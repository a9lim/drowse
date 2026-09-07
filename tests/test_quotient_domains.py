from pathlib import Path
from drowse.core.manifold import ManifoldDomain
import math

import pytest
import torch

from drowse.core.manifold import (
    KleinBottleDomain, ProjectivePlaneDomain, SphereDomain, Manifold,
    domain_from_spec, fit_layer_subspace, fit_sigma_field, subspace_inject,
    invert_parameterization,
)
from drowse.io.manifold_tensors import save_manifold, load_manifold
from drowse.core.monitor import _mean_coordinate_space, _mean_domain_coordinates


@pytest.fixture(params=[KleinBottleDomain, ProjectivePlaneDomain])
def domain(request: pytest.FixtureRequest):
    return request.param()


def equivalent(domain: ManifoldDomain, points: torch.Tensor):
    if isinstance(domain, KleinBottleDomain):
        return torch.stack((points[..., 0] + 2 * math.pi, -points[..., 1]), -1)
    sphere = SphereDomain(2)
    return sphere._unembed(-sphere.embed(points))


def test_quotient_embedding_and_canonicalization(domain: ManifoldDomain):
    points = torch.randn(30, 2, generator=torch.Generator().manual_seed(141), dtype=torch.float64) * 9
    twin = equivalent(domain, points)
    expected = domain.embed(points)
    torch.testing.assert_close(domain.embed(twin), expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(domain.embed(domain.clamp_position(points)), expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(domain.clamp_position(twin), domain.clamp_position(points), atol=1e-12, rtol=1e-12)


def test_quotient_jacobian(domain: ManifoldDomain):
    points = torch.tensor([[0.7, 1.2], [2 * math.pi - 1e-6, 2.3], [1.1, 4.3]], dtype=torch.float64)
    analytic = domain.embed_jacobian(points)
    for axis in range(2):
        delta = torch.zeros_like(points)
        delta[:, axis] = 1e-6
        numerical = (domain.embed(points + delta) - domain.embed(points - delta)) / 2e-6
        torch.testing.assert_close(analytic[..., axis], numerical, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize('fraction', [0., 0.25, 1., -0.4, 2.])
def test_paths_and_transport_ignore_choice_of_representative(domain: ManifoldDomain, fraction: float):
    origin = torch.tensor([0.3, 0.7], dtype=torch.float64)
    target = torch.tensor([2.9, 4.8], dtype=torch.float64)
    points = torch.tensor([[0.1, 0.5], [3.2, 2.2], [6.2, 4.4]], dtype=torch.float64)
    path = domain.geodesic(origin, target, fraction)
    twin_path = domain.geodesic(equivalent(domain, origin), equivalent(domain, target), fraction)
    torch.testing.assert_close(domain.embed(path), domain.embed(twin_path), atol=1e-12, rtol=1e-12)
    moved = domain.translate_foot(points, origin, target, fraction)
    twin_moved = domain.translate_foot(equivalent(domain, points), equivalent(domain, origin), equivalent(domain, target), fraction)
    torch.testing.assert_close(domain.embed(moved), domain.embed(twin_moved), atol=1e-12, rtol=1e-12)
    if fraction == 0:
        torch.testing.assert_close(domain.embed(moved), domain.embed(points), atol=1e-12, rtol=1e-12)


def test_klein_crosses_twisted_seam_continuously():
    domain = KleinBottleDomain()
    origin = torch.tensor([2 * math.pi - 0.2, 0.8], dtype=torch.float64)
    target = torch.tensor([0.2, -0.8], dtype=torch.float64)
    points = domain.geodesic(origin, target, torch.linspace(0, 1, 101, dtype=torch.float64)[:, None])
    assert torch.linalg.vector_norm(torch.diff(domain.embed(points), dim=0), dim=-1).max() < 0.02
    torch.testing.assert_close(points[-1], domain.clamp_position(target))
    assert points[0, 1] < 1
    assert points[-1, 1] > 5


def test_projective_crosses_antipodal_chart_seam_continuously():
    domain = ProjectivePlaneDomain()
    origin = torch.tensor([0.9, 0.7], dtype=torch.float64)
    target = equivalent(domain, origin + torch.tensor([0.1, 0.1], dtype=torch.float64))
    points = domain.geodesic(origin, target, torch.linspace(0, 1, 101, dtype=torch.float64)[:, None])
    assert torch.linalg.vector_norm(torch.diff(domain.embed(points), dim=0), dim=-1).max() < 0.004
    torch.testing.assert_close(domain.embed(points[-1]), domain.embed(target), atol=1e-12, rtol=1e-12)


def test_projective_poles_and_cut_locus_stay_finite():
    domain = ProjectivePlaneDomain()
    points = torch.tensor([[0., 0.], [math.pi, 0.], [math.pi / 2, 0.]], dtype=torch.float64)
    for fraction in [0., 0.5, 1.]:
        result = domain.translate_foot(points, points[0], points[2], fraction)
        assert torch.isfinite(result).all()
        assert torch.isfinite(domain.embed(result)).all()
    torch.testing.assert_close(domain.embed(domain.geodesic(points[0], points[2], 1)), domain.embed(points[2]))


def test_projective_solver_frame_remains_full_rank_at_poles():
    domain = ProjectivePlaneDomain()
    coords = torch.tensor([[0., 0.], [math.pi, 0.], [0.7, 1.3]], dtype=torch.float64)
    jac = domain.tangent_jacobian(coords)
    assert (torch.linalg.svdvals(jac) > 1).all()
    for axis in range(2):
        delta = torch.zeros_like(coords)
        delta[:, axis] = 1e-6
        numerical = (domain.embed(domain.retract_position(coords, delta)) - domain.embed(domain.retract_position(coords, -delta))) / 2e-6
        torch.testing.assert_close(jac[..., axis], numerical, atol=1e-8, rtol=1e-8)


def test_probe_mean_respects_identified_coordinates(domain: ManifoldDomain):
    reference = torch.tensor([0.2, 0.7], dtype=torch.float64)
    points = torch.stack((reference, equivalent(domain, reference)))
    mean = _mean_coordinate_space(domain, points).mean(0)
    result = _mean_domain_coordinates(domain, mean, reference)
    torch.testing.assert_close(domain.embed(result), domain.embed(reference), atol=1e-12, rtol=1e-12)


def test_probe_mean_across_klein_seam_is_not_half_a_turn_away():
    domain = KleinBottleDomain()
    points = torch.tensor([[2*math.pi-0.01, 0.8], [0.01, -0.8]], dtype=torch.float64)
    mean = _mean_coordinate_space(domain, points).mean(0)
    result = _mean_domain_coordinates(domain, mean, points[0])
    assert torch.linalg.vector_norm(domain.embed(result) - domain.embed(points[0])) < 0.04


def test_ambiguous_projective_mean_preserves_reference():
    domain = ProjectivePlaneDomain()
    reference = torch.tensor([1.1, 0.4])
    torch.testing.assert_close(domain.project_mean(torch.zeros(5), reference), reference)


def fitted(domain: ManifoldDomain, nonlinear: bool=False):
    u = torch.arange(12) * (2 * math.pi / 12) + 0.07
    v = torch.arange(10) * (2 * math.pi / 10) + 0.03
    coords = torch.cartesian_prod(u, v)
    if isinstance(domain, ProjectivePlaneDomain):
        coords[:, 0] *= 0.5
    embedded = domain.embed(coords)
    values = torch.cat((embedded, 0.2 * embedded * embedded.roll(1, -1)), -1) if nonlinear else embedded
    centroids = torch.cat((values, torch.zeros(len(coords), 3)), -1) + 20
    sub, _ = fit_layer_subspace(centroids, embedded, n_components=values.shape[1])
    fit_sigma_field({0: sub}, domain, coords, [{0: 0.01 * torch.eye(sub.rank)} for _ in coords], smoothing=0.)
    return coords, sub


def test_fitted_steering_and_serialization_preserve_seams(domain: ManifoldDomain, tmp_path: Path):
    coords, sub = fitted(domain)
    manifold = Manifold(name='quotient', domain=domain, node_labels=[f'n{i}' for i in range(len(coords))], node_coords=coords, layers={0: sub}, mahalanobis_share={0: 1.}, origin={0: coords[0]})
    path = tmp_path / 'quotient.safetensors'
    save_manifold(manifold, path, {'method': 'manifold_discover_spectral', 'fit_mode': 'spectral'})
    restored = load_manifold(path)
    assert restored.domain.to_spec() == domain.to_spec()
    assert isinstance(domain_from_spec(domain.to_spec()), type(domain))
    origin = torch.tensor([6.1, 0.8]) if isinstance(domain, KleinBottleDomain) else torch.tensor([0.9, 0.7])
    target = torch.tensor([0.15, -0.8]) if isinstance(domain, KleinBottleDomain) else equivalent(domain, origin + 0.1)
    h = sub.eval_at(domain.embed(origin))
    for fraction in [0., 0.5, 1.]:
        out, _ = subspace_inject(h, sub, domain, target, origin, fraction, 0., origin=origin, gn_steps=10)
        expected = sub.eval_at(domain.embed(domain.geodesic(origin, target, fraction)))
        torch.testing.assert_close(out, expected, atol=2e-3, rtol=1e-5)
        restored_out, _ = subspace_inject(h, restored.layers[0], restored.domain, equivalent(domain, target), equivalent(domain, origin), fraction, 0., origin=equivalent(domain, origin), gn_steps=10)
        torch.testing.assert_close(restored_out, out, atol=2e-3, rtol=1e-5)


def test_projective_inverse_can_leave_a_polar_seed():
    domain = ProjectivePlaneDomain()
    coords, sub = fitted(domain)
    target = torch.tensor([0.4, 1.7])
    h = sub.eval_at(domain.embed(target))
    query = (h - sub.mean) @ sub.basis.T
    recovered, distance = invert_parameterization(sub, domain, query, coords, n_restarts=1, max_iter=20)
    assert distance < 1e-4
    torch.testing.assert_close(domain.embed(recovered), domain.embed(target), atol=1e-4, rtol=1e-4)
    _, from_pole = subspace_inject(h, sub, domain, target, torch.tensor([0., 0.]), 0., 0., gn_steps=20)
    torch.testing.assert_close(domain.embed(from_pole), domain.embed(target), atol=1e-4, rtol=1e-4)


def test_batched_steering_preserves_off_subspace_residual(domain: ManifoldDomain):
    _, sub = fitted(domain)
    origin = torch.tensor([0.3, 0.7])
    positions = torch.tensor([[0.25, 0.6], [0.4, 0.8], [0.3, 0.7]])
    h = sub.eval_at(domain.embed(positions))
    h[:, -1] += torch.tensor([0.2, -0.3, 0.4])
    for along, onto in [(0., 0.), (0.5, 0.), (0.5, 1.)]:
        out, _ = subspace_inject(h, sub, domain, torch.tensor([6.1, -0.7]), positions, along, onto, origin=origin, gn_steps=10)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out[:, -1], h[:, -1])
        if along == onto == 0:
            torch.testing.assert_close(out, h, atol=1e-5, rtol=1e-5)


def test_seam_transport_preserves_normal_residual_norm(domain: ManifoldDomain):
    _, sub = fitted(domain)
    origin = torch.tensor([6.1, 0.8]) if isinstance(domain, KleinBottleDomain) else torch.tensor([0., 0.])
    target = torch.tensor([0.15, -0.8]) if isinstance(domain, KleinBottleDomain) else torch.tensor([0.2, 1.2])
    tangent = domain.tangent_jacobian(origin)
    frame, _ = torch.linalg.qr(tangent)
    normal = torch.arange(1., domain.embed_dim + 1)
    normal = normal - frame @ (frame.T @ normal)
    normal = 0.03 * normal / normal.norm()
    h = sub.eval_at(domain.embed(origin))
    h[:domain.embed_dim] += normal
    out, _ = subspace_inject(h, sub, domain, target, origin, 1., 0., origin=origin, gn_steps=0)
    moved_surface = sub.eval_at(domain.embed(target))
    torch.testing.assert_close((out - moved_surface).norm(), normal.norm(), atol=2e-5, rtol=1e-4)


def test_nonlinearly_fitted_surface_steers_through_the_seam(domain: ManifoldDomain):
    _, sub = fitted(domain, nonlinear=True)
    origin = torch.tensor([6.1, 0.8]) if isinstance(domain, KleinBottleDomain) else torch.tensor([0.9, 0.7])
    target = torch.tensor([0.15, -0.8]) if isinstance(domain, KleinBottleDomain) else equivalent(domain, origin + 0.1)
    h = sub.eval_at(domain.embed(origin))
    for fraction in [0.25, 0.5, 1.]:
        out, _ = subspace_inject(h, sub, domain, target, origin, fraction, 0., origin=origin, gn_steps=10)
        expected = sub.eval_at(domain.embed(domain.geodesic(origin, target, fraction)))
        torch.testing.assert_close(out, expected, atol=3e-3, rtol=1e-5)


@pytest.mark.parametrize('spec', [{'type': 'klein', 'dim': 2}, {'type': 'projective', 'dim': 3}, {'type': 'projective', 'dim': True}])
def test_quotient_specs_reject_unsupported_dimensions(spec: dict[str, object]):
    with pytest.raises(ValueError):
        domain_from_spec(spec)
