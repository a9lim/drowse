"""Principal-angle residual transport: the on-device n<=2 closed forms.

The curved kernel transports the off-surface residual between tangent frames
by rotating principal-vector pairs.  ``n = 1`` and ``n = 2`` build those pairs
in closed form (no ``linalg.svd`` CPU hop on MPS); the SVD path stays the
reference and the ``n >= 3`` implementation.  Principal vectors are not
unique, so equivalence is checked on the transported residual.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch

from drowse.geometry.inject import (
    _orthonormalize_columns,
    _principal_pairs_1d,
    _principal_pairs_2d,
    _principal_pairs_svd,
    _rotate_planes,
)


def _frames(*shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return _orthonormalize_columns(torch.randn(*shape, generator=g))


def _rot(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]])


def _close_pairs(
    qa: torch.Tensor, qb: torch.Tensor, hn: torch.Tensor,
    closed: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> None:
    got = _rotate_planes(hn, *closed(qa, qb))
    ref = _rotate_planes(hn, *_principal_pairs_svd(qa, qb))
    assert torch.allclose(got, ref, atol=2e-5), (got - ref).abs().max()
    # A rotation never changes the residual's length.
    assert torch.allclose(
        torch.linalg.vector_norm(got, dim=-1),
        torch.linalg.vector_norm(hn, dim=-1), atol=2e-5,
    )


@pytest.mark.parametrize("seed", range(6))
def test_two_d_pairs_match_svd_on_random_frames(seed: int) -> None:
    qa = _frames(4, 7, 2, seed=seed)
    qb = _frames(4, 7, 2, seed=100 + seed)
    hn = torch.randn(4, 7, generator=torch.Generator().manual_seed(seed))
    _close_pairs(qa, qb, hn, _principal_pairs_2d)


def test_two_d_pairs_match_svd_for_small_foot_moves() -> None:
    # The hot-path regime: consecutive feet are close, so every principal
    # angle is small and the rotation is a small correction.
    qa = _frames(3, 9, 2, seed=1)
    drift = 1e-2 * torch.randn(3, 9, 2, generator=torch.Generator().manual_seed(2))
    qb = _orthonormalize_columns(qa + drift)
    hn = torch.randn(3, 9, generator=torch.Generator().manual_seed(3))
    _close_pairs(qa, qb, hn, _principal_pairs_2d)


def test_two_d_pairs_degenerate_frames() -> None:
    qa = _frames(9, 2, seed=4)
    hn = torch.randn(9, generator=torch.Generator().manual_seed(5))
    # Same span, rotated basis: both singular values 1 (fully degenerate) and
    # the transport is the identity.
    same_span = qa @ _rot(0.7)
    a, b, c = _principal_pairs_2d(qa, same_span)
    assert torch.allclose(c, torch.ones(2), atol=1e-6)
    assert torch.allclose(_rotate_planes(hn, a, b, c), hn, atol=1e-5)
    # One shared direction, the other turned by 90°: singular values (1, 0).
    full = _frames(9, 4, seed=6)
    qa2, qb2 = full[:, :2], torch.stack((full[:, 0], full[:, 2]), dim=-1)
    _close_pairs(qa2, qb2, hn, _principal_pairs_2d)
    # Fully orthogonal planes: M = 0, every principal angle is 90° and the
    # pairing is genuinely non-unique, so only the invariants are checked:
    # the residual keeps its length and span(qa) lands in span(qb).
    qa3, qb3 = full[:, :2], full[:, 2:]
    a, b, c = _principal_pairs_2d(qa3, qb3)
    assert torch.allclose(c, torch.zeros(2), atol=1e-6)
    moved = _rotate_planes(qa3[:, 0], a, b, c)
    assert torch.allclose(torch.linalg.vector_norm(moved), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(qb3 @ (qb3.T @ moved), moved, atol=1e-5)


def test_one_d_pairs_match_svd() -> None:
    qa = _frames(5, 6, 1, seed=7)
    qb = _frames(5, 6, 1, seed=8)
    hn = torch.randn(5, 6, generator=torch.Generator().manual_seed(9))
    _close_pairs(qa, qb, hn, _principal_pairs_1d)
    _close_pairs(qa, -qa, hn, _principal_pairs_1d)  # antiparallel: same line


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_two_d_pairs_on_mps_match_cpu() -> None:
    qa = _frames(4, 7, 2, seed=10)
    qb = _frames(4, 7, 2, seed=11)
    hn = torch.randn(4, 7, generator=torch.Generator().manual_seed(12))
    cpu = _rotate_planes(hn, *_principal_pairs_2d(qa, qb))
    mps = _rotate_planes(
        hn.to("mps"), *_principal_pairs_2d(qa.to("mps"), qb.to("mps")),
    ).cpu()
    assert torch.allclose(mps, cpu, atol=1e-5)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_gn_step_closed_form_solves_match_linalg_solve(n: int) -> None:
    """The n<=2 damped normal equations are solved in closed form (no
    device sync); they must agree with ``linalg.solve`` on the same system."""
    from drowse.geometry.domains import BoxAxis, BoxDomain
    from drowse.geometry.invert import (
        DEFAULT_INVERSION_DAMPING,
        _INVERSION_DIAG_FLOOR,
        _gn_step,
    )
    from drowse.geometry.rbf import eval_rbf_jet
    from drowse.geometry.subspace import fit_curved_subspace

    g = torch.Generator().manual_seed(n)
    domain = BoxDomain([BoxAxis(f"t{i}", periodic=True, period=1.0) for i in range(n)])
    grid = torch.linspace(0.0, 1.0, 4)[:-1]
    coords = torch.cartesian_prod(*([grid] * n)).reshape(-1, n)
    sub = fit_curved_subspace(
        torch.randn(coords.shape[0], 12, generator=g), domain.embed(coords),
    ).subspace
    np_, rw, pc = sub.rbf_params()
    p = torch.rand(5, n, generator=g)
    q = torch.randn(5, sub.rank, generator=g)
    got = _gn_step(
        p, q, np_, rw, pc, sub.coord_offset, sub.coord_scale, domain,
        DEFAULT_INVERSION_DAMPING,
    )
    emb = (domain.embed(p) - sub.coord_offset) / sub.coord_scale
    value, jac = eval_rbf_jet(np_, rw, pc, emb)
    j = (jac / sub.coord_scale) @ domain.tangent_jacobian(p)
    jtj = j.transpose(-1, -2) @ j
    diag = torch.diagonal(jtj, dim1=-2, dim2=-1)
    A = jtj + torch.diag_embed(
        DEFAULT_INVERSION_DAMPING * diag.clamp(min=_INVERSION_DIAG_FLOOR)
        + _INVERSION_DIAG_FLOOR
    )
    step = torch.linalg.solve(A, (j.transpose(-1, -2) @ (value - q).unsqueeze(-1)))
    ref = domain.retract_position(p, -step.squeeze(-1))
    assert torch.allclose(got, ref, atol=1e-5)
