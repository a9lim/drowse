from typing import Any
import numpy as np
import pytest

from drowse.core.surface_topology import detect_surface


def sample_surface(kind: str, seed: int):
    rng = np.random.default_rng(seed)
    if kind in ('sphere', 'projective-plane'):
        sphere = rng.normal(size=(256, 3))
        sphere /= np.linalg.norm(sphere, axis=1)[:, None]
        if kind == 'sphere':
            return sphere
        x, y, z = sphere.T
        return np.stack(((x*x-y*y)/np.sqrt(2), (x*x+y*y-2*z*z)/np.sqrt(6),
                         np.sqrt(2)*x*y, np.sqrt(2)*x*z, np.sqrt(2)*y*z), -1)
    u, v = np.meshgrid(np.arange(24), np.arange(16), indexing='ij')
    u = (u.ravel() + rng.uniform(-0.12, 0.12, 384)) * 2*np.pi/24
    v = (v.ravel() + rng.uniform(-0.12, 0.12, 384)) * 2*np.pi/16
    if kind == 'klein-bottle':
        return np.stack(((2+np.cos(v))*np.cos(u), (2+np.cos(v))*np.sin(u),
                         np.sin(v)*np.cos(u/2), np.sin(v)*np.sin(u/2)), -1)
    radius = 0.025 if kind == 'thin-torus' else 1.
    return np.stack((np.cos(u), np.sin(u), radius*np.cos(v), radius*np.sin(v)), -1)


@pytest.mark.parametrize('kind,signature', [
    ('sphere', ((1, 0, 1), (1, 0, 1))),
    ('torus', ((1, 2, 1), (1, 2, 1))),
    ('klein-bottle', ((1, 2, 1), (1, 1, 0))),
    ('projective-plane', ((1, 1, 1), (1, 0, 0))),
])
@pytest.mark.parametrize('seed', [711, 731, 743])
def test_surface_regressions_in_high_dimensional_space(kind: str, signature: tuple[tuple[int, int, int], tuple[int, int, int]], seed: int):
    pytest.importorskip('ripser')
    points = sample_surface(kind, seed)
    rng = np.random.default_rng(seed + 9000)
    rotation, _ = np.linalg.qr(rng.normal(size=(11, points.shape[1])))
    points = points @ rotation.T
    points = points[rng.permutation(len(points))] * 17 + rng.normal(size=11)
    result = detect_surface(points)
    assert result.candidate == kind, result
    assert (result.betti_mod2, result.betti_mod3) == signature
    assert result.relative_persistence >= 0.15


@pytest.mark.parametrize('kind', ['sphere', 'torus', 'klein-bottle', 'projective-plane'])
@pytest.mark.parametrize('seed', [10091, 10093])
def test_held_out_noisy_high_dimensional_surfaces(kind: str, seed: int):
    pytest.importorskip('ripser')
    points = sample_surface(kind, seed)
    rng = np.random.default_rng(seed + 811)
    rotation, _ = np.linalg.qr(rng.normal(size=(17, points.shape[1])))
    points = points @ rotation.T + rng.normal(0, 0.002, size=(len(points), 17))
    result = detect_surface(points)
    assert result.candidate == kind, result


@pytest.mark.parametrize('nu,nv,radius', [(24, 16, 0.6), (48, 8, 0.3)])
@pytest.mark.parametrize('seed', [787, 797])
def test_three_dimensional_doughnut_tori(nu: int, nv: int, radius: float, seed: int):
    pytest.importorskip('ripser')
    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.arange(nu), np.arange(nv), indexing='ij')
    u = (u.ravel() + rng.uniform(-0.1, 0.1, nu*nv)) * 2*np.pi/nu
    v = (v.ravel() + rng.uniform(-0.1, 0.1, nu*nv)) * 2*np.pi/nv
    points = np.stack(((2+radius*np.cos(v))*np.cos(u), (2+radius*np.cos(v))*np.sin(u), radius*np.sin(v)), -1)
    result = detect_surface(points)
    assert result.candidate == 'torus', result


@pytest.mark.parametrize('kind', ['disk', 'mobius', 'cylinder', 'cloud', 'thin-torus', 'disconnected'])
def test_unresolved_data_does_not_receive_a_closed_surface_label(kind: str):
    pytest.importorskip('ripser')
    rng = np.random.default_rng(977)
    if kind == 'thin-torus':
        points = sample_surface(kind, 977)
    elif kind == 'disconnected':
        sphere = sample_surface('sphere', 977)[:128]
        points = np.concatenate((sphere, sphere + np.array([6., 0, 0])))
    elif kind in ('mobius', 'cylinder'):
        u, v = np.meshgrid(np.arange(24)*2*np.pi/24, np.linspace(-0.5, 0.5, 12), indexing='ij')
        u, v = u.ravel(), v.ravel()
        if kind == 'mobius':
            points = np.stack(((2+v*np.cos(u/2))*np.cos(u), (2+v*np.cos(u/2))*np.sin(u), v*np.sin(u/2)), -1)
        else:
            points = np.stack((np.cos(u), np.sin(u), v), -1)
    else:
        points = rng.normal(size=(128, 2 if kind == 'disk' else 7))
    assert detect_surface(points).candidate is None


@pytest.mark.parametrize('points', [np.zeros((32, 3)), np.ones((385, 3)), np.ones((31, 3)), np.ones((40, 1)), np.full((32, 3), np.nan), np.full((32, 3), 1j)])
def test_invalid_or_over_budget_clouds_raise(points: np.ndarray[Any, Any]):
    with pytest.raises(ValueError):
        detect_surface(points)


@pytest.mark.parametrize('threshold', [0, -1, 1, np.nan, np.inf])
def test_invalid_persistence_threshold_raises(threshold: float):
    with pytest.raises(ValueError):
        detect_surface(np.random.default_rng(1).normal(size=(32, 3)), min_relative_persistence=threshold)
