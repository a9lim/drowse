"""Synthetic fitted manifolds for steering tests (no model).

``loop_manifold`` / ``open_manifold`` / ``disk_manifold`` fit curved subspaces
through planar-circle (or random) centroids and stamp the fields a current
fitted artifact carries (Euclidean reference shares, a zero origin, a flat
sigma field), so the hooks and composer accept them as real artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from drowse.geometry.domains import BoxAxis, BoxDomain
from drowse.geometry.manifold import Manifold
from drowse.geometry.rbf import eval_rbf
from drowse.geometry.subspace import fit_curved_subspace

__all__ = [
    "DIM",
    "circle_centroids",
    "disk_manifold",
    "euclidean_shares",
    "loop_manifold",
    "open_manifold",
    "stamp_current_geometry",
]

#: Hidden width of every synthetic manifold here.
DIM = 8


def circle_centroids(k: int, dim: int, scale: float = 1.0) -> torch.Tensor:
    """K activation centroids on a planar circle, with a DC offset.

    The 20.0 common-mode component mimics a real LM activation's large,
    persona-agnostic norm — without it the per-node centroids vary wildly
    in magnitude and the soft norm-cap inside ``subspace_inject`` fires on
    synthetic fits, masking the geometry under test.
    """
    out = torch.zeros(k, dim)
    base = torch.full((dim,), 20.0)
    for i in range(k):
        theta = 2.0 * math.pi * i / k
        out[i] = base.clone()
        out[i, 0] += scale * 2.0 * math.cos(theta)
        out[i, 1] += scale * 2.0 * math.sin(theta)
    return out


def loop_manifold(layers: Sequence[int] = (0, 1)) -> Manifold:
    """A 1-D periodic (loop) manifold over 6 nodes."""
    domain = BoxDomain([BoxAxis("t", periodic=True, period=1.0)])
    node_coords = torch.tensor([[i / 6] for i in range(6)])
    node_params = domain.embed(node_coords)
    manifold = Manifold(
        name="mood",
        domain=domain,
        node_labels=[f"n{i}" for i in range(6)],
        node_coords=node_coords,
        layers={
            L: fit_curved_subspace(circle_centroids(6, DIM, 1.0 + 0.2 * L), node_params).subspace
            for L in layers
        },
        feature_space="raw",
    )
    return stamp_current_geometry(manifold)


def open_manifold(layers: Sequence[int] = (0, 1)) -> Manifold:
    """A 1-D *non-periodic* (open interval) manifold over 6 nodes.

    Same activation pattern as ``loop_manifold`` so share-weighting tests that
    assert share > 1 still hold; the only change is the domain, which disables
    the periodic clamping fix and keeps the pre-fix share-weighted path.
    """
    domain = BoxDomain([BoxAxis("t", periodic=False, lo=0.0, hi=1.0)])
    node_coords = torch.tensor([[i / 6] for i in range(6)])
    node_params = domain.embed(node_coords)
    manifold = Manifold(
        name="mood_open",
        domain=domain,
        node_labels=[f"n{i}" for i in range(6)],
        node_coords=node_coords,
        layers={
            L: fit_curved_subspace(circle_centroids(6, DIM, 1.0 + 0.2 * L), node_params).subspace
            for L in layers
        },
        feature_space="raw",
    )
    return stamp_current_geometry(manifold)


def disk_manifold(layers: Sequence[int] = (0,)) -> Manifold:
    """A 2-D box (disk-like) manifold over a 3x3 grid of 9 nodes."""
    domain = BoxDomain([
        BoxAxis("u", periodic=False, lo=0.0, hi=1.0),
        BoxAxis("v", periodic=False, lo=0.0, hi=1.0),
    ])
    coords = torch.tensor(
        [[x, y] for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)]
    )
    node_params = domain.embed(coords)
    torch.manual_seed(7)
    manifold = Manifold(
        name="disk",
        domain=domain,
        node_labels=[f"n{i}" for i in range(9)],
        node_coords=coords,
        layers={
            L: fit_curved_subspace(20.0 + torch.randn(9, DIM), node_params).subspace
            for L in layers
        },
        feature_space="raw",
    )
    return stamp_current_geometry(manifold)


def stamp_current_geometry(manifold: Manifold) -> Manifold:
    """Give a synthetic fitted manifold the bakes current artifacts require."""
    manifold.mahalanobis_share = euclidean_shares(manifold)
    manifold.origin = {
        layer: torch.zeros(manifold.domain.intrinsic_dim)
        for layer in manifold.layers
    }
    manifold.node_roles = [None] * len(manifold.node_labels)
    manifold.node_kinds = [None] * len(manifold.node_labels)
    for sub in manifold.layers.values():
        if sub.is_affine or sub.node_params is None:
            continue
        sub.sigma_rbf_weights = torch.zeros(
            (sub.node_params.shape[0], 1), dtype=sub.basis.dtype,
        )
        sub.sigma_poly_coeffs = torch.zeros(
            (sub.node_params.shape[1] + 1, 1), dtype=sub.basis.dtype,
        )
        sub.sigma_poly_coeffs[0, 0] = -20.0
    return manifold


def euclidean_shares(manifold: Manifold) -> dict[int, float]:
    """Reference Euclidean ``‖coords‖_F`` share, **mean-1** normalized
    (``Σ = n_layers``) to match ``_manifold_layer_shares``."""
    raw = {}
    for L, sub in manifold.layers.items():
        _np, _rw, _pc = sub.rbf_params()
        c = eval_rbf(_np, _rw, _pc, _np)
        raw[L] = float(torch.linalg.vector_norm(c).item())
    tot = sum(raw.values())
    n = len(raw)
    return {L: v / tot * n for L, v in raw.items()}
