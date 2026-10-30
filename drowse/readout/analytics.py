"""Mahalanobis comparison analytics over per-layer direction profiles.

Two read-only views over named directions (steering profiles or probes), both
in the per-model :class:`~drowse.geometry.whitener.LayerWhitener` metric —
there is no Euclidean path:

- :func:`cross_layer_cosine` — the ``layers_a × layers_b`` whitened cosine
  matrix between two directions;
- :func:`cosine_matrix` — the N×N magnitude-weighted aggregate cosine across
  a pool, the same aggregate :meth:`Profile.cosine_similarity` computes, built
  from one ``Σ⁻¹`` application per name and layer instead of one per pair.

Inputs are CPU tensors; callers pass snapshots so a polled endpoint never
copies off the model device mid-operation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from drowse.geometry.whitener import LayerWhitener

Directions = Mapping[int, torch.Tensor]

_EPS = 1e-12


def cross_layer_cosine(
    a: Directions, b: Directions, whitener: "LayerWhitener",
) -> tuple[list[int], list[int], list[list[float | None]]]:
    """Whitened cosine between every layer of ``a`` and every layer of ``b``.

    Mahalanobis cosine is a single-covariance operation, so each cell is
    whitened in ``a``'s row-layer frame: exact on the layer-aligned diagonal,
    an A-frame read off it.  The whitener must cover every layer of ``a``.
    A cell is ``None`` when either side has near-zero whitened norm or the
    two hidden sizes differ.  Returns ``(layers_a, layers_b, matrix)``.
    """
    layers_a = sorted(a)
    layers_b = sorted(b)
    vecs_b = [b[layer].float().cpu().reshape(-1) for layer in layers_b]
    matrix: list[list[float | None]] = []
    for la in layers_a:
        va = a[la].float().cpu().reshape(-1)
        row: list[float | None] = [None] * len(vecs_b)
        si_va = whitener.apply_inv(la, va).float().reshape(-1)
        aa = max(float(torch.dot(va, si_va).item()), 0.0)
        compatible = [j for j, vb in enumerate(vecs_b) if vb.shape == va.shape]
        if aa >= _EPS and compatible:
            block = torch.stack([vecs_b[j] for j in compatible])
            si_block = whitener.apply_inv(la, block).float().reshape(len(compatible), -1)
            bb = (block * si_block).sum(dim=1).clamp_min(0.0)
            uv = block @ si_va
            denom = (bb * aa).sqrt()
            cosines = uv / denom.clamp_min(_EPS)
            for j, ok, cos in zip(compatible, (denom >= _EPS).tolist(), cosines.tolist(), strict=True):
                if ok:
                    row[j] = round(float(cos), 6)
        matrix.append(row)
    return layers_a, layers_b, matrix


def cosine_matrix(
    pool: Mapping[str, Directions], names: Sequence[str], whitener: "LayerWhitener",
) -> tuple[dict[str, dict[str, float | None]], dict[str, int]]:
    """N×N magnitude-weighted Mahalanobis cosine across ``names``.

    Each pair aggregates over its shared layers as
    ``Σ_L ⟨a_L, b_L⟩_M / Σ_L ‖a_L‖_M ‖b_L‖_M`` (layers with near-zero norm
    on either side drop out).  A name absent from ``pool``, a pair with no
    shared layers, or a pair whose shared layers the whitener does not fully
    cover yields ``None``.  Returns ``(matrix, layers_shared)`` where
    ``layers_shared`` is keyed by the alphabetically sorted ``"a__b"`` pair.
    """
    factors: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor, float]]] = {}
    for name in names:
        if name not in pool:
            continue
        entries: dict[int, tuple[torch.Tensor, torch.Tensor, float]] = {}
        for layer, tensor in pool[name].items():
            vec = tensor.float().cpu().reshape(-1)
            si_vec = whitener.apply_inv(int(layer), vec).float().reshape(-1)
            entries[int(layer)] = (vec, si_vec, max(float(torch.dot(vec, si_vec).item()), 0.0))
        factors[name] = entries

    matrix: dict[str, dict[str, float | None]] = {name: {} for name in names}
    layers_shared: dict[str, int] = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j < i:
                matrix[a][b] = matrix[b][a]
                continue
            if i == j:
                matrix[a][b] = 1.0
                continue
            if a not in factors or b not in factors:
                matrix[a][b] = None
                continue
            shared = sorted(set(factors[a]) & set(factors[b]))
            layers_shared["__".join(sorted((a, b)))] = len(shared)
            if not shared or not whitener.covers_all(shared):
                matrix[a][b] = None
                continue
            num = den = 0.0
            for layer in shared:
                vec_a, _, aa = factors[a][layer]
                _, si_b, bb = factors[b][layer]
                if aa < _EPS or bb < _EPS:
                    continue
                num += float(torch.dot(vec_a, si_b).item())
                den += math.sqrt(aa * bb)
            matrix[a][b] = round(num / den, 6) if den >= _EPS else None
    return matrix, layers_shared
