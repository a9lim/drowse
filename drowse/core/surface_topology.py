"""Synthetic point-cloud surface diagnostics, separate from chart recovery.

Two coefficient fields distinguish orientable surfaces from 2-torsion.
Homology signatures identify a surface only *conditional on* a connected,
closed two-manifold. Finite point samples do not establish that hypothesis;
these results are evidence, never a certificate or an automatic steering chart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class SurfaceEvidence:
    candidate: str | None
    reason: str
    betti_mod2: tuple[int, int, int] | None = None
    betti_mod3: tuple[int, int, int] | None = None
    scale_interval: tuple[float, float] | None = None
    relative_persistence: float = 0.0
    sample_count: int = 0


def _surface_name(b2: tuple[int, ...], b3: tuple[int, ...]) -> str | None:
    if b2[0] != 1 or b3[0] != 1 or b2[2] != 1:
        return None
    if b3[2] == 1 and b2[1] == b3[1] and b2[1] % 2 == 0:
        genus = b2[1] // 2
        return {0: 'sphere', 1: 'torus'}.get(genus, f'orientable-genus-{genus}')
    if b3[2] == 0 and b2[1] == b3[1] + 1:
        genus = b2[1]
        return {1: 'projective-plane', 2: 'klein-bottle'}.get(genus, f'nonorientable-genus-{genus}')
    return None


def detect_surface(points: ArrayLike, *, min_relative_persistence: float = 0.15) -> SurfaceEvidence:
    """Measure closed-surface evidence in explicit Euclidean R^n sample data.

    Input is 32..384 distinct finite points; larger inputs are rejected, not
    silently subsampled. Ripser computes H0/H1/H2 over F2 and F3. A dominant
    finite H2 bar and a simultaneous stable signature in both fields are
    required. Its birth-normalized persistence must dominate the next H2 bar
    by at least 3:1;
    competing signatures and short scale intervals cause abstention.

    The persistence threshold is a fraction of the feature's birth radius,
    not the cloud diameter or a probability/confidence score. This avoids
    discarding a resolved small hole solely because another axis is long.
    Isometries and uniform rescaling leave
    the decision invariant. Noise, sparse coverage, thin necks, boundaries,
    and self-intersections can invalidate a closed-surface interpretation.
    Activation-space callers must whiten explicitly before calling this API.
    """
    if np.iscomplexobj(points):
        raise ValueError('Surface points must be real-valued')
    data = np.asarray(points, dtype=np.float64)
    if data.ndim != 2 or not 32 <= len(data) <= 384 or data.shape[1] < 2:
        raise ValueError('Surface diagnostics need 32..384 points with at least two coordinates')
    if not np.isfinite(data).all():
        raise ValueError('Surface points must be finite')
    if not np.isfinite(min_relative_persistence) or not 0 < min_relative_persistence < 1:
        raise ValueError('min_relative_persistence must be between zero and one')
    from scipy.spatial.distance import pdist, squareform

    # Scale before squaring distances, avoiding overflow for large unit changes.
    magnitude = np.max(np.abs(data))
    if magnitude == 0:
        raise ValueError('Surface points must be distinct')
    data = data / magnitude
    data -= data.mean(axis=0)
    spread = np.max(np.abs(data))
    if spread == 0:
        raise ValueError('Surface points must be distinct')
    data /= spread
    pairwise = pdist(data)
    if pairwise.min() <= 1e-12:
        raise ValueError('Surface points contain duplicate or numerically indistinguishable samples')
    scale = np.median(pairwise)
    distances = squareform(pairwise / scale)
    try:
        from ripser import ripser
    except ImportError as exc:
        raise ImportError('Surface diagnostics require drowse[topology]') from exc

    diagrams = [ripser(distances, distance_matrix=True, maxdim=2, coeff=2)['dgms']]
    h2 = diagrams[0][2]
    if not len(h2):
        return SurfaceEvidence(None, 'No persistent two-dimensional class', sample_count=len(data))
    lifetimes = h2[:, 1] - h2[:, 0]
    relative = lifetimes / h2[:, 0]
    order = np.argsort(relative)
    principal = h2[order[-1]]
    lifetime = float(lifetimes[order[-1]])
    if not np.isfinite(principal).all() or lifetime < min_relative_persistence * principal[0]:
        return SurfaceEvidence(None, 'Two-dimensional persistence is too short or unbounded', sample_count=len(data))
    if len(order) > 1 and relative[order[-1]] < 3 * relative[order[-2]]:
        return SurfaceEvidence(None, 'Competing two-dimensional classes; surface unresolved', sample_count=len(data))

    diagrams.append(ripser(distances, distance_matrix=True, maxdim=2, coeff=3)['dgms'])

    # All critical values are included; no fixed grid can skip a short-lived
    # contradictory class. Trim H2 endpoints so its birth/death do not vote.
    lo, hi = principal[0] + 0.05 * lifetime, principal[1] - 0.05 * lifetime
    events = sorted({float(lo), float(hi)} | {
        float(value) for field in diagrams for barcode in field
        for value in barcode.ravel() if lo < value < hi
    })
    intervals = []
    for left, right in zip(events, events[1:]):
        middle = (left + right) / 2
        signatures = [tuple(int(np.sum((d[:, 0] <= middle) & (middle < d[:, 1]))) for d in field) for field in diagrams]
        key = (_surface_name(*signatures), *signatures)
        if intervals and intervals[-1][0] == key:
            intervals[-1] = (key, intervals[-1][1], right)
        else:
            intervals.append((key, left, right))
    viable = [(key, left, right) for key, left, right in intervals
              if key[0] is not None and right - left >= min_relative_persistence * left]
    if not viable or len({key[0] for key, _, _ in viable}) != 1:
        return SurfaceEvidence(None, 'No unambiguous simultaneous surface signature', relative_persistence=lifetime / float(principal[0]), sample_count=len(data))
    key, left, right = max(viable, key=lambda interval: interval[2] - interval[1])
    return SurfaceEvidence(
        key[0], 'Homology consistent with a connected closed surface; no recovered coordinate chart',
        key[1], key[2], (left, right), (right - left) / left, len(data),
    )
