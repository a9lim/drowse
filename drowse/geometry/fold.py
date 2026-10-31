"""The direction view of fitted rays.

A steering vector is the ``R = 1`` affine case of a manifold.  These functions
move between the two shapes:

- :func:`fold_directions_to_subspace` folds a per-layer direction into a
  neutral-anchored one-pole ray (a derived direction: a merge, a ``~``/``|``
  projection, a registered profile read as a probe);
- :func:`is_foldable_vector_manifold` / :func:`folded_directions` read a
  fitted ray back as per-layer directions (``δ̂_L · share_L``);
- :func:`project_profile` is the LEACE ``~``/``|`` projection of one direction
  profile against another in the Mahalanobis metric.
"""
from __future__ import annotations

from typing import Any

import torch

from drowse.errors import SteeringCompositionError, WhitenerError
from drowse.geometry.domains import CustomDomain
from drowse.geometry.manifold import Manifold
from drowse.geometry.subspace import LayerSubspace

__all__ = [
    "fold_directions_to_subspace",
    "folded_directions",
    "is_foldable_vector_manifold",
    "project_profile",
]


def fold_directions_to_subspace(
    name: str,
    directions: dict[int, torch.Tensor],
    neutral_means: dict[int, torch.Tensor],
    *,
    whitener: "Any",
    label: str = "+",
    feature_space: str = "raw",
) -> Manifold:
    """Fold an arbitrary per-layer *direction* into a neutral-anchored affine
    ``R = 1`` manifold (a one-pole ray).

    Where a bipolar concept's fit (:mod:`drowse.fit.pipeline`) folds a line through two
    pole centroids, a derived direction (a ``merge`` linear-combination, a
    `~`/`|` projection of one concept onto another) has no pole pair — just a
    direction to push along.
    So the subspace is neutral-anchored at ``mean = P_basis(ν_L) = (ν_L·d̂)d̂``
    (the neutral mean projected into the 1-D span, §5) with a single ``+`` pole
    node; steering ``along`` toward the pole pushes the running activation's
    ``d̂`` component away from neutral, and ``!`` (``along``-to-coord-0) slides
    it back to neutral, i.e. ablates the direction.

    Per layer: ``basis = d̂_L``; ``LayerSubspace.node_coords = [[‖d_L‖]]`` —
    the real coord of the pole, a step of ``‖d_L‖`` along ``d̂`` from the
    neutral origin (coord 0). ``share = ‖d_L‖_M``: the whitened direction
    magnitude itself is the budget (a single
    direction has no node cloud to take a μ-centered spread over, so this is
    *not* :func:`~drowse.geometry.subspace.subspace_share`'s √2-scaled form; the
    apply-time normalization handles the cross-layer scale). No
    DLS — a derived direction has no polarity for the straddle test; the
    caller's layer set folds verbatim. No origin store (coord 0, §2).
    Neutral means and a covering whitener are mandatory.
    """
    present = sorted(directions)
    if not present:
        raise ValueError("cannot fold an empty direction profile")
    missing_means = sorted(set(present) - set(neutral_means))
    if missing_means:
        raise WhitenerError(
            f"folding {name!r} requires neutral means for every layer; "
            f"missing {missing_means}"
        )
    if whitener is None or not whitener.covers_all(present):
        raise WhitenerError(
            f"folding {name!r} requires a Mahalanobis whitener covering "
            f"every layer {present}"
        )

    # Every fresh tensor follows the directions' device so the folded manifold
    # is internally device-consistent.  The dispatch-time fold runs over a
    # *loaded* manifold's directions, which may live on the model device (MPS /
    # CUDA); a CPU-default ``node_coords`` would then mismatch the on-device
    # ``basis`` inside ``synthesize_subspace`` (``c_i @ B_i``).  ``ν`` is moved to
    # the same device too so the neutral-anchor projection stays co-located.
    dev = directions[present[0]].device

    layers: dict[int, LayerSubspace] = {}
    mahalanobis_share: dict[int, float] = {}
    for idx in present:
        d = directions[idx].to(torch.float32).reshape(-1)
        norm = float(d.norm())
        if norm <= 1e-12:
            continue
        basis = (d / norm).reshape(1, -1)          # (1, D) unit d̂
        nu = neutral_means[idx].to(
            device=d.device, dtype=torch.float32,
        ).reshape(-1)
        # Neutral-anchored: mean = P_basis(ν) (off-span part of ν dropped);
        # the pole sits at the real coord ‖d‖ along d̂ from the origin.
        mean = (nu @ basis.T) @ basis
        node_coords_L = torch.tensor([[norm]], dtype=torch.float32, device=d.device)
        layers[idx] = LayerSubspace.affine(mean, basis, node_coords=node_coords_L)
        mahalanobis_share[idx] = float(whitener.mahalanobis_norm(idx, d))

    if not layers:
        raise ValueError("cannot fold a direction profile containing only zero vectors")

    # Shared display layout: a single ``+`` pole at coord 1 (the real per-layer
    # pole distance ‖d_L‖ lives on each ``LayerSubspace.node_coords``).
    node_coords = torch.tensor([[1.0]], dtype=torch.float32, device=dev)
    manifold = Manifold(
        name=name,
        domain=CustomDomain(1),
        node_labels=[label],
        node_coords=node_coords,
        layers=layers,
        feature_space=feature_space,
        node_roles=[None],
        node_kinds=[None],
        mahalanobis_share=mahalanobis_share,
    )
    manifold.metadata["share_metric"] = "mahalanobis"
    manifold.validate_runtime_geometry()
    return manifold


def is_foldable_vector_manifold(manifold: Manifold) -> bool:
    """True iff *manifold* is a folded (affine ``R = 1``) steering vector —
    the only shape :func:`folded_directions` accepts.

    A multi-node / curved fit (a rank-``R`` subspace, e.g. the ``personas``
    fan or the ``emotions`` surface) is **not** a single steering direction, so it
    can't participate in the direction-cosine analytics (``correlation`` /
    ``profiles/pairwise``). Callers use this to skip such probes instead of
    letting the fold raise — an empty / layerless manifold is also False.
    """
    layers = manifold.layers
    if not layers:
        return False
    return all(
        sub.is_affine and sub.rank == 1
        for sub in layers.values()
    )


def folded_directions(manifold: Manifold) -> dict[int, torch.Tensor]:
    """Direction view of a folded (affine ``R = 1``) vector manifold.

    Returns ``{L: δ̂_L · share_L}`` — each layer's unit direction scaled by its
    Mahalanobis share — the in-memory steering-vector shape behind
    ``extract()``'s :class:`Profile`, ``manifold compare``/``why``, ``merge``,
    and GGUF export.  The manifold stays the only stored artifact.  Exact for
    scale-invariant reads (per-layer and aggregate cosine); proportional for
    cross-concept magnitude.

    Raises :class:`ValueError` on a curved or multi-dim manifold — the
    single-direction view is only meaningful for an ``R = 1`` ray.
    """
    out: dict[int, torch.Tensor] = {}
    manifold.validate_runtime_geometry()
    share_map = manifold.mahalanobis_share
    for idx, sub in manifold.layers.items():
        if not sub.is_affine or sub.rank != 1:
            raise ValueError(
                "folded_directions requires affine R=1 layers; "
                f"layer {idx} is rank {sub.rank}, affine={sub.is_affine}"
            )
        share = float(share_map[idx])
        out[idx] = sub.basis.reshape(-1).to(torch.float32) * share
    return out


def project_profile(
    base: dict[int, torch.Tensor],
    onto: dict[int, torch.Tensor],
    operator: str,
    *,
    whitener: "Any | None" = None,
) -> dict[int, torch.Tensor]:
    """Per-layer projection of ``base`` against ``onto`` (LEACE only).

    LEACE-style projection in the Mahalanobis metric, per shared layer::

        coef = <base, onto>_M / <onto, onto>_M
        proj = coef * onto                # direction is ``onto``; metric is M

    The output direction is still ``onto``, but the *amount* removed is
    the component along ``onto`` measured in the whitened space.  For
    operator ``"|"``, this is the closed-form LEACE projector for a
    single direction (Belrose et al. 2023, arXiv 2306.03819) — provably
    erases linearly-decodable information along ``onto`` from ``base``
    with minimum collateral damage.

    Operator semantics:

    - ``operator == "~"``   returns ``proj``       (component of base aligned with onto).
    - ``operator == "|"`` returns ``base - proj`` (component of base orthogonal to onto).

    Layers in ``base`` without a matching layer in ``onto``: for ``"|"``
    they pass through unchanged (nothing to project away); for ``"~"`` they
    are dropped (projection onto an absent direction is undefined).

    Near-zero ``||onto|| < 1e-12`` layers are treated the same way: ``"|"``
    passes base through unchanged, ``"~"`` drops the layer.

    The whitener (a :class:`drowse.geometry.whitener.LayerWhitener`) is
    **required** and must cover *every* projected layer (``base ∩ onto``),
    via :meth:`LayerWhitener.covers_all`.  There is no Euclidean path: a
    missing or non-covering whitener raises :class:`WhitenerError`.
    Result tensors are cast back to the source dtype of ``base``.
    """
    if operator not in ("~", "|"):
        raise ValueError(f"unknown projection operator: {operator!r}")
    # Mahalanobis-only: the whitener must cover every layer that actually
    # gets projected (``base ∩ onto``).  No Euclidean fallback — a missing
    # or partial whitener is an error.
    projected_layers = sorted(set(base) & set(onto))
    if whitener is None or not whitener.covers_all(projected_layers):
        raise WhitenerError(
            "project_profile requires a Mahalanobis whitener covering every "
            f"projected layer {projected_layers}; regenerate the neutral "
            "activation cache for this model (the Euclidean path is gone)"
        )
    out: dict[int, torch.Tensor] = {}
    for layer, base_t in base.items():
        onto_t = onto.get(layer)
        if onto_t is None:
            if operator == "|":
                out[layer] = base_t
            continue
        projected = whitener.leace_project(layer, base_t, onto_t, operator)
        # Drop the layer for ``~`` when ``onto`` is degenerate under
        # the Mahalanobis metric — leace_project returns a zero tensor
        # in that case.
        if operator == "~" and torch.all(projected == 0):
            continue
        out[layer] = projected
    if not out:
        raise SteeringCompositionError(
            f"project_profile: no layers produced for operator {operator!r} "
            f"(base layers: {sorted(base.keys())}, "
            f"onto layers: {sorted(onto.keys())})"
        )
    return out
