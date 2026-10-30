"""The fitted manifold container.

A :class:`Manifold` is a domain plus one :class:`~drowse.geometry.subspace.
LayerSubspace` per fitted layer, the node roster (labels, authoring coords,
roles, kinds), the per-layer Mahalanobis share, and — for curved fits — the
per-layer neutral origin foot.  It is the in-memory analogue of a fitted
manifold folder's per-model tensor, built by the extraction pipeline and the
store loaders and consumed by the steering composer and the monitor.

The surrounding geometry lives in sibling modules: :mod:`.domains` (intrinsic
geometry), :mod:`.rbf` (the surface interpolant), :mod:`.subspace` (per-layer
frames and fits), :mod:`.tube` (the sigma field), :mod:`.invert` (the foot
solve), :mod:`.inject` (composition and the steering kernel) and
:mod:`.transfer` (cross-model mapping).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence, cast

import torch

from drowse.errors import UnknownManifoldLabelError
from drowse.geometry.domains import ManifoldDomain
from drowse.geometry.subspace import LayerSubspace

# Numerical fitting semantics are independent of the folder/tensor wire
# format. Bump this whenever PCA/Fisher selection, topology choice, RBF/sigma
# fitting, DLS, or share allocation changes incompatibly.
MANIFOLD_FIT_POLICY_VERSION = 1


class _OmittedNodeRoster(list[str | None]):
    """Default marker: an omitted roster is filled; an explicit ``[]`` is malformed."""


@dataclass
class Manifold:
    """A fitted manifold: a domain + per-layer subspaces, plus identity.

    The in-memory analogue of :class:`drowse.geometry.profile.Profile` for
    manifold steering.  Built by the manifold extraction pipeline and
    consumed by the session and the steering hooks.

    ``feature_space`` is ``"raw"`` for a manifold fitted directly on
    residual-stream activations, or ``"sae-<release>"`` when fitted in an
    SAE feature space (the stored :class:`LayerSubspace` values are then
    already decoded back to model space for runtime use).
    """

    name: str
    domain: ManifoldDomain
    node_labels: list[str]
    node_coords: torch.Tensor   # (K, n) authoring coordinates per node
    layers: dict[int, LayerSubspace]
    feature_space: str = "raw"
    metadata: dict[str, object] = field(default_factory=dict)
    # Per-node assistant-role substitution recorded at fit time, aligned
    # with ``node_labels``.  ``None`` for a given node = "pooled under
    # the standard assistant baseline" — the default, and what every
    # non-role manifold carries.  Used by
    # :meth:`Manifold.nearest_node_role` for role-paired steering.
    node_roles: list[str | None] = field(default_factory=_OmittedNodeRoster)
    # Per-node conceptual ``kind`` — ``"abstract"`` (a trait/quality, e.g.
    # ``happy``) or ``"concrete"`` (an entity, e.g. ``pirate``), aligned with
    # ``node_labels``.  ``None`` = unspecified.  A *generation-time* attribute
    # only: it selects the system template and the elicitation role label
    # (``someone {label}`` vs ``{label}``) when authoring a node's
    # conversational corpus.  It does NOT feed the fit — extraction pools in
    # standard-assistant space (swap-back) regardless — so it is carried for
    # provenance / regeneration, not consumed by the fit-time capture pass.
    node_kinds: list[str | None] = field(default_factory=_OmittedNodeRoster)
    # Per-layer Mahalanobis share weight recorded at fit time —
    # ``share_L = ‖Bᵀ coords_k‖_M`` summed over
    # nodes, the subspace-restricted analogue of vector steering's
    # ``‖d‖_M`` bake score (see ``LayerWhitener.subspace_gram``). Coverage is
    # exact: every fitted layer has one share value.
    # These are raw per-layer scalars with two normalized consumers: the
    # apply-time **steer** weight (normalized to mean 1, ``Σ_L share_L =
    # n_layers``) in ``_manifold_layer_shares``, and the **read** weight
    # (normalized to sum 1) the unified ``Monitor`` uses to combine each
    # layer's geometry into one cross-layer reading — the layer carrying
    # the most steering budget is also the most reliable to read from, so
    # one quantity drives both sides.
    mahalanobis_share: dict[int, float] = field(default_factory=dict)
    # Origin ``O_L`` — the **per-layer** foot of the neutral mean on ``M``, in
    # authoring coordinates ``(n,)``, keyed by curved layer.  Always a point
    # *on* the manifold (an ``invert_parameterization`` result).  It is the
    # cold-start seed for that layer's per-token nearest-point foot-follower
    # (``subspace_inject``) and the translate reference of ``along``.
    # Per-layer rather than a single shared coord because each layer
    # embeds the shared authoring coords into activation space differently (its
    # own PCA + RBF), so neutral's foot genuinely differs by depth — and the
    # hot-path follower runs one Gauss-Newton step from a *single* seed with no
    # restarts (unlike ``invert_parameterization``), so a per-layer seed in the
    # right basin avoids a wrong-basin foot on periodic / curved domains.  Empty
    # dict on a fit with no neutral means available (CPU test stubs); the apply
    # path then seeds that layer's foot at the coord-space origin ``zeros(n)``.
    origin: dict[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # An omitted roster means "None for every node" — the wire format's
        # full-length shape.  ``None`` is a real per-node value.
        if isinstance(self.node_roles, _OmittedNodeRoster):
            self.node_roles = [None] * len(self.node_labels)
        if isinstance(self.node_kinds, _OmittedNodeRoster):
            self.node_kinds = [None] * len(self.node_labels)

    @property
    def is_affine(self) -> bool:
        """True iff the fit is flat — an affine ``%`` joins the merged subspace.

        A fit is all-affine (``fit_mode=pca``) or all-curved (authored /
        spectral) — :meth:`validate_runtime_geometry` enforces it — so the
        first layer decides.
        """
        if not self.layers:
            raise ValueError(f"manifold {self.name!r} has no fitted layers")
        return next(iter(self.layers.values())).is_affine

    @property
    def layer_indices(self) -> list[int]:
        return sorted(self.layers)

    def validate_runtime_geometry(self) -> None:
        """Require the complete fitted geometry consumed by live steering.

        Persistence readers validate their wire schema; this closes the same
        contract over programmatically constructed objects before they enter a
        hot path.  Optionality remains structural only: affine layers have no
        RBF or tube, and SAE-space curved fits deliberately have no raw-space
        tube.  Missing bakes are never interpreted as an older representation.
        """
        if not self.layers:
            raise ValueError(f"manifold {self.name!r} has no fitted layers")
        node_count = len(self.node_labels)
        intrinsic_dim = int(self.domain.intrinsic_dim)
        if self.node_coords.ndim != 2 or self.node_coords.shape != (
            node_count, intrinsic_dim,
        ):
            raise ValueError(
                f"manifold {self.name!r} node_coords must have shape (K, n)"
            )
        for field_name, values in (
            ("node_roles", self.node_roles),
            ("node_kinds", self.node_kinds),
        ):
            if len(values) != len(self.node_labels):
                raise ValueError(
                    f"manifold {self.name!r} {field_name} must align exactly "
                    "with node_labels"
                )
        layer_keys = set(self.layers)
        if set(self.mahalanobis_share) != layer_keys:
            raise ValueError(
                f"manifold {self.name!r} Mahalanobis shares must cover exactly "
                "its fitted layers"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime-loaded artifacts can violate annotations
            or not math.isfinite(value)
            or value <= 0.0
            for value in self.mahalanobis_share.values()
        ):
            raise ValueError(
                f"manifold {self.name!r} Mahalanobis shares must be finite and positive"
            )
        curved = {idx: sub for idx, sub in self.layers.items() if not sub.is_affine}
        if curved and len(curved) != len(self.layers):
            raise ValueError(
                f"manifold {self.name!r} cannot mix affine and curved layers"
            )
        if set(self.origin) != set(curved):
            raise ValueError(
                f"manifold {self.name!r} origins must cover exactly its curved layers"
            )
        for idx, value in self.origin.items():
            if value.ndim != 1 or value.shape != (intrinsic_dim,):
                raise ValueError(
                    f"manifold {self.name!r} origin for layer {idx} must "
                    "have shape (n,)"
                )
        for idx, sub in self.layers.items():
            try:
                sub.validate_structure(
                    feature_space=self.feature_space,
                    expected_node_count=len(self.node_labels),
                )
            except ValueError as exc:
                raise ValueError(
                    f"manifold {self.name!r} layer {idx}: {exc}"
                ) from exc

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "Manifold":
        """Return a copy with every layer tensor on ``device`` in ``dtype``."""
        return Manifold(
            name=self.name,
            domain=self.domain,
            node_labels=list(self.node_labels),
            node_coords=self.node_coords.to(device=device, dtype=dtype),
            layers={
                idx: sub.to(device=device, dtype=dtype)
                for idx, sub in self.layers.items()
            },
            feature_space=self.feature_space,
            metadata=dict(self.metadata),
            node_roles=list(self.node_roles),
            node_kinds=list(self.node_kinds),
            mahalanobis_share=dict(self.mahalanobis_share),
            origin={
                L: o.to(device=device, dtype=dtype)
                for L, o in self.origin.items()
            },
        )

    def _position_tensor(
        self,
        position: "float | Sequence[float] | torch.Tensor",
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Coerce ``float | Sequence[float] | Tensor`` to a ``(n,)`` tensor."""
        n = self.domain.intrinsic_dim
        if isinstance(position, torch.Tensor):
            pos = cast(torch.Tensor, position).to(device=device, dtype=dtype).reshape(-1)
        else:
            if isinstance(position, (int, float)):
                position = (float(position),)
            pos = torch.tensor(
                [float(c) for c in position], dtype=dtype, device=device,
            )
        if pos.shape[0] != n:
            raise ValueError(
                f"manifold {self.name!r} has intrinsic dimension {n}; "
                f"position has {pos.shape[0]} coordinate(s)"
            )
        return pos

    def manifold_point(
        self, layer: int, position: "float | Sequence[float] | torch.Tensor",
    ) -> torch.Tensor:
        """World-space activation ``(D,)`` at authoring coords ``position``."""
        sub = self.layers[layer]
        pos = self._position_tensor(
            position, device=sub.mean.device, dtype=sub.mean.dtype,
        )
        embedded = self.domain.embed(self.domain.clamp_position(pos))
        return sub.eval_at(embedded)

    def resolve_position(
        self,
        position: "float | Sequence[float] | str | torch.Tensor",
    ) -> tuple[float, ...]:
        """Coerce a position payload to a coord tuple.

        Two input shapes are accepted (parser produces both):

        - A coord payload (tuple, list, float, or 1-D tensor) — passthrough
          to a plain coord tuple with arity unchanged.  Arity validation
          against the domain's intrinsic dimension happens downstream in
          :meth:`SteeringManager.add_manifold`.
        - A node-label string (``"pirate"``) — sugar for "the coords of
          the node labeled <s>".  The label is looked up in
          :attr:`node_labels` and the matching row of
          :attr:`node_coords` is returned.  An unknown label raises
          :class:`UnknownManifoldLabelError`.

        Label form makes ``persona%pirate`` a first-class steering term
        in the shared grammar; the bare-name resolver builds on the same
        lookup.
        """
        if isinstance(position, str):
            try:
                idx = self.node_labels.index(position)
            except ValueError:
                raise UnknownManifoldLabelError(
                    f"manifold {self.name!r} has no node labeled "
                    f"{position!r}; known labels: "
                    f"{sorted(self.node_labels)}"
                ) from None
            row = self.node_coords[idx]
            return tuple(float(c) for c in row.tolist())
        if isinstance(position, torch.Tensor):
            tensor_position = cast(torch.Tensor, position)
            return tuple(float(c) for c in tensor_position.reshape(-1).tolist())
        if isinstance(position, (int, float)):
            return (float(position),)
        return tuple(float(c) for c in position)

    def nearest_node_index(
        self, position: "float | Sequence[float] | str | torch.Tensor",
    ) -> int:
        """Index of the node whose authoring coords lie nearest ``position``.

        Distance is the domain's chordal distance between embedded
        points (the same metric the fit pipeline uses for poisedness
        + the RBF kernel), so periodic axes wrap correctly and a
        sphere is measured on its chord.  ``position`` may be a coord
        payload or a node label — labels short-circuit to the matching
        node index without a distance computation.

        Raises ``ValueError`` when the manifold has no nodes recorded
        (a fitted manifold always carries ``node_coords`` from disk).
        """
        if self.node_coords.numel() == 0 or not self.node_labels:
            raise ValueError(
                f"manifold {self.name!r} carries no node coords — cannot "
                f"resolve a nearest node"
            )
        if isinstance(position, str):
            # A node label trivially is the nearest node to itself.
            # Bypass the geometry; surface UnknownManifoldLabelError on
            # a typo through the same channel as resolve_position.
            try:
                return self.node_labels.index(position)
            except ValueError:
                raise UnknownManifoldLabelError(
                    f"manifold {self.name!r} has no node labeled "
                    f"{position!r}; known labels: "
                    f"{sorted(self.node_labels)}"
                ) from None
        # Coerce + embed both sides through the domain so the metric is
        # consistent.  CPU/fp32 — this runs once per scope entry, off
        # the hot path; we don't need to chase the model's device.
        ref_dtype = self.node_coords.dtype
        pos = self._position_tensor(
            position, device=self.node_coords.device, dtype=ref_dtype,
        )
        pos = self.domain.clamp_position(pos)
        embedded_pos = self.domain.embed(pos)                       # (m,)
        clamped_nodes = self.domain.clamp_position(self.node_coords)
        embedded_nodes = self.domain.embed(clamped_nodes)           # (K, m)
        dists = self.domain.distance(embedded_pos, embedded_nodes)  # (K,)
        return int(torch.argmin(dists).item())

    def nearest_node_role(
        self, position: "float | Sequence[float] | str | torch.Tensor",
    ) -> str | None:
        """Role of the node nearest ``position`` — or ``None``.

        Returns ``None`` when the nearest node opts out of role substitution. The
        return value rides through to ``session._active_role`` so the
        generation prefill re-applies the substitution at decode time,
        producing role-paired manifold steering. The role roster is
        exact and aligned with :attr:`node_labels`.
        """
        idx = self.nearest_node_index(position)
        return self.node_roles[idx]


__all__ = ["MANIFOLD_FIT_POLICY_VERSION", "Manifold"]
