"""Steering plans and the forward hooks that run them.

A :class:`SteeringPlan` is the composed, device-agnostic steering state for one
steering-stack head: one merged affine subspace per trigger group
(:class:`AffineTerm`) plus each curved ``%`` manifold (:class:`CurvedTerm`).
:meth:`SteeringPlan.lower` turns it into per-layer injection groups for one
device/dtype, and :meth:`SteeringManager.install` is the only way steering
reaches the model: it either writes the persistent compiled offset buffers
(static pure-push plans) or attaches one transient :class:`SteeringHook` per
steered layer.  :meth:`SteeringManager.clear_all` is its inverse, so the
installed state is always a function of the plan.

Every affine injection — transient single-group, multi-group, or the compiled
offsets — runs through one :class:`AffineOp`; curved terms run the curved
branch of :func:`~drowse.geometry.manifold.subspace_inject`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from drowse.errors import (
    ManifoldArityError,
    OverlappingManifoldError,
    SteeringCompositionError,
)
from drowse.geometry.manifold import (
    BoxDomain,
    LayerSubspace,
    Manifold,
    ManifoldDomain,
    SynthesizedSubspace,
    _ortho_basis,
    subspace_inject,
)
from drowse.steering.triggers import Trigger, TriggerContext


def _trigger_active(trigger: Trigger, ctx: TriggerContext) -> bool:
    """Hot-path trigger predicate with the default preset as a true no-op."""
    return trigger is Trigger.BOTH or trigger.active(ctx)


def _canonical_trigger(trigger: Trigger) -> Trigger:
    """Intern ``Trigger.BOTH`` so the hot path's ``is`` short-circuit holds.

    A trigger equal to BOTH but built elsewhere (a deserializer, a hand-built
    ``Trigger()``) would otherwise miss every fast path.
    """
    return Trigger.BOTH if trigger == Trigger.BOTH else trigger


# --- steering gains ----------------------------------------------------------
# All three ride the mean-1 per-layer share weighting whose contract lives in
# ``_normalize_shares_mean1``:
#
#   affine ``along``   ``eff_along_L = share_L · _SUBSPACE_GAIN``
#   curved ``along``   ``eff_along_L = along · share_L · _MANIFOLD_ALONG_GAIN``
#                      (periodic ``BoxDomain``: unweighted, clamped to [0, 1])
#   ``onto``           ``eff_onto_L  = clamp(onto · share_L · _MANIFOLD_ONTO_GAIN, 0, 1)``

# Affine ``along`` gain (prototype).  ``synthesize_subspace`` emits a
# whitened-unit target per push term, so the average per-layer whitened push is
# ``GAIN·α`` for every target.  Live-calibrated on a gemma-4-12b α-sweep
# (``formal.casual%formal``, ``personas%caveman``, ``personas%hacker``): the
# dead→expressive transition sits at ``E = GAIN·α ≈ 7-8`` for every target,
# but the coherence ceiling varies ~2× (hacker breaks at ``E ≈ 12``, caveman
# ``≈ 17``, formal past ``22``) — the per-target coherence variance a scalar
# gain cannot unify.  Coherence-first: recommended ``α ≈ 0.5`` lands at
# ``E ≈ 8``, so ``GAIN = 16``; ``α ≈ 1.0`` is the strong / over-steered zone.
_SUBSPACE_GAIN = 16.0

# Curved ``along`` gain.  The curved target is raw node coordinates and the
# kernel translates the foot by ``eff_along·(target − origin)``, so
# ``eff_along`` is a fraction of the way to the node (``1.0`` lands on it);
# ``norm_cap = 3·‖h‖`` bounds off-domain RBF extrapolation past it.
# Live-calibrated on a stateless gemma-4-12b ``months_loop%january`` sweep:
# ``along = 1.0`` at gain 4 lands the vivid coherent sweet spot.  Periodic
# ``BoxDomain`` fits drop share-weighting and clamp ``eff_along`` to [0, 1]
# per layer — share ∈ [0.19, 1.47] × 4 would send layers past the node, each
# wrapping to a different one.
_MANIFOLD_ALONG_GAIN = 4.0

# ``onto`` gain (curved off-surface collapse only).  The off-surface residual
# carries per-token variation, so too much ``onto`` erases it and degenerates
# into looping.  Calibrated on the gemma-4-12b ``emotions%dominant`` onto sweep
# (along 0.3): at gain 0.5 ``onto ≈ 0.5`` is the sweet spot and ``onto = 1.0``
# a strong-but-coherent ceiling.
_MANIFOLD_ONTO_GAIN = 0.5

# Max |cosine| between two curved subspaces sharing a layer before they are
# overlapping (each overwrites its own in-subspace component).  The merged
# affine subspace is instead orthogonalized against the curved spans.
_CURVED_ORTHO_TOL = 1e-3

# Gauss-Newton steps on a cold foot (seed at the origin): the prefill fire, or
# the first decode step under a non-prompt trigger.  Warm fires take one.
_MANIFOLD_COLD_GN_STEPS = 4


def _normalize_shares_mean1(raw: dict[int, float]) -> dict[int, float]:
    """Normalize per-layer share scores to **mean 1** (``Σ_L share_L = n_layers``).

    The share weighting every steering gain rides — the one place its contract
    lives.  A raw share (``synth.share`` = the whitened push displacement
    ``‖Δ_L‖_M``, or the baked ``mahalanobis_share`` of a curved fit) says how
    much signal a layer carries; mean-1 normalization makes ``eff_L = share_L ·
    gain`` a per-layer slide ≈ ``gain`` on a typical layer.  That buys
    n_layers-invariance: a 1-layer and a 30-layer fit, or a 4-dim and a 16-dim
    fit, behave comparably at the same α.

    ``along`` is not clamped afterwards — a high-signal layer is meant to
    overshoot, held by the bounded whitened-unit target (affine) or
    ``norm_cap`` (curved).  The exceptions: periodic ``BoxDomain`` curved fits
    drop the weighting, and ``onto`` stays clamped to ``[0, 1]``.

    Inputs must be finite and strictly positive.
    """
    if not raw or any(not math.isfinite(s) or s <= 0.0 for s in raw.values()):
        raise ValueError("layer shares must be a nonempty finite positive mapping")
    n_layers = len(raw)
    total = sum(raw.values())
    return {L: s / total * n_layers for L, s in raw.items()}


def _manifold_layer_shares(manifold: Manifold) -> dict[int, float]:
    """A curved manifold's mean-1 per-layer share from its baked Mahalanobis share.

    Full layer coverage is required: the share is cross-layer normalized, so
    a partial map would compare incommensurable layers.
    """
    baked = manifold.mahalanobis_share
    missing = set(manifold.layers) - set(baked)
    extra = set(baked) - set(manifold.layers)
    if missing or extra:
        raise ValueError(
            f"manifold {manifold.name!r} has non-canonical Mahalanobis shares "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )
    return _normalize_shares_mean1(
        {layer_idx: float(baked[layer_idx]) for layer_idx in manifold.layers}
    )


def _orthogonalize_affine_against(
    sub: LayerSubspace,
    target: torch.Tensor,
    kappa: torch.Tensor,
    curved_basis: torch.Tensor,
) -> "tuple[LayerSubspace, torch.Tensor, torch.Tensor] | None":
    """Project a merged affine subspace out of the curved manifolds' span.

    The curved manifold wins the shared directions (ARCHITECTURE §6); the
    affine slide handles the complement.  Strips the curved-span component
    from the affine basis rows and the push displacement, and carries the
    per-axis collapse mask through the re-orthonormalization (each new axis
    inherits its squared projection onto the old ablate span).  Returns
    ``None`` when the affine span lies entirely inside the curved span.
    """
    basis = sub.basis.to(torch.float32)              # (R, D)
    cb = curved_basis.to(torch.float32)              # (Rc, D)
    delta = basis.T @ target.to(torch.float32)       # (D,) world push displacement
    residual = basis - (basis @ cb.T) @ cb           # (R, D) rows ⟂ curved span
    new_basis, _kept = _ortho_basis(list(residual))
    if new_basis.shape[0] == 0:
        return None
    delta_perp = delta - (delta @ cb.T) @ cb
    new_target = new_basis @ delta_perp              # (R',)
    M = new_basis @ basis.T                          # (R', R)
    new_kappa = (M * M) @ kappa.to(torch.float32)    # (R',)
    return LayerSubspace.affine(sub.mean, new_basis), new_target, new_kappa


@dataclass(frozen=True)
class LoweredAffine:
    """One layer's merged affine steering term in runtime form.

    ``subspace`` is the (possibly curved-orthogonalized) affine subspace,
    ``target`` its ``(R,)`` push coord (every push term's coeff already
    composed in), ``eff_along = share_L · _SUBSPACE_GAIN`` the per-layer
    slide budget, and ``kappa`` the gain-compensated per-axis collapse mask
    (``requested_κ / eff_along``, so ``eff_along · κ`` is exactly the user's
    ablation coefficient).
    """

    subspace: LayerSubspace
    target: torch.Tensor
    eff_along: float
    kappa: torch.Tensor

    @property
    def is_pure_push(self) -> bool:
        """True when no axis ablates, i.e. the injection is a constant add."""
        return not bool(self.kappa.any())

    def offset(self) -> torch.Tensor:
        """The push displacement ``(eff_along · target) @ basis`` (fp32, ``(D,)``)."""
        basis = self.subspace.basis.to(torch.float32)
        return (self.eff_along * self.target.to(torch.float32)) @ basis


def _lower_affine_subspaces(
    synth: SynthesizedSubspace,
    *,
    curved_basis_by_layer: "dict[int, torch.Tensor] | None" = None,
) -> dict[int, LoweredAffine]:
    """Lower one synthesized affine subspace to its per-layer runtime terms.

    **The** affine lowering: mean-1 share normalization, optional
    orthogonalization against the curved manifolds sharing a layer,
    ``eff_along_L = share_L · _SUBSPACE_GAIN``, and the κ gain compensation.
    Both the transient hooks and the compiled offset buffers consume its
    result through :class:`AffineOp` / :meth:`LoweredAffine.offset`.  Runs at
    compose time, never per token.
    """
    layer_set = list(synth.layers)
    if not layer_set:
        return {}
    shares = _normalize_shares_mean1(
        {L: float(synth.share[L]) for L in layer_set}
    )
    lowered: dict[int, LoweredAffine] = {}
    for L in layer_set:
        sub_L = synth.layers[L]
        target = synth.target_coord[L].to(torch.float32)
        # Orthogonalization runs on the user-space κ first; the single scalar
        # division below then preserves the resulting operator exactly.
        requested_kappa = synth.kappa[L].to(torch.float32)
        curved = (
            curved_basis_by_layer.get(L)
            if curved_basis_by_layer is not None
            else None
        )
        if curved is not None:
            res = _orthogonalize_affine_against(
                sub_L, target, requested_kappa, curved,
            )
            if res is None:
                continue
            sub_L, target, requested_kappa = res
        eff_along = shares[L] * _SUBSPACE_GAIN
        lowered[L] = LoweredAffine(
            subspace=sub_L,
            target=target,
            eff_along=eff_along,
            kappa=requested_kappa / eff_along,
        )
    return lowered


class AffineOp:
    """One layer's affine injection: ``h += c − ((h·Bₐᵀ − μBₐᵀ) ⊙ s) Bₐ``.

    The analytic affine slide of :func:`subspace_inject` split at compose time
    into its fixed push ``c = eff_along·target@B`` and its activation-dependent
    ablation, restricted to the nonzero-κ rows ``Bₐ`` with ``s = eff_along·κ``
    (the requested ablation coefficient).  A pure push is therefore one
    in-place ``add_``; a rank-1 ablation is a dot plus an in-place ``addcmul_``
    (MPS dispatch dominates a rank-1 GEMM).  Runs in the residual stream's
    dtype; the constants are computed in fp32 and cast once.

    No ``norm_cap``: a flat fit has no RBF to extrapolate, so the bounded
    displacement cannot push ``‖h‖`` past the curved path's ``3·‖h‖`` guard.
    """

    __slots__ = (
        "_dtype", "const", "mean_proj", "rows", "rows_t", "scale", "scaled_row",
    )

    def __init__(
        self,
        const: "torch.Tensor | None",
        rows: "torch.Tensor | None" = None,
        scale: "torch.Tensor | None" = None,
        mean_proj: "torch.Tensor | None" = None,
    ) -> None:
        self.const = const
        self.rows = rows
        self.scale = scale
        self.mean_proj = mean_proj
        self.rows_t = None if rows is None else rows.T.contiguous()
        self.scaled_row = (
            None if rows is None or scale is None or rows.shape[0] != 1
            else (scale[0] * rows[0]).contiguous()
        )
        carrier = rows if rows is not None else const
        self._dtype = None if carrier is None else carrier.dtype

    @classmethod
    def from_lowered(
        cls,
        low: LoweredAffine,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "AffineOp":
        sub = low.subspace
        const_f32 = low.offset().to(device)
        const = const_f32.to(dtype) if bool(const_f32.any()) else None
        kappa = low.kappa.to(device=device, dtype=torch.float32)
        active = kappa != 0
        if not bool(active.any()):
            return cls(const)
        basis = sub.basis.to(device=device, dtype=torch.float32)
        mean = sub.mean.to(device=device, dtype=torch.float32)
        rows = basis[active]                                      # (Ra, D)
        return cls(
            const,
            rows=rows.to(dtype).contiguous(),
            scale=(low.eff_along * kappa[active]).to(dtype),      # (Ra,)
            mean_proj=(mean @ rows.T).to(dtype),                  # (Ra,)
        )

    def _recast(self, dtype: torch.dtype) -> None:
        # The residual stream can be narrower than the wrapper's first
        # parameter (multimodal / mixed-precision stacks).  Recast once and
        # keep it, so steady state stays allocation-free.
        self._dtype = dtype
        if self.const is not None:
            self.const = self.const.to(dtype)
        if self.rows is not None:
            assert self.scale is not None and self.mean_proj is not None
            self.rows = self.rows.to(dtype)
            self.rows_t = self.rows.T.contiguous()
            self.scale = self.scale.to(dtype)
            self.mean_proj = self.mean_proj.to(dtype)
            if self.scaled_row is not None:
                self.scaled_row = self.scaled_row.to(dtype)

    def apply(self, hidden: torch.Tensor) -> None:
        """Steer ``hidden`` in place."""
        if self._dtype is not hidden.dtype and self._dtype is not None:
            self._recast(hidden.dtype)
        rows = self.rows
        if rows is None:
            if self.const is not None:
                hidden.add_(self.const)
            return
        mean_proj = self.mean_proj
        assert mean_proj is not None
        scaled_row = self.scaled_row
        if scaled_row is not None:
            q = (hidden * rows[0]).sum(dim=-1, keepdim=True) - mean_proj
            if self.const is not None:
                hidden.add_(self.const)
            hidden.addcmul_(q, scaled_row, value=-1)
            return
        assert self.rows_t is not None and self.scale is not None
        q = hidden @ self.rows_t - mean_proj
        if self.const is not None:
            hidden.add_(self.const)
        hidden.sub_((q * self.scale) @ rows)


@dataclass(frozen=True)
class AffineTerm:
    """One trigger group's merged affine subspace, as registered on a plan."""

    name: str
    synth: SynthesizedSubspace
    trigger: Trigger


@dataclass(frozen=True)
class CurvedTerm:
    """One curved ``%`` manifold term, as registered on a plan.

    ``position`` is the resolved authoring coordinate tuple (labels resolved
    at registration); ``shares`` the manifold's mean-1 per-layer share.
    """

    name: str
    manifold: Manifold
    position: tuple[float, ...]
    along: float
    onto: float
    trigger: Trigger
    shares: dict[int, float]


@dataclass(frozen=True)
class AffineGroup:
    """A lowered affine term at one layer."""

    trigger: Trigger
    op: AffineOp


@dataclass(frozen=True)
class CurvedGroup:
    """A lowered curved term at one layer (fp32 tensors on the model device)."""

    trigger: Trigger
    subspace: LayerSubspace
    domain: ManifoldDomain
    target: torch.Tensor
    origin: torch.Tensor
    along: float
    onto: float


Group = AffineGroup | CurvedGroup


class SteeringPlan:
    """Composed steering terms for one stack head, lowered on demand.

    Built by the composer (or directly in tests) through :meth:`add_subspace`
    and :meth:`add_manifold`, which validate as they register; lowering
    (:meth:`lower`) and the compiled offsets (:meth:`static_offsets`) are
    memoized, so a cached plan re-installs without recomposing.
    """

    def __init__(self) -> None:
        self.affine: list[AffineTerm] = []
        self.curved: list[CurvedTerm] = []
        self._lowered: dict[
            tuple[torch.device, torch.dtype], dict[int, tuple[Group, ...]]
        ] = {}
        self._offsets: "dict[int, torch.Tensor] | None | bool" = False

    def __bool__(self) -> bool:
        return bool(self.affine or self.curved)

    def _invalidate(self) -> None:
        self._lowered = {}
        self._offsets = False

    def add_subspace(
        self,
        name: str,
        synth: SynthesizedSubspace,
        *,
        trigger: Trigger = Trigger.BOTH,
    ) -> None:
        """Register a dispatch-synthesized merged affine subspace."""
        self.affine.append(AffineTerm(name, synth, _canonical_trigger(trigger)))
        self._invalidate()

    def add_manifold(
        self,
        name: str,
        manifold: Manifold,
        position: tuple[float, ...] | str,
        along: float,
        onto: float,
        trigger: Trigger = Trigger.BOTH,
    ) -> None:
        """Register a curved-manifold term.

        ``position`` is an authoring coord tuple or a node label (resolved
        here).  ``along`` slides the foot toward the position as a fraction
        in ``[0, 1]``; ``onto`` collapses the off-surface residual.  Both are
        fractions on a curved fit, so a negative value has no meaning and
        raises rather than silently clamping to a no-op (the same expression
        on an affine fit would push away — ``fit_mode=auto`` can resolve
        either way per model).
        """
        if along < 0.0 or onto < 0.0:
            raise SteeringCompositionError(
                f"manifold {name!r} is curved, so its along/onto coefficients "
                f"are fractions in [0, 1]; got along={along:g}, onto={onto:g}. "
                f"Steer toward a different position instead of negating."
            )
        manifold.validate_runtime_geometry()
        resolved = manifold.resolve_position(position)
        domain = manifold.domain
        if len(resolved) != domain.intrinsic_dim:
            raise ManifoldArityError(
                f"manifold {name!r} has a {domain.intrinsic_dim}-dimensional "
                f"domain but the steering position has {len(resolved)} "
                f"coordinate(s)"
            )
        self.curved.append(CurvedTerm(
            name=name,
            manifold=manifold,
            position=tuple(float(c) for c in resolved),
            along=float(along),
            onto=float(onto),
            trigger=_canonical_trigger(trigger),
            shares=_manifold_layer_shares(manifold),
        ))
        self._invalidate()

    def lower(
        self, device: torch.device, dtype: torch.dtype,
    ) -> dict[int, tuple[Group, ...]]:
        """Per-layer injection groups for ``device``/``dtype`` (memoized).

        Curved groups come first at each layer, then the affine groups in
        registration order; the affine subspace is orthogonalized against the
        curved spans so the two compose with zero cross-talk.  Two curved
        manifolds sharing a layer must be near-orthogonal or
        :class:`OverlappingManifoldError` raises.
        """
        memo_key = (device, dtype)
        memo = self._lowered.get(memo_key)
        if memo is not None:
            return memo
        by_layer: dict[int, list[Group]] = {}
        curved_owner: dict[int, str] = {}
        curved_basis_by_layer: dict[int, torch.Tensor] = {}
        for term in self.curved:
            manifold = term.manifold
            domain = manifold.domain
            along = min(1.0, term.along)
            onto = min(1.0, term.onto)
            periodic = isinstance(domain, BoxDomain) and any(
                ax.periodic for ax in domain.axes
            )
            target_coord = domain.clamp_position(
                torch.tensor(term.position, dtype=torch.float32)
            ).to(device)
            for layer_idx, sub in manifold.layers.items():
                B_new = sub.basis.to(torch.float32)
                prev = curved_basis_by_layer.get(layer_idx)
                if prev is not None:
                    cross = float((B_new @ prev.T).abs().max().item())
                    if cross > _CURVED_ORTHO_TOL:
                        raise OverlappingManifoldError(
                            f"manifolds '{curved_owner[layer_idx]}' and "
                            f"'{term.name}' both cover layer {layer_idx} with "
                            f"non-orthogonal subspaces (max |cosine| = "
                            f"{cross:.3f} > {_CURVED_ORTHO_TOL}); curved "
                            f"manifolds sharing a layer must be orthogonal"
                        )
                    curved_basis_by_layer[layer_idx] = torch.cat([prev, B_new])
                else:
                    curved_basis_by_layer[layer_idx] = B_new
                    curved_owner[layer_idx] = term.name
                if layer_idx not in manifold.origin:
                    raise ValueError(
                        f"curved manifold {manifold.name!r} has no neutral "
                        f"origin for layer {layer_idx}"
                    )
                share = term.shares[layer_idx]
                eff_along = (
                    min(1.0, along * _MANIFOLD_ALONG_GAIN) if periodic
                    else along * share * _MANIFOLD_ALONG_GAIN
                )
                eff_onto = min(1.0, onto * share * _MANIFOLD_ONTO_GAIN)
                if eff_along == 0.0 and eff_onto == 0.0:
                    continue
                origin_coord = domain.clamp_position(
                    manifold.origin[layer_idx].reshape(-1).to(torch.float32)
                ).to(device)
                by_layer.setdefault(layer_idx, []).append(CurvedGroup(
                    trigger=term.trigger,
                    subspace=sub.to(device=device, dtype=torch.float32),
                    domain=domain,
                    target=target_coord,
                    origin=origin_coord,
                    along=eff_along,
                    onto=eff_onto,
                ))
        for affine in self.affine:
            lowered = _lower_affine_subspaces(
                affine.synth, curved_basis_by_layer=curved_basis_by_layer,
            )
            for L, low in lowered.items():
                by_layer.setdefault(L, []).append(AffineGroup(
                    trigger=affine.trigger,
                    op=AffineOp.from_lowered(low, device=device, dtype=dtype),
                ))
        result = {L: tuple(groups) for L, groups in by_layer.items()}
        self._lowered[memo_key] = result
        return result

    def static_offsets(self) -> "dict[int, torch.Tensor] | None":
        """Per-layer constant offsets, or ``None`` if not a constant add.

        ``{}`` for an empty plan; a ``{layer: (D,)}`` fp32 map when every term
        is an always-active pure push; ``None`` for anything that needs the
        per-token kernel (curved ``%``, a gate / phase trigger, an ``!``
        ablation).  The offsets come from the same :class:`LoweredAffine` the
        transient :class:`AffineOp` consumes, so compiled and eager pushes are
        one computation.
        """
        if self._offsets is not False:
            return self._offsets  # pyright: ignore[reportReturnType]  # memo sentinel is False only
        offsets: "dict[int, torch.Tensor] | None" = {}
        if self.curved:
            offsets = None
        else:
            for affine in self.affine:
                if affine.trigger is not Trigger.BOTH:
                    offsets = None
                    break
                lowered = _lower_affine_subspaces(affine.synth)
                if not all(low.is_pure_push for low in lowered.values()):
                    offsets = None
                    break
                assert offsets is not None
                for L, low in lowered.items():
                    c = low.offset()
                    offsets[L] = offsets[L] + c if L in offsets else c
        self._offsets = offsets
        return offsets


class SteeringHook:
    """One steered layer: its lowered groups and the forward hook running them.

    The hook short-circuits to a single :meth:`AffineOp.apply` when the layer
    carries exactly one always-active affine group — the dominant case, a
    fixed tensor-op sequence with no :class:`TriggerContext` read (what makes
    a steered generation StaticCache / ``torch.compile`` eligible, see
    :meth:`SteeringManager.static_steerable`).  Otherwise it walks the groups
    in order, skipping inactive triggers and threading each curved group's
    per-token foot.
    """

    def __init__(self) -> None:
        self.groups: tuple[Group, ...] = ()
        self._static: AffineOp | None = None
        self._always_active = True
        # Warm-start foot per group (``None`` = cold, seed at the origin);
        # only curved groups use it.
        self._feet: list[torch.Tensor | None] = []
        self._ctx: TriggerContext | None = None
        self._handle: Any = None

    @property
    def is_static(self) -> bool:
        return self._static is not None

    def recompose(self, groups: "tuple[Group, ...] | list[Group]", ctx: TriggerContext) -> None:
        """Arm this layer with ``groups``; a new group set cold-starts every foot."""
        self.groups = tuple(groups)
        self._ctx = ctx
        self._always_active = all(g.trigger is Trigger.BOTH for g in self.groups)
        only = self.groups[0] if len(self.groups) == 1 else None
        self._static = (
            only.op
            if isinstance(only, AffineGroup) and self._always_active
            else None
        )
        self.reset_feet()

    def reset_feet(self) -> None:
        self._feet = [None] * len(self.groups)

    def hook_fn(self, module: Any, input: Any, output: Any) -> Any:
        static = self._static
        if static is not None:
            static.apply(output if isinstance(output, torch.Tensor) else output[0])
            return output
        groups = self.groups
        ctx = self._ctx
        if not groups or ctx is None:
            return output
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        always = self._always_active
        for i, group in enumerate(groups):
            if not always and not _trigger_active(group.trigger, ctx):
                continue
            if isinstance(group, AffineGroup):
                group.op.apply(hidden)
            else:
                self._apply_curved(i, group, hidden)
        return output

    def _apply_curved(
        self, i: int, group: CurvedGroup, hidden: torch.Tensor,
    ) -> None:
        """Run one curved group with a warm-started nearest-point foot.

        The previous token's refined foot seeds one Gauss-Newton step; a cold
        foot (first fire, or a leading shape that no longer broadcasts — a
        re-prefill or batch change) seeds at the origin with
        :data:`_MANIFOLD_COLD_GN_STEPS`.  The last position's foot carries
        forward, so prefill hands its final prompt position to decode.
        """
        lead = hidden.shape[:-1]
        seed = self._feet[i]
        if seed is not None and seed.shape[:-1] == lead:
            foot_seed = seed
            gn_steps = 1
        else:
            n = int(group.origin.shape[-1])
            foot_seed = group.origin.reshape((1,) * len(lead) + (n,)).expand(*lead, n)
            gn_steps = _MANIFOLD_COLD_GN_STEPS
        h_new, foot = subspace_inject(
            hidden, group.subspace, group.domain, group.target, foot_seed,
            group.along, group.onto, gn_steps=gn_steps, origin=group.origin,
        )
        hidden.copy_(h_new)
        self._feet[i] = foot[..., -1:, :].detach()

    def attach(self, layer_module: torch.nn.Module) -> None:
        """Register the forward hook on a layer module."""
        self._handle = layer_module.register_forward_hook(self.hook_fn)

    def detach(self) -> None:
        """Remove the forward hook."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class SteeringManager:
    """The installed steering state of one model.

    Owns the per-generation :class:`TriggerContext` every hook reads (the
    generation loop mutates it at lifecycle boundaries), the installed
    :class:`SteeringPlan`, and — on compiled CUDA/MPS sessions — the
    persistent per-layer offset buffers traced into the compiled graph.
    """

    def __init__(self) -> None:
        self.hooks: dict[int, SteeringHook] = {}
        self.plan: SteeringPlan | None = None
        self.ctx: TriggerContext = TriggerContext()
        # True iff the installed plan rides the persistent offset buffers
        # (routes the decode loop to the compiled module + StaticCache).
        self.uses_compiled_offsets: bool = False
        self._compiled_offsets: dict[int, torch.Tensor] = {}
        self._compiled_offset_handles: list[Any] = []

    def install(
        self,
        plan: SteeringPlan,
        model_layers: torch.nn.ModuleList,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Make ``plan`` the installed steering, replacing whatever was there.

        A static pure-push plan on a session with adopted offset buffers
        writes them in place (valid on both the compiled graph and the eager
        fallback, which carry the same persistent hooks) and attaches no
        transient hook.  Anything else zeroes the buffers and attaches one
        transient :class:`SteeringHook` per steered layer.  Everything that
        can fail (lowering, overlap checks) runs before any state changes, so
        a raising install leaves the previous plan in place.
        """
        offsets = plan.static_offsets() if self._compiled_offsets else None
        lowered = plan.lower(device, dtype) if offsets is None else {}
        if offsets is not None:
            self._detach_hooks()
            self._write_offsets(offsets)
            self.uses_compiled_offsets = True
        else:
            self._write_offsets({})
            self.uses_compiled_offsets = False
            for idx in list(self.hooks):
                if idx not in lowered:
                    self.hooks.pop(idx).detach()
            for idx, groups in lowered.items():
                hook = self.hooks.get(idx)
                if hook is None:
                    hook = SteeringHook()
                    hook.attach(model_layers[idx])
                    self.hooks[idx] = hook
                hook.recompose(groups, self.ctx)
        self.plan = plan

    def clear_all(self) -> None:
        """Uninstall all steering: detach hooks and zero the offset buffers."""
        self._detach_hooks()
        self._write_offsets({})
        self.uses_compiled_offsets = False
        self.plan = None

    def all_fast_path(self) -> bool:
        """True iff no transient steering hook is attached.

        Holds both unsteered and when the plan rides the compiled offsets, so
        it answers "is graph capture eligible", not "is this unsteered" (read
        :attr:`uses_compiled_offsets` alongside it for that).
        """
        return not self.hooks

    def static_steerable(self) -> bool:
        """True iff steering is attached and every hook is a static affine op.

        Each steered layer then runs one fixed tensor-op sequence with no
        trigger read or foot state, so StaticCache / graph capture stay valid.
        False when unsteered (:meth:`all_fast_path`'s case).
        """
        return bool(self.hooks) and all(h.is_static for h in self.hooks.values())

    def reset_manifold_feet(self) -> None:
        """Cold-start every hook's per-token foot (called at generation start)."""
        for hook in self.hooks.values():
            hook.reset_feet()

    def _detach_hooks(self) -> None:
        for hook in self.hooks.values():
            hook.detach()
        self.hooks.clear()

    # -- persistent compile-clean offset buffers -------------------------------

    def adopt_compiled_offsets(
        self, buffers: dict[int, torch.Tensor], handles: list[Any],
    ) -> None:
        """Adopt the per-layer offset buffers + hook handles.

        Built by :func:`install_persistent_offset_hooks` before
        ``torch.compile`` so the branchless ``add_(offset)`` hooks are traced
        into the captured graph.
        """
        self._compiled_offsets = buffers
        self._compiled_offset_handles = handles

    def has_compiled_offsets(self) -> bool:
        return bool(self._compiled_offsets)

    def _write_offsets(self, offsets: dict[int, torch.Tensor]) -> None:
        # In place (``copy_`` casts to the buffer's device/dtype): reassigning
        # a buffer would change its identity and force a retrace.  Layers the
        # plan doesn't touch are zeroed.
        for L, buf in self._compiled_offsets.items():
            c = offsets.get(L)
            if c is None:
                buf.zero_()
            else:
                buf.copy_(c)

    def detach_compiled_offsets(self) -> None:
        """Remove the persistent offset hooks (compile failure / teardown)."""
        for h in self._compiled_offset_handles:
            h.remove()
        self._compiled_offset_handles = []
        self._compiled_offsets = {}
        self.uses_compiled_offsets = False


def install_persistent_offset_hooks(
    layers: "torch.nn.ModuleList",
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> "tuple[dict[int, torch.Tensor], list[Any]]":
    """Attach one branchless ``hidden.add_(offset)`` forward hook per layer.

    Call **before** ``torch.compile`` so the add is captured in the traced
    graph; :meth:`SteeringManager.install` then updates the zero-initialized
    buffers in place without retracing.  Returns ``(buffers, handles)`` for
    :meth:`SteeringManager.adopt_compiled_offsets`.
    """
    buffers: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def _make(buf: torch.Tensor) -> Any:
        def _hook(module: Any, inp: Any, out: Any) -> Any:
            h = out if isinstance(out, torch.Tensor) else out[0]
            h.add_(buf)
            return out
        return _hook

    for idx in range(len(layers)):
        buf = torch.zeros(hidden_size, device=device, dtype=dtype)
        buffers[idx] = buf
        handles.append(layers[idx].register_forward_hook(_make(buf)))
    return buffers, handles
