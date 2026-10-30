"""SteeringComposer — steering scopes, the steering stack, and plan composition.

Holds the LIFO steering stack and everything that runs at steering push/pop
frequency: opening a scope from a parsed :class:`~drowse.steering.terms.Steering`
(:meth:`SteeringComposer.open_scope` — resolution to registry keys plus the
scope's assistant-role policy), composing the flattened stack head into a
:class:`~drowse.steering.hooks.SteeringPlan` (cached by stack head),
installing it on the :class:`~drowse.steering.hooks.SteeringManager`, and the
probe-gate plumbing.

Its dependencies are explicit: the artifact registry it resolves through, the
neutral baseline it composes against, the steering manager it installs on, the
exclusive model lock, the event bus, and the read-plane sources the per-token
gate callback scores from (:class:`GateSources`).
"""
from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch

from drowse.errors import (
    ConcurrentGenerationError,
    ManifoldArityError,
    RoleBaselineMismatchWarning,
    SteeringExprError,
)
from drowse.events import EventBus, SteeringApplied, SteeringCleared
from drowse.geometry.fold import fold_directions_to_subspace, project_profile
from drowse.geometry.inject import synthesize_subspace
from drowse.steering.grammar import AblationTerm, ManifoldTerm, ProjectedTerm
from drowse.steering.hooks import SteeringManager, SteeringPlan
from drowse.steering.terms import (
    Steering,
    SteeringStackEntry,
    entry_coeff,
    entry_trigger,
    term_entry,
)
from drowse.steering.triggers import Trigger

if TYPE_CHECKING:
    from drowse.geometry.manifold import Manifold
    from drowse.readout.types import GateRef, InstrumentFamily


# Plans retained by stack head.  A ``generate(steering=…)`` composes twice
# (push, and pop back to the outer scope), so a small LRU covers the repeats.
_PLAN_CACHE_SIZE = 16


class ArtifactSource(Protocol):
    """The registry surface steering resolves through."""

    @property
    def profiles(self) -> Mapping[str, dict[int, torch.Tensor]]: ...

    @property
    def manifolds(self) -> "Mapping[str, Manifold]": ...

    def concretize_key(self, key: str) -> str: ...

    def ensure_profile_registered(
        self, name: str, *, role: str = "vector",
    ) -> dict[int, torch.Tensor]: ...

    def ensure_manifold_loaded(self, key: str) -> "Manifold": ...


class NeutralSource(Protocol):
    """The neutral baseline steering composes against."""

    @property
    def means(self) -> dict[int, torch.Tensor]: ...

    @property
    def resident_whitener(self) -> Any: ...

    def ensure(self) -> None: ...

    def require_means(self) -> dict[int, torch.Tensor]: ...


@dataclass(frozen=True)
class GateSources:
    """Read-plane state the per-token probe-gate callback scores from.

    Getters, not values: the session replaces its capture state wholesale per
    generation, and the callback must read the current one on every step.
    """

    monitor: Callable[[], Any]
    capture: Callable[[], Any]
    capture_state: Callable[[], Any]
    incremental_readings: Callable[[], list[Any]]
    incremental_gate_scores: Callable[[], list[dict[str, float]]]
    instruments: Callable[[], Mapping[str, Any]]
    lens_gate_scalars: Callable[..., dict[str, float]]
    sae_gate_scalars: Callable[..., dict[str, float]]


def _fires_in_prefill(trigger: Trigger) -> bool:
    """True iff ``trigger`` touches the prompt residual stream.

    A term steers the prefill exactly when its phase includes the prompt and
    it carries no probe gate (gates report inactive during prefill — there is
    no post-forward score yet).
    """
    return trigger.prompt and trigger.gate is None


def affine_manifold_push(
    manifold: "Manifold", position: "tuple[float, ...] | str",
) -> "tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]":
    """An affine ``%`` term → a rank-R push fragment per layer.

    Both position forms are supported and **equivalent at the nodes**:

    - **Label form** (``position`` a ``str``): the node's per-layer real coords
      (``LayerSubspace.node_coords[idx]``) are the push target on that layer's
      basis — a direct table lookup, no interpolation.
    - **Coord form** (``position`` a coord tuple): the free authoring
      coordinate is mapped into each layer's reduced frame by cardinal RBF
      interpolation over the node layout (:func:`rbf_cardinal_weights` on the
      shared ``Manifold.node_coords``), so the per-layer target is the layout
      blend ``node_coords_L.T @ w``.  The weights are solved once in authoring
      space (layer-agnostic) and are exact at the nodes (``w = e_idx`` at node
      ``idx``), so placing a coord-form term at a node's coords reproduces the
      label-form push toward that node, and interpolates the per-layer target
      between nodes off-node — the flat-subspace analogue of a curved ``%``
      following its RBF surface instead of a straight chord.

    The affine fit's domain is the identity carrier (authoring == embedded), so
    the weights are built directly in authoring-coordinate space.  Arity
    (coord form) is validated against the domain's intrinsic dim, matching the
    curved :meth:`SteeringPlan.add_manifold` path; a non-poised layout (no
    interpolant) raises ``SteeringExprError`` advising the user to steer by
    node label instead.
    """
    from drowse.geometry.rbf import rbf_cardinal_weights

    weights: "torch.Tensor | None" = None
    idx = 0
    if isinstance(position, str):
        idx = manifold.nearest_node_index(position)
    else:
        n = manifold.domain.intrinsic_dim
        pos = tuple(float(c) for c in position)
        if len(pos) != n:
            raise ManifoldArityError(
                f"manifold {manifold.name!r} has a {n}-dimensional domain but "
                f"the steering position has {len(pos)} coordinate(s)"
            )
        query = torch.tensor(pos, dtype=torch.float32)
        try:
            weights = rbf_cardinal_weights(manifold.node_coords, query)
        except ValueError as exc:
            raise SteeringExprError(
                f"manifold {manifold.name!r}: cannot place free coord-form "
                f"position {pos!r} — {exc}; steer by node label instead "
                f"(e.g. '{manifold.name}%<label>')"
            ) from exc

    basis_dirs: dict[int, torch.Tensor] = {}
    coord_dirs: dict[int, torch.Tensor] = {}
    for L, sub in manifold.layers.items():
        if sub.node_coords is None:
            raise SteeringExprError(
                f"affine manifold {manifold.name!r} layer {L} carries no "
                f"per-layer node_coords; re-fit to steer it by node label"
            )
        basis_dirs[L] = sub.basis
        if weights is None:
            coord_dirs[L] = sub.node_coords[idx]
        else:
            w = weights.to(device=sub.node_coords.device, dtype=sub.node_coords.dtype)
            coord_dirs[L] = sub.node_coords.transpose(0, 1) @ w   # (R,)
    return basis_dirs, coord_dirs


def resolve_scope_role(
    entries: Mapping[str, SteeringStackEntry],
    manifolds: "Mapping[str, Manifold]",
) -> tuple[str | None, list[str]]:
    """The assistant-role label a scope's resolved entries imply.

    Two tiers.  Explicit ``:role-<id>`` variants on additive terms must agree
    (mixing roles in one scope has no defined semantics, so a conflict raises
    :class:`SteeringExprError`); a plain term mixed in is allowed but warned
    about, since its extraction baseline was the standard assistant role.
    Each manifold term implies a role through its nearest node; distinct
    implied roles resolve to the highest ``|coeff|`` with a warning.  An
    explicit role wins over a manifold-implied one, again with a warning when
    they disagree.  Nodes that opt out of role substitution abstain.

    Returns ``(role, warning_messages)``.
    """
    role_variants: set[str] = set()
    any_plain_term = False
    for key, entry in entries.items():
        if not isinstance(entry, tuple):
            continue
        if ":" not in key:
            any_plain_term = True
            continue
        variant = key.rsplit(":", 1)[1]
        if variant.startswith("role-"):
            role_variants.add(variant[len("role-"):])
        else:
            # Non-role variants (sae, sae-<rel>) share the standard assistant
            # baseline — plain for baseline-mismatch tracking.
            any_plain_term = True
    if len(role_variants) > 1:
        raise SteeringExprError(
            f"conflicting roles in expression: {sorted(role_variants)}; "
            f"all role-augmented terms must agree on role"
        )
    notes: list[str] = []
    active_role: str | None = next(iter(role_variants), None)
    if active_role is not None and any_plain_term:
        notes.append(
            f"steering scope mixes plain terms with role-augmented "
            f"terms (role={active_role!r}); the plain term's "
            f"extraction baseline was the standard assistant role, "
            f"not {active_role!r}"
        )

    manifold_role: str | None = None
    manifold_role_coeff = -1.0
    disagreement = False
    for entry in entries.values():
        if not isinstance(entry, ManifoldTerm):
            continue
        manifold = manifolds.get(entry.manifold)
        if manifold is None:
            continue
        try:
            implied = manifold.nearest_node_role(entry.position)
        except ValueError:
            implied = None
        if implied is None:
            continue
        abs_c = abs(entry.coeff)
        if manifold_role is None:
            manifold_role, manifold_role_coeff = implied, abs_c
        elif implied == manifold_role:
            manifold_role_coeff = max(manifold_role_coeff, abs_c)
        else:
            disagreement = True
            if abs_c > manifold_role_coeff:
                manifold_role, manifold_role_coeff = implied, abs_c
    if disagreement:
        notes.append(
            f"steering scope has manifold terms implying distinct "
            f"per-node roles; highest-coefficient term wins "
            f"(role={manifold_role!r}, |coeff|={manifold_role_coeff:.3f}). "
            f"Compose with a single manifold term or align the active "
            f"nodes if you want a different role."
        )

    if active_role is None:
        active_role = manifold_role
    elif manifold_role is not None and manifold_role != active_role:
        notes.append(
            f"steering scope has an explicit role={active_role!r} "
            f"and a manifold term implying role={manifold_role!r}; "
            f"explicit role wins, manifold-implied role ignored. "
            f"Use the same role on both, or drop the explicit "
            f":role-{active_role} variant to let the manifold drive."
        )
    return active_role, notes


class SteeringScope:
    """One resolved steering scope (``with session.steering(...)``).

    Entering pushes the resolved entries — and the scope's assistant role —
    onto the composer stack; exiting pops them.  Nested scopes flatten (inner
    entries overwrite outer ones for the inner block) and the innermost scope
    that names a role wins.  The role rides the stack frame, so a failed exit
    cannot leave it bound.
    """

    __slots__ = ("_composer", "_entered", "_entries", "_role")

    def __init__(
        self,
        composer: "SteeringComposer",
        entries: dict[str, SteeringStackEntry],
        role: str | None,
    ) -> None:
        self._composer = composer
        self._entries = entries
        self._role = role
        self._entered = False

    @property
    def entries(self) -> dict[str, SteeringStackEntry]:
        return dict(self._entries)

    @property
    def role(self) -> str | None:
        """The assistant-role label this scope selects (``None`` inherits)."""
        return self._role

    def __enter__(self) -> "SteeringScope":
        self._composer.push(self._entries, role=self._role)
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self, *, internal: bool = False, swallow: bool = False) -> None:
        """Pop this scope if it is entered (idempotent).

        ``internal`` is the generation's own teardown of the scope it opened:
        it passes the mid-generation phase guard that rejects callback
        re-entry.  ``swallow`` drops an ``Exception`` from the pop, for a
        caller already propagating another failure; a ``BaseException``
        always propagates.
        """
        if not self._entered:
            return
        self._entered = False
        try:
            self._composer.pop(internal=internal)
        except Exception:
            if not swallow:
                raise


@dataclass(frozen=True)
class _ProjectionSpec:
    """A resolved ``~``/``|`` term: concrete operand keys and the operator."""

    base: str
    onto: str
    operator: str


@dataclass
class _CachedPlan:
    """A composed plan plus the registry objects it was composed from.

    The plan is reusable exactly while every dependency still resolves to the
    same object (registries replace values rather than mutating them), so a
    re-extract, an eviction, a whitener rebuild, or a neutral refresh misses.
    """

    plan: SteeringPlan
    deps: list[tuple[Callable[[], object], object]]

    def valid(self) -> bool:
        return all(get() is obj for get, obj in self.deps)


class SteeringComposer:
    """Owns the steering stack and steering resolution/composition for a session."""

    def __init__(
        self,
        *,
        artifacts: ArtifactSource,
        neutral: NeutralSource,
        manager: SteeringManager,
        layers: Any,
        device: torch.device,
        dtype: torch.dtype,
        lock: Any,
        events: EventBus,
        generation_phase: Callable[[], str | None],
        on_prefill_steering_change: Callable[[], None],
        gates: GateSources,
    ) -> None:
        """
        ``generation_phase`` returns the phase name while a generation is
        mid-step (``RUNNING`` / ``FINALIZING``) and ``None`` otherwise; a
        stack mutation then is callback re-entry and is refused.
        ``on_prefill_steering_change`` runs when a push or pop leaves the
        prompt prefill steered (the cached prefix is unsteered).
        """
        self._artifacts = artifacts
        self._neutral = neutral
        self._manager = manager
        self._layers = layers
        self._device = device
        self._dtype = dtype
        self._lock = lock
        self._events = events
        self._generation_phase = generation_phase
        self._on_prefill_steering_change = on_prefill_steering_change
        self._gates = gates
        # LIFO stack of resolved per-scope entries; the flattened head (later
        # scopes overwrite earlier keys) is what gets installed.
        self._stack: list[dict[str, SteeringStackEntry]] = []
        # ``(stack depth, role)`` for each frame that names a role.
        self._roles: list[tuple[int, str]] = []
        self._plans: OrderedDict[tuple[Any, ...], _CachedPlan] = OrderedDict()
        # Projection operands by synthetic key, and the projected profile
        # memoized on the identities of its inputs (base, onto, whitener) so
        # plans that use it stay cached.  Projections never enter the registry.
        self._projection_specs: dict[str, _ProjectionSpec] = {}
        self._projections: dict[str, tuple[object, object, object, dict[int, torch.Tensor]]] = {}

    # ------------------------------------------------------------- scopes

    @property
    def active(self) -> bool:
        """True while any steering scope is pushed."""
        return bool(self._stack)

    @property
    def active_role(self) -> str | None:
        """The assistant-role label of the innermost scope that names one."""
        return self._roles[-1][1] if self._roles else None

    def open_scope(
        self, value: "str | Steering", *, stacklevel: int = 1,
    ) -> SteeringScope:
        """Resolve ``value`` into an un-entered :class:`SteeringScope`.

        ``value`` is a steering expression string or a pre-built
        :class:`Steering`.  Every artifact loads here, so an unknown name
        raises before the scope is entered.  Role-policy warnings are issued
        ``stacklevel`` frames above this call.
        """
        steering = Steering.from_value(
            value, profile_names=set(self._artifacts.profiles),
        )
        if steering is None:
            raise TypeError(
                "session.steering() requires a non-None expression string "
                "or Steering instance"
            )
        entries = self.resolve(steering)
        role, notes = resolve_scope_role(entries, self._artifacts.manifolds)
        for note in notes:
            warnings.warn(
                note, RoleBaselineMismatchWarning, stacklevel=stacklevel + 1,
            )
        return SteeringScope(self, entries, role)

    def resolve(self, steering: Steering) -> dict[str, SteeringStackEntry]:
        """Resolve a parsed steering value to stack entries, loading artifacts.

        The one resolution pass after parsing: every vector / ablation target
        concretizes to its registry key (bare ``:sae`` / ``:role`` / ``:from``
        variants pick the unique fitted tensor for this model) and loads,
        every manifold term loads its artifact, and every ``~``/``|`` term
        records its concrete operands (the projection itself is computed at
        composition, against the composed whitener).  Failures raise here —
        an unknown name is :class:`~drowse.errors.ProfileNotRegisteredError` —
        so composition only ever looks loaded artifacts up.
        """
        artifacts = self._artifacts
        entries: dict[str, SteeringStackEntry] = {}
        for name, val in steering.alphas.items():
            if isinstance(val, AblationTerm):
                target = artifacts.concretize_key(val.target)
                artifacts.ensure_profile_registered(target, role="ablation target")
                entries[f"!{target}"] = (
                    val if target == val.target else replace(val, target=target)
                )
                continue
            if isinstance(val, ManifoldTerm):
                manifold_key = artifacts.concretize_key(val.manifold)
                artifacts.ensure_manifold_loaded(manifold_key)
                entries[name] = (
                    val if manifold_key == val.manifold
                    else replace(val, manifold=manifold_key)
                )
                continue
            if isinstance(val, ProjectedTerm):
                spec = _ProjectionSpec(
                    base=artifacts.concretize_key(val.base),
                    onto=artifacts.concretize_key(val.onto),
                    operator=val.operator,
                )
                artifacts.ensure_profile_registered(spec.base, role="projection base")
                artifacts.ensure_profile_registered(spec.onto, role="projection onto")
                if self._projection_specs.get(name) != spec:
                    self._projection_specs[name] = spec
                entries[name] = (float(val.coeff), val.trigger)
                continue
            alpha, trig = term_entry(val, steering.trigger)
            key = name if name in artifacts.profiles else artifacts.concretize_key(name)
            artifacts.ensure_profile_registered(key)
            prev = entries.get(key)
            if isinstance(prev, tuple):
                alpha += prev[0]
            entries[key] = (alpha, trig)
        return entries

    def projection(self, key: str) -> dict[int, torch.Tensor] | None:
        """The projected profile the ``~``/``|`` term ``key`` composed with.

        ``None`` until a plan using it has been composed.  Projections live
        here, never in the artifact registry.
        """
        memo = self._projections.get(key)
        return None if memo is None else memo[3]

    def curved_manifold_terms(self, value: "str | Steering | None") -> list[str]:
        """Registry keys of the curved ``%`` terms in ``value``.

        A curved term's ``along`` is a slide fraction toward a position, so it
        has no sign to invert; a caller that negates steering (loom's
        ``inverted`` regen) can ask which terms that excludes.  Loads each
        manifold term's artifact.
        """
        steering = Steering.from_value(
            value, profile_names=set(self._artifacts.profiles),
        )
        if steering is None:
            return []
        curved: list[str] = []
        for val in steering.alphas.values():
            if not isinstance(val, ManifoldTerm):
                continue
            key = self._artifacts.concretize_key(val.manifold)
            if not self._artifacts.ensure_manifold_loaded(key).is_affine:
                curved.append(key)
        return curved

    # -------------------------------------------------------- stack push / pop

    def push(
        self,
        entries: dict[str, SteeringStackEntry],
        *,
        role: str | None = None,
    ) -> None:
        """Push an entries dict (and its role) and install the new head.

        Transactional: the rebuild composes and lowers before it touches the
        installed plan, so if it raises (a composition error, an overlapping
        manifold) the entry is popped and the outer scope stays installed.

        Thread-safety: acquires the exclusive model lock (blocking), which
        in-flight generations hold for their whole loop, so a push from
        another thread waits instead of mutating hook tensors mid-step.  A
        push from the generation thread while it is mid-step is an
        ``on_token`` callback re-entering the API and raises (the re-entrant
        lock alone would let it through).
        """
        phase = self._generation_phase()
        if phase is not None:
            raise ConcurrentGenerationError(
                "cannot enter session.steering() from inside a generation "
                "callback (e.g. on_token) — the steering stack mutation "
                f"would clobber hook tensors mid-step (gen_phase={phase})"
            )
        with self._lock:
            self._stack.append(dict(entries))
            if role is not None:
                self._roles.append((len(self._stack), role))
            try:
                self.rebuild_hooks()
            except BaseException:
                self._stack.pop()
                self._drop_orphan_roles()
                raise
            # A cached prefix is *unsteered* (``cache_prefix`` refuses to run
            # inside a steering scope), so it only stops matching the
            # pre-attention residual stream when the new head steers the
            # prefill.  A prefill-inactive scope (``@response`` /
            # ``@generated`` / probe-gated) keeps it — which is what lets
            # ``generate_batch`` reuse one prefill across a steered batch.
            if self.steering_active_in_prefill():
                self._on_prefill_steering_change()
        self.emit_steering_applied()

    def pop(self, *, internal: bool = False) -> None:
        """Pop the top of the steering stack and rebuild hooks.

        Mirrors :meth:`push` for thread-safety and the mid-generation guard;
        ``internal`` marks the generation's own teardown of its scope, which
        legitimately runs while the phase is still ``RUNNING`` /
        ``FINALIZING``.

        Unlike :meth:`push`, a failed rebuild cannot keep the previous plan:
        that plan belongs to the scope being exited.  The frame and its role
        still pop and all steering is uninstalled before the error propagates.
        """
        if not self._stack:
            return
        phase = None if internal else self._generation_phase()
        if phase is not None:
            raise ConcurrentGenerationError(
                "cannot exit session.steering() from inside a generation "
                "callback — the steering stack mutation would clobber "
                f"hook tensors mid-step (gen_phase={phase})"
            )
        with self._lock:
            self._stack.pop()
            self._drop_orphan_roles()
            # Fail closed: when the outer head no longer composes (a profile
            # it named was unregistered inside the inner scope), the exited
            # inner plan must not stay installed, so steering goes off and the
            # error propagates.  The next push or pop re-installs from the
            # stack.
            try:
                self.rebuild_hooks()
            except BaseException:
                self._manager.clear_all()
                raise
            if self.steering_active_in_prefill():
                self._on_prefill_steering_change()
        if not self._stack:
            self._events.emit(SteeringCleared())
        else:
            self.emit_steering_applied()

    def _drop_orphan_roles(self) -> None:
        depth = len(self._stack)
        while self._roles and self._roles[-1][0] > depth:
            self._roles.pop()

    def emit_steering_applied(self) -> None:
        """Emit SteeringApplied with alphas-only + full entries.

        ``alphas`` carries the flat ``{name: alpha}`` shape; ``entries``
        carries the full ``{name: (alpha, trigger)}`` mapping.  Ablation
        entries keyed under ``!<target>`` flatten to their ``(coeff,
        trigger)`` pair so subscribers see one uniform tuple shape.
        """
        flat = self.flatten_stack()
        alphas_only: dict[str, float] = {}
        entries_out: dict[str, tuple[float, Trigger]] = {}
        for name, entry in flat.items():
            coeff = entry_coeff(entry)
            alphas_only[name] = coeff
            entries_out[name] = (coeff, entry_trigger(entry))
        self._events.emit(
            SteeringApplied(alphas=alphas_only, entries=entries_out)
        )

    def flatten_stack(self) -> dict[str, SteeringStackEntry]:
        """Collapse the LIFO stack into a single entries dict (later wins)."""
        flat: dict[str, SteeringStackEntry] = {}
        for entry in self._stack:
            flat.update(entry)
        return flat

    # -------------------------------------------------------------- gates

    def steering_needs_probe_gating(self) -> bool:
        """Return True iff any active steering trigger carries a
        :class:`~drowse.steering.triggers.ProbeGate`.

        Cheap pre-flight check that lets generation decide whether to wire
        the per-step score callback at all.
        """
        return any(
            entry_trigger(entry).gate is not None
            for entry in self.flatten_stack().values()
        )

    def _gated_refs(self) -> "list[tuple[str, GateRef]]":
        """Every active probe-gate reference, parsed once into its
        structured form (:func:`parse_gate_ref` — the ONE place the
        scalar-key discrimination lives)."""
        from drowse.readout.types import parse_gate_ref

        refs = []
        for entry in self.flatten_stack().values():
            gate = entry_trigger(entry).gate
            if gate is None:
                continue
            refs.append((gate.probe, parse_gate_ref(gate.probe)))
        return refs

    def gated_probe_names(self) -> set[str]:
        """Registered monitor probe names referenced by active probe gates.

        Only names that are actually attached probes are kept, so a stale /
        non-probe gate key doesn't shrink an otherwise-empty subset to
        "score nothing".  Scopes the per-token step sink to just the gated
        probes when gating is the sole per-token consumer.
        """
        attached = set(self._gates.monitor().probe_names)
        return {
            ref.probe for _key, ref in self._gated_refs()
            if ref.probe in attached
        }

    def gated_probe_keys(self) -> set[str]:
        """Exact monitor scalar keys referenced by active probe gates."""
        attached = set(self._gates.monitor().probe_names)
        return {
            key for key, ref in self._gated_refs()
            if ref.probe in attached
        }

    def gated_family_probe_keys(self, family: "InstrumentFamily") -> set[str]:
        """Exact gate scalar keys referencing one read family's probes.

        A key whose base name is attached to the family is also
        **channel-validated** (``Instrument.validate_gate``): the single-axis
        families produce only the strength channel, so
        ``@when:jlens/word:membership`` raises ``UnsupportedProbeChannelError``
        at generation preflight rather than sitting silently inactive.
        """
        instrument = self._gates.instruments()[family]
        attached = set(instrument.names)
        if not attached:
            return set()
        out: set[str] = set()
        for key, ref in self._gated_refs():
            if ref.probe in attached:
                instrument.validate_gate(ref)
                out.add(key)
        return out

    def gated_lens_probe_keys(self) -> set[str]:
        """Exact gate scalar keys referencing attached J-lens token probes."""
        return self.gated_family_probe_keys("lens")

    def gated_sae_probe_keys(self) -> set[str]:
        """Exact gate scalar keys referencing attached SAE feature probes."""
        return self.gated_family_probe_keys("sae")

    def steering_active_in_prefill(self) -> bool:
        """Return True iff any active steering term fires during prompt prefill.

        When this is False the prefill is numerically identical to the
        *unsteered* prefill, so a cached unsteered prefix KV stays valid for
        reuse; it is the gate the prefix-cache consume path keys on.
        """
        return any(
            _fires_in_prefill(entry_trigger(entry))
            for entry in self.flatten_stack().values()
        )

    def steering_value_prefill_inactive(
        self, value: "str | Steering | None",
    ) -> bool:
        """Return True iff steering ``value`` would not touch the prompt prefill.

        Pre-flight form of :meth:`steering_active_in_prefill` over an
        *incoming* steering value (before any scope push).  ``None`` is
        trivially prefill-inactive; a malformed expression is conservatively
        reported active so the caller skips caching and lets the normal path
        surface the error.
        """
        try:
            s = Steering.from_value(
                value, profile_names=set(self._artifacts.profiles),
            )
        except Exception:
            return False
        if s is None:
            return True
        return not any(_fires_in_prefill(trig) for trig in s.triggers())

    def build_gating_score_callback(self):
        """Return a closure that scores latest captures into a
        ``dict[str, float]`` for ``generate_steered``'s ``score_callback``.

        Geometry gate keys read the capture sink's row for this forward when
        one exists, else score exactly the planned keys from the capture's
        latest per-layer rows (:meth:`Monitor.score_planned_gate_scalars`).
        Keys naming no attached probe score nothing.  Returns an empty dict
        when the capture is empty (e.g. before the first forward) so probe
        gates report inactive instead of seeing stale values.

        Hot path: the returned closure runs per token under an active gate; it
        binds the capture and monitor once and reads the per-generation
        capture state through the :class:`GateSources` getters.
        """
        gates = self._gates
        capture = gates.capture()
        monitor = gates.monitor()
        monitor_gate_keys = self.gated_probe_keys()
        lens_gate_keys = self.gated_lens_probe_keys()
        sae_gate_keys = self.gated_sae_probe_keys()
        has_lens_gates = bool(lens_gate_keys)
        has_sae_gates = bool(sae_gate_keys)
        monitor_gate_plan_cache: dict[
            tuple[frozenset[str], frozenset[str] | None],
            Any,
        ] = {}

        def _score_monitor_gate_keys(
            latest: dict[int, torch.Tensor],
            gate_keys: set[str],
            *,
            probe_names: set[str] | None = None,
        ) -> dict[str, float]:
            if not gate_keys:
                return {}
            frozen_keys = frozenset(gate_keys)
            frozen_probes = (
                frozenset(probe_names) if probe_names is not None else None
            )
            cache_key = (frozen_keys, frozen_probes)
            plan = monitor_gate_plan_cache.get(cache_key)
            if plan is None:
                plan = monitor.plan_gate_scalars(
                    set(frozen_keys),
                    probe_names=(
                        set(frozen_probes) if frozen_probes is not None else None
                    ),
                )
                monitor_gate_plan_cache[cache_key] = plan
            return (
                cast(dict[str, float], monitor.score_planned_gate_scalars(latest, plan))
                if plan else {}
            )

        def _monitor_scalars() -> dict[str, float]:
            incremental_readings = gates.incremental_readings()
            incremental_gate_scores = gates.incremental_gate_scores()
            state = gates.capture_state()
            gating_subset = state.gating_subset
            gate_keys = state.gating_keys or monitor_gate_keys
            if not gate_keys:
                # No gate names an attached geometry probe — nothing the
                # monitor could score would ever fire, so score nothing.
                return {}
            # The step sink already scored this token — reuse its row.  In
            # gating-only-subset mode the rows hold just the gated scalars; in
            # FULL-incremental mode the whole roster's readings.
            if gating_subset and incremental_gate_scores:
                return incremental_gate_scores[-1]
            if state.incremental and incremental_readings:
                scalars = monitor.flat_scalars(incremental_readings[-1])
                missing = gate_keys - set(scalars)
                if missing:
                    latest = capture.latest_per_layer()
                    if latest:
                        scalars.update(
                            _score_monitor_gate_keys(
                                latest,
                                missing,
                                probe_names=gating_subset if gating_subset else None,
                            )
                        )
                return scalars
            latest = capture.latest_per_layer()
            if not latest:
                return {}
            return _score_monitor_gate_keys(
                latest,
                gate_keys,
                probe_names=gating_subset if gating_subset else None,
            )

        def _score(step_id: int) -> dict[str, float]:
            out = _monitor_scalars()
            if has_lens_gates:
                # Once per forward: band lens logits → strength scalars (also
                # stashed step-keyed for the display step to reuse).
                lens_scalars = gates.lens_gate_scalars(
                    lens_gate_keys, step_id=step_id,
                )
                if lens_scalars:
                    out = {**out, **lens_scalars}
            if has_sae_gates:
                sae_scalars = gates.sae_gate_scalars(
                    sae_gate_keys, step_id=step_id,
                )
                if sae_scalars:
                    out = {**out, **sae_scalars}
            return out

        return _score

    # ------------------------------------------------------------- dispatch

    def compose(
        self,
        entries: dict[str, SteeringStackEntry],
        deps: "list[tuple[Callable[[], object], object]] | None" = None,
    ) -> SteeringPlan:
        """Compose resolved stack entries into a fresh :class:`SteeringPlan`.

        Pure with respect to the installed state: nothing on the manager
        changes, so a raising composition leaves the outer scope installed.

        - push terms (vectors, poles, projections, affine ``%``) and ablation
          terms (``!``) are grouped by trigger, and each group is synthesized
          into one merged affine subspace
          (:func:`~drowse.geometry.inject.synthesize_subspace`);
        - each curved ``%`` term registers on its own.

        A vector lowers through the one push path: folded to a neutral-anchored
        one-pole ray and pushed label-form, exactly like an affine ``%`` term.
        Ablations slide their axis to the neutral-anchored origin.  ``deps``
        (when given) collects every registry object read, for the plan cache.

        The neutral baseline is built here if absent: composition runs under
        the push's exclusive lock, so it goes through the unguarded
        ``ensure`` rather than the public getters.
        """
        record = deps.append if deps is not None else (lambda _dep: None)
        artifacts = self._artifacts
        neutral = self._neutral
        neutral.ensure()
        neutral_means = neutral.require_means()
        whitener = neutral.resident_whitener
        record((lambda: neutral.means, neutral_means))
        record((lambda: neutral.resident_whitener, whitener))

        def _profile(name: str, role: str) -> dict[int, torch.Tensor]:
            prof = artifacts.ensure_profile_registered(name, role=role)
            record((lambda: artifacts.profiles.get(name), prof))
            return prof

        def _projected(name: str, spec: _ProjectionSpec) -> dict[int, torch.Tensor]:
            base = _profile(spec.base, "projection base")
            onto = _profile(spec.onto, "projection onto")
            memo = self._projections.get(name)
            if (
                memo is not None and memo[0] is base and memo[1] is onto
                and memo[2] is whitener
            ):
                return memo[3]
            projected = project_profile(base, onto, spec.operator, whitener=whitener)
            self._projections[name] = (base, onto, whitener, projected)
            return projected

        plan = SteeringPlan()
        # trigger -> push [(basis_dirs, coord_dirs, coeff)] / ablate [(dirs, coeff)]
        # Ablation coefficients stay attached to their directions into
        # synthesis (``0.15 !x`` is a partial ablation, not a full one).
        grouped: dict[Trigger, tuple[list[Any], list[Any]]] = {}
        for name, entry in entries.items():
            if isinstance(entry, AblationTerm):
                ablate_prof = _profile(entry.target, "ablation target")
                ablate_dirs = {
                    L: v.to(torch.float32).reshape(-1)
                    for L, v in ablate_prof.items()
                }
                grouped.setdefault(entry.trigger, ([], []))[1].append(
                    (ablate_dirs, entry.coeff),
                )
                continue
            if isinstance(entry, ManifoldTerm):
                manifold = artifacts.ensure_manifold_loaded(entry.manifold)
                record((lambda key=entry.manifold: artifacts.manifolds.get(key), manifold))
                if manifold.is_affine:
                    # Affine ``%`` joins the merged subspace as a rank-R push
                    # toward the position's per-layer coords.
                    basis_dirs, coord_dirs = affine_manifold_push(
                        manifold, entry.position,
                    )
                    grouped.setdefault(entry.trigger, ([], []))[0].append(
                        (basis_dirs, coord_dirs, entry.along),
                    )
                else:
                    plan.add_manifold(
                        entry.manifold, manifold,
                        position=entry.position,
                        along=entry.along, onto=entry.onto,
                        trigger=entry.trigger,
                    )
                continue
            alpha, trigger = entry
            spec = self._projection_specs.get(name)
            directions = (
                _projected(name, spec) if spec is not None
                else _profile(name, "vector")
            )
            folded = fold_directions_to_subspace(
                name, directions, neutral_means, whitener=whitener,
            )
            if not folded.layers:
                continue
            basis_dirs, coord_dirs = affine_manifold_push(
                folded, folded.node_labels[0],
            )
            grouped.setdefault(trigger, ([], []))[0].append(
                (basis_dirs, coord_dirs, alpha),
            )

        for i, (trigger, (push, ablate)) in enumerate(grouped.items()):
            synth = synthesize_subspace(
                push, ablate, neutral_means=neutral_means, whitener=whitener,
            )
            if synth.layers:
                plan.add_subspace(f"__affine__{i}", synth, trigger=trigger)
        return plan

    def plan_for(self, entries: dict[str, SteeringStackEntry]) -> SteeringPlan:
        """The composed plan for a flattened stack head, cached by its entries.

        Stack entries are frozen, so the head itself is the key; a hit is
        reused only while every registry object it was composed from is still
        the one registered.  The plan memoizes its own lowering, so a hit also
        skips :class:`~drowse.steering.hooks.AffineOp` compilation.
        """
        key = tuple(entries.items())
        cached = self._plans.get(key)
        if cached is not None and cached.valid():
            self._plans.move_to_end(key)
            return cached.plan
        deps: list[tuple[Callable[[], object], object]] = []
        for name in entries:
            spec = self._projection_specs.get(name)
            if spec is not None:
                deps.append((lambda n=name: self._projection_specs.get(n), spec))
        plan = self.compose(entries, deps)
        self._plans[key] = _CachedPlan(plan, deps)
        self._plans.move_to_end(key)
        while len(self._plans) > _PLAN_CACHE_SIZE:
            self._plans.popitem(last=False)
        return plan

    def rebuild_hooks(self) -> None:
        """Install the flattened stack head (or clear, when the stack is empty).

        The installed steering is a function of the stack: an empty stack
        uninstalls everything (offset buffers included), and a non-empty one
        installs its composed plan.  Composition and lowering run before the
        manager changes, so a failure leaves the previous plan installed.
        """
        flat = self.flatten_stack()
        if not flat:
            self._manager.clear_all()
            return
        plan = self.plan_for(flat)
        self._manager.install(plan, self._layers, self._device, self._dtype)

    def snapshot_steering_alphas(self) -> dict[str, float]:
        """Flatten the active steering stack for result steering receipts.

        For a plain vector entry, ``entry[0]`` is the signed push alpha.  For
        an ablation it is the ablation strength, and for a
        :class:`~drowse.steering.grammar.ManifoldTerm` it is ``along``: the
        push magnitude toward the position on an affine fit, a slide fraction
        capped at 1 on a curved one.  All are reported as one ``float`` for a
        uniform receipt; callers that need the kind read ``Steering.alphas``.
        """
        snap: dict[str, float] = {}
        for name, entry in self.flatten_stack().items():
            if isinstance(entry, (AblationTerm, ManifoldTerm)):
                snap[name] = entry.coeff
                continue
            snap[name] = entry[0]
        return snap
