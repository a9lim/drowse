"""SteeringComposer — the steering stack, resolution, and plan composition.

Holds the LIFO steering stack and everything that runs at steering push/pop
frequency: resolving a parsed :class:`~drowse.steering.terms.Steering` to
registry keys and loaded artifacts (:meth:`SteeringComposer.resolve`, once per
scope), composing the flattened stack head into a
:class:`~drowse.steering.hooks.SteeringPlan` (cached by stack head), installing
it on the :class:`~drowse.steering.hooks.SteeringManager`, and the probe-gate
plumbing.  It reaches the owning session through ``self._session`` at push/pop
frequency only; the one per-token path is the closure
:meth:`build_gating_score_callback` returns.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, cast

import torch

from drowse.events import SteeringApplied, SteeringCleared
from drowse.session.session import (
    ConcurrentGenerationError,
    GenState,
    ManifoldNotRegisteredError,
    SteeringStackEntry,
    Trigger,
    ProfileNotRegisteredError,
    _PROFILE_ABSENT,
    _affine_manifold_push,
)
from drowse.steering.hooks import SteeringPlan
from drowse.steering.terms import entry_coeff, entry_trigger, term_entry
from drowse.steering.grammar import (
    RESERVED_NAMESPACES,
    AblationTerm,
    ManifoldTerm,
    ProjectedTerm,
)

if TYPE_CHECKING:
    from drowse.readout.types import GateRef, InstrumentFamily
    from drowse.session.session import DrowseSession
    from drowse.steering.terms import Steering


# Variants that name a family of fitted tensors rather than one tensor; they
# concretize against the files on disk for the loaded model.
_BARE_VARIANTS = frozenset({"sae", "from", "role"})

# Plans retained by stack head.  A ``generate(steering=…)`` composes twice
# (push, and pop back to the outer scope), so a small LRU covers the repeats.
_PLAN_CACHE_SIZE = 16


def _split_key(key: str) -> tuple[str | None, str, str]:
    """``[ns/]name[:variant]`` → ``(ns, name, variant)`` (variant ``raw`` if absent)."""
    name_part, variant = key.rsplit(":", 1) if ":" in key else (key, "raw")
    if "/" in name_part:
        ns, name = name_part.split("/", 1)
        return ns, name, variant
    return None, name_part, variant


def _fires_in_prefill(trigger: Trigger) -> bool:
    """True iff ``trigger`` touches the prompt residual stream.

    A term steers the prefill exactly when its phase includes the prompt and
    it carries no probe gate (gates report inactive during prefill — there is
    no post-forward score yet).
    """
    return trigger.prompt and trigger.gate is None


@dataclass
class _CachedPlan:
    """A composed plan plus the registry objects it was composed from.

    The plan is reusable exactly while every dependency still resolves to the
    same object (registries replace values rather than mutating them), so a
    re-extract, ``unsteer``, whitener rebuild, or neutral refresh misses.
    """

    plan: SteeringPlan
    deps: list[tuple[Callable[[], object], object]]

    def valid(self) -> bool:
        return all(get() is obj for get, obj in self.deps)


class SteeringComposer:
    """Owns the steering stack + steering resolution/composition for a session.

    Instantiated once per :class:`~drowse.session.session.DrowseSession` and
    held as ``_steering_composer``; the stack has one owner, this object.
    """

    def __init__(self, session: "DrowseSession") -> None:
        self._session = session
        # LIFO stack of resolved per-scope entries; the flattened head (later
        # scopes overwrite earlier keys) is what gets installed.
        self._stack: list[dict[str, SteeringStackEntry]] = []
        self._plans: OrderedDict[tuple[Any, ...], _CachedPlan] = OrderedDict()
        # Projection memo: syn_key → (base, onto, whitener, projected).  Keeps
        # the registered derived profile's identity stable across scopes so
        # plans that use it stay cached.
        self._projections: dict[str, tuple[object, object, object, object]] = {}

    # ---------------------------------------------------------- resolution

    def resolve(
        self, steering: "Steering",
    ) -> tuple[dict[str, SteeringStackEntry], dict[str, object]]:
        """Resolve a parsed steering value to stack entries, loading artifacts.

        The one resolution pass after parsing: every vector / ablation target
        concretizes to its registry key (bare ``:sae`` / ``:role`` / ``:from``
        variants pick the unique fitted tensor for this model) and loads, every
        manifold term loads its artifact, and ``~``/``|`` projections
        materialize under their synthetic key.  Failures raise here — an
        unknown name is :class:`ProfileNotRegisteredError` — so composition
        only ever looks loaded artifacts up.

        Returns ``(entries, snapshots)``; ``snapshots`` are the projection
        bindings this call replaced, for the scope to restore on exit.
        """
        snapshots = self.materialize_projections(steering)
        profiles = self._session._profiles
        entries: dict[str, SteeringStackEntry] = {}
        for name, val in steering.alphas.items():
            if isinstance(val, AblationTerm):
                target = self.concretize_key(val.target)
                self.ensure_profile_registered(target, role="ablation target")
                entries[f"!{target}"] = (
                    val if target == val.target else replace(val, target=target)
                )
                continue
            if isinstance(val, ManifoldTerm):
                manifold_key = self.concretize_key(val.manifold)
                self.ensure_manifold_loaded(manifold_key)
                entries[name] = (
                    val if manifold_key == val.manifold
                    else replace(val, manifold=manifold_key)
                )
                continue
            alpha, trig = term_entry(val, steering.trigger)
            key = name if name in profiles else self.concretize_key(name)
            self.ensure_profile_registered(key)
            prev = entries.get(key)
            if isinstance(prev, tuple):
                alpha += prev[0]
            entries[key] = (alpha, trig)
        return entries, snapshots

    def concretize_key(self, key: str) -> str:
        """Pin a bare ``:sae`` / ``:role`` / ``:from`` variant to its fitted tensor.

        Any other key is returned unchanged.  Raises
        :class:`~drowse.errors.UnknownVariantError` when no fitted variant
        exists for the loaded model and
        :class:`~drowse.errors.AmbiguousVariantError` when several do.
        """
        ns, name, variant = _split_key(key)
        if variant not in _BARE_VARIANTS:
            return key
        import json

        from drowse.base.paths import (
            manifold_dir, manifolds_dir, parse_tensor_filename, safe_model_id,
        )
        from drowse.errors import AmbiguousVariantError, UnknownVariantError

        if ns is not None:
            roots = [manifold_dir(ns, name)]
        elif manifolds_dir().exists():
            roots = [
                entry / name for entry in manifolds_dir().iterdir()
                if entry.is_dir()
            ]
        else:
            roots = []
        concrete: set[str] = set()
        safe_model = safe_model_id(self._session.model_id)
        for folder in roots:
            if variant == "role":
                sidecar_path = folder / f"{safe_model}.json"
                if not sidecar_path.exists():
                    continue
                with open(sidecar_path) as handle:
                    raw_sidecar = json.load(handle)
                roles = raw_sidecar.get("node_roles") or [
                    None
                ] * int(raw_sidecar.get("node_count", 0))
                if roles and roles[0] is not None and all(
                    role == roles[0] for role in roles
                ):
                    concrete.add(f"role-{roles[0]}")
                continue
            for tensor_path in folder.glob(f"{safe_model}_*.safetensors"):
                parsed = parse_tensor_filename(tensor_path.name)
                if parsed is None or parsed[1] is None:
                    continue
                if parsed[1].startswith(f"{variant}-"):
                    concrete.add(parsed[1])
        qualified = name if ns is None else f"{ns}/{name}"
        if len(concrete) != 1:
            error = UnknownVariantError if not concrete else AmbiguousVariantError
            raise error(
                f"manifold '{qualified}:{variant}' resolves to "
                f"{sorted(concrete) or 'no fitted variants'}"
            )
        return f"{qualified}:{next(iter(concrete))}"

    def materialize_projections(self, steering: "Steering") -> dict[str, object]:
        """Register the derived profile of every ``~``/``|`` term in ``steering``.

        Loads ``base`` and ``onto``, projects with the closed-form LEACE
        projector against the session whitener (Mahalanobis-only:
        ``project_profile`` raises ``WhitenerError`` without a covering
        whitener), and registers the result under the synthetic key
        ``"<base><op><onto>"``.  The projection is memoized on the identities
        of its inputs, so re-entering the same scope reuses one profile.

        Returns ``{syn_key: prior binding or _PROFILE_ABSENT}`` for the bindings
        this call replaced, so the scope can restore them on exit (the
        registry is shared across nested scopes).
        """
        from drowse.fit.capture import project_profile

        profiles = self._session._profiles
        snapshots: dict[str, object] = {}
        whitener = None
        for syn_key, val in steering.alphas.items():
            if not isinstance(val, ProjectedTerm):
                continue
            # Only an actual projection wakes the lazy whitener.
            if whitener is None:
                whitener = self._session.whitener
            base_key = self.concretize_key(val.base)
            onto_key = self.concretize_key(val.onto)
            base = self.ensure_profile_registered(base_key, role="projection base")
            onto = self.ensure_profile_registered(onto_key, role="projection onto")
            memo = self._projections.get(syn_key)
            if memo is not None and memo[0] is base and memo[1] is onto and memo[2] is whitener:
                projected = memo[3]
            else:
                projected = project_profile(
                    base, onto, val.operator, whitener=whitener,
                )
                self._projections[syn_key] = (base, onto, whitener, projected)
            if syn_key not in snapshots:
                snapshots[syn_key] = profiles.get(syn_key, _PROFILE_ABSENT)
            profiles[syn_key] = projected  # pyright: ignore[reportArgumentType]  # memo stores the projected profile dict
        return snapshots

    def ensure_manifold_loaded(self, key: str) -> None:
        """Load the fitted manifold for registry key ``[ns/]name[:variant]``.

        ``raw`` selects the residual-stream tensor, ``sae-<release>`` the SAE
        fit, ``from-<src>`` a cross-model transfer, and ``role-<id>`` the
        canonical tensor with its role baseline checked.  A bare name searches
        every namespace (and refuses a collision).  The loaded manifold is
        validated against the live weights and capture inputs, then promoted
        onto the session device in fp32.  Raises
        :class:`ManifoldNotRegisteredError` on a miss or a stale fit.
        """
        manifolds = self._session._manifolds
        if key in manifolds:
            return
        from drowse.base.paths import (
            encode_release_id, manifold_dir, manifolds_dir,
            tensor_filename_for_variant,
        )
        from drowse.model.fingerprint import loaded_model_fingerprint
        from drowse.store.manifold_schema import sidecar_is_current
        from drowse.store.manifold_tensors import load_manifold

        model_id = self._session.model_id
        ns, name, variant = _split_key(key)
        if ns is not None:
            search_ns = [ns]
        else:
            root = manifolds_dir()
            search_ns = (
                sorted(d.name for d in root.iterdir() if d.is_dir())
                if root.exists() else []
            )
        requested_role = (
            variant[len("role-"):] if variant.startswith("role-") else None
        )
        transferred_from = (
            variant[len("from-"):] if variant.startswith("from-") else None
        )
        fname = tensor_filename_for_variant(model_id, variant)
        if fname is None:
            raise ManifoldNotRegisteredError(
                f"manifold '{key}': unsupported variant '{variant}'"
            )
        # An older artifact format is a miss (refit), not a corrupt artifact.
        matches = [
            (candidate_ns, path)
            for candidate_ns, path in (
                (candidate_ns, manifold_dir(candidate_ns, name) / fname)
                for candidate_ns in search_ns
            )
            if path.exists()
            and sidecar_is_current(path.with_suffix(".json"))
        ]
        if not matches:
            raise ManifoldNotRegisteredError(
                f"manifold '{key}' has no fitted tensor for {model_id}; "
                f"run `drowse manifold fit` first"
            )
        if len(matches) > 1:
            from drowse.store.selectors import AmbiguousSelectorError
            qualified = ", ".join(f"{m_ns}/{name}" for m_ns, _ in matches)
            raise AmbiguousSelectorError(
                f"ambiguous manifold '{name}': matches {qualified}. "
                f"Qualify it with a namespace."
            )
        tensor_path = matches[0][1]
        manifold = load_manifold(tensor_path)
        metadata = manifold.metadata
        live_fingerprint = loaded_model_fingerprint(self._session._model, model_id)
        if transferred_from is not None:
            source_id = metadata.get("source_model_id")
            if (
                not isinstance(source_id, str)
                or encode_release_id(source_id) != transferred_from
            ):
                raise ManifoldNotRegisteredError(
                    f"manifold '{key}' transfer provenance does not match "
                    f"source {transferred_from!r}; recompute the transfer"
                )
        raw_roles = metadata.get("node_roles")
        roles = (
            list(raw_roles)
            if isinstance(raw_roles, list)
            else [None] * len(manifold.node_labels)
        )
        if requested_role is not None and (
            not roles or not all(role == requested_role for role in roles)
        ):
            raise ManifoldNotRegisteredError(
                f"manifold '{key}' was not fitted uniformly with role "
                f"{requested_role!r}; refit it with that role"
            )
        if variant == "raw" and any(role is not None for role in roles):
            raise ManifoldNotRegisteredError(
                f"manifold '{key}' was fitted with role baseline "
                f"{roles!r}; steer it as :role-<name> or refit raw"
            )
        if metadata.get("model_fingerprint") != live_fingerprint:
            raise ManifoldNotRegisteredError(
                f"manifold '{key}' was fitted for different loaded weights "
                f"under {model_id}; run `drowse manifold fit` again"
            )
        if metadata.get("fit_mode", "authored") != "baked":
            from drowse.fit.pipeline import prepare_manifold_capture_identity
            from drowse.store.manifold_folder import ManifoldFolder

            folder_mf = ManifoldFolder.load(
                tensor_path.parent, verify_manifest=False,
            )
            expected_capture = prepare_manifold_capture_identity(
                self._session, folder_mf, live_fingerprint,
            ).capture_sha256
            if metadata.get("capture_sha256") != expected_capture:
                raise ManifoldNotRegisteredError(
                    f"manifold '{key}' capture inputs changed (tokenizer, chat "
                    "template, role framing, or baseline prompts); refit it"
                )
        manifolds[key] = manifold.to(
            device=self._session._device, dtype=torch.float32,
        )

    def ensure_profile_registered(
        self, name: str, *, role: str = "vector",
    ) -> dict[int, torch.Tensor]:
        """Direction profile for registry key ``name``, loading it if absent.

        In order: an in-memory profile already registered (extracts, clones,
        merges, projection derivations); a reserved instrument atom
        (``jlens/<word|#id>`` through the fitted Jacobian lens, ``sae/<id>``
        through the resident SAE decoder); a fitted 2-node ``pca`` manifold
        folded to its per-layer direction.  Raises
        :class:`ProfileNotRegisteredError` when nothing resolves.
        """
        profiles = self._session._profiles
        if name in profiles:
            return profiles[name]
        ns, atom, variant = _split_key(name)
        if ns in RESERVED_NAMESPACES and variant == "raw":
            if ns == "jlens":
                return profiles[self._session.lens.register_direction(atom)]
            if ns == "sae":
                return profiles[self._session.register_sae_direction(atom)]
            raise ProfileNotRegisteredError(
                f"'{ns}/' atoms are reserved and not steerable yet ({name!r})"
            )
        folded = self.try_fold_manifold(name, role=role)
        profiles[name] = folded
        return folded

    def try_fold_manifold(
        self, name: str, *, role: str = "vector",
    ) -> dict[int, torch.Tensor]:
        """Load the fitted manifold ``name`` and fold it to a direction profile.

        A manifold miss becomes :class:`ProfileNotRegisteredError` (chained to
        the miss); any other load failure (ambiguity, corrupt artifact)
        propagates as itself.  A manifold that loads but isn't a 2-node affine
        subspace is a usage error.
        """
        try:
            self.ensure_manifold_loaded(name)
        except ManifoldNotRegisteredError as miss:
            raise ProfileNotRegisteredError(
                f"No profile registered for {role} '{name}' "
                f"({miss.args[0] if miss.args else miss})"
            ) from miss
        from drowse.fit.capture import folded_directions
        try:
            return folded_directions(self._session._manifolds[name])
        except Exception as e:
            raise ProfileNotRegisteredError(
                f"'{name}' is a manifold that does not fold to a single "
                f"steering direction (not a 2-node affine subspace): {e}"
            ) from e

    # -------------------------------------------------------- stack push / pop

    def push(
        self,
        entries: dict[str, SteeringStackEntry],
    ) -> None:
        """Push an entries dict onto the steering stack and install the new head.

        Transactional: the rebuild composes and lowers before it touches the
        installed plan, so if it raises (a composition error, an overlapping
        manifold) the entry is popped and the outer scope stays installed.

        Thread-safety: acquires :attr:`_gen_lock` (blocking), which in-flight
        generations hold for their whole loop, so a push from another thread
        waits instead of mutating hook tensors mid-step.  Phase guard: a push
        from the generation thread while ``RUNNING`` / ``FINALIZING`` is an
        ``on_token`` callback re-entering the API and raises (the RLock alone
        would let it through).  Legitimate callers push during ``PREAMBLE``
        (``_generate_core``'s own scope) or ``IDLE``.
        """
        session = self._session
        # ``_generate_core``'s internal ``steering_cm.__enter__()`` runs
        # during ``PREAMBLE`` (before the ``RUNNING`` flip), so push
        # never needs the ``_internal_steering_pop`` bypass — only the
        # exit path fires under ``RUNNING``/``FINALIZING``.  The check
        # here catches genuine callback reentry where ``on_token`` /
        # ``score_callback`` calls back into ``session.steering(...)``
        # mid-step.
        if session._gen_phase in (GenState.RUNNING, GenState.FINALIZING):
            raise ConcurrentGenerationError(
                "cannot enter session.steering() from inside a generation "
                "callback (e.g. on_token) — the steering stack mutation "
                f"would clobber hook tensors mid-step (gen_phase="
                f"{session._gen_phase.name})"
            )
        with session._gen_lock:
            self._stack.append(dict(entries))
            try:
                # Rebuild through the session boundary after the stack mutation.
                session._rebuild_steering_hooks()
            except BaseException:
                self._stack.pop()
                raise
            # Steering hooks just changed.  A cached prefix is *unsteered*
            # (``cache_prefix`` refuses to run inside a steering scope), so it
            # only stops representing the current pre-attention residual stream
            # when the new stack actually steers the prefill.  A prefill-
            # inactive scope (``@response`` / ``@generated`` / probe-gated)
            # leaves the prompt region untouched, so the unsteered prefix stays
            # valid and we keep it — this is what lets ``generate_batch`` reuse
            # one prefill across a steered batch.  Drop it only when prefill is
            # genuinely steered.
            if self.steering_active_in_prefill():
                session._invalidate_prefix_cache()
        self.emit_steering_applied()

    def pop(self) -> None:
        """Pop the top of the steering stack and rebuild hooks.

        Mirrors :meth:`push` for thread-safety and phase
        guarding: acquires :attr:`_gen_lock` so the rebuild can't fire
        mid-step in another thread's generation, and rejects same-
        thread callback reentry during ``RUNNING`` / ``FINALIZING``.

        Unlike :meth:`push`, a failed rebuild cannot keep the previous plan:
        that plan belongs to the scope being exited.  The stack still pops and
        all steering is uninstalled before the error propagates.
        """
        session = self._session
        if not self._stack:
            return
        # Same internal-vs-callback distinction as ``push``.
        # ``_generate_core``'s finally block sets the flag around the
        # ``steering_cm.__exit__()`` it owns, so its own scope cleanup
        # passes through; callback reentry (on_token, score_callback)
        # from inside the running loop hits the reject path.
        if (
            not session._internal_steering_pop
            and session._gen_phase in (GenState.RUNNING, GenState.FINALIZING)
        ):
            raise ConcurrentGenerationError(
                "cannot exit session.steering() from inside a generation "
                "callback — the steering stack mutation would clobber "
                f"hook tensors mid-step (gen_phase={session._gen_phase.name})"
            )
        with session._gen_lock:
            self._stack.pop()
            # Rebuild through the session boundary after the stack mutation.
            # Fail closed: when the outer head no longer composes (a profile
            # it named was unregistered inside the inner scope), the exited
            # inner plan must not stay installed, so steering goes off and the
            # error propagates.  The next push or pop re-installs from the
            # stack.
            try:
                session._rebuild_steering_hooks()
            except BaseException:
                session._steering.clear_all()
                raise
            # Symmetric with ``push``: the cached prefix is unsteered,
            # so only invalidate when what remains on the stack still steers the
            # prefill.  Popping back to a prefill-inactive (or empty) stack keeps
            # the unsteered prefix valid for the next reuse.
            if self.steering_active_in_prefill():
                session._invalidate_prefix_cache()
        if not self._stack:
            session.events.emit(SteeringCleared())
        else:
            self.emit_steering_applied()

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
        self._session.events.emit(
            SteeringApplied(alphas=alphas_only, entries=entries_out)
        )

    def flatten_stack(self) -> dict[str, SteeringStackEntry]:
        """Collapse the LIFO stack into a single entries dict (later wins)."""
        flat: dict[str, SteeringStackEntry] = {}
        for entry in self._stack:
            flat.update(entry)
        return flat

    def steering_needs_probe_gating(self) -> bool:
        """Return True iff any active steering trigger carries a
        :class:`~drowse.steering.triggers.ProbeGate`.

        Walks the flattened steering stack head through
        :func:`~drowse.steering.terms.entry_trigger`, so every entry shape is
        inspected the same way.  Cheap pre-flight check that lets
        ``_generate_core`` decide whether to wire the per-step score callback
        at all.
        """
        return any(
            entry_trigger(entry).gate is not None
            for entry in self.flatten_stack().values()
        )

    def _gated_refs(self) -> "list[tuple[str, GateRef]]":
        """Every active probe-gate reference, parsed once into its
        structured form (:func:`parse_gate_ref` — the ONE place the
        scalar-key discrimination lives).  Walks the flattened steering
        stack exactly like the historical per-family key walks, which this
        replaces.  Note the structured parse also fixes the old
        ``re.split("[\\[:@~]")`` truncation of variant-suffixed probe names
        (``pirate:role-x`` no longer truncates at ``:role-x``).
        """
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
        "score nothing".  Used by ``_begin_capture`` to scope the per-token
        step sink to just the gated probes when gating is the sole
        per-token consumer (FIX #4).
        """
        attached = set(self._session._monitor.probe_names)
        return {
            ref.probe for _key, ref in self._gated_refs()
            if ref.probe in attached
        }

    def gated_probe_keys(self) -> set[str]:
        """Exact monitor scalar keys referenced by active probe gates."""
        attached = set(self._session._monitor.probe_names)
        return {
            key for key, ref in self._gated_refs()
            if ref.probe in attached
        }

    def gated_family_probe_keys(self, family: "InstrumentFamily") -> set[str]:
        """Exact gate scalar keys referencing one read family's probes.

        Driven by ``session.instruments`` — the registry IS the family
        enumeration, so this is one loop rather than a per-family copy.
        Lens and SAE probes live in their instruments' own registries (the
        readout channels), not the Monitor, so :meth:`gated_probe_keys`'
        attachment test doesn't see them.

        A key whose base name is attached to the family is also
        **channel-validated** here (``Instrument.validate_gate``): the
        single-axis families produce only the strength channel, so
        ``@when:jlens/word:membership`` raises
        ``UnsupportedProbeChannelError`` at generation preflight rather than
        sitting silently inactive.  Geometry accepts every channel, so the
        same call is a no-op there.
        """
        instrument = self._session.instruments[family]
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

        A term touches the prefill residual stream iff its trigger has
        ``prompt=True`` and carries no probe gate (probe gates report
        inactive during prefill — there's no post-forward score yet; see
        :meth:`~drowse.steering.triggers.Trigger.active`).  When this is False the
        prefill is numerically identical to the *unsteered* prefill, so a
        cached unsteered prefix KV (the only kind :meth:`cache_prefix` builds)
        stays valid for reuse.  This is the gate the prefix-cache consume path
        keys on, and the condition under which steering push/pop preserves the
        cache.  Mirrors :meth:`steering_needs_probe_gating`'s stack walk.
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
        *incoming* steering value (before any scope push) — used by
        :meth:`generate_batch` to decide whether a shared-prefix KV cache is
        reusable across the batch.  ``None`` (no steering) is trivially
        prefill-inactive; otherwise a term is prefill-active iff its trigger
        has ``prompt=True`` and no probe gate.  A malformed expression (raises
        on parse) is conservatively reported active so the caller skips
        caching and lets the normal path surface the error.
        """
        from drowse.steering.terms import Steering

        try:
            s = Steering.from_value(
                value, profile_names=set(self._session._profiles),
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
        one exists, else score exactly the planned keys from
        ``session._capture.latest_per_layer()``
        (:meth:`Monitor.score_planned_gate_scalars`).  Keys naming no attached
        probe score nothing.  Returns an empty dict when the capture is empty
        (e.g. before the first forward) so probe gates report inactive instead
        of seeing stale values from a previous gen.

        Caller-side guard: only invoked when
        :meth:`steering_needs_probe_gating` is True, so the no-gate
        path stays at zero overhead.

        Hot path: the returned closure runs per token under an active gate.  It
        binds ``capture``/``monitor`` to locals before the inner ``def`` and
        reads the session's capture state (``_capture_state`` /
        ``_incremental_readings`` / ``_incremental_gate_scores``) through one
        plain ``self._session.`` hop, exactly as the unextracted body did — no
        new per-token indirection.
        """
        session = self._session
        capture = session._capture
        monitor = session._monitor
        # J-lens token probes referenced by active gates score on the lens
        # path (readout-channel strength), not through the Monitor —
        # detected once per generation here, merged into every return below.
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
            incremental_readings = session._incremental_readings
            incremental_gate_scores = session._incremental_gate_scores
            state = session._capture_state
            gating_subset = state.gating_subset
            gate_keys = state.gating_keys or monitor_gate_keys
            if not gate_keys:
                # No gate names an attached geometry probe — readout-family
                # gates, or a gate on a probe nothing attached.  Nothing the
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
                # Once per forward: band lens logits → strength
                # scalars (also stashed step-keyed for the display step to
                # reuse — step identity replaced the freshness handshake).
                lens_scalars = session.lens.current_run.gate_scalars(
                    step_id, None, lens_gate_keys,
                )
                if lens_scalars:
                    out = {**out, **lens_scalars}
            if has_sae_gates:
                sae_scalars = session._score_sae_gate_scalars(
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
        """
        from drowse.fit.capture import fold_directions_to_subspace
        from drowse.geometry.inject import synthesize_subspace

        session = self._session
        record = deps.append if deps is not None else (lambda _dep: None)
        profiles = session._profiles
        manifolds = session._manifolds
        neutral_means = session.layer_means
        whitener = session.whitener
        record((lambda: session.layer_means, neutral_means))
        record((lambda: session.whitener, whitener))

        def _profile(name: str, role: str) -> dict[int, torch.Tensor]:
            prof = self.ensure_profile_registered(name, role=role)
            record((lambda: profiles.get(name), prof))
            return prof

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
                manifold = manifolds.get(entry.manifold)
                if manifold is None:
                    raise ManifoldNotRegisteredError(
                        f"No manifold registered for '{entry.manifold}'"
                    )
                record((lambda key=entry.manifold: manifolds.get(key), manifold))
                if manifold.is_affine:
                    # Affine ``%`` joins the merged subspace as a rank-R push
                    # toward the position's per-layer coords.
                    basis_dirs, coord_dirs = _affine_manifold_push(
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
            folded = fold_directions_to_subspace(
                name, _profile(name, "vector"), neutral_means, whitener=whitener,
            )
            if not folded.layers:
                continue
            basis_dirs, coord_dirs = _affine_manifold_push(
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
        session = self._session
        manager = session._steering
        flat = self.flatten_stack()
        if not flat:
            manager.clear_all()
            return
        plan = self.plan_for(flat)
        manager.install(plan, session._layers, session._device, session._dtype)

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
