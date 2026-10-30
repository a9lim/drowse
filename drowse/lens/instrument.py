"""The J-lens instrument: the lens family's residency, lifecycle and reads.

:class:`LensInstrument` is the one public face of the lens family
(``session.lens``).  It owns:

- the resident lens (:class:`~drowse.lens.runtime.LensRuntime`) and the
  residency transactions over it (refresh from disk, adopt, evict, source
  switch), serialized by :attr:`LensInstrument.state_lock`;
- the fit lifecycle (:class:`~drowse.lens.lifecycle.LensFitter`);
- the offline reads (:meth:`LensInstrument.readout`,
  :meth:`LensInstrument.decompose`, :meth:`LensInstrument.register_direction`,
  :meth:`LensInstrument.token_readout`);
- the readout-channel probe registry, the live workspace readout, and the
  per-generation run (:class:`LensRun`).

Dependencies are explicit constructor arguments: the model, tokenizer and
device handles, the capture buffer the per-step reads consume, the
model-exclusive guard, and a few session callbacks (registry writes, cache
invalidation, profile resolution, the capture forward).  ``replay_host`` is
the session, used only by the token replay's prompt rebuild until the shared
replay lands.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal, Protocol, cast, overload

import torch

from drowse.readout.types import (
    Axis,
    DepthSummary,
    GateRef,
    InstrumentBinding,
    InstrumentFamily,
    InstrumentPlan,
    InstrumentPrep,
    LensLiveState,
    LensPrep,
    ReadRequest,
    ScalarReading,
    UNIT_MEAN_TOKEN_PROBABILITY,
    next_prep_token,
    parse_gate_ref,
    scalar_gate_keys,
    validate_gate_channels,
)
# The readout half of the lens module is import-light by construction (the
# estimator lives in ``jlens_fit``), so the per-decode-step surfaces below bind
# these at module scope instead of re-importing inside each hot call.
from drowse.lens.jlens import (
    aggregate_readout_tensors_from_probabilities,
    pack_readout_rows_to_host,
    readout_probabilities,
    resolve_token_atom,
    token_readout_stats,
    token_readout_stats_from_probabilities,
    transport_rows,
)
from drowse.lens.layers import LayerSelector
from drowse.lens.lifecycle import LensFitRequest, LensFitter, LiveConfig
from drowse.lens.runtime import CaptureHidden, LensRuntime, select_rows

_log = logging.getLogger(__name__)

#: ``exclusive(message, phase_msg=...)``: the session's model-exclusive guard.
ExclusiveGuard = Callable[..., AbstractContextManager[Any]]


class ProfileStore(Protocol):
    """The profile-registry surface the lens writes ``jlens/`` directions
    through (the session's :class:`~drowse.session.registry.ArtifactRegistry`).
    """

    @property
    def profiles(self) -> Mapping[str, Any]: ...

    def put_profile(self, name: str, tensors: Mapping[int, torch.Tensor]) -> None: ...

    def evict_prefix(self, prefix: str) -> list[str]: ...


class LensRun:
    """Per-generation measurement executor for the lens family.

    Owns everything generation-scoped (``protocol.py``): the immutable
    :class:`InstrumentBinding` (probe specs frozen at bind), the resident-
    lens disk-identity pin, the live-active flag, the per-forward stash the
    gate callback and the display step share.
    An **idle** run (``bound=False``) backs out-of-generation reads —
    defensive captures, offline readouts — with live-registry passthrough
    semantics, so every read path has a run to consult.
    """

    def __init__(
        self,
        instrument: "LensInstrument",
        binding: InstrumentBinding,
        *,
        lens: Any = None,
        pinned: bool = False,
        active: bool = True,
        live_state: "Mapping[str, Any] | None" = None,
        bound: bool = False,
    ) -> None:
        self._instrument = instrument
        self.binding = binding
        #: The pinned resident lens (or None — unbound / validated-missing).
        self.lens = lens
        #: True when this generation snapshot-pinned the lens identity.
        self.pinned = pinned
        #: Live-readout activity for this generation.
        self.active = active
        #: The live-readout runtime dict snapshotted at prepare — an
        #: interleaved adoption rebuilds the instrument-level ``live``
        #: against the NEW lens, so a bound run must keep reading the
        #: state that matches its pin (idle runs pass through instead).
        self.live_state = live_state
        #: True for a generation-bound run (idle runs pass through).
        self.bound = bound
        self.step_stash: dict[str, Any] | None = None
        self.last_step_readings: dict[str, ScalarReading] | None = None

    # ------------------------------------------------------------ protocol

    def observe(
        self, step_id: int, hidden: dict[int, torch.Tensor],
    ) -> dict[str, ScalarReading]:
        """Readings for every attached probe from ``hidden``.  The per-step
        reuse between the gate callback and the display step is the
        step-keyed worker stash, not a memo here."""
        del step_id
        return self._instrument.score_probes(hidden)

    def gate_scalars(
        self,
        step_id: int,
        hidden: dict[int, Any] | None,
        gate_keys: frozenset[str] | set[str] | None,
    ) -> dict[str, float]:
        """The gate channels' scalars for this step.  The per-forward
        matrix-level reuse (band logits stashed for the display step)
        stays inside the instrument's worker — the stash lives on this
        run and is keyed by ``step_id``, so the display reuses rows iff
        they came from the same forward.  ``gate_keys=None`` scores the
        full roster (the session forwarder's bare-call shape)."""
        del hidden  # worker reads the capture's latest slices
        return self._instrument.gate_scalars(
            set(gate_keys) if gate_keys is not None else None,
            step_id=step_id,
        )

    def observe_aggregate(
        self, pooled: dict[int, Any],
    ) -> dict[str, ScalarReading]:
        """End-of-generation aggregate at the pooled last-content slice."""
        return self._instrument.score_probes(pooled)

    def close(self) -> None:
        """Release generation-scoped state (stash, pin)."""
        self.step_stash = None
        self.last_step_readings = None
        self.lens = None
        self.pinned = False
        self.active = True
        self.live_state = None


class LensInstrument:
    """Session-lifetime handle for the J-lens read family."""

    family: InstrumentFamily = "lens"

    #: Gate channels a lens probe can produce: the one strength axis.
    _GATE_CHANNELS: tuple[type, ...] = (Axis,)

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        model_id: str,
        device: torch.device,
        n_layers: int,
        capture: Any,
        exclusive: ExclusiveGuard,
        capture_hidden: CaptureHidden,
        pooled_slice: Callable[[list[int]], dict[int, torch.Tensor]],
        registry: ProfileStore,
        on_change: Callable[[Iterable[str]], None],
        resolve_profile: Callable[[str], Mapping[int, torch.Tensor]],
        assert_unsteered: Callable[[], None],
        replay_host: Any = None,
    ) -> None:
        self.model_id = model_id
        self.runtime = LensRuntime(
            model=model, tokenizer=tokenizer, model_id=model_id,
            device=device, n_layers=n_layers,
        )
        #: The capture buffer the per-step reads consume (latest slices).
        self._capture = capture
        self._exclusive = exclusive
        self._capture_hidden = capture_hidden
        self._pooled_slice = pooled_slice
        #: The session's profile registry: ``jlens/<word>`` directions land
        #: here and are evicted on every lens change.
        self._registry = registry
        #: Called with the lens probe names whenever registered directions or
        #: the resident lens change (prefix/analytics/probe-hash caches).
        self._on_change = on_change
        self._resolve_profile = resolve_profile
        self._assert_unsteered = assert_unsteered
        self._replay_host = replay_host
        self._fitter = LensFitter(
            model=model, tokenizer=tokenizer, model_id=model_id, residency=self,
        )
        # Pinned J-lens token probes: name -> {word, token_id, layers}.  NOT
        # monitor probes — they read the lens readout channel (per-layer
        # softmax probability), not a whitened subspace coordinate.
        self.probes: dict[str, dict[str, Any]] = {}
        # Live workspace readout (enable_live): device-resident J_l subset +
        # settings, or None when off.  Runtime residency, not user intent —
        # disabling drops this dict but transported stacks stay in the
        # runtime's device cache.
        self.live: dict[str, Any] | None = None
        # THE lens-state boundary: one reentrant lock covering source
        # refresh/adoption/eviction, registry and live-state mutation,
        # and ``prepare``'s complete snapshot, so the snapshot cannot
        # tear mid-``prepare`` under a concurrent un-locked getter
        # read.  A leaf lock: nothing acquires the model-exclusive guard
        # while holding it (callers hold that first).  Reentrant because
        # adoption runs inside the refresh and ``prepare`` refreshes inside
        # its own hold.
        self.state_lock = threading.RLock()
        # The current per-generation run (idle passthrough until bind()).
        # All generation-scoped state — stash, display readings, active
        # flag, disk-identity pin — lives on it.
        self.current_run = LensRun(
            self, InstrumentBinding(family=self.family),
        )

    # ------------------------------------------------------------- run state
    # Generation-scoped state lives on the current run; these name it.

    @property
    def step_stash(self) -> dict[str, Any] | None:
        return self.current_run.step_stash

    @step_stash.setter
    def step_stash(self, value: "dict[str, Any] | None") -> None:
        self.current_run.step_stash = value

    @property
    def last_step_readings(self) -> "dict[str, ScalarReading] | None":
        return self.current_run.last_step_readings

    @last_step_readings.setter
    def last_step_readings(
        self, value: "dict[str, ScalarReading] | None",
    ) -> None:
        self.current_run.last_step_readings = value

    @property
    def active_for_generation(self) -> bool:
        return self.current_run.active

    @active_for_generation.setter
    def active_for_generation(self, value: bool) -> None:
        self.current_run.active = bool(value)

    @property
    def generation_lens(self) -> Any:
        return self.current_run.lens

    @generation_lens.setter
    def generation_lens(self, value: Any) -> None:
        self.current_run.lens = value

    @property
    def generation_lens_active(self) -> bool:
        return self.current_run.pinned

    @generation_lens_active.setter
    def generation_lens_active(self, value: bool) -> None:
        self.current_run.pinned = bool(value)

    # ------------------------------------------------------------ run lifecycle

    def prepare(self, request: ReadRequest) -> LensPrep:
        """The refresh/pin protocol step.

        Refreshes the resident lens from disk (:meth:`resident`) under pin
        demand BEFORE any plan is taken: the adoption path rewrites the
        live probe layer lists when an external replacement lens landed,
        so the capture plan AND the bind-time spec freeze must both see
        the refreshed registry (a plan taken earlier pairs the new lens
        with stale layers and KeyErrors in the transport stack).

        The prep is the **authoritative snapshot**:
        spec ``layers`` are derived from the prepared lens identity
        itself — never left to a later registry reread — and the live
        runtime dict is captured by reference, because an interleaved
        unpinned refresh (:meth:`has_compatible` runs un-locked on the
        session-info route) can adopt a *newer* disk lens inside the
        prepare→bind window and rewrite both.  ``plan``/``bind`` consume
        only this snapshot, so the run's lens, plan layers, and frozen
        specs agree by construction.

        Pin demand is the registry boolean, not plan emptiness — probes
        whose lens vanished still pin the validated-missing state so
        per-token paths never reopen the sidecar.  The refresh
        short-circuits on a bound run's pin flag, which is why prepare
        requires the prior run closed (a stale pin would suppress the
        very refresh this step exists for).
        """
        if self.current_run.bound:
            raise RuntimeError(
                "LensInstrument.prepare() on a bound run: close the prior "
                "generation's run (_close_instrument_runs) before preparing "
                "the next — a stale pin suppresses the lens disk refresh"
            )
        # The WHOLE snapshot is one atomic lens-state transaction:
        # items, gate demand, pin demand, the getter refresh, spec
        # derivation, the live-state reference, and the sidecar
        # fingerprint are all read under ``state_lock`` — the same lock
        # the refresh/adoption/eviction, the registry mutations, and the
        # live toggles hold.  A concurrent
        # ``has_compatible`` adoption landing between the refresh
        # and the live-state read would pair lens A's specs with lens
        # B's live device stack; a concurrent detach would split gate
        # demand from the captured roster.
        with self.state_lock:
            items = list(self.probes.items())
            names = {name for name, _spec in items}
            gate_hit = any(
                parse_gate_ref(key).probe in names
                for key in request.gate_keys
            )
            pin_demand = bool(
                (request.live and self.live is not None)
                or (items and (request.final_aggregate or gate_hit))
            )
            lens = self.resident() if pin_demand else None
            # Spec layers under a pin come from the prepared identity
            # itself — the atomically consistent source for every probe
            # (the production invariant keeps registry lists identical to
            # ``source_layers``: attach records the full fitted set and
            # adoption rewrites every probe).  A pin-demanded lens
            # without them is structurally broken — layers are what align
            # captures, specs, and Jacobians — so it fails here, at the
            # boundary, not as a mid-generation KeyError.  A vanished
            # lens (None) pins the validated-missing state: its eviction
            # path already zeroed the registry lists the copies reflect.
            if pin_demand and lens is not None and not hasattr(
                lens, "source_layers",
            ):
                raise RuntimeError(
                    "LensInstrument.prepare(): the resident lens has no "
                    "source_layers — a pin-demanded lens must carry its "
                    "fitted layer set"
                )
            if pin_demand and lens is not None:
                layers = [int(layer) for layer in lens.source_layers]
                specs = {
                    name: {**spec, "layers": list(layers)}
                    for name, spec in items
                }
            else:
                specs = {
                    name: {**spec, "layers": list(spec["layers"])}
                    for name, spec in items
                }
            identity = self.runtime.identity
            return LensPrep(
                family=self.family,
                request=request,
                token=next_prep_token(),
                lens=lens,
                pinned=pin_demand,
                specs=specs,
                live_state=self.live,
                fingerprint=identity,
            )

    def bind(self, plan: InstrumentPlan, prep: InstrumentPrep) -> LensRun:
        """Bind an immutable per-generation run from a declared plan.

        The **prep supplies the pin and the spec snapshot** (``prepare``
        — see its docstring for the ordering + snapshot rationale).
        ``bind`` freezes the prep's specs (never the live registry, which
        an interleaved adoption may have rewritten since prepare) and
        installs the pin and live-state reference it is handed.  The
        frozen spec copies own their ``layers`` list — the live
        registry's lists are replaced in place by the adoption/eviction
        paths.
        """
        if not isinstance(prep, LensPrep) or prep.family != self.family:
            raise TypeError(
                "LensInstrument.bind takes the LensPrep its own prepare() "
                f"returned, got {type(prep).__name__} "
                f"(family={getattr(prep, 'family', None)!r})"
            )
        if plan.family != self.family:
            raise ValueError(
                f"LensInstrument.bind: plan family {plan.family!r} is not "
                f"{self.family!r}"
            )
        if plan.prep_token != prep.token:
            raise ValueError(
                "LensInstrument.bind: the plan was not derived from this "
                "prep (prep_token mismatch) — derive the plan from the same "
                "prepare() call"
            )
        run = LensRun(
            self,
            InstrumentBinding(
                family=self.family,
                # ``source`` stays None: the public source label lives in
                # ``active.json`` and reading it here would put a disk hit
                # on every generation; the prep's sidecar identity — taken
                # in the same atomic snapshot as the pin — is the binding's
                # cheap fingerprint (a bind-time live read could stamp a
                # concurrently adopted replacement's identity onto a run
                # pinned to the older lens).
                fingerprint=prep.fingerprint,
                specs={
                    name: {**spec, "layers": list(spec["layers"])}
                    for name, spec in prep.specs.items()
                },
            ),
            lens=prep.lens,
            pinned=prep.pinned,
            active=prep.request.live,
            live_state=prep.live_state,
            bound=True,
        )
        self.current_run = run
        return run

    def close_run(self) -> None:
        """Close the current run and restore the idle passthrough run."""
        self.current_run.close()
        self.current_run = LensRun(
            self, InstrumentBinding(family=self.family),
        )

    def _measurement_specs(self) -> "dict[str, Any] | Any":
        """The spec source for measurement: a bound run's frozen binding
        (immune to concurrent spec mutation — e.g. the refresh's eviction
        path zeroing probe layers from another thread), else the live
        registry."""
        run = self.current_run
        return run.binding.specs if run.bound else self.probes

    def _measurement_live(self) -> "Mapping[str, Any] | None":
        """The live-readout state for measurement: a bound run's
        prepare-time snapshot (an interleaved adoption rebuilds the
        instrument-level ``live`` against the NEW lens — the run must
        keep reading the state that matches its pin), else the live
        config."""
        run = self.current_run
        return run.live_state if run.bound else self.live

    def _measurement_state(self) -> "tuple[Any, dict[str, Any] | Any]":
        """The coherent ``(lens, specs)`` pair for measurement.

        Bound: the run's pin + frozen binding — lock-free, immutable by
        construction (an unpinned bound run falls back to the refresh,
        which locks internally; nothing measures on such a run anyway).
        Idle: ONE state-lock hold refreshes the resident lens and copies
        the registry, so an idle-passthrough read (``observe`` on the
        idle run, offline scoring) cannot pair lens A with a
        concurrently adopted replacement's rewritten layers.  Resolving
        the two separately — the registry un-locked, the resident lens
        under its own hold — lets scoring run with A's lens and B's
        layers, which KeyErrors in the transport stack.
        """
        run = self.current_run
        if run.bound:
            lens = run.lens if run.pinned else self.resident()
            return lens, run.binding.specs
        with self.state_lock:
            lens = self.resident()
            specs = {
                name: {**spec, "layers": list(spec["layers"])}
                for name, spec in self.probes.items()
            }
            return lens, specs

    # ------------------------------------------------------------- residency

    @property
    def resident_lens(self) -> Any:
        """The resident lens as last refreshed (no disk read)."""
        return self.runtime.lens

    @property
    def resident_identity(self) -> str | None:
        """The resident lens's artifact identity (no disk read)."""
        return self.runtime.identity

    def resident(self) -> Any:
        """The model's lens from the active source, or ``None`` when absent.

        Refreshes against disk: a removed, replaced or mismatched artifact
        is evicted along with every derived view, and a new one is loaded
        and adopted.  A generation-bound run's pinned lens short-circuits the
        refresh, so a generation reads one lens throughout.  Fit or fetch a
        lens with ``drowse lens fit|fetch``.
        """
        with self.state_lock:
            run = self.current_run
            if run.pinned:
                return run.lens
            self._refresh(load=True)
            return self.runtime.lens

    def has_compatible(self) -> bool:
        """Whether the active source matches the loaded weights.

        Runs un-locked on the session-info route; a lens that is not
        resident yet is only checked, never paged in.
        """
        with self.state_lock:
            run = self.current_run
            if run.pinned:
                return run.lens is not None
            return self._refresh(load=False)

    def _refresh(self, *, load: bool) -> bool:
        """Reconcile the resident lens with the active source's artifact.

        Returns whether the active source is compatible.  ``load`` pages in
        a compatible lens that is not resident yet.
        """
        from drowse.lens.store import lens_identity, load_lens, load_lens_sidecar

        runtime = self.runtime
        sidecar = load_lens_sidecar(self.model_id)
        if not runtime.matches(sidecar):
            if runtime.lens is not None:
                self.evict()
            if load and sidecar is not None:
                _log.warning(
                    "ignoring stale Jacobian lens for %s: loaded-model "
                    "identity is missing or changed; fetch or re-fit the lens",
                    self.model_id,
                )
            return False
        if runtime.lens is not None and runtime.identity == lens_identity(sidecar):
            return True
        if not load and runtime.lens is None:
            return True
        loaded = load_lens(self.model_id)
        if loaded is None:
            self.evict()
            return False
        if not runtime.matches(loaded[1]):
            self.evict()
            if load:
                _log.warning(
                    "ignoring concurrently replaced Jacobian lens for %s: "
                    "loaded payload does not match the live model",
                    self.model_id,
                )
            return False
        self.adopt(loaded[0], loaded[1])
        return True

    def _require(self) -> Any:
        from drowse.lens.jlens import LensNotFittedError

        lens = self.resident()
        if lens is None:
            raise LensNotFittedError(
                f"no Jacobian lens available for {self.model_id} — run "
                f"`drowse lens fetch {self.model_id}` for a supported official "
                f"lens, or `drowse lens fit {self.model_id}`"
            )
        return lens

    def live_config(self) -> LiveConfig | None:
        """The live readout's ``(layers, uses_all_layers)``, or ``None``."""
        with self.state_lock:
            live = self.live
            if live is None:
                return None
            return list(live["layers"]), bool(live.get("uses_all_layers", False))

    def _drop_directions(self) -> None:
        self._registry.evict_prefix("jlens/")

    def evict(self) -> None:
        """Drop the resident lens and every derived view: registered
        ``jlens/`` directions, probe layer lists, live state, the stash."""
        with self.state_lock:
            self.runtime.clear()
            self.step_stash = None
            self.last_step_readings = None
            self._drop_directions()
            for spec in self.probes.values():
                spec["layers"] = []
            self.live = None
            self._on_change(list(self.probes))

    def adopt(
        self,
        lens: Any,
        sidecar: Mapping[str, Any] | None,
        *,
        live: LiveConfig | None = None,
        restore_live: bool = False,
    ) -> Any:
        """Make ``lens`` resident and rebuild every derived consumer.

        One lens-state transaction, so a concurrent ``prepare`` snapshot
        sees the old world or the new one, never a mix.  The live readout is
        rebuilt from its current configuration, or from ``live`` when
        ``restore_live`` (a fit restores the configuration it evicted).
        """
        with self.state_lock:
            previous = live if restore_live else self.live_config()
            self.runtime.install(lens, sidecar)
            self.step_stash = None
            self.last_step_readings = None
            self._drop_directions()
            readout_layers = [int(layer) for layer in lens.source_layers]
            for spec in self.probes.values():
                spec["layers"] = list(readout_layers)
            if self.probes and readout_layers:
                self.runtime.transport_stack(lens, sorted(readout_layers))
            if previous is not None:
                layers, uses_all_layers = previous
                valid = [layer for layer in layers if layer in lens.jacobians]
                self.enable_live(
                    layers=None if uses_all_layers else (valid or None),
                )
            else:
                self.live = None
            self._on_change(list(self.probes))
            return lens

    def select_source(self, source: str) -> None:
        """Select a prepared local/external lens source and evict the
        resident lens.  Prepare the source first (fit or fetch)."""
        from drowse.lens.sources import use_lens_source

        use_lens_source(self.model_id, source)
        self.evict()

    def close(self) -> None:
        """Session teardown: close the run and drop every resident view."""
        self.close_run()
        with self.state_lock:
            self.probes.clear()
            self.live = None
        self.runtime.close()

    # ------------------------------------------------------------- lifecycle

    def fit(
        self,
        prompts: Sequence[str],
        *,
        kind: str = "J",
        corpus_spec: str = "custom",
        source_layers: LayerSelector = None,
        dim_batch: int | None = None,
        prompt_batch: int | None = None,
        seq_len: int | None = None,
        force: bool = False,
        checkpoint_every: int | None = None,
        on_progress: Callable[[str], None] | None = None,
        cancel_event: Any | None = None,
    ) -> Any:
        """Fit (or resume) this model's ``kind`` lens and make it resident.

        ``kind="J"`` is the standard J-lens at ``local:j``; ``kind="R"`` the
        RelP R-lens at ``local:r`` (the same estimator through the
        LRP-modified backward graph).  Resume is the default: a matching
        durable artifact or checkpoint over a prefix of this corpus is
        extended; ``force`` restarts from zero.  ``source_layers`` is
        ``None``/``"all"`` (every layer below the final one),
        ``"workspace"``, or an index list.  Runs the only backward passes in
        drowse, under the fit lock and the model-exclusive guard.  Returns
        the fitted lens restricted to the requested layers.
        """
        from drowse.lens.jlens import (
            DEFAULT_CHECKPOINT_EVERY,
            DEFAULT_DIM_BATCH,
            DEFAULT_SEQ_LEN,
            JacobianLensCancelled,
        )
        from drowse.lens.store import cleanup_lens_artifacts, lens_fit_lock

        request = LensFitRequest(
            kind=kind,
            corpus_spec=corpus_spec,
            source_layers=source_layers,
            dim_batch=dim_batch or DEFAULT_DIM_BATCH,
            prompt_batch=prompt_batch,
            seq_len=seq_len or DEFAULT_SEQ_LEN,
            force=force,
            checkpoint_every=checkpoint_every or DEFAULT_CHECKPOINT_EVERY,
        )
        name = request.kind.lower()
        with lens_fit_lock(self.model_id, name):
            cleanup_lens_artifacts(self.model_id, name)
            if cancel_event is not None and cancel_event.is_set():
                raise JacobianLensCancelled("Jacobian-lens fit cancelled before start")
            with self._exclusive(
                "session.lens.fit called while another model use is in flight",
                phase_msg="session.lens.fit called while a generation is in flight",
            ):
                self._assert_unsteered()
                return self._fitter.run(
                    prompts, request,
                    on_progress=on_progress, cancel_event=cancel_event,
                )

    # ---------------------------------------------------------- offline reads

    @overload
    def readout(
        self,
        prompt: str,
        *,
        layers: LayerSelector = None,
        positions: Sequence[int] | None = None,
        top_k: int = 10,
        aggregate: Literal[False] = False,
    ) -> dict[int, list[list[tuple[str, float]]]]: ...

    @overload
    def readout(
        self,
        prompt: str,
        *,
        layers: LayerSelector = None,
        positions: Sequence[int] | None = None,
        top_k: int = 10,
        aggregate: Literal[True],
    ) -> tuple[
        dict[int, list[list[tuple[str, float]]]],
        list[list[tuple[str, float, float, float]]],
    ]: ...

    def readout(
        self,
        prompt: str,
        *,
        layers: LayerSelector = None,
        positions: Sequence[int] | None = None,
        top_k: int = 10,
        aggregate: bool = False,
    ) -> (
        dict[int, list[list[tuple[str, float]]]]
        | tuple[
            dict[int, list[list[tuple[str, float]]]],
            list[list[tuple[str, float, float, float]]],
        ]
    ):
        """Lens readout on a raw prompt: the top-``top_k`` tokens per
        (layer, position) with their per-layer probabilities ``p_l``.

        ``layers`` defaults to every fitted layer; ``positions`` to the
        final position (explicit indices, negative from the end, read
        more).  Returns ``{layer: [per-position [(token, p_l), ...]]}``.
        ``aggregate=True`` also returns one layer-aggregated chip list
        ``[(token, strength, com, spread), ...]`` per position, from the
        same logits (:func:`drowse.lens.jlens.aggregate_readout`).
        """
        lens = self._require()
        self.runtime.resolve_layers(lens, layers)
        with self._exclusive(
            "session.lens.readout called while another model use is in flight",
            phase_msg="session.lens.readout called while a generation is in flight",
        ):
            return self.runtime.readout(
                lens, prompt,
                capture_hidden=self._capture_hidden,
                layers=layers, positions=positions, top_k=top_k,
                aggregate=aggregate,
            )

    def decompose(
        self,
        selector: str,
        *,
        k: int = 16,
        layers: LayerSelector = None,
    ) -> dict[int, tuple[float, list[tuple[str, float]]]]:
        """Split a steerable direction into its J-space component, per layer.

        ``selector`` resolves through the ordinary steering resolver, so any
        steerable vector decomposes.  Returns ``{layer: (share, [(token,
        coeff), ...])}``: ``share`` is the fraction of the direction's
        variance carried by its best sparse nonnegative combination of ``k``
        lens vectors (how *verbalizable* it is; ~6-15% is typical for
        concept vectors).
        """
        lens = self._require()
        directions = self._resolve_profile(selector)
        return self.runtime.decompose(
            lens, directions, label=selector, k=k, layers=layers,
        )

    def register_direction(self, word: str) -> str:
        """Register ``W_U[v] @ J_l`` for a single-token word (or ``#<id>``)
        as the profile ``jlens/<word>`` and return its name.

        The direction steers and ablates exactly like an extracted vector;
        this is the resolver behind the lazy ``jlens/`` steering atom.  A
        ``jlens/<word>`` *probe* reads the readout channel instead.
        Idempotent; every fitted layer.
        """
        name = f"jlens/{word}"
        if name in self._registry.profiles:
            return name
        lens = self._require()
        self._registry.put_profile(name, self.runtime.token_directions(lens, word))
        self._on_change(())
        return name

    # -------------------------------------------------------------- registry

    def attach(self, selector: str, *, as_name: str | None = None) -> str:
        """Attach a ``jlens/<word>`` (or ``jlens/#<id>``) token probe.

        Validates the lens artifact + token (the same ``resolve_token_atom``
        contract steering atoms use), records the
        full fitted layer set, and pre-warms the device transport stack so
        the first decode step doesn't hitch on the J_l transfer.  The probe
        reads ONE channel — ``coords = (strength,)``, the mean layer
        probability ``mean_l p_l(v)``.  Cache invalidation (prefix cache,
        probe-hash cache, analytics) is the session's job at its
        ``add_probe`` boundary.
        """
        word = selector.split("/", 1)[1]
        if not word:
            raise ValueError("empty jlens probe word")
        name = as_name if as_name is not None else selector
        runtime = self.runtime
        with self._exclusive(
            "add_probe called while another model operation is in "
            "flight; retry shortly"
        ):
            # Lens-state boundary inside the exclusive section (consistent
            # outer→leaf lock order): the lens read and the registry write
            # land in one lens-state transaction.
            with self.state_lock:
                lens = self._require()
                token_id = self.resolve_token(word)
                readout_layers = [int(l) for l in lens.source_layers]
                runtime.transport_stack(lens, sorted(readout_layers))
                self.probes[name] = {
                    "word": word,
                    "token_id": int(token_id),
                    "layers": readout_layers,
                }
        return name

    def resolve_token(self, word: str) -> int:
        """The vocab id a ``jlens/<word>`` atom names: a single-token word,
        or ``#<id>`` addressing the id directly (the steering-atom contract)."""
        runtime = self.runtime
        return resolve_token_atom(
            runtime.tokenizer, word,
            vocab_size=int(runtime.readout_modules()[0].shape[0]),
        )

    def try_detach(self, name: str) -> bool:
        """Atomic membership-check + detach under the lens-state lock — the
        family's only removal surface.

        The session's ``remove_probe`` dispatch routes lens removals here: a
        bare ``name in probes`` check followed by a delete is two
        un-serialized registry touches, so membership and removal have to
        land inside one hold.  Returns False when the name isn't a lens
        probe.
        """
        with self.state_lock:
            if name not in self.probes:
                return False
            del self.probes[name]
            return True

    def specs(self) -> dict[str, dict[str, Any]]:
        """Snapshot of attached probe specifications (one coherent
        lens-state read — a concurrent detach or adoption cannot tear
        the iteration or mix old/new layer lists)."""
        with self.state_lock:
            return {name: dict(spec) for name, spec in self.probes.items()}

    @property
    def names(self) -> list[str]:
        with self.state_lock:
            return list(self.probes)

    def probe_layers(self, names: set[str] | None = None) -> set[int]:
        """Union of the attached probes' fitted layers (coherent read)."""
        with self.state_lock:
            out: set[int] = set()
            probes = self.probes
            if names is not None:
                probes = {
                    name: probes[name] for name in names if name in probes
                }
            for spec in probes.values():
                out.update(spec["layers"])
            return out

    def validate_gate(self, ref: GateRef) -> None:
        validate_gate_channels(ref, self._GATE_CHANNELS, family=self.family)

    # ---------------------------------------------------------------- planning

    def plan(self, prep: InstrumentPrep) -> InstrumentPlan:
        """Declare the lens family's capture demand for one generation.

        Derived **solely from the prep** — the spec snapshot whose layers
        match the prepared lens identity, and the live state captured at
        prepare — never the live registry, which an interleaved adoption
        may have rewritten since: a plan read off the rewritten registry
        pairs the prep's older lens with the new lens's layers and
        KeyErrors in the transport stack.

        ``latest_layers`` — the live workspace readout's layer set plus the
        pinned probes' fitted band (full band when a finalize aggregate
        will pool it; the gated probes' band when gates are the only
        per-step consumer with final readings disabled).  ``tail_layers``
        is the finalize-pooling demand: pinned probes' aggregates pool the
        last content token from the capture tail ring, which must span the
        probe band at ring depth ``AGG_TAIL_DEPTH``.
        """
        if not isinstance(prep, LensPrep) or prep.family != self.family:
            raise TypeError(
                "LensInstrument.plan takes the LensPrep its own prepare() "
                f"returned, got {type(prep).__name__}"
            )
        request = prep.request
        probes: Mapping[str, Mapping[str, Any]] = prep.specs
        live = prep.live_state if request.live else None
        gate_keys = frozenset(
            key for key in request.gate_keys
            if parse_gate_ref(key).probe in probes
        )

        def _band(names: "set[str] | None" = None) -> set[int]:
            return {
                int(layer)
                for name, spec in probes.items()
                if names is None or name in names
                for layer in spec["layers"]
            }

        latest: set[int] = set()
        tail: set[int] = set()
        if live is not None:
            latest.update(int(layer) for layer in live["layers"])
        if probes and request.final_aggregate:
            band = _band()
            latest.update(band)
            tail.update(band)
        elif gate_keys:
            # Gate-only pinned probes need per-step latest slices, but
            # dormant probes must not keep capture alive when the caller
            # disabled final probe readings.
            gated_names = {parse_gate_ref(key).probe for key in gate_keys}
            latest.update(_band(gated_names))
        return InstrumentPlan(
            family=self.family,
            latest_layers=frozenset(latest),
            tail_layers=frozenset(tail),
            gate_keys=gate_keys,
            final_aggregate=bool(probes and request.final_aggregate),
            prep_token=prep.token,
        )

    def probe_hash(self, name: str) -> str | None:
        """Readout-channel identity digest (no baked tensor exists).

        The channel is the resident lens read through the model's readout
        (output bias, projection, scaling, final soft-capping), so the digest
        folds in the lens artifact's identity: the same word read through a
        different lens — another source, a re-fit — is a different probe.
        """
        with self.state_lock:
            spec = self.probes.get(name)
            if spec is None:
                return None
            return hashlib.sha256(
                repr(
                    (
                        "jlens-readout-v5", self.model_id,
                        spec["word"], spec["token_id"],
                        tuple(spec["layers"]),
                        self.runtime.identity,
                    )
                ).encode("utf-8")
            ).hexdigest()

    # ---------------------------------------------------------------- scoring

    def score_probes(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        only: "set[str] | None" = None,
    ) -> dict[str, ScalarReading]:
        """Score every attached probe from capture hidden slices.

        The capture-slice entry: computes the lens logits itself over the
        probes' fitted layers that ``hidden`` covers.  Callers that already
        hold a calibrated row matrix use :meth:`score_probes_from_rows`.
        Empty when no probe layer is available.
        """
        lens, probes = self._measurement_state()
        if not probes or lens is None:
            return {}
        readout_layers: set[int] = set()
        for spec in probes.values():
            readout_layers.update(spec["layers"])
        layers = sorted(l for l in readout_layers if l in hidden)
        if not layers:
            return {}
        logits = self.runtime.logits_rows(
            lens, [(l, hidden[l]) for l in layers],
        )
        return self._readings_from_rows(
            probes, layers, logits=logits, only=only,
        )

    def score_probes_from_rows(
        self,
        *,
        layers: "Sequence[int]",
        logits: torch.Tensor | None = None,
        probabilities: torch.Tensor | None = None,
        only: "set[str] | None" = None,
    ) -> dict[str, ScalarReading]:
        """Score attached probes from precomputed lens rows aligned to
        ``layers`` — exactly one of ``logits`` / ``probabilities``.

        The entry for callers that already calibrated this forward's matrix
        (the gate callback, the live display step, authored prefill): rows are
        restricted to the probes' fitted layer set, never recomputed.
        """
        if (logits is None) == (probabilities is None):
            raise ValueError(
                "score_probes_from_rows takes lens logits OR probabilities"
            )
        _lens, probes = self._measurement_state()
        if not probes:
            return {}
        readout_layers: set[int] = set()
        for spec in probes.values():
            readout_layers.update(spec["layers"])
        # Restrict precomputed rows (e.g. a custom live-lens layer set) to the
        # probes' fitted layer set.
        keep = [i for i, l in enumerate(layers) if l in readout_layers]
        if not keep:
            return {}
        if len(keep) != len(layers):
            if logits is not None:
                logits = logits[keep]
            if probabilities is not None:
                probabilities = probabilities[keep]
        return self._readings_from_rows(
            probes,
            [layers[i] for i in keep],
            logits=logits,
            probabilities=probabilities,
            only=only,
        )

    def _readings_from_rows(
        self,
        probes: "Mapping[str, Mapping[str, Any]]",
        layers: "Sequence[int]",
        *,
        logits: torch.Tensor | None = None,
        probabilities: torch.Tensor | None = None,
        only: "set[str] | None" = None,
    ) -> dict[str, ScalarReading]:
        """Synthesize the readout-channel readings for ``layers``' rows.

        Returns the family-native :class:`ScalarReading`: ONE channel, the
        mean fitted-layer probability ``mean_l p_l(v)``, its explicit
        ``unit``, the per-layer ``p_l`` trace, and a depth summary that
        names its own mass basis.  Not a ``ProbeReading`` with eight
        neutralized geometry fields — a readout probe has no fraction,
        nearest, residual, assignment or membership to report, and shipping
        those as constants is measurement-shaped noise.
        """
        runtime = self.runtime
        names = [
            name for name in probes
            if only is None or name in only
        ]
        if not names:
            return {}
        token_ids = [probes[n]["token_id"] for n in names]
        rows = probabilities if probabilities is not None else logits
        row_device = cast(torch.Tensor, rows).device
        token_ids_tensor = runtime.long_tensor(token_ids, row_device)
        depth_tensor = runtime.depth_tensor(layers, row_device)
        depths = runtime.depths(layers)
        if probabilities is not None:
            stats = token_readout_stats_from_probabilities(
                probabilities, depths, token_ids,
                token_ids_tensor=token_ids_tensor,
                depth_tensor=depth_tensor,
            )
        else:
            stats = token_readout_stats(
                cast(torch.Tensor, logits), depths, token_ids,
                token_ids_tensor=token_ids_tensor,
                depth_tensor=depth_tensor,
            )
        out: dict[str, ScalarReading] = {}
        for name, (strength, com, spread, per_layer) in zip(names, stats):
            out[name] = ScalarReading(
                value=float(strength),
                unit=UNIT_MEAN_TOKEN_PROBABILITY,
                per_layer={
                    int(l): float(p_l) for l, p_l in zip(layers, per_layer)
                },
                depth=DepthSummary(
                    center=(float(com),),
                    spread=(float(spread),),
                    basis="readout_probability_mass",
                ),
            )
        return out

    def gate_scalars(
        self, gate_keys: "set[str] | None" = None, *, step_id: int = -1,
    ) -> dict[str, float]:
        """Per-forward gate scalars from the latest capture slices.

        Called from the gating score callback (once per decode forward,
        before the token tap). Computes the referenced lens logits, stashes
        them for the display step to reuse (``step_stash``, keyed by
        ``step_id`` — the display reuses rows iff ``stash["step"]`` matches
        its own forward index, so staleness is structural and reuse is
        idempotent), and flattens
        the synthesized readings through :meth:`Monitor.flat_scalars` so the
        gate key space is uniform. Gate-only calls score exact
        selected-token softmax columns; live display calls still calibrate
        the full matrix once for downstream card/aggregate reuse.  Empty
        when nothing is capturable yet.
        """
        lens, probes = self._measurement_state()
        if not probes or lens is None:
            return {}
        latest = self._capture.latest_per_layer()
        if not latest:
            return {}
        only = None
        if gate_keys is not None:
            # ``None`` is the full-roster sentinel; an explicit empty set
            # means "no gated probes" and scores nothing — the
            # None-vs-empty distinction every family's gate entry keeps.
            only = {
                key.split("[", 1)[0]
                for key in gate_keys
                if key.split("[", 1)[0] in probes
            }
            if not only:
                return {}
        live_display_needs_full_probs = bool(
            self._measurement_live() is not None
            and self.active_for_generation
        )
        # When the token tap will immediately need every pinned probe reading
        # for the live payload, compute that superset once in the gate
        # callback and let the display reuse it.  Gate-only calls stay on the
        # narrower requested subset.
        probe_read_only = None if live_display_needs_full_probs else only
        readout_layers: set[int] = set()
        for name, spec in probes.items():
            if probe_read_only is None or name in probe_read_only:
                readout_layers.update(spec["layers"])
        layers = sorted(l for l in readout_layers if l in latest)
        if not layers:
            return {}

        logits = self.runtime.logits_rows(
            lens, [(l, latest[l]) for l in layers],
        )
        probabilities = None
        if live_display_needs_full_probs:
            probabilities = readout_probabilities(logits)
            live_stash: dict[str, Any] = {
                "layers": tuple(layers),
                "logits": logits,
                "probabilities": probabilities,
                "step": step_id,
            }
            self.step_stash = live_stash
        else:
            self.step_stash = {
                "layers": tuple(layers),
                "logits": logits,
                "step": step_id,
            }
        if probabilities is not None:
            readings = self.score_probes_from_rows(
                layers=layers,
                probabilities=probabilities,
                only=probe_read_only,
            )
            live_stash = cast("dict[str, Any]", self.step_stash)
            live_stash["readings"] = readings
            live_stash["readings_layers"] = tuple(layers)
            live_stash["readings_step"] = step_id
            self.last_step_readings = readings
        else:
            readings = self.score_probes_from_rows(
                layers=layers, logits=logits, only=probe_read_only,
            )
        return scalar_gate_keys({
            name: reading.value
            for name, reading in readings.items()
            if only is None or name in only
        })

    def score_aggregate(
        self,
        generated_ids: list[int],
        *,
        pooled: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, ScalarReading]:
        """End-of-gen aggregate pooled at the last content token.

        Shares the session's pooled slice with the monitor roster and the
        SAE family, so all three aggregates read the same position under
        every retention mode.
        """
        # Binding-authoritative guard: a probe detached mid-generation
        # stays in this generation's aggregate roster (mutations apply
        # next generation).
        if not self._measurement_specs() or not generated_ids:
            return {}
        if pooled is None:
            pooled = self._pooled_slice(generated_ids)
        if not pooled:
            return {}
        return self.current_run.observe_aggregate(pooled)

    # ----------------------------------------------------------- live readout

    def enable_live(
        self,
        *,
        layers: "Sequence[int] | None" = None,
    ) -> list[int]:
        """Stream the J-lens readout live during generation.

        The selected layers' ``J_l`` move device-resident here, once;
        ``layers`` defaults to every fitted lens layer.  Attaches no
        forward hooks (the reader consumes the capture's existing
        latest-slice buffers post-forward), so steering fast-path /
        compile eligibility is untouched.  Returns the resolved layer list.
        """
        runtime = self.runtime

        # One lens-state transaction: the lens read and the live-state
        # rebuild must be consistent (adoption calls this while holding
        # the same reentrant lock).
        with self.state_lock:
            lens = self._require()
            uses_all_layers = layers is None
            if layers is None:
                layers = sorted(int(layer) for layer in lens.source_layers)
            else:
                layers = sorted(set(int(l) for l in layers))
                missing = [l for l in layers if l not in lens.jacobians]
                if missing:
                    raise ValueError(
                        f"layers {missing} not in the fitted lens "
                        f"(fitted: {lens.source_layers[0]}.."
                        f"{lens.source_layers[-1]})"
                    )
            device = runtime.device
            layer_list = list(layers)
            if layer_list:
                j_stack = runtime.transport_stack(lens, layer_list, device)
            else:
                sample = next(iter(lens.jacobians.values()))
                j_stack = torch.empty(
                    (0, *sample.shape), device=device, dtype=torch.float32,
                )
            self.live = {
                "layers": layer_list,
                "uses_all_layers": uses_all_layers,
                "J_stack": j_stack,
                "layer_rows": {l: i for i, l in enumerate(layer_list)},
                "readout": runtime.readout_modules()[1],
                "source": self.active_source,
            }
            return list(layers)

    def disable_live(self) -> None:
        """Stop streaming the live readout and free the device J_l copies."""
        with self.state_lock:
            self.live = None

    @property
    def live_layers(self) -> list[int] | None:
        """The live readout's layer list, or ``None`` when it's off."""
        with self.state_lock:
            if self.live is None:
                return None
            return list(self.live["layers"])

    @property
    def active_source(self) -> str | None:
        """The active output-side lens source in the public source syntax.

        THE resolver — one ``active.json`` read, the same label that stamps
        every lens measurement binding.  A listing that answered from the
        prepared-sources scan instead would report ``null`` for an active
        pointer whose artifact is gone while persisted rows still carry its
        label.
        """
        from drowse.lens.sources import LENS_SOURCES

        active = LENS_SOURCES.active(self.model_id, "output")
        return None if active is None else LENS_SOURCES.format(active)

    @property
    def live_state(self) -> LensLiveState:
        layers = self.live_layers
        return LensLiveState(
            enabled=layers is not None,
            layers=tuple(layers) if layers is not None else None,
        )

    def set_live(self, enabled: bool, **kwargs: Any) -> LensLiveState:
        """Toggle the live workspace readout.

        ``layers`` (the family extra) selects the live layer set; omitted
        means every fitted layer.  Disabling with an explicit layer list is
        a caller error, not a silent no-op.
        """
        layers = kwargs.pop("layers", None)
        if kwargs:
            raise TypeError(
                f"lens live takes only 'layers', got {sorted(kwargs)}"
            )
        if not enabled:
            if layers is not None:
                raise TypeError("lens live disable takes no 'layers'")
            self.disable_live()
            return LensLiveState(enabled=False, layers=None)
        resolved = self.enable_live(layers=layers)
        return LensLiveState(enabled=True, layers=tuple(resolved))

    # -------------------------------------------------------------- replay

    def token_readout(
        self,
        node_id: str,
        raw_index: int,
        *,
        top_k: int | None = None,
        layers: "list[int] | str | None" = None,
        apply_steering: bool = True,
        raw: bool = False,
    ) -> dict[str, Any]:
        """The loom-anchored J-lens replay, as the finished
        ``scope="replay"`` measurement envelope.

        :meth:`_replay_readout` runs the replay itself (prompt rebuild, one
        capture forward under the node's recipe steering); this method owns
        the envelope, so the route never reshapes a family-native dict.  The
        replay hands back per-layer *probabilities*, which is exactly what
        ``build_measurements`` takes — there is no exp/log round-trip on
        this hop.
        """
        from drowse.readout.measurements import build_measurements

        width = 8 if top_k is None else int(top_k)
        if not 1 <= width <= 256:
            raise ValueError("top_k must be in [1, 256]")
        out = self._replay_readout(
            node_id,
            raw_index,
            layers=layers if layers is not None else "all",
            top_k=width,
            apply_steering=apply_steering,
            raw=raw,
        )
        readout = out.get("readout", {})
        measurements = build_measurements(
            scope="replay",
            provenance="replayed",
            lens_readout={
                int(layer): [(str(tok), float(p_l)) for tok, p_l, _tid in rows]
                for layer, rows in readout.items()
            },
            lens_token_ids={
                int(layer): [int(tid) for _tok, _p_l, tid in rows]
                for layer, rows in readout.items()
            },
            lens_aggregate=[
                (str(tok), float(strength), float(com), float(spread))
                for tok, strength, com, spread in out.get("aggregate", [])
            ],
            lens_source=self.active_source,
            steering=(out.get("steering") if apply_steering else None),
        )
        return {"measurements": measurements}

    def _replay_readout(
        self,
        node_id: str,
        raw_index: int,
        *,
        layers: LayerSelector = None,
        top_k: int = 8,
        apply_steering: bool = True,
        raw: bool = False,
    ) -> dict[str, Any]:
        """Lens readout at one decode step of a loom node.

        Rebuilds the exact token stream that *produced* the token (the
        node's rendered prompt, as :meth:`DrowseSession.fork_from_token`
        replays it, plus ``raw_token_ids[:raw_index]``), runs one capture
        forward under the node's recipe steering when ``apply_steering``,
        and reads ``softmax(W_U · norm(J_l h))`` at the final position per
        layer.  An always-active affine term reproduces the decode
        injections exactly; phase-split and probe-gated terms fire as the
        trigger context's rest state decides.  ``raw`` selects the flat
        render.

        Returns ``{node_id, raw_index, token_id, token_text, steering,
        readout: {layer: [(token, p_l, id), ...]}, aggregate: [(token,
        strength, com, spread), ...]}``.  Raises
        :class:`~drowse.lens.jlens.LensNotFittedError` with no lens,
        :class:`UnknownNodeError` / :class:`InvalidNodeOperationError` on a
        bad target, ``ValueError`` on an unfitted layer.
        """
        from drowse.loom.errors import InvalidNodeOperationError
        from drowse.loom.recipe import INHERIT_SYSTEM_PROMPT

        host = self._replay_host
        if host is None:
            raise RuntimeError("the lens token replay needs a session replay host")
        runtime = self.runtime
        lens = self._require()
        req = runtime.resolve_layers(lens, layers)
        node = host.tree.get(node_id)
        if node.role not in ("assistant", "user"):
            raise InvalidNodeOperationError(
                f"lens token readout: {node_id!r} is a {node.role} node — "
                f"only a turn with a decode record can be read out"
            )
        raw_ids = node.raw_token_ids
        if not raw_ids:
            raise InvalidNodeOperationError(
                f"lens token readout: {node_id!r} has no raw token record "
                f"(transcript-loaded node)"
            )
        if not 0 <= raw_index < len(raw_ids):
            raise InvalidNodeOperationError(
                f"lens token readout: raw_index {raw_index} out of range "
                f"[0, {len(raw_ids)}) for {node_id!r}"
            )

        recipe = node.recipe
        steering_expr = (
            recipe.steering if (apply_steering and recipe is not None) else None
        )
        # The steering scope opens OUTSIDE the exclusive guard: the
        # composer's push/pop take the generation lock blocking, so nesting
        # the scope inside the guard would self-deadlock.
        scope = host.steering(steering_expr) if steering_expr else nullcontext()
        with scope:
            thinking_req = recipe.thinking if recipe is not None else None
            supports_thinking = host.chat_traits.supports_thinking
            use_thinking = (
                supports_thinking
                if thinking_req is None
                else bool(thinking_req) and supports_thinking
            )
            if raw:
                prompt_ids = host._prepare_input(
                    "", raw=True, parent_node_id=node.parent_id,
                )
            else:
                # Continue-mode rebuild: the history walk to the parent
                # carries every prior turn with its stamped label, so the
                # render is byte-identical to the one that produced the node.
                prompt_ids = host._prepare_input(
                    None,
                    thinking=use_thinking,
                    parent_node_id=node.parent_id,
                    system_prompt=(
                        recipe.system_prompt if recipe is not None
                        else INHERIT_SYSTEM_PROMPT
                    ),
                    user_role=node.role_label if node.role == "user" else None,
                    assistant_role=(
                        node.role_label if node.role == "assistant" else None
                    ),
                    gen_seat=node.role,
                )
            if raw_index > 0:
                prefix = torch.tensor(
                    [[int(t) for t in raw_ids[:raw_index]]],
                    dtype=prompt_ids.dtype, device=prompt_ids.device,
                )
                ids = torch.cat([prompt_ids, prefix], dim=1)
            else:
                ids = prompt_ids
            with self._exclusive(
                "session.lens.token_readout called while another model use "
                "is in flight",
                phase_msg="session.lens.token_readout called while a "
                "generation is in flight",
            ):
                hidden = self._capture_hidden(
                    ids, layer_indices=req, pool_index=ids.shape[1] - 1,
                )
                pair_rows = [(layer, hidden[layer]) for layer in req]
                logits = runtime.logits_rows(lens, pair_rows)
                decoded = runtime.topk_rows(
                    lens, pair_rows, top_k=top_k, logits=logits,
                )
                readout = dict(zip(req, decoded))
                agg = runtime.aggregate_rows(logits, req, top_k=top_k)
        token_id = int(raw_ids[raw_index])
        return {
            "node_id": node_id,
            "raw_index": int(raw_index),
            "token_id": token_id,
            "token_text": str(runtime.tokenizer.decode([token_id])),
            "steering": steering_expr,
            "readout": readout,
            "aggregate": agg,
        }

    def live_readout_step(
        self, *, top_k: int = 8, step_id: int = -1,
    ) -> (
        tuple[
            dict[int, list[tuple[str, float]]],
            list[tuple[str, float, float, float]],
            dict[int, list[int]],
        ]
        | None
    ):
        """One decode step's lens readout from the capture's latest slices.

        Runs post-forward at the token tap (never inside a hook).  Returns
        ``(per_layer, aggregate, token_ids)`` — top-k tokens per selected
        layer scored by per-layer softmax probability (the one strength
        unit every lens surface reports), the layer-aggregated chip list,
        and the vocabulary ids already selected by ``topk``.  Reuses the
        gate callback's stash rows when they came from THIS forward
        (``stash["step"] == step_id`` — staleness is structural, reuse is
        idempotent, and ``step_id < 0`` never matches) and the layer sets
        overlap.
        """
        runtime = self.runtime
        state = self._measurement_live()
        if state is None or not self.active_for_generation:
            return None
        buckets = self._capture.per_layer_buckets()
        layers_present: list[int] = []
        hidden_rows: list[torch.Tensor] = []
        stack_rows: list[int] = []
        layer_rows: dict[int, int] = state["layer_rows"]
        for layer in state["layers"]:
            bucket = buckets.get(layer)
            if not bucket:
                continue
            layers_present.append(layer)
            # Keep the raw bucket reference until after stash reuse is
            # resolved: an exact gate+live cache hit never needs these hidden
            # rows, so it should not pay a dtype/device conversion just to
            # discard them.
            hidden_rows.append(bucket[-1])
            stack_rows.append(layer_rows[layer])
        if not layers_present:
            return None
        stash = self.step_stash
        logits: torch.Tensor | None = None
        probabilities: torch.Tensor | None = None
        cached_logits: dict[int, torch.Tensor] = {}
        cached_probs: dict[int, torch.Tensor] = {}
        if (
            stash is not None
            and step_id >= 0
            and stash.get("step") == step_id
        ):
            stash_layers = tuple(int(layer) for layer in (stash.get("layers") or ()))
            if stash_layers == tuple(layers_present):
                # The common gate+live path: exact row-set match.  Keep the
                # existing zero-copy reuse of the full matrix rather than
                # restacking full-vocab rows.
                logits = stash["logits"]
                probabilities = stash.get("probabilities")
            else:
                # The gate callback may already have computed this forward's
                # band logits before the token tap.  Reuse any overlapping
                # rows rather than requiring the live display layer set to
                # match exactly.  Softmax calibration is per-layer, so cached
                # probability rows compose exactly with newly computed rows.
                for row, layer in enumerate(stash_layers):
                    if layer in layers_present:
                        cached_logits[int(layer)] = stash["logits"][row]
                        probs = stash.get("probabilities")
                        if probs is not None:
                            cached_probs[int(layer)] = probs[row]
        computed_logits: dict[int, torch.Tensor] = {}
        if logits is None:
            missing = [
                (layer, hidden, transport_row)
                for layer, hidden, transport_row in zip(
                    layers_present, hidden_rows, stack_rows, strict=True,
                )
                if layer not in cached_logits
            ]
            if missing:
                J_stack: torch.Tensor = state["J_stack"]
                J = select_rows(
                    J_stack,
                    [row for _layer, _hidden, row in missing],
                )
                H = torch.stack(
                    [
                        hidden.to(torch.float32)
                        for _layer, hidden, _row in missing
                    ],
                    dim=0,
                )
                computed = state["readout"](transport_rows(J, H))
                if not cached_logits and len(missing) == len(layers_present):
                    logits = computed
                else:
                    computed_logits = {
                        layer: computed[row]
                        for row, (layer, _hidden, _transport) in enumerate(missing)
                    }
            if logits is None:
                logits = torch.stack(
                    [
                        cached_logits[layer]
                        if layer in cached_logits else computed_logits[layer]
                        for layer in layers_present
                    ],
                    dim=0,
                )
        if probabilities is None:
            if not cached_probs and not computed_logits:
                probabilities = readout_probabilities(logits)
            else:
                probability_rows: dict[int, torch.Tensor] = dict(cached_probs)
                uncached_prob_layers = [
                    layer for layer in layers_present
                    if layer not in probability_rows
                ]
                if uncached_prob_layers:
                    uncached_logits = torch.stack(
                        [
                            computed_logits[layer]
                            if layer in computed_logits else cached_logits[layer]
                            for layer in uncached_prob_layers
                        ],
                        dim=0,
                    )
                    uncached_probs = readout_probabilities(uncached_logits)
                    probability_rows.update({
                        layer: uncached_probs[row]
                        for row, layer in enumerate(uncached_prob_layers)
                    })
                probabilities = torch.stack(
                    [probability_rows[layer] for layer in layers_present], dim=0,
                )
        # Pinned lens probes ride the same calibrated matrix — per-step
        # readout-channel readings for the payload merge.  A gate callback
        # may already have computed the same readings from these exact rows;
        # reuse them to avoid a second selected-token host sync on the
        # pinned+gated+live path.
        if self._measurement_specs():
            readings_reused = False
            if (
                stash is not None
                and step_id >= 0
                and stash.get("readings_step") == step_id
            ):
                reading_layers = tuple(
                    int(layer) for layer in (stash.get("readings_layers") or ())
                )
                if reading_layers == tuple(layers_present):
                    self.last_step_readings = cast(
                        "dict[str, ScalarReading]",
                        stash.get("readings") or {},
                    )
                    readings_reused = True
            if not readings_reused:
                self.last_step_readings = self.score_probes_from_rows(
                    layers=list(layers_present), probabilities=probabilities,
                )
        else:
            self.last_step_readings = None
        # Display scores are per-layer softmax probabilities — the one
        # strength unit every lens surface reports (softmax is monotone, so
        # the top-k selection is unchanged from the raw-logit ranking).
        k = min(max(int(top_k), 0), int(probabilities.shape[-1]))
        vals, idxs = probabilities.topk(k, dim=-1)
        depth_tensor = runtime.depth_tensor(layers_present, probabilities.device)
        agg_idxs, agg_stats = aggregate_readout_tensors_from_probabilities(
            probabilities,
            runtime.depths(layers_present),
            top_k=k,
            depth_tensor=depth_tensor,
        )
        # Pack the tiny K-wide result through the shared MPS-safe helper and
        # synchronize once.
        n_layers = len(layers_present)
        host_rows = pack_readout_rows_to_host(
            vals,
            idxs,
            agg_stats,
            agg_idxs.reshape(1, -1),
        ).tolist()
        all_vals = host_rows[:n_layers]
        all_idxs = host_rows[n_layers:2 * n_layers]
        agg_host = host_rows[2 * n_layers:]
        out: dict[int, list[tuple[str, float]]] = {}
        token_ids: dict[int, list[int]] = {}
        for row, layer in enumerate(layers_present):
            pairs: list[tuple[str, float]] = []
            for v, i in zip(all_vals[row], all_idxs[row]):
                pairs.append((runtime.decode_id(int(i)), float(v)))
            out[layer] = pairs
            token_ids[layer] = [int(i) for i in all_idxs[row]]
        agg = [
            (
                runtime.decode_id(int(agg_host[3][j])),
                float(agg_host[0][j]),
                float(agg_host[1][j]),
                float(agg_host[2][j]),
            )
            for j in range(k)
        ]
        return out, agg, token_ids

    # ------------------------------------------------------- authored prefill

    def authored_capture(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        top_k: int,
    ) -> tuple[
        dict[int, list[tuple[str, float]]],
        list[tuple[str, float, float, float]],
        dict[int, list[int]],
        dict[str, ScalarReading],
    ] | None:
        """Live J-LENS payload for one retained authored producer row.

        Computation only — the token matching, ``j-1`` producer semantics,
        row ordering, and loom persistence stay in the session's
        authored-prefill orchestration.
        """
        runtime = self.runtime
        state = self._measurement_live()
        if state is None or not self.active_for_generation:
            return None
        layers = [int(layer) for layer in state["layers"] if int(layer) in hidden]
        if not layers:
            return None
        lens, specs = self._measurement_state()
        if lens is None:
            return None
        logits = runtime.logits_rows(
            lens, [(layer, hidden[layer]) for layer in layers],
        )
        probabilities = readout_probabilities(logits)
        readings = self.score_probes_from_rows(
            layers=layers, probabilities=probabilities,
        ) if specs else {}
        k = min(max(int(top_k), 0), int(probabilities.shape[-1]))
        values, indices = probabilities.topk(k, dim=-1)
        value_rows = values.detach().to("cpu").tolist()
        id_rows = indices.detach().to("cpu").tolist()
        per_layer: dict[int, list[tuple[str, float]]] = {}
        token_ids: dict[int, list[int]] = {}
        for row, layer in enumerate(layers):
            per_layer[layer] = [
                (runtime.decode_id(int(tid)), float(value))
                for value, tid in zip(value_rows[row], id_rows[row], strict=True)
            ]
            token_ids[layer] = [int(tid) for tid in id_rows[row]]
        aggregate = runtime.aggregate_rows(
            None, layers, top_k=k, probabilities=probabilities,
        )
        return per_layer, aggregate, token_ids, readings


__all__ = ["LensInstrument"]
