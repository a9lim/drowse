"""ReadPlane — the session's per-generation read side.

One object owns what a generation reads from the residual stream:

- the :class:`~drowse.readout.capture_buffer.HiddenCapture` buffers, this
  generation's :class:`~drowse.readout.planner.CaptureState`, and the per-token
  rows the step sinks append (full / lean readings, gated-subset scalars);
- the instrument transaction (:meth:`ReadPlane.bind_runs` /
  :meth:`ReadPlane.close_runs`) and the read demand
  (:meth:`ReadPlane.resolve_demand`), resolved against the attached rosters;
- applying :func:`~drowse.readout.planner.plan_capture`'s pure decision
  (:meth:`ReadPlane.begin`) and the per-token step sinks it arms;
- the per-step reads the decode loop makes: the token tap's geometry reading
  (:meth:`ReadPlane.geometry_step_readings`) and the probe-gate score callback
  (:meth:`ReadPlane.gate_score_callback`);
- the finalize reads: ONE pooled-slice resolution shared by all three
  families (:meth:`ReadPlane.pooled_slice`), the mode dispatch that turns a
  generation's capture into :class:`~drowse.generation.finalize.FinalReads`
  (:meth:`ReadPlane.finalize`), the batched path's per-row reads, and the
  offline :meth:`ReadPlane.score_hidden`.

Its dependencies are explicit: the capture buffer, the Monitor, the three
read instruments, the model's decoder layers, the tokenizer, and the
steering stack's active gate references.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import torch

from drowse.errors import DrowseError
from drowse.fit.capture import last_content_index
from drowse.generation.finalize import FinalReads
from drowse.generation.results import ProbeReading
from drowse.readout.planner import (
    CaptureMode,
    CapturePlan,
    CaptureState,
    GateDemand,
    ReadDemand,
    plan_capture,
    resolve_read_demand,
)
from drowse.readout.types import AGG_TAIL_DEPTH, InstrumentPlan, ReadRequest

if TYPE_CHECKING:
    from drowse.generation.decode import GenerationState
    from drowse.lens.instrument import LensInstrument
    from drowse.readout.capture_buffer import HiddenCapture
    from drowse.readout.geometry import GeometryInstrument
    from drowse.readout.monitor import Monitor
    from drowse.readout.types import GateRef
    from drowse.sae.instrument import SaeInstrument

GateScoreCallback = Callable[[int], dict[str, float]]


class ReadPlane:
    """Capture planning, per-token reads and finalize reads for one session."""

    def __init__(
        self,
        *,
        capture: "HiddenCapture",
        monitor: "Monitor",
        geometry: "GeometryInstrument",
        lens: "LensInstrument",
        sae: "SaeInstrument",
        layers: Any,
        tokenizer: Any,
        gate_refs: Callable[[], "list[tuple[str, GateRef]]"],
    ) -> None:
        """``layers`` are the model's decoder layers (transient hooks attach
        there); ``gate_refs`` returns the active steering's probe-gate
        references (the composer's stack)."""
        #: The hidden-state buffers every family reads.
        self.capture = capture
        self._monitor = monitor
        self._geometry = geometry
        self._lens = lens
        self._sae = sae
        self._layers = layers
        self._tokenizer = tokenizer
        self._gate_refs = gate_refs
        #: This generation's capture configuration (FULL between generations).
        self.state = CaptureState()
        # Per-token rows the step sinks append: full / lean readings
        # (INCREMENTAL / LEAN_INCREMENTAL), gated-subset scalars
        # (GATING_SUBSET).  Row ``k`` belongs to decode forward ``k``.
        self._rows: list[dict[str, ProbeReading]] = []
        self._gate_rows: list[dict[str, float]] = []

    # ------------------------------------------------------- instrument runs

    def bind_runs(
        self,
        geometry_request: ReadRequest,
        lens_request: ReadRequest,
        sae_request: ReadRequest,
    ) -> tuple[InstrumentPlan, InstrumentPlan, InstrumentPlan]:
        """Run the uniform instrument transaction: close → prepare → plan → bind.

        The defensive close comes first: a stale pin would short-circuit the
        lens disk refresh inside ``prepare``, and a stale binding would keep
        serving frozen specs to idle reads.  ``prepare`` is the
        source-boundary step (the lens refreshes and pins its resident lens
        there and snapshots specs + live config against that identity);
        ``plan``/``bind`` consume the prep only, so an interleaved adoption
        cannot desynchronize them.  ``bind`` freezes each family's probe
        specs, so every token, gate and final aggregate of this generation
        measures the same roster and resident lens.
        """
        self.close_runs()
        geometry_prep = self._geometry.prepare(geometry_request)
        lens_prep = self._lens.prepare(lens_request)
        sae_prep = self._sae.prepare(sae_request)
        geometry_plan = self._geometry.plan(geometry_prep)
        lens_plan = self._lens.plan(lens_prep)
        sae_plan = self._sae.plan(sae_prep)
        self._geometry.bind(geometry_plan, geometry_prep)
        self._lens.bind(lens_plan, lens_prep)
        self._sae.bind(sae_plan, sae_prep)
        return geometry_plan, lens_plan, sae_plan

    def close_runs(self) -> None:
        """Close every family's per-generation run (idempotent teardown).

        Every capture transaction that binds runs must reach this in its
        ``finally``: a bound run leaking past its transaction pins a stale
        lens and freezes SAE units past their generation.
        """
        self._geometry.close_run()
        self._lens.close_run()
        self._sae.close_run()

    # ---------------------------------------------------------------- demand

    def resolve_gates(self) -> GateDemand:
        """The active probe gates, resolved against the attached rosters.

        A key naming an attached J-lens or SAE probe is channel-validated
        (``Instrument.validate_gate``): a gate on a channel the family can
        never produce raises ``UnsupportedProbeChannelError`` here, at
        generation preflight, instead of sitting silently inactive.
        """
        refs = self._gate_refs()
        if not refs:
            return GateDemand()
        attached = set(self._monitor.probe_names)
        return GateDemand(
            monitor_keys=frozenset(
                key for key, ref in refs if ref.probe in attached
            ),
            monitor_probes=frozenset(
                ref.probe for _key, ref in refs if ref.probe in attached
            ),
            lens_keys=_family_gate_keys(self._lens, refs),
            sae_keys=_family_gate_keys(self._sae, refs),
        )

    def resolve_demand(
        self,
        *,
        want_hidden: bool,
        live_scores_on: bool,
        has_monitor_probes: bool,
        loom_attached: bool,
        wants_live_token_scores: bool,
        persists_layer_scores: bool,
        persists_probe_row: bool,
        persists_subspace_coords: bool,
        live_lens_active: bool,
        live_sae_active: bool,
        final_probe_aggregate: bool,
        capture_prompt: bool,
    ) -> ReadDemand:
        """This generation's :class:`ReadDemand` (see
        :func:`~drowse.readout.planner.resolve_read_demand`)."""
        return resolve_read_demand(
            self.resolve_gates(),
            want_hidden=want_hidden,
            live_scores_on=live_scores_on,
            has_monitor_probes=has_monitor_probes,
            loom_attached=loom_attached,
            wants_live_token_scores=wants_live_token_scores,
            persists_layer_scores=persists_layer_scores,
            persists_probe_row=persists_probe_row,
            persists_subspace_coords=persists_subspace_coords,
            live_lens_active=live_lens_active,
            live_sae_active=live_sae_active,
            final_probe_aggregate=final_probe_aggregate,
            capture_prompt=capture_prompt,
        )

    # --------------------------------------------------------------- capture

    def begin(
        self,
        demand: ReadDemand | None = None,
        *,
        widen: bool = False,
        persistent_buffers: "dict[int, torch.Tensor] | None" = None,
    ) -> bool:
        """Bind the instrument runs and attach this generation's capture.

        Returns whether capture is attached.  ``demand`` ``None`` is the
        all-defaults demand (full per-token scoring, a final aggregate, both
        live readouts on, no gates).  ``widen`` (``return_hidden``) covers
        every model layer with full retention.  ``persistent_buffers`` are
        the compile-clean capture buffers, passed only when this generation
        may ride them (compiled clean route, steering lowered to offsets or
        absent).
        """
        demand = ReadDemand() if demand is None else demand
        geometry_plan, lens_plan, sae_plan = self.bind_runs(
            ReadRequest(
                gate_keys=demand.gating_probe_keys,
                per_token_consumers=demand.per_token_full_consumer,
                final_aggregate=demand.final_probe_aggregate,
            ),
            ReadRequest(
                gate_keys=demand.lens_gating_probe_keys,
                live=demand.live_lens_active,
                per_token_consumers=demand.per_token_full_consumer,
                final_aggregate=demand.final_probe_aggregate,
            ),
            ReadRequest(
                gate_keys=demand.sae_gating_probe_keys,
                live=demand.live_sae_active,
                per_token_consumers=demand.per_token_full_consumer,
                final_aggregate=demand.final_probe_aggregate,
            ),
        )
        plan = plan_capture(
            demand,
            widen=widen,
            n_layers=len(self._layers),
            geometry=geometry_plan,
            lens=lens_plan,
            sae=sae_plan,
            has_monitor_probes=bool(self._monitor.probe_names),
            persistent_available=bool(persistent_buffers),
        )
        self._apply(plan, demand, persistent_buffers)
        return plan.attach

    def _apply(
        self,
        plan: CapturePlan,
        demand: ReadDemand,
        persistent_buffers: "dict[int, torch.Tensor] | None",
    ) -> None:
        """Install ``plan`` on the capture and the Monitor."""
        self.state = plan.state()
        self._rows = []
        self._gate_rows = []
        if not plan.attach:
            return
        capture = self.capture
        layers = list(plan.layers)
        capture.clear()
        if plan.persistent:
            assert persistent_buffers is not None
            capture.attach_persistent(layers, persistent_buffers)
        else:
            capture.attach(self._layers, layers)
        if plan.tail_depth is not None:
            capture.set_retention(
                depth=plan.tail_depth,
                step_sink=self._step_sink(plan, demand),
                tail_layers=plan.tail_layers,
            )
        monitor = self._monitor
        if plan.warm:
            # Sequential per-token scoring: curved probes warm-start their
            # nearest-point foot from the previous token, cold-started here.
            monitor.reset_curved_feet()
            monitor.enable_curved_warm(True)
        elif plan.warm is False:
            monitor.enable_curved_warm(False)

    def _step_sink(
        self, plan: CapturePlan, demand: ReadDemand,
    ) -> "Callable[[int, dict[int, torch.Tensor]], None] | None":
        """The per-token step sink ``plan`` arms (rows align 1:1 with decode
        forwards, so every sink appends even an empty row)."""
        monitor = self._monitor
        if plan.sink == "subset":
            gate_plan = monitor.plan_gate_scalars(
                set(demand.gating_probe_keys),
                probe_names=set(plan.gating_subset or ()),
            )
            append_scalars = self._gate_rows.append

            def _score_step_subset(
                step_id: int, latest: dict[int, torch.Tensor],
            ) -> None:
                # Only the exact gate scalar keys (the gate is the sole
                # consumer); finalize pools the full roster from the ring.
                del step_id
                append_scalars(
                    monitor.score_planned_gate_scalars(latest, gate_plan)
                    if latest and gate_plan else {}
                )

            return _score_step_subset
        append_reading = self._rows.append
        if plan.sink == "lean":

            def _score_step_lean(
                step_id: int, latest: dict[int, torch.Tensor],
            ) -> None:
                # Coords only: skips the nearest norm, the assignment softmax
                # and the per-layer host reconstruction.
                del step_id
                append_reading(
                    monitor.score_single_token(latest, coords_only=True)
                    if latest else {}
                )

            return _score_step_lean
        if plan.sink == "full":

            def _score_step(step_id: int, latest: dict[int, torch.Tensor]) -> None:
                # Score while the just-produced slice is the only retained
                # payload; the gate callback and the token tap read the row.
                del step_id
                append_reading(monitor.score_single_token(latest) if latest else {})

            return _score_step
        return None

    def step_callback(self) -> Callable[[int], None]:
        """The decode loop's post-forward ``step_callback``: persistent
        ingest when capture rides the compile-clean buffers, else the
        transient capture's step sink."""
        capture = self.capture
        return (
            capture.ingest_persistent if self.state.persistent
            else capture.fire_step_sink
        )

    def end(self) -> None:
        """Detach the capture hooks (buffers stay readable for finalize)."""
        self.capture.detach()

    def reset(self) -> None:
        """Back to the between-generations state: FULL, no rows."""
        self.state = CaptureState()
        self._rows = []
        self._gate_rows = []

    def close(self) -> None:
        """Session teardown: detach and drop the buffers and rows."""
        self.capture.detach()
        self.capture.clear()
        self.reset()

    # ------------------------------------------------------- per-step reads

    def geometry_step_readings(self, step_id: int) -> "dict[str, ProbeReading] | None":
        """The token tap's geometry reading for this forward.

        When a step sink already scored this forward (rows align 1:1 with
        forwards) the row is read, never rescored: full rows are complete
        readings, lean rows deliberately coords-only.  Otherwise the bound
        geometry run scores the capture's latest slices.
        """
        state = self.state
        rows = self._rows
        if (state.incremental or state.lean) and rows:
            return rows[-1] or None
        latest = {
            layer: bucket[-1]
            for layer, bucket in self.capture.per_layer_buckets().items()
            if bucket
        }
        if not latest:
            return None
        return self._geometry.current_run.observe(step_id, latest) or None

    def gate_score_callback(self) -> GateScoreCallback:
        """The decode loop's ``score_callback`` for probe-gated steering.

        Geometry gate keys read the step sink's row for this forward when one
        exists, else score exactly the planned keys from the latest captured
        slices; J-lens / SAE gate keys score through their bound runs (which
        stash the per-forward rows for the display step).  Keys naming no
        attached probe score nothing, and an empty capture (before the first
        forward) scores nothing, so gates report inactive rather than stale.

        Hot path: runs per token under an active gate.  The gate keys and the
        Monitor are bound once; the capture state and rows are read per call
        (they are this generation's).
        """
        gates = self.resolve_gates()
        capture = self.capture
        monitor = self._monitor
        monitor_gate_keys = set(gates.monitor_keys)
        lens_gate_keys = set(gates.lens_keys)
        sae_gate_keys = set(gates.sae_keys)
        lens, sae = self._lens, self._sae
        plans: dict[tuple[frozenset[str], frozenset[str] | None], Any] = {}

        def _score_monitor_keys(
            latest: dict[int, torch.Tensor],
            gate_keys: "set[str] | frozenset[str]",
            probe_names: "set[str] | frozenset[str] | None",
        ) -> dict[str, float]:
            if not gate_keys:
                return {}
            cache_key = (
                frozenset(gate_keys),
                frozenset(probe_names) if probe_names is not None else None,
            )
            plan = plans.get(cache_key)
            if plan is None:
                plan = monitor.plan_gate_scalars(
                    set(gate_keys),
                    probe_names=set(probe_names) if probe_names is not None else None,
                )
                plans[cache_key] = plan
            return (
                cast("dict[str, float]", monitor.score_planned_gate_scalars(latest, plan))
                if plan else {}
            )

        def _monitor_scalars() -> dict[str, float]:
            state = self.state
            gating_subset = state.gating_subset or None
            gate_keys = state.gating_keys or monitor_gate_keys
            if not gate_keys:
                # No gate names an attached geometry probe.
                return {}
            # The step sink already scored this forward: in gating-subset
            # mode its rows hold just the gated scalars, in incremental mode
            # the whole roster's readings.
            if gating_subset and self._gate_rows:
                return self._gate_rows[-1]
            rows = self._rows
            if state.incremental and rows:
                scalars = monitor.flat_scalars(rows[-1])
                missing = gate_keys - set(scalars)
                if missing:
                    latest = capture.latest_per_layer()
                    if latest:
                        scalars.update(
                            _score_monitor_keys(latest, missing, gating_subset)
                        )
                return scalars
            latest = capture.latest_per_layer()
            if not latest:
                return {}
            return _score_monitor_keys(latest, gate_keys, gating_subset)

        def _score(step_id: int) -> dict[str, float]:
            out = _monitor_scalars()
            if lens_gate_keys:
                # Once per forward: lens logits → strength scalars (also
                # stashed step-keyed for the display step to reuse).
                scalars = lens.current_run.gate_scalars(step_id, None, lens_gate_keys)
                if scalars:
                    out = {**out, **scalars}
            if sae_gate_keys:
                scalars = sae.current_run.gate_scalars(step_id, None, sae_gate_keys)
                if scalars:
                    out = {**out, **scalars}
            return out

        return _score

    # -------------------------------------------------------- finalize reads

    def aggregate_index(
        self, generated_ids: list[int], state: "GenerationState",
    ) -> int | None:
        """The decode forward the end-of-generation aggregate reads.

        The decode loop's recorded visible endpoint when it set one; ``None``
        for a stop-sequence finish whose visible text has no recorded
        endpoint; otherwise the last *content* token (walking back past
        trailing specials).  Generated token ``k`` was produced by forward
        ``k``.
        """
        idx = state.response_aggregate_index
        if idx is not None and 0 <= int(idx) < len(generated_ids):
            return int(idx)
        if state.finish_reason == "stop_sequence" and state.response_text is not None:
            return None
        return last_content_index(generated_ids, self._tokenizer)

    def pooled_slice(
        self,
        generated_ids: list[int],
        state: "GenerationState",
        *,
        stack: "dict[int, torch.Tensor] | None" = None,
    ) -> dict[int, torch.Tensor]:
        """Per-layer ``[D]`` slice at the end-of-generation aggregate position.

        THE finalize-pooling resolution, shared by all three families.  Empty
        when there is no aggregate position or nothing was captured.  Every
        ring mode counts decode forwards, so
        :meth:`HiddenCapture.tail_slice_at` maps the forward index into the
        ring.  FULL retention keeps every forward but never advances that
        counter (the ring mapping would clamp to the last slice), so there the
        stack is indexed directly; ``stack`` reuses one already built.
        """
        agg_fwd = self.aggregate_index(generated_ids, state)
        return self._pooled_at(agg_fwd, stack)

    def _pooled_at(
        self, agg_fwd: int | None, stack: "dict[int, torch.Tensor] | None",
    ) -> dict[int, torch.Tensor]:
        if agg_fwd is None:
            return {}
        if self.state.mode is CaptureMode.FULL:
            rows_by_layer = stack if stack else self.capture.stacked()
            return {
                layer: rows[agg_fwd]
                for layer, rows in rows_by_layer.items()
                if rows.shape[0] > agg_fwd
            }
        return self.capture.tail_slice_at(agg_fwd)

    def finalize(
        self,
        generated_ids: list[int],
        *,
        state: "GenerationState",
        return_hidden: bool,
        return_probe_readings: bool,
    ) -> FinalReads:
        """Turn this generation's capture into its final per-family reads.

        Dispatches on the capture mode: INCREMENTAL reads the scored rows,
        LEAN_INCREMENTAL takes the coord stream from its lean rows and
        re-scores the full aggregate once from the tail ring, aggregate-only
        (and gating-subset) pools once, FULL scores the stack.  The lens and
        SAE aggregates share the one pooled slice.  ``return_hidden`` returns
        the full stack trimmed to the generated length, on CPU.
        """
        monitor_names = list(self._monitor.probe_names)
        mode = self.state.mode
        stack: dict[int, torch.Tensor] = {}
        if generated_ids and (
            return_hidden
            or (return_probe_readings and monitor_names and mode is CaptureMode.FULL)
        ):
            stack = self.capture.stacked()

        agg_fwd: int | None = None
        pooled: dict[int, torch.Tensor] | None = None
        if generated_ids:
            agg_fwd = self.aggregate_index(generated_ids, state)

        def _pooled() -> dict[int, torch.Tensor]:
            nonlocal pooled
            if pooled is None:
                pooled = self._pooled_at(agg_fwd, stack)
            return pooled

        geometry: dict[str, ProbeReading] = {}
        per_token: dict[str, list[float]] | None = None
        if return_probe_readings and monitor_names and generated_ids:
            if mode is CaptureMode.INCREMENTAL:
                geometry, stream = self._score_incremental(
                    generated_ids, monitor_names, agg_fwd,
                )
            elif mode is CaptureMode.LEAN_INCREMENTAL:
                stream = _coord_stream(
                    self._rows[:len(generated_ids)], len(generated_ids),
                    monitor_names,
                )
                geometry = self._geometry_aggregate(monitor_names, _pooled())
            elif self.state.aggregate_only:
                geometry = self._geometry_aggregate(monitor_names, _pooled())
                stream = {}
            elif stack:
                geometry, stream = self._monitor.score_per_token(
                    stack, generated_ids, self._tokenizer,
                    aggregate_index=-1 if agg_fwd is None else agg_fwd,
                )
            else:
                stream = {}
            geometry = dict(geometry)
            per_token = stream or None

        want_readouts = bool(return_probe_readings and generated_ids)
        lens = (
            self._lens.score_aggregate(generated_ids, _pooled())
            if want_readouts and self._lens.names else {}
        )
        sae = (
            self._sae.score_aggregate(generated_ids, _pooled())
            if want_readouts and self._sae.names else {}
        )

        hidden_states: dict[int, torch.Tensor] | None = None
        if return_hidden and generated_ids and stack:
            n = len(generated_ids)
            hidden_states = {}
            for layer, rows in stack.items():
                if rows.shape[0] > n:
                    rows = rows[:n]
                elif rows.shape[0] < n:
                    continue
                hidden_states[layer] = rows.detach().to("cpu")
        return self._reads(geometry, lens, sae, per_token, hidden_states)

    def begin_batch(self, want_readings: bool) -> bool:
        """Bind the runs and arm the batched ``[B, D]`` tail ring.

        The ``generate_batch`` fast path's capture: one request serves all
        three families (the lens prepare pins one lens for every row), and
        the ring covers the union of their declared layers.  Returns whether
        any family will read a per-row aggregate.
        """
        captures = want_readings and bool(
            self._geometry.names or self._lens.names or self._sae.names
        )
        request = ReadRequest(final_aggregate=want_readings)
        plans = self.bind_runs(request, request, request)
        if captures:
            layer_set: set[int] = set()
            for plan in plans:
                layer_set |= plan.latest_layers
                layer_set |= plan.tail_layers
            self.capture.clear()
            if layer_set:
                self.capture.attach_batch_tail(
                    self._layers, sorted(layer_set), depth=AGG_TAIL_DEPTH,
                )
        return captures

    def batch_row_reads(self, row: int, generated_ids: list[int]) -> FinalReads:
        """One batch row's aggregates from its slice of the batched ring.

        The lens/SAE runs measure against their bind-time binding, so a
        per-row callback detaching a probe cannot change later rows (and a
        geometry detach is refused while the batch runs).
        """
        pooled = (
            self.capture.batch_tail_slice_at(
                row, last_content_index(generated_ids, self._tokenizer),
            )
            if generated_ids else {}
        )
        names = list(self._monitor.probe_names)
        return self._reads(
            self._geometry_aggregate(names, pooled) if names else {},
            self._lens.score_aggregate(generated_ids, pooled),
            self._sae.score_aggregate(generated_ids, pooled),
            None,
            None,
        )

    def _reads(
        self,
        geometry: dict[str, ProbeReading],
        lens: dict[str, Any],
        sae: dict[str, Any],
        per_token: dict[str, list[float]] | None,
        hidden_states: dict[int, torch.Tensor] | None,
    ) -> FinalReads:
        """Package the reads with the live sources the envelope reports."""
        live_lens = self._lens.live
        live_sae = self._sae.live
        return FinalReads(
            geometry=geometry,
            lens=lens,
            sae=sae,
            per_token=per_token,
            hidden_states=hidden_states,
            lens_source=(
                live_lens.get("source")
                if lens and isinstance(live_lens, dict) else None
            ),
            sae_source=(
                live_sae.get("source")
                if sae and isinstance(live_sae, dict) else None
            ),
            sae_layer=(
                live_sae.get("layer")
                if sae and isinstance(live_sae, dict) else None
            ),
        )

    def _score_incremental(
        self, generated_ids: list[int], names: list[str], agg_fwd: int | None,
    ) -> tuple[dict[str, ProbeReading], dict[str, list[float]]]:
        """Aggregate + per-token coord stream from the live-scored rows."""
        n = len(generated_ids)
        if not self._rows:
            return _empty_readings(names), {name: [] for name in names}
        rows = self._rows[:n]
        stream = _coord_stream(rows, n, names)
        empty = _empty_readings(names)
        agg_row = rows[agg_fwd] if agg_fwd is not None and agg_fwd < len(rows) else {}
        return {name: agg_row.get(name, empty[name]) for name in names}, stream

    def _geometry_aggregate(
        self, names: list[str], pooled: dict[int, torch.Tensor],
    ) -> dict[str, ProbeReading]:
        """The full roster's aggregate at the pooled slice (one read).

        Curved probes score on the cold foot solve: the pooled position (the
        last *content* token) may differ from the live loop's last forward, so
        a warm foot seeded during per-token scoring would not be the right
        start.
        """
        empty = _empty_readings(names)
        if not pooled:
            return empty
        self._monitor.enable_curved_warm(False)
        scored = self._geometry.current_run.observe_aggregate(pooled)
        # Fill any probe the pool missed (a layer absent from the ring).
        return {name: scored.get(name, empty[name]) for name in names}

    # ---------------------------------------------------------- offline reads

    @overload
    def score_hidden(
        self, hidden: dict[int, torch.Tensor], *, per_token: Literal[False] = False,
    ) -> dict[str, ProbeReading]: ...
    @overload
    def score_hidden(
        self, hidden: dict[int, torch.Tensor], *, per_token: Literal[True],
    ) -> tuple[dict[str, ProbeReading], dict[str, list[float]]]: ...
    def score_hidden(
        self, hidden: dict[int, torch.Tensor], *, per_token: bool = False,
    ) -> (
        dict[str, ProbeReading]
        | tuple[dict[str, ProbeReading], dict[str, list[float]]]
    ):
        """Score the attached geometry probes against pre-captured states.

        Each value ``[D]`` is one state (returns the readings); each ``[T, D]``
        a per-token stack (the aggregate pools row ``T-1``; ``per_token=True``
        also returns the per-probe axis-0 coordinate stream).  Mixed shapes,
        uneven ``T``, a dim mismatch against a covering probe, or an empty
        dict raise :class:`DrowseError`.
        """
        if not hidden:
            raise DrowseError("score_hidden: no layers provided")
        shapes = [v.ndim for v in hidden.values()]
        if len(set(shapes)) > 1:
            by_ndim: dict[int, list[int]] = {}
            for layer_idx, t in hidden.items():
                by_ndim.setdefault(t.ndim, []).append(layer_idx)
            detail = ", ".join(
                f"ndim={n} at layers {ls}" for n, ls in sorted(by_ndim.items())
            )
            raise DrowseError(
                "score_hidden: mixed shapes in input; expected either all "
                f"[D] or all [T, D] across layers ({detail})",
            )
        if shapes[0] not in (1, 2):
            raise DrowseError(
                f"score_hidden: expected [D] or [T, D] tensors, got ndim={shapes[0]}",
            )
        # Dim pre-flight against any probe covering the layer, so a shape
        # mismatch is a DrowseError rather than a raw torch matmul error.
        probe_manifolds = self._geometry.manifolds()
        for layer_idx, t in hidden.items():
            actual_dim = t.shape[-1]
            for probe_name, manifold in probe_manifolds.items():
                sub = manifold.layers.get(layer_idx)
                if sub is None:
                    continue
                expected_dim = sub.mean.shape[-1]
                if expected_dim != actual_dim:
                    raise DrowseError(
                        f"score_hidden: dim mismatch at layer {layer_idx} — "
                        f"got {actual_dim}, probe '{probe_name}' expects "
                        f"{expected_dim}",
                    )
                break  # one covering probe settles the expected dim
        if shapes[0] == 1:
            if per_token:
                raise DrowseError(
                    "score_hidden: per_token=True requires [T, D] input; "
                    "got [D] (single state per layer)",
                )
            return self._monitor.score_single_token(hidden)
        # Uneven T across layers is the only ValueError left past the checks.
        try:
            agg, stream = self._monitor.score_stack(hidden)
        except ValueError as exc:
            raise DrowseError(f"score_hidden: {exc}") from exc
        return (agg, stream) if per_token else agg


def _family_gate_keys(
    instrument: Any, refs: "list[tuple[str, GateRef]]",
) -> frozenset[str]:
    """Gate keys naming one readout family's attached probes, each
    channel-validated against what the family can produce."""
    attached = set(instrument.names)
    if not attached:
        return frozenset()
    out: set[str] = set()
    for key, ref in refs:
        if ref.probe in attached:
            instrument.validate_gate(ref)
            out.add(key)
    return frozenset(out)


def _empty_readings(names: list[str]) -> dict[str, ProbeReading]:
    """A zero reading per probe (no tokens, or nothing pooled)."""
    return {name: ProbeReading(fraction=0.0, nearest=[], coords=()) for name in names}


def _coord_stream(
    rows: list[dict[str, ProbeReading]], n: int, names: list[str],
) -> dict[str, list[float]]:
    """Per-probe axis-0 coord stream (length ``n``, 0.0 for an empty row)."""
    stream: dict[str, list[float]] = {name: [0.0] * n for name in names}
    for i, readings in enumerate(rows):
        for name, reading in readings.items():
            if name in stream:
                stream[name][i] = reading.coords[0] if reading.coords else 0.0
    return stream


__all__ = ["GateScoreCallback", "ReadPlane"]
