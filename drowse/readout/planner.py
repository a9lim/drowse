"""The capture planner: one generation's read demand → its physical capture.

Pure decisions with no side effects, so the whole decision matrix is testable
without a session or a model:

- :func:`resolve_read_demand` folds the generation's consumers (live scores, the
  loom row, the per-layer heatmap, inspector coords, the lens/SAE live
  readouts, probe gates) into one :class:`ReadDemand`;
- :func:`plan_capture` turns a demand plus the three families' declared
  :class:`~drowse.readout.types.InstrumentPlan`\\ s into a :class:`CapturePlan`:
  which layers to hook, persistent buffers or transient hooks, the retention
  (unarmed full stack, a length-1 buffer, or a bounded tail ring), which
  per-token step sink scores, the resulting :class:`CaptureMode`, and the
  curved-foot warm-start policy.

The session's read plane (:class:`drowse.session.read_plane.ReadPlane`) binds
the instrument runs, calls :func:`plan_capture`, and applies the plan to its
:class:`~drowse.readout.capture_buffer.HiddenCapture`.

The capture modes trade *when* scoring runs (per token vs once at finalize)
against *what* the capture retains; every read is still a full per-probe
reading.  Their trigger conditions decide per-token scoring cost, so the
branch order in :func:`plan_capture` is load-bearing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal

from drowse.readout.types import AGG_TAIL_DEPTH, InstrumentPlan


class CaptureMode(Enum):
    """How one generation's capture scores its probes.

    - ``INCREMENTAL`` — a full-reading live consumer wants per-token readings:
      score each token live (full roster) into the read plane's rows.
    - ``LEAN_INCREMENTAL`` — the only per-token consumers read axis-0 coords
      (the loom probe row): score each token ``coords_only`` and re-score the
      full aggregate once at finalize from a bounded tail ring.
    - ``AGGREGATE_ONLY`` — nothing consumes a per-token reading (a stateless
      server gen, or a lens/SAE-only capture with no monitor probes): no
      per-token scoring, pool the last content token once at finalize from a
      bounded tail ring.
    - ``GATING_SUBSET`` — per-token scoring is needed *only* to feed probe
      gates: score just the gated subset per token while a tail ring lets
      finalize pool the full roster once.
    - ``FULL`` — full-retention append (``return_hidden`` widen), or the
      degenerate no-probe / capture-disabled state: distinct clones per step
      so ``stacked()`` builds the full ``[T, D]``.
    """

    INCREMENTAL = "incremental"
    LEAN_INCREMENTAL = "lean_incremental"
    AGGREGATE_ONLY = "aggregate_only"
    GATING_SUBSET = "gating_subset"
    FULL = "full"


@dataclass
class CaptureState:
    """One generation's capture configuration, as the plan left it.

    The mode, the orthogonal ``persistent`` flag (capture rides the always-on
    compile-clean buffers rather than transient hooks) and, for gated
    generations, the gated probe names and exact scalar keys.  Read by the
    finalize dispatch, the gate callback and the token tap.
    """

    mode: CaptureMode = CaptureMode.FULL
    persistent: bool = False
    gating_subset: "frozenset[str] | set[str] | None" = None
    gating_keys: "frozenset[str] | set[str] | None" = None
    final_probe_aggregate: bool = True

    @property
    def incremental(self) -> bool:
        return self.mode is CaptureMode.INCREMENTAL

    @property
    def lean(self) -> bool:
        return self.mode is CaptureMode.LEAN_INCREMENTAL

    @property
    def aggregate_only(self) -> bool:
        # GATING_SUBSET also finalizes off the tail ring (its per-token rows
        # feed only the gate), so it shares the aggregate-only full-roster
        # pool when the caller still wants a final full probe aggregate.
        return (
            self.mode is CaptureMode.AGGREGATE_ONLY
            or (
                self.mode is CaptureMode.GATING_SUBSET
                and self.final_probe_aggregate
            )
        )


@dataclass(frozen=True, slots=True)
class GateDemand:
    """The active probe gates, resolved against the attached rosters.

    ``monitor_keys`` are the exact scalar keys naming an attached geometry
    probe and ``monitor_probes`` those probes' names; ``lens_keys`` /
    ``sae_keys`` the channel-validated keys naming an attached J-lens / SAE
    probe.  A gate naming no attached probe lands in none of them.
    """

    monitor_keys: frozenset[str] = frozenset()
    monitor_probes: frozenset[str] = frozenset()
    lens_keys: frozenset[str] = frozenset()
    sae_keys: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ReadDemand:
    """What one generation's consumers need from the read plane.

    Resolved once per generation (:func:`resolve_read_demand`); every derived
    predicate is computed there and only read by :func:`plan_capture`, so the
    caller and the planner cannot form different opinions about the same
    generation.  The defaults are the "everything, nothing gated" demand a
    bare capture wants.

    - ``need_per_token`` — anything consumes a per-token reading (a probe
      gate, a loom token row, a live-scores client).  False means
      aggregate-only: a bounded tail ring, no per-token scoring, one pooled
      read at finalize.
    - ``per_token_full_consumer`` — a per-token consumer wants the FULL
      roster, not just a gate's scalars.  False with ``need_per_token`` true
      is the gating-only case.
    - ``gating_only_probes`` — the gated probe names when gating is the sole
      per-token consumer; the step sink then scores just that subset.
    - ``gating_probe_keys`` / ``lens_gating_probe_keys`` /
      ``sae_gating_probe_keys`` — the exact gate scalar keys per family.
    - ``lean_per_token`` — the live consumers read only axis-0 coords, so
      per-token scoring drops nearest / assignment / per-layer work.
    - ``final_probe_aggregate`` — an end-of-generation aggregate is wanted.
    - ``live_lens_active`` / ``live_sae_active`` — the live readouts feed a
      consumer this generation.
    - ``capture_prompt`` — authored prompt rows are scored this generation,
      which forces transient capture hooks over the persistent buffers.
    """

    need_per_token: bool = True
    per_token_full_consumer: bool = True
    gating_only_probes: frozenset[str] = frozenset()
    gating_probe_keys: frozenset[str] = frozenset()
    lens_gating_probe_keys: frozenset[str] = frozenset()
    sae_gating_probe_keys: frozenset[str] = frozenset()
    lean_per_token: bool = False
    final_probe_aggregate: bool = True
    live_lens_active: bool = True
    live_sae_active: bool = True
    capture_prompt: bool = False


def resolve_read_demand(
    gates: GateDemand,
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
    """Fold one generation's consumers into its :class:`ReadDemand`."""
    # The four UI / loom / persist consumers each need a per-token reading
    # for the FULL roster.  When none of them is live but a probe gate is,
    # gating is the SOLE per-token consumer: the step sink scores just the
    # gated probes and the full roster is pooled once at finalize.
    per_token_full_consumer = bool(
        (live_scores_on and loom_attached)
        or wants_live_token_scores
        or persists_layer_scores
        or persists_probe_row
        or persists_subspace_coords
    )
    # J-lens / SAE gates score on their own paths (the gate callback reads
    # the latest capture slices), so a readout-only gate doesn't force
    # per-token MONITOR scoring: ``need_per_token`` keys on monitor gates.
    needs_monitor_gating = bool(gates.monitor_keys)
    need_per_token = bool(needs_monitor_gating or per_token_full_consumer)
    gating_only_probes = (
        gates.monitor_probes
        if (needs_monitor_gating and not per_token_full_consumer)
        else frozenset()
    )
    # Lean-incremental: the live consumers present read ONLY the axis-0
    # coord (the loom probe row, the node text/aggregate); none needs a
    # richer per-token reading and there is no probe gate.  Per-token scoring
    # then drops the nearest / assignment / per-layer work, and the full
    # aggregate is re-scored once from the tail ring at finalize.
    full_reading_consumer = bool(
        wants_live_token_scores       # full ProbeReading in TokenEvent
        or persists_layer_scores      # per-layer heatmap row
        or persists_subspace_coords   # inspector whitened coords
    )
    lean_per_token = bool(
        need_per_token
        and not want_hidden
        and not needs_monitor_gating
        and not full_reading_consumer
        and has_monitor_probes
    )
    return ReadDemand(
        need_per_token=need_per_token,
        per_token_full_consumer=per_token_full_consumer,
        gating_only_probes=gating_only_probes,
        gating_probe_keys=gates.monitor_keys,
        lens_gating_probe_keys=gates.lens_keys,
        sae_gating_probe_keys=gates.sae_keys,
        lean_per_token=lean_per_token,
        final_probe_aggregate=final_probe_aggregate,
        live_lens_active=live_lens_active,
        live_sae_active=live_sae_active,
        capture_prompt=capture_prompt,
    )


#: Which per-token step sink a plan arms: the gated subset's exact scalars,
#: the lean coords-only reading, or the full roster reading.
SinkKind = Literal["subset", "lean", "full"]


@dataclass(frozen=True, slots=True)
class CapturePlan:
    """One generation's physical capture, decided by :func:`plan_capture`.

    ``attach`` False means no capture at all (no probes would read anything).
    Otherwise hook ``layers`` — through the persistent compile-clean buffers
    when ``persistent`` — and arm the retention: ``tail_depth`` ``None``
    leaves the capture unarmed (full retention, a clone per step); an int
    arms a ring that deep (``1`` = one in-place buffer), restricted to
    ``tail_layers`` when set, with ``sink`` scoring each forward when set.
    ``warm`` is the curved-foot policy: ``True`` cold-starts the feet and
    enables warm-starting (sequential per-token scoring), ``False`` disables
    it (one-shot reads), ``None`` leaves the Monitor alone.
    """

    attach: bool
    layers: tuple[int, ...] = ()
    persistent: bool = False
    mode: CaptureMode = CaptureMode.FULL
    tail_depth: int | None = None
    tail_layers: frozenset[int] | None = None
    sink: SinkKind | None = None
    warm: bool | None = None
    gating_subset: frozenset[str] | None = None
    gating_keys: frozenset[str] | None = None
    final_probe_aggregate: bool = True

    def state(self) -> CaptureState:
        """The :class:`CaptureState` this plan leaves for the generation."""
        return CaptureState(
            mode=self.mode,
            persistent=self.persistent,
            gating_subset=self.gating_subset,
            gating_keys=self.gating_keys,
            final_probe_aggregate=self.final_probe_aggregate,
        )


def plan_capture(
    demand: ReadDemand,
    *,
    widen: bool,
    n_layers: int,
    geometry: InstrumentPlan,
    lens: InstrumentPlan,
    sae: InstrumentPlan,
    has_monitor_probes: bool,
    persistent_available: bool,
) -> CapturePlan:
    """Decide one generation's capture from its demand and the family plans.

    ``widen`` (``return_hidden``) covers every model layer with full
    retention.  Otherwise capture covers the union of the families' declared
    layers.  ``has_monitor_probes`` is the attached geometry roster;
    ``persistent_available`` says the generation runs compile-clean with the
    persistent capture buffers installed and steering lowered to offsets (or
    absent), so capture can ride those buffers instead of transient hooks.
    """
    gating_keys = demand.gating_probe_keys or None
    final = demand.final_probe_aggregate
    if widen:
        layers: Iterable[int] = range(n_layers)
    else:
        union = (
            geometry.latest_layers
            | lens.latest_layers | lens.tail_layers
            | sae.latest_layers | sae.tail_layers
        )
        if not union:
            # Nothing reads a layer: the degenerate FULL (capture-off) state.
            return CapturePlan(
                attach=False,
                gating_keys=gating_keys,
                final_probe_aggregate=final,
            )
        layers = union
    # Compile-clean persistent capture rides the always-on buffers the
    # decode loop ``copy_``s into; anything that forced the eager path
    # (``return_hidden``, authored prompt rows) uses transient hooks.
    persistent = bool(
        not widen and not demand.capture_prompt and persistent_available
    )
    base = CapturePlan(
        attach=True,
        layers=tuple(sorted(layers)),
        persistent=persistent,
        gating_keys=gating_keys,
        final_probe_aggregate=final,
    )
    monitor = bool(not widen and has_monitor_probes)
    if monitor and demand.need_per_token and demand.gating_only_probes:
        # Gating-only subset: per-token scoring feeds only the probe gates, so
        # the sink scores just the gated probes' exact keys.  A tail ring rides
        # alongside so finalize pools the FULL roster once; a pure control call
        # (no final aggregate) keeps one in-place buffer, its capture already
        # narrowed to the gated probes' layers.
        return replace(
            base,
            mode=CaptureMode.GATING_SUBSET,
            tail_depth=AGG_TAIL_DEPTH if final else 1,
            sink="subset",
            warm=True,
            gating_subset=demand.gating_only_probes,
        )
    if monitor and demand.need_per_token and demand.lean_per_token:
        # Lean-incremental: coords-only rows per token for the loom probe
        # row, plus a tail ring for the one full aggregate at finalize.
        return replace(
            base,
            mode=CaptureMode.LEAN_INCREMENTAL,
            tail_depth=AGG_TAIL_DEPTH,
            sink="lean",
            warm=True,
        )
    if monitor and demand.need_per_token:
        # Full incremental: score the roster once per forward while its slice
        # is the only retained payload.  The lens/SAE finalize aggregates pool
        # from a tail ring (a length-1 buffer can't walk back past trailing
        # specials), spanning only the readout families' tail layers; with
        # final readings off, gates and live readouts need just the latest
        # slice.
        readout_tail = lens.tail_layers | sae.tail_layers
        if (
            lens.final_aggregate or lens.gate_keys
            or sae.final_aggregate or sae.gate_keys
        ):
            return replace(
                base,
                mode=CaptureMode.INCREMENTAL,
                tail_depth=AGG_TAIL_DEPTH if final else 1,
                tail_layers=readout_tail or None,
                sink="full",
                warm=True,
            )
        return replace(
            base, mode=CaptureMode.INCREMENTAL, tail_depth=1, sink="full",
            warm=True,
        )
    if monitor:
        # Aggregate-only: probes attached, no per-token consumer.  A bounded
        # tail ring, nothing scored per token; finalize pools the last content
        # token once, on a cold foot (warm-start is a sequential-live
        # optimization).
        return replace(
            base, mode=CaptureMode.AGGREGATE_ONLY, tail_depth=AGG_TAIL_DEPTH,
            warm=False,
        )
    if not widen and (
        lens.latest_layers or lens.tail_layers
        or sae.latest_layers or sae.tail_layers
    ):
        # Lens/SAE-only capture (live readout and/or pinned readout probes, no
        # monitor probes): a bounded ring keeps the latest slice fresh without
        # full retention; with final readings off, one in-place buffer.  No
        # step sink — the live display runs at the token tap and gates score
        # from the latest slices.
        return replace(
            base,
            mode=CaptureMode.AGGREGATE_ONLY,
            tail_depth=AGG_TAIL_DEPTH if final else 1,
            warm=False,
        )
    # FULL retention: the ``return_hidden`` stack.  Scored one-shot, so
    # curved probes keep the cold foot solve for reproducibility.
    return replace(base, warm=False)


__all__ = [
    "CaptureMode",
    "CapturePlan",
    "CaptureState",
    "GateDemand",
    "ReadDemand",
    "SinkKind",
    "plan_capture",
    "resolve_read_demand",
]
