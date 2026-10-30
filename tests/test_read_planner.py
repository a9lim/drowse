"""The capture planner: read demand → physical capture, as pure decisions.

:func:`~drowse.readout.planner.resolve_read_demand` and
:func:`~drowse.readout.planner.plan_capture` are tested here as tables, with
no session or model; the ``ReadPlane.begin`` tests at the bottom check that
the real instruments' plans flow through the planner onto the capture, and a
few tiny-session generations check the mode a real generation lands in.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from drowse.generation.sampling import SamplingConfig
from drowse.readout.planner import (
    CaptureMode,
    CapturePlan,
    GateDemand,
    ReadDemand,
    plan_capture,
    resolve_read_demand,
)
from drowse.readout.types import AGG_TAIL_DEPTH, InstrumentPlan
from tests.support.sessions import TinySessionFactory, register_vectors


# ---------------------------------------------------------------------------
# resolve_read_demand
# ---------------------------------------------------------------------------

_GATES = GateDemand(
    monitor_keys=frozenset({"gate@x"}), monitor_probes=frozenset({"gate"}),
)


def _demand(gates: GateDemand = GateDemand(), **kwargs: Any) -> ReadDemand:
    consumers: dict[str, Any] = dict(
        want_hidden=False,
        live_scores_on=True,
        has_monitor_probes=True,
        loom_attached=False,
        wants_live_token_scores=False,
        persists_layer_scores=False,
        persists_probe_row=False,
        persists_subspace_coords=False,
        live_lens_active=False,
        live_sae_active=False,
        final_probe_aggregate=True,
        capture_prompt=False,
    )
    consumers.update(kwargs)
    return resolve_read_demand(gates, **consumers)


def test_a_gate_only_generation_routes_to_the_gated_subset() -> None:
    demand = _demand(_GATES)
    assert demand.need_per_token is True
    assert demand.per_token_full_consumer is False
    assert demand.gating_only_probes == {"gate"}
    assert demand.gating_probe_keys == {"gate@x"}
    # Gating forces per-token scoring, but lean is for coord-only consumers.
    assert demand.lean_per_token is False


def test_a_gate_beside_a_full_roster_consumer_is_not_a_subset() -> None:
    demand = _demand(_GATES, loom_attached=True)
    assert demand.per_token_full_consumer is True
    assert demand.gating_only_probes == frozenset()
    assert demand.lean_per_token is False


def test_a_loom_attached_generation_is_full_and_lean() -> None:
    demand = _demand(loom_attached=True)
    assert demand.need_per_token is True
    assert demand.per_token_full_consumer is True
    assert demand.gating_only_probes == frozenset()
    assert demand.lean_per_token is True


def test_nothing_consuming_a_token_drops_per_token_scoring() -> None:
    demand = _demand()
    assert demand.need_per_token is False
    assert demand.per_token_full_consumer is False
    assert demand.lean_per_token is False


@pytest.mark.parametrize("consumer", [
    "wants_live_token_scores", "persists_layer_scores", "persists_subspace_coords",
])
def test_a_full_reading_consumer_leaves_lean_off(consumer: str) -> None:
    demand = _demand(loom_attached=True, **cast("dict[str, Any]", {consumer: True}))
    assert demand.per_token_full_consumer is True
    assert demand.lean_per_token is False


def test_hidden_states_leave_lean_off() -> None:
    demand = _demand(loom_attached=True, want_hidden=True)
    assert demand.need_per_token is True
    assert demand.lean_per_token is False


def test_the_loom_row_alone_is_lean() -> None:
    demand = _demand(persists_probe_row=True)
    assert demand.per_token_full_consumer is True
    assert demand.lean_per_token is True


def test_live_scores_off_masks_the_loom_consumer() -> None:
    demand = _demand(loom_attached=True, live_scores_on=False)
    assert demand.need_per_token is False


def test_readout_only_gates_do_not_force_monitor_scoring() -> None:
    demand = _demand(GateDemand(lens_keys=frozenset({"jlens/g"})))
    assert demand.need_per_token is False
    assert demand.lens_gating_probe_keys == {"jlens/g"}
    assert demand.gating_probe_keys == frozenset()


def test_no_monitor_probes_is_never_lean() -> None:
    assert _demand(loom_attached=True, has_monitor_probes=False).lean_per_token is False


# ---------------------------------------------------------------------------
# plan_capture
# ---------------------------------------------------------------------------


def _fam(
    family: str, latest: tuple[int, ...] = (), tail: tuple[int, ...] = (),
    *, gates: tuple[str, ...] = (), final: bool = False,
) -> InstrumentPlan:
    return InstrumentPlan(
        family=family, latest_layers=frozenset(latest), tail_layers=frozenset(tail),
        gate_keys=frozenset(gates), final_aggregate=final,
    )


_NONE = {"geometry": _fam("geometry"), "lens": _fam("lens"), "sae": _fam("sae")}
_MONITOR = {**_NONE, "geometry": _fam("geometry", (2, 4), final=True)}
_SUBSET: dict[str, Any] = dict(
    need_per_token=True, per_token_full_consumer=False,
    gating_only_probes=frozenset({"gate"}), gating_probe_keys=frozenset({"gate@x"}),
)
_LEAN: dict[str, Any] = dict(need_per_token=True, lean_per_token=True)
_FULL: dict[str, Any] = dict(need_per_token=True)
_AGG: dict[str, Any] = dict(need_per_token=False, per_token_full_consumer=False)


def _plan(
    demand: dict[str, Any], plans: dict[str, InstrumentPlan], *,
    widen: bool = False, monitor: bool = True, persistent: bool = False,
) -> CapturePlan:
    return plan_capture(
        ReadDemand(**demand),
        widen=widen,
        n_layers=6,
        geometry=plans["geometry"],
        lens=plans["lens"],
        sae=plans["sae"],
        has_monitor_probes=monitor,
        persistent_available=persistent,
    )


# (case, demand, plans, has monitor probes) -> (mode, layers, depth, tail
# layers, sink, warm, gating subset)
_MATRIX = [
    ("gated subset + final ring", {**_SUBSET}, _MONITOR, True,
     (CaptureMode.GATING_SUBSET, (2, 4), AGG_TAIL_DEPTH, None, "subset", True, {"gate"})),
    ("gated subset, control call", {**_SUBSET, "final_probe_aggregate": False}, _MONITOR, True,
     (CaptureMode.GATING_SUBSET, (2, 4), 1, None, "subset", True, {"gate"})),
    ("lean loom row", {**_LEAN}, _MONITOR, True,
     (CaptureMode.LEAN_INCREMENTAL, (2, 4), AGG_TAIL_DEPTH, None, "lean", True, None)),
    ("lean even without a final aggregate", {**_LEAN, "final_probe_aggregate": False}, _MONITOR, True,
     (CaptureMode.LEAN_INCREMENTAL, (2, 4), AGG_TAIL_DEPTH, None, "lean", True, None)),
    ("full incremental, length-1", {**_FULL}, _MONITOR, True,
     (CaptureMode.INCREMENTAL, (2, 4), 1, None, "full", True, None)),
    ("full incremental + readout ring",
     {**_FULL}, {**_MONITOR, "lens": _fam("lens", tail=(3,), final=True),
                 "sae": _fam("sae", tail=(5,), final=True)}, True,
     (CaptureMode.INCREMENTAL, (2, 3, 4, 5), AGG_TAIL_DEPTH, {3, 5}, "full", True, None)),
    ("full incremental + lens gate, no final",
     {**_FULL, "final_probe_aggregate": False},
     {**_MONITOR, "lens": _fam("lens", (3,), gates=("jlens/g",))}, True,
     (CaptureMode.INCREMENTAL, (2, 3, 4), 1, None, "full", True, None)),
    ("full incremental + sae gate",
     {**_FULL}, {**_MONITOR, "sae": _fam("sae", (5,), gates=("sae/1",))}, True,
     (CaptureMode.INCREMENTAL, (2, 4, 5), AGG_TAIL_DEPTH, None, "full", True, None)),
    ("aggregate only", {**_AGG}, _MONITOR, True,
     (CaptureMode.AGGREGATE_ONLY, (2, 4), AGG_TAIL_DEPTH, None, None, False, None)),
    ("aggregate only, no final", {**_AGG, "final_probe_aggregate": False}, _MONITOR, True,
     (CaptureMode.AGGREGATE_ONLY, (2, 4), AGG_TAIL_DEPTH, None, None, False, None)),
    ("lens live only", {**_AGG}, {**_NONE, "lens": _fam("lens", (1, 3))}, False,
     (CaptureMode.AGGREGATE_ONLY, (1, 3), AGG_TAIL_DEPTH, None, None, False, None)),
    ("lens live only, no final", {**_AGG, "final_probe_aggregate": False},
     {**_NONE, "lens": _fam("lens", (1, 3))}, False,
     (CaptureMode.AGGREGATE_ONLY, (1, 3), 1, None, None, False, None)),
    ("sae tail only", {**_FULL}, {**_NONE, "sae": _fam("sae", tail=(5,), final=True)}, False,
     (CaptureMode.AGGREGATE_ONLY, (5,), AGG_TAIL_DEPTH, None, None, False, None)),
]


@pytest.mark.parametrize(
    ("demand", "plans", "monitor", "expected"),
    [case[1:] for case in _MATRIX],
    ids=[case[0] for case in _MATRIX],
)
def test_plan_capture_matrix(
    demand: dict[str, Any], plans: dict[str, InstrumentPlan], monitor: bool,
    expected: tuple[Any, ...],
) -> None:
    mode, layers, depth, tail_layers, sink, warm, subset = expected
    plan = _plan(demand, plans, monitor=monitor)
    assert plan.attach is True
    assert plan.mode is mode
    assert plan.layers == layers
    assert plan.tail_depth == depth
    assert plan.tail_layers == (frozenset(tail_layers) if tail_layers else None)
    assert plan.sink == sink
    assert plan.warm is warm
    assert plan.gating_subset == (frozenset(subset) if subset else None)
    state = plan.state()
    assert state.mode is mode
    assert state.final_probe_aggregate is demand.get("final_probe_aggregate", True)


def test_no_declared_layer_attaches_nothing() -> None:
    plan = _plan({**_SUBSET, "final_probe_aggregate": False}, _NONE)
    assert plan.attach is False
    assert plan.layers == ()
    assert plan.warm is None
    state = plan.state()
    assert state.mode is CaptureMode.FULL
    assert state.persistent is False
    assert state.gating_keys == {"gate@x"}
    assert state.final_probe_aggregate is False


def test_widen_covers_every_layer_with_full_retention() -> None:
    plan = _plan({**_LEAN}, _MONITOR, widen=True, persistent=True)
    assert plan.layers == tuple(range(6))
    assert plan.mode is CaptureMode.FULL
    assert plan.tail_depth is None and plan.sink is None
    assert plan.persistent is False
    assert plan.warm is False


def test_widen_still_captures_with_no_probe_layers() -> None:
    plan = _plan({**_AGG}, _NONE, widen=True, monitor=False)
    assert plan.attach is True
    assert plan.layers == tuple(range(6))
    assert plan.mode is CaptureMode.FULL


@pytest.mark.parametrize(("capture_prompt", "available", "expected"), [
    (False, True, True),
    (True, True, False),
    (False, False, False),
])
def test_persistent_buffers_need_availability_and_no_prompt_rows(
    capture_prompt: bool, available: bool, expected: bool,
) -> None:
    plan = _plan(
        {**_FULL, "capture_prompt": capture_prompt}, _MONITOR, persistent=available,
    )
    assert plan.persistent is expected
    assert plan.state().persistent is expected


def test_the_subset_wins_over_lean_and_full() -> None:
    # Branch order is load-bearing: a gated subset is checked before lean.
    plan = _plan({**_SUBSET, "lean_per_token": True}, _MONITOR)
    assert plan.mode is CaptureMode.GATING_SUBSET


def test_monitor_branches_need_attached_monitor_probes() -> None:
    plan = _plan({**_FULL}, {**_MONITOR, "lens": _fam("lens", (3,))}, monitor=False)
    assert plan.mode is CaptureMode.AGGREGATE_ONLY
    assert plan.sink is None


# ---------------------------------------------------------------------------
# Real generations (tiny CPU session)
# ---------------------------------------------------------------------------


def _spy_plans(monkeypatch: pytest.MonkeyPatch) -> list[CapturePlan]:
    import drowse.session.read_plane as read_plane

    plans: list[CapturePlan] = []
    real = read_plane.plan_capture

    def _spy(*args: Any, **kwargs: Any) -> CapturePlan:
        plan = real(*args, **kwargs)
        plans.append(plan)
        return plan

    monkeypatch.setattr(read_plane, "plan_capture", _spy)
    return plans


@pytest.mark.parametrize(("call", "mode"), [
    (dict(stateless=True), CaptureMode.AGGREGATE_ONLY),
    (dict(stateless=False), CaptureMode.LEAN_INCREMENTAL),
    (dict(stateless=False, on_token=lambda *_a: None,
          sampling=SamplingConfig(max_tokens=3, persist_per_layer_scores=True)),
     CaptureMode.INCREMENTAL),
    (dict(stateless=True, sampling=SamplingConfig(max_tokens=3, return_hidden=True)),
     CaptureMode.FULL),
])
def test_a_probed_generation_lands_in_the_planned_mode(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
    call: dict[str, Any], mode: CaptureMode,
) -> None:
    session, _script = make_tiny_session.scripted("Hi there.")
    register_vectors(session, "mood")
    session.add_probe("mood")
    plans = _spy_plans(monkeypatch)
    call.setdefault("sampling", SamplingConfig(max_tokens=3))
    result = session.generate("Say hi.", **call).first
    (plan,) = plans
    assert plan.mode is mode
    assert set(result.probe_readings) == {"mood"}
    # The release leaves the read plane between generations.
    assert session._read_plane.state.mode is CaptureMode.FULL


def test_a_gated_generation_scores_only_the_gated_subset(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _script = make_tiny_session.scripted("Hi there.")
    register_vectors(session, "mood", "calm")
    session.add_probe("mood")
    session.add_probe("calm")
    plans = _spy_plans(monkeypatch)
    result = session.generate(
        "Say hi.", stateless=True, steering="0.3 calm@when:mood>-100",
        sampling=SamplingConfig(max_tokens=3),
    ).first
    (plan,) = plans
    assert plan.mode is CaptureMode.GATING_SUBSET
    assert plan.gating_subset == {"mood"}
    # The full roster is still pooled once at finalize.
    assert set(result.probe_readings) == {"mood", "calm"}


def test_an_unprobed_generation_attaches_no_capture(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _script = make_tiny_session.scripted("Hi there.")
    plans = _spy_plans(monkeypatch)
    session.generate("Say hi.", stateless=True, sampling=SamplingConfig(max_tokens=3))
    (plan,) = plans
    assert plan.attach is False
    assert not session._read_plane.capture.is_transient()


def test_return_hidden_pools_lens_and_geometry_at_the_same_position(
    make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: under ``return_hidden`` (FULL retention) the finalize pool
    clamped the never-advanced ring counter to the LAST forward for the
    lens/SAE aggregates, while geometry read the last content token.  The one
    pooled slice indexes the stack at the aggregate position."""
    from drowse.lens.instrument import LensInstrument

    session, _script = make_tiny_session.scripted("Hi there.")
    captured: dict[str, Any] = {}

    def _score_aggregate(
        ids: list[int], pooled: dict[int, torch.Tensor],
    ) -> dict[str, Any]:
        captured["pooled"] = pooled
        return {}

    monkeypatch.setattr(LensInstrument, "names", property(lambda _self: ["jlens/x"]))
    monkeypatch.setattr(session.lens, "score_aggregate", _score_aggregate)
    result = session.generate(
        "Say hi.", stateless=True,
        sampling=SamplingConfig(max_tokens=32, return_hidden=True),
    ).first

    assert result.hidden_states
    agg = session._read_plane.aggregate_index(result.tokens, session.generation_state)
    stack = session._read_plane.capture.stacked()
    assert agg is not None
    assert set(captured["pooled"]) == set(result.hidden_states)
    for layer, pooled in captured["pooled"].items():
        # The end-of-turn forward follows the last content token, so the
        # clamped ring read (the stack's last row) is a different position.
        assert stack[layer].shape[0] > agg + 1
        assert not torch.equal(stack[layer][-1], stack[layer][agg])
        assert torch.equal(pooled, stack[layer][agg])
        assert torch.equal(pooled.cpu(), result.hidden_states[layer][agg])


# ---------------------------------------------------------------------------
# ReadPlane.begin over the real instruments' plans
# ---------------------------------------------------------------------------


def _probe_session(
    *,
    monitor: Any,
    lens_live: Any = None,
    lens_probes: dict[str, Any] | None = None,
    sae_live: Any = None,
    sae_probes: dict[str, Any] | None = None,
    sae_layer: int | None = None,
) -> Any:
    """A narrow session stub with real lens/SAE instruments, and a real read
    plane over them whose capture is ``stub._capture``."""
    from drowse.session.session import DrowseSession
    from tests.support.sae import make_resident
    from tests.support.stubs import attach_stub_instruments, wire_stub_read_plane

    session: Any = DrowseSession.__new__(DrowseSession)
    session._layers = [object()] * 8
    session._monitor = monitor
    attach_stub_instruments(session)
    lens, sae = session._lens_instrument, session._sae_instrument
    lens.live = lens_live
    lens.probes = dict(lens_probes or {})
    lens.active_for_generation = True
    sae.live = sae_live
    sae.probes = dict(sae_probes or {})
    if sae_layer is not None:
        make_resident(sae.runtime, layer=sae_layer)
    # The stub's in-memory lens stands in for the store refresh: a lens with
    # the declared live + probe layer union (production keeps every probe's
    # layer list equal to the resident lens's ``source_layers``).
    layers = {int(x) for x in (lens_live or {}).get("layers", [])}
    layers |= {int(x) for spec in lens.probes.values() for x in spec.get("layers", [])}
    resident = (
        SimpleNamespace(source_layers=sorted(layers))
        if lens_live is not None or lens.probes else None
    )
    lens.resident = lambda: resident
    wire_stub_read_plane(session)
    return session


class _RecordingCapture:
    """Records what the read plane attaches and arms."""

    def __init__(self) -> None:
        self.attached: list[int] | None = None
        self.depth: int | None = None
        self.sink: Any = None
        self.tail_layers: Any = None

    def clear(self) -> None:
        pass

    def attach(self, _layers: Any, layer_idxs: list[int]) -> None:
        self.attached = list(layer_idxs)

    def attach_persistent(self, layer_idxs: list[int], buffers: Any) -> None:
        raise AssertionError("no persistent buffers were offered")

    def set_retention(
        self, *, depth: int, step_sink: Any = None, tail_layers: Any = None,
    ) -> None:
        self.depth = depth
        self.sink = step_sink
        self.tail_layers = tail_layers


class _QuietMonitor:
    """No monitor probes."""

    probe_names: list[str] = []

    def probe_layers(self, _names: set[str] | None = None) -> set[int]:
        return set()

    def enable_curved_warm(self, _enabled: bool) -> None:
        pass


_NO_FINAL: dict[str, Any] = dict(
    need_per_token=False, per_token_full_consumer=False, final_probe_aggregate=False,
)


def _begin(session: Any, capture: Any, demand: ReadDemand, **kwargs: Any) -> bool:
    session._capture = capture
    return session._read_plane.begin(demand, **kwargs)


def test_lens_live_without_final_aggregate_keeps_one_latest_slice() -> None:
    session = _probe_session(monitor=_QuietMonitor(), lens_live={"layers": [2, 4]})
    capture = _RecordingCapture()
    assert _begin(session, capture, ReadDemand(**_NO_FINAL)) is True
    assert capture.attached == [2, 4]
    assert capture.depth == 1


def test_dormant_lens_probe_without_final_aggregate_attaches_nothing() -> None:
    session = _probe_session(
        monitor=_QuietMonitor(), lens_probes={"jlens/g": {"layers": [3]}},
    )
    capture = _RecordingCapture()
    assert _begin(session, capture, ReadDemand(**_NO_FINAL)) is False
    assert capture.attached is None
    assert session._read_plane.state.final_probe_aggregate is False


def test_lens_gate_without_final_aggregate_attaches_the_gated_band() -> None:
    session = _probe_session(
        monitor=_QuietMonitor(),
        lens_probes={"jlens/g": {"layers": [3]}, "jlens/h": {"layers": [6]}},
    )
    capture = _RecordingCapture()
    demand = ReadDemand(**_NO_FINAL, lens_gating_probe_keys=frozenset({"jlens/g"}))
    assert _begin(session, capture, demand) is True
    # Production bands are uniform (every probe's registry list equals the
    # resident lens's source_layers), so the gate keeps that band alive.
    assert capture.attached == [3, 6]
    assert capture.depth == 1


def test_sae_live_without_final_aggregate_keeps_one_latest_slice() -> None:
    session = _probe_session(monitor=_QuietMonitor(), sae_live={"layer": 5})
    capture = _RecordingCapture()
    _begin(session, capture, ReadDemand(**_NO_FINAL))
    assert capture.attached == [5]
    assert capture.depth == 1


def test_dormant_sae_probe_without_final_aggregate_attaches_nothing() -> None:
    session = _probe_session(
        monitor=_QuietMonitor(), sae_probes={"sae/0": {"feature_id": 0}}, sae_layer=5,
    )
    capture = _RecordingCapture()
    assert _begin(session, capture, ReadDemand(**_NO_FINAL)) is False
    assert capture.attached is None
    assert session._read_plane.state.final_probe_aggregate is False


def test_sae_gate_without_final_aggregate_attaches_the_sae_layer() -> None:
    session = _probe_session(
        monitor=_QuietMonitor(), sae_probes={"sae/0": {"feature_id": 0}}, sae_layer=5,
    )
    capture = _RecordingCapture()
    demand = ReadDemand(**_NO_FINAL, sae_gating_probe_keys=frozenset({"sae/0"}))
    assert _begin(session, capture, demand) is True
    assert capture.attached == [5]
    assert capture.depth == 1
    # The armed retention is a ring, so the recorded mode says so.
    assert session._read_plane.state.mode is CaptureMode.AGGREGATE_ONLY


def test_dormant_monitor_probes_attach_nothing() -> None:
    class Monitor:
        probe_names = ["monitor"]

        def probe_layers(self, _names: set[str] | None = None) -> set[int]:
            raise AssertionError("dormant monitor probes should not widen capture")

    class Capture:
        def attach(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("dormant monitor probes should not attach capture")

    session = _probe_session(monitor=Monitor())
    assert _begin(session, Capture(), ReadDemand(**_NO_FINAL)) is False
    assert session._read_plane.state.final_probe_aggregate is False


def test_begin_takes_per_token_full_consumer_from_the_demand() -> None:
    """``begin`` must not re-derive the demand's predicates: a gate-only
    generation has ``need_per_token`` with no full-roster consumer."""
    from drowse.readout.types import InstrumentPrep
    from tests.support.read_plane import make_read_plane

    class Recording:
        def __init__(self, family: str) -> None:
            self.family = family
            self.requests: list[Any] = []

        def prepare(self, request: Any) -> Any:
            self.requests.append(request)
            return InstrumentPrep(family=self.family, request=request)

        def plan(self, prep: Any) -> Any:
            return InstrumentPlan(family=self.family, prep_token=prep.token)

        def bind(self, _plan: Any, _prep: Any) -> None:
            return None

        def close_run(self) -> None:
            pass

    geometry, lens, sae = Recording("geometry"), Recording("lens"), Recording("sae")
    plane = make_read_plane(
        capture=SimpleNamespace(clear=lambda: None),
        monitor=SimpleNamespace(probe_names=[]),
        geometry=geometry, lens=lens, sae=sae, layers=[object()] * 4,
    )
    demand = ReadDemand(need_per_token=True, per_token_full_consumer=False)
    assert plane.begin(demand) is False
    for instrument in (geometry, lens, sae):
        assert [r.per_token_consumers for r in instrument.requests] == [False]


def test_monitor_final_aggregate_arms_a_cold_aggregate_ring() -> None:
    class Monitor:
        probe_names = ["monitor"]

        def __init__(self) -> None:
            self.layer_query: Any = "unset"
            self.warm: bool | None = None

        def probe_layers(self, names: set[str] | None = None) -> set[int]:
            self.layer_query = names
            return {2, 4}

        def enable_curved_warm(self, enabled: bool) -> None:
            self.warm = enabled

    monitor = Monitor()
    session = _probe_session(monitor=monitor)
    capture = _RecordingCapture()
    demand = ReadDemand(
        need_per_token=False, per_token_full_consumer=False, final_probe_aggregate=True,
    )
    assert _begin(session, capture, demand) is True
    assert monitor.layer_query is None
    assert capture.attached == [2, 4]
    assert capture.depth is not None and capture.depth > 1
    assert capture.sink is None
    assert session._read_plane.state.mode is CaptureMode.AGGREGATE_ONLY
    assert monitor.warm is False


class _GateMonitor:
    probe_names = ["gate", "other"]

    def __init__(self) -> None:
        self.layer_query: Any = "unset"
        self.plan_calls: list[tuple[set[str], Any]] = []
        self.score_plans: list[Any] = []

    def probe_layers(self, names: set[str] | None = None) -> set[int]:
        self.layer_query = names
        return {2} if names == {"gate"} else {2, 5}

    def reset_curved_feet(self) -> None:
        pass

    def enable_curved_warm(self, _enabled: bool) -> None:
        pass

    def plan_gate_scalars(self, gate_keys: set[str], *, probe_names: Any = None) -> Any:
        self.plan_calls.append((
            set(gate_keys), set(probe_names) if probe_names is not None else None,
        ))
        return ("plan",)

    def score_planned_gate_scalars(self, _latest: Any, plan: Any) -> dict[str, float]:
        self.score_plans.append(plan)
        return {"gate": 0.25}

    def score_gate_scalars(self, *_args: Any, **_kwargs: Any) -> dict[str, float]:
        raise AssertionError("GATING_SUBSET should reuse the planned scalars")


_GATE_ONLY: dict[str, Any] = dict(
    need_per_token=True, per_token_full_consumer=False,
    gating_only_probes=frozenset({"gate"}), gating_probe_keys=frozenset({"gate@x"}),
    final_probe_aggregate=False,
)


def test_a_gate_only_control_call_narrows_capture_to_the_gated_probes() -> None:
    monitor = _GateMonitor()
    session = _probe_session(monitor=monitor)
    capture = _RecordingCapture()
    _begin(session, capture, ReadDemand(**_GATE_ONLY))
    assert monitor.layer_query == {"gate"}
    assert capture.attached == [2]
    assert capture.depth == 1 and capture.sink is not None
    state = session._read_plane.state
    assert state.mode is CaptureMode.GATING_SUBSET
    assert state.final_probe_aggregate is False


def test_the_gated_subset_sink_reuses_its_preplanned_gate_scalars() -> None:
    from drowse.readout.types import parse_gate_ref

    monitor = _GateMonitor()
    session = _probe_session(monitor=monitor)
    capture = _RecordingCapture()
    _begin(session, capture, ReadDemand(**_GATE_ONLY))
    assert monitor.plan_calls == [({"gate@x"}, {"gate"})]
    capture.sink(0, {2: torch.ones(4)})
    assert monitor.score_plans == [("plan",)]
    # The gate callback reads the sink's row for this forward.
    plane = session._read_plane
    plane._gate_refs = lambda: [("gate@x", parse_gate_ref("gate@x"))]
    assert plane.gate_score_callback()(0) == {"gate": 0.25}
    assert monitor.score_plans == [("plan",)]


def test_full_incremental_capture_deep_rings_only_the_readout_aggregate_layers() -> None:
    class Monitor:
        probe_names = ["monitor"]

        def probe_layers(self, names: set[str] | None = None) -> set[int]:
            assert names is None
            return {0, 1}

        def reset_curved_feet(self) -> None:
            pass

        def enable_curved_warm(self, _enabled: bool) -> None:
            pass

    session = _probe_session(
        monitor=Monitor(),
        lens_probes={"jlens/g": {"layers": [3]}},
        sae_probes={"sae/5": {"feature_id": 5}}, sae_layer=5,
    )
    capture = _RecordingCapture()
    demand = ReadDemand(need_per_token=True, final_probe_aggregate=True)
    assert _begin(session, capture, demand) is True
    assert capture.attached == [0, 1, 3, 5]
    assert capture.depth is not None and capture.depth > 1
    assert set(capture.tail_layers) == {3, 5}
    assert session._read_plane.state.mode is CaptureMode.INCREMENTAL


def test_persistent_buffers_carry_a_live_lens_compile_clean() -> None:
    session = _probe_session(monitor=_QuietMonitor(), lens_live={"layers": [1, 3]})
    given = {idx: torch.zeros(8) for idx in range(4)}
    persistent: list[int] = []

    class Capture(_RecordingCapture):
        def attach(self, _layers: Any, layer_idxs: list[int]) -> None:
            raise AssertionError("compile-clean capture must not install hooks")

        def attach_persistent(self, layer_idxs: list[int], buffers: Any) -> None:
            assert buffers is given
            persistent.extend(layer_idxs)

    assert _begin(session, Capture(), ReadDemand(), persistent_buffers=given) is True
    assert persistent == [1, 3]
    assert session._read_plane.state.persistent is True
