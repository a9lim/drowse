"""Unified probe wiring: session probe attach/detach and the read plane.

Session-level behavior (``add_probe`` / ``remove_probe``, the geometry-state
lock around roster mutation, the registry's promotion and eviction walks)
runs on a real tiny CPU session whose registry serves a toy curved manifold.
Read-plane behavior (capture widening, sink-row reuse, gate scalars, the
pooled finalize slice) runs on a directly constructed read plane over a real
Monitor and the three real instruments (``_ProbeRig``).
"""
from __future__ import annotations

import threading
import types
from typing import Any, cast

import pytest
import torch

from drowse.geometry.domains import BoxAxis, BoxDomain
from drowse.geometry.subspace import LayerSubspace
from drowse.geometry.manifold import Manifold
from drowse.geometry.subspace import fit_curved_subspace
from drowse.readout.monitor import Monitor
from drowse.generation.results import (
    GenerationResult,
    TokenEvent,
)
from drowse.readout.types import ProbeReading
from drowse.generation.decode import GenerationState
from drowse.readout.planner import CaptureMode, CaptureState, ReadDemand
from tests._whitener import isotropic_whitener
from tests.support.read_plane import ReadRig, gate_refs


def _toy_manifold(*, dim: int = 8, n_layers: int = 2) -> Manifold:
    """1-D BoxDomain manifold with 3 nodes — same shape as the
    test_manifold_monitor helper, copied to keep test files independent."""
    torch.manual_seed(0)
    domain = BoxDomain([BoxAxis("u", periodic=False, lo=-1.0, hi=1.0)])
    coords = torch.tensor([[-1.0], [0.0], [1.0]])
    layers: dict[int, LayerSubspace] = {}
    share: dict[int, float] = {}
    e1 = torch.zeros(dim)
    e1[0] = 1.0
    e2 = torch.zeros(dim)
    e2[1] = 1.0
    for layer_idx in range(n_layers):
        scale = 1.0 + 0.5 * layer_idx
        centroids = torch.stack([
            -scale * e1,
            torch.zeros(dim),
            scale * e1,
        ]) + 0.01 * torch.stack([-e2, torch.zeros(dim), e2])
        fit = fit_curved_subspace(centroids, domain.embed(coords))
        sub, ev_ratio = fit.subspace, fit.explained_variance
        assert sub.node_params is not None
        sub.sigma_rbf_weights = torch.zeros((len(coords), 1))
        sub.sigma_poly_coeffs = torch.zeros((sub.node_params.shape[1] + 1, 1))
        sub.sigma_poly_coeffs[0, 0] = -20.0
        layers[layer_idx] = sub
        share[layer_idx] = ev_ratio
    return Manifold(
        name="toy",
        domain=domain,
        node_labels=["a", "b", "c"],
        node_coords=coords,
        layers=layers,
        node_roles=[None, None, None],
        node_kinds=[None, None, None],
        mahalanobis_share=share,
        origin={layer: torch.zeros(1) for layer in layers},
    )


class _ProbeRig:
    """A read plane over a real Monitor (isotropic whitener on layers 0-3,
    dim 8) and real instruments; the capture is a namespace each test
    fills in, and :meth:`gate` sets the active steering gates."""

    def __init__(self) -> None:
        self.monitor = Monitor(whitener=isotropic_whitener([0, 1, 2, 3], 8))
        self._gates: Any = lambda: []
        self.read = ReadRig(
            monitor=self.monitor,
            capture=types.SimpleNamespace(_per_layer={}),
            gate_refs=lambda: self._gates(),
        )
        self.plane = self.read.plane
        self.geometry = self.read.geometry
        self.lens = self.read.lens
        self.sae = self.read.sae

    @property
    def capture(self) -> Any:
        return self.read.capture.target

    @capture.setter
    def capture(self, value: Any) -> None:
        self.read.capture.target = value

    def attach(self, manifold: Manifold, name: str = "toy") -> None:
        self.monitor.add_probe(name, manifold, top_n=3)

    def gate(self, expr: str) -> None:
        self._gates = gate_refs(expr)


@pytest.fixture
def serve(tiny_session: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Serve toy manifolds through ``tiny_session``'s registry loader (no
    disk): one manifold for every key, or a ``{key: manifold}`` map."""

    def _serve(manifolds: Any) -> None:
        monkeypatch.setattr(
            tiny_session.registry, "ensure_manifold_loaded",
            lambda key: manifolds[key] if isinstance(manifolds, dict) else manifolds,
        )

    return _serve


def _session_toy(session: Any) -> Manifold:
    """The toy manifold at the session's hidden width."""
    return _toy_manifold(dim=int(session.model_info["hidden_dim"]))


# ==================================================== add / remove probe ===

def test_add_probe_registers_via_ensure_loaded(tiny_session: Any, serve: Any) -> None:
    """``add_probe`` rides ``ensure_manifold_loaded`` and lands the
    artifact on the unified ``Monitor``."""
    serve(_session_toy(tiny_session))
    name = tiny_session.add_probe("toy")
    assert name == "toy"
    assert "toy" in tiny_session.monitor.probe_names


def test_add_probe_as_name_override(tiny_session: Any, serve: Any) -> None:
    serve(_session_toy(tiny_session))
    name = tiny_session.add_probe("toy", as_name="affect")
    assert name == "affect"
    assert "affect" in tiny_session.monitor.probe_names


def test_remove_probe(tiny_session: Any, serve: Any) -> None:
    serve(_session_toy(tiny_session))
    tiny_session.add_probe("toy")
    tiny_session.remove_probe("toy")
    assert tiny_session.monitor.probe_names == []


def test_add_probe_invalidates_prefix_cache(tiny_session: Any, serve: Any) -> None:
    serve(_session_toy(tiny_session))
    tiny_session.cache_prefix([{"role": "user", "content": "a shared prefix"}])
    assert tiny_session._engine.prefix_cache
    tiny_session.add_probe("toy")
    assert not tiny_session._engine.prefix_cache


# ==================================================== capture widening ===

def test_read_plane_begin_widens_capture_to_manifold_layers():
    """The read plane's ``begin`` must widen the capture-layer set to the
    union of every attached probe's fit layers."""
    rig = _ProbeRig()
    m = _toy_manifold(n_layers=3)
    rig.attach(m)

    attached_layers: list[int] = []
    sinks: list[Any] = []

    def _attach(layers: Any, layer_indices: list[int]) -> None:
        attached_layers.extend(layer_indices)

    rig.capture.attach = _attach
    rig.capture.clear = lambda: None
    rig.capture.set_retention = (
        lambda *, depth, step_sink=None, tail_layers=None: sinks.append(step_sink)
    )

    ok = rig.plane.begin(ReadDemand())
    assert ok
    # Manifold covers layers 0, 1, 2 — capture must attach to all of them.
    assert set(attached_layers) == {0, 1, 2}
    assert rig.plane.state.mode is CaptureMode.INCREMENTAL
    assert len(sinks) == 1 and sinks[0] is not None


def test_read_plane_begin_without_probes_attaches_nothing():
    """No probes attached → ``begin`` attaches nothing."""
    rig = _ProbeRig()
    rig.capture.attach = lambda *args, **kw: None
    rig.capture.clear = lambda: None
    assert rig.plane.begin(ReadDemand()) is False


def test_read_plane_begin_rides_persistent_capture_for_live_lens():
    """Live J-lens layers ride compile-clean persistent capture buffers."""
    rig = _ProbeRig()
    # Live-lens state lives on the instrument; ``begin`` reads it through
    # the lens family's declared plan.
    rig.lens.live = {"layers": [1, 3]}
    buffers = {idx: torch.zeros(8) for idx in range(4)}

    persistent_layers: list[int] = []
    transient_layers: list[int] = []
    tail_depths: list[int] = []

    def _attach_persistent(
        layer_indices: list[int], given: dict[int, torch.Tensor],
    ) -> None:
        assert given is buffers
        persistent_layers.extend(layer_indices)

    def _attach(_layers: Any, layer_indices: list[int]) -> None:
        transient_layers.extend(layer_indices)

    rig.capture.attach_persistent = _attach_persistent
    cast(Any, rig.capture).attach = _attach
    rig.capture.clear = lambda: None
    rig.capture.set_retention = (
        lambda *, depth, step_sink=None, tail_layers=None: tail_depths.append(depth)
    )

    ok = rig.plane.begin(ReadDemand(), persistent_buffers=buffers)

    assert ok is True
    assert set(persistent_layers) == {1, 3}
    assert transient_layers == []
    assert tail_depths
    assert rig.plane.state.persistent is True


def test_read_plane_begin_ignores_live_lens_without_consumer():
    """A globally enabled live lens should not widen capture for a generation
    that cannot surface ``TokenEvent.lens_readout``."""
    rig = _ProbeRig()
    rig.lens.live = {"layers": [1, 3]}
    rig.capture.attach = lambda *args, **kw: None
    rig.capture.clear = lambda: None

    ok = rig.plane.begin(ReadDemand(live_lens_active=False))

    assert ok is False


def test_geometry_detach_rejects_while_generation_holds_the_lock(tiny_session: Any, serve: Any) -> None:
    """Geometry has no per-generation roster snapshot (Monitor scoring
    walks the live dict), so a detach racing a generation must reject with
    retry-shortly semantics instead of mutating the roster mid-flight —
    the same exclusive section ``attach`` takes."""
    import threading as _threading

    from drowse.errors import ConcurrentExtractionError

    session = tiny_session
    m = _session_toy(session)
    serve(m)
    session.add_probe("toy")
    assert "toy" in session.monitor.probe_names

    # Simulate an in-flight generation: another thread holds the gen lock.
    held = _threading.Event()
    release = _threading.Event()

    def _hold() -> None:
        with session.gen_lock:
            held.set()
            release.wait(timeout=5)

    holder = _threading.Thread(target=_hold)
    holder.start()
    try:
        assert held.wait(timeout=5)
        with pytest.raises(ConcurrentExtractionError, match="retry shortly"):
            session.remove_probe("toy")
        # The roster is untouched — the reject protected the in-flight read.
        assert "toy" in session.monitor.probe_names
    finally:
        release.set()
        holder.join(timeout=5)

    # Idle again: the detach applies normally.
    session.remove_probe("toy")
    assert "toy" not in session.monitor.probe_names


# ================================================== geometry state lock ===

def test_geometry_state_lock_serializes_detach_and_reads(tiny_session: Any, serve: Any) -> None:
    """The geometry-state boundary (the Monitor sibling of the lens
    round-5 fix): the session's ``remove_probe`` geometry branch and the
    coherent read surfaces (``specs``) hold the instrument's state lock —
    an idle detach can no longer land inside a concurrent registry
    iteration or tear it."""
    session = tiny_session
    m = _session_toy(session)
    serve(m)
    session.add_probe("toy")
    inst = session.geometry

    detach_entered = threading.Event()
    read_entered = threading.Event()
    read_out: list[Any] = []

    def _detach() -> None:
        detach_entered.set()
        session.remove_probe("toy")

    def _read_specs() -> None:
        read_entered.set()
        read_out.append(inst.specs())

    with inst.state_lock:
        detacher = threading.Thread(target=_detach)
        reader = threading.Thread(target=_read_specs)
        detacher.start()
        reader.start()
        assert detach_entered.wait(timeout=5.0)
        assert read_entered.wait(timeout=5.0)
        detacher.join(timeout=0.2)
        reader.join(timeout=0.2)
        assert detacher.is_alive()  # both blocked on the geometry-state lock
        assert reader.is_alive()
        assert "toy" in session.monitor.probe_names  # nothing mutated mid-hold
    detacher.join(timeout=5.0)
    reader.join(timeout=5.0)
    assert not detacher.is_alive() and not reader.is_alive()
    assert "toy" not in session.monitor.probe_names
    assert read_out and isinstance(read_out[0], dict)


def test_geometry_idle_run_reads_hold_the_state_lock(tiny_session: Any, serve: Any) -> None:
    """Idle-passthrough reads (an unbound ``GeometryRun.observe``) pair
    with the roster under the state lock — the round-6 idle-coherence
    contract, geometry edition.  The bound per-token path stays
    lock-free (mid-generation mutation is excluded by the detach
    reject contract, not by this lock)."""
    session = tiny_session
    m = _session_toy(session)
    serve(m)
    session.add_probe("toy")
    inst = session.geometry
    assert inst.current_run.bound is False
    hidden = {
        layer_idx: sub.mean + sub.basis[0]
        for layer_idx, sub in m.layers.items()
    }

    entered = threading.Event()
    out: list[Any] = []

    def _idle_read() -> None:
        entered.set()
        out.append(inst.current_run.observe(0, hidden))

    with inst.state_lock:
        reader = threading.Thread(target=_idle_read)
        reader.start()
        assert entered.wait(timeout=5.0)
        reader.join(timeout=0.2)
        assert reader.is_alive()  # blocked on the geometry-state lock
    reader.join(timeout=5.0)
    assert not reader.is_alive()
    assert out and "toy" in out[0]


def test_geometry_roster_reads_do_not_tear_under_concurrent_detach(tiny_session: Any, serve: Any) -> None:
    """The tear repro: ``plan()``'s roster walk (``probe_names`` +
    ``probe_layers``) iterates live ``Monitor._probes`` state and raised
    ``RuntimeError: dictionary changed size during iteration`` under a
    concurrent idle attach/detach before the state lock existed (the
    exact class the lens round-5 fix closed).  Hammer both sides; any
    exception on either thread fails."""
    import time as _time

    from drowse.readout.types import ReadRequest

    session = tiny_session
    manifolds = {f"toy{i}": _session_toy(session) for i in range(6)}
    serve(manifolds)
    for name in manifolds:
        session.add_probe(name)
    inst = session.geometry

    stop = threading.Event()
    errors: list[BaseException] = []

    def _mutate() -> None:
        try:
            i = 0
            while not stop.is_set():
                name = f"toy{i % 6}"
                session.remove_probe(name)
                session.add_probe(name)
                i += 1
        except BaseException as exc:  # noqa: BLE001 — the assertion payload
            errors.append(exc)

    def _read() -> None:
        try:
            while not stop.is_set():
                prep = inst.prepare(ReadRequest(final_aggregate=True))
                inst.plan(prep)
                inst.specs()
                inst.manifolds()
                _ = inst.names
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=_mutate),
        threading.Thread(target=_read),
    ]
    for t in threads:
        t.start()
    _time.sleep(0.5)
    stop.set()
    for t in threads:
        t.join(timeout=10.0)
    assert not any(t.is_alive() for t in threads)
    assert not errors, errors


class _RecordingLock:
    """RLock shim recording hold depth (every production use is ``with``)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.held = 0

    def __enter__(self) -> "_RecordingLock":
        self._lock.acquire()
        self.held += 1
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.held -= 1
        self._lock.release()
        return False


def _record_roster_mutations(
    session: Any, shim: _RecordingLock,
) -> list[tuple[str, bool]]:
    """Wrap the stub monitor's roster mutators to record whether the
    geometry-state shim was held at each call."""
    monitor = session.monitor
    real_remove, real_add = monitor.remove_probe, monitor.add_probe
    held_at_call: list[tuple[str, bool]] = []

    def _rec_remove(name: str) -> None:
        held_at_call.append(("remove", shim.held > 0))
        real_remove(name)

    def _rec_add(name: str, manifold: Any, *, top_n: int = 3) -> None:
        held_at_call.append(("add", shim.held > 0))
        real_add(name, manifold, top_n=top_n)

    monitor.remove_probe = _rec_remove
    monitor.add_probe = _rec_add
    return held_at_call


def test_manifold_promotion_walk_holds_geometry_state_lock(tiny_session: Any, serve: Any) -> None:
    """A fit's probe-promotion walk (the registry's swap listener) mutates
    the Monitor roster outside the instrument's attach/detach — the
    remove+add pair must land under the geometry-state lock so a
    coherent reader can't observe a half-applied promotion."""
    session = tiny_session
    m_old = _session_toy(session)
    serve(m_old)
    session.add_probe("toy")
    session.registry.put_manifold("toy", m_old)
    inst = session.geometry
    shim = _RecordingLock()
    inst.state_lock = shim  # type: ignore[assignment]
    held_at_call = _record_roster_mutations(session, shim)

    m_new = _session_toy(session)
    session.registry.adopt_fitted("default/toy", m_new)

    assert ("remove", True) in held_at_call
    assert ("add", True) in held_at_call
    assert all(held for _op, held in held_at_call)


def test_failed_override_eviction_holds_geometry_state_lock(tiny_session: Any, serve: Any) -> None:
    """An artifact eviction's probe removal is the same out-of-instrument
    roster mutation — it must hold the geometry-state lock too."""
    session = tiny_session
    m_old = _session_toy(session)
    serve(m_old)
    session.add_probe("toy")
    session.registry.put_manifold("toy", m_old)
    inst = session.geometry
    shim = _RecordingLock()
    inst.state_lock = shim  # type: ignore[assignment]
    held_at_call = _record_roster_mutations(session, shim)

    session.registry.evict_artifact("default", "toy", feature="raw")

    assert ("remove", True) in held_at_call
    assert "toy" not in session.monitor.probe_names


# ================================== per-forward reuse of the sink row ===

def _arm_capture(rig: _ProbeRig) -> dict[str, Any]:
    """Minimal capture wiring so the read plane's ``begin`` runs on the
    rig, capturing whichever step sink the plan arms."""
    holder: dict[str, Any] = {}
    rig.capture.attach = lambda *_a, **_k: None
    rig.capture.clear = lambda: None
    rig.capture.set_retention = (
        lambda *, depth, step_sink=None, tail_layers=None: (
            holder.__setitem__("sink", step_sink)
        )
    )
    return holder


def test_gate_callback_reuses_the_full_incremental_sink_row():
    """The composer's FULL-incremental branch reads this forward's sink row —
    no rescore (the monitor is poisoned after the sink to prove it)."""
    rig = _ProbeRig()
    m = _toy_manifold()
    rig.attach(m)
    holder = _arm_capture(rig)
    ok = rig.plane.begin(ReadDemand(need_per_token=True))
    assert ok is True

    latest = {
        layer_idx: sub.mean + sub.basis[0]
        for layer_idx, sub in m.layers.items()
    }
    holder["sink"](0, latest)

    def _poisoned(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise AssertionError(
            "the gate callback must reuse the sink row, not rescore"
        )

    rig.monitor.score_single_token = _poisoned  # type: ignore[method-assign]
    rig.gate("0.3 a@when:toy>0 + 0.3 b@when:toy:fraction>0")
    cb = rig.plane.gate_score_callback()
    flat = cb(0)
    assert "toy" in flat and "toy:fraction" in flat
    rig.geometry.close_run()


def test_token_tap_reuses_the_full_incremental_sink_row():
    """The token tap's FULL-incremental branch reads the sink row too — the
    event's geometry readings ARE the sink's row."""
    from drowse.generation.token_callback import (
        TokenConsumer,
        TokenConsumerOptions,
    )
    from drowse.session.engine import ConsumerPlan, TokenTap

    rig = _ProbeRig()
    m = _toy_manifold()
    rig.attach(m)
    holder = _arm_capture(rig)
    ok = rig.plane.begin(ReadDemand(need_per_token=True))
    assert ok is True

    latest = {
        layer_idx: sub.mean + sub.basis[0]
        for layer_idx, sub in m.layers.items()
    }
    holder["sink"](1, latest)

    def _poisoned(*_a: Any, **_k: Any) -> dict[str, Any]:
        raise AssertionError(
            "the token payload must reuse the sink row, not rescore"
        )

    rig.monitor.score_single_token = _poisoned  # type: ignore[method-assign]
    events: list[Any] = []
    tap = TokenTap(
        cast(Any, types.SimpleNamespace(geometry=rig.geometry, lens=rig.lens, sae=rig.sae)),
        reads=rig.plane,
        state=GenerationState(),
        consumer=TokenConsumer(
            events.append,
            TokenConsumerOptions(live_scores=True, probe_readings=True),
        ),
        plan=ConsumerPlan(
            live_scores_on=True, has_monitor_probes=True,
            wants_live_token_scores=True, persists_layer_scores=False,
            persists_subspace_coords=False, persists_probe_row=False,
            lens=False, sae=False, perplexity_requested=False, need_tap=True,
        ),
        assistant_node_id=None,
        collect_logprobs=False,
        readout_top_k=8,
        recipe_steering=None,
    )
    tap("x", False, 1, None, None, None, 1)
    row = rig.plane._rows[-1]
    assert events[0].probe_readings == row
    assert events[0].probe_readings["toy"] is row["toy"]
    rig.geometry.close_run()


def test_pooled_geometry_aggregate_routes_through_the_run():
    """The aggregate-only finalize read reaches the roster through
    ``current_run.observe_aggregate`` with the one pooled slice — the
    round-2 bypass site, so a run spy pins the route."""
    rig = _ProbeRig()
    m = _toy_manifold()
    rig.attach(m)
    pooled = {
        layer_idx: sub.mean + sub.basis[0]
        for layer_idx, sub in m.layers.items()
    }
    rig.capture.tail_slice_at = lambda _idx: pooled
    run = rig.geometry.current_run
    seen: list[Any] = []
    orig = run.observe_aggregate

    def _spy(p: Any) -> Any:
        seen.append(p)
        return orig(p)

    run.observe_aggregate = _spy  # type: ignore[method-assign]
    plane = rig.plane
    plane.state = CaptureState(mode=CaptureMode.AGGREGATE_ONLY)
    state = GenerationState()
    state.response_aggregate_index = 2
    reads = plane.finalize(
        [1, 2, 3], state=state, return_hidden=False, return_probe_readings=True,
    )
    assert seen and seen[0] is pooled
    assert "toy" in reads.geometry


def test_geometry_run_observe_aggregate_matches_live_read():
    """The finalize aggregate routed through ``GeometryRun.observe_
    aggregate`` is bit-identical to a live ``score_single_token`` read at
    the pooled slice — the protocol claim, pinned on a CURVED probe (the
    foot-solve path, the hard case)."""
    rig = _ProbeRig()
    m = _toy_manifold()
    rig.attach(m)
    pooled = {
        layer_idx: sub.mean + sub.basis[0]
        for layer_idx, sub in m.layers.items()
    }

    via_run = rig.geometry.current_run.observe_aggregate(
        pooled,
    )
    live = rig.monitor.score_single_token(pooled)

    assert set(via_run) == set(live) == {"toy"}
    assert via_run["toy"].coords == live["toy"].coords
    assert via_run["toy"].fraction == live["toy"].fraction
    assert via_run["toy"].residual == live["toy"].residual
    assert via_run["toy"].nearest == live["toy"].nearest


def test_read_plane_pooled_slice_is_one_position_for_every_family():
    """All three families finalize through the read plane's one pooled
    slice, and it lands on the aggregate *forward index* in FULL retention
    too.

    FULL retention never advances the capture's forward counter (only the
    bounded-ring modes do), so a ring lookup would clamp to the last forward —
    the trailing special — instead of the last content token.
    """
    rig = _ProbeRig()
    rows = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    rig.capture = cast(Any, types.SimpleNamespace(
        stacked=lambda: {0: rows},
        tail_slice_at=lambda _idx: {0: rows[-1]},
    ))
    plane = rig.plane
    state = GenerationState()
    state.response_aggregate_index = 1

    plane.state = CaptureState(mode=CaptureMode.FULL)
    full = plane.pooled_slice([1, 2, 3, 4], state)
    assert torch.equal(full[0], rows[1])

    plane.state = CaptureState(mode=CaptureMode.AGGREGATE_ONLY)
    ring = plane.pooled_slice([1, 2, 3, 4], state)
    assert torch.equal(ring[0], rows[-1])

    # A stop-sequence finish with no recorded endpoint has no position.
    state.response_aggregate_index = None
    state.finish_reason = "stop_sequence"
    state.response_text = "x"
    assert plane.pooled_slice([1, 2], state) == {}


# ===================================================== gating callback ===

def _latest_on_subspace(rig: _ProbeRig, m: Any) -> None:
    """Latest-per-layer captures placed inside the manifold's subspace."""
    latest = {L: sub.mean + sub.basis[0] for L, sub in m.layers.items()}
    rig.capture.latest_per_layer = lambda: dict(latest)


def test_gating_callback_emits_probe_scalars():
    """The composer-built gating callback scores the gated keys of attached
    probes from the latest captures — fraction, per-node distance (the
    virtual neutral candidate included) and coordinate axis 0."""
    rig = _ProbeRig()
    m = _toy_manifold()
    rig.attach(m)
    _latest_on_subspace(rig, m)
    rig.gate(
        "0.3 a@when:toy:fraction>0.1 + 0.3 b@when:toy@neutral>-9 "
        "+ 0.3 c@when:toy>0",
    )

    flat = rig.plane.gate_score_callback()(0)

    assert set(flat) == {"toy:fraction", "toy@neutral", "toy"}
    full = rig.monitor.flat_scalars(
        rig.monitor.score_single_token(
            rig.capture.latest_per_layer(),
        ),
    )
    for key, value in flat.items():
        assert value == pytest.approx(full[key], abs=1e-5)


def test_gate_on_unattached_probe_scores_nothing():
    """A gate naming a probe no family attached can never fire, so the
    callback must not fall back to scoring the whole roster every token."""
    rig = _ProbeRig()
    m = _toy_manifold()
    rig.attach(m)
    _latest_on_subspace(rig, m)
    rig.gate("0.3 a@when:ghost>0.1")

    def _poisoned(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("an unowned gate must not score the roster")

    rig.monitor.score_single_token = _poisoned  # type: ignore[method-assign]
    rig.monitor.score_planned_gate_scalars = _poisoned  # type: ignore[method-assign]

    assert rig.plane.gate_score_callback()(0) == {}


def test_gating_callback_empty_capture_returns_empty():
    rig = _ProbeRig()
    m = _toy_manifold()
    rig.attach(m)
    rig.capture.latest_per_layer = lambda: {}

    cb = rig.plane.gate_score_callback()
    assert cb(0) == {}


# ============================================================ aggregate ===

def test_score_aggregate_called_with_correct_captures():
    """Verify ``score_aggregate`` is called on the unified monitor when
    probes are attached and gen produced tokens."""
    from tests._whitener import isotropic_whitener
    m = _toy_manifold()
    mon = Monitor(whitener=isotropic_whitener(list(m.layers), 8))
    mon.add_probe("toy", m)

    # Build a captured stack at node 1's centroid (so coords come back
    # near 0.0).  Recompute the world-space node activation from the RBF
    # (the former ``node_values_world`` cache was removed — only the
    # reduced cache feeds scoring).
    embedded = m.domain.embed(m.domain.clamp_position(m.node_coords.float()))
    captured: dict[int, torch.Tensor] = {}
    for layer_idx, sub in m.layers.items():
        v_world = sub.eval_at(embedded)[1]
        captured[layer_idx] = v_world.unsqueeze(0).repeat(5, 1)

    agg = mon.score_aggregate(captured)
    assert isinstance(agg["toy"], ProbeReading)
    assert agg["toy"].coords[0] == pytest.approx(0.0, abs=0.1)


# ============================================ TokenEvent / GenerationResult ===

def test_token_event_carries_probe_readings_field():
    """``TokenEvent`` accepts ``probe_readings`` and defaults to None."""
    ev = TokenEvent(text="x", token_id=42, index=0)
    assert ev.probe_readings is None
    reading = ProbeReading(fraction=0.5, nearest=[("a", 0.1)])
    ev2 = TokenEvent(
        text="y", token_id=43, index=1,
        probe_readings={"toy": reading},
    )
    assert ev2.probe_readings == {"toy": reading}


def test_generation_result_carries_probe_readings_field():
    """``GenerationResult`` accepts ``probe_readings`` and ``to_dict``
    serializes it."""
    agg = ProbeReading(
        fraction=0.6,
        fraction_per_layer={0: 0.5, 1: 0.7},
        nearest=[("a", 0.2)],
        coords=(0.3,),
        coords_per_layer={0: (0.2,), 1: (0.4,)},
        residual=0.05,
        residual_per_layer={0: 0.04, 1: 0.06},
    )
    result = GenerationResult(
        text="hello", tokens=[1, 2], token_count=2,
        tok_per_sec=10.0, elapsed=0.2,
        probe_readings={"toy": agg},
    )
    d = result.to_dict()
    assert "probe_readings" in d
    assert "toy" in d["probe_readings"]
    assert d["probe_readings"]["toy"]["fraction"] == pytest.approx(0.6)
    assert d["probe_readings"]["toy"]["coords"] == [0.3]


def test_generation_result_default_probe_readings_is_empty_dict():
    result = GenerationResult(
        text="x", tokens=[], token_count=0,
        tok_per_sec=0.0, elapsed=0.0,
    )
    assert result.probe_readings == {}
    assert result.to_dict()["probe_readings"] == {}
