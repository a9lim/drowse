"""Unified probe wiring into DrowseSession.

CPU-only: a full session load needs a GPU + 8GB model download, so
these tests build a stand-in session by binding the real
``DrowseSession`` methods to a minimal stub.  The goal is to verify
wiring shape — capture widening, score-callback merging, finalize
populates ``GenerationResult.probe_readings``, stream populates
``TokenEvent.probe_readings`` — without paying the model-load cost.

4.0 collapse: the former ``TraitMonitor`` + ``ManifoldMonitor`` are one
``Monitor`` and the former ``add_manifold_probe`` / ``remove_manifold_probe``
session surface is the unified ``add_probe`` / ``remove_probe`` over the single
``session._monitor``.  A vector probe is the rank-1 case of the same subspace
readout, so there is no separate vector-monitor to wire — the toy manifold below
covers both shapes.
"""
from __future__ import annotations

import threading
import types
from typing import Any, cast
from unittest.mock import MagicMock

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
from drowse.session.session import DrowseSession
from tests.support.stubs import wire_stub_read_plane, wire_stub_steering
from tests.support.lens import stub_lens_instrument
from tests.support.engine import attach_engine, seed_prefix_cache


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


def _stub_session() -> Any:
    """Build a session-like object with the live methods + minimal state.

    Only the surface the unified-probe wiring touches needs to be real:
    ``_monitor`` (one :class:`Monitor`), the capture stub, the registry, the
    engine's prefix cache, the probe roster and the read plane.  Everything
    else stays as ``MagicMock`` defaults.
    """
    # Manifold reads are Mahalanobis-only now: the monitor needs a covering
    # whitener before ``add_probe`` builds the per-probe whitened factors.  An
    # isotropic one over the toy layers (dim 8, up to 3 layers) suffices.
    from tests._whitener import isotropic_whitener
    _whit = isotropic_whitener([0, 1, 2, 3], 8)

    session = MagicMock(spec=DrowseSession)
    session._monitor = Monitor(whitener=_whit)
    # ``add_probe`` holds the exclusive-GPU lock (``_gen_lock``) around its
    # device-touching factor build and clears the read-side analytics cache.
    session._gen_lock = threading.RLock()
    # Capture has a single ``_per_layer`` dict we read for streaming.
    session._capture = types.SimpleNamespace(
        _per_layer={},
    )
    session._lens_instrument = stub_lens_instrument(session)
    # All three families route through real instruments now: geometry for
    # the add_probe flow (exclusive section + whitener touch + resolve +
    # monitor.add_probe), lens/SAE because the read plane's ``begin``
    # consumes their ``plan()`` demand — a MagicMock plan would poison the
    # capture layer union.  Lens/SAE state (live config, probe registry) lives ON
    # the instruments; tests set ``session._lens_instrument.live`` etc.
    from drowse.readout.geometry import GeometryInstrument
    from tests.support.replay import unwired_replay
    from tests.support.sae import stub_sae_instrument
    session._geometry_instrument = GeometryInstrument(
        session, replay=unwired_replay,
    )
    session._sae_instrument = stub_sae_instrument(session)
    session._lens_instrument.generation_lens = None
    session._lens_instrument.generation_lens_active = False
    session._lens_instrument.live = None
    session._lens_instrument.probes = {}
    session._lens_instrument.step_stash = None
    session._lens_instrument.active_for_generation = True
    session._layers = []
    # The real engine: phase, prefix cache, compiled-routing flags.
    attach_engine(session)

    # Bind the real unified-probe methods we want to exercise.  ``add_probe``
    # rides ``_resolve_probe_manifold`` (which consults the registry's
    # profiles, then its manifold loader), so bind that too; tests serve
    # toy manifolds through the registry loader (``_serve``).
    session.add_probe = types.MethodType(DrowseSession.add_probe, session)
    session.remove_probe = types.MethodType(DrowseSession.remove_probe, session)
    session._resolve_probe_manifold = types.MethodType(
        DrowseSession._resolve_probe_manifold, session,
    )
    # The registry, steering, roster and read-plane collaborators are real;
    # the read plane owns the gating-score callback.
    session._steering = None
    session._device = torch.device("cpu")
    session._dtype = torch.float32
    session._model_info = {"model_id": "stub/model"}
    wire_stub_steering(session)
    wire_stub_read_plane(session)
    return session


def _serve(session: Any, manifolds: Any) -> None:
    """Serve toy manifolds through the registry loader (no disk)."""
    session._registry.ensure_manifold_loaded = (
        lambda key: manifolds[key] if isinstance(manifolds, dict) else manifolds
    )


# ==================================================== add / remove probe ===

def test_add_probe_registers_via_ensure_loaded():
    """``add_probe`` rides ``ensure_manifold_loaded`` and lands the
    artifact on the unified ``Monitor``."""
    session = _stub_session()
    m = _toy_manifold()

    # ``ensure_manifold_loaded`` is the resolution path; stub it to
    # populate ``_manifolds`` instead of hitting disk.
    _serve(session, m)

    name = session.add_probe("toy")
    assert name == "toy"
    assert "toy" in session._monitor.probe_names


def test_add_probe_as_name_override():
    session = _stub_session()
    m = _toy_manifold()
    _serve(session, m)

    name = session.add_probe("toy", as_name="affect")
    assert name == "affect"
    assert "affect" in session._monitor.probe_names


def test_remove_probe():
    session = _stub_session()
    m = _toy_manifold()
    _serve(session, m)

    session.add_probe("toy")
    session.remove_probe("toy")
    assert session._monitor.probe_names == []


def test_add_probe_invalidates_prefix_cache():
    session = _stub_session()
    m = _toy_manifold()
    _serve(session, m)
    seed_prefix_cache(session._engine)
    session.add_probe("toy")
    assert not session._engine.prefix_cache


# ==================================================== capture widening ===

def test_begin_capture_widens_to_manifold_layers():
    """The read plane's ``begin`` must widen the capture-layer set to the
    union of every attached probe's fit layers."""
    session = _stub_session()
    m = _toy_manifold(n_layers=3)
    _serve(session, m)
    session.add_probe("toy")

    attached_layers: list[int] = []
    sinks: list[Any] = []

    def _attach(layers: Any, layer_indices: list[int]) -> None:
        attached_layers.extend(layer_indices)

    session._capture.attach = _attach
    session._capture.clear = lambda: None
    session._capture.set_retention = (
        lambda *, depth, step_sink=None, tail_layers=None: sinks.append(step_sink)
    )

    ok = session._read_plane.begin(ReadDemand())
    assert ok
    # Manifold covers layers 0, 1, 2 — capture must attach to all of them.
    assert set(attached_layers) == {0, 1, 2}
    assert session._read_plane.state.mode is CaptureMode.INCREMENTAL
    assert len(sinks) == 1 and sinks[0] is not None


def test_begin_capture_no_probes_returns_false():
    """No probes attached → ``begin`` attaches nothing."""
    session = _stub_session()
    session._capture.attach = lambda *args, **kw: None
    session._capture.clear = lambda: None
    assert session._read_plane.begin(ReadDemand()) is False


def test_begin_capture_live_lens_uses_persistent_capture_when_available():
    """Live J-lens layers ride compile-clean persistent capture buffers."""
    session = _stub_session()
    # Live-lens state lives on the instrument; ``begin`` reads it through
    # the lens family's declared plan.
    session._lens_instrument.live = {"layers": [1, 3]}
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

    session._capture.attach_persistent = _attach_persistent
    cast(Any, session._capture).attach = _attach
    session._capture.clear = lambda: None
    session._capture.set_retention = (
        lambda *, depth, step_sink=None, tail_layers=None: tail_depths.append(depth)
    )

    ok = session._read_plane.begin(ReadDemand(), persistent_buffers=buffers)

    assert ok is True
    assert set(persistent_layers) == {1, 3}
    assert transient_layers == []
    assert tail_depths
    assert session._read_plane.state.persistent is True


def test_begin_capture_live_lens_ignored_without_consumer():
    """A globally enabled live lens should not widen capture for a generation
    that cannot surface ``TokenEvent.lens_readout``."""
    session = _stub_session()
    session._lens_instrument.live = {"layers": [1, 3]}
    session._capture.attach = lambda *args, **kw: None
    session._capture.clear = lambda: None

    ok = session._read_plane.begin(ReadDemand(live_lens_active=False))

    assert ok is False


def test_geometry_detach_rejects_while_generation_holds_the_lock():
    """Geometry has no per-generation roster snapshot (Monitor scoring
    walks the live dict), so a detach racing a generation must reject with
    retry-shortly semantics instead of mutating the roster mid-flight —
    the same exclusive section ``attach`` takes."""
    import threading as _threading

    from drowse.errors import ConcurrentExtractionError

    session = _stub_session()
    # The MagicMock default no-ops the exclusive section; this test is
    # about exactly that guard, so bind the real one (it needs only
    # ``_gen_lock``, which the fixture provides).
    session._model_exclusive = types.MethodType(
        DrowseSession._model_exclusive, session,
    )
    m = _toy_manifold()
    _serve(session, m)
    session.add_probe("toy")
    assert "toy" in session._monitor.probe_names

    # Simulate an in-flight generation: another thread holds the gen lock.
    held = _threading.Event()
    release = _threading.Event()

    def _hold() -> None:
        with session._gen_lock:
            held.set()
            release.wait(timeout=5)

    holder = _threading.Thread(target=_hold)
    holder.start()
    try:
        assert held.wait(timeout=5)
        with pytest.raises(ConcurrentExtractionError, match="retry shortly"):
            session.remove_probe("toy")
        # The roster is untouched — the reject protected the in-flight read.
        assert "toy" in session._monitor.probe_names
    finally:
        release.set()
        holder.join(timeout=5)

    # Idle again: the detach applies normally.
    session.remove_probe("toy")
    assert "toy" not in session._monitor.probe_names


# ================================================== geometry state lock ===

def test_geometry_state_lock_serializes_detach_and_reads():
    """The geometry-state boundary (the Monitor sibling of the lens
    round-5 fix): the session's ``remove_probe`` geometry branch and the
    coherent read surfaces (``specs``) hold the instrument's state lock —
    an idle detach can no longer land inside a concurrent registry
    iteration or tear it."""
    session = _stub_session()
    m = _toy_manifold()
    _serve(session, m)
    session.add_probe("toy")
    inst = session._geometry_instrument

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
        assert "toy" in session._monitor.probe_names  # nothing mutated mid-hold
    detacher.join(timeout=5.0)
    reader.join(timeout=5.0)
    assert not detacher.is_alive() and not reader.is_alive()
    assert "toy" not in session._monitor.probe_names
    assert read_out and isinstance(read_out[0], dict)


def test_geometry_idle_run_reads_hold_the_state_lock():
    """Idle-passthrough reads (an unbound ``GeometryRun.observe``) pair
    with the roster under the state lock — the round-6 idle-coherence
    contract, geometry edition.  The bound per-token path stays
    lock-free (mid-generation mutation is excluded by the detach
    reject contract, not by this lock)."""
    session = _stub_session()
    m = _toy_manifold()
    _serve(session, m)
    session.add_probe("toy")
    inst = session._geometry_instrument
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


def test_geometry_roster_reads_do_not_tear_under_concurrent_detach():
    """The tear repro: ``plan()``'s roster walk (``probe_names`` +
    ``probe_layers``) iterates live ``Monitor._probes`` state and raised
    ``RuntimeError: dictionary changed size during iteration`` under a
    concurrent idle attach/detach before the state lock existed (the
    exact class the lens round-5 fix closed).  Hammer both sides; any
    exception on either thread fails."""
    import time as _time

    from drowse.readout.types import ReadRequest

    session = _stub_session()
    manifolds = {f"toy{i}": _toy_manifold() for i in range(6)}

    _serve(session, manifolds)
    for name in manifolds:
        session.add_probe(name)
    inst = session._geometry_instrument

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
    monitor = session._monitor
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


def test_manifold_promotion_walk_holds_geometry_state_lock():
    """A fit's probe-promotion walk (the registry's swap listener) mutates
    the Monitor roster outside the instrument's attach/detach — the
    remove+add pair must land under the geometry-state lock so a
    coherent reader can't observe a half-applied promotion."""
    session = _stub_session()
    m_old = _toy_manifold()
    _serve(session, m_old)
    session.add_probe("toy")
    session._registry.put_manifold("toy", m_old)
    session._registry.on_swap(session._roster.swap)
    inst = session._geometry_instrument
    shim = _RecordingLock()
    inst.state_lock = shim  # type: ignore[assignment]
    held_at_call = _record_roster_mutations(session, shim)

    m_new = _toy_manifold()
    session._registry.adopt_fitted("default/toy", m_new)

    assert ("remove", True) in held_at_call
    assert ("add", True) in held_at_call
    assert all(held for _op, held in held_at_call)


def test_failed_override_eviction_holds_geometry_state_lock():
    """An artifact eviction's probe removal is the same out-of-instrument
    roster mutation — it must hold the geometry-state lock too."""
    session = _stub_session()
    m_old = _toy_manifold()
    _serve(session, m_old)
    session.add_probe("toy")
    session._registry.put_manifold("toy", m_old)
    session._registry.on_swap(session._roster.swap)
    inst = session._geometry_instrument
    shim = _RecordingLock()
    inst.state_lock = shim  # type: ignore[assignment]
    held_at_call = _record_roster_mutations(session, shim)

    session._registry.evict_artifact("default", "toy", feature="raw")

    assert ("remove", True) in held_at_call
    assert "toy" not in session._monitor.probe_names


# ================================== per-forward reuse of the sink row ===

def _wire_begin_capture(session: Any) -> dict[str, Any]:
    """Minimal capture wiring so the read plane's ``begin`` runs on the
    fixture stub, capturing whichever step sink the plan arms."""
    holder: dict[str, Any] = {}
    session._capture.attach = lambda *_a, **_k: None
    session._capture.clear = lambda: None
    session._capture.set_retention = (
        lambda *, depth, step_sink=None, tail_layers=None: (
            holder.__setitem__("sink", step_sink)
        )
    )
    return holder


def test_gate_callback_reuses_the_full_incremental_sink_row():
    """The composer's FULL-incremental branch reads this forward's sink row —
    no rescore (the monitor is poisoned after the sink to prove it)."""
    session = _stub_session()
    m = _toy_manifold()
    _serve(session, m)
    session.add_probe("toy")
    holder = _wire_begin_capture(session)
    ok = session._read_plane.begin(ReadDemand(need_per_token=True))
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

    session._monitor.score_single_token = _poisoned  # type: ignore[method-assign]
    _gated(session, "0.3 a@when:toy>0 + 0.3 b@when:toy:fraction>0")
    cb = session._read_plane.gate_score_callback()
    flat = cb(0)
    assert "toy" in flat and "toy:fraction" in flat
    session._geometry_instrument.close_run()


def test_token_tap_reuses_the_full_incremental_sink_row():
    """The token tap's FULL-incremental branch reads the sink row too — the
    event's geometry readings ARE the sink's row."""
    from drowse.generation.token_callback import (
        TokenConsumer,
        TokenConsumerOptions,
    )
    from drowse.session.engine import ConsumerPlan, TokenTap

    session = _stub_session()
    m = _toy_manifold()
    _serve(session, m)
    session.add_probe("toy")
    holder = _wire_begin_capture(session)
    ok = session._read_plane.begin(ReadDemand(need_per_token=True))
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

    session._monitor.score_single_token = _poisoned  # type: ignore[method-assign]
    session.geometry = session._geometry_instrument
    session.lens = session._lens_instrument
    session.sae = session._sae_instrument
    events: list[Any] = []
    tap = TokenTap(
        session,
        reads=session._read_plane,
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
    row = session._read_plane._rows[-1]
    assert events[0].probe_readings == row
    assert events[0].probe_readings["toy"] is row["toy"]
    session._geometry_instrument.close_run()


def test_pooled_geometry_aggregate_routes_through_the_run():
    """The aggregate-only finalize read reaches the roster through
    ``current_run.observe_aggregate`` with the one pooled slice — the
    round-2 bypass site, so a run spy pins the route."""
    session = _stub_session()
    m = _toy_manifold()
    _serve(session, m)
    session.add_probe("toy")
    pooled = {
        layer_idx: sub.mean + sub.basis[0]
        for layer_idx, sub in m.layers.items()
    }
    session._capture.tail_slice_at = lambda _idx: pooled
    run = session._geometry_instrument.current_run
    seen: list[Any] = []
    orig = run.observe_aggregate

    def _spy(p: Any) -> Any:
        seen.append(p)
        return orig(p)

    run.observe_aggregate = _spy  # type: ignore[method-assign]
    plane = session._read_plane
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
    session = _stub_session()
    m = _toy_manifold()
    _serve(session, m)
    session.add_probe("toy")
    pooled = {
        layer_idx: sub.mean + sub.basis[0]
        for layer_idx, sub in m.layers.items()
    }

    via_run = session._geometry_instrument.current_run.observe_aggregate(
        pooled,
    )
    live = session._monitor.score_single_token(pooled)

    assert set(via_run) == set(live) == {"toy"}
    assert via_run["toy"].coords == live["toy"].coords
    assert via_run["toy"].fraction == live["toy"].fraction
    assert via_run["toy"].residual == live["toy"].residual
    assert via_run["toy"].nearest == live["toy"].nearest


def test_pooled_aggregate_slice_is_one_position_for_every_family():
    """All three families finalize through the read plane's one pooled
    slice, and it lands on the aggregate *forward index* in FULL retention
    too.

    FULL retention never advances the capture's forward counter (only the
    bounded-ring modes do), so a ring lookup would clamp to the last forward —
    the trailing special — instead of the last content token.
    """
    session = _stub_session()
    rows = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    session._capture = cast(Any, types.SimpleNamespace(
        stacked=lambda: {0: rows},
        tail_slice_at=lambda _idx: {0: rows[-1]},
    ))
    plane = session._read_plane
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

def _gated(session: Any, expr: str) -> None:
    from drowse.steering.grammar import parse_expr

    session._steering_composer._stack.append(dict(parse_expr(expr).alphas))


def _latest_on_subspace(session: Any, m: Any) -> None:
    """Latest-per-layer captures placed inside the manifold's subspace."""
    latest = {L: sub.mean + sub.basis[0] for L, sub in m.layers.items()}
    session._capture.latest_per_layer = lambda: dict(latest)


def test_gating_callback_emits_probe_scalars():
    """The composer-built gating callback scores the gated keys of attached
    probes from the latest captures — fraction, per-node distance (the
    virtual neutral candidate included) and coordinate axis 0."""
    session = _stub_session()
    m = _toy_manifold()
    _serve(session, m)
    session.add_probe("toy")
    _latest_on_subspace(session, m)
    _gated(
        session,
        "0.3 a@when:toy:fraction>0.1 + 0.3 b@when:toy@neutral>-9 "
        "+ 0.3 c@when:toy>0",
    )

    flat = session._read_plane.gate_score_callback()(0)

    assert set(flat) == {"toy:fraction", "toy@neutral", "toy"}
    full = session._monitor.flat_scalars(
        session._monitor.score_single_token(
            session._capture.latest_per_layer(),
        ),
    )
    for key, value in flat.items():
        assert value == pytest.approx(full[key], abs=1e-5)


def test_gate_on_unattached_probe_scores_nothing():
    """A gate naming a probe no family attached can never fire, so the
    callback must not fall back to scoring the whole roster every token."""
    session = _stub_session()
    m = _toy_manifold()
    _serve(session, m)
    session.add_probe("toy")
    _latest_on_subspace(session, m)
    _gated(session, "0.3 a@when:ghost>0.1")

    def _poisoned(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("an unowned gate must not score the roster")

    session._monitor.score_single_token = _poisoned  # type: ignore[method-assign]
    session._monitor.score_planned_gate_scalars = _poisoned  # type: ignore[method-assign]

    assert session._read_plane.gate_score_callback()(0) == {}


def test_gating_callback_empty_capture_returns_empty():
    session = _stub_session()
    m = _toy_manifold()
    _serve(session, m)
    session.add_probe("toy")
    session._capture.latest_per_layer = lambda: {}

    cb = session._read_plane.gate_score_callback()
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
