"""The lens runtime and instrument: residency, readouts, probes, live step.

Unit tests run a real :class:`~drowse.lens.instrument.LensInstrument` on the
toy lens model (:class:`tests.support.lens.LensHarness`).  Tests that need
the session's generation boundary or all three read families run on the
tiny real session (``tiny_session`` / ``make_tiny_session``).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from drowse.base.lru import BoundedLRU
from drowse.lens.jlens import (
    JacobianLens,
    LensNotFittedError,
    MultiTokenWordError,
)
from drowse.lens.runtime import (
    DECODE_CACHE_MAX,
    DEVICE_STACK_CACHE_MAX,
    READOUT_CHUNK_BYTES,
    SELECTOR_CACHE_MAX,
    select_rows,
)
from drowse.lens.store import lens_manifest_path, remove_lens
from drowse.loom.errors import InvalidNodeOperationError, UnknownNodeError
from drowse.loom.recipe import Recipe
from drowse.loom.tree import LoomTree
from drowse.model.chat import ChatTraits
from drowse.model.fingerprint import loaded_model_fingerprint
from drowse.session.session import ReadDemand
from tests.support.lens import LENS_PROMPTS as _PROMPTS
from tests.support.lens import TOY_MODEL_ID as _MODEL_ID
from tests.support.lens import FakeCapture, LensHarness

pytestmark = pytest.mark.usefixtures("isolated_home")

#: Byte-tokenized prompts long enough for the estimator on the tiny session.
_TINY_PROMPTS = [
    "a first prompt that is long enough for the tiny model",
    "the second prompt, also long enough for the tiny one",
    "and a third one to round out the tiny model's corpus",
    "plus a fourth prompt for the tiny model's resume path",
]


def test_bounded_lru_evicts_least_recently_used() -> None:
    cache: BoundedLRU[str, int] = BoundedLRU(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # refreshes a
    cache.put("c", 3)
    assert "b" not in cache and "a" in cache and "c" in cache
    assert cache.get_or_build("d", lambda: 4) == 4
    assert cache.keys() == ["c", "d"]
    with pytest.raises(ValueError):
        BoundedLRU(0)


def test_readout_chunks_positions_and_matches_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2: an explicit-positions readout runs in position chunks bounded by
    ``READOUT_CHUNK_BYTES`` and returns exactly the one-shot result."""
    import drowse.lens.runtime as runtime_mod

    s = LensHarness()
    s.lens.fit(_PROMPTS)
    prompt = "abcdefghijklmnopqrstuvwxyz"
    positions = list(range(10)) + [-1]
    whole = s.lens.readout(prompt, positions=positions, top_k=4, aggregate=True)

    rows_per_call: list[int] = []
    real_rows = s.lens.runtime.logits_rows

    def _counting(lens: Any, rows: Any) -> torch.Tensor:
        rows_per_call.append(len(rows))
        return real_rows(lens, rows)

    monkeypatch.setattr(s.lens.runtime, "logits_rows", _counting)
    vocab = int(s.model.lm_head.weight.shape[0])
    n_layers = len(s.lens.resident().source_layers)
    # Room for exactly three positions' rows per chunk.
    monkeypatch.setattr(runtime_mod, "READOUT_CHUNK_BYTES", 3 * n_layers * vocab * 4)
    chunked = s.lens.readout(prompt, positions=positions, top_k=4, aggregate=True)

    assert chunked == whole
    assert max(rows_per_call) == 3 * n_layers
    assert sum(rows_per_call) == len(positions) * n_layers
    assert READOUT_CHUNK_BYTES >= 3 * n_layers * vocab * 4


def test_register_direction_and_eviction_report_changes() -> None:
    """The instrument writes ``jlens/`` profiles into the injected registry
    and tells the session (``on_change``) whenever they or the resident lens
    change; eviction drops the directions."""
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    s.lens.probes["jlens/b"] = {"word": "b", "token_id": 2, "layers": [0]}
    s.changes.clear()
    name = s.lens.register_direction("a")
    assert name in s.profiles and s.changes == [[]]
    s.lens.evict()
    assert name not in s.profiles
    assert s.changes[-1] == ["jlens/b"]
    assert s.lens.resident_lens is None


def test_lens_operations_take_the_exclusive_guard() -> None:
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    s.lens.readout("abcdefgh")
    s.lens.attach("jlens/a", as_name=None)
    assert [m.split(" called")[0] for m in s.exclusive_messages] == [
        "session.lens.fit", "session.lens.readout", "add_probe",
    ]


def test_jlens_device_stacks_use_a_bounded_lru_cache() -> None:
    session = LensHarness()
    lens = JacobianLens(
        {0: torch.eye(6), 1: torch.eye(6)},
        n_prompts=1,
        d_model=6,
    )
    device = torch.device("cpu")

    first = session.lens.runtime.transport_stack(lens, [0], device)
    session.lens.runtime.transport_stack(lens, [1], device)
    assert session.lens.runtime.transport_stack(lens, [0], device) is first
    session.lens.runtime.transport_stack(lens, [0, 1], device)

    assert len(session.lens.runtime._stacks) == DEVICE_STACK_CACHE_MAX
    keys = list(session.lens.runtime._stacks)
    assert (id(lens), str(device), (0,)) in keys
    assert (id(lens), str(device), (0, 1)) in keys
    assert (id(lens), str(device), (1,)) not in keys


def test_jlens_token_decode_cache_is_bounded() -> None:
    session = LensHarness()

    for token_id in range(DECODE_CACHE_MAX + 1):
        assert session.lens.runtime.decode_id(token_id) == session.tokenizer.decode([token_id])

    assert len(session.lens.runtime._decoded) == DECODE_CACHE_MAX
    assert 0 not in session.lens.runtime._decoded
    assert DECODE_CACHE_MAX in session.lens.runtime._decoded


def test_jlens_selector_tensor_caches_are_bounded() -> None:
    session = LensHarness(n_layers=SELECTOR_CACHE_MAX + 2)
    for index in range(SELECTOR_CACHE_MAX + 1):
        layers = list(range(index + 1))
        session.lens.runtime.depth_tensor(layers, torch.device("cpu"))
        session.lens.runtime.long_tensor(layers, torch.device("cpu"))

    assert len(session.lens.runtime._depths) == SELECTOR_CACHE_MAX
    assert len(session.lens.runtime._depth_tensors) == SELECTOR_CACHE_MAX
    assert len(session.lens.runtime._long_tensors) == SELECTOR_CACHE_MAX
    assert (0,) not in session.lens.runtime._depths


def test_jlens_property_rejects_legacy_sidecar_without_weight_identity() -> None:
    fitted = LensHarness()
    fitted.lens.fit(_PROMPTS)
    sidecar_path = lens_manifest_path(_MODEL_ID, "j")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar.pop("model_fingerprint")
    sidecar_path.write_text(json.dumps(sidecar))
    assert LensHarness().lens.resident() is None


def test_jlens_property_refreshes_and_evicts_after_external_lifecycle() -> None:
    corpus_b = [f"replacement property prompt {i} with enough content" for i in range(4)]
    session_a = LensHarness()
    resident_a = session_a.lens.fit(_PROMPTS, force=True)
    session_a.profiles["jlens/example"] = {0: torch.ones(6)}
    session_a.lens.probes["jlens/example"] = {
        "word": "example", "token_id": 1, "layers": [0, 1],
    }
    disk_b = LensHarness().lens.fit(corpus_b, force=True)

    assert session_a.lens.has_compatible()
    refreshed = session_a.lens.resident_lens

    assert refreshed is not None
    assert any(
        not torch.equal(resident_a.jacobians[layer], refreshed.jacobians[layer])
        for layer in refreshed.source_layers
    )
    for layer in disk_b.source_layers:
        assert torch.allclose(
            refreshed.jacobians[layer], disk_b.jacobians[layer], atol=2e-3,
        )

    assert remove_lens(_MODEL_ID, "j")
    assert not session_a.lens.has_compatible()
    assert session_a.lens.resident() is None
    assert "jlens/example" not in session_a.profiles
    assert session_a.lens.probes["jlens/example"]["layers"] == []
    assert session_a.lens.live is None


def test_generation_boundary_refreshes_external_lens_once(
    make_tiny_session: Any,
) -> None:
    """A generation snapshots the current pointer before fixing capture layers."""
    session_a = make_tiny_session(warm=False)
    resident_a = session_a.lens.fit(_TINY_PROMPTS, force=True)
    session_a.lens.probes["jlens/example"] = {
        "word": "example", "token_id": 1,
        "layers": list(resident_a.source_layers),
    }

    corpus_b = [
        f"replacement boundary prompt {i} with enough content" for i in range(4)
    ]
    disk_b = make_tiny_session(warm=False).lens.fit(corpus_b, force=True)

    try:
        session_a._begin_capture(ReadDemand(final_probe_aggregate=True))

        snap = session_a.lens.generation_lens
        assert snap is not None and snap is session_a.lens.resident_lens
        assert session_a.lens.generation_lens_active is True
        assert any(
            not torch.equal(resident_a.jacobians[layer], snap.jacobians[layer])
            for layer in snap.source_layers
        )
        for layer in disk_b.source_layers:
            assert torch.allclose(
                snap.jacobians[layer], disk_b.jacobians[layer], atol=2e-3,
            )

        # An external deletion is likewise observed before the next
        # generation, and the validated missing state is pinned (no
        # per-token retry loop).
        assert remove_lens(session_a.model_id, "j")
        session_a._begin_capture(ReadDemand(final_probe_aggregate=True))
        assert session_a.lens.generation_lens_active is True
        assert session_a.lens.generation_lens is None
        assert session_a.lens.resident_lens is None
        assert session_a.lens.probes["jlens/example"]["layers"] == []
    finally:
        session_a._end_capture()
        session_a._close_instrument_runs()


def test_external_lens_replacement_plans_and_freezes_refreshed_layers(
    make_tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refresh-then-plan boundary: an external replacement lens on a
    DISJOINT layer set must be adopted before the capture plan and the
    frozen binding are taken — pairing the new lens with stale layers
    KeyErrors in the transport stack."""
    session_a = make_tiny_session(warm=False)
    session_a.lens.fit(_TINY_PROMPTS, source_layers=[0], force=True)
    session_a.lens.probes["jlens/example"] = {
        "word": "example", "token_id": 1, "layers": [0],
    }

    # Another process replaces the artifact with a disjoint-layer fit.
    corpus_b = [
        f"replacement disjoint prompt {i} with enough content" for i in range(4)
    ]
    disk_b = make_tiny_session(warm=False).lens.fit(
        corpus_b, source_layers=[1], force=True,
    )
    assert list(disk_b.source_layers) == [1]

    attached_layers: list[int] = []
    real_attach = session_a._capture.attach

    def _record_attach(layers: Any, indices: list[int]) -> None:
        attached_layers.extend(indices)
        real_attach(layers, indices)

    monkeypatch.setattr(session_a._capture, "attach", _record_attach)
    try:
        ok = session_a._begin_capture(ReadDemand(final_probe_aggregate=True))

        assert ok is True
        # The adoption rewrote the live registry BEFORE planning/freezing:
        # the capture plan, the frozen binding, and the pinned lens all
        # agree on the replacement's layer set.
        assert session_a.lens.probes["jlens/example"]["layers"] == [1]
        assert attached_layers == [1]
        run = session_a.lens.current_run
        assert run.pinned is True
        assert run.lens is not None
        assert list(run.binding.specs["jlens/example"]["layers"]) == [1]
        assert list(run.lens.source_layers) == [1]
        # And the pinned lens can actually score the frozen band — the
        # stale pairing raised KeyError(0) out of the transport stack.
        hidden_dim = int(session_a.model_info["hidden_dim"])
        readings = session_a.lens.score_probes({1: torch.randn(hidden_dim)})
        assert "jlens/example" in readings
    finally:
        session_a._end_capture()
        session_a._close_instrument_runs()


def test_prepare_is_the_refresh_site_and_supplies_the_pin() -> None:
    """The refresh/pin protocol step: a bare ``prepare()`` — no session
    special-casing around it — adopts an external replacement lens
    (rewriting the live probe layer lists before any plan is taken) and
    returns the ``LensPrep`` snapshot that ``plan``/``bind`` consume."""
    from drowse.readout.types import LensPrep, ReadRequest

    session_a = LensHarness()
    session_a.lens.fit(_PROMPTS, source_layers=[0], force=True)
    session_a.lens.probes["jlens/example"] = {
        "word": "example", "token_id": 1, "layers": [0],
    }
    assert session_a.lens.resident() is not None  # resident before the swap

    corpus_b = [
        f"replacement disjoint prompt {i} with enough content" for i in range(4)
    ]
    LensHarness().lens.fit(corpus_b, source_layers=[1], force=True)

    inst = session_a.lens
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    assert isinstance(prep, LensPrep)
    assert prep.pinned is True
    assert prep.lens is not None and list(prep.lens.source_layers) == [1]
    # The adoption rewrote the live registry inside prepare, and the prep's
    # spec snapshot carries the layers derived from the prepared identity.
    assert session_a.lens.probes["jlens/example"]["layers"] == [1]
    assert list(prep.specs["jlens/example"]["layers"]) == [1]
    run = inst.bind(inst.plan(prep), prep)
    assert run.pinned is True and run.lens is prep.lens
    assert list(run.binding.specs["jlens/example"]["layers"]) == [1]
    session_a.lens.close_run()


def test_interleaved_adoption_cannot_desync_plan_from_pin() -> None:
    """sol's round-3 P1: prepare pins lens A; before plan/bind, the
    un-locked ``has_compatible`` (the session-info route runs it
    without the generation lock) observes a replacement lens B and adopts
    it, rewriting the live probe layer lists.  Because plan and bind
    consume the prep's snapshot — never the live registry — the run still
    measures A's layers with A pinned; the old registry reread paired A
    with B's layers and KeyErrored in the transport stack."""
    from drowse.readout.types import ReadRequest

    session_a = LensHarness()
    session_a.lens.fit(_PROMPTS, source_layers=[0], force=True)
    session_a.lens.probes["jlens/example"] = {
        "word": "example", "token_id": 1, "layers": [0],
    }
    inst = session_a.lens
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    assert prep.lens is not None and list(prep.lens.source_layers) == [0]

    # Another process swaps the artifact; the un-locked path adopts it
    # INSIDE the prepare->bind window.
    corpus_b = [
        f"replacement disjoint prompt {i} with enough content" for i in range(4)
    ]
    LensHarness().lens.fit(corpus_b, source_layers=[1], force=True)
    assert session_a.lens.has_compatible() is True  # adopts B
    assert session_a.lens.probes["jlens/example"]["layers"] == [1]

    plan = inst.plan(prep)
    run = inst.bind(plan, prep)
    # Plan layers, frozen binding, and the pinned lens all agree on A.
    assert set(plan.latest_layers) == {0}
    assert list(run.binding.specs["jlens/example"]["layers"]) == [0]
    assert run.lens is prep.lens
    hidden_dim = int(session_a.model.lm_head.weight.shape[1])
    readings = inst.score_probes({0: torch.randn(hidden_dim)})
    assert "jlens/example" in readings
    session_a.lens.close_run()


def test_bound_run_reads_prepare_time_live_state() -> None:
    """The live-readout runtime dict is part of the prep snapshot: an
    interleaved adoption rebuilds the instrument-level ``live`` against
    the NEW lens, and a bound run must keep reading the state that
    matches its pin.  Idle runs pass through to the live config."""
    from drowse.readout.types import ReadRequest

    session = LensHarness()
    inst = session.lens
    state_a = {"layers": [0]}
    inst.live = state_a
    prep = inst.prepare(ReadRequest(final_aggregate=False, live=True))
    assert prep.live_state is state_a

    inst.live = {"layers": [1]}  # the adoption rebuild lands in the window
    inst.bind(inst.plan(prep), prep)
    assert inst._measurement_live() is state_a
    session.lens.close_run()
    assert inst._measurement_live() is inst.live
    inst.live = None


def test_lens_state_lock_serializes_getter_against_snapshot() -> None:
    """The one lens-state boundary (sol's round-4 P1): the getter's
    refresh/adopt/evict transaction and ``prepare``'s snapshot hold the
    same reentrant lock, so an un-locked read (the session-info route)
    cannot land inside a generation boundary's snapshot — it blocks
    until the snapshot completes."""
    session = LensHarness()
    session.lens.fit(_PROMPTS, force=True)
    inst = session.lens

    entered = threading.Event()
    seen: list[Any] = []

    def _getter_read() -> None:
        entered.set()
        seen.append(session.lens.resident())

    with inst.state_lock:
        reader = threading.Thread(target=_getter_read)
        reader.start()
        assert entered.wait(timeout=5.0)
        reader.join(timeout=0.2)
        assert reader.is_alive()  # blocked on the lens-state lock
    reader.join(timeout=5.0)
    assert not reader.is_alive()
    assert seen and seen[0] is not None


def test_lens_state_lock_serializes_detach_and_reads() -> None:
    """round-5 P1/P2: the lens removal (``try_detach``, the session's
    ``remove_probe`` lens branch and the HTTP DELETE path) and the coherent read surfaces (``specs``) hold
    the lens-state lock — a detach can no longer land inside a
    ``prepare`` snapshot or tear a concurrent registry iteration."""
    session = LensHarness()
    session.lens.probes["jlens/example"] = {
        "word": "example", "token_id": 1, "layers": [0],
    }
    inst = session.lens

    detach_entered = threading.Event()
    read_entered = threading.Event()
    read_out: list[Any] = []

    def _detach() -> None:
        detach_entered.set()
        inst.try_detach("jlens/example")

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
        assert detacher.is_alive()  # both blocked on the lens-state lock
        assert reader.is_alive()
        assert "jlens/example" in inst.probes  # nothing mutated mid-hold
    detacher.join(timeout=5.0)
    reader.join(timeout=5.0)
    assert not detacher.is_alive() and not reader.is_alive()
    assert "jlens/example" not in inst.probes
    assert read_out and isinstance(read_out[0], dict)


def test_idle_measurement_state_is_coherent() -> None:
    """round-6 P1: the idle-passthrough read pairs the lens and the spec
    snapshot in ONE state-lock hold — ``_measurement_specs`` handing out
    the live registry while the lens resolved separately let an idle
    read pair lens A with a concurrently adopted B's rewritten layers.
    The helper's getter read also refreshes, so the pair reflects the
    replacement consistently."""
    from drowse.readout.types import ReadRequest

    session_a = LensHarness()
    session_a.lens.fit(_PROMPTS, source_layers=[0], force=True)
    session_a.lens.probes["jlens/example"] = {
        "word": "example", "token_id": 1, "layers": [0],
    }
    corpus_b = [
        f"replacement idle prompt {i} with enough content" for i in range(4)
    ]
    LensHarness().lens.fit(corpus_b, source_layers=[1], force=True)

    inst = session_a.lens
    lens, specs = inst._measurement_state()
    # One hold: the refreshed lens and the copied specs agree on B.
    assert lens is not None and list(lens.source_layers) == [1]
    assert list(specs["jlens/example"]["layers"]) == [1]
    # The copy is per-call: a later registry mutation can't reach it.
    del inst.probes["jlens/example"]
    assert "jlens/example" in specs

    # And the idle read blocks while the lens-state lock is held.
    entered = threading.Event()
    out: list[Any] = []

    def _idle_read() -> None:
        entered.set()
        out.append(inst._measurement_state())

    with inst.state_lock:
        reader = threading.Thread(target=_idle_read)
        reader.start()
        assert entered.wait(timeout=5.0)
        reader.join(timeout=0.2)
        assert reader.is_alive()
    reader.join(timeout=5.0)
    assert not reader.is_alive() and out

    # A bound run stays lock-free frozen state (prep pin + binding).
    inst.probes["jlens/example"] = {
        "word": "example", "token_id": 1, "layers": [1],
    }
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    inst.bind(inst.plan(prep), prep)
    lens_bound, specs_bound = inst._measurement_state()
    assert prep.pinned and lens_bound is prep.lens
    assert specs_bound is inst.current_run.binding.specs
    session_a.lens.close_run()


def test_prepare_pin_demand_formula() -> None:
    """Pin demand = a live readout, or attached probes with a final
    aggregate or a lens gate — the one formula both generation
    boundaries (``_begin_capture`` and the batch preamble) reduce to."""
    from drowse.readout.types import ReadRequest

    session = LensHarness()
    session.lens.fit(_PROMPTS, force=True)
    inst = session.lens

    # No probes, no live consumer -> unpinned, no disk read taken.
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    assert prep.pinned is False and prep.lens is None

    # Probes + final aggregate (also the batch shape) -> pinned.
    session.lens.probes["jlens/example"] = {
        "word": "example", "token_id": 1, "layers": [0],
    }
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    assert prep.pinned is True and prep.lens is not None

    # Final readings disabled: dormant probes do not pin — a lens gate does.
    prep = inst.prepare(ReadRequest(final_aggregate=False))
    assert prep.pinned is False
    prep = inst.prepare(ReadRequest(
        final_aggregate=False,
        gate_keys=frozenset({"jlens/example"}),
    ))
    assert prep.pinned is True

    # A live-readout consumer pins with an empty probe registry.
    del session.lens.probes["jlens/example"]
    inst.live = {"layers": [0]}
    prep = inst.prepare(ReadRequest(final_aggregate=False, live=True))
    assert prep.pinned is True and prep.request.live is True
    inst.live = None


def test_prepare_on_a_bound_run_raises(tiny_session: Any) -> None:
    """The jlens getter short-circuits on a bound run's pin flag, so a
    prepare taken without closing the prior run would silently skip the
    refresh — every family rejects it instead."""
    from drowse.readout.types import ReadRequest

    session = tiny_session
    request = ReadRequest(final_aggregate=False)
    instruments = (session.lens, session.sae, session.geometry)
    for inst in instruments:
        prep = inst.prepare(request)
        inst.bind(inst.plan(prep), prep)
        with pytest.raises(RuntimeError, match="bound run"):
            inst.prepare(request)
    for inst in instruments:
        inst.close_run()
    for inst in instruments:
        inst.prepare(request)  # idle runs again — every prepare passes


def test_bind_and_plan_reject_foreign_or_missing_preps(tiny_session: Any) -> None:
    """The transaction contract is enforced, not advisory: a bare
    ``bind(plan)`` is a TypeError, a wrong-family prep (including a
    wrong-family ``LensPrep``) is rejected by every family, and plans are
    validated for family provenance too."""
    from drowse.readout.types import (
        InstrumentPlan,
        InstrumentPrep,
        LensPrep,
        ReadRequest,
    )

    session = tiny_session
    request = ReadRequest(final_aggregate=False)

    lens = session.lens
    lens_prep = lens.prepare(request)
    with pytest.raises(TypeError):
        lens.bind(InstrumentPlan(family="lens"))  # type: ignore[call-arg]  # bare bind: prep required
    with pytest.raises(TypeError, match="LensPrep"):
        lens.bind(InstrumentPlan(family="lens"), InstrumentPrep(family="lens"))
    with pytest.raises(TypeError, match="LensPrep"):
        lens.bind(
            InstrumentPlan(family="lens"), LensPrep(family="sae"),
        )
    with pytest.raises(TypeError, match="LensPrep"):
        lens.plan(InstrumentPrep(family="lens"))
    with pytest.raises(ValueError, match="plan family"):
        lens.bind(InstrumentPlan(family="sae"), lens_prep)
    assert lens.current_run.bound is False

    sae = session.sae
    sae_prep = sae.prepare(request)
    with pytest.raises(TypeError):
        sae.bind(InstrumentPlan(family="sae"))  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="family"):
        sae.bind(InstrumentPlan(family="sae"), InstrumentPrep(family="lens"))
    with pytest.raises(TypeError, match="family"):
        sae.plan(InstrumentPrep(family="geometry"))
    with pytest.raises(ValueError, match="plan family"):
        sae.bind(InstrumentPlan(family="lens"), sae_prep)
    assert sae.current_run.bound is False

    geometry = session.geometry
    geometry_prep = geometry.prepare(request)
    with pytest.raises(TypeError):
        geometry.bind(InstrumentPlan(family="geometry"))  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="family"):
        geometry.bind(
            InstrumentPlan(family="geometry"), InstrumentPrep(family="sae"),
        )
    with pytest.raises(TypeError, match="family"):
        geometry.plan(InstrumentPrep(family="lens"))
    with pytest.raises(ValueError, match="plan family"):
        geometry.bind(InstrumentPlan(family="lens"), geometry_prep)
    assert geometry.current_run.bound is False


def test_bind_rejects_same_family_plan_prep_crossing(tiny_session: Any) -> None:
    """sol's round-4 P2: a plan derived from prep A must not bind with
    prep B of the same family — the session's capture union would retain
    A's layers while the run measures B.  The per-preparation token the
    plan echoes is compared at bind, in every family (a hand-built plan,
    which carries no token, is rejected the same way)."""
    from drowse.readout.types import InstrumentPlan, ReadRequest

    session = tiny_session
    request = ReadRequest(final_aggregate=False)
    for inst in (session.lens, session.sae, session.geometry):
        prep_a = inst.prepare(request)
        prep_b = inst.prepare(request)
        plan_a = inst.plan(prep_a)
        assert plan_a.prep_token == prep_a.token != prep_b.token
        with pytest.raises(ValueError, match="prep_token"):
            inst.bind(plan_a, prep_b)
        with pytest.raises(ValueError, match="prep_token"):
            inst.bind(InstrumentPlan(family=inst.family), prep_a)
        assert inst.current_run.bound is False
        inst.bind(plan_a, prep_a)  # the matched pair binds
        inst.close_run()

    # Cross-INSTANCE crossing (round-5): tokens are one process-wide
    # sequence, so a plan from another instrument instance of the same
    # family can't bind here either (per-instance counters both started
    # at 1 and collided).
    other = LensHarness()
    prep_other = other.lens.prepare(request)
    plan_other = other.lens.plan(prep_other)
    prep_local = session.lens.prepare(request)
    assert prep_other.token != prep_local.token
    with pytest.raises(ValueError, match="prep_token"):
        session.lens.bind(plan_other, prep_local)


def test_prepare_rejects_a_layerless_pin_demanded_lens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pin-demanded lens without ``source_layers`` is structurally
    broken (layers align captures, specs, and Jacobians) — it fails at
    the prepare boundary, not as a mid-generation KeyError."""
    from drowse.readout.types import ReadRequest

    session = LensHarness()
    inst = session.lens
    monkeypatch.setattr(inst, "resident", lambda: object())  # no source_layers
    inst.live = {"layers": [0]}
    with pytest.raises(RuntimeError, match="source_layers"):
        inst.prepare(ReadRequest(final_aggregate=False, live=True))


def test_generation_lens_snapshot_avoids_per_token_disk_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe and gate scoring share the boundary snapshot without shard opens."""
    import drowse.lens.store as lens_io

    session = LensHarness()
    lens = session.lens.fit(_PROMPTS, force=True)
    layers = list(lens.source_layers)
    session.lens.probes["jlens/a"] = {
        "word": "a", "token_id": 1, "layers": layers,
    }
    session.lens.generation_lens = lens
    session.lens.generation_lens_active = True
    session.lens.active_for_generation = False

    monkeypatch.setattr(
        lens_io,
        "load_lens_sidecar",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("per-token lens scoring reopened the disk pointer")
        ),
    )
    assert session.lens.resident() is lens
    assert session.lens.has_compatible()

    vocab = int(session.model.lm_head.weight.shape[0])
    probabilities = torch.softmax(torch.randn(len(layers), vocab), dim=-1)
    for _ in range(3):
        readings = session.lens.score_probes_from_rows(
            layers=layers, probabilities=probabilities,
        )
        assert "jlens/a" in readings

    hidden = {
        layer: torch.randn(lens.d_model)
        for layer in layers
    }
    session.capture = SimpleNamespace(latest_per_layer=lambda: hidden)
    for _ in range(3):
        assert "jlens/a" in session.lens.gate_scalars({"jlens/a"})


def test_jlens_property_rejects_changed_loaded_weights() -> None:
    LensHarness().lens.fit(_PROMPTS)
    changed = LensHarness()
    changed_model: Any = changed.model
    with torch.no_grad():
        changed_model.model.layers[0].w1.data.reshape(-1)[1] += 0.125
    assert changed.lens.resident() is None


def test_jlens_property_rechecks_loaded_pointer_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    session = LensHarness()
    live_fp = loaded_model_fingerprint(session.model, _MODEL_ID)
    lens = JacobianLens({0: torch.eye(6)}, n_prompts=1, d_model=6)
    compatible = {
        "model_fingerprint": live_fp, "sites": {"out": {"layers": [0]}},
        "n_prompts": 1,
    }
    replaced = {**compatible, "model_fingerprint": "different-weights"}
    monkeypatch.setattr(lens_io, "load_lens_sidecar", lambda _model: compatible)
    monkeypatch.setattr(lens_io, "load_lens", lambda _model: (lens, replaced))

    assert session.lens.resident() is None


def test_has_compatible_jlens_rechecks_loaded_pointer_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.lens.store as lens_io

    session = LensHarness()
    live_fp = loaded_model_fingerprint(session.model, _MODEL_ID)
    lens = JacobianLens({0: torch.eye(6)}, n_prompts=1, d_model=6)
    compatible = {
        "model_fingerprint": live_fp, "sites": {"out": {"layers": [0]}},
        "n_prompts": 1,
    }
    replaced = {**compatible, "model_fingerprint": "different-weights"}
    session.lens.runtime.lens = lens
    session.lens.runtime.identity = ("old",)
    monkeypatch.setattr(lens_io, "load_lens_sidecar", lambda _model: compatible)
    monkeypatch.setattr(lens_io, "load_lens", lambda _model: (lens, replaced))

    assert not session.lens.has_compatible()


def test_jlens_readout_shape_and_default_position() -> None:
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    seen_pool: list[int | None] = []
    import drowse.fit.capture as _vectors

    real_capture = _vectors._capture_all_hidden_states

    def _spy(model: Any, layers: Any, ids: torch.Tensor, **kw: Any) -> Any:
        pool = kw.get("pool_index")
        seen_pool.append(int(pool) if pool is not None else None)
        return real_capture(model, layers, ids, **kw)

    _vectors._capture_all_hidden_states = _spy
    try:
        out = s.lens.readout("a prompt that is long enough.", top_k=3)
    finally:
        _vectors._capture_all_hidden_states = real_capture
    assert seen_pool == [len(s.tokenizer.encode("a prompt that is long enough.")) - 1]
    assert set(out) == {0, 1}  # 3-layer toy: sources are 0 and 1
    for rows in out.values():
        assert len(rows) == 1  # default: final position only
        assert len(rows[0]) == 3
        # One unit end to end: the per-layer readout PROBABILITY p_l, the
        # same quantity the aggregate's ``strength`` averages.
        token, p_l = rows[0][0]
        assert isinstance(token, str) and 0.0 <= p_l <= 1.0


def test_jlens_readout_aggregate_rides_same_logits() -> None:
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    result = s.lens.readout(
        "a prompt that is long enough.", top_k=3, aggregate=True,
    )
    out, agg = result
    assert set(out) == {0, 1}
    # default position only → one aggregate list
    assert len(agg) == 1
    rows = agg[0]
    assert len(rows) == 3
    strengths = [r[1] for r in rows]
    assert strengths == sorted(strengths, reverse=True)
    for tok, strength, com, spread in rows:
        assert isinstance(tok, str)
        assert 0.0 <= strength <= 1.0
        assert 0.0 <= com <= 1.0
        assert spread >= 0.0
    # Both fitted source layers contribute. Softmax assigns positive mass to
    # every vocabulary item at each layer, so every aggregate row has a
    # non-degenerate depth distribution over L0 and L1.
    for _, _, com, spread in rows:
        assert 0.0 < com < 1 / (3 - 1)
        assert spread > 0.0


def test_jlens_readout_aggregate_multi_position() -> None:
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    result = s.lens.readout(
        "a prompt that is long enough.", positions=[-2, -1], top_k=2,
        aggregate=True,
    )
    out, agg = result
    assert all(len(rows) == 2 for rows in out.values())
    assert len(agg) == 2
    assert all(len(rows) == 2 for rows in agg)


def test_jlens_readout_requires_fitted_lens() -> None:
    s = LensHarness()
    with pytest.raises(LensNotFittedError, match="drowse lens fit"):
        s.lens.readout("a prompt that is long enough.")


def test_jlens_readout_rejects_unfitted_layer() -> None:
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    with pytest.raises(ValueError, match="not in the fitted lens"):
        s.lens.readout("a prompt that is long enough.", layers=[9])


def test_register_jlens_direction_registers_profile() -> None:
    s = LensHarness()
    lens = s.lens.fit(_PROMPTS)
    seen_layers: list[list[int] | None] = []
    real_token_direction = lens.token_direction

    def _spy_token_direction(
        token_id: int,
        unembed: torch.Tensor,
        *,
        layers: list[int] | None = None,
        output_projection: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        seen_layers.append(layers)
        return real_token_direction(token_id, unembed, layers=layers, output_projection=output_projection)

    lens.token_direction = _spy_token_direction  # type: ignore[method-assign]
    name = s.lens.register_direction("g")  # 'g' round-trips in the toy vocab
    assert name == "jlens/g"
    assert seen_layers == [[0, 1]]
    dirs = s.profiles[name]
    assert set(dirs) == {0, 1}
    expected = lens.token_direction(
        s.tokenizer.encode("g")[0], s.model.lm_head.weight,
    )
    for layer, vec in dirs.items():
        assert torch.allclose(vec, expected[layer])
    # idempotent
    assert s.lens.register_direction("g") == name


def test_register_jlens_direction_multi_token_raises() -> None:
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    with pytest.raises(MultiTokenWordError):
        s.lens.register_direction("gg")


# ------------------------------------------------------------- live lens ----


def test_enable_live_lens_defaults_and_disable() -> None:
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    layers = s.lens.enable_live()
    assert layers == [0, 1]
    assert s.lens.live is not None
    assert s.lens.live["layers"] == layers
    assert "J" not in s.lens.live
    s.lens.disable_live()
    assert s.lens.live is None


def test_enable_live_lens_rejects_unfitted_layer() -> None:
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    with pytest.raises(ValueError, match="not in the fitted lens"):
        s.lens.enable_live(layers=[7])


def test_enable_live_lens_requires_lens() -> None:
    s = LensHarness()
    with pytest.raises(LensNotFittedError):
        s.lens.enable_live()


def test_enable_live_lens_registers_no_forward_hooks() -> None:
    """The live lens must not touch the model: no hooks, no wrapping — the
    reader consumes existing capture buffers (compile/fast-path safety)."""
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    before = [
        (len(block._forward_hooks), len(block._forward_pre_hooks))
        for block in s.layers
    ]
    s.lens.enable_live()
    after = [
        (len(block._forward_hooks), len(block._forward_pre_hooks))
        for block in s.layers
    ]
    assert before == after


def test_live_lens_readout_step_reads_latest_slices() -> None:
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    s.lens.enable_live(layers=[0, 1])
    # The per-step reader should use the pre-stacked transport cache, not the
    # per-layer dict.  Replacing the dict entries would have blown up the old
    # per-token ``state["J"][layer].to(...)`` path.
    class Bomb:
        def to(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("live lens readout should use J_stack")

    assert s.lens.live is not None
    s.lens.live["J"] = {0: Bomb(), 1: Bomb()}
    gen = torch.Generator().manual_seed(11)
    s.capture = FakeCapture({
        0: torch.randn(6, generator=gen),
        1: torch.randn(6, generator=gen),
    })

    step = s.lens.live_readout_step(top_k=3)
    assert step is not None
    out, agg, token_ids = step
    assert set(out) == {0, 1}
    assert set(token_ids) == {0, 1}
    for row in out.values():
        assert len(row) == 3
        assert all(isinstance(tok, str) for tok, _ in row)
    assert all(len(row) == 3 for row in token_ids.values())
    assert all(isinstance(token_id, int) for row in token_ids.values() for token_id in row)
    # display scores are per-layer softmax probabilities, descending — the
    # one strength unit every lens surface reports
    scores = [sc for _, sc in out[0]]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= sc <= 1.0 for sc in scores)
    # the aggregate chip list rides the same step: top_k rows of
    # (token, strength, com, spread) with strength descending in [0, 1]
    # and com/spread valid normalized depths
    assert len(agg) == 3
    strengths = [srow[1] for srow in agg]
    assert strengths == sorted(strengths, reverse=True)
    for tok, strength, com, spread in agg:
        assert isinstance(tok, str)
        assert 0.0 <= strength <= 1.0
        assert 0.0 <= com <= 1.0
        assert spread >= 0.0


def test_live_lens_readout_step_avoids_float64_for_mps_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_to = torch.Tensor.to

    def reject_float64(
        self: torch.Tensor, *args: Any, **kwargs: Any,
    ) -> torch.Tensor:
        dtype = kwargs.get("dtype")
        if args and isinstance(args[0], torch.dtype):
            dtype = args[0]
        if dtype is torch.float64:
            raise AssertionError("device-side float64 is unsupported on MPS")
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", reject_float64)
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    s.lens.enable_live(layers=[0, 1])
    s.capture = FakeCapture({0: torch.randn(6), 1: torch.randn(6)})

    step = s.lens.live_readout_step(top_k=3)
    assert step is not None


def test_jlens_row_selector_avoids_copy_for_identity_and_contiguous_rows() -> None:
    tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6)

    identity = select_rows(tensor, [0, 1, 2, 3])
    contiguous = select_rows(tensor, [1, 2])
    gathered = select_rows(tensor, [0, 2])

    assert identity is tensor
    assert contiguous.tolist() == tensor[1:3].tolist()
    assert (
        contiguous.untyped_storage().data_ptr()
        == tensor.untyped_storage().data_ptr()
    )
    assert gathered.tolist() == tensor[[0, 2]].tolist()
    assert (
        gathered.untyped_storage().data_ptr()
        != tensor.untyped_storage().data_ptr()
    )


def test_live_lens_step_normalizes_once_across_all_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The instrument binds the readout primitives at module scope (the per-step
    # surfaces must not re-import), so the counters patch its namespace.
    import drowse.lens.instrument as lens_instrument_module

    jlens_module = lens_instrument_module

    s = LensHarness()
    s.lens.fit(_PROMPTS)
    s.lens.enable_live(layers=[1])
    s.lens.attach("jlens/g", as_name=None)
    s.capture = FakeCapture({1: torch.randn(6)})
    calls = 0
    stat_calls = 0
    original = jlens_module.readout_probabilities
    original_stats = jlens_module.token_readout_stats_from_probabilities

    def counting_probabilities(logits: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(logits)

    def counting_stats(*args: Any, **kwargs: Any) -> Any:
        nonlocal stat_calls
        stat_calls += 1
        return original_stats(*args, **kwargs)

    monkeypatch.setattr(
        jlens_module, "readout_probabilities", counting_probabilities,
    )
    monkeypatch.setattr(
        jlens_module,
        "token_readout_stats_from_probabilities",
        counting_stats,
    )

    # A gated pinned probe calibrates before the token tap and stashes the
    # matrix, keyed by this forward's step id. Cards, pinned readings, and
    # aggregate must all reuse it — the display passes the SAME step.
    scalars = s.lens.gate_scalars(step_id=4)
    assert scalars
    assert s.lens.live_readout_step(top_k=3, step_id=4) is not None
    assert calls == 1
    assert stat_calls == 1
    assert s.lens.last_step_readings is not None
    assert s.lens.last_step_readings["jlens/g"].value == (
        pytest.approx(scalars["jlens/g"])
    )


def test_live_lens_readout_reuses_depth_and_token_selector_tensors() -> None:
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    s.lens.enable_live(layers=[0, 1])
    s.lens.attach("jlens/g", as_name=None)
    s.capture = FakeCapture({
        0: torch.randn(6, generator=torch.Generator().manual_seed(20)),
        1: torch.randn(6, generator=torch.Generator().manual_seed(21)),
    })

    assert s.lens.gate_scalars({"jlens/g"})
    first_depth_ids = {
        key: id(value) for key, value in s.lens.runtime._depth_tensors.items()
    }
    first_selector_ids = {
        key: id(value) for key, value in s.lens.runtime._long_tensors.items()
    }
    assert first_depth_ids
    assert first_selector_ids

    assert s.lens.gate_scalars({"jlens/g"})

    assert {
        key: id(value) for key, value in s.lens.runtime._depth_tensors.items()
    } == first_depth_ids
    assert {
        key: id(value) for key, value in s.lens.runtime._long_tensors.items()
    } == first_selector_ids


def test_live_lens_exact_stash_reuse_skips_hidden_cast() -> None:
    class BombHidden:
        def to(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("exact stash reuse should not cast hidden rows")

    s = LensHarness()
    s.lens.fit(_PROMPTS)
    s.lens.enable_live(layers=[1])
    s.capture = FakeCapture({1: cast(Any, BombHidden())})
    assert s.lens.live is not None
    vocab = int(s.model.lm_head.weight.shape[0])
    logits = torch.randn(1, vocab, generator=torch.Generator().manual_seed(7))
    import drowse.lens.jlens as jlens_module

    probabilities = jlens_module.readout_probabilities(logits)
    s.lens.step_stash = {
        "step": 5,
        "layers": (1,),
        "logits": logits,
        "probabilities": probabilities,
    }

    step = s.lens.live_readout_step(top_k=3, step_id=5)

    assert step is not None
    # Step-keyed reuse is idempotent — a second same-step consumer also
    # rides the stash (the old ``fresh`` flag was consume-once); a
    # different step never matches, so staleness is structural.
    assert s.lens.live_readout_step(top_k=3, step_id=5) is not None
    stash = s.lens.step_stash
    assert stash is not None
    assert stash["step"] == 5


def test_live_lens_reuses_gated_subset_rows_for_wider_display() -> None:
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    s.lens.enable_live(layers=[0, 1])
    s.lens.attach("jlens/g", as_name=None)
    s.lens.probes["jlens/g"]["layers"] = [1]
    s.capture = FakeCapture({
        0: torch.randn(6, generator=torch.Generator().manual_seed(0)),
        1: torch.randn(6, generator=torch.Generator().manual_seed(1)),
    })

    assert s.lens.gate_scalars({"jlens/g"}, step_id=3)
    assert s.lens.step_stash is not None
    assert s.lens.step_stash["layers"] == (1,)
    # Poison the live transport row for the gated layer after the gate callback.
    # The display still covers both live layers, but layer 1 should ride the
    # cached gate logits/probabilities instead of recomputing from this row.
    assert s.lens.live is not None
    row = s.lens.live["layer_rows"][1]
    s.lens.live["J_stack"][row].fill_(float("nan"))

    step = s.lens.live_readout_step(top_k=3, step_id=3)

    assert step is not None
    per_layer, aggregate, token_ids = step
    assert set(per_layer) == {0, 1}
    assert set(token_ids) == {0, 1}
    assert all(torch.isfinite(torch.tensor(score)) for _tok, score in per_layer[1])
    assert all(torch.isfinite(torch.tensor(row[1])) for row in aggregate)


def test_lens_gate_keys_none_vs_empty_contract() -> None:
    """``gate_keys=None`` scores the full roster; an explicit ``set()``
    scores nothing (the worker's old ``if gate_keys:`` treated both as
    full-roster — sol's round-3 P2).  The lens member of the three-family
    contract pin."""
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    s.lens.attach("jlens/g", as_name=None)
    layers = [int(layer) for layer in s.lens.resident().source_layers]
    d_model = next(iter(s.lens.resident().jacobians.values())).shape[0]
    s.capture = FakeCapture({
        layer: torch.randn(
            d_model, generator=torch.Generator().manual_seed(layer),
        )
        for layer in layers
    })
    inst = s.lens
    full = inst.gate_scalars(None, step_id=0)
    assert "jlens/g" in full
    assert inst.gate_scalars(set(), step_id=0) == {}


def test_lens_run_observe_rescores_every_call() -> None:
    """The run's ``observe`` is a pass-through: the gate→display reuse is
    the step-keyed worker stash, so there is no observe memo to go stale
    (a repeated step id with different hidden states rescores)."""
    from drowse.readout.types import ReadRequest

    s = LensHarness()
    s.lens.fit(_PROMPTS)
    s.lens.attach("jlens/g", as_name=None)
    inst = s.lens
    prep = inst.prepare(ReadRequest(final_aggregate=True))
    run = inst.bind(inst.plan(prep), prep)

    calls: list[int] = []

    def _fresh_probes(_hidden: Any, **_kw: Any) -> dict[str, Any]:
        calls.append(1)
        return {"jlens/g": object()}

    inst.score_probes = _fresh_probes  # type: ignore[method-assign]
    for step in (-1, -1, 7, 7):
        run.observe(step, {})
    assert len(calls) == 4
    assert not hasattr(run, "prime_observation")
    inst.close_run()


def test_live_lens_readout_step_none_when_off() -> None:
    s = LensHarness()
    assert s.lens.live_readout_step() is None


# --------------------------------------------------- token readout (loom) ----


_PROMPT_RENDER = "the prompt render, chat shaped."


class TreeLensHarness(LensHarness):
    """A harness whose replay host is itself: a real loom tree plus a
    recorded prompt render and steering scopes."""

    def __init__(self) -> None:
        super().__init__()
        self.chat_traits = ChatTraits(self.tokenizer)
        self.tree = LoomTree(model_id=_MODEL_ID)
        self.prepare_calls: list[dict[str, Any]] = []
        self.steering_scopes: list[Any] = []

    def _prepare_input(
        self,
        input: Any,
        raw: bool = False,
        thinking: bool = False,
        stateless: bool = False,
        parent_node_id: str | None = None,
        user_role: str | None = None,
        assistant_role: str | None = None,
        to_device: bool = True,
        gen_seat: str = "assistant",
        system_prompt: Any = None,
    ) -> torch.Tensor:
        self.prepare_calls.append({
            "input": input, "raw": raw, "thinking": thinking, "system_prompt": system_prompt,
            "parent_node_id": parent_node_id,
            "user_role": user_role, "assistant_role": assistant_role,
            "gen_seat": gen_seat,
        })
        return torch.tensor(
            [self.tokenizer.encode(_PROMPT_RENDER)], dtype=torch.long,
        )

    @contextmanager
    def steering(self, value: Any) -> Iterator[None]:
        self.steering_scopes.append(value)
        yield


def _tree_with_assistant(
    s: TreeLensHarness,
    raw_ids: list[int] | None,
    recipe: Recipe | None = None,
) -> str:
    user_id = s.tree.add_user_turn("a user turn")
    node_id = s.tree.begin_assistant(user_id, recipe=recipe)
    s.tree.finalize_assistant(
        node_id, text="an assistant turn", finish_reason="stop",
        raw_token_ids=raw_ids,
    )
    return node_id


def test_jlens_token_readout_shape_and_position() -> None:
    s = TreeLensHarness()
    s.lens.fit(_PROMPTS)
    raw_ids = s.tokenizer.encode("abcdefg")
    node_id = _tree_with_assistant(s, raw_ids, Recipe(system_prompt="Saved pirate instruction"))

    seen_lens: list[tuple[int, int | None]] = []
    import drowse.fit.capture as _vectors

    real_capture = _vectors._capture_all_hidden_states

    def _spy(model: Any, layers: Any, ids: torch.Tensor, **kw: Any) -> Any:
        pool = kw.get("pool_index")
        seen_lens.append((int(ids.shape[1]), int(pool) if pool is not None else None))
        return real_capture(model, layers, ids, **kw)

    _vectors._capture_all_hidden_states = _spy
    try:
        out = s.lens._replay_readout(node_id, 3, top_k=4)
    finally:
        _vectors._capture_all_hidden_states = real_capture

    prompt_len = len(s.tokenizer.encode(_PROMPT_RENDER))
    # readout position: the forward that PRODUCED the clicked token —
    # prompt + raw[:3], never including the clicked token itself.
    assert seen_lens == [(prompt_len + 3, prompt_len + 2)]
    assert out["node_id"] == node_id
    assert out["raw_index"] == 3
    assert out["token_id"] == raw_ids[3]
    assert out["token_text"] == s.tokenizer.decode([raw_ids[3]])
    assert out["steering"] is None
    assert set(out["readout"]) == {0, 1}  # fitted sources of the 3-layer toy
    for rows in out["readout"].values():
        assert len(rows) == 4
        # Per-layer rows and the aggregate share one unit: the readout
        # probability, so ``strength`` is literally the mean of these.
        tok, p_l, tid = rows[0]
        assert isinstance(tok, str) and 0.0 <= p_l <= 1.0
        assert isinstance(tid, int)
    # The aggregate block rides the same logits across both fitted layers.
    assert len(out["aggregate"]) == 4
    for tok, strength, com, spread in out["aggregate"]:
        assert isinstance(tok, str)
        assert 0.0 <= strength <= 1.0
        assert 0.0 < com < 1 / (3 - 1)
        assert spread > 0.0
    # Continue-mode rebuild (cast model): no text resend — the history walk
    # to the node's parent carries the user turn; the gen header opens the
    # node's own seat.
    assert s.prepare_calls[0]["input"] is None
    assert s.prepare_calls[0]["raw"] is False
    assert s.prepare_calls[0]["gen_seat"] == "assistant"
    assert s.prepare_calls[0]["system_prompt"] == "Saved pirate instruction"
    node = s.tree.get(node_id)
    assert s.prepare_calls[0]["parent_node_id"] == node.parent_id


def test_jlens_token_readout_index_zero_reads_prompt_only() -> None:
    s = TreeLensHarness()
    s.lens.fit(_PROMPTS)
    node_id = _tree_with_assistant(s, s.tokenizer.encode("abc"))

    seen_lens: list[tuple[int, int | None]] = []
    import drowse.fit.capture as _vectors

    real_capture = _vectors._capture_all_hidden_states

    def _spy(model: Any, layers: Any, ids: torch.Tensor, **kw: Any) -> Any:
        pool = kw.get("pool_index")
        seen_lens.append((int(ids.shape[1]), int(pool) if pool is not None else None))
        return real_capture(model, layers, ids, **kw)

    _vectors._capture_all_hidden_states = _spy
    try:
        s.lens._replay_readout(node_id, 0, top_k=2)
    finally:
        _vectors._capture_all_hidden_states = real_capture
    prompt_len = len(s.tokenizer.encode(_PROMPT_RENDER))
    assert seen_lens == [(prompt_len, prompt_len - 1)]


def test_jlens_token_readout_steering_scope() -> None:
    s = TreeLensHarness()
    s.lens.fit(_PROMPTS)
    recipe = Recipe(steering="0.3 formal.casual", thinking=False)
    node_id = _tree_with_assistant(s, s.tokenizer.encode("abcd"), recipe)

    out = s.lens._replay_readout(node_id, 2, top_k=2)
    assert s.steering_scopes == ["0.3 formal.casual"]
    assert out["steering"] == "0.3 formal.casual"

    s.steering_scopes.clear()
    out = s.lens._replay_readout(node_id, 2, top_k=2, apply_steering=False)
    assert s.steering_scopes == []
    assert out["steering"] is None


def test_jlens_token_readout_raw_mode_render() -> None:
    s = TreeLensHarness()
    s.lens.fit(_PROMPTS)
    node_id = _tree_with_assistant(s, s.tokenizer.encode("abcd"))

    s.lens._replay_readout(node_id, 1, top_k=2, raw=True)
    call = s.prepare_calls[0]
    assert call["raw"] is True and call["input"] == ""
    # raw render anchors at the assistant node's parent (the flat prefix)
    assert call["parent_node_id"] == s.tree.get(node_id).parent_id


def test_jlens_token_readout_errors() -> None:
    s = TreeLensHarness()
    node_id = _tree_with_assistant(s, s.tokenizer.encode("abc"))

    with pytest.raises(LensNotFittedError):
        s.lens._replay_readout(node_id, 0)

    s.lens.fit(_PROMPTS)
    user_id = s.tree.get(node_id).parent_id
    assert user_id is not None
    with pytest.raises(UnknownNodeError):
        s.lens._replay_readout("nope", 0)
    # A committed user turn is a valid *shape* under the cast model (user-
    # seat generated nodes are forkable/readable) — but this one has no
    # decode record, so it fails on that instead of its role.
    with pytest.raises(InvalidNodeOperationError, match="no raw token record"):
        s.lens._replay_readout(user_id, 0)
    # The system root stays out of bounds — not a turn at all.
    assert s.tree.root_id is not None
    with pytest.raises(InvalidNodeOperationError, match="only a turn"):
        s.lens._replay_readout(s.tree.root_id, 0)
    with pytest.raises(InvalidNodeOperationError, match="out of range"):
        s.lens._replay_readout(node_id, 3)
    with pytest.raises(InvalidNodeOperationError, match="out of range"):
        s.lens._replay_readout(node_id, -1)
    with pytest.raises(ValueError, match="not in the fitted lens"):
        s.lens._replay_readout(node_id, 0, layers=[9])

    bare = s.tree.begin_assistant(user_id)
    s.tree.finalize_assistant(bare, text="no raw record", finish_reason="stop")
    with pytest.raises(InvalidNodeOperationError, match="no raw token record"):
        s.lens._replay_readout(bare, 0)


def test_score_probes_entries_are_disjoint_and_guarded() -> None:
    """Two entries, no placeholder argument: ``score_probes`` takes capture
    slices, ``score_probes_from_rows`` takes a precomputed matrix.  The
    logits-xor-probabilities rule is a real guard, so ``python -O`` behaves
    identically to a normal run."""
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    s.lens.attach("jlens/g", as_name=None)
    inst = s.lens
    layers = [int(layer) for layer in s.lens.resident().source_layers]
    d_model = next(iter(s.lens.resident().jacobians.values())).shape[0]
    hidden = {
        layer: torch.randn(
            d_model, generator=torch.Generator().manual_seed(layer),
        )
        for layer in layers
    }

    from drowse.lens.jlens import readout_probabilities

    logits = s.lens.runtime.logits_rows(
        s.lens.resident(), [(layer, hidden[layer]) for layer in layers],
    )
    from_slices = inst.score_probes(hidden)
    from_logits = inst.score_probes_from_rows(layers=layers, logits=logits)
    from_probs = inst.score_probes_from_rows(
        layers=layers, probabilities=readout_probabilities(logits),
    )
    assert from_slices["jlens/g"].value == pytest.approx(
        from_logits["jlens/g"].value,
    )
    assert from_logits["jlens/g"].value == pytest.approx(
        from_probs["jlens/g"].value,
    )

    with pytest.raises(ValueError, match="logits OR probabilities"):
        inst.score_probes_from_rows(layers=layers)
    with pytest.raises(ValueError, match="logits OR probabilities"):
        inst.score_probes_from_rows(
            layers=layers, logits=logits,
            probabilities=readout_probabilities(logits),
        )


def test_lens_probe_hash_tracks_the_resident_lens_identity() -> None:
    """The same word read through a different lens is a different probe.

    Transcripts and loom recipes compare probe hashes to decide whether a
    recorded reading is reproducible; a lens swap or re-fit must show up.
    """
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    name = s.lens.attach("jlens/g", as_name=None)
    first = s.lens.probe_hash(name)
    assert first is not None
    assert s.lens.probe_hash(name) == first

    s.lens.runtime.identity = ("another", "lens")
    assert s.lens.probe_hash(name) != first


def test_cuda_transport_scopes_tf32_to_the_matmul(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabling the live lens must not flip process-global matmul precision."""
    from drowse.lens.jlens import _scoped_tf32

    before = torch.get_float32_matmul_precision()
    with _scoped_tf32(torch.device("cuda")):
        assert torch.get_float32_matmul_precision() == "high"
    assert torch.get_float32_matmul_precision() == before
    with _scoped_tf32(torch.device("cpu")):
        assert torch.get_float32_matmul_precision() == before

    set_calls: list[str] = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", set_calls.append)
    s = LensHarness()
    s.lens.fit(_PROMPTS)
    runtime = s.lens.runtime
    runtime.device = torch.device("cuda")  # the stack itself still lands on CPU
    real_stack = runtime.transport_stack
    monkeypatch.setattr(
        runtime, "transport_stack",
        lambda lens, layers, _device=None: real_stack(
            lens, layers, torch.device("cpu"),
        ),
    )
    s.lens.enable_live()
    assert set_calls == []
