"""The generation engine: transaction, call resolution, token tap, stream,
compiled routing, prefix cache, teardown.

Unit tests drive :class:`GenerationEngine`, :class:`TokenTap`,
:class:`CompiledRuntime` and :class:`PrefixCache` with narrow fakes; the
integration tests run real generations on the tiny CPU session.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from drowse.errors import ConcurrentGenerationError, GenerationRequestError
from drowse.generation.compile_runtime import CompiledRuntime
from drowse.generation.decode import GenerationConfig, GenerationState
from drowse.generation.prefix_cache import PrefixCache
from drowse.generation.results import GenerationResult, TokenAlt, TokenEvent
from drowse.generation.sampling import InheritSystemPrompt, SamplingConfig
from drowse.generation.token_callback import TokenConsumer, TokenConsumerOptions
from drowse.generation.types import GenState
from drowse.readout.types import ScalarReading
from drowse.session.engine import (
    ConsumerPlan,
    GenerationEngine,
    PrefixCacheStateError,
    TokenTap,
)
from drowse.steering.terms import Steering
from tests.support.sae import make_resident
from tests.support.sessions import TinySessionFactory

_FOUR = SamplingConfig(max_tokens=4, seed=0)


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------


def _bare_engine(session: Any = None, *, release: Any = None) -> GenerationEngine:
    return GenerationEngine(
        cast(Any, session if session is not None else SimpleNamespace()),
        lock=threading.RLock(),
        runtime=CompiledRuntime(),
        steering=cast(Any, SimpleNamespace()),
        reads=cast(Any, SimpleNamespace()),
        release=release or (lambda: None),
    )


def test_transaction_releases_when_the_body_never_does() -> None:
    """A BaseException before the body's own teardown still runs the
    session release (capture detach) and frees the phase and the lock."""
    released: list[int] = []
    engine = _bare_engine(release=lambda: released.append(1))

    with pytest.raises(KeyboardInterrupt):
        with engine.transaction():
            assert engine.phase is GenState.PREAMBLE
            raise KeyboardInterrupt

    assert released == [1]
    assert engine.phase is GenState.IDLE
    assert engine._lock.acquire(blocking=False)
    engine._lock.release()


class _RecordingScope:
    """A steering scope stand-in recording each teardown's ``swallow`` flag."""

    def __init__(self, exits: list[bool]) -> None:
        self._exits = exits

    def close(self, *, internal: bool = False, swallow: bool = False) -> None:
        assert internal
        self._exits.append(swallow)


def test_transaction_pops_a_scope_the_body_left_open() -> None:
    """A scope still open on failure is popped with ``swallow=True`` so a
    teardown error can't mask the failure already propagating."""
    exits: list[bool] = []
    engine = _bare_engine()

    with pytest.raises(RuntimeError):
        with engine.transaction() as txn:
            txn.steering_cm = _RecordingScope(exits)
            raise RuntimeError("preamble blew up")

    assert exits == [True]


def test_transaction_leaves_a_popped_scope_alone() -> None:
    exits: list[bool] = []
    engine = _bare_engine()

    with engine.transaction() as txn:
        txn.steering_cm = _RecordingScope(exits)
        txn.steering_cm = None

    assert exits == []
    assert engine.phase is GenState.IDLE


def test_transaction_rejects_a_reentrant_generation() -> None:
    """The reentrant lock lets the same thread back in; the phase guard is
    what rejects it."""
    engine = _bare_engine()
    with engine.transaction():
        with pytest.raises(ConcurrentGenerationError):
            with engine.transaction():
                pass  # pragma: no cover - the guard raises on entry
    assert engine.phase is GenState.IDLE


def test_transaction_teardown_frees_the_loom_reservation(tiny_session: Any) -> None:
    """The session release frees the loom reservation along with the phase
    and the lock, even when the body never reached its own teardown."""
    tree = tiny_session.tree
    with pytest.raises(KeyboardInterrupt):
        with tiny_session._engine.transaction():
            tree.reserve(tree.root_id)
            raise KeyboardInterrupt

    assert tree.reservation is None
    assert tiny_session._engine.phase is GenState.IDLE
    assert tiny_session._gen_lock.acquire(blocking=False)
    tiny_session._gen_lock.release()


def test_transaction_release_failure_still_frees_phase_and_lock() -> None:
    def _boom() -> None:
        raise RuntimeError("detach failed")

    engine = _bare_engine(release=_boom)
    engine.compiled_clean_eligible = True
    with pytest.raises(RuntimeError, match="detach failed"):
        with engine.transaction():
            pass
    assert engine.phase is GenState.IDLE
    assert engine.compiled_clean_eligible is False
    assert engine._lock.acquire(blocking=False)
    engine._lock.release()


# ---------------------------------------------------------------------------
# Call resolution
# ---------------------------------------------------------------------------


def _resolver(*, config: GenerationConfig, supports_thinking: bool = True,
              default_return_top_k: int = 0) -> GenerationEngine:
    session = SimpleNamespace(
        config=config,
        registry=SimpleNamespace(profiles={}),
        chat_traits=SimpleNamespace(supports_thinking=supports_thinking),
        tree=SimpleNamespace(cast={}),
        renderer=SimpleNamespace(
            seat_stop_strings=lambda stop, **_kw: stop,
            resolve_system_prompt=lambda prompt: (
                config.system_prompt
                if isinstance(prompt, InheritSystemPrompt) else prompt
            ),
        ),
    )
    engine = _bare_engine(session)
    engine.default_return_top_k = default_return_top_k
    return engine


def test_resolve_call_uses_session_thinking_default() -> None:
    engine = _resolver(config=GenerationConfig(thinking=False))

    assert engine.resolve_call("x", stateless=True).use_thinking_req is False
    assert engine.resolve_call(
        "x", stateless=True, steering=Steering(alphas={}, thinking=True),
    ).use_thinking_req is True
    assert engine.resolve_call(
        "x", stateless=True, thinking=True,
    ).use_thinking_req is True


def test_resolve_call_derives_every_sampling_field_once() -> None:
    engine = _resolver(config=GenerationConfig(max_new_tokens=32, system_prompt="sys"))
    spec = engine.resolve_call(
        "x",
        stateless=True,
        steering="",
        sampling=SamplingConfig(
            max_tokens=5, temperature=0.4, logprobs=2, return_top_k=3,
            stop=("END",), user_role="u", assistant_role="a",
            return_hidden=True, return_probe_readings=False,
            persist_per_layer_scores=True,
        ),
    )
    assert spec.steering_obj is None  # "" composes to unsteered
    assert spec.system_prompt == "sys"  # the inherit sentinel is resolved
    assert spec.gen_config.max_new_tokens == 5
    assert spec.gen_config.temperature == 0.4
    assert spec.lp_count == 3 and spec.collect_logprobs is True
    assert spec.stop_list == ["END"]
    assert (spec.user_role, spec.assistant_role) == ("u", "a")
    assert spec.want_hidden is True and spec.return_probe_readings is False
    assert spec.persist_per_layer_scores is True
    assert spec.readout_top_k == 3

    bare = engine.resolve_call("x", stateless=True, system_prompt=None)
    assert bare.system_prompt is None
    assert bare.return_probe_readings is True and bare.collect_logprobs is False
    assert bare.lp_count is None and bare.stop_list is None


def test_readout_top_k_shares_logit_alternative_width() -> None:
    engine = _resolver(config=GenerationConfig(), default_return_top_k=6)

    assert engine.return_top_k(None) == 6
    assert engine.return_top_k(SamplingConfig(return_top_k=13)) == 13
    # Per-call zero inherits the same session default as the logit pass.
    assert engine.return_top_k(SamplingConfig(return_top_k=0)) == 6
    assert engine.readout_top_k(SamplingConfig(return_top_k=13)) == 13

    # Read-side discovery stays eight wide when alts are disabled.
    engine.default_return_top_k = 0
    assert engine.readout_top_k(None) == 8


# ---------------------------------------------------------------------------
# Token tap
# ---------------------------------------------------------------------------


#: A read plane whose geometry reading must never be asked for.
_NO_GEOMETRY: Any = SimpleNamespace(
    geometry_step_readings=lambda _step: (_ for _ in ()).throw(
        AssertionError("geometry scoring must not run for this generation"),
    ),
)


def _plan(**overrides: Any) -> ConsumerPlan:
    base: dict[str, Any] = dict(
        live_scores_on=True, has_monitor_probes=False,
        wants_live_token_scores=False, persists_layer_scores=False,
        persists_subspace_coords=False, persists_probe_row=False,
        lens=False, sae=False, perplexity_requested=False, need_tap=True,
    )
    base.update(overrides)
    return ConsumerPlan(**base)


def _forbid(name: str) -> Any:
    def _raise(*_a: Any, **_k: Any) -> Any:
        raise AssertionError(f"{name} must not run for this generation")
    return _raise


def _tap_session(*, lens_live: Any = None, sae_live: Any = None,
                 lens_step: Any = None, rows: list[Any] | None = None) -> Any:
    return SimpleNamespace(
        lens=SimpleNamespace(
            live=lens_live, last_step_readings=None,
            live_readout_step=lens_step or _forbid("lens readout"),
        ),
        sae=SimpleNamespace(
            live=sae_live, last_step_readings=None,
            live_readout_step=_forbid("sae readout"),
        ),
        tree=SimpleNamespace(
            append_token=lambda node_id, row, *, thinking: (rows if rows is not None else []).append(
                (node_id, row, thinking),
            ),
        ),
    )


@pytest.mark.parametrize("collect_logprobs", [False, True])
@pytest.mark.parametrize("token_logprobs", [(None,), (-0.4,), (None, -0.4)])
def test_text_only_tap_skips_unconsumed_readouts_and_payloads(
    collect_logprobs: bool, token_logprobs: tuple[float | None, ...],
) -> None:
    """A live lens/SAE the consumer didn't ask for, and a monitor with no
    probes, cost the tap nothing; the logprob rollup still runs."""
    seen: list[tuple[Any, ...]] = []
    tap = TokenTap(
        cast(Any, _tap_session(lens_live={"layers": [0]}, sae_live={"layer": 0})),
        reads=_NO_GEOMETRY,
        state=GenerationState(),
        consumer=lambda *args: seen.append(args),
        plan=_plan(),
        assistant_node_id=None,
        collect_logprobs=collect_logprobs,
        readout_top_k=8,
        recipe_steering=None,
    )
    for lp in token_logprobs:
        tap("x", False, 1, lp, None, None, 0)

    assert seen == [("x", False, 1, lp, None, None) for lp in token_logprobs]
    assert tap.logprobs == (
        [(1, lp, []) for lp in token_logprobs] if collect_logprobs else None
    )
    if None in token_logprobs:
        assert tap.mean_logprob() == (None, None)
    else:
        assert tap.mean_logprob() == (pytest.approx(-0.4), pytest.approx(0.4))


def test_tap_hands_a_consumer_one_event_matching_the_loom_row() -> None:
    """The internal consumer gets the token's envelope and raw index on the
    event itself, and the loom row stores the very same envelope."""
    rows: list[Any] = []
    lens_rows = {3: [(" a", 0.5)]}

    def _lens_step(*, top_k: int, step_id: int) -> Any:
        assert (top_k, step_id) == (8, 4)
        return lens_rows, [(" a", 0.4, 0.5, 0.1)], {3: [7]}

    session = _tap_session(
        lens_live={"layers": [3], "source": "local:default"},
        lens_step=_lens_step, rows=rows,
    )
    session.lens.last_step_readings = {
        "jlens/a": ScalarReading(value=0.25, unit="probability"),
    }
    state = GenerationState()
    state.emit_map = [(4, False)]
    events: list[TokenEvent] = []
    tap = TokenTap(
        cast(Any, session),
        reads=_NO_GEOMETRY,
        state=state,
        consumer=TokenConsumer(
            events.append, TokenConsumerOptions(lens_readout=True, probe_readings=True),
        ),
        plan=_plan(lens=True),
        assistant_node_id="node-1",
        collect_logprobs=False,
        readout_top_k=8,
        recipe_steering="0.3 calm",
    )
    alts = [TokenAlt(id=7, text=" a", logprob=-0.1)]
    tap(" a", False, 7, -0.1, alts, 1.5, 4)

    (event,) = events
    assert (event.text, event.token_id, event.index, event.raw_index) == (" a", 7, 0, 4)
    assert event.top_alts == alts and event.perplexity == 1.5
    assert event.measurements is not None and event.probe_readings is not None
    lens = event.measurements["instruments"]["lens"]
    assert lens["binding"] == {"source": "local:default", "steering": "0.3 calm"}
    assert lens["readings"]["jlens/a"]["value"] == 0.25
    assert set(event.probe_readings) == {"jlens/a"}

    ((node_id, row, thinking),) = rows
    assert (node_id, thinking) == ("node-1", False)
    assert row["measurements"] is event.measurements
    assert row["raw_index"] == 4
    assert row["top_alts"] == [{"id": 7, "text": " a", "logprob": -0.1}]
    assert row["probes"] == {"jlens/a": 0.25}


def test_tap_skips_the_compat_projection_unless_the_consumer_asks() -> None:
    session = _tap_session(
        lens_live={"layers": [3], "source": "local:default"},
        lens_step=lambda **_kw: ({3: [(" a", 0.5)]}, [], {3: [7]}),
    )
    session.lens.last_step_readings = {
        "jlens/a": ScalarReading(value=0.25, unit="probability"),
    }
    events: list[TokenEvent] = []
    tap = TokenTap(
        cast(Any, session),
        reads=_NO_GEOMETRY,
        state=GenerationState(),
        consumer=TokenConsumer(events.append, TokenConsumerOptions(lens_readout=True)),
        plan=_plan(lens=True),
        assistant_node_id=None,
        collect_logprobs=False,
        readout_top_k=8,
        recipe_steering=None,
    )
    tap(" a", False, 7, None, None, None, 0)
    assert events[0].probe_readings is None
    assert events[0].measurements is not None and events[0].raw_index is None


@pytest.mark.parametrize("mode", ["INCREMENTAL", "LEAN_INCREMENTAL"])
def test_tap_reuses_the_capture_sinks_row_instead_of_rescoring(mode: str) -> None:
    from drowse.readout.planner import CaptureMode, CaptureState
    from drowse.readout.types import ProbeReading
    from tests.support.read_plane import ReadRig

    reading = ProbeReading(fraction=0.4, nearest=[("calm", 0.2)], coords=(0.75,))
    session = _tap_session()
    reads = ReadRig(
        capture=SimpleNamespace(per_layer_buckets=_forbid("geometry scoring")),
        geometry=SimpleNamespace(current_run=SimpleNamespace(observe=_forbid("observe"))),
    ).plane
    reads.state = CaptureState(mode=CaptureMode[mode])
    reads._rows = [{"toy": reading}]
    events: list[TokenEvent] = []
    tap = TokenTap(
        cast(Any, session),
        reads=reads,
        state=GenerationState(),
        consumer=TokenConsumer(
            events.append, TokenConsumerOptions(live_scores=True, probe_readings=True),
        ),
        plan=_plan(has_monitor_probes=True, wants_live_token_scores=True),
        assistant_node_id=None,
        collect_logprobs=False,
        readout_top_k=8,
        recipe_steering=None,
    )
    tap("x", False, 1, None, None, None, 0)
    assert events[0].probe_readings == {"toy": reading}
    assert events[0].measurements is not None
    assert events[0].measurements["scores"] == {"toy": pytest.approx(0.75)}


# ---------------------------------------------------------------------------
# Real generations (tiny CPU session)
# ---------------------------------------------------------------------------


def test_serial_fan_runs_every_sibling(make_tiny_session: TinySessionFactory) -> None:
    """A finished generation must not leave the stop flag set: the serial fan
    reads it between siblings, so ``n=3`` returned one result."""
    session, _script = make_tiny_session.scripted("Hi there.")
    runs = session.generate(
        "Say hi.", n=3, sampling=SamplingConfig(max_tokens=4, seed=1, temperature=0.7),
    )
    assert len(runs) == 3
    assert len(set(runs.node_ids)) == 3
    assert not session.generation_state.is_stop_requested()


def test_generate_rejects_a_non_positive_fan(tiny_session: Any) -> None:
    with pytest.raises(GenerationRequestError, match="n must be >= 1"):
        tiny_session.generate("hi", n=0)
    assert GenerationRequestError("x").user_message()[0] == 400


def test_score_choices_types_a_bad_choice_set(tiny_session: Any) -> None:
    with pytest.raises(GenerationRequestError):
        tiny_session.score_choices([{"role": "user", "content": "day?"}], [])
    with pytest.raises(GenerationRequestError):
        tiny_session.score_choices(
            [{"role": "user", "content": "day?"}], ["a", "b"], labels=["a"],
        )


def test_stream_yields_tap_events_and_exposes_the_result(
    make_tiny_session: TinySessionFactory,
) -> None:
    session, _script = make_tiny_session.scripted("Hi there.")
    stream = session.generate_stream("Say hi.", sampling=_FOUR)
    events = list(stream)

    assert "".join(e.text for e in events) == "Hi t"
    assert [e.index for e in events] == [0, 1, 2, 3]
    assert [e.raw_index for e in events] == [0, 1, 2, 3]
    assert stream.result is session.last_result
    assert stream.result is not None and stream.result.text == "Hi t"
    # Every streamed token landed on the loom row with the same raw index.
    node = session.tree.get(session.tree.active_node_id)
    assert [row["raw_index"] for row in node.tokens] == [0, 1, 2, 3]


def test_stream_options_follow_live_scores_and_live_readouts(
    tiny_session: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = GenerationResult(text="ok", tokens=[7], token_count=1, tok_per_sec=1.0, elapsed=1.0)
    envelope = {"version": 1, "scope": "token", "instruments": {}}
    seen: list[TokenConsumerOptions] = []

    def _fake_generate_one(_input: Any, *, on_token: Any, **_kw: Any) -> GenerationResult:
        seen.append(on_token.options)
        on_token.on_event(TokenEvent(text="ok", token_id=7, index=0, measurements=envelope))
        return result

    monkeypatch.setattr(tiny_session._engine, "generate_one", _fake_generate_one)
    stream = tiny_session.generate_stream("p", live_scores=True, live_readouts=False)
    events = list(stream)

    assert (seen[0].lens_readout, seen[0].sae_readout) == (False, False)
    assert (seen[0].live_scores, seen[0].probe_readings) == (True, True)
    assert events[0].measurements is envelope
    assert stream.result is result

    list(tiny_session.generate_stream("p", live_scores=False))
    assert (seen[1].live_scores, seen[1].probe_readings) == (False, False)
    assert (seen[1].lens_readout, seen[1].sae_readout) == (True, True)



def test_close_clears_chat_render_cache(make_tiny_session: TinySessionFactory) -> None:
    session, _script = make_tiny_session.scripted("Hi there.")
    session.generate("Say hi.", sampling=_FOUR)
    assert session.chat_traits._render_cache
    session.close()
    assert not session.chat_traits._render_cache
    assert session.chat_traits._token_text == {}
    assert session.last_result is None


# ---------------------------------------------------------------------------
# Compiled runtime
# ---------------------------------------------------------------------------


def test_compiled_graph_owner_is_thread_local() -> None:
    runtime = CompiledRuntime(cuda_graphs_active=True)
    assert runtime.graph_thread_safe() is True

    observed: list[bool] = []
    worker = threading.Thread(target=lambda: observed.append(runtime.graph_thread_safe()))
    worker.start()
    worker.join()
    assert observed == [False]
    assert CompiledRuntime().graph_thread_safe() is True


class _FakeModel:
    def __init__(self, device: str) -> None:
        self.config = SimpleNamespace(hidden_size=4)
        self._param = SimpleNamespace(device=torch.device(device), dtype=torch.float32)

    def parameters(self) -> Any:
        yield self._param


class _FakeCompiled(_FakeModel):
    def __init__(self, base: _FakeModel) -> None:
        super().__init__(str(base._param.device))
        self._orig_mod = base


@pytest.mark.parametrize("device", ["mps", "cuda"])
@pytest.mark.parametrize("persistent_capture", [False, True])
def test_prepare_installs_capture_hooks_only_for_a_probed_session(
    device: str, persistent_capture: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``probes=[]`` takes the no-capture compiled mode: the offset hooks
    are installed (and handed to steering), the capture hooks are not."""
    from drowse.generation import compile_runtime
    from drowse.model import static_cache

    calls = {"offset": 0, "capture": 0}
    handle = SimpleNamespace(remove=lambda: None)

    def _offset(*_a: Any, **_k: Any) -> Any:
        calls["offset"] += 1
        return {0: torch.zeros(4)}, [handle]

    def _capture(*_a: Any, **_k: Any) -> Any:
        calls["capture"] += 1
        return {0: torch.zeros(4)}, [handle]

    monkeypatch.setattr(compile_runtime, "get_layers", lambda _m: [object()])
    monkeypatch.setattr(compile_runtime.hooks, "install_persistent_offset_hooks", _offset)
    monkeypatch.setattr(
        compile_runtime.capture_buffer, "install_persistent_capture_hooks", _capture,
    )
    monkeypatch.setattr(
        compile_runtime.compile_mod, "compile_with_probe",
        lambda model, _tok, _dev, mode: _FakeCompiled(model),
    )
    monkeypatch.setattr(static_cache, "is_static_cache_supported", lambda _m, _d: (True, None))

    model, runtime = CompiledRuntime.prepare(
        _FakeModel(device), object(),
        compile=True, compile_mode=None, cuda_graphs=False,
        persistent_capture=persistent_capture,
    )

    assert isinstance(model, _FakeCompiled)
    assert runtime.compiled and runtime.static_cache_active
    assert calls == {"offset": 1, "capture": int(persistent_capture)}
    assert bool(runtime.capture_buffers) is persistent_capture
    assert runtime.take_offset_hooks() is not None
    assert runtime.take_offset_hooks() is None


def test_prepare_drops_the_hooks_when_compile_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.generation import compile_runtime

    removed: list[str] = []

    def _installer(kind: str) -> Any:
        return lambda *_a, **_k: (
            {0: torch.zeros(4)}, [SimpleNamespace(remove=lambda: removed.append(kind))],
        )

    monkeypatch.setattr(compile_runtime, "get_layers", lambda _m: [object()])
    monkeypatch.setattr(
        compile_runtime.hooks, "install_persistent_offset_hooks", _installer("offset"),
    )
    monkeypatch.setattr(
        compile_runtime.capture_buffer, "install_persistent_capture_hooks",
        _installer("capture"),
    )
    monkeypatch.setattr(
        compile_runtime.compile_mod, "compile_with_probe",
        lambda model, _tok, _dev, mode: model,  # eager fallback
    )
    _model, runtime = CompiledRuntime.prepare(
        _FakeModel("mps"), object(),
        compile=True, compile_mode=None, cuda_graphs=False, persistent_capture=True,
    )
    assert not runtime.compiled
    assert sorted(removed) == ["capture", "offset"]
    assert runtime.capture_buffers == {} and runtime.take_offset_hooks() is None


def test_runtime_reuses_resets_and_releases_the_generation_static_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.model import static_cache

    class _Cache:
        def __init__(self, capacity: int) -> None:
            self.capacity = capacity
            self.resets = 0

        def reset(self) -> None:
            self.resets += 1

    made: list[_Cache] = []

    def _make(_model: Any, *, max_cache_len: int, device: Any, dtype: Any) -> _Cache:
        del device, dtype
        made.append(_Cache(max_cache_len))
        return made[-1]

    monkeypatch.setattr(static_cache, "make_static_cache", _make)
    runtime = CompiledRuntime()
    model = _FakeModel("cpu")
    cpu = torch.device("cpu")

    first = runtime.acquire_static_cache(model, max_cache_len=512, device=cpu)
    reused = runtime.acquire_static_cache(model, max_cache_len=256, device=cpu)
    grown = runtime.acquire_static_cache(model, max_cache_len=768, device=cpu)

    assert reused is first and cast(_Cache, first).resets == 1
    assert grown is not first
    assert [cache.capacity for cache in made] == [512, 768]

    runtime.close()
    assert runtime._static_cache is None and runtime._static_cache_len == 0


def test_session_close_removes_both_persistent_hook_families(
    tiny_session: Any,
) -> None:
    """``from_pretrained`` installs permanent offset + capture hooks on the
    caller's model; ``close()`` is the only symmetric teardown, and a model
    handed to ``DrowseSession`` outlives the session."""
    from drowse.readout.capture_buffer import install_persistent_capture_hooks
    from drowse.session.session import DrowseSession
    from drowse.steering.hooks import install_persistent_offset_hooks

    model, tokenizer = tiny_session.model, tiny_session.tokenizer
    layers = tiny_session.layers
    hidden = int(tiny_session.model_info["hidden_dim"])
    cpu, dtype = torch.device("cpu"), torch.float32
    before = [len(layer._forward_hooks) for layer in layers]
    offsets, offset_handles = install_persistent_offset_hooks(layers, hidden, cpu, dtype)
    captures, capture_handles = install_persistent_capture_hooks(layers, hidden, cpu, dtype)
    runtime = CompiledRuntime(
        compiled=True,
        capture_buffers=captures, capture_handles=capture_handles,
        offset_buffers=offsets, offset_handles=offset_handles,
    )
    session = DrowseSession(model, tokenizer, probes=[], runtime=runtime)
    assert session._steering.has_compiled_offsets()
    assert [len(layer._forward_hooks) for layer in layers] == [n + 2 for n in before]
    session.cache_prefix(torch.tensor([[1, 2, 3]]))
    assert session._engine.prefix_cache
    # Lens and SAE residency (and the lens's derived caches) are dropped too.
    lens_runtime = session._lens_instrument.runtime
    lens_runtime.transport_stack(
        cast(Any, SimpleNamespace(jacobians={0: torch.eye(2)})), [0], cpu,
    )
    lens_runtime._decoded.put(1, "one")
    lens_runtime.lens = cast(Any, object())
    make_resident(session.sae.runtime, layer=0)
    session.events.subscribe(lambda _event: None)

    session.close()

    assert [len(layer._forward_hooks) for layer in layers] == before
    assert session._steering.has_compiled_offsets() is False
    assert runtime.capture_buffers == {}
    assert not session._engine.prefix_cache
    assert len(lens_runtime._stacks) == 0
    assert len(lens_runtime._decoded) == 0
    assert session._lens_instrument.resident_lens is None
    assert session.sae.runtime.resident is None
    assert session.events._subs == []
    session.close()  # idempotent


# ---------------------------------------------------------------------------
# Prefix cache
# ---------------------------------------------------------------------------


def test_prefix_cache_hits_then_rewinds_a_real_dynamic_cache(tiny_session: Any) -> None:
    model = tiny_session.model
    cache = PrefixCache()
    prefix = torch.tensor([[5, 6, 7, 8]])
    assert cache.build(model, prefix) == 4
    entry = cache.entry
    assert entry is not None and not entry.static

    assert cache.hit(torch.tensor([[5, 6, 7]]), static_eligible=False,
                     required_max_new_tokens=4) is None  # no suffix
    assert cache.hit(torch.tensor([[5, 6, 9, 8, 1]]), static_eligible=False,
                     required_max_new_tokens=4) is None  # mismatch
    hit = cache.hit(torch.tensor([[5, 6, 7, 8, 9, 10]]), static_eligible=False,
                    required_max_new_tokens=4)
    assert hit is not None and hit.prefix_len == 4
    assert hit.suffix_ids.tolist() == [[9, 10]]

    with torch.inference_mode():
        model(input_ids=hit.suffix_ids, past_key_values=hit.past_key_values, use_cache=True)
    assert hit.past_key_values.get_seq_length() == 6
    cache.release(hit)
    assert cache.entry is entry
    assert entry.past_key_values.get_seq_length() == 4

    assert cache.build(model, torch.zeros((1, 0), dtype=torch.long)) == 0
    assert not cache


def test_cached_prefix_generation_matches_full_prefill(tiny_session: Any) -> None:
    messages = [{"role": "user", "content": "A fixed shared instruction prefix."}]
    greedy = SamplingConfig(max_tokens=6, temperature=0.0)
    full = tiny_session.generate(messages, sampling=greedy, stateless=True).first

    assert tiny_session.cache_prefix(messages) > 0
    hit_calls: list[Any] = []
    original_hit = tiny_session._engine.prefix_cache.hit

    def _spy(*args: Any, **kwargs: Any) -> Any:
        result = original_hit(*args, **kwargs)
        hit_calls.append(result)
        return result

    tiny_session._engine.prefix_cache.hit = _spy
    cached = tiny_session.generate(messages, sampling=greedy, stateless=True).first
    again = tiny_session.generate(messages, sampling=greedy, stateless=True).first

    assert all(hit is not None for hit in hit_calls) and len(hit_calls) == 2
    assert cached.tokens == full.tokens == again.tokens


def test_registry_changes_keep_the_unsteered_prefix(tiny_session: Any) -> None:
    """Registering, replacing or dropping artifacts never touches a prefill.

    The cached prefix is the *unsteered* KV (``cache_prefix`` refuses inside a
    scope), and each generation decides reuse itself through
    ``steering_active_in_prefill``.  So registry mutations keep the entry, and
    a generation reusing it, unsteered or under a response-only term,
    decodes exactly what a full prefill does.
    """
    from tests.support.sessions import register_vectors

    messages = [{"role": "user", "content": "A fixed shared instruction prefix."}]
    greedy = SamplingConfig(max_tokens=6, temperature=0.0)
    register_vectors(tiny_session, "a")
    full = tiny_session.generate(messages, sampling=greedy, stateless=True).first
    steered_full = tiny_session.generate(
        messages, sampling=greedy, stateless=True, steering="0.5 a@response",
    ).first

    assert tiny_session.cache_prefix(messages) > 0
    entry = tiny_session._engine.prefix_cache.entry
    register_vectors(tiny_session, "b")
    register_vectors(tiny_session, "a")  # replaced in place
    tiny_session.registry.pop_profile("b")
    tiny_session.evict_artifact("local", "b")
    assert tiny_session._engine.prefix_cache.entry is entry

    hit_calls: list[Any] = []
    original_hit = tiny_session._engine.prefix_cache.hit

    def _spy(*args: Any, **kwargs: Any) -> Any:
        result = original_hit(*args, **kwargs)
        hit_calls.append(result)
        return result

    tiny_session._engine.prefix_cache.hit = _spy
    cached = tiny_session.generate(messages, sampling=greedy, stateless=True).first
    steered = tiny_session.generate(
        messages, sampling=greedy, stateless=True, steering="0.5 a@response",
    ).first
    assert len(hit_calls) == 2 and all(hit is not None for hit in hit_calls)
    assert cached.tokens == full.tokens
    assert steered.tokens == steered_full.tokens


def test_cache_prefix_refuses_inside_a_steering_scope(tiny_session: Any) -> None:
    from tests.support.sessions import register_vectors

    register_vectors(tiny_session, "a")
    with tiny_session.steering("0.5 a"):
        with pytest.raises(PrefixCacheStateError) as excinfo:
            tiny_session.cache_prefix([{"role": "user", "content": "x"}])
    assert excinfo.value.user_message()[0] == 409
    assert tiny_session.cache_prefix(None) == 0


def test_resolve_call_continues_a_matching_leaf_in_place(
    make_tiny_session: TinySessionFactory,
) -> None:
    session, _script = make_tiny_session.scripted("Hi there.")
    session.generate("Say hi.", sampling=_FOUR)
    leaf = session.tree.get(session.tree.active_node_id)
    spec = session._engine.resolve_call(
        None, sampling=replace(_FOUR, max_tokens=3), append_same_role=True,
    )
    assert spec.continuation_node_id == leaf.id
    assert spec.history_anchor == leaf.parent_id
    assert spec.forced_prefix == session.tokenizer.encode(leaf.text, add_special_tokens=False)
    assert spec.gen_config.max_new_tokens == len(spec.forced_prefix) + 3
    assert spec.use_thinking_req is False


def _spec(**overrides: Any) -> Any:
    from drowse.generation.types import CallSpec

    fields: dict[str, Any] = dict(
        input="x", steering_obj=None, sampling=None, system_prompt=None,
        stateless=True, raw=False, gen_seat="assistant", parent_node_id=None,
        history_anchor=None, use_thinking_req=False,
        gen_config=GenerationConfig(max_new_tokens=2), lp_count=None, seed=None,
        stop_list=None, logit_bias=None, presence_penalty=0.0,
        frequency_penalty=0.0, collect_logprobs=False, readout_top_k=8,
        user_role=None, assistant_role=None, return_probe_readings=True,
        want_hidden=False, persist_per_layer_scores=False,
        persist_subspace_coords=False, forced_prefix=None,
    )
    fields.update(overrides)
    return CallSpec(**fields)


def _static_prefix(cache: PrefixCache, past_key_values: Any, *, max_cache_len: int) -> None:
    """Install a StaticCache-backed ``[1, 2]`` prefix entry."""
    cache.build(
        lambda **kw: SimpleNamespace(past_key_values=kw["past_key_values"]),
        torch.tensor([[1, 2]]),
        static_max_cache_len=max_cache_len,
        make_static=lambda _length: past_key_values,
    )
    assert cache.entry is not None and cache.entry.static


def test_static_prefix_hit_requires_static_eligibility_and_headroom() -> None:
    cache = PrefixCache()
    static = object()
    _static_prefix(cache, static, max_cache_len=5)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    assert cache.hit(ids, static_eligible=False, required_max_new_tokens=1) is None
    assert cache.hit(ids, static_eligible=True, required_max_new_tokens=3) is None
    hit = cache.hit(ids, static_eligible=True, required_max_new_tokens=2)
    assert hit is not None
    assert hit.suffix_ids.tolist() == [[3]]
    assert (hit.past_key_values, hit.prefix_len, hit.static) == (static, 2, True)


def test_decode_keeps_the_static_cache_on_a_static_prefix_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.session.engine as engine_mod

    class _Cache:
        # One static layer whose write counter the generation advanced past
        # the 2-token prefix.
        layers = [SimpleNamespace(cumulative_length=torch.tensor(4))]

    session = SimpleNamespace(
        _steering_composer=SimpleNamespace(
            steering_active_in_prefill=lambda: False,
            steering_needs_probe_gating=lambda: False,
        ),
        model=object(), tokenizer=object(), device=torch.device("cpu"),
        config=GenerationConfig(max_new_tokens=2), chat_traits=None,
    )
    engine = GenerationEngine(
        cast(Any, session),
        lock=threading.RLock(),
        runtime=CompiledRuntime(static_cache_active=True),
        steering=cast(Any, SimpleNamespace(
            all_fast_path=lambda: True, static_steerable=lambda: False,
            uses_compiled_offsets=False, ctx=None, hooks={},
        )),
        reads=cast(Any, SimpleNamespace(
            capture=SimpleNamespace(is_transient=lambda: False),
            step_callback=lambda: (lambda _step: None),
        )),
        release=lambda: None,
    )
    cache = _Cache()
    _static_prefix(engine.prefix_cache, cache, max_cache_len=8)
    seen: dict[str, Any] = {}

    def _fake_generate(_model: Any, _tok: Any, input_ids: torch.Tensor,
                       _config: Any, _state: Any, **kwargs: Any) -> list[int]:
        seen["input_ids"] = input_ids.clone()
        seen.update(kwargs)
        return [42]

    monkeypatch.setattr(engine_mod, "generate_steered", _fake_generate)
    out, _elapsed = engine.decode(
        _spec(), torch.tensor([[1, 2, 3]]), tap=None,
        use_thinking=False, want_perplexity=False,
    )

    assert out == [42]
    assert seen["input_ids"].tolist() == [[3]]
    assert seen["past_key_values"] is cache
    assert seen["cache_position_offset"] == 2
    assert seen["use_static_cache"] is True
    # The static rollback rewinds the write counter to the prefix and the
    # entry stays installed for the next reuse.
    assert int(cache.layers[0].cumulative_length) == 2
    assert engine.prefix_cache


def test_cache_prefix_builds_a_static_entry_with_decode_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.session.engine as engine_mod

    class _Model:
        config = object()

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.param = torch.zeros(1, dtype=torch.float16)

        def parameters(self) -> Any:
            return iter([self.param])

        def __call__(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return SimpleNamespace(past_key_values=kwargs.get("past_key_values"))

    static_cache = object()
    made: dict[str, Any] = {}

    def _make(model: Any, *, max_cache_len: int, device: Any, dtype: Any) -> object:
        made.update(model=model, max_cache_len=max_cache_len, device=device, dtype=dtype)
        return static_cache

    monkeypatch.setattr(engine_mod, "make_static_cache", _make)
    model = _Model()
    session = SimpleNamespace(
        _steering_composer=SimpleNamespace(active=False),
        device=torch.device("cpu"),
        model=model,
        config=GenerationConfig(max_new_tokens=4),
    )
    engine = GenerationEngine(
        cast(Any, session), lock=threading.RLock(),
        runtime=CompiledRuntime(static_cache_active=True),
        steering=cast(Any, None),
        reads=cast(Any, SimpleNamespace(end=lambda: None)),
        release=lambda: None,
    )

    assert engine.cache_prefix(
        torch.tensor([[4, 5, 6]]), max_new_tokens=7, prefer_static=True,
    ) == 3
    entry = engine.prefix_cache.entry
    assert entry is not None and entry.static is True
    assert entry.max_cache_len == 10 and entry.past_key_values is static_cache
    assert made == {
        "model": model, "max_cache_len": 10,
        "device": torch.device("cpu"), "dtype": torch.float16,
    }
    call = model.calls[0]
    assert call["past_key_values"] is static_cache
    assert call["cache_position"].tolist() == [0, 1, 2]
