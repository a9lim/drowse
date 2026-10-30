"""Batched generation: ``session.generate_batch`` / ``generate_sweep``.

Tests cover the serial wrapper contract plus compatible stateless fast paths:
ordering, sweep grid shape, and ``applied_steering`` round-trip.

CPU-only.  Mock ``_generate_core`` so we exercise the wrapper logic
without spinning up a real model.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
import pytest

from drowse.generation.results import GenerationResult, ProbeReading
from drowse.steering.composer import SteeringComposer
from drowse.loom.tree import LoomTree
from tests.support.engine import attach_engine

if TYPE_CHECKING:
    from drowse.session.session import DrowseSession


# ---------------------------------------------------------------------------
# Helpers — mock just enough of DrowseSession for the batch path.
# ---------------------------------------------------------------------------


def _make_result(text: str, applied: str | None = None) -> GenerationResult:
    """Cheap GenerationResult stub."""
    return GenerationResult(
        text=text,
        tokens=[1, 2, 3],
        token_count=3,
        tok_per_sec=50.0,
        elapsed=0.06,
        steering_alphas={},
        prompt_tokens=4,
        finish_reason="stop",
        applied_steering=applied,
    )


def _engine_of(session: Any) -> Any:
    """The stub's engine, attaching the real one on first use."""
    return session.__dict__.get("_engine") or attach_engine(session)


def _stub_generate_core(session: DrowseSession, *, capture: list[Any]) -> None:
    """Replace the engine's ``generate_one`` with a stub that records calls.

    Returns one ``GenerationResult`` per call carrying the call's index
    so tests can verify ordering.  Each call appends a ``(input,
    steering)`` tuple to ``capture`` for assertions on the wrapper's
    composition logic.
    """
    counter = {"n": 0}

    def _fake(input: Any, *, steering: Any = None, sampling: Any = None, stateless: bool = False, raw: bool = False, thinking: Any = None, on_token: Any = None, **kwargs: Any) -> GenerationResult:
        # ``kwargs`` swallows additions to ``generate_one``'s signature
        # (``parent_node_id``, ``recipe_override``, ...) so this stub doesn't
        # churn every time the core gains a new optional keyword.
        idx = counter["n"]
        counter["n"] += 1
        capture.append({"input": input, "steering": steering})
        applied = steering if isinstance(steering, str) else None
        return _make_result(f"out_{idx}", applied=applied)

    _engine_of(session).generate_one = _fake


class _NoopSteeringContext:
    def __enter__(self) -> "_NoopSteeringContext":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def _install_noop_steering(session: DrowseSession) -> None:
    session_any = cast(Any, session)
    session_any._profiles = {"a": {}}
    session_any.steering = lambda value: _NoopSteeringContext()
    session_any._steering_composer = SimpleNamespace(
        _stack=[], snapshot_steering_alphas=lambda: {"a": 0.0},
    )
    session_any._exit_internal_steering = (
        lambda steering_cm, *, swallow: steering_cm.__exit__(None, None, None)
    )


class _BatchTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    name_or_path = "batch-test-tokenizer"
    vocab_size = 200
    chat_template = ""
    added_tokens_encoder: dict[str, int] = {}

    def decode(
        self,
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens
        return " ".join(str(i) for i in ids)


class _BatchModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.config = SimpleNamespace(vocab_size=200)
        self.generation_config = SimpleNamespace(eos_token_id=99)

    def named_parameters(self):
        return iter(())

    def named_buffers(self):
        return iter(())

    def generate(self, **kwargs: Any):
        import torch

        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        tail = torch.tensor(
            [
                [10, 11, 99],
                [12, 13, 14],
                [15, 0, 0],
            ],
            dtype=torch.long,
            device=input_ids.device,
        )
        return torch.cat([input_ids, tail[: input_ids.shape[0]]], dim=1)


class _BatchProbeMonitor:
    def __init__(self) -> None:
        self.probe_names = ["mood"]
        self.scored: list[float] = []

    def probe_layers(self) -> set[int]:
        return {0}

    def enable_curved_warm(self, flag: bool) -> None:
        del flag

    def set_subspace_coords(self, flag: bool) -> None:
        del flag

    def score_aggregate(
        self,
        hidden_per_layer: dict[int, Any],
    ) -> dict[str, ProbeReading]:
        value = float(hidden_per_layer[0].reshape(-1)[0])
        self.scored.append(value)
        return {"mood": ProbeReading(fraction=value, nearest=[], coords=(value,))}


class _ProbeBatchModel(_BatchModel):
    def __init__(self, layers: Any) -> None:
        super().__init__()
        self.layers = layers

    def generate(self, **kwargs: Any):
        import torch

        input_ids = kwargs["input_ids"]
        for values in ([1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]):
            hidden = torch.tensor(
                values[: input_ids.shape[0]],
                dtype=torch.float32,
                device=input_ids.device,
            ).reshape(input_ids.shape[0], 1, 1)
            for layer in self.layers:
                layer(hidden)
        return super().generate(**kwargs)


def _fast_batch_session():
    import torch
    from drowse.generation.decode import GenerationConfig
    from drowse.session.session import CaptureState, DrowseSession
    from drowse.steering.triggers import TriggerContext

    s = DrowseSession.__new__(DrowseSession)
    model = _BatchModel()
    s_any = cast(Any, s)
    s_any._model = model
    s._model_info = {"model_id": "batch/test"}
    s._jlens = None
    s_any._tokenizer = _BatchTokenizer()
    s._device = torch.device("cpu")
    s_any._gen_lock = threading.Lock()
    s_any._monitor = SimpleNamespace(probe_names=[], set_subspace_coords=lambda _flag: None)
    s._profiles = {}
    s._manifolds = {}
    s.config = GenerationConfig(
        max_new_tokens=3,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
    )
    events = SimpleNamespace(emitted=[])
    events.emit = lambda event: events.emitted.append(event)
    s_any.events = events
    s_any._steering = SimpleNamespace(
        ctx=TriggerContext(),
        hooks={},
        uses_compiled_offsets=False,
        reset_manifold_feet=lambda: None,
    )
    s._lens_instrument.live = None
    s._lens_instrument.probes = {}
    s._sae_instrument.probes = {}
    s._sae_layer = None
    s._active_gen_reservation = None
    s._capture_state = CaptureState()
    from drowse.readout.capture_buffer import HiddenCapture
    s._capture = HiddenCapture()
    attach_engine(s)
    # Call resolution reads the cast roster even on the stateless fast path.
    s.tree = LoomTree()
    s._incremental_readings = []
    s._incremental_gate_scores = []
    s._readout_long_tensor_cache = {}
    s._internal_steering_pop = False
    s._active_role = None
    s._steering_composer = SteeringComposer(s)

    def _prepare_input(
        input: Any,
        raw: bool = False,
        thinking: bool = False,
        stateless: bool = False,
        parent_node_id: str | None = None,
        user_role: str | None = None,
        assistant_role: str | None = None,
        to_device: bool = True,
        gen_seat: str = "assistant",
        add_generation_prompt: bool = True,
        system_prompt: Any = None,
    ):
        del raw, thinking, stateless, parent_node_id
        del user_role, assistant_role, to_device, gen_seat, add_generation_prompt, system_prompt
        mapping = {
            "alpha": [1, 2],
            "beta": [3, 4, 5],
            "gamma": [6],
        }
        return torch.tensor([mapping[input]], dtype=torch.long)

    s._prepare_input = _prepare_input
    return s, model


def _probe_fast_batch_session():
    import torch
    from drowse.readout.capture_buffer import HiddenCapture

    s, _model = _fast_batch_session()
    s._layers = torch.nn.ModuleList([torch.nn.Identity()])
    s._capture = HiddenCapture()
    s_any = cast(Any, s)
    s_any._monitor = _BatchProbeMonitor()
    model = _ProbeBatchModel(s._layers)
    s_any._model = model
    return s, model


# ---------------------------------------------------------------------------
# session.generate_batch — pure unit tests against a real DrowseSession
# subclass that overrides _generate_core.
# ---------------------------------------------------------------------------


class TestGenerateBatch:
    def _session(self):
        # Construct a minimal DrowseSession by bypassing __init__; the
        # batch methods only need ``_generate_core`` to exist.  These tests
        # pin the SERIAL fan-out contract (prompt order, per-row steering /
        # sampling pass-through), so the one-shot fast path is declined
        # outright rather than through some incidental engine attribute.
        from drowse.session.session import DrowseSession

        s = DrowseSession.__new__(DrowseSession)
        s._steering_composer = SteeringComposer(s)
        cast(Any, s)._generate_batch_fast = lambda *_a, **_kw: None
        return s

    def test_returns_results_in_prompt_order(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        prompts = ["alpha", "beta", "gamma"]
        results = s.generate_batch(prompts)

        assert len(results) == 3
        assert [r.text for r in results] == ["out_0", "out_1", "out_2"]
        assert [c["input"] for c in capture] == prompts

    def test_steering_passes_through_unchanged(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        s.generate_batch(["p1", "p2"], steering="0.3 honest")

        assert all(c["steering"] == "0.3 honest" for c in capture)

    def test_on_result_callback_fires_per_completion(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        seen: list[tuple[int, str]] = []
        s.generate_batch(
            ["p1", "p2", "p3"],
            on_result=lambda idx, result: seen.append((idx, result.text)),
        )

        assert seen == [(0, "out_0"), (1, "out_1"), (2, "out_2")]

    def test_empty_prompt_list_raises(self) -> None:
        s = self._session()
        with pytest.raises(ValueError, match="non-empty list"):
            s.generate_batch([])

    def test_compatible_batch_uses_one_model_generate(self) -> None:
        s, model = _fast_batch_session()

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        _engine_of(s).generate_one = _fail_generate_core
        seen: list[tuple[int, str]] = []

        runset = s.generate_batch(
            ["alpha", "beta", "gamma"],
            thinking=False,
            on_result=lambda idx, result: seen.append((idx, result.text)),
        )

        assert len(model.calls) == 1
        call = model.calls[0]
        assert call["input_ids"].shape == (3, 3)
        assert call["attention_mask"].tolist() == [
            [0, 1, 1],
            [1, 1, 1],
            [0, 0, 1],
        ]
        assert [r.tokens for r in runset] == [[10, 11], [12, 13, 14], [15]]
        assert [r.finish_reason for r in runset] == ["stop", "length", "stop"]
        assert [r.text for r in runset] == ["10 11", "12 13 14", "15"]
        assert runset.grid == [
            {"prompt_index": 0},
            {"prompt_index": 1},
            {"prompt_index": 2},
        ]
        assert runset.node_ids == [None, None, None]
        assert seen == [(0, "10 11"), (1, "12 13 14"), (2, "15")]
        assert runset.metrics["batch_token_count"] == 6
        assert runset.metrics["batch_elapsed"] > 0.0
        assert "batch_tok_per_sec" in runset.metrics
        assert s.last_result is runset[-1]

    def test_fast_path_passes_full_eos_set_to_model_generate(self) -> None:
        # ``model.generate`` must stop on every EOS id the ordinary decode
        # loop recognizes (generation_config + tokenizer + added
        # end-of-turn specials) — the tokenizer's single ``eos_token_id``
        # alone misses e.g. Gemma's ``<end_of_turn>``, and the batch would
        # generate past it to the token cap.
        s, model = _fast_batch_session()
        # A tokenizer carrying an added end-of-turn special beyond the
        # plain eos.
        tok = _BatchTokenizer()
        tok.name_or_path = "batch-test-tokenizer-eos-set"
        tok.added_tokens_encoder = {"<end_of_turn>": 42}
        cast(Any, s)._tokenizer = tok
        model.generation_config = SimpleNamespace(eos_token_id=[99, 98])

        s.generate_batch(["alpha", "beta", "gamma"], thinking=False)

        assert len(model.calls) == 1
        assert model.calls[0]["generation_config"].eos_token_id == [42, 98, 99]

    def test_greedy_batch_allows_seeded_sampling_config(self) -> None:
        from drowse.generation.sampling import SamplingConfig

        s, model = _fast_batch_session()

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        _engine_of(s).generate_one = _fail_generate_core

        runset = s.generate_batch(
            ["alpha", "beta"],
            sampling=SamplingConfig(seed=123),
            thinking=False,
        )

        assert len(model.calls) == 1
        assert [r.tokens for r in runset] == [[10, 11], [12, 13, 14]]

    def test_stochastic_seeded_batch_stays_serial(self) -> None:
        from drowse.generation.sampling import SamplingConfig

        s, model = _fast_batch_session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        runset = s.generate_batch(
            ["alpha", "beta"],
            sampling=SamplingConfig(seed=123, temperature=0.7),
            thinking=False,
        )

        assert len(model.calls) == 0
        assert [r.text for r in runset] == ["out_0", "out_1"]
        assert [c["input"] for c in capture] == ["alpha", "beta"]

    def test_sae_probes_without_layer_fall_back_to_serial_generation(self) -> None:
        # Geometry probes ride the batched aggregate-only capture ring, but an
        # SAE feature probe needs a resident SAE layer to encode against; with
        # none the batch declines to the serial loop rather than reading
        # nothing.
        s, model = _fast_batch_session()
        cast(Any, s)._sae_instrument.probes = {"sae/17": {"feature_id": 17}}
        cast(Any, s)._sae_layer = None
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        runset = s.generate_batch(["alpha", "beta"], thinking=False)

        assert len(model.calls) == 0
        assert [r.text for r in runset] == ["out_0", "out_1"]
        assert [c["input"] for c in capture] == ["alpha", "beta"]

    def test_probe_batch_fast_path_scores_per_row_aggregate(self) -> None:
        s, model = _probe_fast_batch_session()

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        _engine_of(s).generate_one = _fail_generate_core

        runset = s.generate_batch(["alpha", "beta", "gamma"], thinking=False)

        assert len(model.calls) == 1
        assert [r.tokens for r in runset] == [[10, 11], [12, 13, 14], [15]]
        assert [r.probe_readings["mood"].coords for r in runset] == [
            (2.0,),
            (30.0,),
            (100.0,),
        ]
        assert cast(Any, s._monitor).scored == [2.0, 30.0, 100.0]

    def test_probe_batch_fast_path_honors_return_probe_readings_false(self) -> None:
        from drowse.generation.sampling import SamplingConfig

        s, model = _probe_fast_batch_session()

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        _engine_of(s).generate_one = _fail_generate_core

        runset = s.generate_batch(
            ["alpha", "beta", "gamma"],
            sampling=SamplingConfig(return_probe_readings=False),
            thinking=False,
        )

        assert len(model.calls) == 1
        assert [r.tokens for r in runset] == [[10, 11], [12, 13, 14], [15]]
        assert [r.probe_readings for r in runset] == [{}, {}, {}]
        assert cast(Any, s._monitor).scored == []

    def test_sae_readout_probe_batch_fast_path_scores_per_row_aggregate(self) -> None:
        from drowse.sae.backend import MockSaeBackend

        s, model = _probe_fast_batch_session()
        cast(Any, s)._monitor = SimpleNamespace(probe_names=[], set_subspace_coords=lambda _flag: None)
        s._sae_backend = MockSaeBackend(layers=frozenset({0}), d_model=1, d_feature=1)
        s._sae_layer = 0
        s._sae_width = 1
        s._sae_feature_meta = {}
        s._sae_instrument.probes = {
            "sae/0": {
                "feature_id": 0,
                "layer": 0,
                "label": None,
                "max_act": None,
            },
        }

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        _engine_of(s).generate_one = _fail_generate_core

        runset = s.generate_batch(["alpha", "beta", "gamma"], thinking=False)

        assert len(model.calls) == 1
        assert [r.probe_readings["sae/0"].coords for r in runset] == [
            (2.0,),
            (30.0,),
            (100.0,),
        ]

    def test_sae_readout_probe_batch_fast_path_honors_return_probe_readings_false(self) -> None:
        from drowse.generation.sampling import SamplingConfig

        s, model = _probe_fast_batch_session()
        cast(Any, s)._monitor = SimpleNamespace(probe_names=[], set_subspace_coords=lambda _flag: None)
        s._sae_layer = 0
        s._sae_instrument.probes = {
            "sae/0": {
                "feature_id": 0,
                "layer": 0,
                "label": None,
                "max_act": None,
            },
        }

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        _engine_of(s).generate_one = _fail_generate_core

        runset = s.generate_batch(
            ["alpha", "beta", "gamma"],
            sampling=SamplingConfig(return_probe_readings=False),
            thinking=False,
        )

        assert len(model.calls) == 1
        assert [r.probe_readings for r in runset] == [{}, {}, {}]

    def test_lens_readout_probe_batch_fast_path_scores_per_row_aggregate(self) -> None:
        s, model = _probe_fast_batch_session()
        s_any = cast(Any, s)
        s_any._monitor = SimpleNamespace(probe_names=[], set_subspace_coords=lambda _flag: None)
        s_any._lens_instrument.probes = {"jlens/g": {"token_id": 1, "layers": [0]}}
        s_any._lens_instrument.probe_layers = lambda: {0}

        def _score_lens_probes(
            hidden: dict[int, Any],
            **kwargs: Any,
        ) -> dict[str, ProbeReading]:
            del kwargs
            value = float(hidden[0].reshape(-1)[0])
            return {"jlens/g": ProbeReading(0.0, [], coords=(value,))}

        # The batch per-row aggregate routes through the lens run's
        # ``observe_aggregate``, which consumes the INSTRUMENT's
        # ``score_probes`` — stub that (the session forwarder is bypassed).
        s_any._lens_instrument.score_probes = _score_lens_probes

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        _engine_of(s).generate_one = _fail_generate_core

        runset = s.generate_batch(["alpha", "beta", "gamma"], thinking=False)

        assert len(model.calls) == 1
        assert [r.probe_readings["jlens/g"].coords for r in runset] == [
            (2.0,),
            (30.0,),
            (100.0,),
        ]

    def test_lens_readout_probe_batch_fast_path_honors_return_probe_readings_false(self) -> None:
        from drowse.generation.sampling import SamplingConfig

        s, model = _probe_fast_batch_session()
        s_any = cast(Any, s)
        s_any._monitor = SimpleNamespace(probe_names=[], set_subspace_coords=lambda _flag: None)
        s_any._lens_instrument.probes = {"jlens/g": {"token_id": 1, "layers": [0]}}
        s_any._lens_instrument.probe_layers = lambda: {0}

        def _score_lens_probes(*_args: Any, **_kwargs: Any) -> dict[str, ProbeReading]:
            raise AssertionError("return_probe_readings=False should skip lens probes")

        s_any._lens_instrument.score_probes = _score_lens_probes

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        _engine_of(s).generate_one = _fail_generate_core

        runset = s.generate_batch(
            ["alpha", "beta", "gamma"],
            sampling=SamplingConfig(return_probe_readings=False),
            thinking=False,
        )

        assert len(model.calls) == 1
        assert [r.probe_readings for r in runset] == [{}, {}, {}]

    def test_deterministic_fan_uses_batched_generation(self) -> None:
        s, model = _fast_batch_session()

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        _engine_of(s).generate_one = _fail_generate_core

        runset = s.generate("alpha", n=3, stateless=True, thinking=False)

        assert len(model.calls) == 1
        assert model.calls[0]["input_ids"].shape == (3, 2)
        assert runset.kind == "fan"
        assert runset.grid == [{}, {}, {}]
        assert runset.node_ids == [None, None, None]
        assert [r.tokens for r in runset] == [[10, 11], [12, 13, 14], [15]]

    def test_deterministic_fan_honors_return_probe_readings_false(self) -> None:
        from drowse.generation.sampling import SamplingConfig

        s, model = _probe_fast_batch_session()

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        _engine_of(s).generate_one = _fail_generate_core

        runset = s.generate(
            "alpha",
            n=3,
            stateless=True,
            sampling=SamplingConfig(return_probe_readings=False),
            thinking=False,
        )

        assert len(model.calls) == 1
        assert runset.kind == "fan"
        assert [r.probe_readings for r in runset] == [{}, {}, {}]
        assert cast(Any, s._monitor).scored == []

    def test_deterministic_fan_with_seed_uses_batched_generation(self) -> None:
        from drowse.generation.sampling import SamplingConfig

        s, model = _fast_batch_session()

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        _engine_of(s).generate_one = _fail_generate_core

        runset = s.generate(
            "alpha",
            n=3,
            sampling=SamplingConfig(seed=123),
            stateless=True,
            thinking=False,
        )

        assert len(model.calls) == 1
        assert runset.kind == "fan"
        assert [r.tokens for r in runset] == [[10, 11], [12, 13, 14], [15]]


# ---------------------------------------------------------------------------
# Prefix-cache eligibility — the predicates that decide whether a batch can
# share one prefill.  CPU-only: they only read steering triggers, no model.
# ---------------------------------------------------------------------------


class TestPrefixCacheEligibility:
    def _session(self):
        from drowse.session.session import DrowseSession
        session = DrowseSession.__new__(DrowseSession)
        session._steering_composer = SteeringComposer(session)
        session._profiles = {}
        return session

    @pytest.mark.parametrize(
        "expr, expected_inactive",
        [
            (None, True),                          # no steering → reusable
            ("0.5 honest", False),                 # default BOTH steers prefill
            ("0.5 honest@response", True),         # response-phase: prompt untouched
            ("0.5 honest@generated", True),        # decode-only
            ("0.5 honest@after", True),            # after-N: prompt=False
            ("0.5 honest@prompt", False),          # prompt-phase steers prefill
            ("0.5 honest@both", False),            # explicit both
            ("0.5 honest@when:honest.deceptive>0.4", True),  # probe-gated: inactive in prefill
        ],
    )
    def test_steering_value_prefill_inactive(
        self, expr: "str | None", expected_inactive: bool,
    ) -> None:
        s = self._session()
        assert s._steering_value_prefill_inactive(expr) is expected_inactive

    def test_mixed_terms_active_if_any_steers_prefill(self) -> None:
        # One response-phase term + one default-BOTH term → prefill IS steered.
        s = self._session()
        assert s._steering_value_prefill_inactive("0.5 honest@response + 0.3 warm") is False

    def test_malformed_expression_treated_as_active(self) -> None:
        # A parse failure is conservatively prefill-active so the caller skips
        # caching and lets the normal path surface the error.
        s = self._session()
        assert s._steering_value_prefill_inactive("0.5 !nope~bad") is False

    def test_active_in_prefill_reads_the_live_stack(self) -> None:
        from drowse.steering.triggers import Trigger

        s = self._session()
        s._steering_composer._stack = []
        assert s._steering_active_in_prefill() is False
        # Response-phase entry (prompt=False) → prefill untouched.
        s._steering_composer._stack = [{"honest": (0.5, Trigger.GENERATED_ONLY)}]
        assert s._steering_active_in_prefill() is False
        # Default BOTH entry → prefill steered.
        s._steering_composer._stack = [{"honest": (0.5, Trigger.BOTH)}]
        assert s._steering_active_in_prefill() is True
        # A probe-gated trigger reports inactive during prefill.
        gated = Trigger.when("honest.deceptive", ">", 0.4)
        s._steering_composer._stack = [{"honest": (0.5, gated)}]
        assert s._steering_active_in_prefill() is False

    def test_batch_common_prefix_detection_keeps_scalar_walk_on_cpu(self) -> None:
        import torch

        s = self._session()
        common = list(range(32))

        def _prepare_input(
            input: Any,
            raw: bool = False,
            thinking: bool = False,
            stateless: bool = False,
            parent_node_id: str | None = None,
            user_role: str | None = None,
            assistant_role: str | None = None,
            to_device: bool = True,
            system_prompt: Any = None,
        ) -> torch.Tensor:
            suffix = 100 if input == "a" else 101
            return torch.tensor([common + [suffix]], dtype=torch.long)

        cast(Any, s)._prepare_input = _prepare_input

        prefix = s._batch_common_prefix_ids(["a", "b"], raw=False)

        assert prefix is not None
        assert prefix.device.type == "cpu"
        assert prefix.tolist() == common


# ---------------------------------------------------------------------------
# session.generate_sweep — Cartesian product, applied_steering receipts.
# ---------------------------------------------------------------------------


class TestGenerateSweep:
    def _session(self):
        from drowse.session.session import DrowseSession

        s = DrowseSession.__new__(DrowseSession)
        from drowse.generation.decode import GenerationConfig
        s.config = GenerationConfig()
        return s

    def test_single_concept_sweep_yields_one_per_alpha(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        results = s.generate_sweep(
            "describe a sunset",
            sweep={"happy.sad": [-0.4, 0.0, 0.4]},
        )

        assert len(results) == 3
        # Each call's steering string carries the unique alpha + concept.
        steerings = [c["steering"] for c in capture]
        assert steerings == [
            "-0.4 happy.sad",
            "0.0 happy.sad",
            "0.4 happy.sad",
        ]

    def test_two_concept_grid_full_product(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        s.generate_sweep(
            "test",
            sweep={"a": [0.0, 0.3], "b": [0.0, 0.5]},
        )

        # 2 * 2 = 4 results.
        assert len(capture) == 4
        # Order is itertools.product over (a, b): a varies slowest.
        steerings = [c["steering"] for c in capture]
        assert steerings == [
            "0.0 a + 0.0 b",
            "0.0 a + 0.5 b",
            "0.3 a + 0.0 b",
            "0.3 a + 0.5 b",
        ]

    def test_base_steering_composes_under_swept_terms(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        s.generate_sweep(
            "test",
            sweep={"honest": [0.3, 0.6]},
            base_steering="0.2 warm",
        )

        steerings = [c["steering"] for c in capture]
        assert steerings == [
            "0.2 warm + 0.3 honest",
            "0.2 warm + 0.6 honest",
        ]

    def test_on_result_carries_alpha_values(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        seen: list[tuple[int, dict[str, float]]] = []
        s.generate_sweep(
            "test",
            sweep={"a": [0.0, 0.3], "b": [0.5]},
            on_result=lambda idx, result, alphas: seen.append((idx, dict(alphas))),
        )

        assert seen == [
            (0, {"a": 0.0, "b": 0.5}),
            (1, {"a": 0.3, "b": 0.5}),
        ]

    def test_applied_steering_round_trips_canonical(self) -> None:
        s = self._session()
        capture: list[Any] = []
        _stub_generate_core(s, capture=capture)

        results = s.generate_sweep(
            "test", sweep={"honest": [0.4]},
        )

        # Stub propagates ``steering`` to ``applied_steering``; the
        # canonical receipt round-trips through generate_sweep.
        assert results[0].applied_steering == "0.4 honest"

    def test_degenerate_greedy_sweep_uses_batched_generation(self) -> None:
        s, model = _fast_batch_session()
        _install_noop_steering(s)

        def _fail_generate_core(*args: Any, **kwargs: Any) -> GenerationResult:
            raise AssertionError("serial generation path should not run")

        _engine_of(s).generate_one = _fail_generate_core
        seen: list[tuple[int, str, dict[str, float]]] = []

        runset = s.generate_sweep(
            "alpha",
            sweep={"a": [0.0, 0.0, 0.0]},
            thinking=False,
            on_result=lambda idx, result, alphas: seen.append(
                (idx, result.text, dict(alphas)),
            ),
        )

        assert len(model.calls) == 1
        assert model.calls[0]["input_ids"].shape == (3, 2)
        assert runset.kind == "fan"
        assert runset.grid == [{"a": 0.0}, {"a": 0.0}, {"a": 0.0}]
        assert [r.tokens for r in runset] == [[10, 11], [12, 13, 14], [15]]
        assert seen == [
            (0, "10 11", {"a": 0.0}),
            (1, "12 13 14", {"a": 0.0}),
            (2, "15", {"a": 0.0}),
        ]

    def test_empty_sweep_dict_raises(self) -> None:
        s = self._session()
        with pytest.raises(ValueError, match="non-empty"):
            s.generate_sweep("test", sweep={})

    def test_empty_alpha_list_raises(self) -> None:
        s = self._session()
        with pytest.raises(ValueError, match="non-empty list"):
            s.generate_sweep("test", sweep={"a": []})


# ---------------------------------------------------------------------------
# Batch-fast eligibility: only always-on steering can ride HF ``generate``.
# ---------------------------------------------------------------------------


class TestBatchFastSteeringGate:
    """``_batch_fast_steering_is_always_on`` over every entry shape.

    HF ``generate`` runs the whole batch in one call, so there is no per-step
    hook re-evaluation to make a phased or gated trigger mean anything — the
    gate has to see every entry kind, not just the additive ones.
    """

    @staticmethod
    def _gate(value: str | None) -> bool:
        from drowse.session.session import DrowseSession
        from drowse.steering.terms import Steering

        return DrowseSession._batch_fast_steering_is_always_on(
            Steering.from_value(value),
        )

    def test_unsteered_is_eligible(self) -> None:
        assert self._gate(None) is True

    @pytest.mark.parametrize(
        "expr",
        ["0.5 formal.casual", "!gone", "0.5 personas%hacker"],
    )
    def test_default_trigger_entries_are_eligible(self, expr: str) -> None:
        assert self._gate(expr) is True

    @pytest.mark.parametrize(
        "expr",
        [
            "0.5 formal.casual@response",
            "!gone@thinking",
            "0.5 personas%hacker@after",
            "0.5 formal.casual@when:confident.uncertain>0.4",
        ],
    )
    def test_phased_or_gated_entries_are_not_eligible(self, expr: str) -> None:
        assert self._gate(expr) is False

    def test_one_phased_term_disqualifies_the_whole_expression(self) -> None:
        assert self._gate("0.5 formal.casual + 0.3 warm.clinical@response") is False

    def test_a_nondefault_steering_level_trigger_disqualifies(self) -> None:
        """Bare floats inherit ``Steering.trigger``, so it counts too."""
        from drowse.session.session import DrowseSession
        from drowse.steering.terms import Steering
        from drowse.steering.triggers import Trigger

        steering = Steering(
            alphas={"formal.casual": 0.5}, trigger=Trigger.GENERATED_ONLY,
        )
        assert DrowseSession._batch_fast_steering_is_always_on(steering) is False
