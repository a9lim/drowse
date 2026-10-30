"""Multi-row generation: ``generate(n>1)``, ``generate_batch``, ``generate_sweep``.

``BatchRunner`` orchestration (row order, seeds, sweep grids, loom sibling
shape, stop semantics) runs against a fake single-generation engine and a real
``LoomTree``.  The batched ``model.generate`` path, the serial shared-prefix KV
reuse (a real ``DynamicCache``) and the fan run on a real tiny session.
"""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from drowse import LoomTree, Recipe, SamplingConfig, derive_seed_schedule
from drowse.generation.decode import GenerationConfig, GenerationState
from drowse.generation.results import GenerationResult, RunSet
from drowse.session.batch import (
    BatchRunner,
    BatchStopCriteria,
    common_prefix,
    fast_batch_blocked,
    generate_capable_model,
    left_pad,
    pad_token_id,
    steering_always_on,
    trim_generated,
)
from drowse.session.session import PreparedCall
from drowse.steering.composer import SteeringComposer
from drowse.steering.terms import Steering
from drowse.steering.triggers import Trigger
from tests.support.sessions import TinySessionFactory, register_vectors


# ---------------------------------------------------------------------------
# A fake single-generation engine for the runner's orchestration.
# ---------------------------------------------------------------------------


class _Engine:
    """Stands in for ``_generate_core``: records each row, writes the loom
    like the real engine, and ends every row the way the real one does (the
    per-row stop event set on the way out)."""

    def __init__(self, tree: LoomTree, state: GenerationState) -> None:
        self.tree = tree
        self.state = state
        self.calls: list[dict[str, Any]] = []
        self.on_row: Any = None

    def __call__(self, input: Any, **kwargs: Any) -> GenerationResult:
        self.state.reset()  # the engine's preamble
        index = len(self.calls)
        self.calls.append({"input": input, **kwargs})
        if self.on_row is not None:
            self.on_row(index)
        steering = kwargs.get("steering")
        sampling = kwargs.get("sampling")
        if not kwargs.get("stateless"):
            user_id = self.tree.add_user_turn(
                input,
                parent_id=kwargs.get("parent_node_id"),
                role_label=(
                    sampling.user_role
                    if sampling is not None and not kwargs.get("raw") else None
                ),
            )
            node_id = self.tree.begin_assistant(
                user_id, recipe=Recipe(steering=str(steering) if steering else None),
            )
            self.tree.finalize_assistant(
                node_id, text=f"out_{index}", applied_steering=str(steering),
            )
        self.state.stop_requested.set()  # end-of-generation signal
        return GenerationResult(
            text=f"out_{index}", tokens=[1, 2, 3], token_count=3,
            tok_per_sec=50.0, elapsed=0.06, steering_alphas={},
            prompt_tokens=4, finish_reason="stop",
            applied_steering=steering if isinstance(steering, str) else None,
        )


def _runner(**overrides: Any) -> tuple[BatchRunner, _Engine]:
    tree = LoomTree()
    state = GenerationState()
    engine = _Engine(tree, state)
    deps: dict[str, Any] = {
        "generate_one": engine,
        "tree": lambda: tree,
        "gen_state": state,
        "render": lambda *_a, **_k: torch.tensor([[1, 2, 3]]),
        "cache_prefix": lambda *_a, **_k: 0,
        "prefill_inactive": lambda _steering: False,
        "static_cache_active": lambda: False,
        "default_max_tokens": lambda: 8,
    }
    deps.update(overrides)
    return BatchRunner(**deps), engine


class TestRunnerBatch:
    def test_returns_results_in_prompt_order(self) -> None:
        runner, engine = _runner()
        runset = runner.batch(["alpha", "beta", "gamma"])
        assert [r.text for r in runset] == ["out_0", "out_1", "out_2"]
        assert [c["input"] for c in engine.calls] == ["alpha", "beta", "gamma"]
        assert runset.kind == "batch"
        assert runset.grid == [{"prompt_index": i} for i in range(3)]
        assert runset.node_ids == [None, None, None]

    def test_steering_and_sampling_pass_through_unchanged(self) -> None:
        runner, engine = _runner()
        sampling = SamplingConfig(seed=7, temperature=0.5)
        runner.batch(["p1", "p2"], steering="0.3 honest", sampling=sampling)
        assert [c["steering"] for c in engine.calls] == ["0.3 honest"] * 2
        assert [c["sampling"] for c in engine.calls] == [sampling] * 2

    def test_on_result_fires_per_completion(self) -> None:
        runner, _engine = _runner()
        seen: list[tuple[int, str]] = []
        runner.batch(
            ["p1", "p2", "p3"],
            on_result=lambda idx, result: seen.append((idx, result.text)),
        )
        assert seen == [(0, "out_0"), (1, "out_1"), (2, "out_2")]

    @pytest.mark.parametrize("prompts", [[], "not-a-list"])
    def test_rejects_empty_or_non_list_prompts(self, prompts: Any) -> None:
        runner, _engine = _runner()
        with pytest.raises(ValueError, match="non-empty list"):
            runner.batch(prompts)

    def test_shared_prefix_is_cached_and_cleared(self) -> None:
        rows = {"a": list(range(40)) + [100], "b": list(range(40)) + [101]}
        cached: list[Any] = []

        def _cache_prefix(ids: Any, **kwargs: Any) -> int:
            cached.append((ids, kwargs))
            return 0 if ids is None else int(ids.shape[0])

        runner, engine = _runner(
            render=lambda prompt, **_k: torch.tensor([rows[prompt]]),
            cache_prefix=_cache_prefix,
            prefill_inactive=lambda steering: steering is None,
        )
        runner.batch(["a", "b"], sampling=SamplingConfig(max_tokens=5))
        (prefix, kwargs), (cleared, _) = cached
        assert prefix.tolist() == list(range(40))
        assert kwargs == {"max_new_tokens": 5, "prefer_static": False}
        assert cleared is None
        assert len(engine.calls) == 2

    def test_prefill_active_steering_skips_the_prefix_cache(self) -> None:
        cached: list[Any] = []
        runner, _engine = _runner(
            render=lambda prompt, **_k: torch.tensor([list(range(40)) + [len(prompt)]]),
            cache_prefix=lambda ids, **_k: cached.append(ids) or 0,
            prefill_inactive=lambda steering: steering is None,
        )
        runner.batch(["a", "bb"], steering="0.5 honest")
        assert cached == []


class TestRunnerFan:
    def test_fan_returns_every_sibling(self) -> None:
        # Regression: every engine row ends with the per-row stop event set;
        # the fan used to read that as a user stop and return one sibling.
        runner, engine = _runner()
        runset = runner.fan("hi", n=3, sampling=SamplingConfig(seed=11))
        assert runset.kind == "fan"
        assert [r.text for r in runset] == ["out_0", "out_1", "out_2"]
        assert [c["sampling"].seed for c in engine.calls] == derive_seed_schedule(11, 3)

    def test_stateful_fan_lands_siblings_under_one_user_turn(self) -> None:
        runner, engine = _runner()
        runset = runner.fan("hi", n=3)
        (user_id,) = engine.tree.children_of[engine.tree.root_id]
        assert engine.tree.children_of[user_id] == list(runset.node_ids)

    def test_fan_forwards_every_generation_kwarg(self) -> None:
        runner, engine = _runner()
        runner.fan(
            "hi", n=2, steering="0.2 warm", stateless=True, raw=True,
            thinking=False, parent_node_id="p", recipe_override="cool",
            gen_seat="user", system_prompt="sys",
        )
        call = engine.calls[0]
        assert call["steering"] == "0.2 warm"
        assert call["raw"] is True and call["thinking"] is False
        assert call["parent_node_id"] == "p"
        assert call["recipe_override"] == "cool"
        assert call["gen_seat"] == "user"
        assert call["system_prompt"] == "sys"

    def test_rejects_non_positive_n(self) -> None:
        runner, _engine = _runner()
        with pytest.raises(ValueError, match="n must be >= 1"):
            runner.fan("hi", n=0)


@pytest.mark.parametrize("surface", ["fan", "batch", "sweep"])
def test_user_stop_ends_the_series(surface: str) -> None:
    runner, engine = _runner()
    engine.on_row = lambda index: index == 1 and engine.state.request_stop()
    if surface == "fan":
        runset = runner.fan("hi", n=4, stateless=True)
    elif surface == "batch":
        runset = runner.batch(["a", "b", "c", "d"])
    else:
        runset = runner.sweep("hi", {"x": [0.0, 0.1, 0.2, 0.3]})
    assert len(runset) == 2
    assert len(engine.calls) == 2


def test_stale_stop_before_the_series_is_ignored() -> None:
    runner, engine = _runner()
    engine.state.request_stop()
    assert len(runner.fan("hi", n=3, stateless=True)) == 3


class TestRunnerSweep:
    def test_single_concept_sweep_yields_one_per_alpha(self) -> None:
        runner, engine = _runner()
        runset = runner.sweep("describe a sunset", {"happy.sad": [-0.4, 0.0, 0.4]})
        assert len(runset) == 3
        assert [c["steering"] for c in engine.calls] == [
            "-0.4 happy.sad", "0.0 happy.sad", "0.4 happy.sad",
        ]
        assert runset.kind == "fan"

    def test_two_concept_grid_is_the_full_product(self) -> None:
        runner, engine = _runner()
        runset = runner.sweep("test", {"a": [0.0, 0.3], "b": [0.0, 0.5]})
        assert [c["steering"] for c in engine.calls] == [
            "0.0 a + 0.0 b", "0.0 a + 0.5 b", "0.3 a + 0.0 b", "0.3 a + 0.5 b",
        ]
        assert runset.grid == [
            {"a": 0.0, "b": 0.0}, {"a": 0.0, "b": 0.5},
            {"a": 0.3, "b": 0.0}, {"a": 0.3, "b": 0.5},
        ]

    def test_base_steering_composes_under_swept_terms(self) -> None:
        runner, engine = _runner()
        runner.sweep("test", {"honest": [0.3, 0.6]}, base_steering="0.2 warm")
        assert [c["steering"] for c in engine.calls] == [
            "0.2 warm + 0.3 honest", "0.2 warm + 0.6 honest",
        ]

    def test_prebuilt_base_steering_renders_canonically(self) -> None:
        runner, engine = _runner()
        runner.sweep(
            "test", {"honest": [0.3]},
            base_steering=Steering.from_value("0.2 warm"),
        )
        assert engine.calls[0]["steering"] == "0.2 warm + 0.3 honest"

    def test_on_result_carries_alpha_values(self) -> None:
        runner, _engine = _runner()
        seen: list[tuple[int, dict[str, float]]] = []
        runner.sweep(
            "test", {"a": [0.0, 0.3], "b": [0.5]},
            on_result=lambda idx, result, alphas: seen.append((idx, dict(alphas))),
        )
        assert seen == [(0, {"a": 0.0, "b": 0.5}), (1, {"a": 0.3, "b": 0.5})]

    def test_applied_steering_round_trips(self) -> None:
        runner, _engine = _runner()
        assert runner.sweep("test", {"honest": [0.4]})[0].applied_steering == "0.4 honest"

    def test_seed_schedule_is_deterministic(self) -> None:
        runner, engine = _runner()
        runner.sweep(
            "x", {"honest.deceptive": [0.1, 0.2, 0.3]},
            sampling=SamplingConfig(seed=42), stateless=False,
        )
        seeds = [c["sampling"].seed for c in engine.calls]
        assert seeds == derive_seed_schedule(42, 3)

    @pytest.mark.parametrize("sweep", [{}, {"a": []}])
    def test_rejects_empty_sweeps(self, sweep: dict[str, list[float]]) -> None:
        runner, _engine = _runner()
        with pytest.raises(ValueError, match="non-empty"):
            runner.sweep("test", sweep)

    def test_stateful_sweep_lands_siblings_under_one_user_turn(self) -> None:
        runner, engine = _runner()
        runset = runner.sweep(
            "say hi", {"honest.deceptive": [0.0, 0.3, 0.6]}, stateless=False,
        )
        tree = engine.tree
        (user_id,) = tree.children_of[tree.root_id]
        assert tree.nodes[user_id].role == "user"
        assert tree.children_of[user_id] == list(runset.node_ids)
        assert all(tree.nodes[n].role == "assistant" for n in runset.node_ids)
        assert isinstance(runset, RunSet)

    def test_labelled_stateful_sweep_shares_one_labelled_anchor(self) -> None:
        # Label-aware dedup must still land every sibling on the anchor, so
        # the anchor carries the label the rows send.
        runner, engine = _runner()
        runset = runner.sweep(
            "say hi", {"honest.deceptive": [0.0, 0.3]},
            sampling=SamplingConfig(user_role="captain"), stateless=False,
        )
        (anchor_id,) = engine.tree.children_of[engine.tree.root_id]
        assert engine.tree.nodes[anchor_id].role_label == "captain"
        assert engine.tree.children_of[anchor_id] == list(runset.node_ids)


# ---------------------------------------------------------------------------
# The batched ``model.generate`` path: eligibility and pure helpers.
# ---------------------------------------------------------------------------


def _prepared(**fields: Any) -> PreparedCall:
    base = PreparedCall(
        steering_obj=None,
        use_thinking_req=False,
        gen_config=GenerationConfig(max_new_tokens=4),
        lp_count=None,
        seed=None,
        stop_list=None,
        logit_bias=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        logprobs_list=None,
    )
    return replace(base, **fields)


class TestFastBatchEligibility:
    def test_plain_call_is_eligible(self) -> None:
        assert fast_batch_blocked(_prepared(), None) is False
        assert fast_batch_blocked(_prepared(), SamplingConfig()) is False

    @pytest.mark.parametrize("fields", [
        {"use_thinking_req": True},
        {"gen_config": GenerationConfig(max_new_tokens=0)},
        {"lp_count": 0},
        {"stop_list": ["END"]},
        {"logit_bias": {1: 2.0}},
        {"presence_penalty": 0.5},
        {"frequency_penalty": 0.5},
        {"seed": 3, "gen_config": GenerationConfig(max_new_tokens=4, temperature=0.7)},
        {"steering_obj": Steering.from_value("0.5 formal.casual@response")},
    ])
    def test_row_shaped_features_block(self, fields: dict[str, Any]) -> None:
        assert fast_batch_blocked(_prepared(**fields), None) is True

    def test_greedy_seed_stays_eligible(self) -> None:
        greedy = GenerationConfig(max_new_tokens=4, temperature=0.0)
        assert fast_batch_blocked(_prepared(seed=3, gen_config=greedy), None) is False

    @pytest.mark.parametrize("flag", [
        "return_hidden", "persist_per_layer_scores", "persist_subspace_coords",
    ])
    def test_capture_heavy_sampling_flags_block(self, flag: str) -> None:
        assert fast_batch_blocked(_prepared(), SamplingConfig(**{flag: True})) is True


class TestSteeringAlwaysOn:
    """HF ``generate`` runs the whole batch in one call, so there is no
    per-step hook re-evaluation to make a phased or gated trigger mean
    anything: every entry kind must be inspected."""

    @pytest.mark.parametrize(
        "expr", [None, "0.5 formal.casual", "!gone", "0.5 personas%hacker"],
    )
    def test_default_trigger_entries_are_always_on(self, expr: str | None) -> None:
        assert steering_always_on(Steering.from_value(expr)) is True

    @pytest.mark.parametrize("expr", [
        "0.5 formal.casual@response",
        "!gone@thinking",
        "0.5 personas%hacker@after",
        "0.5 formal.casual@when:confident.uncertain>0.4",
        "0.5 formal.casual + 0.3 warm.clinical@response",
    ])
    def test_phased_or_gated_entries_are_not(self, expr: str) -> None:
        assert steering_always_on(Steering.from_value(expr)) is False

    def test_a_steering_level_trigger_counts(self) -> None:
        steering = Steering(
            alphas={"formal.casual": 0.5}, trigger=Trigger.GENERATED_ONLY,
        )
        assert steering_always_on(steering) is False


class TestBatchHelpers:
    def test_trim_cuts_at_eos(self) -> None:
        tokens, finish = trim_generated(
            torch.tensor([10, 11, 99, 0]), eos_ids={99}, pad_id=0, max_new_tokens=4,
        )
        assert (tokens, finish) == ([10, 11], "stop")

    def test_trim_full_row_without_eos_is_length(self) -> None:
        tokens, finish = trim_generated(
            torch.tensor([12, 13, 14]), eos_ids={99}, pad_id=0, max_new_tokens=3,
        )
        assert (tokens, finish) == ([12, 13, 14], "length")

    def test_trim_strips_trailing_pad_as_an_early_stop(self) -> None:
        tokens, finish = trim_generated(
            torch.tensor([15, 0, 0]), eos_ids={99}, pad_id=0, max_new_tokens=3,
        )
        assert (tokens, finish) == ([15], "stop")

    def test_left_pad(self) -> None:
        ids, attention, lengths = left_pad(
            [torch.tensor([1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6])], 0,
        )
        assert ids.tolist() == [[0, 1, 2], [3, 4, 5], [0, 0, 6]]
        assert attention.tolist() == [[0, 1, 1], [1, 1, 1], [0, 0, 1]]
        assert lengths == [2, 3, 1]

    @pytest.mark.parametrize(("tokenizer", "expected"), [
        (SimpleNamespace(pad_token_id=5, eos_token_id=9), 5),
        (SimpleNamespace(pad_token_id=None, eos_token_id=[9, 8]), 9),
        (SimpleNamespace(pad_token_id=None, eos_token_id=None), 0),
    ])
    def test_pad_token_id(self, tokenizer: Any, expected: int) -> None:
        assert pad_token_id(tokenizer) == expected

    def test_common_prefix_keeps_a_suffix_and_stays_on_cpu(self) -> None:
        shared = list(range(32))
        rows = [torch.tensor(shared + [100]), torch.tensor(shared + [101])]
        prefix = common_prefix(rows)
        assert prefix is not None and prefix.device.type == "cpu"
        assert prefix.tolist() == shared
        identical = [torch.tensor(shared), torch.tensor(shared)]
        assert cast(torch.Tensor, common_prefix(identical)).tolist() == shared[:-1]

    def test_common_prefix_below_the_floor_is_none(self) -> None:
        rows = [torch.tensor(list(range(10)) + [1] * 20), torch.tensor(list(range(10)) + [2] * 20)]
        assert common_prefix(rows) is None

    def test_generate_capable_model_unwraps_compiled_modules(self) -> None:
        eager = SimpleNamespace(generate=lambda **_k: None)
        assert generate_capable_model(eager) is eager
        assert generate_capable_model(SimpleNamespace(_orig_mod=eager)) is eager
        assert generate_capable_model(SimpleNamespace()) is None


@pytest.mark.parametrize("score_kind", ["none", "tensor", "tuple"])
def test_stop_criteria_accepts_transformers_score_shapes(score_kind: str) -> None:
    state = GenerationState()
    criterion = BatchStopCriteria(state)
    input_ids = torch.LongTensor([[1, 2], [3, 4]])
    score = torch.FloatTensor([[0.1, 0.9], [0.8, 0.2]])
    scores = None if score_kind == "none" else score if score_kind == "tensor" else (score, score)
    assert criterion(input_ids, scores).tolist() == [False, False]
    state.request_stop()
    assert criterion(input_ids, scores).tolist() == [True, True]
    state.reset()
    assert criterion(input_ids, scores).tolist() == [False, False]


# ---------------------------------------------------------------------------
# Real tiny sessions.
# ---------------------------------------------------------------------------


def _reply(_label: str, prompt_text: str) -> str:
    for word in ("alpha", "beta", "gamma"):
        if word in prompt_text:
            return f"re {word}"
    return "ok"


def _spy_model_generate(monkeypatch: pytest.MonkeyPatch, session: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    real = session.model.generate

    def _spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(session.model, "generate", _spy)
    return calls


def _numbers_blanked(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _numbers_blanked(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_numbers_blanked(item) for item in value]
    if isinstance(value, float):
        return "<float>"
    return value


class TestFastBatchOnTinySession:
    def test_compatible_batch_uses_one_model_generate(
        self, make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session, _script = make_tiny_session.scripted(_reply)
        calls = _spy_model_generate(monkeypatch, session)
        seen: list[tuple[int, str]] = []

        runset = session.generate_batch(
            ["alpha", "beta?", "gamma!!"],
            thinking=False,
            on_result=lambda idx, result: seen.append((idx, result.text)),
        )

        assert len(calls) == 1
        assert calls[0]["input_ids"].shape[0] == 3
        assert calls[0]["attention_mask"][:, 0].tolist() == [0, 0, 1]
        assert [r.text for r in runset] == ["re alpha", "re beta", "re gamma"]
        assert [r.finish_reason for r in runset] == ["stop"] * 3
        assert runset.kind == "batch"
        assert runset.grid == [{"prompt_index": i} for i in range(3)]
        assert runset.node_ids == [None, None, None]
        assert seen == [(0, "re alpha"), (1, "re beta"), (2, "re gamma")]
        assert runset.metrics["batch_token_count"] == sum(r.token_count for r in runset)
        assert session.last_result is runset[-1]
        assert session.tree.rev == 0

    def test_batch_passes_the_full_eos_set(
        self, make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session, _script = make_tiny_session.scripted(_reply)
        calls = _spy_model_generate(monkeypatch, session)
        session.generate_batch(["alpha", "beta"], thinking=False)
        assert calls[0]["generation_config"].eos_token_id == sorted(
            session.chat_traits.eos_ids,
        )

    def test_row_aggregates_match_a_serial_generation(
        self, make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The batch reads each row's aggregate from its slice of the batched
        # tail ring and finalizes through the serial finalizer: readings and
        # the envelope shape must match a one-row generation of the prompt.
        session, _script = make_tiny_session.scripted(_reply)
        register_vectors(session, "mood")
        session.add_probe("mood")
        calls = _spy_model_generate(monkeypatch, session)
        prompts = ["alpha", "beta?", "gamma!!"]

        batch = session.generate_batch(prompts, thinking=False)
        assert len(calls) == 1
        serial = [
            session.generate(prompt, stateless=True, thinking=False).first
            for prompt in prompts
        ]

        for fast, one in zip(batch, serial, strict=True):
            assert fast.tokens == one.tokens
            assert set(fast.probe_readings) == {"mood"}
            assert fast.probe_readings["mood"].coords == pytest.approx(
                one.probe_readings["mood"].coords, abs=1e-4,
            )
            assert fast.probe_readings["mood"].fraction == pytest.approx(
                one.probe_readings["mood"].fraction, abs=1e-4,
            )
            assert fast.measurements is not None
            assert json.dumps(_numbers_blanked(fast.measurements), sort_keys=True) == (
                json.dumps(_numbers_blanked(one.measurements), sort_keys=True)
            )

    def test_return_probe_readings_false_skips_aggregates(
        self, make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session, _script = make_tiny_session.scripted(_reply)
        register_vectors(session, "mood")
        session.add_probe("mood")
        calls = _spy_model_generate(monkeypatch, session)
        runset = session.generate_batch(
            ["alpha", "beta"],
            sampling=SamplingConfig(return_probe_readings=False),
            thinking=False,
        )
        assert len(calls) == 1
        assert [r.probe_readings for r in runset] == [{}, {}]

    def test_blank_steering_means_unsteered(
        self, make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Regression: the fast path handed ``""`` to the grammar, which
        # raised "empty steering expression"; the serial loop reads it as
        # unsteered.
        session, _script = make_tiny_session.scripted(_reply)
        calls = _spy_model_generate(monkeypatch, session)
        runset = session.generate_batch(["alpha", "beta"], steering="", thinking=False)
        assert len(calls) == 1
        assert [r.applied_steering for r in runset] == [None, None]

    def test_cast_member_steers_batched_rows(
        self, make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Regression: the fast path skipped the cast roster, so an assistant
        # cast member's steering reached serial rows but not batched ones.
        session, _script = make_tiny_session.scripted(_reply)
        register_vectors(session, "castvec")
        session.set_cast_member("assistant", steering="0.4 castvec")
        calls = _spy_model_generate(monkeypatch, session)
        runset = session.generate_batch(["alpha", "beta"], thinking=False)
        assert len(calls) == 1
        assert [r.applied_steering for r in runset] == ["0.4 castvec"] * 2
        assert [r.steering_alphas for r in runset] == [{"castvec": 0.4}] * 2
        serial = session.generate("alpha", stateless=True, thinking=False).first
        assert serial.applied_steering == "0.4 castvec"

    def test_sae_probe_without_a_resident_layer_stays_serial(
        self, make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session, _script = make_tiny_session.scripted(_reply)
        session.sae.probes["sae/17"] = {
            "feature_id": 17, "layer": 0, "label": None, "max_act": None,
        }
        assert session._batch._fast.run(
            ["alpha", "beta"], steering=None, sampling=None, thinking=False,
            stateless=True, raw=False, on_result=None,
        ) is None

    @pytest.mark.parametrize("sampling", [
        SamplingConfig(seed=5, temperature=0.7),
        SamplingConfig(stop=("END",)),
        SamplingConfig(logprobs=2),
    ])
    def test_row_shaped_calls_run_serially(
        self,
        sampling: SamplingConfig,
        make_tiny_session: TinySessionFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session, _script = make_tiny_session.scripted(_reply)
        calls = _spy_model_generate(monkeypatch, session)
        runset = session.generate_batch(["alpha", "beta"], sampling=sampling, thinking=False)
        assert calls == []
        assert [r.text for r in runset] == ["re alpha", "re beta"]


class TestSerialBatchOnTinySession:
    def test_shared_prefix_reuses_one_real_dynamic_cache(
        self, make_tiny_session: TinySessionFactory,
    ) -> None:
        # The serial batch prefills the rows' shared prefix once; every row
        # decodes on top of that DynamicCache and rolls it back to the prefix
        # afterwards.  ``script.prompts`` records only forwards that start
        # from an empty cache, so one entry means every row hit the prefix
        # (a failed rollback drops the entry and the next row re-prefills).
        session, script = make_tiny_session.scripted(_reply)
        head = "Answer in one short line, please, and stay on topic: "
        # Distinct suffix lengths: the script keys replies by prompt length.
        prompts = [head + "alpha", head + "beta", head + "gamma!!"]
        sampling = SamplingConfig(seed=5, temperature=0.7, max_tokens=12)

        runset = session.generate_batch(prompts, sampling=sampling)

        assert [r.text for r in runset] == ["re alpha", "re beta", "re gamma"]
        assert len(script.prompts) == 1
        assert head in script.prompts[0]
        assert "alpha" not in script.prompts[0]
        # The batch clears its prefix: the next generation prefills in full.
        session.generate(prompts[0], stateless=True, sampling=sampling)
        assert len(script.prompts) == 2
        assert script.prompts[1].endswith("alpha<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


class TestFanOnTinySession:
    def test_stateful_fan_returns_every_sibling(
        self, make_tiny_session: TinySessionFactory,
    ) -> None:
        # Regression: the serial fan returned one sibling (it read the
        # end-of-generation stop event as a user stop).
        session, _script = make_tiny_session.scripted(_reply)
        runset = session.generate("alpha", n=3, sampling=SamplingConfig(seed=1))
        assert runset.kind == "fan"
        assert [r.text for r in runset] == ["re alpha"] * 3
        (user_id,) = session.tree.children_of[session.tree.root_id]
        assert session.tree.children_of[user_id] == list(runset.node_ids)
        seeds = [session.tree.nodes[n].recipe.sampling.seed for n in runset.node_ids]
        assert seeds == derive_seed_schedule(1, 3)

    def test_greedy_stateless_fan_runs_every_row_serially(
        self, make_tiny_session: TinySessionFactory, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session, _script = make_tiny_session.scripted(_reply)
        calls = _spy_model_generate(monkeypatch, session)
        runset = session.generate(
            "alpha", n=3, stateless=True, thinking=False,
            sampling=SamplingConfig(temperature=0.0),
        )
        assert calls == []
        assert runset.kind == "fan"
        assert [r.text for r in runset] == ["re alpha"] * 3
        assert runset.node_ids == [None, None, None]

    def test_stop_during_a_fan_ends_it(
        self, make_tiny_session: TinySessionFactory,
    ) -> None:
        session, _script = make_tiny_session.scripted(_reply)
        stopped: list[str] = []

        def _on_token(text: str, *_rest: Any) -> None:
            if not stopped:
                stopped.append(text)
                session.stop()

        runset = session.generate("alpha", n=3, stateless=True, on_token=_on_token)
        assert len(runset) == 1
        assert runset[0].text == "r"


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

    def test_static_prefix_hit_requires_static_eligibility_and_headroom(self) -> None:
        import torch
        from drowse.generation.prefix_cache import PrefixCacheEntry

        s = self._session()
        cache = object()
        s._prefix_cache = PrefixCacheEntry(
            prefix_ids_cpu=torch.tensor([1, 2]),
            past_key_values=cache,
            prefix_len=2,
            static=True,
            max_cache_len=5,
        )
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        assert s._try_prefix_cache_hit(
            ids, static_eligible=False, required_max_new_tokens=1,
        ) is None
        assert s._try_prefix_cache_hit(
            ids, static_eligible=True, required_max_new_tokens=3,
        ) is None

        hit = s._try_prefix_cache_hit(
            ids, static_eligible=True, required_max_new_tokens=2,
        )
        assert hit is not None
        suffix, hit_cache, prefix_len, is_static = hit
        assert suffix.tolist() == [[3]]
        assert hit_cache is cache
        assert prefix_len == 2
        assert is_static is True

    def test_generation_loop_keeps_static_cache_on_static_prefix_hit(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import torch
        from drowse.generation.decode import GenerationConfig
        from drowse.generation.prefix_cache import PrefixCacheEntry
        from drowse.session.session import CaptureState

        class _Cache:
            # One static layer whose write counter the generation advanced
            # past the 2-token prefix.
            layers = [SimpleNamespace(cumulative_length=torch.tensor(4))]

        s = self._session()
        cache = _Cache()
        s._prefix_cache = PrefixCacheEntry(
            prefix_ids_cpu=torch.tensor([1, 2]),
            past_key_values=cache,
            prefix_len=2,
            static=True,
            max_cache_len=8,
        )
        s._static_cache_active = True
        s_any = cast(Any, s)
        s_any._steering = SimpleNamespace(
            all_fast_path=lambda: True,
            static_steerable=lambda: False,
            uses_compiled_offsets=False,
            ctx=None,
            hooks={},
        )
        s._steering_active_in_prefill = lambda: False
        s._steering_needs_probe_gating = lambda: False
        s_any._build_gating_score_callback = lambda: None
        s._compiled = False
        s._device = torch.device("cpu")
        s_any._model = object()
        s_any._tokenizer = object()
        s_any._gen_state = object()
        s._capture_state = CaptureState(persistent=False)
        s_any._capture = SimpleNamespace(
            ingest_persistent=lambda: None,
            fire_step_sink=lambda: None,
        )

        seen: dict[str, Any] = {}

        def _fake_generate(
            model: Any,
            tokenizer: Any,
            input_ids: torch.Tensor,
            config: GenerationConfig,
            state: Any,
            **kwargs: Any,
        ) -> list[int]:
            del model, tokenizer, config, state
            seen["input_ids"] = input_ids.clone()
            seen.update(kwargs)
            return [42]

        monkeypatch.setattr("drowse.session.session.generate_steered", _fake_generate)

        out, _elapsed = s._run_generation_loop(
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            GenerationConfig(max_new_tokens=2),
            use_thinking=False,
            want_hidden=False,
            effective_tap=None,
            seed=None,
            stop_list=None,
            logit_bias=None,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            lp_count=None,
        )

        assert out == [42]
        assert seen["input_ids"].tolist() == [[3]]
        assert seen["past_key_values"] is cache
        assert seen["cache_position_offset"] == 2
        assert seen["use_static_cache"] is True
        # The static rollback rewinds the write counter to the prefix and the
        # entry stays installed for the next reuse.
        assert int(cache.layers[0].cumulative_length) == 2
        assert s._prefix_cache is not None

    def test_cache_prefix_can_build_static_cache_entry(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import torch
        from drowse.generation.decode import GenerationConfig

        s = self._session()
        s._static_cache_active = True
        s._device = torch.device("cpu")
        s.config = GenerationConfig(max_new_tokens=4)
        s._steering_composer._stack = []
        s._prefix_cache = None
        from drowse.session.session import GenState
        s._gen_phase = GenState.IDLE
        s._end_capture = lambda: None

        static_cache = object()
        made: dict[str, Any] = {}

        def _make_static_cache(model: Any, max_cache_len: int, device: Any, dtype: Any) -> object:
            made.update({
                "model": model,
                "max_cache_len": max_cache_len,
                "device": device,
                "dtype": dtype,
            })
            return static_cache

        monkeypatch.setattr(
            "drowse.generation.static_cache.make_static_cache", _make_static_cache,
        )

        class _Model:
            config = object()

            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []
                self.param = torch.zeros(1, dtype=torch.float16)

            def parameters(self):
                return iter([self.param])

            def __call__(self, **kwargs: Any) -> Any:
                self.calls.append(kwargs)
                return SimpleNamespace(past_key_values=kwargs.get("past_key_values"))

        model = _Model()
        cast(Any, s)._model = model

        prefix_len = s.cache_prefix(
            torch.tensor([[4, 5, 6]], dtype=torch.long),
            max_new_tokens=7,
            prefer_static=True,
        )

        assert prefix_len == 3
        assert s._prefix_cache is not None
        assert s._prefix_cache.static is True
        assert s._prefix_cache.max_cache_len == 10
        assert s._prefix_cache.past_key_values is static_cache
        assert made == {
            "model": model,
            "max_cache_len": 10,
            "device": torch.device("cpu"),
            "dtype": torch.float16,
        }
        call = model.calls[0]
        assert call["past_key_values"] is static_cache
        assert call["cache_position"].tolist() == [0, 1, 2]


