"""CPU-only generation-loop regressions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from drowse.generation.decode import (
    GenerationConfig,
    GenerationState,
    _PenaltyState,
    generate_steered,
)
from drowse.generation.finalize import MIN_ELAPSED_FOR_RATE
from drowse.model.chat import chat_eos_ids, logits_keep_kwarg
from drowse.readout.types import ProbeReading
from drowse.generation.types import GenState
from drowse.readout.planner import CaptureMode, CaptureState
from tests.support.decode import NoCacheModel, StopTokenizer, scripted_model
from tests.support.read_plane import (
    fake_readout_family,
    finalize,
    gate_refs,
    ReadRig,
)


def test_logits_keep_detection_is_keyed_by_forward_implementation() -> None:
    class CurrentModel:
        def forward(self, input_ids: Any, logits_to_keep: int | None = None) -> Any:
            del input_ids, logits_to_keep

    class LegacyModel:
        def forward(self, input_ids: Any, num_logits_to_keep: int | None = None) -> Any:
            del input_ids, num_logits_to_keep

    assert logits_keep_kwarg(CurrentModel()) == "logits_to_keep"
    assert logits_keep_kwarg(LegacyModel()) == "num_logits_to_keep"

    class KwargsModel:
        def forward(self, input_ids: Any, **kwargs: Any) -> Any:
            del input_ids, kwargs

    assert logits_keep_kwarg(KwargsModel()) is None
    assert logits_keep_kwarg(SimpleNamespace()) is None


def test_eos_ids_are_not_shared_between_models_with_the_same_tokenizer() -> None:
    tokenizer = StopTokenizer()
    first = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=1))
    second = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2))

    assert chat_eos_ids(first, tokenizer) == {1, 3}
    assert chat_eos_ids(second, tokenizer) == {2, 3}
    listed = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[1, 2]))
    assert chat_eos_ids(listed, tokenizer) == {1, 2, 3}


class _SplitStopTokenizer(StopTokenizer):
    name_or_path = "split-stop-tokenizer"
    vocab_size = 5
    _pieces = {
        0: "Hello S",
        1: "TOP",
        2: " ignored",
        3: "",
    }


def test_stop_sequence_trimmed_text_is_final_result_text():
    model: Any = scripted_model([0, 1, 2])
    tokenizer: Any = StopTokenizer()
    state = GenerationState()
    emitted: list[str] = []

    generated_ids = generate_steered(
        model,
        cast(Any, tokenizer),
        torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=5, temperature=0.0),
        state,
        on_token=lambda text, *_args: emitted.append(text),
        stop=[" STOP"],
    )

    assert generated_ids == [0, 1]
    assert emitted == ["Hello"]
    assert state.finish_reason == "stop_sequence"
    assert state.response_text == "Hello"
    assert state.response_aggregate_index == 0

    plane = ReadRig(
        capture=SimpleNamespace(stacked=lambda: {}), tokenizer=tokenizer,
    ).plane
    result, _ = finalize(plane, generated_ids, state=state, tokenizer=tokenizer)
    assert result.text == "Hello"


@pytest.mark.parametrize(
    ("elapsed", "expected"), [(MIN_ELAPSED_FOR_RATE / 2, 0.0), (0.5, 4.0)],
)
def test_finalize_owns_the_rate_floor(elapsed: float, expected: float) -> None:
    # Regression: ``finalize_generation`` defaulted to a 1e-9 floor and only
    # honored ``MIN_ELAPSED_FOR_RATE`` because the session passed it in; a
    # direct call reported a noise rate for a near-instant generation.
    tokenizer = StopTokenizer()
    plane = ReadRig(
        capture=SimpleNamespace(stacked=lambda: {}), tokenizer=tokenizer,
    ).plane
    result, _ = finalize(
        plane, [0, 2], state=GenerationState(), tokenizer=tokenizer, elapsed=elapsed,
    )

    assert result.tok_per_sec == expected


def test_decode_loop_hands_one_step_id_to_sink_gate_and_tap():
    """The loop-owned forward index: the capture sink (``step_callback``),
    the gate callback (``score_callback``), and the token tap
    (``on_token``'s trailing ``step_id``) all receive the SAME value for
    one forward, and the value is ``len(generated_ids)`` before that
    forward — what lets the instrument runs' step-keyed memos pair one
    forward's gate and display reads."""
    from drowse.steering.triggers import TriggerContext

    model: Any = scripted_model([0, 1, 2])
    tokenizer: Any = StopTokenizer()
    state = GenerationState()
    sink_steps: list[int] = []
    gate_steps: list[int] = []
    tap_steps: list[int] = []

    def _gate(step: int) -> dict[str, float]:
        gate_steps.append(step)
        return {}

    generated_ids = generate_steered(
        model,
        cast(Any, tokenizer),
        torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=3, temperature=0.0),
        state,
        on_token=lambda _t, _th, _tid, _lp, _alts, _ppl, step: (
            tap_steps.append(step)
        ),
        trigger_ctx=TriggerContext(),
        score_callback=_gate,
        step_callback=sink_steps.append,
    )

    assert generated_ids == [0, 1, 2]
    assert sink_steps == [0, 1, 2]
    assert gate_steps == [0, 1, 2]
    assert tap_steps == [0, 1, 2]


class _Utf8SplitTokenizer(StopTokenizer):
    """ids 1 and 2 are partial UTF-8 pieces that only decode as a group."""

    name_or_path = "utf8-split-tokenizer"
    vocab_size = 5
    eos_token_id = 4
    all_special_ids = [4]
    _pieces = {0: "A", 1: "�", 2: "�", 3: "B", 4: ""}

    def decode(self, ids: Any, skip_special_tokens: bool = False) -> str:
        ids = [int(t) for t in ids]
        if ids == [1, 2]:
            return "é"
        if ids == [1, 2, 3]:
            return "éB"
        return super().decode(ids, skip_special_tokens)


def test_buffered_utf8_group_emits_carry_the_flush_forwards_step():
    """Partial-UTF-8 tokens buffer in ``pending_ids`` and flush as a group
    — every emit in the group carries the FLUSHING forward's step (the
    group's readings semantics: they all read that forward's captures)."""
    model: Any = scripted_model(
        [0, 1, 2, 3], vocab_size=5, eos_token_id=4,
    )
    tokenizer: Any = _Utf8SplitTokenizer()
    state = GenerationState()
    emits: list[tuple[str, int]] = []

    generated_ids = generate_steered(
        model,
        cast(Any, tokenizer),
        torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=4, temperature=0.0),
        state,
        on_token=lambda text, _th, _tid, _lp, _alts, _ppl, step: (
            emits.append((text, step))
        ),
    )

    assert generated_ids == [0, 1, 2, 3]
    # "A" emitted by forward 0; tokens 1+2 buffer (partial pieces); the
    # completing token 3 flushes the group at forward 3.
    assert emits == [("A", 0), ("éB", 3)]


def test_post_loop_utf8_flush_uses_the_last_forwards_step():
    """A generation ending with a still-buffered partial flushes post-loop
    with ``last_forward_step`` (no fresh forward ran since)."""
    model: Any = scripted_model([0, 1], vocab_size=5, eos_token_id=4)
    tokenizer: Any = _Utf8SplitTokenizer()
    state = GenerationState()
    emits: list[tuple[str, int]] = []

    generated_ids = generate_steered(
        model,
        cast(Any, tokenizer),
        torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=2, temperature=0.0),
        state,
        on_token=lambda text, _th, _tid, _lp, _alts, _ppl, step: (
            emits.append((text, step))
        ),
    )

    assert generated_ids == [0, 1]
    assert emits[0] == ("A", 0)
    # The trailing lone partial flushed after the loop with the last
    # forward's step (1), not a stale or synthetic index.
    assert emits[1][1] == 1


def test_alt_token_decode_does_not_break_partial_utf8_buffering():
    """Top-K alternative decoding shares the lazy decode cache with the emit
    path; a partial-UTF-8 id decoded as an alternative must still buffer when
    it is emitted (no replacement character reaches the stream)."""
    model: Any = scripted_model(
        [0, 1, 2, 3], vocab_size=5, eos_token_id=4,
    )
    tokenizer: Any = _Utf8SplitTokenizer()
    state = GenerationState()
    emits: list[tuple[str, int]] = []

    generated_ids = generate_steered(
        model,
        cast(Any, tokenizer),
        torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=4, temperature=0.0),
        state,
        on_token=lambda text, _th, _tid, _lp, _alts, _ppl, step: (
            emits.append((text, step))
        ),
        logprobs=1,
    )

    assert generated_ids == [0, 1, 2, 3]
    assert emits == [("A", 0), ("éB", 3)]
    assert not any("�" in text for text, _step in emits)


def test_perplexity_is_none_when_not_requested():
    """``want_perplexity=False`` reports ``None`` (not computed), never NaN —
    the value every ``float | None`` consumer is typed for."""
    model: Any = scripted_model([0, 1, 2])
    tokenizer: Any = StopTokenizer()
    state = GenerationState()
    ppls: list[float | None] = []

    generate_steered(
        model,
        cast(Any, tokenizer),
        torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=3, temperature=0.0),
        state,
        on_token=lambda _t, _th, _tid, _lp, _alts, ppl, _step: ppls.append(ppl),
        want_perplexity=False,
    )

    assert ppls
    assert all(p is None for p in ppls)


def test_perplexity_is_a_float_when_requested():
    model: Any = scripted_model([0, 1, 2])
    tokenizer: Any = StopTokenizer()
    state = GenerationState()
    ppls: list[float | None] = []

    generate_steered(
        model,
        cast(Any, tokenizer),
        torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=3, temperature=0.0),
        state,
        on_token=lambda _t, _th, _tid, _lp, _alts, ppl, _step: ppls.append(ppl),
        want_perplexity=True,
    )

    assert ppls
    assert all(isinstance(p, float) and p == p for p in ppls)


def test_stop_sequence_split_across_tokens_trims_final_text():
    model: Any = scripted_model([0, 1, 3], vocab_size=5)
    tokenizer: Any = _SplitStopTokenizer()
    state = GenerationState()
    emitted: list[str] = []

    generated_ids = generate_steered(
        model,
        cast(Any, tokenizer),
        torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=5, temperature=0.0),
        state,
        on_token=lambda text, *_args: emitted.append(text),
        stop=[" STOP"],
    )

    assert generated_ids == [0, 1]
    assert emitted == ["Hello S"]
    assert state.finish_reason == "stop_sequence"
    assert state.response_text == "Hello"
    assert state.response_aggregate_index == 0


def test_stop_sequence_probe_aggregate_uses_visible_endpoint():
    model: Any = scripted_model([0, 1, 2])
    tokenizer: Any = StopTokenizer()
    state = GenerationState()
    generated_ids = generate_steered(
        model,
        cast(Any, tokenizer),
        torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=5, temperature=0.0),
        state,
        on_token=lambda *_args: None,
        stop=[" STOP"],
    )

    visible = ProbeReading(coords=(0.25,), fraction=0.25, nearest=[])
    hidden_stop = ProbeReading(coords=(0.99,), fraction=0.99, nearest=[])

    class Capture:
        def stacked(self) -> dict[int, torch.Tensor]:
            raise AssertionError("incremental stop aggregate should not stack")

    plane = ReadRig(
        capture=Capture(), monitor=SimpleNamespace(probe_names=["toy"]),
        tokenizer=tokenizer,
    ).plane
    plane.state = CaptureState(mode=CaptureMode.INCREMENTAL)
    plane._rows = [{"toy": visible}, {"toy": hidden_stop}]
    result, _ = finalize(plane, generated_ids, state=state, tokenizer=tokenizer)

    assert result.text == "Hello"
    assert result.probe_readings == {"toy": visible}


def test_stop_sequence_only_tap_can_skip_full_token_table():
    class CountingStopTokenizer(StopTokenizer):
        name_or_path = "stop-tokenizer-no-table"

        def __init__(self) -> None:
            self.batch_decode_calls = 0

        def batch_decode(self, ids: Any) -> list[str]:
            self.batch_decode_calls += 1
            return super().batch_decode(ids)

    model: Any = scripted_model([0, 1, 2])
    tokenizer = CountingStopTokenizer()
    state = GenerationState()
    emitted: list[str] = []

    generated_ids = generate_steered(
        model,
        cast(Any, tokenizer),
        torch.tensor([[0]]),
        GenerationConfig(max_new_tokens=5, temperature=0.0),
        state,
        on_token=lambda text, *_args: emitted.append(text),
        stop=[" STOP"],
    )

    assert tokenizer.batch_decode_calls == 0
    assert generated_ids == [0, 1]
    assert emitted == ["Hello"]
    assert state.finish_reason == "stop_sequence"
    assert state.response_text == "Hello"


def test_finalize_reuses_scored_probe_aggregate() -> None:
    """Finalization should not rescore the same probe aggregate after
    ``score_per_token`` already returned the full ``ProbeReading``."""

    reading = ProbeReading(
        fraction=0.5,
        nearest=[("node", 0.25)],
        coords=(0.25,),
        fraction_per_layer={0: 0.5},
        coords_per_layer={0: (0.25,)},
    )

    class Capture:
        def __init__(self) -> None:
            self.calls = 0

        def stacked(self) -> dict[int, torch.Tensor]:
            self.calls += 1
            return {0: torch.randn(2, 4)}

    class Monitor:
        probe_names = ["toy"]

        def score_per_token(
            self,
            captured: dict[int, torch.Tensor],
            generated_ids: list[int],
            tokenizer: Any,
            *,
            aggregate_index: int | None = None,
        ) -> tuple[dict[str, ProbeReading], dict[str, list[float]]]:
            assert list(captured) == [0]
            assert generated_ids == [0, 1]
            assert aggregate_index == 1
            return {"toy": reading}, {"toy": [0.1, 0.25]}

        def score_aggregate(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("aggregate was already scored")

    capture = Capture()
    state = GenerationState()
    state.finish_reason = "length"
    tokenizer = StopTokenizer()
    plane = ReadRig(capture=capture, monitor=Monitor(), tokenizer=tokenizer).plane

    result, per_token = finalize(plane, [0, 1], state=state, tokenizer=tokenizer)

    assert capture.calls == 1
    assert result.probe_readings == {"toy": reading}
    assert per_token == {"toy": [0.1, 0.25]}


def test_finalize_incremental_probe_path_does_not_stack_capture() -> None:
    reading0 = ProbeReading(
        fraction=0.2,
        nearest=[("node", 0.4)],
        coords=(0.1,),
        fraction_per_layer={0: 0.2},
        coords_per_layer={0: (0.1,)},
    )
    reading1 = ProbeReading(
        fraction=0.5,
        nearest=[("node", 0.25)],
        coords=(0.25,),
        fraction_per_layer={0: 0.5},
        coords_per_layer={0: (0.25,)},
    )

    class Capture:
        def stacked(self) -> dict[int, torch.Tensor]:
            raise AssertionError("incremental probe finalization should not stack")

    class Monitor:
        probe_names = ["toy"]

    state = GenerationState()
    state.finish_reason = "length"
    tokenizer = StopTokenizer()
    plane = ReadRig(capture=Capture(), monitor=Monitor(), tokenizer=tokenizer).plane
    plane.state = CaptureState(mode=CaptureMode.INCREMENTAL)
    plane._rows = [{"toy": reading0}, {"toy": reading1}]

    result, per_token = finalize(plane, [0, 1], state=state, tokenizer=tokenizer)

    assert result.probe_readings == {"toy": reading1}
    assert per_token == {"toy": [0.1, 0.25]}


def test_finalize_lean_incremental_probe_path() -> None:
    """LEAN_INCREMENTAL: per-token coord stream from lean rows + full aggregate
    re-scored from the tail ring.

    Key invariants:
    1. The capture's ``stacked()`` is never called (lean uses the tail ring).
    2. The per-token coord stream is populated from the lean per-token rows
       the step sink stored.
    3. The aggregate in the result IS the full ``ProbeReading`` (with ``nearest``
       populated), NOT a lean coords-only row — finalize re-scores the pooled
       slice through the geometry run → ``monitor.score_aggregate``.
    4. The recorded per-token scores hold the coord stream (not the lean reading).
    """

    # Lean per-token rows: only coords populated (nearest empty — the lean path).
    lean0 = ProbeReading(
        fraction=0.15,
        nearest=[],
        coords=(0.1,),
        fraction_per_layer={},
        coords_per_layer={},
    )
    lean1 = ProbeReading(
        fraction=0.3,
        nearest=[],
        coords=(0.3,),
        fraction_per_layer={},
        coords_per_layer={},
    )
    # Full aggregate reading from the tail ring re-score.
    full_agg = ProbeReading(
        fraction=0.3,
        nearest=[("node", 0.2)],
        coords=(0.3,),
        fraction_per_layer={0: 0.3},
        coords_per_layer={0: (0.3,)},
    )

    pooled_slice = {0: torch.randn(4)}

    class Capture:
        def stacked(self) -> dict[int, torch.Tensor]:
            raise AssertionError("LEAN_INCREMENTAL must not call stacked()")

        def tail_slice_at(self, _idx: int) -> dict[int, torch.Tensor]:
            return pooled_slice

    class Monitor:
        probe_names = ["toy"]
        _agg_calls = 0

        def enable_curved_warm(self, _flag: bool) -> None:
            pass

        def score_aggregate(
            self, captured: dict[int, torch.Tensor], **_kw: Any
        ) -> dict[str, ProbeReading]:
            self._agg_calls += 1
            assert captured is pooled_slice
            return {"toy": full_agg}

    monitor = Monitor()
    state = GenerationState()
    state.finish_reason = "length"
    tokenizer = StopTokenizer()
    plane = ReadRig(capture=Capture(), monitor=monitor, tokenizer=tokenizer).plane
    plane.state = CaptureState(mode=CaptureMode.LEAN_INCREMENTAL)
    plane._rows = [{"toy": lean0}, {"toy": lean1}]

    result, per_token = finalize(plane, [0, 1], state=state, tokenizer=tokenizer)

    # Invariant 3: aggregate carries the full reading (nearest populated).
    assert result.probe_readings == {"toy": full_agg}
    assert result.probe_readings["toy"].nearest == [("node", 0.2)]
    # Invariant 2+4: per-token coord stream extracted from lean rows.
    assert per_token == {"toy": [0.1, 0.3]}
    # Invariant 3: monitor.score_aggregate was called exactly once.
    assert monitor._agg_calls == 1


def test_finalize_gating_subset_probe_path() -> None:
    """GATING_SUBSET: per-token scoring is only for probe gates (subset scalars);
    the full-roster aggregate is pooled once from the tail ring at finalize.

    Key invariants:
    1. ``stacked()`` is never called.
    2. ``per_token`` in the result is empty (GATING_SUBSET shares the
       ``aggregate_only`` finalize path — no full per-token stream).
    3. The aggregate reading IS the full ``ProbeReading`` from ``score_aggregate``.
    4. The recorded per-token scores are None (no full per-token stream).
    """

    full_agg = ProbeReading(
        fraction=0.6,
        nearest=[("confident", 0.1)],
        coords=(0.6,),
        fraction_per_layer={0: 0.6},
        coords_per_layer={0: (0.6,)},
    )
    pooled_slice = {0: torch.randn(4)}

    class Capture:
        def stacked(self) -> dict[int, torch.Tensor]:
            raise AssertionError("GATING_SUBSET must not call stacked()")

        def tail_slice_at(self, _idx: int) -> dict[int, torch.Tensor]:
            return pooled_slice

    class Monitor:
        probe_names = ["toy"]
        _agg_calls = 0

        def enable_curved_warm(self, _flag: bool) -> None:
            pass

        def score_aggregate(
            self, captured: dict[int, torch.Tensor], **_kw: Any
        ) -> dict[str, ProbeReading]:
            self._agg_calls += 1
            assert captured is pooled_slice
            return {"toy": full_agg}

    monitor = Monitor()
    state = GenerationState()
    state.finish_reason = "length"
    tokenizer = StopTokenizer()
    plane = ReadRig(capture=Capture(), monitor=monitor, tokenizer=tokenizer).plane
    # GATING_SUBSET: CaptureState.aggregate_only returns True for this mode.
    plane.state = CaptureState(
        mode=CaptureMode.GATING_SUBSET, gating_subset={"toy"}, gating_keys={"toy"},
    )

    result, per_token = finalize(plane, [0, 1], state=state, tokenizer=tokenizer)

    # Invariant 3: full aggregate reading from tail ring.
    assert result.probe_readings == {"toy": full_agg}
    assert result.probe_readings["toy"].nearest == [("confident", 0.1)]
    # Invariant 2+4: no per-token coord stream.
    assert per_token is None
    # Invariant 3: score_aggregate called exactly once.
    assert monitor._agg_calls == 1


def test_finalize_reuses_one_aggregate_pool_for_monitor_lens_and_sae() -> None:
    pooled = {1: torch.ones(4)}

    class Capture:
        def __init__(self) -> None:
            self.tail_calls: list[int] = []

        def tail_slice_at(self, idx: int) -> dict[int, torch.Tensor]:
            self.tail_calls.append(idx)
            return pooled

        def stacked(self) -> dict[int, torch.Tensor]:
            raise AssertionError("shared aggregate pool should avoid stacked fallback")

    capture = Capture()
    monitor_reading = ProbeReading(0.0, [], coords=(0.1,))
    lens_reading = ProbeReading(0.0, [], coords=(0.2,))
    sae_reading = ProbeReading(0.0, [], coords=(0.3,))
    seen: list[tuple[str, bool]] = []

    def _scorer(family: str, reading: Any, name: str) -> Any:
        def _score(*args: Any) -> dict[str, Any]:
            seen.append((family, args[-1] is pooled))
            return {name: reading}
        return _score

    class Monitor:
        probe_names = ["toy"]

        def enable_curved_warm(self, _flag: bool) -> None:
            pass

        score_aggregate = staticmethod(_scorer("monitor", monitor_reading, "toy"))

    state = GenerationState()
    state.finish_reason = "length"
    # The aggregate position: a forward the ring still holds.
    state.response_aggregate_index = 1
    tokenizer = StopTokenizer()
    plane = ReadRig(
        capture=capture,
        monitor=Monitor(),
        lens=fake_readout_family(
            ("jlens/g",), score_aggregate=_scorer("lens", lens_reading, "jlens/g"),
        ),
        sae=fake_readout_family(
            ("sae/0",), score_aggregate=_scorer("sae", sae_reading, "sae/0"),
        ),
        tokenizer=tokenizer,
    ).plane
    plane.state = CaptureState(mode=CaptureMode.AGGREGATE_ONLY)

    result, _ = finalize(plane, [0, 1], state=state, tokenizer=tokenizer)

    assert capture.tail_calls == [1]
    assert seen == [("monitor", True), ("lens", True), ("sae", True)]
    assert result.probe_readings == {
        "toy": monitor_reading,
        "jlens/g": lens_reading,
        "sae/0": sae_reading,
    }


def test_gating_callback_backfills_exact_keys_hidden_by_top_n() -> None:
    """Full per-token readings can truncate label channels; gates still need the
    exact requested scalar keys."""


    class Capture:
        def latest_per_layer(self) -> dict[int, torch.Tensor]:
            return {0: torch.ones(4)}

    class Monitor:
        probe_names = ("toy",)

        def __init__(self) -> None:
            self.plan_calls: list[frozenset[str]] = []
            self.score_calls: list[tuple[str, frozenset[str]]] = []

        def flat_scalars(self, _readings: Any) -> dict[str, float]:
            return {"toy@nearest": -0.1}

        def plan_gate_scalars(
            self,
            gate_keys: set[str],
            *,
            probe_names: set[str] | None = None,
        ) -> tuple[str, frozenset[str]]:
            assert probe_names is None
            planned = frozenset(gate_keys)
            self.plan_calls.append(planned)
            return ("planned", planned)

        def score_planned_gate_scalars(
            self,
            _latest: dict[int, torch.Tensor],
            plan: tuple[str, frozenset[str]],
        ) -> dict[str, float]:
            self.score_calls.append(plan)
            assert plan == ("planned", frozenset({"toy@hidden"}))
            return {"toy@hidden": -2.0}

        def score_gate_scalars(
            self,
            _latest: dict[int, torch.Tensor],
            gate_keys: set[str],
            *,
            probe_names: set[str] | None = None,
        ) -> dict[str, float]:
            raise AssertionError("missing gate keys should use cached plans")

    monitor = Monitor()
    plane = ReadRig(capture=Capture(), monitor=monitor).plane
    plane.state = CaptureState(
        mode=CaptureMode.INCREMENTAL, gating_keys={"toy@nearest", "toy@hidden"},
    )
    # The incremental branch reads the sink's appended row for this forward.
    plane._rows = [{"toy": object()}]

    callback = plane.gate_score_callback()
    scores = callback(0)
    scores_again = callback(1)

    assert scores == {"toy@nearest": -0.1, "toy@hidden": -2.0}
    assert scores_again == scores
    assert monitor.plan_calls == [frozenset({"toy@hidden"})]
    assert monitor.score_calls == [
        ("planned", frozenset({"toy@hidden"})),
        ("planned", frozenset({"toy@hidden"})),
    ]


@pytest.mark.parametrize(
    ("expr", "family", "expected"),
    [
        ("0.3 jlens/g@when:jlens/g>0.4", "lens", {"jlens/g": 0.7}),
        ("0.3 sae/2@when:sae/2>0.4", "sae", {"sae/2": 0.8}),
    ],
)


def test_readout_only_gates_skip_monitor_probe_scoring(
    expr: str,
    family: str,
    expected: dict[str, float],
) -> None:
    class Capture:
        def latest_per_layer(self) -> dict[int, torch.Tensor]:
            raise AssertionError("monitor path should not inspect captures")

    class Monitor:
        probe_names = ("toy",)

        def flat_scalars(self, _readings: Any) -> dict[str, float]:
            raise AssertionError("readout-only gates should skip monitor scalars")

        def score_gate_scalars(self, *_args: Any, **_kwargs: Any) -> dict[str, float]:
            raise AssertionError("readout-only gates should skip monitor gates")

        def score_single_token(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("readout-only gates should skip monitor probes")

    # The lens and SAE gate scalars ride each family's current run.
    name = next(iter(expected))
    families = {
        fam: fake_readout_family(
            (name,) if fam == family else (),
            gate_scalars=lambda _step, _hidden, _keys, fam=fam: (
                dict(expected) if fam == family else {}
            ),
        )
        for fam in ("lens", "sae")
    }
    plane = ReadRig(
        capture=Capture(), monitor=Monitor(), gate_refs=gate_refs(expr), **families,
    ).plane

    assert plane.gate_score_callback()(0) == expected


def test_mixed_monitor_and_lens_gates_score_only_monitor_gate_keys() -> None:
    class Capture:
        def latest_per_layer(self) -> dict[int, torch.Tensor]:
            return {0: torch.ones(4)}

    class Monitor:
        probe_names = ("toy", "other")

        def __init__(self) -> None:
            self.requested: set[str] | None = None

        def plan_gate_scalars(
            self,
            gate_keys: set[str],
            *,
            probe_names: set[str] | None = None,
        ) -> frozenset[str]:
            assert probe_names is None
            self.requested = set(gate_keys)
            return frozenset(gate_keys)

        def score_planned_gate_scalars(
            self,
            _latest: dict[int, torch.Tensor],
            plan: frozenset[str],
        ) -> dict[str, float]:
            assert plan == frozenset({"toy"})
            return {"toy": 0.2}

        def score_single_token(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("exact monitor gate keys should avoid full scoring")

    monitor = Monitor()
    plane = ReadRig(
        capture=Capture(),
        monitor=monitor,
        lens=fake_readout_family(
            ("jlens/g",),
            gate_scalars=lambda _step, _hidden, _keys: {"jlens/g": 0.7},
        ),
        gate_refs=gate_refs("0.3 toy@when:toy>0.1 + 0.2 jlens/g@when:jlens/g>0.4"),
    ).plane

    scores = plane.gate_score_callback()(0)

    assert scores == {"toy": 0.2, "jlens/g": 0.7}
    assert monitor.requested == {"toy"}


def test_stateless_zero_token_probe_result_does_not_use_history() -> None:
    """A stateless empty generation has no current-run probe aggregate."""

    class Capture:
        def stacked(self) -> dict[int, torch.Tensor]:
            raise AssertionError("no generated tokens should not stack capture")

    class Monitor:
        probe_names = ["toy"]

    state = GenerationState()
    state.finish_reason = "length"
    tokenizer = StopTokenizer()
    plane = ReadRig(capture=Capture(), monitor=Monitor(), tokenizer=tokenizer).plane

    result, per_token = finalize(plane, [], state=state, tokenizer=tokenizer)

    assert result.probe_readings == {}
    assert per_token is None


def test_return_probe_readings_false_skips_probe_finalization() -> None:
    class Capture:
        def stacked(self) -> dict[int, torch.Tensor]:
            raise AssertionError("probe finalization disabled")

    class Monitor:
        probe_names = ["toy"]

    def _disabled(family: str) -> Any:
        def _raise(*_args: Any) -> Any:
            raise AssertionError(f"{family} finalization disabled")
        return _raise

    state = GenerationState()
    state.finish_reason = "length"
    tokenizer = StopTokenizer()
    plane = ReadRig(
        capture=Capture(),
        monitor=Monitor(),
        lens=fake_readout_family(("jlens/x",), score_aggregate=_disabled("lens")),
        sae=fake_readout_family(("sae/0",), score_aggregate=_disabled("sae")),
        tokenizer=tokenizer,
    ).plane

    result, _ = finalize(
        plane, [0], state=state, tokenizer=tokenizer, return_probe_readings=False,
    )

    assert result.probe_readings == {}


def test_penalty_state_applies_sparse_counts_on_device():
    logits = torch.zeros(1, 8)
    state = _PenaltyState(max_tokens=4, device=logits.device, dtype=torch.float32)
    state.add(2)
    state.add(5)
    state.add(2)

    state.apply(logits, presence_penalty=0.5, frequency_penalty=0.25)

    assert logits[0, 2].item() == -1.0
    assert logits[0, 5].item() == -0.75
    assert logits[0, 1].item() == 0.0


def test_no_cache_fallback_does_not_cat_each_step(monkeypatch: pytest.MonkeyPatch) -> None:
    model: Any = scripted_model([0, 1, 3], model_cls=NoCacheModel)
    tokenizer: Any = StopTokenizer()
    state = GenerationState()

    def _forbid_cat(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("no-cache fallback should use the preallocated buffer")

    monkeypatch.setattr(torch, "cat", _forbid_cat)
    with pytest.warns(UserWarning, match="no past_key_values"):
        generated_ids = generate_steered(
            model,
            tokenizer,
            torch.tensor([[0, 0]]),
            GenerationConfig(max_new_tokens=3, temperature=0.0),
            state,
        )

    assert generated_ids == [0, 1]
    assert state.finish_reason == "stop"

# ---------------------------------------------------------------------------
# The generation-concurrency guard: ``_gen_lock`` and the engine phase /
# transaction.  Extraction runs forward passes, so it takes
# the same exclusive section generation does; the transaction is the shared
# teardown backstop both sides rely on.
# ---------------------------------------------------------------------------


def _lock_free_for_another_thread(lock: Any) -> bool:
    """Whether a second thread can take ``lock`` (it is an RLock: the owner can re-enter)."""
    import threading

    acquired: list[bool] = []

    def _try() -> None:
        ok = lock.acquire(blocking=False)
        acquired.append(ok)
        if ok:
            lock.release()

    worker = threading.Thread(target=_try)
    worker.start()
    worker.join(5)
    return acquired == [True]


def test_extract_acquires_gen_lock_against_concurrent_generation(
    tiny_session: Any,
) -> None:
    """If ``_gen_lock`` is held by another thread (generation in flight),
    extract must raise ``ConcurrentExtractionError`` rather than reading
    the engine phase and racing the generation that's about to flip it."""
    import threading

    from drowse.errors import ConcurrentExtractionError

    held, release = threading.Event(), threading.Event()

    def _generation() -> None:
        # Simulate "generation just acquired the lock": phase still IDLE
        # because the flip happens after acquire on the generation side.
        with tiny_session._gen_lock:
            held.set()
            release.wait(5)

    worker = threading.Thread(target=_generation)
    worker.start()
    assert held.wait(5)
    try:
        with pytest.raises(ConcurrentExtractionError):
            tiny_session.extract("honest.deceptive")
    finally:
        release.set()
        worker.join(5)


def test_extract_releases_lock_on_path_through_phase_gate(tiny_session: Any) -> None:
    """When extract bails on the phase gate (a generation RUNNING), the lock
    it acquired must be released so subsequent extract attempts can proceed
    once the phase clears."""
    from drowse.errors import ConcurrentExtractionError

    tiny_session._engine.phase = GenState.RUNNING
    try:
        with pytest.raises(ConcurrentExtractionError):
            tiny_session.extract("honest.deceptive")
    finally:
        tiny_session._engine.phase = GenState.IDLE

    # Lock is back to free — another thread can take it.
    assert _lock_free_for_another_thread(tiny_session._gen_lock)
