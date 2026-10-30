"""Fake-model decode-loop regressions for ``generate_steered``."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from drowse.generation.decode import GenerationConfig, GenerationState, generate_steered
from tests.conftest import FakeLogitsModel

# Vocabulary: a bracket-pair thinking family (Mistral-3 Reasoning shape).
_THINK, _END_THINK, _A, _B, _EOS = 0, 1, 2, 3, 4


class _ThinkTokenizer:
    name_or_path = "think-tokenizer"
    vocab_size = 5
    eos_token_id = _EOS
    chat_template = None
    all_special_ids = [_THINK, _END_THINK, _EOS]
    added_tokens_encoder = {"[THINK]": _THINK, "[/THINK]": _END_THINK}
    _pieces = {_THINK: "[THINK]", _END_THINK: "[/THINK]", _A: "a", _B: "b", _EOS: ""}

    def batch_decode(self, ids: Any) -> list[str]:
        return [self._pieces[int(row[0])] for row in ids]

    def decode(self, ids: Any, skip_special_tokens: bool = False) -> str:
        return "".join(
            self._pieces[int(t)] for t in ids
            if not (skip_special_tokens and int(t) in self.all_special_ids)
        )


def _scripted(tokens: list[int]) -> Any:
    step = 0

    def _logits(_ids: torch.Tensor) -> torch.Tensor:
        nonlocal step
        tid = tokens[min(step, len(tokens) - 1)]
        step += 1
        out = torch.full((1, 1, 5), -100.0)
        out[0, 0, tid] = 100.0
        return out

    return FakeLogitsModel(
        _logits,
        with_past_key_values=True,
        config=SimpleNamespace(vocab_size=5),
        generation_config=SimpleNamespace(eos_token_id=_EOS),
    )


def _run(tokens: list[int], max_new_tokens: int) -> tuple[list[int], GenerationState]:
    tokenizer = _ThinkTokenizer()
    state = GenerationState()
    ids = generate_steered(
        _scripted(tokens), cast(Any, tokenizer), torch.tensor([[_A]]),
        GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0),
        state, on_token=lambda *_args: None,
    )
    return ids, state


@pytest.mark.parametrize(
    ("tokens", "max_new_tokens", "finish_reason"),
    [
        # The token budget runs out mid-thought.
        ([_THINK, _A, _B, _A, _B], 4, "length"),
        # EOS mid-thought on a non-channel family terminates the decode.
        ([_THINK, _A, _B, _EOS], 10, "stop"),
    ],
)
def test_decode_ending_inside_thinking_has_an_empty_response(
    tokens: list[int], max_new_tokens: int, finish_reason: str,
) -> None:
    ids, state = _run(tokens, max_new_tokens)
    assert state.finish_reason == finish_reason
    # The response region starts at the end, so the finalizer's
    # ``generated_ids[thinking_end_idx:]`` decodes nothing.
    assert state.thinking_end_idx == len(ids)
    assert ids[state.thinking_end_idx:] == []


def test_closed_thinking_anchors_the_response_after_the_delimiter() -> None:
    ids, state = _run([_THINK, _A, _END_THINK, _B, _EOS], 10)
    assert ids == [_THINK, _A, _END_THINK, _B]
    assert state.finish_reason == "stop"
    assert ids[state.thinking_end_idx:] == [_B]
