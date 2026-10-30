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


# ---------------------------------------------------------------------------
# The batched HF ``generate`` path samples from the decode loop's distribution
# ---------------------------------------------------------------------------


class _PlainTokenizer:
    name_or_path = "plain-tokenizer"
    eos_token_id = None
    chat_template = None
    added_tokens_encoder: dict[str, int] = {}


def _tiny_llama(**generation_defaults: Any) -> Any:
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=1,
        eos_token_id=63,
    )).eval()
    for name, value in generation_defaults.items():
        setattr(model.generation_config, name, value)
    return model


def test_batch_generation_config_reproduces_the_greedy_decode_loop() -> None:
    """A checkpoint's ``repetition_penalty`` / ``top_k`` defaults must not leak
    into the batched HF ``generate`` call: greedy batch output equals the
    decode loop's greedy output token for token."""
    from drowse.generation.decode import hf_generation_config

    # Qwen-style checkpoint defaults that the decode loop never applies.
    model = _tiny_llama(repetition_penalty=1.8, top_k=2, eos_token_id=63)
    prompt = torch.tensor([[1, 5, 9, 13, 5, 9]])
    config = GenerationConfig(max_new_tokens=12, temperature=0.0)

    loop_ids = generate_steered(
        model, cast(Any, _PlainTokenizer()), prompt, config, GenerationState(),
    )
    hf_config = hf_generation_config(
        config, vocab=64, eos_ids=[63], pad_id=0, steering_active=False,
    )
    with torch.inference_mode():
        out = model.generate(
            input_ids=prompt, attention_mask=torch.ones_like(prompt),
            generation_config=hf_config,
        )
    batch_ids = out[0, prompt.shape[1]:].tolist()
    assert batch_ids[:len(loop_ids)] == loop_ids


def test_batch_generation_config_pins_the_loop_sampler() -> None:
    from drowse.generation.decode import hf_generation_config

    model = _tiny_llama(repetition_penalty=1.1, top_k=20, no_repeat_ngram_size=3)
    hf_config = hf_generation_config(
        GenerationConfig(max_new_tokens=4, temperature=0.7, top_p=0.9),
        vocab=64, eos_ids=[63], pad_id=0, steering_active=True,
    )
    resolved, _ = model._prepare_generation_config(hf_config)
    assert resolved.repetition_penalty == 1.0
    assert resolved.no_repeat_ngram_size == 0
    assert resolved.top_k == 64  # the loop's 1024 cap, clipped to the vocab
    assert (resolved.temperature, resolved.top_p) == (0.7, 0.9)
    assert resolved.remove_invalid_values is True


def test_unneutralizable_checkpoint_processors_are_reported() -> None:
    from drowse.generation.decode import hf_unneutralizable_processors

    assert hf_unneutralizable_processors(_tiny_llama()) == []
    assert hf_unneutralizable_processors(
        _tiny_llama(suppress_tokens=[3], min_p=0.05),
    ) == ["min_p", "suppress_tokens"]
