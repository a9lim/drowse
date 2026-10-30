"""Prefix KV-cache reuse + rollback against real transformers caches (CPU).

Each architecture exercises one rollback mechanism of
:class:`~drowse.generation.prefix_cache.PrefixCacheEntry`: full-attention
``crop``, sliding-window and linear-attention conv states through upstream past
recording, linear-attention recurrent states through the snapshot, and the
StaticCache write-counter rewind.  A reused prefix must decode exactly what a
full prefill decodes, across repeated reuses.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from drowse.generation.decode import GenerationConfig, GenerationState, generate_steered
from drowse.generation.prefix_cache import PrefixCacheEntry

_PREFIX = [1, 5, 9, 13, 5, 9, 7, 8, 11, 3]
_SUFFIXES = ([12, 4], [6, 10, 2], [12, 4])
_CONFIG = GenerationConfig(max_new_tokens=6, temperature=0.0)

_TINY = dict(
    hidden_size=32, intermediate_size=64, num_attention_heads=4,
    num_key_value_heads=2, vocab_size=64, pad_token_id=0, bos_token_id=1,
    eos_token_id=63,
)
_ARCHS: dict[str, dict[str, Any]] = {
    "llama": dict(num_hidden_layers=2),
    # A window shorter than the prefix: rewinding needs the recorded tail.
    "mistral": dict(num_hidden_layers=2, sliding_window=4),
    # Short-conv layers: linear-attention conv states only.
    "lfm2": dict(
        num_hidden_layers=4,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
    ),
    # Gated delta-net layers: conv *and* recurrent states.
    "qwen3_next": dict(
        num_hidden_layers=4, head_dim=8,
        linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        num_experts=2, num_experts_per_tok=1, moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        layer_types=[
            "linear_attention", "full_attention",
            "linear_attention", "full_attention",
        ],
    ),
}


class _PlainTokenizer:
    name_or_path = "plain-tokenizer"
    eos_token_id = None
    chat_template = None
    added_tokens_encoder: dict[str, int] = {}


def _model(model_type: str) -> Any:
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.for_model(model_type, **_TINY, **_ARCHS[model_type])
    return AutoModelForCausalLM.from_config(cfg).eval()


def _decode(model: Any, ids: list[int], **kwargs: Any) -> list[int]:
    return generate_steered(
        model, cast(Any, _PlainTokenizer()), torch.tensor([ids]), _CONFIG,
        GenerationState(), **kwargs,
    )


@pytest.mark.parametrize("model_type", list(_ARCHS))
def test_dynamic_prefix_reuse_matches_full_prefill(model_type: str) -> None:
    model = _model(model_type)
    prefix = torch.tensor([_PREFIX])
    with torch.inference_mode():
        prefill = model(input_ids=prefix, use_cache=True)
    entry = PrefixCacheEntry.from_prefill(prefix, prefill.past_key_values)

    for suffix in _SUFFIXES:
        expected = _decode(model, _PREFIX + suffix)
        hit = entry.suffix_for(
            torch.tensor([_PREFIX + suffix]),
            static_eligible=False, required_max_new_tokens=None,
        )
        assert hit is not None and hit.tolist() == [suffix]
        reused = _decode(
            model, suffix,
            past_key_values=entry.past_key_values,
            cache_position_offset=entry.prefix_len,
        )
        assert reused == expected
        assert entry.rollback()


def test_static_prefix_reuse_survives_repeated_reuse() -> None:
    from drowse.generation.static_cache import make_static_cache

    model = _model("llama")
    prefix = torch.tensor([_PREFIX])
    max_cache_len = len(_PREFIX) + 3 + _CONFIG.max_new_tokens
    cache = make_static_cache(
        model, max_cache_len=max_cache_len, device="cpu", dtype=torch.float32,
    )
    with torch.inference_mode():
        model(
            input_ids=prefix, past_key_values=cache, use_cache=True,
            cache_position=torch.arange(len(_PREFIX)),
        )
    entry = PrefixCacheEntry.from_prefill(
        prefix, cache, static=True, max_cache_len=max_cache_len,
    )

    for suffix in _SUFFIXES:
        expected = _decode(model, _PREFIX + suffix)
        reused = _decode(
            model, suffix,
            past_key_values=entry.past_key_values,
            cache_position_offset=entry.prefix_len,
            use_static_cache=True,
        )
        assert reused == expected
        assert entry.rollback()


def test_static_entry_misses_when_ineligible_or_too_long() -> None:
    entry = PrefixCacheEntry(
        prefix_ids_cpu=torch.tensor([1, 2]), past_key_values=object(),
        prefix_len=2, static=True, max_cache_len=5,
    )
    ids = torch.tensor([[1, 2, 3]])
    assert entry.suffix_for(ids, static_eligible=False, required_max_new_tokens=1) is None
    assert entry.suffix_for(ids, static_eligible=True, required_max_new_tokens=3) is None
    hit = entry.suffix_for(ids, static_eligible=True, required_max_new_tokens=2)
    assert hit is not None and hit.tolist() == [[3]]
    # A prefix mismatch or an empty suffix is a miss.
    assert entry.suffix_for(torch.tensor([[1, 9, 3]]), static_eligible=True,
                            required_max_new_tokens=1) is None
    assert entry.suffix_for(torch.tensor([[1, 2]]), static_eligible=True,
                            required_max_new_tokens=1) is None


def test_rollback_reports_a_cache_it_cannot_rewind() -> None:
    entry = PrefixCacheEntry(
        prefix_ids_cpu=torch.tensor([1]), past_key_values=object(), prefix_len=1,
    )
    assert entry.rollback() is False
