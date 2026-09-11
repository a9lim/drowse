"""Regressions for tokenizer isolation and retained conversation buffers."""

from __future__ import annotations

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from drowse.core import generation
from drowse.core.hooks import HiddenCapture, SteeringManager
from drowse.core.session import DrowseSession
from drowse.core.scene import SeatWrapper, TurnGrammar


def tokenizer(swapped: bool = False) -> PreTrainedTokenizerFast:
    vocab = {"[UNK]": 0, "first": 2 if swapped else 1, "second": 1 if swapped else 2}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", chat_template="first")


@pytest.fixture(autouse=True)
def empty_caches():
    generation._chat_input_cache.clear()
    generation._token_table_cache.clear()
    yield
    generation._chat_input_cache.clear()
    generation._token_table_cache.clear()


def test_same_name_and_vocabulary_size_do_not_alias_token_decoding() -> None:
    first, second = tokenizer(), tokenizer(swapped=True)
    assert first.name_or_path == second.name_or_path
    assert first.vocab_size == second.vocab_size
    assert generation._get_token_table(first, 3)[1] == "first"
    assert generation._get_token_table(second, 3)[1] == "second"


def test_chat_render_cache_changes_with_live_template() -> None:
    tok = tokenizer()
    messages = [{"role": "user", "content": "private fixture"}]
    assert generation.build_chat_input(tok, messages).tolist() == [[1]]
    tok.chat_template = "second"
    assert generation.build_chat_input(tok, messages).tolist() == [[2]]


def test_chat_render_cache_changes_with_special_token_template_inputs() -> None:
    tok = tokenizer()
    tok.chat_template = "{{ bos_token }}"
    tok.bos_token = "first"
    messages = [{"role": "user", "content": "private fixture"}]
    assert generation.build_chat_input(tok, messages).tolist() == [[1]]
    tok.bos_token = "second"
    assert generation.build_chat_input(tok, messages).tolist() == [[2]]


def test_chat_render_cache_drops_history_when_tokenizer_is_collected() -> None:
    tok = tokenizer()
    reference = weakref.ref(tok)
    generation.build_chat_input(tok, [{"role": "user", "content": "private fixture"}])
    generation._get_token_table(tok, 3)
    assert generation._chat_input_cache
    del tok
    gc.collect()
    assert reference() is None
    assert not generation._chat_input_cache
    assert not generation._token_table_cache


def test_token_table_rebuilds_when_tokens_are_added() -> None:
    tok = tokenizer()
    generation._get_token_table(tok, 4)
    tok.add_tokens(["third"])
    assert generation._get_token_table(tok, 4)[3] == "third"


def test_chat_cache_uses_the_actual_scene_grammar() -> None:
    tok = tokenizer()
    scene = TurnGrammar(
        model_type="fixture", prelude="", user=SeatWrapper("", "", "", ""),
        assistant=SeatWrapper("", "", "", ""), system=None,
        system_fold_sep=None, gen_extra="",
    )
    messages = [{"role": "user", "content": "first"}]
    assert generation.build_chat_input(tok, messages, scene=scene).tolist() == [[1]]
    changed = replace(scene, prelude="second ")
    assert generation.build_chat_input(tok, messages, scene=changed).tolist() == [[2, 1]]


def test_chat_cache_evicts_by_bytes_and_returns_independent_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    tok = tokenizer()
    monkeypatch.setattr(generation, "_CHAT_INPUT_CACHE_MAX_BYTES", 5000)
    calls = 0
    original = tok.apply_chat_template

    def render(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tok, "apply_chat_template", render)
    first = [{"role": "user", "content": "a" * 500}]
    second = [{"role": "user", "content": "b" * 500}]
    generation.build_chat_input(tok, first).fill_(2)
    assert generation.build_chat_input(tok, first).tolist() == [[1]]
    assert calls == 1
    generation.build_chat_input(tok, second)
    generation.build_chat_input(tok, first)
    assert calls == 3


def test_parallel_chat_cache_use_stays_isolated_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizers = [tokenizer(), tokenizer(swapped=True)]
    monkeypatch.setattr(generation, "_CHAT_INPUT_CACHE_MAX", 2)

    def render(index: int):
        return generation.build_chat_input(
            tokenizers[index % 2], [{"role": "user", "content": str(index % 8)}],
        ).item()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(render, range(64)))
    assert results == [1 if index % 2 == 0 else 2 for index in range(64)]
    assert len(generation._chat_input_cache) <= 2


def test_cached_inputs_preserve_real_transformer_generation() -> None:
    tok = tokenizer()
    model: Any = LlamaForCausalLM(LlamaConfig(
        vocab_size=3, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32, eos_token_id=None, pad_token_id=0,
    )).eval()
    messages = [{"role": "user", "content": "first"}]
    direct: Any = tok.apply_chat_template(messages, return_tensors="pt")
    if not isinstance(direct, torch.Tensor):
        direct = direct["input_ids"]
    assert isinstance(direct, torch.Tensor)
    cold = generation.build_chat_input(tok, messages)
    warm = generation.build_chat_input(tok, messages)
    with torch.inference_mode():
        outputs = [model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=4, do_sample=False,
        ) for ids in (direct, cold, warm)]
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[1], outputs[2])


def test_large_chat_inputs_are_not_retained_in_the_render_cache() -> None:
    tok = tokenizer()
    result = generation.build_chat_input(tok, [{"role": "user", "content": "x" * (9 * 1024 * 1024)}])
    assert result.tolist() == [[1]]
    assert not generation._chat_input_cache


def test_session_close_releases_reusable_generation_device_cache() -> None:
    session: Any = DrowseSession.__new__(DrowseSession)
    session._steering = SteeringManager()
    session._capture_handles = []
    session._capture_buffers = {}
    session._capture = HiddenCapture()
    session._profiles = {}
    session._manifolds = {}
    session._tokenizer = tokenizer()
    generation.build_chat_input(session._tokenizer, [{"role": "user", "content": "private"}])
    tensor = torch.ones(1024)
    reference = weakref.ref(tensor)
    session._generation_static_cache = SimpleNamespace(keys=tensor)
    session._generation_static_cache_len = 1024
    del tensor
    session.close()
    gc.collect()
    assert reference() is None
    assert session._generation_static_cache is None
    assert session._generation_static_cache_len == 0
    assert not generation._chat_input_cache
