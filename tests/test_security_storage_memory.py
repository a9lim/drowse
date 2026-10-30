"""Regressions for tokenizer isolation and retained conversation buffers."""

from __future__ import annotations

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.lens import stub_lens_instrument
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from drowse.generation import decode as generation
from drowse.model import chat as chat_mod
from drowse.model.chat import ChatTraits
from drowse.readout.capture_buffer import HiddenCapture
from drowse.steering.hooks import SteeringManager
from drowse.session.session import DrowseSession
from drowse.model.scene import SeatWrapper, TurnGrammar


def tokenizer(swapped: bool = False) -> PreTrainedTokenizerFast:
    vocab = {"[UNK]": 0, "first": 2 if swapped else 1, "second": 1 if swapped else 2}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", chat_template="first")


def test_token_text_is_owned_by_each_tokenizer_traits() -> None:
    first, second = tokenizer(), tokenizer(swapped=True)
    assert first.name_or_path == second.name_or_path
    assert first.vocab_size == second.vocab_size
    assert ChatTraits(first).token_text(1) == "first"
    assert ChatTraits(second).token_text(1) == "second"


def test_chat_render_cache_changes_with_live_template() -> None:
    tok = tokenizer()
    traits = ChatTraits(tok)
    messages = [{"role": "user", "content": "private fixture"}]
    assert generation.build_chat_input(tok, messages, traits=traits).tolist() == [[1]]
    tok.chat_template = "second"
    assert generation.build_chat_input(tok, messages, traits=traits).tolist() == [[2]]


def test_chat_render_cache_changes_with_special_token_template_inputs() -> None:
    tok = tokenizer()
    tok.chat_template = "{{ bos_token }}"
    tok.bos_token = "first"
    traits = ChatTraits(tok)
    messages = [{"role": "user", "content": "private fixture"}]
    assert generation.build_chat_input(tok, messages, traits=traits).tolist() == [[1]]
    tok.bos_token = "second"
    assert generation.build_chat_input(tok, messages, traits=traits).tolist() == [[2]]


def test_chat_cache_uses_the_actual_scene_grammar() -> None:
    tok = tokenizer()
    traits = ChatTraits(tok)
    scene = TurnGrammar(
        model_type="fixture", prelude="", user=SeatWrapper("", "", "", ""),
        assistant=SeatWrapper("", "", "", ""), system=None,
        system_fold_sep=None, gen_extra="",
    )
    messages = [{"role": "user", "content": "first"}]
    assert generation.build_chat_input(
        tok, messages, scene=scene, traits=traits,
    ).tolist() == [[1]]
    changed = replace(scene, prelude="second ")
    assert generation.build_chat_input(
        tok, messages, scene=changed, traits=traits,
    ).tolist() == [[2, 1]]


def test_chat_cache_evicts_by_bytes_and_returns_independent_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    tok = tokenizer()
    traits = ChatTraits(tok)
    monkeypatch.setattr(chat_mod, "_CHAT_INPUT_CACHE_MAX_BYTES", 5000)
    calls = 0
    original = tok.apply_chat_template

    def render(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tok, "apply_chat_template", render)
    first = [{"role": "user", "content": "a" * 500}]
    second = [{"role": "user", "content": "b" * 500}]
    generation.build_chat_input(tok, first, traits=traits).fill_(2)
    assert generation.build_chat_input(tok, first, traits=traits).tolist() == [[1]]
    assert calls == 1
    generation.build_chat_input(tok, second, traits=traits)
    generation.build_chat_input(tok, first, traits=traits)
    assert calls == 3


def test_parallel_chat_cache_use_stays_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    tok = tokenizer()
    traits = ChatTraits(tok)
    monkeypatch.setattr(chat_mod, "_CHAT_INPUT_CACHE_MAX", 2)

    def render(index: int):
        return generation.build_chat_input(
            tok, [{"role": "user", "content": str(index % 8)}], traits=traits,
        ).item()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(render, range(64)))
    assert results == [1] * 64
    assert len(traits._render_cache) <= 2


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
    traits = ChatTraits(tok)
    cold = generation.build_chat_input(tok, messages, traits=traits)
    warm = generation.build_chat_input(tok, messages, traits=traits)
    with torch.inference_mode():
        outputs = [model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=4, do_sample=False,
        ) for ids in (direct, cold, warm)]
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[1], outputs[2])


def test_large_chat_inputs_are_not_retained_in_the_render_cache() -> None:
    tok = tokenizer()
    traits = ChatTraits(tok)
    result = generation.build_chat_input(
        tok, [{"role": "user", "content": "x" * (9 * 1024 * 1024)}], traits=traits,
    )
    assert result.tolist() == [[1]]
    assert not traits._render_cache


def test_session_close_releases_reusable_generation_device_cache() -> None:
    session: Any = DrowseSession.__new__(DrowseSession)
    session._lens_instrument = stub_lens_instrument(session)
    session._steering = SteeringManager()
    session._capture_handles = []
    session._capture_buffers = {}
    session._capture = HiddenCapture()
    session._profiles = {}
    session._manifolds = {}
    session._tokenizer = tokenizer()
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
