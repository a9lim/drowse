"""PromptRenderer: model type, scene grammar, system prompt, render, seat stops."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from drowse.errors import RoleBaselineMismatchWarning
from drowse.generation.decode import build_chat_input
from drowse.generation.render import PromptRenderer, resolve_model_type
from drowse.generation.sampling import INHERIT_SYSTEM_PROMPT
from drowse.loom.tree import LoomTree
from drowse.model.chat import ChatTraits
from tests.support.tiny_models import byte_tokenizer


class _EchoTokenizer:
    """Minimal tokenizer: encodes text to a tensor of its char codes."""

    name_or_path = "echo-tokenizer"

    def encode(self, text: str, return_tensors: str | None = None) -> Any:
        ids = [ord(c) for c in text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids


def _decode_echo(ids: torch.Tensor) -> str:
    return "".join(chr(int(t)) for t in ids[0])


def _renderer(
    tokenizer: Any,
    tree: LoomTree,
    *,
    model: Any = None,
    system_prompt: dict[str, str | None] | None = None,
    role: dict[str, str | None] | None = None,
) -> PromptRenderer:
    system = system_prompt if system_prompt is not None else {"value": None}
    steer = role if role is not None else {"value": None}
    return PromptRenderer(
        tokenizer=tokenizer,
        model=model,
        traits=ChatTraits(tokenizer),
        device=torch.device("cpu"),
        history=lambda: tree,
        default_system_prompt=lambda: system["value"],
        steering_role=lambda: steer["value"],
    )


def test_raw_render_feeds_flat_active_path() -> None:
    """raw=True walks the loom tree as flat text — no chat template, no role
    markers — and appends the call's own input."""
    tree = LoomTree()
    u1 = tree.add_user_turn("once upon a ")
    a1 = tree.begin_assistant(u1)
    tree.finalize_assistant(a1, text="time")
    renderer = PromptRenderer(
        tokenizer=_EchoTokenizer(),
        model=None,
        traits=SimpleNamespace(),  # pyright: ignore[reportArgumentType]
        device=torch.device("cpu"),
        history=lambda: tree,
        default_system_prompt=lambda: None,
        steering_role=lambda: None,
    )

    ids = renderer.prepare_input(" the fox", raw=True, parent_node_id=a1)
    assert _decode_echo(ids) == "once upon a time the fox"
    # Stateless: the tree is ignored — only the input string is encoded.
    ids = renderer.prepare_input(
        " the fox", raw=True, stateless=True, parent_node_id=a1,
    )
    assert _decode_echo(ids) == " the fox"
    # Empty input is a bare continuation — just the flattened buffer.
    ids = renderer.prepare_input("", raw=True, parent_node_id=a1)
    assert _decode_echo(ids) == "once upon a time"


def test_model_type_prefers_the_text_config() -> None:
    wrapped = SimpleNamespace(config=SimpleNamespace(
        model_type="gemma3", text_config=SimpleNamespace(model_type="gemma3_text"),
    ))
    plain = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    untyped_text = SimpleNamespace(config=SimpleNamespace(
        model_type="mistral3", text_config=SimpleNamespace(model_type=None),
    ))
    assert resolve_model_type(wrapped) == "gemma3_text"
    assert resolve_model_type(plain) == "llama"
    assert resolve_model_type(untyped_text) == "mistral3"
    assert resolve_model_type(None) is None


def test_scene_grammar_resolves_once_and_is_none_without_a_template() -> None:
    tok = byte_tokenizer("llama")
    renderer = _renderer(tok, LoomTree(), model=SimpleNamespace(
        config=SimpleNamespace(model_type="llama"),
    ))
    grammar = renderer.scene_grammar
    assert grammar is not None
    assert renderer.scene_grammar is grammar

    bare = byte_tokenizer("llama")
    bare.chat_template = None
    assert _renderer(bare, LoomTree()).scene_grammar is None


def test_inherited_system_prompt_reads_the_live_default() -> None:
    system: dict[str, str | None] = {"value": "first"}
    renderer = _renderer(byte_tokenizer("llama"), LoomTree(), system_prompt=system)
    assert renderer.resolve_system_prompt(INHERIT_SYSTEM_PROMPT) == "first"
    system["value"] = "second"  # session.config is replaced, not mutated
    assert renderer.resolve_system_prompt(INHERIT_SYSTEM_PROMPT) == "second"
    assert renderer.resolve_system_prompt(None) is None
    assert renderer.resolve_system_prompt("explicit") == "explicit"


def test_render_matches_build_chat_input_over_the_loom_path() -> None:
    tok = byte_tokenizer("llama")
    tree = LoomTree()
    u1 = tree.add_user_turn("hello")
    a1 = tree.begin_assistant(u1)
    tree.finalize_assistant(a1, text="hi there")
    renderer = _renderer(
        tok, tree, system_prompt={"value": "be brief"},
        model=SimpleNamespace(config=SimpleNamespace(model_type="llama")),
    )

    ids = renderer.prepare_input("again", parent_node_id=a1, to_device=False)
    expected = build_chat_input(
        tok,
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "again"},
        ],
        "be brief",
    )
    assert ids.tolist() == expected.tolist()


def test_steering_role_wins_the_generation_header_and_warns_on_conflict() -> None:
    tok = byte_tokenizer("llama")
    tree = LoomTree()
    role: dict[str, str | None] = {"value": "pirate"}
    renderer = _renderer(
        tok, tree, role=role,
        model=SimpleNamespace(config=SimpleNamespace(model_type="llama")),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        steered = renderer.prepare_input("hi", to_device=False)
    labeled = renderer.prepare_input(
        "hi", to_device=False, assistant_role="pirate",
    )
    assert steered.tolist() == labeled.tolist()

    with pytest.warns(RoleBaselineMismatchWarning):
        conflicting = renderer.prepare_input(
            "hi", to_device=False, assistant_role="narrator",
        )
    assert conflicting.tolist() == steered.tolist()

    # Without a scope role the family's default header renders.
    role["value"] = None
    plain = renderer.prepare_input("hi", to_device=False)
    assert plain.tolist() != steered.tolist()
