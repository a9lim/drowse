"""Prompt rendering — one place that turns a conversation into ``input_ids``.

:class:`PromptRenderer` owns everything a generation needs to know about the
loaded chat template: the resolved ``model_type`` (multimodal ``text_config``
unwrapped), the session's validated scene grammar (the cast model's stitcher,
autopsied lazily once), the system-prompt sentinel resolution, the render of a
loom path plus a new turn into ``input_ids``, and the seat-specific stop
strings a non-assistant generation needs.

The renderer reads three pieces of live session state through getters, each
an explicit constructor dependency: the conversation history (the session's
current loom tree, which ``restore_tree`` swaps), the session default system
prompt (``session.config`` is replaced, not mutated), and the steering scope's
role baseline (set and cleared by the steering scope around a generation).
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast

import torch

from drowse.errors import RoleBaselineMismatchWarning
from drowse.generation.decode import build_chat_input
from drowse.generation.sampling import INHERIT_SYSTEM_PROMPT, InheritSystemPrompt
from drowse.model.scene import (
    SceneGrammarError,
    TurnGrammar,
    extract_turn_grammar,
    validate_turn_grammar,
)

if TYPE_CHECKING:
    from drowse.model.chat import ChatTraits

_log = logging.getLogger(__name__)


class ConversationHistory(Protocol):
    """The two reads a render makes of the loom tree."""

    def flat_text(self, leaf_id: str | None = None) -> str: ...

    def messages_for(
        self, leaf_id: str | None = None, *, with_labels: bool = False,
    ) -> list[dict[str, Any]]: ...


def resolve_model_type(model: Any) -> str | None:
    """``model.config.model_type``, preferring a multimodal ``text_config``'s."""
    model_cfg = getattr(model, "config", None)
    if model_cfg is None:
        return None
    text_cfg = getattr(model_cfg, "text_config", None)
    model_type = (
        getattr(text_cfg, "model_type", None) if text_cfg is not None else None
    )
    return model_type or getattr(model_cfg, "model_type", None)


class PromptRenderer:
    """Render conversations for one loaded model and tokenizer."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        traits: "ChatTraits",
        device: torch.device,
        history: Callable[[], ConversationHistory],
        default_system_prompt: Callable[[], str | None],
        steering_role: Callable[[], str | None],
    ) -> None:
        self._tokenizer = tokenizer
        self._traits = traits
        self._device = device
        self._history = history
        self._default_system_prompt = default_system_prompt
        self._steering_role = steering_role
        self.model_type: str | None = resolve_model_type(model)
        self._scene_grammar: TurnGrammar | None = None
        self._scene_grammar_resolved = False

    @property
    def scene_grammar(self) -> TurnGrammar | None:
        """The validated turn grammar for the loaded chat template, or None.

        Lazily runs the template autopsy + byte-exact round-trip validation
        (``model/scene.py``) once.  A passing grammar makes the stitcher the
        render authority — arbitrary seat sequences, per-turn cast labels,
        generation prompts on either seat — with renders bit-identical to
        ``apply_chat_template`` on standard alternating conversations.
        ``None`` = scene mode unavailable (base model, label-free family like
        mistral, or a template shape that defeated the autopsy); rendering
        falls back to the chat-template paths and a one-time warning names
        the reason.
        """
        if not self._scene_grammar_resolved:
            self._scene_grammar_resolved = True
            template = getattr(self._tokenizer, "chat_template", None)
            if template:
                think = (
                    ("<think>", "</think>") if "<think>" in template else None
                )
                try:
                    grammar = extract_turn_grammar(
                        self._tokenizer,
                        self.model_type or "",
                        think_delimiters=think,
                    )
                    validate_turn_grammar(grammar, self._tokenizer)
                    self._scene_grammar = grammar
                except SceneGrammarError as e:
                    _log.warning(
                        "scene mode unavailable for this model (%s); "
                        "rendering falls back to the chat template", e,
                    )
        return self._scene_grammar

    def resolve_system_prompt(
        self, system_prompt: str | None | InheritSystemPrompt,
    ) -> str | None:
        """The effective system prompt: the session default for the sentinel."""
        if isinstance(system_prompt, InheritSystemPrompt):
            return self._default_system_prompt()
        return system_prompt

    def prepare_input(
        self,
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
        system_prompt: str | None | InheritSystemPrompt = INHERIT_SYSTEM_PROMPT,
    ) -> torch.Tensor:
        """Render the loom path to ``parent_node_id`` plus ``input`` to ids.

        ``input`` is a new user turn (``str``), a bare continuation (``None``,
        no new turn: the generation prompt opens directly after the existing
        history), or an explicit message list.  ``stateless`` skips the tree
        walk.  ``raw`` is the flat (base-model / completion) path: no chat
        template, the active-path text verbatim plus ``input``.
        """
        if raw and (isinstance(input, str) or input is None):
            prefix = "" if stateless else self._history().flat_text(parent_node_id)
            encoded = self._tokenizer.encode(
                prefix + (input or ""), return_tensors="pt",
            )
            ids = cast(torch.Tensor, encoded)  # return_tensors="pt" gives a Tensor
            return ids.to(self._device) if to_device else ids
        if isinstance(input, str) or input is None:
            # The model sees the conversation along the requested path, each
            # prior turn under the ``role_label`` it was sent with.
            prior: list[dict[str, Any]] = (
                [] if stateless
                else self._history().messages_for(parent_node_id, with_labels=True)
            )
            if input is None:
                messages = prior
            else:
                messages = prior + [
                    {"role": "user", "content": input, "label": user_role}
                ]
        elif isinstance(input, list):
            messages = list(input)
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")
        gen_role = self._generation_label(
            gen_seat, user_role, assistant_role, warn=add_generation_prompt,
        )
        any_label = gen_role is not None or any(
            isinstance(m, dict) and m.get("label") for m in messages
        )
        ids = build_chat_input(
            self._tokenizer, messages, self.resolve_system_prompt(system_prompt),
            thinking=thinking,
            add_generation_prompt=add_generation_prompt,
            gen_role=gen_role,
            model_type=self.model_type if any_label else None,
            scene=self.scene_grammar,
            traits=self._traits,
            gen_seat=gen_seat,
        )
        return ids.to(self._device) if to_device else ids

    def _generation_label(
        self,
        gen_seat: str,
        user_role: str | None,
        assistant_role: str | None,
        *,
        warn: bool,
    ) -> str | None:
        """The cast label the generation prompt opens under.

        Assistant seat: a role-augmented steering scope's baseline wins so
        steer baseline matches extract baseline; otherwise this send's
        ``assistant_role``.  User seat: the ``user_role`` box (the elicitation
        baseline is an assistant-seat construct).
        """
        if gen_seat != "assistant":
            return user_role
        steer_role = self._steering_role()
        if steer_role is None:
            return assistant_role
        if warn and assistant_role is not None and steer_role != assistant_role:
            # The caller asked for a different header than the steering
            # scope's role baseline; say so rather than silently rendering a
            # turn they didn't label.
            warnings.warn(
                f"steering scope implies role {steer_role!r} but this "
                f"send labels the generated turn {assistant_role!r}; "
                f"the steering role wins for the generation header",
                RoleBaselineMismatchWarning,
                stacklevel=3,
            )
        return steer_role

    def seat_stop_strings(
        self,
        stop_list: list[str] | None,
        *,
        gen_seat: str,
        raw: bool,
    ) -> list[str] | None:
        """:func:`seat_stop_strings` under this renderer's scene grammar."""
        if raw or gen_seat == "assistant":
            return stop_list
        return seat_stop_strings(
            self.scene_grammar, stop_list, gen_seat=gen_seat, raw=raw,
        )


def seat_stop_strings(
    grammar: TurnGrammar | None,
    stop_list: list[str] | None,
    *,
    gen_seat: str,
    raw: bool,
) -> list[str] | None:
    """Add the gen seat's close segment as a stop string when needed.

    Generating into a seat stops on that seat's terminator.  On every
    validated family the two seats share a terminator, which the engine's
    EOS-id union already catches — so the assistant path (and any
    shared-close family) adds nothing and keeps its zero-cost no-stop-list
    fast path.  Only a non-assistant-seat gen on a family whose seat closes
    *differ* (gpt-oss: user ``<|end|>`` vs assistant ``<|return|>``) gains a
    stop string.  Raw / flat mode has no seat closes.
    """
    if raw or gen_seat == "assistant" or grammar is None:
        return stop_list
    close = grammar.seat(gen_seat).close.strip()
    if not close or close == grammar.assistant.close.strip():
        return stop_list
    out = list(stop_list) if stop_list else []
    if close not in out:
        out.append(close)
    return out
