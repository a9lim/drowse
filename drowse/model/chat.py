"""Chat-template and tokenizer facts, computed once per session.

:class:`ChatTraits` is the one place the engine reads a tokenizer's chat
behavior: its stop-token set, its thinking delimiters, whether its template
takes ``enable_thinking``, whether it is a base (template-less) model, the
logits-cap kwarg its model's ``forward`` accepts, a lazy token-text table, and
a bounded render cache for chat inputs.  A session builds it once and hands it
to the decode loop and the renderer; it dies with the session, so nothing
about one tokenizer outlives it or leaks into another.
"""

from __future__ import annotations

import inspect
import logging
import threading
from collections import OrderedDict
from typing import Any, NamedTuple, cast

import torch

log = logging.getLogger(__name__)

# End-of-turn specials some families add as tokens without listing them in
# ``generation_config.eos_token_id`` (Gemma's ``<end_of_turn>``, …).
_EOT_TOKEN_NAMES = frozenset({
    "<end_of_turn>",
    "<|endoftext|>",
    "<|end|>",
    "<|eot_id|>",
    "<turn|>",
    "<|im_end|>",
})

_CHAT_INPUT_CACHE_MAX = 128
_CHAT_INPUT_CACHE_MAX_BYTES = 8 * 1024 * 1024


class ThinkDelimiters(NamedTuple):
    """A family's reasoning-section token ids.

    ``start_id`` opens a thinking section (``None`` when the generation prompt
    already opened it — Qwen-style — or for channel formats that start inside
    one); ``end_id`` closes it; ``response_start_id`` marks where response
    content begins after a multi-token separator (gpt-oss
    ``<|channel|>…<|message|>``); ``starts_in_thinking`` is true when the first
    generated token is already thinking content.
    """

    start_id: int | None = None
    end_id: int | None = None
    response_start_id: int | None = None
    starts_in_thinking: bool = False


NO_THINKING = ThinkDelimiters()


def chat_eos_ids(model: Any, tokenizer: Any) -> set[int]:
    """Every stop id: config eos ∪ tokenizer eos ∪ the template's end-of-turn specials."""
    eos_ids: set[int] = set()
    gen_cfg = getattr(model, "generation_config", None)
    eid = getattr(gen_cfg, "eos_token_id", None)
    if isinstance(eid, int):
        eos_ids.add(eid)
    elif eid is not None:
        eos_ids.update(eid)
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if tokenizer_eos is not None:
        eos_ids.add(int(tokenizer_eos))
    for tok_str, tok_id in getattr(tokenizer, "added_tokens_encoder", {}).items():
        if tok_str in _EOT_TOKEN_NAMES:
            eos_ids.add(tok_id)
    return eos_ids


def takes_enable_thinking(tokenizer: Any) -> bool:
    """Whether the chat template reads an ``enable_thinking`` switch."""
    template = getattr(tokenizer, "chat_template", None) or ""
    return isinstance(template, str) and "enable_thinking" in template


def logits_keep_kwarg(model: Any) -> str | None:
    """The ``logits_to_keep``/``num_logits_to_keep`` kwarg ``model.forward`` names.

    ``None`` for a ``**kwargs``-only signature or a model with no ``forward``
    (a test double called through ``__call__``); the decode loop's first
    forward still retries without the kwarg if a model rejects it at runtime.
    """
    fwd = getattr(model, "forward", None)
    if fwd is None:
        return None
    try:
        params = inspect.signature(fwd).parameters
    except (ValueError, TypeError):  # pragma: no cover — C/builtin forwards
        return None
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return name
    return None


def _render(tokenizer: Any, messages: list[dict[str, Any]], **kwargs: Any) -> str:
    return cast(str, tokenizer.apply_chat_template(messages, tokenize=False, **kwargs))


def _channel_delimiters(tokenizer: Any) -> ThinkDelimiters | None:
    """gpt-oss channel format: ``<|channel|>analysis<|message|>…``, no switch."""
    added = getattr(tokenizer, "added_tokens_encoder", {})
    channel_id = added.get("<|channel|>")
    message_id = added.get("<|message|>")
    if channel_id is None or message_id is None:
        return None
    try:
        gen_prompt = _render(
            tokenizer, [{"role": "user", "content": "hi"}], add_generation_prompt=True,
        )
    except Exception:
        gen_prompt = ""
    starts_in = "<|channel|>" in gen_prompt
    return ThinkDelimiters(
        None if starts_in else channel_id, channel_id, message_id, starts_in,
    )


def _bracket_delimiters(tokenizer: Any) -> ThinkDelimiters | None:
    """Mistral-3 Reasoning ``[THINK]…[/THINK]`` added-token pair, no switch."""
    added = getattr(tokenizer, "added_tokens_encoder", {})
    start_id = added.get("[THINK]")
    end_id = added.get("[/THINK]")
    if start_id is None or end_id is None:
        return None
    return ThinkDelimiters(start_id, end_id, None, False)


def detect_think_delimiters(tokenizer: Any) -> ThinkDelimiters:
    """Detect thinking delimiters by rendering a round-trip through the template.

    Templates without an ``enable_thinking`` switch are checked for the two
    forced-thinking styles (channel, bracket pair).  Switchable templates
    render an assistant turn carrying marker thinking + response content under
    each known message schema; the first special token between the markers is
    the end delimiter, the nearest one before the thinking marker the start
    delimiter, and a start delimiter already present in the generation prompt
    means the model starts inside thinking.
    """
    if not takes_enable_thinking(tokenizer):
        return _channel_delimiters(tokenizer) or _bracket_delimiters(tokenizer) or NO_THINKING

    think_marker = "XTHINKCONTENTX"
    response_marker = "XRESPONSECONTENTX"
    dummy_tc = [{"function": {"name": "x", "arguments": {}}}]
    # Gemma needs reasoning/reasoning_content + tool_calls; Qwen embeds
    # ``<think>…</think>`` in content.  Try each schema until one renders both.
    attempts = [
        {"role": "assistant", "reasoning_content": think_marker, "content": response_marker,
         "tool_calls": dummy_tc},
        {"role": "assistant", "reasoning": think_marker, "content": response_marker,
         "tool_calls": dummy_tc},
        {"role": "assistant", "thought": think_marker, "content": response_marker,
         "tool_calls": dummy_tc},
        {"role": "assistant", "thought": think_marker, "content": response_marker},
        {"role": "assistant", "reasoning_content": think_marker, "content": response_marker},
        {"role": "assistant", "content": f"<think>\n{think_marker}\n</think>\n{response_marker}"},
    ]
    added = getattr(tokenizer, "added_tokens_encoder", {})
    for asst_msg in attempts:
        try:
            rendered = _render(
                tokenizer, [{"role": "user", "content": "hi"}, asst_msg],
                enable_thinking=True,
            )
        except Exception:
            continue
        ti = rendered.find(think_marker)
        ri = rendered.find(response_marker)
        if ti < 0 or ri <= ti:
            continue
        between = rendered[ti + len(think_marker):ri]
        end_pos, end_id = len(between), None
        for tok_str, tok_id in added.items():
            pos = between.find(tok_str)
            if 0 <= pos < end_pos:
                end_pos, end_id = pos, tok_id
        if end_id is None:
            continue
        start_pos, start_tok, start_id = -1, None, None
        for tok_str, tok_id in added.items():
            pos = rendered.rfind(tok_str, 0, ti)
            if pos > start_pos:
                start_pos, start_tok, start_id = pos, tok_str, tok_id
        starts_in_thinking = False
        if start_tok is not None:
            try:
                gen_prompt = _render(
                    tokenizer, [{"role": "user", "content": "hi"}],
                    add_generation_prompt=True, enable_thinking=True,
                )
            except Exception:
                gen_prompt = ""
            if start_tok in gen_prompt:
                starts_in_thinking = True
                start_id = None  # the prompt opened it; nothing to detect
        return ThinkDelimiters(start_id, end_id, None, starts_in_thinking)

    log.warning("thinking supported but could not detect delimiters")
    return NO_THINKING


def _cache_key_bytes(value: Any) -> int:
    if isinstance(value, str):
        return 64 + 4 * len(value)
    if isinstance(value, tuple):
        return 64 + 8 * len(value) + sum(_cache_key_bytes(v) for v in value)
    return 64


class ChatTraits:
    """One tokenizer's (and its model's) chat behavior, read once."""

    def __init__(self, tokenizer: Any, model: Any = None) -> None:
        self.tokenizer = tokenizer
        self.eos_ids: frozenset[int] = frozenset(chat_eos_ids(model, tokenizer))
        self.think: ThinkDelimiters = detect_think_delimiters(tokenizer)
        self.takes_enable_thinking = takes_enable_thinking(tokenizer)
        self.is_base = getattr(tokenizer, "chat_template", None) is None
        self.logits_keep_kwarg = logits_keep_kwarg(model) if model is not None else None
        self._token_text: dict[int, str] = {}
        self._render_lock = threading.Lock()
        self._render_cache: OrderedDict[tuple[Any, ...], tuple[torch.Tensor, int]] = (
            OrderedDict()
        )
        self._render_bytes = 0

    @property
    def supports_thinking(self) -> bool:
        return self.think != NO_THINKING

    @property
    def thinking_optional(self) -> bool:
        """Whether a caller can turn thinking off (the template has the switch).

        Forced-thinking families (gpt-oss channels, Mistral-3 Reasoning's baked
        system prompt, Qwen3-Thinking's hardcoded ``<think>``) think regardless
        of the flag; the state machine still classifies their section.
        """
        return self.supports_thinking and self.takes_enable_thinking

    def adopt_eos(self, model: Any) -> None:
        """Make the model's ``generation_config`` stop on the chat-correct set.

        A multimodal checkpoint's extracted text sub-config can under-specify
        the end-of-turn tokens its template emits (Gemma-4: ``<eos>`` only, while
        every turn ends ``<turn|>``), so HF ``generate`` — the batched paths —
        would run past each turn.  The decode loop already stops on
        :attr:`eos_ids`; this makes both paths agree.
        """
        gen_cfg = getattr(model, "generation_config", None)
        if self.eos_ids and gen_cfg is not None:
            ids = sorted(self.eos_ids)
            gen_cfg.eos_token_id = ids if len(ids) > 1 else ids[0]

    def token_text(self, token_id: int) -> str:
        """The decoded text of one token id, cached for the session."""
        text = self._token_text.get(token_id)
        if text is None:
            text = cast(str, self.tokenizer.decode([token_id]))
            self._token_text[token_id] = text
        return text

    def cached_render(self, key: tuple[Any, ...]) -> torch.Tensor | None:
        with self._render_lock:
            entry = self._render_cache.get(key)
            if entry is None:
                return None
            self._render_cache.move_to_end(key)
            return entry[0].clone()

    def remember_render(self, key: tuple[Any, ...], tensor: torch.Tensor) -> None:
        """Keep a rendered input, bounded by entry count and bytes."""
        cost = _cache_key_bytes(key) + tensor.numel() * tensor.element_size() + 128
        if cost > _CHAT_INPUT_CACHE_MAX_BYTES:
            return
        with self._render_lock:
            old = self._render_cache.pop(key, None)
            if old is not None:
                self._render_bytes -= old[1]
            while self._render_cache and (
                len(self._render_cache) >= _CHAT_INPUT_CACHE_MAX
                or self._render_bytes + cost > _CHAT_INPUT_CACHE_MAX_BYTES
            ):
                _, (_, evicted) = self._render_cache.popitem(last=False)
                self._render_bytes -= evicted
            self._render_cache[key] = (tensor.clone(), cost)
            self._render_bytes += cost

    def clear_caches(self) -> None:
        """Drop the rendered-input cache and the token-text table."""
        with self._render_lock:
            self._render_cache.clear()
            self._render_bytes = 0
        self._token_text.clear()
