"""Shared request helpers for OpenAI, Ollama, and native streaming routes."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from drowse.session.session import DrowseSession

from drowse.errors import DrowseError
from drowse.model.loading import get_unembedding
from drowse.generation.sampling import SamplingConfig
from drowse.steering.terms import Steering


class UnsupportedContentError(ValueError, DrowseError):
    """Non-text content parts submitted to a text-only endpoint."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


class InvalidSamplingError(ValueError, DrowseError):
    def user_message(self) -> tuple[int, str]:
        return 400, str(self)


def validate_logit_bias(sampling: SamplingConfig | None, session: DrowseSession) -> None:
    if sampling is None or not sampling.logit_bias:
        return
    vocabulary_size = get_unembedding(cast("PreTrainedModel", session.model)).shape[0]
    if any(token_id < 0 or token_id >= vocabulary_size for token_id in sampling.logit_bias):
        raise InvalidSamplingError(f"logit_bias token IDs must be between 0 and {vocabulary_size - 1}")


def flatten_content(content: Any) -> str:
    """Concatenate text content parts for text-only endpoints.

    Accepts a string (passed through), a list of content parts (each a
    ``{"type": "text", "text": ...}`` dict or a bare string — non-text
    parts raise :class:`UnsupportedContentError`), ``None`` (→ ``""``,
    the Ollama convention), or any other scalar (stringified).
    """
    if isinstance(content, list):
        pieces: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                pieces.append(str(part.get("text", "")))
            elif isinstance(part, str):
                pieces.append(part)
            else:
                raise UnsupportedContentError(
                    "non-text content parts are not supported by this model"
                )
        return "".join(pieces)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


def compose_request_steering(
    expr: str | None, default: Steering | None,
) -> Steering | None:
    """Compose a per-request steering expression over the server default.

    ``None`` inherits the default.  An explicit empty / whitespace string
    clears it and reaches the session as an *empty* :class:`Steering`, not
    ``None`` — ``None`` means "unset" there, and the cast roster fills unset
    steering with the gen label's standing recipe, so ``""`` is the escape
    hatch that must win.  A non-empty expression parses through the shared
    grammar and overrides the default at the key level: alphas named in both
    come from the request; alphas only in the default pass through.  Thinking
    is never carried here — it travels as the generation's ``thinking=``
    keyword.
    """
    from drowse.steering.grammar import parse_expr

    if expr is None:
        return default
    if not expr.strip():
        return Steering(alphas={})
    alphas: dict[str, Any] = dict(default.alphas) if default is not None else {}
    alphas.update(parse_expr(expr).alphas)
    return Steering(alphas=alphas)


def normalize_stop(
    stop: str | Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Lower a protocol's ``stop`` field to the engine's tuple shape.

    Every protocol accepts either a bare string or a sequence of strings
    (OpenAI ``stop``, Ollama ``options.stop``, the native WS ``sampling.stop``
    list); the engine takes a tuple.  ``None`` and an empty sequence both mean
    "no stop sequences".
    """
    if stop is None:
        return None
    if isinstance(stop, str):
        return (stop,)
    normalized = tuple(str(s) for s in stop)
    return normalized or None


def logprob_count(
    logprobs: bool | int | None, top_logprobs: int | None = None,
) -> int | None:
    """Collapse the two OpenAI logprobs shapes to the engine's int count.

    ``/v1/chat/completions`` sends a bool plus a separate ``top_logprobs``
    count; ``/v1/completions`` sends the count directly as an int.  The engine
    takes one int (``0`` = chosen token only, ``None`` = disabled).
    """
    if isinstance(logprobs, bool):
        return (top_logprobs or 0) if logprobs else None
    if isinstance(logprobs, int):
        return logprobs
    return None


def build_sampling_config(
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    stop: str | Sequence[str] | None = None,
    logit_bias: dict[int, float] | None = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    logprobs: bool | int | None = None,
    top_logprobs: int | None = None,
    return_top_k: int = 0,
    return_probe_readings: bool = True,
    persist_per_layer_scores: bool = False,
    persist_subspace_coords: bool = False,
    user_role: str | None = None,
    assistant_role: str | None = None,
) -> SamplingConfig:
    """Build a :class:`SamplingConfig` from raw per-protocol request fields.

    The single construction site for all three protocols (OpenAI, Ollama, the
    native WebSocket), so the normalizations they share — ``stop`` in either
    string or sequence form, the two OpenAI ``logprobs`` shapes, empty role
    labels meaning "unset" — happen once instead of drifting per surface.
    """
    return SamplingConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
        stop=normalize_stop(stop),
        logit_bias=logit_bias,
        presence_penalty=presence_penalty or 0.0,
        frequency_penalty=frequency_penalty or 0.0,
        logprobs=logprob_count(logprobs, top_logprobs),
        return_top_k=return_top_k,
        return_probe_readings=bool(return_probe_readings),
        persist_per_layer_scores=bool(persist_per_layer_scores),
        persist_subspace_coords=bool(persist_subspace_coords),
        user_role=user_role or None,
        assistant_role=assistant_role or None,
    )


def strict_model_enabled() -> bool:
    return os.environ.get("DROWSE_STRICT_MODEL", "").lower() in (
        "1", "true", "yes", "on",
    )
