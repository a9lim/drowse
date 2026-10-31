"""Drive a generation's token consumer the way the engine's token tap does."""

from __future__ import annotations

from typing import Any

from drowse.generation.results import TokenEvent
from drowse.generation.token_callback import TokenConsumer

__all__ = ["emit_token"]


def emit_token(
    on_token: Any,
    text: str,
    token_id: int,
    *,
    index: int = 0,
    thinking: bool = False,
    logprob: float | None = None,
    top_alts: Any = None,
    perplexity: float | None = None,
    measurements: dict[str, Any] | None = None,
    raw_index: int | None = None,
) -> None:
    """Hand one token to ``on_token`` in the shape the real tap uses.

    An internal :class:`TokenConsumer` (the WebSocket, the stream) receives a
    :class:`TokenEvent`; a plain callback gets the public six arguments.
    """
    if isinstance(on_token, TokenConsumer):
        on_token.on_event(TokenEvent(
            text=text, token_id=token_id, index=index, thinking=thinking,
            logprob=logprob, top_alts=top_alts, perplexity=perplexity,
            measurements=measurements, raw_index=raw_index,
        ))
    else:
        on_token(text, thinking, token_id, logprob, top_alts, perplexity)
