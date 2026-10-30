"""Token-callback contracts: the public callback and the internal consumer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from drowse.generation.results import TokenAlt, TokenEvent


class TokenCallback(Protocol):
    """The public per-token callback users hand to ``generate``."""

    def __call__(self, text: str, is_thinking: bool, token_id: int | None,
                 logprob: float | None, top_alts: list[TokenAlt] | None,
                 perplexity: float | None = None, /) -> None: ...


class StepTokenCallback(Protocol):
    """The decode loop's internal tap contract — ``generate_steered``'s
    ``on_token``.  The trailing ``step_id`` is the loop-owned forward index
    (the same value that forward's ``step_callback`` / ``score_callback``
    received), which the engine's token tap uses to pair the instrument runs'
    step-keyed memos."""

    def __call__(self, text: str, is_thinking: bool, token_id: int | None,
                 logprob: float | None, top_alts: list[TokenAlt] | None,
                 perplexity: float | None, step_id: int, /) -> None: ...


@dataclass(frozen=True)
class TokenConsumerOptions:
    """What an internal consumer wants computed per token.

    ``live_scores`` / ``per_layer_scores`` ask for per-token monitor readings
    (and their per-layer rows); ``lens_readout`` / ``sae_readout`` for the live
    instrument readouts; ``perplexity`` for the sampler perplexity on stateless
    generations; ``probe_readings`` for the cross-family
    :attr:`TokenEvent.probe_readings` compat projection.
    """

    live_scores: bool = False
    per_layer_scores: bool = False
    lens_readout: bool = False
    sae_readout: bool = False
    perplexity: bool = False
    probe_readings: bool = False


@dataclass(frozen=True)
class TokenConsumer:
    """An internal token consumer: receives each token's full :class:`TokenEvent`.

    The event carries the token's measurement envelope and raw index, so the
    stream and the WebSocket read everything from it rather than from session
    state.
    """

    on_event: Callable[[TokenEvent], None]
    options: TokenConsumerOptions = TokenConsumerOptions()


def consumer_options(callback: object | None) -> TokenConsumerOptions:
    return callback.options if isinstance(callback, TokenConsumer) else TokenConsumerOptions()
