"""The OpenAI ``/v1/*`` and Ollama ``/api/*`` compatibility protocols.

Both protocols lower a request onto one :class:`CompatGeneration` and run it
through one driver — :func:`run_generation` (blocking) or
:func:`stream_generation` (an event iterator) — then format the outcome into
their own wire shapes.  The driver owns the session lock, the worker thread,
client-disconnect detection, deterministic stream teardown, and error
classification, so the two protocol modules hold only request models and
formatters.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request

from drowse.generation.results import GenerationResult, TokenEvent
from drowse.generation.sampling import SamplingConfig
from drowse.server.compat.model_names import aliases_for_session, known_model_names
from drowse.server.errors import ErrorInfo, describe_error
from drowse.server.request_helpers import strict_model_enabled
from drowse.server.runtime import run_in_thread, session_lock, stream_events
from drowse.steering.terms import Steering

if TYPE_CHECKING:
    from drowse.session.session import DrowseSession


@dataclass(frozen=True)
class CompatGeneration:
    """One compatibility-protocol generation, lowered and validated."""

    input: str | list[dict[str, str]]
    raw: bool
    sampling: SamplingConfig
    steering: Steering | None
    thinking: bool | None

    def kwargs(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "sampling": self.sampling,
            "steering": self.steering,
            "thinking": self.thinking,
            "stateless": True,
        }


@dataclass(frozen=True)
class StreamStarted:
    """The lock is held and the generation is about to start."""


@dataclass(frozen=True)
class StreamEnd:
    """The stream finished; ``result`` is ``None`` only if the engine recorded none."""

    result: GenerationResult | None
    finish_reason: str | None
    elapsed_ns: int


StreamItem = StreamStarted | TokenEvent | StreamEnd | ErrorInfo


def check_strict_model(session: "DrowseSession", name: str | None) -> None:
    """404 a ``model`` mismatch when ``DROWSE_STRICT_MODEL`` is on."""
    if not strict_model_enabled() or not name:
        return
    if name.lower() not in known_model_names(session):
        hosted = ", ".join(sorted({session.model_id, *aliases_for_session(session)}))
        raise HTTPException(
            404,
            f"model '{name}' not found. drowse hosts: {hosted}. "
            "Unset DROWSE_STRICT_MODEL to accept any model name.",
        )


def probe_reading_aggregate(
    session: "DrowseSession", result: GenerationResult | None,
) -> dict[str, Any]:
    """Attached probes' end-of-generation readings, keyed by probe name.

    The ``x-drowse-probe-readings`` extension both protocols carry.  Pinned
    J-lens and SAE probes live on their own registries but land in
    ``result.probe_readings`` alongside Monitor probes, so the attached set
    unions all three.  Empty when nothing is attached.
    """
    readings = result.probe_readings if result is not None else None
    if not readings:
        return {}
    attached = {*session.monitor.probe_names, *session.lens.names, *session.sae.names}
    return {name: reading.to_dict() for name, reading in readings.items() if name in attached}


def usage_dict(result: GenerationResult) -> dict[str, int]:
    prompt, completion = result.prompt_tokens, result.token_count
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


async def run_generation(app: FastAPI, gen: CompatGeneration) -> tuple[GenerationResult, int]:
    """Generate to completion under the session lock; returns ``(result, elapsed_ns)``.

    Failures propagate to the app's exception handlers, which render them in
    the request's protocol envelope.
    """
    session = app.state.session
    async with session_lock(app):
        start = time.monotonic_ns()
        result = (await run_in_thread(session.generate, gen.input, **gen.kwargs())).first
    return result, time.monotonic_ns() - start


def stream_generation(
    app: FastAPI, gen: CompatGeneration, request: Request,
) -> "aclosing[AsyncGenerator[StreamItem, None]]":
    """``async with stream_generation(...) as events: async for item in events``.

    Wrapped in :func:`contextlib.aclosing` so a protocol formatter that is
    itself closed early (client gone, send failure) closes the driver at once
    — joining the engine worker and releasing the lock before the response
    finishes — instead of leaving it to async-generator finalization.
    """
    return aclosing(_stream_generation(app, gen, request))


async def _stream_generation(
    app: FastAPI, gen: CompatGeneration, request: Request,
) -> AsyncGenerator[StreamItem, None]:
    """Stream one generation as driver events.

    Yields :class:`StreamStarted` once the lock is held, one
    :class:`TokenEvent` per decoded token, then :class:`StreamEnd`.  Any
    failure — including a lock timeout — arrives as one final
    :class:`ErrorInfo` instead of an exception, because headers are already
    on the wire and the protocol must report it in-band.  The iterator stops
    silently once the client disconnects, and the engine worker is always
    joined before the lock is released.
    """
    session = app.state.session
    try:
        async with session_lock(app):
            yield StreamStarted()
            start = time.monotonic_ns()
            stream_iter = session.generate_stream(
                gen.input, live_scores=False, live_readouts=False, **gen.kwargs(),
            )
            try:
                async for event in stream_events(stream_iter):
                    if await request.is_disconnected():
                        return
                    yield event
            finally:
                await run_in_thread(stream_iter.close)
            result = stream_iter.result
            finish = result.finish_reason if result is not None else session.generation_state.finish_reason
            yield StreamEnd(result, finish, time.monotonic_ns() - start)
    except Exception as exc:  # noqa: BLE001 - reported in-band, classified once
        yield describe_error(exc, failure="Generation failed")
