"""Shared streaming-finalization plumbing for the three streaming protocols.

The SSE (OpenAI), NDJSON (Ollama), and WebSocket co-stream paths each close a
generation by deriving the same three things — the finish reason, the token
``usage`` rollup, and the per-attached-probe aggregate reading — and then format
them into protocol-specific wire frames.  This module owns the *derivation* so
the three sites can't drift; each protocol keeps its own framing (raw vs mapped
finish reason, usage chunk vs Ollama duration stats, the ``x-drowse-probe-readings``
extension vs the native ``probe_readings`` block).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

from anyio import CancelScope
from starlette.responses import StreamingResponse
from starlette.types import Send

_T = TypeVar("_T")
_P = ParamSpec("_P")


async def finish_worker(task: asyncio.Future[_T]) -> _T:
    """Join a worker without letting request cancellation abandon its thread."""
    cancelled = False
    with CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return task.result()


async def run_in_thread(func: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    return await finish_worker(asyncio.create_task(asyncio.to_thread(func, *args, **kwargs)))


async def stream_events(iterator: Iterator[_T]) -> AsyncIterator[_T]:
    sentinel = object()
    while True:
        event = await run_in_thread(next, iterator, sentinel)
        if event is sentinel:
            return
        yield cast(_T, event)


class ClosingStreamingResponse(StreamingResponse):
    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                await close()


if TYPE_CHECKING:
    from drowse.core.results import GenerationResult
    from drowse.core.session import DrowseSession


def probe_reading_aggregate(
    session: "DrowseSession", result: "GenerationResult | None",
) -> dict[str, Any]:
    """Per-attached-probe ``ProbeReading.to_dict()`` from ``result``.

    Returns ``{}`` when no result is recorded or no manifold probes are
    attached.  Surfaced under the ``x-drowse-probe-readings`` extension on the
    OpenAI and Ollama responses so vector-probe clients keep working unchanged
    and manifold-aware clients pick up the geometric channel — that vendor
    extension is the only consumer of this flat shape (the native WS ``done``
    frame carries the 5.x envelope alone, see
    :func:`probe_measurements_aggregate`).  Result-parameterized (rather than
    reading ``session.last_result``) so per-sibling done frames score each
    sibling's own result.
    """
    if result is None:
        return {}
    readings = result.probe_readings or {}
    if not readings:
        return {}
    attached = set(session.monitor.probe_names)
    # Pinned J-lens token probes and SAE feature probes live on their own
    # session registries (readout channels, not Monitor probes) but land in
    # ``result.probe_readings`` all the same — without this union the
    # attached-filter silently dropped their end-of-gen aggregates from every
    # streaming done frame.
    attached.update(session.lens.names)
    attached.update(session.sae.names)
    return {
        name: reading.to_dict()
        for name, reading in readings.items()
        if name in attached
    }


def usage_dict(result: "GenerationResult") -> dict[str, int]:
    pt = result.prompt_tokens
    ct = result.token_count
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


def stream_finalizer(
    session: "DrowseSession", result: "GenerationResult | None",
) -> tuple[str | None, dict[str, int] | None, dict[str, Any]]:
    """Derive the shared end-of-stream triple ``(finish_reason, usage, probe_agg)``.

    ``finish_reason`` comes off ``result`` (the engine stamps it there from
    ``GenerationState.finish_reason`` at result-build time, so it equals the
    live gen-state value the non-streaming sites read), falling back to the live
    gen state when no result is recorded.  ``usage`` is the OpenAI-shaped token
    rollup (``None`` when there is no result; Ollama derives its duration stats
    separately).  ``probe_agg`` is :func:`probe_reading_aggregate`.  Each
    protocol formats these into its own wire shape.
    """
    if result is not None:
        finish_reason = result.finish_reason
        usage: dict[str, int] | None = usage_dict(result)
    else:
        finish_reason = session.generation_state.finish_reason
        usage = None
    probe_agg = probe_reading_aggregate(session, result)
    return finish_reason, usage, probe_agg
