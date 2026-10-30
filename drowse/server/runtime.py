"""Server runtime primitives shared by every route module.

The HTTP serializer (one bounded ``asyncio.Lock`` on ``app.state``), the
worker-thread ownership helpers, the engine-busy probe, and the strict native
request base.  Route modules import from here and never from ``app``, so the
factory can import every router without a cycle.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

from anyio import CancelScope
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from starlette.responses import StreamingResponse
from starlette.types import Send

from drowse.errors import DrowseError

if TYPE_CHECKING:
    from drowse.session.session import DrowseSession

_T = TypeVar("_T")
_P = ParamSpec("_P")

#: Bound on waiting for the session lock; a request queued longer gets a 503.
SESSION_LOCK_TIMEOUT_SECONDS = 300
#: The only session id the native tree resolves (one model per server).
SINGLE_SESSION_ID = "default"


class NativeRequest(BaseModel):
    """Strict base for every native ``/drowse/v1`` request body."""

    model_config = ConfigDict(extra="forbid")


class SessionLockTimeout(RuntimeError, DrowseError):
    """The bounded session lock was not acquired in time."""

    def user_message(self) -> tuple[int, str]:
        return (503, "session locked — try again when the current operation finishes")


class EngineBusyError(RuntimeError, DrowseError):
    """The engine gen-lock is held by a worker that outlived its request."""

    def user_message(self) -> tuple[int, str]:
        return (409, "a model operation is in flight; retry shortly")


def resolve_session_id(session_id: str) -> None:
    """404 unless ``session_id`` names the single live session.

    Installed as a router-level dependency on the session-scoped native
    routers, so route bodies never repeat the check.
    """
    if session_id != SINGLE_SESSION_ID:
        raise HTTPException(404, f"session '{session_id}' not found")


def session_router() -> APIRouter:
    """A router under ``/drowse/v1/sessions/{session_id}`` that 404s other ids."""
    return APIRouter(
        prefix="/drowse/v1/sessions/{session_id}",
        dependencies=[Depends(resolve_session_id)],
    )


@asynccontextmanager
async def session_lock(app: FastAPI) -> AsyncIterator[None]:
    """Hold the server's session lock, bounded by the timeout.

    Every generation, model read, and native mutation serializes on this one
    FIFO lock across all three protocols.  A waiter past
    :data:`SESSION_LOCK_TIMEOUT_SECONDS` raises :class:`SessionLockTimeout`
    (503), which each protocol renders in its own envelope or frame.
    """
    lock: asyncio.Lock = app.state.session_lock
    try:
        async with asyncio.timeout(SESSION_LOCK_TIMEOUT_SECONDS):
            await lock.acquire()
    except TimeoutError as exc:
        raise SessionLockTimeout() from exc
    try:
        yield
    finally:
        lock.release()


def refuse_if_busy(session: "DrowseSession") -> None:
    """Raise :class:`EngineBusyError` (409) while the engine gen-lock is held.

    The session lock orders native mutations against each other, but an SSE
    fit or extract whose client went away keeps its worker thread — and the
    ``gen_lock`` it holds — alive until the worker exits.  A non-blocking
    probe refuses a mutation while that thread runs.
    """
    if not session.gen_lock.acquire(blocking=False):
        raise EngineBusyError()
    session.gen_lock.release()


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
    """Run blocking work in a thread that the awaiting request owns until it ends."""
    return await finish_worker(asyncio.create_task(asyncio.to_thread(func, *args, **kwargs)))


async def stream_events(iterator: Iterator[_T]) -> AsyncIterator[_T]:
    """Drain a blocking iterator one item per worker-thread hop."""
    sentinel = object()
    while True:
        event = await run_in_thread(next, iterator, sentinel)
        if event is sentinel:
            return
        yield cast(_T, event)


class ClosingStreamingResponse(StreamingResponse):
    """A streaming response that always closes its body iterator, even on send failure."""

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            close: Any = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                await close()
