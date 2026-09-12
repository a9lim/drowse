"""Shared Server-Sent Events helpers for native long-running routes."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from drowse.server.streaming import ClosingStreamingResponse, finish_worker

ProgressCallback = Callable[[str], None]
ProgressJob = Callable[[ProgressCallback], Awaitable[Any]]
ErrorFormatter = Callable[[Exception], dict[str, Any] | None]
#: Ordered ``(exception type(s), status)`` pairs for the JSON branch, applied
#: by ``isinstance`` — first match wins, so put subclasses first.
JsonErrorMap = Sequence[tuple[type[BaseException] | tuple[type[BaseException], ...], int]]

#: Idle gap after which the SSE worker emits a comment line.  Matches the
#: traits stream: intermediary proxies drop a connection that has been silent
#: for their read timeout (nginx defaults to 60 s), and a single manifold
#: generate / fit progress step can run well past that on MPS.
HEARTBEAT_SECONDS = 15.0
MAX_PROGRESS_MESSAGES = 256


async def _retained_response(
    request: Request, session: Any, job: ProgressJob, request_id: str, *,
    error_message: str, log_message: str, error_formatter: ErrorFormatter | None,
    json_errors: JsonErrorMap, json_progress_key: str | None,
    logger: logging.Logger | None,
) -> Any:
    from drowse.server.app import acquire_session_lock

    ledger = request.app.state.operation_ledger
    ledger.start(request_id, request.url.path, await request.json(), getattr(session, "model_id", None))
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=MAX_PROGRESS_MESSAGES)
    messages: deque[str] = deque(maxlen=MAX_PROGRESS_MESSAGES)
    pending: deque[str] = deque(maxlen=MAX_PROGRESS_MESSAGES)
    pending_lock = threading.Lock()
    scheduled = False
    streaming = "text/event-stream" in request.headers.get("accept", "")

    def flush() -> None:
        nonlocal scheduled
        with pending_lock:
            batch = list(pending)
            pending.clear()
            scheduled = False
        for message in batch:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(("progress", {"message": message}))

    def progress(message: str) -> None:
        nonlocal scheduled
        ledger.progress(request_id, message)
        with pending_lock:
            messages.append(message)
            pending.append(message)
            if scheduled:
                return
            scheduled = True
        loop.call_soon_threadsafe(flush)

    async def run() -> tuple[bool, Any]:
        try:
            if streaming:
                async with session.lock:
                    ledger.running(request_id)
                    payload = await job(progress)
            else:
                async with acquire_session_lock(session) as acquired:
                    if not acquired:
                        raise HTTPException(503, "session locked")
                    ledger.running(request_id)
                    payload = await job(progress)
            if not streaming and json_progress_key is not None and isinstance(payload, dict):
                payload[json_progress_key] = list(messages)
            ledger.complete(request_id, payload)
            terminal = ("done", payload)
        except asyncio.CancelledError:
            ledger.interrupt(request_id, "The server stopped observing this operation before completion was recorded.")
            raise
        except Exception as error:
            formatted = None
            if error_formatter is not None:
                try:
                    formatted = error_formatter(error)
                except Exception:
                    (logger or logging.getLogger("drowse.api")).exception("%s (error formatter crashed)", log_message)
            if isinstance(error, HTTPException):
                formatted = {"message": str(error.detail), "code": "HTTPException", "status": error.status_code}
            elif formatted is None:
                for types, status in json_errors:
                    if isinstance(error, types):
                        formatted = {"message": str(error), "code": type(error).__name__, "status": status}
                        break
            if formatted is not None and "status" not in formatted:
                for types, status in json_errors:
                    if isinstance(error, types):
                        formatted["status"] = status
                        break
            if formatted is None:
                (logger or logging.getLogger("drowse.api")).exception(log_message)
                formatted = {"message": error_message, "code": type(error).__name__, "status": 500}
            ledger.fail(request_id, formatted)
            terminal = ("error", formatted)
        flush()
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(terminal)
        return terminal[0] == "done", terminal[1]

    task = asyncio.create_task(run())
    tasks = request.app.state.operation_tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    if not streaming:
        ok, payload = await finish_worker(task)
        if not ok:
            raise HTTPException(payload.get("status", 400), payload["message"])
        return payload

    async def observe():
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except (TimeoutError, asyncio.TimeoutError):
                yield ": heartbeat\n\n"
                continue
            yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
            if kind in {"done", "error"}:
                return

    return ClosingStreamingResponse(observe(), media_type="text/event-stream")


def progress_sse_response(
    lock: Any,
    job: ProgressJob,
    *,
    error_message: str,
    log_message: str,
    error_formatter: ErrorFormatter | None = None,
    logger: logging.Logger | None = None,
) -> StreamingResponse:
    """Run a progress-reporting async job and stream progress/done/error frames.

    Acquires ``lock`` *without* a timeout — unlike the bounded
    ``acquire_session_lock`` used by request handlers — because the jobs this
    drives (manifold ``fit`` / ``generate``, ``extract``) are inherently
    unbounded.  Use it only for those long-running operations; a bounded
    variant would need a timeout parameter.

    A ``: heartbeat`` comment line goes out every
    :data:`HEARTBEAT_SECONDS` of idle so an intermediary's read timeout can't
    silently drop a client mid-job.
    """
    log = logger or logging.getLogger("drowse.api")

    async def _sse():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=MAX_PROGRESS_MESSAGES)
        pending: deque[str] = deque(maxlen=MAX_PROGRESS_MESSAGES)
        pending_lock = threading.Lock()
        scheduled = False

        def _flush_progress() -> None:
            nonlocal scheduled
            with pending_lock:
                messages = list(pending)
                pending.clear()
                scheduled = False
            for message in messages:
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(("progress", message))

        def _on_progress(msg: str) -> None:
            nonlocal scheduled
            with pending_lock:
                pending.append(msg)
                if scheduled:
                    return
                scheduled = True
            loop.call_soon_threadsafe(_flush_progress)

        def _complete(kind: str, payload: Any) -> None:
            _flush_progress()
            if queue.full():
                queue.get_nowait()
            queue.put_nowait((kind, payload))

        async with lock:
            async def _run() -> None:
                try:
                    payload = await job(_on_progress)
                    _complete("done", payload)
                except Exception as e:
                    err = None
                    if error_formatter is not None:
                        try:
                            err = error_formatter(e)
                        except Exception:
                            log.exception("%s (error formatter crashed)", log_message)
                    if err is None:
                        log.exception(log_message)
                        err = {
                            "message": error_message,
                            "code": type(e).__name__,
                        }
                    _complete("error", err)

            task = asyncio.create_task(_run())
            try:
                while True:
                    try:
                        kind, payload = await asyncio.wait_for(
                            queue.get(), timeout=HEARTBEAT_SECONDS,
                        )
                    except (TimeoutError, asyncio.TimeoutError):
                        # Comment line: not an event, so a spec-compliant
                        # client ignores it while the connection stays warm.
                        yield ": heartbeat\n\n"
                        continue
                    data = {"message": payload} if kind == "progress" else payload
                    yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
                    if kind in {"done", "error"}:
                        break
            finally:
                if not task.done():
                    # Keep the lock owned by the real worker lifetime.  Many
                    # callers run blocking model/artifact work via
                    # ``asyncio.to_thread``; cancelling this wrapper only
                    # cancels the await, not the thread.  Await completion so a
                    # disconnected SSE client cannot release ``session.lock``
                    # while the underlying job is still mutating session state
                    # or writing artifacts.
                    await finish_worker(task)

    return ClosingStreamingResponse(_sse(), media_type="text/event-stream")


async def sse_or_json(
    request: Request,
    session: Any,
    job: ProgressJob,
    *,
    error_message: str,
    log_message: str,
    error_formatter: ErrorFormatter | None = None,
    json_errors: JsonErrorMap = (),
    json_progress_key: str | None = None,
    logger: logging.Logger | None = None,
) -> Any:
    """Run ``job`` as an SSE stream or a plain JSON response.

    The three long-running native routes (``POST /extract``, ``POST
    /manifolds/generate``, ``POST .../fit``) negotiate the same way: SSE on
    ``Accept: text/event-stream``, otherwise a single JSON body produced under
    the bounded session lock.  This owns both branches so the two can't drift
    — the typed error handling is now an explicit argument on each side
    (``error_formatter`` for the SSE ``error`` frame, ``json_errors`` for the
    JSON status mapping) rather than whatever each route happened to catch.

    ``job`` is the same awaitable-taking-``on_progress`` callable
    :func:`progress_sse_response` drives, so any post-processing a route does
    around its worker (registering an extracted profile, re-serializing the
    folder) runs identically on both branches.  ``json_progress_key``, when
    set, attaches the collected progress lines to a dict payload under that
    key — the SSE branch already streams them as ``progress`` frames.
    """
    from drowse.server.app import acquire_session_lock

    request_id = request.headers.get("x-drowse-request-id")
    if request_id:
        return await _retained_response(
            request, session, job, request_id,
            error_message=error_message, log_message=log_message,
            error_formatter=error_formatter, json_errors=json_errors,
            json_progress_key=json_progress_key, logger=logger,
        )

    accept = request.headers.get("accept", "application/json")
    if "text/event-stream" in accept:
        return progress_sse_response(
            session.lock,
            job,
            error_message=error_message,
            log_message=log_message,
            error_formatter=error_formatter,
            logger=logger,
        )

    progress: deque[str] = deque(maxlen=MAX_PROGRESS_MESSAGES)
    async with acquire_session_lock(session) as acquired:
        if not acquired:
            raise HTTPException(503, "session locked")
        try:
            payload = await finish_worker(asyncio.ensure_future(job(progress.append)))
        except HTTPException:
            # Already carries its own status — a job that mapped its own
            # failure wins over the generic table.
            raise
        except Exception as exc:
            for types, status in json_errors:
                if isinstance(exc, types):
                    raise HTTPException(status, str(exc)) from exc
            raise
    if json_progress_key is not None and isinstance(payload, dict):
        payload[json_progress_key] = list(progress)
    return payload
