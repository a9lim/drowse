"""Long-running native operations: one ledger, one runner, one error policy.

Three kinds of work outlive a single request/response exchange, and all of
them classify failures through :func:`~drowse.server.errors.describe_error`
so none surfaces a raw exception string:

- **Progress operations** (``POST /extract``, manifold ``install`` /
  ``generate`` / ``fit``, template ``score``) run through
  :func:`run_operation`.  The job runs in a server-owned task that holds the
  session lock for its real lifetime; the response is a view over it — an SSE
  stream on ``Accept: text/event-stream``, otherwise one awaited JSON body.
- **Receipts.**  An ``X-Drowse-Request-Id`` header (HTTP) or ``request_id``
  field (WebSocket generation) admits the operation into the bounded
  :class:`OperationLedger`, so a client that lost its connection can read the
  outcome back (``GET /operations/{id}`` / ``GET /sessions/{id}/requests/{id}``)
  instead of resubmitting.
- **Preparations** (lens ``fetch``/``fit``, SAE ``fetch``/``train``) are
  polled rather than streamed: a :class:`Preparation` keeps the running-or-last
  status of one operation, and a :class:`PreparationGroup` makes a family's
  preparations mutually exclusive.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict, deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request

from drowse.server.errors import ErrorInfo, describe_error
from drowse.server.runtime import ClosingStreamingResponse, finish_worker, session_lock

ProgressCallback = Callable[[str], None]
ProgressJob = Callable[[ProgressCallback], Awaitable[Any]]
#: Route-specific refinement of :func:`describe_error`; ``None`` defers to it.
ErrorClassifier = Callable[[Exception], ErrorInfo | None]
Admission = Literal["new", "existing", "conflict", "full"]

#: Idle gap after which an SSE stream emits a comment line, so an intermediary
#: read timeout (nginx defaults to 60 s) cannot drop a slow job.
HEARTBEAT_SECONDS = 15.0
MAX_PROGRESS_MESSAGES = 256
MAX_RECEIPT_PROGRESS = 64
MAX_PROGRESS_LENGTH = 1024

_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_TERMINAL = frozenset({"completed", "failed", "interrupted", "cancelled"})


# ---------------------------------------------------------------------------
# The receipt ledger
# ---------------------------------------------------------------------------

def _encoded_size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), default=str))


class OperationLedger:
    """Bounded, thread-safe receipts for HTTP operations and WS generations.

    One record per request id.  Admission is idempotent: resubmitting the same
    parameters under the same id reports ``existing``, different parameters
    ``conflict``.  At most ``max_active`` records may be unfinished; past
    ``max_entries`` the oldest finished record is evicted.  Retained payloads
    (HTTP results, WS ``done``/``error`` frames) share one byte budget; when it
    overflows, the oldest finished payloads are dropped and marked truncated
    while their receipts survive.
    """

    def __init__(
        self,
        *,
        max_entries: int = 256,
        max_active: int = 128,
        max_result_bytes: int = 4 * 1024 * 1024,
        max_total_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if not 1 <= max_active <= max_entries or max_result_bytes < 0 or max_total_bytes < 0:
            raise ValueError("operation ledger limits must satisfy 1 <= max_active <= max_entries")
        self.max_entries = max_entries
        self.max_active = max_active
        self.max_result_bytes = max_result_bytes
        self.max_total_bytes = max_total_bytes
        self._records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._retained: dict[str, int] = {}
        self._lock = threading.RLock()

    @staticmethod
    def validate_id(request_id: str) -> None:
        if not _REQUEST_ID.fullmatch(request_id):
            raise HTTPException(
                400,
                "Operation request ID must contain 1 to 128 letters, digits, "
                "dots, underscores, colons or hyphens",
            )

    def admit(
        self,
        request_id: str,
        parameters: Any,
        *,
        kind: Literal["http", "generate"],
        path: str | None = None,
        model_id: str | None = None,
    ) -> Admission:
        self.validate_id(request_id)
        try:
            fingerprint = hashlib.sha256(json.dumps(
                [kind, path, model_id, parameters], sort_keys=True,
                separators=(",", ":"), ensure_ascii=False, allow_nan=False,
            ).encode("utf-8")).hexdigest()
        except (TypeError, ValueError):
            raise HTTPException(400, "Operation parameters must be valid JSON") from None
        with self._lock:
            prior = self._records.get(request_id)
            if prior is not None:
                return "existing" if prior["fingerprint"] == fingerprint else "conflict"
            active = sum(record["state"] not in _TERMINAL for record in self._records.values())
            if active >= self.max_active:
                return "full"
            if len(self._records) >= self.max_entries:
                evict = next(key for key, record in self._records.items() if record["state"] in _TERMINAL)
                del self._records[evict]
                self._retained.pop(evict, None)
            now = time.time()
            self._records[request_id] = {
                "request_id": request_id, "kind": kind, "fingerprint": fingerprint,
                "path": path, "model_id": model_id,
                "state": "queued" if kind == "http" else "running",
                "created_at": now, "started_at": None, "updated_at": now, "finished_at": None,
                "progress": [], "progress_seq": 0,
                "result": None, "result_bytes": 0, "result_sha256": None, "result_truncated": False,
                "results": [], "error": None,
            }
            return "new"

    # -- transitions (no-ops once a record is terminal) ----------------------

    def _live(self, request_id: str) -> dict[str, Any] | None:
        record = self._records.get(request_id)
        return record if record is not None and record["state"] not in _TERMINAL else None

    def _touch(self, record: dict[str, Any], state: str | None = None) -> None:
        now = time.time()
        if state is not None:
            record["state"] = state
            if state == "running":
                record["started_at"] = record["started_at"] or now
            elif state in _TERMINAL:
                record["finished_at"] = now
        record["updated_at"] = now

    def running(self, request_id: str) -> None:
        with self._lock:
            if (record := self._live(request_id)) is not None:
                self._touch(record, "running")

    def progress(self, request_id: str, message: str) -> None:
        with self._lock:
            if (record := self._live(request_id)) is not None:
                record["progress"].append(message[:MAX_PROGRESS_LENGTH])
                del record["progress"][:-MAX_RECEIPT_PROGRESS]
                record["progress_seq"] += 1
                self._touch(record, "running")

    def complete(self, request_id: str, result: Any) -> None:
        encoded = json.dumps(
            result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        retained = len(encoded) <= min(self.max_result_bytes, self.max_total_bytes)
        with self._lock:
            if (record := self._live(request_id)) is None:
                return
            record.update(
                result=json.loads(encoded) if retained else None,
                result_bytes=len(encoded),
                result_sha256=hashlib.sha256(encoded).hexdigest(),
                result_truncated=not retained,
            )
            self._touch(record, "completed")
            if retained:
                self._retain(request_id, len(encoded))

    def fail(self, request_id: str, error: dict[str, Any]) -> None:
        self._finish_error(request_id, "failed", error)

    def interrupt(self, request_id: str, message: str) -> None:
        self._finish_error(request_id, "interrupted", {"code": "INTERRUPTED", "message": message})

    def _finish_error(self, request_id: str, state: str, error: dict[str, Any]) -> None:
        with self._lock:
            if (record := self._live(request_id)) is None:
                return
            stored: dict[str, Any] = {
                "code": str(error.get("code", "OPERATION_FAILED"))[:128],
                "message": str(error.get("message", "Operation failed"))[:2048],
            }
            status = error.get("status")
            if type(status) is int and 400 <= status <= 599:
                stored["status"] = status
            record["error"] = stored
            self._touch(record, state)

    def observe(self, request_id: str, frame: dict[str, Any]) -> None:
        """Record one WebSocket generation frame against its receipt."""
        with self._lock:
            record = self._records.get(request_id)
            if record is None:
                return
            if frame["type"] == "done":
                record["results"] = [
                    row for row in record["results"] if row["sibling_index"] != frame["sibling_index"]
                ]
                record["results"].append(copy.deepcopy(frame))
                self._retain(request_id, self._retained.get(request_id, 0) + _encoded_size(frame))
            elif frame["type"] == "error":
                record["error"] = copy.deepcopy(frame)
            elif frame["type"] == "request_complete" and record["state"] not in _TERMINAL:
                self._touch(record, frame["state"])

    def settle_disconnected(self, request_id: str, expected: int) -> None:
        """Settle a generation whose client left before ``request_complete``."""
        with self._lock:
            if (record := self._live(request_id)) is None:
                return
            rows = record["results"]
            if record["error"] is not None:
                state = "failed"
            elif len(rows) == expected and all(row["result"].get("finish_reason") != "cancelled" for row in rows):
                state = "completed"
            else:
                state = "cancelled"
            self._touch(record, state)

    def _retain(self, request_id: str, size: int) -> None:
        """Charge ``size`` bytes to ``request_id``; drop the oldest finished payloads past budget."""
        self._retained[request_id] = size
        total = sum(self._retained.values())
        for key in list(self._retained):
            if total <= self.max_total_bytes:
                return
            record = self._records[key]
            if key == request_id or record["state"] not in _TERMINAL:
                continue
            total -= self._retained.pop(key)
            record.update(result=None, results=[], result_truncated=True)

    # -- views ----------------------------------------------------------------

    def receipt(self, request_id: str) -> dict[str, Any]:
        """The HTTP receipt view (``GET /drowse/v1/operations/{id}``)."""
        self.validate_id(request_id)
        with self._lock:
            record = self._records.get(request_id)
            if record is None or record["kind"] != "http":
                raise HTTPException(
                    404, "Operation receipt is unavailable or has expired; completion cannot be inferred",
                )
            view = {key: value for key, value in record.items() if key not in {"kind", "fingerprint", "results"}}
            return copy.deepcopy({"id": request_id, **view, "request_fingerprint": record["fingerprint"]})

    def request_status(self, request_id: str) -> dict[str, Any]:
        """The WebSocket generation view (``GET /drowse/v1/sessions/{id}/requests/{id}``)."""
        with self._lock:
            record = self._records.get(request_id)
            if record is None or record["kind"] != "generate":
                return {"request_id": request_id, "state": "unknown", "results": []}
            view: dict[str, Any] = {
                "request_id": request_id, "state": record["state"], "results": record["results"],
            }
            if record["error"] is not None:
                view["error"] = record["error"]
            return copy.deepcopy(view)


# ---------------------------------------------------------------------------
# Progress operations (SSE or JSON over one server-owned task)
# ---------------------------------------------------------------------------

class ProgressPump:
    """Coalesce worker-thread progress lines onto the event loop.

    ``report`` is safe from any thread; bursts collapse into one loop
    callback.  The queue is bounded (oldest progress dropped first) and the
    terminal frame never waits on a consumer that went away.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, mirror: ProgressCallback | None = None) -> None:
        self._loop = loop
        self._mirror = mirror
        self.queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=MAX_PROGRESS_MESSAGES)
        self.history: deque[str] = deque(maxlen=MAX_PROGRESS_MESSAGES)
        self._pending: deque[str] = deque(maxlen=MAX_PROGRESS_MESSAGES)
        self._lock = threading.Lock()
        self._scheduled = False

    def report(self, message: str) -> None:
        if self._mirror is not None:
            self._mirror(message)
        with self._lock:
            self.history.append(message)
            self._pending.append(message)
            if self._scheduled:
                return
            self._scheduled = True
        self._loop.call_soon_threadsafe(self._flush)

    def _put(self, item: tuple[str, Any]) -> None:
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait(item)

    def _flush(self) -> None:
        with self._lock:
            batch = list(self._pending)
            self._pending.clear()
            self._scheduled = False
        for message in batch:
            self._put(("progress", {"message": message}))

    def finish(self, kind: str, payload: Any) -> None:
        self._flush()
        self._put((kind, payload))

    async def frames(self) -> AsyncGenerator[str, None]:
        """SSE frames until the terminal ``done``/``error``, with idle heartbeats."""
        while True:
            try:
                kind, payload = await asyncio.wait_for(self.queue.get(), timeout=HEARTBEAT_SECONDS)
            except TimeoutError:
                yield ": heartbeat\n\n"
                continue
            yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
            if kind in {"done", "error"}:
                return


@asynccontextmanager
async def _unbounded_lock(app: FastAPI) -> AsyncIterator[None]:
    async with app.state.session_lock:
        yield


class OperationTasks:
    """The server-owned operation tasks, each either queued or started.

    A task is *queued* until it holds the session lock and its job begins.
    Shutdown cancels the queued ones — nobody can still be waiting on an
    operation the server will never answer — and joins the started ones, so a
    worker is never abandoned mid-write.
    """

    def __init__(self) -> None:
        self._started: dict[asyncio.Task[Any], bool] = {}

    def add(self, task: asyncio.Task[Any], *, started: bool = True) -> None:
        self._started[task] = started
        task.add_done_callback(self.discard)

    def discard(self, task: asyncio.Task[Any]) -> None:
        self._started.pop(task, None)

    def start(self, task: asyncio.Task[Any] | None) -> None:
        if task is not None and task in self._started:
            self._started[task] = True

    def queued(self, task: asyncio.Task[Any]) -> bool:
        return self._started.get(task) is False

    def __iter__(self) -> Iterator[asyncio.Task[Any]]:
        return iter(list(self._started))

    def __len__(self) -> int:
        return len(self._started)

    async def join(self) -> None:
        """Cancel queued tasks, then wait out every task's real lifetime."""
        for task, started in list(self._started.items()):
            if not started:
                task.cancel()
        tasks = list(self._started)
        if tasks:
            await finish_worker(asyncio.gather(*tasks, return_exceptions=True))


async def _observe(
    pump: ProgressPump, tasks: OperationTasks, task: asyncio.Task[Any], *, abandonable: bool,
) -> AsyncGenerator[str, None]:
    """The SSE body over ``pump``; an abandoned stream withdraws a still-queued job.

    Without a receipt, the stream is the job's only observer: a client that
    leaves while the job waits for the lock cancels it, so the job never
    writes artifacts nobody asked to keep.  Once the job has started — or
    when a receipt can report the outcome — it runs to completion.
    """
    frames = pump.frames()
    try:
        async for frame in frames:
            yield frame
    finally:
        await frames.aclose()
        if abandonable and tasks.queued(task):
            task.cancel()


async def run_operation(
    request: Request,
    job: ProgressJob,
    *,
    failure: str,
    classify: ErrorClassifier | None = None,
    progress_key: str | None = None,
) -> Any:
    """Run ``job`` under the session lock and answer as SSE or JSON.

    SSE (``Accept: text/event-stream``) waits for the lock without a bound —
    the jobs are inherently long, and heartbeats keep the stream alive — and
    emits ``progress`` / ``done`` / ``error`` frames.  JSON waits at most the
    bounded lock timeout and returns the job's payload (with the collected
    progress lines under ``progress_key`` when set) or raises the classified
    failure.  Either way the job runs in a server-owned task, so a client
    disconnect never releases the lock while the worker still writes, and
    shutdown joins it.  An SSE client without a receipt that leaves before
    the job starts withdraws it (see :func:`_observe`).
    """
    app = request.app
    streaming = "text/event-stream" in request.headers.get("accept", "")
    ledger: OperationLedger = app.state.operations
    request_id = request.headers.get("x-drowse-request-id")
    if request_id:
        admission = ledger.admit(
            request_id, await request.json(), kind="http",
            path=request.url.path, model_id=app.state.session.model_id,
        )
        if admission == "full":
            raise HTTPException(503, "The operation ledger is full of active requests; wait for one to finish")
        if admission != "new":
            raise HTTPException(409, (
                "This request ID already has an operation receipt; read its status instead of resubmitting"
                if admission == "existing"
                else "This request ID belongs to an operation with different parameters"
            ))

    def mirror(message: str) -> None:
        if request_id:
            ledger.progress(request_id, message)

    pump = ProgressPump(asyncio.get_running_loop(), mirror)
    tasks: OperationTasks = app.state.operation_tasks

    async def run() -> tuple[bool, Any]:
        try:
            async with (_unbounded_lock(app) if streaming else session_lock(app)):
                tasks.start(asyncio.current_task())
                if request_id:
                    ledger.running(request_id)
                payload = await job(pump.report)
        except asyncio.CancelledError:
            if request_id:
                ledger.interrupt(request_id, "The server stopped before the operation finished.")
            raise
        except Exception as exc:  # noqa: BLE001 - classified once, reported as a frame or status
            info = (classify(exc) if classify is not None else None) or describe_error(exc, failure=failure)
            if request_id:
                ledger.fail(request_id, info.frame())
            pump.finish("error", info.frame())
            return False, info
        if progress_key is not None and not streaming and isinstance(payload, dict):
            payload[progress_key] = list(pump.history)
        if request_id:
            ledger.complete(request_id, payload)
        pump.finish("done", payload)
        return True, payload

    task = asyncio.create_task(run())
    tasks.add(task, started=False)
    if streaming:
        return ClosingStreamingResponse(
            _observe(pump, tasks, task, abandonable=not request_id), media_type="text/event-stream",
        )
    ok, outcome = await finish_worker(task)
    if not ok:
        raise HTTPException(outcome.status, outcome.message)
    return outcome


# ---------------------------------------------------------------------------
# Polled preparations
# ---------------------------------------------------------------------------

class Preparation:
    """The running-or-last status of one polled background operation.

    ``label`` names the operation in user-facing text (``"lens fit"``).
    ``fields`` names the operation's extra status keys (``live_layers``,
    ``release``, …) — they persist across runs unless a start overwrites
    them.  With ``progress_unit`` and ``progress_pattern`` set, a progress
    line matching the two-integer pattern updates ``{current, total}``.
    """

    def __init__(
        self,
        operation: str,
        *,
        label: str,
        fields: tuple[str, ...],
        cancel_exc: type[BaseException] | None = None,
        progress_unit: str | None = None,
        progress_pattern: re.Pattern[str] | None = None,
    ) -> None:
        self.operation = operation
        self.label = label
        self.fields = fields
        self.cancel_exc = cancel_exc
        self.progress_unit = progress_unit
        self.progress_pattern = progress_pattern
        self.state: dict[str, Any] = {
            "running": False, "message": None, "error": None,
            "started_at": None, "finished_at": None, "current": 0, "total": 0,
            **dict.fromkeys(fields),
        }
        self.task: asyncio.Task[None] | None = None
        self.cancel_event: threading.Event | None = None

    @property
    def cancellable(self) -> bool:
        return self.cancel_exc is not None

    @property
    def running(self) -> bool:
        return bool(self.state["running"])

    def report(self, message: str) -> None:
        """Progress hook for the worker thread: parse counts, mirror the line."""
        if self.progress_pattern is not None:
            match = self.progress_pattern.search(message)
            if match is not None:
                self.state["current"] = int(match.group(1).replace(",", ""))
                self.state["total"] = int(match.group(2).replace(",", ""))
        self.state["message"] = message

    def start(self, body: Callable[[], Awaitable[None]], *, message: str, **fields: Any) -> None:
        """Mark running, stamp ``fields``, and launch ``body`` as a task.

        Any exception the body raises settles through :func:`describe_error`:
        a cooperative cancel to ``cancelled``, anything else to its user-safe
        message — never a raw exception string.
        """
        self.state.update(
            running=True, error=None, started_at=time.time(), finished_at=None,
            message=message, **fields,
        )
        self.cancel_event = threading.Event() if self.cancellable else None

        async def runner() -> None:
            try:
                await finish_worker(asyncio.ensure_future(body()))
            except Exception as exc:  # noqa: BLE001 - settled into the polled status
                if self.cancel_exc is not None and isinstance(exc, self.cancel_exc):
                    self.state.update(message="cancelled", error=None)
                else:
                    failure = f"{self.label} failed"
                    info = describe_error(exc, failure=failure)
                    self.state.update(message=failure, error=info.message)
            finally:
                self.state.update(running=False, finished_at=time.time())

        self.task = asyncio.create_task(runner())

    def request_cancel(self) -> None:
        if not self.running or self.cancel_event is None:
            raise HTTPException(409, f"no {self.label} is running")
        self.cancel_event.set()
        self.state["message"] = "cancelling…"

    async def stop(self) -> None:
        """Signal cooperative cancellation and join the job's real lifetime."""
        if self.cancel_event is not None:
            self.cancel_event.set()
        if self.task is not None and not self.task.done():
            await finish_worker(self.task)

    def wire(self) -> dict[str, Any]:
        st = self.state
        if st["running"]:
            state = "running"
        elif st["error"]:
            state = "error"
        elif st["finished_at"]:
            state = "done"
        else:
            state = "idle"
        progress = (
            {"current": st["current"], "total": st["total"], "unit": self.progress_unit}
            if self.progress_unit is not None else None
        )
        return {
            "state": state,
            "operation": self.operation,
            "progress": progress,
            "message": st["message"],
            "error": st["error"],
            "started_at": st["started_at"],
            "finished_at": st["finished_at"],
            "cancellable": self.cancellable,
            **{key: st[key] for key in self.fields},
        }


IDLE_PREPARATION: dict[str, Any] = {
    "state": "idle", "operation": None, "progress": None, "message": None,
    "error": None, "started_at": None, "finished_at": None, "cancellable": False,
}


class PreparationGroup:
    """A family's mutually exclusive preparations (one may run at a time)."""

    def __init__(self, busy_message: str, *preparations: Preparation) -> None:
        self.busy_message = busy_message
        self.members = {prep.operation: prep for prep in preparations}

    def __iter__(self) -> Iterator[Preparation]:
        return iter(self.members.values())

    def __getitem__(self, operation: str) -> Preparation:
        return self.members[operation]

    def refuse_if_busy(self) -> None:
        if any(prep.running for prep in self):
            raise HTTPException(409, self.busy_message)

    def status(self) -> dict[str, Any]:
        """The running preparation's status, else the most recently active one."""
        for prep in self:
            if prep.running:
                return prep.wire()
        started = [prep for prep in self if prep.state["started_at"]]
        if not started:
            return dict(IDLE_PREPARATION)
        latest = max(started, key=lambda prep: prep.state["finished_at"] or prep.state["started_at"])
        return latest.wire()

    async def stop(self) -> None:
        for prep in self:
            await prep.stop()


def register_operation_routes(app: FastAPI) -> None:
    """Install the ledger, task set, and receipt route; settle live tasks at shutdown."""
    app.state.operations = OperationLedger()
    app.state.operation_tasks = OperationTasks()

    @app.get("/drowse/v1/operations/{request_id}")
    def operation_status(request_id: str) -> dict[str, Any]:
        return app.state.operations.receipt(request_id)

    async def finish_operations() -> None:
        await app.state.operation_tasks.join()

    app.router.on_shutdown.append(finish_operations)
