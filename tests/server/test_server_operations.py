"""The native operation machinery: receipt ledger, progress runner, preparations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

import drowse.server.operations as operations
from drowse.errors import DrowseError, PoisednessError, SourceSpecError
from drowse.server.errors import install_error_handlers
from drowse.server.operations import (
    OperationLedger,
    Preparation,
    PreparationGroup,
    ProgressCallback,
    register_operation_routes,
    run_operation,
)

PATH = "/drowse/v1/manifolds/local/pirate/fit"


def _http(ledger: OperationLedger, request_id: str, body: Any = None, *, path: str = PATH, model: str | None = "model") -> str:
    return ledger.admit(request_id, body if body is not None else {}, kind="http", path=path, model_id=model)


# ---------------------------------------------------------------------------
# OperationLedger — HTTP receipts
# ---------------------------------------------------------------------------

def test_receipt_lifecycle_is_copied_and_keeps_no_request_body() -> None:
    ledger = OperationLedger()
    assert _http(ledger, "fit:1", {"seed": 3, "private_input": "not retained"}) == "new"
    receipt = ledger.receipt("fit:1")
    assert receipt["state"] == "queued" and receipt["request_id"] == "fit:1"
    assert "not retained" not in json.dumps(receipt)
    receipt["progress"].append("external mutation")
    assert ledger.receipt("fit:1")["progress"] == []
    ledger.running("fit:1")
    assert ledger.receipt("fit:1")["started_at"] is not None
    ledger.progress("fit:1", "fitting layers")
    value = {"namespace": "local", "name": "pirate", "scores": [0.3]}
    ledger.complete("fit:1", value)
    value["scores"].append(9)
    completed = ledger.receipt("fit:1")
    assert completed["state"] == "completed"
    assert completed["progress"] == ["fitting layers"]
    assert completed["result"]["scores"] == [0.3]
    assert completed["finished_at"] >= completed["started_at"] >= completed["created_at"]
    assert completed["result_sha256"] == hashlib.sha256(json.dumps(
        completed["result"], sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    ledger.progress("fit:1", "late")
    ledger.fail("fit:1", {"message": "late failure"})
    ledger.complete("fit:1", {"different": True})
    assert ledger.receipt("fit:1") == completed


def test_duplicate_and_conflicting_admissions_never_reexecute() -> None:
    ledger = OperationLedger()
    assert _http(ledger, "same", {"b": 2, "a": {"x": 1}}) == "new"
    assert _http(ledger, "same", {"a": {"x": 1}, "b": 2}) == "existing"
    assert _http(ledger, "same", {"b": 3}) == "conflict"
    assert _http(ledger, "same", {"a": {"x": 1}, "b": 2}, path="/drowse/v1/other") == "conflict"
    assert _http(ledger, "same", {"a": {"x": 1}, "b": 2}, model="other-model") == "conflict"
    ledger.complete("same", {})
    assert _http(ledger, "same", {"a": {"x": 1}, "b": 2}) == "existing"


def test_simultaneous_admission_has_one_owner() -> None:
    ledger = OperationLedger()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _http(ledger, "single-owner"), range(24)))
    assert results.count("new") == 1
    assert results.count("existing") == 23


def test_active_records_are_never_evicted() -> None:
    ledger = OperationLedger(max_entries=2, max_active=1)
    _http(ledger, "active")
    assert _http(ledger, "blocked") == "full"
    ledger.complete("active", {"ok": True})
    assert _http(ledger, "second") == "new"
    ledger.complete("second", {"ok": True})
    assert _http(ledger, "third") == "new"
    with pytest.raises(HTTPException, match="completion cannot be inferred") as expired:
        ledger.receipt("active")
    assert expired.value.status_code == 404
    assert ledger.receipt("third")["state"] == "queued"


def test_progress_failure_and_interruption_are_bounded() -> None:
    ledger = OperationLedger()
    _http(ledger, "progress")
    for index in range(100):
        ledger.progress("progress", f"{index}:" + "x" * 2000)
    receipt = ledger.receipt("progress")
    assert len(receipt["progress"]) == 64
    assert receipt["progress_seq"] == 100
    assert all(len(message) <= 1024 for message in receipt["progress"])
    assert receipt["progress"][0].startswith("36:")
    ledger.fail("progress", {"message": "safe failure", "code": "SAFE", "status": 409, "credential": "secret"})
    failed = ledger.receipt("progress")
    assert failed["state"] == "failed"
    assert failed["error"] == {"message": "safe failure", "code": "SAFE", "status": 409}
    _http(ledger, "interrupted")
    ledger.interrupt("interrupted", "Worker stopped")
    assert ledger.receipt("interrupted")["state"] == "interrupted"


def test_large_results_keep_their_digest_under_one_byte_budget() -> None:
    ledger = OperationLedger(max_result_bytes=100, max_total_bytes=150)
    for request_id in ["first", "second", "large"]:
        _http(ledger, request_id)
    ledger.complete("first", {"text": "a" * 70})
    first = ledger.receipt("first")
    assert not first["result_truncated"]
    ledger.complete("second", {"text": "b" * 70})
    evicted = ledger.receipt("first")
    assert evicted["state"] == "completed"
    assert evicted["result"] is None and evicted["result_truncated"]
    assert evicted["result_sha256"] == first["result_sha256"]
    ledger.complete("large", {"text": "c" * 1000})
    large = ledger.receipt("large")
    assert large["result"] is None and large["result_truncated"]
    assert large["result_bytes"] > 100 and len(large["result_sha256"]) == 64


@pytest.mark.parametrize("request_id", ["", "x" * 129, "a/b", "a b", "a\n", "a?key=secret"])
def test_invalid_ids_are_rejected(request_id: str) -> None:
    with pytest.raises(HTTPException) as error:
        _http(OperationLedger(), request_id)
    assert error.value.status_code == 400


# ---------------------------------------------------------------------------
# OperationLedger — WebSocket generation receipts
# ---------------------------------------------------------------------------

def test_generation_receipts_record_frames_and_settle() -> None:
    ledger = OperationLedger()
    assert ledger.request_status("missing") == {"request_id": "missing", "state": "unknown", "results": []}
    assert ledger.admit("gen", {"prompt": "hi"}, kind="generate") == "new"
    assert ledger.request_status("gen")["state"] == "running"
    ledger.observe("gen", {"type": "done", "sibling_index": 0, "result": {"text": "a", "finish_reason": "stop"}})
    ledger.observe("gen", {"type": "done", "sibling_index": 0, "result": {"text": "b", "finish_reason": "stop"}})
    ledger.settle_disconnected("gen", expected=2)
    status = ledger.request_status("gen")
    assert status["state"] == "cancelled"
    assert [row["result"]["text"] for row in status["results"]] == ["b"]
    snapshot = ledger.request_status("gen")
    snapshot["results"][0]["result"]["text"] = "changed"
    assert ledger.request_status("gen")["results"][0]["result"]["text"] == "b"

    ledger.admit("done", {}, kind="generate")
    ledger.observe("done", {"type": "request_complete", "state": "completed"})
    assert ledger.request_status("done")["state"] == "completed"


def test_receipt_kinds_do_not_alias() -> None:
    ledger = OperationLedger()
    _http(ledger, "shared")
    assert ledger.request_status("shared")["state"] == "unknown"
    ledger.admit("ws", {}, kind="generate")
    with pytest.raises(HTTPException):
        ledger.receipt("ws")


def test_generation_payload_budget_drops_oldest_finished_frames_only() -> None:
    ledger = OperationLedger(max_total_bytes=2500)
    for request_id in ("0", "1", "2"):
        ledger.admit(request_id, {}, kind="generate")
        ledger.observe(request_id, {"type": "done", "sibling_index": 0, "result": {"text": "x" * 1000}})
        ledger.observe(request_id, {"type": "request_complete", "state": "completed"})
    assert ledger.request_status("0")["results"] == []
    assert ledger.request_status("0")["state"] == "completed"
    assert len(ledger.request_status("2")["results"]) == 1


# ---------------------------------------------------------------------------
# run_operation — SSE / JSON over one server-owned task
# ---------------------------------------------------------------------------

def _app() -> FastAPI:
    app = FastAPI()
    app.state.session = SimpleNamespace(model_id="model/test")
    app.state.session_lock = asyncio.Lock()
    install_error_handlers(app)
    register_operation_routes(app)
    return app


def _client(
    job: Callable[[ProgressCallback], Awaitable[Any]], **kwargs: Any,
) -> TestClient:
    app = _app()

    @app.post("/drowse/v1/run")
    async def run_route(request: Request):
        return await run_operation(request, job, failure="Job failed", **kwargs)

    return TestClient(app)


def _sse(client: TestClient) -> str:
    with client.stream("POST", "/drowse/v1/run", headers={"accept": "text/event-stream"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        return b"".join(resp.iter_bytes()).decode("utf-8")


def test_json_by_default_with_optional_progress_lines() -> None:
    async def job(progress: ProgressCallback) -> dict[str, Any]:
        progress("first")
        progress("second")
        return {"ok": True}

    assert _client(job).post("/drowse/v1/run").json() == {"ok": True}
    assert _client(job, progress_key="progress").post("/drowse/v1/run").json() == {
        "ok": True, "progress": ["first", "second"],
    }


def test_sse_streams_progress_then_done() -> None:
    async def job(progress: ProgressCallback) -> dict[str, Any]:
        progress("first")
        progress("second")
        return {"ok": True}

    text = _sse(_client(job))
    assert text.count("event: progress") == 2
    assert '"message": "first"' in text
    assert 'event: done\ndata: {"ok": true}' in text


class _Conflict(RuntimeError, DrowseError):
    def user_message(self) -> tuple[int, str]:
        return 409, "already running"


@pytest.mark.parametrize(("error", "status", "text"), [
    (_Conflict(), 409, "already running"),
    (PoisednessError("nodes are not poised"), 400, "nodes are not poised"),
    (HTTPException(404, "no such folder"), 404, "no such folder"),
    (FileNotFoundError("/secret/cache/path"), 404, "Requested artifact not found"),
    (RuntimeError("/secret/cache/path"), 500, "Job failed. Check the server log for details."),
    (ValueError("/secret/cache/path"), 500, "Job failed. Check the server log for details."),
])
def test_one_error_policy_for_json_and_sse(error: Exception, status: int, text: str) -> None:
    async def job(_progress: ProgressCallback) -> dict[str, Any]:
        raise error

    client = _client(job)
    resp = client.post("/drowse/v1/run")
    assert resp.status_code == status
    assert resp.json() == {"detail": text}
    frames = _sse(client)
    assert "event: error" in frames
    payload = json.loads(frames.split("data: ", 1)[1])
    assert payload["status"] == status and payload["message"] == text
    assert "/secret" not in frames


def test_frame_code_is_the_typed_class_name() -> None:
    """The client keys on the class name; no route-specific classifier."""
    async def job(_progress: ProgressCallback) -> dict[str, Any]:
        raise PoisednessError("nodes are not poised")

    assert '"code": "PoisednessError"' in _sse(_client(job))


def test_sse_heartbeats_while_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(operations, "HEARTBEAT_SECONDS", 0.02)

    async def job(_progress: ProgressCallback) -> dict[str, Any]:
        await asyncio.sleep(0.12)
        return {"ok": True}

    text = _sse(_client(job))
    assert ": heartbeat\n\n" in text
    assert text.count("event:") == 1


def _request(app: FastAPI, *, sse: bool, request_id: str | None = None) -> Request:
    headers = [(b"accept", b"text/event-stream")] if sse else []
    if request_id is not None:
        headers.append((b"x-drowse-request-id", request_id.encode()))

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b'{"name":"example"}', "more_body": False}

    return Request({
        "type": "http", "method": "POST", "path": "/drowse/v1/manifolds/generate",
        "app": app, "headers": headers, "asgi": {"spec_version": "2.4"},
    }, receive)


def test_cancelled_json_request_keeps_the_lock_until_the_worker_finishes() -> None:
    async def scenario() -> None:
        app = _app()
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def worker() -> dict[str, bool]:
            entered.set()
            release.wait(5)
            finished.set()
            return {"written": True}

        async def job(_progress: Any) -> dict[str, bool]:
            return await asyncio.to_thread(worker)

        task = asyncio.create_task(run_operation(_request(app, sse=False), job, failure="failed"))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.sleep(0.02)
            assert app.state.session_lock.locked()
            assert not task.done()
        finally:
            release.set()
            with suppress(asyncio.CancelledError):
                await task
            assert await asyncio.to_thread(finished.wait, 2)
    asyncio.run(scenario())


def test_retained_operation_survives_transport_disconnect_without_reexecution() -> None:
    async def scenario() -> None:
        app = _app()
        release, entered = asyncio.Event(), asyncio.Event()
        calls = 0

        async def job(progress: ProgressCallback) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            progress("working")
            entered.set()
            await release.wait()
            return {"completed": True, "artifact": "local/example"}

        initial = _request(app, sse=True, request_id="owned-job")
        response = await run_operation(initial, job, failure="failed")
        await entered.wait()
        assert app.state.session_lock.locked()
        with pytest.raises(HTTPException) as duplicate:
            await run_operation(_request(app, sse=True, request_id="owned-job"), job, failure="failed")
        assert duplicate.value.status_code == 409

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.body":
                raise OSError("Browser transport disconnected")

        with pytest.raises(ClientDisconnect):
            await response(initial.scope, initial.receive, send)
        assert app.state.session_lock.locked(), "the worker owns the lock after transport teardown"
        assert app.state.operations.receipt("owned-job")["state"] == "running"
        (task,) = app.state.operation_tasks
        release.set()
        await task
        receipt = app.state.operations.receipt("owned-job")
        assert receipt["state"] == "completed"
        assert receipt["result"] == {"completed": True, "artifact": "local/example"}
        assert receipt["progress"] == ["working"]
        assert not app.state.session_lock.locked()
        assert calls == 1
    asyncio.run(scenario())


def test_retained_json_receipt_keeps_result_and_scrubs_untyped_errors() -> None:
    calls = 0

    async def job(_progress: ProgressCallback) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("secret /private/model/path")
        return {"contexts": [{"value": "Mon", "prob_sum": 0.2}]}

    with _client(job) as client:
        result = client.post("/drowse/v1/run", json={}, headers={"X-Drowse-Request-Id": "score-receipt"})
        assert result.status_code == 200
        receipt = client.get("/drowse/v1/operations/score-receipt").json()
        assert receipt["state"] == "completed" and receipt["result"] == result.json()
        assert client.post("/drowse/v1/run", json={}, headers={"X-Drowse-Request-Id": "score-receipt"}).status_code == 409
        failure = client.post("/drowse/v1/run", json={"fail": True}, headers={"X-Drowse-Request-Id": "failed-receipt"})
        assert failure.status_code == 500
        failed = client.get("/drowse/v1/operations/failed-receipt").json()
        assert failed["state"] == "failed"
        assert failed["error"]["message"] == "Job failed. Check the server log for details."
        assert "secret" not in str(failed)
        assert calls == 2


@pytest.mark.parametrize("sse", [False, True])
def test_progress_flood_keeps_bounded_recent_history_and_final_result(sse: bool) -> None:
    async def scenario() -> None:
        app = _app()

        async def job(progress: Any) -> Any:
            for index in range(10_000):
                progress(str(index))
            return {"written": True}

        response = await run_operation(_request(app, sse=sse), job, failure="failed", progress_key="progress")
        if sse:
            frames = [frame async for frame in response.body_iterator]
            assert len(frames) <= 257
            assert '"9999"' in frames[-2]
            assert frames[-1] == 'event: done\ndata: {"written": true}\n\n'
        else:
            assert len(response["progress"]) <= 256
            assert response["progress"][-1] == "9999"
            assert response["written"] is True
    asyncio.run(scenario())


def test_progress_flood_after_disconnect_does_not_block_completion() -> None:
    async def scenario() -> None:
        app = _app()
        release = asyncio.Event()

        async def job(progress: Any) -> Any:
            progress("started")
            await release.wait()
            for index in range(10_000):
                progress(str(index))
            return {"written": True}

        response = await run_operation(_request(app, sse=True), job, failure="failed")
        iterator: Any = response.body_iterator
        assert "started" in await anext(iterator)
        await iterator.aclose()
        release.set()
        (task,) = app.state.operation_tasks
        await asyncio.wait_for(task, 2)
        assert not app.state.session_lock.locked()
    asyncio.run(scenario())


@pytest.mark.parametrize("request_id", [None, "kept"])
def test_sse_client_leaving_while_queued_withdraws_only_an_unreceipted_job(request_id: str | None) -> None:
    """Regression: the job task started at request time and waited for the
    lock unbounded, so an extract/fit whose client had already left still ran
    in full once the lock freed — with nobody observing and no receipt."""
    async def scenario() -> None:
        app = _app()
        calls = 0

        async def job(_progress: Any) -> dict[str, bool]:
            nonlocal calls
            calls += 1
            return {"written": True}

        await app.state.session_lock.acquire()
        response = await run_operation(_request(app, sse=True, request_id=request_id), job, failure="failed")
        (task,) = app.state.operation_tasks
        await asyncio.sleep(0.01)
        assert app.state.operation_tasks.queued(task)
        # The client leaves while the stream waits for its first frame:
        # Starlette's disconnect listener cancels the body iteration.
        body: Any = response.body_iterator
        waiting = asyncio.ensure_future(anext(body))
        await asyncio.sleep(0.01)
        waiting.cancel()
        with suppress(asyncio.CancelledError):
            await waiting
        await body.aclose()
        app.state.session_lock.release()
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        if request_id is None:
            assert task.cancelled() and calls == 0
        else:
            assert calls == 1
            assert app.state.operations.receipt(request_id)["state"] == "completed"
        assert not app.state.session_lock.locked()
    asyncio.run(scenario())


def test_shutdown_cancels_queued_operations_and_joins_started_ones() -> None:
    async def scenario() -> None:
        app = _app()
        entered, release = asyncio.Event(), asyncio.Event()
        calls: list[str] = []

        async def first(_progress: Any) -> dict[str, bool]:
            calls.append("first")
            entered.set()
            await release.wait()
            return {"first": True}

        async def second(_progress: Any) -> dict[str, bool]:
            calls.append("second")
            return {"second": True}

        await run_operation(_request(app, sse=True, request_id="started"), first, failure="failed")
        await entered.wait()
        await run_operation(_request(app, sse=True, request_id="queued"), second, failure="failed")
        shutdown = asyncio.gather(*(hook() for hook in app.router.on_shutdown))
        await asyncio.sleep(0.01)
        assert not shutdown.done(), "a started job is joined, not abandoned"
        release.set()
        await asyncio.wait_for(shutdown, 2)
        assert calls == ["first"]
        assert app.state.operations.receipt("started")["state"] == "completed"
        assert app.state.operations.receipt("queued")["state"] == "interrupted"
    asyncio.run(scenario())


def test_receipt_route_and_shutdown_join() -> None:
    async def scenario() -> None:
        app = _app()
        _http(app.state.operations, "owned", model="model/test")
        finished = False

        async def worker() -> None:
            nonlocal finished
            await asyncio.sleep(0.01)
            finished = True

        task = asyncio.create_task(worker())
        app.state.operation_tasks.add(task)
        for shutdown in app.router.on_shutdown:
            await shutdown()
        assert finished and not task.cancelled()

    asyncio.run(scenario())
    app = _app()
    _http(app.state.operations, "owned")
    with TestClient(app) as client:
        assert client.get("/drowse/v1/operations/owned").json()["request_id"] == "owned"
        assert client.get("/drowse/v1/operations/missing").status_code == 404


# ---------------------------------------------------------------------------
# Preparations
# ---------------------------------------------------------------------------

def test_preparation_failures_never_surface_raw_exception_text() -> None:
    """Regression: a fetch job used to write ``str(exc)`` — a filesystem path —
    into its polled status."""
    async def scenario() -> None:
        prep = Preparation("fetch", label="J-lens fetch", fields=("source",))
        for error, expected in (
            (FileNotFoundError("/home/user/.cache/secret/lens.safetensors"), "Requested artifact not found"),
            (RuntimeError("/home/user/private"), "J-lens fetch failed. Check the server log for details."),
            (SourceSpecError("unknown source 'x'"), "unknown source 'x'"),
            (ValueError("/home/user/private"), "J-lens fetch failed. Check the server log for details."),
        ):
            async def body(error: Exception = error) -> None:
                raise error

            prep.start(body, message="starting…", source="neuronpedia")
            assert prep.task is not None
            await prep.task
            status = prep.wire()
            assert status["state"] == "error"
            assert status["error"] == expected
            assert status["message"] == "J-lens fetch failed"
            assert "/home/user" not in json.dumps(status)
    asyncio.run(scenario())


def test_preparation_progress_cancel_and_group_exclusion() -> None:
    class Cancelled(Exception):
        pass

    async def scenario() -> None:
        fetch = Preparation("fetch", label="SAE fetch", fields=("release",))
        train = Preparation(
            "train", label="SAE training", fields=("name",), cancel_exc=Cancelled,
            progress_unit="tokens", progress_pattern=re.compile(r"trained ([\d,]+)/([\d,]+) tokens"),
        )
        group = PreparationGroup("busy", fetch, train)
        assert group.status()["state"] == "idle"
        with pytest.raises(HTTPException) as idle:
            train.request_cancel()
        assert idle.value.status_code == 409 and "no SAE training is running" in idle.value.detail
        release = asyncio.Event()

        async def body() -> None:
            train.report("trained 1,024/4,096 tokens")
            await release.wait()
            assert train.cancel_event is not None and train.cancel_event.is_set()
            raise Cancelled()

        train.start(body, message="starting…", name="x", current=0, total=4096)
        async with asyncio.timeout(2):
            while train.state["current"] != 1024:
                await asyncio.sleep(0.005)
        with pytest.raises(HTTPException) as busy:
            group.refuse_if_busy()
        assert busy.value.status_code == 409
        status = group.status()
        assert status["operation"] == "train" and status["cancellable"]
        assert status["progress"] == {"current": 1024, "total": 4096, "unit": "tokens"}
        train.request_cancel()
        release.set()
        assert train.task is not None
        await train.task
        settled = group.status()
        assert settled["state"] == "done" and settled["message"] == "cancelled" and settled["error"] is None
    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_task", [False, True])
def test_preparation_task_cancellation_keeps_writer_ownership(cancel_task: bool) -> None:
    async def scenario() -> None:
        entered, release = threading.Event(), threading.Event()
        lock = asyncio.Lock()
        prep = Preparation("fetch", label="fetch", fields=())
        group = PreparationGroup("busy", prep)

        def worker() -> None:
            entered.set()
            assert release.wait(3)

        async def body() -> None:
            async with lock:
                await asyncio.to_thread(worker)

        prep.start(body, message="fetching")
        assert prep.task is not None
        stopper = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            if cancel_task:
                prep.task.cancel()
                await asyncio.sleep(0)
                prep.task.cancel()
            else:
                stopper = asyncio.create_task(group.stop())
            await asyncio.sleep(0.02)
            assert prep.running and lock.locked()
            with pytest.raises(HTTPException):
                group.refuse_if_busy()
        finally:
            release.set()
            with suppress(asyncio.CancelledError):
                await prep.task
            if stopper is not None:
                await stopper
        assert not prep.running and not lock.locked()
    asyncio.run(scenario())
