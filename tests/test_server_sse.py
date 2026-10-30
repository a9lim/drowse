"""Tests for shared native SSE helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

import drowse.server.sse as sse_module
from drowse.server.sse import ProgressCallback, progress_sse_response, sse_or_json
from drowse.server.operation_ledger import register_operation_routes


def _client_for(
    job: Callable[[ProgressCallback], Awaitable[Any]],
    *,
    error_formatter: Callable[[Exception], dict[str, Any] | None] | None = None,
) -> TestClient:
    app = FastAPI()

    @app.get("/sse")
    async def sse_route():
        return progress_sse_response(
            asyncio.Lock(),
            job,
            error_message="job failed",
            log_message="test job failed",
            error_formatter=error_formatter,
        )

    return TestClient(app)


def _body(client: TestClient) -> str:
    with client.stream("GET", "/sse") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        return b"".join(resp.iter_bytes()).decode("utf-8")


def test_progress_sse_streams_progress_then_done() -> None:
    async def job(on_progress: ProgressCallback) -> dict[str, Any]:
        on_progress("first")
        on_progress("second")
        return {"ok": True}

    text = _body(_client_for(job))
    assert text.count("event: progress") == 2
    assert "first" in text
    assert "second" in text
    assert "event: done" in text
    assert '"ok": true' in text


def test_progress_sse_generic_error_is_scrubbed() -> None:
    async def job(_on_progress: ProgressCallback) -> dict[str, Any]:
        raise RuntimeError("/secret/cache/path")

    text = _body(_client_for(job))
    assert "event: error" in text
    assert '"message": "job failed"' in text
    assert '"code": "RuntimeError"' in text
    assert "/secret/cache/path" not in text


def test_progress_sse_typed_error_formatter_can_expose_safe_message() -> None:
    async def job(_on_progress: ProgressCallback) -> dict[str, Any]:
        raise ValueError("safe author-facing message")

    def format_error(e: Exception) -> dict[str, Any] | None:
        if isinstance(e, ValueError):
            return {"message": str(e), "code": "ValueError"}
        return None

    text = _body(_client_for(job, error_formatter=format_error))
    assert "event: error" in text
    assert "safe author-facing message" in text
    assert '"code": "ValueError"' in text


def test_progress_sse_formatter_crash_falls_back_to_generic_error() -> None:
    async def job(_on_progress: ProgressCallback) -> dict[str, Any]:
        raise RuntimeError("/private/path")

    def bad_formatter(_e: Exception) -> dict[str, Any] | None:
        raise AssertionError("formatter bug")

    text = _body(_client_for(job, error_formatter=bad_formatter))
    assert "event: error" in text
    assert '"message": "job failed"' in text
    assert '"code": "RuntimeError"' in text
    assert "/private/path" not in text
    assert "formatter bug" not in text


def test_progress_sse_emits_heartbeat_while_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An idle job keeps the connection warm with SSE comment lines.

    A manifold generate / fit progress step can run past an intermediary's
    default read timeout (nginx: 60 s), which drops the client while the job
    keeps running server-side — the traits stream already heartbeats for
    exactly this reason.
    """
    monkeypatch.setattr(sse_module, "HEARTBEAT_SECONDS", 0.02)

    async def job(_on_progress: ProgressCallback) -> dict[str, Any]:
        await asyncio.sleep(0.12)
        return {"ok": True}

    text = _body(_client_for(job))
    assert ": heartbeat\n\n" in text
    # Heartbeats are comments, not events — the real frames still arrive.
    assert "event: done" in text
    assert text.count("event:") == 1


# ---------------------------------------------------------------------------
# sse_or_json — the shared SSE-vs-JSON negotiation
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()


def _negotiating_client(
    job: Callable[[ProgressCallback], Awaitable[Any]],
    **kwargs: Any,
) -> TestClient:
    app = FastAPI()

    @app.post("/run")
    async def run_route(request: Request):
        return await sse_or_json(
            request,
            _FakeSession(),
            job,
            error_message="job failed",
            log_message="test job failed",
            **kwargs,
        )

    return TestClient(app)


class TestSseOrJson:
    def test_json_branch_by_default(self) -> None:
        async def job(_on_progress: ProgressCallback) -> dict[str, Any]:
            return {"ok": True}

        resp = _negotiating_client(job).post("/run")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_sse_branch_on_accept_header(self) -> None:
        async def job(on_progress: ProgressCallback) -> dict[str, Any]:
            on_progress("step")
            return {"ok": True}

        client = _negotiating_client(job)
        with client.stream(
            "POST", "/run", headers={"accept": "text/event-stream"},
        ) as resp:
            assert resp.headers["content-type"].startswith("text/event-stream")
            text = b"".join(resp.iter_bytes()).decode("utf-8")
        assert "event: progress" in text
        assert "event: done" in text

    def test_json_branch_collects_progress_when_asked(self) -> None:
        """The JSON branch can return what the SSE branch streamed."""
        async def job(on_progress: ProgressCallback) -> dict[str, Any]:
            on_progress("first")
            on_progress("second")
            return {"ok": True}

        resp = _negotiating_client(job, json_progress_key="progress").post("/run")
        assert resp.json() == {"ok": True, "progress": ["first", "second"]}

    def test_json_branch_maps_typed_errors_in_order(self) -> None:
        class Conflict(RuntimeError):
            pass

        async def job(_on_progress: ProgressCallback) -> dict[str, Any]:
            raise Conflict("already running")

        client = _negotiating_client(
            job, json_errors=((Conflict, 409), (ValueError, 400)),
        )
        resp = client.post("/run")
        assert resp.status_code == 409
        assert "already running" in resp.text

    def test_json_branch_maps_value_error(self) -> None:
        async def job(_on_progress: ProgressCallback) -> dict[str, Any]:
            raise ValueError("nodes are not poised")

        client = _negotiating_client(job, json_errors=((ValueError, 400),))
        resp = client.post("/run")
        assert resp.status_code == 400
        assert "poised" in resp.text

    def test_json_branch_passes_through_an_http_exception(self) -> None:
        """A job that already mapped its own failure keeps that status."""
        async def job(_on_progress: ProgressCallback) -> dict[str, Any]:
            raise HTTPException(404, "no such folder")

        client = _negotiating_client(job, json_errors=((ValueError, 400),))
        resp = client.post("/run")
        assert resp.status_code == 404

    def test_json_branch_leaves_unmapped_errors_alone(self) -> None:
        async def job(_on_progress: ProgressCallback) -> dict[str, Any]:
            raise KeyError("unmapped")

        client = _negotiating_client(job, json_errors=((ValueError, 400),))
        with pytest.raises(KeyError):
            client.post("/run")

    def test_sse_branch_uses_the_error_formatter(self) -> None:
        async def job(_on_progress: ProgressCallback) -> dict[str, Any]:
            raise ValueError("nodes are not poised")

        def format_error(e: Exception) -> dict[str, Any] | None:
            if isinstance(e, ValueError):
                return {"message": str(e), "code": "PoisednessError"}
            return None

        client = _negotiating_client(job, error_formatter=format_error)
        with client.stream(
            "POST", "/run", headers={"accept": "text/event-stream"},
        ) as resp:
            text = b"".join(resp.iter_bytes()).decode("utf-8")
        assert "event: error" in text
        assert '"code": "PoisednessError"' in text


def test_retained_operation_survives_transport_disconnect_without_reexecution() -> None:
    async def exercise() -> None:
        app = FastAPI()
        register_operation_routes(app)
        session = _FakeSession()
        release = asyncio.Event()
        entered = asyncio.Event()
        calls = 0

        async def job(progress: ProgressCallback) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            progress("working")
            entered.set()
            await release.wait()
            return {"completed": True, "artifact": "local/example"}

        def request() -> Request:
            async def receive() -> dict[str, Any]:
                return {"type": "http.request", "body": b'{"name":"example"}', "more_body": False}
            return Request({
                "type": "http", "method": "POST", "path": "/drowse/v1/manifolds/generate",
                "app": app, "headers": [(b"accept", b"text/event-stream"), (b"x-drowse-request-id", b"owned-job")],
                "asgi": {"spec_version": "2.4"},
            }, receive)

        initial = request()
        response = await sse_or_json(initial, session, job, error_message="failed", log_message="test")
        await entered.wait()
        assert session.lock.locked()
        with pytest.raises(HTTPException) as duplicate:
            await sse_or_json(request(), session, job, error_message="failed", log_message="test")
        assert duplicate.value.status_code == 409

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.body":
                raise OSError("Browser transport disconnected")

        with pytest.raises(ClientDisconnect):
            await response(initial.scope, initial.receive, send)
        assert session.lock.locked(), "the actual worker owns the lock after transport teardown"
        assert app.state.operation_ledger.get("owned-job")["state"] == "running"
        assert len(app.state.operation_tasks) == 1
        task = next(iter(app.state.operation_tasks))
        release.set()
        await task
        receipt = app.state.operation_ledger.get("owned-job")
        assert receipt["state"] == "completed"
        assert receipt["result"] == {"completed": True, "artifact": "local/example"}
        assert receipt["progress"] == ["working"]
        assert not session.lock.locked()
        assert calls == 1

    asyncio.run(exercise())


def test_retained_json_receipt_keeps_result_and_scrubs_untyped_errors() -> None:
    app = FastAPI()
    register_operation_routes(app)
    session = _FakeSession()
    calls = 0

    @app.post("/drowse/v1/test/score")
    async def score(request: Request) -> Any:
        async def job(_progress: ProgressCallback) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if (await request.json()).get("fail"):
                raise RuntimeError("secret /private/model/path")
            return {"contexts": [{"value": "Mon", "prob_sum": 0.2}]}
        return await sse_or_json(request, session, job, error_message="scoring failed", log_message="test scoring failed")

    with TestClient(app) as client:
        result = client.post("/drowse/v1/test/score", json={}, headers={"X-Drowse-Request-Id": "score-receipt"})
        assert result.status_code == 200
        receipt = client.get("/drowse/v1/operations/score-receipt").json()
        assert receipt["state"] == "completed"
        assert receipt["result"] == result.json()
        assert client.post("/drowse/v1/test/score", json={}, headers={"X-Drowse-Request-Id": "score-receipt"}).status_code == 409
        failure = client.post("/drowse/v1/test/score", json={"fail": True}, headers={"X-Drowse-Request-Id": "failed-receipt"})
        assert failure.status_code == 500
        failed = client.get("/drowse/v1/operations/failed-receipt").json()
        assert failed["state"] == "failed"
        assert failed["error"]["message"] == "scoring failed"
        assert "secret" not in str(failed)
        assert calls == 2
