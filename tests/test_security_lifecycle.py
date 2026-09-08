"""Adversarial regressions for native server access and worker lifetimes."""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator, cast
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from starlette.types import Message

from drowse.core.generation import GenerationConfig, GenerationState, generate_steered
from drowse.core.results import GenerationResult
from drowse.core.session import DrowseSession
from drowse.server import create_app
from drowse.server.sse import progress_sse_response
from tests.test_generation import _CurrentSessionStub, _StopTokenizer, _scripted_model
from tests.test_server_loom import _StubSession

STREAM_PATH = "/drowse/v1/sessions/default/stream"


@asynccontextmanager
async def _socket(session: _StubSession, send_gate: asyncio.Event | None = None) -> AsyncIterator[tuple[Any, Any, asyncio.Task[Any]]]:
    incoming: asyncio.Queue[Message] = asyncio.Queue()
    outgoing: asyncio.Queue[Message] = asyncio.Queue()
    app = create_app(cast(DrowseSession, session))
    scope = {
        "type": "websocket", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "scheme": "ws", "path": STREAM_PATH, "raw_path": STREAM_PATH.encode(),
        "root_path": "", "query_string": b"", "headers": [(b"host", b"localhost")],
        "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 80),
        "subprotocols": [], "state": {},
    }
    incoming.put_nowait({"type": "websocket.connect"})

    async def send(message: Message) -> None:
        if send_gate is not None and message["type"] == "websocket.send":
            await send_gate.wait()
        await outgoing.put(message)

    task = asyncio.create_task(app(scope, incoming.get, send))
    assert (await asyncio.wait_for(outgoing.get(), 2))["type"] == "websocket.accept"
    try:
        yield incoming, outgoing, task
    finally:
        incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})
        with suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)


def test_foreign_origin_cannot_subscribe_or_mutate() -> None:
    session = _StubSession()
    with TestClient(create_app(cast(DrowseSession, session))) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(STREAM_PATH, headers={"Origin": "https://untrusted.example"}):
                pass
    assert session.tree.rev == 0


def test_websocket_flood_is_rejected_while_session_locked() -> None:
    async def scenario() -> None:
        session = _StubSession()
        await session.lock.acquire()
        try:
            async with _socket(session) as (incoming, outgoing, _task):
                try:
                    for _ in range(128):
                        incoming.put_nowait({"type": "websocket.receive", "text": json.dumps({
                            "type": "submit", "text": "x" * 4096,
                            "authored_role": "user",
                        })})
                    async with asyncio.timeout(1):
                        while True:
                            frame = await outgoing.get()
                            if frame["type"] == "websocket.close":
                                assert frame["code"] in (1008, 1009, 1013)
                                break
                finally:
                    session.lock.release()
        finally:
            if session.lock.locked():
                session.lock.release()
        assert session.tree.rev == 0
    asyncio.run(scenario())


def test_stream_close_before_first_read_waits_for_worker_and_survives_reset() -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    closed = threading.Event()
    generated: list[int] = []
    errors: list[BaseException] = []
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._gen_state = GenerationState()
    session._monitor = SimpleNamespace(probe_names=[])
    session._last_token_probe_payload = {}

    def worker(*_args: Any, **kwargs: Any) -> GenerationResult:
        import torch
        entered.set()
        assert release.wait(3)
        session._gen_state.reset()
        try:
            generated.extend(generate_steered(
                _scripted_model([0] * 20), cast(Any, _StopTokenizer()),
                torch.tensor([[0]]), GenerationConfig(max_new_tokens=20, temperature=0),
                session._gen_state,
            ))
            return GenerationResult(text="", tokens=generated, token_count=len(generated),
                                    tok_per_sec=0, elapsed=0)
        finally:
            finished.set()

    session._generate_core = worker
    stream = session.generate_stream("prompt")
    assert entered.wait(2)

    def close_stream() -> None:
        try:
            stream.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            closed.set()

    closer = threading.Thread(target=close_stream)
    closer.start()
    try:
        returned_before_worker = closed.wait(0.1)
    finally:
        release.set()
        closer.join(3)
        assert finished.wait(3)
    assert not errors
    assert closed.is_set()
    assert not returned_before_worker
    assert generated == []
    stream.close()


def test_sse_disconnect_keeps_lock_until_real_worker_exits() -> None:
    async def scenario() -> None:
        lock = asyncio.Lock()
        started = asyncio.Event()
        disconnect = asyncio.Event()
        release = threading.Event()
        finished = threading.Event()
        loop = asyncio.get_running_loop()

        def worker() -> dict[str, bool]:
            loop.call_soon_threadsafe(started.set)
            try:
                assert release.wait(3)
                return {"ok": True}
            finally:
                finished.set()

        async def job(_progress: Any) -> dict[str, bool]:
            return await asyncio.to_thread(worker)

        async def receive() -> dict[str, str]:
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(_message: Any) -> None:
            pass

        response = progress_sse_response(lock, job, error_message="failed", log_message="failed")
        task = asyncio.create_task(response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send))
        await asyncio.wait_for(started.wait(), 2)
        disconnect.set()
        acquired = asyncio.create_task(lock.acquire())
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(acquired), 0.1)
            assert not finished.is_set()
        finally:
            release.set()
            await asyncio.wait_for(task, 2)
            await asyncio.wait_for(acquired, 2)
            lock.release()
        assert finished.is_set()
    asyncio.run(scenario())


def test_idle_authenticated_socket_disconnect_does_not_stop_other_client() -> None:
    session = _StubSession()
    session._block_until_stop = True
    app = create_app(cast(DrowseSession, session), api_key="test-secret")
    with TestClient(app) as client:
        with client.websocket_connect(STREAM_PATH + "?token=test-secret") as active:
            active.send_json({"type": "generate", "input": "hello"})
            while active.receive_json()["type"] != "token":
                pass
            try:
                with client.websocket_connect(STREAM_PATH + "?token=test-secret") as idle:
                    idle.send_json({"type": "invalid"})
                    assert idle.receive_json()["type"] == "error"
                assert not session._stop_event.is_set()
            finally:
                active.send_json({"type": "stop"})
                while active.receive_json()["type"] not in ("done", "error"):
                    pass


def test_unexpected_commit_error_is_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _StubSession()

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("/home/private-user/private-model-cache/config.json secret=synthetic")

    monkeypatch.setattr(session, "append_turn", fail)
    with TestClient(create_app(cast(DrowseSession, session))) as client:
        with client.websocket_connect(STREAM_PATH) as ws:
            ws.send_json({"type": "submit", "text": "hello", "authored_role": "user"})
            while True:
                message = ws.receive_json()
                if message["type"] == "error":
                    assert message["status"] == 500
                    assert "private-user" not in message["message"]
                    assert "synthetic" not in message["message"]
                    break


@pytest.mark.parametrize(("origin", "allowed", "accepted"), [
    ("http://testserver", [], True),
    ("http://TESTSERVER:80", [], True),
    ("https://testserver", [], False),
    ("null", [], False),
    ("http://testserver/path", [], False),
    ("http://testserver@untrusted.example", [], False),
    ("http://testserver untrusted.example", [], False),
    ("https://untrusted.example", ["*"], False),
    ("https://dashboard.example", ["https://dashboard.example"], True),
])
@pytest.mark.parametrize("key", [None, "test-secret"])
def test_websocket_origin_policy(origin: str, allowed: list[str], accepted: bool, key: str | None) -> None:
    app = create_app(cast(DrowseSession, _StubSession()), cors_origins=allowed, api_key=key)
    path = STREAM_PATH + ("?token=" + key if key else "")
    with TestClient(app) as client:
        if accepted:
            with client.websocket_connect(path, headers={"Origin": origin}) as ws:
                ws.send_json({"type": "invalid"})
                assert ws.receive_json()["status"] == 400
        else:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(path, headers={"Origin": origin}):
                    pass


def test_unkeyed_dns_rebinding_host_is_rejected() -> None:
    app = create_app(cast(DrowseSession, _StubSession()))
    headers = {"Host": "rebind.example", "Origin": "http://rebind.example"}
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(STREAM_PATH, headers=headers):
                pass
        assert client.get("/v1/models", headers=headers).status_code == 403


def test_websocket_duplicate_origin_is_rejected() -> None:
    import httpx
    with TestClient(create_app(cast(DrowseSession, _StubSession()))) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(STREAM_PATH, headers=httpx.Headers([
                ("Origin", "http://testserver"), ("Origin", "https://untrusted.example"),
            ])):
                pass


def test_inbound_budget_fifo_and_reserved_stop() -> None:
    from drowse.server.ws_stream import (
        _InboundBuffer, _InvalidInbound, _Stop, MAX_WS_PENDING_BYTES, MAX_WS_PENDING_MESSAGES,
    )

    async def scenario() -> None:
        queue = _InboundBuffer()
        messages = [_InvalidInbound(str(i)) for i in range(MAX_WS_PENDING_MESSAGES)]
        for message in messages:
            assert queue.put(message, 1)
        assert not queue.put(_InvalidInbound("overflow"), 1)
        for _ in range(100):
            assert queue.put(_Stop(), 10)
        assert isinstance(await queue.get(control_only=True), _Stop)
        assert [await queue.get() for _ in messages] == messages
        assert queue.pending_bytes == 0
        assert queue.put(messages[0], MAX_WS_PENDING_BYTES)
        assert not queue.put(messages[1], 1)
        queue.close(1013)
        assert queue.pending_bytes == 0 and not queue.pending
        assert not queue.put(messages[0], 1)
    asyncio.run(scenario())


def test_websocket_oversized_message_closes_without_mutation() -> None:
    from drowse.server.ws_stream import MAX_WS_MESSAGE_BYTES

    async def scenario() -> None:
        session = _StubSession()
        async with _socket(session) as (incoming, outgoing, _):
            incoming.put_nowait({"type": "websocket.receive", "text": json.dumps({
                "type": "submit", "text": "x" * MAX_WS_MESSAGE_BYTES, "authored_role": "user",
            })})
            frame = await asyncio.wait_for(outgoing.get(), 2)
            assert frame["type"] == "websocket.close" and frame["code"] == 1009
        assert session.tree.rev == 0
    asyncio.run(scenario())


def test_queued_socket_disconnect_never_runs_its_request() -> None:
    async def scenario() -> None:
        session = _StubSession()
        await session.lock.acquire()
        try:
            async with _socket(session) as (incoming, outgoing, task):
                incoming.put_nowait({"type": "websocket.receive", "text": json.dumps({
                    "type": "submit", "text": "must not persist", "authored_role": "user",
                })})
                assert (await asyncio.wait_for(outgoing.get(), 2))["type"] == "websocket.send"
                incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})
                await asyncio.wait_for(task, 2)
                assert session.lock.locked()
        finally:
            session.lock.release()
        assert session.tree.rev == 0
        assert not session.generation_state.is_stop_requested()
    asyncio.run(scenario())


def test_active_socket_disconnect_joins_its_worker() -> None:
    session = _StubSession()
    session._block_until_stop = True
    with TestClient(create_app(cast(DrowseSession, session))) as client:
        with client.websocket_connect(STREAM_PATH) as ws:
            ws.send_json({"type": "generate", "input": "hello"})
            while ws.receive_json()["type"] != "token":
                pass
        assert not session.lock.locked()
        assert session.tree.get(session.tree.active_node_id).finish_reason is not None
        assert not session.generation_state.is_stop_requested()


def test_repeated_cancellation_keeps_worker_lock() -> None:
    from drowse.server.streaming import finish_worker

    async def scenario() -> None:
        lock = asyncio.Lock()
        entered = asyncio.Event()
        release = threading.Event()
        finished = threading.Event()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(3)
            finished.set()

        async def request() -> None:
            async with lock:
                task = asyncio.create_task(asyncio.to_thread(worker))
                try:
                    await asyncio.shield(task)
                finally:
                    await finish_worker(task)

        request_task = asyncio.create_task(request())
        await entered.wait()
        try:
            for _ in range(3):
                request_task.cancel()
                await asyncio.sleep(0)
                assert lock.locked() and not finished.is_set()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await request_task
        assert finished.is_set() and not lock.locked()
    asyncio.run(scenario())


def test_closed_stream_does_not_cancel_a_later_generation() -> None:
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._gen_state = GenerationState()
    session._monitor = SimpleNamespace(probe_names=[])
    session._last_token_probe_payload = {}
    session._generate_core = lambda *args, **kwargs: GenerationResult(
        text="", tokens=[], token_count=0, tok_per_sec=0, elapsed=0,
    )
    stream = session.generate_stream("prompt")
    assert list(stream) == []
    later = threading.Event()
    with session._gen_state.cancellation_scope(later):
        session._gen_state.reset()
        stream.close()
        stream.close()
        assert not session._gen_state.is_stop_requested()


def test_serve_default_is_loopback_and_remote_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from drowse.cli import parse_args
    from drowse.cli.runners.serve import _run_serve
    import drowse.cli.runners as runners

    assert parse_args(["serve", "model"]).host == "127.0.0.1"
    monkeypatch.delenv("DROWSE_API_KEY", raising=False)
    monkeypatch.setattr(runners, "_load_effective_config", lambda args: None)
    for host in ("0.0.0.0", "::", "192.168.1.10"):
        args = parse_args(["serve", "model", "--host", host])
        with pytest.raises(SystemExit) as exc:
            _run_serve(args)
        assert exc.value.code == 2


def test_sse_send_failure_keeps_lock_until_real_worker_exits() -> None:
    from starlette.requests import ClientDisconnect

    async def scenario() -> None:
        lock = asyncio.Lock()
        started = asyncio.Event()
        failed_send = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        finished = threading.Event()

        def worker(progress: Any) -> None:
            loop.call_soon_threadsafe(started.set)
            progress("started")
            assert release.wait(3)
            finished.set()

        async def job(progress: Any) -> None:
            await asyncio.to_thread(worker, progress)

        async def receive() -> dict[str, str]:
            return {"type": "http.disconnect"}

        async def send(message: Any) -> None:
            if message["type"] == "http.response.body":
                failed_send.set()
                raise OSError("disconnected")

        response = progress_sse_response(lock, job, error_message="failed", log_message="failed")
        task = asyncio.create_task(response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send))
        await asyncio.wait_for(started.wait(), 2)
        await asyncio.wait_for(failed_send.wait(), 2)
        try:
            await asyncio.sleep(0)
            assert lock.locked()
            assert not task.done()
        finally:
            release.set()
            with pytest.raises(ClientDisconnect):
                await task
        assert finished.is_set() and not lock.locked()
    asyncio.run(scenario())


def test_outbound_queue_bounds_pending_thread_callbacks() -> None:
    from drowse.server.ws_stream import _OutboundQueue, MAX_WS_OUTBOUND_EVENTS

    async def scenario() -> None:
        overflows: list[bool] = []
        queue = _OutboundQueue[int](lambda: overflows.append(True))
        for i in range(10_000):
            queue.put(i)
        await asyncio.sleep(0)
        assert overflows == [True]
        assert [await queue.get() for _ in range(MAX_WS_OUTBOUND_EVENTS)] == list(range(MAX_WS_OUTBOUND_EVENTS))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.get(), 0.01)
        queue.close()
    asyncio.run(scenario())


def test_slow_tree_subscriber_is_disconnected() -> None:
    from drowse.core.events import LoomMutated
    from drowse.server.ws_stream import MAX_WS_OUTBOUND_EVENTS

    async def scenario() -> None:
        session = _StubSession()
        async with _socket(session, asyncio.Event()) as (_, outgoing, task):
            for rev in range(MAX_WS_OUTBOUND_EVENTS + 10):
                session.events.emit(LoomMutated(op="cast", rev=rev))
            frame = await asyncio.wait_for(outgoing.get(), 2)
            assert frame["type"] == "websocket.close" and frame["code"] == 1013
            await asyncio.wait_for(task, 2)
        assert not session.generation_state.is_stop_requested()
    asyncio.run(scenario())


@pytest.mark.parametrize(("url", "origin", "allowed"), [
    ("wss://[::1]", "https://[::1]:443", True),
    ("ws://[::1]:8000", "http://[::1]:8000", True),
    ("ws://[::1]:8000", "http://[::1]", False),
])
def test_ipv6_websocket_origins(url: str, origin: str, allowed: bool) -> None:
    headers = {"Host": url.partition("://")[2], "Origin": origin}
    target = url.replace("[::1]", "testserver") + STREAM_PATH
    with TestClient(create_app(cast(DrowseSession, _StubSession()))) as client:
        if allowed:
            with client.websocket_connect(target, headers=headers) as ws:
                ws.send_json({"type": "invalid"})
                assert ws.receive_json()["status"] == 400
        else:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(target, headers=headers):
                    pass


@pytest.mark.parametrize("cli_key", [None, "cli-secret"])
def test_remote_serve_honors_key_precedence_and_transport_limit(monkeypatch: pytest.MonkeyPatch, cli_key: str | None) -> None:
    from drowse.cli import parse_args
    import drowse.cli.runners as runners
    import drowse.cli.runners.serve as serve
    import drowse.server as server
    import uvicorn
    from drowse.server.ws_stream import MAX_WS_MESSAGE_BYTES

    captured: dict[str, Any] = {}
    monkeypatch.setenv("DROWSE_API_KEY", "env-secret")
    for name in ("_load_effective_config", "_print_startup", "_print_model_info"):
        monkeypatch.setattr(runners, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(runners, "_make_session", lambda args: object())
    for name in ("_setup_steering_vectors", "_warmup_session", "_enable_serve_live_lens_if_compatible"):
        monkeypatch.setattr(serve, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "create_app", lambda *args, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: captured.update(kwargs))
    argv = ["serve", "model", "--host", "0.0.0.0", "--no-web"]
    if cli_key:
        argv += ["--api-key", cli_key]
    serve._run_serve(parse_args(argv))
    assert captured["api_key"] == (cli_key or "env-secret")
    assert captured["host"] == "0.0.0.0"
    assert captured["ws_max_size"] == MAX_WS_MESSAGE_BYTES


def test_socket_lifecycles_release_subscriptions() -> None:
    session = _StubSession()
    baseline = len(session.events._subs)
    with TestClient(create_app(cast(DrowseSession, session))) as client:
        for _ in range(30):
            with client.websocket_connect(STREAM_PATH) as ws:
                ws.send_json({"type": "invalid"})
                assert ws.receive_json()["status"] == 400
            assert len(session.events._subs) == baseline


@pytest.mark.parametrize("read_first", [True, False])
def test_stream_reports_worker_error_once_and_close_is_idempotent(read_first: bool) -> None:
    session: Any = _CurrentSessionStub.__new__(_CurrentSessionStub)
    session._gen_state = GenerationState()
    session._monitor = SimpleNamespace(probe_names=[])
    session._last_token_probe_payload = {}

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("worker failed")

    session._generate_core = fail
    stream = session.generate_stream("prompt")
    with pytest.raises(RuntimeError, match="worker failed"):
        if read_first:
            list(stream)
        else:
            stream.close()
    stream.close()
