"""Adversarial checks for HTTP and artifact trust boundaries."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from starlette.types import Message

from drowse.core.session import DrowseSession
from drowse.server import create_app
from tests.test_server_loom import _StubSession

SESSION = "/drowse/v1/sessions/default"


def test_cross_origin_simple_post_cannot_clear_conversation() -> None:
    session = _StubSession()
    session.append_turn(None, "private synthetic conversation", role="user")
    before = session.tree.to_dict()
    with TestClient(create_app(cast(DrowseSession, session))) as client:
        response = client.post(SESSION + "/clear", headers={"Origin": "https://untrusted.example"})
    assert response.status_code == 403
    assert session.tree.to_dict() == before


def test_local_dashboard_cannot_be_framed() -> None:
    with TestClient(create_app(cast(DrowseSession, _StubSession()), web=True)) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.headers.get("x-frame-options") == "DENY"
    assert "frame-ancestors 'none'" in response.headers.get("content-security-policy", "")


def test_private_api_responses_are_not_cacheable() -> None:
    with TestClient(create_app(cast(DrowseSession, _StubSession()))) as client:
        response = client.get(SESSION + "/tree")
    assert response.status_code == 200
    assert "no-store" in response.headers.get("cache-control", "")


def test_unauthenticated_body_is_never_read() -> None:
    async def scenario() -> None:
        reads: list[bool] = []
        sent: list[Message] = []
        app = create_app(cast(DrowseSession, _StubSession()), api_key="synthetic-secret")
        scope: dict[str, Any] = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1", "scheme": "http", "method": "POST",
            "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
            "root_path": "", "query_string": b"",
            "headers": [(b"host", b"localhost"), (b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 80),
        }

        async def receive() -> Message:
            reads.append(True)
            return {"type": "http.request", "body": b'{"invalid":', "more_body": False}

        async def send(message: Message) -> None:
            sent.append(message)

        await app(scope, receive, send)
        assert not reads
        assert sent[0]["status"] == 401
    asyncio.run(scenario())


def test_metadata_only_lens_fetch_never_executes_repository_code(tmp_path: Path) -> None:
    from drowse.io.lens_sources import _resolve_model_for_fetch

    marker = tmp_path / "repository-code-executed"
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "security_fixture",
        "auto_map": {"AutoConfig": "configuration_fixture.FixtureConfig"},
        "hidden_size": 4, "num_hidden_layers": 2,
    }))
    (model / "configuration_fixture.py").write_text(
        "from pathlib import Path\n"
        "from transformers import PretrainedConfig\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "class FixtureConfig(PretrainedConfig):\n"
        "    model_type = 'security_fixture'\n"
    )
    with pytest.raises(ValueError):
        _resolve_model_for_fetch(str(model))
    assert not marker.exists()


@pytest.mark.parametrize("chunked", [False, True])
def test_http_body_limit_checks_declared_and_streamed_bytes(chunked: bool) -> None:
    app = create_app(cast(DrowseSession, _StubSession()), max_request_bytes=128)
    payload = json.dumps({"messages": [{"role": "user", "content": "x" * 256}]}).encode()
    content: Any = iter([payload[:90], payload[90:]]) if chunked else payload
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", content=content, headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_cors_wildcard_cannot_expose_conversations_or_authorize_mutations(method: str) -> None:
    app = create_app(cast(DrowseSession, _StubSession()), cors_origins=["*"])
    with TestClient(app) as client:
        response = client.request(method, SESSION + ("/tree" if method == "GET" else "/clear"), headers={"Origin": "https://untrusted.example"})
    assert response.status_code == 403


@pytest.mark.parametrize("origin", [None, "http://testserver", "https://trusted.example"])
def test_http_origin_policy_preserves_trusted_clients(origin: str | None) -> None:
    app = create_app(cast(DrowseSession, _StubSession()), cors_origins=["https://trusted.example"])
    with TestClient(app) as client:
        response = client.post(SESSION + "/clear", headers={"Origin": origin} if origin else {})
    assert response.status_code == 204


def test_websocket_auth_subprotocol_keeps_credentials_out_of_url() -> None:
    key = "synthetic +/ secret"
    encoded = base64.urlsafe_b64encode(key.encode()).decode().rstrip("=")
    app = create_app(cast(DrowseSession, _StubSession()), api_key=key)
    with TestClient(app) as client:
        with client.websocket_connect(SESSION + "/stream", subprotocols=["drowse.v1", "drowse.auth." + encoded]) as socket:
            assert socket.accepted_subprotocol == "drowse.v1"
            socket.send_json({"type": "invalid"})
            assert socket.receive_json()["type"] == "error"


@pytest.mark.parametrize("key", ["synthetic-secret", "invalid-secret"])
def test_legacy_websocket_query_credentials_are_removed_before_access_logging(key: str) -> None:
    from starlette.websockets import WebSocket
    from drowse.server.app import ws_auth_ok

    app = create_app(cast(DrowseSession, _StubSession()), api_key="synthetic-secret")
    scope: Any = {"type": "websocket", "app": app, "scheme": "ws", "path": SESSION + "/stream",
                  "headers": [(b"host", b"localhost")], "server": ("127.0.0.1", 80),
                  "query_string": f"token={key}&view=tree".encode(), "subprotocols": []}
    async def unused_receive() -> Message:
        return {"type": "websocket.disconnect"}

    async def unused_send(message: Message) -> None:
        pass

    assert ws_auth_ok(WebSocket(scope, unused_receive, unused_send)) == (key == "synthetic-secret")
    assert scope["query_string"] == b"view=tree"


@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
def test_install_errors_never_disclose_filesystem_paths(monkeypatch: pytest.MonkeyPatch, accept: str) -> None:
    import drowse.server.manifold_routes as routes

    def fail(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("/private/synthetic-user/secret-model-cache/manifest.json")

    monkeypatch.setattr(routes, "install_manifold", fail)
    monkeypatch.setattr(routes, "refuse_if_busy", lambda session: None)
    with TestClient(create_app(cast(DrowseSession, _StubSession()))) as client:
        response = client.post("/drowse/v1/manifolds/install", json={"target": "synthetic/missing"}, headers={"Accept": accept})
    assert response.status_code == (404 if accept == "application/json" else 200)
    assert "/private/" not in response.text


@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
def test_install_transport_errors_never_disclose_signed_urls(monkeypatch: pytest.MonkeyPatch, accept: str) -> None:
    import drowse.server.manifold_routes as routes

    def fail(*args: Any, **kwargs: Any) -> None:
        try:
            raise RuntimeError("https://synthetic.example/model?token=synthetic-download-credential")
        except RuntimeError as exc:
            raise routes.ManifoldHFError(f"synthetic/model: download failed ({exc})") from exc

    monkeypatch.setattr(routes, "install_manifold", fail)
    monkeypatch.setattr(routes, "refuse_if_busy", lambda session: None)
    with TestClient(create_app(cast(DrowseSession, _StubSession()))) as client:
        response = client.post("/drowse/v1/manifolds/install", json={"target": "synthetic/model"}, headers={"Accept": accept})
    assert response.status_code == (502 if accept == "application/json" else 200)
    assert "synthetic-download-credential" not in response.text


@pytest.mark.parametrize("length, status", [("-1", 400), ("9" * 10000, 413), ("000" * 2000 + "129", 413)])
def test_invalid_or_extreme_content_length_fails_closed(length: str, status: int) -> None:
    with TestClient(create_app(cast(DrowseSession, _StubSession()), max_request_bytes=128)) as client:
        response = client.post("/v1/chat/completions", content=b"{}", headers={"Content-Length": length})
    assert response.status_code == status


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_body_limit_rejected_at_app_creation(limit: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        create_app(cast(DrowseSession, _StubSession()), max_request_bytes=limit)


@pytest.mark.parametrize("transport", ["http", "websocket"])
def test_remote_clients_cannot_bypass_missing_key_with_loopback_host(transport: str) -> None:
    from starlette.websockets import WebSocketDisconnect

    with TestClient(create_app(cast(DrowseSession, _StubSession())), client=("198.51.100.10", 12345)) as client:
        if transport == "http":
            assert client.get("/v1/models", headers={"Host": "localhost"}).status_code == 403
        else:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(SESSION + "/stream", headers={"Host": "localhost"}):
                    pass


def test_authenticated_remote_clients_remain_supported() -> None:
    with TestClient(create_app(cast(DrowseSession, _StubSession()), api_key="synthetic-key"), client=("198.51.100.10", 12345)) as client:
        assert client.get("/v1/models", headers={"Authorization": "Bearer synthetic-key"}).status_code == 200
