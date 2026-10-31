"""HTTP trust boundaries: framing, CSP, caching, auth, request validation and limits."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from starlette.types import Message

from drowse.session.session import DrowseSession
from drowse.server import create_app
from tests.support.server import StubSession
from types import SimpleNamespace
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from tests.support.server import mock_session

SESSION = "/drowse/v1/sessions/default"


def test_cross_origin_simple_post_cannot_clear_conversation() -> None:
    session = StubSession()
    session.append_turn(None, "private synthetic conversation", role="user")
    before = session.tree.to_dict()
    with TestClient(create_app(cast(DrowseSession, session))) as client:
        response = client.post(SESSION + "/clear", headers={"Origin": "https://untrusted.example"})
    assert response.status_code == 403
    assert session.tree.to_dict() == before


def test_local_dashboard_cannot_be_framed() -> None:
    with TestClient(create_app(cast(DrowseSession, StubSession()), web=True)) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.headers.get("x-frame-options") == "DENY"
    assert "frame-ancestors 'none'" in response.headers.get("content-security-policy", "")


def test_private_api_responses_are_not_cacheable() -> None:
    with TestClient(create_app(cast(DrowseSession, StubSession()))) as client:
        response = client.get(SESSION + "/tree")
    assert response.status_code == 200
    assert "no-store" in response.headers.get("cache-control", "")


def test_unauthenticated_body_is_never_read() -> None:
    async def scenario() -> None:
        reads: list[bool] = []
        sent: list[Message] = []
        app = create_app(cast(DrowseSession, StubSession()), api_key="synthetic-secret")
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
    from drowse.lens.sources import _resolve_model_for_fetch

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
    app = create_app(cast(DrowseSession, StubSession()), max_request_bytes=128)
    payload = json.dumps({"messages": [{"role": "user", "content": "x" * 256}]}).encode()
    content: Any = iter([payload[:90], payload[90:]]) if chunked else payload
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", content=content, headers={"Content-Type": "application/json"})
    assert response.status_code == 413
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_cors_wildcard_cannot_expose_conversations_or_authorize_mutations(method: str) -> None:
    app = create_app(cast(DrowseSession, StubSession()), cors_origins=["*"])
    with TestClient(app) as client:
        response = client.request(method, SESSION + ("/tree" if method == "GET" else "/tree/reset"), headers={"Origin": "https://untrusted.example"})
    assert response.status_code == 403


@pytest.mark.parametrize("origin", [None, "http://testserver", "https://trusted.example"])
def test_http_origin_policy_preserves_trusted_clients(origin: str | None) -> None:
    app = create_app(cast(DrowseSession, StubSession()), cors_origins=["https://trusted.example"])
    with TestClient(app) as client:
        response = client.post(SESSION + "/tree/reset", headers={"Origin": origin} if origin else {})
    assert response.status_code == 204


def test_websocket_auth_subprotocol_keeps_credentials_out_of_url() -> None:
    key = "synthetic +/ secret"
    encoded = base64.urlsafe_b64encode(key.encode()).decode().rstrip("=")
    app = create_app(cast(DrowseSession, StubSession()), api_key=key)
    with TestClient(app) as client:
        with client.websocket_connect(SESSION + "/stream", subprotocols=["drowse.v1", "drowse.auth." + encoded]) as socket:
            assert socket.accepted_subprotocol == "drowse.v1"
            socket.send_json({"type": "invalid"})
            assert socket.receive_json()["type"] == "error"


@pytest.mark.parametrize("key", ["synthetic-secret", "invalid-secret"])
def test_websocket_query_credentials_are_rejected_and_scrubbed_before_access_logging(key: str) -> None:
    """URL credentials never authenticate, and a stale client's ``?token=``
    is removed from the scope the access log reads, valid key or not."""
    from starlette.websockets import WebSocket
    from drowse.server.security import ws_auth_ok

    app = create_app(cast(DrowseSession, StubSession()), api_key="synthetic-secret")
    scope: Any = {"type": "websocket", "app": app, "scheme": "ws", "path": SESSION + "/stream",
                  "headers": [(b"host", b"localhost")], "server": ("127.0.0.1", 80),
                  "query_string": f"token={key}&view=tree".encode(), "subprotocols": []}
    async def unused_receive() -> Message:
        return {"type": "websocket.disconnect"}

    async def unused_send(message: Message) -> None:
        pass

    assert ws_auth_ok(WebSocket(scope, unused_receive, unused_send)) is False
    assert scope["query_string"] == b"view=tree"


@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
def test_install_errors_never_disclose_filesystem_paths(monkeypatch: pytest.MonkeyPatch, accept: str) -> None:
    import drowse.server.manifold_routes as routes

    def fail(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("/private/synthetic-user/secret-model-cache/manifest.json")

    monkeypatch.setattr(routes, "install_manifold", fail)
    monkeypatch.setattr(routes, "refuse_if_busy", lambda session: None)
    with TestClient(create_app(cast(DrowseSession, StubSession()))) as client:
        response = client.post("/drowse/v1/manifolds/install", json={"target": "synthetic/missing"}, headers={"Accept": accept})
    assert response.status_code == (404 if accept == "application/json" else 200)
    assert "/private/" not in response.text


@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
def test_install_transport_errors_never_disclose_signed_urls(monkeypatch: pytest.MonkeyPatch, accept: str) -> None:
    import drowse.server.manifold_routes as routes

    from drowse.base.hf import HFError

    def fail(*args: Any, **kwargs: Any) -> None:
        try:
            raise RuntimeError("https://synthetic.example/model?token=synthetic-download-credential")
        except RuntimeError as exc:
            raise HFError(f"synthetic/model: download failed ({exc})") from exc

    monkeypatch.setattr(routes, "install_manifold", fail)
    monkeypatch.setattr(routes, "refuse_if_busy", lambda session: None)
    with TestClient(create_app(cast(DrowseSession, StubSession()))) as client:
        response = client.post("/drowse/v1/manifolds/install", json={"target": "synthetic/model"}, headers={"Accept": accept})
    assert response.status_code == (502 if accept == "application/json" else 200)
    assert "synthetic-download-credential" not in response.text


@pytest.mark.parametrize("length, status", [("-1", 400), ("9" * 10000, 413), ("000" * 2000 + "129", 413)])
def test_invalid_or_extreme_content_length_fails_closed(length: str, status: int) -> None:
    with TestClient(create_app(cast(DrowseSession, StubSession()), max_request_bytes=128)) as client:
        response = client.post("/v1/chat/completions", content=b"{}", headers={"Content-Length": length})
    assert response.status_code == status


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_body_limit_rejected_at_app_creation(limit: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        create_app(cast(DrowseSession, StubSession()), max_request_bytes=limit)


@pytest.mark.parametrize("transport", ["http", "websocket"])
def test_remote_clients_cannot_bypass_missing_key_with_loopback_host(transport: str) -> None:
    from starlette.websockets import WebSocketDisconnect

    with TestClient(create_app(cast(DrowseSession, StubSession())), client=("198.51.100.10", 12345)) as client:
        if transport == "http":
            assert client.get("/v1/models", headers={"Host": "localhost"}).status_code == 403
        else:
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(SESSION + "/stream", headers={"Host": "localhost"}):
                    pass


def test_authenticated_remote_clients_remain_supported() -> None:
    with TestClient(create_app(cast(DrowseSession, StubSession()), api_key="synthetic-key"), client=("198.51.100.10", 12345)) as client:
        assert client.get("/v1/models", headers={"Authorization": "Bearer synthetic-key"}).status_code == 200


def test_dashboard_can_submit_feedback_without_broadening_other_connections():
    with TestClient(create_app(cast(DrowseSession, StubSession()), web=True)) as client:
        response = client.get("/")
    policy = response.headers["content-security-policy"]
    connections = next(directive.strip().split()[1:] for directive in policy.split(";")
                       if directive.strip().startswith("connect-src "))
    assert connections == ["'self'", "https://www.neuronpedia.org", "https://drowse.ai/api/contact"]
    assert "form-action 'self'" in policy


# ---------------------------------------------------------------------------
# Request validation and limits
# ---------------------------------------------------------------------------


INVALID_SAMPLING = [
    {"temperature": float("nan")}, {"temperature": float("inf")},
    {"top_p": float("nan")}, {"top_p": -0.01}, {"top_p": 1.01},
    {"presence_penalty": float("inf")}, {"frequency_penalty": float("nan")},
    {"logit_bias": {"1": float("nan")}}, {"logit_bias": {"-1": 2}},
    {"max_tokens": -1},
]

@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
@pytest.mark.parametrize("error_type, status", [(FileNotFoundError, 404), (PermissionError, 500)])
def test_retained_install_errors_do_not_disclose_paths(
    monkeypatch: pytest.MonkeyPatch, accept: str, error_type: type[Exception], status: int,
) -> None:
    import drowse.server.manifold_routes as routes

    def fail(*args: Any, **kwargs: Any) -> None:
        raise error_type("/private/synthetic-user/secret-model-cache/manifest.json")

    monkeypatch.setattr(routes, "install_manifold", fail)
    monkeypatch.setattr(routes, "refuse_if_busy", lambda session: None)
    with TestClient(create_app(cast(DrowseSession, StubSession()))) as client:
        response = client.post(
            "/drowse/v1/manifolds/install", json={"target": "synthetic/missing"},
            headers={"Accept": accept, "X-Drowse-Request-ID": "retained-install"},
        )
        receipt = client.get("/drowse/v1/operations/retained-install")
    assert response.status_code == (status if accept == "application/json" else 200)
    assert receipt.json()["state"] == "failed"
    assert "/private/" not in response.text
    assert "/private/" not in receipt.text

@pytest.mark.parametrize("sampling", INVALID_SAMPLING)
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions"])
def test_invalid_openai_sampling_never_starts_inference(path: str, sampling: dict[str, Any]) -> None:
    session = mock_session()
    with TestClient(create_app(session)) as client:
        response = client.post(path, content=json.dumps({
            "messages": [{"role": "user", "content": "test"}], "prompt": "test",
            "stream": True, **sampling,
        }), headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    session.generate.assert_not_called()
    session.generate_stream.assert_not_called()

@pytest.mark.parametrize("sampling", INVALID_SAMPLING)
@pytest.mark.parametrize("frame_type", ["generate", "submit"])
def test_invalid_websocket_sampling_keeps_connection_usable(frame_type: str, sampling: dict[str, Any]) -> None:
    session = StubSession()
    original = session.tree.to_dict()
    with TestClient(create_app(cast(DrowseSession, session))) as client:
        with client.websocket_connect("/drowse/v1/sessions/default/stream") as socket:
            socket.send_text(json.dumps({"type": frame_type, "sampling": sampling}))
            error = socket.receive_json()
            assert error["type"] == "error" and error["status"] == 400
            socket.send_json({"type": "unknown"})
            assert socket.receive_json()["status"] == 400
    assert session.tree.to_dict() == original

@pytest.mark.parametrize("sampling", [
    {"temperature": float("nan")}, {"top_p": float("inf")},
    {"top_p": -1}, {"top_p": 2}, {"max_tokens": -1},
])
def test_invalid_session_sampling_cannot_poison_defaults(sampling: dict[str, Any]) -> None:
    session = StubSession()
    original = session.config
    with TestClient(create_app(cast(DrowseSession, session))) as client:
        response = client.patch("/drowse/v1/sessions/default", content=json.dumps(sampling),
                                headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert session.config is original

@pytest.mark.parametrize("option, value", [
    ("num_predict", float("inf")), ("top_k", float("inf")),
    ("seed", "not-an-integer"), ("seed", float("inf")),
    ("seed", 1.5), ("num_predict", 1.5), ("top_k", 1.5),
    ("top_p", -0.1), ("top_p", 1.1),
])
def test_invalid_ollama_options_return_client_error(option: str, value: Any) -> None:
    session = mock_session()
    with TestClient(create_app(session)) as client:
        response = client.post("/api/generate", content=json.dumps({
            "model": "test/model", "prompt": "test", "options": {option: value},
        }), headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    session.generate.assert_not_called()
    session.generate_stream.assert_not_called()

def test_sampling_validation_preserves_boundary_and_sentinel_meanings() -> None:
    from drowse.server.compat.ollama import OllamaGenerateRequest, resolve_options
    from drowse.server.compat.openai import ChatCompletionRequest
    from drowse.server.session_models import PatchSessionRequest
    from drowse.server.ws_models import WSSamplingParams, build_sampling_config

    sampling: dict[str, Any] = {"temperature": 0, "top_p": 0, "max_tokens": 0}
    assert ChatCompletionRequest(messages=[], **sampling).max_tokens == 0
    assert PatchSessionRequest(**sampling, top_k=None).top_k is None
    config = build_sampling_config(WSSamplingParams(**sampling, top_k=0))
    assert config is not None and config.temperature == 0
    for count in [-2, -1, 0]:
        body = OllamaGenerateRequest(options={
            "temperature": -1, "top_p": 1, "top_k": -1, "num_predict": count, "seed": "42",
        })
        gen = resolve_options(body, None, "test", raw=False)
        assert gen.sampling.max_tokens == count
        assert gen.sampling.top_k is None
        assert gen.sampling.temperature == -1
        assert gen.sampling.seed == 42

def test_dropped_ollama_option_names_cannot_forge_log_lines(caplog: pytest.LogCaptureFixture) -> None:
    from drowse.server.compat.ollama import OllamaGenerateRequest, resolve_options

    body = OllamaGenerateRequest(options={"min_p\nINFO forged entry": 0.1})
    with caplog.at_level("DEBUG", logger="drowse.server.compat.ollama"):
        resolve_options(body, None, "test", raw=False)
    dropped = [r.getMessage() for r in caplog.records if "unsupported options" in r.getMessage()]
    assert dropped == ["ollama: unsupported options dropped: min_pINFO forged entry"]

@pytest.mark.parametrize("frame_type", ["generate", "submit"])
def test_websocket_fan_size_is_bounded_before_seed_allocation(frame_type: str) -> None:
    session = StubSession()
    before = session.tree.to_dict()
    with TestClient(create_app(cast(DrowseSession, session))) as client:
        with client.websocket_connect("/drowse/v1/sessions/default/stream") as socket:
            socket.send_json({"type": frame_type, "n": 1025})
            error = socket.receive_json()
            assert error["type"] == "error" and error["status"] == 400
            assert "1024" in error["message"]
            socket.send_json({"type": "unknown"})
            assert socket.receive_json()["status"] == 400
    assert session.tree.to_dict() == before

@pytest.mark.parametrize("transport", ["http", "websocket"])
def test_bias_ids_are_validated_against_the_output_vocabulary(transport: str) -> None:
    session: Any = StubSession()
    session.model = SimpleNamespace(get_output_embeddings=lambda: SimpleNamespace(weight=SimpleNamespace(shape=(32, 16))))
    original = session.tree.to_dict()
    with TestClient(create_app(cast(DrowseSession, session))) as client:
        if transport == "http":
            response = client.post("/v1/completions", json={"prompt": "test", "logit_bias": {"32": 1}})
            assert response.status_code == 400
            assert "between 0 and 31" in response.text
        else:
            with client.websocket_connect("/drowse/v1/sessions/default/stream") as socket:
                socket.send_json({"type": "generate", "sampling": {"logit_bias": {"32": 1}}})
                error = socket.receive_json()
                assert error["status"] == 400 and "between 0 and 31" in error["message"]
                socket.send_json({"type": "unknown"})
                assert socket.receive_json()["status"] == 400
    assert session.tree.to_dict() == original

def test_bias_validation_preserves_valid_values() -> None:
    from drowse.server.request_helpers import build_sampling_config, validate_logit_bias

    session: Any = StubSession()
    session.model = SimpleNamespace(get_output_embeddings=lambda: SimpleNamespace(weight=SimpleNamespace(shape=(32, 16))))
    bias = {0: -100.0, 31: 100.0}
    sampling = build_sampling_config(logit_bias=bias)
    validate_logit_bias(sampling, cast(DrowseSession, session))
    assert sampling.logit_bias is bias

def _operation_app() -> FastAPI:
    import asyncio

    from drowse.server.errors import install_error_handlers
    from drowse.server.operations import register_operation_routes

    app = FastAPI()
    app.state.session = SimpleNamespace(model_id="synthetic/model")
    app.state.session_lock = asyncio.Lock()
    install_error_handlers(app)
    register_operation_routes(app)
    return app

@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("accept", ["application/json", "text/event-stream"])
def test_retained_typed_transport_errors_keep_safe_messages(wrapped: bool, accept: str) -> None:
    from drowse.base.hf import HFError
    from drowse.server.operations import run_operation

    app = _operation_app()

    @app.post("/drowse/v1/test")
    async def fail(request: Request) -> Any:
        async def job(progress: Any) -> None:
            error = HFError("https://synthetic.example/download?token=secret-upstream-credential")
            error.__cause__ = RuntimeError("synthetic transport failure")
            if wrapped:
                raise HTTPException(502, str(error)) from error
            raise error

        return await run_operation(request, job, failure="failed")

    with TestClient(app) as client:
        result = client.post("/drowse/v1/test", json={}, headers={
            "X-Drowse-Request-ID": "safe-error", "Accept": accept,
        })
        receipt = client.get("/drowse/v1/operations/safe-error")
    assert result.status_code == (502 if accept == "application/json" else 200)
    assert receipt.json()["error"]["status"] == 502
    assert "secret-upstream-credential" not in result.text
    assert "secret-upstream-credential" not in receipt.text

def test_retained_operation_keeps_typed_error_status() -> None:
    from drowse.errors import SteeringCompositionError
    from drowse.server.operations import run_operation

    app = _operation_app()

    @app.post("/drowse/v1/test")
    async def fail(request: Request) -> Any:
        async def job(progress: Any) -> None:
            raise SteeringCompositionError("incompatible steering geometry")

        return await run_operation(request, job, failure="failed")

    with TestClient(app) as client:
        result = client.post("/drowse/v1/test", json={}, headers={"X-Drowse-Request-ID": "typed-status"})
        receipt = client.get("/drowse/v1/operations/typed-status")
    assert result.status_code == 422
    assert receipt.json()["error"]["status"] == 422
