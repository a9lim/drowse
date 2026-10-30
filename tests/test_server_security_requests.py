from __future__ import annotations

import json
from typing import Any, cast
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from drowse.session.session import DrowseSession
from drowse.server import create_app
from tests.test_server_loom import _StubSession


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
    with TestClient(create_app(cast(DrowseSession, _StubSession()))) as client:
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
    from tests.test_server import _mock_session

    session = _mock_session()
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
    session = _StubSession()
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
    session = _StubSession()
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
    from tests.test_server import _mock_session

    session = _mock_session()
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


@pytest.mark.parametrize("frame_type", ["generate", "submit"])
def test_websocket_fan_size_is_bounded_before_seed_allocation(frame_type: str) -> None:
    session = _StubSession()
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
    session: Any = _StubSession()
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

    session: Any = _StubSession()
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
