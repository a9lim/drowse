from typing import cast
import time

import pytest
from fastapi.testclient import TestClient

from drowse.core.session import DrowseSession
from drowse.server import create_app
from tests.test_drowse_api import _mock_session, TestWebSocket as _WebSocketFixture
from tests.test_server_loom import _StubSession


STREAM = "/drowse/v1/sessions/default/stream"
REQUESTS = "/drowse/v1/sessions/default/requests"


def test_completed_request_reconciles_after_reconnect_without_duplicate_compute():
    session = _mock_session()
    _WebSocketFixture()._attach_generate(session, ["ahoy"])
    payload = {"type": "generate", "request_id": "recover", "input": "hi", "n": 2}
    with TestClient(create_app(session)) as client:
        for _ in range(2):
            with client.websocket_connect(STREAM) as ws:
                ws.send_json(payload)
                while ws.receive_json()["type"] != "request_complete":
                    pass
        receipt = client.get(f"{REQUESTS}/recover").json()
        assert receipt["state"] == "completed"
        assert len(receipt["results"]) == 2
        with client.websocket_connect(STREAM) as ws:
            ws.send_json({**payload, "input": "different"})
            assert ws.receive_json()["code"] == "REQUEST_ID_CONFLICT"
        assert client.get(f"{REQUESTS}/missing").json()["state"] == "unknown"
    assert session.generate.call_count == 2, "replayed fan receipts do not generate another fan"


def test_disconnected_active_request_preserves_cancelled_result_without_reexecution():
    session = _StubSession()
    session._block_until_stop = True
    payload = {"type": "generate", "request_id": "disconnect", "input": "hi", "stateless": False}
    with TestClient(create_app(cast(DrowseSession, session))) as client:
        with client.websocket_connect(STREAM) as ws:
            ws.send_json(payload)
            while ws.receive_json()["type"] != "token":
                pass
            assert client.get(f"{REQUESTS}/disconnect").json()["state"] == "running"
        receipt = client.get(f"{REQUESTS}/disconnect").json()
        assert receipt["state"] == "cancelled"
        assert len(receipt["results"]) == 1
        assert receipt["results"][0]["result"]["finish_reason"] == "cancelled"
        node_id = receipt["results"][0]["node_id"]
        assert node_id is not None
        assert session.tree.get(node_id).finish_reason == "stop"
        tree_revision = session.tree.rev
        with client.websocket_connect(STREAM) as ws:
            ws.send_json(payload)
            assert ws.receive_json() == receipt["results"][0]
            assert ws.receive_json()["state"] == "cancelled"
        assert session.tree.rev == tree_revision


def test_system_prompt_can_be_restored_to_null():
    session = _mock_session()
    with TestClient(create_app(session)) as client:
        assert client.patch("/drowse/v1/sessions/default", json={"system_prompt": "pirate"}).json()["config"]["system_prompt"] == "pirate"
        assert client.patch("/drowse/v1/sessions/default", json={"system_prompt": None}).json()["config"]["system_prompt"] is None


@pytest.mark.parametrize("prompt", [None, "", "Speak like a pirate"])
def test_correlated_submit_carries_per_call_prompt_without_mutating_defaults(prompt: str | None):
    session = _mock_session()
    _WebSocketFixture()._attach_generate(session, ["ahoy"])
    original = session.config.system_prompt
    with TestClient(create_app(session)) as client:
        with client.websocket_connect(STREAM) as ws:
            ws.send_json({"type": "submit", "request_id": "prompt-test", "text": "hi", "authored_role": "user", "generated_role": "assistant", "system_prompt": prompt})
            while ws.receive_json()["type"] != "request_complete":
                pass
    assert session.generate.call_args.kwargs["system_prompt"] == prompt
    assert session.config.system_prompt == original


@pytest.mark.parametrize("seed", [-(2**53 - 1), 2**53 - 1])
def test_correlated_single_generation_preserves_large_signed_seed(seed: int):
    session = _mock_session()
    _WebSocketFixture()._attach_generate(session, ["ahoy"])
    with TestClient(create_app(session)) as client:
        with client.websocket_connect(STREAM) as ws:
            ws.send_json({"type": "generate", "request_id": "seed-test", "input": "hi", "sampling": {"seed": seed}})
            while True:
                frame = ws.receive_json()
                if frame["type"] == "request_complete":
                    break
    assert frame["state"] == "completed"
    assert session.generate.call_args.kwargs["sampling"].seed == seed


def test_correlated_generation_waits_for_all_siblings():
    session = _mock_session()
    _WebSocketFixture()._attach_generate(session, ["ahoy"])
    with TestClient(create_app(session)) as client:
        with client.websocket_connect(STREAM) as ws:
            ws.send_json({"type": "generate", "request_id": "pirate-test", "input": "hi", "n": 2})
            frames = []
            while True:
                frame = ws.receive_json()
                frames.append(frame)
                if frame["type"] == "request_complete":
                    break
    assert all(frame.get("request_id") == "pirate-test" for frame in frames)
    assert len([frame for frame in frames if frame["type"] == "done"]) == 2
    assert frames[-1] == {"type": "request_complete", "request_id": "pirate-test",
                          "state": "completed", "completed_siblings": 2}


def test_correlated_validation_error_is_terminal_without_generation():
    session = _mock_session()
    with TestClient(create_app(session)) as client:
        with client.websocket_connect(STREAM) as ws:
            ws.send_json({"type": "generate", "request_id": "invalid", "input": "hi", "n": 0})
            error = ws.receive_json()
            complete = ws.receive_json()
    assert error["type"] == "error" and error["request_id"] == "invalid"
    assert complete == {"type": "request_complete", "request_id": "invalid",
                        "state": "failed", "completed_siblings": 0}
    session.generate.assert_not_called()


def test_correlated_commit_preserves_request_id_after_submit_normalization():
    session = _StubSession()
    with TestClient(create_app(cast(DrowseSession, session))) as client:
        with client.websocket_connect(STREAM) as ws:
            ws.send_json({"type": "submit", "request_id": "append", "text": "ahoy", "authored_role": "user"})
            frames = []
            while True:
                frame = ws.receive_json()
                frames.append(frame)
                if frame["type"] == "request_complete":
                    break
    assert frames[-1]["state"] == "completed"
    assert [frame for frame in frames if frame["type"] == "done"][0]["request_id"] == "append"


def test_scoped_stop_does_not_cancel_another_request():
    session = _StubSession()
    session._block_until_stop = True
    with TestClient(create_app(cast(DrowseSession, session))) as client:
        with client.websocket_connect(STREAM) as ws:
            ws.send_json({"type": "generate", "request_id": "active", "input": "hi", "stateless": False, "n": 2})
            while ws.receive_json()["type"] != "token":
                pass
            ws.send_json({"type": "stop", "request_id": "old-job"})
            time.sleep(0.03)
            assert not session._stop_event.is_set()
            ws.send_json({"type": "stop", "request_id": "active"})
            while True:
                frame = ws.receive_json()
                if frame["type"] == "request_complete":
                    break
    assert frame["state"] == "cancelled"
    assert frame["completed_siblings"] == 1
