"""Security checks for model code and detached artifact workers."""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from drowse.model import load as model_module


@pytest.mark.parametrize("trusted", [False, True])
def test_model_configuration_requires_explicit_code_trust(tmp_path: Path, trusted: bool) -> None:
    marker = tmp_path / "configuration-executed"
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 2, "intermediate_size": 16,
        "auto_map": {"AutoConfig": "configuration_fixture.FixtureConfig"},
    }))
    (tmp_path / "configuration_fixture.py").write_text(
        "from pathlib import Path\n"
        "from transformers import LlamaConfig\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "class FixtureConfig(LlamaConfig):\n    pass\n"
    )
    plan = model_module._resolve_load_plan(str(tmp_path), quantize=None, device="cpu", dtype=None, trust_remote_code=trusted)
    assert plan.config.model_type == "llama"
    assert marker.exists() is trusted
    assert plan.tokenizer_kwargs["trust_remote_code"] is trusted


@pytest.mark.parametrize(("explicit", "environment", "trusted"), [(None, "", False), (None, "true", True), (False, "1", False), (True, "", True), (None, "false", False)])
def test_tokenizer_code_trust_is_explicit_and_overrides_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool | None, environment: str, trusted: bool) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    marker = tmp_path / "tokenizer-executed"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2}))
    Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")).save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]",
        "auto_map": {"AutoTokenizer": [None, "tokenization_fixture.FixtureTokenizer"]},
    }))
    (tmp_path / "tokenization_fixture.py").write_text(
        "from pathlib import Path\n"
        "from transformers import PreTrainedTokenizerFast\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "class FixtureTokenizer(PreTrainedTokenizerFast):\n    pass\n"
    )
    monkeypatch.setattr(model_module, "_materialize_model", lambda plan: object())
    monkeypatch.setattr(model_module, "_finalize_model", lambda model, *args, **kwargs: model)
    monkeypatch.setenv("DROWSE_TRUST_REMOTE_CODE", environment)
    _, tokenizer = model_module.load_model(str(tmp_path), device="cpu", trust_remote_code=explicit)
    assert tokenizer.encode("hello") == [1]
    assert marker.exists() is trusted


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions", "/api/chat", "/api/generate"])
@pytest.mark.parametrize("stream", [False, True])
def test_http_inference_leaves_event_loop_responsive(path: str, stream: bool) -> None:
    import httpx
    from drowse.generation.results import GenerationResult
    from drowse.server import create_app
    from tests.test_server_loom import _StubSession

    async def scenario() -> None:
        entered, release = threading.Event(), threading.Event()
        session: Any = _StubSession()
        result = GenerationResult(text="ok", tokens=[1], token_count=1, tok_per_sec=1, elapsed=0.1, finish_reason="stop")

        def block() -> None:
            entered.set()
            release.wait(2)

        def generate(*args: Any, **kwargs: Any) -> Any:
            block()
            return SimpleNamespace(first=result)

        class Stream:
            def __init__(self) -> None:
                self.result = result

            def __iter__(self) -> Any:
                return self

            def __next__(self) -> Any:
                block()
                raise StopIteration

            def close(self) -> None:
                pass

        iterator = Stream()
        session.generate = generate
        session.generate_stream = lambda *args, **kwargs: iterator
        app = create_app(session)
        # Release from outside the event loop so the pre-fix failure cannot hang pytest.
        timer = threading.Timer(0.3, release.set)
        timer.start()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://localhost") as client:
            task = asyncio.create_task(client.post(path, json={
                "model": session.model_id, "stream": stream, "prompt": "hi",
                "messages": [{"role": "user", "content": "hi"}],
            }))
            try:
                assert await asyncio.to_thread(entered.wait, 2)
                assert not release.is_set(), "inference blocked the ASGI event loop"
                assert (await client.get("/v1/models")).status_code == 200
            finally:
                release.set()
                timer.cancel()
                response = await task
                assert response.status_code == 200, response.text
    asyncio.run(scenario())


def test_queued_openai_stream_does_not_start_model_before_owning_lock() -> None:
    import httpx
    from drowse.server import create_app
    from tests.test_server_loom import _StubSession

    async def scenario() -> None:
        session: Any = _StubSession()
        starts: list[bool] = []
        session.generate_stream = lambda *args, **kwargs: starts.append(True)
        app = create_app(session)
        lock = app.state.session_lock
        await lock.acquire()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://localhost") as client:
            task = asyncio.create_task(client.post("/v1/chat/completions", json={
                "model": session.model_id, "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }))
            try:
                await asyncio.sleep(0.03)
                assert starts == []
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                lock.release()
    asyncio.run(scenario())


@pytest.mark.parametrize("value", ["true", 1, [], {}])
def test_model_code_trust_rejects_ambiguous_non_booleans(value: Any) -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        model_module.load_model("unused/model", trust_remote_code=value)


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/api/chat"])
def test_http_send_failure_joins_stream_close_without_blocking_loop(path: str) -> None:
    from starlette.requests import ClientDisconnect
    from drowse.server import create_app
    from tests.test_server_loom import _StubSession

    async def scenario() -> None:
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        session: Any = _StubSession()

        class Stream:
            result = None

            def __iter__(self) -> Any:
                return self

            def __next__(self) -> Any:
                return SimpleNamespace(text="content", thinking=False, probe_readings=None)

            def close(self) -> None:
                entered.set()
                assert release.wait(3)
                closed.set()

        session.generate_stream = lambda *args, **kwargs: Stream()
        app = create_app(session)
        scope: Any = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
            "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
            "root_path": "", "query_string": b"", "headers": [(b"host", b"localhost"), (b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 80),
        }
        incoming: asyncio.Queue[Any] = asyncio.Queue()
        incoming.put_nowait({"type": "http.request", "body": json.dumps({
            "model": session.model_id, "stream": True, "messages": [{"role": "user", "content": "hi"}],
        }).encode()})

        async def send(message: Any) -> None:
            if message["type"] == "http.response.body" and b"content" in message["body"]:
                raise OSError("disconnected")

        task = asyncio.create_task(app(scope, incoming.get, send))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            assert app.state.session_lock.locked() and not task.done()
            assert not closed.is_set()
        finally:
            release.set()
            with pytest.raises(ClientDisconnect):
                await task
        assert closed.is_set() and not app.state.session_lock.locked()
    asyncio.run(scenario())


