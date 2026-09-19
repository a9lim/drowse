from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

from drowse.core.session import DrowseSession

from drowse.server.request_ledger import RequestLedger


def test_active_receipts_are_bounded_without_losing_admitted_work() -> None:
    ledger = RequestLedger(max_active=2)
    assert ledger.claim("one", {}) == "new"
    assert ledger.claim("two", {}) == "new"
    assert ledger.claim("three", {}) == "full"
    assert ledger.claim("one", {}) == "existing"
    assert ledger.claim("one", {"changed": True}) == "conflict"
    assert ledger.get("three")["state"] == "unknown"
    ledger.interrupt("one")
    assert ledger.claim("three", {}) == "new"
    assert ledger.get("two")["state"] == "running"


def test_receipt_storage_does_not_reserialize_unrelated_results(monkeypatch: Any) -> None:
    ledger = RequestLedger(retention=2)
    for index in range(3):
        request_id = str(index)
        ledger.claim(request_id, {})
        ledger.observe(request_id, {"type": "done", "sibling_index": 0, "result": {"text": "x" * 1000}})
        ledger.observe(request_id, {"type": "request_complete", "state": "completed"})
    assert ledger.get("0")["state"] == "unknown"
    assert sum(ledger._sizes.values()) == len(json.dumps(ledger.records))
    original = json.dumps
    serialized = []

    def track(value: Any, **kwargs: Any) -> str:
        serialized.append(value)
        return original(value, **kwargs)

    monkeypatch.setattr("drowse.server.request_ledger.json.dumps", track)
    ledger.claim("new", {"prompt": "short"})
    assert not any(value is ledger.records or value is ledger.records["1"] or value is ledger.records["2"]
                   for value in serialized)


def test_receipt_admission_and_snapshots_are_safe_across_http_worker_threads() -> None:
    ledger = RequestLedger()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: ledger.claim("shared", {"prompt": "test"}), range(64)))
    assert results.count("new") == 1
    assert results.count("existing") == 63
    ledger.observe("shared", {"type": "done", "sibling_index": 0, "result": {"text": "preserved"}})
    snapshot = ledger.get("shared")
    snapshot["results"][0]["result"]["text"] = "changed"
    assert ledger.get("shared")["results"][0]["result"]["text"] == "preserved"


def test_full_websocket_receipt_ledger_rejects_before_dispatch(monkeypatch: Any) -> None:
    from drowse.server import ws_stream
    from drowse.server.ws_models import WSGenerateMessage

    async def scenario() -> None:
        ledger = RequestLedger(max_active=1)
        ledger.claim("active", {})
        frames: list[Any] = []

        async def send(frame: Any) -> None:
            frames.append(frame)

        async def unexpected(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("full ledger must reject before dispatch")

        monkeypatch.setattr(ws_stream, "_ws_dispatch_generate", unexpected)
        await ws_stream._ws_handle_generate(cast(DrowseSession, None), WSGenerateMessage(type="generate", request_id="blocked"),
                                           None, ws_stream._InboundBuffer(), send, unexpected, ledger)
        assert frames[0]["status"] == 503 and frames[0]["code"] == "REQUEST_QUEUE_FULL"
        assert frames[1]["state"] == "failed" and frames[1]["completed_siblings"] == 0
        assert ledger.get("active")["state"] == "running"
        assert ledger.get("blocked")["state"] == "unknown"

    asyncio.run(scenario())
