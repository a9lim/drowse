from __future__ import annotations

import asyncio
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from drowse.server.operation_ledger import OperationLedger, register_operation_routes

PATH = "/drowse/v1/manifolds/local/pirate/fit"


def test_receipt_lifecycle_copies_and_request_privacy() -> None:
    ledger = OperationLedger()
    receipt = ledger.start("fit:1", PATH, {"seed": 3, "private_input": "not retained"}, "model/one")
    assert receipt["state"] == "queued"
    assert "not retained" not in json.dumps(receipt)
    receipt["progress"].append("external mutation")
    assert ledger.get("fit:1")["progress"] == []
    assert ledger.running("fit:1")["started_at"] is not None
    assert ledger.progress("fit:1", "fitting layers")["progress"] == ["fitting layers"]
    value = {"namespace": "local", "name": "pirate", "scores": [0.3]}
    completed = ledger.complete("fit:1", value)
    value["scores"].append(9)
    assert completed["state"] == "completed"
    assert completed["result"]["scores"] == [0.3]
    assert completed["finished_at"] >= completed["started_at"] >= completed["created_at"]
    assert completed["result_sha256"] == hashlib.sha256(json.dumps(
        completed["result"], sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    assert ledger.progress("fit:1", "late") == completed
    assert ledger.fail("fit:1", {"message": "late failure"}) == completed
    assert ledger.complete("fit:1", {"different": True}) == completed


def test_duplicate_and_conflicting_receipts_never_reexecute() -> None:
    ledger = OperationLedger()
    ledger.start("same", PATH, {"b": 2, "a": {"x": 1}}, "model")
    with pytest.raises(HTTPException, match="read its status") as duplicate:
        ledger.start("same", PATH, {"a": {"x": 1}, "b": 2}, "model")
    assert duplicate.value.status_code == 409
    for path, body, model in [
        (PATH, {"b": 3}, "model"),
        ("/drowse/v1/other", {"a": {"x": 1}, "b": 2}, "model"),
        (PATH, {"a": {"x": 1}, "b": 2}, "other-model"),
    ]:
        with pytest.raises(HTTPException, match="different parameters") as conflict:
            ledger.start("same", path, body, model)
        assert conflict.value.status_code == 409
    ledger.complete("same", {})
    with pytest.raises(HTTPException, match="read its status"):
        ledger.start("same", PATH, {"a": {"x": 1}, "b": 2}, "model")


def test_simultaneous_admission_has_one_owner() -> None:
    ledger = OperationLedger()

    def admit(_: int) -> int:
        try:
            ledger.start("single-owner", PATH, {}, None)
            return 200
        except HTTPException as error:
            return error.status_code

    with ThreadPoolExecutor(max_workers=8) as executor:
        statuses = list(executor.map(admit, range(24)))
    assert statuses.count(200) == 1
    assert statuses.count(409) == 23


def test_active_receipts_are_not_evicted_and_terminal_eviction_is_explicit() -> None:
    ledger = OperationLedger(max_entries=2)
    ledger.start("active", PATH, {}, None)
    ledger.running("active")
    ledger.start("terminal", PATH, {}, None)
    with pytest.raises(HTTPException) as full:
        ledger.start("blocked", PATH, {}, None)
    assert full.value.status_code == 503
    ledger.complete("terminal", {"ok": True})
    ledger.start("replacement", PATH, {}, None)
    assert ledger.get("active")["state"] == "running"
    with pytest.raises(HTTPException, match="completion cannot be inferred") as expired:
        ledger.get("terminal")
    assert expired.value.status_code == 404


def test_progress_failure_and_interruption_are_bounded() -> None:
    ledger = OperationLedger()
    ledger.start("progress", PATH, {}, None)
    for index in range(100):
        ledger.progress("progress", f"{index}:" + "x" * 2000)
    receipt = ledger.get("progress")
    assert len(receipt["progress"]) == 64
    assert receipt["progress_seq"] == 100
    assert all(len(message) <= 1024 for message in receipt["progress"])
    assert receipt["progress"][0].startswith("36:")
    failed = ledger.fail("progress", {"message": "safe failure", "code": "SAFE", "status": 409, "credential": "secret"})
    assert failed["state"] == "failed"
    assert failed["error"] == {"message": "safe failure", "code": "SAFE", "status": 409}
    ledger.start("interrupted", PATH, {}, None)
    assert ledger.interrupt("interrupted", "Worker stopped")["state"] == "interrupted"


def test_large_results_preserve_completion_digest_and_aggregate_bound() -> None:
    ledger = OperationLedger(max_result_bytes=100, max_total_result_bytes=150)
    for request_id in ["first", "second", "large"]:
        ledger.start(request_id, PATH, {}, None)
    first = ledger.complete("first", {"text": "a" * 70})
    assert not first["result_truncated"]
    ledger.complete("second", {"text": "b" * 70})
    evicted_payload = ledger.get("first")
    assert evicted_payload["state"] == "completed"
    assert evicted_payload["result"] is None
    assert evicted_payload["result_truncated"]
    assert evicted_payload["result_sha256"] == first["result_sha256"]
    large = ledger.complete("large", {"text": "c" * 1000})
    assert large["state"] == "completed"
    assert large["result"] is None and large["result_truncated"]
    assert large["result_bytes"] > 100
    assert len(large["result_sha256"]) == 64
    assert sum(record["result_bytes"] for record in [ledger.get("first"), ledger.get("second"), large]
               if not record["result_truncated"]) <= 150


@pytest.mark.parametrize("request_id", ["", "x" * 129, "a/b", "a b", "a\n", "a?key=secret"])
def test_invalid_ids_are_rejected(request_id: str) -> None:
    with pytest.raises(HTTPException) as error:
        OperationLedger().start(request_id, PATH, {}, None)
    assert error.value.status_code == 400


def test_route_inherits_auth_and_reads_current_app_ledger() -> None:
    def require_key(request: Request) -> None:
        if request.headers.get("authorization") != "Bearer test-key":
            raise HTTPException(401, "Unauthorized")

    app = FastAPI(dependencies=[Depends(require_key)])
    register_operation_routes(app)
    app.state.operation_ledger.start("owned", PATH, {}, "model")
    with TestClient(app) as client:
        assert client.get("/drowse/v1/operations/owned").status_code == 401
        assert client.get("/drowse/v1/operations/owned", headers={"Authorization": "Bearer wrong"}).status_code == 401
        response = client.get("/drowse/v1/operations/owned", headers={"Authorization": "Bearer test-key"})
        assert response.status_code == 200
        assert response.json()["request_id"] == "owned"
        app.state.operation_ledger = OperationLedger()
        assert client.get("/drowse/v1/operations/owned", headers={"Authorization": "Bearer test-key"}).status_code == 404


def test_shutdown_joins_operation_tasks_without_cancelling_them() -> None:
    async def run() -> None:
        app = FastAPI()
        register_operation_routes(app)
        finished = False

        async def worker() -> None:
            nonlocal finished
            await asyncio.sleep(0.01)
            finished = True

        task = asyncio.create_task(worker())
        app.state.operation_tasks.add(task)
        for shutdown in app.router.on_shutdown:
            await shutdown()
        assert finished
        assert not task.cancelled()

    asyncio.run(run())
