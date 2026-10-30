from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from typing import Any

from fastapi import FastAPI, HTTPException

from drowse.server.streaming import finish_worker

_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_TERMINAL = frozenset({"completed", "failed", "interrupted"})
MAX_PROGRESS_MESSAGES = 64
MAX_PROGRESS_LENGTH = 1024


class OperationLedger:
    def __init__(
        self,
        *,
        max_entries: int = 128,
        max_result_bytes: int = 4 * 1024 * 1024,
        max_total_result_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if max_entries < 1 or max_result_bytes < 0 or max_total_result_bytes < 0:
            raise ValueError("Operation ledger limits must be nonnegative with at least one entry")
        self.max_entries = max_entries
        self.max_result_bytes = max_result_bytes
        self.max_total_result_bytes = max_total_result_bytes
        self._records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._retained_bytes: dict[str, int] = {}
        self._lock = threading.RLock()

    def start(self, request_id: str, path: str, body: Any, model_id: str | None) -> dict[str, Any]:
        self._validate_id(request_id)
        if not path.startswith("/drowse/v1/") or len(path) > 2048 or "?" in path or "#" in path:
            raise HTTPException(400, "Operation path must be a native API path without query parameters")
        if model_id is not None and len(model_id) > 512:
            raise HTTPException(400, "Operation model identity is invalid")
        try:
            fingerprint = hashlib.sha256(json.dumps(
                [path, model_id, body], sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ).encode("utf-8")).hexdigest()
        except (TypeError, ValueError):
            raise HTTPException(400, "Operation parameters must be valid JSON") from None
        with self._lock:
            existing = self._records.get(request_id)
            if existing is not None:
                if existing["request_fingerprint"] != fingerprint:
                    raise HTTPException(409, "This request ID belongs to an operation with different parameters")
                raise HTTPException(409, "This request ID already has an operation receipt; read its status instead of resubmitting")
            if len(self._records) >= self.max_entries:
                evict = next((key for key, record in self._records.items() if record["state"] in _TERMINAL), None)
                if evict is None:
                    raise HTTPException(503, "The operation ledger is full of active requests; wait for one to finish")
                del self._records[evict]
                self._retained_bytes.pop(evict, None)
            now = time.time()
            record: dict[str, Any] = {
                "id": request_id, "request_id": request_id, "path": path,
                "model_id": model_id, "request_fingerprint": fingerprint,
                "state": "queued", "created_at": now, "started_at": None,
                "updated_at": now, "finished_at": None, "progress": [], "progress_seq": 0,
                "result": None, "error": None, "result_bytes": 0,
                "result_sha256": None, "result_truncated": False,
            }
            self._records[request_id] = record
            return copy.deepcopy(record)

    def running(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._require(request_id)
            if record["state"] not in _TERMINAL:
                now = time.time()
                record["state"] = "running"
                record["started_at"] = record["started_at"] or now
                record["updated_at"] = now
            return copy.deepcopy(record)

    def progress(self, request_id: str, message: str) -> dict[str, Any]:
        with self._lock:
            record = self._require(request_id)
            if record["state"] not in _TERMINAL:
                now = time.time()
                record["state"] = "running"
                record["started_at"] = record["started_at"] or now
                record["updated_at"] = now
                record["progress"].append(message[:MAX_PROGRESS_LENGTH])
                del record["progress"][:-MAX_PROGRESS_MESSAGES]
                record["progress_seq"] += 1
            return copy.deepcopy(record)

    def complete(self, request_id: str, result: Any) -> dict[str, Any]:
        chunks: list[bytes] = []
        size = 0
        digest = hashlib.sha256()
        encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        for chunk in encoder.iterencode(result):
            encoded = chunk.encode("utf-8")
            digest.update(encoded)
            size += len(encoded)
            if size <= self.max_result_bytes:
                chunks.append(encoded)
            else:
                chunks.clear()
        retained = size <= min(self.max_result_bytes, self.max_total_result_bytes)
        payload = json.loads(b"".join(chunks)) if retained else None
        with self._lock:
            record = self._require(request_id)
            if record["state"] in _TERMINAL:
                return copy.deepcopy(record)
            if retained:
                for key, old in self._records.items():
                    if sum(self._retained_bytes.values()) + size <= self.max_total_result_bytes:
                        break
                    if self._retained_bytes.pop(key, 0):
                        old["result"] = None
                        old["result_truncated"] = True
                self._retained_bytes[request_id] = size
            record.update(
                state="completed", result=payload, result_bytes=size,
                result_sha256=digest.hexdigest(), result_truncated=not retained,
                error=None, finished_at=time.time(),
            )
            record["updated_at"] = record["finished_at"]
            return copy.deepcopy(record)

    def fail(self, request_id: str, error: dict[str, Any]) -> dict[str, Any]:
        return self._finish_error(request_id, "failed", error)

    def interrupt(self, request_id: str, message: str) -> dict[str, Any]:
        return self._finish_error(request_id, "interrupted", {"code": "INTERRUPTED", "message": message})

    def _finish_error(self, request_id: str, state: str, error: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            record = self._require(request_id)
            if record["state"] not in _TERMINAL:
                record.update(state=state, error={
                    "code": str(error.get("code", "OPERATION_FAILED"))[:128],
                    "message": str(error.get("message", "Operation failed"))[:2048],
                }, finished_at=time.time())
                status = error.get("status")
                if type(status) is int and 400 <= status <= 599:
                    record["error"]["status"] = status
                record["updated_at"] = record["finished_at"]
            return copy.deepcopy(record)

    def get(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._require(request_id))

    @staticmethod
    def _validate_id(request_id: str) -> None:
        if not _REQUEST_ID.fullmatch(request_id):
            raise HTTPException(400, "Operation request ID must contain 1 to 128 letters, digits, dots, underscores, colons or hyphens")

    def _require(self, request_id: str) -> dict[str, Any]:
        self._validate_id(request_id)
        record = self._records.get(request_id)
        if record is None:
            raise HTTPException(404, "Operation receipt is unavailable or has expired; completion cannot be inferred")
        return record


def register_operation_routes(app: FastAPI) -> None:
    if not hasattr(app.state, "operation_ledger"):
        app.state.operation_ledger = OperationLedger()
    if not hasattr(app.state, "operation_tasks"):
        app.state.operation_tasks = set()

    @app.get("/drowse/v1/operations/{request_id}")
    async def operation_status(request_id: str) -> dict[str, Any]:
        return app.state.operation_ledger.get(request_id)

    async def finish_operations() -> None:
        tasks = list(app.state.operation_tasks)
        if tasks:
            await finish_worker(asyncio.gather(*tasks, return_exceptions=True))

    app.router.on_shutdown.append(finish_operations)
