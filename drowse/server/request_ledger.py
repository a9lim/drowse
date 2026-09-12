"""Bounded request receipts retained across client connection lifetimes."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
import json
from typing import Any


class RequestLedger:
    def __init__(self, retention: int = 100) -> None:
        self.records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.retention = retention

    def claim(self, request_id: str, payload: dict[str, Any]) -> str:
        fingerprint = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        prior = self.records.get(request_id)
        if prior is not None:
            return "existing" if prior["fingerprint"] == fingerprint else "conflict"
        self.records[request_id] = {
            "request_id": request_id, "state": "running", "results": [],
            "fingerprint": fingerprint,
        }
        self._prune()
        return "new"

    def observe(self, request_id: str, frame: dict[str, Any]) -> None:
        record = self.records[request_id]
        if frame["type"] == "done":
            record["results"] = [row for row in record["results"] if row["sibling_index"] != frame["sibling_index"]]
            record["results"].append(deepcopy(frame))
        elif frame["type"] == "error":
            record["error"] = deepcopy(frame)
        elif frame["type"] == "request_complete":
            record["state"] = frame["state"]
        self._prune()

    def interrupt(self, request_id: str) -> None:
        if self.records[request_id]["state"] == "running":
            self.records[request_id]["state"] = "interrupted"
        self._prune()

    def settle_disconnected(self, request_id: str, expected: int) -> None:
        record = self.records[request_id]
        if record["state"] != "running":
            return
        rows = record["results"]
        record["state"] = "failed" if "error" in record else (
            "completed" if len(rows) == expected and all(row["result"].get("finish_reason") != "cancelled" for row in rows)
            else "cancelled"
        )
        self._prune()

    def get(self, request_id: str) -> dict[str, Any]:
        record = self.records.get(request_id)
        if record is None:
            return {"request_id": request_id, "state": "unknown", "results": []}
        return deepcopy({key: value for key, value in record.items() if key != "fingerprint"})

    def _prune(self) -> None:
        finished = [key for key, value in self.records.items() if value["state"] != "running"]
        for key in finished[:max(0, len(finished) - self.retention)]:
            del self.records[key]
        size = len(json.dumps(self.records))
        for key in finished:
            if size <= 16 * 1024 * 1024:
                break
            if key in self.records:
                size -= len(json.dumps(self.records[key]))
                del self.records[key]
