from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

import pytest


DEFAULT_GPU_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

_T = TypeVar("_T")


def gpu_model_id() -> str:
    return os.environ.get("DROWSE_TEST_MODEL", DEFAULT_GPU_MODEL_ID)


def load_or_skip_inaccessible(loader: Callable[[], _T], model_id: str) -> _T:
    try:
        return loader()
    except Exception as exc:
        if _is_access_error(exc):
            pytest.skip(f"model access is unavailable for {model_id}: {exc}")
        raise


def _is_access_error(exc: BaseException) -> bool:
    for current in _exception_chain(exc):
        name = type(current).__name__.lower()
        message = str(current).lower()
        if name == "gatedrepoerror":
            return True
        if "gated repo" in message or "gated repository" in message:
            return True
        if "403 client error" in message and (
            "access" in message or "authorized" in message or "gated" in message
        ):
            return True
        if "401 client error" in message and (
            "authentication" in message or "authorized" in message or "token" in message
        ):
            return True
    return False


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain
