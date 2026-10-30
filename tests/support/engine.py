"""Give a hand-built ``DrowseSession`` stub the real generation collaborators.

Prefer :class:`~tests.support.sessions.TinySessionFactory` for new tests; this
keeps the remaining narrow session stubs (built with ``DrowseSession.__new__``
or ``MagicMock(spec=DrowseSession)``) wired the way ``__init__`` wires a real
session, so they read the engine's phase, prefix cache and compiled routing
instead of session attributes that no longer exist.
"""

from __future__ import annotations

import threading
from typing import Any

from drowse.generation.compile_runtime import CompiledRuntime
from drowse.session.engine import GenerationEngine

__all__ = ["attach_engine", "seed_prefix_cache"]


def attach_engine(stub: Any, *, runtime: CompiledRuntime | None = None) -> GenerationEngine:
    """Attach ``stub._runtime`` and ``stub._engine`` and return the engine.

    The engine shares the stub's ``_gen_lock`` when it has one (a fresh
    reentrant lock otherwise) and its ``_steering`` / ``_read_plane``; its
    teardown release is a no-op.
    """
    lock = stub.__dict__.get("_gen_lock") if hasattr(stub, "__dict__") else None
    if lock is None:
        lock = threading.RLock()
        stub._gen_lock = lock
    stub._runtime = runtime if runtime is not None else CompiledRuntime()
    stub._engine = GenerationEngine(
        stub,
        lock=lock,
        runtime=stub._runtime,
        steering=stub.__dict__.get("_steering"),
        reads=stub.__dict__.get("_read_plane"),
        release=lambda: None,
    )
    return stub._engine


def seed_prefix_cache(engine: GenerationEngine) -> None:
    """Fill the engine's prefix-cache slot (for invalidation tests)."""
    from types import SimpleNamespace

    import torch

    engine.prefix_cache.build(
        lambda **_kwargs: SimpleNamespace(past_key_values=object()),
        torch.tensor([[1, 2]]),
    )
    assert engine.prefix_cache
