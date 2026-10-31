"""Direct construction of a :class:`~drowse.session.engine.GenerationEngine`.

:func:`engine_over` builds the engine over ``host`` (the session surface its
call composition reads, e.g. ``tree`` and ``config``) with an inert compiled
runtime, no steering manager, no read plane and a no-op teardown release.
Prefer :class:`~tests.support.sessions.TinySessionFactory` when the test
needs a whole generation.
"""

from __future__ import annotations

import threading
from typing import Any, cast

from drowse.generation.compile_runtime import CompiledRuntime
from drowse.session.engine import GenerationEngine

__all__ = ["engine_over"]


def engine_over(
    host: Any,
    *,
    lock: Any = None,
    runtime: CompiledRuntime | None = None,
    steering: Any = None,
    reads: Any = None,
) -> GenerationEngine:
    """A generation engine over ``host``."""
    return GenerationEngine(
        cast(Any, host),
        lock=lock if lock is not None else threading.RLock(),
        runtime=runtime if runtime is not None else CompiledRuntime(),
        steering=cast(Any, steering),
        reads=cast(Any, reads),
        release=lambda: None,
    )
