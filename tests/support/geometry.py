"""Geometry test support: a real :class:`GeometryInstrument` over a stub.

:func:`stub_geometry_instrument` wires the instrument's explicit dependencies
from a hand-built session stub's attributes, looked up per call (``_monitor``,
``_model_exclusive``, ``_resolve_probe_manifold``, ``whitener``), so a test can
assign them before or after building the instrument.  Prefer ``tiny_session``
for new tests.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from drowse.readout.geometry import GeometryInstrument
from tests.support.replay import unwired_replay

__all__ = ["stub_geometry_instrument"]


class _LateMonitor:
    """Forwards every Monitor call to ``stub._monitor``, looked up per call."""

    def __init__(self, stub: Any) -> None:
        self._stub = stub

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stub._monitor, name)


def stub_geometry_instrument(
    stub: Any, *, replay: Any = None, monitor: Any = None,
) -> GeometryInstrument:
    """A geometry instrument reading ``stub``'s attributes at call time.

    ``monitor`` pins a Monitor instead of forwarding to ``stub._monitor``.
    The exclusive section is the stub's ``_model_exclusive`` when it has one,
    else a no-op.
    """

    def exclusive(*args: Any, **kwargs: Any) -> Any:
        guard = getattr(stub, "_model_exclusive", None)
        return nullcontext() if guard is None else guard(*args, **kwargs)

    return GeometryInstrument(
        monitor=_LateMonitor(stub) if monitor is None else monitor,  # type: ignore[arg-type]
        exclusive=exclusive,
        resolve_manifold=lambda selector: stub._resolve_probe_manifold(selector),
        ensure_whitener=lambda: getattr(stub, "whitener", None),
        replay=unwired_replay if replay is None else replay,
    )
