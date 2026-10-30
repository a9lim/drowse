"""A real :class:`~drowse.session.read_plane.ReadPlane` over test fakes.

:func:`make_read_plane` builds the read plane with explicit fakes for the
capture, the Monitor and the three instruments (only the members a test
exercises need to exist), :func:`finalize` runs its finalize reads through
:func:`~drowse.generation.finalize.finalize_generation` the way the engine
does, and :func:`gate_refs` turns a steering expression into the active gate
references the composer would hand the read plane.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

from drowse.events import EventBus
from drowse.generation.decode import GenerationState
from drowse.generation.finalize import finalize_generation
from drowse.generation.results import GenerationResult

__all__ = ["fake_readout_family", "finalize", "gate_refs", "make_read_plane"]


def fake_readout_family(
    names: tuple[str, ...] = (),
    *,
    live: Any = None,
    score_aggregate: Callable[..., dict[str, Any]] | None = None,
    gate_scalars: Callable[..., dict[str, float]] | None = None,
) -> Any:
    """A J-lens / SAE instrument fake: names, live state, the finalize
    aggregate, the bound run's gate scalars and an accept-all gate check."""
    return SimpleNamespace(
        names=list(names),
        live=live,
        score_aggregate=score_aggregate or (lambda _ids, _pooled: {}),
        current_run=SimpleNamespace(
            gate_scalars=gate_scalars or (lambda _step, _hidden, _keys: {}),
        ),
        validate_gate=lambda _ref: None,
    )


def make_read_plane(
    *,
    capture: Any = None,
    monitor: Any = None,
    geometry: Any = None,
    lens: Any = None,
    sae: Any = None,
    layers: Any = (),
    tokenizer: Any = None,
    gate_refs: Callable[[], list[Any]] = lambda: [],
) -> Any:
    """A ``ReadPlane`` over fakes.  The default geometry instrument routes
    its run's reads to ``monitor`` (as the real idle/bound run does)."""
    from drowse.session.read_plane import ReadPlane

    monitor = monitor if monitor is not None else SimpleNamespace(probe_names=[])
    if geometry is None:
        geometry = SimpleNamespace(
            names=list(getattr(monitor, "probe_names", [])),
            manifolds=lambda: dict(getattr(monitor, "manifolds", {})),
            current_run=SimpleNamespace(
                observe=lambda _step, hidden: monitor.score_single_token(hidden),
                observe_aggregate=lambda pooled: monitor.score_aggregate(pooled),
            ),
        )
    return ReadPlane(
        capture=cast(Any, capture if capture is not None else SimpleNamespace()),
        monitor=cast(Any, monitor),
        geometry=cast(Any, geometry),
        lens=cast(Any, lens if lens is not None else fake_readout_family()),
        sae=cast(Any, sae if sae is not None else fake_readout_family()),
        layers=layers,
        tokenizer=tokenizer,
        gate_refs=gate_refs,
    )


def finalize(
    plane: Any,
    generated_ids: list[int],
    *,
    state: GenerationState,
    tokenizer: Any,
    elapsed: float = 1.0,
    return_hidden: bool = False,
    return_probe_readings: bool = True,
    **kwargs: Any,
) -> tuple[GenerationResult, dict[str, list[float]] | None]:
    """Score ``plane``'s capture and finalize a stateless result, as the
    engine does; returns the result and the recorded per-token scores."""
    recorded: dict[str, Any] = {}
    reads = plane.finalize(
        generated_ids,
        state=state,
        return_hidden=return_hidden,
        return_probe_readings=return_probe_readings,
    )
    kwargs.setdefault("stateless", True)
    result = finalize_generation(
        generated_ids, elapsed, {},
        state=state,
        reads=reads,
        tokenizer=tokenizer,
        events=EventBus(),
        record=lambda _result, per_token: recorded.update(per_token=per_token),
        **kwargs,
    )
    return result, recorded["per_token"]


def gate_refs(expr: str) -> Callable[[], list[Any]]:
    """The composer's ``gate_refs`` for the steering expression ``expr``."""
    from drowse.readout.types import parse_gate_ref
    from drowse.steering.grammar import parse_expr
    from drowse.steering.terms import entry_trigger

    refs = []
    for entry in parse_expr(expr).alphas.values():
        gate = entry_trigger(cast(Any, entry)).gate
        if gate is not None:
            refs.append((gate.probe, parse_gate_ref(gate.probe)))
    return lambda: list(refs)
