"""A real :class:`~drowse.generation.read_plane.ReadPlane` and its instruments.

:class:`ReadRig` constructs the read plane, the three read instruments and
the probe roster with the explicit dependencies a session wires, over a
:class:`CaptureSlot` whose target a test points at a fake capture (or swaps
mid-test).  Any instrument (or the Monitor) can be replaced by a fake.
:func:`finalize` runs the plane's finalize reads through
:func:`~drowse.generation.finalize.finalize_generation` the way the engine
does, and :func:`gate_refs` turns a steering expression into the active gate
references the composer would hand the read plane.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

from drowse.events import EventBus
from drowse.generation.decode import GenerationState
from drowse.generation.finalize import finalize_generation
from drowse.generation.results import GenerationResult

__all__ = [
    "CaptureSlot",
    "ReadRig",
    "fake_readout_family",
    "finalize",
    "gate_refs",
    "geometry_instrument",
]


class CaptureSlot:
    """The capture a read plane and its instruments are built over.

    Every attribute read forwards to ``target`` at call time, so a test can
    point the slot at a fake capture after construction (or swap it).
    """

    def __init__(self, target: Any = None) -> None:
        self.target = target if target is not None else SimpleNamespace()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)


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


def geometry_instrument(
    monitor: Any, *, replay: Any = None, lock: Any = None,
) -> Any:
    """A real geometry instrument over ``monitor``; ``lock`` is the
    generation lock its exclusive section takes.  It resolves no manifolds
    and builds no whitener: attach probes on ``monitor`` directly."""
    from drowse.readout.geometry import GeometryInstrument
    from tests.support.replay import unwired_replay

    guard = lock if lock is not None else threading.RLock()

    @contextmanager
    def exclusive(_message: str, **_kwargs: Any) -> Iterator[None]:
        with guard:
            yield

    def resolve_manifold(selector: str) -> Any:
        raise AssertionError(
            f"read rig geometry resolves no manifolds (asked for {selector!r}); "
            "attach probes on the monitor"
        )

    return GeometryInstrument(
        monitor=monitor,
        exclusive=exclusive,
        resolve_manifold=resolve_manifold,
        ensure_whitener=lambda: None,
        replay=unwired_replay if replay is None else replay,
    )


class ReadRig:
    """One read plane over real (or substituted) instruments.

    - ``capture``: the :class:`CaptureSlot` every reader shares.
    - ``monitor``: the geometry engine (default: an empty ``Monitor``).
    - ``geometry`` / ``lens`` / ``sae``: the instruments (real by default;
      pass a fake to replace one).
    - ``roster``: the probe roster; ``roster_changes`` counts its change hook.
    - ``plane``: the :class:`ReadPlane` under test.
    - ``lock``: the generation lock the geometry exclusive section takes.
    """

    def __init__(
        self,
        *,
        capture: Any = None,
        monitor: Any = None,
        geometry: Any = None,
        lens: Any = None,
        sae: Any = None,
        layers: Any = (),
        tokenizer: Any = None,
        gate_refs: Callable[[], list[Any]] = lambda: [],
        replay: Any = None,
    ) -> None:
        from drowse.readout.monitor import Monitor
        from drowse.readout.roster import ProbeRoster
        from drowse.generation.read_plane import ReadPlane
        from tests.support.lens import lens_instrument
        from tests.support.sae import sae_instrument

        self.capture = CaptureSlot(capture)
        self.lock = threading.RLock()
        self.monitor = monitor if monitor is not None else Monitor()
        self.geometry = (
            geometry if geometry is not None
            else geometry_instrument(self.monitor, replay=replay, lock=self.lock)
        )
        self.lens = (
            lens if lens is not None
            else lens_instrument(capture=self.capture, n_layers=len(layers), replay=replay)
        )
        self.sae = (
            sae if sae is not None
            else sae_instrument(
                SimpleNamespace(model_id="stub/model", layers=layers),
                capture=self.capture, replay=replay,
            )
        )
        self.roster_changes = 0
        self.roster = ProbeRoster(
            geometry=self.geometry, lens=self.lens, sae=self.sae,
            on_change=self._roster_changed,
        )
        self.plane = ReadPlane(
            capture=cast(Any, self.capture),
            monitor=self.monitor,
            geometry=self.geometry,
            lens=self.lens,
            sae=self.sae,
            layers=layers,
            tokenizer=tokenizer,
            gate_refs=gate_refs,
        )

    def _roster_changed(self) -> None:
        self.roster_changes += 1


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
