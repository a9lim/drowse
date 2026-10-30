"""A real :class:`~drowse.steering.composer.SteeringComposer` without a model.

:class:`SteeringRig` constructs the composer with the same explicit
dependencies a session wires (an :class:`ArtifactRegistry`, a
:class:`NeutralBaseline`, a :class:`SteeringManager`, the generation lock and
event bus) so steering-stack, projection and lowering tests run on the
collaborator under test instead of a hand-assembled session.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, cast

import torch

from drowse.events import EventBus
from drowse.generation.compile_runtime import CompiledRuntime
from drowse.session.engine import GenerationEngine
from drowse.session.neutral import NeutralBaseline
from drowse.session.registry import ArtifactRegistry, ModelBinding
from drowse.steering.composer import SteeringComposer, SteeringScope
from drowse.steering.hooks import SteeringManager
from drowse.steering.terms import Steering

__all__ = ["SteeringRig"]


@contextmanager
def _no_lock(*_args: Any, **_kwargs: Any) -> Iterator[None]:
    yield


class SteeringRig:
    """One composer plus the collaborators it steers through.

    - ``composer`` / ``registry`` / ``neutral`` / ``manager``: the real objects.
    - ``layers``: the decoder layers hooks attach to (any module list).
    - ``prefill_changes``: how often the composer reported a prefill-steering
      change (the session drops its prefix cache there).
    - :meth:`attach_engine` builds ``engine``, a :class:`GenerationEngine`
      sharing the rig's lock and manager, for generation-transaction tests;
      the rig's phase guard reads that engine's phase.
    """

    def __init__(
        self,
        *,
        layers: Any = (),
        profiles: Mapping[str, Mapping[int, torch.Tensor]] | None = None,
        manifolds: Mapping[str, Any] | None = None,
        layer_means: Mapping[int, torch.Tensor] | None = None,
        whitener: Any = None,
        model: Any = None,
        tokenizer: Any = None,
        model_id: str = "stub/model",
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        manager: SteeringManager | None = None,
    ) -> None:
        device = device or torch.device("cpu")
        self.layers = layers
        self.lock = threading.RLock()
        self.events = EventBus()
        self.manager = manager if manager is not None else SteeringManager()
        self.prefill_changes = 0
        self.engine: GenerationEngine | None = None
        self.registry = ArtifactRegistry(ModelBinding(
            model=model, tokenizer=tokenizer, model_id=model_id,
            device=device, dtype=dtype,
        ))
        for name, tensors in (profiles or {}).items():
            self.registry.put_profile(name, tensors)
        for key, manifold in (manifolds or {}).items():
            self.registry.put_manifold(key, manifold)
        self.neutral = NeutralBaseline(
            model=model, tokenizer=tokenizer, layers=layers, model_id=model_id,
            exclusive=_no_lock, steered=lambda: self.manager.plan is not None,
        )
        if layer_means is not None or whitener is not None:
            self.neutral.adopt(dict(layer_means or {}), whitener)
        self.composer = SteeringComposer(
            artifacts=self.registry, neutral=self.neutral, manager=self.manager,
            layers=layers, device=device, dtype=dtype,
            lock=self.lock, events=self.events,
            generation_phase=self._phase,
            on_prefill_steering_change=self._prefill_changed,
        )

    def steering(self, value: str | Steering) -> SteeringScope:
        """``session.steering`` over the rig: an un-entered scope."""
        return self.composer.open_scope(value)

    def attach_engine(self, runtime: CompiledRuntime | None = None) -> GenerationEngine:
        """Build ``engine`` over the rig's lock and steering manager."""
        self.engine = GenerationEngine(
            cast(Any, None),
            lock=self.lock,
            runtime=runtime if runtime is not None else CompiledRuntime(),
            steering=self.manager,
            reads=cast(Any, None),
            release=lambda: None,
        )
        return self.engine

    def _phase(self) -> str | None:
        name = getattr(getattr(self.engine, "phase", None), "name", None)
        return name if name in ("RUNNING", "FINALIZING") else None

    def _prefill_changed(self) -> None:
        self.prefill_changes += 1
