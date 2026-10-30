"""Give hand-built session stubs the collaborators a real session wires.

Real sessions build an :class:`~drowse.session.registry.ArtifactRegistry`, a
:class:`~drowse.session.neutral.NeutralBaseline` and a
:class:`~drowse.steering.composer.SteeringComposer` in ``__init__``.  Suites
that still assemble ``DrowseSession.__new__`` skeletons (or duck-typed stubs)
call :func:`wire_stub_steering` to get the same three objects, built from the
stub's own attributes.  Prefer ``tiny_session`` for new tests.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import torch

from drowse.events import EventBus
from drowse.session.neutral import NeutralBaseline
from drowse.session.registry import ArtifactRegistry, ModelBinding
from drowse.steering.composer import GateSources, SteeringComposer
from drowse.steering.hooks import SteeringManager

__all__ = ["attach_stub_instruments", "wire_stub_steering"]


def attach_stub_instruments(stub: Any) -> Any:
    """Construct the three read instruments on ``stub`` (as ``__init__`` does).

    The lens instrument takes its dependencies from the stub's attributes at
    construction (see :func:`tests.support.lens.stub_lens_kwargs`), so set
    ``_model`` / ``_tokenizer`` / ``_layers`` first when a test needs them.
    """
    from drowse.readout.geometry import GeometryInstrument
    from tests.support.lens import stub_lens_instrument
    from tests.support.sae import stub_sae_instrument

    d = stub.__dict__
    if "_geometry_instrument" not in d:
        stub._geometry_instrument = GeometryInstrument(stub)
    if "_lens_instrument" not in d:
        stub._lens_instrument = stub_lens_instrument(stub)
    if "_sae_instrument" not in d:
        stub._sae_instrument = stub_sae_instrument(stub)
    return stub


def _model_id(stub: Any) -> str:
    info = getattr(stub, "_model_info", None)
    if isinstance(info, Mapping) and info.get("model_id"):
        return str(info["model_id"])
    model_id = stub.__dict__.get("model_id") if hasattr(stub, "__dict__") else None
    return str(model_id or getattr(type(stub), "model_id", None) or "stub/model")


@contextmanager
def _no_lock(*_args: Any, **_kwargs: Any):
    yield


def wire_stub_steering(
    stub: Any,
    *,
    profiles: Mapping[str, Mapping[int, torch.Tensor]] | None = None,
    manifolds: Mapping[str, Any] | None = None,
    layer_means: Mapping[int, torch.Tensor] | None = None,
    whitener: Any = None,
) -> SteeringComposer:
    """Install ``_registry`` / ``_neutral`` / ``_steering_composer`` on ``stub``.

    Missing session state the collaborators need (``_gen_lock``, ``events``,
    ``_steering``, ``_device``, ``_dtype``) is created on the stub first.
    ``profiles`` / ``manifolds`` seed the registry; ``layer_means`` /
    ``whitener`` seed a resident neutral baseline.  Gate sources, atom
    resolvers and the phase guard read the stub's attributes at call time.
    """
    d = stub.__dict__
    if "_gen_lock" not in d:
        stub._gen_lock = threading.RLock()
    if "events" not in d:
        stub.events = EventBus()
    if d.get("_steering") is None:
        stub._steering = SteeringManager()
    device = d.get("_device") or torch.device("cpu")
    dtype = d.get("_dtype") or torch.float32
    stub._device, stub._dtype = device, dtype
    layers = d.get("_layers", [])

    registry = ArtifactRegistry(ModelBinding(
        model=d.get("_model"), tokenizer=d.get("_tokenizer"),
        model_id=_model_id(stub), device=device, dtype=dtype,
    ))
    registry.on_change(lambda: getattr(stub, "_invalidate_prefix_cache", lambda: None)())
    registry.set_atom_resolver("jlens", lambda atom: stub.lens.register_direction(atom))
    registry.set_atom_resolver("sae", lambda atom: stub.sae.register_direction(atom))
    if hasattr(stub, "_fold_profile_probe"):
        registry.set_probe_folder(lambda key, tensors: stub._fold_profile_probe(key, tensors))
    for name, tensors in (profiles or {}).items():
        registry.put_profile(name, tensors)
    for key, manifold in (manifolds or {}).items():
        registry.put_manifold(key, manifold)

    def _steered() -> bool:
        manager = getattr(stub, "_steering", None)
        return getattr(manager, "plan", None) is not None

    exclusive = getattr(stub, "_model_exclusive", None) or _no_lock
    neutral = NeutralBaseline(
        model=d.get("_model"), tokenizer=d.get("_tokenizer"), layers=layers,
        model_id=_model_id(stub), exclusive=exclusive, steered=_steered,
    )
    if layer_means is not None or whitener is not None:
        neutral.adopt(dict(layer_means or {}), whitener)

    def _phase() -> str | None:
        phase = getattr(stub, "_gen_phase", None)
        name = getattr(phase, "name", None)
        return name if name in ("RUNNING", "FINALIZING") else None

    composer = SteeringComposer(
        artifacts=registry, neutral=neutral, manager=stub._steering,
        layers=layers, device=device, dtype=dtype,
        lock=stub._gen_lock, events=stub.events,
        generation_phase=_phase,
        on_prefill_steering_change=lambda: getattr(
            stub, "_invalidate_prefix_cache", lambda: None,
        )(),
        gates=GateSources(
            monitor=lambda: stub._monitor,
            capture=lambda: stub._capture,
            capture_state=lambda: stub._capture_state,
            incremental_readings=lambda: stub._incremental_readings,
            incremental_gate_scores=lambda: stub._incremental_gate_scores,
            instruments=lambda: stub.instruments,
            lens_gate_scalars=lambda keys, *, step_id: stub.lens.current_run.gate_scalars(
                step_id, None, keys,
            ),
            sae_gate_scalars=lambda keys, *, step_id: stub.sae.current_run.gate_scalars(
                step_id, None, keys,
            ),
        ),
    )
    stub._registry = registry
    stub._neutral = neutral
    stub._steering_composer = composer
    return composer
