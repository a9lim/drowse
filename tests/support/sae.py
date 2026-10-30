"""SAE test support: an in-memory backend and direct instrument construction.

- :class:`MockSaeBackend` — an in-memory ``SaeBackend`` (identity encode/decode
  by default) for CPU tests.
- :func:`sae_instrument` — a real :class:`~drowse.sae.instrument.SaeInstrument`
  over a model handle and a capture, with a no-op exclusive section and
  recorded registry effects.
- :func:`make_resident` — install a backend as the runtime's residency without
  the provider/disk load path (the load path itself is covered through
  ``session.sae.load`` with ``resolve_sae_backend`` patched).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, cast

import torch

__all__ = ["MockSaeBackend", "make_resident", "sae_instrument"]


@dataclass
class MockSaeBackend:
    """In-memory SAE backend for CPU-only tests.

    Default is identity encode/decode with ``d_feature == d_model``. Pass
    ``encode_fn`` / ``decode_fn`` for non-trivial layer-level transforms.
    """
    layers: frozenset[int]
    d_model: int
    d_feature: int | None = None
    release: str = "mock-release"
    revision: str | None = None
    fingerprint: str | None = None
    encode_fn: Callable[[int, torch.Tensor], torch.Tensor] | None = None
    decode_fn: Callable[[int, torch.Tensor], torch.Tensor] | None = None
    sae_ids_by_layer: dict[str, str] = field(default_factory=dict)
    repo_id: str | None = None
    neuronpedia_id: str | None = None
    neuronpedia_ids_by_layer: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.d_feature is None:
            self.d_feature = self.d_model
        if not self.sae_ids_by_layer:
            self.sae_ids_by_layer = {
                str(layer): f"mock-layer-{layer}" for layer in self.layers
            }
        if (
            self.fingerprint is None
            and self.encode_fn is None
            and self.decode_fn is None
        ):
            import hashlib
            import json

            self.fingerprint = hashlib.sha256(json.dumps({
                "backend": "mock-identity",
                "release": self.release,
                "revision": self.revision,
                "layers": sorted(self.layers),
                "d_model": self.d_model,
                "d_feature": self.d_feature,
            }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def encode_layer(self, idx: int, h: torch.Tensor) -> torch.Tensor:
        if self.encode_fn is not None:
            return self.encode_fn(idx, h)
        return h

    def decode_layer(self, idx: int, f: torch.Tensor) -> torch.Tensor:
        if self.decode_fn is not None:
            return self.decode_fn(idx, f)
        return f

    def feature_count(self, idx: int) -> int:
        if idx not in self.layers:
            raise KeyError(idx)
        assert self.d_feature is not None
        return int(self.d_feature)

    def feature_direction(self, idx: int, feature_id: int) -> torch.Tensor:
        width = self.feature_count(idx)
        if not 0 <= feature_id < width:
            raise IndexError(feature_id)
        one_hot = torch.zeros(width)
        one_hot[feature_id] = 1
        return self.decode_layer(idx, one_hot)


def sae_instrument(
    handle: Any = None,
    *,
    capture: Any = None,
    registry: Any = None,
    replay: Callable[..., tuple[Any, Any]] | None = None,
    cls: type | None = None,
) -> Any:
    """A real SAE instrument + runtime.

    ``handle`` is the runtime's model handle (default: a ``stub/model`` with
    no layers); give it whatever model surface the test exercises
    (``layers`` for depth summaries, ``device``/``dtype`` for directions).
    ``capture`` is what the per-step reads see (default: none captured).
    ``cls`` substitutes an :class:`SaeInstrument` subclass.  Registry effects
    land on the instrument's ``effects`` dict: ``directions`` (``sae/<id>``
    -> tensors), ``evictions`` (count) and ``forgotten`` (probe names); with
    ``registry`` the directions are also registered there, as on a session.
    """
    from types import SimpleNamespace

    from drowse.sae.instrument import SaeInstrument
    from drowse.sae.runtime import SaeRuntime
    from tests.support.lens import FakeCapture
    from tests.support.replay import unwired_replay

    if handle is None:
        handle = SimpleNamespace(model_id="stub/model", layers=[])
    effects: dict[str, Any] = {"directions": {}, "evictions": 0, "forgotten": []}

    def put(name: str, tensors: Any) -> None:
        effects["directions"][name] = tensors
        if registry is not None:
            registry.put_profile(name, tensors)

    def evict() -> None:
        effects["directions"].clear()
        effects["evictions"] += 1
        if registry is not None:
            registry.evict_prefix("sae/")

    instrument = (cls or SaeInstrument)(
        capture if capture is not None else FakeCapture({}),
        SaeRuntime(
            handle,
            exclusive=lambda *_a, **_k: nullcontext(),
            assert_unsteered=lambda: None,
        ),
        replay=replay or unwired_replay,
        put_direction=put,
        evict_directions=evict,
        forget_probe_hashes=effects["forgotten"].extend,
    )
    instrument.effects = effects  # type: ignore[attr-defined]
    return instrument


def make_resident(
    runtime: Any,
    backend: Any = None,
    *,
    layer: int,
    width: int | None = None,
    feature_meta: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """Make ``backend`` (default: a 4-wide identity mock covering ``layer``)
    the runtime's residency, bypassing resolution and persistence."""
    from drowse.sae.runtime import SaeResidency

    if backend is None:
        backend = MockSaeBackend(layers=frozenset({layer}), d_model=4)
    resident = SaeResidency(
        backend,
        layer,
        int(width if width is not None else backend.feature_count(layer)),
        dict(feature_meta or {}),
    )
    runtime._swap(resident, None)
    return resident
