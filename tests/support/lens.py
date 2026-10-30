"""A real :class:`~drowse.lens.instrument.LensInstrument` on a toy model.

:class:`LensHarness` builds the instrument with the same explicit
dependencies the session wires, on the toy lens model from
``tests/_jlens_toys`` (or any model passed in), and records what the session
callbacks would have done.  Unit tests of the lens runtime, the fit
lifecycle and the instrument use it instead of borrowing session methods.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch

from tests._jlens_toys import CharTokenizer, frozen_toy

__all__ = [
    "FakeCapture",
    "LENS_PROMPTS",
    "LensHarness",
    "TOY_MODEL_ID",
    "lens_instrument",
    "lens_kwargs",
]

TOY_MODEL_ID = "toy/jlens-model"

#: A small corpus whose documents clear the estimator's minimum length.
LENS_PROMPTS = [
    "a first prompt that is long enough..",
    "the second prompt, also long enough.",
    "and a third one to round out corpus.",
    "plus a fourth for the resume checks.",
]


class FakeCapture:
    """Minimal HiddenCapture stand-in: ``per_layer_buckets`` -> latest slices."""

    def __init__(self, slices: Mapping[int, torch.Tensor]) -> None:
        self._buckets = {layer: [tensor] for layer, tensor in slices.items()}

    def per_layer_buckets(self) -> dict[int, list[torch.Tensor]]:
        return self._buckets

    def latest_per_layer(self) -> dict[int, torch.Tensor]:
        return {layer: rows[-1] for layer, rows in self._buckets.items()}


class _CaptureSlot:
    """Forwards the instrument's capture reads to ``harness.capture``."""

    def __init__(self, harness: "LensHarness") -> None:
        self._harness = harness

    def latest_per_layer(self) -> dict[int, torch.Tensor]:
        return self._harness.capture.latest_per_layer()

    def per_layer_buckets(self) -> dict[int, list[torch.Tensor]]:
        return self._harness.capture.per_layer_buckets()


def _toy_registry(model_id: str, *, model: Any = None, tokenizer: Any = None) -> Any:
    """A real :class:`~drowse.session.registry.ArtifactRegistry` for a toy."""
    from drowse.session.registry import ArtifactRegistry, ModelBinding

    return ArtifactRegistry(ModelBinding(
        model=model, tokenizer=tokenizer, model_id=model_id,
        device=torch.device("cpu"), dtype=torch.float32,
    ))


@contextmanager
def _no_exclusive(_message: str, *, phase_msg: str | None = None) -> Iterator[None]:
    del phase_msg
    yield


def lens_kwargs(
    *,
    model: Any = None,
    tokenizer: Any = None,
    layers: Any = (),
    n_layers: int | None = None,
    model_id: str = "stub/model",
    capture: Any = None,
    registry: Any = None,
    replay: Any = None,
) -> dict[str, Any]:
    """The :class:`LensInstrument` constructor arguments, with inert defaults.

    ``capture`` defaults to an empty :class:`FakeCapture`, ``registry`` to a
    fresh artifact registry (profiles resolve from it), ``replay`` to
    :func:`~tests.support.replay.unwired_replay`; the capture forward runs
    ``model`` over ``layers``.  For tests (and instrument subclasses) that
    construct the instrument themselves.
    """
    from tests.support.replay import unwired_replay

    registry = registry if registry is not None else _toy_registry(model_id)

    def _capture_hidden(input_ids: torch.Tensor, **kwargs: Any) -> dict[int, torch.Tensor]:
        from drowse.fit import capture as fit_capture

        return fit_capture._capture_all_hidden_states(model, layers, input_ids, **kwargs)

    return dict(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        device=torch.device("cpu"),
        n_layers=len(layers) if n_layers is None else n_layers,
        capture=capture if capture is not None else FakeCapture({}),
        exclusive=_no_exclusive,
        capture_hidden=_capture_hidden,
        registry=registry,
        on_change=lambda _names: None,
        resolve_profile=lambda selector: registry.profiles[selector],
        assert_unsteered=lambda: None,
        replay=unwired_replay if replay is None else replay,
    )


def lens_instrument(**kwargs: Any) -> Any:
    """A real :class:`LensInstrument` built from :func:`lens_kwargs`."""
    from drowse.lens.instrument import LensInstrument

    return LensInstrument(**lens_kwargs(**kwargs))


class LensHarness:
    """One ``LensInstrument`` plus the session-side state it writes to.

    - ``lens``: the instrument.
    - ``registry``: the profile registry ``register_direction`` writes
      (``profiles`` is its read-only view).
    - ``changes``: every ``on_change`` call's probe names.
    - ``capture``: what the per-step reads see (assign a :class:`FakeCapture`).
    - ``exclusive_messages``: every model-exclusive section entered.

    ``replay`` is the instrument's token replay (unwired by default; a
    :class:`tests.support.replay.ReplayRig` supplies a real one).
    """

    def __init__(
        self,
        *,
        n_layers: int = 3,
        model: Any = None,
        layers: Any = None,
        tokenizer: Any = None,
        model_id: str = TOY_MODEL_ID,
        replay: Any = None,
    ) -> None:
        from drowse.lens.instrument import LensInstrument
        from tests.support.replay import unwired_replay

        self.model = frozen_toy(n_layers=n_layers) if model is None else model
        self.tokenizer = CharTokenizer() if tokenizer is None else tokenizer
        self.layers = self.model.model.layers if layers is None else layers
        self.device = torch.device("cpu")
        self.model_id = model_id
        self.registry = _toy_registry(
            model_id, model=self.model, tokenizer=self.tokenizer,
        )
        self.changes: list[list[str]] = []
        self.capture: Any = FakeCapture({})
        self.exclusive_messages: list[str] = []
        self.lens = LensInstrument(
            model=self.model,
            tokenizer=self.tokenizer,
            model_id=model_id,
            device=self.device,
            n_layers=len(self.layers),
            capture=_CaptureSlot(self),
            exclusive=self._exclusive,
            capture_hidden=self._capture_hidden,
            registry=self.registry,
            on_change=lambda names: self.changes.append(list(names)),
            resolve_profile=lambda selector: self.registry.profiles[selector],
            assert_unsteered=lambda: None,
            replay=unwired_replay if replay is None else replay,
        )

    @property
    def profiles(self) -> Mapping[str, Any]:
        """Read-only view of the registry's profiles."""
        return self.registry.profiles

    def _capture_hidden(self, input_ids: torch.Tensor, **kwargs: Any) -> dict[int, torch.Tensor]:
        """The session's capture forward, looked up per call so a test can
        spy on ``drowse.fit.capture._capture_all_hidden_states``."""
        from drowse.fit import capture

        return capture._capture_all_hidden_states(self.model, self.layers, input_ids, **kwargs)

    @contextmanager
    def _exclusive(self, message: str, *, phase_msg: str | None = None) -> Iterator[None]:
        del phase_msg
        self.exclusive_messages.append(message)
        yield
