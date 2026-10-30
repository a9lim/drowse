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
    "stub_lens_instrument",
    "stub_lens_kwargs",
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


class _StubCapture:
    """Forwards the instrument's capture reads to ``stub._capture``."""

    def __init__(self, stub: Any) -> None:
        self._stub = stub

    def latest_per_layer(self) -> dict[int, torch.Tensor]:
        return self._stub._capture.latest_per_layer()

    def per_layer_buckets(self) -> dict[int, list[torch.Tensor]]:
        return self._stub._capture.per_layer_buckets()


def stub_lens_instrument(stub: Any) -> Any:
    """A ``LensInstrument`` for a narrow ``DrowseSession.__new__`` stub.

    Every dependency is looked up on the stub when it is used, so a test can
    assign ``stub._capture`` / ``stub._pooled_aggregate_slice`` after wiring.
    The resident lens comes from the store as usual.
    """
    from drowse.lens.instrument import LensInstrument

    return LensInstrument(**stub_lens_kwargs(stub))


def stub_lens_kwargs(stub: Any) -> dict[str, Any]:
    """The ``LensInstrument`` constructor arguments for a session stub (for
    test fakes that subclass the instrument)."""
    class _StubProfiles(dict):  # type: ignore[type-arg]
        pass

    @contextmanager
    def _exclusive(_message: str, *, phase_msg: str | None = None) -> Iterator[None]:
        del phase_msg
        yield

    def _capture_hidden(input_ids: torch.Tensor, **kwargs: Any) -> dict[int, torch.Tensor]:
        from drowse.fit import capture

        return capture._capture_all_hidden_states(
            stub._model, stub._layers, input_ids, **kwargs,
        )

    profiles = getattr(stub, "_profiles", None)
    return dict(
        model=getattr(stub, "_model", None),
        tokenizer=getattr(stub, "_tokenizer", None),
        model_id="stub/model",
        device=torch.device("cpu"),
        n_layers=len(getattr(stub, "_layers", None) or ()),
        capture=_StubCapture(stub),
        exclusive=_exclusive,
        capture_hidden=_capture_hidden,
        pooled_slice=lambda ids: stub._pooled_aggregate_slice(ids),
        profiles=profiles if isinstance(profiles, dict) else _StubProfiles(),
        on_change=lambda _names: None,
        resolve_profile=lambda selector: stub.ensure_profile_registered(selector),
        assert_unsteered=lambda: None,
        replay_host=stub,
    )


class LensHarness:
    """One ``LensInstrument`` plus the session-side state it writes to.

    - ``lens``: the instrument.
    - ``profiles``: the profile registry ``register_direction`` writes.
    - ``changes``: every ``on_change`` call's probe names.
    - ``capture``: what the per-step reads see (assign a :class:`FakeCapture`).
    - ``pooled``: what the finalize pool hands back.
    - ``exclusive_messages``: every model-exclusive section entered.
    """

    def __init__(
        self,
        *,
        n_layers: int = 3,
        model: Any = None,
        layers: Any = None,
        tokenizer: Any = None,
        model_id: str = TOY_MODEL_ID,
        replay_host: Any = None,
    ) -> None:
        from drowse.lens.instrument import LensInstrument

        self.model = frozen_toy(n_layers=n_layers) if model is None else model
        self.tokenizer = CharTokenizer() if tokenizer is None else tokenizer
        self.layers = self.model.model.layers if layers is None else layers
        self.device = torch.device("cpu")
        self.model_id = model_id
        self.profiles: dict[str, Any] = {}
        self.changes: list[list[str]] = []
        self.capture: Any = FakeCapture({})
        self.pooled: dict[int, torch.Tensor] = {}
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
            pooled_slice=lambda _ids: self.pooled,
            profiles=self.profiles,
            on_change=lambda names: self.changes.append(list(names)),
            resolve_profile=self.profiles.__getitem__,
            assert_unsteered=lambda: None,
            replay_host=self if replay_host is None else replay_host,
        )

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
