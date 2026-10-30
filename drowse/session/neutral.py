"""NeutralBaseline — the per-model neutral means and Mahalanobis whitener.

Both read the one fp32 neutral-activation cache
(:func:`~drowse.store.neutral.load_or_compute_neutral_activations`): the
probe-centering mean is its per-layer ``X.mean(0)`` and the whitener's
covariance is built from the same stack, so one load produces both.

Two entry points:

- :meth:`NeutralBaseline.ensure` is unguarded.  A caller that already holds the
  exclusive model lock (steering composition, generation preflight, session
  construction) builds the baseline through it.  It still refuses to *compute*
  while steering hooks are installed, since the neutral forward pass would read
  steered activations and persist them.
- :meth:`layer_means` and :meth:`whitener` are the guarded public getters: a
  build from another thread takes the exclusive lock non-blocking and refuses
  during a generation or inside a steering scope.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

import torch

from drowse.errors import ConcurrentExtractionError, DrowseError

if TYPE_CHECKING:
    from drowse.geometry.whitener import LayerWhitener

_log = logging.getLogger(__name__)

ExclusiveSection = Callable[..., AbstractContextManager[None]]


class NeutralBaseline:
    """Lazily built neutral means + whitener for one loaded model."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        layers: Any,
        model_id: str,
        exclusive: ExclusiveSection,
        steered: Callable[[], bool],
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._layers = layers
        self._model_id = model_id
        self._exclusive = exclusive
        self._steered = steered
        self._means: dict[int, torch.Tensor] = {}
        self._whitener: LayerWhitener | None = None
        self._load_error: BaseException | None = None
        self._on_whitener: list[Callable[["LayerWhitener"], None]] = []

    # ------------------------------------------------------------- state

    @property
    def means(self) -> dict[int, torch.Tensor]:
        """The resident means (``{}`` until built).  No build, no guard."""
        return self._means

    @property
    def resident_whitener(self) -> "LayerWhitener | None":
        """The resident whitener (``None`` until built).  No build, no guard."""
        return self._whitener

    @property
    def ready(self) -> bool:
        return bool(self._means) and self._whitener is not None

    def on_whitener(self, callback: Callable[["LayerWhitener"], None]) -> None:
        """Call ``callback(whitener)`` whenever a whitener is built."""
        self._on_whitener.append(callback)

    def adopt(
        self, means: dict[int, torch.Tensor], whitener: "LayerWhitener | None",
    ) -> None:
        """Install an externally computed baseline (means + whitener)."""
        self._means = dict(means)
        self._whitener = whitener
        self._load_error = None
        if whitener is not None:
            for callback in self._on_whitener:
                callback(whitener)

    def reset(self) -> None:
        """Drop the resident baseline (session teardown)."""
        self._means = {}
        self._whitener = None
        self._load_error = None

    # ------------------------------------------------------------ build

    def ensure(self, *, on_progress: Callable[[str], None] | None = None) -> None:
        """Make the means and the whitener resident with one neutral load.

        Unguarded: the caller holds the exclusive model lock.  A failed load
        leaves the means empty (``require_means`` raises) and the whitener
        ``None``; a whitener that cannot be built from loaded activations is
        logged and left ``None``, since every Mahalanobis consumer raises its
        own ``WhitenerError`` without one.
        """
        if self.ready:
            return
        if self._steered():
            raise ConcurrentExtractionError(
                "artifact-producing model operations cannot run inside an "
                "active steering scope; exit session.steering() first"
            )
        from drowse.geometry.whitener import LayerWhitener
        from drowse.store.neutral import load_or_compute_neutral_activations

        try:
            acts = load_or_compute_neutral_activations(
                self._model, self._tokenizer, self._layers,
                model_id=self._model_id, on_progress=on_progress,
            )
        except Exception as exc:
            self._load_error = exc
            _log.warning(
                "neutral baseline: load failed (%s); Mahalanobis-only fitting, "
                "projection, and probe reads are unavailable. Error: %s",
                type(exc).__name__, exc,
            )
            return
        self._load_error = None
        if not self._means:
            self._means = {idx: X.mean(dim=0) for idx, X in acts.items()}
        if self._whitener is not None:
            return
        try:
            whitener = LayerWhitener.from_neutral_activations(acts, self._means)
        except Exception as exc:
            _log.warning(
                "whitener: build failed (%s); Mahalanobis-only fitting, "
                "projection, and probe reads are unavailable. Error: %s",
                type(exc).__name__, exc,
            )
            return
        self._whitener = whitener
        for callback in self._on_whitener:
            callback(whitener)

    def require_means(self) -> dict[int, torch.Tensor]:
        """The means, built if absent (unguarded); raises when they cannot be."""
        if not self._means:
            self.ensure()
        if not self._means:
            if self._load_error is not None:
                raise DrowseError(
                    "failed to build neutral layer means"
                ) from self._load_error
            raise DrowseError("neutral layer-mean bootstrap returned empty")
        return self._means

    # ---------------------------------------------------- guarded getters

    def layer_means(self) -> dict[int, torch.Tensor]:
        """Per-layer neutral means, built on first use under the exclusive lock.

        Raises :class:`~drowse.errors.DrowseError` when the neutral cache
        cannot be built, and :class:`~drowse.errors.ConcurrentExtractionError`
        when a build is needed while a model op, a generation, or a steering
        scope is active.
        """
        if not self._means:
            with self._exclusive(
                "layer_means requested while another model use is in flight",
                phase_msg="layer_means requested while generation is in flight",
            ):
                return self.require_means()
        return self._means

    def whitener(self) -> "LayerWhitener | None":
        """The Mahalanobis whitener, built on first use under the exclusive lock.

        ``None`` when it cannot be built (the consumers raise ``WhitenerError``);
        :class:`~drowse.errors.ConcurrentExtractionError` when a build is needed
        while a model op, a generation, or a steering scope is active.
        """
        if self._whitener is None:
            with self._exclusive(
                "whitener requested while another model use is in flight",
                phase_msg="whitener requested while generation is in flight",
            ):
                self.ensure()
        return self._whitener
