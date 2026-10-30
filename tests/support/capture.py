"""A model-free harness for the manifold fit pipeline's activation capture.

:func:`install_capture` replaces the pipeline's batched capture seam
(``drowse.fit.capture._encode_and_capture_all_batch``) with a synthetic
per-row encoder and pins the baseline prompts to one, so any node corpus
length is a valid multiple.  :class:`StubHandle` is the ``ModelHandle`` the
pipeline fits against: identity layers, a capture tokenizer, and a covering
synthetic whitener over the given (or synthetic) neutral means.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import Any

import pytest
import torch

__all__ = [
    "DIM",
    "N_LAYERS",
    "CaptureTokenizer",
    "StubHandle",
    "batched",
    "install_capture",
    "label_encoder",
    "label_encoder_batch",
]

DIM = 8
N_LAYERS = 4

RowEncoder = Callable[..., dict[int, torch.Tensor]]


class CaptureTokenizer:
    """Character-code ids, enough for the capture's tokenization step."""

    chat_template = None
    pad_token_id = 0
    eos_token_id = 0
    bos_token_id = 0
    all_special_ids: list[int] = []
    added_tokens_encoder: dict[str, int] = {}

    def __call__(
        self, texts: list[str], *, add_special_tokens: bool = False,
    ) -> dict[str, list[list[int]]]:
        del add_special_tokens
        return {
            "input_ids": [
                [1 + (ord(char) % 97) for char in text] or [1] for text in texts
            ],
        }


def label_encoder(
    model: Any, tokenizer: Any, prompt: str, response: str, layers: Any,
    device: Any, **_kwargs: Any,
) -> dict[int, torch.Tensor]:
    """Synthetic per-layer activations, deterministic per node label.

    Ignores the baseline ``prompt`` and keys off the ``response`` (the node
    corpus entry, ``"<label> statement i"``)."""
    label = response.split()[0]
    seed = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(DIM, generator=g)
    out: dict[int, torch.Tensor] = {}
    for layer in range(len(layers)):
        v = base.clone()
        v[2] = 0.5 * layer            # layer-varying offset
        out[layer] = v + 0.01 * torch.randn(DIM)
    return out


def batched(row_encoder: RowEncoder) -> RowEncoder:
    """The batched capture seam over a per-row encoder: ``{layer: (B, D)}``,
    rows encoded in order (so RNG consumption matches a per-row capture)."""

    def _batch(
        model: Any, tokenizer: Any, prompts: Any, responses: Any, layers: Any,
        device: Any, **kwargs: Any,
    ) -> dict[int, torch.Tensor]:
        rows = [
            row_encoder(model, tokenizer, p, r, layers, device, **kwargs)
            for p, r in zip(prompts, responses)
        ]
        return {
            idx: torch.stack([row[idx] for row in rows])
            for idx in range(len(layers))
        }

    return _batch


label_encoder_batch = batched(label_encoder)


class StubHandle:
    """Minimal ModelHandle for ``ManifoldExtractionPipeline``.

    Real (identity) layer modules host the fit-scoped capture hooks; the
    batched capture seam is monkeypatched, so they never fire.  The covering
    whitener (activation-space fits require one) is synthesized over
    ``layer_means`` (default: synthetic means).
    """

    def __init__(
        self,
        *,
        layer_means: Mapping[int, torch.Tensor] | None = None,
        model_id: str = "stub-model",
    ) -> None:
        from tests._whitener import synthetic_means, synthetic_whitener

        self.model_id = model_id
        # A real nn.Module so the protocol's ``model: nn.Module`` is met.
        self.model: torch.nn.Module = torch.nn.Linear(1, 1)
        self.tokenizer: Any = CaptureTokenizer()
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.layers: Any = torch.nn.ModuleList(
            torch.nn.Identity() for _ in range(N_LAYERS)
        )
        self.layer_means: dict[int, torch.Tensor] = (
            {L: t.clone() for L, t in layer_means.items()} if layer_means is not None
            else synthetic_means(range(N_LAYERS), DIM)
        )
        self.whitener = synthetic_whitener(
            range(N_LAYERS), DIM, means=self.layer_means,
        )


def install_capture(
    monkeypatch: pytest.MonkeyPatch, encoder_batch: RowEncoder = label_encoder_batch,
) -> None:
    """Route the fit pipeline's capture through ``encoder_batch`` and pin the
    baseline prompts to a single one."""
    from drowse.fit import capture

    monkeypatch.setattr(capture, "_encode_and_capture_all_batch", encoder_batch)
    monkeypatch.setattr(capture, "_load_baseline_prompts", lambda: ["baseline prompt"])
