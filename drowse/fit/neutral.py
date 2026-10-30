"""Load or capture a model's neutral activations.

The compute half of the per-model neutral cache: render the neutral corpus,
derive the cache identity (rendered token rows, loaded weights, checkpoint
identity, layer set), reuse a validated cache whose identity matches, and run
the capture forward on a miss.  The cache file format, validation and
publication are pure IO in :mod:`drowse.store.neutral`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable

import torch

from drowse.fit import capture as _capture
from drowse.model.fingerprint import loaded_model_fingerprint
from drowse.store.neutral import (
    NEUTRAL_CAPTURE_VERSION,
    load_validated_neutral_cache,
    neutral_cache_matches,
    neutral_fit_lock,
    publish_neutral_cache,
)

log = logging.getLogger(__name__)


def load_or_compute_neutral_activations(
    model: Any,
    tokenizer: Any,
    layers: Any,
    *,
    model_id: str,
    force: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> dict[int, torch.Tensor]:
    """Single-flight neutral-cache load/capture, returning activations only.

    ``on_progress`` narrates the cache-miss capture loop (per forward chunk);
    a cache hit reports nothing because it runs no model.
    """
    activations, _sidecar = load_or_compute_neutral_activations_with_metadata(
        model, tokenizer, layers, model_id=model_id, force=force,
        on_progress=on_progress,
    )
    return activations


def load_or_compute_neutral_activations_with_metadata(
    model: Any,
    tokenizer: Any,
    layers: Any,
    *,
    model_id: str,
    force: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Return activations plus the sidecar proven in the same transaction.

    Alignment fitting needs both the materialized neutral rows and their stable
    cache identity.  Returning the already-validated sidecar avoids an immediate
    second full-file digest pass through
    :func:`~drowse.store.neutral.load_validated_neutral_cache`.  The
    activation-only API above remains the convenience surface for ordinary
    whitener/probe callers.
    """
    with neutral_fit_lock(model_id):
        return _load_or_compute_neutral_activations_with_metadata_locked(
            model, tokenizer, layers, model_id=model_id, force=force,
            on_progress=on_progress,
        )


def _load_or_compute_neutral_activations_with_metadata_locked(
    model: Any,
    tokenizer: Any,
    layers: Any,
    *,
    model_id: str,
    force: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Disk-cached per-statement activations for one model.

    Cache identity binds the exact rendered token rows and pooling positions,
    the loaded model weights, and the fitted layer set. The tensor payload is
    digest-verified before reuse. A stale cache is recomputed and replaced.

    Returns ``({layer_idx: [N, D] fp32 CPU tensor}, validated_sidecar)``.
    Stored fp32 on disk so the whitener covariance is built and inverted at
    full precision and the compute path (cache miss) and cache-hit path return
    bit-identical tensors (no precision seam — bf16's ~0.4% input error landed
    on exactly the inverted quantity).  fp16 was abandoned because its 65504
    ceiling overflows gemma-3's extreme late-layer channels to ±inf (poisoning
    ``λ`` / ``K``); fp32 has the range and, unlike the former bf16 store, no
    input-precision loss, at the cost of ~2× disk on this one cache.
    """
    pairs = _capture._neutral_pairs()
    prompts = [prompt for prompt, _ in pairs]
    responses = [response for _, response in pairs]
    prepared = _capture._prepare_capture_batch(
        tokenizer, prompts, responses,
    )
    rendered_payload = [
        {"input_ids": ids[0].tolist(), "content_end": int(content_end)}
        for ids, content_end in prepared
    ]
    capture_sha = hashlib.sha256(json.dumps(
        {"capture_version": NEUTRAL_CAPTURE_VERSION, "rows": rendered_payload},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    model_fingerprint = loaded_model_fingerprint(model, model_id)
    base_model = getattr(model, "_orig_mod", model)
    model_source_fingerprint = getattr(
        base_model, "_drowse_source_fingerprint", None,
    )
    identity = {
        "capture_sha256": capture_sha,
        "model_fingerprint": model_fingerprint,
        "model_source_fingerprint": model_source_fingerprint,
    }

    if not force:
        try:
            out, sc = load_validated_neutral_cache(model_id)
        except FileNotFoundError:
            pass
        except Exception as e:
            log.warning("Corrupt neutral activations cache for %s, recomputing: %s", model_id, e)
        else:
            if neutral_cache_matches(
                sc, layers=list(range(len(layers))), **identity,
            ):
                log.debug("Loaded cached neutral activations for %s", model_id)
                return out, sc
            log.info(
                "Neutral activations stale (model/token/corpus/layer identity changed); "
                "recomputing for %s",
                model_id,
            )

    log.info("Computing neutral activations (one-time per model)...")
    if on_progress is not None:
        on_progress(
            f"Computing neutral activations for {model_id} "
            f"(one-time per model, {len(pairs)} prompts)..."
        )
    activations = _capture.compute_neutral_activations(
        model, tokenizer, layers, rendered=prepared, on_progress=on_progress,
    )
    sidecar = publish_neutral_cache(model_id, activations, **identity)
    return activations, sidecar
