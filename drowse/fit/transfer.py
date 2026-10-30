"""Cross-model transfer alignment: load or fit a source->target map end to end.

:func:`load_or_fit_transfer_alignment` is the single-flight orchestration
behind ``drowse manifold transfer``.  It proves both models' neutral caches
(:mod:`drowse.store.neutral`) without hashing payloads, reuses a cached
alignment map (:mod:`drowse.store.alignment`) when it covers the requested
layers, fits only the missing layers (:mod:`drowse.geometry.procrustes`), and
returns the target model's identity-matched Mahalanobis metric alongside the
map, so the caller re-bakes the transferred manifold in target units.

Concurrency contract (load-bearing):

* The directional ``alignment_fit_lock`` is held for the whole transaction.
* A model is loaded only on a neutral-cache miss, and then under that model's
  ``neutral_fit_lock`` acquired *before* the load, so two transfers sharing a
  cold model incur one load and one capture.
* Rows are proven by a header-only sidecar preflight plus an identity re-check
  after materialization; a cache writer racing the preflight restarts the
  transaction under the outer lock with the new identity.
* Unrequested layers are dropped as soon as shared coverage is known, so a
  narrow transfer never retains both models' full neutral rosters.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch

from drowse.geometry.procrustes import (
    AlignmentError,
    LayerAlignment,
    alignment_quality,
    fit_alignment,
)
from drowse.store.alignment import (
    alignment_fit_lock,
    load_alignment_map,
    save_alignment_map,
)
from drowse.fit.neutral import (
    _load_or_compute_neutral_activations_with_metadata_locked,
)
from drowse.store.neutral import (
    load_validated_neutral_cache,
    neutral_cache_identity,
    neutral_fit_lock,
    validate_neutral_cache_metadata,
)

Rows = dict[int, torch.Tensor]


@dataclass(frozen=True)
class TransferAlignment:
    """A complete source->target alignment plus the target-side metric.

    ``alignment`` and ``quality_per_layer`` cover exactly the requested shared
    layers.  ``target_whitener`` / ``target_layer_means`` are built from the
    same target neutral rows whose identity is ``target_identity``.
    """

    alignment: dict[int, LayerAlignment]
    quality_per_layer: dict[int, float]
    source_identity: dict[str, Any]
    target_identity: dict[str, Any]
    target_whitener: Any
    target_layer_means: dict[int, torch.Tensor]


@contextmanager
def _loaded_model(model_id: str) -> Iterator[tuple[Any, Any, Any]]:
    """Load ``model_id`` just long enough for one neutral capture.

    A bare ``load_model`` (not a session): the capture needs only the model,
    tokenizer, and layer stack, and the loader stamps the source fingerprint
    the neutral cache binds.
    """
    from drowse.model.arch import get_layers
    from drowse.model.load import load_model

    model, tokenizer = load_model(model_id, device="auto")
    try:
        yield model, tokenizer, get_layers(model)
    finally:
        del model, tokenizer


def _proven_sidecar(model_id: str) -> dict[str, Any] | None:
    """Prove cache/source identity without hashing any tensor payload."""
    from drowse.model.fingerprint import model_source_fingerprint

    try:
        source = model_source_fingerprint(model_id)
        if source is None:
            return None
        sidecar = validate_neutral_cache_metadata(model_id)
        if sidecar.get("model_source_fingerprint") != source:
            return None
        return sidecar
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        return None


def _load_proven_rows(
    model_id: str, layers: set[int] | None,
) -> tuple[Rows, dict[str, Any]] | None:
    before = _proven_sidecar(model_id)
    if before is None:
        return None
    try:
        rows, after = load_validated_neutral_cache(
            model_id, requested_layers=layers,
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        return None
    if neutral_cache_identity(before) != neutral_cache_identity(after):
        return None
    return rows, after


def _load_or_capture_rows(
    model_id: str, layers: set[int] | None,
) -> tuple[Rows, dict[str, Any]]:
    cached_rows = _load_proven_rows(model_id, layers)
    if cached_rows is not None:
        return cached_rows
    # Single-flight starts before model construction, not merely before the
    # neutral forward. Two directional transfers sharing a cold model cache
    # therefore incur one large model load as well as one capture.
    with neutral_fit_lock(model_id):
        cached_rows = _load_proven_rows(model_id, layers)
        if cached_rows is not None:
            return cached_rows
        with _loaded_model(model_id) as (model, tokenizer, model_layers):
            all_rows, sidecar = (
                _load_or_compute_neutral_activations_with_metadata_locked(
                    model, tokenizer, model_layers,
                    model_id=model_id, force=False,
                )
            )
    if layers is None:
        return all_rows, sidecar
    return (
        {layer: value for layer, value in all_rows.items() if layer in layers},
        sidecar,
    )


def _with_target_metric(
    alignment: dict[int, LayerAlignment],
    quality: dict[int, float],
    source_identity: dict[str, Any],
    target_identity: dict[str, Any],
    target_rows: Rows,
) -> TransferAlignment:
    from drowse.geometry.whitener import LayerWhitener

    means = {layer: rows.mean(dim=0) for layer, rows in target_rows.items()}
    return TransferAlignment(
        alignment=alignment,
        quality_per_layer=quality,
        source_identity=source_identity,
        target_identity=target_identity,
        target_whitener=LayerWhitener.from_neutral_activations(target_rows, means),
        target_layer_means=means,
    )


def load_or_fit_transfer_alignment(
    src_model: str,
    tgt_model: str,
    *,
    force: bool = False,
    requested_layers: Sequence[int] | None = None,
) -> TransferAlignment:
    """Single-flight a complete alignment plus its target metric.

    Raises :class:`~drowse.geometry.procrustes.AlignmentError` when the two
    models share no requested layer or the fit is degenerate.
    """
    with alignment_fit_lock(src_model, tgt_model):
        return _load_or_fit_locked(
            src_model, tgt_model, force=force,
            requested_layers=requested_layers,
        )


def _load_or_fit_locked(
    src_model: str,
    tgt_model: str,
    *,
    force: bool,
    requested_layers: Sequence[int] | None,
) -> TransferAlignment:
    wanted_arg = (
        None if requested_layers is None
        else {int(layer) for layer in requested_layers}
    )

    src_sidecar = _proven_sidecar(src_model)
    tgt_sidecar = _proven_sidecar(tgt_model)
    src_seed_rows: Rows | None = None
    tgt_seed_rows: Rows | None = None
    if src_sidecar is None:
        src_seed_rows, src_sidecar = _load_or_capture_rows(src_model, None)
    if tgt_sidecar is None:
        tgt_seed_rows, tgt_sidecar = _load_or_capture_rows(tgt_model, None)

    src_identity = neutral_cache_identity(src_sidecar)
    tgt_identity = neutral_cache_identity(tgt_sidecar)
    available = set(int(layer) for layer in src_sidecar.get("layers", [])) & set(
        int(layer) for layer in tgt_sidecar.get("layers", [])
    )
    wanted = available if wanted_arg is None else (available & wanted_arg)
    if not wanted:
        raise AlignmentError(
            f"{src_model} and {tgt_model} have no requested shared layers"
        )

    # A cold cache fill necessarily captures the model's full neutral roster,
    # but a narrow transfer must not retain every layer from both models for
    # the rest of the transaction. Rebinding here drops unrequested tensor
    # owners as soon as shared coverage is known.
    if src_seed_rows is not None:
        src_seed_rows = {layer: src_seed_rows[layer] for layer in wanted}
    if tgt_seed_rows is not None:
        tgt_seed_rows = {layer: tgt_seed_rows[layer] for layer in wanted}

    cached = load_alignment_map(
        src_model, tgt_model,
        source_identity=src_identity, target_identity=tgt_identity,
        requested_layers=(set() if force else wanted),
    )
    cached_map: dict[int, LayerAlignment] = {}
    cached_sidecar: dict[str, Any] = {}
    if cached is not None:
        cached_map, cached_sidecar = cached
    complete_hit = not force and set(cached_map) >= wanted
    if complete_hit:
        # The cached map proves the source side; only the target rows are still
        # needed to reconstruct the exact target metric.
        src_seed_rows = None

    # Target rows are always needed for the exact target-metric re-bake, but
    # only for layers the source profile can actually transfer.
    tgt_loaded = (
        ({layer: tgt_seed_rows[layer] for layer in wanted}, tgt_sidecar)
        if tgt_seed_rows is not None else _load_proven_rows(tgt_model, wanted)
    )
    if tgt_loaded is None:
        tgt_loaded = _load_or_capture_rows(tgt_model, wanted)
    tgt_acts, tgt_sidecar = tgt_loaded
    tgt_seed_rows = None
    if neutral_cache_identity(tgt_sidecar) != tgt_identity:
        # A cache writer raced the metadata preflight. Restart this directional
        # transaction under the outer alignment fit lock with the new identity.
        return _load_or_fit_locked(
            src_model, tgt_model, force=force, requested_layers=requested_layers,
        )

    raw_quality = cached_sidecar.get("quality_per_layer") or {}
    quality_per_layer = {
        int(layer): float(value) for layer, value in raw_quality.items()
        if value is not None
    }
    result_map = dict(cached_map)
    if not complete_hit:
        missing = wanted if force else (wanted - set(cached_map))
        src_loaded = (
            ({layer: src_seed_rows[layer] for layer in missing}, src_sidecar)
            if src_seed_rows is not None
            else _load_proven_rows(src_model, missing)
        )
        if src_loaded is None:
            src_loaded = _load_or_capture_rows(src_model, missing)
        src_acts, src_sidecar = src_loaded
        src_seed_rows = None
        if neutral_cache_identity(src_sidecar) != src_identity:
            return _load_or_fit_locked(
                src_model, tgt_model, force=force,
                requested_layers=requested_layers,
            )
        tgt_missing = {layer: tgt_acts[layer] for layer in missing}
        fitted = fit_alignment(
            src_acts, tgt_missing,
            requested_layers=missing,
            available_shared_layers=available,
        )
        fitted_quality = alignment_quality(fitted, src_acts, tgt_missing)
        quality_per_layer.update(fitted_quality)
        save_alignment_map(
            fitted, src_model, tgt_model,
            source_identity=src_identity, target_identity=tgt_identity,
            quality_per_layer=fitted_quality, extend=bool(cached_sidecar),
        )
        result_map.update(fitted)

    return _with_target_metric(
        {layer: result_map[layer] for layer in sorted(wanted)},
        {
            layer: quality_per_layer[layer]
            for layer in sorted(wanted) if layer in quality_per_layer
        },
        src_identity,
        tgt_identity,
        tgt_acts,
    )
