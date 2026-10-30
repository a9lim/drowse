"""Per-model neutral-activation cache.

The model's activations on the bundled neutral statements, one ``[N, D]`` fp32
row block per layer, stored as immutable per-layer shard generations behind an
atomic JSON pointer (:mod:`drowse.base.shards`).  It is the single per-model
neutral artifact: the probe-centering means are its per-layer ``X.mean(0)``,
the :class:`~drowse.geometry.whitener.LayerWhitener` covariance is built from
the stack, and cross-model transfer aligns two models' caches.

Cache identity binds the exact rendered token rows, the loaded model weights
(``model_fingerprint``), the off-model checkpoint identity
(``model_source_fingerprint``), and the layer set.  Readers never mutate the
store; superseded generations are collected by the next writer.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import (
    load as load_safetensors,
    save as save_safetensors,
)

from drowse.base.atomic import fsync_directory, write_bytes_atomic, write_json_atomic
from drowse.base.paths import model_dir
from drowse.base.shards import (
    cleanup_generations,
    fit_lock,
    generation_path,
    json_pointer_matches as _json_pointer_matches,
    shard_paths,
)

log = logging.getLogger(__name__)

_NEUTRAL_LABEL = "neutral activation cache"
_NEUTRAL_ACTS_NAME = "neutral_activations"
_NEUTRAL_CACHE_FORMAT_VERSION = 4
_NEUTRAL_CAPTURE_VERSION = 1

_SHA256 = re.compile(r"[0-9a-f]{64}")

_NEUTRAL_SIDECAR_FIELDS = {
    "method", "format_version", "capture_version", "capture_sha256",
    "model_fingerprint", "model_source_fingerprint", "tensor_sha256",
    "tensor_files", "layers", "tensor_schema", "n_prompts", "n_layers",
}


def _neutral_acts_paths(model_id: str) -> tuple[Path, Path]:
    """Return the stable lock anchor and atomic pointer sidecar."""
    md = model_dir(model_id)
    return (
        md / f"{_NEUTRAL_ACTS_NAME}.safetensors",
        md / f"{_NEUTRAL_ACTS_NAME}.json",
    )


def _neutral_shard_paths(
    anchor: Path, sidecar: Mapping[str, Any], layers: list[int],
) -> dict[int, Path]:
    return shard_paths(anchor, sidecar, layers, label=_NEUTRAL_LABEL)


def _cleanup_neutral_generations(
    anchor: Path, sidecar: Mapping[str, Any],
) -> None:
    cleanup_generations(anchor, sidecar, label=_NEUTRAL_LABEL)


@contextmanager
def neutral_fit_lock(model_id: str) -> Iterator[None]:
    """Single-flight lock for one model's expensive neutral forward pass."""
    ts_path, _ = _neutral_acts_paths(model_id)
    with fit_lock(ts_path):
        yield


def validate_neutral_cache_metadata(model_id: str) -> dict[str, Any]:
    """Validate a neutral cache's identity and tensor headers without paging data.

    ``safe_open`` checks each shard's declared key/shape/dtype from its mmap
    header; payload digests and finiteness are proven by the materializing
    :func:`load_validated_neutral_cache`.  This is the cheap identity preflight
    for exact transfer repeats and source-to-loaded fingerprint bridging.
    """
    from drowse.base.atomic import artifact_lock

    ts_path, _ = _neutral_acts_paths(model_id)
    with artifact_lock(ts_path):
        return _validate_neutral_cache_metadata_locked(model_id)


def _validate_neutral_cache_metadata_locked(model_id: str) -> dict[str, Any]:
    """Header validation with the neutral pair lock already held."""
    ts_path, sc_path = _neutral_acts_paths(model_id)
    if not sc_path.exists():
        raise FileNotFoundError(f"neutral activation cache missing for {model_id}")
    with open(sc_path) as handle:
        sidecar = json.load(handle)
    if not isinstance(sidecar, dict) or set(sidecar) != _NEUTRAL_SIDECAR_FIELDS:
        raise ValueError("neutral activation cache sidecar has a non-current schema")
    version = sidecar["format_version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != _NEUTRAL_CACHE_FORMAT_VERSION
        or isinstance(sidecar["capture_version"], bool)
        or not isinstance(sidecar["capture_version"], int)
        or sidecar["capture_version"] != _NEUTRAL_CAPTURE_VERSION
        or sidecar["method"] != "neutral_activations"
        or any(
            not isinstance(sidecar[key], str) or not sidecar[key]
            for key in (
                "capture_sha256", "model_fingerprint",
                "model_source_fingerprint",
            )
        )
    ):
        raise ValueError("neutral activation cache identity mismatch")
    layers = sidecar["layers"]
    schema = sidecar["tensor_schema"]
    if (
        not isinstance(layers, list)
        or not layers
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in layers
        )
        or layers != sorted(set(layers))
        or not isinstance(schema, dict)
        or set(schema) != {str(layer) for layer in layers}
        or isinstance(sidecar["n_prompts"], bool)
        or not isinstance(sidecar["n_prompts"], int)
        or sidecar["n_prompts"] <= 0
        or isinstance(sidecar["n_layers"], bool)
        or not isinstance(sidecar["n_layers"], int)
        or sidecar["n_layers"] != len(layers)
    ):
        raise ValueError("neutral activation cache sidecar has no layer schema")
    expected_n = sidecar["n_prompts"]
    paths = _neutral_shard_paths(ts_path, sidecar, layers)
    digests = sidecar.get("tensor_sha256")
    if (
        not isinstance(digests, Mapping)
        or set(digests) != {str(layer) for layer in layers}
    ):
        raise ValueError("neutral activation cache has no tensor digests")
    for layer, path in paths.items():
        digest = digests.get(str(layer))
        if (
            not isinstance(digest, str) or not _SHA256.fullmatch(digest)
            or not path.exists()
        ):
            raise ValueError(f"invalid neutral shard for layer {layer}")
        key = f"layer_{layer}"
        spec = schema[str(layer)]
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            if set(tensors.keys()) != {key}:
                raise ValueError("neutral activation cache layer keys do not match sidecar")
            view = tensors.get_slice(key)
            shape = list(view.get_shape())
            dtype = view.get_dtype()
        if (
            not isinstance(spec, dict)
            or set(spec) != {"shape", "dtype"}
            or not isinstance(spec["shape"], list)
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                for dim in spec["shape"]
            )
            or dtype != "F32"
            or spec["dtype"] != "torch.float32"
            or shape != spec["shape"]
            or len(shape) != 2
            or shape[0] != expected_n
        ):
            raise ValueError(
                f"neutral activation cache layer {layer} header failed validation"
            )
    return sidecar


def load_validated_neutral_cache(
    model_id: str,
    *,
    requested_layers: Iterable[int] | None = None,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Materialize and fully validate a neutral cache without loading the model."""
    from drowse.base.atomic import artifact_lock

    ts_path, _ = _neutral_acts_paths(model_id)
    with artifact_lock(ts_path):
        sidecar = _validate_neutral_cache_metadata_locked(model_id)
        all_layers = sorted(int(layer) for layer in sidecar["layers"])
        selected = (
            all_layers if requested_layers is None else sorted(
                set(all_layers) & {int(layer) for layer in requested_layers}
            )
        )
        paths = _neutral_shard_paths(ts_path, sidecar, all_layers)
        digests = sidecar["tensor_sha256"]
        tensors: dict[str, torch.Tensor] = {}
        for layer in selected:
            payload = paths[layer].read_bytes()
            if hashlib.sha256(payload).hexdigest() != digests[str(layer)]:
                raise ValueError("neutral activation cache payload digest mismatch")
            shard = load_safetensors(payload)
            tensors[f"layer_{layer}"] = shard[f"layer_{layer}"]
    out: dict[int, torch.Tensor] = {}
    schema = sidecar["tensor_schema"]
    expected_n = int(sidecar.get("n_prompts", -1))
    for key, tensor in tensors.items():
        layer = int(key.split("_", 1)[1])
        expected_shape = schema.get(str(layer), {}).get("shape")
        if (
            tensor.dtype != torch.float32
            or list(tensor.shape) != expected_shape
            or tensor.ndim != 2
            or int(tensor.shape[0]) != expected_n
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"neutral activation cache layer {layer} failed validation")
        out[layer] = tensor
    return out, sidecar


def neutral_cache_identity(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Stable identity subset embedded into dependent alignment artifacts."""
    keys = (
        "format_version", "capture_version", "model_fingerprint",
        "model_source_fingerprint",
        "capture_sha256", "tensor_sha256", "layers", "tensor_schema",
        "n_prompts",
    )
    required = {*keys, "tensor_files", "method"}
    if (
        not required <= set(sidecar)
        or sidecar["format_version"] != _NEUTRAL_CACHE_FORMAT_VERSION
        or sidecar["method"] != "neutral_activations"
    ):
        raise ValueError("neutral cache identity requires current validated metadata")
    identity = {key: sidecar[key] for key in keys}
    identity["tensor_files"] = sidecar["tensor_files"]
    return identity


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
    :func:`load_validated_neutral_cache`.  The activation-only public API above
    remains the convenience surface for ordinary whitener/probe callers.
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
    from drowse.model.fingerprint import loaded_model_fingerprint
    from drowse.fit.capture import (
        _neutral_pairs,
        _prepare_capture_batch,
        compute_neutral_activations,
    )

    ts_path, sc_path = _neutral_acts_paths(model_id)
    md = model_dir(model_id)
    md.mkdir(parents=True, exist_ok=True)

    pairs = _neutral_pairs()
    prompts = [prompt for prompt, _ in pairs]
    responses = [response for _, response in pairs]
    prepared = _prepare_capture_batch(
        tokenizer, prompts, responses,
    )
    rendered_payload = [
        {"input_ids": ids[0].tolist(), "content_end": int(content_end)}
        for ids, content_end in prepared
    ]
    capture_sha = hashlib.sha256(json.dumps(
        {"capture_version": _NEUTRAL_CAPTURE_VERSION, "rows": rendered_payload},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    model_fingerprint = loaded_model_fingerprint(model, model_id)
    base_model = getattr(model, "_orig_mod", model)
    model_source_fingerprint = getattr(
        base_model, "_drowse_source_fingerprint", None,
    )
    expected_layers = list(range(len(layers)))

    if not force and sc_path.exists():
        try:
            out, sc = load_validated_neutral_cache(model_id)
            identity_matches = (
                sc.get("format_version") == _NEUTRAL_CACHE_FORMAT_VERSION
                and sc.get("capture_version") == _NEUTRAL_CAPTURE_VERSION
                and sc.get("capture_sha256") == capture_sha
                and sc.get("model_fingerprint") == model_fingerprint
                and sc.get("model_source_fingerprint") == model_source_fingerprint
                and sc.get("layers") == expected_layers
            )
            if identity_matches:
                log.debug("Loaded cached neutral activations for %s", model_id)
                return out, sc
            else:
                log.info(
                    "Neutral activations stale (model/token/corpus/layer identity changed); "
                    "recomputing for %s",
                    model_id,
                )
        except Exception as e:
            log.warning("Corrupt neutral activations cache for %s, recomputing: %s", model_id, e)

    log.info("Computing neutral activations (one-time per model)...")
    if on_progress is not None:
        on_progress(
            f"Computing neutral activations for {model_id} "
            f"(one-time per model, {len(pairs)} prompts)..."
        )
    activations = compute_neutral_activations(
        model, tokenizer, layers, rendered=prepared, on_progress=on_progress,
    )

    # Persist fp32 — the whitener covariance is built and inverted from these,
    # so the compute path (this fresh fp32 store) and the cache-hit path stay
    # bit-identical (no precision seam) and the inversion is full-precision.
    # fp32 carries the exponent range fp16 lacked (gemma-3's late layers exceed
    # the fp16 max of 65504 and overflowed to ±inf) without bf16's input loss,
    # at ~2× disk on this one cache.
    fp32 = {
        int(idx): t.contiguous().to(torch.float32).cpu()
        for idx, t in activations.items()
    }
    from drowse.base.atomic import artifact_lock

    with artifact_lock(ts_path):
        tensor_files: dict[str, str] = {}
        tensor_sha256: dict[str, str] = {}
        created: list[Path] = []
        try:
            for idx, tensor in sorted(fp32.items()):
                path = generation_path(ts_path, idx)
                payload = save_safetensors({f"layer_{idx}": tensor})
                write_bytes_atomic(path, payload)
                created.append(path)
                tensor_files[str(idx)] = path.name
                tensor_sha256[str(idx)] = hashlib.sha256(payload).hexdigest()
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        sidecar: dict[str, Any] = {
            "method": "neutral_activations",
            "format_version": _NEUTRAL_CACHE_FORMAT_VERSION,
            "capture_version": _NEUTRAL_CAPTURE_VERSION,
            "capture_sha256": capture_sha,
            "model_fingerprint": model_fingerprint,
            "model_source_fingerprint": model_source_fingerprint,
            "tensor_sha256": tensor_sha256,
            "tensor_files": tensor_files,
            "layers": sorted(activations),
            "tensor_schema": {
                str(idx): {"shape": list(t.shape), "dtype": str(t.dtype)}
                for idx, t in sorted(activations.items())
            },
            "n_prompts": (
                next(iter(activations.values())).shape[0] if activations else 0
            ),
            "n_layers": len(activations),
        }
        if set(sidecar) != _NEUTRAL_SIDECAR_FIELDS:
            raise AssertionError("neutral writer produced a non-canonical sidecar")
        try:
            # Persist every new directory entry before publishing the pointer.
            # A crash can therefore expose either the complete old generation
            # or the complete new one, never a pointer to absent shards.
            fsync_directory(sc_path.parent)
            write_json_atomic(sc_path, sidecar)
        except BaseException:
            if not _json_pointer_matches(sc_path, sidecar):
                for path in created:
                    path.unlink(missing_ok=True)
            raise
        # The pointer rename is now authoritative. Preserve its generations if
        # this durability barrier itself fails; deleting them would break the
        # successfully replaced pointer immediately.
        fsync_directory(sc_path.parent)
        _cleanup_neutral_generations(ts_path, sidecar)
        fsync_directory(sc_path.parent)
    return activations, sidecar
