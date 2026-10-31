"""Per-(source, target) cross-model alignment-map cache.

A fitted :class:`~drowse.geometry.procrustes.LayerAlignment` map persists under
the *target* model's directory as ``alignments/<safe_src>.json`` (an atomic
pointer) plus one immutable ``left``/``right``/``offset`` safetensors shard per
layer (:mod:`drowse.base.shards`).  The pointer embeds both models' neutral
cache identities (:func:`drowse.store.neutral.neutral_cache_identity`), so a
re-captured neutral cache on either side invalidates the map.  Readers never
mutate the store; superseded generations are collected by the next writer.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import (
    load as load_safetensors,
    save as save_safetensors,
)

from drowse.base.atomic import fsync_directory, write_bytes_atomic, write_json_atomic
from drowse.base.paths import model_dir, safe_model_id
from drowse.base.shards import (
    cleanup_generations,
    fit_lock,
    generation_path,
    json_pointer_matches as _json_pointer_matches,
    shard_paths,
)
from drowse.geometry.procrustes import AlignmentMap, LayerAlignment

log = logging.getLogger(__name__)

_ALIGNMENT_LABEL = "alignment cache"
_ALIGNMENT_CACHE_FORMAT_VERSION = 5
_SHA256 = re.compile(r"[0-9a-f]{64}")

_ALIGNMENT_SIDECAR_FIELDS = {
    "format_version", "method", "source_model_id", "target_model_id",
    "shared_layers", "tensor_schema", "source_neutral_identity",
    "target_neutral_identity", "tensor_files", "tensor_sha256",
    "quality_per_layer",
}


def _alignment_anchor_paths(
    src_model_id: str, tgt_model_id: str,
) -> tuple[Path, Path]:
    """Stable lock anchor and atomic v5 pointer sidecar."""
    md = model_dir(tgt_model_id)
    al_dir = md / "alignments"
    src_safe = safe_model_id(src_model_id)
    return (al_dir / f"{src_safe}.safetensors", al_dir / f"{src_safe}.json")


def _alignment_shard_paths(
    anchor: Path, sidecar: Mapping[str, Any], layers: list[int],
) -> dict[int, Path]:
    return shard_paths(anchor, sidecar, layers, label=_ALIGNMENT_LABEL)


def _cleanup_alignment_generations(anchor: Path, sidecar: Mapping[str, Any]) -> None:
    cleanup_generations(anchor, sidecar, label=_ALIGNMENT_LABEL)


@contextmanager
def alignment_fit_lock(src_model_id: str, tgt_model_id: str) -> Iterator[None]:
    """Single-flight a complete directional alignment fit, including loads."""
    ts_path, _ = _alignment_anchor_paths(src_model_id, tgt_model_id)
    with fit_lock(ts_path):
        yield


def save_alignment_map(
    M: AlignmentMap,
    src_model_id: str,
    tgt_model_id: str,
    *,
    source_identity: dict[str, Any],
    target_identity: dict[str, Any],
    quality_per_layer: dict[int, float] | None = None,
    extend: bool = False,
) -> None:
    """Persist a fitted alignment map.

    Writes one bounded safetensors shard per layer, then atomically switches the
    JSON pointer.  Payload digests are computed from the exact bytes being
    written, so publication never rereads a complete shard.
    """
    anchor, sc_path = _alignment_anchor_paths(src_model_id, tgt_model_id)
    anchor.parent.mkdir(parents=True, exist_ok=True)

    normalized = {int(idx): value for idx, value in M.items()}
    if not normalized:
        raise ValueError("cannot save an empty alignment map")
    from drowse.base.atomic import artifact_lock

    with artifact_lock(anchor):
        tensor_files: dict[str, str] = {}
        tensor_sha256: dict[str, str] = {}
        tensor_schema: dict[str, Any] = {}
        merged_quality: dict[str, float | None] = {}
        if extend and sc_path.exists():
            try:
                with open(sc_path) as handle:
                    current = json.load(handle)
                if (
                    isinstance(current, dict)
                    and set(current) == _ALIGNMENT_SIDECAR_FIELDS
                    and current.get("format_version") == _ALIGNMENT_CACHE_FORMAT_VERSION
                    and current.get("method") == "sharded_factorized_affine_alignment"
                    and current.get("source_model_id") == src_model_id
                    and current.get("target_model_id") == tgt_model_id
                    and current.get("source_neutral_identity") == source_identity
                    and current.get("target_neutral_identity") == target_identity
                ):
                    current_layers = sorted(
                        int(layer) for layer in current.get("shared_layers", [])
                    )
                    current_paths = _alignment_shard_paths(
                        anchor, current, current_layers,
                    )
                    current_digests = current.get("tensor_sha256")
                    current_schema = current.get("tensor_schema")
                    if not isinstance(current_digests, Mapping) or not isinstance(
                        current_schema, Mapping,
                    ):
                        raise ValueError("alignment pointer has no shard metadata")
                    for layer in current_layers:
                        try:
                            digest = current_digests.get(str(layer))
                            spec = current_schema.get(str(layer))
                            path = current_paths[layer]
                            if (
                                not isinstance(digest, str)
                                or not _SHA256.fullmatch(digest)
                                or not isinstance(spec, Mapping) or not path.exists()
                            ):
                                continue
                            # Header-only proof is enough here: this writer is
                            # under the artifact lock and carries the immutable
                            # generation forward byte-for-byte. The newly fitted
                            # mapping below replaces a damaged requested layer.
                            with safe_open(
                                str(path), framework="pt", device="cpu",
                            ) as shard:
                                if set(shard.keys()) != {"left", "right", "offset"}:
                                    continue
                            tensor_files[str(layer)] = path.name
                            tensor_sha256[str(layer)] = digest
                            tensor_schema[str(layer)] = dict(spec)
                        except (OSError, RuntimeError, ValueError, KeyError):
                            continue
                    raw_quality = current.get("quality_per_layer") or {}
                    if isinstance(raw_quality, Mapping):
                        merged_quality.update({
                            str(layer): float(value)
                            for layer, value in raw_quality.items()
                            if value is not None
                        })
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                log.warning("could not extend current alignment generation: %s", exc)
                tensor_files.clear()
                tensor_sha256.clear()
                tensor_schema.clear()
                merged_quality.clear()
        created: list[Path] = []
        try:
            for idx, alignment in sorted(normalized.items()):
                path = generation_path(anchor, idx)
                payload = save_safetensors({
                    "left": alignment.left.contiguous().cpu(),
                    "right": alignment.right.contiguous().cpu(),
                    "offset": alignment.offset.contiguous().cpu(),
                })
                write_bytes_atomic(path, payload)
                created.append(path)
                tensor_files[str(idx)] = path.name
                tensor_sha256[str(idx)] = hashlib.sha256(payload).hexdigest()
                tensor_schema[str(idx)] = {
                    "left": list(alignment.left.shape),
                    "right": list(alignment.right.shape),
                    "offset": list(alignment.offset.shape),
                }
        except BaseException:
            for path in created:
                path.unlink(missing_ok=True)
            raise
        sidecar: dict[str, Any] = {
            "format_version": _ALIGNMENT_CACHE_FORMAT_VERSION,
            "method": "sharded_factorized_affine_alignment",
            "source_model_id": src_model_id,
            "target_model_id": tgt_model_id,
            "shared_layers": sorted(int(layer) for layer in tensor_files),
            "tensor_schema": tensor_schema,
            "source_neutral_identity": source_identity,
            "target_neutral_identity": target_identity,
            "tensor_files": tensor_files,
            "tensor_sha256": tensor_sha256,
            "quality_per_layer": {},
        }
        if quality_per_layer:
            merged_quality.update({
                str(layer): round(float(q), 6)
                for layer, q in quality_per_layer.items()
            })
        for layer in sidecar["shared_layers"]:
            merged_quality.setdefault(str(layer), None)
        sidecar["quality_per_layer"] = merged_quality
        if set(sidecar) != _ALIGNMENT_SIDECAR_FIELDS:
            raise AssertionError("alignment writer produced a non-canonical sidecar")
        try:
            fsync_directory(sc_path.parent)
            write_json_atomic(sc_path, sidecar)
        except BaseException:
            if not _json_pointer_matches(sc_path, sidecar):
                for path in created:
                    path.unlink(missing_ok=True)
            raise
        # The pointer is now authoritative. If directory fsync itself fails,
        # preserve the generations it names; deleting them here would turn a
        # successfully replaced pointer into an immediately broken cache.
        fsync_directory(sc_path.parent)
        _cleanup_alignment_generations(anchor, sidecar)
        fsync_directory(sc_path.parent)


def load_alignment_map(
    src_model_id: str, tgt_model_id: str,
    *,
    source_identity: dict[str, Any],
    target_identity: dict[str, Any],
    requested_layers: Iterable[int] | None = None,
) -> tuple[dict[int, LayerAlignment], dict[str, Any]] | None:
    """Load all or selected factor shards after identity/header preflight.

    ``requested_layers`` materializes only the requested intersection with the
    cache's shared layers.  The returned sidecar remains the complete pointer.
    """

    anchor, sc_path = _alignment_anchor_paths(src_model_id, tgt_model_id)
    from drowse.base.atomic import artifact_lock

    with artifact_lock(anchor):
        if not sc_path.exists():
            return None
        try:
            with open(sc_path) as f:
                sidecar = json.load(f)
            if not isinstance(sidecar, dict) or set(sidecar) != _ALIGNMENT_SIDECAR_FIELDS:
                return None
            # Identity mismatch is the common stale-cache case.  Reject it before
            # hashing or materializing a potentially multi-GiB payload.
            if (
                sidecar.get("format_version") != _ALIGNMENT_CACHE_FORMAT_VERSION
                or sidecar.get("method") != "sharded_factorized_affine_alignment"
                or sidecar.get("source_model_id") != src_model_id
                or sidecar.get("target_model_id") != tgt_model_id
                or sidecar.get("source_neutral_identity") != source_identity
                or sidecar.get("target_neutral_identity") != target_identity
            ):
                return None
            expected_layers = sidecar["shared_layers"]
            schema = sidecar["tensor_schema"]
            if (
                not isinstance(expected_layers, list)
                or not expected_layers
                or any(
                    isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
                    for layer in expected_layers
                )
                or expected_layers != sorted(set(expected_layers))
                or not isinstance(schema, dict)
                or set(schema) != {str(layer) for layer in expected_layers}
                or not isinstance(sidecar["quality_per_layer"], dict)
            ):
                return None
            all_layers = expected_layers
            paths = _alignment_shard_paths(anchor, sidecar, all_layers)
            digests = sidecar.get("tensor_sha256")
            layer_keys = {str(layer) for layer in all_layers}
            quality = sidecar["quality_per_layer"]
            if (
                not isinstance(digests, Mapping)
                or set(digests) != layer_keys
                or set(quality) != layer_keys
                or any(
                    value is not None and (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                    )
                    for value in quality.values()
                )
            ):
                return None
            # Validate declared headers independently before payload materialization.
            # One damaged layer must remain repairable without discarding the
            # pointer metadata for unrelated immutable factors.
            valid_headers: set[int] = set()
            for layer in all_layers:
                spec = schema.get(str(layer), {})
                digest = digests.get(str(layer))
                if (
                    not isinstance(spec, dict)
                    or set(spec) != {"left", "right", "offset"}
                    or any(
                        not isinstance(spec[part], list)
                        or not spec[part]
                        or any(
                            isinstance(dim, bool) or not isinstance(dim, int)
                            or dim <= 0 for dim in spec[part]
                        )
                        for part in ("left", "right", "offset")
                    )
                    or len(spec["left"]) != 2
                    or len(spec["right"]) != 2
                    or len(spec["offset"]) != 1
                    or spec["left"][1] != spec["right"][0]
                    or spec["left"][0] != spec["offset"][0]
                    or not isinstance(digest, str)
                    or not _SHA256.fullmatch(digest)
                    or not paths[layer].exists()
                ):
                    continue
                try:
                    with safe_open(
                        str(paths[layer]), framework="pt", device="cpu",
                    ) as shard:
                        if set(shard.keys()) != {"left", "right", "offset"}:
                            continue
                        if any(
                            list(shard.get_slice(part).get_shape()) != spec.get(part)
                            or shard.get_slice(part).get_dtype() != "F32"
                            for part in ("left", "right", "offset")
                        ):
                            continue
                    valid_headers.add(layer)
                except (OSError, RuntimeError, ValueError, KeyError):
                    continue
            selected = (
                all_layers if requested_layers is None else sorted(
                    set(all_layers) & {int(layer) for layer in requested_layers}
                )
            )
            M: dict[int, LayerAlignment] = {}
            for layer in selected:
                if layer not in valid_headers:
                    continue
                try:
                    payload = paths[layer].read_bytes()
                    if hashlib.sha256(payload).hexdigest() != digests[str(layer)]:
                        continue
                    tensors = load_safetensors(payload)
                    left = tensors["left"]
                    right = tensors["right"]
                    offset = tensors["offset"]
                    if (
                        left.ndim != 2 or right.ndim != 2 or offset.ndim != 1
                        or left.shape[1] != right.shape[0]
                        or left.shape[0] != offset.shape[0]
                        or not all(
                            bool(torch.isfinite(t).all())
                            for t in (left, right, offset)
                        )
                    ):
                        continue
                    M[layer] = LayerAlignment(left, right, offset)
                except (OSError, RuntimeError, ValueError, KeyError):
                    continue
            return M, sidecar
        except Exception as e:
            log.warning(
                "Corrupt alignment map cache (%s → %s): %s",
                src_model_id, tgt_model_id, e,
            )
            return None
