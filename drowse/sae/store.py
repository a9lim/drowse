"""Drowse-owned sparse-autoencoder artifacts.

Provider-owned SAEs stay in SAELens/Hugging Face caches. This module owns only
SAEs trained by Drowse itself under ``models/<safe>/sae/local/<name>``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from drowse.base.atomic import artifact_lock, fsync_directory, write_json_atomic
from drowse.base.integrity import hash_file, is_sha256_digest, verify_integrity
from drowse.base.source_registry import LOCAL, SourceRef
from drowse.sae.sources import SAE_SOURCES, sae_release

LOCAL_SAE_FORMAT_VERSION = 1
_MANIFEST_FIELDS = {
    "format_version",
    "kind",
    "name",
    "release",
    "model_id",
    "model_fingerprint",
    "model_source_fingerprint",
    "layer",
    "d_model",
    "d_sae",
    "activation",
    "tensor_file",
    "tensor_sha256",
    "corpus_spec",
    "corpus_sha256",
    "tokens_trained",
    "seq_len",
    "batch_size",
    "learning_rate",
    "l1_coefficient",
    "dead_feature_threshold",
}


def local_sae_manifest_path(model_id: str, name: str) -> Path:
    return SAE_SOURCES.local_dir(model_id, name) / "manifest.json"


def _validate_manifest(
    payload: Any,
    model_id: str,
    name: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise ValueError("local SAE manifest has an invalid schema")
    if (
        payload.get("format_version") != LOCAL_SAE_FORMAT_VERSION
        or payload.get("kind") != "local"
        or payload.get("name") != name
        or payload.get("release") != sae_release(SourceRef(LOCAL, name))
        or payload.get("model_id") != model_id
        or payload.get("activation") != "relu"
    ):
        raise ValueError("local SAE manifest has invalid identity metadata")
    for key in ("model_fingerprint", "model_source_fingerprint", "corpus_spec", "corpus_sha256"):
        value = payload.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"local SAE manifest has invalid {key}")
    for key in (
        "layer",
        "d_model",
        "d_sae",
        "tokens_trained",
        "seq_len",
        "batch_size",
    ):
        value = payload.get(key)
        minimum = 0 if key == "layer" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"local SAE manifest has invalid {key}")
    for key in ("learning_rate", "l1_coefficient", "dead_feature_threshold"):
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not torch.isfinite(torch.tensor(float(value))).item()
            or float(value) < 0
        ):
            raise ValueError(f"local SAE manifest has invalid {key}")
    tensor_file = payload.get("tensor_file")
    if (
        not isinstance(tensor_file, str)
        or Path(tensor_file).name != tensor_file
        or not tensor_file.endswith(".safetensors")
    ):
        raise ValueError("local SAE manifest has invalid tensor_file")
    if not is_sha256_digest(payload.get("tensor_sha256")):
        raise ValueError("local SAE manifest has invalid tensor digest")
    return payload


def load_local_sae_manifest(model_id: str, name: str) -> dict[str, Any] | None:
    """The validated manifest of a trained SAE, or ``None``.

    Metadata-only: the schema and the payload's safetensors header (keys,
    shapes, dtype) are checked, but the multi-GB payload is not hashed —
    listing and selection run on every dashboard refresh.  The digest is
    verified where the bytes are actually used, in
    :func:`load_local_sae_tensors`.
    """
    path = local_sae_manifest_path(model_id, name)
    if not path.exists():
        return None
    try:
        payload = _validate_manifest(json.loads(path.read_text()), model_id, name)
        tensor_path = path.parent / payload["tensor_file"]
        if not tensor_path.exists():
            return None
        expected = {
            "W_enc": (payload["d_model"], payload["d_sae"]),
            "W_dec": (payload["d_sae"], payload["d_model"]),
            "b_enc": (payload["d_sae"],),
            "b_dec": (payload["d_model"],),
        }
        with safe_open(str(tensor_path), framework="pt", device="cpu") as tensors:
            if set(tensors.keys()) != set(expected):
                return None
            for key, shape in expected.items():
                view = tensors.get_slice(key)
                if tuple(view.get_shape()) != shape or view.get_dtype() != "F32":
                    return None
        return payload
    except (OSError, ValueError, TypeError, KeyError):
        return None


def load_local_sae_tensors(
    model_id: str,
    name: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]] | None:
    """Load a trained SAE's weights after verifying their recorded digest.

    The digest check rides :func:`verify_integrity`'s stat-identity cache,
    so a warm reload of an unchanged payload does not re-hash it.
    """
    manifest = load_local_sae_manifest(model_id, name)
    if manifest is None:
        return None
    root = SAE_SOURCES.local_dir(model_id, name)
    ok, _bad = verify_integrity(
        root, {manifest["tensor_file"]: manifest["tensor_sha256"]},
    )
    if not ok:
        return None
    tensors = load_file(str(root / manifest["tensor_file"]))
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        return None
    return tensors, manifest


def save_local_sae(
    model_id: str,
    name: str,
    tensors: Mapping[str, torch.Tensor],
    *,
    model_fingerprint: str | None,
    model_source_fingerprint: str | None,
    layer: int,
    corpus_spec: str,
    corpus_sha256: str,
    tokens_trained: int,
    seq_len: int,
    batch_size: int,
    learning_rate: float,
    l1_coefficient: float,
    dead_feature_threshold: float,
    force: bool = False,
    activate: bool = True,
) -> Path:
    """Publish a Drowse-trained SAE and (by default) make it the active source.

    ``activate=False`` writes the artifact without selecting it — the same
    opt-out ``fetch_lens_source`` and ``save_sae_metadata`` have, for a
    caller that trains one SAE while another stays resident.
    """
    required = {"W_enc", "W_dec", "b_enc", "b_dec"}
    if set(tensors) != required:
        raise ValueError(f"local SAE tensors must be exactly {sorted(required)}")
    w_enc = tensors["W_enc"]
    w_dec = tensors["W_dec"]
    if w_enc.ndim != 2 or w_dec.ndim != 2 or tuple(w_dec.shape) != tuple(reversed(w_enc.shape)):
        raise ValueError("local SAE encoder/decoder shapes do not agree")
    d_model, d_sae = (int(w_enc.shape[0]), int(w_enc.shape[1]))
    expected = {
        "W_enc": (d_model, d_sae),
        "W_dec": (d_sae, d_model),
        "b_enc": (d_sae,),
        "b_dec": (d_model,),
    }
    clean: dict[str, torch.Tensor] = {}
    for key, shape in expected.items():
        value = tensors[key].detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f"local SAE tensor {key} is invalid")
        clean[key] = value

    ref = SourceRef(LOCAL, name)
    root = SAE_SOURCES.local_dir(model_id, name)
    manifest_path = root / "manifest.json"
    with artifact_lock(manifest_path):
        if manifest_path.exists() and not force:
            raise FileExistsError(f"local SAE {name!r} already exists; pass --force to replace it")
        root.mkdir(parents=True, exist_ok=True)
        generation = f"layer-{int(layer)}.gen-{os.urandom(12).hex()}.safetensors"
        tensor_path = root / generation
        tmp = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
        save_file(clean, str(tmp))
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, tensor_path)
        digest = hash_file(tensor_path)
        payload = {
            "format_version": LOCAL_SAE_FORMAT_VERSION,
            "kind": "local",
            "name": name,
            "release": sae_release(ref),
            "model_id": model_id,
            "model_fingerprint": model_fingerprint,
            "model_source_fingerprint": model_source_fingerprint,
            "layer": int(layer),
            "d_model": d_model,
            "d_sae": d_sae,
            "activation": "relu",
            "tensor_file": generation,
            "tensor_sha256": digest,
            "corpus_spec": corpus_spec,
            "corpus_sha256": corpus_sha256,
            "tokens_trained": int(tokens_trained),
            "seq_len": int(seq_len),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "l1_coefficient": float(l1_coefficient),
            "dead_feature_threshold": float(dead_feature_threshold),
        }
        _validate_manifest(payload, model_id, name)
        write_json_atomic(manifest_path, payload)
        for old in root.glob("*.safetensors"):
            if old != tensor_path:
                old.unlink(missing_ok=True)
        fsync_directory(root)
    if activate:
        SAE_SOURCES.select(model_id, ref)
    return manifest_path


def remove_local_sae(model_id: str, name: str) -> bool:
    """Delete a Drowse-trained SAE, unpublishing it if it was selected."""
    root = SAE_SOURCES.local_dir(model_id, name)
    manifest = root / "manifest.json"
    with artifact_lock(manifest):
        removed = root.exists()
        if root.exists():
            for path in sorted(root.glob("*")):
                if path.is_file():
                    path.unlink(missing_ok=True)
            try:
                root.rmdir()
            except OSError:
                pass
    # Unpublish after the artifact is gone, so the selection can never outlive
    # what it points at.
    SAE_SOURCES.clear_if_active(model_id, SourceRef(LOCAL, name))
    return removed


def list_local_saes(model_id: str) -> list[dict[str, Any]]:
    active = SAE_SOURCES.active(model_id)
    rows: list[dict[str, Any]] = []
    for name in SAE_SOURCES.local_names(model_id):
        manifest = load_local_sae_manifest(model_id, name)
        if manifest is None:
            continue
        ref = SourceRef(LOCAL, name)
        rows.append(
            {
                "source": SAE_SOURCES.format(ref),
                "kind": LOCAL,
                "name": name,
                "active": active == ref,
                "path": str(local_sae_manifest_path(model_id, name)),
                "layer": manifest["layer"],
                "features": manifest["d_sae"],
            }
        )
    return rows
