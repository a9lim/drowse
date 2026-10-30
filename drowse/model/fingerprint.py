"""Source and loaded-state identities for cache keys.

``model_source_fingerprint`` proves a checkpoint's identity before any weight
is read (Hub commit or exact local file hashes); ``loaded_model_fingerprint``
identifies the weights a live module actually holds.
"""

from __future__ import annotations

import hashlib
import json
import threading
import weakref
from pathlib import Path
from typing import Any, Iterable, cast

import torch
from transformers import AutoConfig

from drowse.model.arch import text_extractable
from drowse.model.device import detect_device, resolve_dtype

_MODEL_FILE_HASH_CACHE_MAX = 4096
_MODEL_FILE_HASH_CACHE: dict[str, tuple[int, int, int, int, int, str]] = {}
_MODEL_FINGERPRINT_CACHE: "weakref.WeakKeyDictionary[Any, tuple[object, str, str]]" = (
    weakref.WeakKeyDictionary()
)
_MODEL_FINGERPRINT_LOCK = threading.Lock()
_SOURCE_FINGERPRINT_VERSION = 1


def _model_state_signature(model: Any) -> object:
    """Cheap mutation signature for memoizing an exact loaded-state digest."""
    config = getattr(model, "config", None)
    config_payload = (
        config.to_dict() if config is not None and hasattr(config, "to_dict")
        else repr(config)
    )
    rows: list[tuple[object, ...]] = []
    named_parameters = getattr(model, "named_parameters", None)
    parameter_rows: Any
    if callable(named_parameters):
        parameter_rows = named_parameters()
    else:
        parameters = getattr(model, "parameters", None)
        parameters_any: Any = parameters
        parameter_rows = (
            (
                (str(index), tensor)
                for index, tensor in enumerate(
                    cast(Iterable[Any], parameters_any())
                )
            )
            if callable(parameters_any) else ()
        )
    named_buffers = getattr(model, "named_buffers", None)
    buffer_rows: Any = named_buffers() if callable(named_buffers) else ()
    for kind, tensors in (
        ("parameter", parameter_rows),
        ("buffer", buffer_rows),
    ):
        for name, tensor in tensors:
            try:
                pointer = int(tensor.data_ptr())
            except (RuntimeError, TypeError, AttributeError):
                pointer = id(tensor)
            rows.append((
                kind, name, tuple(int(dim) for dim in tensor.shape),
                str(tensor.dtype), str(tensor.device), pointer,
                int(getattr(tensor, "_version", 0)),
            ))
    return (
        hashlib.sha256(json.dumps(
            config_payload, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")).hexdigest(),
        tuple(rows),
    )


def invalidate_loaded_model_fingerprint(model: Any) -> None:
    """Mark out-of-band weight/storage mutation before persistent-cache use.

    Drowse-owned models are immutable after load; steering is per-call and does
    not mutate parameters. Integrations that replace adapters or write through
    ``Tensor.data`` must call this hook because those writes can bypass
    PyTorch's tensor ``_version`` counter.
    """
    base = getattr(model, "_orig_mod", model)
    setattr(base, "_drowse_state_invalidated", True)
    with _MODEL_FINGERPRINT_LOCK:
        _MODEL_FINGERPRINT_CACHE.pop(base, None)


def _local_model_files(path: Path) -> list[tuple[str, int, str]]:
    """Exact local checkpoint/config identity with process-local hash reuse."""
    if not path.is_dir():
        return []
    from drowse.base.integrity import hash_file

    found: dict[str, Path] = {}
    ignored_parts = {".git", ".locks", "__pycache__"}
    for file_path in path.rglob("*"):
        relative_path = file_path.relative_to(path)
        if (
            file_path.is_file()
            and not any(part in ignored_parts for part in relative_path.parts)
            and file_path.suffix not in {".lock", ".tmp", ".pyc"}
        ):
            # trust_remote_code and custom tokenizers may consume arbitrary
            # adjacent resources. A filename whitelist cannot prove source
            # identity; hash every stable regular snapshot file instead.
            found[str(relative_path)] = file_path
    result: list[tuple[str, int, str]] = []
    for relative, file_path in sorted(found.items()):
        stat = file_path.stat()
        key = str(file_path.resolve())
        cached = _MODEL_FILE_HASH_CACHE.pop(key, None)
        # Size+mtime can be restored after an in-place same-size rewrite.
        # ctime catches that rewrite; device+inode also distinguish atomic
        # replacement at the same path without forcing another full hash when
        # the snapshot is genuinely unchanged.
        identity = (
            int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns),
            int(stat.st_dev), int(stat.st_ino),
        )
        if cached is not None and cached[:-1] == identity:
            file_hash = cached[-1]
            _MODEL_FILE_HASH_CACHE[key] = cached
        else:
            file_hash = hash_file(file_path)
            while len(_MODEL_FILE_HASH_CACHE) >= _MODEL_FILE_HASH_CACHE_MAX:
                _MODEL_FILE_HASH_CACHE.pop(next(iter(_MODEL_FILE_HASH_CACHE)))
            _MODEL_FILE_HASH_CACHE[key] = (*identity, file_hash)
        result.append((relative, int(stat.st_size), file_hash))
    return result


def resolve_model_revision(model_id: str) -> str | None:
    """The immutable Hub commit ``model_id``'s default branch points to.

    Transformers 5.18 stopped recording the resolved commit on configs (it
    drops ``_commit_hash`` at construction), so Drowse resolves it itself and
    pins every later read to it.  Online this asks the Hub and records the
    ref in the cache; offline ``huggingface_hub`` answers from that cached
    ref.  A local directory has no commit (it is identified by file hashes)
    and returns ``None``.  Hub failures propagate — a missing, gated, or
    unreachable repo raises with ``huggingface_hub``'s own message.
    """
    if Path(model_id).expanduser().is_dir():
        return None
    from huggingface_hub import HfApi

    return str(HfApi().resolve_revision(model_id).resolved)


def _pinned_revision(model_id: str) -> str | None:
    """:func:`resolve_model_revision`, failing open to an unpinned read."""
    try:
        return resolve_model_revision(model_id)
    except Exception:
        # Transformers' own resolver fails open the same way: the config read
        # that follows reports the repo problem in its own words, or serves
        # an already-cached snapshot.
        return None


def model_source_fingerprint(
    model_id: str, *, quantize: str | None = None, device: str = "auto",
    dtype: torch.dtype | str | None = None, config: Any | None = None,
    revision: str | None = None,
    parameter_dtype: torch.dtype | str | None = None,
) -> str | None:
    """Pre-load-verifiable identity for a checkpoint and load representation.

    Hub models are pinned by the resolved immutable commit; local models by
    exact checkpoint/config/tokenizer file hashes.  ``None`` means the source
    cannot be proven without loading weights, so callers must decline any
    metadata-only cache shortcut.

    ``config`` and ``revision`` describe one snapshot: the loader passes both
    from its plan; a preflight passes neither, and the commit is resolved
    here with the config read pinned to it.

    The hashed config payload is canonicalized to the *post-load* view: on the
    multimodal text-extraction path the load mutates the plan config in place
    (``_materialize_model`` fills ``text_config._name_or_path`` and
    ``from_config`` writes the load dtype onto ``text_config``), and
    ``_finalize_model`` stamps from that mutated object.  Applying the same
    two writes to the payload dict here is a no-op on the stamp side and makes
    a pre-load recompute reproduce the stamp, so metadata-only preflights can
    prove identity without loading weights.
    """
    try:
        resolved_device = detect_device(device)
        effective_quantize = quantize if resolved_device == "cuda" else None
        resolved_load_dtype = resolve_dtype(dtype, resolved_device)
        if config is None:
            if revision is None:
                revision = _pinned_revision(model_id)
            config = AutoConfig.from_pretrained(
                model_id, trust_remote_code=False,
                **({"revision": revision} if revision else {}),
            )
        resolved_commit = revision
        requested_path = Path(model_id).expanduser()
        raw_config_path = str(getattr(config, "_name_or_path", "")).strip()
        config_path = Path(raw_config_path).expanduser() if raw_config_path else None
        local_path = requested_path if requested_path.is_dir() else config_path
        local_files = _local_model_files(local_path) if local_path is not None else []
        if not resolved_commit and not local_files:
            return None
        effective_dtype = str(
            parameter_dtype
            if parameter_dtype is not None
            else (
                "quantized" if effective_quantize is not None
                else resolved_load_dtype
            )
        )
        canonical_id = (
            str(requested_path.resolve()) if requested_path.exists() else model_id
        )
        config_payload: dict[str, Any] = (
            config.to_dict() if hasattr(config, "to_dict")
            else {"model_type": getattr(config, "model_type", None)}
        )
        text_cfg = getattr(config, "text_config", None)
        text_payload = config_payload.get("text_config")
        if text_extractable(text_cfg) and isinstance(text_payload, dict):
            # Same predicate as ``LoadPlan.extract_text_model``: this config
            # takes the text-extraction path, whose in-place plan-config
            # mutations the canonical payload must carry (see docstring).
            # The extraction always hands ``from_config`` the plan-resolved
            # dtype, even under quantization.
            if not text_payload.get("_name_or_path"):
                text_payload["_name_or_path"] = model_id
            for key in ("dtype", "torch_dtype"):
                if key in text_payload:
                    text_payload[key] = str(resolved_load_dtype).removeprefix(
                        "torch.",
                    )
        payload = {
            "version": _SOURCE_FINGERPRINT_VERSION,
            "model_id": canonical_id,
            "resolved_commit": resolved_commit,
            "local_files": local_files,
            "config": config_payload,
            "quantize": effective_quantize,
            "parameter_dtype": effective_dtype,
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")).hexdigest()
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError):
        return None


def loaded_model_fingerprint(model: Any, model_id: str) -> str:
    """Stable cache identity for the weights loaded under ``model_id``.

    A Hugging Face id alone is mutable (and local paths are more so). Fold the
    resolved config/commit and exact local shard hashes together with distributed
    samples from *every* exposed parameter. When neither immutable source exists
    (an in-memory/opaque model), stream an exact full-state hash in bounded host
    chunks rather than permit an unprovable persistent cache hit.
    """
    base = getattr(model, "_orig_mod", model)
    config = getattr(base, "config", getattr(model, "config", None))
    config_payload = (
        config.to_dict() if config is not None and hasattr(config, "to_dict")
        else {
            "model_type": getattr(config, "model_type", None),
            "name_or_path": getattr(config, "_name_or_path", None),
            "commit": getattr(config, "_commit_hash", None),
        }
    )
    state_signature = _model_state_signature(base)
    with _MODEL_FINGERPRINT_LOCK:
        cached = _MODEL_FINGERPRINT_CACHE.get(base)
    if cached is not None and cached[:2] == (state_signature, model_id):
        return cached[2]
    # ``named_*`` follows module/registration order, which is not an identity
    # property.  Some HF modules build equivalent buffer registries from sets
    # (Gemma-3's shared rotary buffers are the live case), so two identical
    # loads can expose the same names in a different order.  Canonicalize once
    # and reuse that ordering for both the schema and the exact-state fallback.
    named_parameters = sorted(base.named_parameters(), key=lambda item: item[0])
    named_buffers = sorted(base.named_buffers(), key=lambda item: item[0])
    parameter_schema = [
        (name, tuple(int(dim) for dim in parameter.shape), str(parameter.dtype))
        for name, parameter in named_parameters
    ]
    buffer_schema = [
        (name, tuple(int(dim) for dim in buffer.shape), str(buffer.dtype))
        for name, buffer in named_buffers
    ]
    trusted_source = getattr(base, "_drowse_source_fingerprint", None)
    payload = {
        "model_id": model_id,
        "class": f"{type(base).__module__}.{type(base).__qualname__}",
        "config": config_payload,
        "trusted_source": trusted_source,
        "parameter_schema": parameter_schema,
        "buffer_schema": buffer_schema,
    }
    digest = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8"))
    loaded_signature = getattr(base, "_drowse_loaded_state_signature", None)
    if (
        not trusted_source
        or state_signature != loaded_signature
        or bool(getattr(base, "_drowse_state_invalidated", False))
    ):
        # No trusted load provenance exists (programmatically constructed,
        # separately initialized, externally wrapped, or mutated after drowse
        # loaded it). Persistent reuse is safe only with an exact state hash.
        # Stream bounded chunks instead of cloning the whole model. Clean
        # drowse-owned loads use the exact immutable source fingerprint above.
        tensors = named_parameters + named_buffers
        for name, tensor in tensors:
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(repr(tuple(int(d) for d in tensor.shape)).encode("utf-8"))
            flat = tensor.detach().reshape(-1)
            for start in range(0, flat.numel(), 1_048_576):
                chunk = flat[start:start + 1_048_576].to(
                    device="cpu",
                ).contiguous()
                # Byte-view preserves the original dtype bit pattern (including
                # float64 ULPs and integer/bool buffers); float32 promotion would
                # make distinct live states collide.
                digest.update(chunk.view(torch.uint8).numpy().tobytes())
    fingerprint = digest.hexdigest()
    with _MODEL_FINGERPRINT_LOCK:
        _MODEL_FINGERPRINT_CACHE[base] = (state_signature, model_id, fingerprint)
    return fingerprint
