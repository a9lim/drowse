"""Hardened safetensors reading and multimodal text-model extraction.

The loader's one hand-written checkpoint path: a bounded, duplicate-key-
rejecting safetensors header parser, an uncached ``pread`` tensor reader, a
shard-index validator, and the text-only extraction that builds a multimodal
checkpoint's language model on-device shard by shard.
"""

from __future__ import annotations

import json
import os
import stat
from contextlib import suppress
from pathlib import PureWindowsPath
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from drowse.errors import DrowseError

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2,
               torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)

# safetensors dtype tag -> torch dtype, for the manual (mmap-free) reader.
_ST_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}
_ST_DTYPE_BITS = {
    "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6,
    "BOOL": 8, "U8": 8, "I8": 8, "F8_E4M3": 8, "F8_E5M2": 8,
    "F8_E8M0": 8, "F8_E4M3FNUZ": 8, "F8_E5M2FNUZ": 8,
    "I16": 16, "U16": 16, "F16": 16, "BF16": 16,
    "I32": 32, "U32": 32, "F32": 32,
    "I64": 64, "U64": 64, "F64": 64, "C64": 64,
}
_ST_HEADER_MAX_BYTES = 100_000_000
_CHECKPOINT_INDEX_MAX_BYTES = 64 * 1024 * 1024


def _checkpoint_json(raw: bytes) -> Any:
    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate checkpoint metadata key {key!r}")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=unique_keys)


def checkpoint_shard_names(index_path: str) -> list[str]:
    with os.fdopen(open_uncached(index_path), "rb") as handle:
        raw = handle.read(_CHECKPOINT_INDEX_MAX_BYTES + 1)
    if len(raw) > _CHECKPOINT_INDEX_MAX_BYTES:
        raise ValueError("checkpoint shard index exceeds the metadata limit")
    index = _checkpoint_json(raw)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint shard index requires a non-empty weight_map")
    names: set[str] = set()
    for name in weight_map.values():
        if (
            not isinstance(name, str) or not name.endswith(".safetensors")
            or "\\" in name or "\x00" in name or PureWindowsPath(name).drive
            or any(part in {"", ".", ".."} for part in name.split("/"))
        ):
            raise ValueError(f"unsafe checkpoint shard name {name!r}")
        names.add(name)
    return sorted(names)


def read_safetensors_header(path: str) -> tuple[dict[str, Any], int]:
    """Parse a safetensors header: ``(tensor_metadata, data_section_offset)``.

    The format is an 8-byte little-endian header length, that many bytes
    of JSON (name → ``{dtype, shape, data_offsets}``), then the packed
    tensor data.  The ``__metadata__`` entry is dropped.
    """
    with os.fdopen(open_uncached(path), "rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        prefix = f.read(8)
        n = int.from_bytes(prefix, "little")
        if len(prefix) != 8 or not 0 < n <= _ST_HEADER_MAX_BYTES or n > file_size - 8:
            raise ValueError("invalid or oversized safetensors header")
        header = _checkpoint_json(f.read(n))
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    header.pop("__metadata__", None)
    data_bytes = file_size - 8 - n
    intervals = []
    for name, meta in header.items():
        if not isinstance(meta, dict) or set(meta) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"invalid safetensors metadata for {name!r}")
        dtype, shape, offsets = meta["dtype"], meta["shape"], meta["data_offsets"]
        if (
            not isinstance(dtype, str) or dtype not in _ST_DTYPE_BITS
            or not isinstance(shape, list)
            or any(type(dim) is not int or not 0 <= dim < 2**64 for dim in shape)
            or not isinstance(offsets, list) or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
        ):
            raise ValueError(f"invalid safetensors shape/dtype/offsets for {name!r}")
        start, end = offsets
        expected_bits = 0 if 0 in shape else _ST_DTYPE_BITS[dtype]
        for dim in shape:
            expected_bits *= dim
            if expected_bits > data_bytes * 8:
                raise ValueError(f"invalid safetensors byte extent for {name!r}")
        if (
            not 0 <= start <= end <= data_bytes or expected_bits % 8
            or end - start != expected_bits // 8
        ):
            raise ValueError(f"invalid safetensors byte extent for {name!r}")
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("safetensors data has overlapping or non-contiguous tensors")
        cursor = end
    if cursor != data_bytes:
        raise ValueError("safetensors data does not match its declared tensors")
    return header, 8 + n


def open_uncached(path: str) -> int:
    """Open ``path`` read-only, asking the OS not to cache its pages.

    On Apple Silicon the page cache shares the same physical RAM as MPS,
    so mmap-reading a 62 GB checkpoint fills the cache *on top of* the
    61 GB on-device model — ~2× the model size, which forces the unified
    pool into compression/swap.  ``F_NOCACHE`` (macOS) reads straight
    through without populating the unified buffer cache, holding the
    host-side transient to a single tensor.  Elsewhere the page cache is
    reclaimable and doesn't compete with discrete VRAM, so a plain fd is
    fine.
    """
    import os
    import sys
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"checkpoint {path!r} is not a regular file")
    except BaseException:
        os.close(fd)
        raise
    if sys.platform == "darwin":
        import fcntl
        with suppress(OSError):
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    return fd


def read_st_tensor(fd: int, header: dict[str, Any], data_start: int, name: str) -> torch.Tensor:
    """Read one tensor by name from an already-open safetensors fd.

    ``pread`` at the tensor's recorded offset, wrap the bytes as a CPU
    tensor (zero-copy view of the freshly-read buffer), reshape.  No mmap,
    so with an :func:`open_uncached` fd the bytes never enter the page
    cache.
    """
    import os
    import warnings
    meta = header[name]
    dtype = _ST_DTYPES.get(meta["dtype"])
    if dtype is None:
        raise ValueError(f"unsupported safetensors dtype {meta['dtype']!r} for selected tensor {name!r}")
    start, end = meta["data_offsets"]
    nbytes = end - start
    offset = data_start + start
    # A single pread can't exceed INT_MAX bytes on macOS (EINVAL), and a
    # bf16 embedding is ~2.8 GB, so read in capped chunks.
    _CHUNK = 1 << 28  # 256 MiB
    chunks: list[bytes] = []
    got = 0
    while got < nbytes:
        block = os.pread(fd, min(_CHUNK, nbytes - got), offset + got)
        if not block:
            break
        chunks.append(block)
        got += len(block)
    if got != nbytes:
        raise ValueError(f"safetensors tensor {name!r} is truncated")
    if nbytes == 0:
        return torch.empty(meta["shape"], dtype=dtype)
    buf = chunks[0] if len(chunks) == 1 else b"".join(chunks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # frombuffer warns on read-only bytes
        flat = torch.frombuffer(buf, dtype=dtype)
    return flat.reshape(meta["shape"])


class NoTextWeightsExtracted(DrowseError, RuntimeError):
    """No text-model weights matched the ``language_model.`` layout.

    Raised by :func:`load_text_from_multimodal` so the caller can fall
    back to a standard full-model load rather than return a model full of
    random initialization.
    """


def load_text_from_multimodal(
    model_id: str,
    text_config: Any,
    dtype: torch.dtype,
    device: str,
    revision: str | None = None,
):
    """Load just the text model from a multimodal checkpoint.

    Multimodal checkpoints store the text-model weights under a
    ``language_model.`` path *segment* — either as a leading prefix
    (Mistral-3 / Ministral: ``language_model.model.layers…``) or nested
    inside the composite model (Gemma-3 / Gemma-4:
    ``model.language_model.layers…``).  This function builds a text-only
    model directly on the target device, then streams each safetensors
    shard tensor-by-tensor, drops the ``language_model.`` segment to
    recover the text model's own parameter names, dequantizes FP8
    weights, and copies each into the preallocated parameter. A top-level
    ``lm_head`` is retained; unrelated vision tower / projector weights
    are skipped.

    Two memory disciplines matter on Apple Silicon, where CPU and MPS
    share one unified-memory pool:

    * **On-device construction** — ``from_config`` under
      ``torch.device(device)`` allocates the model once, on the target
      device, so there is never a second full-model copy.  The standard
      ``from_pretrained(device_map={"": "mps"})`` path stages the entire
      state dict in CPU RAM *and* copies it to MPS before freeing the CPU
      side — ~2× the model size at peak (>128 GB for gemma-4-31B).
    * **Cache-bypassing reads** — each shard is read with ``pread`` on an
      :func:`open_uncached` fd (``F_NOCACHE`` on macOS) rather than
      mmap'd via ``safe_open``.  mmap would fault the whole 62 GB
      checkpoint into the page cache, which on unified memory collides
      with the on-device model; reading straight through holds the
      host-side transient to a single tensor.

    Raises :class:`NoTextWeightsExtracted` if nothing matched, so the
    caller can fall back to a full multimodal load.
    """
    import gc
    import os
    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
    from transformers.utils.hub import cached_file

    # Prefer the sharded index; fall back to a single `model.safetensors`
    # for repos that ship consolidated weights (e.g. Ministral-3-3B).
    index_path = cached_file(
        model_id, SAFE_WEIGHTS_INDEX_NAME, revision=revision,
        _raise_exceptions_for_missing_entries=False,
    )
    if index_path is not None:
        shard_names = checkpoint_shard_names(index_path)
        # Resolve each shard through cached_file so HF hub downloads it
        # if it isn't already local (the index can land before the shards).
        shard_paths = [
            cached_file(model_id, name, revision=revision)
            for name in shard_names
        ]
    else:
        single_path = cached_file(model_id, SAFE_WEIGHTS_NAME, revision=revision)
        shard_paths = [single_path]

    # Create directly on target device — avoids a CPU copy that would
    # spike RSS and eat into MPS's unified memory budget.
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(text_config, dtype=dtype)

    # named_parameters de-duplicates tied storage. Nonpersistent buffers
    # may legitimately be initialized by the model rather than the checkpoint.
    targets: dict[str, torch.Tensor] = dict(model.named_parameters())
    required = set(targets)
    targets.update(model.named_buffers())
    loaded: set[str] = set()
    matched = 0
    with torch.no_grad():
        for sf in shard_paths:
            # cached_file returns str | None; None means a missing shard —
            # skip it. In practice the single-shard path raises before here.
            if sf is None:
                continue
            header, data_start = read_safetensors_header(sf)
            # Read in on-disk offset order so an uncached fd still streams
            # sequentially (no per-tensor seek thrash).
            names = sorted(header, key=lambda nm: header[nm]["data_offsets"][0])
            fd = open_uncached(sf)
            try:
                for name in names:
                    if name in {"lm_head.weight", "lm_head.bias"}:
                        key = name
                    elif "language_model." in name:
                        # Both language_model.model.layers… and
                        # model.language_model.layers… map to model.layers….
                        key = name.replace("language_model.", "", 1)
                    else:
                        continue  # vision tower / projector — skip
                    if key.endswith((".weight_scale_inv", ".activation_scale")):
                        continue
                    target = targets.get(key)
                    if target is None:
                        continue
                    if tuple(header[name]["shape"]) != tuple(target.shape):
                        raise ValueError(f"checkpoint tensor {name!r} has the wrong shape")
                    v = read_st_tensor(fd, header, data_start, name)
                    # Dequantize FP8: real_weight = weight.to(dtype) * scale
                    if v.dtype in _FP8_DTYPES:
                        sk = name + "_scale_inv"
                        scale = (read_st_tensor(fd, header, data_start, sk)
                                 if sk in header else None)
                        v = v.to(dtype) * scale.to(dtype) if scale is not None else v.to(dtype)
                    target.copy_(v)  # CPU→device cast + copy into preallocated param
                    loaded.add(key)
                    if "language_model." in name:
                        matched += 1
                    del v
            finally:
                os.close(fd)
            if device == "mps":
                torch.mps.empty_cache()
            gc.collect()

    if matched == 0:
        raise NoTextWeightsExtracted(
            f"no text-model weights matched the language_model. layout in {model_id!r}"
        )
    missing = required - loaded
    if missing:
        raise ValueError(f"checkpoint is missing text-model parameters: {sorted(missing)!r}")

    if device == "mps":
        torch.mps.empty_cache()

    return model
