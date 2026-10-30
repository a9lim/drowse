"""Content-addressed activation capture cache for manifold fits.

A fit's residual activations depend only on its capture identity —
``capture_sha256``: the loaded model, the node partition, and every rendered
token row — not on fit geometry, hyperparameters, layer scope, or which folder
asked.  Captures are therefore cached per model under that digest and shared by
every folder whose corpus renders the same rows, as one self-describing
safetensors shard per kind and layer under ``models/<safe_model>/manifold_capture/``:

- ``<sha>.centroids.layer_<L>.safetensors`` — the ``(K, D)`` fp32 node centroids;
- ``<sha>.rows.layer_<L>.safetensors`` — the ``(N, D)`` source-dtype response
  rows a curved fit's σ-field reads.

Each shard is fsynced and published by atomic replace, and carries its own
identity (format, digest, node sizes) in the safetensors header.  There is no
pointer, journal, or generation: a shard that is missing, truncated, foreign,
or malformed is a per-layer miss, and the worst failure is a recapture.  A
size bound evicts whole digests, least recently written first.
"""
from __future__ import annotations

import json
import os
import struct
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save as serialize_safetensors

from drowse.base import atomic
from drowse.base.atomic import fsync_directory, write_bytes_atomic
from drowse.base.paths import model_dir

if TYPE_CHECKING:
    from drowse.store.manifold_tensors import ActivationRowStore

CAPTURE_CACHE_FORMAT = "5"

_DTYPE_NAMES = {
    torch.bool: "BOOL", torch.uint8: "U8", torch.int8: "I8",
    torch.int16: "I16", torch.int32: "I32", torch.int64: "I64",
    torch.float16: "F16", torch.bfloat16: "BF16",
    torch.float32: "F32", torch.float64: "F64",
}
_READ_ERRORS = (OSError, ValueError, RuntimeError, KeyError, SafetensorError)
_STREAM_CHUNK_BYTES = 2_097_152


def capture_cache_dir(model_id: str) -> Path:
    return model_dir(model_id) / "manifold_capture"


class CaptureCache:
    """The cached captures of one token identity on one model."""

    def __init__(
        self, model_id: str, capture_sha256: str, node_sizes: Sequence[int],
    ) -> None:
        self.root = capture_cache_dir(model_id)
        self.stem = capture_sha256
        self.node_sizes = [int(size) for size in node_sizes]
        self._metadata = {
            "format": CAPTURE_CACHE_FORMAT,
            "capture_sha256": capture_sha256,
            "node_sizes": json.dumps(self.node_sizes),
        }

    def path(self, kind: str, layer: int) -> Path:
        return self.root / f"{self.stem}.{kind}.layer_{int(layer)}.safetensors"

    @contextmanager
    def capturing(self) -> Iterator[None]:
        """Serialize capture of this identity across threads and processes.

        Correctness never depends on it (every shard publishes atomically);
        it only keeps two fits of one corpus from running the same model pass.
        """
        with atomic.artifact_lock(self.root / self.stem):
            yield

    # -- reads ---------------------------------------------------------------

    def load_centroids(self, layers: Sequence[int]) -> dict[int, torch.Tensor]:
        """Every requested layer whose centroid shard is valid."""
        out: dict[int, torch.Tensor] = {}
        for layer in layers:
            try:
                value = self._read_tensor("centroids", layer).to(torch.float32)
            except _READ_ERRORS:
                continue
            if (
                value.ndim == 2
                and int(value.shape[0]) == len(self.node_sizes)
                and bool(torch.isfinite(value).all())
            ):
                out[int(layer)] = value
        return out

    def load_rows(self, layers: Sequence[int]) -> "ActivationRowStore | None":
        """A memory-mapped row store over every requested layer that is valid."""
        from drowse.store.manifold_tensors import ActivationRowStore

        stores: list[ActivationRowStore] = []
        for layer in layers:
            path = self.path("rows", layer)
            try:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    self._check_header(handle, layer)
                stores.append(
                    ActivationRowStore.load_shards({int(layer): path}, self.node_sizes),
                )
            except _READ_ERRORS:
                continue
        if not stores:
            return None
        return stores[0] if len(stores) == 1 else ActivationRowStore.combine_disjoint(stores)

    def _read_tensor(self, kind: str, layer: int) -> torch.Tensor:
        with safe_open(str(self.path(kind, layer)), framework="pt", device="cpu") as handle:
            self._check_header(handle, layer)
            return handle.get_tensor(f"layer_{int(layer)}")

    def _check_header(self, handle: object, layer: int) -> None:
        if (
            handle.metadata() != self._metadata  # type: ignore[attr-defined]
            or list(handle.keys()) != [f"layer_{int(layer)}"]  # type: ignore[attr-defined]
        ):
            raise ValueError("capture shard identity mismatch")

    # -- writes --------------------------------------------------------------

    def save_centroids(self, stacks: Mapping[int, torch.Tensor]) -> None:
        for layer, value in stacks.items():
            payload = serialize_safetensors(
                {f"layer_{int(layer)}": value.detach().to("cpu").contiguous()},
                metadata=self._metadata,
            )
            write_bytes_atomic(self.path("centroids", layer), payload)

    def save_rows(self, rows: "ActivationRowStore", layers: Sequence[int]) -> None:
        for layer in layers:
            _write_streamed(
                f"layer_{int(layer)}", rows.flat_rows(int(layer)),
                self.path("rows", layer), self._metadata,
            )

    # -- bound ---------------------------------------------------------------

    def prune(self) -> None:
        """Evict whole digests, least recently written first, to the size bound.

        ``DROWSE_MANIFOLD_CAPTURE_CACHE_GB`` (default 8; ``<= 0`` disables) is
        the bound.  This identity is never evicted.  Lock-free: a reader that
        loses a shard to eviction sees a miss and recaptures.
        """
        try:
            limit_gb = float(os.environ.get("DROWSE_MANIFOLD_CAPTURE_CACHE_GB", "8"))
        except ValueError:
            limit_gb = 8.0
        if limit_gb <= 0.0 or not self.root.is_dir():
            return
        limit = int(limit_gb * 1024**3)
        sizes: dict[str, int] = {}
        written: dict[str, int] = {}
        files: dict[str, list[Path]] = {}
        for path in self.root.iterdir():
            stem = path.name.split(".", 1)[0]
            if len(stem) != 64 or not path.is_file():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            files.setdefault(stem, []).append(path)
            sizes[stem] = sizes.get(stem, 0) + stat.st_size
            written[stem] = max(written.get(stem, 0), stat.st_mtime_ns)
        total = sum(sizes.values())
        for stem in sorted(
            (stem for stem in sizes if stem != self.stem), key=written.__getitem__,
        ):
            if total <= limit:
                break
            for path in files[stem]:
                path.unlink(missing_ok=True)
            total -= sizes[stem]


def _write_streamed(
    key: str, tensor: torch.Tensor, path: Path, metadata: Mapping[str, str],
) -> None:
    """Publish one possibly multi-GiB tensor shard with bounded host memory."""
    dtype_name = _DTYPE_NAMES.get(tensor.dtype)
    if dtype_name is None:
        raise TypeError(f"unsupported capture dtype {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError("capture shards must be contiguous")
    nbytes = int(tensor.numel() * tensor.element_size())
    header = json.dumps(
        {
            "__metadata__": dict(metadata),
            key: {
                "dtype": dtype_name,
                "shape": [int(dim) for dim in tensor.shape],
                "data_offsets": [0, nbytes],
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    byte_view = tensor.detach().reshape(-1).view(torch.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(struct.pack("<Q", len(header)))
            handle.write(header)
            for start in range(0, byte_view.numel(), _STREAM_CHUNK_BYTES):
                chunk = byte_view[start:start + _STREAM_CHUNK_BYTES]
                handle.write(chunk.to(device="cpu").numpy().tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)
