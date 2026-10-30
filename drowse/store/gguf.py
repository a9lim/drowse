"""GGUF interchange: the control-vector codec plus the manifold export driver.

Two layers live here.  The **codec** — ``write_gguf_profile`` /
``read_gguf_profile`` — round-trips a per-layer direction dict through a
llama.cpp control-vector file.  The **driver** — ``export_gguf_manifold`` —
is the folder-level orchestrator over that codec: it folds a fitted 2-node
``pca`` manifold to a single direction and writes one ``.gguf`` per model,
with ``_resolve_model_hint`` deriving llama.cpp's architecture string.
GGUF is the specialized llama.cpp interchange export; ``.drowse`` carries
complete manifold closures without folding them to a control vector.

Wire format matches the llama.cpp control-vector convention (same as
repeng's export_gguf):

  general.architecture     = "controlvector"
  controlvector.model_hint = <architecture string, e.g. "llama" / "gemma3">
  controlvector.layer_count = uint32
  direction.<layer_idx>     = float32 tensor, shape (hidden_dim,)

Because drowse bakes per-layer PCA shares into the stored direction
magnitudes at extraction time, the tensors written here are already in
llama.cpp's expected form: a uniform `--control-vector-scaled` scalar on
the consumer side reproduces drowse's layer-weighted injection without
any extra metadata slot.

Repeng-exported GGUFs are unit-normed (no share baking); reading them
yields a profile that injects uniformly across layers at apply time.
That's the correct behavior for repeng vectors — it's the semantic they
were exported with. Both paths go through the same loader.

The ``gguf`` Python package is an optional dependency; this module
lazy-imports it and raises a clear error if it's missing.
"""
from __future__ import annotations

from typing import Any, Optional

import logging
from pathlib import Path

import torch

from drowse.errors import DrowseError
from drowse.base.paths import safe_model_id, unsafe_model_id

log = logging.getLogger(__name__)

_ARCH = "controlvector"
_MODEL_HINT_KEY = f"{_ARCH}.model_hint"
_LAYER_COUNT_KEY = f"{_ARCH}.layer_count"
_TENSOR_PREFIX = "direction."


class GGUFNotInstalled(ImportError, DrowseError):
    """Raised when the optional ``gguf`` package is not available."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


def _import_gguf():
    try:
        import gguf
    except ImportError as e:
        raise GGUFNotInstalled(
            "GGUF support requires the 'gguf' package. "
            "Install with: pip install 'drowse.ai[gguf]'"
        ) from e
    return gguf


def write_gguf_profile(
    profile: dict[int, torch.Tensor],
    path: str | Path,
    *,
    model_hint: str,
) -> Path:
    """Write a baked profile to a GGUF file at ``path``.

    Args:
        profile: layer_idx -> baked direction tensor (any dtype; cast to fp32).
        path: output file path.
        model_hint: architecture string for llama.cpp compatibility — usually
            ``model.config.model_type`` of the base model (e.g. "llama",
            "gemma3", "qwen2").

    Returns the written path.
    """
    gguf = _import_gguf()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(str(path), _ARCH)
    writer.add_string(_MODEL_HINT_KEY, model_hint)
    writer.add_uint32(_LAYER_COUNT_KEY, len(profile))
    for layer_idx in sorted(profile.keys()):
        vec = profile[layer_idx].detach().to(dtype=torch.float32, device="cpu").contiguous()
        writer.add_tensor(f"{_TENSOR_PREFIX}{layer_idx}", vec.numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    log.info("Wrote GGUF profile (%d layers) to %s", len(profile), path)
    return path


def read_gguf_profile(path: str | Path) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    """Read a GGUF control-vector file.

    Returns ``(profile, metadata)`` in the shape :func:`save_profile` accepts:
    ``method="profile"`` plus ``source_format``/``model_hint`` provenance from
    the GGUF header.

    Raises:
        GGUFNotInstalled: if the ``gguf`` package isn't installed.
        ValueError: if the file isn't a control-vector GGUF.
    """
    gguf = _import_gguf()
    path = Path(path)
    reader = gguf.GGUFReader(str(path))

    arch_field = reader.get_field("general.architecture")
    if arch_field is None or not len(arch_field.parts):
        log.warning("%s: missing general.architecture field", path)
    else:
        arch = str(bytes(arch_field.parts[-1]), encoding="utf-8", errors="replace")
        if arch != _ARCH:
            raise ValueError(
                f"{path}: general.architecture is {arch!r}, expected {_ARCH!r} "
                f"(is this actually a control-vector file?)"
            )

    hint_field = reader.get_field(_MODEL_HINT_KEY)
    if hint_field is None or not len(hint_field.parts):
        raise ValueError(f"{path}: missing {_MODEL_HINT_KEY} field")
    model_hint = str(bytes(hint_field.parts[-1]), encoding="utf-8", errors="replace")

    profile: dict[int, torch.Tensor] = {}
    for tensor in reader.tensors:
        if not tensor.name.startswith(_TENSOR_PREFIX):
            continue
        try:
            layer = int(tensor.name.split(".", 1)[1])
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"{path}: invalid direction tensor name {tensor.name!r}"
            ) from e
        # tensor.data is a numpy memmap slice; copy to own the buffer.
        profile[layer] = torch.from_numpy(tensor.data.copy())

    if not profile:
        raise ValueError(f"{path}: no direction.* tensors found")

    # A GGUF carries no drowse sidecar; it loads as a plain ``profile`` whose
    # provenance records where it came from, so ``Profile.load(x.gguf)``
    # saves back to safetensors without a schema error.
    metadata = {
        "method": "profile",
        "source_format": "gguf",
        "model_hint": model_hint,
    }
    return profile, metadata


def _resolve_model_hint(safe_id: str) -> str:
    """Derive llama.cpp's ``controlvector.model_hint`` from a safe_model_id.

    Strategy: load the base model's config via ``transformers.AutoConfig``
    (cache-first, network fallback) and return ``config.model_type``.  That's
    the same string llama.cpp's loader keys off when matching a control
    vector to a loaded model (e.g. ``"llama"``, ``"gemma2"``, ``"qwen2"``).

    Raises RuntimeError with actionable guidance if the config can't be
    resolved — callers should surface the ``--model-hint`` flag as the
    escape hatch.
    """
    hf_id = unsafe_model_id(safe_id)
    try:
        from transformers import AutoConfig
    except ImportError as e:  # pragma: no cover — transformers is a hard dep
        raise RuntimeError(
            f"could not resolve model_hint for {hf_id!r}: transformers missing ({e})"
        ) from e
    try:
        cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=False)
    except Exception as e:
        raise RuntimeError(
            f"could not resolve model_hint for {hf_id!r}: {e}. "
            f"Pass --model-hint <arch> explicitly (e.g. 'llama', 'gemma2', 'qwen2')."
        ) from e
    mt = getattr(cfg, "model_type", None)
    if not mt:
        raise RuntimeError(
            f"{hf_id}: config has no model_type field; pass --model-hint explicitly"
        )
    return str(mt)


def export_gguf_manifold(
    ns: str,
    name: str,
    *,
    model_scope: Optional[str],
    output: Optional[str],
    model_hint: Optional[str],
) -> list[Path]:
    """Export a fitted 2-node ``pca`` manifold to GGUF by folding it to a vector.

    Sources each per-model profile by folding the manifold
    (:func:`~drowse.geometry.fold.folded_directions`), then writes a
    llama.cpp control-vector GGUF via :func:`write_gguf_profile`.
    ``model_scope`` restricts to one base model; without it, every fitted
    ``raw`` tensor is exported (one ``.gguf`` per model).

    ``output`` is a ``.gguf`` file path (single model only) or a directory;
    ``None`` means the current working directory, like ``.drowse`` export.
    A directory receives ``<name>.gguf`` for one model and
    ``<name>.<safe_model_id>.gguf`` per model otherwise.  An export is a
    derived interchange file, never part of the artifact store: a
    destination inside ``$DROWSE_HOME`` is refused, so the manifold folder
    and its integrity manifest are never touched.
    """
    from drowse.base.paths import (
        drowse_home, manifold_dir, parse_tensor_filename, tensor_filename,
    )
    from drowse.store.manifold_tensors import load_manifold
    from drowse.geometry.fold import folded_directions

    mdir = manifold_dir(ns, name)
    if not (mdir / "manifold.json").is_file():
        raise FileNotFoundError(f"manifold {ns}/{name} not found at {mdir}")

    raw_models: set[str] = set()
    for tensor in mdir.glob("*.safetensors"):
        parsed = parse_tensor_filename(tensor.name)
        if parsed is not None and parsed[1] is None:  # raw fitted → foldable
            raw_models.add(parsed[0])

    if model_scope is not None:
        sid = safe_model_id(model_scope)
        if sid not in raw_models:
            raise RuntimeError(
                f"{ns}/{name}: no fitted manifold tensor for {model_scope}"
            )
        targets = [sid]
    else:
        targets = sorted(raw_models)
        if not targets:
            raise RuntimeError(
                f"{ns}/{name}: no fitted manifold tensors to export"
            )

    out_path = Path(output).expanduser() if output else Path.cwd()
    to_file = out_path.suffix == ".gguf"
    if to_file and len(targets) > 1:
        raise RuntimeError(
            "multi-model export needs a directory or no --output; "
            f"got file path {out_path}"
        )
    destinations = {
        sid: (
            out_path if to_file
            else out_path / (
                f"{name}.gguf" if len(targets) == 1 else f"{name}.{sid}.gguf"
            )
        )
        for sid in targets
    }
    home = drowse_home().resolve(strict=False)
    for dest in destinations.values():
        if dest.resolve(strict=False).is_relative_to(home):
            raise RuntimeError(
                f"GGUF export path {dest} is inside the drowse store {home}; "
                "pass --output <path> outside it"
            )

    written: list[Path] = []
    for sid in targets:
        manifold = load_manifold(
            mdir / tensor_filename(
                sid, release=None, model_id_is_safe=True,
            )
        )
        try:
            profile = folded_directions(manifold)
        except Exception as e:
            raise RuntimeError(
                f"{ns}/{name}: manifold does not fold to a single steering "
                f"direction (not a 2-node affine subspace): {e}"
            ) from e
        hint = model_hint or _resolve_model_hint(sid)
        written.append(
            write_gguf_profile(profile, destinations[sid], model_hint=hint)
        )
    return written
