"""GGUF export: fold a fitted 2-node manifold to a llama.cpp control vector.

``export_gguf_manifold`` is the folder-level driver over the control-vector
codec in :mod:`drowse.geometry.gguf`: it folds a fitted 2-node ``pca``
manifold to a single direction and writes one ``.gguf`` per model, with
``_resolve_model_hint`` deriving llama.cpp's architecture string.  GGUF is the
specialized llama.cpp interchange export; ``.drowse`` carries complete
manifold closures without folding them to a control vector.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from drowse.base.paths import safe_model_id, unsafe_model_id
from drowse.geometry.gguf import write_gguf_profile


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
