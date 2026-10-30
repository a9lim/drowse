"""Weight-free proof that a ``manifold fit`` would be a no-op.

The manifold counterpart of ``lens fit``'s model-free preflight: every
component of the fit's cache key is re-derived from the folder, the
checkpoint's published files, and the tokenizer, then matched against the
digest-proven fitted sidecar — without instantiating the transformer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from drowse.base.paths import tensor_filename
from drowse.fit.pipeline import (
    MANIFOLD_CAPTURE_VERSION,
    offline_fit_identity,
    resolve_fit_layer_indices,
)
from drowse.store.manifold_folder import ManifoldFolder, read_proven_sidecar
from drowse.store.manifold_schema import ManifoldFormatError
from drowse.store.manifold_tensors import fitted_tensor_layers


@dataclass(frozen=True)
class ManifoldFitProof:
    """Proof that one ``manifold fit`` target is already exact, made off-model.

    Returned only when *every* component of the fit's cache key was
    independently established without instantiating the transformer; every
    field is read from the digest-proven sidecar.
    """

    folder: Path
    model_id: str
    tensor_path: Path
    manifold_name: str
    fit_mode: str
    fitted_layers: tuple[int, ...]
    node_labels: tuple[str, ...]


def _proven_neutral_loaded_fingerprint(
    model_id: str, source_fingerprint: str,
) -> str | None:
    """Bridge a proven checkpoint source to the loaded fingerprint it produces.

    The neutral-activation cache was written by a real load, binding
    ``model_source_fingerprint`` (recomputable off-model) to the
    ``model_fingerprint`` that load produced, so matching the source half
    establishes the loaded half without a second load.
    """
    from drowse.store.neutral import validate_neutral_cache_metadata

    try:
        sidecar = validate_neutral_cache_metadata(model_id)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        return None
    if sidecar.get("model_source_fingerprint") != source_fingerprint:
        return None
    loaded = sidecar.get("model_fingerprint")
    return loaded if isinstance(loaded, str) and loaded else None


def preflight_manifold_fit_noop(
    folder: Path | str,
    *,
    model_id: str,
    layer_indices: "Any" = None,
    sae: str | None = None,
    fit_mode: str | None = None,
    hyperparams: Mapping[str, Any] | None = None,
    quantize: str | None = None,
    device: str = "auto",
) -> ManifoldFitProof | None:
    """Prove an existing fit is exact for ``model_id`` without loading weights.

    A non-``None`` return means re-running the fit would publish the identical
    tensor.  ``None`` means **unproven**, not "stale": an SAE fit, a discover
    override (which rewrites ``manifold.json`` inside the fit's own lock), a
    checkpoint with no provable source, a missing neutral cache, or a
    tokenizer that will not load all return it, and the caller falls through
    to the ordinary model-loading fit.

    Proven, in the order the fit keys on them: ``nodes_sha256`` (recomputed
    from the folder), ``model_source_fingerprint`` (recomputed from config /
    checkpoint files), ``model_fingerprint`` (bridged through the neutral
    cache), the tokenizer-render half of ``capture_sha256``
    (``capture_version`` + ``capture_render_sha256`` +
    ``baseline_prompts_sha256``), ``fit_policy_version``, and the requested
    layer set against the config's layer count.  Nothing here mutates.
    """
    from drowse.geometry.manifold import MANIFOLD_FIT_POLICY_VERSION
    from drowse.model.arch import config_model_shape
    from drowse.model.fingerprint import model_source_fingerprint

    if sae is not None or fit_mode is not None or hyperparams:
        return None
    folder = Path(folder)
    try:
        mf = ManifoldFolder.load(folder, verify_manifest=False)
    except (ManifoldFormatError, OSError, ValueError):
        return None
    if mf.fit_mode == "baked":
        return None
    source_fingerprint = model_source_fingerprint(model_id, quantize=quantize, device=device)
    if source_fingerprint is None:
        return None
    loaded_fingerprint = _proven_neutral_loaded_fingerprint(model_id, source_fingerprint)
    if loaded_fingerprint is None:
        return None
    try:
        n_layers, _hidden = config_model_shape(model_id)
        requested_layers = resolve_fit_layer_indices(layer_indices, n_layers)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        return None
    try:
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    except Exception:
        # Hub / custom tokenizer resolution can raise anything; every failure
        # means "cannot prove".
        return None
    identity = offline_fit_identity(
        mf, tokenizer=tokenizer, model_type=getattr(config, "model_type", None),
    )
    if identity is None:
        return None
    tensor_path = folder / tensor_filename(model_id)
    sidecar = read_proven_sidecar(folder, tensor_path)
    if sidecar is None or (
        sidecar.nodes_sha256 != identity.nodes_sha256
        or tuple(sidecar.node_labels) != identity.node_labels
        or sidecar.feature_space != "raw"
        or sidecar.sae_release is not None
        or sidecar.model_fingerprint != loaded_fingerprint
        or sidecar.model_source_fingerprint != source_fingerprint
        or sidecar.capture_version != MANIFOLD_CAPTURE_VERSION
        # A fit whose handle had no callable tokenizer stamps both capture
        # digests ``None``: a real fit, but not a provable one.
        or sidecar.capture_sha256 is None
        or sidecar.capture_render_sha256 != identity.capture_render_sha256
        or sidecar.baseline_prompts_sha256 != identity.baseline_prompts_sha256
        or sidecar.fit_policy_version != MANIFOLD_FIT_POLICY_VERSION
        or sidecar.evaluated_layers != requested_layers
    ):
        return None
    layers = fitted_tensor_layers(tensor_path)
    if layers is None or list(layers) != sorted(sidecar.fitted_layers):
        return None
    return ManifoldFitProof(
        folder=folder,
        model_id=model_id,
        tensor_path=tensor_path,
        manifold_name=mf.name,
        fit_mode=sidecar.fit_mode,
        fitted_layers=layers,
        node_labels=tuple(sidecar.node_labels),
    )
