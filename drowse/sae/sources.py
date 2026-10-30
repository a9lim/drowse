"""Source family + local metadata cache for a session-resident SAE.

Weights remain in the normal Hugging Face cache owned by SAELens. Drowse only
persists the resolved release/layer binding and optional per-feature metadata
(Neuronpedia display labels + ``maxActApprox``) under
``models/<safe>/sae/bindings``. Drowse-trained weights are a separate local
artifact family under ``sae/local`` (:mod:`drowse.sae.store`).

:data:`SAE_SOURCES` is the family's :class:`~drowse.base.source_registry.SourceFamily`
— the same paths, grammar, and locked selection as the lens family — and the
ONE parser of both SAE grammars: the public *source* (``local:NAME`` /
``saelens:RELEASE``) and the *release* string the backend loads
(``local:NAME`` / a bare provider release; :func:`parse_sae_release`).
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import re
from typing import Any

from drowse.base.atomic import artifact_lock, write_json_atomic
from drowse.base.integrity import NAME_REGEX
from drowse.base.paths import decode_release_id, encode_release_id
from drowse.base.source_registry import (
    LOCAL_PREFIX,
    SourceFamily,
    SourceRef,
)

SAE_RUNTIME_FORMAT_VERSION = 3
SAE_FEATURE_META_FORMAT_VERSION = 4
SAE_SOURCE_FORMAT_VERSION = 1
SAELENS = "saelens"
_RUNTIME_FIELDS = {
    "layer", "width", "revision", "fingerprint", "sae_id", "repo_id",
    "neuronpedia_id",
}

#: The SAE source family: ``local:NAME`` artifacts and ``saelens:RELEASE``
#: bindings (filed under the release's reversible encoding), one selection.
SAE_SOURCES = SourceFamily(
    label="SAE",
    dirname="sae",
    format_version=SAE_SOURCE_FORMAT_VERSION,
    provider_kind=SAELENS,
    provider_prefix=f"{SAELENS}:",
    local_name=NAME_REGEX,
    provider_name_ok=lambda name: bool(name.strip()),
    binding_stem=encode_release_id,
)


def parse_sae_release(release: str) -> SourceRef:
    """The release string the backend loads, as a ref.

    ``local:NAME`` is a Drowse-trained artifact; anything else is a bare
    SAELens release id.
    """
    text = release.strip()
    if text.startswith(LOCAL_PREFIX):
        return SAE_SOURCES.parse(text)
    return SAE_SOURCES.parse(f"{SAELENS}:{text}")


def sae_release(ref: SourceRef) -> str:
    """Inverse of :func:`parse_sae_release`: ``local:NAME`` or the bare release."""
    return SAE_SOURCES.format(ref) if ref.local else ref.name


def sae_features_path(model_id: str, release: str) -> Path:
    return SAE_SOURCES.bindings_dir(model_id) / (
        f"{encode_release_id(release)}-features.json"
    )


def load_active_sae(model_id: str) -> tuple[str, dict[str, Any] | None] | None:
    """Resolve the active SAE selection to ``(release, provider_metadata)``.

    Returns ``None`` when nothing is selected; ``metadata`` is the validated
    runtime binding for a provider source (carrying the pinned commit and the
    explicitly selected hook layer) and ``None`` for a local artifact, whose
    manifest is the binding.
    """
    active = SAE_SOURCES.active(model_id)
    if active is None:
        return None
    metadata = None if active.local else load_sae_metadata(model_id, active.name)
    return sae_release(active), metadata


def save_sae_metadata(
    model_id: str, release: str, payload: dict[str, Any], *,
    activate: bool = True,
) -> Path:
    """Persist a provider release's runtime binding under the artifact lock.

    ``activate=False`` records the binding without selecting it — the same
    opt-out ``fetch_lens_source`` has, for a caller that fetches or
    refreshes one source while another stays resident.
    """
    if set(payload) != _RUNTIME_FIELDS:
        raise ValueError(
            f"SAE runtime metadata fields must be {sorted(_RUNTIME_FIELDS)}"
        )
    full = {
        **payload,
        "format_version": SAE_RUNTIME_FORMAT_VERSION,
        "model_id": model_id,
        "release": release,
    }
    if not _validate_runtime_payload(full, model_id, release):
        raise ValueError("invalid SAE runtime metadata values")
    path = SAE_SOURCES.binding_path(model_id, release)
    with artifact_lock(path):
        write_json_atomic(path, full)
    if activate:
        SAE_SOURCES.select(model_id, SourceRef(SAELENS, release))
    return path


def _validate_runtime_payload(
    payload: Any, model_id: str, release: str,
) -> bool:
    expected = {"format_version", "model_id", "release", *_RUNTIME_FIELDS}
    if not isinstance(payload, dict) or set(payload) != expected:
        return False
    if (
        payload["format_version"] != SAE_RUNTIME_FORMAT_VERSION
        or payload["model_id"] != model_id
        or payload["release"] != release
        or isinstance(payload["layer"], bool)
        or not isinstance(payload["layer"], int)
        or payload["layer"] < 0
        or isinstance(payload["width"], bool)
        or not isinstance(payload["width"], int)
        or payload["width"] <= 0
        or not isinstance(payload["revision"], str)
        or not payload["revision"]
        or not isinstance(payload["fingerprint"], str)
        or not payload["fingerprint"]
    ):
        return False
    return all(
        payload[key] is None or (
            isinstance(payload[key], str) and bool(payload[key].strip())
        )
        for key in ("sae_id", "repo_id", "neuronpedia_id")
    )


def load_sae_metadata(model_id: str, release: str) -> dict[str, Any] | None:
    path = SAE_SOURCES.binding_path(model_id, release)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not _validate_runtime_payload(payload, model_id, release):
        return None
    return payload


def _validate_feature_entry(value: Any) -> dict[str, Any] | None:
    """Validate one exact current ``{label, max_act}`` feature row."""
    if not isinstance(value, dict) or set(value) != {"label", "max_act"}:
        return None
    label = value["label"]
    if label is not None and not (isinstance(label, str) and label.strip()):
        return None
    max_act = value["max_act"]
    if max_act is not None and (
        isinstance(max_act, bool)
        or not isinstance(max_act, (int, float))
        or not math.isfinite(float(max_act))
        or float(max_act) <= 0
    ):
        return None
    return {"label": label, "max_act": None if max_act is None else float(max_act)}


def _feature_source(
    model_id: str, release: str, source: dict[str, Any] | None,
) -> dict[str, Any] | None:
    source = source if source is not None else load_sae_metadata(model_id, release)
    return {key: source[key] for key in sorted(_RUNTIME_FIELDS)} if source is not None else None


def load_sae_feature_meta(
    model_id: str, release: str, *, source: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load the current ``{feature_id: {label, max_act}}`` metadata cache."""
    path = sae_features_path(model_id, release)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if (
        not isinstance(payload, dict)
        or set(payload) != {"format_version", "model_id", "release", "source", "features"}
        or payload["format_version"] != SAE_FEATURE_META_FORMAT_VERSION
        or payload["model_id"] != model_id
        or payload["release"] != release
        or payload["source"] != _feature_source(model_id, release, source)
    ):
        return {}
    features = payload["features"]
    if not isinstance(features, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in features.items():
        if not isinstance(key, str) or not re.fullmatch(r"0|[1-9][0-9]*", key):
            return {}
        entry = _validate_feature_entry(value)
        if entry is not None:
            out[key] = entry
        else:
            return {}
    return out


def save_sae_feature_meta(
    model_id: str, release: str, features: dict[str, dict[str, Any]],
    *, source: dict[str, Any] | None = None,
) -> Path:
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in features.items():
        key_value: Any = key
        if (
            not isinstance(key_value, str)
            or not re.fullmatch(r"0|[1-9][0-9]*", key_value)
        ):
            raise ValueError(f"invalid SAE feature id {key!r}")
        # The resident session may carry ephemeral lookup state (currently
        # ``checked``). Persist only the two fields in the cache contract.
        row = {"label": value.get("label"), "max_act": value.get("max_act")}
        entry = _validate_feature_entry(row)
        if entry is None:
            raise ValueError(f"invalid SAE feature metadata row {key!r}")
        normalized[key] = entry
    path = sae_features_path(model_id, release)
    with artifact_lock(path):
        write_json_atomic(path, {
            "format_version": SAE_FEATURE_META_FORMAT_VERSION,
            "model_id": model_id,
            "release": release,
            "source": _feature_source(model_id, release, source),
            "features": normalized,
        })
    return path


def load_sae_source(model_id: str, ref: SourceRef) -> dict[str, Any] | None:
    """A source's validated record: a local manifest or a provider binding."""
    if ref.local:
        from drowse.sae.store import load_local_sae_manifest

        return load_local_sae_manifest(model_id, ref.name)
    return load_sae_metadata(model_id, ref.name)


def use_sae_source(model_id: str, source: str) -> SourceRef:
    """Select a source from the public ``local:NAME`` / ``saelens:RELEASE`` grammar."""
    ref = SAE_SOURCES.parse(source)
    if load_sae_source(model_id, ref) is None:
        raise FileNotFoundError(
            f"SAE source {SAE_SOURCES.format(ref)} is not prepared for {model_id}"
        )
    SAE_SOURCES.select(model_id, ref)
    return ref


def remove_sae_source(model_id: str, ref: SourceRef) -> bool:
    """Delete a local artifact or forget a provider binding.

    A provider removal never touches SAELens/HF cache bytes; either way the
    selection is unpublished only when it named exactly this source.
    """
    if ref.local:
        from drowse.sae.store import remove_local_sae

        return remove_local_sae(model_id, ref.name)
    metadata = SAE_SOURCES.binding_path(model_id, ref.name)
    features = sae_features_path(model_id, ref.name)
    with artifact_lock(metadata):
        removed = metadata.exists() or features.exists()
        metadata.unlink(missing_ok=True)
        features.unlink(missing_ok=True)
    SAE_SOURCES.clear_if_active(model_id, ref)
    return removed


def list_sae_sources(model_id: str) -> list[dict[str, Any]]:
    from drowse.sae.store import list_local_saes

    rows = list_local_saes(model_id)
    active = SAE_SOURCES.active(model_id)
    for path in sorted(SAE_SOURCES.bindings_dir(model_id).glob("_z*.json")):
        if path.name.endswith("-features.json"):
            continue
        try:
            release = decode_release_id(path.stem)
        except ValueError:
            continue
        metadata = load_sae_metadata(model_id, release)
        if metadata is None:
            continue
        neuronpedia_id = metadata["neuronpedia_id"]
        description_source = None
        if isinstance(neuronpedia_id, str) and "/" in neuronpedia_id and metadata["repo_id"] and metadata["sae_id"]:
            model, dictionary = neuronpedia_id.split("/", 1)
            description_source = {
                "model": model, "source": dictionary,
                "repository": metadata["repo_id"], "saeId": metadata["sae_id"],
            }
        ref = SourceRef(SAELENS, release)
        rows.append({
            "source": SAE_SOURCES.format(ref),
            "kind": SAELENS,
            "name": release,
            "active": active == ref,
            "path": str(path),
            "layer": metadata["layer"],
            "features": metadata["width"],
            "revision": metadata["revision"],
            "description_source": description_source,
        })
    return rows
