"""Neuronpedia feature-metadata client.

One HTTP read per feature, ``GET /api/feature/<model>/<source>/<index>``,
returns the display label (the first non-empty explanation) and
``maxActApprox``, the corpus-max activation that normalizes a feature's
strength channel to ~[0, 1].

The response is checked against the binding that asked for it: the model id,
the source id, the feature index and, when the binding knows them, the
dictionary's HF repo and SAELens id.  A row describing some other dictionary
is rejected rather than cached under this one.

This module does network and validation only.  It takes no session, no lock
and no disk path, and its callers run it outside the exclusive-GPU section
(steering never fetches; see :meth:`drowse.sae.runtime.SaeRuntime.check_feature`).
"""

from __future__ import annotations

import json
import math
from typing import Any
from urllib.parse import quote

API_ROOT = "https://www.neuronpedia.org/api/feature"
USER_AGENT = "drowse-sae-meta/1"
DEFAULT_TIMEOUT = 10.0

#: Binding key -> the Neuronpedia ``source`` field that must agree with it.
_DICTIONARY_FIELDS = (("repo_id", "hfRepoId"), ("sae_id", "saelensSaeId"))


def split_neuronpedia_id(neuronpedia_id: Any) -> tuple[str, str] | None:
    """``"<model>/<source>"`` -> ``(model, source)``, or ``None`` when the
    binding names no Neuronpedia dictionary."""
    if not isinstance(neuronpedia_id, str) or "/" not in neuronpedia_id:
        return None
    model, source = neuronpedia_id.split("/", 1)
    return model, source


def feature_url(model: str, source: str, feature_id: int) -> str:
    return f"{API_ROOT}/{quote(model, safe='')}/{quote(source, safe='')}/{int(feature_id)}"


def parse_feature_payload(
    payload: Any,
    *,
    model: str,
    source: str,
    feature_id: int,
    repo_id: str | None = None,
    sae_id: str | None = None,
) -> dict[str, Any] | None:
    """Validate one feature response -> ``{label, max_act, checked}``.

    ``None`` when the payload does not describe exactly this feature of this
    dictionary.  A valid payload always yields an entry, even one with no
    label and no ``maxActApprox``: ``checked`` records that the feature was
    asked about, so a feature Neuronpedia has no data for is not re-fetched
    on every validation.
    """
    if (
        not isinstance(payload, dict)
        or payload.get("modelId") != model
        or payload.get("layer") != source
        or str(payload.get("index")) != str(feature_id)
        or not isinstance(payload.get("explanations"), list)
    ):
        return None
    dictionary = payload.get("source")
    expected = {"repo_id": repo_id, "sae_id": sae_id}
    if not isinstance(dictionary, dict) or any(
        expected[key] is not None and dictionary.get(field) != expected[key]
        for key, field in _DICTIONARY_FIELDS
    ):
        return None
    label = None
    for row in payload["explanations"]:
        if not isinstance(row, dict):
            continue
        description = row.get("description")
        if isinstance(description, str) and description.strip():
            label = description.strip()
            break
    raw_max = payload.get("maxActApprox")
    max_act: float | None = None
    if isinstance(raw_max, (int, float)) and not isinstance(raw_max, bool):
        value = float(raw_max)
        if math.isfinite(value) and value > 0:
            max_act = value
    return {"label": label, "max_act": max_act, "checked": True}


def fetch_feature(
    neuronpedia_id: str | None,
    feature_id: int,
    *,
    repo_id: str | None = None,
    sae_id: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any] | None:
    """Fetch and validate one feature's metadata.

    ``None`` when the binding names no Neuronpedia dictionary, on any
    transport failure (the caller retries next time), or when the response
    fails validation.
    """
    parts = split_neuronpedia_id(neuronpedia_id)
    if parts is None:
        return None
    model, source = parts
    from huggingface_hub import get_session

    try:
        response = get_session().get(
            feature_url(model, source, feature_id),
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        payload = json.loads(response.content)
    except Exception:  # noqa: BLE001 — best effort; any failure retries later
        return None
    return parse_feature_payload(
        payload, model=model, source=source, feature_id=feature_id,
        repo_id=repo_id, sae_id=sae_id,
    )


__all__ = [
    "API_ROOT",
    "fetch_feature",
    "feature_url",
    "parse_feature_payload",
    "split_neuronpedia_id",
]
