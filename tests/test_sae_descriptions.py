from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from drowse.io import sae as sae_io
from tests.test_sae_runtime import _session


def _source(layer: int = 1) -> dict[str, Any]:
    return {
        "layer": layer, "width": 4, "revision": "commit", "fingerprint": "weights",
        "sae_id": f"layer_{layer}", "repo_id": "org/sae",
        "neuronpedia_id": f"test-model/{layer}-dictionary",
    }


def test_cached_feature_labels_do_not_cross_sae_layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    sae_io.save_sae_metadata("org/model", "release", _source(1))
    sae_io.save_sae_feature_meta("org/model", "release", {"2": {"label": "layer one", "max_act": 3.0}})
    assert sae_io.load_sae_feature_meta("org/model", "release")["2"]["label"] == "layer one"
    sae_io.save_sae_metadata("org/model", "release", _source(2))
    assert sae_io.load_sae_feature_meta("org/model", "release") == {}


@pytest.mark.parametrize("patch", [{"fingerprint": "different"}, {"revision": "new"}, {"sae_id": "other-l0"}])
def test_cached_labels_require_the_same_dictionary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patch: dict[str, str],
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    sae_io.save_sae_metadata("org/model", "release", _source())
    sae_io.save_sae_feature_meta("org/model", "release", {"2": {"label": "original", "max_act": 3.0}})
    sae_io.save_sae_metadata("org/model", "release", {**_source(), **patch})
    assert sae_io.load_sae_feature_meta("org/model", "release") == {}


@pytest.mark.parametrize("lookup", ["batch", "validate"])
def test_a_cached_maximum_does_not_suppress_a_missing_description(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lookup: str,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    session = _session()
    session._model_info = {"model_id": "org/model"}
    session._sae_feature_meta["2"] = {"label": None, "max_act": 10.0}
    requested = []

    def fetch(idx: int) -> dict[str, Any]:
        requested.append(idx)
        return {"label": "published description", "max_act": 10.0, "checked": True}

    monkeypatch.setattr(session, "_fetch_neuronpedia_feature", fetch)
    result = session.fetch_sae_feature_meta([2])["2"] if lookup == "batch" else session.validate_sae_feature(2)
    assert result["label"] == "published description"
    assert requested == [2]


@pytest.mark.parametrize("patch", [
    {"modelId": "other-model"}, {"layer": "other-dictionary"}, {"index": "3"},
    {"source": {"hfRepoId": "other/repo", "saelensSaeId": "layer_1"}},
    {"source": {"hfRepoId": "org/sae", "saelensSaeId": "other-l0"}},
])
def test_neuronpedia_payload_must_identify_the_requested_feature(
    monkeypatch: pytest.MonkeyPatch, patch: dict[str, Any],
) -> None:
    import huggingface_hub

    session = _session()
    backend = session._sae_backend
    assert backend is not None
    backend.neuronpedia_ids_by_layer = {"1": "test-model/1-dictionary"}
    backend.sae_ids_by_layer = {"1": "layer_1"}
    backend.repo_id = "org/sae"
    payload = {
        "modelId": "test-model", "layer": "1-dictionary", "index": "2",
        "source": {"hfRepoId": "org/sae", "saelensSaeId": "layer_1"},
        "explanations": [{"description": "the wrong feature"}], "maxActApprox": 4.0,
        **patch,
    }
    response = SimpleNamespace(content=json.dumps(payload), raise_for_status=lambda: None)
    monkeypatch.setattr(huggingface_hub, "get_session", lambda: SimpleNamespace(get=lambda *_a, **_k: response))
    assert session._fetch_neuronpedia_feature(2) is None


@pytest.mark.parametrize("network_failure", [False, True])
def test_late_metadata_does_not_label_a_replacement_sae(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, network_failure: bool,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    session = _session()
    session._model_info = {"model_id": "org/model"}

    def fetch(_idx: int) -> dict[str, Any] | None:
        session._sae_layer = 2
        session._sae_feature_meta = {}
        return None if network_failure else {"label": "old layer", "max_act": 4.0, "checked": True}

    monkeypatch.setattr(session, "_fetch_neuronpedia_feature", fetch)
    assert session.fetch_sae_feature_meta([1, 2]) == {}
    assert session._sae_feature_meta == {}


def test_prepared_source_exposes_exact_neuronpedia_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    sae_io.save_sae_metadata("org/model", "release", _source())
    sources = sae_io.list_sae_sources("org/model")
    assert sources[0]["description_source"] == {
        "model": "test-model", "source": "1-dictionary", "repository": "org/sae", "saeId": "layer_1",
    }
