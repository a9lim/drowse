from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from drowse.sae import neuronpedia
from drowse.sae import sources as sae_io
from tests.support.sae import MockSaeBackend, make_resident, sae_instrument


def _instrument(feature_meta: dict[str, dict[str, Any]] | None = None) -> Any:
    """A real SAE instrument over a stub, with a 4-wide mock dictionary
    resident at layer 1 under the Neuronpedia binding ``test-model/1-dictionary``."""
    inst = sae_instrument(SimpleNamespace(model_id="org/model"))
    make_resident(
        inst.runtime,
        MockSaeBackend(
            layers=frozenset({1}), d_model=4, repo_id="org/sae",
            sae_ids_by_layer={"1": "layer_1"},
            neuronpedia_ids_by_layer={"1": "test-model/1-dictionary"},
        ),
        layer=1,
        feature_meta=feature_meta,
    )
    return inst


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
    inst = _instrument({"2": {"label": None, "max_act": 10.0}})
    requested = []

    def fetch(_np_id: Any, idx: int, **_kw: Any) -> dict[str, Any]:
        requested.append(idx)
        return {"label": "published description", "max_act": 10.0, "checked": True}

    monkeypatch.setattr(neuronpedia, "fetch_feature", fetch)
    result = inst.fetch_meta([2])["2"] if lookup == "batch" else inst.validate_feature(2)
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

    payload = {
        "modelId": "test-model", "layer": "1-dictionary", "index": "2",
        "source": {"hfRepoId": "org/sae", "saelensSaeId": "layer_1"},
        "explanations": [{"description": "the right feature"}], "maxActApprox": 4.0,
    }
    seen_urls: list[str] = []

    def serve(body: dict[str, Any]) -> Any:
        response = SimpleNamespace(content=json.dumps(body), raise_for_status=lambda: None)

        def get(url: str, **_kwargs: Any) -> Any:
            seen_urls.append(url)
            return response

        return SimpleNamespace(get=get)

    def fetch() -> dict[str, Any] | None:
        return neuronpedia.fetch_feature(
            "test-model/1-dictionary", 2, repo_id="org/sae", sae_id="layer_1",
        )

    monkeypatch.setattr(huggingface_hub, "get_session", lambda: serve(payload))
    assert fetch() == {"label": "the right feature", "max_act": 4.0, "checked": True}
    assert seen_urls == [
        "https://www.neuronpedia.org/api/feature/test-model/1-dictionary/2",
    ]
    monkeypatch.setattr(
        huggingface_hub, "get_session", lambda: serve({**payload, **patch}),
    )
    assert fetch() is None


@pytest.mark.parametrize("network_failure", [False, True])
def test_late_metadata_does_not_label_a_replacement_sae(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, network_failure: bool,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    inst = _instrument()
    inst.probes["sae/1"] = {"feature_id": 1, "layer": 1, "label": None, "max_act": None}

    def fetch(_np_id: Any, _idx: int, **_kw: Any) -> dict[str, Any] | None:
        make_resident(inst.runtime, layer=2)
        return None if network_failure else {"label": "old layer", "max_act": 4.0, "checked": True}

    monkeypatch.setattr(neuronpedia, "fetch_feature", fetch)
    assert inst.fetch_meta([1, 2]) == {}
    assert inst.runtime.resident.feature_meta == {}
    assert inst.probes["sae/1"]["label"] is None
    with pytest.raises(ValueError, match="changed while loading"):
        inst.validate_feature(3)


def test_prepared_source_exposes_exact_neuronpedia_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    sae_io.save_sae_metadata("org/model", "release", _source())
    sources = sae_io.list_sae_sources("org/model")
    assert sources[0]["description_source"] == {
        "model": "test-model", "source": "1-dictionary", "repository": "org/sae", "saeId": "layer_1",
    }
