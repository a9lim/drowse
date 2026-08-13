from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from saklas.io.atomic import write_json_atomic


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAKLAS_HOME", str(tmp_path / "saklas"))


def test_external_lens_stays_in_provider_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io import lens_sources as sources
    from saklas.io.lens import load_lens, load_lens_sidecar

    provider = tmp_path / "hf-cache"
    provider.mkdir()
    config = provider / "config.yaml"
    config.write_text("hf_model_name: org/model\n")
    checkpoint = provider / "lens.pt"
    torch.save(
        {
            "J": {0: torch.eye(4, dtype=torch.float16)},
            "n_prompts": 12,
            "source_layers": [0],
            "d_model": 4,
        },
        checkpoint,
    )
    import hashlib

    binding = sources.ExternalLensBinding(
        name="neuronpedia",
        model_id="org/model",
        model_revision="model-commit",
        repo_id="neuronpedia/jacobian-lens",
        repo_revision="lens-commit",
        checkpoint="model/lens.pt",
        config_file="model/config.yaml",
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        corpus="Salesforce/wikitext:wikitext-103-raw-v1:train",
        n_prompts=12,
        seq_len=128,
        dim_batch=8,
        d_model=4,
        source_layers=(0,),
    )
    write_json_atomic(
        sources.lens_binding_path("org/model", "neuronpedia"),
        binding.to_json(),
    )
    sources.set_active_lens_source("org/model", "huggingface", "neuronpedia")

    def download(_repo: str, filename: str, **_kwargs: object) -> str:
        return str(config if filename.endswith("config.yaml") else checkpoint)

    monkeypatch.setattr(sources, "_hf_hub_download", download)
    sidecar = load_lens_sidecar("org/model")
    assert sidecar is not None and sidecar["_source"]["provider"] == "neuronpedia"
    loaded = load_lens("org/model")
    assert loaded is not None
    lens, _ = loaded
    assert lens.source_layers == [0]
    assert lens.jacobians[0].dtype == torch.float32
    assert not list((tmp_path / "saklas").rglob("*.safetensors"))
    assert checkpoint.exists()
    rows = sources.list_lens_sources("org/model")
    assert rows[0]["source"] == "neuronpedia"
    assert rows[0]["repo_id"] == "neuronpedia/jacobian-lens"
    assert sources.remove_external_lens_binding("org/model")
    assert checkpoint.exists()  # provider cache ownership is preserved


def test_local_lens_layout_and_source_selection() -> None:
    from saklas.core.jlens import JacobianLens
    from saklas.io.lens import lens_paths, save_lens
    from saklas.io.lens_sources import list_lens_sources, load_active_lens_source

    save_lens(
        JacobianLens({0: torch.eye(3)}, n_prompts=2, d_model=3),
        "org/model",
        corpus_spec="test",
        corpus_sha256="a" * 64,
        seq_len=8,
        dim_batch=1,
        skip_first=0,
    )
    _tensor, manifest = lens_paths("org/model")
    assert manifest.parts[-4:] == ("jlens", "local", "default", "manifest.json")
    assert load_active_lens_source("org/model") == {
        "format_version": 1,
        "model_id": "org/model",
        "kind": "local",
        "name": "default",
    }
    rows = list_lens_sources("org/model")
    assert len(rows) == 1 and rows[0]["source"] == "local:default" and rows[0]["active"]


def test_lens_source_default_preference_order() -> None:
    from saklas.io.lens_sources import lens_source_preference_key

    shuffled = [
        "local:default",
        "workspace-j",
        "local:relp",
        "neuronpedia",
        "workspace-r",
    ]
    assert sorted(shuffled, key=lens_source_preference_key) == [
        "workspace-r",
        "neuronpedia",
        "workspace-j",
        "local:relp",
        "local:default",
    ]


def test_lens_registry_requires_an_active_source() -> None:
    from saklas.core.jlens import JacobianLens
    from saklas.io.lens import load_lens, save_lens
    from saklas.io.lens_sources import lens_active_path

    save_lens(
        JacobianLens({0: torch.eye(3)}, n_prompts=2, d_model=3),
        "org/model",
        corpus_spec="test",
        corpus_sha256="a" * 64,
        seq_len=8,
        dim_batch=1,
        skip_first=0,
    )
    lens_active_path("org/model").unlink()
    assert load_lens("org/model") is None


def test_external_lens_identity_binds_model_commit_and_shape() -> None:
    from saklas.core.session import _jlens_matches_loaded_model

    model = SimpleNamespace(
        config=SimpleNamespace(
            _commit_hash="model-commit",
            hidden_size=4,
            num_hidden_layers=3,
        )
    )
    sidecar = {
        "model_fingerprint": None,
        "d_model": 4,
        "source_layers": [0, 1],
        "_source": {
            "kind": "huggingface",
            "model_id": "org/model",
            "model_revision": "model-commit",
        },
    }
    assert _jlens_matches_loaded_model(sidecar, model, "org/model")
    sidecar["_source"]["model_revision"] = "changed"
    assert not _jlens_matches_loaded_model(sidecar, model, "org/model")


def _workspace_payload(
    estimator: str = "relp", model_id: str = "org/model",
) -> dict:
    import json

    config = {"estimator": estimator}
    if estimator == "relp":
        config["rules"] = {
            "ln_rule": True,
            "identity_rule": True,
            "half_rule": True,
            "half_rule_beta": 0.5,
        }
    return {
        "J": {L: torch.eye(4, dtype=torch.float16) for L in range(3)},
        "n_prompts": 25,
        "source_layers": [0, 1, 2],
        "d_model": 4,
        "provenance": {
            "model_id": model_id,
            "dataset_id": "NeelNanda/pile-10k",
            "target_layer": 2,
            "t_max": 128,
            "n_prompts": 25,
            "skip_first": 4,
            "config_json": json.dumps(config),
        },
    }


def _mock_workspace_hub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict,
    *,
    files: list[str] | None = None,
) -> Path:
    from saklas.io import lens_sources as sources

    checkpoint = tmp_path / "workspace-lens.pt"
    torch.save(payload, checkpoint)
    api = SimpleNamespace(
        model_info=lambda repo, revision: SimpleNamespace(sha="lens-commit"),
        list_repo_files=lambda repo, revision: (
            files if files is not None else ["model/r-lens/lens.pt", "model/j-lens/lens.pt"]
        ),
    )
    monkeypatch.setattr(sources, "_hf_api", lambda: api)
    monkeypatch.setattr(
        sources, "_hf_hub_download",
        lambda _repo, _filename, **_kwargs: str(checkpoint),
    )
    monkeypatch.setattr(
        sources, "_resolve_model_for_fetch",
        lambda _model: ("model-commit", 4, 3),
    )
    return checkpoint


def test_workspace_lens_fetch_binds_loads_and_switches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io import lens_sources as sources
    from saklas.io.lens import load_lens, load_lens_sidecar

    checkpoint = _mock_workspace_hub(
        monkeypatch, tmp_path, _workspace_payload("relp"),
    )
    binding = sources.fetch_workspace_lens("org/model", "workspace-r")
    assert isinstance(binding, sources.WorkspaceLensBinding)
    assert binding.arm == "r-lens" and binding.estimator == "relp"
    assert binding.corpus == "NeelNanda/pile-10k"
    assert binding.seq_len == 128 and binding.skip_first == 4
    assert binding.target_layer == 2
    assert len(binding.checkpoint_sha256) == 64
    assert "ln_rule" in binding.rules_json

    sidecar = load_lens_sidecar("org/model")
    assert sidecar is not None
    assert sidecar["method"] == "workspace_relp"
    assert sidecar["_source"]["provider"] == "workspace-lenses"
    assert sidecar["_source"]["arm"] == "r-lens"
    loaded = load_lens("org/model")
    assert loaded is not None
    lens, _ = loaded
    assert lens.source_layers == [0, 1, 2]
    assert not list((tmp_path / "saklas").rglob("*.safetensors"))
    assert checkpoint.exists()

    torch.save(_workspace_payload("standard"), checkpoint)
    j_binding = sources.fetch_workspace_lens("org/model", "workspace-j")
    assert j_binding.estimator == "standard"
    rows = sources.list_lens_sources("org/model")
    by_name = {row["source"]: row for row in rows}
    assert by_name["workspace-r"]["provider"] == "workspace-lenses"
    assert by_name["workspace-r"]["estimator"] == "relp"
    assert by_name["workspace-j"]["active"]
    assert not by_name["workspace-r"]["active"]

    sources.use_lens_source("org/model", "workspace-r")
    active = sources.load_active_lens_source("org/model")
    assert active is not None and active["name"] == "workspace-r"

    assert sources.remove_external_lens_binding("org/model", "workspace-j")
    assert checkpoint.exists()  # provider cache ownership is preserved


def test_workspace_fetch_rejects_arm_estimator_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io import lens_sources as sources

    _mock_workspace_hub(monkeypatch, tmp_path, _workspace_payload("standard"))
    with pytest.raises(ValueError, match="estimator"):
        sources.fetch_workspace_lens("org/model", "workspace-r")


def test_workspace_fetch_rejects_model_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io import lens_sources as sources

    _mock_workspace_hub(
        monkeypatch, tmp_path,
        _workspace_payload("relp", model_id="org/other-model"),
    )
    with pytest.raises(ValueError, match="not the requested model"):
        sources.fetch_workspace_lens("org/model", "workspace-r")


def test_workspace_fetch_reports_published_models_on_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from saklas.io import lens_sources as sources

    _mock_workspace_hub(
        monkeypatch, tmp_path, _workspace_payload("relp"),
        files=["gemma-3-27b-it/r-lens/lens.pt"],
    )
    with pytest.raises(FileNotFoundError, match="gemma-3-27b-it"):
        sources.fetch_workspace_lens("org/model", "workspace-r")


def test_fetch_lens_source_dispatch_and_use_rejection() -> None:
    from saklas.io import lens_sources as sources

    with pytest.raises(ValueError, match="unknown J-lens fetch source"):
        sources.fetch_lens_source("org/model", "nonsense")
    with pytest.raises(ValueError, match="local:NAME"):
        sources.use_lens_source("org/model", "workspace-r")  # nothing fetched
