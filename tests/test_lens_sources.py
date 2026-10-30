from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from drowse.base.atomic import write_json_atomic
from drowse.base.source_registry import SourceRef
from drowse.lens.jlens import JacobianLens
from drowse.lens.key import LensKey
from drowse.lens.store import LensProvenance, save_lens

_PROV = LensProvenance(
    corpus_spec="test", corpus_sha256="a" * 64, seq_len=8, dim_batch=1,
)


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path / "drowse"))


def _neuronpedia_binding(config: Path, checkpoint: Path) -> Any:
    from drowse.lens import sources

    return sources.LensBinding(
        name="neuronpedia",
        model_id="org/model",
        model_revision="model-commit",
        repo_id="neuronpedia/jacobian-lens",
        repo_revision="lens-commit",
        files={LensKey("J"): sources.BoundLensFile(
            "model/lens.pt", hashlib.sha256(checkpoint.read_bytes()).hexdigest(), (0,),
        )},
        d_model=4,
        n_prompts=12,
        corpus="Salesforce/wikitext:wikitext-103-raw-v1:train",
        seq_len=128,
        skip_first=0,
        provenance={
            "config_file": "model/config.yaml",
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "dim_batch": 8,
        },
    )


def test_external_lens_stays_in_provider_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.lens import sources
    from drowse.lens.store import load_lens, load_lens_sidecar

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
    binding = _neuronpedia_binding(config, checkpoint)
    write_json_atomic(
        sources.LENS_SOURCES.binding_path("org/model", "neuronpedia"),
        binding.to_json(),
    )
    assert sources.load_external_lens_binding("org/model", "neuronpedia") == binding
    sources.use_lens_source("org/model", "neuronpedia")

    def download(_repo: str, filename: str, **_kwargs: object) -> str:
        return str(config if filename.endswith("config.yaml") else checkpoint)

    monkeypatch.setattr(sources, "_hf_hub_download", download)
    sidecar = load_lens_sidecar("org/model")
    assert sidecar is not None and sidecar["_source"]["provider"] == "neuronpedia"
    assert sidecar["kind"] == "J" and sidecar["sites"] == {"out": {"layers": [0]}}
    loaded = load_lens("org/model")
    assert loaded is not None
    lens, _ = loaded
    assert lens.source_layers == [0] and lens.key == LensKey("J")
    assert lens.jacobians[0].dtype == torch.float32
    assert not list((tmp_path / "drowse").rglob("*.safetensors"))
    assert checkpoint.exists()
    rows = sources.list_lens_sources("org/model")
    assert rows[0]["source"] == "neuronpedia"
    assert rows[0]["repo_id"] == "neuronpedia/jacobian-lens"
    assert rows[0]["lens"] == "J" and rows[0]["active"]
    assert sources.remove_lens_source(
        "org/model", SourceRef("huggingface", "neuronpedia"),
    )
    assert sources.LENS_SOURCES.active("org/model", "output") is None
    assert checkpoint.exists()  # provider cache ownership is preserved


def test_pre_v7_binding_reads_as_absent(tmp_path: Path) -> None:
    from drowse.lens import sources

    config = tmp_path / "config.yaml"
    config.write_text("x")
    checkpoint = tmp_path / "lens.pt"
    checkpoint.write_bytes(b"x")
    payload = _neuronpedia_binding(config, checkpoint).to_json()
    payload["format_version"] = 1
    write_json_atomic(
        sources.LENS_SOURCES.binding_path("org/model", "neuronpedia"), payload,
    )
    assert sources.load_external_lens_binding("org/model", "neuronpedia") is None


def test_local_lens_layout_and_source_selection() -> None:
    from drowse.lens.sources import LENS_SOURCES, list_lens_sources
    from drowse.lens.store import lens_manifest_path

    save_lens(JacobianLens({0: torch.eye(3)}, n_prompts=2, d_model=3), "org/model", _PROV)
    manifest = lens_manifest_path("org/model", "j")
    assert manifest.parts[-4:] == ("lens", "local", "j", "manifest.json")
    assert json.loads(LENS_SOURCES.active_path("org/model").read_text()) == {
        "format_version": 2,
        "model_id": "org/model",
        "output": {"kind": "local", "name": "j"},
        "input": None,
    }
    rows = list_lens_sources("org/model")
    assert len(rows) == 1 and rows[0]["source"] == "local:j" and rows[0]["active"]


def test_selection_is_per_side_so_output_and_input_lenses_coexist() -> None:
    """An R-lens and an S-lens read opposite sides of a unit: both stay active."""
    from drowse.lens.sources import LENS_SOURCES, remove_lens_source, use_lens_source

    for kind in ("R", "S", "J"):
        save_lens(
            JacobianLens({0: torch.eye(3)}, n_prompts=2, d_model=3, key=LensKey(kind)),
            "org/model", _PROV,
        )
    # The last publication of each side wins its own slot only.
    assert LENS_SOURCES.active("org/model", "output") == SourceRef("local", "j")
    assert LENS_SOURCES.active("org/model", "input") == SourceRef("local", "s")
    use_lens_source("org/model", "local:r")
    assert LENS_SOURCES.active("org/model", "output") == SourceRef("local", "r")
    assert LENS_SOURCES.active("org/model", "input") == SourceRef("local", "s")

    assert remove_lens_source("org/model", SourceRef("local", "s"))
    assert LENS_SOURCES.active("org/model", "input") is None
    assert LENS_SOURCES.active("org/model", "output") == SourceRef("local", "r")


def test_lens_source_default_preference_order() -> None:
    from drowse.lens.sources import lens_source_preference_key

    shuffled = [
        "local:j-chat",
        "local:j",
        "workspace-j",
        "local:r",
        "neuronpedia",
        "workspace-r",
    ]
    assert sorted(shuffled, key=lens_source_preference_key) == [
        "workspace-r",
        "neuronpedia",
        "workspace-j",
        "local:r",
        "local:j",
        "local:j-chat",
    ]


def test_lens_registry_requires_an_active_source() -> None:
    from drowse.lens.sources import LENS_SOURCES
    from drowse.lens.store import load_lens

    save_lens(JacobianLens({0: torch.eye(3)}, n_prompts=2, d_model=3), "org/model", _PROV)
    LENS_SOURCES.active_path("org/model").unlink()
    assert load_lens("org/model") is None


def test_neuronpedia_fetch_pins_the_resolved_model_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transformers >=5.18 configs carry no ``_commit_hash``.

    The binding's model revision comes from the Hub commit lookup, and the
    dimension check reads the config pinned to that same commit.
    """
    import huggingface_hub
    import transformers
    from huggingface_hub import ResolvedRevision

    from drowse.lens import sources
    commit = "0123456789abcdef0123456789abcdef01234567"

    class _Api:
        def resolve_revision(self, repo_id: str, **_kwargs: Any) -> ResolvedRevision:
            assert repo_id == "org/model"
            return ResolvedRevision(resolved=commit, repo_id=repo_id)

    config_reads: list[dict[str, Any]] = []

    def _read_config(_cls: Any, _model_id: str, **kwargs: Any) -> SimpleNamespace:
        config_reads.append(kwargs)
        return SimpleNamespace(hidden_size=4, num_hidden_layers=3)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", classmethod(_read_config),
    )

    snapshot = tmp_path / "snapshot"
    config_dir = snapshot / "model" / "jlens" / "Salesforce-wikitext"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text("hf_model_name: org/model\n")
    checkpoint = tmp_path / "lens.pt"
    torch.save(
        {
            "J": {0: torch.eye(4)},
            "n_prompts": 12,
            "source_layers": [0],
            "d_model": 4,
        },
        checkpoint,
    )
    files = [
        "model/jlens/Salesforce-wikitext/config.yaml",
        "model/jlens/Salesforce-wikitext/model_jacobian_lens.pt",
    ]
    api = SimpleNamespace(
        model_info=lambda repo, revision: SimpleNamespace(sha="lens-commit"),
        list_repo_files=lambda repo, revision: files,
    )
    monkeypatch.setattr(sources, "_hf_api", lambda: api)
    monkeypatch.setattr(
        sources, "_hf_snapshot_download", lambda _repo, **_kwargs: str(snapshot),
    )
    monkeypatch.setattr(
        sources, "_hf_hub_download",
        lambda _repo, _filename, **_kwargs: str(checkpoint),
    )

    binding = sources.fetch_lens_source("org/model", "neuronpedia")
    assert binding.model_revision == commit
    assert binding.repo_revision == "lens-commit"
    assert binding.key == LensKey("J")
    bound = binding.files[LensKey("J")]
    assert bound.path == "model/jlens/Salesforce-wikitext/model_jacobian_lens.pt"
    assert bound.sha256 == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert config_reads == [{"trust_remote_code": False, "revision": commit}]
    assert sources.load_external_lens_binding("org/model", "neuronpedia") == binding
    assert sources.LENS_SOURCES.active("org/model", "output") == SourceRef(
        "huggingface", "neuronpedia",
    )


def test_external_lens_identity_binds_model_commit_and_shape() -> None:
    from drowse.session.session import _jlens_matches_loaded_model

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
        "sites": {"out": {"layers": [0, 1]}},
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
) -> dict[str, Any]:
    config: dict[str, Any] = {"estimator": estimator}
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
    payload: dict[str, Any],
    *,
    files: list[str] | None = None,
) -> Path:
    from drowse.lens import sources
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
    from drowse.lens import sources
    from drowse.lens.store import load_lens, load_lens_sidecar

    checkpoint = _mock_workspace_hub(
        monkeypatch, tmp_path, _workspace_payload("relp"),
    )
    binding = sources.fetch_lens_source("org/model", "workspace-r")
    assert binding.key == LensKey("R") and binding.provider == "workspace-lenses"
    assert binding.files[LensKey("R")].path == "model/r-lens/lens.pt"
    assert len(binding.files[LensKey("R")].sha256) == 64
    assert binding.corpus == "NeelNanda/pile-10k"
    assert binding.seq_len == 128 and binding.skip_first == 4
    assert binding.provenance["estimator"] == "relp"
    assert binding.provenance["target_layer"] == 2
    assert "ln_rule" in binding.provenance["rules_json"]
    assert sources.LENS_SOURCES.binding_path("org/model", "workspace-r").exists()
    on_disk = json.loads(
        sources.LENS_SOURCES.binding_path("org/model", "workspace-r").read_text(),
    )
    assert set(on_disk["files"]) == {"R:out"}

    sidecar = load_lens_sidecar("org/model")
    assert sidecar is not None
    assert sidecar["kind"] == "R"
    assert sidecar["_source"]["provider"] == "workspace-lenses"
    assert sidecar["_source"]["files"]["R:out"]["path"] == "model/r-lens/lens.pt"
    loaded = load_lens("org/model")
    assert loaded is not None
    lens, _ = loaded
    assert lens.source_layers == [0, 1, 2] and lens.key == LensKey("R")
    assert not list((tmp_path / "drowse").rglob("*.safetensors"))
    assert checkpoint.exists()

    torch.save(_workspace_payload("standard"), checkpoint)
    j_binding = sources.fetch_lens_source("org/model", "workspace-j")
    assert j_binding.provenance["estimator"] == "standard"
    rows = sources.list_lens_sources("org/model")
    by_name = {row["source"]: row for row in rows}
    assert by_name["workspace-r"]["provider"] == "workspace-lenses"
    assert by_name["workspace-r"]["lens"] == "R"
    assert by_name["workspace-j"]["active"]
    assert not by_name["workspace-r"]["active"]

    sources.use_lens_source("org/model", "workspace-r")
    assert sources.LENS_SOURCES.active("org/model", "output") == SourceRef(
        "huggingface", "workspace-r",
    )

    assert sources.remove_lens_source(
        "org/model", SourceRef("huggingface", "workspace-j"),
    )
    assert checkpoint.exists()  # provider cache ownership is preserved


def test_workspace_fetch_rejects_arm_estimator_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.lens import sources
    _mock_workspace_hub(monkeypatch, tmp_path, _workspace_payload("standard"))
    with pytest.raises(ValueError, match="estimator"):
        sources.fetch_lens_source("org/model", "workspace-r")


def test_workspace_fetch_rejects_model_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.lens import sources
    _mock_workspace_hub(
        monkeypatch, tmp_path,
        _workspace_payload("relp", model_id="org/other-model"),
    )
    with pytest.raises(ValueError, match="not the requested model"):
        sources.fetch_lens_source("org/model", "workspace-r")


def test_workspace_fetch_reports_published_models_on_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.lens import sources
    _mock_workspace_hub(
        monkeypatch, tmp_path, _workspace_payload("relp"),
        files=["gemma-3-27b-it/r-lens/lens.pt"],
    )
    with pytest.raises(FileNotFoundError, match="gemma-3-27b-it"):
        sources.fetch_lens_source("org/model", "workspace-r")


def test_fetch_lens_source_dispatch_and_use_rejection() -> None:
    from drowse.lens import sources
    with pytest.raises(ValueError, match="unknown lens fetch source"):
        sources.fetch_lens_source("org/model", "nonsense")
    with pytest.raises(FileNotFoundError, match="not prepared"):
        sources.use_lens_source("org/model", "workspace-r")  # nothing fetched
    with pytest.raises(ValueError, match="local:NAME"):
        sources.use_lens_source("org/model", "nonsense")
    with pytest.raises(ValueError, match="invalid local lens name"):
        sources.use_lens_source("org/model", "local:default")
