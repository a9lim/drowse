"""``scripts/migrate_lens_v7.py`` on a synthetic v6 DROWSE_HOME."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from drowse.base.paths import model_dir
from drowse.base.source_registry import SourceRef
from drowse.lens.key import LensKey

_MODEL = "org/model"
_D = 4
_SHA = "a" * 64


def _script() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / "migrate_lens_v7.py"
    spec = importlib.util.spec_from_file_location("migrate_lens_v7", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    return tmp_path


def _v6_policy(*, relp: bool, pre_fix: bool = False) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "method": "jlens_cotangent_sum",
        "skip_first_positions": 16,
        "corpus_preprocess_version": 1,
        "doc_chars": 4000,
    }
    if relp and not pre_fix:
        from drowse.lens.relp import RELP_RULES

        policy["method"] = "relp_cotangent_sum"
        policy["backward_rules"] = dict(RELP_RULES)
    return policy


def _write_v6_lens(
    directory: Path, matrices: dict[int, torch.Tensor], *, relp: bool,
    checkpoint: bool = False, pre_fix: bool = False, n_prompts: int = 3,
) -> None:
    """Exactly what the v6 writer left on disk (its shard bytes are v7's)."""
    from drowse.lens.store import _save_fp32_square_safetensors_atomic

    directory.mkdir(parents=True, exist_ok=True)
    stem = "jlens.partial" if checkpoint else "jlens"
    files: dict[str, str] = {}
    digests: dict[str, str] = {}
    for layer, matrix in matrices.items():
        name = f"{stem}.layer-{layer}.gen-{os.urandom(4).hex()}.safetensors"
        digests[str(layer)] = _save_fp32_square_safetensors_atomic(
            directory / name, layer, _D,
            lambda _l, start, end, m=matrix: m[start:end].contiguous(),
        )
        files[str(layer)] = name
    policy = _v6_policy(relp=relp, pre_fix=pre_fix)
    pointer = {
        "format_version": 6,
        "method": policy["method"],
        "n_prompts": n_prompts,
        "d_model": _D,
        "source_layers": sorted(matrices),
        "dtype": "float32",
        "corpus_spec": "test",
        "corpus_sha256": _SHA,
        "corpus_hash_kind": "token_ids_v1",
        "seq_len": 128,
        "dim_batch": 8,
        "skip_first_positions": 16,
        "estimator_policy": policy,
        "tensor_sha256": digests,
        "tensor_files": files,
        "raw_corpus_sha256": None,
        "raw_prompt_count": 4,
        "usable_prompt_count": 4,
        "model_layer_count": 3,
        "model_fingerprint": "weights",
        "model_source_fingerprint": None,
        "checkpoint": checkpoint,
        "base_n_prompts": 0 if checkpoint else None,
        "partial_n_prompts": n_prompts if checkpoint else None,
        "consumed_prefix_sha256": "b" * 64 if checkpoint else None,
    }
    name = "checkpoint.json" if checkpoint else "manifest.json"
    (directory / name).write_text(json.dumps(pointer))


def _matrices(seed: int) -> dict[int, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {layer: torch.randn(_D, _D, generator=gen) for layer in (0, 1)}


def _v6_home() -> dict[str, dict[int, torch.Tensor]]:
    root = model_dir(_MODEL) / "jlens"
    std, relp, ckpt = _matrices(0), _matrices(1), _matrices(2)
    _write_v6_lens(root / "local" / "default", std, relp=False)
    _write_v6_lens(root / "local" / "relp", relp, relp=True)
    _write_v6_lens(
        root / "local" / "relp", ckpt, relp=True, checkpoint=True, n_prompts=2,
    )
    (root / "local" / "relp" / ".locks").mkdir()
    (root / "local" / "relp" / ".locks" / "jlens.fit.lock").write_bytes(b"")
    (root / "bindings").mkdir(parents=True)
    (root / "bindings" / "workspace-r.json").write_text(json.dumps({
        "format_version": 1, "kind": "huggingface",
        "provider": "workspace-lenses", "name": "workspace-r",
        "model_id": _MODEL, "model_revision": "model-commit",
        "repo_id": "camilablank/workspace-lenses", "repo_revision": "lens-commit",
        "checkpoint": "model/r-lens/lens.pt", "checkpoint_sha256": "c" * 64,
        "arm": "r-lens", "estimator": "relp", "rules_json": "{}",
        "corpus": "NeelNanda/pile-10k", "n_prompts": 25, "seq_len": 128,
        "skip_first": 4, "target_layer": 2, "d_model": _D,
        "source_layers": [0, 1, 2],
    }))
    (root / "bindings" / "neuronpedia.json").write_text(json.dumps({
        "format_version": 1, "kind": "huggingface", "provider": "neuronpedia",
        "name": "neuronpedia", "model_id": _MODEL,
        "model_revision": "model-commit",
        "repo_id": "neuronpedia/jacobian-lens", "repo_revision": "np-commit",
        "checkpoint": "model/lens.pt", "config_file": "model/config.yaml",
        "config_sha256": "d" * 64, "corpus": "wikitext", "n_prompts": 12,
        "seq_len": 128, "dim_batch": 8, "d_model": _D, "source_layers": [0],
    }))
    (root / "active.json").write_text(json.dumps({
        "format_version": 1, "model_id": _MODEL, "kind": "local", "name": "relp",
    }))
    return {"std": std, "relp": relp, "ckpt": ckpt}


def test_migrates_a_v6_home_without_touching_shard_bytes(_home: Path) -> None:
    from drowse.lens.sources import LENS_SOURCES, load_external_lens_binding
    from drowse.lens.store import (
        load_lens, load_lens_checkpoint, load_local_lens, load_local_lens_sidecar,
    )

    originals = _v6_home()
    old_root = model_dir(_MODEL) / "jlens"
    shard_bytes = {
        path.name: path.read_bytes()
        for path in old_root.rglob("*.safetensors")
    }
    module = _script()
    lines: list[str] = []

    assert module.migrate_home(
        _home, dry_run=True, log=lines.append, digest_of=lambda *_a: None,
    ) == 1
    assert old_root.exists() and not (model_dir(_MODEL) / "lens").exists()

    assert module.migrate_home(
        _home, log=lines.append, digest_of=lambda *_a: None,
    ) == 1
    assert not old_root.exists()
    assert any("re-run `drowse lens fetch <model> neuronpedia`" in l for l in lines)

    for name, kind, matrices in (("j", "J", originals["std"]), ("r", "R", originals["relp"])):
        loaded = load_local_lens(_MODEL, name)
        assert loaded is not None, name
        lens, manifest = loaded
        assert manifest["kind"] == kind and lens.key == LensKey(kind)
        for layer, matrix in matrices.items():
            assert torch.equal(lens.jacobians[layer], matrix)
    checkpoint = load_lens_checkpoint(_MODEL, "r")
    assert checkpoint is not None and checkpoint[0].n_prompts == 2
    assert torch.equal(checkpoint[0].jacobians[1], originals["ckpt"][1])
    # Shards were renamed, never rewritten.
    new_root = model_dir(_MODEL) / "lens"
    migrated = {path.name: path for path in new_root.rglob("*.safetensors")}
    assert len(migrated) == len(shard_bytes)
    for old_name, data in shard_bytes.items():
        new_name = "out." + old_name[len("jlens."):]
        assert migrated[new_name].read_bytes() == data
    assert not list(new_root.rglob("jlens.fit.lock"))  # v6 locks stay behind

    assert LENS_SOURCES.active(_MODEL, "output") == SourceRef("local", "r")
    assert LENS_SOURCES.active(_MODEL, "input") is None
    loaded_active = load_lens(_MODEL)
    assert loaded_active is not None and loaded_active[0].key == LensKey("R")
    binding = load_external_lens_binding(_MODEL, "workspace-r")
    assert binding is not None and binding.key == LensKey("R")
    assert binding.files[LensKey("R")].sha256 == "c" * 64
    assert load_external_lens_binding(_MODEL, "neuronpedia") is None
    sidecar = load_local_lens_sidecar(_MODEL, "r")
    assert sidecar is not None and sidecar["estimator_policy"]["rules"] is not None

    # Idempotent: a migrated home has nothing left to do.
    assert module.migrate_home(_home, log=lines.append) == 0


def test_pre_fix_relp_pointer_migrates_as_an_r_lens(_home: Path) -> None:
    from drowse.lens.relp import RELP_RULES
    from drowse.lens.store import load_local_lens

    matrices = _matrices(3)
    _write_v6_lens(
        model_dir(_MODEL) / "jlens" / "local" / "relp", matrices,
        relp=True, pre_fix=True,
    )
    assert _script().migrate_home(_home, log=lambda _line: None) == 1
    loaded = load_local_lens(_MODEL, "r")
    assert loaded is not None
    lens, manifest = loaded
    assert lens.key == LensKey("R")
    assert manifest["estimator_policy"]["rules"] == RELP_RULES


def test_neuronpedia_binding_is_hashed_from_the_local_cache(_home: Path) -> None:
    from drowse.lens.sources import load_external_lens_binding

    _v6_home()
    digests: list[tuple[str, str, str]] = []

    def digest_of(repo: str, filename: str, revision: str) -> str:
        digests.append((repo, filename, revision))
        return "e" * 64

    _script().migrate_home(_home, log=lambda _line: None, digest_of=digest_of)
    assert digests == [("neuronpedia/jacobian-lens", "model/lens.pt", "np-commit")]
    binding = load_external_lens_binding(_MODEL, "neuronpedia")
    assert binding is not None
    assert binding.files[LensKey("J")].sha256 == "e" * 64
    assert binding.provenance["config_sha256"] == "d" * 64


def test_refuses_a_model_with_both_trees(_home: Path) -> None:
    module = _script()
    _v6_home()
    (model_dir(_MODEL) / "lens").mkdir()
    with pytest.raises(module.MigrationError, match="both jlens/ and lens/"):
        module.migrate_home(_home, log=lambda _line: None)
