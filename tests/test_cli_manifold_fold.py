"""Manifold folding for the disk-side inspection verbs.

Bundled & user concepts ship as 2-node ``pca`` manifolds, so ``manifold
compare`` / ``manifold why`` fold a fitted manifold tensor. These tests
synthesize a fitted 2-node manifold on disk and exercise the runners end-to-end.

The fixture uses the production ``fit_affine_subspace`` primitive and seeds a
per-model neutral cache for the Mahalanobis-only compare path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import pytest
import torch

from drowse.cli.context import CliError
from drowse.cli.main import main
from drowse.cli.verbs.manifold import _fold, _fold_all
from drowse.geometry.domains import CustomDomain
from drowse.geometry.subspace import fit_affine_subspace
from drowse.geometry.manifold import MANIFOLD_FIT_POLICY_VERSION, Manifold
from drowse.store.manifold_tensors import save_manifold
from drowse.base.paths import manifold_dir, model_dir, tensor_filename

_MODEL = "test/model"
_LAYERS = (2, 5)
_DIM = 8


@pytest.fixture(autouse=True)
def _isolated_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> Generator[None, None, None]:
    from drowse.store import selectors as _sel
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    _sel.invalidate()
    _seed_neutral_cache(_MODEL)
    yield
    _sel.invalidate()


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def _seed_neutral_cache(model_id: str, *, n: int = 64, seed: int = 5) -> None:
    """Write a per-model neutral-activation disk cache.

    ``manifold compare`` builds its (mandatory) whitener via
    ``LayerWhitener.from_cache(model_id)`` — there is no Euclidean path — so the
    sharded disk cache must exist for the layers the folded manifolds occupy. The
    probe-centering mean is derived from these activations (``X.mean(0)``), so
    there is no separate ``layer_means`` cache to seed.
    """
    import json

    from safetensors.torch import save_file
    from drowse.base.integrity import hash_file

    md = model_dir(model_id)
    md.mkdir(parents=True, exist_ok=True)
    acts: dict[str, torch.Tensor] = {}
    for L in _LAYERS:
        g = torch.Generator().manual_seed(seed * 13 + L)
        scale = 0.5 + torch.rand(_DIM, generator=g, dtype=torch.float32)
        mu = torch.randn(_DIM, generator=g, dtype=torch.float32) * 0.1
        X = torch.randn(n, _DIM, generator=g, dtype=torch.float32) * scale + mu
        acts[f"layer_{L}"] = X
    tensor_files: dict[str, str] = {}
    tensor_sha256: dict[str, str] = {}
    for layer in _LAYERS:
        tensor_path = md / f"neutral_activations.layer-{layer}.gen-test.safetensors"
        save_file({f"layer_{layer}": acts[f"layer_{layer}"]}, str(tensor_path))
        tensor_files[str(layer)] = tensor_path.name
        tensor_sha256[str(layer)] = hash_file(tensor_path)
    (md / "neutral_activations.json").write_text(json.dumps({
        "method": "neutral_activations",
        "format_version": 4,
        "capture_version": 1,
        "capture_sha256": "test-capture",
        "model_fingerprint": "test-fingerprint",
        "model_source_fingerprint": "test-source",
        "tensor_sha256": tensor_sha256,
        "tensor_files": tensor_files,
        "layers": list(_LAYERS),
        "tensor_schema": {
            str(layer): {
                "shape": list(acts[f"layer_{layer}"].shape),
                "dtype": "torch.float32",
            }
            for layer in _LAYERS
        },
        "n_prompts": n,
        "n_layers": len(_LAYERS),
    }))


def _write_fitted_manifold(
    ns: str, name: str, *, seed: int = 0, d: int = _DIM,
) -> Path:
    """Synthesize a fitted 2-node affine manifold tensor on disk (no model).

    Built via the production ``fit_affine_subspace`` per layer — node 0 = pos,
    node 1 = neg — exactly as ``ManifoldExtractionPipeline`` does for a 2-node
    ``pca`` fit, so the on-disk shape matches a real fit.
    """
    g = torch.Generator().manual_seed(seed)
    d0, d1 = _unit(torch.randn(d, generator=g)), _unit(torch.randn(d, generator=g))
    pos = {2: 5.0 + 1.1 * d0, 5: 3.0 + 0.6 * d1}
    neg = {2: 5.0 - 1.1 * d0, 5: 3.0 - 0.6 * d1}
    neutral = {2: 5.0 + 0.2 * d0, 5: 3.0 - 0.1 * d1}
    pos_label, neg_label = (
        name.split(".", 1) if "." in name else (name, f"{name}_neg")
    )
    layers = {}
    share = {}
    for L in _LAYERS:
        cent = torch.stack([pos[L].float(), neg[L].float()])  # (2, D), node 0 = pos
        _fit = fit_affine_subspace(
            cent, neutral_mean=neutral[L].float(), orient_to=0,
        )
        sub, mu_coords = _fit.subspace, _fit.mu_coords
        layers[L] = sub
        share[L] = float(torch.linalg.norm(mu_coords))  # any positive spread
    mfld = Manifold(
        name=name,
        domain=CustomDomain(1),
        node_labels=[pos_label, neg_label],
        node_coords=torch.tensor([[1.0], [-1.0]]),
        layers=layers,
        mahalanobis_share=share,
    )
    from drowse.store.manifold_authoring import create_discover_manifold_folder

    folder = manifold_dir(ns, name)
    if not (folder / "manifold.json").exists():
        folder = create_discover_manifold_folder(
            ns, name, "", fit_mode="pca",
            node_corpora={pos_label: ["positive"], neg_label: ["negative"]},
        )
    path = folder / tensor_filename(_MODEL, release=None)
    metadata: dict[str, object] = {
        "method": "manifold_discover_pca", "fit_mode": "pca",
        "share_metric": "euclidean",
        "model_fingerprint": "test-fingerprint",
    }
    manifest = folder / "manifold.json"
    if manifest.exists():
        from drowse.store.manifold_folder import ManifoldFolder

        metadata["nodes_sha256"] = ManifoldFolder.load(
            folder, verify_manifest=False,
        ).nodes_sha256()
        metadata["fit_policy_version"] = MANIFOLD_FIT_POLICY_VERSION
    save_manifold(mfld, path, metadata)
    if manifest.exists():
        from drowse.store.manifold_folder import ManifoldFolder

        ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
            path, path.with_suffix(".json"),
        )
    return path


# ---------------------------------------------------------------------------
# the io-level fold helper
# ---------------------------------------------------------------------------

class TestFoldHelper:
    def test_fold_returns_profile_for_fitted_manifold(self) -> None:
        _write_fitted_manifold("default", "happy.sad")
        folded = _fold("happy.sad", _MODEL)
        assert sorted(folded.profile.layers) == [2, 5]
        assert (folded.display, folded.namespace, folded.name) == ("happy.sad", "default", "happy.sad")

    def test_fold_unfitted_is_a_miss_with_a_fit_hint(self) -> None:
        with pytest.raises(CliError) as exc:
            _fold("happy.sad", _MODEL)
        assert exc.value.exit_code == 1 and "drowse manifold fit" in str(exc.value)

    def test_fold_bare_name_collision_raises(self) -> None:
        from drowse.store.selectors import AmbiguousSelectorError

        _write_fitted_manifold("default", "happy.sad")
        _write_fitted_manifold("alice", "happy.sad")
        with pytest.raises(AmbiguousSelectorError):
            _fold("happy.sad", _MODEL)
        folded = _fold("alice/happy.sad", _MODEL)
        assert (folded.display, folded.namespace) == ("alice/happy.sad", "alice")

    def test_fold_reads_transfer_and_role_variants(self) -> None:
        """Regression: compare/why folded only raw and ``sae-*`` tensors, so a
        transferred (``:from-<src>``) or role (``:role-<r>``) vector reported
        "no fitted manifold" although its tensor existed."""
        import shutil

        from drowse.base.paths import encode_release_id
        from drowse.store.manifold_tensors import load_manifold, save_manifold

        from drowse.store.manifold_folder import ManifoldFolder

        folder = _write_fitted_manifold("default", "happy.sad").parent
        raw = folder / tensor_filename(_MODEL)
        moved = folder / tensor_filename(_MODEL, transferred_from="src/model")
        manifold = load_manifold(raw)

        def publish(path: Path, metadata: dict[str, Any]) -> None:
            save_manifold(manifold, path, metadata)
            ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
                path, path.with_suffix(".json"),
            )

        publish(moved, {
            **manifold.metadata, "method": "manifold_procrustes_transfer",
            "source_model_id": "src/model", "source_model_fingerprint": "src-fp",
        })
        folded = _fold(f"happy.sad:from-{encode_release_id('src/model')}", _MODEL)
        assert folded.variant.startswith("from-") and sorted(folded.profile.layers) == [2, 5]
        with pytest.raises(CliError, match="no fitted manifold"):
            _fold(f"happy.sad:from-{encode_release_id('other/model')}", _MODEL)

        with pytest.raises(CliError, match="role"):
            _fold("happy.sad:role-pirate", _MODEL)
        publish(raw, {**manifold.metadata, "node_roles": ["pirate", "pirate"]})
        assert _fold("happy.sad:role-pirate", _MODEL).display == "happy.sad:role-pirate"
        shutil.rmtree(folder)

    def test_fold_all_keys_by_namespace(self) -> None:
        _write_fitted_manifold("default", "happy.sad", seed=1)
        _write_fitted_manifold("alice", "happy.sad", seed=2)
        _write_fitted_manifold("default", "warm.clinical", seed=3)
        assert set(_fold_all(_MODEL)) == {
            ("alice", "happy.sad"), ("default", "happy.sad"), ("default", "warm.clinical"),
        }


# ---------------------------------------------------------------------------
# manifold why — folds a fitted manifold
# ---------------------------------------------------------------------------

class TestWhyFold:
    def test_why_on_fitted_manifold(self, capsys: pytest.CaptureFixture[str]) -> None:
        _write_fitted_manifold("default", "happy.sad")
        main(["manifold", "why", "happy.sad", "-m", _MODEL])
        assert "happy.sad" in capsys.readouterr().out

    def test_why_unfitted_manifold_nudges_to_fit(self, capsys: pytest.CaptureFixture[str]) -> None:
        manifold_dir("default", "happy.sad").mkdir(parents=True, exist_ok=True)
        with pytest.raises(SystemExit) as exc:
            main(["manifold", "why", "happy.sad", "-m", _MODEL])
        assert exc.value.code == 1
        assert "manifold fit" in capsys.readouterr().err

    @pytest.mark.parametrize(("ref", "display"), [
        ("happy.sad", "happy.sad"), ("default/happy.sad", "default/happy.sad"),
    ])
    def test_why_json_display(
        self, ref: str, display: str, capsys: pytest.CaptureFixture[str],
    ) -> None:
        import json

        _write_fitted_manifold("default", "happy.sad")
        main(["manifold", "why", ref, "-m", _MODEL, "-j"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["concept"] == display
        assert {row["layer"] for row in payload["layers"]} == {2, 5}


# ---------------------------------------------------------------------------
# manifold compare — folds fitted manifolds (named + 1-arg rank-all).
# Mahalanobis-only: the fixture seeds the per-model neutral cache.
# ---------------------------------------------------------------------------

class TestCompareFold:
    def test_compare_two_fitted_manifolds(self, capsys: pytest.CaptureFixture[str]) -> None:
        _write_fitted_manifold("default", "happy.sad", seed=1)
        _write_fitted_manifold("default", "warm.clinical", seed=2)
        main(["manifold", "compare", "happy.sad", "warm.clinical", "-m", _MODEL])
        out = capsys.readouterr().out
        assert "happy.sad" in out and "warm.clinical" in out

    def test_compare_one_arg_ranks_fitted_manifolds(self, capsys: pytest.CaptureFixture[str]) -> None:
        _write_fitted_manifold("default", "happy.sad", seed=1)
        _write_fitted_manifold("default", "warm.clinical", seed=2)
        _write_fitted_manifold("default", "angry.calm", seed=3)
        main(["manifold", "compare", "happy.sad", "-m", _MODEL])
        out = capsys.readouterr().out
        assert "warm.clinical" in out and "angry.calm" in out
        assert "happy.sad  " not in out  # the target never ranks against itself

    def test_compare_explicit_namespace_collision_keeps_both(self, capsys: pytest.CaptureFixture[str]) -> None:
        _write_fitted_manifold("default", "happy.sad", seed=1)
        _write_fitted_manifold("alice", "happy.sad", seed=2)
        main(["manifold", "compare", "default/happy.sad", "alice/happy.sad", "-m", _MODEL])
        out = capsys.readouterr().out
        assert "default/happy.sad" in out and "alice/happy.sad" in out

    def test_compare_one_arg_rank_all_keeps_namespace_collision(self, capsys: pytest.CaptureFixture[str]) -> None:
        _write_fitted_manifold("default", "happy.sad", seed=1)
        _write_fitted_manifold("alice", "warm.clinical", seed=2)
        _write_fitted_manifold("default", "warm.clinical", seed=3)
        main(["manifold", "compare", "happy.sad", "-m", _MODEL])
        out = capsys.readouterr().out
        assert "alice/warm.clinical" in out and "default/warm.clinical" in out


def _make_full_manifold(ns: str, name: str, *, seed: int = 0) -> Path:
    """A complete manifold folder (manifold.json + nodes) plus a fitted tensor."""
    from drowse.store.manifold_authoring import create_discover_manifold_folder
    from drowse.store.manifold_folder import ManifoldFolder, hash_manifold_files
    pos_label, neg_label = (
        name.split(".", 1) if "." in name else (name, f"{name}_neg")
    )
    create_discover_manifold_folder(
        ns, name, f"test {name}", fit_mode="pca",
        node_corpora={pos_label: ["a", "b"], neg_label: ["c", "d"]},
        hyperparams={"max_dim": 1, "var_threshold": 0.70},
    )
    folder = _write_fitted_manifold(ns, name, seed=seed).parent
    # Refresh the manifest so the fitted tensor passes integrity on load.
    ManifoldFolder.load(folder).write_metadata(files=hash_manifold_files(folder))
    return folder


# ---------------------------------------------------------------------------
# manifold bake — folds fitted manifold components
# ---------------------------------------------------------------------------

class TestMergeFold:
    def test_merge_folds_manifold_components(self) -> None:
        from drowse.fit.bake import merge_into_manifold
        from drowse.base.paths import safe_model_id

        _write_fitted_manifold("default", "happy.sad", seed=1)
        _write_fitted_manifold("default", "warm.clinical", seed=2)
        dst = merge_into_manifold(
            "merged",
            "0.5 default/happy.sad + 0.5 default/warm.clinical",
            model=_MODEL,
        )
        assert (dst / f"{safe_model_id(_MODEL)}.safetensors").exists()

    def test_merge_missing_component_errors(self) -> None:
        from drowse.errors import MergeError
        from drowse.fit.bake import merge_into_manifold

        _write_fitted_manifold("default", "happy.sad", seed=1)
        with pytest.raises(MergeError):
            merge_into_manifold(
                "merged",
                "0.5 default/happy.sad + 0.5 default/nonexistent",
                model=_MODEL,
            )


# ---------------------------------------------------------------------------
# pack export gguf — folds a fitted manifold
# ---------------------------------------------------------------------------

class TestGgufFold:
    def test_export_gguf_folds_manifold(self, tmp_path: Path) -> None:
        pytest.importorskip("gguf")  # writing the GGUF needs the optional extra
        from drowse.store.gguf import export_gguf_manifold

        _make_full_manifold("default", "happy.sad")
        # ``tmp_path`` is DROWSE_HOME here; exports must land outside it.
        out = tmp_path.parent / f"{tmp_path.name}-out" / "happy.gguf"
        written = export_gguf_manifold(
            "default", "happy.sad",
            model_scope=_MODEL, output=str(out), model_hint="llama",
        )
        assert written == [out]
        assert out.is_file()

    def test_export_gguf_defaults_to_cwd_and_leaves_folder_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression: a default GGUF export used to land inside the manifold
        folder and extend ``manifold.json::files`` with the ``.gguf``, which
        then made ``.drowse`` export of that manifold fail."""
        pytest.importorskip("gguf")
        from drowse.store.archive import export_drowse_archive
        from drowse.store.gguf import export_gguf_manifold

        folder = _make_full_manifold("local", "happy.sad")
        manifest_before = (folder / "manifold.json").read_bytes()
        listing_before = sorted(p.name for p in folder.iterdir())
        cwd = tmp_path.parent / f"{tmp_path.name}-cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)

        written = export_gguf_manifold(
            "local", "happy.sad",
            model_scope=_MODEL, output=None, model_hint="llama",
        )

        assert written == [cwd / "happy.sad.gguf"]
        assert written[0].is_file()
        assert (folder / "manifold.json").read_bytes() == manifest_before
        assert sorted(p.name for p in folder.iterdir()) == listing_before
        archive = export_drowse_archive("local", "happy.sad", output=cwd / "x.drowse")
        assert archive.is_file()

    def test_export_gguf_refuses_destination_inside_drowse_home(
        self, tmp_path: Path,
    ) -> None:
        from drowse.store.gguf import export_gguf_manifold

        folder = _make_full_manifold("local", "happy.sad")
        with pytest.raises(RuntimeError, match="inside the drowse store"):
            export_gguf_manifold(
                "local", "happy.sad",
                model_scope=_MODEL, output=str(folder), model_hint="llama",
            )
        assert not list(folder.glob("*.gguf"))

    def test_export_gguf_unfitted_errors(self, tmp_path: Path) -> None:
        from drowse.store.gguf import export_gguf_manifold

        # manifold.json but no fitted tensor for the model.
        from drowse.store.manifold_authoring import create_discover_manifold_folder
        create_discover_manifold_folder(
            "default", "happy.sad", "x", fit_mode="pca",
            node_corpora={"happy": ["a"], "sad": ["b"]},
            hyperparams={"max_dim": 1},
        )
        with pytest.raises(RuntimeError, match="no fitted manifold"):
            export_gguf_manifold(
                "default", "happy.sad",
                model_scope=_MODEL, output=str(tmp_path / "x.gguf"),
                model_hint="llama",
            )

    def test_export_preflight_skips_unrelated_variant_hashes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from drowse.store import gguf as gguf_io
        from drowse.base import integrity
        from drowse.store.gguf import export_gguf_manifold
        from drowse.store.manifold_folder import ManifoldFolder

        folder = _make_full_manifold("default", "happy.sad")
        raw = folder / tensor_filename(_MODEL)
        unrelated = folder / tensor_filename(_MODEL, release="unrelated")
        unrelated.write_bytes(raw.read_bytes())
        unrelated.with_suffix(".json").write_bytes(
            raw.with_suffix(".json").read_bytes()
        )
        ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
            unrelated, unrelated.with_suffix(".json"),
        )
        integrity._FINGERPRINT_CACHE.clear()
        real_hash = integrity.hash_file
        hashed: list[str] = []

        def track_hash(path: Path) -> str:
            hashed.append(Path(path).name)
            return real_hash(Path(path))

        def fake_write(_profile: object, path: Path, **_kwargs: object) -> Path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"gguf")
            return Path(path)

        monkeypatch.setattr(integrity, "hash_file", track_hash)
        monkeypatch.setattr(gguf_io, "write_gguf_profile", fake_write)
        export_gguf_manifold(
            "default", "happy.sad", model_scope=_MODEL,
            output=str(tmp_path.parent / f"{tmp_path.name}-out" / "out.gguf"),
            model_hint="llama",
        )

        assert raw.name in hashed and raw.with_suffix(".json").name in hashed
        assert unrelated.name not in hashed
        assert unrelated.with_suffix(".json").name not in hashed


def test_default_probe_preflight_skips_unrelated_variant_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.store.manifold_tensors import load_manifold
    from drowse.base import integrity
    from drowse.store.manifold_folder import ManifoldFolder
    import drowse.store.bundled as bundled_module
    import drowse.store.bundled_probes as probes_module

    folder = _make_full_manifold("default", "probe")
    raw = folder / tensor_filename(_MODEL)
    unrelated = folder / tensor_filename(_MODEL, release="unrelated")
    unrelated.write_bytes(raw.read_bytes())
    unrelated.with_suffix(".json").write_bytes(raw.with_suffix(".json").read_bytes())
    ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
        unrelated, unrelated.with_suffix(".json"),
    )
    integrity._FINGERPRINT_CACHE.clear()
    real_hash = integrity.hash_file
    hashed: list[str] = []

    def track_hash(path: Path) -> str:
        hashed.append(Path(path).name)
        return real_hash(Path(path))

    monkeypatch.setattr(integrity, "hash_file", track_hash)
    monkeypatch.setattr(probes_module, "load_default_manifolds", lambda: {})
    monkeypatch.setattr(bundled_module, "bundled_manifold_names", lambda: ["probe"])
    probes = probes_module.bootstrap_probe_roster(
        [], model_id=_MODEL, load=lambda _key: load_manifold(raw),
        fit=lambda _folder, _progress: None, include_fitted_defaults=True,
    )

    assert "default/probe" in probes
    assert raw.name in hashed and raw.with_suffix(".json").name in hashed
    assert unrelated.name not in hashed
    assert unrelated.with_suffix(".json").name not in hashed
