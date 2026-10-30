"""On-disk manifold format, authoring, lifecycle, and bundled tests — CPU-only."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

import torch

from drowse.store.alignment import LayerAlignment
from drowse.store.manifold_schema import (
    MANIFOLD_FORMAT_VERSION,
    BakedManifoldError,
    ManifoldFormatError,
    ManifoldSidecar,
    min_nodes,
)
from drowse.store.manifold_folder import (
    DiscoverGenerationPlan,
    ManifoldFolder,
    hash_manifold_files,
)
from drowse.store.manifold_authoring import (
    append_discover_manifold_node,
    BakedTensor,
    publish_baked_manifold,
    create_discover_manifold_folder,
    create_manifold_folder,
    init_discover_manifold_folder,
    iter_manifold_folders,
    merge_discover_manifolds,
    plan_discover_generation,
    update_manifold_folder,
)
from drowse.store.manifold_lifecycle import (
    clear_manifold_tensors,
    manifold_summary,
    preflight_transfer_manifold,
    refresh_manifold,
    remove_manifold_folder,
    transfer_manifold,
)
from drowse.base.paths import safe_model_id, tensor_filename

_STUB_TENSOR = tensor_filename("stub/model")
_STUB_SIDECAR = Path(_STUB_TENSOR).with_suffix(".json").name


def _alignment(dense: torch.Tensor) -> LayerAlignment:
    target_dim, source_dim = dense.shape
    return LayerAlignment(dense, torch.eye(source_dim), torch.zeros(target_dim))


def _alignments(dense: dict[int, torch.Tensor]) -> dict[int, LayerAlignment]:
    return {layer: _alignment(value) for layer, value in dense.items()}


def _box1d(periodic: bool, labels: list[str]) -> dict[str, Any]:
    """A 1-D box domain spec with evenly-spaced node coords."""
    k = len(labels)
    axis = (
        {"name": "t", "periodic": True, "period": 1.0, "lo": 0.0, "hi": 1.0}
        if periodic
        else {"name": "t", "periodic": False, "period": 1.0, "lo": 0.0, "hi": 1.0}
    )
    if periodic:
        coords = [[i / k] for i in range(k)]
    else:
        coords = [[i / (k - 1) if k > 1 else 0.0] for i in range(k)]
    return {
        "domain": {"type": "box", "axes": [axis]},
        "nodes": [
            {"label": label, "coords": coords[i]}
            for i, label in enumerate(labels)
        ],
    }


def _author_manifold(
    root: Path,
    *,
    name: str = "mood",
    periodic: bool = True,
    labels: list[str] | None = None,
    files: dict[str, Any] | None = None,
    domain: dict[str, Any] | None = None,
    nodes: list[dict[str, Any]] | None = None,
) -> Path:
    """Hand-author an exact current manifold folder; return its path."""
    if labels is None:
        labels = ["calm", "uneasy", "afraid", "frantic"]
    folder = root / name
    (folder / "nodes").mkdir(parents=True)
    spec = _box1d(periodic, labels)
    if domain is not None:
        spec["domain"] = domain
    if nodes is not None:
        spec["nodes"] = nodes
    for node in spec["nodes"]:
        node.setdefault("role", None)
        node.setdefault("kind", None)
    for idx, node in enumerate(spec["nodes"]):
        statements = [f"{node['label']} statement {i}" for i in range(3)]
        (folder / "nodes" / f"{idx:02d}_{node['label']}.json").write_text(
            json.dumps(statements)
        )
    meta = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": name,
        "description": "a mood manifold",
        "fit_mode": "authored",
        "domain": spec["domain"],
        "nodes": spec["nodes"],
        "files": files if files is not None else {},
        "source": "local",
        "tags": [],
        "template_ref": None,
    }
    (folder / "manifold.json").write_text(json.dumps(meta))
    return folder


def _sidecar_payload(
    *,
    name: str = "mood",
    labels: list[str] | None = None,
    domain: dict[str, Any] | None = None,
    fit_mode: str = "authored",
) -> dict[str, Any]:
    """Exact current fitted-sidecar payload for persistence unit tests."""
    node_labels = labels or ["a", "b", "c"]
    return {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": name,
        "method": "manifold_pca",
        "drowse_version": "0",
        "domain": domain or {
            "type": "box",
            "axes": [
                {"name": "t", "periodic": False, "period": 1.0, "lo": 0.0, "hi": 1.0},
            ],
        },
        "node_count": len(node_labels),
        "node_labels": node_labels,
        "feature_space": "raw",
        "fit_mode": fit_mode,
        "hyperparams": {},
        "diagnostics": {},
        "node_spread_per_layer": {},
        "mahalanobis_share_per_layer": {},
        "origin_per_layer": {},
        "nodes_sha256": None,
        "sae_release": None,
        "sae_revision": None,
        "sae_fingerprint": None,
        "sae_ids_by_layer": {},
        "sae_full_coverage": False,
        "model_fingerprint": None,
        "context_binding_sha256": None,
        "model_source_fingerprint": None,
        "capture_sha256": None,
        "capture_version": None,
        "capture_render_sha256": None,
        "baseline_prompts_sha256": None,
        "fitted_layers": [],
        "fit_policy_version": None,
        "share_metric": None,
        "subspace_metric": None,
        "rbf_smoothing_per_layer": {},
        "sigma_field_per_layer": {},
        "resolved_fit_mode": None,
        "topology_winner": None,
        "topology_candidates": [],
        "node_roles": [None] * len(node_labels),
        "node_kinds": [None] * len(node_labels),
        "components": None,
        "bake_policy": None,
        "source_model_id": None,
        "source_model_fingerprint": None,
        "transfer_quality_estimate": None,
    }


def test_min_nodes_per_dimension():
    assert min_nodes(1) == 3
    assert min_nodes(2) == 5
    assert min_nodes(3) == 7


def test_load_minimal_manifold(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    mf = ManifoldFolder.load(folder)
    assert mf.name == "mood"
    assert mf.domain["type"] == "box"
    assert mf.node_labels == ["calm", "uneasy", "afraid", "frantic"]
    assert len(mf.node_coords) == 4
    assert mf.description == "a mood manifold"


def test_node_groups(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    mf = ManifoldFolder.load(folder)
    groups = mf.node_groups()
    assert [label for label, _ in groups] == mf.node_labels
    label, statements = groups[0]
    assert label == "calm"
    assert statements == ["calm statement 0", "calm statement 1",
                          "calm statement 2"]


def test_missing_manifold_json_raises(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(tmp_path / "empty")


def test_stale_format_version_raises(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    meta = json.loads((folder / "manifold.json").read_text())
    meta["format_version"] = 2
    (folder / "manifold.json").write_text(json.dumps(meta))
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def test_newer_format_version_raises(tmp_path: Path):
    # A manifold authored by a future Drowse must not load silently.
    folder = _author_manifold(tmp_path)
    meta = json.loads((folder / "manifold.json").read_text())
    meta["format_version"] = MANIFOLD_FORMAT_VERSION + 1
    (folder / "manifold.json").write_text(json.dumps(meta))
    with pytest.raises(ManifoldFormatError, match="need exactly"):
        ManifoldFolder.load(folder)


@pytest.mark.parametrize("field", ["format_version", "name", "fit_mode"])
def test_current_manifest_requires_identity_fields(tmp_path: Path, field: str) -> None:
    folder = _author_manifold(tmp_path)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop(field)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def test_domain_required(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    meta = json.loads((folder / "manifold.json").read_text())
    del meta["domain"]
    (folder / "manifold.json").write_text(json.dumps(meta))
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def test_too_few_nodes_raises(tmp_path: Path):
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(
            _author_manifold(tmp_path, labels=["calm", "afraid"])
        )


def test_too_few_nodes_for_2d(tmp_path: Path):
    # A 2-D domain needs min_nodes(2) == 5; four nodes is too few.
    domain = {
        "type": "box",
        "axes": [
            {"name": "u", "periodic": False, "period": 1.0, "lo": 0.0, "hi": 1.0},
            {"name": "v", "periodic": False, "period": 1.0, "lo": 0.0, "hi": 1.0},
        ],
    }
    nodes = [
        {"label": f"n{i}", "coords": [x, y]}
        for i, (x, y) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)])
    ]
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(
            _author_manifold(tmp_path, domain=domain, nodes=nodes)
        )


def test_node_coords_dim_mismatch_raises(tmp_path: Path):
    # 1-D domain but a node carrying two coordinates.
    nodes = [
        {"label": "a", "coords": [0.0, 0.0]},
        {"label": "b", "coords": [0.5]},
        {"label": "c", "coords": [1.0]},
    ]
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(_author_manifold(tmp_path, nodes=nodes))


def test_bad_node_label_raises(tmp_path: Path):
    nodes = [
        {"label": "calm", "coords": [0.0]},
        {"label": "Uneasy", "coords": [0.5]},   # uppercase invalid
        {"label": "afraid", "coords": [1.0]},
    ]
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(_author_manifold(tmp_path, nodes=nodes))


def test_duplicate_labels_raises(tmp_path: Path):
    nodes = [
        {"label": "calm", "coords": [0.0]},
        {"label": "calm", "coords": [0.5]},
        {"label": "afraid", "coords": [1.0]},
    ]
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(_author_manifold(tmp_path, nodes=nodes))


def test_poisedness_soft_warning(tmp_path: Path):
    # Five nodes on a 2-D domain but all collinear — a soft warning, not
    # a hard error (the hard error is raised later at fit time).
    domain = {
        "type": "box",
        "axes": [
            {"name": "u", "periodic": False, "period": 1.0, "lo": 0.0, "hi": 1.0},
            {"name": "v", "periodic": False, "period": 1.0, "lo": 0.0, "hi": 1.0},
        ],
    }
    nodes = [
        {"label": f"n{i}", "coords": [t, t]}      # the diagonal line
        for i, t in enumerate([0.0, 0.25, 0.5, 0.75, 1.0])
    ]
    from drowse.store.manifold_authoring import authoring_advisories

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # loading never warns; advice is authoring-time
        mf = ManifoldFolder.load(_author_manifold(tmp_path, domain=domain, nodes=nodes))
    advisories = authoring_advisories(mf.name, mf.domain, mf.node_coords)
    assert any("poised" in advisory for advisory in advisories)


def test_missing_node_file_raises(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    (folder / "nodes" / "00_calm.json").unlink()
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def _add_dummy_tensor(folder: Path) -> None:
    """Write a placeholder fitted tensor + sidecar (no model needed)."""
    (folder / _STUB_TENSOR).write_bytes(b"placeholder-tensor")
    (folder / _STUB_SIDECAR).write_text(json.dumps(_sidecar_payload(
        labels=["calm", "uneasy", "afraid", "frantic"],
        domain={"type": "box", "axes": [
            {"name": "t", "periodic": True, "period": 1.0, "lo": 0.0, "hi": 1.0},
        ]},
    )))


def test_integrity_check_catches_tampering(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    files = hash_manifold_files(folder)
    meta = json.loads((folder / "manifold.json").read_text())
    meta["files"] = files
    (folder / "manifold.json").write_text(json.dumps(meta))
    ManifoldFolder.load(folder)
    (folder / _STUB_TENSOR).write_bytes(b"tampered")
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


def test_missing_sidecar_raises(tmp_path: Path):
    # A fitted .safetensors without its .json sidecar is a corrupt folder.
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    (folder / _STUB_SIDECAR).unlink()
    with pytest.raises(ManifoldFormatError, match="no sidecar"):
        ManifoldFolder.load(folder)


def test_node_corpus_edits_do_not_trip_integrity(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    files = hash_manifold_files(folder)
    meta = json.loads((folder / "manifold.json").read_text())
    meta["files"] = files
    (folder / "manifold.json").write_text(json.dumps(meta))
    (folder / "nodes" / "00_calm.json").write_text(json.dumps(["edited"]))
    ManifoldFolder.load(folder)


def test_nodes_sha256_stable_and_sensitive(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    mf = ManifoldFolder.load(folder)
    h1 = mf.nodes_sha256()
    assert h1 == ManifoldFolder.load(folder).nodes_sha256()
    (folder / "nodes" / "00_calm.json").write_text(json.dumps(["changed"]))
    h2 = ManifoldFolder.load(folder).nodes_sha256()
    assert h1 != h2


def test_fit_parse_can_skip_historical_manifest_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    mf = ManifoldFolder.load(folder)
    mf.write_metadata()

    import drowse.store.manifold_folder as folder_module

    def _must_not_verify(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("fit-specific parse hashed the full manifest")

    monkeypatch.setattr(folder_module, "verify_integrity", _must_not_verify)
    loaded = ManifoldFolder.load(folder, verify_manifest=False)
    assert loaded.name == mf.name


def test_nodes_sha256_sensitive_to_coords(tmp_path: Path):
    # Moving a node's authoring coordinate must invalidate the hash even
    # when the corpus is untouched — the fit depends on the geometry.
    folder = _author_manifold(tmp_path)
    h1 = ManifoldFolder.load(folder).nodes_sha256()
    meta = json.loads((folder / "manifold.json").read_text())
    meta["nodes"][1]["coords"] = [0.123]
    (folder / "manifold.json").write_text(json.dumps(meta))
    h2 = ManifoldFolder.load(folder).nodes_sha256()
    assert h1 != h2


def test_write_metadata_populates_files(tmp_path: Path):
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    mf = ManifoldFolder.load(folder)
    assert mf.files == {}
    mf.write_metadata()
    # Metadata-only rewrites preserve trusted entries; fitted writers name
    # their exact successful outputs explicitly.
    assert mf.files == {}
    mf.update_file_hashes(
        folder / _STUB_TENSOR, folder / _STUB_SIDECAR,
    )
    assert _STUB_TENSOR in mf.files
    assert _STUB_SIDECAR in mf.files
    assert not any(k.startswith("nodes/") for k in mf.files)
    reloaded = ManifoldFolder.load(folder)
    assert reloaded.files == mf.files


def test_update_file_hashes_only_reads_new_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    _add_dummy_tensor(folder)
    mf = ManifoldFolder.load(folder)
    mf.write_metadata()
    new_tensor = folder / "other-model.safetensors"
    new_sidecar = folder / "other-model.json"
    new_tensor.write_bytes(b"new tensor")
    new_sidecar.write_bytes((folder / _STUB_SIDECAR).read_bytes())

    from drowse.store import manifold_folder as folder_mod
    real_hash = folder_mod.hash_file
    hashed: list[Path] = []

    def _spy(path: Path) -> str:
        hashed.append(Path(path))
        return real_hash(path)

    monkeypatch.setattr(folder_mod, "hash_file", _spy)
    mf.update_file_hashes(new_tensor, new_sidecar)

    assert hashed == [new_tensor, new_sidecar]
    assert ManifoldFolder.load(folder).files == mf.files


def test_update_file_hashes_rejects_unreadable_future_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = MANIFOLD_FORMAT_VERSION + 1
    manifest["files"] = ["future-schema"]
    manifest_path.write_text(json.dumps(manifest))
    tensor = folder / "new-model.safetensors"
    tensor.write_bytes(b"future tensor")

    from drowse.store import manifold_folder as folder_mod
    def _unexpected_hash(_path: Path) -> str:
        pytest.fail("an unreadable manifest must be rejected before payload hashing")

    monkeypatch.setattr(folder_mod, "hash_file", _unexpected_hash)
    with pytest.raises(ManifoldFormatError, match="need exactly"):
        mf.update_file_hashes(tensor)

    assert json.loads(manifest_path.read_text()) == manifest


def test_bundle_drift_ignores_local_fit_transaction_state() -> None:
    from drowse.store.bundled import payload_drift_sha256
    from drowse.store.manifold_schema import LOCAL_STATE_KEYS

    base = {"format_version": MANIFOLD_FORMAT_VERSION, "name": "mood", "files": {}}
    local = {
        **base,
        "files": {"model.safetensors": "a" * 64},
        "artifact_id": "local-generation",
        "fit_epochs": {"model:raw": 3},
    }
    assert payload_drift_sha256(
        json.dumps(base).encode(), strip_keys=LOCAL_STATE_KEYS,
    ) == payload_drift_sha256(json.dumps(local).encode(), strip_keys=LOCAL_STATE_KEYS)


def test_update_file_hashes_merges_latest_manifest_from_stale_instances(
    tmp_path: Path,
) -> None:
    """Concurrent fit snapshots cannot drop each other's fitted pair."""
    folder = _author_manifold(tmp_path)
    first = ManifoldFolder.load(folder)
    second = ManifoldFolder.load(folder)
    paths: list[Path] = []
    sidecar_payload = _sidecar_payload(
        labels=["calm", "uneasy", "afraid", "frantic"],
    )
    for stem in ("model-a", "model-b"):
        tensor = folder / f"{stem}.safetensors"
        sidecar = folder / f"{stem}.json"
        tensor.write_bytes(stem.encode())
        sidecar.write_text(json.dumps(sidecar_payload))
        paths.extend((tensor, sidecar))

    first.update_file_hashes(*paths[:2])
    second.update_file_hashes(*paths[2:])

    files = ManifoldFolder.load(folder).files
    assert set(paths_item.name for paths_item in paths) <= set(files)


def test_update_file_hashes_does_not_launder_unrelated_untracked_pair(
    tmp_path: Path,
) -> None:
    folder = _author_manifold(tmp_path)
    sidecar_payload = _sidecar_payload(
        labels=["calm", "uneasy", "afraid", "frantic"],
    )
    stale_tensor = folder / "stale.safetensors"
    stale_sidecar = folder / "stale.json"
    stale_tensor.write_bytes(b"untrusted")
    stale_sidecar.write_text(json.dumps(sidecar_payload))
    good_tensor = folder / "good.safetensors"
    good_sidecar = folder / "good.json"
    good_tensor.write_bytes(b"new")
    good_sidecar.write_text(json.dumps(sidecar_payload))

    mf = ManifoldFolder.load(folder, verify_manifest=False)
    mf.update_file_hashes(good_tensor, good_sidecar)

    files = ManifoldFolder.load(folder, verify_manifest=False).files
    assert set(files) == {"good.safetensors", "good.json"}


def test_manifold_sidecar_load(tmp_path: Path):
    path = tmp_path / "m.json"
    payload = _sidecar_payload(
        labels=["a", "b", "c", "d"],
        domain={"type": "sphere", "dim": 2},
    )
    payload.update({
        "method": "manifold_sae",
        "drowse_version": "3.1.0",
        "feature_space": "sae-gemma",
        "nodes_sha256": "deadbeef",
        "sae_release": "gemma",
        "sae_revision": "rev-1",
        "sae_fingerprint": "fingerprint",
        "sae_ids_by_layer": {"0": "mock-layer-0"},
    })
    path.write_text(json.dumps(payload))
    sc = ManifoldSidecar.load(path)
    assert sc.method == "manifold_sae"
    assert sc.domain == {"type": "sphere", "dim": 2}
    assert sc.node_count == 4
    assert sc.feature_space == "sae-gemma"
    assert sc.nodes_sha256 == "deadbeef"
    assert sc.sae_release == "gemma"
    # Optional diagnostics are empty when the current writer has none.
    assert sc.node_spread_per_layer == {}


def test_manifold_sidecar_node_spread_round_trips(tmp_path: Path):
    """The per-layer signal profile (``node_spread_per_layer``) round-trips."""
    path = tmp_path / "m.json"
    payload = _sidecar_payload(
        labels=["happy", "sad"],
        domain={"type": "custom", "embed_dim": 1, "bounds": None},
        fit_mode="pca",
    )
    payload.update({
        "method": "manifold_discover_pca",
        "drowse_version": "4.0.0",
        "fitted_layers": [5, 12, 20],
        "node_spread_per_layer": {"5": 0.5, "12": 8.25, "20": 3.0},
    })
    path.write_text(json.dumps(payload))
    sc = ManifoldSidecar.load(path)
    assert sc.node_spread_per_layer == {"5": 0.5, "12": 8.25, "20": 3.0}
    # The peak layer is the one with the largest whitened between-node spread.
    peak = max(sc.node_spread_per_layer.items(), key=lambda kv: kv[1])
    assert peak[0] == "12"


def test_manifold_sidecar_distinguishes_evaluated_from_dls_fitted_layers(
    tmp_path: Path,
) -> None:
    """Pre-DLS diagnostics may cover layers absent from the tensor roster."""
    path = tmp_path / "m.json"
    payload = _sidecar_payload(
        labels=["happy", "sad"],
        domain={"type": "custom", "embed_dim": 1, "bounds": None},
        fit_mode="pca",
    )
    payload.update({
        "method": "manifold_discover_pca",
        "fitted_layers": [5, 20],
        "node_spread_per_layer": {"5": 0.5, "12": 8.25, "20": 3.0},
        "mahalanobis_share_per_layer": {"5": 0.25, "20": 1.5},
    })
    path.write_text(json.dumps(payload))

    sc = ManifoldSidecar.load(path)

    assert sc.fitted_layers == [5, 20]
    assert sc.evaluated_layers == [5, 12, 20]


def test_manifold_sidecar_topology_provenance_round_trips(tmp_path: Path):
    """``auto``-fit topology provenance survives the save → load round-trip.

    Regression: ``save_manifold`` filters metadata through a key whitelist, and
    ``resolved_fit_mode`` / ``topology_winner`` / ``topology_candidates`` (built
    by the extraction pipeline for an ``auto`` fit) were absent from it — so they
    were silently dropped from the on-disk sidecar even though the documented
    contract said the resolved geometry is recorded there.  The in-memory
    ``manifold.metadata`` carried them (``.update`` after save), which is why the
    extraction tests stayed green while the round-trip lost them.
    """
    import torch
    from drowse.geometry.manifold import (
        BoxAxis, BoxDomain, Manifold, fit_layer_subspace,
    )
    from drowse.store.manifold_tensors import load_manifold, save_manifold

    g = torch.Generator().manual_seed(0)
    domain = BoxDomain([BoxAxis("t", periodic=False, lo=0.0, hi=1.0)])
    coords = torch.tensor([[0.0], [0.5], [1.0]])
    embedded = domain.embed(coords)
    layers = {}
    for layer in (4, 5):
        sub, _ev_ratio = fit_layer_subspace(torch.randn(3, 6, generator=g), embedded)
        layers[layer] = sub
        man = Manifold(
            name="autotopo", domain=domain, node_labels=["a", "b", "c"],
            node_coords=coords, layers=layers,
            mahalanobis_share={4: 1.0, 5: 2.0},
            origin={
                    layer: torch.zeros(domain.intrinsic_dim)
                for layer, sub in layers.items() if not sub.is_affine
            },
            feature_space="sae-test",
        )
    candidates = [
        {"name": "flat-pca", "fit_mode": "pca", "intrinsic_dim": 2,
         "score": 12.5, "viable": True, "reason": None},
        {"name": "spectral", "fit_mode": "spectral", "intrinsic_dim": 1,
         "score": 40.0, "viable": True, "reason": None},
    ]
    out = tmp_path / "autotopo.safetensors"
    save_manifold(man, out, {
        "method": "manifold_discover_auto", "nodes_sha256": "x",
        "fit_mode": "auto", "resolved_fit_mode": "pca",
        "topology_winner": "flat-pca", "topology_candidates": candidates,
        "sae_release": "test", "sae_revision": "rev-1",
        "sae_fingerprint": "fingerprint",
        "sae_ids_by_layer": {"4": "mock-4", "5": "mock-5"},
    })
    # The written sidecar JSON carries the provenance (the whitelist passes it).
    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert sidecar["resolved_fit_mode"] == "pca"
    assert sidecar["topology_winner"] == "flat-pca"
    assert [c["name"] for c in sidecar["topology_candidates"]] == ["flat-pca", "spectral"]
    # And it surfaces on the loaded manifold's metadata (load reads the whole
    # sidecar back into ``metadata``).
    loaded = load_manifold(out)
    assert loaded.metadata["resolved_fit_mode"] == "pca"
    assert loaded.metadata["topology_winner"] == "flat-pca"


def test_save_manifold_rejects_unknown_metadata_key(tmp_path: Path):
    """A provenance key outside the schema must fail loudly, not vanish.

    The canonical-shape assert after the copy loop catches a *missing* sidecar
    field but is structurally blind to an extra one: extras are dropped before
    it runs.  A fit author adding a key would get no error and no persisted
    field.  The read-side ``_tensor_sha256`` annotation stays tolerated so a
    load → transfer → save round-trip still works.
    """
    import torch
    from drowse.geometry.manifold import (
        BoxAxis, BoxDomain, Manifold, fit_layer_subspace,
    )
    from drowse.store.manifold_tensors import load_manifold, save_manifold

    g = torch.Generator().manual_seed(0)
    domain = BoxDomain([BoxAxis("t", periodic=False, lo=0.0, hi=1.0)])
    coords = torch.tensor([[0.0], [0.5], [1.0]])
    sub, _ = fit_layer_subspace(torch.randn(3, 6, generator=g), domain.embed(coords))
    man = Manifold(
        name="unknownmeta", domain=domain, node_labels=["a", "b", "c"],
        node_coords=coords, layers={4: sub}, mahalanobis_share={4: 1.0},
        origin={4: torch.zeros(domain.intrinsic_dim)}, feature_space="sae-test",
    )
    sae_provenance = {
        "sae_release": "test", "sae_revision": "rev-1",
        "sae_fingerprint": "fingerprint", "sae_ids_by_layer": {"4": "mock-4"},
    }
    out = tmp_path / "unknownmeta.safetensors"
    with pytest.raises(ValueError, match="unknown key"):
        save_manifold(man, out, {
            "method": "manifold_sae", "fit_mode": "authored",
            "monopolar": True, **sae_provenance,
        })
    assert not out.exists()

    save_manifold(man, out, {
        "method": "manifold_sae", "fit_mode": "authored",
        "_tensor_sha256": "0" * 64, **sae_provenance,
    })
    assert "_tensor_sha256" not in json.loads(out.with_suffix(".json").read_text())
    assert load_manifold(out).name == "unknownmeta"


# --------------------------------------------------- authoring (create/update) ---

def _author_nodes(labels: list[str]) -> list[dict[str, Any]]:
    """A well-spread node list for a 1-D box, statements inline."""
    out = []
    for i, label in enumerate(labels):
        out.append({
            "label": label,
            "coords": [i / (len(labels) - 1)],
            "statements": [f"{label} statement {j}" for j in range(3)],
        })
    return out


def test_create_manifold_folder_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = _author_nodes(["calm", "uneasy", "afraid"])
    folder, advisories = create_manifold_folder(
        "local", "mood", "a mood axis", domain, nodes,
    )
    assert folder.exists()
    mf = ManifoldFolder.load(folder)
    assert mf.name == "mood"
    assert mf.node_labels == ["calm", "uneasy", "afraid"]
    assert mf.files == {}
    assert dict(mf.node_groups())["calm"][0] == "calm statement 0"
    # A well-spread 1-D layout draws no poisedness advisory.
    assert advisories == []


def test_create_manifold_folder_conflict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = _author_nodes(["a", "b", "c"])
    create_manifold_folder("local", "dup", "", domain, nodes)
    with pytest.raises(FileExistsError):
        create_manifold_folder("local", "dup", "", domain, nodes)


def test_create_manifold_folder_too_few_nodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    with pytest.raises(ManifoldFormatError):
        create_manifold_folder(
            "local", "thin", "", domain, _author_nodes(["a", "b"]),
        )


def test_create_manifold_folder_bad_coords_arity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = [
        {"label": "a", "coords": [0.0, 0.0], "statements": ["x"]},
        {"label": "b", "coords": [0.5], "statements": ["y"]},
        {"label": "c", "coords": [1.0], "statements": ["z"]},
    ]
    with pytest.raises(ManifoldFormatError):
        create_manifold_folder("local", "bad", "", domain, nodes)


def test_create_manifold_folder_empty_statements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = _author_nodes(["a", "b", "c"])
    nodes[1]["statements"] = []
    with pytest.raises(ManifoldFormatError):
        create_manifold_folder("local", "nostmt", "", domain, nodes)


def test_create_manifold_folder_bad_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    with pytest.raises(ManifoldFormatError):
        create_manifold_folder(
            "Bad Namespace", "m", "", domain, _author_nodes(["a", "b", "c"]),
        )


def test_update_manifold_folder_statements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = _author_nodes(["a", "b", "c"])
    folder, _ = create_manifold_folder("local", "mood", "", domain, nodes)
    sha_before = ManifoldFolder.load(folder).nodes_sha256()

    nodes[0]["statements"].append("a fresh statement")
    update_manifold_folder(folder, description="edited", nodes=nodes)
    mf = ManifoldFolder.load(folder)
    assert mf.description == "edited"
    assert "a fresh statement" in dict(mf.node_groups())["a"]
    # Editing the corpus must invalidate the staleness key.
    assert mf.nodes_sha256() != sha_before


def test_update_manifold_folder_relabels_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    relabeled = _author_nodes(["x", "y", "z"])
    update_manifold_folder(folder, nodes=relabeled)
    mf = ManifoldFolder.load(folder)
    assert mf.node_labels == ["x", "y", "z"]
    # No orphaned node files from the old labels.
    assert sorted(p.name for p in (folder / "nodes").iterdir()) == [
        "00_x.json", "01_y.json", "02_z.json",
    ]


def test_update_manifold_folder_does_not_hash_fitted_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    _fake_fit_tensor(folder, "test/model")
    from drowse.base import integrity
    hashed: list[Path] = []

    def unexpected_hash(path: Path) -> str:
        hashed.append(Path(path))
        raise AssertionError("metadata-only authoring hashed a fitted payload")

    monkeypatch.setattr(integrity, "hash_file", unexpected_hash)
    update_manifold_folder(folder, description="edited")
    assert hashed == []


def test_malformed_json_raises_format_error(tmp_path: Path):
    # A corrupt manifest must surface as ManifoldFormatError, not a bare
    # JSONDecodeError — the HTTP routes and iter_manifold_folders only
    # guard against the former.
    folder = _author_manifold(tmp_path)
    (folder / "manifold.json").write_text("{ not valid json")
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)


@pytest.mark.parametrize("root", ["[]", "null", "1"])
def test_non_object_manifest_raises_format_error(
    tmp_path: Path, root: str,
) -> None:
    folder = _author_manifold(tmp_path)
    (folder / "manifold.json").write_text(root)

    with pytest.raises(ManifoldFormatError, match="JSON object"):
        ManifoldFolder.load(folder)


def test_non_object_manifest_rejected_by_locked_metadata_writers(
    tmp_path: Path,
) -> None:
    folder = _author_manifold(tmp_path)
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    manifest_path = folder / "manifold.json"
    manifest_path.write_text("[]")
    tensor = folder / "new-model.safetensors"
    tensor.write_bytes(b"payload")

    with pytest.raises(ManifoldFormatError, match="JSON object"):
        mf.update_file_hashes(tensor)
    with pytest.raises(ManifoldFormatError, match="JSON object"):
        mf.write_metadata()


def test_write_metadata_rejects_concurrent_future_manifest(
    tmp_path: Path,
) -> None:
    folder = _author_manifold(tmp_path)
    stale = ManifoldFolder.load(folder, verify_manifest=False)
    manifest_path = folder / "manifold.json"
    future = json.loads(manifest_path.read_text())
    future["format_version"] = MANIFOLD_FORMAT_VERSION + 1
    manifest_path.write_text(json.dumps(future))

    with pytest.raises(ManifoldFormatError, match="need exactly"):
        stale.write_metadata()

    assert json.loads(manifest_path.read_text()) == future


def test_non_object_sidecar_raises_format_error(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_text("[]")

    with pytest.raises(ManifoldFormatError, match="JSON object"):
        ManifoldSidecar.load(path)


@pytest.mark.parametrize("version", [None, MANIFOLD_FORMAT_VERSION + 1])
def test_sidecar_requires_exact_current_format(
    tmp_path: Path, version: int | None,
) -> None:
    path = tmp_path / "model.json"
    payload = _sidecar_payload()
    if version is None:
        payload.pop("format_version")
    else:
        payload["format_version"] = version
    path.write_text(json.dumps(payload))

    with pytest.raises(ManifoldFormatError, match="need exactly"):
        ManifoldSidecar.load(path)


def test_sidecar_requires_current_identity_fields(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    payload = _sidecar_payload()
    payload.pop("fit_mode")
    path.write_text(json.dumps(payload))

    with pytest.raises(ManifoldFormatError, match="exact schema"):
        ManifoldSidecar.load(path)


def test_sidecar_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    payload = _sidecar_payload()
    payload["legacy_diagnostics"] = {}
    path.write_text(json.dumps(payload))

    with pytest.raises(ManifoldFormatError, match="exact schema"):
        ManifoldSidecar.load(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mahalanobis_share_per_layer", {"garbage": 1.0}),
        ("origin_per_layer", {"0": "not-coordinates"}),
        ("topology_candidates", [42]),
        ("components", {"0:x": {"selector": "x", "alpha": 1.0}}),
        ("components", {"0:x": {
            "selector": "x", "alpha": 1.0, "tensor_sha256": None,
        }}),
        ("diagnostics", {"invented_metric": 1.0}),
        ("hyperparams", {"invented_knob": 1}),
    ],
)
def test_sidecar_rejects_malformed_nested_contracts(
    tmp_path: Path, field: str, value: object,
) -> None:
    path = tmp_path / "model.json"
    payload = _sidecar_payload()
    payload["fitted_layers"] = [0]
    payload[field] = value
    path.write_text(json.dumps(payload))

    with pytest.raises(ManifoldFormatError):
        ManifoldSidecar.load(path)


def test_sidecar_rejects_unknown_method(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    payload = _sidecar_payload()
    payload["method"] = "invented_current_method"
    path.write_text(json.dumps(payload))

    with pytest.raises(ManifoldFormatError, match="method/fit_mode"):
        ManifoldSidecar.load(path)


def test_manifest_and_nodes_reject_unknown_fields(tmp_path: Path) -> None:
    folder = _author_manifold(tmp_path)
    manifest = json.loads((folder / "manifold.json").read_text())
    manifest["legacy_vectors"] = []
    (folder / "manifold.json").write_text(json.dumps(manifest))
    with pytest.raises(ManifoldFormatError, match="unknown field"):
        ManifoldFolder.load(folder)

    del manifest["legacy_vectors"]
    manifest["nodes"][0]["statements"] = ["inline legacy corpus"]
    (folder / "manifold.json").write_text(json.dumps(manifest))
    with pytest.raises(ManifoldFormatError, match="fields must be"):
        ManifoldFolder.load(folder)


def test_iter_manifold_folders_skips_malformed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "good", "", domain, _author_nodes(["a", "b", "c"]),
    )
    bad = folder.parent / "broken"
    (bad / "nodes").mkdir(parents=True)
    (bad / "manifold.json").write_text("{ truncated")
    found = {mf.name for _ns, mf in iter_manifold_folders()}
    assert found == {"good"}


def test_iter_manifold_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    create_manifold_folder("local", "one", "", domain, _author_nodes(["a", "b", "c"]))
    create_manifold_folder("shared", "two", "", domain, _author_nodes(["a", "b", "c"]))
    found = {(ns, mf.name) for ns, mf in iter_manifold_folders()}
    assert found == {("local", "one"), ("shared", "two")}
    only_local = {(ns, mf.name) for ns, mf in iter_manifold_folders("local")}
    assert only_local == {("local", "one")}


def test_iter_manifold_folders_is_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.store import manifold_folder as manifold_folder_module
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "one", "", domain, _author_nodes(["a", "b", "c"]),
    )
    _add_dummy_tensor(folder)
    ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
        folder / _STUB_TENSOR,
    )

    def _must_not_verify(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("ordinary manifold discovery verified payload hashes")

    monkeypatch.setattr(
        manifold_folder_module, "verify_integrity", _must_not_verify,
    )
    assert [(ns, mf.name) for ns, mf in iter_manifold_folders()] == [
        ("local", "one"),
    ]
    assert manifold_summary(folder)["name"] == "one"
    with pytest.raises(AssertionError, match="verified payload hashes"):
        ManifoldFolder.load(folder)


# ============================================================ discover mode ===

def _discover_corpora(labels: list[str]) -> dict[str, list[str]]:
    """``{label: [statement, ...]}`` for the discover authoring shape."""
    return {
        label: [f"{label} statement {i}" for i in range(3)]
        for label in labels
    }


def test_create_discover_manifold_folder_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A freshly authored discover folder loads with the expected shape."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "personas", "five personas",
        fit_mode="pca",
        node_corpora=_discover_corpora(
            ["pirate", "caveman", "assistant", "scholar", "robot"],
        ),
        hyperparams={"max_dim": 8, "var_threshold": 0.70},
    )
    mf = ManifoldFolder.load(folder)
    assert mf.name == "personas"
    assert mf.fit_mode == "pca"
    assert mf.is_discover
    assert mf.domain == {}
    assert mf.node_coords == []
    assert mf.node_labels == [
        "pirate", "caveman", "assistant", "scholar", "robot",
    ]
    assert mf.hyperparams == {"max_dim": 8, "var_threshold": 0.70}
    groups = mf.node_groups()
    assert [label for label, _ in groups] == mf.node_labels
    assert all(len(stmts) == 3 for _, stmts in groups)


def test_create_discover_manifold_rejects_unknown_fit_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    with pytest.raises(ManifoldFormatError, match="fit_mode"):
        create_discover_manifold_folder(
            "local", "bad", "",
            fit_mode="banana",
            node_corpora=_discover_corpora(["a", "b", "c"]),
        )


def test_create_discover_manifold_rejects_authored_fit_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``authored`` is not a valid discover-mode fit_mode."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    with pytest.raises(ManifoldFormatError, match="fit_mode"):
        create_discover_manifold_folder(
            "local", "wrong", "",
            fit_mode="authored",
            node_corpora=_discover_corpora(["a", "b", "c"]),
        )


def test_discover_manifold_rejects_coords_on_nodes(tmp_path: Path):
    """A discover folder must not carry per-node ``coords`` — those are derived."""
    folder = tmp_path / "leaky"
    (folder / "nodes").mkdir(parents=True)
    (folder / "nodes" / "00_a.json").write_text(json.dumps(["a says"]))
    (folder / "manifold.json").write_text(json.dumps({
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": "leaky",
        "description": "",
        "fit_mode": "pca",
        "hyperparams": {"max_dim": 4},
        "nodes": [{"label": "a", "coords": [0.5], "role": None, "kind": None}],
        "files": {},
        "source": "local", "tags": [], "template_ref": None,
    }))
    with pytest.raises(ManifoldFormatError, match="fields must be"):
        ManifoldFolder.load(folder)


def test_discover_manifold_rejects_domain_field(tmp_path: Path):
    """A discover folder must not carry a ``domain`` — coords are derived."""
    folder = tmp_path / "leaky2"
    (folder / "nodes").mkdir(parents=True)
    (folder / "nodes" / "00_a.json").write_text(json.dumps(["a says"]))
    (folder / "manifold.json").write_text(json.dumps({
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": "leaky2",
        "description": "",
        "fit_mode": "spectral",
        "hyperparams": {"max_dim": 4},
        "domain": {"type": "box", "axes": [
            {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0},
        ]},
        "nodes": [{"label": "a"}],
        "files": {},
    }))
    with pytest.raises(ManifoldFormatError, match="unknown field"):
        ManifoldFolder.load(folder)


def test_discover_nodes_sha256_sensitive_to_hyperparams(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A hyperparam edit invalidates the fit cache — the staleness key changes."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "pca_a", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["a", "b", "c"]),
        hyperparams={"max_dim": 8, "var_threshold": 0.70},
    )
    h1 = ManifoldFolder.load(folder).nodes_sha256()
    data = json.loads((folder / "manifold.json").read_text())
    data["hyperparams"] = {"max_dim": 4, "var_threshold": 0.70}
    (folder / "manifold.json").write_text(json.dumps(data))
    h2 = ManifoldFolder.load(folder).nodes_sha256()
    assert h1 != h2, "hyperparam change must invalidate the fit cache"


def test_discover_nodes_sha256_sensitive_to_fit_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Switching ``pca`` ↔ ``spectral`` invalidates the fit cache."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "mode_a", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["a", "b", "c"]),
        hyperparams={"max_dim": 8},
    )
    h_pca = ManifoldFolder.load(folder).nodes_sha256()
    data = json.loads((folder / "manifold.json").read_text())
    data["fit_mode"] = "spectral"
    (folder / "manifold.json").write_text(json.dumps(data))
    h_spec = ManifoldFolder.load(folder).nodes_sha256()
    assert h_pca != h_spec, "fit_mode change must invalidate the fit cache"


def test_discover_nodes_sha256_sensitive_to_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The standing invariant: a corpus edit invalidates the fit cache."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "corp_a", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["a", "b", "c"]),
    )
    h1 = ManifoldFolder.load(folder).nodes_sha256()
    (folder / "nodes" / "00_a.json").write_text(
        json.dumps(["a says something else"])
    )
    h2 = ManifoldFolder.load(folder).nodes_sha256()
    assert h1 != h2


def test_discover_write_metadata_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A re-written discover folder loads identically."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "rw", "round-trip",
        fit_mode="spectral",
        node_corpora=_discover_corpora(["a", "b", "c"]),
        hyperparams={"max_dim": 6, "k_nn": 5},
    )
    mf = ManifoldFolder.load(folder)
    mf.write_metadata()
    again = ManifoldFolder.load(folder)
    assert again.fit_mode == "spectral"
    assert again.hyperparams == {"max_dim": 6, "k_nn": 5}
    assert again.node_labels == mf.node_labels
    assert again.domain == {}
    assert again.node_coords == []


def test_discover_create_rejects_cross_method_hyperparams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    with pytest.raises(ManifoldFormatError, match="k_nn"):
        create_discover_manifold_folder(
            "local", "sanity_pca", "",
            fit_mode="pca",
            node_corpora=_discover_corpora(["a", "b", "c"]),
            hyperparams={"max_dim": 8, "var_threshold": 0.70, "k_nn": 5},
        )


def test_discover_create_rejects_unknown_hyperparams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    with pytest.raises(ManifoldFormatError, match="foo"):
        create_discover_manifold_folder(
            "local", "sanity_spec", "",
            fit_mode="spectral",
            node_corpora=_discover_corpora(["a", "b", "c"]),
            hyperparams={"max_dim": 8, "k_nn": 5, "foo": "bar"},
        )


# ---------------------------------------------------------------------------
# merge_discover_manifolds
# ---------------------------------------------------------------------------


def test_merge_discover_unions_nodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Two discover sources merge into a fresh discover folder whose
    node corpus is the union (in source order) of both inputs.
    """
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    create_discover_manifold_folder(
        "local", "src_a", "first heap",
        fit_mode="pca",
        node_corpora=_discover_corpora(["alpha", "beta"]),
        hyperparams={"max_dim": 8, "var_threshold": 0.7},
    )
    create_discover_manifold_folder(
        "local", "src_b", "second heap",
        fit_mode="pca",
        node_corpora=_discover_corpora(["gamma", "delta"]),
        hyperparams={"max_dim": 8, "var_threshold": 0.7},
    )
    target = merge_discover_manifolds(
        "local", "combined", "merged heap",
        sources=[("local", "src_a"), ("local", "src_b")],
    )
    mf = ManifoldFolder.load(target)
    assert mf.is_discover
    assert mf.fit_mode == "pca"
    assert mf.description == "merged heap"
    # Source order preserved, then per-source label order.
    assert mf.node_labels == ["alpha", "beta", "gamma", "delta"]
    # Hyperparams inherited from the first source.
    assert mf.hyperparams == {"max_dim": 8, "var_threshold": 0.7}
    # Every node carries the source corpus statements (3 each from
    # ``_discover_corpora``).
    groups = dict(mf.node_groups())
    for label in mf.node_labels:
        assert len(groups[label]) == 3


def test_merge_discover_does_not_hash_source_fitted_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folders = [
        create_discover_manifold_folder(
            "local", name, "", fit_mode="pca",
            node_corpora=_discover_corpora(labels),
        )
        for name, labels in (
            ("src_a", ["a", "b", "c"]),
            ("src_b", ["d", "e", "f"]),
        )
    ]
    for folder in folders:
        _fake_fit_tensor(folder, "test/model")
    from drowse.base import integrity
    hashed: list[Path] = []

    def unexpected_hash(path: Path) -> str:
        hashed.append(Path(path))
        raise AssertionError("metadata-only merge hashed a fitted payload")

    monkeypatch.setattr(integrity, "hash_file", unexpected_hash)
    merge_discover_manifolds(
        "local", "combined", "", sources=[
            ("local", "src_a"), ("local", "src_b"),
        ],
    )
    assert hashed == []


def test_merge_discover_refuses_authored_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Authored manifolds carry user-declared geometry; merge refuses
    them because there's no shared coordinate system to reconcile.
    """
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    _author_manifold(tmp_path / "manifolds" / "local", name="authored_src")
    create_discover_manifold_folder(
        "local", "discover_src", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["a", "b", "c"]),
    )
    with pytest.raises(ManifoldFormatError, match="authored"):
        merge_discover_manifolds(
            "local", "combined", "",
            sources=[("local", "authored_src"), ("local", "discover_src")],
        )


def test_merge_discover_refuses_label_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Same label in two sources isn't auto-renamed — refuse so the
    user resolves the collision deliberately (renaming hides
    provenance otherwise).
    """
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    create_discover_manifold_folder(
        "local", "src_a", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["shared", "a_only"]),
    )
    create_discover_manifold_folder(
        "local", "src_b", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["shared", "b_only"]),
    )
    with pytest.raises(ManifoldFormatError, match="collision"):
        merge_discover_manifolds(
            "local", "combined", "",
            sources=[("local", "src_a"), ("local", "src_b")],
        )


def test_merge_discover_refuses_mixed_fit_modes_without_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Sources with different fit_modes require an explicit override —
    the merged folder has one ``fit_mode``, so picking implicitly
    would silently lose information.
    """
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    create_discover_manifold_folder(
        "local", "src_pca", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["a", "b"]),
    )
    create_discover_manifold_folder(
        "local", "src_spec", "",
        fit_mode="spectral",
        node_corpora=_discover_corpora(["c", "d"]),
    )
    with pytest.raises(ManifoldFormatError, match="mixed fit_modes"):
        merge_discover_manifolds(
            "local", "combined", "",
            sources=[("local", "src_pca"), ("local", "src_spec")],
        )

    # Explicit override succeeds.
    target = merge_discover_manifolds(
        "local", "combined", "",
        sources=[("local", "src_pca"), ("local", "src_spec")],
        fit_mode="pca",
    )
    mf = ManifoldFolder.load(target)
    assert mf.fit_mode == "pca"


def test_merge_discover_refuses_missing_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A non-existent source raises FileNotFoundError before any folder
    is written — atomic-on-failure discipline."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    create_discover_manifold_folder(
        "local", "exists", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["a", "b"]),
    )
    with pytest.raises(FileNotFoundError):
        merge_discover_manifolds(
            "local", "combined", "",
            sources=[("local", "exists"), ("local", "missing")],
        )


def test_merge_discover_force_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An existing destination raises FileExistsError without
    ``force=True``; with ``force=True`` it's rebuilt clean.
    """
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    for nm, labels in (("src_a", ["a", "b"]), ("src_b", ["c", "d"])):
        create_discover_manifold_folder(
            "local", nm, "",
            fit_mode="pca",
            node_corpora=_discover_corpora(labels),
        )
    sources = [("local", "src_a"), ("local", "src_b")]
    merge_discover_manifolds("local", "combined", "v1", sources=sources)
    with pytest.raises(FileExistsError):
        merge_discover_manifolds("local", "combined", "v2", sources=sources)
    # force=True rebuilds.
    target = merge_discover_manifolds(
        "local", "combined", "v2", sources=sources, force=True,
    )
    mf = ManifoldFolder.load(target)
    assert mf.description == "v2"


def test_merge_discover_refuses_under_two_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Single-source 'merge' is meaningless — refuse with a clear error."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    create_discover_manifold_folder(
        "local", "only", "",
        fit_mode="pca",
        node_corpora=_discover_corpora(["a", "b"]),
    )
    with pytest.raises(ValueError, match=">= 2"):
        merge_discover_manifolds(
            "local", "combined", "",
            sources=[("local", "only")],
        )


# ============================================================ A1: label regex ===
#
# Node labels are stricter than the artifact NAME_REGEX — they drop ``.``
# (reserved as the bipolar-pole separator and unaddressable via ``%label``).


def test_dotted_node_label_rejected_authored_load(tmp_path: Path):
    """A dotted label is invalid even though it matches the old NAME_REGEX.

    ``deer.wolf`` passes ``^[a-z][a-z0-9._-]{0,63}$`` but the dot is the
    bipolar separator and the steering-expr lexer can't address it via
    ``%deer.wolf`` — so it must be rejected at load.
    """
    nodes = [
        {"label": "calm", "coords": [0.0]},
        {"label": "deer.wolf", "coords": [0.5]},   # dot — invalid
        {"label": "afraid", "coords": [1.0]},
    ]
    with pytest.raises(ManifoldFormatError, match="grammar-addressable"):
        ManifoldFolder.load(_author_manifold(tmp_path, nodes=nodes))


def test_dotted_node_label_rejected_create(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``create_manifold_folder`` rejects a dotted label up front."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = _author_nodes(["a", "b", "c"])
    nodes[1]["label"] = "a.b"
    with pytest.raises(ManifoldFormatError, match="grammar-addressable"):
        create_manifold_folder("local", "dotted", "", domain, nodes)


def test_dotted_node_label_rejected_discover_create(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``create_discover_manifold_folder`` rejects a dotted label too."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    with pytest.raises(ManifoldFormatError, match="grammar-addressable"):
        create_discover_manifold_folder(
            "local", "dotted_disc", "",
            fit_mode="pca",
            node_corpora=_discover_corpora(["alpha", "be.ta", "gamma"]),
        )


def test_underscore_and_hyphen_labels_still_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The stricter regex still admits ``_`` and ``-`` — only ``.`` is dropped."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    nodes = _author_nodes(["snake_case", "kebab-case", "plain"])
    folder, _ = create_manifold_folder("local", "ok_labels", "", domain, nodes)
    mf = ManifoldFolder.load(folder)
    assert mf.node_labels == ["snake_case", "kebab-case", "plain"]


# ============================================================ B3: lifecycle ===
#
# remove_manifold_folder / clear_manifold_tensors / refresh_manifold —
# the folder-addressed rm / clear / refresh lifecycle.


def _fake_fit_tensor(folder: Path, model_id: str, *, release: str | None = None) -> Path:
    """Drop a placeholder fitted tensor + sidecar for ``model_id``.

    No model needed — the lifecycle functions only glob/parse filenames.
    """
    from drowse.base.paths import tensor_filename
    ts = folder / tensor_filename(model_id, release=release)
    sc = folder / Path(tensor_filename(model_id, release=release)).with_suffix(".json").name
    ts.write_bytes(b"placeholder-tensor")
    sc.write_text(json.dumps(_sidecar_payload(name=folder.name)))
    # Trust only this successful pair; metadata rewrites never scan-and-bless
    # unrelated fitted files.
    ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(ts, sc)
    return ts


def test_clear_manifold_tensors_removes_tensors_keeps_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    capture_sha = "c" * 64
    sidecar_path = fitted.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["capture_sha256"] = capture_sha
    sidecar_path.write_text(json.dumps(sidecar))
    ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
        fitted, sidecar_path,
    )
    from drowse.base.paths import model_dir

    capture_dir = model_dir("google/gemma-3-4b-it") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.centroids.layer_0.safetensors"
    capture_file.write_bytes(b"cached")
    n = clear_manifold_tensors("local", "mood")
    assert n == 2  # tensor + sidecar
    assert not list(folder.glob("*.safetensors"))
    assert not capture_file.exists()
    # Corpus + manifest survive.
    assert (folder / "manifold.json").exists()
    assert (folder / "nodes").is_dir()
    mf = ManifoldFolder.load(folder)
    assert mf.node_labels == ["a", "b", "c"]
    # The integrity manifest no longer references the removed files.
    assert not any(k.endswith(".safetensors") for k in mf.files)


def test_shared_capture_survives_until_last_fitted_owner_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folders = [
        create_manifold_folder(
            "local", name, "", domain, _author_nodes(["a", "b", "c"]),
        )[0]
        for name in ("mood-a", "mood-b")
    ]
    capture_sha = "e" * 64
    for folder in folders:
        fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
        sidecar_path = fitted.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["capture_sha256"] = capture_sha
        sidecar_path.write_text(json.dumps(sidecar))
        ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(
            fitted, sidecar_path,
        )

    from drowse.base.paths import model_dir

    capture_dir = model_dir("google/gemma-3-4b-it") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.rows.layer_0.safetensors"
    capture_file.write_bytes(b"shared")

    clear_manifold_tensors("local", "mood-a")
    assert capture_file.exists()
    remove_manifold_folder("local", "mood-b")
    assert not capture_file.exists()


def test_clear_reaps_capture_when_sidecar_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "model/a")
    fitted.with_suffix(".json").unlink()
    from drowse.base.paths import model_dir

    capture_sha = "1" * 64
    capture_dir = model_dir("model/a") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.rows.layer_0.safetensors"
    capture_file.write_bytes(b"orphan")

    clear_manifold_tensors("local", "mood")
    assert not capture_file.exists()


def test_clear_reaps_capture_when_both_fitted_halves_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "model/a")
    fitted.unlink()
    fitted.with_suffix(".json").unlink()
    from drowse.base.paths import model_dir

    capture_sha = "4" * 64
    capture_dir = model_dir("model/a") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.rows.layer_0.safetensors"
    capture_file.write_bytes(b"orphan")

    assert clear_manifold_tensors("local", "mood") == 0
    assert not capture_file.exists()


def test_scoped_clear_reaps_orphan_capture_with_no_fitted_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fit killed after capture publication leaves only the capture.

    No fitted file and no manifest entry names the model, so the scoped
    model itself must seed the sweep (``clear -m`` / ``refresh -m``).
    """
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    create_manifold_folder("local", "mood", "", domain, _author_nodes(["a", "b", "c"]))
    from drowse.base.paths import model_dir

    capture_dir = model_dir("org/model") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    orphan = capture_dir / f"{'7' * 64}.rows"
    orphan.write_bytes(b"orphan")
    unrelated_dir = model_dir("org/other") / "manifold_capture"
    unrelated_dir.mkdir(parents=True)
    unrelated = unrelated_dir / f"{'8' * 64}.rows"
    unrelated.write_bytes(b"other model")

    assert clear_manifold_tensors("local", "mood", "org/model") == 0
    assert not orphan.exists()
    assert unrelated.exists()


def test_rm_reaps_capture_when_sidecar_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "model/a")
    fitted.with_suffix(".json").write_text("{")
    from drowse.base.paths import model_dir

    capture_sha = "2" * 64
    capture_dir = model_dir("model/a") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.rows.layer_0.safetensors"
    capture_file.write_bytes(b"orphan")

    remove_manifold_folder("local", "mood")
    assert not capture_file.exists()


def test_rm_reaps_capture_when_both_fitted_halves_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "model/a")
    fitted.unlink()
    fitted.with_suffix(".json").unlink()
    from drowse.base.paths import model_dir

    capture_sha = "5" * 64
    capture_dir = model_dir("model/a") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.rows.layer_0.safetensors"
    capture_file.write_bytes(b"orphan")

    remove_manifold_folder("local", "mood")
    assert not capture_file.exists()


def test_corrupt_owner_cleanup_preserves_capture_with_readable_shared_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folders = [
        create_manifold_folder(
            "local", name, "", domain, _author_nodes(["a", "b", "c"]),
        )[0]
        for name in ("broken-owner", "live-owner")
    ]
    broken = _fake_fit_tensor(folders[0], "model/a")
    broken.with_suffix(".json").write_text("{")
    live = _fake_fit_tensor(folders[1], "model/a")
    capture_sha = "3" * 64
    live_sidecar = live.with_suffix(".json")
    sidecar = json.loads(live_sidecar.read_text())
    sidecar["capture_sha256"] = capture_sha
    live_sidecar.write_text(json.dumps(sidecar))
    ManifoldFolder.load(folders[1], verify_manifest=False).update_file_hashes(
        live, live_sidecar,
    )
    from drowse.base.paths import model_dir

    capture_dir = model_dir("model/a") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.rows.layer_0.safetensors"
    capture_file.write_bytes(b"shared")

    remove_manifold_folder("local", "broken-owner")
    assert capture_file.exists()


def test_clear_manifold_tensors_repairs_orphan_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    fitted.unlink()

    assert clear_manifold_tensors("local", "mood") == 1
    assert not fitted.with_suffix(".json").exists()
    loaded = ManifoldFolder.load(folder)
    assert not any("gemma-3-4b-it" in name for name in loaded.files)


def test_clear_manifold_tensors_repairs_orphan_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    fitted.with_suffix(".json").unlink()

    assert clear_manifold_tensors("local", "mood") == 1
    assert not fitted.exists()
    ManifoldFolder.load(folder)


def test_clear_manifold_tensors_repairs_corrupt_fitted_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    fitted.write_bytes(b"same path, different fitted bytes")

    with pytest.raises(ManifoldFormatError, match="integrity"):
        ManifoldFolder.load(folder)
    assert clear_manifold_tensors("local", "mood") == 2
    ManifoldFolder.load(folder)


def test_scoped_clear_never_launders_unselected_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    corrupt = _fake_fit_tensor(folder, "model/a")
    _fake_fit_tensor(folder, "model/b")
    corrupt.write_bytes(b"tampered but not selected")

    assert clear_manifold_tensors("local", "mood", "model/b") == 2
    with pytest.raises(ManifoldFormatError, match="integrity"):
        ManifoldFolder.load(folder)


def test_clear_manifold_tensors_variant_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    _fake_fit_tensor(folder, "google/gemma-3-4b-it", release="gemma-scope")
    # variant="raw" only drops the canonical tensor, leaves the SAE one.
    n = clear_manifold_tensors("local", "mood", variant="raw")
    assert n == 2
    remaining = sorted(p.name for p in folder.glob("*.safetensors"))
    assert remaining == [tensor_filename("google/gemma-3-4b-it", release="gemma-scope")]


def test_clear_manifold_tensors_model_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A model scope deletes only that model's tensors, keeping others."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    _fake_fit_tensor(folder, "Qwen/Qwen3-4B")
    # Scope to gemma — only its tensor + sidecar go.
    n = clear_manifold_tensors("local", "mood", "google/gemma-3-4b-it")
    assert n == 2
    remaining = sorted(p.name for p in folder.glob("*.safetensors"))
    assert remaining == [tensor_filename("Qwen/Qwen3-4B")]
    # The integrity manifest still references the surviving model's tensor.
    mf = ManifoldFolder.load(folder)
    assert tensor_filename("Qwen/Qwen3-4B") in mf.files


def test_clear_manifold_tensors_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        clear_manifold_tensors("local", "nope")


def test_remove_manifold_folder_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    result = remove_manifold_folder("local", "mood")
    assert not folder.exists()
    assert result["removed"] is True
    assert result["source"] == "local"
    assert result["rematerializes_on_restart"] is False


def test_remove_manifold_folder_removes_referenced_capture_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    capture_sha = "d" * 64
    sidecar_path = fitted.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["capture_sha256"] = capture_sha
    sidecar_path.write_text(json.dumps(sidecar))
    ManifoldFolder.load(folder, verify_manifest=False).write_metadata()
    from drowse.base.paths import model_dir

    capture_dir = model_dir("google/gemma-3-4b-it") / "manifold_capture"
    capture_dir.mkdir(parents=True)
    capture_file = capture_dir / f"{capture_sha}.rows.layer_0.safetensors"
    capture_file.write_bytes(b"cached")
    remove_manifold_folder("local", "mood")
    assert not capture_file.exists()


def test_remove_manifold_folder_bundled_namespace_rematerializes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A folder under ``default/`` reports the bundled-respawn flag."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "default", "shipped", "", domain, _author_nodes(["a", "b", "c"]),
    )
    result = remove_manifold_folder("default", "shipped")
    assert not folder.exists()
    assert result["rematerializes_on_restart"] is True


def test_remove_manifold_folder_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        remove_manifold_folder("local", "nope")


def test_refresh_manifold_skips_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A local manifold has no upstream — refresh is a silent skip."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    assert refresh_manifold("local", "mood") == "skipped"


def test_metadata_only_lifecycle_does_not_hash_fitted_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    _fake_fit_tensor(folder, "test/model")
    from drowse.base import integrity
    hashed: list[Path] = []

    def unexpected_hash(path: Path) -> str:
        hashed.append(Path(path))
        raise AssertionError("metadata-only lifecycle hashed a fitted payload")

    monkeypatch.setattr(integrity, "hash_file", unexpected_hash)
    assert refresh_manifold("local", "mood") == "skipped"
    assert remove_manifold_folder("local", "mood")["removed"] is True
    assert hashed == []


def test_refresh_manifold_hf_repulls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An ``hf://``-sourced manifold re-pulls via pull_manifold."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "alice", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    # Stamp an hf:// source the way pull_manifold would.
    mf = ManifoldFolder.load(folder)
    mf.source = "hf://alice/mood@v1"
    mf.write_metadata()

    captured: dict[str, Any] = {}
    import drowse.store.hub as hfm

    def fake_pull(coord: str, *, target_folder: Path, force: bool, revision: str | None = None) -> Path:
        captured.update(coord=coord, target=target_folder,
                        force=force, revision=revision)
        return target_folder

    monkeypatch.setattr(hfm, "pull_manifold", fake_pull)
    assert refresh_manifold("alice", "mood") == "hf"
    assert captured["coord"] == "alice/mood"
    assert captured["revision"] == "v1"
    assert captured["target"] == folder


def test_refresh_manifold_model_scope_clears_fit_no_repull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A scoped refresh drops just the model's fit pair, never re-pulling.

    Scoped refresh drops tensors only: HF pulls are whole-repo,
    so a single-model refresh is a tensors-only delete (re-fits on next
    use), even on an ``hf://``-sourced manifold.
    """
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "alice", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    # Even on an hf:// source, a scoped refresh must NOT re-pull.
    mf = ManifoldFolder.load(folder)
    mf.source = "hf://alice/mood@v1"
    mf.write_metadata()
    _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    _fake_fit_tensor(folder, "Qwen/Qwen3-4B")

    import drowse.store.hub as hfm

    def boom(*a: object, **k: object) -> None:  # pragma: no cover - asserted not to fire
        raise AssertionError("scoped refresh must not re-pull from HF")

    monkeypatch.setattr(hfm, "pull_manifold", boom)

    assert refresh_manifold("alice", "mood", model_scope="google/gemma-3-4b-it") == "scoped"
    # Only gemma's fit pair is gone; Qwen's survives.
    remaining = sorted(p.name for p in folder.glob("*.safetensors"))
    assert remaining == [tensor_filename("Qwen/Qwen3-4B")]


def test_refresh_manifold_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        refresh_manifold("local", "nope")


def test_manifold_pair_lock_identity_is_external_stable_and_bounded(
    tmp_path: Path,
) -> None:
    from drowse.store.manifold_folder import manifold_pair_lock_path

    folder = tmp_path / "manifolds" / "local" / "mood"
    tensor = folder / ("model_" + "x" * 220 + ".safetensors")
    before = manifold_pair_lock_path(tensor)
    folder.mkdir(parents=True)
    folder.rmdir()
    after = manifold_pair_lock_path(tensor)

    assert before == after
    assert before.parent == folder.parent
    assert len(before.name) < 100


@pytest.mark.parametrize("operation", ["clear", "rm"])
def test_lifecycle_mutations_wait_for_stable_pair_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    from drowse.store.manifold_folder import manifold_pair_lock

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")

    started = threading.Event()
    done = threading.Event()
    errors: list[BaseException] = []

    def mutate() -> None:
        started.set()
        try:
            if operation == "clear":
                clear_manifold_tensors("local", "mood")
            else:
                remove_manifold_folder("local", "mood")
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            done.set()

    with manifold_pair_lock(fitted):
        worker = threading.Thread(target=mutate)
        worker.start()
        assert started.wait(1.0)
        assert not done.wait(0.1)
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert errors == []
    assert done.is_set()


def test_hf_refresh_holds_no_pair_lock_while_repulling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refresh must not hold pair locks across the re-pull.

    ``pull_manifold`` takes the manifest lock (and, forced, the pair locks in
    manifest-then-pairs order).  Refresh used to wrap it in the pair locks,
    inverting that global order against ``load_manifold`` (manifest, then
    pair): two processes could each hold one lock and wait forever.
    """
    from drowse.store.manifold_folder import manifold_pair_lock
    import drowse.store.hub as hfm

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "alice", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    mf = ManifoldFolder.load(folder)
    mf.source = "hf://alice/mood@v1"
    mf.write_metadata()
    pair_free: list[bool] = []

    def fake_pull(coord: str, *, target_folder: Path, force: bool, revision: Any = None) -> Path:
        acquired: list[bool] = []

        def probe() -> None:
            with manifold_pair_lock(fitted):
                acquired.append(True)

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join(timeout=1.0)
        pair_free.append(bool(acquired))
        return target_folder

    monkeypatch.setattr(hfm, "pull_manifold", fake_pull)
    assert refresh_manifold("alice", "mood") == "hf"
    assert pair_free == [True]


def test_force_authoring_reset_waits_for_stable_pair_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.store.manifold_folder import manifold_pair_lock

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "mood", "", fit_mode="pca",
        node_corpora={"old": ["old response"]},
    )
    fitted = _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    started = threading.Event()
    done = threading.Event()
    errors: list[BaseException] = []

    def reset() -> None:
        started.set()
        try:
            plan_discover_generation(
                folder, "mood", "", fit_mode="pca", labels=["new"],
                force=True,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            done.set()

    with manifold_pair_lock(fitted):
        worker = threading.Thread(target=reset)
        worker.start()
        assert started.wait(1.0)
        assert not done.wait(0.1)
        assert fitted.exists()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert errors == []
    assert done.is_set()
    assert not fitted.exists()
    manifest = json.loads((folder / "manifold.json").read_text())
    assert manifest["nodes"] == [{"label": "new", "role": None, "kind": None}]


# ============================================================ B6a: transfer ===
#
# transfer_manifold applies a GIVEN per-layer Procrustes map to a fitted
# manifold's subspace, writing a _from-<safe_src> variant tensor.


def test_rectangular_affine_transfer_preserves_points_frame_and_steering(
    tmp_path: Path,
) -> None:
    import torch
    from drowse.geometry.manifold import (
        CustomDomain, LayerSubspace, Manifold,
        transfer_manifold_subspaces,
    )
    from drowse.store.manifold_tensors import load_manifold, save_manifold
    from drowse.store.alignment import LayerAlignment

    basis = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    node_coords = torch.tensor([[0.0, 0.0], [1.0, -0.5], [-0.25, 1.5]])
    source = Manifold(
        name="rect-flat", domain=CustomDomain(2), node_labels=["a", "b", "c"],
        node_coords=node_coords,
        layers={0: LayerSubspace.affine(
            torch.tensor([0.5, -1.0, 2.0, 0.25]), basis,
            node_coords=node_coords,
        )}, mahalanobis_share={0: 1.0},
    )
    dense = torch.tensor([
        [2.0, 0.2, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0],
        [0.3, 0.0, 1.0, 0.0], [0.0, -0.4, 0.0, 1.0],
        [0.1, 0.2, 0.0, 0.0], [0.0, 0.3, 0.0, 0.0],
    ])
    alignment = LayerAlignment(
        dense, torch.eye(4), torch.linspace(-0.5, 0.5, 6),
    )
    target = transfer_manifold_subspaces(
        source, {0: alignment}, whitener=_target_whitener(dim=6, layers=(0,)),
        target_layer_means={0: torch.zeros(6)},
        from_model="src", to_model="tgt",
    )
    for point in (torch.tensor([0.2, -0.4]), node_coords[1]):
        assert torch.allclose(
            target.manifold_point(0, point),
            alignment.apply_points(source.manifold_point(0, point)),
            atol=1e-5,
        )
    target_sub = target.layers[0]
    assert torch.allclose(
        target_sub.basis @ target_sub.basis.T, torch.eye(2), atol=1e-5,
    )
    assert target_sub.node_coords is not None
    assert torch.allclose(
        target_sub.node_coords[1] @ target_sub.basis,
        alignment.apply_vector(node_coords[1] @ basis),
        atol=1e-5,
    )
    # The explicit affine coordinate reparameterization is an on-disk contract,
    # not merely an in-memory transfer detail.
    path = tmp_path / "rect-flat.safetensors"
    save_manifold(target, path, {
        "method": "manifold_procrustes_transfer",
        "source_model_id": "src", "source_model_fingerprint": "src-fp",
    })
    loaded = load_manifold(path, verify_manifest=False)
    assert torch.allclose(
        loaded.manifold_point(0, [0.2, -0.4]),
        target.manifold_point(0, [0.2, -0.4]),
        atol=1e-6,
    )


def test_monopolar_affine_transfer_rebakes_positive_world_direction_share() -> None:
    from drowse.geometry.manifold import (
        CustomDomain, LayerSubspace, Manifold, transfer_manifold_subspaces,
    )
    from drowse.store.alignment import LayerAlignment

    basis = torch.tensor([[1.0, 0.0, 0.0]])
    real_coord = torch.tensor([[2.5]])
    source = Manifold(
        name="ray", domain=CustomDomain(1), node_labels=["positive"],
        node_coords=torch.tensor([[1.0]]),
        layers={0: LayerSubspace.affine(
            torch.zeros(3), basis, node_coords=real_coord,
        )},
        mahalanobis_share={0: 1.0},
    )
    whitener = _target_whitener(dim=3, layers=(0,))
    transferred = transfer_manifold_subspaces(
        source,
        {0: LayerAlignment(torch.eye(3), torch.eye(3), torch.zeros(3))},
        whitener=whitener, target_layer_means=whitener.layer_means,
        from_model="src", to_model="tgt",
    )
    transferred_coords = transferred.layers[0].node_coords
    assert transferred_coords is not None
    expected_world = transferred_coords[0] @ transferred.layers[0].basis
    assert transferred.mahalanobis_share[0] == pytest.approx(
        whitener.mahalanobis_norm(0, expected_world), rel=1e-5,
    )
    assert transferred.mahalanobis_share[0] > 0.0


def test_rectangular_curved_transfer_rejects_anisotropic_tube() -> None:
    import torch
    from dataclasses import replace
    from drowse.geometry.manifold import (
        BoxAxis, BoxDomain, Manifold, fit_layer_subspace,
        fit_rbf_interpolant, transfer_manifold_subspaces,
    )
    from drowse.store.alignment import LayerAlignment

    domain = BoxDomain([BoxAxis("t", periodic=False, lo=0.0, hi=1.0)])
    coords = torch.tensor([[0.0], [0.5], [1.0]])
    embedded = domain.embed(coords)
    centroids = torch.tensor([
        [0.0, 0.0, 1.0, -1.0],
        [0.8, 0.4, 0.3, -0.2],
        [0.1, 1.2, -0.5, 0.5],
    ])
    sub, _ = fit_layer_subspace(centroids, embedded)
    assert sub.node_params is not None
    sigma_rw, sigma_pc = fit_rbf_interpolant(
        sub.node_params, torch.zeros(3, 1),
    )
    sub = replace(sub, sigma_rbf_weights=sigma_rw, sigma_poly_coeffs=sigma_pc)
    source = Manifold(
        name="rect-curve", domain=domain, node_labels=["a", "b", "c"],
        node_coords=coords, layers={0: sub},
        node_roles=[None, None, None], node_kinds=[None, None, None],
        mahalanobis_share={0: 1.0},
        origin={0: torch.zeros(domain.intrinsic_dim)},
    )
    dense = torch.tensor([
        [1.8, 0.0, 0.0, 0.0], [0.2, 0.7, 0.0, 0.0],
        [0.0, 0.1, 1.0, 0.0], [0.0, 0.0, 0.2, 1.0],
        [0.3, 0.0, 0.0, 0.0], [0.0, -0.2, 0.0, 0.0],
    ])
    alignment = LayerAlignment(dense, torch.eye(4), torch.arange(6).float() / 10)
    with pytest.raises(ValueError, match="anisotropic tube transfer"):
        transfer_manifold_subspaces(
            source, {0: alignment},
            whitener=_target_whitener(dim=6, layers=(0,)),
            target_layer_means={0: torch.zeros(6)},
            from_model="src", to_model="tgt",
        )


def test_rectangular_transfer_rejects_collapsed_subspace_rank() -> None:
    import torch
    from drowse.geometry.manifold import CustomDomain, LayerSubspace, Manifold, transfer_manifold_subspaces
    from drowse.store.alignment import LayerAlignment

    source = Manifold(
        name="collapsed", domain=CustomDomain(2), node_labels=["a", "b"],
        node_coords=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        layers={0: LayerSubspace.affine(
            torch.zeros(3), torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            node_coords=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        )}, mahalanobis_share={0: 1.0},
    )
    collapsed = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    alignment = LayerAlignment(collapsed, torch.eye(3), torch.zeros(2))
    with pytest.raises(ValueError, match="rank-deficient"):
        transfer_manifold_subspaces(
            source, {0: alignment},
            whitener=_target_whitener(dim=2, layers=(0,)),
            target_layer_means={0: torch.zeros(2)},
            from_model="src", to_model="tgt",
        )


def test_rectangular_transfer_rejects_oblique_rank_collapse() -> None:
    """Rank detection uses singular values, not unpivoted QR diagonals."""
    import torch
    from drowse.geometry.manifold import (
        CustomDomain, LayerSubspace, Manifold, transfer_manifold_subspaces,
    )
    from drowse.store.alignment import LayerAlignment

    source = Manifold(
        name="oblique-collapse", domain=CustomDomain(3),
        node_labels=["a", "b", "c"], node_coords=torch.eye(3),
        layers={0: LayerSubspace.affine(
            torch.zeros(3), torch.eye(3), node_coords=torch.eye(3),
        )}, mahalanobis_share={0: 1.0},
    )
    generator = torch.Generator().manual_seed(30061)
    collapsed = (
        torch.randn(3, 2, generator=generator)
        @ torch.randn(2, 3, generator=generator)
    )
    alignment = LayerAlignment(collapsed, torch.eye(3), torch.zeros(3))

    with pytest.raises(ValueError, match="rank-deficient"):
        transfer_manifold_subspaces(
            source, {0: alignment},
            whitener=_target_whitener(dim=3, layers=(0,)),
            target_layer_means={0: torch.zeros(3)},
            from_model="src", to_model="tgt",
        )


def _target_whitener(*, dim: int = 6, layers: tuple[int, ...] = (4, 5, 6)):
    """A target-model whitener over the transferred layers.

    Transfer is Mahalanobis-only now (the share re-bake is mandatory; there is
    no Euclidean rebake), so every ``transfer_manifold`` call needs a target
    whitener covering the transferred layers.
    """
    import torch
    from drowse.geometry.whitener import LayerWhitener
    g = torch.Generator().manual_seed(99)
    acts = {L: torch.randn(120, dim, generator=g) for L in layers}
    means = {L: torch.zeros(dim) for L in layers}
    return LayerWhitener.from_neutral_activations(acts, means)


def _fit_real_manifold(folder: Path, model_id: str, *, dim: int = 6, seed: int = 0):
    """Fit a tiny real authored manifold on synthetic centroids and save it.

    Returns the path of the written canonical tensor.
    """
    import torch
    from dataclasses import replace
    from drowse.geometry.manifold import (
        MANIFOLD_FIT_POLICY_VERSION, BoxAxis, BoxDomain, Manifold,
        fit_layer_subspace, fit_rbf_interpolant,
    )
    from drowse.store.manifold_tensors import save_manifold

    g = torch.Generator().manual_seed(seed)
    domain = BoxDomain([BoxAxis("t", periodic=False, lo=0.0, hi=1.0)])
    coords = torch.tensor([[0.0], [0.5], [1.0]])
    embedded = domain.embed(coords)
    layers = {}
    origins = {}
    for layer in (4, 5, 6):
        centroids = torch.randn(3, dim, generator=g)
        sub, _ev_ratio = fit_layer_subspace(centroids, embedded)
        assert sub.node_params is not None
        sigma_rw, sigma_pc = fit_rbf_interpolant(
            sub.node_params, torch.zeros(3, 1),
        )
        sub = replace(
            sub,
            sigma_rbf_weights=sigma_rw,
            sigma_poly_coeffs=sigma_pc,
        )
        layers[layer] = sub
        origins[layer] = torch.zeros(domain.intrinsic_dim)
    man = Manifold(
        name=folder.name,
        domain=domain,
        node_labels=["a", "b", "c"],
        node_coords=coords,
        layers=layers,
        # A per-model whitened share, so the transfer test can assert it
        # is re-baked in target space (Σ is per-model; the source metric is
        # invalid in target space).
        mahalanobis_share={4: 1.0, 5: 2.0, 6: 3.0},
        origin=origins,
    )
    from drowse.base.paths import tensor_filename
    out = folder / tensor_filename(model_id)
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    save_manifold(man, out, {
        "method": "manifold_pca", "nodes_sha256": mf.nodes_sha256(),
        "model_fingerprint": f"fp:{model_id}",
        "fit_policy_version": MANIFOLD_FIT_POLICY_VERSION,
    })
    mf.update_file_hashes(out, out.with_suffix(".json"))
    return out


def test_transfer_preflight_returns_locked_source_layer_header(
    tmp_path: Path,
) -> None:
    folder = _author_manifold(tmp_path)
    _fit_real_manifold(folder, "src/model", dim=4)

    proof = preflight_transfer_manifold(
        folder, from_model="src/model", to_model="tgt/model",
    )
    assert proof.layers == (4, 5, 6)


def test_transfer_rejects_source_generation_changed_after_preflight(
    tmp_path: Path,
) -> None:
    folder = _author_manifold(tmp_path)
    _fit_real_manifold(folder, "src/model", dim=4, seed=0)
    proof = preflight_transfer_manifold(
        folder, from_model="src/model", to_model="tgt/model",
    )
    _fit_real_manifold(folder, "src/model", dim=4, seed=1)

    with pytest.raises(ManifoldFormatError, match="changed while alignment"):
        transfer_manifold(
            folder,
            from_model="src/model",
            to_model="tgt/model",
            alignment=_alignments({layer: torch.eye(4) for layer in proof.layers}),
            source_model_fingerprint="fp:src/model",
            target_model_fingerprint="fp:tgt/model",
            whitener=_target_whitener(dim=4),
            target_layer_means=_target_whitener(dim=4).layer_means,
            expected_source_proof=proof,
        )

    from drowse.base.paths import tensor_filename

    assert not (
        folder / tensor_filename("tgt/model", transferred_from="src/model")
    ).exists()


def test_transfer_preflight_wraps_corrupt_manifest(tmp_path: Path) -> None:
    folder = _author_manifold(tmp_path)
    _fit_real_manifold(folder, "src/model", dim=4)
    (folder / "manifold.json").write_text("{")

    with pytest.raises(ManifoldFormatError, match="manifest is unreadable"):
        preflight_transfer_manifold(
            folder, from_model="src/model", to_model="tgt/model",
        )


@pytest.mark.parametrize("root", ["[]", "null", '"manifest"'])
def test_transfer_preflight_rejects_non_object_manifest_before_payload_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root: str,
) -> None:
    folder = _author_manifold(tmp_path)
    _fit_real_manifold(folder, "src/model", dim=4)
    (folder / "manifold.json").write_text(root)
    monkeypatch.setattr(
        "drowse.base.integrity.verify_integrity",
        lambda *_args, **_kwargs: pytest.fail(
            "non-object manifest reached payload hashing"
        ),
    )

    with pytest.raises(ManifoldFormatError, match="JSON object"):
        preflight_transfer_manifold(
            folder, from_model="src/model", to_model="tgt/model",
        )


@pytest.mark.parametrize("digest", [None, 7, "abc", "g" * 64])
def test_transfer_preflight_rejects_invalid_selected_digest_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, digest: object,
) -> None:
    folder = _author_manifold(tmp_path)
    source = _fit_real_manifold(folder, "src/model", dim=4)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][source.name] = digest
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "drowse.base.integrity.verify_integrity",
        lambda *_args, **_kwargs: pytest.fail(
            "malformed selected digest reached payload hashing"
        ),
    )

    with pytest.raises(ManifoldFormatError, match="invalid sha256"):
        preflight_transfer_manifold(
            folder, from_model="src/model", to_model="tgt/model",
        )


def test_transfer_preflight_rejects_future_manifest_before_payload_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = _author_manifold(tmp_path)
    _fit_real_manifold(folder, "src/model", dim=4)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = MANIFOLD_FORMAT_VERSION + 1
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(
        "drowse.base.integrity.verify_integrity",
        lambda *_args, **_kwargs: pytest.fail(
            "future manifest reached payload hashing"
        ),
    )
    with pytest.raises(ManifoldFormatError, match="need exactly"):
        preflight_transfer_manifold(
            folder, from_model="src/model", to_model="tgt/model",
        )


def test_strict_fitted_load_rejects_non_object_manifest(tmp_path: Path) -> None:
    from drowse.store.manifold_tensors import load_manifold

    folder = _author_manifold(tmp_path)
    source = _fit_real_manifold(folder, "src/model", dim=4)
    (folder / "manifold.json").write_text("[]")

    with pytest.raises(ManifoldFormatError, match="JSON object"):
        load_manifold(source)


@pytest.mark.parametrize(
    ("payload", "message"), [("{", "unreadable"), ("[]", "JSON object")],
)
def test_strict_fitted_load_normalizes_trusted_corrupt_sidecar(
    tmp_path: Path, payload: str, message: str,
) -> None:
    from drowse.store.manifold_tensors import load_manifold
    from drowse.base.integrity import hash_file

    folder = _author_manifold(tmp_path)
    source = _fit_real_manifold(folder, "src/model", dim=4)
    sidecar = source.with_suffix(".json")
    sidecar.write_text(payload)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][sidecar.name] = hash_file(sidecar)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ManifoldFormatError, match=message):
        load_manifold(source)


def test_strict_fitted_load_rejects_invalid_digest_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.store.manifold_tensors import load_manifold

    folder = _author_manifold(tmp_path)
    source = _fit_real_manifold(folder, "src/model", dim=4)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][source.name] = "invalid"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "drowse.base.integrity.verify_integrity",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid digest reached fitted payload hashing"
        ),
    )

    with pytest.raises(ManifoldFormatError, match="invalid sha256"):
        load_manifold(source)


def test_transfer_manifold_identity_alignment_preserves_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """An identity alignment leaves the per-layer subspace unchanged and
    writes the transferred tensor at the ``_from-<safe_src>`` filename."""
    import torch
    from drowse.store.manifold_tensors import load_manifold

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    src_model = "google/gemma-3-4b-it"
    tgt_model = "Qwen/Qwen2.5-7B-Instruct"
    src_tensor = _fit_real_manifold(folder, src_model, dim=6)
    src_man = load_manifold(src_tensor)

    # Identity map per fitted layer.
    align = _alignments({L: torch.eye(6) for L in src_man.layers})

    target_whitener = _target_whitener()
    out = transfer_manifold(
        folder, from_model=src_model, to_model=tgt_model,
        alignment=align, transfer_quality_estimate=0.9,
        source_model_fingerprint=f"fp:{src_model}",
        target_model_fingerprint=f"fp:{tgt_model}",
        whitener=target_whitener,
        target_layer_means=target_whitener.layer_means,
    )
    # Filename uses the transfer variant suffix.
    from drowse.base.paths import tensor_filename
    assert out.name == tensor_filename(tgt_model, transferred_from=src_model)

    tgt_man = load_manifold(out)
    assert sorted(tgt_man.layers) == sorted(src_man.layers)
    for L in src_man.layers:
        assert torch.allclose(tgt_man.layers[L].mean, src_man.layers[L].mean, atol=1e-5)
        assert torch.allclose(tgt_man.layers[L].basis, src_man.layers[L].basis, atol=1e-5)
    # The source's per-model Mahalanobis share is invalid in target space, so
    # it is RE-BAKED in the target metric (mandatory now — never dropped).
    assert src_man.mahalanobis_share  # source had one
    assert set(tgt_man.mahalanobis_share.keys()) == set(src_man.layers)
    assert tgt_man.mahalanobis_share != src_man.mahalanobis_share
    # Provenance lands in the sidecar.
    with open(out.with_suffix(".json")) as f:
        sc = json.load(f)
    assert sc["method"] == "manifold_procrustes_transfer"
    assert sc["source_model_id"] == src_model
    assert sc["transfer_quality_estimate"] == pytest.approx(0.9)
    # Folder integrity manifest now covers the transferred tensor.
    mf = ManifoldFolder.load(folder)
    assert out.name in mf.files


def test_transfer_manifold_rebakes_share_in_target_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A target whitener covering the transferred layers re-bakes the
    Mahalanobis share in target space (instead of dropping it) and records
    ``share_metric == "mahalanobis"``."""
    import torch
    from drowse.geometry.whitener import LayerWhitener
    from drowse.store.manifold_tensors import load_manifold

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    src_model = "google/gemma-3-4b-it"
    tgt_model = "Qwen/Qwen2.5-7B-Instruct"
    src_tensor = _fit_real_manifold(folder, src_model, dim=6)
    src_man = load_manifold(src_tensor)
    align = _alignments({L: torch.eye(6) for L in src_man.layers})

    # Target whitener over the fitted layers {4, 5, 6}.
    g = torch.Generator().manual_seed(99)
    acts = {L: torch.randn(120, 6, generator=g) for L in src_man.layers}
    means = {L: torch.zeros(6) for L in src_man.layers}
    w = LayerWhitener.from_neutral_activations(acts, means)

    out = transfer_manifold(
        folder, from_model=src_model, to_model=tgt_model,
        alignment=align, whitener=w, target_layer_means=w.layer_means,
        source_model_fingerprint=f"fp:{src_model}",
        target_model_fingerprint=f"fp:{tgt_model}",
    )
    tgt_man = load_manifold(out)

    # Share recomputed (not dropped), one per transferred layer.
    assert set(tgt_man.mahalanobis_share.keys()) == set(src_man.layers)
    for L in src_man.layers:
        assert tgt_man.mahalanobis_share[L] > 0.0
    # They are *target*-metric values, not the source's hand-set share.
    assert tgt_man.mahalanobis_share != src_man.mahalanobis_share
    with open(out.with_suffix(".json")) as f:
        sc = json.load(f)
    assert sc["share_metric"] == "mahalanobis"


def test_transfer_manifold_rotation_maps_subspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A known rotation Q maps mean → Q@mean and basis → basis@Q^T, so the
    transferred world-space activation at a node equals Q applied to the
    source activation."""
    import torch
    from drowse.store.manifold_tensors import load_manifold

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    src_model = "src/model"
    tgt_model = "tgt/model"
    src_tensor = _fit_real_manifold(folder, src_model, dim=6, seed=3)
    src_man = load_manifold(src_tensor)

    # Random orthogonal Q via QR.
    A = torch.randn(6, 6, generator=torch.Generator().manual_seed(7))
    Q, _ = torch.linalg.qr(A)
    align = _alignments({L: Q for L in src_man.layers})

    target_whitener = _target_whitener()
    out = transfer_manifold(
        folder, from_model=src_model, to_model=tgt_model, alignment=align,
        source_model_fingerprint=f"fp:{src_model}",
        target_model_fingerprint=f"fp:{tgt_model}",
        whitener=target_whitener,
        target_layer_means=target_whitener.layer_means,
    )
    tgt_man = load_manifold(out)
    # World-space activation at node 0 should be Q @ (source activation).
    for L in src_man.layers:
        src_pt = src_man.manifold_point(L, [0.0]).to(torch.float32)
        tgt_pt = tgt_man.manifold_point(L, [0.0]).to(torch.float32)
        assert torch.allclose(tgt_pt, Q @ src_pt, atol=1e-4)


def test_transfer_manifold_drops_uncovered_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Layers the alignment doesn't cover are dropped from the transfer."""
    import torch
    from drowse.store.manifold_tensors import load_manifold

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    src_model, tgt_model = "src/m", "tgt/m"
    _fit_real_manifold(folder, src_model, dim=6)  # writes the on-disk source fit
    # Cover only layer 5 of {4, 5, 6}.
    align = _alignments({5: torch.eye(6)})
    target_whitener = _target_whitener(layers=(5,))
    out = transfer_manifold(
        folder, from_model=src_model, to_model=tgt_model, alignment=align,
        source_model_fingerprint=f"fp:{src_model}",
        target_model_fingerprint=f"fp:{tgt_model}",
        whitener=target_whitener,
        target_layer_means=target_whitener.layer_means,
    )
    tgt_man = load_manifold(out)
    assert sorted(tgt_man.layers) == [5]


def test_transfer_manifold_missing_source_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import torch
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    with pytest.raises(FileNotFoundError):
        transfer_manifold(
            folder, from_model="never/fitted", to_model="tgt/m",
            alignment=_alignments({0: torch.eye(4)}),
            source_model_fingerprint="fp:never/fitted",
            target_model_fingerprint="fp:tgt/m",
            target_layer_means={},
        )


def test_transfer_manifold_empty_alignment_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    _fit_real_manifold(folder, "src/m", dim=6)
    with pytest.raises(ManifoldFormatError, match="empty"):
        transfer_manifold(
            folder, from_model="src/m", to_model="tgt/m", alignment={},
            source_model_fingerprint="fp:src/m",
            target_model_fingerprint="fp:tgt/m",
            target_layer_means={},
        )


def test_transfer_manifold_no_overlap_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """An alignment covering no fitted layer raises rather than write empty."""
    import torch
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    _fit_real_manifold(folder, "src/m", dim=6)  # layers 4,5,6
    with pytest.raises(ManifoldFormatError, match="covered none"):
        transfer_manifold(
            folder, from_model="src/m", to_model="tgt/m",
            alignment=_alignments({0: torch.eye(6), 1: torch.eye(6)}),
            source_model_fingerprint="fp:src/m",
            target_model_fingerprint="fp:tgt/m",
            target_layer_means={},
        )


def test_transfer_manifold_refuses_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    import torch
    from drowse.store.manifold_tensors import load_manifold

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    src_tensor = _fit_real_manifold(folder, "src/m", dim=6)
    align = _alignments({L: torch.eye(6) for L in load_manifold(src_tensor).layers})
    w = _target_whitener()
    transfer_manifold(
        folder, from_model="src/m", to_model="tgt/m", alignment=align, whitener=w,
        source_model_fingerprint="fp:src/m", target_layer_means=w.layer_means,
        target_model_fingerprint="fp:tgt/m",
    )
    import drowse.geometry.manifold as manifold_mod

    real_transfer = manifold_mod.transfer_manifold_subspaces
    monkeypatch.setattr(
        manifold_mod, "transfer_manifold_subspaces",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("trusted target performed transfer compute")
        ),
    )
    with pytest.raises(FileExistsError):
        transfer_manifold(
            folder, from_model="src/m", to_model="tgt/m", alignment=align, whitener=w,
            source_model_fingerprint="fp:src/m", target_layer_means=w.layer_means,
            target_model_fingerprint="fp:tgt/m",
        )
    # force=True overwrites cleanly.
    monkeypatch.setattr(manifold_mod, "transfer_manifold_subspaces", real_transfer)
    transfer_manifold(
        folder, from_model="src/m", to_model="tgt/m", alignment=align,
        source_model_fingerprint="fp:src/m",
        target_model_fingerprint="fp:tgt/m",
        whitener=w, target_layer_means=w.layer_means, force=True,
    )


def test_transfer_retries_pair_committed_before_manifest_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unproven transferred pair is repaired without requiring force."""
    import torch
    from drowse.store.manifold_tensors import load_manifold
    from drowse.base.paths import tensor_filename

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    src_tensor = _fit_real_manifold(folder, "src/m", dim=6)
    align = _alignments({
        layer: torch.eye(6) for layer in load_manifold(src_tensor).layers
    })
    whitener = _target_whitener()
    original = ManifoldFolder.update_file_hashes
    failed = False

    def fail_transfer_manifest_once(
        self: ManifoldFolder, *paths: Path,
    ) -> None:
        nonlocal failed
        if not failed and any("_from-" in path.name for path in paths):
            failed = True
            raise RuntimeError("injected post-pair manifest failure")
        original(self, *paths)

    monkeypatch.setattr(
        ManifoldFolder, "update_file_hashes", fail_transfer_manifest_once,
    )
    kwargs = dict(
        folder=folder, from_model="src/m", to_model="tgt/m",
        alignment=align, whitener=whitener,
        target_layer_means=whitener.layer_means,
        source_model_fingerprint="fp:src/m",
        target_model_fingerprint="fp:tgt/m",
    )
    with pytest.raises(RuntimeError, match="post-pair manifest"):
        transfer_manifold(**kwargs)

    target = folder / tensor_filename("tgt/m", transferred_from="src/m")
    assert target.exists()
    assert target.with_suffix(".json").exists()
    assert target.name not in json.loads(
        (folder / "manifold.json").read_text(),
    )["files"]

    retried = transfer_manifold(**kwargs)
    assert retried == target
    loaded = load_manifold(retried)
    assert loaded.metadata["source_model_id"] == "src/m"
    assert loaded.metadata["source_model_fingerprint"] == "fp:src/m"
    ManifoldFolder.load(folder)


# ============================================================ B6b: summary ===


def test_manifold_summary_authored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "a mood axis", domain, _author_nodes(["a", "b", "c"]),
    )
    _fake_fit_tensor(folder, "google/gemma-3-4b-it")
    summ = manifold_summary(folder)
    assert summ["namespace"] == "local"
    assert summ["name"] == "mood"
    assert summ["description"] == "a mood axis"
    assert summ["source"] == "local"
    assert summ["fit_mode"] == "authored"
    assert summ["is_discover"] is False
    assert summ["domain"]["type"] == "box"
    assert summ["domain_label"] == "box(1d)"
    assert summ["intrinsic_dim"] == 1
    assert summ["min_nodes"] == 3
    assert summ["node_count"] == 3
    assert summ["node_labels"] == ["a", "b", "c"]
    assert len(summ["node_coords"]) == 3
    assert summ["node_roles"] == [None, None, None]
    assert summ["hyperparams"] == {}
    gemma_safe = safe_model_id("google/gemma-3-4b-it")
    assert summ["fitted_models"] == [gemma_safe]
    assert summ["tensor_variants"][gemma_safe] == ["raw"]


def test_manifold_summary_discover_unfitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "personas", "five personas",
        fit_mode="pca",
        node_corpora=_discover_corpora(
            ["pirate", "caveman", "assistant", "scholar", "robot"],
        ),
        hyperparams={"max_dim": 8, "var_threshold": 0.70},
    )
    summ = manifold_summary(folder)
    assert summ["fit_mode"] == "pca"
    assert summ["is_discover"] is True
    # Discover folder carries no top-level geometry on disk.
    assert summ["domain"] == {}
    assert summ["domain_label"] == "discover-pca"
    assert summ["intrinsic_dim"] == 0
    assert summ["min_nodes"] is None
    assert summ["node_coords"] == []
    assert summ["hyperparams"] == {"max_dim": 8, "var_threshold": 0.70}
    assert summ["fitted_models"] == []


def test_manifold_summary_reports_transfer_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A transferred tensor surfaces under tensor_variants as ``from-...``."""
    import torch
    from drowse.store.manifold_tensors import load_manifold

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    src_tensor = _fit_real_manifold(folder, "src/m", dim=6)
    align = _alignments({L: torch.eye(6) for L in load_manifold(src_tensor).layers})
    target_whitener = _target_whitener()
    transfer_manifold(
        folder, from_model="src/m", to_model="tgt/m", alignment=align,
        source_model_fingerprint="fp:src/m",
        target_model_fingerprint="fp:tgt/m",
        whitener=target_whitener,
        target_layer_means=target_whitener.layer_means,
    )
    summ = manifold_summary(folder)
    src_safe = safe_model_id("src/m")
    from drowse.base.paths import encode_release_id
    tgt_safe = safe_model_id("tgt/m")
    assert summ["tensor_variants"][src_safe] == ["raw"]
    assert summ["tensor_variants"][tgt_safe] == [
        f"from-{encode_release_id('src/m')}"
    ]
    # The transfer provenance is on the per-tensor block, opt-in. It keys on
    # the full tensor *stem*, so the variant gets its own entry.
    assert "fitted" not in summ
    detail = manifold_summary(folder, include_fits=True)
    stems = {f["stem"]: f for f in detail["fitted"]}
    variant_stem = f"{tgt_safe}_from-{encode_release_id('src/m')}"
    assert set(stems) == {src_safe, variant_stem}
    assert stems[variant_stem]["method"] == "manifold_procrustes_transfer"
    assert stems[variant_stem]["share_metric"] == "mahalanobis"


def test_manifold_summary_reports_template_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A templated manifold must be distinguishable from a hand-authored one.

    Its corpus is a derived materialization whose authoring source of truth
    lives in the template — exactly what a user needs to know before editing
    ``nodes/`` by hand.
    """
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.store.manifold_authoring import create_manifold_from_template
    from drowse.store.templates import create_template_folder

    create_template_folder(
        "local", "weekday", slot="[DAY]",
        values=["Monday", "Tuesday", "Wednesday"],
        contexts=[{
            "turns": [{"role": "user", "content": "what day is it?"}],
            "assistant": "today is [DAY]",
        }],
    )
    plain = create_discover_manifold_folder(
        "local", "handmade", "d", fit_mode="pca",
        node_corpora={"a": ["x"], "b": ["y"], "c": ["z"]},
    )
    assert manifold_summary(plain)["template_ref"] is None

    derived = create_manifold_from_template(
        "local", "fromtmpl", "d", template_ref="weekday", fit_mode="pca",
    )
    assert manifold_summary(derived)["template_ref"] == "local/weekday"


def test_manifold_fit_summary_surfaces_auto_topology_evidence(tmp_path: Path):
    """An ``auto`` fit's flat-vs-curved evidence must be readable, not just stored.

    ``resolved_fit_mode`` / ``topology_winner`` / ``topology_candidates`` are
    the whole record of why ``select_topology`` chose the geometry it did;
    they were written and schema-validated but no reader lifted them.
    """
    import torch
    from drowse.geometry.manifold import (
        BoxAxis, BoxDomain, Manifold, fit_layer_subspace,
    )
    from drowse.store.manifold_schema import ManifoldSidecar
    from drowse.store.manifold_tensors import save_manifold
    from drowse.store.manifold_lifecycle import manifold_fit_summary

    g = torch.Generator().manual_seed(0)
    domain = BoxDomain([BoxAxis("t", periodic=False, lo=0.0, hi=1.0)])
    coords = torch.tensor([[0.0], [0.5], [1.0]])
    sub, _ = fit_layer_subspace(torch.randn(3, 6, generator=g), domain.embed(coords))
    man = Manifold(
        name="autotopo2", domain=domain, node_labels=["a", "b", "c"],
        node_coords=coords, layers={4: sub}, mahalanobis_share={4: 1.0},
        origin={4: torch.zeros(domain.intrinsic_dim)}, feature_space="sae-test",
    )
    candidates = [
        {"name": "flat-pca", "fit_mode": "pca", "intrinsic_dim": 2,
         "score": 12.5, "viable": True, "reason": None},
    ]
    out = tmp_path / "autotopo2.safetensors"
    save_manifold(man, out, {
        "method": "manifold_discover_auto", "fit_mode": "auto",
        "resolved_fit_mode": "pca", "topology_winner": "flat-pca",
        "topology_candidates": candidates,
        "share_metric": "mahalanobis", "subspace_metric": "mahalanobis",
        "sae_release": "test", "sae_revision": "rev-1",
        "sae_fingerprint": "fingerprint", "sae_ids_by_layer": {"4": "mock-4"},
    })

    sc = ManifoldSidecar.load(out.with_suffix(".json"))
    assert sc.resolved_fit_mode == "pca"
    assert sc.topology_winner == "flat-pca"
    assert [c["name"] for c in sc.topology_candidates] == ["flat-pca"]
    assert sc.share_metric == "mahalanobis"
    assert sc.subspace_metric == "mahalanobis"

    entry = manifold_fit_summary(sc, "autotopo2")
    assert entry["resolved_fit_mode"] == "pca"
    assert entry["topology_winner"] == "flat-pca"
    assert [c["name"] for c in entry["topology_candidates"]] == ["flat-pca"]
    assert entry["share_metric"] == "mahalanobis"
    assert entry["subspace_metric"] == "mahalanobis"
    # Empty diagnostics stay omitted so a pinned/authored fit reads terse.
    assert "rbf_smoothing" not in entry
    assert "sigma_field" not in entry


# --------------------------------------------- streaming discover writers ---

def test_init_and_append_discover_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The streaming skeleton + per-node append round-trips through load."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    labels = ["alpha", "beta", "gamma", "default"]
    folder = init_discover_manifold_folder(
        "local", "stream", "streamed discover", fit_mode="pca",
        labels=labels, hyperparams={"max_dim": 4},
    )
    # Skeleton present, no node corpus yet.
    assert (folder / "manifold.json").exists()
    assert (folder / "nodes").is_dir()
    assert list((folder / "nodes").glob("*.json")) == []

    for i, label in enumerate(labels):
        append_discover_manifold_node(
            folder, i, label, [f"{label} one", f"{label} two"],
        )

    mf = ManifoldFolder.load(folder)
    assert mf.is_discover
    assert mf.node_labels == labels
    assert dict(mf.node_groups())["beta"] == ["beta one", "beta two"]


def test_create_discover_does_not_write_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Current all-at-once discover authoring has no scenario provenance."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "noscn", "d", fit_mode="pca",
        node_corpora={"a": ["x"], "b": ["y"], "default": ["z"]},
    )
    assert not (folder / "scenarios.json").exists()


def test_init_discover_rejects_duplicate_and_bad_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    with pytest.raises(ManifoldFormatError):
        init_discover_manifold_folder(
            "local", "dup", "", fit_mode="pca", labels=["a", "a"],
        )
    with pytest.raises(ManifoldFormatError):
        init_discover_manifold_folder(
            "local", "bad", "", fit_mode="pca", labels=["a", "bad.label"],
        )
    with pytest.raises(ManifoldFormatError):
        init_discover_manifold_folder(
            "local", "wrongmode", "", fit_mode="authored", labels=["a", "b"],
        )


def test_append_discover_rejects_empty_statements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = init_discover_manifold_folder(
        "local", "emptychk", "", fit_mode="pca", labels=["a", "b"],
    )
    with pytest.raises(ManifoldFormatError):
        append_discover_manifold_node(folder, 0, "a", [])
    with pytest.raises(ManifoldFormatError):
        append_discover_manifold_node(folder, 0, "a", ["   "])


# ----------------------------------------- discover generation planner ---

def test_plan_fresh_creates_skeleton_all_pending(tmp_path: Path):
    folder = tmp_path / "m"
    plan = plan_discover_generation(
        folder, "m", "desc", fit_mode="pca",
        labels=["a", "b", "c"], hyperparams={"max_dim": 4},
    )
    assert isinstance(plan, DiscoverGenerationPlan)
    assert not plan.resumed
    assert plan.pending == ("a", "b", "c")
    assert plan.added == ()
    assert plan.index_of == {"a": 0, "b": 1, "c": 2}
    assert (folder / "manifold.json").exists()
    assert (folder / "nodes").is_dir()


def test_plan_force_reset_and_skeleton_share_manifest_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.store.manifold_folder import _locked_manifest

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = tmp_path / "manifolds" / "local" / "m"
    first = plan_discover_generation(
        folder, "m", "old", fit_mode="pca", labels=["a", "b"],
    )
    append_discover_manifold_node(folder, first.index_of["a"], "a", ["old"])

    started = threading.Event()
    done = threading.Event()
    errors: list[BaseException] = []

    def reset() -> None:
        started.set()
        try:
            plan_discover_generation(
                folder, "m", "new", fit_mode="pca", labels=["c", "d"],
                force=True,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            done.set()

    with _locked_manifest(folder):
        worker = threading.Thread(target=reset)
        worker.start()
        assert started.wait(1.0)
        assert not done.wait(0.1)
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert errors == []
    assert done.is_set()
    data = json.loads((folder / "manifold.json").read_text())
    assert data["description"] == "new"
    assert [node["label"] for node in data["nodes"]] == ["c", "d"]
    assert not (folder / "nodes" / "00_a.json").exists()


def test_plan_force_validates_before_removing_prior_folder(tmp_path: Path) -> None:
    folder = tmp_path / "m"
    plan_discover_generation(
        folder, "m", "kept", fit_mode="pca", labels=["a", "b"],
    )
    before = (folder / "manifold.json").read_bytes()

    with pytest.raises(ManifoldFormatError):
        plan_discover_generation(
            folder, "m", "bad", fit_mode="pca", labels=["not.valid", "b"],
            force=True,
        )

    assert (folder / "manifold.json").read_bytes() == before


def test_plan_force_recovers_partial_folder_without_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = tmp_path / "manifolds" / "local" / "m"
    folder.mkdir(parents=True)
    (folder / "orphan").write_text("partial")

    plan = plan_discover_generation(
        folder, "m", "fresh", fit_mode="pca", labels=["a", "b"], force=True,
    )

    assert plan.pending == ("a", "b")
    assert not (folder / "orphan").exists()
    assert (folder / "manifold.json").exists()


def test_plan_force_failure_mid_skeleton_keeps_prior_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forced re-plan swaps the skeleton in; it never rmtrees first.

    The old path removed the prior folder and then wrote the skeleton in
    place, so a failure in between lost the corpus and every fit.
    """
    from drowse.store import manifold_authoring

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = tmp_path / "manifolds" / "local" / "m"
    plan = plan_discover_generation(folder, "m", "kept", fit_mode="pca", labels=["a", "b"])
    for label in ("a", "b"):
        append_discover_manifold_node(folder, plan.index_of[label], label, [f"{label}."])
    before = {p.name: p.read_bytes() for p in folder.rglob("*.json")}

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("injected skeleton failure")

    monkeypatch.setattr(manifold_authoring, "write_json_atomic", boom)
    with pytest.raises(OSError, match="injected"):
        plan_discover_generation(
            folder, "m", "new", fit_mode="pca", labels=["c", "d"], force=True,
        )
    assert {p.name: p.read_bytes() for p in folder.rglob("*.json")} == before
    assert ManifoldFolder.load(folder).node_labels == ["a", "b"]


def test_plan_resume_reports_only_missing(tmp_path: Path):
    folder = tmp_path / "m"
    labels = ["a", "b", "c", "d"]
    plan = plan_discover_generation(folder, "m", "d", fit_mode="pca", labels=labels)
    # Simulate a kill after 2 of 4 nodes.
    for lbl in ["a", "b"]:
        append_discover_manifold_node(folder, plan.index_of[lbl], lbl, [f"{lbl} one"])
    plan2 = plan_discover_generation(folder, "m", "d", fit_mode="pca", labels=labels)
    assert plan2.resumed
    assert plan2.pending == ("c", "d")           # only the missing ones
    assert plan2.added == ()
    assert plan2.index_of == {"a": 0, "b": 1, "c": 2, "d": 3}


def test_plan_add_nodes_extends_manifest(tmp_path: Path):
    folder = tmp_path / "m"
    labels = ["a", "b", "c"]
    plan = plan_discover_generation(folder, "m", "d", fit_mode="pca", labels=labels)
    for lbl in labels:
        append_discover_manifold_node(folder, plan.index_of[lbl], lbl, [f"{lbl} x"])
    # Re-plan with a superset roster — add-nodes.
    plan2 = plan_discover_generation(
        folder, "m", "d", fit_mode="pca", labels=labels + ["d", "e"],
    )
    assert plan2.resumed
    assert plan2.added == ("d", "e")
    assert plan2.pending == ("d", "e")            # only the new labels
    mj = json.loads((folder / "manifold.json").read_text())
    assert [n["label"] for n in mj["nodes"]] == ["a", "b", "c", "d", "e"]
    for lbl in ["d", "e"]:
        append_discover_manifold_node(folder, plan2.index_of[lbl], lbl, [f"{lbl} x"])
    mf = ManifoldFolder.load(folder)
    assert mf.node_labels == ["a", "b", "c", "d", "e"]


def test_plan_partial_rejected_by_load_but_resumable(tmp_path: Path):
    """A partial fails strict load yet the lenient planner resumes it."""
    folder = tmp_path / "m"
    plan_discover_generation(folder, "m", "d", fit_mode="pca", labels=["a", "b"])
    append_discover_manifold_node(folder, 0, "a", ["x"])
    with pytest.raises(ManifoldFormatError):
        ManifoldFolder.load(folder)
    plan2 = plan_discover_generation(folder, "m", "d", fit_mode="pca", labels=["a", "b"])
    assert plan2.pending == ("b",)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda data: data.update({"legacy": True}), "unknown field"),
        (lambda data: data.pop("format_version"), "need exactly"),
        (lambda data: data.update({"name": "other"}), "not 'm'"),
        (lambda data: data["nodes"][0].update({"coords": [0.0]}), "fields must be"),
    ],
)
def test_plan_resume_requires_exact_partial_manifest(
    tmp_path: Path, mutation: Any, match: str,
) -> None:
    folder = tmp_path / "m"
    plan_discover_generation(folder, "m", "d", fit_mode="pca", labels=["a", "b"])
    data = json.loads((folder / "manifold.json").read_text())
    mutation(data)
    (folder / "manifold.json").write_text(json.dumps(data))

    with pytest.raises(ManifoldFormatError, match=match):
        plan_discover_generation(
            folder, "m", "d", fit_mode="pca", labels=["a", "b"],
        )


def test_plan_rejects_non_discover_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "authored", "d", domain, _author_nodes(["a", "b", "c"]),
    )
    with pytest.raises(ManifoldFormatError):
        plan_discover_generation(
            folder, "authored", "d", fit_mode="pca", labels=["x", "y"],
        )


def _bundled_axis_package(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, description: str = "v1",
) -> Path:
    """Fake ``drowse/data/manifolds/axis`` package data under ``root``.

    ``DROWSE_HOME`` is ``root``'s parent, so ``default/axis`` materializes at
    ``<root parent>/manifolds/default/axis``.
    """
    from drowse.store import bundled

    monkeypatch.setenv("DROWSE_HOME", str(root.parent))

    pkg = root / "axis"
    (pkg / "nodes").mkdir(parents=True, exist_ok=True)
    (pkg / "manifold.json").write_text(json.dumps({
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": "axis",
        "description": description,
        "fit_mode": "pca",
        "hyperparams": {},
        "nodes": [
            {"label": "pos", "role": None, "kind": None},
            {"label": "neg", "role": None, "kind": None},
        ],
        "files": {},
        "source": "bundled", "tags": [], "template_ref": None,
    }))
    (pkg / "nodes" / "00_pos.json").write_text('["a"]')
    (pkg / "nodes" / "01_neg.json").write_text('["b"]')
    monkeypatch.setattr(
        bundled._resources, "files",
        lambda package: root if package == "drowse.data.manifolds" else None,
    )
    return pkg


def _fake_fit(target: Path) -> dict[str, str]:
    """Drop a fitted pair into ``target`` and record its proofs."""
    tensor = target / tensor_filename("test/model")
    tensor.write_bytes(b"tensor-bytes")
    tensor.with_suffix(".json").write_text("{}")
    manifest = json.loads((target / "manifold.json").read_text())
    manifest["files"] = {tensor.name: "aa" * 32, tensor.with_suffix(".json").name: "bb" * 32}
    (target / "manifold.json").write_text(json.dumps(manifest))
    return manifest["files"]


def test_bundled_copy_mirrors_only_package_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package-data materialization must not mirror local metadata files."""
    from drowse.store import bundled

    pkg = _bundled_axis_package(tmp_path / "bundled", monkeypatch)
    (pkg / "extra.json").write_text('{"extra": []}')
    (pkg / ".DS_Store").write_text("finder")
    (pkg / "nodes" / ".DS_Store").write_text("finder")
    (pkg / "nodes" / "notes.txt").write_text("not package data")

    default_dir = tmp_path / "manifolds" / "default"
    bundled._materialize_manifold("axis", force=False)
    target = default_dir / "axis"
    assert sorted(p.name for p in target.iterdir()) == ["manifold.json", "nodes"]
    assert sorted(p.name for p in (target / "nodes").iterdir()) == [
        "00_pos.json", "01_neg.json",
    ]
    ManifoldFolder.load(target)


def test_bundled_refresh_ignores_local_fit_proofs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit proofs in ``files`` must not read as a bundle update."""
    from drowse.store import bundled

    _bundled_axis_package(tmp_path / "bundled", monkeypatch)
    default_dir = tmp_path / "manifolds" / "default"
    bundled._materialize_manifold("axis", force=False)
    target = default_dir / "axis"
    proofs = _fake_fit(target)
    before = (target / "manifold.json").stat().st_ino

    bundled._materialize_manifold("axis", force=False)

    assert json.loads((target / "manifold.json").read_text())["files"] == proofs
    assert (target / "manifold.json").stat().st_ino == before, "no-drift must not rewrite"


def test_bundled_update_carries_tensor_proofs_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine bundle update keeps fitted pairs and their proofs only."""
    from drowse.store import bundled

    root = tmp_path / "bundled"
    _bundled_axis_package(root, monkeypatch)
    default_dir = tmp_path / "manifolds" / "default"
    bundled._materialize_manifold("axis", force=False)
    target = default_dir / "axis"
    proofs = _fake_fit(target)
    manifest = json.loads((target / "manifold.json").read_text())
    manifest["files"].update({
        "gone.safetensors": "cc" * 32,   # file no longer on disk
        "manifold.json": "dd" * 32,      # never carried
    })
    (target / "manifold.json").write_text(json.dumps(manifest))
    (target / "nodes" / "00_pos.json").write_text('["edited"]')

    _bundled_axis_package(root, monkeypatch, description="v2")
    with pytest.warns(UserWarning, match="refreshed bundled manifold default/axis"):
        bundled._materialize_manifold("axis", force=False)

    refreshed = json.loads((target / "manifold.json").read_text())
    assert refreshed["description"] == "v2"
    assert refreshed["files"] == proofs
    assert (target / tensor_filename("test/model")).read_bytes() == b"tensor-bytes"
    assert (target / "nodes" / "00_pos.json").read_text() == '["a"]'


@pytest.mark.parametrize("damage", ["manifest", "node"])
def test_bundled_interrupted_copy_is_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    """A folder left without its manifest or a declared node file is drift.

    Before the fix a missing manifest made the materializer return early
    forever, and a missing node file (with matching manifest hashes) was
    skipped — either way the bundled manifold vanished from the index.
    """
    from drowse.store import bundled

    _bundled_axis_package(tmp_path / "bundled", monkeypatch)
    default_dir = tmp_path / "manifolds" / "default"
    bundled._materialize_manifold("axis", force=False)
    target = default_dir / "axis"
    if damage == "manifest":
        (target / "manifold.json").unlink()
    else:
        (target / "nodes" / "01_neg.json").unlink()

    with pytest.warns(UserWarning, match="refreshed bundled manifold"):
        bundled._materialize_manifold("axis", force=False)
    assert ManifoldFolder.load(target).node_labels == ["pos", "neg"]


def test_bundled_copy_failure_keeps_previous_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-copy is staged: a failure mid-copy leaves the old folder intact."""
    from drowse.store import bundled

    root = tmp_path / "bundled"
    _bundled_axis_package(root, monkeypatch)
    default_dir = tmp_path / "manifolds" / "default"
    bundled._materialize_manifold("axis", force=False)
    target = default_dir / "axis"
    _bundled_axis_package(root, monkeypatch, description="v2")

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("injected copy failure")

    monkeypatch.setattr(bundled, "write_json_atomic", boom)
    with pytest.raises(OSError, match="injected"):
        bundled._materialize_manifold("axis", force=False)
    assert json.loads((target / "manifold.json").read_text())["description"] == "v1"
    assert ManifoldFolder.load(target).node_labels == ["pos", "neg"]
    from drowse.base.staging import staging_root

    assert not any(staging_root().iterdir())


def test_pack_refresh_restores_bundled_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``refresh_manifold`` on a bundled folder re-copies the shipped corpus.

    It used to call the process-scoped materializer, which had already run, so
    an edited or deleted node file was never restored while reporting success.
    """
    from drowse.store import bundled

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    bundled.reset_bundled_state()
    bundled.materialize_bundled_artifacts()
    target = tmp_path / "manifolds" / "default" / "formal.casual"
    shipped = (target / "nodes" / "00_formal.json").read_text()
    (target / "nodes" / "00_formal.json").write_text('["edited"]')
    (target / "nodes" / "01_casual.json").unlink()
    proofs = _fake_fit(target)

    assert refresh_manifold("default", "formal.casual") == "bundled"

    assert (target / "nodes" / "00_formal.json").read_text() == shipped
    assert (target / "nodes" / "01_casual.json").is_file()
    assert json.loads((target / "manifold.json").read_text())["files"] == proofs
    assert ManifoldFolder.load(target, verify_manifest=False).name == "formal.casual"


def test_bundled_manifold_names_skips_incomplete_package_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial generation directory must not materialize as a bundled manifold."""
    from drowse.store import bundled

    root = tmp_path / "bundled"
    complete = root / "complete"
    (complete / "nodes").mkdir(parents=True)
    (complete / "manifold.json").write_text(json.dumps({
        "nodes": [{"label": "alpha"}, {"label": "beta"}],
    }))
    (complete / "nodes" / "00_alpha.json").write_text('["a"]')
    (complete / "nodes" / "01_beta.json").write_text('["b"]')

    partial = root / "partial"
    (partial / "nodes").mkdir(parents=True)
    (partial / "manifold.json").write_text(json.dumps({
        "nodes": [{"label": "alpha"}, {"label": "beta"}],
    }))
    (partial / "nodes" / "00_alpha.json").write_text('["a"]')

    junk = root / "junk"
    junk.mkdir()
    (junk / "manifold.json").write_text("{not json")

    monkeypatch.setattr(
        bundled._resources, "files",
        lambda package: root if package == "drowse.data.manifolds" else None,
    )
    assert bundled.bundled_manifold_names() == ["complete"]


# =================================================== baked (corpus-less) ===
#
# A baked manifold is a pre-fitted, corpus-less artifact (merge output /
# imported control vector): geometry frozen in the tensor, no ``nodes/``
# corpus, never re-fits.


def _baked_manifold(name: str = "merged", n_layers: int = 3):
    """Build an affine R=1 Manifold (the merge/import shape) via the fold."""
    from drowse.fit.capture import fold_directions_to_subspace

    directions = {i: torch.randn(8) for i in range(n_layers)}
    whitener = _target_whitener(dim=8, layers=tuple(range(n_layers)))
    return (
        fold_directions_to_subspace(
            name, directions, whitener.layer_means,
            label=name, whitener=whitener,
        ),
        directions,
    )


_MERGE_COMPONENTS = {
    "0:default/x": {"selector": "default/x", "alpha": 1.0, "tensor_sha256": "0" * 64},
}


def _publish_baked(manifold: Any, *, model_id: str = "test/model", force: bool = False) -> Path:
    return publish_baked_manifold(
        "local", "merged", "a merged vector",
        [BakedTensor(
            model_id=model_id, manifold=manifold, model_fingerprint=f"fp:{model_id}",
            components=_MERGE_COMPONENTS,
        )],
        method="merge", force=force,
    )


def test_baked_manifold_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.store.manifold_tensors import load_manifold
    from drowse.fit.capture import folded_directions

    manifold, directions = _baked_manifold("merged")
    folder = _publish_baked(manifold)
    mf = ManifoldFolder.load(folder)
    assert mf.fit_mode == "baked"
    assert mf.is_discover is False
    assert mf.node_labels == ["merged"]
    assert mf.files, "files manifest should carry the published proofs"
    assert not (folder / "nodes").exists()
    assert mf.tensor_models() == [safe_model_id("test/model")]

    # The tensor folds back to the per-layer directions it was baked from.
    (tensor,) = list(folder.glob("*.safetensors"))
    folded = folded_directions(load_manifold(tensor))
    assert set(folded) == set(directions)


def test_baked_publication_hashes_each_new_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.store import manifold_authoring as authoring_module
    real_hash = authoring_module.hash_file
    hashed: list[Path] = []

    def count_hash(path: Path) -> str:
        hashed.append(Path(path))
        return real_hash(Path(path))

    monkeypatch.setattr(authoring_module, "hash_file", count_hash)
    manifold, _ = _baked_manifold("merged")
    _publish_baked(manifold)

    tensor = tensor_filename("test/model")
    assert [path.name for path in hashed] == [tensor, Path(tensor).with_suffix(".json").name]


def test_baked_publication_crash_leaves_no_partial_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure mid-publication leaves neither a partial nor a discoverable folder."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.base.paths import manifold_dir
    from drowse.store import manifold_tensors

    manifold, _directions = _baked_manifold("merged")
    real_save = manifold_tensors.save_manifold

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("injected baked publication failure")

    monkeypatch.setattr(manifold_tensors, "save_manifold", fail)
    with pytest.raises(OSError, match="injected"):
        _publish_baked(manifold)
    assert not manifold_dir("local", "merged").exists()
    assert list(iter_manifold_folders()) == []

    monkeypatch.setattr(manifold_tensors, "save_manifold", real_save)
    folder = _publish_baked(manifold)
    assert ManifoldFolder.load(folder).tensor_models() == [safe_model_id("test/model")]


def test_baked_publication_refuses_existing_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    manifold, _directions = _baked_manifold("merged")
    folder = _publish_baked(manifold)
    with pytest.raises(FileExistsError, match="already exists"):
        _publish_baked(manifold)
    replaced = _publish_baked(manifold, force=True)
    assert replaced == folder
    assert ManifoldFolder.load(folder).tensor_models() == [safe_model_id("test/model")]


def test_baked_publication_replaces_manifestless_partial_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A directory with no manifest is a crash leftover, not a manifold."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    from drowse.base.paths import manifold_dir

    folder = manifold_dir("local", "merged")
    folder.mkdir(parents=True)
    (folder / "stale.safetensors").write_bytes(b"partial")
    (folder / "stale.json").write_text("{bad sidecar")
    manifold, _directions = _baked_manifold("merged")

    created = _publish_baked(manifold)
    assert not (created / "stale.safetensors").exists()
    assert ManifoldFolder.load(created).tensor_models() == [safe_model_id("test/model")]


def test_baked_manifold_clear_and_scoped_refresh_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    manifold, _ = _baked_manifold("merged")
    _publish_baked(manifold)
    # Both tensor-deleting ops refuse — a baked manifold can't re-fit, so
    # clearing its tensor would destroy the only copy of its geometry.
    with pytest.raises(BakedManifoldError, match="rm"):
        clear_manifold_tensors("local", "merged")
    with pytest.raises(BakedManifoldError):
        refresh_manifold("local", "merged", model_scope="test/model")


def test_baked_manifold_node_groups_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    manifold, _ = _baked_manifold("merged")
    mf = ManifoldFolder.load(_publish_baked(manifold))
    with pytest.raises(ManifoldFormatError, match="no node corpus"):
        mf.node_groups()


def test_baked_manifold_rejects_coords_on_node(tmp_path: Path):
    """A baked node is label-only — coords would be ambiguous against the
    frozen tensor geometry, so the loader refuses them."""
    folder = tmp_path / "merged"
    (folder).mkdir()
    payload = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": "merged",
        "description": "",
        "fit_mode": "baked",
        "domain": {"type": "custom", "embed_dim": 1, "bounds": None},
        "nodes": [{"label": "merged", "coords": [1.0], "role": None, "kind": None}],
        "files": {},
        "source": "local", "tags": [], "template_ref": None,
    }
    (folder / "manifold.json").write_text(json.dumps(payload))
    with pytest.raises(ManifoldFormatError, match="fields must be"):
        ManifoldFolder.load(folder)


def test_baked_manifold_requires_tensor(tmp_path: Path):
    """A tensor-less baked folder is corrupt — the tensor is its only geometry."""
    folder = tmp_path / "merged"
    folder.mkdir()
    payload = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": "merged",
        "description": "",
        "fit_mode": "baked",
        "domain": {"type": "custom", "embed_dim": 1, "bounds": None},
        "nodes": [{"label": "merged", "role": None, "kind": None}],
        "files": {},
        "source": "local", "tags": [], "template_ref": None,
    }
    (folder / "manifold.json").write_text(json.dumps(payload))
    with pytest.raises(ManifoldFormatError, match="no fitted tensor"):
        ManifoldFolder.load(folder)


def test_baked_manifold_requires_domain(tmp_path: Path):
    folder = tmp_path / "merged"
    folder.mkdir()
    payload = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": "merged",
        "description": "",
        "fit_mode": "baked",
        "nodes": [{"label": "merged", "role": None, "kind": None}],
        "files": {},
        "source": "local", "tags": [], "template_ref": None,
    }
    (folder / "manifold.json").write_text(json.dumps(payload))
    with pytest.raises(ManifoldFormatError, match="needs a 'domain'"):
        ManifoldFolder.load(folder)


# ============================================================ regressions ===


def test_transfer_drops_source_capture_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``_from-`` sidecar must not claim the source fit's checkpoint/capture."""
    from drowse.store.manifold_tensors import load_manifold

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    domain = {"type": "box", "axes": [
        {"name": "t", "periodic": False, "lo": 0.0, "hi": 1.0}]}
    folder, _ = create_manifold_folder(
        "local", "mood", "", domain, _author_nodes(["a", "b", "c"]),
    )
    src_model, tgt_model = "google/gemma-3-4b-it", "Qwen/Qwen2.5-7B-Instruct"
    src_tensor = _fit_real_manifold(folder, src_model, dim=6)
    sidecar_path = src_tensor.with_suffix(".json")
    payload = json.loads(sidecar_path.read_text())
    payload.update({
        "model_source_fingerprint": "src:checkpoint",
        "capture_sha256": "1" * 64,
        "capture_version": 3,
        "capture_render_sha256": "2" * 64,
        "baseline_prompts_sha256": "3" * 64,
    })
    sidecar_path.write_text(json.dumps(payload))
    ManifoldFolder.load(folder, verify_manifest=False).update_file_hashes(src_tensor, sidecar_path)

    whitener = _target_whitener()
    out = transfer_manifold(
        folder, from_model=src_model, to_model=tgt_model,
        alignment=_alignments({L: torch.eye(6) for L in load_manifold(src_tensor).layers}),
        source_model_fingerprint=f"fp:{src_model}",
        target_model_fingerprint=f"fp:{tgt_model}",
        whitener=whitener, target_layer_means=whitener.layer_means,
    )
    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert sidecar["model_fingerprint"] == f"fp:{tgt_model}"
    for key in (
        "model_source_fingerprint", "capture_sha256", "capture_version",
        "capture_render_sha256", "baseline_prompts_sha256",
    ):
        assert sidecar[key] is None, key


def test_merge_fit_mode_override_does_not_inherit_foreign_hyperparams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``manifold merge --method pca`` over spectral sources used to fail.

    The sources' ``k_nn`` was inherited into a ``pca`` folder, which the
    exact hyperparameter whitelist rejects.
    """
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    for name, labels in (("a", ["x1", "x2", "x3"]), ("b", ["y1", "y2", "y3"])):
        create_discover_manifold_folder(
            "local", name, "", fit_mode="spectral",
            node_corpora={label: [f"{label}."] for label in labels},
            hyperparams={"k_nn": 2},
        )
    merged = merge_discover_manifolds(
        "local", "ab", "", sources=[("local", "a"), ("local", "b")], fit_mode="pca",
    )
    mf = ManifoldFolder.load(merged)
    assert mf.fit_mode == "pca"
    assert mf.hyperparams == {}

    same_mode = merge_discover_manifolds(
        "local", "ab2", "", sources=[("local", "a"), ("local", "b")],
    )
    assert ManifoldFolder.load(same_mode).hyperparams == {"k_nn": 2}


def test_update_manifold_folder_refuses_nodes_on_non_authored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PATCH ``nodes`` on a discover folder used to escape as a plain ValueError (500)."""
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "d", "", fit_mode="pca",
        node_corpora={"x": ["x."], "y": ["y."]},
    )
    with pytest.raises(ManifoldFormatError, match="only an authored manifold"):
        update_manifold_folder(folder, nodes=_author_nodes(["a", "b", "c"]))
    assert not (folder / "nodes" / "00_a.json").exists()
    update_manifold_folder(folder, description="still editable")
    assert ManifoldFolder.load(folder).description == "still editable"


def test_bad_sidecar_domain_is_a_format_error_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One sidecar with a bogus domain must not break discovery.

    The domain check raised a plain ``ValueError`` through
    ``iter_manifold_folders``, breaking ``pack ls`` and silently disabling
    bare-label steering for every manifold.
    """
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "d", "", fit_mode="pca",
        node_corpora={"x": ["x."], "y": ["y."]},
    )
    tensor = folder / tensor_filename("some/model")
    tensor.write_bytes(b"x")
    payload = ManifoldSidecar.build(
        name="d", method="manifold_discover_pca",
        domain={"type": "custom", "embed_dim": 1, "bounds": None},
        node_labels=["x", "y"], fit_mode="pca",
    ).to_payload()
    payload["domain"] = {"type": "bogus"}
    tensor.with_suffix(".json").write_text(json.dumps(payload))

    listed = [mf.name for _ns, mf in iter_manifold_folders()]
    assert listed == ["d"]
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    assert mf.tensor_models() == []
    with pytest.raises(ManifoldFormatError, match="invalid domain"):
        mf.sidecar(tensor.stem)


def test_discovery_ignores_staging_and_misnamed_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crash-left stages and backups are never discovered as duplicates."""
    import shutil

    from drowse.base.staging import backup_path, staging_root

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    folder = create_discover_manifold_folder(
        "local", "foo", "", fit_mode="pca",
        node_corpora={"a": ["a."], "b": ["b."]},
    )
    for leftover in (backup_path(folder), staging_root() / "in-flight" / "foo"):
        shutil.copytree(folder, leftover)
    shutil.copytree(folder, folder.with_name("foo.copy"))

    assert [(ns, mf.name) for ns, mf in iter_manifold_folders()] == [("local", "foo")]
