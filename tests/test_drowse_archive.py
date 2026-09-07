from __future__ import annotations

import json
import hashlib
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch

from drowse.core.manifold import (
    MANIFOLD_FIT_POLICY_VERSION,
    CustomDomain,
    LayerSubspace,
    Manifold,
)
from drowse.io.manifold_authoring import (
    create_discover_manifold_folder,
    create_manifold_folder,
    create_manifold_from_template,
    iter_manifold_folders,
)
from drowse.io.manifold_tensors import save_manifold
from drowse.io.manifolds import ManifoldFolder, ManifoldFormatError
from drowse.io.paths import manifold_dir, drowse_home, tensor_filename
from drowse.io.drowse_archive import (
    DROWSE_ARCHIVE_FORMAT_VERSION,
    DrowseArchiveError,
    _begin_transaction,
    _promote_transaction,
    _tensor_shapes_and_finiteness,
    _transaction_backup_path,
    _transaction_stage_path,
    _validate_member_path,
    _validate_zip_infos,
    export_drowse_archive,
    install_drowse_archive,
    recover_drowse_archive_transactions,
)
from drowse.io.templates import (
    TemplateFolder,
    create_template_folder,
    template_dir,
)


@pytest.fixture(autouse=True)
def _home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path / "home"))


def _discover(name: str = "demo") -> Path:
    return create_discover_manifold_folder(
        "local",
        name,
        "demo manifold",
        fit_mode="pca",
        node_corpora={"calm": ["calm"], "alert": ["alert"]},
        hyperparams={"max_dim": 1},
    )


def _template_manifold() -> tuple[Path, Path]:
    template = create_template_folder(
        "local",
        "weekday",
        slot="[DAY]",
        values=["Monday", "Tuesday"],
        contexts=[{
            "turns": [{"role": "user", "content": "what day is it?"}],
            "assistant": "today is [DAY]",
        }],
    )
    manifold = create_manifold_from_template(
        "local",
        "days",
        "days",
        template_ref="local/weekday",
        fit_mode="pca",
    )
    assert template.path is not None
    return manifold, template.path


def _fit(folder: Path, *, finite: bool = True) -> Path:
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    basis = torch.tensor([[1.0, 0.0]])
    if not finite:
        basis[0, 0] = float("nan")
    node_coords = torch.tensor([[-1.0], [1.0]])
    manifold = Manifold(
        name=mf.name,
        domain=CustomDomain(1),
        node_labels=list(mf.node_labels),
        node_coords=node_coords,
        layers={
            0: LayerSubspace.affine(
                mean=torch.zeros(2), basis=basis,
                node_coords=node_coords.clone(),
            ),
        },
        feature_space="raw",
        mahalanobis_share={0: 1.0},
    )
    path = folder / tensor_filename("test/model")
    save_manifold(
        manifold,
        path,
        {
            "method": "manifold_discover_pca",
            "fit_mode": "pca",
            "hyperparams": dict(mf.hyperparams),
            "nodes_sha256": mf.nodes_sha256(),
            "model_fingerprint": "test-model",
            "fit_policy_version": MANIFOLD_FIT_POLICY_VERSION,
        },
    )
    mf.update_file_hashes(path, path.with_suffix(".json"))
    return path


def _archive_entries(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def _write_entries(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)


def _rewrite_manifest(
    source: Path, target: Path, mutate: Any,
) -> None:
    entries = _archive_entries(source)
    manifest = json.loads(entries["pack.json"])
    mutate(manifest, entries)
    entries["pack.json"] = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    _write_entries(target, entries)


def _replace_manifested_json(
    manifest: dict[str, Any],
    entries: dict[str, bytes],
    path: str,
    payload: dict[str, Any] | list[Any],
) -> None:
    content = (json.dumps(payload, sort_keys=True) + "\n").encode()
    entries[path] = content
    manifest["files"][path] = {
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _rewrite_fit_pair(
    folder: Path,
    tensor: Path,
    *,
    tensor_name: str,
    sidecar_updates: dict[str, Any] | None = None,
) -> Path:
    sidecar = tensor.with_suffix(".json")
    sidecar_payload = json.loads(sidecar.read_text())
    sidecar_payload.update(sidecar_updates or {})
    sidecar.write_text(json.dumps(sidecar_payload))
    destination = folder / tensor_name
    destination_sidecar = destination.with_suffix(".json")
    tensor.rename(destination)
    sidecar.rename(destination_sidecar)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"].pop(tensor.name)
    manifest["files"].pop(sidecar.name)
    manifest["files"][destination.name] = hashlib.sha256(
        destination.read_bytes(),
    ).hexdigest()
    manifest["files"][destination_sidecar.name] = hashlib.sha256(
        destination_sidecar.read_bytes(),
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    return destination


def test_round_trip_preserves_manifold_and_exact_manifest(tmp_path: Path) -> None:
    folder = _discover()
    output = tmp_path / "demo.drowse"

    assert export_drowse_archive("local", "demo", output=output) == output
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("pack.json"))
    assert manifest["format_version"] == DROWSE_ARCHIVE_FORMAT_VERSION
    assert manifest["kind"] == "drowse-manifold"
    assert manifest["primary"] == "manifolds/local/demo"
    assert manifest["template"] is None
    assert set(manifest["files"]) == names - {"pack.json"}
    assert "manifolds/local/demo/manifold.json" in names

    shutil.rmtree(folder)
    installed = install_drowse_archive(output)
    assert installed == manifold_dir("local", "demo")
    assert ManifoldFolder.load(installed).node_labels == ["calm", "alert"]


def test_monopolar_discover_fit_exports_and_installs(tmp_path: Path) -> None:
    folder = create_discover_manifold_folder(
        "local",
        "monopolar",
        "monopolar manifold",
        fit_mode="pca",
        node_corpora={"calm": ["calm"]},
        hyperparams={"max_dim": 1, "var_threshold": 0.7},
    )
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    node_coords = torch.tensor([[1.0]])
    manifold = Manifold(
        name=mf.name,
        domain=CustomDomain(1),
        node_labels=list(mf.node_labels),
        node_coords=node_coords,
        layers={
            0: LayerSubspace.affine(
                mean=torch.zeros(2),
                basis=torch.tensor([[1.0, 0.0]]),
                node_coords=node_coords.clone(),
            ),
        },
        feature_space="raw",
        mahalanobis_share={0: 1.0},
    )
    tensor = folder / tensor_filename("test/model")
    save_manifold(
        manifold,
        tensor,
        {
            "method": "manifold_monopolar",
            "fit_mode": mf.fit_mode,
            "hyperparams": dict(mf.hyperparams),
            "diagnostics": {},
            "nodes_sha256": mf.nodes_sha256(),
            "model_fingerprint": "test-model",
            "fit_policy_version": MANIFOLD_FIT_POLICY_VERSION,
            "share_metric": "mahalanobis",
            "subspace_metric": "euclidean",
        },
    )
    mf.update_file_hashes(tensor, tensor.with_suffix(".json"))

    output = tmp_path / "monopolar.drowse"
    export_drowse_archive("local", "monopolar", output=output)
    shutil.rmtree(folder)
    installed = install_drowse_archive(output)

    sidecar = ManifoldFolder.load(installed).sidecar(tensor.stem)
    assert sidecar.method == "manifold_monopolar"
    assert sidecar.fit_mode == "pca"
    assert sidecar.hyperparams == {"max_dim": 1, "var_threshold": 0.7}
    assert sidecar.diagnostics == {}


def test_install_rejects_duplicate_keys_in_inner_json(tmp_path: Path) -> None:
    _discover()
    valid = tmp_path / "valid.drowse"
    invalid = tmp_path / "invalid.drowse"
    export_drowse_archive("local", "demo", output=valid)

    def mutate(manifest: dict[str, Any], entries: dict[str, bytes]) -> None:
        path = "manifolds/local/demo/manifold.json"
        entries[path] = entries[path].replace(
            b'"name": "demo",',
            b'"name": "demo",\n  "name": "demo",',
            1,
        )
        manifest["files"][path] = {
            "size": len(entries[path]),
            "sha256": hashlib.sha256(entries[path]).hexdigest(),
        }

    _rewrite_manifest(valid, invalid, mutate)

    with pytest.raises(DrowseArchiveError, match="duplicate key 'name'"):
        install_drowse_archive(invalid)


def test_template_closure_installs_transactionally(tmp_path: Path) -> None:
    folder, template = _template_manifold()
    output = tmp_path / "days.drowse"
    export_drowse_archive("local", "days", output=output)
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("pack.json"))
    assert manifest["template"] == "templates/local/weekday"
    assert "templates/local/weekday/template.json" in manifest["files"]

    shutil.rmtree(folder)
    shutil.rmtree(template)
    install_drowse_archive(output)
    assert ManifoldFolder.load(manifold_dir("local", "days")).template_ref == (
        "local/weekday"
    )
    assert TemplateFolder.load(template_dir("local", "weekday")).values == (
        "Monday", "Tuesday",
    )


def test_export_rejects_unqualified_template_ref(tmp_path: Path) -> None:
    folder, _template = _template_manifold()
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["template_ref"] = "weekday"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "days.drowse"
    with pytest.raises(DrowseArchiveError, match="namespace-qualified"):
        export_drowse_archive("local", "days", output=output)


def test_different_existing_template_blocks_whole_install(tmp_path: Path) -> None:
    folder, template = _template_manifold()
    output = tmp_path / "days.drowse"
    export_drowse_archive("local", "days", output=output)
    shutil.rmtree(folder)
    shutil.rmtree(template)
    create_template_folder(
        "local",
        "weekday",
        slot="[DAY]",
        values=["Friday", "Saturday"],
        contexts=[{
            "turns": [{"role": "user", "content": "day?"}],
            "assistant": "[DAY]",
        }],
    )

    with pytest.raises(RuntimeError, match="different referenced template"):
        install_drowse_archive(output)
    assert not manifold_dir("local", "days").exists()
    assert TemplateFolder.load(template_dir("local", "weekday")).values == (
        "Friday", "Saturday",
    )


def test_matching_template_symlink_is_never_reused(tmp_path: Path) -> None:
    folder, template = _template_manifold()
    template_bytes = (template / "template.json").read_bytes()
    output = tmp_path / "days.drowse"
    export_drowse_archive("local", "days", output=output)
    shutil.rmtree(folder)
    shutil.rmtree(template)
    external = tmp_path / "template.json"
    external.write_bytes(template_bytes)
    template.mkdir(parents=True)
    (template / "template.json").symlink_to(external)

    with pytest.raises(DrowseArchiveError, match="not a regular file"):
        install_drowse_archive(output)
    assert not manifold_dir("local", "days").exists()


def test_matching_template_with_unexpected_entries_is_never_reused(
    tmp_path: Path,
) -> None:
    folder, template = _template_manifold()
    output = tmp_path / "days.drowse"
    export_drowse_archive("local", "days", output=output)
    shutil.rmtree(folder)
    (template / "unexpected.txt").write_text("unowned")

    with pytest.raises(DrowseArchiveError, match="exact template closure"):
        install_drowse_archive(output)
    assert not manifold_dir("local", "days").exists()


def test_force_replaces_template_and_manifold_together(tmp_path: Path) -> None:
    folder, template = _template_manifold()
    output = tmp_path / "days.drowse"
    export_drowse_archive("local", "days", output=output)
    shutil.rmtree(folder)
    shutil.rmtree(template)
    create_template_folder(
        "local",
        "weekday",
        slot="[DAY]",
        values=["Friday", "Saturday"],
        contexts=[{
            "turns": [{"role": "user", "content": "day?"}],
            "assistant": "[DAY]",
        }],
    )

    install_drowse_archive(output, force=True)
    assert ManifoldFolder.load(manifold_dir("local", "days")).name == "days"
    assert TemplateFolder.load(template_dir("local", "weekday")).values == (
        "Monday", "Tuesday",
    )


def test_force_refuses_to_replace_template_shared_by_another_manifold(
    tmp_path: Path,
) -> None:
    folder, template = _template_manifold()
    output = tmp_path / "days.drowse"
    export_drowse_archive("local", "days", output=output)
    shutil.rmtree(folder)
    shutil.rmtree(template)
    create_template_folder(
        "local",
        "weekday",
        slot="[DAY]",
        values=["Friday", "Saturday"],
        contexts=[{
            "turns": [{"role": "user", "content": "day?"}],
            "assistant": "[DAY]",
        }],
    )
    create_manifold_from_template(
        "local",
        "other",
        "other days",
        template_ref="local/weekday",
        fit_mode="pca",
    )

    with pytest.raises(RuntimeError, match="cannot replace shared template"):
        install_drowse_archive(output, force=True)

    assert not manifold_dir("local", "days").exists()
    assert TemplateFolder.load(template_dir("local", "weekday")).values == (
        "Friday", "Saturday",
    )


def test_as_rewrites_manifold_and_fitted_sidecar_identity(tmp_path: Path) -> None:
    folder = _discover()
    _fit(folder)
    output = tmp_path / "demo.drowse"
    export_drowse_archive("local", "demo", output=output)
    shutil.rmtree(folder)

    installed = install_drowse_archive(output, as_="shared/renamed")
    loaded = ManifoldFolder.load(installed)
    assert installed == manifold_dir("shared", "renamed")
    assert loaded.name == "renamed"
    with open(installed / tensor_filename("test/model").replace(
        ".safetensors", ".json",
    )) as handle:
        sidecar = json.load(handle)
    assert sidecar["name"] == "renamed"


@pytest.mark.parametrize("namespace", ["jlens", "sae"])
def test_install_rejects_reserved_destination_namespace(
    tmp_path: Path, namespace: str,
) -> None:
    folder = _discover()
    output = tmp_path / "demo.drowse"
    export_drowse_archive("local", "demo", output=output)
    shutil.rmtree(folder)

    with pytest.raises(DrowseArchiveError, match="reserved"):
        install_drowse_archive(output, as_=f"{namespace}/renamed")


def test_install_rejects_reserved_source_namespace(tmp_path: Path) -> None:
    _discover()
    valid = tmp_path / "valid.drowse"
    bad = tmp_path / "bad.drowse"
    export_drowse_archive("local", "demo", output=valid)

    def mutate(manifest: dict[str, Any], entries: dict[str, bytes]) -> None:
        old = "manifolds/local/demo"
        new = "manifolds/jlens/demo"
        manifest["primary"] = new
        for path in [name for name in entries if name.startswith(old + "/")]:
            replacement = new + path[len(old):]
            entries[replacement] = entries.pop(path)
            manifest["files"][replacement] = manifest["files"].pop(path)

    _rewrite_manifest(valid, bad, mutate)

    with pytest.raises(DrowseArchiveError, match="reserved"):
        install_drowse_archive(bad)


def test_install_resets_imported_fit_transaction_ownership(tmp_path: Path) -> None:
    folder = _discover()
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_id"] = "publisher-owned"
    manifest["fit_epochs"] = {"*:all": 91}
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "demo.drowse"
    export_drowse_archive("local", "demo", output=output)
    with zipfile.ZipFile(output) as archive:
        archived = json.loads(
            archive.read("manifolds/local/demo/manifold.json")
        )
    assert archived["artifact_id"] == "publisher-owned"
    assert archived["fit_epochs"] == {"*:all": 91}

    shutil.rmtree(folder)
    installed = install_drowse_archive(output)
    local_manifest = json.loads((installed / "manifold.json").read_text())
    assert "artifact_id" not in local_manifest
    assert "fit_epochs" not in local_manifest


def test_export_rejects_nonfinite_fitted_tensor(tmp_path: Path) -> None:
    folder = _discover()
    _fit(folder, finite=False)
    with pytest.raises(DrowseArchiveError, match="non-finite"):
        export_drowse_archive("local", "demo", output=tmp_path / "bad.drowse")


def test_export_requires_drowse_archive_suffix(tmp_path: Path) -> None:
    _discover()
    with pytest.raises(DrowseArchiveError, match="must end in .drowse"):
        export_drowse_archive("local", "demo", output=tmp_path / "demo.zip")


def test_export_refuses_destination_inside_manifold_closure() -> None:
    folder = _discover()
    destination = folder / "nested.drowse"
    with pytest.raises(DrowseArchiveError, match="outside its manifold closure"):
        export_drowse_archive("local", "demo", output=destination)
    assert not destination.exists()


def test_export_refuses_destination_inside_template_closure() -> None:
    _folder, template = _template_manifold()
    destination = template / "nested.drowse"
    with pytest.raises(DrowseArchiveError, match="outside its template closure"):
        export_drowse_archive("local", "days", output=destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "hyperparams",
    [{"k_nn": 3}, {"max_dim": True}, {"var_threshold": float("nan")}],
)
def test_discover_hyperparameters_are_mode_validated(
    hyperparams: dict[str, object],
) -> None:
    with pytest.raises(ManifoldFormatError):
        create_discover_manifold_folder(
            "local",
            "invalid",
            "invalid",
            fit_mode="pca",
            node_corpora={"calm": ["calm"], "alert": ["alert"]},
            hyperparams=hyperparams,
        )


def test_export_rejects_sae_filename_for_raw_feature_space(tmp_path: Path) -> None:
    folder = _discover()
    tensor = _fit(folder)
    _rewrite_fit_pair(
        folder,
        tensor,
        tensor_name=tensor_filename("test/model", release="release-a"),
    )

    with pytest.raises(DrowseArchiveError, match="filename variant"):
        export_drowse_archive("local", "demo", output=tmp_path / "bad.drowse")


def test_export_rejects_transfer_filename_with_different_source(
    tmp_path: Path,
) -> None:
    folder = _discover()
    tensor = _fit(folder)
    _rewrite_fit_pair(
        folder,
        tensor,
        tensor_name=tensor_filename(
            "test/model", transferred_from="source/model-a",
        ),
        sidecar_updates={
            "method": "manifold_procrustes_transfer",
            "source_model_id": "source/model-b",
            "source_model_fingerprint": "source-b-fingerprint",
            "transfer_quality_estimate": 0.95,
        },
    )

    with pytest.raises(DrowseArchiveError, match="filename variant"):
        export_drowse_archive("local", "demo", output=tmp_path / "bad.drowse")


def test_export_validates_affine_map_against_domain_embedding_dimension(
    tmp_path: Path,
) -> None:
    folder = _discover()
    tensor = _fit(folder)
    sidecar_path = tensor.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["domain"] = {"type": "sphere", "dim": 1}
    sidecar_path.write_text(json.dumps(sidecar))
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][sidecar_path.name] = hashlib.sha256(
        sidecar_path.read_bytes(),
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DrowseArchiveError, match="needs affine_map"):
        export_drowse_archive("local", "demo", output=tmp_path / "bad.drowse")


@pytest.mark.parametrize(
    ("relative", "content", "match"),
    [
        ("checkpoint.bin", b"checkpoint", "not part of manifold v10"),
        ("nested/hidden.safetensors", b"hidden", "not a top-level fitted file"),
    ],
)
def test_export_rejects_unknown_manifested_manifold_files(
    tmp_path: Path,
    relative: str,
    content: bytes,
    match: str,
) -> None:
    folder = _discover()
    path = folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][relative] = hashlib.sha256(content).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DrowseArchiveError, match=match):
        export_drowse_archive("local", "demo", output=tmp_path / "bad.drowse")


def test_export_parses_every_node_corpus(tmp_path: Path) -> None:
    folder = _discover()
    (folder / "nodes" / "00_calm.json").write_text("{}")

    with pytest.raises(DrowseArchiveError, match="node corpus failed validation"):
        export_drowse_archive("local", "demo", output=tmp_path / "bad.drowse")


@pytest.mark.parametrize("field", ["node_roles", "node_kinds", "nodes_sha256"])
def test_export_binds_fitted_sidecar_to_manifold_inputs(
    tmp_path: Path, field: str,
) -> None:
    folder = _discover()
    tensor = _fit(folder)
    sidecar_path = tensor.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    if field == "node_roles":
        sidecar[field] = ["guide", "guide"]
    elif field == "node_kinds":
        sidecar[field] = ["abstract", "abstract"]
    else:
        sidecar[field] = "f" * 64
    sidecar_path.write_text(json.dumps(sidecar))
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][sidecar_path.name] = hashlib.sha256(
        sidecar_path.read_bytes(),
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DrowseArchiveError, match="identity|node hash"):
        export_drowse_archive("local", "demo", output=tmp_path / "bad.drowse")


def test_export_rejects_tensor_rank_larger_than_hidden_dimension(
    tmp_path: Path,
) -> None:
    from safetensors.torch import load_file, save_file

    folder = _discover()
    tensor = _fit(folder)
    values = load_file(str(tensor))
    values["layer_0.basis"] = torch.zeros(3, 2)
    save_file(values, str(tensor))
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][tensor.name] = hashlib.sha256(
        tensor.read_bytes(),
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DrowseArchiveError, match="invalid mean/basis"):
        export_drowse_archive("local", "demo", output=tmp_path / "bad.drowse")


def test_tensor_shape_is_bounded_before_chunk_iteration(tmp_path: Path) -> None:
    path = tmp_path / "hostile.safetensors"
    header = json.dumps({
        "x": {
            "dtype": "F32",
            "shape": [18_446_744_073_709_551_615, 0],
            "data_offsets": [0, 0],
        },
    }, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(len(header).to_bytes(8, "little") + header)

    with pytest.raises(DrowseArchiveError, match="zero or negative dimension"):
        _tensor_shapes_and_finiteness(path)


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "C:/drive",
        "manifolds\\local\\demo",
        "manifolds//demo",
        "manifolds/./demo",
        "manifolds/../demo",
        "bad\x00path",
    ],
)
def test_rejects_unsafe_member_paths(name: str) -> None:
    with pytest.raises(DrowseArchiveError):
        _validate_member_path(name)


def test_rejects_duplicate_and_casefold_colliding_paths() -> None:
    duplicate = [zipfile.ZipInfo("pack.json"), zipfile.ZipInfo("pack.json")]
    with pytest.raises(DrowseArchiveError, match="duplicate"):
        _validate_zip_infos(duplicate)
    collision = [zipfile.ZipInfo("pack.json"), zipfile.ZipInfo("PACK.JSON")]
    with pytest.raises(DrowseArchiveError, match="case-insensitively"):
        _validate_zip_infos(collision)


def test_rejects_nul_truncated_raw_zip_path() -> None:
    info = zipfile.ZipInfo("pack.json\x00ignored")
    assert info.filename == "pack.json"
    with pytest.raises(DrowseArchiveError, match="raw path"):
        _validate_zip_infos([info])


def test_rejects_symlink_special_and_encrypted_entries() -> None:
    symlink = zipfile.ZipInfo("pack.json")
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(DrowseArchiveError, match="regular file"):
        _validate_zip_infos([symlink])

    special = zipfile.ZipInfo("pack.json")
    special.create_system = 3
    special.external_attr = (stat.S_IFIFO | 0o600) << 16
    with pytest.raises(DrowseArchiveError, match="regular file"):
        _validate_zip_infos([special])

    encrypted = zipfile.ZipInfo("pack.json")
    encrypted.flag_bits |= 0x1
    with pytest.raises(DrowseArchiveError, match="encrypted"):
        _validate_zip_infos([encrypted])


def test_rejects_entry_total_ratio_and_count_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.io.drowse_archive as module

    oversized = zipfile.ZipInfo("pack.json")
    oversized.file_size = 11
    oversized.compress_size = 11
    monkeypatch.setattr(module, "DROWSE_ARCHIVE_MAX_FILE_BYTES", 10)
    with pytest.raises(DrowseArchiveError, match="larger"):
        _validate_zip_infos([oversized])

    monkeypatch.setattr(module, "DROWSE_ARCHIVE_MAX_FILE_BYTES", 100)
    monkeypatch.setattr(module, "DROWSE_ARCHIVE_MAX_TOTAL_BYTES", 15)
    first = zipfile.ZipInfo("one")
    first.file_size = first.compress_size = 10
    second = zipfile.ZipInfo("two")
    second.file_size = second.compress_size = 10
    with pytest.raises(DrowseArchiveError, match="total"):
        _validate_zip_infos([first, second])

    monkeypatch.setattr(module, "DROWSE_ARCHIVE_MAX_FILE_BYTES", 1000)
    monkeypatch.setattr(module, "DROWSE_ARCHIVE_MAX_TOTAL_BYTES", 1000)
    ratio = zipfile.ZipInfo("ratio")
    ratio.file_size = 101
    ratio.compress_size = 1
    with pytest.raises(DrowseArchiveError, match="100:1"):
        _validate_zip_infos([ratio])

    monkeypatch.setattr(module, "DROWSE_ARCHIVE_MAX_ENTRIES", 1)
    with pytest.raises(DrowseArchiveError, match="entries"):
        _validate_zip_infos([zipfile.ZipInfo("one"), zipfile.ZipInfo("two")])


def test_rejects_payload_hash_mismatch(tmp_path: Path) -> None:
    _discover()
    valid = tmp_path / "valid.drowse"
    bad = tmp_path / "bad.drowse"
    export_drowse_archive("local", "demo", output=valid)
    entries = _archive_entries(valid)
    payload = next(name for name in entries if name.endswith("00_calm.json"))
    original = entries[payload]
    entries[payload] = bytes([original[0] ^ 1]) + original[1:]
    _write_entries(bad, entries)

    shutil.rmtree(manifold_dir("local", "demo"))
    with pytest.raises(DrowseArchiveError, match="SHA-256"):
        install_drowse_archive(bad)


@pytest.mark.parametrize("as_", [None, "shared/renamed"])
def test_rejects_primary_identity_mismatch_before_optional_rewrite(
    tmp_path: Path, as_: str | None,
) -> None:
    _discover()
    valid = tmp_path / "valid.drowse"
    bad = tmp_path / "bad.drowse"
    export_drowse_archive("local", "demo", output=valid)

    def mutate(manifest: dict[str, Any], entries: dict[str, bytes]) -> None:
        path = "manifolds/local/demo/manifold.json"
        payload = json.loads(entries[path])
        payload["name"] = "other"
        _replace_manifested_json(manifest, entries, path, payload)

    _rewrite_manifest(valid, bad, mutate)
    shutil.rmtree(manifold_dir("local", "demo"))
    with pytest.raises(DrowseArchiveError, match="contains manifold identity"):
        install_drowse_archive(bad, as_=as_)


def test_rejects_template_path_identity_mismatch(tmp_path: Path) -> None:
    folder, template = _template_manifold()
    valid = tmp_path / "valid.drowse"
    bad = tmp_path / "bad.drowse"
    export_drowse_archive("local", "days", output=valid)

    def mutate(manifest: dict[str, Any], entries: dict[str, bytes]) -> None:
        path = "templates/local/weekday/template.json"
        payload = json.loads(entries[path])
        payload["name"] = "weekend"
        _replace_manifested_json(manifest, entries, path, payload)

    _rewrite_manifest(valid, bad, mutate)
    shutil.rmtree(folder)
    shutil.rmtree(template)
    with pytest.raises(DrowseArchiveError, match="contains identity 'weekend'"):
        install_drowse_archive(bad)


def test_rejects_pack_source_that_contradicts_manifold(tmp_path: Path) -> None:
    folder = _discover()
    valid = tmp_path / "valid.drowse"
    bad = tmp_path / "bad.drowse"
    export_drowse_archive("local", "demo", output=valid)

    def mutate(manifest: dict[str, Any], _entries: dict[str, bytes]) -> None:
        manifest["source"] = {
            "uri": "bundled", "repository": None, "revision": None,
        }

    _rewrite_manifest(valid, bad, mutate)
    shutil.rmtree(folder)
    with pytest.raises(DrowseArchiveError, match="does not match manifold.json"):
        install_drowse_archive(bad)


def test_rejects_unpinned_hugging_face_provenance(tmp_path: Path) -> None:
    folder = _discover()
    valid = tmp_path / "valid.drowse"
    bad = tmp_path / "bad.drowse"
    export_drowse_archive("local", "demo", output=valid)

    def mutate(manifest: dict[str, Any], _entries: dict[str, bytes]) -> None:
        manifest["source"] = {
            "uri": "hf://owner/repo",
            "repository": "owner/repo",
            "revision": None,
        }

    _rewrite_manifest(valid, bad, mutate)
    shutil.rmtree(folder)
    with pytest.raises(DrowseArchiveError, match="immutable revision"):
        install_drowse_archive(bad)


def test_rejects_mutable_hugging_face_revision_provenance(tmp_path: Path) -> None:
    _discover()
    valid = tmp_path / "valid.drowse"
    bad = tmp_path / "bad.drowse"
    export_drowse_archive("local", "demo", output=valid)

    def mutate(manifest: dict[str, Any], entries: dict[str, bytes]) -> None:
        source = {
            "uri": "hf://owner/repo@main",
            "repository": "owner/repo",
            "revision": "main",
        }
        manifest["source"] = source
        path = "manifolds/local/demo/manifold.json"
        payload = json.loads(entries[path])
        payload["source"] = source["uri"]
        _replace_manifested_json(manifest, entries, path, payload)

    _rewrite_manifest(valid, bad, mutate)

    with pytest.raises(DrowseArchiveError, match="commit SHA"):
        install_drowse_archive(bad)


def test_accepts_immutable_hugging_face_commit_provenance(tmp_path: Path) -> None:
    folder = _discover()
    commit = "a" * 40
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"] = f"hf://owner/repo@{commit}"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "demo.drowse"

    export_drowse_archive("local", "demo", output=output)

    with zipfile.ZipFile(output) as archive:
        pack = json.loads(archive.read("pack.json"))
    assert pack["source"] == {
        "uri": f"hf://owner/repo@{commit}",
        "repository": "owner/repo",
        "revision": commit,
    }


def test_rejects_unknown_root_even_when_manifested(tmp_path: Path) -> None:
    _discover()
    valid = tmp_path / "valid.drowse"
    bad = tmp_path / "bad.drowse"
    export_drowse_archive("local", "demo", output=valid)

    def mutate(manifest: dict[str, Any], entries: dict[str, bytes]) -> None:
        content = b"nope"
        entries["models/weights.bin"] = content
        import hashlib

        manifest["files"]["models/weights.bin"] = {
            "size": len(content), "sha256": hashlib.sha256(content).hexdigest(),
        }

    _rewrite_manifest(valid, bad, mutate)
    shutil.rmtree(manifold_dir("local", "demo"))
    with pytest.raises(DrowseArchiveError, match="outside its closure"):
        install_drowse_archive(bad)


def test_rejects_missing_template_closure(tmp_path: Path) -> None:
    folder, _template = _template_manifold()
    valid = tmp_path / "valid.drowse"
    bad = tmp_path / "bad.drowse"
    export_drowse_archive("local", "days", output=valid)

    def mutate(manifest: dict[str, Any], entries: dict[str, bytes]) -> None:
        template_path = "templates/local/weekday/template.json"
        entries.pop(template_path)
        manifest["files"].pop(template_path)
        manifest["template"] = None

    _rewrite_manifest(valid, bad, mutate)
    shutil.rmtree(folder)
    with pytest.raises(DrowseArchiveError, match="missing its template closure"):
        install_drowse_archive(bad)


def test_promote_transaction_rolls_back_both_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "a" * 32
    target_a = tmp_path / "manifolds" / "local" / "a"
    target_b = tmp_path / "templates" / "local" / "b"
    stage_a = _transaction_stage_path(tmp_path, token, "manifold")
    stage_b = _transaction_stage_path(tmp_path, token, "template")
    for path, value in ((target_a, "old-a"), (target_b, "old-b")):
        path.mkdir(parents=True)
        (path / "value").write_text(value)
    _begin_transaction(
        [(stage_a, target_a), (stage_b, target_b)], token=token, root=tmp_path,
    )
    for path, value in ((stage_a, "new-a"), (stage_b, "new-b")):
        path.mkdir(parents=True)
        (path / "value").write_text(value)
    original_rename = Path.rename

    def fail_second_stage(self: Path, target: Path) -> Path:
        if self == stage_b and target == target_b:
            raise OSError("injected")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_second_stage)
    with pytest.raises(DrowseArchiveError, match="promote"):
        _promote_transaction(
            [(stage_a, target_a), (stage_b, target_b)],
            token=token,
            journal_root=tmp_path,
            transaction_locked=True,
        )
    assert (target_a / "value").read_text() == "old-a"
    assert (target_b / "value").read_text() == "old-b"
    assert not list((tmp_path / ".drowse-transactions").glob("*.json"))


def test_promote_transaction_rolls_back_python_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "b" * 32
    target = tmp_path / "manifolds" / "local" / "a"
    stage = _transaction_stage_path(tmp_path, token, "manifold")
    target.mkdir(parents=True)
    (target / "value").write_text("old")
    _begin_transaction([(stage, target)], token=token, root=tmp_path)
    stage.mkdir(parents=True)
    (stage / "value").write_text("new")
    original_rename = Path.rename

    def fail_stage(self: Path, destination: Path) -> Path:
        if self == stage and destination == target:
            raise RuntimeError("injected")
        return original_rename(self, destination)

    monkeypatch.setattr(Path, "rename", fail_stage)
    with pytest.raises(RuntimeError, match="injected"):
        _promote_transaction(
            [(stage, target)], token=token, journal_root=tmp_path,
            transaction_locked=True,
        )
    assert (target / "value").read_text() == "old"
    assert not list((tmp_path / ".drowse-transactions").glob("*.json"))


def test_recover_crash_left_two_target_transaction(tmp_path: Path) -> None:
    token = "c" * 32
    target_a = tmp_path / "manifolds" / "local" / "a"
    target_b = tmp_path / "templates" / "local" / "b"
    stage_a = _transaction_stage_path(tmp_path, token, "manifold")
    stage_b = _transaction_stage_path(tmp_path, token, "template")
    backup_a = _transaction_backup_path(tmp_path, token, "manifold")
    backup_b = _transaction_backup_path(tmp_path, token, "template")
    for path, value in (
        (target_a, "old-a"), (target_b, "old-b"),
        (stage_a, "new-a"), (stage_b, "new-b"),
    ):
        path.mkdir(parents=True)
        (path / "value").write_text(value)
    target_a.rename(backup_a)
    target_b.rename(backup_b)
    stage_a.rename(target_a)
    journal_dir = tmp_path / ".drowse-transactions"
    journal_dir.mkdir(exist_ok=True)
    journal = {
        "format_version": 1,
        "token": token,
        "phase": "backed_up",
        "items": [
            {
                "kind": "manifold",
                "stage": stage_a.relative_to(tmp_path).as_posix(),
                "target": target_a.relative_to(tmp_path).as_posix(),
                "backup": backup_a.relative_to(tmp_path).as_posix(),
                "had_existing": True,
            },
            {
                "kind": "template",
                "stage": stage_b.relative_to(tmp_path).as_posix(),
                "target": target_b.relative_to(tmp_path).as_posix(),
                "backup": backup_b.relative_to(tmp_path).as_posix(),
                "had_existing": True,
            },
        ],
    }
    (journal_dir / f"{token}.json").write_text(json.dumps(journal))

    recover_drowse_archive_transactions(tmp_path)

    assert (target_a / "value").read_text() == "old-a"
    assert (target_b / "value").read_text() == "old-b"
    assert not stage_a.exists()
    assert not stage_b.exists()
    assert not backup_a.exists()
    assert not backup_b.exists()
    assert not (journal_dir / f"{token}.json").exists()


def test_staging_journal_precedes_copy_and_stage_is_not_discoverable() -> None:
    source = _discover()
    root = drowse_home()
    target = manifold_dir("local", "installed")
    token = "d" * 32
    stage = _transaction_stage_path(root, token, "manifold")

    journal = _begin_transaction([(stage, target)], token=token, root=root)
    assert journal.is_file()
    shutil.copytree(source, stage)
    assert stage.parent == root / ".drowse-transactions"
    assert [name for name, _folder in iter_manifold_folders()] == ["local"]

    recover_drowse_archive_transactions(root)
    assert not stage.exists()
    assert not journal.exists()
    assert not target.exists()


def test_install_cleans_stage_when_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drowse.io.drowse_archive as module

    folder = _discover()
    archive = tmp_path / "demo.drowse"
    export_drowse_archive("local", "demo", output=archive)
    shutil.rmtree(folder)
    original_copy = module._copy_tree

    def fail_after_copy(source: Path, destination: Path) -> None:
        original_copy(source, destination)
        raise RuntimeError("injected copy failure")

    monkeypatch.setattr(module, "_copy_tree", fail_after_copy)
    with pytest.raises(RuntimeError, match="copy failure"):
        install_drowse_archive(archive)

    transactions = drowse_home() / ".drowse-transactions"
    assert not list(transactions.glob("*.json"))
    assert not list(transactions.glob("*.stage"))
    assert not manifold_dir("local", "demo").exists()


def test_recovery_sweeps_legacy_stage_before_discovery() -> None:
    source = _discover()
    token = "e" * 32
    legacy = source.with_name(f"demo.drowse-stage-{token}")
    shutil.copytree(source, legacy)
    assert len(list(iter_manifold_folders())) == 2

    recover_drowse_archive_transactions()

    assert not legacy.exists()
    assert len(list(iter_manifold_folders())) == 1


@pytest.mark.parametrize(
    "relative",
    [Path(".drowse-transactions"), Path("manifolds"), Path("templates")],
)
def test_recovery_rejects_symlinked_owned_roots(
    tmp_path: Path,
    relative: Path,
) -> None:
    root = tmp_path / "root"
    external = tmp_path / "external"
    root.mkdir()
    external.mkdir()
    marker = external / "keep"
    marker.write_text("external")
    (root / relative).parent.mkdir(parents=True, exist_ok=True)
    (root / relative).symlink_to(external, target_is_directory=True)

    with pytest.raises(DrowseArchiveError, match="symbolic link"):
        recover_drowse_archive_transactions(root)

    assert marker.read_text() == "external"


def test_bootstrap_recovers_pending_staging_transaction() -> None:
    from drowse.io.bootstrap import materialize_bundled_artifacts

    source = _discover()
    root = drowse_home()
    target = manifold_dir("local", "installed")
    token = "f" * 32
    stage = _transaction_stage_path(root, token, "manifold")
    journal = _begin_transaction([(stage, target)], token=token, root=root)
    shutil.copytree(source, stage)

    materialize_bundled_artifacts()

    assert not stage.exists()
    assert not journal.exists()


def test_cli_export_drowse_archive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from drowse.cli.main import main

    _discover()
    output = tmp_path / "cli.drowse"
    main(["pack", "export", "archive", "local/demo", "-o", str(output)])
    assert output.is_file()
    assert f"Wrote {output}" in capsys.readouterr().out


def test_install_manifold_routes_local_archive(tmp_path: Path) -> None:
    from drowse.io.hf_manifolds import install_manifold

    folder = _discover()
    output = tmp_path / "demo.drowse"
    export_drowse_archive("local", "demo", output=output)
    shutil.rmtree(folder)
    assert install_manifold(str(output)) == manifold_dir("local", "demo")


def test_install_manifold_rejects_missing_local_archive(tmp_path: Path) -> None:
    from drowse.io.hf_manifolds import install_manifold

    missing = tmp_path / "missing.drowse"
    with pytest.raises(FileNotFoundError, match="Drowse archive not found"):
        install_manifold(str(missing))


@pytest.mark.parametrize("coordinate", [True, float("nan"), float("inf")])
def test_authored_coordinates_must_be_finite_non_boolean(coordinate: object) -> None:
    domain = {
        "type": "box",
        "axes": [{"name": "x", "periodic": False, "lo": 0.0, "hi": 1.0}],
    }
    nodes = [
        {"label": "a", "coords": [coordinate], "statements": ["a"]},
        {"label": "b", "coords": [0.5], "statements": ["b"]},
        {"label": "c", "coords": [1.0], "statements": ["c"]},
    ]
    with pytest.raises(ManifoldFormatError, match="needs 'coords'"):
        create_manifold_folder("local", "invalid", "", domain, nodes)


@pytest.mark.parametrize("coordinate", [True, float("nan"), float("inf")])
def test_manifold_loader_rejects_nonfinite_or_boolean_coords(
    coordinate: object,
) -> None:
    domain = {
        "type": "box",
        "axes": [{"name": "x", "periodic": False, "lo": 0.0, "hi": 1.0}],
    }
    nodes = [
        {"label": "a", "coords": [0.0], "statements": ["a"]},
        {"label": "b", "coords": [0.5], "statements": ["b"]},
        {"label": "c", "coords": [1.0], "statements": ["c"]},
    ]
    folder, _ = create_manifold_folder("local", "invalid", "", domain, nodes)
    manifest_path = folder / "manifold.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["nodes"][0]["coords"] = [coordinate]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ManifoldFormatError, match="needs 'coords'"):
        ManifoldFolder.load(folder, verify_manifest=False)
