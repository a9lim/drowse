"""The one install pipeline, exercised over every source.

``local`` installs a manifold folder, ``archive`` a ``.drowse``, and ``hf`` a
fake Hub repo whose single root archive is the same ``.drowse``.  Every source
must land the same validated closure under the same destination policy.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from drowse.base import staging
from drowse.base.paths import drowse_home, manifold_dir, tensor_filename
from drowse.store import hub
from drowse.store import install as install_mod
from drowse.store.archive import DrowseArchiveError, export_drowse_archive
from drowse.store.install import ManifoldInstallConflict, install_manifold
from drowse.store.manifold_authoring import (
    create_discover_manifold_folder,
    iter_manifold_folders,
)
from drowse.store.manifold_folder import ManifoldFolder

MODEL = "google/gemma-2-2b-it"
SHA = "0123456789abcdef0123456789abcdef01234567"
SOURCES = ("local", "archive", "hf")


def write_fitted_manifold(
    folder: Path, model_id: str, *, publish_manifest: bool = True,
) -> Path:
    """Write one identity-consistent fitted 1-D affine pair into ``folder``."""
    import torch

    from drowse.geometry.domains import CustomDomain
    from drowse.geometry.manifold import MANIFOLD_FIT_POLICY_VERSION, Manifold
    from drowse.geometry.subspace import LayerSubspace
    from drowse.store.manifold_tensors import save_manifold

    mf = ManifoldFolder.load(folder, verify_manifest=False)
    node_coords = torch.tensor([[-1.0], [1.0]])
    manifold = Manifold(
        name=mf.name,
        domain=CustomDomain(1),
        node_labels=list(mf.node_labels),
        node_coords=node_coords,
        layers={
            0: LayerSubspace.affine(
                mean=torch.zeros(2), basis=torch.tensor([[1.0, 0.0]]),
                node_coords=node_coords.clone(),
            ),
        },
        feature_space="raw",
        mahalanobis_share={0: 1.0},
    )
    path = folder / tensor_filename(model_id)
    save_manifold(manifold, path, {
        "method": "manifold_discover_pca",
        "fit_mode": "pca",
        "hyperparams": dict(mf.hyperparams),
        "nodes_sha256": mf.nodes_sha256(),
        "model_fingerprint": f"fp:{model_id}",
        "fit_policy_version": MANIFOLD_FIT_POLICY_VERSION,
    })
    if publish_manifest:
        mf.update_file_hashes(path, path.with_suffix(".json"))
    return path


def author_manifold(name: str = "mood", *, fitted: bool = True) -> Path:
    folder = create_discover_manifold_folder(
        "local", name, "a mood axis", fit_mode="pca",
        node_corpora={"calm": ["easy"], "afraid": ["tense"]},
        hyperparams={"max_dim": 1},
    )
    if fitted:
        write_fitted_manifold(folder, MODEL)
    return folder


class FakeHub:
    """A Hub serving one repo whose single root file is an exported archive."""

    def __init__(self, archive: Path, *, tags: tuple[str, ...] = ("drowse-manifold",),
                 extra: tuple[str, ...] = ()) -> None:
        self.archive = archive
        self.tags = list(tags)
        self.files = [archive.name, "manifold.json", "README.md", *extra]
        self.downloads: list[tuple[str, str, str]] = []
        self.revisions: list[str | None] = []

    def model_info(self, coord: str, revision: str | None = None) -> Any:
        self.revisions.append(revision)
        return SimpleNamespace(
            id=coord, sha=SHA, tags=self.tags,
            siblings=[SimpleNamespace(rfilename=name) for name in self.files],
        )

    def download(self, coord: str, filename: str, *, revision: str) -> str:
        self.downloads.append((coord, filename, revision))
        return str(self.archive)


@pytest.fixture
def homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Author under one home, install into another."""
    author = tmp_path / "author"
    target = tmp_path / "home"
    monkeypatch.setenv("DROWSE_HOME", str(author))
    return {"author": author, "home": target, "tmp": tmp_path}


def _use_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    from drowse.store import selectors

    monkeypatch.setenv("DROWSE_HOME", str(home))
    selectors.invalidate()


def stage_source(
    kind: str, folder: Path, homes: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, FakeHub | None]:
    """Prepare ``folder`` (in the author home) as an install target of ``kind``."""
    if kind == "local":
        snapshot = homes["tmp"] / "src" / folder.name
        shutil.copytree(folder, snapshot)
        _use_home(monkeypatch, homes["home"])
        return str(snapshot), None
    archive = homes["tmp"] / "dist" / f"{folder.name}.drowse"
    export_drowse_archive(folder.parent.name, folder.name, output=archive)
    _use_home(monkeypatch, homes["home"])
    if kind == "archive":
        return str(archive), None
    fake = FakeHub(archive)
    monkeypatch.setattr(hub, "_hf_api", lambda: fake)
    monkeypatch.setattr(hub, "_hf_hub_download", fake.download)
    return f"alice/{folder.name}", fake


# --------------------------------------------------------------- sources --


@pytest.mark.parametrize("kind", SOURCES)
def test_every_source_installs_the_same_validated_closure(
    kind: str, homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.store.manifold_tensors import load_manifold

    target, fake = stage_source(kind, author_manifold(), homes, monkeypatch)
    installed = install_manifold(target)

    assert installed == manifold_dir("local", "mood")
    folder = ManifoldFolder.load(installed)
    assert folder.source == ("hf://alice/mood@" + SHA if kind == "hf" else "local")
    assert load_manifold(installed / tensor_filename(MODEL)) is not None
    assert [mf.name for _ns, mf in iter_manifold_folders()] == ["mood"]
    assert not any(staging.staging_root().glob("*/manifolds"))
    if fake is not None:
        assert fake.downloads == [("alice/mood", "mood.drowse", SHA)]


@pytest.mark.parametrize("kind", SOURCES)
def test_as_renames_manifest_and_every_fitted_sidecar(
    kind: str, homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.store.manifold_tensors import load_manifold

    target, _ = stage_source(kind, author_manifold(), homes, monkeypatch)
    installed = install_manifold(target, "team/calm")

    assert installed == manifold_dir("team", "calm")
    assert ManifoldFolder.load(installed).name == "calm"
    sidecar = json.loads((installed / tensor_filename(MODEL)).with_suffix(".json").read_text())
    assert sidecar["name"] == "calm"
    assert load_manifold(installed / tensor_filename(MODEL)).name == "calm"


@pytest.mark.parametrize("kind", SOURCES)
def test_existing_destination_needs_force(
    kind: str, homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _ = stage_source(kind, author_manifold(), homes, monkeypatch)
    installed = install_manifold(target)
    marker = installed / "nodes"
    with pytest.raises(ManifoldInstallConflict, match="already exists"):
        install_manifold(target)
    assert marker.is_dir()
    assert install_manifold(target, force=True) == installed
    ManifoldFolder.load(installed)


@pytest.mark.parametrize(
    "as_",
    ["jlens/target", "sae/target", "default/target", "../target",
     "/target", "local\n/target", "local/target\n", "bad"],
)
@pytest.mark.parametrize("kind", ("archive", "hf"))
def test_bad_or_reserved_destinations_are_refused_before_writing(
    kind: str, as_: str, homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, fake = stage_source(kind, author_manifold(), homes, monkeypatch)
    with pytest.raises(ValueError):
        install_manifold(target, as_)
    if fake is not None:
        assert fake.downloads == []  # refused before any network traffic
    assert not (homes["home"] / "manifolds").exists() or not any(
        (homes["home"] / "manifolds").rglob("manifold.json")
    )


def test_install_rejects_escaping_namespace_link(
    homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, _ = stage_source("archive", author_manifold(), homes, monkeypatch)
    outside = homes["tmp"] / "outside"
    outside.mkdir()
    (homes["home"] / "manifolds").mkdir(parents=True)
    (homes["home"] / "manifolds" / "local").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the manifold directory"):
        install_manifold(target)
    assert list(outside.iterdir()) == []


# ------------------------------------------------------------ local source --


@pytest.mark.parametrize("kind", ["file", "directory", "dangling"])
def test_local_source_links_are_refused_before_copying(
    kind: str, homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = author_manifold(fitted=False)
    external = homes["tmp"] / "private"
    if kind == "file":
        external.write_text("private")
        (folder / "nodes" / "extra.json").symlink_to(external)
    elif kind == "directory":
        external.mkdir()
        (folder / "linked").symlink_to(external, target_is_directory=True)
    else:
        (folder / "nodes" / "dangling.json").symlink_to(homes["tmp"] / "missing")
    _use_home(monkeypatch, homes["home"])
    with pytest.raises(ValueError, match="symbolic link"):
        install_manifold(str(folder))
    assert not manifold_dir("local", "mood").exists()


def test_local_source_special_file_is_refused(
    homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = author_manifold(fitted=False)
    os.mkfifo(folder / "pipe")
    _use_home(monkeypatch, homes["home"])
    with pytest.raises(ValueError, match="not a regular file"):
        install_manifold(str(folder))


def test_local_source_unmanifested_pair_is_refused(
    homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = author_manifold(fitted=False)
    write_fitted_manifold(folder, MODEL, publish_manifest=False)
    _use_home(monkeypatch, homes["home"])
    with pytest.raises(DrowseArchiveError, match="unproved or noncanonical"):
        install_manifold(str(folder))
    assert not manifold_dir("local", "mood").exists()


def test_local_install_onto_itself_needs_force_and_keeps_the_only_copy(
    homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = author_manifold()
    with pytest.raises(ManifoldInstallConflict):
        install_manifold(str(folder))
    assert install_manifold(str(folder), force=True) == folder
    ManifoldFolder.load(folder)


def test_local_folder_only_carries_its_closure(
    homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = author_manifold()
    (folder / "notes.txt").write_text("scratch")
    (folder / "export.gguf").write_bytes(b"gguf")
    _use_home(monkeypatch, homes["home"])
    installed = install_manifold(str(folder))
    assert not (installed / "notes.txt").exists()
    assert not (installed / "export.gguf").exists()


# ----------------------------------------------- staging / crash recovery --


def test_crash_between_swap_renames_is_invisible_and_recovered(
    homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a crash-left stage or backup used to sit beside the live
    folder (``<name>.staging`` / ``<name>.bak``) and be discovered as a
    duplicate manifold.  Both now live under ``$DROWSE_HOME/.staging``."""
    target, _ = stage_source("archive", author_manifold(), homes, monkeypatch)
    installed = install_manifold(target)
    real_promote = staging.promote

    def crash_after_backup(stage: Path, destination: Path) -> None:
        backup = staging.backup_path(destination)
        backup.parent.mkdir(parents=True, exist_ok=True)
        destination.rename(backup)
        raise KeyboardInterrupt("crash between the two renames")

    monkeypatch.setattr(install_mod, "promote", crash_after_backup)
    with pytest.raises(KeyboardInterrupt):
        install_manifold(target, force=True)
    monkeypatch.setattr(install_mod, "promote", real_promote)

    assert not installed.exists()
    assert list(iter_manifold_folders()) == []
    with pytest.raises(ManifoldInstallConflict, match="already exists"):
        install_manifold(target)  # recovers the backup first
    assert ManifoldFolder.load(installed).name == "mood"
    assert [mf.name for _ns, mf in iter_manifold_folders()] == ["mood"]


def test_failed_copy_preserves_the_installed_folder(
    homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder = author_manifold()
    _use_home(monkeypatch, homes["home"])
    installed = install_manifold(str(folder))
    before = sorted(p.name for p in installed.iterdir())

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(install_mod.shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="disk full"):
        install_manifold(str(folder), force=True)
    assert sorted(p.name for p in installed.iterdir()) == before
    ManifoldFolder.load(installed)


def test_force_install_waits_for_a_held_pair_lock(
    homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.store.manifold_folder import manifold_pair_lock

    target, _ = stage_source("archive", author_manifold(), homes, monkeypatch)
    installed = install_manifold(target)
    tensor = installed / tensor_filename(MODEL)
    done = threading.Event()
    errors: list[BaseException] = []

    def reinstall() -> None:
        try:
            install_manifold(target, force=True)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            done.set()

    with manifold_pair_lock(tensor):
        worker = threading.Thread(target=reinstall)
        worker.start()
        assert not done.wait(0.3)
        assert tensor.is_file()
    worker.join(timeout=10)
    assert done.is_set() and errors == []


# --------------------------------------------------------------------- hf --


@pytest.mark.parametrize("kind", SOURCES)
def test_templated_manifold_installs_with_its_template_closure(
    kind: str, homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: an HF pull copied only the manifold folder, leaving a
    dangling ``template_ref`` whose every fit then read as stale."""
    from drowse.store.manifold_authoring import create_manifold_from_template
    from drowse.store.templates import create_template_folder, resolve_template

    create_template_folder(
        "local", "weekday", slot="[DAY]", values=["Monday", "Tuesday"],
        contexts=[{
            "turns": [{"role": "user", "content": "what day is it?"}],
            "assistant": "today is [DAY]",
        }],
    )
    folder = create_manifold_from_template(
        "local", "days", "days", template_ref="local/weekday", fit_mode="pca",
    )
    target, _ = stage_source(kind, folder, homes, monkeypatch)
    if kind == "local":
        # A bare folder carries no template: it must already be installed.
        from drowse.store.templates import TemplateNotFoundError

        with pytest.raises(TemplateNotFoundError):
            install_manifold(target)
        return
    installed = install_manifold(target)
    assert ManifoldFolder.load(installed).template_ref == "local/weekday"
    assert resolve_template("local/weekday").node_labels() == ["monday", "tuesday"]


def test_hf_install_pins_the_commit_and_exports_back(
    homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: an HF install stamped ``hf://repo`` (or ``@main``), which
    ``.drowse`` export rejects as mutable provenance."""
    target, fake = stage_source("hf", author_manifold(), homes, monkeypatch)
    install_manifold(f"{target}@main")
    assert fake is not None and fake.revisions == ["main"]
    assert ManifoldFolder.load(manifold_dir("local", "mood")).source == (
        f"hf://alice/mood@{SHA}"
    )
    exported = export_drowse_archive(
        "local", "mood", output=homes["tmp"] / "again.drowse",
    )
    assert exported.is_file()


@pytest.mark.parametrize(
    ("tags", "extra", "match"),
    [
        (("other",), (), "not tagged"),
        (("drowse-manifold",), ("second.drowse",), "exactly one root"),
    ],
)
def test_hf_repo_must_follow_the_archive_convention(
    tags: tuple[str, ...], extra: tuple[str, ...], match: str,
    homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drowse.base.hf import HFError

    archive = homes["tmp"] / "mood.drowse"
    archive.write_bytes(b"unused")
    fake = FakeHub(archive, tags=tags, extra=extra)
    monkeypatch.setattr(hub, "_hf_api", lambda: fake)
    monkeypatch.setattr(hub, "_hf_hub_download", fake.download)
    _use_home(monkeypatch, homes["home"])
    with pytest.raises(HFError, match=match):
        install_manifold("alice/mood")
    assert fake.downloads == []


def test_missing_archive_and_bare_name_are_rejected(
    homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_home(monkeypatch, homes["home"])
    with pytest.raises(FileNotFoundError, match="Drowse archive not found"):
        install_manifold(str(homes["tmp"] / "missing.drowse"))
    with pytest.raises(ValueError, match="owner"):
        install_manifold("not-a-coord")


def test_staging_root_is_outside_every_artifact_root(
    homes: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_home(monkeypatch, homes["home"])
    root = staging.staging_root()
    assert root.parent == drowse_home()
    assert root.name not in {"manifolds", "templates", "models"}
