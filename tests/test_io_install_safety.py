import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from drowse.store import hub as hfm
from drowse.store.manifolds import ManifoldFolder, create_discover_manifold_folder


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path / "author"))
    folder = create_discover_manifold_folder(
        "local", "source", "", fit_mode="pca",
        node_corpora={"positive": ["yes"], "negative": ["no"]},
    )
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path / "home"))
    return folder


@pytest.mark.parametrize("kind", ["file", "directory", "dangling", "cycle"])
def test_local_install_rejects_source_links_before_copying(source: Path, tmp_path: Path, kind: str):
    external = tmp_path / "private"
    if kind == "file":
        external.write_text("private data")
        (source / "extra.json").symlink_to(external)
    elif kind == "directory":
        external.mkdir()
        (external / "secret").write_text("private data")
        (source / "extra").symlink_to(external, target_is_directory=True)
    elif kind == "dangling":
        (source / "extra.json").symlink_to(external)
    else:
        (source / "extra").symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        hfm.install_manifold(str(source), as_="local/target")

    assert not (tmp_path / "home" / "manifolds" / "local" / "target").exists()


@pytest.mark.parametrize("suffix", [".staging", ".bak"])
def test_local_install_preserves_source_nested_in_reserved_sibling(source: Path, tmp_path: Path, suffix: str):
    target = tmp_path / "home" / "manifolds" / "local" / "target"
    reserved = target.with_name("target" + suffix)
    nested_source = reserved / "nested" / "source"
    nested_source.parent.mkdir(parents=True)
    shutil.copytree(source, nested_source)

    assert hfm.install_manifold(str(nested_source), as_="local/target", force=True) == target
    assert ManifoldFolder.load(target).node_labels == ["positive", "negative"]


def test_local_install_rejects_destination_inside_source(source: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DROWSE_HOME", str(source / "cache"))

    def reject_copy(*args: Any, **kwargs: Any):
        pytest.fail("recursive source-to-descendant copy reached copytree")

    monkeypatch.setattr(hfm.shutil, "copytree", reject_copy)
    with pytest.raises(ValueError, match="inside the source"):
        hfm.install_manifold(str(source), as_="local/target")
    assert not (source / "cache").exists()


def test_local_install_rejects_special_file_before_loading(source: Path, tmp_path: Path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable")
    (source / "manifold.json").unlink()
    os.mkfifo(source / "manifold.json")
    with pytest.raises(ValueError, match="not a regular file"):
        hfm.install_manifold(str(source), as_="local/target")
    assert not (tmp_path / "home").exists()


@pytest.mark.parametrize("target_kind", ["local", "hf"])
def test_install_rejects_escaping_namespace_link(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_kind: str,
):
    outside = tmp_path / "private"
    outside.mkdir()
    target = outside / "target"
    target.mkdir()
    (target / "keep").write_text("untouched")
    manifold_root = tmp_path / "home" / "manifolds"
    manifold_root.mkdir(parents=True)
    (manifold_root / "local").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(hfm, "_hf_snapshot_download", lambda **kwargs: pytest.fail("unexpected network"))
    with pytest.raises(ValueError, match="outside the manifold directory"):
        hfm.install_manifold(str(source) if target_kind == "local" else "owner/source", as_="local/target", force=True)
    assert (target / "keep").read_text() == "untouched"
    assert list(outside.iterdir()) == [target]


@pytest.mark.parametrize("target_kind", ["local", "hf"])
@pytest.mark.parametrize("as_", ["/target", "../target", "local/target\n", "local\n/target", "jlens/target", "sae/target"])
def test_install_rejects_invalid_or_reserved_namespace(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target_kind: str, as_: str,
):
    def reject_download(*args: Any, **kwargs: Any):
        pytest.fail("invalid destination reached the network")

    monkeypatch.setattr(hfm, "_hf_snapshot_download", reject_download)
    target = str(source) if target_kind == "local" else "owner/source"
    with pytest.raises(ValueError):
        hfm.install_manifold(target, as_=as_)
    assert not (tmp_path / "home").exists()
