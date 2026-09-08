"""Private, independent staging for application state writes."""

from __future__ import annotations

import os
import gzip
import json
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from drowse.io import atomic
from drowse.core import loom


def test_loom_import_bounds_main_json_before_parsing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(loom.LoomTree().to_dict()) + " " * 4096)
    monkeypatch.setattr(loom, "_TREE_MAX_LOAD_BYTES", 1024, raising=False)
    with pytest.raises(loom.LoomTreeError, match="import limit"):
        loom.LoomTree.load(path)


def test_loom_import_bounds_decompressed_token_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "tree.json"
    payload = loom.LoomTree().to_dict()
    payload["token_sidecar"] = "tree.tokens.json.gz"
    path.write_text(json.dumps(payload))
    sidecar = tmp_path / "tree.tokens.json.gz"
    expanded = json.dumps({"token_sidecar_format": 2, "nodes": {}}).encode() + b" " * 65536
    sidecar.write_bytes(gzip.compress(expanded))
    assert sidecar.stat().st_size < 1024
    monkeypatch.setattr(loom, "_TOKEN_SIDECAR_MAX_LOAD_BYTES", 1024, raising=False)
    with pytest.raises(loom.LoomTreeError, match="import limit"):
        loom.LoomTree.load(path)


def test_loom_import_can_explicitly_allow_a_larger_trusted_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "tree.json"
    tree = loom.LoomTree()
    tree.add_user_turn("keep this conversation")
    tree.save(path)
    monkeypatch.setattr(loom, "_TREE_MAX_LOAD_BYTES", 64)
    restored = loom.LoomTree.load(path, max_bytes=path.stat().st_size)
    assert restored.messages_for() == tree.messages_for()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_loom_import_rejects_invalid_size_overrides(tmp_path: Path, limit: object) -> None:
    from typing import cast

    with pytest.raises(ValueError, match="positive integer"):
        loom.LoomTree.load(tmp_path / "not-opened.json", max_bytes=cast(int, limit))


@pytest.mark.parametrize("compressed", [b"invalid gzip", gzip.compress(b"{}")[:-5]])
def test_loom_import_reports_corrupt_token_sidecars(tmp_path: Path, compressed: bytes) -> None:
    path = tmp_path / "tree.json"
    payload = loom.LoomTree().to_dict()
    payload["token_sidecar"] = "tree.tokens.json.gz"
    path.write_text(json.dumps(payload))
    (tmp_path / "tree.tokens.json.gz").write_bytes(compressed)
    with pytest.raises(loom.LoomTreeError, match="complete gzip"):
        loom.LoomTree.load(path)


def test_atomic_write_does_not_follow_a_planted_staging_symlink(tmp_path: Path) -> None:
    target = tmp_path / "conversation.json"
    victim = tmp_path / "unrelated.txt"
    victim.write_text("preserve me")
    planted = target.with_suffix(".json.tmp")
    planted.symlink_to(victim)

    atomic.write_json_atomic(target, {"private": "conversation"})

    assert victim.read_text() == "preserve me"
    assert not target.is_symlink()
    assert target.read_text() == '{\n  "private": "conversation"\n}\n'
    assert planted.is_symlink()


@pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")
def test_atomic_state_write_is_private_even_with_permissive_umask(tmp_path: Path) -> None:
    target = tmp_path / "conversation.json"
    previous = os.umask(0)
    try:
        atomic.write_json_atomic(target, {"private": "conversation"})
    finally:
        os.umask(previous)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_simultaneous_atomic_writes_publish_complete_independent_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state.json"
    barrier = threading.Barrier(2)
    replace = atomic.os.replace
    published: list[bytes] = []
    guard = threading.Lock()

    def publish(source: Path, destination: Path) -> None:
        barrier.wait(timeout=3)
        with guard:
            published.append(Path(source).read_bytes())
            replace(source, destination)

    monkeypatch.setattr(atomic.os, "replace", publish)
    payloads = [b"a" * 8192, b"b" * 8192]
    with ThreadPoolExecutor(max_workers=2) as pool:
        writes = [pool.submit(atomic.write_bytes_atomic, target, payload) for payload in payloads]
        for write in writes:
            write.result(timeout=5)
    assert sorted(published) == payloads
    assert target.read_bytes() in payloads
    assert sorted(path.name for path in tmp_path.iterdir()) == ["state.json"]


def test_failed_publication_cleans_staging_and_preserves_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state.json"
    target.write_bytes(b"original")

    def fail_replace(source: Path, destination: Path) -> None:
        raise PermissionError("injected publication failure")

    monkeypatch.setattr(atomic.os, "replace", fail_replace)
    with pytest.raises(PermissionError):
        atomic.write_bytes_atomic(target, b"replacement")
    assert target.read_bytes() == b"original"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["state.json"]


def test_atomic_write_supports_long_valid_destination_names(tmp_path: Path) -> None:
    target = tmp_path / ("x" * 240 + ".json")
    atomic.write_bytes_atomic(target, b"complete")
    assert target.read_bytes() == b"complete"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory sync")
def test_atomic_write_syncs_payload_before_publication_and_directory_afterward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    original_replace = atomic.os.replace

    def sync(fd: int) -> None:
        events.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "payload")

    def publish(source: Path, destination: Path) -> None:
        events.append("publish")
        original_replace(source, destination)

    monkeypatch.setattr(atomic.os, "fsync", sync)
    monkeypatch.setattr(atomic.os, "replace", publish)
    atomic.write_bytes_atomic(tmp_path / "state.json", b"complete")
    assert events == ["payload", "publish", "directory"]


def test_atomic_write_preserves_original_when_payload_sync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state.json"
    target.write_bytes(b"original")

    def fail_sync(fd: int) -> None:
        raise OSError("injected storage failure")

    monkeypatch.setattr(atomic.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="storage failure"):
        atomic.write_bytes_atomic(target, b"replacement")
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]
