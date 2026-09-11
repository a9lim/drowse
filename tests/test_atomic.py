"""Atomic-write helper + crash-recovery semantics for ~/.drowse state."""
from __future__ import annotations

import json
import ctypes
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from drowse.io import atomic
from drowse.io.atomic import write_bytes_atomic, write_json_atomic


def test_artifact_lock_registry_releases_inactive_paths(tmp_path: Path):
    for index in range(64):
        with atomic.artifact_lock(tmp_path / f"artifact-{index}"):
            pass
    gc.collect()
    assert not [path for path in atomic._ARTIFACT_LOCKS if path.is_relative_to(tmp_path)]


def test_artifact_lock_waiters_share_the_live_reentrant_lock(tmp_path: Path):
    path = tmp_path / "artifact"
    entered = threading.Event()
    finished = threading.Event()
    order = []

    def waiting_writer():
        entered.set()
        with atomic.artifact_lock(path):
            order.append("waiter")
        finished.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with atomic.artifact_lock(path):
            with atomic.artifact_lock(path):
                future = pool.submit(waiting_writer)
                assert entered.wait(2)
                gc.collect()
                assert not finished.wait(0.05)
                order.append("owner")
        future.result(timeout=2)
    assert order == ["owner", "waiter"]


def test_write_json_atomic_creates_file(tmp_path: Path):
    path = tmp_path / "x.json"
    write_json_atomic(path, {"a": 1, "b": [2, 3]})
    assert path.is_file()
    assert json.loads(path.read_text()) == {"a": 1, "b": [2, 3]}
    # Trailing newline matches the prior json.dump + f.write("\n") convention.
    assert path.read_text().endswith("\n")


def test_write_json_atomic_overwrites(tmp_path: Path):
    path = tmp_path / "x.json"
    write_json_atomic(path, {"v": 1})
    write_json_atomic(path, {"v": 2})
    assert json.loads(path.read_text()) == {"v": 2}


def test_write_json_atomic_no_orphan_tmp(tmp_path: Path):
    path = tmp_path / "x.json"
    write_json_atomic(path, {"v": 1})
    assert list(tmp_path.iterdir()) == [path]


def test_write_json_atomic_creates_parent(tmp_path: Path):
    path = tmp_path / "nested" / "deep" / "x.json"
    write_json_atomic(path, {"v": 1})
    assert path.is_file()


def test_write_bytes_atomic_basic(tmp_path: Path):
    path = tmp_path / "blob.bin"
    write_bytes_atomic(path, b"\x00\x01\x02")
    assert path.read_bytes() == b"\x00\x01\x02"
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_staging_uses_the_destination_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "subdir" / "state.json"
    replace = atomic.os.replace

    def publish(source: Path, destination: Path) -> None:
        assert Path(source).parent == destination.parent
        replace(source, destination)

    monkeypatch.setattr(atomic.os, "replace", publish)
    write_json_atomic(target, {"state": "complete"})
    assert target.is_file()

def test_atomic_overwrite_preserves_prior_on_simulated_crash(tmp_path: Path):
    """If the .tmp file is written but the ``os.replace`` step never lands
    (the canonical crash window), the original file is byte-identical to
    what it was before the write started."""
    path = tmp_path / "x.json"
    write_json_atomic(path, {"version": 1})
    original_bytes = path.read_bytes()

    # Simulate a partial write: stage a new tempfile but don't replace.
    tmp = tmp_path / ".x.json.orphan.tmp"
    tmp.write_text('{"version": 2, "trunc')

    # The "kill" window: tmp exists, original is untouched.
    assert path.read_bytes() == original_bytes
    assert tmp.exists()

    # And the original still loads cleanly.
    assert json.loads(path.read_text()) == {"version": 1}


def test_releasable_artifact_lock_can_reacquire_after_early_release(
    tmp_path: Path,
) -> None:
    transaction = atomic.ReleasableArtifactLock(tmp_path / "capture")

    assert transaction.acquire() is transaction
    transaction.release()
    assert transaction.acquire() is transaction
    transaction.release()
    transaction.release()  # idempotent final cleanup


class _WinCall:
    def __init__(self, result: int) -> None:
        self.result = result
        self.restype = None

    def __call__(self, *_args: object) -> int:
        return self.result


@pytest.mark.parametrize(
    "wait_result, expected", [(0x102, True), (0, False), (0xFFFFFFFF, True)],
)
def test_windows_process_probe_never_uses_os_kill(
    monkeypatch: pytest.MonkeyPatch, wait_result: int, expected: bool,
) -> None:
    class _Kernel:
        OpenProcess = _WinCall(123)
        WaitForSingleObject = _WinCall(wait_result)
        CloseHandle = _WinCall(1)

    class _Windll:
        kernel32 = _Kernel()

    monkeypatch.setattr(ctypes, "windll", _Windll(), raising=False)
    monkeypatch.setattr(atomic.os, "name", "nt")
    monkeypatch.setattr(
        atomic.os, "kill",
        lambda *_: (_ for _ in ()).throw(AssertionError("os.kill is unsafe here")),
    )
    assert atomic._process_exists(987654) is expected


def test_windows_access_denied_process_is_treated_as_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Kernel:
        OpenProcess = _WinCall(0)
        WaitForSingleObject = _WinCall(0)
        CloseHandle = _WinCall(1)
        GetLastError = _WinCall(5)

    class _Windll:
        kernel32 = _Kernel()

    monkeypatch.setattr(ctypes, "windll", _Windll(), raising=False)
    assert atomic._windows_process_exists(42)
