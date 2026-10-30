import builtins
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from drowse.base import atomic
def test_copy_file_atomic_is_bounded_exact_and_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source.bin"
    payload = bytes(range(256)) * (1024 * 12 + 17)
    source.write_bytes(payload)
    destination = tmp_path / "nested" / "copy.bin"
    original_open = builtins.open
    reads = []

    class BoundedReader:
        def __enter__(self):
            self.handle = original_open(source, "rb")
            return self

        def __exit__(self, *args: Any):
            self.handle.close()

        def read(self, size: int = -1):
            assert 0 < size <= 1024 * 1024
            reads.append(size)
            return self.handle.read(size)

    def guarded_open(path: Any, mode: str = "r", *args: Any, **kwargs: Any):
        if Path(path) == source and mode == "rb":
            return BoundedReader()
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    atomic.copy_file_atomic(source, destination)
    assert hashlib.sha256(destination.read_bytes()).digest() == hashlib.sha256(payload).digest()
    assert len(reads) >= 4
    assert list(destination.parent.iterdir()) == [destination]
    if os.name != "nt":
        assert destination.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("phase", ["copy", "publish"])
def test_copy_file_atomic_failure_preserves_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str):
    source = tmp_path / "source.bin"
    source.write_bytes(b"replacement")
    destination = tmp_path / "copy.bin"
    destination.write_bytes(b"original")

    def fail(*args: Any, **kwargs: Any):
        raise OSError("injected failure")

    if phase == "copy":
        monkeypatch.setattr(atomic.shutil, "copyfileobj", fail)
    else:
        monkeypatch.setattr(atomic.os, "replace", fail)
    with pytest.raises(OSError, match="injected failure"):
        atomic.copy_file_atomic(source, destination)
    assert destination.read_bytes() == b"original"
    assert set(tmp_path.iterdir()) == {source, destination}
