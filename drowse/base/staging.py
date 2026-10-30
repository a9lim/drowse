"""Crash-recoverable staging for whole-directory artifact installs.

Every install builds its new tree in a private scratch area under
``$DROWSE_HOME/.staging/`` — never beside the live artifact, so a crash-left
stage or backup is invisible to discovery — and promotes it with two renames:
``target -> backup`` then ``stage -> target``.  The backup path is
deterministic per target (``.staging/backup/<target relative to home>``), so a
crash between the two renames is recovered by :func:`recover_backup` before the
next install of that target decides anything.  Callers hold the target's
artifact lock across recovery, policy, and promotion.
"""

from __future__ import annotations

import os
import shutil
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Iterator

from drowse.base.atomic import fsync_directory
from drowse.base.paths import drowse_home


def staging_root() -> Path:
    """The scratch root shared by every install (outside every artifact root)."""
    return drowse_home() / ".staging"


@contextmanager
def staging_area() -> Iterator[Path]:
    """A fresh private scratch directory, removed when the install finishes."""
    area = staging_root() / uuid.uuid4().hex
    area.mkdir(parents=True)
    try:
        yield area
    finally:
        shutil.rmtree(area, ignore_errors=True)


def backup_path(target: Path) -> Path:
    """Deterministic swap backup for an artifact directory under the home."""
    home = drowse_home().resolve(strict=False)
    relative = Path(target).resolve(strict=False).relative_to(home)
    return staging_root() / "backup" / relative


def recover_backup(target: Path) -> bool:
    """Restore a crash-left swap backup when its target is missing.

    Returns whether a backup was restored.  A backup beside a live target is a
    completed swap whose cleanup was interrupted; it is discarded.
    """
    backup = backup_path(target)
    if not backup.exists():
        return False
    if Path(target).exists():
        shutil.rmtree(backup)
        return False
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    backup.rename(target)
    fsync_directory(Path(target).parent)
    return True


def fsync_tree(root: Path) -> None:
    """One durability pass over a fully built stage before it is promoted."""
    directories: list[Path] = []
    for current, _subdirectories, filenames in os.walk(root):
        directories.append(Path(current))
        for filename in filenames:
            with open(Path(current) / filename, "rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        fsync_directory(directory)


def promote(stage: Path, target: Path) -> None:
    """Atomically replace ``target`` with the fully built ``stage``.

    The previous target (if any) is parked at :func:`backup_path` for the
    duration of the swap and restored if the second rename fails.
    """
    stage, target = Path(stage), Path(target)
    fsync_tree(stage)
    backup = backup_path(target)
    if backup.exists():
        shutil.rmtree(backup)
    had_existing = target.exists()
    if had_existing:
        backup.parent.mkdir(parents=True, exist_ok=True)
        target.rename(backup)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        stage.rename(target)
    except BaseException:
        if had_existing and not target.exists():
            with suppress(OSError):
                backup.rename(target)
        raise
    fsync_directory(target.parent)
    if had_existing:
        shutil.rmtree(backup, ignore_errors=True)
