"""Crash-recoverable stage-and-swap for whole artifact folders.

An artifact folder ``<root>/<ns>/<name>`` is replaced by building its new
content under ``<root>/.staging/`` and renaming it into place, so readers see
either the old folder or the complete new one.  The staging and backup
directories live outside every namespace directory: a crash-left half-built
tree is never discovered as an installed artifact.  Their names are
deterministic per target so a later attempt recovers or clears them.
"""

from __future__ import annotations

import shutil
from contextlib import suppress
from pathlib import Path
from typing import Callable


ErrorFactory = Callable[[str], Exception]
StageBuilder = Callable[[Path], None]

STAGING_DIRNAME = ".staging"


def staging_paths(target_folder: Path) -> tuple[Path, Path]:
    """``(staging, backup)`` directories for ``<root>/<ns>/<name>``.

    Both live under ``<root>/.staging/`` and are keyed ``<ns>~<name>`` (``~``
    is outside the artifact-name grammar, so keys cannot collide).
    """
    target = Path(target_folder)
    root = target.parent.parent / STAGING_DIRNAME
    key = f"{target.parent.name}~{target.name}"
    return root / f"{key}.staging", root / f"{key}.bak"


def recover_backup(target_folder: Path, *, label: str, make_error: ErrorFactory) -> None:
    """Restore a crash-left backup when the destination itself is missing.

    A swap that died between moving the old folder aside and promoting the
    new one leaves only the backup; callers run this before their conflict
    policy so the recovered folder counts as an existing destination.
    """
    _staging, backup = staging_paths(target_folder)
    if Path(target_folder).exists() or not backup.exists():
        return
    try:
        Path(target_folder).parent.mkdir(parents=True, exist_ok=True)
        backup.rename(target_folder)
    except OSError as error:
        raise make_error(f"{label}: could not recover the previous install ({error})") from error


def stage_verify_swap(
    target_folder: Path,
    *,
    force: bool,
    label: str,
    build: StageBuilder,
    make_error: ErrorFactory,
) -> Path:
    """Build ``target_folder`` in staging and atomically promote it.

    The caller owns artifact-specific staging and validation through
    ``build(staging)``.  This helper owns the shared choreography: recover a
    previous backup when the destination is missing, clear stale staging,
    build a fully validated tree, then promote via ``target -> backup`` and
    ``staging -> target`` with best-effort restore on failure.
    """
    target_folder = Path(target_folder)
    staging, backup = staging_paths(target_folder)
    recover_backup(target_folder, label=label, make_error=make_error)

    if target_folder.exists() and not force:
        raise make_error(f"{target_folder} exists; pass force=True to overwrite")

    if staging.exists():
        shutil.rmtree(staging)
    if backup.exists():
        shutil.rmtree(backup)

    staging.mkdir(parents=True, exist_ok=True)
    try:
        build(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    target_folder.parent.mkdir(parents=True, exist_ok=True)
    had_existing = target_folder.exists()
    if had_existing:
        try:
            target_folder.rename(backup)
        except OSError as e:
            shutil.rmtree(staging, ignore_errors=True)
            raise make_error(
                f"{label}: could not move existing install aside ({e})"
            ) from e

    try:
        staging.rename(target_folder)
    except OSError as e:
        if had_existing and backup.exists() and not target_folder.exists():
            with suppress(OSError):
                backup.rename(target_folder)
        shutil.rmtree(staging, ignore_errors=True)
        raise make_error(
            f"{label}: could not promote staging into place ({e})"
        ) from e

    if had_existing:
        shutil.rmtree(backup, ignore_errors=True)

    return target_folder
