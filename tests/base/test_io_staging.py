from pathlib import Path

import pytest

from drowse.base import staging


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    return tmp_path


def _stage(text: str) -> Path:
    stage = staging.staging_root() / "build"
    stage.mkdir(parents=True)
    (stage / "payload.txt").write_text(text)
    return stage


def test_promote_installs_a_new_target(tmp_path: Path) -> None:
    target = tmp_path / "manifolds" / "local" / "happy"
    staging.promote(_stage("new"), target)
    assert (target / "payload.txt").read_text() == "new"
    assert not staging.backup_path(target).exists()


def test_promote_replaces_and_discards_the_backup(tmp_path: Path) -> None:
    target = tmp_path / "manifolds" / "local" / "happy"
    target.mkdir(parents=True)
    (target / "payload.txt").write_text("old")
    staging.promote(_stage("new"), target)
    assert (target / "payload.txt").read_text() == "new"
    assert not staging.backup_path(target).exists()


def test_failed_promotion_restores_the_previous_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "manifolds" / "local" / "happy"
    target.mkdir(parents=True)
    (target / "payload.txt").write_text("old")
    stage = _stage("new")
    real_rename = Path.rename

    def rename(self: Path, destination: Path) -> Path:
        if self == stage:
            raise OSError("injected promotion failure")
        return real_rename(self, destination)

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(OSError, match="promotion failure"):
        staging.promote(stage, target)
    assert (target / "payload.txt").read_text() == "old"


def test_backup_lives_outside_the_artifact_namespace(tmp_path: Path) -> None:
    target = tmp_path / "manifolds" / "local" / "happy"
    backup = staging.backup_path(target)
    assert backup == tmp_path / ".staging" / "backup" / "manifolds" / "local" / "happy"
    assert not backup.is_relative_to(tmp_path / "manifolds")


def test_recover_backup_restores_a_missing_target(tmp_path: Path) -> None:
    target = tmp_path / "manifolds" / "local" / "happy"
    backup = staging.backup_path(target)
    backup.mkdir(parents=True)
    (backup / "payload.txt").write_text("old")
    assert staging.recover_backup(target) is True
    assert (target / "payload.txt").read_text() == "old"
    assert not backup.exists()


def test_recover_backup_discards_a_backup_beside_a_live_target(tmp_path: Path) -> None:
    target = tmp_path / "manifolds" / "local" / "happy"
    target.mkdir(parents=True)
    (target / "payload.txt").write_text("live")
    backup = staging.backup_path(target)
    backup.mkdir(parents=True)
    assert staging.recover_backup(target) is False
    assert (target / "payload.txt").read_text() == "live"
    assert not backup.exists()


def test_staging_area_is_removed_after_use() -> None:
    with staging.staging_area() as area:
        (area / "scratch").write_text("x")
        assert area.parent == staging.staging_root()
    assert not area.exists()
