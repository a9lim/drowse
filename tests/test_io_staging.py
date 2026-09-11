from pathlib import Path

import pytest

from drowse.io.staging import stage_verify_swap


class _InstallError(RuntimeError):
    pass


def _err(message: str) -> _InstallError:
    return _InstallError(message)


def test_stage_verify_swap_promotes_valid_staging(tmp_path: Path) -> None:
    target = tmp_path / "packs" / "happy"

    def build(staging: Path) -> None:
        (staging / "payload.txt").write_text("new")

    out = stage_verify_swap(
        target,
        force=False,
        label="user/happy",
        build=build,
        make_error=_err,
    )

    assert out == target
    assert (target / "payload.txt").read_text() == "new"
    assert not target.with_name("happy.staging").exists()
    assert not target.with_name("happy.bak").exists()


def test_stage_verify_swap_build_failure_preserves_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "packs" / "happy"
    target.mkdir(parents=True)
    (target / "payload.txt").write_text("old")

    def build(staging: Path) -> None:
        (staging / "payload.txt").write_text("new")
        raise _InstallError("broken staging")

    with pytest.raises(_InstallError, match="broken staging"):
        stage_verify_swap(
            target,
            force=True,
            label="user/happy",
            build=build,
            make_error=_err,
        )

    assert (target / "payload.txt").read_text() == "old"
    assert not target.with_name("happy.staging").exists()
    assert not target.with_name("happy.bak").exists()


def test_stage_verify_swap_recovers_backup_before_new_build(tmp_path: Path) -> None:
    target = tmp_path / "packs" / "happy"
    backup = target.with_name("happy.bak")
    backup.mkdir(parents=True)
    (backup / "payload.txt").write_text("old")

    def build(staging: Path) -> None:
        (staging / "payload.txt").write_text("new")
        raise _InstallError("broken staging")

    with pytest.raises(_InstallError, match="broken staging"):
        stage_verify_swap(
            target,
            force=True,
            label="user/happy",
            build=build,
            make_error=_err,
        )

    assert (target / "payload.txt").read_text() == "old"
    assert not target.with_name("happy.staging").exists()
    assert not backup.exists()


def test_stage_verify_swap_refuses_existing_target_without_force(tmp_path: Path) -> None:
    target = tmp_path / "packs" / "happy"
    target.mkdir(parents=True)

    def build(_staging: Path) -> None:
        raise AssertionError("existing target should fail before staging")

    with pytest.raises(_InstallError, match="exists; pass force=True"):
        stage_verify_swap(
            target,
            force=False,
            label="user/happy",
            build=build,
            make_error=_err,
        )


def test_failed_backup_recovery_keeps_the_only_good_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "happy"
    backup = tmp_path / "happy.bak"
    backup.mkdir()
    (backup / "payload.txt").write_text("only good copy")
    original_rename = Path.rename
    builds: list[Path] = []

    def rename(path: Path, destination: Path) -> Path:
        if path == backup:
            raise PermissionError("injected recovery failure")
        return original_rename(path, destination)

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(_InstallError):
        stage_verify_swap(target, force=True, label="happy", build=builds.append, make_error=_err)
    assert (backup / "payload.txt").read_text() == "only good copy"
    assert builds == []


def test_recovered_install_is_not_overwritten_without_force(tmp_path: Path) -> None:
    target = tmp_path / "happy"
    backup = tmp_path / "happy.bak"
    backup.mkdir()
    (backup / "payload.txt").write_text("recovered")

    def build(staging: Path) -> None:
        (staging / "payload.txt").write_text("replacement")

    with pytest.raises(_InstallError, match="exists; pass force=True"):
        stage_verify_swap(target, force=False, label="happy", build=build, make_error=_err)
    assert (target / "payload.txt").read_text() == "recovered"
