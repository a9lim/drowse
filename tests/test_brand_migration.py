from __future__ import annotations

import json
from pathlib import Path

import pytest

from drowse.io.brand_migration import migrate_legacy_home, migrate_legacy_record


@pytest.mark.parametrize("name", ["polythetic", "saklas"])
def test_legacy_home_reuses_data_without_invalidating_digests(tmp_path: Path, name: str) -> None:
    legacy = tmp_path / f".{name}"
    manifest = legacy / "manifolds" / "local" / "demo" / "manifold.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({
            "saklas_version": "5.3.0",
            "kind": "saklas-manifold",
            "producer": {"name": "saklas"},
            "archive": "demo.saklaspack",
        }),
        encoding="utf-8",
    )

    original = manifest.read_bytes()
    assert migrate_legacy_home(tmp_path) == legacy
    assert manifest.read_bytes() == original
    assert not (tmp_path / ".drowse").exists()


def test_existing_drowse_home_is_never_merged_or_overwritten(tmp_path: Path) -> None:
    legacy = tmp_path / ".saklas"
    current = tmp_path / ".drowse"
    legacy.mkdir()
    current.mkdir()
    (legacy / "legacy.txt").write_text("legacy", encoding="utf-8")
    (current / "current.txt").write_text("current", encoding="utf-8")

    assert migrate_legacy_home(tmp_path) == current
    assert (legacy / "legacy.txt").read_text(encoding="utf-8") == "legacy"
    assert (current / "current.txt").read_text(encoding="utf-8") == "current"


@pytest.mark.parametrize("name", ["polythetic", "saklas"])
def test_record_migration_only_changes_brand_contracts(name: str) -> None:
    value = {
        f"{name}_version": "5.3.0",
        "producer": {"name": name},
        "text": name,
        "source": f"https://github.com/example/{name}-reference",
    }
    assert migrate_legacy_record(value) == {
        "drowse_version": "5.3.0",
        "producer": {"name": "drowse"},
        "text": name,
        "source": f"https://github.com/example/{name}-reference",
    }


def test_conflicting_version_is_not_silently_accepted() -> None:
    value = {"drowse_version": "new", "polythetic_version": "old"}
    assert migrate_legacy_record(value) == value


def test_fresh_install_uses_drowse(tmp_path: Path) -> None:
    assert migrate_legacy_home(tmp_path) == tmp_path / ".drowse"
