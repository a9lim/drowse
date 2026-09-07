"""Read compatibility for data saved under earlier product names."""

from __future__ import annotations

from pathlib import Path
from typing import Any


LEGACY_NAMES = ("polythetic", "saklas")


def migrate_legacy_home(home: Path) -> Path:
    current = home / ".drowse"
    if current.exists():
        return current
    for name in LEGACY_NAMES:
        legacy = home / f".{name}"
        if legacy.is_dir() and not legacy.is_symlink():
            return legacy
    return current


def migrate_legacy_record(value: Any) -> Any:
    return _migrate_record(value)


def _migrate_record(value: Any, parent: str = "") -> Any:
    if isinstance(value, list):
        return [_migrate_record(entry, parent) for entry in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, entry in value.items():
        current_key = key
        for name in LEGACY_NAMES:
            for suffix in ("version", "transcript"):
                if key == f"{name}_{suffix}":
                    current_key = f"drowse_{suffix}"
        if current_key != key and current_key in value:
            if value[current_key] == entry:
                continue
            current_key = key
        if key == "kind" and entry in tuple(f"{name}-manifold" for name in LEGACY_NAMES):
            entry = "drowse-manifold"
        if parent == "producer" and key == "name" and entry in LEGACY_NAMES:
            entry = "drowse"
        result[current_key] = _migrate_record(entry, key)
    return result
