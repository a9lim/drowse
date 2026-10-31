"""Bundled artifacts: package data materialized into ``~/.drowse/``.

Drowse ships two artifact kinds inside the wheel — templates under
``drowse/data/templates/`` and manifolds under ``drowse/data/manifolds/`` —
which materialize into the ``default/`` namespace of ``~/.drowse/``.
:func:`materialize_bundled_artifacts` is the single entry point every
bootstrap site calls.  It is a per-``DROWSE_HOME`` process-scoped no-op after
its first call, and it owns the load-bearing order: templates before
manifolds, because a bundled manifold may ``template_ref`` a bundled
template (its fit resolves that ref).

The process scope exists because drift detection cannot tell "the bundle
changed under the user" from "the user overrode a default manifold's
hyperparameters via the CLI earlier in this invocation"; bundle updates ship
via pip and need a restart anyway.  :func:`refresh_bundled_manifold` is the
explicit, unguarded re-copy behind ``drowse pack refresh default/<name>``.

A manifold is copied through stage-and-swap, so a crash never leaves a
half-copied folder in ``default/``.  Fitted tensor pairs are local state and
survive a re-copy together with their manifest proofs; the per-tensor
``nodes_sha256`` decides whether an old fit is still current.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import warnings
from importlib import resources as _resources
from pathlib import Path
from typing import Any

from drowse.base.atomic import write_bytes_atomic, write_json_atomic
from drowse.base.integrity import canonical_json
from drowse.base.paths import drowse_home, manifolds_dir, templates_dir
from drowse.store.manifold_folder import (
    locked_manifold_folder,
    manifold_folder_tensor_paths,
    replace_manifold_folder,
)
from drowse.store.manifold_schema import (
    LOCAL_STATE_KEYS,
    MANIFOLD_FORMAT_VERSION,
    ManifoldFormatError,
    node_filename,
)

_log = logging.getLogger(__name__)

# The ``DROWSE_HOME`` the last :func:`materialize_bundled_artifacts` covered.
_materialized_home: Path | None = None


def reset_bundled_state() -> None:
    """Forget the process-scope guard so the next bootstrap runs again."""
    global _materialized_home
    _materialized_home = None


def materialize_bundled_artifacts() -> None:
    """Materialize bundled templates, then bundled manifolds (once per home)."""
    global _materialized_home
    home = drowse_home()
    if _materialized_home == home:
        return
    _materialized_home = home

    from drowse.store.selectors import invalidate as invalidate_selector_index

    home.mkdir(parents=True, exist_ok=True)
    _materialize_templates()
    for name in bundled_manifold_names():
        _materialize_manifold(name, force=False)
    # The installed roster may have changed; the resolver's memo must not
    # serve the pre-materialization walk.
    invalidate_selector_index()


def payload_drift_sha256(data: bytes, *, strip_keys: tuple[str, ...] = ()) -> str:
    """Bundle-drift sha256 of a JSON payload, cosmetic differences ignored.

    Hashes the canonical-JSON form with ``strip_keys`` (local transaction
    state) removed; unparseable bytes hash raw, so they read as drift.
    """
    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return hashlib.sha256(data).hexdigest()
    if isinstance(parsed, dict):
        for key in strip_keys:
            parsed.pop(key, None)
    return hashlib.sha256(canonical_json(parsed)).hexdigest()


# ============================================================ manifolds ===


def _bundled_manifest_node_files(pkg_root: Any) -> list[str] | None:
    """Declared node corpus filenames of a bundled manifest, or ``None`` if invalid."""
    try:
        with pkg_root.joinpath("manifold.json").open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (AttributeError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    nodes = payload.get("nodes") if isinstance(payload, dict) else None
    if not isinstance(nodes, list):
        return None
    names: list[str] = []
    for idx, node in enumerate(nodes):
        label = node.get("label") if isinstance(node, dict) else None
        if not isinstance(label, str):
            return None
        names.append(node_filename(idx, label))
    return names


def _bundled_manifest_complete(pkg_root: Any) -> bool:
    """True when a package-data manifold has every node corpus it declares."""
    node_files = _bundled_manifest_node_files(pkg_root)
    if node_files is None:
        return False
    nodes_root = pkg_root.joinpath("nodes")
    return not node_files or (
        nodes_root.is_dir() and all(nodes_root.joinpath(n).is_file() for n in node_files)
    )


def bundled_manifold_names() -> list[str]:
    """Complete manifolds shipped under ``drowse/data/manifolds/``."""
    try:
        root = _resources.files("drowse.data.manifolds")
    except (ModuleNotFoundError, FileNotFoundError):
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and _bundled_manifest_complete(p)
    )


def _manifold_drift(pkg_root: Any, target: Path) -> str | None:
    """Why the materialized copy at ``target`` no longer matches the bundle.

    ``None`` means current.  A missing folder or manifest, an unreadable or
    older-format manifest, a changed manifest (local state stripped), or a
    declared node corpus file that is missing on disk all count as drift.
    Node *content* edits do not: a corpus edit under ``default/`` persists
    until the bundle itself moves.
    """
    manifest = target / "manifold.json"
    if not manifest.is_file():
        return "fresh install" if not target.exists() else "manifold.json missing"
    try:
        on_disk = manifest.read_bytes()
        payload = json.loads(on_disk)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return "manifold.json unreadable"
    fmt = payload.get("format_version") if isinstance(payload, dict) else None
    if isinstance(fmt, bool) or fmt != MANIFOLD_FORMAT_VERSION:
        return f"v{fmt}->v{MANIFOLD_FORMAT_VERSION} (format_version)"
    shipped = pkg_root.joinpath("manifold.json").read_bytes()
    if (
        payload_drift_sha256(on_disk, strip_keys=LOCAL_STATE_KEYS)
        != payload_drift_sha256(shipped, strip_keys=LOCAL_STATE_KEYS)
    ):
        return "manifest content changed"
    node_files = _bundled_manifest_node_files(pkg_root) or []
    if any(not (target / "nodes" / name).is_file() for name in node_files):
        return "node corpus file missing"
    return None


def _link_or_copy(source: Path, dest: Path) -> None:
    """Hard-link a fitted file into staging (same filesystem), else copy it."""
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def _stage_bundled_manifold(pkg_root: Any, target: Path, staging: Path) -> None:
    """Build a fresh copy of the bundle in ``staging``, carrying local fits."""
    payload = json.loads(pkg_root.joinpath("manifold.json").read_bytes())
    old_files: dict[str, Any] = {}
    if (target / "manifold.json").is_file():
        try:
            old_payload = json.loads((target / "manifold.json").read_bytes())
            if isinstance(old_payload, dict) and isinstance(old_payload.get("files"), dict):
                old_files = old_payload["files"]
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            old_files = {}
    carried: dict[str, Any] = {}
    for tensor in manifold_folder_tensor_paths(target):
        for half in (tensor, tensor.with_suffix(".json")):
            if half.is_file():
                _link_or_copy(half, staging / half.name)
                if half.name in old_files:
                    carried[half.name] = old_files[half.name]
    shipped_files = payload.get("files")
    payload["files"] = {**carried, **(shipped_files if isinstance(shipped_files, dict) else {})}
    nodes_dir = staging / "nodes"
    nodes_dir.mkdir()
    pkg_nodes = pkg_root.joinpath("nodes")
    if pkg_nodes.is_dir():
        for node_file in pkg_nodes.iterdir():
            if node_file.is_file() and node_file.name.endswith(".json"):
                write_bytes_atomic(nodes_dir / node_file.name, node_file.read_bytes())
    write_json_atomic(staging / "manifold.json", payload)


def _materialize_manifold(name: str, *, force: bool) -> None:
    """Bring ``default/<name>`` up to date with the bundle (or re-copy it)."""
    target = manifolds_dir() / "default" / name
    pkg_root = _resources.files("drowse.data.manifolds").joinpath(name)
    with locked_manifold_folder(target):
        reason = "explicit refresh" if force else _manifold_drift(pkg_root, target)
        if reason is None:
            return
        existed = target.exists()
        replace_manifold_folder(
            target, lambda stage: _stage_bundled_manifold(pkg_root, target, stage),
        )
    if existed and not force:
        warnings.warn(
            f"refreshed bundled manifold default/{name} — {reason}; local edits "
            f"to its node corpus were overwritten (fork under local/ to keep a "
            f"custom corpus)",
            UserWarning,
            stacklevel=3,
        )
        _log.warning("refreshed bundled manifold default/%s — %s", name, reason)


def refresh_bundled_manifold(name: str) -> None:
    """Re-copy one bundled manifold from package data, unconditionally.

    The explicit ``pack refresh`` path: bypasses the process-scope guard and
    the drift check, restoring the shipped manifest and every shipped node
    corpus file while keeping the folder's fitted pairs and their proofs.
    """
    from drowse.store.selectors import invalidate as invalidate_selector_index

    if name not in bundled_manifold_names():
        raise ManifoldFormatError(f"no bundled manifold named {name!r} ships with drowse")
    _materialize_manifold(name, force=True)
    invalidate_selector_index()


# ============================================================ templates ===
#
# None ship at present; the machinery is dormant by design until one does.


def _bundled_template_valid(pkg_root: Any) -> bool:
    """True when a package-data template parses and passes format validation."""
    from drowse.store.templates import TemplateFolder, TemplateFormatError

    try:
        with pkg_root.joinpath("template.json").open(encoding="utf-8") as handle:
            payload = json.load(handle)
        TemplateFolder.from_payload(payload)
    except (
        AttributeError, OSError, json.JSONDecodeError, UnicodeDecodeError,
        TemplateFormatError, ValueError,
    ):
        return False
    return True


def bundled_template_names() -> list[str]:
    """Valid templates shipped under ``drowse/data/templates/``."""
    try:
        root = _resources.files("drowse.data.templates")
    except (ModuleNotFoundError, FileNotFoundError):
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and _bundled_template_valid(p)
    )


def _materialize_templates() -> None:
    """Copy bundled templates into ``templates/default/``.

    Fresh install copies ``template.json``; a bundle update (canonical hash or
    ``format_version`` drift) re-copies it, keeping a ``.json.bak`` of the
    replaced bytes.  An unreadable on-disk copy is left alone.
    """
    from drowse.store.templates import TEMPLATE_FORMAT_VERSION

    names = bundled_template_names()
    if not names:
        return
    root = _resources.files("drowse.data.templates")
    default_dir = templates_dir() / "default"
    default_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        bundled_bytes = root.joinpath(name).joinpath("template.json").read_bytes()
        manifest = default_dir / name / "template.json"
        if not manifest.exists():
            manifest.parent.mkdir(parents=True, exist_ok=True)
            write_bytes_atomic(manifest, bundled_bytes)
            continue
        try:
            on_disk_bytes = manifest.read_bytes()
            on_disk_payload = json.loads(on_disk_bytes)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        fmt = on_disk_payload.get("format_version") if isinstance(on_disk_payload, dict) else None
        format_stale = isinstance(fmt, bool) or fmt != TEMPLATE_FORMAT_VERSION
        if not format_stale and payload_drift_sha256(on_disk_bytes) == payload_drift_sha256(bundled_bytes):
            continue
        write_bytes_atomic(manifest.with_suffix(".json.bak"), on_disk_bytes)
        write_bytes_atomic(manifest, bundled_bytes)
        reason = (
            f"v{fmt}->v{TEMPLATE_FORMAT_VERSION} (format_version)"
            if format_stale else "content changed"
        )
        warnings.warn(
            f"refreshed bundled template default/{name} — {reason}; local edits "
            f"were overwritten (fork under a non-default namespace to keep a "
            f"custom template)",
            UserWarning,
            stacklevel=3,
        )
        _log.warning("refreshed bundled template default/%s — %s", name, reason)
