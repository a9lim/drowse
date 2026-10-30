"""Bundled-template materialization infra — CPU-only, content-independent.

Tests the bundled-template half of :mod:`drowse.store.bundled` against a
synthetic fixture template injected via ``_resources``, so coverage doesn't depend on whatever
template-derived manifold happens to ship.  No template-derived bundled manifold
currently ships (colors was the first candidate, pulled after it resolved to a
flat lexical scatter rather than a hue ring); the end-to-end ``template_ref``
wiring test returns to this suite when a real one lands.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import drowse.store.bundled as bundled
from drowse.base.paths import templates_dir
from drowse.store.bundled import bundled_template_names
from drowse.store.templates import TemplateFolder, resolve_template


def materialize_bundled_templates() -> None:
    bundled._materialize_templates()

_DEMO = {
    "format_version": 2,
    "name": "demo",
    "slot": "[X]",
    "values": ["alpha", "beta"],
    "contexts": [
        {"turns": [{"role": "user", "content": "pick one"}], "assistant": "[X]"},
    ],
    "description": "",
    "source": "bundled",
    "tags": [],
}


class _FakeResources:
    """Stand-in for ``importlib.resources`` pointing at a temp package-data root."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._manifolds = root.parent / "no-manifolds"
        self._manifolds.mkdir(parents=True, exist_ok=True)

    def files(self, pkg: str) -> Path:
        if pkg == "drowse.data.manifolds":
            return self._manifolds
        assert pkg == "drowse.data.templates"
        return self._root


def _write_template(
    pkg_root: Path, payload: dict[str, object], name: str = "demo",
) -> None:
    d = pkg_root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "template.json").write_text(json.dumps(payload))


def _wire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pkg_root: Path) -> None:
    pkg_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(bundled, "_resources", _FakeResources(pkg_root))
    bundled.reset_bundled_state()


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #


def test_empty_package_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _wire(monkeypatch, tmp_path, tmp_path / "pkg")  # no template written
    assert bundled_template_names() == []
    materialize_bundled_templates()  # must not crash, must not create default/
    assert not (templates_dir() / "default").exists()


def test_valid_template_advertised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pkg = tmp_path / "pkg"
    _wire(monkeypatch, tmp_path, pkg)
    _write_template(pkg, _DEMO)
    assert bundled_template_names() == ["demo"]


def test_invalid_template_not_advertised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    pkg = tmp_path / "pkg"
    _wire(monkeypatch, tmp_path, pkg)
    _write_template(pkg, {**_DEMO, "values": ["only_one"]})  # < 2 values: invalid
    assert bundled_template_names() == []


# --------------------------------------------------------------------------- #
# materialization
# --------------------------------------------------------------------------- #


def test_fresh_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pkg = tmp_path / "pkg"
    _wire(monkeypatch, tmp_path, pkg)
    _write_template(pkg, _DEMO)
    materialize_bundled_templates()
    manifest = templates_dir() / "default" / "demo" / "template.json"
    assert manifest.exists()
    tmpl = TemplateFolder.load(manifest.parent)
    assert tmpl.name == "demo" and list(tmpl.values) == ["alpha", "beta"]
    # bare-name resolves to the materialized default copy
    assert resolve_template("demo").slot == "[X]"


def test_process_scope_no_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pkg = tmp_path / "pkg"
    _wire(monkeypatch, tmp_path, pkg)
    _write_template(pkg, _DEMO)
    bundled.materialize_bundled_artifacts()
    manifest = templates_dir() / "default" / "demo" / "template.json"
    manifest.unlink()
    bundled.materialize_bundled_artifacts()  # guard set for this home → no re-copy
    assert not manifest.exists()


def test_process_scope_guard_is_per_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    pkg = tmp_path / "pkg"
    _wire(monkeypatch, tmp_path, pkg)
    _write_template(pkg, _DEMO)
    bundled.materialize_bundled_artifacts()
    assert (templates_dir() / "default" / "demo" / "template.json").exists()

    monkeypatch.setenv("DROWSE_HOME", str(tmp_path / "second-home"))
    bundled.materialize_bundled_artifacts()
    assert (templates_dir() / "default" / "demo" / "template.json").exists()


def test_bundle_update_recopies_and_backs_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    pkg = tmp_path / "pkg"
    _wire(monkeypatch, tmp_path, pkg)
    _write_template(pkg, _DEMO)  # bundled: 2 values
    # A stale on-disk copy with different content — canonical hash differs.
    target = templates_dir() / "default" / "demo"
    target.mkdir(parents=True)
    (target / "template.json").write_text(json.dumps({
        **_DEMO, "values": ["gamma", "delta", "epsilon"],
    }))
    with pytest.warns(UserWarning, match="refreshed bundled template default/demo"):
        materialize_bundled_templates()
    refreshed = TemplateFolder.load(target)
    assert list(refreshed.values) == ["alpha", "beta"]   # bundled content won
    assert (target / "template.json.bak").exists()        # old copy preserved


# --------------------------------------------------------------------------- #
# the ordering invariant lives in one place
# --------------------------------------------------------------------------- #


def test_bootstrap_entry_point_runs_templates_before_manifolds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The order is load-bearing: a bundled manifold may ``template_ref`` a
    bundled template, and its fit raises ``TemplateNotFoundError`` if the
    template has not materialized yet."""
    order: list[str] = []
    monkeypatch.setenv("DROWSE_HOME", str(tmp_path))
    bundled.reset_bundled_state()
    monkeypatch.setattr(bundled, "_materialize_templates", lambda: order.append("templates"))
    monkeypatch.setattr(bundled, "bundled_manifold_names", lambda: ["probe"])
    monkeypatch.setattr(
        bundled, "_materialize_manifold",
        lambda name, *, force: order.append(f"manifold:{name}"),
    )
    bundled.materialize_bundled_artifacts()
    assert order == ["templates", "manifold:probe"]
