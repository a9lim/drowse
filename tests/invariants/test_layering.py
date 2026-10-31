"""The package layer order: no drowse module imports a higher layer.

An AST walk over every module under ``drowse/``.  Every import counts — module
level, function level and ``if TYPE_CHECKING:`` alike: a deferred import is
still a dependency, and a type-only one still couples the lower layer to the
higher one's shape (type such a parameter against a narrow ``Protocol`` at the
lower layer instead).

Lower layers come first; a module may import its own layer or any lower one.

A second walk keeps the session's collaborators honest: outside
``drowse/session/`` no code reaches a session private (``session._x`` /
``sess._x``, ``<x>.session._x``, ``self._session._x``); collaborators and
frontends take explicit dependencies or the public API instead.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "drowse"

#: Lowest first.  An entry names a package (``geometry``) or a module inside
#: one (``steering.grammar``); the longest matching entry wins.
LAYERS: tuple[tuple[str, ...], ...] = (
    ("errors", "naming", "events"),
    ("base",),
    ("geometry",),
    ("model",),
    ("store",),
    # Parsing an expression resolves bare atoms against installed manifolds
    # (store.selectors), so the grammar sits above the store.
    ("steering", "steering.grammar", "steering.terms", "steering.triggers"),
    ("readout",),
    ("lens", "sae"),
    ("steering.composer", "steering.hooks"),
    ("generation",),
    ("fit",),
    ("loom",),
    ("session",),
    ("cli", "server", "web", "notebook"),
)
#: ``drowse/__init__.py`` (the public facade) and ``__main__`` sit on top.
TOP = len(LAYERS)

#: Upward imports that cannot move yet: ``(importer, imported module) -> why``.
#: Every entry must still occur (see ``test_allowlist_is_current``), so moving
#: the code forces deleting its entry.
ALLOWED: dict[tuple[str, str], str] = {}


def _module_name(path: Path, root: Path = ROOT) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _layer(module: str) -> int | None:
    parts = module.split(".")[1:]
    if not parts or parts == ["__main__"]:
        return TOP
    best: tuple[int, int] | None = None
    for index, group in enumerate(LAYERS):
        for entry in group:
            key = entry.split(".")
            if parts[: len(key)] == key and (best is None or len(key) > best[1]):
                best = (index, len(key))
    return None if best is None else best[0]


def _imports(
    path: Path, modules: set[str], root: Path = ROOT,
) -> list[tuple[int, str]]:
    """``(line, imported drowse module)`` for every import in ``path``."""
    me = _module_name(path, root)
    package = me.split(".") if path.name == "__init__.py" else me.split(".")[:-1]
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                source = ".".join(base + ([node.module] if node.module else []))
            else:
                source = node.module or ""
            for alias in node.names:
                # The root's version constant: every submodule import has
                # already initialized ``drowse/__init__``, and pyproject reads
                # the version from there (one place to bump).
                if source == "drowse" and alias.name == "__version__":
                    continue
                submodule = f"{source}.{alias.name}"
                found.append((node.lineno, submodule if submodule in modules else source))
    found.sort()
    return [(line, target) for line, target in found if target.split(".")[0] == "drowse"]


def _is_private(attr: str) -> bool:
    return attr.startswith("_") and not attr.startswith("__")


def _session_private_reaches(path: Path) -> list[tuple[int, str]]:
    """``(line, source)`` for every ``session._x`` / ``sess._x`` /
    ``<x>.session._x`` / ``self._session._x`` attribute access in ``path``."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if not (isinstance(node, ast.Attribute) and _is_private(node.attr)):
            continue
        owner = node.value
        if (isinstance(owner, ast.Name) and owner.id in ("session", "sess")) or (
            isinstance(owner, ast.Attribute) and owner.attr in ("session", "_session")
        ):
            found.append((node.lineno, ast.unparse(node)))
    return found


def _scan() -> tuple[list[str], set[tuple[str, str]], list[str]]:
    paths = sorted(PACKAGE.rglob("*.py"))
    modules = {_module_name(path) for path in paths}
    violations: list[str] = []
    allowed_seen: set[tuple[str, str]] = set()
    unassigned: list[str] = []
    for path in paths:
        me = _module_name(path)
        mine = _layer(me)
        if mine is None:
            unassigned.append(me)
            continue
        for line, target in _imports(path, modules):
            theirs = _layer(target)
            if theirs is None or theirs <= mine:
                continue
            if (me, target) in ALLOWED:
                allowed_seen.add((me, target))
                continue
            violations.append(
                f"{path.relative_to(ROOT)}:{line}: {me} (layer {mine}) imports "
                f"{target} (layer {theirs})"
            )
    return violations, allowed_seen, unassigned


def test_every_module_has_a_layer() -> None:
    _, _, unassigned = _scan()
    assert not unassigned, f"assign these modules a layer in LAYERS: {unassigned}"


def test_no_module_imports_a_higher_layer() -> None:
    violations, _, _ = _scan()
    assert not violations, "upward imports:\n" + "\n".join(violations)


def test_allowlist_is_current() -> None:
    _, allowed_seen, _ = _scan()
    stale = sorted(set(ALLOWED) - allowed_seen)
    assert not stale, f"delete these ALLOWED entries, the imports are gone: {stale}"


@pytest.mark.parametrize(("source", "expected"), [
    ("from drowse.fit.capture import x\n", [(1, "drowse.fit.capture")]),
    ("def f():\n    from drowse.fit import capture\n", [(2, "drowse.fit.capture")]),
    ("import drowse.loom.tree\n", [(1, "drowse.loom.tree")]),
    (
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from drowse.loom import tree\n",
        [(3, "drowse.loom.tree")],
    ),
    (
        "import typing\nif typing.TYPE_CHECKING:\n    pass\nelse:\n    import drowse.loom.tree\n",
        [(5, "drowse.loom.tree")],
    ),
    ("from drowse import __version__\n", []),
])
def test_scanner_reads_every_import(
    tmp_path: Path, source: str, expected: list[tuple[int, str]],
) -> None:
    """The walk sees module-level, function-level and type-only imports."""
    module = tmp_path / "drowse" / "geometry" / "probe.py"
    module.parent.mkdir(parents=True)
    module.write_text(source)
    modules = {"drowse.fit.capture", "drowse.loom.tree", "drowse.geometry.probe"}
    assert _imports(module, modules, tmp_path) == expected


def test_no_session_private_reaches_outside_the_session_package() -> None:
    """Collaborators take explicit dependencies; only ``drowse/session/``
    reads the session's private attributes."""
    session_pkg = PACKAGE / "session"
    reaches = [
        f"{path.relative_to(ROOT)}:{line}: {source}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if session_pkg not in path.parents
        for line, source in _session_private_reaches(path)
    ]
    assert not reaches, "session-private reaches:\n" + "\n".join(reaches)


@pytest.mark.parametrize(("source", "expected"), [
    ("session._monitor.add(x)\n", [(1, "session._monitor")]),
    ("self._session._engine\n", [(1, "self._session._engine")]),
    ("app.state.session._read_plane\n", [(1, "app.state.session._read_plane")]),
    ("sess._gen_lock.acquire()\n", [(1, "sess._gen_lock")]),
    ("# session._monitor\nx = 'session._monitor'\n", []),
    ("my_session._monitor\n", []),
    ("self._session.monitor\n", []),
    ("session.__class__\n", []),
    ("other._monitor\n", []),
])
def test_session_private_scanner(
    tmp_path: Path, source: str, expected: list[tuple[int, str]],
) -> None:
    module = tmp_path / "probe.py"
    module.write_text(source)
    assert _session_private_reaches(module) == expected
