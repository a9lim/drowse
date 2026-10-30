"""The package layer order: no drowse module imports a higher layer.

An AST walk over every module under ``drowse/``.  Every runtime import counts —
module level and function level alike, since a deferred import is still a
dependency.  Imports under ``if TYPE_CHECKING:`` are exempt: they never run,
so they create no dependency, only an annotation.

Lower layers come first; a module may import its own layer or any lower one.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
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


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _runtime_imports(
    path: Path, modules: set[str], root: Path = ROOT,
) -> list[tuple[int, str]]:
    """``(line, imported drowse module)`` for every runtime import in ``path``."""
    me = _module_name(path, root)
    package = me.split(".") if path.name == "__init__.py" else me.split(".")[:-1]
    found: list[tuple[int, str]] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and _is_type_checking(child.test):
                for stmt in child.orelse:
                    visit_stmt(stmt)
                continue
            visit_stmt(child)

    def visit_stmt(node: ast.AST) -> None:
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
        elif isinstance(node, ast.If) and _is_type_checking(node.test):
            for stmt in node.orelse:
                visit_stmt(stmt)
        else:
            visit(node)

    visit(ast.parse(path.read_text(), filename=str(path)))
    return [(line, target) for line, target in found if target.split(".")[0] == "drowse"]


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
        for line, target in _runtime_imports(path, modules):
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
    ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from drowse.loom import tree\n", []),
    ("from drowse import __version__\n", []),
])
def test_scanner_reads_runtime_imports(
    tmp_path: Path, source: str, expected: list[tuple[int, str]],
) -> None:
    """The walk sees module- and function-level imports and skips type-only ones."""
    module = tmp_path / "drowse" / "geometry" / "probe.py"
    module.parent.mkdir(parents=True)
    module.write_text(source)
    modules = {"drowse.fit.capture", "drowse.loom.tree", "drowse.geometry.probe"}
    assert _runtime_imports(module, modules, tmp_path) == expected
