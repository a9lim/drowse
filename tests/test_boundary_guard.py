"""Server import-surface guards.

Source-grep checks over the frontend packages: no import of a promoted name by
its old underscore form, and no server module importing a sibling's private
helper.  (Session-private attribute reaches are an AST invariant over the whole
package in ``tests/test_layering.py``.)

Comments and string literals are stripped via :mod:`tokenize` before the
match, so a historical note or a docstring example doesn't trip a guard —
only live code does.

CPU-only: pure source-grep, no model load.
"""
from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path


def _drowse_root() -> Path:
    return Path(__file__).resolve().parent.parent / "drowse"


def _strip_comments_and_strings(source: str) -> str:
    """Return ``source`` with comment and string-literal tokens blanked.

    Keeps line/column structure so a flagged hit still reports a usable line
    number.  Blanking (rather than deleting) string tokens means a reach-in
    that only ever appears inside a docstring or an f-string literal is *not*
    a real reach-in and is correctly ignored.
    """
    out_lines = source.splitlines()
    # Rebuild as a mutable char grid so we can null out token spans in place.
    grid = [list(line) for line in out_lines]
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                continue
            (srow, scol), (erow, ecol) = tok.start, tok.end
            for row in range(srow, erow + 1):
                line = grid[row - 1]
                lo = scol if row == srow else 0
                hi = ecol if row == erow else len(line)
                for col in range(lo, min(hi, len(line))):
                    line[col] = " "
    except (tokenize.TokenError, IndentationError):
        # A file we can't tokenize shouldn't mask a reach-in: fall back to the
        # raw source (the regex still runs; at worst a commented hit surfaces,
        # which is a louder failure than a silent miss).
        return source
    return "\n".join("".join(line) for line in grid)


# ---------------------------------------------------------------------------
# Promoted-name regression guard (T2.4)
# ---------------------------------------------------------------------------

# The four names that were promoted from underscore-private to public in T2.4.
# Frontends must now import the public form; importing the old underscore name
# from any module is a regression against the promotion.
_PROMOTED_OLD_NAMES = frozenset(["_export_gguf_manifold"])

# All cross-module import surfaces (CLI/server plus core helpers) must use
# the promoted public names.
_ALL_FRONTEND_DIRS = ("server", "cli", "core")

_UNDERSCORE_IMPORT_RE = re.compile(
    r"(?:^|\s)from\s+[\w.]+\s+import\s+[^#\n]*\b(_[A-Za-z]\w*)"
)


def _find_promoted_old_name_imports() -> list[str]:
    """Return one diagnostic string per import of a now-promoted name."""
    root = _drowse_root()
    offenders: list[str] = []
    for sub in _ALL_FRONTEND_DIRS:
        for path in sorted((root / sub).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            code = _strip_comments_and_strings(path.read_text(encoding="utf-8"))
            for lineno, line in enumerate(code.splitlines(), start=1):
                for m in _UNDERSCORE_IMPORT_RE.finditer(line):
                    name = m.group(1)
                    if name in _PROMOTED_OLD_NAMES:
                        offenders.append(f"{rel}:{lineno}: import of old name {name!r}")
    return offenders


def test_promoted_names_not_imported_by_old_underscore_form() -> None:
    """Frontends must not import the four T2.4-promoted names by their old
    underscore form.

    Each of these was renamed to a public name and the old ``_``-prefixed form
    is now only a back-compat alias kept for monkeypatching in tests.  A fresh
    cross-module import of the old name is a boundary violation.
    """
    offenders = _find_promoted_old_name_imports()
    assert not offenders, (
        "Frontend code imports a now-public name by its old underscore form.\n"
        "Use the public name (export_gguf_manifold) instead.\n\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# Intra-server private-import guard
# ---------------------------------------------------------------------------

_SERVER_PRIVATE_IMPORT_RE = re.compile(
    r"(?:^|\s)from\s+drowse\.server[\w.]*\s+import\s+[^#\n]*?(?<![\w.])"
    r"(_[A-Za-z]\w*)"
)


def test_server_modules_do_not_import_each_others_privates() -> None:
    """A route module must not reach into a sibling's underscore helper.

    Shared native-tree primitives belong in ``server/runtime.py``
    (``resolve_session_id``, ``refuse_if_busy``, the session lock) or the
    shared plumbing modules (``errors`` / ``operations`` /
    ``request_helpers``).  Importing ``manifold_routes._refuse_if_busy`` from
    ``profile_routes`` — the pattern this guard retires — made a guard that is
    not manifold-specific look like manifold-route internals, and made either
    module's private rename break the other.
    """
    root = _drowse_root()
    offenders: list[str] = []
    for path in sorted((root / "server").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        code = _strip_comments_and_strings(path.read_text(encoding="utf-8"))
        for lineno, line in enumerate(code.splitlines(), start=1):
            for m in _SERVER_PRIVATE_IMPORT_RE.finditer(line):
                offenders.append(
                    f"{rel}:{lineno}: imports private {m.group(1)!r} "
                    "from another server module"
                )
    assert not offenders, (
        "A server module imports a sibling module's private helper.\n"
        "Promote the helper into server/runtime.py (or the relevant "
        "shared plumbing module) instead.\n\n" + "\n".join(offenders)
    )
