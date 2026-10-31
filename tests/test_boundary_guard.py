"""Server import-surface guards.

A source-grep check over ``drowse/server``: no server module imports a
sibling's private helper.  (Session-private attribute reaches are an AST
invariant over the whole package in ``tests/test_layering.py``.)

Comments and string literals are stripped via :mod:`tokenize` before the
match, so a historical note or a docstring example doesn't trip the guard —
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
