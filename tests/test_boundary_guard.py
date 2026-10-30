"""Server ↔ engine boundary guard.

The HTTP server is a frontend over :class:`DrowseSession`; it must talk to the
engine through its public API, never reach past it into a
``DrowseSession`` private (``session._monitor``, ``session._profiles``, …).
Reaching in couples a frontend to the engine's internal layout, so a private
rename silently breaks it — exactly the migration this test guards against
regressing.  Every former reach-in has a public accessor now (``monitor``,
``manifolds``, ``profiles``, ``model_info``, ``generation_state``,
``gen_lock``, ``joint_logprob_cache``,
``ensure_manifold_loaded``, ``ensure_profile_registered``); a fresh reach-in
should use those instead of being added to the allowlist below.

The grep is **session-scoped**, not "any ``self._x``": a widget or route
controller is free to keep its own privates.  Only attribute access whose
*receiver* is a session-object expression counts — the session is bound to a
bare ``session`` / ``sess`` local or reached through ``self._session``,
``self.session``, or ``.session``.

Comments and string literals are stripped via :mod:`tokenize` before the
match, so a historical note (``# v2.2 had self._messages = session._history``)
or a docstring example doesn't trip the guard — only live code does.

CPU-only: pure source-grep, no model load.
"""
from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

_FRONTEND_DIRS = ("server",)

# Receiver expressions that denote the DrowseSession object, each matched
# right before ``._<attr>``.  Two left-boundary regimes, so the dotted and the
# bare forms can't bleed into each other:
#   * a *dotted* receiver (``self._session`` / ``self.session`` / any
#     ``…​.session``) is anchored on the leading ``.``/``self`` — so
#     ``app.state.session._gen_lock`` matches via the ``.session`` arm;
#   * a *bare* ``session`` / ``sess`` local must NOT be preceded by ``.`` or a
#     word char, so ``my_session`` / ``subsession`` / ``x.session`` don't slip
#     through the bare arm (``x.session`` is still caught by the dotted arm).
_SESSION_PRIVATE_RE = re.compile(
    r"(?:"
    r"(?:self|[\w\]\)])\.session"  # self.session / app.state.session / …
    r"|self\._session"             # an explicit private session binding
    r"|(?<![\w.])(?:session|sess)"  # a bare session/sess local
    r")"
    r"\._[A-Za-z]\w*"
)


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


def _iter_frontend_files() -> list[Path]:
    root = _drowse_root()
    files: list[Path] = []
    for sub in _FRONTEND_DIRS:
        files.extend(sorted((root / sub).rglob("*.py")))
    assert files, "no frontend python files found — path drift?"
    return files


def _find_reach_ins() -> dict[str, list[str]]:
    """Map ``<relpath>:<receiver>._<attr>`` -> list of human-readable hits."""
    root = _drowse_root()
    hits: dict[str, list[str]] = {}
    for path in _iter_frontend_files():
        rel = path.relative_to(root).as_posix()
        code = _strip_comments_and_strings(path.read_text(encoding="utf-8"))
        for lineno, line in enumerate(code.splitlines(), start=1):
            for match in _SESSION_PRIVATE_RE.finditer(line):
                token = match.group(0)
                key = f"{rel}:{token}"
                hits.setdefault(key, []).append(f"{rel}:{lineno}: {token}")
    return hits


def test_no_frontend_session_private_reach_ins() -> None:
    """No ``drowse/server`` code reaches a DrowseSession private."""
    hits = _find_reach_ins()
    if hits:
        report = "\n".join(
            line for lines in hits.values() for line in lines
        )
        raise AssertionError(
            "frontend code reaches past the DrowseSession public API.\n"
            "Use the public accessor (monitor / manifolds / profiles / "
            "model_info / generation_state / gen_lock / "
            "joint_logprob_cache / "
            "ensure_manifold_loaded / ensure_profile_registered, …) at the "
            "call site.\n\n" + report
        )


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
