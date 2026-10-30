"""Shared runner context: errors, narration, effective config, sessions, names.

One place for what every verb group needs:

- :class:`CliError` — a runner ``raise``s it (or lets any ``DrowseError``
  escape); ``main`` maps it to one stderr line and an exit code.
- :func:`note` / :func:`progress` — narration goes to **stderr**, so a verb's
  stdout carries only its result (a ``-j`` payload stays one JSON document).
- :func:`effective_config` — ``~/.drowse/config.yaml`` + ``-c`` files composed
  under the CLI flags (CLI > YAML > runner default).
- :func:`open_session` — the one session-construction path: narrated load,
  YAML sampling defaults applied, closed on exit.
- :func:`resolve_manifold` / :func:`existing_template` / :func:`split_new` —
  addressing an existing manifold or template (cross-namespace) vs authoring a
  fresh ``ns/name`` (bare → ``local``); :func:`split_ref` validates the shape
  of every such reference first.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from drowse.errors import DrowseError

if TYPE_CHECKING:
    from drowse.cli.config_file import ConfigFile
    from drowse.session.session import DrowseSession
    from drowse.store.manifold_folder import ManifoldFolder
    from drowse.store.templates import TemplateFolder


class CliError(DrowseError):
    """A user-facing CLI failure carrying its process exit code.

    ``exit_code`` 2 is a usage / bad-input error (the argparse convention),
    1 a runtime miss (not found, nothing to act on).
    """

    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code

    def user_message(self) -> tuple[int, str]:
        return (404 if self.exit_code == 1 else 400, str(self))


@contextmanager
def user_errors(*types: type[Exception], exit_code: int = 1) -> Iterator[None]:
    """Report a backend's plain stdlib exceptions as one-line CLI errors.

    For the io calls that still signal user-facing failures with bare
    ``ValueError`` / ``RuntimeError`` instead of a ``DrowseError``.
    """
    try:
        yield
    except types as e:
        raise CliError(str(e), exit_code=exit_code) from e


def exit_code_for(status: int) -> int:
    """HTTP-style ``user_message`` status → process exit code.

    404 is a miss (1); every other 4xx is bad input (2, the argparse usage
    code); 5xx and anything else is a runtime failure (1).
    """
    if status == 404:
        return 1
    if 400 <= status < 500:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def note(message: str) -> None:
    """Narration (startup, model info, announcements) — always stderr."""
    print(message, file=sys.stderr, flush=True)


def progress(message: str) -> None:
    """The indented progress sink every long operation reports through."""
    note(f"  {message}")


def emit_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Effective config
# ---------------------------------------------------------------------------

def effective_config(
    args: argparse.Namespace, *, default_max_tokens: int = 1024,
    yaml_vectors: bool = False,
) -> "ConfigFile":
    """Compose ``~/.drowse/config.yaml`` + ``-c`` files and stamp ``args``.

    Precedence is CLI > YAML > runner default for every field: ``model`` (only
    filled when the positional was omitted), ``max_tokens`` (verbs carrying
    ``--max-tokens`` default it to ``None`` so unset is distinguishable;
    ``default_max_tokens`` floors it), ``compile`` / ``cuda_graphs`` /
    ``top_k_alts`` (YAML ``return_top_k``), plus the YAML-only sampling
    defaults ``temperature`` / ``top_p`` / ``thinking`` / ``system_prompt``
    that :func:`open_session` applies to the session.

    YAML ``vectors:`` is the *server's* default steering: only
    ``yaml_vectors=True`` (``serve``) lets it fill an unset ``-S``.  A
    research verb (``experiment fan``) composes exactly the ``-S`` on its
    command line, so a global ``~/.drowse/config.yaml`` serve default never
    silently shifts a sweep.  Finishes by installing any HF-namespaced
    reference in the effective steering expression.
    """
    from drowse.cli.config_file import ConfigFile, ensure_vectors_installed

    extras = [Path(p) for p in (args.config or [])]
    composed = ConfigFile.effective(extras, include_default=True)
    if getattr(args, "model", None) is None:
        args.model = composed.model
    if getattr(args, "max_tokens", None) is None:
        args.max_tokens = (
            composed.max_tokens if composed.max_tokens is not None
            else default_max_tokens
        )
    args.temperature = composed.temperature
    args.top_p = composed.top_p
    args.thinking = composed.thinking
    args.system_prompt = composed.system_prompt
    if composed.compile:
        args.compile = True
    if composed.cuda_graphs:
        args.cuda_graphs = True
    if getattr(args, "top_k_alts", None) is None and composed.return_top_k is not None:
        args.top_k_alts = composed.return_top_k
    if yaml_vectors and args.steer is None:
        args.steer = composed.vectors
    steer = getattr(args, "steer", None)
    if steer:
        ensure_vectors_installed(steer, strict=args.strict)
    return composed


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def resolve_probes(raw: list[str] | None) -> list[str] | None:
    """``--probes`` → ``from_pretrained(probes=)``.

    Unset and ``all`` are ``None`` (the session's default roster: tagged
    concept axes plus every fitted bundled multi-node manifold); ``none`` or an
    empty list disable probes; a category list passes through verbatim.
    """
    if raw is None or raw == ["all"]:
        return None
    if raw in (["none"], []):
        return []
    return raw


def apply_generation_defaults(session: Any, args: argparse.Namespace) -> None:
    """Rebind the session's frozen ``GenerationConfig`` with YAML sampling
    defaults (``temperature`` / ``top_p`` / ``thinking``) — the fields
    ``from_pretrained`` has no parameter for."""
    overrides = {
        key: value
        for key in ("temperature", "top_p", "thinking")
        if (value := getattr(args, key, None)) is not None
    }
    if overrides:
        session.config = replace(session.config, **overrides)


@contextmanager
def open_session(
    args: argparse.Namespace, *, roster: bool = False,
) -> Iterator["DrowseSession"]:
    """Load ``args.model`` and yield the session, closing it on exit.

    Narrates the load on stderr (construction is slow — HF load, then with
    ``roster=True`` the neutral/whitener capture and a fit for every bundled
    concept the default probe roster still needs).  ``roster=False`` (every
    verb that never reads probe readings) constructs with ``probes=[]``.
    """
    from drowse.session.session import DrowseSession

    tail = f" ({args.quantize})" if args.quantize else ""
    note(f"Loading model: {args.model}{tail}")
    session = DrowseSession.from_pretrained(
        args.model,
        device=args.device,
        quantize=args.quantize,
        probes=resolve_probes(args.probes) if roster else [],
        system_prompt=getattr(args, "system_prompt", None),
        max_tokens=getattr(args, "max_tokens", None) or 1024,
        dls=not getattr(args, "no_dls", False),
        compile=getattr(args, "compile", False),
        cuda_graphs=getattr(args, "cuda_graphs", False),
        return_top_k=getattr(args, "top_k_alts", None) or 0,
        on_progress=progress,
    )
    with session:
        apply_generation_defaults(session, args)
        info = session.model_info
        note(f"Architecture: {info['model_type']}")
        note(f"Layers: {info['num_layers']}, Hidden dim: {info['hidden_dim']}")
        note(f"VRAM: {info['vram_used_gb']:.1f} GB")
        note(f"Loaded {len(session.probes)} probes")
        yield session


# ---------------------------------------------------------------------------
# Manifold names
# ---------------------------------------------------------------------------

def iter_manifolds(namespace: str | None = None) -> Iterator[tuple[str, "ManifoldFolder"]]:
    """Every installed ``(namespace, ManifoldFolder)``, bundled ones included.

    CLI verbs resolve folders before any session exists, so the bundled
    materializer runs here (a process-scoped no-op after the first call) —
    otherwise a first ``drowse manifold fit default/personas`` would miss it.
    """
    from drowse.store.bundled import materialize_bundled_artifacts
    from drowse.store.manifold_authoring import iter_manifold_folders

    materialize_bundled_artifacts()
    yield from iter_manifold_folders(namespace)


def split_ref(ref: str) -> tuple[str | None, str]:
    """``[ns/]name`` → ``(ns or None, name)``.

    Anything but one or two plain path components (an empty side, ``.`` /
    ``..``, a further separator) is a one-line usage error here, never an
    unsafe-path ``ValueError`` from the filesystem layer below.
    """
    ns, sep, name = ref.partition("/")
    parts = (ns, name) if sep else (ref,)
    if any(not p or p in {".", ".."} or "/" in p or "\\" in p for p in parts):
        raise CliError(f"bad reference {ref!r} (want NAME or NS/NAME)")
    return (ns, name) if sep else (None, ref)


def resolve_manifold(ref: str) -> tuple[str, str]:
    """Address an **existing** manifold: ``(namespace, name)``.

    ``ns/name`` pins verbatim (existence is the backend's check); a bare name
    walks every installed namespace, so it reaches bundled ``default/``.  A
    malformed reference exits 2, a miss exits 1, and a cross-namespace
    collision raises :class:`~drowse.store.selectors.AmbiguousSelectorError`
    (exit 2).
    """
    from drowse.store.bundled import materialize_bundled_artifacts
    from drowse.store.selectors import AmbiguousSelectorError

    ns, name = split_ref(ref)
    # Bundled folders materialize first, so even a pinned ``default/<name>``
    # resolves before any session has run the bootstrap.
    materialize_bundled_artifacts()
    if ns is not None:
        return ns, name
    matches = [found for found, mf in iter_manifolds() if mf.name == name]
    if not matches:
        raise CliError(f"manifold '{name}' not found", exit_code=1)
    if len(matches) > 1:
        qualified = ", ".join(f"{found}/{name}" for found in matches)
        raise AmbiguousSelectorError(
            f"ambiguous manifold '{name}': matches {qualified}"
        )
    return matches[0], name


def existing_folder(ref: str) -> Path:
    """:func:`resolve_manifold` → the folder, which must hold a ``manifold.json``."""
    from drowse.base.paths import manifold_dir

    ns, name = resolve_manifold(ref)
    folder = manifold_dir(ns, name)
    if not (folder / "manifold.json").exists():
        raise CliError(f"manifold {ns}/{name} not found at {folder}", exit_code=1)
    return folder


def folder_or_manifold(ref: str) -> Path:
    """A manifold folder **path** or an installed manifold reference.

    An existing directory is taken as a path (and must hold a
    ``manifold.json``); anything else resolves like :func:`existing_folder`.
    """
    path = Path(ref)
    if path.is_dir() or (path / "manifold.json").exists():
        if not (path / "manifold.json").is_file():
            raise CliError(f"no manifold.json in {path}")
        return path
    return existing_folder(ref)


def split_new(ref: str) -> tuple[str, str]:
    """Name a manifold or template **being authored**: bare → ``local/``."""
    ns, name = split_ref(ref)
    return (ns or "local"), name


def existing_template(ref: str) -> "TemplateFolder":
    """An installed template ``[ns/]name`` (a bare name resolves across
    namespaces); a malformed reference exits 2 before any lookup."""
    from drowse.store.templates import resolve_template

    split_ref(ref)
    return resolve_template(ref)


def check_steering(expression: str | None) -> None:
    """Parse ``expression`` once so a typo fails before any model load."""
    if expression:
        from drowse.steering.grammar import parse_expr

        parse_expr(expression)


def parse_layers(raw: str | None) -> "list[int] | str | None":
    """A ``--layers`` spec: explicit indices, or ``workspace`` / ``sample`` / ``all``."""
    if raw is None:
        return None
    lowered = raw.strip().lower()
    if lowered in {"workspace", "sample", "all"}:
        return lowered
    try:
        layers = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError:
        raise CliError(
            f"bad --layers value {raw!r} (want e.g. 12,24,36 or workspace)"
        ) from None
    if not layers:
        raise CliError("--layers must name at least one layer")
    return layers
