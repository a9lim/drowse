"""``drowse serve`` and its startup policies.

Startup order: effective config → bind/model checks → parse ``-S`` (YAML
``vectors:`` fills an unset flag) → load the session → preflight the default
steering through the session (a missing concept fails here, with a ``manifold
extract`` hint, instead of on the first request) → ``create_app`` → warm-up
(``--compile`` only) → live J-lens → live SAE (dashboard only) → uvicorn.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Sequence

from drowse.cli.context import CliError, effective_config, note, open_session
from drowse.cli.spec import (
    CONFIG, INJECTION, MODEL, MODEL_LOAD, PROBES, STEER, TOP_K_ALTS, Verb, arg,
    bounded_int,
)


def preflight_steering(session: Any, expression: str, model_id: str) -> Any:
    """Parse ``expression`` and compose it once against ``session``.

    Entering and leaving a steering scope resolves every atom through the
    session's own bare-atom tiers and loads every referenced artifact — exactly
    what a request would do — without authoring anything.  Returns the parsed
    :class:`~drowse.steering.terms.Steering` for ``create_app``.
    """
    from drowse.session.session import (
        ManifoldNotRegisteredError, ProfileNotRegisteredError,
    )
    from drowse.steering.grammar import parse_expr

    steering = parse_expr(expression)
    try:
        with session.steering(steering):
            pass
    except (ProfileNotRegisteredError, ManifoldNotRegisteredError) as e:
        _status, message = e.user_message()
        raise CliError(
            f"default steering {expression!r}: {message} — fit it first, e.g. "
            f"`drowse manifold extract <concept> -m {model_id}`",
            exit_code=1,
        ) from e
    return steering


def warm_up(session: Any) -> None:
    """One short stateless generation so dynamo's shape promotion fires on a
    realistic prefill before the first request (``--compile`` only — eager
    sessions have nothing to specialize)."""
    import time

    from drowse.generation.sampling import SamplingConfig

    note("Warming up compiled generation kernels...")
    start = time.monotonic()
    try:
        session.generate(
            "Please respond briefly.",
            sampling=SamplingConfig(max_tokens=32), stateless=True,
        )
    except Exception as e:  # noqa: BLE001 — never block serve startup
        note(f"  warm-up skipped: {e}")
        return
    note(f"  warmed in {time.monotonic() - start:.1f}s")


def enable_live_lens(session: Any) -> bool:
    """Stream the live J-lens readout when a compatible lens is selected.

    The persisted ``lens use`` selection is the user's choice and is never
    rewritten: an active source is used when it matches the loaded weights and
    otherwise left alone (live readout off, with a hint).  Only when nothing is
    selected does serve adopt the best compatible prepared source, in
    :func:`~drowse.lens.sources.lens_source_preference_key` order.
    """
    try:
        from drowse.lens.sources import (
            LENS_SOURCES, lens_root, lens_source_label, list_lens_sources,
            load_active_lens_source,
        )

        model_id = session.model_id
        active = load_active_lens_source(model_id)
        if active is not None:
            if not session.has_compatible_jlens():
                note(
                    f"Live J-lens readout: off — the active source "
                    f"{lens_source_label(active)} does not match the loaded "
                    f"weights (`drowse lens use {model_id} <source>` to switch)"
                )
                return False
        else:
            compatible = False
            for row in list_lens_sources(model_id):
                try:
                    session.select_jlens_source(str(row["source"]))
                    compatible = session.has_compatible_jlens()
                except Exception:  # noqa: BLE001 — try the next prepared source
                    continue
                if compatible:
                    break
            if not compatible:
                tried = load_active_lens_source(model_id)
                if tried is not None:
                    LENS_SOURCES.clear_if_active(
                        lens_root(model_id), model_id, tried["kind"], tried["name"],
                    )
                return False
        layers = session.lens.set_live(True).layers or ()
        note(f"Live J-lens readout: on ({len(layers)} fitted layers)")
        return True
    except Exception as e:  # noqa: BLE001 — never block serve startup
        note(f"Live J-lens readout: enable failed ({e})")
        return False


def best_sae_release(rows: Sequence[dict[str, Any]]) -> str | None:
    """The strongest provider default for the dashboard runtime.

    Provider-hosted, Neuronpedia-labelled, official releases win; canonical
    releases beat broad variant sets, and a curated base release beats its
    ``-all`` sibling.  The name tie-break keeps the choice stable.
    """
    candidates = [
        row for row in rows
        if row.get("source") == "saelens"
        and isinstance(row.get("release"), str)
        and bool(row["release"].strip())
    ]
    if not candidates:
        return None

    def rank(row: dict[str, Any]) -> tuple[bool, bool, bool, bool, str]:
        release = str(row["release"])
        repo_id = row.get("repo_id")
        official = isinstance(repo_id, str) and repo_id.startswith("google/")
        return (
            not official,
            not bool(row.get("neuronpedia")),
            "canonical" not in release,
            release.endswith("-all"),
            release,
        )

    return str(min(candidates, key=rank)["release"])


def enable_live_sae(session: Any) -> bool:
    """Attach the active / a prepared / the best provider SAE and go live."""
    try:
        info = session.sae_info
        if info is None:
            from drowse.sae.sources import list_sae_sources

            # Prepared artifacts are the cheapest and most intentional
            # default; the active one first.
            rows = sorted(
                list_sae_sources(session.model_id),
                key=lambda row: (not bool(row.get("active")), row.get("kind") != "local"),
            )
            for row in rows:
                source = row.get("source")
                if not isinstance(source, str) or not source:
                    continue
                try:
                    info = session.load_sae(source.removeprefix("saelens:"))
                except Exception:  # noqa: BLE001 — try the next prepared source
                    continue
                break
        if info is None:
            from drowse.sae.backend import list_sae_releases

            release = best_sae_release(list_sae_releases(session.model_id))
            if release is None:
                return False
            info = session.load_sae(release)
        state = session.sae.set_live(True)
        note(f"Live SAE readout: on ({info.get('release', 'active')} at L{state.layer})")
        return True
    except Exception as e:  # noqa: BLE001 — never block serve startup
        note(f"Live SAE readout: enable failed ({e})")
        return False


def run(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        raise CliError(
            "server dependencies missing; run `pip install --upgrade drowse.ai`",
            exit_code=1,
        ) from None
    from drowse.server.app import is_loopback_host

    effective_config(args)
    api_key = args.api_key or os.environ.get("DROWSE_API_KEY")
    if not is_loopback_host(args.host) and not api_key:
        raise CliError("non-loopback binding requires --api-key or DROWSE_API_KEY")
    if not args.model:
        raise CliError(
            "model required: pass a HuggingFace repo id (e.g. `drowse serve "
            "google/gemma-2-2b-it`) or a -c setup.yaml with a `model:` field"
        )
    if args.steer:
        from drowse.steering.grammar import parse_expr

        parse_expr(args.steer)  # a typo fails before the model load

    with open_session(args, roster=True) as session:
        steering = (
            preflight_steering(session, args.steer, args.model)
            if args.steer else None
        )
        from drowse.server import create_app
        from drowse.server.ws_stream import MAX_WS_MESSAGE_BYTES

        web = not args.no_web
        app = create_app(
            session, default_steering=steering, cors_origins=args.cors or None,
            api_key=api_key, web=web,
        )
        if args.compile:
            warm_up(session)
        enable_live_lens(session)
        # API-only serves neither acquire the optional SAE dependency nor
        # download provider weights implicitly.
        if web:
            enable_live_sae(session)

        base = f"http://{args.host}:{args.port}"
        print(f"\nServing on {base}")
        print(f"OpenAI-compatible:  {base}/v1")
        print(f"Ollama-compatible:  {base}/api")
        print(f"API docs:           {base}/docs")
        if args.port != 11434:
            print("Tip: for drop-in Ollama compatibility, run with `--port 11434`.")
        uvicorn.run(
            app, host=args.host, port=args.port, log_level="info",
            ws_max_size=MAX_WS_MESSAGE_BYTES,
        )


GROUP = Verb(
    "serve",
    "Start the OpenAI + Ollama API server + analytics dashboard at /",
    run=run,
    description="Start the OpenAI + Ollama compatible API server and the "
                "dashboard at /.",
    args=(
        MODEL.but(nargs="?", default=None,
                  help="HuggingFace model ID or local path (optional when a "
                       "-c YAML supplies `model:`)"),
        MODEL_LOAD,
        PROBES,
        arg("-H", "--host", default="127.0.0.1",
            help="Bind address (non-loopback requires an API key)"),
        arg("-P", "--port", type=bounded_int(1, 65535), default=8000,
            help="Bind port"),
        STEER.but(help="Default steering expression (overrides YAML "
                       'vectors:), e.g. "0.5 honest + 0.3 warm"'),
        arg("-C", "--cors", action="append", default=[], metavar="ORIGIN",
            help="CORS allowed origin (repeatable)"),
        arg("-k", "--api-key", default=None, metavar="KEY",
            help="Require Bearer token auth; falls back to $DROWSE_API_KEY"),
        arg("--no-web", action="store_true",
            help="Skip the dashboard mount at / (API-only mode)"),
        INJECTION,
        TOP_K_ALTS,
        CONFIG,
    ),
)
