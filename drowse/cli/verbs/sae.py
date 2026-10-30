"""``drowse sae <verb>`` — per-model sparse-autoencoder sources: train a
local one, fetch a provider one, then the shared ls/show/use/rm lifecycle.
Sources are ``local:NAME`` or ``saelens:RELEASE``."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from drowse.cli.context import CliError, emit_json, note, open_session, progress, user_errors
from drowse.cli.spec import (
    FORCE, JSON, MODEL, MODEL_LOAD, YES, Verb, arg, nonnegative_int, positive_int,
)

_SOURCES_HELP = "local:NAME or saelens:RELEASE"


def _sae_source(model_id: str, source: str | None) -> Any:
    """``local:NAME`` / ``saelens:RELEASE`` / ``None`` (the active source) →
    a :class:`~drowse.base.source_registry.SourceRef`."""
    from drowse.sae.sources import SAE_SOURCES

    if source is None:
        ref = SAE_SOURCES.active(model_id)
        if ref is None:
            raise CliError(f"no SAE source for {model_id}", exit_code=1)
        return ref
    with user_errors(ValueError, exit_code=2):
        return SAE_SOURCES.parse(source)


def run_train(args: argparse.Namespace) -> None:
    from drowse.lens.corpus import DEFAULT_LENS_CORPUS, read_corpus_file, stream_lens_corpus
    from drowse.sae.backend import select_runtime_layer

    if args.learning_rate <= 0 or args.l1 < 0 or args.dead_threshold < 0:
        raise CliError("--learning-rate must be > 0 and the sparsity values >= 0")
    n_docs = max(1, math.ceil(args.tokens / args.seq_len))
    if args.corpus is not None:
        path = Path(args.corpus)
        docs, spec = read_corpus_file(path, n_docs), f"file:{path.name}"
    else:
        note(f"Streaming {n_docs} documents from {DEFAULT_LENS_CORPUS}...")
        docs, spec = stream_lens_corpus(n_docs)
    with open_session(args) as session:
        n_layers = len(session.layers)
        layer = select_runtime_layer(frozenset(range(n_layers)), n_layers, args.layer)
        result = session.sae.train(
            args.name, docs, layer=layer, corpus_spec=spec,
            tokens=args.tokens, seq_len=args.seq_len, batch_size=args.batch_size,
            d_sae=args.width, expansion=args.expansion,
            learning_rate=args.learning_rate, l1_coefficient=args.l1,
            dead_feature_threshold=args.dead_threshold, seed=args.seed,
            force=args.force, on_progress=progress,
        )
    if args.json_output:
        emit_json(result)
        return
    metrics = result["metrics"]
    print(
        f"Trained {result['source']} for {args.model}: L{layer}, "
        f"{metrics['d_sae']} features, {metrics['tokens_trained']:,} tokens"
    )
    print(f"Artifact: {result['artifact']}")


def run_fetch(args: argparse.Namespace) -> None:
    """Resolve a provider release and pin its binding without loading the
    base model: compatibility comes from the release entry, layer coverage
    and decoder width from the published config."""
    from drowse.model.arch import config_model_shape
    from drowse.sae.backend import resolve_sae_backend
    from drowse.sae.sources import save_sae_metadata

    ref = _sae_source(args.model, args.source)
    if ref.local:
        raise CliError("source must be saelens:RELEASE")
    release = ref.name
    with user_errors(OSError, ValueError, exit_code=2):
        n_layers, hidden_dim = config_model_shape(args.model)
    # Announce before the provider download (silent with progress bars off).
    note(f"Fetching {args.source} into the Hugging Face cache...")
    resolved = resolve_sae_backend(
        release, model_id=args.model, n_layers=n_layers, hidden_dim=hidden_dim,
        layer=args.layer, revision=args.revision,
    )
    layer, width, binding = resolved.layer, resolved.width, resolved.binding
    save_sae_metadata(args.model, release, binding)
    if args.json_output:
        emit_json({
            "model": args.model, "source": args.source, "layer": layer,
            "width": width, "revision": binding["revision"], "repo_id": binding["repo_id"],
        })
        return
    print(f"Fetched {args.source} for {args.model}: L{layer}, {width} features")
    note("Provider weights remain in the Hugging Face cache; binding is active.")


def run_ls(args: argparse.Namespace) -> None:
    from drowse.sae.sources import list_sae_sources

    rows = list_sae_sources(args.model)
    if args.json_output:
        emit_json(rows)
        return
    if not rows:
        print(f"No SAE sources for {args.model}.")
    for row in rows:
        print(f"{'*' if row['active'] else ' '} {row['source']}  L{row['layer']}  {row['features']} features")


def run_show(args: argparse.Namespace) -> None:
    from drowse.sae.sources import SAE_SOURCES, load_sae_source

    ref = _sae_source(args.model, args.source)
    source = SAE_SOURCES.format(ref)
    payload = load_sae_source(args.model, ref)
    if payload is None:
        raise CliError(f"no SAE source {source} for {args.model}", exit_code=1)
    is_active = SAE_SOURCES.active(args.model) == ref
    if args.json_output:
        emit_json({"model": args.model, "source": source, "active": is_active, **payload})
        return
    print(f"SAE {source} for {args.model}{' (active)' if is_active else ''}")
    print(f"  layer:    {payload['layer']}")
    print(f"  features: {payload.get('d_sae', payload.get('width'))}")
    if ref.local:
        print(f"  tokens:   {payload['tokens_trained']}")
        print(f"  corpus:   {payload['corpus_spec']}")
    else:
        print(f"  revision: {payload['revision']}")
        print("  payload:  Hugging Face cache (provider-owned)")


def run_use(args: argparse.Namespace) -> None:
    from drowse.sae.sources import use_sae_source

    with user_errors(ValueError, exit_code=2):
        use_sae_source(args.model, args.source)
    print(f"Active SAE for {args.model}: {args.source}")


def run_rm(args: argparse.Namespace) -> None:
    from drowse.sae.sources import SAE_SOURCES, remove_sae_source

    ref = _sae_source(args.model, args.source)
    source = SAE_SOURCES.format(ref)
    if not args.yes:
        what = "local SAE" if ref.local else "external binding"
        raise CliError(f"refusing to remove {what} {source}; pass -y/--yes to confirm")
    if not remove_sae_source(args.model, ref):
        raise CliError(f"no SAE source {source} for {args.model}", exit_code=1)
    if ref.local:
        print(f"Removed Drowse-owned SAE {source}.")
    else:
        print(f"Forgot {source} binding; the Hugging Face cache was not modified.")


_MODEL_POS = arg("model", help="Model ID")
_SOURCE_OPT = arg("source", nargs="?", default=None, help=f"{_SOURCES_HELP} (default: active)")

GROUP = Verb(
    "sae",
    "Sparse autoencoders (train/fetch/ls/show/use/rm)",
    description="Train, fetch, select, and inspect per-model sparse-autoencoder sources.",
    verbs=(
        Verb(
            "train", "Train a local residual-post sparse autoencoder", run=run_train,
            args=(
                MODEL,
                arg("name", help="Local SAE name"),
                arg("--corpus", default=None, metavar="FILE",
                    help="One text document per line (default: stream FineWeb-Edu)"),
                arg("--layer", type=nonnegative_int, default=None,
                    help="Residual-post layer (default: nearest 65%% depth in the workspace band)"),
                arg("--tokens", type=positive_int, default=1_000_000, metavar="N"),
                arg("--seq-len", type=positive_int, default=128, metavar="T"),
                arg("--batch-size", type=positive_int, default=8, metavar="B"),
                arg("--width", type=positive_int, default=None, metavar="N"),
                arg("--expansion", type=positive_int, default=8, metavar="K"),
                arg("--learning-rate", type=float, default=3e-4, metavar="LR"),
                arg("--l1", type=float, default=1e-3, metavar="LAMBDA"),
                arg("--dead-threshold", type=float, default=1e-6, metavar="T"),
                arg("--seed", type=int, default=0),
                FORCE, MODEL_LOAD, JSON,
            ),
        ),
        Verb(
            "fetch", "Fetch an external SAE into its provider cache", run=run_fetch,
            description="Pure IO like `lens fetch`: provider weights stay in the "
                        "Hugging Face cache and Drowse pins a small binding. The "
                        "base model is never loaded.",
            args=(
                _MODEL_POS,
                arg("source", help="saelens:RELEASE"),
                arg("--layer", type=nonnegative_int, default=None,
                    help="Explicit covered hook layer (default: nearest 65%% depth)"),
                arg("--revision", default=None, metavar="REV",
                    help="Provider revision to resolve and pin (default: the release's own)"),
                JSON,
            ),
        ),
        Verb("ls", "List local and external SAE sources for a model", run=run_ls,
             args=(_MODEL_POS, JSON)),
        Verb("show", "Show the active or selected SAE source", run=run_show,
             args=(_MODEL_POS, _SOURCE_OPT, JSON)),
        Verb("use", "Select an already available SAE source", run=run_use,
             args=(_MODEL_POS, arg("source", help=_SOURCES_HELP))),
        Verb("rm", "Remove a local SAE or forget an external binding", run=run_rm,
             args=(_MODEL_POS, _SOURCE_OPT, YES)),
    ),
)
