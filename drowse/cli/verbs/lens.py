"""``drowse lens <verb>`` — the per-model Jacobian lens: fit, the
local/external source lifecycle, and the two readouts (``top``,
``decompose``)."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drowse.cli.context import (
    CliError, emit_json, note, open_session, parse_layers, progress, user_errors,
)
from drowse.cli.spec import (
    FORCE, JSON, MODEL, MODEL_LOAD, MODEL_REQ, YES, OneOf, Verb, arg,
    layers_arg, positive_int,
)
from drowse.lens.key import FITTABLE_KINDS, LensKey  # torch-free

if TYPE_CHECKING:
    from drowse.lens.corpus import LensCorpus


#: ``--kind`` → the fit's backward rules, one entry per fittable
#: :class:`~drowse.lens.key.LensKey` kind.  The lower-case letter is also the
#: local artifact name (``local:r`` / ``local:j``); upper-cased, it is the
#: kind the estimator policy is recorded under.
LENS_KINDS: dict[str, str] = {
    kind.lower(): "relp" if LensKey(kind).relp else "standard"
    for kind in FITTABLE_KINDS
}

_SOURCES_HELP = (
    "local:r, local:j (or local:<kind>-<tag>), or a fetched binding "
    "(neuronpedia, workspace-r, workspace-j)"
)


# ---------------------------------------------------------------------------
# Corpora (shared with ``sae train``)
# ---------------------------------------------------------------------------

def read_text_corpus(path: Path, n: int) -> list[str]:
    """Up to ``n`` documents of a local corpus file
    (:func:`~drowse.lens.corpus.read_corpus_file`), as CLI errors."""
    from drowse.lens.corpus import read_corpus_file

    if not path.exists():
        raise CliError(f"corpus file not found: {path}")
    docs = read_corpus_file(path, n)
    if not docs:
        raise CliError(f"corpus {path} has no non-empty documents")
    return docs


def _pinned_hub_corpus(args: argparse.Namespace) -> "LensCorpus | None":
    """The Hub corpus ``lens fit`` streams (``--hf-corpus``, repeated = a
    weighted mix; unset = the default web-text sample), pinned once for both
    the no-op proof and the stream.  ``None`` for ``--corpus FILE``."""
    from drowse.lens.corpus import parse_lens_corpus, pin_lens_corpus

    if args.corpus is not None:
        return None
    with user_errors(ValueError, exit_code=2):
        corpus = parse_lens_corpus(args.hf_corpus)
    return pin_lens_corpus(corpus)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def fit_covers_layers(sidecar: dict[str, Any], requested: "list[int] | str | None") -> bool:
    """Whether a fitted sidecar's source layers cover the requested set."""
    from drowse.lens.store import lens_layers

    fitted = set(lens_layers(sidecar))
    if not fitted:
        return False
    if isinstance(requested, list):
        return fitted >= set(requested)
    n_layers = sidecar.get("model_layer_count")
    if not isinstance(n_layers, int) or n_layers < 2:
        return False
    if requested is None or requested == "all":
        return fitted >= set(range(n_layers - 1))
    if requested == "workspace":
        from drowse.model.arch import workspace_layer_indices

        return fitted >= set(workspace_layer_indices(range(n_layers - 1), n_layers))
    return False


def lens_fit_is_noop(
    args: argparse.Namespace,
    requested: "list[int] | str | None",
    *,
    docs: list[str] | None = None,
    hf_corpus: "LensCorpus | None" = None,
) -> bool:
    """Prove the requested fit already exists, without loading weights.

    Runs under the fit's own lock, so the proof, the checkpoint reap, and the
    report are one transaction.  The corpus identity is ``docs`` for a file
    corpus, or the pinned ``hf_corpus`` spec plus prompt count before
    streaming.
    """
    from drowse.lens.jlens import DEFAULT_SEQ_LEN
    from drowse.lens.store import (
        lens_artifact_size, lens_estimator_policy, lens_fit_lock, lens_layers,
        lens_manifest_path, lens_payloads_match, load_local_lens_sidecar,
        remove_subsumed_lens_checkpoint,
    )
    from drowse.model.fingerprint import model_source_fingerprint

    if args.force:
        return False
    name = args.kind
    with lens_fit_lock(args.model, name):
        sidecar = load_local_lens_sidecar(args.model, name)
        if sidecar is None or sidecar.get("estimator_policy") != lens_estimator_policy(
            name.upper(),
        ):
            return False
        source_fp = model_source_fingerprint(args.model, quantize=args.quantize, device=args.device)
        if (
            source_fp is None
            or sidecar.get("model_source_fingerprint") != source_fp
            or sidecar.get("seq_len") != (args.seq_len or DEFAULT_SEQ_LEN)
            or not fit_covers_layers(sidecar, requested)
        ):
            return False
        if docs is not None:
            # The same identity ``DrowseSession.fit_jlens`` records.
            raw_sha = hashlib.sha256(repr(docs).encode("utf-8")).hexdigest()
            if (
                sidecar.get("raw_corpus_sha256") != raw_sha
                or sidecar.get("raw_prompt_count") != len(docs)
            ):
                return False
        elif hf_corpus is None or (
            sidecar.get("corpus_spec") != hf_corpus.spec
            or sidecar.get("raw_prompt_count") != int(args.prompts)
        ):
            return False
        usable = int(sidecar.get("usable_prompt_count", -1))
        if usable < 0 or int(sidecar.get("n_prompts", -1)) < usable:
            return False
        if not lens_payloads_match(args.model, sidecar, name=name):
            return False
        remove_subsumed_lens_checkpoint(args.model, name=name, verified_final_sidecar=sidecar)
        size_mb = lens_artifact_size(args.model, name=name) / 1024**2
        note("Already fitted for this corpus and exact model source — nothing to do.")
        print(
            f"Fitted Jacobian lens: {len(lens_layers(sidecar))} layers, "
            f"{sidecar.get('n_prompts', usable)} prompts, d_model={sidecar.get('d_model', '?')}"
        )
        print(f"Artifact: {lens_manifest_path(args.model, name)} ({size_mb:.0f} MB across layer shards)")
        return True


def run_fit(args: argparse.Namespace) -> None:
    from drowse.lens.corpus import stream_lens_corpus
    from drowse.lens.store import lens_artifact_size, lens_manifest_path

    requested = parse_layers(args.layers)
    if requested == "sample":
        raise CliError(
            "--layers sample is not a wall-time optimization; use --layers "
            "workspace or an explicit comma-separated band"
        )
    rules = LENS_KINDS[args.kind]
    hf_corpus = _pinned_hub_corpus(args)
    if hf_corpus is None:
        path = Path(args.corpus)
        docs = read_text_corpus(path, args.prompts)
        spec = f"file:{path.name}"
        if lens_fit_is_noop(args, requested, docs=docs):
            return
    else:
        if lens_fit_is_noop(args, requested, hf_corpus=hf_corpus):
            return
        note(f"Streaming {args.prompts} documents from {hf_corpus}...")
        docs, spec = stream_lens_corpus(args.prompts, hf_corpus)

    with open_session(args) as session:
        note(f"Fitting the {'R-lens (RelP backward rules)' if rules == 'relp' else 'standard J-lens'}"
             f"{' from zero (-f)' if args.force else ''}.")
        lens = session.fit_jlens(
            docs, corpus_spec=spec, source_layers=requested,
            dim_batch=args.dim_batch, prompt_batch=args.prompt_batch,
            seq_len=args.seq_len, force=args.force,
            checkpoint_every=args.checkpoint_every, backward_rules=rules,
            on_progress=progress,
        )
    size_mb = lens_artifact_size(args.model, name=args.kind) / 1024**2
    print(
        f"Fitted Jacobian lens: {len(lens.source_layers)} layers, "
        f"{lens.n_prompts} prompts, d_model={lens.d_model}"
    )
    print(f"Artifact: {lens_manifest_path(args.model, args.kind)} ({size_mb:.0f} MB across layer shards)")


# ---------------------------------------------------------------------------
# source lifecycle
# ---------------------------------------------------------------------------

def _lens_source(model_id: str, source: str | None) -> Any:
    """``local:<kind>[-<tag>]`` / a binding name / ``None`` (the active
    output-side source) → a :class:`~drowse.base.source_registry.SourceRef`."""
    from drowse.lens.sources import LENS_SOURCES

    if source is None:
        ref = LENS_SOURCES.active(model_id, "output")
        if ref is None:
            raise CliError(f"no lens source for {model_id}", exit_code=1)
        return ref
    with user_errors(ValueError, exit_code=2):
        return LENS_SOURCES.parse(source)


def run_fetch(args: argparse.Namespace) -> None:
    from drowse.lens.sources import LENS_PROVIDERS, fetch_lens_source

    known = sorted(LENS_PROVIDERS)
    if args.source not in known:
        raise CliError(f"source must be one of {', '.join(known)}")
    note(f"Fetching the {args.source} lens for {args.model} into the Hugging Face cache...")
    binding = fetch_lens_source(
        args.model, args.source, repo_id=args.repo, revision=args.revision, force=args.force,
    )
    if args.json_output:
        emit_json(binding.to_json())
        return
    bound = binding.files[binding.key]
    print(
        f"Fetched {binding.name} for {args.model}: {binding.key.label} lens, "
        f"{len(bound.layers)} layers, {binding.n_prompts} prompts"
    )
    print(f"  provider: {binding.repo_id}@{binding.repo_revision}")
    print(f"  checkpoint: {bound.path}")
    note("Provider payload remains in the Hugging Face cache; binding is active.")


def run_ls(args: argparse.Namespace) -> None:
    from drowse.lens.sources import list_lens_sources

    rows = list_lens_sources(args.model)
    if args.json_output:
        emit_json(rows)
        return
    if not rows:
        print(f"No lens sources for {args.model}.")
    for row in rows:
        print(f"{'*' if row['active'] else ' '} {row['source']}")


def run_use(args: argparse.Namespace) -> None:
    from drowse.lens.sources import use_lens_source

    with user_errors(ValueError, exit_code=2):
        use_lens_source(args.model, args.source)
    print(f"Active lens for {args.model}: {args.source}")


def run_show(args: argparse.Namespace) -> None:
    from drowse.lens.sources import LENS_SOURCES, load_lens_source_sidecar
    from drowse.lens.store import lens_artifact_size, lens_layers

    ref = _lens_source(args.model, args.source)
    label = LENS_SOURCES.format(ref)
    sidecar = load_lens_source_sidecar(args.model, ref)
    if sidecar is None:
        raise CliError(f"no lens source {label} for {args.model}", exit_code=1)
    path = LENS_SOURCES.path_of(args.model, ref)
    size_mb = lens_artifact_size(args.model, name=ref.name) / 1024**2 if ref.local else None
    is_active = ref in {LENS_SOURCES.active(args.model, side) for side in ("output", "input")}
    if args.json_output:
        emit_json({
            "model": args.model, "source": label, "path": str(path),
            "size_mb": None if size_mb is None else round(size_mb, 1),
            "active": is_active, **sidecar,
        })
        return
    layers = lens_layers(sidecar)
    skip_first = (
        sidecar["estimator_policy"]["skip_first_positions"] if ref.local
        else sidecar.get("skip_first_positions", "?")
    )
    print(f"{sidecar['kind']}-lens {label} for {args.model}{' (active)' if is_active else ''}")
    print(f"  layers:   {layers[0]}..{layers[-1]} ({len(layers)})")
    print(f"  d_model:  {sidecar['d_model']}")
    print(f"  prompts:  {sidecar.get('n_prompts', '?')}")
    corpus = f"  corpus:   {sidecar.get('corpus_spec', '?')}"
    if sidecar.get("corpus_sha256"):
        corpus += f" (sha256 {str(sidecar['corpus_sha256'])[:12]}…)"
    print(corpus)
    print(f"  seq_len:  {sidecar.get('seq_len', '?')}, skip_first: {skip_first}")
    if size_mb is None:
        meta = sidecar["_source"]
        print(f"  source:   {meta['provider']} ({meta['repo_id']}@{meta['repo_revision']})")
        print(f"  binding:  {path}")
        print("  payload:  Hugging Face cache (provider-owned)")
    else:
        print(f"  artifact: {path} ({size_mb:.0f} MB across layer shards)")


def run_rm(args: argparse.Namespace) -> None:
    from drowse.lens.sources import LENS_SOURCES, remove_lens_source

    ref = _lens_source(args.model, args.source)
    label = LENS_SOURCES.format(ref)
    if not args.yes:
        what = "local lens" if ref.local else "external binding"
        raise CliError(f"refusing to remove {what} {label}; pass -y/--yes to confirm")
    if not remove_lens_source(args.model, ref):
        raise CliError(f"no lens source {label} for {args.model}", exit_code=1)
    if ref.local:
        print(f"Removed Drowse-owned lens {label} for {args.model}.")
    else:
        print(f"Forgot {label} binding; the Hugging Face cache was not modified.")


# ---------------------------------------------------------------------------
# readouts
# ---------------------------------------------------------------------------

def run_top(args: argparse.Namespace) -> None:
    layers = parse_layers(args.layers)
    positions = args.position or [-1]
    with open_session(args) as session:
        lens = session.jlens
        if lens is None:
            raise CliError(
                f"no lens for {args.model} — run `drowse lens fetch {args.model}` "
                f"or `drowse lens fit {args.model}`",
                exit_code=1,
            )
        per_layer, aggregate = session.jlens_readout(
            args.prompt, layers=layers if layers is not None else list(lens.source_layers),
            positions=args.position, top_k=args.top_k, aggregate=True,
        )
    if args.json_output:
        emit_json({
            "model": args.model,
            "prompt": args.prompt,
            "positions": positions,
            # Per-layer softmax → mean-probability strength and a
            # probability-weighted depth center of mass, strength-descending.
            "aggregate": [
                [
                    {"token": t, "strength": round(s, 6), "com": round(c, 4), "spread": round(sp, 4)}
                    for t, s, c, sp in rows
                ]
                for rows in aggregate
            ],
            # Per-layer rows carry the unit the aggregate averages: the
            # per-layer readout probability p_l.
            "layers": {
                str(layer): [[{"token": t, "strength": round(p, 6)} for t, p in row] for row in rows]
                for layer, rows in per_layer.items()
            },
        })
        return
    for i, pos in enumerate(positions):
        if len(positions) > 1:
            print(f"\nposition {pos}:")
        print("  aggregate (all displayed layers):")
        for t, s, c, sp in aggregate[i]:
            print(f"    {t.strip() or repr(t):<20} strength {s:.3f}   com {c:.2f} ±{sp:.2f}")
        print("  per-layer:")
        for layer in sorted(per_layer):
            print(f"  L{layer:>3}  {'  '.join(t.strip() or repr(t) for t, _ in per_layer[layer][i])}")


def run_decompose(args: argparse.Namespace) -> None:
    layers = parse_layers(args.layers)
    with open_session(args) as session:
        out = session.jspace_decompose(args.selector, k=args.top_k, layers=layers)
    mean_share = sum(share for share, _ in out.values()) / len(out)
    if args.json_output:
        emit_json({
            "selector": args.selector, "model": args.model, "k": args.top_k,
            "mean_share": round(mean_share, 4),
            "layers": {
                str(layer): {
                    "share": round(share, 4),
                    "tokens": [{"token": t, "coeff": round(c, 4)} for t, c in tokens],
                }
                for layer, (share, tokens) in out.items()
            },
        })
        return
    print(f"J-space share of '{args.selector}' (k={args.top_k}):")
    print(f"  mean over {len(out)} layers: {mean_share:.1%}")
    for layer in sorted(out):
        share, tokens = out[layer]
        print(f"  L{layer:>3}  {share:>6.1%}  {'  '.join(t.strip() or repr(t) for t, _ in tokens[:6])}")


_MODEL_POS = arg("model", help="Model ID")
_SOURCE_OPT = arg("source", nargs="?", default=None, help=f"{_SOURCES_HELP} (default: active)")

GROUP = Verb(
    "lens",
    "Jacobian lens (fit/fetch/ls/show/use/top/decompose/rm)",
    description="Per-model Jacobian lens: fit the residual→output transport, read "
                "out what intermediate activations are disposed to say, and split "
                "directions into their verbalizable (J-space) component.",
    verbs=(
        Verb(
            "fit", "Fit the model's Jacobian lens over a text corpus",
            run=run_fit,
            description=(
                "Fit J_l = E[dh_final/dh_l] per source layer over a text corpus: "
                "one forward + ceil(d_model/dim_batch) backward passes per document. "
                "--kind r (default) fits the RelP R-lens at local:r, --kind j the "
                "standard J-lens at local:j. Interrupted fits checkpoint and "
                "resume; an exact repeat is proven without loading the model."
            ),
            args=(
                MODEL,
                OneOf((
                    arg("--corpus", default=None, metavar="FILE",
                        help="One document per line (JSONL 'text' lines work too). "
                             "Default: stream FineWeb-Edu sample-10BT (needs "
                             "`pip install 'drowse.ai[hf]'`)"),
                    arg("--hf-corpus", action="append", default=None,
                        metavar="REPO[:CONFIG][@REV][,KEY=VALUE...]",
                        help="Stream a Hub dataset's train split instead, pinned to "
                             "a commit. Options: text_field=F, files=K (round-robin "
                             "over K data files), seed=S; repeat with weight=W to mix"),
                )),
                arg("--kind", choices=sorted(LENS_KINDS), default="r",
                    help="Estimator: r = RelP R-lens (default), j = standard J-lens"),
                arg("--prompts", type=positive_int, default=100, metavar="N",
                    help="Corpus documents to average over (default 100; the paper used 1000)"),
                arg("--seq-len", type=positive_int, default=None, metavar="T",
                    help="Truncate each document to T tokens (default 128)"),
                arg("--dim-batch", type=positive_int, default=None, metavar="K",
                    help="Output dims per backward pass (default 8; halves on OOM)"),
                arg("--prompt-batch", type=positive_int, default=None, metavar="B",
                    help="Prompts per autograd graph (default 4; 2 on MPS)"),
                arg("--checkpoint-every", type=positive_int, default=None, metavar="N",
                    help="Resumable checkpoint every N usable prompts (default 25)"),
                layers_arg("Fit only these source layers (default: every layer below "
                           "the final one); workspace = the 40-90%% band, the one "
                           "real wall-time lever"),
                FORCE.but(help="Restart from zero (default: resume a matching partial fit)"),
                MODEL_LOAD,
            ),
        ),
        Verb(
            "fetch", "Fetch an external lens into its provider cache",
            run=run_fetch,
            description="Pin an external lens's repository and base-model revisions "
                        "and write only a small binding; the payload stays in the "
                        "Hugging Face cache. No model load.",
            args=(
                _MODEL_POS,
                arg("source", nargs="?", default="neuronpedia",
                    help="neuronpedia (default), workspace-r (RelP), or workspace-j"),
                arg("--revision", default="main", help="Provider revision to resolve and pin"),
                arg("--repo", default=None, metavar="REPO",
                    help="Override the provider repository"),
                FORCE.but(help="Refresh the binding"),
                JSON,
            ),
        ),
        Verb("ls", "List local and external lens sources for a model", run=run_ls,
             args=(_MODEL_POS, JSON)),
        Verb("show", "Show the active or selected lens source", run=run_show,
             args=(_MODEL_POS, _SOURCE_OPT, JSON)),
        Verb("use", "Select an already available lens source", run=run_use,
             args=(_MODEL_POS, arg("source", help=_SOURCES_HELP))),
        Verb(
            "top", "Lens readout on a prompt: top vocabulary tokens per layer",
            run=run_top,
            description="One forward pass on a raw prompt (no chat template), each "
                        "selected layer decoded through softmax(W_U · norm(J_l h)), "
                        "plus the layer-aggregated view.",
            args=(
                MODEL,
                arg("prompt", help="Raw prompt to read out"),
                arg("-k", "--top-k", type=positive_int, default=8, metavar="K",
                    help="Tokens per (layer, position) readout (default 8)"),
                layers_arg("Layer indices or a mode (default: every fitted layer)",
                           metavar="L1,L2,...|sample|workspace|all"),
                arg("--position", type=int, action="append", default=None, metavar="P",
                    help="Token position to read (repeatable; negative from the end; "
                         "default: final)"),
                MODEL_LOAD, JSON,
            ),
        ),
        Verb(
            "decompose", "Split a steerable direction into its J-space component",
            run=run_decompose,
            description="Best sparse nonnegative combination of k J-lens token "
                        "vectors per layer: the variance share (how verbalizable "
                        "the direction is) and the contributing tokens.",
            args=(
                arg("selector", help="Steerable direction (concept, ns/name, or jlens/<word>)"),
                MODEL_REQ,
                arg("-k", "--top-k", type=positive_int, default=16, metavar="K",
                    help="Sparsity budget (default 16)"),
                layers_arg("Layer indices or a mode (default: every fitted layer "
                           "the direction covers)"),
                MODEL_LOAD, JSON,
            ),
        ),
        Verb(
            "rm", "Remove a local lens or forget an external binding",
            run=run_rm,
            description="Delete a local lens, or forget an external binding without "
                        "touching its provider cache.",
            args=(_MODEL_POS, _SOURCE_OPT, YES),
        ),
    ),
)
