"""``drowse experiment <verb>`` — repeatable research runs: an alpha-grid
fan, transcript replay, and the behavior-manifold naturalness eval."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from drowse.cli.context import (
    CliError, check_steering, effective_config, emit_json, open_session, progress,
)
from drowse.cli.spec import (
    CONFIG, INJECTION, JSON, MODEL, MODEL_LOAD, PROBES, STEER, TOP_K_ALTS, Verb,
    arg, positive_int,
)

# ---------------------------------------------------------------------------
# fan
# ---------------------------------------------------------------------------

def parse_grid(raw_terms: list[str]) -> dict[str, list[float]]:
    """``CONCEPT=ALPHAS`` terms → ``{concept: alphas}`` (comma lists,
    ``start:stop:step``, or ``linspace(a, b, n)``)."""
    from drowse.cli.alpha_grid import AlphaListError, parse_alpha_list

    grid: dict[str, list[float]] = {}
    for raw in raw_terms:
        name, sep, alpha_text = raw.partition("=")
        name = name.strip()
        if not sep or not name:
            raise CliError(f"grid term must be CONCEPT=ALPHAS, got {raw!r}")
        try:
            grid[name] = [float(a) for a in parse_alpha_list(alpha_text)]
        except AlphaListError as e:
            raise CliError(f"{name}: {e}") from e
    return grid


def run_fan(args: argparse.Namespace) -> None:
    grid = parse_grid(args.grid)  # typos fail before the model load
    effective_config(args, default_max_tokens=256)
    check_steering(args.steer)
    total = math.prod(len(alphas) for alphas in grid.values())

    def report(idx: int, result: Any, alphas: dict[str, float]) -> None:
        row = ", ".join(f"{k}={v:+.3f}" for k, v in alphas.items())
        progress(f"{idx + 1}/{total}: {row} ({result.token_count} tokens)")

    with open_session(args, roster=True) as session:
        runset = session.generate_sweep(
            args.prompt, grid, base_steering=args.steer, stateless=False,
            on_result=report,
        )
    if args.json_output:
        emit_json(runset.to_dict())
        return
    print(f"experiment fan: {len(runset)} run(s)")
    for idx, result in enumerate(runset):
        node_id = runset.node_ids[idx] if idx < len(runset.node_ids) else None
        row = runset.grid[idx] if idx < len(runset.grid) else {}
        node = f" node={node_id[:8]}" if node_id else ""
        print(
            f"{idx:>3}: {', '.join(f'{k}={v:+.3f}' for k, v in row.items())}{node} "
            f"tokens={result.token_count} finish={result.finish_reason}"
        )


# ---------------------------------------------------------------------------
# transcript run
# ---------------------------------------------------------------------------

def run_transcript(args: argparse.Namespace) -> None:
    """Print :meth:`~drowse.loom.transcript.Transcript.replay`: every
    generated turn regenerated under its imported parent with its recorded
    recipe, and the recorded → replayed reading drift."""
    from drowse.loom.transcript import Transcript

    path = Path(args.path)
    if not path.is_file():
        raise CliError(f"{path}: file not found")
    transcript = Transcript.load(path)
    effective_config(args, default_max_tokens=256)
    if not args.model:
        if not transcript.model_id:
            raise CliError("model required (pass <model> or record `model_id` in the transcript)")
        args.model = transcript.model_id

    with open_session(args, roster=True) as session:
        replay = transcript.replay(session, mode="default", strict=args.strict)
        print(
            f"transcript: {len(transcript.turns)} turns loaded "
            f"(leaf: {replay.leaf_id[:8]}, {len(replay.steps)} regenerated)\n"
        )
        for idx, step in enumerate(replay.steps):
            recorded = session.tree.get(step.recorded_id)
            replayed = session.tree.get(step.replay_id)
            print(f"--- replay turn {idx} ({recorded.role}) ---")
            print(f"recorded: {recorded.text[:120]}")
            print(f"replayed: {replayed.text[:120]}")
            if step.diff.readings:
                print("  readings drift (top 5):")
                for delta in step.diff.readings[:5]:
                    print(
                        f"    {delta.name:<32}  Δ {delta.delta:+.4f}  "
                        f"(recorded {delta.a_value:+.4f} → got {delta.b_value:+.4f})"
                    )
            print()


# ---------------------------------------------------------------------------
# naturalness
# ---------------------------------------------------------------------------

def run_naturalness(args: argparse.Namespace) -> None:
    """Print :func:`~drowse.fit.naturalness.evaluate`: the steered
    trajectory's per-step Bhattacharyya distance to a behavior-space manifold
    fit over the node corpus — low is natural."""
    from drowse.fit.naturalness import evaluate, preflight

    # Everything checkable without weights fails before the model load.
    folder = preflight(args.manifold, args.steer, args.compare_linear)
    effective_config(args, default_max_tokens=128)

    with open_session(args) as session:
        report = evaluate(
            session, folder, args.prompt, args.steer, args.compare_linear,
            max_tokens=args.max_tokens, on_progress=progress,
        )
    if args.json_output:
        emit_json(report.to_dict())
        return
    print("behavior-manifold naturalness  (lower = more natural)")
    for score in report.scores:
        print(f"  {score.label:<14} mean D_B={score.mean:.4f}  max D_B={score.max:.4f}")
    if len(report.scores) == 2:
        delta = report.scores[1].mean - report.scores[0].mean
        closer = "manifold" if delta > 0 else "linear"
        print(f"  -> {closer} steering stays closer to the behavior manifold (Δmean={delta:+.4f})")


_MAX_TOKENS = "--max-tokens"

GROUP = Verb(
    "experiment", "Run and replay experiment trees",
    description="Repeatable research runs.",
    verbs=(
        Verb(
            "fan", "Run an alpha grid as one experiment", run=run_fan,
            description="Generate one loom sibling per alpha-grid row.",
            args=(
                MODEL,
                arg("prompt", help="Prompt to fan out"),
                arg("-g", "--grid", action="append", required=True, metavar="CONCEPT=ALPHAS",
                    help="Alpha grid term, repeatable (e.g. happy.sad=-0.4,0,0.4)"),
                STEER.but(help="Fixed steering composed under every grid row "
                               "(YAML vectors: is serve-only and never applies)"),
                MODEL_LOAD, PROBES,
                arg(_MAX_TOKENS, type=positive_int, default=None,
                    help="Max tokens per grid row (default: 256)"),
                JSON, INJECTION, TOP_K_ALTS, CONFIG,
            ),
        ),
        Verb(
            "transcript", "Replay saved transcripts",
            verbs=(
                Verb(
                    "run", "Replay a transcript and report probe-reading drift",
                    run=run_transcript,
                    description="Import a YAML transcript as a fresh branch, "
                                "regenerate each assistant turn under its imported "
                                "parent with the recorded recipe, and report the "
                                "per-turn reading drift.",
                    args=(
                        arg("path", help="Path to a drowse transcript YAML file"),
                        MODEL.but(nargs="?", default=None,
                                  help="Model ID (default: the transcript's model_id)"),
                        MODEL_LOAD, PROBES,
                        arg(_MAX_TOKENS, type=positive_int, default=None,
                            help="Max tokens per replayed turn (default: 256)"),
                        INJECTION,
                        CONFIG[0],
                        arg("-s", "--strict", action="store_true",
                            help="Refuse to replay when the probe roster drifted"),
                    ),
                ),
            ),
        ),
        Verb(
            "naturalness", "Score a steered generation's behavior-manifold naturalness",
            run=run_naturalness,
            description="Fit a behavior-space manifold over a manifold's node "
                        "corpus (on its authored layout, or a discover manifold's "
                        "per-model fitted one) and report the mean / max "
                        "Bhattacharyya distance of a steered trajectory to it "
                        "(low = natural).",
            args=(
                MODEL,
                arg("prompt", help="Prompt to generate from"),
                arg("--manifold", required=True, metavar="NAME",
                    help="Manifold (name, ns/name, or folder) whose node corpus "
                         "seeds the behavior manifold (>= 3 nodes)"),
                STEER.but(required=True, help="Steering expression to evaluate (e.g. 'mood%%0.5')"),
                arg("--compare-linear", action="store_true",
                    help="Also score the straight-chord baseline (-S must be one '%%' term)"),
                arg(_MAX_TOKENS, type=positive_int, default=None,
                    help="Max tokens per scored run (default: 128)"),
                MODEL_LOAD, JSON, INJECTION, CONFIG,
            ),
        ),
    ),
)
