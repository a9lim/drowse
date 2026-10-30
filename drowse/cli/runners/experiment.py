"""``drowse experiment <verb>`` runners (fan / transcript / naturalness)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import drowse.cli.runners as _pkg
from drowse.cli.parsers import _EXPERIMENT_VERBS
from drowse.cli.runners.shared import _print_verb_menu, _drowse_error_exit

#: The one nested verb under ``experiment transcript`` — its own menu table.
_TRANSCRIPT_VERBS = [
    ("run", "Replay a transcript on the current session"),
]


@_drowse_error_exit
def _run_experiment(args: argparse.Namespace) -> None:
    """Dispatch ``drowse experiment <verb>``."""
    cmd = getattr(args, "experiment_cmd", None)
    if cmd is None:
        _print_verb_menu("experiment", _EXPERIMENT_VERBS)
        sys.exit(0)
    # Registered-subparser invariant: argparse rejects an unknown verb.
    _EXPERIMENT_RUNNERS[cmd](args)


def _run_experiment_naturalness(args: argparse.Namespace) -> None:
    """Score a steered generation's behavior-manifold naturalness."""
    import json as _json

    from drowse.fit.naturalness import evaluate, preflight

    _pkg._load_effective_config(args, default_max_tokens=128)
    # Reject a bad manifold / --compare-linear expression before the model load.
    folder = preflight(args.manifold, args.steer, args.compare_linear)
    _pkg._print_startup(args)
    session = _pkg._make_session(args)
    _pkg._print_model_info(session)
    report = evaluate(
        session, folder, args.prompt, args.steer, args.compare_linear,
        max_tokens=args.max_tokens, on_progress=print,
    )
    if args.json_output:
        print(_json.dumps(report.to_dict(), indent=2))
        return
    print()
    print("behavior-manifold naturalness  (lower = more natural)")
    for score in report.scores:
        print(
            f"  {score.label:<14} mean D_B={score.mean:.4f}  "
            f"max D_B={score.max:.4f}"
        )
    if len(report.scores) == 2:
        delta = report.scores[1].mean - report.scores[0].mean
        verdict = (
            "manifold steering stays closer to the behavior manifold"
            if delta > 0 else
            "linear steering stays closer to the behavior manifold"
        )
        print(f"  -> {verdict} (Δmean={delta:+.4f})")


def _run_experiment_transcript(args: argparse.Namespace) -> None:
    """Dispatch ``drowse experiment transcript <verb>``."""
    cmd = getattr(args, "transcript_cmd", None)
    if cmd is None:
        _print_verb_menu("experiment transcript", _TRANSCRIPT_VERBS)
        sys.exit(0)
    # Registered-subparser invariant: argparse rejects an unknown verb.
    _TRANSCRIPT_RUNNERS[cmd](args)


def _parse_grid_terms(raw_terms: list[str]) -> dict[str, list[float]]:
    from drowse.cli.alpha_grid import AlphaListError, parse_alpha_list

    grid: dict[str, list[float]] = {}
    for raw in raw_terms:
        if "=" not in raw:
            print(
                f"experiment fan: grid term must be CONCEPT=ALPHAS, got {raw!r}",
                file=sys.stderr,
            )
            sys.exit(2)
        name, alpha_text = raw.split("=", 1)
        name = name.strip()
        if not name:
            print("experiment fan: grid concept cannot be empty", file=sys.stderr)
            sys.exit(2)
        try:
            alphas = parse_alpha_list(alpha_text)
        except AlphaListError as e:
            print(f"experiment fan: {name}: {e}", file=sys.stderr)
            sys.exit(2)
        if not alphas:
            print(f"experiment fan: {name}: alpha list is empty", file=sys.stderr)
            sys.exit(2)
        grid[name] = [float(a) for a in alphas]
    return grid


def _run_experiment_fan(args: argparse.Namespace) -> None:
    import json as _json

    _pkg._load_effective_config(args, default_max_tokens=256)
    _pkg._print_startup(args)
    session = _pkg._make_session(args)
    _pkg._print_model_info(session)

    grid = _parse_grid_terms(args.grid)

    # A grid is sequential generations; without a per-completion line the
    # user sees nothing between "Loaded N probes" and the final table.  JSON
    # mode stays silent so stdout remains a single parseable document.
    total = 1
    for alphas in grid.values():
        total *= len(alphas)

    def _report(idx: int, result: Any, alphas: dict[str, float]) -> None:
        row = ", ".join(f"{k}={v:+.3f}" for k, v in alphas.items())
        print(
            f"  {idx + 1}/{total}: {row} ({result.token_count} tokens)",
            flush=True,
        )

    runset = session.generate_sweep(
        args.prompt,
        grid,
        base_steering=args.base_steering,
        stateless=False,
        on_result=None if args.json_output else _report,
    )
    if args.json_output:
        print(_json.dumps(runset.to_dict(), indent=2))
        return
    print(f"experiment fan: {len(runset)} run(s)")
    for idx, result in enumerate(runset):
        node_id = runset.node_ids[idx] if idx < len(runset.node_ids) else None
        row = runset.grid[idx] if idx < len(runset.grid) else {}
        row_str = ", ".join(f"{k}={v:+.3f}" for k, v in row.items())
        node_str = f" node={node_id[:8]}" if node_id else ""
        print(
            f"{idx:>3}: {row_str}{node_str} "
            f"tokens={result.token_count} finish={result.finish_reason}"
        )


def _run_transcript_run(args: argparse.Namespace) -> None:
    from drowse.loom.transcript import (
        Transcript, TranscriptError,
    )

    transcript_path = Path(args.path)
    if not transcript_path.is_file():
        print(f"transcript run: {transcript_path}: file not found", file=sys.stderr)
        sys.exit(2)
    try:
        transcript = Transcript.load(transcript_path)
    except TranscriptError as e:
        print(f"transcript run: {e}", file=sys.stderr)
        sys.exit(2)

    _pkg._load_effective_config(args, default_max_tokens=256)
    if not args.model:
        if transcript.model_id:
            args.model = transcript.model_id
        else:
            print(
                "transcript run: model required (pass <model> or include "
                "`model_id` in the transcript)",
                file=sys.stderr,
            )
            sys.exit(2)

    _pkg._print_startup(args)
    session = _pkg._make_session(args)
    _pkg._print_model_info(session)

    # The transcript lands as a fresh branch under the synthetic root and
    # every generated turn regenerates under its recipe with the branch's
    # full history before it.
    try:
        replay = transcript.replay(session, mode="default", strict=args.strict)
    except TranscriptError as e:
        print(f"transcript run: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"transcript: {len(transcript.turns)} turns loaded "
          f"(leaf: {replay.leaf_id[:8]}, {len(replay.steps)} regenerated)")
    print()
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


# The three research verbs, plus the one nested ``transcript`` leaf.  Declared
# after their runners so the tables carry bound functions.
_EXPERIMENT_RUNNERS = {
    "fan":         _run_experiment_fan,
    "transcript":  _run_experiment_transcript,
    "naturalness": _run_experiment_naturalness,
}

_TRANSCRIPT_RUNNERS = {
    "run": _run_transcript_run,
}
