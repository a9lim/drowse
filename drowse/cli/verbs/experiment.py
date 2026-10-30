"""``drowse experiment <verb>`` — repeatable research runs: an alpha-grid
fan, transcript replay, and the behavior-manifold naturalness eval."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drowse.cli.context import (
    CliError, check_steering, effective_config, emit_json, existing_folder,
    note, open_session, progress,
)
from drowse.cli.spec import (
    CONFIG, INJECTION, JSON, MODEL, MODEL_LOAD, PROBES, STEER, TOP_K_ALTS, Verb,
    arg, positive_int,
)

if TYPE_CHECKING:
    import torch

    from drowse.geometry.manifold import ManifoldDomain


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
    """Import a transcript as a fresh branch, then regenerate every
    assistant turn under its imported parent — with the full conversation
    history the recording had — and report the probe-reading drift."""
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
        leaf_id = transcript.import_into(session, mode="default", strict=args.strict)
        turns = [t for t in transcript.turns if t.role != "system"]
        branch = session.tree.path_to(leaf_id)[-len(turns):] if turns else []
        print(f"transcript: {len(turns)} turns loaded (leaf: {leaf_id[:8]})\n")
        for idx, (turn, node) in enumerate(zip(turns, branch, strict=True)):
            if turn.role != "assistant" or node.parent_id is None:
                continue
            recipe = turn.recipe
            print(f"--- replay turn {idx} ---")
            result = session.generate(
                None, parent_node_id=node.parent_id, append_same_role=False,
                steering=recipe.steering if recipe is not None else None,
                sampling=recipe.sampling if recipe is not None else None,
            ).first
            print(f"assistant: {result.text[:120]}")
            if turn.readings:
                actual = {
                    name: (reading.coords[0] if reading.coords else 0.0)
                    for name, reading in result.probe_readings.items()
                }
                deltas = sorted(
                    (
                        (name, actual.get(name, 0.0) - expected, expected, actual.get(name, 0.0))
                        for name, expected in turn.readings.items()
                    ),
                    key=lambda row: abs(row[1]), reverse=True,
                )
                print("  readings drift (top 5):")
                for name, d, ev, av in deltas[:5]:
                    print(f"    {name:<32}  Δ {d:+.4f}  (expected {ev:+.4f} → got {av:+.4f})")
            print()


# ---------------------------------------------------------------------------
# naturalness
# ---------------------------------------------------------------------------

def behavior_inputs(
    folder: Path,
) -> "tuple[list[tuple[str, list[str]]], ManifoldDomain, torch.Tensor]":
    """The node corpora + authored geometry the behavior manifold is fit on.

    Only the authoring geometry and corpus are read, so fitted payloads are
    not hashed (``verify_manifest=False``).  A discover folder has no
    authored geometry — its coordinates are per-model.
    """
    import torch

    from drowse.geometry.manifold import domain_from_spec
    from drowse.store.manifolds import ManifoldFolder

    mf = ManifoldFolder.load(folder, verify_manifest=False)
    if mf.is_discover or not mf.domain:
        raise CliError(
            f"{folder.parent.name}/{folder.name} is a discover-mode manifold; the "
            f"naturalness eval needs an authored domain and node coordinates"
        )
    coords = torch.tensor(mf.node_coords, dtype=torch.float32)
    return mf.node_groups(), domain_from_spec(mf.domain), coords


def _single_manifold_term(expression: str) -> Any:
    from drowse.steering.grammar import ManifoldTerm, parse_expr

    terms = list(parse_expr(expression).alphas.values())
    if len(terms) != 1 or not isinstance(terms[0], ManifoldTerm):
        raise CliError("--compare-linear needs -S to be a single manifold ('%') term")
    return terms[0]


def run_naturalness(args: argparse.Namespace) -> None:
    """Fit a behavior-space manifold over the node corpus (each node's mean
    next-token distribution in Hellinger space) and score a steered
    trajectory's per-step Bhattacharyya distance to it — low is natural."""
    import torch

    from drowse.fit.naturalness import (
        compute_node_behavior_centroid, compute_trajectory_distributions,
        fit_behavior_manifold, trajectory_naturalness,
    )
    from drowse.generation.sampling import SamplingConfig
    from drowse.geometry.profile import Profile

    target = Path(args.manifold)
    folder = target if (target / "manifold.json").exists() else existing_folder(args.manifold)
    node_groups, domain, node_coords = behavior_inputs(folder)
    linear_term = _single_manifold_term(args.steer) if args.compare_linear else None
    effective_config(args, default_max_tokens=128)

    with open_session(args) as session:
        note(f"Fitting the behavior manifold ({len(node_groups)} nodes)...")
        centroids = [
            compute_node_behavior_centroid(session.model, session.tokenizer, session.device, texts)
            for _label, texts in node_groups
        ]
        behavior = fit_behavior_manifold(torch.stack(centroids), domain.embed(node_coords))
        sampling = SamplingConfig(max_tokens=args.max_tokens, seed=0)

        def score(steer: str) -> tuple[float, float]:
            text = session.generate(args.prompt, steering=steer, sampling=sampling).first.text
            trajectory = compute_trajectory_distributions(
                session.model, session.tokenizer, session.device, text,
            )
            per_step = trajectory_naturalness(trajectory, behavior, domain, node_coords)
            return float(per_step.mean()), float(per_step.max())

        rows: list[dict[str, Any]] = []
        mean_d, max_d = score(args.steer)
        rows.append({"label": "manifold", "steering": args.steer,
                     "mean_bhattacharyya": mean_d, "max_bhattacharyya": max_d})
        if linear_term is not None:
            # The straight activation-space chord from node 0 to the term's
            # position, per layer — what plain additive steering would do.
            session.ensure_manifold_loaded(linear_term.manifold)
            manifold = session.manifolds[linear_term.manifold]
            position = manifold.resolve_position(linear_term.position)
            origin = manifold.node_coords[0]
            session.steer("__manifold_linear_baseline__", Profile({
                layer: manifold.manifold_point(layer, position) - manifold.manifold_point(layer, origin)
                for layer in manifold.layer_indices
            }))
            mean_l, max_l = score(f"{linear_term.coeff:g} __manifold_linear_baseline__")
            label = (
                linear_term.position if isinstance(linear_term.position, str)
                else ",".join(f"{c:g}" for c in position)
            )
            rows.append({"label": "linear-chord", "steering": f"chord@{label}",
                         "mean_bhattacharyya": mean_l, "max_bhattacharyya": max_l})

    if args.json_output:
        emit_json({"prompt": args.prompt, "results": rows})
        return
    print("behavior-manifold naturalness  (lower = more natural)")
    for r in rows:
        print(
            f"  {r['label']:<14} mean D_B={r['mean_bhattacharyya']:.4f}  "
            f"max D_B={r['max_bhattacharyya']:.4f}"
        )
    if len(rows) == 2:
        delta = rows[1]["mean_bhattacharyya"] - rows[0]["mean_bhattacharyya"]
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
                               "(overrides YAML vectors:)"),
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
            description="Fit a behavior-space manifold over an authored manifold's "
                        "node corpus and report the mean / max Bhattacharyya "
                        "distance of a steered trajectory to it (low = natural).",
            args=(
                MODEL,
                arg("prompt", help="Prompt to generate from"),
                arg("--manifold", required=True, metavar="NAME",
                    help="Authored manifold (name, ns/name, or folder) whose node "
                         "corpus seeds the behavior manifold"),
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
