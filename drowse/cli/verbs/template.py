"""``drowse template <verb>`` — the standalone templated-completion artifact
(a slot + candidate values + multi-turn contexts)."""

from __future__ import annotations

import argparse
import json

from drowse.cli.context import (
    CliError, check_steering, emit_json, note, open_session, split_new,
)
from drowse.cli.spec import FORCE, JSON, MODEL_LOAD, MODEL_REQ, STEER, YES, Verb, arg


def _canonical_context(entry: object) -> dict[str, object]:
    """A ``{turns, assistant}`` context, or the single-turn ``{user,
    assistant}`` sugar normalized to it."""
    if not isinstance(entry, dict):
        raise CliError("each context must be an object")
    if "turns" in entry:
        return {"turns": entry["turns"], "assistant": entry.get("assistant")}
    if "user" in entry:
        return {
            "turns": [{"role": "user", "content": entry["user"]}],
            "assistant": entry.get("assistant"),
        }
    raise CliError(
        "each context needs 'turns' (multi-turn) or 'user' (single-turn "
        "sugar), plus 'assistant'"
    )


def run_create(args: argparse.Namespace) -> None:
    from drowse.store.templates import create_template_folder

    namespace, name = split_new(args.name)
    try:
        with open(args.contexts_file) as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        raise CliError(f"cannot read --contexts: {e}", exit_code=1) from e
    if not isinstance(raw, list) or not raw:
        raise CliError(
            "--contexts must be a non-empty JSON list of contexts "
            '({"turns": [...], "assistant": ...} or {"user": ..., "assistant": ...})'
        )
    contexts = [_canonical_context(entry) for entry in raw]
    try:
        create_template_folder(
            namespace, name, slot=args.slot, values=list(args.values),
            contexts=contexts, description=args.description, force=args.force,
        )
    except FileExistsError:
        raise CliError(
            f"{namespace}/{name} already exists -- pass -f/--force to overwrite",
            exit_code=1,
        ) from None
    print(
        f"wrote template {namespace}/{name} ({len(args.values)} values x "
        f"{len(contexts)} contexts)"
    )
    note(f"  -> `drowse template score {namespace}/{name} -m MODEL` to read the value distribution")


def run_ls(args: argparse.Namespace) -> None:
    from drowse.store.templates import iter_template_folders

    rows = list(iter_template_folders())
    if args.json_output:
        emit_json([t.summary() for t in rows])
        return
    if not rows:
        print("no templates installed under ~/.drowse/templates/")
        return
    for t in rows:
        ns = t.path.parent.name if t.path else "?"
        print(
            f"  {ns}/{t.name}  [slot {t.slot!r}, {len(t.values)} values x "
            f"{len(t.contexts)} contexts]"
        )
        if t.description:
            print(f"    {t.description}")


def run_show(args: argparse.Namespace) -> None:
    from drowse.store.templates import resolve_template

    t = resolve_template(args.name)
    if args.json_output:
        payload = t.summary()
        payload["contexts"] = [
            {"turns": list(c.turns), "assistant": c.assistant} for c in t.contexts
        ]
        emit_json(payload)
        return
    print(f"{t.name}  (slot {t.slot!r})")
    if t.description:
        print(f"  {t.description}")
    print(f"  values ({len(t.values)}): {', '.join(t.values)}")
    print(f"  contexts ({len(t.contexts)}):")
    for i, c in enumerate(t.contexts):
        turns = " / ".join(f"{turn['role']}: {turn['content']}" for turn in c.turns)
        print(f"    [{i}] {turns}")
        print(f"        assistant: {c.assistant}")


def run_rm(args: argparse.Namespace) -> None:
    from drowse.store.templates import remove_template_folder, resolve_template

    t = resolve_template(args.name)
    namespace = t.path.parent.name if t.path else "local"
    if not args.yes:
        raise CliError(f"refusing to remove template {namespace}/{t.name}; pass -y/--yes to confirm")
    if not remove_template_folder(namespace, t.name):
        raise CliError(f"no template {namespace}/{t.name}", exit_code=1)
    print(f"Removed template {namespace}/{t.name}")


def run_score(args: argparse.Namespace) -> None:
    """The restricted-choice value distribution, per context."""
    from drowse.generation.scoring import score_template
    from drowse.store.templates import resolve_template

    tmpl = resolve_template(args.name)
    check_steering(args.steer)
    with open_session(args) as session:
        per_ctx = score_template(session, tmpl, steering=args.steer)
    if args.json_output:
        emit_json({
            "template": tmpl.name, "model": args.model, "steering": args.steer,
            "contexts": [sc.to_dict() for sc in per_ctx],
        })
        return
    steer_note = f"  (steering: {args.steer})" if args.steer else ""
    print(f"{tmpl.name} on {args.model}{steer_note}")
    for i, sc in enumerate(per_ctx):
        print(f"\n  context [{i}]:")
        for c in sc.ranked(by=args.by):
            prob = c.prob_sum if args.by == "sum" else c.prob_mean
            print(
                f"    {c.label:14s} P={prob:6.3f}  sum={c.sum_logprob:8.3f}  "
                f"mean={c.mean_logprob:8.3f}  n={c.n_tokens}  "
                f"{'#' * int(round(prob * 30))}"
            )


_NAME = arg("name", help="Template name (or ns/name)")

GROUP = Verb(
    "template",
    "Templated completions (create/ls/show/score/rm)",
    description="Standalone templated-completion artifact: author, inspect, and "
                "score the restricted-choice value distribution.",
    verbs=(
        Verb(
            "create", "Author a template (slot + values + multi-turn contexts)",
            run=run_create,
            description=(
                "Author ~/.drowse/templates/<ns>/<name>/: a slot token, candidate "
                "values, and contexts whose final assistant turn carries the slot. "
                "`template score` reads its value distribution; a manifold can "
                "`template_ref` it (`manifold from-template`). No model load."
            ),
            args=(
                arg("name", help="Template name (ns/name for a non-local namespace)"),
                arg("--slot", required=True, metavar="TOKEN",
                    help="Placeholder substituted per value, e.g. '[DAY]'; exactly "
                         "once in each final assistant turn, never in history"),
                arg("--values", nargs="+", required=True, metavar="VALUE",
                    help="Candidate slot values (>= 2); each value's slug is its node label"),
                arg("--contexts", dest="contexts_file", required=True, metavar="PATH",
                    help='JSON list of {"turns": [{"role", "content"}...], '
                         '"assistant": "...slot..."} or {"user": ..., "assistant": ...}'),
                arg("--description", default="", metavar="TEXT"),
                FORCE.but(help="Overwrite an existing template folder"),
            ),
        ),
        Verb("ls", "List installed templates", run=run_ls, args=(JSON,)),
        Verb("show", "Show a template's slot, values, and contexts", run=run_show,
             args=(_NAME, JSON)),
        Verb(
            "score", "Score the value distribution against each context (loads a model)",
            run=run_score,
            description=(
                "For every context, the model's restricted-choice probability over "
                "the candidate slot-fills, joint (sum) and length-normalized (mean). "
                "With -S, scores under steering: did it shift the distribution, not "
                "just the argmax?"
            ),
            args=(
                _NAME, MODEL_REQ, MODEL_LOAD,
                STEER.but(help="Score under this steering expression"),
                arg("--by", choices=("sum", "mean"), default="sum",
                    help="Ranking statistic: joint logprob (sum) or length-normalized (mean)"),
                JSON,
            ),
        ),
        Verb("rm", "Remove a template folder", run=run_rm, args=(_NAME, YES)),
    ),
)
