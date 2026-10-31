"""``drowse pack <verb>`` — the manifold lifecycle (install / share / inspect /
remove).  Pure IO over ``~/.drowse/manifolds/``; no verb loads a model."""

from __future__ import annotations

import argparse
from typing import Any

from drowse.cli.context import (
    CliError, emit_json, existing_folder, iter_manifolds, note, progress,
    resolve_manifold, user_errors,
)
from drowse.cli.spec import FORCE, JSON, MODEL_OPT, VERBOSE, YES, Verb, arg


def run_ls(args: argparse.Namespace) -> None:
    from drowse.store.manifold_schema import domain_label

    rows = list(iter_manifolds(args.namespace))
    if args.json_output:
        emit_json([
            {
                "namespace": ns, "name": mf.name, "domain": mf.domain,
                "nodes": mf.node_labels, "fitted_models": mf.tensor_models(),
            }
            for ns, mf in rows
        ])
        return
    if not rows:
        print("no manifolds installed under ~/.drowse/manifolds/")
        return
    for ns, mf in rows:
        models = ", ".join(mf.tensor_models()) or "(unfitted)"
        print(f"  {ns}/{mf.name}  [{domain_label(mf.domain)}, {len(mf.node_labels)} nodes]  {models}")
        if args.verbose and mf.description:
            print(f"    {mf.description}")


def _print_fit(fit: dict[str, Any]) -> None:
    tag = f"{fit['method']}, {fit['feature_space']}"
    if fit.get("fit_mode") and fit["fit_mode"] != "authored":
        tag += f", fit_mode={fit['fit_mode']}"
    print(f"    {fit['stem']}  ({tag})")
    diag = fit.get("diagnostics") or {}
    k = diag.get("picked_k")
    if diag and fit.get("fit_mode") == "pca":
        cumvar = diag.get("cumulative_variance") or []
        if k and 1 <= k <= len(cumvar):
            print(
                f"      pca: picked_k={k}, cumvar@k={cumvar[k - 1]:.3f} "
                f"(threshold={diag.get('threshold', '?')})"
            )
    elif diag and fit.get("fit_mode") == "spectral":
        bits = [f"picked_k={k}"]
        for key, label in (("gap_magnitude", "gap"), ("bandwidth", "bandwidth")):
            if diag.get(key) is not None:
                bits.append(f"{label}={diag[key]:.3g}")
        if diag.get("k_nn") is not None:
            bits.append(f"k_nn={diag['k_nn']}")
        print(f"      spectral: {', '.join(bits)}")
    if fit.get("hyperparams"):
        hp = ", ".join(f"{k}={v}" for k, v in sorted(fit["hyperparams"].items()))
        print(f"      hyperparams: {hp}")
    # Where the concept's whitened between-node spread concentrates.
    spread = fit.get("node_spread") or {}
    if spread:
        ranked = sorted(
            ((int(layer), float(v)) for layer, v in spread.items()),
            key=lambda kv: kv[1], reverse=True,
        )
        print(f"      signal by layer (peak): {', '.join(f'L{l}={v:.2f}' for l, v in ranked[:5])}")


def run_show(args: argparse.Namespace) -> None:
    """One manifold's geometry, nodes, and per-model fits.

    ``-j`` emits the server's ``manifold_summary`` shape; the text view also
    surfaces a discover folder's per-model derived node coordinates.
    """
    from drowse.store.manifold_tensors import load_manifold
    from drowse.store.manifold_folder import ManifoldFolder
    from drowse.store.manifold_schema import domain_label
    from drowse.store.manifold_lifecycle import manifold_fit_summary, manifold_summary

    folder = existing_folder(args.name)
    if args.json_output:
        emit_json(manifold_summary(folder, include_fits=True))
        return
    # An inventory view: fitted payload hashes are verified where they are
    # used, not where they are listed.
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    fits = [manifold_fit_summary(mf.sidecar(stem), stem) for stem in mf.tensor_models()]
    roles = mf.roles  # padded to the label count

    print(f"{folder.parent.name}/{mf.name}")
    if mf.description:
        print(f"  {mf.description}")
    if mf.is_discover:
        coords: list[list[float]] = []
        if fits:
            try:
                coords = load_manifold(mf.tensor_path(fits[0]["stem"])).node_coords.tolist()
            except (FileNotFoundError, KeyError, ValueError):
                coords = []
        print(f"  fit_mode: {mf.fit_mode} (discover — coords derived per-model)")
        if mf.hyperparams:
            print(f"  hyperparams: {', '.join(f'{k}={v}' for k, v in sorted(mf.hyperparams.items()))}")
        print("  nodes:")
        for i, label in enumerate(mf.node_labels):
            where = (
                f"({', '.join(f'{c:.3g}' for c in coords[i])})" if i < len(coords)
                else "(coords pending fit)"
            )
            role = f"  [role={roles[i]}]" if roles[i] else ""
            print(f"    {label}  {where}{role}")
    else:
        print(f"  domain: {domain_label(mf.domain)}")
        print("  nodes:")
        for label, node_coords, role in zip(mf.node_labels, mf.node_coords, roles, strict=True):
            tail = f"  [role={role}]" if role else ""
            print(f"    {label}  ({', '.join(f'{c:g}' for c in node_coords)}){tail}")
    if not fits:
        print("  fitted models: (none — run `drowse manifold fit`)")
        return
    print("  fitted models:")
    for fit in fits:
        _print_fit(fit)


def run_install(args: argparse.Namespace) -> None:
    from drowse.store.install import install_manifold

    # Hub progress bars vanish off-TTY; announce before the network call.
    note(f"Installing {args.target}...")
    with user_errors(ValueError, exit_code=2):
        dst = install_manifold(args.target, args.as_target, force=args.force, on_progress=progress)
    print(f"Installed {args.target} -> {dst}")


def run_search(args: argparse.Namespace) -> None:
    from drowse.store.hub import search_manifolds

    matching = f" matching {args.query!r}" if args.query else ""
    note(f"Searching Hugging Face for manifolds{matching}...")
    with user_errors(ImportError, OSError):
        rows = search_manifolds(args.query or None)
    if args.json_output:
        emit_json(rows)
        return
    if not rows:
        print("(no matches)")
        return
    for r in rows:
        tail = (
            r.get("description", "") if args.verbose
            else ", ".join(r["tensor_models"]) or "(unfitted)"
        )
        print(
            f"{r['name']:<24} {r['namespace']:<12} [hf]          "
            f"{r['domain_label']:<14} {r['node_count']:>3} nodes  {tail}"
        )


def run_push(args: argparse.Namespace) -> None:
    from drowse.base.hf import resolve_target_coord
    from drowse.store.hub import push_manifold

    folder = existing_folder(args.selector)
    ns, name = folder.parent.name, folder.name
    # ``--as owner/name`` wins, else ``<whoami>/<name>``.
    with user_errors(ValueError, RuntimeError, exit_code=2):
        coord = resolve_target_coord(name, args.as_target)
        note(f"{'Dry-run: resolving push of' if args.dry_run else 'Pushing'} {ns}/{name} to {coord}...")
        repo_url, sha = push_manifold(
            folder, coord, private=args.private, model_scope=args.model,
            variant=args.variant, dry_run=args.dry_run,
        )
    if sha:
        print(f"Pushed {coord} -> {repo_url} @ {sha[:12]}")
    elif args.dry_run:
        print(f"Dry-run: would push {coord} -> {repo_url}")
    else:
        print(f"Pushed {coord} -> {repo_url}")


def run_rm(args: argparse.Namespace) -> None:
    from drowse.store.manifold_lifecycle import remove_manifold_folder

    ns, name = resolve_manifold(args.selector)
    if not args.yes:
        tail = (
            " (re-materializes on restart)" if ns == "default"
            else " (this cannot be undone)"
        )
        raise CliError(f"refusing to remove manifold {ns}/{name}{tail}; pass -y/--yes to confirm")
    result = remove_manifold_folder(ns, name)
    suffix = (
        " (re-materializes on next session start)"
        if result.get("rematerializes_on_restart") else ""
    )
    print(f"Removed {result['namespace']}/{result['name']}{suffix}")


def run_clear(args: argparse.Namespace) -> None:
    from drowse.store.manifold_lifecycle import clear_manifold_tensors

    ns, name = resolve_manifold(args.selector)
    n = clear_manifold_tensors(ns, name, args.model, variant=args.variant)
    print(f"Deleted {n} files")


def run_refresh(args: argparse.Namespace) -> None:
    from drowse.store.manifold_lifecycle import refresh_manifold

    ns, name = resolve_manifold(args.selector)
    tier = refresh_manifold(ns, name, model_scope=args.model)
    print({
        "scoped": f"Refreshed {ns}/{name} ({args.model} tensors cleared — re-fits on next use)",
        "skipped": f"{ns}/{name}: local source, nothing to refresh",
        "bundled": f"Refreshed {ns}/{name} (bundled — re-materialized from package data)",
        "hf": f"Refreshed {ns}/{name} (re-pulled from HF)",
    }[tier])


def run_export_gguf(args: argparse.Namespace) -> None:
    from drowse.store.gguf import export_gguf_manifold

    ns, name = resolve_manifold(args.name)
    with user_errors(RuntimeError, ValueError):
        written = export_gguf_manifold(
            ns, name, model_scope=args.model, output=args.output, model_hint=args.model_hint,
        )
    for path in written:
        print(f"Wrote {path}")


def run_export_archive(args: argparse.Namespace) -> None:
    from drowse.store.archive import export_drowse_archive

    ns, name = resolve_manifold(args.name)
    with user_errors(RuntimeError, ValueError):
        written = export_drowse_archive(ns, name, output=args.output)
    print(f"Wrote {written}")


_NAME = arg("name", help="Manifold name (or ns/name)")
_SELECTOR = arg("selector", help="Manifold name (or ns/name)")

GROUP = Verb(
    "pack",
    "Manifold lifecycle (ls/show/install/search/push/rm/clear/refresh/export)",
    description="Manifold install / share / inspect / removal lifecycle.",
    verbs=(
        Verb("ls", "List installed manifolds", run=run_ls, args=(
            arg("--namespace", default=None, metavar="NS",
                help="Restrict to one namespace"),
            JSON, VERBOSE.but(help="Include descriptions"),
        )),
        Verb("show", "Show a manifold's nodes and fitted models", run=run_show,
             args=(_NAME, JSON)),
        Verb(
            "install", "Install a manifold from HF or a local folder",
            run=run_install,
            description="Pull a drowse-manifold repo (<ns>/<name>[@revision]) "
                        "or copy-install a local folder or .drowse archive.",
            args=(
                arg("target", help="<ns>/<name>[@revision], manifold folder, or .drowse path"),
                arg("-a", "--as", dest="as_target", default=None, metavar="NS/NAME",
                    help="Install under a different (fully qualified) namespace/name"),
                FORCE.but(help="Overwrite an existing installation"),
            ),
        ),
        Verb("search", "Search the HuggingFace hub for manifolds", run=run_search, args=(
            arg("query", nargs="?", default="", help="Search text (matched against HF model ids)"),
            JSON, VERBOSE.but(help="Show descriptions instead of fitted models"),
        )),
        Verb(
            "push", "Push a manifold to HF as a model repo",
            run=run_push,
            description="Push a manifold (corpus always, fitted tensors filtered "
                        "by -m/--variant) as a drowse-manifold-tagged model repo.",
            args=(
                _SELECTOR,
                arg("-a", "--as", dest="as_target", default=None, metavar="OWNER/NAME",
                    help="Target HF coord (default: <whoami>/<name>)"),
                MODEL_OPT.but(help="Push only this base model's tensors"),
                # Long-only: push is the one publishing verb, and -p/-d mean
                # --probes/--device everywhere else.
                arg("--private", action="store_true", help="Create the repo as private"),
                arg("--dry-run", action="store_true",
                    help="Resolve the coord and report what would be pushed"),
                arg("--variant", choices=["raw", "sae", "all"], default="raw",
                    help="Tensor variants to push (default: raw; SAE variants are opt-in)"),
            ),
        ),
        Verb(
            "rm", "Fully remove a manifold folder", run=run_rm,
            description="Remove a whole manifold folder. Bundled (default/) "
                        "manifolds re-materialize on the next session start.",
            args=(_SELECTOR, YES),
        ),
        Verb(
            "clear", "Delete per-model fitted tensors for a manifold", run=run_clear,
            description="Delete fitted tensors (they re-fit on next use), keeping "
                        "manifold.json and the node corpus.",
            args=(
                _SELECTOR,
                MODEL_OPT.but(help="Only this model's tensors (default: all models)"),
                arg("--variant", choices=["raw", "sae", "all"], default="all",
                    help="Tensor variants to delete (default: all)"),
            ),
        ),
        Verb(
            "refresh", "Re-pull / re-materialize a manifold from its source",
            run=run_refresh,
            description="Re-pull from the source (local: skipped, bundled: "
                        "re-materialized, hf: re-pulled). With -m, only drop that "
                        "model's fitted tensors so it re-fits on next use.",
            args=(_SELECTOR, MODEL_OPT.but(help="Scoped refresh for one model")),
        ),
        Verb(
            "export", "Export a manifold as GGUF or .drowse",
            verbs=(
                Verb(
                    "gguf", "Export a folded 2-node manifold to llama.cpp GGUF",
                    run=run_export_gguf,
                    args=(
                        _NAME, MODEL_OPT,
                        arg("-o", "--output", default=None, metavar="PATH"),
                        arg("--model-hint", default=None, metavar="HINT"),
                    ),
                ),
                Verb(
                    "archive", "Export a manifold (+ referenced template) as .drowse",
                    run=run_export_archive,
                    description="Write one complete manifold closure as a validated "
                                ".drowse archive. Checksums establish integrity, not "
                                "publisher identity.",
                    args=(_NAME, arg("-o", "--output", default=None, metavar="FILE")),
                ),
            ),
        ),
    ),
)
