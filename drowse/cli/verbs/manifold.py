"""``drowse manifold <verb>`` — the steering-vector / manifold compute surface.

A steering vector is the 2-node flat case of a manifold, so extraction,
corpus generation, fitting, baking, merging, transfer, and the disk-side
inspection verbs (``compare`` / ``why``) all live here.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

from drowse.cli.context import (
    CliError, emit_json, existing_folder, folder_or_manifold, note,
    open_session, parse_layers, progress, resolve_manifold, split_new,
    user_errors,
)
from drowse.cli.spec import (
    DESCRIPTION, FORCE, JSON, MODEL_LOAD, MODEL_OPT, MODEL_REQ, NO_DLS, VERBOSE,
    OneOf, Verb, arg, layers_arg, positive_int, smoothing,
)

if TYPE_CHECKING:
    from drowse.geometry.profile import Profile


# ---------------------------------------------------------------------------
# extract / generate / from-template — authoring
# ---------------------------------------------------------------------------

_KIND_HELP = (
    "Conceptual kind (default: abstract). Selects the generation system "
    "template + elicitation role label: abstract -> 'someone {c}', concrete "
    "-> '{art} {c}', custom -> the --system template (no role swap)."
)
_SYSTEM_HELP = (
    "Custom elicitation system prompt for --kind custom; {c} is the concept "
    "(e.g. \"You are the month of {c}; speak as that month.\"). The framing "
    "rides the system prompt, so it works on every model family."
)
KIND = arg("--kind", choices=("abstract", "concrete", "custom"),
           default="abstract", help=_KIND_HELP)
SYSTEM = arg("--system", dest="custom_system", default=None,
             metavar="TEMPLATE", help=_SYSTEM_HELP)


def _require_custom_system(args: argparse.Namespace) -> None:
    if args.kind == "custom" and not args.custom_system:
        raise CliError(
            "--kind custom requires --system (a template with a {c} "
            'placeholder, e.g. "You are the month of {c}; speak as that month.")'
        )


def _fit_noop(
    args: argparse.Namespace, folder: Path, *,
    layers: "list[int] | str | None" = None,
    sae: str | None = None,
    fit_mode: str | None = None,
    hyperparams: dict[str, object] | None = None,
) -> Any:
    """The weight-free proof that this fit is already exact, or ``None``."""
    from drowse.store.manifold_lifecycle import preflight_manifold_fit_noop

    if args.force or not (folder / "manifold.json").exists():
        return None
    return preflight_manifold_fit_noop(
        folder, model_id=args.model, layer_indices=layers,
        sae=sae, fit_mode=fit_mode, hyperparams=hyperparams,
        quantize=args.quantize, device=args.device,
    )


def _corpus_roles_match(folder: Path, role: str | None) -> bool:
    """Whether every node of an existing folder was authored under ``role``
    (``extract --role`` must not reuse a corpus with another baseline)."""
    from drowse.store.manifold_folder import ManifoldFolder, ManifoldFormatError

    try:
        mf = ManifoldFolder.load(folder, verify_manifest=False)
    except (ManifoldFormatError, OSError, ValueError):
        return False
    roles = list(mf.node_roles) + [None] * (len(mf.node_labels) - len(mf.node_roles))
    return bool(roles) and all(r == role for r in roles)


def run_extract(args: argparse.Namespace) -> None:
    from drowse.base.paths import manifold_dir, tensor_filename
    from drowse.naming import canonical_concept_name

    if len(args.concept) > 2:
        raise CliError(f"expected 1 or 2 concepts (got {len(args.concept)})")
    if args.sae and args.role:
        raise CliError(
            "--sae and --role are mutually exclusive (role substitution bakes "
            "into the node corpora; SAE reconstructs the centroids)"
        )
    _require_custom_system(args)
    namespace, raw = split_new(args.concept[0])
    baseline = args.concept[1] if len(args.concept) == 2 else None
    canonical = canonical_concept_name(raw, baseline)
    folder = manifold_dir(namespace, canonical)
    variant_tail = f":role-{args.role}" if args.role else ""

    # Bare file existence proves nothing: the weight-free preflight checks
    # sidecar integrity, corpus / role / template identity, the token-exact
    # render and the source fingerprint, and falls through when any is unproven.
    if _corpus_roles_match(folder, args.role):
        proof = _fit_noop(args, folder, sae=args.sae)
        if proof is not None:
            note("Already fitted for this corpus and exact model source — nothing to do.")
            print(f"extracted {canonical}{variant_tail} -> {proof.tensor_path}")
            return

    with open_session(args) as session, user_errors(ValueError):
        name, _profile = session.extract(
            raw, baseline,
            kind=args.kind, custom_system=args.custom_system, force=args.force,
            sae=args.sae, role=args.role, namespace=namespace,
            on_progress=progress,
        )
    tensor = folder / tensor_filename(args.model, release=args.sae)
    print(f"extracted {name} -> {tensor}")


def run_generate(args: argparse.Namespace) -> None:
    """Author a discover folder and fill one in-character corpus per concept.

    Generation and fitting are separate steps on purpose: a flaky generation
    leaves inspectable corpora, and ``manifold fit`` resumes from them.
    """
    from drowse.base.paths import manifold_dir
    from drowse.store.manifold_authoring import (
        append_discover_manifold_node, plan_discover_generation,
    )

    _require_custom_system(args)
    if len(args.concepts) < 2:
        raise CliError("need >= 2 concepts (a one-node discover manifold is meaningless)")
    namespace, name = split_new(args.name)
    folder = manifold_dir(namespace, name)
    # --role-per-node: each slug doubles as its node's assistant-role
    # substitution (a persona manifold pooled in role-baselined space).
    node_roles = (
        {c: c for c in args.concepts} if args.role_per_node else None
    )
    plan = plan_discover_generation(
        folder, name, args.description, fit_mode="pca",
        labels=list(args.concepts), node_roles=node_roles,
        node_kinds={c: args.kind for c in args.concepts}, force=args.force,
    )
    n_total = len(plan.index_of)
    if not plan.pending:
        print(f"manifold {namespace}/{name} already complete ({n_total} nodes)")
        note("pass -f/--force to regenerate")
        return
    if plan.resumed and len(plan.pending) < n_total:
        note(
            f"resuming {namespace}/{name}: {n_total - len(plan.pending)}/"
            f"{n_total} nodes present, generating {len(plan.pending)}"
        )
    if plan.added:
        note(f"adding {len(plan.added)} new node(s): {', '.join(plan.added)}")

    with open_session(args) as session:
        note(
            f"generating {args.samples_per_prompt} response(s) per baseline "
            f"prompt for {len(plan.pending)} concept(s)..."
        )
        for concept in plan.pending:
            corpora = session.generate_responses(
                [concept], [args.kind],
                roles={concept: concept} if node_roles else None,
                custom_system=args.custom_system,
                samples_per_prompt=args.samples_per_prompt,
                on_progress=progress,
            )
            append_discover_manifold_node(
                folder, plan.index_of[concept], concept, corpora[concept],
            )
    print(f"wrote {namespace}/{name} ({n_total} nodes)")
    note(f"  -> run `drowse manifold fit {namespace}/{name} -m MODEL` to fit")


def run_from_template(args: argparse.Namespace) -> None:
    """Pure IO: expand a template's values x contexts into node corpora and
    write a discover folder that ``template_ref``-erences it."""
    from drowse.store.manifold_authoring import create_manifold_from_template
    from drowse.store.templates import resolve_template

    tmpl = resolve_template(args.template)
    ref = f"{tmpl.path.parent.name}/{tmpl.name}" if tmpl.path else tmpl.name
    namespace, name = split_new(args.name or tmpl.name)
    hyperparams = {
        key: value
        for key, value in (("max_dim", args.max_dim), ("var_threshold", args.var_threshold))
        if value is not None
    }
    try:
        create_manifold_from_template(
            namespace, name, args.description, template_ref=ref,
            fit_mode=args.fit_mode, hyperparams=hyperparams or None,
            force=args.force,
        )
    except FileExistsError:
        raise CliError(
            f"{namespace}/{name} already exists -- pass -f/--force to overwrite",
            exit_code=1,
        ) from None
    print(
        f"wrote {namespace}/{name} ({len(tmpl.values)} nodes x "
        f"{len(tmpl.contexts)} contexts, template_ref={ref}, "
        f"fit_mode={args.fit_mode})"
    )
    note(f"  -> run `drowse manifold fit {namespace}/{name} -m MODEL` to fit")


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

#: Discover-mode override flags → ``hyperparams`` keys (``--method`` is the
#: fit-mode override and travels separately).
_HYPERPARAMS = (
    "max_dim", "min_dim", "var_threshold", "k_nn", "bandwidth",
    "max_subspace_dim", "smoothing", "persistence_frac",
)


def run_fit(args: argparse.Namespace) -> None:
    """Fit an authored or discover-mode manifold (name or folder path).

    Discover overrides are merged into ``manifold.json`` inside the fit's own
    lock (the one that derives the cache key); against an authored folder
    they are an error, as on the server.
    """
    from drowse.store.manifold_folder import ManifoldFolder, domain_label

    layers = parse_layers(args.layers)
    if layers == "sample":
        raise CliError("--layers takes explicit indices, workspace, or all")
    folder = folder_or_manifold(args.target)
    mf = ManifoldFolder.load(folder, verify_manifest=False)

    hyperparams = {k: getattr(args, k) for k in _HYPERPARAMS if getattr(args, k) is not None}
    overriding = bool(hyperparams) or args.method is not None
    if overriding and not mf.is_discover:
        raise CliError(
            f"fit_mode/hyperparameter overrides are discover-mode only; "
            f"{folder} is authored"
        )
    fit_mode = args.method if overriding else None
    params = hyperparams if overriding else None

    # An override patch is deliberately not proven: it rewrites manifold.json
    # inside the fit lock, so the on-disk hash the preflight sees is not the
    # one the fit compares against.
    proof = _fit_noop(
        args, folder, layers=layers, sae=args.sae, fit_mode=fit_mode, hyperparams=params,
    )
    if proof is not None:
        note("Already fitted for this corpus and exact model source — nothing to do.")
        print(
            f"fitted manifold '{proof.manifold_name}' "
            f"({len(proof.fitted_layers)} layers, {len(proof.node_labels)} "
            f"nodes, {proof.fit_mode}) -> {proof.tensor_path}"
        )
        return

    with open_session(args) as session, user_errors(ValueError):
        manifold = session.fit(
            folder, sae=args.sae, layers=layers, fit_mode=fit_mode,
            hyperparams=params, force=args.force, on_progress=progress,
        )
    fitted_mode = getattr(manifold, "metadata", {}).get("fit_mode", mf.fit_mode)
    tail = f", fit_mode={fitted_mode}" if mf.is_discover else ""
    print(
        f"fitted manifold '{manifold.name}' ({len(manifold.layers)} layers, "
        f"{len(manifold.node_labels)} nodes, "
        f"{domain_label(manifold.domain.to_spec())}, "
        f"{manifold.feature_space}{tail})"
    )


# ---------------------------------------------------------------------------
# bake / merge / transfer
# ---------------------------------------------------------------------------

def run_bake(args: argparse.Namespace) -> None:
    from drowse.fit.bake import merge_into_manifold

    dst = merge_into_manifold(
        args.name, args.expression, model=args.model,
        force=args.force, strict=args.strict,
    )
    print(f"Baked manifold written to {dst}")


def run_merge(args: argparse.Namespace) -> None:
    from drowse.store.manifold_authoring import merge_discover_manifolds

    if len(args.sources) < 2:
        raise CliError("need >= 2 sources (the union of one manifold is a no-op)")
    target_ns, target_name = split_new(args.name)
    sources = [resolve_manifold(s) for s in args.sources]
    with user_errors(ValueError):
        dst = merge_discover_manifolds(
            target_ns, target_name, args.description,
            sources=sources, fit_mode=args.method, force=args.force,
        )
    print(f"Merged manifold written to {dst}")
    note(f"  -> run `drowse manifold fit {target_ns}/{target_name} -m MODEL` to fit")


def run_transfer(args: argparse.Namespace) -> None:
    """Fit (or load) the per-layer Procrustes alignment between the two
    models' neutral activations, then map the fitted manifold through it."""
    from drowse.base.paths import encode_release_id
    from drowse.store.alignment import load_or_fit_transfer_alignment
    from drowse.store.manifold_lifecycle import (
        preflight_transfer_manifold, transfer_manifold,
    )

    folder = existing_folder(args.name)
    ns, name = folder.parent.name, folder.name
    source_proof = preflight_transfer_manifold(
        folder, from_model=args.src_model, to_model=args.tgt_model, force=args.force,
    )
    (
        alignment, quality_per_layer, _path, source_identity, target_identity,
        target_whitener, target_layer_means,
    ) = load_or_fit_transfer_alignment(
        args.src_model, args.tgt_model, force=args.force,
        label="manifold transfer",
        requested_layers=list(source_proof.layers) if source_proof is not None else None,
    )
    median_quality = (
        statistics.median(float(q) for q in quality_per_layer.values())
        if quality_per_layer else None
    )
    out_path = transfer_manifold(
        folder,
        from_model=args.src_model,
        to_model=args.tgt_model,
        alignment=alignment,
        transfer_quality_estimate=median_quality,
        source_model_fingerprint=source_identity["model_fingerprint"],
        target_model_fingerprint=target_identity["model_fingerprint"],
        whitener=target_whitener,
        target_layer_means=target_layer_means,
        force=args.force,
        expected_source_proof=source_proof,
    )
    if args.json_output:
        emit_json({
            "namespace": ns,
            "name": name,
            "source_model": args.src_model,
            "target_model": args.tgt_model,
            "tensor": str(out_path),
            "transferred_layers": sorted(alignment),
            "median_transfer_quality": (
                round(median_quality, 4) if median_quality is not None else None
            ),
        })
        return
    quality = f"{median_quality:.3f}" if median_quality is not None else "n/a"
    print(
        f"Transferred manifold {ns}/{name} from {args.src_model} -> {args.tgt_model}\n"
        f"  layers:           {len(alignment)} shared\n"
        f"  median quality:   {quality} (R^2 across shared layers)\n"
        f"  tensor:           {out_path}\n"
        f"  variant suffix:   :from-{encode_release_id(args.src_model)}"
    )


# ---------------------------------------------------------------------------
# compare / why — fold fitted 2-node manifolds to steering vectors
# ---------------------------------------------------------------------------

def _variant_tensor(model_id: str, variant: str) -> str:
    """The fitted-tensor filename a selector variant addresses."""
    from drowse.base.paths import tensor_filename

    if variant.startswith("sae-"):
        return tensor_filename(model_id, release=variant[len("sae-"):])
    if variant.startswith("from-"):
        return tensor_filename(
            model_id, transferred_from=variant[len("from-"):],
            transferred_from_is_encoded=True,
        )
    if variant == "raw" or variant.startswith("role-"):
        return tensor_filename(model_id)
    raise CliError(
        f"variant ':{variant}' needs its full form here "
        f"(:sae-<release>, :from-<src>, :role-<name>)"
    )


class _Folded(NamedTuple):
    display: str
    namespace: str
    name: str
    variant: str
    profile: "Profile"


def _fold(ref: str, model_id: str) -> _Folded:
    """Fold the fitted manifold ``[ns/]name[:variant]`` to a steering vector.

    A bare name searches every namespace (a collision raises); the display
    keeps the namespace only when the reference typed one.  Role variants
    check the recorded per-node roles, transfer variants their source model.
    """
    from drowse.base.paths import encode_release_id, manifold_dir, manifolds_dir
    from drowse.fit.capture import folded_directions
    from drowse.geometry.profile import Profile
    from drowse.store.manifold_tensors import load_manifold
    from drowse.store.selectors import AmbiguousSelectorError, canonicalize_atom

    typed_ns, _, rest = ref.rpartition("/")
    name, variant = canonicalize_atom(rest)
    filename = _variant_tensor(model_id, variant)
    if typed_ns:
        namespaces = [typed_ns]
    else:
        root = manifolds_dir()
        namespaces = sorted(d.name for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    hits = [ns for ns in namespaces if (manifold_dir(ns, name) / filename).is_file()]
    if not hits:
        raise CliError(
            f"no fitted manifold for '{ref}' with model {model_id} — fit it "
            f"first: `drowse manifold fit {typed_ns + '/' if typed_ns else ''}"
            f"{name} -m {model_id}`",
            exit_code=1,
        )
    if len(hits) > 1:
        raise AmbiguousSelectorError(
            f"ambiguous manifold '{name}': matches "
            + ", ".join(f"{ns}/{name}" for ns in hits)
        )
    ns = hits[0]
    manifold = load_manifold(manifold_dir(ns, name) / filename)
    metadata = manifold.metadata
    if variant.startswith("role-"):
        roles = metadata.get("node_roles")
        if not isinstance(roles, list) or not roles or any(
            r != variant[len("role-"):] for r in roles
        ):
            raise CliError(f"'{ref}' was not fitted uniformly with that role", exit_code=1)
    if variant.startswith("from-"):
        source = metadata.get("source_model_id")
        if not isinstance(source, str) or encode_release_id(source) != variant[len("from-"):]:
            raise CliError(f"'{ref}' transfer provenance does not match", exit_code=1)
    try:
        profile = Profile(folded_directions(manifold))
    except Exception as e:
        raise CliError(
            f"'{ref}' does not fold to a single steering direction "
            f"(not a 2-node affine fit): {e}"
        ) from e
    display = f"{ns}/{name}" if typed_ns else name
    if variant != "raw":
        display = f"{display}:{variant}"
    return _Folded(display, ns, name, variant, profile)


def _fold_all(model_id: str) -> "dict[tuple[str, str], Profile]":
    """Every installed manifold whose raw fit for ``model_id`` folds to a
    single direction, keyed by ``(namespace, name)``.  (No bundled
    materialization: it lands corpora, never fitted tensors.)"""
    from drowse.base.paths import tensor_filename
    from drowse.fit.capture import folded_directions
    from drowse.geometry.profile import Profile
    from drowse.store.manifold_tensors import load_manifold
    from drowse.store.manifold_authoring import iter_manifold_folders

    filename = tensor_filename(model_id)
    out: dict[tuple[str, str], Profile] = {}
    for ns, mf in iter_manifold_folders():
        tensor = mf.folder / filename
        if not tensor.is_file():
            continue
        try:
            out[(ns, mf.name)] = Profile(folded_directions(load_manifold(tensor)))
        except Exception:  # noqa: BLE001 — curved / rank > 1: not a vector
            continue
    return out


def _expand_compare_refs(raw_refs: list[str]) -> list[str]:
    """Expand bulk selectors (``tag:x`` / ``namespace:x`` / ``all``) to
    ``ns/name`` references, carrying a ``:variant`` suffix across."""
    from drowse.base.paths import VARIANT_SUFFIX_RE
    from drowse.store.selectors import parse, resolve

    refs: list[str] = []
    for raw in raw_refs:
        head, sep, tail = raw.rpartition(":")
        has_variant = bool(sep and head and VARIANT_SUFFIX_RE.match(tail))
        base, suffix = (head, f":{tail}") if has_variant else (raw, "")
        selector = parse(base)
        if selector.kind == "name":
            refs.append(raw)
        else:
            refs.extend(f"{c.namespace}/{c.name}{suffix}" for c in resolve(selector))
    return refs


def _layer_rows(per_layer: dict[int, float]) -> dict[str, float]:
    return {str(layer): round(value, 6) for layer, value in sorted(per_layer.items())}


def run_compare(args: argparse.Namespace) -> None:
    """Mahalanobis cosine between steering vectors: rank every other fitted
    vector (one reference), pairwise (two), or the N×N matrix (three+)."""
    from drowse.geometry.whitener import LayerWhitener

    whitener = LayerWhitener.from_cache(args.model)

    def cos(a: "Profile", b: "Profile") -> float:
        return a.cosine_similarity(b, whitener=whitener)

    def per_layer(a: "Profile", b: "Profile") -> dict[int, float]:
        return a.cosine_similarity(b, per_layer=True, whitener=whitener)

    folded: dict[str, _Folded] = {}
    for ref in _expand_compare_refs(args.concepts):
        try:
            entry = _fold(ref, args.model)
        except CliError as e:
            note(f"warning: {e}; skipping")
            continue
        folded[entry.display] = entry
    if not folded:
        raise CliError("no loadable profiles found", exit_code=1)

    if len(args.concepts) == 1 and len(folded) == 1:
        (target,) = folded.values()
        _rank_all(args, target, cos, per_layer)
        return
    if len(folded) < 2:
        raise CliError("need at least 2 loadable profiles to compare", exit_code=1)

    profiles = {name: entry.profile for name, entry in folded.items()}
    names = list(profiles)
    if len(names) == 2:
        a_name, b_name = names
        a, b = profiles[a_name], profiles[b_name]
        sim = cos(a, b)
        layers = per_layer(a, b) if args.verbose else {}
        if args.json_output:
            payload: dict[str, Any] = {
                "a": a_name, "b": b_name, "model": args.model,
                "similarity": round(sim, 6),
            }
            if args.verbose:
                payload["per_layer"] = _layer_rows(layers)
            emit_json(payload)
            return
        print(f"{a_name} ~ {b_name}: {sim:+.4f}")
        for layer, value in sorted(layers.items()):
            print(f"  layer {layer:>3}: {value:+.4f}")
        return

    # Cosine is symmetric: fill the upper triangle once and mirror it.
    matrix = {a: {a: 1.0} for a in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            matrix[a][b] = matrix[b][a] = cos(profiles[a], profiles[b])
    if args.json_output:
        payload = {
            "model": args.model, "concepts": names,
            "matrix": {a: {b: round(matrix[a][b], 6) for b in names} for a in names},
        }
        if args.verbose:
            payload["per_layer"] = {
                f"{a}|{b}": _layer_rows(per_layer(profiles[a], profiles[b]))
                for i, a in enumerate(names) for b in names[i + 1:]
            }
        emit_json(payload)
        return
    width = max(len(n) for n in names)
    print(" " * (width + 2) + "  ".join(f"{n:>{width}}" for n in names))
    for a in names:
        row = "  ".join(f"{matrix[a][b]:>{width}.4f}" for b in names)
        print(f"{a:<{width}}  {row}")


def _rank_all(
    args: argparse.Namespace, target: _Folded,
    cos: "Callable[[Profile, Profile], float]",
    per_layer: "Callable[[Profile, Profile], dict[int, float]]",
) -> None:
    """Rank every other fitted vector for the model against ``target``.

    Display names are bare unless they collide — with another namespace's
    vector or with the target's own name — in which case they are qualified.
    """
    from collections import Counter

    pool = {
        key: profile for key, profile in _fold_all(args.model).items()
        if key != (target.namespace, target.name) or target.variant != "raw"
    }
    counts = Counter(name for _ns, name in pool)
    others = {
        (f"{ns}/{name}" if counts[name] > 1 or name == target.name else name): profile
        for (ns, name), profile in sorted(pool.items())
    }
    if not others:
        raise CliError(f"no other fitted vectors for model {args.model}", exit_code=1)
    ranked = sorted(
        ((name, cos(target.profile, p)) for name, p in others.items()),
        key=lambda item: item[1], reverse=True,
    )
    top3 = (
        {name: per_layer(target.profile, others[name]) for name, _ in ranked[:3]}
        if args.verbose else {}
    )
    if args.json_output:
        payload: dict[str, Any] = {
            "target": target.display, "model": args.model,
            "similarities": [{"name": n, "similarity": round(s, 6)} for n, s in ranked],
        }
        if args.verbose:
            payload["per_layer_top3"] = {n: _layer_rows(rows) for n, rows in top3.items()}
        emit_json(payload)
        return
    width = max(len(n) for n, _ in ranked)
    print(f"{target.display} vs all installed ({args.model}):")
    for name, score in ranked:
        print(f"  {name:<{width}}  {score:+.4f}")
    if top3:
        print("\n  per-layer (top 3):")
        for name, rows in top3.items():
            print(f"    {name}:")
            for layer, value in sorted(rows.items()):
                print(f"      layer {layer:>3}: {value:+.4f}")


#: Rows in the ``manifold why`` histogram: any model's full layer profile
#: collapses into a fixed-height block.
HIST_BUCKETS = 16


def bucketize(norms: list[tuple[int, float]], buckets: int) -> list[tuple[int, int, float]]:
    """Collapse layer-sorted ``(layer, norm)`` pairs into ``buckets`` evenly
    sized ``(lo_layer, hi_layer, mean_norm)`` groups (one per layer when
    there are fewer layers than buckets)."""
    n = len(norms)
    if n <= buckets:
        return [(layer, layer, value) for layer, value in norms]
    out: list[tuple[int, int, float]] = []
    for i in range(buckets):
        chunk = norms[(i * n) // buckets:((i + 1) * n) // buckets]
        out.append((chunk[0][0], chunk[-1][0], sum(v for _, v in chunk) / len(chunk)))
    return out


def run_why(args: argparse.Namespace) -> None:
    """Per-layer ``‖baked‖`` of a steering vector, as a 16-bucket histogram."""
    import shutil

    folded = _fold(args.concept, args.model)
    name = folded.display
    mags = sorted((layer, float(t.norm().item())) for layer, t in folded.profile.items())
    if args.json_output:
        emit_json({
            "concept": name, "model": args.model, "total_layers": len(mags),
            "layers": [{"layer": l, "magnitude": round(m, 6)} for l, m in mags],
        })
        return
    print(f"{name} ({len(mags)} layers, {args.model}):")
    print("  LAYERS (mean ||baked|| per bucket):")
    if not mags:
        return
    buckets = bucketize(mags, HIST_BUCKETS)
    max_norm = max(v for _, _, v in buckets) or 1.0
    label_w = max(2, len(str(buckets[-1][1])))

    def label(lo: int, hi: int) -> str:
        return f"L{lo:0{label_w}}" if lo == hi else f"L{lo:0{label_w}}-{hi:0{label_w}}"

    label_col = max(len(label(lo, hi)) for lo, hi, _ in buckets)
    bar_w = max(12, shutil.get_terminal_size((80, 24)).columns - 16 - label_col)
    for lo, hi, norm in buckets:
        filled = min(int(norm / max_norm * bar_w), bar_w)
        print(f"    {label(lo, hi):<{label_col}}  {'█' * filled}{'░' * (bar_w - filled)}  {norm:>8.3f}")


# ---------------------------------------------------------------------------
# The verb table
# ---------------------------------------------------------------------------

_DISCOVER = "Discover-mode override"

GROUP = Verb(
    "manifold",
    "Manifold compute (extract/generate/from-template/fit/bake/merge/transfer/compare/why)",
    description="Steering-vector / manifold authoring, fitting, and analysis.",
    verbs=(
        Verb(
            "extract", "Extract a steering vector (a 2-node flat manifold)",
            run=run_extract,
            description=(
                "Generate in-character corpora for one concept (a 1-node ray "
                "against the neutral mean) or two poles, and fit them as a "
                "flat `pca` manifold under ~/.drowse/manifolds/<ns>/<name>/ "
                "(ns defaults to local; prefix the first concept with `ns/`)."
            ),
            args=(
                arg("concept", nargs="+", metavar="CONCEPT",
                    help="One concept ('happy.sad', 'agentic') or two poles "
                         "('happy' 'sad'); prefix with ns/ to author outside local/"),
                MODEL_REQ, MODEL_LOAD, FORCE,
                OneOf((
                    arg("--sae", default=None, metavar="RELEASE",
                        help="Extract in a SAELens SAE release's feature space"),
                    arg("--role", default=None, metavar="SLUG",
                        help="Role-augmented extraction: render under a chat "
                             "template whose assistant-role label is SLUG; "
                             "selected later as :role-SLUG"),
                )),
                KIND, SYSTEM, NO_DLS,
            ),
        ),
        Verb(
            "generate", "Author a discover-mode manifold from a concept list",
            run=run_generate,
            description=(
                "For every concept, the loaded model answers the shared baseline "
                "prompts in character, writing one corpus per node of a fresh "
                "discover folder; fit it next with `drowse manifold fit`. A "
                "partial run resumes."
            ),
            args=(
                arg("name", help="Manifold name (ns/name for a non-local namespace)"),
                arg("--concepts", nargs="+", required=True, metavar="CONCEPT",
                    help="Concept slugs to generate corpora for (>= 2)"),
                KIND, SYSTEM,
                arg("--samples-per-prompt", type=positive_int, default=1, metavar="K",
                    help="In-character responses per baseline prompt (default: 1)"),
                arg("--role-per-node", action="store_true",
                    help="Use each concept slug as its node's assistant-role "
                         "substitution (a persona manifold; needs a family "
                         "with a substitutable role header)"),
                DESCRIPTION, FORCE.but(help="Overwrite an existing manifold folder"),
                MODEL_REQ, MODEL_LOAD,
            ),
        ),
        Verb(
            "from-template", "Author a discover manifold from a standalone template",
            run=run_from_template,
            description=(
                "Expand a template's values x contexts into per-value node "
                "corpora and write a discover folder that `template_ref`s it, so "
                "the fit uses the template's multi-turn contexts as elicitation "
                "prefixes. No model load."
            ),
            args=(
                arg("template", help="Template name (or ns/name)"),
                arg("--name", default=None, metavar="MANIFOLD",
                    help="Manifold to write (default: the template name)"),
                arg("--fit-mode", choices=("pca", "spectral", "auto"), default="auto",
                    help="Discover fit mode (default: auto — detects periodic "
                         "axes, right for cyclic categories)"),
                arg("--max-dim", type=positive_int, default=None, metavar="N",
                    help="Cap the discovered intrinsic dimension"),
                arg("--var-threshold", type=float, default=None, metavar="T",
                    help="PCA cumulative-variance threshold"),
                DESCRIPTION, FORCE.but(help="Overwrite an existing manifold folder"),
            ),
        ),
        Verb(
            "fit", "Fit an authored or discover-mode manifold",
            run=run_fit,
            description=(
                "Pool per-node centroids, fit the per-layer subspace (+ RBF "
                "surface for curved fits), and write the per-model tensor. The "
                "target is a manifold name or folder path. Discover folders "
                "accept the override flags, merged into manifold.json inside "
                "the fit lock; an exact repeat is proven without loading the model."
            ),
            args=(
                arg("target", help="Manifold name or folder path"),
                MODEL_REQ, MODEL_LOAD, FORCE,
                layers_arg("Fit only these layers (default: all; workspace = "
                           "the 40-90%% depth band)"),
                arg("--sae", default=None, metavar="RELEASE",
                    help="Fit in a SAELens SAE feature space"),
                arg("--method", choices=("pca", "spectral", "auto"), default=None,
                    help=f"{_DISCOVER}: fit_mode (auto picks flat vs curved and "
                         "detects periodic axes)"),
                arg("--max-dim", type=positive_int, default=None, metavar="N",
                    help=f"{_DISCOVER}: cap the intrinsic dimension (default 8)"),
                arg("--min-dim", type=positive_int, default=None, metavar="N",
                    help=f"{_DISCOVER} (spectral/auto): floor the intrinsic dimension"),
                arg("--var-threshold", type=float, default=None, metavar="T",
                    help=f"{_DISCOVER} (pca): cumulative-variance threshold (default 0.70)"),
                arg("--k-nn", type=positive_int, default=None, metavar="K",
                    help=f"{_DISCOVER} (spectral): k-NN graph degree"),
                arg("--bandwidth", type=float, default=None, metavar="SIGMA",
                    help=f"{_DISCOVER} (spectral): heat-kernel bandwidth"),
                arg("--max-subspace-dim", type=positive_int, default=None, metavar="R",
                    help=f"{_DISCOVER} (spectral/auto): per-layer RBF subspace cap (default 64)"),
                arg("--smoothing", type=smoothing, default=None, metavar="LAMBDA",
                    help=f"{_DISCOVER} (spectral/auto): 'auto' (GCV), '0' "
                         "(exact interpolation), or a float λ"),
                arg("--persistence-frac", type=float, default=None, metavar="F",
                    help=f"{_DISCOVER} (auto): H1 loop-significance threshold (default 0.5)"),
                NO_DLS,
            ),
        ),
        Verb(
            "bake", "Bake existing vectors into a new manifold",
            run=run_bake,
            description=(
                "Bake an additive expression of namespace-qualified fitted "
                "vectors into a corpus-less fit_mode=baked manifold under "
                "local/. Projections, ablations, triggers and % terms need a "
                "live model and are rejected."
            ),
            args=(
                arg("name", help="New manifold name (written under local/)"),
                arg("expression", help='Additive expression, e.g. "0.3 ns/a + 0.4 ns/b"'),
                FORCE,
                arg("-s", "--strict", action="store_true",
                    help="Require identical layer coverage across components"),
                MODEL_OPT,
            ),
        ),
        Verb(
            "merge", "Union discover-mode manifolds' nodes into a new manifold",
            run=run_merge,
            description=(
                "Union the node corpora of two or more discover-mode manifolds "
                "into a fresh unfitted discover folder (label collisions raise); "
                "fit it next with `drowse manifold fit`."
            ),
            args=(
                arg("name", help="New manifold name (or ns/name)"),
                arg("sources", nargs="+", metavar="SOURCE",
                    help="Existing discover-mode manifolds (name or ns/name)"),
                DESCRIPTION,
                arg("--method", choices=("pca", "spectral", "auto"), default=None,
                    help="Merged fit_mode (default: the sources' shared mode)"),
                FORCE.but(help="Overwrite an existing manifold folder"),
            ),
        ),
        Verb(
            "transfer", "Transfer a manifold to another model via Procrustes",
            run=run_transfer,
            description=(
                "Fit a per-layer Procrustes alignment between two models' "
                "neutral activations and map a fitted manifold into the target, "
                "written as the target's :from-<src> variant."
            ),
            args=(
                arg("name", help="Manifold name (or ns/name)"),
                arg("--from", dest="src_model", required=True, metavar="SRC_MODEL",
                    help="Model the manifold was fitted on"),
                arg("--to", dest="tgt_model", required=True, metavar="TGT_MODEL",
                    help="Model to transfer it to"),
                FORCE.but(help="Recompute alignment + transfer even when cached"),
                JSON,
            ),
        ),
        Verb(
            "compare", "Cosine similarity between steering vectors",
            run=run_compare,
            description=(
                "Mahalanobis cosine between fitted vectors: one reference ranks "
                "every other fitted vector, two compare pairwise, three or more "
                "print the matrix. Needs the model's cached neutral activations."
            ),
            args=(
                arg("concepts", nargs="+",
                    help="Vector references (name, ns/name[:variant], tag:x, namespace:x, all)"),
                MODEL_REQ, VERBOSE.but(help="Add per-layer breakdowns"), JSON,
            ),
        ),
        Verb(
            "why", "Show which layers contribute most to a steering vector",
            run=run_why,
            description="Per-layer ‖baked‖ histogram (16 buckets) of a fitted vector.",
            args=(
                arg("concept", help="Vector reference (name or ns/name[:variant])"),
                MODEL_REQ, JSON,
            ),
        ),
    ),
)
