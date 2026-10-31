"""Discovery and authoring for manifold folders.

:func:`iter_manifold_folders` is the one discovery walk.  The create / init /
append / plan / merge / update functions are the shared backend of ``drowse
manifold`` and the ``/drowse/v1/manifolds`` routes; every whole folder they
write (an authored, discover, or baked manifold) is built in staging and
swapped into place, so a crash never leaves a half-written manifold behind.
The streaming discover skeleton (:func:`init_discover_manifold_folder` /
:func:`plan_discover_generation` + :func:`append_discover_manifold_node`) is
the one deliberate exception: it fills its corpus in place, and a skeleton
with missing corpus files is simply not discoverable until complete.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from drowse.base.atomic import write_json_atomic
from drowse.base.integrity import NAME_REGEX, canonical_json, hash_file
from drowse.base.paths import manifold_dir, manifolds_dir, tensor_filename
from drowse.geometry.domains import domain_from_spec
from drowse.store.manifold_folder import (
    DiscoverGenerationPlan,
    ManifoldFolder,
    _locked_manifest,
    locked_manifold_folder,
    replace_manifold_folder,
    write_node_corpus,
)
from drowse.store.manifold_schema import (
    FIT_MODES_DISCOVER,
    MERGE_BAKE_POLICY,
    ManifestNode,
    ManifoldFormatError,
    load_manifest_data,
    manifest_payload,
    min_nodes,
    node_filename,
    parse_manifest,
    require_discover_mode,
    resolve_node_maps,
    sanitize_hyperparams,
    spec_payload,
    validate_node_kind,
    validate_node_label,
    validate_node_labels,
    validate_node_role,
)
# No manifold may be authored under an instrument namespace (it would shadow it).
from drowse.naming import RESERVED_NAMESPACES
# Every authoring entry point that changes the installed manifold roster drops
# the resolver's memoized walks itself — callers carry no invalidation duty.
from drowse.store.selectors import invalidate as invalidate_selector_index
from drowse.errors import RequestError


def _lock_namespace_name(func: Any) -> Any:
    """Serialize a mutation whose first two arguments are namespace and name."""
    signature = inspect.signature(func)
    first, second = tuple(signature.parameters)[:2]

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        folder = manifold_dir(str(bound.arguments[first]), str(bound.arguments[second]))
        with locked_manifold_folder(folder):
            return func(*args, **kwargs)
    return wrapped


def _lock_folder_arg(func: Any) -> Any:
    """Serialize a mutation whose first argument is a manifold folder."""
    signature = inspect.signature(func)
    first = next(iter(signature.parameters))

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        with _locked_manifest(Path(bound.arguments[first])):
            return func(*args, **kwargs)
    return wrapped


# ============================================================ discovery ===


def iter_manifold_folders(
    namespace: Optional[str] = None,
) -> Iterator[tuple[str, ManifoldFolder]]:
    """Yield ``(namespace, ManifoldFolder)`` for every installed manifold.

    Walks ``~/.drowse/manifolds/<ns>/<name>/``.  Only directories whose names
    are valid artifact names count (staging, lock, and backup areas never
    match), and a folder must be named after its manifest: anything else is
    not an installed manifold.  Malformed folders are skipped, so one bad
    manifold never breaks a listing.  Discovery is metadata-only — fitted
    payloads are proven at explicit use / publish / install boundaries.
    """
    root = manifolds_dir()
    if not root.is_dir():
        return
    for ns_dir in sorted(root.iterdir()):
        if not ns_dir.is_dir() or not NAME_REGEX.match(ns_dir.name):
            continue
        if namespace is not None and ns_dir.name != namespace:
            continue
        for mdir in sorted(ns_dir.iterdir()):
            if not NAME_REGEX.match(mdir.name) or not (mdir / "manifold.json").is_file():
                continue
            try:
                mf = ManifoldFolder.load(mdir, verify_manifest=False)
            except ManifoldFormatError:
                continue
            if mf.name == mdir.name:
                yield ns_dir.name, mf


def _validate_ns_name(namespace: str, name: str) -> None:
    """Shared name/namespace gate for every folder-creation entry point."""
    if not NAME_REGEX.match(name):
        raise ManifoldFormatError(
            f"manifold name {name!r} invalid; must match {NAME_REGEX.pattern}"
        )
    if not NAME_REGEX.match(namespace):
        raise ManifoldFormatError(
            f"manifold namespace {namespace!r} invalid; must match {NAME_REGEX.pattern}"
        )
    if namespace in RESERVED_NAMESPACES:
        raise ManifoldFormatError(
            f"manifold namespace {namespace!r} is reserved for instrument "
            f"atoms ({', '.join(sorted(RESERVED_NAMESPACES))}), not "
            f"manifold folders"
        )


# ============================================================ validation ===


def _validate_statements(name: str, label: str, statements: Any) -> list[str]:
    if (
        not isinstance(statements, list)
        or not statements
        or not all(isinstance(s, str) and s.strip() for s in statements)
    ):
        raise ManifoldFormatError(
            f"manifold {name!r} node {label!r} needs a non-empty list of "
            f"non-blank statement strings"
        )
    return [str(s) for s in statements]


def _authored_nodes(
    name: str, domain: Any, nodes: Any,
) -> tuple[list[ManifestNode], list[tuple[str, list[str]]], list[tuple[float, ...]]]:
    """Validate the authoring shape ``[{label, coords, statements, role?, kind?}]``.

    Returns the manifest nodes, the per-node corpora and the validated
    per-node coords (every authored node has them).
    """
    n = domain.intrinsic_dim
    floor = min_nodes(n)
    if not isinstance(nodes, list) or len(nodes) < floor:
        raise ManifoldFormatError(
            f"manifold {name!r} ({n}-D domain) needs >= {floor} nodes, "
            f"got {len(nodes) if isinstance(nodes, list) else nodes!r}"
        )
    manifest_nodes: list[ManifestNode] = []
    corpora: list[tuple[str, list[str]]] = []
    node_coords: list[tuple[float, ...]] = []
    for entry in nodes:
        if not isinstance(entry, dict):
            raise ManifoldFormatError(f"manifold {name!r} node {entry!r} must be an object")
        label = validate_node_label(name, entry.get("label"))
        coords = entry.get("coords")
        if (
            not isinstance(coords, list)
            or len(coords) != n
            or not all(
                not isinstance(c, bool) and isinstance(c, (int, float))
                and math.isfinite(float(c))
                for c in coords
            )
        ):
            raise ManifoldFormatError(
                f"manifold {name!r} node {label!r} needs 'coords' of "
                f"{n} number(s), got {coords!r}"
            )
        corpora.append((label, _validate_statements(name, label, entry.get("statements"))))
        node_coords.append(tuple(float(c) for c in coords))
        manifest_nodes.append(ManifestNode(
            label=label,
            role=validate_node_role(name, label, entry.get("role")),
            kind=validate_node_kind(name, label, entry.get("kind")),
            coords=node_coords[-1],
        ))
    validate_node_labels(name, [node.label for node in manifest_nodes])
    return manifest_nodes, corpora, node_coords


def authoring_advisories(
    name: str, domain_spec: dict[str, Any], node_coords: Sequence[Sequence[float]],
) -> list[str]:
    """Soft warnings on an authored node layout — never raises.

    Poisedness (the embedded coordinates failing to affinely span the
    embedding) is a hard error at fit time; flagging it here gives the author
    the feedback before a fit is paid for.  A non-periodic box axis with
    fewer than three distinct values cannot show curvature.
    """
    import torch

    from drowse.geometry.domains import BoxDomain

    domain = domain_from_spec(domain_spec)
    embedded = domain.embed(torch.tensor([list(c) for c in node_coords], dtype=torch.float32))
    rank = int(torch.linalg.matrix_rank(embedded - embedded.mean(dim=0, keepdim=True)))
    advisories: list[str] = []
    if rank != domain.embed_dim:
        advisories.append(
            f"manifold {name!r}: node coordinates have affine rank {rank} "
            f"but the embedding is {domain.embed_dim}-dimensional — the fit "
            f"will fail as not poised; spread the nodes across every axis"
        )
    if isinstance(domain, BoxDomain):
        for i, axis in enumerate(domain.axes):
            distinct = {round(row[i], 9) for row in node_coords}
            if not axis.periodic and len(distinct) < 3:
                advisories.append(
                    f"manifold {name!r}: axis {axis.name!r} has only "
                    f"{len(distinct)} distinct coordinate value(s) — "
                    f"curvature along it cannot be fitted"
                )
    return advisories


# ============================================================ publication ===


def _publish_folder(
    folder: Path,
    build: Callable[[Path], None],
    *,
    force: bool,
    label: str,
) -> Path:
    """Build a whole manifold folder in staging and swap it into place.

    An existing manifest without ``force`` is a :class:`FileExistsError`; a
    directory with no manifest is a crash leftover and is replaced.
    """
    with locked_manifold_folder(folder):
        if (folder / "manifold.json").exists() and not force:
            raise FileExistsError(f"manifold {label} already exists")
        replace_manifold_folder(folder, build)
    invalidate_selector_index()
    return folder


def _corpus_builder(
    payload: dict[str, Any], corpora: list[tuple[str, list[str]]],
) -> Callable[[Path], None]:
    def build(staging: Path) -> None:
        write_node_corpus(staging, corpora)
        write_json_atomic(staging / "manifold.json", payload)
        ManifoldFolder.load(staging)
    return build


@_lock_namespace_name
def create_manifold_folder(
    namespace: str,
    name: str,
    description: str,
    domain_spec: dict[str, Any],
    nodes: list[dict[str, Any]],
) -> tuple[Path, list[str]]:
    """Author a fresh authored-geometry manifold folder.

    ``domain_spec`` is the ``manifold.json`` ``domain`` tagged union and
    ``nodes`` the authoring shape ``[{label, coords, statements, role?,
    kind?}]``.  Returns ``(folder, advisories)`` — the soft poisedness /
    flat-axis warnings, so the UI can flag a deficient layout before a fit.
    Raises :class:`ManifoldFormatError` on invalid input and
    :class:`FileExistsError` when the manifold exists.
    """
    from drowse.geometry.domains import normalize_domain_spec

    _validate_ns_name(namespace, name)
    try:
        domain = domain_from_spec(normalize_domain_spec(domain_spec))
    except (ValueError, KeyError) as exc:
        raise ManifoldFormatError(f"invalid manifold domain: {exc}") from exc
    manifest_nodes, corpora, node_coords = _authored_nodes(name, domain, nodes)
    payload = manifest_payload(
        name=name, description=description, fit_mode="authored",
        nodes=manifest_nodes, domain=domain.to_spec(),
    )
    folder = _publish_folder(
        manifold_dir(namespace, name), _corpus_builder(payload, corpora),
        force=False, label=f"{namespace}/{name}",
    )
    return folder, authoring_advisories(name, payload["domain"], node_coords)


@_lock_namespace_name
def create_discover_manifold_folder(
    namespace: str,
    name: str,
    description: str,
    *,
    fit_mode: str,
    node_corpora: dict[str, list[str]],
    hyperparams: Optional[dict[str, Any]] = None,
    node_roles: Optional[dict[str, str | None]] = None,
    node_kinds: Optional[dict[str, str | None]] = None,
    template_ref: Optional[str] = None,
    tags: Sequence[str] = (),
    force: bool = False,
) -> Path:
    """Author a fresh discover-mode manifold from ``{label: [statement, ...]}``.

    Coordinates are derived per model at fit time, so there is no domain and
    no authoring advisory here.  ``force`` replaces an existing folder in the
    same staged swap.  Raises :class:`ManifoldFormatError` on invalid input
    and :class:`FileExistsError` when the manifold exists without ``force``.
    """
    _validate_ns_name(namespace, name)
    require_discover_mode(name, fit_mode)
    if not isinstance(node_corpora, dict) or not node_corpora:
        raise ManifoldFormatError(
            f"discover manifold {name!r} needs a non-empty {{label: [statements]}} dict"
        )
    labels = validate_node_labels(name, list(node_corpora))
    corpora = [
        (label, _validate_statements(name, label, node_corpora[label])) for label in labels
    ]
    roles, kinds = resolve_node_maps(name, labels, node_roles, node_kinds)
    payload = manifest_payload(
        name=name, description=description, fit_mode=fit_mode,
        nodes=[ManifestNode(label, roles[label], kinds[label]) for label in labels],
        hyperparams=sanitize_hyperparams(fit_mode, hyperparams),
        tags=tags, template_ref=template_ref,
    )
    return _publish_folder(
        manifold_dir(namespace, name), _corpus_builder(payload, corpora),
        force=force, label=f"{namespace}/{name}",
    )


@_lock_namespace_name
def create_manifold_from_template(
    namespace: str,
    name: str,
    description: str,
    *,
    template_ref: str,
    fit_mode: str,
    hyperparams: Optional[dict[str, Any]] = None,
    force: bool = False,
) -> Path:
    """Author a discover manifold whose node corpora derive from a template.

    Resolves the template, expands ``values × contexts`` into per-value node
    corpora (the slot-filled assistant turns, ``corpus[i]`` aligned to
    ``contexts[i]``), and writes a discover folder carrying both that corpus
    and the ``template_ref`` the fit resolves for its elicitation prefixes.
    """
    from drowse.store.templates import resolve_template

    _validate_ns_name(namespace, name)
    require_discover_mode(name, fit_mode)
    tmpl = resolve_template(template_ref)
    ref = f"{tmpl.path.parent.name}/{tmpl.name}" if tmpl.path else tmpl.name
    return create_discover_manifold_folder(
        namespace, name, description,
        fit_mode=fit_mode,
        node_corpora=tmpl.node_corpora(),
        hyperparams=sanitize_hyperparams(fit_mode, hyperparams),
        template_ref=ref,
        force=force,
    )


@dataclass(frozen=True)
class BakedTensor:
    """One model's fitted tensor for :func:`publish_baked_manifold`.

    ``manifold`` is an already-fitted :class:`drowse.geometry.manifold.Manifold`
    (typically the affine ray from ``fold_directions_to_subspace``);
    ``components`` is merge provenance.  ``model_id_is_safe`` marks a
    ``model_id`` that is already a safe filename component.
    """

    model_id: str
    manifold: Any
    model_fingerprint: str
    components: Optional[dict[str, Any]] = None
    model_id_is_safe: bool = False


@_lock_namespace_name
def publish_baked_manifold(
    namespace: str,
    name: str,
    description: str,
    tensors: Sequence[BakedTensor],
    *,
    method: str,
    tags: Sequence[str] = (),
    source: str = "local",
    force: bool = False,
) -> Path:
    """Publish a corpus-less ``fit_mode="baked"`` manifold in one staged swap.

    The single producer for baked folders (merge outputs).  Every per-model
    tensor pair and the manifest with its proofs are written and verified in
    staging, then the whole folder swaps into place — there is no partially
    published state to resume.  ``method`` is the sidecar provenance tag
    (``"merge"`` stamps the merge bake policy).  Raises
    :class:`FileExistsError` when the manifold exists and ``force`` is unset.
    """
    from drowse.store.manifold_tensors import save_manifold

    _validate_ns_name(namespace, name)
    if not tensors:
        raise ValueError("a baked manifold needs at least one fitted tensor")
    first = tensors[0].manifold
    labels = list(first.node_labels)
    if any(list(t.manifold.node_labels) != labels for t in tensors):
        raise ValueError("every baked tensor must carry the same node labels")
    outputs = []
    for item in tensors:
        if not item.model_fingerprint:
            raise ValueError("baked manifold tensors require a proven model fingerprint")
        outputs.append((
            tensor_filename(item.model_id, model_id_is_safe=item.model_id_is_safe), item,
        ))
    # Provenance only — a baked folder never re-fits — keyed exactly as
    # ``ManifoldFolder.nodes_sha256`` hashes a baked folder.
    nodes_sha = hashlib.sha256(
        canonical_json({"fit_mode": "baked", "node_labels": labels})
    ).hexdigest()

    def build(staging: Path) -> None:
        files: dict[str, str] = {}
        for filename, item in outputs:
            meta: dict[str, object] = {
                "method": method, "fit_mode": "baked", "nodes_sha256": nodes_sha,
                "model_fingerprint": item.model_fingerprint,
            }
            share_metric = item.manifold.metadata.get("share_metric")
            if share_metric:
                meta["share_metric"] = share_metric
            if item.components:
                meta["components"] = item.components
            if method == "merge":
                meta["bake_policy"] = MERGE_BAKE_POLICY
            tensor_path = staging / filename
            save_manifold(item.manifold, tensor_path, meta)
            for half in (tensor_path, tensor_path.with_suffix(".json")):
                files[half.name] = hash_file(half)
        write_json_atomic(staging / "manifold.json", manifest_payload(
            name=name, description=description, fit_mode="baked",
            nodes=[ManifestNode(label) for label in labels],
            domain=first.domain.to_spec(), files=files, source=source, tags=tags,
        ))
        # Structural check only: the pairs were just written by the
        # validating writer and hashed above.
        ManifoldFolder.load(staging, verify_manifest=False)

    return _publish_folder(
        manifold_dir(namespace, name), build, force=force, label=f"{namespace}/{name}",
    )


# ============================================================ streaming skeleton ===


@_lock_namespace_name
def init_discover_manifold_folder(
    namespace: str,
    name: str,
    description: str,
    *,
    fit_mode: str,
    labels: list[str],
    hyperparams: Optional[dict[str, Any]] = None,
    node_roles: Optional[dict[str, str | None]] = None,
    node_kinds: Optional[dict[str, str | None]] = None,
) -> Path:
    """Create a discover skeleton for streaming node writes.

    Writes ``manifold.json`` (label-only nodes) and an empty ``nodes/``; node
    corpora then arrive one file at a time through
    :func:`append_discover_manifold_node`, so a crash keeps finished nodes.
    Raises :class:`FileExistsError` when the manifold exists.
    """
    _validate_ns_name(namespace, name)
    require_discover_mode(name, fit_mode)
    labels = validate_node_labels(name, labels)
    roles, kinds = resolve_node_maps(name, labels, node_roles, node_kinds)
    folder = manifold_dir(namespace, name)
    if (folder / "manifold.json").exists():
        raise FileExistsError(f"manifold {namespace}/{name} already exists")
    (folder / "nodes").mkdir(parents=True, exist_ok=True)
    write_json_atomic(folder / "manifold.json", manifest_payload(
        name=name, description=description, fit_mode=fit_mode,
        nodes=[ManifestNode(label, roles[label], kinds[label]) for label in labels],
        hyperparams=sanitize_hyperparams(fit_mode, hyperparams),
    ))
    invalidate_selector_index()
    return folder


@_lock_folder_arg
def append_discover_manifold_node(
    folder: Path, index: int, label: str, statements: list[str],
) -> None:
    """Write one discover node corpus to ``nodes/NN_<label>.json`` atomically.

    ``index`` is the node's position in the skeleton's label order.  There
    is deliberately no whole-corpus swap: a run that dies part-way keeps the
    nodes it already finished, and the node that completes the corpus is
    what makes the manifold discoverable.
    """
    validate_node_label("discover manifold", label)
    corpus = _validate_statements("discover manifold", label, statements)
    nodes_dir = Path(folder) / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(nodes_dir / node_filename(index, label), corpus)
    invalidate_selector_index()


@_lock_folder_arg
def plan_discover_generation(
    folder: Path,
    name: str,
    description: str,
    *,
    fit_mode: str,
    labels: list[str],
    hyperparams: Optional[dict[str, Any]] = None,
    node_roles: Optional[dict[str, str | None]] = None,
    node_kinds: Optional[dict[str, str | None]] = None,
    tags: Optional[list[str]] = None,
    force: bool = False,
) -> DiscoverGenerationPlan:
    """Ensure a streamable discover skeleton covering ``labels``; report pending nodes.

    Resume and add-nodes in one — the single planner every discover-generate
    surface calls:

    - a fresh ``folder`` gets a label-only skeleton and every label is pending;
    - an existing discover folder keeps its node order, appends labels new to
      it (its fits go stale through the corpus hash), and reports as pending
      the declared labels whose corpus file is absent;
    - ``force`` removes the prior folder inside the same manifest transaction.

    The existing manifest is parsed without demanding corpus files, so the
    planner can inspect the very partial it resumes.  Description (and
    ``tags``, when given) refresh to the caller's value; existing fit
    hyperparameters, roles, and kinds are kept.
    """
    folder = Path(folder)
    require_discover_mode(name, fit_mode)
    labels = validate_node_labels(name, labels)
    roles, kinds = resolve_node_maps(name, labels, node_roles, node_kinds)
    meta_path = folder / "manifold.json"
    if force or not meta_path.exists():
        skeleton = manifest_payload(
            name=name, description=description, fit_mode=fit_mode,
            nodes=[ManifestNode(label, roles[label], kinds[label]) for label in labels],
            hyperparams=sanitize_hyperparams(fit_mode, hyperparams), tags=tags or (),
        )
        if force and folder.exists():
            # One swap replaces the prior folder with the empty skeleton:
            # there is no window where the old folder is half-removed.
            def build(stage: Path) -> None:
                (stage / "nodes").mkdir()
                write_json_atomic(stage / "manifold.json", skeleton)

            replace_manifold_folder(folder, build)
        else:
            (folder / "nodes").mkdir(parents=True, exist_ok=True)
            write_json_atomic(meta_path, skeleton)
        invalidate_selector_index()
        return DiscoverGenerationPlan(
            folder=folder,
            index_of={label: i for i, label in enumerate(labels)},
            pending=tuple(labels),
            added=(),
            resumed=False,
            kinds=dict(kinds),
            roles=dict(roles),
        )

    spec = parse_manifest(load_manifest_data(meta_path), location=str(meta_path))
    if spec.fit_mode not in FIT_MODES_DISCOVER:
        raise ManifoldFormatError(
            f"manifold at {folder} is not discover-mode; cannot stream-generate into it"
        )
    if spec.name != name:
        raise ManifoldFormatError(f"manifold at {folder} is named {spec.name!r}, not {name!r}")
    existing = spec.labels
    new_labels = [label for label in labels if label not in existing]

    def has_corpus(index: int, label: str) -> bool:
        return (folder / "nodes" / node_filename(index, label)).exists()

    # Kind is generation provenance: a node about to be generated with no
    # recorded kind takes the caller's, so its corpus and manifest agree.
    nodes = [
        replace(node, kind=kinds.get(node.label))
        if node.kind is None and not has_corpus(i, node.label) else node
        for i, node in enumerate(spec.nodes)
    ] + [ManifestNode(label, roles[label], kinds[label]) for label in new_labels]
    updated = spec_payload(spec)
    updated["description"] = description
    if tags is not None:
        updated["tags"] = [str(tag) for tag in tags]
    updated["nodes"] = [node.payload() for node in nodes]
    if updated != spec_payload(spec):
        write_json_atomic(meta_path, updated)
        invalidate_selector_index()
    (folder / "nodes").mkdir(exist_ok=True)
    index_of = {node.label: i for i, node in enumerate(nodes)}
    return DiscoverGenerationPlan(
        folder=folder,
        index_of=index_of,
        pending=tuple(
            node.label for i, node in enumerate(nodes) if not has_corpus(i, node.label)
        ),
        added=tuple(new_labels),
        resumed=True,
        kinds={node.label: node.kind for node in nodes},
        roles={node.label: node.role for node in nodes},
    )


# ============================================================ merge / update ===


@_lock_namespace_name
def merge_discover_manifolds(
    target_namespace: str,
    target_name: str,
    target_description: str,
    *,
    sources: list[tuple[str, str]],
    fit_mode: Optional[str] = None,
    hyperparams: Optional[dict[str, Any]] = None,
    force: bool = False,
) -> Path:
    """Union N discover manifolds' node corpora into one fresh, unfitted folder.

    Discover sources only: authored manifolds carry user-declared geometry
    that two unrelated coordinate systems cannot share, while discover
    coordinates are re-derived per model from the pooled centroids.  A label
    present in two sources raises (labels carry provenance; no silent
    renames).

    ``fit_mode`` defaults to the sources' common mode and is required when
    they disagree.  ``hyperparams`` replaces them wholesale; otherwise the
    first source fitted in the target mode supplies them, and a mode no
    source used starts from the defaults (a mode's hyperparameters never
    carry across modes).
    """
    if len(sources) < 2:
        raise RequestError(f"merge needs >= 2 sources, got {len(sources)}: {sources!r}")
    label = f"{target_namespace}/{target_name}"
    folders: list[tuple[str, ManifoldFolder]] = []
    for ns, name in sources:
        path = manifold_dir(ns, name)
        if not (path / "manifold.json").exists():
            raise FileNotFoundError(f"merge source {ns}/{name} not found at {path}")
        mf = ManifoldFolder.load(path, verify_manifest=False)
        if not mf.is_discover:
            raise ManifoldFormatError(
                f"merge source {ns}/{name} is {mf.fit_mode!r}; merge supports "
                f"discover-mode (autofitted) manifolds only — authored manifolds "
                f"carry user-declared geometry that isn't mergeable without a "
                f"shared coordinate system."
            )
        folders.append((f"{ns}/{name}", mf))

    owner: dict[str, str] = {}
    collisions: list[str] = []
    for src_key, mf in folders:
        for node_label in mf.node_labels:
            if node_label in owner:
                collisions.append(f"{node_label!r} in {owner[node_label]} and {src_key}")
            owner.setdefault(node_label, src_key)
    if collisions:
        raise ManifoldFormatError(
            f"merge {label}: label collisions — {'; '.join(collisions)}. "
            f"Rename one side before merging."
        )

    source_modes = sorted({mf.fit_mode for _, mf in folders})
    if fit_mode is None:
        if len(source_modes) > 1:
            raise ManifoldFormatError(
                f"merge {label}: sources have mixed fit_modes ({source_modes}); "
                f"pass fit_mode= to pick one."
            )
        fit_mode = source_modes[0]
    require_discover_mode(target_name, fit_mode)
    if hyperparams is None:
        hyperparams = next(
            (dict(mf.hyperparams) for _, mf in folders if mf.fit_mode == fit_mode), {},
        )

    node_corpora: dict[str, list[str]] = {}
    node_roles: dict[str, str | None] = {}
    node_kinds: dict[str, str | None] = {}
    for _src, mf in folders:
        groups = dict(mf.node_groups())
        for idx, node_label in enumerate(mf.node_labels):
            node_corpora[node_label] = list(groups[node_label])
            node_roles[node_label] = mf.roles[idx]
            node_kinds[node_label] = mf.kinds[idx]

    target = manifold_dir(target_namespace, target_name)
    if (target / "manifold.json").exists() and not force:
        raise FileExistsError(f"manifold {label} already exists; pass force=True to overwrite")
    return create_discover_manifold_folder(
        target_namespace, target_name, target_description,
        fit_mode=fit_mode,
        node_corpora=node_corpora,
        hyperparams=hyperparams,
        node_roles=node_roles,
        node_kinds=node_kinds,
        force=force,
    )


@_lock_folder_arg
def update_manifold_folder(
    folder: Path,
    *,
    description: Optional[str] = None,
    nodes: Optional[list[dict[str, Any]]] = None,
) -> tuple[Path, list[str]]:
    """Re-author an existing manifold folder.

    ``description`` applies to any manifold.  ``nodes`` (authored manifolds
    only — a discover corpus is regenerated, a baked manifold has none) fully
    replaces the labels, coords, and corpus; existing fits go stale through
    ``nodes_sha256`` and the next fit overwrites them.  Returns
    ``(folder, advisories)``.
    """
    folder = Path(folder)
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    if nodes is not None and mf.fit_mode != "authored":
        raise ManifoldFormatError(
            f"manifold {mf.name!r} is {mf.fit_mode!r}; only an authored manifold's "
            f"node list can be re-authored in place"
        )
    if description is not None:
        mf.description = description
    if nodes is not None:
        manifest_nodes, corpora, node_coords = _authored_nodes(
            mf.name, domain_from_spec(mf.domain), nodes,
        )
        write_node_corpus(folder, corpora)
        mf.node_labels = [node.label for node in manifest_nodes]
        mf.node_coords = [list(coords) for coords in node_coords]
        mf.node_roles = [node.role for node in manifest_nodes]
        mf.node_kinds = [node.kind for node in manifest_nodes]
    mf.write_metadata()
    invalidate_selector_index()
    advisories = (
        authoring_advisories(mf.name, mf.domain, mf.node_coords)
        if mf.fit_mode == "authored" else []
    )
    return folder, advisories
