"""Native ``/drowse/v1/manifolds/*`` routes — manifold steering artifacts.

A manifold is a top-level artifact (labeled nodes on a domain), not
session-scoped, so these routes live beside the pack routes rather than
under ``/sessions/{id}``.  The exception is ``fit``, which needs the
loaded model — it runs ``session.fit`` under the session
lock and streams progress like ``/extract`` does.

Authoring (create / update) writes ``manifold.json`` + ``nodes/*.json``
through :mod:`drowse.store.manifold_folder`; steering a fitted manifold needs no
route — a ``%`` term in any steering expression already loads the
artifact lazily on scope entry.
"""
from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import FastAPI, HTTPException, Request
from pydantic import ConfigDict, Field

from drowse.geometry.domains import domain_from_spec
from drowse.model.fingerprint import loaded_model_fingerprint
from drowse.session.session import DrowseSession
from drowse.store.hub import search_manifolds
from drowse.store.install import install_manifold
from drowse.store.manifold_folder import ManifoldFolder
from drowse.store.manifold_schema import ManifoldFormatError, domain_label, min_nodes
from drowse.store.manifold_authoring import (
    create_discover_manifold_folder,
    create_manifold_folder,
    create_manifold_from_template,
    iter_manifold_folders,
    merge_discover_manifolds,
    plan_discover_generation,
)
from drowse.store.manifold_lifecycle import manifold_summary, remove_manifold_folder
from drowse.base.paths import manifold_dir
from drowse.server.operations import ProgressCallback, run_operation
from drowse.server.runtime import NativeRequest, refuse_if_busy, run_in_thread, session_lock
from drowse.server.response_models import (
    ManifoldDeleteResponse,
    ManifoldFitInfo,
    ManifoldInfo,
    ManifoldListResponse,
    ManifoldSearchResponse,
    RemoteManifoldInfo,
)


# ----------------------------------------------------------- request models ---

class BoxAxisSpec(NativeRequest):
    name: str = "axis"
    periodic: bool = False
    period: float = 1.0
    lo: float = 0.0
    hi: float = 1.0


class DomainSpec(NativeRequest):
    """Built-in authoring domains (custom immersions are JSON-authored)."""

    type: Literal["box", "sphere", "klein", "projective"]
    axes: list[BoxAxisSpec] | None = None
    dim: int | None = None


class NodeSpec(NativeRequest):
    label: str
    coords: list[float]
    statements: list[str]
    # Optional per-node assistant-role substitution.  ``None`` (default)
    # = standard assistant baseline.  When set, this node's centroid is
    # pooled with the chat-template's assistant-role label replaced by
    # this slug.  Engine validates the slug shape (``[a-z0-9._-]+``)
    # and the family's support at fit time.
    role: str | None = None


class DiscoverNodeSpec(NativeRequest):
    """A node in a discover-mode authoring payload — label + statements only.

    No ``coords``: coords are derived per-model at fit time from the
    pooled centroids.  ``role`` (optional) carries the per-node
    assistant-role substitution; see :class:`NodeSpec`.
    """

    label: str
    statements: list[str]
    role: str | None = None


class CreateManifoldRequest(NativeRequest):
    namespace: str = "local"
    name: str
    description: str = ""
    domain: DomainSpec
    nodes: list[NodeSpec]


class CreateDiscoverManifoldRequest(NativeRequest):
    """Author a discover-mode manifold from supplied per-concept corpora.

    The user provides labeled statement corpora; the fit derives node
    coordinates per-model via PCA or spectral embedding when the
    matching ``POST .../fit`` runs.
    """

    namespace: str = "local"
    name: str
    description: str = ""
    fit_mode: Literal["pca", "spectral", "auto"] = "pca"
    nodes: list[DiscoverNodeSpec]
    hyperparams: dict[str, Any] = {}


class CreateManifoldFromTemplateRequest(NativeRequest):
    """Author a discover manifold derived from a standalone template."""

    namespace: str = "local"
    name: str
    description: str = ""
    fit_mode: Literal["pca", "spectral", "auto"] = "auto"
    template_ref: str
    hyperparams: dict[str, Any] = {}
    force: bool = False


class GenerateManifoldRequest(NativeRequest):
    """LLM-author a discover-mode manifold from a flat concept list.

    Wraps :meth:`DrowseSession.generate_manifold` — produces one
    conversational corpus per concept (in-character responses to the shared
    baseline prompts, selected by ``kind``).  No coords supplied; the fit
    derives them per-model.
    """

    namespace: str = "local"
    name: str
    description: str = ""
    concepts: list[str]
    kind: Literal["abstract", "concrete", "custom"] = "abstract"
    # Required when ``kind == "custom"`` — the free-form elicitation system
    # prompt (``{c}`` = the concept), no role swap (pools in standard space).
    custom_system: str | None = None
    samples_per_prompt: int = 1
    # Seeds the sampler: each node's corpus is reproducible from ``seed`` and
    # its node index (``None``: unseeded), like ``manifold generate --seed``.
    seed: int | None = None
    fit_mode: Literal["pca", "spectral", "auto"] = "pca"
    hyperparams: dict[str, Any] = {}
    force: bool = False
    # ``role_per_node=True`` (the GUI "use slug as per-node role"
    # checkbox) means each ``concepts[i]`` slug doubles as that node's
    # assistant-role substitution at fit time — producing a persona
    # manifold where the geometry captures persona-relative structure
    # in role-baselined activation space.
    role_per_node: bool = False


class FitManifoldRequest(NativeRequest):
    """Body for ``POST /manifolds/{ns}/{name}/fit``.

    For authored manifolds only ``sae`` is honored.
    For discover-mode manifolds ``fit_mode`` and ``hyperparams`` can
    override the folder's stored values; when provided, the folder
    manifest is rewritten *before* the fit so the cache key reflects
    the actual fit inputs.
    """

    sae: str | None = None
    layers: list[int] | Literal["workspace", "all"] | None = None
    force: bool = False
    # Discover-mode override fields — ignored when the folder is authored.
    fit_mode: Literal["pca", "spectral", "auto"] | None = None
    hyperparams: dict[str, Any] | None = None


class InstallManifoldRequest(NativeRequest):
    """Body for ``POST /manifolds/install``.

    ``target`` is an HF coord (``owner/name[@revision]``) or a local folder path;
    ``as_`` re-namespaces the destination; ``force`` overwrites an existing
    folder at the destination.
    """

    target: str
    as_: str | None = Field(default=None, alias="as")
    force: bool = False

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class MergeManifoldSource(NativeRequest):
    """One source folder in a manifold merge — fully qualified ``ns/name``."""

    namespace: str
    name: str


class MergeManifoldRequest(NativeRequest):
    """Body for ``POST /manifolds/merge``.

    Restricted to discover-mode (autofitted) sources by design — see
    :func:`drowse.store.manifold_authoring.merge_discover_manifolds`.  Pools the
    sources' node corpora + roles into one heap, writes a fresh
    discover folder; the next ``POST .../fit`` derives coords from the
    combined heap.
    """

    namespace: str = "local"
    name: str
    description: str = ""
    sources: list[MergeManifoldSource]
    # Defaults to the sources' shared fit_mode if they agree, else
    # required.  Server-side reconciliation lives in
    # ``merge_discover_manifolds``.
    fit_mode: Literal["pca", "spectral", "auto"] | None = None
    hyperparams: dict[str, Any] | None = None
    force: bool = False


def _intrinsic_dim(spec: dict[str, Any]) -> int:
    try:
        return domain_from_spec(spec).intrinsic_dim
    except (ValueError, KeyError):
        return 0


def _resolve_intrinsic_dim(mf: ManifoldFolder, session_stem: str) -> tuple[int, dict[str, Any]]:
    """Resolve the manifold's intrinsic dimension + effective domain spec.

    Authored folders carry the domain inline on ``manifold.json``; the
    answer is just ``_intrinsic_dim(mf.domain)`` and ``mf.domain`` itself.

    Discover folders carry an empty top-level ``domain`` — the real
    ``CustomDomain(picked_k)`` is materialized at fit time and lives on
    the per-model sidecar.  When a fit exists for the loaded model we
    read the sidecar's ``domain`` so the wire reports the actual dim
    (otherwise the webui's manifold strip renders zero controls on an
    otherwise-fitted persona manifold).

    Returns ``(intrinsic_dim, domain_spec)``.  When no fit is available
    on a discover folder we still report 0 / ``{}`` — the strip's
    "unfitted" warning covers that state.
    """
    n = _intrinsic_dim(mf.domain)
    if n > 0:
        return n, mf.domain
    if mf.is_discover and session_stem in mf.tensor_models():
        try:
            sc_domain = mf.sidecar(session_stem).domain
            return _intrinsic_dim(sc_domain), sc_domain
        except (KeyError, ManifoldFormatError):
            pass
    return 0, mf.domain


def _fitted_geometry(path: Path) -> tuple[list[list[float]], bool | None]:
    """Read a fitted tensor's shared layout + flat/curved family cheaply.

    Returns ``(node_coords, is_affine)``.  Both answers live in the
    safetensors *header* plus one small ``(K, n)`` entry, so this opens the
    file lazily instead of deserializing the whole per-layer
    ``mean``/``basis``/RBF payload the way ``load_manifold`` does — the list
    route calls this once per installed manifold.

    The flat/curved discriminator is the same one ``load_manifold`` keys on:
    the *absence* of any ``layer_<L>.node_params`` entry marks an affine fit.
    ``([], None)`` when the artifact is missing or unreadable — the caller
    reports an unresolved geometry rather than failing the whole listing.
    """
    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            keys = set(tensors.keys())
            if "node_coords" not in keys:
                return [], None
            is_affine = not any(k.endswith(".node_params") for k in keys)
            coords = tensors.get_tensor("node_coords")
    except (OSError, SafetensorError, ValueError, KeyError):
        return [], None
    return [[float(x) for x in row] for row in coords.tolist()], is_affine


def _manifold_json(
    mf: ManifoldFolder,
    session: DrowseSession,
    *,
    full: bool,
    fingerprint: Callable[[], str] | None = None,
) -> ManifoldInfo:
    """Serialize a :class:`ManifoldFolder` for the wire.

    The session-independent fields come straight from
    :func:`drowse.store.manifold_lifecycle.manifold_summary` so this route and the
    CLI ``manifold show -j`` emit byte-identical values for every
    key they share (``namespace`` / ``name`` / ``description`` /
    ``source`` / ``fit_mode`` / ``is_discover`` / ``node_count`` /
    ``node_labels`` / ``node_roles`` / ``hyperparams`` / ``fitted_models``
    / ``tensor_variants``).  The server then layers the *session-aware*
    extras on top: ``fitted_for_session`` / ``stale``, plus — for a
    discover folder fitted on the loaded model — the materialized
    per-model geometry (``domain`` / ``domain_label`` / ``intrinsic_dim``
    / ``min_nodes`` / ``node_coords``) that ``manifold_summary`` leaves
    empty by design (it can't read the per-model safetensors without a
    session).

    ``full`` adds per-node statements and per-tensor fit detail — the
    list route omits both to stay light.  ``fingerprint`` supplies the
    loaded-weight fingerprint, so a listing computes it once, not per row.
    """
    out = cast(ManifoldInfo, manifold_summary(mf.folder))

    from drowse.base.paths import tensor_filename

    session_stem = tensor_filename(session.model_id).removesuffix(".safetensors")
    fitted_models = out["fitted_models"]
    fitted_for_session = session_stem in fitted_models
    n, effective_domain = _resolve_intrinsic_dim(mf, session_stem)

    stale = False
    if fitted_for_session:
        if fingerprint is None:
            fingerprint = _fingerprint(session)
        try:
            sidecar = mf.sidecar(session_stem)
            # The (memoized) weight fingerprint is far cheaper than hashing
            # every node corpus, so it short-circuits first.
            stale = (
                sidecar.model_fingerprint != fingerprint()
                or sidecar.nodes_sha256 != mf.nodes_sha256()
            )
        except (KeyError, ManifoldFormatError, OSError):
            stale = True

    # Roles are index-aligned with ``node_labels``; ``manifold_summary``
    # already padded them (``ManifoldFolder.roles``) so a consumer can ``zip``
    # against ``node_labels`` without a length check.  Reuse below.
    node_roles_padded = out["node_roles"]

    # For fitted discover folders the derived per-model coords live in
    # the safetensors, not on the folder.  Read them once and share
    # between the list-level ``node_coords`` field and the detail-level
    # ``nodes`` block — the list shape needs them so the manifold rack
    # strip's snap-to-node action can sync the position sliders to the
    # picked node's actual coords (otherwise label-form selections show
    # zeros on every axis).  Header + one small entry, not a full
    # per-layer deserialization (see :func:`_fitted_geometry`).
    derived_coords: list[list[float]] = []
    resolved_affine: bool | None = None
    if fitted_for_session and mf.is_discover:
        derived_coords, resolved_affine = _fitted_geometry(
            mf.tensor_path(session_stem),
        )

    # Session-aware geometry override.  ``manifold_summary`` reports the
    # empty/discover form for a discover folder (no session, no per-model
    # read); the server lifts the materialized ``CustomDomain(picked_k)``
    # spec + derived coords from the per-model sidecar/tensor when a fit
    # exists so the rack strip can build N sliders and snap to a node.
    # Authored folders already have the right values from the summary, so
    # only touch these on the discover path.
    if mf.is_discover:
        out["domain"] = effective_domain
        out["domain_label"] = (
            domain_label(effective_domain)
            # An unfitted discover folder has no geometry yet.  Preserve the
            # summary's useful ``discover-<mode>`` label instead of replacing
            # it with the technically-derived but user-hostile ``?(0d)``.
            if effective_domain else out["domain_label"]
        )
        out["intrinsic_dim"] = n
        out["min_nodes"] = min_nodes(n) if n > 0 else None
        out["node_coords"] = derived_coords

    # Resolved flat/curved geometry for the rack family split.  An
    # ``auto`` discover folder's family is per-model and only known
    # post-fit, so the client can't route it to the subspace (flat) vs
    # manifold (curved) drawer off ``fit_mode`` alone.  Surface the
    # resolved mode: ``"pca"`` (flat) / ``"spectral"`` (curved) for a
    # fitted discover folder, the concrete ``fit_mode`` for an authored /
    # baked folder, and ``None`` when an ``auto`` folder isn't fitted for
    # this model yet (the client shows an unresolved auto manifold in
    # both drawers until a fit pins the geometry).
    if mf.is_discover:
        out["resolved_fit_mode"] = (
            None if resolved_affine is None
            else ("pca" if resolved_affine else "spectral")
        )
    else:
        out["resolved_fit_mode"] = mf.fit_mode

    # Session-only extras layered on top of the shared summary.
    out["fitted_for_session"] = fitted_for_session
    out["stale"] = stale

    if full:
        groups = dict(mf.node_groups())
        if mf.fit_mode == "authored":
            out["nodes"] = [
                {
                    "label": label,
                    "coords": list(coords),
                    "statements": list(groups.get(label, [])),
                    "role": role,
                }
                for label, coords, role in zip(
                    mf.node_labels, mf.node_coords, node_roles_padded,
                    strict=True,
                )
            ]
        else:
            # Discover: nodes carry the derived per-model coords loaded
            # above (or ``None`` per node when no fit exists yet).
            out["nodes"] = [
                {
                    "label": label,
                    "coords": (
                        derived_coords[i] if i < len(derived_coords) else None
                    ),
                    "statements": list(groups.get(label, [])),
                    "role": node_roles_padded[i],
                }
                for i, label in enumerate(mf.node_labels)
            ]
        fitted: list[ManifoldFitInfo] = []
        for stem in fitted_models:
            try:
                sc = mf.sidecar(stem)
            except (KeyError, ManifoldFormatError):
                # No pair for this stem, or a stale / malformed sidecar:
                # sidecars load lazily, so a bad fit is skipped, not a 500.
                continue
            entry: ManifoldFitInfo = {
                "stem": stem,
                "method": sc.method,
                "feature_space": sc.feature_space,
                "node_count": sc.node_count,
                "nodes_sha256": sc.nodes_sha256,
                "fit_mode": sc.fit_mode,
            }
            if sc.hyperparams:
                entry["hyperparams"] = sc.hyperparams
            if sc.diagnostics:
                entry["diagnostics"] = sc.diagnostics
            fitted.append(entry)
        out["fitted"] = fitted

    return out


def _fingerprint(session: DrowseSession) -> Callable[[], str]:
    """The loaded-weight fingerprint, computed at most once per caller."""
    return functools.cache(lambda: loaded_model_fingerprint(session.model, session.model_id))


def _find_manifold(namespace: str, name: str) -> ManifoldFolder:
    """Load one manifold folder by namespace + name, or 404."""
    folder = manifold_dir(namespace, name)
    if not (folder / "manifold.json").exists():
        raise HTTPException(404, f"manifold {namespace}/{name} not found")
    return ManifoldFolder.load(folder, verify_manifest=False)


# ------------------------------------------------------------------- routes ---

def register_manifold_routes(app: FastAPI) -> None:
    """Mount the ``/drowse/v1/manifolds/*`` tree onto ``app``."""

    session: DrowseSession = app.state.session

    def detail(namespace: str, name: str) -> ManifoldInfo:
        return _manifold_json(_find_manifold(namespace, name), session, full=True)

    @app.get("/drowse/v1/manifolds")
    def list_manifolds() -> ManifoldListResponse:
        """List every installed manifold with per-session fit status."""
        fingerprint = _fingerprint(session)
        return {"manifolds": [
            _manifold_json(mf, session, full=False, fingerprint=fingerprint)
            for _ns, mf in iter_manifold_folders()
        ]}

    @app.get("/drowse/v1/manifolds/{namespace}/{name}")
    def get_manifold(namespace: str, name: str) -> ManifoldInfo:
        """One manifold: domain, nodes with statements, per-tensor fit detail."""
        return detail(namespace, name)

    @app.post("/drowse/v1/manifolds", status_code=201)
    def create_manifold(req: CreateManifoldRequest) -> ManifoldInfo:
        """Author an authored-geometry manifold (``manifold.json`` + ``nodes/``).

        The response adds ``advisories`` — soft poisedness / flat-axis
        warnings, so a deficient layout surfaces before a fit is paid for.
        """
        folder, advisories = create_manifold_folder(
            req.namespace, req.name, req.description,
            req.domain.model_dump(exclude_none=True), [n.model_dump() for n in req.nodes],
        )
        body = _manifold_json(ManifoldFolder.load(folder), session, full=True)
        body["advisories"] = advisories
        return body

    @app.post("/drowse/v1/manifolds/discover", status_code=201)
    def create_discover_manifold(req: CreateDiscoverManifoldRequest) -> ManifoldInfo:
        """Author a discover manifold from labeled corpora; coords derive at fit time."""
        folder = create_discover_manifold_folder(
            req.namespace, req.name, req.description,
            fit_mode=req.fit_mode,
            node_corpora={n.label: list(n.statements) for n in req.nodes},
            hyperparams=req.hyperparams,
            node_roles={n.label: n.role for n in req.nodes},
        )
        return _manifold_json(ManifoldFolder.load(folder), session, full=True)

    @app.post("/drowse/v1/manifolds/from-template", status_code=201)
    async def create_manifold_from_template_route(req: CreateManifoldFromTemplateRequest) -> ManifoldInfo:
        """Author a discover manifold from an existing template artifact.

        ``force`` replaces an existing folder, so it runs like ``merge``: under
        the session lock and refused while the engine is busy, which keeps a
        generation from starting between the overwrite and the eviction of
        every in-memory consumer of the old folder.
        """
        def create() -> ManifoldInfo:
            folder = create_manifold_from_template(
                req.namespace, req.name, req.description,
                template_ref=req.template_ref,
                fit_mode=req.fit_mode,
                hyperparams=req.hyperparams or None,
                force=req.force,
            )
            if req.force:
                session.evict_artifact(req.namespace, req.name)
            return _manifold_json(ManifoldFolder.load(folder), session, full=True)

        if not req.force:
            return await run_in_thread(create)
        async with session_lock(app):
            refuse_if_busy(session)
            return await run_in_thread(create)

    @app.get("/drowse/v1/manifolds/search")
    def search_remote_manifolds(q: str = "", limit: int = 20) -> ManifoldSearchResponse:
        """Search the HF hub for ``drowse-manifold`` repos matching ``q``.

        A missing ``huggingface_hub`` is 503; an HF transport error is 502.
        """
        rows = search_manifolds(q or None)
        if limit > 0:
            rows = rows[:limit]
        return {"query": q, "results": cast(list[RemoteManifoldInfo], rows)}

    @app.post("/drowse/v1/manifolds/merge", status_code=201)
    async def merge_manifold(req: MergeManifoldRequest) -> ManifoldInfo:
        """Pool N discover manifolds' node corpora into one unfitted discover folder.

        Discover sources only — authored geometry is not mergeable without a
        shared coordinate system.  Held under the session lock, and refused
        while an engine operation holds the gen-lock, so a fit cannot race the
        corpus read.
        """
        if len(req.sources) < 2:
            raise HTTPException(400, f"manifold merge: need >= 2 sources, got {len(req.sources)}")
        async with session_lock(app):
            refuse_if_busy(session)
            folder = await run_in_thread(
                merge_discover_manifolds,
                req.namespace, req.name, req.description,
                sources=[(s.namespace, s.name) for s in req.sources],
                fit_mode=req.fit_mode, hyperparams=req.hyperparams, force=req.force,
            )
            session.evict_artifact(req.namespace, req.name)
        return detail(folder.parent.name, folder.name)

    @app.post("/drowse/v1/manifolds/install", status_code=201, response_model=ManifoldInfo)
    async def install_remote_manifold(req: InstallManifoldRequest, request: Request):
        """Install a manifold from an HF coord (``owner/name[@rev]``) or a local folder.

        ``as`` re-namespaces the destination; ``force`` overwrites.  An HF
        pull carries a per-model payload, so this is a progress operation
        (SSE or JSON) whose result is the installed manifold's detail.
        """
        def install(on_progress: ProgressCallback) -> ManifoldInfo:
            folder = install_manifold(req.target, req.as_, force=req.force, on_progress=on_progress)
            if req.force:
                # A forced install replaces the folder under every in-memory
                # consumer of the previous one.
                session.evict_artifact(folder.parent.name, folder.name)
            return detail(folder.parent.name, folder.name)

        async def job(on_progress: ProgressCallback) -> ManifoldInfo:
            # ``run_operation`` holds the session lock for the whole job, so
            # no generation starts between this check and the eviction.
            refuse_if_busy(session)
            return await run_in_thread(install, on_progress)

        return await run_operation(request, job, failure="Manifold install failed")

    @app.post("/drowse/v1/manifolds/generate", status_code=201, response_model=ManifoldInfo)
    async def generate_manifold(req: GenerateManifoldRequest, request: Request):
        """LLM-author a discover manifold: one in-character corpus per concept.

        The default resumes — missing nodes fill and new concepts append;
        ``force`` starts clean.  ``role_per_node`` makes each concept slug its
        node's assistant-role substitution (a persona manifold).  Pair with
        ``POST .../fit``.
        """
        if len(req.concepts) < 2:
            raise HTTPException(
                400, "manifold generate: need >= 2 concepts (a discover manifold is meaningless with one node)",
            )
        if req.kind == "custom" and not req.custom_system:
            raise HTTPException(
                400, "kind='custom' requires custom_system (a system template with a {c} placeholder)",
            )
        folder = manifold_dir(req.namespace, req.name)
        node_roles = {c: c for c in req.concepts} if req.role_per_node else None

        def generate(on_progress: ProgressCallback) -> ManifoldInfo:
            plan = plan_discover_generation(
                folder, req.name, req.description,
                fit_mode=req.fit_mode, labels=list(req.concepts),
                hyperparams=req.hyperparams, node_roles=node_roles,
                node_kinds={c: req.kind for c in req.concepts},
                force=req.force,
            )
            # Evicts the folder's in-memory consumers itself, and refuses
            # (409) while another model op or a generation is in flight.
            session.generate_manifold(
                plan, custom_system=req.custom_system,
                samples_per_prompt=req.samples_per_prompt,
                seed=req.seed,
                on_progress=on_progress,
            )
            body = detail(req.namespace, req.name)
            body["done"] = True
            return body

        async def job(on_progress: ProgressCallback) -> ManifoldInfo:
            return await run_in_thread(generate, on_progress)

        return await run_operation(request, job, failure="Manifold generate failed")

    @app.delete("/drowse/v1/manifolds/{namespace}/{name}")
    async def delete_manifold(namespace: str, name: str) -> ManifoldDeleteResponse:
        """Remove a manifold folder (``remove_manifold_folder``, shared with ``pack rm``).

        Held under the session lock and refused while a fit thread holds the
        gen-lock — deleting ``nodes/`` mid-fit would corrupt the read.
        """
        if not (manifold_dir(namespace, name) / "manifold.json").exists():
            raise HTTPException(404, f"manifold {namespace}/{name} not found")
        async with session_lock(app):
            refuse_if_busy(session)
            session.evict_artifact(namespace, name)
            return cast(
                ManifoldDeleteResponse,
                await run_in_thread(remove_manifold_folder, namespace, name),
            )

    @app.post("/drowse/v1/manifolds/{namespace}/{name}/fit", response_model=ManifoldInfo)
    async def fit_manifold(namespace: str, name: str, req: FitManifoldRequest, request: Request):
        """Fit the manifold for the loaded model (a progress operation, SSE or JSON).

        Discover folders accept ``fit_mode`` / ``hyperparams`` overrides that
        the pipeline merges into ``manifold.json`` inside the transaction that
        derives the cache key.
        """
        folder = manifold_dir(namespace, name)
        if not (folder / "manifold.json").exists():
            raise HTTPException(404, f"manifold {namespace}/{name} not found")

        def fit(on_progress: ProgressCallback) -> ManifoldInfo:
            manifold = session.fit(
                folder, sae=req.sae, layers=req.layers, fit_mode=req.fit_mode,
                hyperparams=req.hyperparams, force=req.force, on_progress=on_progress,
            )
            # ``session.fit`` already moved every in-memory consumer to the
            # new fit (loaded manifolds, folded profiles, attached probes).
            body = detail(namespace, name)
            body["done"] = True
            body["layers_fitted"] = len(manifold.layers)
            body["feature_space"] = manifold.feature_space
            return body

        async def job(on_progress: ProgressCallback) -> ManifoldInfo:
            return await run_in_thread(fit, on_progress)

        return await run_operation(request, job, failure="Manifold fit failed")
