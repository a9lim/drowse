"""ManifoldExtractionPipeline — fit a steering manifold from an authored corpus.

The one extraction pipeline: a steering vector is the 2-node ``pca`` case of a
manifold, so concept extraction (``session.extract``) and manifold fitting
share it.  A fit runs four stages over small values:

1. **inputs** (:class:`FitInputs`) — the authoring snapshot taken under the
   folder lock (:mod:`drowse.store.fit_publish`), the token-exact
   :class:`CaptureIdentity`, the loaded-model fingerprints, the resolved SAE
   backend.  A fitted tensor whose sidecar matches all of it is served as is.
2. **capture** — per-node centroids (and, for curved fits, response rows) from
   the content-addressed :class:`~drowse.fit.capture_cache.CaptureCache`; only
   missing layers run one fit-wide model pass.
3. **geometry** — pure tensor work: the monopolar fold, or whitened node
   geometry → resolved topology → per-layer affine/curved subspaces →
   σ-field → origin.
4. **publish** — compare-and-swap against the authoring revision, then
   ``ManifoldExtracted``.

Dependencies are passed structurally via the :class:`ModelHandle` protocol
plus an :class:`EventBus`; ``DrowseSession`` satisfies the protocol, so
construction reads ``ManifoldExtractionPipeline(self, self.events)``.  The
session gates re-entry against generation before forwarding.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

import torch

from drowse.events import EventBus, ManifoldExtracted
from drowse.fit import capture as _capture
from drowse.fit.capture_cache import CaptureCache
from drowse.geometry.manifold import MANIFOLD_FIT_POLICY_VERSION
from drowse.geometry.topology import PcaDiagnostics, SpectralDiagnostics
from drowse.model.arch import workspace_layer_indices
from drowse.model.fingerprint import loaded_model_fingerprint
from drowse.sae.backend import SaeBackend
from drowse.base.paths import tensor_filename
from drowse.store.fit_publish import (
    AuthoringSnapshot,
    assert_authoring_revision,
    fit_target_lock,
    load_published_fit,
    publish_fit_if_current,
    take_authoring_snapshot,
)

if TYPE_CHECKING:
    from drowse.geometry.manifold import Manifold
    from drowse.geometry.subspace import LayerSubspace
    from drowse.store.manifold_folder import ManifoldFolder
    from drowse.store.manifold_schema import ManifoldSidecar
    from drowse.store.manifold_tensors import ActivationRowStore


# Schema revision of the manifold fit's capture identity — the payload
# ``prepare_manifold_capture_identity`` hashes into ``capture_sha256``.  Bumping
# it invalidates every fitted tensor *and* every cached capture, so it is the
# knob for "the residuals a corpus produces are no longer the same".  It also
# rides the fitted sidecar as ``capture_version`` so a weight-free preflight can
# prove the stored identity was built under this exact schema.
MANIFOLD_CAPTURE_VERSION = 3

# Fit modes whose geometry may come out curved and so needs response rows for
# the σ-field (``auto`` only if topology selection resolves curved).
_ROW_CONSUMING_MODES = frozenset({"authored", "spectral", "auto"})


class ModelHandle(Protocol):
    """Read-only surface the pipeline needs from the live HF session.

    Held as a *handle*, not a copy — the pipeline must see the same model
    object the session uses; otherwise device-side state diverges.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def model(self) -> torch.nn.Module: ...

    @property
    def tokenizer(self) -> Any: ...  # PreTrainedTokenizerBase

    @property
    def device(self) -> torch.device: ...

    @property
    def layers(self) -> torch.nn.ModuleList: ...

    @property
    def whitener(self) -> Any: ...

    @property
    def layer_means(self) -> dict[int, torch.Tensor]: ...


def _default_prompts() -> list[str]:
    """The shared baseline prompt set (looked up per call; tests substitute it)."""
    return list(_capture._load_baseline_prompts())


def resolve_fit_layer_indices(
    layer_indices: "Sequence[int] | str | None", n_layers: int,
) -> list[int]:
    """Normalize a ``layer_indices`` request against a model's layer count.

    ``None``/``"all"`` is every layer, ``"workspace"`` the 40–90% band, any
    other string a comma-separated index list.  Shared with the weight-free fit
    preflight, which resolves the same request against the *config's* layer
    count: the two must agree exactly or a proof could cover a different roster
    than the fit would evaluate.
    """
    if layer_indices is None or layer_indices == "all":
        requested = list(range(n_layers))
    elif layer_indices == "workspace":
        requested = workspace_layer_indices(range(n_layers), n_layers)
    elif isinstance(layer_indices, str):
        try:
            requested = sorted({
                int(part.strip()) for part in layer_indices.split(",")
                if part.strip()
            })
        except ValueError as exc:
            raise ValueError(
                "layer_indices must be 'all', 'workspace', or a "
                "comma-separated integer list"
            ) from exc
    else:
        requested = sorted({int(idx) for idx in layer_indices})
    if not requested:
        raise ValueError("layer_indices must name at least one layer")
    if any(idx < 0 or idx >= n_layers for idx in requested):
        raise ValueError(
            f"layer_indices must lie in [0, {n_layers}); got {requested}"
        )
    return requested


def baseline_prompts_sha256(
    baseline_prompts: "Sequence[str | list[dict[str, str]]]",
) -> str:
    """Digest the exact shared elicitation prompt set a fit paired rows against.

    Plain strings (the shared baseline set) and multi-turn message lists (a
    templated manifold's resolved contexts) hash through one canonical JSON
    encoding, so the channel means the same thing for both corpus sources.
    """
    return hashlib.sha256(json.dumps(
        list(baseline_prompts), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def capture_render_sha256(
    node_groups: "Sequence[tuple[str, list[str]]]",
    prepared_rows: "Sequence[tuple[torch.Tensor, int]]",
) -> str:
    """Digest the tokenizer-render half of the capture identity.

    Exactly the node partition plus every rendered row's token ids and pool
    index — the part of ``capture_sha256`` that a tokenizer, the baseline
    prompts, and the node corpora reproduce with **no weights loaded**.  The
    complementary model-identity half rides ``model_fingerprint`` +
    ``model_source_fingerprint``, so together the two prove ``capture_sha256``
    equality without instantiating the transformer.
    """
    digest = hashlib.sha256(json.dumps(
        {"node_sizes": [len(rows) for _label, rows in node_groups]},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))
    for ids, content_end in prepared_rows:
        digest.update(repr((ids[0].tolist(), int(content_end))).encode("utf-8"))
    return digest.hexdigest()


def prepare_capture_rows(
    tokenizer: Any,
    *,
    node_groups: "Sequence[tuple[str, list[str]]]",
    baseline_prompts: "Sequence[str | list[dict[str, str]]]",
    node_roles: "Sequence[str | None]",
    model_type: str | None,
) -> tuple[list[tuple[torch.Tensor, int]], str]:
    """Render + tokenize every fit row and digest that render.

    The single owner of the row flattening (``response[i]`` answers
    ``baseline_prompt[i % k]``, per-node role carried through) that both the
    live fit and the weight-free preflight must agree on byte-for-byte.  Needs
    a tokenizer, never a model.
    """
    from drowse.fit.capture import _prepare_capture_batch

    flat_prompts: list[str | list[dict[str, str]]] = []
    flat_responses: list[str] = []
    flat_roles: list[str | None] = []
    k_prompts = len(baseline_prompts)
    for (_label, responses), role in zip(node_groups, node_roles, strict=True):
        for row_idx, response in enumerate(responses):
            flat_prompts.append(baseline_prompts[row_idx % k_prompts])
            flat_responses.append(response)
            flat_roles.append(role)
    prepared_rows = _prepare_capture_batch(
        tokenizer, flat_prompts, flat_responses,
        roles=flat_roles, model_type=model_type,
    )
    return prepared_rows, capture_render_sha256(node_groups, prepared_rows)



def resolve_elicitation_prompts(
    mf: "ManifoldFolder",
) -> tuple[list[str | list[dict[str, str]]], str | None]:
    """The prompts a folder's node responses answer, plus the template digest.

    A templated manifold's corpus answers its template's multi-turn contexts;
    every other folder answers the shared baseline prompt set.
    """
    if mf.template_ref is None:
        return list(_default_prompts()), None
    from drowse.store.templates import resolve_template

    tmpl = resolve_template(mf.template_ref)
    return [context.messages() for context in tmpl.contexts], tmpl.sha256()


@dataclass(frozen=True)
class OfflineFitIdentity:
    """The fit-identity digests derivable with no model weights loaded.

    Every field is something :class:`ManifoldExtractionPipeline` would compute
    for the same folder, reproduced from the folder, the shared baseline
    prompts (or a templated manifold's resolved contexts), and a tokenizer.
    What is *not* here — the loaded-model fingerprint and therefore
    ``capture_sha256`` itself — is exactly what the preflight proves through
    the neutral cache's ``(source, loaded)`` fingerprint binding instead.
    """

    nodes_sha256: str
    baseline_prompts_sha256: str
    capture_render_sha256: str
    node_labels: tuple[str, ...]


def offline_fit_identity(
    mf: "ManifoldFolder", *, tokenizer: Any, model_type: str | None,
) -> OfflineFitIdentity | None:
    """Recompute a folder's weight-free fit identity, or ``None`` if it cannot be.

    Returns ``None`` — never a partial answer — when any input the fit would
    consume cannot be reproduced offline: an unresolvable ``template_ref``, a
    templated corpus the fit would re-derive before hashing, per-node roles with
    no ``model_type`` to resolve their headers, or a tokenizer that cannot
    render.  A caller must treat ``None`` as "load the model", because a partial
    identity is precisely what would serve a stale fit after a corpus or
    baseline-prompt edit.
    """
    node_roles = mf.roles
    if any(role is not None for role in node_roles) and model_type is None:
        return None
    try:
        node_groups = mf.node_groups()
        baseline_prompts, template_sha = resolve_elicitation_prompts(mf)
        if mf.template_ref is not None:
            from drowse.store.templates import resolve_template

            # The fit re-derives (and rewrites) a drifted templated corpus
            # under the manifest lock before it hashes anything, so a mismatch
            # here means the on-disk corpus is not what the fit would hash.
            if dict(node_groups) != resolve_template(mf.template_ref).node_corpora():
                return None
        if not baseline_prompts:
            return None
        nodes_sha = mf.nodes_sha256(resolved_template_sha256=template_sha)
        _rows, render_sha = prepare_capture_rows(
            tokenizer,
            node_groups=node_groups,
            baseline_prompts=baseline_prompts,
            node_roles=node_roles,
            model_type=model_type,
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError):
        return None
    return OfflineFitIdentity(
        nodes_sha256=nodes_sha,
        baseline_prompts_sha256=baseline_prompts_sha256(baseline_prompts),
        capture_render_sha256=render_sha,
        node_labels=tuple(label for label, _rows in node_groups),
    )


@dataclass(frozen=True)
class CaptureIdentity:
    """Everything that determines a fit's residual activations, rendered once.

    ``capture_sha256`` digests the loaded model's identity, the node partition,
    and every rendered row's token ids and pool index — never fit geometry or
    hyperparameters, so geometry-only refits share one cached capture.  The
    two weight-free halves (``capture_render_sha256``,
    ``baseline_prompts_sha256``) let a preflight prove it off-model.
    """

    node_groups: list[tuple[str, list[str]]]
    baseline_prompts: list[str | list[dict[str, str]]]
    prepared_rows: list[tuple[torch.Tensor, int]]
    node_roles: list[str | None]
    node_kinds: list[str | None]
    model_type: str | None
    capture_sha256: str
    capture_render_sha256: str
    baseline_prompts_sha256: str

    @property
    def node_sizes(self) -> list[int]:
        return [len(rows) for _label, rows in self.node_groups]


def prepare_manifold_capture_identity(
    handle: ModelHandle,
    mf: "ManifoldFolder",
    model_fingerprint: str,
    *,
    node_groups: list[tuple[str, list[str]]] | None = None,
    baseline_prompts: list[str | list[dict[str, str]]] | None = None,
) -> CaptureIdentity:
    """Render/tokenize all fit rows and return their exact capture identity.

    ``node_groups`` / ``baseline_prompts`` pin an authoring snapshot; omitted,
    they are read from the folder (the steer-time staleness check).
    """
    model = handle.model
    base_model = getattr(model, "_orig_mod", model)
    config = getattr(base_model, "config", getattr(model, "config", None))
    groups = (
        [(label, list(rows)) for label, rows in node_groups]
        if node_groups is not None else mf.node_groups()
    )
    node_roles = mf.roles
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if any(role is not None for role in node_roles) and model_type is None:
        raise ValueError(
            f"manifold {mf.name!r} carries per-node roles but model config "
            "has no 'model_type' — cannot resolve role headers"
        )
    prompts = (
        list(baseline_prompts) if baseline_prompts is not None
        else resolve_elicitation_prompts(mf)[0]
    )
    prepared_rows, render_sha = prepare_capture_rows(
        handle.tokenizer,
        node_groups=groups,
        baseline_prompts=prompts,
        node_roles=node_roles,
        model_type=model_type,
    )
    capture_hash = hashlib.sha256(json.dumps(
        {
            "model_id": handle.model_id,
            "model_class": (
                f"{type(base_model).__module__}.{type(base_model).__qualname__}"
            ),
            "commit": getattr(config, "_commit_hash", None),
            "name_or_path": getattr(config, "_name_or_path", None),
            "model_type": getattr(config, "model_type", None),
            "capture_version": MANIFOLD_CAPTURE_VERSION,
            "fingerprint": model_fingerprint,
        },
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))
    capture_hash.update(json.dumps(
        {"node_sizes": [len(rows) for _label, rows in groups]},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))
    for ids, content_end in prepared_rows:
        capture_hash.update(repr((ids[0].tolist(), int(content_end))).encode("utf-8"))
    return CaptureIdentity(
        node_groups=groups,
        baseline_prompts=prompts,
        prepared_rows=prepared_rows,
        node_roles=node_roles,
        node_kinds=mf.kinds,
        model_type=model_type,
        capture_sha256=capture_hash.hexdigest(),
        capture_render_sha256=render_sha,
        baseline_prompts_sha256=baseline_prompts_sha256(prompts),
    )


def _diagnostics_to_dict(
    diag: PcaDiagnostics | SpectralDiagnostics,
) -> dict[str, Any]:
    """Exhaustively serialize the current diagnostics tagged union."""
    if isinstance(diag, PcaDiagnostics):
        return {
            "per_component_variance": [
                float(x) for x in diag.per_component_variance.tolist()
            ],
            "cumulative_variance": [
                float(x) for x in diag.cumulative_variance.tolist()
            ],
            "picked_k": int(diag.picked_k),
            "threshold": float(diag.threshold),
        }
    out: dict[str, Any] = {
        "eigenvalues": [float(x) for x in diag.eigenvalues.tolist()],
        "picked_k": int(diag.picked_k),
        "gap_index": int(diag.gap_index),
        "gap_magnitude": float(diag.gap_magnitude),
        "bandwidth": float(diag.bandwidth),
        "k_nn": int(diag.k_nn),
        "component_count": int(diag.component_count),
        "heuristic_k": int(diag.heuristic_k),
        "pinned": bool(diag.pinned),
    }
    if diag.min_dim is not None:
        out["min_dim"] = int(diag.min_dim)
    return out



@dataclass(frozen=True)
class _WhitenedNodeGeometry:
    """Per-layer whitened between-node geometry over the pooled centroids.

    ``rows`` is ``Σ_L⁻¹ X̃_L`` and ``grams`` the symmetrized
    ``G_L = X̃_L Σ_L⁻¹ X̃_Lᵀ`` of the node-mean-centered centroids, computed
    once here because four later stages want them: discover coordinate
    derivation and topology scoring (whose consensus Gram is the mean of
    ``grams``), the Fisher basis in the per-layer fit, and the neutral-layout
    anchoring.

    ``node_spread`` is ``tr(G_L) = Σ_k ‖x̃_k‖²_M`` — the total whitened node
    spread at layer L in background-σ² units, comparable across layers.  It is
    a *diagnostic* layer profile (where does this concept live?), distinct from
    the apply-time ``mahalanobis_share``, which restricts the same idea to the
    steerable subspace: a layer with large ``tr(G_L)`` but a small share is one
    whose fitted subspace is dropping concept signal.
    """

    rows: dict[int, torch.Tensor]
    grams: dict[int, torch.Tensor]
    centroid_means: dict[int, torch.Tensor]
    node_spread: dict[int, float]


def _whitened_node_geometry(
    stacks: Mapping[int, torch.Tensor],
    fit_layers: Sequence[int],
    *,
    whitener: Any,
) -> _WhitenedNodeGeometry:
    """Whiten each layer's ``(K, D)`` centroid stack; see the dataclass."""
    rows: dict[int, torch.Tensor] = {}
    grams: dict[int, torch.Tensor] = {}
    centroid_means: dict[int, torch.Tensor] = {}

    def _one(idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep the full K×D temporaries in a short-lived frame; otherwise the
        # final loop iteration remains bound after the roster clears.
        raw = stacks[idx].to(torch.float32)
        centroid_mean = raw.mean(dim=0, keepdim=True)
        xc = raw - centroid_mean
        sinv_xc = whitener.apply_inv(idx, xc).to(torch.float32)
        gram = xc @ sinv_xc.transpose(0, 1)
        return sinv_xc, 0.5 * (gram + gram.transpose(0, 1)), centroid_mean

    for idx in fit_layers:
        rows[idx], grams[idx], centroid_means[idx] = _one(idx)
    return _WhitenedNodeGeometry(
        rows=rows,
        grams=grams,
        centroid_means=centroid_means,
        node_spread={
            idx: float(grams[idx].diagonal().sum()) for idx in fit_layers
        },
    )


@dataclass(frozen=True)
class _ResolvedGeometry:
    """The geometry a fit runs on, resolved before any layer is fit.

    ``fit_mode`` on disk is a *request*; this is the answer.  ``authored``
    supplies its own domain and coords, ``pca``/``spectral`` derive coords from
    the consensus Gram, and ``auto`` defers the whole
    flat-vs-curved-vs-periodic choice to ``select_topology``.  All three
    converge on the same things the per-layer fit needs: a domain, the shared
    node layout, its embedding, the resolved mode to branch on, the sidecar
    ``method``, and the discover provenance block — plus the curved-only knobs
    and, for ``auto``, the winner's precomputed artifacts.
    """

    domain: Any
    node_coords: torch.Tensor
    node_params: torch.Tensor
    #: What the per-layer fit branches on: ``mf.fit_mode`` verbatim for
    #: authored/pca/spectral, and what ``select_topology`` picked for ``auto``.
    effective_fit_mode: str
    method: str
    discover_metadata: dict[str, Any]
    #: Penalized-RBF λ selection for the curved surface.  ``None`` ⇒ exact
    #: interpolation — the authored contract (node = exact steering target) and
    #: the flat ``pca`` path, which has no RBF at all.
    curved_smoothing: float | str | None
    #: Curved-only per-layer PCA width cap.  A flat fit's subspace *is* its
    #: k-dim layout span, so the affine branch hard-sets ``n_components`` and
    #: ignores any override.
    max_subspace_dim_override: int | None
    #: ``auto``-only: the winner's already-factorized RBF plan and Fisher
    #: bases, carried out of topology scoring so the final layer fit doesn't
    #: repeat the layout QR/eigh/LU.
    rbf_plan: Any = None
    fisher_bases: dict[int, torch.Tensor] = field(default_factory=dict)


def _resolve_fit_geometry(
    mf: Any,
    *,
    K: int,
    fit_layers: Sequence[int],
    stacks: Mapping[int, torch.Tensor],
    layer_grams: Mapping[int, torch.Tensor],
    whitened_rows: dict[int, torch.Tensor],
    whitener: Any,
    sae_backend: "SaeBackend | None",
    progress: Callable[[str], None],
) -> _ResolvedGeometry:
    """Resolve the fit's geometry — the one step that differs by ``fit_mode``.

    Pure computation over the pooled centroid stacks: no locks, no IO, no
    publication.  Raises ``ValueError`` when the resolved intrinsic dim needs
    more nodes than the corpus carries (``k + 1`` to span a flat affine
    subspace, ``min_nodes(k) = 2k+1`` for a curved RBF surface).
    """
    from drowse.geometry.domains import CustomDomain, domain_from_spec
    from drowse.geometry.subspace import DEFAULT_N_COMPONENTS
    from drowse.geometry.topology import discover_coords, select_topology
    from drowse.store.manifold_schema import min_nodes, sanitize_hyperparams

    discover_metadata: dict[str, Any] = {}
    # ``max_subspace_dim`` caps the per-layer PCA subspace in the
    # **curved** fit (``fit_curved_subspace``; default
    # :data:`DEFAULT_N_COMPONENTS` = 64).  Smaller values constrain the
    # dim count that ``subspace_inject`` displaces at steer time — finer-
    # grained steering control at the cost of representing less per-layer
    # activation variance.  It is a *curved-only* knob: a flat (``pca``)
    # fit's per-layer subspace is exactly its k-dim layout span (the
    # affine span *is* the layout), so ``max_subspace_dim`` is not a
    # ``pca`` hyperparam — the affine branch hard-sets ``n_components`` to
    # the derived intrinsic dim and ignores any override.  Authored
    # manifolds don't currently route hyperparams so they inherit 64.
    max_subspace_dim_override: int | None = None
    # ``smoothing`` (curved discover only): penalized-RBF λ selection for
    # the per-layer surface.  ``None`` ⇒ exact interpolation — the authored
    # contract (node = exact steering target) and the flat ``pca`` path
    # (no RBF).  A curved ``spectral`` discover fit defaults to GCV
    # (``"auto"``), trading exactness for a surface that doesn't chase
    # noise in the per-node centroids.
    curved_smoothing: float | str | None = None

    # ``effective_fit_mode`` is the *resolved* geometry the per-layer fit
    # branches on: it equals ``mf.fit_mode`` for authored / pca / spectral,
    # and the topology ``select_topology`` picks for ``fit_mode="auto"``.
    effective_fit_mode = mf.fit_mode
    auto_rbf_plan = None
    auto_fisher_bases: dict[int, torch.Tensor] = {}
    if mf.fit_mode == "authored":
        domain = domain_from_spec(mf.domain)
        node_coords = torch.tensor(mf.node_coords, dtype=torch.float32)
        node_params = domain.embed(node_coords)
        method = "manifold_sae" if sae_backend is not None else "manifold_pca"
    elif mf.fit_mode == "auto":
        # Auto: pick the discover geometry per-model — flat (pca) vs curved
        # (spectral) by GCV, plus periodic (BoxDomain) axes via persistent
        # homology.  ``select_topology`` returns the resolved fit_mode +
        # coords + domain; the per-layer fit below runs unchanged on the
        # resolved mode.  Sphere is authored-only (not an auto candidate).
        st_hyper = sanitize_hyperparams("auto", dict(mf.hyperparams))
        consensus_gram = torch.stack(
            [layer_grams[idx] for idx in fit_layers]
        ).mean(dim=0)
        progress(
            f"Selecting topology across {len(fit_layers)} layers "
            f"({K} centroids)..."
        )
        choice = select_topology(
            {idx: stacks[idx] for idx in fit_layers},
            {idx: layer_grams[idx] for idx in fit_layers},
            consensus_gram,
            whitener=whitener,
            whitened_rows=whitened_rows,
            max_dim=int(st_hyper.get("max_dim", 8)),
            smoothing=st_hyper.get("smoothing", "auto"),
            persistence_frac=float(st_hyper.get("persistence_frac", 0.5)),
            score_dim=int(
                st_hyper.get("max_subspace_dim", DEFAULT_N_COMPONENTS)
            ),
        )
        effective_fit_mode = choice.fit_mode
        auto_rbf_plan = choice.rbf_plan
        auto_fisher_bases = choice.fisher_bases
        domain = choice.domain
        node_coords = choice.coords
        node_params = domain.embed(node_coords)
        k = int(node_coords.shape[1])
        floor = (k + 1) if effective_fit_mode == "pca" else min_nodes(k)
        if floor > K:
            raise ValueError(
                f"auto manifold {mf.name!r}: resolved topology "
                f"{choice.winner_name!r} (k={k}) needs >= {floor} nodes, "
                f"got K={K}"
            )
        if effective_fit_mode == "spectral":
            curved_smoothing = st_hyper.get("smoothing", "auto")
        method = (
            "manifold_discover_sae" if sae_backend is not None
            else "manifold_discover_auto"
        )
        discover_metadata = {
            "fit_mode": "auto",
            "resolved_fit_mode": effective_fit_mode,
            "hyperparams": dict(st_hyper),
            "topology_winner": choice.winner_name,
            "topology_candidates": [
                {
                    "name": c.name,
                    "fit_mode": c.fit_mode,
                    "intrinsic_dim": c.intrinsic_dim,
                    "score": (c.score if math.isfinite(c.score) else None),
                    "viable": c.viable,
                    "reason": c.reason,
                }
                for c in choice.candidates
            ],
        }
        # Emit the winner's coordinate diagnostics (pca per-component
        # variance / spectral eigenvalues) so the inspector renders the
        # same bars a pinned pca/spectral fit does — the auto resolution
        # otherwise leaves the panel blank.
        if choice.diagnostics is not None:
            discover_metadata["diagnostics"] = _diagnostics_to_dict(
                choice.diagnostics
            )
    else:
        # Discover: derive coords from the per-node centroids, layer-
        # agnostically — there is no reference layer.  The same coords feed
        # the per-layer RBF as the manifold's intrinsic coordinates —
        # wrapped in a ``CustomDomain(k)`` with identity embed so the
        # existing fit machinery handles them unchanged.
        # Sanitize against the per-mode whitelist (single source of truth
        # in ``store.manifold_schema``) so a stale on-disk ``manifold.json`` carrying
        # a since-removed key (e.g. the old ``anchor_origin``) can't reach
        # ``discover_coords`` as an unexpected kwarg.  Author/CLI paths
        # already sanitize; this guards legacy + hand-edited folders.
        hyperparams = sanitize_hyperparams(mf.fit_mode, dict(mf.hyperparams))
        # ``max_subspace_dim`` is consumed by the curved per-layer fit,
        # not by ``discover_coords`` — pop it before the discover call so
        # the dispatcher doesn't get an unexpected kwarg.  (Sanitized out
        # of ``pca`` folders above; the affine branch ignores it regardless
        # — see the affine fit below.)
        if "max_subspace_dim" in hyperparams:
            max_subspace_dim_override = int(hyperparams.pop("max_subspace_dim"))
        # ``smoothing`` is consumed by the penalized curved fit
        # (``fit_curved_subspace`` → ``RbfFitPlan.fit``), not by
        # ``discover_coords`` — pop it before the dispatch so it isn't an
        # unexpected kwarg.  Only the curved (``spectral``) path has an RBF
        # surface to smooth; the flat (``pca``) path's affine span has no
        # interpolant, so smoothing never applies there.
        if mf.fit_mode == "spectral":
            curved_smoothing = hyperparams.pop("smoothing", "auto")
        else:
            hyperparams.pop("smoothing", None)
        # Consensus Gram: the mean over every fit layer of that layer's
        # whitened, node-mean-centered (K, K) Gram ``X̃_L Σ_L⁻¹ X̃_Lᵀ``
        # (the ``layer_grams`` already computed for the spread profile).
        # Whitening puts each layer in common (background-σ) units, so the
        # raw average is **signal-weighted** — a layer where the nodes
        # aren't separated contributes a near-zero Gram and drops out, while
        # a layer where the concept is strongly represented dominates.  PCA
        # eigendecomposes this Gram; spectral reads its pairwise distances.
        # The (K, K) Gram is the layer-invariant object, so this is exactly
        # the single-reference-layer derivation generalized to the whole
        # stack — ``%`` positions and node labels live in one coordinate
        # system distilled from all layers, wherever the concept's signal
        # happens to concentrate.
        progress(
            f"Deriving {mf.fit_mode} coords across {len(fit_layers)} "
            f"layers ({K} centroids)..."
        )
        consensus_gram = torch.stack(
            [layer_grams[idx] for idx in fit_layers]
        ).mean(dim=0)
        derived_coords, diagnostics = discover_coords(
            consensus_gram, method=mf.fit_mode, **hyperparams,
        )
        k = derived_coords.shape[1]
        # Node-count floor.  The curved (spectral) path fits an RBF
        # surface and needs the poisedness floor ``min_nodes(k) = 2k+1``.
        # The flat (``pca``) path fits an *affine* subspace with no RBF —
        # it only needs ``k+1`` affinely-independent centroids to span a
        # k-dim subspace, so a rank-1 (k=1) fit is valid at K=2: that is a
        # difference-of-means steering vector (ARCHITECTURE §1/§5, "a
        # vector = a 2-node fit_mode=pca folder").
        if mf.fit_mode == "pca":
            floor = k + 1
            floor_reason = "to span the affine subspace"
        else:
            floor = min_nodes(k)
            floor_reason = "for the RBF fit"
        if floor > K:
            raise ValueError(
                f"discover manifold {mf.name!r}: picked k={k} needs "
                f">= {floor} nodes {floor_reason}, got K={K}"
            )
        # The derived coords come out PCA-mean-centered (origin = the node
        # centroid).  The per-layer fit below neutral-anchors each layer's
        # *real* reduced frame (coord 0 = neutral), and the shared display
        # layout is re-anchored on neutral in step 4a after the fit, so the
        # display/`%`-authoring origin matches the steer origin.
        domain = CustomDomain(k)
        node_coords = derived_coords
        node_params = derived_coords  # identity embedding
        method = (
            "manifold_discover_sae" if sae_backend is not None
            else f"manifold_discover_{mf.fit_mode}"
        )
        # Record what we ran with — fit_mode + hyperparams (with
        # any derived defaults filled in, e.g. spectral's resolved
        # ``k_nn``/``bandwidth``) + the diagnostics for the sidecar
        # and the inspector surfaces.
        resolved_hyperparams = dict(hyperparams)
        if max_subspace_dim_override is not None:
            resolved_hyperparams["max_subspace_dim"] = max_subspace_dim_override
        if curved_smoothing is not None:
            resolved_hyperparams["smoothing"] = curved_smoothing
        if isinstance(diagnostics, SpectralDiagnostics):
            resolved_hyperparams["k_nn"] = int(diagnostics.k_nn)
            resolved_hyperparams["bandwidth"] = float(
                diagnostics.bandwidth,
            )
        discover_metadata = {
            "fit_mode": mf.fit_mode,
            "hyperparams": resolved_hyperparams,
            "diagnostics": _diagnostics_to_dict(diagnostics),
        }

    return _ResolvedGeometry(
        domain=domain,
        node_coords=node_coords,
        node_params=node_params,
        effective_fit_mode=effective_fit_mode,
        method=method,
        discover_metadata=discover_metadata,
        curved_smoothing=curved_smoothing,
        max_subspace_dim_override=max_subspace_dim_override,
        rbf_plan=auto_rbf_plan,
        fisher_bases=auto_fisher_bases,
    )



@dataclass(frozen=True)
class FitInputs:
    """Everything a fit of one target is computed from and keyed on."""

    snapshot: AuthoringSnapshot
    identity: CaptureIdentity
    tensor_path: Path
    model_fingerprint: str
    #: The exact checkpoint source the loader proved for these weights; ``None``
    #: for an in-memory or opaque model, which can then never be preflight-proven.
    model_source_fingerprint: str | None
    requested_layers: list[int]
    #: ``layer_indices`` was given explicitly (an SAE fit otherwise covers
    #: exactly the release's layers).
    layer_scope_explicit: bool
    sae_backend: SaeBackend | None
    #: The SAE revision the cache key compares (the request's, or the passed
    #: backend's own).
    sae_revision: str | None

    @property
    def manifold(self) -> "ManifoldFolder":
        return self.snapshot.manifold

    @property
    def feature_space(self) -> str:
        if self.sae_backend is None:
            return "raw"
        return f"sae-{self.sae_backend.release}"

    def accepts(self, sidecar: "ManifoldSidecar") -> bool:
        """Whether a published fit was computed from exactly these inputs.

        ``nodes_sha256`` folds in the corpus plus the domain/coords (authored)
        or fit_mode/hyperparams (discover); ``capture_sha256`` the rendered
        rows and loaded weights; the SAE revision does not ride the filename,
        so it is checked here.
        """
        backend = self.sae_backend
        release = backend.release if backend is not None else None
        if (
            sidecar.nodes_sha256 != self.snapshot.nodes_sha256
            or sidecar.model_fingerprint != self.model_fingerprint
            or sidecar.capture_sha256 != self.identity.capture_sha256
            or sidecar.fit_policy_version != MANIFOLD_FIT_POLICY_VERSION
            or sidecar.sae_release != release
        ):
            return False
        if backend is None:
            return (
                sidecar.sae_revision is None
                and sidecar.evaluated_layers == self.requested_layers
            )
        if sidecar.sae_fingerprint != backend.fingerprint:
            return False
        if self.sae_revision is not None and sidecar.sae_revision != self.sae_revision:
            return False
        if self.layer_scope_explicit:
            return sidecar.evaluated_layers == self.requested_layers
        return bool(sidecar.sae_full_coverage)

    def base_metadata(
        self, *, method: str, fitted_layers: list[int], subspace_metric: str,
    ) -> dict[str, Any]:
        """The sidecar keys every fit writes.

        ``subspace_metric`` is a basis *label*: the monopolar fold keeps the
        raw δ̂ basis (``concept − ν`` cancels common-mode by differencing, like
        DiM) and reports ``euclidean``; every other fit selects a whitened
        basis.  The share is always Mahalanobis — the whitener gate guarantees
        it.
        """
        ident = self.identity
        metadata: dict[str, Any] = {
            "method": method,
            "nodes_sha256": self.snapshot.nodes_sha256,
            "model_fingerprint": self.model_fingerprint,
            "model_source_fingerprint": self.model_source_fingerprint,
            "capture_sha256": ident.capture_sha256,
            "capture_version": MANIFOLD_CAPTURE_VERSION,
            "capture_render_sha256": ident.capture_render_sha256,
            "baseline_prompts_sha256": ident.baseline_prompts_sha256,
            "fitted_layers": fitted_layers,
            "fit_policy_version": MANIFOLD_FIT_POLICY_VERSION,
            # Provenance only (nothing branches on these at load).
            "share_metric": "mahalanobis",
            "subspace_metric": subspace_metric,
        }
        backend = self.sae_backend
        if backend is not None:
            metadata["sae_release"] = backend.release
            metadata["sae_revision"] = backend.revision
            metadata["sae_fingerprint"] = backend.fingerprint
            metadata["sae_ids_by_layer"] = dict(
                getattr(backend, "sae_ids_by_layer", {})
            )
            metadata["sae_full_coverage"] = not self.layer_scope_explicit
        # Per-node roles/kinds ride the sidecar for `manifold show` and the
        # inspectors, ``node_labels`` order; absent when no node carries one.
        if any(role is not None for role in ident.node_roles):
            metadata["node_roles"] = list(ident.node_roles)
        if any(kind is not None for kind in ident.node_kinds):
            metadata["node_kinds"] = list(ident.node_kinds)
        return metadata


@dataclass
class _Captured:
    """Resident capture for one fit: centroids plus optional response rows."""

    cache: CaptureCache
    stacks: dict[int, torch.Tensor]
    rows: "ActivationRowStore | None"
    #: Row layers already durable in the cache (loaded from it).
    durable_rows: set[int] = field(default_factory=set)

    def take_stacks(self) -> dict[int, torch.Tensor]:
        """Move centroid ownership to the caller, leaving no aliases here."""
        stacks, self.stacks = self.stacks, {}
        return stacks

    def close(self) -> None:
        if self.rows is not None:
            self.rows.close()
            self.rows = None


def _union_rows(
    first: "ActivationRowStore | None", second: "ActivationRowStore | None",
) -> "ActivationRowStore | None":
    if first is None or second is None:
        return first if second is None else second
    from drowse.store.manifold_tensors import ActivationRowStore

    return ActivationRowStore.combine_disjoint([first, second])


def _sae_reconstruct(
    backend: SaeBackend, layer: int, x: torch.Tensor, device: torch.device,
) -> torch.Tensor:
    """Round-trip ``(n, D)`` activations through one SAE layer, fp32 on CPU."""
    with torch.no_grad():
        features = backend.encode_layer(layer, x.to(device))
        return backend.decode_layer(layer, features).detach().to("cpu", torch.float32)


class ManifoldExtractionPipeline:
    """Fit a steering manifold for the handle's model from a manifold folder.

    Feature space: ``sae=None`` fits on residual-stream centroids.
    ``sae="<release>"`` (or a backend) reconstructs each centroid through the
    SAE before the fit and restricts the manifold to the SAE's covered layers.
    Either way the fitted :class:`~drowse.geometry.subspace.LayerSubspace` is
    model-space, so the steering hook never touches the SAE.
    """

    __slots__ = ("_events", "_handle")

    def __init__(self, model_handle: ModelHandle, events: EventBus) -> None:
        self._handle = model_handle
        self._events = events

    def fit(
        self,
        folder: str | Path,
        *,
        sae: str | SaeBackend | None = None,
        sae_revision: str | None = None,
        layer_indices: "Sequence[int] | str | None" = None,
        fit_mode: str | None = None,
        hyperparams: Mapping[str, object] | None = None,
        force: bool = False,
        on_progress: Callable[[str], None] | None = None,
        emit_event: bool = True,
        dls: bool = True,
    ) -> "Manifold":
        """Fit (or load the current fit of) ``folder`` for the handle's model.

        A published tensor whose sidecar matches every fit input short-circuits
        the forward passes; ``force=True`` bypasses it and re-pools the selected
        layers (the discover analogue of ``manifold extract -f``).  The lock is
        target-scoped, so two model/variant targets in one folder compute
        concurrently and readers keep the prior artifact; the folder lock is
        held only for the authoring snapshot and the final compare-and-swap.

        ``emit_event=False`` suppresses ``ManifoldExtracted`` for a caller that
        emits its own enriched event (the session's vector path).  ``dls=False``
        keeps every flat-fit axis instead of pruning to the ones whose node
        projections straddle the neutral baseline; curved fits never run DLS.
        """
        if sae_revision is not None and sae is None:
            raise ValueError("sae_revision requires an SAE release")
        progress = on_progress or (lambda _message: None)
        started = time.perf_counter()
        folder = Path(folder)
        release = sae if isinstance(sae, str) else (
            sae.release if sae is not None else None
        )
        target = folder / tensor_filename(self._handle.model_id, release=release)
        with fit_target_lock(folder, target):
            snapshot = take_authoring_snapshot(
                folder, target, fit_mode=fit_mode, hyperparams=hyperparams,
                default_prompts=_default_prompts,
            )
            return self._fit_snapshot(
                snapshot, target, sae=sae, sae_revision=sae_revision,
                layer_indices=layer_indices, force=force, progress=progress,
                emit_event=emit_event, dls=dls, started=started,
            )

    def _fit_snapshot(
        self,
        snapshot: AuthoringSnapshot,
        target: Path,
        *,
        sae: str | SaeBackend | None,
        sae_revision: str | None,
        layer_indices: "Sequence[int] | str | None",
        force: bool,
        progress: Callable[[str], None],
        emit_event: bool,
        dls: bool,
        started: float,
    ) -> "Manifold":
        """Everything after the authoring snapshot, under the target lock."""
        folder = target.parent
        inputs = self._fit_inputs(
            snapshot, target, sae=sae, sae_revision=sae_revision,
            layer_indices=layer_indices,
        )
        if not force:
            cached = load_published_fit(folder, target, inputs.accepts)
            if cached is not None:
                assert_authoring_revision(
                    folder, target, snapshot.revision, _default_prompts,
                )
                progress(f"Loaded cached manifold '{inputs.manifold.name}'.")
                if emit_event:
                    self._events.emit(ManifoldExtracted(
                        name=inputs.manifold.name, manifold=cached,
                        metadata=dict(cached.metadata),
                    ))
                return cached

        fit_layers = self._fit_layers(inputs)
        self._require_neutral_baseline(inputs.manifold.name, fit_layers)
        captured = self._capture(inputs, fit_layers, force=force, progress=progress)
        try:
            manifold, metadata = self._fit_geometry(
                inputs, captured, fit_layers, dls=dls, progress=progress,
            )
        finally:
            captured.close()
        publish_fit_if_current(
            folder, expected_revision=snapshot.revision, manifold=manifold,
            tensor_path=target, metadata=metadata,
            default_prompts=_default_prompts,
        )
        manifold.metadata.update(metadata)
        if emit_event:
            self._events.emit(ManifoldExtracted(
                name=manifold.name, manifold=manifold, metadata=metadata,
            ))
        progress(f"Fit complete in {time.perf_counter() - started:.1f}s.")
        return manifold

    # -- stage 1: inputs -----------------------------------------------------

    def _fit_inputs(
        self,
        snapshot: AuthoringSnapshot,
        tensor_path: Path,
        *,
        sae: str | SaeBackend | None,
        sae_revision: str | None,
        layer_indices: "Sequence[int] | str | None",
    ) -> FitInputs:
        handle = self._handle
        model = handle.model
        model_fingerprint = loaded_model_fingerprint(model, handle.model_id)
        # A string release resolves through SAELens metadata before a hit is
        # accepted (transform identity is part of the key); the backend stays
        # one-layer lazy, so a valid hit loads no SAE weights.
        backend: SaeBackend | None
        if isinstance(sae, str):
            from drowse.sae.backend import load_sae_backend

            backend = load_sae_backend(
                sae, revision=sae_revision, model_id=handle.model_id,
                device=handle.device,
            )
            key_revision = sae_revision
        else:
            backend = sae
            key_revision = sae.revision if sae is not None else None
        identity = prepare_manifold_capture_identity(
            handle, snapshot.manifold, model_fingerprint,
            node_groups=snapshot.node_groups,
            baseline_prompts=snapshot.baseline_prompts,
        )
        if backend is not None and any(r is not None for r in identity.node_roles):
            raise ValueError(
                "SAE-backed and role-baselined manifold fits are mutually exclusive"
            )
        source_fp = getattr(
            getattr(model, "_orig_mod", model), "_drowse_source_fingerprint", None,
        )
        return FitInputs(
            snapshot=snapshot,
            identity=identity,
            tensor_path=tensor_path,
            model_fingerprint=model_fingerprint,
            model_source_fingerprint=(
                source_fp if isinstance(source_fp, str) and source_fp else None
            ),
            requested_layers=resolve_fit_layer_indices(
                layer_indices, len(handle.layers),
            ),
            layer_scope_explicit=layer_indices is not None,
            sae_backend=backend,
            sae_revision=key_revision,
        )

    def _fit_layers(self, inputs: FitInputs) -> list[int]:
        """The layers to fit — validated against SAE coverage before any capture."""
        backend = inputs.sae_backend
        if backend is None:
            return inputs.requested_layers
        from drowse.errors import SaeCoverageError

        covered = set(backend.layers) & set(range(len(self._handle.layers)))
        if inputs.layer_scope_explicit:
            missing = sorted(set(inputs.requested_layers) - covered)
            if missing:
                raise SaeCoverageError(
                    f"SAE release {backend.release!r} does not cover "
                    f"requested layers {missing}"
                )
            return inputs.requested_layers
        if not covered:
            raise SaeCoverageError(
                f"SAE release {backend.release!r} covers no layers "
                f"of {self._handle.model_id}"
            )
        return sorted(covered)

    def _require_neutral_baseline(self, name: str, fit_layers: list[int]) -> None:
        """Gate the mandatory whitener + neutral means before the model pass."""
        from drowse.errors import WhitenerError

        whitener = self._handle.whitener
        if whitener is None or not whitener.covers_all(fit_layers):
            raise WhitenerError(
                f"manifold {name!r}: the Mahalanobis whitener must cover "
                f"every fit layer (regenerate the neutral activation cache "
                f"for {self._handle.model_id!r})"
            )
        missing = sorted(set(fit_layers) - set(self._handle.layer_means))
        if missing:
            raise WhitenerError(
                f"manifold {name!r}: neutral layer means must cover every "
                f"fit layer (missing {missing})"
            )

    # -- stage 2: capture ----------------------------------------------------

    def _capture(
        self,
        inputs: FitInputs,
        fit_layers: list[int],
        *,
        force: bool,
        progress: Callable[[str], None],
    ) -> _Captured:
        """Cached centroids (+ rows) for every fit layer, topping up misses.

        Response rows are retained only when the fit may come out curved and
        raw.  A pinned-curved fit captures and persists missing rows up front,
        sharing the centroid pass; ``auto`` spools rows only alongside missing
        centroids and leaves persistence and any gap to :meth:`_curved_rows`,
        so a fit that resolves flat never writes its speculative spool.
        ``force`` treats the selected layers as misses.
        """
        ident = inputs.identity
        mode = inputs.manifold.fit_mode
        cache = CaptureCache(
            self._handle.model_id, ident.capture_sha256, ident.node_sizes,
        )
        retain_rows = inputs.sae_backend is None and mode in _ROW_CONSUMING_MODES
        with cache.capturing():
            stacks = {} if force else cache.load_centroids(fit_layers)
            rows = (
                cache.load_rows(sorted(stacks))
                if retain_rows and not force else None
            )
            durable_rows = set(rows.layer_indices) if rows is not None else set()
            needed = set(fit_layers) - set(stacks)
            if retain_rows and mode != "auto":
                needed |= set(fit_layers) - durable_rows
            if needed:
                fresh, new_rows = self._run_capture(
                    ident, sorted(needed), retain_rows=retain_rows,
                    progress=progress,
                )
                stacks.update(fresh)
                rows = _union_rows(rows, new_rows)
                try:
                    cache.save_centroids(fresh)
                    if new_rows is not None and mode != "auto":
                        cache.save_rows(new_rows, new_rows.layer_indices)
                        durable_rows.update(new_rows.layer_indices)
                except (OSError, RuntimeError, ValueError) as exc:
                    # The capture is resident and valid; a cache write failure
                    # costs the next fit a recapture, never this fit.
                    progress(f"Activation capture cache write skipped: {exc}")
            elif fit_layers:
                progress("Reusing token-exact manifold activation capture cache...")
        try:
            cache.prune()
        except OSError as exc:
            progress(f"Activation capture cache prune skipped: {exc}")
        return _Captured(cache, stacks, rows, durable_rows)

    def _run_capture(
        self,
        ident: CaptureIdentity,
        layers: list[int],
        *,
        retain_rows: bool,
        progress: Callable[[str], None],
    ) -> tuple[dict[int, torch.Tensor], "ActivationRowStore | None"]:
        """One fit-wide batched model pass over every node row."""
        from drowse.fit.capture import _ReusablePooledCapture
        from drowse.fit.node_stats import compute_manifold_node_stats

        handle = self._handle
        started = time.perf_counter()
        n_rows = sum(ident.node_sizes)
        progress(
            f"Pooling {len(ident.node_groups)} nodes in fit-wide batches "
            f"({n_rows} responses, layers {layers})..."
        )
        with _ReusablePooledCapture(handle.model, handle.layers, layers) as context:
            centroids, rows = compute_manifold_node_stats(
                handle.model, handle.tokenizer, handle.layers, handle.device,
                ident.node_groups, ident.baseline_prompts,
                roles=ident.node_roles, model_type=ident.model_type,
                layer_indices=layers, retain_rows=retain_rows,
                prepared_rows=ident.prepared_rows,
                capture_context=context,
            )
        fresh = {
            idx: centroids.pop(idx).to(torch.float32).contiguous()
            for idx in layers
        }
        progress(
            f"Captured {n_rows} rows across {len(layers)} layers in "
            f"{time.perf_counter() - started:.1f}s."
        )
        return fresh, rows

    def _curved_rows(
        self,
        inputs: FitInputs,
        captured: _Captured,
        layers: list[int],
        progress: Callable[[str], None],
    ) -> "ActivationRowStore":
        """Rows for every curved layer, recapturing any gap; persist new rows."""
        with captured.cache.capturing():
            have = set(captured.rows.layer_indices) if captured.rows else set()
            missing = sorted(set(layers) - have)
            if missing:
                _centroids, extra = self._run_capture(
                    inputs.identity, missing, retain_rows=True, progress=progress,
                )
                captured.rows = _union_rows(captured.rows, extra)
            rows = captured.rows
            if rows is None:
                raise RuntimeError("curved raw manifold fit has no activation rows")
            fresh = sorted(set(rows.layer_indices) - captured.durable_rows)
            try:
                captured.cache.save_rows(rows, fresh)
                captured.durable_rows.update(fresh)
            except (OSError, RuntimeError, ValueError) as exc:
                progress(f"Activation-row cache write skipped: {exc}")
        return rows

    # -- stage 3: geometry ---------------------------------------------------

    def _fit_geometry(
        self,
        inputs: FitInputs,
        captured: _Captured,
        fit_layers: list[int],
        *,
        dls: bool,
        progress: Callable[[str], None],
    ) -> tuple["Manifold", dict[str, Any]]:
        if inputs.manifold.fit_mode == "pca" and len(inputs.identity.node_groups) == 1:
            return self._fit_monopolar(inputs, captured, fit_layers, progress)
        return self._fit_nodes(
            inputs, captured, fit_layers, dls=dls, progress=progress,
        )

    def _fit_monopolar(
        self,
        inputs: FitInputs,
        captured: _Captured,
        fit_layers: list[int],
        progress: Callable[[str], None],
    ) -> tuple["Manifold", dict[str, Any]]:
        """A 1-node ``pca`` folder: the concept against the neutral baseline.

        With no second pole to span an affine subspace, the steering direction
        is the concept centroid's displacement from the model's neutral mean ν,
        folded into a 1-node neutral-anchored ray — no discover coords, no
        DLS, no synthetic second node.  ``concept − ν`` cancels common-mode
        like DiM, so the raw δ̂ basis is appropriate.
        """
        from drowse.geometry.fold import fold_directions_to_subspace

        mf = inputs.manifold
        backend = inputs.sae_backend
        means = self._handle.layer_means
        label = inputs.identity.node_groups[0][0]
        stacks = captured.take_stacks()
        directions: dict[int, torch.Tensor] = {}
        for idx in fit_layers:
            concept = stacks.pop(idx)
            if backend is not None:
                concept = _sae_reconstruct(backend, idx, concept, self._handle.device)
            directions[idx] = (
                concept.reshape(-1) - means[idx].to(torch.float32).reshape(-1)
            )
        progress(
            f"Folding monopolar '{label}' − neutral across "
            f"{len(directions)} layers..."
        )
        manifold = fold_directions_to_subspace(
            mf.name, directions, means, whitener=self._handle.whitener,
            label=label, feature_space=inputs.feature_space,
        )
        # The fold primitive is role-agnostic; carry role/kind so steer-time
        # role substitution (``nearest_node_role``) and provenance work.
        manifold.node_roles = list(inputs.identity.node_roles)
        manifold.node_kinds = list(inputs.identity.node_kinds)
        metadata = inputs.base_metadata(
            method=(
                "manifold_monopolar_sae" if backend is not None
                else "manifold_monopolar"
            ),
            fitted_layers=list(fit_layers),
            subspace_metric="euclidean",
        )
        metadata.update({
            "fit_mode": mf.fit_mode,
            "hyperparams": dict(mf.hyperparams),
            "diagnostics": {},
        })
        return manifold, metadata

    def _fit_nodes(
        self,
        inputs: FitInputs,
        captured: _Captured,
        fit_layers: list[int],
        *,
        dls: bool,
        progress: Callable[[str], None],
    ) -> tuple["Manifold", dict[str, Any]]:
        """K ≥ 2 nodes: whitened geometry → topology → per-layer subspaces."""
        from drowse.geometry.manifold import Manifold

        mf = inputs.manifold
        handle = self._handle
        whitener = handle.whitener
        backend = inputs.sae_backend
        n_nodes = len(inputs.identity.node_groups)

        # Per-layer (K, D) fp32 centroid stacks, SAE-reconstructed when a
        # backend is set — one layer at a time, so no raw roster outlives its
        # decoded replacement.
        stacks = captured.take_stacks()
        if backend is not None:
            for idx in fit_layers:
                stacks[idx] = _sae_reconstruct(
                    backend, idx, stacks.pop(idx), handle.device,
                )
        # Whitened between-node geometry, computed once and shared by
        # discovery, topology scoring, the Fisher basis, and the neutral layout.
        whitened = _whitened_node_geometry(stacks, fit_layers, whitener=whitener)
        geometry = _resolve_fit_geometry(
            mf,
            K=n_nodes,
            fit_layers=fit_layers,
            stacks=stacks,
            layer_grams=whitened.grams,
            whitened_rows=whitened.rows,
            whitener=whitener,
            sae_backend=backend,
            progress=progress,
        )
        node_coords = geometry.node_coords
        progress(f"Fitting RBF interpolant across {len(fit_layers)} layers...")
        rbf_smoothing: dict[int, dict[str, float]] = {}
        rbf_plan: Any = None
        if geometry.effective_fit_mode == "pca":
            layer_subs, shares = self._fit_affine_layers(
                stacks, whitened, geometry, fit_layers, dls=dls,
            )
            node_coords = _neutral_centered_layout(
                node_coords, whitened, handle.layer_means, fit_layers,
            )
        else:
            layer_subs, shares, rbf_smoothing, rbf_plan = self._fit_curved_layers(
                stacks, whitened, geometry, fit_layers,
            )
        # The K×D×L centroid rosters and their whitened siblings have no
        # consumer past this point; release them before the row pass.
        geometry.fisher_bases.clear()
        stacks.clear()
        whitened.rows.clear()
        whitened.grams.clear()
        whitened.centroid_means.clear()

        sigma_field: dict[int, dict[str, float]] = {}
        origin: dict[int, torch.Tensor] = {}
        if geometry.effective_fit_mode != "pca":
            if backend is None and layer_subs:
                sigma_field = self._fit_sigma(
                    inputs, captured, layer_subs, geometry, node_coords,
                    rbf_plan, progress,
                )
            origin = _neutral_origin(
                layer_subs, geometry.domain, node_coords, handle.layer_means,
            )

        manifold = Manifold(
            name=mf.name,
            domain=geometry.domain,
            node_labels=[label for label, _ in inputs.identity.node_groups],
            node_coords=node_coords,
            layers=layer_subs,
            feature_space=inputs.feature_space,
            node_roles=list(inputs.identity.node_roles),
            node_kinds=list(inputs.identity.node_kinds),
            mahalanobis_share=shares,
            origin=origin,
        )
        # ``save_manifold`` derives ``fitted_layers`` from the tensor roster;
        # ``node_spread_per_layer`` keeps the complete evaluated roster, so
        # DLS-pruned layers stay distinguishable from never-fitted ones.
        metadata = inputs.base_metadata(
            method=geometry.method,
            fitted_layers=sorted(layer_subs),
            subspace_metric="mahalanobis",
        )
        # Diagnostic layer profile: the whitened between-node spread
        # ``tr(G_L)`` — surfaced by `manifold show`, consumed by nothing.
        metadata["node_spread_per_layer"] = {
            str(idx): whitened.node_spread[idx] for idx in fit_layers
        }
        if rbf_smoothing:
            # Penalized-RBF provenance (GCV λ + effective dof per layer).
            metadata["rbf_smoothing_per_layer"] = {
                str(idx): info for idx, info in rbf_smoothing.items()
            }
        if sigma_field:
            # σ-field summary per layer; the σ-RBF tensors ride the safetensors.
            metadata["sigma_field_per_layer"] = {
                str(idx): info for idx, info in sigma_field.items()
            }
        metadata.update(geometry.discover_metadata)
        return manifold, metadata

    def _fit_affine_layers(
        self,
        stacks: dict[int, torch.Tensor],
        whitened: "_WhitenedNodeGeometry",
        geometry: "_ResolvedGeometry",
        fit_layers: list[int],
        *,
        dls: bool,
    ) -> tuple[dict[int, "LayerSubspace"], dict[int, float]]:
        """Flat fit: a whitened/Fisher rank-``k`` affine subspace per layer.

        The steerable subspace is exactly the ``k``-dim layout span, so
        ``n_components`` is the layout width (no ``max_subspace_dim`` knob);
        the frame is neutral-anchored with real per-layer node coords.  DLS
        then keeps each axis whose node projections straddle neutral
        (``dls=False`` keeps every axis); a layer with no surviving axis drops.
        """
        from drowse.fit.capture import compute_dls_axes
        from drowse.geometry.subspace import fit_affine_subspace, subspace_share

        whitener = self._handle.whitener
        means = self._handle.layer_means
        width = int(geometry.node_coords.shape[1])
        raw_fits = {}
        for idx in fit_layers:
            fit = fit_affine_subspace(
                stacks[idx], neutral_mean=means[idx],
                whitener=whitener, layer=idx,
                whitened_gram=whitened.grams[idx],
                whitened_rows=whitened.rows[idx],
                orient_to=0,
                basis_override=geometry.fisher_bases.get(idx),
                n_components=width,
            )
            raw_fits[idx] = (fit.subspace, fit.mu_coords)
        kept = compute_dls_axes(
            {idx: stacks[idx] for idx in raw_fits},
            {idx: sub.basis for idx, (sub, _mu) in raw_fits.items()},
            means if dls else None,
        )
        layer_subs: dict[int, LayerSubspace] = {}
        shares: dict[int, float] = {}
        for idx, (sub, mu_coords) in raw_fits.items():
            axes = sorted(kept.get(idx, set()))
            if not axes:
                continue
            if len(axes) < sub.rank:
                sub = sub.select_axes(axes)
                mu_coords = mu_coords[:, axes]
            layer_subs[idx] = sub
            # Per-layer budget = the μ-centered (anchor-independent) whitened
            # spread: signal spread, not where neutral sits.
            shares[idx] = subspace_share(
                mu_coords, sub.basis, whitener=whitener, layer=idx,
            )
        return layer_subs, shares

    def _fit_curved_layers(
        self,
        stacks: dict[int, torch.Tensor],
        whitened: "_WhitenedNodeGeometry",
        geometry: "_ResolvedGeometry",
        fit_layers: list[int],
    ) -> tuple[
        dict[int, "LayerSubspace"], dict[int, float],
        dict[int, dict[str, float]], Any,
    ]:
        """Curved fit: a neutral-anchored RBF surface per layer.

        ``curved_smoothing`` is ``None`` for authored (exact interpolation:
        node = steering target) and the GCV / fixed-λ selector for spectral.
        """
        from drowse.geometry.rbf import RbfFitPlan
        from drowse.geometry.subspace import fit_curved_subspace, subspace_share

        whitener = self._handle.whitener
        means = self._handle.layer_means
        plan = geometry.rbf_plan or RbfFitPlan.build(
            geometry.node_params, smoothing=geometry.curved_smoothing,
        )
        extra: dict[str, Any] = {}
        if geometry.max_subspace_dim_override is not None:
            extra["n_components"] = geometry.max_subspace_dim_override
        layer_subs: dict[int, LayerSubspace] = {}
        shares: dict[int, float] = {}
        smoothing: dict[int, dict[str, float]] = {}
        for idx in fit_layers:
            fit = fit_curved_subspace(
                stacks[idx], plan,
                whitener=whitener, layer=idx,
                neutral_mean=means[idx],
                whitened_gram=whitened.grams[idx],
                whitened_rows=whitened.rows[idx],
                smoothing=geometry.curved_smoothing,
                basis_override=geometry.fisher_bases.get(idx),
                **extra,
            )
            layer_subs[idx] = fit.subspace
            # μ-centered share (NOT ``eval_rbf(node_params)`` — the surface is
            # neutral-anchored, so its node values aren't μ-centered).
            shares[idx] = subspace_share(
                fit.mu_coords, fit.subspace.basis, whitener=whitener, layer=idx,
            )
            if fit.rbf is not None:
                smoothing[idx] = fit.rbf.info()
        return layer_subs, shares, smoothing, plan

    def _fit_sigma(
        self,
        inputs: FitInputs,
        captured: _Captured,
        layer_subs: dict[int, "LayerSubspace"],
        geometry: "_ResolvedGeometry",
        node_coords: torch.Tensor,
        rbf_plan: Any,
        progress: Callable[[str], None],
    ) -> dict[int, dict[str, float]]:
        """The fuzzy-manifold σ-field over retained rows (curved + raw only).

        Each node's within-node reduced covariance yields one off-surface σ;
        a ``log σ`` RBF onto each ``LayerSubspace`` (mutated in place) gives
        the surface the tube thickness soft-``onto`` steers into and the
        monitor reads as a soft-assignment bandwidth.
        """
        from drowse.fit.node_stats import compute_store_reduced_covariances
        from drowse.geometry.tube import fit_sigma_field

        rows = self._curved_rows(inputs, captured, sorted(layer_subs), progress)
        progress(
            f"Fitting within-node σ-field across {len(layer_subs)} layers "
            f"({len(inputs.identity.node_groups)} nodes, retained activation rows)..."
        )
        covariances = compute_store_reduced_covariances(rows, layer_subs)
        captured.close()
        return fit_sigma_field(
            layer_subs, geometry.domain, node_coords, covariances,
            smoothing=(
                geometry.curved_smoothing
                if geometry.curved_smoothing is not None else "auto"
            ),
            rbf_plan=rbf_plan,
        )


def _neutral_centered_layout(
    node_coords: torch.Tensor,
    whitened: "_WhitenedNodeGeometry",
    layer_means: Mapping[int, torch.Tensor],
    fit_layers: Sequence[int],
) -> torch.Tensor:
    """Re-anchor a flat fit's display layout on neutral (a pure translation).

    ``discover_coords`` centers the layout on the node centroid, so a
    coord-form ``% 0,…,0`` would mean "the average node".  Subtracting
    neutral's landmark-MDS projection into the consensus layout — from its
    whitened, node-mean-centered cross-Gram column
    ``gᵢ = mean_L (ν_L − μ_L)ᵀ Σ_L⁻¹ (c_{L,i} − μ_L)`` — makes ``% 0,…,0`` read
    as neutral and shares the geometry plot's origin.  Steering is unchanged:
    the affine push blends per-layer neutral-anchored coords with
    translation-invariant cardinal weights.
    """
    from drowse.geometry.topology import neutral_layout_coord

    cross_gram = torch.stack([
        whitened.rows[idx] @ (
            layer_means[idx].to(torch.float32).reshape(1, -1)
            - whitened.centroid_means[idx]
        ).reshape(-1)
        for idx in fit_layers
    ]).mean(dim=0)
    return (node_coords - neutral_layout_coord(node_coords, cross_gram)).contiguous()


def _neutral_origin(
    layer_subs: Mapping[int, "LayerSubspace"],
    domain: Any,
    node_coords: torch.Tensor,
    layer_means: Mapping[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """Per-layer foot of the neutral mean on a curved surface, authoring coords.

    The cold-start foot seed.  Flat fits store none: neutral's foot on an
    affine subspace is reduced coord 0 (the ``!`` ablation target).
    """
    from drowse.geometry.invert import neutral_origin

    return {
        idx: neutral_origin(sub, domain, layer_means[idx], node_coords)
        for idx, sub in layer_subs.items()
    }
