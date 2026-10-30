"""The manifold artifact contracts: ``manifold.json`` and the fitted sidecar.

Pure schema — no filesystem transactions.  Every reader and writer of a
manifold folder goes through the two pairs defined here:

- :func:`parse_manifest` / :func:`manifest_payload` — the one validator and
  the one builder for ``manifold.json``.  :class:`ManifestSpec` is the parsed
  form; :class:`drowse.store.manifold_folder.ManifoldFolder` wraps it with the
  folder's corpus, integrity, and fitted pairs.
- :class:`ManifoldSidecar` — the one field list of the per-model fitted
  sidecar.  :meth:`ManifoldSidecar.build` is the writer's constructor,
  :meth:`ManifoldSidecar.from_payload` the reader's, and both run the same
  exact-schema validator, so a sidecar that loads is exactly a sidecar the
  writer could have produced.  The wire format is mirrored by hand in
  drowse.ai (``webui/src/hosted/artifacts/drowseArchive.ts``); a field or
  version change crosses that seam.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from drowse.base.integrity import LABEL_REGEX, NAME_REGEX, is_sha256_digest
from drowse.errors import DrowseError
from drowse.model.role_templates import _ROLE_SLUG_RE


class ManifoldFormatError(ValueError, DrowseError):
    """Raised when a manifold folder is malformed or fails integrity."""


class BakedManifoldError(ValueError, DrowseError):
    """Raised when an operation invalid for a corpus-less baked manifold runs.

    A baked manifold (``fit_mode == "baked"``) has no node corpus and its
    geometry lives only in the per-model tensor, so a re-fit is impossible.
    Tensor-deleting lifecycle ops (``clear``, scoped ``refresh``) would
    therefore destroy the only copy of the geometry irreversibly — they
    refuse with this error and point the caller at ``manifold rm`` instead.
    """


# Current manifold artifact format.  v10 splits the fit's capture identity into
# separately provable halves — ``capture_version`` + ``capture_render_sha256``
# (the tokenizer render) alongside ``model_source_fingerprint`` (the
# pre-load-verifiable checkpoint identity) and ``baseline_prompts_sha256`` — so
# a fit no-op can be proven without loading weights.  Readers require exactly
# this version; an older fitted sidecar is a stale cache miss, an older
# manifest a regenerate error.
MANIFOLD_FORMAT_VERSION = 10

# ``authored``: the user supplies a domain + per-node coords (curved RBF fit).
# Discover modes derive coordinates per model at fit time: ``pca`` (flat),
# ``spectral`` (curved), ``auto`` (``select_topology`` picks per model).
# ``baked``: a corpus-less, pre-fitted artifact (merge outputs) whose geometry
# lives only in the per-model tensor; it never re-fits and is not discover.
FIT_MODES_DISCOVER: frozenset[str] = frozenset({"pca", "spectral", "auto"})
FIT_MODES_ALL: frozenset[str] = frozenset({"authored", "baked"}) | FIT_MODES_DISCOVER
MERGE_BAKE_POLICY = "additive_union_v1"

# Manifest keys that carry *local* transaction state rather than authored
# content: ``files`` accumulates per-model fit proofs, ``artifact_id`` and
# ``fit_epochs`` are per-install identity/epoch counters.  Bundle-drift hashes
# and published/exported manifests strip them.
LOCAL_STATE_KEYS: tuple[str, ...] = ("files", "artifact_id", "fit_epochs")

_MANIFEST_COMMON_FIELDS = frozenset({
    "format_version", "name", "description", "fit_mode", "nodes", "files",
    "source", "tags", "artifact_id", "fit_epochs", "template_ref",
})

# Per-mode hyperparameter whitelists.  ``max_dim`` caps the derived layout dim
# (for a flat ``pca`` fit the steerable subspace *is* that span, so it is the
# only dim knob).  ``max_subspace_dim`` and ``min_dim`` apply only to curved
# fits; ``auto`` accepts the union its candidate fits consume plus
# ``persistence_frac`` (the H1 loop-significance threshold).  Coordinate
# derivation is layer-agnostic, so there is no reference-layer knob.
HYPERPARAMS_BY_MODE: dict[str, frozenset[str]] = {
    "pca": frozenset({"max_dim", "var_threshold"}),
    "spectral": frozenset({
        "max_dim", "min_dim", "k_nn", "bandwidth", "max_subspace_dim",
        "smoothing",
    }),
    "auto": frozenset({
        "max_dim", "var_threshold", "min_dim", "k_nn", "bandwidth",
        "max_subspace_dim", "smoothing", "persistence_frac",
    }),
}
_INT_HYPERPARAMS = frozenset({"max_dim", "min_dim", "k_nn", "max_subspace_dim"})
_FLOAT_HYPERPARAMS = frozenset({"var_threshold", "bandwidth", "persistence_frac"})

# Which fitted ``method`` may carry which ``fit_mode``.  Mirrored by drowse.ai.
METHOD_FIT_MODES: dict[str, frozenset[str]] = {
    "manifold_pca": frozenset({"authored"}),
    "manifold_sae": frozenset({"authored"}),
    "manifold_monopolar": frozenset({"pca"}),
    "manifold_monopolar_sae": frozenset({"pca"}),
    "manifold_discover_auto": frozenset({"auto"}),
    "manifold_discover_pca": frozenset({"pca"}),
    "manifold_discover_spectral": frozenset({"spectral"}),
    "manifold_discover_sae": frozenset({"pca", "spectral", "auto"}),
    "manifold_procrustes_transfer": FIT_MODES_ALL,
    "merge": frozenset({"baked"}),
}

NODE_KINDS = ("abstract", "concrete", "custom")


def validate_manifold_format_version(
    value: Any, *, location: str = "manifold",
) -> int:
    """Validate the shared readable/writable manifold format boundary."""
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value != MANIFOLD_FORMAT_VERSION
    ):
        raise ManifoldFormatError(
            f"{location} has format_version={value!r}; "
            f"need exactly {MANIFOLD_FORMAT_VERSION}. Regenerate or re-fit it "
            "with the current drowse."
        )
    return value


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def sanitize_hyperparams(
    fit_mode: str, hyperparams: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate and return hyperparameters for one exact discover fit mode.

    Single source of truth for the per-method whitelist: a key the selected
    dispatcher would not consume raises rather than landing in the manifest.
    """
    if hyperparams is None:
        return {}
    allowed = HYPERPARAMS_BY_MODE.get(fit_mode)
    if allowed is None:
        raise ManifoldFormatError(f"unknown fit_mode {fit_mode!r}")
    if not isinstance(hyperparams, Mapping):
        raise ManifoldFormatError(f"fit_mode {fit_mode!r} hyperparams must be an object")
    invalid = set(hyperparams) - allowed
    if invalid:
        raise ManifoldFormatError(
            f"fit_mode {fit_mode!r} does not accept hyperparameter(s): "
            f"{', '.join(sorted(invalid))}"
        )
    for key, value in hyperparams.items():
        if key in _INT_HYPERPARAMS:
            valid = isinstance(value, int) and not isinstance(value, bool) and value > 0
        elif key in _FLOAT_HYPERPARAMS:
            valid = _finite_number(value)
        else:  # smoothing
            valid = value == "auto" or _finite_number(value)
        if not valid:
            raise ManifoldFormatError(
                f"fit_mode {fit_mode!r} hyperparameter {key!r} has invalid "
                f"value {value!r}"
            )
    return dict(hyperparams)


def domain_label(spec: Mapping[str, Any]) -> str:
    """Short ``type(Nd)`` label for a manifold domain spec dict."""
    kind = spec.get("type", "?")
    if kind == "klein":
        return "Klein bottle (2D)"
    if kind == "projective":
        return "Projective plane / RP² (2D)"
    if kind == "box":
        n = len(spec.get("axes", []))
    elif kind == "sphere":
        n = int(spec.get("dim", 0))
    elif kind == "custom":
        n = int(spec.get("embed_dim", 0))
    else:
        n = 0
    return f"{kind}({n}d)"


def min_nodes(n: int) -> int:
    """Minimum node count for an ``n``-dimensional curved manifold.

    ``2*n + 1`` — one center node plus two extremes per authoring axis, so
    every axis is sampled at three levels and curvature is fittable.  This is
    *necessary* but not *sufficient*: the nodes must also be poised (affinely
    span the embedding), which the RBF fit enforces.
    """
    return 2 * n + 1


def node_filename(index: int, label: str) -> str:
    """Canonical zero-padded node corpus filename: ``NN_<label>.json``."""
    return f"{index:02d}_{label}.json"


def validate_node_label(name: str, label: Any) -> str:
    """Validate one node label against the grammar-addressable label subset."""
    if not isinstance(label, str) or not LABEL_REGEX.match(label):
        raise ManifoldFormatError(
            f"manifold {name!r} node label {label!r} invalid; a node label is "
            f"a grammar-addressable identifier (no '.', reserved as the "
            f"bipolar separator) — must match {LABEL_REGEX.pattern}"
        )
    return label


def validate_node_labels(name: str, labels: Any) -> list[str]:
    """Validate a non-empty, duplicate-free label list, preserving order."""
    if not isinstance(labels, list) or not labels:
        raise ManifoldFormatError(f"manifold {name!r} needs at least one node label")
    out = [validate_node_label(name, label) for label in labels]
    if len(set(out)) != len(out):
        raise ManifoldFormatError(f"manifold {name!r} has duplicate node labels")
    return out


def validate_node_role(name: str, label: str, role: Any) -> str | None:
    """Validate an optional per-node assistant-role substitution slug.

    ``None`` means "use the standard assistant baseline".  Family support is a
    fit-time concern (the folder is model-agnostic).
    """
    if role is None:
        return None
    if not isinstance(role, str) or not _ROLE_SLUG_RE.match(role):
        raise ManifoldFormatError(
            f"manifold {name!r} node {label!r} role {role!r} invalid; "
            f"must match {_ROLE_SLUG_RE.pattern}"
        )
    return role


def validate_node_kind(name: str, label: str, kind: Any) -> str | None:
    """Validate an optional per-node generation ``kind`` (provenance only)."""
    if kind is None:
        return None
    if not isinstance(kind, str) or kind not in NODE_KINDS:
        raise ManifoldFormatError(
            f"manifold {name!r} node {label!r} kind {kind!r} invalid; "
            f"must be one of {NODE_KINDS}"
        )
    return kind


def resolve_node_maps(
    name: str,
    labels: Iterable[str],
    roles: Mapping[str, Any] | None,
    kinds: Mapping[str, Any] | None,
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """Validate optional ``{label: role}`` / ``{label: kind}`` maps for a roster.

    Keys outside the roster raise; missing labels default to ``None``.
    """
    labels = list(labels)
    resolved: list[dict[str, str | None]] = []
    for field_name, mapping, validate in (
        ("node_roles", roles, validate_node_role),
        ("node_kinds", kinds, validate_node_kind),
    ):
        mapping = mapping or {}
        unknown = set(mapping) - set(labels)
        if unknown:
            raise ManifoldFormatError(
                f"manifold {name!r} {field_name} carries labels not in the "
                f"roster: {sorted(unknown)}"
            )
        resolved.append({
            label: validate(name, label, mapping.get(label)) for label in labels
        })
    return resolved[0], resolved[1]


def require_discover_mode(name: str, fit_mode: Any) -> str:
    """Raise unless ``fit_mode`` is one of the discover modes."""
    if fit_mode not in FIT_MODES_DISCOVER:
        raise ManifoldFormatError(
            f"discover manifold {name!r} fit_mode {fit_mode!r} invalid; "
            f"expected one of {sorted(FIT_MODES_DISCOVER)}"
        )
    return fit_mode


# ============================================================ manifest ===


@dataclass(frozen=True)
class ManifestNode:
    """One ``manifold.json`` node entry.  ``coords`` is authored-only."""

    label: str
    role: str | None = None
    kind: str | None = None
    coords: tuple[float, ...] | None = None

    def payload(self) -> dict[str, Any]:
        if self.coords is None:
            return {"label": self.label, "role": self.role, "kind": self.kind}
        return {
            "label": self.label, "coords": [float(c) for c in self.coords],
            "role": self.role, "kind": self.kind,
        }


@dataclass(frozen=True)
class ManifestSpec:
    """The parsed, validated content of one ``manifold.json``."""

    name: str
    description: str
    fit_mode: str
    nodes: tuple[ManifestNode, ...]
    domain: dict[str, Any]
    hyperparams: dict[str, Any]
    files: dict[str, str]
    source: str
    tags: tuple[str, ...]
    template_ref: str | None
    artifact_id: str | None = None
    fit_epochs: dict[str, Any] | None = None

    @property
    def labels(self) -> list[str]:
        return [node.label for node in self.nodes]


def manifest_payload(
    *,
    name: str,
    description: str,
    fit_mode: str,
    nodes: Iterable[ManifestNode],
    domain: Mapping[str, Any] | None = None,
    hyperparams: Mapping[str, Any] | None = None,
    files: Mapping[str, str] | None = None,
    source: str = "local",
    tags: Iterable[str] = (),
    template_ref: str | None = None,
    artifact_id: str | None = None,
    fit_epochs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The one builder for a ``manifold.json`` payload.

    Authored and baked manifests carry ``domain``; discover manifests carry
    ``hyperparams``.  The local-state keys ``artifact_id`` / ``fit_epochs``
    appear only when set.
    """
    payload: dict[str, Any] = {
        "format_version": MANIFOLD_FORMAT_VERSION,
        "name": name,
        "description": description,
        "fit_mode": fit_mode,
    }
    if fit_mode in FIT_MODES_DISCOVER:
        payload["hyperparams"] = dict(hyperparams or {})
    else:
        payload["domain"] = dict(domain or {})
    payload["nodes"] = [node.payload() for node in nodes]
    payload["files"] = dict(files or {})
    payload["source"] = source or "local"
    payload["tags"] = [str(tag) for tag in tags]
    payload["template_ref"] = template_ref
    if artifact_id is not None:
        payload["artifact_id"] = artifact_id
    if fit_epochs is not None:
        payload["fit_epochs"] = dict(fit_epochs)
    return payload


def spec_payload(spec: ManifestSpec) -> dict[str, Any]:
    """Serialize a :class:`ManifestSpec` through :func:`manifest_payload`."""
    return manifest_payload(
        name=spec.name, description=spec.description, fit_mode=spec.fit_mode,
        nodes=spec.nodes, domain=spec.domain, hyperparams=spec.hyperparams,
        files=spec.files, source=spec.source, tags=spec.tags,
        template_ref=spec.template_ref, artifact_id=spec.artifact_id,
        fit_epochs=spec.fit_epochs,
    )


def load_manifest_data(path: Path) -> dict[str, Any]:
    """Read ``manifold.json`` as a version-checked JSON object."""
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ManifoldFormatError(f"{path}: manifest is unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifoldFormatError(f"{path}: manifest must be a JSON object")
    validate_manifold_format_version(data.get("format_version"), location=str(path))
    return data


def parse_manifest(data: Any, *, location: str = "manifold.json") -> ManifestSpec:
    """Validate one ``manifold.json`` payload against the exact current schema.

    Pure schema: corpus presence and the ``files`` integrity map are checked by
    :meth:`drowse.store.manifold_folder.ManifoldFolder.load`, which is why a
    partially generated discover skeleton can still be parsed here.
    """
    from drowse.geometry.manifold import domain_from_spec

    if not isinstance(data, dict):
        raise ManifoldFormatError(f"{location} must be a JSON object")
    validate_manifold_format_version(data.get("format_version"), location=location)
    name = data.get("name")
    if not isinstance(name, str) or not NAME_REGEX.match(name):
        raise ManifoldFormatError(
            f"manifold name {name!r} invalid; must match {NAME_REGEX.pattern}"
        )
    fit_mode = data.get("fit_mode")
    if fit_mode not in FIT_MODES_ALL:
        raise ManifoldFormatError(
            f"manifold {name!r} fit_mode {fit_mode!r} invalid; "
            f"expected one of {sorted(FIT_MODES_ALL)}"
        )
    discover = fit_mode in FIT_MODES_DISCOVER
    mode_field = "hyperparams" if discover else "domain"
    unknown = set(data) - _MANIFEST_COMMON_FIELDS - {mode_field}
    if unknown:
        raise ManifoldFormatError(
            f"manifold {name!r} has unknown field(s): {sorted(unknown)}"
        )
    for key, expected in (
        ("description", str), ("files", dict), ("source", str), ("tags", list),
        ("nodes", list),
    ):
        if not isinstance(data.get(key), expected):
            raise ManifoldFormatError(
                f"manifold {name!r} field {key!r} must be {expected.__name__}"
            )
    if not data["source"]:
        raise ManifoldFormatError(f"manifold {name!r} field 'source' must be non-empty")
    if not all(isinstance(tag, str) for tag in data["tags"]):
        raise ManifoldFormatError(f"manifold {name!r} 'tags' must be a list of strings")
    artifact_id = data.get("artifact_id")
    if "artifact_id" in data and (not isinstance(artifact_id, str) or not artifact_id):
        raise ManifoldFormatError(
            f"manifold {name!r} field 'artifact_id' must be a non-empty str"
        )
    fit_epochs = data.get("fit_epochs")
    if "fit_epochs" in data and not isinstance(fit_epochs, dict):
        raise ManifoldFormatError(f"manifold {name!r} field 'fit_epochs' must be dict")
    if "template_ref" not in data:
        raise ManifoldFormatError(f"manifold {name!r} needs explicit 'template_ref'")
    template_ref = data["template_ref"]
    if template_ref is not None:
        if not isinstance(template_ref, str) or not template_ref:
            raise ManifoldFormatError(
                f"manifold {name!r} 'template_ref' must be a non-empty string"
            )
        if not discover:
            raise ManifoldFormatError(
                f"manifold {name!r} carries a 'template_ref' but fit_mode is "
                f"{fit_mode!r}; templated corpora are a discover-mode feature "
                f"({sorted(FIT_MODES_DISCOVER)})"
            )

    raw_nodes = data["nodes"]
    if not raw_nodes:
        raise ManifoldFormatError(f"manifold {name!r} needs a non-empty 'nodes' list")
    domain: dict[str, Any] = {}
    hyperparams: dict[str, Any] = {}
    n_coords: int | None = None
    if discover:
        if not isinstance(data.get("hyperparams"), dict):
            raise ManifoldFormatError(f"manifold {name!r} field 'hyperparams' must be dict")
        hyperparams = sanitize_hyperparams(fit_mode, data["hyperparams"])
    else:
        domain = data.get("domain")
        if not isinstance(domain, dict) or not domain:
            raise ManifoldFormatError(f"{fit_mode} manifold {name!r} needs a 'domain' object")
        try:
            parsed_domain = domain_from_spec(domain)
        except (ValueError, KeyError, TypeError) as exc:
            raise ManifoldFormatError(
                f"manifold {name!r} has an invalid domain: {exc}"
            ) from exc
        if fit_mode == "authored":
            n_coords = parsed_domain.intrinsic_dim
            floor = min_nodes(n_coords)
            if len(raw_nodes) < floor:
                raise ManifoldFormatError(
                    f"manifold {name!r} ({n_coords}-D domain) needs a 'nodes' "
                    f"list of >= {floor} entries, got {len(raw_nodes)}"
                )
    expected_node = (
        {"label", "role", "kind"} if n_coords is None
        else {"label", "coords", "role", "kind"}
    )
    nodes: list[ManifestNode] = []
    for entry in raw_nodes:
        if not isinstance(entry, dict) or set(entry) != expected_node:
            raise ManifoldFormatError(
                f"manifold {name!r} node fields must be {sorted(expected_node)}"
            )
        label = validate_node_label(name, entry["label"])
        coords: tuple[float, ...] | None = None
        if n_coords is not None:
            raw_coords = entry["coords"]
            if (
                not isinstance(raw_coords, list)
                or len(raw_coords) != n_coords
                or not all(_finite_number(c) for c in raw_coords)
            ):
                raise ManifoldFormatError(
                    f"manifold {name!r} node {label!r} needs 'coords' of "
                    f"{n_coords} number(s), got {raw_coords!r}"
                )
            coords = tuple(float(c) for c in raw_coords)
        nodes.append(ManifestNode(
            label=label,
            role=validate_node_role(name, label, entry["role"]),
            kind=validate_node_kind(name, label, entry["kind"]),
            coords=coords,
        ))
    if len({node.label for node in nodes}) != len(nodes):
        raise ManifoldFormatError(f"manifold {name!r} has duplicate node labels")
    return ManifestSpec(
        name=name,
        description=data["description"],
        fit_mode=fit_mode,
        nodes=tuple(nodes),
        domain=dict(domain),
        hyperparams=hyperparams,
        files=dict(data["files"]),
        source=data["source"],
        tags=tuple(data["tags"]),
        template_ref=template_ref,
        artifact_id=artifact_id,
        fit_epochs=dict(fit_epochs) if isinstance(fit_epochs, dict) else None,
    )


# ============================================================ fitted sidecar ===


@dataclass(kw_only=True)
class ManifoldSidecar:
    """JSON sidecar beside a fitted per-model manifold tensor.

    The single field list of the sidecar wire format, in on-disk key order.
    Defaults are the canonical "not set" values.  Everything here is
    provenance — the runtime hot path reads the tensor, never the sidecar —
    but each field is what lets a later reader prove a cached fit current, or
    lets ``pack show`` / the inspector explain how it was reached.
    """

    format_version: int = MANIFOLD_FORMAT_VERSION
    name: str
    # ``manifold_pca`` / ``manifold_sae`` (authored), ``manifold_discover_*``
    # (discover), ``manifold_monopolar*`` (1-node ray), ``merge`` (baked),
    # ``manifold_procrustes_transfer`` (cross-model transfer).
    method: str
    drowse_version: str
    domain: dict[str, Any]
    node_count: int
    node_labels: list[str]
    feature_space: str = "raw"
    fit_mode: str = "authored"
    hyperparams: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Per-layer whitened between-node spread ``{str(L): tr(G_L)}`` over every
    # *evaluated* layer (a superset of ``fitted_layers`` when DLS pruned some).
    node_spread_per_layer: dict[str, Any] = field(default_factory=dict)
    # Apply-time per-layer Mahalanobis share and the authoring-coordinate foot
    # of the neutral mean, both keyed by fitted layer.
    mahalanobis_share_per_layer: dict[str, Any] = field(default_factory=dict)
    origin_per_layer: dict[str, Any] = field(default_factory=dict)
    # Staleness key over the fit's inputs (see ``ManifoldFolder.nodes_sha256``).
    nodes_sha256: Optional[str] = None
    sae_release: Optional[str] = None
    sae_revision: Optional[str] = None
    sae_fingerprint: Optional[str] = None
    sae_ids_by_layer: dict[str, str] = field(default_factory=dict)
    sae_full_coverage: bool = False
    # Loaded-weights identity, and the pre-load-verifiable checkpoint identity
    # (resolved Hub commit or local file hashes + load representation).
    model_fingerprint: Optional[str] = None
    model_source_fingerprint: Optional[str] = None
    # ``capture_sha256`` is the whole capture identity; ``capture_version`` +
    # ``capture_render_sha256`` are its weight-free tokenizer-render half, and
    # ``baseline_prompts_sha256`` names the paired prompts directly.
    # ``context_binding_sha256`` is written only by the browser runtime.
    capture_sha256: Optional[str] = None
    capture_version: Optional[int] = None
    context_binding_sha256: Optional[str] = None
    capture_render_sha256: Optional[str] = None
    baseline_prompts_sha256: Optional[str] = None
    fitted_layers: list[int] = field(default_factory=list)
    fit_policy_version: Optional[int] = None
    # ``mahalanobis`` on an ordinary fit; a monopolar fold labels its raw-δ̂
    # basis ``euclidean`` (a basis-selection label, not a metric fallback).
    share_metric: Optional[str] = None
    subspace_metric: Optional[str] = None
    # Curved-fit summaries: GCV smoothing ``{L: {lambda, edf, gcv}}`` and the
    # fuzzy-manifold tube ``{L: {sigma_mean, sigma_min, sigma_max, lambda}}``.
    rbf_smoothing_per_layer: dict[str, Any] = field(default_factory=dict)
    sigma_field_per_layer: dict[str, Any] = field(default_factory=dict)
    # ``fit_mode="auto"`` only: the resolved geometry and the ranked
    # GCV + persistent-homology candidates behind the choice.
    resolved_fit_mode: Optional[str] = None
    topology_winner: Optional[str] = None
    topology_candidates: list[dict[str, Any]] = field(default_factory=list)
    # Per-node role / kind provenance, aligned with ``node_labels``.
    node_roles: list[str | None] = field(default_factory=list)
    node_kinds: list[str | None] = field(default_factory=list)
    # Merge provenance (``method == "merge"`` only).
    components: Optional[dict[str, Any]] = None
    bake_policy: Optional[str] = None
    # Cross-model transfer provenance (``manifold_procrustes_transfer`` only).
    source_model_id: Optional[str] = None
    source_model_fingerprint: Optional[str] = None
    transfer_quality_estimate: Optional[float] = None

    @property
    def evaluated_layers(self) -> list[int]:
        """Layers evaluated by the fit before DLS pruning.

        Multi-node activation fits measure ``node_spread_per_layer`` before
        selecting the retained tensor layers; artifact kinds without that
        diagnostic (baked, monopolar) evaluate exactly their fitted roster.
        """
        if self.node_spread_per_layer:
            return sorted(int(layer) for layer in self.node_spread_per_layer)
        return list(self.fitted_layers)

    def to_payload(self) -> dict[str, Any]:
        """The exact on-disk JSON object."""
        return asdict(self)

    @classmethod
    def from_payload(
        cls, data: Any, *, location: str = "manifold sidecar",
    ) -> "ManifoldSidecar":
        """Validate an on-disk payload and build the sidecar from it."""
        return cls(**validate_sidecar_payload(data, location=location))

    @classmethod
    def build(
        cls,
        *,
        name: str,
        method: str,
        domain: dict[str, Any],
        node_labels: list[str],
        **provenance: Any,
    ) -> "ManifoldSidecar":
        """The writer's constructor: derive the identity fields and validate.

        ``provenance`` takes any non-derived sidecar field; per-node
        ``node_roles`` / ``node_kinds`` default to all-``None``.
        """
        from drowse import __version__

        count = len(node_labels)
        provenance.setdefault("node_roles", [None] * count)
        provenance.setdefault("node_kinds", [None] * count)
        sidecar = cls(
            name=name, method=method, drowse_version=__version__,
            domain=domain, node_count=count, node_labels=list(node_labels),
            **provenance,
        )
        validate_sidecar_payload(sidecar.to_payload(), location="manifold writer sidecar")
        return sidecar

    @classmethod
    def load(cls, path: Path) -> "ManifoldSidecar":
        """Read and validate one fitted sidecar file.

        The version check is hoisted so an artifact from an older format says
        so rather than reporting whichever field the current schema misses.
        """
        location = f"manifold sidecar {path}"
        try:
            with open(path) as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ManifoldFormatError(f"{location} is unreadable: {exc}") from exc
        if not isinstance(data, dict):
            raise ManifoldFormatError(f"{location} must be a JSON object")
        validate_manifold_format_version(data.get("format_version"), location=location)
        return cls.from_payload(data, location=location)


SIDECAR_FIELDS: frozenset[str] = frozenset(f.name for f in fields(ManifoldSidecar))

# Identity fields the tensor writer derives from the ``Manifold`` itself; a
# caller's metadata can never override them.
SIDECAR_DERIVED_FIELDS: frozenset[str] = frozenset({
    "format_version", "name", "drowse_version", "domain", "node_count",
    "node_labels", "feature_space", "fitted_layers",
    "mahalanobis_share_per_layer", "origin_per_layer",
})


def sidecar_is_current(path: Path) -> bool:
    """Whether a fitted sidecar carries the current schema generation.

    A cheap version peek, deliberately narrower than the full validator: only a
    well-formed *older or newer* stamp counts as stale.  Unreadable bytes and a
    missing or non-integer ``format_version`` return ``True`` so the strict
    loader still runs and reports what is actually wrong — a format bump must
    invalidate fits, never launder corruption.
    """
    try:
        with open(path) as handle:
            value = json.load(handle).get("format_version")
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return True
    if isinstance(value, bool) or not isinstance(value, int):
        return True
    return value == MANIFOLD_FORMAT_VERSION


def _validate_layer_map(
    value: Any, *, location: str, field_name: str, fitted: set[int],
    validate_value: Any, require_fitted: bool = True,
) -> None:
    if not isinstance(value, dict):
        raise ManifoldFormatError(f"{location} field {field_name!r} must be an object")
    for raw_layer, item in value.items():
        if (
            not isinstance(raw_layer, str)
            or not raw_layer.isascii()
            or not raw_layer.isdecimal()
            or str(int(raw_layer)) != raw_layer
            or require_fitted and int(raw_layer) not in fitted
        ):
            raise ManifoldFormatError(
                f"{location} field {field_name!r} has invalid fitted-layer key {raw_layer!r}"
            )
        if not validate_value(item):
            raise ManifoldFormatError(
                f"{location} field {field_name!r} has invalid value for layer {raw_layer}"
            )


_DIAGNOSTIC_LISTS = frozenset({"per_component_variance", "cumulative_variance", "eigenvalues"})
_DIAGNOSTIC_INTS = frozenset({
    "picked_k", "gap_index", "k_nn", "component_count", "heuristic_k", "min_dim",
})
_DIAGNOSTIC_FIELDS = _DIAGNOSTIC_LISTS | _DIAGNOSTIC_INTS | frozenset({
    "threshold", "gap_magnitude", "bandwidth", "pinned",
})


def _validate_diagnostics(value: Any, *, location: str) -> None:
    if not isinstance(value, dict):
        raise ManifoldFormatError(f"{location} diagnostics must be an object")
    if set(value) - _DIAGNOSTIC_FIELDS:
        raise ManifoldFormatError(f"{location} diagnostics has unknown fields")
    for key, item in value.items():
        if key in _DIAGNOSTIC_LISTS:
            valid = isinstance(item, list) and bool(item) and all(_finite_number(v) for v in item)
        elif key in _DIAGNOSTIC_INTS:
            valid = isinstance(item, int) and not isinstance(item, bool) and item >= 0
        elif key == "pinned":
            valid = isinstance(item, bool)
        else:
            valid = _finite_number(item)
        if not valid:
            raise ManifoldFormatError(f"{location} diagnostics field {key!r} is invalid")


_TOPOLOGY_CANDIDATE_FIELDS = frozenset({
    "name", "fit_mode", "intrinsic_dim", "score", "viable", "reason",
})


def _validate_topology_candidates(value: Any, *, location: str) -> None:
    if not isinstance(value, list):
        raise ManifoldFormatError(f"{location} topology_candidates must be an array")
    for candidate in value:
        if not isinstance(candidate, dict) or set(candidate) != _TOPOLOGY_CANDIDATE_FIELDS:
            raise ManifoldFormatError(f"{location} has invalid topology candidate schema")
        if (
            not isinstance(candidate["name"], str) or not candidate["name"]
            or candidate["fit_mode"] not in {"pca", "spectral"}
            or isinstance(candidate["intrinsic_dim"], bool)
            or not isinstance(candidate["intrinsic_dim"], int)
            or candidate["intrinsic_dim"] < 1
            or candidate["score"] is not None and not _finite_number(candidate["score"])
            or not isinstance(candidate["viable"], bool)
            or candidate["reason"] is not None and not isinstance(candidate["reason"], str)
        ):
            raise ManifoldFormatError(f"{location} has invalid topology candidate values")


def _validate_components(value: Any, *, location: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or not value:
        raise ManifoldFormatError(f"{location} components must be null or non-empty object")
    for coord, item in value.items():
        if (
            not isinstance(coord, str) or not coord
            or not isinstance(item, dict)
            or set(item) != {"selector", "alpha", "tensor_sha256"}
            or not isinstance(item["selector"], str) or not item["selector"]
            or not _finite_number(item["alpha"])
            or not is_sha256_digest(item["tensor_sha256"])
        ):
            raise ManifoldFormatError(f"{location} has invalid component provenance")


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_sidecar_payload(
    data: Any, *, location: str = "manifold sidecar",
) -> dict[str, Any]:
    """Validate every field of the exact current fitted-sidecar contract."""
    from drowse.geometry.manifold import validate_domain_spec

    if not isinstance(data, dict) or set(data) != SIDECAR_FIELDS:
        raise ManifoldFormatError(f"{location} does not match the current exact schema")
    validate_manifold_format_version(data["format_version"], location=location)
    try:
        validate_domain_spec(data["domain"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ManifoldFormatError(f"{location} has an invalid domain: {exc}") from exc
    for key in ("name", "method", "drowse_version", "feature_space", "fit_mode"):
        if not isinstance(data[key], str) or not data[key]:
            raise ManifoldFormatError(f"{location} field {key!r} must be non-empty str")
    if data["fit_mode"] not in FIT_MODES_ALL:
        raise ManifoldFormatError(f"{location} has invalid fit_mode")
    allowed_fit_modes = METHOD_FIT_MODES.get(data["method"])
    if allowed_fit_modes is None or data["fit_mode"] not in allowed_fit_modes:
        raise ManifoldFormatError(f"{location} has invalid method/fit_mode combination")
    labels = data["node_labels"]
    if (
        isinstance(data["node_count"], bool)
        or not isinstance(data["node_count"], int)
        or not isinstance(labels, list)
        or not labels
        or data["node_count"] != len(labels)
        or any(not isinstance(label, str) or not LABEL_REGEX.match(label) for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ManifoldFormatError(f"{location} has invalid node identity")
    params = data["hyperparams"]
    if not isinstance(params, dict):
        raise ManifoldFormatError(f"{location} hyperparams must be an object")
    if data["fit_mode"] in FIT_MODES_DISCOVER:
        sanitize_hyperparams(data["fit_mode"], params)
    elif params:
        raise ManifoldFormatError(f"{location} {data['fit_mode']} fit cannot carry hyperparams")
    _validate_diagnostics(data["diagnostics"], location=location)
    if not isinstance(data["sae_full_coverage"], bool):
        raise ManifoldFormatError(f"{location} sae_full_coverage must be bool")
    nullable_strings = (
        "nodes_sha256", "sae_release", "sae_revision", "sae_fingerprint",
        "model_fingerprint", "model_source_fingerprint", "capture_sha256",
        "share_metric", "subspace_metric", "resolved_fit_mode",
        "topology_winner", "bake_policy", "source_model_id",
        "source_model_fingerprint",
    )
    if any(
        data[key] is not None and (not isinstance(data[key], str) or not data[key])
        for key in nullable_strings
    ):
        raise ManifoldFormatError(f"{location} has invalid nullable string metadata")
    # The preflight-provable digests are writer-produced sha256: a malformed one
    # must fail at load, not quietly mismatch forever.
    if any(
        data[key] is not None and not is_sha256_digest(data[key])
        for key in (
            "context_binding_sha256", "capture_render_sha256", "baseline_prompts_sha256",
        )
    ):
        raise ManifoldFormatError(f"{location} has invalid capture digest metadata")
    for key in ("capture_version", "fit_policy_version"):
        if data[key] is not None and not _non_negative_int(data[key]):
            raise ManifoldFormatError(f"{location} has invalid {key}")
    quality = data["transfer_quality_estimate"]
    if quality is not None and not _finite_number(quality):
        raise ManifoldFormatError(f"{location} has invalid transfer quality")
    fitted = data["fitted_layers"]
    if (
        not isinstance(fitted, list)
        or not all(_non_negative_int(v) for v in fitted)
        or fitted != sorted(set(fitted))
    ):
        raise ManifoldFormatError(f"{location} has invalid fitted_layers")
    fitted_set = set(fitted)
    non_negative = lambda value: _finite_number(value) and float(value) >= 0  # noqa: E731
    for field_name, validate_value, require_fitted in (
        # Measured before DLS prunes layers: keys may be a strict superset of
        # ``fitted_layers``, which is what proves a dropped layer was evaluated.
        ("node_spread_per_layer", non_negative, False),
        ("mahalanobis_share_per_layer", non_negative, True),
        ("origin_per_layer", lambda value: (
            isinstance(value, list) and bool(value)
            and all(_finite_number(coord) for coord in value)
        ), True),
        ("sae_ids_by_layer", lambda value: isinstance(value, str) and bool(value), False),
        ("rbf_smoothing_per_layer", lambda value: (
            isinstance(value, dict) and set(value) == {"lambda", "edf", "gcv"}
            and all(_finite_number(item) for item in value.values())
        ), True),
        ("sigma_field_per_layer", lambda value: (
            isinstance(value, dict)
            and set(value) == {"sigma_mean", "sigma_min", "sigma_max", "lambda"}
            and all(non_negative(item) for item in value.values())
        ), True),
    ):
        _validate_layer_map(
            data[field_name], location=location, field_name=field_name,
            fitted=fitted_set, validate_value=validate_value,
            require_fitted=require_fitted,
        )
    roles, kinds = data["node_roles"], data["node_kinds"]
    if (
        not isinstance(roles, list) or not isinstance(kinds, list)
        or len(roles) != len(labels) or len(kinds) != len(labels)
    ):
        raise ManifoldFormatError(f"{location} node provenance is misaligned")
    for label, role, kind in zip(labels, roles, kinds, strict=True):
        validate_node_role(data["name"], label, role)
        validate_node_kind(data["name"], label, kind)
    _validate_topology_candidates(data["topology_candidates"], location=location)
    _validate_components(data["components"], location=location)
    is_auto = data["fit_mode"] == "auto"
    auto_complete = bool(
        data["resolved_fit_mode"] is not None
        and data["topology_winner"] is not None
        and data["topology_candidates"]
    )
    if is_auto != auto_complete:
        raise ManifoldFormatError(f"{location} has inconsistent auto-topology provenance")
    if not is_auto and (
        data["resolved_fit_mode"] is not None
        or data["topology_winner"] is not None
        or data["topology_candidates"]
    ):
        raise ManifoldFormatError(f"{location} non-auto fit carries topology provenance")
    if data["method"] == "merge":
        if (
            data["fit_mode"] != "baked" or data["components"] is None
            or data["bake_policy"] != MERGE_BAKE_POLICY
        ):
            raise ManifoldFormatError(f"{location} has incomplete merge provenance")
    elif data["components"] is not None or data["bake_policy"] is not None:
        raise ManifoldFormatError(f"{location} non-merge fit carries merge provenance")
    transfer_values = (
        data["source_model_id"], data["source_model_fingerprint"],
        data["transfer_quality_estimate"],
    )
    if data["method"] == "manifold_procrustes_transfer":
        if transfer_values[0] is None or transfer_values[1] is None:
            raise ManifoldFormatError(f"{location} has incomplete transfer provenance")
    elif any(value is not None for value in transfer_values):
        raise ManifoldFormatError(f"{location} non-transfer fit carries transfer provenance")
    sae_values = (data["sae_release"], data["sae_revision"], data["sae_fingerprint"])
    if data["feature_space"] == "raw":
        if (
            any(value is not None for value in sae_values)
            or data["sae_ids_by_layer"] or data["sae_full_coverage"]
        ):
            raise ManifoldFormatError(f"{location} raw fit carries SAE provenance")
    elif (
        not data["feature_space"].startswith("sae-")
        or data["sae_release"] is None
        or not data["sae_ids_by_layer"]
    ):
        raise ManifoldFormatError(f"{location} has incomplete SAE provenance")
    return data


__all__ = [
    "BakedManifoldError",
    "FIT_MODES_ALL",
    "FIT_MODES_DISCOVER",
    "HYPERPARAMS_BY_MODE",
    "LOCAL_STATE_KEYS",
    "MANIFOLD_FORMAT_VERSION",
    "MERGE_BAKE_POLICY",
    "METHOD_FIT_MODES",
    "ManifestNode",
    "ManifestSpec",
    "ManifoldFormatError",
    "ManifoldSidecar",
    "SIDECAR_DERIVED_FIELDS",
    "SIDECAR_FIELDS",
    "domain_label",
    "load_manifest_data",
    "manifest_payload",
    "min_nodes",
    "node_filename",
    "parse_manifest",
    "require_discover_mode",
    "resolve_node_maps",
    "sanitize_hyperparams",
    "sidecar_is_current",
    "spec_payload",
    "validate_manifold_format_version",
    "validate_node_kind",
    "validate_node_label",
    "validate_node_labels",
    "validate_node_role",
    "validate_sidecar_payload",
]
