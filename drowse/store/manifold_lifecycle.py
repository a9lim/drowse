"""Lifecycle (rm / clear / refresh), cross-model transfer, and the summary.

Every function here is addressed by a manifold folder or ``(namespace, name)``
and builds on the folder primitives in :mod:`drowse.store.manifold_folder`
(manifest CAS, pair scan / proof, the manifest-then-pairs lock order).  Each
entry point that can change the installed roster drops the selector index
itself, so callers carry no invalidation duty.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from drowse.base.atomic import write_json_atomic
from drowse.base.integrity import is_sha256_digest
from drowse.base.staging import staging_area
from drowse.base.paths import (
    manifold_dir,
    manifolds_dir,
    models_dir,
    safe_model_id,
    tensor_filename,
)
from drowse.errors import DrowseError, ManifoldExistsError, ManifoldNotFoundError
from drowse.geometry.manifold import domain_from_spec
from drowse.store.manifold_folder import (
    ManifoldFolder,
    _locked_manifest,
    fitted_name_selected,
    fitted_pair_files,
    locked_manifold_pairs,
    prove_pair,
    read_manifest,
)
from drowse.store.manifold_schema import (
    BakedManifoldError,
    ManifoldFormatError,
    ManifoldSidecar,
    domain_label,
    min_nodes,
)
from drowse.store.manifold_tensors import fitted_tensor_layers
from drowse.store.selectors import invalidate as invalidate_selector_index


# ============================================================ capture GC ===


def _capture_owner(sidecar_path: Path) -> tuple[str, str] | None:
    """``(safe_model, capture_sha256)`` a fitted sidecar owns, if readable."""
    parsed = fitted_name_selected(sidecar_path.name)
    if parsed is None:
        return None
    try:
        with open(sidecar_path) as handle:
            capture_sha = json.load(handle).get("capture_sha256")
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return None
    return (parsed[0], capture_sha) if is_sha256_digest(capture_sha) else None


def _capture_group_is_referenced(safe_model: str, capture_sha: str) -> bool:
    """Whether any surviving fitted sidecar still owns a shared capture stem."""
    for sidecar_path in manifolds_dir().glob("*/*/*.json"):
        if sidecar_path.name != "manifold.json" and _capture_owner(sidecar_path) == (
            safe_model, capture_sha,
        ):
            return True
    return False


def _remove_unreferenced_capture_groups(
    capture_groups: set[tuple[str, str]],
) -> None:
    """Garbage-collect content-addressed captures only after the last owner."""
    from drowse.base.atomic import artifact_has_live_lease, artifact_lock

    for safe_model, capture_sha in capture_groups:
        cache_dir = models_dir() / safe_model / "manifold_capture"
        # The capture lock spans cache publication through fitted-sidecar
        # publication; scanning and deleting under it means GC cannot race a
        # folder that is about to establish ownership.  A fit still consuming
        # mapped row shards holds a process lease; a later pass reaps it.
        with artifact_lock(cache_dir / capture_sha):
            if artifact_has_live_lease(cache_dir / capture_sha):
                continue
            if _capture_group_is_referenced(safe_model, capture_sha):
                continue
            for cached in cache_dir.glob(f"{capture_sha}.*"):
                cached.unlink()


def _capture_groups_for_models(safe_models: set[str]) -> set[tuple[str, str]]:
    """Every cache stem of the models touched by a destructive operation.

    Independent of fitted-sidecar readability: clear/rm are repair surfaces,
    so an orphaned tensor or corrupt sidecar must not make its capture
    permanently undiscoverable.  Ownership is checked again, under the stem
    lock, immediately before deletion.
    """
    groups: set[tuple[str, str]] = set()
    for safe_model in safe_models:
        try:
            cached_paths = list((models_dir() / safe_model / "manifold_capture").iterdir())
        except OSError:
            continue
        for cached in cached_paths:
            stem = cached.name.split(".", 1)[0]
            if is_sha256_digest(stem):
                groups.add((safe_model, stem))
    return groups


def _delete_fitted_files(
    files: list[Path], manifest_names: list[str], *, scoped_model: str | None = None,
) -> set[tuple[str, str]]:
    """Unlink ``files`` (pair locks held by the caller); return the GC set.

    ``manifest_names`` are the selected ``files`` entries — after a crash or a
    manual deletion they are the only surviving model hint.  ``scoped_model``
    (a safe model id) is the model a scoped clear names: its captures are
    swept even when no fitted file or manifest entry survives, which is
    exactly what a fit killed after capture publication leaves behind.
    """
    affected_models = {
        parsed[0]
        for name in [*manifest_names, *(path.name for path in files)]
        if (parsed := fitted_name_selected(name)) is not None
    }
    if scoped_model is not None:
        affected_models.add(scoped_model)
    capture_groups = {
        owner for path in files
        if path.suffix == ".json" and (owner := _capture_owner(path)) is not None
    }
    for path in files:
        path.unlink(missing_ok=True)
    return capture_groups | _capture_groups_for_models(affected_models)


# ============================================================ rm / clear ===


def clear_manifold_tensors(
    namespace: str, name: str, model_scope: Optional[str] = None, *, variant: str = "all",
) -> int:
    """Delete a manifold's selected fitted pairs, keeping its corpus.

    ``variant`` selects ``raw`` / ``sae`` / ``from`` / ``all`` tensors and
    ``model_scope`` (a raw model id) one model's.  Orphaned halves are cleared
    too — this is the repair path — and only the selected ``files`` entries
    are dropped, so an unrelated corrupt pair is never re-blessed.  Bumps the
    selected ``fit_epochs`` key so an in-flight fit of that target cannot
    publish over the clear.  Returns the number of files deleted.
    """
    folder = manifold_dir(namespace, name)
    with _locked_manifest(folder):
        if not (folder / "manifold.json").exists():
            raise ManifoldNotFoundError(f"manifold {namespace}/{name} not found at {folder}")
        manifest = read_manifest(folder)
        if manifest.get("fit_mode") == "baked":
            raise BakedManifoldError(
                f"manifold {namespace}/{name} is baked (corpus-less, cannot "
                f"re-fit); clearing its tensors is irreversible — use "
                f"`manifold rm {namespace}/{name}` to delete it"
            )
        manifest_files = manifest.get("files", {})
        if not isinstance(manifest_files, dict):
            raise ManifoldFormatError("manifold files manifest must be an object")
        selected = [
            entry for entry in manifest_files
            if fitted_name_selected(entry, variant=variant, model_scope=model_scope)
        ]
        files = fitted_pair_files(folder, variant=variant, model_scope=model_scope)
        scoped_model = safe_model_id(model_scope) if model_scope is not None else None
        with locked_manifold_pairs(files):
            capture_groups = _delete_fitted_files(files, selected, scoped_model=scoped_model)
            manifest["files"] = {
                entry: digest for entry, digest in manifest_files.items()
                if entry not in selected
            }
            epochs = manifest.get("fit_epochs", {})
            epochs = dict(epochs) if isinstance(epochs, dict) else {}
            epoch_key = f"{scoped_model or '*'}:{variant}"
            epochs[epoch_key] = int(epochs.get(epoch_key, 0)) + 1
            manifest["fit_epochs"] = epochs
            write_json_atomic(folder / "manifold.json", manifest)
    _remove_unreferenced_capture_groups(capture_groups)
    invalidate_selector_index()
    return len(files)


def remove_manifold_folder(namespace: str, name: str) -> dict[str, Any]:
    """Remove a whole manifold folder (``rm``).

    A bundled manifold (``default/`` or ``source == "bundled"``) re-materializes
    on the next session start; ``rematerializes_on_restart`` lets a caller say
    so.  Returns ``{namespace, name, source, removed, rematerializes_on_restart}``.
    """
    folder = manifold_dir(namespace, name)
    with _locked_manifest(folder):
        if not (folder / "manifold.json").exists():
            raise ManifoldNotFoundError(f"manifold {namespace}/{name} not found at {folder}")
        try:
            manifest = read_manifest(folder)
        except ManifoldFormatError:
            manifest = {}
        source = manifest.get("source")
        if not isinstance(source, str) or not source:
            source = "bundled" if namespace == "default" else "local"
        manifest_files = manifest.get("files", {})
        names = list(manifest_files) if isinstance(manifest_files, dict) else []
        files = fitted_pair_files(folder)
        with locked_manifold_pairs(files), staging_area() as area:
            capture_groups = _delete_fitted_files(files, names)
            # One rename takes the folder out of every namespace; the scratch
            # area's cleanup deletes it, so a crash mid-delete never leaves a
            # half-removed manifold behind.
            folder.rename(area / folder.name)
    _remove_unreferenced_capture_groups(capture_groups)
    invalidate_selector_index()
    return {
        "namespace": namespace,
        "name": name,
        "source": source,
        "removed": True,
        "rematerializes_on_restart": namespace == "default" or source == "bundled",
    }


def refresh_manifold(
    namespace: str, name: str, *, model_scope: Optional[str] = None, force: bool = True,
) -> str:
    """Re-pull / re-materialize a manifold from its source tier.

    - ``model_scope`` set: clear just that model's fitted pairs so it re-fits
      from the corpus; returns ``"scoped"``.
    - ``bundled`` (or ``default/``): re-copy the shipped manifest and corpus,
      keeping fitted pairs; returns ``"bundled"``.
    - ``hf://<owner>/<name>[@rev]``: re-pull into the same folder; ``"hf"``.
    - anything else has no upstream; returns ``"skipped"``.

    Each tier takes its own locks in the global manifest-then-pairs order;
    no outer lock is held across the re-pull.
    """
    folder = manifold_dir(namespace, name)
    if not (folder / "manifold.json").exists():
        raise ManifoldNotFoundError(f"manifold {namespace}/{name} not found at {folder}")
    if model_scope is not None:
        clear_manifold_tensors(namespace, name, model_scope, variant="all")
        return "scoped"
    # The manifest alone decides the tier: a damaged folder (say, a deleted
    # node file) is exactly what a refresh must still be able to repair.
    source = read_manifest(folder).get("source")
    source = source if isinstance(source, str) else "local"
    if namespace == "default" or source == "bundled":
        from drowse.store.bundled import refresh_bundled_manifold

        refresh_bundled_manifold(name)
        return "bundled"
    if source.startswith("hf://"):
        from drowse.base.hf import split_revision
        from drowse.store.hub import pull_manifold

        coord, revision = split_revision(source[len("hf://"):])
        pull_manifold(coord, target_folder=folder, force=force, revision=revision)
        return "hf"
    return "skipped"


# ============================================================ cross-model transfer ===
#
# Writes a ``_from-<encoded src>`` variant tensor.  The caller builds the
# per-layer Procrustes alignment (both models loaded — a session/CLI concern);
# this only applies it to a fitted manifold and publishes the result.


@dataclass(frozen=True)
class TransferSourceProof:
    """Immutable source-pair generation selected before alignment work."""

    tensor_name: str
    tensor_sha256: str
    sidecar_sha256: str
    layers: tuple[int, ...]


# Sidecar fields that describe how the *source* fit was produced.  A
# transferred tensor was never captured on the target model, so it carries
# none of the source's checkpoint / capture identity (its own identity is
# ``model_fingerprint`` + the transfer provenance), and a merge's component
# digests name source-model tensors.  Keeping them would make the ``_from-``
# sidecar lie about the weights it belongs to.
_SOURCE_ONLY_FIELDS = (
    "model_source_fingerprint", "capture_sha256", "capture_version",
    "capture_render_sha256", "baseline_prompts_sha256", "context_binding_sha256",
    "components", "bake_policy",
)


def _transfer_paths(folder: Path, from_model: str, to_model: str) -> tuple[Path, Path]:
    return (
        folder / tensor_filename(from_model),
        folder / tensor_filename(to_model, transferred_from=from_model),
    )


def _transfer_preflight_locked(
    folder: Path, *, from_model: str, to_model: str, force: bool,
) -> tuple[Path, Path, dict[str, Any], dict[str, str], bool]:
    """Prove the source pair and classify the target with locks already held."""
    source, target = _transfer_paths(folder, from_model, to_model)
    manifest = read_manifest(folder)
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        raise ManifoldFormatError("manifold files manifest must be an object")
    if not prove_pair(folder, source, files):
        raise ManifoldNotFoundError(
            f"manifold {folder.name!r} has no trusted fit for source model "
            f"{from_model!r} at {source}"
        )
    target_trusted = prove_pair(folder, target, files)
    if target_trusted and not force:
        raise ManifoldExistsError(f"{target} already exists; pass force=True to overwrite")
    return source, target, manifest, dict(files), target_trusted


def preflight_transfer_manifold(
    folder: Path, *, from_model: str, to_model: str, force: bool = False,
) -> TransferSourceProof:
    """Prove transfer viability and return the locked source-fit layer header."""
    folder = Path(folder)
    with _locked_manifest(folder):
        with locked_manifold_pairs(list(_transfer_paths(folder, from_model, to_model))):
            source, _target, _manifest, files, _trusted = _transfer_preflight_locked(
                folder, from_model=from_model, to_model=to_model, force=force,
            )
            layers = fitted_tensor_layers(source)
            if layers is None:
                raise ManifoldFormatError(f"source fit {source} has an unreadable tensor header")
            if not layers:
                raise ManifoldFormatError(f"source fit {source} carries no fitted layer means")
            return TransferSourceProof(
                tensor_name=source.name,
                tensor_sha256=files[source.name],
                sidecar_sha256=files[source.with_suffix(".json").name],
                layers=layers,
            )


def transfer_manifold(
    folder: Path,
    *,
    from_model: str,
    to_model: str,
    alignment: dict[int, Any],
    transfer_quality_estimate: Optional[float] = None,
    source_model_fingerprint: str,
    target_model_fingerprint: str,
    whitener: "Any | None" = None,
    target_layer_means: Mapping[int, torch.Tensor],
    force: bool = False,
    expected_source_proof: TransferSourceProof | None = None,
) -> Path:
    """Map a fitted manifold into another model's space and publish it.

    Reads the source-model fit, maps each layer's affine subspace through the
    supplied factorized alignment (:func:`drowse.geometry.manifold.
    transfer_manifold_subspaces` owns the math and the mandatory target-metric
    share re-bake — a missing or non-covering target ``whitener`` raises
    ``WhitenerError``), and writes the ``_from-`` variant pair plus its
    manifest proof in one locked transaction.  Layers the alignment does not
    cover are dropped.  ``expected_source_proof`` pins the source generation
    the alignment was prepared against.  An unproven (crash-left) target pair
    is repaired by overwriting; a proven one needs ``force``.
    """
    from drowse.geometry.manifold import transfer_manifold_subspaces
    from drowse.store.manifold_tensors import load_manifold, save_manifold

    folder = Path(folder)
    with _locked_manifest(folder):
        with locked_manifold_pairs(list(_transfer_paths(folder, from_model, to_model))):
            src_tensor, out_path, manifest, files, target_trusted = (
                _transfer_preflight_locked(
                    folder, from_model=from_model, to_model=to_model, force=force,
                )
            )
            if expected_source_proof is not None and (
                src_tensor.name,
                files.get(src_tensor.name),
                files.get(src_tensor.with_suffix(".json").name),
            ) != (
                expected_source_proof.tensor_name,
                expected_source_proof.tensor_sha256,
                expected_source_proof.sidecar_sha256,
            ):
                raise ManifoldFormatError(
                    "source manifold fit changed while alignment was prepared; "
                    "retry the transfer against the current source generation"
                )
            if not alignment:
                raise ManifoldFormatError(
                    f"transfer_manifold: alignment map for {from_model!r} → "
                    f"{to_model!r} is empty"
                )
            if not source_model_fingerprint or not target_model_fingerprint:
                raise ManifoldFormatError(
                    "transfer requires proven source and target model fingerprints"
                )
            src = load_manifold(src_tensor)
            if src.metadata.get("model_fingerprint") != source_model_fingerprint:
                raise ManifoldFormatError(
                    "transfer source manifold was fitted for different loaded weights; "
                    "refit the source manifold before transfer"
                )
            try:
                transferred = transfer_manifold_subspaces(
                    src, alignment, whitener=whitener,
                    target_layer_means=target_layer_means,
                    from_model=from_model, to_model=to_model,
                )
            except DrowseError:
                raise
            except ValueError as exc:
                raise ManifoldFormatError(f"transfer_manifold: {exc}") from exc

            metadata: dict[str, object] = dict(src.metadata)
            for key in _SOURCE_ONLY_FIELDS:
                metadata[key] = None
            metadata["method"] = "manifold_procrustes_transfer"
            metadata["source_model_id"] = from_model
            metadata["source_model_fingerprint"] = source_model_fingerprint
            metadata["model_fingerprint"] = target_model_fingerprint
            # The share was re-baked in the target metric; ``subspace_metric``
            # keeps the source value (the basis was selected there and only
            # rotated here).
            metadata["share_metric"] = "mahalanobis"
            if transfer_quality_estimate is not None:
                metadata["transfer_quality_estimate"] = float(transfer_quality_estimate)
            sidecar_path = out_path.with_suffix(".json")
            if not target_trusted:
                out_path.unlink(missing_ok=True)
                sidecar_path.unlink(missing_ok=True)
                files.pop(out_path.name, None)
                files.pop(sidecar_path.name, None)
                manifest["files"] = files
                write_json_atomic(folder / "manifold.json", manifest)
            current = ManifoldFolder.load(folder, verify_manifest=False)
            metadata["nodes_sha256"] = current.nodes_sha256()
            save_manifold(transferred, out_path, metadata)
            current.update_file_hashes(out_path, sidecar_path)
            return out_path


# ============================================================ summary ===


def manifold_fit_summary(sidecar: ManifoldSidecar, stem: str) -> dict[str, Any]:
    """Serialize one fitted tensor's sidecar for inspection surfaces.

    The per-tensor detail block behind ``pack show`` and the inspector: the
    provenance the fit stamped (an ``auto`` fit's resolved geometry and ranked
    topology evidence, the share / subspace metrics).  Empty fields are
    omitted so a pinned fit stays terse.
    """
    entry: dict[str, Any] = {
        "stem": stem,
        "method": sidecar.method,
        "feature_space": sidecar.feature_space,
        "node_count": sidecar.node_count,
        "nodes_sha256": sidecar.nodes_sha256,
        "fit_mode": sidecar.fit_mode,
        "fitted_layers": list(sidecar.fitted_layers),
    }
    for key, value in (
        ("hyperparams", sidecar.hyperparams),
        ("diagnostics", sidecar.diagnostics),
        ("node_spread", sidecar.node_spread_per_layer),
        ("share_metric", sidecar.share_metric),
        ("subspace_metric", sidecar.subspace_metric),
        ("resolved_fit_mode", sidecar.resolved_fit_mode),
        ("topology_winner", sidecar.topology_winner),
        ("topology_candidates", sidecar.topology_candidates),
        ("rbf_smoothing", sidecar.rbf_smoothing_per_layer),
        ("sigma_field", sidecar.sigma_field_per_layer),
    ):
        if value:
            entry[key] = value
    return entry


def manifold_summary(
    folder: Path, *, include_fits: bool = False,
) -> dict[str, Any]:
    """Session-independent summary of a manifold folder.

    The shared serializer behind ``manifold show -j`` and ``GET
    /drowse/v1/manifolds/{ns}/{name}``.  Metadata only: an authored folder
    reports its domain / intrinsic dim / coords; a discover or baked folder
    reports ``{}`` / ``0`` / ``[]`` (its geometry is per-model, in the fitted
    tensor).  ``fitted`` (opt-in, one sidecar read per fit) carries one
    :func:`manifold_fit_summary` block per readable fitted stem.
    """
    folder = Path(folder)
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    if mf.fit_mode == "authored":
        domain = mf.domain
        domain_lbl = domain_label(domain)
        intrinsic = domain_from_spec(domain).intrinsic_dim
    else:
        domain = {}
        domain_lbl = "baked" if mf.fit_mode == "baked" else f"discover-{mf.fit_mode}"
        intrinsic = 0

    tensor_variants: dict[str, list[str]] = {}
    for tensor in sorted(folder.glob("*.safetensors")):
        parsed = fitted_name_selected(tensor.name)
        if parsed is not None:
            tensor_variants.setdefault(parsed[0], []).append(parsed[1] or "raw")

    out: dict[str, Any] = {
        "namespace": folder.parent.name,
        "name": mf.name,
        "description": mf.description,
        "source": mf.source,
        "tags": list(mf.tags),
        "template_ref": mf.template_ref,
        "fit_mode": mf.fit_mode,
        "is_discover": mf.is_discover,
        "domain": domain,
        "domain_label": domain_lbl,
        "intrinsic_dim": intrinsic,
        "min_nodes": min_nodes(intrinsic) if intrinsic else None,
        "node_count": len(mf.node_labels),
        "node_labels": list(mf.node_labels),
        "node_coords": [list(c) for c in mf.node_coords],
        "node_roles": mf.roles,
        "node_kinds": mf.kinds,
        "hyperparams": dict(mf.hyperparams),
        "fitted_models": list(tensor_variants),
        "tensor_variants": tensor_variants,
    }
    if include_fits:
        out["fitted"] = [
            manifold_fit_summary(mf.sidecar(stem), stem) for stem in mf.tensor_models()
        ]
    return out
