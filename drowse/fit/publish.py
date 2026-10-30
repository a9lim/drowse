"""Folder transactions around a manifold fit: snapshot in, compare-and-swap out.

A fit computes for minutes against one manifold folder that readers keep
using and authors may edit.  This module owns the two short folder
transactions that bracket that compute:

- :func:`take_authoring_snapshot` — under the folder's manifest lock: repair an
  interrupted one-sided tensor pair, merge the request's discover overrides
  into ``manifold.json``, re-materialize a drifted templated corpus, and record
  the exact authoring revision the fit runs against;
- :func:`publish_fit_if_current` — under the same lock again: publish the
  fitted tensor/sidecar pair only if that revision is still current, else
  raise :class:`ManifoldAuthoringChangedError` (409, the caller retries).

Between them, :func:`load_published_fit` reads the currently published pair
for the fitted-tensor fast path under the pair lock alone, and
:func:`fit_target_lock` single-flights fits of one target.  The fit pipeline
itself (:mod:`drowse.fit.pipeline`) never touches folder files directly.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from drowse.base.atomic import artifact_lock, write_json_atomic
from drowse.base.paths import parse_tensor_filename
from drowse.errors import DrowseError
from drowse.store.manifold_folder import (
    ManifoldFolder,
    _locked_manifest,
    manifold_pair_lock,
    write_node_corpus,
)
from drowse.store.manifold_schema import ManifoldSidecar, sanitize_hyperparams

if TYPE_CHECKING:
    from drowse.geometry.manifold import Manifold


class ManifoldAuthoringChangedError(RuntimeError, DrowseError):
    """The authoring inputs changed while a target was being fitted.

    Semantically the conflict :class:`~drowse.session.session.ConcurrentExtractionError`
    models, reached from the other direction: the fit is fine, the folder moved
    under it.  Reports 409 so the caller retries against the current revision
    instead of reading it as a server fault.  ``RuntimeError`` stays in the
    MRO, so existing catch sites are unaffected.
    """

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or self.__class__.__name__)


@dataclass(frozen=True)
class AuthoringSnapshot:
    """One exact authoring revision of a folder, as a fit consumes it.

    ``baseline_prompts`` are the elicitation prompts every node response is
    paired against: a templated manifold's resolved multi-turn contexts, else
    the shared baseline prompt set.  ``revision`` digests everything a fit of
    this target depends on — folder identity, the target's fit epochs, the
    corpus/geometry/hyperparameter hash, and the prompts — and is what
    publication compares against.
    """

    manifold: ManifoldFolder
    nodes_sha256: str
    revision: str
    node_groups: list[tuple[str, list[str]]]
    baseline_prompts: list[str | list[dict[str, str]]]


@contextmanager
def fit_target_lock(folder: Path, target_path: Path) -> Iterator[None]:
    """Single-flight fits of one folder target (model × variant).

    Keyed outside the removable folder: ``rm`` + recreate must not mint a fresh
    lock inode while a stale fit still owns the old one.  Distinct targets in
    one folder fit concurrently.
    """
    key = hashlib.sha256(
        f"{Path(folder).resolve(strict=False)}\0{target_path.name}".encode()
    ).hexdigest()
    with artifact_lock(Path(folder).parent / f".drowse-fit-{key}"):
        yield


def take_authoring_snapshot(
    folder: Path,
    target_path: Path,
    *,
    fit_mode: str | None,
    hyperparams: Mapping[str, object] | None,
    default_prompts: Callable[[], Sequence[str]],
) -> AuthoringSnapshot:
    """Prepare the folder for a fit of ``target_path`` and snapshot it.

    ``fit_mode`` / ``hyperparams`` are this request's discover overrides; they
    merge against the manifest re-read under the lock, so two concurrent
    override requests cannot lose an update before either derives its cache
    key.  ``default_prompts`` supplies the shared baseline prompt set for a
    non-templated folder.
    """
    with _locked_manifest(folder):
        with manifold_pair_lock(target_path):
            _repair_incomplete_target_locked(folder, target_path)
        _apply_fit_overrides_locked(
            folder, fit_mode=fit_mode, hyperparams=hyperparams,
        )
        return _authoring_snapshot_locked(folder, target_path, default_prompts)


def assert_authoring_revision(
    folder: Path,
    target_path: Path,
    expected_revision: str,
    default_prompts: Callable[[], Sequence[str]],
) -> None:
    """Raise unless ``expected_revision`` is still the folder's current one."""
    with _locked_manifest(folder):
        current = _authoring_snapshot_locked(folder, target_path, default_prompts)
    _check_revision(expected_revision, current.revision)


def publish_fit_if_current(
    folder: Path,
    *,
    expected_revision: str,
    manifold: "Manifold",
    tensor_path: Path,
    metadata: dict[str, Any],
    default_prompts: Callable[[], Sequence[str]],
) -> None:
    """CAS-publish a fitted pair while the authoring revision is unchanged."""
    from drowse.store.manifold_tensors import save_manifold

    with _locked_manifest(folder):
        current = _authoring_snapshot_locked(folder, tensor_path, default_prompts)
        _check_revision(expected_revision, current.revision)
        save_manifold(manifold, tensor_path, metadata)
        current.manifold.update_file_hashes(
            tensor_path, tensor_path.with_suffix(".json"),
        )


def load_published_fit(
    folder: Path,
    tensor_path: Path,
    accept: Callable[[ManifoldSidecar], bool],
) -> "Manifold | None":
    """The currently published pair for ``tensor_path`` if ``accept`` takes it.

    Clear/rm/refresh take the folder lock and then this same stable pair lock.
    The fast path takes only the pair lock: it proves the *current* manifest
    pair, reads the sidecar, and loads the tensor in one transaction, avoiding
    both a proof->load deletion window and a folder/pair lock inversion.  Any
    unreadable or unproven pair is simply not a hit.
    """
    from drowse.base.integrity import verify_integrity
    from drowse.store.manifold_tensors import _load_manifold_locked

    sidecar_path = tensor_path.with_suffix(".json")
    with manifold_pair_lock(tensor_path):
        try:
            with open(folder / "manifold.json") as handle:
                files = json.load(handle).get("files", {})
            expected = {
                name: files[name]
                for name in (tensor_path.name, sidecar_path.name)
                if isinstance(files, dict) and name in files
            }
            if (
                len(expected) != 2
                or not tensor_path.exists()
                or not sidecar_path.exists()
                or not verify_integrity(folder, expected)[0]
            ):
                return None
            sidecar = ManifoldSidecar.load(sidecar_path)
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return None
        if not accept(sidecar):
            return None
        return _load_manifold_locked(tensor_path, verify_manifest=False)


def _check_revision(expected: str, actual: str) -> None:
    if actual != expected:
        raise ManifoldAuthoringChangedError(
            f"manifold authoring changed during fit ({expected[:12]}"
            f" -> {actual[:12]}); retry against the current revision"
        )


def _apply_fit_overrides_locked(
    folder: Path,
    *,
    fit_mode: str | None,
    hyperparams: Mapping[str, object] | None,
) -> None:
    """Merge discover-fit overrides into ``manifold.json`` (lock held)."""
    if fit_mode is None and hyperparams is None:
        return
    mf = ManifoldFolder.load(folder, verify_manifest=False)
    if not mf.is_discover:
        raise ValueError(
            "fit_mode/hyperparams overrides are discover-mode only; "
            f"{folder} is authored"
        )
    requested_mode = fit_mode or mf.fit_mode
    if requested_mode not in {"pca", "spectral", "auto"}:
        raise ValueError(f"invalid discover fit mode {requested_mode!r}")
    # Hyperparameters belong to their discriminated fit mode. A mode change
    # starts a fresh parameter set; carrying the previous mode's keys would
    # either be ignored or violate the exact current schema.
    requested_hyperparams = (
        {} if fit_mode is not None and fit_mode != mf.fit_mode
        else dict(mf.hyperparams)
    )
    if hyperparams is not None:
        requested_hyperparams.update(hyperparams)
    requested_hyperparams = sanitize_hyperparams(
        requested_mode, requested_hyperparams,
    )
    if (
        requested_mode == mf.fit_mode
        and requested_hyperparams == mf.hyperparams
    ):
        return

    with open(folder / "manifold.json") as handle:
        data = json.load(handle)
    data["fit_mode"] = requested_mode
    data["hyperparams"] = requested_hyperparams
    # Discover nodes are label-only.  Preserve role/kind provenance while
    # stripping any authored geometry left by a mode transition.
    data["nodes"] = [
        {
            "label": label,
            "role": mf.node_roles[idx],
            "kind": mf.node_kinds[idx],
        }
        for idx, label in enumerate(mf.node_labels)
    ]
    data.pop("domain", None)
    write_json_atomic(folder / "manifold.json", data)


def _repair_incomplete_target_locked(folder: Path, tensor_path: Path) -> None:
    """Remove an interrupted one-sided fitted pair and its stale proofs."""
    sidecar_path = tensor_path.with_suffix(".json")
    if tensor_path.exists() == sidecar_path.exists():
        return
    tensor_path.unlink(missing_ok=True)
    sidecar_path.unlink(missing_ok=True)
    manifest_path = folder / "manifold.json"
    with open(manifest_path) as handle:
        payload = json.load(handle)
    files = payload.get("files", {})
    if isinstance(files, dict):
        files = dict(files)
        files.pop(tensor_path.name, None)
        files.pop(sidecar_path.name, None)
        payload["files"] = files
        write_json_atomic(manifest_path, payload)


def _authoring_snapshot_locked(
    folder: Path,
    target_path: Path,
    default_prompts: Callable[[], Sequence[str]],
) -> AuthoringSnapshot:
    """Materialize template inputs and snapshot one exact authoring revision."""
    manifest_path = folder / "manifold.json"
    with open(manifest_path) as handle:
        identity_payload = json.load(handle)
    artifact_id = identity_payload.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        artifact_id = uuid.uuid4().hex
        identity_payload["artifact_id"] = artifact_id
        identity_payload.setdefault("fit_epochs", {})
        write_json_atomic(manifest_path, identity_payload)

    parsed = parse_tensor_filename(target_path.name)
    if parsed is None:
        raise ValueError(f"invalid manifold tensor target {target_path.name!r}")
    safe_model, raw_variant = parsed
    variant = "raw" if raw_variant is None else raw_variant.split("-", 1)[0]
    fit_epochs_raw = identity_payload.get("fit_epochs", {})
    fit_epochs = fit_epochs_raw if isinstance(fit_epochs_raw, dict) else {}
    epoch_keys = (
        "*:all", f"*:{variant}", f"{safe_model}:all",
        f"{safe_model}:{variant}", target_path.name,
    )
    target_epochs = {key: int(fit_epochs.get(key, 0)) for key in epoch_keys}

    mf = ManifoldFolder.load(folder, verify_manifest=False)
    template_sha: str | None = None
    baseline: list[str | list[dict[str, str]]]
    if mf.template_ref is not None:
        from drowse.store.templates import resolve_template

        tmpl = resolve_template(mf.template_ref)
        template_sha = tmpl.sha256()
        expected_corpora = tmpl.node_corpora()
        if dict(mf.node_groups()) != expected_corpora:
            write_node_corpus(folder, list(expected_corpora.items()))
            mf.node_labels = list(expected_corpora)
            mf.node_roles = [None] * len(mf.node_labels)
            mf.node_kinds = [None] * len(mf.node_labels)
            mf.write_metadata()
            mf = ManifoldFolder.load(folder, verify_manifest=False)
        baseline = [context.messages() for context in tmpl.contexts]
    else:
        baseline = list(default_prompts())
    nodes_sha = mf.nodes_sha256(resolved_template_sha256=template_sha)
    revision = hashlib.sha256(json.dumps(
        {
            "artifact_id": artifact_id,
            "target_epochs": target_epochs,
            "nodes_sha256": nodes_sha,
            "baseline_prompts": baseline,
        },
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    return AuthoringSnapshot(
        manifold=mf,
        nodes_sha256=nodes_sha,
        revision=revision,
        node_groups=mf.node_groups(),
        baseline_prompts=baseline,
    )
