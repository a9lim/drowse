"""A manifold folder on disk: corpus, integrity, fitted pairs, and locks.

```
~/.drowse/manifolds/<ns>/<name>/
  manifold.json               # the manifest (schema: drowse.store.manifold_schema)
  nodes/NN_<label>.json       # one JSON list of corpus strings per node
  <safe_model>.safetensors    # fitted per-model subspace (+ .json sidecar)
  <safe_model>_sae-<rel>.safetensors   # SAE-feature-space variant
  <safe_model>_from-<src>.safetensors  # cross-model transfer variant
```

The schema of ``manifold.json`` and of the fitted sidecar lives in
:mod:`drowse.store.manifold_schema`; this module owns the filesystem side —
:class:`ManifoldFolder`, the manifest read-modify-write CAS, the fitted-pair
scan / proof primitives, and the lock order every mutator shares (manifest
lock, then stable pair locks in tensor-path order).  Tensor save/load lives in
:mod:`drowse.store.manifold_tensors`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from drowse.base.atomic import artifact_lock, write_json_atomic
from drowse.base.integrity import (
    canonical_json,
    hash_file,
    is_sha256_digest,
    verify_integrity,
)
from drowse.base.paths import parse_tensor_filename, safe_model_id
from drowse.base.staging import promote, recover_backup, staging_area
from drowse.store.manifold_schema import (
    FIT_MODES_DISCOVER,
    MANIFOLD_FORMAT_VERSION,
    ManifestNode,
    ManifestSpec,
    ManifoldFormatError,
    ManifoldSidecar,
    load_manifest_data,
    manifest_payload,
    node_filename,
    parse_manifest,
    sidecar_is_current,
)

_log = logging.getLogger(__name__)


# ============================================================== locks ===


@contextmanager
def _locked_manifest(folder: Path) -> Iterator[None]:
    """Serialize cross-process manifest read-modify-write for one folder.

    The lock file lives beside the folder (``<ns>/<name>.manifest``) so a
    whole-folder swap or ``rm`` cannot unlink a held lock inode.
    """
    with artifact_lock(folder.parent / f"{folder.name}.manifest"):
        yield


def manifold_pair_lock_path(tensor_path: Path) -> Path:
    """Stable, bounded logical lock path for one fitted tensor pair.

    The lock lives under the manifold namespace rather than inside the
    removable manifold folder, so ``rm`` and stage-swap refreshes cannot unlink
    a held lock inode and let a later process acquire a fresh inode for the
    same logical pair.  The digest bounds the filename for long tensor names.
    """
    tensor_path = Path(tensor_path).expanduser().resolve(strict=False)
    identity = f"{tensor_path.parent}\0{tensor_path.name}".encode()
    digest = hashlib.sha256(identity).hexdigest()
    return tensor_path.parent.parent / f".drowse-pair-{digest}"


@contextmanager
def manifold_pair_lock(tensor_path: Path) -> Iterator[None]:
    """Lock a fitted tensor/sidecar pair using stable external identity."""
    with artifact_lock(manifold_pair_lock_path(tensor_path)):
        yield


@contextmanager
def locked_manifold_pairs(paths: list[Path]) -> Iterator[None]:
    """Acquire the pair locks for ``paths`` in one deterministic global order."""
    with ExitStack() as stack:
        for tensor_path in sorted({pair_tensor_path(path) for path in paths}):
            stack.enter_context(manifold_pair_lock(tensor_path))
        yield


@contextmanager
def destructive_manifold_folder_transaction(folder: Path) -> Iterator[None]:
    """Quiesce fitted readers before a whole-folder reset or stage swap.

    Global mutation order is manifest then stable pair locks, with pair locks in
    deterministic tensor-path order.  The pair locks live outside the removable
    folder, so holding this context across ``rmtree``/rename/recreation prevents
    a new inode from bypassing a reader that still owns the old logical pair.
    The manifest lock is re-entrant.
    """
    folder = Path(folder)
    with _locked_manifest(folder):
        with locked_manifold_pairs(manifold_folder_tensor_paths(folder)):
            yield


def reset_manifold_folder(folder: Path) -> None:
    """Remove ``folder`` only after all extant fitted pairs are quiescent."""
    folder = Path(folder)
    with destructive_manifold_folder_transaction(folder):
        if folder.exists():
            shutil.rmtree(folder)


@contextmanager
def locked_manifold_folder(folder: Path) -> Iterator[None]:
    """Take ``folder``'s manifest lock and recover a crash-left swap backup.

    Every whole-folder writer enters through here, so a swap that died between
    parking the old folder and promoting the new one is restored before the
    writer's conflict policy or drift check reads the destination.
    """
    folder = Path(folder)
    with _locked_manifest(folder):
        recover_backup(folder)
        yield


def replace_manifold_folder(folder: Path, build: Callable[[Path], None]) -> None:
    """Replace ``folder`` wholesale with the tree ``build`` writes.

    ``build`` fills a fresh directory in a private staging area outside every
    namespace (:func:`drowse.base.staging.staging_area`), which is promoted
    over ``folder`` in one swap, so readers see the old folder or the complete
    new one and a crash never leaves a half-built manifold discoverable.  The
    swap runs under the destructive folder transaction, quiescing readers of
    every extant fitted pair.  Callers hold :func:`locked_manifold_folder`.
    """
    folder = Path(folder)
    with destructive_manifold_folder_transaction(folder), staging_area() as area:
        stage = area / folder.name
        stage.mkdir()
        build(stage)
        promote(stage, folder)


# ====================================================== fitted pairs ===


def pair_tensor_path(path: Path) -> Path:
    """Canonicalize either half of a fitted pair to its tensor path."""
    path = Path(path)
    return path.with_suffix(".safetensors") if path.suffix == ".json" else path


def tensor_variant_matches(variant: str | None, filter_: str) -> bool:
    """Whether a parsed tensor variant passes a ``raw|sae|from|all`` filter.

    ``variant`` is what :func:`drowse.base.paths.parse_tensor_filename`
    returns: ``None`` for the canonical fit, ``"sae-<rel>"`` or
    ``"from-<src>"`` for the variants.
    """
    if filter_ == "all":
        return True
    if filter_ == "raw":
        return variant is None
    if filter_ in {"sae", "from"}:
        return variant is not None and variant.startswith(f"{filter_}-")
    return False


def fitted_name_selected(
    filename: str, *, variant: str = "all", model_scope: str | None = None,
) -> tuple[str, str | None] | None:
    """Parse either half's filename and apply the variant / model filters.

    Returns the parsed ``(safe_model, variant)`` when selected, else ``None``.
    ``model_scope`` is a raw model id.
    """
    parsed = parse_tensor_filename(pair_tensor_path(Path(filename)).name)
    if parsed is None:
        return None
    if model_scope is not None and parsed[0] != safe_model_id(model_scope):
        return None
    return parsed if tensor_variant_matches(parsed[1], variant) else None


def manifold_folder_tensor_paths(
    folder: Path, *, variant: str = "all", model_scope: str | None = None,
) -> list[Path]:
    """Canonical tensor paths for every fitted or orphaned pair in ``folder``.

    Scans both halves independently (an interrupted publication or a corrupt
    folder may hold only one) and canonicalizes sidecars back to their tensor
    path, so each logical pair appears exactly once.  The manifest's ``files``
    map is deliberately not consulted: destructive operations are also the
    repair path for pairs that never acquired a proof.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    tensors: set[Path] = set()
    for candidate in folder.iterdir():
        if candidate.name == "manifold.json" or candidate.suffix not in {".json", ".safetensors"}:
            continue
        if fitted_name_selected(candidate.name, variant=variant, model_scope=model_scope):
            tensors.add(pair_tensor_path(candidate))
    return sorted(tensors)


def fitted_pair_files(
    folder: Path, *, variant: str = "all", model_scope: str | None = None,
) -> list[Path]:
    """The existing halves (tensor and/or sidecar) of every selected pair."""
    out: list[Path] = []
    for tensor in manifold_folder_tensor_paths(folder, variant=variant, model_scope=model_scope):
        out.extend(p for p in (tensor, tensor.with_suffix(".json")) if p.exists())
    return out


def prove_pair(folder: Path, tensor_path: Path, files: dict[str, Any]) -> bool:
    """Whether ``files`` carries valid digests that match both halves on disk.

    The one fitted-pair proof.  A pair missing either digest is unproven
    (``False``); a present but malformed digest is a corrupt manifest and
    raises :class:`ManifoldFormatError`.
    """
    names = (tensor_path.name, tensor_path.with_suffix(".json").name)
    if any(name not in files for name in names):
        return False
    for name in names:
        if not is_sha256_digest(files[name]):
            raise ManifoldFormatError(
                f"manifold integrity manifest has invalid sha256 for {name!r}"
            )
    return verify_integrity(Path(folder), {name: files[name] for name in names})[0]


def read_proven_sidecar(folder: Path, tensor_path: Path) -> ManifoldSidecar | None:
    """Read one pair's sidecar only once the current manifest proves both files.

    Takes the stable pair lock, proves the *current* manifest's digests for
    exactly this pair, then reads.  An unproven, half-published, malformed, or
    foreign pair reads as no sidecar at all.
    """
    folder = Path(folder)
    with manifold_pair_lock(tensor_path):
        try:
            files = read_manifest(folder).get("files", {})
            if not isinstance(files, dict) or not prove_pair(folder, tensor_path, files):
                return None
            return ManifoldSidecar.load(tensor_path.with_suffix(".json"))
        except (OSError, ManifoldFormatError):
            return None


def hash_manifold_files(folder: Path) -> dict[str, str]:
    """``{name: sha256}`` for every top-level fitted file in ``folder``.

    Skips ``manifold.json`` and the ``nodes/`` corpus: editing the corpus is
    the normal re-fit trigger (tracked by the per-tensor ``nodes_sha256``), so
    it must never fail an integrity check.
    """
    return {
        entry.name: hash_file(entry)
        for entry in sorted(Path(folder).iterdir())
        if entry.is_file() and entry.name != "manifold.json"
    }


def read_manifest(folder: Path) -> dict[str, Any]:
    """Read ``<folder>/manifold.json`` as a version-checked JSON object.

    The raw-dict read for repair and CAS paths that must not demand a fully
    valid folder; :meth:`ManifoldFolder.load` is the validating read.
    """
    return load_manifest_data(Path(folder) / "manifold.json")


def write_node_corpus(folder: Path, corpora: list[tuple[str, list[str]]]) -> None:
    """(Re)write ``nodes/`` from ``[(label, statements), ...]`` via a staged swap.

    The new corpus is written to ``nodes.staging/`` in full and then swapped
    in, so an IO error mid-corpus leaves the existing ``nodes/`` untouched.
    """
    nodes_dir = Path(folder) / "nodes"
    staging = Path(folder) / "nodes.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for idx, (label, statements) in enumerate(corpora):
        write_json_atomic(staging / node_filename(idx, label), [str(s) for s in statements])
    if nodes_dir.exists():
        shutil.rmtree(nodes_dir)
    staging.rename(nodes_dir)


# ============================================================ folder ===


class _StaleFit(ManifoldFormatError):
    """A fitted sidecar from an older format: a cache miss, not corruption."""



@dataclass
class ManifoldFolder:
    """A manifold artifact folder: the parsed manifest plus its corpus and fits.

    ``fit_mode == "authored"`` folders carry a ``domain`` spec and per-node
    ``node_coords``; discover folders (``pca`` / ``spectral`` / ``auto``)
    carry ``hyperparams`` and derive coordinates per model at fit time, so
    their ``domain`` / ``node_coords`` are empty until a fitted sidecar
    supplies them; ``baked`` folders carry a display ``domain`` and no corpus.

    Fitted sidecars load lazily: discovery and selector indexing never parse
    them, and one malformed sidecar makes only that fit unavailable
    (:meth:`tensor_models` omits it, :meth:`sidecar` raises) instead of hiding
    the whole manifold.
    """

    folder: Path
    name: str
    description: str
    domain: dict[str, Any]
    node_labels: list[str]
    node_coords: list[list[float]]
    files: dict[str, str]
    fit_mode: str = "authored"
    hyperparams: dict[str, Any] = field(default_factory=dict)
    # Provenance tier: ``"local"``; ``"bundled"`` (shipped under
    # ``drowse/data/manifolds/``, materialized into ``default/``); or
    # ``"hf://<owner>/<name>[@rev]"`` (pulled from the hub).  ``refresh``
    # dispatches on it.
    source: str = "local"
    # Per-node assistant-role substitution and generation kind, aligned with
    # ``node_labels``; ``None`` = standard assistant baseline / unspecified.
    node_roles: list[str | None] = field(default_factory=list)
    node_kinds: list[str | None] = field(default_factory=list)
    # Category tags (``tag:`` selectors and the bundled probe roster).
    tags: list[str] = field(default_factory=list)
    # ``<ns>/<name>`` of the template a templated discover manifold derives its
    # corpus from; the fit resolves its multi-turn contexts as elicitation
    # prefixes and folds its content hash into ``nodes_sha256``.
    template_ref: str | None = None
    _tensor_stems: tuple[str, ...] = ()
    _sidecars: dict[str, ManifoldSidecar | ManifoldFormatError] = field(
        default_factory=dict, repr=False,
    )

    @property
    def is_discover(self) -> bool:
        """True when this folder's coords are derived per-model rather than authored."""
        return self.fit_mode in FIT_MODES_DISCOVER

    @classmethod
    def from_spec(cls, folder: Path, spec: ManifestSpec) -> "ManifoldFolder":
        return cls(
            folder=Path(folder),
            name=spec.name,
            description=spec.description,
            domain=dict(spec.domain),
            node_labels=spec.labels,
            node_coords=[list(node.coords) for node in spec.nodes if node.coords is not None],
            files=dict(spec.files),
            fit_mode=spec.fit_mode,
            hyperparams=dict(spec.hyperparams),
            source=spec.source,
            node_roles=[node.role for node in spec.nodes],
            node_kinds=[node.kind for node in spec.nodes],
            tags=list(spec.tags),
            template_ref=spec.template_ref,
        )

    @classmethod
    def load(cls, folder: Path, *, verify_manifest: bool = True) -> "ManifoldFolder":
        """Parse and structurally validate a manifold folder.

        Runtime/install/push callers keep the default full integrity walk over
        ``files``; metadata-only callers (inventory, authoring, lifecycle
        routing, fit) pass ``verify_manifest=False`` and prove only the pair
        they consume.  Every declared node corpus file must exist (except for
        a baked manifold, which has no corpus but must carry a fitted pair).
        """
        folder = Path(folder)
        meta_path = folder / "manifold.json"
        if not meta_path.exists():
            raise ManifoldFormatError(f"no manifold.json in {folder}")
        spec = parse_manifest(load_manifest_data(meta_path), location=str(meta_path))
        if spec.files and verify_manifest:
            ok, bad = verify_integrity(folder, spec.files)
            if not ok:
                raise ManifoldFormatError(
                    f"manifold integrity check failed in {folder}: tampered/missing {bad}"
                )
        inst = cls.from_spec(folder, spec)
        if spec.fit_mode != "baked":
            for idx in range(len(inst.node_labels)):
                path = inst.node_path(idx)
                if not path.exists():
                    raise ManifoldFormatError(
                        f"manifold {spec.name!r} missing node corpus file {path}"
                    )
        tensors = [t for t in manifold_folder_tensor_paths(folder) if t.exists()]
        inst._tensor_stems = tuple(
            tensor.stem for tensor in tensors if tensor.with_suffix(".json").exists()
        )
        if spec.fit_mode == "baked" and not inst._tensor_stems:
            raise ManifoldFormatError(
                f"baked manifold {spec.name!r} has no fitted tensor — a corpus-less "
                f"manifold carries its geometry only in the per-model tensor"
            )
        if verify_manifest:
            # The strict read (install, push, runtime): every fitted pair must
            # be whole and every current-format sidecar valid.
            for tensor in tensors:
                if tensor.stem not in inst._tensor_stems:
                    raise ManifoldFormatError(
                        f"manifold {spec.name!r} tensor {tensor.name} has no sidecar"
                    )
                loaded = inst._load_sidecar(tensor.stem)
                if isinstance(loaded, ManifoldFormatError) and not isinstance(loaded, _StaleFit):
                    raise loaded
        return inst

    # -- manifest ------------------------------------------------------------

    def spec(self) -> ManifestSpec:
        """This folder's authored state as a :class:`ManifestSpec`."""
        roles, kinds = self.roles, self.kinds
        authored = self.fit_mode == "authored"
        return ManifestSpec(
            name=self.name,
            description=self.description,
            fit_mode=self.fit_mode,
            nodes=tuple(
                ManifestNode(
                    label=label, role=roles[idx], kind=kinds[idx],
                    coords=tuple(self.node_coords[idx]) if authored else None,
                )
                for idx, label in enumerate(self.node_labels)
            ),
            domain=dict(self.domain),
            hyperparams=dict(self.hyperparams),
            files=dict(self.files),
            source=self.source or "local",
            tags=tuple(self.tags),
            template_ref=self.template_ref,
        )

    def write_metadata(self, *, files: Optional[dict[str, str]] = None) -> None:
        """Rewrite the authored manifest while preserving trusted fitted hashes.

        Fitted writers must call :meth:`update_file_hashes` with their exact
        outputs; a metadata-only rewrite never scans and blesses unrelated
        files.  The locked latest manifest supplies ``files`` (and the local
        ``artifact_id`` / ``fit_epochs`` state) unless ``files`` is explicit.
        """
        with _locked_manifest(self.folder):
            latest = read_manifest(self.folder)
            if files is None:
                files = latest.get("files", {})
                if not isinstance(files, dict):
                    raise ManifoldFormatError("manifold files manifest is not an object")
            self.files = dict(files)
            spec = self.spec()
            write_json_atomic(self.folder / "manifold.json", manifest_payload(
                name=spec.name, description=spec.description, fit_mode=spec.fit_mode,
                nodes=spec.nodes, domain=spec.domain, hyperparams=spec.hyperparams,
                files=spec.files, source=spec.source, tags=spec.tags,
                template_ref=spec.template_ref,
                artifact_id=latest.get("artifact_id"),
                fit_epochs=latest.get("fit_epochs"),
            ))

    def update_file_hashes(self, *paths: Path) -> None:
        """Record the digests of newly written fitted files in the manifest CAS.

        Only the named paths are hashed: implicitly hashing other pairs here
        would launder an interrupted or tampered artifact merely because
        another model was fitted.
        """
        resolved_paths = [Path(path) for path in paths]
        for resolved in resolved_paths:
            if resolved.parent != self.folder or not resolved.is_file():
                raise ValueError(
                    f"manifest update path must be a fitted file in {self.folder}: {resolved}"
                )
        with _locked_manifest(self.folder):
            payload = read_manifest(self.folder)
            latest = payload.get("files", {})
            if not isinstance(latest, dict):
                raise ManifoldFormatError("manifold files manifest is not an object")
            files = dict(latest)
            for resolved in resolved_paths:
                files[resolved.name] = hash_file(resolved)
            payload["format_version"] = MANIFOLD_FORMAT_VERSION
            payload["files"] = files
            write_json_atomic(self.folder / "manifold.json", payload)
            self.files = files

    def _padded(self, values: list[str | None]) -> list[str | None]:
        if len(values) == len(self.node_labels):
            return list(values)
        return [None] * len(self.node_labels)

    @property
    def roles(self) -> list[str | None]:
        """``node_roles`` aligned with ``node_labels`` (``None``-padded)."""
        return self._padded(self.node_roles)

    @property
    def kinds(self) -> list[str | None]:
        """``node_kinds`` aligned with ``node_labels`` (``None``-padded)."""
        return self._padded(self.node_kinds)

    # -- node corpus ---------------------------------------------------------

    def node_path(self, index: int) -> Path:
        return self.folder / "nodes" / node_filename(index, self.node_labels[index])

    def node_groups(self) -> list[tuple[str, list[str]]]:
        """Return ``[(label, statements), ...]`` in node order."""
        if self.fit_mode == "baked":
            raise ManifoldFormatError(f"baked manifold {self.name!r} has no node corpus to read")
        groups: list[tuple[str, list[str]]] = []
        for idx, label in enumerate(self.node_labels):
            with open(self.node_path(idx)) as handle:
                statements = json.load(handle)
            if (
                not isinstance(statements, list) or not statements
                or not all(isinstance(s, str) for s in statements)
            ):
                raise ManifoldFormatError(
                    f"manifold {self.name!r} node {label!r} must be a non-empty "
                    f"JSON list of statement strings"
                )
            groups.append((label, list(statements)))
        return groups

    def nodes_sha256(self, *, resolved_template_sha256: str | None = None) -> str:
        """Stable hash of every input that determines a fit's output.

        The staleness key a fitted sidecar records.  Labels, corpus bytes,
        roles, and kinds always participate; authored folders add the domain
        and coordinates, discover folders the fit mode and hyperparameters,
        and a templated folder the resolved template's content hash (the ref
        string when the template cannot be resolved — the fit then fails
        loudly).  A baked folder never re-fits, so its key is provenance only.
        """
        h = hashlib.sha256()
        if self.fit_mode == "baked":
            h.update(canonical_json({"fit_mode": "baked", "node_labels": list(self.node_labels)}))
            return h.hexdigest()
        h.update(canonical_json(list(self.node_labels)))
        for idx in range(len(self.node_labels)):
            h.update(self.node_path(idx).read_bytes())
        if self.fit_mode == "authored":
            h.update(canonical_json(self.domain))
            h.update(canonical_json(self.node_coords))
        else:
            h.update(canonical_json({"fit_mode": self.fit_mode, "hyperparams": self.hyperparams}))
        h.update(canonical_json(self.node_roles))
        h.update(canonical_json(self.node_kinds))
        if self.template_ref is not None:
            if resolved_template_sha256 is None:
                from drowse.store.templates import (
                    AmbiguousTemplateError,
                    TemplateNotFoundError,
                    resolve_template,
                )

                try:
                    resolved_template_sha256 = resolve_template(self.template_ref).sha256()
                except (TemplateNotFoundError, AmbiguousTemplateError):
                    h.update(canonical_json({"template_ref": self.template_ref}))
                    return h.hexdigest()
            h.update(resolved_template_sha256.encode())
        return h.hexdigest()

    # -- fitted tensors ------------------------------------------------------

    def _load_sidecar(self, stem: str) -> ManifoldSidecar | ManifoldFormatError:
        cached = self._sidecars.get(stem)
        if cached is None:
            path = self.tensor_path(stem).with_suffix(".json")
            if self.fit_mode != "baked" and not sidecar_is_current(path):
                # A fit from an older format is stale, not corrupt: the corpus
                # is intact, so it reads as "not fitted for that model" and the
                # next fit overwrites it.  A baked folder has nothing to re-fit
                # from, so it keeps the strict read.
                cached = _StaleFit(f"fitted sidecar {path.name} is from an older format")
            else:
                try:
                    cached = ManifoldSidecar.load(path)
                except ManifoldFormatError as exc:
                    _log.warning("manifold %r: skipping unreadable fit %s: %s", self.name, stem, exc)
                    cached = exc
            self._sidecars[stem] = cached
        return cached

    def tensor_models(self) -> list[str]:
        """Stems (``<safe_model>[_sae-<rel>|_from-<src>]``) with a readable current fit."""
        return [
            stem for stem in self._tensor_stems
            if isinstance(self._load_sidecar(stem), ManifoldSidecar)
        ]

    def tensor_path(self, stem: str) -> Path:
        return self.folder / f"{stem}.safetensors"

    def sidecar(self, stem: str) -> ManifoldSidecar:
        """The validated sidecar of one fitted stem.

        Raises ``KeyError`` for a stem with no fitted pair on disk and
        :class:`ManifoldFormatError` for a stale or malformed sidecar.
        """
        if stem not in self._tensor_stems:
            raise KeyError(stem)
        loaded = self._load_sidecar(stem)
        if isinstance(loaded, ManifoldFormatError):
            raise loaded
        return loaded


@dataclass(frozen=True)
class DiscoverGenerationPlan:
    """Resume / extend plan for streaming discover-manifold generation.

    Returned by :func:`drowse.store.manifold_authoring.plan_discover_generation`.
    ``index_of`` maps each label to its on-disk node index; ``pending`` is the
    declared labels whose corpus file is not yet on disk (node order);
    ``added`` is the labels appended this call; ``resumed`` is ``True`` when
    an existing ``manifold.json`` was found.
    """

    folder: Path
    index_of: dict[str, int]
    pending: tuple[str, ...]
    added: tuple[str, ...]
    resumed: bool
