"""The one manifold install pipeline: local folder, ``.drowse``, or Hugging Face.

Every source only *materializes* a closure into a private staging area
(:func:`drowse.base.staging.staging_area`, outside every artifact root):

* a ``.drowse`` is validated and extracted by
  :func:`drowse.store.archive.extract_drowse_archive`;
* a local folder has its exact closure (manifest, corpora, manifested fitted
  pairs, referenced template) copied by :func:`_materialize_local_closure`;
* a Hugging Face coord resolves to a commit, downloads the repo's single root
  ``.drowse`` (:mod:`drowse.store.hub`), and takes the archive path.

:func:`install_closure` then applies one policy to all three: destination
``local/<name>`` unless ``as_`` says otherwise (never a reserved or the
bundled ``default`` namespace), identity rewrite for a rename, ``source``
stamped ``local`` or ``hf://<repo>@<sha>``, every fitted pair proven by
:func:`drowse.store.archive.validate_fitted_pair` on the exact bytes being
published, and template-then-manifold promotion under the manifest, template,
and pair locks.  A crash between the two promotions leaves a valid template
beside the previous manifold generation (a stale fit, never corruption).
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

from drowse.base.atomic import write_json_atomic
from drowse.base.integrity import NAME_REGEX, hash_file
from drowse.base.paths import manifold_dir, manifolds_dir
from drowse.base.staging import promote, recover_backup, staging_area
from drowse.errors import DrowseError, RequestError, SourceSpecError
from drowse.naming import RESERVED_NAMESPACES
from drowse.store.archive import (
    LOCAL_ONLY_MANIFOLD_KEYS,
    ArchiveClosure,
    DrowseArchiveError,
    _fitted_tensor_paths,
    _load_exact_json,
    _METADATA_JSON_MAX_BYTES,
    _payload_paths,
    _preflight_manifold_codec_sizes,
    _read_limited,
    extract_drowse_archive,
    validate_closure,
    validate_fitted_pair,
)
from drowse.store.manifold_folder import (
    ManifoldFolder,
    ManifoldFormatError,
    _locked_manifest,
    manifold_folder_tensor_paths,
    manifold_pair_lock,
)
from drowse.store.templates import template_dir, template_lock

#: Optional per-stage narration sink (CLI printer, SSE route).
ProgressCallback = Callable[[str], None]

#: The bundled namespace is owned by the package materializer; an install
#: there would be overwritten (or resurrected) by the next bootstrap.
_BUNDLED_NAMESPACE = "default"


class ManifoldInstallConflict(RuntimeError, DrowseError):
    """The install target (or its template) exists and ``force`` is off."""

    def user_message(self) -> tuple[int, str]:
        return (409, str(self) or self.__class__.__name__)


def _report(on_progress: ProgressCallback | None, message: str) -> None:
    if on_progress is not None:
        on_progress(message)


# ------------------------------------------------------------- entry point --


def install_manifold(
    target: str | Path,
    as_: Optional[str] = None,
    *,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Install a manifold from a ``.drowse`` path, a local folder, or an HF coord.

    ``target`` is a local ``.drowse`` file, a local manifold folder, or
    ``<owner>/<repo>[@revision]``.  ``as_`` (``<ns>/<name>``) overrides the
    default ``local/<manifold name>`` destination; ``force`` replaces an
    existing destination.  Returns the installed folder.
    """
    _report(on_progress, f"Resolving {target}...")
    path = Path(target).expanduser()
    if path.suffix.lower() == ".drowse":
        with staging_area() as area:
            _report(on_progress, f"Validating {path.name}...")
            closure = extract_drowse_archive(path, area)
            return install_closure(
                closure, as_=as_, force=force, source="local",
                on_progress=on_progress,
            )
    if path.is_dir():
        with staging_area() as area:
            _report(on_progress, f"Validating {path.name}...")
            closure = _materialize_local_closure(path, area)
            return install_closure(
                closure, as_=as_, force=force, source="local",
                on_progress=on_progress,
            )
    from drowse.base.hf import split_revision
    from drowse.store.hub import pull_manifold

    coord, revision = split_revision(str(target))
    if "/" not in coord:
        raise SourceSpecError(
            "install target must be a .drowse file, a manifold folder, or "
            f"'<owner>/<repo>[@revision]': {str(target)!r}"
        )
    if as_ is not None:
        # Refuse a bad destination before any network traffic.
        _install_destination(*_split_as(as_))
    return pull_manifold(
        coord, revision=revision, as_=as_, force=force, on_progress=on_progress,
    )


# ------------------------------------------------------ local materializer --


def _validate_local_tree(source: Path) -> None:
    if source.is_symlink():
        raise RequestError(f"local manifold source {source} is a symbolic link")
    for root, directories, files in os.walk(source, followlinks=False):
        for name in directories + files:
            path = Path(root) / name
            if path.is_symlink():
                raise RequestError(f"local manifold source {path} is a symbolic link")
            if not path.is_dir() and not path.is_file():
                raise RequestError(
                    f"local manifold source {path} is not a regular file or directory"
                )


def _materialize_local_closure(source: Path, area: Path) -> ArchiveClosure:
    """Copy a local folder's exact closure (+ referenced template) into ``area``.

    Only the closure travels: ``manifold.json``, the node corpora, and the
    manifested fitted pairs.  An unmanifested tensor fails, exactly as it
    would fail ``.drowse`` export.
    """
    from drowse.store.templates import resolve_template

    _validate_local_tree(source)
    try:
        _preflight_manifold_codec_sizes(source)
        folder = ManifoldFolder.load(source)
    except (ManifoldFormatError, DrowseArchiveError) as exc:
        raise RequestError(f"{source}: source folder is not a manifold ({exc})") from exc
    manifold_stage = area / "manifolds" / "local" / folder.name
    for path in _payload_paths(folder):
        destination = manifold_stage / path.relative_to(folder.folder)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    template_stage = None
    template_identity = None
    if folder.template_ref is not None:
        template = resolve_template(folder.template_ref)
        if template.path is None:
            raise ValueError(f"template {folder.template_ref} has no on-disk path")
        template_identity = (template.path.parent.name, template.name)
        template_stage = area / "templates" / "/".join(template_identity)
        template_stage.mkdir(parents=True)
        shutil.copyfile(template.path / "template.json", template_stage / "template.json")
    staged, staged_template = validate_closure(
        manifold_stage,
        expected_name=folder.name,
        template_path=template_stage,
        template_identity=template_identity,
    )
    return ArchiveClosure(
        manifold=manifold_stage,
        folder=staged,
        identity=("local", folder.name),
        template=template_stage,
        template_folder=staged_template,
        template_identity=template_identity,
    )


# ---------------------------------------------------------------- policy --


def _split_as(as_: str) -> tuple[str, str]:
    parts = as_.split("/")
    if len(parts) != 2:
        raise RequestError(f"as_ must be '<ns>/<name>', got {as_!r}")
    return parts[0], parts[1]


def _install_destination(namespace: str, name: str) -> Path:
    for label, component in (("namespace", namespace), ("name", name)):
        if not NAME_REGEX.fullmatch(component):
            raise RequestError(
                f"install target {label} {component!r} doesn't match NAME_REGEX "
                f"{NAME_REGEX.pattern}"
            )
    if namespace in RESERVED_NAMESPACES:
        raise RequestError(f"install target namespace {namespace!r} is reserved")
    if namespace == _BUNDLED_NAMESPACE:
        raise RequestError(
            f"install target namespace {namespace!r} is the bundled namespace; "
            "install under another namespace"
        )
    target = manifold_dir(namespace, name)
    if not target.resolve(strict=False).is_relative_to(
        manifolds_dir().resolve(strict=False)
    ):
        raise ValueError("install target resolves outside the manifold directory")
    return target


@contextmanager
def _install_locks(target: Path, template_target: Path | None) -> Iterator[None]:
    """Manifest lock, template lock, crash recovery, then every pair lock."""
    with ExitStack() as stack:
        stack.enter_context(_locked_manifest(target))
        if template_target is not None:
            stack.enter_context(template_lock(template_target))
            recover_backup(template_target)
        recover_backup(target)
        for tensor in manifold_folder_tensor_paths(target):
            stack.enter_context(manifold_pair_lock(tensor))
        yield


def _same_template(staged: Path, installed: Path) -> bool:
    entries = sorted(path.name for path in installed.iterdir())
    if entries != ["template.json"]:
        raise DrowseArchiveError(
            f"template target {installed} is not an exact template closure"
        )
    target = installed / "template.json"
    if target.is_symlink() or not target.is_file():
        raise DrowseArchiveError(f"template target {target} is not a regular file")
    return hash_file(staged / "template.json") == hash_file(target)


def _template_dependents(template_ref: str, *, excluding: Path) -> list[str]:
    """Installed manifolds other than ``excluding`` that reference the template."""
    root = manifolds_dir()
    if not root.is_dir():
        return []
    excluded = excluding.resolve(strict=False)
    dependents: list[str] = []
    for namespace in sorted(root.iterdir()):
        if not NAME_REGEX.fullmatch(namespace.name) or not namespace.is_dir():
            continue
        for candidate in sorted(namespace.iterdir()):
            manifest = candidate / "manifold.json"
            if (
                not NAME_REGEX.fullmatch(candidate.name)
                or candidate.resolve(strict=False) == excluded
                or not manifest.is_file()
            ):
                continue
            payload = _load_exact_json(
                _read_limited(
                    manifest,
                    maximum=_METADATA_JSON_MAX_BYTES,
                    label=f"manifold metadata {manifest}",
                ),
                label=f"manifold metadata {manifest}",
            )
            if isinstance(payload, dict) and payload.get("template_ref") == template_ref:
                dependents.append(f"{namespace.name}/{candidate.name}")
    return dependents


def _template_needs_install(
    closure: ArchiveClosure, template_target: Path, manifold_target: Path,
    *, force: bool,
) -> bool:
    assert closure.template is not None and closure.template_identity is not None
    if template_target.is_symlink() or (
        template_target.exists() and not template_target.is_dir()
    ):
        raise DrowseArchiveError(
            f"template target {template_target} is not a regular directory"
        )
    if not template_target.exists():
        return True
    if _same_template(closure.template, template_target):
        return False
    if not force:
        raise ManifoldInstallConflict(
            f"{template_target} already contains a different referenced "
            "template; pass force=True to replace the closure"
        )
    template_ref = "/".join(closure.template_identity)
    dependents = _template_dependents(template_ref, excluding=manifold_target)
    if dependents:
        raise ManifoldInstallConflict(
            f"cannot replace shared template {template_ref}; referenced by "
            f"{', '.join(dependents)}"
        )
    return True


def _rewrite_identity(
    stage: Path, *, name: str, source: str,
) -> ManifoldFolder:
    """Give the staged closure its installed identity and provenance.

    Drops local-only transaction keys, renames the manifest *and* every fitted
    sidecar (a sidecar repeats its manifold's name) when the destination name
    differs, stamps ``source``, and re-proves the manifest's hash map.
    """
    manifest = stage / "manifold.json"
    payload = json.loads(manifest.read_text())
    for key in LOCAL_ONLY_MANIFOLD_KEYS:
        payload.pop(key, None)
    write_json_atomic(manifest, payload)
    staged = ManifoldFolder.load(stage)
    files = dict(staged.files)
    if staged.name != name:
        for stem in staged.tensor_models():
            sidecar_path = staged.tensor_path(stem).with_suffix(".json")
            sidecar = json.loads(sidecar_path.read_text())
            sidecar["name"] = name
            write_json_atomic(sidecar_path, sidecar)
            files[sidecar_path.name] = hash_file(sidecar_path)
        staged.name = name
    staged.source = source
    staged.write_metadata(files=files)
    try:
        return ManifoldFolder.load(stage)
    except ManifoldFormatError as exc:
        raise DrowseArchiveError(
            f"staged manifold failed re-validation after identity rewrite ({exc})"
        ) from exc


def install_closure(
    closure: ArchiveClosure,
    *,
    as_: Optional[str],
    force: bool,
    source: str,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Publish a validated staged closure into the store (see module docs)."""
    from drowse.store.selectors import invalidate as invalidate_selector_index

    namespace, name = (
        _split_as(as_) if as_ is not None else ("local", closure.folder.name)
    )
    target = _install_destination(namespace, name)
    template_target = (
        template_dir(*closure.template_identity)
        if closure.template_identity is not None else None
    )
    label = f"{namespace}/{name}"
    with _install_locks(target, template_target):
        if target.exists() and not force:
            raise ManifoldInstallConflict(
                f"{target} already exists; pass force=True or as_=<ns>/<name>"
            )
        install_template = template_target is not None and _template_needs_install(
            closure, template_target, target, force=force,
        )
        _report(on_progress, f"Staging {label}...")
        staged = _rewrite_identity(closure.manifold, name=name, source=source)
        for tensor in _fitted_tensor_paths(staged):
            validate_fitted_pair(
                tensor, staged, template_sha256=closure.template_sha256,
            )
        _report(on_progress, f"Installing {label}...")
        if install_template:
            assert closure.template is not None and template_target is not None
            promote(closure.template, template_target)
        promote(closure.manifold, target)
    invalidate_selector_index()
    return target
