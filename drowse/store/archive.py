"""The ``.drowse`` manifold transport archive: codec, validation, and export.

A ``.drowse`` file is a stored ZIP whose ``pack.json`` (format v1) names one
primary manifold closure (``manifolds/<ns>/<name>/``: ``manifold.json``, the
node corpora, and every manifested fitted tensor/sidecar pair) plus, for a
templated manifold, its exact template closure
(``templates/<ns>/<name>/template.json``), with a size and sha256 per entry.
The same wire format is read and written by the drowse.ai browser edition.

:func:`extract_drowse_archive` validates the ZIP (path canonicality, entry
kinds, size/ratio/count bounds, duplicate-key JSON) and streams every entry
once — hashing while writing — into a caller-owned staging directory.
:func:`validate_closure` proves an on-disk closure exact, and
:func:`validate_fitted_pair` proves one fitted pair; the install pipeline
(:mod:`drowse.store.install`) runs them on everything it publishes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator

import torch
from safetensors import SafetensorError, safe_open

from drowse.errors import DrowseError
from drowse.base.atomic import fsync_directory
from drowse.base.integrity import NAME_REGEX, hash_file
from drowse.store.manifold_folder import (
    ManifoldFolder,
    ManifoldFormatError,
    _locked_manifest,
    load_manifold_sidecar_data,
    manifold_folder_tensor_paths,
    manifold_pair_lock,
)
from drowse.base.paths import manifold_dir
from drowse.store.templates import TemplateFolder, template_dir, template_lock


DROWSE_ARCHIVE_FORMAT_VERSION = 1
DROWSE_ARCHIVE_MAX_ENTRIES = 4_096
DROWSE_ARCHIVE_MAX_FILE_BYTES = 512 * 1024 * 1024
DROWSE_ARCHIVE_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
DROWSE_ARCHIVE_MAX_COMPRESSION_RATIO = 100
_PACK_MANIFEST_MAX_BYTES = 1024 * 1024
_METADATA_JSON_MAX_BYTES = 4 * 1024 * 1024
_NODE_JSON_MAX_BYTES = 64 * 1024 * 1024
_SAFETENSORS_HEADER_MAX_BYTES = 1024 * 1024
_SAFETENSORS_MAX_KEYS = 4_096
_TENSOR_VALIDATION_CHUNK_BYTES = 16 * 1024 * 1024
_TENSOR_MAX_ELEMENTS = DROWSE_ARCHIVE_MAX_FILE_BYTES // 4
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HF_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")

#: ``manifold.json`` keys that identify one *installed* folder generation
#: (fit transactions), never a transported artifact.  Export drops them and
#: install strips them from whatever an archive carries.
LOCAL_ONLY_MANIFOLD_KEYS = ("artifact_id", "fit_epochs")


class DrowseArchiveError(ValueError, DrowseError):
    """A ``.drowse`` archive is malformed or unsafe."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


@dataclass(frozen=True)
class ArchiveClosure:
    """A validated manifold closure laid out under a staging directory.

    ``manifold`` / ``template`` are the closure directories; ``identity`` is
    the archive-declared ``(namespace, name)`` of the primary manifold and
    ``template_identity`` that of its template.
    """

    manifold: Path
    folder: ManifoldFolder
    identity: tuple[str, str]
    template: Path | None = None
    template_folder: TemplateFolder | None = None
    template_identity: tuple[str, str] | None = None

    @property
    def template_sha256(self) -> str | None:
        return (
            self.template_folder.sha256()
            if self.template_folder is not None else None
        )


def _load_exact_json(raw: bytes, *, label: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise DrowseArchiveError(f"{label} has duplicate key {key!r}")
            out[key] = value
        return out

    try:
        return json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except UnicodeDecodeError as exc:
        raise DrowseArchiveError(f"{label} is not UTF-8 ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise DrowseArchiveError(f"{label} is invalid JSON ({exc})") from exc


def _load_pack_json(raw: bytes) -> dict[str, Any]:
    data = _load_exact_json(raw, label="pack.json")
    if not isinstance(data, dict):
        raise DrowseArchiveError("pack.json must be a JSON object")
    return data


def _read_limited(path: Path, *, maximum: int, label: str) -> bytes:
    try:
        with open(path, "rb") as handle:
            raw = handle.read(maximum + 1)
    except OSError as exc:
        raise DrowseArchiveError(f"{label} is unreadable ({exc})") from exc
    if len(raw) > maximum:
        raise DrowseArchiveError(f"{label} is larger than {maximum} bytes")
    return raw


def _archive_path(value: Any, *, root: str) -> tuple[str, str, str]:
    if not isinstance(value, str):
        raise DrowseArchiveError(f"pack.json {root!r} path must be a string")
    _validate_member_path(value)
    parts = PurePosixPath(value).parts
    if len(parts) != 3 or parts[0] != root:
        raise DrowseArchiveError(
            f"pack.json path {value!r} must be {root}/<namespace>/<name>"
        )
    namespace, name = parts[1], parts[2]
    for label, component in (("namespace", namespace), ("name", name)):
        if not NAME_REGEX.fullmatch(component):
            raise DrowseArchiveError(
                f"pack.json {root} {label} {component!r} must match "
                f"{NAME_REGEX.pattern}"
            )
    if root == "manifolds":
        _validate_manifold_identity(namespace, name)
    return value, namespace, name


def _validate_manifold_identity(namespace: str, name: str) -> None:
    from drowse.store.manifold_authoring import _validate_ns_name

    try:
        _validate_ns_name(namespace, name)
    except ManifoldFormatError as exc:
        raise DrowseArchiveError(str(exc)) from exc


def _validate_member_path(name: str) -> None:
    if not name or "\x00" in name:
        raise DrowseArchiveError("archive contains a blank or NUL-containing path")
    if "\\" in name:
        raise DrowseArchiveError(f"archive path {name!r} contains a backslash")
    if name.startswith("/") or _DRIVE_RE.match(name):
        raise DrowseArchiveError(f"archive path {name!r} is absolute")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise DrowseArchiveError(f"archive path {name!r} is not canonical")
    if PurePosixPath(name).is_absolute():
        raise DrowseArchiveError(f"archive path {name!r} is absolute")


def _validate_zip_infos(infos: list[zipfile.ZipInfo]) -> dict[str, zipfile.ZipInfo]:
    if not infos:
        raise DrowseArchiveError("archive is empty")
    if len(infos) > DROWSE_ARCHIVE_MAX_ENTRIES:
        raise DrowseArchiveError(
            f"archive has {len(infos)} entries; maximum is "
            f"{DROWSE_ARCHIVE_MAX_ENTRIES}"
        )
    by_name: dict[str, zipfile.ZipInfo] = {}
    casefolded: dict[str, str] = {}
    total = 0
    for info in infos:
        name = info.filename
        original = getattr(info, "orig_filename", name)
        if original != name:
            raise DrowseArchiveError(
                f"archive raw path {original!r} is not represented exactly"
            )
        _validate_member_path(original)
        _validate_member_path(name)
        if name in by_name:
            raise DrowseArchiveError(f"archive contains duplicate path {name!r}")
        folded = name.casefold()
        if folded in casefolded:
            raise DrowseArchiveError(
                f"archive paths {casefolded[folded]!r} and {name!r} collide "
                "case-insensitively"
            )
        if info.is_dir():
            raise DrowseArchiveError(f"archive entry {name!r} must be a regular file")
        if info.flag_bits & 0x1:
            raise DrowseArchiveError(f"archive entry {name!r} is encrypted")
        mode = info.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if kind not in (0, stat.S_IFREG):
            raise DrowseArchiveError(f"archive entry {name!r} is not a regular file")
        if info.file_size > DROWSE_ARCHIVE_MAX_FILE_BYTES:
            raise DrowseArchiveError(
                f"archive entry {name!r} is larger than 512 MiB"
            )
        total += info.file_size
        if total > DROWSE_ARCHIVE_MAX_TOTAL_BYTES:
            raise DrowseArchiveError("archive expands beyond the 2 GiB total limit")
        if info.file_size and (
            info.compress_size == 0
            or info.file_size / info.compress_size
            > DROWSE_ARCHIVE_MAX_COMPRESSION_RATIO
        ):
            raise DrowseArchiveError(
                f"archive entry {name!r} exceeds the 100:1 compression limit"
            )
        by_name[name] = info
        casefolded[folded] = name
    return by_name


def _validate_pack_manifest(
    data: dict[str, Any], infos: dict[str, zipfile.ZipInfo],
) -> tuple[str, tuple[str, str], str | None, tuple[str, str] | None, dict[str, Any]]:
    expected = {
        "format_version", "kind", "producer", "primary", "template",
        "source", "files",
    }
    if set(data) != expected:
        raise DrowseArchiveError("pack.json does not match the exact v1 schema")
    version = data["format_version"]
    if isinstance(version, bool) or version != DROWSE_ARCHIVE_FORMAT_VERSION:
        raise DrowseArchiveError(
            f"pack.json format_version must be {DROWSE_ARCHIVE_FORMAT_VERSION}"
        )
    if data["kind"] != "drowse-manifold":
        raise DrowseArchiveError("pack.json kind must be 'drowse-manifold'")
    producer = data["producer"]
    if (
        not isinstance(producer, dict)
        or set(producer) != {"name", "version"}
        or producer["name"] != "drowse"
        or not isinstance(producer["version"], str)
        or not producer["version"]
    ):
        raise DrowseArchiveError("pack.json producer must name a drowse version")

    primary, primary_ns, primary_name = _archive_path(
        data["primary"], root="manifolds",
    )
    template_value = data["template"]
    template: str | None = None
    template_identity: tuple[str, str] | None = None
    if template_value is not None:
        template, template_ns, template_name = _archive_path(
            template_value, root="templates",
        )
        template_identity = (template_ns, template_name)

    _validate_source_provenance(data["source"])

    files = data["files"]
    if not isinstance(files, dict) or not files:
        raise DrowseArchiveError("pack.json files must be a non-empty object")
    expected_names = set(infos) - {"pack.json"}
    if set(files) != expected_names:
        raise DrowseArchiveError(
            "pack.json files must describe every non-manifest archive entry exactly"
        )
    primary_prefix = primary + "/"
    template_path = None if template is None else template + "/template.json"
    for name, record in files.items():
        _validate_member_path(name)
        if not (
            name.startswith(primary_prefix)
            or template_path is not None and name == template_path
        ):
            raise DrowseArchiveError(f"archive entry {name!r} is outside its closure")
        if not isinstance(record, dict) or set(record) != {"size", "sha256"}:
            raise DrowseArchiveError(f"pack.json file record for {name!r} is invalid")
        size = record["size"]
        digest = record["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise DrowseArchiveError(f"pack.json size for {name!r} is invalid")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise DrowseArchiveError(f"pack.json sha256 for {name!r} is invalid")
        if infos[name].file_size != size:
            raise DrowseArchiveError(f"pack.json size for {name!r} does not match ZIP")
    if f"{primary}/manifold.json" not in files:
        raise DrowseArchiveError("archive closure has no manifold.json")
    if template is None:
        if any(name.startswith("templates/") for name in files):
            raise DrowseArchiveError("archive has an undeclared template closure")
    elif template_path not in files:
        raise DrowseArchiveError("archive template closure has no template.json")
    return (
        primary, (primary_ns, primary_name), template, template_identity,
        files,
    )


def _validate_source_provenance(value: Any) -> dict[str, str | None]:
    if not isinstance(value, dict) or set(value) != {
        "uri", "repository", "revision",
    }:
        raise DrowseArchiveError("pack.json source does not match the exact v1 schema")
    uri = value["uri"]
    if (
        not isinstance(uri, str)
        or not uri
        or "\x00" in uri
        or len(uri.encode("utf-8")) > 4_096
    ):
        raise DrowseArchiveError("pack.json source.uri must be a short non-empty string")
    repository = value["repository"]
    revision = value["revision"]
    for field, item in (("repository", repository), ("revision", revision)):
        if item is not None and (
            not isinstance(item, str)
            or not item
            or "\x00" in item
            or len(item.encode("utf-8")) > 1_024
        ):
            raise DrowseArchiveError(
                f"pack.json source.{field} must be a short string or null"
            )
    if uri.startswith("hf://"):
        if repository is None or revision is None:
            raise DrowseArchiveError(
                "Hugging Face source provenance requires an immutable revision"
            )
        if not _HF_COMMIT_RE.fullmatch(revision):
            raise DrowseArchiveError(
                "Hugging Face source provenance revision must be an immutable "
                "40-character commit SHA"
            )
        if uri != f"hf://{repository}@{revision}":
            raise DrowseArchiveError(
                "pack.json source fields do not describe the same Hugging Face source"
            )
    elif repository is not None or revision is not None:
        raise DrowseArchiveError(
            "non-Hugging Face source provenance cannot name a repository or revision"
        )
    return {"uri": uri, "repository": repository, "revision": revision}


def _source_provenance(source: str) -> dict[str, str | None]:
    repository = None
    revision = None
    if source.startswith("hf://"):
        from drowse.base.hf import split_revision

        repository, revision = split_revision(source[len("hf://"):])
        if revision is None:
            raise DrowseArchiveError(
                "Hugging Face manifold source must be pinned to a revision"
            )
    return _validate_source_provenance({
        "uri": source,
        "repository": repository,
        "revision": revision,
    })


def _stream_entry(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, *, expected: dict[str, Any],
    output: Path | None = None,
) -> None:
    digest = hashlib.sha256()
    count = 0
    handle = None
    try:
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            handle = open(output, "xb")
        with archive.open(info, "r") as source:
            while chunk := source.read(1024 * 1024):
                count += len(chunk)
                if count > expected["size"] or count > DROWSE_ARCHIVE_MAX_FILE_BYTES:
                    raise DrowseArchiveError(
                        f"archive entry {info.filename!r} exceeds its declared size"
                    )
                digest.update(chunk)
                if handle is not None:
                    handle.write(chunk)
        if count != expected["size"]:
            raise DrowseArchiveError(
                f"archive entry {info.filename!r} has the wrong size"
            )
        if digest.hexdigest() != expected["sha256"]:
            raise DrowseArchiveError(
                f"archive entry {info.filename!r} failed SHA-256 verification"
            )
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise DrowseArchiveError(
            f"could not read archive entry {info.filename!r} ({exc})"
        ) from exc
    finally:
        if handle is not None:
            handle.close()


def extract_drowse_archive(archive_path: Path, root: Path) -> ArchiveClosure:
    """Validate a ``.drowse`` and extract its closure under ``root``.

    Each entry is streamed exactly once: hashed and size-checked while it is
    written, so a bad entry aborts mid-copy and the caller discards ``root``.
    The extracted manifold directory doubles as the install stage.
    """
    source = Path(archive_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Drowse archive not found: {source}")
    if source.suffix.lower() != ".drowse":
        raise DrowseArchiveError("archive path must end in .drowse")
    try:
        archive = zipfile.ZipFile(source, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise DrowseArchiveError(f"could not open {source} ({exc})") from exc
    with archive:
        infos = _validate_zip_infos(archive.infolist())
        pack_info = infos.get("pack.json")
        if pack_info is None:
            raise DrowseArchiveError("archive has no pack.json")
        if pack_info.file_size > _PACK_MANIFEST_MAX_BYTES:
            raise DrowseArchiveError("pack.json is larger than 1 MiB")
        try:
            raw_pack = archive.read(pack_info)
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise DrowseArchiveError(f"could not read pack.json ({exc})") from exc
        data = _load_pack_json(raw_pack)
        primary, identity, template, template_identity, files = (
            _validate_pack_manifest(data, infos)
        )
        for name, record in files.items():
            output = root.joinpath(*PurePosixPath(name).parts)
            _stream_entry(archive, infos[name], expected=record, output=output)
    manifold_path = root.joinpath(*PurePosixPath(primary).parts)
    template_path = (
        root.joinpath(*PurePosixPath(template).parts)
        if template is not None else None
    )
    folder, loaded_template = validate_closure(
        manifold_path,
        expected_name=identity[1],
        template_path=template_path,
        template_identity=template_identity,
    )
    if data["source"] != _source_provenance(folder.source or "local"):
        raise DrowseArchiveError(
            "pack.json source provenance does not match manifold.json"
        )
    return ArchiveClosure(
        manifold=manifold_path,
        folder=folder,
        identity=identity,
        template=template_path,
        template_folder=loaded_template,
        template_identity=template_identity,
    )


def validate_closure(
    manifold_path: Path,
    *,
    expected_name: str,
    template_path: Path | None,
    template_identity: tuple[str, str] | None,
) -> tuple[ManifoldFolder, TemplateFolder | None]:
    """Prove an on-disk manifold (+ template) closure exact and self-consistent.

    Duplicate-key-free bounded JSON, the folder's integrity manifest, the
    identity, the template binding and corpus agreement, and an exact file set
    (no file outside ``manifold.json`` + node corpora + manifested pairs).
    Fitted pairs are proven separately by :func:`validate_fitted_pair`.
    """
    _preflight_manifold_codec_sizes(manifold_path)
    try:
        folder = ManifoldFolder.load(manifold_path)
    except (ManifoldFormatError, OSError, ValueError) as exc:
        raise DrowseArchiveError(f"manifold closure failed validation ({exc})") from exc
    if folder.name != expected_name:
        raise DrowseArchiveError(
            f"manifold closure {expected_name!r} contains manifold identity "
            f"{folder.name!r}"
        )
    loaded_template: TemplateFolder | None = None
    if folder.template_ref is None:
        if template_path is not None:
            raise DrowseArchiveError(
                "closure declares a template for a manifold without template_ref"
            )
    else:
        if template_path is None or template_identity is None:
            raise DrowseArchiveError(
                "templated manifold is missing its template closure"
            )
        expected_ref = "/".join(template_identity)
        if folder.template_ref != expected_ref:
            raise DrowseArchiveError(
                f"manifold template_ref {folder.template_ref!r} does not match "
                f"closure {expected_ref!r}"
            )
        template_entries = sorted(path.name for path in template_path.iterdir())
        if template_entries != ["template.json"]:
            raise DrowseArchiveError(
                f"template closure {template_path} is not exactly template.json"
            )
        raw_template = _read_limited(
            template_path / "template.json",
            maximum=_METADATA_JSON_MAX_BYTES,
            label="closure template.json",
        )
        _load_exact_json(raw_template, label="closure template.json")
        try:
            loaded_template = TemplateFolder.load(template_path)
        except (OSError, ValueError) as exc:
            raise DrowseArchiveError(
                f"closure template failed validation ({exc})"
            ) from exc
        if loaded_template.name != template_identity[1]:
            raise DrowseArchiveError(
                f"template closure {expected_ref!r} contains identity "
                f"{loaded_template.name!r}"
            )
        try:
            node_groups = dict(folder.node_groups())
        except (OSError, ValueError) as exc:
            raise DrowseArchiveError(
                f"closure node corpus failed validation ({exc})"
            ) from exc
        if (
            folder.node_labels != loaded_template.node_labels()
            or node_groups != loaded_template.node_corpora()
        ):
            raise DrowseArchiveError(
                "templated manifold corpus does not match its template closure"
            )
    expected_files = {
        path.relative_to(manifold_path).as_posix()
        for path in _payload_paths(folder)
    }
    actual_files: set[str] = set()
    for path in manifold_path.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise DrowseArchiveError(
                f"closure path {path} is not a regular file or directory"
            )
        if path.is_file():
            actual_files.add(path.relative_to(manifold_path).as_posix())
    if actual_files != expected_files:
        extras = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        raise DrowseArchiveError(
            "manifold closure is not exact "
            f"(extra={extras}, missing={missing})"
        )
    return folder, loaded_template


def _validate_safetensors_header(path: Path) -> None:
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            prefix = handle.read(8)
    except OSError as exc:
        raise DrowseArchiveError(
            f"fitted tensor {path.name} is unreadable ({exc})"
        ) from exc
    if len(prefix) != 8:
        raise DrowseArchiveError(f"fitted tensor {path.name} has a truncated header")
    header_bytes = int.from_bytes(prefix, "little")
    if (
        header_bytes <= 0
        or header_bytes > _SAFETENSORS_HEADER_MAX_BYTES
        or header_bytes + 8 > size
    ):
        raise DrowseArchiveError(
            f"fitted tensor {path.name} has an invalid or oversized header"
        )


def _tensor_shapes_and_finiteness(path: Path) -> dict[str, tuple[int, ...]]:
    _validate_safetensors_header(path)
    shapes: dict[str, tuple[int, ...]] = {}
    try:
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            keys = list(tensors.keys())
            if len(keys) > _SAFETENSORS_MAX_KEYS:
                raise DrowseArchiveError(
                    f"fitted tensor {path.name} has too many tensor keys"
                )
            for key in keys:
                view = tensors.get_slice(key)
                shape = tuple(int(v) for v in view.get_shape())
                if view.get_dtype() != "F32":
                    raise DrowseArchiveError(
                        f"fitted tensor {path.name} key {key!r} must be fp32"
                    )
                if len(shape) > 2:
                    raise DrowseArchiveError(
                        f"fitted tensor {path.name} key {key!r} has rank > 2"
                    )
                if not shape:
                    raise DrowseArchiveError(
                        f"fitted tensor {path.name} key {key!r} is scalar"
                    )
                if any(dimension <= 0 for dimension in shape):
                    raise DrowseArchiveError(
                        f"fitted tensor {path.name} key {key!r} has a zero "
                        "or negative dimension"
                    )
                elements = 1
                for dimension in shape:
                    if dimension > _TENSOR_MAX_ELEMENTS // elements:
                        raise DrowseArchiveError(
                            f"fitted tensor {path.name} key {key!r} is too large"
                        )
                    elements *= dimension
                if len(shape) == 1:
                    chunk_elements = max(
                        1, _TENSOR_VALIDATION_CHUNK_BYTES // 4,
                    )
                    chunks = (
                        view[start:min(start + chunk_elements, shape[0])]
                        for start in range(0, shape[0], chunk_elements)
                    )
                else:
                    columns = max(1, _TENSOR_VALIDATION_CHUNK_BYTES // 4)
                    if shape[1] <= columns:
                        rows = max(1, columns // max(shape[1], 1))
                        chunks = (
                            view[start:min(start + rows, shape[0])]
                            for start in range(0, shape[0], rows)
                        )
                    else:
                        chunks = (
                            view[row:row + 1, start:min(start + columns, shape[1])]
                            for row in range(shape[0])
                            for start in range(0, shape[1], columns)
                        )
                for chunk in chunks:
                    if not bool(torch.isfinite(chunk).all()):
                        raise DrowseArchiveError(
                            f"fitted tensor {path.name} key {key!r} is non-finite"
                        )
                    del chunk
                shapes[key] = shape
    except (OSError, OverflowError, ValueError, SafetensorError) as exc:
        if isinstance(exc, DrowseArchiveError):
            raise
        raise DrowseArchiveError(
            f"fitted tensor {path.name} is unreadable ({exc})"
        ) from exc
    return shapes


def validate_fitted_pair(
    path: Path,
    folder: ManifoldFolder,
    *,
    template_sha256: str | None = None,
) -> None:
    """Validate one fitted tensor/sidecar pair against its manifold closure.

    Header bounds, fp32 dtype, per-field shapes, finiteness, the layer roster,
    sidecar identity against ``manifold.json``, the corpus/template hash, and
    the filename-variant provenance.  Every install and export path runs it
    on every pair it publishes.
    """
    from drowse.geometry.manifold import domain_from_spec
    from drowse.base.paths import decode_release_id, parse_tensor_filename

    parsed_filename = parse_tensor_filename(path.name)
    if parsed_filename is None:
        raise DrowseArchiveError(f"fitted tensor filename {path.name!r} is invalid")
    sidecar_path = path.with_suffix(".json")
    _read_limited(
        sidecar_path,
        maximum=_METADATA_JSON_MAX_BYTES,
        label=f"fitted sidecar {sidecar_path.name}",
    )
    sidecar = load_manifold_sidecar_data(sidecar_path)
    variant = parsed_filename[1]
    if variant is None:
        if (
            sidecar["feature_space"] != "raw"
            or sidecar["method"] == "manifold_procrustes_transfer"
        ):
            raise DrowseArchiveError(
                f"fitted tensor {path.name} filename variant does not match "
                "its feature-space/source provenance"
            )
    else:
        variant_kind, encoded_identity = variant.split("-", 1)
        decoded_identity = decode_release_id(encoded_identity)
        if variant_kind == "sae":
            matches_variant = (
                sidecar["feature_space"] == f"sae-{decoded_identity}"
                and sidecar["sae_release"] == decoded_identity
                and sidecar["method"] != "manifold_procrustes_transfer"
            )
        else:
            matches_variant = (
                variant_kind == "from"
                and sidecar["feature_space"] == "raw"
                and sidecar["method"] == "manifold_procrustes_transfer"
                and sidecar["source_model_id"] == decoded_identity
            )
        if not matches_variant:
            raise DrowseArchiveError(
                f"fitted tensor {path.name} filename variant does not match "
                "its feature-space/source provenance"
            )
    if (
        sidecar["name"] != folder.name
        or sidecar["node_labels"] != folder.node_labels
        or sidecar["fit_mode"] != folder.fit_mode
        or sidecar["node_roles"] != folder.node_roles
        or sidecar["node_kinds"] != folder.node_kinds
    ):
        raise DrowseArchiveError(
            f"fitted tensor {path.name} identity does not match manifold.json"
        )
    if folder.fit_mode in {"authored", "baked"} and sidecar["domain"] != folder.domain:
        raise DrowseArchiveError(
            f"fitted tensor {path.name} domain does not match manifold.json"
        )
    if folder.template_ref is not None and template_sha256 is None:
        raise DrowseArchiveError(
            f"fitted tensor {path.name} needs its template closure to be validated"
        )
    expected_nodes_sha256 = folder.nodes_sha256(
        resolved_template_sha256=template_sha256,
    )
    if sidecar["nodes_sha256"] != expected_nodes_sha256:
        raise DrowseArchiveError(
            f"fitted tensor {path.name} node hash does not match its closure"
        )
    shapes = _tensor_shapes_and_finiteness(path)
    domain = domain_from_spec(sidecar["domain"])
    node_count = len(sidecar["node_labels"])
    if shapes.pop("node_coords", None) != (node_count, domain.intrinsic_dim):
        raise DrowseArchiveError(
            f"fitted tensor {path.name} has invalid node_coords shape"
        )
    fields = {
        "mean", "basis", "node_coords", "affine_map", "node_params",
        "rbf_weights", "poly_coeffs", "coord_offset", "coord_scale",
        "sigma_rbf_weights", "sigma_poly_coeffs",
    }
    by_layer: dict[int, dict[str, tuple[int, ...]]] = {}
    for key, shape in shapes.items():
        if key.count(".") != 1:
            raise DrowseArchiveError(f"fitted tensor {path.name} has invalid key {key!r}")
        head, field = key.split(".", 1)
        raw_layer = head.removeprefix("layer_")
        if (
            not head.startswith("layer_") or not raw_layer.isascii()
            or not raw_layer.isdecimal() or len(raw_layer) > 10
            or str(int(raw_layer)) != raw_layer or field not in fields
        ):
            raise DrowseArchiveError(f"fitted tensor {path.name} has invalid key {key!r}")
        by_layer.setdefault(int(raw_layer), {})[field] = shape
    if set(by_layer) != set(sidecar["fitted_layers"]) or not by_layer:
        raise DrowseArchiveError(
            f"fitted tensor {path.name} layers do not match its sidecar"
        )
    shares = sidecar["mahalanobis_share_per_layer"]
    if set(shares) != {str(layer) for layer in by_layer} or any(
        float(value) <= 0 for value in shares.values()
    ):
        raise DrowseArchiveError(
            f"fitted tensor {path.name} has invalid Mahalanobis shares"
        )
    hidden_dim: int | None = None
    curved_layers: set[int] = set()
    for layer, parts in by_layer.items():
        mean = parts.get("mean")
        basis = parts.get("basis")
        if (
            mean is None or len(mean) != 1 or basis is None or len(basis) != 2
            or basis[0] < 1 or mean[0] != basis[1] or basis[0] > basis[1]
        ):
            raise DrowseArchiveError(
                f"fitted tensor {path.name} layer {layer} has invalid mean/basis"
            )
        rank, dim = basis
        if hidden_dim is None:
            hidden_dim = dim
        elif hidden_dim != dim:
            raise DrowseArchiveError(
                f"fitted tensor {path.name} has inconsistent hidden dimensions"
            )
        curved = "node_params" in parts
        if curved:
            curved_layers.add(layer)
            required = {
                "mean", "basis", "node_params", "rbf_weights", "poly_coeffs",
                "coord_offset", "coord_scale",
            }
            sigma = {"sigma_rbf_weights", "sigma_poly_coeffs"}
            if not required.issubset(parts) or set(parts) - required - sigma:
                raise DrowseArchiveError(
                    f"fitted tensor {path.name} layer {layer} has invalid curved fields"
                )
            if bool(set(parts) & sigma) != sigma.issubset(parts):
                raise DrowseArchiveError(
                    f"fitted tensor {path.name} layer {layer} has incomplete sigma fields"
                )
            m = domain.embed_dim
            if (
                parts["node_params"] != (node_count, m)
                or parts["rbf_weights"] != (node_count, rank)
                or parts["poly_coeffs"] != (m + 1, rank)
                or parts["coord_offset"] != (m,)
                or parts["coord_scale"] != (m,)
            ):
                raise DrowseArchiveError(
                    f"fitted tensor {path.name} layer {layer} has invalid curved shapes"
                )
            if sidecar["feature_space"] == "raw" and not sigma.issubset(parts):
                raise DrowseArchiveError(
                    f"fitted tensor {path.name} raw curved layer {layer} lacks sigma fields"
                )
            if sigma.issubset(parts) and (
                parts["sigma_rbf_weights"] != (node_count, 1)
                or parts["sigma_poly_coeffs"] != (m + 1, 1)
            ):
                raise DrowseArchiveError(
                    f"fitted tensor {path.name} layer {layer} has invalid sigma shapes"
                )
        else:
            allowed = {"mean", "basis", "node_coords", "affine_map"}
            if set(parts) - allowed or parts.get("node_coords") != (node_count, rank):
                raise DrowseArchiveError(
                    f"fitted tensor {path.name} layer {layer} has invalid affine fields"
                )
            affine_map = parts.get("affine_map")
            if affine_map is None and domain.embed_dim != rank:
                raise DrowseArchiveError(
                    f"fitted tensor {path.name} layer {layer} needs affine_map"
                )
            if affine_map is not None and affine_map != (domain.embed_dim, rank):
                raise DrowseArchiveError(
                    f"fitted tensor {path.name} layer {layer} has invalid affine_map"
                )
    if curved_layers and curved_layers != set(by_layer):
        raise DrowseArchiveError(
            f"fitted tensor {path.name} mixes affine and curved layers"
        )
    origins = sidecar["origin_per_layer"]
    expected_origins = {str(layer) for layer in curved_layers}
    if set(origins) != expected_origins or any(
        len(value) != domain.intrinsic_dim for value in origins.values()
    ):
        raise DrowseArchiveError(f"fitted tensor {path.name} has invalid origins")


def _preflight_manifold_codec_sizes(folder: Path) -> None:
    if folder.is_symlink() or not folder.is_dir():
        raise DrowseArchiveError(f"manifold path {folder} is not a regular directory")
    manifest = folder / "manifold.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise DrowseArchiveError(f"manifold metadata {manifest} is not a regular file")
    raw_manifest = _read_limited(
        manifest,
        maximum=_METADATA_JSON_MAX_BYTES,
        label=f"manifold metadata {manifest}",
    )
    _load_exact_json(raw_manifest, label=f"manifold metadata {manifest}")
    for path in folder.rglob("*.json"):
        if path.is_symlink() or not path.is_file():
            raise DrowseArchiveError(f"manifold JSON path {path} is not a regular file")
        if path == manifest:
            continue
        try:
            relative = path.relative_to(folder)
        except ValueError as exc:
            raise DrowseArchiveError(f"manifold JSON path {path} escapes {folder}") from exc
        maximum = (
            _NODE_JSON_MAX_BYTES
            if relative.parts and relative.parts[0] == "nodes"
            else _METADATA_JSON_MAX_BYTES
        )
        raw = _read_limited(
            path,
            maximum=maximum,
            label=f"manifold JSON {relative}",
        )
        _load_exact_json(raw, label=f"manifold JSON {relative}")


def _fitted_tensor_paths(folder: ManifoldFolder) -> list[Path]:
    from drowse.base.paths import parse_tensor_filename

    declared = set(folder.files)
    tensors: set[str] = set()
    for relative in declared:
        _validate_member_path(relative)
        if len(PurePosixPath(relative).parts) != 1:
            raise DrowseArchiveError(
                f"manifold files entry {relative!r} is not a top-level fitted file"
            )
        if relative.endswith(".safetensors"):
            if parse_tensor_filename(relative) is None:
                raise DrowseArchiveError(
                    f"manifold files entry {relative!r} is not a fitted tensor"
                )
            tensors.add(relative)
        elif relative.endswith(".json"):
            tensor_name = str(Path(relative).with_suffix(".safetensors"))
            if parse_tensor_filename(tensor_name) is None:
                raise DrowseArchiveError(
                    f"manifold files entry {relative!r} is not a fitted sidecar"
                )
        else:
            raise DrowseArchiveError(
                f"manifold files entry {relative!r} is not part of manifold v10"
            )
    expected = {
        name
        for tensor in tensors
        for name in (tensor, str(Path(tensor).with_suffix(".json")))
    }
    if declared != expected:
        raise DrowseArchiveError(
            "manifold files must contain complete canonical tensor/sidecar pairs"
        )
    if folder.fit_mode == "baked" and not tensors:
        raise DrowseArchiveError("baked manifold closure has no fitted tensor")
    return [folder.folder / name for name in sorted(tensors)]


def _payload_paths(folder: ManifoldFolder) -> list[Path]:
    _read_limited(
        folder.folder / "manifold.json",
        maximum=_METADATA_JSON_MAX_BYTES,
        label="manifold.json",
    )
    paths = [folder.folder / "manifold.json"]
    if folder.fit_mode != "baked":
        nodes_dir = folder.folder / "nodes"
        if nodes_dir.is_symlink() or not nodes_dir.is_dir():
            raise DrowseArchiveError("manifold nodes path is not a regular directory")
        node_paths = [
            folder.node_path(i) for i in range(len(folder.node_labels))
        ]
        for path in node_paths:
            _read_limited(
                path,
                maximum=_NODE_JSON_MAX_BYTES,
                label=f"manifold node {path.name}",
            )
        try:
            folder.node_groups()
        except (OSError, ValueError) as exc:
            raise DrowseArchiveError(
                f"manifold node corpus failed validation ({exc})"
            ) from exc
        paths.extend(node_paths)
    for tensor in _fitted_tensor_paths(folder):
        paths.extend((tensor, tensor.with_suffix(".json")))
    unique: dict[Path, None] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise DrowseArchiveError(f"manifold closure file {path} is not regular")
        unique[path] = None
    discovered_tensors = set(folder.folder.glob("*.safetensors"))
    expected_tensors = set(_fitted_tensor_paths(folder))
    if discovered_tensors != expected_tensors:
        raise DrowseArchiveError(
            "manifold folder has an unproved or noncanonical fitted tensor"
        )
    paths = sorted(unique)
    if len(paths) + 1 > DROWSE_ARCHIVE_MAX_ENTRIES:
        raise DrowseArchiveError(
            f"manifold closure has more than {DROWSE_ARCHIVE_MAX_ENTRIES - 1} files"
        )
    total = 0
    for path in paths:
        size = path.stat().st_size
        if size > DROWSE_ARCHIVE_MAX_FILE_BYTES:
            raise DrowseArchiveError(f"manifold closure file {path} is larger than 512 MiB")
        total += size
    if total > DROWSE_ARCHIVE_MAX_TOTAL_BYTES:
        raise DrowseArchiveError("manifold closure is larger than 2 GiB")
    return paths


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _write_zip_path(archive: zipfile.ZipFile, archive_path: str, path: Path) -> None:
    with archive.open(_zip_info(archive_path), "w") as target, open(path, "rb") as source:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def _resolve_transport_template(folder: ManifoldFolder) -> TemplateFolder | None:
    from drowse.store.templates import resolve_template

    if folder.template_ref is None:
        return None
    parts = folder.template_ref.split("/")
    if len(parts) != 2 or any(
        not NAME_REGEX.fullmatch(component) for component in parts
    ):
        raise DrowseArchiveError(
            "Drowse archive requires a namespace-qualified template_ref"
        )
    expected_path = template_dir(parts[0], parts[1])
    template_file = expected_path / "template.json"
    if (
        expected_path.is_symlink()
        or not expected_path.is_dir()
        or template_file.is_symlink()
        or not template_file.is_file()
    ):
        raise DrowseArchiveError("referenced template path is not a regular directory")
    raw_template = _read_limited(
        template_file,
        maximum=_METADATA_JSON_MAX_BYTES,
        label="referenced template.json",
    )
    _load_exact_json(raw_template, label="referenced template.json")
    loaded = resolve_template(folder.template_ref)
    if loaded.path is None:
        raise DrowseArchiveError("referenced template has no on-disk path")
    if (
        loaded.name != parts[1]
        or loaded.path.resolve(strict=False) != expected_path.resolve(strict=False)
        or loaded.path.is_symlink()
    ):
        raise DrowseArchiveError(
            "referenced template path and embedded identity do not match template_ref"
        )
    return loaded


@contextmanager
def _source_closure_locks(folder_path: Path) -> Iterator[None]:
    """Freeze one installed manifold (+ its template) for a consistent read.

    Global order: manifest lock, then the template lock, then every fitted
    pair lock.  The template is computable from the locked manifest because
    transportable ``template_ref`` values are namespace-qualified.
    """
    with ExitStack() as stack:
        stack.enter_context(_locked_manifest(folder_path))
        ref = ManifoldFolder.load(folder_path, verify_manifest=False).template_ref
        if ref is not None:
            parts = ref.split("/")
            if len(parts) != 2 or any(
                not NAME_REGEX.fullmatch(component) for component in parts
            ):
                raise DrowseArchiveError(
                    "Drowse archive requires a namespace-qualified template_ref"
                )
            stack.enter_context(template_lock(template_dir(*parts)))
        for tensor in manifold_folder_tensor_paths(folder_path):
            stack.enter_context(manifold_pair_lock(tensor))
        yield


def _transport_manifest_bytes(
    folder: ManifoldFolder, kept_files: set[str],
) -> bytes:
    """``manifold.json`` as transported: local-only keys dropped, ``files``
    narrowed to the exported pairs."""
    raw = _read_limited(
        folder.folder / "manifold.json",
        maximum=_METADATA_JSON_MAX_BYTES,
        label="manifold.json",
    )
    payload = _load_exact_json(raw, label="manifold.json")
    for key in LOCAL_ONLY_MANIFOLD_KEYS:
        payload.pop(key, None)
    payload["files"] = {
        name: digest for name, digest in payload["files"].items()
        if name in kept_files
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def export_drowse_archive(
    namespace: str,
    name: str,
    *,
    output: str | Path | None = None,
    include_tensor: Callable[[str], bool] | None = None,
) -> Path:
    """Export one installed manifold and its exact template closure.

    ``output`` is a ``.drowse`` path or a directory (default: the working
    directory), receiving ``<name>.drowse``.  ``include_tensor`` narrows the
    exported fitted pairs by tensor filename (``drowse pack push``'s model /
    variant filter); the transported ``manifold.json`` is narrowed to match.
    """
    from drowse import __version__

    _validate_manifold_identity(namespace, name)
    folder_path = manifold_dir(namespace, name)
    if not (folder_path / "manifold.json").is_file():
        raise FileNotFoundError(f"manifold {namespace}/{name} not found at {folder_path}")
    destination = (
        Path(output).expanduser()
        if output is not None else Path.cwd() / f"{name}.drowse"
    )
    if destination.exists() and destination.is_dir():
        destination = destination / f"{name}.drowse"
    if destination.suffix.lower() != ".drowse":
        raise DrowseArchiveError("export path must end in .drowse")
    if destination.resolve(strict=False).is_relative_to(
        folder_path.resolve(strict=False)
    ):
        raise DrowseArchiveError(
            "Drowse archive export path must be outside its manifold closure"
        )

    with _source_closure_locks(folder_path):
        _preflight_manifold_codec_sizes(folder_path)
        folder = ManifoldFolder.load(folder_path)
        if folder.name != name:
            raise DrowseArchiveError(
                f"manifold path {namespace}/{name} contains identity "
                f"{folder.name!r}"
            )
        template_folder = _resolve_transport_template(folder)
        template_path = (
            template_folder.path if template_folder is not None else None
        )
        if template_path is not None and destination.resolve(
            strict=False,
        ).is_relative_to(template_path.resolve(strict=False)):
            raise DrowseArchiveError(
                "Drowse archive export path must be outside its template closure"
            )
        if template_folder is not None and (
            folder.node_labels != template_folder.node_labels()
            or dict(folder.node_groups()) != template_folder.node_corpora()
        ):
            raise DrowseArchiveError(
                "templated manifold corpus does not match its template"
            )
        template_sha256 = (
            template_folder.sha256() if template_folder is not None else None
        )
        tensors = [
            tensor for tensor in _fitted_tensor_paths(folder)
            if include_tensor is None or include_tensor(tensor.name)
        ]
        for tensor in tensors:
            validate_fitted_pair(tensor, folder, template_sha256=template_sha256)
        kept = {
            item.name for tensor in tensors
            for item in (tensor, tensor.with_suffix(".json"))
        }
        manifest_path = folder.folder / "manifold.json"
        closure_paths = [
            path for path in _payload_paths(folder)
            if path.parent != folder.folder
            or path == manifest_path or path.name in kept
        ]
        primary = f"manifolds/{namespace}/{name}"
        payloads: list[tuple[str, Path | bytes]] = [
            (
                f"{primary}/{path.relative_to(folder.folder).as_posix()}",
                _transport_manifest_bytes(folder, kept)
                if path == manifest_path else path,
            )
            for path in closure_paths
        ]
        template_key = None
        if template_folder is not None and template_path is not None:
            template_key = f"templates/{template_path.parent.name}/{template_folder.name}"
            template_file = template_path / "template.json"
            if template_file.is_symlink() or not template_file.is_file():
                raise DrowseArchiveError(
                    "referenced template.json is not a regular file"
                )
            _read_limited(
                template_file,
                maximum=_METADATA_JSON_MAX_BYTES,
                label="referenced template.json",
            )
            payloads.append((f"{template_key}/template.json", template_file))
        _write_archive(
            destination,
            payloads,
            primary=primary,
            template=template_key,
            source=_source_provenance(folder.source or "local"),
            producer_version=__version__,
        )
    return destination


def _write_archive(
    destination: Path,
    payloads: list[tuple[str, Path | bytes]],
    *,
    primary: str,
    template: str | None,
    source: dict[str, str | None],
    producer_version: str,
) -> None:
    if len(payloads) + 1 > DROWSE_ARCHIVE_MAX_ENTRIES:
        raise DrowseArchiveError(
            f"Drowse archive would exceed {DROWSE_ARCHIVE_MAX_ENTRIES} entries"
        )
    files: dict[str, dict[str, Any]] = {}
    payload_bytes = 0
    for archive_path, item in payloads:
        _validate_member_path(archive_path)
        if isinstance(item, bytes):
            size, digest = len(item), hashlib.sha256(item).hexdigest()
        else:
            size, digest = item.stat().st_size, hash_file(item)
        if size > DROWSE_ARCHIVE_MAX_FILE_BYTES:
            raise DrowseArchiveError(
                f"archive payload {archive_path!r} is larger than 512 MiB"
            )
        payload_bytes += size
        files[archive_path] = {"size": size, "sha256": digest}
    manifest = {
        "format_version": DROWSE_ARCHIVE_FORMAT_VERSION,
        "kind": "drowse-manifold",
        "producer": {"name": "drowse", "version": producer_version},
        "primary": primary,
        "template": template,
        "source": source,
        "files": files,
    }
    raw_manifest = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    if len(raw_manifest) > _PACK_MANIFEST_MAX_BYTES:
        raise DrowseArchiveError("pack.json would be larger than 1 MiB")
    if payload_bytes + len(raw_manifest) > DROWSE_ARCHIVE_MAX_TOTAL_BYTES:
        raise DrowseArchiveError("Drowse archive would expand beyond 2 GiB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with zipfile.ZipFile(temp_path, "w", allowZip64=True) as archive:
            archive.writestr(_zip_info("pack.json"), raw_manifest)
            for archive_path, item in sorted(payloads, key=lambda pair: pair[0]):
                if isinstance(item, bytes):
                    archive.writestr(_zip_info(archive_path), item)
                else:
                    _write_zip_path(archive, archive_path, item)
        with open(temp_path, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        fsync_directory(destination.parent)
    finally:
        temp_path.unlink(missing_ok=True)
