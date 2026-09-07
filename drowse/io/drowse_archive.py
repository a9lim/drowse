"""Secure ``.drowse`` manifold transport archives."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator

import torch
from safetensors import SafetensorError, safe_open

from drowse.core.errors import DrowseError
from drowse.io.atomic import artifact_lock, fsync_directory, write_json_atomic
from drowse.io.integrity import NAME_REGEX, hash_file
from drowse.io.manifold_folder import (
    ManifoldFolder,
    ManifoldFormatError,
    load_manifold_sidecar_data,
    manifold_folder_tensor_paths,
    manifold_pair_lock,
)
from drowse.io.paths import manifold_dir, drowse_home
from drowse.io.templates import TemplateFolder, template_dir


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
_TRANSACTION_FORMAT_VERSION = 1
_LEGACY_TRANSACTION_RE = re.compile(
    r"^(?P<name>[a-z][a-z0-9._-]{0,63})\.drowse-"
    r"(?P<kind>stage|bak)-(?P<token>[0-9a-f]{32})$"
)

ProgressCallback = Callable[[str], None]


class DrowseArchiveError(ValueError, DrowseError):
    """A ``.drowse`` archive is malformed or unsafe."""

    def user_message(self) -> tuple[int, str]:
        return (400, str(self) or self.__class__.__name__)


def _report(on_progress: ProgressCallback | None, message: str) -> None:
    if on_progress is not None:
        on_progress(message)


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
    from drowse.io.manifold_authoring import _validate_ns_name

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
    from drowse.io.brand_migration import migrate_legacy_record

    data = migrate_legacy_record(data)
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


def _validate_fitted_tensor(
    path: Path,
    folder: ManifoldFolder,
    *,
    resolved_template_sha256: str | None = None,
) -> None:
    from drowse.core.manifold import domain_from_spec
    from drowse.io.paths import decode_release_id, parse_tensor_filename

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
    if folder.template_ref is not None and resolved_template_sha256 is None:
        raise DrowseArchiveError(
            f"fitted tensor {path.name} needs its template closure to be validated"
        )
    expected_nodes_sha256 = folder.nodes_sha256(
        resolved_template_sha256=resolved_template_sha256,
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


def _validate_extracted_closure(
    root: Path,
    primary: str,
    source_identity: tuple[str, str],
    template: str | None,
    template_identity: tuple[str, str] | None,
    source_provenance: dict[str, Any],
) -> tuple[ManifoldFolder, TemplateFolder | None]:
    manifold_path = root.joinpath(*PurePosixPath(primary).parts)
    _preflight_manifold_codec_sizes(manifold_path)
    try:
        folder = ManifoldFolder.load(manifold_path)
    except (ManifoldFormatError, OSError, ValueError) as exc:
        raise DrowseArchiveError(f"archive manifold failed validation ({exc})") from exc
    if folder.name != source_identity[1]:
        raise DrowseArchiveError(
            f"archive primary {primary!r} contains manifold identity "
            f"{folder.name!r}"
        )
    expected_source = _source_provenance(folder.source or "local")
    if source_provenance != expected_source:
        raise DrowseArchiveError(
            "pack.json source provenance does not match manifold.json"
        )
    loaded_template: TemplateFolder | None = None
    if folder.template_ref is None:
        if template is not None:
            raise DrowseArchiveError(
                "archive declares a template for a manifold without template_ref"
            )
    else:
        if template is None or template_identity is None:
            raise DrowseArchiveError(
                "templated manifold is missing its template closure"
            )
        expected_ref = "/".join(template_identity)
        if folder.template_ref != expected_ref:
            raise DrowseArchiveError(
                f"manifold template_ref {folder.template_ref!r} does not match "
                f"closure {expected_ref!r}"
            )
        template_path = root.joinpath(*PurePosixPath(template).parts)
        raw_template = _read_limited(
            template_path / "template.json",
            maximum=_METADATA_JSON_MAX_BYTES,
            label="archive template.json",
        )
        _load_exact_json(raw_template, label="archive template.json")
        try:
            loaded_template = TemplateFolder.load(template_path)
        except (OSError, ValueError) as exc:
            raise DrowseArchiveError(
                f"archive template failed validation ({exc})"
            ) from exc
        if loaded_template.name != template_identity[1]:
            raise DrowseArchiveError(
                f"archive template path {template!r} contains identity "
                f"{loaded_template.name!r}"
            )
        try:
            node_groups = dict(folder.node_groups())
        except (OSError, ValueError) as exc:
            raise DrowseArchiveError(
                f"archive node corpus failed validation ({exc})"
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
                f"archive manifold path {path} is not a regular file or directory"
            )
        if path.is_file():
            actual_files.add(path.relative_to(manifold_path).as_posix())
    if actual_files != expected_files:
        extras = sorted(actual_files - expected_files)
        missing = sorted(expected_files - actual_files)
        raise DrowseArchiveError(
            "archive manifold closure is not exact "
            f"(extra={extras}, missing={missing})"
        )
    template_sha256 = (
        loaded_template.sha256() if loaded_template is not None else None
    )
    for tensor in _fitted_tensor_paths(folder):
        _validate_fitted_tensor(
            tensor,
            folder,
            resolved_template_sha256=template_sha256,
        )
    return folder, loaded_template


def _source_provenance(source: str) -> dict[str, str | None]:
    repository = None
    revision = None
    if source.startswith("hf://"):
        from drowse.io.hf import split_revision

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


def _fitted_tensor_paths(folder: ManifoldFolder) -> list[Path]:
    from drowse.io.paths import parse_tensor_filename

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


@contextmanager
def _closure_locks(
    manifold_path: Path, template_path: Path | None,
) -> Iterator[None]:
    lock_paths = [
        manifold_path.parent / f"{manifold_path.name}.manifest",
    ]
    if template_path is not None:
        lock_paths.append(template_path.parent / f"{template_path.name}.template")
    with ExitStack() as stack:
        for lock_path in sorted(lock_paths, key=lambda path: str(path.resolve(strict=False))):
            stack.enter_context(artifact_lock(lock_path))
        for tensor in manifold_folder_tensor_paths(manifold_path):
            stack.enter_context(manifold_pair_lock(tensor))
        yield


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
    from drowse.io.templates import resolve_template

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


def export_drowse_archive(
    namespace: str,
    name: str,
    *,
    output: str | Path | None = None,
) -> Path:
    """Export one manifold and its exact referenced-template closure."""
    from drowse import __version__
    from drowse.io.bootstrap import materialize_bundled_artifacts
    _validate_manifold_identity(namespace, name)
    materialize_bundled_artifacts()
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

    for _attempt in range(8):
        _preflight_manifold_codec_sizes(folder_path)
        initial = ManifoldFolder.load(folder_path, verify_manifest=False)
        initial_template = _resolve_transport_template(initial)
        initial_template_path = (
            initial_template.path if initial_template is not None else None
        )
        with _closure_locks(folder_path, initial_template_path):
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
            if template_path != initial_template_path:
                continue
            if template_path is not None and destination.resolve(
                strict=False,
            ).is_relative_to(template_path.resolve(strict=False)):
                raise DrowseArchiveError(
                    "Drowse archive export path must be outside its template closure"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
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
            closure_paths = _payload_paths(folder)
            for tensor in _fitted_tensor_paths(folder):
                _validate_fitted_tensor(
                    tensor,
                    folder,
                    resolved_template_sha256=template_sha256,
                )
            primary = f"manifolds/{namespace}/{name}"
            payloads: list[tuple[str, Path]] = [
                (
                    f"{primary}/{path.relative_to(folder.folder).as_posix()}",
                    path,
                )
                for path in closure_paths
            ]
            template_key = None
            if template_folder is not None and template_path is not None:
                template_ns = template_path.parent.name
                template_key = f"templates/{template_ns}/{template_folder.name}"
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
            if len(payloads) + 1 > DROWSE_ARCHIVE_MAX_ENTRIES:
                raise DrowseArchiveError(
                    f"Drowse archive would exceed {DROWSE_ARCHIVE_MAX_ENTRIES} entries"
                )
            payload_bytes = 0
            for archive_path, path in payloads:
                _validate_member_path(archive_path)
                size = path.stat().st_size
                if size > DROWSE_ARCHIVE_MAX_FILE_BYTES:
                    raise DrowseArchiveError(
                        f"archive payload {archive_path!r} is larger than 512 MiB"
                    )
                payload_bytes += size
            files = {
                archive_path: {
                    "size": path.stat().st_size,
                    "sha256": hash_file(path),
                }
                for archive_path, path in payloads
            }
            manifest = {
                "format_version": DROWSE_ARCHIVE_FORMAT_VERSION,
                "kind": "drowse-manifold",
                "producer": {"name": "drowse", "version": __version__},
                "primary": primary,
                "template": template_key,
                "source": _source_provenance(folder.source or "local"),
                "files": files,
            }
            raw_manifest = (
                json.dumps(manifest, sort_keys=True, indent=2) + "\n"
            ).encode()
            if len(raw_manifest) > _PACK_MANIFEST_MAX_BYTES:
                raise DrowseArchiveError("pack.json would be larger than 1 MiB")
            if payload_bytes + len(raw_manifest) > DROWSE_ARCHIVE_MAX_TOTAL_BYTES:
                raise DrowseArchiveError("Drowse archive would expand beyond 2 GiB")
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            os.close(fd)
            temp_path = Path(temp_name)
            try:
                with zipfile.ZipFile(temp_path, "w", allowZip64=True) as archive:
                    archive.writestr(_zip_info("pack.json"), raw_manifest)
                    for archive_path, path in sorted(payloads):
                        _write_zip_path(archive, archive_path, path)
                with open(temp_path, "rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temp_path, destination)
                fsync_directory(destination.parent)
            finally:
                temp_path.unlink(missing_ok=True)
            return destination
    raise DrowseArchiveError("referenced template changed repeatedly during export")


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    files = [path for path in destination.rglob("*") if path.is_file()]
    for path in files:
        with open(path, "rb") as handle:
            os.fsync(handle.fileno())
    directories = [
        path for path in destination.rglob("*") if path.is_dir()
    ]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        fsync_directory(path)
    fsync_directory(destination)
    fsync_directory(destination.parent)


def _same_template(source: Path, destination: Path) -> bool:
    try:
        entries = list(destination.iterdir())
    except OSError as exc:
        raise DrowseArchiveError(
            f"template target {destination} is unreadable ({exc})"
        ) from exc
    if len(entries) != 1 or entries[0].name != "template.json":
        raise DrowseArchiveError(
            f"template target {destination} is not an exact template closure"
        )
    target = entries[0]
    if target.is_symlink() or not target.is_file():
        raise DrowseArchiveError(
            f"template target {target} is not a regular file"
        )
    return hash_file(source / "template.json") == hash_file(target)


def _other_template_dependents(
    root: Path,
    template_ref: str,
    *,
    excluding: Path,
) -> list[str]:
    manifold_root = root / "manifolds"
    if not manifold_root.exists():
        return []
    if manifold_root.is_symlink() or not manifold_root.is_dir():
        raise DrowseArchiveError(
            f"Drowse archive manifold root {manifold_root} is not a regular directory"
        )
    excluded = excluding.resolve(strict=False)
    dependents: list[str] = []
    for namespace in sorted(manifold_root.iterdir()):
        if not NAME_REGEX.fullmatch(namespace.name):
            continue
        if namespace.is_symlink() or not namespace.is_dir():
            raise DrowseArchiveError(
                f"Drowse archive manifold namespace {namespace} is not a regular directory"
            )
        for candidate in sorted(namespace.iterdir()):
            if not NAME_REGEX.fullmatch(candidate.name):
                continue
            if candidate.resolve(strict=False) == excluded:
                continue
            if candidate.is_symlink() or not candidate.is_dir():
                raise DrowseArchiveError(
                    f"Drowse archive manifold path {candidate} is not a regular directory"
                )
            manifest = candidate / "manifold.json"
            if not manifest.exists():
                continue
            if manifest.is_symlink() or not manifest.is_file():
                raise DrowseArchiveError(
                    f"Drowse archive manifold metadata {manifest} is not a regular file"
                )
            try:
                payload = json.loads(_read_limited(
                    manifest,
                    maximum=_METADATA_JSON_MAX_BYTES,
                    label=f"manifold metadata {manifest}",
                ))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DrowseArchiveError(
                    f"manifold metadata {manifest} is invalid ({exc})"
                ) from exc
            if not isinstance(payload, dict):
                raise DrowseArchiveError(
                    f"manifold metadata {manifest} must be an object"
                )
            if payload.get("template_ref") == template_ref:
                dependents.append(f"{namespace.name}/{candidate.name}")
    return dependents


def _reset_installed_manifold_transaction_state(folder: Path) -> None:
    manifest_path = folder / "manifold.json"
    try:
        payload = json.loads(_read_limited(
            manifest_path,
            maximum=_METADATA_JSON_MAX_BYTES,
            label=f"staged manifold metadata {manifest_path}",
        ))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DrowseArchiveError(
            f"staged manifold metadata {manifest_path} is invalid ({exc})"
        ) from exc
    if not isinstance(payload, dict):
        raise DrowseArchiveError(
            f"staged manifold metadata {manifest_path} must be an object"
        )
    payload.pop("artifact_id", None)
    payload.pop("fit_epochs", None)
    write_json_atomic(manifest_path, payload)


def _transaction_dir(root: Path) -> Path:
    return root / ".drowse-transactions"


def _validate_transaction_roots(root: Path) -> None:
    for path, label in (
        (_transaction_dir(root), "transaction directory"),
        (root / "manifolds", "manifold root"),
        (root / "templates", "template root"),
    ):
        if path.is_symlink():
            raise DrowseArchiveError(f"Drowse archive {label} {path} is a symbolic link")
        if path.exists() and not path.is_dir():
            raise DrowseArchiveError(f"Drowse archive {label} {path} is not a directory")


def _transaction_stage_path(root: Path, token: str, kind: str) -> Path:
    return _transaction_dir(root) / f"{token}.{kind}.stage"


def _transaction_backup_path(root: Path, token: str, kind: str) -> Path:
    return _transaction_dir(root) / f"{token}.{kind}.backup"


def _relative_transaction_path(root: Path, path: Path) -> str:
    root = root.resolve(strict=False)
    path = path.resolve(strict=False)
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise DrowseArchiveError(f"transaction path {path} escapes {root}") from exc
    _validate_member_path(relative)
    return relative


def _resolve_transaction_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str):
        raise DrowseArchiveError("transaction journal path must be a string")
    _validate_member_path(value)
    path = root.joinpath(*PurePosixPath(value).parts).resolve(strict=False)
    try:
        path.relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise DrowseArchiveError(f"transaction journal path {value!r} escapes root") from exc
    return path


def _write_transaction_journal(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)
    fsync_directory(path.parent)


def _load_transaction_journal(
    root: Path, path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = _load_pack_json(_read_limited(
        path,
        maximum=_METADATA_JSON_MAX_BYTES,
        label=f"transaction journal {path}",
    ))
    if set(data) != {"format_version", "token", "phase", "items"}:
        raise DrowseArchiveError(f"transaction journal {path} has an invalid schema")
    if data["format_version"] != _TRANSACTION_FORMAT_VERSION:
        raise DrowseArchiveError(f"transaction journal {path} has an invalid version")
    if (
        not isinstance(data["token"], str)
        or not re.fullmatch(r"[0-9a-f]{32}", data["token"])
        or path.stem != data["token"]
        or data["phase"] not in {
            "staging", "prepared", "backed_up", "committed",
        }
        or not isinstance(data["items"], list)
        or not 1 <= len(data["items"]) <= 2
    ):
        raise DrowseArchiveError(f"transaction journal {path} has invalid values")
    items: list[dict[str, Any]] = []
    for index, raw in enumerate(data["items"]):
        if not isinstance(raw, dict) or set(raw) != {
            "kind", "stage", "target", "backup", "had_existing",
        }:
            raise DrowseArchiveError(f"transaction journal {path} has an invalid item")
        expected_kind = "manifold" if index == 0 else "template"
        if raw["kind"] != expected_kind or not isinstance(raw["had_existing"], bool):
            raise DrowseArchiveError(f"transaction journal {path} has an invalid item")
        target = _resolve_transaction_path(root, raw["target"])
        expected_root = "manifolds" if expected_kind == "manifold" else "templates"
        relative_target = target.relative_to(root.resolve(strict=False))
        if (
            len(relative_target.parts) != 3
            or relative_target.parts[0] != expected_root
            or any(
                not NAME_REGEX.fullmatch(component)
                for component in relative_target.parts[1:]
            )
        ):
            raise DrowseArchiveError(f"transaction journal {path} has an invalid target")
        stage = _resolve_transaction_path(root, raw["stage"])
        backup = _resolve_transaction_path(root, raw["backup"])
        if (
            stage != _transaction_stage_path(root, data["token"], expected_kind)
            or backup
            != _transaction_backup_path(root, data["token"], expected_kind)
        ):
            raise DrowseArchiveError(f"transaction journal {path} has invalid staging paths")
        items.append({
            "kind": expected_kind,
            "stage": stage,
            "target": target,
            "backup": backup,
            "had_existing": raw["had_existing"],
        })
    return data, items


def _remove_transaction_tree(path: Path) -> None:
    if path.is_symlink():
        raise DrowseArchiveError(f"transaction path {path} is a symbolic link")
    if path.exists():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            raise DrowseArchiveError(f"transaction path {path} is not a directory")


def _recover_transaction_journal(root: Path, journal: Path) -> None:
    data, items = _load_transaction_journal(root, journal)
    manifold_target = items[0]["target"]
    template_target = items[1]["target"] if len(items) == 2 else None
    with _closure_locks(manifold_target, template_target):
        committed = data["phase"] == "committed" and all(
            item["target"].is_dir() and not item["target"].is_symlink()
            for item in items
        )
        if committed:
            for item in items:
                _remove_transaction_tree(item["stage"])
                _remove_transaction_tree(item["backup"])
        else:
            for item in reversed(items):
                target = item["target"]
                backup = item["backup"]
                if backup.exists():
                    _remove_transaction_tree(target)
                    backup.rename(target)
                    fsync_directory(target.parent)
                    fsync_directory(backup.parent)
                elif data["phase"] in {"backed_up", "committed"}:
                    if item["had_existing"]:
                        raise DrowseArchiveError(
                            f"transaction {data['token']} lost backup for {target}"
                        )
                    _remove_transaction_tree(target)
                _remove_transaction_tree(item["stage"])
        journal.unlink(missing_ok=True)
        fsync_directory(journal.parent)


def _recover_legacy_transaction_orphans(root: Path) -> None:
    for artifact_root, metadata_name in (
        (root / "manifolds", "manifold.json"),
        (root / "templates", "template.json"),
    ):
        if not artifact_root.exists():
            continue
        if artifact_root.is_symlink() or not artifact_root.is_dir():
            raise DrowseArchiveError(
                f"Drowse archive artifact root {artifact_root} is not a regular directory"
            )
        for namespace in sorted(artifact_root.iterdir()):
            if not namespace.is_dir() or namespace.is_symlink():
                continue
            for candidate in sorted(namespace.iterdir()):
                match = _LEGACY_TRANSACTION_RE.fullmatch(candidate.name)
                if match is None:
                    continue
                if candidate.is_symlink() or not candidate.is_dir():
                    raise DrowseArchiveError(
                        f"legacy transaction path {candidate} is not a directory"
                    )
                metadata = candidate / metadata_name
                if not metadata.is_file() or metadata.is_symlink():
                    continue
                try:
                    payload = json.loads(_read_limited(
                        metadata,
                        maximum=_METADATA_JSON_MAX_BYTES,
                        label=f"legacy transaction metadata {metadata}",
                    ))
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict) or payload.get("name") != match["name"]:
                    continue
                target = candidate.with_name(match["name"])
                lock_suffix = (
                    "manifest" if metadata_name == "manifold.json" else "template"
                )
                with artifact_lock(target.parent / f"{target.name}.{lock_suffix}"):
                    if match["kind"] == "bak" and not target.exists():
                        candidate.rename(target)
                        fsync_directory(target.parent)
                    else:
                        _remove_transaction_tree(candidate)
                        fsync_directory(candidate.parent)


def _recover_transaction_orphans(root: Path) -> None:
    directory = _transaction_dir(root)
    if directory.exists():
        stage_pattern = re.compile(
            r"^[0-9a-f]{32}\.(?:manifold|template)\.stage$"
        )
        backup_pattern = re.compile(
            r"^[0-9a-f]{32}\.(?:manifold|template)\.backup$"
        )
        for path in sorted(directory.iterdir()):
            if stage_pattern.fullmatch(path.name):
                _remove_transaction_tree(path)
            elif backup_pattern.fullmatch(path.name):
                raise DrowseArchiveError(
                    f"orphaned transaction backup {path} has no recovery journal"
                )
            elif re.fullmatch(r"[0-9a-f]{32}\.json\.tmp", path.name):
                if path.is_dir() or path.is_symlink():
                    raise DrowseArchiveError(
                        f"transaction temporary path {path} is not a regular file"
                    )
                path.unlink(missing_ok=True)
        fsync_directory(directory)
    _recover_legacy_transaction_orphans(root)


def _recover_transactions_locked(root: Path) -> None:
    directory = _transaction_dir(root)
    if directory.exists():
        for journal in sorted(directory.glob("*.json")):
            _recover_transaction_journal(root, journal)
    _recover_transaction_orphans(root)


@contextmanager
def _transaction_guard(root: Path) -> Iterator[None]:
    root = root.resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    _validate_transaction_roots(root)
    with artifact_lock(root / ".drowse-transactions"):
        _validate_transaction_roots(root)
        _recover_transactions_locked(root)
        yield


def recover_drowse_archive_transactions(root: Path | None = None) -> None:
    """Recover crash-left multi-artifact install journals under ``root``."""
    with _transaction_guard(Path(root) if root is not None else drowse_home()):
        pass


def _transaction_items(
    root: Path,
    staged: list[tuple[Path, Path]],
    *,
    token: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, (stage, target) in enumerate(staged):
        kind = "manifold" if index == 0 else "template"
        if target.is_symlink() or target.exists() and not target.is_dir():
            raise DrowseArchiveError(
                f"Drowse archive {kind} target {target} is not a regular directory"
            )
        expected_stage = _transaction_stage_path(root, token, kind)
        if stage.resolve(strict=False) != expected_stage.resolve(strict=False):
            raise DrowseArchiveError(
                f"Drowse archive {kind} stage must be outside live artifact namespaces"
            )
        backup = _transaction_backup_path(root, token, kind)
        items.append({
            "kind": kind,
            "stage": _relative_transaction_path(root, stage),
            "target": _relative_transaction_path(root, target),
            "backup": _relative_transaction_path(root, backup),
            "had_existing": target.exists(),
        })
    return items


def _begin_transaction(
    staged: list[tuple[Path, Path]], *, token: str, root: Path,
) -> Path:
    items = _transaction_items(root, staged, token=token)
    journal = _transaction_dir(root) / f"{token}.json"
    if journal.exists():
        raise DrowseArchiveError(f"Drowse archive transaction {token} already exists")
    for item in items:
        _remove_transaction_tree(_resolve_transaction_path(root, item["stage"]))
        _remove_transaction_tree(_resolve_transaction_path(root, item["backup"]))
    _write_transaction_journal(journal, {
        "format_version": _TRANSACTION_FORMAT_VERSION,
        "token": token,
        "phase": "staging",
        "items": items,
    })
    return journal


def _promote_transaction(
    staged: list[tuple[Path, Path]], *, token: str,
    journal_root: Path | None = None, transaction_locked: bool = False,
) -> None:
    if not staged or len(staged) > 2:
        raise DrowseArchiveError("Drowse archive transaction needs one or two targets")
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise DrowseArchiveError("Drowse archive transaction token is invalid")
    if journal_root is None:
        roots: set[Path] = set()
        for _stage, target in staged:
            resolved_target = target.resolve(strict=False)
            if len(resolved_target.parents) < 3:
                raise DrowseArchiveError("Drowse archive transaction target is too shallow")
            roots.add(resolved_target.parents[2])
        if len(roots) != 1:
            raise DrowseArchiveError("Drowse archive transaction targets have different roots")
        journal_root = roots.pop()
    root = journal_root.resolve(strict=False)
    if not transaction_locked:
        with _transaction_guard(root):
            _promote_transaction(
                staged, token=token, journal_root=root, transaction_locked=True,
            )
        return
    journal = _transaction_dir(root) / f"{token}.json"
    if not journal.exists():
        raise DrowseArchiveError(
            f"Drowse archive transaction {token} has no durable staging journal"
        )
    payload, loaded_items = _load_transaction_journal(root, journal)
    expected_items = _transaction_items(root, staged, token=token)
    comparable = [
        {
            "kind": item["kind"],
            "stage": _relative_transaction_path(root, item["stage"]),
            "target": _relative_transaction_path(root, item["target"]),
            "backup": _relative_transaction_path(root, item["backup"]),
            "had_existing": item["had_existing"],
        }
        for item in loaded_items
    ]
    if payload["phase"] != "staging" or comparable != expected_items:
        raise DrowseArchiveError(
            f"Drowse archive transaction {token} does not match its journal"
        )
    items = expected_items
    if any(
        (
            _resolve_transaction_path(root, item["stage"]).is_symlink()
            or not _resolve_transaction_path(root, item["stage"]).is_dir()
        )
        for item in items
    ):
        raise DrowseArchiveError("Drowse archive transaction has an incomplete stage")
    payload["phase"] = "prepared"
    _write_transaction_journal(journal, payload)
    try:
        for item in items:
            target = _resolve_transaction_path(root, item["target"])
            backup = _resolve_transaction_path(root, item["backup"])
            if item["had_existing"]:
                _remove_transaction_tree(backup)
                target.rename(backup)
                fsync_directory(target.parent)
                fsync_directory(backup.parent)
        payload["phase"] = "backed_up"
        _write_transaction_journal(journal, payload)
        for stage, target in staged:
            stage.rename(target)
            fsync_directory(target.parent)
            fsync_directory(stage.parent)
        payload["phase"] = "committed"
        _write_transaction_journal(journal, payload)
    except BaseException as exc:
        try:
            _recover_transaction_journal(root, journal)
        except BaseException as recovery_exc:
            raise DrowseArchiveError(
                f"Drowse archive promotion failed and recovery was incomplete "
                f"({recovery_exc})"
            ) from recovery_exc
        if isinstance(exc, OSError):
            raise DrowseArchiveError(
                f"could not promote Drowse archive transaction ({exc})"
            ) from exc
        raise
    _recover_transaction_journal(root, journal)


def install_drowse_archive(
    archive_path: str | Path,
    as_: str | None = None,
    *,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Verify, stage, and transactionally install a local ``.drowse``."""
    from drowse.io.hf_manifolds import (
        ManifoldInstallConflict,
        _rewrite_staged_manifold_name,
    )
    from drowse.io.selectors import invalidate as invalidate_selector_index

    source_archive = Path(archive_path).expanduser()
    if not source_archive.is_file():
        raise FileNotFoundError(f"Drowse archive not found: {source_archive}")
    if source_archive.suffix.lower() not in {".drowse", ".polythetic", ".saklaspack"}:
        raise DrowseArchiveError("archive path must end in .drowse")
    _report(on_progress, f"Validating {source_archive.name}...")
    root = drowse_home()
    root.mkdir(parents=True, exist_ok=True)
    try:
        archive = zipfile.ZipFile(source_archive, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise DrowseArchiveError(f"could not open {source_archive} ({exc})") from exc
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
        primary, source_identity, template, template_identity, files = (
            _validate_pack_manifest(data, infos)
        )
        for name, record in files.items():
            _stream_entry(archive, infos[name], expected=record)
        with tempfile.TemporaryDirectory(prefix=".drowse-", dir=root) as temp:
            extracted = Path(temp)
            for name, record in files.items():
                output = extracted.joinpath(*PurePosixPath(name).parts)
                _stream_entry(archive, infos[name], expected=record, output=output)
            folder, loaded_template = _validate_extracted_closure(
                extracted,
                primary,
                source_identity,
                template,
                template_identity,
                data["source"],
            )

            src_ns, src_name = source_identity
            if as_ is None:
                dst_ns, dst_name = src_ns, src_name
            else:
                parts = as_.split("/")
                if len(parts) != 2 or any(
                    not NAME_REGEX.fullmatch(component) for component in parts
                ):
                    raise DrowseArchiveError(
                        f"as_ must be '<namespace>/<name>' with both components "
                        f"matching {NAME_REGEX.pattern}"
                    )
                dst_ns, dst_name = parts
                _validate_manifold_identity(dst_ns, dst_name)
            dst = manifold_dir(dst_ns, dst_name)
            tmpl_dst = (
                template_dir(*template_identity)
                if template_identity is not None else None
            )
            token = uuid.uuid4().hex
            manifold_stage = _transaction_stage_path(root, token, "manifold")
            template_stage = (
                _transaction_stage_path(root, token, "template")
                if tmpl_dst is not None else None
            )
            with _transaction_guard(root):
                with _closure_locks(dst, tmpl_dst):
                    if dst.exists() and not force:
                        raise ManifoldInstallConflict(
                            f"{dst} already exists; pass force=True or "
                            "as_=<ns>/<name>"
                        )
                    install_template = False
                    if tmpl_dst is not None and loaded_template is not None:
                        if tmpl_dst.is_symlink() or (
                            tmpl_dst.exists() and not tmpl_dst.is_dir()
                        ):
                            raise DrowseArchiveError(
                                f"template target {tmpl_dst} is not a regular directory"
                            )
                        source_template = loaded_template.path
                        assert source_template is not None
                        if tmpl_dst.exists() and _same_template(
                            source_template, tmpl_dst,
                        ):
                            install_template = False
                        elif tmpl_dst.exists() and not force:
                            raise ManifoldInstallConflict(
                                f"{tmpl_dst} already contains a different "
                                "referenced template; pass force=True to "
                                "replace the closure"
                            )
                        else:
                            if tmpl_dst.exists():
                                assert template_identity is not None
                                template_ref = "/".join(template_identity)
                                dependents = _other_template_dependents(
                                    root,
                                    template_ref,
                                    excluding=dst,
                                )
                                if dependents:
                                    joined = ", ".join(dependents)
                                    raise ManifoldInstallConflict(
                                        f"cannot replace shared template "
                                        f"{template_ref}; referenced by {joined}"
                                    )
                            install_template = True
                    staged: list[tuple[Path, Path]] = [(manifold_stage, dst)]
                    if install_template:
                        assert template_stage is not None
                        assert tmpl_dst is not None
                        staged.append((template_stage, tmpl_dst))
                    journal = _begin_transaction(staged, token=token, root=root)
                    try:
                        _report(on_progress, f"Staging {src_ns}/{src_name}...")
                        _copy_tree(folder.folder, manifold_stage)
                        _reset_installed_manifold_transaction_state(manifold_stage)
                        staged_folder = ManifoldFolder.load(manifold_stage)
                        if as_ is not None:
                            rewritten_files = _rewrite_staged_manifold_name(
                                staged_folder, dst_name,
                            )
                            staged_folder.write_metadata(files=rewritten_files)
                            staged_folder = ManifoldFolder.load(manifold_stage)
                        template_sha256 = (
                            loaded_template.sha256()
                            if loaded_template is not None else None
                        )
                        for tensor in _fitted_tensor_paths(staged_folder):
                            _validate_fitted_tensor(
                                tensor,
                                staged_folder,
                                resolved_template_sha256=template_sha256,
                            )
                        if install_template:
                            assert template_stage is not None
                            assert loaded_template is not None
                            assert loaded_template.path is not None
                            _copy_tree(loaded_template.path, template_stage)
                            staged_template = TemplateFolder.load(template_stage)
                            assert template_identity is not None
                            if staged_template.name != template_identity[1]:
                                raise DrowseArchiveError(
                                    "staged template identity changed during copy"
                                )
                        _report(on_progress, f"Installing {src_ns}/{src_name}...")
                        _promote_transaction(
                            staged,
                            token=token,
                            journal_root=root,
                            transaction_locked=True,
                        )
                    except BaseException:
                        if journal.exists():
                            _recover_transaction_journal(root, journal)
                        raise
    invalidate_selector_index()
    return dst
