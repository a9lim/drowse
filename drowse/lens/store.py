"""Drowse-fitted lens artifacts (format v7): save/load under ``models/<safe_id>/lens/``.

A lens is a per-model transport — one matrix per layer for each fitted site —
not a per-concept artifact.  Lenses fitted by Drowse are owned by Drowse and
live in the model's lens source family (:data:`~drowse.lens.sources.LENS_SOURCES`)::

    ~/.drowse/models/<safe_model_id>/lens/local/<kind>[-<tag>]/
      manifest.json                              # atomic shard pointer
      checkpoint.json                            # only during a resumable fit
      <site>.layer-<L>.gen-<id>.safetensors      # immutable fp32 layer shards
      <site>.partial.layer-<L>.gen-<id>.safetensors

One artifact is one estimator run: one :class:`~drowse.lens.key.LensKey` kind
with one or more sites.  The manifest names the kind, and per site the
layers, shard files, and per-shard sha256::

    {"format_version": 7, "kind": "R",
     "sites": {"out": {"layers": [...], "tensor_files": {...}, "tensor_sha256": {...}}},
     "estimator_policy": {"direction", "rules", "skip_first_positions",
                          "corpus_preprocess_version", "doc_chars"},
     "n_prompts", "d_model", "dtype", <provenance>, "checkpoint", "consumed_prefix_sha256"}

The directory name carries the kind (``local:r`` is always an R-lens), so a
fit can never land in, or resume from, another kind's shards.  fp32 on disk
and in memory keeps resumed fits numerically identical to uninterrupted ones.
The token-id corpus sha256 and loaded-model fingerprint make a different
corpus or mutable model revision read as stale; ``n_prompts`` is what lets an
interrupted fit resume (a checkpoint is the estimator's raw sums, averaged
stripe by stripe while it streams, so it is self-contained).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import torch
from safetensors import safe_open

from drowse.base.atomic import artifact_lock, fsync_directory, write_json_atomic
from drowse.base.shards import (
    fit_lock,
    generation_path as _new_layer_generation,
    json_pointer_matches as _json_pointer_matches,
    shard_paths,
)
from drowse.base.source_registry import LOCAL, SourceRef
from drowse.lens.corpus import LENS_CORPUS_PREPROCESS_VERSION, LENS_DOC_CHARS
from drowse.lens.jlens import SKIP_FIRST_POSITIONS, JacobianLens
from drowse.lens.key import LENS_KINDS, LENS_SITES, LensKey, local_lens_kind

log = logging.getLogger(__name__)

LENS_FORMAT_VERSION = 7
# Artifact label for the shared shard primitive's error/log wording.
_LENS_LABEL = "lens"
#: Stable lock anchors (never payload files): the durable pointer's and the
#: checkpoint's.  Payload shards are named per site.
_LOCK_STEM = "lens"
_CHECKPOINT_LOCK_STEM = "lens.partial"

_PROVENANCE_FIELDS = (
    "corpus_spec", "corpus_sha256", "corpus_hash_kind", "seq_len", "dim_batch",
    "raw_corpus_sha256", "raw_prompt_count", "usable_prompt_count",
    "model_layer_count", "model_fingerprint", "model_source_fingerprint",
)
_MANIFEST_FIELDS = frozenset({
    "format_version", "kind", "sites", "estimator_policy", "n_prompts",
    "d_model", "dtype", "checkpoint", "consumed_prefix_sha256",
    *_PROVENANCE_FIELDS,
})
_SITE_FIELDS = frozenset({"layers", "tensor_files", "tensor_sha256"})


@dataclass(frozen=True)
class LensProvenance:
    """What a fit was run on — recorded verbatim in every manifest it writes.

    Built once per fit and handed to every save: the durable artifact, each
    checkpoint, and a top-up all record the same identity.  ``skip_first``
    is the estimator's attention-sink cut; it lands in the estimator policy,
    so a fit at another value never resumes or reuses this one's shards.
    """

    corpus_spec: str
    corpus_sha256: str
    seq_len: int
    dim_batch: int
    corpus_hash_kind: str = "token_ids_v1"
    skip_first: int = SKIP_FIRST_POSITIONS
    raw_corpus_sha256: str | None = None
    raw_prompt_count: int | None = None
    usable_prompt_count: int | None = None
    model_layer_count: int | None = None
    model_fingerprint: str | None = None
    model_source_fingerprint: str | None = None

    def manifest_fields(self) -> dict[str, Any]:
        fields = asdict(self)
        del fields["skip_first"]
        return fields


def lens_estimator_policy(
    kind: str, *, skip_first: int = SKIP_FIRST_POSITIONS,
) -> dict[str, Any]:
    """Fit semantics that must match before any final/checkpoint reuse.

    The kind fixes the direction and the backward rules; the rule *set* is
    recorded too, so a RelP rule change reads as a different estimator.
    """
    key = LensKey(kind)
    rules: dict[str, Any] | None = None
    if key.relp:
        from drowse.lens.relp import RELP_RULES

        rules = dict(RELP_RULES)
    return {
        "direction": key.side,
        "rules": rules,
        "skip_first_positions": int(skip_first),
        "corpus_preprocess_version": LENS_CORPUS_PREPROCESS_VERSION,
        "doc_chars": LENS_DOC_CHARS,
    }


def lens_layers(sidecar: Mapping[str, Any], site: str = "out") -> list[int]:
    """The fitted layers of one site of a validated lens manifest/descriptor."""
    sites = sidecar.get("sites")
    if not isinstance(sites, Mapping) or not isinstance(sites.get(site), Mapping):
        return []
    return [int(layer) for layer in sites[site].get("layers", [])]


def lens_identity(sidecar: Mapping[str, Any] | None) -> str | None:
    """Stable identity binding a resident lens to its artifact pointer.

    The digest of the canonical manifest (or external descriptor): any
    republication — a re-fit, a top-up, another source — changes it.
    """
    if sidecar is None:
        return None
    canonical = json.dumps(
        dict(sidecar), sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# -- transaction proofs -----------------------------------------------------

@dataclass(frozen=True)
class _LensPayloadProof:
    """Transaction-scoped proof that one immutable pointer was byte-verified.

    Constructed only after a successful load or pointer publication.  Callers
    may reuse it while holding ``lens_fit_lock``; public/unverified callers
    pass no proof and retain full digest verification.
    """

    anchor: str
    pointer_sha256: str


def _lens_payload_proof(
    anchor: Path, sidecar: Mapping[str, Any],
) -> _LensPayloadProof:
    return _LensPayloadProof(
        anchor=str(anchor.resolve(strict=False)),
        pointer_sha256=cast(str, lens_identity(sidecar)),
    )


def _proof_matches(
    proof: _LensPayloadProof | None,
    anchor: Path,
    sidecar: Mapping[str, Any],
) -> bool:
    return proof == _lens_payload_proof(anchor, sidecar) if proof is not None else False


# -- paths ------------------------------------------------------------------

def _local_dir(model_id: str, name: str) -> Path:
    from drowse.lens.sources import LENS_SOURCES

    return LENS_SOURCES.local_dir(model_id, name)


def _anchor(directory: Path, *, checkpoint: bool) -> Path:
    stem = _CHECKPOINT_LOCK_STEM if checkpoint else _LOCK_STEM
    return directory / f"{stem}.safetensors"


def _pointer(directory: Path, *, checkpoint: bool) -> Path:
    return directory / ("checkpoint.json" if checkpoint else "manifest.json")


def _site_anchor(directory: Path, site: str, *, checkpoint: bool) -> Path:
    """The per-site stem the immutable generations are named after."""
    stem = f"{site}.partial" if checkpoint else site
    return directory / f"{stem}.safetensors"


def lens_manifest_path(
    model_id: str, name: str, *, checkpoint: bool = False,
) -> Path:
    """The durable (or checkpoint) pointer of a local lens."""
    return _pointer(_local_dir(model_id, name), checkpoint=checkpoint)


def _site_tensor_paths(
    directory: Path,
    sidecar: Mapping[str, Any],
    site: str,
    layers: list[int],
    *,
    require_exact_keys: bool = True,
) -> dict[int, Path]:
    return shard_paths(
        directory / "pointer",
        sidecar["sites"][site],
        layers,
        label=_LENS_LABEL,
        require_exact_keys=require_exact_keys,
    )


def lens_artifact_size(
    model_id: str, *, name: str, checkpoint: bool = False,
) -> int:
    """Total payload bytes named by the current pointer."""
    directory = _local_dir(model_id, name)
    anchor = _anchor(directory, checkpoint=checkpoint)
    with artifact_lock(anchor):
        current = _load_sidecar_at(
            model_id, directory, _pointer(directory, checkpoint=checkpoint),
            label="lens checkpoint" if checkpoint else "lens",
            checkpoint=checkpoint, name=name,
        )
        if current is None:
            return 0
        paths: set[Path] = set()
        for site in current["sites"]:
            paths.update(_site_tensor_paths(
                directory, current, site, lens_layers(current, site),
            ).values())
        return sum(path.stat().st_size for path in paths)


def lens_payloads_match(
    model_id: str, sidecar: Mapping[str, Any], *, name: str,
) -> bool:
    """Validate exact payload digests without materializing fp32 matrices."""
    from drowse.base.integrity import hash_file

    directory = _local_dir(model_id, name)
    try:
        for site, entry in sidecar["sites"].items():
            expected = entry.get("tensor_sha256")
            if not isinstance(expected, Mapping):
                return False
            paths = _site_tensor_paths(
                directory, sidecar, site, lens_layers(sidecar, site),
            )
            if not all(
                isinstance(expected.get(str(layer)), str)
                and hash_file(path) == expected[str(layer)]
                for layer, path in paths.items()
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


@contextmanager
def lens_fit_lock(model_id: str, name: str):
    """Serialize the complete per-artifact fit/lifecycle transaction cross-process."""
    with fit_lock(_anchor(_local_dir(model_id, name), checkpoint=False)):
        yield


# -- generation GC ----------------------------------------------------------

def _referenced_tensor_names(directory: Path) -> set[str] | None:
    """Read both pointers, failing closed if an existing pointer is unreadable."""
    out: set[str] = set()
    for sidecar_path in (
        _pointer(directory, checkpoint=False),
        _pointer(directory, checkpoint=True),
    ):
        if not sidecar_path.exists():
            continue
        try:
            with open(sidecar_path) as handle:
                sidecar = json.load(handle)
            sites = sidecar.get("sites") if isinstance(sidecar, dict) else None
            if not isinstance(sites, Mapping):
                return None
            for entry in sites.values():
                files = entry.get("tensor_files") if isinstance(entry, Mapping) else None
                if not isinstance(files, Mapping):
                    return None
                for filename in files.values():
                    if isinstance(filename, str) and Path(filename).name == filename:
                        out.add(filename)
        except (OSError, TypeError, json.JSONDecodeError):
            return None
    return out


def _cleanup_unreferenced_generations(directory: Path) -> None:
    """Remove unreferenced generations and crash-left streaming temporaries."""
    keep = _referenced_tensor_names(directory)
    if keep is None:
        # A transient pointer read failure cannot authorize deletion of any
        # potentially live multi-GiB generation.
        return
    for path in (
        *directory.glob("*.layer-*.gen-*.safetensors"),
        *directory.glob("*.layer-*.gen-*.safetensors.tmp"),
    ):
        if path.name not in keep:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                # Pointer publication is already complete. Generation GC is
                # opportunistic and must not turn a valid fit into a failure.
                log.warning("could not remove old lens generation %s: %s", path, exc)


def cleanup_lens_artifacts(model_id: str, name: str) -> None:
    """Reap crash-left lens shards before a new fit allocates replacements."""
    with lens_fit_lock(model_id, name):
        _cleanup_unreferenced_generations(_local_dir(model_id, name))


# -- validation -------------------------------------------------------------

def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_count(value: Any, *, minimum: int = 0) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _validate_manifest(
    sidecar: Any, *, checkpoint: bool, name: str | None,
) -> dict[str, Any]:
    """Exact v7 schema checks that need no payload bytes; raises ``ValueError``."""
    if not isinstance(sidecar, dict) or set(sidecar) != _MANIFEST_FIELDS:
        raise ValueError("lens manifest does not match the v7 schema")
    if sidecar["format_version"] != LENS_FORMAT_VERSION:
        raise ValueError(
            f"lens manifest has format_version {sidecar['format_version']!r} "
            f"(need exactly {LENS_FORMAT_VERSION})"
        )
    kind = sidecar["kind"]
    if kind not in LENS_KINDS:
        raise ValueError(f"lens manifest has unknown kind {kind!r}")
    if name is not None and local_lens_kind(name) != kind:
        raise ValueError(f"local:{name} holds a {kind} lens")
    policy = sidecar["estimator_policy"]
    skip_first = policy.get("skip_first_positions") if isinstance(policy, dict) else None
    if not _is_count(skip_first) or policy != lens_estimator_policy(
        kind, skip_first=cast(int, skip_first),
    ):
        raise ValueError("lens manifest has a different estimator policy")
    if (
        sidecar["dtype"] != "float32"
        or not _is_count(sidecar["d_model"], minimum=1)
        or not _is_count(sidecar["n_prompts"])
        or not _is_count(sidecar["seq_len"])
        or not _is_count(sidecar["dim_batch"])
        or not isinstance(sidecar["checkpoint"], bool)
        or sidecar["checkpoint"] != checkpoint
        or any(
            not isinstance(sidecar[key], str) or not sidecar[key]
            for key in ("corpus_spec", "corpus_sha256", "corpus_hash_kind")
        )
    ):
        raise ValueError("lens manifest has invalid shape metadata")
    for key in (
        "raw_corpus_sha256", "model_fingerprint", "model_source_fingerprint",
        "consumed_prefix_sha256",
    ):
        if sidecar[key] is not None and (
            not isinstance(sidecar[key], str) or not sidecar[key]
        ):
            raise ValueError(f"lens manifest has invalid {key}")
    for key in ("corpus_sha256", "raw_corpus_sha256", "consumed_prefix_sha256"):
        if sidecar[key] is not None and not _is_digest(sidecar[key]):
            raise ValueError(f"lens manifest has an invalid {key} digest")
    for key in ("raw_prompt_count", "usable_prompt_count", "model_layer_count"):
        if sidecar[key] is not None and not _is_count(sidecar[key]):
            raise ValueError(f"lens manifest has invalid {key}")
    if checkpoint:
        if sidecar["n_prompts"] <= 0 or sidecar["consumed_prefix_sha256"] is None:
            raise ValueError("lens checkpoint is missing consumed-prefix identity")
    elif sidecar["consumed_prefix_sha256"] is not None:
        raise ValueError("final lens carries checkpoint-only metadata")
    sites = sidecar["sites"]
    if not isinstance(sites, dict) or not sites or set(sites) - set(LENS_SITES):
        raise ValueError("lens manifest has invalid sites")
    for site, entry in sites.items():
        LensKey(kind, site)
        if not isinstance(entry, dict) or set(entry) != _SITE_FIELDS:
            raise ValueError(f"lens site {site!r} has an invalid schema")
        layers = entry["layers"]
        if (
            not isinstance(layers, list) or not layers
            or any(not _is_count(layer) for layer in layers)
            or layers != sorted(set(layers))
        ):
            raise ValueError(f"lens site {site!r} has invalid layers")
        digests = entry["tensor_sha256"]
        if (
            not isinstance(digests, dict)
            or set(digests) != {str(layer) for layer in layers}
            or not all(_is_digest(digest) for digest in digests.values())
        ):
            raise ValueError(f"lens site {site!r} digest keys do not match its layers")
    return sidecar


def _load_sidecar_at(
    model_id: str, directory: Path, sc_path: Path, *, label: str,
    checkpoint: bool, name: str | None,
) -> dict[str, Any] | None:
    """A validated pointer whose every shard exists with the right header.

    Self-healing like the neutral-activation cache: any deviation logs a
    warning and reads as "no lens" — the caller decides whether to error or
    re-fit — rather than crashing the session.
    """
    if not sc_path.exists():
        return None
    try:
        with open(sc_path) as handle:
            sidecar = _validate_manifest(
                json.load(handle), checkpoint=checkpoint, name=name,
            )
        d_model = sidecar["d_model"]
        for site in sidecar["sites"]:
            layers = lens_layers(sidecar, site)
            for layer, path in _site_tensor_paths(
                directory, sidecar, site, layers,
            ).items():
                if not path.exists():
                    raise ValueError(f"missing tensor generation {path.name}")
                with safe_open(str(path), framework="pt", device="cpu") as tensors:
                    key = f"layer_{layer}"
                    if list(tensors.keys()) != [key]:
                        raise ValueError("tensor layer keys do not match the manifest")
                    view = tensors.get_slice(key)
                    if tuple(view.get_shape()) != (d_model, d_model):
                        raise ValueError(f"{site} {key} shape does not match d_model")
                    if view.get_dtype() != "F32":
                        raise ValueError(f"{site} {key} is not float32")
        return sidecar
    except Exception as exc:
        log.warning(
            "ignoring %s for %s (%s) — re-fit with `drowse lens fit`",
            label, model_id, exc,
        )
        return None


def load_local_lens_sidecar(model_id: str, name: str) -> dict[str, Any] | None:
    """A Drowse-owned lens's validated manifest, without its matrices."""
    directory = _local_dir(model_id, name)
    with artifact_lock(_anchor(directory, checkpoint=False)):
        return _load_sidecar_at(
            model_id, directory, _pointer(directory, checkpoint=False),
            label="lens", checkpoint=False, name=name,
        )


def load_lens_checkpoint_sidecar(model_id: str, name: str) -> dict[str, Any] | None:
    """Validated checkpoint metadata without materializing its matrices."""
    directory = _local_dir(model_id, name)
    with artifact_lock(_anchor(directory, checkpoint=True)):
        return _load_sidecar_at(
            model_id, directory, _pointer(directory, checkpoint=True),
            label="lens checkpoint", checkpoint=True, name=name,
        )


def load_lens_sidecar(model_id: str) -> dict[str, Any] | None:
    """Metadata for the active output-side lens source (local or external)."""
    from drowse.lens.sources import LENS_SOURCES, load_external_lens_sidecar

    active = LENS_SOURCES.active(model_id, "output")
    if active is None:
        return None
    if active.local:
        return load_local_lens_sidecar(model_id, active.name)
    return load_external_lens_sidecar(model_id, active.name)


# -- save -------------------------------------------------------------------

def save_lens(
    lens: JacobianLens,
    model_id: str,
    provenance: LensProvenance,
    *,
    name: str | None = None,
    durable: bool = True,
    reuse_layers: Iterable[int] | None = None,
    _verified_reuse_proof: _LensPayloadProof | None = None,
) -> Path:
    """Persist a fitted lens and atomically point at its layer generations.

    ``name`` defaults to the lens kind's directory (``R`` → ``local:r``); a
    tagged name must carry the same kind.  ``reuse_layers`` names matrices
    known to be byte-identical to the current durable artifact (the
    missing-layer top-up path): their shards are carried forward, every
    other layer is written anew.  Publication makes the lens active.
    """
    from drowse.lens.sources import LENS_SOURCES

    name = lens.key.kind.lower() if name is None else name
    if local_lens_kind(name) != lens.key.kind:
        raise ValueError(f"cannot save a {lens.key.kind} lens as local:{name}")
    directory = _local_dir(model_id, name)
    with lens_fit_lock(model_id, name):
        with artifact_lock(_anchor(directory, checkpoint=False)):
            path = _save_components(
                lens.jacobians, lens.n_prompts, lens.d_model, directory,
                key=lens.key, provenance=provenance, durable=durable,
                checkpoint=False, name=name, reuse_layers=reuse_layers,
                verified_reuse_proof=_verified_reuse_proof,
            )
    LENS_SOURCES.select(model_id, SourceRef(LOCAL, name))
    return path


def save_lens_at(
    lens: JacobianLens, directory: Path, provenance: LensProvenance,
) -> Path:
    """Write a lens artifact into an arbitrary directory (no selection).

    The packaging path for a lens that ships outside ``DROWSE_HOME`` — the
    same v7 manifest and shards :func:`save_lens` publishes.
    """
    with artifact_lock(_anchor(directory, checkpoint=False)):
        return _save_components(
            lens.jacobians, lens.n_prompts, lens.d_model, directory,
            key=lens.key, provenance=provenance, durable=True,
            checkpoint=False, name=None,
        )


def save_lens_checkpoint(
    sums: Mapping[int, torch.Tensor],
    n_prompts: int,
    d_model: int,
    model_id: str,
    provenance: LensProvenance,
    *,
    kind: str,
    consumed_prefix_sha256: str,
    name: str | None = None,
    _proof_out: list[_LensPayloadProof] | None = None,
) -> Path:
    """Write a self-contained checkpoint directly from raw estimator sums.

    Averaging happens one stripe at a time while the fp32 rows stream, so
    checkpointing never holds a second full lens beside the estimator.
    """
    name = kind.lower() if name is None else name
    if local_lens_kind(name) != kind:
        raise ValueError(f"cannot checkpoint a {kind} lens as local:{name}")
    directory = _local_dir(model_id, name)
    with lens_fit_lock(model_id, name), artifact_lock(_anchor(directory, checkpoint=True)):
        return _save_components(
            sums, n_prompts, d_model, directory,
            key=LensKey(kind), provenance=provenance, durable=True,
            checkpoint=True, name=name, raw_sum_count=int(n_prompts),
            consumed_prefix_sha256=consumed_prefix_sha256,
            proof_out=_proof_out,
        )


def _save_components(
    matrices: Mapping[int, torch.Tensor],
    n_prompts: int,
    d_model: int,
    directory: Path,
    *,
    key: LensKey,
    provenance: LensProvenance,
    durable: bool,
    checkpoint: bool,
    name: str | None,
    raw_sum_count: int | None = None,
    reuse_layers: Iterable[int] | None = None,
    verified_reuse_proof: _LensPayloadProof | None = None,
    consumed_prefix_sha256: str | None = None,
    proof_out: list[_LensPayloadProof] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    sc_path = _pointer(directory, checkpoint=checkpoint)
    site_anchor = _site_anchor(directory, key.site, checkpoint=checkpoint)
    layer_ids = sorted(int(idx) for idx in matrices)
    policy = lens_estimator_policy(key.kind, skip_first=provenance.skip_first)

    def _rows(idx: int, start: int, end: int) -> torch.Tensor:
        value = matrices[idx][start:end].to(device="cpu", dtype=torch.float32)
        if raw_sum_count is not None:
            # Stripe-local averaging: checkpointing never clones a complete
            # fp32 layer or keeps a complete artifact mapping alive.
            value = value.clone().mul_(1.0 / max(int(raw_sum_count), 1))
        return value.contiguous()

    tensor_files: dict[str, str] = {}
    tensor_sha256: dict[str, str] = {}
    reuse_set = {int(layer) for layer in reuse_layers or ()}
    if reuse_set:
        current = _load_sidecar_at(
            "current", directory, sc_path, label="lens reuse source",
            checkpoint=checkpoint, name=name,
        )
        if (
            current is not None
            and current["kind"] == key.kind
            and key.site in current["sites"]
            and current["d_model"] == int(d_model)
            and current["n_prompts"] == int(n_prompts)
            and current["corpus_sha256"] == provenance.corpus_sha256
            and current["corpus_hash_kind"] == provenance.corpus_hash_kind
            and current["seq_len"] == int(provenance.seq_len)
            and current["model_fingerprint"] == provenance.model_fingerprint
            and current["estimator_policy"] == policy
        ):
            from drowse.base.integrity import hash_file

            current_layers = lens_layers(current, key.site)
            current_paths = _site_tensor_paths(
                directory, current, key.site, current_layers,
            )
            current_digests = current["sites"][key.site]["tensor_sha256"]
            pointer_verified = _proof_matches(
                verified_reuse_proof, _anchor(directory, checkpoint=checkpoint),
                current,
            )
            for layer in sorted(reuse_set & set(layer_ids) & set(current_layers)):
                digest = current_digests[str(layer)]
                path = current_paths[layer]
                try:
                    matches = path.exists() and (
                        pointer_verified or hash_file(path) == digest
                    )
                except OSError:
                    matches = False
                if matches:
                    tensor_files[str(layer)] = path.name
                    tensor_sha256[str(layer)] = digest
    created: list[Path] = []
    try:
        for layer in layer_ids:
            if str(layer) in tensor_files:
                continue
            generation_path = _new_layer_generation(site_anchor, layer)
            digest = _save_fp32_square_safetensors_atomic(
                generation_path, layer, d_model, _rows, durable=durable,
            )
            created.append(generation_path)
            tensor_files[str(layer)] = generation_path.name
            tensor_sha256[str(layer)] = digest
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    sidecar: dict[str, Any] = {
        "format_version": LENS_FORMAT_VERSION,
        "kind": key.kind,
        "sites": {
            key.site: {
                "layers": layer_ids,
                "tensor_files": tensor_files,
                "tensor_sha256": tensor_sha256,
            },
        },
        "estimator_policy": policy,
        "n_prompts": int(n_prompts),
        "d_model": int(d_model),
        "dtype": "float32",
        **provenance.manifest_fields(),
        "checkpoint": checkpoint,
        "consumed_prefix_sha256": consumed_prefix_sha256,
    }
    _validate_manifest(sidecar, checkpoint=checkpoint, name=name)
    try:
        # Every immutable layer generation is complete before this one atomic
        # pointer switch. A pointer failure leaves the prior pointer and its
        # shards untouched, and the failed new generations are discarded.
        fsync_directory(directory)
        write_json_atomic(sc_path, sidecar)
    except BaseException:
        # A signal may unwind just after the atomic replace. Never delete
        # immutable generations the live pointer already names.
        if not _json_pointer_matches(sc_path, sidecar):
            for path in created:
                path.unlink(missing_ok=True)
        raise
    # Make the new pointer rename durable before deleting any generation the
    # previous pointer named. Otherwise power loss can roll the directory back
    # to an old pointer whose shards GC already removed.
    fsync_directory(directory)
    if proof_out is not None:
        proof_out[:] = [
            _lens_payload_proof(_anchor(directory, checkpoint=checkpoint), sidecar),
        ]
    _cleanup_unreferenced_generations(directory)
    fsync_directory(directory)
    return sc_path


def _save_fp32_square_safetensors_atomic(
    path: Path,
    layer: int,
    d_model: int,
    rows: Callable[[int, int, int], torch.Tensor],
    *,
    durable: bool = True,
) -> str:
    """Stream one square fp32 layer tensor into an atomic safetensors file.

    ``safetensors.torch.save_file`` requires a complete tensor mapping, which
    would make checkpoint RSS include another full lens beside the fp32
    estimator.  The wire format is deliberately simple: an 8-byte
    little-endian JSON-header length, a space-padded JSON header, then the
    contiguous payload, written in 256-row fp32 stripes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    prefix, raw_header = _lens_safetensors_header(layer, d_model)
    digest = hashlib.sha256()
    try:
        with open(tmp, "wb", buffering=0) as f:
            f.write(prefix)
            digest.update(prefix)
            f.write(raw_header)
            digest.update(raw_header)
            for start in range(0, d_model, 256):
                block = rows(layer, start, min(start + 256, d_model))
                expected = (min(start + 256, d_model) - start, d_model)
                if block.dtype != torch.float32 or tuple(block.shape) != expected:
                    raise ValueError(
                        f"streamed layer {layer} rows have "
                        f"{tuple(block.shape)} {block.dtype}; expected "
                        f"{expected} float32"
                    )
                # Both file.write and hashlib accept the buffer protocol; keep
                # the stripe tensor alive and avoid a second payload-sized copy.
                payload = memoryview(cast(Any, block.numpy())).cast("B")
                f.write(payload)
                digest.update(payload)
            if durable:
                os.fsync(f.fileno())
    except BaseException:
        with suppress(FileNotFoundError):
            tmp.unlink()
        raise
    os.replace(tmp, path)
    return digest.hexdigest()


def _lens_safetensors_header(layer: int, d_model: int) -> tuple[bytes, bytes]:
    """Canonical writer header bytes, shared by save and load-time digest."""
    nbytes = int(d_model) * int(d_model) * 4
    raw_header = json.dumps(
        {
            f"layer_{int(layer)}": {
                "dtype": "F32",
                "shape": [int(d_model), int(d_model)],
                "data_offsets": [0, nbytes],
            },
        },
        separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    padded_len = (len(raw_header) + 7) // 8 * 8
    raw_header += b" " * (padded_len - len(raw_header))
    return struct.pack("<Q", padded_len), raw_header


# -- load -------------------------------------------------------------------

def load_local_lens(
    model_id: str, name: str,
) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Load a Drowse-owned lens's ``out`` site when usable, else ``None``."""
    loaded = _load_lens_verified(model_id, name=name)
    return None if loaded is None else (loaded[0], loaded[1])


def load_lens(model_id: str) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Load the active output-side lens source (local or external)."""
    from drowse.lens.sources import LENS_SOURCES, load_external_lens

    active = LENS_SOURCES.active(model_id, "output")
    if active is None:
        return None
    if active.local:
        return load_local_lens(model_id, active.name)
    return load_external_lens(model_id, active.name)


def _load_lens_verified(
    model_id: str,
    *,
    name: str,
    requested_layers: set[int] | None = None,
) -> tuple[JacobianLens, dict[str, Any], _LensPayloadProof | None] | None:
    """Load a durable lens plus a proof reusable under the fit transaction."""
    directory = _local_dir(model_id, name)
    anchor = _anchor(directory, checkpoint=False)
    with lens_fit_lock(model_id, name), artifact_lock(anchor):
        sidecar = _load_sidecar_at(
            model_id, directory, _pointer(directory, checkpoint=False),
            label="lens", checkpoint=False, name=name,
        )
        loaded = _load_lens_at(
            model_id, directory, sidecar, label="lens",
            requested_layers=requested_layers,
        )
        if loaded is None:
            return None
        lens, verified = loaded
        _cleanup_unreferenced_generations(directory)
        fully_verified = (
            requested_layers is None
            or requested_layers >= set(lens_layers(verified))
        )
        return lens, verified, (
            _lens_payload_proof(anchor, verified) if fully_verified else None
        )


def load_lens_checkpoint(
    model_id: str, name: str,
) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Load a resumable partial lens, or ``None`` when absent/unusable."""
    directory = _local_dir(model_id, name)
    with lens_fit_lock(model_id, name), artifact_lock(_anchor(directory, checkpoint=True)):
        sidecar = _load_sidecar_at(
            model_id, directory, _pointer(directory, checkpoint=True),
            label="lens checkpoint", checkpoint=True, name=name,
        )
        loaded = _load_lens_at(
            model_id, directory, sidecar, label="lens checkpoint",
        )
        if loaded is not None:
            _cleanup_unreferenced_generations(directory)
        return loaded


def _load_lens_at(
    model_id: str,
    directory: Path,
    sidecar: dict[str, Any] | None,
    *,
    label: str,
    requested_layers: set[int] | None = None,
    site: str = "out",
) -> tuple[JacobianLens, dict[str, Any]] | None:
    """Materialize one site's matrices, verifying each shard's digest."""
    if sidecar is None:
        return None
    try:
        if site not in sidecar["sites"]:
            raise ValueError(f"lens has no {site!r} site")
        d_model = int(sidecar["d_model"])
        durable_layers = lens_layers(sidecar, site)
        if requested_layers is not None and requested_layers - set(durable_layers):
            raise ValueError("requested layers are absent from lens")
        layers = (
            durable_layers if requested_layers is None
            else [layer for layer in durable_layers if layer in requested_layers]
        )
        if not layers:
            raise ValueError("requested layers are absent from lens")
        paths = _site_tensor_paths(
            directory, sidecar, site, layers,
            require_exact_keys=(layers == durable_layers),
        )
        digests = sidecar["sites"][site]["tensor_sha256"]
        jacobians: dict[int, torch.Tensor] = {}
        for layer in layers:
            prefix, raw_header = _lens_safetensors_header(layer, d_model)
            digest = hashlib.sha256(prefix)
            digest.update(raw_header)
            with safe_open(str(paths[layer]), framework="pt", device="cpu") as tensors:
                raw = tensors.get_tensor(f"layer_{layer}")
            if raw.dtype != torch.float32 or tuple(raw.shape) != (d_model, d_model):
                raise ValueError(f"layer {layer} has shape {tuple(raw.shape)}")
            for start in range(0, d_model, 256):
                block = raw[start:start + 256].contiguous()
                digest.update(memoryview(cast(Any, block.numpy())).cast("B"))
            if digest.hexdigest() != digests[str(layer)]:
                raise ValueError(f"layer {layer} failed tensor digest validation")
            if not bool(torch.isfinite(raw).all()):
                raise ValueError(f"layer {layer} contains non-finite values")
            jacobians[layer] = raw
        lens = JacobianLens(
            jacobians, n_prompts=int(sidecar["n_prompts"]), d_model=d_model,
            key=LensKey(sidecar["kind"], site),
        )
        return lens, sidecar
    except Exception as exc:
        log.warning(
            "ignoring %s for %s (%s) — re-fit with `drowse lens fit`",
            label, model_id, exc,
        )
        return None


# -- checkpoint promotion / reaping ----------------------------------------

def promote_lens_checkpoint(
    model_id: str,
    *,
    name: str,
    n_prompts: int,
    source_layers: list[int],
    corpus_sha256: str,
    corpus_hash_kind: str,
    seq_len: int,
    d_model: int,
    model_fingerprint: str,
    _verified_proof: _LensPayloadProof | None = None,
) -> bool:
    """Promote a complete checkpoint to the durable lens without rewriting it.

    Terminal checkpoint cadence can coincide with fit completion.  That shard
    set is already the complete fp32 artifact, so serializing the in-memory
    lens again would duplicate a potentially multi-GiB write.  Promote only
    after cheap exact identity checks; callers fall back to
    :func:`save_lens` on ``False``.
    """
    from drowse.lens.sources import LENS_SOURCES

    directory = _local_dir(model_id, name)
    checkpoint_anchor = _anchor(directory, checkpoint=True)
    final_anchor = _anchor(directory, checkpoint=False)
    checkpoint_sc = _pointer(directory, checkpoint=True)
    final_sc = _pointer(directory, checkpoint=False)
    with (
        lens_fit_lock(model_id, name),
        artifact_lock(checkpoint_anchor),
        artifact_lock(final_anchor),
    ):
        sidecar = _load_sidecar_at(
            model_id, directory, checkpoint_sc, label="lens checkpoint",
            checkpoint=True, name=name,
        )
        if (
            sidecar is None
            or sidecar["n_prompts"] != int(n_prompts)
            or lens_layers(sidecar) != sorted(int(layer) for layer in source_layers)
            or sidecar["corpus_sha256"] != corpus_sha256
            or sidecar["corpus_hash_kind"] != corpus_hash_kind
            or sidecar["seq_len"] != int(seq_len)
            or sidecar["d_model"] != int(d_model)
            or sidecar["model_fingerprint"] != model_fingerprint
        ):
            return False
        if not _proof_matches(
            _verified_proof, checkpoint_anchor, sidecar,
        ) and not lens_payloads_match(model_id, sidecar, name=name):
            return False
        # Re-fsync the checkpoint payloads before publishing the final pointer.
        for site in sidecar["sites"]:
            for path in set(_site_tensor_paths(
                directory, sidecar, site, lens_layers(sidecar, site),
            ).values()):
                with open(path, "rb") as handle:
                    os.fsync(handle.fileno())
        fsync_directory(directory)
        durable = {**sidecar, "checkpoint": False, "consumed_prefix_sha256": None}
        # Publish only the pointer. The immutable shards stay referenced by
        # the checkpoint until this succeeds, so any failure preserves both
        # the prior durable lens and the complete resumable checkpoint.
        write_json_atomic(final_sc, durable)
        fsync_directory(directory)
        checkpoint_sc.unlink(missing_ok=True)
        fsync_directory(directory)
        _cleanup_unreferenced_generations(directory)
        fsync_directory(directory)
    LENS_SOURCES.select(model_id, SourceRef(LOCAL, name))
    return True


def _final_lens_subsumes_checkpoint(
    final: Mapping[str, Any], checkpoint: Mapping[str, Any],
) -> bool:
    """Whether the durable pointer contains every result in ``checkpoint``."""
    semantic_keys = (
        "kind", "corpus_sha256", "corpus_hash_kind", "seq_len", "d_model",
        "model_fingerprint", "estimator_policy",
    )
    if any(final.get(key) != checkpoint.get(key) for key in semantic_keys):
        return False
    checkpoint_layers = set(lens_layers(checkpoint))
    return bool(
        checkpoint_layers
        and set(lens_layers(final)) >= checkpoint_layers
        and int(final["n_prompts"]) >= int(checkpoint["n_prompts"]) > 0
    )


def remove_subsumed_lens_checkpoint(
    model_id: str,
    *,
    name: str,
    verified_final_sidecar: Mapping[str, Any] | None = None,
) -> bool:
    """Reap a crash-left checkpoint only when the durable lens proves it redundant.

    Final publication and checkpoint unlink are two filesystem operations; a
    process death between them leaves two complete shard sets.  This keeps a
    farther-ahead or semantically different checkpoint and validates the
    final payload before discarding the recovery point.  A caller that just
    validated the exact current final sidecar may pass it to avoid hashing a
    multi-GiB lens twice.
    """
    directory = _local_dir(model_id, name)
    with (
        lens_fit_lock(model_id, name),
        artifact_lock(_anchor(directory, checkpoint=False)),
        artifact_lock(_anchor(directory, checkpoint=True)),
    ):
        checkpoint_sc = _pointer(directory, checkpoint=True)
        checkpoint = _load_sidecar_at(
            model_id, directory, checkpoint_sc, label="lens checkpoint",
            checkpoint=True, name=name,
        )
        if checkpoint is None:
            return False
        final = _load_sidecar_at(
            model_id, directory, _pointer(directory, checkpoint=False),
            label="lens", checkpoint=False, name=name,
        )
        if final is None or not _final_lens_subsumes_checkpoint(final, checkpoint):
            return False
        caller_verified = (
            verified_final_sidecar is not None
            and dict(verified_final_sidecar) == final
        )
        if not caller_verified and not lens_payloads_match(model_id, final, name=name):
            return False
        checkpoint_sc.unlink(missing_ok=True)
        # The pointer removal must be durable before any shard it named is
        # collected, or a crash can resurrect a pointer whose bytes are gone.
        fsync_directory(directory)
        _cleanup_unreferenced_generations(directory)
        fsync_directory(directory)
        return True


def remove_lens_checkpoint(model_id: str, name: str) -> bool:
    """Delete a resumable checkpoint. Returns True when anything was removed."""
    directory = _local_dir(model_id, name)
    with lens_fit_lock(model_id, name), artifact_lock(_anchor(directory, checkpoint=True)):
        before = set(directory.glob("*.safetensors"))
        checkpoint_sc = _pointer(directory, checkpoint=True)
        removed = checkpoint_sc.exists()
        checkpoint_sc.unlink(missing_ok=True)
        if directory.exists():
            fsync_directory(directory)
            _cleanup_unreferenced_generations(directory)
            fsync_directory(directory)
        return removed or before != set(directory.glob("*.safetensors"))


def remove_lens(model_id: str, name: str) -> bool:
    """Delete a local lens artifact. Returns True when anything was removed.

    The selection is unpublished only when it named exactly this lens.
    """
    from drowse.lens.sources import LENS_SOURCES

    directory = _local_dir(model_id, name)
    with (
        lens_fit_lock(model_id, name),
        artifact_lock(_anchor(directory, checkpoint=False)),
        artifact_lock(_anchor(directory, checkpoint=True)),
    ):
        pointers = [
            _pointer(directory, checkpoint=False),
            _pointer(directory, checkpoint=True),
        ]
        shards = [
            *directory.glob("*.layer-*.gen-*.safetensors"),
            *directory.glob("*.layer-*.gen-*.safetensors.tmp"),
        ]
        removed = any(path.exists() for path in (*pointers, *shards))
        # Remove the atomic pointers first: after this logical deletion, a
        # best-effort shard GC failure cannot expose a partially deleted lens.
        for pointer in pointers:
            pointer.unlink(missing_ok=True)
        if directory.exists():
            fsync_directory(directory)
        for path in shards:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not remove lens shard %s: %s", path, exc)
        if directory.exists():
            fsync_directory(directory)
    LENS_SOURCES.clear_if_active(model_id, SourceRef(LOCAL, name))
    return removed
