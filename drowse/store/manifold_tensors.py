"""The fitted-manifold tensor codec and the activation-row spool.

:func:`save_manifold` / :func:`load_manifold` read and write one fitted
tensor/sidecar pair (atomic, fsynced, sidecar first) under the stable pair
lock; the sidecar schema is :class:`drowse.store.manifold_schema.ManifoldSidecar`.
:class:`ActivationRowStore` is the mmap-backed per-layer row spool curved
fitting consumes.  Folder-level orchestration (manifest, proofs, locks) lives
in :mod:`drowse.store.manifold_folder`.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from drowse.geometry.manifold import (
    MANIFOLD_FIT_POLICY_VERSION,
    LayerSubspace,
    Manifold,
    domain_from_spec,
)
from drowse.store.manifold_folder import (
    ManifoldFolder,
    _locked_manifest,
    manifold_pair_lock,
    prove_pair,
)
from drowse.store.manifold_schema import (
    SIDECAR_DERIVED_FIELDS,
    SIDECAR_FIELDS,
    ManifoldFormatError,
    ManifoldSidecar,
)

log = logging.getLogger(__name__)


# ------------------------------------------ activation-row spool ---


class ActivationRowStore:
    """Temporary layer-major pooled-row spool for curved manifold fitting.

    A curved fit needs per-response rows only after the centroid-derived basis
    exists.  Keeping ``nodes × responses × layers × d_model`` in resident fp32
    is a multi-GiB cliff, while dropping it makes auto-curved fitting repeat the
    complete model pass.  This store writes one mmap-backed tensor per layer in
    the model output dtype, indexed by the original flat corpus row.  fp16/bf16
    storage is lossless relative to the source residual; fp32 promotion happens
    only when covariance math consumes a node slice.
    """

    def __init__(self, node_sizes: Sequence[int], *, spool: bool = True) -> None:
        self.node_sizes = [int(size) for size in node_sizes]
        self.offsets: list[int] = []
        offset = 0
        for size in self.node_sizes:
            self.offsets.append(offset)
            offset += size
        self.total_rows = offset
        self._tmp: tempfile.TemporaryDirectory[str] | None = (
            tempfile.TemporaryDirectory(prefix="drowse-manifold-rows-")
            if spool else None
        )
        self._layers: dict[int, torch.Tensor] = {}
        self._owners: list[ActivationRowStore] = []
        self._closed = False

    @classmethod
    def load_shards(
        cls,
        paths: "dict[int, Path]",
        node_sizes: Sequence[int],
    ) -> "ActivationRowStore":
        """Load independently persisted per-layer activation-row shards.

        Capture-cache v4 stores one safetensors file per layer so a scoped fit
        neither maps nor rewrites unrelated multi-GiB row payloads.  Every
        shard must carry exactly its named ``layer_<L>`` tensor; the caller
        validates the tensor's exact digest against the atomically published
        capture metadata.
        """
        if not paths:
            raise ValueError("activation-row shard selection must not be empty")
        store = cls(node_sizes, spool=False)
        for idx, path in sorted(paths.items()):
            expected_key = f"layer_{int(idx)}"
            with safe_open(str(path), framework="pt", device="cpu") as tensors:
                keys = list(tensors.keys())
                if keys != [expected_key]:
                    raise ValueError(
                        f"activation-row shard at {path} has keys {keys}, "
                        f"expected [{expected_key!r}]"
                    )
                shape = tuple(
                    int(dim) for dim in tensors.get_slice(expected_key).get_shape()
                )
                if len(shape) != 2 or shape[0] != store.total_rows:
                    raise ValueError(
                        f"invalid activation-row shard tensor {expected_key!r} "
                        f"at {path}"
                    )
                store._layers[int(idx)] = tensors.get_tensor(expected_key)
        return store

    @classmethod
    def combine_disjoint(
        cls, stores: Sequence["ActivationRowStore"],
    ) -> "ActivationRowStore":
        """Combine disjoint layer stores by view, transferring ownership.

        Sharded cache top-ups often pair many immutable cached layers with one
        newly captured temporary layer. Copying all ``N x D`` rows into a third
        mmap merely to present one layer-major roster doubles multi-GiB I/O.
        This composite aliases the existing tensors and keeps their stores
        alive until the composite closes; no payload bytes move.
        """
        if not stores:
            raise ValueError("activation-row combine needs at least one store")
        node_sizes = list(stores[0].node_sizes)
        if any(store._closed for store in stores):
            raise RuntimeError("cannot combine a closed activation-row store")
        if any(store.node_sizes != node_sizes for store in stores[1:]):
            raise ValueError("activation-row stores must share node sizes")
        layers: dict[int, torch.Tensor] = {}
        for store in stores:
            overlap = set(layers) & set(store._layers)
            if overlap:
                raise ValueError(
                    f"activation-row stores overlap layers {sorted(overlap)}"
                )
            layers.update(store._layers)
        combined = cls(node_sizes, spool=False)
        combined._layers = layers
        combined._owners = list(stores)
        return combined

    def _layer(self, idx: int, *, dim: int, dtype: torch.dtype) -> torch.Tensor:
        existing = self._layers.get(idx)
        if existing is not None:
            if existing.shape != (self.total_rows, dim) or existing.dtype != dtype:
                raise ValueError(f"activation-row shape/dtype changed at layer {idx}")
            return existing
        if self._closed:
            raise RuntimeError("activation-row store is closed")
        assert self._tmp is not None
        path = Path(self._tmp.name) / f"layer_{idx}.bin"
        rows = torch.from_file(
            str(path), shared=True, size=self.total_rows * dim, dtype=dtype,
        ).reshape(self.total_rows, dim)
        self._layers[idx] = rows
        return rows

    def write(
        self, idx: int, flat_indices: torch.Tensor, rows: torch.Tensor,
    ) -> None:
        host = rows.detach().to(device="cpu")
        target = self._layer(idx, dim=int(host.shape[1]), dtype=host.dtype)
        target.index_copy_(0, flat_indices.to(dtype=torch.long, device="cpu"), host)

    def node_rows(self, node_idx: int) -> dict[int, torch.Tensor]:
        start = self.offsets[node_idx]
        end = start + self.node_sizes[node_idx]
        return {idx: rows[start:end] for idx, rows in self._layers.items()}

    @property
    def layer_indices(self) -> list[int]:
        return sorted(self._layers)

    def flat_rows(self, idx: int) -> torch.Tensor:
        return self._layers[idx]

    def __iter__(self) -> Iterator[dict[int, torch.Tensor]]:
        for node_idx in range(len(self.node_sizes)):
            yield self.node_rows(node_idx)

    def close(self) -> None:
        if self._closed:
            return
        self._layers.clear()
        owners = getattr(self, "_owners", [])
        self._owners = []
        for owner in owners:
            owner.close()
        if self._tmp is not None:
            self._tmp.cleanup()
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ------------------------------------------------------------- save/load ---

def _replace_manifold_file(source: Path, target: Path) -> None:
    """Atomic replace seam used by publication failure-injection tests."""

    os.replace(source, target)

def fitted_tensor_layers(tensor_path: Path) -> tuple[int, ...] | None:
    """Fitted layer roster off the safetensors header alone (no payload read).

    ``None`` when the header is unreadable.
    """
    from safetensors import SafetensorError

    try:
        with safe_open(str(tensor_path), framework="pt", device="cpu") as tensors:
            return tuple(sorted({
                int(key.split(".", 1)[0].split("_", 1)[1])
                for key in tensors.keys()
                if key.startswith("layer_") and key.endswith(".mean")
            }))
    except (OSError, ValueError, SafetensorError):
        return None


# Read-side annotations ``load_manifold`` stamps onto a loaded manifold's
# ``metadata`` that must never reach disk.  ``_tensor_sha256`` is the digest the
# integrity manifest proved for the payload just read; ``fit.bake`` consumes it
# as component provenance.  A round-trip (load -> transfer -> save) carries it
# along, so the writer tolerates exactly these keys.
_TRANSIENT_METADATA_KEYS = frozenset({"_tensor_sha256"})


def save_manifold(
    manifold: Manifold, path: str | Path, metadata: dict[str, object],
) -> None:
    """Save a fitted manifold as ``.safetensors`` + a ``.json`` sidecar.

    The safetensors payload carries, per layer ``L``: ``layer_<L>.mean``,
    ``layer_<L>.basis``, ``layer_<L>.node_params``,
    ``layer_<L>.rbf_weights``, ``layer_<L>.poly_coeffs``,
    ``layer_<L>.coord_offset``, ``layer_<L>.coord_scale``; plus a shared
    bare ``node_coords`` tensor.  The sidecar carries the manifold identity
    (name, domain spec, ordered node labels, feature space), the per-layer
    ``origin_per_layer`` (the authoring-coordinate foot of the neutral mean,
    keyed by layer), plus the provenance fields in ``metadata``.

    **Affine (flat / folded-vector) layers** write ``layer_<L>.mean`` +
    ``layer_<L>.basis`` and optional ``layer_<L>.affine_map`` — there is no RBF
    triple and ordinary-fit coord normalization is identity (reconstructed by
    :meth:`LayerSubspace.affine` on load).  The *absence* of ``node_params``
    on disk is the read-side affine marker ``load_manifold`` keys on, so a
    folded-vector artifact and a fitted manifold share one save/load path.
    """
    # The writer derives every identity field from ``manifold`` itself; the
    # caller's metadata supplies provenance only.  A key outside the sidecar
    # schema would otherwise vanish silently — a new provenance field must be
    # added to the schema instead.
    unknown = set(metadata) - SIDECAR_FIELDS - _TRANSIENT_METADATA_KEYS
    if unknown:
        raise ValueError(
            f"manifold writer metadata has unknown key(s): {sorted(unknown)}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifold.validate_runtime_geometry()

    # Every drowse safetensor artifact is stored fp32 (cast at the writer);
    # fits already produce fp32, so the casts are idempotent guarantees.
    tensors: dict[str, torch.Tensor] = {
        "node_coords": manifold.node_coords.contiguous().to(torch.float32).cpu(),
    }
    for idx, sub in manifold.layers.items():
        tensors[f"layer_{idx}.mean"] = sub.mean.contiguous().to(torch.float32).cpu()
        tensors[f"layer_{idx}.basis"] = sub.basis.contiguous().to(torch.float32).cpu()
        if sub.is_affine:
            # Flat (folded-vector / subspace) layer: no RBF surface, and the
            # coord normalization is identity — rebuilt from the basis shape
            # by ``LayerSubspace.affine`` on load.  Persist mean + basis (the
            # *absence* of ``node_params`` on disk is the affine marker), plus
            # the per-layer **real, neutral-anchored** node coords ``(K, R)``
            # when present — the steer-target source (§5).
            if sub.node_coords is not None:
                tensors[f"layer_{idx}.node_coords"] = (
                    sub.node_coords.contiguous().to(torch.float32).cpu()
                )
            if sub.affine_map is not None:
                tensors[f"layer_{idx}.affine_map"] = (
                    sub.affine_map.contiguous().to(torch.float32).cpu()
                )
            continue
        np_, rw, pc = sub.rbf_params()
        tensors[f"layer_{idx}.node_params"] = np_.contiguous().to(torch.float32).cpu()
        tensors[f"layer_{idx}.rbf_weights"] = rw.contiguous().to(torch.float32).cpu()
        tensors[f"layer_{idx}.poly_coeffs"] = pc.contiguous().to(torch.float32).cpu()
        tensors[f"layer_{idx}.coord_offset"] = sub.coord_offset.contiguous().to(torch.float32).cpu()
        tensors[f"layer_{idx}.coord_scale"] = sub.coord_scale.contiguous().to(torch.float32).cpu()
        # Fuzzy-manifold σ-field (curved fits with the within-node spread pass).
        # The log-σ RBF over the same normalized ``node_params``; its *absence*
        # on disk is the current SAE-space "zero-thickness wire" marker.
        if sub.sigma_rbf_weights is not None and sub.sigma_poly_coeffs is not None:
            tensors[f"layer_{idx}.sigma_rbf_weights"] = (
                sub.sigma_rbf_weights.contiguous().to(torch.float32).cpu()
            )
            tensors[f"layer_{idx}.sigma_poly_coeffs"] = (
                sub.sigma_poly_coeffs.contiguous().to(torch.float32).cpu()
            )
    provenance = {
        key: value for key, value in metadata.items()
        if key in SIDECAR_FIELDS and key not in SIDECAR_DERIVED_FIELDS
    }
    provenance.setdefault("method", "manifold_pca")
    sidecar = ManifoldSidecar.build(
        name=manifold.name,
        domain=manifold.domain.to_spec(),
        node_labels=list(manifold.node_labels),
        feature_space=manifold.feature_space,
        fitted_layers=sorted(manifold.layers),
        mahalanobis_share_per_layer={
            str(idx): float(v) for idx, v in manifold.mahalanobis_share.items()
        },
        origin_per_layer={
            str(idx): [float(c) for c in o.reshape(-1).tolist()]
            for idx, o in manifold.origin.items()
        },
        **provenance,
    ).to_payload()

    with manifold_pair_lock(path):
        sidecar_path = path.with_suffix(".json")
        tensor_fd, tensor_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
        )
        os.close(tensor_fd)
        sidecar_fd, sidecar_name = tempfile.mkstemp(
            prefix=f".{sidecar_path.name}.", suffix=".tmp", dir=path.parent,
        )
        tensor_tmp = Path(tensor_name)
        sidecar_tmp = Path(sidecar_name)
        try:
            # Stage and fsync *both* payloads before replacing either canonical
            # path.  Commit the sidecar first: an interruption on a first fit
            # then leaves an ignored orphan sidecar, never the tensor-without-
            # sidecar shape the folder loader must reject.  On replacement an
            # interrupted mixed pair fails its old manifest proof and the next
            # target fit deterministically overwrites it.
            save_file(tensors, str(tensor_tmp))
            with open(tensor_tmp, "rb") as handle:
                os.fsync(handle.fileno())
            sidecar_bytes = (json.dumps(sidecar, indent=2) + "\n").encode()
            with os.fdopen(sidecar_fd, "wb", closefd=True) as handle:
                sidecar_fd = -1
                handle.write(sidecar_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            _replace_manifold_file(sidecar_tmp, sidecar_path)
            _replace_manifold_file(tensor_tmp, path)
            # Make both directory entries durable before the manifest CAS can
            # publish their hashes as the current pair.
            try:
                dir_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                # Directory fsync is unavailable on some platforms; both files
                # themselves are still fsynced and retry recovery remains safe.
                pass
        finally:
            if sidecar_fd >= 0:
                os.close(sidecar_fd)
            tensor_tmp.unlink(missing_ok=True)
            sidecar_tmp.unlink(missing_ok=True)
    log.info("Saved manifold %r (%d layers) to %s",
             manifold.name, len(manifold.layers), path)


def load_manifold(
    path: str | Path, *, verify_manifest: bool = True,
) -> Manifold:
    """Load a fitted manifold, proving its manifest pair by default.

    With ``verify_manifest`` the folder manifest must prove both halves of
    this pair, and a non-baked fit must also be current for the live corpus
    (``nodes_sha256``) and the current numerical fit policy.
    """
    resolved = Path(path)
    if verify_manifest:
        # Global lock order for fitted artifacts is folder -> tensor pair.
        with _locked_manifest(resolved.parent):
            with manifold_pair_lock(resolved):
                return _load_manifold_locked(resolved, verify_manifest=True)
    with manifold_pair_lock(resolved):
        return _load_manifold_locked(resolved, verify_manifest=False)


def _load_manifold_locked(
    path: str | Path, *, verify_manifest: bool = True,
) -> Manifold:
    """Read one fitted manifold while its tensor/sidecar pair lock is held."""
    path = Path(path)
    folder_view: ManifoldFolder | None = None
    if verify_manifest and (path.parent / "manifold.json").exists():
        folder_view = ManifoldFolder.load(path.parent, verify_manifest=False)
        if not prove_pair(path.parent, path, folder_view.files):
            raise ManifoldFormatError(
                f"manifold integrity manifest has no proof for fitted pair {path.name}"
            )
    tensors = load_file(str(path))
    sidecar = ManifoldSidecar.load(path.with_suffix(".json")).to_payload()
    if folder_view is not None:
        sidecar["_tensor_sha256"] = folder_view.files[path.name]
        if sidecar["fit_mode"] != "baked":
            if sidecar["nodes_sha256"] != folder_view.nodes_sha256():
                raise ManifoldFormatError(
                    f"fitted manifold {path.name} is stale for the live corpus/"
                    "domain/template inputs; refit before loading"
                )
            if sidecar["fit_policy_version"] != MANIFOLD_FIT_POLICY_VERSION:
                raise ManifoldFormatError(
                    f"fitted manifold {path.name} uses an older numerical fit "
                    "policy; refit it with the current drowse"
                )
    # The shared, layer-agnostic node_coords tensor — pop it before the
    # per-layer key split, which assumes ``layer_<idx>.<field>``.
    node_coords = tensors.pop("node_coords", None)
    if node_coords is None:
        raise ManifoldFormatError("fitted manifold tensor is missing node_coords")

    by_layer: dict[int, dict[str, torch.Tensor]] = {}
    allowed_fields = {
        "mean", "basis", "node_coords", "affine_map", "node_params",
        "rbf_weights", "poly_coeffs", "coord_offset", "coord_scale",
        "sigma_rbf_weights", "sigma_poly_coeffs",
    }
    for key, tensor in tensors.items():
        if key.count(".") != 1:
            raise ManifoldFormatError(f"invalid fitted manifold tensor key {key!r}")
        head, field_name = key.split(".", 1)
        raw_layer = head.removeprefix("layer_")
        if (
            not head.startswith("layer_") or not raw_layer.isascii()
            or not raw_layer.isdecimal() or str(int(raw_layer)) != raw_layer
            or field_name not in allowed_fields
        ):
            raise ManifoldFormatError(f"invalid fitted manifold tensor key {key!r}")
        idx = int(raw_layer)
        by_layer.setdefault(idx, {})[field_name] = tensor

    tensor_layers = set(by_layer)
    sidecar_layers = set(sidecar["fitted_layers"])
    if not tensor_layers or tensor_layers != sidecar_layers:
        raise ManifoldFormatError(
            "fitted manifold tensor layers do not match sidecar fitted_layers"
        )

    layers: dict[int, LayerSubspace] = {}
    for idx, parts in by_layer.items():
        if "node_params" not in parts:
            if not {"mean", "basis"}.issubset(parts) or set(parts) - {
                "mean", "basis", "node_coords", "affine_map",
            }:
                raise ManifoldFormatError(
                    f"affine manifold layer {idx} has invalid tensor fields"
                )
            # Affine (flat / folded-vector) layer — only mean + basis on disk
            # (the coord normalization is identity, rebuilt from the basis
            # shape), plus the per-layer real node coords when the writer
            # stamped them.  Read side of ``save_manifold``'s affine branch.
            layers[idx] = LayerSubspace.affine(
                mean=parts["mean"], basis=parts["basis"],
                node_coords=parts.get("node_coords"),
                affine_map=parts.get("affine_map"),
            )
            continue
        required_curved = {
            "mean", "basis", "node_params", "rbf_weights", "poly_coeffs",
            "coord_offset", "coord_scale",
        }
        sigma_fields = {"sigma_rbf_weights", "sigma_poly_coeffs"}
        if (
            not required_curved.issubset(parts)
            or set(parts) - required_curved - sigma_fields
            or bool(set(parts) & sigma_fields) != sigma_fields.issubset(parts)
        ):
            raise ManifoldFormatError(
                f"curved manifold layer {idx} has invalid tensor fields"
            )
        layers[idx] = LayerSubspace(
            mean=parts["mean"],
            basis=parts["basis"],
            node_params=parts["node_params"],
            rbf_weights=parts["rbf_weights"],
            poly_coeffs=parts["poly_coeffs"],
            coord_offset=parts["coord_offset"],
            coord_scale=parts["coord_scale"],
            # Fuzzy-manifold σ-field. Current raw curved fits require this
            # pair; SAE-space curved fits deliberately omit the raw-space tube.
            sigma_rbf_weights=parts.get("sigma_rbf_weights"),
            sigma_poly_coeffs=parts.get("sigma_poly_coeffs"),
        )

    domain = domain_from_spec(sidecar["domain"])
    maha_raw = sidecar["mahalanobis_share_per_layer"]
    mahalanobis_share: dict[int, float] = {
        int(k): float(v) for k, v in maha_raw.items()
    }

    # Per-layer origin ``O_L`` (authoring-coordinate foot of neutral).
    origin_raw = sidecar["origin_per_layer"]
    origin: dict[int, torch.Tensor] = {
        int(k): torch.tensor([float(c) for c in v], dtype=torch.float32)
        for k, v in origin_raw.items()
    }

    manifold = Manifold(
        name=sidecar["name"],
        domain=domain,
        node_labels=list(sidecar["node_labels"]),
        node_coords=node_coords,
        layers=layers,
        feature_space=sidecar["feature_space"],
        metadata=sidecar,
        node_roles=list(sidecar["node_roles"]),
        node_kinds=list(sidecar["node_kinds"]),
        mahalanobis_share=mahalanobis_share,
        origin=origin,
    )
    manifold.validate_runtime_geometry()
    return manifold
