"""ArtifactRegistry — the session's in-memory steering artifacts.

Two registries live here: baked steering *profiles* (name → per-layer
direction tensors) and loaded *manifolds* (registry key ``[ns/]name[:variant]``
→ :class:`~drowse.geometry.manifold.Manifold`).  The public surface hands out
read-only :class:`~collections.abc.Mapping` views; every mutation goes through
a method here, so the dependent caches move with it:

- an *invalidating* mutation (``put_profile`` / ``pop_profile`` /
  ``evict_prefix`` / ``adopt_fitted`` / ``evict_artifact`` / ``clear``) drops
  the analytics CPU snapshots and fires the ``on_change`` listeners (the
  session wires its prefix-cache invalidation there);
- a manifold *swap* (``adopt_fitted`` / ``evict_artifact``) also fires the
  ``on_swap`` listeners with the identities of the replaced objects, so the
  session can refresh or drop monitor probes that read them;
- lazy loads (``ensure_manifold_loaded`` / ``ensure_profile_registered``)
  memoize what is already on disk and fire nothing.

Loading validates a fitted tensor against the live weights and capture inputs
before it is admitted, and ``concretize_key`` pins a bare ``:sae`` / ``:role``
/ ``:from`` variant to the unique fitted tensor for this model.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from drowse.errors import ManifoldNotRegisteredError, ProfileNotRegisteredError
from drowse.geometry.manifold import Manifold
from drowse.geometry.profile import Profile
from drowse.steering.grammar import RESERVED_NAMESPACES

Tensors = dict[int, torch.Tensor]

# Variants that name a family of fitted tensors rather than one tensor; they
# concretize against the files on disk for the loaded model.
_BARE_VARIANTS = frozenset({"sae", "from", "role"})


@dataclass(frozen=True)
class ModelBinding:
    """The loaded model an artifact must match before it is admitted.

    Also the read-only handle :func:`~drowse.fit.pipeline.prepare_manifold_capture_identity`
    renders the staleness check through.
    """

    model: Any
    tokenizer: Any
    model_id: str
    device: torch.device
    dtype: torch.dtype


def split_key(key: str) -> tuple[str | None, str, str]:
    """``[ns/]name[:variant]`` → ``(ns, name, variant)`` (variant ``raw`` if absent)."""
    name_part, variant = key.rsplit(":", 1) if ":" in key else (key, "raw")
    if "/" in name_part:
        ns, name = name_part.split("/", 1)
        return ns, name, variant
    return None, name_part, variant


def artifact_key_matches(
    key: str, namespace: str, name: str, feature: str | None = None,
) -> bool:
    """True iff registry ``key`` reads the artifact folder ``namespace/name``.

    Both the qualified and the bare form match (a bare name resolves across
    namespaces, so a bare cache entry may hold this folder's tensor).
    ``feature`` narrows to one tensor family: ``"raw"`` (the residual-stream
    fit, read raw or role-baselined) or ``"sae-<release>"``; cross-model
    ``from-`` transfers are independent artifacts and match only when
    ``feature`` is ``None`` (every variant).
    """
    head, variant = key.rsplit(":", 1) if ":" in key else (key, "raw")
    if head not in (name, f"{namespace}/{name}"):
        return False
    if feature is None:
        return True
    if variant.startswith("sae-"):
        return variant == feature
    if variant == "raw" or variant.startswith("role-"):
        return feature == "raw"
    return False


class ArtifactRegistry:
    """In-memory profiles and manifolds for one loaded model."""

    def __init__(self, binding: ModelBinding) -> None:
        self._binding = binding
        self._profiles: dict[str, Tensors] = {}
        self._manifolds: dict[str, Manifold] = {}
        self._on_change: list[Callable[[], None]] = []
        self._on_swap: list[Callable[[set[int], Manifold | None], None]] = []
        self._atom_resolvers: dict[str, Callable[[str], str]] = {}
        # Read-side analytics snapshots (CPU copies), dropped on any
        # invalidating mutation.  ``_epoch`` advances with every drop so a
        # snapshot built across a concurrent mutation is never stored.
        self._analytics: dict[str, Profile] = {}
        self._epoch = 0
        self._analytics_lock = threading.Lock()

    # ------------------------------------------------------------- views

    @property
    def profiles(self) -> Mapping[str, Tensors]:
        """Read-only view of the profile registry (name → per-layer tensors)."""
        return MappingProxyType(self._profiles)

    @property
    def manifolds(self) -> Mapping[str, Manifold]:
        """Read-only view of the loaded manifolds (registry key → Manifold)."""
        return MappingProxyType(self._manifolds)

    # --------------------------------------------------------- listeners

    def on_change(self, callback: Callable[[], None]) -> None:
        """Call ``callback`` after every invalidating mutation."""
        self._on_change.append(callback)

    def on_swap(self, callback: Callable[[set[int], Manifold | None], None]) -> None:
        """Call ``callback(old_ids, replacement)`` when loaded manifolds are
        replaced (``replacement`` is the promoted fit) or evicted (``None``)."""
        self._on_swap.append(callback)

    def set_atom_resolver(self, namespace: str, resolver: Callable[[str], str]) -> None:
        """Route a reserved ``<namespace>/<atom>`` profile key to ``resolver``.

        The resolver registers the direction (through :meth:`put_profile`)
        and returns its registry key.
        """
        self._atom_resolvers[namespace] = resolver

    def _changed(self) -> None:
        self.invalidate_analytics()
        for callback in self._on_change:
            callback()

    def _swapped(self, old_ids: set[int], replacement: Manifold | None) -> None:
        if not old_ids:
            return
        for callback in self._on_swap:
            callback(old_ids, replacement)

    # --------------------------------------------------------- mutations

    def promote(self, tensors: Mapping[int, torch.Tensor]) -> Tensors:
        """Per-layer tensors on the model device and dtype."""
        b = self._binding
        return {int(idx): vec.to(b.device, b.dtype) for idx, vec in tensors.items()}

    def put_profile(self, name: str, tensors: "Mapping[int, torch.Tensor] | Profile") -> None:
        """Register (or replace) the steering profile ``name``."""
        source = tensors.as_dict() if isinstance(tensors, Profile) else tensors
        self._profiles[name] = dict(source)
        self._changed()

    def put_manifold(self, key: str, manifold: Manifold) -> None:
        """Register an in-memory manifold under registry key ``key``."""
        self._manifolds[key] = manifold
        self._changed()

    def pop_profile(self, name: str) -> Tensors | None:
        """Unregister ``name``; returns the removed tensors, if any."""
        removed = self._profiles.pop(name, None)
        self._changed()
        return removed

    def evict_prefix(self, prefix: str) -> list[str]:
        """Unregister every profile whose key starts with ``prefix``."""
        removed = [key for key in self._profiles if key.startswith(prefix)]
        for key in removed:
            del self._profiles[key]
        self._changed()
        return removed

    def adopt_fitted(self, folder: "str | Path", manifold: Manifold) -> None:
        """Move every in-memory consumer of a freshly fitted artifact to it.

        Loaded manifolds that read the same tensor family are replaced by the
        promoted fit, folded profiles are re-folded (or dropped when the fit
        no longer folds to one direction), and ``on_swap`` listeners refresh
        whatever held the old objects.
        """
        from drowse.fit.capture import folded_directions, is_foldable_vector_manifold

        path = Path(folder)
        ns, name = path.parent.name, path.name
        feature = manifold.feature_space
        keys = [
            key for key in self._manifolds
            if artifact_key_matches(key, ns, name, feature)
        ]
        old_ids = {id(self._manifolds[key]) for key in keys}
        promoted: Manifold | None = None
        if keys:
            promoted = manifold.to(device=self._binding.device, dtype=torch.float32)
            for key in keys:
                self._manifolds[key] = promoted
        profile_keys = [
            key for key in self._profiles
            if artifact_key_matches(key, ns, name, feature)
        ]
        if profile_keys and is_foldable_vector_manifold(manifold):
            folded = self.promote(folded_directions(manifold))
            for key in profile_keys:
                self._profiles[key] = dict(folded)
        else:
            for key in profile_keys:
                del self._profiles[key]
        self._swapped(old_ids, promoted)
        self._changed()

    def evict_artifact(
        self, namespace: str, name: str, *, feature: str | None = None,
    ) -> None:
        """Drop every in-memory consumer of the artifact folder ``namespace/name``.

        ``feature`` (``"raw"`` / ``"sae-<release>"``) narrows the eviction to
        one tensor family; ``None`` evicts every variant.  The next steer or
        probe attach reloads from disk.
        """
        keys = [
            key for key in self._manifolds
            if artifact_key_matches(key, namespace, name, feature)
        ]
        old_ids = {id(self._manifolds.pop(key)) for key in keys}
        for key in [
            key for key in self._profiles
            if artifact_key_matches(key, namespace, name, feature)
        ]:
            del self._profiles[key]
        self._swapped(old_ids, None)
        self._changed()

    def clear(self) -> None:
        """Drop everything (session teardown)."""
        self._profiles.clear()
        self._manifolds.clear()
        self._changed()

    # ---------------------------------------------------------- analytics

    def invalidate_analytics(self) -> None:
        """Drop the analytics CPU snapshots (rebuilt lazily on the next poll)."""
        with self._analytics_lock:
            self._epoch += 1
            self._analytics.clear()

    def analytics_snapshot(
        self,
        name: str,
        *,
        live: Callable[[], "Mapping[int, torch.Tensor] | None"],
        lock: Any,
    ) -> Profile | None:
        """CPU snapshot of ``name``'s direction for read-side analytics, cached.

        ``live`` returns the (possibly device-resident) per-layer tensors; it
        runs under ``lock`` — the exclusive-GPU lock — acquired non-blocking,
        so a poll defers (``None``) instead of racing a model op on PyTorch's
        single MPS command buffer.  The snapshot is cached only when no
        invalidating mutation landed while it was built.
        """
        with self._analytics_lock:
            cached = self._analytics.get(name)
            epoch = self._epoch
        if cached is not None:
            return cached
        if not lock.acquire(blocking=False):
            return None
        try:
            tensors = live()
            if tensors is None:
                return None
            snap = Profile({
                int(L): t.detach().float().cpu() for L, t in tensors.items()
            })
            with self._analytics_lock:
                if self._epoch == epoch:
                    self._analytics[name] = snap
        finally:
            lock.release()
        return snap

    # ------------------------------------------------------------ loading

    def concretize_key(self, key: str) -> str:
        """Pin a bare ``:sae`` / ``:role`` / ``:from`` variant to its fitted tensor.

        Any other key is returned unchanged.  Raises
        :class:`~drowse.errors.UnknownVariantError` when no fitted variant
        exists for the loaded model and
        :class:`~drowse.errors.AmbiguousVariantError` when several do.
        """
        ns, name, variant = split_key(key)
        if variant not in _BARE_VARIANTS:
            return key
        from drowse.base.paths import (
            manifold_dir, manifolds_dir, parse_tensor_filename, safe_model_id,
        )
        from drowse.errors import AmbiguousVariantError, UnknownVariantError

        if ns is not None:
            roots = [manifold_dir(ns, name)]
        elif manifolds_dir().exists():
            roots = [
                entry / name for entry in manifolds_dir().iterdir()
                if entry.is_dir()
            ]
        else:
            roots = []
        concrete: set[str] = set()
        safe_model = safe_model_id(self._binding.model_id)
        for folder in roots:
            if variant == "role":
                sidecar_path = folder / f"{safe_model}.json"
                if not sidecar_path.exists():
                    continue
                with open(sidecar_path) as handle:
                    raw_sidecar = json.load(handle)
                roles = raw_sidecar.get("node_roles") or [
                    None
                ] * int(raw_sidecar.get("node_count", 0))
                if roles and roles[0] is not None and all(
                    role == roles[0] for role in roles
                ):
                    concrete.add(f"role-{roles[0]}")
                continue
            for tensor_path in folder.glob(f"{safe_model}_*.safetensors"):
                parsed = parse_tensor_filename(tensor_path.name)
                if parsed is None or parsed[1] is None:
                    continue
                if parsed[1].startswith(f"{variant}-"):
                    concrete.add(parsed[1])
        qualified = name if ns is None else f"{ns}/{name}"
        if len(concrete) != 1:
            error = UnknownVariantError if not concrete else AmbiguousVariantError
            raise error(
                f"manifold '{qualified}:{variant}' resolves to "
                f"{sorted(concrete) or 'no fitted variants'}"
            )
        return f"{qualified}:{next(iter(concrete))}"

    def ensure_manifold_loaded(self, key: str) -> Manifold:
        """The fitted manifold for registry key ``[ns/]name[:variant]``, loading it if absent.

        ``raw`` selects the residual-stream tensor, ``sae-<release>`` the SAE
        fit, ``from-<src>`` a cross-model transfer, and ``role-<id>`` the
        canonical tensor with its role baseline checked.  A bare name searches
        every namespace (and refuses a collision).  The loaded manifold is
        validated against the live weights and capture inputs, then promoted
        onto the model device in fp32.  Raises
        :class:`ManifoldNotRegisteredError` on a miss or a stale fit.
        """
        loaded = self._manifolds.get(key)
        if loaded is not None:
            return loaded
        from drowse.base.paths import (
            encode_release_id, manifold_dir, manifolds_dir,
            tensor_filename_for_variant,
        )
        from drowse.model.fingerprint import loaded_model_fingerprint
        from drowse.store.manifold_schema import sidecar_is_current
        from drowse.store.manifold_tensors import load_manifold

        model_id = self._binding.model_id
        ns, name, variant = split_key(key)
        if ns is not None:
            search_ns = [ns]
        else:
            root = manifolds_dir()
            search_ns = (
                sorted(d.name for d in root.iterdir() if d.is_dir())
                if root.exists() else []
            )
        requested_role = (
            variant[len("role-"):] if variant.startswith("role-") else None
        )
        transferred_from = (
            variant[len("from-"):] if variant.startswith("from-") else None
        )
        fname = tensor_filename_for_variant(model_id, variant)
        if fname is None:
            raise ManifoldNotRegisteredError(
                f"manifold '{key}': unsupported variant '{variant}'"
            )
        # An older artifact format is a miss (refit), not a corrupt artifact.
        matches = [
            (candidate_ns, path)
            for candidate_ns, path in (
                (candidate_ns, manifold_dir(candidate_ns, name) / fname)
                for candidate_ns in search_ns
            )
            if path.exists()
            and sidecar_is_current(path.with_suffix(".json"))
        ]
        if not matches:
            raise ManifoldNotRegisteredError(
                f"manifold '{key}' has no fitted tensor for {model_id}; "
                f"run `drowse manifold fit` first"
            )
        if len(matches) > 1:
            from drowse.store.selectors import AmbiguousSelectorError
            qualified = ", ".join(f"{m_ns}/{name}" for m_ns, _ in matches)
            raise AmbiguousSelectorError(
                f"ambiguous manifold '{name}': matches {qualified}. "
                f"Qualify it with a namespace."
            )
        tensor_path = matches[0][1]
        manifold = load_manifold(tensor_path)
        metadata = manifold.metadata
        live_fingerprint = loaded_model_fingerprint(self._binding.model, model_id)
        if transferred_from is not None:
            source_id = metadata.get("source_model_id")
            if (
                not isinstance(source_id, str)
                or encode_release_id(source_id) != transferred_from
            ):
                raise ManifoldNotRegisteredError(
                    f"manifold '{key}' transfer provenance does not match "
                    f"source {transferred_from!r}; recompute the transfer"
                )
        raw_roles = metadata.get("node_roles")
        roles = (
            list(raw_roles)
            if isinstance(raw_roles, list)
            else [None] * len(manifold.node_labels)
        )
        if requested_role is not None and (
            not roles or not all(role == requested_role for role in roles)
        ):
            raise ManifoldNotRegisteredError(
                f"manifold '{key}' was not fitted uniformly with role "
                f"{requested_role!r}; refit it with that role"
            )
        if variant == "raw" and any(role is not None for role in roles):
            raise ManifoldNotRegisteredError(
                f"manifold '{key}' was fitted with role baseline "
                f"{roles!r}; steer it as :role-<name> or refit raw"
            )
        if metadata.get("model_fingerprint") != live_fingerprint:
            raise ManifoldNotRegisteredError(
                f"manifold '{key}' was fitted for different loaded weights "
                f"under {model_id}; run `drowse manifold fit` again"
            )
        if metadata.get("fit_mode", "authored") != "baked":
            from drowse.fit.pipeline import prepare_manifold_capture_identity
            from drowse.store.manifold_folder import ManifoldFolder

            folder_mf = ManifoldFolder.load(
                tensor_path.parent, verify_manifest=False,
            )
            expected_capture = prepare_manifold_capture_identity(
                self._binding, folder_mf, live_fingerprint,
            ).capture_sha256
            if metadata.get("capture_sha256") != expected_capture:
                raise ManifoldNotRegisteredError(
                    f"manifold '{key}' capture inputs changed (tokenizer, chat "
                    "template, role framing, or baseline prompts); refit it"
                )
        promoted = manifold.to(device=self._binding.device, dtype=torch.float32)
        self._manifolds[key] = promoted
        return promoted

    def ensure_profile_registered(
        self, name: str, *, role: str = "vector",
    ) -> Tensors:
        """Direction profile for registry key ``name``, loading it if absent.

        In order: an in-memory profile already registered (extracts, bakes,
        instrument atoms); a reserved instrument atom (``jlens/<word|#id>``,
        ``sae/<id>``) through its registered resolver; a fitted 2-node ``pca``
        manifold folded to its per-layer direction.  Raises
        :class:`ProfileNotRegisteredError` when nothing resolves.
        """
        profile = self._profiles.get(name)
        if profile is not None:
            return profile
        ns, atom, variant = split_key(name)
        if ns in RESERVED_NAMESPACES and variant == "raw":
            resolver = self._atom_resolvers.get(ns)
            if resolver is None:
                raise ProfileNotRegisteredError(
                    f"'{ns}/' atoms are reserved and not steerable yet ({name!r})"
                )
            return self._profiles[resolver(atom)]
        folded = self.fold_manifold(name, role=role)
        self._profiles[name] = folded
        return folded

    def fold_manifold(self, name: str, *, role: str = "vector") -> Tensors:
        """Load the fitted manifold ``name`` and fold it to a direction profile.

        A manifold miss becomes :class:`ProfileNotRegisteredError` (chained to
        the miss); any other load failure (ambiguity, corrupt artifact)
        propagates as itself.  A manifold that loads but isn't a 2-node affine
        subspace is a usage error.
        """
        try:
            manifold = self.ensure_manifold_loaded(name)
        except ManifoldNotRegisteredError as miss:
            raise ProfileNotRegisteredError(
                f"No profile registered for {role} '{name}' "
                f"({miss.args[0] if miss.args else miss})"
            ) from miss
        from drowse.fit.capture import folded_directions
        try:
            return folded_directions(manifold)
        except Exception as e:
            raise ProfileNotRegisteredError(
                f"'{name}' is a manifold that does not fold to a single "
                f"steering direction (not a 2-node affine subspace): {e}"
            ) from e
