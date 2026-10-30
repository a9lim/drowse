"""The resident SAE runtime: one dictionary, one hook layer, its metadata.

A session keeps at most one SAE source resident, at one residual-post hook
layer, so the unqualified ``sae/<id>`` grammar always names one feature.
Provider weights stay in the SAELens / Hugging Face caches and Drowse-trained
weights under ``DROWSE_HOME`` (:mod:`drowse.sae.store`).  The runtime holds the
lazy backend, the selected layer, its width and the per-feature Neuronpedia
metadata cache (:class:`SaeResidency`), and owns the lifecycle that changes
them: ``load`` / ``unload`` / ``train`` / ``restore_active``.

:class:`~drowse.sae.instrument.SaeInstrument` owns the runtime and shares its
``state_lock``.  The runtime's dependencies are explicit: a read-only model
handle (the public model surface), the exclusive-GPU section, and the guard
that refuses artifact work inside a steering scope.  A residency change runs
the caller's ``on_swap`` inside the same lock hold that swaps the backend, so
the owner drops state keyed by the old release (attached probes, registered
``sae/<id>`` directions) atomically with the swap.

Metadata is best effort and never on the steering path.  ``check_feature`` is
a local range check: steering runs under the exclusive-GPU section and needs
only the decoder row, so it never waits on Neuronpedia.  ``validate_feature``
and ``fetch_meta`` fetch outside every lock, then publish under the state lock
only if the residency they read from is still current.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable, Mapping, Sequence, Sized
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from drowse.base.exclusive import ExclusiveGuard
from drowse.errors import SaeFeatureError, SaeNotLoadedError
from drowse.model.fingerprint import loaded_model_fingerprint
from drowse.sae import neuronpedia
from drowse.sae.backend import LocalSaeBackend, resolve_sae_backend, sae_binding

_log = logging.getLogger(__name__)

#: Concurrent Neuronpedia requests for one metadata backfill.
_FETCH_WORKERS = 8


class SaeModelHandle(Protocol):
    """The read-only model surface the runtime needs (the session's public
    properties satisfy it).  Read on use, never copied."""

    @property
    def model_id(self) -> str: ...

    @property
    def model(self) -> torch.nn.Module: ...

    @property
    def tokenizer(self) -> Any: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def layers(self) -> Sized: ...

    @property
    def model_info(self) -> Mapping[str, Any]: ...


#: Called under the state lock with the residency already swapped.
SwapCallback = Callable[[], None]
#: Called under the state lock with newly cached ``{str(id): entry}`` rows.
FetchedCallback = Callable[[dict[str, dict[str, Any]]], None]


@dataclass(frozen=True, eq=False)
class SaeResidency:
    """One resident SAE: backend, hook layer, width, metadata cache.

    The binding is immutable; ``feature_meta`` is the mutable per-feature
    cache (``{str(id): {label, max_act[, checked]}}``).  Each load makes a
    new residency, so its identity is the "is this source still current"
    token every metadata write checks before it publishes.
    """

    backend: Any
    layer: int
    width: int
    feature_meta: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def info(self) -> dict[str, Any]:
        """``{release, layer, width, revision, fingerprint, sae_id, repo_id,
        neuronpedia_id}`` — the public residency description."""
        return {
            "release": self.backend.release,
            **sae_binding(self.backend, self.layer, self.width),
        }


def _needs_fetch(entry: Mapping[str, Any] | None) -> bool:
    """A feature is fetched until it has both fields or was asked about."""
    return entry is None or (
        (entry.get("max_act") is None or not entry.get("label"))
        and not entry.get("checked")
    )


class SaeRuntime:
    """Resident-SAE state and lifecycle for one session."""

    def __init__(
        self,
        handle: SaeModelHandle,
        *,
        exclusive: ExclusiveGuard,
        assert_unsteered: Callable[[], None],
    ) -> None:
        self._handle = handle
        #: The exclusive-GPU section (non-blocking; raises when busy).
        self.exclusive = exclusive
        self._assert_unsteered = assert_unsteered
        #: The SAE-state boundary, shared with the owning instrument.  A
        #: reentrant leaf lock: never held across a forward or a fetch.
        self.state_lock = threading.RLock()
        self._resident: SaeResidency | None = None

    # ------------------------------------------------------------ residency

    @property
    def resident(self) -> SaeResidency | None:
        return self._resident

    @property
    def layer(self) -> int | None:
        resident = self._resident
        return None if resident is None else resident.layer

    @property
    def info(self) -> dict[str, Any] | None:
        resident = self._resident
        return None if resident is None else resident.info

    @property
    def model_id(self) -> str:
        return self._handle.model_id

    @property
    def n_layers(self) -> int:
        return len(self._handle.layers)

    def require(self) -> SaeResidency:
        resident = self._resident
        if resident is None:
            raise SaeNotLoadedError(
                "no SAE loaded for this session — select one with "
                "`drowse sae use MODEL SOURCE` "
                "or POST /drowse/v1/sessions/default/sae/load"
            )
        return resident

    def _swap(
        self, resident: SaeResidency | None, on_swap: SwapCallback | None,
    ) -> None:
        with self.state_lock:
            self._resident = resident
            if on_swap is not None:
                on_swap()

    # ------------------------------------------------------------ lifecycle

    def load(
        self,
        release: str,
        *,
        layer: int | None = None,
        revision: str | None = None,
        on_swap: SwapCallback | None = None,
    ) -> dict[str, Any]:
        """Make one local or SAELens source the resident SAE.

        The deterministic default layer is the covered layer nearest 65%
        model depth, preferring the workspace band.  ``revision`` pins a
        provider release (a binding's recorded commit); ``None`` resolves the
        release's current Hub head.  The source binding is published before
        the swap, so a binding or metadata failure leaves the previous source
        resident.
        """
        from drowse.sae.sources import (
            SAE_SOURCES,
            load_sae_feature_meta,
            parse_sae_release,
            save_sae_metadata,
        )

        release = release.strip()
        if not release:
            raise ValueError("SAE release must not be empty")
        handle = self._handle
        with self.exclusive(
            "SAE load called while another model operation is in flight; retry shortly"
        ):
            resolved = resolve_sae_backend(
                release,
                model_id=handle.model_id,
                n_layers=len(handle.layers),
                hidden_dim=int(handle.model_info["hidden_dim"]),
                layer=layer,
                revision=revision,
                device=handle.device,
                dtype=handle.dtype,
                model_fingerprint=lambda: loaded_model_fingerprint(
                    handle.model, handle.model_id,
                ),
            )
            binding = resolved.binding
            local = isinstance(resolved.backend, LocalSaeBackend)
            feature_meta = (
                {} if local
                else load_sae_feature_meta(handle.model_id, release, source=binding)
            )
            if local:
                SAE_SOURCES.select(handle.model_id, parse_sae_release(release))
            else:
                save_sae_metadata(handle.model_id, release, binding)
            self._swap(
                SaeResidency(
                    resolved.backend, resolved.layer, resolved.width, feature_meta,
                ),
                on_swap,
            )
        return self.info or {}

    def unload(self, *, on_swap: SwapCallback | None = None) -> None:
        """Release the resident SAE."""
        with self.exclusive(
            "SAE unload called while another model operation is in flight; retry shortly"
        ):
            self._swap(None, on_swap)

    def close(self) -> None:
        """Session teardown: drop the resident backend.  Takes no lock and
        runs no callbacks (the session clears its registries itself)."""
        self._resident = None

    def restore_active(self, *, on_swap: SwapCallback | None = None) -> None:
        """Make an explicit ``sae use`` selection resident again at startup.

        A provider binding pins the commit it was fetched at and the last
        explicitly selected hook layer, so the restore lands exactly that
        measurement surface, offline from the provider cache and never moved
        to upstream HEAD.  Failure is non-fatal: a cache eviction must not
        keep the model from starting.
        """
        from drowse.sae.sources import load_active_sae

        active = load_active_sae(self._handle.model_id)
        if active is None:
            return
        release, metadata = active
        try:
            self.load(
                release,
                layer=None if metadata is None else metadata["layer"],
                revision=None if metadata is None else metadata["revision"],
                on_swap=on_swap,
            )
        except Exception as exc:  # noqa: BLE001 — never block startup
            _log.warning(
                "could not restore active SAE %s for %s: %s",
                release, self._handle.model_id, exc,
            )

    def train(
        self,
        name: str,
        documents: Sequence[str],
        *,
        layer: int,
        corpus_spec: str = "user",
        tokens: int = 1_000_000,
        seq_len: int = 128,
        batch_size: int = 8,
        d_sae: int | None = None,
        expansion: int = 8,
        learning_rate: float = 3e-4,
        l1_coefficient: float = 1e-3,
        dead_feature_threshold: float = 1e-6,
        seed: int = 0,
        force: bool = False,
        on_progress: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
        on_swap: SwapCallback | None = None,
    ) -> dict[str, Any]:
        """Train and persist a Drowse-owned residual-post SAE, then load it.

        Returns ``{source, artifact, metrics, runtime}``; ``runtime`` is the
        new residency's :attr:`info`.
        """
        from drowse.base.source_registry import LOCAL, SourceRef
        from drowse.model.arch import get_layers
        from drowse.sae.sources import SAE_SOURCES, sae_release
        from drowse.sae.store import save_local_sae
        from drowse.sae.training import train_residual_sae

        handle = self._handle
        local_name = name.strip()
        SAE_SOURCES.local_dir(handle.model_id, local_name)  # validates the name
        release = sae_release(SourceRef(LOCAL, local_name))
        docs = list(documents)
        if not docs:
            raise ValueError("SAE training needs at least one corpus document")
        corpus_sha = hashlib.sha256(repr(docs).encode("utf-8")).hexdigest()
        with self.exclusive(
            "SAE train called while another model use is in flight",
            phase_msg="SAE train called while a generation is in flight",
        ):
            self._assert_unsteered()
            model = handle.model
            fit_model: Any = getattr(model, "_orig_mod", model)
            tensors, metrics = train_residual_sae(
                fit_model,
                handle.tokenizer,
                list(get_layers(fit_model)),
                docs,
                layer=layer,
                tokens=tokens,
                seq_len=seq_len,
                batch_size=batch_size,
                d_sae=d_sae,
                expansion=expansion,
                learning_rate=learning_rate,
                l1_coefficient=l1_coefficient,
                dead_feature_threshold=dead_feature_threshold,
                seed=seed,
                on_progress=on_progress,
                cancel_event=cancel_event,
            )
            manifest = save_local_sae(
                handle.model_id,
                local_name,
                tensors,
                model_fingerprint=loaded_model_fingerprint(
                    fit_model, handle.model_id,
                ),
                model_source_fingerprint=getattr(
                    fit_model, "_drowse_source_fingerprint", None,
                ),
                layer=layer,
                corpus_spec=corpus_spec,
                corpus_sha256=corpus_sha,
                tokens_trained=int(metrics["tokens_trained"]),
                seq_len=seq_len,
                batch_size=batch_size,
                learning_rate=learning_rate,
                l1_coefficient=l1_coefficient,
                dead_feature_threshold=dead_feature_threshold,
                force=force,
            )
        runtime = self.load(release, layer=layer, on_swap=on_swap)
        return {
            "source": release,
            "artifact": str(manifest),
            "metrics": metrics,
            "runtime": runtime,
        }

    # ------------------------------------------------------------- features

    def check_feature(self, feature_id: int | str) -> tuple[int, SaeResidency]:
        """Parse ``feature_id`` and range-check it against the resident width.

        Local only: no metadata and no network.  This is the steering path's
        check (it runs under the exclusive-GPU section and needs only the
        decoder row).
        """
        try:
            idx = int(feature_id)
        except (TypeError, ValueError) as exc:
            raise SaeFeatureError(
                f"SAE feature id must be an integer: {feature_id!r}"
            ) from exc
        resident = self.require()
        if not 0 <= idx < resident.width:
            raise SaeFeatureError(
                f"SAE feature {idx} out of range [0, {resident.width}) "
                f"for layer {resident.layer}"
            )
        return idx, resident

    def validate_feature(
        self,
        feature_id: int | str,
        *,
        on_fetched: FetchedCallback | None = None,
    ) -> dict[str, Any]:
        """Range-check a feature and resolve its label and ``max_act``.

        Fetches from Neuronpedia (outside every lock) when the cache has no
        complete entry and the feature was never asked about.  Returns
        ``{id, label, layer, max_act}``.
        """
        idx, resident = self.check_feature(feature_id)
        with self.state_lock:
            entry = resident.feature_meta.get(str(idx))
        if _needs_fetch(entry):
            entry = self._fetch(resident, [idx], on_fetched).get(str(idx)) or entry
        entry = entry or {}
        with self.state_lock:
            if self._resident is not resident:
                raise SaeFeatureError(
                    "SAE source changed while loading feature metadata; try again"
                )
        return {
            "id": idx,
            "label": entry.get("label"),
            "layer": resident.layer,
            "max_act": entry.get("max_act"),
        }

    def fetch_meta(
        self,
        feature_ids: Sequence[int],
        *,
        on_fetched: FetchedCallback | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Fetch and cache Neuronpedia metadata for a set of feature ids.

        The dashboard's discovery backfill, run between generations with the
        ids the live top-k surfaced, never from the decode loop.  Cached ids
        skip the network; misses fetch concurrently.  Returns
        ``{str(id): {label, max_act}}`` for every requested id that has
        metadata afterwards (out-of-range ids are dropped: the top-k cannot
        produce one).  Empty when the source changed meanwhile.
        """
        with self.state_lock:
            resident = self.require()
            seen: set[int] = set()
            wanted: list[int] = []
            for raw in feature_ids:
                idx = int(raw)
                if not 0 <= idx < resident.width or idx in seen:
                    continue
                seen.add(idx)
                if _needs_fetch(resident.feature_meta.get(str(idx))):
                    wanted.append(idx)
        if wanted:
            self._fetch(resident, wanted, on_fetched)
        with self.state_lock:
            if self._resident is not resident:
                return {}
            return {
                str(idx): {"label": entry.get("label"), "max_act": entry.get("max_act")}
                for idx in sorted(seen)
                if (entry := resident.feature_meta.get(str(idx))) is not None
            }

    def _fetch(
        self,
        resident: SaeResidency,
        feature_ids: Sequence[int],
        on_fetched: FetchedCallback | None,
    ) -> dict[str, dict[str, Any]]:
        """Fetch ``feature_ids`` (no lock held), then cache what arrived.

        A transport failure leaves an id absent (retried next time).  The
        cache write, its disk mirror and ``on_fetched`` land only if
        ``resident`` is still the current residency.
        """
        from drowse.sae.sources import save_sae_feature_meta

        info = resident.info

        def fetch_one(idx: int) -> dict[str, Any] | None:
            return neuronpedia.fetch_feature(
                info.get("neuronpedia_id"), idx,
                repo_id=info.get("repo_id"), sae_id=info.get("sae_id"),
            )

        ids = list(feature_ids)
        if len(ids) == 1:
            rows = [fetch_one(ids[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(_FETCH_WORKERS, len(ids))) as pool:
                rows = list(pool.map(fetch_one, ids))
        fetched = {
            str(idx): entry for idx, entry in zip(ids, rows) if entry is not None
        }
        if not fetched:
            return {}
        with self.state_lock:
            if self._resident is not resident:
                return {}
            resident.feature_meta.update(fetched)
            save_sae_feature_meta(
                self._handle.model_id, resident.backend.release,
                resident.feature_meta, source=info,
            )
            if on_fetched is not None:
                on_fetched(fetched)
        return fetched

    def label(self, feature_id: int) -> str | None:
        """The cached Neuronpedia label, or ``None``."""
        resident = self._resident
        entry = resident.feature_meta.get(str(feature_id)) if resident else None
        label = entry.get("label") if entry else None
        return label if isinstance(label, str) and label else None

    def max_act(self, feature_id: int) -> float | None:
        """The cached Neuronpedia ``maxActApprox``, the strength unit.

        ``None`` when metadata is missing (offline / not on Neuronpedia); every
        readout surface for that feature then stays raw.
        """
        resident = self._resident
        entry = resident.feature_meta.get(str(feature_id)) if resident else None
        max_act = entry.get("max_act") if entry else None
        if isinstance(max_act, (int, float)) and float(max_act) > 0:
            return float(max_act)
        return None

    # ------------------------------------------------------------ tensors

    def feature_direction(
        self, feature_id: int | str,
    ) -> tuple[int, int, torch.Tensor]:
        """``(id, layer, W_dec[id])`` on the model's device and dtype.

        Range-checked locally (:meth:`check_feature`); never fetches.
        """
        idx, resident = self.check_feature(feature_id)
        direction = resident.backend.feature_direction(resident.layer, idx)
        handle = self._handle
        return idx, resident.layer, direction.detach().to(handle.device, handle.dtype)

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Encode one residual row at the resident layer -> ``(width,)``.

        Detached: SAE reads are diagnostics, probes and steering metadata,
        never a training surface, and some third-party backends return
        grad-tracked activations even in an inference-only session.
        """
        resident = self.require()
        row = hidden.unsqueeze(0) if hidden.ndim == 1 else hidden
        acts = resident.backend.encode_layer(resident.layer, row)
        if acts.ndim == 2 and acts.shape[0] == 1:
            acts = acts[0]
        acts = acts.reshape(-1)
        if int(acts.numel()) != resident.width:
            raise ValueError(
                f"SAE encoder returned {acts.numel()} features; "
                f"expected {resident.width}"
            )
        return acts.detach()


__all__ = [
    "SaeModelHandle",
    "SaeResidency",
    "SaeRuntime",
]
