"""The SAE instrument: the public face of the SAE family.

Owns everything SAE-shaped on a session (``session.sae``):

* the resident dictionary, through its :class:`~drowse.sae.runtime.SaeRuntime`
  (``load`` / ``unload`` / ``train`` / ``restore_active``, ``info``, feature
  validation and Neuronpedia metadata, ``register_direction`` for ``sae/<id>``
  steering atoms);
* the probe registry, the live-discovery config, the per-forward stash and the
  per-generation active flag;
* the read surfaces: attach, per-step scoring, gate scalars, the finalize
  aggregate, the live display step, authored-prefill computation and the
  loom-anchored token replay.

Dependencies are explicit constructor arguments: the read plane's capture
(the per-step reads take its latest slices), the runtime, the loom token
replay, and three registry callbacks for the ``sae/<id>`` steering directions
and the probe-hash cache.  The finalize aggregate takes the read plane's
pooled slice.

**An SAE probe emits exactly one channel**: ``<name>`` / ``<name>[0]``, the
normalized strength.  There is no ``:fraction`` or ``:membership`` — a
feature activation is not a geometry reading, and emitting constants under
those names would be measurement-shaped noise.  A gate on a channel the
family cannot produce is a composition-preflight error
(:meth:`SaeInstrument.validate_gate`), never a silently-constant comparison.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any, Mapping, TYPE_CHECKING

import torch

from drowse.base.lru import BoundedLRU
from drowse.readout.replay import TokenReplay
from drowse.readout.types import (
    Axis,
    DepthSummary,
    GateRef,
    InstrumentBinding,
    InstrumentFamily,
    InstrumentPlan,
    InstrumentPrep,
    ReadRequest,
    SaeLiveState,
    ScalarReading,
    UNIT_ACTIVATION_OVER_MAX,
    UNIT_RAW_ACTIVATION,
    next_prep_token,
    parse_gate_ref,
    scalar_gate_keys,
    validate_gate_channels,
)

if TYPE_CHECKING:
    from drowse.readout.protocol import LatestSlices
    from drowse.sae.runtime import SaeRuntime

#: Device selector tensors kept for pinned-probe gathers (one per roster).
_SELECTOR_CACHE_MAX = 64


class SaeRun:
    """Per-generation measurement executor for the SAE family.

    Owns the generation-scoped state (``protocol.py``): the immutable
    :class:`InstrumentBinding` — probe specs frozen at bind with the
    normalization unit (``max_act``) **resolved** into the snapshot — the
    live-discovery snapshot, the live-active flag, and the per-forward encode
    stash.  The frozen specs close the metadata-backfill race:
    :meth:`SaeInstrument.fetch_meta` mutates attached specs and the meta cache
    without the generation lock, so a running generation must measure against
    its bind-time snapshot, never the live registry.  An idle run
    (``bound=False``) reads the live registry — between generations a
    Neuronpedia refresh still changes the unit immediately.
    """

    def __init__(
        self,
        instrument: "SaeInstrument",
        binding: InstrumentBinding,
        *,
        active: bool = True,
        live_state: "Mapping[str, Any] | None" = None,
        bound: bool = False,
    ) -> None:
        self._instrument = instrument
        self.binding = binding
        self.active = active
        #: The live-discovery config snapshotted at bind — a library caller
        #: on another thread can toggle the instrument-level ``live``
        #: mid-decode, and a bound run must keep reading the state its
        #: generation started with (idle runs pass through instead).
        self.live_state = live_state
        self.bound = bound
        self.step_stash: dict[str, Any] | None = None
        self.last_step_readings: dict[str, ScalarReading] | None = None

    # ------------------------------------------------------------ protocol

    def gate_scalars(
        self,
        step_id: int,
        hidden: dict[int, Any] | None,
        gate_keys: frozenset[str] | set[str] | None,
    ) -> dict[str, float]:
        """The gate channels' scalars for this step (the encode stash the
        display step reuses lives on this run, keyed by ``step_id``).
        ``gate_keys=None`` scores the full roster (the session forwarder's
        bare-call shape)."""
        del hidden  # worker reads the capture's latest slices
        return self._instrument.gate_scalars(
            set(gate_keys) if gate_keys is not None else None,
            step_id=step_id,
        )

    def observe_aggregate(
        self, pooled: dict[int, Any],
    ) -> dict[str, ScalarReading]:
        """End-of-generation aggregate at the pooled last-content slice."""
        return self._instrument.score_probes(pooled)

    def close(self) -> None:
        """Release generation-scoped state (stash, live snapshot)."""
        self.step_stash = None
        self.last_step_readings = None
        self.active = True
        self.live_state = None


class SaeInstrument:
    """Session-lifetime handle for the SAE read family."""

    family: InstrumentFamily = "sae"

    #: Gate channels an SAE probe can produce: the one activation axis.
    _GATE_CHANNELS: tuple[type, ...] = (Axis,)

    def __init__(
        self,
        capture: "LatestSlices",
        runtime: "SaeRuntime",
        *,
        replay: TokenReplay,
        put_direction: Callable[[str, dict[int, torch.Tensor]], None],
        evict_directions: Callable[[], object],
        forget_probe_hashes: Callable[[Sequence[str]], None],
    ) -> None:
        # The read plane's capture: the per-step reads take its latest slices.
        self._capture = capture
        # Device feature-id selector tensors, keyed by (device, ids).
        self._index_tensors: BoundedLRU[
            tuple[str, tuple[int, ...]], torch.Tensor
        ] = BoundedLRU(_SELECTOR_CACHE_MAX)
        #: The resident dictionary and its lifecycle.
        self.runtime = runtime
        # The loom-anchored token replay (the producing forward's capture);
        # this family supplies the feature read.
        self._replay = replay
        # Registry effects of a residency change: ``sae/<id>`` directions are
        # registry profiles, and probe digests live in the probe roster.
        self._put_direction = put_direction
        self._evict_directions = evict_directions
        self._forget_probe_hashes = forget_probe_hashes
        # Pinned SAE feature probes: name -> {feature_id, layer, label,
        # max_act}.  NOT monitor probes — they read the encoder channel.
        self.probes: dict[str, dict[str, Any]] = {}
        # Live feature discovery: {layer, source}, or None when off. The
        # generation's resolved readout width is shared with logit alts and
        # J-lens, so it is deliberately not session-lifetime instrument state.
        self.live: dict[str, Any] | None = None
        # THE SAE-state boundary (the runtime's lock, shared): one reentrant
        # leaf lock covering registry mutation (attach/detach, the metadata
        # backfill, load/unload probe eviction together with the residency
        # swap), the live toggles, and the coherent snapshots
        # (specs/names/probe_hash, the plan roster read, the idle
        # ``_measurement_specs`` / ``_measurement_live`` copies).  An
        # un-locked reader iterating the registry tears under a concurrent
        # detach.  Never held on per-token paths — bound runs read their
        # frozen binding and live snapshot.
        self.state_lock = runtime.state_lock
        # The current per-generation run (idle passthrough until bind()).
        self.current_run = SaeRun(
            self, InstrumentBinding(family=self.family),
        )

    # ------------------------------------------------------------- lifecycle

    @property
    def info(self) -> dict[str, Any] | None:
        """The resident SAE (``{release, layer, width, revision, fingerprint,
        sae_id, repo_id, neuronpedia_id}``), or ``None`` when none is loaded."""
        return self.runtime.info

    def load(
        self, release: str, *, layer: int | None = None,
        revision: str | None = None,
    ) -> dict[str, Any]:
        """Make one local or SAELens source the resident SAE (see
        :meth:`SaeRuntime.load`).  Feature ids belong to the resident
        release, so a swap evicts ``sae/<id>`` directions and attached
        probes rather than silently reusing ids."""
        return self.runtime.load(
            release, layer=layer, revision=revision, on_swap=self._on_swap,
        )

    def unload(self) -> None:
        """Release the resident SAE and every feature handle keyed by it."""
        self.runtime.unload(on_swap=self._on_swap)

    def restore_active(self) -> None:
        """Reload the persisted ``sae use`` selection (non-fatal)."""
        self.runtime.restore_active(on_swap=self._on_swap)

    def train(
        self, name: str, documents: Sequence[str], *, layer: int,
        **options: Any,
    ) -> dict[str, Any]:
        """Train a local residual-post SAE and make it resident (see
        :meth:`SaeRuntime.train` for the options)."""
        return self.runtime.train(
            name, documents, layer=layer, on_swap=self._on_swap, **options,
        )

    def validate_feature(self, feature_id: int | str) -> dict[str, Any]:
        """``{id, label, layer, max_act}`` for one resident feature,
        fetching Neuronpedia metadata when it is not cached."""
        return self.runtime.validate_feature(
            feature_id, on_fetched=self._on_fetched,
        )

    def fetch_meta(
        self, feature_ids: Sequence[int],
    ) -> dict[str, dict[str, Any]]:
        """Fetch and cache Neuronpedia metadata for a set of feature ids
        (the dashboard's discovery backfill; see :meth:`SaeRuntime.fetch_meta`)."""
        return self.runtime.fetch_meta(feature_ids, on_fetched=self._on_fetched)

    def register_direction(self, feature_id: int | str) -> str:
        """Register ``W_dec[id]`` at the resident hook layer as the steering
        profile ``sae/<id>`` and return that name.

        Range-checked locally; never fetches metadata.  The steering composer
        calls this while it holds the exclusive-GPU lock, and a direction
        needs only the decoder row.
        """
        idx, layer, direction = self.runtime.feature_direction(feature_id)
        name = f"sae/{idx}"
        self._put_direction(name, {layer: direction})
        return name

    def _on_swap(self) -> None:
        """Drop everything keyed by the old release (state lock held, inside
        the exclusive section, residency already swapped)."""
        cleared = list(self.probes)
        self.probes.clear()
        self.live = None
        self.step_stash = None
        self.last_step_readings = None
        self._evict_directions()
        self._forget_probe_hashes(cleared)

    def _on_fetched(self, fetched: dict[str, dict[str, Any]]) -> None:
        """Reflect newly fetched metadata onto attached probes (state lock
        held).  ``max_act`` is part of the readout-channel identity (it sets
        the strength unit), so the probe's cached digest is dropped too."""
        refreshed: list[str] = []
        for name, spec in list(self.probes.items()):
            entry = fetched.get(str(spec.get("feature_id")))
            if entry is None:
                continue
            # Whole-dict replacement (one atomic store): idle readers hold
            # per-call spec snapshots or shared references, and a field-level
            # mutation could hand them a half-updated unit.
            self.probes[name] = {
                **spec,
                "label": entry.get("label"),
                "max_act": entry.get("max_act"),
            }
            refreshed.append(name)
        self._forget_probe_hashes(refreshed)

    # ------------------------------------------------------------- run state

    @property
    def step_stash(self) -> dict[str, Any] | None:
        return self.current_run.step_stash

    @step_stash.setter
    def step_stash(self, value: "dict[str, Any] | None") -> None:
        self.current_run.step_stash = value

    @property
    def last_step_readings(self) -> "dict[str, ScalarReading] | None":
        return self.current_run.last_step_readings

    @last_step_readings.setter
    def last_step_readings(
        self, value: "dict[str, ScalarReading] | None",
    ) -> None:
        self.current_run.last_step_readings = value

    @property
    def active_for_generation(self) -> bool:
        return self.current_run.active

    @active_for_generation.setter
    def active_for_generation(self, value: bool) -> None:
        self.current_run.active = bool(value)

    # ------------------------------------------------------------ run lifecycle

    def prepare(self, request: ReadRequest) -> InstrumentPrep:
        """Generation-boundary prep — no disk-backed source to refresh
        (backend residency is explicit lifecycle, :meth:`load` /
        :meth:`train`) and no lens-style identity coupling between the
        registry and the source, so the prep only carries the request
        forward for ``plan``/``bind``.  The bound-run guard keeps the
        transaction contract uniform across families."""
        if self.current_run.bound:
            raise RuntimeError(
                "SaeInstrument.prepare() on a bound run: close the prior "
                "generation's run (ReadPlane.close_runs) first"
            )
        return InstrumentPrep(
            family=self.family,
            request=request,
            token=next_prep_token(),
        )

    def bind(self, plan: InstrumentPlan, prep: InstrumentPrep) -> SaeRun:
        """Bind an immutable per-generation run from a declared plan.

        The binding freezes each attached probe's spec **with its
        normalization unit resolved**: ``max_act`` falls back to the
        session metadata cache at bind time, so a mid-generation
        Neuronpedia backfill (which mutates specs and cache without the
        generation lock) cannot change what a running generation measures.

        The live-discovery config is snapshotted in the same ``state_lock``
        hold, for the same reason the lens snapshots its live state: the
        toggles are reachable from any thread (the HTTP routes serialize
        behind the session lock, a library caller does not), and a flip
        landing mid-decode must not change what the running generation reads.
        """
        if prep.family != self.family:
            raise TypeError(
                "SaeInstrument.bind takes the InstrumentPrep its own "
                f"prepare() returned, got family={prep.family!r}"
            )
        if plan.family != self.family:
            raise ValueError(
                f"SaeInstrument.bind: plan family {plan.family!r} is not "
                f"{self.family!r}"
            )
        if plan.prep_token != prep.token:
            raise ValueError(
                "SaeInstrument.bind: the plan was not derived from this "
                "prep (prep_token mismatch) — derive the plan from the same "
                "prepare() call"
            )
        live_active = prep.request.live
        runtime = self.runtime
        specs: dict[str, dict[str, Any]] = {}
        with self.state_lock:
            probe_items = list(self.probes.items())
            live_state = dict(self.live) if self.live is not None else None
            resident = runtime.resident
        for name, spec in probe_items:
            frozen = dict(spec)
            max_act = frozen.get("max_act")
            if not (isinstance(max_act, (int, float)) and float(max_act) > 0):
                frozen["max_act"] = runtime.max_act(int(frozen["feature_id"]))
            specs[name] = frozen
        backend = resident.backend if resident is not None else None
        run = SaeRun(
            self,
            InstrumentBinding(
                family=self.family,
                source=getattr(backend, "release", None),
                fingerprint=getattr(backend, "revision", None),
                specs=specs,
            ),
            active=live_active,
            live_state=live_state,
            bound=True,
        )
        self.current_run = run
        return run

    def close_run(self) -> None:
        """Close the current run and restore the idle passthrough run."""
        self.current_run.close()
        self.current_run = SaeRun(
            self, InstrumentBinding(family=self.family),
        )

    def _measurement_specs(self) -> "Mapping[str, Mapping[str, Any]]":
        """The spec source for measurement: a bound run's frozen binding
        (race-immune), else a per-call coherent snapshot of the live
        registry (idle passthrough).  Handing out the live dict lets an idle
        read tear mid-iteration under a concurrent detach, and a metadata
        backfill can rewrite a unit mid-read."""
        run = self.current_run
        if run.bound:
            return run.binding.specs
        with self.state_lock:
            return {
                name: dict(spec) for name, spec in self.probes.items()
            }

    def _measurement_live(self) -> "Mapping[str, Any] | None":
        """The live-discovery state for measurement: a bound run's bind-time
        snapshot (a toggle from another thread must not change what a running
        generation reads), else a coherent copy of the live config."""
        run = self.current_run
        if run.bound:
            return run.live_state
        with self.state_lock:
            return dict(self.live) if self.live is not None else None

    # -------------------------------------------------------------- registry

    def attach(self, selector: str, *, as_name: str | None = None) -> str:
        """Attach a resident SAE feature as a one-channel readout probe.

        Validation may fetch the feature's Neuronpedia metadata (outside
        every lock): ``max_act`` is the probe's strength unit.  Cache
        invalidation (probe-hash cache, analytics) is the probe roster's
        job (:class:`~drowse.readout.roster.ProbeRoster`).
        """
        raw_id = selector.split("/", 1)[1]
        validated = self.validate_feature(raw_id)
        idx = int(validated["id"])
        name = as_name if as_name is not None else f"sae/{idx}"
        with self.runtime.exclusive(
            "add_probe called while another model operation is in flight; retry shortly"
        ):
            with self.state_lock:
                self.probes[name] = {
                    "feature_id": idx,
                    "layer": int(validated["layer"]),
                    "label": validated.get("label"),
                    "max_act": validated.get("max_act"),
                }
        return name

    def try_detach(self, name: str) -> bool:
        """Atomic membership-check + detach under the registry lock — the
        family's only removal surface (the session's ``remove_probe``
        dispatch; a bare check + direct delete is two un-serialized
        registry touches)."""
        with self.state_lock:
            if name not in self.probes:
                return False
            del self.probes[name]
            return True

    def specs(self) -> dict[str, dict[str, Any]]:
        """Snapshot of attached probe specifications (coherent — a
        concurrent detach or backfill cannot tear the iteration)."""
        with self.state_lock:
            return {name: dict(spec) for name, spec in self.probes.items()}

    @property
    def names(self) -> list[str]:
        with self.state_lock:
            return list(self.probes)

    def validate_gate(self, ref: GateRef) -> None:
        validate_gate_channels(ref, self._GATE_CHANNELS, family=self.family)

    # ---------------------------------------------------------------- planning

    def plan(self, prep: InstrumentPrep) -> InstrumentPlan:
        """Declare the SAE family's capture demand for one generation.

        The family reads one resident hook layer: live discovery captures
        the live config's layer; pinned probes capture the resident
        runtime layer whenever a finalize aggregate or an active gate will
        read them (consulted only when probes are attached).
        The prep carries the request.  Unlike the lens there is no
        registry↔source identity to snapshot at prepare, so demand reads the
        live registry and live config under one ``state_lock`` hold; a detach
        or toggle racing that read only shrinks or widens capture, because
        ``bind`` freezes what the run actually measures.
        """
        if prep.family != self.family:
            raise TypeError(
                "SaeInstrument.plan takes the InstrumentPrep its own "
                f"prepare() returned, got family={prep.family!r}"
            )
        request = prep.request
        with self.state_lock:
            live = (
                dict(self.live)
                if (request.live and self.live is not None) else None
            )
            probes = dict(self.probes)
        gate_keys = frozenset(
            key for key in request.gate_keys
            if parse_gate_ref(key).probe in probes
        )
        latest: set[int] = set()
        tail: set[int] = set()
        if live is not None:
            latest.add(int(live["layer"]))
        if probes and (request.final_aggregate or gate_keys):
            layer = self.runtime.layer
            if layer is not None:
                latest.add(int(layer))
                if request.final_aggregate:
                    tail.add(int(layer))
        return InstrumentPlan(
            family=self.family,
            latest_layers=frozenset(latest),
            tail_layers=frozenset(tail),
            gate_keys=gate_keys,
            final_aggregate=bool(probes and request.final_aggregate),
            prep_token=prep.token,
        )

    def probe_hash(self, name: str) -> str | None:
        """Readout-channel identity digest.

        v2: the channel is normalized strength (activation / maxActApprox)
        when metadata exists, raw activation otherwise — ``max_act`` is
        part of the channel identity (it sets the unit), so drift
        detection catches a unit change.
        """
        with self.state_lock:
            spec = self.probes.get(name)
            if spec is None:
                return None
            spec = dict(spec)
        runtime = self.runtime
        info = runtime.info or {}
        return hashlib.sha256(repr((
            "sae-readout-v2", runtime.model_id, info.get("fingerprint"),
            info.get("release"), spec["layer"],
            spec["feature_id"],
            runtime.max_act(int(spec["feature_id"])),
        )).encode("utf-8")).hexdigest()

    # ---------------------------------------------------------------- scoring

    def score_probes(
        self, hidden: dict[int, torch.Tensor] | None = None,
        *, only: "set[str] | None" = None,
    ) -> dict[str, ScalarReading]:
        """Score attached probes from capture hidden slices.

        The capture-slice entry: encodes the resident hook layer itself.
        Callers holding this forward's encode already use
        :meth:`score_probes_from_activations`.
        """
        if not self._measurement_specs():
            return {}
        layer = self.runtime.require().layer
        if hidden is None or layer not in hidden:
            return {}
        return self.score_probes_from_activations(
            self.runtime.encode(hidden[layer]), only=only,
        )

    def score_probes_from_activations(
        self,
        activations: torch.Tensor,
        *,
        only: "set[str] | None" = None,
        raw_by_fid: Mapping[int, float] | None = None,
    ) -> dict[str, ScalarReading]:
        """Score attached probes from an already-encoded activation vector.

        The entry for callers that encoded this forward once and share it
        (the gate callback, the live display step, authored prefill);
        ``raw_by_fid`` additionally seeds values a caller already transferred.

        Returns the family-native :class:`ScalarReading`: ONE channel whose
        ``unit`` says which normalization actually applied (the Neuronpedia
        corpus max when cached, raw activation otherwise) and whose depth
        summary names its single-layer basis.  A feature activation is not
        a geometry reading, so it carries none of those fields.
        """
        specs = self._measurement_specs()
        if not specs:
            return {}
        layer = self.runtime.require().layer
        live_fallback = not self.current_run.bound
        depth = DepthSummary(
            center=(layer / max(self.runtime.n_layers - 1, 1),),
            spread=(0.0,),
            basis="single_layer",
        )
        out: dict[str, ScalarReading] = {}
        for name, fid, _raw_value, value in self.probe_values(
            activations, only=only, raw_by_fid=raw_by_fid,
        ):
            normalized = self._resolve_max_act(
                specs.get(name) or {}, fid, live_fallback,
            ) is not None
            out[name] = ScalarReading(
                value=float(value),
                unit=(
                    UNIT_ACTIVATION_OVER_MAX if normalized
                    else UNIT_RAW_ACTIVATION
                ),
                per_layer={int(layer): float(value)},
                depth=depth,
            )
        return out

    def _resolve_max_act(
        self, spec: "Mapping[str, Any]", fid: int, live_fallback: bool,
    ) -> float | None:
        """The feature's normalization constant, or ``None`` when there is
        none (offline / unlisted feature -> the reading is raw activation).

        A bound run's unit was resolved into the binding snapshot at bind,
        so the un-locked Neuronpedia backfill cannot change a running
        generation's unit; the live cache fallback applies only between
        generations.  ONE resolver, shared by the value path and the unit
        the reading reports, so the two cannot disagree.
        """
        max_act = spec.get("max_act")
        if isinstance(max_act, (int, float)) and float(max_act) > 0:
            return float(max_act)
        if live_fallback:
            return self.runtime.max_act(int(fid))
        return None

    def probe_values(
        self,
        activations: torch.Tensor,
        *,
        only: "set[str] | None" = None,
        raw_by_fid: Mapping[int, float] | None = None,
    ) -> list[tuple[str, int, float, float]]:
        """Pinned SAE probe values as ``(name, fid, raw, normalized)``.

        ``raw_by_fid`` is a per-forward cache from a caller that has already
        transferred selected feature activations (currently the live top-k
        readout). Normalization reads the **measurement specs**: a bound
        run's bind-time snapshot (a mid-generation Neuronpedia backfill
        cannot change the unit under a running generation), else the live
        registry (between generations a refresh changes the unit
        immediately, as before).
        """
        specs = self._measurement_specs()
        live_fallback = not self.current_run.bound
        names = [
            name for name in specs
            if only is None or name in only
        ]
        if not names:
            return []
        fids = [int(specs[name]["feature_id"]) for name in names]
        raw_values_by_fid: dict[int, float] = {
            int(fid): float(value)
            for fid, value in (raw_by_fid or {}).items()
        }
        missing_fids = [
            fid for fid in fids
            if fid not in raw_values_by_fid
        ]
        if missing_fids:
            device = activations.device
            fid_tensor = self._index_tensors.get_or_build(
                (str(device), tuple(missing_fids)),
                lambda: torch.tensor(missing_fids, dtype=torch.long, device=device),
            )
            # One host transfer for every not-already-read pinned SAE probe
            # value.  Live readout top-k rows seed ``raw_by_fid`` first, so
            # pinned cards that came from the visible top-k avoid a second
            # selected-feature gather + CPU transfer.
            raw_values = (
                activations.index_select(0, fid_tensor)
                .detach()
                .to("cpu")
                .tolist()
            )
            for fid, raw_value in zip(missing_fids, raw_values, strict=True):
                raw_values_by_fid[int(fid)] = float(raw_value)
        out: list[tuple[str, int, float, float]] = []
        for name, fid in zip(names, fids, strict=True):
            spec = specs[name]
            raw_value = float(raw_values_by_fid[fid])
            value = raw_value
            # The ONE channel is normalized strength — ``activation /
            # maxActApprox`` ∈ ~[0,1], apples-to-apples across features like
            # the lens probes' mean fitted-layer probability. Raw activation
            # only when no metadata exists (offline / not on Neuronpedia).
            # A bound run's unit was resolved into the snapshot at bind; the
            # live cache fallback applies only between generations.
            max_act = self._resolve_max_act(spec, fid, live_fallback)
            if max_act is not None:
                value = value / max_act
            out.append((name, fid, raw_value, value))
        return out

    def gate_scalars(
        self, gate_keys: "set[str] | None" = None, *, step_id: int = -1,
    ) -> dict[str, float]:
        """Per-forward SAE gate scalars from the latest capture slice.

        Emits ONLY the channels an SAE probe actually measures — ``<name>``
        and ``<name>[0]`` (the normalized strength axis).  The historical
        fake constants (``:fraction`` 0.0 / ``:membership`` 1.0) are gone:
        a gate referencing a channel this family can never produce is a
        composition-preflight error (:meth:`validate_gate`), never a
        silently-constant comparison.  The encode stash is keyed by
        ``step_id``, so the display step reuses it iff it came from the
        same forward: staleness is structural, reuse is idempotent.
        """
        specs = self._measurement_specs()
        if not specs:
            return {}
        layer = self.runtime.require().layer
        only = None
        if gate_keys is not None:
            # ``None`` is the full-roster sentinel; an explicit empty set
            # means "no gated probes" and scores nothing — the
            # None-vs-empty distinction every family's gate entry keeps.
            only = {
                key.split("[", 1)[0]
                for key in gate_keys
                if key.split("[", 1)[0] in specs
            }
            if not only:
                return {}
        latest = self._capture.latest_per_layer()
        if layer not in latest:
            return {}
        acts = self.runtime.encode(latest[layer])
        values = self.probe_values(acts, only=only)
        self.step_stash = {
            "activations": acts,
            "step": step_id,
            "raw_by_fid": {
                int(fid): float(raw_value)
                for _name, fid, raw_value, _value in values
            },
        }
        return scalar_gate_keys({
            name: value for name, _fid, _raw_value, value in values
        })

    def score_aggregate(
        self, generated_ids: list[int], pooled: dict[int, torch.Tensor],
    ) -> dict[str, ScalarReading]:
        """End-of-gen aggregate at ``pooled``, the read plane's slice at the
        last content token (shared with the monitor roster and the lens
        family, so all three aggregates read the same position)."""
        # Binding-authoritative guard: a probe detached mid-generation
        # stays in this generation's aggregate roster (mutations apply
        # next generation).
        if not self._measurement_specs() or not generated_ids or not pooled:
            return {}
        return self.current_run.observe_aggregate(pooled)

    # ----------------------------------------------------------- live readout

    def enable_live(self) -> dict[str, Any]:
        """Enable the one-matvec live feature readout at the resident layer."""
        from drowse.sae.sources import SAE_SOURCES, parse_sae_release

        resident = self.runtime.require()
        layer = resident.layer
        source = SAE_SOURCES.format(parse_sae_release(str(resident.backend.release)))
        with self.state_lock:
            self.live = {
                "layer": layer,
                "source": source,
            }
        return {"layer": layer}

    def disable_live(self) -> None:
        with self.state_lock:
            self.live = None

    @property
    def is_live(self) -> bool:
        with self.state_lock:
            return self.live is not None

    @property
    def active_source(self) -> str | None:
        """The resident SAE release normalized to ``local:``/``saelens:``.

        Residency is the source here: an SAE with no weights loaded has no
        active source, and :attr:`info` is the one place that identity lives.
        """
        from drowse.sae.sources import SAE_SOURCES, parse_sae_release

        info = self.runtime.info
        if not info:
            return None
        release = info.get("release")
        if not release:
            return None
        return SAE_SOURCES.format(parse_sae_release(str(release)))

    @property
    def live_state(self) -> SaeLiveState:
        with self.state_lock:
            live = self.live
            if live is None:
                return SaeLiveState(enabled=False)
            return SaeLiveState(
                enabled=True,
                layer=(
                    int(live["layer"]) if live.get("layer") is not None
                    else None
                ),
                source=live.get("source"),
            )

    def set_live(self, enabled: bool, **kwargs: Any) -> SaeLiveState:
        """Toggle the live feature-discovery readout.

        Takes no family extras: the readout width follows the generation's
        resolved alternatives (``return_top_k``), and the layer is the
        resident hook layer — neither is an instrument-local dial.
        """
        if kwargs:
            raise TypeError(
                "sae live takes no extras (readout width follows the "
                f"generation's alternatives), got {sorted(kwargs)}"
            )
        if enabled:
            self.enable_live()
        else:
            self.disable_live()
        return self.live_state

    # -------------------------------------------------------------- replay

    def token_readout(
        self,
        node_id: str,
        raw_index: int,
        *,
        top_k: int | None = None,
        layers: "list[int] | str | None" = None,
        apply_steering: bool = True,
        raw: bool = False,
    ) -> dict[str, Any]:
        """The loom-anchored SAE feature replay, as the finished
        ``scope="replay"`` measurement envelope.

        Rebuilds the forward that produced the token (under the node's
        recorded steering when ``apply_steering``; ``raw`` selects the flat
        render) and reads the top-``top_k`` features at the resident hook
        layer.  ``layers`` is **rejected**, not ignored: the SAE reads its
        ONE resident hook layer, so a layer selection is a request this
        family cannot honor.
        """
        from drowse.readout.measurements import build_measurements

        if layers is not None:
            raise ValueError(
                "sae token-readout takes no layers (the readout is at the "
                "resident hook layer)"
            )
        k = 8 if top_k is None else int(top_k)
        resident = self.runtime.require()
        layer = resident.layer
        if not 1 <= k <= min(resident.width, 256):
            raise ValueError(f"top_k must be in [1, {min(resident.width, 256)}]")

        def read(hidden: dict[int, torch.Tensor]) -> list[tuple[int, float]]:
            acts = self.runtime.encode(hidden[layer])
            values, indices = torch.topk(acts, k=k)
            return [
                (int(idx), float(value))
                for value, idx in zip(values.cpu(), indices.cpu())
            ]

        token, top = self._replay(
            node_id, raw_index, layers=[layer], read=read,
            apply_steering=apply_steering, raw=raw,
            label="session.sae.token_readout",
        )
        measurements = build_measurements(
            scope="replay",
            provenance="replayed",
            sae_features=[
                (fid, activation, self.runtime.label(fid), self.runtime.max_act(fid))
                for fid, activation in top
            ],
            sae_source=self.active_source,
            sae_layer=layer,
            steering=token.steering,
        )
        return {"measurements": measurements}

    def live_readout_step(
        self, *, top_k: int = 8, step_id: int = -1,
    ) -> list[tuple[int, float, str | None, float | None]] | None:
        """One decode step's live feature top-k from the latest capture slice.

        Reuses the gate callback's encoded activations + raw values when the
        stash came from THIS forward (``stash["step"] == step_id`` — one
        encode shared by gates, pinned probes, and the live display on a
        step; staleness is structural, so reuse is idempotent and
        ``step_id < 0`` never matches).
        """
        state = self._measurement_live()
        if state is None or not self.active_for_generation:
            return None
        layer = int(state["layer"])
        buckets = self._capture.per_layer_buckets()
        if not buckets.get(layer):
            return None
        stash = self.step_stash
        stashed_raw_by_fid: dict[int, float] = {}
        if (
            stash is not None
            and step_id >= 0
            and stash.get("step") == step_id
        ):
            acts = stash["activations"]
            stashed_raw_by_fid = {
                int(fid): float(value)
                for fid, value in (stash.get("raw_by_fid") or {}).items()
            }
        else:
            acts = self.runtime.encode(buckets[layer][-1])
        k = min(int(top_k), int(acts.numel()))
        values, indices = torch.topk(acts, k=k)
        value_list = values.detach().to("cpu").tolist()
        id_list = indices.detach().to("cpu").tolist()
        raw_by_fid = {
            fid: value for fid, value in zip(id_list, value_list, strict=True)
        }
        if stashed_raw_by_fid:
            raw_by_fid = {**stashed_raw_by_fid, **raw_by_fid}
        if self._measurement_specs():
            self.last_step_readings = self.score_probes_from_activations(
                acts, raw_by_fid=raw_by_fid,
            )
        else:
            self.last_step_readings = None
        # Rows carry ``max_act`` (cached-only — the decode loop never fetches)
        # so clients can render the normalized 0..1 strength beside the raw
        # activation; ``None`` until the metadata backfill lands.
        return [
            (
                int(idx),
                float(value),
                self.runtime.label(int(idx)),
                self.runtime.max_act(int(idx)),
            )
            for value, idx in zip(value_list, id_list, strict=True)
        ]

    # ------------------------------------------------------- authored prefill

    def authored_capture(
        self,
        hidden: dict[int, torch.Tensor],
        *,
        top_k: int,
    ) -> tuple[
        list[tuple[int, float, str | None, float | None]],
        dict[str, ScalarReading],
    ] | None:
        """Live SAE payload for one retained authored producer row.

        Computation only — the token-matching orchestration stays in the
        session's authored-prefill path.
        """
        state = self._measurement_live()
        if state is None or not self.active_for_generation:
            return None
        layer = int(state["layer"])
        if layer not in hidden:
            return None
        activations = self.runtime.encode(hidden[layer])
        k = min(int(top_k), int(activations.numel()))
        values, indices = torch.topk(activations, k=k)
        value_list = values.detach().to("cpu").tolist()
        id_list = indices.detach().to("cpu").tolist()
        raw_by_fid = {
            int(fid): float(value)
            for fid, value in zip(id_list, value_list, strict=True)
        }
        readings = self.score_probes_from_activations(
            activations, raw_by_fid=raw_by_fid,
        ) if self._measurement_specs() else {}
        rows = [
            (
                int(fid),
                float(value),
                self.runtime.label(int(fid)),
                self.runtime.max_act(int(fid)),
            )
            for value, fid in zip(value_list, id_list, strict=True)
        ]
        return rows, readings


__all__ = ["SaeInstrument"]
