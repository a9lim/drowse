"""Shared vocabulary for the read-side instruments.

The three read families — geometry (Monitor subspace probes), the Jacobian
lens readout channel, and SAE feature reads — speak the types defined here.
:mod:`drowse.readout.protocol` carries the contract they share.

* **GateRef** is the structured view of a probe-gate scalar key.  The
  steering grammar stores the verbatim string in ``ProbeGate.probe`` and the
  per-step runtime lookup is a plain ``dict.get`` on that string (hot-path
  discipline) — ``GateRef`` exists for *composition preflight*: a family
  validates that a referenced channel is one it can actually produce, so
  ``@when:sae/123:membership`` is a 400 at compose time instead of a
  silently-inactive gate.  ``parse_gate_ref`` is the ONE place the key-shape
  discrimination lives (it inverts the key family ``Monitor.flat_scalars``
  emits).
* **ScalarReading** is the honest reading shape for the single-channel
  families (lens strength, SAE feature activation).  The geometry family
  carries the full :class:`ProbeReading`.
  ``ScalarReading.to_probe_reading()`` is the compatibility projection
  older callers consume; the versioned measurement envelope carries the
  family-native scalar shape.
* **DepthSummary carries its basis** because ``depth_com`` means three
  mathematically unrelated things across families (share-weighted
  coordinate mass / readout probability mass / a single-layer constant);
  a bare float invites cross-family comparison that is meaningless.
* **InstrumentPlan declares capture demand, not mechanics** — the
  session-side capture planner unions plans and picks physical retention
  (incremental vs tail ring vs full stack); the
  ``INCREMENTAL -> set_tail_with_sink`` upgrade is cross-instrument
  resource sharing and stays out of the families.
* **InstrumentBinding is an immutable per-generation snapshot** of
  source identity + attached specs, so mid-generation mutations (e.g.
  the SAE metadata backfill, which runs without the generation lock)
  cannot change what a running generation is measuring.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Mapping, Union

from drowse.errors import UnsupportedProbeChannelError


#: The three read families.  THE enumeration — ``session.instruments`` is
#: keyed by it, the server validates ``{family}`` against that registry, and
#: the payload layer's per-family slots are pinned to it by test.
InstrumentFamily = Literal["geometry", "lens", "sae"]


#: Depth of the bounded capture tail ring finalize aggregates pool from —
#: deep enough to walk back past trailing special tokens to the last
#: content token.  It lives in the plan vocabulary because a family's
#: ``tail_layers`` demand is only meaningful at this depth; the session
#: planner's own constant aliases this value.
AGG_TAIL_DEPTH = 8

# ONE process-wide per-preparation token sequence.  Per-instance sequences
# collide across instrument instances (both start at 1), which lets a plan
# from one instrument bind with a prep from another; a single shared
# sequence makes every token unique process-wide.
_PREP_TOKENS = itertools.count(1)


def next_prep_token() -> int:
    """Draw the next process-wide unique per-preparation token."""
    return next(_PREP_TOKENS)

DepthBasis = Literal[
    # geometry: mass = share_weight_L * |coord_L[axis]| (monitor _depth_stats)
    "share_weighted_coord_mass",
    # lens: mass = the token's per-layer readout probability p_l
    "readout_probability_mass",
    # sae: the resident hook layer's normalized depth, a constant
    "single_layer",
]


# --------------------------------------------------------------------------
# Gate channels
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Axis:
    """A coordinate axis of the probe's primary reading (axis 0 is the
    bare-name channel; lens/SAE readings have exactly this one axis)."""

    index: int = 0


@dataclass(frozen=True)
class Fraction:
    """The manifold subspace-fraction channel (``<probe>:fraction``)."""


@dataclass(frozen=True)
class Membership:
    """The fuzzy-manifold tube-density channel (``<probe>:membership``)."""


@dataclass(frozen=True)
class Distance:
    """Negated whitened distance to a named node (``<probe>@<label>``)."""

    label: str


@dataclass(frozen=True)
class Assignment:
    """Soft node-assignment probability (``<probe>~<label>``)."""

    label: str


GateChannel = Union[Axis, Fraction, Membership, Distance, Assignment]

_CHANNEL_WORDS = {
    Axis: "coordinate axis",
    Fraction: "subspace fraction",
    Membership: "tube membership",
    Distance: "label distance",
    Assignment: "soft assignment",
}


@dataclass(frozen=True)
class GateRef:
    """Structured view of one probe-gate scalar key.

    ``probe`` is the full attached probe name exactly as registered
    (namespace and variant segments included — ``jlens/fake``,
    ``default/emotions``, ``pirate:role-x``).  ``scalar_key()`` formats
    back to the canonical string ``Monitor.flat_scalars`` emits.

    Round-trip: ``parse_gate_ref(ref.scalar_key()) == ref`` always;
    ``parse_gate_ref(key).scalar_key() == key`` except that an explicit
    ``[0]`` axis normalizes to the bare-name form (both spellings read
    the same channel).
    """

    probe: str
    channel: GateChannel

    def scalar_key(self) -> str:
        ch = self.channel
        if isinstance(ch, Axis):
            return self.probe if ch.index == 0 else f"{self.probe}[{ch.index}]"
        if isinstance(ch, Fraction):
            return f"{self.probe}:fraction"
        if isinstance(ch, Membership):
            return f"{self.probe}:membership"
        if isinstance(ch, Distance):
            return f"{self.probe}@{ch.label}"
        return f"{self.probe}~{ch.label}"

    def describe_channel(self) -> str:
        return _CHANNEL_WORDS[type(self.channel)]


def parse_gate_ref(key: str) -> GateRef:
    """Parse a verbatim gate scalar key into its structured form.

    The inverse of the key family ``Monitor.flat_scalars`` emits.  Safe
    on namespaced (``ns/name``) and variant-suffixed (``name:role-x``)
    probe names: ``@`` and ``~`` cannot appear inside a probe name or a
    node label (``NAME_REGEX`` forbids both), and ``:`` is treated as a
    channel separator only for the two exact channel words.
    """
    if key.endswith("]"):
        head, bracket, idx = key[:-1].rpartition("[")
        if bracket and head and idx.isdigit():
            return GateRef(head, Axis(int(idx)))
    if key.endswith(":fraction"):
        return GateRef(key[: -len(":fraction")], Fraction())
    if key.endswith(":membership"):
        return GateRef(key[: -len(":membership")], Membership())
    for sep, ctor in (("@", Distance), ("~", Assignment)):
        if sep in key:
            head, _, label = key.partition(sep)
            if head and label:
                return GateRef(head, ctor(label))
    return GateRef(key, Axis(0))


def validate_gate_channels(
    ref: GateRef,
    allowed: tuple[type, ...],
    *,
    family: str,
) -> None:
    """Raise :class:`UnsupportedProbeChannelError` when ``ref.channel``
    is not one of ``allowed`` for this family.

    The runtime rule is unchanged for *supported* channels: a valid
    channel whose value is temporarily absent this step (prefill, capture
    not yet available) stays quietly inactive.  This guard rejects only
    configuration that can never fire.
    """
    if not isinstance(ref.channel, allowed):
        raise UnsupportedProbeChannelError(
            f"probe gate '{ref.scalar_key()}' references the "
            f"{ref.describe_channel()} channel, which a {family} probe "
            f"does not produce"
        )


# --------------------------------------------------------------------------
# Readings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DepthSummary:
    """Per-axis depth center of mass (+ spread) with its mass basis.

    ``basis`` names what the mass *is* — the three families weight depth
    by mathematically unrelated quantities, so a consumer must never
    compare centers across bases.
    """

    center: tuple[float, ...]
    spread: tuple[float, ...]
    basis: DepthBasis


#: The lens strength unit — ``mean_l p_l(v)`` over the fitted layers.
UNIT_MEAN_TOKEN_PROBABILITY = "mean_token_probability"
#: The SAE strength unit when Neuronpedia ``maxActApprox`` is cached.
UNIT_ACTIVATION_OVER_MAX = "activation_over_max"
#: The SAE strength unit with no metadata (offline / unlisted feature).
UNIT_RAW_ACTIVATION = "raw_activation"


@dataclass
class ProbeReading:
    """Subspace-probe reading for one attached fit (flat or curved).

    The **single** unified readout: a live per-token reading (gate / stream)
    and the end-of-generation aggregate are the same shape — the aggregate
    is just this reading evaluated at the pooled last-content token.  Every
    field is populated for both flat and curved fits (the old flat/curved
    asymmetry — flat-no-nearest, curved-no-coords — is gone; both run the
    full per-layer geometry each token):

    * ``coords`` — EV-weighted whitened in-subspace coordinate in the fit's
      domain frame (``M_R⁻¹ B Σ⁻¹ (h − mean)`` mapped through the affine
      inverse for a flat fit, the ``invert_parameterization`` foot solve for
      a curved fit), an ``R``-tuple for a rank-``R`` fit (a 1-tuple for a
      2-node concept axis).  Neutral-anchored, pole-normalized at rank-1
      (``1.0`` at the positive node).
    * ``fraction`` — ``||P_M(h−mean)||_M / ||h−mean||_M`` EV-weighted across
      layers, the share of the centered activation living in the subspace,
      ∈ [0, 1].
    * ``nearest`` — top-N node labels by EV-weighted whitened distance,
      ascending (ranked by **raw** whitened distance — literally nearest).  The
      reported distance is rescaled into the probe's **typical label-spacing**
      units (divided by ``AttachedManifoldProbe.label_scale``, the median node
      nearest-neighbor whitened distance), so an ``@label`` gate threshold is
      portable across probes (raw whitened distance spans ~60× by fit); the scale
      is a per-probe constant, so the ranking is unchanged.  The synthetic label
      ``"neutral"`` competes here as the frame anchor (the per-model neutral
      mean): it is never a stored node, only a candidate the readout ranks, so it
      surfaces when the activation sits closer to the origin than to any node.
      Suppressed when a manifold already carries a real node named ``neutral``.
    * ``residual`` — EV-weighted normalized off-manifold residual
      ``dist_L / ||c_L||``; identically ``0.0`` for a flat fit (the surface
      fills its subspace), the off-surface distance for a curved fit.

    The ``*_per_layer`` maps carry the un-EV-collapsed per-layer trace of
    the same three geometric quantities (coords / fraction / residual).

    **Fuzzy-manifold readout** (the soft, distributional view of ``nearest`` /
    ``residual``):

    * ``assignment`` — a soft node-assignment posterior: ``(label, prob)`` over
      the candidate nodes (+ the neutral anchor),
      ``softmax(−d²_M / 2τ² − R·log(τ))`` with a per-node bandwidth ``τ`` and
      Gaussian log-volume correction (a curved fit's within-node σ-field mapped into the
      whitened metric, a flat fit's local layout scale).  The *distributional*
      counterpart to the argmax ``nearest`` — ships the shape instead of the
      winner.  Top-N by probability, descending; sums to ≤ 1 (the reported
      head of the full simplex).  Empty when no bandwidth is available.
    * ``membership`` — graded tube-fit ∈ [0, 1]: ``exp(−residual² / 2σ²)`` at the
      foot under the within-node thickness ``σ(z)``, EV-averaged.  Distinguishes
      *off-surface* (a real residual relative to a thin tube) from
      *on-surface-but-diffuse* — the density taper a hard ``residual`` threshold
      can't express.  ``1.0`` for a flat fit (the surface fills its subspace) and
      for a curved fit with no σ-field (no tube information).

    **Depth statistics** (where in the layer stack the probe reads):

    * ``depth_com`` / ``depth_spread`` — per-axis depth center of mass (+ std)
      of the per-layer coordinate trace, in normalized depth (0 = first
      block, 1 = last).  Mass per layer is ``share_weight_L · |coord_L|`` —
      where the probe's signal actually reads, weighted the same way the
      cross-layer aggregate coordinate is.  Aligned with ``coords`` (one
      entry per domain axis).  Empty when the reading carries no per-layer
      trace (lean per-token modes) or the monitor wasn't given the model
      depth.
    """
    fraction: float
    nearest: list[tuple[str, float]]
    coords: tuple[float, ...] = ()
    residual: float = 0.0
    fraction_per_layer: dict[int, float] = field(default_factory=dict)
    coords_per_layer: dict[int, tuple[float, ...]] = field(default_factory=dict)
    residual_per_layer: dict[int, float] = field(default_factory=dict)
    assignment: list[tuple[str, float]] = field(default_factory=list)
    membership: float = 1.0
    depth_com: tuple[float, ...] = ()
    depth_spread: tuple[float, ...] = ()
    # Per-layer reduced subspace coords in the **whitened** frame
    # (``cdist_query = c @ chol``, the same metric ``node_white`` lives in) — the
    # current hidden state's position for the probe-inspector geometry plot +
    # fading trajectory trail.  Empty by default; populated only when the session
    # sets ``Monitor.set_subspace_coords(True)`` (the ``persist_subspace_coords``
    # generate flag, on while the dashboard inspector is open), so the default
    # hot path neither computes nor serializes it.  Keyed by layer index → the
    # layer's ``(R,)`` whitened coords (R = that layer's subspace rank).
    subspace_coords_per_layer: dict[int, tuple[float, ...]] = field(
        default_factory=dict,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fraction": self.fraction,
            "nearest": [[label, dist] for label, dist in self.nearest],
            "coords": list(self.coords),
            "residual": self.residual,
            "fraction_per_layer": {
                str(k): float(v) for k, v in self.fraction_per_layer.items()
            },
            "coords_per_layer": {
                str(k): list(v) for k, v in self.coords_per_layer.items()
            },
            "residual_per_layer": {
                str(k): float(v) for k, v in self.residual_per_layer.items()
            },
            "assignment": [[label, prob] for label, prob in self.assignment],
            "membership": self.membership,
            "depth_com": list(self.depth_com),
            "depth_spread": list(self.depth_spread),
            "subspace_coords_per_layer": {
                str(k): list(v)
                for k, v in self.subspace_coords_per_layer.items()
            },
        }


@dataclass
class ScalarReading:
    """One-channel reading for the lens and SAE families.

    ``unit`` makes the scalar's meaning explicit:

    * ``"mean_token_probability"`` — lens strength, ``mean_l p_l(v)`` in
      ``[0, 1]``;
    * ``"activation_over_max"`` — SAE activation normalized by the
      Neuronpedia ``maxActApprox`` corpus max, ~``[0, 1]``;
    * ``"raw_activation"`` — SAE activation with no metadata available
      (offline / unlisted feature), comparable across tokens for one
      feature only.
    """

    value: float
    unit: str
    per_layer: dict[int, float] = field(default_factory=dict)
    depth: DepthSummary | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The family-native wire shape carried inside the measurement
        envelope's ``instruments.lens.readings`` / ``.sae.readings``.

        Deliberately NOT the ``ProbeReading`` shape: a one-channel reading
        has no fraction / nearest / residual / assignment / membership, and
        shipping those as constants is the "constants masquerading as
        measurements" pattern the 5.x break removed from the gate path.
        """
        out: dict[str, Any] = {
            "value": round(float(self.value), 6),
            "unit": str(self.unit),
            "per_layer": {
                str(layer): round(float(v), 6)
                for layer, v in self.per_layer.items()
            },
        }
        if self.depth is not None:
            out["depth"] = {
                "center": [round(float(c), 6) for c in self.depth.center],
                "spread": [round(float(s), 6) for s in self.depth.spread],
                "basis": str(self.depth.basis),
            }
        else:
            out["depth"] = None
        if self.meta:
            out["meta"] = dict(self.meta)
        return out

    def to_probe_reading(self) -> "ProbeReading":
        """Project this reading into the shared ``ProbeReading`` shape
        (geometry fields defaulted — ``fraction``/``residual`` 0,
        ``nearest``/``assignment`` empty, ``membership`` 1.0).

        The compatibility channel for consumers that key probes across
        families by one reading type; the versioned measurement envelope
        carries the native single-channel scalar reading instead.
        """
        return ProbeReading(
            fraction=0.0,
            nearest=[],
            coords=(self.value,),
            residual=0.0,
            coords_per_layer={
                layer: (v,) for layer, v in self.per_layer.items()
            },
            depth_com=tuple(self.depth.center) if self.depth else (),
            depth_spread=tuple(self.depth.spread) if self.depth else (),
        )


#: What a family's ``score_probes`` returns: the geometry family's full
#: whitened :class:`ProbeReading`, or the single-axis
#: families' :class:`ScalarReading`.  The three helpers below are the only
#: places a consumer needs to care which.
Reading = Union[ProbeReading, ScalarReading]


def reading_axis0(reading: "Reading") -> float:
    """The reading's primary scalar — a ``ScalarReading``'s ``value`` or a
    ``ProbeReading``'s coordinate axis 0.  This is the number the flat
    cross-family ``scores`` view and the loom's aggregate row carry."""
    if isinstance(reading, ScalarReading):
        return float(reading.value)
    coords = reading.coords
    return float(coords[0]) if coords else 0.0


def reading_per_layer_axis0(reading: "Reading") -> dict[int, float]:
    """The per-layer trace of :func:`reading_axis0`."""
    if isinstance(reading, ScalarReading):
        return {
            int(layer): float(value)
            for layer, value in reading.per_layer.items()
        }
    return {
        int(layer): float(coord[0] if coord else 0.0)
        for layer, coord in reading.coords_per_layer.items()
    }


def as_probe_reading(reading: "Reading") -> "ProbeReading":
    """Project any reading into the shared ``ProbeReading`` shape.

    The COMPATIBILITY boundary, and only that: ``TokenEvent.probe_readings``
    (a cross-family dict keyed by one reading type) and the OpenAI/Ollama
    ``x-drowse-probe-readings`` vendor extension, whose shape is a user
    contract.  It must not run on the native per-token path — the versioned
    measurement envelope carries each family's own reading shape, which is
    the whole point of :class:`ScalarReading`.
    """
    if isinstance(reading, ScalarReading):
        return reading.to_probe_reading()
    return reading


def scalar_gate_keys(values: Mapping[str, float]) -> dict[str, float]:
    """Flatten one-channel probe values into gate scalars.

    The :meth:`Monitor.flat_scalars` counterpart for the single-axis
    families: ONLY the real strength channel (``<name>`` and its explicit
    ``<name>[0]`` spelling).  No ``:fraction`` / ``:membership`` constants —
    a gate on a channel the family cannot produce is a composition-preflight
    error (``validate_gate``), never a silently-constant comparison.
    """
    out: dict[str, float] = {}
    for name, value in values.items():
        scalar = float(value)
        out[name] = scalar
        out[f"{name}[0]"] = scalar
    return out


# --------------------------------------------------------------------------
# Live state
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LiveState:
    """One family's live-readout intent — the return of ``set_live`` and
    the value of ``live_state``.

    The three families genuinely differ in what "live" carries (the lens
    resolves a layer list, the SAE reports its resident layer + source,
    geometry is a bare all-or-nothing switch), so the shape is a small
    frozen per-family record rather than a bare bool.  ``to_dict`` is the
    wire block every surface (the ``/live`` route, the ``/instruments``
    listing, session info) emits for the family.
    """

    enabled: bool
    family: ClassVar[InstrumentFamily]

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": bool(self.enabled)}


@dataclass(frozen=True)
class GeometryLiveState(LiveState):
    """The CAA per-token monitor-scoring switch — all-or-nothing."""

    family: ClassVar[InstrumentFamily] = "geometry"


@dataclass(frozen=True)
class LensLiveState(LiveState):
    """The workspace readout: ``layers`` is the resolved live layer list
    (``None`` when off)."""

    family: ClassVar[InstrumentFamily] = "lens"
    layers: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "layers": list(self.layers) if self.layers is not None else None,
        }


@dataclass(frozen=True)
class SaeLiveState(LiveState):
    """The feature-discovery readout at the resident hook layer."""

    family: ClassVar[InstrumentFamily] = "sae"
    layer: int | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "layer": int(self.layer) if self.layer is not None else None,
            "source": self.source,
        }


# --------------------------------------------------------------------------
# Plans, preps, bindings, live configs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ReadRequest:
    """What the session knows about a generation's read demand when it
    plans capture — the input to ``prepare`` and, through the prep, to
    ``plan``."""

    gate_keys: frozenset[str] = frozenset()
    live: bool = False
    per_token_consumers: bool = False
    final_aggregate: bool = True


@dataclass(frozen=True)
class InstrumentPrep:
    """A family's generation-boundary source snapshot — produced by
    ``Instrument.prepare`` (after the prior run is closed), consumed by
    the same family's ``plan`` and ``bind``.

    Family-opaque to the session planner: it threads the prep from
    prepare through plan into bind without reading it, so
    source-boundary work (a disk refresh, a pin decision, a spec
    snapshot) is protocol shape rather than session special-casing.
    ``request`` carries the generation's read demand forward — ``plan``
    derives solely from the prep, so a live-registry mutation landing
    between prepare and bind cannot desynchronize the plan from the
    binding.

    ``token`` is the per-preparation identity, drawn from ONE
    process-wide sequence (``next_prep_token``; per-instance sequences
    would collide across instrument instances, letting a plan from one
    instrument bind with a prep from another): the plan a prep derives
    echoes it as ``prep_token``, and ``bind`` refuses a plan/prep pair
    from different prepare() calls — same-family crossing would let the
    session's capture union retain one prep's layers while the run
    measures another's.
    """

    family: str
    request: ReadRequest = field(default_factory=ReadRequest)
    token: int = 0


@dataclass(frozen=True)
class LensPrep(InstrumentPrep):
    """The lens family's prep: the refresh/pin decision AND the
    authoritative source/spec snapshot.

    ``prepare`` refreshes the resident lens from disk (``lens.resident()``) under
    pin demand — the adoption path rewrites live probe layer lists when
    an external replacement lens landed, which is exactly why the read
    must precede planning and the spec freeze.  Because the pin itself
    only lands at ``bind``, an interleaved **unpinned** getter read (the
    un-locked ``lens.has_compatible()`` on the session-info route) can
    adopt a *newer* disk lens inside the prepare→bind window and rewrite
    the live registry again; the prep therefore carries everything
    downstream steps may consume:

    * ``lens`` — the refreshed resident lens (``None`` when demand was
      absent or the artifact is validated-missing); ``pinned`` records
      the demand itself, so per-token paths never reopen the sidecar
      even for a vanished lens.
    * ``specs`` — the probe registry snapshot with ``layers`` derived
      from ``lens`` (the prepared identity), never reread from the live
      registry by ``plan``/``bind``.
    * ``live_state`` — the live-readout runtime dict as of prepare
      (adoption rebuilds it against the new lens; the bound run must
      keep reading the one that matches its pin).
    * ``fingerprint`` — the resident sidecar identity captured in the
      same atomic snapshot, so the binding's provenance matches the
      lens the run actually measures.  A bind-time live read would
      stamp a concurrently adopted replacement's identity onto a run
      pinned to the older lens.

    The whole snapshot is taken under the instrument's ``state_lock``
    — the one lens-state boundary that also serializes the getter's
    refresh/adoption/eviction and the registry/live mutations — so it
    cannot tear mid-``prepare``.
    """

    lens: Any = None
    pinned: bool = False
    specs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    live_state: Mapping[str, Any] | None = None
    fingerprint: str | None = None


@dataclass(frozen=True)
class InstrumentPlan:
    """A family's declared capture demand for one generation.

    Demand, not mechanics: the session-side planner unions plans across
    instruments and chooses the physical retention mode
    (``set_incremental`` / ``set_tail_with_sink`` / ``set_aggregate_tail``
    / full stack).  Cross-instrument interactions (a lens aggregate
    upgrading geometry's incremental capture to a tail ring) are the
    planner's business, never a family's.

    Every field here is read by that planner.  Demand a family can state but
    nobody consumes is not demand — it is a field that drifts out of sync
    with the retention it claims to describe.
    """

    family: str
    #: Layers whose latest slice must be captured every forward.
    latest_layers: frozenset[int] = frozenset()
    #: Layers the finalize aggregate pools from the bounded tail ring.
    tail_layers: frozenset[int] = frozenset()
    #: Gate scalar keys naming this family's attached probes.
    gate_keys: frozenset[str] = frozenset()
    #: Whether an end-of-generation aggregate will read this family.
    final_aggregate: bool = False
    #: The token of the prep this plan derived from (``InstrumentPrep.
    #: token``); ``bind`` refuses a plan/prep pair whose tokens differ.
    prep_token: int | None = None


@dataclass(frozen=True)
class InstrumentBinding:
    """Immutable per-generation snapshot of what a run is measuring:
    source identity, its disk/revision fingerprint, and the attached
    probe specs frozen at bind time."""

    family: str
    source: str | None = None
    fingerprint: str | None = None
    specs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


__all__ = [
    "AGG_TAIL_DEPTH",
    "Assignment",
    "Axis",
    "DepthBasis",
    "DepthSummary",
    "Distance",
    "Fraction",
    "GateChannel",
    "GateRef",
    "GeometryLiveState",
    "InstrumentBinding",
    "InstrumentFamily",
    "InstrumentPlan",
    "LensLiveState",
    "LiveState",
    "Membership",
    "ReadRequest",
    "Reading",
    "SaeLiveState",
    "ScalarReading",
    "UNIT_ACTIVATION_OVER_MAX",
    "UNIT_MEAN_TOKEN_PROBABILITY",
    "UNIT_RAW_ACTIVATION",
    "as_probe_reading",
    "parse_gate_ref",
    "reading_axis0",
    "reading_per_layer_axis0",
    "scalar_gate_keys",
    "validate_gate_channels",
]
