from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import torch

from drowse.readout.types import Axis, Distance, Fraction, Membership, parse_gate_ref
from drowse.geometry.manifold import (
    BoxDomain, SphereDomain, KleinBottleDomain, ProjectivePlaneDomain,
    invert_parameterization,
)
from drowse.readout.monitor_attach import (
    DEFAULT_NEAREST_TOP_N,
    _FRACTION_EPSILON,
    AttachedManifoldProbe,
    _FactorStack,
    _attach_manifold_probe,
    _layer_geometry,
)
from drowse.readout.monitor_flat import _FlatEngine, stack_hidden
from drowse.generation.results import ProbeReading

if TYPE_CHECKING:
    from drowse.geometry.manifold import Manifold

_MAX_HISTORY = 8

#: Tokens per batched chunk of the FULL-mode coordinate stream — bounds the
#: ``(n_layers, T, D)`` working set on long generations.
_STREAM_CHUNK = 64


def _empty_reading() -> ProbeReading:
    return ProbeReading(fraction=0.0, nearest=[], coords=())


def _mean_coordinate_space(domain: Any, coords: torch.Tensor) -> torch.Tensor:
    if isinstance(domain, (SphereDomain, KleinBottleDomain, ProjectivePlaneDomain)) or (
        isinstance(domain, BoxDomain) and any(axis.periodic for axis in domain.axes)
    ):
        return domain.embed(coords)
    return coords


def _mean_domain_coordinates(
    domain: Any, mean: torch.Tensor, reference: torch.Tensor,
) -> torch.Tensor:
    if isinstance(domain, (KleinBottleDomain, ProjectivePlaneDomain)):
        return domain.project_mean(mean, reference)
    if isinstance(domain, SphereDomain):
        position = domain.clamp_position(domain._unembed(mean))
        return torch.where(torch.linalg.vector_norm(mean) < 1e-6, reference, position)
    if not isinstance(domain, BoxDomain) or not any(axis.periodic for axis in domain.axes):
        return mean
    parts = []
    offset = 0
    for axis, spec in enumerate(domain.axes):
        if spec.periodic:
            cosine, sine = mean[offset], mean[offset + 1]
            position = torch.remainder(torch.atan2(sine, cosine) * spec.period / (2 * torch.pi), spec.period)
            parts.append(torch.where(torch.hypot(sine, cosine) < 1e-6, reference[axis], position))
            offset += 2
        else:
            parts.append(mean[offset])
            offset += 1
    return torch.stack(parts)


@dataclass(frozen=True, slots=True)
class _ProbeGateScalarPlan:
    """Attach/generation-stable parse of the scalar channels a gate consumes."""

    probe_name: str
    coord_axes: tuple[tuple[int, str], ...] = ()
    fraction_key: str | None = None
    membership_key: str | None = None
    dist_requests: tuple[tuple[str, int], ...] = ()
    assign_requests: tuple[tuple[str, int], ...] = ()
    dist_index: torch.Tensor | None = None
    assign_index: torch.Tensor | None = None


@dataclass(eq=False)
class _GatePlan:
    """A generation's monitor gate plan: per-probe channel parses.

    Flat probes are scored together through a :class:`_FlatEngine` over just
    the gated flat probes (one batched pass, one gather, one transfer); curved
    probes through the need-masked per-probe read.  ``_gather`` caches the
    flat gather layout against the engine it was derived for.
    """

    items: tuple[_ProbeGateScalarPlan, ...]
    flat_names: frozenset[str]
    _gather: "tuple[_FlatEngine, torch.Tensor, list[tuple[int, str, str, int]], bool, bool] | None" = field(
        default=None, repr=False,
    )

    def __bool__(self) -> bool:
        return bool(self.items)


@dataclass
class _CurvedTerms:
    """Device-side pieces of one curved probe's read (see ``_curved_terms``)."""

    shared: list[int]
    w_shared: dict[int, float]
    frac_mean: torch.Tensor
    coords_mean: torch.Tensor | None = None
    mem_mean: torch.Tensor | None = None
    resid_mean: torch.Tensor | None = None
    dist_acc: torch.Tensor | None = None
    frac_terms: list[torch.Tensor] = field(default_factory=list)
    resid_terms: list[torch.Tensor] = field(default_factory=list)
    coord_terms: list[torch.Tensor] = field(default_factory=list)
    cq_terms: list[torch.Tensor] = field(default_factory=list)


def _depth_stats(
    coords_per_layer: dict[int, tuple[float, ...]],
    weights: dict[int, float],
    denom: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-axis depth center of mass (+ std) of a per-layer coordinate trace.

    Mass per (layer, axis) is ``weight_L · |coord_L[axis]|`` — where in depth
    the probe's signal actually reads, weighted the same way the cross-layer
    aggregate coordinate is (the Mahalanobis share).  Depths are
    ``layer / denom`` (``denom = n_layers − 1``), so 0 = first block, 1 =
    last.  Empty tuples when the trace is empty or ``denom`` unset; a
    zero-mass axis (activation at neutral) reports ``(0.0, 0.0)``.  The flat
    roster computes the same statistic on device
    (:meth:`_FlatEngine.depth_moments`); this host form serves curved probes.
    """
    if not coords_per_layer or denom <= 0.0:
        return (), ()
    n_dim = max(len(c) for c in coords_per_layer.values())
    coms: list[float] = []
    spreads: list[float] = []
    for axis in range(n_dim):
        total = 0.0
        first_moment = 0.0
        for layer, coords in coords_per_layer.items():
            if axis >= len(coords):
                continue
            m = weights.get(layer, 0.0) * abs(coords[axis])
            total += m
            first_moment += m * (layer / denom)
        if total <= 1e-12:
            coms.append(0.0)
            spreads.append(0.0)
            continue
        com = first_moment / total
        var = 0.0
        for layer, coords in coords_per_layer.items():
            if axis >= len(coords):
                continue
            m = weights.get(layer, 0.0) * abs(coords[axis])
            var += m * ((layer / denom) - com) ** 2
        coms.append(com)
        spreads.append(max(var / total, 0.0) ** 0.5)
    return tuple(coms), tuple(spreads)


class Monitor:
    """Reads probes as whitened subspace coordinates — flat and curved alike.

    Each probe is a :class:`~drowse.geometry.manifold.Manifold`; every read —
    live per token (gate / stream) and the end-of-generation aggregate —
    produces one full :class:`ProbeReading` (``coords`` + ``fraction`` +
    ``nearest`` + ``residual`` + assignment / membership, their per-layer
    traces and depth statistics).  A flat (affine) probe recovers ``coords``
    through its affine reduced→domain map (``residual`` 0); a curved probe
    through the :func:`invert_parameterization` foot solve.

    Execution: one stacked ``Σ⁻¹h`` apply per read covers every layer
    (:class:`_FactorStack`, the single device copy of the whitener's
    factors); the whole flat roster is then one batched pass
    (:class:`~drowse.readout.monitor_flat._FlatEngine`) and one host
    transfer.  Curved probes keep the per-probe foot solve, warm-started
    across decode tokens on the sequential live path
    (:meth:`enable_curved_warm`).

    Coordinates are **domain-frame**: each layer's reduced coordinate is
    mapped into the shared domain *before* share-averaging, because raw
    per-layer coords live in per-layer ``‖δ_L‖`` units.  At rank-1 this is
    the pole-normalized coordinate (``1.0`` at the positive node).

    The Mahalanobis whitener is mandatory and covers every probed layer;
    there is no Euclidean readout (on real LMs the Euclidean metric is
    rogue-dominated — a wrong answer, not a degraded one).
    """

    def __init__(self, probe_manifolds: dict[str, "Manifold"] | None = None,
                 whitener: Any = None,
                 n_layers: int | None = None):
        """
        probe_manifolds: probe name → :class:`Manifold` to attach at once.
        whitener: the :class:`~drowse.geometry.whitener.LayerWhitener`;
            mandatory at attach (covers every probed layer or raise).
        n_layers: the model's block count — the depth axis for the
            ``depth_com``/``depth_spread`` statistics (``None`` leaves them
            empty).
        """
        self._probes: dict[str, AttachedManifoldProbe] = {}
        self._whitener: Any = whitener
        self._depth_denom: float = (
            float(n_layers - 1) if n_layers is not None and n_layers > 1 else 0.0
        )
        # Device → the factor stack every read and attach on that device uses.
        self._stacks: dict[torch.device, _FactorStack] = {}
        # Per-device read state, rebuilt lazily after any roster change: the
        # stack covering the whole roster, and flat engines keyed by the probe
        # subset they read (``None`` = the full flat roster).
        self._read_stacks: dict[torch.device, _FactorStack] = {}
        self._engines: dict[
            tuple[torch.device, frozenset[str] | None], _FlatEngine | None
        ] = {}

        # Per-coordinate history. Rich result payloads use the pooled
        # ``ProbeReading`` directly; scalar UI projections derive from this
        # single history rather than maintaining a parallel summary model.
        self.history: dict[str, deque[tuple[float, ...]]] = {}

        # Curved-probe per-token foot-follow, keyed ``(probe, layer)``.  Only
        # the sequential live-scoring path enables it (``enable_curved_warm``);
        # one-off reads stay on the cold, reproducible solve.
        self._curved_feet: dict[tuple[str, int], torch.Tensor] = {}
        self._curved_warm: bool = False
        # Probe-inspector trail: stamp each reading's per-layer whitened query
        # coords (``subspace_coords_per_layer``) — on only while a client asks.
        self._emit_subspace_coords: bool = False

        if probe_manifolds:
            for _name, _m in probe_manifolds.items():
                self.add_probe(_name, _m)

    @property
    def probe_names(self) -> list[str]:
        """Probe names in insertion order."""
        return list(self._probes.keys())

    @property
    def manifolds(self) -> dict[str, "Manifold"]:
        """Attached probe manifolds: name -> :class:`Manifold`."""
        return {n: p.manifold for n, p in self._probes.items()}

    def probe_layers(self, names: set[str] | None = None) -> set[int]:
        """Union of fit-layer indices across attached probes (or ``names``)."""
        out: set[int] = set()
        probes = (
            (p for n, p in self._probes.items() if n in names)
            if names is not None else self._probes.values()
        )
        for probe in probes:
            out.update(probe.manifold.layers.keys())
        return out

    @property
    def whitener(self) -> Any:
        """The wired :class:`LayerWhitener` (required for reads)."""
        return self._whitener

    def set_whitener(self, whitener: Any) -> None:
        """Wire (or clear) the Mahalanobis whitener.

        Every attached probe is re-attached under the new whitener — its
        whitened factors, soft-assignment bandwidths and label scale all
        derive from the metric — atomically: if the new whitener cannot carry
        the roster (e.g. it misses a probed layer), the Monitor keeps the old
        whitener and roster and the error propagates.  Same-instance re-wiring
        is a no-op.
        """
        if whitener is self._whitener:
            return
        old = (self._whitener, self._stacks, self._probes)
        self._whitener = whitener
        self._stacks = {}
        self._invalidate()
        try:
            self._probes = {
                name: _attach_manifold_probe(
                    name, probe.manifold, top_n=probe.top_n,
                    whitener=whitener, stack_for=self._stack_for,
                )
                for name, probe in self._probes.items()
            }
        except Exception:
            self._whitener, self._stacks, self._probes = old
            self._invalidate()
            raise
        self._curved_feet.clear()

    def enable_curved_warm(self, flag: bool) -> None:
        """Enable/disable curved-probe per-token foot warm-starting.

        On only for the sequential live-scoring path (one read per decode
        step, in order); one-off reads stay on the cold, bit-reproducible
        ``invert_parameterization`` solve.
        """
        self._curved_warm = bool(flag)

    def set_subspace_coords(self, flag: bool) -> None:
        """Enable/disable the probe-inspector per-layer whitened-coords trail."""
        self._emit_subspace_coords = bool(flag)

    def reset_curved_feet(self) -> None:
        """Cold-start every curved probe's per-token foot (call per generation)."""
        self._curved_feet.clear()

    # ------------------------------------------------------------ factors

    def _stack_for(
        self, device: torch.device, layers: "list[int] | set[int]",
    ) -> _FactorStack:
        """The factor stack on ``device``, grown to cover ``layers``."""
        stack = self._stacks.get(device)
        if stack is None:
            stack = _FactorStack.build(self._whitener, sorted(set(layers)), device)
        else:
            grown = stack.grown(self._whitener, layers)
            if grown is stack:
                return stack
            stack = grown
        self._stacks[device] = stack
        self._invalidate()
        return stack

    def _invalidate(self) -> None:
        """Drop the per-device read state (rebuilt lazily on the next read)."""
        self._read_stacks = {}
        self._engines = {}

    def _read_stack(self, device: torch.device) -> _FactorStack:
        stack = self._read_stacks.get(device)
        if stack is None:
            stack = self._stack_for(device, self.probe_layers())
            self._read_stacks[device] = stack
        return stack

    def _engine(
        self, device: torch.device, names: frozenset[str] | None = None,
    ) -> _FlatEngine | None:
        """The batched engine over the flat probes in ``names`` (all if None)."""
        key = (device, names)
        if key in self._engines:
            return self._engines[key]
        stack = self._read_stack(device)
        probes = [
            p for n, p in self._probes.items()
            if p.is_affine and (names is None or n in names)
        ]
        engine = (
            _FlatEngine(probes, stack, depth_denom=self._depth_denom)
            if probes else None
        )
        self._engines[key] = engine
        return engine

    # -------------------------------------------------------------- reads

    def _score_full(
        self, hidden_per_layer: dict[int, torch.Tensor],
        *, coords_only: bool = False,
    ) -> dict[str, ProbeReading]:
        """Full readings for every attached probe at one state.

        ``coords_only`` returns the lean reading (``coords`` + ``fraction``;
        no nearest / assignment / residual / membership / traces) for the
        axis-0 live consumers; the full aggregate is re-scored at finalize.
        """
        if not hidden_per_layer or not self._probes:
            return {}
        device = next(iter(hidden_per_layer.values())).device
        stack = self._read_stack(device)
        H, present = stack_hidden(stack, hidden_per_layer)
        sih = stack.apply_stack(H)
        out: dict[str, ProbeReading] = {}
        engine = self._engine(device)
        if engine is not None:
            out.update(self._flat_readings(engine, H, present, sih, coords_only))
        for name, probe in self._probes.items():
            if name not in out:
                out[name] = self._curved_reading(
                    probe, H, sih, stack, present, 0, coords_only=coords_only,
                )
        return {name: out[name] for name in self._probes}

    def _flat_readings(
        self,
        engine: _FlatEngine,
        H: torch.Tensor,
        present: frozenset[int],
        sih: torch.Tensor,
        coords_only: bool,
    ) -> dict[str, ProbeReading]:
        """The flat roster's readings from one batched pass + one transfer."""
        P = engine.P
        nd_total = engine.nd_total
        if coords_only:
            r = engine.read(H, present, sih=sih)
            blob = torch.cat(
                [r.frac.reshape(-1), r.coords.reshape(-1)],
            ).detach().cpu().tolist()
            out: dict[str, ProbeReading] = {}
            for pi, name in enumerate(engine.names):
                if present.isdisjoint(engine.probe_layers[pi]):
                    out[name] = _empty_reading()
                    continue
                g = P + engine.nd_off[pi]
                out[name] = ProbeReading(
                    fraction=blob[pi],
                    nearest=[],
                    coords=tuple(blob[g:g + engine.nd_list[pi]]),
                    residual=0.0,
                    assignment=[],
                    membership=1.0,
                )
            return out

        subspace = self._emit_subspace_coords
        r = engine.read(
            H, present, sih=sih, dist=True, traces=True, subspace=subspace,
        )
        assert r.dist is not None and r.frac_seg is not None
        assert r.coords_seg is not None
        tw = engine.topk_width
        if tw:
            nd_sorted, ni_sorted = torch.topk(
                r.dist, k=tw, largest=False, sorted=True,
            )
            # Rank by raw whitened distance (``nearest`` stays literally
            # nearest), report in units of the probe's label spacing.
            nd_sorted = nd_sorted / engine.label_scale
        else:
            nd_sorted = r.dist.new_zeros((P, 0))
            ni_sorted = torch.zeros((P, 0), device=r.dist.device, dtype=torch.long)
        if tw and engine.band_any:
            ap_sorted, ai_sorted = torch.topk(
                engine.assignment_probs(r.dist), k=tw, largest=True, sorted=True,
            )
        else:
            ap_sorted = r.dist.new_zeros((P, tw))
            ai_sorted = torch.zeros((P, tw), device=r.dist.device, dtype=torch.long)
        parts = [
            r.frac.reshape(-1), r.coords.reshape(-1),
            nd_sorted.reshape(-1), ni_sorted.to(torch.float32).reshape(-1),
            ap_sorted.reshape(-1), ai_sorted.to(torch.float32).reshape(-1),
        ]
        with_depth = self._depth_denom > 0.0
        if with_depth:
            com, spread = engine.depth_moments(r.coords_seg, r.pres_el)
            parts += [com, spread]
        parts += [r.frac_seg.reshape(-1), r.coords_seg.reshape(-1)]
        if subspace:
            assert r.cq_seg is not None
            parts.append(r.cq_seg.reshape(-1))
        blob = torch.cat(parts).detach().cpu().tolist()

        o_frac, o_coords = 0, P
        o_nd = o_coords + nd_total
        o_ni = o_nd + P * tw
        o_ap = o_ni + P * tw
        o_ai = o_ap + P * tw
        o_com = o_ai + P * tw
        o_spread = o_com + (nd_total if with_depth else 0)
        o_fseg = o_spread + (nd_total if with_depth else 0)
        o_cseg = o_fseg + engine.S
        o_cq = o_cseg + int(r.coords_seg.shape[0])
        out = {}
        for pi, name in enumerate(engine.names):
            segs = [seg for seg in engine.probe_segs[pi] if seg[0] in present]
            if not segs:
                out[name] = _empty_reading()
                continue
            nd = engine.nd_list[pi]
            g = engine.nd_off[pi]
            labels = engine.labels[pi]
            row = pi * tw
            top_n = engine.top_n[pi]
            nearest = [
                (labels[int(round(blob[o_ni + row + j]))], blob[o_nd + row + j])
                for j in range(top_n)
            ]
            assignment = (
                [
                    (labels[int(round(blob[o_ai + row + j]))], blob[o_ap + row + j])
                    for j in range(top_n)
                ]
                if engine.band_present[pi] else []
            )
            reading = ProbeReading(
                fraction=blob[o_frac + pi],
                nearest=nearest,
                coords=tuple(blob[o_coords + g:o_coords + g + nd]),
                residual=0.0,
                fraction_per_layer={L: blob[o_fseg + s] for L, s, _c, _q, _r in segs},
                coords_per_layer={
                    L: tuple(blob[o_cseg + c:o_cseg + c + nd])
                    for L, _s, c, _q, _r in segs
                },
                residual_per_layer={L: 0.0 for L, *_ in segs},
                assignment=assignment,
                membership=1.0,
                depth_com=(
                    tuple(blob[o_com + g:o_com + g + nd]) if with_depth else ()
                ),
                depth_spread=(
                    tuple(blob[o_spread + g:o_spread + g + nd]) if with_depth else ()
                ),
            )
            if subspace:
                reading.subspace_coords_per_layer = {
                    L: tuple(blob[o_cq + q:o_cq + q + rk])
                    for L, _s, _c, q, rk in segs
                }
            out[name] = reading
        return out

    def _curved_terms(
        self,
        probe: AttachedManifoldProbe,
        H: torch.Tensor,
        sih: torch.Tensor,
        stack: _FactorStack,
        present: frozenset[int],
        col: int,
        *,
        foot: bool,
        dist: bool = False,
        membership: bool = False,
        traces: bool = False,
        subspace: bool = False,
    ) -> _CurvedTerms | None:
        """One curved probe's share-weighted per-layer read, on device.

        The need flags gate the work: ``foot`` runs the nearest-point solve
        (coords; required for ``membership`` / ``traces``), ``dist`` the
        per-candidate whitened distances, ``traces`` the per-layer fraction /
        coord / residual terms, ``subspace`` the inspector's whitened queries.
        ``H``/``sih`` are the read's ``(n, T, D)`` stacks; ``col`` picks the token.
        """
        manifold = probe.manifold
        shared = [L for L in manifold.layers if L in present]
        if not shared:
            return None
        sw = probe.share_weights
        total_w = sum(sw[L] for L in shared)
        w_shared = {L: sw[L] / total_w for L in shared}
        n_dim = manifold.domain.intrinsic_dim
        reference_layer = max(shared, key=lambda layer: w_shared[layer])
        reference_coord: torch.Tensor | None = None
        terms: _CurvedTerms | None = None
        for layer_idx in shared:
            i = stack.index[layer_idx]
            wh = probe.whitened[layer_idx]
            frac_t, cq, c = _layer_geometry(
                probe, layer_idx, H[i, col], sih[i, col],
            )
            w = w_shared[layer_idx]
            frac_t = frac_t.reshape(1)
            if terms is None:
                terms = _CurvedTerms(
                    shared=shared, w_shared=w_shared, frac_mean=w * frac_t,
                )
            else:
                terms.frac_mean = terms.frac_mean + w * frac_t
            if traces:
                terms.frac_terms.append(frac_t)
            if subspace:
                terms.cq_terms.append(cq.reshape(-1))
            if dist:
                # Neutral rides ``node_white_aug`` as the K-th candidate.
                d = torch.linalg.vector_norm(
                    wh.node_white_aug - cq.reshape(1, -1), dim=-1,
                ) * w
                terms.dist_acc = d if terms.dist_acc is None else terms.dist_acc + d
            if not foot:
                continue
            sub = manifold.layers[layer_idx]
            foot_key = (probe.name, layer_idx)
            warm = self._curved_feet.get(foot_key) if self._curved_warm else None
            pos, res = invert_parameterization(
                sub, manifold.domain, c.reshape(1, -1),
                manifold.node_coords, warm_start=warm,
            )
            if self._curved_warm:
                self._curved_feet[foot_key] = pos.detach()
            coord_t = pos.reshape(-1)[:n_dim]
            if coord_t.numel() < n_dim:
                coord_t = torch.cat(
                    [coord_t, coord_t.new_zeros(n_dim - coord_t.numel())],
                )
            if layer_idx == reference_layer:
                reference_coord = coord_t
            mean_coord = w * _mean_coordinate_space(manifold.domain, coord_t)
            terms.coords_mean = (
                mean_coord if terms.coords_mean is None
                else terms.coords_mean + mean_coord
            )
            res_flat = res.reshape(-1)[:1]
            if membership:
                # Tube-fit membership ``exp(−res²/2σ²)`` at the foot (both raw
                # reduced-space distances); no σ-field ⇒ full membership.
                if sub.has_sigma:
                    sig_foot = sub.sigma_at(
                        manifold.domain.embed(pos.reshape(1, -1)),
                    ).reshape(1)
                    mem_t = torch.exp(
                        -(res_flat ** 2)
                        / (2.0 * (sig_foot ** 2).clamp(min=_FRACTION_EPSILON))
                    )
                else:
                    mem_t = frac_t.new_ones(1)
                terms.mem_mean = (
                    w * mem_t if terms.mem_mean is None
                    else terms.mem_mean + w * mem_t
                )
            if traces:
                par_norm = torch.linalg.vector_norm(c)
                resid_t = torch.where(
                    par_norm < _FRACTION_EPSILON,
                    torch.zeros_like(res_flat),
                    res_flat / par_norm.clamp(min=_FRACTION_EPSILON),
                )
                terms.resid_terms.append(resid_t)
                terms.coord_terms.append(coord_t)
                terms.resid_mean = (
                    w * resid_t if terms.resid_mean is None
                    else terms.resid_mean + w * resid_t
                )
        assert terms is not None
        if terms.coords_mean is not None:
            terms.coords_mean = _mean_domain_coordinates(
                manifold.domain, terms.coords_mean,
                cast(torch.Tensor, reference_coord),
            )
        return terms

    def _curved_reading(
        self,
        probe: AttachedManifoldProbe,
        H: torch.Tensor,
        sih: torch.Tensor,
        stack: _FactorStack,
        present: frozenset[int],
        col: int,
        *,
        coords_only: bool = False,
    ) -> ProbeReading:
        """One curved probe's :class:`ProbeReading` (one host transfer)."""
        subspace = self._emit_subspace_coords and not coords_only
        terms = self._curved_terms(
            probe, H, sih, stack, present, col,
            foot=True, dist=not coords_only, membership=not coords_only,
            traces=not coords_only, subspace=subspace,
        )
        if terms is None:
            return _empty_reading()
        n_dim = probe.manifold.domain.intrinsic_dim
        assert terms.coords_mean is not None
        if coords_only:
            lean = torch.cat(
                [terms.frac_mean.reshape(-1), terms.coords_mean.reshape(-1)],
            ).detach().cpu().tolist()
            return ProbeReading(
                fraction=lean[0], nearest=[], coords=tuple(lean[1:1 + n_dim]),
                residual=0.0, assignment=[], membership=1.0,
            )
        assert terms.mem_mean is not None and terms.resid_mean is not None
        assert terms.dist_acc is not None
        Kc = int(terms.dist_acc.numel())
        req = int(probe.top_n)
        top_n = min(max(req if req >= 0 else Kc + req, 0), Kc)
        parts: list[torch.Tensor] = [
            terms.frac_mean, terms.resid_mean, terms.coords_mean, terms.mem_mean,
            torch.cat(terms.frac_terms), torch.cat(terms.resid_terms),
            torch.stack(terms.coord_terms, 0).reshape(-1),
        ]
        has_assign = False
        if top_n:
            # Rank by raw whitened distance; report in label spacings.
            nd_t, ni_t = torch.topk(terms.dist_acc, k=top_n, largest=False, sorted=True)
            parts += [nd_t / float(probe.label_scale), ni_t.to(torch.float32)]
            tau, lvb = probe.assign_bandwidth, probe.assign_logvol_bias
            if tau is not None and lvb is not None and tau.numel() == Kc == lvb.numel():
                # softmax(−d²/(2τ²) − R·log τ): the isotropic Gaussian-mixture
                # posterior with a uniform node prior.
                logits = -(terms.dist_acc ** 2) / (
                    2.0 * (tau ** 2).clamp(min=_FRACTION_EPSILON)
                ) + lvb
                ap_t, ai_t = torch.topk(
                    torch.softmax(logits, dim=0), k=top_n, largest=True, sorted=True,
                )
                parts += [ap_t, ai_t.to(torch.float32)]
                has_assign = True
        if subspace:
            parts += terms.cq_terms
        flat = torch.cat([p.reshape(-1) for p in parts]).detach().cpu().tolist()

        shared = terms.shared
        L = len(shared)
        o = 0
        frac_mean, residual_mean = flat[0], flat[1]
        coords_mean = tuple(flat[2:2 + n_dim])
        membership = flat[2 + n_dim]
        o = 3 + n_dim
        frac_layer = flat[o:o + L]
        o += L
        resid_layer = flat[o:o + L]
        o += L
        coord_layer = flat[o:o + L * n_dim]
        o += L * n_dim
        labels = probe.candidate_labels
        nearest: list[tuple[str, float]] = []
        assignment: list[tuple[str, float]] = []
        if top_n:
            nearest = [
                (labels[int(round(flat[o + top_n + j]))], flat[o + j])
                for j in range(top_n)
            ]
            o += 2 * top_n
            if has_assign:
                assignment = [
                    (labels[int(round(flat[o + top_n + j]))], flat[o + j])
                    for j in range(top_n)
                ]
                o += 2 * top_n
        coords_per_layer = {
            layer: tuple(coord_layer[i * n_dim:(i + 1) * n_dim])
            for i, layer in enumerate(shared)
        }
        depth_com, depth_spread = _depth_stats(
            coords_per_layer, terms.w_shared, self._depth_denom,
        )
        reading = ProbeReading(
            fraction=frac_mean,
            nearest=nearest,
            coords=coords_mean,
            residual=residual_mean,
            fraction_per_layer=dict(zip(shared, frac_layer, strict=True)),
            coords_per_layer=coords_per_layer,
            residual_per_layer=dict(zip(shared, resid_layer, strict=True)),
            assignment=assignment,
            membership=membership,
            depth_com=depth_com,
            depth_spread=depth_spread,
        )
        if subspace:
            sub_coords: dict[int, tuple[float, ...]] = {}
            for layer, cq in zip(shared, terms.cq_terms, strict=True):
                n = int(cq.numel())
                sub_coords[layer] = tuple(flat[o:o + n])
                o += n
            reading.subspace_coords_per_layer = sub_coords
        return reading

    # -------------------------------------------------------------- gates

    def _plan_probe_gate_scalars(
        self,
        probe: AttachedManifoldProbe,
        gate_keys: set[str],
    ) -> _ProbeGateScalarPlan | None:
        """Parse one probe's requested gate scalar keys once per generation."""
        name = probe.name
        coord_axes: dict[int, str] = {}
        fraction_key = membership_key = None
        dist_labels: dict[str, str] = {}
        assign_labels: dict[str, str] = {}
        for key in gate_keys:
            ref = parse_gate_ref(key)
            if ref.probe != name:
                continue
            channel = ref.channel
            if isinstance(channel, Axis):
                coord_axes[channel.index] = key
            elif isinstance(channel, Fraction):
                fraction_key = key
            elif isinstance(channel, Membership):
                membership_key = key
            elif isinstance(channel, Distance):
                dist_labels[channel.label] = key
            else:
                assign_labels[channel.label] = key
        if not (coord_axes or fraction_key or membership_key or dist_labels or assign_labels):
            return None
        label_to_idx = probe.label_to_candidate_idx
        dist_requests = tuple(
            (key, label_to_idx[label])
            for label, key in dist_labels.items() if label in label_to_idx
        )
        assign_requests = tuple(
            (key, label_to_idx[label])
            for label, key in assign_labels.items() if label in label_to_idx
        )
        plan_device = (
            next(iter(probe.whitened.values())).node_white_aug.device
            if probe.whitened else torch.device("cpu")
        )
        return _ProbeGateScalarPlan(
            probe_name=name,
            coord_axes=tuple(sorted(coord_axes.items())),
            fraction_key=fraction_key,
            membership_key=membership_key,
            dist_requests=dist_requests,
            assign_requests=assign_requests,
            dist_index=(
                torch.tensor([i for _k, i in dist_requests], device=plan_device, dtype=torch.long)
                if dist_requests else None
            ),
            assign_index=(
                torch.tensor([i for _k, i in assign_requests], device=plan_device, dtype=torch.long)
                if assign_requests else None
            ),
        )

    def plan_gate_scalars(
        self,
        gate_keys: set[str],
        *,
        probe_names: set[str] | None = None,
    ) -> _GatePlan:
        """Pre-parse exact monitor gate scalar keys for the decode hot path.

        Every key goes through :func:`parse_gate_ref` (the one key-shape
        parser); ``probe_names`` narrows the plan to a roster subset.
        """
        if not gate_keys or not self._probes:
            return _GatePlan(items=(), flat_names=frozenset())
        names = (
            set(probe_names)
            if probe_names is not None
            else {parse_gate_ref(k).probe for k in gate_keys}
        )
        items: list[_ProbeGateScalarPlan] = []
        for name, probe in self._probes.items():
            if name not in names:
                continue
            plan = self._plan_probe_gate_scalars(probe, gate_keys)
            if plan is not None:
                items.append(plan)
        flat_names = frozenset(
            it.probe_name for it in items if self._probes[it.probe_name].is_affine
        )
        return _GatePlan(items=tuple(items), flat_names=flat_names)

    def _flat_gate_layout(
        self, plan: _GatePlan, engine: _FlatEngine,
    ) -> "tuple[_FlatEngine, torch.Tensor, list[tuple[int, str, str, int]], bool, bool]":
        """Gather layout of a plan's flat keys over ``engine``'s read.

        The read is packed as ``[frac (P) | coords (Σnd) | dist/label_scale
        (P·Kmax) | assignment probs (P·Kmax)]``; each key gathers one entry.
        Slots are ``(probe index, key, kind, blob position)``.
        """
        if plan._gather is not None and plan._gather[0] is engine:
            return plan._gather
        P, Kmax = engine.P, engine.Kmax
        o_coords, o_dist = P, P + engine.nd_total
        o_prob = o_dist + P * Kmax
        index: list[int] = []
        slots: list[tuple[int, str, str, int]] = []
        need_dist = need_prob = False
        local = {name: pi for pi, name in enumerate(engine.names)}

        def _slot(pi: int, key: str, kind: str, pos: int) -> None:
            slots.append((pi, key, kind, len(index)))
            index.append(pos)

        for item in plan.items:
            pi = local.get(item.probe_name)
            if pi is None:
                continue
            if item.fraction_key is not None:
                _slot(pi, item.fraction_key, "value", pi)
            for axis, key in item.coord_axes:
                if axis < engine.nd_list[pi]:
                    _slot(pi, key, "value", o_coords + engine.nd_off[pi] + axis)
            if item.membership_key is not None:
                # A flat surface fills its subspace: membership is exactly 1.
                slots.append((pi, item.membership_key, "one", -1))
            for key, cand in item.dist_requests:
                need_dist = True
                _slot(pi, key, "negate", o_dist + pi * Kmax + cand)
            if engine.band_present[pi]:
                for key, cand in item.assign_requests:
                    need_dist = need_prob = True
                    _slot(pi, key, "value", o_prob + pi * Kmax + cand)
        gather = torch.tensor(index, device=engine.stack.device, dtype=torch.long)
        plan._gather = (engine, gather, slots, need_dist, need_prob)
        return plan._gather

    def _curved_gate_scalars(
        self,
        probe: AttachedManifoldProbe,
        H: torch.Tensor,
        sih: torch.Tensor,
        stack: _FactorStack,
        present: frozenset[int],
        plan: _ProbeGateScalarPlan,
    ) -> dict[str, float]:
        """A curved probe's planned gate scalars — only the geometry they need.

        ``:fraction`` and ``@label`` / ``~label`` gates skip the foot solve;
        coord and membership gates run it.
        """
        need_coords = bool(plan.coord_axes)
        need_membership = plan.membership_key is not None
        need_dist = bool(plan.dist_requests or plan.assign_requests)
        terms = self._curved_terms(
            probe, H, sih, stack, present, 0,
            foot=need_coords or need_membership,
            dist=need_dist, membership=need_membership,
        )
        if terms is None:
            return {}
        parts: list[torch.Tensor] = []
        keys: list[tuple[str, bool]] = []      # (key, negate)
        if plan.fraction_key is not None:
            parts.append(terms.frac_mean.reshape(1))
            keys.append((plan.fraction_key, False))
        if need_coords and terms.coords_mean is not None:
            for axis, key in plan.coord_axes:
                if axis < int(terms.coords_mean.numel()):
                    parts.append(terms.coords_mean[axis].reshape(1))
                    keys.append((key, False))
        if need_membership and terms.mem_mean is not None:
            parts.append(terms.mem_mean.reshape(1))
            keys.append((cast(str, plan.membership_key), False))
        dist_acc = terms.dist_acc
        if plan.dist_requests and dist_acc is not None and plan.dist_index is not None:
            idx = plan.dist_index.to(dist_acc.device)
            parts.append(dist_acc.index_select(0, idx) / probe.label_scale)
            keys += [(key, True) for key, _i in plan.dist_requests]
        tau, lvb = probe.assign_bandwidth, probe.assign_logvol_bias
        if (
            plan.assign_requests and dist_acc is not None
            and plan.assign_index is not None
            and tau is not None and lvb is not None
            and tau.numel() == dist_acc.numel() == lvb.numel()
        ):
            logits = -(dist_acc ** 2) / (
                2.0 * (tau ** 2).clamp(min=_FRACTION_EPSILON)
            ) + lvb
            probs = torch.softmax(logits, dim=0)
            parts.append(probs.index_select(0, plan.assign_index.to(probs.device)))
            keys += [(key, False) for key, _i in plan.assign_requests]
        if not parts:
            return {}
        flat = torch.cat([p.reshape(-1) for p in parts]).detach().cpu().tolist()
        return {
            key: (-float(v) if negate else float(v))
            for (key, negate), v in zip(keys, flat, strict=True)
        }

    def score_planned_gate_scalars(
        self,
        hidden_per_layer: dict[int, torch.Tensor],
        plan: _GatePlan,
    ) -> dict[str, float]:
        """Score a pre-parsed gate plan: only the gated probes do any work.

        :meth:`plan_gate_scalars` parses the key space once per generation;
        this runs every token — the gated flat probes as one batched pass and
        one transfer, each gated curved probe through its need-masked read.
        """
        if not hidden_per_layer or not self._probes or not plan.items:
            return {}
        device = next(iter(hidden_per_layer.values())).device
        stack = self._read_stack(device)
        H, present = stack_hidden(stack, hidden_per_layer)
        sih = stack.apply_stack(H)
        out: dict[str, float] = {}
        if plan.flat_names:
            engine = self._engine(device, plan.flat_names)
            if engine is not None:
                out.update(self._flat_gate_scalars(engine, plan, H, present, sih))
        for item in plan.items:
            if item.probe_name in plan.flat_names:
                continue
            probe = self._probes.get(item.probe_name)
            if probe is None:
                continue
            out.update(self._curved_gate_scalars(probe, H, sih, stack, present, item))
        return out

    def _flat_gate_scalars(
        self,
        engine: _FlatEngine,
        plan: _GatePlan,
        H: torch.Tensor,
        present: frozenset[int],
        sih: torch.Tensor,
    ) -> dict[str, float]:
        _eng, gather, slots, need_dist, need_prob = self._flat_gate_layout(plan, engine)
        if not slots:
            return {}
        live = [not present.isdisjoint(layers) for layers in engine.probe_layers]
        values: list[float] = []
        if gather.numel():
            r = engine.read(H, present, sih=sih, dist=need_dist)
            parts = [r.frac.reshape(-1), r.coords.reshape(-1)]
            if need_dist:
                assert r.dist is not None
                parts.append((r.dist / engine.label_scale).reshape(-1))
                parts.append(
                    engine.assignment_probs(r.dist).reshape(-1) if need_prob
                    else r.dist.new_zeros(engine.P * engine.Kmax)
                )
            values = torch.cat(parts).index_select(0, gather).detach().cpu().tolist()
        out: dict[str, float] = {}
        for pi, key, kind, pos in slots:
            if not live[pi]:
                continue
            if kind == "one":
                out[key] = 1.0
            elif kind == "negate":
                out[key] = -float(values[pos])
            else:
                out[key] = float(values[pos])
        return out

    # ---------------------------------------------------- public scoring

    def _score_tokens(
        self,
        hidden_per_layer: dict[int, torch.Tensor],
        accumulate: bool = True,
        *,
        coords_only: bool = False,
    ) -> dict[str, ProbeReading]:
        """Score every probe → ``{name: ProbeReading}``; ``accumulate`` folds
        the coords into history."""
        out = self._score_full(hidden_per_layer, coords_only=coords_only)
        if out and accumulate:
            self._apply_accumulate(out)
        return out

    def _apply_accumulate(
        self, readings: dict[str, ProbeReading],
    ) -> None:
        """Fold per-probe aggregate coordinates into monitor history."""
        for name, reading in readings.items():
            if name not in self.history:
                continue
            coords = reading.coords or (0.0,)
            self.history[name].append(tuple(coords))

    def accumulate_readings(
        self, readings: dict[str, ProbeReading],
    ) -> None:
        """Fold an already-scored aggregate into history/stats."""
        self._apply_accumulate(readings)

    def score_single_token(
        self, hidden_per_layer: dict[int, torch.Tensor],
        *, coords_only: bool = False,
    ) -> dict[str, "ProbeReading"]:
        """Per-probe full reading for a single token (no accumulate).

        The live per-token read: ``{name: ProbeReading}`` for every probe.
        ``coords_only`` returns the lean coords + fraction reading for the
        axis-0-only live consumers.
        """
        return self._score_tokens(
            hidden_per_layer, accumulate=False, coords_only=coords_only,
        )

    def measure_from_hidden(
        self, hidden_per_layer: dict[int, torch.Tensor], accumulate: bool = True,
    ) -> dict[str, "ProbeReading"]:
        """Score probes from pre-captured hidden states (no forward pass)."""
        return self._score_tokens(hidden_per_layer, accumulate=accumulate)

    @staticmethod
    def flat_scalars(
        readings: dict[str, "ProbeReading"],
    ) -> dict[str, float]:
        """Flatten per-probe readings into namespaced gate-callback scalars.

        The single gate emitter for every probe shape: ``"<probe>"`` (coord
        axis 0), ``"<probe>[<i>]"`` per axis, ``"<probe>:fraction"``,
        ``"<probe>@<label>"`` per nearest node as ``-distance`` (larger =
        closer), ``"<probe>~<label>"`` per assigned node as the soft-assignment
        probability, and ``"<probe>:membership"`` (tube fit).  The inverse is
        :func:`~drowse.readout.types.parse_gate_ref`.
        """
        out: dict[str, float] = {}
        for name, reading in readings.items():
            coords = reading.coords
            if coords:
                out[name] = coords[0]
                for i, c in enumerate(coords):
                    out[f"{name}[{i}]"] = c
            out[f"{name}:fraction"] = reading.fraction
            for label, dist in reading.nearest:
                out[f"{name}@{label}"] = -dist
            for label, prob in reading.assignment:
                out[f"{name}~{label}"] = prob
            out[f"{name}:membership"] = reading.membership
        return out

    def _per_token_coord_stream(
        self, captured: dict[int, torch.Tensor], n: int,
    ) -> dict[str, list[float]]:
        """Axis-0 domain-coordinate stream per probe over ``n`` tokens.

        The flat roster reads every token of a chunk in one batched pass and
        one transfer; curved probes keep a per-token foot solve.  Row ``i``
        reads ``captured[L][i]`` where it exists — a layer shorter than ``n``
        (EOS overshoot raggedness) simply drops out of that token's
        share-weighted mean, and a token no layer covers reads ``0.0``.
        """
        per_token: dict[str, list[float]] = {name: [0.0] * n for name in self._probes}
        if not self._probes or not captured or n <= 0:
            return per_token
        device = next(iter(captured.values())).device
        stack = self._read_stack(device)
        engine = self._engine(device)
        curved = [
            p for name, p in self._probes.items()
            if engine is None or name not in engine.names
        ]
        lengths = {L: int(h.shape[0]) for L, h in captured.items()}
        needed = self.probe_layers()
        axis0 = (
            torch.tensor(engine.nd_off, device=device, dtype=torch.long)
            if engine is not None else None
        )
        D = int(stack.X.shape[-1])
        for t0 in range(0, n, _STREAM_CHUNK):
            tc = min(_STREAM_CHUNK, n - t0)
            rows: list[torch.Tensor] = []
            ragged = False
            for L in stack.layers:
                h = captured.get(L)
                avail = max(0, min(lengths.get(L, 0) - t0, tc))
                ragged = ragged or (avail < tc and L in needed)
                if h is None or avail == 0:
                    rows.append(torch.zeros((tc, D), device=device, dtype=torch.float32))
                    continue
                block = h[t0:t0 + avail].to(torch.float32)
                if avail < tc:
                    block = torch.cat([block, block.new_zeros((tc - avail, D))])
                rows.append(block)
            H = torch.stack(rows)                                 # (n, tc, D)
            sih = stack.apply_stack(H)
            present = frozenset(L for L in stack.layers if lengths.get(L, 0) > t0)
            pres_t = None
            if ragged:
                pres_t = torch.tensor(
                    [
                        [1.0 if lengths.get(L, 0) > t0 + t else 0.0 for t in range(tc)]
                        for L in stack.layers
                    ],
                    device=device, dtype=torch.float32,
                )
            if engine is not None and axis0 is not None:
                r = engine.read(H, present, pres_t=pres_t, sih=sih)
                vals = r.coords.index_select(0, axis0).detach().cpu().tolist()
                for pi, name in enumerate(engine.names):
                    per_token[name][t0:t0 + tc] = vals[pi]
            for probe in curved:
                for t in range(tc):
                    present_t = frozenset(
                        L for L in present if lengths[L] > t0 + t
                    )
                    reading = self._curved_reading(
                        probe, H, sih, stack, present_t, t, coords_only=True,
                    )
                    per_token[probe.name][t0 + t] = (
                        reading.coords[0] if reading.coords else 0.0
                    )
        return per_token

    def score_per_token(
        self,
        captured: dict[int, torch.Tensor],
        generated_ids: list[int],
        tokenizer: Any,
        *,
        accumulate: bool = True,
        aggregate_index: int | None = None,
    ) -> tuple[dict[str, "ProbeReading"], dict[str, list[float]]]:
        """Score probes per generated token using pre-captured hidden states.

        ``captured[layer_idx]`` is a ``(n, dim)`` stack where row ``k`` is
        the hidden state that produced generated token ``k``.  Returns
        ``(aggregate_readings, per_token_coord_stream)``: the aggregate is
        the full :class:`ProbeReading` per probe pooled at the last
        non-special token (or ``aggregate_index``; negative = none); the
        stream is the per-token axis-0 domain coordinate.
        """
        n = len(generated_ids)
        if n == 0 or not captured:
            return (
                {name: _empty_reading() for name in self._probes},
                {name: [] for name in self._probes},
            )
        per_token = self._per_token_coord_stream(captured, n)
        if aggregate_index is not None and int(aggregate_index) < 0:
            return {name: _empty_reading() for name in self._probes}, per_token
        if aggregate_index is None:
            from drowse.fit.capture import last_content_index
            agg_idx = last_content_index(generated_ids, tokenizer)
        else:
            agg_idx = max(0, min(int(aggregate_index), n - 1))
        agg_hidden = {
            layer_idx: h[agg_idx] for layer_idx, h in captured.items()
            if h.shape[0] > agg_idx
        }
        return self._score_tokens(agg_hidden, accumulate=accumulate), per_token

    def score_stack(
        self,
        captured: dict[int, torch.Tensor],
        *,
        agg_index: int | None = None,
        accumulate: bool = False,
    ) -> tuple[dict[str, "ProbeReading"], dict[str, list[float]]]:
        """Score probes over a pre-captured ``[T, D]`` stack per layer.

        Like :meth:`score_per_token` without ``generated_ids`` / tokenizer —
        the aggregate pools row ``agg_index`` (default the last row).
        """
        empty_agg = {name: _empty_reading() for name in self._probes}
        if not captured:
            return empty_agg, {name: [] for name in self._probes}

        any_h = next(iter(captured.values()))
        if any_h.ndim != 2:
            raise ValueError(
                f"score_stack expects [T, D] per layer; got shape {tuple(any_h.shape)}",
            )
        n = any_h.shape[0]
        if n == 0:
            return empty_agg, {name: [] for name in self._probes}
        for layer_idx, h in captured.items():
            if h.ndim != 2 or h.shape[0] != n:
                raise ValueError(
                    f"score_stack: layer {layer_idx} has shape {tuple(h.shape)}, "
                    f"expected [{n}, D]",
                )
        agg_row = n - 1 if agg_index is None else int(agg_index)
        if not (0 <= agg_row < n):
            raise ValueError(
                f"score_stack: agg_index={agg_index} out of range for T={n}",
            )
        agg_hidden = {layer_idx: h[agg_row] for layer_idx, h in captured.items()}
        agg = self._score_tokens(agg_hidden, accumulate=accumulate)
        return agg, self._per_token_coord_stream(captured, n)

    def add_probe(self, name: str, manifold: "Manifold", *, top_n: int = DEFAULT_NEAREST_TOP_N):
        """Register a :class:`Manifold` probe — flat (any rank) or curved.

        Pre-caches the per-layer node values, share weights and whitened
        factors (:func:`_attach_manifold_probe`); the wired whitener must
        cover the fit's layers.  The Woodbury factors come from (and, for a
        new layer, grow) the Monitor's shared :class:`_FactorStack`.
        """
        is_new = name not in self._probes
        self._probes[name] = _attach_manifold_probe(
            name, manifold, top_n=top_n, whitener=self._whitener,
            stack_for=self._stack_for,
        )
        self._invalidate()
        if is_new:
            self.history[name] = deque(maxlen=_MAX_HISTORY)

    def remove_probe(self, name: str):
        """Detach ``name``.  The shared factor stack stays: the remaining
        probes (and any later attach on the same layers) keep reading the one
        device copy rather than re-uploading it."""
        self._probes.pop(name, None)
        self._invalidate()
        self._curved_feet = {
            key: foot for key, foot in self._curved_feet.items() if key[0] != name
        }
        if name in self.history:
            del self.history[name]

    def attached_probes(self) -> dict[str, "AttachedManifoldProbe"]:
        """Live attached-probe map (read-only view) — name → probe."""
        return dict(self._probes)

    def reset_history(self):
        for name in self._probes:
            self.history[name].clear()

    def score_aggregate(
        self,
        captured_per_layer: dict[int, torch.Tensor],
        *,
        agg_index: int | None = None,
    ) -> dict[str, ProbeReading]:
        """End-of-generation aggregate over pooled captures — all probes.

        ``captured_per_layer[L]`` is the per-layer ``[T, D]`` stack (or a
        single ``[D]`` state).  The pooled activation is one row — the **last
        non-special token** (``agg_index``, default the final row), the same
        single-state discipline extraction uses — read through the same full
        scorer as a live single-token read.
        """
        if not captured_per_layer or not self._probes:
            return {}
        pooled: dict[int, torch.Tensor] = {}
        for layer_idx, stack in captured_per_layer.items():
            if stack is None or stack.numel() == 0:
                continue
            if stack.ndim == 1:
                pooled[layer_idx] = stack
            else:
                t = stack.shape[0]
                row = t - 1 if agg_index is None else max(0, min(agg_index, t - 1))
                pooled[layer_idx] = stack[row]
        return self._score_full(pooled)

    def probe_geometry(
        self, name: str, *, grid_resolution: int = 24,
    ) -> dict[str, Any]:
        """Static geometry payload for one attached probe (inspector plot).

        Everything in the **whitened frame** the reads use: per-layer node
        centroids (``wh.node_white``), the neutral anchor (``wh.neutral_white``),
        a top-3 PCA rotation of the node cloud when ``rank >= 3``, and — for a
        curved fit of intrinsic dim 1/2 embedded in a higher-rank subspace — the
        manifold curve/surface sampled and whitened into the same frame.  Pairs
        with the per-token ``ProbeReading.subspace_coords_per_layer`` (the live
        point + trail), which carries the same ``cdist_query`` whitened coord.
        CPU/fp32 for the SVD + grid (MPS-fallback discipline); off the hot path.
        """
        probe = self._probes.get(name)
        if probe is None:
            raise KeyError(name)
        manifold = probe.manifold
        domain = manifold.domain
        n = int(domain.intrinsic_dim)
        layers_out: dict[str, Any] = {}
        ranks: set[int] = set()
        for layer_idx in sorted(manifold.layers):
            wh = probe.whitened.get(layer_idx)
            if wh is None:
                continue
            sub = manifold.layers[layer_idx]
            node_white = wh.node_white.detach().to(torch.float32).cpu()   # (K, R)
            R = int(node_white.shape[1])
            ranks.add(R)
            neutral_white = (
                wh.neutral_white.detach().to(torch.float32).cpu().reshape(-1)
            )
            pca_rotation: list[list[float]] | None = None
            ev_pcs: list[float] | None = None
            if R >= 3 and node_white.shape[0] >= 2:
                centered = node_white - node_white.mean(dim=0, keepdim=True)
                try:
                    _u, s, vh = torch.linalg.svd(centered, full_matrices=False)
                    pca_rotation = vh[:3].transpose(0, 1).contiguous().tolist()
                    var = s ** 2
                    tot = float(var.sum().clamp(min=1e-12))
                    ev_pcs = [float(v) / tot for v in var[:3].tolist()]
                except Exception:
                    pca_rotation, ev_pcs = None, None
            overlay: dict[str, Any] | None = None
            if (
                (not sub.is_affine)
                and n in (1, 2)
                and n < R
                and isinstance(domain, BoxDomain)
            ):
                overlay = self._sample_overlay(sub, domain, wh, n, grid_resolution)
            layers_out[str(layer_idx)] = {
                "layer": layer_idx,
                "rank": R,
                "intrinsic_dim": n,
                "is_affine": bool(sub.is_affine),
                "node_white": node_white.tolist(),
                "neutral_white": neutral_white.tolist(),
                "pca_rotation": pca_rotation,
                "explained_variance_pcs": ev_pcs,
                "mahalanobis_share": float(
                    manifold.mahalanobis_share[layer_idx],
                ),
                "overlay": overlay,
            }
        return {
            "name": name,
            "manifold": manifold.name,
            "intrinsic_dim": n,
            "is_affine": bool(probe.is_affine),
            "node_labels": list(manifold.node_labels),
            "rank_uniform": len(ranks) <= 1,
            "layers": layers_out,
        }

    @staticmethod
    def _sample_overlay(
        sub: Any, domain: BoxDomain, wh: Any, n: int, res: int,
    ) -> dict[str, Any] | None:
        """Sample a curved manifold's curve (n=1) / surface (n=2), whitened.

        Returns the sampled points already in the ``node_white`` frame
        (``r @ chol``), so the frontend overlays them on the same plot without a
        second transform.  Off the hot path — CPU/fp32 throughout.
        """
        cpu = torch.device("cpu")
        sub_c = sub.to(device=cpu, dtype=torch.float32)
        chol = wh.chol.detach().to(torch.float32).cpu()       # (R, R)
        basis = sub_c.basis                                   # (R, D)
        mean = sub_c.mean                                     # (D,)
        axes = domain.axes

        def _axis_grid(ax: Any) -> torch.Tensor:
            if ax.periodic:
                return torch.linspace(
                    0.0, float(ax.period), res + 1, dtype=torch.float32,
                )[:-1]
            return torch.linspace(
                float(ax.lo), float(ax.hi), res, dtype=torch.float32,
            )

        def _whiten(coords: torch.Tensor) -> torch.Tensor:
            world = sub_c.eval_at(domain.embed(domain.clamp_position(coords)))
            reduced = (world - mean) @ basis.transpose(0, 1)  # (.., R)
            return reduced @ chol                             # (.., R) whitened

        if n == 1:
            g = _axis_grid(axes[0]).reshape(-1, 1)            # (S, 1)
            return {"kind": "curve", "points": _whiten(g).tolist()}
        gu, gv = _axis_grid(axes[0]), _axis_grid(axes[1])
        uu, vv = torch.meshgrid(gu, gv, indexing="ij")
        grid = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)  # (nu*nv, 2)
        return {
            "kind": "surface",
            "points": _whiten(grid).tolist(),
            "grid_shape": [int(gu.numel()), int(gv.numel())],
        }
