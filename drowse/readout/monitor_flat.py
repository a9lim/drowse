"""The batched flat-roster read: every affine probe, every layer, one pass.

A flat (affine) probe's per-layer read is the same small algebra for every
(probe, layer) *segment*: project ``Σ⁻¹h`` onto the layer basis, apply the
``(R, R)`` M-projection, map to domain coords, measure the whitened distance to
each node.  :class:`_FlatEngine` lays the segments out once per roster and runs
the whole read as a fixed handful of batched kernels:

* one stacked Woodbury apply over every layer (:meth:`_FactorStack.apply_stack`),
* one ``bmm`` projecting ``Σ⁻¹h`` onto every segment's basis rows and mean,
* per *bucket* of segments sharing ``(rank, n_dim, candidates)``, batched small
  matrices — rank-1 segments (every bipolar concept) are elementwise,
* share-weighted cross-layer sums as dense ``(P, S)`` matmuls,

and the caller pulls the result to the host in one transfer.  The kernel count
depends on the number of buckets, not on probes × layers, and nothing is padded
to the roster-wide widest rank or node count.  Every read takes an optional
token axis ``T``, so the FULL-mode per-token coordinate stream is the same
kernels over a ``(n, T, D)`` stack instead of ``T`` separate reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from drowse.readout.monitor_attach import (
    _FRACTION_EPSILON,
    AttachedManifoldProbe,
    _FactorStack,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _last_row(h: torch.Tensor) -> torch.Tensor:
    """A single-token read's state: ``[D]`` as is, else the last row."""
    return h if h.ndim == 1 else h.reshape(-1, h.shape[-1])[-1]


def stack_hidden(
    stack: _FactorStack, hidden: "Mapping[int, torch.Tensor]",
) -> tuple[torch.Tensor, frozenset[int]]:
    """``(H, present)``: one token's state per stack layer as ``(n, 1, D)`` fp32.

    Layers the read did not capture are zero rows (exact for the Woodbury
    apply; their segments are masked out of every weighted sum by the reader).
    One stack + one cast, not a cast per layer.
    """
    rows: list[torch.Tensor | None] = []
    present: set[int] = set()
    dtypes: set[torch.dtype] = set()
    for layer in stack.layers:
        h = hidden.get(layer)
        if h is None:
            rows.append(None)
            continue
        h = _last_row(h)
        present.add(layer)
        dtypes.add(h.dtype)
        rows.append(h)
    dtype = dtypes.pop() if len(dtypes) == 1 else torch.float32
    zero: torch.Tensor | None = None
    filled: list[torch.Tensor] = []
    for h in rows:
        if h is None:
            if zero is None:
                zero = stack.X.new_zeros(stack.X.shape[-1], dtype=dtype)
            filled.append(zero)
        else:
            filled.append(h if h.dtype == dtype else h.to(dtype))
    H = torch.stack(filled).unsqueeze(1)
    return H.to(device=stack.device, dtype=torch.float32), frozenset(present)


@dataclass(frozen=True, eq=False)
class _Bucket:
    """Segments sharing ``(rank r, n_dim nd, candidate count kc)``."""

    n: int
    r: int
    nd: int
    kc: int
    seg: slice            # this bucket's segments, in segment order
    rows: slice           # this bucket's basis rows, in row order
    M: torch.Tensor       # (n, r, r)   M_R⁻¹
    S: torch.Tensor       # (n, nd, r)  reduced → domain map
    b: torch.Tensor       # (n, nd, 1)
    cholT: torch.Tensor   # (n, r, r)   whitened query is cholᵀ c
    nodes: torch.Tensor   # (n, kc, r)  whitened node (+ neutral) coords
    Aw: torch.Tensor      # (Pb, n)     per-probe share weights over the bucket
    cand_index: torch.Tensor  # (Pb * kc,) flat candidate slots


@dataclass(frozen=True)
class FlatRead:
    """Device tensors of one batched read (``T`` = token axis)."""

    frac: torch.Tensor                  # (P, T)
    coords: torch.Tensor                # (Σnd, T)  probe-major domain coords
    present: frozenset[int]             # layers the read saw
    sih: torch.Tensor                   # (n, T, D) Σ⁻¹h per stack layer
    frac_seg: torch.Tensor | None = None    # (S, T)
    coords_seg: torch.Tensor | None = None  # (Σseg_nd, T)
    cq_seg: torch.Tensor | None = None      # (ΣR, T) whitened queries
    dist: torch.Tensor | None = None        # (P, Kmax) raw, +inf padded
    pres_el: torch.Tensor | None = None     # (Σseg_nd, T) or None = all


class _FlatEngine:
    """Precomputed segment layout + factors for one flat roster on one device."""

    def __init__(
        self,
        probes: "list[AttachedManifoldProbe]",
        stack: _FactorStack,
        *,
        depth_denom: float,
    ) -> None:
        device = stack.device
        f32 = torch.float32
        self.stack = stack
        self.probes = probes
        self.names = tuple(p.name for p in probes)
        P = len(probes)
        self.P = P
        self.depth_denom = depth_denom

        # ---- per-probe layout --------------------------------------------
        self.nd_list = [int(p.manifold.domain.intrinsic_dim) for p in probes]
        self.nd_off: list[int] = []
        off = 0
        for nd in self.nd_list:
            self.nd_off.append(off)
            off += nd
        self.nd_total = off
        self.labels = [list(p.candidate_labels) for p in probes]
        kc_list = [len(lbl) for lbl in self.labels]
        self.top_n: list[int] = []
        for p, kc in zip(probes, kc_list, strict=True):
            req = int(p.top_n)
            tn = req if req >= 0 else kc + req
            self.top_n.append(min(max(tn, 0), kc))
        self.topk_width = max(self.top_n, default=0)
        self.cand_off: list[int] = []
        off = 0
        for kc in kc_list:
            self.cand_off.append(off)
            off += kc
        n_cand = off
        self.Kmax = max(kc_list, default=0)

        # ---- segments, bucketed by (r, nd, kc) ---------------------------
        segs: list[tuple[tuple[int, int, int], int, int]] = []   # (key, pi, layer)
        for pi, p in enumerate(probes):
            for layer, wh in p.whitened.items():
                key = (int(wh.basis.shape[0]), self.nd_list[pi], kc_list[pi])
                segs.append((key, pi, layer))
        segs.sort(key=lambda t: (t[0], t[1], stack.index[t[2]]))
        S = len(segs)
        self.S = S
        self.seg_probe = [pi for _k, pi, _l in segs]
        self.seg_layer = [layer for _k, _p, layer in segs]
        self.layers_used = frozenset(self.seg_layer)

        # Basis + mean rows per stack layer, zero-padded to the widest layer.
        n_store = len(stack.layers)
        rows_at = [0] * n_store
        means_at = [0] * n_store
        for _k, _pi, layer in segs:
            li = stack.index[layer]
            rows_at[li] += _k[0]
            means_at[li] += 1
        Rl = max(rows_at, default=0)
        width = Rl + max(means_at, default=0)
        D = int(stack.X.shape[-1])
        W = torch.zeros((n_store, max(width, 1), D), device=device, dtype=f32)
        fill_rows = [0] * n_store
        fill_means = [0] * n_store
        row_idx: list[int] = []
        mean_idx: list[int] = []
        gmean: list[torch.Tensor] = []
        mm: list[torch.Tensor] = []
        seg_store: list[int] = []
        w_seg: list[float] = []
        elem_seg: list[int] = []
        elem_slot: list[int] = []
        depth_el: list[float] = []
        # per probe: (layer, seg index, coord elem offset, cq row offset, r)
        self.probe_segs: list[list[tuple[int, int, int, int, int]]] = [
            [] for _ in range(P)
        ]
        self.probe_layers = [frozenset(p.whitened) for p in probes]
        buckets: list[_Bucket] = []
        coord_el = 0
        row_pos = 0
        s = 0
        while s < S:
            key = segs[s][0]
            e = s
            while e < S and segs[e][0] == key:
                e += 1
            r, nd, kc = key
            Ms, Ss, bs, cholTs, nodes = [], [], [], [], []
            bucket_probes: list[int] = []
            for si in range(s, e):
                _k, pi, layer = segs[si]
                probe = probes[pi]
                wh = probe.whitened[layer]
                li = stack.index[layer]
                basis = wh.basis.to(device, f32)
                mean = wh.mean.to(device, f32)
                W[li, fill_rows[li]:fill_rows[li] + r] = basis
                row_idx.extend(range(
                    li * W.shape[1] + fill_rows[li],
                    li * W.shape[1] + fill_rows[li] + r,
                ))
                fill_rows[li] += r
                W[li, Rl + fill_means[li]] = mean
                mean_idx.append(li * W.shape[1] + Rl + fill_means[li])
                fill_means[li] += 1
                sim = stack.apply(layer, mean)
                gmean.append(basis @ sim)
                mm.append((mean * sim).sum().reshape(1))
                seg_store.append(li)
                w = float(probe.share_weights[layer])
                w_seg.append(w)
                Ms.append(wh.m_r_inv.to(device, f32))
                # Every flat layer carries its affine reduced → domain map.
                assert wh.coord_S is not None and wh.coord_b is not None
                Ss.append(wh.coord_S.to(device, f32))
                bs.append(wh.coord_b.to(device, f32))
                cholTs.append(wh.chol.to(device, f32).transpose(0, 1))
                nodes.append(wh.node_white_aug.to(device, f32))
                self.probe_segs[pi].append((layer, si, coord_el, row_pos, r))
                for a in range(nd):
                    elem_seg.append(si)
                    elem_slot.append(self.nd_off[pi] + a)
                    depth_el.append(
                        layer / depth_denom if depth_denom > 0 else 0.0
                    )
                coord_el += nd
                row_pos += r
                if not bucket_probes or bucket_probes[-1] != pi:
                    bucket_probes.append(pi)
            n = e - s
            Aw = torch.zeros((len(bucket_probes), n), device=device, dtype=f32)
            local = {pi: j for j, pi in enumerate(bucket_probes)}
            for si in range(s, e):
                Aw[local[segs[si][1]], si - s] = w_seg[si]
            cand_index = torch.tensor(
                [
                    self.cand_off[pi] + k
                    for pi in bucket_probes for k in range(kc)
                ],
                device=device, dtype=torch.long,
            )
            first_row = row_pos - n * r
            buckets.append(_Bucket(
                n=n, r=r, nd=nd, kc=kc,
                seg=slice(s, e), rows=slice(first_row, row_pos),
                M=torch.stack(Ms), S=torch.stack(Ss),
                b=torch.stack(bs).unsqueeze(-1),
                cholT=torch.stack(cholTs), nodes=torch.stack(nodes),
                Aw=Aw, cand_index=cand_index,
            ))
            s = e
        self.buckets = tuple(buckets)
        self.W = W
        self.row_idx = torch.tensor(row_idx, device=device, dtype=torch.long)
        self.mean_idx = torch.tensor(mean_idx, device=device, dtype=torch.long)
        self.gmean = (
            torch.cat(gmean).unsqueeze(1) if gmean
            else torch.zeros((0, 1), device=device, dtype=f32)
        )
        self.mm = (
            torch.cat(mm).unsqueeze(1) if mm
            else torch.zeros((0, 1), device=device, dtype=f32)
        )
        self.seg_store = torch.tensor(seg_store, device=device, dtype=torch.long)
        self.cand_index = (
            torch.cat([b.cand_index for b in buckets]) if buckets
            else torch.zeros(0, device=device, dtype=torch.long)
        )
        self.n_cand = n_cand

        # ---- share-weighted cross-layer sums as dense matmuls --------------
        wt = torch.tensor(w_seg, device=device, dtype=f32)
        A_p = torch.zeros((P, S), device=device, dtype=f32)
        if S:
            A_p[torch.tensor(self.seg_probe, device=device), torch.arange(S, device=device)] = wt
        self.A_p = A_p
        n_el = len(elem_seg)
        A_nd = torch.zeros((self.nd_total, n_el), device=device, dtype=f32)
        if n_el:
            es = torch.tensor(elem_seg, device=device, dtype=torch.long)
            A_nd[torch.tensor(elem_slot, device=device), torch.arange(n_el, device=device)] = wt[es]
            self.elem_seg = es
        else:
            self.elem_seg = torch.zeros(0, device=device, dtype=torch.long)
        self.A_nd = A_nd
        self.elem_slot = torch.tensor(elem_slot, device=device, dtype=torch.long)
        self.slot_probe = torch.tensor(
            [pi for pi in range(P) for _a in range(self.nd_list[pi])],
            device=device, dtype=torch.long,
        )
        self.cand_probe = torch.tensor(
            [pi for pi in range(P) for _k in range(kc_list[pi])],
            device=device, dtype=torch.long,
        )
        self.depth_el = torch.tensor(depth_el, device=device, dtype=f32).unsqueeze(1)
        wsum = A_p.sum(dim=1, keepdim=True).clamp(min=_FRACTION_EPSILON)   # (P, 1)
        self.inv_wsum = 1.0 / wsum
        self.inv_wsum_slot = self.inv_wsum.index_select(0, self.slot_probe)

        # ---- nearest / assignment over the padded (P, Kmax) candidates -----
        # ``pad_index`` gathers each probe's candidates into a (P, Kmax) grid;
        # padding points at a trailing +inf slot (never ranked, zero probability).
        pad_index = [n_cand] * (P * self.Kmax)
        band = torch.ones((P, max(self.Kmax, 1)), device=device, dtype=f32)
        logvol = torch.zeros((P, max(self.Kmax, 1)), device=device, dtype=f32)
        self.band_present: list[bool] = []
        for pi, p in enumerate(probes):
            kc = kc_list[pi]
            for k in range(kc):
                pad_index[pi * self.Kmax + k] = self.cand_off[pi] + k
            bw, lvb = p.assign_bandwidth, p.assign_logvol_bias
            ok = bw is not None and lvb is not None and bw.numel() == kc == lvb.numel()
            if ok:
                assert bw is not None and lvb is not None
                band[pi, :kc] = bw.to(device, f32)
                logvol[pi, :kc] = lvb.to(device, f32)
            self.band_present.append(ok)
        self.pad_index = torch.tensor(pad_index, device=device, dtype=torch.long)
        self.band = band[:, :self.Kmax]
        self.logvol = logvol[:, :self.Kmax]
        self.band_any = any(self.band_present)
        self.label_scale = torch.tensor(
            [float(p.label_scale) for p in probes], device=device, dtype=f32,
        ).clamp(min=_FRACTION_EPSILON).unsqueeze(1)
        self.inf = torch.full((1,), float("inf"), device=device, dtype=f32)

    # ------------------------------------------------------------------ read

    def read(
        self,
        H: torch.Tensor,
        present: frozenset[int],
        *,
        pres_t: torch.Tensor | None = None,
        sih: torch.Tensor | None = None,
        dist: bool = False,
        traces: bool = False,
        subspace: bool = False,
    ) -> FlatRead:
        """Batched read of ``H`` ``(n, T, D)`` (the stack's layers, fp32).

        ``present`` names the layers ``H`` actually carries; a missing layer's
        segments drop out of every weighted sum.  ``pres_t`` ``(n, T)`` is the
        per-token form (a ragged ``[T, D]`` capture).  ``dist`` (single-token
        only) adds the share-weighted whitened distance to every candidate.
        """
        T = int(H.shape[1])
        if sih is None:
            sih = self.stack.apply_stack(H)                       # (n, T, D)
        GY = torch.bmm(self.W, sih.transpose(1, 2)).reshape(-1, T)
        g = GY.index_select(0, self.row_idx) - self.gmean         # (ΣR, T)
        hsm = GY.index_select(0, self.mean_idx)                   # (S, T)
        hsih = (H * sih).sum(-1)                                  # (n, T)
        x_m2 = (
            hsih.index_select(0, self.seg_store) - 2.0 * hsm + self.mm
        ).clamp(min=_FRACTION_EPSILON)

        # Segment presence (None = every segment present at every token).
        pres: torch.Tensor | None = None
        if pres_t is not None:
            pres = pres_t.index_select(0, self.seg_store)          # (S, T)
        elif not self.layers_used <= present:
            mask = torch.tensor(
                [L in present for L in self.stack.layers],
                device=H.device, dtype=torch.float32,
            )
            pres = mask.index_select(0, self.seg_store).unsqueeze(1)   # (S, 1)

        par2_parts: list[torch.Tensor] = []
        coord_parts: list[torch.Tensor] = []
        cq_parts: list[torch.Tensor] = []
        dist_parts: list[torch.Tensor] = []
        for b in self.buckets:
            gb = g[b.rows].reshape(b.n, b.r, T)
            if b.r == 1:
                cb = b.M * gb                                     # (n, 1, T)
                coords_b = b.S * cb + b.b                         # (n, nd, T)
            else:
                cb = torch.bmm(b.M, gb)
                coords_b = torch.baddbmm(b.b, b.S, cb)
            par2_parts.append((gb * cb).sum(1))                   # (n, T)
            coord_parts.append(coords_b.reshape(b.n * b.nd, T))
            if dist or subspace:
                cq = b.cholT * cb if b.r == 1 else torch.bmm(b.cholT, cb)
                if subspace:
                    cq_parts.append(cq.reshape(b.n * b.r, T))
                if dist:
                    d = torch.linalg.vector_norm(
                        b.nodes - cq.transpose(1, 2), dim=-1,
                    )                                             # (n, kc)
                    if pres is not None:
                        d = d * pres[b.seg]
                    dist_parts.append((b.Aw @ d).reshape(-1))
        par2 = torch.cat(par2_parts)                              # (S, T)
        coords_seg = torch.cat(coord_parts)                       # (Σseg_nd, T)
        frac_seg = (
            par2.clamp(min=0.0).sqrt() / x_m2.sqrt()
        ).clamp(0.0, 1.0)

        pres_el: torch.Tensor | None = None
        if pres is None:
            frac = (self.A_p @ frac_seg) * self.inv_wsum
            coords = (self.A_nd @ coords_seg) * self.inv_wsum_slot
            inv_cand = self.inv_wsum.index_select(0, self.cand_probe) if dist else None
        else:
            pres_el = pres.index_select(0, self.elem_seg)
            wsum = (self.A_p @ pres).clamp(min=_FRACTION_EPSILON)     # (P, T)
            frac = (self.A_p @ (frac_seg * pres)) / wsum
            coords = (
                (self.A_nd @ (coords_seg * pres_el))
                / wsum.index_select(0, self.slot_probe)
            )
            inv_cand = (1.0 / wsum).index_select(0, self.cand_probe) if dist else None

        dist_pad: torch.Tensor | None = None
        if dist:
            if T != 1:
                raise ValueError("candidate distances are single-token reads")
            assert inv_cand is not None
            cand = torch.zeros(
                self.n_cand, device=H.device, dtype=torch.float32,
            ).index_add_(0, self.cand_index, torch.cat(dist_parts))
            cand = cand * inv_cand.reshape(-1)
            dist_pad = torch.cat([cand, self.inf]).index_select(
                0, self.pad_index,
            ).reshape(self.P, self.Kmax)

        return FlatRead(
            frac=frac,
            coords=coords,
            present=present,
            sih=sih,
            frac_seg=frac_seg if traces else None,
            coords_seg=coords_seg if traces else None,
            cq_seg=torch.cat(cq_parts) if subspace else None,
            dist=dist_pad,
            pres_el=pres_el,
        )

    # ------------------------------------------------------- derived heads

    def assignment_probs(self, dist: torch.Tensor) -> torch.Tensor:
        """``softmax(−d²/(2τ²) − R·log τ)`` per probe over its candidates."""
        logits = -(dist ** 2) / (
            2.0 * (self.band ** 2).clamp(min=_FRACTION_EPSILON)
        ) + self.logvol
        return torch.softmax(logits, dim=1)

    def depth_moments(
        self, coords_seg: torch.Tensor, pres_el: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per (probe, axis) depth center of mass + spread, ``(Σnd,)`` each.

        Mass is ``share_weight_L · |coord_L[axis]|`` at depth ``L / (n_layers−1)``
        — :func:`~drowse.readout.monitor._depth_stats` on device; a zero-mass
        axis reads ``(0, 0)``.
        """
        mass = coords_seg[:, :1].abs()
        if pres_el is not None:
            mass = mass * pres_el[:, :1]
        mom = self.A_nd @ torch.cat([mass, mass * self.depth_el], dim=1)  # (Σnd, 2)
        total = mom[:, 0]
        ok = total > 1e-12
        safe = torch.where(ok, total, torch.ones_like(total))
        com = mom[:, 1] / safe
        dev2 = (self.depth_el[:, 0] - com.index_select(0, self.elem_slot)) ** 2
        var = (self.A_nd @ (mass[:, 0] * dev2)) / safe
        zero = torch.zeros_like(com)
        return (
            torch.where(ok, com, zero),
            torch.where(ok, var.clamp(min=0.0).sqrt(), zero),
        )
