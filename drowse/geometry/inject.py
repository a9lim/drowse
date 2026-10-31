"""Dispatch-time composition and the steering kernel.

:func:`synthesize_subspace` composes every affine steering term of one
trigger group (vectors, poles, ``~``/``|`` projections, ``!`` ablations,
affine ``%``) into one :class:`SynthesizedSubspace` per layer;
:func:`subspace_inject` is the along/onto kernel every steering term lowers
to.  Composition runs on CPU fp32 — the fragments are tiny ``(R, D)`` rows —
and moves only the finished per-layer tensors to the model device, so a push
costs no per-row host syncs.  The kernel is hot-path code: no host syncs,
fp32 arithmetic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import torch

from drowse.errors import WhitenerError
from drowse.geometry.domains import ManifoldDomain
from drowse.geometry.invert import DEFAULT_INVERSION_DAMPING, _gn_step
from drowse.geometry.rbf import eval_rbf, eval_rbf_jet
from drowse.geometry.subspace import LayerSubspace

if TYPE_CHECKING:
    from drowse.geometry.whitener import LayerWhitener

# Per-position guard for the principal-angle residual transport in
# :func:`_frame_rotation_transport`: a principal plane whose ``sin θ`` is below
# this is treated as a no-op rotation (its in-plane direction ``nᵢ`` is a 0/0),
# which is also exactly the ``p_new == p`` identity-at-rest case.
_ROTATE_EPSILON: float = 1e-6


def _ortho_basis(
    rows: torch.Tensor, *, eps: float = 1e-6,
) -> tuple[torch.Tensor, list[int]]:
    """Ordered Gram-Schmidt orthonormalization of ``rows`` ``(N, D)``.

    Processes rows in order; each is unit-normalized first (a basis of the
    *span* — magnitudes are irrelevant), and a row whose residual after
    removing its projection onto the accepted rows falls below ``eps`` is
    dropped (collinear / duplicate directions add no axis).  Unit space makes
    ``eps`` a scale-free tolerance (fp32 leaves ~1e-7 on a truly parallel
    row).  Returns ``(B, kept)``: the ``(R, D)`` orthonormal basis on the input
    device and the indices of the rows that became axes, in order.

    Ordering is load-bearing for :func:`synthesize_subspace`: push rows before
    ablation rows keep the push displacement inside the earlier axes, so the
    ablation-only complement carries a target of ~0.  Compose-time only; the
    loop runs on CPU (one transfer in, one out) so a device input pays no
    per-row host sync.
    """
    device = rows.device
    work = rows.detach().to(device="cpu", dtype=torch.float32)
    accepted: list[torch.Tensor] = []
    kept: list[int] = []
    for i, d in enumerate(work):
        dn = float(torch.linalg.vector_norm(d))
        if dn < 1e-12:                 # exact-zero input — no direction
            continue
        v = d / dn
        for b in accepted:
            v = v - (v @ b) * b
        nv = float(torch.linalg.vector_norm(v))
        if nv < eps:                   # collinear with an accepted row
            continue
        accepted.append(v / nv)
        kept.append(i)
    basis = (
        torch.stack(accepted) if accepted
        else work.new_zeros((0, work.shape[-1]))
    )
    return basis.to(device), kept


@dataclass
class SynthesizedSubspace:
    """A per-layer affine subspace synthesized from an active steering term set.

    The dispatch-time analogue of a fitted :class:`Manifold`: instead of loading
    one artifact, the session composes the whole active steering expression into
    a single per-layer subspace + ``along`` target.  This is what lets steering
    keep its superposition semantics under the two-op kernel — one manifold per
    layer holds, because dispatch builds exactly *one* derived subspace per layer
    from every active term (rather than one manifold per concept, which would
    collide at shared layers, ``OverlappingManifoldError``).

    Per layer the subspace spans the union of every active term's directions
    (push + ablation).  The ``along`` step **translates** the foot by the
    ``target_coord`` offset on each *push* axis (preserving the per-token
    spread → coherent strong steer) and **collapses** the foot toward ``0`` on
    each *ablation* axis (removing the requested fraction of that component).
    ``kappa`` stores the requested per-axis ablation coefficient: ``0`` on push
    axes, ``1`` for full mean replacement, fractional for partial ablation, and
    signed/outside ``[0, 1]`` for the grammar's extrapolating forms.  The hook
    divides it by the affine push gain before calling the shared kernel, so the
    kernel's ``along · kappa`` product is exactly the user coefficient rather
    than the affine steering gain.  ``share`` is the
    un-normalized per-layer budget weight — the push-displacement magnitude
    ``‖Δ_L‖_M`` under the required covering whitener; a pure-ablation layer
    weights by the summed
    ablation magnitude instead.  The apply path normalizes it across layers
    (mean-1).  ``target_coord`` is correspondingly a whitened-unit direction on
    the whitened path, with magnitude carried by ``share``.

    Fields are keyed by layer index; only layers carrying at least one
    non-degenerate active direction (and present in ``neutral_means``) appear.
    """

    layers: dict[int, "LayerSubspace"]        # affine: mean = neutral_L, basis = ortho span
    target_coord: dict[int, torch.Tensor]     # (R_L,) the along target (poles / 0)
    share: dict[int, float]                   # ‖Δ_L‖, un-normalized budget weight
    kappa: dict[int, torch.Tensor] = field(   # (R_L,) per-axis requested ablation coefficient
        default_factory=dict
    )

    def __post_init__(self) -> None:
        # An empty synth is the explicit no-op result when every active
        # fragment cancels or degenerates; SteeringManager skips it.
        keys = set(self.layers)
        for name, mapping in (
            ("target_coord", self.target_coord),
            ("share", self.share),
            ("kappa", self.kappa),
        ):
            if set(mapping) != keys:
                raise ValueError(
                    f"SynthesizedSubspace {name} must cover exactly its layers"
                )
        # Numeric checks accumulate on each tensor's own device and resolve
        # with one host read per device, not several per layer.  (They avoid
        # ``torch.cat``: the MPS Metal blit path can segfault on the mixed
        # view/stride shapes dispatch-time synthesis produces.)
        not_orthonormal: dict[torch.device, torch.Tensor] = {}
        not_finite: dict[torch.device, torch.Tensor] = {}
        for layer, sub in self.layers.items():
            if not sub.is_affine:
                raise ValueError("SynthesizedSubspace layers must be affine")
            if sub.mean.ndim != 1:
                raise ValueError("synthesized mean must have shape (D,)")
            if sub.basis.ndim != 2 or sub.basis.shape[0] < 1:
                raise ValueError("synthesized basis must have shape (R, D), R >= 1")
            rank = sub.rank
            dim = int(sub.basis.shape[1])
            if sub.mean.shape != (dim,):
                raise ValueError("synthesized mean width must match basis width")
            if sub.coord_offset.shape != (rank,) or sub.coord_scale.shape != (rank,):
                raise ValueError(
                    "synthesized coord_offset/coord_scale must have shape (R,)"
                )
            if (
                sub.node_coords is not None
                or sub.affine_map is not None
                or sub.sigma_rbf_weights is not None
                or sub.sigma_poly_coeffs is not None
                or sub.node_params is not None
                or sub.rbf_weights is not None
                or sub.poly_coeffs is not None
            ):
                raise ValueError(
                    "synthesized affine spans cannot carry artifact surface fields"
                )
            if self.target_coord[layer].shape != (rank,):
                raise ValueError("SynthesizedSubspace target_coord must have shape (R,)")
            if self.kappa[layer].shape != (rank,):
                raise ValueError("SynthesizedSubspace kappa must have shape (R,)")
            share = self.share[layer]
            if (
                isinstance(share, bool)
                or not isinstance(share, (int, float))  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime values can violate annotations
                or not math.isfinite(share)
                or share <= 0.0
            ):
                raise ValueError(
                    "SynthesizedSubspace shares must be finite and positive"
                )
            device = sub.basis.device
            gram = sub.basis @ sub.basis.T
            identity = torch.eye(rank, dtype=gram.dtype, device=device)
            bad_gram = ~torch.isclose(gram, identity, atol=1e-5, rtol=1e-5).all()
            bad_values = ~(
                torch.isfinite(sub.mean).all()
                & torch.isfinite(sub.basis).all()
                & torch.isfinite(self.target_coord[layer]).all().to(device)
                & torch.isfinite(self.kappa[layer]).all().to(device)
            )
            not_orthonormal[device] = (
                bad_gram if device not in not_orthonormal
                else not_orthonormal[device] | bad_gram
            )
            not_finite[device] = (
                bad_values if device not in not_finite
                else not_finite[device] | bad_values
            )
        for device, bad_gram in not_orthonormal.items():
            # One host read per device encodes both flags.
            flags = int((bad_gram.to(torch.uint8) * 2 + not_finite[device].to(torch.uint8)).item())
            if flags & 2:
                raise ValueError("synthesized basis rows must be orthonormal")
            if flags & 1:
                raise ValueError("SynthesizedSubspace tensors must be finite")



def synthesize_subspace(
    push: Sequence[
        tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], float]
    ],
    ablate: Sequence[tuple[dict[int, torch.Tensor], float]],
    neutral_means: dict[int, torch.Tensor],
    *,
    whitener: "LayerWhitener",
    eps: float = 1e-9,
) -> SynthesizedSubspace:
    """Compose an active steering term set into one affine subspace per layer.

    Each **push** term is an affine subspace fragment
    ``(basis_rows, coord_target, coeff)``:

    - ``basis_rows`` — per-layer ``(R_i, D)`` orthonormal rows (a steering
      *vector* is rank-1, ``(1, D)``; ``personas%pirate`` is rank-8, ``(8, D)``).
    - ``coord_target`` — per-layer ``(R_i,)`` target position *in that fragment's
      own basis* (a pole / node coordinate, origin-relative).
    - ``coeff`` — the signed strength (the blend fraction / α).

    Each **ablation** term is ``(directions, coeff)``: a per-layer ``(R_i, D)``
    (or ``(D,)``) direction set plus the signed fraction to remove.  Its target
    is the origin (``0``).  ``neutral_means`` supplies each layer's anchor
    (``mean``); it must cover every participating layer.

    Per layer (over the union of layers any term touches):

    - Orthonormalize the push span, project ablation directions out of it (push
      wins a shared direction), and build the reduced symmetric ablation
      operator ``A = Σ coeffᵢ uᵢuᵢᵀ`` on the remaining span.  Diagonalizing
      ``A`` gives an orthonormal ablation basis and one exact collapse
      coefficient per axis, including partial, repeated, and negative terms.
    - World push displacement ``Δ = Σ_push coeffᵢ·(coord_targetᵢ @ basis_rowsᵢ)``
      — each fragment's own ``(R_i,) @ (R_i, D) = (D,)`` world vector, scaled by
      its coeff.  ``target = B @ Δ`` is its coordinate in the merged basis.
      Because ``Δ`` lives in the push span and the ablation-only axes are its
      orthogonal complement, those axes get ``target ≈ 0``.
    - Per-axis ``kappa`` ``(R,)`` — ``0`` on the push span and the eigenvalues of
      ``A`` on the ablation-only complement.  The apply path gain-compensates
      these coefficients before the kernel evaluates
      ``p_new = q + along·(target − kappa·q)``.
    - ``share = ‖Δ‖`` (the world displacement magnitude); a pure-ablation layer
      uses a positive magnitude proxy because gain compensation makes the
      resulting collapse independent of cross-layer share normalization.

    **Whitened normalization (Mahalanobis-only).**  ``whitener`` is required and
    must cover every synthesized layer; the push is normalized in the Mahalanobis
    metric ``M_R = B Σ⁻¹ Bᵀ`` (the engine-wide read/fit metric):

    - ``share = ‖Δ‖_M`` (whitened displacement) — the cross-layer profile weights
      by *whitened* signal, matching the baked ``mahalanobis_share`` rather than
      the raw activation distance.
    - ``target = (B @ Δ) / ‖Δ‖_M`` — a **whitened-unit** direction (``‖B@target‖_M
      = 1``), so the apply path's ``eff_along_L = mean1(share)·gain`` puts the same
      *whitened* slide on every target.  The push still aims at the node centroid
      (whitening a direction toward a fixed point is metric-invariant — only the
      calibration changes), it is just measured in std-units instead of the
      raw-Euclidean scale.  Every target then receives one uniform whitened
      budget (``Σ_L eff_along_L = gain·n_layers``), distributed across layers by
      where its signal lives; ``along`` is a scale-stable strength knob.

    The metric gate is all-or-nothing: missing neutral means or partial whitener
    coverage raise :class:`~drowse.geometry.whitener.WhitenerError` before
    synthesis, so a cross-layer profile can never mix metrics and there is no
    Euclidean path.  ``share`` carries the per-layer magnitude and ``target``
    carries only the whitened-unit direction.

    The strengths live in ``target`` (per-axis), not in a single ``along`` — the
    caller picks ``along`` (the overall slide, the existing manifold-``%``
    knob) and the per-layer (mean-1) share normalization at apply time.  Pure
    tensor, fp32, no model/IO coupling — the dispatch synthesizer (which routes
    the result through ``subspace_inject`` with a ``CustomDomain(R)`` per layer)
    is the only consumer.
    """

    all_layers: set[int] = set()
    for basis_dirs, _coords, _c in push:
        all_layers |= basis_dirs.keys()
    for dirs, _coeff in ablate:
        all_layers |= dirs.keys()

    present_layers = sorted(all_layers)
    missing_means = sorted(all_layers - set(neutral_means))
    if missing_means:
        raise WhitenerError(
            f"steering synthesis requires neutral means for every layer; "
            f"missing {missing_means}"
        )
    if not whitener.covers_all(present_layers):
        raise WhitenerError(
            "steering synthesis requires a Mahalanobis whitener covering "
            f"every layer {present_layers}"
        )

    def _cpu(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to(device="cpu", dtype=torch.float32)

    layers: dict[int, LayerSubspace] = {}
    target_coord: dict[int, torch.Tensor] = {}
    share: dict[int, float] = {}
    kappa: dict[int, torch.Tensor] = {}
    finished: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for L in present_layers:
        # Every fragment is co-located on CPU: fitted manifolds follow the
        # model device while J-lens shards and SAE payloads are CPU-backed, and
        # the arithmetic below is tiny.  Only the finished layer moves to the
        # neutral mean's (runtime) device.
        neutral = neutral_means[L].to(torch.float32).reshape(-1)
        width = neutral.numel()
        zeros = torch.zeros(width)

        # Push fragments present at this layer: their basis rows (for the span)
        # and per fragment the ``(coeff, world_dir)`` pair — ``world_dir = coords
        # @ basis`` is the raw (coeff-free) neutral→node displacement; ``coeff``
        # stays separate so the whitened path can unit-normalize the direction
        # (strip the node's raw distance) while still scaling by the strength.
        push_blocks: list[torch.Tensor] = []
        push_frags: list[tuple[float, torch.Tensor]] = []
        for basis_dirs, coord_dirs, coeff in push:
            if abs(float(coeff)) < eps:
                continue
            B_i = basis_dirs.get(L)
            if B_i is None:
                continue
            B_i = _cpu(B_i)
            if B_i.ndim == 1:
                B_i = B_i.reshape(1, -1)
            if float(torch.linalg.matrix_norm(B_i)) < eps:
                continue
            c_i = coord_dirs.get(L)
            # No target coords for this layer ⇒ no displacement, but the rows
            # still join the span (a degenerate push = ablation).
            c_i = B_i.new_zeros(B_i.shape[0]) if c_i is None else _cpu(c_i).reshape(-1)
            push_blocks.append(B_i)
            push_frags.append((float(coeff), c_i @ B_i))

        ablate_rows: list[tuple[float, torch.Tensor]] = []   # (coeff, raw row)
        for dirs, coeff in ablate:
            coeff_f = float(coeff)
            if abs(coeff_f) < eps:
                continue
            d = dirs.get(L)
            if d is None:
                continue
            d = _cpu(d)
            if d.ndim == 1:
                d = d.reshape(1, -1)
            for row in d:
                if float(torch.linalg.vector_norm(row)) >= eps:
                    ablate_rows.append((coeff_f, row))

        raw_delta = zeros
        for cf, wd in push_frags:
            raw_delta = raw_delta + cf * wd
        # One batched Woodbury apply for every Mahalanobis norm this layer
        # needs: the ablation rows, each push world direction, and the raw
        # coeff-weighted push displacement (last row).
        m_norms = whitener.mahalanobis_norms(L, torch.stack(
            [row for _cf, row in ablate_rows]
            + [wd for _cf, wd in push_frags]
            + [raw_delta]
        )).tolist()
        n_ab = len(ablate_rows)
        frag_norms = m_norms[n_ab:n_ab + len(push_frags)]

        # A strictly positive proxy is sufficient for a pure-ablation share:
        # hook-time gain compensation cancels share from the ablation amount.
        ablate_share = sum(
            abs(cf) * max(m_norms[i], eps) for i, (cf, _row) in enumerate(ablate_rows)
        )
        ablate_frags = [
            (cf, row / float(torch.linalg.vector_norm(row))) for cf, row in ablate_rows
        ]

        # The active push target in world coordinates.  If all pushes cancel,
        # they must not suppress an ablation on the same axis.
        world_target = zeros
        for (cf, wd), wn in zip(push_frags, frag_norms, strict=True):
            if wn > eps:
                world_target = world_target + (cf / wn) * wd
        has_push = float(torch.linalg.vector_norm(world_target)) >= eps

        if has_push:
            B_push, _ = _ortho_basis(torch.cat(push_blocks))
        else:
            B_push = zeros.new_zeros((0, width))

        # Push wins shared directions.  The remaining ablation operator is
        # diagonalized in its own small span so coefficients survive exactly;
        # a per-axis mask cannot represent unequal non-orthogonal terms.
        projected_ablate: list[tuple[float, torch.Tensor]] = []
        for coeff_f, unit in ablate_frags:
            residual = unit
            if B_push.shape[0]:
                residual = residual - (residual @ B_push.T) @ B_push
            rn = float(torch.linalg.vector_norm(residual))
            if rn >= eps:
                projected_ablate.append((coeff_f, residual / rn))

        B_ab = zeros.new_zeros((0, width))
        ablate_eigenvalues = zeros.new_zeros((0,))
        if projected_ablate:
            B_ab_span, _ = _ortho_basis(
                torch.stack([row for _cf, row in projected_ablate]),
            )
            reduced = B_ab_span.new_zeros((B_ab_span.shape[0], B_ab_span.shape[0]))
            for coeff_f, row in projected_ablate:
                coord = B_ab_span @ row
                reduced = reduced + coeff_f * torch.outer(coord, coord)
            eigenvalues, eigenvectors = torch.linalg.eigh(reduced)
            keep = eigenvalues.abs() >= eps
            if bool(keep.any()):
                ablate_eigenvalues = eigenvalues[keep]
                B_ab = eigenvectors[:, keep].T @ B_ab_span

        if B_push.shape[0] == 0 and B_ab.shape[0] == 0:
            continue
        basis = torch.cat((B_push, B_ab), dim=0)

        if has_push:
            # The (whitened) raw displacement magnitude is the per-layer
            # profile weight; its absolute scale cancels under the apply-time
            # mean-1 normalization, leaving the across-layer shape.
            share_L = m_norms[-1]
            if share_L < eps:
                share_L = sum(abs(cf) * wn for (cf, _wd), wn in zip(push_frags, frag_norms))
            target = basis @ world_target                   # ablation axes ≈ 0
        else:
            share_L = ablate_share
            target = basis.new_zeros(basis.shape[0])
        if share_L < eps:
            continue
        axis_kappa = torch.cat((basis.new_zeros(B_push.shape[0]), ablate_eigenvalues))
        finished.append((L, neutral, basis, target, axis_kappa))
        share[L] = float(share_L)

    # One host→device transfer per device: pack every finished layer's
    # (basis, target, kappa) into a flat CPU buffer and slice views back out.
    by_device: dict[torch.device, list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    for item in finished:
        by_device.setdefault(item[1].device, []).append(item)
    for device, items in by_device.items():
        packed = torch.cat([
            torch.cat((basis.reshape(-1), target, axis_kappa))
            for _L, _neutral, basis, target, axis_kappa in items
        ]).to(device)
        offset = 0
        for L, neutral, basis, target, axis_kappa in items:
            rank, width = basis.shape
            span = rank * width
            layers[L] = LayerSubspace.affine(
                mean=neutral, basis=packed[offset:offset + span].view(rank, width),
            )
            offset += span
            target_coord[L] = packed[offset:offset + rank]
            offset += rank
            kappa[L] = packed[offset:offset + rank]
            offset += rank

    order = sorted(layers)
    return SynthesizedSubspace(
        layers={L: layers[L] for L in order},
        target_coord={L: target_coord[L] for L in order},
        share={L: share[L] for L in order},
        kappa={L: kappa[L] for L in order},
    )


def _soft_norm_cap_scale(
    norm_pre: torch.Tensor, norm_post_sq: torch.Tensor, norm_cap: float,
) -> torch.Tensor:
    """Per-row shrink ``min(1, norm_cap·‖h‖ / ‖h_new‖)``.

    The curved kernel's soft cap against off-domain RBF extrapolation.
    ``norm_pre`` is ``‖h‖`` and ``norm_post_sq`` is ``‖h_new‖²``, both
    ``(.., 1)``; the result is the ``(.., 1)`` scale (``1`` within the cap).
    Affine injection is uncapped.
    """
    return (
        norm_cap * norm_pre
        / norm_post_sq.clamp_min(0.0).sqrt().clamp(min=1e-6)
    ).clamp_(max=1.0)


def _orthonormalize_columns(
    m: torch.Tensor, *, eps: float = _ROTATE_EPSILON,
) -> torch.Tensor:
    """Modified Gram-Schmidt orthonormalization of the columns of ``m`` (.., R, n).

    Returns ``(.., R, n)`` with orthonormal columns spanning the same range.
    Pure matmul / elementwise — **no** ``torch.linalg.qr``, which is
    unimplemented on the MPS backend (no autograd-fallback either), so this runs
    natively on every device.  ``n`` is the small intrinsic dim (≤ a handful),
    so the Python loop is cheap.  A column that collapses to ~0 after
    orthogonalization (a rank-deficient frame at a fold) is zeroed rather than
    amplified by the norm division — the downstream principal-angle SVD treats
    the resulting zero overlap row as a 90° angle, i.e. no transport in it.
    """
    cols: list[torch.Tensor] = []
    for i in range(m.shape[-1]):
        v = m[..., i]                                       # (.., R)
        for u in cols:
            v = v - (u * v).sum(dim=-1, keepdim=True) * u
        norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        v = torch.where(norm > eps, v / norm.clamp(min=eps), torch.zeros_like(v))
        cols.append(v)
    return torch.stack(cols, dim=-1)                        # (.., R, n)


def _principal_pairs_1d(
    qa: torch.Tensor, qb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Principal pair of two lines: ``a = qa``, ``b = ±qb`` with ``aᵀb = c ≥ 0``.

    The 1×1 overlap ``[c0] = qaᵀqb`` has the trivial SVD ``|c0|`` with the
    sign absorbed into ``b`` — exactly what ``linalg.svd`` returns, on-device.
    """
    c0 = (qa * qb).sum(dim=-2, keepdim=True)       # (.., 1, 1) = qaᵀqb
    sign = torch.where(c0 >= 0.0, torch.ones_like(c0), -torch.ones_like(c0))
    return qa, qb * sign, c0.squeeze(-2).abs().clamp(0.0, 1.0)


def _principal_pairs_2d(
    qa: torch.Tensor, qb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Principal pairs of two planes from a closed-form 2×2 SVD, on-device.

    ``M = qaᵀqb``; the right singular vectors are the eigenvectors of the
    symmetric ``MᵀM`` at angle ``θ = ½·atan2(2r, p − q)``.  ``u₁ = Mv₁/‖Mv₁‖``
    and ``u₂ = ±u₁^⊥`` with the sign that makes ``σ₂ = u₂ᵀMv₂ ≥ 0``, so ``U``
    is exactly orthonormal and ``aᵢᵀbⱼ = σᵢδᵢⱼ`` holds to the eigenvector
    accuracy.  Replaces the MPS CPU hop of ``linalg.svd`` for the 2-D case.
    """
    m = qa.transpose(-1, -2) @ qb                  # (.., 2, 2)
    g = m.transpose(-1, -2) @ m                    # (.., 2, 2) MᵀM
    theta = 0.5 * torch.atan2(2.0 * g[..., 0, 1], g[..., 0, 0] - g[..., 1, 1])
    cs, sn = torch.cos(theta), torch.sin(theta)
    v = torch.stack((
        torch.stack((cs, -sn), dim=-1),
        torch.stack((sn, cs), dim=-1),
    ), dim=-2)                                     # (.., 2, 2) columns v₁, v₂
    mv = m @ v                                     # columns Mv₁, Mv₂
    s1 = torch.linalg.vector_norm(mv[..., 0], dim=-1)          # (..,)
    fallback = torch.zeros_like(mv[..., 0])
    fallback[..., 0] = 1.0                          # M ≈ 0: any orthonormal U
    u1 = torch.where(
        (s1 > _ROTATE_EPSILON).unsqueeze(-1),
        mv[..., 0] / s1.clamp(min=_ROTATE_EPSILON).unsqueeze(-1),
        fallback,
    )
    u1_perp = torch.stack((-u1[..., 1], u1[..., 0]), dim=-1)
    d2 = (u1_perp * mv[..., 1]).sum(dim=-1)        # signed σ₂
    u2 = u1_perp * torch.where(d2 >= 0.0, 1.0, -1.0).unsqueeze(-1)
    u = torch.stack((u1, u2), dim=-1)              # (.., 2, 2)
    c = torch.stack((s1, d2.abs()), dim=-1).clamp(0.0, 1.0)
    return qa @ u, qb @ v, c


def _principal_pairs_svd(
    qa: torch.Tensor, qb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Principal pairs from ``linalg.svd`` of the ``n×n`` overlap (``n ≥ 3``).

    ``linalg.svd`` is unimplemented on Metal, so on MPS the tiny overlap hops
    to CPU explicitly (one round trip per call).
    """
    m = qa.transpose(-1, -2) @ qb
    if m.device.type == "mps":
        u, s, vh = (t.to(m.device) for t in torch.linalg.svd(m.cpu()))
    else:
        u, s, vh = torch.linalg.svd(m)
    return qa @ u, qb @ vh.transpose(-1, -2), s.clamp(-1.0, 1.0)


def _rotate_planes(
    Hn: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
) -> torch.Tensor:
    """Apply the planar rotations ``aᵢ → bᵢ`` (``cos θᵢ = cᵢ``) to ``Hn``."""
    s = (1.0 - c * c).clamp(min=0.0).sqrt()        # (.., n) sin θᵢ
    # nᵢ = unit(bᵢ − cᵢ aᵢ): the in-(aᵢ,bᵢ)-plane direction ⊥ aᵢ that aᵢ rotates
    # toward (‖bᵢ − cᵢ aᵢ‖ = sᵢ).  Guard the sᵢ≈0 (no-rotation) planes — their
    # nᵢ is a 0/0 and their planar rotation is the identity anyway.
    perp = b - c.unsqueeze(-2) * a                 # (.., R, n)
    perp_norm = torch.linalg.vector_norm(perp, dim=-2, keepdim=True)  # (.., 1, n)
    n_dir = perp / perp_norm.clamp(min=_ROTATE_EPSILON)  # (.., R, n) unit nᵢ
    # Hn's components in each plane and the planar-rotation change
    # (aᵢ → cᵢaᵢ + sᵢnᵢ, nᵢ → −sᵢaᵢ + cᵢnᵢ):
    #   Δ = Σᵢ [(cᵢ−1)αᵢ − sᵢβᵢ] aᵢ + [sᵢαᵢ + (cᵢ−1)βᵢ] nᵢ.
    alpha = (a * Hn.unsqueeze(-1)).sum(dim=-2)     # (.., n)  aᵢᵀ Hn
    beta = (n_dir * Hn.unsqueeze(-1)).sum(dim=-2)  # (.., n)  nᵢᵀ Hn
    active = (s > _ROTATE_EPSILON).to(Hn.dtype)    # (.., n) skip sᵢ≈0 planes
    d_a = ((c - 1.0) * alpha - s * beta) * active  # (.., n)
    d_n = (s * alpha + (c - 1.0) * beta) * active  # (.., n)
    delta = (
        (a * d_a.unsqueeze(-2)).sum(dim=-1) + (n_dir * d_n.unsqueeze(-2)).sum(dim=-1)
    )                                              # (.., R)
    return Hn + delta


def _frame_rotation_transport(
    Hn: torch.Tensor,      # (.., R) off-surface residual to transport
    j_old: torch.Tensor,   # (.., R, n) tangent at the old foot p
    j_new: torch.Tensor,   # (.., R, n) tangent at the new foot p_new
) -> torch.Tensor:
    """Transport ``Hn`` from the tangent frame at the old foot to the frame at
    the new foot by the minimal (principal-angle) orthogonal rotation between
    the two tangent subspaces.

    The rotation maps ``span(j_old) → span(j_new)`` (and hence their normal
    complements) by rotating each pair of principal vectors ``aᵢ → bᵢ`` in its
    own plane, identity on the orthogonal complement of both frames.  Two
    load-bearing guarantees follow:

    - **Identity at rest.**  When the foot doesn't move (``p_new == p``, i.e.
      ``along == 0``) the frames coincide, every principal angle is 0, and the
      rotation is *exactly* the identity — so ``subspace_inject`` returns its
      input untouched regardless of foot-solve accuracy.  Anything that
      reprojects ``Hn`` onto the normal space instead corrupts an off-neutral
      activation by the residual's tangential part, which never vanishes at an
      approximate foot; the guarantee is why this is a rotation.
    - **No information loss.**  A rotation preserves ``‖Hn‖`` and discards no
      component; the residual's full content rides the frame as it turns.

    All ``O(R·n)`` per position, no ``(R, R)`` matrix materialized.  fp32.  The
    principal pairs are closed-form on-device for ``n ≤ 2``; ``n ≥ 3`` takes
    the SVD (a CPU hop on MPS).
    """
    # Orthonormal bases of the two tangent spaces (basis-choice-arbitrary; the
    # principal pairs are invariant to that choice).  Modified Gram-Schmidt,
    # not ``torch.linalg.qr`` — the latter is unimplemented on MPS.
    qa = _orthonormalize_columns(j_old)            # (.., R, n)
    qb = _orthonormalize_columns(j_new)            # (.., R, n)
    n = qa.shape[-1]
    pairs = (
        _principal_pairs_1d if n == 1
        else _principal_pairs_2d if n == 2
        else _principal_pairs_svd
    )
    return _rotate_planes(Hn, *pairs(qa, qb))


def subspace_inject(
    h: torch.Tensor,
    subspace: LayerSubspace,
    domain: ManifoldDomain,
    target_coord: torch.Tensor,
    foot_seed: torch.Tensor,
    along: "float | torch.Tensor",
    onto: "float | torch.Tensor",
    *,
    gn_steps: int = 1,
    norm_cap: float = 3.0,
    damping: float = DEFAULT_INVERSION_DAMPING,
    mean_proj: "torch.Tensor | None" = None,
    origin: "torch.Tensor | None" = None,
    kappa: "float | torch.Tensor" = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The unified two-operation manifold injection — drowse's one steering kernel.

    Decomposes ``h = mean + H_m + H_n + H_o`` against the layer's affine
    subspace and its RBF surface ``M``, then applies two near-orthogonal
    operations (``along`` is unclamped on the affine path; ``onto`` is in ``[0, 1]``):

    - **along** (``a``): **translate** the projected foot ``p*`` by the fixed
      neutral→target offset (scaled by ``a``) via
      :meth:`ManifoldDomain.translate_foot` — every token's foot shifts by the
      same displacement, which preserves the per-token in-subspace spread and
      keeps strong steer coherent (collapsing each foot onto the absolute target
      instead erases that spread and degenerates into looping).  Then transport
      the off-manifold residual ``H_n`` from the tangent frame at the old foot to
      the frame at the new foot by the minimal orthogonal (principal-angle)
      rotation between the two tangent subspaces
      (:func:`_frame_rotation_transport`) — *exactly* the identity when the foot
      didn't move, so the curved path is identity at ``along == 0`` regardless of
      foot-solve accuracy; norm-preserving and lossless.  Tangential /
      directional; leaves ``H_o`` untouched.  By moving *on the surface* it never
      cuts through the off-manifold low-density region.

    ``origin`` is the neutral foot — the translate reference; it is coord 0 for
    an affine fit, so the affine path ignores it.  ``kappa`` is the per-axis
    collapse blend on the affine path: scalar ``0.0`` ⇒ pure translate, or a
    ``(R,)`` mask (from :func:`synthesize_subspace`) whose ``κ=1`` ablation axes
    collapse toward 0 while ``κ=0`` push axes translate.  (Curved manifolds are
    push-only, so they take the scalar-0 / pure-translate :meth:`translate_foot`.)
    - **onto** (``o``): collapse the off-surface residual ``H_n`` toward the
      surface within the subspace.  With no σ-field (zero-thickness wire) this
      scales ``H_n`` by ``(1 − o)`` — at ``o = 1`` the activation lands on the
      mean surface.  With a fuzzy-manifold σ-field (:meth:`~drowse.geometry.subspace.LayerSubspace.sigma_at`)
      it instead shrinks ``‖H_n‖`` toward the local within-node thickness
      ``σ(z)`` (the *tube*), so ``o = 1`` lands one-σ off the wire — a sample-like
      point on the surface's typical set — direction preserved and never
      expanding a residual already inside the tube.  Vacuous when the surface
      fills its subspace (``H_n ≈ 0``, every affine term).

    The off-*subspace* residual ``H_o`` is **always kept verbatim**.  That is the
    composition invariant: ``H_o`` is the orthogonal complement of *this*
    subspace, i.e. every composing neighbor's span, so touching it would couple
    otherwise-orthogonal terms.  ``~``/``|`` semantics live inside the subspace
    instead, as push/ablation axes of the merged affine basis.

    All subspace arithmetic runs in **reduced (R-dim) coordinates**; because
    ``basis`` is orthonormal, ``‖H_n_reduced‖ = ‖H_n‖`` exactly, so the
    transport's norm-preservation holds and the cost is ``O(R)`` not ``O(D)``.
    fp32 throughout (fp16 sum-of-squares overflows at large ``D``).

    ``foot_seed`` ``(.., n)`` is the warm start for the nearest-point foot — the
    previous token's foot on the hot path, refined by ``gn_steps`` shared
    :func:`_gn_step` iterations (one per token in steady state).  Returns
    ``(h_new, foot)`` where ``foot`` ``(.., n)`` is *this* token's refined
    pre-slide foot, to seed the next token.

    Order is fixed **along → onto**: the transport (along) must run before
    ``onto`` scales the transported residual.  No global norm preservation —
    ``onto`` is *meant* to shrink ``‖h − mean‖``, and the apply-time (mean-1)
    share normalization controls the per-α magnitude; a soft cap
    ``‖h_new‖ ≤ norm_cap·‖h‖`` guards only against off-domain RBF extrapolation
    blowup.
    """
    h_f32 = h.to(torch.float32)
    mean = subspace.mean.to(torch.float32)            # (D,)
    basis = subspace.basis.to(torch.float32)          # (R, D)
    target = target_coord.to(device=h_f32.device, dtype=torch.float32)

    if subspace.is_affine:
        # --- flat (folded-vector) shortcut --------------------------------
        # The surface fills the subspace, so the injected target is already in
        # the layer's orthonormal reduced frame: the foot is ``q`` exactly,
        # ``H_n ≡ 0`` and ``onto`` is vacuous (ignored).  Ordinary fits have an
        # identity authoring→reduced map; rectangular transfers bake their
        # explicit map into the resolved per-layer target before this kernel.
        # No GN solve, no RBF eval, no tangent Gram-solve — the cost the common
        # steering case can't pay.
        #
        # ``q = (h − mean)·basisᵀ = h·basisᵀ − mean·basisᵀ`` is computed
        # **without** the full-width ``centered = h − mean`` temporary: the
        # ``(R,)`` ``mean_proj = mean·basisᵀ`` is a tiny reduced-space constant
        # (one dot product at R=1).  Hot callers (the single-affine fast hook)
        # precompute it once at ``recompose`` and thread it in; everyone else
        # gets the inline ``mean @ basis.T``.  ALONG translates ``q`` by the
        # fixed offset ``along·target``; the off-subspace residual ``H_o`` is
        # kept verbatim (``h_new = h + Δ·basis`` is a fresh tensor, never an
        # in-place mutation of ``h``).
        mp = mean_proj if mean_proj is not None else (mean @ basis.T)  # (R,)
        q = h_f32 @ basis.T - mp                        # (.., R)
        # Per-axis κ-blend of the foot toward ``target`` (``origin`` is span-coord
        # 0 in the ν-anchored affine frame; an affine fit is always on the flat
        # ``CustomDomain`` — identity clamp — so this analytic is exact):
        #   push axis   (κ=0): ``q + along·target``        — translate by the
        #     fixed offset, preserving the per-token in-subspace spread (coherent);
        #   ablate axis (κ=1): ``q + along·(0 − q)``       — collapse the
        #     component toward 0 (remove the ablated direction).
        # ``kappa`` is a scalar (``0.0`` ⇒ pure translate) or a per-axis ``(R,)``
        # mask from ``synthesize_subspace`` (0 push / 1 ablate).
        p_new = q + along * (target - kappa * q)        # (.., n==R)
        h_new = h_f32 + ((p_new - q) @ basis)           # keep H_o verbatim
        # No soft norm cap on the affine branch.  The cap only ever guarded
        # off-domain RBF *extrapolation* blowup; a flat affine fit has no RBF
        # surface (``clamp_position`` is identity on the flat ``CustomDomain``),
        # so ``p_new`` is always in-frame and the displacement ``(p_new − q)@basis``
        # is a bounded steering offset added to a large-norm residual stream — it
        # cannot plausibly push ``‖h_new‖`` past ``3·‖h‖``.  This is the same
        # reasoning (and now the same behavior) as the runtime affine path
        # (``drowse.steering.hooks.AffineOp``), which never caps; the
        # mixed push+ablate (κ≠0) affine term only *shrinks* the ablated axes, so
        # it can't grow the norm past the pure-push case either.  Dropping it
        # removes the two per-fire full-width norm reductions on the affine kernel
        # path (curved fits keep the cap below).
        # Return fp32; the caller's ``hidden.copy_(h_new)`` downcasts to the
        # model dtype on the copy, so an explicit ``.to(h.dtype)`` here would
        # only allocate a redundant full-width model-dtype temporary per fire.
        return h_new, q

    # Reduced coords of ``h`` without a full-width ``h − mean`` temporary: the
    # basis is orthonormal, so ``q = h·Bᵀ − mean·Bᵀ`` and the whole update is
    # one reduced displacement ``δ`` added back as ``h + δ·B``.
    mp = mean_proj if mean_proj is not None else (mean @ basis.T)  # (R,)
    q = h_f32 @ basis.T - mp                           # (.., R)

    np_, rw, pc = subspace.rbf_params()
    np_ = np_.to(torch.float32)
    rw = rw.to(torch.float32)
    pc = pc.to(torch.float32)
    offset = subspace.coord_offset.to(torch.float32)  # (m,)
    scale = subspace.coord_scale.to(torch.float32)    # (m,)

    # --- foot p* : nearest point on M to q, warm-started from foot_seed ---
    p = foot_seed
    for _ in range(int(gn_steps)):
        p = _gn_step(p, q, np_, rw, pc, offset, scale, domain, damping)

    along_zero = not isinstance(along, torch.Tensor) and float(along) == 0.0
    onto_zero = not isinstance(onto, torch.Tensor) and float(onto) == 0.0

    emb_old = (domain.embed(p) - offset) / scale       # (.., m)

    # --- ALONG: translate the foot along the surface ---
    # Translate p* by the parallel-transported neutral→target offset (scaled by
    # ``along``), preserving the per-token foot spread rather than collapsing
    # every foot onto the absolute target.  ``origin`` is this layer's neutral
    # foot; falls back to coord 0.  When along == 0 the transport is exactly
    # the identity, so the tangent frames (and the n>=2 SVD) are skipped.
    if along_zero:
        foot_red = eval_rbf(np_, rw, pc, emb_old)      # (.., R)
        emb_new = emb_old
        foot_new_red = foot_red
        Hn_trans = q - foot_red                        # off-manifold-in-subspace
    else:
        foot_red, jac_old = eval_rbf_jet(np_, rw, pc, emb_old)
        org = origin if origin is not None else torch.zeros_like(target)
        p_new = domain.translate_foot(p, org, target, along)  # (.., n)
        emb_new = (domain.embed(p_new) - offset) / scale       # (.., m)
        foot_new_red, jac_new = eval_rbf_jet(np_, rw, pc, emb_new)
        # World-reduced tangent columns of M at the old/new feet: dRBF/dcoord
        # chained through the embedding Jacobian.
        j_old = (jac_old / scale) @ domain.tangent_jacobian(p)      # (.., R, n)
        j_new = (jac_new / scale) @ domain.tangent_jacobian(p_new)  # (.., R, n)
        Hn_trans = _frame_rotation_transport(q - foot_red, j_old, j_new)

    # --- ONTO: collapse the transported off-manifold residual toward the tube ---
    # Zero-thickness wire (no sigma field — SAE-curved fits): scale ``H_n`` by
    # ``(1 − o)``, landing on the mean surface at ``o = 1``.  Fuzzy (sigma field
    # present): shrink ``‖H_n‖`` toward the local within-node thickness
    # ``σ(z)`` instead, so ``o = 1`` lands one-σ off the wire — the surface's
    # typical set — direction preserved, never expanding a residual already
    # inside the tube.  ``σ = 0`` reproduces the ``(1 − o)`` collapse exactly.
    if onto_zero:
        Hn_final = Hn_trans
    else:
        sigma = subspace.sigma_at(emb_new, normalized=True)  # (..,) σ(z)
        hn_norm = torch.linalg.vector_norm(Hn_trans, dim=-1)  # (..,)
        shrink = torch.clamp(
            1.0 - sigma / hn_norm.clamp(min=1e-6), min=0.0,
        )                                                     # (..,)
        onto_scale = 1.0 - onto * shrink                     # (..,) in [1−o, 1]
        Hn_final = onto_scale.unsqueeze(-1) * Hn_trans        # (.., R)

    # The off-subspace residual ``H_o`` rides ``h`` untouched: it is every
    # composing neighbor's span, so only the in-subspace displacement is added.
    delta = foot_new_red + Hn_final - q                       # (.., R)
    # Soft cap ``‖h_new‖ ≤ norm_cap·‖h‖`` against off-domain RBF extrapolation,
    # with ``‖h + δ·B‖² = ‖h‖² + 2 (q + mp)·δ + ‖δ‖²`` evaluated in reduced
    # space (orthonormal ``B``): one full-width reduction, one full-width write.
    norm_pre = torch.linalg.vector_norm(h_f32, dim=-1, keepdim=True)   # (.., 1)
    norm_post_sq = (
        norm_pre * norm_pre
        + 2.0 * ((q + mp) * delta).sum(dim=-1, keepdim=True)
        + (delta * delta).sum(dim=-1, keepdim=True)
    )
    cap = _soft_norm_cap_scale(norm_pre, norm_post_sq, norm_cap)  # (.., 1)
    h_new = (delta @ basis).add_(h_f32).mul_(cap)        # (.., D)
    # Return fp32 (see the affine branch): the caller's ``hidden.copy_`` does
    # the model-dtype downcast, so no per-fire ``.to(h.dtype)`` temporary.
    return h_new, p


__all__ = [
    "SynthesizedSubspace",
    "subspace_inject",
    "synthesize_subspace",
]
