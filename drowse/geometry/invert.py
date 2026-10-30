"""Nearest-point projection onto a curved manifold surface (the foot solve).

The paper's ``s⁻¹`` map: minimize ``‖s(p) − q‖`` over authoring coords ``p``
by damped Gauss-Newton (Levenberg-Marquardt) with the analytic RBF Jacobian
chained through the domain's embedding Jacobian.  :func:`_gn_step` is the one
LM body, shared by :func:`invert_parameterization` (the read side, the fit
origin, transfer) and :func:`~drowse.geometry.inject.subspace_inject` (one
warm-started step per token on the steering hot path).
"""
from __future__ import annotations

import torch

from drowse.geometry.domains import ManifoldDomain
from drowse.geometry.rbf import eval_rbf, eval_rbf_jet
from drowse.geometry.subspace import LayerSubspace

# The solve is warm-started from the nearest fit node(s) and Marquardt-damped,
# so a fixed dozen iterations converges in authoring-coord space independent
# of intrinsic dimension.
DEFAULT_INVERSION_MAX_ITER = 12
DEFAULT_INVERSION_RESTARTS = 3
# Warm-started inversion (curved-probe per-token foot-follow): a one-decode-step
# activation drift leaves the carried foot near this token's nearest point, so
# a handful of LM steps (plus one nearest-node restart as a basin-jump safety
# net) converges where the cold 12-iter / 3-restart search would.
DEFAULT_INVERSION_WARM_ITER = 4
DEFAULT_INVERSION_DAMPING = 1e-3
# Absolute floor added to the LM normal-equation diagonal so a locally
# rank-deficient Jacobian (a fold/pinch, or a flat authoring direction)
# still yields a solvable, well-conditioned system.
_INVERSION_DIAG_FLOOR = 1e-9


def _gn_step(
    p: torch.Tensor,
    q: torch.Tensor,
    node_params: torch.Tensor,
    rbf_weights: torch.Tensor,
    poly_coeffs: torch.Tensor,
    coord_offset: torch.Tensor,
    coord_scale: torch.Tensor,
    domain: ManifoldDomain,
    damping: float,
) -> torch.Tensor:
    """One damped Gauss-Newton (Levenberg-Marquardt) step of ``argmin_p ‖s(p) − q‖``.

    ``p`` ``(.., n)`` are authoring coords, ``q`` ``(.., R)`` the reduced-space
    targets (broadcastable against ``p``'s leading dims — the inversion fans
    ``p`` over ``S`` restarts against a ``(.., 1, R)`` query).  The RBF tensors
    are already on ``p``'s device/dtype.  Returns the retracted post-step
    position.  The ``n ≤ 2`` normal equations are solved in closed form: a
    batched ``linalg.solve`` checks for singularity with a host sync, which on
    the steering hot path would drain the device queue mid-forward.
    """
    emb = (domain.embed(p) - coord_offset) / coord_scale          # (.., m)
    value, jac = eval_rbf_jet(node_params, rbf_weights, poly_coeffs, emb)
    resid = value - q                                             # (.., R)
    j_auth = (jac / coord_scale) @ domain.tangent_jacobian(p)     # (.., R, n)
    jt = j_auth.transpose(-1, -2)                                 # (.., n, R)
    jtj = jt @ j_auth                                             # (.., n, n)
    jtr = jt @ resid.unsqueeze(-1)                                # (.., n, 1)
    n_dim = jtj.shape[-1]
    diag = torch.diagonal(jtj, dim1=-2, dim2=-1)                  # (.., n)
    reg = damping * diag.clamp(min=_INVERSION_DIAG_FLOOR) + _INVERSION_DIAG_FLOOR
    if n_dim == 1:
        step = jtr[..., 0] / (jtj[..., 0] + reg)                  # (.., 1)
    elif n_dim == 2:
        # Damped JᵀJ is SPD, so Cramer's rule is safe — and, unlike a batched
        # ``linalg.solve``, it never syncs the device mid-forward.
        a00 = jtj[..., 0, 0] + reg[..., 0]
        a11 = jtj[..., 1, 1] + reg[..., 1]
        a01 = jtj[..., 0, 1]
        r0, r1 = jtr[..., 0, 0], jtr[..., 1, 0]
        det = a00 * a11 - a01 * jtj[..., 1, 0]
        step = torch.stack((a11 * r0 - a01 * r1, a00 * r1 - jtj[..., 1, 0] * r0), dim=-1)
        step = step / det.unsqueeze(-1)                           # (.., 2)
    else:
        A = jtj + torch.diag_embed(reg)                           # (.., n, n)
        bsz = A.shape[:-2]
        # The flattened batch also dodges ``linalg.solve``'s size-1-leading-
        # batch out-resize warning on MPS.
        step = torch.linalg.solve(
            A.reshape(-1, n_dim, n_dim).contiguous(),
            jtr.reshape(-1, n_dim, 1).contiguous(),
        ).reshape(*bsz, n_dim)                                    # (.., n)
    return domain.retract_position(p, -step)


def invert_parameterization(
    subspace: LayerSubspace,
    domain: ManifoldDomain,
    query: torch.Tensor,
    node_coords: torch.Tensor,
    *,
    max_iter: int = DEFAULT_INVERSION_MAX_ITER,
    n_restarts: int = DEFAULT_INVERSION_RESTARTS,
    warm_start: torch.Tensor | None = None,
    warm_iter: int = DEFAULT_INVERSION_WARM_ITER,
    damping: float = DEFAULT_INVERSION_DAMPING,
    node_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-point projection of ``query`` ``(.., R)`` onto the fitted surface.

    Returns ``(positions (.., n), dist (..,))``: the authoring coords whose
    interpolant value is nearest each query in reduced-coordinate space, and
    that distance.  ``node_coords`` ``(K, n)`` are the manifold's authoring
    node coordinates (the warm-start seeds).

    Cold path: seed each query at its ``n_restarts`` nearest fit nodes (in
    reduced space) and take ``max_iter`` LM steps; the best final residual per
    query wins, guarding against a fold/periodic local minimum.  Warm path
    (``warm_start``, the curved-probe foot-follower): ``warm_iter`` steps over
    two restarts — the carried previous-token foot and the single nearest fit
    node.  ``node_values`` ``(K, R)`` optionally supplies the surface values at
    the fit nodes (the seed metric), which per-token callers can cache.
    """
    n = domain.intrinsic_dim
    R = query.shape[-1]
    lead = query.shape[:-1]
    device = query.device
    dtype = query.dtype if query.dtype.is_floating_point else torch.float32
    flat = query.reshape(-1, R).to(dtype)  # (N, R)
    N = flat.shape[0]

    # The caller may hand a subspace still resident on its load device.
    np_, rw, pc = subspace.rbf_params()
    np_ = np_.to(device=device, dtype=dtype)  # (K, m)
    rw = rw.to(device=device, dtype=dtype)
    pc = pc.to(device=device, dtype=dtype)
    offset = subspace.coord_offset.to(device=device, dtype=dtype)  # (m,)
    scale = subspace.coord_scale.to(device=device, dtype=dtype)  # (m,)

    node_coords = node_coords.to(device=device, dtype=dtype)  # (K, n)
    K = node_coords.shape[0]

    node_vals = (
        eval_rbf(np_, rw, pc, np_) if node_values is None
        else node_values.to(device=device, dtype=dtype)
    )  # (K, R)
    if warm_start is not None:
        ws = warm_start.to(device=device, dtype=dtype).reshape(N, n)  # (N, n)
        near1 = torch.cdist(flat, node_vals).argmin(dim=-1)  # (N,)
        p = domain.clamp_position(
            torch.stack([ws, node_coords[near1]], dim=1)  # (N, 2, n)
        )
        iters = int(warm_iter)
    else:
        restarts = max(1, min(int(n_restarts), K))
        seed_idx = torch.cdist(flat, node_vals).topk(
            restarts, dim=-1, largest=False,
        ).indices  # (N, S)
        p = domain.clamp_position(node_coords[seed_idx])  # (N, S, n)
        iters = int(max_iter)
    q = flat.unsqueeze(1)  # (N, 1, R) -- broadcasts over the S restarts

    for _ in range(iters):
        p = _gn_step(p, q, np_, rw, pc, offset, scale, domain, damping)

    final = eval_rbf(np_, rw, pc, (domain.embed(p) - offset) / scale)
    final_res = torch.linalg.vector_norm(final - q, dim=-1)  # (N, S)
    best = final_res.argmin(dim=-1)  # (N,)
    pos = p.gather(1, best[:, None, None].expand(N, 1, n)).squeeze(1)  # (N, n)
    dist = final_res.gather(1, best[:, None]).squeeze(1)  # (N,)
    return pos.reshape(*lead, n), dist.reshape(lead)


__all__ = [
    "DEFAULT_INVERSION_DAMPING",
    "DEFAULT_INVERSION_MAX_ITER",
    "DEFAULT_INVERSION_RESTARTS",
    "DEFAULT_INVERSION_WARM_ITER",
    "invert_parameterization",
]
