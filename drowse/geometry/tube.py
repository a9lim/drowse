"""The fuzzy-manifold sigma field — a curved fit's tube thickness.

A node's corpus scatters off the mean surface; the off-surface (normal)
part of that within-node scatter is the tube thickness ``σ``.  The pipeline
computes each node's reduced within-node covariance from the retained
activation rows (:func:`drowse.fit.node_stats.compute_store_reduced_covariances`),
:func:`off_surface_vars` reduces it to one normal-space variance per node,
and :func:`fit_sigma_field` interpolates ``log σ`` with an RBF over the same
layout as the surface.  Soft ``onto`` steers into the tube and the monitor
reads it as the ``:membership`` channel.
"""
from __future__ import annotations

import torch

from drowse.geometry.domains import ManifoldDomain
from drowse.geometry.rbf import RbfFitPlan, eval_rbf_jacobian
from drowse.geometry.subspace import LayerSubspace


def _reduced_tangents(
    sub: "LayerSubspace", domain: "ManifoldDomain", coords: torch.Tensor,
) -> torch.Tensor:
    """Batched reduced-space surface tangents ``(K, R, n)``.

    Computes the RBF Jacobian chained through the domain's embedding Jacobian
    for every node at once — ``embed`` / ``embed_jacobian`` are batch-generic
    and ``eval_rbf_jacobian`` is vectorized, so the σ-field pass covers all K
    node tangent frames for a layer in one tensor sweep.
    """
    np_, rw, pc = sub.rbf_params()
    coords_f = coords.to(torch.float32)
    emb = sub._normalize(domain.embed(coords_f))                  # (K, m)
    j_red = eval_rbf_jacobian(np_, rw, pc, emb) / sub.coord_scale  # (K, R, m)
    return j_red @ domain.tangent_jacobian(coords_f)              # (K, R, n)


def off_surface_vars(
    covs: torch.Tensor, tangents: torch.Tensor, R: int, n: int,
) -> torch.Tensor:
    """Mean within-node variance in the off-surface (normal) directions.

    ``covs`` is ``(K, R, R)`` — each node's reduced within-node covariance —
    and ``tangents`` ``(K, R, n)`` its surface tangent.  Projects each
    covariance onto the normal complement of the *actual* tangent range and
    returns ``tr(P cov)/(R − rank(t))`` per node: the scatter that lives off
    the surface (tangential scatter is the node sliding along it, not
    thickness).  Degenerates to the isotropic ``tr(cov)/R`` only when the
    tangent fills the reduced subspace.  Clamped non-negative.
    """
    if covs.ndim != 3 or covs.shape[-2:] != (R, R):
        raise ValueError(
            f"covariances must have shape (K, {R}, {R}), got {tuple(covs.shape)}"
        )
    if tangents.shape != (covs.shape[0], R, n):
        raise ValueError(
            f"tangents must have shape ({covs.shape[0]}, {R}, {n}), "
            f"got {tuple(tangents.shape)}"
        )
    # One economy SVD supplies both the numerical rank and the projector onto
    # the *actual* local tangent range.  An RBF surface may have a fold/pinch
    # where rank(T) < n; using nominal ``R-n`` there overstates tube variance.
    # Match ``torch.linalg.matrix_rank``'s default relative tolerance so this
    # fused path preserves the old rank boundary without a second decomposition.
    U, singular_values, _ = torch.linalg.svd(tangents, full_matrices=False)
    relative_tol = (
        singular_values[..., :1]
        * (max(R, n) * torch.finfo(singular_values.dtype).eps)
    )
    active = singular_values > relative_tol
    tangent_rank = active.sum(dim=-1)
    tangent_frame = U * active.unsqueeze(-2)
    proj = tangent_frame @ U.transpose(-1, -2)
    normal = torch.eye(R, dtype=covs.dtype, device=covs.device) - proj
    normal_dof = R - tangent_rank
    normal_var = torch.einsum("kij,kji->k", normal, covs) / normal_dof.clamp_min(1)
    # A full-row-rank tangent fills the fitted subspace: there is no normal
    # direction to average. Preserve the established isotropic fallback.
    isotropic = torch.diagonal(covs, dim1=-2, dim2=-1).sum(dim=-1) / max(R, 1)
    return torch.where(normal_dof > 0, normal_var, isotropic).clamp(min=0.0)


def fit_sigma_field(
    layer_subs: "dict[int, LayerSubspace]",
    domain: "ManifoldDomain",
    node_coords: torch.Tensor,
    node_covs: "list[dict[int, torch.Tensor]]",
    *,
    smoothing: float | str | None = "auto",
    floor_frac: float = 1e-3,
    rbf_plan: RbfFitPlan | None = None,
) -> dict[int, dict[str, float]]:
    """Attach a fuzzy-manifold ``log σ`` RBF to each curved layer (mutates them).

    Reduces the per-node within-node covariances (from
    :func:`drowse.fit.node_stats.compute_store_reduced_covariances`) to one
    off-surface ``σ`` per node per layer (:func:`off_surface_vars`), then fits a
    *separate* penalized
    ``r**3`` RBF over the **same normalized** ``node_params`` interpolating the
    per-node ``log σ`` and writes it onto ``sub.sigma_rbf_weights`` /
    ``sub.sigma_poly_coeffs``.  Returns ``{layer: {"sigma_mean", "sigma_min",
    "sigma_max", "lambda"}}`` for the sidecar.

    ``floor_frac`` floors each layer's per-node ``σ²`` at ``floor_frac × median``
    so a degenerate (single-sample / collapsed) node can't drive ``log σ → −∞``.
    Smoothing defaults to GCV ``"auto"`` — the σ-field is noisier than the mean
    surface (a second-moment estimate from ~48 samples), so a regularized
    interpolant is the right default; ``0`` makes it exact at the nodes.
    """
    K = int(node_coords.shape[0])
    n = int(domain.intrinsic_dim)
    coords_f = node_coords.to(torch.float32)
    info: dict[int, dict[str, float]] = {}
    if not layer_subs:
        return info
    if rbf_plan is None:
        np_, _rw, _pc = next(iter(layer_subs.values())).rbf_params()
        rbf_plan = RbfFitPlan.build(np_, smoothing=smoothing)
    for idx, sub in layer_subs.items():
        R = sub.rank
        covs = torch.stack(
            [node_covs[kidx][idx].to(torch.float32) for kidx in range(K)],
            dim=0,
        )                                                        # (K, R, R)
        tangents = _reduced_tangents(sub, domain, coords_f)      # (K, R, n)
        raw = off_surface_vars(covs, tangents, R, n).to(torch.float32)
        floor = floor_frac * float(raw.median().clamp(min=1e-12))
        sigma = raw.clamp(min=floor).sqrt()                      # (K,) σ (std)
        log_sigma = torch.log(sigma).reshape(K, 1)               # (K, 1)
        fit = rbf_plan.fit(log_sigma, smoothing=smoothing)
        sub.sigma_rbf_weights = fit.weights
        sub.sigma_poly_coeffs = fit.poly
        info[idx] = {
            "sigma_mean": float(sigma.mean()),
            "sigma_min": float(sigma.min()),
            "sigma_max": float(sigma.max()),
            "lambda": fit.lam,
        }
    return info


__all__ = ["fit_sigma_field", "off_surface_vars"]
