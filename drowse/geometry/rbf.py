"""The ``r**3`` polyharmonic RBF — the per-layer surface of a curved manifold.

Per layer a curved manifold interpolates reduced activation coordinates with
one ``r**3`` polyharmonic RBF plus an affine polynomial term.  The kernel is
conditionally positive definite of order 2 in every dimension; with the
affine term it gives a smooth (C^2) interpolant, and at ``n == 1`` over an
open axis it is exactly the natural cubic spline.

:class:`RbfFitPlan` is the one factorization.  It depends only on the node
layout, so a multi-layer fit builds it once and every layer (and the sigma
field) is right-hand-side work:

* exact interpolation (``smoothing=None`` / ``0``) — the saddle LU;
* a fixed penalty ``λ`` (a float, on the mean-kernel-magnitude scale);
* ``"auto"`` — ``λ`` selected by generalized cross-validation through the
  Demmler-Reinsch eigensystem of the reduced kernel.

:func:`eval_rbf`, :func:`eval_rbf_jacobian` and :func:`eval_rbf_jet` are the
evaluation side; they are reachable from the generation hot path and are free
of host syncs.  Fitting is CPU/fp32: the saddle system is symmetric
indefinite (LU, never Cholesky) and unsupported on MPS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from drowse.errors import ManifoldSpecError, PoisednessError

# Log-spaced GCV grid over the mean off-diagonal kernel magnitude.
_GCV_GRID = (-6.0, 3.0, 40)


def _check_poised(node_params: torch.Tensor) -> tuple[int, int]:
    """Validate affine poisedness for an RBF fit; return ``(K, m)``.

    The saddle system is nonsingular exactly when the node coordinates are
    poised for affine interpolation (their affine rank equals ``m``); the
    penalty conditions only the kernel block, so the same check guards every
    smoothing mode.
    """
    K, m = node_params.shape
    if m + 1 > K:
        raise PoisednessError(
            f"RBF poisedness failure: {K} nodes cannot determine an affine "
            f"term in {m} embedding dimensions (need >= {m + 1})"
        )
    centered = node_params - node_params.mean(dim=0, keepdim=True)
    if int(torch.linalg.matrix_rank(centered)) != m:
        raise PoisednessError(
            f"RBF poisedness failure: the {K} node coordinates do not "
            f"affinely span the {m}-dim embedding space (they lie in a "
            f"lower-dimensional flat); spread the nodes across every axis"
        )
    return K, m


@dataclass(frozen=True)
class RbfFit:
    """One fitted RBF: weights plus the smoothing provenance.

    ``gcv`` is ``-1`` unless ``λ`` was GCV-selected.
    """

    weights: torch.Tensor   # (K, R)
    poly: torch.Tensor      # (m+1, R)
    lam: float
    edf: float
    gcv: float

    def info(self) -> dict[str, float]:
        """The ``{"lambda", "edf", "gcv"}`` sidecar record."""
        return {"lambda": self.lam, "edf": self.edf, "gcv": self.gcv}


@dataclass(frozen=True)
class RbfFitPlan:
    """Layout-only RBF factorization shared by every layer of a curved fit.

    ``node_params`` is unit-box normalized (``(raw − coord_offset) /
    coord_scale``); the RBF kernel amplifies coordinate scale, so the
    normalization is mandatory.  ``E``/``Q`` are the kernel and polynomial
    blocks, ``q2``/``gamma``/``eigenvectors`` the Demmler-Reinsch system of the
    reduced kernel ``G = Q2ᵀ E Q2`` (``Q2`` an orthonormal basis of
    ``null(Qᵀ)``), and ``fixed_lu`` the saddle LU at the plan's own fixed
    ``λ`` (``None`` for an ``"auto"`` plan).
    """

    node_params: torch.Tensor
    coord_offset: torch.Tensor
    coord_scale: torch.Tensor
    E: torch.Tensor
    Q: torch.Tensor
    e_scale: float
    grid: torch.Tensor
    q2: torch.Tensor
    gamma: torch.Tensor
    eigenvectors: torch.Tensor
    fixed_lambda: float | None
    fixed_lu: torch.Tensor | None
    fixed_pivots: torch.Tensor | None

    @classmethod
    def build(
        cls,
        node_params: torch.Tensor,
        *,
        smoothing: float | str | None = None,
    ) -> "RbfFitPlan":
        """Factorize the layout ``node_params`` ``(K, m)`` (raw embedded coords).

        ``smoothing`` picks the ``λ`` whose saddle LU is cached; any other
        smoothing still fits through :meth:`fit`, at the cost of one solve.
        """
        raw = node_params.to(device="cpu", dtype=torch.float32)
        K, m = _check_poised(raw)
        lo = raw.min(dim=0).values
        scale = (raw.max(dim=0).values - lo).clamp(min=1e-9)
        normalized = ((raw - lo) / scale).contiguous()
        E = torch.cdist(normalized, normalized).pow(3)
        Q = torch.cat([torch.ones(K, 1, dtype=torch.float32), normalized], dim=1)
        # Scale λ by the mean off-diagonal kernel magnitude (diag(E) = 0) so
        # the smoothing knob is invariant to coordinate scale.
        denom = K * K - K
        e_scale = float(E.abs().sum() / denom) if denom > 0 else 1.0
        if not math.isfinite(e_scale) or e_scale <= 0.0:
            e_scale = 1.0
        lo_exp, hi_exp, n_grid = _GCV_GRID
        grid = e_scale * torch.logspace(lo_exp, hi_exp, n_grid, dtype=E.dtype)
        mp1 = m + 1
        if K > mp1:
            q_full, _ = torch.linalg.qr(Q, mode="complete")
            q2 = q_full[:, mp1:]
            g = q2.transpose(0, 1) @ E @ q2
            g = 0.5 * (g + g.transpose(0, 1))          # symmetrize vs roundoff
            gamma, eigenvectors = torch.linalg.eigh(g)  # γⱼ ≥ 0 (cond. PSD)
            gamma = gamma.clamp_min(0.0)
        else:
            q2 = torch.empty(K, 0, dtype=E.dtype)
            gamma = torch.empty(0, dtype=E.dtype)
            eigenvectors = torch.empty(0, 0, dtype=E.dtype)
        plan = cls(
            node_params=normalized, coord_offset=lo, coord_scale=scale,
            E=E, Q=Q, e_scale=e_scale, grid=grid, q2=q2, gamma=gamma,
            eigenvectors=eigenvectors, fixed_lambda=None, fixed_lu=None,
            fixed_pivots=None,
        )
        fixed = plan._fixed_lambda(smoothing)
        if fixed is None:
            return plan
        lu, pivots = torch.linalg.lu_factor(plan._saddle(fixed))
        return replace(plan, fixed_lambda=fixed, fixed_lu=lu, fixed_pivots=pivots)

    @property
    def node_count(self) -> int:
        return int(self.node_params.shape[0])

    def normalize(self, embedded: torch.Tensor) -> torch.Tensor:
        """Map raw embedded coords into this plan's unit box."""
        return (embedded - self.coord_offset) / self.coord_scale

    def _fixed_lambda(self, smoothing: float | str | None) -> float | None:
        if smoothing is None:
            return 0.0
        if smoothing == "auto":
            return None
        if isinstance(smoothing, (int, float)) and not isinstance(smoothing, bool):
            return float(smoothing) * self.e_scale
        raise ManifoldSpecError(
            f"smoothing must be 'auto', a float, or 0/None; got {smoothing!r}"
        )

    def _saddle(self, lam: float) -> torch.Tensor:
        K = self.node_count
        mp1 = int(self.Q.shape[1])
        A = self.E + lam * torch.eye(K, dtype=self.E.dtype)
        return torch.cat([
            torch.cat([A, self.Q], dim=1),
            torch.cat([
                self.Q.transpose(0, 1),
                torch.zeros(mp1, mp1, dtype=self.E.dtype),
            ], dim=1),
        ], dim=0)

    def _shrink_ratios(self, lams: torch.Tensor) -> torch.Tensor:
        """``λᵢ / (γⱼ + λᵢ)`` — the eigenvalues of ``I − S_λ`` on ``null(Qᵀ)``."""
        return lams.unsqueeze(1) / (self.gamma.unsqueeze(0) + lams.unsqueeze(1))

    def _projected_sq(self, values: torch.Tensor) -> torch.Tensor:
        """``Σ_r bⱼᵣ²`` with ``b = Uᵀ Q2ᵀ Y`` — the residual spectrum of ``Y``."""
        b = self.eigenvectors.transpose(0, 1) @ (self.q2.transpose(0, 1) @ values)
        return b.pow(2).sum(dim=1)

    def edf(self, lam: float) -> float:
        """Effective degrees of freedom ``tr S_λ = K − Σⱼ λ/(γⱼ+λ)``."""
        if lam == 0.0 or self.gamma.numel() == 0:
            return float(self.node_count)
        ratios = self._shrink_ratios(torch.tensor([lam], dtype=self.E.dtype))
        return float(self.node_count) - float(ratios.sum().item())

    def gcv(self, values: torch.Tensor, lam: float) -> float:
        """GCV score ``K · ‖(I − S_λ) Y‖²_F / [tr(I − S_λ)]²`` at a fixed ``λ``."""
        if self.gamma.numel() == 0 or lam == 0.0:
            return math.inf
        values = values.to(device="cpu", dtype=torch.float32)
        ratios = self._shrink_ratios(torch.tensor([lam], dtype=self.E.dtype))[0]
        tr = float(ratios.sum().item())
        rss = float((ratios.pow(2) * self._projected_sq(values)).sum().item())
        return self.node_count * rss / (tr * tr) if tr > 0.0 else math.inf

    def select_lambda(self, values: torch.Tensor) -> tuple[float, float, float]:
        """Pick ``λ`` minimizing GCV over the log grid; ``(λ*, edf, gcv)``.

        ``V(λ) = K · ‖(I − S_λ) Y‖²_F / [tr(I − S_λ)]²``, the multi-output
        shared-smoother form.  ``λ = 0`` is excluded (``tr(I − S) = 0`` there,
        an indeterminate ``0/0``); the smallest grid ``λ`` is the
        near-interpolating limit.  Demmler-Reinsch collapses every grid point
        to scalar evals of ``γⱼ/(γⱼ+λ)``: ``I − S_λ = λ·Q2 (G + λI)⁻¹ Q2ᵀ``.
        """
        K = self.node_count
        if self.gamma.numel() == 0:
            # Q square ⇒ no penalized null space; the polynomial fits every
            # node exactly (S = I) and GCV is 0/0 over the whole grid.
            return float(self.grid[0].item()), float(K), math.inf
        values = values.to(device="cpu", dtype=torch.float32)
        ratios = self._shrink_ratios(self.grid)                     # (n_grid, null)
        tr = ratios.sum(dim=1)                                      # tr(I − S_λ)
        rss = (ratios.pow(2) * self._projected_sq(values).unsqueeze(0)).sum(dim=1)
        gcv = torch.where(tr > 0.0, K * rss / (tr * tr), torch.full_like(tr, math.inf))
        best = int(torch.argmin(gcv).item())                        # smallest λ on ties
        return (
            float(self.grid[best].item()),
            float(K) - float(tr[best].item()),
            float(gcv[best].item()),
        )

    def solve(
        self, values: torch.Tensor, lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve the penalized saddle for ``values`` ``(K, R)``; ``(w, c)``.

        ``[[E + λI, Q], [Qᵀ, 0]] [w; c] = [values; 0]`` — reusing the cached
        LU when ``λ`` is the plan's own fixed penalty.
        """
        values = values.to(device="cpu", dtype=torch.float32)
        K = self.node_count
        mp1 = int(self.Q.shape[1])
        rhs = torch.cat([
            values, torch.zeros(mp1, values.shape[1], dtype=values.dtype),
        ], dim=0)
        if self.fixed_lambda is not None and lam == self.fixed_lambda:
            assert self.fixed_lu is not None and self.fixed_pivots is not None
            sol = torch.linalg.lu_solve(self.fixed_lu, self.fixed_pivots, rhs)
        else:
            sol = torch.linalg.solve(self._saddle(lam), rhs)
        return sol[:K].contiguous(), sol[K:].contiguous()

    def fit(
        self, values: torch.Tensor, *, smoothing: float | str | None = None,
    ) -> RbfFit:
        """Fit the (penalized) interpolant of ``values`` ``(K, R)`` over the layout.

        The thin-plate / Duchon smoothing spline: minimize
        ``‖y − Ew − Qc‖² + λ wᵀEw`` subject to ``Qᵀw = 0``.  ``λ = 0``
        (``smoothing`` ``None``/``0``) interpolates the node values exactly;
        ``λ > 0`` shrinks toward the affine trend.  ``"auto"`` GCV-selects
        ``λ``; a float is a fixed ``λ`` on the mean-kernel-magnitude scale.
        """
        fixed = self._fixed_lambda(smoothing)
        if fixed is None:
            lam, edf, gcv = self.select_lambda(values)
        else:
            lam, edf, gcv = fixed, self.edf(fixed), -1.0
        w, c = self.solve(values, lam)
        return RbfFit(weights=w, poly=c, lam=float(lam), edf=float(edf), gcv=float(gcv))


def eval_rbf(
    node_params: torch.Tensor,
    rbf_weights: torch.Tensor,
    poly_coeffs: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the interpolant at ``query`` ``(.., m)`` (embedded + normalized).

    ``node_params`` ``(K, m)``, ``rbf_weights`` ``(K, R)``, ``poly_coeffs``
    ``(m+1, R)``.  Returns ``(.., R)``.  Hot-path safe: no host sync.
    """
    diff = query.unsqueeze(-2) - node_params  # (.., K, m)
    r = torch.linalg.vector_norm(diff, dim=-1)  # (.., K)
    phi = r.pow(3)
    rbf_part = torch.matmul(phi, rbf_weights)  # (.., R)
    aug = torch.cat(
        [torch.ones_like(query[..., :1]), query], dim=-1,
    )  # (.., m+1)
    poly_part = torch.matmul(aug, poly_coeffs)  # (.., R)
    return rbf_part + poly_part


def eval_rbf_jacobian(
    node_params: torch.Tensor,
    rbf_weights: torch.Tensor,
    poly_coeffs: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Analytic Jacobian ``d s / d query``: ``(.., m) -> (.., R, m)``.

    ``d/dx[r**3] = 3 r (x − p_j)``; the polynomial contributes its linear
    coefficients.  No autograd.
    """
    diff = query.unsqueeze(-2) - node_params  # (.., K, m)
    r = torch.linalg.vector_norm(diff, dim=-1)  # (.., K)
    grad_phi = 3.0 * r.unsqueeze(-1) * diff  # (.., K, m)
    j_rbf = torch.einsum("kr,...km->...rm", rbf_weights, grad_phi)  # (.., R, m)
    return j_rbf + poly_coeffs[1:].T


def eval_rbf_jet(
    node_params: torch.Tensor,
    rbf_weights: torch.Tensor,
    poly_coeffs: torch.Tensor,
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused :func:`eval_rbf` + :func:`eval_rbf_jacobian` sharing ``diff``/``r``.

    Returns ``(value (.., R), jacobian (.., R, m))``, numerically identical to
    the two separate calls.  The foot solve needs both at every step, so the
    fusion halves the distance work on the curved hot path.
    """
    diff = query.unsqueeze(-2) - node_params  # (.., K, m)
    r = torch.linalg.vector_norm(diff, dim=-1)  # (.., K)
    value = torch.matmul(r.pow(3), rbf_weights) + torch.matmul(
        torch.cat([torch.ones_like(query[..., :1]), query], dim=-1), poly_coeffs,
    )
    grad_phi = 3.0 * r.unsqueeze(-1) * diff
    jac = torch.einsum("kr,...km->...rm", rbf_weights, grad_phi) + poly_coeffs[1:].T
    return value, jac


def rbf_cardinal_weights(
    node_coords: torch.Tensor, query: torch.Tensor,
) -> torch.Tensor:
    """Cardinal ``r**3``-RBF weights ``w(z) ∈ ℝ^K`` of a layout at ``query``.

    For any per-node values ``Y`` ``(K, R)`` the exact interpolant of ``Y`` at
    ``query`` equals ``w @ Y``.  The weights depend only on the layout and the
    query, so one solve serves every layer; they are exact at the nodes
    (``w = e_i`` at node ``i``) and a partition of unity.  This is the
    flat-manifold coord-form analogue of a curved fit's surface: applied to a
    flat fit's per-layer node coords it reproduces label-form steering at the
    nodes and follows the learned layout between them.

    CPU/fp32, once per steering compose.  Raises ``ValueError`` when the
    layout is not affinely poised (no interpolant exists).
    """
    query = query.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    plan = RbfFitPlan.build(node_coords.detach(), smoothing=None)
    eye = torch.eye(plan.node_count, dtype=torch.float32)
    rbf_weights, poly_coeffs = plan.solve(eye, 0.0)
    return eval_rbf(plan.node_params, rbf_weights, poly_coeffs, plan.normalize(query))


__all__ = [
    "RbfFit",
    "RbfFitPlan",
    "eval_rbf",
    "eval_rbf_jacobian",
    "eval_rbf_jet",
    "rbf_cardinal_weights",
]
