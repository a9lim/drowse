"""Per-layer manifold subspaces and the fits that produce them.

A :class:`LayerSubspace` is one layer's frame: an affine PCA subspace of the
activation space (``mean`` + orthonormal ``basis`` rows), optionally carrying
an RBF surface from the manifold's embedded domain coordinates into that
subspace's reduced coordinates (a curved fit), and optionally a
fuzzy-manifold sigma field (the within-node tube thickness).

The two fits share the basis selection (:func:`pca_basis`, whitened/Fisher
when a covering whitener is supplied) and the neutral-anchored frame; only
what is built on top differs:

* :func:`fit_affine_subspace` — flat, the surface fills the span (a steering
  vector is the ``K = 2`` case);
* :func:`fit_curved_subspace` — an ``r**3`` RBF surface over an
  :class:`~drowse.geometry.rbf.RbfFitPlan` layout.

:func:`subspace_share` turns the μ-centered node coords either fit returns
into the per-layer Mahalanobis budget weight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

from drowse.geometry.rbf import RbfFit, RbfFitPlan, eval_rbf

if TYPE_CHECKING:
    from drowse.geometry.whitener import LayerWhitener


# Default PCA width for a fitted manifold subspace.  Matches the paper's
# 64-dimensional reduction.  Clamped down to ``min(64, K-1, rank)`` when
# the node count ``K`` is small -- ``K`` centered centroids span a space
# of rank at most ``K-1``.
DEFAULT_N_COMPONENTS = 64


@dataclass
class LayerSubspace:
    """Per-layer reduced frame and RBF interpolant for one manifold.

    ``mean`` and ``basis`` define an affine PCA subspace of the
    activation space; ``node_params`` / ``rbf_weights`` / ``poly_coeffs``
    define the ``r**3`` RBF interpolant from the manifold's embedded
    domain coordinates to that subspace's reduced coordinates.
    ``coord_offset`` / ``coord_scale`` carry the unit-box normalization
    applied to the embedded coordinates before the fit (the RBF kernel
    amplifies coordinate scale, so normalization is mandatory).

    **Affine (flat) case.**  When ``node_params`` / ``rbf_weights`` /
    ``poly_coeffs`` are ``None`` the subspace carries no RBF surface — the
    "surface" *is* the whole affine subspace (a folded steering vector at
    ``n = R``).  Ordinary fits use the identity authoring→reduced map;
    rectangular cross-model transfers may carry an explicit ``affine_map`` so
    ``manifold_point(c) = mean + c @ affine_map @ basis`` while ``basis`` stays
    orthonormal.  The
    affine case has ``H_n ≡ 0`` (the surface fills its subspace), so
    ``subspace_inject`` takes an analytic shortcut that skips the
    Gauss-Newton foot solve, the RBF eval, and the tangent Gram-solve —
    load-bearing for throughput, since a folded vector is the common
    steering case and the curved per-token solve would blow the
    throughput invariant.  Build one via :meth:`affine`; query via
    :attr:`is_affine`.
    """

    mean: torch.Tensor                  # (D,)   centering mean over the node centroids
    basis: torch.Tensor                 # (R, D) orthonormal PCA rows
    node_params: torch.Tensor | None    # (K, m) normalized embedded coords; None = affine
    rbf_weights: torch.Tensor | None    # (K, R) RBF weights; None = affine
    poly_coeffs: torch.Tensor | None    # (m+1, R) affine polynomial coeffs; None = affine
    coord_offset: torch.Tensor          # (m,)   unit-box normalization offset
    coord_scale: torch.Tensor           # (m,)   unit-box normalization scale
    node_coords: torch.Tensor | None = None
    # (K, R) per-layer **real, neutral-anchored** reduced node coordinates
    # ``(c_i − ν*)·basisᵀ`` — affine (flat) subspaces only; ``None`` on a
    # curved subspace (which carries the shared ``Manifold.node_coords`` +
    # RBF instead).  This is the per-layer steer target source: a flat
    # subspace's pole / node sits at distance ∝ ‖δ_L‖ from the neutral
    # origin here, so the synthesizer reads ``node_coords[index]`` per layer
    # as the ``along`` target.  The shared ``Manifold.node_coords`` stays the
    # label/display layout; *these* are the neutral-anchored geometry.
    affine_map: torch.Tensor | None = None
    # (m, R) authoring-to-reduced coordinate map for affine subspaces.  ``None``
    # is the canonical identity map (m == R). Cross-model transfer may need a
    # non-isometric map after orthonormalizing a rectangular mapped basis; this
    # explicit factor preserves the exact world surface without weakening the
    # runtime's orthonormal-basis invariant.
    sigma_rbf_weights: torch.Tensor | None = None
    sigma_poly_coeffs: torch.Tensor | None = None
    # The **fuzzy-manifold σ-field** (raw curved subspaces only). A separate
    # ``r**3`` RBF over
    # the **same** normalized ``node_params`` that interpolates per-node
    # ``log σ`` — the within-node off-surface activation spread (the corpus a
    # node produces scatters off the mean surface; ``σ`` is that scatter's
    # normal-projected std, a tube thickness).  Kept separate from the surface
    # RBF (rather than appended as an extra value column) so the ``(R,)``-shape
    # contracts the surface consumers rely on are untouched; ``sigma_at`` is the
    # one extra ``eval_rbf`` (``O(K)``) paid only on the already-slow curved
    # path. ``sigma_rbf_weights`` is ``(K, 1)``, ``sigma_poly_coeffs`` is
    # ``(m+1, 1)``. Affine and SAE-curved fits do not model a raw-activation
    # tube and therefore carry neither sigma tensor.

    @property
    def rank(self) -> int:
        return int(self.basis.shape[0])

    @property
    def is_affine(self) -> bool:
        """True for a flat (no-RBF) subspace — the surface fills its span.

        The analytic-shortcut marker: ``subspace_inject`` and ``eval_at``
        branch on this to skip all RBF / Gauss-Newton machinery.  A folded
        steering vector (``n = R``, the chord) is built affine; every
        RBF-fitted manifold (curved, or merely space-filling like a discover
        fit at ``R = n``) is not.
        """
        return self.node_params is None

    def validate_structure(
        self, *, feature_space: str, expected_node_count: int,
    ) -> None:
        """Validate the exact affine/curved discriminated tensor shape."""
        if self.mean.ndim != 1:
            raise ValueError("LayerSubspace mean must have shape (D,)")
        if self.basis.ndim != 2 or self.basis.shape[0] < 1:
            raise ValueError("LayerSubspace basis must have shape (R, D) with R >= 1")
        rank, dim = self.basis.shape
        if self.mean.shape[0] != dim:
            raise ValueError("LayerSubspace mean dimension must match basis width")
        if self.coord_offset.ndim != 1 or self.coord_scale.ndim != 1:
            raise ValueError("LayerSubspace coordinate normalization must be vectors")
        if self.coord_offset.shape != self.coord_scale.shape:
            raise ValueError("LayerSubspace coord_offset/coord_scale shapes must match")
        embed_dim = int(self.coord_offset.shape[0])
        surface = (self.node_params, self.rbf_weights, self.poly_coeffs)
        surface_present = tuple(value is not None for value in surface)
        sigma_present = (
            self.sigma_rbf_weights is not None,
            self.sigma_poly_coeffs is not None,
        )
        if any(surface_present) and not all(surface_present):
            raise ValueError("curved LayerSubspace requires the complete RBF triple")
        if sigma_present[0] != sigma_present[1]:
            raise ValueError("LayerSubspace sigma tensors must be present as a pair")
        if not any(surface_present):
            if self.node_coords is None:
                raise ValueError("affine LayerSubspace requires node_coords")
            if self.node_coords.shape != (expected_node_count, rank):
                raise ValueError(
                    "affine LayerSubspace node_coords must have shape (K, R)"
                )
            if self.affine_map is None:
                if embed_dim != rank:
                    raise ValueError(
                        "affine LayerSubspace without affine_map requires m == R"
                    )
            elif self.affine_map.shape != (embed_dim, rank):
                raise ValueError("affine_map must have shape (m, R)")
            if any(sigma_present):
                raise ValueError("affine LayerSubspace cannot carry a sigma field")
            return
        if self.node_coords is not None or self.affine_map is not None:
            raise ValueError(
                "curved LayerSubspace cannot carry affine node_coords/affine_map"
            )
        if feature_space == "raw" and not all(sigma_present):
            raise ValueError("raw curved LayerSubspace requires a sigma field")
        assert self.node_params is not None
        assert self.rbf_weights is not None
        assert self.poly_coeffs is not None
        if self.node_params.shape != (expected_node_count, embed_dim):
            raise ValueError("curved node_params must have shape (K, m)")
        if self.rbf_weights.shape != (expected_node_count, rank):
            raise ValueError("curved rbf_weights must have shape (K, R)")
        if self.poly_coeffs.shape != (embed_dim + 1, rank):
            raise ValueError("curved poly_coeffs must have shape (m + 1, R)")
        if all(sigma_present):
            assert self.sigma_rbf_weights is not None
            assert self.sigma_poly_coeffs is not None
            if self.sigma_rbf_weights.shape != (expected_node_count, 1):
                raise ValueError("sigma_rbf_weights must have shape (K, 1)")
            if self.sigma_poly_coeffs.shape != (embed_dim + 1, 1):
                raise ValueError("sigma_poly_coeffs must have shape (m + 1, 1)")

    @classmethod
    def affine(
        cls,
        mean: torch.Tensor,
        basis: torch.Tensor,
        *,
        node_coords: torch.Tensor | None = None,
        affine_map: torch.Tensor | None = None,
    ) -> "LayerSubspace":
        """Build a flat (affine, no-RBF) subspace from ``mean`` + ``basis``.

        By default authoring coordinates map to reduced coordinates by identity
        (``coord_offset = 0``, ``coord_scale = 1``), so
        ``eval_at(c) = c @ basis + mean`` exactly.  ``affine_map`` optionally
        supplies an ``(m, R)`` map after cross-model reparameterization.
        ``basis`` is ``(R, D)``;
        the implied intrinsic dimension is ``n = R`` (the surface fills the
        span).  Backs the folded-vector / flat-subspace representation.
        ``node_coords`` ``(K, R)`` carries the per-layer real
        neutral-anchored node positions (the steer-target source);
        ``None`` for a bare span with no associated nodes.
        """
        r = int(basis.shape[0])
        m = r
        if affine_map is not None:
            if affine_map.ndim != 2 or affine_map.shape[1] != r:
                raise ValueError(
                    f"affine_map must have shape (m, {r}), got "
                    f"{tuple(affine_map.shape)}"
                )
            m = int(affine_map.shape[0])
        return cls(
            mean=mean,
            basis=basis,
            node_params=None,
            rbf_weights=None,
            poly_coeffs=None,
            coord_offset=torch.zeros(m, device=mean.device, dtype=mean.dtype),
            coord_scale=torch.ones(m, device=mean.device, dtype=mean.dtype),
            node_coords=node_coords,
            affine_map=affine_map,
        )

    def select_axes(self, kept: Sequence[int]) -> "LayerSubspace":
        """Restrict an affine subspace to a subset of its basis rows (DLS prune).

        Slices ``basis`` + ``node_coords`` to the ``kept`` axis indices and
        recomputes ``mean`` as the projection of the *same* anchor into the
        reduced span (``mean' = P_{basis[kept]}(anchor)``) — recovered from the
        stored ``mean`` without the raw anchor, since the anchor's per-axis
        coords are ``mean @ basisᵀ`` (the basis is orthonormal).  Affine only
        — a curved subspace's per-axis pruning would force an RBF re-fit, so
        the split is flat-yes / curved-no.  ``kept`` indexes into
        ``[0, rank)``; an empty ``kept`` is a caller error (drop the layer).

        Bakes the DLS keep set into the stored basis at fit time (the N-node
        analogue of the folded vector dropping a non-discriminative *layer*),
        so the steer/apply path needs no separate per-axis mask.
        """
        if not self.is_affine:
            raise ValueError(
                "select_axes is affine-only (curved has no per-axis DLS)"
            )
        idx = list(kept)
        if not idx:
            raise ValueError(
                "select_axes: empty kept set — drop the layer instead"
            )
        if (
            any(isinstance(axis, bool) for axis in idx)
            or any(axis < 0 or axis >= self.rank for axis in idx)
            or len(set(idx)) != len(idx)
        ):
            raise ValueError(
                f"select_axes: kept axes must be unique indices in [0, {self.rank})"
            )
        basis_k = self.basis[idx].contiguous()                 # (|kept|, D)
        anchor_coords = self.mean @ self.basis.T               # (R,) anchor coords
        mean_k = anchor_coords[idx] @ basis_k                  # P_{basis[kept]}(anchor)
        nc_k = (
            self.node_coords[:, idx].contiguous()
            if self.node_coords is not None else None
        )
        if self.affine_map is not None:
            amap_k = self.affine_map[:, idx].contiguous()
        elif idx != list(range(self.rank)):
            amap_k = torch.eye(
                self.rank,
                device=self.basis.device,
                dtype=self.basis.dtype,
            )[:, idx].contiguous()
        else:
            amap_k = None
        return LayerSubspace.affine(
            mean_k, basis_k, node_coords=nc_k, affine_map=amap_k,
        )

    def rbf_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The validated ``(node_params, rbf_weights, poly_coeffs)`` triple.

        Raises on an affine (flat) subspace — every RBF call site operates on
        the curved path, and the affine path has analytic equivalents that
        must never reach the interpolant.  Doubles as a guardrail: a stray
        affine subspace routed through the RBF machinery raises loudly
        instead of silently dereferencing ``None``.
        """
        np_, rw, pc = self.node_params, self.rbf_weights, self.poly_coeffs
        if np_ is None or rw is None or pc is None:
            raise ValueError(
                "LayerSubspace.rbf_params() called on an affine (flat) "
                "subspace; the affine path has analytic equivalents and must "
                "not reach the RBF interpolant"
            )
        return np_, rw, pc

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "LayerSubspace":
        """Return a copy with every tensor on ``device`` in ``dtype``."""
        def _cast(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t.to(device=device, dtype=dtype)
        return LayerSubspace(
            mean=self.mean.to(device=device, dtype=dtype),
            basis=self.basis.to(device=device, dtype=dtype),
            node_params=_cast(self.node_params),
            rbf_weights=_cast(self.rbf_weights),
            poly_coeffs=_cast(self.poly_coeffs),
            coord_offset=self.coord_offset.to(device=device, dtype=dtype),
            coord_scale=self.coord_scale.to(device=device, dtype=dtype),
            node_coords=_cast(self.node_coords),
            affine_map=_cast(self.affine_map),
            sigma_rbf_weights=_cast(self.sigma_rbf_weights),
            sigma_poly_coeffs=_cast(self.sigma_poly_coeffs),
        )

    @property
    def has_sigma(self) -> bool:
        """True iff this subspace carries a fuzzy-manifold σ-field.

        A raw curved subspace fitted with the within-node spread pass; ``False``
        for affine and SAE-curved fits, which do not model raw tube density.
        """
        return self.sigma_rbf_weights is not None and self.sigma_poly_coeffs is not None

    def _normalize(self, embedded: torch.Tensor) -> torch.Tensor:
        return (embedded - self.coord_offset) / self.coord_scale

    def eval_at(self, embedded: torch.Tensor) -> torch.Tensor:
        """World-space activation ``(.., D)`` at embedded domain coords ``(.., m)``."""
        if self.is_affine:
            # Flat: the canonical representation has an identity
            # authoring→reduced map.  Rectangular cross-model transfer may
            # carry an explicit map after orthonormalizing the target basis.
            reduced = embedded if self.affine_map is None else embedded @ self.affine_map
            return reduced @ self.basis + self.mean
        reduced = eval_rbf(*self.rbf_params(), self._normalize(embedded))
        return reduced @ self.basis + self.mean

    def sigma_at(
        self, embedded: torch.Tensor, *, normalized: bool = False,
    ) -> torch.Tensor:
        """Within-node off-surface spread ``σ`` at embedded coords ``(.., m)``.

        Interpolates the per-node ``log σ`` field through the σ-RBF over the
        same normalized ``node_params``, then exponentiates — so the result is
        a positive ``(..,)`` thickness in the layer's reduced-coordinate units
        (the same units ``H_n`` is measured in, since the basis is
        orthonormal).  ``normalized=True`` takes coords already mapped through
        :meth:`_normalize` (the injection kernel has them in hand).  Returns
        an all-zeros ``(..,)`` when the subspace carries no σ-field
        (:attr:`has_sigma` false): affine and SAE-curved fits.  Hot-path safe
        (one extra ``eval_rbf``, no ``.item()`` / host sync).
        """
        lead = embedded.shape[:-1]
        sw, sp, np_ = self.sigma_rbf_weights, self.sigma_poly_coeffs, self.node_params
        if sw is None or sp is None or np_ is None:
            return torch.zeros(lead, device=embedded.device, dtype=embedded.dtype)
        dev, dt = embedded.device, embedded.dtype
        norm = embedded if normalized else (
            (embedded - self.coord_offset.to(dev, dt)) / self.coord_scale.to(dev, dt)
        )
        log_sigma = eval_rbf(
            np_.to(dev, dt), sw.to(dev, dt), sp.to(dev, dt), norm,
        )  # (.., 1)
        return torch.exp(log_sigma).squeeze(-1)  # (..,)


def pca_basis(
    X: torch.Tensor,
    *,
    n_components: int = DEFAULT_N_COMPONENTS,
    whitener: "LayerWhitener | None" = None,
    layer: int | None = None,
    whitened_gram: torch.Tensor | None = None,
    whitened_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """μ-centered PCA basis — whitened/Fisher, or Euclidean without a whitener.

    ``X`` is the ``(K, D)`` **μ-centered** centroid scatter (never the neutral
    anchor: anchor-centering injects the neutral offset as a spurious axis and
    breaks PCA@2 ≡ DiM).  Returns ``(basis, ev_ratio)``: ``basis`` is ``(R, D)``
    orthonormal rows, ``R = min(n_components, K-1, rank)``.

    With ``whitener``/``layer`` the basis is the Fisher discriminant — the
    generalized eigenproblem ``(S_b, Σ)`` via the Woodbury ``Σ⁻¹``: eigvecs
    ``a`` of ``G = X Σ⁻¹ Xᵀ`` give directions ``Σ⁻¹ Xᵀ a``, re-expressed in a
    Euclidean-orthonormal basis by QR (span-preserving) so the hot path is
    unchanged.  Dividing by background variance cancels the rogue
    (massive-activation) dims raw PCA chases.  ``ev_ratio`` is the retained
    fraction of whitened between-variance.  ``whitened_gram`` / ``whitened_rows``
    may supply a precomputed ``X Σ⁻¹ Xᵀ`` / ``Σ⁻¹X`` so a fit shares one
    Woodbury application across Gram, basis, and layout anchoring.

    Without a whitener: ordinary SVD of ``X`` (raw inter-node variance).  Only
    tests and offline diagnostics take this branch; every activation-space fit
    in the pipeline passes a covering whitener.
    """
    K = int(X.shape[0])
    if whitener is not None and layer is not None:
        G = (
            whitened_gram.to(dtype=torch.float32, device="cpu")
            if whitened_gram is not None
            else whitener.subspace_gram(layer, X)
        )                                               # (K, K) = X Σ⁻¹ Xᵀ
        if G.shape != (K, K):
            raise ValueError(
                f"whitened_gram shape {tuple(G.shape)} does not match "
                f"centered scatter shape ({K}, {K})"
            )
        mu, A = torch.linalg.eigh(G)                    # ascending
        mu_pos = mu.clamp_min(0.0)
        rank = int((mu_pos > 1e-6 * mu_pos[-1].clamp(min=1e-12)).sum().item())
        R = max(1, min(n_components, K - 1, rank))
        top = torch.argsort(mu, descending=True)[:R]
        if whitened_rows is not None:
            sinv_x = whitened_rows.to(dtype=torch.float32, device="cpu")
            if sinv_x.shape != X.shape:
                raise ValueError(
                    f"whitened_rows shape {tuple(sinv_x.shape)} does not match "
                    f"centered scatter shape {tuple(X.shape)}"
                )
            directions = A[:, top].transpose(0, 1) @ sinv_x
        else:
            XtA = X.transpose(0, 1) @ A[:, top]         # (D, R) = Xᵀ a_r
            directions = whitener.apply_inv(
                layer, XtA.transpose(0, 1).contiguous(),
            )                                           # (R, D) = Σ⁻¹ Xᵀ a_r
        basis = torch.linalg.qr(
            directions.transpose(0, 1)
        ).Q.transpose(0, 1).contiguous()                # (R, D)
        total_w = float(mu_pos.sum().item())
        retained_w = float(mu_pos[top].sum().item())
        ev_ratio = retained_w / total_w if total_w > 1e-12 else 1.0
    else:
        _, S, Vh = torch.linalg.svd(X, full_matrices=False)
        rank = int((1e-6 * S[0].clamp(min=1e-12) < S).sum().item())
        R = max(1, min(n_components, K - 1, rank))
        basis = Vh[:R].contiguous()                     # (R, D)
        total_var = float(S.pow(2).sum().item())
        retained_var = float(S[:R].pow(2).sum().item())
        ev_ratio = retained_var / total_var if total_var > 1e-12 else 1.0
    return basis, ev_ratio


@dataclass(frozen=True)
class LayerFit:
    """One layer's fit: the subspace plus what the pipeline bakes from it.

    ``mu_coords`` ``(K, R)`` are the μ-centered reduced node coords —
    :func:`subspace_share`'s input (the anchor-independent budget weight).
    ``explained_variance`` is the retained (whitened) between-variance
    fraction, a diagnostic.  ``rbf`` is the penalized-surface record of a
    smoothed curved fit; ``None`` for flat and exact-interpolation fits.
    """

    subspace: LayerSubspace
    mu_coords: torch.Tensor
    explained_variance: float
    rbf: RbfFit | None = None


def _frame(
    centroids: torch.Tensor,
    *,
    n_components: int,
    whitener: "LayerWhitener | None",
    layer: int | None,
    neutral_mean: torch.Tensor | None,
    whitened_gram: torch.Tensor | None,
    whitened_rows: torch.Tensor | None,
    basis_override: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Shared basis + anchor of both fits: ``(mu, X, basis, anchor, ev_ratio)``.

    The basis comes from the μ-centered scatter (the PCA@2 ≡ DiM caveat); the
    frame anchors at ``neutral_mean`` when given, else at the centroid mean.
    """
    mu = centroids.mean(dim=0)
    X = centroids - mu
    if basis_override is None:
        basis, ev_ratio = pca_basis(
            X, n_components=n_components, whitener=whitener, layer=layer,
            whitened_gram=whitened_gram, whitened_rows=whitened_rows,
        )
    else:
        basis = basis_override[: min(n_components, basis_override.shape[0])].to(
            device="cpu", dtype=torch.float32,
        ).contiguous()
        ev_ratio = 1.0  # the planned Fisher basis carries no diagnostic
    anchor = (
        neutral_mean.to(device=centroids.device, dtype=torch.float32).reshape(-1)
        if neutral_mean is not None else mu
    )
    return mu, X, basis, anchor, ev_ratio


def fit_affine_subspace(
    centroids: torch.Tensor,
    *,
    neutral_mean: torch.Tensor | None = None,
    n_components: int = DEFAULT_N_COMPONENTS,
    whitener: "LayerWhitener | None" = None,
    layer: int | None = None,
    whitened_gram: torch.Tensor | None = None,
    whitened_rows: torch.Tensor | None = None,
    orient_to: int | None = 0,
    basis_override: torch.Tensor | None = None,
) -> LayerFit:
    """Fit a flat (affine, no-RBF) subspace from per-node centroids ``(K, D)``.

    ``fit_mode=pca`` produces these, and a steering vector is the ``K = 2``
    case.  The basis is the μ-centered PCA (:func:`pca_basis`); the frame is
    **neutral-anchored**:

    - ``mean = P_basis(anchor)`` — the anchor's projection into the span;
    - ``node_coords = (centroids − anchor)·basisᵀ`` — real, anchor-relative
      reduced coords (neutral sits at coord 0, a node at distance ∝ ‖δ_L‖).

    ``orient_to`` flips each basis row so node ``orient_to``'s μ-centered
    projection is non-negative (the ``K = 2`` node-0-is-pos case reproduces
    DiM's ``+δ̂``); ``None`` keeps the raw SVD/QR sign.
    """
    centroids = centroids.to(torch.float32)
    K = int(centroids.shape[0])
    if K < 2:
        raise ValueError(f"an affine subspace needs >= 2 nodes, got {K}")
    mu, X, basis, anchor, ev_ratio = _frame(
        centroids, n_components=n_components, whitener=whitener, layer=layer,
        neutral_mean=neutral_mean, whitened_gram=whitened_gram,
        whitened_rows=whitened_rows, basis_override=basis_override,
    )
    if orient_to is not None:
        proj = basis @ (centroids[orient_to] - mu)      # (R,)
        signs = torch.where(proj < 0, -1.0, 1.0)        # flip rows facing away
        basis = (basis * signs.unsqueeze(1)).contiguous()
    mean = (anchor @ basis.T) @ basis                   # P_basis(anchor) (D,)
    node_coords = (centroids - anchor) @ basis.T        # (K, R) anchor-relative
    return LayerFit(
        subspace=LayerSubspace.affine(mean, basis, node_coords=node_coords),
        mu_coords=X @ basis.T,
        explained_variance=ev_ratio,
    )


def subspace_share(
    mu_coords: torch.Tensor,
    basis: torch.Tensor,
    *,
    whitener: "LayerWhitener",
    layer: int,
) -> float:
    """Per-layer budget share — the whitened μ-centered node spread.

    ``share_L = sqrt(Σ_k coords_kᵀ M_R coords_k)`` with ``M_R = B Σ⁻¹ Bᵀ``
    (:meth:`~drowse.geometry.whitener.LayerWhitener.subspace_gram`): the
    spread of the node centroids around their *own* mean, restricted to the
    subspace and measured in the Mahalanobis metric.  Anchor-independent — the
    budget measures signal spread, not where neutral sits.  The apply path
    normalizes it to mean 1 across layers.  At ``K = 2`` / ``R = 1`` this is
    ``‖δ_L‖_M / √2``, so the normalized profile is the DiM one exactly.
    """
    mu_coords = mu_coords.to(torch.float32)
    if mu_coords.ndim == 1:
        mu_coords = mu_coords.reshape(-1, 1)
    M_R = whitener.subspace_gram(layer, basis.to(torch.float32))  # (R, R)
    quad = float((mu_coords @ M_R * mu_coords).sum().clamp_min(0.0).item())
    return quad ** 0.5


def fit_curved_subspace(
    centroids: torch.Tensor,
    layout: "RbfFitPlan | torch.Tensor",
    *,
    smoothing: float | str | None = None,
    n_components: int = DEFAULT_N_COMPONENTS,
    whitener: "LayerWhitener | None" = None,
    layer: int | None = None,
    neutral_mean: torch.Tensor | None = None,
    whitened_gram: torch.Tensor | None = None,
    whitened_rows: torch.Tensor | None = None,
    basis_override: torch.Tensor | None = None,
) -> LayerFit:
    """Fit a PCA subspace + ``r**3`` RBF surface for one layer (curved).

    ``centroids`` is ``(K, D)``; ``layout`` is the shared
    :class:`~drowse.geometry.rbf.RbfFitPlan` over the ``(K, m)`` embedded node
    coordinates (or those raw coordinates, planned here with ``smoothing``).
    The activations reduce to ``R = min(n_components, K-1, rank)`` components
    (:func:`pca_basis` — Fisher with a covering whitener) and the RBF maps the
    normalized coordinates to the **anchor-relative** reduced coords, so
    neutral lands at reduced-coord 0 and the surface passes through each
    centroid's in-span projection.

    ``smoothing=None`` is exact interpolation (the authored contract: node =
    exact steering target); ``"auto"`` or a float fits the penalized surface
    and records ``λ``/edf/GCV on :attr:`LayerFit.rbf`.
    """
    centroids = centroids.to(torch.float32)
    K = centroids.shape[0]
    if K < 3:
        raise ValueError(f"a manifold needs >= 3 nodes to fit, got {K}")
    plan = (
        layout if isinstance(layout, RbfFitPlan)
        else RbfFitPlan.build(layout, smoothing=smoothing)
    )
    mu, X, basis, anchor, ev_ratio = _frame(
        centroids, n_components=n_components, whitener=whitener, layer=layer,
        neutral_mean=neutral_mean, whitened_gram=whitened_gram,
        whitened_rows=whitened_rows, basis_override=basis_override,
    )
    mean = (anchor @ basis.T) @ basis           # P_basis(anchor) (D,)
    # Project the μ-centered roster once; the anchor-relative RBF targets
    # differ only by one reduced offset.
    mu_coords = X @ basis.T
    coords = mu_coords + ((mu - anchor) @ basis.T)
    fit = plan.fit(coords, smoothing=smoothing)
    sub = LayerSubspace(
        # Each layer's persistent payload owns these tiny tensors: safetensors
        # rejects aliases across keys.
        mean=mean, basis=basis, node_params=plan.node_params.clone(),
        rbf_weights=fit.weights, poly_coeffs=fit.poly,
        coord_offset=plan.coord_offset.clone(),
        coord_scale=plan.coord_scale.clone(),
    )
    return LayerFit(
        subspace=sub, mu_coords=mu_coords, explained_variance=ev_ratio,
        rbf=None if smoothing is None else fit,
    )


__all__ = [
    "DEFAULT_N_COMPONENTS",
    "LayerFit",
    "LayerSubspace",
    "fit_affine_subspace",
    "fit_curved_subspace",
    "pca_basis",
    "subspace_share",
]
