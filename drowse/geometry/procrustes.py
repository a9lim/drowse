"""Cross-model activation alignment: per-layer Procrustes / least squares.

Pure tensor math with no persistence.  :func:`fit_alignment` fits a compact
affine map ``A_L(x) = M_L x + b_L`` per shared layer from two models' matched
neutral activations (orthogonal Procrustes at matched hidden dim, minimum-norm
least squares otherwise); :func:`alignment_quality` scores it as per-layer R².
The map is applied by ``geometry.transfer.transfer_manifold_subspaces``,
persisted by :mod:`drowse.store.alignment`, and orchestrated end to end by
:mod:`drowse.fit.transfer`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch

from drowse.errors import DrowseError


@dataclass(frozen=True)
class LayerAlignment:
    """Compact affine source->target activation map for one layer.

    ``linear(x) = left @ (right @ x)`` stores a rank-``r`` map in
    ``O(r * (D_src + D_tgt))`` rather than materializing ``D_tgt * D_src``.
    ``offset`` is applied only to activation *points*.  Directions intentionally
    use the linear component alone.
    """

    left: torch.Tensor       # (D_tgt, r)
    right: torch.Tensor      # (r, D_src)
    offset: torch.Tensor     # (D_tgt,)

    def __post_init__(self) -> None:
        if self.left.ndim != 2 or self.right.ndim != 2 or self.offset.ndim != 1:
            raise ValueError("alignment factors must be 2-D, 2-D, and 1-D")
        if self.left.shape[1] != self.right.shape[0]:
            raise ValueError("alignment factor ranks do not match")
        if self.left.shape[0] != self.offset.shape[0]:
            raise ValueError("alignment offset does not match target dimension")
        if self.left.shape[1] == 0:
            raise ValueError("alignment rank must be positive")
        device = self.left.device
        object.__setattr__(
            self, "left", self.left.to(device=device, dtype=torch.float32).contiguous(),
        )
        object.__setattr__(
            self, "right", self.right.to(device=device, dtype=torch.float32).contiguous(),
        )
        object.__setattr__(
            self, "offset", self.offset.to(device=device, dtype=torch.float32).contiguous(),
        )

    @property
    def source_dim(self) -> int:
        return int(self.right.shape[1])

    @property
    def target_dim(self) -> int:
        return int(self.left.shape[0])

    @property
    def rank(self) -> int:
        return int(self.left.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.target_dim, self.source_dim)

    @property
    def device(self) -> torch.device:
        return self.left.device

    def apply_vector(self, value: torch.Tensor) -> torch.Tensor:
        value = value.to(device=self.device, dtype=torch.float32)
        return self.left @ (self.right @ value)

    def apply_vectors(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the linear component to row-major ``(..., D_src)`` vectors."""
        value = value.to(device=self.device, dtype=torch.float32)
        return (value @ self.right.transpose(0, 1)) @ self.left.transpose(0, 1)

    def apply_points(self, value: torch.Tensor) -> torch.Tensor:
        value = value.to(device=self.device, dtype=torch.float32)
        return self.apply_vectors(value) + self.offset

    def to_dense(self) -> torch.Tensor:
        return self.left @ self.right


AlignmentMap = Mapping[int, LayerAlignment]


class AlignmentError(ValueError, DrowseError):
    """Raised when an alignment can't be fit (insufficient shared layers)."""

    def user_message(self) -> tuple[int, str]:
        return (422, str(self) or self.__class__.__name__)


def fit_alignment(
    src_acts: dict[int, torch.Tensor],
    tgt_acts: dict[int, torch.Tensor],
    *,
    min_shared_layers: int = 10,
    requested_layers: Iterable[int] | None = None,
    available_shared_layers: Iterable[int] | None = None,
) -> dict[int, LayerAlignment]:
    """Per-layer compact affine mapping ``A_L : ℝ^D_src → ℝ^D_tgt``.

    For each layer the two models share, fits ``A_L(x) = M_L x + b_L`` such
    that matched neutral activation *points* map source→target.  Directions
    use ``M_L`` alone, so translations never contaminate profile transfer.

    * **Same hidden dim:** evidence-supported orthogonal Procrustes, factored
      through economy QR row-space bases and a small ``<=N`` SVD.  No arbitrary
      unobserved ``D-N`` complement is invented.
    * **Different hidden dim:** minimum-norm least squares from the thin SVD of
      ``X_src``.  The result stays as rank-sized factors instead of a dense
      ``D_tgt × D_src`` matrix.

    The translation is ``b_L = mean_tgt - M_L mean_src``.  Returns compact
    :class:`LayerAlignment` objects over the shared-layer set; tensors are fp32
    on CPU.  Raises :class:`AlignmentError` when fewer than
    ``min_shared_layers`` layers are common to both models — partial
    alignment under that threshold tends to produce noisy transfers.
    """
    materialized_shared = set(src_acts.keys()) & set(tgt_acts.keys())
    available = (
        materialized_shared if available_shared_layers is None
        else {int(layer) for layer in available_shared_layers}
    )
    if len(available) < min_shared_layers:
        raise AlignmentError(
            f"alignment requires >= {min_shared_layers} shared layers; "
            f"src={sorted(src_acts.keys())}, tgt={sorted(tgt_acts.keys())}"
        )
    shared = sorted(
        materialized_shared if requested_layers is None else (
            materialized_shared & {int(layer) for layer in requested_layers}
        )
    )
    if not shared:
        raise AlignmentError("alignment has no requested layers shared by both models")

    out: dict[int, LayerAlignment] = {}
    for layer in shared:
        X_src = src_acts[layer].to(torch.float32)
        X_tgt = tgt_acts[layer].to(torch.float32)
        if X_src.shape[0] != X_tgt.shape[0]:
            raise AlignmentError(
                f"layer {layer}: src has N={X_src.shape[0]} prompts, "
                f"tgt has N={X_tgt.shape[0]}; expected matched."
            )

        # Center both sides — Procrustes is translation-invariant only
        # under shared origin.
        src_mean = X_src.mean(dim=0)
        tgt_mean = X_tgt.mean(dim=0)
        X_src_c = X_src - src_mean
        X_tgt_c = X_tgt - tgt_mean

        D_src = X_src_c.shape[1]
        D_tgt = X_tgt_c.shape[1]

        if D_src == D_tgt:
            # Low-rank orthogonal Procrustes.  Factor the two row spaces first,
            # then SVD only their <=N cross-core; the unobserved D-N complement
            # has no evidence and is deliberately not invented/persisted.
            Qs, Rs = torch.linalg.qr(X_src_c.transpose(0, 1), mode="reduced")
            Qt, Rt = torch.linalg.qr(X_tgt_c.transpose(0, 1), mode="reduced")
            U, S, Vh = torch.linalg.svd(Rt @ Rs.transpose(0, 1), full_matrices=False)
            tol = torch.finfo(S.dtype).eps * max(D_src, X_src_c.shape[0]) * float(S.max())
            rank = int((S > tol).sum())
            if rank == 0:
                raise AlignmentError(f"layer {layer}: centered alignment has rank 0")
            left = (Qt @ U[:, :rank]).contiguous()
            right = (Vh[:rank] @ Qs.transpose(0, 1)).contiguous()
        else:
            # Minimum-norm least squares in factored form:
            # M = X_tgt^T U S^-1 V^T for X_src = U S V^T.
            U, S, Vh = torch.linalg.svd(X_src_c, full_matrices=False)
            tol = torch.finfo(S.dtype).eps * max(X_src_c.shape) * float(S.max())
            rank = int((S > tol).sum())
            if rank == 0:
                raise AlignmentError(f"layer {layer}: centered alignment has rank 0")
            left = (
                X_tgt_c.transpose(0, 1) @ U[:, :rank]
            ).div(S[:rank].unsqueeze(0)).contiguous()
            right = Vh[:rank].contiguous()

        linear_src_mean = left @ (right @ src_mean)
        offset = (tgt_mean - linear_src_mean).contiguous()
        out[layer] = LayerAlignment(left, right, offset)
    return out


def alignment_quality(
    M: AlignmentMap,
    src_acts: dict[int, torch.Tensor],
    tgt_acts: dict[int, torch.Tensor],
) -> dict[int, float]:
    """Per-layer R² between the transferred src activations and tgt.

    For each layer ``L`` shared by ``M``, computes
    ``1 - ||X_tgt - X_src @ M_L^T||^2 / ||X_tgt||^2`` on centered data.
    Values near 1.0 mean the linear map captures the cross-model
    geometry; values near 0.0 mean transferred probes will be noisy.

    The median over shared layers becomes a transferred manifold sidecar's
    ``transfer_quality_estimate``; the full per-layer dict rides the
    alignment cache and the CLI JSON output.
    """
    out: dict[int, float] = {}
    for layer, M_L in M.items():
        X_src = src_acts[layer].to(torch.float32)
        X_tgt = tgt_acts[layer].to(torch.float32)
        X_src_c = X_src - X_src.mean(dim=0, keepdim=True)
        X_tgt_c = X_tgt - X_tgt.mean(dim=0, keepdim=True)
        # Translation cancels under centering; score the linear geometry.
        X_pred = M_L.apply_vectors(X_src_c)
        residual = (X_tgt_c - X_pred).pow(2).sum().item()
        total = X_tgt_c.pow(2).sum().item()
        if total <= 1e-12:
            out[layer] = 0.0
        else:
            out[layer] = float(1.0 - residual / total)
    return out
