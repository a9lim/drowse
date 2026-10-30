"""Sparse decomposition of a direction against a lens dictionary.

A *dictionary* is a set of atoms in residual space that is never
materialized: a vocabulary-sized ``[V, d]`` matrix would be gigabytes on a
real model, so a dictionary exposes only the three operations the pursuit
needs — correlate a residual with every atom, form one atom, and the atom
norms.  The pursuit is the paper's gradient-pursuit decomposition: greedily
pick the atom that best correlates with the residual, re-solve the
coefficients over the selected atoms, repeat.

One dictionary exists today, :class:`TokensOut` — ``W_U J_l``, the output
side's token directions.  The interface is the hook for the others the lens
model implies: ``tokens-in`` (input-side footprints ``K_l ẽ_v``) and
``neurons`` (``W_down`` columns, *signed* because a neuron has two poles).
A signed dictionary selects by ``|score|`` and solves its coefficients
unconstrained; a nonnegative one keeps the NNLS step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch


class LensDictionary(Protocol):
    """Atoms ``D[i] ∈ R^d`` addressed by index, never materialized."""

    #: The dictionary's name on the wire (``tokens-out``, …).
    kind: str
    #: Whether coefficients are constrained nonnegative (tokens) or signed.
    nonneg: bool
    device: torch.device

    def scores(self, residual: torch.Tensor) -> torch.Tensor:
        """``D @ residual`` over every atom (fp32 ``[V]``)."""
        ...

    def atom(self, index: int) -> torch.Tensor:
        """One atom ``D[index]`` (fp32 ``[d]``)."""
        ...

    def norms(self) -> torch.Tensor:
        """``‖D[i]‖`` for every atom (fp32 ``[V]``)."""
        ...


def token_atom_norms(
    jacobian: torch.Tensor, unembed: torch.Tensor, *, chunk: int = 8192,
) -> torch.Tensor:
    """Per-atom norms ``‖W_U[v] @ J‖`` for the whole vocabulary, computed in
    chunks so the ``[vocab, d]`` dictionary never exists in full."""
    norms = torch.empty(unembed.shape[0], device=unembed.device)
    for start in range(0, unembed.shape[0], chunk):
        block = unembed[start : start + chunk].float() @ jacobian
        norms[start : start + chunk] = block.norm(dim=-1)
    return norms


@dataclass
class TokensOut:
    """``W_U J_l``: the tokens a layer-``l`` residual is disposed to say.

    ``atom_norms`` may carry a cached ``‖W_U[v] J_l‖`` (the lens caches them
    per layer); otherwise they are computed in chunks on first use.
    """

    jacobian: torch.Tensor
    unembed: torch.Tensor
    atom_norms: torch.Tensor | None = None
    kind: str = field(default="tokens-out", init=False)
    nonneg: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        self.jacobian = self.jacobian.to(
            device=self.unembed.device, dtype=torch.float32,
        )

    @property
    def device(self) -> torch.device:
        return self.unembed.device

    def scores(self, residual: torch.Tensor) -> torch.Tensor:
        return (
            self.unembed @ (self.jacobian @ residual).to(self.unembed.dtype)
        ).float()

    def atom(self, index: int) -> torch.Tensor:
        return self.unembed[index].float() @ self.jacobian

    def norms(self) -> torch.Tensor:
        if self.atom_norms is None:
            self.atom_norms = token_atom_norms(self.jacobian, self.unembed)
        return self.atom_norms.to(device=self.device, dtype=torch.float32)


@dataclass(frozen=True)
class SparseDecomposition:
    """One layer's sparse split of a direction against a dictionary.

    ``share`` is the fraction of the direction's variance the selected atoms
    carry; ``atoms`` is ``[(index, coeff), ...]`` by descending magnitude —
    vocabulary ids for a token dictionary.
    """

    layer: int
    share: float
    atoms: list[tuple[int, float]]
    dictionary: str

    def __repr__(self) -> str:
        return (
            f"SparseDecomposition({self.dictionary}, layer={self.layer}, "
            f"share={self.share:.3f}, k={len(self.atoms)})"
        )


def sparse_decompose(
    target: torch.Tensor,
    dictionary: LensDictionary,
    *,
    layer: int,
    k: int = 16,
    nnls_iters: int = 200,
) -> SparseDecomposition:
    """Greedy sparse pursuit of ``target`` against ``dictionary``.

    Atom scores are normalized by the atom norms (raw inner products would
    bias selection toward large-norm atoms), and only the ≤k selected atoms
    are formed.  After each selection the coefficients re-solve over the
    selected atoms — nonnegative least squares for a nonnegative dictionary
    (projected gradient with the exact Lipschitz step; the problem is tiny),
    ordinary least squares for a signed one.  Selection stops early when no
    atom correlates with the residual in an admissible direction.
    """
    device = dictionary.device
    t = target.detach().to(device=device, dtype=torch.float32)
    t_norm_sq = float(t.pow(2).sum())
    if t_norm_sq == 0.0:
        return SparseDecomposition(layer, 0.0, [], dictionary.kind)
    norms = dictionary.norms().clamp(min=1e-12)

    selected: list[int] = []
    rows: list[torch.Tensor] = []
    coeffs = torch.zeros(0, device=device)
    residual = t.clone()
    for _ in range(k):
        scores = dictionary.scores(residual) / norms
        if not dictionary.nonneg:
            scores = scores.abs()
        if selected:
            scores[torch.tensor(selected, device=device)] = -torch.inf
        best = int(scores.argmax())
        if float(scores[best]) <= 0.0:
            break
        selected.append(best)
        rows.append(dictionary.atom(best))
        A = torch.stack(rows)  # [s, d]
        gram = A @ A.T
        b = A @ t
        if dictionary.nonneg:
            c = torch.cat([coeffs, coeffs.new_zeros(1)])
            solved = _try_unconstrained_nonnegative(gram, b, device=device)
            if solved is None:
                # CPU hop: eigvalsh is unimplemented on MPS, and the gram is ≤ k×k.
                lipschitz = float(
                    torch.linalg.eigvalsh(gram.cpu())[-1].clamp(min=1e-12)
                )
                for _ in range(nnls_iters):
                    c = torch.clamp(c - (gram @ c - b) / lipschitz, min=0.0)
            else:
                c = solved
        else:
            c = _solve_tiny(gram, b).to(device=device)
        coeffs = c
        residual = t - A.T @ coeffs

    share = max(0.0, 1.0 - float(residual.pow(2).sum()) / t_norm_sq)
    pairs = sorted(
        (
            (idx, float(cf)) for idx, cf in zip(selected, coeffs)
            if float(cf) != 0.0 and (float(cf) > 0.0 or not dictionary.nonneg)
        ),
        key=lambda p: -abs(p[1]),
    )
    return SparseDecomposition(layer, share, pairs, dictionary.kind)


def _solve_tiny(gram: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Ridge-stabilized normal-equation solve on CPU (the system is ≤ k×k)."""
    g_cpu = gram.detach().to("cpu", torch.float32)
    eye = torch.eye(g_cpu.shape[0], dtype=torch.float32)
    return torch.linalg.solve(g_cpu + 1e-7 * eye, b.detach().to("cpu", torch.float32))


def _try_unconstrained_nonnegative(
    gram: torch.Tensor,
    b: torch.Tensor,
    *,
    device: torch.device,
    tol: float = 1e-7,
) -> torch.Tensor | None:
    """Exact tiny least-squares solve when its coefficients are already >= 0.

    The greedy step solves ``min ||A^T c - t||²`` over the selected atoms.
    If the unconstrained normal-equation solution is nonnegative, it is also
    the NNLS optimum, so the projected-gradient loop is pure overhead.  If
    any coefficient is negative or the tiny system is singular, the caller
    keeps the projected-gradient fallback.
    """
    try:
        sol = _solve_tiny(gram, b)
    except RuntimeError:
        return None
    if not bool(torch.isfinite(sol).all()):
        return None
    if bool((sol < -tol).any()):
        return None
    return sol.clamp(min=0.0).to(device=device, dtype=torch.float32)
