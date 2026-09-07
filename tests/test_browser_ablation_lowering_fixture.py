from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from drowse.core.hooks import _lower_affine_subspaces
from drowse.core.mahalanobis import LayerWhitener
from drowse.core.manifold import synthesize_subspace


FIXTURE = json.loads(
    (
        Path(__file__).parents[1]
        / "browser-runtime"
        / "fixtures"
        / "affine-ablation-lowering-v1.json"
    ).read_text()
)
COMPOSED = json.loads(
    (
        Path(__file__).parents[1]
        / "browser-runtime"
        / "fixtures"
        / "affine-composed-lowering-v1.json"
    ).read_text()
)


def test_browser_rank_one_ablation_matches_python() -> None:
    whitener, means = _whitener()
    directions = {
        row["layer"]: torch.tensor(row["basis"], dtype=torch.float32)
        * row["mahalanobisShare"]
        for row in COMPOSED["manifolds"]["demo"]["layers"]
    }
    synthesized = synthesize_subspace(
        [],
        [(directions, FIXTURE["coefficient"])],
        means,
        whitener=whitener,
    )
    lowered = _lower_affine_subspaces(synthesized)

    for expected in FIXTURE["layers"]:
        actual = lowered[expected["layer"]]
        assert actual.subspace.basis[0].tolist() == pytest.approx(expected["basis"])
        assert actual.subspace.mean.tolist() == pytest.approx(expected["neutral"])
        assert actual.target.tolist() == pytest.approx([0.0])
        collapse = actual.eff_along * actual.kappa
        assert collapse.tolist() == pytest.approx([FIXTURE["coefficient"]])
        residual = torch.tensor(expected["input"], dtype=torch.float32)
        coordinate = actual.subspace.basis @ (residual - actual.subspace.mean)
        output = residual + actual.eff_along * (
            actual.target - actual.kappa * coordinate
        ) @ actual.subspace.basis
        assert output.tolist() == pytest.approx(expected["expected"])


def _whitener() -> tuple[LayerWhitener, dict[int, torch.Tensor]]:
    centered = {}
    small_inverse = {}
    ridge = {}
    means = {}
    for row in COMPOSED["whiteners"]:
        layer = row["layer"]
        basis = torch.tensor(row["basis"], dtype=torch.float32)
        eigenvalues = torch.tensor(row["eigenvalues"], dtype=torch.float32)
        rank = len(eigenvalues)
        centered[layer] = torch.diag(torch.sqrt(rank * eigenvalues)) @ basis
        small_inverse[layer] = torch.diag(
            1 / (rank * (eigenvalues + row["ridge"]))
        )
        ridge[layer] = row["ridge"]
        means[layer] = torch.tensor(row["mean"], dtype=torch.float32)
    return LayerWhitener(centered, small_inverse, ridge, means), means
