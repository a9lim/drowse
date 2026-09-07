from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from drowse.core.hooks import _lower_affine_subspaces
from drowse.core.manifold import synthesize_subspace
from tests._whitener import isotropic_whitener


FIXTURE = json.loads(
    (
        Path(__file__).parents[1]
        / "browser-runtime"
        / "fixtures"
        / "affine-rank-one-lowering-v1.json"
    ).read_text()
)


def test_browser_affine_lowering_fixture_matches_python() -> None:
    layers = FIXTURE["layers"]
    basis = {
        row["layer"]: torch.tensor([row["basis"]], dtype=torch.float32)
        for row in layers
    }
    coordinates = {
        row["layer"]: torch.tensor(
            [row["nodeCoordinates"][row["selectedNode"]]], dtype=torch.float32
        )
        for row in layers
    }
    means = {
        row["layer"]: torch.tensor(row["neutral"], dtype=torch.float32)
        for row in layers
    }
    whitener = isotropic_whitener(means, len(layers[0]["basis"]))
    synthesized = synthesize_subspace(
        [(basis, coordinates, FIXTURE["coefficient"])],
        [],
        means,
        whitener=whitener,
    )
    lowered = _lower_affine_subspaces(synthesized)

    for row in layers:
        layer = lowered[row["layer"]]
        assert layer.target.tolist() == pytest.approx([row["expectedTarget"]])
        assert layer.eff_along == pytest.approx(row["expectedAlong"])
        assert layer.kappa.tolist() == [0.0]
