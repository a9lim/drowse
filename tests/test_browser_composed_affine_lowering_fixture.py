from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from drowse.core.capture import project_profile
from drowse.core.hooks import _lower_affine_subspaces
from drowse.core.mahalanobis import LayerWhitener
from drowse.core.manifold import synthesize_subspace


FIXTURE = json.loads(
    (
        Path(__file__).parents[1]
        / "browser-runtime"
        / "fixtures"
        / "affine-composed-lowering-v1.json"
    ).read_text()
)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_browser_composed_affine_lowering_matches_python(case: dict[str, object]) -> None:
    whitener, means = _whitener()
    pushes = []
    for term in cast(list[dict[str, Any]], case["terms"]):
        base = _profile(term["base"], folded=term["projection"] is not None)
        projection = term["projection"]
        if projection is not None:
            onto = _profile(projection[1:], folded=True)
            base = project_profile(base, onto, projection[0], whitener=whitener)
        basis = {}
        coordinates = {}
        for layer, direction in base.items():
            norm = float(direction.norm())
            if norm <= 1e-9:
                continue
            basis[layer] = (direction / norm).reshape(1, -1)
            coordinates[layer] = torch.tensor([norm], dtype=torch.float32)
        pushes.append((basis, coordinates, term["coefficient"]))

    synthesized = synthesize_subspace(pushes, [], means, whitener=whitener)
    lowered = _lower_affine_subspaces(synthesized)
    for expected in cast(list[dict[str, Any]], case["expectedLayers"]):
        layer = expected["layer"]
        actual = lowered[layer]
        world_target = actual.target @ actual.subspace.basis
        target = float(world_target.norm())
        basis = world_target / target
        delta = actual.eff_along * world_target
        assert basis.tolist() == pytest.approx(expected["basis"], abs=1e-6)
        assert actual.subspace.mean.tolist() == pytest.approx(expected["neutral"], abs=1e-7)
        assert target == pytest.approx(expected["target"], abs=1e-6)
        assert actual.eff_along == pytest.approx(expected["along"], abs=1e-6)
        assert delta.tolist() == pytest.approx(expected["delta"], abs=1e-5)


def _whitener() -> tuple[LayerWhitener, dict[int, torch.Tensor]]:
    centered = {}
    small_inverse = {}
    ridge = {}
    means = {}
    for row in FIXTURE["whiteners"]:
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


def _profile(selector: list[str], *, folded: bool) -> dict[int, torch.Tensor]:
    manifold_name, label = selector
    manifold = FIXTURE["manifolds"][manifold_name]
    label_index = manifold["labels"].index(label)
    result = {}
    for row in manifold["layers"]:
        basis = torch.tensor(row["basis"], dtype=torch.float32)
        magnitude = (
            row["mahalanobisShare"]
            if folded
            else row["nodeCoordinates"][label_index]
        )
        result[row["layer"]] = basis * magnitude
    return result
