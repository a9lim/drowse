import json
from pathlib import Path

import pytest
import torch

from drowse.core.manifold import eval_rbf, fit_rbf_smoothed, prepare_rbf_fit_plan
from drowse.core.mahalanobis import LayerWhitener
from drowse.core.manifold import CustomDomain
from drowse.core.topology import (
    SpectralDiagnostics,
    derive_spectral_coords,
    select_topology,
)


FIXTURE = (
    Path(__file__).parents[1]
    / "browser-runtime"
    / "fixtures"
    / "topology-rbf-parity-v1.json"
)


def test_browser_topology_rbf_fixture_matches_python_source_of_truth() -> None:
    fixture = json.loads(FIXTURE.read_text())
    node_count = fixture["nodeCount"]
    output_dimensions = fixture["outputDimensions"]
    targets = torch.tensor(fixture["targets"], dtype=torch.float32).reshape(
        node_count, output_dimensions,
    )
    centered = targets - targets.mean(dim=0, keepdim=True)
    selection = fixture["selection"]
    coordinates, diagnostics = derive_spectral_coords(
        centered @ centered.transpose(0, 1),
        max_dim=selection["maxDimensions"],
        min_dim=selection["minDimensions"],
    )

    expected = fixture["expected"]
    assert coordinates.shape == (node_count, expected["intrinsicDimensions"])
    assert diagnostics.k_nn == expected["kNN"]
    assert diagnostics.bandwidth == pytest.approx(expected["bandwidth"], abs=1e-6)
    assert diagnostics.gap_magnitude == pytest.approx(
        expected["gapMagnitude"], abs=1e-6,
    )
    assert diagnostics.eigenvalues.tolist() == pytest.approx(
        expected["eigenvalues"], abs=1e-6,
    )
    for pair in expected["pairDistances"]:
        distance = torch.linalg.vector_norm(
            coordinates[pair["left"]] - coordinates[pair["right"]],
        )
        assert float(distance) == pytest.approx(pair["distance"], abs=1e-6)

    plan = prepare_rbf_fit_plan(coordinates, smoothing=selection["smoothing"])
    weights, polynomial, info = fit_rbf_smoothed(
        plan.node_params,
        targets,
        smoothing=selection["smoothing"],
        plan=plan,
    )
    evaluated = eval_rbf(
        plan.node_params,
        weights,
        polynomial,
        plan.node_params,
    )
    assert info["lambda"] == pytest.approx(expected["lambda"], abs=1e-6)
    assert info["edf"] == pytest.approx(
        expected["effectiveDegreesOfFreedom"], abs=1e-5,
    )
    assert evaluated.flatten().tolist() == pytest.approx(
        # Spectral coordinates and the RBF solve each contribute fp32 roundoff.
        expected["evaluatedTargets"], abs=5e-6,
    )

    whitener = LayerWhitener(
        {0: torch.zeros(1, output_dimensions, dtype=torch.float32)},
        {0: torch.ones(1, 1, dtype=torch.float32)},
        {0: 1.0},
        {0: torch.zeros(output_dimensions, dtype=torch.float32)},
    )
    gram = whitener.subspace_gram(0, centered)
    choice = select_topology(
        {0: centered},
        {0: gram},
        gram,
        whitener=whitener,
        max_dim=selection["maxDimensions"],
        smoothing=selection["smoothing"],
        persistence_frac=selection["persistenceFraction"],
    )
    assert choice.winner_name == expected["winnerName"]
    assert choice.fit_mode == expected["fitMode"]
    assert isinstance(choice.domain, CustomDomain)
    assert expected["domainKind"] == "custom"
    assert [
        {
            "name": candidate.name,
            "intrinsicDimensions": candidate.intrinsic_dim,
            "viable": candidate.viable,
        }
        for candidate in choice.candidates
    ] == expected["candidates"]
    assert isinstance(choice.diagnostics, SpectralDiagnostics)
    assert choice.diagnostics.k_nn == expected["kNN"]
    assert choice.diagnostics.bandwidth == pytest.approx(
        expected["bandwidth"], abs=1e-6,
    )
