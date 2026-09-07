from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from drowse.core.jlens import JacobianLens
from drowse.core.manifold import (
    BoxAxis,
    BoxDomain,
    LayerSubspace,
    SphereDomain,
    subspace_inject,
)
from drowse.core.sae import LocalSaeBackend
from drowse.core.steering_expr import parse_expr
from drowse.core.triggers import TriggerContext


FIXTURE = (
    Path(__file__).parents[1]
    / "browser-runtime"
    / "fixtures"
    / "structured-program-parity-v1.json"
)


def _curve_subspace(curve: dict[str, Any]) -> LayerSubspace:
    return LayerSubspace(
        mean=torch.tensor(curve["neutral"], dtype=torch.float32),
        basis=torch.tensor(curve["basis"], dtype=torch.float32),
        node_params=torch.tensor(curve["nodeParameters"], dtype=torch.float32),
        rbf_weights=torch.tensor(curve["rbfWeights"], dtype=torch.float32),
        poly_coeffs=torch.tensor(curve["polynomial"], dtype=torch.float32),
        coord_offset=torch.tensor(curve["coordinateOffset"], dtype=torch.float32),
        coord_scale=torch.tensor(curve["coordinateScale"], dtype=torch.float32),
    )


def _curve_domain(curve: dict[str, Any]) -> SphereDomain | BoxDomain:
    if curve["domainKind"] == "sphere":
        return SphereDomain(len(curve["axes"]))
    return BoxDomain([
        BoxAxis(
            name=f"axis_{index}",
            periodic=axis["periodic"],
            period=axis["period"],
            lo=axis["lowerBound"],
            hi=axis["upperBound"],
        )
        for index, axis in enumerate(curve["axes"])
    ])


def _run_curve(
    residual: torch.Tensor,
    curve: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    return subspace_inject(
        residual,
        _curve_subspace(curve),
        _curve_domain(curve),
        torch.tensor(curve["target"], dtype=torch.float32),
        torch.tensor(curve["origin"], dtype=torch.float32),
        float(curve["along"]),
        float(curve["onto"]),
        gn_steps=4,
        origin=torch.tensor(curve["origin"], dtype=torch.float32),
    )


def test_multi_curve_periodic_fixture_matches_python_kernel() -> None:
    fixture = json.loads(FIXTURE.read_text())["multiCurve"]
    residual = torch.tensor(fixture["residual"], dtype=torch.float32)
    feet = []
    for curve in fixture["curves"]:
        residual, foot = _run_curve(residual, curve)
        feet.append(foot.tolist())

    assert residual.tolist() == pytest.approx(
        fixture["expected"]["residual"], abs=3e-6,
    )
    for actual, expected in zip(
        feet, fixture["expected"]["preSlideFeet"], strict=True,
    ):
        assert actual == pytest.approx(expected, abs=3e-6)


def test_sphere_curve_fixture_matches_python_kernel() -> None:
    fixture = json.loads(FIXTURE.read_text())["sphereCurve"]
    residual, foot = _run_curve(
        torch.tensor(fixture["residual"], dtype=torch.float32),
        fixture["curve"],
    )
    assert residual.tolist() == pytest.approx(
        fixture["expected"]["residual"], abs=3e-6,
    )
    assert foot.tolist() == pytest.approx(
        fixture["expected"]["preSlideFoot"], abs=3e-6,
    )


def test_sae_fixture_matches_local_drowse_backend() -> None:
    fixture = json.loads(FIXTURE.read_text())["sae"]
    weights = {
        "W_enc": torch.tensor(fixture["encoderDirection"], dtype=torch.float32).reshape(-1, 1),
        "W_dec": torch.tensor(fixture["decoderDirection"], dtype=torch.float32).reshape(1, -1),
        "b_enc": torch.tensor([fixture["encoderBias"]], dtype=torch.float32),
        "b_dec": torch.tensor(fixture["decoderBias"], dtype=torch.float32),
    }
    backend = LocalSaeBackend(
        release="fixture",
        revision="fixture",
        fingerprint="fixture",
        layers=frozenset({0}),
        model_fingerprint="fixture",
        _loader=lambda: weights,
    )
    activation = backend.encode_layer(
        0, torch.tensor([fixture["residual"]], dtype=torch.float32),
    )[0, 0]
    assert float(activation / fixture["maxAct"]) == pytest.approx(
        fixture["expectedNormalizedActivation"], abs=1e-7,
    )
    assert backend.feature_direction(0, 0).tolist() == pytest.approx(
        fixture["expectedDecoderDirection"], abs=1e-7,
    )


def test_jlens_fixture_matches_python_token_direction() -> None:
    fixture = json.loads(FIXTURE.read_text())["jlens"]
    lens = JacobianLens(
        {0: torch.tensor(fixture["jacobian"], dtype=torch.float32)},
        n_prompts=1,
        d_model=len(fixture["unembeddingRow"]),
    )
    unembedding = torch.tensor([fixture["unembeddingRow"]], dtype=torch.float32)
    assert lens.token_direction(0, unembedding)[0].tolist() == pytest.approx(
        fixture["expectedDirection"], abs=1e-7,
    )


@pytest.mark.parametrize("gate_index", [0, 1])
def test_gate_fixture_matches_python_trigger(gate_index: int) -> None:
    fixture = json.loads(FIXTURE.read_text())["gates"][gate_index]
    steering = parse_expr(fixture["expression"])
    trigger = next(iter(steering.normalized_entries().values()))[1]
    for case in fixture["contexts"]:
        context = TriggerContext(
            is_prefill=case["prefill"],
            thinking=case["thinking"],
            gen_step=case["generatedTokens"],
            probe_scores=case["scores"],
        )
        assert trigger.active(context) is case["active"]
