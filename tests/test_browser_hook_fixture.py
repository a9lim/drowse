from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


_FIXTURE = Path(__file__).parents[1] / "browser-runtime" / "fixtures" / "post-block-rank-one-v2.json"


@pytest.mark.parametrize("case_index", range(len(json.loads(_FIXTURE.read_text())["cases"])))
def test_browser_rank_one_hook_fixture_matches_python_fp32(case_index: int) -> None:
    fixture = json.loads(_FIXTURE.read_text())
    case = fixture["cases"][case_index]
    residual = torch.tensor(case["residual"], dtype=torch.float32)
    basis = torch.tensor(case["basis"], dtype=torch.float32)
    neutral = torch.tensor(case["neutral"], dtype=torch.float32)
    from drowse.core.manifold import CustomDomain, LayerSubspace, subspace_inject

    coordinate = torch.dot(residual - neutral, basis)
    result, _ = subspace_inject(
        residual,
        LayerSubspace.affine(neutral, basis.reshape(1, -1)),
        CustomDomain(1),
        torch.tensor([case["target"]], dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
        case["along"],
        0.0,
        kappa=torch.tensor([case["collapse"]], dtype=torch.float32),
    )
    probe = torch.dot(
        result - torch.tensor(case["probeNeutral"], dtype=torch.float32),
        torch.tensor(case["probeBasis"], dtype=torch.float32),
    )

    assert coordinate.item() == pytest.approx(case["expectedCoordinate"], abs=1e-7)
    assert result.tolist() == pytest.approx(case["expectedResidual"], abs=1e-7)
    assert probe.item() == pytest.approx(case["expectedProbe"], abs=1e-7)
