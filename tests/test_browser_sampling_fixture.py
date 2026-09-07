import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from drowse.core.generation import (
    GenerationConfig,
    _effective_topk,
    _sampler_candidates,
)


FIXTURE = (
    Path(__file__).parents[1]
    / "browser-runtime"
    / "fixtures"
    / "sampling-parity-v1.json"
)


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text())["cases"])
def test_browser_sampling_fixture_matches_python_source_of_truth(
    case: dict[str, Any],
) -> None:
    logits = torch.tensor([case["logits"]], dtype=torch.float32)
    config = GenerationConfig(
        temperature=case["temperature"],
        top_p=case["topP"],
        top_k=case["topK"] or None,
    )
    token_ids, probabilities = _sampler_candidates(
        logits,
        config,
        _effective_topk(config, logits.shape[-1]),
    )
    support = probabilities > 0
    support_ids = token_ids[support].tolist()
    support_probabilities = probabilities[support].tolist()
    expected = case["expected"]
    assert support_ids == expected["supportTokenIds"]
    assert support_probabilities == pytest.approx(
        expected["supportProbabilities"], abs=1e-7,
    )

    entropy = -sum(probability * math.log(probability) for probability in support_probabilities)
    assert entropy == pytest.approx(expected["entropyNats"], abs=1e-7)
    assert math.exp(entropy) == pytest.approx(expected["perplexity"], abs=1e-7)
    for sample in expected["samples"]:
        cumulative = 0.0
        sampled_id = support_ids[-1]
        sampled_probability = support_probabilities[-1]
        for token_id, probability in zip(
            support_ids, support_probabilities, strict=True,
        ):
            cumulative += probability
            if sample["uniform"] < cumulative:
                sampled_id = token_id
                sampled_probability = probability
                break
        assert sampled_id == sample["tokenId"]
        assert math.log(sampled_probability) == pytest.approx(
            sample["logprob"], abs=1e-7,
        )
