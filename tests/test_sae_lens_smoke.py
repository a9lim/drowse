"""Opt-in end-to-end smoke test for the SAE pipeline.

Gated on DROWSE_TEST_SAELENS=1 because it requires ``sae_lens`` installed
and downloads a real SAE release (hundreds of MB). Not on CI by default.
"""
from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.gpu_gemma,
    pytest.mark.skipif(
        os.environ.get("DROWSE_TEST_SAELENS") != "1",
        reason="opt in via DROWSE_TEST_SAELENS=1",
    ),
]


def test_end_to_end_sae_extraction_and_generate():
    import torch

    from drowse import DrowseSession

    has_gpu = torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if not has_gpu:
        pytest.skip("requires CUDA or MPS")

    session = DrowseSession.from_pretrained(
        "google/gemma-2-2b-it", device="auto", probes=[],
    )

    # Small, canonical GemmaScope release — one SAE per covered layer.
    name, profile = session.extract(
        "happy.sad",
        sae="gemma-scope-2b-pt-res-canonical",
    )
    assert ":sae-" in name
    assert len(profile) > 0

    vanilla = session.generate("Describe a rainy afternoon.").first.text
    with session.steering(f"0.3 {name}"):
        steered = session.generate("Describe a rainy afternoon.").first.text
    assert vanilla != steered, "SAE-steered output should differ from vanilla"
