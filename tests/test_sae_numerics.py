from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from drowse.core.sae import LocalSaeBackend, SaeLensBackend


@pytest.mark.parametrize("size", ["270m", "1b", "4b"])
@pytest.mark.parametrize("variant", ["it", "pt"])
def test_gemma_scope_native_encoding_matches_uncentered_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: str, variant: str,
):
    pytest.importorskip("sae_lens")
    from sae_lens.loading import pretrained_sae_loaders as provider
    from sae_lens.saes.jumprelu_sae import JumpReLUSAE, JumpReLUSAEConfig

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"architecture": "jump_relu", "model_name": f"gemma-3-{size}-{variant}"}))
    monkeypatch.setattr(provider, "hf_hub_download", lambda *args, **kwargs: str(config_path))
    monkeypatch.setattr(provider, "get_safetensors_tensor_shapes", lambda *args: {"w_enc": (3, 4), "w_dec": (4, 3)})
    config = provider.get_gemma_3_config_from_hf(
        f"google/gemma-scope-2-{size}-{variant}", "resid_post/layer_1/width_16k/l0_small", "cpu",
    )
    assert config["apply_b_dec_to_input"] is False
    assert config["normalize_activations"] == "none"
    sae = JumpReLUSAE(JumpReLUSAEConfig(
        d_in=3, d_sae=4, apply_b_dec_to_input=config["apply_b_dec_to_input"],
    ))
    with torch.no_grad():
        sae.W_enc.copy_(torch.tensor([[1., 0., -2., 1.], [0., 1., 1., 0.], [1., -1., 0., .5]]))
        sae.b_enc.copy_(torch.tensor([.5, -1., 2., 0.]))
        sae.threshold.copy_(torch.tensor([.5, 1., 2., .5]))
        sae.W_dec.copy_(torch.arange(12).reshape(4, 3) / 10)
        sae.b_dec.copy_(torch.tensor([50., -20., 10.]))
    backend = SaeLensBackend("test", None, None, frozenset({1}), lambda _: sae)
    h = torch.tensor([[0., 0., 0.], [1., 2., -1.], [-3., .5, 2.]])
    pre = h.numpy() @ sae.W_enc.detach().numpy() + sae.b_enc.detach().numpy()
    expected = np.maximum(pre, 0) * (pre > sae.threshold.detach().numpy())
    np.testing.assert_allclose(backend.encode_layer(1, h).detach().numpy(), expected, atol=1e-6)
    decoded = expected @ sae.W_dec.detach().numpy() + sae.b_dec.detach().numpy()
    np.testing.assert_allclose(backend.decode_layer(1, torch.from_numpy(expected)).detach().numpy(), decoded, atol=1e-6)
    torch.testing.assert_close(backend.feature_direction(1, 2), sae.W_dec[2])
    assert expected[0].tolist() == [0., 0., 0., 0.]


def test_local_sae_keeps_its_centered_convention():
    rng = np.random.default_rng(924)
    arrays = {"W_enc": rng.normal(size=(5, 7)), "W_dec": rng.normal(size=(7, 5)),
              "b_enc": rng.normal(size=7), "b_dec": rng.normal(size=5)}
    weights = {key: torch.tensor(value, dtype=torch.float32) for key, value in arrays.items()}
    backend = LocalSaeBackend("local:test", None, None, frozenset({0}), None, lambda: weights)
    h = rng.normal(size=(4, 5)).astype(np.float32)
    expected = np.maximum((h - arrays["b_dec"]) @ arrays["W_enc"] + arrays["b_enc"], 0)
    actual = backend.encode_layer(0, torch.from_numpy(h))
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(backend.decode_layer(0, actual).numpy(), expected @ arrays["W_dec"] + arrays["b_dec"], atol=2e-6, rtol=1e-5)
