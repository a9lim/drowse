import importlib.util
import sys
from pathlib import Path
from typing import cast

import pytest
import numpy as np
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig


RUNTIME = Path(__file__).parents[1] / "browser-runtime"
for name, filename in [("mlc_q4_torch", "mlc_q4_torch.py"),
                       ("base_checkpoint_reference", "compare-base-checkpoint.py")]:
    spec = importlib.util.spec_from_file_location(name, RUNTIME / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
REFERENCE = sys.modules["base_checkpoint_reference"]


@pytest.mark.parametrize("quantization,dtype", [("q4f16_1", torch.float16), ("q4f32_1", torch.float32)])
def test_converted_reference_retains_source_geometry(tmp_path: Path, quantization: str, dtype: torch.dtype):
    config = Gemma3TextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        sliding_window=7, max_position_embeddings=128,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000},
            "full_attention": {"rope_type": "default", "rope_theta": 1000000},
        },
        bos_token_id=2, eos_token_id=1, pad_token_id=0,
    )
    config.save_pretrained(tmp_path)
    original = cast(torch.nn.Module, Gemma3ForCausalLM(config)).to(dtype=dtype).eval()
    state = {key: value.clone() for key, value in original.state_dict().items()}
    for key in state:
        if key.endswith("norm.weight"):
            state[key] += 1
    model = REFERENCE.load_source_architecture_reference(tmp_path, state, quantization)
    assert model.config.sliding_window == 7
    assert model.config.max_position_embeddings == 128
    assert model.config.layer_types == ["sliding_attention", "full_attention"]
    assert model.config.rope_parameters == config.rope_parameters
    assert next(model.parameters()).dtype == dtype
    rotary = model.model.rotary_emb
    local = rotary.sliding_attention_inv_freq
    global_ = rotary.full_attention_inv_freq
    assert not torch.equal(local, global_)
    assert config.layer_types is not None
    for layer_type in config.layer_types:
        expected, _ = rotary.compute_default_rope_parameters(config, layer_type=layer_type)
        torch.testing.assert_close(getattr(rotary, f"{layer_type}_inv_freq"), expected)
    torch.testing.assert_close(model.model.embed_tokens.embed_scale,
                               torch.tensor(config.hidden_size ** 0.5, dtype=dtype))
    with torch.inference_mode():
        assert torch.isfinite(model(torch.tensor([[2, 5, 6]])).logits).all()


def test_converted_reference_requires_exact_state_closure(tmp_path: Path):
    Gemma3TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=1, head_dim=8).save_pretrained(tmp_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        REFERENCE.load_source_architecture_reference(tmp_path, {}, "q4f32_1")


def test_converted_reference_rejects_unsupported_precision(tmp_path: Path):
    Gemma3TextConfig().save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="only Gemma 3 q4"):
        REFERENCE.load_source_architecture_reference(tmp_path, {}, "q0f32")


def test_probability_comparison_preserves_small_differences_on_large_logits():
    reference = np.array([1500.0, 1500.001, 1499.5], dtype=np.float64)
    actual = reference + np.array([0.0004, -0.0004, 0.0])
    result = REFERENCE.probability_metrics(reference, actual)
    assert result["totalVariation"] > 1e-4
    assert result["logitCount"] == 3
    assert REFERENCE.probability_metrics(reference, reference + 256)["totalVariation"] < 1e-12


@pytest.mark.parametrize("actual", [[1, 2], [[1, 2, 3]], [1, float("nan"), 3], [1, float("inf"), 3]])
def test_probability_comparison_rejects_incomplete_or_nonfinite_vocab(actual: object):
    with pytest.raises(ValueError):
        REFERENCE.probability_metrics([1, 2, 3], actual)


def test_probability_comparison_includes_padded_vocabulary_mass():
    result = REFERENCE.probability_metrics([0, 0, 8], [0, 0, -8])
    assert result["totalVariation"] > .99


@pytest.mark.parametrize("threshold", ["nan", "inf", "0", "-0.1", "1.01"])
def test_probability_threshold_rejects_invalid_values(threshold: str):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        REFERENCE.probability_threshold(threshold)


def test_probability_threshold_retains_strict_accuracy_bound():
    assert REFERENCE.probability_threshold("0.0001") == 1e-4


def test_float64_attention_preserves_sub_float32_score_differences():
    query = torch.tensor([[[[1000., 1.]]]], dtype=torch.float64)
    key = torch.tensor([[[[65., .00001], [65., -.00001]]]], dtype=torch.float64)
    value = torch.tensor([[[[1., 0.], [0., 1.]]]], dtype=torch.float64)
    output, probabilities = REFERENCE.float64_attention(torch.nn.Identity().eval(), query, key, value, None, 1.)
    expected = torch.tensor([.00001, -.00001], dtype=torch.float64).softmax(-1)
    assert probabilities.dtype == torch.float64
    torch.testing.assert_close(probabilities[0, 0, 0], expected, atol=1e-11, rtol=0)
    torch.testing.assert_close(output[0, 0, 0], expected, atol=1e-11, rtol=0)
    assert probabilities[0, 0, 0, 0] > .5


def test_float64_reference_preserves_causality_and_cached_decoding():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(19)
    config = GPTNeoXConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=2,
                          max_position_embeddings=128, attention_dropout=0., hidden_dropout=0.)
    model = GPTNeoXForCausalLM(config).double().eval()
    REFERENCE.enable_float64_reference(model)
    ids = torch.tensor([[2, 5, 9, 11]])
    with torch.inference_mode():
        full = model(ids, use_cache=False).logits
        prefix = model(ids[:, :2], use_cache=True)
        assert prefix.past_key_values is not None
        tail = model(ids[:, 2:], past_key_values=prefix.past_key_values, use_cache=True).logits
        alternative = model(torch.tensor([[2, 5, 20, 31]]), use_cache=False).logits
    torch.testing.assert_close(full[:, :2], alternative[:, :2], atol=1e-13, rtol=0)
    torch.testing.assert_close(full[:, :2], prefix.logits, atol=1e-13, rtol=0)
    torch.testing.assert_close(full[:, 2:], tail, atol=1e-13, rtol=0)
    rotary = model.gpt_neox.rotary_emb
    cos, sin = rotary(None, torch.tensor([[0, 1, 63]]))
    assert cos.dtype == sin.dtype == rotary.inv_freq.dtype == torch.float64
    assert cos[0, 2, 0].item() == pytest.approx(np.cos(63.), abs=1e-15)


def test_nonrotary_key_bias_cancels_but_rotary_bias_does_not():
    torch.manual_seed(23)
    query = torch.randn(2, 7, 8, dtype=torch.float64)
    key = torch.randn_like(query)
    bias = torch.randn(2, 1, 8, dtype=torch.float64) * 64
    phase = torch.arange(7, dtype=torch.float64)[None, :, None]
    def rotate(x: torch.Tensor):
        return torch.cat((x[..., :2] * phase.cos() - x[..., 2:4] * phase.sin(),
                          x[..., 2:4] * phase.cos() + x[..., :2] * phase.sin(), x[..., 4:]), -1)
    q = rotate(query)
    original = (q @ rotate(key + bias).transpose(-1, -2)).softmax(-1)
    centered_bias = bias.clone()
    centered_bias[..., 4:] = 0
    centered = (q @ rotate(key + centered_bias).transpose(-1, -2)).softmax(-1)
    wrongly_centered = (q @ rotate(key).transpose(-1, -2)).softmax(-1)
    torch.testing.assert_close(original, centered, atol=1e-13, rtol=0)
    assert (original - wrongly_centered).abs().max() > .1
