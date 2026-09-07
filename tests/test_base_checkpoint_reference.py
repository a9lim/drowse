import importlib.util
import sys
from pathlib import Path
from typing import cast

import pytest
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
