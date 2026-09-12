from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types
from collections.abc import Iterator, Mapping
from typing import Any, Self

import numpy as np
import pytest
import torch


_MODULE_PATH = Path(__file__).parents[1] / "browser-runtime" / "mlc_q4_torch.py"
_SPEC = importlib.util.spec_from_file_location("mlc_q4_torch", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_dequantize_q4_group_layout() -> None:
    values = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
        ]
        * 2,
        dtype=np.uint32,
    )
    packed = np.zeros(4, dtype=np.uint32)
    for offset in range(8):
        packed |= values[offset::8] << np.uint32(offset * 4)
    decoded = _MODULE.dequantize_q4(
        packed.reshape(1, 4),
        np.array([[0.5]], dtype=np.float16),
    )
    np.testing.assert_allclose(decoded[0], (values.astype(np.float32) - 7) * 0.5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_metal_linear_row_batches_preserve_exact_fma_results(dtype: torch.dtype):
    generator = torch.Generator().manual_seed(812)
    inputs = torch.randn(2, 40, 64, generator=generator, dtype=dtype).to("mps")
    weight = torch.randn(96, 64, generator=generator, dtype=dtype).to("mps")
    reference = torch.empty(2, 40, 96, dtype=dtype, device="mps")
    shader = _MODULE._mlc_metal_linear()
    kernel = shader.exact_f16_linear if dtype == torch.float16 else shader.exact_f32_linear
    kernel(reference, inputs, weight, 64, 96)
    linear = _MODULE.MlcLinear(torch.nn.Parameter(weight, requires_grad=False), None)
    actual = linear(inputs)
    torch.mps.synchronize()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_tensor_cache_reads_exact_shard_extent(tmp_path: Path) -> None:
    values = np.array([1.5, -2.0], dtype=np.float16)
    (tmp_path / "params_shard_0.bin").write_bytes(values.tobytes())
    (tmp_path / "tensor-cache.json").write_text(
        json.dumps(
            {
                "metadata": {"ParamSize": 1},
                "records": [
                    {
                        "dataPath": "params_shard_0.bin",
                        "nbytes": values.nbytes,
                        "records": [
                            {
                                "name": "value",
                                "shape": [2],
                                "dtype": "float16",
                                "nbytes": values.nbytes,
                                "byteOffset": 0,
                            }
                        ],
                    }
                ],
            }
        )
    )
    cache = _MODULE.MlcTensorCache(tmp_path)
    np.testing.assert_array_equal(cache.tensor("value"), values)
    with pytest.raises(ValueError, match="metadata has an invalid extent"):
        payload = json.loads((tmp_path / "tensor-cache.json").read_text())
        payload["records"][0]["records"][0]["nbytes"] += 2
        (tmp_path / "tensor-cache.json").write_text(json.dumps(payload))
        _MODULE.MlcTensorCache(tmp_path).tensor("value")


def test_dequantize_rejects_incompatible_scale_shape() -> None:
    with pytest.raises(ValueError, match="incompatible shapes"):
        _MODULE.dequantize_q4(
            np.zeros((2, 4), dtype=np.uint32),
            np.zeros((2, 2), dtype=np.float16),
        )


@pytest.mark.parametrize("tied", [True, False])
def test_qwen35_exact_weights_preserve_gates_and_hybrid_state(tied: bool):
    config = {
        "tie_word_embeddings": tied, "hidden_size": 4, "head_dim": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "intermediate_size": 3, "num_hidden_layers": 2,
        "full_attention_interval": 2,
    }

    class Cache:
        def q4(self, name: str):
            rows = 12 if name.endswith("c_attn") else 6 if name.endswith("gate_up_proj") else 4
            offset = 100 if name == "lm_head" else 0
            return torch.arange(rows * 4, dtype=torch.float16).reshape(rows, 4) + offset

        def tensor(self, name: str):
            shape = (8, 1, 4) if name.endswith("conv1d_weight") else (2,)
            return np.full(shape, 0.12345678, dtype=np.float32)

    cache = Cache()
    state = _MODULE.qwen35_state_dict(cache, config, np.float16)
    full = "model.layers.1.self_attn"
    qkv = cache.q4(f"{full}.c_attn")
    torch.testing.assert_close(state[f"{full}.q_proj.weight"], qkv[:8])
    torch.testing.assert_close(state[f"{full}.k_proj.weight"], qkv[8:10])
    torch.testing.assert_close(state[f"{full}.v_proj.weight"], qkv[10:])
    linear = "model.layers.0.linear_attn"
    for projection in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
        torch.testing.assert_close(state[f"{linear}.{projection}.weight"], cache.q4(f"{linear}.{projection}"))
    assert state[f"{linear}.A_log"].dtype == torch.float32
    assert state[f"{linear}.dt_bias"].dtype == torch.float32
    assert state[f"{linear}.conv1d.weight"].shape == (8, 1, 4)
    assert state[f"{linear}.norm.weight"].dtype == torch.float16
    assert state[f"{linear}.norm.weight"][0].item() < 1
    assert (state["lm_head.weight"] is state["model.embed_tokens.weight"]) is tied
    for layer in range(2):
        mlp = f"model.layers.{layer}.mlp"
        torch.testing.assert_close(state[f"{mlp}.gate_proj.weight"], cache.q4(f"{mlp}.gate_up_proj")[:3])
        torch.testing.assert_close(state[f"{mlp}.up_proj.weight"], cache.q4(f"{mlp}.gate_up_proj")[3:])


@pytest.mark.parametrize("invalid", ["c_attn", "gate_up_proj"])
def test_qwen35_exact_weights_reject_incompatible_fused_layout(invalid: str):
    class Cache:
        def q4(self, name: str):
            rows = 1 if name.endswith(invalid) else 6
            return torch.zeros(rows, 4)

        def tensor(self, name: str):
            return np.zeros(4, dtype=np.float32)

    with pytest.raises(ValueError, match="fused MLP shape|gated QKV shape"):
        _MODULE.qwen35_state_dict(Cache(), {
            "tie_word_embeddings": True, "hidden_size": 4, "head_dim": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "intermediate_size": 3, "num_hidden_layers": 1,
            "full_attention_interval": 1,
        }, np.float16)


def test_special_token_config_preserves_scalar_and_multi_eos_ids() -> None:
    assert _MODULE.special_token_config(
        {"bos_token_id": None, "eos_token_id": [248044, 248046], "pad_token_id": 248044}
    ) == {"bos_token_id": None, "eos_token_id": [248044, 248046], "pad_token_id": 248044}
    with pytest.raises(ValueError, match="bos_token_id"):
        _MODULE.special_token_config({"eos_token_id": 2, "pad_token_id": 2})
    assert _MODULE.special_token_config(
        {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 2,
        }
    ) == {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 2}
    assert _MODULE.special_token_config(
        {
            "bos_token_id": 2,
            "eos_token_id": [1, 106],
            "pad_token_id": 0,
        }
    ) == {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0}
    with pytest.raises(ValueError, match="eos_token_id"):
        _MODULE.special_token_config(
            {
                "bos_token_id": 1,
                "eos_token_id": [],
                "pad_token_id": 0,
            }
        )


def test_build_transformers_model_binds_exact_special_token_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConfig:
        def __init__(self, **values: Any) -> None:
            self.__dict__.update(values)

    class FakeModel:
        def __init__(self, config: FakeConfig) -> None:
            self.config = config
            self.lm_head = types.SimpleNamespace()

        def load_state_dict(
            self,
            state_dict: Mapping[str, torch.Tensor],
            strict: bool = False,
            assign: bool = False,
        ) -> tuple[list[str], list[str]]:
            assert state_dict == {}
            assert strict is False
            assert assign is True
            return [], []

        def tie_weights(self) -> None:
            pass

        def modules(self) -> Iterator[Self]:
            yield self

        def named_buffers(
            self,
            *,
            recurse: bool = True,
        ) -> Iterator[tuple[str, torch.Tensor]]:
            assert isinstance(recurse, bool)
            return iter(())

        def named_children(self) -> Iterator[tuple[str, Self]]:
            return iter(())

        def requires_grad_(self, requires_grad: bool = True) -> Self:
            assert requires_grad is False
            return self

        def to(self, device: str | torch.device) -> Self:
            assert str(device) == "cpu"
            return self

        def eval(self) -> Self:
            return self

    fake_transformers = types.SimpleNamespace(
        Gemma3ForCausalLM=FakeModel,
        Gemma3TextConfig=FakeConfig,
        LlamaConfig=FakeConfig,
        LlamaForCausalLM=FakeModel,
        Qwen3Config=FakeConfig,
        Qwen3ForCausalLM=FakeModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        _MODULE,
        "transformers_state_dict",
        lambda _directory: (
            {},
            {
                "architecture": "qwen3",
                "quantization": "q4f32_1",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "rms_norm_eps": 1e-6,
                "context_window_size": 128,
                "rope_theta": 10_000,
                "attention_bias": False,
                "bos_token_id": 2,
                "eos_token_id": [1, 7],
                "pad_token_id": 0,
            },
        ),
    )
    model = _MODULE.build_transformers_model(Path("unused"))
    assert model.config.bos_token_id == 2
    assert model.config.eos_token_id == [1, 7]
    assert model.config.pad_token_id == 0
    assert model.lm_head.output_dtype == torch.float32


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("scaling", [None, {"rope_type": "linear", "factor": 8.0}])
@pytest.mark.parametrize("quantization", ["q4f32_1", "q0f32"])
def test_gemma_reference_preserves_local_geometry_and_materializes_rope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool, scaling: dict[str, object] | None, quantization: str):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    common = {
        "vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 4, "rms_norm_eps": 1e-6, "attention_bias": False,
        "hidden_activation": "gelu_pytorch_tanh", "query_pre_attn_scalar": 4,
        "bos_token_id": 2, "eos_token_id": [1, 7], "pad_token_id": 0,
    }
    global_rope = {"rope_type": "default", "rope_theta": 1_000_000, **(scaling or {})}
    reference = Gemma3ForCausalLM(Gemma3TextConfig(
        **common, max_position_embeddings=128, sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={"full_attention": global_rope, "sliding_attention": {"rope_type": "default", "rope_theta": 10_000}},
    )).eval()
    reference.config._attn_implementation = "eager"
    state = reference.state_dict()

    class Cache:
        def __init__(self, directory: Path):
            assert directory == tmp_path

        def q4(self, name: str):
            name = name.removeprefix("language_model.")
            if name.endswith(".gate_up_proj"):
                return torch.cat([state[name.replace("gate_up_proj", "gate_proj") + ".weight"], state[name.replace("gate_up_proj", "up_proj") + ".weight"]])
            return state[name + ".weight"]

        def tensor(self, name: str):
            name = name.removeprefix("language_model.")
            if name.endswith(".gate_up_proj.weight"):
                return torch.cat([state[name.replace("gate_up_proj", "gate_proj")], state[name.replace("gate_up_proj", "up_proj")]]).numpy()
            value = state[name]
            return (value + 1 if "norm.weight" in name else value).numpy()

    geometry = {"rope_local_base_freq": 10_000, "sliding_window_pattern": 2, "rope_scaling": scaling}
    text_config = {**common, "context_window_size": 128, "position_embedding_base": 1_000_000}
    text_config.update({"sliding_window": 4, "kwargs": geometry} if legacy else {"sliding_window_size": 4, **geometry})
    (tmp_path / "mlc-chat-config.json").write_text(json.dumps({
        "model_type": "gemma3_text", "quantization": quantization, **common,
        "model_config": {"text_config": text_config, "vocab_size": 32, "context_window_size": 128, "sliding_window_size": -1},
    }))
    monkeypatch.setattr(_MODULE, "MlcTensorCache", Cache)
    model = _MODULE.build_transformers_model(tmp_path)
    assert model.config.sliding_window == 4
    assert model.config.layer_types == reference.config.layer_types
    assert model.config.rope_parameters == reference.config.rope_parameters
    actual_buffers = dict(model.model.rotary_emb.named_buffers())
    for name, expected in reference.model.rotary_emb.named_buffers():
        torch.testing.assert_close(actual_buffers[name], expected, rtol=0, atol=0)
    input_ids = torch.tensor([[2, 3, 4, 5, 6, 8, 9, 10]])
    with torch.inference_mode():
        torch.testing.assert_close(model(input_ids).logits, reference(input_ids).logits, rtol=1e-5, atol=1e-6)


def test_browser_residual_correction_preserves_gradient_and_replaces_value() -> None:
    layers = [torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False)]
    for layer in layers:
        layer.weight.data.copy_(torch.eye(2))
    corrector = _MODULE.BrowserResidualCorrector(layers)
    references = torch.tensor(
        [
            [[3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0]],
        ]
    )
    corrector.set(references)
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    result = layers[1](layers[0](values))
    torch.testing.assert_close(result, references[1].unsqueeze(0))
    result.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
    corrector.close()


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_tied_mlc_head_accumulates_and_returns_fp32(device: str) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    weight = torch.nn.Parameter(torch.ones((1, 3), dtype=torch.float16, device=device))
    values = torch.tensor([[2048, 1, -2048]], dtype=torch.float16, device=device)
    head = _MODULE.MlcLinear(weight, None, output_dtype=torch.float32)
    result = head(values)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result.cpu(), torch.tensor([[1.0]]), rtol=0, atol=0)


def test_browser_residual_correction_rejects_partial_or_nonfinite_rows() -> None:
    corrector = _MODULE.BrowserResidualCorrector([torch.nn.Linear(2, 2, bias=False)])
    with pytest.raises(ValueError, match="layer shape"):
        corrector.set(torch.zeros(2, 1, 2))
    with pytest.raises(ValueError, match="non-finite"):
        corrector.set(torch.tensor([[[float("nan"), 0.0]]]))
    corrector.close()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_mlc_linear_runs_exact_metal_kernel_for_supported_dtype(
    dtype: torch.dtype,
) -> None:
    weight = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0], [-3.0, 4.0]], dtype=dtype, device="mps"),
        requires_grad=False,
    )
    values = torch.tensor([[[5.0, 6.0]]], dtype=dtype, device="mps")

    result = _MODULE.MlcLinear(weight, None)(values)

    torch.testing.assert_close(
        result.cpu(),
        torch.tensor([[[17.0, 9.0]]], dtype=dtype),
        rtol=0,
        atol=0,
    )
