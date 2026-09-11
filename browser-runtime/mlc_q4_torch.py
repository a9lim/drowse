from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


_DTYPES = {
    "float16": np.dtype("<f2"),
    "float32": np.dtype("<f4"),
    "int32": np.dtype("<i4"),
    "uint32": np.dtype("<u4"),
}


class MlcTensorCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        payload = json.loads((directory / "tensor-cache.json").read_text())
        records: dict[str, tuple[Path, dict[str, Any]]] = {}
        for shard in payload.get("records", []):
            path = directory / shard["dataPath"]
            if not path.is_file() or path.stat().st_size != shard["nbytes"]:
                raise ValueError(f"MLC tensor shard is missing or truncated: {path}")
            for record in shard.get("records", []):
                name = record.get("name")
                if not isinstance(name, str) or name in records:
                    raise ValueError("MLC tensor cache repeats or omits a tensor name")
                records[name] = (path, record)
        if len(records) != payload.get("metadata", {}).get("ParamSize"):
            raise ValueError("MLC tensor cache parameter count does not match its metadata")
        self.records = records

    def tensor(self, name: str) -> np.ndarray:
        try:
            path, record = self.records[name]
            dtype = _DTYPES[record["dtype"]]
            shape = tuple(int(value) for value in record["shape"])
            encoding = record.get("format", "raw")
            byte_offset = int(record["byteOffset"])
            byte_length = int(record["nbytes"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"MLC tensor metadata is invalid for {name}") from error
        if encoding not in {"f32-to-bf16", "raw"}:
            raise ValueError(f"MLC tensor metadata has an unsupported encoding for {name}")
        encoded_dtype = np.dtype("<u2") if encoding == "f32-to-bf16" and dtype == np.dtype("<f4") else dtype
        expected = int(np.prod(shape, dtype=np.int64)) * encoded_dtype.itemsize
        if expected != byte_length or byte_offset < 0 or byte_offset + byte_length > path.stat().st_size:
            raise ValueError(f"MLC tensor metadata has an invalid extent for {name}")
        value = np.memmap(path, dtype=encoded_dtype, mode="r", offset=byte_offset, shape=shape)
        if encoded_dtype != dtype:
            return (np.asarray(value, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        return value

    def q4(self, name: str) -> torch.Tensor:
        weight = self.tensor(f"{name}.q_weight")
        scale = self.tensor(f"{name}.q_scale")
        return torch.from_numpy(dequantize_q4(weight, scale))


class BrowserResidualCorrector:
    def __init__(self, layers: list[torch.nn.Module]) -> None:
        if not layers:
            raise ValueError("browser residual correction requires model layers")
        self.layers = layers
        self.references: torch.Tensor | None = None
        self.handles = [
            layer.register_forward_hook(self._capture(layer_index)) for layer_index, layer in enumerate(layers)
        ]

    def set(self, references: torch.Tensor) -> None:
        if references.ndim != 3 or references.shape[0] != len(self.layers):
            raise ValueError("browser residual correction has an incompatible layer shape")
        if references.shape[1] == 0 or references.shape[2] == 0:
            raise ValueError("browser residual correction cannot be empty")
        if not torch.isfinite(references).all():
            raise ValueError("browser residual correction contains non-finite values")
        parameter = next(self.layers[0].parameters())
        self.references = references.to(device=parameter.device, dtype=parameter.dtype)

    def clear(self) -> None:
        self.references = None

    def close(self) -> None:
        self.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _capture(self, layer_index: int):
        def correct(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            if self.references is None:
                return None
            actual = output[0] if isinstance(output, tuple) else output
            if not isinstance(actual, torch.Tensor) or actual.shape[0] != 1:
                raise ValueError("browser residual correction requires a batch of one prompt")
            expected = self.references[layer_index]
            if actual.shape[1:] != expected.shape:
                raise ValueError(
                    f"browser residual correction shape changed at layer {layer_index}: "
                    f"{tuple(actual.shape[1:])} != {tuple(expected.shape)}"
                )
            corrected = actual + (expected.unsqueeze(0) - actual.detach())
            if isinstance(output, tuple):
                return (corrected, *output[1:])
            return corrected

        return correct


class MlcRMSNorm(torch.nn.Module):
    def __init__(self, weight: torch.nn.Parameter, epsilon: float) -> None:
        super().__init__()
        self.weight = weight
        self.variance_epsilon = epsilon

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_f32 = hidden_states.to(torch.float32)
        variance = hidden_f32.square().mean(dim=-1, keepdim=True)
        normalized = hidden_f32 * torch.rsqrt(variance + self.variance_epsilon)
        return (normalized * self.weight.to(torch.float32)).to(input_dtype)


class MlcLinear(torch.nn.Module):
    def __init__(self, weight: torch.nn.Parameter, bias: torch.nn.Parameter | None) -> None:
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.device.type == "mps" and self.bias is None:
            if inputs.dtype == torch.float16 and self.weight.dtype == torch.float16:
                return _MlcF16Linear.apply(inputs, self.weight)
            if inputs.dtype == torch.float32 and self.weight.dtype == torch.float32:
                return _MlcF32Linear.apply(inputs, self.weight)
            raise ValueError("MLC q4 linear input and weight dtypes do not match")
        output = torch.nn.functional.linear(
            inputs.to(torch.float32),
            self.weight.to(torch.float32),
            None if self.bias is None else self.bias.to(torch.float32),
        )
        return output.to(inputs.dtype)


class _MlcF16Linear(torch.autograd.Function):
    @staticmethod
    def forward(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if inputs.dtype != torch.float16 or weight.dtype != torch.float16:
            raise ValueError("MLC q4 linear expects fp16 inputs and weights")
        inputs = inputs.contiguous()
        weight = weight.contiguous()
        output = torch.empty((*inputs.shape[:-1], weight.shape[0]), dtype=torch.float16, device="mps")
        _mlc_metal_linear().exact_f16_linear(
            output,
            inputs,
            weight,
            int(inputs.shape[-1]),
            int(weight.shape[0]),
        )
        return output

    @staticmethod
    def setup_context(context: Any, inputs: tuple[torch.Tensor, torch.Tensor], output: torch.Tensor) -> None:
        del output
        _values, weight = inputs
        context.save_for_backward(weight)

    @staticmethod
    def backward(context: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (weight,) = context.saved_tensors
        grad_input = torch.matmul(
            grad_output.to(torch.float32),
            weight.to(torch.float32),
        ).to(grad_output.dtype)
        return grad_input, None


class _MlcF32Linear(torch.autograd.Function):
    @staticmethod
    def forward(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if inputs.dtype != torch.float32 or weight.dtype != torch.float32:
            raise ValueError("MLC q4 linear expects fp32 inputs and weights")
        inputs = inputs.contiguous()
        weight = weight.contiguous()
        output = torch.empty(
            (*inputs.shape[:-1], weight.shape[0]),
            dtype=torch.float32,
            device="mps",
        )
        _mlc_metal_linear().exact_f32_linear(
            output,
            inputs,
            weight,
            int(inputs.shape[-1]),
            int(weight.shape[0]),
        )
        return output

    @staticmethod
    def setup_context(
        context: Any,
        inputs: tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        del output
        _values, weight = inputs
        context.save_for_backward(weight)

    @staticmethod
    def backward(
        context: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        (weight,) = context.saved_tensors
        grad_input = torch.matmul(grad_output, weight)
        return grad_input, None


_METAL_LINEAR = None


def _mlc_metal_linear():
    global _METAL_LINEAR
    if _METAL_LINEAR is None:
        _METAL_LINEAR = torch.mps.compile_shader(
            """
#include <metal_stdlib>
using namespace metal;

kernel void exact_f16_linear(
    device half* output [[buffer(0)]],
    device const half* inputs [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    constant uint& input_width [[buffer(3)]],
    constant uint& output_width [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  const uint row = index / output_width;
  const uint column = index - row * output_width;
  half value = half(0.0);
  for (uint inner = 0; inner < input_width; ++inner) {
    value = fma(inputs[row * input_width + inner],
                weight[column * input_width + inner], value);
  }
  output[index] = value;
}

kernel void exact_f32_linear(
    device float* output [[buffer(0)]],
    device const float* inputs [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    constant uint& input_width [[buffer(3)]],
    constant uint& output_width [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  const uint row = index / output_width;
  const uint column = index - row * output_width;
  float value = 0.0f;
  for (uint inner = 0; inner < input_width; ++inner) {
    value = fma(inputs[row * input_width + inner],
                weight[column * input_width + inner], value);
  }
  output[index] = value;
}
"""
        )
    return _METAL_LINEAR


def dequantize_q4(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    if weight.dtype != np.dtype("uint32") or scale.dtype not in {
        np.dtype("float16"),
        np.dtype("float32"),
    }:
        raise ValueError("MLC q4 expects uint32 weights and float16 or float32 scales")
    if weight.ndim < 1 or scale.shape[:-1] != weight.shape[:-1] or weight.shape[-1] * 8 != scale.shape[-1] * 32:
        raise ValueError("MLC q4 packed weights and scales have incompatible shapes")
    unpacked = np.empty((*weight.shape[:-1], weight.shape[-1] * 8), dtype=scale.dtype)
    words = np.asarray(weight, dtype=np.uint32)
    for offset in range(8):
        unpacked[..., offset::8] = ((words >> np.uint32(offset * 4)) & np.uint32(15)).astype(scale.dtype)
    grouped = unpacked.reshape(*scale.shape, 32)
    grouped -= np.array(7, dtype=scale.dtype)
    grouped *= np.asarray(scale)[..., None]
    return np.ascontiguousarray(unpacked)


def transformers_state_dict(directory: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    chat = json.loads((directory / "mlc-chat-config.json").read_text())
    architecture = chat.get("model_type")
    raw_config = chat.get("model_config")
    quantization = chat.get("quantization")
    if architecture not in {"gemma3_text", "llama", "qwen3", "qwen3_5"} or not isinstance(raw_config, dict):
        raise ValueError("only the pinned Gemma 3, Llama, Qwen3, and Qwen3.5 q4 browser architectures are supported")
    if quantization not in {"q4f16_1", "q4f32_1"} and not (
        quantization == "q0f32" and architecture == "gemma3_text"
    ):
        raise ValueError("only q4f16_1, q4f32_1, and Gemma q0f32 browser weights are supported")
    if architecture == "gemma3_text":
        text_config = raw_config.get("text_config")
        if not isinstance(text_config, dict):
            raise ValueError("Gemma 3 browser configuration has no text_config")
        config = {
            **text_config,
            "vocab_size": raw_config.get("vocab_size"),
            "context_window_size": raw_config.get("context_window_size", text_config.get("context_window_size")),
            "sliding_window_size": text_config.get("sliding_window_size", text_config.get("sliding_window")),
        }
        source_prefix = "language_model."
    else:
        config = raw_config
        source_prefix = ""
    cache = MlcTensorCache(directory)
    direct_dtype = np.float16 if quantization == "q4f16_1" else np.float32
    matrix = cache.q4 if quantization != "q0f32" else lambda name: direct(cache, f"{name}.weight", np.float32)
    if architecture == "qwen3_5":
        return qwen35_state_dict(cache, config, direct_dtype), {
            "architecture": architecture, "quantization": quantization,
            **config, **special_token_config(chat),
        }
    state: dict[str, torch.Tensor] = {}
    embedding = matrix(f"{source_prefix}model.embed_tokens")
    state["model.embed_tokens.weight"] = embedding
    state["lm_head.weight"] = embedding
    hidden_size = int(config["hidden_size"])
    head_dim = int(config["head_dim"])
    q_width = int(config["num_attention_heads"]) * head_dim
    kv_width = int(config["num_key_value_heads"]) * head_dim
    intermediate = int(config["intermediate_size"])
    layer_count = int(config["num_hidden_layers"])
    for layer in range(layer_count):
        source = f"{source_prefix}model.layers.{layer}"
        target = f"model.layers.{layer}"
        state[f"{target}.input_layernorm.weight"] = direct(cache, f"{source}.input_layernorm.weight", direct_dtype)
        state[f"{target}.post_attention_layernorm.weight"] = direct(
            cache, f"{source}.post_attention_layernorm.weight", direct_dtype
        )
        if architecture == "gemma3_text":
            state[f"{target}.pre_feedforward_layernorm.weight"] = direct(
                cache, f"{source}.pre_feedforward_layernorm.weight", direct_dtype
            )
            state[f"{target}.post_feedforward_layernorm.weight"] = direct(
                cache, f"{source}.post_feedforward_layernorm.weight", direct_dtype
            )
            for projection, width in (("q", q_width), ("k", kv_width), ("v", kv_width)):
                value = matrix(f"{source}.self_attn.{projection}_proj")
                if tuple(value.shape) != (width, hidden_size):
                    raise ValueError(f"MLC Gemma {projection.upper()} projection shape is invalid at layer {layer}")
                state[f"{target}.self_attn.{projection}_proj.weight"] = value
        else:
            fused_attention = "qkv_proj" if architecture == "llama" else "c_attn"
            qkv = matrix(f"{source}.self_attn.{fused_attention}")
            if tuple(qkv.shape) != (q_width + 2 * kv_width, hidden_size):
                raise ValueError(f"MLC fused QKV shape is invalid at layer {layer}")
            state[f"{target}.self_attn.q_proj.weight"] = qkv[:q_width]
            state[f"{target}.self_attn.k_proj.weight"] = qkv[q_width : q_width + kv_width]
            state[f"{target}.self_attn.v_proj.weight"] = qkv[q_width + kv_width :]
        state[f"{target}.self_attn.o_proj.weight"] = matrix(f"{source}.self_attn.o_proj")
        gate_up = matrix(f"{source}.mlp.gate_up_proj")
        if tuple(gate_up.shape) != (2 * intermediate, hidden_size):
            raise ValueError(f"MLC fused gate/up shape is invalid at layer {layer}")
        state[f"{target}.mlp.gate_proj.weight"] = gate_up[:intermediate]
        state[f"{target}.mlp.up_proj.weight"] = gate_up[intermediate:]
        state[f"{target}.mlp.down_proj.weight"] = matrix(f"{source}.mlp.down_proj")
        if architecture in {"gemma3_text", "qwen3"}:
            state[f"{target}.self_attn.q_norm.weight"] = direct(
                cache, f"{source}.self_attn.q_norm.weight", direct_dtype
            )
            state[f"{target}.self_attn.k_norm.weight"] = direct(
                cache, f"{source}.self_attn.k_norm.weight", direct_dtype
            )
    state["model.norm.weight"] = direct(cache, f"{source_prefix}model.norm.weight", direct_dtype)
    return state, {
        "architecture": architecture,
        "quantization": quantization,
        **config,
        **special_token_config(chat),
    }


def qwen35_state_dict(cache, config, direct_dtype):
    state = {"model.embed_tokens.weight": cache.q4("model.embed_tokens")}
    state["lm_head.weight"] = (state["model.embed_tokens.weight"]
                               if config["tie_word_embeddings"] else cache.q4("lm_head"))
    hidden = int(config["hidden_size"])
    head_dim = int(config["head_dim"])
    q_width = 2 * int(config["num_attention_heads"]) * head_dim
    kv_width = int(config["num_key_value_heads"]) * head_dim
    intermediate = int(config["intermediate_size"])
    for layer in range(int(config["num_hidden_layers"])):
        name = f"model.layers.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            key = f"{name}.{norm}.weight"
            state[key] = direct(cache, key, direct_dtype)
        gate_up = cache.q4(f"{name}.mlp.gate_up_proj")
        if tuple(gate_up.shape) != (2 * intermediate, hidden):
            raise ValueError(f"MLC Qwen3.5 fused MLP shape is invalid at layer {layer}")
        state[f"{name}.mlp.gate_proj.weight"] = gate_up[:intermediate]
        state[f"{name}.mlp.up_proj.weight"] = gate_up[intermediate:]
        state[f"{name}.mlp.down_proj.weight"] = cache.q4(f"{name}.mlp.down_proj")
        if (layer + 1) % int(config["full_attention_interval"]) == 0:
            attention = f"{name}.self_attn"
            qkv = cache.q4(f"{attention}.c_attn")
            if tuple(qkv.shape) != (q_width + 2 * kv_width, hidden):
                raise ValueError(f"MLC Qwen3.5 gated QKV shape is invalid at layer {layer}")
            state[f"{attention}.q_proj.weight"] = qkv[:q_width]
            state[f"{attention}.k_proj.weight"] = qkv[q_width:q_width + kv_width]
            state[f"{attention}.v_proj.weight"] = qkv[q_width + kv_width:]
            state[f"{attention}.o_proj.weight"] = cache.q4(f"{attention}.o_proj")
            for norm in ("q_norm", "k_norm"):
                key = f"{attention}.{norm}.weight"
                state[key] = direct(cache, key, direct_dtype)
        else:
            attention = f"{name}.linear_attn"
            for projection in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
                state[f"{attention}.{projection}.weight"] = cache.q4(f"{attention}.{projection}")
            for key in ("A_log", "dt_bias"):
                state[f"{attention}.{key}"] = direct(cache, f"{attention}.{key}", np.float32)
            state[f"{attention}.conv1d.weight"] = direct(cache, f"{attention}.conv1d_weight", direct_dtype)
            state[f"{attention}.norm.weight"] = direct(cache, f"{attention}.norm.weight", direct_dtype)
    state["model.norm.weight"] = direct(cache, "model.norm.weight", direct_dtype)
    return state


def build_transformers_model(directory: Path, device: str = "cpu") -> torch.nn.Module:
    from transformers import (
        Gemma3ForCausalLM,
        Gemma3TextConfig,
        LlamaConfig,
        LlamaForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    state, config = transformers_state_dict(directory)
    model_dtype = torch.float16 if config["quantization"] == "q4f16_1" else torch.float32
    common = {
        "vocab_size": int(config["vocab_size"]),
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "num_attention_heads": int(config["num_attention_heads"]),
        "num_key_value_heads": int(config["num_key_value_heads"]),
        "head_dim": int(config["head_dim"]),
        "rms_norm_eps": float(config["rms_norm_eps"]),
        "tie_word_embeddings": True,
        "torch_dtype": model_dtype,
        "bos_token_id": config["bos_token_id"],
        "eos_token_id": config["eos_token_id"],
        "pad_token_id": config["pad_token_id"],
    }
    if config["architecture"] == "llama":
        model_config = LlamaConfig(
            **common,
            max_position_embeddings=int(config["context_window_size"]),
            rope_theta=float(config["position_embedding_base"]),
        )
        model_type = LlamaForCausalLM
    elif config["architecture"] == "qwen3":
        model_config = Qwen3Config(
            **common,
            max_position_embeddings=int(config["context_window_size"]),
            rope_theta=float(config["rope_theta"]),
            attention_bias=bool(config["attention_bias"]),
        )
        model_type = Qwen3ForCausalLM
    else:
        kwargs = config.get("kwargs")
        if not isinstance(kwargs, dict):
            kwargs = {}
        layer_types = kwargs.get("layer_types")
        if not isinstance(layer_types, list) or len(layer_types) != common["num_hidden_layers"]:
            pattern = int(config.get("sliding_window_pattern", kwargs.get("sliding_window_pattern", kwargs.get("_sliding_window_pattern", 6))))
            layer_types = [
                "full_attention" if (layer + 1) % pattern == 0 else "sliding_attention"
                for layer in range(common["num_hidden_layers"])
            ]
        rope_theta = float(config["position_embedding_base"])
        local_rope_theta = float(config.get("rope_local_base_freq", kwargs.get("rope_local_base_freq", 10_000)))
        global_rope = {"rope_type": "default", "rope_theta": rope_theta}
        global_rope.update(config.get("rope_scaling") or kwargs.get("rope_scaling") or {})
        model_config = Gemma3TextConfig(
            **common,
            max_position_embeddings=int(config["context_window_size"]),
            hidden_activation=str(config["hidden_activation"]),
            attention_bias=bool(config["attention_bias"]),
            query_pre_attn_scalar=int(config["query_pre_attn_scalar"]),
            sliding_window=int(config["sliding_window_size"]),
            layer_types=layer_types,
            rope_parameters={
                "full_attention": global_rope,
                "sliding_attention": {"rope_type": "default", "rope_theta": local_rope_theta},
            },
        )
        model_type = Gemma3ForCausalLM
    model_config._attn_implementation = "eager"
    with torch.device("meta"):
        model = model_type(model_config)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    allowed_missing = {key for key in missing if key.endswith("rotary_emb.inv_freq")}
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(
            f"MLC q4 state does not close the Transformers model: missing={missing}, unexpected={unexpected}"
        )
    materialize_rotary_embeddings(model)
    for module in model.modules():
        for name, buffer in tuple(module.named_buffers(recurse=False)):
            if not buffer.is_meta:
                continue
            if name == "embed_scale":
                module._buffers[name] = torch.tensor(
                    common["hidden_size"] ** 0.5,
                    dtype=model_dtype,
                )
    remaining_meta = [name for name, value in model.named_buffers() if value.is_meta]
    if remaining_meta:
        raise ValueError(f"MLC q4 model has unmaterialized buffers: {remaining_meta}")
    model.tie_weights()
    replace_rms_norms(model)
    replace_linears(model)
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    model._drowse_mlc_architecture = config["architecture"]
    model._drowse_mlc_quantization = config["quantization"]
    return model


def materialize_rotary_embeddings(module: torch.nn.Module) -> None:
    for name, child in tuple(module.named_children()):
        if child.__class__.__name__.endswith("RotaryEmbedding"):
            with torch.device("cpu"):
                setattr(module, name, type(child)(child.config))
        else:
            materialize_rotary_embeddings(child)


def replace_rms_norms(module: torch.nn.Module) -> None:
    for name, child in tuple(module.named_children()):
        if child.__class__.__name__.endswith("RMSNorm"):
            epsilon = getattr(child, "variance_epsilon", getattr(child, "eps", None))
            weight = getattr(child, "weight", None)
            if not isinstance(epsilon, float) or not isinstance(weight, torch.nn.Parameter):
                raise ValueError(f"cannot reproduce MLC RMSNorm semantics for {name}")
            setattr(module, name, MlcRMSNorm(weight, epsilon))
        else:
            replace_rms_norms(child)


def replace_linears(module: torch.nn.Module) -> None:
    for name, child in tuple(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, MlcLinear(child.weight, child.bias))
        else:
            replace_linears(child)


def direct(cache: MlcTensorCache, name: str, dtype: np.dtype[Any]) -> torch.Tensor:
    value = np.array(cache.tensor(name), copy=True)
    if not np.isfinite(value).all():
        raise ValueError(f"MLC tensor contains non-finite values: {name}")
    return torch.from_numpy(value.astype(dtype, copy=False))


def special_token_config(chat: dict[str, Any]) -> dict[str, int | list[int] | None]:
    result: dict[str, int | list[int] | None] = {}
    for field in ("bos_token_id", "pad_token_id"):
        value = chat.get(field)
        if field == "bos_token_id" and field in chat and value is None:
            result[field] = None
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"MLC chat configuration has no valid {field}")
        result[field] = value
    eos = chat.get("eos_token_id")
    if isinstance(eos, int) and not isinstance(eos, bool) and eos >= 0:
        result["eos_token_id"] = eos
    elif (
        isinstance(eos, list)
        and eos
        and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in eos)
    ):
        result["eos_token_id"] = list(eos)
    else:
        raise ValueError("MLC chat configuration has no valid eos_token_id")
    return result


def runtime_source_fingerprint(directory: Path) -> str:
    digest = hashlib.sha256()
    for name in ("mlc-chat-config.json", "tensor-cache.json"):
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()
