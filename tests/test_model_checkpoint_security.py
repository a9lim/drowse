import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from drowse.model import loading as model_mod
from tests.test_model_loading import _remap_to_layout, _tiny_text_config


@pytest.mark.parametrize("shard", ["../outside.safetensors", "/tmp/outside.safetensors", "C:/weights.safetensors", "sub\\weights.safetensors", "./weights.safetensors", "sub//weights.safetensors", "weights.bin", "", None, 1])
def test_index_rejects_unsafe_shard_names_before_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shard: Any):
    index = tmp_path / SAFE_WEIGHTS_INDEX_NAME
    index.write_text(json.dumps({"weight_map": {"language_model.model.weight": shard}}))
    calls = []

    def cached_file(model: str, name: str, **kwargs: Any):
        calls.append(name)
        if name == SAFE_WEIGHTS_INDEX_NAME:
            return str(index)
        pytest.fail("unsafe index shard reached file resolution")

    monkeypatch.setattr("transformers.utils.hub.cached_file", cached_file)
    with pytest.raises(ValueError, match="shard"):
        model_mod._load_text_from_multimodal("fixture/model", _tiny_text_config(False), torch.float32, "cpu")
    assert calls == [SAFE_WEIGHTS_INDEX_NAME]


def test_local_index_traversal_cannot_read_an_external_checkpoint(tmp_path: Path):
    config = _tiny_text_config(False)
    reference = model_mod.AutoModelForCausalLM.from_config(config)
    save_file(_remap_to_layout(reference.state_dict(), "mistral"), str(tmp_path / "outside.safetensors"))
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps({
        "weight_map": {name: "../outside.safetensors" for name in reference.state_dict()},
    }))
    with pytest.raises(ValueError, match="shard"):
        model_mod._load_text_from_multimodal(str(model_root), config, torch.float32, "cpu")


@pytest.mark.parametrize("kind", ["index", "shard"])
def test_checkpoint_rejects_fifo_without_blocking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation unavailable")
    path = tmp_path / (SAFE_WEIGHTS_INDEX_NAME if kind == "index" else SAFE_WEIGHTS_NAME)
    os.mkfifo(path)
    def cached_file(model: str, name: str, **kwargs: Any):
        return str(path) if name == path.name else None
    monkeypatch.setattr("transformers.utils.hub.cached_file", cached_file)
    with pytest.raises(ValueError, match="regular file"):
        model_mod._load_text_from_multimodal("fixture/model", _tiny_text_config(False), torch.float32, "cpu")


@pytest.mark.parametrize("damage", ["missing", "broadcast"])
def test_partial_or_broadcast_checkpoint_is_not_a_loaded_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str):
    config = _tiny_text_config(False)
    reference = model_mod.AutoModelForCausalLM.from_config(config)
    state = _remap_to_layout(reference.state_dict(), "mistral")
    key = "language_model.model.embed_tokens.weight"
    if damage == "missing":
        del state[key]
    else:
        state[key] = state[key][:1].clone()
    path = tmp_path / SAFE_WEIGHTS_NAME
    save_file(state, str(path))
    monkeypatch.setattr("transformers.utils.hub.cached_file", lambda model, name, **kwargs: str(path) if name == SAFE_WEIGHTS_NAME else None)
    with pytest.raises(ValueError, match="missing|shape"):
        model_mod._load_text_from_multimodal("fixture/model", config, torch.float32, "cpu")


@pytest.mark.parametrize("header", [
    {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 8]}},
    {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [-1, 7]}},
    {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 20]}},
    {"weight": {"dtype": "F32", "shape": [True, 2], "data_offsets": [0, 8]}},
    {"weight": {"dtype": "F32", "shape": [-2], "data_offsets": [0, 8]}},
    {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "b": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
])
def test_manual_header_reader_rejects_malformed_layout(tmp_path: Path, header: dict[str, Any]):
    raw = json.dumps(header).encode()
    path = tmp_path / "bad.safetensors"
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0" * 8)
    with pytest.raises(ValueError, match="safetensors"):
        model_mod._read_safetensors_header(str(path))


def test_manual_header_reader_bounds_length_before_reading(tmp_path: Path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes((2**40).to_bytes(8, "little"))
    with pytest.raises(ValueError, match="header"):
        model_mod._read_safetensors_header(str(path))


def test_tied_checkpoint_alias_and_nonpersistent_buffers_are_optional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = _tiny_text_config(True)
    original_factory = model_mod.AutoModelForCausalLM.from_config
    reference = original_factory(config)
    state = _remap_to_layout(reference.state_dict(), "gemma")
    del state["language_model.lm_head.weight"]
    weights = tmp_path / SAFE_WEIGHTS_NAME
    save_file(state, str(weights))

    def with_runtime_buffer(*args: Any, **kwargs: Any):
        model = original_factory(*args, **kwargs)
        model.register_buffer("runtime_cache", torch.ones(3), persistent=False)
        return model

    monkeypatch.setattr(model_mod.AutoModelForCausalLM, "from_config", with_runtime_buffer)
    monkeypatch.setattr("transformers.utils.hub.cached_file", lambda model, name, **kwargs: str(weights) if name == SAFE_WEIGHTS_NAME else None)
    loaded = model_mod._load_text_from_multimodal("fixture/model", config, torch.float32, "cpu")
    assert torch.equal(loaded.runtime_cache, torch.ones(3))
    for name, tensor in reference.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight


def test_tensor_reader_rejects_post_header_truncation(tmp_path: Path):
    path = tmp_path / SAFE_WEIGHTS_NAME
    save_file({"weight": torch.arange(4)}, str(path))
    header, offset = model_mod._read_safetensors_header(str(path))
    with path.open("r+b") as handle:
        handle.truncate(offset + 8)
    fd = model_mod._open_uncached(str(path))
    try:
        with pytest.raises(ValueError, match="truncated"):
            model_mod._read_st_tensor(fd, header, offset, "weight")
    finally:
        os.close(fd)


def test_scalar_and_empty_tensor_headers_remain_valid(tmp_path: Path):
    path = tmp_path / SAFE_WEIGHTS_NAME
    values = {"empty": torch.empty(0, 3), "scalar": torch.tensor(2.0)}
    save_file(values, str(path))
    header, offset = model_mod._read_safetensors_header(str(path))
    fd = model_mod._open_uncached(str(path))
    try:
        for name, expected in values.items():
            assert torch.equal(model_mod._read_st_tensor(fd, header, offset, name), expected)
    finally:
        os.close(fd)


def test_valid_nested_shards_and_hf_blob_symlinks_preserve_exact_weights(tmp_path: Path):
    config = _tiny_text_config(False)
    reference = model_mod.AutoModelForCausalLM.from_config(config)
    state = _remap_to_layout(reference.state_dict(), "mistral")
    model_root = tmp_path / "snapshot"
    shards = model_root / "shards"
    shards.mkdir(parents=True)
    blob = tmp_path / "blob"
    save_file(state, str(blob))
    (shards / "weights.safetensors").symlink_to(blob)
    (model_root / SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps({
        "weight_map": {name: "shards/weights.safetensors" for name in state},
    }))
    loaded = model_mod._load_text_from_multimodal(str(model_root), config, torch.float32, "cpu")
    for name, tensor in reference.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64, torch.complex64, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz])
def test_ignored_vision_dtype_does_not_restrict_text_loading(tmp_path: Path, dtype: torch.dtype):
    config = _tiny_text_config(False)
    reference = model_mod.AutoModelForCausalLM.from_config(config)
    state = _remap_to_layout(reference.state_dict(), "mistral")
    state["model.vision_tower.buffer"] = torch.zeros(2, dtype=dtype)
    save_file(state, str(tmp_path / SAFE_WEIGHTS_NAME))

    loaded = model_mod._load_text_from_multimodal(str(tmp_path), config, torch.float32, "cpu")

    for name, tensor in reference.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor), name


@pytest.mark.parametrize("dtype,bits", [
    ("F4", 4), ("F6_E2M3", 6), ("F6_E3M2", 6), ("F8_E8M0", 8),
    ("F8_E4M3FNUZ", 8), ("F8_E5M2FNUZ", 8), ("U16", 16),
    ("U32", 32), ("U64", 64), ("C64", 64),
])
def test_valid_unselected_dtypes_validate_but_do_not_silently_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: str, bits: int):
    path = tmp_path / SAFE_WEIGHTS_NAME
    raw = json.dumps({"vision": {"dtype": dtype, "shape": [8], "data_offsets": [0, bits]}}).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(bits))
    with safe_open(str(path), framework="pt") as archive:
        assert archive.keys() == ["vision"]
    header, offset = model_mod._read_safetensors_header(str(path))

    def no_read(*args: Any):
        pytest.fail("unsupported selected dtype was read before rejection")

    monkeypatch.setattr(os, "pread", no_read)
    fd = model_mod._open_uncached(str(path))
    try:
        with pytest.raises(ValueError, match="unsupported.*selected tensor"):
            model_mod._read_st_tensor(fd, header, offset, "vision")
    finally:
        os.close(fd)


@pytest.mark.parametrize("shape,dtype", [([2**64, 0], "F32"), ([2**32] * 256, "F32"), ([1], "F4")])
def test_shape_validation_rejects_unbounded_or_misaligned_dimensions(tmp_path: Path, shape: list[int], dtype: str):
    header = {"vision": {"dtype": dtype, "shape": shape, "data_offsets": [0, 1]}}
    raw = json.dumps(header).encode()
    path = tmp_path / SAFE_WEIGHTS_NAME
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0")
    with pytest.raises(ValueError, match="safetensors"):
        model_mod._read_safetensors_header(str(path))


def test_outer_head_alone_does_not_identify_a_supported_text_layout(tmp_path: Path):
    config = _tiny_text_config(False)
    reference = model_mod.AutoModelForCausalLM.from_config(config)
    save_file(reference.state_dict(), str(tmp_path / SAFE_WEIGHTS_NAME))
    with pytest.raises(model_mod._NoTextWeightsExtracted):
        model_mod._load_text_from_multimodal(str(tmp_path), config, torch.float32, "cpu")


@pytest.mark.parametrize("architecture", ["mistral3", "gemma3"])
@pytest.mark.parametrize("tie", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_actual_multimodal_wrapper_preserves_head_weights_and_logits(tmp_path: Path, architecture: str, tie: bool, sharded: bool):
    from transformers import AutoConfig, AutoModelForImageTextToText

    text_config = {
        "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
        "vocab_size": 32, "tie_word_embeddings": tie, "pad_token_id": 0,
    }
    vision_config = {
        "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
        "num_attention_heads": 2, "image_size": 16, "patch_size": 8,
    }
    extra = {"mm_tokens_per_image": 4} if architecture == "gemma3" else {}
    config = AutoConfig.for_model(
        architecture, text_config=text_config, vision_config=vision_config,
        tie_word_embeddings=tie, **extra,
    )
    reference = AutoModelForImageTextToText.from_config(config).eval()
    state = {name: value.clone() for name, value in reference.state_dict().items()}
    assert "lm_head.weight" in state
    if tie:
        del state["lm_head.weight"]
    if sharded:
        body = {name: value for name, value in state.items() if name != "lm_head.weight"}
        head = {name: value for name, value in state.items() if name == "lm_head.weight"}
        save_file(body, str(tmp_path / "body.safetensors"))
        save_file(head, str(tmp_path / "head.safetensors"))
        (tmp_path / SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps({
            "weight_map": {name: "head.safetensors" if name in head else "body.safetensors" for name in state},
        }))
    else:
        save_file(state, str(tmp_path / SAFE_WEIGHTS_NAME))

    loaded = model_mod._load_text_from_multimodal(str(tmp_path), config.text_config, torch.float32, "cpu").eval()

    reference_state = reference.state_dict()
    for name, value in loaded.state_dict().items():
        source = "model.language_model." + name[len("model."):] if name.startswith("model.") else name
        assert torch.equal(value, reference_state[source]), name
    assert (loaded.lm_head.weight is loaded.model.embed_tokens.weight) == tie
    with torch.no_grad():
        tokens = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(loaded(tokens).logits, reference(tokens).logits, rtol=0, atol=0)


def _generic_plan(folder: Path, config: Any) -> model_mod.LoadPlan:
    return model_mod.LoadPlan(
        model_id=str(folder), device="cpu", dtype=torch.float32, quantize=None,
        revision=None, pin_kwargs={}, tokenizer_kwargs={}, config=config,
        text_config=None, extract_text_model=False,
        load_kwargs={"dtype": torch.float32, "device_map": {"": "cpu"},
                     "attn_implementation": "eager", "trust_remote_code": False,
                     "local_files_only": True},
    )


@pytest.mark.parametrize("tie", [False, True])
def test_generic_load_preserves_exact_weights_logits_and_runtime_buffers(tmp_path: Path, tie: bool):
    config = _tiny_text_config(tie)
    reference = model_mod.AutoModelForCausalLM.from_config(config, attn_implementation="eager").eval()
    state = reference.state_dict()
    state["training_only.unused"] = torch.ones(1)
    reference.save_pretrained(str(tmp_path), state_dict=state, safe_serialization=True)

    loaded = model_mod._materialize_model(_generic_plan(tmp_path, config)).eval()

    for name, value in reference.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value), name
    for name, value in reference.named_buffers():
        assert torch.equal(dict(loaded.named_buffers())[name], value), name
    assert (loaded.lm_head.weight is loaded.model.embed_tokens.weight) == tie
    with torch.no_grad():
        tokens = torch.tensor([[1, 2, 3]])
        torch.testing.assert_close(loaded(tokens).logits, reference(tokens).logits, rtol=0, atol=0)


@pytest.mark.parametrize("tie", [False, True])
def test_generic_load_rejects_missing_required_weights(tmp_path: Path, tie: bool):
    config = _tiny_text_config(tie)
    reference = model_mod.AutoModelForCausalLM.from_config(config)
    state = reference.state_dict()
    if tie:
        del state["model.embed_tokens.weight"]
    del state["lm_head.weight"]
    reference.save_pretrained(str(tmp_path), state_dict=state, safe_serialization=True)
    with pytest.raises(model_mod._IncompleteCheckpointError, match="missing model parameters"):
        model_mod._materialize_model(_generic_plan(tmp_path, config))


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_missing_parameter_error_never_uses_backend_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str):
    model = torch.nn.Module()
    model.register_parameter("CONVERSION", torch.nn.Parameter(torch.ones(1)))
    plan = _generic_plan(tmp_path, _tiny_text_config(False))
    plan.device = device
    attempts = []

    def from_pretrained(*args: Any, **kwargs: Any):
        attempts.append(kwargs)
        return model, {"missing_keys": ["CONVERSION"]}

    monkeypatch.setattr(model_mod.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    with pytest.raises(model_mod._IncompleteCheckpointError):
        model_mod._materialize_model(plan)
    assert len(attempts) == 1
    assert attempts[0]["output_loading_info"] is True


@pytest.mark.parametrize("kind", ["attention", "dtype", "device", "quantized"])
def test_checked_generic_load_preserves_backend_options_and_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str):
    model = torch.nn.Linear(2, 2)
    plan = _generic_plan(tmp_path, _tiny_text_config(False))
    plan.load_kwargs.update(dtype=torch.bfloat16, attn_implementation="sdpa", revision="pinned")
    if kind == "device":
        plan.device = "mps"
        plan.load_kwargs["device_map"] = {"": "mps"}
    if kind == "quantized":
        plan.quantize = "4bit"
        plan.load_kwargs["quantization_config"] = object()
    attempts = []
    moves = []

    def from_pretrained(*args: Any, **kwargs: Any):
        attempts.append(dict(kwargs))
        if kind == "attention" and len(attempts) == 1:
            raise ValueError("model does not support an attention implementation")
        if kind == "dtype" and len(attempts) == 1:
            raise RuntimeError("unsupported backend dtype")
        if kind == "device" and len(attempts) <= 2:
            raise RuntimeError("CONVERSION backend failure")
        return model, {"missing_keys": ["optional_buffer"], "unexpected_keys": ["training_only"]}

    def to_device(device: str):
        moves.append(device)
        return model

    monkeypatch.setattr(model_mod.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(model, "to", to_device)
    assert model_mod._materialize_model(plan) is model
    assert all(row["output_loading_info"] and row["revision"] == "pinned" for row in attempts)
    if kind == "attention":
        assert [row["attn_implementation"] for row in attempts] == ["sdpa", "eager"]
    elif kind == "dtype":
        assert [row["dtype"] for row in attempts] == [torch.bfloat16, torch.float32]
    elif kind == "device":
        assert [row["device_map"] for row in attempts] == [{"": "mps"}, {"": "mps"}, {"": "cpu"}]
        assert moves == ["mps"]
    else:
        assert len(attempts) == 1
        assert attempts[0]["quantization_config"] is plan.load_kwargs["quantization_config"]
