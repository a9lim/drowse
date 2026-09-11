import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "browser-runtime/forks/build-production-webgpu.py"
SPEC = importlib.util.spec_from_file_location("build_production_webgpu", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_webllm_gemma_uses_global_context_without_top_level_sliding_window():
    config = SimpleNamespace(sliding_window_size=512)
    assert TOOL.webllm_sliding_window_size("gemma3_text", config) == -1
    assert TOOL.webllm_sliding_window_size("llama", config) == 512


def test_production_quantizations_cover_f32_compute_for_fp16_unsafe_models():
    assert TOOL.PRODUCTION_QUANTIZATIONS == ("q4f16_1", "q4f32_1", "q0f32")


def test_gpt2_browser_shape_preserves_native_fields_and_adds_canonical_dimensions():
    config = SimpleNamespace(n_layer=12, n_embd=768, asdict=lambda: {"n_layer": 12, "n_embd": 768})
    assert TOOL.browser_model_config("gpt2", config) == {
        "n_layer": 12, "n_embd": 768, "num_hidden_layers": 12, "hidden_size": 768,
    }
    assert TOOL.browser_model_config("gpt_neox", config) == {"n_layer": 12, "n_embd": 768}


def test_production_builder_matches_checked_in_overlay():
    manifest = json.loads((SCRIPT.parent / "manifest.json").read_text())
    overlay = next(item for item in manifest["overlays"] if item["baseCommit"] == TOOL.MLC_BASE_COMMIT)
    assert TOOL.RESULT_DIGESTS == overlay["resultFiles"]
    tvm_overlay = next(item for item in manifest["overlays"] if item["id"] == "tvm-webgpu-readonly")
    assert TOOL.TVM_RESULT_DIGESTS == tvm_overlay["resultFiles"]
    assert "python/tvm/relax/frontend/nn/llm/_kernel_common.py" in TOOL.TVM_RESULT_DIGESTS
    assert "python/tvm/relax/frontend/nn/llm/_decode_kernels.py" in TOOL.TVM_RESULT_DIGESTS
    for name, digest in TOOL.HYBRID_ADAPTER_DIGESTS.items():
        assert TOOL.sha256(SCRIPT.parent / name) == digest


@pytest.mark.parametrize("architecture,limit", [("gpt2", 1024), ("gpt_neox", 2048), ("gemma3_text", 32768)])
def test_base_context_and_raw_policy_are_checked_before_conversion(tmp_path: Path, architecture: str, limit: int):
    config = {"n_positions": limit, "max_position_embeddings": limit, "eos_token_id": 0}
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = SimpleNamespace(architecture=architecture, model_type="base", conv_template="raw",
                           context_window_size=min(limit, 2048), prefill_chunk_size=1024)
    TOOL.validate_completion_policy(args, tmp_path)
    args.context_window_size = limit + 1
    with pytest.raises(SystemExit, match="position limit"):
        TOOL.validate_completion_policy(args, tmp_path)
    args.context_window_size = min(limit, 2048)
    args.prefill_chunk_size = args.context_window_size + 1
    with pytest.raises(SystemExit, match="prefill chunk"):
        TOOL.validate_completion_policy(args, tmp_path)
    args.prefill_chunk_size = 1024
    args.conv_template = "chatml"
    with pytest.raises(SystemExit, match="raw"):
        TOOL.validate_completion_policy(args, tmp_path)


@pytest.mark.parametrize("eos", [None, True, -1, [], [0, None], "0"])
def test_base_completion_rejects_invalid_source_eos(tmp_path: Path, eos: object):
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": eos}))
    with pytest.raises(SystemExit, match="EOS"):
        TOOL.source_eos_ids(tmp_path)


def test_base_completion_retains_zero_and_generation_config_eos(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 0}))
    assert TOOL.source_eos_ids(tmp_path) == [0]
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [7, 0, 7]}))
    assert TOOL.source_eos_ids(tmp_path) == [0, 7]


def test_base_special_tokens_do_not_invent_zero_padding_or_chat_eos():
    tokenizer = SimpleNamespace(bos_token_id=50256, pad_token_id=None)
    assert TOOL.base_special_token_config(tokenizer, [50256]) == {
        "bos_token_id": 50256, "eos_token_id": 50256, "pad_token_id": 50256,
    }
    tokenizer = SimpleNamespace(bos_token_id=None, pad_token_id=248044)
    assert TOOL.base_special_token_config(tokenizer, [248044, 248046]) == {
        "bos_token_id": None, "eos_token_id": [248044, 248046], "pad_token_id": 248044,
    }


@pytest.mark.parametrize("prefix", [[], [2]])
def test_completion_prefix_keeps_the_tokenizers_special_token_policy(prefix: list[int]):
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens:
        (prefix if add_special_tokens else []) + ([4, 5] if text else []))
    assert TOOL.completion_prefix_ids(tokenizer) == prefix


def test_completion_prefix_rejects_a_tokenizer_that_appends_eos():
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens:
        ([4, 5] if text else []) + ([1] if add_special_tokens else []))
    with pytest.raises(SystemExit, match="non-prefix special tokens"):
        TOOL.completion_prefix_ids(tokenizer)


def test_base_adapter_provenance_is_architecture_specific():
    assert "modelAdapterFiles" not in TOOL.toolchain_manifest("qwen3")
    for architecture in ("gpt2", "gpt_neox"):
        manifest = TOOL.toolchain_manifest(architecture)
        assert manifest["modelAdapterFiles"] == TOOL.BASE_ADAPTER_DIGESTS
        assert manifest["modelSourceFiles"] == TOOL.BASE_MODEL_SOURCE_DIGESTS[architecture]
    hybrid = TOOL.toolchain_manifest("qwen3_5")
    assert hybrid["modelAdapterFiles"] == TOOL.HYBRID_ADAPTER_DIGESTS
    assert hybrid["modelSourceFiles"] == TOOL.BASE_MODEL_SOURCE_DIGESTS["qwen3_5"]
    assert "qwen35_adapter.py" not in TOOL.toolchain_manifest("gpt2")["modelAdapterFiles"]


def test_qwen35_base_policy_uses_the_text_backbone_not_vision_or_chat(tmp_path: Path):
    config = {"model_type": "qwen3_5", "max_position_embeddings": 4096, "eos_token_id": 99,
              "text_config": {"model_type": "qwen3_5_text", "max_position_embeddings": 2048,
                              "eos_token_id": 248044}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = SimpleNamespace(architecture="qwen3_5", model_type="base", conv_template="raw",
                           context_window_size=2048, prefill_chunk_size=2048)
    TOOL.validate_completion_policy(args, tmp_path)
    assert TOOL.source_eos_ids(tmp_path) == [248044]
    assert TOOL.derive_thinking_profile(tmp_path, "qwen3_5") is None
    args.context_window_size = 4096
    with pytest.raises(SystemExit, match="position limit"):
        TOOL.validate_completion_policy(args, tmp_path)
    args.model_type, args.conv_template = "chat", "chatml"
    with pytest.raises(SystemExit, match="base classification"):
        TOOL.validate_completion_policy(args, tmp_path)


@pytest.mark.parametrize("format,dtype", [("f32-to-bf16", "float32"), ("raw", "float16"), ("raw", "bfloat16")])
def test_fp32_reference_build_rejects_lossy_tensor_storage(tmp_path: Path, format: str, dtype: str):
    path = tmp_path / "tensor-cache.json"
    path.write_text(json.dumps({"records": [{"records": [{"format": format, "dtype": dtype}]}]}))
    with pytest.raises(SystemExit, match="lossless raw"):
        TOOL.validate_weight_encoding(tmp_path, "q0f32")
    TOOL.validate_weight_encoding(tmp_path, "q4f16_1")
    path.write_text(json.dumps({"records": [{"records": [{"format": "raw", "dtype": "float32"}]}]}))
    TOOL.validate_weight_encoding(tmp_path, "q0f32")


def test_active_vocabulary_cannot_exceed_model_rows():
    assert TOOL.bounded_active_vocab_size(262_145, 262_144) == 262_144
    assert TOOL.bounded_active_vocab_size(32_000, 32_768) == 32_000


@pytest.mark.parametrize("architecture,config", [
    ("qwen3", {"model_type": "qwen3"}),
    ("gemma3_text", {"model_type": "gemma3", "text_config": {"model_type": "gemma3_text"}}),
])
def test_validate_source_requires_one_exact_hugging_face_revision(tmp_path: Path, architecture: str, config: dict[str, object]):
    revision = "a" * 40
    metadata = tmp_path / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    files = {
        "config.json": json.dumps(config).encode(),
        "model.safetensors": b"weights",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
    }
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
        (metadata / f"{name}.metadata").write_text(f"{revision}\netag\n0\n")

    TOOL.validate_source(tmp_path, revision, architecture)
    with pytest.raises(SystemExit, match="came from"):
        TOOL.validate_source(tmp_path, "b" * 40, architecture)
    if architecture == "gemma3_text":
        text_config = config["text_config"]
        assert isinstance(text_config, dict)
        text_config["model_type"] = "foreign_text"
        (tmp_path / "config.json").write_text(json.dumps(config))
        with pytest.raises(SystemExit, match="source model type"):
            TOOL.validate_source(tmp_path, revision, architecture)


def test_gemma_wrapped_text_backbone_supplies_position_limit_and_eos(tmp_path: Path):
    config = {"model_type": "gemma3", "text_config": {
        "model_type": "gemma3_text", "max_position_embeddings": 131072, "eos_token_id": 106,
    }}
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = SimpleNamespace(architecture="gemma3_text", model_type="chat", conv_template="gemma3_instruction",
                           context_window_size=4096, prefill_chunk_size=2048)
    TOOL.validate_completion_policy(args, tmp_path)
    assert TOOL.source_eos_ids(tmp_path) == [106]
    args.context_window_size = 131073
    with pytest.raises(SystemExit, match="position limit"):
        TOOL.validate_completion_policy(args, tmp_path)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("source_key,compiled_key,wrong", [
    ("rope_theta", "position_embedding_base", 10000),
    ("rope_local_base_freq", "rope_local_base_freq", 1000000),
    ("rope_scaling", "rope_scaling", None),
    ("sliding_window", "sliding_window_size", -1),
    ("sliding_window_pattern", "sliding_window_pattern", 5),
])
def test_gemma_conversion_preserves_source_attention_settings(tmp_path: Path, wrapped: bool,
                                                            source_key: str, compiled_key: str, wrong: object):
    source = {"rope_theta": 1000000, "rope_local_base_freq": 10000,
              "rope_scaling": {"rope_type": "linear", "factor": 8},
              "sliding_window": 1024, "sliding_window_pattern": 6}
    compiled = {"position_embedding_base": 1000000, "rope_local_base_freq": 10000,
                "rope_scaling": source["rope_scaling"], "sliding_window_size": 1024, "sliding_window_pattern": 6}
    (tmp_path / "config.json").write_text(json.dumps({"text_config": source} if wrapped else source))
    output = tmp_path / "mlc-chat-config.json"
    output.write_text(json.dumps({"model_config": {"text_config": compiled}}))
    TOOL.validate_gemma_attention_config(tmp_path, tmp_path)
    compiled[compiled_key] = wrong
    output.write_text(json.dumps({"model_config": {"text_config": compiled}}))
    with pytest.raises(SystemExit, match=source_key):
        TOOL.validate_gemma_attention_config(tmp_path, tmp_path)


def test_validate_source_closes_sharded_weights_and_optional_tokenizer_inputs(tmp_path: Path):
    revision = "c" * 40
    metadata = tmp_path / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    files = {
        "config.json": json.dumps({"model_type": "llama"}).encode(),
        "model.safetensors.index.json": json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        ).encode(),
        "model-00001-of-00002.safetensors": b"one",
        "model-00002-of-00002.safetensors": b"two",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "special_tokens_map.json": b"{}",
        "generation_config.json": b"{}",
    }
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
        (metadata / f"{name}.metadata").write_text(f"{revision}\netag\n0\n")

    TOOL.validate_source(tmp_path, revision, "llama")
    assert TOOL.conversion_source_files(tmp_path) == sorted(files)


def test_verify_build_manifest_closes_and_hashes_every_artifact(tmp_path: Path):
    artifact = tmp_path / "mlc-chat-config.json"
    artifact.write_bytes(b"config")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(json.dumps({
        "added_tokens": [
            {"id": 151667, "content": "<think>"},
            {"id": 151668, "content": "</think>"},
        ],
    }))
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(json.dumps({
        "chat_template": "{% if enable_thinking %}<think>{% else %}</think>{% endif %}",
    }))
    thinking_profile = {
        "start": "<think>",
        "end": "</think>",
        "startsInThinking": False,
        "startTokenIds": [151667],
        "endTokenIds": [151668],
    }
    manifest = {
        "schemaVersion": 1,
        "runtimeAbi": "drowse-web-runtime-v1",
        "hookAbi": "post-block-residual-v4",
        "structuredHookProfile": TOOL.STANDARD_STRUCTURED_HOOK_PROFILE,
        "thinkingProfile": thinking_profile,
        "architecture": "qwen3",
        "quantization": "q4f16_1",
        "toolchain": {
            "mlcLlmBaseCommit": TOOL.MLC_BASE_COMMIT,
            "mlcOverlayFiles": TOOL.RESULT_DIGESTS,
            "tvmCommit": TOOL.TVM_COMMIT,
            "tvmFfiCommit": TOOL.TVM_FFI_COMMIT,
            "tvmOverlayFiles": TOOL.TVM_RESULT_DIGESTS,
        },
        "files": [
            {
                "path": artifact.name,
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            },
            {
                "path": tokenizer.name,
                "bytes": tokenizer.stat().st_size,
                "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
            },
            {
                "path": tokenizer_config.name,
                "bytes": tokenizer_config.stat().st_size,
                "sha256": hashlib.sha256(tokenizer_config.read_bytes()).hexdigest(),
            },
        ],
    }
    (tmp_path / "drowse-build.json").write_text(json.dumps(manifest))

    TOOL.verify_build_manifest(tmp_path, "qwen3", "q4f16_1")
    with pytest.raises(SystemExit, match="structuredHookProfile"):
        TOOL.verify_build_manifest(tmp_path, "qwen3", "q4f16_1", "unlisted")
    artifact.write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="failed verification"):
        TOOL.verify_build_manifest(tmp_path, "qwen3", "q4f16_1")


def test_thinking_profile_is_tokenizer_derived_and_architecture_bound(tmp_path: Path):
    (tmp_path / "tokenizer.json").write_text(json.dumps({
        "added_tokens": [
            {"id": 151667, "content": "<think>"},
            {"id": 151668, "content": "</think>"},
        ],
    }))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "enable_thinking <think> </think>",
    }))
    assert TOOL.derive_thinking_profile(tmp_path, "qwen3") == {
        "start": "<think>",
        "end": "</think>",
        "startsInThinking": False,
        "startTokenIds": [151667],
        "endTokenIds": [151668],
    }
    assert TOOL.derive_thinking_profile(tmp_path, "llama") is None

    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "<think> </think>",
    }))
    with pytest.raises(SystemExit, match="thinking controls"):
        TOOL.derive_thinking_profile(tmp_path, "qwen3")


def test_verify_compiled_library_requires_wasm_and_every_drowse_export(tmp_path: Path):
    output = tmp_path / "model.wasm"
    tile_shader = (
        b"// Function: drowse_exact_top8_tiles_kernel\n"
        + b"\n".join(
            f"@group(0) @binding({binding}) var<storage, read_write> buffer_{binding}: array<f32>;".encode()
            for binding in range(3)
        )
        + b"\n"
        + TOOL.TOPK_WORKGROUP_LINE
        + b"\nfn drowse_exact_top8_tiles_kernel() {\n"
        + b"var<workgroup> best_values : array<f32, 64>;\n"
        + b"var<workgroup> best_indices : array<i32, 64>;\n"
        + b"workgroupBarrier();\nfor (var rank : i32 = 0i; rank < 8i; rank++) {\n"
        + b"for (var step : i32 = 0i; step < 256i; step++) {}\n}\n}\n"
    )
    merge_shader = (
        b"// Function: drowse_exact_top8_merge_kernel\n"
        + b"\n".join(
            f"@group(0) @binding({binding}) var<storage, read_write> buffer_{binding}: array<f32>;".encode()
            for binding in range(4)
        )
        + b"\n"
        + TOOL.TOPK_WORKGROUP_LINE
        + b"\nfn drowse_exact_top8_merge_kernel() {\n"
        + b"var<workgroup> best_values : array<f32, 64>;\n"
        + b"var<workgroup> best_indices : array<i32, 64>;\n"
        + b"workgroupBarrier();\nfor (var rank : i32 = 0i; rank < 8i; rank++) {\n"
        + b"for (var candidate : i32 = 0i; candidate < podArgs.candidates_per_row; candidate++) {}\n}\n}\n"
    )
    transport_shader = (
        b"// Function: drowse_jlens_transport_kernel\n"
        + b"\n".join(
            f"@group(0) @binding({binding}) var<storage, read> buffer_{binding}: array<f32>;".encode()
            for binding in range(3)
        )
        + b"\n"
        + TOOL.JLENS_TRANSPORT_WORKGROUP_LINE
        + b"\nfn drowse_jlens_transport_kernel() {\n"
        + b"var<workgroup> partial : array<f32, 128>;\nworkgroupBarrier();\nfor (var source_coordinate : i32 = 0i; source_coordinate < 8i; source_coordinate++) {\n"
        + b"let transported = fma(1.0f, 1.0f, 0.0f);\n}\n}\n"
    )
    geometry_shader = (
        b"// Function: drowse_geometry_measurements_kernel\n"
        + b"\n".join(
            f"@group(0) @binding({binding}) var<storage, read> buffer_{binding}: array<f32>;".encode()
            for binding in range(TOOL.MAX_WEBGPU_STORAGE_BINDINGS_PER_STAGE)
        )
        + b"\n@compute @workgroup_size(1) fn main() {}\n"
    )
    output.write_bytes(
        b"\x00asm" + b"".join(
            len(function.encode()).to_bytes(8, "little") + function.encode()
            for function in sorted(TOOL.REQUIRED_FUNCTIONS)
        ) + tile_shader + merge_shader + transport_shader + geometry_shader
    )
    TOOL.verify_compiled_library(output)

    output.write_bytes(output.read_bytes().replace(b"fma(", b"mul(", 1))
    with pytest.raises(SystemExit, match="bounded cooperative fp32 schedule"):
        TOOL.verify_compiled_library(output)

    output.write_bytes(b"not wasm")
    with pytest.raises(SystemExit, match="valid WebAssembly"):
        TOOL.verify_compiled_library(output)

    function = "drowse_prefill"
    output.write_bytes(
        b"\x00asm" + len(function.encode()).to_bytes(8, "little") + function.encode()
    )
    with pytest.raises(SystemExit, match="lacks Drowse VM function metadata"):
        TOOL.verify_compiled_library(output)


def test_copy_notices_packages_full_declared_model_license(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "README.md").write_text("---\nlicense: apache-2.0\n---\n# Model\n")

    TOOL.copy_notices(source, output)

    license_record = json.loads((output / "MODEL-LICENSE.json").read_text())
    assert license_record == {
        "declaredIn": "README.md",
        "licenseText": "LICENSE.model",
        "licenseTextSource": "drowse-standard-text",
        "schemaVersion": 1,
        "spdx": "apache-2.0",
    }
    assert "Apache License" in (output / "LICENSE.model").read_text()

    (source / "LICENSE").write_text("source terms")
    TOOL.copy_notices(source, output)
    assert (output / "LICENSE.model").read_text() == "source terms"
    assert json.loads((output / "MODEL-LICENSE.json").read_text())["licenseTextSource"] == (
        "source-snapshot"
    )

    (source / "README.md").write_text("---\nlicense: mit\n---\n")
    with pytest.raises(SystemExit, match="unsupported model license"):
        TOOL.copy_notices(source, output)

    TOOL.copy_notices(source, output, "openai-community/gpt2")
    assert "Software Copyright (c) 2019 OpenAI" in (output / "LICENSE.model").read_text()
    assert json.loads((output / "MODEL-LICENSE.json").read_text())["licenseId"] == (
        "LicenseRef-GPT2-Modified-MIT"
    )
