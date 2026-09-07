#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path


MLC_BASE_COMMIT = "9fa644f54b04983adea4d0168f49fc6af4a893ba"
TVM_COMMIT = "68ba2b31c2a6d202fcd44e5780ef10ce08721dd5"
TVM_FFI_COMMIT = "12dbf053b3d9ba4ebd9da3123b1aeca79cf74229"
EMCC_VERSION = "3.1.56"
EMCC_REVISION = "cf90417346b78455089e64eb909d71d091ecc055"
RESULT_DIGESTS = {
    "python/mlc_llm/interface/convert_weight.py": "dffc4af8487f54087bc27f9a35f959df318a70c4c33326b65d4a6d9f6479a1b5",
    "python/mlc_llm/interface/compile.py": "435876d96e9ba6fbc3f02dd6f5ed858df559ebded169ce91c5cdd8151e610d32",
    "python/mlc_llm/op/moe_misc.py": "4141be404d6d31b1ee152ef495fd5bdfc75b16af968ad01e07c3d1d26637ef92",
    "python/mlc_llm/model/llama/llama_model.py": "2a34402c81074ce0a773f9a73dc95aea20a7a0533b2561708d32ceeb72aec7dc",
    "python/mlc_llm/model/qwen3/qwen3_model.py": "0621190a80fc12ec58ca3fd5d439411135815c4f4f6596a5a0d258475b5c7291",
    "python/mlc_llm/model/gemma3/gemma3_model.py": "fd219c4779b1a7497fa533b264d5c77a973c5a1f46f41ed29923c312d6220f18",
    "python/mlc_llm/model/drowse_hooks.py": "bc1763ba32ef304b4c45be05bcbd2f2eb1240fc5fd7f95f48e025d96dce76c9e",
}
TVM_RESULT_DIGESTS = {
    "python/tvm/relax/frontend/nn/llm/_decode_kernels.py": "4cfba82db3cd92e0b02081bd9679f24663fdca327de767966ae50c135c92e3ce",
    "python/tvm/relax/frontend/nn/llm/_kernel_common.py": "f5bc8999ae5a6f06e96da1c8262fedc5ae53230c61439285be0cdb0828a43012",
    "python/tvm/relax/frontend/nn/llm/_prefill_kernels.py": "cbb41ed61ece7c8741c73dcd3690635c66fd7cdb71a755b78d1b70e0c5ad992e",
    "src/backend/webgpu/codegen/codegen_webgpu.cc": "753ff21703a741cf7a8bc49678f4737127f04fa1f28f0fef06aefb4859b5d4a7",
    "src/runtime/vm/paged_kv_cache.cc": "90ec1ed27fa2cc3faf426f60ab46e1b5c6ba352de177d50db168a4357dba3ed2",
    "tests/python/codegen/test_target_codegen_webgpu.py": "3da351a5fd03a1a0b05ed926581fe3d46e5160c23f926f931b0a8776d3f2e9c9"
}
REQUIRED_FUNCTIONS = {
    "drowse_hook_profile",
    "drowse_prefill",
    "drowse_decode",
    "drowse_batch_prefill",
    "drowse_batch_decode",
    "drowse_capture_prefill",
    "drowse_capture_decode",
    "drowse_capture_batch_prefill",
    "drowse_capture_batch_decode",
    "drowse_rank_one_capture_prefill_v1",
    "drowse_rank_one_capture_decode_v1",
    "drowse_rank_one_capture_batch_prefill_v1",
    "drowse_rank_one_capture_batch_decode_v1",
    "drowse_structured_batch_prefill",
    "drowse_structured_batch_decode",
    "drowse_geometry_batch_prefill",
    "drowse_geometry_batch_decode",
    "drowse_curved_batch_prefill",
    "drowse_curved_batch_decode",
    "drowse_jlens_probabilities",
    "drowse_jlens_readout_accumulate",
    "drowse_jlens_readout_topk",
    "drowse_jlens_directions",
    "drowse_sae_readout_accumulate",
    "drowse_sae_jump_relu_readout_accumulate",
}
GEOMETRY_SHADER_PREFIX = b"// Function: drowse_geometry_measurements"
TOPK_TILE_SHADER_PREFIX = b"// Function: drowse_exact_top8_tiles"
TOPK_MERGE_SHADER_PREFIX = b"// Function: drowse_exact_top8_merge"
JLENS_TRANSPORT_SHADER_PREFIX = b"// Function: drowse_jlens_transport"
TOPK_WORKGROUP_LINE = b"@compute @workgroup_size(1, 1, 1)"
JLENS_TRANSPORT_WORKGROUP_LINE = b"@compute @workgroup_size(1, 1, 1)"
MAX_WEBGPU_STORAGE_BINDINGS_PER_STAGE = 8
TOKENIZER_SOURCE_FILES = [
    "tokenizer.model",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
]
OPTIONAL_SOURCE_FILES = ["generation_config.json", "LICENSE", "README.md"]
STANDARD_STRUCTURED_HOOK_PROFILE = "standard-v3"
THINKING_DELIMITERS = ("<think>", "</think>")
PRODUCTION_QUANTIZATIONS = ("q4f16_1", "q4f32_1", "q0f32")
PRODUCTION_ARCHITECTURES = ("qwen3", "llama", "gemma3_text", "gpt2", "gpt_neox", "qwen3_5")
BASE_ADAPTER_DIGESTS = {
    "base_model_adapters.py": "eed323ea949e87c537ca7b1c49c3db2841f26711cadd3921e3d46a5cd45eaca7",
}
HYBRID_ADAPTER_DIGESTS = {
    **BASE_ADAPTER_DIGESTS,
    "qwen35_adapter.py": "02f78f5acc870f27d46efd1dea5551dbddf67cd5e18b7280cb71fb0ef0b77fdf",
    "hybrid_state_kernels.py": "7075a4b80b11093503ca5048074095eb41c39cade72edf4fa6a0f69b2fdfb8a3",
}
BASE_MODEL_SOURCE_DIGESTS = {
    "gpt2": {
        "python/mlc_llm/model/gpt2/gpt2_loader.py": "d9e91af06a275b8b4e6b2419e51c9670702fddfb3ad9e3fecdf63901310e4c36",
        "python/mlc_llm/model/gpt2/gpt2_model.py": "b46e88b1c6eba7a4ad1e971792c19f89c547f98f44ebe761de09e2726929d6c5",
    },
    "gpt_neox": {
        "python/mlc_llm/model/gpt_neox/gpt_neox_loader.py": "0440155050e86cba299b8b6ce18cc016dfcc25e790e3cfdc3b209f86b99cd3c9",
        "python/mlc_llm/model/gpt_neox/gpt_neox_model.py": "219b09c91012214c58e9e82a273d1ea02521b2111d84db110a3575bcdeb9d7d2",
    },
    "qwen3_5": {
        "python/mlc_llm/model/qwen35/qwen35_model.py": "f8cabfa42d659643e6068f1c47160fc8e54269593ca93243a5f75ea7b507f2ac",
        "python/mlc_llm/model/qwen35/qwen35_loader.py": "67c7d0e9b285438a6d5c92e508346d9916c0c782cf758fb212497679bdbb1dd1",
    },
}


def main() -> None:
    args = parse_args()
    mlc_repository = args.mlc_repository.resolve(strict=True)
    tvm_repository = args.tvm_repository.resolve(strict=True)
    validate_toolchain(mlc_repository, tvm_repository)
    bootstrap_mlc_package(mlc_repository)
    if args.architecture in BASE_MODEL_SOURCE_DIGESTS:
        verify_files(mlc_repository, BASE_MODEL_SOURCE_DIGESTS[args.architecture], "base model source")
        verify_files(Path(__file__).resolve().parent, adapter_digests(args.architecture), "base model adapter")
        from base_model_adapters import register_candidate_models
        register_candidate_models(include_hybrid=args.architecture == "qwen3_5")
    if args.command == "convert":
        convert(args, mlc_repository, tvm_repository)
    else:
        compile_library(args, mlc_repository, tvm_repository)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlc-repository", required=True, type=Path)
    parser.add_argument("--tvm-repository", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--source", required=True, type=Path)
    convert_parser.add_argument("--source-repository", required=True)
    convert_parser.add_argument("--source-revision", required=True)
    convert_parser.add_argument(
        "--architecture", required=True, choices=PRODUCTION_ARCHITECTURES
    )
    convert_parser.add_argument(
        "--quantization", default="q4f16_1", choices=PRODUCTION_QUANTIZATIONS
    )
    convert_parser.add_argument("--context-window-size", type=int, choices=[1024, 2048, 4096], default=4096)
    convert_parser.add_argument("--prefill-chunk-size", type=int, choices=[1024, 2048, 4096], default=2048)
    convert_parser.add_argument("--conv-template", required=True)
    convert_parser.add_argument("--model-type", choices=["chat", "base"], default="chat")
    convert_parser.add_argument(
        "--structured-hook-profile",
        choices=[STANDARD_STRUCTURED_HOOK_PROFILE],
        default=STANDARD_STRUCTURED_HOOK_PROFILE,
    )
    convert_parser.add_argument("--output", required=True, type=Path)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--model", required=True, type=Path)
    compile_parser.add_argument(
        "--architecture", required=True, choices=PRODUCTION_ARCHITECTURES
    )
    compile_parser.add_argument(
        "--quantization", default="q4f16_1", choices=PRODUCTION_QUANTIZATIONS
    )
    compile_parser.add_argument(
        "--structured-hook-profile",
        choices=[STANDARD_STRUCTURED_HOOK_PROFILE],
        default=STANDARD_STRUCTURED_HOOK_PROFILE,
    )
    compile_parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def validate_toolchain(mlc_repository: Path, tvm_repository: Path) -> None:
    require_commit(mlc_repository, MLC_BASE_COMMIT, "MLC-LLM")
    require_commit(tvm_repository, TVM_COMMIT, "TVM")
    require_commit(tvm_repository / "3rdparty/tvm-ffi", TVM_FFI_COMMIT, "TVM-FFI")
    verify_files(mlc_repository, RESULT_DIGESTS, "MLC overlay")
    verify_portable_topk_source(mlc_repository)
    verify_files(tvm_repository, TVM_RESULT_DIGESTS, "TVM WebGPU overlay")
    expected_roots = [(tvm_repository / "python").resolve(), (mlc_repository / "python").resolve()]
    import_roots = {Path(path).resolve() for path in sys.path if path}
    missing = [str(path) for path in expected_roots if path not in import_roots]
    if missing:
        raise SystemExit("exact source roots are missing from PYTHONPATH: " + os.pathsep.join(missing))
    os.environ["MLC_LLM_SOURCE_DIR"] = str(mlc_repository)
    os.environ.setdefault("SKIP_LOADING_MLCLLM_SO", "1")


def require_commit(repository: Path, expected: str, label: str) -> None:
    actual = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise SystemExit(f"{label} checkout must be exact commit {expected}, found {actual}")


def verify_files(repository: Path, expected_files: dict[str, str], label: str) -> None:
    for relative_path, expected in expected_files.items():
        actual = sha256(repository / relative_path)
        if actual != expected:
            raise SystemExit(f"{label} result digest differs for {relative_path}")


def verify_portable_topk_source(repository: Path) -> None:
    source = (repository / "python/mlc_llm/op/moe_misc.py").read_text(encoding="utf-8")
    if (
        source.count("TX = 1024") < 2
        or "T.buffer_store(local_top_k_index, -1, indices=[t])" not in source
        or "T.buffer_store(local_top_k_index, t, indices=[-1])" in source
    ):
        raise SystemExit("generic MLC MoE top-k differs from upstream scheduling or safe initialization")
    hooks = (repository / "python/mlc_llm/model/drowse_hooks.py").read_text(encoding="utf-8")
    for fragment in (
        "EXACT_READOUT_TOPK_BLOCK_SIZE = 256",
        "candidate_count = T.ceildiv(column_count, EXACT_READOUT_TOPK_BLOCK_SIZE)",
        '"drowse_exact_top8_tiles"',
        '"drowse_exact_top8_merge"',
        '"drowse_jlens_transport"',
        "_exact_readout_not_selected",
        "T.And(",
        "T.Or(",
    ):
        if fragment not in hooks:
            raise SystemExit(f"Drowse exact tiled top-8 omits {fragment}")
    exact_source = hooks.split("def exact_readout_topk", 1)[-1].split(
        "def structured_geometry_payload_layout", 1
    )[0]
    if 'scope="shared"' in exact_source or "tvm_storage_sync" in exact_source:
        raise SystemExit("Drowse exact tiled top-8 must not use workgroup storage or barriers")
    for relative_path in (
        "python/mlc_llm/model/llama/llama_model.py",
        "python/mlc_llm/model/qwen3/qwen3_model.py",
    ):
        model = (repository / relative_path).read_text(encoding="utf-8")
        if (
            model.count("exact_readout_topk") != 5
            or model.count("transport_jlens_hidden") != 2
            or "op_ext.moe_misc.gating_topk(" in model
        ):
            raise SystemExit(f"{relative_path} does not route exact readouts through Drowse top-8")


def bootstrap_mlc_package(mlc_repository: Path) -> None:
    if "mlc_llm" in sys.modules:
        raise SystemExit("mlc_llm was imported before the source-only production bootstrap")
    package = types.ModuleType("mlc_llm")
    package.__path__ = [str(mlc_repository / "python/mlc_llm")]
    package.__package__ = "mlc_llm"
    sys.modules["mlc_llm"] = package


def convert(args, mlc_repository: Path, tvm_repository: Path) -> None:
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    validate_new_output(output, directory=True)
    validate_source(source, args.source_revision, args.architecture)
    validate_completion_policy(args, source)

    import tvm
    from mlc_llm.interface.convert_weight import convert_weight
    from mlc_llm.model import MODELS
    from mlc_llm.quantization import QUANTIZATION

    require_llvm(tvm)
    stage = Path(tempfile.mkdtemp(prefix=f"{output.name}.partial-", dir=output.parent))
    try:
        config = source / "config.json"
        model = MODELS[args.architecture]
        quantization = QUANTIZATION[args.quantization]
        convert_weight(
            config=config,
            quantization=quantization,
            model=model,
            device=tvm.runtime.cpu(),
            source=source_weight_files(source)[0],
            source_format="huggingface-safetensor",
            output=stage,
            encode_format="raw" if args.quantization == "q0f32" else "f32-to-bf16",
        )
        validate_weight_encoding(stage, args.quantization)

        generate_config(args, source, stage, model, quantization)
        complete_config(stage, source)
        copy_notices(source, stage, args.source_repository)
        write_build_manifest(args, source, stage)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps(output_summary(output), sort_keys=True))


def validate_source(source: Path, expected_revision: str, architecture: str) -> None:
    if not source.is_dir():
        raise SystemExit(f"source is not a directory: {source}")
    if len(expected_revision) != 40 or any(character not in "0123456789abcdef" for character in expected_revision):
        raise SystemExit("source revision must be a lowercase 40-character commit")
    for filename in conversion_source_files(source):
        path = source / filename
        metadata = source / ".cache/huggingface/download" / f"{filename}.metadata"
        if not path.is_file() or not metadata.is_file():
            raise SystemExit(f"exact Hugging Face snapshot is missing {filename}")
        revision = metadata.read_text(encoding="utf-8").splitlines()[0]
        if revision != expected_revision:
            raise SystemExit(f"{filename} came from {revision}, expected {expected_revision}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != architecture:
        raise SystemExit(
            f"source model type is {config.get('model_type')!r}, expected {architecture!r}"
        )
    if architecture == "qwen3_5" and config.get("text_config", {}).get("model_type") != "qwen3_5_text":
        raise SystemExit("Qwen 3.5 source requires its explicit text backbone configuration")


def detect_tokenizer_info(tokenizer_path: Path) -> dict:
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))

    def contains_byte_level(value) -> bool:
        if isinstance(value, dict):
            return value.get("type") == "ByteLevel" or any(contains_byte_level(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_byte_level(item) for item in value)
        return False

    return {
        "token_postproc_method": "byte_level" if contains_byte_level(tokenizer.get("pre_tokenizer")) else "byte_fallback",
        "prepend_space_in_encode": False,
        "strip_space_in_decode": False,
    }


def source_weight_files(source: Path) -> tuple[Path, list[str]]:
    single = source / "model.safetensors"
    if single.is_file():
        return single, [single.name]
    index = source / "model.safetensors.index.json"
    if not index.is_file():
        raise SystemExit("exact Hugging Face snapshot has no safetensor weights")
    values = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
    if not isinstance(values, dict) or not values:
        raise SystemExit("safetensor index has no weight map")
    shards = sorted(set(values.values()))
    if any(not isinstance(name, str) or Path(name).name != name for name in shards):
        raise SystemExit("safetensor index contains an unsafe shard path")
    missing = [name for name in shards if not (source / name).is_file()]
    if missing:
        raise SystemExit(f"safetensor index is missing shard: {missing[0]}")
    return index, [index.name, *shards]


def conversion_source_files(source: Path) -> list[str]:
    files = ["config.json", *source_weight_files(source)[1]]
    files.extend(name for name in TOKENIZER_SOURCE_FILES if (source / name).is_file())
    files.extend(name for name in OPTIONAL_SOURCE_FILES if (source / name).is_file())
    return sorted(set(files))


def generate_config(args, source: Path, output: Path, model, quantization) -> None:
    from mlc_llm.conversation_template import ConvTemplateRegistry
    from mlc_llm.interface.compiler_flags import ModelConfigOverride
    from mlc_llm.protocol.mlc_chat_config import MLCChatConfig, MLC_CHAT_SYSTEM_DEFAULT
    from mlc_llm.protocol.conversation_protocol import Conversation, MessagePlaceholders
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(source), use_fast=True, local_files_only=True)
    completion_prefix = completion_prefix_ids(tokenizer) if args.model_type == "base" else None
    if args.model_type == "base":
        conversation = Conversation(
            name="drowse-base", system_template="", system_message="",
            roles={"user": "", "assistant": ""}, seps=[""],
            role_content_sep="", role_empty_sep="", stop_str=[],
            stop_token_ids=source_eos_ids(source),
            system_prefix_token_ids=completion_prefix,
        )
    elif args.conv_template == "smollm2":
        conversation = Conversation(
            name="smollm2",
            system_template=f"<|im_start|>system\n{MessagePlaceholders.SYSTEM.value}<|im_end|>\n",
            system_message="You are a helpful AI assistant named SmolLM, trained by Hugging Face",
            roles={"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"},
            seps=["<|im_end|>\n"],
            role_content_sep="\n",
            role_empty_sep="\n",
            stop_str=["<|im_end|>"],
            stop_token_ids=[2],
        )
    else:
        conversation = ConvTemplateRegistry.get_conv_template(args.conv_template)
    if conversation is None:
        raise SystemExit(f"unknown MLC conversation template: {args.conv_template}")
    model_config = ModelConfigOverride(
        context_window_size=args.context_window_size,
        sliding_window_size=None,
        prefill_chunk_size=args.prefill_chunk_size,
        attention_sink_size=None,
        max_batch_size=1,
        tensor_parallel_shards=1,
        pipeline_parallel_stages=1,
        disaggregation=False,
    ).apply(model.config.from_file(source / "config.json"))
    config = MLCChatConfig(
        model_type=model.name,
        quantization=quantization.name,
        model_config=browser_model_config(args.architecture, model_config),
        vocab_size=model_config.vocab_size,
        active_vocab_size=model_config.vocab_size,
        context_window_size=model_config.context_window_size,
        sliding_window_size=webllm_sliding_window_size(args.architecture, model_config),
        prefill_chunk_size=model_config.prefill_chunk_size,
        attention_sink_size=getattr(model_config, "attention_sink_size", -1),
        tensor_parallel_shards=model_config.tensor_parallel_shards,
        pipeline_parallel_stages=getattr(model_config, "pipeline_parallel_stages", 1),
        conv_template=conversation,
        model_task=model.model_task,
        embedding_metadata=model.embedding_metadata,
    )
    for filename in TOKENIZER_SOURCE_FILES:
        path = source / filename
        if path.is_file():
            shutil.copyfile(path, output / filename)
            config.tokenizer_files.append(filename)
    config.tokenizer_info = detect_tokenizer_info(source / "tokenizer.json")
    for filename in ["generation_config.json", "config.json"]:
        path = source / filename
        if path.is_file():
            values = json.loads(path.read_text(encoding="utf-8"))
            for key, value in values.items():
                if hasattr(config, key) and getattr(config, key) is None:
                    setattr(config, key, value)
    config.active_vocab_size = bounded_active_vocab_size(
        len(tokenizer),
        model_config.vocab_size,
    )
    for key, value in MLC_CHAT_SYSTEM_DEFAULT.items():
        if getattr(config, key) is None:
            setattr(config, key, value)
    payload = config.model_dump(by_alias=True)
    if completion_prefix is not None:
        payload["drowse_completion_prefix_token_ids"] = completion_prefix
    (output / "mlc-chat-config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def completion_prefix_ids(tokenizer) -> list[int]:
    prefix = tokenizer.encode("", add_special_tokens=True)
    text = "I love marmots because"
    raw = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.encode(text, add_special_tokens=True) != prefix + raw:
        raise SystemExit("base tokenizer requires unsupported non-prefix special tokens")
    return prefix


def webllm_sliding_window_size(architecture: str, model_config) -> int:
    if architecture == "gemma3_text":
        return -1
    return getattr(model_config, "sliding_window_size", -1)


def browser_model_config(architecture: str, model_config) -> dict:
    result = model_config.asdict()
    if architecture == "gpt2":
        result["num_hidden_layers"] = model_config.n_layer
        result["hidden_size"] = model_config.n_embd
    return result


def bounded_active_vocab_size(tokenizer_size: int, model_vocab_size: int) -> int:
    return min(tokenizer_size, model_vocab_size)


def validate_weight_encoding(model: Path, quantization: str) -> None:
    if quantization != "q0f32":
        return
    cache = json.loads((model / "tensor-cache.json").read_text(encoding="utf-8"))
    parameters = [parameter for shard in cache.get("records", []) for parameter in shard.get("records", [])]
    if not parameters or any(item.get("dtype") != "float32" or item.get("format") != "raw" for item in parameters):
        raise SystemExit("q0f32 requires lossless raw float32 tensor storage")


def complete_config(output: Path, source: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(source), use_fast=True, local_files_only=True)
    special_ids = set(tokenizer.all_special_ids)
    special_ids.update(tokenizer.added_tokens_encoder.values())
    config_path = output / "mlc-chat-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "drowse_completion_prefix_token_ids" in config:
        eos_ids = source_eos_ids(source)
        config.update(base_special_token_config(tokenizer, eos_ids))
        special_ids.update(eos_ids)
        special_ids.add(config["pad_token_id"])
    config["drowse_capture_special_token_ids"] = sorted(special_ids)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def base_special_token_config(tokenizer, eos_ids):
    return {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": eos_ids[0] if len(eos_ids) == 1 else eos_ids,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0],
    }


def source_eos_ids(source: Path) -> list[int]:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") == "qwen3_5":
        config = dict(config["text_config"])
    generation = source / "generation_config.json"
    if generation.is_file():
        config.update(json.loads(generation.read_text(encoding="utf-8")))
    eos = config.get("eos_token_id")
    values = eos if isinstance(eos, list) else [eos]
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise SystemExit("base completion requires explicit source EOS token IDs")
    return sorted(set(values))


def validate_completion_policy(args, source: Path) -> None:
    if (args.model_type == "base") != (args.conv_template == "raw"):
        raise SystemExit("base models require --conv-template raw; chat models require a chat template")
    if args.architecture in BASE_MODEL_SOURCE_DIGESTS and args.model_type != "base":
        raise SystemExit("candidate base architectures require explicit base classification")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if args.architecture == "qwen3_5":
        config = config["text_config"]
    limit = config.get("n_positions") if args.architecture == "gpt2" else config.get("max_position_embeddings")
    if type(limit) is not int or limit <= 0 or args.context_window_size > limit:
        raise SystemExit("requested context exceeds the source model's verified position limit")
    if args.architecture == "gpt2" and args.context_window_size != limit:
        raise SystemExit("GPT-2 context must match its learned position embedding table")
    if args.prefill_chunk_size > args.context_window_size:
        raise SystemExit("prefill chunk exceeds the selected context window")
    if args.model_type == "base":
        source_eos_ids(source)


def copy_notices(source: Path, output: Path, source_repository: str | None = None) -> None:
    readme = source / "README.md"
    if readme.is_file():
        shutil.copyfile(readme, output / "README.source.md")
    license_id = model_card_license(readme)
    if license_id == "mit" and source_repository == "openai-community/gpt2":
        license_text = Path(__file__).resolve().parent / "licenses/GPT2-Modified-MIT.txt"
        shutil.copyfile(license_text, output / "LICENSE.model")
        (output / "MODEL-LICENSE.json").write_text(json.dumps({
            "schemaVersion": 1,
            "licenseId": "LicenseRef-GPT2-Modified-MIT",
            "modelCardLicense": license_id,
            "declaredIn": "README.md",
            "licenseText": "LICENSE.model",
            "licenseTextSource": "upstream-repository",
            "sourceUrl": "https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/LICENSE",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if license_id not in {"apache-2.0", "gemma"}:
        raise SystemExit("source model card declares an unsupported model license")
    source_license = source / "LICENSE"
    if source_license.is_file():
        license_text = source_license
        license_text_source = "source-snapshot"
    elif license_id == "apache-2.0":
        license_text = Path(__file__).resolve().parent / "licenses/Apache-2.0.txt"
        license_text_source = "drowse-standard-text"
    else:
        license_text = readme
        license_text_source = "model-card-terms-reference"
    if not license_text.is_file():
        raise SystemExit("the declared model license notice is unavailable")
    shutil.copyfile(license_text, output / "LICENSE.model")
    if license_id == "apache-2.0":
        record = {
            "schemaVersion": 1,
            "spdx": license_id,
            "declaredIn": "README.md",
            "licenseText": "LICENSE.model",
            "licenseTextSource": license_text_source,
        }
    else:
        record = {
            "schemaVersion": 1,
            "licenseId": license_id,
            "declaredIn": "README.md",
            "licenseText": "LICENSE.model",
            "licenseTextSource": license_text_source,
            "termsUrl": "https://ai.google.dev/gemma/terms",
        }
    (output / "MODEL-LICENSE.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def model_card_license(readme: Path) -> str | None:
    if not readme.is_file():
        return None
    lines = readme.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, separator, value = line.partition(":")
        if separator and key.strip() == "license":
            return value.strip().strip("'\"").lower()
    return None


def write_build_manifest(args, source: Path, output: Path) -> None:
    tokenizer_config = json.loads((source / "tokenizer_config.json").read_text(encoding="utf-8"))
    source_files = conversion_source_files(source)
    manifest = {
        "schemaVersion": 1,
        "runtimeAbi": "drowse-web-runtime-v1",
        "hookAbi": "post-block-residual-v4",
        "structuredHookProfile": args.structured_hook_profile,
        "thinkingProfile": derive_thinking_profile(source, args.architecture),
        "architecture": args.architecture,
        "quantization": args.quantization,
        "contextWindowSize": args.context_window_size,
        "prefillChunkSize": args.prefill_chunk_size,
        "source": {
            "repository": args.source_repository,
            "revision": args.source_revision,
            "files": [
                {"path": name, "bytes": (source / name).stat().st_size, "sha256": sha256(source / name)}
                for name in source_files
            ],
            "chatTemplateSha256": hashlib.sha256(
                str(tokenizer_config.get("chat_template", "")).encode("utf-8")
            ).hexdigest(),
        },
        "toolchain": toolchain_manifest(args.architecture),
    }
    if args.model_type == "base":
        manifest["modelType"] = "base"
        compiled_config = json.loads((output / "mlc-chat-config.json").read_text(encoding="utf-8"))
        manifest["promptPolicy"] = {
            "mode": "raw", "stopTokenIds": source_eos_ids(source),
            "prefixTokenIds": compiled_config["drowse_completion_prefix_token_ids"],
        }
    if args.architecture == "qwen3_5":
        manifest["stateAbi"] = "kv-rnn-v1"
    if args.quantization == "q0f32":
        manifest["weightEncoding"] = "raw"
    manifest["files"] = artifact_files(output)
    (output / "drowse-build.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def compile_library(args, mlc_repository: Path, tvm_repository: Path) -> None:
    model = args.model.resolve(strict=True)
    output = args.output.resolve()
    validate_new_output(output, directory=False)
    if not (model / "mlc-chat-config.json").is_file() or not (model / "drowse-build.json").is_file():
        raise SystemExit("model directory lacks verified Drowse build metadata")
    verify_build_manifest(
        model,
        args.architecture,
        args.quantization,
        args.structured_hook_profile,
    )
    verify_emcc()

    import tvm
    from mlc_llm.cli.compile import main as mlc_compile

    require_llvm(tvm)
    with tempfile.TemporaryDirectory(prefix=f".{output.stem}.", dir=output.parent) as staging:
        staged_output = Path(staging) / output.name
        mlc_compile(
            [
                str(model),
                "--quantization",
                args.quantization,
                "--model-type",
                args.architecture,
                "--device",
                "webgpu",
                "--output",
                str(staged_output),
            ]
        )
        verify_compiled_library(staged_output)
        os.replace(staged_output, output)
    print(
        json.dumps(
            {
                "architecture": args.architecture,
                "bytes": output.stat().st_size,
                "output": str(output),
                "quantization": args.quantization,
                "sha256": sha256(output),
            },
            sort_keys=True,
        )
    )


def verify_compiled_library(output: Path) -> None:
    wasm = output.read_bytes()
    if not wasm.startswith(b"\x00asm"):
        raise SystemExit("compiler did not produce a valid WebAssembly container")
    for function in REQUIRED_FUNCTIONS:
        encoded = function.encode()
        vm_function_marker = len(encoded).to_bytes(8, "little") + encoded
        if vm_function_marker not in wasm:
            raise SystemExit(f"compiled WebAssembly lacks Drowse VM function metadata: {function}")
    if b"local_top_k_index[-1i]" in wasm:
        raise SystemExit("compiled WebAssembly contains an out-of-bounds top-k shader store")
    if b"@compute @workgroup_size(1024" in wasm:
        raise SystemExit("dense launch-model WebAssembly contains an unexpected 1024-thread shader")
    verify_topk_shader_workgroups(wasm)
    verify_jlens_transport_shader(wasm)
    verify_geometry_shader_bindings(wasm)


def verify_topk_shader_workgroups(wasm: bytes) -> None:
    counts = {}
    for label, prefix, expected_bindings, required_fragments in (
        (
            "tile",
            TOPK_TILE_SHADER_PREFIX,
            3,
            (
                b"var local_values : array<f32, 8>;",
                b"var local_indices : array<i32, 8>;",
                b"rank < 8i",
                b"step < 256i",
            ),
        ),
        (
            "merge",
            TOPK_MERGE_SHADER_PREFIX,
            4,
            (
                b"var local_values : array<f32, 8>;",
                b"var local_indices : array<i32, 8>;",
                b"rank < 8i",
                b"candidate < podArgs.candidates_per_row",
            ),
        ),
    ):
        position = 0
        kernels = 0
        while True:
            start = wasm.find(prefix, position)
            if start < 0:
                break
            next_kernel = wasm.find(b"// Function:", start + len(prefix))
            end = next_kernel if next_kernel >= 0 else min(len(wasm), start + 131072)
            shader = wasm[start:end]
            header_end = shader.find(b"\n")
            if header_end < 0 or not shader[:header_end].endswith(b"_kernel"):
                raise SystemExit(f"compiled exact top-8 {label} shader name is invalid")
            workgroup = shader.find(b"@compute @workgroup_size(", 0, 2048)
            line_end = shader.find(b"\n", workgroup, 2048)
            if workgroup < 0 or line_end < 0 or shader[workgroup:line_end] != TOPK_WORKGROUP_LINE:
                raise SystemExit(f"compiled exact top-8 {label} shader is not single-thread tiled")
            bindings = shader[:workgroup].count(b"var<storage")
            if (
                bindings != expected_bindings
                or b"var<workgroup>" in shader
                or b"workgroupBarrier();" in shader
                or b"candidate_thread" in shader
                or b"array<f32, 2048>" in shader
                or b"array<i32, 2048>" in shader
                or any(fragment not in shader for fragment in required_fragments)
            ):
                raise SystemExit(
                    f"compiled exact top-8 {label} shader violates the barrier-free tiled schedule: "
                    f"bindings={bindings}"
                )
            kernels += 1
            position = end
        counts[label] = kernels
    if any(count == 0 for count in counts.values()):
        raise SystemExit(f"compiled WebAssembly lacks exact tiled top-8 shaders: {counts}")


def verify_jlens_transport_shader(wasm: bytes) -> None:
    position = 0
    kernels = 0
    while True:
        start = wasm.find(JLENS_TRANSPORT_SHADER_PREFIX, position)
        if start < 0:
            break
        next_kernel = wasm.find(
            b"// Function:", start + len(JLENS_TRANSPORT_SHADER_PREFIX)
        )
        end = next_kernel if next_kernel >= 0 else min(len(wasm), start + 131072)
        shader = wasm[start:end]
        workgroup = shader.find(b"@compute @workgroup_size(", 0, 2048)
        line_end = shader.find(b"\n", workgroup, 2048)
        bindings = shader[:workgroup].count(b"var<storage")
        if (
            workgroup < 0
            or line_end < 0
            or shader[workgroup:line_end] != JLENS_TRANSPORT_WORKGROUP_LINE
            or bindings != 3
            or b"var<workgroup>" in shader
            or b"workgroupBarrier();" in shader
            or b"fma(" not in shader
            or b"source_coordinate" not in shader
        ):
            raise SystemExit(
                "compiled J-lens transport shader violates the barrier-free fp32 schedule: "
                f"bindings={bindings}"
            )
        kernels += 1
        position = end
    if kernels == 0:
        raise SystemExit("compiled WebAssembly lacks the Drowse J-lens transport shader")


def verify_geometry_shader_bindings(wasm: bytes) -> None:
    position = 0
    kernels = []
    while True:
        start = wasm.find(GEOMETRY_SHADER_PREFIX, position)
        if start < 0:
            break
        line_end = wasm.find(b"\n", start)
        compute = wasm.find(b"@compute", line_end)
        if line_end < 0 or compute < 0:
            raise SystemExit("compiled geometry WebGPU shader metadata is truncated")
        name = wasm[start + len(b"// Function: ") : line_end].decode("ascii")
        bindings = wasm[line_end:compute].count(b"var<storage")
        kernels.append((name, bindings))
        position = compute + len(b"@compute")
    if not kernels:
        raise SystemExit("compiled WebAssembly lacks a Drowse geometry WebGPU shader")
    for name, bindings in kernels:
        if bindings != MAX_WEBGPU_STORAGE_BINDINGS_PER_STAGE:
            raise SystemExit(
                f"compiled WebGPU shader {name} uses {bindings} storage bindings; "
                f"expected {MAX_WEBGPU_STORAGE_BINDINGS_PER_STAGE}"
            )


def verify_build_manifest(
    model: Path,
    architecture: str,
    quantization: str,
    structured_hook_profile: str = STANDARD_STRUCTURED_HOOK_PROFILE,
) -> None:
    manifest = json.loads((model / "drowse-build.json").read_text(encoding="utf-8"))
    expected = {
        "schemaVersion": 1,
        "runtimeAbi": "drowse-web-runtime-v1",
        "hookAbi": "post-block-residual-v4",
        "structuredHookProfile": structured_hook_profile,
        "thinkingProfile": derive_thinking_profile(model, architecture),
        "architecture": architecture,
        "quantization": quantization,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise SystemExit(f"model build manifest has invalid {key}")
    if architecture == "qwen3_5" and manifest.get("stateAbi") != "kv-rnn-v1":
        raise SystemExit("Qwen 3.5 build manifest lacks its hybrid state ABI")
    toolchain = manifest.get("toolchain", {})
    expected_toolchain = toolchain_manifest(architecture)
    if toolchain != expected_toolchain:
        raise SystemExit("model build manifest does not match the active exact toolchain")
    if quantization == "q0f32":
        if manifest.get("weightEncoding") != "raw":
            raise SystemExit("q0f32 build manifest must declare lossless raw storage")
        validate_weight_encoding(model, quantization)
    if architecture in BASE_MODEL_SOURCE_DIGESTS and manifest.get("modelType") != "base":
        raise SystemExit("base model build manifest lacks explicit classification")
    if manifest.get("modelType") == "base":
        config = json.loads((model / "mlc-chat-config.json").read_text(encoding="utf-8"))
        conversation = config.get("conv_template", {})
        policy = manifest.get("promptPolicy", {})
        if (policy.get("mode") != "raw" or not policy.get("stopTokenIds")
                or conversation.get("name") != "drowse-base"
                or conversation.get("stop_token_ids") != policy["stopTokenIds"]):
            raise SystemExit("base model prompt policy differs from its compiled configuration")
        if "prefixTokenIds" in policy and (
            config.get("drowse_completion_prefix_token_ids") != policy["prefixTokenIds"]
            or conversation.get("system_prefix_token_ids") != policy["prefixTokenIds"]
        ):
            raise SystemExit("base model generation and capture prefixes disagree")
    declared = manifest.get("files")
    if not isinstance(declared, list) or not declared:
        raise SystemExit("model build manifest has no artifact closure")
    actual_paths = {
        str(path.relative_to(model))
        for path in model.rglob("*")
        if path.is_file() and path.name != "drowse-build.json"
    }
    declared_paths = {entry.get("path") for entry in declared if isinstance(entry, dict)}
    if declared_paths != actual_paths:
        raise SystemExit("model build manifest artifact closure differs from disk")
    for entry in declared:
        path = model / entry["path"]
        if path.stat().st_size != entry.get("bytes") or sha256(path) != entry.get("sha256"):
            raise SystemExit(f"model artifact failed verification: {entry['path']}")


def adapter_digests(architecture: str) -> dict:
    return HYBRID_ADAPTER_DIGESTS if architecture == "qwen3_5" else BASE_ADAPTER_DIGESTS


def toolchain_manifest(architecture: str) -> dict:
    result = {
        "mlcLlmBaseCommit": MLC_BASE_COMMIT,
        "mlcOverlayFiles": RESULT_DIGESTS,
        "tvmCommit": TVM_COMMIT,
        "tvmFfiCommit": TVM_FFI_COMMIT,
        "tvmOverlayFiles": TVM_RESULT_DIGESTS,
    }
    if architecture in BASE_MODEL_SOURCE_DIGESTS:
        result["modelAdapterFiles"] = adapter_digests(architecture)
        result["modelSourceFiles"] = BASE_MODEL_SOURCE_DIGESTS[architecture]
    return result


def derive_thinking_profile(model: Path, architecture: str) -> dict | None:
    if architecture in {"llama", "gemma3_text", "gpt2", "gpt_neox", "qwen3_5"}:
        return None
    if architecture != "qwen3":
        raise SystemExit(f"thinking profile is undefined for architecture {architecture}")
    tokenizer_path = model / "tokenizer.json"
    tokenizer_config_path = model / "tokenizer_config.json"
    if not tokenizer_path.is_file() or not tokenizer_config_path.is_file():
        raise SystemExit("Qwen3 thinking attestation requires tokenizer.json and tokenizer_config.json")
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    chat_template = tokenizer_config.get("chat_template")
    if (
        not isinstance(chat_template, str)
        or "enable_thinking" not in chat_template
        or any(delimiter not in chat_template for delimiter in THINKING_DELIMITERS)
    ):
        raise SystemExit("Qwen3 chat template does not expose verified thinking controls")
    added_tokens = tokenizer.get("added_tokens")
    if not isinstance(added_tokens, list):
        raise SystemExit("Qwen3 tokenizer has no added-token table for thinking delimiters")
    token_ids = []
    for delimiter in THINKING_DELIMITERS:
        matches = [
            token.get("id")
            for token in added_tokens
            if isinstance(token, dict) and token.get("content") == delimiter
        ]
        if (
            len(matches) != 1
            or not isinstance(matches[0], int)
            or isinstance(matches[0], bool)
            or matches[0] < 0
        ):
            raise SystemExit(f"Qwen3 tokenizer does not map {delimiter} to one exact token ID")
        token_ids.append(matches[0])
    if token_ids[0] == token_ids[1]:
        raise SystemExit("Qwen3 thinking delimiters share a token ID")
    return {
        "start": THINKING_DELIMITERS[0],
        "end": THINKING_DELIMITERS[1],
        "startsInThinking": False,
        "startTokenIds": [token_ids[0]],
        "endTokenIds": [token_ids[1]],
    }


def validate_new_output(output: Path, directory: bool) -> None:
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise SystemExit(f"output parent does not exist: {output.parent}")
    if not directory and output.suffix != ".wasm":
        raise SystemExit("compiled output must use the .wasm extension")


def verify_emcc() -> None:
    executable = shutil.which("emcc")
    if executable is None:
        raise SystemExit("emcc 3.1.56 is required in PATH")
    first_line = subprocess.check_output([executable, "--version"], text=True).splitlines()[0]
    if EMCC_VERSION not in first_line or EMCC_REVISION not in first_line:
        raise SystemExit(f"emcc must be exact version {EMCC_VERSION} ({EMCC_REVISION})")


def require_llvm(tvm) -> None:
    if tvm.target.codegen.llvm_version_major() != 18:
        raise SystemExit("TVM compiler must be built against LLVM 18")


def artifact_files(root: Path) -> list[dict]:
    return [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "drowse-build.json"
    ]


def output_summary(output: Path) -> dict:
    files = artifact_files(output)
    return {
        "bytes": sum(file["bytes"] for file in files),
        "files": len(files),
        "output": str(output),
        "manifestSha256": sha256(output / "drowse-build.json"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
