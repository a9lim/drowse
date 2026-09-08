#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
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
    "python/mlc_llm/model/drowse_hooks.py": "4fadef9db1c875c81ca9cde76dfdbc477a3755f4d982ba9d097ccc730f6778b1",
}
TVM_RESULT_DIGESTS = {
    "python/tvm/relax/frontend/nn/llm/_decode_kernels.py": "4cfba82db3cd92e0b02081bd9679f24663fdca327de767966ae50c135c92e3ce",
    "python/tvm/relax/frontend/nn/llm/_kernel_common.py": "01d9c8b9fd1fec50ad6ecc6e91b3ff33921c754a1c7c7f0f072b7c0b37c86690",
    "src/backend/webgpu/codegen/codegen_webgpu.cc": "753ff21703a741cf7a8bc49678f4737127f04fa1f28f0fef06aefb4859b5d4a7",
    "tests/python/codegen/test_target_codegen_webgpu.py": "3da351a5fd03a1a0b05ed926581fe3d46e5160c23f926f931b0a8776d3f2e9c9",
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


def main() -> None:
    args = parse_args()
    mlc_repository = args.mlc_repository.resolve(strict=True)
    tvm_repository = args.tvm_repository.resolve(strict=True)
    output = args.output.resolve()
    validate_output(output)
    require_commit(mlc_repository, MLC_BASE_COMMIT, "MLC-LLM")
    require_commit(tvm_repository, TVM_COMMIT, "TVM")
    require_commit(tvm_repository / "3rdparty/tvm-ffi", TVM_FFI_COMMIT, "TVM-FFI")
    verify_result_files(mlc_repository)
    verify_portable_topk_source(mlc_repository)
    verify_result_files(tvm_repository, TVM_RESULT_DIGESTS, "TVM WebGPU overlay")
    emcc = verify_emcc()
    verify_wasm_runtime(mlc_repository)
    ensure_import_roots(mlc_repository, tvm_repository)
    os.environ["MLC_LLM_SOURCE_DIR"] = str(mlc_repository)
    os.environ.setdefault("SKIP_LOADING_MLCLLM_SO", "1")

    try:
        import tvm
        from tvm import relax
        from tvm.s_tir.dlight.base import transform as dlight_transform

        bootstrap_mlc_package(mlc_repository)
        from mlc_llm import compiler_pass as _compiler_pass  # noqa: F401
        from mlc_llm import op as op_ext
        from mlc_llm.interface.compiler_flags import IPCAllReduceStrategyType
        from mlc_llm.model.gemma3.gemma3_model import (
            Gemma3Config,
            Gemma3ForCausalLM,
            Gemma3TextConfig,
        )
        from mlc_llm.model.llama.llama_model import LlamaConfig, LlamaForCausalLM
        from mlc_llm.model.qwen3.qwen3_model import Qwen3Config, Qwen3LMHeadModel
        from mlc_llm.quantization import QUANTIZATION, make_quantization_functions
    except (ImportError, OSError) as error:
        raise SystemExit(f"exact MLC/TVM compiler toolchain is unavailable: {error}") from error
    if tvm.target.codegen.llvm_version_major() != 18:
        raise SystemExit("TVM compiler must be built against LLVM 18")
    install_dlight_diagnostic(dlight_transform)

    target = tvm.target.Target(
        {
            "kind": "webgpu",
            "host": {
                "kind": "llvm",
                "mtriple": "wasm32-unknown-unknown-wasm",
            },
        }
    )
    with target:
        op_ext.enable(
            target=target,
            flashinfer=False,
            faster_transformer=False,
            cutlass=False,
        )
        model, model_type, config = create_model(
            args.architecture,
            Qwen3Config,
            Qwen3LMHeadModel,
            LlamaConfig,
            LlamaForCausalLM,
            Gemma3Config,
            Gemma3TextConfig,
            Gemma3ForCausalLM,
            args.quantization,
            QUANTIZATION,
            make_quantization_functions,
            args.context_window_size,
        )
        module, named_params, ext_mods = model.export_tvm(
            spec=model.get_default_spec(),
            allow_extern=True,
        )

    verify_exports(module, args.architecture)
    if args.ir_stats_only:
        functions = []
        for global_var in module.get_global_vars():
            name = global_var.name_hint
            function = module[global_var]
            functions.append({
                "name": name,
                "scriptBytes": len(function.script(show_meta=False).encode()),
                "type": type(function).__name__,
            })
        print(json.dumps({
            "architecture": args.architecture,
            "functionCount": len(functions),
            "functions": sorted(functions, key=lambda item: item["scriptBytes"], reverse=True),
            "quantization": args.quantization,
        }, sort_keys=True))
        return
    prepare_parameters(named_params)
    metadata = create_metadata(model_type, config, named_params, args.quantization)
    num_hidden_layers = (
        config.num_hidden_layers
        if hasattr(config, "num_hidden_layers")
        else config.text_config.num_hidden_layers
    )
    variable_bounds = {
        "total_seq_len": config.context_window_size,
        "seq_len": config.prefill_chunk_size,
        "batch_size": 1,
        "jlens_layers": num_hidden_layers,
        "sae_features": 16384,
    }
    cuda_graph_hints = {
        "batch_decode": ["batch_size"],
        "batch_decode_to_last_hidden_states": ["batch_size"],
        "batch_verify": ["batch_size", "seq_len"],
        "batch_verify_to_last_hidden_states": ["batch_size", "seq_len"],
    }
    pipeline = relax.get_pipeline(
        "mlc_llm",
        target=target,
        flashinfer=False,
        cublas_gemm=False,
        faster_transformer=False,
        allreduce_strategy=IPCAllReduceStrategyType.NONE,
        variable_bounds=variable_bounds,
        cuda_graph_symbolic_capture_hints=cuda_graph_hints,
        additional_tirs={},
        ext_mods=ext_mods,
        metadata=metadata,
        debug_dump=None,
    )
    runtime_bitcode = mlc_repository / "web/dist/wasm/mlc_wasm_runtime.bc"
    with tvm.transform.PassContext(config={"tirx.disable_cse_tir": True}):
        executable = relax.build(
            module,
            target=target,
            relax_pipeline=pipeline,
            system_lib=True,
        )
        executable.export_library(str(output), libs=[str(runtime_bitcode)])

    wasm = output.read_bytes()
    if len(wasm) == 0 or wasm[:4] != b"\x00asm":
        output.unlink(missing_ok=True)
        raise SystemExit("compiler did not produce a valid WebAssembly container")
    verify_wasm(output, wasm, emcc)
    digest = hashlib.sha256(wasm).hexdigest()
    print(
        json.dumps(
            {
                "architecture": args.architecture,
                "bytes": output.stat().st_size,
                "contextWindowSize": args.context_window_size,
                "output": str(output),
                "quantization": args.quantization,
                "sha256": digest,
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlc-repository", required=True, type=Path)
    parser.add_argument("--tvm-repository", required=True, type=Path)
    parser.add_argument(
        "--architecture", required=True, choices=["qwen3", "llama", "gemma3_text"]
    )
    parser.add_argument(
        "--quantization",
        choices=["q0f32", "q4f32_1", "q4f16_1"],
        default="q0f32",
    )
    parser.add_argument("--context-window-size", type=int, choices=[16, 2048, 4096], default=16)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ir-stats-only", action="store_true")
    return parser.parse_args()


def validate_output(output: Path) -> None:
    if output.suffix != ".wasm":
        raise SystemExit("output must use the .wasm extension")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    if not output.parent.is_dir():
        raise SystemExit(f"output parent does not exist: {output.parent}")


def require_commit(repository: Path, expected: str, label: str) -> None:
    actual = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise SystemExit(f"{label} checkout must be exact commit {expected}, found {actual}")


def verify_result_files(repository: Path, expected_files=RESULT_DIGESTS, label="MLC overlay") -> None:
    for relative_path, expected in expected_files.items():
        path = repository / relative_path
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
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


def verify_emcc() -> Path:
    executable = shutil.which("emcc")
    if executable is None:
        raise SystemExit("emcc 3.1.56 is required in PATH")
    try:
        output = subprocess.check_output([executable, "--version"], text=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit("emcc 3.1.56 is required in PATH") from error
    first_line = output.splitlines()[0] if output else ""
    if f" {EMCC_VERSION} " not in f" {first_line} " or EMCC_REVISION not in first_line:
        raise SystemExit(
            f"emcc must be exact version {EMCC_VERSION} ({EMCC_REVISION}), found: {first_line}"
        )
    return Path(executable).resolve()


def verify_wasm(output: Path, wasm: bytes, emcc: Path) -> None:
    for function in REQUIRED_FUNCTIONS:
        encoded = function.encode()
        vm_function_marker = len(encoded).to_bytes(8, "little") + encoded
        if vm_function_marker not in wasm:
            output.unlink(missing_ok=True)
            raise SystemExit(f"compiled WebAssembly lacks Drowse VM function metadata: {function}")
    if b"local_top_k_index[-1i]" in wasm:
        output.unlink(missing_ok=True)
        raise SystemExit("compiled WebAssembly contains an out-of-bounds top-k shader store")
    if b"@compute @workgroup_size(1024" in wasm:
        output.unlink(missing_ok=True)
        raise SystemExit("dense launch-model WebAssembly contains an unexpected 1024-thread shader")
    verify_topk_shader_workgroups(wasm)
    verify_jlens_transport_shader(wasm)
    verify_geometry_shader_bindings(wasm)
    wasm_opt = emcc.parents[1] / "bin/wasm-opt"
    if not wasm_opt.is_file():
        output.unlink(missing_ok=True)
        raise SystemExit(f"exact Emscripten Binaryen validator is missing: {wasm_opt}")
    validation = subprocess.run(
        [str(wasm_opt), str(output), "--enable-bulk-memory", "-o", os.devnull],
        text=True,
        capture_output=True,
        check=False,
    )
    if validation.returncode != 0:
        output.unlink(missing_ok=True)
        detail = validation.stderr.strip().splitlines()
        raise SystemExit("WebAssembly validation failed: " + (detail[-1] if detail else "unknown error"))


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


def verify_wasm_runtime(repository: Path) -> None:
    runtime = repository / "web/dist/wasm/mlc_wasm_runtime.bc"
    if not runtime.is_file() or runtime.stat().st_size == 0:
        raise SystemExit(
            "missing mlc_wasm_runtime.bc; run the exact MLC web/prep_emcc_deps.sh first"
        )


def ensure_import_roots(mlc_repository: Path, tvm_repository: Path) -> None:
    expected = [(tvm_repository / "python").resolve(), (mlc_repository / "python").resolve()]
    import_roots = {Path(path).resolve() for path in sys.path if path}
    missing = [str(path) for path in expected if path not in import_roots]
    if missing:
        raise SystemExit(
            "exact source roots are missing from PYTHONPATH: " + os.pathsep.join(missing)
        )


def bootstrap_mlc_package(mlc_repository: Path) -> None:
    if "mlc_llm" in sys.modules:
        raise SystemExit("mlc_llm was imported before the source-only compiler bootstrap")
    package = types.ModuleType("mlc_llm")
    package.__path__ = [str(mlc_repository / "python/mlc_llm")]
    package.__package__ = "mlc_llm"
    sys.modules["mlc_llm"] = package
    model_package = types.ModuleType("mlc_llm.model")
    model_package.__path__ = [str(mlc_repository / "python/mlc_llm/model")]
    model_package.__package__ = "mlc_llm.model"
    sys.modules["mlc_llm.model"] = model_package


def install_dlight_diagnostic(dlight_transform) -> None:
    original = dlight_transform._apply_rules

    def apply_with_context(func, target, rules, tunable):
        try:
            return original(func, target, rules, tunable)
        except Exception as error:
            symbol = func.attrs.get("global_symbol", "<anonymous>")
            body = func.script(show_meta=False)
            raise RuntimeError(
                f"WebGPU scheduling failed for TIR function {symbol}: {body[:1200]}"
            ) from error

    dlight_transform._apply_rules = apply_with_context


def create_model(
    architecture,
    qwen_config_class,
    qwen_model_class,
    llama_config_class,
    llama_model_class,
    gemma_config_class,
    gemma_text_config_class,
    gemma_model_class,
    quantization_name,
    quantization_registry,
    make_quantization_functions,
    context_window_size,
):
    common = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "vocab_size": 16,
        "context_window_size": context_window_size,
        "prefill_chunk_size": context_window_size,
        "head_dim": 32,
    }
    if architecture == "qwen3":
        config = qwen_config_class(
            hidden_act="silu",
            attention_bias=False,
            rope_theta=10000,
            **common,
        )
        model_class = qwen_model_class
        model_type = "qwen3"
    elif architecture == "llama":
        config = llama_config_class(**common)
        model_class = llama_model_class
        model_type = "llama"
    else:
        text_config = gemma_text_config_class(
            hidden_size=common["hidden_size"],
            intermediate_size=common["intermediate_size"],
            num_attention_heads=common["num_attention_heads"],
            num_hidden_layers=common["num_hidden_layers"],
            num_key_value_heads=common["num_key_value_heads"],
            rms_norm_eps=common["rms_norm_eps"],
            context_window_size=common["context_window_size"],
            prefill_chunk_size=common["prefill_chunk_size"],
            head_dim=common["head_dim"],
            sliding_window_size=common["context_window_size"],
            query_pre_attn_scalar=common["head_dim"],
        )
        config = gemma_config_class(
            text_config=text_config,
            vocab_size=common["vocab_size"],
            is_text_model=True,
        )
        model_class = gemma_model_class
        model_type = "gemma3_text"
    quantization = quantization_registry[quantization_name]
    quantize = make_quantization_functions(model_class)[quantization.kind]
    model, _ = quantize(config, quantization)
    return model, model_type, config


def verify_exports(module, architecture: str) -> None:
    names = {global_var.name_hint for global_var in module.get_global_vars()}
    missing = sorted(REQUIRED_FUNCTIONS - names)
    if missing:
        raise SystemExit(f"{architecture} export lacks Drowse functions: {', '.join(missing)}")


def prepare_parameters(named_params) -> None:
    for _, parameter in named_params:
        parameter.attrs["preprocs"] = list(parameter.attrs.get("preprocs", []))
        parameter.attrs["pipeline_stages"] = [0]


def create_metadata(model_type, config, named_params, quantization):
    def parameter_metadata(name, parameter):
        return {
            "name": name,
            "shape": [dimension if isinstance(dimension, int) else dimension.name for dimension in parameter.shape],
            "dtype": str(parameter.dtype),
            "preprocs": parameter.attrs["preprocs"],
            "pipeline_stages": parameter.attrs["pipeline_stages"],
        }

    return {
        "model_type": model_type,
        "quantization": quantization,
        "context_window_size": config.context_window_size,
        "sliding_window_size": getattr(config, "sliding_window_size", -1),
        "attention_sink_size": getattr(config, "attention_sink_size", -1),
        "prefill_chunk_size": config.prefill_chunk_size,
        "tensor_parallel_shards": config.tensor_parallel_shards,
        "pipeline_parallel_stages": getattr(config, "pipeline_parallel_stages", 1),
        "disaggregation": getattr(config, "disaggregation", False),
        "kv_state_kind": "kv_cache",
        "max_batch_size": 1,
        "active_vocab_size": None,
        "model_task": "chat",
        "params": [parameter_metadata(name, parameter) for name, parameter in named_params],
    }


if __name__ == "__main__":
    main()
