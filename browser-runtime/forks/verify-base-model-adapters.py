#!/usr/bin/env python3
"""Export candidate base architectures using the installed MLC/TVM toolchain."""

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlc-repository", type=Path, required=True)
    parser.add_argument("--architecture", choices=("gpt2", "gpt_neox", "qwen3_5"), required=True)
    parser.add_argument("--quantization", choices=("q0f32", "q4f16_1"), default="q0f32")
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument("--readout-golden", action="store_true")
    validation.add_argument("--webgpu-codegen", action="store_true")
    args = parser.parse_args()
    root = args.mlc_repository.resolve(strict=True)
    for name, relative in (("mlc_llm", "python/mlc_llm"), ("mlc_llm.model", "python/mlc_llm/model")):
        package = types.ModuleType(name)
        package.__path__ = [str(root / relative)]
        package.__package__ = name
        sys.modules[name] = package

    from mlc_llm.model.gpt2.gpt2_model import GPT2Config, GPT2LMHeadModel
    from mlc_llm.model.gpt_neox.gpt_neox_model import GPTNeoXConfig, GPTNeoXForCausalLM
    from base_model_adapters import CANDIDATE_MODELS, register_candidate_models

    if args.architecture == "gpt2":
        config = GPT2Config(vocab_size=64, n_embd=256 if args.webgpu_codegen else 32, n_layer=2, n_head=4,
                            layer_norm_epsilon=1e-5, context_window_size=64)
        native_class = GPT2LMHeadModel
    elif args.architecture == "gpt_neox":
        config = GPTNeoXConfig(vocab_size=64, hidden_size=256 if args.webgpu_codegen else 32, intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=4,
                              layer_norm_eps=1e-5, use_parallel_residual=True,
                              rotary_pct=0.25, context_window_size=64)
        native_class = GPTNeoXForCausalLM
    else:
        from mlc_llm.model.qwen35.qwen35_model import Qwen35Config, Qwen35LMHeadModel
        from qwen35_adapter import InstrumentedQwen35LMHeadModel
        config = Qwen35Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                              head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
                              linear_num_key_heads=2, linear_num_value_heads=2,
                              context_window_size=64)
        native_class = Qwen35LMHeadModel
        CANDIDATE_MODELS["qwen3_5"] = InstrumentedQwen35LMHeadModel

    native = native_class(config)
    candidate = CANDIDATE_MODELS[args.architecture](config)
    register_candidate_models(include_hybrid=args.architecture == "qwen3_5")
    from mlc_llm.model.model import MODELS
    assert MODELS[args.architecture].model is CANDIDATE_MODELS[args.architecture]
    native_parameters = {name: (tuple(map(str, value.shape)), value.dtype)
                         for name, value in native.named_parameters()}
    candidate_parameters = {name: (tuple(map(str, value.shape)), value.dtype)
                            for name, value in candidate.named_parameters()}
    assert candidate_parameters == native_parameters, "Adapter changed checkpoint parameter paths"
    from mlc_llm.quantization import QUANTIZATION, make_quantization_functions
    quantization = QUANTIZATION[args.quantization]
    candidate, _ = make_quantization_functions(CANDIDATE_MODELS[args.architecture])[
        quantization.kind](config, quantization)
    module, _, _ = candidate.export_tvm(spec=candidate.get_default_spec(), allow_extern=True)
    names = {var.name_hint for var in module.get_global_vars()}
    expected = {name for name in candidate.get_default_spec().method_names
                if name.startswith("drowse_")}
    assert len(expected) >= 25 and expected <= names, "Missing instrumentation exports"
    # Structural guards only; numerical parity needs a separate inference run.
    source = module.script()
    if args.architecture != "qwen3_5":
        assert "layer_norm" in source
    if args.architecture == "gpt2":
        assert "get_query_positions" in source
    if args.architecture == "qwen3_5":
        import tvm
        from tvm.relax.type import AnyType
        for name, spec in zip(candidate.get_default_spec().method_names,
                              candidate.get_default_spec().method_specs):
            if name.startswith("drowse_") and "paged_kv_cache" in spec.arg_names:
                assert "rnn_state" in spec.arg_names
                result = module[name].ret_ty
                assert isinstance(result, tvm.ir.TupleType)
                assert isinstance(result.fields[2], AnyType), name
        verify_hybrid_state()
    if args.readout_golden:
        if args.quantization != "q0f32" or args.architecture == "qwen3_5":
            raise SystemExit("The readout golden currently requires q0f32")
        verify_readout(candidate, args.architecture)
    if args.webgpu_codegen:
        verify_webgpu_codegen(candidate, config, args.architecture, args.quantization)
    print(json.dumps({"architecture": args.architecture, "quantization": args.quantization,
                      "validation": "webgpu-codegen-only" if args.webgpu_codegen else "compiler-export-only",
                      "parameter_count": len(native_parameters), "hook_exports": sorted(expected),
                      "adapter_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in (["qwen35_adapter.py", "hybrid_state_kernels.py"]
                                       if args.architecture == "qwen3_5" else ["base_model_adapters.py"])},
                      "readout_cpu_golden": args.readout_golden,
                      "webgpu_codegen": args.webgpu_codegen,
                      "real_model_inference": False}, indent=2))


def verify_webgpu_codegen(candidate, config, architecture, quantization):
    import importlib.util
    import tvm
    from tvm import relax
    from mlc_llm import compiler_pass as _compiler_pass  # noqa: F401 — registers the MLC pipeline
    from mlc_llm import op as op_ext
    from mlc_llm.interface.compiler_flags import IPCAllReduceStrategyType

    path = Path(__file__).with_name("compile-tiny-webgpu.py")
    spec = importlib.util.spec_from_file_location("candidate_codegen_helpers", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    target = tvm.target.Target({"kind": "webgpu", "host": {
        "kind": "llvm", "mtriple": "wasm32-unknown-unknown-wasm"}})
    with target:
        op_ext.enable(target=target, flashinfer=False, faster_transformer=False, cutlass=False)
        module, params, ext_mods = candidate.export_tvm(
            spec=candidate.get_default_spec(), allow_extern=True)
    helper.prepare_parameters(params)
    metadata = helper.create_metadata(architecture, config, params, quantization)
    metadata["kv_state_kind"] = "hybrid" if architecture == "qwen3_5" else "kv_cache"
    layers = config.n_layer if architecture == "gpt2" else config.num_hidden_layers
    pipeline = relax.get_pipeline("mlc_llm", target=target, flashinfer=False,
        cublas_gemm=False, faster_transformer=False,
        allreduce_strategy=IPCAllReduceStrategyType.NONE,
        variable_bounds={"total_seq_len": config.context_window_size,
            "seq_len": config.prefill_chunk_size, "batch_size": 1,
            "jlens_layers": layers, "sae_features": 32768},
        cuda_graph_symbolic_capture_hints={}, additional_tirs={},
        ext_mods=ext_mods, metadata=metadata, debug_dump=None)
    with tvm.transform.PassContext(config={"tirx.disable_cse_tir": True}):
        relax.build(module, target=target, relax_pipeline=pipeline, system_lib=True)
    print(f"{architecture}: WebGPU/wasm target code generation passed; not linked or executed")


def verify_hybrid_state():
    import numpy as np
    import tvm
    from tvm.runtime import tensor
    from hybrid_state_kernels import state_get, state_set

    scenarios = 0
    for shape in ((7,), (2, 3), (2, 3, 4)):
        for dtype in ("float32", "float16"):
            for capacity in (1, 2, 3, 7):
                module = tvm.tirx.build(tvm.IRModule({
                    "rnn_state_get_0": state_get(shape, dtype, 3, capacity, 0),
                    "rnn_state_set_0": state_set(shape, dtype, 3, capacity, 0),
                }), target=tvm.target.Target("llvm"))
                for selected in ([1], [2, 0], [2, 0, 1]):
                    expected = np.arange(3 * capacity * np.prod(shape), dtype=dtype).reshape(
                        3, capacity, *shape)
                    slots = np.asarray(selected, dtype="int32")
                    history = np.arange(len(slots), dtype="int32") % capacity
                    device_storage = tensor(expected.copy())
                    device_slots = tensor(slots)
                    output = tensor(np.zeros((len(slots), *shape), dtype=dtype))
                    for step in range(capacity * 2 + 1):
                        module["rnn_state_get_0"](device_storage, device_slots, tensor(history), output)
                        np.testing.assert_array_equal(output.numpy(), expected[slots, history])
                        updates = np.arange(len(slots) * np.prod(shape), dtype=dtype).reshape(
                            len(slots), *shape) - step - 3
                        module["rnn_state_set_0"](device_storage, device_slots, tensor(history), tensor(updates))
                        history = (history + 1) % capacity
                        expected[slots, history] = updates
                        np.testing.assert_array_equal(device_storage.numpy(), expected)
                    scenarios += 1
    print(f"Hybrid state: {scenarios} CPU gather/scatter scenarios passed (fp32/fp16, "
          "1D/2D/3D, singleton/partial/full batches, repeated ring wrap, untouched slots)")


def verify_readout(model, architecture):
    import numpy as np
    import tvm
    from tvm import relax
    from tvm.runtime import tensor

    specs = model.get_default_spec()
    selected = {name: spec for name, spec in zip(specs.method_names, specs.method_specs)
                if name in ("drowse_jlens_probabilities", "drowse_jlens_directions")}
    module, parameters, _ = model.export_tvm(spec=selected, allow_extern=True)
    rng = np.random.default_rng(482)
    arrays = {}
    for name, parameter in parameters:
        shape = tuple(64 if str(size) == "vocab_size" else int(size) for size in parameter.shape)
        arrays[name] = rng.normal(0, 0.1, shape).astype(parameter.dtype)
    norm_prefix = "transformer.ln_f" if architecture == "gpt2" else "gpt_neox.final_layer_norm"
    head_name = "lm_head.weight" if architecture == "gpt2" else "embed_out.weight"
    arrays[f"{norm_prefix}.weight"] += 1
    hidden = rng.normal(2, 1, (2, 32)).astype("float32")
    jacobians = rng.normal(0, 0.2, (2, 32, 32)).astype("float32")
    token_ids = np.arange(8, dtype="int32")
    layer_ids = np.asarray([1, 0], dtype="int32")
    transported = np.einsum("lij,lj->li", jacobians, hidden[layer_ids])
    centered = transported - transported.mean(axis=-1, keepdims=True)
    normalized = centered / np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + 1e-5)
    normalized = normalized * arrays[f"{norm_prefix}.weight"] + arrays[f"{norm_prefix}.bias"]
    logits = normalized @ arrays[head_name].T
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    # Execute the same transport kernel serially for the CPU reference check.
    # This does not validate its GPU scheduling or GPU numerical behavior.
    import importlib.util
    helper_spec = importlib.util.spec_from_file_location("hook_verification", Path(__file__).with_name("verify-mlc-hook.py"))
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    module = helper.strip_gpu_thread_bindings(module, tvm)
    vm = relax.VirtualMachine(relax.build(module, target=tvm.target.Target("llvm")), tvm.cpu())
    packed = [tensor(arrays[name], device=tvm.cpu()) for name, _ in parameters]
    actual = vm["drowse_jlens_probabilities"](
        tensor(hidden), tensor(jacobians), tensor(token_ids), tensor(layer_ids), packed).numpy()
    np.testing.assert_allclose(actual, probabilities[:, token_ids], atol=2e-6, rtol=2e-5)
    directions = vm["drowse_jlens_directions"](tensor(jacobians), tensor(token_ids), packed).numpy()
    np.testing.assert_allclose(directions, arrays[head_name][token_ids][None, :, :] @ jacobians,
                               atol=2e-6, rtol=2e-5)
    print(f"{architecture}: compiled LayerNorm J-lens probabilities and directions match NumPy")


if __name__ == "__main__":
    main()
