#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

MLC_BASE_COMMIT = "9fa644f54b04983adea4d0168f49fc6af4a893ba"
TVM_COMMIT = "68ba2b31c2a6d202fcd44e5780ef10ce08721dd5"
TVM_FFI_COMMIT = "12dbf053b3d9ba4ebd9da3123b1aeca79cf74229"
RESULT_DIGESTS = {
    "python/mlc_llm/interface/convert_weight.py": "dffc4af8487f54087bc27f9a35f959df318a70c4c33326b65d4a6d9f6479a1b5",
    "python/mlc_llm/interface/compile.py": "435876d96e9ba6fbc3f02dd6f5ed858df559ebded169ce91c5cdd8151e610d32",
    "python/mlc_llm/op/moe_misc.py": "4141be404d6d31b1ee152ef495fd5bdfc75b16af968ad01e07c3d1d26637ef92",
    "python/mlc_llm/model/llama/llama_model.py": "2a34402c81074ce0a773f9a73dc95aea20a7a0533b2561708d32ceeb72aec7dc",
    "python/mlc_llm/model/qwen3/qwen3_model.py": "0621190a80fc12ec58ca3fd5d439411135815c4f4f6596a5a0d258475b5c7291",
    "python/mlc_llm/model/gemma3/gemma3_model.py": "fd219c4779b1a7497fa533b264d5c77a973c5a1f46f41ed29923c312d6220f18",
    "python/mlc_llm/model/drowse_hooks.py": "4fadef9db1c875c81ca9cde76dfdbc477a3755f4d982ba9d097ccc730f6778b1",
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


def main() -> None:
    args = parse_args()
    mlc_repository = args.mlc_repository.resolve(strict=True)
    tvm_repository = args.tvm_repository.resolve(strict=True)
    fixture_path = args.fixture.resolve(strict=True)
    require_commit(mlc_repository, MLC_BASE_COMMIT, "MLC-LLM")
    require_commit(tvm_repository, TVM_COMMIT, "TVM")
    require_commit(tvm_repository / "3rdparty/tvm-ffi", TVM_FFI_COMMIT, "TVM-FFI")
    verify_result_files(mlc_repository)
    verify_portable_topk_source(mlc_repository)
    verify_exact_parallel_topk_cpu_golden()
    ensure_import_roots(mlc_repository, tvm_repository)
    os.environ.setdefault("SKIP_LOADING_MLCLLM_SO", "1")

    try:
        import tvm
        from tvm import relax
        from tvm.relax.frontend import nn
        from tvm.relax.frontend.nn import op, spec
        from tvm.runtime import tensor as tvm_tensor
        bootstrap_mlc_package(mlc_repository)
        from mlc_llm import op as op_ext
        from mlc_llm.model.gemma3.gemma3_model import (
            Gemma3Config,
            Gemma3ForCausalLM,
            Gemma3TextConfig,
        )
        from mlc_llm.model.llama.llama_model import LlamaConfig, LlamaForCausalLM
        from mlc_llm.model.qwen3.qwen3_model import Qwen3Config, Qwen3LMHeadModel
        from mlc_llm.model.drowse_hooks import (
            accumulate_jlens_readout,
            accumulate_sae_jump_relu_readout,
            accumulate_sae_readout,
            apply_rank_one_hook,
            apply_structured_affine_hook,
            apply_structured_curved_hook,
            capture_residual_positions,
            finalize_jlens_readout,
            measure_structured_geometry,
            pack_structured_curve_parameters,
            structured_geometry_payload_layout,
            structured_hook_profile_descriptor,
        )
    except (ImportError, OSError) as error:
        raise SystemExit(f"exact MLC/TVM toolchain is unavailable: {error}") from error
    if tvm.target.codegen.llvm_version_major() != 18:
        raise SystemExit("TVM compiler must be built against LLVM 18")

    if args.export_only is not None:
        model, label = create_tiny_model(
            args.export_only,
            Qwen3Config,
            Qwen3LMHeadModel,
            LlamaConfig,
            LlamaForCausalLM,
            Gemma3Config,
            Gemma3TextConfig,
            Gemma3ForCausalLM,
        )
        verify_model_exports(model, label, tvm)
        print(f"{label} exact-readout exports verified")
        sys.stdout.flush()
        os._exit(0)

    if args.readout_only:
        verify_readout_discovery_goldens(
            nn,
            spec,
            relax,
            tvm,
            tvm_tensor,
            op_ext.moe_misc.gating_topk,
            accumulate_jlens_readout,
            finalize_jlens_readout,
            accumulate_sae_jump_relu_readout,
            accumulate_sae_readout,
        )

    verify_isolated_model_exports(args)
    verify_exact_readout_profile(
        nn,
        spec,
        relax,
        tvm,
        tvm_tensor,
        structured_hook_profile_descriptor,
    )
    fixture = json.loads(fixture_path.read_text())
    if fixture.get("hookAbi") != "post-block-residual-v4" or fixture.get("hiddenSize") != 4:
        raise SystemExit("unexpected rank-one hook fixture identity")

    class HookModule(nn.Module):
        def forward(
            self,
            residual,
            enabled,
            basis,
            neutral,
            target,
            along,
            collapse,
            probe_basis,
            probe_neutral,
        ):
            return apply_rank_one_hook(
                residual,
                0,
                enabled,
                basis,
                neutral,
                target,
                along,
                collapse,
                probe_basis,
                probe_neutral,
            )

    hidden_size = fixture["hiddenSize"]
    module, _, _ = HookModule().export_tvm(
        spec={
            "forward": {
                "residual": spec.Tensor([1, 1, hidden_size], "float32"),
                "enabled": spec.Tensor([1], "uint32"),
                "basis": spec.Tensor([1, hidden_size], "float32"),
                "neutral": spec.Tensor([1, hidden_size], "float32"),
                "target": spec.Tensor([1], "float32"),
                "along": spec.Tensor([1], "float32"),
                "collapse": spec.Tensor([1], "float32"),
                "probe_basis": spec.Tensor([1, hidden_size], "float32"),
                "probe_neutral": spec.Tensor([1, hidden_size], "float32"),
            }
        },
        allow_extern=True,
    )
    executable = relax.build(module, target=tvm.target.Target("c"))
    vm = relax.VirtualMachine(executable, tvm.cpu())

    for case in fixture["cases"]:
        arrays = [
            np.asarray(case["residual"], dtype="float32").reshape(1, 1, hidden_size),
            np.asarray([1], dtype="uint32"),
            np.asarray(case["basis"], dtype="float32").reshape(1, hidden_size),
            np.asarray(case["neutral"], dtype="float32").reshape(1, hidden_size),
            np.asarray([case["target"]], dtype="float32"),
            np.asarray([case["along"]], dtype="float32"),
            np.asarray([case["collapse"]], dtype="float32"),
            np.asarray(case["probeBasis"], dtype="float32").reshape(1, hidden_size),
            np.asarray(case["probeNeutral"], dtype="float32").reshape(1, hidden_size),
        ]
        result, probe = vm["forward"](*(tvm_tensor(value, device=tvm.cpu()) for value in arrays))
        np.testing.assert_allclose(
            result.numpy().reshape(hidden_size),
            np.asarray(case["expectedResidual"], dtype="float32"),
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"{case['name']} residual",
        )
        np.testing.assert_allclose(
            probe.numpy().reshape(-1),
            np.asarray([case["expectedProbe"]], dtype="float32"),
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"{case['name']} probe",
        )

    class CaptureModule(nn.Module):
        def forward(self, residual, positions):
            return capture_residual_positions(residual, positions)

    capture_module, _, _ = CaptureModule().export_tvm(
        spec={
            "forward": {
                "residual": spec.Tensor([1, 3, hidden_size], "float16"),
                "positions": spec.Tensor([2], "int32"),
            }
        },
        allow_extern=True,
    )
    capture_executable = relax.build(capture_module, target=tvm.target.Target("llvm"))
    capture_vm = relax.VirtualMachine(capture_executable, tvm.cpu())
    residual = np.arange(3 * hidden_size, dtype="float16").reshape(1, 3, hidden_size)
    positions = np.asarray([0, 2], dtype="int32")
    capture = capture_vm["forward"](
        tvm_tensor(residual, device=tvm.cpu()),
        tvm_tensor(positions, device=tvm.cpu()),
    )
    if str(capture.dtype) != "float32" or list(capture.shape) != [1, 2, hidden_size]:
        raise SystemExit("selective residual capture returned an unexpected dtype or shape")
    np.testing.assert_array_equal(
        capture.numpy(),
        residual[:, [0, 2], :].astype("float32"),
    )

    verify_structured_hook_golden(
        nn,
        spec,
        relax,
        tvm,
        tvm_tensor,
        op,
        apply_structured_affine_hook,
        apply_structured_curved_hook,
        pack_structured_curve_parameters,
    )
    verify_packed_geometry_golden(
        nn,
        spec,
        relax,
        tvm,
        tvm_tensor,
        measure_structured_geometry,
        structured_geometry_payload_layout,
    )
    verify_isolated_readout_goldens(args)
    print(
        "exact-source MLC rank-one, structured, packed geometry, "
        "multidimensional/periodic/sphere curved, exact J-lens/SAE top-8, "
        "capture, and compiled C/LLVM golden checks passed"
    )


def verify_readout_discovery_goldens(
    nn,
    spec,
    relax,
    tvm,
    tvm_tensor,
    topk,
    accumulate_jlens_readout,
    finalize_jlens_readout,
    accumulate_sae_jump_relu_readout,
    accumulate_sae_readout,
) -> None:
    model_layers = 4
    vocab_size = 16
    fitted_layers = 3

    class JlensReadoutModule(nn.Module):
        def forward(
            self,
            probabilities,
            layer_ids,
            probability_sum,
            depth_sum,
            depth_square_sum,
            fitted_layer_count,
        ):
            sums = accumulate_jlens_readout(
                probabilities,
                layer_ids,
                probability_sum,
                depth_sum,
                depth_square_sum,
                model_layers,
            )
            token_ids, stats = finalize_jlens_readout(
                *sums,
                fitted_layer_count,
                topk,
            )
            layer_top_probabilities, layer_top_token_ids = topk(probabilities, 8)
            return (
                *sums,
                layer_top_token_ids,
                layer_top_probabilities,
                token_ids,
                stats,
            )

    jlens_module, _, _ = JlensReadoutModule().export_tvm(
        spec={
            "forward": {
                "probabilities": spec.Tensor([fitted_layers, vocab_size], "float32"),
                "layer_ids": spec.Tensor([fitted_layers], "int32"),
                "probability_sum": spec.Tensor([vocab_size], "float32"),
                "depth_sum": spec.Tensor([vocab_size], "float32"),
                "depth_square_sum": spec.Tensor([vocab_size], "float32"),
                "fitted_layer_count": spec.Tensor([1], "int32"),
            }
        },
        allow_extern=True,
    )
    jlens_module = strip_gpu_thread_bindings(jlens_module, tvm)
    jlens_vm = relax.VirtualMachine(
        relax.build(jlens_module, target=tvm.target.Target("c")), tvm.cpu()
    )
    probabilities = np.asarray(
        [
            np.linspace(0.001, 0.031, vocab_size),
            np.linspace(0.034, 0.004, vocab_size),
            np.asarray(
                [
                    0.006,
                    0.012,
                    0.003,
                    0.044,
                    0.005,
                    0.038,
                    0.007,
                    0.021,
                    0.009,
                    0.032,
                    0.011,
                    0.027,
                    0.013,
                    0.024,
                    0.015,
                    0.019,
                ]
            ),
        ],
        dtype="float32",
    )
    layer_ids = np.asarray([0, 2, 3], dtype="int32")
    zero = np.zeros(vocab_size, dtype="float32")
    actual = jlens_vm["forward"](
        tvm_tensor(probabilities, device=tvm.cpu()),
        tvm_tensor(layer_ids, device=tvm.cpu()),
        tvm_tensor(zero, device=tvm.cpu()),
        tvm_tensor(zero, device=tvm.cpu()),
        tvm_tensor(zero, device=tvm.cpu()),
        tvm_tensor(np.asarray([fitted_layers], dtype="int32"), device=tvm.cpu()),
    )
    probability_sum = probabilities.sum(axis=0)
    depths = layer_ids.astype("float32") / (model_layers - 1)
    depth_sum = (probabilities * depths[:, None]).sum(axis=0)
    depth_square_sum = (probabilities * depths[:, None] ** 2).sum(axis=0)
    expected_ids = np.argsort(-probability_sum, kind="stable")[:8]
    expected_strength = probability_sum[expected_ids] / fitted_layers
    expected_center = depth_sum[expected_ids] / probability_sum[expected_ids]
    expected_spread = np.sqrt(
        np.maximum(
            depth_square_sum[expected_ids] / probability_sum[expected_ids]
            - expected_center**2,
            0,
        )
    )
    np.testing.assert_allclose(actual[0].numpy(), probability_sum, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual[1].numpy(), depth_sum, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual[2].numpy(), depth_square_sum, rtol=1e-6, atol=1e-7)
    expected_layer_ids = np.argsort(-probabilities, axis=1, kind="stable")[:, :8]
    np.testing.assert_array_equal(actual[3].numpy(), expected_layer_ids)
    np.testing.assert_allclose(
        actual[4].numpy(),
        np.take_along_axis(probabilities, expected_layer_ids, axis=1),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_array_equal(actual[5].numpy().reshape(-1), expected_ids)
    np.testing.assert_allclose(
        actual[6].numpy(),
        np.stack([expected_strength, expected_center, expected_spread]),
        rtol=1e-5,
        atol=1e-6,
    )

    hidden_size = 4
    feature_counts = (9, 11)

    class SaeReadoutModule(nn.Module):
        def forward(
            self,
            hidden_states,
            encoder_a,
            encoder_bias_a,
            encoder_b,
            encoder_bias_b,
            decoder_bias,
            layer_id,
            offset_a,
            offset_b,
            prior_values,
            prior_feature_ids,
        ):
            values, feature_ids = accumulate_sae_readout(
                hidden_states,
                encoder_a,
                encoder_bias_a,
                decoder_bias,
                layer_id,
                offset_a,
                prior_values,
                prior_feature_ids,
                topk,
            )
            return accumulate_sae_readout(
                hidden_states,
                encoder_b,
                encoder_bias_b,
                decoder_bias,
                layer_id,
                offset_b,
                values,
                feature_ids,
                topk,
            )

    sae_module, _, _ = SaeReadoutModule().export_tvm(
        spec={
            "forward": {
                "hidden_states": spec.Tensor([model_layers, hidden_size], "float32"),
                "encoder_a": spec.Tensor([hidden_size, feature_counts[0]], "float32"),
                "encoder_bias_a": spec.Tensor([feature_counts[0]], "float32"),
                "encoder_b": spec.Tensor([hidden_size, feature_counts[1]], "float32"),
                "encoder_bias_b": spec.Tensor([feature_counts[1]], "float32"),
                "decoder_bias": spec.Tensor([hidden_size], "float32"),
                "layer_id": spec.Tensor([1], "int32"),
                "offset_a": spec.Tensor([1], "int32"),
                "offset_b": spec.Tensor([1], "int32"),
                "prior_values": spec.Tensor([1, 8], "float32"),
                "prior_feature_ids": spec.Tensor([1, 8], "int32"),
            }
        },
        allow_extern=True,
    )
    sae_module = strip_gpu_thread_bindings(sae_module, tvm)
    sae_vm = relax.VirtualMachine(
        relax.build(sae_module, target=tvm.target.Target("c")), tvm.cpu()
    )
    hidden_states = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.4],
            [0.5, 0.25, -0.75, 1.25],
            [-1.0, 0.75, 0.5, -0.25],
            [0.2, 0.4, 0.6, 0.8],
        ],
        dtype="float32",
    )
    decoder_bias = np.asarray([0.05, -0.1, 0.2, 0.0], dtype="float32")
    encoder_a = (np.arange(hidden_size * feature_counts[0], dtype="float32").reshape(
        hidden_size, feature_counts[0]
    ) - 17) / 19
    encoder_b = (np.arange(hidden_size * feature_counts[1], dtype="float32").reshape(
        hidden_size, feature_counts[1]
    )[::-1] - 21) / 17
    encoder_bias_a = np.linspace(-0.15, 0.11, feature_counts[0], dtype="float32")
    encoder_bias_b = np.linspace(0.13, -0.17, feature_counts[1], dtype="float32")
    selected_layer = np.asarray([2], dtype="int32")
    actual_values, actual_ids = sae_vm["forward"](
        tvm_tensor(hidden_states, device=tvm.cpu()),
        tvm_tensor(encoder_a, device=tvm.cpu()),
        tvm_tensor(encoder_bias_a, device=tvm.cpu()),
        tvm_tensor(encoder_b, device=tvm.cpu()),
        tvm_tensor(encoder_bias_b, device=tvm.cpu()),
        tvm_tensor(decoder_bias, device=tvm.cpu()),
        tvm_tensor(selected_layer, device=tvm.cpu()),
        tvm_tensor(np.asarray([0], dtype="int32"), device=tvm.cpu()),
        tvm_tensor(np.asarray([feature_counts[0]], dtype="int32"), device=tvm.cpu()),
        tvm_tensor(np.full((1, 8), -1e30, dtype="float32"), device=tvm.cpu()),
        tvm_tensor(np.zeros((1, 8), dtype="int32"), device=tvm.cpu()),
    )
    centered = hidden_states[selected_layer[0]] - decoder_bias
    activations = np.concatenate(
        [
            np.maximum(centered @ encoder_a + encoder_bias_a, 0),
            np.maximum(centered @ encoder_b + encoder_bias_b, 0),
        ]
    )
    expected_ids = np.argsort(-activations, kind="stable")[:8]
    np.testing.assert_array_equal(actual_ids.numpy().reshape(-1), expected_ids)
    np.testing.assert_allclose(
        actual_values.numpy().reshape(-1),
        activations[expected_ids],
        rtol=1e-6,
        atol=1e-7,
    )

    class JumpSaeReadoutModule(nn.Module):
        def forward(
            self,
            hidden_states,
            encoder,
            encoder_bias,
            encoder_threshold,
            decoder_bias,
            layer_id,
            feature_offset,
            prior_values,
            prior_feature_ids,
        ):
            return accumulate_sae_jump_relu_readout(
                hidden_states,
                encoder,
                encoder_bias,
                encoder_threshold,
                decoder_bias,
                layer_id,
                feature_offset,
                prior_values,
                prior_feature_ids,
                topk,
            )

    jump_module, _, _ = JumpSaeReadoutModule().export_tvm(
        spec={
            "forward": {
                "hidden_states": spec.Tensor([model_layers, hidden_size], "float32"),
                "encoder": spec.Tensor([hidden_size, feature_counts[0]], "float32"),
                "encoder_bias": spec.Tensor([feature_counts[0]], "float32"),
                "encoder_threshold": spec.Tensor([feature_counts[0]], "float32"),
                "decoder_bias": spec.Tensor([hidden_size], "float32"),
                "layer_id": spec.Tensor([1], "int32"),
                "feature_offset": spec.Tensor([1], "int32"),
                "prior_values": spec.Tensor([1, 8], "float32"),
                "prior_feature_ids": spec.Tensor([1, 8], "int32"),
            }
        },
        allow_extern=True,
    )
    jump_module = strip_gpu_thread_bindings(jump_module, tvm)
    jump_vm = relax.VirtualMachine(
        relax.build(jump_module, target=tvm.target.Target("c")), tvm.cpu()
    )
    jump_threshold = np.linspace(0.05, 0.45, feature_counts[0], dtype="float32")
    jump_values, jump_ids = jump_vm["forward"](
        tvm_tensor(hidden_states, device=tvm.cpu()),
        tvm_tensor(encoder_a, device=tvm.cpu()),
        tvm_tensor(encoder_bias_a, device=tvm.cpu()),
        tvm_tensor(jump_threshold, device=tvm.cpu()),
        tvm_tensor(decoder_bias, device=tvm.cpu()),
        tvm_tensor(selected_layer, device=tvm.cpu()),
        tvm_tensor(np.asarray([0], dtype="int32"), device=tvm.cpu()),
        tvm_tensor(np.full((1, 8), -1e30, dtype="float32"), device=tvm.cpu()),
        tvm_tensor(np.zeros((1, 8), dtype="int32"), device=tvm.cpu()),
    )
    jump_pre = centered @ encoder_a + encoder_bias_a
    jump_activations = np.where(jump_pre > jump_threshold, jump_pre, 0)
    jump_expected_ids = np.argsort(-jump_activations, kind="stable")[:8]
    np.testing.assert_array_equal(
        jump_ids.numpy().reshape(-1),
        jump_expected_ids,
    )
    np.testing.assert_allclose(
        jump_values.numpy().reshape(-1),
        jump_activations[jump_expected_ids],
        rtol=1e-6,
        atol=1e-7,
    )
    print("isolated exact J-lens/SAE top-8 compiled C golden checks passed")
    sys.stdout.flush()
    os._exit(0)


def verify_exact_readout_profile(
    nn,
    spec,
    relax,
    tvm,
    tvm_tensor,
    structured_hook_profile_descriptor,
) -> None:
    class ProfileModule(nn.Module):
        def forward(self):
            return structured_hook_profile_descriptor(2, 4)

    module, _, _ = ProfileModule().export_tvm(
        spec={"forward": {"$": {"param_mode": "none", "effect_mode": "none"}}},
        allow_extern=True,
    )
    vm = relax.VirtualMachine(
        relax.build(module, target=tvm.target.Target("c")), tvm.cpu()
    )
    descriptor = vm["forward"]().numpy()
    if descriptor.shape != (31,):
        raise SystemExit("Drowse exact-readout profile descriptor has the wrong length")
    np.testing.assert_array_equal(descriptor[:7], [0x53414B4C, 3, 3, 4, 3, 2, 4])
    np.testing.assert_array_equal(descriptor[-3:], [1, 8, 16384])


def verify_structured_hook_golden(
    nn,
    spec,
    relax,
    tvm,
    tvm_tensor,
    op,
    apply_structured_affine_hook,
    apply_structured_curved_hook,
    pack_structured_curve_parameters,
) -> None:
    hidden_size = 2
    groups = 4
    rank = 8
    probes = 8
    curves = 4
    nodes = 32
    intrinsic = 4
    embed = 8

    class StructuredModule(nn.Module):
        def forward(
            self,
            residual,
            affine_active,
            affine_basis,
            affine_neutral,
            affine_target,
            affine_along,
            affine_kappa,
            probe_kind,
            probe_direction,
            probe_bias,
            probe_threshold,
        ):
            return apply_structured_affine_hook(
                residual,
                0,
                affine_active,
                affine_basis,
                affine_neutral,
                affine_target,
                affine_along,
                affine_kappa,
                probe_kind,
                probe_direction,
                probe_bias,
                probe_threshold,
            )

    structured_module, _, _ = StructuredModule().export_tvm(
        spec={
            "forward": {
                "residual": spec.Tensor([1, 1, hidden_size], "float32"),
                "affine_active": spec.Tensor([1, groups], "float32"),
                "affine_basis": spec.Tensor([1, groups, rank, hidden_size], "float32"),
                "affine_neutral": spec.Tensor([1, groups, hidden_size], "float32"),
                "affine_target": spec.Tensor([1, groups, rank], "float32"),
                "affine_along": spec.Tensor([1, groups], "float32"),
                "affine_kappa": spec.Tensor([1, groups, rank], "float32"),
                "probe_kind": spec.Tensor([1, probes], "uint32"),
                "probe_direction": spec.Tensor([1, probes, hidden_size], "float32"),
                "probe_bias": spec.Tensor([1, probes], "float32"),
                "probe_threshold": spec.Tensor([1, probes], "float32"),
            }
        },
        allow_extern=True,
    )
    structured_module = strip_gpu_thread_bindings(structured_module, tvm)
    structured_vm = relax.VirtualMachine(
        relax.build(structured_module, target=tvm.target.Target("c")),
        tvm.cpu(),
    )
    affine_active = np.zeros((1, groups), dtype="float32")
    affine_active[0, 0] = 1
    affine_basis = np.zeros((1, groups, rank, hidden_size), dtype="float32")
    affine_basis[0, 0, 0] = [1, 0]
    affine_basis[0, 0, 1] = [0, 1]
    affine_target = np.zeros((1, groups, rank), dtype="float32")
    affine_target[0, 0, :2] = [1, 2]
    affine_along = np.zeros((1, groups), dtype="float32")
    affine_along[0, 0] = 0.5
    affine_kappa = np.zeros((1, groups, rank), dtype="float32")
    affine_kappa[0, 0, :2] = [1, 1]
    probe_kind = np.zeros((1, probes), dtype="uint32")
    probe_kind[0, :4] = [1, 2, 3, 4]
    probe_direction = np.zeros((1, probes, hidden_size), dtype="float32")
    probe_direction[0, 0] = [1, 0]
    probe_direction[0, 1] = [1, 1]
    probe_direction[0, 3] = [0, 1]
    probe_bias = np.zeros((1, probes), dtype="float32")
    probe_bias[0, 1] = -1.5
    probe_threshold = np.zeros((1, probes), dtype="float32")
    probe_threshold[0, 3] = 1.0
    affine_neutral = np.zeros((1, groups, hidden_size), dtype="float32")
    values = [
        np.asarray([[[0.2, 0.4]]], dtype="float32"),
        affine_active,
        affine_basis,
        affine_neutral,
        affine_target,
        affine_along,
        affine_kappa,
        probe_kind,
        probe_direction,
        probe_bias,
        probe_threshold,
    ]
    residual, measurements = structured_vm["forward"](
        *(tvm_tensor(value, device=tvm.cpu()) for value in values)
    )
    np.testing.assert_allclose(
        residual.numpy().reshape(hidden_size), [0.6, 1.2], rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        measurements.numpy()[:2], [0.6, 0.3], rtol=1e-6, atol=1e-7
    )
    np.testing.assert_array_equal(measurements.numpy()[2:3], [0.0])
    np.testing.assert_allclose(measurements.numpy()[3:4], [1.2], rtol=1e-6, atol=1e-7)
    blocked_values = list(values)
    blocked_threshold = probe_threshold.copy()
    blocked_threshold[0, 3] = 1.3
    blocked_values[10] = blocked_threshold
    _, blocked_measurements = structured_vm["forward"](
        *(tvm_tensor(value, device=tvm.cpu()) for value in blocked_values)
    )
    np.testing.assert_array_equal(blocked_measurements.numpy()[3:4], [0.0])
    inactive_values = list(values)
    inactive_values[1] = np.zeros((1, groups), dtype="float32")
    inactive_residual, _ = structured_vm["forward"](
        *(tvm_tensor(value, device=tvm.cpu()) for value in inactive_values)
    )
    np.testing.assert_allclose(
        inactive_residual.numpy().reshape(hidden_size),
        [0.2, 0.4],
        rtol=1e-6,
        atol=1e-7,
    )

    class CurvedModule(nn.Module):
        def __init__(self, decode=False):
            super().__init__()
            self.decode = decode

        def forward(
            self,
            residual,
            affine_active,
            affine_basis,
            affine_neutral,
            affine_target,
            affine_along,
            affine_kappa,
            probe_kind,
            probe_direction,
            probe_bias,
            probe_threshold,
            curve_active,
            curve_rank,
            curve_intrinsic_dim,
            curve_embed_dim,
            curve_node_count,
            curve_basis,
            curve_neutral,
            curve_domain_kind,
            curve_node_parameters,
            curve_rbf_weights,
            curve_polynomial,
            curve_coordinate_offset,
            curve_coordinate_scale,
            curve_origin,
            curve_target,
            curve_along,
            curve_onto,
            curve_bounds,
            curve_axis_periodic,
            curve_axis_period,
            curve_sigma_present,
            curve_sigma_rbf_weights,
            curve_sigma_polynomial,
            curve_damping,
            curve_feet,
        ):
            curve_parameters = pack_structured_curve_parameters(
                curve_active,
                curve_intrinsic_dim,
                curve_node_parameters,
                curve_rbf_weights,
                curve_polynomial,
                curve_coordinate_offset,
                curve_coordinate_scale,
                curve_origin,
                curve_target,
                curve_along,
                curve_onto,
                curve_bounds,
                curve_axis_periodic,
                curve_axis_period,
                curve_sigma_present,
                curve_sigma_rbf_weights,
                curve_sigma_polynomial,
                curve_damping,
            )
            return apply_structured_curved_hook(
                residual,
                0,
                self.decode,
                affine_active,
                affine_basis,
                affine_neutral,
                affine_target,
                affine_along,
                affine_kappa,
                probe_kind,
                probe_direction,
                probe_bias,
                probe_threshold,
                curve_basis,
                curve_neutral,
                curve_domain_kind,
                curve_parameters,
                curve_feet,
            )

    curved_forward_spec = {
        "residual": spec.Tensor([1, 1, hidden_size], "float32"),
        "affine_active": spec.Tensor([1, groups], "float32"),
        "affine_basis": spec.Tensor([1, groups, rank, hidden_size], "float32"),
        "affine_neutral": spec.Tensor([1, groups, hidden_size], "float32"),
        "affine_target": spec.Tensor([1, groups, rank], "float32"),
        "affine_along": spec.Tensor([1, groups], "float32"),
        "affine_kappa": spec.Tensor([1, groups, rank], "float32"),
        "probe_kind": spec.Tensor([1, probes], "uint32"),
        "probe_direction": spec.Tensor([1, probes, hidden_size], "float32"),
        "probe_bias": spec.Tensor([1, probes], "float32"),
        "probe_threshold": spec.Tensor([1, probes], "float32"),
        "curve_active": spec.Tensor([1, curves], "float32"),
        "curve_rank": spec.Tensor([1, curves], "uint32"),
        "curve_intrinsic_dim": spec.Tensor([1, curves], "uint32"),
        "curve_embed_dim": spec.Tensor([1, curves], "uint32"),
        "curve_node_count": spec.Tensor([1, curves], "uint32"),
        "curve_basis": spec.Tensor([1, curves, rank, hidden_size], "float32"),
        "curve_neutral": spec.Tensor([1, curves, hidden_size], "float32"),
        "curve_domain_kind": spec.Tensor([1, curves], "uint32"),
        "curve_node_parameters": spec.Tensor([1, curves, nodes, embed], "float32"),
        "curve_rbf_weights": spec.Tensor([1, curves, nodes, rank], "float32"),
        "curve_polynomial": spec.Tensor([1, curves, embed + 1, rank], "float32"),
        "curve_coordinate_offset": spec.Tensor([1, curves, embed], "float32"),
        "curve_coordinate_scale": spec.Tensor([1, curves, embed], "float32"),
        "curve_origin": spec.Tensor([1, curves, intrinsic], "float32"),
        "curve_target": spec.Tensor([1, curves, intrinsic], "float32"),
        "curve_along": spec.Tensor([1, curves], "float32"),
        "curve_onto": spec.Tensor([1, curves], "float32"),
        "curve_bounds": spec.Tensor([1, curves, intrinsic, 2], "float32"),
        "curve_axis_periodic": spec.Tensor([1, curves, intrinsic], "uint32"),
        "curve_axis_period": spec.Tensor([1, curves, intrinsic], "float32"),
        "curve_sigma_present": spec.Tensor([1, curves], "uint32"),
        "curve_sigma_rbf_weights": spec.Tensor([1, curves, nodes], "float32"),
        "curve_sigma_polynomial": spec.Tensor([1, curves, embed + 1], "float32"),
        "curve_damping": spec.Tensor([1, curves], "float32"),
        "curve_feet": spec.Tensor([1, curves, intrinsic], "float32"),
    }
    curved_module, _, _ = CurvedModule().export_tvm(
        spec={"forward": curved_forward_spec},
        allow_extern=True,
    )
    curved_module = strip_gpu_thread_bindings(curved_module, tvm)
    curved_vm = relax.VirtualMachine(
        relax.build(curved_module, target=tvm.target.Target("c")), tvm.cpu()
    )
    curve_active = np.zeros((1, curves), dtype="float32")
    curve_active[0, :2] = 1
    curve_rank = np.zeros((1, curves), dtype="uint32")
    curve_rank[0, :2] = 2
    curve_intrinsic_dim = np.zeros((1, curves), dtype="uint32")
    curve_intrinsic_dim[0, :2] = [1, 2]
    curve_embed_dim = np.zeros((1, curves), dtype="uint32")
    curve_embed_dim[0, :2] = [2, 4]
    curve_node_count = np.zeros((1, curves), dtype="uint32")
    curve_node_count[0, :2] = 2
    curve_basis = np.zeros((1, curves, rank, hidden_size), dtype="float32")
    curve_basis[0, :2, 0] = [1, 0]
    curve_basis[0, :2, 1] = [0, 1]
    curve_polynomial = np.zeros((1, curves, embed + 1, rank), dtype="float32")
    curve_polynomial[0, 0, 1, 0] = 1
    curve_polynomial[0, 1, 1, 0] = 1
    curve_polynomial[0, 1, 3, 1] = 1
    curve_origin = np.zeros((1, curves, intrinsic), dtype="float32")
    curve_target = np.zeros((1, curves, intrinsic), dtype="float32")
    curve_target[0, 0, 0] = 0.3
    curve_target[0, 1, :2] = [0.1, -0.1]
    curve_along = np.zeros((1, curves), dtype="float32")
    curve_along[0, :2] = 1
    curve_bounds = np.zeros((1, curves, intrinsic, 2), dtype="float32")
    curve_bounds[..., 0] = -2
    curve_bounds[..., 1] = 2
    curve_periods = np.ones((1, curves, intrinsic), dtype="float32")
    curve_feet = np.zeros((1, curves, intrinsic), dtype="float32")
    curve_values = [
        np.asarray([[[0.2, 0.3]]], dtype="float32"),
        np.zeros((1, groups), dtype="float32"),
        affine_basis,
        affine_neutral,
        affine_target,
        affine_along,
        affine_kappa,
        probe_kind,
        probe_direction,
        probe_bias,
        probe_threshold,
        curve_active,
        curve_rank,
        curve_intrinsic_dim,
        curve_embed_dim,
        curve_node_count,
        curve_basis,
        np.zeros((1, curves, hidden_size), dtype="float32"),
        np.ones((1, curves), dtype="uint32"),
        np.zeros((1, curves, nodes, embed), dtype="float32"),
        np.zeros((1, curves, nodes, rank), dtype="float32"),
        curve_polynomial,
        np.zeros((1, curves, embed), dtype="float32"),
        np.ones((1, curves, embed), dtype="float32"),
        curve_origin,
        curve_target,
        curve_along,
        np.zeros((1, curves), dtype="float32"),
        curve_bounds,
        np.zeros((1, curves, intrinsic), dtype="uint32"),
        curve_periods,
        np.zeros((1, curves), dtype="uint32"),
        np.zeros((1, curves, nodes), dtype="float32"),
        np.zeros((1, curves, embed + 1), dtype="float32"),
        np.full((1, curves), 1e-3, dtype="float32"),
        curve_feet,
    ]
    curved, curve_measurements, next_foot = curved_vm["forward"](
        *(tvm_tensor(value, device=tvm.cpu()) for value in curve_values)
    )
    np.testing.assert_allclose(
        curved.numpy().reshape(hidden_size), [0.6, 0.2], rtol=3e-3, atol=3e-4
    )
    np.testing.assert_allclose(
        curve_measurements.numpy()[:2], [0.6, 0.0], rtol=3e-3, atol=3e-4
    )
    np.testing.assert_allclose(
        next_foot.numpy()[0, :2], [0.2, 0.0], rtol=3e-3, atol=3e-4
    )
    np.testing.assert_allclose(
        next_foot.numpy()[1, :2], [0.5, 0.3], rtol=3e-3, atol=3e-4
    )

    periodic_values = list(curve_values)
    periodic_active = np.zeros((1, curves), dtype="float32")
    periodic_active[0, 0] = 1
    periodic_intrinsic = np.zeros((1, curves), dtype="uint32")
    periodic_intrinsic[0, 0] = 1
    periodic_polynomial = np.zeros((1, curves, embed + 1, rank), dtype="float32")
    periodic_polynomial[0, 0, 1, 0] = 1
    periodic_polynomial[0, 0, 2, 1] = 1
    periodic_bounds = np.zeros((1, curves, intrinsic, 2), dtype="float32")
    periodic_bounds[..., 1] = 1
    periodic_axes = np.zeros((1, curves, intrinsic), dtype="uint32")
    periodic_axes[0, 0, 0] = 1
    periodic_target = np.zeros((1, curves, intrinsic), dtype="float32")
    periodic_target[0, 0, 0] = 0.25
    periodic_along = np.zeros((1, curves), dtype="float32")
    periodic_along[0, 0] = 1
    periodic_values[0] = np.asarray([[[1.0, 0.0]]], dtype="float32")
    periodic_values[11] = periodic_active
    periodic_values[13] = periodic_intrinsic
    periodic_values[21] = periodic_polynomial
    periodic_values[25] = periodic_target
    periodic_values[26] = periodic_along
    periodic_values[28] = periodic_bounds
    periodic_values[29] = periodic_axes
    periodic, _, periodic_foot = curved_vm["forward"](
        *(tvm_tensor(value, device=tvm.cpu()) for value in periodic_values)
    )
    np.testing.assert_allclose(
        periodic.numpy().reshape(hidden_size), [0.0, 1.0], rtol=3e-3, atol=3e-4
    )
    np.testing.assert_allclose(periodic_foot.numpy()[0, 0], 0.0, atol=3e-4)

    sphere_values = list(curve_values)
    sphere_active = np.zeros((1, curves), dtype="float32")
    sphere_active[0, 0] = 1
    sphere_rank = np.zeros((1, curves), dtype="uint32")
    sphere_rank[0, 0] = 2
    sphere_intrinsic = np.zeros((1, curves), dtype="uint32")
    sphere_intrinsic[0, 0] = 1
    sphere_embed = np.zeros((1, curves), dtype="uint32")
    sphere_embed[0, 0] = 2
    sphere_nodes = np.zeros((1, curves), dtype="uint32")
    sphere_nodes[0, 0] = 2
    sphere_basis = np.zeros((1, curves, rank, hidden_size), dtype="float32")
    sphere_basis[0, 0, 0] = [1, 0]
    sphere_basis[0, 0, 1] = [0, 1]
    sphere_domain = np.zeros((1, curves), dtype="uint32")
    sphere_domain[0, 0] = 2
    sphere_polynomial = np.zeros((1, curves, embed + 1, rank), dtype="float32")
    sphere_polynomial[0, 0, 1, 0] = 1
    sphere_polynomial[0, 0, 2, 1] = 1
    sphere_origin = np.zeros((1, curves, intrinsic), dtype="float32")
    sphere_target = np.zeros((1, curves, intrinsic), dtype="float32")
    sphere_target[0, 0, 0] = np.pi / 2
    sphere_along = np.zeros((1, curves), dtype="float32")
    sphere_along[0, 0] = 1
    sphere_bounds = np.zeros((1, curves, intrinsic, 2), dtype="float32")
    sphere_bounds[0, 0, 0] = [0, 2 * np.pi]
    sphere_axis_periodic = np.zeros((1, curves, intrinsic), dtype="uint32")
    sphere_axis_periodic[0, 0, 0] = 1
    sphere_periods = np.ones((1, curves, intrinsic), dtype="float32")
    sphere_periods[0, 0, 0] = 2 * np.pi
    sphere_values[0] = np.asarray([[[1.0, 0.0]]], dtype="float32")
    sphere_values[11] = sphere_active
    sphere_values[12] = sphere_rank
    sphere_values[13] = sphere_intrinsic
    sphere_values[14] = sphere_embed
    sphere_values[15] = sphere_nodes
    sphere_values[16] = sphere_basis
    sphere_values[18] = sphere_domain
    sphere_values[21] = sphere_polynomial
    sphere_values[24] = sphere_origin
    sphere_values[25] = sphere_target
    sphere_values[26] = sphere_along
    sphere_values[28] = sphere_bounds
    sphere_values[29] = sphere_axis_periodic
    sphere_values[30] = sphere_periods
    sphere, _, sphere_foot = curved_vm["forward"](
        *(tvm_tensor(value, device=tvm.cpu()) for value in sphere_values)
    )
    np.testing.assert_allclose(
        sphere.numpy().reshape(hidden_size), [0.0, 1.0], rtol=3e-3, atol=3e-4
    )
    np.testing.assert_allclose(sphere_foot.numpy()[0, 0], 0.0, atol=3e-4)

    decode_module, _, _ = CurvedModule(decode=True).export_tvm(
        spec={"forward": curved_forward_spec},
        allow_extern=True,
    )
    decode_module = strip_gpu_thread_bindings(decode_module, tvm)
    decode_vm = relax.VirtualMachine(
        relax.build(decode_module, target=tvm.target.Target("c")), tvm.cpu()
    )
    curve_values[11] = np.zeros((1, curves), dtype="float32")
    curve_values[-1] = np.full((1, curves, intrinsic), 0.4, dtype="float32")
    inactive, _, inactive_foot = decode_vm["forward"](
        *(tvm_tensor(value, device=tvm.cpu()) for value in curve_values)
    )
    np.testing.assert_allclose(
        inactive.numpy().reshape(hidden_size), [0.2, 0.3], rtol=1e-6, atol=1e-7
    )
    np.testing.assert_array_equal(
        inactive_foot.numpy(), np.full((curves, intrinsic), 0.4, dtype="float32")
    )

    class JlensModule(nn.Module):
        def forward(self, hidden_states, jacobian, layer_ids, token_ids, lm_head):
            layer_count = jacobian.shape[0]
            selected_hidden = op.take(hidden_states, layer_ids, axis=0)
            transported = op.matmul(
                op.reshape(selected_hidden, (layer_count, 1, hidden_size)),
                op.permute(jacobian, axes=[0, 2, 1]),
            )
            normalized = transported / op.sqrt(
                op.sum(transported * transported, axis=-1, keepdims=True)
                / hidden_size
                + 1e-6
            )
            logits = op.matmul(normalized, op.permute(lm_head, axes=[1, 0]))
            probabilities = op.softmax(logits.astype("float32"), axis=-1)
            selected_rows = op.take(lm_head, token_ids, axis=0)
            directions = op.matmul(
                op.reshape(selected_rows, (1, 3, hidden_size)), jacobian
            )
            return (
                op.reshape(
                    op.take(probabilities, token_ids, axis=2),
                    (layer_count, 3),
                ),
                directions,
            )

    jlens_module, _, _ = JlensModule().export_tvm(
        spec={
            "forward": {
                "hidden_states": spec.Tensor([2, hidden_size], "float32"),
                "jacobian": spec.Tensor(
                    ["jlens_layers", hidden_size, hidden_size], "float32"
                ),
                "layer_ids": spec.Tensor(["jlens_layers"], "int32"),
                "token_ids": spec.Tensor([3], "int32"),
                "lm_head": spec.Tensor([5, hidden_size], "float32"),
            }
        },
        allow_extern=True,
    )
    jlens_vm = relax.VirtualMachine(
        relax.build(jlens_module, target=tvm.target.Target("c")), tvm.cpu()
    )
    hidden = np.asarray([[1.0, 2.0], [-1.0, 0.5]], dtype="float32")
    jacobians = np.asarray(
        [[[1.0, 0.0], [0.0, 1.0]], [[0.5, 1.0], [-1.0, 0.25]]],
        dtype="float32",
    )
    token_ids = np.asarray([4, 0, 2], dtype="int32")
    lm_head = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.5, -0.5]],
        dtype="float32",
    )
    layer_ids = np.asarray([1, 0], dtype="int32")
    actual, actual_directions = jlens_vm["forward"](
        tvm_tensor(hidden, device=tvm.cpu()),
        tvm_tensor(jacobians, device=tvm.cpu()),
        tvm_tensor(layer_ids, device=tvm.cpu()),
        tvm_tensor(token_ids, device=tvm.cpu()),
        tvm_tensor(lm_head, device=tvm.cpu()),
    )
    actual = actual.numpy()
    actual_directions = actual_directions.numpy()
    transported = np.matmul(
        hidden[layer_ids, None, :], np.swapaxes(jacobians, 1, 2)
    )
    normalized = transported / np.sqrt(
        np.mean(transported * transported, axis=-1, keepdims=True) + 1e-6
    )
    logits = np.matmul(normalized, lm_head.T)
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = probabilities[:, 0, :][:, token_ids]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual.sum(axis=1), expected.sum(axis=1), rtol=1e-6)
    expected_directions = np.matmul(lm_head[token_ids][None, :, :], jacobians)
    np.testing.assert_allclose(
        actual_directions, expected_directions, rtol=1e-6, atol=1e-7
    )


def verify_packed_geometry_golden(
    nn,
    spec,
    relax,
    tvm,
    tvm_tensor,
    measure_structured_geometry,
    structured_geometry_payload_layout,
) -> None:
    layers = 1
    hidden_size = 2
    geometry_probes = 8
    whitener_rank = 96
    rank = 8
    candidates = 33
    intrinsic = 4
    curve_nodes = 32
    curve_stride = 671
    output_stride = 41
    header_stride = 7
    layout = structured_geometry_payload_layout(layers, hidden_size)

    class GeometryModule(nn.Module):
        def forward(
            self,
            hidden,
            whitening_rank,
            whitening_ridge,
            whitening_basis,
            whitening_correction,
            header,
            payload,
            feet,
        ):
            return measure_structured_geometry(
                hidden,
                False,
                whitening_rank,
                whitening_ridge,
                whitening_basis,
                whitening_correction,
                header,
                payload,
                feet,
            )

    module, _, _ = GeometryModule().export_tvm(
        spec={
            "forward": {
                "hidden": spec.Tensor([layers, hidden_size], "float32"),
                "whitening_rank": spec.Tensor([layers], "uint32"),
                "whitening_ridge": spec.Tensor([layers], "float32"),
                "whitening_basis": spec.Tensor(
                    [layers, whitener_rank, hidden_size], "float32"
                ),
                "whitening_correction": spec.Tensor(
                    [layers, whitener_rank], "float32"
                ),
                "header": spec.Tensor(
                    [layers, geometry_probes, header_stride], "uint32"
                ),
                "payload": spec.Tensor([layout["elements"]], "float32"),
                "feet": spec.Tensor(
                    [layers, geometry_probes, intrinsic], "float32"
                ),
            }
        },
        allow_extern=True,
    )
    module = strip_gpu_thread_bindings(module, tvm)
    vm = relax.VirtualMachine(
        relax.build(module, target=tvm.target.Target("c")), tvm.cpu()
    )

    hidden = np.asarray([[0.25, 0.5]], dtype="float32")
    whitening_rank_value = np.asarray([2], dtype="uint32")
    whitening_ridge = np.ones((layers,), dtype="float32")
    whitening_basis = np.zeros(
        (layers, whitener_rank, hidden_size), dtype="float32"
    )
    whitening_basis[0, 0] = [1, 0]
    whitening_basis[0, 1] = [0, 1]
    whitening_correction = np.zeros((layers, whitener_rank), dtype="float32")
    header = np.zeros(
        (layers, geometry_probes, header_stride), dtype="uint32"
    )
    header[0, 0] = [1, 1, 1, 1, 2, 0, 0]
    header[0, 1] = [1, 2, 1, 1, 2, 2, 1]
    payload = np.zeros((layout["elements"],), dtype="float32")

    def payload_view(name, shape):
        start = layout[name]
        size = int(np.prod(shape))
        return payload[start : start + size].reshape(shape)

    basis = payload_view(
        "basis", (layers, geometry_probes, rank, hidden_size)
    )
    gram_inverse = payload_view(
        "gram_inverse", (layers, geometry_probes, rank, rank)
    )
    cholesky = payload_view(
        "cholesky", (layers, geometry_probes, rank, rank)
    )
    node_white = payload_view(
        "node_white", (layers, geometry_probes, candidates, rank)
    )
    coord_map = payload_view(
        "coord_map", (layers, geometry_probes, intrinsic, rank)
    )
    coord_bias = payload_view(
        "coord_bias", (layers, geometry_probes, intrinsic)
    )
    curve_parameters = payload_view(
        "curve_parameters", (layers, geometry_probes, curve_stride)
    )
    curve_node_coords = payload_view(
        "curve_node_coords", (layers, geometry_probes, curve_nodes, intrinsic)
    )
    curve_node_values = payload_view(
        "curve_node_values", (layers, geometry_probes, curve_nodes, rank)
    )
    for probe in (0, 1):
        basis[0, probe, 0] = [1, 0]
        gram_inverse[0, probe, 0, 0] = 1
        cholesky[0, probe, 0, 0] = 1
    node_white[0, 0, 1, 0] = 2
    coord_map[0, 0, 0, 0] = 2
    coord_bias[0, 0, 0] = -1
    node_white[0, 1, 1, 0] = 1
    curve_parameters[0, 1, 0] = 1
    curve_parameters[0, 1, 1] = 1
    curve_parameters[0, 1, 514 + rank] = 1
    curve_parameters[0, 1, 594 : 594 + 8] = 1
    curve_parameters[0, 1, 612 : 614] = [-2, 2]
    curve_parameters[0, 1, 624 : 624 + intrinsic] = 1
    curve_parameters[0, 1, 670] = 1e-3
    curve_node_coords[0, 1, 1, 0] = 1
    curve_node_values[0, 1, 1, 0] = 1
    feet = np.zeros((layers, geometry_probes, intrinsic), dtype="float32")
    feet[0, 0] = [0.1, 0.2, 0.3, 0.4]

    measurements, next_feet = vm["forward"](
        tvm_tensor(hidden, device=tvm.cpu()),
        tvm_tensor(whitening_rank_value, device=tvm.cpu()),
        tvm_tensor(whitening_ridge, device=tvm.cpu()),
        tvm_tensor(whitening_basis, device=tvm.cpu()),
        tvm_tensor(whitening_correction, device=tvm.cpu()),
        tvm_tensor(header, device=tvm.cpu()),
        tvm_tensor(payload, device=tvm.cpu()),
        tvm_tensor(feet, device=tvm.cpu()),
    )
    actual = measurements.numpy().reshape(layers, geometry_probes, output_stride)
    expected_fraction = 0.25 / np.sqrt(0.25 * 0.25 + 0.5 * 0.5)
    np.testing.assert_allclose(
        actual[0, 0, :10],
        [1, expected_fraction, 0, 1, -0.5, 0, 0, 0, 0.25, 1.75],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        actual[0, 1, :10],
        [1, expected_fraction, 0, 1, 0.25, 0, 0, 0, 0.25, 0.75],
        rtol=2e-3,
        atol=2e-4,
    )
    np.testing.assert_array_equal(actual[0, 2:], 0)
    expected_feet = feet.copy()
    expected_feet[0, 1, 0] = 0.25
    np.testing.assert_allclose(
        next_feet.numpy(), expected_feet, rtol=2e-3, atol=2e-4
    )


def create_tiny_model(
    architecture,
    Qwen3Config,
    Qwen3LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    Gemma3Config,
    Gemma3TextConfig,
    Gemma3ForCausalLM,
):
    if architecture == "qwen3":
        return (
            Qwen3LMHeadModel(
                Qwen3Config(
                    hidden_act="silu",
                    hidden_size=4,
                    intermediate_size=8,
                    attention_bias=False,
                    num_attention_heads=2,
                    num_hidden_layers=2,
                    num_key_value_heads=1,
                    rms_norm_eps=1e-6,
                    rope_theta=10000,
                    vocab_size=16,
                    context_window_size=16,
                    head_dim=2,
                )
            ),
            "Qwen3",
        )
    if architecture == "llama":
        return (
            LlamaForCausalLM(
                LlamaConfig(
                    hidden_size=4,
                    intermediate_size=8,
                    num_attention_heads=2,
                    num_hidden_layers=2,
                    rms_norm_eps=1e-6,
                    vocab_size=16,
                    context_window_size=16,
                    num_key_value_heads=1,
                    head_dim=2,
                )
            ),
            "Llama",
        )
    return (
        Gemma3ForCausalLM(
            Gemma3Config(
                text_config=Gemma3TextConfig(
                    hidden_size=4,
                    intermediate_size=8,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=2,
                    sliding_window_size=16,
                    query_pre_attn_scalar=2,
                    context_window_size=16,
                    prefill_chunk_size=16,
                ),
                vocab_size=16,
                is_text_model=True,
            )
        ),
        "Gemma3",
    )


def verify_isolated_model_exports(args) -> None:
    for architecture in ("qwen3", "llama", "gemma3_text"):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mlc-repository",
                str(args.mlc_repository),
                "--tvm-repository",
                str(args.tvm_repository),
                "--fixture",
                str(args.fixture),
                "--export-only",
                architecture,
            ],
            check=True,
        )


def verify_isolated_readout_goldens(args) -> None:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mlc-repository",
            str(args.mlc_repository),
            "--tvm-repository",
            str(args.tvm_repository),
            "--fixture",
            str(args.fixture),
            "--readout-only",
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlc-repository", required=True, type=Path)
    parser.add_argument("--tvm-repository", required=True, type=Path)
    parser.add_argument("--export-only", choices=["qwen3", "llama", "gemma3_text"])
    parser.add_argument("--readout-only", action="store_true")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=repository_root / "browser-runtime/fixtures/post-block-rank-one-v2.json",
    )
    return parser.parse_args()


def strip_gpu_thread_bindings(module, tvm):
    """Execute workgroup phases serially on CPU, retaining per-lane local state."""
    tir = tvm.tirx
    def has_sync(node):
        found = []
        tir.stmt_functor.post_order_visit(node, lambda n: found.append(n) if isinstance(n, tvm.ir.Call) and n.op.name == "tirx.tvm_storage_sync" else None)
        return bool(found)

    def lower(function):
        # Resolve block axes before moving a phase across its thread loop.
        def cast_bound_axes(node):
            if isinstance(node, tir.SBlockRealize):
                values = [tir.Cast(axis.var.ty.dtype, value)
                          for axis, value in zip(node.block.iter_vars, node.iter_values)]
                return tir.SBlockRealize(values, node.predicate, node.block)
            return None
        function = function.with_body(tir.stmt_functor.ir_transform(function.body, None, cast_bound_axes))
        one = tvm.IRModule({"main": function})
        one = tvm.s_tir.transform.ConvertBlocksToOpaque()(one)
        function = one["main"]
        lanes = []
        tir.stmt_functor.post_order_visit(function.body, lambda n: lanes.append(n) if isinstance(n, tir.For) and n.thread_binding is not None and n.thread_binding.thread_tag == "threadIdx.x" else None)
        if not lanes:
            return function
        extent = lanes[0].extent
        locals_ = {}
        def find_buffers(node):
            if isinstance(node, tir.SBlock):
                for buf in node.alloc_buffers:
                    if buf.scope() == "local":
                        locals_[buf] = tir.decl_buffer((extent, *buf.shape), buf.dtype, name=buf.name)
                    elif buf.scope() == "shared":
                        locals_[buf] = tir.decl_buffer(buf.shape, buf.dtype, name=buf.name)
        tir.stmt_functor.post_order_visit(function.body, find_buffers)
        thread = lanes[0].loop_var
        def buffers(node):
            if isinstance(node, tir.BufferLoad) and node.buffer in locals_:
                indices = [thread, *node.indices] if node.buffer.scope() == "local" else node.indices
                return tir.BufferLoad(locals_[node.buffer], indices)
            if isinstance(node, tir.BufferStore) and node.buffer in locals_:
                indices = [thread, *node.indices] if node.buffer.scope() == "local" else node.indices
                return tir.BufferStore(locals_[node.buffer], node.value, indices)
            if isinstance(node, tir.SBlock):
                return tir.SBlock(node.iter_vars, node.reads, node.writes, node.name_hint, node.body, node.init,
                    [locals_.get(b,b) for b in node.alloc_buffers], node.match_buffers, node.annotations)
            return None
        body = tir.stmt_functor.ir_transform(function.body, None, buffers)
        def serial_loop(node, body):
            return tir.For(node.loop_var,node.min,node.extent,tir.ForKind.SERIAL,body,annotations=node.annotations,step=node.step)
        def lane_loop(node):
            return tir.For(thread,0,extent,tir.ForKind.SERIAL,node)
        def phases(node):
            if not has_sync(node):
                return lane_loop(node)
            if isinstance(node,tir.Evaluate):
                return tir.Evaluate(0)
            if isinstance(node,tir.SeqStmt):
                result, pending = [], []
                for statement in node.seq:
                    if has_sync(statement):
                        if pending:
                            result.append(lane_loop(pending[0] if len(pending) == 1 else tir.SeqStmt(pending)))
                            pending = []
                        result.append(phases(statement))
                    else:
                        pending.append(statement)
                if pending:
                    result.append(lane_loop(pending[0] if len(pending) == 1 else tir.SeqStmt(pending)))
                return result[0] if len(result) == 1 else tir.SeqStmt(result)
            if isinstance(node,tir.For):
                return serial_loop(node,phases(node.body))
            if isinstance(node,tir.IfThenElse):
                return tir.IfThenElse(node.condition,phases(node.then_case),phases(node.else_case) if node.else_case is not None else None)
            if isinstance(node,tir.SBlockRealize):
                b=node.block
                return tir.SBlockRealize(node.iter_values,node.predicate,tir.SBlock(b.iter_vars,b.reads,b.writes,b.name_hint,phases(b.body),b.init,b.alloc_buffers,b.match_buffers,b.annotations))
            raise ValueError(f"unsupported cooperative CPU phase: {type(node)}")
        def threads(node):
            if isinstance(node,tir.For) and node.kind == tir.ForKind.THREAD_BINDING:
                if node.thread_binding.thread_tag == "threadIdx.x":
                    return phases(node.body)
                return serial_loop(node,node.body)
            return None
        body=tir.stmt_functor.ir_transform(body,None,threads)
        return function.with_body(body)
    for var in list(module.get_global_vars()):
        if isinstance(module[var],tir.PrimFunc):
            module.update_func(var,lower(module[var]))
    return module


def require_commit(repository: Path, expected: str, label: str) -> None:
    actual = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise SystemExit(f"{label} checkout must be exact commit {expected}, found {actual}")


def verify_result_files(repository: Path) -> None:
    for relative_path, expected in RESULT_DIGESTS.items():
        path = repository / relative_path
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"MLC overlay result digest differs for {relative_path}")


def verify_portable_topk_source(repository: Path) -> None:
    moe_source = (repository / "python/mlc_llm/op/moe_misc.py").read_text(encoding="utf-8")
    if (
        moe_source.count("TX = 1024") < 2
        or "T.buffer_store(local_top_k_index, -1, indices=[t])" not in moe_source
        or "T.buffer_store(local_top_k_index, t, indices=[-1])" in moe_source
    ):
        raise SystemExit("generic MLC MoE top-k differs from upstream scheduling or safe initialization")
    hooks_source = (
        repository / "python/mlc_llm/model/drowse_hooks.py"
    ).read_text(encoding="utf-8")
    exact_source = hooks_source.split("def exact_readout_topk", 1)[-1].split(
        "def structured_geometry_payload_layout", 1
    )[0]
    for fragment in (
        "EXACT_READOUT_TOPK_BLOCK_SIZE = 256",
        "candidate_count = T.ceildiv(column_count, EXACT_READOUT_TOPK_BLOCK_SIZE)",
        '"drowse_exact_top8_tiles"',
        '"drowse_exact_top8_merge"',
        "_exact_readout_is_better",
        "while candidate_count > 1:",
        'for thread in T.thread_binding(0, 64, "threadIdx.x"):',
        "T.And(",
        "T.Or(",
    ):
        if fragment not in hooks_source:
            raise SystemExit(f"Drowse exact tiled top-8 omits {fragment}")
    if 'scope="shared"' not in exact_source or "tvm_storage_sync" not in exact_source:
        raise SystemExit("Drowse exact tiled top-8 requires bounded cooperative reduction")
    for relative_path in (
        "python/mlc_llm/model/llama/llama_model.py",
        "python/mlc_llm/model/qwen3/qwen3_model.py",
    ):
        model_source = (repository / relative_path).read_text(encoding="utf-8")
        if (
            model_source.count("exact_readout_topk") != 5
            or "op_ext.moe_misc.gating_topk(" in model_source
        ):
            raise SystemExit(f"{relative_path} does not route every exact readout through Drowse top-8")


def verify_exact_parallel_topk_cpu_golden() -> None:
    block_size = 256
    top_k = 8

    def better(value, index, best_value, best_index):
        return index >= 0 and (
            value > best_value
            or (value == best_value and (best_index < 0 or index < best_index))
        )

    def insert(values, indices, value, index):
        for slot in range(top_k):
            if better(value, index, values[slot], indices[slot]):
                values[slot + 1 :] = values[slot:-1]
                indices[slot + 1 :] = indices[slot:-1]
                values[slot] = value
                indices[slot] = index
                break

    def parallel_top8(row):
        candidate_values = []
        candidate_indices = []
        for start in range(0, row.shape[0], block_size):
            values = np.full(top_k, -np.inf, dtype="float32")
            indices = np.full(top_k, -1, dtype="int32")
            for index in range(start, min(start + block_size, row.shape[0])):
                insert(values, indices, row[index], index)
            candidate_values.append(values)
            candidate_indices.append(indices)
        values = np.full(top_k, -np.inf, dtype="float32")
        indices = np.full(top_k, -1, dtype="int32")
        for block_values, block_indices in zip(candidate_values, candidate_indices):
            for value, index in zip(block_values, block_indices):
                insert(values, indices, value, index)
        return values, indices

    rng = np.random.default_rng(20260829)
    rows = rng.random((3, 49153), dtype=np.float32)
    rows[0, :4096] = 0.5
    rows[0, [0, 255, 256, 511, 512, 767, 768, 1023, 1024]] = 1.0
    rows[1, :] = 0.0
    rows[2, [19, 275, 531, 787, 1043, 1299, 1555, 1811, 2067]] = 2.0
    for row in rows:
        actual_values, actual_indices = parallel_top8(row)
        expected_indices = np.argsort(-row, kind="stable")[:top_k]
        np.testing.assert_array_equal(actual_indices, expected_indices)
        np.testing.assert_array_equal(actual_values, row[expected_indices])


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
        raise SystemExit("mlc_llm was imported before the source-only verification bootstrap")
    package = types.ModuleType("mlc_llm")
    package.__path__ = [str(mlc_repository / "python/mlc_llm")]
    package.__package__ = "mlc_llm"
    sys.modules["mlc_llm"] = package
    model_package = types.ModuleType("mlc_llm.model")
    model_package.__path__ = [str(mlc_repository / "python/mlc_llm/model")]
    model_package.__package__ = "mlc_llm.model"
    sys.modules["mlc_llm.model"] = model_package


def verify_model_exports(model, label: str, tvm):
    module, _, _ = model.export_tvm(spec=model.get_default_spec(), allow_extern=True)
    names = {global_var.name_hint for global_var in module.get_global_vars()}
    missing = sorted(REQUIRED_FUNCTIONS - names)
    if missing:
        raise SystemExit(f"{label} export lacks Drowse functions: {', '.join(missing)}")
    geometry_kernels = []
    for global_var in module.get_global_vars():
        function = module[global_var]
        if (
            isinstance(function, tvm.tirx.PrimFunc)
            and global_var.name_hint.startswith("drowse_geometry_measurements")
        ):
            geometry_kernels.append((global_var.name_hint, len(function.params)))
    if not geometry_kernels:
        raise SystemExit(f"{label} export lacks the packed Drowse geometry kernel")
    for name, parameters in geometry_kernels:
        if parameters != 7:
            raise SystemExit(
                f"{label} {name} exposes {parameters} packed buffer parameters; expected 7"
            )
    return module


if __name__ == "__main__":
    main()
