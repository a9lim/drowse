#!/usr/bin/env python3
"""Export numerical WebGPU fixtures with the pinned MLC/TVM overlay."""

import argparse
import importlib.util
import json
import pathlib
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--mlc-repository", required=True, type=pathlib.Path)
parser.add_argument("--tvm-repository", required=True, type=pathlib.Path)
parser.add_argument("--output", required=True, type=pathlib.Path)
args = parser.parse_args()
mlc = args.mlc_repository.resolve(strict=True)
tvm_root = args.tvm_repository.resolve(strict=True)
sys.path[:0] = [str(tvm_root / "python"), str(mlc / "python")]
verification_spec = importlib.util.spec_from_file_location(
    "verify_mlc", pathlib.Path(__file__).with_name("verify-mlc-hook.py")
)
verify_mlc = importlib.util.module_from_spec(verification_spec)
verification_spec.loader.exec_module(verify_mlc)
verify_mlc.require_commit(mlc, verify_mlc.MLC_BASE_COMMIT, "MLC-LLM")
verify_mlc.require_commit(tvm_root, verify_mlc.TVM_COMMIT, "TVM")
verify_mlc.verify_result_files(mlc)
import tvm
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import spec

verify_mlc.bootstrap_mlc_package(mlc)
from mlc_llm.model.drowse_hooks import (
    apply_structured_affine_hook,
    exact_readout_topk,
    transport_jlens_hidden,
    _reconstruct_curve_tir,
    STRUCTURED_CURVE_PARAMETER_STRIDE,
)

out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
metadata = []


def emit(module, label, metadata_fields):
    for name in module.get_global_vars():
        if name.name_hint.startswith(
            ("drowse_structured_affine", "drowse_exact_top8", "drowse_jlens_transport", "drowse_curve_reconstruct")
        ):
            function = module[name].with_attr("global_symbol", "main")
            compiled = tvm.compile(
                tvm.tirx.transform.ForceNarrowIndexToInt32()(tvm.IRModule({"main": function})),
                target=tvm.target.Target("webgpu"),
            )
            source = compiled.mod.imports[0].inspect_source()
            filename = f"{label}-{name.name_hint}.wgsl"
            (out / filename).write_text(source)
            metadata.append({"file": filename, "kernel": name.name_hint, **metadata_fields})
            print("compiled", filename, flush=True)


class Affine(nn.Module):
    def forward(self, residual, active, basis, neutral, target, along, kappa, kind, direction, bias, threshold):
        return apply_structured_affine_hook(
            residual, 0, active, basis, neutral, target, along, kappa, kind, direction, bias, threshold
        )


for h, b, t in [(2, 1, 1), (65, 1, 1), (512, 1, 1), (2560, 1, 1), (128, 1, 3)]:
    shapes = {
        "residual": [b, t, h],
        "active": [1, 4],
        "basis": [1, 4, 8, h],
        "neutral": [1, 4, h],
        "target": [1, 4, 8],
        "along": [1, 4],
        "kappa": [1, 4, 8],
        "kind": [1, 8],
        "direction": [1, 8, h],
        "bias": [1, 8],
        "threshold": [1, 8],
    }
    module, _, _ = Affine().export_tvm(
        spec={"forward": {k: spec.Tensor(v, "uint32" if k == "kind" else "float32") for k, v in shapes.items()}},
        allow_extern=True,
    )
    emit(module, f"affine-{h}-{b}-{t}", {"family": "affine", "h": h, "b": b, "t": t})


class TopK(nn.Module):
    def forward(self, scores):
        return exact_readout_topk(scores)


for columns in [17, 257, 70001, 262144]:
    module, _, _ = TopK().export_tvm(
        spec={"forward": {"scores": spec.Tensor([2, columns], "float32")}}, allow_extern=True
    )
    emit(module, f"topk-{columns}", {"family": "topk", "rows": 2, "columns": columns})
module, _, _ = TopK().export_tvm(
    spec={"forward": {"scores": spec.Tensor([2, "columns"], "float32")}}, allow_extern=True
)
emit(module, "topk-dynamic", {"family": "topk-dynamic", "rows": 2})


class Transport(nn.Module):
    def forward(self, hidden, jacobians):
        return transport_jlens_hidden(hidden, jacobians)


for h in [65, 512, 2560]:
    module, _, _ = Transport().export_tvm(
        spec={"forward": {"hidden": spec.Tensor([3, h], "float32"), "jacobians": spec.Tensor([3, h, h], "float32")}},
        allow_extern=True,
    )
    emit(module, f"transport-{h}", {"family": "transport", "h": h, "layers": 3})


class Curve(nn.Module):
    def forward(self, residual, basis, neutral, q, coordinates, parameters):
        return _reconstruct_curve_tir(residual, basis, neutral, q, coordinates, parameters, 0, 0)


for h in [65, 512, 2560]:
    shapes = {
        "residual": [1, 3, h],
        "basis": [1, 4, 8, h],
        "neutral": [1, 4, h],
        "q": [1, 3, 8],
        "coordinates": [1, 3, 8],
        "parameters": [1, 4, STRUCTURED_CURVE_PARAMETER_STRIDE],
    }
    module, _, _ = Curve().export_tvm(
        spec={"forward": {k: spec.Tensor(v, "float32") for k, v in shapes.items()}}, allow_extern=True
    )
    emit(module, f"curve-{h}", {"family": "curve", "h": h, "t": 3, "stride": STRUCTURED_CURVE_PARAMETER_STRIDE})
(out / "manifest.json").write_text(json.dumps(metadata, indent=2))
print("all kernels compiled")
