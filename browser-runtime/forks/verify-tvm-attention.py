#!/usr/bin/env python3

"""Check the actual prefill softmax macros serially; GPU synchronization is not tested."""

import argparse
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tvm
from tvm.relax.frontend.nn.llm import _kernel_common
from tvm.runtime import tensor
from tvm.script import tirx as T


def compile_update(macros, mode, local=False, absolute_causal=False):
    init, _, causal, _, valid_length, _, _, padded_left = macros

    @T.prim_func(s_tir=True)
    def check(
        scores: T.Buffer((1, 4), "float32"),
        maximum: T.Buffer((1,), "float32"),
        denominator: T.Buffer((1,), "float32"),
        old_maximum: T.Buffer((1,), "float32"),
        output: T.Buffer((1, 1), "float32"),
        new_m: T.Buffer((1,), "float32"),
        prev_m: T.Buffer((1,), "float32"),
        new_d: T.Buffer((1,), "float32"),
        row: T.int32,
        valid: T.int32,
    ):
        init(maximum, denominator, output, 0, 0)
        if mode == 0:
            if local or absolute_causal:
                causal(scores, maximum, denominator, old_maximum,
                       new_m, prev_m, new_d, 0, 0, row, 0, 0 if absolute_causal else 1, valid, 4, valid - 4)
            else:
                causal(scores, maximum, denominator, old_maximum,
                       new_m, prev_m, new_d, 0, 0, row, 0, 1, valid, 4)
        elif mode == 1:
            valid_length(scores, maximum, denominator, old_maximum,
                         new_m, prev_m, new_d, 0, 0, row, 0, valid, 4, 4)
        else:
            padded_left(scores, maximum, denominator, old_maximum,
                        new_m, prev_m, new_d, 0, 0, row, 0, valid, 4, 4)

    def remove_serial_barrier(node):
        if (isinstance(node, tvm.tirx.Evaluate)
                and isinstance(node.value, tvm.ir.expr.Call)
                and node.value.op.name == "tirx.tvm_storage_sync"):
            return tvm.tirx.Evaluate(0)
        return None

    body = tvm.tirx.stmt_functor.ir_transform(check.body, None, remove_serial_barrier)
    return tvm.tirx.build(tvm.IRModule({"check": check.with_body(body)}), target="llvm")


def verify(module, local_window=0, absolute_causal=False):
    macros = module._make_prefill_macros(1, 1, 4, 1, 1, 1, 1, local_window, absolute_causal) if absolute_causal else module._make_prefill_macros(1, 1, 4, 1, 1, 1, 1, local_window) if local_window else module._make_prefill_macros(1, 1, 4, 1, 1, 1, 1)
    cases = [
        ("causal-negative", 0, 0, 4, [True, False, False, False]),
        ("right-padding-negative", 1, 0, 2, [True, True, False, False]),
        ("left-padding-negative", 2, 2, 2, [False, False, True, False]),
        ("causal-empty", 0, 0, 0, [False] * 4),
        ("right-padding-empty", 1, 0, 0, [False] * 4),
        ("left-padding-empty", 2, 0, 2, [False] * 4),
    ]
    if local_window:
        cases = [
            ("local-before-window", 0, 0, 4, [True, False, False, False]),
            ("local-at-window", 0, 1, 4, [True, True, False, False]),
            ("local-past-window", 0, 2, 4, [False, True, True, False]),
            ("local-final-window", 0, 3, 4, [False, False, True, True]),
            ("local-empty", 0, 0, 0, [False] * 4),
        ]
    kernels = [compile_update(macros, mode, bool(local_window), absolute_causal) for mode in range(3)]
    passed = []
    for name, mode, row, valid, mask in cases:
        mask = np.asarray(mask)
        values = np.where(mask, [-80000, -80001, -80002, -80003], 50000).astype("float32")
        buffers = [tensor(values.reshape(1, 4))]
        buffers.extend(tensor(np.zeros(shape, dtype="float32"))
                       for shape in [(1,), (1,), (1,), (1, 1), (1,), (1,), (1,)])
        kernels[mode]["check"](*buffers, row, valid)
        probabilities = buffers[0].numpy()[0] / buffers[2].numpy()[0]
        expected = np.zeros(4, dtype="float64")
        if mask.any():
            weights = np.exp2(values[mask].astype("float64") - values[mask].max())
            expected[mask] = weights / weights.sum()
        np.testing.assert_allclose(probabilities, expected, atol=1e-7, rtol=1e-6, err_msg=name)
        assert all(np.isfinite(buffer.numpy()).all() for buffer in buffers), name
        passed.append(name)
    print(json.dumps({"validation": "serial-prefill-macro-numerics", "gpuSynchronization": False,
                      "localWindow": local_window,
                      "absoluteCausal": absolute_causal,
                      "passed": passed}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-baseline", type=Path)
    parser.add_argument("--local-window", action="store_true")
    parser.add_argument("--absolute-causal", action="store_true")
    args = parser.parse_args()
    if args.upstream_baseline is None:
        verify(_kernel_common, 2 if args.local_window else 0, args.absolute_causal)
        return
    source = subprocess.check_output([
        "git", "-C", str(args.upstream_baseline), "show",
        "68ba2b31c2a6d202fcd44e5780ef10ce08721dd5:python/tvm/relax/frontend/nn/llm/_kernel_common.py",
    ], text=True)
    with tempfile.TemporaryDirectory(prefix="drowse-attention-baseline-") as directory:
        path = Path(directory) / "_kernel_common.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(
            "tvm.relax.frontend.nn.llm._attention_baseline", path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        verify(module)


if __name__ == "__main__":
    main()
