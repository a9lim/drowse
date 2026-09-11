# WebGPU kernel validation

Export fixtures inside the pinned compiler environment after applying the fork overlays:

```sh
python browser-runtime/forks/export-webgpu-kernel-goldens.py \
  --mlc-repository /path/to/mlc-llm \
  --tvm-repository /path/to/tvm \
  --output /tmp/drowse-kernels
```

After installing the web UI dependencies, run the emitted WGSL on the installed Chrome WebGPU adapter:

```sh
node browser-runtime/forks/verify-webgpu-kernel-goldens.mjs \
  /tmp/drowse-kernels/manifest.json /path/to/web-llm /tmp/drowse-kernel-results.json
```

The second command also compiles the native sampling and curve-control shaders from the supplied WebLLM checkout. It uses normal headless Chrome without GPU-enabling flags. Missing WebGPU support fails the check.

The fixtures cover inactive, push, ablation, and sequential affine groups; curved reconstruction and norm clipping; parallel J-lens transport; static and dynamic-width exact instrument top-eight; sampling top-k capacities 1 through 1024; and sparse curve-mask updates. Shapes include non-multiple hidden widths, tied scores, tail blocks, multiple positions, and vocabulary widths up to 262144. Token indices must match the CPU ordering exactly, including lower-ID tie breaks. Numeric comparisons use fixed absolute and relative tolerances, declared in the verifier before execution. Validation errors and device loss fail the run.

These are kernel checks. They do not attest a released model, signed catalog, optional fitted instrument pack, or physical iPhone. Rebuilding a model changes its runtime identity; fitted pack and release provenance must be rebuilt and validated through the existing release tools before deployment.
