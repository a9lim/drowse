# Browser compiler image

Build the exact native Linux feasibility toolchain from the repository root:

```bash
docker build \
  --file browser-runtime/compiler/Dockerfile \
  --tag drowse-browser-compiler:feasibility \
  browser-runtime
```

The recipe supports `linux/amd64` and `linux/arm64` builds with separately pinned
official LLVM archives, so TVM and LLVM compile natively on each platform.
Emscripten 3.1.56 does not publish Linux ARM64 binaries, so the ARM64 image runs only
the pinned Emscripten tools through AMD64 translation. On Apple silicon with Colima,
use Rosetta for that narrow compatibility layer:

```bash
colima stop
colima start --vm-type vz --vz-rosetta
```

The build pins the Ubuntu base digest, LLVM archive, Emscripten SDK commit,
TVM/TVM-FFI sources, MLC source, and both Drowse overlays. Its final build gate
runs the C VM hook golden and LLVM fp16-capture golden, then compiles fp32 and
production-q4f16 Qwen3 and Llama WebGPU fixture libraries.

The local image ID is feasibility evidence only. A production runtime lock must
name an immutable registry reference and manifest digest for the published
image; it must not use a local image ID.

The image also contains `/opt/drowse/build-production-webgpu.py`. Its `convert`
command accepts only an exact Hugging Face snapshot whose required files carry
the requested 40-character revision in the local Hub metadata. It writes a new
q4f16 tensor cache, tokenizer/config closure, source notices, the complete
model-license text, and a
content-addressed `drowse-build.json`; it never overwrites an output directory.
The `compile` command re-verifies that complete manifest, compiles into a
same-filesystem staging directory, rejects a library missing any Drowse hook or
capture export, and promotes only the validated WASM to the requested path.

The pinned TVM/MLC pipeline does not currently emit byte-identical WASM across
otherwise identical invocations: serialized VM and generated-function ordering
can vary while the verified model behavior remains unchanged. Treat the emitted
SHA-256 as the identity of that exact release candidate. Do not promote the
runtime lock until either the compiler output is canonicalized reproducibly or
the release process records and physically validates one immutable emitted
library per model variant.

For an exact upstream MLC closure, first rewrite its browser-unsafe capacity
defaults and bind it to the Drowse build manifest. The source metadata directory
must be the immutable source-model snapshot containing the tokenizer files:

```bash
npm --prefix webui run adopt:hosted:mlc-model -- \
  qwen3-4b mlc-ai/Qwen3-4B-q4f16_1-MLC UPSTREAM_COMMIT \
  /absolute/mlc-snapshot /absolute/qwen3-source-metadata \
  /absolute/output/qwen3-4b-converted

docker run --rm \
  -v /absolute/output/qwen3-4b-converted:/model:ro \
  -v /absolute/output:/output \
  drowse-browser-compiler:feasibility \
  'python /opt/drowse/build-production-webgpu.py \
    --mlc-repository /opt/mlc-llm --tvm-repository /opt/tvm compile \
    --model /model --architecture qwen3 --quantization q4f16_1 \
    --structured-hook-profile standard-v3 \
    --output /output/qwen3-4b-model.wasm'
```

The adoption step fixes the production closure at context 4096, prefill chunk
2048, and batch size 1. The converter remains available when no exact upstream
MLC tensor closure exists.
