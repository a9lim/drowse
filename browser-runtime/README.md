# Drowse browser runtime

The hosted application must not infer with stock WebLLM. Production loading is
enabled only when `runtime-lock.json` has `status: "verified"` and the exact
package, overlay, toolchain, model, and compiled-library closure is pinned.

The runtime lock is `verified`. Exact converted-model revisions, compiled
libraries, package archive, overlay closure, runtime ABI, hook ABI, tokenizer,
and chat-template identities are bound by digest. The public signed catalog
contains only immutable packages that match those identities.

The browser core-pack compiler produces the required neutral whitener and seed
manifold pack for each release model. Its raw rank-one path shares Python
goldens for Mahalanobis normalization, mean-normalized per-layer share, and
uncapped affine gain. The structured path lowers independent
push/ablation groups, prior-step gates, response phases, SAE and J-lens
programs, box/custom/sphere curved manifolds, and exact v3 geometry reads.
Curved steering retains its separate safety cap. Every published pack passes the
semantic validator and a physical WebGPU runtime check before catalog inclusion.

The checked-in TypeScript parser accepts the full Python steering grammar. The
current compiler supports rank-one and composed affine pushes, mixed or
multiaxis mean-replacement ablation, Mahalanobis `~`/`|` projections,
response/prompt/count/thinking phases, prior-step scalar gates, and up to four
curved terms per layer. Curves may use bounded, unbounded, periodic, or spherical
domains and may have up to four intrinsic dimensions. Curved steering can itself
be phase-gated or controlled by a prior SAE, J-lens probability, scalar affine
probe, or exact geometry channel from the preceding token. The `standard-v2`
and `standard-v3` profiles emit `drowse-structured-v3`: in addition to four independently gated
affine groups, rank eight, eight scalar probes, four curves, thirty-two nodes per
curve, four intrinsic dimensions, and eight embedded dimensions, it reserves
eight geometry probes, whitener rank 96, thirty-three candidates per probe, and
a separate 41-float output record per layer and geometry probe. The older
`standard-v1` profile continues to emit `drowse-structured-v2` and fails closed
for multidimensional, curved, membership, and node-distance geometry reads.
`standard-v3` additionally attests `exact-readout-v1`, a fixed top-eight
full-vocabulary J-lens readout and chunked full-dictionary SAE readout. A
`standard-v2` library remains valid for geometry but is rejected before loading
a precomputed J-lens or SAE pack.

The v3 geometry path produces domain coordinates, subspace fraction,
normalized off-surface residual, tube membership, and raw whitened distance to
every candidate. The host derives normalized nearest-label distances and the
Gaussian assignment posterior from fixed, validated schema data. Coordinate,
fraction, membership, distance, and assignment gates therefore use exact
previous-token values instead of scalar approximations. Sphere fits use an
explicit domain tag and the same hyperspherical embedding, Jacobian, clamping,
log/parallel-transport/exp translation, and nearest-foot solve as the Python
domain. These TypeScript and compiled-kernel contracts are implemented but do
not by themselves constitute production-model or physical-device verification.

Optional instrument packs are executable inputs rather than capability
placeholders. A local SAE v1 fp32 artifact provides `W_dec[id]` steering and the
exact `relu((h - b_dec) @ W_enc + b_enc)` probe used by `sae/<id>` gates. A
pack may also include `packs/sae/features.json`, an exact v1 sidecar with
`{format_version, model_id, release, features}` and sparse
`{label, max_act}` rows. The loader rejects the complete pack when any row is
malformed or outside the SAE width. When `max_act` exists, the GPU probe, gate,
and scalar reading use `activation / max_act`; discovery rows retain the raw
activation and carry the same label and unit. Packs without the sidecar retain
the Python-compatible raw-activation/null-metadata behavior. Live discovery
evaluates the complete installed encoder on the GPU in bounded feature chunks
and reports the exact global top eight; it does not train or modify the SAE. A
local J-lens v6 artifact provides its unchanged `J_l` shards. The worker uses
the model's complete quantized or fp32 language head to compute an exact
full-vocabulary softmax on the GPU, reports exact per-layer and layer-aggregated
top-eight rows, and selects requested token probabilities for probes and gates.
Steering selectors are not limited to the pack's curated word
list: before synchronous lowering, the worker resolves each requested word
with the loaded tokenizer, requires the Python-compatible leading-space or bare
form to round-trip as exactly one token, and derives `W_U[v] @ J_l` through the
loaded GPU runtime in batches of at most eight token IDs. Partial J-lenses are
passed in fitted source-layer order and partitioned into dynamic
`[chunk, hidden, hidden]` inputs bounded by the smaller of the adapter's buffer
and storage-binding limits. The runtime conservatively uses one matrix per
chunk because larger dynamic J-lens batches are not stable across supported MLC
WebGPU VMs. The worker uploads every validated fitted matrix exactly once when
it installs the structured hook program and keeps the chunks GPU-resident for
that program's lifetime. Each probability read or gate then makes one exact
full-vocabulary call per chunk, combines all readbacks behind one device
synchronization, and scatters the rows back to model-layer order.
Program replacement, clearing, unload, device loss, and worker termination
dispose the resident inputs. Loading a J-lens pack fails before generation if
even one fp32 Jacobian cannot fit either adapter limit. Instrument artifacts are
checked against the catalog closure, schema, tensor keys, fp32 shapes, layer
map, and digests before lowering. During program installation the bridge
borrows the validated host array only while constructing the bounded chunks; it
neither retains the complete lens nor re-uploads Jacobians on token reads.

Qwen chat models carry tokenizer-derived `<think>` delimiter IDs in the runtime
lock. Loading retokenizes the delimiters with the installed tokenizer and
attests the result before the generation state machine enables thinking. This
permits exact `@thinking` and `@after` control changes at streamed token
boundaries. Models without an attested thinking profile reject thinking-specific
programs before generation. The browser derives required features from every
expression and fails before generation when a loaded library or pack cannot
implement them exactly.

## Runtime source overlays

`forks/manifest.json` binds the current Drowse MLC-LLM, WebLLM, and TVM WebGPU
changes to exact upstream commits, base trees, patch bytes, and resulting file
digests. The `a9lim/mlc-llm-drowse` and `a9lim/web-llm-drowse` repositories may
mirror those result trees, but a mirror commit is not part of the executable
release identity. The checked-in overlay closure is reproducible without
depending on mutable repository state.

The runtime lock hashes this manifest and the compiler Dockerfile, repeats the
exact TVM, TVM-FFI, Emscripten, and LLVM identities, and binds the vendored
WebLLM archive by its exact SHA-256, package name, version, and repository. The
converted model manifests bind the compiled WebGPU libraries by SHA-256. A
published compiler image or fork mirror can be recorded when one exists, but
neither is required to run or reproduce the checked-in overlay and artifact
closure.

Apply an overlay only to a clean checkout at its exact upstream commit:

```bash
node browser-runtime/forks/apply-fork-overlay.mjs \
  mlc-llm-drowse /path/to/mlc-llm --check
node browser-runtime/forks/apply-fork-overlay.mjs \
  mlc-llm-drowse /path/to/mlc-llm --apply
```

Use `web-llm-drowse` for the WebLLM overlay. Apply the TVM backport before
building the compiler:

```bash
node browser-runtime/forks/apply-fork-overlay.mjs \
  tvm-webgpu-readonly /path/to/tvm --apply
```

The command rejects a different commit or tree, any tracked or untracked
worktree change, a modified patch, and any resulting file whose digest differs
from the manifest.

## Hosted repository provisioning

The runtime and distribution locks target the `logitsml` Hugging Face namespace.
Changing this configuration does not transfer repositories or publish artifacts.
The pinned revisions must exist there, and the catalog must be signed for the new
artifact URLs before downloads can work. Check repositories without mutation, or
create any missing public repositories with the authenticated owner account:

```bash
cd webui
npm run provision:hosted:check
npm run provision:hosted:create
```

After converting and compiling a model, assemble its exact runtime closure and
validate it against `runtime-lock.json`:

```bash
npm run assemble:hosted:model -- CONVERTED_DIR MODEL.wasm PUBLISH_DIR
npm run build:hosted:whitener -- MODEL_ID PUBLISH_DIR MODEL.wasm WEBLLM.js CORE_DIR --manifold-source ../browser-runtime/release-inputs/core-manifolds/welcoming.detached
npm run validate:hosted:core-pack -- MODEL_ID CORE_DIR
npm run build:hosted:sae -- MODEL_ID PUBLISH_DIR MODEL.wasm WEBLLM.js CORPUS.json SAE_DIR
npm run build:hosted:jlens -- MODEL_ID PUBLISH_DIR MODEL.wasm WEBLLM.js CORPUS.json WORDS.json JLENS_DIR
npm run capture:hosted:residuals -- MODEL_ID PUBLISH_DIR MODEL.wasm WEBLLM.js INPUT.json CAPTURE_DIR
python3 ../browser-runtime/validate_mlc_q4_parity.py PUBLISH_DIR CAPTURE_DIR
npm run assemble:hosted:pack -- PACK_DIR
npm run validate:hosted:instrument-pack -- MODEL_ID sae|jlens PACK_DIR
npm run publish:hosted:model -- MODEL_ID PUBLISH_DIR
npm run publish:hosted:pack -- MODEL_ID core|sae|jlens PACK_DIR
```

The production-q4 comparator reports probe error against the activation scale:
for each layer it divides the absolute scalar-projection difference by the
larger residual norm times the probe-direction norm. This keeps the 1% gate
well-defined when a valid probe projection is near zero. It separately requires
the browser probe readback to reproduce the projection of its own captured
residual and reports the cancellation-sensitive whole-trace ratio as a
diagnostic only. Schema-v3 captures also run a disabled and enabled program
through the same rank-one VM function and reject any pre-enabled-layer delta.

Candidate libraries that do not yet match the checked-in production lock can be
captured against an explicit engineering lock without editing repository state:

```bash
npm run prepare:hosted:candidate-lock -- MODEL_ID PUBLISH_DIR /tmp/candidate-runtime-lock.json
npm run capture:hosted:residuals -- MODEL_ID PUBLISH_DIR MODEL.wasm WEBLLM.js INPUT.json CAPTURE_DIR \
  --runtime-lock /tmp/candidate-runtime-lock.json
npm run build:hosted:whitener -- MODEL_ID PUBLISH_DIR MODEL.wasm WEBLLM.js CORE_DIR \
  --manifold-source ../browser-runtime/release-inputs/core-manifolds/welcoming.detached \
  --runtime-lock /tmp/candidate-runtime-lock.json
npm run validate:hosted:core-pack -- MODEL_ID CORE_DIR \
  --runtime-lock /tmp/candidate-runtime-lock.json
```

The preparation tool starts from the repository lock, validates the assembled
model closure, and writes a new lock with only that model's library digest
replaced. It refuses to overwrite an existing file. The capture records both the
exact-byte and canonical status-free lock digests and marks the source as
`override`. It revalidates the complete lock schema, model identity, build
provenance, and every local artifact before Chrome starts. An override capture
can be compared locally, but the parity validator refuses to turn it into a
release-evidence receipt.

Instrument builders can optionally record a corpus closure produced by
`prepare_hosted_release_corpus.py` for research bookkeeping:

```bash
npm run build:hosted:sae -- MODEL_ID PUBLISH_DIR MODEL.wasm WEBLLM.js \
  RELEASE_DIR/sae-corpus.json SAE_DIR \
  --release-corpus RELEASE_DIR/release-corpus.json \
  --release-corpus-source APPROVED-SOURCE.json \
  --release-license-evidence LICENSE-EVIDENCE
npm run build:hosted:jlens -- MODEL_ID PUBLISH_DIR MODEL.wasm WEBLLM.js \
  RELEASE_DIR/jlens-corpus.json RELEASE_DIR/jlens-words.json JLENS_DIR \
  --release-corpus RELEASE_DIR/release-corpus.json \
  --release-corpus-source APPROVED-SOURCE.json \
  --release-license-evidence LICENSE-EVIDENCE
npm run validate:hosted:instrument-pack -- MODEL_ID sae|jlens PACK_DIR --release
```

When supplied, the builders verify and copy those bookkeeping files. They are
not required for catalog generation or upload. Production admission instead
uses the ordinary instrument validator: exact runtime/context compatibility,
safe tensor shapes and dtypes, finite values, readable manifests, and successful
loading by the browser instrument registry.

The offline SAE pack builder mirrors Python's local-training default: it rounds 65%
model depth with Python's ties-to-even rule, then resolves that slot through the
runtime lock's `layerMap`. The current locks select layer 18 for Qwen3-1.7B and
layer 23 for Qwen3-4B. An explicit `--layer` overrides the default and must name
a layer in that map.

The publisher is check-only unless `--upload` is supplied. Uploading creates a
Hugging Face commit, so it must be an explicit release action:

```bash
npm run publish:hosted:model -- MODEL_ID PUBLISH_DIR --upload
npm run publish:hosted:pack -- MODEL_ID core|sae|jlens PACK_DIR --upload
```

Record the returned immutable revision as `convertedRevision`, generate the
catalog artifact list from that revision, and rerun the distribution preflight.
Model publication alone is insufficient: the matching required core pack,
signed catalog, and distribution lock must also be published before production
loading is enabled. Precomputed J-lens and SAE packs are optional model-specific
capabilities and are listed only when a compatible release already exists.

The whitener builder is a physical WebGPU release tool. It renders the bundled
96-row neutral corpus through the exact q4 browser runtime, captures the
post-block residual from every compiled layer, fits the reduced Mahalanobis
representation with the shipped Rust/WASM kernel, and writes separately bound
2K and 4K `neutral-whitener` pairs. It refuses an existing output directory and
checks the model manifest, runtime identity, layer map, transfer digest, and
context bindings. The required `--manifold-source` must name a v10 PCA discover
folder; the builder captures and fits its nodes through the same q4 runtime and
produces the seed browser-fitted `.drowse` needed for first-run steering. The
independent validator checks both context closures and the complete browser
archive before the pack manifest is assembled.

The same checked-in `welcoming.detached` source is used for all three model
builds. Its 48 response pairs are aligned to the canonical baseline prompts,
carry machine-readable authorship and CC0-1.0 provenance, and contain no fitted
model tensors. Each build produces a separate runtime-fingerprinted archive.

The SAE builder is offline developer release tooling and is never shipped as a
hosted user feature. It renders a release
corpus through the same exact q4 browser runtime, captures explicit token
positions at the selected post-block layer, streams the fp32 activations to a
bounded local file, trains only the ReLU SAE parameters outside the hosted
application, emits the existing local-v1 schema, assembles its immutable file
closure, and reopens the first and last features through the production browser
registry. `--feature-metadata FEATURES.json` adds the validated optional
sidecar to that closure. Raw activation rows remain local and are removed after
the validated pack is produced.

The J-lens builder is likewise offline developer release tooling and is never
shipped as a hosted user feature. It renders and tokenizes its release corpus in the exact q4
browser runtime, captures every post-block residual at every retained token
position, and streams those rows to a bounded local file. The offline fitter
dequantizes the same MLC tensor cache, reproduces its fp16 accumulation on MPS,
and forward-corrects each block onto the captured browser trajectory while
retaining the dequantized graph's derivative. It writes deterministic local-v6
fp32 layer shards plus exact dequantized language-head rows for a curated
single-token seed vocabulary used by the UI and pack preflight. The production
registry also resolves arbitrary compatible single-token selectors against the
loaded model's language head; the curated rows are not a selector allowlist.
The builder assembles the pack and reopens it through that registry. The capture
and fit are offline developer release tooling; J-lens fitting remains
unavailable inside the hosted application itself. Neither builder is imported
by the PWA or callable through its worker protocol: browser users can only
download, validate, activate, read, probe, gate, and steer with compatible
precomputed packs.

Browser J-lens packs retain up to eight evenly spaced source layers from an
existing provider lens, including the first and last available block. This
preserves a depth-spanning readout while bounding download size, browser memory,
and GPU uploads. The launch catalog does not fit J-lenses; it packages compatible
precomputed provider artifacts and records the exact source-layer map.

After building a model, core pack, and J-lens pack, the physical instrument
harness loads all three through the same browser loaders, enables live J-lens
probabilities, transfers the complete per-layer Jacobian program once into
bounded GPU-resident chunks, and generates with a J-lens steering term without
making an external request:

```bash
npm run test:browser-runtime:instruments -- \
  --model-id qwen3-1.7b \
  --model-directory MODEL_DIR --model-library MODEL.wasm --webllm WEBLLM.js \
  --core-directory CORE_DIR --jlens-directory JLENS_DIR \
  --sae-directory SAE_DIR
```

Use the full production-model harness for one fail-closed Chrome pass over the
complete local runtime path:

```bash
npm run test:browser-runtime:full -- \
  --model-id qwen3-1.7b \
  --model-directory MODEL_DIR --model-library MODEL.wasm --webllm WEBLLM.js \
  --core-directory CORE_DIR --jlens-directory JLENS_DIR \
  --sae-directory SAE_DIR --context-tokens 2048 \
  --output /tmp/drowse-real-browser-e2e.json
```

The command rehashes every local manifest object before opening Chrome, requires
a confirmed hardware adapter with at least 10 storage buffers per shader stage,
then measures ordinary generation, selective
post-block capture, exact forced-token replay and compact cross-token scoring,
an alternate-token fork, flat steering gated independently by geometry, J-lens
probability, and SAE activation signals, J-lens and SAE steering/readout,
a small spectral fit backed by the OPFS activation spool, nearest-node role
selection and named-role generation, curved steering,
streamed stop, unload/reload, and a final reload from reopened OPFS files while
all HTTP(S) fetches are blocked. It fails when any phase is absent, reordered,
unsupported, unmeasured, or emits a WebGPU/device-loss error. There is no
`--skip-fitting` or missing-instrument mode.

Its JSON is deliberately marked `releaseEvidence: false`: one local desktop
run is useful physical integration evidence, but it cannot claim Android,
cross-platform, numeric-parity, or full release-matrix results. Release records
still go through `record:hosted:evidence` and the revision-bound evidence
schemas.

Build the catalog from an exact release spec only after every model and pack has
an immutable revision. Model directories contain `hosted-artifacts.json`; each
core or optional-instrument directory contains `hosted-pack-artifacts.json`
with exact relative paths, byte sizes, and SHA-256 digests. The builder rehashes
every file, closes all runtime-lock context profiles, derives runtime/context
identity digests, and validates the completed catalog with the browser parser:

```bash
npm run build:hosted:catalog -- RELEASE_SPEC.json catalog.json
node scripts/sign-catalog.mjs catalog.json PRIVATE_KEY.pem KEY_ID catalog.sig.json
npm run publish:hosted:catalog -- catalog.json catalog.sig.json
```

Apply `tvm-webgpu-readonly` to the pinned TVM checkout before building its
compiler. Its read-only buffer fix was merged as Apache TVM PR 20113. Without
that fix, typed buffer aliases are emitted as `read_write`, and current Chrome rejects
the generated paged-KV shader because a `workgroupBarrier` appears under control
flow derived from that storage binding.

The same overlay also carries a local attention numerical fix, separate from
that upstream PR: GPU prefill and decode initialize their running maximum at
the minimum finite float32 value and assign masked positions exactly zero
softmax weight. The former `-50000` cutoff is above valid attention scores in
Pythia-70M, corrupting both attention normalization and masked-token exclusion.
The compiler verifies the exact hashes of both modified kernel sources.

MLC commit `9fa644f5` merged typed-buffer compiler passes after its checked-in
TVM gitlink was last updated. Those passes explicitly depend on Apache TVM
`68ba2b31`, so the feasibility toolchain pins that external TVM commit and its
`tvm-ffi` submodule rather than incorrectly compiling against MLC's older
gitlink. The overlay manifest records the exact TVM base, upstream WebGPU
backport, TVM-FFI, Emscripten, and LLVM identities.

The optional exact-source MLC check requires that pinned source-built TVM
compiler. It fails when the toolchain is unavailable instead of reporting a
skipped gate:

```bash
TVM_DEVICE_BACKEND_AUTOLOAD=0 \
TVM_LIBRARY_PATH=/path/to/exact-tvm-build/lib \
SKIP_LOADING_MLCLLM_SO=1 \
PYTHONPATH=/path/to/tvm/python:/path/to/mlc-llm/python \
  /path/to/venv/bin/python browser-runtime/forks/verify-mlc-hook.py \
    --mlc-repository /path/to/mlc-llm \
    --tvm-repository /path/to/tvm
```

That check proves the exact Qwen3, Llama, and Gemma 3 graph exports contain the four
steering and four selective-capture entry points and executes the shared
rank-one and fp32 capture goldens through a compiled C TVM VM. It does not
prove WebGPU, q4, lifecycle, long-prefill, or physical-device feasibility.

For the WebGPU compile gate, first source exact Emscripten 3.1.56 and rebuild
both support runtimes against the pinned TVM checkout:

```bash
export TVM_SOURCE_DIR=/path/to/tvm
make -C /path/to/mlc-llm/web clean
make -C /path/to/tvm/web clean
(cd /path/to/mlc-llm && ./web/prep_emcc_deps.sh)
```

Then compile each architecture fixture with the LLVM-18-enabled TVM compiler:

```bash
TVM_DEVICE_BACKEND_AUTOLOAD=0 \
TVM_LIBRARY_PATH=/path/to/tvm-build/lib \
PYTHONPATH=/path/to/tvm/python:/path/to/mlc-llm/python \
  /path/to/venv/bin/python browser-runtime/forks/compile-tiny-webgpu.py \
    --mlc-repository /path/to/mlc-llm --tvm-repository /path/to/tvm \
    --architecture qwen3 --output /tmp/drowse-tiny-qwen3.wasm
```

Repeat with `--architecture llama` and `--architecture gemma3_text` for the
architecture regression fixtures. The compiler requires a new output path,
validates the resulting WebAssembly with the matching Emscripten Binaryen tool,
and verifies that every rank-one, structured, curved, capture, and J-lens VM
function remains in the library. The three launch models use `q4f16_1`; pass
`--context-window-size 4096` for the long-prefill Qwen3 fixture.

Create the matching deterministic tensor cache with the same architecture,
quantization, and context arguments. The fixture uses nonzero embeddings and
unit normalization weights so the browser gate exercises an actual rank-one
residual change; q4 caches use valid offset-packed int4 values and scales:

```bash
TVM_DEVICE_BACKEND_AUTOLOAD=0 \
TVM_LIBRARY_PATH=/path/to/tvm-build/lib \
PYTHONPATH=/path/to/tvm/python:/path/to/mlc-llm/python \
  /path/to/venv/bin/python browser-runtime/forks/create-tiny-webgpu-model.py \
    --mlc-repository /path/to/mlc-llm --tvm-repository /path/to/tvm \
    --architecture qwen3 --quantization q4f16_1 \
    --output /tmp/drowse-tiny-qwen3-model
```

After building the WebLLM overlay, run the compiled library through a dedicated
browser worker:

```bash
node webui/scripts/browser-runtime-feasibility.mjs \
  --webllm /path/to/web-llm/lib/index.js \
  --model /tmp/drowse-tiny-qwen3-model \
  --library /tmp/drowse-tiny-qwen3.wasm \
  --architecture qwen3 --quantization q4f16_1 --reloads 3
```

Pass `--browser-channel chrome` or `--browser-channel msedge` to run the same
gate against an installed production browser instead of Playwright's bundled
Chromium.

The harness fails on uncaptured WebGPU errors and checks ordinary generation,
rank-one and structured affine injection, SAE probes, live gates, simultaneous
curves, multidimensional and periodic curves, full-vocabulary J-lens
probabilities, hook clearing, streamed stop, repeated load/unload, and worker
termination. For the shape-cache regression, compile the Qwen3 q4f16 fixture
at a 4096-token context and run separate
`--prompt-tokens 2048` and `--prompt-tokens 4095` cases. That result belongs to
the Qwen long-prefill regression; Qwen3-1.7B and Qwen3-4B also have independent
tiny-fp32 and production-q4 parity coverage.

Pass `--verified-artifacts` to preload each fixture object exactly once and
provide it to the fork through a caller-owned, read-only artifact cache. This
exercises configuration, tokenizer, model-library, and tensor-shard loading
without allowing WebLLM to create a second browser cache. The harness rejects
any missing object, any unexpected URL, or a repeated network fetch. This mode
uses a worker-side configured `MLCEngine`; caller-owned caches are deliberately
not sent through structured-clone worker messages.

Add `--selected-adapter --reloads 1` to prove that the production device is
requested from the exact adapter supplied by Drowse. WebGPU adapters are
single-use for device creation in current Chromium, so the hosted capability
gate destroys its calibration device, then acquires a fresh adapter and rejects
it unless the hardware status, identity, features, and bucketed limits match the
checked adapter. Repeated load/unload testing likewise requests a fresh adapter
per load.

These deterministic tiny checks prove fp32 and q4 browser execution only. They
do not satisfy production-model numeric parity, Android, device-loss, or the
physical release matrix.

Production-size compilation must also exercise a launch model with its real
layer count. A two-layer fixture cannot detect per-stage WebGPU binding growth.
Probe and capture aggregation therefore write every layer into one in-place
output tensor; returning a single `concat` over all layer tensors is forbidden
because a 28-layer Qwen model would require 29 storage bindings in one shader,
above the baseline WebGPU per-stage limit. The physical harness discovers all
tokenizer files and tensor shards from the converted metadata and requires each
caller-owned artifact to be fetched exactly once.

Every converted `mlc-chat-config.json` must also contain a sorted, unique
`drowse_capture_special_token_ids` array produced from the source tokenizer's
complete `all_special_ids` and `added_tokens_encoder` union. Browser loading
checks that this list covers every `tokenizer.json` added-token ID and each
configured BOS, EOS, and padding ID before capture can run. This prevents
last-content pooling from silently depending on tokenizer heuristics.

`distribution-lock.json` independently gates the signed catalog and download
origins. It embeds the current and next Ed25519 public keys, the minimum catalog
sequence, and the exact Hugging Face artifact redirect origin exercised by the
full distribution preflight. Its verified state authorizes catalog downloads.

`runtime-feasibility-evidence.json`, `authoring-evidence.json`, and
`benchmark-evidence.json` remain optional engineering records. They are useful
for regression tracking and support-policy decisions, but their status,
producer registrations, source ancestry, and matrix completeness do not enable
or block downloads, generation, fitting, or a release build.

The current launch catalog contains Gemma 3 270M, Qwen3-1.7B, and Qwen3-4B. Each release model
must include an already-released compatible J-lens, selected automatically in
the first-run flow. The UI advertises only context profiles physically exercised for each
model. Authoring-evidence ledgers and a fixed 48-cell benchmark matrix are not
download gates, but each published model still requires a real-browser load,
generation, steering, unload, and offline-reload pass. Models that list an
existing J-lens or SAE pack also require the matching browser instrument pass.

## Release transaction

1. Exercise the advertised context profiles for all three launch models.
2. Publish each model's immutable converted files, required core pack, and an
   already-released exact-model J-lens that passes the browser pack validator.
   Add compatible precomputed SAE packs when available; never fit either
   instrument as part of the browser release process.
3. Fill the model revision and artifact SHA-256 fields in `runtime-lock.json`
   and set its status to `verified`.
4. Generate and sign the catalog, fill the exact Hugging Face redirect origins,
   and set `distribution-lock.json` to `verified`.
5. Run the release checker and distribution preflight:

   ```bash
   node webui/scripts/check-runtime-lock.mjs --release
   node webui/scripts/preflight-hosted-distribution.mjs
   ```

Parity, authoring, lifecycle, and cross-platform runs remain ordinary regression
tests and local reports; they are not part of the release transaction.

`runtimeIdentitySha256` is SHA-256 over canonical, recursively key-sorted JSON
for the runtime identity record. A context binding is the same digest over
`{"contextTokens":N,"runtimeIdentitySha256":"..."}`. Catalog validation
recomputes both values before a pack can be activated.

Each runtime identity digest is bound to exactly one catalog artifact with the
matching `converted_manifest`, `tokenizer`, `chat_template`, or
`model_library` role. Weight shards are closed by the converted manifest; a
signed identity claim that does not match those artifact digests is rejected.

The catalog signing key is never stored in this repository. The signing tool
accepts an Ed25519 PKCS#8 PEM key path and signs the exact `catalog.json` bytes.
## Cloudflare Pages deployment

Cloudflare credentials, Pages project creation, Access policy, DNS, deployment,
and production publication are external release gates. The repository builds and
verifies static output but does not perform any of those actions automatically.
`webui/wrangler.jsonc` names `webui/dist-hosted` as the upload directory and
contains no credentials.

### Access-controlled preview

1. From `webui`, run `npm ci` and `npm run build:hosted`. This always emits the
   feasibility-preview copy, `X-Robots-Tag: noindex, nofollow`, and a blocking
   `robots.txt`.
2. Upload `webui/dist-hosted` to a Cloudflare Pages preview deployment and put
   the preview hostname behind a Cloudflare Access policy. Do not promote this
   build to the public production hostname.
3. Verify the deployed hostname, including Pages-applied headers, protection of
   `/`, `/app`, the manifest, and service worker from unauthenticated requests,
   and offline navigation:

   ```bash
   npx playwright install chromium
   export CF_ACCESS_CLIENT_ID=...
   export CF_ACCESS_CLIENT_SECRET=...
   npm run verify:hosted:deployment -- https://PREVIEW_HOST --channel preview
   ```

   For non-interactive verification behind Access, provision a scoped service
   token outside the repository and expose both `CF_ACCESS_CLIENT_ID` and
   `CF_ACCESS_CLIENT_SECRET` only to this command. The verifier sends the pair
   to same-origin requests, browser navigation, and service-worker requests;
   cross-origin browser requests are blocked so the token cannot leave the
   preview origin, and the verifier never prints it.

### Public release

1. Complete the runtime artifact, signed distribution, and hosted security
   checks above. Authoring evidence, source ancestry attestations, and a complete
   physical benchmark matrix are not required.
2. From `webui`, run `npm ci` and `npm run build:hosted:release`. The release
   build fails closed unless the locks and distribution preflight pass, emits
   public indexing policy, and records the exact 40-character source revision.
3. Upload the resulting `webui/dist-hosted` without rebuilding it to an
   Access-protected Pages preview. Configure a preview-host-only Cloudflare
   Response Header Transform Rule that sets `X-Robots-Tag: noindex, nofollow`;
   do not apply that rule to the production hostname. Provision a scoped Access
   service token outside the repository and run the candidate check. This
   verifies release metadata while requiring both Access protection and the
   externally applied noindex header:

   ```bash
   npx playwright install chromium
   export CF_ACCESS_CLIENT_ID=...
   export CF_ACCESS_CLIENT_SECRET=...
   npm run verify:hosted:deployment -- https://RELEASE_CANDIDATE_HOST \
     --channel candidate
   ```

4. Publish the same byte-identical `dist-hosted` artifact to the production
   environment and remove Access only from the intended public hostname. Unset
   the service-token environment variables, then prove the public deployment is
   anonymously reachable and no longer carries noindex:

   ```bash
   unset CF_ACCESS_CLIENT_ID CF_ACCESS_CLIENT_SECRET
   npm run verify:hosted:deployment -- https://RELEASE_HOST --channel release
   ```

The post-deploy verifier accepts only an HTTPS origin. Preview and candidate
checks require Access service credentials and prove that the shell's entry
routes and bootstrap assets deny unauthenticated requests. Candidates must have
release metadata, an independently supplied source revision, and noindex.
Public release checks forbid Access credentials and noindex. `--revision` is an
optional spot check, not a release prerequisite. Every mode checks `/`, `/app`,
the complete deployed COOP/COEP/CORP/CSP/Permissions-Policy set,
indexing policy, the exact required PWA icon set, asset MIME and cache headers,
route rendering, `crossOriginIsolated`, service-worker readiness, and a
never-before-fetched offline `/app` URL. A successful local build does not
substitute for this check.
