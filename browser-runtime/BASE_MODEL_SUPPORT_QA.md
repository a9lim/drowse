# Base-model integration status

**Latest status (2026-09-06, 16:06 UTC recheck):** All four base models pass signed installation, ordinary and core-steered generation, full-page saved-session reopening, and further generation at the original local app origin. Catalog sequence 7 fixes the existing-client rollback rejection. Additional fixes cover model reload approval, zero-probability replay persistence, and cross-model conversation isolation. The app is rebuilt locally, not deployed to a production website. The historical blocked recheck below is superseded by the final section.

The sections before “Signed base installs and long-context validation” record candidate work on 2026-09-05; their not-published statements do not describe the current release.

## Implemented

- GPT-2 and GPT-NeoX adapters use the upstream decoder blocks and preserve their checkpoint parameter paths. GPT-2 adds learned absolute positions using the cache's query positions before the first block. GPT-NeoX retains the upstream parallel-residual and partial-RoPE implementation. Both retain LayerNorm rather than substituting RMSNorm.
- Qwen 3.5's candidate adapter threads recurrent state through each native hybrid decoder layer and returns it alongside KV cache before instrumentation outputs. The candidate recurrent-state gather/scatter kernels avoid the pinned compiler's undefined local-buffer error in the upstream state kernels.
- An opt-in compiler entrypoint registers candidate implementations and their quantizers while preserving the upstream checkpoint loaders. Qwen requires the explicit `--hybrid-state-abi` flag; current production instrumentation consumers do **not** support its output layout yet.
- Base sessions always use raw text, including continuation and token replay. Chat-message arrays are rejected before generation. An old UI preference cannot turn a base session into a templated chat.
- The completion workspace offers multiline editing, Continue text, Save edit, Discard edit, and token inspection. Enter inserts a newline; Cmd/Ctrl+Enter continues text. Saving a boundary deletion selects the shorter Loom path. Editing below a shared parent creates a new node instead of modifying the original branch. Failed submissions retain the draft.
- `[BASE]` appears beside model names in the model picker, saved-chat selector, and workbench model card. Optional model-type metadata survives autosave and `.polytheticchat` export/import; legacy records remain accepted without guessing their type.

The UI and writing skill guidance informed the completion terminology, control grouping, and reuse of existing theme tokens rather than a separate visual theme for base models.

## Verification

| Check | GPT-2 | GPT-NeoX / Pythia | Qwen 3.5 |
| --- | --- | --- | --- |
| Native parameter names, shapes, dtypes preserved | Passed | Passed | Passed |
| All 25 instrumentation entry points export to TVM IR, FP32 | Passed | Passed | Passed |
| All 25 entry points export after q4f16_1 quantization | Passed | Passed | Passed |
| Compiled FP32 LayerNorm J-lens probabilities and directions versus NumPy | Passed | Passed | Not tested |
| Compiled recurrent-state gather/scatter versus NumPy | N/A | N/A | Passed |
| Real checkpoint generation / WebGPU parity | Not tested | Not tested | Not tested |

The recurrent-state test covers 1D, 2D, and 3D state tensors, noncontiguous sequence slots, history-ring wraparound, and preservation of untouched slots. The J-lens CPU test serializes GPU thread bindings to exercise arithmetic; it does not certify GPU scheduling. These tests use small synthetic configurations and deterministic tensors, not downloaded model checkpoints.

Browser testing used the development-only `?layoutFixture=base` route. Verified raw completion, saving a shortened path, continued generation, multiline Enter behavior, token-detail access, light/dark appearance, and the `[BASE]` badge in the saved-chat drawer. Its output is a deterministic fixture, not real Pythia inference.

Svelte checks, browser-Loom tests, engine-adapter tests, conversation-library tests, backup round trips, base-mode enforcement tests, theme/contrast/interface checks, and hosted/native builds passed. Builds retain the existing large-chunk warning.

## Reproduce compiler checks

Use the project's patched MLC/TVM compiler environment. `--mlc-repository` points to its MLC source tree; the adapters require the current `polythetic_*` hook names, not an older `saklas_*` overlay.

```sh
python browser-runtime/forks/verify-base-model-adapters.py --mlc-repository /path/to/mlc-llm --architecture gpt2 --readout-golden
python browser-runtime/forks/verify-base-model-adapters.py --mlc-repository /path/to/mlc-llm --architecture gpt_neox --readout-golden
python browser-runtime/forks/verify-base-model-adapters.py --mlc-repository /path/to/mlc-llm --architecture qwen3_5
```

Repeat for each architecture with `--quantization q4f16_1` (without `--readout-golden`). The script prints hashes of the adapter files and explicitly reports `real_model_inference: false`.

`forks/compile-base-candidate.py` accepts the normal MLC `compile`, `convert_weight`, and `gen_config` arguments. Qwen's local candidate registration additionally requires `--hybrid-state-abi`. This wrapper does not bypass production attestation or publish artifacts.

## Remaining before usable downloads

1. Compile actual WebGPU/WASM libraries and convert exact, pinned model checkpoints. Validate quantized prefill/decode logits, captures, EOS behavior, context overflow, and checkpoint-to-instrument coordinates. GPT-2's actual position limit is 1024, so the existing production builder's 2048/4096-only context choices need extending.
2. Complete the WebLLM hybrid instrumentation consumer: pass both states to capture and rank-one-capture calls; read measurements/captures after both returned states; test reset, cancellation, repeated prefill, branching, and replay. Ordinary hybrid generation support alone does not cover these calls.
3. Implement exact Top-K SAE encoding for the selected Pythia and Qwen packs, including global selection across chunks. ReLU is not a valid replacement. Resolve GPT-2 SAE preprocessing and capture-coordinate provenance before treating its dictionary as compatible.
4. Package and validate matching core/J-lens/SAE artifacts, measure browser memory/performance, then produce new release attestations and signed catalog entries. Existing instruction-model binaries and fingerprints must not be relabeled as base models.

The app deliberately continues to withhold unverified base downloads. This task has not established end-to-end production support for these three model families.

## Extended validation — 2026-09-05

This pass adds executable failure reproductions and target code generation. **No real checkpoint was generated with in a browser.** Compiler success is not a generation-parity result, and the CPU state tests do not validate the full GatedDeltaNet computation.

### Results

- All three candidate architectures pass FP32 and q4f16_1 WebGPU/wasm-target code generation through the MLC compiler pipeline, including all 25 instrumentation exports. These executables were not linked to a browser runtime or executed on a GPU. GPT-2 and NeoX use 256-wide synthetic configurations for this check; the smaller 8-wide attention heads used for their IR tests exceed a WebGPU decode-kernel thread limit and are not suitable GPU fixtures.
- GPT-2 and NeoX FP32 J-lens CPU probability/direction goldens pass again.
- Qwen state gather/scatter passes **72 scenarios**: FP32/FP16; 1D, 2D and 3D state; ring capacities 1, 2, 3 and 7; singleton, partial and full batches; reordered/noncontiguous sequence slots; more than two full ring cycles; exact preservation of all untouched storage. These compare compiled CPU kernels to NumPy after every update.
- The core-pack suite passes **67 tests**, including new rejection checks for `topk`, `top_k` and `batch_topk` SAE manifests. This verifies safe rejection, **not Top-K support**. There is still no K=16/K=50 global encoder to test, nor coverage for Top-K chunk merging, ties, normalization or steering through those dictionaries.
- The new Qwen consumer-contract suite runs actual methods extracted from the attested `llm_chat.ts` source with fake device objects: **40 pass, 6 fail**. Four failures reproduce omitted recurrent state in ordinary/rank-one capture, for both prefill and decode. Two reproduce capture/measurement reads at the wrong return offset. Single/batch KV-only and RNN-only controls and plain/steered hybrid forwarding pass. Injected capture failures balance both state-forward scopes without advancing the recorded cache length.
- The consumer source SHA-256 is `9f9ea3b62c96f5755fe591f22b06cf1b18574c7988fffb82d4f47eea921caa5f`, matching `forks/manifest.json`. The test is intentionally red while these production-consumer contracts remain broken; it is not added to the default passing suite.
- All **43 runtime scripts** were run independently to avoid fail-fast masking. Initially 41 passed. The streaming-generation script's manually initialized runtime lacked its capability record, so its concurrency test never reached generation. Supplying the fixture's existing capabilities fixes that test, bringing the result to **42/43**. The remaining worker-runtime failure is `missing fixture method manifolds.inspectSurface`; later tests within that script remain unverified.
- Additional full-runtime contract tests (12), hosted-SAE tooling and residual-capture tooling pass. Svelte/checks and the hosted build pass, with the existing large-chunk advisory. The full-runtime contract tests validate the harness, not a real browser inference run.

### Reproduce the added checks

```sh
# Run each architecture separately in the patched compiler environment.
python browser-runtime/forks/verify-base-model-adapters.py \
  --mlc-repository /path/to/mlc-llm --architecture gpt2 \
  --quantization q4f16_1 --webgpu-codegen
# Repeat with gpt_neox and qwen3_5. Omit --webgpu-codegen for IR/state tests.

cd webui
node scripts/browser-core-pack.test.mjs
node scripts/web-llm-generation.test.mjs
node scripts/qwen-hybrid-contract.test.mjs /path/to/web-llm/src/llm_chat.ts
```

`--readout-golden` and `--webgpu-codegen` are mutually exclusive. The latter runs code generation only and does not attest a toolchain, link a runnable browser library, publish files or modify production locks. The Qwen contract suite's nonzero exit is an expected **readiness failure**, not a passing integration result.

### Still required for the requested end-to-end validation

Implement global Top-K SAE encoding and hybrid capture/output handling first. Then link candidate WebGPU libraries, convert pinned checkpoints, load them through the actual browser consumer, and compare prefill/decode logits and captures against a reference. Real browser checks must cover EOS/context limits, cancellation, reset, repeated generation, Loom branching and forced replay without stale recurrent state. None of those real-model browser outcomes is established by this pass.

## Runtime repair and completion interface pass — 2026-09-05

This section supersedes the earlier red consumer-contract and missing fixture-method results. No public release or real-checkpoint browser inference is claimed.

### Implemented and packaged

- Hybrid capture and rank-one capture now pass both KV and recurrent state. Instrument outputs are read after the complete returned-state prefix, rather than assuming a single state. Ordinary instrumented generation uses the same offset convention.
- State-forward scopes and TVM scopes are closed after embedding, forward, and partial state-begin failures. The regression covers the case where the second state cannot begin without attempting to end a state that never began.
- The rebuilt runtime is `@polythetic/web-llm@0.2.84-polythetic.33`; the web app installs the new vendor tarball. The previous `.32` tarball was not overwritten. Fork overlay, manifest, package lock and runtime identity were regenerated together; no old GPU evidence was relabeled as validating the new runtime.
- Runtime package SHA-256: `f0c6cb71ea4bff6ce7a3fc7d55166658c217f24f106addcd7d74820e2fc63b2b`.
- Fork manifest SHA-256: `97ff7b31f5bd4f561a492748da9e2ffd43101fd3df7e03d9b6c8f445b37dff0f`.
- Runtime identity: `72d368fc8b9100c6773e519366ec57b505e8d29597ec9e66ac0ad9f7b2cbe446`.
- Fake artifact routing now exposes `manifolds.inspectSurface`; all 113 worker orchestration checks pass.
- Browser Loom treats `terminalReason: external_stop` as cancellation even when its own stop flag was not set. The stored node and UI preserve partial text and report stopped rather than complete.
- Base interface fixture sessions use unique roots to avoid test-run autosave collisions. The visible streaming delay is a fixture-only interaction aid, not a performance measurement.

### Verification and provenance

- Rebuilt WebLLM source: `npm run build` passes; Jest passes 20 suites / 300 tests. Logs: `/tmp/polythetic-hybrid-runtime-build.log`, `/tmp/polythetic-hybrid-runtime-jest.log`.
- `cd webui && node scripts/qwen-hybrid-contract.test.mjs`: **53/53** against the installed bundle. An optional path still allows the same tests against `src/llm_chat.ts`, which also passes. The harness extracts the actual consumer methods and compiled async helper; its TVM/device objects are fakes. It validates the calling contract and cleanup, **not GPU numerical correctness**. Log: `/tmp/polythetic-qwen-bundled-contract.log`.
- `npm run test:fork-overlays` and `node scripts/check-runtime-lock.mjs` pass. Release checking was not completed: `--release` requires an explicit release revision. No fresh release attestation was produced.
- `npm run check` passes, including Svelte with zero errors/warnings, contrast and theme policies, backup round trips, and base/instruct mode enforcement. `npm run build:hosted` passes, including 291-file isolation; the existing chunk-size advisory remains.
- Earlier full-chain runtime verification passed all 43 scripts. After adding the Qwen contract test, the first independent 44-script pass returned 43 successes plus a native SIGSEGV after conversation-library assertions completed. Other chained runs also encountered native Node/Rolldown SIGBUS/SIGSEGV in the instrument test. These were not assertion successes or GPU failures. Both small test harnesses now disable unnecessary dependency discovery, and the instrument test no longer loads the app configuration. Five consecutive conversation-library processes subsequently exited successfully. Final chained output: `/tmp/polythetic-final-runtime-no-discovery.log`.
- Browser interaction coverage and the six-domain interface review are recorded in [BASE_MODEL_INTERFACE_REVIEW.md](BASE_MODEL_INTERFACE_REVIEW.md). The tested browser runtime is deterministic and explicitly marked as a development fixture.
- **Final result:** after the test-harness changes above, `npm run test:runtime` completes all **44 scripts** sequentially with exit code 0. This includes the new installed-bundle Qwen contract test and external-stop regression. Log: `/tmp/polythetic-final-runtime-no-discovery.log`. This supersedes the intermediate 43/44 result, not the real-model validation limitations.

### Download availability check

Anonymous HEAD requests to each pinned repository's `hosted-artifacts.json` returned **HTTP 401**:

| Catalog model | Repository under `logitsml/` | Pinned revision |
| --- | --- | --- |
| Gemma 3 270M Instruct | `polythetic-web-gemma3-270m` | `ab831929b3266cc3f09c4a2b81c0090193455bd6` |
| Gemma 3 1B Instruct | `polythetic-web-gemma3-1b` | `f4a082a06d5947572eb6a2753fa4093fd222904c` |
| Gemma 3 4B Instruct | `polythetic-web-gemma3-4b` | `b54703553af5b3f1fbbed016a895de01a988f5f6` |
| Qwen 3 1.7B | `polythetic-web-qwen3-1.7b` | `cda345a4200f08cca112502d99c700b04026fefe` |
| Qwen 3 4B | `polythetic-web-qwen3-4b` | `87ef453995eacc71c475dd319193760bafcedded` |

The status alone does not distinguish private repositories from unavailable or otherwise access-controlled repositories. Existing locally cached files may still work; public installation was not established. No account settings, credentials or repository visibility were changed.

### Still blocking “all models usable”

1. True Top-K SAE encoding is still unsupported. The passing pack suite tests rejection, not K=16/K=50 encoding, globally selected feature masks, ties, preprocessing or correct gated/steered reads.
2. GPT-2, Pythia/GPT-NeoX and Qwen 3.5 candidate libraries still need actual checkpoint conversion, linking, and WebGPU generation/capture/replay parity tests. Synthetic compiler exports do not establish these outcomes. GPT-2's 1024-position limit also needs production-builder support before packaging it.
3. The pinned public artifact URLs must be made accessible or replaced with validated accessible artifacts, and matched instrument packs must be packaged. Public publishing and fresh release attestations have not happened.

Do not enable these unverified downloads or claim arbitrary base architecture support on the strength of these interface and consumer-contract tests.

## Optional base-model instruments — 2026-09-05

Base-model setup now separates generation readiness from tool availability. Signed catalogs may contain base-model/core closures with no J-lens or SAE, or optional instruments covering only some contexts. Supplied packs still undergo the existing strict compatibility, schema, and integrity checks. Fresh base installs leave tools unselected; already installed tools retain their selection. A selected tool's hardware limits and the combined instrument-memory budget still apply. Existing chat-model setup behavior is unchanged.

Validation: 65 runtime-foundation tests, 45 shell-controller tests, catalog-builder tests, and `npm run check` passed. Tests cover core-only installation, incomplete optional coverage, default selection, and combined-budget rejection after explicit selection. These are setup tests, not new checkpoint inference evidence.

## Real GPT-2 / Pythia checkpoint pass — 2026-09-05

This section supersedes the earlier statements that no base checkpoints had been
converted or run in a browser. These are **local Chromium smoke and numerical
checks**, not Safari results, instrument validation, or release certification.
No model downloads were enabled in the public catalog and nothing was published.

### Conversion and compiler changes

- The production builder now accepts GPT-2 and GPT-NeoX adapters, GPT-2's real
  1024-position limit, explicit base/raw completion policy, source EOS ID zero,
  and canonical browser dimensions alongside GPT-2's native config fields.
- `q0f32` now explicitly requests raw float32 tensor storage. The converter's
  default BF16 storage is lossy even with float32 computation; merely selecting
  `q0f32` did not previously guarantee lossless converted weights. All 76 raw
  Pythia tensors were checked against the source, accounting for the native QKV
  reorder, and matched exactly.
- Pythia exposed a TVM GPU attention bug: valid late-layer scores fall below the
  hardcoded `-50000` initial maximum and masked-score sentinel. Prefill and decode
  now initialize at the minimum finite float32 value and give masked entries
  exactly zero weight. This is a local fix, separate from the read-only-buffer
  upstream backport in the same overlay.
- `verify-tvm-attention.py` compiles and executes the actual prefill macros on
  CPU with synchronization barriers removed. All six negative-score / empty-tile
  cases pass; the original pinned source fails the first causal case, assigning
  weights `[0, .25, .25, .25]` instead of `[1, 0, 0, 0]`. This tests serial
  arithmetic, not GPU synchronization. The compiler image runs this regression.

### Physical-browser evidence

Both tests used Chrome 152 with an Apple `metal-3` WebGPU adapter, local
integrity-checked files, no persistent model cache, and the exact prefix
`I love marmots because`. The harness generated 24 greedy tokens, captured every
prompt position at every block, repeated capture, regenerated after capture,
and unloaded the model. The independent reference uses the same pinned source
files, CPU float32, eager Hugging Face attention, and identical tokenizer IDs.

| Candidate | Source revision | Result |
| --- | --- | --- |
| GPT-2 small, raw float32, 1024 context | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | Corrected attention build: all 24 greedy tokens match; logit RMSE `3.219e-5`; worst layer/position residual relative L2 `1.952e-6`. Repeat capture is exact and post-capture generation matches. |
| Pythia-70M deduped, raw float32, 2048 context | `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c` | Corrected attention build: all 24 greedy tokens match; centered-logit RMSE `5.400e-4`; worst layer/position residual relative L2 `7.273e-5`. Repeat capture is exact and post-capture generation matches. |

Pythia's new library SHA-256 is
`b037136145cdff7d39ce2b43d711c754c2427cf0201fe2fd8d7c879827a98e97`;
the conversion manifest SHA-256 is
`b434d2f07751fab67b8da94efd1c907a9b9bd31ceed9961fe33f21079bdd7afa`.
GPT-2's new library SHA-256 is
`84f2b9e5793facbea2c6262f0db3a714e27c63a43dbf7e0d421ec44a14b33142`;
its conversion manifest SHA-256 is
`e1a1f757872a0908ce30ed1851c6c714393fe776b986d0f4202794eacab7bccb`.
The raw model closures occupy about 284 MB (Pythia) and 655 MB (GPT-2, including
the materialized tied head). Earlier q4f16 candidates did not establish acceptable
checkpoint parity and remain unverified; finite output alone was insufficient.

Local diagnostic records:

- `/tmp/polythetic-pythia-checkpoint-comparison-attention.json`
- `/tmp/polythetic-gpt2-checkpoint-comparison-attention.json`
- `/tmp/polythetic-attention-macro-test.log` and
  `/tmp/polythetic-attention-macro-baseline.log`

The harness and checker are reproducible with
`webui/scripts/base-model-browser-harness.mjs` and
`browser-runtime/compare-base-checkpoint.py`. Their reports intentionally do not
claim release certification. Safari, EOS/context-limit behavior, cancellation,
steering and instrument coordinates, Qwen hybrid inference, and Gemma PT remain
outside these results. Old GPU attestations were not restamped after the
toolchain change.

Final local checks pass: all 44 runtime scripts, `npm run check`, the hosted
build (291-file isolation), 25 production-toolchain tests, overlay integrity,
and the non-release runtime-lock check. Logs use the
`/tmp/polythetic-base-final-*` and `/tmp/polythetic-attention-*` prefixes.

## Gemma PT and Qwen 3.5 checkpoint diagnostics — 2026-09-05

These are local candidate results, not catalog additions or release attestations.
The Gemma source download was verified against revision
`fcf18a2a879aab110ca39f8bffbccd5d49d8eb29` with `hf cache verify`: all ten
requested files passed. Account access is no longer blocking this checkpoint.

### Runtime and builder changes

- Qwen 3.5 base now participates in the production builder with the explicit
  `kv-rnn-v1` state ABI and architecture-specific adapter/source digests.
  EOS and position limits come from the nested text config, not the multimodal
  wrapper. Base completions do not acquire an instruction-model thinking profile.
- Raw completion text now retains a tokenizer's required special-token prefix.
  Gemma's prefix is BOS ID `2`; the actual text remains exactly
  `I love marmots because`. The builder checks that special-token encoding is
  a pure prefix, pins it in the build manifest, and aligns generation and capture.
  The runtime counts prefix tokens against the context limit. Word tokenization
  remains prefix-free, and chat templates are not applied to base completions.
- The verified artifact cache rejects invalid, mismatched or missing required
  BOS metadata. This does not loosen model/tool compatibility or integrity checks.
- The local runtime package is now `.34`, SHA-256
  `79e9e88edd7213370da67d464fa490254f5fe30bb2d7949c6e477b5eb02642fd`.
  Fork manifest SHA-256 is
  `3e17cab1323a86b16df3f4d16b5a84494a0d5cc973d2534b59f9211eccb56852`;
  runtime identity is
  `2efbf2380150a1d59740596ec0e1951c6f713e7ceeb15ab4abc330fd2ee025b1`.
  Earlier packages and GPU evidence were retained, not relabeled.

### Actual Chromium results

Chrome 152 / Apple `metal-3` ran these pinned, integrity-checked local files.
All prompts used the required marmot prefix. The harness unloads after each run.

| Candidate | Observed result | Limitation |
| --- | --- | --- |
| Qwen 3.5 2B Base, `q4f16_1`, 2048 context, revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` | 24-token generation, all 24-layer prompt captures, exact repeat capture, identical 24 token IDs after capture | Original-checkpoint float32 diagnostic differs after the first 12 generated tokens; logit RMSE `0.4154`. A matched-quantization reference is still required. |
| Gemma 3 1B PT, `q4f16_1`, without BOS | Generation and repeat capture pass, but completion repeats “because” | The original checkpoint also repeats without BOS; this was an invalid prompt policy, not useful evidence of ordinary base completion quality. |
| Gemma 3 1B PT, raw `q0f32`, without BOS | Model loading exits with `Program terminated with exit(1)` while fetching parameters | No generation or numerical result. The embedding exceeds the runtime's 1 GiB requested buffer limit; the adapter advertises a larger limit, but the precise exit cause was not captured. Production admission was not loosened. |
| Gemma 3 1B PT, `q4f16_1`, with BOS | Generation is rejected for invalid probabilities; all first-step logits are non-finite | The BOS residual overflows at zero-based block 11, followed by non-finite values across the next block. This is a failure, not a smoke pass. |
| Gemma 3 1B PT, `q4f32_1`, with BOS | Finite 24-token generation, all 26-layer prompt captures, exact repeat capture, identical 24 token IDs after capture | Float32 computation fixes the observed overflow. Source-architecture numerical and long-context correctness are separate requirements. |

For the BOS run, capture records a maximum finite BOS magnitude of `62176` at
block 10, one non-finite coordinate at block 11, and entirely non-finite
residuals from block 12 onward. A CPU reference using the exact dequantized
float16 browser weights **and the original source architecture** reproduces
the same onset. The original float32 checkpoint produces finite completions.
This establishes a float16 range problem, independently of the separate
positional-encoding issue below. The float32-compute q4 candidate passes the
four Chromium smoke/replay checks. Its closure is 602,358,293 bytes (float32
computation, 4-bit packed weights; float32 scale payloads use the converter's
BF16 storage encoding). Library SHA-256:
`2427408ca735cdb751e568a7ea4772ce2bb1166092066ddc4f4dc1a35a94ace1`;
conversion manifest SHA-256:
`30e8862c6836da6969a37cddda334a59cb0fbf70cbc42e338a82cdb47609a65b`.

The exact-dequantized/source-architecture reference is **not a numerical pass**:
first-step logit RMSE is `0.12869`, relative L2 `0.01735`; worst per-position
residual relative L2 is `0.04698`. The first 11 greedy token IDs agree before
the reference selects “love” and the browser selects “have”. This discrepancy
cannot be attributed simply to comparing q4 with original unquantized weights:
both sides of this diagnostic use the same converted q4 tensors. The known
local-RoPE mismatch remains a concrete compiler issue; its contribution has
not yet been isolated by an A/B compiler fix. Record:
`/tmp/polythetic-gemma-pt-q4f32-source-comparison.json`.

Qwen's library SHA-256 is
`18eca88b2bff223bdf0103df3fe45c9b7f3d079bfd5f01faeaefdcf6a676a1f0`;
its conversion manifest SHA-256 is
`996c751dad97539878a67005399ad1674fb0deb7e0a01d33512cb972aee7729c`.
The closure is about 1.083 GB. These earlier smoke reports identify the exact
runtime bundle they used; they do not certify the subsequent `.34` package.

### Independent reference and unresolved Gemma geometry

`compare-base-checkpoint.py --converted DIRECTORY` adds a Gemma-only diagnostic
using exactly the hashed, dequantized browser tensors with the **source config**.
It preserves different local/global RoPE frequencies, source sliding-window
width and layer types. Norms account for the converter's already-folded `+1`,
and linear operations use float32 accumulation before the configured output
cast. It does not inject browser residuals into the reference or change source
geometry to match a suspect compiler. Non-finite outputs cannot yield an
accuracy result. The original-checkpoint reference remains the default.

The pinned MLC Gemma cache applies normal-mode RoPE with global frequency
`1e6` before attention; its inline-mode local `1e4` setting does not correct
that pre-rotation. Source Gemma 1B requires local `1e4`. The TVM layer-sliding
path also contains a hardcoded 1024-token window, whereas the 1B source uses
512, and initial ragged prefill needs a local-window mask. These remain
unresolved; short-prompt output cannot certify long-context Gemma fidelity.

### Regression checks and records

- WebLLM: all 20 suites / 307 tests pass, including seven completion-prefix and
  context-budget regressions. Build and local packaging pass.
- App: `npm run check`, all 44 runtime scripts, and the hosted build pass
  (293-file isolation; existing chunk-size advisory). A native Node shutdown
  assertion in the session-persistence test was removed by disabling unnecessary
  test-server dependency discovery; three repeated processes then exited cleanly.
- Production builder: 29 tests pass; non-release runtime-lock checking passes.
- Source-config reference plus existing q4 helpers: 13 tests pass, covering both
  compute precisions, distinct local/global frequencies, source window policy,
  exact state closure and unsupported-precision rejection.
- Diagnostic logs: `/tmp/polythetic-gemma-f16-source-geometry.log`,
  `/tmp/polythetic-gemma-bos-original-reference.log`,
  `/tmp/polythetic-qwen35-checkpoint-comparison-v1.json`, and
  `/tmp/polythetic-completion-prefix-*.log`.

No public visibility, publishing, model catalog availability, project version,
or old release attestation was changed. The nine-model Chromium/Safari milestone
and full instrument compatibility remain incomplete.

## Updated-runtime lifecycle checks — 2026-09-05

The harness's optional `--lifecycle` mode additionally interrupts a real stream
after three sampled tokens, checks that partial text is retained, and compares
a fresh greedy run to the original. It then submits an independently tokenized
overlong marmot-prefixed prompt, requires `ContextWindowSizeExceededError`, and
checks fresh generation again. These are engine-level tests, not Loom branching
or natural-EOS tests.

| Browser and checkpoint | Result |
| --- | --- |
| Chrome 152, GPT-2 `q0f32`, 1024 context | All six smoke/lifecycle checks pass. Three tokens retained on cancellation; 2054-token input rejected; all 24 output IDs match after recovery. |
| Safari 26.6.2, GPT-2 `q0f32`, 1024 context | All six checks pass on the Apple WebGPU adapter. Logits and all prompt captures are bit-for-bit identical to Chromium, with identical 24-token output. |
| Chrome 152, Pythia-70M deduped `q0f32`, 2048 context | All six checks pass. Three tokens retained on cancellation; 6150-token input rejected; all 24 output IDs match after recovery. |
| Safari, Pythia | Not run in this pass: Safari became user-active while the test tab was being selected, so UI automation was paused. |

The new Chromium GPT-2 and Pythia reports also have **zero difference** in
logits and every captured value from their respective earlier `.33` reports
that were checked against the original float32 checkpoints. These actual `.34`
runs provide updated evidence; old attestation files were not rewritten.

Records: GPT-2 reports `run-1.json` (Chromium) and `run-2.json` (Safari) under
`/var/folders/q6/8tf71hmx2qq_v74d8rvwg1ww0000gn/T/polythetic-base-browser-T9QfkO/`;
Pythia report `run-1.json` under
`/var/folders/q6/8tf71hmx2qq_v74d8rvwg1ww0000gn/T/polythetic-base-browser-NnkXbJ/`.
The combined production-builder/source-reference/q4-helper suite passes all
42 tests. Overlay integrity, Python compilation and harness JavaScript syntax
checks pass. These results still do not cover the full nine-model roster,
long-context attention fidelity, model instruments or release certification.
The current runtime identity is
`e2ad88fea0f132ebf6dc8ce6887f325861a7c3ff6562de5ad33bbc9f0ef65002`.

### Access and remaining work

The pinned Qwen 3.5 2B Base source
`b1485b2fa6dfa1287294f269f5fb618e03d52d7c` is downloaded locally, including the
4.2 GiB weight shard; it has not been converted or run. The production compiler
still needs hybrid-adapter registration, nested text-config/EOS handling, and
architecture-specific source/adapter provenance. Its cached config is a
multimodal wrapper; the requested text backbone must not be mistaken for a
Qwen3 architecture or a different checkpoint.

An exact-source download of `google/gemma-3-1b-pt` revision
`fcf18a2a879aab110ca39f8bffbccd5d49d8eb29` failed with **Access denied; repository
requires approval**. User-approved access or a locally supplied checkpoint is
required. No terms were accepted, account permissions changed, or replacement
checkpoint substituted.

The overall milestone is incomplete. Remaining scope includes the other real
model checks, Top-K SAE and instrument parity, min-p/repetition-penalty plumbing,
the remaining completion controls and nondestructive Loom split/local exports,
Safari coverage, and fresh candidate release evidence. Comparison experiments
remain explicitly excluded; publishing remains unauthorized.

## Signed base installs and long-context validation — 2026-09-06

This release work follows the request to enable in-app installation. The
following model repositories are public, with model files and their required
core geometry pack bound to the same immutable commit:

| Model | Computation | Context | Published commit |
| --- | --- | --- | --- |
| GPT-2 Base | `q0f32` | 1024 | `787a18eed89e1c5e342382c6b88fb5f4dcd5f978` |
| Pythia 70M Deduped Base | `q0f32` | 2048 | `b312ef57bd3852969fd43c8254ceb93b79317d27` |
| Gemma 3 1B PT | `q4f32_1` | 2048 | `b771368a80809580545d340358de1ceab28632f4` |
| Qwen 3.5 2B Base | `q4f32_1` | 2048 | `4a839d1c2a91f397a91bc25dbaa07825930adea7` |

Repositories are `logitsml/drowse-web-<model-id>` on Hugging Face. Catalog
sequence 3 at `logitsml/drowse-web-catalog` includes these four models and
required core packs only. Every published model/core file passed full-byte
size/SHA-256 checks and anonymous download, range, and CORS preflight. The
catalog uses the new Drowse current/next signing-key pair; only public keys are
in the repository. Earlier chat-model runtime pins remain unchanged and are
not silently relabeled as base models.

The catalog's immutable publication is
`9ed3c63c8840fd93a53fcf8012c9c2d46315a5aa`; the updated runtime identity is
`8cf065c1ace9e8e2bc35bcbde1f5f0b8bf039284dc746a1f22ed47e41630415c`.

The model picker now assesses a base model against its highest supported
context no larger than the requested context. This makes GPT-2's native 1024
profile installable while preserving strict actual-load admission, memory
checks, and the policy that base models are never automatically recommended
as chat assistants. Missing J-lens or SAE packs do not block base generation
or core steering.

### Gemma attention repair and independent accuracy

Gemma's paged-KV path now uses the source model's actual rotary frequency and
absolute query positions, including chunk offsets. Its 512-token local-window
mask and global-layer mask enforce absolute causality even when the underlying
append-before-attention prefill call supplies `causal=0`. The missing causal
mask had let early query positions attend to later tokens; changing rotary
parameters alone did not repair it. Eleven serialized kernel checks exercise
the masked path, including local/global and chunk-boundary cases.

The corrected final `q4f32_1` library passes actual Chrome/Metal generation,
capture, repeat, cancellation, context rejection/recovery, branch/reset, and
forced-EOS/recovery checks. The source-architecture reference consumes the
exact converted quantized weights, without modified source attention settings
or residual corrections. The float16-compute Gemma build is not released.

| Final model | Input tokens | Relative logit L2 error | Maximum per-layer capture relative L2 | Greedy output |
| --- | ---: | ---: | ---: | --- |
| GPT-2 | 996 | `1.70e-7` | `9.83e-7` | 24/24 source-checkpoint IDs match |
| Pythia | 1926 | `1.29e-6` | `3.00e-4` | 24/24 source-checkpoint IDs match |
| Gemma PT, short | 6 | `9.66e-7` | `1.29e-6` | 24/24 matched-quantization IDs match |
| Gemma PT, long | 1806 | `7.07e-6` | `2.63e-6` | 24/24 matched-quantization IDs match |
| Qwen 3.5 Base, short | 6 | `1.75e-6` | `3.10e-6` | 24/24 matched-quantization IDs match |
| Qwen 3.5 Base, long | 1626 | `2.53e-5` | `1.16e-5` | 24/24 matched-quantization IDs match |

Long captures sample positions on both sides of 128, 512, and 1024 when
present, including the last input token. Gemma's worst sampled-position
relative error is `1.74e-5`. These checks cover the advertised browser profiles,
not the source models' larger maximum contexts or every possible prompt.

### Core-only runtime, cache, and app checks

All four final model/core closures pass the production runtime
worker harness: SHA-verified OPFS staging, 24-token baseline generation,
physically active core steering, a changed finite conditional logprob for the
same forced token, and an unchanged subsequent baseline. A new runtime and
core compiler then reopen freshly read OPFS file handles with all model HTTP
requests blocked. Baseline and steered token IDs reproduce, with zero network
attempts and zero device losses. Optional instrument count is zero.

This network-blocked test verifies model reload, not offline page navigation.
Separately, the real app's signed-catalog installer was exercised for all four
models: download/open, ordinary and core-steered completion, full page
reload, saved-conversation reopening, and further generation all pass. The
offline-ready notice initially covered the completion button in the short
test viewport; dismissing it restored normal clicks without an engine change.
Gemma and Qwen at steering strength 0.5 can repeat heavily; Gemma's 0.1 smoke
completion was readable. These are functional steering checks, not a
semantic-quality study.

### Qwen matched-quantization and replay investigation

The final FP16-compute Qwen build is withheld. Some repeated captures and
branch replays differ by up to `0.171875`, despite matching greedy tokens and
other successful runs. Passing runs alone are not treated as repeatability
evidence.

A freshly converted/compiled `q4f32_1` build retains the exact pinned
`Qwen/Qwen3.5-2B-Base` text backbone and hybrid KV/recurrent-state ABI. At 1626
input tokens its independent source-config reference with the exact quantized
weights matches all 24 output IDs: relative logit error `2.53e-5`, maximum
per-layer capture relative error `1.16e-5`, and worst sampled-position relative
error `2.73e-5`.

An eight-branch diagnostic run also preserves greedy generation after capture,
cancellation, rejected context, explicit reset, and forced EOS. Its small
within-browser residual differences exceed the original absolute-only `1e-5`
check: maximum absolute difference `5.15e-5`, maximum per-layer relative L2
`3.00e-6`. That diagnostic remains labeled `diagnostic-only`, not retroactively
passed. A separate explicit FP32 replay policy requires **both** maximum
absolute error at most `1e-4` and relative L2 at most `1e-5` in **every layer**;
the strict default is unchanged and FP16 builds cannot opt into the FP32
policy. This is numerical repeatability, not a bit-exact claim.

A fresh long-context run passes all eight smoke/lifecycle checks and eight
branch replays under that declared FP32 policy. Repeated capture differs by
at most `2.48e-5` absolute / `2.11e-6` per-layer relative L2. Across the eight
branches, maxima are `8.78e-5` absolute and `2.68e-6` per-layer relative L2;
every comparison satisfies both limits. All 24 greedy output IDs reproduce
after capture, cancellation, rejected context, reset, and forced EOS. A fresh
short-context run passes the same lifecycle checks plus three branch replays
with bit-identical captures. This does not establish bit-exact long replay or
all-prompt/platform fidelity.

The final Qwen core-only production worker test also passes: 24-token baseline
and core-steered generation, a changed finite logprob for the same forced
token, unchanged subsequent baseline, and fresh runtime/core-compiler reload
from verified OPFS handles with all model HTTP requests blocked. Reopened
baseline and steered sequences reproduce, with zero network attempts, zero
device losses, and zero optional instruments installed.

The final combined runtime suite passes all 44 runners. The production
builder/source-reference/quantized-weight suite passes 47 tests. Fork-overlay
integrity, Svelte checking, hosted build and isolation checks pass; the existing
large-chunk build warning remains. Standalone replay-policy tests cover both
bounds, individual-layer isolation, shape mismatches, and non-finite values.

Compact artifact identities, public preflight receipts, reference metrics,
browser lifecycle results, rejected diagnostics, core-only offline reloads,
and actual app-check outcomes are recorded in
[`base-model-release-evidence.json`](base-model-release-evidence.json).
Qwen's final actual app check used catalog sequence 3: the 1.1 GB download
verified and opened without optional packs, generated 24 tokens normally and
with the included core direction, survived a full-page reload, reopened the
saved two-branch completion, and generated 24 further tokens. No warning or
error was reported in the browser console.

The model packages and signed catalog were published to Hugging Face. The app
was rebuilt and verified locally; this work does not deploy a production
website, publish git commits, or certify the older nine-model release roster.

### Existing-origin upgrade recheck — blocked

At `http://127.0.0.1:4173/app?choose=1`, the in-app browser initially used an
older cached app. The normal **Update and reload** flow successfully loaded
the current Drowse build, but the model catalog still failed admission. A
temporary console diagnostic identified the exact error: **The catalog
sequence is older than the last accepted release**. The diagnostic was then
removed. No catalog state, model files, or saved conversations were cleared,
and signature and rollback enforcement were not weakened.

The public catalog still returns sequence 3 at immutable commit
`9ed3c63c8840fd93a53fcf8012c9c2d46315a5aa`. It is valid for fresh clients but
cannot replace a higher sequence already accepted by this existing client.
The corrective release must use a newer signed sequence and then pass the
actual installer, load, generation, and reopen checks at the existing origin.
That publication has not been performed during this recheck.

The unavailable-download message previously displayed a model's descriptive
blurb in place of the global catalog failure. It now prioritizes the actual
download failure, with six regression assertions covering global failures,
model-specific setup issues, and no selected model. The full 44-runner runtime
suite also passed during this recheck, before this message-only change; it
does not supersede the observed upgrade failure.

## Existing-origin upgrade and lifecycle repair — 2026-09-06

The original `http://127.0.0.1:4173` app accepted the normal **Update and
reload** flow and signed catalog sequence **7** without clearing catalog
history, model files, or saved conversations. The published catalog commit is
`da488a3de729bc649ca70465b1da56409a0b98eb` in
`logitsml/drowse-web-catalog`. Its four model/core entries are unchanged from
sequence 3. Public preflight verified **126 immutable files / 2,731,824,423
bytes**. Signature verification and rollback protection remain enabled.

### Reproduced failures and repairs

- The catalog publisher now authenticates the previous published catalog,
  rejects a non-advancing sequence, and rechecks the repository revision before
  uploading. The app's minimum accepted sequence is 7; regression coverage
  includes an existing legacy-key client with a sequence-6 high-water mark.
- Normal unload/takeover no longer erases compatibility approval. Each model
  load still requests a fresh GPU adapter and checks it against the approved
  fingerprint. Stateful regressions reproduce the old second-load failure and
  verify both unload/takeover reloads plus rejection of a changed adapter.
- Forced replay at temperature zero can legitimately produce a sampler
  log-probability of negative infinity. Raw analysis callbacks retain that
  value. JSON-facing token statistics use null, non-finite display alternatives
  are omitted, and the aggregate is null if a response log-probability is
  non-finite. This avoids corrupting the saved session or inventing a finite
  probability. The exact previously failing GPT-2 continuation now generates
  and autosaves successfully, including after unload/reopen.
- Tree revision ordering is now scoped to both model and session identity.
  Previously, two different models sharing session `default` could cause the
  lower-revision model's tree to be ignored. Switching now adopts the correct
  tree and clears the previous model's live speed/status counters; stale
  revisions within the same model/session remain rejected.

### Actual app checks

All checks used real downloaded models, not layout fixtures. The initial raw
prompt was `I love marmots because`, temperature 0, with 24 newly generated
tokens per completion. The steering condition used the included
`default/welcoming.detached%welcoming` core direction at strength 0.1.

| Model | Signed install/open | Ordinary + core-steered generation | Saved reopen + continuation |
| --- | --- | --- | --- |
| GPT-2 Base | Passed | Passed | Passed |
| Pythia 70M Deduped Base | Passed | Passed | Passed |
| Gemma 3 1B PT | Passed | Passed | Passed |
| Qwen 3.5 2B Base | Passed | Passed | Passed |

Qwen's 1.1 GB download was also paused at approximately 21%, resumed, verified,
and opened. Direct in-app switches among the installed models passed without
requiring a page refresh. Original saved user work remained present. No new
browser warnings/errors appeared during the final four-model verification.

The 44-runner runtime suite, publisher regressions, Svelte/UI checks, and both
native and hosted builds pass. Existing bundle-size warnings remain.
[`base-model-upgrade-evidence.json`](base-model-upgrade-evidence.json) records
this app-level recheck separately from the earlier numerical/model-artifact
attestations. These are functional smoke checks on this device, not a
semantic-quality evaluation or an all-browser guarantee. Optional J-lens and
SAE packs are not installed or required for these core-only checks. No git
push, production website deployment, or version bump was performed.
