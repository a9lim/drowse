# Hosted Polythetic user-journey audit

Audit date: 2026-08-30

This document records the end-to-end state of the hosted application. It separates four kinds of evidence so a deterministic fixture or unit test is never presented as proof of a production model release:

- **Hands-on**: exercised through the rendered application in the Codex in-app browser.
- **Browser E2E**: exercised in Chromium through Playwright, including IndexedDB, OPFS, workers, service workers, offline mode, accessibility, mobile, and RTL layouts.
- **Physical WebGPU**: exercised with the current Gemma 3 270M q4f32 runtime, model files, core manifold pack, provider J-lens pack, and Gemma Scope 2 SAE pack at the 2K context profile on the local Apple/Metal WebGPU adapter.
- **Contract/math**: deterministic TypeScript, Python, or Rust/WASM coverage for lower-level behavior.

The current result is: **the local development build and current Gemma physical
path pass, but the public hosted release remains intentionally locked**. The
model artifacts and signed distribution have not been published, so the
non-fixture `/app` correctly refuses model downloads. Cross-platform and
authoring evidence are optional QA, not release gates.

## User journey catalogue

| User journey | Result | Evidence and notes |
|---|---|---|
| Open `/` | Pass | Hands-on: landing page rendered the project purpose, local-processing promise, supported-device explanation, model tiers, source, license, and `Open Polythetic` action with a coherent heading hierarchy and skip link. Browser E2E also covers routing, 404 handling, 320 px reflow, and automated accessibility. |
| Open `/app` for the first time | Pass | Hands-on: compatibility check completed on a real WebGPU-capable browser and retained the adapter for the runtime path. The UI explained secure context, storage, browser, and hardware facts without calling advisory memory values VRAM. |
| Visit `/app` while production distribution is unverified | Pass, fail-closed | Hands-on: compatible hardware was reported as compatible, while model downloads remained disabled with an explicit distribution-verification explanation. This is the intended current public behavior. |
| Receive a model recommendation | Pass in fixture and policy tests | The selection and unsafe/uncertain distinction are covered by onboarding and policy tests. Unknown devices remain conservative; confirmed OOM history requires explicit retry. Production recommendations remain disabled until the signed model distribution is provisioned. |
| Choose optional tools on first install | Pass | Hands-on: J-lens and SAE were selected independently, their sizes were included, and the required core response controls were always included. Browser E2E covers omit, add, remove, reinstall, family replacement, runtime compatibility, and resident-memory rejection. |
| Start a download | Pass in fixture and physical signed-catalog run | Fixture onboarding showed exact bytes and clear source/license copy. The physical production-app harness admitted an ephemeral signed catalog, verified 27 signed files/25 unique hashes, and installed 330,178,828 bytes across model, core, J-lens, and SAE assets. |
| See progress, ETA, cancellation, and resume | Pass | Runtime and browser tests cover sequential files, one-second EWMA, stall handling, cancellation checkpoints, offline pause/resume, exact `Content-Range` validation, ignored ranges, changed ETags, hash/size corruption, quota checks, and crash recovery. |
| Open the installed model | Pass | Hands-on fixture opened the shared workbench. Physical WebGPU opened Gemma 3 270M q4f32 from verified OPFS content and later reopened it after explicit unload and worker/page recreation. |
| Send a chat message | Pass | Hands-on fixture streamed and completed two prompts. The physical production-app run completed online and offline responses using the real q4f32 backend. The 270M checkpoint is the fastest tier, not a quality tier. |
| Stop generation | Pass | Browser E2E covers the visible stop flow and reload-during-generation recovery. Physical WebGPU interrupted a 64-token request after one token and reported `finishReason: abort`. |
| Repeat generation and recover from reload | Pass | Browser E2E covers repeated generate/stop/reload cycles and removal of partial assistant output after mid-generation reload. The authoritative user turn remains stable. |
| Reroll a reply | Pass | Hands-on: reroll created a second assistant sibling under the same user turn with its changed steering recipe visible. Browser E2E independently asserts the sibling topology. |
| Explore conversation branches | Pass | Hands-on: the branch map showed depth, alternatives, steering deltas, and log-probability summaries. Keyboard-scoped tree navigation, filtering, starring, notes, branch mutation, and restoration are covered by E2E/runtime tests. |
| Compare two branches | Pass | Hands-on: side-by-side comparison rendered recipe delta, per-token log-probability changes, approximate KL, rank changes, and J-lens/SAE reading deltas. |
| Save, open, export, and import a conversation | Pass | Save/open drawers are included in responsive/accessibility coverage. Browser E2E round-trips transcript YAML through the worker and verifies authoritative loom persistence. |
| Change sampling | Pass | Hands-on: creativity, maximum length, raw/chat behavior, roles, and help text were visible; unsupported reasoning was disabled. Contract tests verify capability-filtered thinking modes and sampling persistence. |
| Change speakers and roles | Pass | Hands-on: role seats and cast editing opened correctly. Physical curved generation selected the named role `distant_voice`; explicit and inferred role-baseline conflicts fail before generation. |
| Add a concept or flat response direction | Pass in runtime/physical path | The fixture has no installed concept inventory, so this control is deliberately empty in the hands-on fixture. The physical core pack loaded `default/welcoming.detached`; the physical gate applied flat gated geometry, and shared Python/browser lowering fixtures cover push, projection, erasure, ablation, DLS, and affine composition. |
| Add a curved mood or scale | Pass in runtime/physical path | Physical WebGPU captured ten rows, spooled them through OPFS, ran the Rust/WASM fitting lifecycle, installed `local/browser_e2e_curve`, and generated with an active curved and geometry slot. Multidimensional, periodic, sphere, sigma, carried-foot, overlap, and orthogonalization cases are covered by structured hook/runtime tests. |
| Add J-lens steering or readings | Pass with the provider pack | Hands-on J-lens controls and readings remained usable. Physical Gemma attached `jlens/fake`, loaded 17 fitted layer matrices and the curated token dictionary, emitted exact probability readouts, and restored them after unload/offline reopen. |
| Add SAE steering or readings | Pass with Gemma Scope 2 | Hands-on SAE controls and readings remained usable. Physical Gemma loaded the layer-12 16,384-feature JumpReLU dictionary, emitted exact full-dictionary feature measurements, and restored it after unload/offline reopen. Browser SAE training remains absent. |
| Combine controls and use gates | Pass | Hands-on recipe combined J-lens and SAE terms. Contract tests cover manifold, SAE, J-lens, ablation, phases, prior-step dynamic gates, full-vocabulary J-lens probability gates, multidimensional/periodic curves, overlapping-curve rejection, and fixed-width GPU lanes. Physical WebGPU activated geometry, lens, and SAE gate families in one run. |
| Inspect a generated token | Pass | Hands-on: selecting `Polythetic` opened J-lens aggregate/per-layer data, SAE feature strength and metadata, explicit empty geometry state, and an actionable unavailable-alternatives state. Keyboard entry and focus restoration are covered by E2E. |
| Replay or fork a token | Pass | Physical forced replay scored alternatives, replayed two tokens, selected an alternate first token, and created a one-token fork. Browser E2E verifies recoverable replay failure when a fixture intentionally lacks the capability. |
| Inspect geometry and correlations | Pass | Hands-on empty states explained missing inputs instead of failing. Runtime tests cover Mahalanobis geometry, correlations, pairwise analysis, cross-layer comparison, and exclusion of J-lens/SAE readouts from residual-direction correlations. |
| Author a manifold | Pass in automated browser/runtime coverage | Browser E2E validates authored, discover, and template-backed forms; delayed validation focuses each blocking field. Runtime tests cover generate, extract, fit, merge, exact artifact publication, and rollback. A real physical curved fit passed. |
| Score prompt templates | Pass in automated browser/runtime coverage | The fitting lifecycle supports exact template scoring and cancellation. Browser artifact tests persist the exact template closure and reject conflicting identities or malformed storage. |
| Import, export, search, or delete `.polythetic` controls | Pass | Hands-on pack manager exposed import and Hugging Face discovery with publisher-integrity wording. Browser tests stream immutable HF packs through OPFS, bind provenance, round-trip exports, and roll back malformed installs. Python and browser ZIP validation cover traversal, collisions, links, special files, compression bombs, tensor schema, checksums, and limits. |
| View model health | Pass | Hands-on health drawer showed model/runtime, generation, loom revision/depth, artifacts, probe rows, matrix state, and warnings without console errors. |
| Close a drawer | Fixed and passing | The backdrop and visible close button previously shared the same accessible name. The backdrop is now pointer-only and `aria-hidden`; the visible button and Escape remain accessible. A regression test asserts one `Close drawer` action plus backdrop dismissal. |
| Manage the model and storage | Pass | Hands-on: close model, change model, model deletion, pack deletion, quota/persistence status, technical details, and clear-all were distinct. Browser E2E performs deletion in disposable storage and verifies that only the selected model, pack, session, or Polythetic-owned data is removed. |
| Use multiple tabs | Pass | Browser E2E verifies idle cooperative takeover, repeated transfer, and explicit approval before interrupting a busy owner. A requester times out rather than pretending Web Locks can force eviction. |
| Recover from device loss or OOM | Pass in orchestration tests | Worker tests cover one clean unknown-loss reload, changed-artifact rejection, verified-object eviction, loss during load/restore/generation, cleanup escalation, stale callbacks, and confirmed-OOM history. These destructive cases are not induced on the physical developer GPU. |
| Reload and work offline | Pass | Browser E2E covers PWA shell offline navigation and fixture generation without redownload. Physical production-app and lower-level gates both reopened the real installed model offline with zero artifact/network requests during the quiet window. |
| Receive a PWA update | Pass | E2E verifies update detection, explicit consent, clean runtime unload before reload, and preservation of local data. |
| Use mobile, keyboard, screen reader semantics, or RTL | Pass in automation | All hosted command and context drawers were tested at 320 px; touch targets, composer size, bidi fields, radio arrow behavior, focus return, landmark hierarchy, and RTL mirroring passed. Automated WCAG checks passed on landing, onboarding, workbench, every command drawer, and context-launched drawers. |
| Keep prompts and activations local | Pass in tested flows | Browser E2E asserts online fixture generation sends neither prompts nor activations. The physical production-app audit found no prompt uploads or unexpected external requests; the installed offline run made zero artifact requests. |
| Use the Python server/dashboard | Pass | The isolated default Vite build remained separate from the hosted PWA. Package isolation verified the 24-file dashboard closure, and the full Python suite preserved HTTP/WebSocket, CLI, server, artifact, SAE, J-lens, manifold, and bundled-dashboard behavior. |

## Defects found and fixed

### Duplicate accessible drawer-close action

Every open drawer exposed two elements named `Close drawer`: the visible close button and the full-screen backdrop. Assistive technology and semantic browser automation could choose the backdrop, whose center was covered by the drawer, leaving the drawer open.

The backdrop remains clickable for pointer dismissal but is now `aria-hidden` and no longer declares button semantics or a duplicate keyboard handler. The visible close button and Escape are the two clear keyboard-accessible dismissal paths. The regression exercises both the named close button and a click on the uncovered backdrop.

### Gemma fp16 overflow

Gemma 3 270M produced non-finite late-layer residuals in fp16 in both PyTorch and the browser runtime. The hosted variant now uses q4f32 weights and fp32 compute, does not require `shader-f16`, and has a separate runtime fingerprint. The real q4f32 model, core pack, J-lens pack, and SAE pack pass the production-app and 12-stage physical gates.

### Missing exact JumpReLU VM registration

The compiled model exported the exact JumpReLU readout kernel, but WebLLM did not load it into the VM function registry. The registry now includes `polythetic_sae_jump_relu_readout_accumulate`, with overlay and physical-generation regression coverage.

### Gemma named-role rendering

Named assistant roles initially supported only ChatML `<|im_start|>` headers. Gemma uses `<start_of_turn>model`; the role renderer now preserves both prefix families, de-slugs the label, and leaves ordinary unnamed chat rendering unchanged. The physical curved-fit and named-role generation stages pass.

### Production-q4 parity ordering and probe metric

The first parity fixture compared outputs from separate ordinary-capture and
rank-one-capture VM functions. Small compiled-graph drift could therefore appear
before the enabled layer and corrupt the steering cosine. Schema-v3 captures now
run disabled and enabled programs through the same rank-one VM function, require
zero pre-enabled delta, and compare enabled-minus-disabled steering.

Probe parity is normalized by the activation scale rather than by the norm of a
possibly near-zero scalar trace. Browser probe readback is separately checked
against the dot product of its own captured residual. Fresh SmolLM2, Gemma, and
Qwen3 candidates all pass the unchanged 1% capture/probe and 0.99 cosine gates.

### Exact q4f32 comparator dispatch

The local PyTorch/Metal comparator accepted Gemma q4f32 metadata but routed every
bias-free linear through its fp16-only kernel. It now dispatches float16 and
float32 tensors to separate exact Metal FMA kernels. The fresh Gemma comparison
passes with maximum capture error `8.29e-6`, maximum activation-normalized probe
error `2.13e-7`, steering cosine `1.0000005`, and exact greedy tokens.

### Minimal-tokenizer capture compatibility

Trailing ChatML whitespace trimming initially assumed every tokenizer exposed
`decode()`. The capture path now retains the older special-token-only behavior
for minimal tokenizer implementations while real tokenizers still trim the
terminal newline. The affected 151-test capture/manifold/neutral-cache cluster
and the complete Python suites pass.

No other application error, warning, failed request, uncaught page error, or WebGPU error was observed in the hands-on or physical-browser audits.

## Verification results

| Gate | Result |
|---|---|
| Hands-on landing, first run, fixture install, workbench, generation, steering, probes, reload, drawers, reroll, branch comparison, transcript, storage, and browser-console audit | Pass; browser console had 0 warnings and 0 errors |
| Playwright hosted E2E | 79 passed |
| Svelte/type/theme/runtime-boundary/interface policy | Pass; 0 Svelte errors and 0 warnings; 239 source files checked |
| Hosted runtime, downloader, worker, storage, authoring, release-tool, and dependency audit | Pass; `npm run test:hosted`; 0 high-severity npm vulnerabilities and 0 total reported vulnerabilities |
| Hosted production build isolation | Pass; 40 files; PWA generated with 39 precache entries |
| Python-dashboard production build isolation | Pass; 24 files |
| Wheel and source distribution dirty-cache isolation | Pass; wheel and sdist built; 24 dashboard files verified |
| Rust/WASM | 37 tests passed; Clippy passed with warnings denied |
| Python regression suite | Pass; non-GPU: 3,465 passed and 20 skipped; real MPS: 48 passed and 3 skipped |
| Physical production-q4 parity | Pass locally for SmolLM2-360M, Gemma 3 270M, and Qwen3-1.7B; exact greedy tokens, ordered rank-one steering, capture/probe tolerance, and steering cosine gates all passed |
| Physical production-app Gemma 3 270M gate | Pass; verified install/load/reopen, provider J-lens and Gemma Scope 2 SAE measurements, online/offline generation, 0 artifact requests after install, empty network/error audits |
| Physical 12-stage Gemma WebGPU gate | Pass; all required checks passed in order, including replay, capture, flat gated instruments, curved fitting/generation with a named role, stop, reload, and offline reuse |
| Local model/context WebGPU matrix | Current Gemma 2K macOS/Chrome path passed; broader device coverage remains optional support QA |
| Instrument pack validation | Pass; the precomputed provider J-lens and Gemma Scope 2 JumpReLU SAE load and execute on the GPU without browser-side fitting or training |
| Public release guard | Correctly fails closed on unpublished model artifacts and the unprovisioned signed distribution; evidence registries are informational |

Physical reports from this audit:

- `/private/tmp/polythetic-gemma-production-app-candidate-v4.json`
- `/private/tmp/polythetic-gemma-full-runtime-candidate-v4.json`
- `/private/tmp/polythetic-gemma-candidate-capture-dense-v1/`
- `/private/tmp/polythetic-smollm2-capture-dense-probe-v1/`
- `/private/tmp/polythetic-qwen3-1.7b-candidate-20260830/capture-v3/`

The JSON reports explicitly set `releaseEvidence: false`, and the captures use
candidate runtime-lock overrides. They remain useful local engineering checks.
Their throughput fields come from the harness's short ordinary-generation
prompt; they are diagnostics, not a long-context benchmark claim.

## Remaining public-release blockers

The application itself is locally functional, but deployment remains
download-locked until these practical requirements are resolved:

1. Choose the model subset to ship and publish immutable converted revisions
   for those models. A Gemma-only 2K launch is valid.
2. Publish each shipped model's required core pack and any compatible
   precomputed J-lens or SAE packs that already exist.
   Publish the SAE pack only when SAE features are offered for that model.
3. Fill the artifact hashes/revisions and mark `runtime-lock.json` verified.
4. Publish and sign the production catalog, fill the observed artifact redirect
   origins, and mark `distribution-lock.json` verified.
5. Validate Hugging Face CORS/range/redirect behavior and the production
   Cloudflare CSP/COOP/COEP headers at the deployed origin.

Fork and compiler attestations, authoring evidence, evidence producers, Git
ancestry checks, all-three-model closure, mandatory 4K support, and the 48-cell
benchmark matrix are no longer public-release blockers.

Until those are complete, a compatible visitor sees a truthful compatibility result and an unavailable download action. That is the correct behavior, not a user-facing runtime failure.

## Deliberate hosted limits

These are not bugs and should remain hidden or explicitly unavailable in hosted mode:

- SAE training.
- J-lens fitting.
- OpenAI/Ollama/server HTTP endpoints.
- Python server-session administration.
- CPU-only or cloud inference fallback.
