# lens/

The Jacobian lens family (Gurnee et al., "Verbalizable Representations Form a
Global Workspace in Language Models", Transformer Circuits 2026). It is a
**per-model** artifact. A lens holds one fp32 matrix per source layer, `J_l =
E[∂h_final/∂h_l]`, averaged over positions and a text corpus. Three surfaces
read it: the vocabulary readout `softmax(W_U · norm(J_l h))`, the
`jlens/<word>` steering atom `W_U[v] @ J_l`, and the J-space decomposition of a
steerable direction. Inside a session the face is `session.lens`
(`LensInstrument`). The package sits beside `sae`, above `readout` and below
`steering.composer`, and imports only `errors`, `base`, `model` and `readout`
(`tests/invariants/test_layering.py`, which counts `TYPE_CHECKING` imports).

| Module | Owns |
|---|---|
| `key.py` | `LensKey` (kind × site), `FITTABLE_KINDS`, local-name grammar. Torch-free. |
| `jlens.py` | `JacobianLens`, readout math, token-atom resolution, error types, fit defaults. Imports only torch and errors (decode-step code imports it). |
| `decompose.py` | `LensDictionary`, `TokensOut` (`W_U J_l`), `sparse_decompose` |
| `fit.py` | `fit_jacobian_lens`, the only backward passes in drowse. No runtime path imports it. |
| `relp.py` | `relp_backward_rules`, `RELP_RULES` (the R-lens backward graph) |
| `corpus.py` | Hub corpora (pin and stream in a worker), `read_corpus_file`. `sae train` uses it too. |
| `lifecycle.py` | `plan_lens_fit` (pure), `LensFitter` (executor), `prove_lens_fit_noop` (model-free) |
| `store.py` | Format-v7 local artifacts: save/load/checkpoint/promote/remove, `LensProvenance`, `lens_estimator_policy`, `lens_identity` |
| `sources.py` | `LENS_SOURCES` per-side selection, provider bindings and fetch, default source order |
| `runtime.py` | `LensRuntime` (resident lens, identity, bounded caches, readout primitives), `matches_loaded_model` |
| `instrument.py` | `LensInstrument` (residency, fit, reads, probes, gates, live, replay) and `LensRun` (per-generation state) |

## Identity: `LensKey`

A lens's **kind** is the Jacobian's direction crossed with its backward rules:
J (output, standard), R (output, RelP), K/S (input, standard/RelP). Its
**site** is `out` (block output) or `mid` (post-attention residual). Only J/R
at `out` have an estimator. Every store, binding and selection path is already
keyed on `LensKey`, and every readout surface here assumes the output side.

- A local directory name carries its kind (`^[jrks](-tag)?$`: `r`, `j`,
  `r-chat`). Every writer refuses a mismatch, so a fit cannot land in or resume
  from another kind's shards. Fits always write the untagged `local:j` or
  `local:r`; a tagged name appears only through a direct `save_lens(name=…)`.
- `JacobianLens` defaults to `key=LensKey("J")` and casts every matrix to fp32.
  Code that builds an R-lens by hand must pass `key=LensKey("R")`.

## On disk (format v7)

```
models/<safe_model_id>/lens/
  active.json            per-side selection {format_version: 2, model_id, output, input}
  bindings/<name>.json   commit-pinned provider binding (format 2)
  local/<kind>[-<tag>]/
    manifest.json        atomic pointer;  checkpoint.json only during a resumable fit
    out[.partial].layer-<L>.gen-<id>.safetensors    immutable fp32 [d, d] shards
```

The manifest schema is exact. It records `kind`, `sites.out.{layers,
tensor_files, tensor_sha256}`, `n_prompts`, `d_model`, the `LensProvenance`
fields and `estimator_policy`. The provenance fields are the corpus spec, the
token-id sha256 (`token_ids_v1`), raw corpus digest and counts, `seq_len`,
`dim_batch`, layer count, and the loaded-weight and source fingerprints. The
policy is `{direction, rules, skip_first_positions, corpus_preprocess_version,
doc_chars}`, with `rules = RELP_RULES` for RelP kinds. There is no `method`
field.

- **The policy is checked by equality on every load.** Editing `RELP_RULES`,
  `LENS_CORPUS_PREPROCESS_VERSION` or `LENS_DOC_CHARS` makes every stored lens
  of the affected kinds load as "no lens". Change them only for a genuinely new
  estimator.
- **Loads fail soft.** A schema, header, digest or finiteness failure logs
  "re-fit with `drowse lens fit`" and returns `None`. The caller chooses
  between `LensNotFittedError` and a re-fit. Shards load one at a time, so a
  selective load reads only the requested layers.
- **Pre-6.0 homes** (`models/<id>/jlens/`, v6, `local/default`/`local/relp`)
  are neither read nor auto-migrated. Run `scripts/migrate_lens_v7.py --home
  <dir>` (`--dry-run` first). It renames shards without touching their bytes.
- **Durability.** New generations are fsynced, then the pointer is swapped
  atomically, then the directory is fsynced before old generations are
  collected. Collection is best-effort and deletes nothing when a pointer is
  unreadable. `lens_fit_lock` is cross-process and spans a whole fit or
  removal, on a `.fit` sibling so pointer reads aren't blocked.
- `save_lens` and checkpoint promotion publish **and select** for the lens's
  side. `save_lens_at(lens, dir, provenance)` writes the same layout anywhere
  without selecting it; it is the packaging path.

## Sources and selection (`sources.py`)

`LENS_SOURCES` is a `base.source_registry.SourceFamily` with one active source
per side (`output`, `input`). A source is `local:<name>` or a bare provider
name.

| Source | Key | Repository |
|---|---|---|
| `neuronpedia` | J | `neuronpedia/jacobian-lens`: the paper lens, matched on its config's `hf_model_name`, config sha256 pinned, `_n1000.pt` preferred |
| `workspace-r` | R | `camilablank/workspace-lenses` `<model>/r-lens/lens.pt`; embedded provenance must name this model and estimator `relp` |
| `workspace-j` | J | the same repository, `j-lens` arm, estimator `standard`, same forwards as `workspace-r` |

- `fetch_lens_source` is pure IO. It pins the publisher and base-model
  commits, reads the config at that commit (`trust_remote_code=False`), loads
  the `.pt` with `weights_only=True`, validates dims, layers and finiteness,
  and writes only the binding. The payload stays in the HF cache.
  `activate=True` selects it.
- `load_external_lens` is offline. It rechecks the Neuronpedia config digest,
  `n_prompts` and the layer roster. **It does not re-hash the checkpoint**
  against the binding's sha256.
- Default order (`lens_source_preference_key`): `workspace-r`, `neuronpedia`,
  `workspace-j`, then `local:r`, `local:j`, then tagged fits. Listings and
  serve's auto-selection (`cli/verbs/serve.py::enable_live_lens`) use it.
- A fit, a promotion or a fetch replaces its side's selection. Removal
  unselects only an exact match. `select_source` evicts the resident lens.
- `matches_loaded_model` decides compatibility. A local fit must match the
  loaded-weight fingerprint exactly. An external lens must match the model id,
  the live `config._commit_hash`, `d_model` and the layer range, so a model
  loaded without a Hub commit never matches one.

## Estimator (`fit.py`)

- **Its own autograd forward.** Inference tensors never re-enter autograd, so
  the fit runs under `enable_grad`. Blocks below the lowest source carry no
  graph. `_MeanSourceProbe` is an identity whose derivative collapses each
  source gradient to its position mean. Gradients come from
  `autograd.grad(final, sources)`, never `backward()` + `retain_grad()`, whose
  `.grad` accumulation across the multi-backward loop corrupts rows. A
  terminal hook raises `_FitForwardComplete` (a `BaseException`) to skip the
  final norm and the vocabulary head.
- **Weighting.** Cotangents go in at every valid target position. Sources are
  averaged within a prompt and summed across prompts, so prompts are weighted
  exactly equally even at ragged lengths. The first `SKIP_FIRST_POSITIONS =
  16` positions (attention sinks) and the last position are excluded. Prompts
  of 17 tokens or fewer are dropped before fitting, so `n_prompts` counts
  consumed documents.
- **Widths.** `prompt_batch` defaults to 4 (2 on MPS, where 2 measured 1.72×
  faster than 1 on M5 Max with gemma-3-4b). `dim_batch` defaults to 8 (93.6
  s/prompt there, 96.9 at 32); its FLOPs are width-invariant, so it trades
  memory for per-pass overhead. `vjp_mode="auto"` falls back from batched
  VJPs to the replicated reference path, with `prompt_batch=1`, when the
  backend lacks vmap coverage.
- **Memory.** Rows stage through bounded stripes (at most 256 rows and 128 MiB
  per slot) into a CPU fp32 accumulator; CUDA double-buffers them to overlap
  device-to-host copies. On OOM the fit first drops the second CUDA slot,
  then splits the prompt group (keeping committed rows and never retrying a
  width that failed), then halves `dim_batch`.
- **MPS.** Metal reports queue exhaustion asynchronously: nothing raises and
  the work comes back as zeros. Hence `torch.mps.synchronize()` every
  `_MPS_SYNC_EVERY_PASSES = 4` passes, a zero-row guard before each commit,
  and `empty_cache` only at checkpoint cadence.
- Cancellation is checked between prompt groups (completed progress is
  checkpointed) and around every VJP block. Teardown drops hooks, graph and
  stripes, then flushes the allocator, so a server's next decode doesn't
  inherit the working set.
- `kind="R"` applies the RelP rules itself; don't wrap it again.
  `initial_lens` hands its matrices to the estimator, which scales them to
  sums in place, so don't reuse the passed lens.
- The cost is about `2·d_model` forward-equivalents of backward per prompt.
  Restricting source layers (`workspace`, an index list) is the only lever
  that removes work. A fit defaults to every layer below the final one and
  refuses `sample`.

## RelP rules (`relp.py`)

The R-lens is the same estimator run through an LRP-modified backward graph
(RelP, arXiv:2508.21258). Forwards are bit-identical and only gradients
differ. The R-lens reads early layers more faithfully. The rules:

- **LN-rule.** RMSNorm backpropagates as the diagonal
  `weight·rsqrt(mean(x²)+eps)`.
- **Identity-rule.** The detached `act(x)/x` factor replaces `act'(x)`.
- **Half-rule.** `act(gate)·up` splits β = 0.5 per factor.

Each rewrite routes the module's own output through a custom `Function`, so
there is no forward arithmetic to drift. The `w` vs `1+w` RMSNorm convention
is detected empirically, and every patched module verifies its factor on its
first forward. Scope is dense models: one gated MLP and direct RMSNorm
children per block, with q/k norms left untouched. MoE, fused gate-up
projections and instance-level `forward` overrides raise
`RelpUnsupportedError` (422). Patching is visible to the whole process, so
hold the model exclusively and pass the eager module list.

## Corpora (`corpus.py`)

The default is `HuggingFaceFW/fineweb-edu:sample-10BT`, which needs the
optional `datasets` package (`pip install 'drowse.ai[hf]'`). A source is
written `REPO[:CONFIG][@REV][,text_field=F,files=K,seed=S,weight=W]`; several
sources form a weighted mix. Documents are sliced to `LENS_DOC_CHARS = 4000`.
A corpus is **pinned to a commit and named config before it is read**, and its
`spec` is the recorded provenance. The first N documents of a longer read
equal a shorter read's, which is what makes token-prefix resume sound. Pinning
and streaming run in a spawned worker: cancellation can abandon a blocked Hub
call, pyarrow's I/O pool can't hang interpreter exit, and spawn is safe beside
MPS. An empty file corpus raises `EmptyCorpusError` (400).

## Fit lifecycle (`lifecycle.py`)

`plan_lens_fit` decides from metadata alone. The identity it compares is the
token-id rows sha256, `seq_len`, the weight fingerprint and the policy.

- **`noop`**: the durable artifact covers the corpus and the requested layers.
  A leftover checkpoint is reaped only if the verified final artifact
  subsumes it.
- **`topup`**: fit only the missing layers and union them in. Unchanged
  shards carry forward after a re-hash, or without one under a load proof.
- **`resume`**: continue from the durable artifact or checkpoint over a
  corpus prefix (`consumed_prefix_sha256`), whichever is farther ahead.
- **`fresh`**: nothing reusable, or `force`.

Extending the corpus of an artifact that holds layers the request omits is
refused, since one artifact has one corpus identity. Pass the full layer set or
`force`. When a payload fails to load, the executor drops it and re-plans.

`prove_lens_fit_noop` proves a no-op without the model. A file corpus is
identified by its raw digest, a pinned Hub corpus by spec plus prompt count
(before streaming). The source fingerprint is computed last and the payload
digests are verified. The CLI preflight and `fit_is_noop` (the server) both
call it. A fit evicts the resident lens before the estimator owns the base and
restores it on failure. A complete terminal checkpoint is promoted, not
rewritten.

## Residency and the generation protocol

`LensRuntime` holds the resident lens, its identity (`lens_identity`, the
sha256 of the canonical manifest or binding descriptor) and `BoundedLRU`
caches (two device transport stacks, decoded tokens, depth and selector
tensors). `LensInstrument` takes every dependency explicitly: model, capture
(`LatestSlices`), `ExclusiveGuard`, `capture_hidden`, the registry
(`ProfileStore`), `on_change`, `resolve_profile`, `assert_unsteered` and
`TokenReplay`.

- **`state_lock` is the lens-state boundary.** It is a reentrant leaf lock
  over refresh, adopt, evict, registry and live mutations, and `prepare`'s
  snapshot. Take the model-exclusive guard first, never while holding it.
- `resident()` reconciles with disk on every call: it evicts a removed,
  replaced or incompatible artifact and adopts a new one. A bound run's pin
  short-circuits it. `adopt` and `evict` rebuild probe layers, the live
  stack and the stash in one transaction, and evict every `jlens/` profile.
- **Per generation:** `prepare` → `plan` → `bind` → `close_run`. `prepare`
  refuses while a run is bound, because a stale pin would suppress its
  refresh. Its `LensPrep` is the atomic snapshot (lens, specs, live state,
  fingerprint). `plan` and `bind` read only the prep, never the registry: a
  concurrent adoption pairs lens A with lens B's layers, which fails with a
  `KeyError` in the transport stack. A run is pinned when live readout is on
  or when probes feed a final aggregate or a gate.
- `fit`, `readout` and `attach` run under the exclusive guard. `fit` takes
  `lens_fit_lock` first and calls `assert_unsteered`.

## Readout math

`LogitReadout(transport_rows(J, h))` reuses the model's own final path (norm,
OPT projection, logit scale, soft-cap). One unit runs through every surface:
the per-layer softmax probability `p_l(v)`.

- `strength = mean_l p_l(v)` over fitted layers, in [0, 1], with uniform
  weights (softmax already lets a confident layer dominate). `com`/`spread`
  are depth moments weighted by the same `p_l`, where depth = `layer /
  (n_layers − 1)`. Readouts are sharp (median per-layer max ≈ 0.8 on
  gemma-3-4b) and depth changes *which* token leads, so a token's profile over
  depth is its depth signal.
- Aggregate top-k ranks full-vocabulary strength. A union of per-layer top-ks
  would miss a token that is mid-pack everywhere.
- Gate and probe reads take exact columns against the full `logsumexp`, with
  no `[layers, vocab]` probability matrix. Results leave the device in one
  packed fp32 transfer (MPS has no float64). CUDA transport uses scoped TF32.
  Offline readouts chunk positions under `READOUT_CHUNK_BYTES` (256 MiB).

## Atoms, probes, gates

- **Reserved namespaces** (`naming.RESERVED_NAMESPACES`: `jlens`, `slens`,
  `sae`, `neuron`, `head`) name instrument units. Names stay verbatim and
  case-sensitive (`jlens/Paris` ≠ `jlens/paris`), take no `:variant`, skip the
  manifold tiers, and no manifold may be authored or installed under them.
  `slens` (input side, including `#<id>`) parses but has no resolver yet;
  steering it raises `ProfileNotRegisteredError`.
- **Steering atom.** `jlens/<word>` or `jlens/#<id>` resolves lazily through
  the registry's `jlens` atom resolver into `register_direction`. That yields
  `W_U[v] @ J_l` on every fitted layer (`W_U P J_l` under an OPT projection),
  registered as an ordinary profile that pushes, ablates and composes like
  any term. A word must be one token: the leading-space piece is tried first,
  and the match must survive a decode round trip (else `MultiTokenWordError`
  lists the pieces). `#<id>` is bounds-checked against the unembedding
  (`InvalidTokenIdError`). Lens atoms run hot. On gemma-3-4b α ≈ 0.3 was the
  coherent sweet spot and α ≥ 0.5 over-steered into repetition, because the
  atom is one sharp token direction rather than a distributed contrast.
- **Probe.** `add_probe("jlens/…")` routes through `readout/roster.py` to
  `attach`, never the Monitor: no whitener, no fold. It yields a
  `ScalarReading` (unit `mean_token_probability`) with a per-layer `p_l` trace
  and a depth summary (basis `readout_probability_mass`). `probe_hash` folds
  in the lens identity, so the same word under another source or a re-fit is
  a different probe. The final aggregate pools the same last-content-token
  slice as the Monitor and SAE.
- **Gate.** `@when:jlens/<word> > x` (also `[0]` or `#<id>`) fires only for an
  attached probe. The strength channel is the only one
  (`_GATE_CHANNELS = (Axis,)`); `:fraction`, `:membership`, `@label` and
  `~label` raise `UnsupportedProbeChannelError` at preflight. Gate scalars are
  computed post-forward and stashed by `step_id`, and the display step reuses
  them only from the same forward. A lens gate forces per-step lens reads
  whatever the live toggle says, but never per-token Monitor scoring.

## Live readout, replay, decomposition

- `set_live(True, layers=…)` (all fitted layers by default) puts the `J_l`
  stack on the device and attaches **no hooks**. `live_readout_step` reads
  the capture's latest slices post-forward at the token tap, so the steering
  fast path and compile eligibility are untouched. It reuses same-forward gate
  rows and syncs to the host once. The library default is off;
  `authored_capture` does the same for authored-prefill rows.
- `token_readout(node_id, raw_index, top_k=None, …)` replays the producing
  forward through `TokenReplay` (`apply_steering=False` reads the unsteered
  counterfactual). It returns the `scope="replay"` measurement envelope.
  `top_k=None` means 8; valid values are 1 to 256.
- `decompose(selector, k=16)` is greedy pursuit against `TokensOut`, never
  materialized, with chunked and cached atom norms. Selection is
  norm-normalized. It re-solves NNLS but takes the exact unconstrained solve
  when that is already nonnegative, and tiny solves run on CPU. `share` is the
  variance fraction the atoms carry, typically 6–15% for concept vectors.
  `LensDictionary` leaves room for `tokens-in` and signed `neurons`.

## Errors, seams, tests

- Errors: `JacobianLensError` (422), `LensNotFittedError` (404),
  `JacobianLensCancelled` (409), `RelpUnsupportedError` (422, in
  `errors.py`). `MultiTokenWordError`, `InvalidTokenIdError`,
  `EmptyCorpusError` and `SourceSpecError` are `RequestError`s (400). `HFError`
  (502) means an unresolvable repo commit.
- **External callers** make these names and the v7 layout an interface.
  neuron-lens calls `load_local_lens`, `save_lens`, `LensProvenance`,
  `JacobianLens(key=)`, `fit_jacobian_lens(kind=)`, `relp_backward_rules` and
  `RELP_RULES`. drowse.ai's hosted fit (`browser-runtime/fit_hosted_jlens.py`)
  calls `fit_jacobian_lens` and `save_lens_at`, and its browser reader
  accepts v7 packs only (`sites.out`, `out.layer-L` shards).
- Tests: CPU tests in `tests/lens/` (one file per module plus
  `test_migrate_lens_v7.py`) run on the toy model in
  `tests/support/jlens_toys.py`. They use `tests/support/lens.py`
  (`LensHarness`, `lens_instrument(**kw)`, `lens_kwargs(**kw)`,
  `FakeCapture`), which builds a real `LensInstrument` with session-shaped
  dependencies. Generation-boundary tests use `tiny_session`.
  `tests/gpu/test_jlens_gpu.py` micro-fits a real model (SmolLM2-360M or
  `DROWSE_TEST_MODEL`).
