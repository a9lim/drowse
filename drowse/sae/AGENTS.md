# drowse/sae — sparse autoencoders

Provider (SAELens) and Drowse-trained dictionaries, their per-model source
registry, the one resident dictionary a session reads, the Neuronpedia client,
local training, and the `sae` read instrument (`session.sae`,
`session.instruments["sae"]`). The instrument contract is in
`drowse/readout/AGENTS.md`; the source shape shared with the lens is
`drowse/base/source_registry.py`.

`sae` shares a layer with `lens` (above `readout`, below `steering.composer`):
it never imports `generation`, `fit`, `loom` or `session`, type-only imports
included (`tests/invariants/test_layering.py`). `DrowseSession.__init__` wires
every dependency explicitly.

| Module | Owns |
|---|---|
| `backend.py` | `SaeBackend` protocol, `SaeLensBackend` / `LocalSaeBackend`, provider resolution + commit pinning, `resolve_sae_backend`, `select_runtime_layer`, `list_sae_releases`, `sae_binding` |
| `sources.py` | `SAE_SOURCES` (a `SourceFamily`), both source grammars, provider bindings, the Neuronpedia feature cache, use/remove/list |
| `store.py` | Drowse-trained artifacts under `sae/local/<name>/` |
| `runtime.py` | `SaeRuntime` / `SaeResidency`: residency lifecycle (load/unload/train/restore), feature validation + metadata |
| `neuronpedia.py` | One validated HTTP read per feature; no session, lock or disk |
| `instrument.py` | `SaeInstrument` + `SaeRun`: probes, gates, live discovery, aggregate, replay, `sae/<id>` directions |
| `training.py` | `train_residual_sae`, `SaeTrainingCancelled` |

`store.py` imports `sources.py` at module level; `sources.py` imports `store.py`
only inside functions. Keep that direction.

## The `[sae]` extra

`sae = ["sae-lens>=6.0,<7.0", "nltk>=3.10.3"]` (nltk pinned past its CVE'd
releases). SAELens is not a base dependency. It is imported lazily on provider
paths only (`load_sae_backend` for a non-`local:` release, `list_sae_releases`).
Local training and loading, `sae/<id>` steering, probes and Neuronpedia
(through `huggingface_hub`) work without it. Missing SAELens raises
`SaeBackendImportError` with the install hint, except that `list_sae_releases`
returns `[]` when local artifacts exist. `sae train` without `--corpus` streams
FineWeb-Edu via `drowse/lens/corpus.py`, which needs `[hf]`. SAELens 6 shims
(registry location, `from_pretrained_with_cfg_and_sparsity` vs
`from_pretrained`, string `dtype`) live in `backend.py`.

## Sources and disk layout

`SAE_SOURCES` is the one parser of two grammars; never hand-build either.

- **Public source:** `local:NAME` (`NAME_REGEX`) or `saelens:RELEASE`. A
  release only has to be non-blank (it may carry slashes and mixed case).
- **Backend release string:** `local:NAME` or the bare provider release.
  `parse_sae_release` / `sae_release` convert it to and from a `SourceRef`.

```
models/<safe_model_id>/sae/
  active.json                      one selection per model (SAE_SOURCE_FORMAT_VERSION 1)
  bindings/_z<b32>.json            provider binding (SAE_RUNTIME_FORMAT_VERSION 3)
  bindings/_z<b32>-features.json   Neuronpedia cache (SAE_FEATURE_META_FORMAT_VERSION 4)
  local/<name>/manifest.json       trained artifact (LOCAL_SAE_FORMAT_VERSION 1)
  local/<name>/layer-<L>.gen-<hex>.safetensors   fp32 W_enc, W_dec, b_enc, b_dec
```

`_z<b32>` is `base.paths.encode_release_id` (lowercase unpadded Base32,
exactly reversible). Provider weights stay in the SAELens / Hugging Face cache;
Drowse never copies them into `DROWSE_HOME` or writes into a provider cache.

- **Binding.** `sae_binding(backend, layer, width)` is the one shape
  (`{layer, width, revision, fingerprint, sae_id, repo_id, neuronpedia_id}`)
  behind the binding file (plus `format_version`, `model_id`, `release`),
  `session.sae.info` and the feature-cache key. Readers demand the exact key
  set and valid values, and anything else reads as absent, never half-trusted.
  `save_sae_metadata(activate=True)` also selects the binding.
- **Feature cache.** `{format_version, model_id, release, source, features}`,
  where `source` is the seven binding fields, so a cache from another layer,
  revision or dictionary is ignored whole. Rows are exactly `{label, max_act}`
  under canonical decimal ids, and one bad row voids the file. The in-memory
  `checked` flag is not persisted.
- **Local artifact.** The manifest schema is exact (`activation: "relu"`, one
  `layer`, model and source fingerprints, corpus spec + sha256, hyperparameters,
  `tensor_file` + `tensor_sha256`). `save_local_sae` writes a fresh
  generation-named payload, fsyncs and hashes it, writes the manifest atomically
  under `artifact_lock`, then unlinks older payloads. It raises
  `FileExistsError` without `force`. `load_local_sae_manifest` checks the
  schema and safetensors header only and never hashes the payload, because
  listing runs on every dashboard refresh. `load_local_sae_tensors` verifies
  the digest (through `verify_integrity`'s stat cache) and rejects non-finite
  weights.
- **Lifecycle.** `use_sae_source` selects only a prepared source
  (`FileNotFoundError` otherwise). `remove_sae_source` deletes a local folder,
  or for a provider only its binding and feature cache. It clears the
  selection afterwards, and only if the selection named that source.
  `list_sae_sources` returns local plus binding rows; a binding row carries
  `description_source` `{model, source, repository, saeId}` when known.

## Backends (`backend.py`)

`SaeBackend`: `release`, `revision`, `fingerprint`, `layers` (0-indexed blocks),
`encode_layer(idx, (N, D)) → (N, F)`, `decode_layer`, `feature_count`,
`feature_direction` (the decoder row).

- **`SaeLensBackend`** resolves the registry eagerly and loads weights lazily,
  keeping exactly one layer resident: fits visit layers one at a time, and a
  whole release can outweigh the model. Encode/decode delegate to the provider
  module. Never reimplement them: Gemma Scope 2 JumpReLU encodes uncentered
  (`apply_b_dec_to_input=False`) while local SAEs subtract `b_dec`
  (`tests/sae/test_sae_numerics.py` pins both).
- **`LocalSaeBackend`** loads once, on first use, at the model's device and
  dtype (fp32 when none is given). It encodes `relu((h − b_dec) W_enc + b_enc)`,
  covers only its trained layer, and uses the payload sha256 as both `revision`
  and `fingerprint`.

`load_sae_backend` resolves a provider release:

- **Registry and model.** An unknown release raises `SaeReleaseNotFoundError`
  with difflib suggestions. `_model_names_match` compares the last path segment
  case-insensitively and accepts a substring either way; a mismatch raises
  `SaeModelMismatchError`.
- **One SAE per layer.** `_canonical_layer_map` ranks each layer's candidates:
  Neuronpedia-hosted first, then narrowest width, then smallest L0
  (`small < medium < big`), then id. Hosting comes first because it supplies
  the labels and the `maxActApprox` strength unit; an unhosted pick silently
  loses the normalized channel. It warns when it had to choose.
- **Immutable identity.** With a `repo_id`, the revision resolves to a Hub
  commit (`model_info(..., timeout=3).sha`). A full 40-hex revision skips the
  round trip, which is what lets a pinned restore run offline. If no commit
  resolves, every layer load raises: mutable weights are never mixed or
  persisted. Without a `repo_id`, only a registry `content_fingerprint` /
  `fingerprint` supplies identity. `fingerprint` hashes `{release, repo_id,
  resolved_commit, sae_ids_by_layer, sae_lens_version, conversion_func,
  config_overrides, dtype}` and is what fitted-tensor reuse compares.
- **Pinning.** The commit is passed as `revision=` when the loader accepts it.
  Otherwise `_commit_pinned_converter` clones the release's converter into
  isolated globals with every Hub read pinned (no process-wide monkeypatch).
  If neither is possible, the load is refused.
- **Per-layer checks.** The loaded config's `hook_layer` must equal the
  registry mapping. `_validate_residual_hook` rejects hook names ending in
  neither `.hook_resid_post` nor `layers.<L>.output`; a config with no hook
  name falls through to the width check.

**Residual-post only.** Capture is the block output. An attention/MLP SAE fed
that tensor fails on width or, when widths agree, silently returns meaningless
activations. Discovery (`list_sae_releases`, metadata-only) drops releases
whose names mark attention/MLP/transcoder families (`_release_hook_kind`); the
loaded SAE stays authoritative.

`resolve_sae_backend(...) → ResolvedSae(backend, layer, width)` is the one path
behind `sae fetch` and `SaeRuntime.load`. It rejects a local artifact whose
`model_fingerprint` differs from the loaded weights (before its weights load),
picks the layer, materializes that layer to read its width, and checks
decoder-row width against the model's hidden size. Persisting is the caller's
job. `sae fetch` therefore downloads and materializes one SAE layer on CPU but
never instantiates the base model (shape from `config_model_shape`).

`select_runtime_layer(available, n_layers, requested)`: an explicit layer must
be covered (`SaeCoverageError`). Otherwise the covered layer nearest
`0.65·(n_layers − 1)` wins, restricted to the 0.4–0.9 depth band when any
covered layer lies in it, with the lower layer on ties. The choice is
deterministic, so restores and the unqualified `sae/<id>` stay stable across
launches; `sae train` uses it for its default `--layer`. `sae_device_str`
maps `mps:0` to `mps` (safetensors and SAELens reject the indexed form).

## Residency (`runtime.py`)

A session holds at most one dictionary at one hook layer:
`SaeResidency(backend, layer, width, feature_meta)`, frozen with `eq=False`, so
each load's identity is the "still current?" token. Feature ids mean something
only against that release and layer, which is why `sae/<id>` takes no
qualifier.

- **Dependencies.** `SaeModelHandle` (the session's public `model_id`, `model`,
  `tokenizer`, `device`, `dtype`, `layers`, `model_info["hidden_dim"]`), the
  exclusive-GPU section (`base.exclusive.ExclusiveGuard`), and
  `SteeringComposer.assert_unsteered`. `state_lock` is a reentrant leaf lock
  shared with the instrument, never held across a forward or a fetch.
- **`load`** holds the exclusive section across provider resolution and the
  weight download, which is why the server switches sources only as a
  background `fetch` preparation (a direct switch returns 409). The order is
  resolve → load the feature cache (provider only) → publish binding +
  selection → swap and `on_swap` under `state_lock`. A failure before the swap
  leaves the previous source resident. Every successful load rewrites
  `active.json`.
- **`restore_active`** runs at the end of `DrowseSession.__init__`. It reloads
  the binding's pinned commit and layer, so the restore never drifts to
  upstream HEAD and works offline from the provider cache. Failure only logs a
  warning.
- **`train`** rejects an empty corpus (`RequestError`) and records
  `sha256(repr(docs))`. It runs in the exclusive section with `phase_msg`
  (refused during a generation) and calls `assert_unsteered()`, because an
  open steering scope would bake into the artifact. It trains on
  `model._orig_mod` when compiled and saves with the model and source
  fingerprints. It then loads the result at the trained layer, in its own
  exclusive section, and returns `{source, artifact, metrics, runtime}`.
- **Feature metadata.** `check_feature` (parse + range, `SaeFeatureError`) is
  the steering path's only check. `validate_feature` and `fetch_meta` fetch
  outside every lock (at most `_FETCH_WORKERS = 8` concurrent requests), then
  publish under `state_lock` only if the residency is still current: a late
  response never labels a successor SAE. A feature is fetched when it is
  uncached, or lacks a label or `max_act` and was never `checked`.
  `fetch_meta` is the dashboard's between-generation backfill, never called
  from the decode loop.
- **Tensors.** `encode` returns detached activations (some providers return
  grad-tracked ones). `feature_direction` returns `W_dec[id]` on the model's
  device and dtype.

## Neuronpedia (`neuronpedia.py`)

`GET https://www.neuronpedia.org/api/feature/<model>/<source>/<index>` via
`huggingface_hub.get_session()` (UA `drowse-sae-meta/1`, 10 s timeout).
`<model>/<source>` is the binding's `neuronpedia_id`. The response must match
`modelId`, `layer` (= source) and `index`, and `source.hfRepoId` /
`source.saelensSaeId` whenever the binding knows them; a mismatch returns
`None` and nothing is cached. The label is the first non-blank explanation;
`max_act` is `maxActApprox` when finite and positive. A transport failure
returns `None` and is retried later. Local SAEs have no Neuronpedia id, so they
always read in the raw unit.

## The instrument (`instrument.py`)

The constructor takes the read plane's capture (`LatestSlices`), the runtime,
the loom `TokenReplay`, and three callbacks: `put_direction` /
`evict_directions` (the artifact registry) and `forget_probe_hashes` (the probe
roster).

**One channel.** An SAE probe emits `<name>` and `<name>[0]`: normalized
strength `activation / maxActApprox`, roughly in [0, 1] and comparable across
features. It reads raw activation when `max_act` is unknown, and
`ScalarReading.unit` (`activation_over_max` / `raw_activation`) says which.
`per_layer` is `{layer: value}` and depth is `layer/(n_layers − 1)` with spread
0 and basis `single_layer`. There is no `:fraction`, `:membership`, `@label` or
`~label` channel. `validate_gate` admits only the axis, so such a gate fails at
generation preflight (`ReadPlane.resolve_gates`, `UnsupportedProbeChannelError`). Never emit constant
placeholder channels.

- **Probes.** `attach("sae/<id>", as_name=)` validates first (which may fetch,
  outside locks), then inserts under the exclusive section and `state_lock`.
  The default name is `sae/<int id>`, the spec `{feature_id, layer, label,
  max_act}`. These are not Monitor probes and use no whitener. `try_detach` is
  the only removal.
- **Per-generation run** (`prepare → plan → bind → close_run`, driven by
  `drowse/generation/read_plane.py`). `plan` captures the live layer, plus the
  resident layer when probes feed a final aggregate (tail ring) or a gate.
  `bind` freezes specs with `max_act` resolved from the cache and snapshots
  the live config, so a backfill or a cross-thread toggle cannot change what a
  running generation measures. Idle reads take a locked snapshot of the live
  registry, so a refresh between generations applies at once. A probe detached
  mid-generation stays in that generation's aggregate. `prepare` on a bound
  run raises.
- **One encode per step.** `gate_scalars` encodes the latest slice and stashes
  `{activations, step, raw_by_fid}`. `live_readout_step` and pinned probes
  reuse it only when `step` matches their `step_id` (negative never matches),
  so staleness is structural. `probe_values` gathers every not-yet-read pinned
  id with one `index_select` and one host transfer, caching selector tensors
  in a `BoundedLRU(64)`. `gate_keys=None` scores the full roster and an empty
  set scores nothing. `score_aggregate` reads the pooled last-content-token
  slice, the same position as the monitor and lens.
- **Live discovery.** The live config is `{layer, source}` at the resident
  layer. `set_live` takes no extras (`TypeError`), because the top-k width
  follows the generation's resolved alternatives (`return_top_k`). Rows are
  `(id, raw activation, label, max_act)`, with label and `max_act` from the
  cache only. `authored_capture` is the same computation for authored-prefill
  rows.
- **Replay.** `token_readout(node_id, raw_index, top_k=None → 8, layers=None,
  …)` rebuilds the producing forward through `TokenReplay` and returns the
  `scope="replay"` envelope. `top_k` must lie in `[1, min(width, 256)]`, and a
  `layers` argument raises `RequestError` rather than being ignored.
- **Steering atoms.** The grammar reads `sae/<int>` as a feature id (`#` ids
  and variants are rejected). The registry's `sae` atom resolver calls
  `register_direction`, which puts `W_dec[id]` at the resident layer as the
  one-layer profile `sae/<id>`; it lowers like any vector term. It runs while
  the composer holds the exclusive section, so it range-checks and never
  fetches. Steering never clamps or decodes features inside the forward.
- **Swap.** `_on_swap` runs under `state_lock` inside the exclusive section,
  after the residency has swapped. It clears probes, the live config and the
  stash, evicts every `sae/` registry profile and forgets probe hashes. Never
  carry feature ids across a swap. `_on_fetched` replaces each matching probe
  spec dict whole (idle readers may hold references) and forgets its hash.
- **Identity.** `probe_hash` hashes `("sae-readout-v2", model_id,
  fingerprint, release, layer, feature_id, max_act)`; `max_act` is included
  because it sets the unit. `active_source` derives from residency, not
  `active.json`, and is `None` when nothing is loaded.

## Local training (`training.py`)

`train_residual_sae` fits a one-layer ReLU SAE on block outputs, for models with
no published SAE.

- **Capture.** Forwards run under `inference_mode` with `use_cache=False`. A
  hook on the chosen block raises `_SaeCaptureComplete` to skip the remaining
  blocks and the LM head. It is a `BaseException` because a broad
  `except Exception` in third-party modeling code would swallow it and run
  every batch through the full forward. Capture must happen exactly once and
  keeps only unmasked tokens, in fp32.
- **Corpus.** Documents are truncated to `seq_len` (under 2 tokens dropped),
  right-padded, and cycled in epochs until `tokens`.
- **Optimization.** Decoder rows start as seeded unit-norm vectors, with
  `W_enc = W_decᵀ`, `b_enc = 0` and `b_dec` the first batch's mean. Adam
  minimizes `MSE + l1·mean|f|`. Decoder rows are renormalized every step;
  otherwise L1 is evaded by shrinking the encoder and growing the decoder.
  Width is `d_sae`, or `d_model × expansion`.
- **Defaults** (runtime and CLI): tokens 1,000,000, seq_len 128, batch 8,
  expansion 8, lr 3e-4, l1 1e-3, dead threshold 1e-6, seed 0.
- **Device work.** Fire counts accumulate on the device (a per-step host copy
  of `[tokens, width]` dominated accelerator time), and each step makes one
  host transfer, for its three scalars.
- **Cancellation.** `cancel_event` is checked before each batch and after
  capture, raising `SaeTrainingCancelled` (`RuntimeError` + `DrowseError`, 409,
  the `JacobianLensCancelled` contract).

## Fits in SAE space

`--sae REL` / `sae=` on extract and fit is documented in `drowse/fit/AGENTS.md`.
The pipeline loads its own backend with `load_sae_backend`, not the resident
runtime, and round-trips each centroid through it one layer at a time over the
release's covered layers. It writes `<safe>_sae-<encoded release>.safetensors`
in model space, so hooks never touch the SAE. It cannot be role-baselined.

## Errors

| Error | Base | HTTP |
|---|---|---|
| `SaeBackendImportError` | `ImportError` | 400 |
| `SaeReleaseNotFoundError`, `SaeModelMismatchError`, `SaeCoverageError`, `SaeFeatureError`, `SourceSpecError` (blank release), `UnsupportedProbeChannelError` | `RequestError` | 400 |
| `SaeNotLoadedError` | `RuntimeError` | 404 |
| `SaeTrainingCancelled` | `RuntimeError` | 409 |

The bare `ValueError`s in `backend.py` and `runtime.py` are engine invariants
(an unresolvable commit, a local payload that changed during load, an encoder
width mismatch), which the server scrubs to a 500. Anything a client can
trigger with a bad value must be a `RequestError` subclass.

## Surfaces

- **CLI** (`drowse/cli/verbs/sae.py`): `sae use` only writes the selection,
  and the weights load at the next session start via `restore_active`.
- **Server** (`drowse/server/instrument_routes.py`): unlike the CLI's pure-IO
  fetch, the `fetch` preparation loads into the live session and enables live
  discovery.
- **`drowse serve`** in web mode calls `enable_live_sae` (the active or a
  prepared source, else `best_sae_release`). `--no-web` never imports SAELens
  or downloads provider weights implicitly.

## Testing

CPU tests live in `tests/sae/` (backends and pinning, the local store and
trainer, Neuronpedia validation and races, encoder numerics, runtime and
instrument). `tests/support/sae.py` provides `MockSaeBackend`,
`sae_instrument(...)` (a real instrument and runtime with a no-op exclusive
section and recorded registry effects) and `make_resident(runtime, backend,
layer=...)`. Session-level tests load a mock through `session.sae.load` with
`drowse.sae.runtime.resolve_sae_backend` monkeypatched. Tests that need SAELens
`importorskip` it. The real end-to-end run, `tests/gpu/test_sae_lens_smoke.py`,
is opt-in via `DROWSE_TEST_SAELENS=1` and is marked `gpu_gemma`.
