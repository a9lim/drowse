# fit/

Corpora in, fitted manifold tensors out: the one extraction pipeline, corpus
generation, the neutral-activation capture behind the whitener, the offline
bake, cross-model alignment orchestration, and the naturalness eval. `fit`
sits above `generation` and below `loom`/`session`
(`tests/invariants/test_layering.py`), so it never imports the session; hosts
are Protocols (`ModelHandle`, `NaturalnessHost`) that `DrowseSession`
satisfies. The math it calls is in `geometry/AGENTS.md` (Fisher basis, frame,
share, discover coords, topology, RBF, σ-field, fold). The folder format,
sidecar schema, `method` table and locks are in `store/AGENTS.md`. The
session-side glue (`session/extraction.py::Extraction`: namespace resolution,
the exclusive guard, adopt/evict) is in `session/AGENTS.md`.

## Module map

| Module | Owns |
|---|---|
| `pipeline.py` | `ManifoldExtractionPipeline`, `prepare_manifold_capture_identity`/`CaptureIdentity`, `prepare_capture_rows`, `offline_fit_identity`, `FitInputs.accepts` (cache key), `MANIFOLD_CAPTURE_VERSION` |
| `publish.py` | Folder transactions: `fit_target_lock`, `take_authoring_snapshot`, `load_published_fit`, `publish_fit_if_current`, `ManifoldAuthoringChangedError` (409) |
| `preflight.py` | `preflight_manifold_fit_noop`: weight-free proof a fit is already exact |
| `capture.py` | Pooled residual capture, `_LENGTH_DIRECTIVE`, prompt/neutral corpus loaders, `compute_neutral_activations`, `compute_dls_axes` |
| `node_stats.py` | `compute_manifold_node_stats` (centroids + optional row spool), `compute_store_reduced_covariances` (σ-field input) |
| `capture_cache.py` | `CaptureCache`: content-addressed per-layer capture shards |
| `corpus.py` | `CorpusGenerator`, `KIND_TEMPLATES`, `system_for`, `role_for`, `split_composite_source` |
| `neutral.py` | `load_or_compute_neutral_activations[_with_metadata]` (compute half; `store/neutral.py` is the IO) |
| `bake.py` | `merge_into_manifold`, `shared_models` |
| `transfer.py` | `load_or_fit_transfer_alignment` → `TransferAlignment` |
| `naturalness.py` | `preflight`, `evaluate` (`experiment naturalness`) |

## One pipeline

A steering vector is the 2-node flat case of a manifold, so `session.extract`
and `manifold fit` both run `ManifoldExtractionPipeline(handle, events).fit`.
`extract` authors a discover `pca` folder with `{max_dim: 1, var_threshold:
0.7}`: two nodes for a bipolar axis (node 0 the positive pole; a composite
`pos.neg` is split by `split_composite_source`), one for a monopolar concept.
It returns the folded `Profile` (`geometry.fold.folded_directions`); the
folder is the only stored artifact. Stages: **inputs** (snapshot, identity,
fingerprints, SAE backend; an accepting published fit returns as is),
**capture**, **geometry** (pure tensors), **publish** (CAS). The pipeline
touches folder files only through `publish.py`.

## Conversational corpora

A node corpus is `list[str]` aligned `response[i] ↔ prompt[i % k]` against the
48 bundled `baseline_prompts.json` (a `~/.drowse/` override replaces it
outright) or a `template_ref` folder's contexts. Lengths must be non-empty
multiples of `k` (`ManifoldSpecError`). `CorpusGenerator` elicits in
character: system `f"{_LENGTH_DIRECTIVE} {system_for(c, kind)}"` under a
swapped assistant label (an explicit `role` beats `role_for`):

| `kind` | System (`{c}` humanized) | Label |
|---|---|---|
| `abstract` (default) | `You are someone {c}. Respond exactly as someone {c} would.` | `someone_<slug>` |
| `concrete` | `You are {art} {c}. Respond exactly as {art} {c} would.` | `<slug>` |
| `custom` | caller template (`{c}`, `{art}`) | none |

`abstract`/`concrete` raise `RoleSubstitutionUnsupportedError` without a
substitutable role header (Mistral); `custom` works everywhere. Capture
renders `[system: _LENGTH_DIRECTIVE, *history, assistant: response]`. The
persona is generation-only and its signal rides the text, so nodes pool in
standard-assistant space. Only an explicit manifest `role` (the
`:role-<name>` fit) is substituted at capture too. So `personas` (concrete,
no role) needs a role-capable family to *regenerate* but fits on any family.
`kind` is provenance: no geometry reads it, but it is in `nodes_sha256`.

`_LENGTH_DIRECTIVE` ("Answer in one short paragraph.") must stay identical at
all four sites (node and neutral generation, node and neutral capture). As
common-mode it cancels in `node − neutral`, matches capture framing to
generation framing, and keeps answers under the 256-token cap.

Sampling is HF `model.generate` over left-padded batches of 16,
`temperature=1.0`, `top_p=0.9`, samples-outer/prompts-inner. A row that
samples EOS first is retried alone, greedily with `min_new_tokens=8`, because
dropping it misaligns later rows and keeping it installs a blank; a second
blank raises. `fill(plan, seed=)` seeds node `i` with `seed + i` and writes
each node when done, so a resumed run equals a full one (plan from
`store.manifold_authoring.plan_discover_generation`). `CorpusGenerator` holds
no lock: `Extraction` guards it, but
`session.generate_responses`/`generate_neutral_responses` call it bare. The
neutral corpus uses the directive alone with the standard label.
`scripts/regenerate_bundled.py` drives all bundled corpora (fitting is a
separate `manifold fit`).

## Identity and the cache key

`nodes_sha256` (`ManifoldFolder.nodes_sha256`) covers labels, corpus bytes, roles,
kinds, domain+coords (authored) or fit_mode+hyperparams (discover), and the
resolved template hash. `capture_sha256` covers the loaded-model identity (id,
class, commit, model_type, loaded fingerprint, `MANIFOLD_CAPTURE_VERSION`),
node sizes, and every rendered row's token ids and pool index. It excludes
geometry, layer scope and folder, so geometry-only refits and identical
corpora share a capture. Its weight-free halves are `capture_render_sha256`
and `baseline_prompts_sha256`. `prepare_capture_rows` is the single
row-flattening owner for the fit and the preflight.

`FitInputs.accepts` matches `nodes_sha256`, `model_fingerprint`,
`capture_sha256`, `fit_policy_version` and `sae_release`. Raw fits also need
`evaluated_layers == requested` (the pre-DLS roster). SAE fits need the SAE
fingerprint, the revision when one is requested, and exact layers for an
explicit scope (else `sae_full_coverage`). `session/registry.py` recomputes
`capture_sha256` on every manifold load and refuses a mismatch, so the render
must stay deterministic.

| Constant | Bump invalidates |
|---|---|
| `MANIFOLD_CAPTURE_VERSION = 3` (`pipeline.py`) | all fitted tensors and all cached captures |
| `MANIFOLD_FIT_POLICY_VERSION = 1` (`geometry/manifold.py`) | fitted tensors only |
| `CAPTURE_CACHE_FORMAT = "5"` (`capture_cache.py`) | capture shards |
| `NEUTRAL_CAPTURE_VERSION = 1` (`store/neutral.py`) | neutral caches |

Gotchas: `dls` is not in the key, so toggling `--no-dls` on a fitted target
returns the cached tensor (and the preflight proves a no-op) unless forced.
Sidecars record no neutral identity, so regenerating the neutral corpus or
cache invalidates no fit.

Layers resolve via `model.layers.resolve_model_layers` (`all`, `workspace` =
the 40–90% band, or indices; `sample` is refused). After the cache check and
before any forward, `_require_neutral_baseline` raises `WhitenerError` unless
the whitener `covers_all` fit layers and `layer_means` covers them; there is
no Euclidean fallback. With `sae=`, each centroid is reconstructed through the
SAE one layer at a time. A default SAE fit covers exactly the release's layers
(`SaeCoverageError`). SAE fits exclude roles, keep no rows (so no σ-field) and
stay model-space, so hooks never touch the SAE.

## Capture

Rows are rendered and tokenized once on CPU (thinking disabled where the
template takes `enable_thinking`) and pool at `model.tokens.last_content_index`,
the last non-special token before chat-template trailers. `compute_manifold_node_stats` streams all rows across
node boundaries, length-sorted, in right-padded batches that start at
`_CAPTURE_BATCH_MAX = 64` and halve on OOM (never growing back).
Right-padding is exact because the pooled position attends only to real
tokens; the mask is built only for ragged chunks. `_ReusablePooledCapture`
registers hooks once per fit and gathers `(B, D)` inside the hook.
`_CaptureComplete` is a `BaseException` so wrappers' `except Exception` can't
swallow it; it aborts after the last requested block, so the final norm and LM
head never run. Centroid-only fits reduce per node on device in fp32. Curved
raw fits spool source-dtype rows to a layer-major mmap `ActivationRowStore`.

`CaptureCache` writes
`models/<safe>/manifold_capture/<sha>.{centroids,rows}.layer_<L>.safetensors`.
Shards are self-describing, fsynced and atomically replaced, with no pointer
or journal. A missing, foreign or malformed shard is a per-layer miss (worst
case: recapture). Validation is header, shape and finiteness, not a payload
digest. Write failures only log. `prune()` evicts whole digests, oldest first
and never the current one, down to `DROWSE_MANIFOLD_CAPTURE_CACHE_GB`
(default 8; `<= 0` off), and `pack clear`/`rm` GC unreferenced groups. Rows
are kept only for raw `authored`/`spectral`/`auto` fits. `auto` spools rows
only alongside missing centroids and persists or tops them up only if it
resolves curved. `force` treats the selected layers as misses.

## Geometry orchestration

**Monopolar** (1-node `pca`): `concept − ν` per layer is folded by
`fold_directions_to_subspace` into a neutral-anchored ray. There are no
discover coords, no DLS and no synthetic second node, and the fit needs ν
(`WhitenerError` without it).

**K ≥ 2**: `_whitened_node_geometry` computes `Σ⁻¹X̃` and `G_L = X̃Σ⁻¹X̃ᵀ` of
the node-mean-centered centroids once per layer, plus `node_spread = tr(G_L)`.
Discovery, topology, the Fisher basis and the neutral layout all reuse them.
`_resolve_fit_geometry` is the only mode branch:

| `fit_mode` | Coords / domain | Per-layer fit |
|---|---|---|
| `authored` | manifest | curved, exact (node = steering target) |
| `pca` | `derive_pca_coords(mean_L G_L)`, `CustomDomain(k)` | flat affine |
| `spectral` | `derive_spectral_coords(mean_L G_L)` | curved, GCV smoothing |
| `auto` | `select_topology`: flat, curved, or periodic `BoxDomain` | as resolved |

The consensus Gram `mean_L G_L` is signal-weighted and needs no reference
layer. Node floors are `k+1` flat and `min_nodes(k) = 2k+1` plus poisedness
curved (`ManifoldSpecError`/`PoisednessError`). `HYPERPARAMS_BY_MODE`
whitelists keys per mode. **`auto` forwards only `max_dim`, `smoothing`,
`persistence_frac` and `max_subspace_dim`** (as the scoring/Fisher width). It
accepts `var_threshold`/`min_dim`/`k_nn`/`bandwidth` and hashes them into the
identity, but never uses them.

**Flat**: `fit_affine_subspace` with `n_components` = the layout width (a flat
subspace *is* its layout span, so `max_subspace_dim` does not apply) and
`orient_to=0` (node 0 positive, so a 2-node fit is `+δ̂`: PCA@2 ≡ DiM). DLS
then prunes axes and `subspace_share` is baked on the survivors. The display
layout is translated onto neutral (`_neutral_centered_layout`) so `% 0,…,0`
reads as neutral; steering is unchanged. No origin is stored (neutral's foot
is coord 0).

**Curved**: `fit_curved_subspace` with one `RbfFitPlan` (auto reuses its
scoring plan and Fisher bases). Smoothing is `None` (exact) for authored and
`"auto"` (GCV) otherwise; `0` means exact and a float is a fixed λ. The width
is `max_subspace_dim` (default 64). Raw fits then read the retained rows for
the σ-field, and each layer stores `origin`, the neutral foot that seeds the
solve. There is no DLS, since a curved fit has no polarity.

**DLS** (`compute_dls_axes`; Dang & Ngo 2026, Selective Steering, Eq. 9) keeps
axis `d̂_r` at layer L iff `{(c_i − ν_L)·d̂_r}` straddles zero. Same-side axes
encode intensity, not polarity, and at K=2 this is exactly the bipolar
opposite-sign test. Edge cases: `layer_means=None` (`dls=False`) keeps every
axis; a zero-norm axis drops; a layer with no ν keeps every checkable axis; if
nothing passes anywhere, every checkable axis is kept with a `UserWarning`; a
layer left empty drops. `tests/fixtures/browser/dls-axes-v1.json` pins this
and drowse.ai replays it, so a semantic change means regenerating the golden.

## Publish and concurrency

`fit_target_lock` is per (folder, tensor filename) and keyed at
`<namespace>/.drowse-fit-<sha>`, outside the removable folder. Distinct
targets fit concurrently, and readers keep the old pair.
`take_authoring_snapshot` runs under the manifest lock. It repairs a
one-sided pair, merges discover overrides (a mode change resets hyperparams),
re-materializes a drifted templated corpus, mints `artifact_id`, and hashes
`revision = H(artifact_id, target fit_epochs, nodes_sha256, prompts)`. A cache
hit takes only the pair lock and then re-asserts the revision; compute holds
no folder lock. `publish_fit_if_current` raises
`ManifoldAuthoringChangedError` (409, retry) on drift. `pack clear` bumps
`fit_epochs` and rm+recreate mints a new `artifact_id`, so a paused fit can
never publish over either.

`preflight_manifold_fit_noop` runs before the CLI opens a session and returns
a proof or `None`; `None` means *unproven*, not stale. It declines SAE fits,
overrides, `baked`, an unprovable source, a missing neutral cache, an
unloadable tokenizer and templated drift. It proves every key component
offline, bridging the loaded fingerprint through the neutral cache sidecar,
which a real load bound to the source fingerprint.

## Neutral activations

`load_or_compute_neutral_activations` is single-flight under
`neutral_fit_lock`, spanning recheck, capture and publish. A stale cache is
recomputed, and a corrupt one warns and is recomputed. It renders the neutral
pairs (`response[i] ↔ prompt[i % k]`), captures every layer with the same
batched path as node capture, and returns fp32 CPU `[N, D]` per layer.
`session/neutral.py` derives the means and the whitener from it (format and
the fp32 rationale: `store/AGENTS.md`).

## Bake

`merge_into_manifold(name, expression, model, *, force, strict,
expected_model_fingerprint)` writes `local/<name>` (`fit_mode="baked"`).

- **Grammar.** Only namespace-qualified scalar `+`/`-` terms. `!`, `%`,
  `~`/`|`, triggers and bare names raise `MergeError` (400): projection is
  Mahalanobis-only and needs an identity-matched whitener, so an offline
  stand-in would change semantics.
- **Components.** Fitted 2-node `pca` manifolds folded to directions.
  `:role-<r>` needs a uniform role, `raw` rejects role fits, and `:from-<src>`
  must match `source_model_id`.
- **Combining.** `linear_sum` unions layers (an absent component adds 0, as in
  live composition); `strict` demands equal coverage.
- **Fold and publish.** Folding uses each model's persisted whitener
  (`load_neutral_whitener`, no model load). Component fingerprints must agree,
  and the whole staged folder swaps in.
- **Callers.** CLI `manifold bake` is offline (no `-m` = every shared model).
  `session.bake` scopes to the loaded model and guards its fingerprint. There
  is no server route.

## Transfer alignment

`load_or_fit_transfer_alignment(src, tgt, *, force, requested_layers)` returns
the per-layer map, R² quality, both neutral identities, and the target
whitener and means for the re-bake (`store.manifold_lifecycle.transfer_manifold`).
The concurrency is load-bearing:

- The directional `alignment_fit_lock` is held throughout.
- A model loads only on a neutral miss, under its `neutral_fit_lock` taken
  before the load, so concurrent cold transfers share one load.
- Header-only proofs are re-checked after load; a racing writer restarts the
  transaction.
- Unrequested layers drop early.
- `force` refits the map, not an exact neutral cache.

## Naturalness eval

`preflight` is model-free: it resolves a folder or a unique selector and
rejects `baked`, fewer than 3 nodes, and `--compare-linear` without a single
`%` term. `evaluate(host, ...)` then:

- **Node centroids.** The unsteered next-token distributions at each
  response's positions (capture framing), averaged per response and then per
  node, one unbatched forward per response.
- **Behavior manifold.** Centroids in Hellinger space (`√p`), fitted with
  Euclidean PCA + exact r³ RBF over the manifold's per-model layout. Output
  space has no rogue dims, so there is no whitener. A discover folder's layout
  calls `host.fit(folder)`, a full fit if unfitted.
- **Trajectory.** One steered generation (`seed=0`), then the unsteered
  distributions along its text, scored per step as the Bhattacharyya distance
  to the nearest manifold point (low = natural).
- **`compare_linear`.** Also scores the node-0→position chord, steered as
  profile `__manifold_linear_baseline__`.

## Fit behavior notes

The bundled set is listed in `store/AGENTS.md`; the α regime and gain
calibration are in `steering/AGENTS.md`. Observed on fitted bundled manifolds
(qualitative; MPS is not deterministic):

- The whitened fit stays coherent at strong push where a Euclidean fit
  loop-collapses: caveman reaches terse primitive grammar, hacker a guarded
  intrusion register.
- `personas` and `emotions` resolve flat under `auto` on gemma-4-12b. The
  `months` ring is authored because a per-model auto fit GCV-prefers flat.
- Persona blending, low-α drift and the near-1-D persona fan are the open
  frontiers in `ARCHITECTURE.md` §17.

## Testing

`tests/fit/` runs on CPU: `tests/support/capture.py::install_capture` patches
`drowse.fit.capture._encode_and_capture_all_batch`, and `StubHandle` is the
`ModelHandle`. Elsewhere: extraction glue `tests/support/extraction.py`,
corpora `tests/session/test_generate_responses.py`, session bake
`tests/session/test_session_bake.py`, neutral cache
`tests/store/test_neutral_cache.py`, DLS golden
`tests/browser/test_browser_parity.py`, GPU `tests/gpu/test_smoke.py` and
`tests/gpu/test_session.py`.

## Deliberately absent

No extraction method besides DiM (PCA@2) and no standalone
`{positive, negative}` corpus artifact; no Euclidean activation-space fit; no
reference-layer, `anchor_origin` or flat `max_subspace_dim` knob; no separate
`layer_means` cache; no synthetic monopolar node; no `pca` tensor variant.
`--method` exists only on discover fit/merge.
