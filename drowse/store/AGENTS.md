# drowse/store

On-disk artifacts and their distribution: manifold folders, the selector index
and bare-atom ladder, templates, bundled materialization, the `.drowse` archive,
install, Hugging Face push/pull/search, GGUF export, and the per-model neutral and
alignment caches.

Layer: above `model`, below `steering.grammar` (`parse_expr` resolves bare atoms
through `store.selectors`). Store imports only `errors`/`naming`/`events`, `base`,
`geometry`, `model` and itself; `tests/invariants/test_layering.py` counts
function-level and `TYPE_CHECKING` imports too. No barrel module: import from the
submodule. This package owns formats and folder primitives; the fit's folder
transactions (authoring snapshot, compare-and-swap publish) are `fit/publish.py`,
the weight-free fit no-op proof `fit/preflight.py`, the capture cache
`fit/capture_cache.py`.

## On-disk layout

All paths resolve through `base.paths.drowse_home()` (`$DROWSE_HOME`, default
`~/.drowse`); `ensure_within` re-checks each user-supplied component.

```
$DROWSE_HOME/
  config.yaml                      CLI/server defaults (cli/config_file.py)
  baseline_prompts.json            override for drowse/data/baseline_prompts.json
  neutral_statements.json          override for drowse/data/neutral_statements.json
  templates/<ns>/<name>/template.json
  manifolds/<ns>/<name>/
    manifold.json
    nodes/NN_<label>.json          one JSON list of strings per node
    <safe>.safetensors + .json               raw fit
    <safe>_sae-<enc>.safetensors + .json     SAE feature-space fit
    <safe>_from-<enc>.safetensors + .json    cross-model transfer
  {manifolds,templates}/<ns>/.locks/         lock files, outside the removable folder
  models/<safe>/
    neutral_activations.json       pointer -> neutral_activations.layer-<L>.gen-<hex>.safetensors
    alignments/<safe_src>.json     pointer -> <safe_src>.layer-<L>.gen-<hex>.safetensors
    manifold_capture/              content-addressed fit captures (fit/capture_cache.py)
    lens/, sae/                    source families (lens/AGENTS.md, sae/AGENTS.md)
  .staging/<uuid>/                 scratch for one whole-folder build (base/staging.py)
  .staging/backup/<path>           the parked old folder during a swap
```

`<safe>` is `safe_model_id` (`_z` + unpadded urlsafe base64); `<enc>` is
`encode_release_id` (`_z` + unpadded lowercase base32) of the SAE release or
transfer source id. `tensor_filename`/`parse_tensor_filename` (`base/paths.py`)
escape separators inside components; a non-canonical name parses as `None` and is
not a fitted file. One kind per file, no method suffix, no `pca` variant;
`role-<x>` is a selector suffix only and addresses the canonical tensor.

## The manifold folder

**`manifold.json`** (`MANIFOLD_FORMAT_VERSION = 10`, exact;
`manifold_schema.parse_manifest` validates, `manifest_payload` builds). Keys:
`format_version, name, description, fit_mode, nodes, files, source, tags,
template_ref` (key required, value nullable), plus `hyperparams` (discover) or
`domain` (authored/baked), plus optional local state `artifact_id`/`fit_epochs`.
Unknown keys raise. A node is exactly `{label, role, kind}` (`+ coords` if
authored). Names follow `NAME_REGEX = ^[a-z][a-z0-9._-]{0,63}$`; labels the
stricter `LABEL_REGEX = ^[a-z][a-z0-9_-]{0,63}$`, because `.` is the bipolar
separator and labels are typed after `%`. `role` (assistant-role substitution
slug) is a fit input; `kind` (`abstract|concrete|custom`) is provenance the fit
never reads. A folder is discoverable only if its directory name equals `name`.

**Fit modes.** `authored` (domain + coords, curved RBF, at least
`min_nodes(n) = 2n+1` nodes, poised at fit time); discover `pca` (flat),
`spectral` (curved), `auto` (`select_topology` picks per model); `baked`
(corpus-less, geometry only in the tensor, never re-fits). `sanitize_hyperparams`
(`HYPERPARAMS_BY_MODE`) raises on a knob the mode would ignore rather than
recording it. `template_ref` is discover-only. A generated corpus aligns
`response[i]` with `baseline_prompt[i % 48]`; a templated one aligns `corpus[i]`
with `contexts[i]`.

**Fitted pair.** `save_manifold` writes fp32: shared `node_coords`; per layer
`layer_<L>.mean`/`.basis` plus either the affine fields (`node_coords`, optional
`affine_map`) or the curved RBF set (plus the `sigma_*` pair, required on raw
curved fits, absent on SAE-space ones). Missing `node_params` is the affine
marker; one tensor never mixes the two. `ManifoldSidecar` is the one sidecar field
list, validated by exact key set on read and write; its identity fields
(`SIDECAR_DERIVED_FIELDS`) come from the `Manifold`, caller metadata supplies
provenance only, and an unknown key raises. Provenance is all-or-nothing against
`METHOD_FIT_MODES`:

| `method` | `fit_mode` | required block |
| --- | --- | --- |
| `manifold_pca`, `manifold_sae` | authored | |
| `manifold_monopolar[_sae]` | pca | 1-node ray, `subspace_metric = euclidean` |
| `manifold_discover_{pca,spectral,auto}` | its own | auto: `resolved_fit_mode`, `topology_winner`, `topology_candidates` |
| `manifold_discover_sae` | any discover | |
| `manifold_procrustes_transfer` | any | `source_model_id`, `source_model_fingerprint` |
| `merge` | baked | `components`, `bake_policy = additive_union_v1` |

An `sae-*` feature space needs `sae_release` + `sae_ids_by_layer`; a raw fit
carries no SAE field. `node_spread_per_layer` is measured before DLS, so its keys
may exceed `fitted_layers`. The hot path reads the tensor, never the sidecar; the
sidecar proves a cached fit current. `model_source_fingerprint`,
`capture_version`, `capture_render_sha256` and `baseline_prompts_sha256` let
`fit/preflight.py` prove a no-op without weights; only the browser writes
`context_binding_sha256`.

**Integrity and staleness.** `files` maps each top-level fitted file to its
sha256; `manifold.json` and `nodes/` are excluded because a corpus edit is the
normal re-fit trigger. `prove_pair` is the one pair proof. `nodes_sha256()` is the
staleness key: labels, corpus bytes, roles, kinds; domain + coords (authored) or
fit_mode + hyperparams (discover); the resolved template's `sha256()` (the bare
ref if unresolvable, so the fit fails loudly). A baked folder hashes a
provenance-only sentinel. `load_manifold(verify_manifest=True)` refuses an
unproven pair and, for non-baked fits, a `nodes_sha256` mismatch or a
`fit_policy_version` other than `geometry.manifold.MANIFOLD_FIT_POLICY_VERSION`.
A manifest at another version is a hard `ManifoldFormatError`; a sidecar at
another version is a stale cache miss (`tensor_models()` omits it, the next fit
overwrites it) except in a baked folder. `sidecar_is_current` reads unreadable
bytes as current so the strict loader reports the real fault: a format bump
invalidates fits but never launders corruption.

`ManifoldFolder.load` always checks the exact schema and every declared corpus
file (a baked folder instead needs a fitted pair). `verify_manifest=True`
(runtime, install, push) also walks all of `files` and needs a sidecar per
tensor; `False` (discovery, authoring, lifecycle, summaries) proves only what the
caller later consumes. Sidecars load lazily, so a malformed fit hides only itself.

**Locks and publication.** Global order: manifest lock
(`manifold_folder._locked_manifest`, re-entrant), template lock, then pair locks
sorted by tensor path. Pair locks are digest-named in `<ns>/.locks/`, so `rm` or a
swap cannot unlink a held lock inode. `save_manifold` fsyncs both halves and
replaces the sidecar before the tensor (an interrupted first fit leaves an
ignored orphan sidecar, never a bare tensor); the writer then records exactly its
outputs with `update_file_hashes`, so fitting one model never blesses another
model's interrupted pair. Whole-folder writes build in `.staging/<uuid>` and swap
via `replace_manifold_folder` under `locked_manifold_folder`, which first
restores a crash-parked backup of that target.

## Authoring and lifecycle

Authoring refuses `naming.RESERVED_NAMESPACES` (`jlens, slens, sae, neuron,
head`): a folder there would shadow instrument atoms. Every roster-changing call
in this package invalidates the selector index itself.

- `create_manifold_folder` (authored) returns `(folder, advisories)`;
  `authoring_advisories` flags a non-poised layout or an axis with fewer than 3
  distinct values before a fit is paid for. `update_manifold_folder` edits the
  description, or the nodes of an authored folder.
- `create_discover_manifold_folder(force=)` replaces in one swap;
  `create_manifold_from_template` expands values × contexts and stores a
  namespace-qualified `template_ref`.
- `publish_baked_manifold` is the only producer of baked folders (merge output):
  pairs and proofs are built in staging, so there is no partial state.
- `init_discover_manifold_folder` / `append_discover_manifold_node` /
  `plan_discover_generation` are the streaming exception: the skeleton fills in
  place, stays undiscoverable until its last corpus file lands, and keeps
  finished nodes after a crash; the planner resumes or appends labels, and
  `force` swaps in an empty skeleton.
- `merge_discover_manifolds` unions ≥2 discover sources into an unfitted folder;
  authored sources and label collisions raise, mixed modes need `fit_mode=`, and
  hyperparameters never cross modes.

`source` is `local`, `bundled` or `hf://<owner>/<repo>@<sha>`; `manifold_upstream`
is the one tier predicate (any `default/` folder counts as bundled).

- `clear_manifold_tensors(..., model_scope, variant=raw|sae|from|all)` is the
  repair path: it deletes orphaned halves too, drops only the selected `files`
  entries, and bumps `fit_epochs["<safe or *>:<variant>"]` so an in-flight fit
  (`fit/publish.py`) cannot publish over it.
- `remove_manifold_folder` renames into scratch, then deletes; a `default/` or
  bundled folder reports `rematerializes_on_restart`.
- `refresh_manifold` returns `scoped` (model scope given: clear that model),
  `bundled` (re-copy manifest + corpus, keep pairs and proofs), `hf` (re-pull the
  repo's **current default revision**, not the ref first installed), or `skipped`.
- Baked folders refuse clear and scoped refresh (`BakedManifoldError`): the
  tensor is the only copy. Use rm.
- After clear/rm, a `manifold_capture/` digest is deleted only when no surviving
  sidecar anywhere names `(model, capture_sha256)`, checked under the capture lock.
- `transfer_manifold` re-proves the source against the `TransferSourceProof` from
  `preflight_transfer_manifold`, refuses a source fitted for other weights, maps
  through `geometry/transfer.py` (target whitener mandatory), strips source-only
  capture identity and writes the `_from-` pair; an unproven target is
  overwritten, a proven one needs `force`. Map fitting is `fit/transfer.py`.
- `manifold_summary` backs `pack show -j` and `GET /drowse/v1/manifolds/{ns}/{name}`;
  discover and baked folders report `domain {}`, `intrinsic_dim 0`,
  `node_coords []` because their geometry is per-model.

## Selectors

`selectors.parse`: `all`; `default` (= `namespace:default`); `tag:`,
`namespace:`, `model:`; `<ns>/<name>` or `<name>` with an optional `:<variant>`,
validated against `base.paths.VARIANT_SUFFIX_RE` (`raw|sae[-rel]|role[-x]|from[-src]`)
and dropped. In `resolve`, a bare name in two namespaces raises
`AmbiguousSelectorError`; `model:X` matches a fitted tensor for X of any variant.

One memoized walk per manifolds root yields `concepts`, `labels`
(`(ns, manifold, label)`) and `names` (2-node `pca` only). The memo is keyed on a
stat fingerprint (folder listings, `manifold.json` inode + mtime, `nodes/` mtime),
so another process's edits show on the next resolve; in-process writers also call
`invalidate()` for same-tick edits.

`resolve_bare_atom` owns the ladder `steering/grammar.py` calls, in order:

1. **label**, plain slug only (variant `raw`, no typed namespace, no `.`): a
   node-label hit becomes `<ns>/<manifold>%<label>`. A folder scan, so unfitted
   manifolds resolve; every bipolar pole is a node label.
2. **name**, variant `raw`: a 2-node `pca` manifold name (`formal.casual`) maps to
   node 0, the positive pole.
3. **pole**: `canonicalize_atom` peels `:variant` and slugs via
   `naming.canonical_concept_name`.

A collision in tier 1 or 2 raises `AmbiguousSelectorError`; any other failure
propagates rather than falling through to tier 3.

## Templates

`templates/<ns>/<name>/template.json`, `TEMPLATE_FORMAT_VERSION = 2`, exact keys
`format_version, name, slot, values, contexts, description, source, tags`; a
context is exactly `{turns: [{role, content}], assistant}` with roles
`system|user|assistant`. Load enforces: the slot appears **exactly once** in each
`assistant` and **never** in a history turn (history is common-mode across
values, so the value lives only where it is read); the last history turn is
`user`; at least 2 non-blank values, each slugging (`[^a-z0-9]+ → _`, trimmed) to
a unique `LABEL_REGEX` node label.

`node_corpora()` is the manifold corpus, `score_inputs()` the `ChoiceScorer` feed
(split at the slot). `sha256()` covers format version, slot, values and contexts
(not name, description, source, tags) and folds into a referencing manifold's
`nodes_sha256`. `resolve_template` takes `<ns>/<name>` or a bare name
(`AmbiguousTemplateError` on collision, `TemplateNotFoundError` on a miss).
`create_template_folder` validates the whole body before writing.

## Bundled artifacts

`bundled.materialize_bundled_artifacts()` is the single entry point (session, CLI
context, config loading), a no-op after its first call per process per home
(`reset_bundled_state()` for tests): in-process drift cannot be told apart from a
CLI hyperparameter override. Templates materialize **before** manifolds because a
bundled manifold may `template_ref` a bundled template
(`test_bootstrap_entry_point_runs_templates_before_manifolds`).
`bundled_manifold_names()` lists a package folder only when its manifest and
every declared node file exist, so a partial regeneration is never exposed.

A manifold re-copies into `default/<name>` on drift: missing folder, older-format
manifest, changed manifest (`payload_drift_sha256`, canonical JSON with
`LOCAL_STATE_KEYS` stripped), or a missing node file. Corpus edits under
`default/` persist until the bundle changes, then are overwritten with a warning
(fork under `local/` to keep them). Fitted pairs and proofs carry across
(hard-linked into staging) and `nodes_sha256` decides whether they stay current.
`refresh_bundled_manifold` is the unconditional re-copy behind `pack refresh`.
Templates re-copy on hash or format drift and keep a `.json.bak`; none ships yet,
and the path stays because a template-derived manifold is the one artifact that
authors model-free.

Shipped in `drowse/data/manifolds/`, all `source: "bundled"`, unfitted:

- **17 bipolar concepts** (2-node `pca`, `max_dim 1`, node 0 = positive pole),
  independent signed axes tagged `epistemic` (confident.uncertain,
  honest.deceptive, curious.disinterested), `alignment` (refusing.compliant,
  sycophantic.blunt, sincere.manipulative), `register` (formal.casual,
  direct.indirect, verbose.concise, creative.conventional, humorous.serious,
  warm.clinical, technical.accessible), `cultural` (masculine.feminine,
  individualist.collectivist, traditional.progressive, religious.secular). Known
  model-level entanglement, cross-model robust: masculine.feminine ↔
  traditional.progressive (weighted cosine ≈ +0.5–0.6).
- **`personas`**: `auto`, `max_dim 8`, 107 assistant-baselined persona nodes
  (Assistant Axis paper, arXiv 2601.10387).
- **`emotions`**: `auto`, `max_dim 3`, 20 mood nodes spanning the PAD octants.
- **`months`**: `authored` periodic 1-D `BoxDomain` (period 12), nodes at 0…11,
  December wrapping to January; declared because an auto-fit prefers flat by GCV.

`drowse/data/` also holds `baseline_prompts.json` (48 prompts) and
`neutral_statements.json` (a multiple of 48). `scripts/regenerate_bundled.py`
regenerates corpora and the neutral baseline; fitting is a separate
`manifold fit`. `bundled_probes.bootstrap_probe_roster` fits-or-loads the tagged
axes (keyed by bare name) and attaches multi-node defaults only if already fitted
(keyed `default/<name>`; fitting 107 nodes would block startup). A per-manifold
failure is logged, never fatal.

## Distribution

**`.drowse` v1** (`archive`): a stored ZIP with fixed timestamps; `pack.json` =
`{format_version: 1, kind: "drowse-manifold", producer, primary:
"manifolds/<ns>/<name>", template: "templates/<ns>/<name>" | null, source: {uri,
repository, revision}, files: {path: {size, sha256}}}`. It carries one closure
(manifest, corpora, manifested pairs) and its one template. An `hf://` source
names a 40-hex commit, and the `pack.json` source must equal the manifest's.
Extraction validates every entry (canonical relative paths, regular files, the
`DROWSE_ARCHIVE_*` size/count/ratio bounds, duplicate-free JSON) and hashes while
streaming. `validate_closure` demands the exact file set
and, for a templated manifold, a corpus equal to the template's.
`validate_fitted_pair` proves each pair: variant against provenance, identity
against the manifest, `nodes_sha256`, fp32, finite, shapes, positive shares.
Export strips `artifact_id`/`fit_epochs`, narrows `files`, requires a
namespace-qualified `template_ref`, and refuses an output inside the closure.
Checksums prove integrity, not publisher identity.

**Install** (`install.install_manifold`): a `.drowse` path, a local folder (exact
closure only; symlinks and unmanifested tensors fail) or `<owner>/<repo>[@rev]`.
Each source stages a closure; `install_closure` applies one policy. The
destination is `local/<name>` unless `as_` is given (the archive's primary
namespace is ignored; `default/` and reserved namespaces are refused). A rename
rewrites the manifest and sidecar `name`s and re-proves; `source` is stamped
`local` or `hf://<repo>@<sha>`; every pair is validated, then the template is
promoted, then the manifold. A differing installed template needs `force` and no
other dependent; an existing target without `force` raises
`ManifoldInstallConflict` (409).

**Hub** (`hub`): a repo is a Hub *model* repo tagged `drowse-manifold` holding
exactly one root `<name>.drowse` (the only file install reads), a summary
`manifold.json`, and a `README.md` whose frontmatter `drowse:` block carries the
search-row fields; drowse.ai shares the convention. Pull pins the revision to its
commit. Push exports with a variant filter (default `raw`) and an optional model
scope, then uploads in one commit, deleting stale archives and loose layouts.
Search is one `list_models` call with card data, at most 20 rows, skipping repos
that would not install.

**GGUF** (`gguf.export_gguf_manifold`): folds each raw 2-node fit
(`geometry.fold.folded_directions`) to one llama.cpp control vector per model;
multi-rank or curved fits are rejected. Output is a `.gguf` (single model) or a
directory (`<name>.gguf`, or `<name>.<safe>.gguf` per model), default cwd; a
destination inside `$DROWSE_HOME` is refused, so the folder and its manifest are
never touched. `model_hint` comes from `AutoConfig.model_type`
(`trust_remote_code=False`); `--model-hint` is the escape hatch.

## Per-model caches

Both use `base/shards.py`: immutable per-layer generations behind an atomic JSON
pointer, `fit_lock` single-flighting the expensive transaction. Readers never
mutate; the next writer collects old generations once the new pointer is durable.

- **`neutral`** (format 4, capture version 1): `[N, D]` fp32 per layer, the single
  per-model neutral artifact. Probe-centering means are its `X.mean(0)` and the
  whitener is built from it (no `layer_means` cache). fp32 because late gemma-3
  layers exceed the fp16 max (65504) and bf16 loses input precision. Identity
  binds `capture_sha256`, `model_fingerprint`, `model_source_fingerprint` and the
  layer set. `load_neutral_whitener` is the one offline whitener loader
  (`manifold compare`, offline bake): it never loads a model and raises
  `WhitenerError` with a populate hint. Capture on a miss is `fit/neutral.py`.
- **`alignment`** (format 5): `models/<safe_tgt>/alignments/<safe_src>.json` over
  per-layer `left`/`right`/`offset` factor shards. The pointer embeds both models'
  `neutral_cache_identity`, so re-capturing either neutral cache invalidates it.
  `extend=True` writes only new layers; load rejects an identity mismatch before
  reading payloads and skips a damaged layer without dropping the rest.

## Errors and gotchas

`ManifoldFormatError` and `BakedManifoldError` (`ValueError`, default status)
report 400 via `server/errors.py::_drowse_status`; `TemplateFormatError`,
`AmbiguousTemplateError`, `SelectorError` and `DrowseArchiveError` are
`RequestError` (400); `TemplateNotFoundError` is 404; `HFError` 502 (scrubbed
when wrapped). Authoring over an existing manifold or template raises a bare
`FileExistsError`.

- Discovery is metadata-only: prove a pair where it is consumed, published or
  installed.
- A crash between a swap's two renames leaves the target missing until the next
  locked writer of that target restores `.staging/backup/…`; bundled folders
  recover at the next session start, and discovery does not sweep.
- `write_node_corpus` swaps `nodes/` in place; a crash mid-swap leaves a
  corpus-less folder, which discovery skips.
- drowse.ai (`webui/src/hosted/artifacts/drowseArchive.ts`) hand-mirrors the
  sidecar format, `METHOD_FIT_MODES`, `RESERVED_NAMESPACES` and `pack.json` v1, so
  changing any of them crosses that seam. `tests/browser_drowse_archive_bridge.py`
  is the Python end of its interchange test; keep its path, arguments and stdout.
- `hash_manifold_files` blesses every file in a folder; it is for tests only.

## Testing

CPU-only, in `tests/store/`: `test_manifolds_io.py` covers format, authoring,
lifecycle, locks and bundled manifolds; the rest are one file per module
(`test_hf` patches `store.hub._hf_api`). Fit-side folder transactions are tested
in `tests/fit/`.
