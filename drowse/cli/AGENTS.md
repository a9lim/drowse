# drowse/cli

The `drowse` console script (`drowse.cli.main:main`; `python -m drowse` is the
same entry point): eight verb groups — `serve`, `manifold`, `pack`, `config`,
`experiment`, `template`, `lens`, `sae` — as thin frontends over `DrowseSession`
and the `store`/`fit`/`lens`/`sae` packages. The CLI is in the top layer beside
`server`, `web` and `notebook` (`tests/invariants/test_layering.py`): it may
import any lower layer and never reaches a session private (`session._x`).

## Module map

| File | Owns |
|---|---|
| `main.py` | `build_parser`, `parse_args`, `main` — the one error boundary |
| `spec.py` | `Arg`/`arg`, `OneOf`, `Verb`; argument types (`positive_int`, `nonnegative_int`, `bounded_int`, `smoothing`); the shared flag specs (the reserved short-flag table); `attach` |
| `context.py` | `CliError`, `exit_code_for`, `user_errors`; `note` / `progress` / `emit_json`; `effective_config`; `open_session`, `resolve_probes`; name resolution; `check_steering`, `parse_layers` |
| `config_file.py` | `ConfigFile` (setup-YAML schema, one validator per key in `_VALIDATORS`), `compose`, `ensure_vectors_installed`, `ConfigFileError` |
| `alpha_grid.py` | `parse_alpha_list` for `experiment fan` grids |
| `verbs/<group>.py` | One `GROUP: Verb` tree per group, its `run_*` runners, and verb-local helpers |

## Parser and dispatch

- A group is a `Verb` with `verbs=`, a leaf a `Verb` with `run=` and `args=`.
  `attach` walks the tree once: a leaf gets `set_defaults(func=run,
  command="<verb path>")`, a group a `func` that prints its own help. A leaf
  without `run` raises at build.
- `Arg` is one `add_argument` call held as data; `Arg.but(**kw)` copies a shared
  spec with keywords replaced (how a verb changes a shared flag's help,
  `required` or `nargs`). `OneOf` is a mutually exclusive group. Nested tuples
  flatten, so `MODEL_LOAD`, `CONFIG` and `INJECTION` are tuples of specs.
- Bare `drowse` or a bare group prints that level's help and exits 0. An unknown
  verb is an argparse error (exit 2). No `argv[0]` peeking, no bare-model
  fallback: `drowse google/gemma-2-2b-it` is an argparse error.
- Building the parser stays torch-free: verb modules import only torch-free
  modules at top level (`drowse.cli.*`, `drowse.errors`, `drowse.lens.key`), and
  every runner imports the engine in its body (`test_help_does_not_import_torch`).

## Errors and output

`main` is the one error boundary. A handled failure prints one stderr line,
`drowse <verb path>: <message>`:

| Escaping exception | Exit |
|---|---|
| `CliError` | its `exit_code`: 2 = usage / bad input (default), 1 = runtime miss |
| other `DrowseError` | `exit_code_for(user_message()[0])`: 404 → 1, other 4xx → 2, 5xx → 1 |
| `FileNotFoundError` / `FileExistsError` | 1 |
| `KeyboardInterrupt` | 130 |
| anything else | traceback — a bug; raise a typed error |

The server maps the same `user_message()` status, so CLI and HTTP agree on what
is bad input: every 400-status error (the `RequestError` family, `WhitenerError`)
exits 2.

`user_errors(*types, exit_code=1)` turns a backend's bare stdlib exception into
a `CliError`. `RequestError` subclasses `ValueError`, so a
`user_errors(ValueError)` block also re-codes typed request errors (`extract`,
`fit`, `merge` and the exports exit 1 for them): type the backend's error rather
than widening a wrap.

stdout carries only the verb's result. Narration (`note`), progress
(`progress`, the indented `on_progress` sink for every long operation) and
warnings go to stderr; `-j` prints exactly one JSON document (`emit_json`).

## Config loading

`-c/--config PATH` (repeatable) and `-s/--strict` ride `serve` and the three
`experiment` verbs. `effective_config(args, *, default_max_tokens=1024,
yaml_vectors=False)` composes `$DROWSE_HOME/config.yaml` (if present) then each
`-c` file, field by field, later wins (`vectors` is replaced, never
concatenated). Precedence is CLI > YAML > runner default:

| Field | Rule |
|---|---|
| `model` | fills only an omitted positional (`serve`, `transcript run`); `transcript run` then falls back to the transcript's `model_id` |
| `max_tokens` | `--max-tokens` defaults to `None` so unset is visible; floors: `serve` 1024 (no flag), `fan` and `transcript run` 256, `naturalness` 128; elsewhere `open_session` uses 1024 |
| `compile`, `cuda_graphs` | YAML `true` switches on; YAML cannot switch off a CLI flag |
| `return_top_k` | fills `--top-k-alts` when unset |
| `temperature`, `top_p`, `thinking`, `system_prompt` | YAML-only session defaults (`system_prompt` via `from_pretrained`, the rest rebind `session.config`) |
| `vectors` | **serve only** (`yaml_vectors=True`) fills an unset `-S`; research verbs use exactly their own `-S`, so a serve default never shifts a sweep |

Then `ensure_vectors_installed(steer, strict=args.strict)` walks
`steering.grammar.referenced_selectors`: reserved readout namespaces
(`naming.RESERVED_NAMESPACES`) are skipped, a bare reference counts as resolved
when a manifold label or name tier matches (an ambiguity is a match), a missing
HF `ns/name` installs, a missing `local/` or `default/` one cannot. Misses warn;
`-s` raises `ConfigFileError` (exit 2).

`ConfigFile.load` rejects unknown keys (a new key needs a dataclass field and a
`_VALIDATORS` entry; an assert keeps them equal), refuses non-boolean booleans
(a quoted `"false"` would flip an opt-in), and parses `vectors` at load. `config
validate FILE` adds `ensure_vectors_installed(strict=True, dry_run=True)`
(nothing installs) and exits 2 on any failure.

## Sessions

`open_session(args, *, roster=False)` is the one construction path: narrated on
stderr, built from `device`, `quantize`, `system_prompt`, `max_tokens`, `dls =
not --no-dls`, `compile`, `cuda_graphs`, `return_top_k`, closed on exit.

- `roster=True` only for verbs that read probe readings: `serve`, `experiment
  fan`, `experiment transcript run`. `-p`: unset or `all` → `None` (tagged
  concept axes plus already-fitted bundled multi-node manifolds), `none` or
  empty → `[]`, categories verbatim. Everything else constructs with
  `probes=[]`, which also skips the construction-time neutral capture and fits.
- `-d/-q` (`MODEL_LOAD`) exist exactly on the verbs that load weights: `serve`,
  `manifold extract|generate|fit`, `template score`, `lens fit|top|decompose`,
  `sae train`, every `experiment` verb. `manifold transfer` has neither but loads
  a model (default device) when it must capture a missing neutral cache.

## Fail before the load

Everything checkable without weights runs before `open_session`: serve's bind
check (non-loopback `--host` needs `-k/--api-key` or `$DROWSE_API_KEY`) and
model presence; every `-S` parse; fan's grid; `fit.naturalness.preflight` (not
`baked`, ≥ 3 nodes, `--compare-linear` needs one `%` term); sae train's numeric
ranges; extract's arity and `--kind custom` needing `--system`.

**Exact no-op proofs skip the load.** `manifold extract|fit` run
`fit.preflight.preflight_manifold_fit_noop`, `lens fit` runs
`lens.lifecycle.prove_lens_fit_noop` before streaming its corpus; a proof prints
the cache-hit line and exits 0, and `-f` skips it. Never short-circuit on file
existence or a model id: a Hub repo is mutable, and the fit identity also covers
corpus, tokenizer render, role and SAE transform. A discover override is never
proven (it rewrites `manifold.json` inside the fit lock), and `extract` tries
the proof only when every existing node carries the requested `--role`.
`manifold generate` plans first (`plan_discover_generation`): a complete folder
costs no load, a partial one regenerates only its missing nodes.

Gap: the per-mode hyperparameter whitelist
(`store/manifold_schema.py::HYPERPARAMS_BY_MODE`) is enforced inside the fit
transaction, so `manifold fit X --method pca --k-nn 5` fails only after the
model loads.

## Name resolution

| Helper | Callers | Semantics |
|---|---|---|
| `split_ref` | every reference, first | `NAME` or `NS/NAME`; an empty side, `.`/`..` or another separator exits 2 |
| `resolve_manifold`, `existing_folder` | `merge` sources, `transfer`, `pack show`, `push`, `rm`, `clear`, `refresh`, `export` | `ns/name` pins verbatim; a bare name walks every namespace; miss exits 1, collision `AmbiguousSelectorError` (2) |
| `folder_or_manifold` | `manifold fit` | an existing directory is a folder (must hold `manifold.json`), else a reference |
| `split_new` | `extract`, `generate`, `from-template`, `merge` target, `template create` | authoring: bare → `local/` |
| `existing_template` | `template show`, `score`, `rm`; `from-template` | `store.templates.resolve_template`, cross-namespace |

Resolution runs `store.bundled.materialize_bundled_artifacts` first: CLI verbs
resolve folders before any session exists, and a first `drowse manifold fit
default/personas` would otherwise miss the bundled folder.

Gotcha: `split_new` always hands `session.extract` a namespace, so the CLI never
takes `resolve_extraction_target`'s fit-in-place path. A bare `manifold extract
formal.casual` authors `local/formal.casual` beside `default/formal.casual`,
making the bare name ambiguous; qualify it to fit the installed folder.
`manifold bake` always writes `local/<name>`.

`compare` / `why` resolve through `_fold`: `[ns/]name[:variant]` → that
model's fitted tensor file. Variants need their full form (`:raw`,
`:role-<slug>`, `:sae-<release>`, `:from-<src>`), role and transfer variants are
checked against the sidecar, and only a 2-node affine fit folds.

## `serve` startup

Bind check → `effective_config(yaml_vectors=True)` → model check → parse `-S` →
`open_session(roster=True)` → `enable_live_lens` → `preflight_steering` →
`create_app` → `warm_up` (`--compile` only) → `enable_live_sae` (not with
`--no-web`) → `uvicorn.run`. Lens, SAE and warm-up failures are narrated and
never block startup.

- The live lens precedes the preflight because `jlens/<word>` atoms resolve
  against the selected lens source. `preflight_steering` enters and leaves
  `session.steering(expr)` once — resolving and loading every atom as a request
  would, authoring nothing; a missing concept exits 1 with an `extract` hint.
- Live lens: a selected output-side source (`lens use`) is never rewritten
  (incompatible → live off with a hint). With none selected, serve tries prepared
  output-side sources in `lens.sources.lens_source_preference_key` order and the
  first compatible one stays selected in `active.json` (the lens runtime has no
  in-memory selection); if none fits, the trial selection is cleared.
- Live SAE: the active source, then other prepared ones (local first), then
  `best_sae_release` (official `google/` repo, Neuronpedia-labelled, `canonical`,
  curated base over its `-all` sibling, name). `--no-web` skips this discovery
  and any download; an explicit `sae use` selection still restores at session
  construction.

## Reserved short flags

One letter, one long form, tree-wide (`test_short_flags_keep_one_meaning_tree_wide`):

| Flag | Long | Flag | Long | Flag | Long |
|---|---|---|---|---|---|
| `-m` | `--model` | `-c` | `--config` | `-f` | `--force` |
| `-d` | `--device` | `-s` | `--strict` | `-j` | `--json` |
| `-q` | `--quantize` | `-S` | `--steer` | `-v` | `--verbose` |
| `-p` | `--probes` | `-g` | `--grid` | `-y` | `--yes` |
| `-H` | `--host` | `-a` | `--as` | `-o` | `--output` |
| `-P` | `--port` | `-C` | `--cors` | `-k` | below |

- `-k` is the one double-booked letter: `--api-key` on `serve`, `--top-k` on
  `lens top|decompose`; they never share a parser.
- `-s` is always `--strict`, strict about something per verb: unresolvable
  steering references (config block), identical layer coverage (`manifold
  bake`), probe-roster drift (`transcript run`).
- `pack push` keeps `--private` and `--dry-run` long-only: it is the one verb
  that publishes, and `-p`/`-d` mean `--probes`/`--device` everywhere else.
- Reuse `spec.py`'s shared specs instead of redeclaring a flag.

## Verb notes

`drowse <verb> -h` lists every flag and default; these are the semantics it
does not carry.

**manifold**
- `extract CONCEPT [CONCEPT]`: one argument is monopolar or a dotted `pos.neg`,
  two are poles; an `ns/` prefix on the first picks the namespace. `--sae` and
  `--role` are exclusive; `--role` bakes into the corpora and reports a
  `:role-<slug>` name. `-f` re-authors the corpora and refits.
- `generate NAME --concepts C...` (≥ 2) writes a `pca` discover folder; `--seed`
  makes each node's corpus reproducible from S and its node index;
  `--role-per-node` uses each slug as its node's role substitution. Fitting is a
  separate step so a flaky generation leaves inspectable corpora.
- `fit TARGET`: `--layers` takes indices, `workspace` (the 40–90% depth band)
  or `all`, never `sample`. Overrides apply only to discover folders (authored
  exits 2); only the supplied patch reaches `session.fit`, which merges it under
  the fit lock — the CLI never writes `manifold.json`. `-f` bypasses the tensor
  cache, needed because `fit` does not re-author corpora: an unchanged corpus
  always cache-hits, even after a code-level fit change.
- `bake NAME EXPR` accepts only namespace-qualified additive scalar terms; `-m`
  limits it to one model, else one tensor per model every component shares.
- `merge NAME SOURCE...` (≥ 2) unions discover corpora into an unfitted folder;
  `--method` is required when the sources' fit modes disagree.
- `transfer` maps a fit through a per-layer Procrustes alignment
  (`fit.transfer.load_or_fit_transfer_alignment`) into the target's
  `:from-<src>` tensor, re-baked in the target's Mahalanobis metric.
- `compare REF... -m` reads cached neutral activations only (missing →
  `WhitenerError`, exit 2): one reference ranks every other foldable fit, two
  are pairwise, three or more print a matrix; `tag:` / `namespace:` / `all`
  expand and keep a `:variant` suffix. `why` histograms the folded direction's
  per-layer norm in 16 buckets.

**pack** — pure IO; `install`, `search`, `push` and an `hf` refresh reach the
Hub. `rm` needs `-y` in every namespace, `default/` included: a `local/` fit
cost real time, so the guard is uniform (likewise `template rm`, `lens rm`, `sae
rm`; nothing prompts). `push --variant` defaults to `raw` (SAE variants are
opt-in), `clear --variant` to `all`. `refresh -m` only drops that model's
tensors; otherwise `local` is skipped, bundled re-materializes, `hf` re-pulls.
`export gguf` folds a fitted 2-node `pca` manifold to a control vector.

**template** — `create --contexts FILE` takes a JSON list of `{turns,
assistant}` or the single-turn `{user, assistant}` sugar (`_canonical_context`).
`score` is the only template verb that loads a model, and has no `-c`.

**lens** — corpus specs, sources and estimators are in `drowse/lens/AGENTS.md`.
- `fit MODEL`: `--kind r` (default, RelP R-lens, `local:r`) or `--kind j`
  (standard, `local:j`); `--corpus FILE` (one document per line, JSON lines
  contribute `text`) or repeatable `--hf-corpus`, else FineWeb-Edu
  `sample-10BT` (needs `drowse.ai[hf]`). A Hub corpus is pinned once and the
  pinned spec serves both the no-op proof and the stream. Unset numeric flags take
  the `lens.jlens` defaults. A fit never backprops below its shallowest source
  layer, so a `--layers` band is the one wall-time lever; `sample` is rejected
  because it includes layer 0 and saves nothing.
  `-f` restarts from zero, otherwise a matching partial fit resumes.
- `fetch` never loads the model: the payload stays in the Hugging Face cache and
  Drowse writes, and activates, a commit-pinned binding. `show` / `rm` default to
  the active output-side source; removing a binding never touches the cache.
- `top` reads a raw prompt (no chat template); `decompose SELECTOR -m` takes the
  model as a flag, unlike the rest of the group.

**sae** — `train`'s `--layer` defaults to the layer nearest 65% depth within the
40–90% band (`sae.backend.select_runtime_layer`); `--width` overrides
`--expansion`. `fetch` has no `-d/-q`: it reads only the model config and the
release (`sae.backend.resolve_sae_backend`), and an uncovered `--layer` exits 2.

**experiment** — `fan` alphas are a comma list, `start:stop:step` or
`linspace(a, b, n)`, one loom sibling per grid row with `-S` composed under each.
`transcript run` prints `Transcript.replay` fail-fast (the library's
`isolate=True` is not exposed). `naturalness` prints `fit.naturalness.evaluate`
and runs without a probe roster.

## Deliberately absent

Each belongs to a design the engine does not have; do not propose them.

- No `vector`, `subspace`, `discover` or `tui` verbs; no top-level `transcript`;
  no `manifold template` (authoring is `manifold from-template`).
- No `--steer-mode`, `--theta-max`, `--legacy`, `--injection-mode`,
  `--projection-metric`, `--metric` or `--ridge-scale`; no gain, lever or origin
  flags. One injection kernel, Mahalanobis-only projections and `compare`, no
  Euclidean fallback; the only steering knob beyond coefficients is `--no-dls`.
- `--method` only on discover fitting and merging. Extraction has no method knob
  (difference-of-means, a 2-node `pca` fit, is the only one).
- One spelling each: a namespace is an `ns/` prefix (the only `--namespace` is
  `pack ls`'s filter), the lens estimator is `--kind`, steering is `-S/--steer`.
- No interactive prompts, no `-j` on `rm` / `use`, no `-d/-q` on verbs that
  never load weights, no serve-side probe-attach step.

## Testing

`tests/cli/` is CPU-only and model-free. `test_cli_verbs.py` pins the parser:
eight groups, `-h` on every parser, the short-flag table, bare groups exit 0,
rejected argv for absent surfaces, parsed namespaces, torch-free help — a new
verb or flag usually needs a case there. `test_cli_flags.py` covers runners over
`tests.support.cli` (`FakeSession`, `install_fake_session`, `poison_session` to
assert nothing loads): the error boundary, config precedence, `-j` purity, serve
policies, fail-before-load and no-op proofs. `test_cli_manifold.py` (lifecycle
over mocked io), `test_cli_manifold_fold.py` (compare / why on a synthesized
fit), `test_config_file.py` and `test_alpha_grid.py` cover the rest.
