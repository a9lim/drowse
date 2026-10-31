# AGENTS.md

## What this is

`drowse` is a Python library and dual-protocol HTTP server for activation
steering and trait monitoring on Hugging Face causal LMs. One `drowse serve`
port carries OpenAI `/v1/*`, Ollama `/api/*`, the native `/drowse/v1/*` API
(REST, SSE, one WebSocket) and the Svelte dashboard at `/`. Two frontends share
one engine: `DrowseSession` (programmatic) and `drowse serve` (HTTP plus the
dashboard).

Steering signal comes from representation engineering, unified under one
artifact family, the **manifold**: labeled nodes placed on a domain and fitted
to a per-layer, neutral-anchored subspace in the Mahalanobis metric. A
difference-of-means steering vector is the 2-node flat case and a monopolar
concept the 1-node case; `personas` is a 107-node flat fan, `emotions` a
20-node affect manifold, `months` an authored periodic ring. Every steering
term (vectors, poles, `~`/`|` projections, `!` ablations, `%` manifold
positions, J-lens and SAE atoms) lowers at generation time to one per-layer
injection whose semantics are `geometry/inject.py::subspace_inject`.
Coefficients are per call; the model is never mutated. Three read families
measure the same forward: geometry probes (the whitened `Monitor`), the
Jacobian lens and SAEs.

Read `ARCHITECTURE.md` before touching the engine: it is the cross-cutting
design. Every package has its own `AGENTS.md`, auto-loaded when you work in
that directory. `README.md` is the user-facing guide (it ships to PyPI).

## Version and releases

`__version__` in `drowse/__init__.py` is the one version; `pyproject.toml`
reads it through `[tool.setuptools.dynamic]`. Version bumps are user-owned:
never bump as part of feature work. A push to `main` that changes the version
makes `.github/workflows/release.yml` tag `v$VERSION`, build, publish the
`drowse.ai` distribution to PyPI by trusted publishing, and cut a GitHub
release. A push without a bump is a no-op.

## Commands

```bash
pip install -e ".[dev]"            # editable + pytest/ruff/pyright/build
# extras: sae (SAELens 6.x), hf (datasets: streamed lens/SAE corpora), gguf,
#         cuda + flash (Linux/NVIDIA), research, notebook

drowse serve [MODEL] [-S EXPR] [-p CATS...] [-H HOST] [-P PORT] [-k KEY] [-C ORIGIN]
             [--no-web] [--no-dls] [--compile] [--cuda-graphs] [--top-k-alts N] [-c PATH] [-s]
drowse manifold extract CONCEPT [CONCEPT] -m MODEL [--kind abstract|concrete|custom] [--system T]
                        [--role SLUG | --sae RELEASE] [--no-dls] [-f]   # ns/ prefix authors outside local/
drowse manifold generate NAME --concepts C... -m MODEL [--kind K] [--system T] [--samples-per-prompt K]
                         [--seed S] [--role-per-node] [--description TEXT] [-f]
drowse manifold from-template TEMPLATE [--name N] [--fit-mode pca|spectral|auto] [--max-dim N]
                              [--var-threshold T] [--description TEXT] [-f]
drowse manifold fit TARGET -m MODEL [--layers L1,L2|workspace|all] [--sae REL] [--method pca|spectral|auto]
                    [--max-dim N] [--min-dim N] [--var-threshold T] [--k-nn K] [--bandwidth S]
                    [--max-subspace-dim R] [--smoothing auto|0|LAMBDA] [--persistence-frac F] [--no-dls] [-f]
drowse manifold bake NAME "0.3 ns/a + 0.5 ns/b" [-m MODEL] [-s] [-f]   # offline, writes local/NAME
drowse manifold merge NAME SRC... [--method M] [--description TEXT] [-f]
drowse manifold transfer NAME --from SRC_MODEL --to TGT_MODEL [-f] [-j]
drowse manifold compare CONCEPT... -m MODEL [-v] [-j]
drowse manifold why CONCEPT -m MODEL [-j]
drowse pack ls [--namespace NS] [-j|-v] | show NAME [-j] | search [QUERY] [-j|-v]
drowse pack install TARGET [-a NS/NAME] [-f]     # ns/name[@rev] on the Hub, a folder, or a .drowse
drowse pack push SEL [-a OWNER/NAME] [-m MODEL] [--variant raw|sae|all] [--private] [--dry-run]
drowse pack rm SEL -y | clear SEL [-m MODEL] [--variant V] | refresh SEL [-m MODEL]
drowse pack export gguf NAME [-m MODEL] [-o PATH] [--model-hint HINT] | export archive NAME [-o FILE]
drowse experiment fan MODEL PROMPT -g CONCEPT=A,B,C [-S EXPR] [--max-tokens N]
drowse experiment transcript run PATH.yaml [MODEL] [--max-tokens N]
drowse experiment naturalness MODEL PROMPT --manifold NAME -S EXPR [--compare-linear]
drowse template create NAME --slot TOKEN --values V... --contexts PATH [--description TEXT] [-f]
drowse template ls | show NAME | rm NAME [-y] | score NAME -m MODEL [-S EXPR] [--by sum|mean]
drowse lens fit MODEL [--kind r|j] [--corpus FILE | --hf-corpus REPO[:CFG][@REV][,K=V...]...] [--prompts N]
                [--seq-len T] [--dim-batch K] [--prompt-batch B] [--checkpoint-every N] [--layers L] [-f]
drowse lens fetch MODEL [neuronpedia|workspace-r|workspace-j] [--repo R] [--revision REV] [-f]
drowse lens ls MODEL | show MODEL [SRC] | use MODEL SRC | rm MODEL [SRC] [-y]
drowse lens top MODEL "PROMPT" [-k K] [--layers L|sample|workspace|all] [--position P]
drowse lens decompose SELECTOR -m MODEL [-k K] [--layers L]
drowse sae train MODEL NAME [--corpus FILE] [--layer L] [--tokens N] [...training knobs]
drowse sae fetch MODEL saelens:RELEASE [--layer L] [--revision REV]
drowse sae ls MODEL | show MODEL [SRC] | use MODEL SRC | rm MODEL [SRC] [-y]
drowse config show [-c PATH...] [--no-default] [-m MODEL] | validate FILE

pytest -q -m "not gpu"             # the CI suite (CPU)
pytest -q -n auto -m "not perf"    # everything this machine can run
pytest -m perf -p no:xdist         # the throughput gates, serially
ruff check . && pyright
python scripts/generate_webui_types.py [--check]          # PYTHONPATH=. in a worktree
cd webui && npm ci && npm run check && npm run build      # Node 24.x; commit drowse/web/dist
python scripts/migrate_lens_v7.py --home DIR [--dry-run]  # pre-6.0 lens homes
python scripts/regenerate_bundled.py [TARGETS...]         # bundled corpora + neutral baseline
```

Model-loading verbs also take `-d/--device` and `-q/--quantize 4bit|8bit`;
most read verbs take `-j/--json`. The root parser has exactly these eight
verbs; a bare verb or group prints its help and exits 0, and `drowse MODEL` is
an argparse error. Precedence is CLI > YAML > runner default; every verb with
`-c` composes `$DROWSE_HOME/config.yaml` first. Flags, config keys, the serve
startup order and the reserved short-flag table are in `drowse/cli/AGENTS.md`.

## Deliberately absent

Rejected designs; do not re-propose them. No Euclidean fallback anywhere in
activation space. No extraction method besides difference-of-means (a 2-node
`pca` fit; PCA@2 ≡ DiM) and no `pca` selector variant. No second injection
kernel, angular mode, `injection_mode`/`theta_max`, lever or per-term gain
knob. No `vector` verb, `--steer-mode`, `--legacy`, `--projection-metric` or
`--ridge-scale`; `--method` exists only on discover fit and merge, and the
only steering knob beyond coefficients is `--no-dls`. No in-place migration
of old persisted formats (the one converter is an explicit script) and no
deprecation shims or second spellings.

## Package map and layer order

`tests/invariants/test_layering.py` enforces this order, lowest first. A
module may import its own layer or any lower one. Every import counts:
module-level, function-level and `if TYPE_CHECKING:`. A lower layer that needs
a higher object types it against a narrow `Protocol` of its own
(`fit.naturalness.NaturalnessHost`, `generation.render.ConversationHistory`)
and receives it by injection. `ALLOWED` is empty. Outside `drowse/session/`,
no code reads a session private (`session._x`, `sess._x`, `<x>.session._x`,
`self._session._x`). `drowse/__init__.py` (the public facade) sits on top.

| Layer | Package | Owns | Doc |
|---|---|---|---|
| 0 | `errors.py`, `naming.py`, `events.py` | the error family; canonical names and `RESERVED_NAMESPACES`; the synchronous `EventBus` | below |
| 1 | `base/` | torch-free primitives: paths and `$DROWSE_HOME`, atomic writes and artifact locks, integrity, shards, staging, Hub access, `SourceFamily`, `ExclusiveGuard`, LRU | below |
| 2 | `geometry/` | pure tensor math: frames, domains, RBF, the kernel, whitener, topology, fold/`Profile`, transfer | `drowse/geometry/AGENTS.md` |
| 3 | `model/` | loading, fingerprints, layer discovery, logits path, layer selectors, chat traits, roles, scene grammar, compile/StaticCache | `drowse/model/AGENTS.md` |
| 4 | `store/` | manifold folders and schema, selectors and the bare-atom ladder, templates, bundled materialization, archive, install, Hub, GGUF export, neutral/alignment caches | `drowse/store/AGENTS.md` |
| 5 | `steering.grammar`, `.terms`, `.triggers` | the expression language and its IR | `drowse/steering/AGENTS.md` |
| 6 | `readout/` | instrument contract, read vocabulary, `Monitor`/`GeometryInstrument`, capture planner and buffers, roster, measurement envelope | `drowse/readout/AGENTS.md` |
| 7 | `lens/`, `sae/` | the Jacobian lens and SAE families: sources, fits, runtimes, instruments | `drowse/lens/AGENTS.md`, `drowse/sae/AGENTS.md` |
| 8 | `steering.composer`, `.hooks` | scopes, resolution, composition, lowering, gains, hooks | `drowse/steering/AGENTS.md` |
| 9 | `generation/` | decode loop and sampler, rendering, prefix cache, compile routing, `ReadPlane`, results, completion scoring | `drowse/generation/AGENTS.md` |
| 10 | `fit/` | the one extraction pipeline, corpora, capture, publish, preflight, neutral compute, bake, transfer, naturalness | `drowse/fit/AGENTS.md` |
| 11 | `loom/` | the conversation tree, recipes and `compose_call`, cast, ops, token replay, diff, filter, transcripts, joint logprobs | `drowse/loom/AGENTS.md` |
| 12 | `session/` | `DrowseSession`, the engine, batch runs, registry, neutral baseline, extraction glue | `drowse/session/AGENTS.md` |
| 13 | `cli/`, `server/`, `web/`, `notebook/` | frontends | `drowse/{cli,server,web}/AGENTS.md` |

Two placements are easy to get wrong. `store` sits below the grammar because
`parse_expr` resolves bare atoms against installed manifolds
(`store.selectors`). `steering` spans two layers: the grammar, IR and triggers
sit low, while the composer and hooks sit above `readout` (gate references)
and `lens`/`sae`; a lower layer that must refuse work under steering takes
`SteeringComposer.assert_unsteered` as an injected callable. A new subpackage
needs a `LAYERS` entry.

Unlisted homes: `errors.py` holds the cross-module errors (subsystem-local ones
live beside their code); `naming.py` holds `canonical_concept_name`,
`BIPOLAR_SEP = "."` and `RESERVED_NAMESPACES` (`jlens`, `slens`, `sae`,
`neuron`, `head`); `base/exclusive.py::ExclusiveGuard` is the one
exclusive-section type; `base/source_registry.py::SourceFamily` is the shared
per-model source shape for lens and SAE. `notebook/` (optional `[notebook]`
extra) holds plotly/pandas helpers over the public result types. `drowse/data/`
holds the bundled manifolds, `baseline_prompts.json` (48 prompts) and
`neutral_statements.json`. `webui/` (repo root, no `AGENTS.md` of its own) is
documented by `drowse/web/AGENTS.md`; its build is committed under
`drowse/web/dist/`. `tests/AGENTS.md` covers the suite.

## Cross-cutting rules

**One authoritative path.** One extraction pipeline
(`fit/pipeline.py::ManifoldExtractionPipeline`), one injection kernel, one
expression grammar (`steering/grammar.py`) for Python, YAML, HTTP, CLI and loom
recipes, one sampler (`generation/decode.py`; the three places that reproduce
it, listed in `drowse/generation/AGENTS.md`, change with it), one error classifier
(`server/errors.py::describe_error`), one layer-selector parser
(`model/layers.py`), one pooling position (`model/tokens.py::last_content_index`),
one measurement envelope (`readout/measurements.py`). Replace or delete the old
path when you add a new one; breaking compatibility is acceptable.

**Mahalanobis only.** Every activation-space metric (fit basis, share, steering
synthesis, `~`/`|`, monitor reads, `manifold compare`, transfer) needs a
`LayerWhitener` covering every touched layer (`covers_all`) and raises
`WhitenerError` otherwise. On real LMs the Euclidean metric is dominated by a
few massive-activation channels, so a fallback would be a wrong answer, not a
degraded one. The deliberate Euclidean exceptions are listed in
`drowse/geometry/AGENTS.md`. Neutral activations are cached fp32: fp16
overflows late layers, and `covers_all` must mean finite factors everywhere.

**Exclusive GPU.** One reentrant `_gen_lock` serializes all model work, because
MPS's single global command buffer is not thread-safe. Collaborators that run
model work take an `ExclusiveGuard`; a busy section raises
`ConcurrentGenerationError`/`ConcurrentExtractionError` (409) rather than
queueing. Artifact-producing work (extraction, fits, corpus generation, lens
fit, SAE train, neutral compute) refuses inside a steering scope
(`SteeringComposer.assert_unsteered`), since the hooks would bake into the
artifact. Details: `drowse/session/AGENTS.md`.

**Errors.** Every drowse exception subclasses `DrowseError` and keeps its
stdlib base; `user_message()` returns `(http_status, text)`. Input a caller can
get wrong raises `RequestError` (400, text verbatim) or a subclass
(`GenerationRequestError`, `SourceSpecError`, `ManifoldSpecError`,
`PoisednessError`, `SteeringExprError`, …) where it is validated; composition
failures after parsing are `SteeringCompositionError` (422). A bare
`ValueError` is a broken invariant: the server logs it and answers a scrubbed
500. The wire `code` is the exception class name. The CLI maps 404 → exit 1,
other 4xx → 2, 5xx → 1. `tests/invariants/test_user_message.py` enforces the
family.

**Persisted formats are exact.** Readers accept only the current version and
never migrate (a derived cache recomputes instead): `manifold.json` v10,
`template.json` v2, `.drowse` v1, loom `tree_format` 2 with token sidecar 2,
transcripts v2, lens format v7, neutral cache v4.
`MANIFOLD_FIT_POLICY_VERSION` (`geometry/manifold.py`) stamps every fit, and
bumping it makes every fitted tensor refuse to load. Every `SamplingConfig`
field is persisted in recipes and required on read, so adding one breaks saved
trees and the browser seam. Pre-6.0 lens homes need
`scripts/migrate_lens_v7.py`.

**The browser seam.** [stdlogits/drowse.ai](https://github.com/stdlogits/drowse.ai)
pins this repo as a submodule and runs `webui/src` with a WebGPU runtime. Its
interface here:

- the `RuntimeClient` contract (`webui/src/lib/runtime/contracts.ts`), the five
  `@runtime-*` aliases (`webui/vite.config.ts` and `tsconfig.json`), and any
  shared webui module it imports by path or whose text its tests read
  (`drowse/web/AGENTS.md`);
- the byte-frozen goldens `tests/fixtures/browser/*`,
  `tests/fixtures/browser_steering_expression.json` and
  `tests/browser_drowse_archive_bridge.py`, at fixed paths, recomputed by
  `tests/browser/test_browser_parity.py` (`tests/AGENTS.md`);
- semantics it reimplements: the grammar, `compose_call`
  (`loom/recipe.py`), the tree filter (`loom/filter.py`), the sampler, the
  measurement envelope, the affine lowering, DLS, the Fisher fit and topology;
- names it mirrors: `RESERVED_NAMESPACES`, `store/manifold_schema.py`
  `METHOD_FIT_MODES`, the manifold sidecar and `.drowse` formats, lens format
  v7 (`sites.out`, `out.layer-L` shards);
- Python it calls: `lens.fit.fit_jacobian_lens` and
  `lens.store.save_lens_at`/`LensProvenance`.

A semantic change to any of these regenerates the golden in the same change
and is an interface change for drowse.ai. A refactor that would need a golden
regenerated is not a refactor.

**Performance invariants.** `tests/gpu/test_smoke.py` gates steered ≥ 85% of
vanilla tok/s (`test_throughput_regression`) and ablation ≥ 80%.

- Hooks write in place, never `.item()` or sync the host, allocate nothing per
  call beyond their own tensors; validation and constants belong in
  compose/lower (`drowse/steering/AGENTS.md`).
- Generation runs under `torch.inference_mode()`; the J-lens fit is the only
  backward pass in drowse.
- Norms, covariances and fit-time math are fp32.
- One device→host transfer per decoded token; per-token scoring runs after the
  forward, never inside a capture hook, and only when a consumer needs it
  (`drowse/readout/AGENTS.md` has the capture-mode table).
- The always-active affine case is an analytic fast path (`AffineOp`); only a
  static pure-push plan rides the compiled graph through the persistent offset
  buffers. Any transient hook routes a compiled session to the eager module on
  a DynamicCache. Curved `%` terms are the slowest path (per-token foot solve,
  no compile); prefer affine terms where coherence allows.
- Top-p runs over `torch.topk` (default pool 1024, top-k applied first), never
  a full-vocab sort.

**MPS discipline.** Fit-time geometry runs on CPU in fp32;
`torch.mps.empty_cache()` between extraction batches; `torch.mps.synchronize()` at the end of every decode, or a
rapid regenerate recommits an in-flight Metal command buffer. Metal reports
queue exhaustion as an asynchronous error that silently zeroes work, so long
MPS loops drain the queue periodically and guard against zero rows (the J-lens
fit loop is the reference pattern).

**Testing.** Most of the suite runs on CPU against a real `DrowseSession` on a
tiny generated checkpoint (`tiny_session`); route tests that need no behavior
use the autospec `tests.support.server.mock_session`. GPU tests (`gpu` marker)
load `HuggingFaceTB/SmolLM2-360M-Instruct` unless `DROWSE_TEST_MODEL` names
another; `gpu_gemma` needs gated Gemma access. Every bug fix lands with a
regression test. A test never touches the real home (an autouse fixture sets a
scratch `DROWSE_HOME`). CI runs ruff, pyright, the types `--check`, the CPU
suite on Python 3.11–3.13, the package-isolation build and the webui build
with a dist-drift check. Details: `tests/AGENTS.md`.

## State and environment

All state lives under `$DROWSE_HOME` (default `~/.drowse`): `config.yaml`,
prompt/neutral overrides, `templates/`, `manifolds/<ns>/<name>/` and
`models/<safe_model_id>/` (neutral activations, alignments, capture cache,
`lens/`, `sae/`). The full tree is in `drowse/store/AGENTS.md`; the lens and
SAE subtrees in their packages. Conversations are never autosaved; they are
browser downloads or explicit `LoomTree.save(path)` targets.

| Variable | Effect |
|---|---|
| `DROWSE_HOME` | relocates all state (there is no `cache_dir=`) |
| `DROWSE_TRUST_REMOTE_CODE` | the only way to let the CLI/server run a model repo's Python |
| `DROWSE_API_KEY` | bearer key when `--api-key` is not given |
| `DROWSE_STRICT_MODEL` | OpenAI/Ollama 404 on a model-name mismatch |
| `DROWSE_MAX_REQUEST_BYTES` | request body cap (default 64 MiB) |
| `DROWSE_MANIFOLD_CAPTURE_CACHE_GB` | fit capture cache bound (default 8) |
| `DROWSE_TEST_MODEL`, `DROWSE_TEST_SAELENS`, `DROWSE_TEST_LIVE_TOKENIZERS` | test knobs (`tests/AGENTS.md`) |
