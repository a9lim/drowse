# drowse/session

`DrowseSession` (`session.py`) is a facade over one loaded model. It owns the
model handle, the exclusive-GPU lock, the loom tree, the event bus and the
generation defaults, builds every collaborator with explicit dependencies, and
delegates each public method to one of them. New behavior goes into a
collaborator with explicit constructor dependencies, not into session state.

Outside this package nothing reads a session private (`session._x`, `sess._x`,
`<x>.session._x`, `self._session._x`); modules inside it may (the engine calls
`_start_loom_assistant`, `_plan_authored_prompt_capture`, `_steering_composer`).
`tests/invariants/test_layering.py` enforces this and pins `session` directly
below `cli`/`server`/`web`/`notebook`. Two collaborators take the session as a
Protocol-typed handle, so keep those properties stable: `Extraction`
(`fit.pipeline.ModelHandle`: `model`, `tokenizer`, `model_id`, `device`,
`layers`, `whitener`, `layer_means`) and `SaeRuntime` (`sae.runtime.SaeModelHandle`).

| Module | Owns |
|---|---|
| `session.py` | wiring, `_model_exclusive`, probe resolution, loom-node stamping, the public API, `close()` |
| `engine.py` | `GenerationEngine`: transaction, `CallSpec` resolution, `TokenTap`, decode routing, the prefix-cache slot, the stream worker, `last_result`; `PrefixCacheStateError` (409) |
| `batch.py` | `BatchRunner` (fan / batch / sweep over `generate_one`), `FastBatchPath` (one batched HF `generate`) |
| `registry.py` | `ArtifactRegistry`: in-memory profiles and loaded manifolds, swaps, probe folds, load validation, analytics snapshots |
| `neutral.py` | `NeutralBaseline`: neutral means + Mahalanobis whitener from one load |
| `extraction.py` | `Extraction`: extract / extract_from_corpora / fit / generate_manifold / bake glue over `fit.pipeline` |
| `authored_capture.py` | measuring authored loom text the first time a prefill re-reads it |

Wired here, documented with their own package: `SteeringComposer`, `ReadPlane`
(`generation/read_plane.py`), `PromptRenderer`, `CompiledRuntime`,
`ChoiceScorer`, `CorpusGenerator`, `TokenReplayer` (`loom/replay.py`),
`JointLogprobReplayer`, `Monitor` / `GeometryInstrument` / `ProbeRoster`,
`LensInstrument`, `SaeInstrument` / `SaeRuntime`.

## Construction

`from_pretrained` = `model.load.load_model` → `CompiledRuntime.prepare`
(persistent offset and capture hooks go on before `torch.compile`; capture only
when probes will attach, `probes is None or bool(probes)`) → `__init__`.
`__init__(model, tokenizer, *, runtime=None, ...)` wraps a pre-loaded model; no
`runtime` means `CompiledRuntime.for_model` (no compilation). No `cache_dir=`:
`$DROWSE_HOME` relocates every path. `trust_remote_code=None` defers to
`DROWSE_TRUST_REMOTE_CODE`; `compile` / `cuda_graphs` / `compile_mode` are
opt-in (`generation/compile_runtime.py`). `on_progress(str)` narrates model
load, the neutral pass and each bundled fit (silent by default).

`__init__` wires chat traits (the tokenizer's EOS set becomes the model's
generation default), layers, `config`, `SteeringManager`, `PromptRenderer`,
`events`, `HiddenCapture`, `_gen_lock`, `tree`, then calls
`materialize_bundled_artifacts()` (always, even with `probes=[]`), then builds
registry → neutral → corpus → composer → extraction → scorer → Monitor → token
replay → instruments → roster → read plane → engine → joint-logprob replayer →
batch runner. Several are wired with lambdas naming attributes built later
(`self._roster`, `self._engine`, `self._lens_instrument`): model work in
`__init__` must come after the wiring block.

`config` is a frozen `GenerationConfig` (`max_new_tokens` from `max_tokens`,
default 1024): rebind it with `dataclasses.replace`; an in-flight generation
keeps its own. A per-call `SamplingConfig` overrides fields for one call only.

The probe roster is built last (`store/bundled_probes.py::bootstrap_probe_roster`,
neutral baseline first). `probes=None` attaches the bundled 2-node axes tagged
in `PROBE_CATEGORIES` (`epistemic`, `alignment`, `register`, `cultural`),
fitting stale ones through `self.fit`, plus every bundled multi-node manifold
*already* fitted for this model (attach-only: fitting `personas` would block
startup for minutes). A list attaches those categories only; `probes=[]`
attaches nothing and skips the construction-time neutral build. A failing
manifold is logged and skipped. `sae.restore_active()` then reloads the
persisted `sae use` selection.

`close()` / `__exit__` removes everything the session attached, including the
permanent offset and capture hooks, so a caller-owned model stays usable, and
drops the prefix cache, instrument residency, registry, roster digests, neutral
baseline and event subscribers.

## The exclusive-GPU section

PyTorch's MPS backend drives every device op through one global command buffer
that is not thread-safe; two threads committing it abort the process. So one
`threading.RLock`, `_gen_lock`, serializes all model work.
`_model_exclusive(busy_msg, *, phase_msg=None, exc=ConcurrentExtractionError)`
is the guard every collaborator receives (`base/exclusive.py::ExclusiveGuard`):
it acquires non-blocking, raises `exc(busy_msg)` when another thread holds the
lock, and with `phase_msg` also raises while a generation is in flight.

| Caller | Acquisition |
|---|---|
| a generation (`GenerationEngine.transaction`) | non-blocking; `ConcurrentGenerationError` (409) |
| extract, fit, `generate_manifold`, bake, `evict_artifact`, completion scoring, token replay, lens fit/readout, SAE train, guarded neutral getters | with `phase_msg`; `ConcurrentExtractionError` (409) |
| probe attach (every family), geometry detach, SAE load/unload | without `phase_msg` (busy check only) |
| steering push/pop (`SteeringScope` enter/exit) | blocking: waits out an in-flight generation |
| `analytics_profile` | non-blocking; returns `None` instead of raising |
| frontends checking for a running generation | `session.gen_lock.acquire(blocking=False)` |

The lock is reentrant so a generation can push its own steering scope while
holding it. Same-thread re-entry from inside a generation (an `on_token`
callback entering `session.steering()` or starting another generation) is
rejected by the engine's `GenState` phase (`IDLE → PREAMBLE → RUNNING →
FINALIZING → IDLE`), not by the lock. Open a steering scope before entering the
exclusive section, not inside it (`ChoiceScorer` and `TokenReplayer` do).

Artifact-producing work refuses inside a steering scope, since the hooks would
bake steering into the artifact: extraction, fits, corpus generation, lens fits
and SAE training call `SteeringComposer.assert_unsteered()`, and
`NeutralBaseline.ensure` refuses to compute while a steering plan is installed
(both `ConcurrentExtractionError`). The server's request serializer
(`app.state.session_lock`, an `asyncio.Lock`) is a separate, server-owned lock.

## Steering scopes

`session.steering(value)` returns an un-entered `SteeringScope`. `value` is an
expression string or a `Steering`; `None` or a dict raises `TypeError`. Every
artifact resolves and loads at the call, so an unknown name raises
(`ProfileNotRegisteredError`) before the `with`. Nesting flattens (inner
entries override outer ones for the inner block); the innermost scope naming a
role wins.

A generation opens its own scope over the call's steering: resolved before the
loom node is created (a bad expression leaves the tree untouched), entered
after it, and popped before finalize, so finalize reads run unsteered.

## Generation (`engine.py`)

`generate_one` = `resolve_call` (frozen `CallSpec`) → `_plan_consumers` /
`_plan_reads` (`ConsumerPlan`, `ReadDemand`, authored-capture plan) → decode
with a `TokenTap` → `finalize` (records `last_result`), inside `transaction()`,
the single-flight guard. Its teardown is exception-hard: pop a leftover scope
(swallowing a teardown error while another failure propagates), run the
session's `_release_generation` (read plane ended, closed and reset, loom
reservation cleared), then set `IDLE` and release the lock unconditionally.

- **Composition** is `loom/recipe.py::compose_call` (rules documented there):
  cast member < call kwargs < regen anchor < regen modifier, with the call
  owning the observation knobs and the seed resolving modifier > call > anchor
  > cast. The engine supplies the tiers: the cast member under
  `sampling.user_role` (user seat) or `sampling.assistant_role`, falling back
  to the seat name (raw mode applies no cast), and for `recipe_override` the
  `tree.anchor_recipe(parent_node_id)` plus the modifier derived from it.
  Composed `steering=""` is unsteered.
- **Same-role continuation** (`append_same_role=True`, default on
  `generate`/`generate_stream`, off for fan/fork/prefill): with `input=None` (or
  `""` raw) and a selected leaf in the same seat and role label, that node is
  regenerated in place with its text as forced prefix, `max_tokens` grown by
  the prefix length, thinking off.
- **Thinking**: explicit `thinking` > the steering's `thinking` >
  `config.thinking` > `chat_traits.supports_thinking`.
- **Widths**: `return_top_k` is per-call when positive, else the session
  default; the shared logit/J-lens/SAE readout width is that or
  `DEFAULT_READOUT_TOP_K = 8`.

Decode routing:

- **Prefix-cache hit** decodes only the suffix. It needs no `return_hidden`, no
  thinking, and no active term that fires in prefill
  (`steering_active_in_prefill`): the cached prefix is unsteered.
- **StaticCache** is eligible when the runtime has it, graph capture is
  thread-safe, and steering is absent or entirely the static single-affine fast
  path. The reserve is `max(call, config) max_new_tokens`, with the total
  bucketed to 256 tokens so most requests share one allocation and one set of
  Dynamo guards. Allocation failure warns and falls back to DynamicCache.
- **The compiled module** runs only when the live hook topology matches the
  traced one: no transient capture hooks, and steering absent, all fast path,
  or lowered to the persistent offsets. Otherwise the eager `_orig_mod` runs on
  a DynamicCache, which avoids a Dynamo retrace per topology.

`TokenTap` is skipped (`ConsumerPlan.need_tap`) when nothing consumes it: its
presence forces a per-step log-softmax and entropy sync. `on_token` is either
the public six-argument callback `(text, is_thinking, token_id, logprob,
top_alts, perplexity)`, which never receives measurements, or an internal
`TokenConsumer` that gets each `TokenEvent` (envelope + `raw_index`) with the
measurements its options request. `generate_stream` builds one (`live_scores=`
/ `live_readouts=`) and runs `generate_one` on a worker thread; closing or
exhausting the stream cancels via `GenerationState.cancellation_scope`, joins,
and re-raises the worker's error when it produced no result.

`cache_prefix(messages | ids | None, ...)` fills the engine's **single** prefix
slot: messages render with `add_generation_prompt=False`, capture is suspended,
and the entry owns its cache (never the reusable generation StaticCache). It
raises `PrefixCacheStateError` inside a steering scope and
`ConcurrentGenerationError` mid-generation. A mismatch misses (full prefill),
never mis-hits. A steering push or pop that leaves prefill steered clears it.

## Multi-row runs (`batch.py`)

- `generate(n>1)` → `BatchRunner.fan`: per-sibling seeds from
  `derive_seed_schedule`. A stateful fan with no parent re-selects the starting
  node before each row, so rows land as siblings under one user turn; that node
  is placement only, never the regen anchor, so `recipe_override` composes as
  at `n=1`.
- `generate_batch` (stateless by default) takes `FastBatchPath` (one left-padded
  HF `generate`) when there are ≥2 stateless rows, the resolved spec passes
  `fast_batch_blocked` (the one eligibility predicate: no thinking,
  logprobs/alternatives, stop strings, logit bias, penalties, seeded sampling,
  hidden or per-layer persistence, and only always-active triggers), an SAE
  probe has a resident layer, and the model has no logits processor the decode
  loop cannot neutralize. Otherwise rows run serially; rows sharing at least
  `PREFIX_CACHE_MIN_TOKENS` (16) rendered tokens under prefill-inactive steering
  prefill that prefix once. That uses the single prefix slot, replacing any
  user-cached prefix, and clears it afterwards.
- `generate_sweep`: the Cartesian product of `{concept: [alphas]}`, each row
  under `base_steering + "α concept" terms` with a derived seed; stateful sweeps
  hang every row under one shared anchor.
- `stop()` ends the whole series: the runner compares
  `GenerationState.stop_count` across rows (every row sets its own stop event on
  the way out); the fast path bridges it through `BatchStopCriteria`.

## Artifacts

**Registry** (`registry.py`). `profiles` / `manifolds` are read-only `Mapping`
views; every mutation is a method and drops the analytics snapshots (the
unsteered prefix cache survives). Lazy loads (`ensure_*`) memoize and fire
nothing. A replace or evict (`adopt_fitted`, `evict_artifact`, or `put_profile`
over a profile a probe was folded from) fires `on_swap({id(old): new | None})`
→ `ProbeRoster.swap` → `GeometryInstrument.refresh_swapped`, which re-attaches
each affected probe on the replacement or detaches it when evicted. Folds handed out
by `fold_probe` are tracked weakly per key, so evicting or refitting an artifact
reaches probes that never held a loaded manifold. Replacements are built before
the first mutation, so a failed fold leaves the registry unchanged.

`ensure_manifold_loaded(key)` admits a tensor only when its sidecar is current
(an older format is a miss), a `from-` variant's provenance matches, `role-<r>`
was fitted uniformly with role `r` (and `raw` with none), `model_fingerprint`
matches the live weights, and (except `baked`) `capture_sha256` matches the
live tokenizer, chat template, role framing and baseline prompts. A miss or
stale fit is `ManifoldNotRegisteredError` (404, refit hint); a bare name fitted
in two namespaces is `AmbiguousSelectorError`. Admitted manifolds are promoted
to fp32 on the model device. `concretize_key` pins a bare `:sae` / `:role` /
`:from` variant to the unique fitted tensor. `ensure_profile_registered(name)`
tries an in-memory profile, then a reserved-namespace atom resolver (only
`jlens` and `sae` are installed; `slens`/`neuron`/`head` raise "reserved and not
steerable yet"), then a fitted 2-node `pca` manifold folded to one direction.

**Neutral baseline** (`neutral.py`). One load of the fp32 neutral-activation
cache yields both the means (`X.mean(0)`) and the whitener. `ensure()` is
unguarded, for callers already holding the lock (composition, generation
preflight, construction); `session.layer_means` / `session.whitener` are the
guarded getters. A failed load leaves the means empty (`layer_means` raises
`DrowseError`); an unbuildable whitener is logged and stays `None`, and every
Mahalanobis consumer then raises `WhitenerError`. A new whitener re-points the
Monitor via `GeometryInstrument.adopt_whitener`.

**Extraction** (`extraction.py`). `extract`, `extract_from_corpora`, `fit`,
`generate_manifold` hold the exclusive section and refuse under steering;
`bake` holds the section but runs no forward
(`fit/bake.py::merge_into_manifold`, fingerprint-guarded).

- Without `namespace`, an installed concept is fitted where it lives, never
  shadowed by a second `local/` copy. `force` on a bundled or hub-installed
  folder needs `namespace='local'` (`pack refresh` would restore its corpus); a
  role mismatch on a reused folder is a `RequestError`.
- `sae` and `role` are exclusive. The returned name carries the variant tail
  (`:sae-<release>` / `:role-<slug>`); steer with that name.
- Author-or-reuse is decided under the folder's manifest lock; corpus
  generation (the slow step) runs outside it and the loop re-decides.
- The vector path emits exactly one `ManifoldExtracted` carrying the folded
  `Profile` (the pipeline's own event is suppressed).
- A fit `adopt_fitted`s every in-memory consumer. A fit that fails after its
  `fit_mode`/`hyperparams` overrides rewrote the manifest evicts the old fit's
  consumers; `generate_manifold` always evicts the folder, even part-way; `bake`
  evicts `local/<name>`, then registers the profile.

`session.evict_artifact(ns, name, *, feature=None)` is what frontends call after
a folder changes on disk: loaded manifolds, folded profiles, tracked probe folds
and analytics snapshots go, geometry probes reading them are detached, and the
next steer or attach loads from disk.

## Reads

`session.instruments` maps `geometry` / `lens` / `sae` to the
`readout.protocol.Instrument` implementations (also `session.geometry` /
`.lens` / `.sae`); the server's `{family}` routes iterate and validate against
it. `ProbeRoster`, `ReadPlane` and `TokenTap` name the three families
explicitly, so a fourth family touches them too.

`add_probe(selector, *, as_name=None, top_n=3)` routes through `ProbeRoster`:
`jlens/…` → lens, `sae/…` → SAE, anything else → geometry. Geometry resolution
(`_resolve_probe_manifold`) tries a registered profile (folded to a 1-node ray),
then a fitted manifold on disk, and as a last resort **extracts** the bare
concept (generates a corpus and fits it under `local/`). Lens/SAE detaches are
safe mid-generation (bound runs keep a frozen roster). A roster change only
invalidates analytics snapshots. Node recipes stamp the Monitor (geometry)
probes and their hashes; lens and SAE probes are not in the receipt.
`analytics_names()` / `analytics_profile()` serve read-side correlation (a
registered vector wins a same-named probe; multi-node and curved probes are
excluded; snapshots are CPU copies). `score_hidden(hidden, *, per_token=False)`
scores attached geometry probes against a `return_hidden=True` capture.

**Authored capture** (`authored_capture.py`). Authored loom text
(`recipe is None`) has no token rows of its own; the first generation whose
prefill renders it measures those positions from the same forward and writes
them via `LoomTree.set_authored_token_scores`. Nothing is tokenized unless the
Monitor or a live lens/SAE consumer will record. Matching walks backward,
bounded before the generation header, and retries whitespace-stripped text
(Gemma templates trim content). The first suffix token after a prefix-cache hit
has no producer row and stays unscored. Rows persist on the first step
callback, before the step sink advances curved feet. Row and envelope shapes
are the loom token-row contract the browser runtime replays.

## Conversation

`tree` is a `LoomTree`. Session methods exist only where they add something:
`append_turn` (scene grammar), `fork_from_token` / `prefill_assistant` (planned
in `loom/ops.py`, run through the engine), `set_cast_member` (validates steering
now via `CastMember.from_parts`), `restore_tree`, `joint_logprobs`. Everything
else is on the tree (`tree.remove_cast_member`, `tree.rewind()`,
`tree.reset()`) or `loom.diff.node_diff(session.tree, a, b)`. `restore_tree`
rebinds `session.tree`, so collaborators read it through a callable
(`tree=lambda: self.tree`) and never cache it; the caller guarantees no
generation is in flight. `joint_logprobs` is uncached; the server caches it in
`joint_logprob_cache`, which the session clears on `LoomMutated` ops `edit` /
`delete` / `reset` / `finalize_assistant` and on `restore_tree`.

## Public API contracts

- `generate` / `generate_batch` / `generate_sweep` return a list-like `RunSet`
  (`node_ids`, `grid`, `kind`, `metrics`, `.first`); `generate_stream` a
  `GenerationStream` (`TokenEvent`s, `.result`). `last_result.applied_steering`
  is the canonical expression (round-trips through `parse_expr`).
- `steering=` takes `str | Steering | None`. `system_prompt` defaults to the
  `INHERIT_SYSTEM_PROMPT` sentinel (explicit `None` clears). `stateless=True`
  never touches the tree. `gen_seat="user"` generates the user seat and
  `input=None` continues the current leaf (rules: `generation/render.py`,
  `loom/ops.py`).
- `extract` / `extract_from_corpora` return `(name, Profile)` (the folded
  view), `fit(folder)` the `Manifold`; `bake` returns `(name, Profile)` and
  registers it (CLI `manifold bake` calls `merge_into_manifold` directly; no
  server route bakes). `registry.put_profile` / `pop_profile` register or drop
  an in-memory vector.
- `events` is a synchronous `EventBus`: subscribers run on the emit thread;
  async consumers hop with `loop.call_soon_threadsafe`.
- `from drowse import X` is the stable path; private submodule paths are not.

## Errors

The error policy (the `DrowseError` family, `user_message()`, `RequestError`
for caller input, a bare `ValueError` as a scrubbed 500, the CLI exit codes) is
the root `AGENTS.md` "Errors" rule; `user_message()` defaults to
`(500, str(self))`. The session's own conflicts are the 409s above
(`ConcurrentGenerationError`, `ConcurrentExtractionError`,
`PrefixCacheStateError`).

## Testing

- `tests/session/` (CPU): engine, batch, registry and loading, authored
  capture, cast, conversation, extraction, bake, gen state, probe resolution.
- Prefer a real session over `DrowseSession.__new__` skeletons: `tiny_session`
  / `make_tiny_session` (`tests/support/sessions.py::TinySessionFactory`, real
  `from_pretrained` on a tiny CPU checkpoint, neutral baseline warmed;
  `.scripted(replies)` for deterministic output). Narrow rigs:
  `tests.support.engine.engine_over`, `tests.support.extraction.make_extraction`.
  Route tests use `tests.support.server.mock_session`, an autospec that fails on
  removed or renamed members.
- GPU: `tests/gpu/test_session.py`, and `tests/gpu/test_smoke.py` (owns the
  throughput regression), on `HuggingFaceTB/SmolLM2-360M-Instruct` unless
  `DROWSE_TEST_MODEL` names another model.
