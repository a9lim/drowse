# drowse/generation

The decode loop and what one generation reads or produces: sampler, prompt
rendering, prefix KV cache, compile/StaticCache routing, the read plane, token
callbacks and payloads, results, finalization, and completion scoring. The
orchestration (`GenerationEngine`, `TokenTap`, `CallSpec` resolution, decode
routing, the transaction) is `session/engine.py`; the multi-row runner is
`session/batch.py` (see `drowse/session/AGENTS.md`).

Layer: above `readout`, `lens`, `sae`, `steering.composer`/`hooks`; below `fit`,
`loom`, `session`. `tests/invariants/test_layering.py` counts `TYPE_CHECKING`
imports too, so collaborators here type against a narrow `Protocol`
(`render.ConversationHistory`) or a lower-layer class.

| Module | Owns |
|---|---|
| `decode.py` | `generate_steered`, `GenerationConfig`, `GenerationState`, the sampler (`_sampler_candidates`, `_effective_topk`, `_PenaltyState`), `build_chat_input`, `hf_generation_config` / `hf_unneutralizable_processors` |
| `sampling.py` | `SamplingConfig` (frozen per-call knobs, `merged_with`), `INHERIT_SYSTEM_PROMPT` |
| `types.py` | `GenState`, `CallSpec`, `GenerationTransaction`, `GenerationStream` |
| `render.py` | `PromptRenderer`, `resolve_model_type`, `seat_stop_strings` |
| `prefix_cache.py` | `PrefixCache` (one slot: build / hit / release), `PrefixCacheEntry`, `PREFIX_CACHE_MIN_TOKENS = 16` |
| `compile_runtime.py` | `CompiledRuntime`: compile, persistent hooks, StaticCache / CUDA-graph routing |
| `read_plane.py` | `ReadPlane`: capture apply, instrument transaction, step sinks, gate callback, pooled slice, finalize reads, `score_hidden` |
| `token_callback.py` | `TokenCallback` (public, 6 args), `StepTokenCallback` (+ `step_id`), `TokenConsumer` / `TokenConsumerOptions` |
| `token_payloads.py` | `TokenProbePayload`: per-family reading slots → the `scope="token"` envelope |
| `finalize.py` | `FinalReads`, `finalize_generation`, `MIN_ELAPSED_FOR_RATE = 0.1` |
| `results.py` | `GenerationResult`, `RunSet`, `TokenEvent`, `TokenAlt`, `ProbeReadings`, `ResultCollector` |
| `scoring.py` | `ChoiceScorer`, `ChoiceScores` / `ChoiceScore`, `shared_prefix_len` |

## Decode loop (`decode.py::generate_steered`)

Runs on a worker thread under `torch.inference_mode()`. Each iteration: stop
check → `step_id = len(generated_ids)` → write `TriggerContext` (`is_prefill`,
`thinking`, `gen_step`) → forward → `step_callback(step_id)` →
`score_callback(step_id)` into `trigger_ctx.probe_scores` → sample →
thinking/stop bookkeeping → `on_token(..., step_id)`.

- **One step identity.** Sink, gate callback and tap receive the same `step_id`;
  instrument runs key per-forward stashes on it, so one forward's gate and
  display reads pair without rescoring. A buffered UTF-8 flush carries the
  flushing forward's id.
- **Gates lag one forward:** scalars scored after forward `k` steer `k+1`; the
  prefill forward sees none.
- **Scoring is post-forward.** Firing the host read from the capture hook at the
  deepest probe layer would drain the device queue before the remaining layers
  and LM head are enqueued; after the forward the captures are equally fresh.
- **One device→host transfer per token**, unavoidable (EOS, thinking and stop
  logic need the id on host). Entropy, chosen logprob and top-K ride the same
  packed transfer (int64 ids + fp32 bits viewed as int32); with none requested it
  is one `.item()`. `want_perplexity` saves compute, not a sync.

### Sampler order

1. `nan_to_num_(nan=0, posinf=100, neginf=-100)`, **only under steering** (steered
   states can leave fp16 range). Finite logits are never clipped: clipping a large
   common offset flattens valid distributions (Pythia).
2. Presence/frequency penalties on raw logits, before temperature (OpenAI
   semantics); `_PenaltyState` counts emitted completion tokens, thinking
   included, suppressed delimiters not.
3. `logit_bias`.
4. `temperature <= 0`: argmax, as a one-token pool with p = 1.
5. `logits / T`, then `topk(k)`, `k = _effective_topk`: the user's `top_k` as a
   hard cap, else **1024** (nucleus sampling is insensitive past it), capped at
   vocab. Top-k caps the pool before top-p (llama.cpp/Ollama order); never a
   full-vocab sort.
6. Softmax; zero entries whose preceding cumulative mass is `>= top_p`; floor
   top-1; renormalize; `multinomial`.

The pool is **fixed width** with the excluded tail zeroed in place; a mask select
would make the shape data-dependent and add a second sync per token. Consumers
treat zero-probability entries as excluded. Logprobs, `TokenAlt`s and
perplexity (`exp` of entropy) describe this **sampler** distribution; the
completion scorer is the raw-distribution read.

**Three other places reproduce this sampler; change them together.**
`hf_generation_config` is the explicit HF `GenerationConfig` for the
`session/batch.py` fast path: processors pinned neutral, `top_k =
_effective_topk`, greedy as `do_sample=False`, `remove_invalid_values` only under
steering. HF fills a `None` field of an explicit config from the checkpoint, so
`hf_unneutralizable_processors` lists checkpoint processors that cannot be
neutralized (`min_p`, `suppress_tokens`, …), and any present sends the batch to
the serial loop. `loom/joint_logprobs.py` calls `_sampler_candidates`,
`_effective_topk` and `_PenaltyState` but re-implements the penalty/bias step
(`_LogitAdjust`). The drowse.ai browser runtime is pinned by
`tests/fixtures/browser/sampling-parity-v1.json`.

### Forks, thinking, stops

- `forced_prefix` overrides the fed-back token for the first `N` steps while the
  multinomial draw still runs, so re-seeding keeps the RNG stream bit-identical
  through the fork. Pass the exact raw ids, delimiters included. A forced id
  outside the pool reports logprob `None`, never a raw-model probability.
- `seed` seeds the global torch RNG. MPS is not bitwise deterministic.
- The thinking machine (`IDLE`/`PREAMBLE`/`THINKING`/`RESPONSE_PREAMBLE`) runs off
  `ChatTraits.think` even with `thinking=False`: channel formats (gpt-oss) emit
  their analysis channel regardless. With `thinking=False`, `starts_in_thinking`
  is forced off only for templates with an `enable_thinking` switch;
  forced-thinking families think regardless. For channel formats EOS inside
  thinking is a separator, not a stop. Any exit mid-thought sets
  `thinking_end_idx = len(generated_ids)`, so the response decodes empty.
- Stop sequences match non-thinking text only, against a tail of
  `max(len(s)) - 1` chars plus the current emit. A hit emits the pre-stop part and
  sets `state.response_text` (exact visible text) and `response_aggregate_index`.
  **Matching runs only when `on_token` is set**; the engine forces a tap for any
  stop list, and a direct caller must too.
- A token whose lone decode contains U+FFFD is buffered and flushed as a group
  (`token_id = -1`, no logprob). Text comes from the lazy `ChatTraits.token_text`
  cache, never a full-vocab table.
- `finish_reason`: `"stop"` (EOS or external stop), `"length"`, `"stop_sequence"`.

### KV paths

- DynamicCache: `cache_position` is omitted, not `None`; HF derives positions
  from the cache length and an explicit `None` breaks that.
- StaticCache: the caller passes a preallocated cache. `cache_position` is an
  arange on prefill, then one 1-element buffer advanced by `fill_` from a
  Python-int counter (stable address for graph replay, no `.item()`).
- Prefix hit: `cache_position_offset` = prefix length; `input_ids` is the suffix.
- A model returning no `past_key_values` on prefill drops to O(N²) re-feeding
  through a preallocated `prompt + max_new_tokens` buffer, with one warning.
- `logits_to_keep=1` avoids the `[1, T, V]` prefill logits; a runtime
  `TypeError` retries once without it.
- `finally: torch.mps.synchronize()`, or a rapid regenerate can recommit an
  in-flight Metal command buffer.

`GenerationState`: the stop event plus a thread-local `cancellation_scope`
(closing one stream cannot stop a later one), `emit_map` (`(generated_ids index,
is_thinking)` per emission; `TokenEvent.raw_index` reads it), `thinking_end_idx`,
`finish_reason`, `response_text`, `response_aggregate_index`. `reset()` keeps
`stop_count`, which the batch runner compares across rows to tell a user stop from
a row finishing.

`GenerationConfig` (frozen; `max_new_tokens=1024`, `temperature=1.0`,
`top_p=0.9`, `top_k=None`) is the session default, changed by rebinding
`session.config = replace(...)`. A per-call `SamplingConfig` composes into a local
copy. `merged_with` lets the other side's **non-default** fields win, so a merge
cannot reset a field to its default.

## Rendering (`render.py`, `build_chat_input`)

`PromptRenderer.prepare_input` is the one place a conversation becomes
`input_ids`. `input` is a new user turn (`str`, labelled `user_role`), `None` (no
new turn) or a message list; anything else is `GenerationRequestError` (400).
`raw=True` with `str`/`None` input renders `tree.flat_text(...) + input` with no
template; the caller chooses `raw`, nothing forces it for base models.
`INHERIT_SYSTEM_PROMPT` reads the live session default; `None` means none.

- Generation label: on the assistant seat a role-augmented steering scope's role
  wins (steer baseline = extract baseline), with `RoleBaselineMismatchWarning`
  when the send's `assistant_role` differs; the user seat uses `user_role`.
- `scene_grammar` runs the template autopsy and round-trip validation
  (`model/scene.py`) lazily, once; `None` means unavailable. `build_chat_input`'s
  routing (stitcher, `apply_chat_template`, per-turn role splice,
  `render_scene_raw`) is specified with the scene grammar in
  `drowse/model/AGENTS.md`.
- `build_chat_input(traits=...)` reads the session's bounded render cache
  (`ChatTraits`); the key, built here, covers messages, labels, thinking, seat,
  scene, model type, the live template, special tokens and added-token count.
- `seat_stop_strings` adds a seat's close as a stop string only when it differs
  from the assistant's (gpt-oss `<|end|>` vs `<|return|>`); otherwise the EOS
  union stops it and the no-stop-list fast path survives.

## Prefix cache (`prefix_cache.py`)

The one opt-in prompt-prefix KV slot. `build` prefills with no hooks of its own;
the engine's `cache_prefix` runs it outside any steering scope with capture
suspended, so the KV is a pure function of the tokens (when a generation may
reuse it: `drowse/session/AGENTS.md`). A hit needs an exact token prefix **and** a
non-empty suffix (the first sample reads the suffix's last logit); a static entry
also needs a StaticCache-eligible generation that fits `max_cache_len`. A
mismatch misses, never mis-hits. A failed StaticCache build or prefill falls back
to a DynamicCache entry.

`release(hit)` must follow every reuse; a failed rollback drops the entry, and an
unreleased dynamic entry keeps the decoded tail. Rollback by cache-layer kind:
full attention `crop(-n)`; sliding-window and conv states through upstream
`activate_past_recording`, armed **after** the prefill (recording a long prefill
is the memory spike), then `crop`; recurrent states snapshotted and restored in
place; StaticCache by resetting each layer's `cumulative_length` (a sliding layer
that wrapped cannot rewind). A static entry never records past states, which would
reassign buffers a compiled graph captured. No transformers class is patched.

## Compile routing (`compile_runtime.py`)

`CompiledRuntime.prepare` compiles only on CUDA/MPS. It installs the persistent
offset hooks and (unless built with `probes=[]`) persistent last-token capture
hooks **before** `torch.compile`, so they are traced into the graph; later hooks
are not retraced. When `model/compile.py::compile_with_probe` falls back to eager
the persistent hooks are removed (pure overhead without a graph).
`compile_mode=None` picks `"reduce-overhead"` only when CUDA graphs are
supported, else `"default"` (MPS: fusion, no graph capture). `for_model` routes a
model handed in directly.

- `static_cache_active` iff (compiled or CUDA graphs active) and the
  architecture's StaticCache constructs.
- `graph_thread_safe()`: CUDA graph trees are thread-local, warmed on the
  constructing thread; with graphs active, other threads (stream and server
  workers) run eager.
- `take_offset_hooks()` hands the offset buffers to `SteeringManager` for good.
- `acquire_static_cache` returns one identity-stable cache, reset in place and
  regrown only on demand; Dynamo guards key on the K/V tensors, so a fresh cache
  per request recompiles.
- `compiled_clean_capture(device, want_hidden=)`: the persistent capture buffers
  are usable (`return_hidden` keeps transient full-retention capture).

Which generation takes StaticCache or the compiled module is
`GenerationEngine.decode` (`drowse/session/AGENTS.md`); in short, curved, gated
and phased steering never compiles.

## Read plane (`read_plane.py`)

`ReadPlane` applies one generation's read side over explicit dependencies
(capture, Monitor, the three instruments, decoder layers, tokenizer, composer
`gate_refs`). The instrument transaction rules, the gate channel validation and
the capture-mode table (including each mode's finalize read) are in
`drowse/readout/AGENTS.md`; this module is their executor.

- `begin` runs `bind_runs`, plans, attaches capture, arms the ring and step sink,
  and resets or disables curved warm-start. Sinks append one row per forward,
  empty rows included, so row `k` is forward `k`. `step_callback()` is
  `ingest_persistent` on the persistent route, else `fire_step_sink`.
- `geometry_step_readings` reads the sink's row and never rescores (lean rows are
  deliberately coords-only); without a row it observes the latest slices through
  the bound geometry run.
- `gate_score_callback` binds the gate keys once and scores only keys naming
  attached probes: from the subset row, from the incremental row (backfilling
  exact keys the row omits), or from the latest slices; lens/SAE keys through
  their bound runs, which stash the row for the display step. An empty capture
  scores nothing, so gates read inactive, never stale.
- `aggregate_index` / `pooled_slice` is the one finalize pooling for all three
  families: the loop's visible endpoint when set; `None` for a stop-sequence finish
  without one; else `model/tokens.py::last_content_index` over `generated_ids`
  (thinking tokens count, so a run ending mid-thought pools a thinking token).
  Forward `k` produced token `k`. Ring modes map through `tail_slice_at`; FULL
  indexes the stack directly (it never advances the ring counter).
- The finalize geometry aggregate uses a **cold** curved foot solve: the pooled
  position may not be the last forward a warm foot followed. `return_hidden`
  returns the stack trimmed to the generated length, on CPU.
- `begin_batch` / `batch_row_reads` serve the HF fast batch from a `[B, D]` ring;
  `score_hidden` is the offline read (bad input raises `DrowseError`).

## Tokens, payloads, results

- `TokenCallback` is positional six-argument and carries no readings.
  `TokenConsumer` receives whole `TokenEvent`s and declares what it reads in
  `TokenConsumerOptions` (`live_scores`, `per_layer_scores`, `lens_readout`,
  `sae_readout`, `perplexity`, `probe_readings`), which lets the engine skip
  unconsumed work.
- `TokenProbePayload` keeps one reading slot per `InstrumentFamily`.
  `to_token_payload` returns `{"measurements": envelope}` from
  `readout/measurements.py::build_measurements`, no aliases beside it.
  `all_readings` (the compat projection) runs only for a consumer that set
  `probe_readings`; `has_readings` is the cheap check.
- `TokenEvent.raw_index` is the `generated_ids` index behind the emission (the fork
  join key); `measurements` the token envelope; `probe_readings` feeds the
  OpenAI/Ollama `x-drowse-probe-readings` extension. The engine never sets
  `TokenEvent.finish_reason`; read it from the result.
- Per-token logprobs need a width: the engine passes `max(return_top_k, logprobs)`
  when the alternative width (per call, else the session default) is positive,
  else `SamplingConfig.logprobs`; `None` captures nothing even with a consumer.
  `GenerationResult.logprobs` is filled only when `SamplingConfig.logprobs` is set.
- `finalize_generation` never touches the capture. From scored `FinalReads` it
  decodes `generated_ids[thinking_end_idx:]` (or the exact `response_text` after a
  stop), builds the `scope="aggregate"` envelope and the `probe_readings` compat
  dict, calls `record` (the engine's `last_result`), emits `ProbeScored`, then
  finalizes the loom node. `thinking_text` is stamped only when the scene grammar
  has `think_open`; elsewhere a stamped block would make later renders raise.
- `GenerationResult.applied_steering` round-trips through `parse_expr`; `to_dict`
  omits `hidden_states` and `logprobs`. `RunSet` is a `list[GenerationResult]`
  with `node_ids`, `grid`, `kind`, `metrics`; `.first` raises on empty;
  `probe_summary()` feeds `notebook.plots.plot_trait_history`.

## Completion scoring (`scoring.py`)

For a context and a fixed candidate set, each candidate's joint `log P(candidate |
context)`: the logit counterpart of a template-derived manifold fit (the fit pools
each slot-fill's activation, the scorer reads its logprob). Template format:
`drowse/store/AGENTS.md`. `session.score_choices` / `score_template` delegate to
`ChoiceScorer`; `steering=` answers whether steering shifted the distribution,
not just the argmax.

- **Raw model distribution:** fp32 `log_softmax`, temperature 1, no top-k/top-p.
- Reports `sum_logprob` and length-normalized `mean_logprob`, each with its own
  softmax over the set (`prob_sum`, `prob_mean`); length bias is real, so neither
  view is chosen silently. A degenerate candidate (no distinct token, so
  `sum_logprob` 0.0, the largest value) is masked to `-inf` in both softmaxes; an
  all-degenerate set returns zeros.
- `shared_prefix_len` recovers each span, absorbing the boundary-token merge
  (`"Monday"` vs `"Tuesday"`); `loom/joint_logprobs.py` reuses it.
- One right-padded, `use_cache=False` teacher-forced forward per `_SCORE_BATCH =
  16` rows, `logits_to_keep` trimmed to the completion-producing tail (the batch
  bounds the `[n, tail, vocab]` output; the gather offset holds whether or not the
  model honored the kwarg).
- Renders with plain `build_chat_input` (no scene grammar, labels or cache).
  `system_prompt=None` means **no** system prompt, not the session default. Text
  after a template's slot is not scored.
- The steering scope opens **outside** the exclusive-GPU section
  (`base/exclusive.py::ExclusiveGuard`): composer push/pop take the same lock
  blocking, so the other nesting self-deadlocks. Every forward holds the section
  against a concurrent model op racing the hooks and the MPS command buffer.

## Tests

CPU, `tests/generation/`: `test_decode` (thinking exits; the HF config
reproduces the greedy loop on a checkpoint carrying `repetition_penalty` and
`top_k`), `test_generation` (step identity, UTF-8 flush, stop trimming, finalize
per capture mode, gate callback, no-KV fallback), `test_prefix_cache` (each
rollback mechanism on a real cache: llama, sliding mistral, lfm2, qwen3_next,
static llama), plus scoring, render, results, sampling, logits and payloads.
Elsewhere: `tests/readout/test_read_planner.py` (rig
`tests/support/read_plane.py::ReadRig`), `tests/session/test_engine.py`
(routing), `tests/browser/test_browser_parity.py` (sampler parity).
`tests/gpu/test_smoke.py::test_throughput_regression` requires steered decode at
≥ 85% of vanilla tok/s; changes to the loop, capture or routing need that
real-model run, not only the CPU suite.
