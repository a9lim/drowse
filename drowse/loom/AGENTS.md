# drowse/loom — AGENTS.md

The conversation layer: the branching tree a session generates into
(`LoomTree`), the receipt on each generated turn (`Recipe`), and the one rule
that composes a generation's effective recipe (`compose_call`). It also holds
the cast roster, conversation ops, token replay, diff, joint logprobs, the
filter grammar and YAML transcripts. Nothing here owns a model. Neighbours:
`session/engine.py` (`GenerationEngine.compose_call` gathers the tiers,
`TokenTap` writes rows), `session/session.py` (`_start_loom_assistant` stamps
recipes; the verbs, `restore_tree`, the joint-logprob cache),
`session/authored_capture.py`, `generation/{finalize,render}.py`,
`model/scene.py` (seats, labels, think delimiters), `server/tree_routes.py`.

## Module map

| Module | Owns |
|---|---|
| `tree.py` | `LoomNode`, `LoomTree` (mutators, reservation, cast roster, persistence), `TokenRow`, `TREE_FORMAT_VERSION`, `TOKEN_SIDECAR_FORMAT_VERSION` |
| `recipe.py` | `Recipe`, `CastMember`, `compose_call`, `CALL_ONLY_SAMPLING_FIELDS`, regen modifiers, `derive_seed_schedule` |
| `ops.py` | `append_turn`, seating/thinking checks, `resolve_token_anchor`, `plan_fork`/`plan_prefill` (→ `ForcedTurn`), `begin_generation_turn` |
| `replay.py` | `TokenReplayer` (implements `readout/replay.py::TokenReplay`) |
| `diff.py` | `text_diff`, `readings_diff`, `per_token_diff`, `node_diff`, `steering_delta` |
| `filter.py` | `parse_filter`, `filter_tree`, `node_token_series`, `FilterParseError` |
| `transcript.py` | `Transcript`, `Turn`, `ProbeRef`, replay results, `Transcript*Error` |
| `joint_logprobs.py` | `JointLogprobReplayer`, `JointLogprobs`, `SamplerRows`, `reorient_for_request` |
| `errors.py` | `LoomTreeError` 400, `UnknownNodeError` (KeyError) 404, `InvalidNodeOperationError` (ValueError) 400, `MutationDuringGenerationError` (RuntimeError) 409, `RecipeOverrideError` (RequestError) 400; strict-reader helpers |

## Layering

`loom` sits above `fit` and below `session`. `tests/invariants/test_layering.py`
enforces this, `TYPE_CHECKING` imports included. Lower layers never import
loom: `generation/render.py` types the tree as `ConversationHistory`, and
`generation/finalize.py` takes `tree: Any`.

Collaborators take the tree as a getter (`tree: Callable[[], LoomTree]`),
because `restore_tree` replaces `session.tree`. Only `transcript.py` takes a
duck-typed `session`: it uses `tree`, `model_id`, `config.system_prompt`,
`probe_hashes()` and `generate`.

## Tree data model

- **Root.** The root is a synthetic, empty, parentless `system` node.
  `messages_for` and `flat_text` skip it; a system prompt is never a node.
- **Provenance is recipe presence.** A node with a `Recipe` is generated in
  either seat, since the model can generate into the user seat.
  `recipe is None` is authored. `rewind`, `anchor_recipe`,
  `set_authored_token_scores`, transcript import and joint logprobs key on
  this. Never infer provenance from `role`.
- **`role_label`** is the cast label the turn was sent under (`None` = family
  default). It is stamped at creation and immutable, and is unrelated to the
  transient `SteeringComposer.active_role`.
- **`thinking_text`** is committed by the author or stamped at finalize, and
  finalize stamps it only when the scene grammar has `think_open`. A block the
  family can't re-render would make every later render raise.
- **Token rows.** `tokens`/`thinking_tokens` hold opaque row dicts from
  `TokenTap` (browser-seam shape), never mutated, so a shallow list copy is a
  stable snapshot. `raw_token_ids` is the exact decode sequence, delimiters
  and unmerged partial UTF-8 included: the forceable prefix for forks and
  replays, joined to rows by `raw_index`. Authored and imported turns have
  `tokens=None` until authored capture adds rows (which have no `raw_index`).
- **Readings.** `aggregate_readings` holds axis-0 finalize aggregates from all
  three read families; `mean_logprob`/`mean_surprise` cover the response span.
- **Edits.** `edit` rewrites `text` only. The recipe, rows and
  `raw_token_ids` keep the original decode, so forks, replay and joint
  logprobs read the original; `edit_count`/`edited_at` are the only signal.
  `append_authored` instead clears the receipt, making the node authored.

## Mutation rules

Public methods take the tree's `RLock`. Each mutation bumps `rev` and emits
one `LoomMutated` (`drowse/events.py`); surfaces refetch on a `rev` gap.
No-op calls emit nothing, and hot-loop `append_token` neither bumps `rev` nor
emits (deltas ride the token stream). Ops: `add_user`, `navigate` (also
`rewind` and a dedup hit), `begin_assistant` (also `begin_continuation`, as
`updated`), `finalize_assistant`, `capture_authored`, `edit` (also
`append_authored`), `branch`, `delete`, `star`, `note`, `reset`, `import`
(`graft`), `cast` (no node ids), `restore` (`announce_replacement`).

**Reservation.** `begin_generation_turn` reserves the parent of the node a
generation streams into, and `session._release_generation` clears it in a
`finally`. While set, `edit`, `append_authored` and `delete_subtree` refuse
the reserved node, its ancestors and its descendants (siblings of the
streaming node included), and `reset` refuses outright, all with
`MutationDuringGenerationError` (409). New turns, `branch`, `graft`,
`navigate`, star, note and cast ops stay free. `replacing` doesn't check it,
so `session.restore_tree`'s caller must exclude generation (the server holds
its session lock).

| Op | Behavior worth knowing |
|---|---|
| `add_user_turn` | Reuses a user child with identical text, `role_label` and `thinking_text` (regen and prefill rely on this) |
| `add_assistant_turn` | Authored turn through the same begin/finalize pair: `tokens=None`, `raw_token_ids` kept, so it stays forkable |
| `begin_continuation` | Resets a node in place; the caller replays its old text as the forced prefix |
| `branch` | Authored sibling. Copies `role_label`; copies `thinking_text` only for unchanged text; `role=` overrides the seat; refuses the root |
| `graft` | Validates all turns (user/assistant seats, role slugs) first, then chain + absent cast labels + note in one `rev` |
| `delete_subtree` | Re-seats an active node inside the subtree on the surviving parent |
| `reset` | New root (old id in `removed`); the cast survives and `rev` keeps counting |
| `rewind` | Up one submission: two nodes from a generated node, one from an authored one. No-op only at the root |

## Persistence

`tree_format` 2 and `token_sidecar_format` 2 are required exactly. Readers are
strict (`errors.require_fields`: every written field required, unknown keys
rejected) and never migrate. `from_dict` also checks structure (`children_of`
matches node ids, one agreeing parent, all reachable from a parentless
`system` root, no cycles, all three token fields or none).

`to_dict()` omits token payloads; `GET /tree` sends `include_tokens=True` and
`PUT /tree` reads it back. `save` writes the JSON atomically after a gzip
`<stem>.tokens.json.gz` sidecar when any node has rows or `raw_token_ids`
(else removes a stale one). `load` caps the JSON at 64 MiB and the
decompressed sidecar at 256 MiB (`max_bytes=` overrides both). The reader
skips `structural`/`observed` cast entries, so the server's effective-roster
JSON round-trips. `LoomTree.replacing` refuses another model's tree (null
`model_id` passes), keeps the event bus and `session_id`, and sets
`rev = max(prev, saved) + 1`. The owner installs it, then calls
`announce_replacement` (WS subscribers serialize `session.tree` on that
event). There is no autosave.

## Recipes and composition (`recipe.py`)

A stamped `Recipe` carries `steering` (canonical; `None` when unsteered),
`sampling` (the composed config, including the seat labels and observation
knobs), `thinking`, `system_prompt` (resolved; `None` in raw mode), and
`probes`/`probe_hashes` (geometry probes only; a transcript header uses
`session.probe_hashes()`, all families). The seed lives only in
`sampling.seed`: on disk a top-level `seed` mirrors it and wins on read, and a
seed without a sampling block is `seed_only` (writes `"sampling": null`; build
with `Recipe.from_parts`). Every `SamplingConfig` field is persisted and
required on read, so a new sampling field changes saved trees, transcripts
and the browser seam. An absent `system_prompt` means inherit at use; `None`
and `""` are captured values.

`compose_call` is the single composition rule (the browser edition mirrors
it). Tiers, weakest first: `cast`, the call's kwargs, `anchor`, `modifier`. A
stronger tier sets `steering`/`thinking` when not `None` (`""` is explicit
unsteered and blocks weaker tiers) and `system_prompt` when not inherited, and
merges `sampling` field-wise via `SamplingConfig.merged_with` (a default field
is unset, so no tier can reset a field to its default). Two exceptions:
`CALL_ONLY_SAMPLING_FIELDS` (`logprobs`, `return_hidden`, `return_top_k`, both
`persist_*` flags, `return_probe_readings`) always come from the call, so a
recipe never decides what a consumer captures; and the seed resolves in the
order modifier > call > anchor > cast, so fanned per-sibling seeds beat the
anchor (N distinct samples) while an unseeded one-shot regen reproduces it.

`GenerationEngine.compose_call` supplies the tiers. The cast member is
`tree.cast[label]`, the label being the call's seat role label or else the
seat name (a member named `assistant` covers every unlabeled assistant turn);
raw mode skips the roster. Anchor and modifier exist only with
`recipe_override`; the anchor is `anchor_recipe(parent_node_id)`, the parent's
recipe if generated, else the nearest generated ancestor's. A regen passes the
regenerated node's parent, so the anchor is never that node or a sibling. A
composed `""` becomes no steering before parsing.

Modifiers (`Recipe.compose_modifier`): `unsteered` (steering `""`), `inverted`
(every coefficient negated; `along` on a `%` term, `onto` kept), `reseed`
(fresh 31-bit seed), `cool`/`hot` (temperature 0.3/1.2), a `Recipe`
passthrough, or a partial `"field=value, …"` over `steering`, `thinking`
(`true`/`false`), `seed`, `temperature` ≥0, `top_p` [0,1], `top_k` ≥0,
`max_tokens` ≥1, `presence_penalty`/`frequency_penalty` [-2,2]. It splits only
at `, <ident>=` (steering may contain commas) and parse-validates steering;
errors are `RecipeOverrideError`.

Gotchas: a modifier seed beats the call, so `seed=42` on a fan pins every
sibling and `reseed` gives each a fresh seed instead of the schedule.
`inverted` on a curved `%` term composes, but the steering plan refuses the
negated `along` (`SteeringCompositionError`, 422); curvature is per-model, so
the refusal is the composer's.

`derive_seed_schedule(base, n)`: `n=1` returns the base (31-bit), `n>1`
derives cross-machine-deterministic `blake2b(base, i)` seeds, `None` draws
entropy. Each sibling stamps its own seed (`session/batch.py`, the WS fan).

## Cast roster

`tree.cast` maps label → `CastMember` (optional standing `Recipe` + `notes`).
`cast_roster()` is the effective view: structural `user`/`assistant`, every
label observed anywhere in the tree (not just the active path, so speakers
don't flicker on navigation), then configured members on top. Roster ops are
decoration-tier and validate the label as a role slug.
`LoomTree.set_cast_member` checks only the label; `session.set_cast_member`
builds through `CastMember.from_parts`, which parse-validates steering at
authoring time. A fieldless member contributes nothing; steering `""` pins a
label unsteered.

## Conversation operations (`ops.py`)

- `append_turn` is the one authored-append rule. Same seat and label as the
  parent concatenates via `append_authored` (`raw_token_ids` = tokenization
  of the full text); otherwise a new user turn (deduping) or authored
  assistant turn lands. Seating is free in scene mode (`grammar` set) and flat
  mode. A legacy chat template renders only alternation, so there a user turn
  may not hang under a user node and an assistant turn must.
- `check_thinking_commit` raises `SceneThinkingUnsupportedError` (400) at
  commit time when the grammar has no `think_open`.
- `begin_generation_turn` opens and reserves. A `str` input commits the user
  turn first (label from `sampling.user_role`); an assistant-seat send refuses
  a user parent even in scene mode (commit with `append_turn`, then generate
  from `None`). Flat mode lands a non-empty span as its own node and lets an
  empty one extend the parent. `continuation_node_id` reuses a message.
- `resolve_token_anchor` is the one `(node_id, raw_index)` validation, shared
  by forks and replay; `TokenAnchor` snapshots `raw_ids`.
- `plan_fork` replays `raw_ids[:raw_index]`, forces `alt_token_id` or
  `replacement_text`'s tokens (exactly one), then samples under the node's
  recipe steering, sampling, thinking and system prompt. The original seed
  re-applies by default and the sampler still draws every forced step
  (`generation/decode.py`), so the RNG stays aligned through the fork; `seed=`
  asks for a new ending. `max_tokens` grows by the forced span. The sibling
  keeps the node's parent, seat and `role_label`.
- `plan_prefill` opens an assistant reply to a user node with the text,
  thinking off, re-sending the user text so dedup reuses the node. It is
  Python-only (`session.prefill_assistant`); forks reach the WS via `generate`.

Gotchas: a fork passes the recipe's steering verbatim, so a node stamped
unsteered (`None`) picks up its label's cast steering if a member now exists
(transcript replay passes `""` for exactly this reason). Prefill carries
neither the node's `role_label` (unless `sampling.user_role` matches) nor its
`thinking_text`, so a labelled or thinking-carrying user node gets a duplicate
sibling.

## Token replay (`replay.py`)

`TokenReplayer` backs every family's `token_readout`. It resolves the anchor,
rebuilds the node's continue-mode render to its parent (stamped labels, recipe
thinking and system prompt, node seat), appends `raw_ids[:raw_index]`, runs
one capture forward, and hands `{layer: (hidden,) fp32}` pooled at the
producing position to the family's `read`. The steering scope opens
**outside** the exclusive-GPU section (`base/exclusive.py::ExclusiveGuard`):
composer push/pop take the generation lock blocking, so nesting would
deadlock. `layers` and `read` resolve inside it. `apply_steering=False` reads
the unsteered counterfactual. Always-active affine terms reproduce exactly;
phase/gated terms fire as the trigger context's rest state decides.
`raw=True` picks the flat render, which the node does not record.

## Diff (`diff.py`)

`text_diff` is word-level `SequenceMatcher(autojunk=False)` on
whitespace-split tokens (replace = delete then insert; whitespace not
preserved). `readings_diff` is per-probe `b − a` sorted by `|Δ|`, an absent
side reading 0.0 with originals kept. `per_token_diff` aligns only where
cumulative UTF-8 byte offsets and texts both match; tails are unaligned at
index `-1`. `node_diff` sets `parent_id` only for siblings. `steering_delta`
subtracts by canonical `Steering.alphas` key (bare atoms resolve against
installed manifolds) and returns `""` for no label, never a sentinel.

## Filter grammar (`filter.py`)

```
filter := clause ("," clause)*                              # AND
clause := [("agg" | "any" | "last") ":"] probe op number    # default agg
op     := ">" | ">=" | "<" | "<="
probe  := [ns "/"] NAME ["." NAME] | "sae/" INT
```

Deliberately separate from `@when:`, which gates live per-step readings; this
filters what the loom persisted. `agg:` reads `aggregate_readings`. `any:`
(max for `>`/`>=`, min for `<`/`<=`) and `last:` read the node's own rows,
thinking first, from `measurements.scores` (never the flat `probes` map),
skipping rows without the probe. A missing probe makes the clause false. No
OR, negation or grouping; compose with `LoomTree.filter(pred)`. Keys go
through `readout/types.py::parse_gate_ref`: `name[0]` normalizes to `name`,
and any other channel is a `FilterParseError` (400), since only axis 0 is
persisted. Names must be ASCII `[ns/]name[.pole]` or `sae/<id>`, so a
`jlens/#<id>` probe, a non-ASCII lens word or a variant-suffixed name can't
be filtered. The browser ports this grammar; change both together.

## Transcripts (`transcript.py`)

YAML, `drowse_transcript: 2` exactly. Header: `model_id`, `system_prompt`
(the session default; per-turn prompts ride the recipes), optional `cast`,
`probes` (`name` + `sha256`). Turns: `role` (seat), `text`, optional
`speaker` (`role_label`), `thinking`, `recipe`, `readings`. The reader is
strict (`TranscriptFormatError`).

`import_into` is one `graft`, so everything lands or nothing does. Turns with
a recipe re-attach as generated nodes in their seat; system turns are
skipped; imported nodes have no rows or `raw_token_ids` (not forkable or
replayable). New cast labels land, and a label the roster holds with a
different member keeps the session's and is flagged. Modes: `default`
(branch off the root), `here` (under the active node), `merge` (under the
deepest active-path user turn matching the transcript's user turn at the same
position, else the root; the matched prefix is skipped). Guards warn and note
the first grafted node: model mismatch (raises `TranscriptModelMismatch`
under `merge`), system-prompt mismatch, missing probes, probe drift
(`strict=True` raises `TranscriptProbeDriftError`).

`replay` imports, then regenerates each generated turn as a sibling under its
own parent (full history in context) with the recorded steering (`""` when
unsteered), sampling, seed, thinking, system prompt and seat, and diffs
recorded → replayed (`node_diff`, `a` = recorded). Fail-fast; `isolate=True`
records `DrowseError` failures and continues. CLI: `drowse experiment
transcript run`.

## Joint logprobs (`joint_logprobs.py`)

`JointLogprobReplayer.compute(a, b)` (`session.joint_logprobs`) force-replays
both nodes under their stamped recipes and reports, per aligned position,
both chosen-token logprobs, `lp_a_in_b`/`lp_b_in_a`, `rank_changed` and
`approx_kl`.

- Prompts rebuild as the node was opened (labels, seat, system prompt;
  thinking from recipe, else steering, else model default). A rowless node
  recovers tokens from the closed render, split at the shared prefix. Rows
  with `token_id < 0` (buffered partial UTF-8) are skipped; thinking is forced
  but its rows dropped.
- Rows are sampler support pools (`SamplerRows`): effective top-k with
  renormalized logprobs after the decode loop's logit edits (NaN/±inf clamp,
  penalties, `logit_bias`). Off-pool is `-inf`, exact for the sampler, with no
  dense `[rows, vocab]` table.
- Position-independent steering (`all_fast_path()` or `static_steerable()`)
  with no penalties replays teacher-forced in `_REPLAY_CHUNK = 128` chunks;
  otherwise one forced token per forward drives the trigger context, with a
  gate-only read-plane binding when probe gates need it (runs close in a
  `finally`).
- Alignment is `shared_prefix_len` then `per_token_diff`; indices are relative
  to the divergence point and cross values exist only on aligned rows.
  `approx_kl` is KL(P_A‖P_B) over the union of each side's top
  `_KL_TOP_K = 32`; infinite KL reports `None`.
- The caller serializes against generation (server session lock). Unknown ids
  raise a bare `KeyError`; the route pre-checks.
- Caching is the server's: `session.joint_logprob_cache`, keyed by
  `_cache_key` (sorted pair), LRU of 8 (`server/tree_routes.py`). The session
  clears it on `edit`, `delete`, `reset`, `finalize_assistant`, restore and
  close. `reorient_for_request` swaps columns but keeps `approx_kl`, which
  stays in the first-computed orientation.

## Browser seam

The browser edition (`stdlogits/drowse.ai`) ports the tree ops,
`compose_call`, the filter grammar, diff and transcripts.
`tests/fixtures/browser/loom-operation-transcript-v1.json` (checked by
`tests/browser/test_browser_parity.py`) pins a shared script's events (`op`,
`rev`, ids, `active_node_id`) and final tree; token rows carry the envelope
pinned by `measurement-envelope-v1.json`. Changing an event, the recipe, node
or tree JSON, the composition rule or the filter grammar is an interface
change there.

## Testing

All CPU. `tests/loom/` covers the tree, reservation
(`test_loom_concurrency.py`), ops, composition (`test_recipe_override.py`),
diff, filter, transcripts, rows, replay and joint logprobs.
`tests/session/test_cast.py` and `test_conversation.py` run on `tiny_session`
(a real tiny model). Routes: `tests/server/test_server_loom.py`; browser
golden: `tests/browser/test_browser_parity.py`.
