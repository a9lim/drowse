# drowse/readout — AGENTS.md

The read side's lower half: the shared read vocabulary, the instrument
contract all three read families implement, the geometry family (the whitened
`Monitor` and its `GeometryInstrument` adapter), the hidden-state capture
buffers, the pure capture planner, the probe roster, the versioned measurement
envelope, and Mahalanobis cosine analytics. The narrative of how reads fit a
generation is the read-plane section of `ARCHITECTURE.md`; this file is the
contract an editor needs.

Lives elsewhere: `generation/read_plane.py::ReadPlane` applies capture plans,
arms the step sinks, builds the gate callback and runs the finalize reads;
`loom/replay.py::TokenReplayer` implements the `TokenReplay` protocol;
`lens/instrument.py` and `sae/instrument.py` are the other two families;
`geometry/whitener.py` is the metric; `session/session.py` wires it all.

## Module map

| Module | Owns |
|---|---|
| `types.py` | `InstrumentFamily`, gate channels + `GateRef`/`parse_gate_ref`, `ProbeReading`, `ScalarReading`, `DepthSummary`, `LiveState` subclasses, `ReadRequest`/`InstrumentPrep`/`LensPrep`/`InstrumentPlan`/`InstrumentBinding`, `AGG_TAIL_DEPTH`, `next_prep_token` |
| `protocol.py` | `Instrument`, `InstrumentRun`, `LatestSlices` (runtime-checkable Protocols) |
| `replay.py` | `TokenReplay` protocol, `ReplayedToken`, `ReplayLayers` |
| `planner.py` | `GateDemand`, `ReadDemand`, `resolve_read_demand`, `CapturePlan`, `plan_capture`, `CaptureMode`, `CaptureState` — pure, no side effects |
| `capture_buffer.py` | `HiddenCapture`, `install_persistent_capture_hooks` |
| `monitor.py` | `Monitor`: every geometry read and gate scalar |
| `monitor_attach.py` | attach-time algebra: `AttachedManifoldProbe`, `_LayerWhiten`, `_FactorStack`, bandwidths, `NEUTRAL_LABEL`; plus the per-token `_layer_geometry` |
| `monitor_flat.py` | `_FlatEngine`: the batched read over every flat probe |
| `geometry.py` | `GeometryInstrument`, `GeometryRun` |
| `roster.py` | `ProbeRoster` and its `GeometryFamily`/`ReadoutFamily` protocols |
| `measurements.py` | `build_measurements`, `live_sources`, the envelope TypedDicts |
| `analytics.py` | `cross_layer_cosine`, `cosine_matrix` (whitened; used by `server/profile_routes.py`) |

## Layering

`readout` sits above `steering.{grammar,terms,triggers}` and below `lens`/`sae`
(`tests/invariants/test_layering.py`; `TYPE_CHECKING` imports count). It must
not import `lens`, `sae`, `steering.composer`/`hooks`, `generation` or
anything above; collaborators from those layers are typed against the
Protocols here (`LatestSlices`, `TokenReplay`, `GeometryFamily`,
`ReadoutFamily`). `monitor_attach.py` must never import `monitor.py`.
Dependencies arrive through constructors; nothing here touches a session.

## The instrument contract (`protocol.py`)

`Instrument` carries only surface every family answers the same way: `family`,
`state_lock`, `current_run`, `prepare`/`plan`/`bind`/`close_run`, `names`,
`specs`, `try_detach`, `probe_hash`, `validate_gate`, `active_source`,
`live_state`/`set_live`, `token_readout`. Anything one family cannot answer
uniformly stays family-native and off the Protocol: geometry's `top_n`, the
lens's `probe_layers`, the SAE's `probe_values`. `attach` is therefore not on
`Instrument`; it is on the roster's `GeometryFamily`/`ReadoutFamily`.

**Transaction.** Every capture transaction runs `close_run → prepare → plan →
bind` per family (`ReadPlane.bind_runs`) and reaches `close_runs` in a
`finally`. Rules every family enforces:

- `prepare` raises on a still-bound run (a stale lens pin would short-circuit
  the very refresh `prepare` exists for). Source refresh precedes planning
  because lens adoption may rewrite live probe specs.
- Each prep draws a token from one process-wide sequence (`next_prep_token`);
  per-instance counters would collide across instruments. The plan echoes it
  as `prep_token`; `bind` refuses a plan/prep pair whose family or token
  disagree.
- `plan` derives from the prep only, never the live registry.

`InstrumentRun` is `binding` (frozen source + specs), `bound`,
`observe_aggregate(pooled)` and `close`. The unbound idle passthrough run backs
out-of-generation reads. There is no per-run reading memo: within one forward
the decode loop passes the same `step_id` (`len(generated_ids)` before the
forward) to the capture sink, the gate callback and the token tap, and reuse
is the sink's row (geometry) or the lens/SAE step-keyed stash.

**Locks.** Each family's `state_lock` is a reentrant leaf: roster mutation and
the coherent read surfaces (`names`, `specs`, `plan`/`bind` roster reads, idle
run reads) take it; callers take the exclusive-GPU section first; it is never
taken on the bound per-token path. Mid-generation lens/SAE detaches are safe
(the bound run measures its frozen binding). Geometry has no frozen roster, so
geometry attach/detach hold the exclusive section and a detach during a
generation fails fast with `ConcurrentExtractionError` (409, "retry shortly").

**`token_readout`** returns the finished `{"measurements": …}` replay envelope.
`top_k=None` means the family default (8 for the ranked lens/SAE reads). A knob
a family cannot honor raises `RequestError` (400), never silently ignored:
geometry rejects `top_k` and `layers`, SAE rejects `layers`.

**`validate_gate`** guards channels a family can never produce. Lens and SAE
accept only the strength axis (`validate_gate_channels`); geometry accepts
everything. It runs at generation preflight in `ReadPlane.resolve_gates`, for
gates naming an attached lens/SAE probe.

`session.instruments` is the family enumeration. `generation/token_payloads.py::_FAMILY_SLOTS`
and `server/instrument_routes.py::CAPABILITIES` are pinned to it by test, so a
new family is a registry entry plus its implementation (full checklist: the
change map in `ARCHITECTURE.md`).

## Vocabulary (`types.py`)

Gate scalar keys; `parse_gate_ref` is the only key-shape parser and inverts
what `Monitor.flat_scalars` emits:

| Key | Channel | Producers |
|---|---|---|
| `<probe>` / `<probe>[i]` | `Axis(i)` (`[0]` normalizes to the bare form) | all |
| `<probe>:fraction` | `Fraction` | geometry |
| `<probe>:membership` | `Membership` | geometry |
| `<probe>@<label>` | `Distance` (negated, label-spacing units) | geometry |
| `<probe>~<label>` | `Assignment` (probability) | geometry |

`<probe>` is the attached name verbatim (`jlens/fake`, `default/emotions`,
`pirate:role-x`). Parsing is safe because `NAME_REGEX` forbids `@`/`~` in names
and labels, and `:` splits only on the two exact channel words. The runtime
gate lookup stays a plain `dict.get` on the verbatim string; `GateRef` is for
preflight and planning.

Readings: geometry produces `ProbeReading`; lens and SAE produce `ScalarReading`
(`value` + explicit `unit` — `mean_token_probability`, `activation_over_max`,
`raw_activation` — + `per_layer` + `DepthSummary`). `DepthSummary.basis` travels
with the numbers because the three families weight depth by unrelated masses;
never compare centers across bases. `as_probe_reading` /
`ScalarReading.to_probe_reading` is a compatibility projection used only for
`TokenEvent.probe_readings` and the OpenAI/Ollama `x-drowse-probe-readings`
body key; never on the native per-token path. `reading_axis0` is the number
the flat `scores` view and loom rows carry. `scalar_gate_keys` emits only
`<name>` and `<name>[0]` for the single-axis families.

`InstrumentPlan` is demand, not mechanics (`latest_layers`, `tail_layers`,
`gate_keys`, `final_aggregate`); physical retention and cross-family upgrades
are the planner's. Do not add a demand field `plan_capture` does not read.
`AGG_TAIL_DEPTH = 8` is the finalize ring depth, deep enough to walk back past
trailing special tokens to the last content token.

## Capture planning (`planner.py`)

`resolve_read_demand` folds a generation's consumers into one `ReadDemand`
(the engine masks every per-token geometry consumer when the geometry live
toggle is off):

- `per_token_full_consumer` = (live scores on and loom attached) or live token
  scores or persisted per-layer scores or persisted probe row or persisted
  inspector coords.
- `need_per_token` = a geometry gate is active or `per_token_full_consumer`.
  Lens/SAE gates score on their own paths and never force geometry scoring.
- `gating_only_probes` = the gated geometry probes when gates are the only
  per-token consumer.
- `lean_per_token` = per-token need, no `return_hidden`, no geometry gate,
  geometry probes attached, and no full-reading consumer (live token scores,
  per-layer heatmap, inspector coords): every per-token consumer reads axis 0.

`plan_capture` layers: `return_hidden` (`widen`) captures every model layer;
otherwise the union of geometry `latest_layers` and lens/SAE `latest_layers |
tail_layers`, and an empty union attaches nothing. The branch order is
load-bearing; the first match wins ("monitor" = geometry probes attached and
not widened; "ring 8" = `AGG_TAIL_DEPTH`):

| Mode | When | Retention | Step sink | Curved feet | Finalize (in `ReadPlane`) |
|---|---|---|---|---|---|
| `GATING_SUBSET` | monitor, per-token need, `gating_only_probes` | ring 8 if final aggregate, else 1 | `subset`: planned gate scalars only | warm | pool full roster once from ring |
| `LEAN_INCREMENTAL` | monitor, per-token need, `lean_per_token` | ring 8 | `lean`: `coords_only` rows | warm | coord stream from rows; full aggregate re-scored once from ring |
| `INCREMENTAL` | monitor, per-token need | 1; with a lens/SAE final aggregate or gate, ring 8 (1 if final off) limited to their tail layers when any | `full` rows | warm | aggregate = the row at the pooled position |
| `AGGREGATE_ONLY` | monitor, no per-token need | ring 8 | none | cold | pool once |
| `AGGREGATE_ONLY` | no monitor; lens/SAE declared layers | ring 8 if final, else 1 | none | cold | lens/SAE pool once |
| `FULL` | `return_hidden` | unarmed: clone every step | none | cold | `Monitor.score_per_token` over the stack |

Capture rides the persistent compile-clean buffers iff not widened, no
authored prompt rows (`capture_prompt`), and the caller passed buffers
(compiled-clean route with steering absent or lowered to offsets); otherwise
transient hooks. `CaptureState` is what the plan leaves for finalize, the gate
callback and the token tap; its `aggregate_only` is true for `AGGREGATE_ONLY`
and for `GATING_SUBSET` with a final aggregate.

A lean row is coords + fraction only. The demand rule above is what keeps a
lean row from ever reaching a full-reading consumer; widen it before adding a
consumer that reads more than axis 0.

## Capture buffers (`capture_buffer.py`)

Hooks do device work only: detach the last-position `(D,)` slice, then clone it
or `copy_` it into a preallocated buffer. Retention (`set_retention`, after
`attach`/`attach_persistent`) is one policy: unarmed = full retention for
`stacked()`; depth 1 = one in-place buffer per layer; depth > 1 = a ring,
restricted to `tail_layers` when set (other layers stay length-1 and are
omitted from `tail_slice_at`). The step sink runs after the forward returns, so
its host read never drains the device mid-forward. Ring modes count forwards
(the highest hooked layer advances the count; `ingest_persistent` does it on
the persistent path); `FULL` does not, so pooling indexes its stack directly.
`install_persistent_capture_hooks` covers every layer and must run before
`torch.compile`; `ingest_persistent` replays the same retention and sink
post-forward. Prompt rows (`set_prompt_positions`) exist only on transient
hooks. `attach_batch_tail` is the `generate_batch` `[B, D]` ring.
`is_transient()` is the compiled-clean routing gate.

## The Monitor

Every probe is a `Manifold`; every read (live token, aggregate, replay) is one
full `ProbeReading` from the same scorer (an aggregate is a live read at the
pooled token, differing only in a curved probe's cold foot start).
Mahalanobis-only: the wired whitener must cover every probed layer or attach
raises `WhitenerError`; there is no Euclidean path.

Reading fields (cross-layer means are share-weighted; `share_weights` are the
fit's `mahalanobis_share`, normalized):

| Field | Flat (affine) probe | Curved probe |
|---|---|---|
| `coords` | affine map of `M_R⁻¹ B Σ⁻¹(h − mean)` to the domain frame | `invert_parameterization` foot |
| `fraction` | `‖P_M x‖_M / ‖x‖_M` in `[0, 1]` | same |
| `residual` | `0.0` | normalized off-surface distance |
| `membership` | `1.0` | `exp(−res²/2σ(z)²)` at the foot |
| `nearest` | top-`top_n` by raw whitened distance, reported ÷ `label_scale` | same |
| `assignment` | `softmax(−d²/2τ² − R·log τ)`, top-`top_n` | same |

- Coordinates are domain-frame: each layer's reduced coord is mapped to the
  domain before averaging (raw per-layer coords are in `‖δ_L‖` units). Rank-1
  reads `1.0` at the positive node. Periodic, sphere, Klein and projective
  domains average in the embedding and map back; a near-zero mean falls back to
  the highest-share layer's coordinate instead of inventing a midpoint.
- `label_scale` is the median of the per-node bandwidths: each node's
  nearest-neighbor whitened distance for a flat (or σ-less) fit, the node's
  whitened tube thickness σ for a curved fit with a σ-field. Raw whitened
  distance spans ~60× across fits, so `@label` thresholds are portable only
  after this rescale. It is a per-probe constant, so ranking stays by raw
  distance; a single-node fit keeps 1.0 (raw distance).
- The `−R·log τ` log-volume bias is the Gaussian normalization; without it a
  diffuse-τ node swallows probability from far away. τ and the bias are
  precomputed at attach.
- `neutral` (`NEUTRAL_LABEL`) is a virtual candidate: the frame anchor in the
  same whitened metric (zero for affine fits, the baked `origin` through
  `eval_at → basis → chol` for curved). It rides `node_white_aug` as the last
  candidate in both paths, is never a stored node, and is suppressed when the
  manifold has a real node named `neutral`.
- `depth_com`/`depth_spread` need `n_layers` at construction; mass is
  `share_L · |coord_L[axis]|` at depth `L/(n_layers−1)`.
- `subspace_coords_per_layer` (inspector trail) is emitted only while
  `set_subspace_coords(True)`; the engine sets it per generation.

**Execution.** `_FactorStack` is the single device copy of the whitener's
Woodbury factors per device, grown rather than duplicated (detach keeps it),
and assembled on the host before one upload. A probe on a device manifold
attaches on the host too and uploads packed (`geometry/packing.py`), so the
roster stays a few device buffers instead of thousands of small tensors. A
read is one stacked `Σ⁻¹h` apply over every layer; the flat roster is then one
`_FlatEngine` pass (segments bucketed by rank, dim and candidate count; rank-1
is elementwise) and one host transfer; each curved probe adds its foot solve
and one transfer. Curved feet warm-start per `(probe, layer)` only while
`enable_curved_warm(True)` (sequential per-token scoring). Aggregates, replays
and `FULL` stacks read cold so one-shot reads stay reproducible. Curved
probes are the expensive reads: a foot solve per layer per token.

**Gates.** `plan_gate_scalars` parses keys once per generation;
`score_planned_gate_scalars` computes only the requested channels (curved
`:fraction`, `@label` and `~label` skip the foot solve). The planned path
scores any candidate label regardless of `top_n`; `flat_scalars` emits labels
from the top-N lists only, and in `INCREMENTAL` mode the gate callback fills
missing keys from the planned path. A key naming an unattached probe or an unknown label is dropped:
the gate stays inactive with no error.

**Attach.** `add_probe` validates runtime geometry (a raw-space curved fit
needs its σ-field; SAE-space curved fits have none and read `membership`
1.0), requires finite positive shares, and builds every factor up front.
`set_whitener` re-attaches the whole roster atomically (failure keeps the old
whitener and roster; same instance is a no-op), since bandwidths and
`label_scale` derive from the metric.

## GeometryInstrument

Constructor deps: `monitor`, `exclusive` (`base/exclusive.py::ExclusiveGuard`),
`resolve_manifold`, `ensure_whitener`, `replay`. `attach` enters the exclusive
section, builds the whitener, resolves the manifold, then adds under
`state_lock`. `refresh_swapped` follows the artifact registry's `on_swap`
(re-attach on the replacement with the same `top_n`, detach on eviction) in one
lock hold. `adopt_whitener` is the neutral cache's whitener hook. `plan`
returns no layers for an empty or dormant roster (no gate, no per-token
consumer, no final aggregate) and narrows to the gated probes' layers when
gates are the sole consumer and final readings are off. The binding freezes
names only. `is_live` (default on) is the all-or-nothing per-token switch;
`set_live` rejects extras with `TypeError`. `probe_hash` is sha256 over
fp32 bytes: the folded direction for a foldable fit, else per-layer
mean/basis/node coords. `token_readout` captures the roster's layer union
(resolved inside the exclusive section) and scores with a cold foot.

## ProbeRoster

`jlens/…` routes to the lens, `sae/…` to the SAE, anything else to geometry.
`detach` asks lens, then SAE, then geometry, and an unknown name is a no-op (the
server 404s first). Digests are cached per family until `forget`; `changed`
drops them and runs `on_change` (the session invalidates analytics; the prefix
cache survives). Names are not checked for uniqueness across families; a
collision shadows (hash and detach consult lens first; the flat `scores` join
keeps the last family written).

## Measurement envelope (`measurements.py`)

The one wire record for token frames (`scope="token"`), final aggregates
(`"aggregate"`) and replays (`"replay"`, `provenance="replayed"`);
`MEASUREMENTS_VERSION = 1`. Build rules: `None` when nothing was measured; a
family channel exists only when it has readings or a readout; lens/SAE channels
always carry `binding` (`source`, `steering`, SAE `layer`), geometry only when
the caller passes one (replay); `readings` are each family's native
`to_dict()`; `readout` is the native discovery surface; `scores` is the axis-0
join rounded to 6 places. The lens readout holds `p_l`, and this is the one
place it becomes the wire's `logprob` (floored at `1e-45` so JSON stays
finite). `ProbeReadingDict`'s last five keys are `NotRequired` because
persisted loom rows may lack them. `live_sources` is the one builder of
`lens_source`/`sae_source`/`sae_layer`; an aggregate binds a family's source
only when it carries that family's readings.

Wire drift is a test failure: `tests/readout/test_measurements_envelope.py`
pins key sets for every scope; `tests/browser/test_browser_parity.py` compares
against the hand-maintained golden
`tests/fixtures/browser/measurement-envelope-v1.json`, which the drowse.ai
runtime replays (change both together); the TypedDicts feed
`scripts/generate_webui_types.py`, gated by
`tests/invariants/test_webui_types_parity.py`.

## Testing

All CPU, in `tests/readout/`: `test_read_planner.py` (demand rules, the
`plan_capture` matrix, `ReadPlane` integration), `test_hidden_capture_retention.py`,
`test_persistent_capture.py` (persistent ≡ transient), `test_manifold_monitor.py`
and `test_monitor_whitened_readout.py` (reading semantics, neutral anchor,
gate scalars, whitened metric),
`test_score_hidden.py`, `test_instruments_types.py` (gate keys, protocol
conformance), `test_probe_roster.py`, `test_geometry_replay.py`,
`test_measurements_envelope.py`. Helpers: `tests/support/read_plane.py::ReadRig`,
`tests/support/geometry.py::stub_geometry_instrument`,
`tests/support/whitener.py`.
