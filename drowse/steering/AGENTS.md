# drowse/steering

The steering language and its lowering: the expression grammar and its IR
(`Steering`), triggers and probe gates, the composer that turns a steering
stack into a `SteeringPlan`, and the forward hooks that run it. The kernel
math (`subspace_inject`, `synthesize_subspace`, the fold and projection
primitives) lives in `drowse/geometry/`; this package decides what gets
injected, where, when, and how strongly.

| Module | Owns |
|---|---|
| `grammar.py` | Lexer, parser, fold; `parse_expr`, `format_expr`, `referenced_selectors`; term types `ProjectedTerm`, `AblationTerm`, `ManifoldTerm`; `DEFAULT_COEFF = 0.5`, `DEFAULT_ABLATION_COEFF = 1.0` |
| `terms.py` | `Steering` (frozen IR), `AlphaEntry`, `SteeringStackEntry`, `term_entry` / `entry_trigger` / `entry_coeff` |
| `triggers.py` | `Trigger` + presets, `ProbeGate`, `TriggerContext` |
| `composer.py` | `SteeringComposer` (stack, scopes, resolution, composition, plan cache, stack queries), `SteeringScope`, `resolve_scope_role`, `affine_manifold_push`, the `ArtifactSource` / `NeutralSource` protocols |
| `hooks.py` | `SteeringPlan`, `_lower_affine_subspaces`, `AffineOp`, `SteeringHook`, `SteeringManager`, `install_persistent_offset_hooks`, the gain constants |

Layering (`tests/invariants/test_layering.py`): `grammar`/`terms`/`triggers`
sit above `store` (parsing resolves bare atoms against installed manifolds via
`store.selectors`) and below `readout`; `composer`/`hooks` sit above
`lens`/`sae` and below `generation`. Lower layers that must refuse work under
steering take `SteeringComposer.assert_unsteered` as an injected callable.

Flow: text → `parse_expr` → `Steering` → `SteeringComposer.open_scope`
(`resolve` + role policy) → `SteeringScope.__enter__` → `push` →
`rebuild_hooks` → `plan_for` (cache) → `compose` → `SteeringPlan` →
`SteeringManager.install` → `SteeringPlan.lower` → `AffineOp` / curved groups
on `SteeringHook`s, or the compiled offset buffers.

## Expression grammar (authoritative)

Every live surface (Python, YAML `vectors:`, HTTP, CLI, loom recipes) parses
through `parse_expr`. `fit/bake.py::_parse_merge_expr` walks the private AST
(`_lex`, `_Parser`, `_Term`/`_Selector`/`_Atom`) to accept only
namespace-qualified additive scalar terms; keep those fields stable or update
it.

```text
expr       := [sign] term (sign term)*
sign       := "+" | "-"
term       := [coeffs ["*"]] ["!"] selector ["@" trigger]
coeffs     := NUM ["," signed_num]           # 2 values only on a % term: along,onto
selector   := atom [("~" | "|") atom | "%" position]
position   := signed_num ("," signed_num)* | IDENT     # coord list | node label
atom       := ("jlens" | "slens") "/" "#" INT          # exact vocab id
            | "sae" "/" INT                            # SAE feature
            | [IDENT "/"] IDENT ["." IDENT] [":" IDENT]  # ns / name . name : variant
trigger    := phase ["&" gate] | gate
phase      := preset | ("first" | "after") ":" INT
preset     := "both" | "before" | "prompt" | "after" | "thinking" | "response" | "generated"
gate       := "when" ":" probe op signed_num
probe      := [IDENT "/"] pname ( ":" ("fraction" | "membership")
                                | ("@" | "~") IDENT
                                | ["." IDENT] ["[" INT "]"] )
pname      := IDENT | "#" INT (jlens/slens) | INT (sae)
op         := ">" | ">=" | "<" | "<="
signed_num := ["+" | "-"] NUM
```

Lexical rules: `NUM` is unsigned (`1`, `.5`, `2.`, `1e-3`); a term's sign
comes from the leading/separating `+`/`-` and applies to every value of a
comma run (`-0.6,0.3 m%x` is `along=-0.6, onto=-0.3`). `IDENT` is
`[A-Za-z][A-Za-z0-9_]*`, and a `-` between two identifier characters joins
segments, so `a-b` is one name (slugged to `a_b`) and `0.5 a-0.3 b` is a parse
error; write `a - b`. Multi-word names use `_`; quotes, `==`, and `a.b.c` are
errors. `SteeringExprError` (400) carries the failing `col`.

| Form | IR value (`Steering.alphas` key) | Lowered as |
|---|---|---|
| `α x` | `ManifoldTerm` via the bare-atom ladder, else `α` / `(α, Trigger)` under `x` | push |
| `α a~b` / `α a\|b` | `ProjectedTerm` under `a~b` / `a\|b` | Mahalanobis LEACE `project_profile`: keep / remove the component of `a` along `b`, then push |
| `α !x` | `AblationTerm` under `!x` | `h' = h − α(h·d̂ − μ·d̂)d̂`; bare `!x` is α = 1 |
| `along[,onto] m%pos` | `ManifoldTerm` under `m%pos` | affine fit: push fragment; curved fit: own term |

Omitted coefficient: 0.5 (`DEFAULT_COEFF`), or 1.0 after `!`. Parse-time
rejections: `!` with `~`/`|`/`%`; `%` with `~`/`|` or a second `%`; chained
projections; a two-value comma run on anything that does not resolve to a `%`
term (a bare node label does: `0.6,0.3 pirate` is legal); more than two
values; more than one `@` clause or gate per term. Repeated keys merge: equal
triggers sum coefficients (`along` and `onto` separately), differing triggers
raise, and one key used both as a plain term and as a projection raises.
Projection, ablation and `%` terms default to `Trigger.BOTH`; a plain term
without `@` stores a bare float that inherits `Steering.trigger`.

### Triggers

| `@` form | `Trigger` | prefill | thinking tokens | response tokens |
|---|---|---|---|---|
| none, `@both` | `Trigger.BOTH` | yes | yes | yes |
| `@before` = `@prompt` | `Trigger.PROMPT_ONLY` | yes | no | no |
| `@response` = `@generated` | `Trigger.GENERATED_ONLY` | no | yes | yes |
| `@after` | `Trigger.AFTER_THINKING` | no | no | yes |
| `@thinking` | `Trigger.THINKING_ONLY` | no | yes | no |
| `@first:N` / `@after:N` | `Trigger.first(N)` / `Trigger.after(N)` | no | `gen_step < N` / `gen_step >= N` | same |
| `@when:G` | `Trigger.when(probe, op, t)` | no | while G holds | while G holds |
| `@<phase>&when:G` | `replace(<phase>, gate=…)` | no | phase window while G holds | same |

`@response` includes thinking tokens; only bare `@after` excludes them, and
`@after:N` is a token window, not a phase. `gen_step` counts raw generated
positions from 0, thinking included. A gate can never fire during prefill, so
`@before&when:…` parses but never fires.

### Probe gates

The gate's probe string is stored verbatim in `ProbeGate.probe` and must
equal a scalar key the read side writes into `TriggerContext.probe_scores`,
i.e. the attached probe's exact name plus a channel suffix (a probe attached
as `default/emotions` is gated as `default/emotions@happy`).

| Shape | Channel (`readout.types`) | Produced by |
|---|---|---|
| `p`, `a.b`, `p[i]` | `Axis(i)`; bare = axis 0 | geometry (one key per coord axis, domain-frame); `jlens/<w>` / `jlens/#id` (mean fitted-layer probability); `sae/<id>` (normalized strength). Lens and SAE emit axis 0 only; a higher index validates but never fires. |
| `p:fraction` | `Fraction` | geometry: share of the centered activation in the subspace, [0, 1] |
| `p:membership` | `Membership` | geometry: tube-fit density, [0, 1] |
| `p@label` | `Distance` | geometry: negated distance in typical label spacings (larger = closer); only for nodes in the reading's `nearest` list; `neutral` names the frame anchor |
| `p~label` | `Assignment` | geometry: soft-assignment probability, [0, 1] |

Channel semantics and units are owned by `drowse/readout/AGENTS.md`. A dotted
probe name takes only the axis forms (`a.b:fraction` and `a.b@x` are parse
errors). Evaluation (`Trigger.active`): a missing key is inactive, never an
error. The decode loop replaces `probe_scores` after every forward, so a gate
on forward t reads the reading scored after forward t−1 (the prefill for the
first decode step). Validation happens at generation preflight in
`ReadPlane.resolve_gates` (`generation/read_plane.py`): a gate naming a probe
attached to the lens or SAE family on a channel that family cannot produce
raises `UnsupportedProbeChannelError` (400); geometry accepts every channel;
a gate naming no attached probe is silently inactive.

### Round trip

`format_expr` renders the canonical form: resolved registry keys (`0.5 pirate`
→ `0.5 default/personas%pirate`), `%g` numbers (six significant digits, so
lossy beyond that), no tag for `Trigger.BOTH`, canonical preset spellings
(`before`, `response`), `!x` / `-!x` for ±1 ablations, and a `%` term's
`onto` rendered relative to `along`'s sign so mixed-sign terms (the loom
`inverted` regen negates only `along`) re-parse exactly.
`parse_expr(format_expr(s)) == s` holds for every IR the parser produces and
every trigger built from the presets, `first`/`after`/`when`, or
`replace(preset, gate=…)`. Any other hand-built `Trigger` has no grammar form
and `format_expr` raises `SteeringExprError`; because results stamp
`applied_steering = str(steering)` (as do loom recipes and scoring), such a
trigger fails there. `referenced_selectors` returns the pre-resolution concept
atoms `(namespace, concept, variant)`, skipping `%` terms (pack-install checks
in `cli/config_file.py`).

The browser edition reimplements this grammar.
`tests/fixtures/browser_steering_expression.json` (accepted expressions with
their `referenced_selectors`, plus rejected ones) is pinned by
`tests/browser/test_browser_parity.py`; a grammar change updates the fixture
and is an interface change for drowse.ai.

## Names, atoms and resolution

- **Canonical names** (`drowse/naming.py`): `canonical_concept_name` lowercases
  and slugs `[^a-z0-9]+ → _`; bipolar poles join with `BIPOLAR_SEP = "."`
  (`formal.casual`), and a pre-joined `a.b` slugs each side. `.` is the
  separator because `~` is the projection operator and HF repo names reject
  it. `NAME_REGEX = ^[a-z][a-z0-9._-]{0,63}$` (`base/integrity.py`) governs
  names and namespaces; node labels use `LABEL_REGEX` (no `.`) so `%label` and
  bare labels stay unambiguous; `@` is forbidden (HF revision separator).
- **Reserved namespaces** (`RESERVED_NAMESPACES`: `jlens`, `slens`, `sae`,
  `neuron`, `head`) name instrument units: the name is kept verbatim and
  case-preserving (`jlens/Paris` ≠ `jlens/paris`), takes no `:variant` (not
  even `:raw`), and skips every manifold tier. The session wires resolvers for
  `jlens` and `sae` (`ArtifactRegistry.set_atom_resolver`); `slens`, `neuron`
  and `head` parse but raise `ProfileNotRegisteredError` when steered.
- **Variants**: `raw` (default), `sae` / `sae-<release>`, `role` /
  `role-<id>` (role-augmented fit; its role becomes the scope's assistant
  label), `from` / `from-<src>` (cross-model transfer). There is no `pca`
  variant. Non-`%` atoms are checked against `VARIANT_SUFFIX_RE` at parse
  time; a `%` operand's variant is checked when the manifold loads.

Plain-term resolution in `_fold`, in order:

1. Reserved namespace → key `ns/name`.
2. `profile_names` override: if the canonical key exactly matches an in-memory
   profile (`open_scope` passes the registry's names: extracts, bakes,
   instrument atoms, and any 2-node manifold the registry already folded as an
   `!`/`~`/`|` operand), it is a plain vector. So `0.5 formal.casual` is a `%`
   push in a fresh session but a folded-vector push after
   `session.extract("formal.casual")` or `!formal.casual`, and the two weight
   layers differently (folded `mahalanobis_share` versus the pole's
   neutral-anchored coordinate).
3. `store.selectors.resolve_bare_atom`: **label tier** (bare slug only: no
   typed namespace, `raw`, no `.`) → a unique node label across installed
   manifolds becomes `<ns>/<manifold>%<label>`; **name tier** (`raw`) → a
   2-node `pca` manifold name becomes `%` toward node 0, the + pole;
   **pole tier** → `canonicalize_atom` (slug + variant), a plain vector key.
   Collisions in either manifold tier raise `AmbiguousSelectorError`. The
   ladder reads a memoized walk of installed manifold folders, so labels
   resolve whether or not the manifold is fitted for the loaded model.

Projection operands and ablation targets skip both manifold tiers: they must
fold to a single direction (a 2-node `pca` manifold name, an instrument atom,
or an in-memory profile), so write `!formal.casual`, not `!formal`. A `%`
left operand is a manifold name verbatim: no slugging, no pole resolution.
Keys carry `<ns>/` only when typed and `:<variant>` only when not `raw`.

## Composition (`composer.py`)

`SteeringComposer` reaches the session only through its constructor
arguments: the `ArtifactSource` and `NeutralSource` protocols (the session's
`ArtifactRegistry` and `NeutralBaseline`), the manager, the exclusive
generation lock, the event bus, and two callbacks.

**Resolution.** `open_scope(value)` parses (with `profile_names`), then
`resolve()` concretizes and loads every artifact: `concretize_key` pins a bare
`:sae`/`:role`/`:from` to the one fitted variant (`UnknownVariantError` 404 /
`AmbiguousVariantError` 400); `ensure_profile_registered` returns an in-memory
profile, an instrument atom, or a fitted 2-node manifold folded to its
direction; `ensure_manifold_loaded` loads `%` targets. Unknown names and labels
raise here, before the scope is entered. `resolve_scope_role` then picks the
scope's assistant-role label: explicit `:role-<id>` keys must agree (else
`SteeringExprError`), plain keys mixed in warn (`RoleBaselineMismatchWarning`);
each `%` term implies its nearest node's role, disagreements resolve to the
highest `|along|` with a warning, and an explicit role beats an implied one.

**Scopes and the stack.** Entering a `SteeringScope` pushes its frame (and
role) and reinstalls the flattened head: later frames overwrite equal keys,
and the innermost frame naming a role sets `active_role`. `push` refuses to run
from inside a generation callback (`generation_phase()` non-None →
`ConcurrentGenerationError`, 409), then takes the generation lock, so
another thread's push waits for an in-flight generation. It is transactional:
if composition or install raises, the frame pops and the outer plan stays
installed, so composition errors surface on `__enter__`, not at `open_scope`.
`pop` is fail-closed: if the outer head no longer composes, all steering is
cleared before the error propagates. `pop(internal=True)` is the generation's
own teardown and bypasses the phase guard. Each change emits
`SteeringApplied` (or `SteeringCleared` when the stack empties), and a head
that steers the prefill calls `on_prefill_steering_change` (the session drops
its unsteered prefix cache).

**`compose(entries)`** builds a fresh plan without touching installed state.
It calls `neutral.ensure()` (the first steer on a fresh model may build the
neutral baseline) and needs the means plus a whitener covering every steered
layer (`WhitenerError` otherwise; there is no Euclidean path).

| Entry | Lowering |
|---|---|
| `(α, trigger)` vector | profile → `fold_directions_to_subspace` (neutral-anchored rank-1 ray, + pole at `‖d_L‖`) → `affine_manifold_push(folded, "+")`, coefficient α |
| `ProjectedTerm` | `project_profile(base, onto, op, whitener=…)`, memoized on the identities of base, onto and whitener and never registered → folded and pushed like a vector |
| `AblationTerm` | target profile's per-layer direction(s) → ablate fragment with its coefficient |
| affine `ManifoldTerm` | `affine_manifold_push(manifold, position)`, coefficient `along` (`onto` is vacuous and dropped) |
| curved `ManifoldTerm` | `SteeringPlan.add_manifold` (its own term) |

Push and ablate fragments group by trigger equality; each group becomes one
`synthesize_subspace` call (one merged affine subspace per layer, push wins a
direction shared with an ablation) registered as `__affine__<i>`. Distinct
triggers produce distinct groups, each with its own share normalization.
`affine_manifold_push` with a label uses that node's per-layer `node_coords`;
with coordinates it blends the per-layer coords by cardinal RBF weights over
the shared layout (exact at nodes). Coordinate arity raises
`ManifoldArityError`; a non-poised layout raises `SteeringExprError` advising
the label form.

**Plan cache.** `plan_for` keys on `tuple(entries.items())` (entries are
frozen) in an LRU of `_PLAN_CACHE_SIZE = 16`, enough for the push/pop pair of
each `generate(steering=…)`. A hit is reused only while every registry object
composition read is still the identical object (`_CachedPlan.deps`;
registries replace values, so a re-extract, eviction, whitener rebuild or
neutral refresh misses). Plans memoize their lowering per `(device, dtype)`
and their static offsets.

**Stack queries.** `gate_refs()` → `(scalar key, GateRef)` pairs for the read
plane; `steering_needs_probe_gating()` → the engine wires the gate score
callback only when true; `steering_active_in_prefill()` (a term steers prefill
iff `trigger.prompt and trigger.gate is None`) gates prefix-cache reuse;
`steering_value_prefill_inactive(value)` answers the same for an incoming
value (batch prefill sharing; a malformed value reports active);
`snapshot_steering_alphas()` feeds result receipts (`along` for `%`);
`assert_unsteered()` raises `ConcurrentExtractionError` for artifact-producing
work inside a scope, whose hooks would bake into the artifact (injected into
extraction and fits, the lens fit and the SAE runtime; the neutral baseline
checks for an installed plan itself).

## Lowering and hooks (`hooks.py`)

`SteeringPlan.lower(device, dtype)` places curved groups first at each layer.
Two curved spans sharing a layer with max |cosine| above
`_CURVED_ORTHO_TOL = 1e-3` raise `OverlappingManifoldError` (422). Each affine
term then goes through `_lower_affine_subspaces`, the single affine lowering:
mean-1 share normalization, `_orthogonalize_affine_against` the curved span at
that layer (the curved term owns shared directions; κ is carried through the
re-orthonormalization; a fully contained span drops the layer),
`eff_along = share_L · _SUBSPACE_GAIN`, and `kappa = requested_κ / eff_along`,
so `eff_along · κ` is exactly the user's ablation coefficient and ablation
strength is independent of share and gain. Both the transient `AffineOp` and
the compiled offsets (`LoweredAffine.offset`) consume this result; tests pin
their parity.

### Gains

`_normalize_shares_mean1` makes `Σ_L share_L = n_layers`, so a 1-layer or
30-layer fit, 4-dim or 16-dim, lands in the same α band. The raw share is
`SynthesizedSubspace.share` (the whitened push displacement `‖Δ_L‖_M`; a
positive proxy on pure-ablation layers) or a curved fit's baked
`mahalanobis_share`, which must cover exactly its fitted layers.
`synthesize_subspace` makes each push target whitened-unit, so every target
gets the same whitened budget (`Σ_L eff_along_L = gain · n_layers`; rationale
in `ARCHITECTURE.md` §6.4). There is no lever or N correction and no clamp or
water-fill on affine `along`; a high-share layer is meant to overshoot,
bounded by the whitened-unit target.

| Constant | Value | Applies as | Calibration |
|---|---|---|---|
| `_SUBSPACE_GAIN` | 16.0 | affine: `eff_along_L = share_L · 16`, α already folded into the target | gemma-4-12b α sweep over `formal.casual%formal`, `personas%caveman`, `personas%hacker`: dead → expressive at `16·α ≈ 7–8` for every target, coherence ceilings about 2× apart (hacker ≈ 12, caveman ≈ 17, formal > 22). Coherence-first: α ≈ 0.5 lands ≈ 8, α ≈ 1.0 is the strong / over-steer zone. Prototype. |
| `_MANIFOLD_ALONG_GAIN` | 4.0 | curved: `eff_along_L = along · share_L · 4`; periodic `BoxDomain`: `min(1, along · 4)` on every layer | stateless gemma-4-12b `months_loop%january` sweep: `along = 1.0` is the vivid coherent sweet spot. `eff_along = 1` lands on the node; `norm_cap = 3·‖h‖` bounds overshoot. |
| `_MANIFOLD_ONTO_GAIN` | 0.5 | curved: `eff_onto_L = min(1, onto · share_L · 0.5)` | gemma-4-12b `emotions%dominant` sweep at along 0.3: onto ≈ 0.5 sweet spot, 1.0 strong but coherent. Too much `onto` erases per-token variation and loops. |

Curved `along` and `onto` are fractions: each caps at 1 before gain, and a
negative value raises `SteeringCompositionError` (422) in `add_manifold`
because a curved position has no opposite. Periodic fits drop share weighting
because share × gain would send some layers past the node and each would wrap
to a different one; consequently every `along ≥ 0.25` lands all layers on the
node. `fit_mode=auto` can resolve flat on one model and curved on another, so
the same negative `%` coefficient pushes away on one and raises on the other.
Recommended affine α is about 0.5, toward 1.0 for stronger expression.

### Hooks and installation

`AffineOp` applies `h += c − ((h·Bₐᵀ − μBₐᵀ) ⊙ s) Bₐ`, with the fixed push
`c = eff_along · target @ B` and the activation-dependent ablation restricted
to nonzero-κ rows `Bₐ` (`s = eff_along · κ`). A pure push is one `add_`; a
rank-1 ablation is a dot plus `addcmul_` (MPS dispatch makes this cheaper than
a tiny GEMM). Constants are built in fp32 and cast once to the residual dtype
(re-cast once if the residual is narrower). There is no norm cap on this path:
a flat fit has no RBF to extrapolate.

`SteeringHook.recompose` arms a layer. One always-active `AffineGroup` makes
the hook static: `hook_fn` is a single `AffineOp.apply` with no
`TriggerContext` read. Otherwise it walks the groups, skipping inactive
triggers (`_trigger_active` short-circuits on `is Trigger.BOTH`, and
`_canonical_trigger` interns equal-to-BOTH triggers at registration so that
identity holds). Curved groups run `subspace_inject` in fp32 with a
warm-started foot: the previous token's foot seeds one Gauss-Newton step; a
cold foot (first fire, or a shape change from a re-prefill or batch change)
seeds at the origin with `_MANIFOLD_COLD_GN_STEPS = 4`. The last position's
foot carries from prefill into decode; `reset_manifold_feet()` cold-starts
every generation.

`SteeringManager.install(plan, …)`: when the session adopted compiled offset
buffers (`install_persistent_offset_hooks` runs before `torch.compile` on
CUDA/MPS; `adopt_compiled_offsets`) and `plan.static_offsets()` returns a map
(no curved term, every group `Trigger.BOTH`, every layer a pure push), it
writes the buffers in place (`copy_`; reassigning would change identity and
force a retrace; untouched layers are zeroed), attaches no transient hook and
sets `uses_compiled_offsets`. Otherwise it zeroes the buffers and attaches or
re-arms one transient `SteeringHook` per steered layer. Lowering and
validation run before any state changes, so a raising install keeps the
previous plan; `clear_all()` is the inverse, and an empty stack always
clears. Routing predicates for `session/engine.py`: `all_fast_path()` (no
transient hook: unsteered or on offsets, so the compiled graph is valid) and
`static_steerable()` (hooks attached, every one static, so StaticCache stays
eligible). On a compiled session any transient hook routes to the eager
original module.

`TriggerContext` is `SteeringManager.ctx`. `generation/decode.py::generate_steered`
sets `is_prefill`, `thinking` and `gen_step` before each forward and replaces
`probe_scores` after it; hooks only read it.

Hot-path rules (gated by `tests/gpu/test_smoke.py::test_throughput_regression`,
steered ≥ 85% of vanilla tok/s): hooks write the layer output in place, never
call `.item()` or otherwise sync the host, and create nothing per call beyond
the op's own tensors. Validation, overlap checks and constant preparation
belong in compose or lower.

## Tests

All CPU, in `tests/steering/`: grammar and round trips
(`test_steering_expr.py`, `test_manifold_grammar.py`), triggers
(`test_triggers.py`), scopes on a real session and on
`tests/support/composer.py::SteeringRig` (`test_steering_context.py`,
`test_steering_context_ablation.py`), installed state and offset buffers
(`test_steering_install.py`), affine lowering, gain, κ compensation and
compiled/eager parity (`test_subspace_steering.py`), `AffineOp` against
`subspace_inject` (`test_hooks_triggers.py`), curved lowering and foot state
(`test_manifold_steering.py`). Kernel math lives in `tests/geometry/`.
