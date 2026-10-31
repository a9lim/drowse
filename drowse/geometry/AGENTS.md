# drowse/geometry

Activation-space geometry: the manifold/subspace representation behind every
steering term and geometry probe, the one steering kernel, the RBF surface and
its penalized fit, coordinate discovery and topology selection, the Mahalanobis
whitener, the direction view (fold, `Profile`, GGUF) and cross-model transfer.

Pure tensor math. Geometry sits directly above `errors`/`naming`/`events` and
`base` in the layer order and imports nothing else from drowse
(`tests/invariants/test_layering.py`; `drowse.__version__` is exempt). The
manifold tensor codec is `store/manifold_tensors.py`; capture and fit
orchestration are `fit/` (`pipeline.py`, `node_stats.py`,
`capture.py::compute_dls_axes`); the gains that scale the kernel are in
`steering/hooks.py`. The only file I/O here is the `Profile` safetensors/GGUF
codec.

## Module map

Internal order: `domains`, `rbf` → `subspace` → `invert`, `manifold`, `tube` →
`inject`, `topology`, `fold`, `transfer`. `packing` sits under `manifold`;
`whitener`, `procrustes`, `gguf` and `profile` stand alone.

| Module | Owns |
|---|---|
| `domains.py` | `ManifoldDomain`; `BoxDomain`/`BoxAxis`, `SphereDomain`, `KleinBottleDomain`, `ProjectivePlaneDomain`, `CustomDomain`; the `manifold.json` domain union (`validate_domain_spec`, `normalize_domain_spec`, `domain_from_spec`). |
| `rbf.py` | `RbfFitPlan` (the one layout factorization: exact, fixed-λ, GCV), `RbfFit`, `eval_rbf`/`eval_rbf_jacobian`/`eval_rbf_jet`, `rbf_cardinal_weights`. |
| `subspace.py` | `LayerSubspace`, `pca_basis`, `fit_affine_subspace`, `fit_curved_subspace`, `LayerFit`, `subspace_share`, `DEFAULT_N_COMPONENTS = 64`. |
| `manifold.py` | `Manifold`, `MANIFOLD_FIT_POLICY_VERSION`. |
| `packing.py` | `pack_to_device`: moves an object tree's tensors to a device as one buffer per dtype (what `Manifold.to` and probe attach use off the CPU). |
| `invert.py` | `_gn_step` (the one damped Gauss-Newton step), `invert_parameterization`, `neutral_origin`. |
| `tube.py` | The σ-field: `off_surface_vars`, `fit_sigma_field`. |
| `inject.py` | `synthesize_subspace`/`SynthesizedSubspace` (composition) and `subspace_inject` (the kernel). |
| `topology.py` | `derive_pca_coords`, `derive_spectral_coords`, `discover_coords`, `neutral_layout_coord`, `select_topology`. |
| `whitener.py` | `LayerWhitener`, `DEFAULT_RIDGE_SCALE`. |
| `fold.py` | `fold_directions_to_subspace`, `folded_directions`, `is_foldable_vector_manifold`, `project_profile`. |
| `profile.py` | `Profile`, `save_profile`/`load_profile`, `linear_sum`. |
| `gguf.py` | llama.cpp control-vector codec, `GGUFNotInstalled`. |
| `procrustes.py` | `fit_alignment`, `LayerAlignment`, `alignment_quality`, `AlignmentError`. |
| `transfer.py` | `transfer_manifold_subspaces`, the `LayerMap` protocol. |

## Rules

- **Mahalanobis only.** Every activation-space metric is the whitener's: the
  Fisher basis, share bake, steering synthesis, `~`/`|`, `Profile.cosine_similarity`,
  transfer re-bake, and (in `readout/`) monitor reads and `manifold compare`.
  Consumers check `whitener.covers_all(layers)` for every layer they touch and
  raise `WhitenerError` otherwise. There is no Euclidean fallback: on real LMs
  the Euclidean metric is dominated by a few massive-activation ("rogue")
  channels, so it is a wrong answer, not a degraded one. The deliberate
  Euclidean code: `pca_basis` without a whitener (the behavior-space fit in
  `fit/naturalness.py`, which lives in output-distribution space, and tests),
  `Profile.projected_away` (plain Gram-Schmidt, no in-repo caller), and a
  whitener layer whose neutrals are exactly zero (λ = 1, Σ_reg = I).
- **μ-center the scatter, never anchor-center it.** `pca_basis` takes centroids
  minus their own mean. Centering on neutral injects the neutral offset as a
  spurious axis and breaks PCA@2 ≡ difference-of-means.
- **Every fitted frame is neutral-anchored.** `mean = P_basis(ν_L)` and reduced
  coordinates are measured from it, so coordinate 0 is neutral. Only the origin
  is neutral; the basis comes from the μ-centered scatter.
- **Fit-time math is CPU fp32** (whitener, RBF saddle, PCA, topology). Norms are
  fp32 everywhere: fp16 sum-of-squares overflows at hidden sizes ≥ 2048.
- **Hot-path code never syncs the host.** `subspace_inject`, `eval_rbf*`,
  `_gn_step` and `LayerSubspace.sigma_at` run inside forward hooks: no
  `.item()`, no `float(tensor)`, no Python branching on tensor values. Known
  exceptions, both curved n ≥ 3 only: `_gn_step` uses `torch.linalg.solve` (its
  singularity check syncs) and `_principal_pairs_svd` hops to CPU on MPS.
- **Errors.** Request-reachable failures raise typed errors (`drowse/errors.py`,
  plus `AlignmentError` and `GGUFNotInstalled` here). A bare `ValueError` means
  a broken internal invariant and surfaces as a scrubbed 500.
- **`MANIFOLD_FIT_POLICY_VERSION`** stamps every fitted sidecar. Bump it when
  PCA/Fisher selection, topology choice, RBF or σ fitting, DLS or share
  allocation changes incompatibly. A bump makes every existing fitted tensor
  refuse to load and misses every fit cache (`fit/preflight.py`,
  `fit/pipeline.py`, `store/manifold_tensors.py`).
- **Browser seam.** drowse.ai's WebGPU runtime replays this package from the
  goldens in `tests/fixtures/browser/` (Fisher fit, DLS, affine lowering,
  RBF/topology parity, topology adversarial set), recomputed by
  `tests/browser/test_browser_parity.py`. A semantic change to the kernel,
  `synthesize_subspace`, `fit_affine_subspace`, `subspace_share`, `RbfFitPlan`
  or the topology functions that test imports regenerates the golden in the
  same change.

## Frames: `LayerSubspace` and `Manifold`

`LayerSubspace` is one layer's frame: `mean (D,)`, orthonormal `basis (R, D)`,
unit-box normalization `coord_offset`/`coord_scale (m,)`, and one of two exact
shapes (`validate_structure`):

- **Affine (flat)**, `is_affine ⇔ node_params is None`: `node_coords (K, R)`
  holds the real neutral-anchored per-layer node positions (the steering
  target), and `affine_map (m, R)` is `None` (identity) except after a
  rectangular transfer. `select_axes(kept)` bakes a DLS prune into the basis
  (affine only; a curved prune would need an RBF refit). `rbf_params()` raises,
  keeping flat frames out of RBF code.
- **Curved**: `node_params (K, m)`, `rbf_weights (K, R)`, `poly_coeffs (m+1, R)`.
  Raw-space curved subspaces also require the σ pair (`sigma_rbf_weights (K, 1)`,
  `sigma_poly_coeffs (m+1, 1)`); SAE-space curved fits have none.

`Manifold` adds the domain, `node_labels` and `node_coords (K, n)` (the shared
authoring/display layout), `node_roles`/`node_kinds` (exactly K; kinds are
generation-time provenance only), `feature_space` (`raw` or `sae-<release>`,
already decoded to model space), `mahalanobis_share` (one finite positive
scalar per fitted layer) and `origin` (the neutral foot of each curved layer,
exactly those layers). A fit is all-affine or all-curved.
`validate_runtime_geometry` closes the persisted contract over programmatically
built objects before a hot path sees them. `nearest_node_index`/`nearest_node_role`
use chordal distance between embedded coordinates, so periodic axes wrap.

## Domains

A domain embeds n authoring coordinates into R^m (`embed`, `embed_jacobian`,
`clamp_position`), supplies the foot-solve frame (`tangent_jacobian`,
`retract_position`) and implements `translate_foot`, the kernel's `along` step:
every token's foot moves by the same neutral→target offset, parallel-transported
to the foot.

| Domain | Geometry | Notes |
|---|---|---|
| `BoxDomain` | intervals × circles (box, cylinder, n-torus) | Periodic axes embed as `(cos, sin)` and translate along the short arc. Open axes clamp to `[lo, hi]` (default `[0, 1]`), so authored coordinates must sit inside the bounds. |
| `SphereDomain` | Sⁿ ⊂ R^(n+1), hyperspherical angles | Chordal metric (chordal r³ stays conditionally positive definite); log/exp translate with parallel transport. |
| `KleinBottleDomain` | flat quotient `(u + 2π, v) ~ (u, −v)` in R⁴ | Quotient-aware lift; cut-locus ties break deterministically. |
| `ProjectivePlaneDomain` | RP² via the Veronese embedding in R⁵ | Spherical angles of a canonical representative; foot steps in a cover frame. |
| `CustomDomain` | identity embed and clamp | Carrier for discover coordinates and synthesized affine subspaces. `bounds` are serialized for UIs; clamping ignores them. |

Sphere, Klein and projective are authored-only. Klein and projective
`project_mean` map an extrinsic mean back onto the domain; `readout/monitor.py`
uses them (and its own sphere and periodic-box cases) to average a coordinate
across layers. `validate_domain_spec` accepts only the exact current wire
shape; `normalize_domain_spec` fills authoring defaults first.

## Fitting a frame

`pca_basis(X, whitener=, layer=)` with a covering whitener solves the Fisher
(LDA) problem `(S_b, Σ)`: eigenvectors `a` of `G = X Σ⁻¹ Xᵀ` give directions
`Σ⁻¹ Xᵀ a`, re-orthonormalized by QR so the runtime sees a Euclidean-orthonormal
basis. Dividing by background variance cancels the rogue channels raw PCA
chases, the same cancellation DiM gets by differencing.
`R = min(n_components, K − 1, rank)`. `whitened_gram`/`whitened_rows` let one
Woodbury apply serve Gram, basis and layout anchoring; `basis_override` reuses
the Fisher bases `select_topology` computed.

- `fit_affine_subspace` (flat: `fit_mode=pca`, every steering vector as K = 2):
  `node_coords = (centroids − ν)·basisᵀ`; `orient_to=0` makes node 0 project
  non-negative, so a 2-node axis is `+δ̂`.
- `fit_curved_subspace` (authored/spectral, K ≥ 3): the same frame plus an RBF
  to the neutral-anchored reduced coordinates. `smoothing=None` is exact (the
  authored contract: a node is an exact steering target); `"auto"`/float fits
  the penalized surface and records λ/edf/GCV on `LayerFit.rbf`.
- `subspace_share = sqrt(Σ_k c_kᵀ (B Σ⁻¹ Bᵀ) c_k)` over μ-centered coordinates:
  the per-layer budget is signal spread, independent of where neutral sits. At
  K = 2 it is `‖δ_L‖_M / √2`, so the mean-1 profile equals DiM's. The monitor
  weights its cross-layer read by the same share (normalized to sum 1).

## RBF surface

One `r³` polyharmonic RBF plus an affine polynomial per layer: conditionally
positive definite of order 2 in every dimension, C², the natural cubic spline at
n = 1. `RbfFitPlan.build(node_params, smoothing=)` depends only on the layout,
so a multi-layer fit factorizes once and each layer (and the σ-field) is
right-hand-side work. It normalizes to the unit box (mandatory: the kernel
amplifies scale), raises `PoisednessError` unless K ≥ m + 1 and the nodes
affinely span R^m, and caches the saddle LU (symmetric indefinite, so LU, never
Cholesky) at its own fixed λ.

The penalized fit is the thin-plate/Duchon spline `[[E + λI, Q], [Qᵀ, 0]]`. λ is
scaled by the mean off-diagonal kernel magnitude, so a float `smoothing` is
scale-invariant. `"auto"` minimizes the multi-output GCV
`K·‖(I − S_λ)Y‖² / tr(I − S_λ)²` over 40 log-spaced points in `[1e-6, 1e3]`
through the Demmler-Reinsch eigensystem; λ = 0 is excluded and ties go to the
smallest λ. Smoothing changes coefficients only, never the evaluators.

`rbf_cardinal_weights(node_coords, query)` gives exact interpolation weights
(exact at nodes, a partition of unity); the composer uses them for coordinate
`%` on a flat manifold (per-layer target `node_coords_Lᵀ @ w`).

## The σ-field

`fit_sigma_field` turns each node's reduced within-node covariance
(`fit/node_stats.py::compute_store_reduced_covariances`) into one off-surface σ:
`off_surface_vars` projects onto the normal complement of the *actual* local
tangent range (one economy SVD gives rank and projector, so folds with
rank < n stay honest) and divides by the normal degrees of freedom. σ² is
floored at `1e-3 × median`, and `log σ` gets its own RBF over the same layout,
smoothed (GCV unless the fit pins λ) because a second moment from about 48
samples is noisy. The field is isotropic. `onto` steers into it; the monitor
reads it as `:membership`.

## Foot solve

`_gn_step` is one Levenberg-Marquardt step of `argmin_p ‖s(p) − q‖` with the
analytic RBF Jacobian chained through `tangent_jacobian` and a diagonal floor
for folds; n ≤ 2 solves in closed form so it never syncs.
`invert_parameterization`: cold, 12 iterations from the 3 nearest nodes, best
residual wins (guards periodic and fold minima); warm (the monitor's follower),
4 iterations from the carried foot plus the nearest node. `neutral_origin`
projects ν onto a curved layer's surface. Origins are per layer because each
layer embeds the coordinates differently, and the steering follower takes one
step with no restarts, so its seed must already be in the right basin.

## Composition and the kernel

`synthesize_subspace(push, ablate, neutral_means, *, whitener)` composes every
affine term of one trigger group (vectors, poles, `~`/`|` results, `!`, affine
`%`) into one `SynthesizedSubspace` per layer, so terms sharing a layer
superpose instead of colliding. On CPU fp32, per layer:

- Push rows `(basis_rows, coord_target, coeff)` and ablation rows `(dirs, coeff)`
  are orthonormalized push-first (`_ortho_basis`); push wins a shared direction.
- The leftover ablation operator `A = Σ coeffᵢ uᵢuᵢᵀ` is diagonalized, so
  partial, repeated, negative and non-orthogonal ablations get exact per-axis
  `kappa` (0 on push axes).
- **Whitened along-normalization.** `target = B @ Σ coeffᵢ·wdᵢ/‖wdᵢ‖_M`, and
  `share = ‖Δ‖_M` of the raw coefficient-weighted displacement. Every target
  gets the same whitened budget (`Σ_L eff_along_L = gain·n_layers`, linear in
  α), spread by where its signal lives (the rationale is `ARCHITECTURE.md`
  §6.4). A pure-ablation layer gets a positive proxy share, which the hook's κ
  compensation (`κ / eff_along`) cancels.
- Missing neutral means or partial coverage raise `WhitenerError` up front. An
  empty result is the explicit no-op. Finished layers move to the device in one
  packed transfer per device.

`subspace_inject(h, subspace, domain, target, foot_seed, along, onto, …)` splits
`h = mean + H_m + H_n + H_o` (surface foot, off-surface in-subspace residual,
off-subspace residual) and works in reduced coordinates (orthonormal basis, so
norms are exact and cost is O(R)):

- **along** translates the foot by `along·(target − origin)` via
  `domain.translate_foot`. A fixed offset preserves the per-token in-subspace
  spread; lerping each foot onto the absolute target erases it and
  loop-collapses at strong push. On the curved path `H_n` then moves to the new
  tangent frame by the minimal principal-angle rotation
  (`_frame_rotation_transport`): norm-preserving, lossless, and exactly the
  identity when the foot does not move, so `along = 0` returns `h` whatever the
  foot-solve accuracy (a reprojection would leak the residual's tangential part
  at an approximate foot). Frames use modified Gram-Schmidt (`linalg.qr` is
  unimplemented on MPS); principal pairs are closed-form on device for n ≤ 2.
- **onto** then shrinks `H_n`: by `(1 − onto)` without a σ-field (onto = 1
  lands on the wire), or toward σ(z) with one (onto = 1 lands one σ off the
  wire, on the typical set; direction kept; a residual inside the tube is never
  expanded). Order is fixed: along, then onto.
- **`H_o` is kept verbatim.** It is every composing neighbor's span, so a vector
  and N orthogonal manifolds compose with zero cross-talk.
- **Affine shortcut.** On a flat subspace the foot is `q`, `H_n ≡ 0`, `onto` is
  vacuous: `p_new = q + along·(target − κ·q)`, `h_new = h + (p_new − q)·B`. No
  foot solve, no RBF, no norm cap (a flat fit cannot extrapolate).
- **Curved norm cap** `‖h_new‖ ≤ norm_cap·‖h‖` (3) guards off-domain RBF
  extrapolation. When it fires it scales the whole output, `H_o` included.

The kernel returns fp32 (the hook's `copy_` downcasts) plus this token's refined
pre-slide foot, which seeds the next token. It applies no gain and no clamp:
per-layer `eff_along`/`eff_onto`, share normalization, `_SUBSPACE_GAIN`,
`_MANIFOLD_ALONG_GAIN`, `_MANIFOLD_ONTO_GAIN`, the periodic clamp and the
cold-start step count belong to `steering/hooks.py`. At runtime only curved
terms call `subspace_inject`. Affine terms run `steering/hooks.py::AffineOp`, a
compose-time split of the affine branch that
`tests/steering/test_hooks_triggers.py::test_affine_op_matches_unified_kernel`
pins equal to it, so change the two together.

## Discover coordinates and topology

Discover coordinates are layer-agnostic: the pipeline averages every fit
layer's whitened, node-mean-centered `(K, K)` Gram into a consensus Gram, and a
layer that does not separate the nodes contributes nearly nothing.
`derive_pca_coords` keeps the smallest eigen-prefix crossing `var_threshold`
(0.70), capped at `max_dim` (8). `derive_spectral_coords` builds a union k-NN
heat-kernel graph on the same distances (`k_nn = max(5, ⌈log K⌉)`, bandwidth =
median neighbor distance) and picks k at the eigenvalue-*ratio* cliff
(absolute gaps grow like k² on S¹ and over-pick); `min_dim` floors the pick
because one dominant Fiedler mode makes the cliff undershoot. Spectral needs
K ≥ 4 and a connected graph and is noisy below about 50 nodes.
`neutral_layout_coord` locates ν in a flat layout by landmark MDS; subtracting
it makes `% 0,…,0` read as neutral without changing steering.

`select_topology` (`fit_mode=auto`) makes two decoupled decisions, because one
reconstruction score always crowns the highest-dimensional candidate:

1. **Flat vs curved.** Summed per-layer GCV (`K·RSS/(K − edf)²`) of the flat OLS
   fit and the penalized curved fit, both predicting the same Fisher-reduced
   targets. The curved candidate is floored to the flat dimension: starved of
   coordinates, a curved manifold in a k-plane loses to the flat fit it beats at
   matched dimension.
2. **Periodic axes.** Vietoris-Rips H1 over a window of twice the longest MST
   edge (exact GF(2) reduction) counts persistent bars (`persistence_frac`,
   0.5). Charts come from a distance-MDS span (`_untangle_periodic_eigenspaces`
   separates equal-radius torus factors) or Laplacian eigenpairs that must be
   near-circular and not harmonics (`_is_angular_harmonic`), and must pass
   ambient-rank, phase-coverage and false-neighbor checks
   (`_periodic_neighborhoods_preserved`). When H1 counts nothing,
   `_faint_cycle_coords` tries a tour for a faint uniform ring or a clustered
   ring (months, weekdays). Its `_CYCLE_*` guards trade two known false
   negatives (a very loose cluster heap, an ellipse beyond 6:1) for near-zero
   false positives on grids, fans, arcs, lines and blobs.

A viable periodic candidate (finite GCV no worse than the mean-only baseline)
becomes a `BoxDomain` of 2π axes and wins outright, since a plane approximates a
circle but loses its closure. Otherwise curved wins only on strictly lower GCV.
`TopologyChoice` carries the ranked candidates for the sidecar plus the
winner's diagnostics, `RbfFitPlan` and Fisher bases. This proposes periodic
charts; it does not certify topology. H1 cannot tell a circle from a cylinder or
Möbius strip, and sphere sampling holes can pass it. Detection abstains above
K = 128; an exceeded 500,000-triangle budget is recorded as a non-viable
`periodic-unresolved` candidate. An abstention is not evidence of trivial
topology.

## Whitener

`LayerWhitener` keeps per layer the centered neutral rows `X (N, D)`, the ridge
`λ_L = ‖X‖²_F/(N·D) · ridge_scale` (the sample covariance's mean diagonal;
`DEFAULT_RIDGE_SCALE = 1.0`) and `K = (NλI + XXᵀ)⁻¹` by Cholesky (jitter retry).
`apply_inv` is `Σ⁻¹v = (1/λ)(v − Xᵀ K X v)`: O(ND), no D×D matrix, CPU fp32,
batched over leading dimensions. N is the neutral corpus size (a multiple of
the 48 baseline prompts), so Σ is rank-deficient and the ridge is required. A
layer whose centered rows or `K` are non-finite is left out, which is why
`covers_all` can be trusted as "finite factors everywhere". Built by
`store/neutral.py::load_neutral_whitener` from the fp32 neutral cache, or by
the session's lazy neutral state.

Primitives: `apply_inv`, `mahalanobis_norm(s)`, `mahalanobis_cosine`,
`leace_project` (Belrose et al. 2023; `~` keeps `⟨b, d⟩_M/⟨d, d⟩_M · d`, `|`
erases it), `subspace_gram(layer, B) = B Σ⁻¹ Bᵀ` (the reduced metric behind the
share and whitened reads) and `woodbury_factors` (device-resident factors for
the monitor's on-device per-token apply).

## Direction view: fold, `Profile`, GGUF

A steering vector is the R = 1 affine manifold. `fold_directions_to_subspace`
folds a per-layer direction (a merge, a `~`/`|` result, a registered profile, a
monopolar `concept − ν`) into a neutral-anchored one-pole ray: `basis = d̂`,
pole at coordinate `‖d_L‖`, share `‖d_L‖_M` (not the √2 form; one direction has
no node cloud), no DLS, no origin. `folded_directions` reads an R = 1 fit back
as `{L: δ̂_L · share_L}`: the `Profile` behind `extract()`, `manifold
compare`/`why`, merges and GGUF export, exact for scale-invariant reads and
proportional for magnitudes. `is_foldable_vector_manifold` gates its callers.
`project_profile` is the per-layer LEACE `~`/`|` (`|` passes layers missing from
`onto` through, `~` drops them, an empty result raises
`SteeringCompositionError`).

`Profile` wraps `dict[int, Tensor]`. On disk: fp32 `layer_<i>` safetensors plus
an exact five-field sidecar (`format_version`, `drowse_version`, `method` ∈
{`profile`, `manifold_pca`, `merge`}, `tensor_sha256`, JSON-safe `provenance`),
written atomically under `artifact_lock`. The version is
`base/integrity.py::PROFILE_FORMAT_VERSION`, matched exactly, no migration.
`load_profile` flattens `provenance`, so load → save round-trips. `linear_sum`
merges over the union of layers (`strict=True` requires equal coverage).

`gguf.py` reads and writes llama.cpp control vectors (`controlvector.model_hint`,
`direction.<L>` fp32; `gguf` is an optional lazy import).
`store/gguf.py::export_gguf_manifold` exports 2-node flat fits as
`folded_directions`. llama.cpp applies one uniform scale, so the exported
per-layer weighting is the Mahalanobis share: an approximation of drowse's
injection, which budgets by the whitened neutral→node displacement. Unit-norm
repeng files load and inject uniformly.

## Cross-model transfer

`fit_alignment` fits `A_L(x) = M_L x + b_L` per shared layer from matched neutral
activations (`fit/transfer.py` orchestrates; `store/alignment.py` caches). Equal
hidden sizes use orthogonal Procrustes factored through the two row spaces, so
no unobserved D − N complement is invented; unequal sizes use minimum-norm least
squares. `LayerAlignment` keeps rank-sized factors and applies the offset to
points only. Fewer than 10 shared layers raises `AlignmentError`.
`alignment_quality` is per-layer centered R²; its median is the sidecar's
`transfer_quality_estimate`.

`transfer_manifold_subspaces` maps means as points and bases as directions,
QR-reparameterizes back to an orthonormal basis and folds the companion matrix
into `affine_map` or the RBF coefficients, so world points are preserved. It
rejects rank-collapsing maps and, on curved layers, non-isometric companions (a
scalar σ cannot represent an anisotropic tube). It re-bakes the share with the
target whitener (required) and re-solves curved origins against the target
neutral means. Uncovered layers drop.

## Testing

`tests/geometry/` is CPU-only, one file per module family;
`test_manifold_math.py` holds the kernel identities (identity at along = 0,
`H_o` verbatim, norm cap, soft onto) and `test_frame_transport.py` checks the
closed-form n ≤ 2 principal pairs against SVD. Cross-package pins:
`tests/steering/test_hooks_triggers.py` (AffineOp equals the kernel) and
`tests/browser/test_browser_parity.py` (goldens).
