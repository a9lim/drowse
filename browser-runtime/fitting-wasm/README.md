# Drowse browser fitting kernel

This crate is an incomplete numerical foundation for browser-side,
non-training Drowse workflows. It has no filesystem, network, model-runtime,
or UI dependency. All inputs are row-major `f64` buffers. Public operations
reject shape mismatches, non-finite values, and requests beyond fixed output,
work-memory, or cubic-compute ceilings.

Implemented kernels:

- neutral-mean centering and application to new rows;
- full-dimensional Mahalanobis whitening for
  `X^T X / N + ridge * I`, represented by its low-rank covariance span plus
  the exact `1 / sqrt(ridge)` complement used by Python;
- neutral-anchored affine Fisher fitting from pooled node centroids and a
  serialized, already-verified per-layer whitener, including Python's
  whitened Gram, stable-rank policy, deterministic orientation, reduced node
  coordinates, explained-variance diagnostic, and Mahalanobis layer share;
- deterministic PCA with stable component ordering and signs;
- symmetric union-k-nearest-neighbor graphs, heat-kernel adjacency, normalized
  graph Laplacians, and Laplacian-eigenmap spectral coordinates;
- bounded Vietoris-Rips H1 persistence, guarded faint/clustered-cycle recovery,
  spectral periodic-axis coordination with harmonic rejection, and automatic
  flat-PCA versus curved-spectral versus periodic-torus selection;
- affine positioning plus explicitly named Euclidean projection and ablation
  primitives;
- explicitly named Euclidean Pearson-column correlations and pairwise
  distances;
- normalized `r^3` polyharmonic RBF interpolation, fixed smoothing, standalone
  automatic GCV selection, and serializable reusable fit plans over Python's
  40-point penalty grid, with the Python fp32 affine-poisedness policy;
- incremental per-node reduced covariance over bounded fp32 activation chunks,
  plus curved-surface tangent projection and fixed or automatic fuzzy sigma
  RBF fitting;
- stable sum- and mean-log-probability normalization for template scores.

The native functions in `drowse_fitting_wasm::kernel` are the source of truth.
The crate-root `wasm-bindgen` API is a thin typed-array adapter over those
functions. Matrix buffers are always row-major. PCA and whitener bases are
`components × feature_dimension`; score/output buffers are `rows × columns`.
`MahalanobisWhitener.transform` returns a full `rows × feature_dimension`
buffer: Euclidean dot products after that transform equal the regularized
Mahalanobis metric, including directions outside the neutral sample span.
`fitAffineFisher` consumes that same serialized metric without rebuilding a
neutral corpus. The browser coordinator composes it with DLS axis selection,
consensus layout anchoring, curved-winner RBF fitting, streamed raw covariance,
sigma-field fitting, and self-validated flat artifact publication. Production
capture and the transaction that installs fitted artifacts into the active
runtime remain orchestration release gates. Curved fits now include the
neutral nearest-point origin and self-validated curved artifact publication.
The topology result exposes the winning selector diagnostics without
reconstructing them from the chosen coordinates: all retained PCA variance
components and threshold, or the spectral eigenvalues, eigengap, bandwidth,
k-nearest-neighbor count, heuristic and selected dimensions, authored floor,
and pinned state. Fixed smoothing is scored through the fixed smoother; omitted
smoothing selects a penalty independently for every target layer using the
Python GCV grid. Exact smoothing (`0`) is rejected as an automatic curved
winner when its interpolation score is undefined.

The RBF model exposes the scaled penalty as `lambda`, effective degrees of
freedom, and its GCV score (`-1` for final fixed/exact fits); the caller's
smoothing multiplier is not stored under that name.

`prepareRbfFitPlan` validates and fp32-normalizes one node layout, then retains
its kernel, null-space spectral basis, grid ratios, and residual traces. The
plan exposes a validated serialization contract so a stateless worker can
restore the exact winning geometry. `RbfFitPlan.fitSmoothed` and
`fitAutoSmoothed` reuse those operators for every layer and sigma-field value
matrix without repeating the geometry decomposition or assuming that layers
share an output width. The native `kernel::RbfFitPlan` provides the same
operations. Standalone exact, fixed-smoothing, and automatic-smoothing
functions remain available.

`buildKnnGraph` accepts a validated symmetric pairwise-distance matrix and
retains an undirected edge when either endpoint selects the other among its
`k` nearest neighbors. Equal-distance ties use node id as a deterministic
browser-side tie break; Python's `torch.topk` tie order is unspecified, so
cross-runtime conformance is defined by graph and embedding invariants when a
fixture contains ties. `buildNormalizedLaplacian` reads distances from a
symmetrized fp32 Gram, uses the median retained-edge distance by default, and
returns the symmetric heat-kernel weights, degrees, normalized Laplacian, and
all nontrivial eigenpairs. `deriveSpectralEmbedding` applies Python's
eigenvalue-ratio dimension heuristic and optional authored-dimension floor.
Its coordinates are eigenvectors, so consumers and goldens must compare
pairwise distances or projection matrices rather than raw signs or bases.

`detectPeriodicTopology` proposes a periodic chart using the same bounded
heuristics as Python. Exact GF(2) Rips reduction supplies candidate H1 bars,
including sufficiently persistent finite bars. Distance-MDS quadratic forms
separate mixed circle factors; a checked Laplacian fallback rejects vanishing
or non-circular eigenpairs and harmonics. Ambient-rank, phase-coverage, and
neighborhood/collapse checks reject unsupported charts. The faint-ring tour
fallback remains heuristic, with an additional ambient-rank guard.

Neither H1 nor a passing chart check certifies a homeomorphism. An unresolved
result must not be interpreted as flat topology. In particular, spheres and
Mobius strips are not classified, and some sampled 3D or very thin tori remain
unresolved. Detection is capped at 128 nodes; exhausted triangle budgets raise
instead of silently truncating the complex. See
[validation scope and remaining work](../TOPOLOGY_VALIDATION_2026-09-05.md).

`selectTopologyFromTargets` consumes a consensus Gram plus concatenated
row-major `[node][reduced_dimension]` buffers and an element-offset table for
the layers. Per-layer reduced dimensions may differ, matching Python's stable
Fisher-rank behavior. The buffers must already be the shared
whitened/Fisher-reduced targets that Python's selector constructs. It
derives the flat PCA layout, floors the spectral dimension to the flat
dimension, compares OLS and RBF candidates with Python's GCV formulas and
40-point automatic-smoothing grid. A supported periodic candidate takes
priority over a linear approximation only if its GCV improves on a mean-only
prediction baseline. The result retains the curved winner's
serializable RBF plan; the coordinator restores and reuses it for each
variable-width layer and scalar sigma field under the same fixed/automatic
smoothing policy used during selection. Candidate scores and selector
diagnostics are cross-runtime golden-tested on non-degenerate fixtures. Eigenvector signs
and rotations, and therefore raw angle values on a repeated eigenvalue, are
compared only through topology and geometry invariants.

The Euclidean, spectral, and topology exports are bounded numerical primitives,
not complete Drowse profile projection, correlation, pairwise geometry, or
authoring operations. The topology selector deliberately starts after
whitened/Fisher target construction and stops before artifact publication.
Those surrounding operations remain release-gated because the Python pipeline
composes Mahalanobis/LEACE semantics, model-bound state, activation-row
transactions, and artifact identity.

Build and test:

```bash
cargo test --manifest-path browser-runtime/fitting-wasm/Cargo.toml
cargo clippy --manifest-path browser-runtime/fitting-wasm/Cargo.toml --all-targets -- -D warnings
cargo build --manifest-path browser-runtime/fitting-wasm/Cargo.toml --target wasm32-unknown-unknown --release
```

The hosted build packages this crate with the exact `wasm-bindgen-cli 0.2.127`
toolchain into `webui/public-hosted/wasm`. The generated manifest binds the JS
and WebAssembly bytes to the crate lockfile, and CI regenerates them before
accepting a change. `webui/src/hosted/fitting/fitting.worker.ts` loads those
bindings in a one-job worker, can read a committed activation layer directly
from OPFS, and terminates on cancellation or timeout. This proves the browser
transport boundary only; it does not enable the authoring capability.

See `RELEASE_GATES.md`. This crate must not enable the hosted fitting
capability or be represented as browser fitting parity.
