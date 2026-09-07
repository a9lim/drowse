# Topology validation — 5 September 2026

## Status

This report records the earlier periodic-chart implementation. The subsequent
[synthetic surface and quotient-domain update](SYNTHETIC_SURFACE_VALIDATION_2026-09-05.md)
adds native two-field surface evidence and seam-aware Klein/RP2 steering.
Synthetic R^n validation is the acceptance scope; real-model validation is no
longer a prerequisite for geometric detection.

The SAE source-contract assertion is fixed. Automatic topology discovery has
been hardened in Python and Rust/WASM, but it is **not an arbitrary-manifold
classifier** and remains disabled in the hosted production backend. Linear
fitting and explicitly authored geometry remain available.

The `torus-Tn` identifier is retained for artifact compatibility. It names a
proposed periodic parameterization, not a certified homeomorphism or a semantic
claim about a model's representations. A PCA or spectral fallback is an
approximation, not evidence that the source manifold is flat or contractible.

## Repairs

- SAE source listings omit absent optional metadata rather than returning keys
  with `undefined`. Real layer lists, description provenance, and explicit null
  description bindings survive. Tests exercise both the in-process contract and
  its JSON representation.
- The H1 counter no longer discards every finite bar. A small torus cycle can
  die inside the observation window without being noise. Lifetime and an
  interior connectivity margin screen candidates; these thresholds remain
  heuristics, not calibrated confidence probabilities.
- A triangle-budget failure now raises. Dropping filling triangles can invent
  cycles, so truncated reductions must never be treated as evidence.
- Constant quadratic forms in a distance-MDS span recover separate circle
  factors when an arbitrary eigensolver basis mixes them. The checked
  Laplacian fallback rejects vanishing/radially inconsistent eigenpairs and
  harmonic duplicates. Geometry checks reject collapsed neighborhoods,
  insufficient phase coverage, duplicate samples, and impossible ambient rank.
- The selector will not truncate an excessive H1 count into a supported torus
  dimension. A periodic fit must also beat a mean-only GCV prediction baseline.
  A plane can reconstruct a circle perfectly, so comparing only against flat
  reconstruction would incorrectly discard periodic structure.
- Explicit errors remain errors or unresolved diagnostics, not successful
  topology classifications. Existing authored geometry and the injection
  kernel are unchanged.

## Reproducible checks

`fixtures/topology-adversarial-v1.csv` is shared by the Python and Rust tests.
Its 20 cases run in original, reordered, and rescaled variants. Positive cases
also require recovered phase coherence above 0.95, with distinct matches for
the torus axes; merely returning the expected axis count is insufficient.

| Fixture family | What the tests establish |
| --- | --- |
| Circles, noisy circles, 6:1 ellipse | One periodic axis with accurate sampled phases |
| Product tori, including radii 1:0.3 and unequal sample counts | Two independent axes, without mixed-eigenvector folds |
| Very thin, disconnected 16×8 product-torus sample | Refusal, not a fabricated connected manifold |
| Sampled 3D doughnut tori | Currently unresolved; these are false-negative limitations, not successful detection |
| Wide and thin Möbius strips, cylinders | Tested collapsed-circle interpretations are rejected; surface type is not inferred |
| Fibonacci spheres and 50 seeded random spheres | Tested false periodic interpretations are rejected; no sphere classification claim |
| Open arc, grid, line | No periodic interpretation in the tested variants |
| Invalid matrices, duplicate points, invalid thresholds, exhausted budgets | Explicit validation or chart refusal |

The shipped WASM exports additionally run thin-torus, thin-Möbius, sphere,
reordering, and poor-prediction-baseline checks under Node's WebAssembly engine.
This is not a live WebGPU model experiment or browser visual QA.

Commands, from the repository root:

```sh
.venv/bin/python -m pytest tests/test_topology_adversarial.py tests/test_manifold_topology.py tests/test_browser_topology_rbf_fixture.py tests/test_manifold_discover.py tests/test_manifold_math.py tests/test_manifold_extraction.py -q
cargo test --release --manifest-path browser-runtime/fitting-wasm/Cargo.toml
cargo clippy --manifest-path browser-runtime/fitting-wasm/Cargo.toml --all-targets -- -D warnings
npm --prefix webui run build:fitting-wasm
npm --prefix webui run test:runtime
npm --prefix webui run build:hosted
```

Existing fitted artifacts are not rewritten or re-certified. Explicitly refit
old automatic fits with `force=True` / `-f`; do not assume a cached fit has
passed the new checks. No package version was changed.

## Required before general topology detection can ship

Verification used the verify-changes workflow to trace the SAE contract,
topology consumers, and artifact boundary. Final results for this change:

- 382 focused Python tests passed.
- 38 Rust tests passed, including 60 shared adversarial fixture variants;
  Clippy passed with warnings denied.
- The complete `test:runtime` command passed, including the new WASM checks.
- Hosted build, build isolation, preview-shell checks, and reproducible WASM
  asset verification passed. Svelte reported zero errors and zero warnings.
- The broader `npm run check` command is **not green**: its separate
  `color-contrast.test.mjs:148` favicon/interaction-accent assertion fails.
  Those styling files were not changed for this task.

### Remaining implementation and validation work

1. A richer, validated representation: higher homology, boundary and
   orientability evidence, and compatible charts/atlas transitions. H1 alone
   cannot distinguish a circle, cylinder, and Möbius strip. The current global
   box/sphere domain choices are not an arbitrary-topology atlas.
2. A larger-sample, bounded sparse-persistence/circular-coordinate pipeline.
   The current 128-node ceiling cannot resolve every thin or long torus.
   Missing geometric features must lead to abstention, not inferred certainty.
3. Independent held-out geometry benchmarks: varied nonuniform sampling,
   missing regions, contamination, aspect ratios, and embeddings. The included
   regression fixtures helped develop these guards and are not an independent
   validation set. Thresholds require sensitivity analysis and false-positive
   calibration before a confidence score is meaningful.
4. Browser/runtime parity and validated chart recovery for any newly enabled
   automatic domain. Native synthetic diagnostics alone do not implement the
   compiled browser domain operations. Real-model semantic/behavioral claims
   are outside the requested synthetic-geometry acceptance scope.

Finite-sample recovery needs sampling and geometric assumptions; it does not
identify every arbitrary manifold from an arbitrary point cloud. See
[Niyogi, Smale and Weinberger, homology recovery from samples](https://math.uchicago.edu/~shmuel/NSW1.pdf).
For principled circle-valued coordinates rather than arbitrary eigenpair
angles, see [de Silva, Morozov and Vejdemo-Johansson](https://www.sci.utah.edu/~beiwang/teaching/cs6170-spring-2017/SilvaMorozovJohansson_2011.pdf).
