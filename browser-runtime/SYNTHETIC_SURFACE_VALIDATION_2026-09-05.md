# Synthetic surfaces and quotient steering

## Scope

Validation uses synthetic manifold-shaped data in R^n. Real-model experiments
are not required for this scope. No semantic or model-behavior claims follow
from these checks.

Two capabilities are deliberately separate:

- `polythetic.core.surface_topology.detect_surface(points)` measures evidence
  for a closed surface from unlabelled point coordinates. It does not receive
  the generating surface's name or parameter coordinates.
- `KleinBottleDomain` and `ProjectivePlaneDomain` provide native, seam-aware
  steering when the node coordinates are authored. A detected homology
  signature is not automatically converted into a coordinate chart.

The hosted app remains gated. Its archive validator and compiled hook ABI
support box, sphere, and custom domains, not these new quotient tags. No model
bundle, browser artifact, deployment, or saved user manifold was rewritten.

## Detection

The optional `topology` extra installs Ripser and SciPy. Detection computes
Vietoris–Rips H0, H1, and H2 over F2 and F3, using the full distance matrix of
32–384 finite, distinct points. Larger inputs raise instead of silently
subsampling away thin features. Inputs are explicitly Euclidean synthetic
coordinates; activation-space callers must supply a whitened representation.

Birth-normalized persistence compares features at their own scale. A dominant
F2 H2 bar must exceed competing bars by 3:1. Both fields must have a simultaneous
surface signature over a scale interval at least 15% of its starting radius.
Every barcode endpoint participates in the interval sweep; a coarse grid
cannot hide an intervening contradictory class. F3 is computed only after the
F2 screen passes. These are conservative heuristic thresholds, not calibrated
confidence probabilities.

| Conditional surface interpretation | F2 Betti numbers | F3 Betti numbers |
| --- | --- | --- |
| Sphere | 1, 0, 1 | 1, 0, 1 |
| Torus | 1, 2, 1 | 1, 2, 1 |
| Real projective plane | 1, 1, 1 | 1, 0, 0 |
| Klein bottle | 1, 2, 1 | 1, 1, 0 |

These signatures distinguish the listed **connected closed two-manifolds**.
Homology alone does not prove that an arbitrary point cloud samples such a
manifold, nor determine arbitrary topology. Boundary surfaces, intersecting
immersions, non-manifold complexes, missing regions, and insufficient sampling
remain outside a certified classification claim. `candidate=None` means
unresolved, not flat or contractible. Higher-genus signature names are
implemented but have not been positively validated here.

`scale_interval` is expressed in median-pairwise-distance units.
`relative_persistence` is interval width divided by its starting radius.

## Seam-correct native geometry

- Klein bottle: `(u + 2π, v) ~ (u, -v)`, with a crossing-free embedding in R4.
  Shortest deck lifts include the orientation-reversing seam, and tangent
  translation flips the second component when required. Its connection is the
  flat quotient metric, not the metric induced by the R4 embedding.
- Projective plane: antipodal unit vectors, represented by a trace-free
  Veronese embedding in R5. Paths select the nearest antipodal lift. Local
  two-dimensional tangent frames and sphere retractions keep the foot solver
  full-rank at spherical-coordinate poles.
- The existing RBF fit, unified injection kernel, off-subspace preservation,
  sigma-field calculation, and safetensors codec consume the new domains.
  Probe summaries average in embedding space and project back, rather than
  averaging across a coordinate seam. Projective means with tied top
  eigenvalues retain the reference coordinate.
- Cut loci have genuinely non-unique shortest paths. The implementation makes
  a deterministic choice; it does not promise globally continuous path choices
  or encode arbitrary winding-number instructions.

Native domain specifications are `{"type": "klein"}` and
`{"type": "projective", "dim": 2}`. Both use two authoring coordinates.
Their artifact tags are rejected by older/hosted readers rather than being
misinterpreted as periodic boxes.

## Reproducible validation

```sh
.venv/bin/python -m pytest tests/test_surface_topology.py tests/test_quotient_domains.py -q
.venv/bin/python -m pytest tests/test_topology_adversarial.py tests/test_manifold_topology.py tests/test_browser_topology_rbf_fixture.py tests/test_manifold_discover.py tests/test_manifold_math.py tests/test_manifold_extraction.py tests/test_manifold_monitor.py tests/test_manifold_steering.py tests/test_manifolds_io.py -q
.venv/bin/ruff check polythetic/core/manifold.py polythetic/core/monitor.py polythetic/core/surface_topology.py tests/test_quotient_domains.py tests/test_surface_topology.py
```

Regression cases include random spheres/RP2, stratified Klein/product-torus
samples embedded in 11 dimensions with shuffled rows and changed units, and
3D doughnut tori with major radius 2 and minor radii 0.6 and 0.3. These cases
informed development and are not independent holdouts. The 3D cases use 24×16
and 48×8 samples respectively, with parameter jitter.

After freezing the algorithm, seeds 10091 and 10093 test the four main families
in R17 with independent Gaussian coordinate noise of standard deviation 0.002.
These are held-out seed/noise variants of the same generators, not an
independently designed benchmark or broad noise-tolerance calibration.

Negative cases include a disk-like cloud, a cylinder, a Möbius strip, a
seven-dimensional cloud, disconnected spheres, and a very thin 1:0.025
product torus. Refusal on the thin torus is a recovery limitation, not a
successful identification. Geometry tests cover equivalent representatives,
seam continuity, Jacobians, poles, ambiguous means, batched injection,
off-subspace residual preservation, zero-strength identity, and native
save/load round trips.

Final local results:

- 41 surface-detection tests passed in 197.69 seconds, including all eight
  frozen noisy holdouts and all four jittered 3D doughnut-torus cases.
- 34 quotient-geometry tests passed in 0.95 seconds, including nonlinear RBF
  fits and preservation of normal-residual norms across seams.
- 652 related existing math, extraction, topology, monitor, steering, and
  artifact tests passed (run together with the first 30 quotient tests: 682
  passed in 21.14 seconds). One expected authoring-rank warning was emitted.
- Ruff, Python compilation, and targeted Pyright passed (zero errors/warnings,
  with the project interpreter selected). A local wheel built successfully and
  included the new module and optional dependency. No package version changed.

These timings are local test-run measurements, not browser latency guarantees.
The full unrelated Python/server/GPU suite and hosted runtime suite were not
rerun for this native-only update.

## Remaining work

Automatic chart recovery, the corresponding browser compiler/ABI operations,
and synthetic parity tests of those compiled operations are still outstanding.
The point-cloud diagnostic is opt-in Python, not connected to the existing
`fit_mode="auto"` selector. Do not advertise arbitrary-manifold detection or
enable hosted automatic steering on the strength of this native implementation.

References: [Ripser API](https://ripser.scikit-tda.org/en/latest/reference/stubs/ripser.Rips.html),
[Ripser algorithm](https://arxiv.org/abs/1908.02518),
[Hatcher, Algebraic Topology](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf).
