# Manifold steering validation — 2026-09-05

## Repairs

- **Cross-layer geometry:** periodic coordinates now use circular means, and sphere coordinates use an embedded-vector mean. Ordinary coordinates retain their weighted arithmetic mean. Previously, phases 0.99 and 0.01 averaged to 0.50: the opposite side of the circle. The repair covers browser readouts, native full/lean monitor readings, and gate scalars.
- **Ambiguous means:** when embedded directions cancel, the aggregate retains the highest-weight observed layer (first on a tie). This is a deterministic representative, not a confident estimate of a unique mean. Per-layer readings remain available.
- **Fit cost:** explicitly selected PCA returns after the PCA candidate is evaluated. It no longer computes unused spectral embeddings and curved RBF candidates. Automatic selection is unchanged; no wall-clock speedup is claimed.
- **UI accuracy:** zero residual no longer implies a flat manifold. Flat/curved labels use attached artifact metadata; absent metadata is shown as generic geometry.

## Verification

- 301 native tests across topology, manifold math, monitor, steering, gates, probe sessions, and geometry replay passed.
- Rust fitting tests and strict Clippy passed. Rebuilt hosted WASM assets pass the reproducibility check.
- Browser regressions cover periodic seams, rotation, weighted and ambiguous means, sphere poles, missing layer buffers, readout/gate agreement, abrupt warm-start jumps, and zero-strength identity.
- Browser fitting, activation-spool recovery, worker cancellation/crashes, coordinator, provenance, and WASM golden checks passed.
- Python type/lint checks and Svelte type checking passed.
- Geometry UI tests passed in Chromium and WebKit, including known-curved versus unknown metadata, keyboard layer selection, and light/dark layouts at 320px and 1280px. Generated screenshots were inspected. Native and hosted dashboard builds passed (existing large-chunk warnings remain).

### Real-model numerical smoke test

`scripts/check_manifold_model.py` uses cached SmolLM2-360M-Instruct on the Mac GPU, with real residual activations at layer 16. It tests flat and controlled curved injection at strengths 0, +0.3, and −0.3. Outputs remained finite; zero strength preserved baseline within numerical tolerance; removing each hook restored baseline exactly. Both nonzero signs changed logits.

The circle is constructed in a real activation frame: this checks numerical integration, not discovery of a semantic circle. It does not establish generation quality, calibrated membership, or generalization to other models.

Run with:

```sh
.venv/bin/python scripts/check_manifold_model.py --device mps
```

## Remaining boundaries

Hosted automatic topology discovery remains explicitly disabled in `browserModelBackend.ts`. Linear PCA and explicitly authored geometry remain supported. Passing synthetic topology tests is not sufficient evidence to release automatic semantic shape detection.

Before enabling that feature, validate topology selection on held-out, real-model corpora, including noisy flat data and negative controls. Check stability across resampling and layer subsets; expose inconclusive evidence rather than forcing a shape label. Residual and membership describe fit to the supplied artifact, not proof of a globally correct manifold.

The full browser runtime command currently stops at an unrelated SAE source-metadata deep-equality assertion in `drowse-web-llm-backend.test.mjs`: actual metadata includes optional `model_layers` and `description_source` keys with undefined values, while the expected object omits them. Remaining manifold tests were run separately; the entire runtime suite is not claimed green.
