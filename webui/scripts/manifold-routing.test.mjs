import assert from "node:assert/strict";
import { createServer } from "vite";

const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { fitGeometry, isDiscoverFit, isDiscoverFitMode, steeringDrawerFor } = await server.ssrLoadModule("/src/lib/manifolds/fitKind.ts");
  const row = (fit_mode, resolved_fit_mode, is_discover) => ({ fit_mode, resolved_fit_mode, is_discover });

  // ``auto`` folders (the bundled personas / emotions) route by their resolved geometry.
  assert.equal(fitGeometry(row("auto", "pca", true)), "flat");
  assert.equal(fitGeometry(row("auto", "spectral", true)), "curved");
  assert.equal(fitGeometry(row("auto", null, true)), "unresolved");
  assert.equal(fitGeometry(row("pca", null, true)), "flat", "an unfitted pca folder is flat before any fit");
  assert.equal(fitGeometry(row("spectral", null, true)), "curved");
  assert.equal(fitGeometry(row("baked", "baked", false)), "flat");
  assert.equal(fitGeometry(row("authored", "authored", false)), "curved");
  assert.equal(steeringDrawerFor(row("auto", "spectral", true)), "manifolds");
  assert.equal(steeringDrawerFor(row("auto", null, true)), "subspace");
  assert.equal(steeringDrawerFor(undefined), "subspace");
  for (const mode of ["pca", "spectral", "auto"]) assert.ok(isDiscoverFitMode(mode), mode);
  for (const mode of ["authored", "baked"]) assert.ok(!isDiscoverFitMode(mode), mode);
  assert.ok(isDiscoverFit(row("auto", "pca", true)));

  // The rack inspector renders diagnostics for an ``auto`` fit instead of
  // calling it authored (regression: the gate omitted ``auto``).
  const { render } = await server.ssrLoadModule("svelte/server");
  const { default: DiagnosticsPanel } = await server.ssrLoadModule("/src/lib/manifolds/DiagnosticsPanel.svelte");
  const manifold = {
    ...row("auto", "pca", true), namespace: "default", name: "personas", domain: { type: "custom" },
    fitted: [{
      stem: "m", method: "pca", feature_space: "raw", node_count: 3, nodes_sha256: null, fit_mode: "auto",
      diagnostics: { per_component_variance: [0.6, 0.3, 0.1], cumulative_variance: [0.6, 0.9, 1], picked_k: 2, threshold: 0.7 },
    }],
  };
  const fitted = render(DiagnosticsPanel, { props: { manifold } }).body;
  assert.match(fitted, /pca · k=2/);
  assert.match(fitted, /aria-label="pca diagnostics"/);
  const unfitted = render(DiagnosticsPanel, { props: { manifold: { ...manifold, resolved_fit_mode: null, fitted: [] } } }).body;
  assert.match(unfitted, /fit for diagnostics/);

  // Trigger pill: one state per engine preset, labels match the engine phases.
  const { TRIGGER_ORDER, TRIGGER_LABEL, nextTrigger } = await server.ssrLoadModule("/src/panels/rack/triggers.ts");
  assert.deepEqual(TRIGGER_ORDER, ["BOTH", "BEFORE", "AFTER", "THINKING", "RESPONSE"]);
  assert.match(TRIGGER_LABEL.BEFORE, /prompt prefill only/);
  assert.doesNotMatch(TRIGGER_LABEL.BEFORE, /response/);
  assert.match(TRIGGER_LABEL.RESPONSE, /thinking included/);
  assert.equal(nextTrigger("RESPONSE"), "BOTH");

  // Racks saved with the old alias pill values still import, as the canonical preset.
  const { serializeExpression } = await server.ssrLoadModule("/src/lib/expression.ts");
  const rack = new Map([["default/formal.casual", { mode: "subspace", coords: [1], label: "formal", variant: "raw", enabled: true, ablate: false, trigger: "BEFORE" }]]);
  assert.match(serializeExpression(rack, 0.5), /@before$/);
  console.log("Manifold routing (auto fits, diagnostics gate, steering drawer) and trigger presets passed");
} finally { await server.close(); }
