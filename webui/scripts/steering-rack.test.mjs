import assert from "node:assert/strict";
import { createServer } from "vite";

const server = await createServer({
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  // The profiles service the correlation-freshness block below drives.
  let roster = ["a"];
  let pendingCorrelation = null;
  installRuntimeClient({
    mode: "http",
    profiles: {
      list: async () => ({ profiles: roster.map(name => ({ name, layers: [0], metadata: {} })) }),
      correlation: () => new Promise(resolve => { pendingCorrelation = resolve; }),
    },
  });
  const steering = await server.ssrLoadModule("/src/lib/stores/steering.svelte.ts");
  const { genStatus } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const { pendingActions, drainNextPendingAction } = await server.ssrLoadModule("/src/lib/stores/pending.svelte.ts");
  const { steerRack, patchSteerEntry, setSteerCoords, setSteerLabel, removeSteerEntry, addSubspaceToRack, addManifoldToRack, addSaeToRack, currentSteeringExpression } = steering;
  steerRack.catalog = [{
    namespace: "default", name: "emotions", node_count: 3, node_labels: ["calm", "angry", "glad"],
    node_coords: [[0, 0], [1, 0], [0, 1]], intrinsic_dim: 2, domain: { type: "custom" },
  }, {
    namespace: "default", name: "formal.casual", node_count: 2, node_labels: ["formal", "casual"],
    node_coords: [[1], [-1]], intrinsic_dim: 1, domain: { type: "custom" },
  }];
  addSubspaceToRack("default/formal.casual");
  addManifoldToRack("default/emotions");
  addSaeToRack(17);

  // Label binding mirrors node coords; coords drop the label; both only touch geometric terms.
  setSteerLabel("default/emotions", "glad");
  assert.deepEqual(steerRack.entries.get("default/emotions"), { ...steerRack.entries.get("default/emotions"), label: "glad", coords: [0, 1] });
  setSteerCoords("default/emotions", [0.5, 0.25]);
  assert.equal(steerRack.entries.get("default/emotions").label, null);
  setSteerLabel("default/emotions", "unknown");
  assert.deepEqual(steerRack.entries.get("default/emotions").coords, [0.5, 0.25], "an uncatalogued label keeps the coords");
  assert.equal(steerRack.entries.get("default/emotions").label, "unknown", "and resolves server-side");
  setSteerLabel("default/emotions", null);
  setSteerLabel("sae/17", "glad");
  assert.equal(steerRack.entries.get("sae/17").label, undefined, "atoms have no geometry");

  // Mode-restricted patches leave other families alone.
  patchSteerEntry("default/formal.casual", { blend: 0.9 }, ["manifold"]);
  assert.equal(steerRack.entries.get("default/formal.casual").blend, undefined);
  patchSteerEntry("default/emotions", { blend: 0.8, onto: 0.2 }, ["manifold"]);
  patchSteerEntry("sae/17", { alpha: 0.4, ablate: true }, ["sae"]);
  patchSteerEntry("default/formal.casual", { trigger: "BEFORE" });
  assert.equal(currentSteeringExpression(),
    "0.5 default/formal.casual%formal@before + 0.4 !sae/17 + 0.8,0.2 default/emotions%0.5,0.25");

  // Edits during a generation queue and coalesce into one bubble.
  genStatus.active = true;
  patchSteerEntry("sae/17", { enabled: false });
  patchSteerEntry("sae/17", { alpha: 0.6 }, ["sae"]);
  assert.equal(pendingActions.queue.length, 1);
  assert.equal(steerRack.entries.get("sae/17").enabled, true);
  genStatus.active = false;
  await drainNextPendingAction();
  assert.deepEqual([steerRack.entries.get("sae/17").enabled, steerRack.entries.get("sae/17").alpha], [false, 0.6]);
  removeSteerEntry("sae/17");
  assert.equal(steerRack.entries.has("sae/17"), false);

  // Correlation freshness: nothing fetches per generation, but a generation
  // (which can register vectors and fill null cells) or a vector-roster change
  // leaves the cached matrix stale, so the open drawer refetches.
  const { correlationCache, markCorrelationStale, refreshCorrelation, refreshVectorList } = steering;
  const matrix = names => ({ names, matrix: {}, layers_shared: {} });
  const fresh = () => correlationCache.fetched === correlationCache.revision;
  await refreshVectorList();
  let request = refreshCorrelation(null);
  pendingCorrelation(matrix(["a"]));
  await request;
  assert.deepEqual(steerRack.correlation.names, ["a"]);
  assert.ok(fresh());

  markCorrelationStale();
  assert.deepEqual(steerRack.correlation.names, ["a"], "a stale matrix stays on screen");
  assert.ok(!fresh(), "but the open drawer refetches it");
  request = refreshCorrelation(null);
  markCorrelationStale();
  pendingCorrelation(matrix(["stale"]));
  await request;
  assert.deepEqual(steerRack.correlation.names, ["a"], "a response older than the last stale mark is dropped");
  assert.ok(!fresh());

  await refreshVectorList();
  assert.ok(steerRack.correlation, "an unchanged vector roster keeps the matrix");
  roster = ["a", "b"];
  await refreshVectorList();
  assert.equal(steerRack.correlation, null, "a registered vector joins the pool");
  console.log("Steering rack: one patch path for every term, label binding, mode guards, queued edits, and correlation freshness passed");
} finally { await server.close(); }
