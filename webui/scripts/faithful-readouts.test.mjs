import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const flush = () => new Promise((resolve) => setImmediate(resolve));
const envelope = { measurements: { version: 1, scope: "replay", provenance: "replayed", instruments: {
  lens: { binding: { source: "test", steering: null }, readings: {}, readout: { layers: [], aggregate: [] } },
  sae: { binding: { source: "test", steering: null, layer: 0 }, readings: {}, readout: { features: [] } },
  geometry: { binding: { source: "test", steering: null }, readings: {} },
} } };

try {
  const requests = [];
  let replay = (...args) => new Promise((resolve, reject) => requests.push({ args, resolve, reject }));
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({ mode: "browser", instruments: {
    tokenReadout: (...args) => replay(...args),
  } });
  const { cachedTokenReadout, invalidateTokenReadoutCache, tokenReadoutCacheEpoch } =
    await server.ssrLoadModule("/src/lib/runtime/tokenReadoutCache.ts");

  const active = cachedTokenReadout("lens", "active", 1, {});
  const queued = cachedTokenReadout("lens", "obsolete", 2, {});
  const unaffected = cachedTokenReadout("sae", "other-family", 3, {});
  const rejectedActive = assert.rejects(active, /conversation or reading source changed/);
  const rejectedQueued = assert.rejects(queued, /conversation or reading source changed/);
  invalidateTokenReadoutCache("lens");
  await Promise.all([rejectedActive, rejectedQueued]);
  const options = { topK: 5, steered: true };
  const fresh = cachedTokenReadout("lens", "active", 1, options);
  options.steered = false;
  assert.equal(requests.length, 1, "invalidation must not overlap GPU jobs");
  requests[0].resolve(envelope);
  await flush();
  assert.equal(requests.length, 2);
  assert.equal(requests[1].args[1], "active");
  assert.equal(requests[1].args[3].steered, true, "queued options retain their keyed identity");
  requests[1].resolve(envelope);
  await fresh;
  await flush();
  assert.equal(requests[2].args[1], "other-family");
  requests[2].resolve(envelope);
  await unaffected;
  await flush();
  assert.equal(requests.some(({ args }) => args[1] === "obsolete"), false);
  await cachedTokenReadout("lens", "active", 1, { topK: 5, steered: true });
  assert.equal(requests.length, 3, "only the fresh result is reused");

  const { ReplayReadout } = await server.ssrLoadModule("/src/drawers/token/readout.svelte.ts");
  const view = new ReplayReadout();
  let attempts = 0;
  replay = async () => {
    if (++attempts === 1) throw new Error("temporary read failure");
    return envelope;
  };
  const map = () => ({ data: { strength: 0.4 }, source: "test-source" });
  view.replay("lens", "retry", 0, {}, map);
  await flush();
  assert.match(view.error, /temporary read failure/);
  view.replay("lens", "retry", 0, {}, map);
  await flush();
  assert.equal(attempts, 2);
  assert.deepEqual(view.data, { strength: 0.4 });
  assert.equal(view.origin, "replayed");
  view.dispose();
  assert.equal(view.data, null);
  replay = async () => ({ measurements: { ...envelope.measurements, instruments: {} } });
  await assert.rejects(cachedTokenReadout("lens", "missing-result", 0, {}), /No reading was returned/);
  await flush();
  replay = async () => envelope;
  assert.deepEqual(await cachedTokenReadout("lens", "missing-result", 0, {}), envelope,
    "a missing result is neither measured zero nor a reusable cache hit");
  await flush();

  const { applyTreeSnapshot, applyTreeDelta, loomTree } =
    await server.ssrLoadModule("/src/lib/stores/loom.svelte.ts");
  const node = (id, parent_id) => ({
    id, parent_id, role: parent_id ? "assistant" : "system", text: id,
    recipe: null, applied_steering: null, finish_reason: "stop", tokens: [],
  });
  const snapshot = {
    tree_format: 2, drowse_version: "test", session_id: "default", model_id: "fixture",
    root_id: "root", active_node_id: "a", rev: 1,
    nodes: [node("root", null), node("a", "root")],
    children_of: { root: ["a"], a: [] }, cast: {},
  };
  applyTreeSnapshot(snapshot);
  const epoch = tokenReadoutCacheEpoch("lens");
  applyTreeDelta({ rev: 2, updated: [{ ...loomTree.nodes.get("a"), starred: true }] });
  assert.equal(tokenReadoutCacheEpoch("lens"), epoch, "stars do not alter replay inputs");
  applyTreeDelta({ rev: 3, updated: [{ ...loomTree.nodes.get("root"), text: "new system prompt" }] });
  assert.equal(tokenReadoutCacheEpoch("lens"), epoch + 1, "ancestor edits invalidate descendant replays");
  applyTreeSnapshot({ ...snapshot, rev: 4 });
  assert.equal(tokenReadoutCacheEpoch("lens"), epoch + 2, "same node IDs in restored chats are not cache hits");
  applyTreeSnapshot({ ...snapshot, rev: 5, model_id: "other-model" }, { reconcileEdgeLabels: true });
  assert.equal(tokenReadoutCacheEpoch("lens"), epoch + 3, "model swaps invalidate every family");

  const { highlightScoreFor } = await server.ssrLoadModule("/src/lib/highlight.ts");
  assert.equal(highlightScoreFor({ text: "x", score: 0.9, probes: { old: 0.9 } }, "new"), undefined);
  assert.equal(highlightScoreFor({ text: "x", perLayerScores: { 9: { probe: 0.8 } } }, "probe"), undefined);
  assert.equal(highlightScoreFor({ text: "x", probes: { probe: 0 } }, "probe"), 0);
  assert.equal(highlightScoreFor({ text: "x", coordsByProbe: { probe: [0.1, -0.2] } }, "probe[1]"), -0.2);

  const { probeRack, highlightScale, probeEntryForDisplay, highlightState } =
    await server.ssrLoadModule("/src/lib/stores/probes.svelte.ts");
  const { tokenHoverState } = await server.ssrLoadModule("/src/lib/stores/instruments.svelte.ts");
  probeRack.entries.set("word alias", { info: { family: "lens", name: "word alias", word: "word", layers: [0] },
    current: 0.5, previous: 0.5, sparkline: [0.5], reading: null, aggregate: null });
  probeRack.entries.set("feature alias", { info: { family: "sae", name: "feature alias", feature_id: 7, max_act: 10, layers: [0] },
    current: 5, previous: 5, sparkline: [5], reading: { value: 5, unit: "raw_activation", per_layer: {}, depth: null }, aggregate: null });
  probeRack.active = ["word alias", "feature alias"];
  assert.equal(highlightScale("word alias"), 1);
  assert.equal(highlightScale("feature alias"), 5, "reading units override newly cached feature metadata");
  const { highlightStyleFor } = await server.ssrLoadModule("/src/lib/highlight.ts");
  highlightState.target = "word alias";
  assert.match(highlightStyleFor({ text: "x", probes: { "word alias": 0.5 } }).backgroundColor, /pillar-lens/);
  tokenHoverState.active = true;
  tokenHoverState.lensAggregate = [[" word", 0.9, 0.5, 0.1]];
  assert.equal(probeEntryForDisplay("word alias").reading, null, "hover cannot borrow a whitespace variant's probability");
  tokenHoverState.lensAggregate = [["word", 0.2, 0.5, 0.1]];
  assert.equal(probeEntryForDisplay("word alias").reading.value, 0.2);
  tokenHoverState.saeReadout = [{ id: 7, activation: 3, max_act: null }];
  assert.equal(probeEntryForDisplay("feature alias").reading.unit, "raw_activation");
  assert.equal(probeEntryForDisplay("feature alias").reading.value, 3, "historical rows are not normalized using current metadata");
  tokenHoverState.active = false;
  probeRack.active = [];
  probeRack.entries.clear();
  highlightState.target = null;

  const { sparklinePaths } = await server.ssrLoadModule("/src/lib/charts/sparklinePath.ts");
  assert.deepEqual(sparklinePaths([], 60, 16, 1), { line: "", area: "" });
  assert.deepEqual(sparklinePaths([null], 60, 16, 1), { line: "", area: "" });
  assert.equal(sparklinePaths([1], 60, 16, 1).line, "M 0 0.00 L 60 0.00");
  assert.equal(sparklinePaths([0], 60, 16, 1).line, "M 0 8.00 L 60 8.00");
  const gaps = sparklinePaths([1, 0.5, null, 0.25, 0], 40, 16, 1);
  assert.equal(gaps.line, "M 0.00 0.00 L 10.00 4.00 M 30.00 6.00 L 40.00 8.00");
  assert.equal((gaps.area.match(/Z/g) ?? []).length, 2, "area fills cannot bridge missing measurements");
  assert.deepEqual(sparklinePaths([NaN, Infinity], 60, 16, 1), { line: "", area: "" });

  const { render } = await server.ssrLoadModule("svelte/server");
  const { default: LogitsTab } = await server.ssrLoadModule("/src/drawers/token/LogitsTab.svelte");
  const chosen = { text: "x", tokenId: 1, logprob: Math.log(0.9996),
    topAlts: [{ text: "x", id: 1, logprob: Math.log(0.9996) }] };
  const logitHtml = render(LogitsTab, { props: { token: chosen, nodeId: null } }).body;
  assert.match(logitHtml, /not a measure of factual accuracy/);
  assert.doesNotMatch(logitHtml, /Only one token remained|choice was certain/);
  assert.doesNotMatch(render(LogitsTab, { props: { token: { ...chosen, tokenId: 2 }, nodeId: null } }).body,
    /This token has almost all/, "a top alternative's probability is not the chosen token's probability");
  const { default: LensTab } = await server.ssrLoadModule("/src/drawers/token/LensTab.svelte");
  const { default: SaeTab } = await server.ssrLoadModule("/src/drawers/token/SaeTab.svelte");
  const scalar = { value: 0.25, unit: "mean_token_probability", per_layer: {}, depth: null };
  const readout = {
    data: { token_id: 2, token_text: " word", steering: null,
      aggregate: [{ token: " word", strength: 0.1, com: 0.5, spread: 0.1 }],
      layers: [{ layer: 0, tokens: [{ token: "word", id: 1, logprob: Math.log(0.8) }] }],
      layer: 0, features: [],
    },
    origin: "captured", source: "original", progress: null, loading: false, error: null,
  };
  const props = {
    readout, steered: true, jlensFitted: true, saeLoaded: true, hasReplayContext: true,
    replayAvailable: true, modelId: "fixture", pinned: { "original-probe": scalar },
  };
  const lensHtml = render(LensTab, { props }).body;
  assert.match(lensHtml, /original-probe/);
  assert.match(lensHtml, /0\/1 layers/, "a whitespace variant is not the aggregate token's layer probability");
  assert.match(render(SaeTab, { props }).body, /original-probe/);
  readout.origin = "replayed";
  readout.source = "new-source";
  assert.doesNotMatch(render(LensTab, { props }).body, /original-probe/);
  assert.doesNotMatch(render(SaeTab, { props }).body, /original-probe/);
  readout.data.token_text = "word";
  assert.doesNotMatch(render(LensTab, { props }).body, /class="badge"[^>]*>generated/);

  const { default: JLensProbeCard } = await server.ssrLoadModule("/src/panels/rack/JLensProbeCard.svelte");
  const { default: SaeProbeCard } = await server.ssrLoadModule("/src/panels/rack/SaeProbeCard.svelte");
  const card = { token: "word", probeName: "custom lens alias", strength: null,
    com: null, spread: null, series: [], cells: [], pinned: true };
  const lensCard = render(JLensProbeCard, { props: card }).body;
  assert.match(lensCard, /Not measured/);
  assert.match(lensCard, /Unpin probe custom lens alias/);
  assert.doesNotMatch(lensCard, /Strength 0/);
  const saeCard = render(SaeProbeCard, { props: {
    id: 7, probeName: "custom feature alias", layer: 0, value: 0, measured: false,
    series: [], pinned: true,
  } }).body;
  assert.match(saeCard, /Not measured/);
  assert.match(saeCard, /Unpin probe custom feature alias/);
  assert.doesNotMatch(saeCard, /Activation 0/);

  console.log("Faithful readout checks passed (cache lifetimes, provenance, token identity, missing data, and aliases)");
} finally {
  await server.close();
}
