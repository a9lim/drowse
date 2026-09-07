import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

const geometry = (name, { affine = true, dim = 1 } = {}) => ({
  family: "geometry",
  name,
  manifold: `local/${name}`,
  top_n: 3,
  layers: [0, 1],
  node_labels: ["low", "high"],
  node_count: 2,
  domain: { type: "custom", embed_dim: dim, bounds: null },
  intrinsic_dim: dim,
  feature_space: "raw",
  is_affine: affine,
  node_coords: [[-1], [1]],
});

try {
  const { savedProbeSummary } = await server.ssrLoadModule("/src/lib/probeHistory.ts");
  const curve = geometry("curve", { affine: false, dim: 2 });
  assert.deepEqual(savedProbeSummary(curve, { curve: 2.5 }), {
    value: null, coordinates: [2.5], fraction: null,
  });
  assert.deepEqual(savedProbeSummary(curve, {
    curve: 2.5, "curve[1]": -0.4, "curve:fraction": 0.12,
  }), { value: 0.12, coordinates: [2.5, -0.4], fraction: 0.12 });
  assert.deepEqual(savedProbeSummary(geometry("flat"), {
    flat: -3, "flat:fraction": 0.04,
  }), { value: -3, coordinates: [-3], fraction: 0.04 });
  assert.deepEqual(savedProbeSummary({ family: "lens", name: "word" }, { word: 0.02 }), {
    value: 0.02, coordinates: [], fraction: null,
  });
  assert.deepEqual(savedProbeSummary(curve, { curve: NaN, "curve:fraction": Infinity }), {
    value: null, coordinates: [], fraction: null,
  });
  const compareSource = readFileSync(
    new URL("../src/drawers/CompareDrawer.svelte", import.meta.url),
    "utf8",
  );
  const correlationSource = readFileSync(
    new URL("../src/drawers/CorrelationDrawer.svelte", import.meta.url),
    "utf8",
  );
  const probeInspectorSource = readFileSync(
    new URL("../src/drawers/ProbeInspectorDrawer.svelte", import.meta.url),
    "utf8",
  );
  assert.match(compareSource, /comparisonProfileNames/);
  assert.match(correlationSource, /Omitted unsupported probes:/);
  assert.doesNotMatch(correlationSource, /every active probe/);
  assert.match(
    probeInspectorSource,
    /runtimeOperationAvailability\(\s*"probe_subspace_trails"/,
  );
  assert.match(probeInspectorSource, /live trail unavailable here/);
  assert.match(probeInspectorSource, /The fitted geometry remains available/);

  const {
    comparisonProfileNames,
    omittedAnalyticsProbes,
  } = await server.ssrLoadModule("/src/lib/profileAnalytics.ts");
  const {
    manifoldSelectorOptions,
    manifoldUsage,
    manifoldUsageMessage,
    selectorReferencesManifold,
  } = await server.ssrLoadModule("/src/lib/manifolds/selectors.ts");
  const { instrumentPackDependencies } = await server.ssrLoadModule(
    "/src/lib/runtime/instrumentPackSafety.ts",
  );

  const instrumentEntries = new Map([
    ["jlens/answer", { mode: "jlens", ablate: false, alpha: 0.3, trigger: "BOTH", enabled: false }],
    ["sae/7", { mode: "sae", ablate: false, alpha: 0.5, trigger: "BOTH", enabled: true }],
  ]);
  const instrumentProbes = [
    { family: "lens", name: "answer watch" },
    { family: "sae", name: "feature seven" },
  ];
  assert.deepEqual(instrumentPackDependencies({
    kind: "jlens",
    steeringExpression: "0.5 calm@when:jlens/fake>0.01",
    steeringEntries: instrumentEntries,
    probes: instrumentProbes,
  }), [
    "J-lens steering: jlens/answer",
    "J-lens readings: answer watch",
  ]);
  assert.deepEqual(instrumentPackDependencies({
    kind: "sae",
    steeringExpression: "0.5 calm@when:sae/9>0.2",
    steeringEntries: new Map(),
    probes: [],
  }), ["SAE steering expression or condition"]);
  assert.deepEqual(instrumentPackDependencies({
    kind: null,
    steeringExpression: "0.5 calm@when:jlens/fake>0.01 + 0.2 other@when:sae/9>0.2",
    steeringEntries: new Map(),
    probes: [],
  }), [
    "J-lens steering expression or condition",
    "SAE steering expression or condition",
  ]);
  assert.deepEqual(instrumentPackDependencies({
    kind: "jlens",
    steeringExpression: "0.5 myjlens/foo",
    steeringEntries: new Map(),
    probes: [],
  }), []);

  const manifold = {
    namespace: "local",
    name: "demo",
    description: "",
    source: "local",
    tags: [],
    template_ref: null,
    fit_mode: "pca",
    is_discover: true,
    domain: { type: "custom", embed_dim: 1 },
    domain_label: "1D",
    intrinsic_dim: 1,
    min_nodes: 2,
    node_count: 2,
    node_labels: ["low", "high"],
    node_coords: [[-1], [1]],
    node_roles: [null, null],
    node_kinds: [null, null],
    hyperparams: {},
    fitted_models: ["fixture/model"],
    tensor_variants: { "fixture/model": ["raw", "sae-release-a"] },
    fitted_for_session: true,
    stale: false,
    resolved_fit_mode: "pca",
  };
  assert.deepEqual(
    manifoldSelectorOptions(manifold, "fixture/model", true).map((option) => [
      option.selector,
      option.variant,
      option.available,
    ]),
    [
      ["local/demo", "raw", true],
      ["local/demo:sae-release-a", "sae-release-a", true],
    ],
  );
  assert.deepEqual(
    manifoldSelectorOptions({ ...manifold, node_roles: ["scholar", "scholar"] }, "fixture/model", true)
      .map((option) => [option.selector, option.available]),
    [
      ["local/demo:role-scholar", true],
      ["local/demo:sae-release-a", true],
    ],
  );
  const mixedRoleOptions = manifoldSelectorOptions(
    { ...manifold, node_roles: ["pirate", "scholar"] },
    "fixture/model",
    true,
  );
  assert.equal(mixedRoleOptions.every((option) => option.available), true);
  assert.equal(mixedRoleOptions[0].label, "nearest node role");
  assert.equal(selectorReferencesManifold("local/demo:sae-release-a", manifold), true);
  assert.equal(selectorReferencesManifold("demo:role-scholar", manifold), true);
  assert.equal(selectorReferencesManifold("local/other", manifold), false);

  const probeInfo = [
    geometry("rank one"),
    geometry("curved", { affine: false }),
    geometry("fan", { dim: 2 }),
    {
      family: "lens",
      name: "next hello",
      layers: [0, 1],
      intrinsic_dim: 1,
      feature_space: "readout",
      word: "hello",
      token_id: 0,
    },
    {
      family: "sae",
      name: "feature one",
      layers: [1],
      intrinsic_dim: 1,
      feature_space: "sae-readout",
      feature_id: 1,
      label: null,
      max_act: null,
    },
  ];
  const entries = new Map(probeInfo.map((info) => [info.name, { info }]));
  const active = [...entries.keys()];

  assert.deepEqual(
    comparisonProfileNames(
      ["local/demo", "rank one"],
      active,
      entries,
      false,
    ),
    ["local/demo", "rank one"],
  );
  assert.deepEqual(
    comparisonProfileNames(["local/demo"], active, entries, true),
    ["local/demo", "rank one"],
  );
  assert.deepEqual(
    omittedAnalyticsProbes(active, entries, ["local/demo", "rank one"]),
    [
      { name: "curved", reason: "curved manifold, not one direction" },
      { name: "fan", reason: "multidimensional subspace, not one direction" },
      { name: "feature one", reason: "SAE readout, not a residual direction" },
      { name: "next hello", reason: "J-lens readout, not a residual direction" },
    ],
  );
  const collidingReadout = {
    ...probeInfo.find((info) => info.family === "lens"),
    name: "local/demo",
  };
  assert.deepEqual(
    omittedAnalyticsProbes(
      ["local/demo"],
      new Map([["local/demo", { info: collidingReadout }]]),
      ["local/demo"],
    ),
    [{ name: "local/demo", reason: "J-lens readout, not a residual direction" }],
  );

  let probeRoster = [geometry("rank one")];
  let resolveCorrelation;
  const correlationCalls = [];
  const attachCalls = [];
  let tokenReadoutCalls = 0;
  let tokenReadoutImpl = async () => {
    throw new Error("fixture replay unavailable");
  };
  const runtime = {
    mode: "browser",
    profiles: {
      list: async () => ({
        profiles: [
          {
            name: "local/demo:role-scholar",
            layers: [0, 1],
            metadata: { selector_key: "local/demo:role-scholar" },
          },
          {
            name: "local/demo:sae-release",
            layers: [0, 1],
            metadata: { selector_key: "local/demo:sae-release" },
          },
        ],
      }),
      correlation: async (names) => {
        correlationCalls.push(names);
        return await new Promise((resolve) => { resolveCorrelation = resolve; });
      },
    },
    probes: {
      list: async () => ({ probes: probeRoster }),
      attach: async (request) => {
        attachCalls.push(structuredClone(request));
        const info = {
          ...geometry(request.name ?? request.selector),
          manifold: request.selector,
        };
        probeRoster = [...probeRoster, info];
        return info;
      },
      detach: async (name) => {
        probeRoster = probeRoster.filter((probe) => probe.name !== name);
      },
    },
    instruments: {
      tokenReadout: async (...args) => {
        tokenReadoutCalls += 1;
        return tokenReadoutImpl(...args);
      },
    },
  };
  const { installRuntimeClient } = await server.ssrLoadModule(
    "/src/lib/runtime/registry.ts",
  );
  installRuntimeClient(runtime);
  const steering = await server.ssrLoadModule(
    "/src/lib/stores/steering.svelte.ts",
  );
  const probes = await server.ssrLoadModule(
    "/src/lib/stores/probes.svelte.ts",
  );
  const instruments = await server.ssrLoadModule(
    "/src/lib/stores/instruments.svelte.ts",
  );
  const session = await server.ssrLoadModule(
    "/src/lib/stores/session.svelte.ts",
  );

  for (let id = 0; id < instruments.MAX_SAE_HISTORY_FEATURES + 8; id += 1) {
    instruments.recordSaeReadoutFrame([{
      id,
      activation: id,
      label: `feature ${id}`,
      max_act: id + 1,
    }]);
  }
  assert.equal(instruments.saeState.history.size, instruments.MAX_SAE_HISTORY_FEATURES);
  assert.equal(instruments.saeState.history.has(0), false);
  assert.equal(
    instruments.saeState.history.has(instruments.MAX_SAE_HISTORY_FEATURES + 7),
    true,
  );
  assert.equal(instruments.saeState.meta.has(0), false);
  instruments.recordSaeReadoutFrame([{
    id: instruments.MAX_SAE_HISTORY_FEATURES + 7,
    activation: 999,
    label: null,
    max_act: null,
  }]);
  assert.deepEqual(
    instruments.saeState.history.get(instruments.MAX_SAE_HISTORY_FEATURES + 7),
    [instruments.MAX_SAE_HISTORY_FEATURES + 7, 999],
  );
  instruments.saeState.readout = [];
  instruments.saeState.history.clear();
  instruments.saeState.meta.clear();

  steering.steerRack.profiles.set("stale", {
    name: "stale",
    layers: [0],
    metadata: {},
  });
  await steering.refreshVectorList();
  assert.deepEqual(steering.vectorsState.names, [
    "local/demo:role-scholar",
    "local/demo:sae-release",
  ]);
  assert.deepEqual([...steering.steerRack.profiles.keys()], steering.vectorsState.names);

  const staleRefresh = steering.refreshCorrelation(null);
  await Promise.resolve();
  assert.deepEqual(correlationCalls, [null]);
  await probes.attachProbe("local/new", { name: "new alias" });
  assert.deepEqual(attachCalls.at(-1), {
    selector: "local/new",
    name: "new alias",
    top_n: undefined,
  });
  assert.equal(steering.steerRack.correlation, null);
  resolveCorrelation({
    names: ["stale"],
    matrix: { stale: { stale: 1 } },
    layers_shared: {},
  });
  await staleRefresh;
  assert.equal(steering.steerRack.correlation, null);

  steering.steerRack.correlation = {
    names: ["new alias"],
    matrix: { "new alias": { "new alias": 1 } },
    layers_shared: {},
  };
  await probes.detachProbe("new alias");
  assert.equal(steering.steerRack.correlation, null);

  steering.steerRack.correlation = {
    names: ["rank one"],
    matrix: { "rank one": { "rank one": 1 } },
    layers_shared: {},
  };
  await probes.refreshProbeList();
  assert.equal(steering.steerRack.correlation, null);

  steering.steerRack.catalog = [manifold];
  steering.addSubspaceToRack("local/demo", "sae-release-a");
  assert.equal(
    steering.currentSteeringExpression(),
    "0.5 local/demo:sae-release-a%low",
  );
  await probes.attachProbe("local/demo:sae-release-a", { name: "friendly alias" });
  const usage = manifoldUsage(manifold, steering.steerRack.entries, probes.probeRack.entries);
  assert.deepEqual(usage, {
    rack: ["local/demo"],
    probes: ["friendly alias"],
  });
  assert.match(manifoldUsageMessage(manifold, usage), /friendly alias/);

  session.sessionState.info = {
    model_id: "fixture/model",
    jlens_fitted: true,
    instruments: [],
  };
  instruments.beginTokenHover({ text: "token", rawIndex: 3 }, "node-a");
  await new Promise((resolve) => setTimeout(resolve, 180));
  assert.equal(tokenReadoutCalls, 0);
  assert.equal(instruments.tokenHoverState.lensError, null);
  assert.equal(instruments.tokenHoverState.lensLoading, false);
  instruments.endTokenHover();
  await new Promise((resolve) => setTimeout(resolve, 60));

  session.sessionState.info.instruments = [{
    family: "lens",
    live: { enabled: false, layers: null },
    source: "fixture",
    probes: [],
    capabilities: {
      sources: true,
      preparations: [],
      token_readout: true,
      source_switch: false,
    },
  }];
  instruments.beginTokenHover({ text: "token", rawIndex: 3 }, "node-a");
  await new Promise((resolve) => setTimeout(resolve, 180));
  assert.equal(tokenReadoutCalls, 1);
  assert.match(
    instruments.tokenHoverState.lensError,
    /fixture replay unavailable/,
  );
  assert.equal(instruments.tokenHoverState.lensLoading, false);
  instruments.endTokenHover();
  await new Promise((resolve) => setTimeout(resolve, 60));
  assert.equal(instruments.tokenHoverState.lensError, null);

  let releaseReadout;
  tokenReadoutImpl = () => new Promise((resolve) => { releaseReadout = resolve; });
  const {
    cachedTokenReadout,
    invalidateTokenReadoutCache,
  } = await server.ssrLoadModule(
    "/src/lib/runtime/tokenReadoutCache.ts",
  );
  const first = cachedTokenReadout("lens", "node-b", 4, {
    topK: 8,
    steered: true,
    raw: false,
    layers: "all",
  });
  const second = cachedTokenReadout("lens", "node-b", 4, {
    topK: 8,
    steered: true,
    raw: false,
    layers: "all",
  });
  assert.equal(tokenReadoutCalls, 2);
  releaseReadout({
    measurements: {
      version: 1,
      scope: "replay",
      provenance: "replayed",
      instruments: { lens: { readout: { layers: [], aggregate: [] } } },
    },
  });
  assert.deepEqual(await first, await second);
  await cachedTokenReadout("lens", "node-b", 4, {
    topK: 8,
    steered: true,
    raw: false,
    layers: "all",
  });
  assert.equal(tokenReadoutCalls, 2);

  const replayEnvelope = {
    measurements: {
      version: 1,
      scope: "replay",
      provenance: "replayed",
      instruments: { lens: { readout: { layers: [], aggregate: [] } } },
    },
  };
  tokenReadoutImpl = async () => replayEnvelope;
  invalidateTokenReadoutCache("lens");
  await cachedTokenReadout("lens", "node-b", 4, {
    topK: 8,
    steered: true,
    raw: false,
    layers: "all",
  });
  assert.equal(tokenReadoutCalls, 3);
  await Promise.resolve();
  await Promise.resolve();

  const releases = new Map();
  const callOrder = [];
  tokenReadoutImpl = (_family, nodeId) => new Promise((resolve) => {
    callOrder.push(nodeId);
    releases.set(nodeId, resolve);
  });
  const activeReadout = cachedTokenReadout("lens", "node-active", 2, {});
  const backgroundReadout = cachedTokenReadout(
    "lens",
    "node-background",
    2,
    {},
    undefined,
    "background",
  );
  const queuedProgress = [];
  const selectedReadout = cachedTokenReadout(
    "lens",
    "node-selected",
    2,
    {},
    (event) => queuedProgress.push(event),
  );
  assert.deepEqual(callOrder, ["node-active"]);
  assert.equal(queuedProgress.at(-1)?.data?.phase, "queued");
  assert.match(queuedProgress.at(-1)?.data?.message, /1 token reading/);

  releases.get("node-active")(replayEnvelope);
  await activeReadout;
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(callOrder, ["node-active", "node-selected"]);
  releases.get("node-selected")(replayEnvelope);
  await selectedReadout;
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(callOrder, [
    "node-active",
    "node-selected",
    "node-background",
  ]);
  releases.get("node-background")(replayEnvelope);
  await backgroundReadout;

  console.log("Browser analytics UI/store checks passed");
} finally {
  await server.close();
}
