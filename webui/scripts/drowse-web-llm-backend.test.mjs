import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const { DrowseWebLlmBackend } = await server.ssrLoadModule(
    "/src/hosted/runtime/drowseWebLlmBackend.ts",
  );
  const { BrowserInstrumentRuntime } = await server.ssrLoadModule(
    "/src/hosted/runtime/browserInstrumentRuntime.ts",
  );
  const calls = [];
  const runtime = {
    runtimeCapabilities() {
      return {
        topK: true,
        forcedReplay: true,
        replayScoring: true,
        tokenizer: true,
        namedRoles: true,
        userSeatGeneration: true,
        sceneStitching: true,
        defaultUserRole: "user",
        defaultAssistantRole: "model",
      };
    },
    thinkingProfileForSession() { return null; },
    async load(request) {
      calls.push(["load", request]);
      return { prefillTokensPerSecond: 12, decodeTokensPerSecond: 6 };
    },
    async unload() { calls.push(["unload"]); },
    async stop() { calls.push(["stop"]); },
    generationPerformance() {
      return { prefillTokensPerSecond: 12, decodeTokensPerSecond: 6 };
    },
    async tokenizeText(text) { return text.trim() === "yes" ? [7] : [8]; },
    async decodeTokens(tokenIds) { return tokenIds[0] === 7 ? " yes" : " other"; },
    async resolveJlensTokenDirections() { return new Float32Array(); },
    async streamGeneration(plan, onToken) {
      calls.push(["generate", plan]);
      await onToken({
        text: "answer",
        thinking: false,
        tokenId: null,
        logprob: -0.2,
        perplexity: Math.exp(0.2),
        rawIndex: null,
      });
      return {
        text: "answer",
        tokens: 1,
        finishReason: "stop",
        usage: { promptTokens: 2, completionTokens: 1, totalTokens: 3 },
        meanLogprob: -0.2,
        meanSurprise: 0.2,
        prefillTokensPerSecond: 12,
        decodeTokensPerSecond: 6,
      };
    },
  };
  const compiled = [];
  const deltas = [];
  const authoring = [];
  let id = 0;
  const instrumentLive = { geometry: true, lens: false, lensLayers: null, sae: false };
  const attachedProbes = [];
  const preparedJlensWords = [];
  const initialProbeHash = "1".repeat(64);
  const refreshedLensProbeHash = "2".repeat(64);
  const refreshedGeometryProbeHash = "3".repeat(64);
  let saeSourceMetadata = {};
  const instrumentCompiler = {
    listProfiles() {
      return { profiles: [{ name: "local/demo", layers: [0, 1], metadata: {} }] };
    },
    getProfile(name) {
      if (name !== "local/demo") throw new Error("unknown profile");
      return { name, layers: [0, 1], metadata: {} };
    },
    profileCorrelation(names) {
      const selected = names?.length ? names : ["local/demo"];
      return {
        names: selected,
        matrix: Object.fromEntries(selected.map((name) => [name, { [name]: 1 }])),
        layers_shared: {},
      };
    },
    profilePairwise(a, b) {
      return {
        a,
        b,
        metric: "mahalanobis",
        layers_a: [0, 1],
        layers_b: [0, 1],
        matrix: [[1, 0], [0, 1]],
        model: "fixture-model",
      };
    },
    instrumentDescriptor() {
      return {
        jlens: {
          source: "fixture-jlens",
          displayName: "Fixture J-lens",
          layers: [0, 1],
          words: [{ word: "yes", tokenId: 7 }],
        },
        sae: {
          source: "fixture-sae",
          displayName: "Fixture SAE",
          layer: 1,
          features: 4,
          ...saeSourceMetadata,
        },
      };
    },
    instrumentLiveState() { return { ...instrumentLive }; },
    listProbes() { return structuredClone(attachedProbes); },
    probeHashes() {
      return Object.fromEntries(attachedProbes.map((probe) => [probe.name, initialProbeHash]));
    },
    hasAttachedProbes() { return attachedProbes.length > 0; },
    async prepareJlensWords(words, resolver) {
      await Promise.resolve();
      assert.equal(resolver, runtime);
      preparedJlensWords.push([...words]);
    },
    attachProbe(request) {
      const word = request.selector.slice("jlens/".length);
      const probe = {
        family: "lens",
        name: request.name || request.selector,
        layers: [0, 1],
        intrinsic_dim: 1,
        feature_space: "readout",
        word,
        token_id: 7,
      };
      attachedProbes.push(probe);
      return structuredClone(probe);
    },
    detachProbe(name) {
      const index = attachedProbes.findIndex((probe) => probe.name === name);
      if (index >= 0) attachedProbes.splice(index, 1);
    },
    probeGeometry() { throw new Error("fixture has no geometry probe"); },
    listManifolds() {
      return [{ namespace: "local", name: "demo" }];
    },
    setInstrumentLive(family, enabled, layers) {
      instrumentLive[family] = enabled;
      if (family === "lens") {
        instrumentLive.lensLayers = enabled
          ? [...layers ?? this.instrumentDescriptor().jlens.layers]
          : null;
      }
    },
    validateLensToken(word) {
      if (word !== "yes") throw new Error("unknown token");
      return { word, token_id: 7 };
    },
    validateSaeFeature(featureId) {
      if (featureId > 3) throw new Error("unknown feature");
      return { id: featureId, label: null, layer: 1, max_act: null };
    },
  };
  const refreshedAttachedProbes = [];
  const refreshedCompiler = {
    ...instrumentCompiler,
    listProfiles() {
      return {
        profiles: [
          { name: "local/demo", layers: [0, 1], metadata: {} },
          { name: "local/fitted", layers: [0, 1], metadata: {} },
        ],
      };
    },
    listManifolds() {
      return [
        { namespace: "local", name: "demo" },
        { namespace: "local", name: "fitted" },
      ];
    },
    listProbes() { return structuredClone(refreshedAttachedProbes); },
    probeHashes() {
      return Object.fromEntries(refreshedAttachedProbes.map((probe) => [
        probe.name,
        probe.family === "geometry" ? refreshedGeometryProbeHash : refreshedLensProbeHash,
      ]));
    },
    attachProbe(request) {
      const probe = request.selector.startsWith("jlens/")
        ? {
            family: "lens",
            name: request.name || request.selector,
            layers: [0, 1],
            intrinsic_dim: 1,
            feature_space: "readout",
            word: request.selector.slice("jlens/".length),
            token_id: 7,
          }
        : {
            family: "geometry",
            name: request.name || request.selector,
            manifold: request.selector,
            top_n: request.top_n ?? 3,
            layers: [0, 1],
            node_labels: ["positive", "negative"],
            node_count: 2,
            domain: {},
            intrinsic_dim: 1,
            feature_space: "residual",
            is_affine: true,
            node_coords: [[1], [-1]],
          };
      refreshedAttachedProbes.push(probe);
      return structuredClone(probe);
    },
  };
  const instruments = new BrowserInstrumentRuntime(instrumentCompiler, runtime);
  let steeringCompiler = instrumentCompiler;
  let instrumentRefreshes = 0;
  let rejectDirectRefresh = false;
  const backend = new DrowseWebLlmBackend(
    { DROWSE_HOOK_ABI: "post-block-residual-v4", MLCEngine: class {} },
    {
      drowseVersion: "5.3.0-test",
      runtime,
      now: () => 123,
      createId: (kind) => `${kind}-${++id}`,
      prepareLoad() { return instruments; },
      compileSteering(context) {
        compiled.push({
          ...context,
          profiles: steeringCompiler.listProfiles().profiles.map((profile) => profile.name),
        });
        return { hookAbi: "post-block-residual-v4" };
      },
      steeringDelta(context) {
        deltas.push(context);
        return `${context.parent ?? "none"}->${context.child ?? "none"}`;
      },
      async authoringRequest(context) {
        authoring.push(context);
        context.onProgress({ event: "progress", data: { stage: "capturing" } });
        return { fitted: true };
      },
      async refreshAfterAuthoring() {
        await instruments.replaceCompiler(refreshedCompiler);
        steeringCompiler = refreshedCompiler;
        instrumentRefreshes += 1;
      },
      async refreshManifolds() {
        if (rejectDirectRefresh) {
          throw Object.assign(new Error("fixture refresh rejected"), {
            code: "ARTIFACT_REFRESH_REJECTED",
          });
        }
        await instruments.replaceCompiler(refreshedCompiler);
        steeringCompiler = refreshedCompiler;
        instrumentRefreshes += 1;
      },
    },
  );

  const service = (service, method, args = []) => ({ service, method, args });
  await assert.rejects(
    backend.request(
      service("sessions", "get"),
      () => {},
      new AbortController().signal,
    ),
    (error) => error.code === "MODEL_NOT_LOADED",
  );

  const loadProgress = [];
  const request = loadRequest({ onProgress: (event) => loadProgress.push(event) });
  assert.deepEqual(await backend.load(request), {
    prefillTokensPerSecond: 12,
    decodeTokensPerSecond: 6,
  });
  assert.deepEqual(loadProgress, [
    {
      event: "progress",
      data: {
        kind: "model_load",
        phase: "runtime_ready",
        message: "Model runtime ready",
      },
    },
    {
      event: "progress",
      data: {
        kind: "model_load",
        phase: "workbench_preparing",
        message: "Preparing response controls and insights",
      },
    },
    {
      event: "progress",
      data: {
        kind: "model_load",
        phase: "workbench_ready",
        message: "Workbench ready",
      },
    },
  ]);
  const session = await backend.request(
    service("sessions", "get"),
    () => {},
    new AbortController().signal,
  );
  assert.equal(session.model_id, "fixture-model");
  assert.equal(session.dtype, "q4f16_1");
  assert.equal(session.created, 123);
  assert.deepEqual(session.config, {
    temperature: 1,
    top_p: 0.9,
    top_k: null,
    max_tokens: 1024,
    system_prompt: null,
    thinking: null,
  });
  assert.equal(session.supports_thinking, false);
  assert.equal(session.is_base_model, false);
  assert.equal(session.role_substitution_supported, true);
  assert.equal(session.user_role_supported, true);
  assert.equal(session.scene_mode, true);
  assert.equal(session.default_user_role, "user");
  assert.equal(session.default_assistant_role, "model");
  assert.equal(session.instruments.length, 3);
  assert.equal(session.jlens_fitted, true);
  assert.deepEqual(
    session.instruments.find((item) => item.family === "lens").capabilities.preparations,
    [],
  );
  assert.equal(
    session.instruments.find((item) => item.family === "lens").capabilities.source_switch,
    false,
  );
  assert.deepEqual(session.profiles, ["local/demo"]);
  assert.deepEqual(await backend.request(
    service("profiles", "list"),
    () => {},
    new AbortController().signal,
  ), { profiles: [{ name: "local/demo", layers: [0, 1], metadata: {} }] });
  assert.deepEqual(await backend.request(
    service("profiles", "correlation", [["local/demo"]]),
    () => {},
    new AbortController().signal,
  ), {
    names: ["local/demo"],
    matrix: { "local/demo": { "local/demo": 1 } },
    layers_shared: {},
  });
  assert.equal((await backend.request(
    service("profiles", "pairwise", ["local/demo", "local/demo"]),
    () => {},
    new AbortController().signal,
  )).model, "fixture-model");
  assert.equal(session.instruments.every((item) => item.capabilities.token_readout === true), true);
  const capturedLens = {
    version: 1,
    scope: "token",
    provenance: "captured",
    instruments: {
      lens: {
        binding: { source: "fixture-jlens", steering: null },
        readout: { layers: [], aggregate: [] },
      },
    },
  };
  assert.deepEqual(
    await instruments.request(
      service("instruments", "tokenReadout", ["lens", "node", 0]),
      () => capturedLens,
    ),
    { measurements: capturedLens },
  );
  assert.deepEqual(
    await instruments.request(
      service("instruments", "tokenReadout", [
        "lens", "node", 0, { steered: true },
      ]),
      () => capturedLens,
    ),
    { measurements: capturedLens },
  );
  const captureOnlyInstruments = new BrowserInstrumentRuntime(instrumentCompiler);
  await assert.rejects(
    captureOnlyInstruments.request(
      service("instruments", "tokenReadout", [
        "lens", "node", 0, { steered: false },
      ]),
      () => capturedLens,
    ),
    (error) => error.code === "TOKEN_REPLAY_UNAVAILABLE",
  );
  const subsetReplay = new BrowserInstrumentRuntime(instrumentCompiler);
  subsetReplay.configureTokenReplay(async ({ family, options }) => {
    assert.equal(family, "lens");
    assert.equal(options.layers, "1");
    assert.deepEqual(instrumentCompiler.instrumentLiveState().lensLayers, [1]);
    return {
      version: 1,
      scope: "token",
      provenance: "captured",
      instruments: {
        lens: {
          binding: { source: "fixture-jlens", steering: null },
          readout: {
            layers: [{
              layer: 1,
              tokens: [{ token: "yes", id: 7, logprob: Math.log(0.75) }],
            }],
            aggregate: [{ token: "yes", strength: 0.75, com: 1, spread: 0 }],
          },
        },
      },
    };
  });
  const subsetReplayResult = await subsetReplay.request(
    service("instruments", "tokenReadout", [
      "lens", "node", 0, { layers: "1", topK: 1 },
    ]),
    () => capturedLens,
  );
  assert.deepEqual(
    subsetReplayResult.measurements.instruments.lens.readout.layers.map((row) => row.layer),
    [1],
  );
  assert.deepEqual(
    subsetReplayResult.measurements.instruments.lens.readout.aggregate.map((row) => row.token),
    ["yes"],
  );
  assert.deepEqual(instrumentCompiler.instrumentLiveState(), {
    geometry: true,
    lens: true,
    lensLayers: [0, 1],
    sae: true,
  });
  let exactWidthReplayCalls = 0;
  const exactWidthReplay = new BrowserInstrumentRuntime(instrumentCompiler);
  exactWidthReplay.configureTokenReplay(async ({ family, options }) => {
    exactWidthReplayCalls += 1;
    assert.equal(family, "lens");
    const width = options.topK;
    assert.ok(width === 5 || width === 8);
    const tokens = Array.from({ length: width }, (_, index) => ({
      token: `token-${index}`,
      id: index,
      logprob: -index - 0.25,
    }));
    return {
      version: 1,
      scope: "token",
      provenance: "captured",
      instruments: {
        lens: {
          binding: { source: "fixture-jlens", steering: null },
          readout: {
            layers: [0, 1].map((layer) => ({
              layer,
              tokens: tokens.map((token) => ({ ...token })),
            })),
            aggregate: tokens.map((token, index) => ({
              token: token.token,
              strength: 1 / (index + 1),
              com: 0.5,
              spread: 0.5,
            })),
          },
        },
      },
    };
  });
  for (const width of [5, 8]) {
    const replay = await exactWidthReplay.request(
      service("instruments", "tokenReadout", [
        "lens", "node", 0, { layers: "all", topK: width },
      ]),
      () => capturedLens,
    );
    assert.deepEqual(
      replay.measurements.instruments.lens.readout.layers.map((row) => row.tokens.length),
      [width, width],
    );
    assert.equal(replay.measurements.instruments.lens.readout.aggregate.length, width);
  }
  const replayCallsBeforeCapacityFailure = exactWidthReplayCalls;
  await assert.rejects(
    exactWidthReplay.request(
      service("instruments", "tokenReadout", [
        "lens", "node", 0, { layers: "all", topK: 9 },
      ]),
      () => capturedLens,
    ),
    (error) => error.code === "EXACT_READOUT_TOP_K_EXCEEDS_CAPACITY",
  );
  assert.equal(exactWidthReplayCalls, replayCallsBeforeCapacityFailure);
  await assert.rejects(
    captureOnlyInstruments.request(
      service("instruments", "tokenReadout", [
        "lens", "node", 0, { raw: false },
      ]),
      () => capturedLens,
    ),
    (error) => error.code === "TOKEN_REPLAY_UNAVAILABLE",
  );
  await assert.rejects(
    captureOnlyInstruments.request(
      service("instruments", "tokenReadout", [
        "lens", "node", 0, { topK: 5 },
      ]),
      () => capturedLens,
    ),
    (error) => error.code === "TOKEN_REPLAY_UNAVAILABLE",
  );
  await assert.rejects(
    captureOnlyInstruments.request(
      service("instruments", "tokenReadout", ["lens", "node", 1]),
      () => null,
    ),
    (error) => error.code === "TOKEN_REPLAY_UNAVAILABLE",
  );
  assert.deepEqual(await backend.request(
    service("instruments", "validateLensToken", ["yes"]),
    () => {},
    new AbortController().signal,
  ), { word: "yes", token_id: 7 });
  const attachedLens = await backend.request(
    service("probes", "attach", [{ selector: "jlens/yes", name: "watch yes" }]),
    () => {},
    new AbortController().signal,
  );
  assert.equal(attachedLens.name, "watch yes");
  assert.deepEqual(instruments.probeHashes(), { "watch yes": initialProbeHash });
  assert.deepEqual(preparedJlensWords, [["yes"], ["yes"]]);
  assert.deepEqual(
    (await backend.request(
      service("probes", "list"),
      () => {},
      new AbortController().signal,
    )).probes.map((probe) => probe.name),
    ["watch yes"],
  );
  await backend.refreshManifolds(new AbortController().signal);
  assert.equal(instrumentRefreshes, 1);
  assert.deepEqual(instruments.probeHashes(), { "watch yes": refreshedLensProbeHash });
  assert.deepEqual(
    (await backend.request(
      service("sessions", "get"),
      () => {},
      new AbortController().signal,
    )).profiles,
    ["local/demo", "local/fitted"],
  );
  assert.deepEqual(
    (await backend.request(
      service("probes", "list"),
      () => {},
      new AbortController().signal,
    )).probes.map((probe) => probe.name),
    ["watch yes"],
  );
  assert.deepEqual(
    (await backend.request(
      service("sessions", "get"),
      () => {},
      new AbortController().signal,
    )).probes,
    ["watch yes"],
  );
  assert.deepEqual(await backend.request(
    service("instruments", "sources", ["sae"]),
    () => {},
    new AbortController().signal,
  ), {
    sources: [{
      source: "fixture-sae",
      name: "Fixture SAE",
      kind: "catalog",
      provider: "catalog",
      active: true,
      layer: 1,
      features: 4,
    }],
  });
  saeSourceMetadata = {
    modelLayers: [0, 1],
    descriptionSource: {
      model: "fixture-model", source: "fixture-sae", repository: "fixture/sae", folder: "layer_1",
    },
  };
  const describedSaeSources = await backend.request(
    service("instruments", "sources", ["sae"]),
    () => {},
    new AbortController().signal,
  );
  assert.deepEqual(describedSaeSources.sources[0].model_layers, saeSourceMetadata.modelLayers);
  assert.deepEqual(describedSaeSources.sources[0].description_source, saeSourceMetadata.descriptionSource);
  assert.deepEqual(JSON.parse(JSON.stringify(describedSaeSources)), describedSaeSources);
  saeSourceMetadata = { descriptionSource: null };
  assert.equal((await backend.request(
    service("instruments", "sources", ["sae"]), () => {}, new AbortController().signal,
  )).sources[0].description_source, null);
  saeSourceMetadata = {};
  await assert.rejects(
    backend.request(
      service("instruments", "activateInstalledPack", [
        "lens",
        { source: "workspace-r" },
      ]),
      () => {},
      new AbortController().signal,
    ),
    (error) => error.code === "INSTRUMENT_PACK_SOURCE_MISMATCH",
  );
  assert.deepEqual(await backend.request(
    service("instruments", "activateInstalledPack", [
      "lens",
      { source: "fixture-jlens" },
    ]),
    () => {},
    new AbortController().signal,
  ), {
    state: "active",
    family: "lens",
    source: "fixture-jlens",
    live: { enabled: true, layers: [0, 1] },
    contextBindingSha256: null,
    reloadRequired: false,
  });

  const fitProgress = [];
  assert.deepEqual(await backend.request(
    service("manifolds", "fit", ["local", "fixture", {}]),
    (event) => fitProgress.push(event),
    new AbortController().signal,
  ), { fitted: true });
  assert.equal(authoring.length, 1);
  assert.equal(authoring[0].loadRequest, request);
  assert.equal(authoring[0].runtime, runtime);
  assert.deepEqual(fitProgress, [{ event: "progress", data: { stage: "capturing" } }]);
  assert.equal(instrumentRefreshes, 2);
  assert.deepEqual(preparedJlensWords, [["yes"], ["yes"], ["yes"]]);
  assert.deepEqual(
    (await backend.request(
      service("sessions", "get"),
      () => {},
      new AbortController().signal,
    )).profiles,
    ["local/demo", "local/fitted"],
  );
  assert.deepEqual(
    (await backend.request(
      service("manifolds", "list"),
      () => {},
      new AbortController().signal,
    )).manifolds.map((manifold) => manifold.name),
    ["demo", "fitted"],
  );
  assert.deepEqual(
    (await backend.request(
      service("probes", "list"),
      () => {},
      new AbortController().signal,
    )).probes.map((probe) => probe.name),
    ["watch yes"],
  );
  const fittedProbe = await backend.request(
    service("probes", "attach", [{ selector: "local/fitted", name: "watch fitted" }]),
    () => {},
    new AbortController().signal,
  );
  assert.equal(fittedProbe.family, "geometry");
  assert.deepEqual(instruments.probeHashes(), {
    "watch yes": refreshedLensProbeHash,
    "watch fitted": refreshedGeometryProbeHash,
  });
  assert.deepEqual(
    (await backend.request(
      service("sessions", "get"),
      () => {},
      new AbortController().signal,
    )).probes,
    ["watch fitted", "watch yes"],
  );
  const sessionBeforeRejectedRefresh = await backend.request(
    service("sessions", "get"),
    () => {},
    new AbortController().signal,
  );
  rejectDirectRefresh = true;
  await assert.rejects(
    backend.refreshManifolds(new AbortController().signal),
    (error) => error.code === "ARTIFACT_REFRESH_REJECTED",
  );
  assert.deepEqual(
    await backend.request(
      service("sessions", "get"),
      () => {},
      new AbortController().signal,
    ),
    sessionBeforeRejectedRefresh,
  );

  const events = [];
  let generationAdmissions = 0;
  assert.deepEqual(await backend.generate(
    {
      type: "submit",
      text: "question",
      authored_role: "user",
      generated_role: "assistant",
      steering: "0.5 local/fitted",
    },
    (event) => events.push(event),
    () => { generationAdmissions += 1; },
  ), { prefillTokensPerSecond: 12, decodeTokensPerSecond: 6 });
  assert.equal(generationAdmissions, 1);
  assert.deepEqual(events.map((event) => event.type), [
    "started", "tree_mutated", "tree_mutated", "token", "tree_mutated", "done",
  ]);
  assert.equal(compiled.length, 1);
  assert.equal(compiled[0].expression, "0.5 local/fitted");
  assert.equal(compiled[0].loadRequest, request);
  assert.deepEqual(compiled[0].profiles, ["local/demo", "local/fitted"]);
  assert.equal(calls.find((call) => call[0] === "generate")[1].hookProgram.hookAbi, "post-block-residual-v4");

  const tree = await backend.request(
    service("tree", "get"),
    () => {},
    new AbortController().signal,
  );
  assert.equal(tree.nodes.length, 3);
  assert.equal(tree.nodes[2].text, "answer");
  assert.deepEqual(tree.nodes[2].recipe.probe_hashes, {
    "watch fitted": refreshedGeometryProbeHash,
    "watch yes": refreshedLensProbeHash,
  });
  assert.deepEqual(await backend.request(
    service("tree", "edgeLabel", [tree.nodes[1].id, tree.nodes[2].id]),
    () => {},
    new AbortController().signal,
  ), { label: "none->0.5 local/fitted" });
  assert.equal(deltas.length, 1);
  assert.equal(deltas[0].parent, null);
  assert.equal(deltas[0].child, "0.5 local/fitted");
  assert.equal(deltas[0].loadRequest, request);
  const transcript = await backend.request(
    service("tree", "transcriptExport", [tree.nodes[2].id]),
    () => {},
    new AbortController().signal,
  );
  assert.equal(transcript.yaml.includes(refreshedGeometryProbeHash), true);
  assert.equal(transcript.yaml.includes(refreshedLensProbeHash), true);
  const transcriptMutationEvents = [];
  assert.deepEqual(await backend.request(
    service("tree", "transcriptLoad", [transcript.yaml, "default", true]),
    (event) => transcriptMutationEvents.push(event),
    new AbortController().signal,
  ).then((result) => result.guards), []);
  assert.equal(transcriptMutationEvents.length, 1);
  assert.equal(transcriptMutationEvents[0].event, "tree_mutated");
  assert.equal(transcriptMutationEvents[0].data.op, "restore");
  assert.equal(transcriptMutationEvents[0].data.cast.assistant.origin, "structural");
  assert.deepEqual(await backend.request(
    service("instruments", "setLive", ["lens", { enabled: true, layers: [1] }]),
    () => {},
    new AbortController().signal,
  ), { enabled: true, layers: [1] });
  assert.deepEqual(
    (await backend.request(
      service("sessions", "get"),
      () => {},
      new AbortController().signal,
    )).instruments.find((item) => item.family === "lens").live,
    { enabled: true, layers: [1] },
  );
  await assert.rejects(
    backend.request(
      service("instruments", "setLive", ["lens", { enabled: false, layers: [1] }]),
      () => {},
      new AbortController().signal,
    ),
    (error) => error.code === "INVALID_INSTRUMENT_REQUEST" && /disable/u.test(error.message),
  );
  await assert.rejects(
    backend.request(
      service("instruments", "setLive", ["geometry", { enabled: true, layers: [0] }]),
      () => {},
      new AbortController().signal,
    ),
    (error) => error.code === "INVALID_INSTRUMENT_REQUEST",
  );
  await assert.rejects(
    backend.request(
      service("instruments", "setLive", ["sae", { enabled: true, layers: [1] }]),
      () => {},
      new AbortController().signal,
    ),
    (error) => error.code === "INVALID_INSTRUMENT_REQUEST",
  );
  await assert.rejects(
    backend.request(
      service("instruments", "setLive", ["lens", { enabled: true, topK: 4 }]),
      () => {},
      new AbortController().signal,
    ),
    (error) => error.code === "INVALID_INSTRUMENT_REQUEST",
  );
  assert.deepEqual(await backend.request(
    service("instruments", "setLive", ["lens", { enabled: true }]),
    () => {},
    new AbortController().signal,
  ), { enabled: true, layers: [0, 1] });
  await backend.generate(
    {
      type: "submit",
      text: "another question",
      authored_role: "user",
      generated_role: "assistant",
      steering: null,
    },
    () => {},
  );
  assert.equal(compiled.at(-1).expression, "");
  const generationCalls = calls.filter((call) => call[0] === "generate").length;
  assert.equal(await backend.generate(
    {
      type: "submit",
      text: "authored only",
      authored_role: "user",
      generated_role: null,
      steering: null,
    },
    () => {},
  ), undefined);
  assert.equal(
    calls.filter((call) => call[0] === "generate").length,
    generationCalls,
  );
  await backend.stop();
  assert.equal(calls.at(-1)[0], "stop");
  await backend.unload();
  assert.equal(calls.at(-1)[0], "unload");

  for (const mode of ["resolve", "reject"]) {
    let signalStarted;
    let settle;
    const started = new Promise((resolve) => { signalStarted = resolve; });
    const cancelRuntime = {
      ...runtime,
      async load() {
        return { prefillTokensPerSecond: 12, decodeTokensPerSecond: 6 };
      },
      async unload() {},
      async streamGeneration(_plan, onToken) {
        await onToken({
          text: "partial",
          thinking: false,
          tokenId: 7,
          logprob: -0.3,
          perplexity: Math.exp(0.3),
          rawIndex: 0,
        });
        const interrupted = new Promise((resolve, reject) => {
          settle = mode === "resolve"
            ? resolve
            : () => reject(new DOMException("interrupted", "AbortError"));
        });
        signalStarted();
        await interrupted;
        return {
          text: "partial",
          thinkingText: null,
          tokens: 1,
          finishReason: "stop",
          usage: { promptTokens: 2, completionTokens: 1, totalTokens: 3 },
          meanLogprob: -0.3,
          meanSurprise: 0.3,
          prefillTokensPerSecond: 12,
          decodeTokensPerSecond: 6,
        };
      },
      async stop() { settle?.(); },
    };
    const cancelBackend = new DrowseWebLlmBackend(
      { DROWSE_HOOK_ABI: "post-block-residual-v4", MLCEngine: class {} },
      {
        drowseVersion: "5.3.0-test",
        runtime: cancelRuntime,
        createId: ids(`cancel-${mode}`),
      },
    );
    await cancelBackend.load(loadRequest());
    const cancelEvents = [];
    const pending = cancelBackend.generate({
      type: "submit",
      text: "preserve this prompt",
      authored_role: "user",
      generated_role: "assistant",
    }, (event) => cancelEvents.push(event));
    await started;
    await cancelBackend.stop("user");
    await pending;
    const cancelledTree = await cancelBackend.request(
      service("tree", "get"),
      () => {},
      new AbortController().signal,
    );
    assert.equal(cancelledTree.nodes[2].text, "partial");
    assert.equal(cancelledTree.nodes[2].finish_reason, "cancelled");
    assert.equal(cancelEvents.find((event) => event.type === "done").result.finish_reason, "cancelled");
    assert.equal(cancelEvents.some(
      (event) => event.type === "tree_mutated" && event.op === "delete",
    ), false);
    await cancelBackend.unload();
  }

  let retryUnloadFailure = true;
  let retryUnloadAttempts = 0;
  const retryUnloadBackend = new DrowseWebLlmBackend(
    { DROWSE_HOOK_ABI: "post-block-residual-v4", MLCEngine: class {} },
    {
      drowseVersion: "5.3.0-test",
      runtime: {
        ...runtime,
        async unload() {
          retryUnloadAttempts += 1;
          if (retryUnloadFailure) throw new Error("fixture backend unload failed");
        },
      },
      now: () => 127,
      createId: ids("retry-unload"),
    },
  );
  await retryUnloadBackend.load(loadRequest());
  const retrySession = await retryUnloadBackend.request(
    service("sessions", "get"),
    () => {},
    new AbortController().signal,
  );
  await assert.rejects(
    retryUnloadBackend.unload(),
    /fixture backend unload failed/,
  );
  assert.deepEqual(await retryUnloadBackend.request(
    service("sessions", "get"),
    () => {},
    new AbortController().signal,
  ), retrySession);
  retryUnloadFailure = false;
  await retryUnloadBackend.unload();
  assert.equal(retryUnloadAttempts, 2);
  await assert.rejects(
    retryUnloadBackend.request(
      service("sessions", "get"),
      () => {},
      new AbortController().signal,
    ),
    (error) => error.code === "MODEL_NOT_LOADED",
  );

  let setupCleanupFailure = true;
  let setupCleanupAttempts = 0;
  const setupCleanupBackend = new DrowseWebLlmBackend(
    { DROWSE_HOOK_ABI: "post-block-residual-v4", MLCEngine: class {} },
    {
      drowseVersion: "5.3.0-test",
      runtime: {
        ...runtime,
        async unload() {
          setupCleanupAttempts += 1;
          if (setupCleanupFailure) throw new Error("fixture setup cleanup failed");
        },
      },
      prepareLoad() { throw new Error("fixture instrument setup failed"); },
      now: () => 128,
      createId: ids("setup-cleanup"),
    },
  );
  await assert.rejects(
    setupCleanupBackend.load(loadRequest()),
    (error) => {
      assert.equal(error.code, "RUNTIME_CLEANUP_FAILED");
      assert.match(error.message, /fixture instrument setup failed/);
      assert.match(error.message, /fixture setup cleanup failed/);
      return true;
    },
  );
  assert.equal(setupCleanupAttempts, 1);
  setupCleanupFailure = false;
  await setupCleanupBackend.unload();
  assert.equal(setupCleanupAttempts, 2);

  const uncompiled = new DrowseWebLlmBackend(
    { DROWSE_HOOK_ABI: "post-block-residual-v4", MLCEngine: class {} },
    { drowseVersion: "5.3.0-test", runtime, now: () => 124, createId: ids("uncompiled") },
  );
  await uncompiled.load(loadRequest());
  await assert.rejects(
    uncompiled.generate(
      {
        type: "submit",
        text: "question",
        authored_role: "user",
        generated_role: "assistant",
        steering: "0.5 honest",
      },
      () => {},
    ),
    (error) => error.code === "STEERING_COMPILER_UNAVAILABLE",
  );
  await uncompiled.unload();

  const plainTemplateBackend = new DrowseWebLlmBackend(
    { DROWSE_HOOK_ABI: "post-block-residual-v4", MLCEngine: class {} },
    {
      drowseVersion: "5.3.0-test",
      runtime: {
        ...runtime,
        runtimeCapabilities() {
          return {
            topK: true,
            forcedReplay: true,
            replayScoring: true,
            tokenizer: true,
            namedRoles: false,
            userSeatGeneration: false,
          };
        },
      },
      now: () => 125,
      createId: ids("plain-template"),
    },
  );
  await plainTemplateBackend.load(loadRequest());
  const plainTemplateSession = await plainTemplateBackend.request(
    service("sessions", "get"),
    () => {},
    new AbortController().signal,
  );
  assert.equal(plainTemplateSession.role_substitution_supported, false);
  assert.equal(plainTemplateSession.user_role_supported, false);
  assert.equal(plainTemplateSession.scene_mode, false);
  await plainTemplateBackend.unload();

  const baseModelBackend = new DrowseWebLlmBackend(
    { DROWSE_HOOK_ABI: "post-block-residual-v4", MLCEngine: class {} },
    {
      drowseVersion: "5.3.0-test",
      runtime: {
        ...runtime,
        runtimeCapabilities() {
          return {
            topK: true,
            forcedReplay: true,
            replayScoring: true,
            tokenizer: true,
            namedRoles: true,
            userSeatGeneration: true,
            sceneStitching: true,
            baseModel: true,
          };
        },
      },
      now: () => 126,
      createId: ids("base-model"),
    },
  );
  await baseModelBackend.load(loadRequest());
  const baseModelSession = await baseModelBackend.request(
    service("sessions", "get"),
    () => {},
    new AbortController().signal,
  );
  assert.equal(baseModelSession.is_base_model, true);
  assert.equal(baseModelSession.role_substitution_supported, false);
  assert.equal(baseModelSession.user_role_supported, false);
  assert.equal(baseModelSession.scene_mode, false);
  await baseModelBackend.unload();

  const unattestedThinkingBackend = new DrowseWebLlmBackend(
    { DROWSE_HOOK_ABI: "post-block-residual-v4", MLCEngine: class {} },
    { drowseVersion: "5.3.0-test", runtime, now: () => 126, createId: ids("unattested-thinking") },
  );
  await unattestedThinkingBackend.load(loadRequest({
    variant: {
      ...loadRequest().variant,
      thinkingProfile: {
        start: "<think>",
        end: "</think>",
        startsInThinking: false,
        startTokenIds: [1],
        endTokenIds: [2],
      },
    },
  }));
  const unattestedThinkingSession = await unattestedThinkingBackend.request(
    service("sessions", "get"),
    () => {},
    new AbortController().signal,
  );
  assert.equal(unattestedThinkingSession.supports_thinking, false);
  await unattestedThinkingBackend.unload();

  await backend.load(loadRequest({ runtimeClass: "apple-mobile-webkit" }));
  const appleMobileSession = await backend.request(
    service("sessions", "get"),
    () => {},
    new AbortController().signal,
  );
  assert.equal(appleMobileSession.config.max_tokens, 256);
  const cappedAppleMobileSession = await backend.request(
    service("sessions", "patch", [{ max_tokens: 8_192 }]),
    () => {},
    new AbortController().signal,
  );
  assert.equal(cappedAppleMobileSession.config.max_tokens, 256);
  await backend.generate({
    type: "submit",
    text: "Keep it short",
    authored_role: "user",
    generated_role: "assistant",
    raw: false,
    sampling: { max_tokens: 8_192 },
  }, () => {});
  const appleMobilePlan = calls.findLast(([kind]) => kind === "generate")[1];
  assert.equal(appleMobilePlan.sampling.max_tokens, 256);
  assert.equal(appleMobilePlan.maxOutputTokens, 256);

  const aborted = new AbortController();
  aborted.abort(new DOMException("cancelled", "AbortError"));
  await assert.rejects(
    backend.load(loadRequest({ signal: aborted.signal })),
    (error) => error.name === "AbortError",
  );
  assert.equal(calls.at(-1)[0], "unload");

  console.log("Drowse WebLLM backend composition checks passed");
} finally {
  await server.close();
}

function loadRequest(overrides = {}) {
  return {
    model: {
      id: "fixture-model",
      displayName: "Fixture",
      description: "fixture",
      sourceUrl: "https://example.test/model",
      license: "Apache-2.0",
      languages: ["en"],
      variants: [],
    },
    variant: {
      id: "fixture-q4",
      thinkingProfile: null,
      runtimeIdentity: {
        quantization: "q4f16_1",
        hookAbi: "post-block-residual-v4",
      },
    },
    requiredCorePack: { id: "fixture-core" },
    contextTokens: 2048,
    adapter: {},
    artifacts: [],
    optionalPacks: [],
    activationSpool: {},
    signal: new AbortController().signal,
    onDeviceLost() {},
    ...overrides,
  };
}

function ids(prefix) {
  let value = 0;
  return (kind) => `${prefix}-${kind}-${++value}`;
}
