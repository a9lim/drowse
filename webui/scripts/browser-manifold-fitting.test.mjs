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
  const {
    BrowserManifoldFitting,
    resolveBrowserFitLayerIndices,
  } = await server.ssrLoadModule(
    "/src/hosted/runtime/browserManifoldFitting.ts",
  );
  const {
    compilerBackedFittingDependencies,
    prepareBrowserInstrumentDictionaries,
    withInstalledManifoldArchives,
  } = await server.ssrLoadModule(
    "/src/hosted/runtime/browserModelBackend.ts",
  );
  const leasedArchives = [new Blob(["one"]), new Blob(["two"])];
  let leasedIds = null;
  const compileLeaseResult = await withInstalledManifoldArchives({
    async request() { throw new Error("public export must not be used for compiler reads"); },
    async withDrowseArchiveArchives(ids, _signal, consumer) {
      leasedIds = [...ids];
      return consumer(leasedArchives);
    },
  }, ["manifolds/local/one", "manifolds/local/two"], new AbortController().signal,
  async (archives) => archives.reduce((sum, archive) => sum + archive.size, 0));
  assert.equal(compileLeaseResult, 6);
  assert.deepEqual(leasedIds, ["manifolds/local/one", "manifolds/local/two"]);
  const loadedWhiteners = new Map([[0, whitener()]]);
  let compilerLookups = 0;
  const validatedArchives = [];
  const compiler = {
    fittingWhiteners() { return loadedWhiteners; },
    exactSaeFitting(selector) { return { selector }; },
    async withInstalledManifolds(archives, signal) {
      signal.throwIfAborted();
      validatedArchives.push(...archives);
      return this;
    },
  };
  const compilerDependencies = compilerBackedFittingDependencies(async () => {
    compilerLookups += 1;
    return compiler;
  });
  assert.equal(await compilerDependencies.loadWhiteners(loadRequest()), loadedWhiteners);
  assert.deepEqual(
    await compilerDependencies.loadExactSae(loadRequest(), "sae/test", {}),
    { selector: "sae/test" },
  );
  assert.equal(compilerLookups, 2);
  const fittedArchive = new Blob(["fitted"]);
  await compilerDependencies.validateFittedPack(loadRequest(), fittedArchive);
  assert.deepEqual(validatedArchives, [fittedArchive]);
  assert.equal(compilerLookups, 3);
  const dictionaryCalls = [];
  const dictionaryProgress = [];
  const dictionaryRequest = {
    ...loadRequest(),
    onProgress(event) { dictionaryProgress.push(event); },
  };
  const saeDictionary = { bindingId: "fixture-sae" };
  const jlensDictionary = { bindingId: "fixture-jlens" };
  await prepareBrowserInstrumentDictionaries(dictionaryRequest, {
    saeGpuDictionary() { return saeDictionary; },
    jlensGpuDictionary() { return jlensDictionary; },
  }, {
    async setSaeDictionary(dictionary) {
      assert.equal(dictionary, saeDictionary);
      dictionaryCalls.push("stage-sae");
    },
    async setJlensDictionary(dictionary) {
      assert.equal(dictionary, jlensDictionary);
      dictionaryCalls.push("stage-jlens");
    },
    async warmSaeDictionary() { dictionaryCalls.push("upload-sae"); },
    async warmJlensDictionary() { dictionaryCalls.push("upload-jlens"); },
  });
  assert.deepEqual(dictionaryCalls, [
    "stage-sae",
    "stage-jlens",
    "upload-sae",
    "upload-jlens",
  ]);
  assert.deepEqual(dictionaryProgress.map((event) => event.data.phase), [
    "instrument_dictionaries_staging",
    "instrument_dictionaries_staged",
    "sae_dictionary_uploading",
    "sae_dictionary_ready",
    "jlens_dictionary_uploading",
    "jlens_dictionary_ready",
  ]);
  const mobileDictionaryCalls = [];
  await prepareBrowserInstrumentDictionaries({
    ...loadRequest(), runtimeClass: "apple-mobile-webkit",
  }, {
    saeGpuDictionary() { return saeDictionary; },
    jlensGpuDictionary() { return jlensDictionary; },
  }, {
    async setSaeDictionary(dictionary) {
      assert.equal(dictionary, saeDictionary);
      mobileDictionaryCalls.push("stage-sae");
    },
    async setJlensDictionary(dictionary) {
      assert.equal(dictionary, jlensDictionary);
      mobileDictionaryCalls.push("stage-jlens");
    },
    async warmSaeDictionary() { assert.fail("Do not warm unused SAE data on iOS"); },
    async warmJlensDictionary() { assert.fail("Do not warm unused J-lens data on iOS"); },
  });
  assert.deepEqual(mobileDictionaryCalls, ["stage-sae", "stage-jlens"]);
  const mobileAbort = new AbortController();
  await assert.rejects(prepareBrowserInstrumentDictionaries({
    ...loadRequest(), runtimeClass: "apple-mobile-webkit", signal: mobileAbort.signal,
  }, {
    saeGpuDictionary() { return saeDictionary; },
    jlensGpuDictionary() { return jlensDictionary; },
  }, {
    async setSaeDictionary() { mobileAbort.abort(); },
    async setJlensDictionary() { assert.fail("Stop staging after cancellation"); },
    async warmSaeDictionary() { assert.fail("Cancelled"); },
    async warmJlensDictionary() { assert.fail("Cancelled"); },
  }), { name: "AbortError" });
  const {
    defaultTuning,
    maxDimensionValidationMessage,
    tuningMessages,
  } = await server.ssrLoadModule("/src/drawers/manifold/shared.ts");
  const { installRuntimeCapabilities } = await server.ssrLoadModule(
    "/src/lib/runtime/registry.ts",
  );
  assert.equal(defaultTuning().maxDim, 8);
  assert.equal(defaultTuning(null).maxDim, 8);
  assert.equal(defaultTuning(4).maxDim, 4);
  assert.equal(defaultTuning(16).maxDim, 8);
  assert.equal(maxDimensionValidationMessage(9, null), null);
  installRuntimeCapabilities({ limits: { manifoldFitMaxIntrinsicDim: 4 } });
  assert.equal(defaultTuning().maxDim, 4);
  assert.equal(maxDimensionValidationMessage(4, 4), null);
  assert.equal(
    maxDimensionValidationMessage(5, 4),
    "Hosted browser fitting supports at most 4 dimensions.",
  );
  assert.deepEqual(
    tuningMessages({ ...defaultTuning(4), maxDim: 5 }, 4),
    ["Hosted browser fitting supports at most 4 dimensions."],
  );
  workspaceLayerSelectionCheck(resolveBrowserFitLayerIndices);
  const {
    browserFittedAuthoredPack,
    browserFittedFlatDiscoverPack,
  } = await server.ssrLoadModule(
    "/src/hosted/artifacts/fittedAuthoring.ts",
  );
  const { readVerifiedDrowseArchiveFile, validateDrowseArchive } = await server.ssrLoadModule(
    "/src/hosted/artifacts/drowseArchive.ts",
  );
  const artifactCalls = [];
  const fitLifecycle = [];
  const source = manifold("auto", { max_dim: 2 });
  const fitted = { ...source, fitted_for_session: true, layers_fitted: 1 };
  let getCount = 0;
  const artifacts = {
    async request(request) {
      artifactCalls.push(request);
      if (request.service === "manifolds" && request.method === "get") {
        getCount += 1;
        return getCount === 1 ? source : fitted;
      }
      if (request.service === "manifolds" && request.method === "drowseArchiveInstall") {
        fitLifecycle.push("install");
        assert.equal(request.args[0] instanceof Blob, true);
        assert.deepEqual(request.args[1], { force: true });
        return { installed: true };
      }
      throw new Error(`unexpected artifact request ${request.service}.${request.method}`);
    },
  };
  const preparedRows = [];
  const runtime = {
    async prepareCaptureRows(rows) {
      preparedRows.push(...rows);
      return rows.map((_row, index) => ({ inputIds: [1, 10 + index, 2], position: 2 }));
    },
    async capturePreparedRow() {
      throw new Error("the fake coordinator must not execute GPU capture");
    },
  };
  const plans = [];
  const flatInputs = [];
  const progress = [];
  const fitting = new BrowserManifoldFitting(artifacts, "browser-test", {
    async validateFittedPack(_request, archive) {
      assert.equal(archive instanceof Blob, true);
      fitLifecycle.push("validate");
    },
    async loadWhiteners() {
      return new Map([[0, whitener()]]);
    },
    createCoordinator(request) {
      assert.equal(request.contextTokens, 2048);
      return {
        async captureTopologyFoundation(plan, _capture, options) {
          plans.push(plan);
          options.onProgress?.({
            stage: "capturing",
            layer: null,
            completed: plan.descriptor.layers[0].rows,
            total: plan.descriptor.layers[0].rows,
          });
          return flatFoundation();
        },
      };
    },
    async buildFlatPack(input) {
      flatInputs.push(input);
      return new Blob(["fitted-pack"]);
    },
  });
  const result = await fitting.request({
    request: {
      service: "manifolds",
      method: "fit",
      args: ["local", "demo", {
        layers: [0],
        fit_mode: "auto",
        hyperparams: { max_dim: 3, persistence_frac: 0.25 },
      }],
    },
    loadRequest: loadRequest(),
    runtime,
    onProgress(event) { progress.push(event); },
    signal: new AbortController().signal,
  });

  assert.equal(result, fitted);
  assert.deepEqual(fitLifecycle, ["validate", "install"]);
  assert.deepEqual(artifactCalls.map((call) => `${call.service}.${call.method}`), [
    "manifolds.get",
    "manifolds.drowseArchiveInstall",
    "manifolds.get",
  ]);
  assert.equal(preparedRows.length, 4);
  assert.equal(preparedRows[0].system, "Answer in one short paragraph.");
  assert.deepEqual(preparedRows[0].messages, [
    { role: "user", content: "Who are you?" },
    { role: "assistant", content: "calm one" },
  ]);
  assert.deepEqual([...plans[0].groupOffsets], [0, 2, 4]);
  assert.equal(plans[0].descriptor.layers[0].rows, 4);
  assert.equal(plans[0].descriptor.layers[0].width, 2);
  assert.equal(plans[0].maxDimensions, 3);
  assert.equal(plans[0].persistenceFraction, 0.25);
  assert.equal(flatInputs.length, 1);
  assert.equal(flatInputs[0].manifold.fitMode, "auto");
  assert.deepEqual(flatInputs[0].manifold.hyperparams, {
    max_dim: 3,
    persistence_frac: 0.25,
  });
  assert.match(flatInputs[0].identity.captureSha256, /^[a-f0-9]{64}$/);
  assert.match(flatInputs[0].identity.captureRenderSha256, /^[a-f0-9]{64}$/);
  assert.equal(flatInputs[0].identity.contextBindingSha256, "b".repeat(64));
  assert.equal(flatInputs[0].identity.modelSourceFingerprint, "d".repeat(64));
  assert.deepEqual(flatInputs[0].closure, {
    source: { uri: "local", repository: null, revision: null },
    tags: [],
    template: null,
  });
  assert.deepEqual([...flatInputs[0].consensusGram], [1, -1, -1, 1]);
  assert.equal(flatInputs[0].evaluatedLayers.get(0).whitenedGram[0], 1);
  assert.equal(progress[0].event, "progress");

  await assert.rejects(
    fitting.request({
      request: {
        service: "manifolds",
        method: "fit",
        args: ["local", "demo", { fit_mode: "auto", hyperparams: { max_dim: 5 } }],
      },
      loadRequest: loadRequest(),
      runtime,
      onProgress() {},
      signal: new AbortController().signal,
    }),
    (error) => error.code === "BROWSER_FIT_PROFILE_LIMIT" && /up to 4 dimensions/.test(error.message),
  );
  assert.equal(preparedRows.length, 4);
  await assert.rejects(
    fitting.request({
      request: {
        service: "manifolds",
        method: "fit",
        args: ["local", "demo", { mystery: true }],
      },
      loadRequest: loadRequest(),
      runtime,
      onProgress() {},
      signal: new AbortController().signal,
    }),
    (error) => error.code === "INVALID_FIT_REQUEST" && /mystery/.test(error.message),
  );
  await assert.rejects(
    fitting.request({
      request: {
        service: "manifolds",
        method: "fit",
        args: ["local", "demo", { fit_mode: "pca", hyperparams: { k_nn: 3 } }],
      },
      loadRequest: loadRequest(),
      runtime,
      onProgress() {},
      signal: new AbortController().signal,
    }),
    (error) => error.code === "INVALID_FIT_REQUEST" && /k_nn/.test(error.message),
  );
  await assert.rejects(
    fitting.request({
      request: {
        service: "manifolds",
        method: "fit",
        args: ["local", "demo", { sae: "installed-sae" }],
      },
      loadRequest: loadRequest(),
      runtime,
      onProgress() {},
      signal: new AbortController().signal,
    }),
    (error) => error.code === "BROWSER_SAE_FITTING_PORT_UNAVAILABLE" &&
      /exact centroid encode\/decode port/.test(error.message),
  );

  const productionFitting = new BrowserManifoldFitting(artifacts, "browser-test", {
    allowAutomaticTopologyDiscovery: false,
    createCoordinator() {
      throw new Error("production policy must reject before fitting starts");
    },
  });
  await assert.rejects(
    productionFitting.request({
      request: {
        service: "manifolds",
        method: "fit",
        args: ["local", "demo", { fit_mode: "spectral" }],
      },
      loadRequest: loadRequest(),
      runtime,
      onProgress() {},
      signal: new AbortController().signal,
    }),
    (error) => error.code === "BROWSER_AUTOMATIC_TOPOLOGY_UNAVAILABLE",
  );

  await generatedManifoldCheck(BrowserManifoldFitting, defaultTuning().maxDim);
  await generatedProfileLimitPreflightCheck(BrowserManifoldFitting);
  await resumedGenerationCheck(BrowserManifoldFitting);
  await cancelledFitRollbackCheck(BrowserManifoldFitting);
  await orderedTemplateCaptureCheck(BrowserManifoldFitting);
  await exactTemplateScoringCheck(BrowserManifoldFitting);
  await authoredFittingCheck(
    BrowserManifoldFitting,
    browserFittedAuthoredPack,
    validateDrowseArchive,
    readVerifiedDrowseArchiveFile,
  );
  await monopolarExtractionCheck(
    BrowserManifoldFitting,
    browserFittedFlatDiscoverPack,
  );
  await saeExtractionCheck(
    BrowserManifoldFitting,
    browserFittedFlatDiscoverPack,
  );

  console.log("Browser manifold fitting, generation, and extraction checks passed");
} finally {
  await server.close();
}

function manifold(fitMode, hyperparams) {
  return {
    namespace: "local",
    name: "demo",
    description: "fixture",
    source: "local",
    tags: [],
    template_ref: null,
    fit_mode: fitMode,
    is_discover: true,
    domain: { type: "custom", embed_dim: 1 },
    domain_label: "custom(1d)",
    intrinsic_dim: 1,
    min_nodes: 2,
    node_count: 2,
    node_labels: ["calm", "alert"],
    node_coords: [],
    node_roles: [null, null],
    node_kinds: [null, null],
    hyperparams,
    fitted_models: [],
    tensor_variants: {},
    fitted_for_session: false,
    stale: false,
    resolved_fit_mode: null,
    nodes: [
      { label: "calm", coords: null, statements: ["calm one", "calm two"], role: null },
      { label: "alert", coords: null, statements: ["alert one", "alert two"], role: null },
    ],
  };
}

function workspaceLayerSelectionCheck(resolveLayers) {
  assert.deepEqual(resolveLayers("workspace", [0, 1, 2, 3]), [2]);
  assert.deepEqual(resolveLayers("workspace", [0, 1, 2, 3, 4, 5]), [2, 3, 4]);
  assert.deepEqual(
    resolveLayers("workspace", Array.from({ length: 11 }, (_, layer) => layer)),
    [4, 5, 6, 7, 8, 9],
  );
  assert.deepEqual(
    resolveLayers("workspace", Array.from({ length: 32 }, (_, layer) => layer)),
    Array.from({ length: 15 }, (_, index) => index + 13),
  );
  assert.deepEqual(resolveLayers("workspace", [0]), [0]);
  assert.deepEqual(resolveLayers("workspace", [0, 1]), [0, 1]);
  assert.deepEqual(resolveLayers("workspace", [0, 1, 2]), [1]);
  assert.deepEqual(resolveLayers("all", [0, 1, 2, 3]), [0, 1, 2, 3]);
  assert.deepEqual(resolveLayers([3, 1], [0, 1, 2, 3]), [1, 3]);
}

async function cancelledFitRollbackCheck(BrowserManifoldFitting) {
  const source = manifold("pca", { max_dim: 1 });
  const fitted = { ...source, fitted_for_session: true, layers_fitted: 1 };
  let installed = false;
  let installs = 0;
  let validationCalls = 0;
  let activeController;
  const fitting = new BrowserManifoldFitting({
    async request(request) {
      if (request.service === "manifolds" && request.method === "get") {
        return installed ? fitted : source;
      }
      if (request.service === "manifolds" && request.method === "drowseArchiveInstall") {
        installs += 1;
        installed = true;
        return { installed: true };
      }
      throw new Error(`unexpected cancellation artifact request ${request.service}.${request.method}`);
    },
  }, "browser-cancel-test", {
    async loadWhiteners() {
      return new Map([[0, whitener()]]);
    },
    createCoordinator() {
      return {
        async captureTopologyFoundation() {
          return flatFoundation();
        },
      };
    },
    async buildFlatPack() {
      return new Blob(["staged-fitted-pack"]);
    },
    async validateFittedPack() {
      validationCalls += 1;
      if (validationCalls === 1) activeController.abort();
    },
  });
  const runtime = {
    async prepareCaptureRows(rows) {
      return rows.map((_row, index) => ({ inputIds: [1, index + 2], position: 1 }));
    },
    async capturePreparedRow() {
      throw new Error("the cancellation fixture must not execute GPU capture");
    },
  };
  const fit = (signal) => fitting.request({
    request: {
      service: "manifolds",
      method: "fit",
      args: ["local", "demo", { layers: [0], fit_mode: "pca" }],
    },
    loadRequest: loadRequest(),
    runtime,
    onProgress() {},
    signal,
  });

  activeController = new AbortController();
  await assert.rejects(fit(activeController.signal), { name: "AbortError" });
  assert.equal(installed, false);
  assert.equal(installs, 0);

  activeController = new AbortController();
  assert.equal(await fit(activeController.signal), fitted);
  assert.equal(installed, true);
  assert.equal(installs, 1);
}

function loadRequest() {
  return {
    model: { id: "fixture-model" },
    variant: {
      structuredHookProfile: "standard-v2",
      runtimeIdentitySha256: "a".repeat(64),
      runtimeIdentity: {
        sourceModel: "fixture/source",
        sourceRevision: "c".repeat(40),
        convertedManifestSha256: "d".repeat(64),
        quantization: "fixture-fp32",
        tokenizerSha256: "e".repeat(64),
        chatTemplateSha256: "f".repeat(64),
        modelLibrarySha256: "1".repeat(64),
        runtimeAbi: "fixture-runtime-v1",
        hookAbi: "post-block-residual-v4",
        hiddenSize: 2,
        layerMap: [0],
      },
      contextProfiles: [{ contextTokens: 2048, bindingSha256: "b".repeat(64) }],
    },
    requiredCorePack: {},
    contextTokens: 2048,
    artifacts: [],
    activationSpool: {},
    signal: new AbortController().signal,
  };
}

function whitener() {
  return {
    columns: 2,
    rank: 2,
    ridge: 0.1,
    mean: new Float64Array(2),
    basis: new Float64Array([1, 0, 0, 1]),
    eigenvalues: new Float64Array([1, 1]),
    inverseScales: new Float64Array([1, 1]),
  };
}

function flatFoundation() {
  const layer = flatLayer();
  return {
    identity: {
      runtimeIdentitySha256: "a".repeat(64),
      contextBindingSha256: "b".repeat(64),
      captureSha256: "2".repeat(64),
    },
    layers: new Map([[0, layer]]),
    captureRetained: false,
    consensusGram: new Float64Array([1, -1, -1, 1]),
    topology: {
      operation: "topology",
      winnerName: "flat-pca",
      fitMode: "pca",
      intrinsicDimensions: 1,
      periodicDimensions: 0,
      persistentLoops: 0,
      usedFaintCycle: false,
      coordinates: new Float64Array([-1, 1]),
      embeddedCoordinates: new Float64Array([-1, 1]),
      candidates: [],
      diagnostics: {
        kind: "pca",
        perComponentVariance: new Float64Array([1]),
        cumulativeVariance: new Float64Array([1]),
        pickedDimensions: 1,
        threshold: 0.7,
      },
      winnerPlan: null,
    },
    dlsKept: null,
    neutralLayoutCoordinate: null,
    finalAffineLayers: new Map([[0, { ...layer, affineMap: null }]]),
    anchoredNodeCoordinates: new Float64Array([-1, 1]),
    curvedSurfaceLayers: null,
  };
}

function flatLayer() {
  return {
    operation: "affine_fisher",
    nodeCount: 2,
    columns: 2,
    components: 1,
    centroidMean: new Float64Array(2),
    mean: new Float64Array(2),
    basis: new Float64Array([1, 0]),
    nodeCoordinates: new Float64Array([-1, 1]),
    muCoordinates: new Float64Array([-1, 1]),
    whitenedGram: new Float64Array([1, -1, -1, 1]),
    neutralCrossGram: new Float64Array(2),
    explainedVariance: 1,
    mahalanobisShare: Math.sqrt(2),
  };
}

async function generatedManifoldCheck(BrowserManifoldFitting, defaultMaxDim) {
  const harness = authoringHarness();
  const fitting = new BrowserManifoldFitting(harness.artifacts, "browser-test", {
    async buildDiscoverPack(input) {
      harness.published.push(structuredClone(input));
      harness.current = detailFromMerged(input);
      return new Blob([`discover-${harness.published.length}`]);
    },
  });
  const progress = [];
  const result = await fitting.request({
    request: {
      service: "manifolds",
      method: "generate",
      args: [{
        namespace: "local",
        name: "personas",
        concepts: ["calm", "fox"],
        description: "two generated nodes",
        kind: "concrete",
        samples_per_prompt: 1,
        fit_mode: "auto",
        hyperparams: { max_dim: defaultMaxDim, persistence_frac: 0.25 },
        role_per_node: true,
      }],
    },
    loadRequest: loadRequest(),
    runtime: harness.runtime,
    onProgress(event) { progress.push(event); },
    signal: new AbortController().signal,
  });

  assert.equal(harness.generationPlans.length % 2, 0);
  const promptsPerConcept = harness.generationPlans.length / 2;
  assert.ok(promptsPerConcept > 1);
  assert.equal(harness.published.length, 2);
  assert.equal(harness.published[0].nodes.length, 1);
  assert.equal(harness.published[1].nodes.length, 2);
  assert.deepEqual(harness.published[1].nodes.map((node) => [node.label, node.role, node.kind]), [
    ["calm", "calm", "concrete"],
    ["fox", "fox", "concrete"],
  ]);
  assert.equal(harness.published[1].fitMode, "auto");
  assert.deepEqual(harness.published[1].hyperparams, {
    max_dim: 4,
    persistence_frac: 0.25,
  });
  assert.equal(harness.generationPlans[0].generationRoleName, "calm");
  assert.deepEqual(harness.generationPlans[0].sampling, {
    temperature: 1,
    top_p: 0.9,
    max_tokens: 256,
  });
  assert.equal(harness.generationPlans[0].thinking, false);
  assert.equal(
    harness.generationPlans[0].input.messages[0].content,
    "Answer in one short paragraph. You are a calm. Respond exactly as a calm would.",
  );
  assert.equal(harness.generationPlans[promptsPerConcept].generationRoleName, "fox");
  assert.equal(result.node_count, 2);
  assert.equal(progress.filter((event) => event.event === "progress" && event.data?.message).length,
    harness.generationPlans.length);
}

async function generatedProfileLimitPreflightCheck(BrowserManifoldFitting) {
  const harness = authoringHarness();
  const fitting = new BrowserManifoldFitting(harness.artifacts, "browser-test");
  await assert.rejects(
    fitting.request({
      request: {
        service: "manifolds",
        method: "generate",
        args: [{
          namespace: "local",
          name: "too_wide",
          concepts: ["calm", "alert"],
          fit_mode: "auto",
          hyperparams: { max_dim: 5 },
        }],
      },
      loadRequest: loadRequest(),
      runtime: harness.runtime,
      onProgress() {},
      signal: new AbortController().signal,
    }),
    (error) => error.code === "BROWSER_FIT_PROFILE_LIMIT" && /up to 4 dimensions/.test(error.message),
  );
  assert.equal(harness.generationPlans.length, 0);
  assert.equal(harness.published.length, 0);
}

async function resumedGenerationCheck(BrowserManifoldFitting) {
  const harness = authoringHarness();
  harness.current = detailFromMerged({
    namespace: "local",
    name: "personas",
    description: "old",
    fitMode: "auto",
    hyperparams: { max_dim: 2, persistence_frac: 0.4 },
    nodes: [{
      label: "calm",
      statements: ["retained response"],
      role: null,
      kind: "abstract",
    }],
  });
  const fitting = new BrowserManifoldFitting(harness.artifacts, "browser-test", {
    async buildDiscoverPack(input) {
      harness.published.push(structuredClone(input));
      harness.current = detailFromMerged(input);
      return new Blob(["resumed"]);
    },
  });
  await fitting.request({
    request: {
      service: "manifolds",
      method: "generate",
      args: [{
        namespace: "local",
        name: "personas",
        description: "refreshed",
        concepts: ["calm", "alert"],
        kind: "concrete",
        fit_mode: "pca",
        hyperparams: { max_dim: 1 },
      }],
    },
    loadRequest: loadRequest(),
    runtime: harness.runtime,
    onProgress() {},
    signal: new AbortController().signal,
  });

  assert.equal(harness.published.length, 1);
  assert.equal(harness.published[0].fitMode, "auto");
  assert.deepEqual(harness.published[0].hyperparams, {
    max_dim: 2,
    persistence_frac: 0.4,
  });
  assert.equal(harness.published[0].description, "refreshed");
  assert.deepEqual(harness.published[0].nodes[0], {
    label: "calm",
    statements: ["retained response"],
    role: null,
    kind: "abstract",
  });
  assert.equal(harness.published[0].nodes[1].kind, "concrete");
}

async function exactTemplateScoringCheck(BrowserManifoldFitting) {
  const template = {
    namespace: "local",
    name: "weekday",
    slot: "[DAY]",
    n_values: 2,
    n_contexts: 1,
    values: [" Monday", " Tuesday"],
    labels: ["monday", "tuesday"],
    description: "weekday fixture",
    tags: [],
    contexts: [{
      turns: [{ role: "user", content: "Complete the sentence." }],
      assistant: "Today is[DAY]",
    }],
  };
  const tokenizations = new Map([
    ["Today is", [10, 20]],
    ["Today is Monday", [10, 21, 31]],
    ["Today is Tuesday", [10, 22, 41, 42]],
  ]);
  const logprobs = new Map([
    [21, -0.5],
    [31, -0.25],
    [22, -0.2],
    [41, -0.3],
    [42, -0.4],
  ]);
  const plans = [];
  const compiled = [];
  const runtime = {
    async tokenizeText(text) {
      const ids = tokenizations.get(text);
      assert.ok(ids, `unexpected tokenizer input ${JSON.stringify(text)}`);
      return [...ids];
    },
    async streamGeneration(plan) {
      plans.push(plan);
      const forced = plan.replay.forcedPrefixTokenIds;
      for (let index = 0; index < forced.length; index += 1) {
        const tokenId = forced[index];
        await plan.onRawToken({
          text: `#${tokenId}`,
          tokenId,
          logprob: logprobs.get(tokenId) ?? -1,
          rawIndex: index,
          topAlts: null,
          replayScore: {
            emittedTokenId: tokenId,
            sampledTokenId: 999,
            forcedTokenId: tokenId,
            requestedLogprobs: plan.replay.scoreTokenIds.map((requested) => ({
              tokenId: requested,
              logprob: logprobs.get(requested) ?? -10,
            })),
            argmax: { tokenId: 999, logprob: -0.01 },
            topLogprobs: [{ tokenId: 999, logprob: -0.01 }],
          },
        });
      }
      return {
        text: "",
        thinkingText: null,
        tokens: forced.length,
        finishReason: "length",
        usage: {
          promptTokens: 1,
          completionTokens: forced.length,
          totalTokens: forced.length + 1,
        },
        meanLogprob: null,
        meanSurprise: null,
        prefillTokensPerSecond: null,
        decodeTokensPerSecond: null,
      };
    },
    async prepareCaptureRows() { throw new Error("unexpected capture preparation"); },
    async capturePreparedRow() { throw new Error("unexpected capture"); },
  };
  const fitting = new BrowserManifoldFitting({
    async request(request) {
      assert.deepEqual(request, {
        service: "templates",
        method: "get",
        args: ["local", "weekday"],
      });
      return template;
    },
  }, "browser-test");
  const result = await fitting.request({
    request: {
      service: "templates",
      method: "score",
      args: ["local", "weekday", " 0.5 calm "],
    },
    loadRequest: loadRequest(),
    runtime,
    onProgress() {},
    signal: new AbortController().signal,
    async compileSteering(expression) {
      compiled.push(expression);
      return { hookAbi: "post-block-residual-v4" };
    },
  });

  assert.deepEqual(compiled, ["0.5 calm"]);
  assert.deepEqual(plans.map((plan) => plan.replay.forcedPrefixTokenIds), [
    [10, 21, 31],
    [10, 22, 41, 42],
  ]);
  assert.deepEqual(plans.map((plan) => plan.replay.scoreTokenIds), [
    [21, 31],
    [22, 41, 42],
  ]);
  assert.equal(plans.every((plan) =>
    plan.generationSeat === "assistant" &&
    plan.sampling.temperature === 1 &&
    plan.sampling.top_p === 1 &&
    plan.sampling.top_k === 0 &&
    plan.thinking === undefined &&
    plan.steeringExpression === "0.5 calm"
  ), true);
  const [monday, tuesday] = result.contexts[0].choices;
  assert.deepEqual(
    [monday.n_tokens, monday.sum_logprob, monday.mean_logprob],
    [2, -0.75, -0.375],
  );
  assert.deepEqual(
    [tuesday.n_tokens, tuesday.sum_logprob, tuesday.mean_logprob],
    [3, -0.9, -0.3],
  );
  assert.ok(monday.prob_sum > tuesday.prob_sum);
  assert.ok(monday.prob_mean < tuesday.prob_mean);
  assert.ok(Math.abs(monday.prob_sum + tuesday.prob_sum - 1) < 1e-12);
  assert.ok(Math.abs(monday.prob_mean + tuesday.prob_mean - 1) < 1e-12);
}

async function orderedTemplateCaptureCheck(BrowserManifoldFitting) {
  const source = {
    ...manifold("pca", { max_dim: 1 }),
    template_ref: "local/ordered-template",
  };
  const template = {
    namespace: "local",
    name: "ordered-template",
    slot: "[VALUE]",
    n_values: 2,
    n_contexts: 1,
    values: ["calm", "alert"],
    labels: ["calm", "alert"],
    description: "ordered capture fixture",
    tags: [],
    contexts: [{
      turns: [
        { role: "system", content: "first history system" },
        { role: "user", content: "first question" },
        { role: "assistant", content: "first answer" },
        { role: "system", content: "second history system" },
        { role: "user", content: "final question" },
      ],
      assistant: "[VALUE]",
    }],
  };
  let capturedRows = null;
  const fitting = new BrowserManifoldFitting({
    async request(request) {
      if (request.service === "manifolds" && request.method === "get") return source;
      if (request.service === "templates" && request.method === "get") {
        assert.deepEqual(request.args, ["local", "ordered-template"]);
        return template;
      }
      throw new Error(`unexpected artifact request ${request.service}.${request.method}`);
    },
  }, "browser-test", {
    async loadWhiteners() {
      return new Map([[0, whitener()]]);
    },
    createCoordinator() {
      throw new Error("capture preparation should fail first");
    },
  });

  await assert.rejects(
    fitting.request({
      request: {
        service: "manifolds",
        method: "fit",
        args: ["local", "demo", { layers: [0] }],
      },
      loadRequest: loadRequest(),
      runtime: {
        async prepareCaptureRows(rows) {
          capturedRows = structuredClone(rows);
          throw new Error("ordered rows captured");
        },
        async capturePreparedRow() {
          throw new Error("unexpected capture");
        },
      },
      onProgress() {},
      signal: new AbortController().signal,
    }),
    /ordered rows captured/,
  );
  assert.equal(capturedRows[0].system, "Answer in one short paragraph.");
  assert.deepEqual(capturedRows[0].messages, [
    { role: "system", content: "first history system" },
    { role: "user", content: "first question" },
    { role: "assistant", content: "first answer" },
    { role: "system", content: "second history system" },
    { role: "user", content: "final question" },
    { role: "assistant", content: "calm one" },
  ]);
}

async function authoredFittingCheck(
  BrowserManifoldFitting,
  browserFittedAuthoredPack,
  validateDrowseArchive,
  readVerifiedDrowseArchiveFile,
) {
  for (const useSae of [false, true]) {
    const source = authoredManifold();
    const fitted = { ...source, fitted_for_session: true, layers_fitted: 1 };
    let archive = null;
    let getCount = 0;
    const plans = [];
    const artifacts = {
      async request(request) {
        if (request.service === "manifolds" && request.method === "get") {
          getCount += 1;
          return getCount === 1 ? source : fitted;
        }
        if (request.service === "manifolds" && request.method === "drowseArchiveInstall") {
          archive = request.args[0];
          return { installed: true };
        }
        throw new Error(`unexpected artifact request ${request.service}.${request.method}`);
      },
    };
    const runtime = {
      async prepareCaptureRows(rows) {
        return rows.map((_row, index) => ({ inputIds: [1, index + 2], position: 1 }));
      },
      async capturePreparedRow() {
        throw new Error("the authored coordinator owns capture");
      },
      async exactSaeFitting(selector) {
        if (!useSae) return null;
        assert.equal(selector, "installed-sae");
        return {
          layers: [0],
          transformCentroids(layer, centroids) {
            assert.equal(layer, 0);
            return {
              ...centroids,
              values: Float64Array.from(centroids.values, (value) => value + 1),
            };
          },
          provenance: {
            release: "local:test",
            revision: "8".repeat(64),
            fingerprint: "9".repeat(64),
            idsByLayer: new Map([[0, "local:test:layer-0"]]),
            fullCoverage: true,
          },
        };
      },
    };
    const fitting = new BrowserManifoldFitting(artifacts, "browser-test", {
      async validateFittedPack() {},
      async loadWhiteners() {
        return new Map([[0, whitener()]]);
      },
      createCoordinator() {
        return {
          async captureAuthoredFoundation(plan) {
            plans.push(plan);
            assert.deepEqual([...plan.coordinates], [-1, 0, 1]);
            assert.deepEqual([...plan.embeddedCoordinates], [-1, 0, 1]);
            assert.equal(plan.fitSigma, !useSae);
            if (useSae) {
              assert.deepEqual(
                [...plan.transformCentroids(0, {
                  rows: 3,
                  columns: 2,
                  values: new Float64Array([1, 2, 3, 4, 5, 6]),
                }).values],
                [2, 3, 4, 5, 6, 7],
              );
            }
            const layer = authoredCurvedLayer(!useSae);
            return {
              identity: {
                runtimeIdentitySha256: plan.descriptor.runtimeIdentitySha256,
                contextBindingSha256: plan.descriptor.contextBindingSha256,
                captureSha256: plan.descriptor.captureSha256,
              },
              layers: new Map([[0, layer]]),
              captureRetained: false,
              curvedSurfaceLayers: new Map([[0, layer]]),
            };
          },
        };
      },
      async buildAuthoredPack(input) {
        return browserFittedAuthoredPack(input);
      },
    });
    const result = await fitting.request({
      request: {
        service: "manifolds",
        method: "fit",
        args: ["local", "authored", useSae ? { sae: "installed-sae" } : { layers: [0] }],
      },
      loadRequest: loadRequest(),
      runtime,
      onProgress() {},
      signal: new AbortController().signal,
    });
    assert.equal(result, fitted);
    assert.equal(plans.length, 1);
    const verified = await validateDrowseArchive(archive);
    assert.equal(verified.fittedArtifacts.length, 1);
    const artifact = verified.fittedArtifacts[0];
    assert.equal(artifact.variant, useSae ? "sae" : "raw");
    const sidecar = JSON.parse(new TextDecoder().decode(
      await readVerifiedDrowseArchiveFile(verified, artifact.sidecarPath),
    ));
    assert.equal(sidecar.fit_mode, "authored");
    assert.equal(sidecar.method, useSae ? "manifold_sae" : "manifold_pca");
    assert.deepEqual(sidecar.domain, source.domain);
    assert.equal(sidecar.feature_space, useSae ? "sae-local:test" : "raw");
    assert.equal(sidecar.model_fingerprint, "a".repeat(64));
    assert.equal(sidecar.context_binding_sha256, "b".repeat(64));
    assert.equal(sidecar.model_source_fingerprint, "d".repeat(64));
    assert.deepEqual(sidecar.node_spread_per_layer, { 0: 14 });
    assert.deepEqual(sidecar.rbf_smoothing_per_layer, {
      0: { lambda: 0, edf: 3, gcv: -1 },
    });
    assert.deepEqual(Object.keys(sidecar.sigma_field_per_layer), useSae ? [] : ["0"]);
  }
}

function authoredManifold() {
  return {
    namespace: "local",
    name: "authored",
    description: "authored fixture",
    source: "local",
    tags: [],
    template_ref: null,
    fit_mode: "authored",
    is_discover: false,
    domain: { type: "custom", embed_dim: 1, bounds: null },
    domain_label: "custom(1d)",
    intrinsic_dim: 1,
    min_nodes: 3,
    node_count: 3,
    node_labels: ["low", "middle", "high"],
    node_coords: [[-1], [0], [1]],
    node_roles: [null, null, null],
    node_kinds: [null, null, null],
    hyperparams: {},
    fitted_models: [],
    tensor_variants: {},
    fitted_for_session: false,
    stale: false,
    resolved_fit_mode: null,
    nodes: [
      { label: "low", coords: [-1], statements: ["low one", "low two"], role: null },
      { label: "middle", coords: [0], statements: ["middle one", "middle two"], role: null },
      { label: "high", coords: [1], statements: ["high one", "high two"], role: null },
    ],
  };
}

function authoredCurvedLayer(withSigma) {
  const surface = {
    inputDimensions: 1,
    outputDimensions: 2,
    nodeCount: 3,
    lambda: 0,
    effectiveDegreesOfFreedom: 3,
    gcv: -1,
    nodes: new Float64Array([0, 0.5, 1]),
    coordinateOffset: new Float64Array([-1]),
    coordinateScale: new Float64Array([2]),
    weights: new Float64Array(6),
    polynomial: new Float64Array([-1, 0, 2, 0]),
  };
  const sigmaSurface = withSigma ? {
    ...surface,
    outputDimensions: 1,
    lambda: 0.1,
    effectiveDegreesOfFreedom: 2,
    gcv: 0.1,
    weights: new Float64Array(3),
    polynomial: new Float64Array(2),
  } : null;
  return {
    operation: "affine_fisher",
    nodeCount: 3,
    columns: 2,
    components: 2,
    centroidMean: new Float64Array(2),
    mean: new Float64Array(2),
    basis: new Float64Array([1, 0, 0, 1]),
    nodeCoordinates: new Float64Array([-1, 0, 0, 0, 1, 0]),
    muCoordinates: new Float64Array([-1, 0, 0, 0, 1, 0]),
    whitenedGram: new Float64Array([
      1, 0, 0,
      0, 4, 0,
      0, 0, 9,
    ]),
    neutralCrossGram: new Float64Array(3),
    explainedVariance: 1,
    mahalanobisShare: Math.sqrt(2),
    surface,
    sigmaSurface,
    sigmaSummary: withSigma ? { mean: 1, min: 1, max: 1, lambda: 0.1 } : null,
    origin: new Float64Array([0]),
    originDistance: 0,
  };
}

async function monopolarExtractionCheck(
  BrowserManifoldFitting,
  browserFittedFlatDiscoverPack,
) {
  const harness = authoringHarness();
  let fittedArchive = null;
  const flatInputs = [];
  const fitting = new BrowserManifoldFitting(harness.artifacts, "browser-test", {
    async validateFittedPack() {},
    async loadWhiteners() {
      return new Map([[0, {
        ...whitener(),
        mean: new Float64Array([1, 1]),
      }]]);
    },
    createCoordinator() {
      return {
        async captureNodeCentroids(plan) {
          return {
            identity: {
              runtimeIdentitySha256: plan.descriptor.runtimeIdentitySha256,
              contextBindingSha256: plan.descriptor.contextBindingSha256,
              captureSha256: plan.descriptor.captureSha256,
            },
            layers: new Map([[0, {
              rows: 1,
              columns: 2,
              values: new Float64Array([3, 4]),
            }]]),
            captureRetained: false,
          };
        },
        async captureTopologyFoundation() {
          throw new Error("monopolar extraction must not run topology selection");
        },
      };
    },
    async buildDiscoverPack(input) {
      harness.published.push(structuredClone(input));
      harness.current = detailFromMerged(input);
      return new Blob(["monopolar-source"]);
    },
    async buildFlatPack(input) {
      flatInputs.push(input);
      fittedArchive = await browserFittedFlatDiscoverPack(input);
      return fittedArchive;
    },
  });
  harness.onInstall = (archive) => {
    if (archive === fittedArchive) harness.exported = archive;
  };
  const result = await fitting.request({
    request: {
      service: "profiles",
      method: "extract",
      args: [{ concept: "calm", namespace: "local", kind: "abstract" }, "default"],
    },
    loadRequest: loadRequest(),
    runtime: harness.runtime,
    onProgress() {},
    signal: new AbortController().signal,
  });

  assert.equal(result.done, true);
  assert.equal(result.canonical, "local/calm");
  assert.deepEqual(result.profile, {
    name: "local/calm",
    layers: [0],
    metadata: {
      method: "manifold_pca",
      name: "calm",
      share_metric: "mahalanobis",
    },
  });
  assert.equal(harness.published[0].nodes.length, 1);
  assert.equal(harness.published[0].nodes[0].role, null);
  assert.equal(harness.generationPlans[0].generationRoleName, "someone_calm");
  assert.equal(flatInputs.length, 1);
  const layer = flatInputs[0].layers.get(0);
  assert.ok(Math.abs(layer.nodeCoordinates[0] - Math.sqrt(13)) < 1e-12);
  assert.ok(Math.abs(layer.basis[0] - 2 / Math.sqrt(13)) < 1e-12);
  assert.ok(Math.abs(layer.basis[1] - 3 / Math.sqrt(13)) < 1e-12);
  assert.deepEqual([...flatInputs[0].nodeCoordinates], [1]);
}

async function saeExtractionCheck(
  BrowserManifoldFitting,
  browserFittedFlatDiscoverPack,
) {
  const harness = authoringHarness();
  let fittedArchive = null;
  const flatInputs = [];
  const exactSae = {
    layers: [0],
    transformCentroids(_layer, centroids) {
      return { ...centroids, values: centroids.values.slice() };
    },
    provenance: {
      release: "fixture-sae",
      revision: "8".repeat(64),
      fingerprint: "9".repeat(64),
      idsByLayer: new Map([[0, "fixture-sae:layer-0"]]),
      fullCoverage: false,
    },
  };
  const fitting = new BrowserManifoldFitting(harness.artifacts, "browser-test", {
    async validateFittedPack() {},
    async loadWhiteners() {
      return new Map([[0, { ...whitener(), mean: new Float64Array([1, 1]) }]]);
    },
    async loadExactSae(_request, selector) {
      return selector === "fixture-sae" ? exactSae : null;
    },
    createCoordinator() {
      return {
        async captureNodeCentroids(plan) {
          return {
            identity: {
              runtimeIdentitySha256: plan.descriptor.runtimeIdentitySha256,
              contextBindingSha256: plan.descriptor.contextBindingSha256,
              captureSha256: plan.descriptor.captureSha256,
            },
            layers: new Map([[0, {
              rows: 1,
              columns: 2,
              values: new Float64Array([3, 4]),
            }]]),
            captureRetained: false,
          };
        },
        async captureTopologyFoundation() {
          throw new Error("monopolar SAE extraction must not run topology selection");
        },
      };
    },
    async buildDiscoverPack(input) {
      harness.published.push(structuredClone(input));
      harness.current = detailFromMerged(input);
      return new Blob(["sae-monopolar-source"]);
    },
    async buildFlatPack(input) {
      flatInputs.push(input);
      fittedArchive = await browserFittedFlatDiscoverPack(input);
      return fittedArchive;
    },
  });
  harness.onInstall = (archive) => {
    if (archive === fittedArchive) harness.exported = archive;
  };
  const result = await fitting.request({
    request: {
      service: "profiles",
      method: "extract",
      args: [{
        concept: "calm",
        namespace: "local",
        kind: "abstract",
        sae: "fixture-sae",
      }, "default"],
    },
    loadRequest: loadRequest(),
    runtime: harness.runtime,
    onProgress() {},
    signal: new AbortController().signal,
  });
  assert.equal(result.canonical, "local/calm:sae-fixture-sae");
  assert.equal(result.profile.name, "local/calm:sae-fixture-sae");
  assert.equal(result.profile.metadata.name, "calm:sae-fixture-sae");
  assert.equal(flatInputs.length, 1);
  assert.equal(flatInputs[0].sae.release, "fixture-sae");
  assert.equal(flatInputs[0].sae.fullCoverage, true);
  assert.deepEqual([...flatInputs[0].sae.idsByLayer], [[0, "fixture-sae:layer-0"]]);

  await assert.rejects(
    fitting.request({
      request: {
        service: "profiles",
        method: "extract",
        args: [{ concept: "other", sae: "fixture-sae", role: "pirate" }, "default"],
      },
      loadRequest: loadRequest(),
      runtime: harness.runtime,
      onProgress() {},
      signal: new AbortController().signal,
    }),
    (error) => error.code === "INVALID_EXTRACT_REQUEST" && /mutually exclusive/.test(error.message),
  );
}

function authoringHarness() {
  const harness = {
    current: null,
    exported: null,
    published: [],
    generationPlans: [],
    onInstall: null,
  };
  harness.artifacts = {
    async request(request) {
      if (request.service !== "manifolds") {
        throw new Error(`unexpected artifact request ${request.service}.${request.method}`);
      }
      if (request.method === "list") {
        return { manifolds: harness.current ? [harness.current] : [] };
      }
      if (request.method === "get") {
        assert.ok(harness.current);
        return harness.current;
      }
      if (request.method === "drowseArchiveInstall") {
        harness.onInstall?.(request.args[0]);
        return { installed: true };
      }
      if (request.method === "drowseArchiveExport") {
        assert.ok(harness.exported);
        return harness.exported;
      }
      throw new Error(`unexpected artifact request ${request.service}.${request.method}`);
    },
  };
  harness.runtime = {
    async streamGeneration(plan) {
      harness.generationPlans.push(plan);
      return {
        text: ` response ${harness.generationPlans.length} `,
        thinkingText: null,
        tokens: 1,
        finishReason: "stop",
        usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
        meanLogprob: null,
        meanSurprise: null,
        prefillTokensPerSecond: null,
        decodeTokensPerSecond: null,
      };
    },
    async prepareCaptureRows(rows) {
      return rows.map((_row, index) => ({ inputIds: [1, index + 2], position: 1 }));
    },
    async capturePreparedRow() {
      throw new Error("fake coordinator owns capture");
    },
  };
  return harness;
}

function detailFromMerged(input) {
  return {
    namespace: input.namespace,
    name: input.name,
    description: input.description,
    source: "local",
    tags: [],
    template_ref: null,
    fit_mode: input.fitMode,
    is_discover: true,
    domain: {},
    domain_label: `discover-${input.fitMode}`,
    intrinsic_dim: 0,
    min_nodes: null,
    node_count: input.nodes.length,
    node_labels: input.nodes.map((node) => node.label),
    node_coords: [],
    node_roles: input.nodes.map((node) => node.role),
    node_kinds: input.nodes.map((node) => node.kind),
    hyperparams: structuredClone(input.hyperparams),
    fitted_models: [],
    tensor_variants: {},
    fitted_for_session: false,
    stale: false,
    resolved_fit_mode: null,
    nodes: input.nodes.map((node) => ({
      label: node.label,
      coords: null,
      statements: [...node.statements],
      role: node.role,
    })),
  };
}
