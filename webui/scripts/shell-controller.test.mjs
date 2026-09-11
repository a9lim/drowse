import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import {
  FIXTURE_SIGNATURE,
  catalogBytes,
  catalogFixture,
} from "./catalog-fixture.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

const tests = [];

function test(name, run) {
  tests.push({ name, run });
}

function capabilities(overrides = {}) {
  return {
    checkedAt: 1,
    supported: true,
    secureContext: true,
    dedicatedWorker: true,
    indexedDb: true,
    opfs: true,
    webLocks: true,
    broadcastChannel: true,
    webAssembly: true,
    webGpu: {
      available: true,
      fallback: "hardware",
      features: ["shader-f16"],
      limits: {
        maxBufferSize: 16 * 1024 * 1024,
        maxStorageBufferBindingSize: 16 * 1024 * 1024,
        maxStorageBuffersPerShaderStage: 10,
        maxComputeWorkgroupStorageSize: 65_536,
      },
      adapterInfo: { vendor: "fixture" },
    },
    storage: {
      quotaBytes: 2_000_000_000,
      usageBytes: 100_000_000,
      availableBytes: 1_900_000_000,
      persisted: true,
    },
    signals: {
      deviceMemoryGiB: 8,
      logicalCpuCount: 8,
      mobile: false,
      language: "en-US",
      calibrationScore: 10,
    },
    deviceSignature: "fixture-device",
    runtimeVersion: "drowse-web-test-v1",
    issues: [],
    operations: {
      download: { available: true, reasons: [] },
      generation: { available: true, reasons: [] },
      fitting: { available: true, reasons: [] },
      manifold_artifacts: { available: false, reasons: [] },
      jlens_fitting: { available: false, reasons: [] },
      sae_training: { available: false, reasons: [] },
      session_admin: { available: false, reasons: [] },
      server_endpoints: { available: false, reasons: [] },
    },
    ...overrides,
  };
}

function recommendation(document) {
  const model = document.models[0];
  const variant = model.variants[0];
  return {
    model,
    variant,
    context: variant.contextProfiles[0],
    installed: false,
    eligible: true,
    proven: true,
    hardFailures: [],
    advisories: [],
    requiredStorageBytes: variant.downloadBytes + variant.requiredCorePackBytes,
    recommended: true,
    reason: "This measured configuration is the safest fit for this device.",
  };
}

function optionalInstrumentPack(variant, id, kind, bytes, sha256) {
  return {
    id,
    kind,
    displayName: kind === "jlens" ? "J-lens readouts" : "SAE features",
    license: "AGPL-3.0-or-later",
    sourceRepository: "a9lim/drowse-web-fixture",
    sourceRevision: variant.files[0].revision,
    bytes,
    required: false,
    runtimeIdentitySha256: variant.runtimeIdentitySha256,
    compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
    files: [{
      ...variant.files[0],
      path: `packs/${id}.bin`,
      role: "instrument",
      bytes,
      sha256,
    }],
  };
}

function removeFixtureJlens(variant) {
  variant.packs = variant.packs.filter((pack) => pack.kind !== "jlens");
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function createBundle(overrides = {}, document = catalogFixture()) {
  const verifiedCatalog = {
    document,
    exactBytes: new TextEncoder().encode(JSON.stringify(document)),
    signature: FIXTURE_SIGNATURE,
    allowDownloads: true,
    stale: false,
  };
  const calls = [];
  const catalogRequests = [];
  let runtimeSnapshot = {
    lifecycle: "uninitialized",
    modelVariantId: null,
    contextTokens: null,
    selectedModelVariantId: null,
    installedModelVariantIds: [],
    installedPackIds: [],
    loadRecords: [],
    download: { phase: "idle", startedAt: null, finishedAt: null, error: null },
    generation: { phase: "idle", startedAt: null, finishedAt: null, error: null },
    fitting: { phase: "idle", startedAt: null, finishedAt: null, error: null },
    error: null,
  };
  const runtimeListeners = new Set();
  const controller = {
    get snapshot() {
      return runtimeSnapshot;
    },
    subscribe(listener) {
      runtimeListeners.add(listener);
      listener(runtimeSnapshot);
      return () => runtimeListeners.delete(listener);
    },
    async check() {
      calls.push("check");
      return capabilities();
    },
    async catalog(request) {
      calls.push("catalog");
      catalogRequests.push(request);
      return verifiedCatalog;
    },
    recommend(catalog) {
      calls.push("recommend");
      assert.deepEqual(
        catalog.document.models.map(({ id }) => id),
        verifiedCatalog.document.models
          .map(({ id }) => id),
      );
      return [recommendation(document)];
    },
    async requestPersistence() {
      calls.push("persist");
      return true;
    },
    async download(modelVariantId, onProgress) {
      calls.push(`download:${modelVariantId}`);
      return { modelVariantId, installed: true, cancelled: false };
    },
    async downloadPack(modelVariantId, packId, onProgress) {
      calls.push(`download-pack:${modelVariantId}:${packId}`);
      return { modelVariantId, packId, installed: true, cancelled: false };
    },
    async cancelDownload() {
      calls.push("cancel");
    },
    async refreshStorage() {
      calls.push("storage");
      return capabilities().storage;
    },
    async load(modelVariantId, contextTokens) {
      calls.push(`load:${modelVariantId}:${contextTokens}`);
    },
    async unload() {
      calls.push("unload");
    },
    ...overrides,
  };
  let disposed = 0;
  return {
    calls,
    catalogRequests,
    bundle: {
      controller,
      runtime: {},
      async dispose() {
        disposed += 1;
      },
    },
    disposed: () => disposed,
    emitRuntime(patch) {
      runtimeSnapshot = { ...runtimeSnapshot, ...patch };
      for (const listener of runtimeListeners) listener(runtimeSnapshot);
    },
  };
}

const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: {
    connection: { downlink: 10 },
    storage: { persist: async () => true },
  },
});

try {
  const { createShellController } = await server.ssrLoadModule(
    "/hosted/shell-controller.ts",
  );
  const {
    OPTIONAL_PACK_RESIDENT_BUDGET_BYTES,
    optionalPackHardwareBlock,
  } = await server.ssrLoadModule(
    "/src/hosted/runtime/optionalPackCompatibility.ts",
  );

  test("model deletion resets setup and preserves analysis packs", async () => {
    let removed;
    const fixture = createBundle({
      async deleteModel(id) {
        removed = id;
        fixture.emitRuntime({ installedModelVariantIds: [], selectedModelVariantId: null, lifecycle: "unloaded" });
      },
    });
    const shell = createShellController(fixture.bundle);
    await shell.check();
    const id = shell.current().models[0].id;
    const packs = shell.current().models[0].firstRunPacks.map(pack => pack.id);
    fixture.emitRuntime({ installedModelVariantIds: [id], installedPackIds: packs });
    assert.equal(shell.current().models[0].installed, true);
    await shell.deleteModel(id);
    assert.equal(removed, id);
    assert.equal(shell.current().models[0].installed, false);
    assert.equal(shell.current().models[0].setupComplete, false);
    assert.ok(shell.current().models[0].firstRunPacks.every(pack => pack.installed));
    assert.equal(shell.current().download.phase, "idle");
    assert.ok(fixture.calls.includes("storage"));
    shell.dispose();
  });

  test("model deletion failure retains the installed state", async () => {
    const fixture = createBundle({ async deleteModel() { throw new Error("Storage is busy"); } });
    const shell = createShellController(fixture.bundle);
    await shell.check();
    const id = shell.current().models[0].id;
    fixture.emitRuntime({ installedModelVariantIds: [id] });
    await assert.rejects(shell.deleteModel(id), /Storage is busy/);
    assert.equal(shell.current().models[0].installed, true);
    shell.dispose();
  });

  test("verified catalog unlocks downloads and recommends its measured model", async () => {
    const { bundle, calls, catalogRequests } = createBundle();
    const shell = createShellController(bundle);

    await shell.check();

    const snapshot = shell.current();
    assert.equal(snapshot.phase, "supported");
    assert.equal(snapshot.headline, "This device is ready");
    assert.equal(snapshot.download.available, true);
    assert.equal(snapshot.download.phase, "idle");
    assert.deepEqual(snapshot.models.filter((model) => model.catalogAvailable !== false).map((model) => model.id), [
      "smollm2-360m-q4f16",
    ]);
    assert.equal(snapshot.models[0].fit, "recommended");
    assert.ok(snapshot.models[0].estimatedDownloadSeconds);
    assert.deepEqual(calls, ["check", "catalog", "recommend"]);
    assert.deepEqual(catalogRequests, [{ offline: false, preferCached: false }]);
  });

  test("a base-only catalog keeps all four chat models visible without authorizing their downloads", async () => {
    const document = catalogFixture();
    document.models[0].modelType = "base";
    removeFixtureJlens(document.models[0].variants[0]);
    const { bundle, calls } = createBundle({}, document);
    const shell = createShellController(bundle);
    await shell.check();

    const snapshot = shell.current();
    assert.equal(snapshot.models.filter((model) => model.modelType === "base").length, 1);
    const chats = snapshot.models.filter((model) => model.modelType !== "base");
    assert.deepEqual(chats.map((model) => model.modelId), [
      "gemma3-1b-instruct", "gemma3-4b-instruct", "qwen3-1.7b", "qwen3-4b",
    ]);
    for (const chat of chats) {
      assert.equal(chat.catalogAvailable, false);
      assert.equal(chat.fit, "blocked");
      assert.equal(chat.setupComplete, false);
      assert.match(chat.reason, /not in the current signed download catalog/);
      await assert.rejects(shell.download(chat.id), /not in the current signed download catalog/);
    }
    assert.equal(calls.some((call) => call.startsWith("download:")), false);
    assert.equal(snapshot.download.available, true, "signed base downloads stay available");
    shell.dispose();
  });

  test("a restored signed chat entry replaces its unavailable listing without duplicates", async () => {
    let includeChat = false;
    const { bundle } = createBundle({
      recommend(verified) {
        const base = recommendation(verified.document);
        base.model = { ...base.model, modelType: "base" };
        if (!includeChat) return [base];
        return [base, {
          ...base,
          model: { ...base.model, id: "gemma3-1b-instruct", modelType: "chat", displayName: "Gemma 3 1B" },
          variant: { ...base.variant, id: "gemma3-1b-instruct-restored" },
        }];
      },
    });
    const shell = createShellController(bundle);
    await shell.check();
    assert.equal(shell.current().models.find((model) => model.modelId === "gemma3-1b-instruct").catalogAvailable, false);
    includeChat = true;
    await shell.check();
    const matches = shell.current().models.filter((model) => model.modelId === "gemma3-1b-instruct");
    assert.equal(matches.length, 1);
    assert.equal(matches[0].id, "gemma3-1b-instruct-restored");
    assert.notEqual(matches[0].catalogAvailable, false);
    assert.notEqual(matches[0].fit, "blocked");
    assert.equal(shell.current().models.filter((model) => model.modelType !== "base").length, 4);
    shell.dispose();
  });

  test("base models include available SAE features on mobile and work without an SAE", async () => {
    const document = catalogFixture();
    document.models[0].modelType = "base";
    const variant = document.models[0].variants[0];
    variant.packs.push(optionalInstrumentPack(variant, "base-sae", "sae", 64, "a".repeat(64)));
    const { bundle, calls } = createBundle({
      async check() { return capabilities({ signals: { ...capabilities().signals, appleMobile: true } }); },
    }, document);
    const shell = createShellController(bundle);
    await shell.check();
    const option = shell.current().models[0];
    assert.equal(option.modelType, "base");
    assert.match(option.context, /Text completion/);
    assert.deepEqual(option.firstRunPacks.map((pack) => [pack.kind, pack.requiredForSetup, pack.selected]),
      [["jlens", false, false], ["sae", true, true]]);
    await shell.download(variant.id);
    assert.ok(calls.includes(`download-pack:${variant.id}:base-sae`));
    assert.ok(!calls.includes(`download-pack:${variant.id}:smollm2-jlens`));
    assert.equal(shell.current().models[0].setupComplete, true);

    variant.packs = variant.packs.filter((pack) => pack.kind !== "sae");
    const unavailable = createShellController(createBundle({}, document).bundle);
    await unavailable.check();
    assert.notEqual(unavailable.current().models[0].fit, "blocked");
    assert.equal(unavailable.current().models[0].setupIssue, undefined);
    variant.packs = variant.packs.filter((pack) => pack.kind === "core");
    const generationOnly = createShellController(createBundle({}, document).bundle);
    await generationOnly.check();
    assert.notEqual(generationOnly.current().models[0].fit, "blocked");
    await generationOnly.download(variant.id);
    assert.equal(generationOnly.current().models[0].setupComplete, true);
  });

  test("base models cannot bypass the combined instrument memory budget", async () => {
    const document = catalogFixture();
    document.models[0].modelType = "base";
    const variant = document.models[0].variants[0];
    variant.packs.push(optionalInstrumentPack(variant, "oversize-sae", "sae",
      OPTIONAL_PACK_RESIDENT_BUDGET_BYTES, "a".repeat(64)));
    const shell = createShellController(createBundle({
      async check() {
        const device = capabilities();
        device.webGpu.limits.maxBufferSize = 64 * 1024 * 1024;
        device.webGpu.limits.maxStorageBufferBindingSize = 64 * 1024 * 1024;
        return device;
      },
    }, document).bundle);
    await shell.check();
    assert.notEqual(shell.current().models[0].fit, "blocked");
    const lens = shell.current().models[0].firstRunPacks.find((pack) => pack.kind === "jlens");
    assert.throws(() => shell.setOptionalPackSelected(variant.id, lens.id, true));
    assert.throws(() => shell.setOptionalPackSelected(variant.id, "oversize-sae", false), /SAE is required/);
    shell.setOptionalPackSelected(variant.id, lens.id, false);
    await shell.download(variant.id);
    assert.equal(shell.current().models[0].setupComplete, true);
  });

  test("Firefox is called incompatible when every signed model exceeds its limits", async () => {
    const limitReason =
      "smollm2-360m-q4f16 requires maxStorageBuffersPerShaderStage >= 10; this adapter reports 9";
    const { bundle } = createBundle({
      async check() {
        return capabilities({
          signals: {
            ...capabilities().signals,
            runtimeClass: "desktop-gecko",
            appleMobile: false,
          },
        });
      },
      recommend(verified) {
        return [{
          ...recommendation(verified.document),
          eligible: false,
          proven: false,
          recommended: false,
          hardFailures: [{
            code: "WEBGPU_LIMIT_TOO_LOW",
            message: limitReason,
            severity: "hard",
          }],
          reason: limitReason,
        }];
      },
    });
    const shell = createShellController(bundle);

    await shell.check();

    const snapshot = shell.current();
    assert.equal(snapshot.phase, "supported");
    assert.equal(snapshot.headline, "Firefox is incompatible with the current model runtime");
    assert.equal(
      snapshot.detail,
      "Firefox is incompatible with the current signed model runtime. This browser provides 9 of the 10 graphics buffers this model needs. Update your browser and graphics driver, then run the device check again. If it still fails, use another supported browser.",
    );
    assert.equal(snapshot.models[0].fit, "blocked");
    assert.equal(snapshot.download.available, false);
    assert.equal(snapshot.download.phase, "locked");
    assert.equal(
      snapshot.download.reason,
      "Firefox is incompatible with the current signed model runtime on this device.",
    );
  });

  test("Firefox remains eligible when its adapter satisfies the signed runtime limits", async () => {
    const { bundle } = createBundle({
      async check() {
        return capabilities({
          signals: {
            ...capabilities().signals,
            runtimeClass: "desktop-gecko",
            appleMobile: false,
          },
        });
      },
    });
    const shell = createShellController(bundle);

    await shell.check();

    const snapshot = shell.current();
    assert.equal(snapshot.headline, "This device is ready");
    assert.equal(snapshot.models[0].fit, "recommended");
    assert.equal(snapshot.download.available, true);
  });

  test("GPT-2 Base includes its parameter count in the model label", async () => {
    const { bundle } = createBundle({
      recommend(verified) {
        const entry = recommendation(verified.document);
        entry.model.id = "gpt2-base";
        entry.model.displayName = "GPT-2 Base";
        return [entry];
      },
    });
    const shell = createShellController(bundle);
    await shell.check();
    assert.equal(shell.current().models.find(model => model.modelId === "gpt2-base").name, "GPT-2 Base (124M)");
  });

  test("Gemma 3 270M remains available from the signed catalog", async () => {
    const { bundle } = createBundle({
      recommend(verified) {
        const retained = recommendation(verified.document);
        const removed = structuredClone(retained);
        removed.model.id = "gemma3-270m-instruct";
        removed.model.displayName = "Gemma 3 270M";
        removed.variant.id = "gemma3-270m-instruct-q4f32_1";
        return [removed, retained];
      },
    });
    const shell = createShellController(bundle);

    await shell.check();

    assert.deepEqual(shell.current().models.filter((model) => model.catalogAvailable !== false).map((model) => model.id), [
      "gemma3-270m-instruct-q4f32_1",
      "smollm2-360m-q4f16",
    ]);
  });

  test("model choices are grouped by family instead of interleaved by size", async () => {
    const { bundle } = createBundle({
      recommend(verified) {
        const base = recommendation(verified.document);
        const option = (modelId, displayName, variantId, tier, downloadBytes, recommended) => ({
          ...structuredClone(base),
          model: {
            ...structuredClone(base.model),
            id: modelId,
            displayName,
          },
          variant: {
            ...structuredClone(base.variant),
            id: variantId,
            tier,
            downloadBytes,
          },
          recommended,
        });
        return [
          option("gemma3-1b-instruct", "Gemma 3 1B", "gemma3-1b-q4", "balanced", 693, true),
          option("qwen3-1.7b", "Qwen3-1.7B", "qwen3-1.7b-q4", "balanced", 1_200, false),
          option("gemma3-4b-instruct", "Gemma 3 4B", "gemma3-4b-q4", "quality", 2_500, false),
          option("qwen3-4b", "Qwen3-4B", "qwen3-4b-q4", "quality", 2_600, false),
        ];
      },
    });
    const shell = createShellController(bundle);

    await shell.check();

    assert.deepEqual(shell.current().models.map((model) => model.id), [
      "gemma3-1b-q4",
      "gemma3-4b-q4",
      "qwen3-1.7b-q4",
      "qwen3-4b-q4",
    ]);
  });

  test("compatible hardware is not described as release-ready when the runtime is locked", async () => {
    const { bundle } = createBundle({
      async check() {
        return capabilities({
          operations: {
            ...capabilities().operations,
            download: {
              available: false,
              reasons: [{
                code: "SIGNED_DISTRIBUTION_UNAVAILABLE",
                message: "Signed model downloads have not passed release checks",
                severity: "hard",
              }],
            },
            generation: {
              available: false,
              reasons: [{
                code: "CUSTOM_MLC_BACKEND_UNAVAILABLE",
                message: "The local model engine has not passed release checks",
                severity: "hard",
              }],
            },
          },
        });
      },
    });
    const shell = createShellController(bundle);

    await shell.check();

    const snapshot = shell.current();
    assert.equal(snapshot.phase, "supported");
    assert.equal(snapshot.headline, "This hardware is compatible");
    assert.equal(snapshot.download.available, false);
    assert.equal(snapshot.runtime.available, false);
    assert.equal(snapshot.checks.find(({ id }) => id === "runtime")?.state, "warn");
  });

  test("a verified runtime is ready in a local preview as well as a release build", async () => {
    const { bundle } = createBundle();
    const shell = createShellController(bundle);
    await shell.check();
    const runtime = shell.current().checks.find(({ id }) => id === "runtime");
    assert.equal(runtime.state, "pass");
    assert.equal(runtime.detail, "The tested local engine is ready to load a model.");
  });

  test("data saver suppresses the pre-download network estimate", async () => {
    navigator.connection.saveData = true;
    try {
      const { bundle } = createBundle();
      const shell = createShellController(bundle);

      await shell.check();

      assert.equal(shell.current().models[0].estimatedDownloadSeconds, undefined);
    } finally {
      delete navigator.connection.saveData;
    }
  });

  test("confirmed OOM remains selectable only through an explicit retry", async () => {
    const seenOptions = [];
    const { bundle } = createBundle({
      recommend(verified) {
        return [{
          ...recommendation(verified.document),
          eligible: false,
          proven: false,
          recommended: false,
          hardFailures: [{
            code: "CONFIRMED_OOM",
            message: "This exact configuration previously ran out of memory",
            severity: "hard",
          }],
          reason: "This exact configuration previously ran out of memory",
        }];
      },
      async download(modelVariantId, _onProgress, options) {
        seenOptions.push(structuredClone(options));
        return { modelVariantId, installed: true, cancelled: false };
      },
    });
    const shell = createShellController(bundle);
    await shell.check();

    assert.equal(shell.current().models[0].fit, "uncertain");
    assert.equal(shell.current().models[0].requiresOomRetry, true);
    await shell.download("smollm2-360m-q4f16", { explicitUnsafeOverride: true });
    assert.deepEqual(seenOptions, [{
      contextTokens: 2048,
      explicitUnsafeOverride: true,
    }]);
  });

  test("ownership check requires both Web Locks and BroadcastChannel", async () => {
    const { bundle, calls } = createBundle({
      async check() {
        calls.push("check");
        return capabilities({
          supported: false,
          broadcastChannel: false,
          issues: [{
            code: "BROADCAST_CHANNEL_UNAVAILABLE",
            message: "BroadcastChannel is unavailable",
            severity: "hard",
          }],
        });
      },
    });
    const shell = createShellController(bundle);

    await shell.check();

    const snapshot = shell.current();
    assert.equal(snapshot.phase, "unsupported");
    assert.equal(
      snapshot.detail,
      "This browser cannot coordinate model use between Drowse tabs. Drowse will not switch to cloud processing.",
    );
    assert.equal(snapshot.checks.find(({ id }) => id === "ownership")?.state, "fail");
    assert.match(
      snapshot.checks.find(({ id }) => id === "ownership")?.detail ?? "",
      /another Drowse tab/,
    );
    assert.deepEqual(calls, ["check"]);
  });

  test("an insecure origin is explained as unsupported instead of a failed check", async () => {
    const { bundle, calls } = createBundle({
      async check() {
        calls.push("check");
        return capabilities({
          supported: false,
          secureContext: false,
          issues: [{
            code: "INSECURE_CONTEXT",
            message: "Drowse requires a secure HTTPS context",
            severity: "hard",
          }],
        });
      },
    });
    const shell = createShellController(bundle);

    await shell.check();

    const snapshot = shell.current();
    assert.equal(snapshot.phase, "unsupported");
    assert.equal(
      snapshot.detail,
      "Open Drowse over a secure HTTPS connection or on localhost. Drowse will not switch to cloud processing.",
    );
    assert.equal(snapshot.checks.find(({ id }) => id === "secure")?.state, "fail");
    assert.deepEqual(calls, ["check"]);
  });

  test("graphics checks show the selected adapter and conditional Windows guidance", async () => {
    const { windowsIntelGpuHint, windowsIntelGen9Blocked } = await server.ssrLoadModule("/src/lib/runtime/gpuRecovery.ts");
    const hint = windowsIntelGpuHint("intel", "Windows NT 10.0");
    assert.match(hint, /If this laptop has a dedicated GPU/u);
    assert.match(hint, /chrome:\/\/flags\/#force-high-performance-gpu/u);
    const chromeWindows = "Windows NT 10.0 Chrome/151.0";
    assert.equal(windowsIntelGen9Blocked("intel", "gen-9", chromeWindows), true);
    assert.equal(windowsIntelGen9Blocked("intel", "gen-12", chromeWindows), false);
    assert.equal(windowsIntelGen9Blocked("nvidia", "gen-9", chromeWindows), false);
    assert.equal(windowsIntelGen9Blocked("intel", "gen-9", "Macintosh Chrome/151.0"), false);
    assert.equal(windowsIntelGen9Blocked("intel", "gen-9", "Windows Firefox/150.0"), false);
    assert.equal(windowsIntelGen9Blocked("intel", undefined, chromeWindows), false);
    assert.equal(windowsIntelGpuHint("intel", "Macintosh"), null);
    assert.equal(windowsIntelGpuHint("nvidia", "Windows NT 10.0"), null);
    assert.equal(windowsIntelGpuHint(undefined, "Windows NT 10.0"), null);
    const { bundle } = createBundle({
      async check() {
        return capabilities({
          webGpu: { ...capabilities().webGpu, adapterInfo: { vendor: "intel", architecture: "gen-9" } },
          issues: [{ code: "WINDOWS_INTEL_GPU", message: hint, severity: "advisory" }],
        });
      },
    });
    const shell = createShellController(bundle);
    await shell.check();
    const graphics = shell.current().checks.find(({ id }) => id === "webgpu");
    assert.equal(graphics.state, "warn");
    assert.match(graphics.detail, /intel \/ gen-9/u);
    assert.match(graphics.detail, /does not guarantee stability/u);
    assert.match(graphics.detail, /chrome.exe/u);
  });

  test("the Windows Intel Gen9 block keeps its GPU-switch instructions in the shell", async () => {
    const { WINDOWS_INTEL_GEN9_BLOCK } = await server.ssrLoadModule("/src/lib/runtime/gpuRecovery.ts");
    const failure = { code: "WEBGPU_WINDOWS_INTEL_GEN9_BLOCKED", message: WINDOWS_INTEL_GEN9_BLOCK, severity: "hard" };
    const { bundle } = createBundle({
      async check() {
        const base = capabilities();
        return capabilities({
          supported: false,
          webGpu: { ...base.webGpu, adapterInfo: { vendor: "intel", architecture: "gen-9" } },
          issues: [failure],
          operations: { ...base.operations, generation: { available: false, reasons: [failure] } },
        });
      },
    });
    const shell = createShellController(bundle);
    await shell.check();
    const graphics = shell.current().checks.find(({ id }) => id === "webgpu");
    assert.equal(graphics.state, "fail");
    assert.match(graphics.detail, /Selected GPU: intel \/ gen-9/u);
    assert.match(graphics.detail, /force-high-performance-gpu/u);
    assert.equal(shell.current().runtime.available, false);
  });

  test("a failed compute calibration marks graphics support as unavailable", async () => {
    const calibrationFailure = {
      code: "WEBGPU_CALIBRATION_FAILED",
      message: "The WebGPU compute calibration failed: validation error",
      severity: "hard",
    };
    const { bundle } = createBundle({
      async check() {
        return capabilities({
          supported: false,
          signals: {
            ...capabilities().signals,
            runtimeClass: "desktop-gecko",
            appleMobile: false,
            calibrationScore: null,
          },
          issues: [calibrationFailure],
        });
      },
    });
    const shell = createShellController(bundle);

    await shell.check();

    const graphics = shell.current().checks.find(({ id }) => id === "webgpu");
    assert.equal(graphics?.state, "fail");
    assert.equal(
      graphics?.detail,
      "Selected GPU: fixture. Drowse could not confirm compatible hardware-accelerated graphics. Update your browser and graphics driver, then run the device check again.",
    );
  });

  test("persistence precedes download and progress reaches the shell snapshot", async () => {
    const progressVariant = catalogFixture().models[0].variants[0];
    const firstRunBytes = progressVariant.downloadBytes +
      progressVariant.requiredCorePackBytes +
      progressVariant.packs.find((pack) => pack.kind === "jlens").bytes;
    const progress = {
      modelVariantId: "smollm2-360m-q4f16",
      files: [{
        path: "weights/params.bin",
        bytesReceived: 512,
        bytesTotal: 700,
        resumable: true,
        verification: "hashing",
      }],
      bytesReceived: 512,
      bytesTotal: progressVariant.downloadBytes + progressVariant.requiredCorePackBytes,
      throughputBytesPerSecond: 256,
      etaSeconds: [3, 5],
      calculatingEta: false,
      stalled: false,
      offline: false,
      resumable: true,
    };
    const seen = [];
    const { bundle, calls } = createBundle({
      async requestPersistence() {
        calls.push("persist");
        return true;
      },
      async download(modelVariantId, onProgress) {
        calls.push(`download:${modelVariantId}`);
        onProgress(progress);
        assert.deepEqual(seen.at(-1).download.progress, {
          ...progress,
          bytesTotal: firstRunBytes,
          etaSeconds: [5, 10],
        });
        return { modelVariantId, installed: true, cancelled: false };
      },
    });
    const shell = createShellController(bundle);
    shell.subscribe((snapshot) => seen.push(structuredClone(snapshot)));
    await shell.check();

    await shell.download("smollm2-360m-q4f16");

    assert.ok(
      calls.indexOf("persist") < calls.indexOf("download:smollm2-360m-q4f16"),
    );
    assert.ok(seen.some((snapshot) =>
      snapshot.download.phase === "requesting_persistence"
    ));
    assert.ok(seen.some((snapshot) =>
      snapshot.download.phase === "downloading" &&
      snapshot.download.progress?.bytesReceived === 512
    ));
    assert.equal(shell.current().download.phase, "installed");
  });

  test("first run installs every selected compatible instrument pack with aggregate progress", async () => {
    const document = catalogFixture();
    const variant = document.models[0].variants[0];
    removeFixtureJlens(variant);
    const baseFirstRunBytes = variant.downloadBytes + variant.requiredCorePackBytes;
    const binding = variant.contextProfiles[0].bindingSha256;
    const provenance = {
      license: "AGPL-3.0-or-later",
      sourceRepository: "a9lim/drowse-web-fixture",
      sourceRevision: variant.files[0].revision,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [binding],
    };
    variant.packs.push(
      {
        id: "fixture-jlens",
        kind: "jlens",
        displayName: "J-lens readouts",
        ...provenance,
        bytes: 40,
        files: [{
          ...variant.files[0],
          path: "packs/jlens.bin",
          role: "instrument",
          bytes: 40,
          sha256: "c".repeat(64),
        }],
      },
      {
        id: "fixture-sae",
        kind: "sae",
        displayName: "SAE features",
        ...provenance,
        bytes: 60,
        files: [{
          ...variant.files[0],
          path: "packs/sae.bin",
          role: "instrument",
          bytes: 60,
          sha256: "d".repeat(64),
        }],
      },
    );
    const progress = (path, bytes, packId) => ({
      modelVariantId: variant.id,
      ...(packId ? { packId } : {}),
      files: [{
        path,
        bytesReceived: bytes,
        bytesTotal: bytes,
        resumable: false,
        verification: "verified",
      }],
      bytesReceived: bytes,
      bytesTotal: bytes,
      throughputBytesPerSecond: 100,
      etaSeconds: [0, 0],
      calculatingEta: false,
      stalled: false,
      offline: false,
      resumable: false,
    });
    const fixture = createBundle({
      async download(modelVariantId, onProgress) {
        fixture.calls.push(`download:${modelVariantId}`);
        onProgress(progress("model-and-core", baseFirstRunBytes));
        return { modelVariantId, installed: true, cancelled: false };
      },
      async downloadPack(modelVariantId, packId, onProgress) {
        fixture.calls.push(`download-pack:${modelVariantId}:${packId}`);
        const bytes = packId === "fixture-jlens" ? 40 : 60;
        onProgress(progress(`packs/${packId}.bin`, bytes, packId));
        return { modelVariantId, packId, installed: true, cancelled: false };
      },
    }, document);
    const shell = createShellController(fixture.bundle);
    await shell.check();

    assert.equal(shell.current().models[0].firstRunBytes, baseFirstRunBytes + 100);
    assert.deepEqual(
      shell.current().models[0].firstRunPacks.map((pack) => pack.id),
      ["fixture-jlens", "fixture-sae"],
    );
    assert.deepEqual(
      shell.current().models[0].firstRunPacks.map((pack) => pack.selected),
      [true, true],
    );

    await shell.download(variant.id);

    assert.deepEqual(fixture.calls.slice(-5), [
      "persist",
      "storage",
      `download:${variant.id}`,
      `download-pack:${variant.id}:fixture-jlens`,
      `download-pack:${variant.id}:fixture-sae`,
    ]);
    assert.equal(shell.current().download.phase, "installed");
    assert.equal(shell.current().download.progress.bytesReceived, baseFirstRunBytes + 100);
    assert.equal(shell.current().download.progress.bytesTotal, baseFirstRunBytes + 100);
    assert.deepEqual(
      shell.current().download.progress.files.map((file) => file.path),
      ["model-and-core", "packs/fixture-jlens.bin", "packs/fixture-sae.bin"],
    );
    assert.equal(shell.current().models[0].setupComplete, true);
  });

  test("available SAE features are required and included in exact download bytes", async () => {
    const document = catalogFixture();
    const variant = document.models[0].variants[0];
    removeFixtureJlens(variant);
    const modelBytes = variant.downloadBytes + variant.requiredCorePackBytes;
    variant.packs.push(
      optionalInstrumentPack(variant, "fixture-jlens", "jlens", 40, "c".repeat(64)),
      optionalInstrumentPack(variant, "fixture-sae", "sae", 60, "d".repeat(64)),
    );
    const fixture = createBundle({}, document);
    const shell = createShellController(fixture.bundle);
    await shell.check();

    assert.equal(shell.current().models[0].remainingDownloadBytes, modelBytes + 100);
    assert.throws(() => shell.setOptionalPackSelected(variant.id, "fixture-sae", false), /SAE is required/);
    assert.deepEqual(
      shell.current().models[0].firstRunPacks.map((pack) => [pack.id, pack.selected]),
      [["fixture-jlens", true], ["fixture-sae", true]],
    );
    assert.equal(shell.current().models[0].firstRunBytes, modelBytes + 100);
    assert.equal(shell.current().models[0].remainingDownloadBytes, modelBytes + 100);

    await shell.download(variant.id);

    assert.ok(fixture.calls.includes(`download-pack:${variant.id}:fixture-jlens`));
    assert.equal(
      fixture.calls.includes(`download-pack:${variant.id}:fixture-sae`),
      true,
    );
    assert.equal(shell.current().models[0].setupComplete, true);
    assert.equal(shell.current().models[0].remainingDownloadBytes, 0);
  });

  test("compatible SAE setup is required on iPhone and preview desktop browsers", async () => {
    for (const signals of [{ appleMobile: true }, { runtimeClass: "desktop-webkit" }, { runtimeClass: "desktop-gecko" }]) {
      const document = catalogFixture();
      const variant = document.models[0].variants[0];
      variant.packs.push(optionalInstrumentPack(variant, "fixture-sae", "sae", 60, "d".repeat(64)));
      const shell = createShellController(createBundle({
        async check() { return capabilities({ signals: { ...capabilities().signals, ...signals } }); },
      }, document).bundle);
      await shell.check();
      const model = shell.current().models[0];
      const sae = model.firstRunPacks.find(pack => pack.kind === "sae");
      assert.equal(sae.requiredForSetup, true);
      assert.equal(sae.selected, true);
      assert.doesNotMatch(model.toolNotice ?? "", /separate download/);
      assert.throws(() => shell.setOptionalPackSelected(variant.id, sae.id, false), /SAE is required/);
      shell.dispose();
    }
  });

  test("an installed alternative SAE satisfies the required feature setup", async () => {
    const document = catalogFixture();
    const variant = document.models[0].variants[0];
    variant.packs.push(
      optionalInstrumentPack(variant, "fixture-sae-a", "sae", 40, "c".repeat(64)),
      optionalInstrumentPack(variant, "fixture-sae-b", "sae", 70, "d".repeat(64)),
    );
    const fixture = createBundle({}, document);
    fixture.emitRuntime({ installedPackIds: ["fixture-sae-b"] });
    const shell = createShellController(fixture.bundle);
    await shell.check();
    assert.deepEqual(shell.current().models[0].firstRunPacks.filter(pack => pack.kind === "sae")
      .map(pack => [pack.id, pack.requiredForSetup, pack.selected, pack.installed]), [
        ["fixture-sae-b", true, true, true],
        ["fixture-sae-a", false, false, false],
      ]);
    await shell.download(variant.id);
    assert.ok(!fixture.calls.some(call => call.includes("download-pack:") && call.includes("fixture-sae")));
    assert.equal(shell.current().models[0].setupComplete, true);
    shell.dispose();
  });

  test("a model with no released J-lens is blocked before download", async () => {
    const document = catalogFixture();
    const variant = document.models[0].variants[0];
    removeFixtureJlens(variant);
    const fixture = createBundle({}, document);
    const shell = createShellController(fixture.bundle);
    await shell.check();

    const option = shell.current().models[0];
    assert.equal(option.fit, "blocked");
    assert.match(option.setupIssue, /no compatible precomputed J-lens/u);
    assert.equal(option.toolNotice, undefined);
    assert.deepEqual(option.firstRunPacks, []);

    await assert.rejects(() => shell.download(variant.id));
    assert.equal(
      fixture.calls.some((call) => call === `download:${variant.id}`),
      false,
    );
    assert.equal(shell.current().models[0].setupComplete, false);
  });

  test("choosing another pack in the same family replaces the pending choice", async () => {
    const document = catalogFixture();
    const variant = document.models[0].variants[0];
    variant.packs.push(
      optionalInstrumentPack(variant, "fixture-sae-a", "sae", 40, "c".repeat(64)),
      optionalInstrumentPack(variant, "fixture-sae-b", "sae", 70, "d".repeat(64)),
    );
    const shell = createShellController(createBundle({}, document).bundle);
    await shell.check();

    assert.deepEqual(
      shell.current().models[0].firstRunPacks.map((pack) => [pack.id, pack.selected]),
      [["smollm2-jlens", true], ["fixture-sae-a", true], ["fixture-sae-b", false]],
    );
    shell.setOptionalPackSelected(variant.id, "fixture-sae-b", true);
    assert.equal(shell.current().models[0].firstRunPacks.find(pack => pack.id === "fixture-sae-b").requiredForSetup, true);
    assert.equal(shell.current().models[0].firstRunPacks.find(pack => pack.id === "fixture-sae-a").requiredForSetup, false);
    assert.throws(() => shell.setOptionalPackSelected(variant.id, "fixture-sae-b", false), /SAE is required/);
    assert.deepEqual(
      shell.current().models[0].firstRunPacks.map((pack) => [pack.id, pack.selected]),
      [["smollm2-jlens", true], ["fixture-sae-a", false], ["fixture-sae-b", true]],
    );
    assert.equal(
      shell.current().models[0].firstRunBytes,
      variant.downloadBytes + variant.requiredCorePackBytes + 40 + 70,
    );
  });

  test("standard and R-lens packs can be installed together for live swapping", async () => {
    const document = catalogFixture();
    const variant = document.models[0].variants[0];
    variant.packs.push(
      optionalInstrumentPack(variant, "smollm2-rlens", "jlens", 70, "d".repeat(64)),
    );
    const fixture = createBundle({}, document);
    fixture.emitRuntime({ installedPackIds: ["smollm2-jlens"] });
    const shell = createShellController(fixture.bundle);
    await shell.check();

    assert.deepEqual(
      shell.current().models[0].firstRunPacks
        .filter((pack) => pack.kind === "jlens")
        .map((pack) => [pack.id, pack.selected, pack.installed]),
      [
        ["smollm2-jlens", true, true],
        ["smollm2-rlens", false, false],
      ],
    );
    assert.doesNotThrow(() =>
      shell.setOptionalPackSelected(variant.id, "smollm2-rlens", true)
    );
  });

  test("an installed pack blocks a second pack in the same family", async () => {
    const document = catalogFixture();
    const variant = document.models[0].variants[0];
    variant.packs.push(
      optionalInstrumentPack(variant, "fixture-sae-a", "sae", 40, "c".repeat(64)),
      optionalInstrumentPack(variant, "fixture-sae-b", "sae", 70, "d".repeat(64)),
    );
    const fixture = createBundle({}, document);
    fixture.emitRuntime({ installedPackIds: ["fixture-sae-a"] });
    const shell = createShellController(fixture.bundle);
    await shell.check();

    assert.throws(
      () => shell.setOptionalPackSelected(variant.id, "fixture-sae-b", true),
      /Remove SAE features before choosing another feature pack/u,
    );
    assert.deepEqual(
      shell.current().models[0].firstRunPacks.map((pack) => [pack.id, pack.selected]),
      [["smollm2-jlens", true], ["fixture-sae-a", true], ["fixture-sae-b", false]],
    );
  });

  test("persistence denial warns but does not block a verified installation", async () => {
    const fixture = createBundle({
      async requestPersistence() {
        fixture.calls.push("persist");
        return false;
      },
      async refreshStorage() {
        fixture.calls.push("storage");
        return capabilities({
          storage: { ...capabilities().storage, persisted: false },
        }).storage;
      },
    });
    const shell = createShellController(fixture.bundle);
    await shell.check();

    await shell.download("smollm2-360m-q4f16");

    assert.equal(shell.current().download.phase, "installed");
    assert.equal(shell.current().download.persistenceDenied, true);
    assert.equal(shell.current().storage.persisted, false);
    assert.ok(fixture.calls.includes("download:smollm2-360m-q4f16"));
  });

  test("persistence can be retried after installation without downloading again", async () => {
    let persistenceRequests = 0;
    const fixture = createBundle({
      async requestPersistence() {
        fixture.calls.push("persist");
        persistenceRequests += 1;
        return persistenceRequests > 1;
      },
      async refreshStorage() {
        fixture.calls.push("storage");
        return capabilities({
          storage: {
            ...capabilities().storage,
            persisted: persistenceRequests > 1,
          },
        }).storage;
      },
    });
    const shell = createShellController(fixture.bundle);
    await shell.check();
    await shell.download("smollm2-360m-q4f16");

    assert.equal(shell.current().download.persistenceDenied, true);
    const storageReadsBeforeRetry = fixture.calls.filter((call) => call === "storage").length;
    assert.equal(await shell.retryPersistence(), true);
    assert.equal(shell.current().download.persistenceDenied, false);
    assert.equal(shell.current().storage.persisted, true);
    assert.equal(
      fixture.calls.filter((call) => call === "storage").length,
      storageReadsBeforeRetry,
    );
    assert.equal(
      fixture.calls.filter((call) => call === "download:smollm2-360m-q4f16").length,
      1,
    );
  });

  test("late storage approval clears the notice without repeating the download", async () => {
    let grant;
    const fixture = createBundle({
      async requestPersistence(onLateGranted) { grant = onLateGranted; return false; },
      async refreshStorage() { return { ...capabilities().storage, persisted: false }; },
    });
    const shell = createShellController(fixture.bundle);
    await shell.check();
    await shell.download("smollm2-360m-q4f16");
    assert.equal(shell.current().storage.persisted, false);
    grant();
    assert.equal(shell.current().storage.persisted, true);
    assert.equal(shell.current().download.persistenceDenied, false);
    assert.equal(shell.current().download.phase, "installed");
    assert.equal(fixture.calls.filter(call => call === "download:smollm2-360m-q4f16").length, 1);
    await shell.retryPersistence();
    assert.equal(shell.current().storage.persisted, false);
    grant();
    assert.equal(shell.current().storage.persisted, true);
    shell.dispose();
    const snapshot = shell.current();
    grant();
    assert.equal(shell.current(), snapshot);
  });

  test("a stale quota check cannot overwrite late storage approval", async () => {
    let grant;
    let finishStorage;
    const fixture = createBundle({
      async requestPersistence(onLateGranted) { grant = onLateGranted; return false; },
      refreshStorage() { return new Promise(resolve => { finishStorage = resolve; }); },
    });
    const shell = createShellController(fixture.bundle);
    await shell.check();
    const download = shell.download("smollm2-360m-q4f16");
    await waitFor(() => finishStorage);
    grant();
    finishStorage({ ...capabilities().storage, persisted: false });
    await download;
    assert.equal(shell.current().storage.persisted, true);
    assert.equal(shell.current().download.persistenceDenied, false);
    shell.dispose();
  });

  test("optional pack admission types SAE buffer and concurrent resident-budget failures", () => {
    const document = catalogFixture();
    const variant = document.models[0].variants[0];
    const base = {
      displayName: "fixture instrument",
      bytes: 64,
      required: false,
      compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
      files: [],
    };
    const saeFileBytes = 64 * 1024 * 1024;
    const sae = {
      ...base,
      id: "fixture-sae",
      kind: "sae",
      bytes: saeFileBytes,
      files: [{ bytes: saeFileBytes }],
    };
    const normalWebGpu = capabilities().webGpu;
    const adapterLimitBytes = 16 * 1024 * 1024;
    const bufferBlock = optionalPackHardwareBlock(
      sae,
      variant,
      capabilities({
        webGpu: {
          ...normalWebGpu,
          limits: {
            ...normalWebGpu.limits,
            maxBufferSize: adapterLimitBytes,
            maxStorageBufferBindingSize: adapterLimitBytes,
          },
        },
      }),
    );

    assert.equal(bufferBlock.code, "SAE_DEVICE_BUFFER_LIMIT");
    assert.equal(bufferBlock.requiredBytes, 32 * 1024 * 1024);
    assert.equal(bufferBlock.limitBytes, adapterLimitBytes);
    assert.deepEqual(bufferBlock.packIds, [sae.id]);

    const largeJlens = {
      ...base,
      id: "fixture-jlens-large",
      kind: "jlens",
      bytes: Math.ceil(OPTIONAL_PACK_RESIDENT_BUDGET_BYTES * 0.6),
    };
    const largeSae = {
      ...sae,
      id: "fixture-sae-large",
      bytes: Math.ceil(OPTIONAL_PACK_RESIDENT_BUDGET_BYTES * 0.6),
    };
    const alternateLargeJlens = {
      ...largeJlens,
      id: "fixture-rlens-large",
      bytes: Math.ceil(OPTIONAL_PACK_RESIDENT_BUDGET_BYTES * 0.7),
    };
    const ampleWebGpu = {
      ...normalWebGpu,
      limits: {
        ...normalWebGpu.limits,
        maxBufferSize: 64 * 1024 * 1024,
        maxStorageBufferBindingSize: 64 * 1024 * 1024,
      },
    };
    const lowMemorySignal = capabilities({
      webGpu: ampleWebGpu,
      signals: { ...capabilities().signals, deviceMemoryGiB: null },
    });
    const highMemorySignal = capabilities({
      webGpu: ampleWebGpu,
      signals: { ...capabilities().signals, deviceMemoryGiB: 64 },
    });
    const lowSignalBlock = optionalPackHardwareBlock(
      largeSae,
      variant,
      lowMemorySignal,
      [largeJlens],
    );
    const highSignalBlock = optionalPackHardwareBlock(
      largeSae,
      variant,
      highMemorySignal,
      [largeJlens],
    );

    assert.deepEqual(highSignalBlock, lowSignalBlock);
    assert.equal(lowSignalBlock.code, "OPTIONAL_PACK_RESIDENT_BUDGET_EXCEEDED");
    assert.ok(lowSignalBlock.requiredBytes > lowSignalBlock.limitBytes);
    assert.deepEqual(lowSignalBlock.packIds, [largeJlens.id, largeSae.id]);
    assert.match(lowSignalBlock.message, /Remove one installed tool/u);
    assert.equal(
      optionalPackHardwareBlock(
        alternateLargeJlens,
        variant,
        highMemorySignal,
        [largeJlens],
      ),
      null,
    );

    const qwenVariant = structuredClone(variant);
    qwenVariant.runtimeIdentity.hiddenSize = 2048;
    qwenVariant.runtimeIdentity.layerMap = Array.from({ length: 28 }, (_, index) => index);
    const qwenJlensBytes = 2048 * 2048 * 4 * 28;
    const qwenJlens = {
      ...largeJlens,
      id: "qwen3-1.7b-jlens",
      bytes: qwenJlensBytes,
      files: [{ bytes: qwenJlensBytes }],
    };
    const qwenSae = {
      ...largeSae,
      id: "qwen3-1.7b-sae",
      bytes: OPTIONAL_PACK_RESIDENT_BUDGET_BYTES - qwenJlensBytes + 1,
      files: [{
        bytes: OPTIONAL_PACK_RESIDENT_BUDGET_BYTES - qwenJlensBytes + 1,
      }],
    };
    assert.equal(
      optionalPackHardwareBlock(qwenJlens, qwenVariant, highMemorySignal),
      null,
    );
    assert.equal(
      optionalPackHardwareBlock(qwenSae, qwenVariant, highMemorySignal),
      null,
    );

    const wideVariant = structuredClone(qwenVariant);
    wideVariant.runtimeIdentity.hiddenSize = 3072;
    const wideJlensMatrixBytes = 3072 * 3072 * 4;
    const wideJlens = {
      ...qwenJlens,
      id: "wide-model-jlens",
      bytes: wideJlensMatrixBytes,
      files: [{ bytes: wideJlensMatrixBytes }],
    };
    assert.ok(wideJlensMatrixBytes > 32 * 1024 * 1024);
    const wideJlensBlock = optionalPackHardwareBlock(
      wideJlens,
      wideVariant,
      highMemorySignal,
    );
    assert.equal(wideJlensBlock.code, "JLENS_DEVICE_BUFFER_LIMIT");
    assert.equal(wideJlensBlock.requiredBytes, wideJlensMatrixBytes);
    assert.equal(wideJlensBlock.limitBytes, 32 * 1024 * 1024);
    assert.match(wideJlensBlock.message, /larger graphics buffer/u);

    assert.equal(
      optionalPackHardwareBlock(
        qwenSae,
        qwenVariant,
        highMemorySignal,
        [qwenJlens],
      ).code,
      "OPTIONAL_PACK_RESIDENT_BUDGET_EXCEEDED",
    );

    const partialQwenJlens = {
      ...qwenJlens,
      id: "qwen3-1.7b-partial-jlens",
      bytes: 2 * 2048 * 2048 * 4,
      files: [{ bytes: 2 * 2048 * 2048 * 4 }],
    };
    assert.equal(
      optionalPackHardwareBlock(
        qwenSae,
        qwenVariant,
        highMemorySignal,
        [partialQwenJlens],
      ),
      null,
    );
  });

  test("first run never auto-selects optional packs beyond the resident safety budget", async () => {
    const document = catalogFixture();
    const variant = document.models[0].variants[0];
    removeFixtureJlens(variant);
    const binding = variant.contextProfiles[0].bindingSha256;
    const packBytes = Math.ceil(OPTIONAL_PACK_RESIDENT_BUDGET_BYTES * 0.6);
    const provenance = {
      license: "AGPL-3.0-or-later",
      sourceRepository: "a9lim/drowse-web-fixture",
      sourceRevision: variant.files[0].revision,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [binding],
    };
    variant.packs.push(
      {
        id: "fixture-jlens-resident",
        kind: "jlens",
        displayName: "J-lens resident fixture",
        ...provenance,
        bytes: packBytes,
        files: [{
          ...variant.files[0],
          path: "packs/jlens-resident.bin",
          role: "instrument",
          bytes: packBytes,
          sha256: "e".repeat(64),
        }],
      },
      {
        id: "fixture-sae-resident",
        kind: "sae",
        displayName: "SAE resident fixture",
        ...provenance,
        bytes: packBytes,
        files: [{
          ...variant.files[0],
          path: "packs/sae-resident.bin",
          role: "instrument",
          bytes: packBytes,
          sha256: "f".repeat(64),
        }],
      },
    );
    const normalWebGpu = capabilities().webGpu;
    const shell = createShellController(createBundle({
      async check() {
        return capabilities({
          webGpu: {
            ...normalWebGpu,
            limits: {
              ...normalWebGpu.limits,
              maxBufferSize: 64 * 1024 * 1024,
              maxStorageBufferBindingSize: 64 * 1024 * 1024,
            },
          },
        });
      },
    }, document).bundle);

    await shell.check();

    assert.deepEqual(
      shell.current().models[0].firstRunPacks.map((pack) => [pack.id, pack.selected]),
      [
        ["fixture-jlens-resident", true],
      ],
    );
    assert.equal(
      shell.current().models[0].firstRunBytes,
      variant.downloadBytes + variant.requiredCorePackBytes + packBytes,
    );
  });

  test("a J-lens that exceeds GPU limits is omitted without blocking the model", async () => {
    const document = catalogFixture();
    const variant = document.models[0].variants[0];
    removeFixtureJlens(variant);
    const baseFirstRunBytes = variant.downloadBytes + variant.requiredCorePackBytes;
    const provenance = {
      license: "AGPL-3.0-or-later",
      sourceRepository: "a9lim/drowse-web-fixture",
      sourceRevision: variant.files[0].revision,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
    };
    variant.packs.push(
      {
        id: "fixture-jlens-too-large",
        kind: "jlens",
        displayName: "J-lens readouts",
        ...provenance,
        bytes: 40,
        files: [{
          ...variant.files[0],
          path: "packs/jlens-too-large.bin",
          role: "instrument",
          bytes: 40,
          sha256: "c".repeat(64),
        }],
      },
      {
        id: "fixture-sae-supported",
        kind: "sae",
        displayName: "SAE features",
        ...provenance,
        bytes: 60,
        files: [{
          ...variant.files[0],
          path: "packs/sae-supported.bin",
          role: "instrument",
          bytes: 60,
          sha256: "d".repeat(64),
        }],
      },
    );
    const normalWebGpu = capabilities().webGpu;
    const fixture = createBundle({
      async check() {
        return capabilities({
          webGpu: {
            ...normalWebGpu,
            limits: {
              ...normalWebGpu.limits,
              maxBufferSize: 1_048_576,
              maxStorageBufferBindingSize: 1_048_576,
            },
          },
        });
      },
    }, document);
    const shell = createShellController(fixture.bundle);
    await shell.check();

    assert.equal(shell.current().models[0].firstRunBytes, baseFirstRunBytes + 60);
    assert.deepEqual(
      shell.current().models[0].firstRunPacks.map((pack) => pack.id),
      ["fixture-sae-supported"],
    );
    assert.equal(shell.current().models[0].fit, "blocked");
    assert.match(shell.current().models[0].setupIssue, /cannot safely load the J-lens required/u);
    assert.equal(shell.current().models[0].toolNotice, undefined);
    await assert.rejects(() => shell.download(variant.id));
    assert.equal(fixture.calls.includes(`download:${variant.id}`), false);
  });

  test("shell defers exact first-run quota admission to resumable worker state", async () => {
    const document = catalogFixture();
    const variant = document.models[0].variants[0];
    variant.packs.push({
      id: "fixture-large-sae",
      kind: "sae",
      displayName: "SAE features",
      license: "AGPL-3.0-or-later",
      sourceRepository: "a9lim/drowse-web-fixture",
      sourceRevision: variant.files[0].revision,
      bytes: 100_000_000,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
      files: [{
        ...variant.files[0],
        path: "packs/large-sae.bin",
        role: "instrument",
        bytes: 100_000_000,
        sha256: "e".repeat(64),
      }],
    });
    const normalWebGpu = capabilities().webGpu;
    const fixture = createBundle({
      async check() {
        return capabilities({
          webGpu: {
            ...normalWebGpu,
            limits: {
              ...normalWebGpu.limits,
              maxBufferSize: 64 * 1024 * 1024,
              maxStorageBufferBindingSize: 64 * 1024 * 1024,
            },
          },
        });
      },
      async refreshStorage() {
        fixture.calls.push("storage");
        return {
          quotaBytes: 400_000_000,
          usageBytes: 50_000_000,
          availableBytes: 300_000_000,
          persisted: true,
        };
      },
      async download(modelVariantId) {
        fixture.calls.push(`download:${modelVariantId}`);
        throw new Error("The exact remaining download does not fit after reconciliation");
      },
    }, document);
    const shell = createShellController(fixture.bundle);
    await shell.check();

    await assert.rejects(
      () => shell.download(variant.id),
      /exact remaining download does not fit/i,
    );

    assert.deepEqual(fixture.calls.slice(-3), [
      "persist",
      "storage",
      `download:${variant.id}`,
    ]);
    assert.equal(
      fixture.calls.some((call) => call.startsWith("download-pack:")),
      false,
    );
    assert.equal(shell.current().download.phase, "failed");
  });

  test("unknown shell quota is rechecked authoritatively by the worker", async () => {
    const fixture = createBundle({
      async refreshStorage() {
        fixture.calls.push("storage");
        return {
          quotaBytes: null,
          usageBytes: null,
          availableBytes: null,
          persisted: true,
        };
      },
      async download(modelVariantId) {
        fixture.calls.push(`download:${modelVariantId}`);
        throw new Error("Browser storage availability is unknown");
      },
    });
    const shell = createShellController(fixture.bundle);
    await shell.check();

    await assert.rejects(
      () => shell.download("smollm2-360m-q4f16"),
      /storage availability is unknown/i,
    );

    assert.deepEqual(fixture.calls.slice(-3), [
      "persist",
      "storage",
      "download:smollm2-360m-q4f16",
    ]);
  });

  test("an installed model requires its released J-lens before opening", async () => {
    const document = catalogFixture();
    const variant = document.models[0].variants[0];
    variant.packs.push({
      id: "fixture-sae",
      kind: "sae",
      displayName: "SAE features",
      license: "AGPL-3.0-or-later",
      sourceRepository: "a9lim/drowse-web-fixture",
      sourceRevision: variant.files[0].revision,
      bytes: 60,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
      files: [{
        ...variant.files[0],
        path: "packs/sae.bin",
        role: "instrument",
        bytes: 60,
        sha256: "d".repeat(64),
      }],
    });
    const fixture = createBundle({
      recommend(verified) {
        fixture.calls.push("recommend");
        return [{ ...recommendation(verified.document), installed: true }];
      },
    }, document);
    const shell = createShellController(fixture.bundle);
    await shell.check();

    assert.equal(shell.current().models[0].installed, true);
    assert.equal(shell.current().models[0].setupComplete, false);
    assert.equal(shell.current().models[0].firstRunPacks[0].selected, true);
    assert.equal(shell.current().models[0].firstRunPacks[0].requiredForSetup, true);
    assert.equal(shell.current().models[0].remainingDownloadBytes, 100);
    assert.throws(
      () => shell.setOptionalPackSelected(variant.id, "smollm2-jlens", false),
      /J-lens is required/u,
    );
    assert.equal(shell.current().models[0].setupComplete, false);
    assert.throws(() => shell.setOptionalPackSelected(variant.id, "fixture-sae", false), /SAE is required/);
    assert.equal(shell.current().models[0].setupComplete, false);
    assert.equal(shell.current().models[0].remainingDownloadBytes, 100);
    await shell.download(variant.id);

    assert.equal(
      fixture.calls.some((call) => call === `download:${variant.id}`),
      false,
    );
    assert.ok(fixture.calls.includes(
      `download-pack:${variant.id}:fixture-sae`,
    ));
    assert.ok(fixture.calls.includes(`download-pack:${variant.id}:smollm2-jlens`));
    assert.equal(shell.current().models[0].setupComplete, true);
  });

  test("cancelled downloads settle as paused after the runtime checkpoints them", async () => {
    const pending = deferred();
    let modelVariantId;
    const { bundle, calls } = createBundle({
      download(id) {
        calls.push(`download:${id}`);
        modelVariantId = id;
        return pending.promise;
      },
      async cancelDownload() {
        calls.push("cancel");
        pending.resolve({
          modelVariantId,
          installed: false,
          cancelled: true,
        });
      },
    });
    const shell = createShellController(bundle);
    await shell.check();

    const downloading = shell.download("smollm2-360m-q4f16");
    await waitFor(() => shell.current().download.phase === "downloading");
    await shell.cancelDownload();
    assert.ok(["cancelling", "paused"].includes(shell.current().download.phase));
    await downloading;

    assert.equal(shell.current().download.phase, "paused");
    assert.match(shell.current().download.reason, /downloaded files were kept/i);
    assert.equal(calls.filter((call) => call === "cancel").length, 1);
  });

  test("catalog failures leave a supported device download-locked", async () => {
    const { bundle } = createBundle({
      async catalog() {
        throw Object.assign(new Error("Ed25519 verification returned false"), {
          code: "CATALOG_SIGNATURE_INVALID",
        });
      },
    });
    const shell = createShellController(bundle);

    await shell.check();

    const snapshot = shell.current();
    assert.equal(snapshot.phase, "supported");
    assert.equal(snapshot.download.available, false);
    assert.equal(snapshot.download.phase, "locked");
    assert.match(snapshot.detail, /could not verify the model list/i);
    assert.match(snapshot.download.reason, /could not verify the model list/i);
    assert.doesNotMatch(snapshot.detail, /Ed25519|verification returned false/i);
  });

  test("installed models open with their exact measured context", async () => {
    const { bundle, calls } = createBundle({
      recommend(verified) {
        calls.push("recommend");
        return [{ ...recommendation(verified.document), installed: true }];
      },
    });
    const shell = createShellController(bundle);
    await shell.check();

    await shell.open("smollm2-360m-q4f16");

    assert.equal(shell.current().runtime.phase, "ready");
    assert.ok(calls.includes("load:smollm2-360m-q4f16:2048"));
  });

  test("installed models remain openable from an offline last-known-good catalog", async () => {
    const { bundle, calls } = createBundle({
      async catalog() {
        calls.push("catalog");
        return {
          document: catalogFixture(),
          exactBytes: catalogBytes(),
          signature: FIXTURE_SIGNATURE,
          allowDownloads: false,
          stale: true,
        };
      },
      recommend(verified) {
        calls.push("recommend");
        return [{ ...recommendation(verified.document), installed: true }];
      },
    });
    const shell = createShellController(bundle);
    await shell.check();
    assert.equal(shell.current().download.available, false);

    await shell.open("smollm2-360m-q4f16");

    assert.equal(shell.current().runtime.phase, "ready");
  });

  test("open rejects uninstalled and runtime-disabled models", async () => {
    const ordinary = createBundle();
    const shell = createShellController(ordinary.bundle);
    await shell.check();
    await assert.rejects(
      shell.open("smollm2-360m-q4f16"),
      /Install and verify/,
    );

    const disabled = createBundle({
      async check() {
        return capabilities({
          operations: {
            ...capabilities().operations,
            generation: {
              available: false,
              reasons: [{
                code: "RUNTIME_DISABLED",
                message: "Fixture generation is disabled",
                severity: "hard",
              }],
            },
          },
        });
      },
      recommend(verified) {
        return [{ ...recommendation(verified.document), installed: true }];
      },
    });
    const disabledShell = createShellController(disabled.bundle);
    await disabledShell.check();
    await assert.rejects(
      disabledShell.open("smollm2-360m-q4f16"),
      /local model engine is not available in this browser/i,
    );
  });

  test("load failures stay on onboarding with a retryable error", async () => {
    const { bundle } = createBundle({
      recommend(verified) {
        return [{ ...recommendation(verified.document), installed: true }];
      },
      async load() {
        throw new Error("fixture load failed");
      },
    });
    const shell = createShellController(bundle);
    await shell.check();
    await assert.rejects(shell.open("smollm2-360m-q4f16"), /fixture load failed/);
    assert.equal(shell.current().runtime.phase, "failed");
    assert.equal(shell.current().runtime.reason, "fixture load failed");
  });

  test("PWA reload preparation waits for a clean runtime unload", async () => {
    const gate = deferred();
    const fixture = createBundle({
      async unload() {
        fixture.calls.push("unload:started");
        await gate.promise;
        fixture.calls.push("unload:complete");
      },
    });
    const shell = createShellController(fixture.bundle);
    await shell.check();
    fixture.emitRuntime({
      lifecycle: "ready",
      modelVariantId: "smollm2-360m-q4f16",
    });

    let prepared = false;
    const preparing = shell.prepareForReload().then(() => {
      prepared = true;
    });
    await Promise.resolve();
    assert.equal(prepared, false);
    assert.equal(shell.current().runtime.phase, "ready");

    gate.resolve();
    await preparing;
    assert.equal(prepared, true);
    assert.equal(shell.current().runtime.phase, "unloaded");
    assert.equal(shell.current().runtime.modelVariantId, undefined);
    assert.deepEqual(fixture.calls.slice(-2), ["unload:started", "unload:complete"]);
  });

  test("PWA reload can recover after the runtime worker has already failed", async () => {
    const fixture = createBundle({
      async unload() {
        throw new Error("Runtime worker is closed");
      },
    });
    const shell = createShellController(fixture.bundle);
    await shell.check();
    fixture.emitRuntime({
      lifecycle: "failed",
      modelVariantId: null,
      error: { code: "RUNTIME_WORKER_FAILED", message: "The runtime worker failed" },
    });

    await shell.prepareForReload();

    assert.equal(shell.current().runtime.phase, "unloaded");
    assert.equal(shell.current().runtime.modelVariantId, undefined);
  });

  test("an incompatible saved session requires an explicit destructive retry", async () => {
    const attempts = [];
    const { bundle } = createBundle({
      recommend(verified) {
        return [{ ...recommendation(verified.document), installed: true }];
      },
      async load(_modelVariantId, _contextTokens, options = {}) {
        attempts.push(structuredClone(options));
        if (!options.resetSession) {
          throw Object.assign(new Error("saved session is incompatible"), {
            body: {
              error: { code: "LOCAL_SESSION_INCOMPATIBLE" },
            },
          });
        }
      },
    });
    const shell = createShellController(bundle);
    await shell.check();

    await assert.rejects(
      shell.open("smollm2-360m-q4f16"),
      /saved session is incompatible/,
    );
    assert.equal(shell.current().runtime.resetSessionAvailable, true);

    await shell.open("smollm2-360m-q4f16", { resetSession: true });
    assert.equal(shell.current().runtime.phase, "ready");
    assert.equal(shell.current().runtime.resetSessionAvailable, false);
    assert.deepEqual(attempts, [
      { explicitUnsafeOverride: false, resetSession: false },
      { explicitUnsafeOverride: false, resetSession: true },
    ]);
  });

  test("allowing a busy takeover waits for the runtime to unload", async () => {
    const fixture = createBundle();
    let confirmBusyTakeover;
    const shell = createShellController(undefined, {
      runtimeFactory(options) {
        confirmBusyTakeover = options.confirmBusyTakeover;
        return fixture.bundle;
      },
    });
    const generation = {
      phase: "running",
      startedAt: 1,
      finishedAt: null,
      error: null,
    };

    const decision = confirmBusyTakeover({
      ...fixture.bundle.controller.snapshot,
      lifecycle: "ready",
      generation,
    });
    assert.equal(shell.current().takeover.phase, "requested");
    assert.match(shell.current().takeover.reason, /generating a response/i);

    let released = false;
    const allowing = shell.allowBusyTakeover().then(() => {
      released = true;
    });
    assert.equal(await decision, true);
    assert.equal(shell.current().takeover.phase, "allowing");
    await Promise.resolve();
    assert.equal(released, false);

    fixture.emitRuntime({
      lifecycle: "unloaded",
      modelVariantId: null,
      generation: {
        phase: "complete",
        startedAt: 1,
        finishedAt: 2,
        error: null,
      },
    });
    await allowing;
    assert.equal(released, true);
    assert.equal(shell.current().takeover.phase, "idle");
  });

  test("denying a busy takeover resolves its decision exactly once", async () => {
    const fixture = createBundle();
    let confirmBusyTakeover;
    const shell = createShellController(undefined, {
      runtimeFactory(options) {
        confirmBusyTakeover = options.confirmBusyTakeover;
        return fixture.bundle;
      },
    });
    const decision = confirmBusyTakeover({
      ...fixture.bundle.controller.snapshot,
      lifecycle: "loading",
    });

    assert.equal(shell.current().takeover.phase, "requested");
    assert.match(shell.current().takeover.reason, /loading a model/i);
    shell.denyBusyTakeover();
    shell.denyBusyTakeover();

    assert.equal(await decision, false);
    assert.equal(shell.current().takeover.phase, "idle");
  });

  test("dispose denies an unresolved busy takeover", async () => {
    const fixture = createBundle();
    let confirmBusyTakeover;
    const shell = createShellController(undefined, {
      runtimeFactory(options) {
        confirmBusyTakeover = options.confirmBusyTakeover;
        return fixture.bundle;
      },
    });
    const decision = confirmBusyTakeover({
      ...fixture.bundle.controller.snapshot,
      lifecycle: "checking",
    });

    shell.dispose();

    assert.equal(await decision, false);
    await Promise.resolve();
    assert.equal(fixture.disposed(), 1);
  });

  test("overlapping busy takeover requests reject the later requester", async () => {
    const fixture = createBundle();
    let confirmBusyTakeover;
    const shell = createShellController(undefined, {
      runtimeFactory(options) {
        confirmBusyTakeover = options.confirmBusyTakeover;
        return fixture.bundle;
      },
    });
    const busy = {
      ...fixture.bundle.controller.snapshot,
      lifecycle: "ready",
      fitting: {
        phase: "running",
        startedAt: 1,
        finishedAt: null,
        error: null,
      },
    };

    const firstDecision = confirmBusyTakeover(busy);
    const overlappingDecision = confirmBusyTakeover(busy);

    assert.equal(await overlappingDecision, false);
    assert.equal(shell.current().takeover.phase, "requested");
    assert.match(shell.current().takeover.reason, /fitting a manifold/i);
    shell.denyBusyTakeover();
    assert.equal(await firstDecision, false);
  });

  test("injected runtime bundles never install the shell takeover prompt", async () => {
    const fixture = createBundle();
    let factoryCalls = 0;
    const shell = createShellController(fixture.bundle, {
      runtimeFactory() {
        factoryCalls += 1;
        throw new Error("injected runtimes own their takeover callback");
      },
    });

    fixture.emitRuntime({
      lifecycle: "ready",
      generation: {
        phase: "running",
        startedAt: 1,
        finishedAt: null,
        error: null,
      },
    });
    await shell.allowBusyTakeover();
    shell.denyBusyTakeover();

    assert.equal(factoryCalls, 0);
    assert.equal(shell.current().takeover.phase, "idle");
  });

  test("dispose delegates to the injected runtime bundle", async () => {
    const { bundle, disposed } = createBundle();
    const shell = createShellController(bundle);

    shell.dispose();
    await Promise.resolve();

    assert.equal(disposed(), 1);
  });

  for (const { name, run } of tests) {
    await run();
    console.log(`ok - ${name}`);
  }
  console.log(`${tests.length} hosted shell controller tests passed`);
} finally {
  if (originalNavigator) {
    Object.defineProperty(globalThis, "navigator", originalNavigator);
  } else {
    delete globalThis.navigator;
  }
  await server.close();
}

async function waitFor(predicate, timeoutMs = 1_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = predicate();
    if (value) return value;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error("Timed out waiting for fixture state");
}
