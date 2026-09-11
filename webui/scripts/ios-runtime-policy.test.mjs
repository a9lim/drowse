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

function replaceGlobal(name, value) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, {
    configurable: true,
    writable: true,
    value,
  });
  return () => {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  };
}

function memoryOpfs({ supportsResume = true } = {}) {
  const files = new Map();
  let scopedDirectory = null;
  const scoped = () => ({
    async getFileHandle(name) {
      if (!files.has(name)) files.set(name, "");
      return {
        async createWritable(options = {}) {
          let contents = options.keepExistingData ? files.get(name) ?? "" : "";
          let position = 0;
          return {
            async write(value) {
              const text = String(value);
              contents = contents.slice(0, position) + text +
                contents.slice(position + text.length);
              position += text.length;
            },
            ...(supportsResume
              ? {
                  async seek(next) { position = next; },
                  async truncate(size) {
                    contents = contents.length > size
                      ? contents.slice(0, size)
                      : contents.padEnd(size, "\0");
                  },
                }
              : {}),
            async close() { files.set(name, contents); },
            async abort() {},
          };
        },
        async getFile() { return new Blob([files.get(name) ?? ""]); },
      };
    },
    async *entries() {
      for (const name of files.keys()) yield [name, {}];
    },
  });
  return {
    files,
    directory: {
      async getDirectoryHandle() {
        scopedDirectory = scopedDirectory ?? scoped();
        return scopedDirectory;
      },
      async removeEntry(_name, options = {}) {
        if (!scopedDirectory || options.recursive !== true) {
          throw new Error("Missing OPFS canary directory");
        }
        files.clear();
        scopedDirectory = null;
      },
    },
  };
}

function gpuDevice({ stallWorkDoneAt = null, stallMapAt = null } = {}) {
  const state = {
    submissions: 0,
    workDone: 0,
    maps: 0,
    copies: 0,
  };
  return {
    state,
    queue: {
      submit() { state.submissions += 1; },
      async onSubmittedWorkDone() {
        state.workDone += 1;
        if (state.workDone === stallWorkDoneAt) await new Promise(() => {});
      },
    },
    createBuffer(descriptor) {
      const contents = new ArrayBuffer(Number(descriptor.size));
      return {
        async mapAsync() {
          state.maps += 1;
          if (state.maps === stallMapAt) await new Promise(() => {});
        },
        getMappedRange(offset = 0, size = contents.byteLength - offset) {
          return contents.slice(offset, offset + size);
        },
        unmap() {},
        destroy() {},
        contents,
      };
    },
    createShaderModule() { return {}; },
    createComputePipeline() {
      return { getBindGroupLayout() { return {}; } };
    },
    createBindGroup() { return {}; },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        copyBufferToBuffer() { state.copies += 1; },
        finish() { return {}; },
      };
    },
    destroy() {},
  };
}

function adapter({ fallback = undefined, makeDevice = gpuDevice } = {}) {
  return {
    features: new Set(["shader-f16"]),
    limits: {
      maxBufferSize: 64 * 1024 * 1024,
      maxStorageBufferBindingSize: 64 * 1024 * 1024,
      maxStorageBuffersPerShaderStage: 10,
      maxComputeWorkgroupStorageSize: 65_536,
      maxComputeWorkgroupSizeX: 256,
      maxComputeInvocationsPerWorkgroup: 256,
    },
    ...(fallback === undefined ? {} : { isFallbackAdapter: fallback }),
    async requestDevice() { return makeDevice(); },
  };
}

function operationAvailability() {
  return {
    download: { available: true, reasons: [] },
    generation: { available: true, reasons: [] },
    fitting: { available: true, reasons: [] },
    manifold_artifacts: { available: true, reasons: [] },
    probe_subspace_trails: { available: true, reasons: [] },
    jlens_fitting: { available: false, reasons: [] },
    sae_training: { available: false, reasons: [] },
    session_admin: { available: false, reasons: [] },
    server_endpoints: { available: false, reasons: [] },
  };
}

function runtimeCapabilities(overrides = {}) {
  return {
    checkedAt: 1,
    supported: true,
    secureContext: true,
    crossOriginIsolated: true,
    dedicatedWorker: true,
    indexedDb: true,
    opfs: true,
    webLocks: true,
    broadcastChannel: true,
    webAssembly: true,
    webGpu: {
      available: true,
      fallback: "unknown",
      features: ["shader-f16"],
      limits: {
        maxBufferSize: 64 * 1024 * 1024,
        maxStorageBufferBindingSize: 64 * 1024 * 1024,
      },
      adapterInfo: null,
    },
    storage: {
      quotaBytes: 4_000_000_000,
      usageBytes: 0,
      availableBytes: 4_000_000_000,
      persisted: false,
    },
    signals: {
      deviceMemoryGiB: null,
      logicalCpuCount: 6,
      mobile: true,
      runtimeClass: "apple-mobile-webkit",
      appleMobile: true,
      language: "en-US",
      calibrationScore: 100,
    },
    deviceSignature: "iphone-fixture",
    runtimeVersion: "fixture-runtime",
    issues: [{
      code: "ADAPTER_KIND_UNKNOWN",
      message: "Adapter metadata unavailable; compute passed",
      severity: "advisory",
    }],
    operations: operationAvailability(),
    limits: { manifoldFitMaxIntrinsicDim: 4 },
    ...overrides,
  };
}

function model(id, displayName, tier) {
  const variantId = `${id}-q4`;
  return {
    id,
    displayName,
    description: "fixture",
    sourceUrl: "https://huggingface.co/fixture/model",
    license: "Apache-2.0",
    languages: ["en"],
    variants: [{
      id: variantId,
      tier,
      structuredHookProfile: "standard-v1",
      contextProfiles: [{
        contextTokens: 2048,
        bindingSha256: "b".repeat(64),
        minimumCalibrationScore: null,
        minimumDeviceMemoryGiB: null,
        expectedPrefillTokensPerSecond: [1, 2],
        expectedDecodeTokensPerSecond: [1, 2],
        measuredDevices: 1,
      }],
      downloadBytes: 1_000,
      requiredCorePackBytes: 100,
      requirements: { features: [], limits: {} },
      runtimeIdentity: { hiddenSize: 1024 },
      runtimeIdentitySha256: id.padEnd(64, "0").slice(0, 64),
      files: [],
      packs: [],
    }],
  };
}

function pack(variant, id, kind, bytes) {
  return {
    id,
    kind,
    displayName: kind === "jlens" ? "J-lens readouts" : "Gemma Scope features",
    license: "Apache-2.0",
    sourceRepository: "fixture/packs",
    sourceRevision: "a".repeat(40),
    bytes,
    required: false,
    runtimeIdentitySha256: variant.runtimeIdentitySha256,
    compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
    files: [{
      path: `${id}.bin`,
      role: "instrument",
      url: "https://example.test/file",
      revision: "a".repeat(40),
      bytes,
      sha256: (kind === "jlens" ? "1" : "2").repeat(64),
    }],
  };
}

try {
  const capabilityModule = await server.ssrLoadModule(
    "/src/hosted/runtime/capabilities.ts",
  );
  const recommendationModule = await server.ssrLoadModule(
    "/src/lib/runtime/recommendation.ts",
  );
  const shellModule = await server.ssrLoadModule("/hosted/shell-controller.ts");
  const outputTokenPolicy = await server.ssrLoadModule(
    "/src/lib/runtime/outputTokenPolicy.ts",
  );

  assert.equal(
    capabilityModule.isAppleMobileBrowser(
      "Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X)",
      "iPhone",
      5,
    ),
    true,
  );
  assert.equal(
    capabilityModule.isAppleMobileBrowser(
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Mobile/15E148 Safari/604.1",
      "MacIntel",
      5,
    ),
    true,
  );
  assert.equal(
    capabilityModule.isAppleMobileBrowser(
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Mobile/15E148 Safari/604.1",
      "",
      undefined,
    ),
    true,
  );
  assert.equal(
    capabilityModule.isAppleMobileBrowser(
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Mobile/15E148 Safari/604.1",
      "",
      5,
    ),
    true,
  );
  assert.equal(
    capabilityModule.isAppleMobileBrowser(
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Version/26.0 Safari/605.1.15",
      "MacIntel",
      0,
    ),
    false,
  );
  assert.equal(
    capabilityModule.browserRuntimeClass(
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Version/26.0 Safari/605.1.15",
      "MacIntel",
      0,
    ),
    "desktop-webkit",
  );
  assert.equal(
    capabilityModule.browserRuntimeClass(
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/27.0 Safari/605.1.15",
      "iPad",
      undefined,
    ),
    "apple-mobile-webkit",
  );
  assert.equal(
    capabilityModule.browserRuntimeClass(
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:155.0) Gecko/20100101 Firefox/155.0",
      "MacIntel",
      0,
    ),
    "desktop-gecko",
  );
  assert.equal(
    capabilityModule.browserRuntimeClass(
      "Mozilla/5.0 (Android 16; Mobile; rv:155.0) Gecko/155.0 Firefox/155.0",
      "Linux armv8l",
      5,
    ),
    "other",
  );
  assert.equal(
    capabilityModule.browserRuntimeClass(
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:155.0) Gecko/20100101 Firefox/155.0",
      "Win32",
      0,
    ),
    "desktop-gecko",
  );
  assert.equal(
    capabilityModule.browserRuntimeClass(
      "Mozilla/5.0 (X11; Linux x86_64; rv:155.0) Gecko/20100101 Firefox/155.0",
      "Linux x86_64",
      0,
    ),
    "desktop-gecko",
  );
  assert.equal(
    outputTokenPolicy.outputTokenLimitForRuntime("apple-mobile-webkit", 4096),
    4096,
  );
  assert.equal(
    outputTokenPolicy.outputTokenLimitForRuntime("desktop-webkit", 16384),
    16384,
  );
  assert.equal(
    outputTokenPolicy.outputTokenLimitForRuntime("desktop-gecko", 32768),
    32768,
  );
  assert.equal(
    outputTokenPolicy.outputTokenLimitForRuntime("desktop-chromium"),
    Number.MAX_SAFE_INTEGER,
  );

  const opfs = memoryOpfs();
  const calibratedDevices = [];
  const restoreGlobals = [
    replaceGlobal("isSecureContext", true),
    replaceGlobal("crossOriginIsolated", true),
    replaceGlobal("Worker", class Worker {}),
    replaceGlobal("indexedDB", {}),
    replaceGlobal("BroadcastChannel", class BroadcastChannel {}),
    replaceGlobal("GPUBufferUsage", {
      STORAGE: 1,
      COPY_SRC: 2,
      COPY_DST: 4,
      MAP_READ: 8,
    }),
    replaceGlobal("GPUMapMode", { READ: 1 }),
    replaceGlobal("navigator", {
      userAgent:
        "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Mobile/15E148 Safari/604.1",
      platform: "iPhone",
      maxTouchPoints: 5,
      language: "en-US",
      hardwareConcurrency: 6,
      onLine: true,
      locks: { request() {} },
      storage: {
        async getDirectory() { return opfs.directory; },
        async estimate() { return { quota: 4_000_000_000, usage: 0 }; },
        async persisted() { return false; },
      },
      gpu: {
        async requestAdapter() {
          return adapter({
            makeDevice() {
              const device = gpuDevice();
              calibratedDevices.push(device);
              return device;
            },
          });
        },
      },
    }),
  ];
  try {
    const originalStorage = globalThis.navigator.storage;
    try {
      for (const persisted of [
        () => new Promise(() => {}),
        () => { throw new Error("Protection status unavailable"); },
        async () => { throw new Error("Protection status rejected"); },
      ]) {
        globalThis.navigator.storage = { ...originalStorage, persisted };
        const estimate = await capabilityModule.storageEstimate(20);
        assert.equal(estimate.availableBytes, 4_000_000_000);
        assert.equal(estimate.persisted, null);
      }
      globalThis.navigator.storage = {
        ...originalStorage,
        estimate: () => new Promise(() => {}),
        persisted: async () => true,
      };
      const unavailable = await capabilityModule.storageEstimate(20);
      assert.equal(unavailable.availableBytes, null);
      assert.equal(unavailable.persisted, true);
    } finally {
      globalThis.navigator.storage = originalStorage;
    }
    const checker = new capabilityModule.BrowserCapabilityChecker();
    const checked = await checker.check();
    assert.equal(checked.supported, true);
    assert.equal(checked.opfs, true);
    assert.equal(opfs.files.size, 0);
    assert.equal(checked.signals.appleMobile, true);
    assert.equal(checked.signals.runtimeClass, "apple-mobile-webkit");
    assert.equal(checked.signals.mobile, true);
    assert.ok(checked.issues.some(
      ({ code, severity }) => code === "ADAPTER_KIND_UNKNOWN" && severity === "advisory",
    ));
    assert.equal(calibratedDevices[0].state.submissions, 8);
    assert.equal(calibratedDevices[0].state.workDone, 4);
    assert.equal(calibratedDevices[0].state.maps, 4);
    assert.equal(calibratedDevices[0].state.copies, 4);
    assert.equal(await checker.adapterForLoad() instanceof Object, true);

    const originalNavigatorIdentity = {
      userAgent: globalThis.navigator.userAgent,
      platform: globalThis.navigator.platform,
      maxTouchPoints: globalThis.navigator.maxTouchPoints,
    };
    try {
      Object.assign(globalThis.navigator, {
        userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/27.0 Safari/605.1.15",
        platform: "iPad",
        maxTouchPoints: undefined,
      });
      const ipadWorker = await new capabilityModule.BrowserCapabilityChecker().check();
      assert.equal(ipadWorker.supported, true);
      assert.equal(ipadWorker.signals.appleMobile, true);
      assert.equal(ipadWorker.signals.mobile, true);
      assert.equal(ipadWorker.signals.runtimeClass, "apple-mobile-webkit");
    } finally {
      Object.assign(globalThis.navigator, originalNavigatorIdentity);
    }

    const originalRequestAdapter = globalThis.navigator.gpu.requestAdapter;
    const changedAdapter = adapter();
    changedAdapter.features = new Set(["shader-f16", "timestamp-query"]);
    globalThis.navigator.gpu.requestAdapter = async () => changedAdapter;
    await assert.rejects(
      () => checker.adapterForLoad(),
      (error) => error?.code === "WEBGPU_ADAPTER_CHANGED",
    );
    globalThis.navigator.gpu.requestAdapter = originalRequestAdapter;

    const originalGetDirectory = globalThis.navigator.storage.getDirectory;
    globalThis.navigator.storage.getDirectory = async () => new Promise(() => {});
    const failedStorage = await new capabilityModule.BrowserCapabilityChecker({
      opfsTimeoutMs: 10,
    }).check();
    assert.equal(failedStorage.opfs, false);
    assert.equal(failedStorage.supported, false);
    assert.ok(failedStorage.issues.some(
      ({ code, severity }) => code === "OPFS_READ_WRITE_FAILED" && severity === "hard",
    ));
    globalThis.navigator.storage.getDirectory = originalGetDirectory;

    const incompleteOpfs = memoryOpfs({ supportsResume: false });
    globalThis.navigator.storage.getDirectory = async () => incompleteOpfs.directory;
    const failedResumeStorage = await new capabilityModule.BrowserCapabilityChecker().check();
    assert.equal(failedResumeStorage.opfs, false);
    assert.equal(failedResumeStorage.supported, false);
    assert.equal(incompleteOpfs.files.size, 0);
    assert.ok(failedResumeStorage.issues.some(
      ({ code, severity }) => code === "OPFS_READ_WRITE_FAILED" && severity === "hard",
    ));
    globalThis.navigator.storage.getDirectory = originalGetDirectory;

    globalThis.navigator.gpu.requestAdapter = async () => adapter({ fallback: false });
    const restoreUsage = replaceGlobal("GPUBufferUsage", undefined);
    try {
      const unverifiedChecker = new capabilityModule.BrowserCapabilityChecker();
      const unverified = await unverifiedChecker.check();
      assert.equal(unverified.supported, false);
      assert.ok(unverified.issues.some(
        ({ code, severity }) =>
          code === "WEBGPU_CALIBRATION_UNAVAILABLE" && severity === "hard",
      ));
      await assert.rejects(
        () => unverifiedChecker.adapterForLoad(),
        (error) => error?.code === "COMPATIBILITY_CHECK_REQUIRED",
      );
    } finally {
      restoreUsage();
    }

    const stalledDevice = gpuDevice({ stallWorkDoneAt: 2 });
    globalThis.navigator.gpu.requestAdapter = async () => adapter({
      fallback: false,
      makeDevice: () => stalledDevice,
    });
    const timedOutCalibration = await new capabilityModule.BrowserCapabilityChecker({
      calibrationTimeoutMs: 10,
    }).check();
    assert.equal(timedOutCalibration.supported, false);
    assert.ok(timedOutCalibration.issues.some(
      ({ code, severity }) => code === "WEBGPU_CALIBRATION_TIMEOUT" && severity === "hard",
    ));
    assert.equal(stalledDevice.state.maps, 1);
    assert.equal(stalledDevice.state.submissions, 3);

    const stalledReadbackDevice = gpuDevice({ stallMapAt: 2 });
    globalThis.navigator.gpu.requestAdapter = async () => adapter({
      fallback: false,
      makeDevice: () => stalledReadbackDevice,
    });
    const timedOutReadback = await new capabilityModule.BrowserCapabilityChecker({
      calibrationTimeoutMs: 10,
    }).check();
    assert.equal(timedOutReadback.supported, false);
    assert.ok(timedOutReadback.issues.some(
      ({ code, severity }) => code === "WEBGPU_CALIBRATION_TIMEOUT" && severity === "hard",
    ));
    assert.equal(stalledReadbackDevice.state.maps, 2);
    assert.equal(stalledReadbackDevice.state.submissions, 4);

    const failingDevice = gpuDevice();
    failingDevice.createComputePipeline = () => {
      throw new Error("fixture pipeline failure");
    };
    globalThis.navigator.gpu.requestAdapter = async () => adapter({
      fallback: false,
      makeDevice: () => failingDevice,
    });
    const failedCalibration = await new capabilityModule.BrowserCapabilityChecker().check();
    assert.equal(failedCalibration.supported, false);
    assert.ok(failedCalibration.issues.some(
      ({ code, message, severity }) =>
        code === "WEBGPU_CALIBRATION_FAILED" &&
        severity === "hard" &&
        /fixture pipeline failure/.test(message),
    ));

    const mobileUserAgent = globalThis.navigator.userAgent;
    const mobilePlatform = globalThis.navigator.platform;
    const mobileTouchPoints = globalThis.navigator.maxTouchPoints;
    globalThis.navigator.userAgent =
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Version/26.0 Safari/605.1.15";
    globalThis.navigator.platform = "MacIntel";
    globalThis.navigator.maxTouchPoints = 0;
    globalThis.navigator.gpu.requestAdapter = async () => adapter({ fallback: false });
    const restoreDesktopUsage = replaceGlobal("GPUBufferUsage", undefined);
    try {
      const desktopWithoutCalibration =
        await new capabilityModule.BrowserCapabilityChecker().check();
      assert.equal(desktopWithoutCalibration.supported, false);
      assert.equal(desktopWithoutCalibration.signals.appleMobile, false);
      assert.equal(desktopWithoutCalibration.signals.runtimeClass, "desktop-webkit");
      assert.ok(desktopWithoutCalibration.issues.some(
        ({ code, severity }) =>
          code === "WEBGPU_CALIBRATION_UNAVAILABLE" && severity === "hard",
      ));
    } finally {
      restoreDesktopUsage();
      globalThis.navigator.userAgent = mobileUserAgent;
      globalThis.navigator.platform = mobilePlatform;
      globalThis.navigator.maxTouchPoints = mobileTouchPoints;
    }

    globalThis.navigator.userAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/140.0.0.0";
    globalThis.navigator.gpu.requestAdapter = async () => ({
      ...adapter({ fallback: false }),
      info: { vendor: "intel", architecture: "gen-9" },
    });
    const windowsChecker = new capabilityModule.BrowserCapabilityChecker();
    const windowsIntel = await windowsChecker.check();
    assert.ok(windowsIntel.issues.some(({ code, severity, message }) =>
      code === "WEBGPU_WINDOWS_INTEL_GEN9_BLOCKED" && severity === "hard" && /force-high-performance-gpu/u.test(message)
    ));
    assert.equal(windowsIntel.supported, false);
    assert.equal(windowsIntel.operations.generation.available, false);
    await assert.rejects(windowsChecker.adapterForLoad());
    assert.equal(windowsIntel.webGpu.adapterInfo.architecture, "gen-9");
    globalThis.navigator.gpu.requestAdapter = async () => ({
      ...adapter({ fallback: false }),
      info: { vendor: "nvidia", architecture: "pascal" },
    });
    const switchedGpu = await windowsChecker.check();
    assert.equal(switchedGpu.supported, true);
    assert.notEqual(switchedGpu.deviceSignature, windowsIntel.deviceSignature);
    assert.equal((await windowsChecker.adapterForLoad()).info.vendor, "nvidia");
    globalThis.navigator.userAgent = mobileUserAgent;

    globalThis.navigator.gpu.requestAdapter = async () => adapter({ fallback: true });
    const fallback = await new capabilityModule.BrowserCapabilityChecker().check();
    assert.equal(fallback.supported, false);
    assert.ok(fallback.issues.some(
      ({ code, severity }) => code === "SOFTWARE_ADAPTER" && severity === "hard",
    ));
  } finally {
    for (const restore of restoreGlobals.reverse()) restore();
  }

  const oneB = model("gemma3-1b-instruct", "Gemma 3 1B", "balanced");
  const fourB = model("gemma3-4b-instruct", "Gemma 3 4B", "quality");
  const iosCapabilities = runtimeCapabilities();
  const unproven = recommendationModule.rankCatalogModels([oneB, fourB], {
    capabilities: iosCapabilities,
    contextTokens: 2048,
    preference: "speed",
    language: "en-US",
    loadRecords: [],
  });
  const oneBAssessment = unproven.find(({ model }) => model.id === oneB.id);
  const fourBAssessment = unproven.find(({ model }) => model.id === fourB.id);
  assert.equal(oneBAssessment.eligible, true);
  assert.equal(oneBAssessment.recommended, false);
  assert.ok(oneBAssessment.advisories.some(({ code }) => code === "PROFILE_UNMEASURED"));
  assert.equal(fourBAssessment.eligible, false);
  assert.ok(fourBAssessment.hardFailures.some(
    ({ code }) => code === "APPLE_MOBILE_MODEL_TOO_LARGE",
  ));

  const variant = oneB.variants[0];
  const proven = recommendationModule.rankCatalogModels([oneB], {
    capabilities: iosCapabilities,
    contextTokens: 2048,
    preference: "speed",
    language: "en-US",
    loadRecords: [{
      modelVariantId: variant.id,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      deviceSignature: iosCapabilities.deviceSignature,
      contextTokens: 2048,
      result: "success",
      prefillTokensPerSecond: 2,
      decodeTokensPerSecond: 2,
      recordedAt: 1,
    }],
  });
  assert.equal(proven[0].proven, true);
  assert.equal(proven[0].recommended, true);
  assert.equal(
    proven[0].advisories.some(({ code }) => code === "PROFILE_UNMEASURED"),
    false,
  );

  for (const [runtimeClass, deviceSignature, browserName] of [
    ["desktop-webkit", "safari-mac-fixture", "Safari"],
    ["desktop-gecko", "firefox-mac-fixture", "Firefox"],
  ]) {
    const desktopCapabilities = runtimeCapabilities({
      signals: {
        ...iosCapabilities.signals,
        mobile: false,
        runtimeClass,
        appleMobile: false,
      },
      deviceSignature,
    });
    const desktopUnproven = recommendationModule.rankCatalogModels([oneB], {
      capabilities: desktopCapabilities,
      contextTokens: 2048,
      preference: "balanced",
      language: "en-US",
      loadRecords: [],
    });
    assert.equal(desktopUnproven[0].eligible, true);
    assert.equal(desktopUnproven[0].recommended, false);
    assert.match(desktopUnproven[0].reason, new RegExp(browserName));
    assert.ok(desktopUnproven[0].advisories.some(
      ({ code }) => code === "PROFILE_UNMEASURED",
    ));

    const desktopProven = recommendationModule.rankCatalogModels([oneB], {
      capabilities: desktopCapabilities,
      contextTokens: 2048,
      preference: "balanced",
      language: "en-US",
      loadRecords: [{
        modelVariantId: variant.id,
        runtimeIdentitySha256: variant.runtimeIdentitySha256,
        deviceSignature,
        contextTokens: 2048,
        result: "success",
        prefillTokensPerSecond: 2,
        decodeTokensPerSecond: 2,
        recordedAt: 1,
      }],
    });
    assert.equal(desktopProven[0].proven, true);
    assert.equal(desktopProven[0].recommended, true);
    assert.equal(
      desktopProven[0].advisories.some(({ code }) => code === "PROFILE_UNMEASURED"),
      false,
    );
  }

  const firefox155Model = structuredClone(oneB);
  firefox155Model.variants[0].requirements.limits.maxStorageBuffersPerShaderStage = 10;
  const firefox155Capabilities = runtimeCapabilities({
    webGpu: {
      ...iosCapabilities.webGpu,
      limits: {
        ...iosCapabilities.webGpu.limits,
        maxStorageBuffersPerShaderStage: 9,
      },
    },
    signals: {
      ...iosCapabilities.signals,
      mobile: false,
      runtimeClass: "desktop-gecko",
      appleMobile: false,
    },
    deviceSignature: "firefox-155-mac-fixture",
  });
  const firefox155Assessment = recommendationModule.assessModelVariant(
    firefox155Model,
    firefox155Model.variants[0],
    {
      capabilities: firefox155Capabilities,
      contextTokens: 2048,
      loadRecords: [],
    },
  );
  assert.equal(firefox155Assessment.eligible, false);
  assert.ok(firefox155Assessment.hardFailures.some(
    ({ code, message }) =>
      code === "WEBGPU_LIMIT_TOO_LOW" &&
      message.includes("maxStorageBuffersPerShaderStage >= 10") &&
      message.includes("reports 9"),
  ));

  variant.packs = [
    pack(variant, "gemma-jlens", "jlens", 4 * 1024 * 1024),
    pack(variant, "gemma-sae", "sae", 8 * 1024 * 1024),
  ];
  const catalog = {
    document: {
      schemaVersion: 1,
      sequence: 1,
      issuedAt: new Date().toISOString(),
      expiresAt: new Date(Date.now() + 60_000).toISOString(),
      runtimeAbi: "fixture-runtime",
      models: [model("gemma3-270m-instruct", "Gemma 3 270M", "fastest"), oneB],
    },
    exactBytes: new Uint8Array(),
    signature: { schemaVersion: 1, algorithm: "Ed25519", keyId: "fixture", signature: "" },
    allowDownloads: true,
    stale: false,
  };
  const runtimeSnapshot = {
    lifecycle: "unloaded",
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
  let recommendedCatalog;
  let recommendedPreference;
  const fakeController = {
    snapshot: runtimeSnapshot,
    subscribe(listener) { listener(runtimeSnapshot); return () => {}; },
    async check() { return iosCapabilities; },
    async catalog() { return catalog; },
    recommend(nextCatalog, preference) {
      recommendedCatalog = nextCatalog;
      recommendedPreference = preference;
      return [{
        model: oneB,
        variant,
        context: variant.contextProfiles[0],
        installed: false,
        eligible: true,
        proven: false,
        hardFailures: [],
        advisories: [{ code: "PROFILE_UNMEASURED", message: "Needs first load", severity: "advisory" }],
        requiredStorageBytes: 1_100,
        recommended: false,
        reason: "Needs first load",
      }];
    },
  };
  const shell = shellModule.createShellController({
    controller: fakeController,
    runtime: {},
  });
  await shell.check();
  assert.deepEqual(
    recommendedCatalog.document.models.map(({ id }) => id),
    ["gemma3-270m-instruct", "gemma3-1b-instruct"],
  );
  assert.equal(recommendedPreference, "speed");
  const shellModel = shell.current().models[0];
  assert.equal(shellModel.firstRunPacks.find(({ kind }) => kind === "jlens").selected, true);
  assert.equal(shellModel.firstRunPacks.find(({ kind }) => kind === "sae").selected, false);
  assert.match(shellModel.toolNotice, /Live word readouts start off/i);
  assert.match(shellModel.toolNotice, /separate download/i);

  console.log("ok - Safari admission uses capabilities and a real OPFS canary");
  console.log("ok - Apple mobile calibration repeats compute submissions and mapped readbacks");
  console.log("ok - Apple mobile model policy requires evidence and blocks 4B models");
  console.log("ok - Apple mobile setup retains 270M, prefers speed, and defers SAE packs");
  console.log("ok - Safari and Firefox previews require exact-browser load evidence");
  console.log("ok - output limits follow context capacity across browsers");
  console.log("ok - Firefox 155 remains blocked when its adapter reports 9 of 10 buffers");
} finally {
  await server.close();
}
