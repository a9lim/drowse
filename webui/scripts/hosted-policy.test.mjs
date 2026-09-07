import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const runtimeLock = JSON.parse(
  await readFile(new URL("../../browser-runtime/runtime-lock.json", import.meta.url), "utf8"),
);
const distributionLock = JSON.parse(
  await readFile(new URL("../../browser-runtime/distribution-lock.json", import.meta.url), "utf8"),
);
runtimeLock.status = "verified";
distributionLock.status = "verified";
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  plugins: [{
    name: "verified-release-fixtures",
    enforce: "pre",
    transform(source, id) {
      if (id.endsWith("/runtimeLock.ts") || id.endsWith("/distributionConfig.ts")) {
        return source.replace(/\b(?:runtimeLock|distributionLock)\.status\b/gu, '"verified"');
      }
    },
  }],
  server: { middlewareMode: true, watch: null },
});

function capabilities(overrides = {}) {
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
      fallback: "hardware",
      features: [],
      limits: {},
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
    runtimeVersion: runtimeLock.runtimeAbi,
    issues: [],
    operations: {
      download: { available: true, reasons: [] },
      generation: { available: true, reasons: [] },
      fitting: { available: true, reasons: [] },
      manifold_artifacts: { available: false, reasons: [] },
      probe_subspace_trails: { available: true, reasons: [] },
      jlens_fitting: { available: false, reasons: [] },
      sae_training: { available: false, reasons: [] },
      session_admin: { available: false, reasons: [] },
      server_endpoints: { available: false, reasons: [] },
    },
    ...overrides,
  };
}

function model(id, language = "en") {
  const runtimeIdentitySha256 = id.padEnd(64, id.at(-1) ?? "0").slice(0, 64);
  return {
    id: `model-${id}`,
    displayName: id,
    description: "policy fixture",
    sourceUrl: "https://huggingface.co/fixture/model",
    license: "Apache-2.0",
    languages: [language],
    variants: [{
      id: `variant-${id}`,
      tier: "balanced",
      structuredHookProfile: "standard-v1",
      contextProfiles: [{
        contextTokens: 2048,
        bindingSha256: "b".repeat(64),
        minimumCalibrationScore: 1,
        minimumDeviceMemoryGiB: 1,
        expectedPrefillTokensPerSecond: [1, 2],
        expectedDecodeTokensPerSecond: [1, 2],
        measuredDevices: 1,
      }],
      downloadBytes: 1_000,
      requiredCorePackBytes: 100,
      requirements: { features: [], limits: {} },
      runtimeIdentity: {},
      runtimeIdentitySha256,
      files: [],
      packs: [],
    }],
  };
}

function successfulLoad(modelFixture, decodeTokensPerSecond) {
  const variant = modelFixture.variants[0];
  return {
    modelVariantId: variant.id,
    runtimeIdentitySha256: variant.runtimeIdentitySha256,
    deviceSignature: "fixture-device",
    contextTokens: 2048,
    result: "success",
    prefillTokensPerSecond: 10,
    decodeTokensPerSecond,
    recordedAt: 1,
  };
}

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

function memoryOpfs() {
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
            async seek(next) { position = next; },
            async truncate(size) {
              contents = contents.length > size
                ? contents.slice(0, size)
                : contents.padEnd(size, "\0");
            },
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

class FakeBroadcastChannel {
  static channels = new Map();

  constructor(name) {
    this.name = name;
    this.listeners = new Set();
    const channels = FakeBroadcastChannel.channels.get(name) ?? new Set();
    channels.add(this);
    FakeBroadcastChannel.channels.set(name, channels);
  }

  addEventListener(type, listener) {
    if (type === "message") this.listeners.add(listener);
  }

  postMessage(data) {
    for (const channel of FakeBroadcastChannel.channels.get(this.name) ?? []) {
      if (channel === this) continue;
      queueMicrotask(() => {
        for (const listener of channel.listeners) listener({ data });
      });
    }
  }

  close() {
    FakeBroadcastChannel.channels.get(this.name)?.delete(this);
  }
}

class FakeLockManager {
  active = false;
  waiting = [];
  requestCount = 0;

  request(name, options, callback) {
    this.requestCount += 1;
    if (options.ifAvailable && options.signal) {
      throw new TypeError("signal and ifAvailable cannot be combined");
    }
    return new Promise((resolve, reject) => {
      const entry = { name, options, callback, resolve, reject };
      if (options.ifAvailable && this.active) {
        Promise.resolve(callback(null)).then(resolve, reject);
      } else if (this.active) {
        this.waiting.push(entry);
        options.signal?.addEventListener("abort", () => {
          const index = this.waiting.indexOf(entry);
          if (index >= 0) this.waiting.splice(index, 1);
          reject(new DOMException("Lock request aborted", "AbortError"));
        }, { once: true });
      } else {
        this.grant(entry);
      }
    });
  }

  grant(entry) {
    this.active = true;
    Promise.resolve(entry.callback({ name: entry.name })).then(
      (value) => {
        this.active = false;
        entry.resolve(value);
        const next = this.waiting.shift();
        if (next) this.grant(next);
      },
      (error) => {
        this.active = false;
        entry.reject(error);
        const next = this.waiting.shift();
        if (next) this.grant(next);
      },
    );
  }
}

try {
  const recommendation = await server.ssrLoadModule(
    "/src/lib/runtime/recommendation.ts",
  );
  const capabilityModule = await server.ssrLoadModule(
    "/src/hosted/runtime/capabilities.ts",
  );
  const authoringModule = await server.ssrLoadModule(
    "/src/hosted/runtime/browserAuthoring.ts",
  );
  const catalogModule = await server.ssrLoadModule(
    "/src/lib/runtime/catalog.ts",
  );
  const ownershipModule = await server.ssrLoadModule(
    "/src/hosted/runtime/ownership.ts",
  );
  const randomIdModule = await server.ssrLoadModule(
    "/src/hosted/runtime/randomId.ts",
  );
  const runtimeLockModule = await server.ssrLoadModule(
    "/src/hosted/runtime/runtimeLock.ts",
  );
  const uiCapabilityModule = await server.ssrLoadModule(
    "/src/lib/runtime/ui-capabilities.ts",
  );

  assert.equal(
    capabilityModule.generationReleaseIssue(false, false).code,
    "CUSTOM_MLC_BACKEND_UNAVAILABLE",
  );
  assert.equal(
    capabilityModule.generationReleaseIssue(true, false).code,
    "CUSTOM_MLC_BACKEND_UNINTEGRATED",
  );
  assert.equal(capabilityModule.generationReleaseIssue(true, true), null);
  assert.equal(authoringModule.BROWSER_AUTHORING_IMPLEMENTATION_INTEGRATED, true);
  assert.equal(authoringModule.BROWSER_AUTHORING_RUNTIME_INTEGRATED, true);
  assert.equal(
    authoringModule.BROWSER_AUTHORING_RELEASE_VERIFIED,
    true,
  );
  const authoringReady = {
    runtimeIntegrated: true,
    modelBackendIntegrated: true,
  };
  assert.equal(capabilityModule.browserAuthoringAvailableForChannel({
    ...authoringReady,
  }), true);
  assert.equal(capabilityModule.browserAuthoringAvailableForChannel({
    ...authoringReady,
  }), true);
  assert.equal(capabilityModule.browserAuthoringAvailableForChannel({
    ...authoringReady,
  }), true);
  assert.equal(capabilityModule.browserAuthoringAvailableForChannel({
    ...authoringReady,
    modelBackendIntegrated: false,
  }), false);
  assert.equal(
    capabilityModule.structuredHookProfilesSupportSubspaceTrails(["standard-v1"]),
    false,
  );
  assert.equal(
    capabilityModule.structuredHookProfilesSupportSubspaceTrails(["standard-v2", "standard-v3"]),
    true,
  );
  assert.equal(
    capabilityModule.structuredHookProfilesSupportSubspaceTrails(["unknown-profile"]),
    false,
  );
  assert.equal(
    capabilityModule.structuredHookProfilesSupportSubspaceTrails(
      runtimeLock.models.map((entry) => entry.structuredHookProfile),
    ),
    true,
  );
  console.log("ok - generation readiness reports the exact missing runtime requirement");

  const installed = model("installed");
  const installedAssessment = recommendation.assessModelVariant(
    installed,
    installed.variants[0],
    {
      capabilities: capabilities({
        storage: {
          quotaBytes: null,
          usageBytes: null,
          availableBytes: null,
          persisted: null,
        },
      }),
      contextTokens: 2048,
      installedModelVariantIds: [installed.variants[0].id],
    },
  );
  assert.equal(installedAssessment.eligible, true);
  assert.equal(installedAssessment.requiredStorageBytes, 0);
  assert.equal(
    installedAssessment.hardFailures.some(({ code }) => code === "STORAGE_UNKNOWN"),
    false,
  );
  console.log("ok - installed models remain eligible when storage is unknown");

  const languageMatch = model("language", "en");
  const faster = model("faster", "fr");
  const ranked = recommendation.rankCatalogModels(
    [languageMatch, faster],
    {
      capabilities: capabilities(),
      contextTokens: 2048,
      preference: "balanced",
      language: "en-US",
      loadRecords: [
        successfulLoad(languageMatch, 8),
        successfulLoad(faster, 16),
      ],
    },
  );
  assert.equal(ranked[0].variant.id, faster.variants[0].id);
  console.log("ok - local decode speed ranks before browser language");

  assert.equal(runtimeLockModule.BROWSER_RUNTIME_LOCK.runtimeAbi, runtimeLock.runtimeAbi);

  const restoreCryptoWithoutRandomUuid = replaceGlobal("crypto", {
    getRandomValues(bytes) {
      bytes.set(Array.from({ length: bytes.length }, (_, index) => index));
      return bytes;
    },
  });
  try {
    assert.equal(
      randomIdModule.randomUuid(),
      "00010203-0405-4607-8809-0a0b0c0d0e0f",
    );
  } finally {
    restoreCryptoWithoutRandomUuid();
  }
  console.log("ok - browser identifiers retain UUID v4 entropy fallback without randomUUID");

  const opfs = memoryOpfs();
  const restore = [
    replaceGlobal("isSecureContext", true),
    replaceGlobal("crossOriginIsolated", true),
    replaceGlobal("Worker", class Worker {}),
    replaceGlobal("indexedDB", {}),
    replaceGlobal("BroadcastChannel", FakeBroadcastChannel),
    replaceGlobal("GPUBufferUsage", undefined),
    replaceGlobal("navigator", {
      userAgent: "Mozilla/5.0 Chrome/140.0.0.0 Safari/537.36",
      language: "en-US",
      hardwareConcurrency: 8,
      deviceMemory: 8,
      locks: { request() {} },
      storage: {
        async getDirectory() { return opfs.directory; },
        async estimate() { return { quota: 1_000_000, usage: 0 }; },
        async persisted() { return true; },
      },
      gpu: {
        async requestAdapter() {
          return {
            features: [],
            limits: Object.fromEntries(
              catalogModule.CATALOG_WEBGPU_LIMIT_NAMES.map((name, index) => [name, index + 1]),
            ),
            async requestDevice() {
              return { destroy() {} };
            },
          };
        },
      },
    }),
  ];
  try {
    const result = await new capabilityModule.BrowserCapabilityChecker().check();
    assert.equal(opfs.files.size, 0);
    assert.equal(result.runtimeVersion, runtimeLock.runtimeAbi);
    assert.equal(result.limits.manifoldFitMaxIntrinsicDim, 4);
    assert.equal(result.crossOriginIsolated, true);
    assert.equal(result.broadcastChannel, true);
    assert.equal(result.webGpu.fallback, "unknown");
    assert.deepEqual(
      Object.keys(result.webGpu.limits).sort(),
      [...catalogModule.CATALOG_WEBGPU_LIMIT_NAMES].sort(),
    );
    assert.equal(result.supported, false);
    assert.equal(result.operations.download.available, false);
    assert.deepEqual(result.operations.probe_subspace_trails, result.operations.generation);
    assert.equal(result.operations.manifold_artifacts.available, false);
    assert.ok(result.operations.manifold_artifacts.reasons.some(
      ({ code }) => code === "ADAPTER_KIND_UNKNOWN",
    ));
    assert.equal(
      result.operations.download.reasons.some(
        ({ code }) => code === "SIGNED_DISTRIBUTION_UNAVAILABLE",
      ),
      distributionLock.status !== "verified" || runtimeLock.status !== "verified",
    );
    assert.ok(
      result.issues.some(
        ({ code, severity }) =>
          code === "ADAPTER_KIND_UNKNOWN" && severity === "hard",
      ),
    );
    const makeHardwareAdapter = (vendor) => {
      const state = { deviceRequests: 0 };
      return {
        state,
        adapter: {
          features: [],
          info: { vendor, architecture: "fixture", isFallbackAdapter: false },
          limits: Object.fromEntries(
            catalogModule.CATALOG_WEBGPU_LIMIT_NAMES.map((name, index) => [name, index + 1]),
          ),
          async requestDevice() {
            state.deviceRequests += 1;
            return { destroy() {} };
          },
        },
      };
    };
    const checkedAdapter = makeHardwareAdapter("fixture");
    const loadAdapter = makeHardwareAdapter("fixture");
    let adapterRequests = 0;
    globalThis.navigator.gpu.requestAdapter = async () => {
      adapterRequests += 1;
      return adapterRequests === 1 ? checkedAdapter.adapter : loadAdapter.adapter;
    };
    const freshChecker = new capabilityModule.BrowserCapabilityChecker();
    const freshCapabilities = await freshChecker.check();
    assert.equal(freshCapabilities.supported, true, JSON.stringify(freshCapabilities.issues));
    assert.equal(freshCapabilities.operations.fitting.available, true);
    assert.deepEqual(freshCapabilities.operations.jlens_fitting.available, false);
    assert.deepEqual(freshCapabilities.operations.sae_training.available, false);
    assert.deepEqual(freshCapabilities.webGpu.adapterInfo, {
      vendor: "fixture",
      architecture: "fixture",
    });
    assert.equal(await freshChecker.adapterForLoad(), loadAdapter.adapter);
    assert.equal(adapterRequests, 2);
    assert.equal(checkedAdapter.state.deviceRequests, 1);
    assert.equal(loadAdapter.state.deviceRequests, 0);

    const changedChecked = makeHardwareAdapter("fixture-a");
    const changedLoad = makeHardwareAdapter("fixture-b");
    let changedRequests = 0;
    globalThis.navigator.gpu.requestAdapter = async () =>
      ++changedRequests === 1 ? changedChecked.adapter : changedLoad.adapter;
    const changedChecker = new capabilityModule.BrowserCapabilityChecker();
    assert.equal((await changedChecker.check()).supported, true);
    await assert.rejects(
      changedChecker.adapterForLoad(),
      (error) => error.code === "WEBGPU_ADAPTER_CHANGED",
    );

    const restoreSecureContext = replaceGlobal("isSecureContext", false);
    try {
      const insecure = await new capabilityModule.BrowserCapabilityChecker().check();
      assert.equal(insecure.secureContext, false);
      assert.equal(insecure.supported, false);
      assert.ok(
        insecure.issues.some(
          ({ code, severity }) => code === "INSECURE_CONTEXT" && severity === "hard",
        ),
      );
    } finally {
      restoreSecureContext();
    }
    const restoreBroadcastChannel = replaceGlobal("BroadcastChannel", undefined);
    try {
      const missingBroadcast = await new capabilityModule.BrowserCapabilityChecker().check();
      assert.equal(missingBroadcast.broadcastChannel, false);
      assert.equal(missingBroadcast.supported, false);
      assert.ok(
        missingBroadcast.issues.some(
          ({ code, severity }) =>
            code === "BROADCAST_CHANNEL_UNAVAILABLE" && severity === "hard",
        ),
      );
    } finally {
      restoreBroadcastChannel();
    }
    const restoreIsolation = replaceGlobal("crossOriginIsolated", false);
    try {
      const missingIsolation = await new capabilityModule.BrowserCapabilityChecker().check();
      assert.equal(missingIsolation.crossOriginIsolated, false);
      assert.equal(missingIsolation.supported, false);
      assert.ok(
        missingIsolation.issues.some(
          ({ code, severity }) =>
            code === "CROSS_ORIGIN_ISOLATION_REQUIRED" && severity === "hard",
        ),
      );
    } finally {
      restoreIsolation();
    }
    globalThis.navigator.gpu.requestAdapter = () => new Promise(() => {});
    const timedOut = await new capabilityModule.BrowserCapabilityChecker({
      adapterTimeoutMs: 1,
    }).check();
    assert.ok(
      timedOut.issues.some(
        ({ code, severity }) =>
          code === "WEBGPU_ADAPTER_TIMEOUT" && severity === "hard",
      ),
    );
  } finally {
    for (const restoreGlobal of restore.reverse()) restoreGlobal();
  }
console.log("ok - capability policy requires a compute check for unidentified adapters");

  const artifactBlocked = capabilities();
  artifactBlocked.operations.manifold_artifacts = {
    available: false,
    reasons: [{
      code: "MANIFOLD_ARTIFACT_BRIDGE_UNAVAILABLE",
      message: "Fixture artifact bridge unavailable",
      severity: "hard",
    }],
  };
  assert.deepEqual(
    uiCapabilityModule.drawerAvailability("manifold_pack", artifactBlocked),
    {
      available: false,
      reason: "This model’s response-control files are incomplete or incompatible. Remove the model, download it again, and retry.",
    },
  );
  assert.deepEqual(
    uiCapabilityModule.drawerAvailability("template_lab", artifactBlocked),
    {
      available: false,
      reason: "This model’s response-control files are incomplete or incompatible. Remove the model, download it again, and retry.",
    },
  );
  const fittingBlocked = capabilities();
  fittingBlocked.operations.fitting = {
    available: false,
    reasons: [{
      code: "WASM_FITTING_UNAVAILABLE",
      message: "Fixture fitting unavailable",
      severity: "hard",
    }],
  };
  fittingBlocked.operations.manifold_artifacts = { available: true, reasons: [] };
  assert.deepEqual(
    uiCapabilityModule.drawerAvailability("manifold_pack", fittingBlocked),
    { available: true, reason: null },
  );
  assert.deepEqual(
    uiCapabilityModule.drawerAvailability("manifold_merge", fittingBlocked),
    { available: true, reason: null },
  );
  assert.deepEqual(
    uiCapabilityModule.drawerAvailability("template_lab", fittingBlocked),
    {
      available: false,
      reason: "That building method is not available for this model in the browser.",
    },
  );
  console.log("ok - hosted authoring drawers are filtered before unsupported requests");

  const lockManager = new FakeLockManager();
  const restoreOwnershipGlobals = [
    replaceGlobal("BroadcastChannel", FakeBroadcastChannel),
    replaceGlobal("navigator", { locks: lockManager }),
  ];
  try {
    const first = new ownershipModule.GpuRuntimeOwnership({
      channelName: "ownership-release-test",
    });
    assert.equal(await first.acquire(), true);
    assert.equal(first.isOwner, true);
    first.release();
    assert.equal(first.isOwner, false);
    while (lockManager.active) await Promise.resolve();
    first.close();
    console.log("ok - ownership release updates local state synchronously");

    const concurrent = new ownershipModule.GpuRuntimeOwnership({
      channelName: "ownership-concurrent-test",
    });
    const requestsBeforeConcurrentAcquire = lockManager.requestCount;
    assert.deepEqual(
      await Promise.all([concurrent.acquire(), concurrent.acquire()]),
      [true, true],
    );
    assert.equal(lockManager.requestCount, requestsBeforeConcurrentAcquire + 1);
    assert.equal(concurrent.isOwner, true);
    concurrent.close();
    console.log("ok - concurrent ownership acquisition shares one lock request");

    const owner = new ownershipModule.GpuRuntimeOwnership({
      channelName: "ownership-takeover-test",
      onTakeoverRequest: () => true,
    });
    const requester = new ownershipModule.GpuRuntimeOwnership({
      channelName: "ownership-takeover-test",
      takeoverTimeoutMs: 100,
    });
    assert.equal(await owner.acquire(), true);
    assert.equal(await requester.acquire(), true);
    assert.equal(owner.isOwner, false);
    assert.equal(requester.isOwner, true);
    owner.close();
    requester.close();
    console.log("ok - cooperative takeover transfers the exclusive runtime lock");

    const blocker = new ownershipModule.GpuRuntimeOwnership({
      channelName: "ownership-close-blocker",
      onTakeoverRequest: () => false,
    });
    const closingRequester = new ownershipModule.GpuRuntimeOwnership({
      channelName: "ownership-close-requester",
      takeoverTimeoutMs: 1_000,
    });
    assert.equal(await blocker.acquire(), true);
    const pendingAcquire = closingRequester.acquire();
    while (lockManager.waiting.length === 0) await Promise.resolve();
    closingRequester.close();
    assert.equal(await pendingAcquire, false);
    assert.equal(closingRequester.isOwner, false);
    await assert.rejects(closingRequester.acquire(), /coordinator is closed/);
    blocker.close();
    console.log("ok - closing ownership aborts a pending lock acquisition");

    let releaseShared;
    let markSharedStarted;
    const sharedStarted = new Promise((resolve) => {
      markSharedStarted = resolve;
    });
    const shared = navigator.locks.request(
      ownershipModule.HOSTED_DESTRUCTIVE_LOCK,
      { mode: "shared" },
      async () => {
        markSharedStarted();
        await new Promise((resolve) => {
          releaseShared = resolve;
        });
      },
    );
    await sharedStarted;
    let destructiveStarted = false;
    const destructive = ownershipModule.withHostedDestructiveLock(async () => {
      destructiveStarted = true;
    });
    await Promise.resolve();
    assert.equal(destructiveStarted, false);
    releaseShared();
    await shared;
    await destructive;
    assert.equal(destructiveStarted, true);
    console.log("ok - destructive runtime changes wait for shared activity");

    let releaseExclusive;
    let markExclusiveStarted;
    const exclusiveStarted = new Promise((resolve) => {
      markExclusiveStarted = resolve;
    });
    const exclusive = ownershipModule.withHostedDestructiveLock(async () => {
      markExclusiveStarted();
      await new Promise((resolve) => {
        releaseExclusive = resolve;
      });
    });
    await exclusiveStarted;
    let artifactStarted = false;
    const artifact = ownershipModule.withHostedDestructiveSharedLock(async () => {
      artifactStarted = true;
    });
    await Promise.resolve();
    assert.equal(artifactStarted, false);
    releaseExclusive();
    await Promise.all([exclusive, artifact]);
    assert.equal(artifactStarted, true);
    console.log("ok - artifact operations wait for destructive clear ownership");
  } finally {
    for (const restoreGlobal of restoreOwnershipGlobals.reverse()) restoreGlobal();
    FakeBroadcastChannel.channels.clear();
  }
  console.log("10 hosted policy tests passed");
} finally {
  await server.close();
}
