import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
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

class FakeWorkerScope {
  messages = [];
  timeline = [];
  listener = null;

  postMessage(message) {
    this.messages.push(message);
    this.timeline.push(
      message.kind === "event"
        ? `event:${message.event}`
        : `response:${message.requestId}:${message.ok ? "ok" : "error"}`,
    );
  }

  addEventListener(type, listener) {
    assert.equal(type, "message");
    this.listener = listener;
  }

  send(request) {
    assert.ok(this.listener);
    this.listener({ data: request });
  }
}

class FakeTransportWorker {
  listeners = new Map([
    ["message", new Set()],
    ["error", new Set()],
    ["messageerror", new Set()],
  ]);
  posted = [];
  terminated = false;

  postMessage(message) {
    this.posted.push(message);
  }

  addEventListener(type, listener) {
    this.listeners.get(type).add(listener);
  }

  removeEventListener(type, listener) {
    this.listeners.get(type).delete(listener);
  }

  terminate() { this.terminated = true; }

  emit(data) {
    for (const listener of this.listeners.get("message")) listener({ data });
  }
}

try {
  const contracts = await server.ssrLoadModule("/src/lib/runtime/contracts.ts");
  const { ApiError } = await server.ssrLoadModule("/src/lib/runtime/errors.ts");
  const { isFittingCancellation } = await server.ssrLoadModule(
    "/src/lib/runtime/fittingCancellation.ts",
  );
  const catalogModule = await server.ssrLoadModule("/src/lib/runtime/catalog.ts");
  const workerModule = await server.ssrLoadModule("/src/hosted/runtime/worker.ts");
  const downloaderModule = await server.ssrLoadModule(
    "/src/hosted/runtime/downloader.ts",
  );
  const browserModule = await server.ssrLoadModule(
    "/src/hosted/runtime/browserRuntimeClient.ts",
  );
  const artifactModule = await server.ssrLoadModule(
    "/src/hosted/artifacts/service.ts",
  );
  const fakeRuntimeModule = await server.ssrLoadModule(
    "/src/hosted/runtime/fakeRuntime.ts",
  );
  const fakeModelBackendModule = await server.ssrLoadModule(
    "/src/hosted/runtime/fakeModelBackend.ts",
  );
  const operationPolicyModule = await server.ssrLoadModule(
    "/src/hosted/runtime/operationPolicy.ts",
  );
  const { OPTIONAL_PACK_RESIDENT_BUDGET_BYTES } = await server.ssrLoadModule(
    "/src/hosted/runtime/optionalPackCompatibility.ts",
  );
  const document = catalogFixture();
  const verifiedCatalog = {
    document,
    exactBytes: catalogBytes(),
    signature: FIXTURE_SIGNATURE,
    allowDownloads: true,
    stale: false,
  };
  const variant = document.models[0].variants[0];
  const optionalPackProvenance = {
    license: "AGPL-3.0-or-later",
    sourceRepository: "a9lim/drowse-web-fixture",
    sourceRevision: variant.files[0].revision,
  };
  const progress = {
    modelVariantId: variant.id,
    files: [],
    bytesReceived: 512,
    bytesTotal: variant.downloadBytes + variant.requiredCorePackBytes,
    throughputBytesPerSecond: 256,
    etaSeconds: [3, 5],
    calculatingEta: false,
    stalled: false,
    offline: false,
    resumable: true,
  };
  const result = {
    modelVariantId: variant.id,
    objectHashes: [],
    downloadedBytes: progress.bytesTotal,
    totalBytes: progress.bytesTotal,
    installedAt: 1,
  };

  test("authoring UI recognizes only the typed fitting cancellation failure", () => {
    assert.equal(isFittingCancellation(new ApiError(
      409,
      "worker:FITTING_CANCELLED",
      "cancelled",
      {
        detail: "The browser fitting job was cancelled",
        error: { code: "FITTING_CANCELLED" },
      },
    )), true);
    assert.equal(isFittingCancellation({ code: "FITTING_CANCELLED" }), true);
    assert.equal(isFittingCancellation(new ApiError(
      500,
      "worker:FITTING_FAILED",
      "failed",
      { error: { code: "FITTING_FAILED" } },
    )), false);
    assert.equal(isFittingCancellation(new DOMException("cancelled", "AbortError")), false);
  });

  test("hosted authoring forms expose one typed fitting cancel action without changing HTTP mode", async () => {
    const [templated, templateLab] = await Promise.all([
      readFile("src/drawers/manifold/TemplatedForm.svelte", "utf8"),
      readFile("src/drawers/TemplateLabDrawer.svelte", "utf8"),
    ]);
    for (const source of [templated, templateLab]) {
      assert.match(source, /getHostedController\(\)/);
      assert.match(source, /hostedController\s*&&/);
      assert.match(source, /hostedController\.cancelFitting\(\)/);
      assert.match(source, /isFittingCancellation\(e\)/);
      assert.match(source, /<MorphText\s+text=\{cancelling\w* \? "cancelling…" : "cancel"\}/);
    }
    assert.match(templated, /onclick=\{cancelFit\}/);
    assert.match(templateLab, /onclick=\{cancelScore\}/);
  });

  function compatibleCapabilities(overrides = {}) {
    return {
      checkedAt: Date.now(),
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
        quotaBytes: 1_000_000_000,
        usageBytes: 0,
        availableBytes: 1_000_000_000,
        persisted: true,
      },
      signals: {
        deviceMemoryGiB: 8,
        logicalCpuCount: 8,
        mobile: false,
        runtimeClass: "desktop-chromium",
        appleMobile: false,
        language: "en-US",
        calibrationScore: 10,
      },
      deviceSignature: "fixture-device",
      runtimeVersion: document.runtimeAbi,
      issues: [],
      operations: {
        download: { available: true, reasons: [] },
        generation: { available: true, reasons: [] },
        fitting: { available: false, reasons: [] },
        manifold_artifacts: { available: true, reasons: [] },
        jlens_fitting: { available: false, reasons: [] },
        sae_training: { available: false, reasons: [] },
        session_admin: { available: false, reasons: [] },
        server_endpoints: { available: false, reasons: [] },
      },
      ...overrides,
    };
  }

  const downloadPayload = (overrides = {}) => ({
    modelVariantId: variant.id,
    contextTokens: 2048,
    ...overrides,
  });

  function request(requestId, command, payload) {
    return {
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      requestId,
      command,
      payload,
    };
  }

  function harness(downloader, overrides = {}) {
    const scope = new FakeWorkerScope();
    const loadRecords = [];
    const activationSpool = overrides.activationSpool ?? { async clear() {} };
    const manifests = [
      ...variant.files,
      ...variant.packs.find((pack) => pack.kind === "core").files,
    ];
    const installs = structuredClone(overrides.installs ?? [{
      id: variant.id,
      kind: "model",
      objectHashes: manifests.map((file) => file.sha256),
      installedAt: 1,
    }]);
    let selectedModelVariantId = overrides.selectedModelVariantId ?? null;
    const runtimeDownloader = {
      async download(...args) {
        const downloadResult = await downloader.download(...args);
        selectedModelVariantId = downloadResult.modelVariantId;
        return downloadResult;
      },
      async downloadPack(...args) {
        return downloader.downloadPack(...args);
      },
      cancel() {
        return downloader.cancel();
      },
    };
    new workerModule.HostedRuntimeWorker(scope, {
      contentStore: {
        async initialize() { await overrides.onContentStoreInitialize?.(); },
        async listInstalls() {
          if (overrides.listInstalls) {
            return overrides.listInstalls({ installs });
          }
          return structuredClone(installs);
        },
        async listLoadRecords() {
          if (overrides.listLoadRecords) {
            return overrides.listLoadRecords({ loadRecords });
          }
          return structuredClone(loadRecords);
        },
        async recordLoad(record) {
          if (overrides.recordLoad) {
            await overrides.recordLoad(structuredClone(record), loadRecords);
            return;
          }
          loadRecords.push(structuredClone(record));
        },
        async verifiedFile(sha256) {
          if (overrides.verifiedFile) {
            return overrides.verifiedFile(sha256, { installs, manifests });
          }
          const manifest = manifests.find((file) => file.sha256 === sha256);
          return manifest ? new Blob([new Uint8Array(manifest.bytes)]) : null;
        },
        async selectedModelVariantId() {
          if (overrides.selectedModelVariantIdRead) {
            return overrides.selectedModelVariantIdRead(selectedModelVariantId);
          }
          return selectedModelVariantId;
        },
        async setSelectedModelVariantId(value) {
          if (overrides.setSelectedModelVariantId) {
            await overrides.setSelectedModelVariantId(value);
          }
          selectedModelVariantId = value;
        },
        async removeInstall(id, expectedKind) {
          const index = installs.findIndex((install) => install.id === id);
          if (index < 0) return [];
          assert.equal(installs[index].kind, expectedKind);
          const [removed] = installs.splice(index, 1);
          if (removed.kind === "model" && selectedModelVariantId === id) {
            selectedModelVariantId = null;
          }
          return removed.objectHashes;
        },
        async collectGarbage() {
          return overrides.collectGarbage ? overrides.collectGarbage() : [];
        },
        async clear() {
          selectedModelVariantId = null;
          installs.length = 0;
        },
      },
      catalogRepository: {
        async get(request) {
          if (overrides.catalogGet) return overrides.catalogGet(request);
          return overrides.catalog ?? verifiedCatalog;
        },
        async withDownloadAuthorization(catalog, operation) {
          if (overrides.withDownloadAuthorization) {
            return overrides.withDownloadAuthorization(catalog, operation);
          }
          return operation();
        },
        async clear() { await overrides.catalogClear?.(); },
      },
      capabilityChecker: {
        async check() {
          overrides.onCapabilityCheck?.();
          if (overrides.checkError) throw overrides.checkError;
          if (overrides.capabilityCheck) return overrides.capabilityCheck();
          return overrides.capabilities ?? compatibleCapabilities();
        },
        clearAdapter() { overrides.onClearAdapter?.(); },
        adapterForLoad() {
          return overrides.adapterForLoad?.() ?? { fixture: true };
        },
      },
      storageEstimator: overrides.storageEstimator ?? (async () => ({
        quotaBytes: 1_000_000_000,
        usageBytes: 0,
        availableBytes: 1_000_000_000,
        persisted: true,
      })),
      downloader: runtimeDownloader,
      activationSpool,
      artifactService: overrides.artifactService ?? {
        handles() { return false; },
        async request() { throw new Error("unexpected artifact request"); },
        async clear() {},
      },
      modelBackend: overrides.modelBackend,
      sessionPersistence: overrides.sessionPersistence ?? {
        async claim() { return { status: "missing" }; },
        async save(input, ownerEpoch) {
          return {
            ...structuredClone(input),
            schemaVersion: 2,
            ownerEpoch,
            updatedAt: Date.now(),
          };
        },
        async delete() { return false; },
        async clear() {},
        close() {},
      },
      sessionCoordinatorFactory: overrides.sessionCoordinatorFactory ?? (() => ({
        async restore() { return { status: "not_found" }; },
        async persist() {},
        async close() {},
      })),
      clearRuntimeStorage: overrides.clearRuntimeStorage ?? (async () => {}),
    });
    return Object.assign(scope, {
      installs,
      loadRecords,
      activationSpool,
      setSelectedModelVariantId(value) {
        selectedModelVariantId = value;
      },
    });
  }

  async function response(scope, requestId) {
    return waitFor(() => scope.messages.find(
      (message) => message.kind === "response" && message.requestId === requestId,
    ));
  }

  async function acceptCatalog(scope) {
    scope.send(request("catalog", "catalog", {}));
    assert.equal((await response(scope, "catalog")).ok, true);
    scope.messages.length = 0;
  }

  async function prepareDownload(scope) {
    await prepareCapabilities(scope);
    await acceptCatalog(scope);
  }

  async function prepareCapabilities(scope) {
    scope.send(request("check", "check", undefined));
    assert.equal((await response(scope, "check")).ok, true);
    scope.messages.length = 0;
  }

  test("worker reports an unverified hosted distribution as unavailable", async () => {
    const scope = harness({
      async download() {},
      async downloadPack() {},
      cancel() { return false; },
    }, {
      catalogGet() {
        throw new catalogModule.CatalogValidationError(
          "HOSTED_DISTRIBUTION_NOT_VERIFIED",
          "The hosted Drowse artifact distribution has not been published yet",
        );
      },
    });
    scope.send(request("catalog-unverified", "catalog", {}));
    const rejected = await response(scope, "catalog-unverified");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "HOSTED_DISTRIBUTION_NOT_VERIFIED");
    assert.equal(rejected.error.status, 503);
    assert.equal(rejected.error.recoverable, false);
  });

  test("worker orders authoritative download progress and installation", async () => {
    let quotaChecked = false;
    const scope = harness({
      async download(_catalog, modelVariantId, options) {
        assert.equal(modelVariantId, variant.id);
        quotaChecked = await options.checkQuota({
          modelVariantId,
          file: variant.files[0],
          fileIndex: 0,
          fileCount: variant.files.length + 1,
          remainingFileBytes: variant.files[0].bytes,
          remainingManifestBytes: progress.bytesTotal,
        });
        options.onProgress(progress);
        return result;
      },
      cancel() { return false; },
    });
    await prepareDownload(scope);
    scope.send(request("download", "download", downloadPayload()));
    const completed = await response(scope, "download");

    assert.equal(quotaChecked, true);
    assert.deepEqual(completed.result, {
      modelVariantId: variant.id,
      installed: true,
      cancelled: false,
    });
    assert.deepEqual(
      scope.messages.map((message) =>
        message.kind === "event" ? `${message.event}:${message.payload.download?.phase ?? ""}` :
          `response:${message.requestId}`
      ),
      ["status:running", "progress:", "status:complete", "response:download"],
    );
  });

  test("worker defers resumable-storage admission to exact downloader state", async () => {
    let downloadCalls = 0;
    const scope = harness({
      async download(_catalog, modelVariantId, options) {
        downloadCalls += 1;
        assert.equal(await options.checkQuota({
          modelVariantId,
          file: variant.files[0],
          fileIndex: 0,
          fileCount: 1,
          remainingFileBytes: 1,
          remainingManifestBytes: 1,
        }), true);
        return result;
      },
      cancel() { return false; },
    }, {
      capabilities: compatibleCapabilities({
        storage: {
          quotaBytes: 1,
          usageBytes: 0,
          availableBytes: 1,
          persisted: true,
        },
      }),
      storageEstimator: async () => ({
        quotaBytes: 300 * 1024 * 1024,
        usageBytes: 0,
        availableBytes: 300 * 1024 * 1024,
        persisted: true,
      }),
    });
    await prepareDownload(scope);
    scope.send(request("resume-storage", "download", downloadPayload()));
    assert.equal((await response(scope, "resume-storage")).ok, true);
    assert.equal(downloadCalls, 1);
  });

  test("worker installs an exact optional pack with correlated progress", async () => {
    const optionalPack = {
      id: "smollm2-sae",
      kind: "sae",
      displayName: "SmolLM2 SAE",
      ...optionalPackProvenance,
      bytes: 320,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
      files: [{
        path: "packs/sae.safetensors",
        role: "instrument",
        url: `${variant.files[0].url.replace("weights/params.bin", "packs/sae.safetensors")}`,
        revision: variant.files[0].revision,
        bytes: 320,
        sha256: "9".repeat(64),
      }],
    };
    const packCatalog = structuredClone(verifiedCatalog);
    packCatalog.document.models[0].variants[0].packs.push(optionalPack);
    const packProgress = {
      ...progress,
      packId: optionalPack.id,
      bytesReceived: optionalPack.bytes,
      bytesTotal: optionalPack.bytes,
      files: [{
        path: optionalPack.files[0].path,
        bytesReceived: optionalPack.bytes,
        bytesTotal: optionalPack.bytes,
        resumable: false,
        verification: "verified",
      }],
    };
    let scope;
    let quotaChecked = false;
    const downloader = {
      async download() { return result; },
      async downloadPack(_catalog, modelVariantId, packId, options) {
        assert.equal(modelVariantId, variant.id);
        assert.equal(packId, optionalPack.id);
        quotaChecked = await options.checkQuota({
          modelVariantId,
          packId,
          file: optionalPack.files[0],
          fileIndex: 0,
          fileCount: 1,
          remainingFileBytes: optionalPack.bytes,
          remainingManifestBytes: optionalPack.bytes,
        });
        options.onProgress(packProgress);
        scope.installs.push({
          id: packId,
          kind: "pack",
          objectHashes: [optionalPack.files[0].sha256],
          installedAt: 2,
        });
        return {
          modelVariantId,
          packId,
          objectHashes: [optionalPack.files[0].sha256],
          downloadedBytes: optionalPack.bytes,
          totalBytes: optionalPack.bytes,
          installedAt: 2,
        };
      },
      cancel() { return false; },
    };
    scope = harness(downloader, {
      catalog: packCatalog,
      selectedModelVariantId: variant.id,
    });
    await prepareDownload(scope);
    scope.send(request("download-pack", "download_pack", {
      modelVariantId: variant.id,
      packId: optionalPack.id,
    }));
    const completed = await response(scope, "download-pack");

    assert.equal(quotaChecked, true);
    assert.deepEqual(completed.result, {
      modelVariantId: variant.id,
      packId: optionalPack.id,
      installed: true,
      cancelled: false,
    });
    assert.deepEqual(scope.installs.find((install) => install.id === optionalPack.id), {
      id: optionalPack.id,
      kind: "pack",
      objectHashes: [optionalPack.files[0].sha256],
      installedAt: 2,
    });
    assert.deepEqual(
      scope.messages.map((message) =>
        message.kind === "event"
          ? `${message.event}:${message.payload.download?.phase ?? message.payload.packId ?? ""}`
          : `response:${message.requestId}`
      ),
      ["status:running", `progress:${optionalPack.id}`, "status:complete", "response:download-pack"],
    );
  });

  test("worker rejects and skips a J-lens whose matrix exceeds adapter buffer limits", async () => {
    const optionalPack = {
      id: "smollm2-jlens-too-large",
      kind: "jlens",
      displayName: "SmolLM2 J-lens",
      ...optionalPackProvenance,
      bytes: 128,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
      files: [{
        path: "packs/jlens-too-large.safetensors",
        role: "instrument",
        url: variant.files[0].url.replace(
          "weights/params.bin",
          "packs/jlens-too-large.safetensors",
        ),
        revision: variant.files[0].revision,
        bytes: 128,
        sha256: "8".repeat(64),
      }],
    };
    const packCatalog = structuredClone(verifiedCatalog);
    packCatalog.document.models[0].variants[0].packs.push(optionalPack);
    const normalWebGpu = compatibleCapabilities().webGpu;
    const limitedCapabilities = compatibleCapabilities({
      webGpu: {
        ...normalWebGpu,
        limits: {
          ...normalWebGpu.limits,
          maxBufferSize: 1_048_576,
          maxStorageBufferBindingSize: 1_048_576,
        },
      },
    });
    let downloadCalls = 0;
    const rejectedScope = harness({
      async download() { return result; },
      async downloadPack() {
        downloadCalls += 1;
        throw new Error("hardware-incompatible J-lens must not download");
      },
      cancel() { return false; },
    }, {
      catalog: packCatalog,
      capabilities: limitedCapabilities,
      selectedModelVariantId: variant.id,
    });
    await prepareDownload(rejectedScope);
    rejectedScope.send(request("download-oversize-jlens", "download_pack", {
      modelVariantId: variant.id,
      packId: optionalPack.id,
    }));
    const rejected = await response(rejectedScope, "download-oversize-jlens");

    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "JLENS_DEVICE_BUFFER_LIMIT");
    assert.equal(downloadCalls, 0);

    let loadInput;
    const loadScope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      catalog: packCatalog,
      capabilities: limitedCapabilities,
      installs: [
        {
          id: variant.id,
          kind: "model",
          objectHashes: [
            ...variant.files,
            ...variant.packs.find((pack) => pack.kind === "core").files,
          ].map((file) => file.sha256),
          installedAt: 1,
        },
        {
          id: optionalPack.id,
          kind: "pack",
          objectHashes: [optionalPack.files[0].sha256],
          installedAt: 2,
        },
      ],
      modelBackend: {
        async load(input) {
          loadInput = input;
          return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
        },
        async request() { return null; },
        async generate() {},
        async stop() {},
        async unload() {},
      },
    });
    await prepareDownload(loadScope);
    loadScope.send(request("load-with-oversize-jlens", "load", downloadPayload()));
    assert.equal((await response(loadScope, "load-with-oversize-jlens")).ok, true);
    assert.deepEqual(loadInput.optionalPacks, []);
  });

  test("worker rejects unsafe concurrent pack downloads and loads only a safe resident subset", async () => {
    const packBytes = Math.ceil(OPTIONAL_PACK_RESIDENT_BUDGET_BYTES * 0.6);
    const common = {
      ...optionalPackProvenance,
      bytes: packBytes,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
    };
    const jlens = {
      ...common,
      id: "smollm2-jlens-resident",
      kind: "jlens",
      displayName: "SmolLM2 resident J-lens",
      files: [{
        path: "packs/jlens-resident.safetensors",
        role: "instrument",
        url: variant.files[0].url.replace(
          "weights/params.bin",
          "packs/jlens-resident.safetensors",
        ),
        revision: variant.files[0].revision,
        bytes: packBytes,
        sha256: "1".repeat(64),
      }],
    };
    const sae = {
      ...common,
      id: "smollm2-sae-resident",
      kind: "sae",
      displayName: "SmolLM2 resident SAE",
      files: [{
        path: "packs/sae-resident.safetensors",
        role: "instrument",
        url: variant.files[0].url.replace(
          "weights/params.bin",
          "packs/sae-resident.safetensors",
        ),
        revision: variant.files[0].revision,
        bytes: packBytes,
        sha256: "2".repeat(64),
      }],
    };
    const packCatalog = structuredClone(verifiedCatalog);
    const catalogVariant = packCatalog.document.models[0].variants[0];
    catalogVariant.packs.push(jlens, sae);
    const normalWebGpu = compatibleCapabilities().webGpu;
    const residentCapabilities = compatibleCapabilities({
      webGpu: {
        ...normalWebGpu,
        limits: {
          ...normalWebGpu.limits,
          maxBufferSize: 64 * 1024 * 1024,
          maxStorageBufferBindingSize: 64 * 1024 * 1024,
        },
      },
    });
    const baseInstall = {
      id: variant.id,
      kind: "model",
      objectHashes: [
        ...variant.files,
        ...variant.packs.find((pack) => pack.kind === "core").files,
      ].map((file) => file.sha256),
      installedAt: 1,
    };
    const jlensInstall = {
      id: jlens.id,
      kind: "pack",
      objectHashes: jlens.files.map((file) => file.sha256),
      installedAt: 2,
    };
    let downloadCalls = 0;
    const downloadScope = harness({
      async download() { return result; },
      async downloadPack() {
        downloadCalls += 1;
        throw new Error("resident-budget-incompatible pack must not download");
      },
      cancel() { return false; },
    }, {
      catalog: packCatalog,
      capabilities: residentCapabilities,
      selectedModelVariantId: variant.id,
      installs: [baseInstall, jlensInstall],
    });
    await prepareDownload(downloadScope);
    downloadScope.send(request("download-over-resident-budget", "download_pack", {
      modelVariantId: variant.id,
      packId: sae.id,
    }));
    const rejected = await response(downloadScope, "download-over-resident-budget");

    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "OPTIONAL_PACK_RESIDENT_BUDGET_EXCEEDED");
    assert.match(rejected.error.message, /Remove one installed tool/u);
    assert.equal(downloadCalls, 0);

    let loadInput;
    const allFiles = [
      ...catalogVariant.files,
      ...catalogVariant.packs.flatMap((pack) => pack.files),
    ];
    const loadScope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      catalog: packCatalog,
      capabilities: residentCapabilities,
      selectedModelVariantId: variant.id,
      installs: [
        baseInstall,
        jlensInstall,
        {
          id: sae.id,
          kind: "pack",
          objectHashes: sae.files.map((file) => file.sha256),
          installedAt: 3,
        },
      ],
      verifiedFile(sha256) {
        const file = allFiles.find((candidate) => candidate.sha256 === sha256);
        return file ? { size: file.bytes } : null;
      },
      modelBackend: {
        async load(input) {
          loadInput = input;
          return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
        },
        async request() { return null; },
        async generate() {},
        async stop() {},
        async unload() {},
      },
    });
    await prepareDownload(loadScope);
    loadScope.send(request("load-over-resident-budget", "load", downloadPayload()));
    assert.equal((await response(loadScope, "load-over-resident-budget")).ok, true);
    assert.deepEqual(
      loadInput.optionalPacks.map(({ pack }) => pack.id),
      [jlens.id],
    );
  });

  test("worker republishes a durably committed pack when metadata refresh fails", async () => {
    const optionalPack = {
      id: "smollm2-sae-durable",
      kind: "sae",
      displayName: "SmolLM2 durable SAE",
      ...optionalPackProvenance,
      bytes: 64,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
      files: [{
        path: "packs/sae-durable.safetensors",
        role: "instrument",
        url: variant.files[0].url.replace(
          "weights/params.bin",
          "packs/sae-durable.safetensors",
        ),
        revision: variant.files[0].revision,
        bytes: 64,
        sha256: "a".repeat(64),
      }],
    };
    const packCatalog = structuredClone(verifiedCatalog);
    packCatalog.document.models[0].variants[0].packs.push(optionalPack);
    let scope;
    let reads = 0;
    const downloader = {
      async download() { return result; },
      async downloadPack(_catalog, modelVariantId, packId) {
        scope.installs.push({
          id: packId,
          kind: "pack",
          objectHashes: [optionalPack.files[0].sha256],
          installedAt: 2,
        });
        return {
          modelVariantId,
          packId,
          objectHashes: [optionalPack.files[0].sha256],
          downloadedBytes: optionalPack.bytes,
          totalBytes: optionalPack.bytes,
          installedAt: 2,
        };
      },
      cancel() { return false; },
    };
    scope = harness(downloader, {
      catalog: packCatalog,
      selectedModelVariantId: variant.id,
      listInstalls({ installs }) {
        reads += 1;
        if (reads === 2) throw new Error("fixture pack metadata refresh failed");
        return structuredClone(installs);
      },
    });
    await prepareDownload(scope);
    scope.send(request("pack-committed-refresh-failure", "download_pack", {
      modelVariantId: variant.id,
      packId: optionalPack.id,
    }));
    const rejected = await response(scope, "pack-committed-refresh-failure");

    assert.equal(rejected.ok, false);
    assert.match(rejected.error.message, /pack metadata refresh failed/);
    const status = scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload;
    assert.deepEqual(status.installedPackIds, [optionalPack.id]);
    assert.equal(status.download.phase, "failed");
  });

  test("worker reconciles a same-ID pack against the current signed manifest", async () => {
    const optionalPack = {
      id: "smollm2-sae-current",
      kind: "sae",
      displayName: "SmolLM2 current SAE",
      ...optionalPackProvenance,
      bytes: 96,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
      files: [{
        path: "packs/sae-current.safetensors",
        role: "instrument",
        url: variant.files[0].url.replace(
          "weights/params.bin",
          "packs/sae-current.safetensors",
        ),
        revision: variant.files[0].revision,
        bytes: 96,
        sha256: "c".repeat(64),
      }],
    };
    const packCatalog = structuredClone(verifiedCatalog);
    packCatalog.document.models[0].variants[0].packs.push(optionalPack);
    let scope;
    let downloadCalls = 0;
    const downloader = {
      async download() { return result; },
      async downloadPack(_catalog, modelVariantId, packId) {
        downloadCalls += 1;
        const install = scope.installs.find((candidate) => candidate.id === packId);
        assert.ok(install);
        install.objectHashes = [optionalPack.files[0].sha256];
        return {
          modelVariantId,
          packId,
          objectHashes: [optionalPack.files[0].sha256],
          downloadedBytes: optionalPack.bytes,
          totalBytes: optionalPack.bytes,
          installedAt: 3,
        };
      },
      cancel() { return false; },
    };
    scope = harness(downloader, {
      catalog: packCatalog,
      selectedModelVariantId: variant.id,
      installs: [
        {
          id: variant.id,
          kind: "model",
          objectHashes: [
            ...variant.files,
            ...variant.packs.find((pack) => pack.kind === "core").files,
          ].map((file) => file.sha256),
          installedAt: 1,
        },
        {
          id: optionalPack.id,
          kind: "pack",
          objectHashes: ["d".repeat(64)],
          installedAt: 2,
        },
      ],
    });
    await prepareDownload(scope);
    scope.send(request("download-current-pack", "download_pack", {
      modelVariantId: variant.id,
      packId: optionalPack.id,
    }));
    const completed = await response(scope, "download-current-pack");

    assert.equal(completed.ok, true);
    assert.equal(downloadCalls, 1);
    assert.deepEqual(
      scope.installs.find((install) => install.id === optionalPack.id).objectHashes,
      [optionalPack.files[0].sha256],
    );
  });

  test("worker cancellation during optional-pack preflight never starts a download", async () => {
    const optionalPack = {
      id: "smollm2-jlens-cancel",
      kind: "jlens",
      displayName: "SmolLM2 cancelled J-lens",
      ...optionalPackProvenance,
      bytes: 64,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
      files: [{
        path: "packs/jlens-cancel.safetensors",
        role: "instrument",
        url: variant.files[0].url.replace(
          "weights/params.bin",
          "packs/jlens-cancel.safetensors",
        ),
        revision: variant.files[0].revision,
        bytes: 64,
        sha256: "e".repeat(64),
      }],
    };
    const packCatalog = structuredClone(verifiedCatalog);
    packCatalog.document.models[0].variants[0].packs.push(optionalPack);
    let releasePreflight;
    let reportPreflight;
    const preflightStarted = new Promise((resolve) => (reportPreflight = resolve));
    const preflightRelease = new Promise((resolve) => (releasePreflight = resolve));
    const scope = harness({
      async download() { return result; },
      async downloadPack() { assert.fail("cancelled preflight must not download"); },
      cancel() { return false; },
    }, {
      catalog: packCatalog,
      selectedModelVariantId: variant.id,
      async listInstalls({ installs }) {
        reportPreflight();
        await preflightRelease;
        return structuredClone(installs);
      },
    });
    await prepareDownload(scope);
    scope.send(request("pack-preflight", "download_pack", {
      modelVariantId: variant.id,
      packId: optionalPack.id,
    }));
    await preflightStarted;
    scope.send(request("cancel-pack-preflight", "cancel", undefined));
    assert.equal((await response(scope, "cancel-pack-preflight")).ok, true);
    releasePreflight();
    const cancelled = await response(scope, "pack-preflight");

    assert.deepEqual(cancelled.result, {
      modelVariantId: variant.id,
      packId: optionalPack.id,
      installed: false,
      cancelled: true,
    });
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" && message.payload.download?.phase === "running",
    ), false);
  });

  test("worker refuses optional packs for an uninstalled or stale model", async () => {
    const optionalPack = {
      id: "smollm2-jlens",
      kind: "jlens",
      displayName: "SmolLM2 J-lens",
      ...optionalPackProvenance,
      bytes: 128,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
      files: [{
        path: "packs/jlens.safetensors",
        role: "instrument",
        url: variant.files[0].url.replace("weights/params.bin", "packs/jlens.safetensors"),
        revision: variant.files[0].revision,
        bytes: 128,
        sha256: "b".repeat(64),
      }],
    };
    const packCatalog = structuredClone(verifiedCatalog);
    packCatalog.document.models[0].variants[0].packs.push(optionalPack);
    const downloader = {
      async download() { return result; },
      async downloadPack() { assert.fail("rejected pack must not download"); },
      cancel() { return false; },
    };
    const missing = harness(downloader, {
      catalog: packCatalog,
      installs: [],
      selectedModelVariantId: variant.id,
    });
    await prepareDownload(missing);
    missing.send(request("pack-model-missing", "download_pack", {
      modelVariantId: variant.id,
      packId: optionalPack.id,
    }));
    assert.equal(
      (await response(missing, "pack-model-missing")).error.code,
      "PACK_MODEL_NOT_INSTALLED",
    );

    const staleModel = harness(downloader, {
      catalog: packCatalog,
      selectedModelVariantId: variant.id,
      installs: [{
        id: variant.id,
        kind: "model",
        objectHashes: ["f".repeat(64)],
        installedAt: 1,
      }],
    });
    await prepareDownload(staleModel);
    staleModel.send(request("pack-model-stale", "download_pack", {
      modelVariantId: variant.id,
      packId: optionalPack.id,
    }));
    assert.equal(
      (await response(staleModel, "pack-model-stale")).error.code,
      "PACK_MODEL_BINDING_MISMATCH",
    );
  });

  test("worker completes an optional pack for an installed non-selected model", async () => {
    const optionalPack = {
      id: "smollm2-unselected-sae",
      kind: "sae",
      displayName: "SmolLM2 background SAE",
      ...optionalPackProvenance,
      bytes: 128,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
      files: [{
        path: "packs/unselected-sae.safetensors",
        role: "instrument",
        url: variant.files[0].url.replace(
          "weights/params.bin",
          "packs/unselected-sae.safetensors",
        ),
        revision: variant.files[0].revision,
        bytes: 128,
        sha256: "e".repeat(64),
      }],
    };
    const packCatalog = structuredClone(verifiedCatalog);
    packCatalog.document.models[0].variants[0].packs.push(optionalPack);
    let scope;
    const downloader = {
      async download() { return result; },
      async downloadPack(_catalog, modelVariantId, packId, options) {
        await options.authorizeSelection(new AbortController().signal);
        scope.installs.push({
          id: packId,
          kind: "pack",
          objectHashes: [optionalPack.files[0].sha256],
          installedAt: 2,
        });
        return {
          modelVariantId,
          packId,
          objectHashes: [optionalPack.files[0].sha256],
          downloadedBytes: optionalPack.bytes,
          totalBytes: optionalPack.bytes,
          installedAt: 2,
        };
      },
      cancel() { return false; },
    };
    scope = harness(downloader, {
      catalog: packCatalog,
      selectedModelVariantId: null,
    });
    await prepareDownload(scope);
    scope.send(request("pack-model-unselected", "download_pack", {
      modelVariantId: variant.id,
      packId: optionalPack.id,
    }));

    const completed = await response(scope, "pack-model-unselected");
    assert.equal(completed.ok, true);
    assert.equal(completed.result.installed, true);
    assert.equal(scope.installs.some((install) => install.id === optionalPack.id), true);
  });

  test("worker reauthorizes an optional pack after destructive-lock admission", async () => {
    const optionalPack = {
      id: "smollm2-raced-sae",
      kind: "sae",
      displayName: "SmolLM2 raced SAE",
      ...optionalPackProvenance,
      bytes: 128,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
      files: [{
        path: "packs/raced-sae.safetensors",
        role: "instrument",
        url: variant.files[0].url.replace("weights/params.bin", "packs/raced-sae.safetensors"),
        revision: variant.files[0].revision,
        bytes: 128,
        sha256: "c".repeat(64),
      }],
    };
    const packCatalog = structuredClone(verifiedCatalog);
    packCatalog.document.models[0].variants[0].packs.push(optionalPack);
    let scope;
    const downloader = {
      async download() { return result; },
      async downloadPack(_catalog, _modelVariantId, _packId, options) {
        scope.installs.splice(
          scope.installs.findIndex((install) => install.id === variant.id),
          1,
        );
        await options.authorizeSelection(new AbortController().signal);
        assert.fail("a pack must not install after its model was deleted");
      },
      cancel() { return false; },
    };
    scope = harness(downloader, {
      catalog: packCatalog,
      selectedModelVariantId: variant.id,
    });
    await prepareDownload(scope);
    scope.send(request("pack-model-deleted-race", "download_pack", {
      modelVariantId: variant.id,
      packId: optionalPack.id,
    }));
    const rejected = await response(scope, "pack-model-deleted-race");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "PACK_MODEL_NOT_INSTALLED");
    assert.equal(scope.installs.some((install) => install.id === optionalPack.id), false);
  });

  test("worker rejects correlatable invalid requests immediately", async () => {
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    });
    scope.send({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      requestId: "invalid",
      command: "unknown",
      payload: undefined,
    });
    const rejected = await response(scope, "invalid");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "INVALID_WORKER_REQUEST");
  });

  test("runtime service inventory is complete and unknown RPC methods fail at the boundary", async () => {
    const runtime = new fakeRuntimeModule.DeterministicFakeRuntime();
    const artifactService = new artifactModule.BrowserDrowseArchiveService(
      undefined,
      undefined,
      "test",
    );
    for (const [serviceName, methods] of Object.entries(
      contracts.RUNTIME_SERVICE_METHODS,
    )) {
      const service = runtime[serviceName];
      assert.ok(service, `missing fixture service ${serviceName}`);
      for (const method of Object.keys(methods)) {
        assert.equal(
          typeof service[method],
          "function",
          `missing fixture method ${serviceName}.${method}`,
        );
        const requestPayload = { service: serviceName, method, args: [] };
        const route = operationPolicyModule.runtimeServiceRoute(requestPayload);
        assert.ok(route, `missing browser route ${serviceName}.${method}`);
        assert.equal(
          artifactService.handles(requestPayload),
          route === "artifact" || route === "hybrid_manifold_read",
          `artifact route drift for ${serviceName}.${method}`,
        );
      }
      assert.equal(service.__unknown_runtime_method__, undefined);
    }
    await runtime.dispose();

    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    });
    scope.send(request("unknown-service-method", "request", {
      service: "instruments",
      method: "startPreparaton",
      args: ["lens", { operation: "fetch" }],
    }));
    const rejected = await response(scope, "unknown-service-method");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "INVALID_WORKER_REQUEST");

    const worker = new FakeTransportWorker();
    const transport = new browserModule.WorkerRpcTransport(worker, {
      requestTimeoutMs: 0,
    });
    const client = new browserModule.BrowserRuntimeClient(transport, {
      ownsTransport: false,
    });
    assert.equal(typeof client.instruments.startPreparation, "function");
    assert.equal(client.instruments.startPreparaton, undefined);
    await client.dispose();
    transport.dispose();
  });

  test("worker accepts and forwards explicit offline catalog state", async () => {
    let received;
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      catalogGet(request) {
        received = request;
        return verifiedCatalog;
      },
    });
    scope.send(request("offline-catalog", "catalog", {
      offline: true,
      preferCached: true,
    }));
    const accepted = await response(scope, "offline-catalog");
    assert.equal(accepted.ok, true);
    assert.deepEqual(received, { offline: true, preferCached: true });
  });

  test("worker requires a successful capability check before download", async () => {
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    });
    await acceptCatalog(scope);
    scope.send(request("download", "download", downloadPayload()));
    const rejected = await response(scope, "download");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "COMPATIBILITY_CHECK_REQUIRED");
  });

  test("worker republishes a durably committed model install when metadata refresh fails", async () => {
    let scope;
    let reads = 0;
    const downloader = {
      async download() {
        scope.installs.push({
          id: variant.id,
          kind: "model",
          objectHashes: variant.files.map((file) => file.sha256),
          installedAt: 2,
        });
        return result;
      },
      cancel() { return false; },
    };
    scope = harness(downloader, {
      installs: [],
      listInstalls({ installs }) {
        reads += 1;
        if (reads === 1) throw new Error("fixture metadata refresh failed");
        return structuredClone(installs);
      },
    });
    await prepareDownload(scope);
    scope.send(request("download-committed-refresh-failure", "download", downloadPayload()));
    const rejected = await response(scope, "download-committed-refresh-failure");
    assert.equal(rejected.ok, false);
    assert.match(rejected.error.message, /metadata refresh failed/);
    const status = scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload;
    assert.deepEqual(status.installedModelVariantIds, [variant.id]);
    assert.equal(status.download.phase, "failed");
  });

  test("worker never publishes a selected model removed after download commit", async () => {
    let scope;
    const downloader = {
      async download() {
        scope.installs.push({
          id: variant.id,
          kind: "model",
          objectHashes: variant.files.map((file) => file.sha256),
          installedAt: 2,
        });
        return result;
      },
      cancel() { return false; },
    };
    scope = harness(downloader, {
      installs: [],
      listInstalls({ installs }) {
        installs.length = 0;
        scope.setSelectedModelVariantId(null);
        return [];
      },
    });
    await prepareDownload(scope);
    scope.send(request("download-cleared-after-commit", "download", downloadPayload()));
    const completed = await response(scope, "download-cleared-after-commit");

    assert.deepEqual(completed.result, {
      modelVariantId: variant.id,
      installed: false,
      cancelled: false,
    });
    const status = scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload;
    assert.equal(status.selectedModelVariantId, null);
    assert.deepEqual(status.installedModelVariantIds, []);
  });

  test("worker rejects a superseded catalog before starting a download", async () => {
    let downloads = 0;
    const scope = harness({
      async download(_catalog, _modelVariantId, options) {
        return options.withCatalogAuthorization(async () => {
          downloads += 1;
          return result;
        });
      },
      cancel() { return false; },
    }, {
      async withDownloadAuthorization() {
        throw Object.assign(new Error("refresh the accepted catalog"), {
          code: "CATALOG_SUPERSEDED",
          recoverable: true,
          status: 409,
        });
      },
    });
    await prepareDownload(scope);
    scope.send(request("download-superseded", "download", downloadPayload()));
    const rejected = await response(scope, "download-superseded");
    assert.equal(rejected.error.code, "CATALOG_SUPERSEDED");
    assert.equal(downloads, 0);
  });

  test("worker reports authorization-time catalog expiry as a recoverable conflict", async () => {
    let downloads = 0;
    const scope = harness({
      async download(_catalog, _modelVariantId, options) {
        return options.withCatalogAuthorization(async () => {
          downloads += 1;
          return result;
        });
      },
      cancel() { return false; },
    }, {
      async withDownloadAuthorization() {
        throw new catalogModule.CatalogValidationError(
          "CATALOG_EXPIRED",
          "The accepted catalog expired while queued",
        );
      },
    });
    await prepareDownload(scope);
    scope.send(request("download-expired-at-admission", "download", downloadPayload()));
    const rejected = await response(scope, "download-expired-at-admission");

    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "CATALOG_EXPIRED");
    assert.equal(rejected.error.recoverable, true);
    assert.equal(rejected.error.status, 409);
    assert.equal(downloads, 0);
  });

  test("worker preserves a loaded runtime when compatibility is rechecked", async () => {
    let checkCalls = 0;
    let unloadCalls = 0;
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      onCapabilityCheck() { checkCalls += 1; },
      modelBackend: {
        async load() {
          return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
        },
        async request() { return null; },
        async generate() {},
        async stop() {},
        async unload() { unloadCalls += 1; },
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-before-recheck", "load", downloadPayload()));
    assert.equal((await response(scope, "load-before-recheck")).ok, true);
    scope.messages.length = 0;

    scope.send(request("check-while-loaded", "check", undefined));
    const rejected = await response(scope, "check-while-loaded");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "RUNTIME_IN_USE");
    assert.equal(checkCalls, 1);
    assert.equal(unloadCalls, 0);
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" && message.event === "status",
    ), false);
  });

  test("worker preserves a ready backend when a model switch fails preflight", async () => {
    const switchCatalog = structuredClone(verifiedCatalog);
    const alternate = structuredClone(variant);
    alternate.id = "fixture-alternate";
    switchCatalog.document.models[0].variants.push(alternate);
    let loads = 0;
    let unloads = 0;
    let requests = 0;
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      catalog: switchCatalog,
      modelBackend: {
        async load() {
          loads += 1;
          return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
        },
        async request() {
          requests += 1;
          return { active: true };
        },
        async generate() {},
        async stop() {},
        async unload() { unloads += 1; },
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-before-failed-switch", "load", downloadPayload()));
    assert.equal((await response(scope, "load-before-failed-switch")).ok, true);
    scope.messages.length = 0;

    scope.send(request("failed-switch", "load", {
      modelVariantId: alternate.id,
      contextTokens: 2048,
    }));
    const rejected = await response(scope, "failed-switch");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "MODEL_NOT_INSTALLED");
    assert.equal(loads, 1);
    assert.equal(unloads, 0);
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" && message.event === "status",
    ), false);

    scope.send(request("request-after-failed-switch", "request", {
      service: "sessions",
      method: "get",
      args: [],
    }));
    assert.deepEqual(
      (await response(scope, "request-after-failed-switch")).result,
      { active: true },
    );
    assert.equal(requests, 1);
  });

  test("worker reinitializes content storage after external invalidation", async () => {
    let initializeCalls = 0;
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      onContentStoreInitialize() { initializeCalls += 1; },
    });

    scope.send(request("storage-first", "storage", undefined));
    assert.equal((await response(scope, "storage-first")).ok, true);
    scope.send(request("storage-after-versionchange", "storage", undefined));
    assert.equal((await response(scope, "storage-after-versionchange")).ok, true);
    assert.equal(initializeCalls, 2);
  });

  test("worker enforces measured context and unsafe override at admission", async () => {
    const calls = [];
    const scope = harness({
      async download() {
        calls.push("download");
        return result;
      },
      cancel() { return false; },
    }, {
      capabilities: compatibleCapabilities({
        signals: {
          deviceMemoryGiB: 8,
          logicalCpuCount: 8,
          mobile: false,
          language: "en-US",
          calibrationScore: 1,
        },
      }),
    });
    await prepareDownload(scope);
    scope.send(request("wrong-context", "download", downloadPayload({ contextTokens: 4096 })));
    assert.equal((await response(scope, "wrong-context")).error.code, "CONTEXT_UNSUPPORTED");
    scope.send(request("unsafe", "download", downloadPayload()));
    assert.equal((await response(scope, "unsafe")).error.code, "UNSAFE_MODEL_OVERRIDE_REQUIRED");
    scope.send(request("override", "download", downloadPayload({ explicitUnsafeOverride: true })));
    assert.equal((await response(scope, "override")).ok, true);
    assert.deepEqual(calls, ["download"]);
  });

  test("worker rechecks catalog expiry when a download starts", async () => {
    const expiredCatalog = {
      ...verifiedCatalog,
      document: {
        ...verifiedCatalog.document,
        expiresAt: "2026-02-01T00:00:00.000Z",
      },
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { catalog: expiredCatalog });
    await prepareDownload(scope);
    scope.send(request("download", "download", downloadPayload()));
    assert.equal((await response(scope, "download")).error.code, "CATALOG_EXPIRED");
  });

  test("worker settles a thrown capability check as failed", async () => {
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { checkError: new Error("fixture check failure") });
    scope.send(request("check", "check", undefined));
    const rejected = await response(scope, "check");
    const status = scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    );
    assert.equal(rejected.ok, false);
    assert.equal(status.payload.lifecycle, "failed");
    assert.equal(status.payload.error.message, "fixture check failure");
  });

  test("worker routes Drowse pack list, install, export, and delete with progress", async () => {
    const calls = [];
    let finishInstall;
    const pack = {
      id: "manifolds/local/worker_fixture",
      namespace: "local",
      name: "worker_fixture",
      template: null,
      installedAt: 1,
      producerVersion: "test",
      source: { uri: "local", repository: null, revision: null },
    };
    const artifactService = new artifactModule.BrowserDrowseArchiveService({
      async initialize() { calls.push("initialize"); },
      async list() { calls.push("list"); return [pack]; },
      async install(source, options) {
        calls.push(`install:${source.size}:${options.force}`);
        options.onProgress({
          phase: "verifying",
          path: "manifolds/local/worker_fixture/manifold.json",
          verifiedBytes: 8,
          totalBytes: 16,
        });
        await new Promise((resolve) => { finishInstall = resolve; });
        calls.push("install:committed");
        return pack;
      },
      async export(primary) {
        calls.push(`export:${primary}`);
        return new Blob([primary]);
      },
      async remove(primary) { calls.push(`remove:${primary}`); return true; },
      async clear() { calls.push("clear"); },
      async close() {},
    }, undefined, "test");
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { artifactService });
    await prepareCapabilities(scope);

    scope.send(request("artifact-list", "request", {
      service: "manifolds",
      method: "drowseArchiveList",
      args: [],
    }));
    assert.deepEqual((await response(scope, "artifact-list")).result, { packs: [pack] });

    scope.messages.length = 0;
    const source = new Blob(["archive"]);
    scope.send(request("artifact-install", "request", {
      service: "manifolds",
      method: "drowseArchiveInstall",
      args: [source, { force: true }],
    }));
    const progressEvent = await waitFor(() => scope.messages.find(
      (message) => message.kind === "event" && message.event === "progress",
    ));
    assert.deepEqual(progressEvent.payload, {
      phase: "verifying",
      path: "manifolds/local/worker_fixture/manifold.json",
      verifiedBytes: 8,
      totalBytes: 16,
    });
    assert.equal(scope.messages.some(
      (message) => message.kind === "response" && message.requestId === "artifact-install",
    ), false);
    finishInstall();
    assert.deepEqual((await response(scope, "artifact-install")).result, pack);
    assert.ok(calls.indexOf("install:committed") < calls.length);

    scope.send(request("artifact-export", "request", {
      service: "manifolds",
      method: "drowseArchiveExport",
      args: [pack.id],
    }));
    assert.equal(
      await (await response(scope, "artifact-export")).result.text(),
      pack.id,
    );
    scope.send(request("artifact-delete", "request", {
      service: "manifolds",
      method: "drowseArchiveDelete",
      args: [pack.id],
    }));
    assert.deepEqual((await response(scope, "artifact-delete")).result, {
      id: pack.id,
      removed: true,
    });
    assert.deepEqual(calls, [
      "initialize", "list",
      "initialize", `install:${source.size}:true`, "install:committed",
      "initialize", `export:${pack.id}`,
      "initialize", `remove:${pack.id}`,
    ]);
  });

  test("worker returns artifact validation failures without invoking the backend", async () => {
    let backendRequests = 0;
    const artifactService = new artifactModule.BrowserDrowseArchiveService({
      async initialize() {},
      async list() {
        throw Object.assign(new Error("fixture manifest rejected"), {
          code: "MANIFEST_INVALID",
          recoverable: false,
          status: 400,
        });
      },
      async install() { throw new Error("not reached"); },
      async export() { throw new Error("not reached"); },
      async remove() { throw new Error("not reached"); },
      async clear() {},
      async close() {},
    }, undefined, "test");
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      artifactService,
      modelBackend: {
        async load() { return { prefillTokensPerSecond: null, decodeTokensPerSecond: null }; },
        async request() { backendRequests += 1; return null; },
        async generate() {},
        async stop() {},
        async unload() {},
      },
    });
    await prepareCapabilities(scope);
    scope.send(request("artifact-invalid", "request", {
      service: "manifolds",
      method: "drowseArchiveExport",
      args: ["../escape"],
    }));
    const rejected = await response(scope, "artifact-invalid");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "WORKER_OPERATION_FAILED");
    assert.match(rejected.error.message, /must be manifolds/);
    assert.equal(backendRequests, 0);
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" && message.event === "progress",
    ), false);

    scope.send(request("artifact-coded-failure", "request", {
      service: "manifolds",
      method: "drowseArchiveList",
      args: [],
    }));
    const coded = await response(scope, "artifact-coded-failure");
    assert.equal(coded.ok, false);
    assert.deepEqual(coded.error, {
      code: "MANIFEST_INVALID",
      message: "fixture manifest rejected",
      recoverable: false,
      status: 400,
    });
    assert.equal(backendRequests, 0);
  });

  test("loaded manifold listings retain fitted runtime metadata over local authoring rows", async () => {
    const unfitted = { namespace: "local", name: "curve", fitted_for_session: false };
    const fitted = {
      ...unfitted,
      fitted_for_session: true,
      layers_fitted: 26,
      intrinsic_dim: 1,
      resolved_fit_mode: "authored",
    };
    const localOnly = { namespace: "local", name: "draft", fitted_for_session: false };
    const core = { namespace: "default", name: "concept", fitted_for_session: true };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      artifactService: {
        handles(input) { return input.service === "manifolds" && input.method === "list"; },
        async request() { return { manifolds: [unfitted, localOnly] }; },
        async clear() {},
      },
      modelBackend: {
        async load() { return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 }; },
        async request() { return { manifolds: [core, fitted] }; },
        async generate() {},
        async stop() {},
        async unload() {},
      },
    });
    await prepareDownload(scope);
    scope.send(request("list-before-model", "request", {
      service: "manifolds", method: "list", args: [],
    }));
    assert.deepEqual((await response(scope, "list-before-model")).result, {
      manifolds: [unfitted, localOnly],
    });
    scope.send(request("load-fitted-list", "load", downloadPayload()));
    assert.equal((await response(scope, "load-fitted-list")).ok, true);
    scope.send(request("list-after-model", "request", {
      service: "manifolds", method: "list", args: [],
    }));
    assert.deepEqual((await response(scope, "list-after-model")).result, {
      manifolds: [core, fitted, localOnly],
    });
  });

  test("loaded manifold mutations refresh the runtime before reporting success", async () => {
    const events = [];
    const mutations = [
      "create",
      "createDiscover",
      "createFromTemplate",
      "delete",
      "install",
      "merge",
      "drowseArchiveInstall",
      "drowseArchiveDelete",
    ];
    const artifactService = {
      handles(input) {
        return input.service === "manifolds" &&
          [...mutations, "search"].includes(input.method);
      },
      async request(input) {
        events.push(`artifact:${input.method}`);
        return { method: input.method, committed: true };
      },
      async clear() {},
    };
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async refreshManifolds() { events.push("refresh"); },
      async request() { return null; },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { artifactService, modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-mutation-refresh", "load", downloadPayload()));
    assert.equal((await response(scope, "load-mutation-refresh")).ok, true);
    scope.messages.length = 0;

    for (const [index, method] of mutations.entries()) {
      const requestId = `mutation-refresh-${index}`;
      scope.send(request(requestId, "request", {
        service: "manifolds",
        method,
        args: [],
      }));
      assert.deepEqual((await response(scope, requestId)).result, {
        method,
        committed: true,
      });
    }
    scope.send(request("nonmutation-no-refresh", "request", {
      service: "manifolds",
      method: "search",
      args: [],
    }));
    assert.equal((await response(scope, "nonmutation-no-refresh")).ok, true);
    assert.deepEqual(events, [
      ...mutations.flatMap((method) => [`artifact:${method}`, "refresh"]),
      "artifact:search",
    ]);
  });

  test("loaded manifold deletion rejects registered probe aliases before committing", async () => {
    let artifactCommits = 0;
    let probes = [{
      family: "geometry",
      name: "friendly alias",
      manifold: "local/demo:sae-release-a",
    }];
    const artifactService = {
      handles(input) {
        return input.service === "manifolds" &&
          ["delete", "drowseArchiveDelete"].includes(input.method);
      },
      async request(input) {
        artifactCommits += 1;
        return input.method === "delete"
          ? { namespace: "local", name: "demo", removed: true }
          : { id: "manifolds/local/demo", removed: true };
      },
      async clear() {},
    };
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async refreshManifolds() {},
      async request(input) {
        if (input.service === "probes" && input.method === "list") {
          return { probes };
        }
        throw new Error(`unexpected backend request ${input.service}.${input.method}`);
      },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { artifactService, modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-delete-guard", "load", downloadPayload()));
    assert.equal((await response(scope, "load-delete-guard")).ok, true);
    scope.messages.length = 0;

    scope.send(request("delete-in-use", "request", {
      service: "manifolds",
      method: "delete",
      args: ["local", "demo"],
    }));
    const blocked = await response(scope, "delete-in-use");
    assert.equal(blocked.ok, false);
    assert.equal(blocked.error.code, "MANIFOLD_IN_USE");
    assert.match(blocked.error.message, /friendly alias/);
    assert.deepEqual(blocked.error.detail.probes, ["friendly alias"]);
    assert.equal(artifactCommits, 0);

    scope.send(request("pack-delete-in-use", "request", {
      service: "manifolds",
      method: "drowseArchiveDelete",
      args: ["manifolds/local/demo"],
    }));
    assert.equal((await response(scope, "pack-delete-in-use")).error.code, "MANIFOLD_IN_USE");
    assert.equal(artifactCommits, 0);

    probes = [];
    scope.send(request("delete-detached", "request", {
      service: "manifolds",
      method: "delete",
      args: ["local", "demo"],
    }));
    assert.deepEqual((await response(scope, "delete-detached")).result, {
      namespace: "local",
      name: "demo",
      removed: true,
    });
    assert.equal(artifactCommits, 1);
  });

  test("committed mutation refresh failure invalidates the model with a typed reload requirement", async () => {
    let artifactCommits = 0;
    let backendRequests = 0;
    let backendUnloads = 0;
    const artifactService = {
      handles(input) {
        return input.service === "manifolds" && input.method === "install";
      },
      async request() {
        artifactCommits += 1;
        return { installed: true };
      },
      async clear() {},
    };
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async refreshManifolds() {
        throw Object.assign(new Error("fixture compiler rejected the installed pack"), {
          code: "CORE_PACK_INVALID",
          recoverable: true,
          status: 409,
        });
      },
      async request() { backendRequests += 1; return null; },
      async generate() {},
      async stop() {},
      async unload() { backendUnloads += 1; },
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { artifactService, modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-refresh-failure", "load", downloadPayload()));
    assert.equal((await response(scope, "load-refresh-failure")).ok, true);
    scope.messages.length = 0;

    scope.send(request("committed-refresh-failure", "request", {
      service: "manifolds",
      method: "install",
      args: [],
    }));
    const rejected = await response(scope, "committed-refresh-failure");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "ARTIFACT_REFRESH_REQUIRED");
    assert.equal(rejected.error.detail.artifactCommitted, true);
    assert.equal(rejected.error.detail.modelReloadRequired, true);
    assert.deepEqual(rejected.error.detail.committedChange, {
      service: "manifolds",
      method: "install",
      result: { installed: true },
    });
    assert.equal(rejected.error.detail.refreshFailure.code, "CORE_PACK_INVALID");
    assert.equal(artifactCommits, 1);
    assert.equal(backendUnloads, 1);
    const invalidated = scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    );
    assert.equal(invalidated.payload.lifecycle, "unloaded");
    assert.equal(invalidated.payload.modelVariantId, null);

    scope.send(request("blocked-until-reload", "request", {
      service: "profiles",
      method: "list",
      args: [],
    }));
    const blocked = await response(scope, "blocked-until-reload");
    assert.equal(blocked.ok, false);
    assert.equal(blocked.error.code, "MODEL_RUNTIME_NOT_READY");
    assert.equal(backendRequests, 0);
  });

  test("ordinary model requests remain isolated on the model backend", async () => {
    const backendRequests = [];
    const artifactService = new artifactModule.BrowserDrowseArchiveService({
      async initialize() { throw new Error("ordinary requests must not initialize artifacts"); },
      async list() { throw new Error("not reached"); },
      async install() { throw new Error("not reached"); },
      async export() { throw new Error("not reached"); },
      async remove() { throw new Error("not reached"); },
      async clear() {},
      async close() {},
    }, undefined, "test");
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request(input) {
        backendRequests.push(input);
        return { profiles: ["backend"] };
      },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { artifactService, modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-artifact-isolation", "load", downloadPayload()));
    assert.equal((await response(scope, "load-artifact-isolation")).ok, true);
    scope.messages.length = 0;
    scope.send(request("ordinary-profiles", "request", {
      service: "profiles",
      method: "list",
      args: [],
    }));
    assert.deepEqual((await response(scope, "ordinary-profiles")).result, {
      profiles: ["backend"],
    });
    scope.send(request("wrong-artifact-namespace", "request", {
      service: "templates",
      method: "drowseArchiveList",
      args: [],
    }));
    const wrongNamespace = await response(scope, "wrong-artifact-namespace");
    assert.equal(wrongNamespace.ok, false);
    assert.equal(wrongNamespace.error.code, "INVALID_WORKER_REQUEST");
    assert.deepEqual(backendRequests, [
      { service: "profiles", method: "list", args: [] },
    ]);
  });

  test("manifold reads merge loaded core packs with browser-local artifacts", async () => {
    const local = {
      namespace: "local",
      name: "saved",
      description: "local",
    };
    const core = {
      namespace: "default",
      name: "calm.focused",
      description: "required core",
    };
    const artifactService = {
      handles(input) {
        return input.service === "manifolds" && ["list", "get"].includes(input.method);
      },
      async request(input) {
        if (input.method === "list") return { manifolds: [local] };
        if (input.args[0] === local.namespace && input.args[1] === local.name) return local;
        throw new Error("local manifold not found");
      },
      async clear() {},
    };
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request(input) {
        if (input.method === "list") return { manifolds: [core] };
        if (input.args[0] === core.namespace && input.args[1] === core.name) return core;
        throw Object.assign(new Error("core manifold not found"), {
          code: "CORE_PACK_SELECTOR_NOT_FOUND",
        });
      },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { artifactService, modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-manifold-merge", "load", downloadPayload()));
    assert.equal((await response(scope, "load-manifold-merge")).ok, true);
    scope.messages.length = 0;

    scope.send(request("manifold-list-merged", "request", {
      service: "manifolds",
      method: "list",
      args: [],
    }));
    assert.deepEqual(
      (await response(scope, "manifold-list-merged")).result.manifolds,
      [core, local],
    );
    scope.send(request("manifold-get-core", "request", {
      service: "manifolds",
      method: "get",
      args: [core.namespace, core.name],
    }));
    assert.deepEqual((await response(scope, "manifold-get-core")).result, core);
    scope.send(request("manifold-get-local", "request", {
      service: "manifolds",
      method: "get",
      args: [local.namespace, local.name],
    }));
    assert.deepEqual((await response(scope, "manifold-get-local")).result, local);
  });

  test("fixture backend lets the worker fall back to browser-local manifolds", async () => {
    const local = {
      namespace: "local",
      name: "saved",
      description: "browser-local fixture manifold",
    };
    const artifactService = {
      handles(input) {
        return input.service === "manifolds" && ["list", "get"].includes(input.method);
      },
      async request(input) {
        if (input.method === "list") return { manifolds: [local] };
        if (input.args[0] === local.namespace && input.args[1] === local.name) return local;
        throw new Error("local manifold not found");
      },
      async clear() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      artifactService,
      modelBackend: new fakeModelBackendModule.DeterministicBrowserModelBackend(),
    });
    await prepareDownload(scope);
    scope.send(request("load-fixture-artifact-fallback", "load", downloadPayload()));
    assert.equal((await response(scope, "load-fixture-artifact-fallback")).ok, true);
    scope.messages.length = 0;
    scope.send(request("fixture-manifold-get-local", "request", {
      service: "manifolds",
      method: "get",
      args: [local.namespace, local.name],
    }));
    assert.deepEqual((await response(scope, "fixture-manifold-get-local")).result, local);
  });

  test("clear-all waits for the artifact service clear commit", async () => {
    let clearStarted = false;
    let finishClear;
    const artifactService = {
      handles() { return false; },
      async request() { throw new Error("not reached"); },
      async clear() {
        clearStarted = true;
        await new Promise((resolve) => { finishClear = resolve; });
      },
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { artifactService });
    scope.send(request("clear-artifacts", "clear", undefined));
    await waitFor(() => clearStarted);
    assert.equal(scope.messages.some(
      (message) => message.kind === "response" && message.requestId === "clear-artifacts",
    ), false);
    finishClear();
    assert.equal((await response(scope, "clear-artifacts")).ok, true);
  });

  test("clear-all waits for exact runtime cache cleanup after catalog removal", async () => {
    const calls = [];
    let finishRuntimeStorageClear;
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      async catalogClear() { calls.push("catalog"); },
      async clearRuntimeStorage() {
        calls.push("runtime-storage");
        await new Promise((resolve) => { finishRuntimeStorageClear = resolve; });
      },
    });
    scope.send(request("clear-runtime-storage", "clear", undefined));
    await waitFor(() => calls.includes("runtime-storage"));
    assert.deepEqual(calls, ["catalog", "runtime-storage"]);
    assert.equal(scope.messages.some(
      (message) => message.kind === "response" &&
        message.requestId === "clear-runtime-storage",
    ), false);
    finishRuntimeStorageClear();
    assert.equal((await response(scope, "clear-runtime-storage")).ok, true);
  });

  test("clear-all reports legacy runtime storage cleanup failure", async () => {
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      async clearRuntimeStorage() {
        throw new Error("legacy WebLLM database is blocked");
      },
    });
    scope.send(request("clear-runtime-storage-failure", "clear", undefined));
    const failed = await response(scope, "clear-runtime-storage-failure");
    assert.equal(failed.ok, false);
    assert.match(failed.error.message, /legacy WebLLM database is blocked/u);
  });

  test("clear-all stops before other stores when an activation writer survives unload", async () => {
    let artifactClears = 0;
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      activationSpool: {
        async clear() { throw new Error("activation writer still open"); },
      },
      artifactService: {
        handles() { return false; },
        async request() { throw new Error("not reached"); },
        async clear() { artifactClears += 1; },
      },
    });
    scope.send(request("clear-active-capture", "clear", undefined));
    const failed = await response(scope, "clear-active-capture");
    assert.equal(failed.ok, false);
    assert.match(failed.error.message, /activation writer still open/);
    assert.equal(artifactClears, 0);
    assert.deepEqual(scope.installs.map((install) => install.id), [variant.id]);
  });

  test("clear-all reports committed content state when a later store fails", async () => {
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      artifactService: {
        handles() { return false; },
        async request() { throw new Error("not reached"); },
        async clear() { throw new Error("artifact clear failed"); },
      },
    });
    scope.send(request("clear-partial-failure", "clear", undefined));
    const failed = await response(scope, "clear-partial-failure");
    assert.equal(failed.ok, false);
    assert.match(failed.error.message, /artifact clear failed/);
    const finalState = scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload;
    assert.deepEqual(finalState.installedModelVariantIds, []);
    assert.deepEqual(finalState.installedPackIds, []);
    assert.equal(finalState.selectedModelVariantId, null);
  });

  test("fake runtime mirrors started-first stateful and mutation-free stateless streams", async () => {
    const runtime = new fakeRuntimeModule.DeterministicFakeRuntime({
      response: "fixture",
    });
    const messages = [];
    const unsubscribe = runtime.events.subscribe((message) => messages.push(message));
    await runtime.events.open();

    runtime.events.send({
      type: "generate",
      input: [{ role: "user", content: "fixture prompt" }],
      stateless: false,
    });
    await waitFor(() => messages.some(
      (message) => message.type === "done" || message.type === "error",
    ));
    assert.equal(
      messages.find((message) => message.type === "error")?.message,
      undefined,
    );
    assert.deepEqual(messages.map((message) => message.type), [
      "started", "tree_mutated", "token", "tree_mutated", "done",
    ]);
    assert.equal(messages[0].node_id, null);
    const statefulTree = await runtime.tree.get();

    messages.length = 0;
    runtime.events.send({
      type: "generate",
      input: [{ role: "user", content: "fixture prompt" }],
      stateless: true,
    });
    await waitFor(() => messages.some((message) => message.type === "done"));
    assert.deepEqual(messages.map((message) => message.type), [
      "started", "token", "done",
    ]);
    assert.equal(messages[0].node_id, null);
    assert.equal(messages[1].node_id, statefulTree.active_node_id);
    assert.equal(messages[2].node_id, statefulTree.active_node_id);
    assert.deepEqual(await runtime.tree.get(), statefulTree);

    unsubscribe();
    await runtime.dispose();
  });

  test("fake runtime exposes only its installed catalog instrument packs", async () => {
    const runtime = new fakeRuntimeModule.DeterministicFakeRuntime({
      instrumentPacks: [
        { id: "fixture-jlens", kind: "jlens", displayName: "Fixture J-lens" },
        { id: "fixture-sae", kind: "sae", displayName: "Fixture SAE" },
      ],
    });
    assert.deepEqual(await runtime.tree.replayCapabilities(), {
      jointLogprobs: { available: true, reason: null },
    });
    await assert.rejects(
      runtime.manifolds.get("local", "saved"),
      (error) => error.code === "CORE_PACK_SELECTOR_NOT_FOUND",
    );
    const session = await runtime.sessions.get();
    assert.equal(session.jlens_fitted, true);
    const lensBlock = session.instruments.find((row) => row.family === "lens");
    const saeBlock = session.instruments.find((row) => row.family === "sae");
    assert.equal(lensBlock.source, "fixture-jlens");
    assert.deepEqual(lensBlock.capabilities.preparations, []);
    assert.equal(lensBlock.capabilities.source_switch, false);
    assert.equal(session.instruments.find((row) => row.family === "sae").source, "fixture-sae");
    assert.deepEqual(saeBlock.capabilities.preparations, []);
    assert.deepEqual((await runtime.instruments.sources("lens")).sources.map((row) => row.source), [
      "fixture-jlens",
    ]);
    assert.deepEqual((await runtime.instruments.sources("sae")).sources.map((row) => row.source), [
      "fixture-sae",
    ]);
    assert.equal((await runtime.instruments.preparationStatus("lens")).state, "idle");
    await assert.rejects(
      runtime.instruments.activateInstalledPack("lens", {
        source: "workspace-r",
      }),
      (error) => error.code === "INSTRUMENT_PACK_SOURCE_MISMATCH",
    );
    await assert.rejects(
      runtime.instruments.activateInstalledPack("lens", {
        source: "fixture-jlens",
        contextBindingSha256: "wrong-context",
      }),
      (error) => error.code === "INSTRUMENT_PACK_CONTEXT_MISMATCH",
    );
    const activated = await runtime.instruments.activateInstalledPack("lens", {
      source: "fixture-jlens",
    });
    assert.equal(activated.state, "active");
    assert.equal(activated.source, "fixture-jlens");
    assert.equal(activated.contextBindingSha256, "fixture-context-binding");
    assert.equal(activated.reloadRequired, false);
    assert.match(
      (await runtime.instruments.cancelPreparation("lens")).message,
      /Model settings/u,
    );
    await assert.rejects(
      runtime.instruments.startPreparation("lens", {
        operation: "fetch",
        source: "fixture-jlens",
      }),
      (error) => error.code === "HOSTED_PACK_DOWNLOAD_MANAGED",
    );
    assert.deepEqual((await runtime.instruments.setLive("lens", { enabled: true })).layers, [0, 1]);
    const lensToken = await runtime.instruments.validateLensToken("fixture");
    assert.equal(lensToken.word, "fixture");
    const lensProbe = await runtime.probes.attach({ selector: "jlens/fixture" });
    assert.equal(lensProbe.family, "lens");
    const saeFeature = await runtime.instruments.validateSaeFeature(7);
    assert.equal(saeFeature.id, 7);
    const saeProbe = await runtime.probes.attach({ selector: "sae/7" });
    assert.equal(saeProbe.family, "sae");
    await runtime.instruments.setLive("sae", { enabled: true });
    const geometryProbe = await runtime.probes.attach({ selector: "fixture/calm.focused" });
    assert.equal(geometryProbe.family, "geometry");
    assert.deepEqual((await runtime.probes.list()).probes.map((probe) => probe.name), [
      "jlens/fixture",
      "sae/7",
      "fixture/calm.focused",
    ]);
    assert.equal(
      (await runtime.instruments.tokenReadout("lens", "root", 0)).measurements.provenance,
      "replayed",
    );

    const messages = [];
    runtime.events.subscribe((message) => messages.push(message));
    await runtime.events.open();
    runtime.events.send({
      type: "generate",
      input: [{ role: "user", content: "measure every fixture instrument" }],
      stateless: false,
    });
    await waitFor(() => messages.some(
      (message) => message.type === "done" || message.type === "error",
    ));
    assert.equal(
      messages.find((message) => message.type === "error")?.message,
      undefined,
    );
    const token = messages.find((message) => message.type === "token");
    assert.equal(token.raw_index, 0);
    assert.equal(Number.isSafeInteger(token.token_id), true);
    assert.equal(token.top_alts, null);
    assert.equal(token.measurements.scope, "token");
    assert.equal(token.measurements.provenance, "captured");
    assert.equal(
      token.measurements.instruments.geometry.readings["fixture/calm.focused"].coords.length,
      1,
    );
    assert.equal(
      token.measurements.instruments.lens.readings["jlens/fixture"].unit,
      "mean_token_probability",
    );
    assert.equal(token.measurements.instruments.lens.readout.layers.length, 2);
    assert.equal(
      token.measurements.instruments.sae.readings["sae/7"].unit,
      "activation_over_max",
    );
    assert.ok(token.measurements.instruments.sae.readout.features.length >= 2);
    const done = messages.find((message) => message.type === "done");
    assert.equal(done.result.measurements.scope, "aggregate");
    assert.equal(done.result.measurements.scores["sae/7"] > 0, true);

    const generated = (await runtime.tree.get()).nodes.find(
      (node) => node.role === "assistant" && node.finish_reason !== null,
    );
    assert.ok(generated);
    assert.equal(generated.raw_token_ids.length, generated.tokens.length);
    assert.equal(generated.tokens[0].raw_index, 0);
    assert.equal(generated.tokens[0].top_alts, undefined);
    assert.equal(generated.tokens[0].measurements.scope, "token");
    assert.equal(generated.aggregate_readings["fixture/calm.focused"] < 1, true);
    await runtime.dispose();
  });

  test("worker refreshes installed state after a verified object is evicted during load", async () => {
    const manifests = [
      ...variant.files,
      ...variant.packs.find((pack) => pack.kind === "core").files,
    ];
    const missingHash = manifests[0].sha256;
    const survivingVerifiedHashes = new Set(
      manifests.slice(1).map((manifest) => manifest.sha256),
    );
    let garbageCollections = 0;
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      selectedModelVariantId: variant.id,
      installs: [
        {
          id: variant.id,
          kind: "model",
          objectHashes: manifests.map((manifest) => manifest.sha256),
          installedAt: 1,
        },
        {
          id: "fixture-sae",
          kind: "pack",
          objectHashes: [],
          installedAt: 2,
        },
      ],
      verifiedFile(sha256, state) {
        if (sha256 === missingHash) {
          const install = state.installs.findIndex((entry) => entry.id === variant.id);
          assert.notEqual(install, -1);
          state.installs.splice(install, 1);
          return null;
        }
        const manifest = state.manifests.find((entry) => entry.sha256 === sha256);
        return manifest ? new Blob([new Uint8Array(manifest.bytes)]) : null;
      },
      collectGarbage() {
        garbageCollections += 1;
        survivingVerifiedHashes.clear();
        return [];
      },
    });
    await prepareDownload(scope);
    scope.send(request("storage-before-eviction", "storage", undefined));
    assert.equal((await response(scope, "storage-before-eviction")).ok, true);
    scope.messages.length = 0;

    scope.send(request("load-after-eviction", "load", downloadPayload()));
    const rejected = await response(scope, "load-after-eviction");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "VERIFIED_MODEL_OBJECT_MISSING");
    assert.equal(rejected.error.recoverable, true);
    const statuses = scope.messages.filter(
      (message) => message.kind === "event" && message.event === "status",
    );
    const refreshed = statuses.find(
      (message) => message.payload.installedModelVariantIds.length === 0,
    );
    assert.ok(refreshed);
    assert.equal(refreshed.payload.selectedModelVariantId, null);
    assert.deepEqual(refreshed.payload.installedPackIds, ["fixture-sae"]);
    const failed = statuses.at(-1).payload;
    assert.equal(failed.lifecycle, "failed");
    assert.equal(failed.error.code, "VERIFIED_MODEL_OBJECT_MISSING");
    assert.deepEqual(failed.installedModelVariantIds, []);
    assert.equal(failed.selectedModelVariantId, null);
    assert.equal(garbageCollections, 0);
    assert.equal(survivingVerifiedHashes.size, manifests.length - 1);
  });

  test("worker binds a same-ID model install to the current signed manifest", async () => {
    const currentCatalog = structuredClone(verifiedCatalog);
    const currentVariant = currentCatalog.document.models[0].variants[0];
    currentVariant.files[0].sha256 = "0".repeat(64);
    const currentManifests = [
      ...currentVariant.files,
      ...currentVariant.packs.find((pack) => pack.kind === "core").files,
    ];
    let verifiedReads = 0;
    let backendLoads = 0;
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      catalog: currentCatalog,
      verifiedFile(sha256) {
        verifiedReads += 1;
        const manifest = currentManifests.find((file) => file.sha256 === sha256);
        return manifest ? new Blob([new Uint8Array(manifest.bytes)]) : null;
      },
      modelBackend: {
        async load() {
          backendLoads += 1;
          return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
        },
        async request() { return null; },
        async generate() {},
        async stop() {},
        async unload() {},
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-stale-model-binding", "load", downloadPayload()));

    const rejected = await response(scope, "load-stale-model-binding");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "MODEL_INSTALL_BINDING_MISMATCH");
    assert.equal(rejected.error.recoverable, true);
    assert.equal(verifiedReads, 0);
    assert.equal(backendLoads, 0);
  });

  test("worker loads only verified artifacts and streams backend events", async () => {
    const calls = [];
    let scope;
    const backend = {
      async load(input) {
        calls.push(`load:${input.variant.id}:${input.contextTokens}`);
        assert.equal(input.artifacts.length, variant.files.length + 1);
        assert.deepEqual(input.optionalPacks, []);
        assert.equal(input.activationSpool, scope.activationSpool);
        assert.equal(input.runtimeClass, "desktop-chromium");
        assert.ok(input.artifacts.every((artifact) => artifact.file.size === artifact.manifest.bytes));
        assert.deepEqual(input.adapter, { fixture: true });
        input.onProgress({
          event: "progress",
          data: {
            kind: "model_load",
            phase: "webllm_initialization",
            progress: 0.5,
          },
        });
        input.onProgress({
          event: "progress",
          data: {
            kind: "model_load",
            phase: "core_pack_preparing",
          },
        });
        input.onProgress({
          event: "progress",
          data: {
            kind: "model_load",
            phase: "jlens_dictionary_uploading",
          },
        });
        return {
          prefillTokensPerSecond: 30,
          decodeTokensPerSecond: 12,
        };
      },
      async request(input, onProgress) {
        calls.push(`request:${input.service}.${input.method}`);
        if (input.service === "tree" && input.method === "navigate") {
          onProgress({
            event: "tree_mutated",
            data: {
              type: "tree_mutated",
              op: "navigate",
              rev: 3,
              added: [],
              removed: [],
              updated: [],
              active_node_id: "node-1",
              cast: {},
            },
          });
        } else {
          onProgress({ event: "progress", data: { step: 1 } });
        }
        return { id: "fixture-session" };
      },
      async generate(_input, emit) {
        calls.push("generate");
        emit({
          type: "started",
          generation_id: "generation-1",
          node_id: null,
          sibling_index: 0,
          sibling_count: 1,
        });
        emit({ type: "tree_mutated", op: "begin_assistant", rev: 1 });
        emit({
          type: "token",
          text: "local",
          thinking: false,
          token_id: 1,
          node_id: "node-1",
        });
        emit({ type: "tree_mutated", op: "finalize", rev: 2 });
        emit({
          type: "done",
          node_id: "node-1",
          sibling_index: 0,
          sibling_count: 1,
          result: {
            text: "local",
            tokens: 1,
            finish_reason: "stop",
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          },
        });
        return {
          prefillTokensPerSecond: 24,
          decodeTokensPerSecond: 9,
        };
      },
      async stop() { calls.push("stop"); },
      async unload() { calls.push("unload"); },
    };
    scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: backend,
      sessionCoordinatorFactory: () => ({
        async restore(identity, mode, compatibleIdentities) {
          calls.push(`restore:${identity.modelVariantId}:${identity.contextTokens}`);
          assert.equal(identity.runtimeIdentitySha256, variant.runtimeIdentitySha256);
          assert.equal(typeof identity.ownerEpoch, "string");
          assert.equal(mode, "replace");
          assert.deepEqual(compatibleIdentities, []);
          return { status: "not_found" };
        },
        async persist() { scope.timeline.push("session:persist"); },
        async close() { calls.push("session:close"); },
      }),
    });
    await prepareDownload(scope);
    scope.send(request("load", "load", {
      modelVariantId: variant.id,
      contextTokens: 2048,
    }));
    assert.equal((await response(scope, "load")).ok, true);
    assert.equal(scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload.lifecycle, "ready");
    assert.deepEqual(scope.loadRecords.map((record) => record.result), ["success"]);
    assert.equal(scope.loadRecords[0].decodeTokensPerSecond, 12);
    const loadProgress = scope.messages.filter(
      (message) => message.kind === "event" && message.event === "progress",
    );
    assert.deepEqual(loadProgress.map((message) => message.requestId), [
      "load",
      "load",
      "load",
    ]);
    assert.deepEqual(loadProgress.map((message) => message.payload), [
      {
        kind: "model_load",
        phase: "webllm_initialization",
        progress: 0.5,
      },
      {
        kind: "model_load",
        phase: "core_pack_preparing",
      },
      {
        kind: "model_load",
        phase: "jlens_dictionary_uploading",
      },
    ]);
    assert.deepEqual(calls, [
      `load:${variant.id}:2048`,
      `restore:${variant.id}:2048`,
    ]);

    scope.messages.length = 0;
    scope.send(request("service", "request", {
      service: "sessions",
      method: "get",
      args: [],
    }));
    assert.deepEqual((await response(scope, "service")).result, { id: "fixture-session" });
    assert.deepEqual(
      scope.messages.filter((message) => message.kind === "event").map((message) => message.event),
      ["progress"],
    );

    scope.messages.length = 0;
    scope.timeline.length = 0;
    scope.send(request("mutate", "request", {
      service: "tree",
      method: "navigate",
      args: ["node-1"],
    }));
    assert.equal((await response(scope, "mutate")).ok, true);
    assert.deepEqual(scope.timeline, [
      "event:tree_mutated",
      "session:persist",
      "response:mutate:ok",
    ]);
    const mutation = scope.messages.find(
      (message) => message.kind === "event" && message.event === "tree_mutated",
    );
    assert.equal(mutation.requestId, "mutate");
    assert.equal(mutation.generationId, null);
    assert.equal(mutation.payload.op, "navigate");
    assert.equal(mutation.payload.active_node_id, "node-1");

    scope.messages.length = 0;
    scope.timeline.length = 0;
    scope.send(request("generate", "generate", { type: "generate" }));
    assert.equal((await response(scope, "generate")).ok, true);
    const events = scope.messages.filter((message) => message.kind === "event");
    assert.deepEqual(events.map((message) => message.event), [
      "status", "started", "tree_mutated", "token", "tree_mutated", "done", "status",
    ]);
    assert.deepEqual(events.slice(1, 6).map((message) => message.generationId), [
      "generation-1", "generation-1", "generation-1", "generation-1", "generation-1",
    ]);
    assert.deepEqual(scope.timeline, [
      "event:status",
      "event:started",
      "event:tree_mutated",
      "event:token",
      "session:persist",
      "event:tree_mutated",
      "event:done",
      "event:status",
      "response:generate:ok",
    ]);
    assert.equal(scope.loadRecords.length, 2);
    assert.deepEqual(scope.loadRecords[1], {
      modelVariantId: variant.id,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      deviceSignature: "fixture-device",
      contextTokens: 2048,
      result: "success",
      prefillTokensPerSecond: 24,
      decodeTokensPerSecond: 9,
      recordedAt: scope.loadRecords[1].recordedAt,
    });
    assert.equal(Number.isSafeInteger(scope.loadRecords[1].recordedAt), true);
    assert.deepEqual(events.at(-1).payload.loadRecords, scope.loadRecords);

    scope.send(request("stop", "stop", undefined));
    await response(scope, "stop");
    scope.send(request("unload", "unload", undefined));
    await response(scope, "unload");
    assert.deepEqual(calls, [
      `load:${variant.id}:2048`,
      `restore:${variant.id}:2048`,
      "request:sessions.get",
      "request:tree.navigate",
      "generate",
      "stop",
      "session:close",
      "unload",
    ]);
  });

  for (const release of ["unload", "takeover"]) {
    test(`worker retains compatibility approval after ${release} and validates each fresh adapter`, async () => {
      let approved = false;
      let adapterChanged = false;
      let loads = 0;
      let adapterRequests = 0;
      let checks = 0;
      const scope = harness({
        async download() { return result; },
        cancel() { return false; },
      }, {
        onCapabilityCheck() { approved = true; checks += 1; },
        onClearAdapter() { approved = false; },
        adapterForLoad() {
          adapterRequests += 1;
          if (!approved) throw Object.assign(new Error("Run compatibility first"), { code: "COMPATIBILITY_CHECK_REQUIRED" });
          if (adapterChanged) throw Object.assign(new Error("The adapter changed"), { code: "WEBGPU_ADAPTER_CHANGED" });
          return { fixture: true };
        },
        modelBackend: {
          async load() {
            loads += 1;
            return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
          },
          async request() { return null; },
          async generate() {},
          async stop() {},
          async unload() {},
        },
      });
      await prepareDownload(scope);
      for (let attempt = 0; attempt < 2; attempt += 1) {
        scope.send(request(`load-${attempt}`, "load", downloadPayload()));
        const loaded = await response(scope, `load-${attempt}`);
        assert.equal(loaded.ok, true, JSON.stringify(loaded));
        scope.send(request(`release-${attempt}`, release, undefined));
        assert.equal((await response(scope, `release-${attempt}`)).ok, true);
      }
      assert.equal(checks, 1);
      assert.equal(loads, 2);
      assert.equal(adapterRequests, 2);

      adapterChanged = true;
      scope.send(request("load-changed-adapter", "load", downloadPayload()));
      const rejected = await response(scope, "load-changed-adapter");
      assert.equal(rejected.ok, false);
      assert.equal(rejected.error.code, "WEBGPU_ADAPTER_CHANGED");
      assert.equal(adapterRequests, 3);
      assert.equal(loads, 2);
    });
  }

  test("worker binds Apple-mobile sessions to the selected context capacity", async () => {
    let coordinatorOptions;
    const desktopCapabilities = compatibleCapabilities();
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      capabilities: compatibleCapabilities({
        signals: {
          ...desktopCapabilities.signals,
          mobile: true,
          runtimeClass: "apple-mobile-webkit",
          appleMobile: true,
        },
      }),
      modelBackend: {
        async load() {
          return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
        },
        async request() { return null; },
        async generate() {},
        async stop() {},
        async unload() {},
      },
      sessionCoordinatorFactory(_persistence, _execute, options) {
        coordinatorOptions = options;
        return {
          async restore() { return { status: "not_found" }; },
          async persist() {},
          async close() {},
        };
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-apple-mobile", "load", downloadPayload({
      explicitUnsafeOverride: true,
    })));
    const loadResponse = await response(scope, "load-apple-mobile");
    assert.equal(loadResponse.ok, true, JSON.stringify(loadResponse));
    assert.equal(coordinatorOptions.maxOutputTokens, 2048);
  });

  test("worker unloads after session cleanup fails and preserves the load error", async () => {
    let unloads = 0;
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: {
        async load() {
          return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
        },
        async request() { return null; },
        async generate() {},
        async stop() {},
        async unload() { unloads += 1; },
      },
      sessionCoordinatorFactory: () => ({
        async restore() { throw new Error("fixture restore failed"); },
        async persist() {},
        async close() { throw new Error("fixture close failed"); },
      }),
    });
    await prepareDownload(scope);
    scope.send(request("load-with-session-cleanup-failure", "load", downloadPayload()));
    const rejected = await response(scope, "load-with-session-cleanup-failure");
    assert.equal(rejected.ok, false);
    assert.match(rejected.error.message, /fixture restore failed/);
    assert.equal(unloads, 1);

    scope.send(request("check-after-session-cleanup-failure", "check", undefined));
    assert.equal((await response(scope, "check-after-session-cleanup-failure")).ok, true);
  });

  test("worker leaves ownership nonterminal when failed load cleanup cannot dispose the backend", async () => {
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: {
        async load() { throw new Error("fixture load failed"); },
        async request() { return null; },
        async generate() {},
        async stop() {},
        async unload() { throw new Error("fixture unload failed"); },
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-cleanup-fatal", "load", downloadPayload()));
    const rejected = await response(scope, "load-cleanup-fatal");
    assert.equal(rejected.error.code, "RUNTIME_CLEANUP_FAILED");
    assert.equal(rejected.error.recoverable, false);
    assert.match(rejected.error.message, /fixture load failed/);
    assert.match(rejected.error.message, /fixture unload failed/);
    assert.equal(scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload.lifecycle, "loading");
  });

  test("worker removes an optional pack whose verified object was evicted", async () => {
    const packCatalog = structuredClone(verifiedCatalog);
    const packVariant = packCatalog.document.models[0].variants[0];
    const optionalPack = {
      id: "fixture-evicted-sae",
      kind: "sae",
      displayName: "Evicted fixture SAE",
      ...optionalPackProvenance,
      bytes: 64,
      required: false,
      runtimeIdentitySha256: packVariant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [packVariant.contextProfiles[0].bindingSha256],
      files: [{
        path: "packs/evicted.safetensors",
        role: "instrument",
        url: packVariant.files[0].url.replace(
          "weights/params.bin",
          "packs/evicted.safetensors",
        ),
        revision: packVariant.files[0].revision,
        bytes: 64,
        sha256: "9".repeat(64),
      }],
    };
    packVariant.packs.push(optionalPack);
    const modelHashes = [
      ...packVariant.files,
      ...packVariant.packs.find((pack) => pack.kind === "core").files,
    ].map((file) => file.sha256);
    let loadedOptionalPacks;
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      catalog: packCatalog,
      installs: [
        { id: packVariant.id, kind: "model", objectHashes: modelHashes, installedAt: 1 },
        {
          id: optionalPack.id,
          kind: "pack",
          objectHashes: [optionalPack.files[0].sha256],
          installedAt: 2,
        },
      ],
      verifiedFile(sha256, state) {
        if (sha256 === optionalPack.files[0].sha256) return null;
        const manifest = state.manifests.find((file) => file.sha256 === sha256);
        return manifest ? new Blob([new Uint8Array(manifest.bytes)]) : null;
      },
      modelBackend: {
        async load(input) {
          loadedOptionalPacks = input.optionalPacks;
          return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
        },
        async request() { return null; },
        async generate() {},
        async stop() {},
        async unload() {},
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-after-optional-eviction", "load", downloadPayload()));
    assert.equal((await response(scope, "load-after-optional-eviction")).ok, true);
    assert.deepEqual(loadedOptionalPacks, []);
    assert.deepEqual(scope.installs.map((install) => install.id), [packVariant.id]);
    assert.deepEqual(scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload.installedPackIds, []);
  });

  test("worker passes only exact compatible optional instrument packs to the backend", async () => {
    const optionalPack = {
      id: "smollm2-jlens-load",
      kind: "jlens",
      displayName: "SmolLM2 J-lens",
      ...optionalPackProvenance,
      bytes: 128,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [variant.contextProfiles[0].bindingSha256],
      files: [
        {
          path: "packs/jlens-load.safetensors",
          role: "instrument",
          url: variant.files[0].url.replace("weights/params.bin", "packs/jlens-load.safetensors"),
          revision: variant.files[0].revision,
          bytes: 64,
          sha256: "1".repeat(64),
        },
        {
          path: "packs/jlens-load-copy.safetensors",
          role: "instrument",
          url: variant.files[0].url.replace("weights/params.bin", "packs/jlens-load-copy.safetensors"),
          revision: variant.files[0].revision,
          bytes: 64,
          sha256: "1".repeat(64),
        },
      ],
    };
    const stalePack = {
      ...structuredClone(optionalPack),
      id: "smollm2-sae-stale",
      kind: "sae",
      displayName: "Stale SAE",
      files: [{
        ...optionalPack.files[0],
        path: "packs/sae-stale.safetensors",
        sha256: "2".repeat(64),
      }],
    };
    const packCatalog = structuredClone(verifiedCatalog);
    packCatalog.document.models[0].variants[0].packs.push(optionalPack, stalePack);
    const baseManifests = [
      ...variant.files,
      ...variant.packs.find((pack) => pack.kind === "core").files,
    ];
    let loadInput;
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      catalog: packCatalog,
      installs: [
        {
          id: variant.id,
          kind: "model",
          objectHashes: baseManifests.map((file) => file.sha256),
          installedAt: 1,
        },
        {
          id: optionalPack.id,
          kind: "pack",
          objectHashes: [optionalPack.files[0].sha256],
          installedAt: 2,
        },
        {
          id: stalePack.id,
          kind: "pack",
          objectHashes: ["e".repeat(64)],
          installedAt: 3,
        },
      ],
      verifiedFile(sha256, state) {
        const manifest = [...state.manifests, ...optionalPack.files]
          .find((candidate) => candidate.sha256 === sha256);
        return manifest ? new Blob([new Uint8Array(manifest.bytes)]) : null;
      },
      modelBackend: {
        async load(input) {
          loadInput = input;
          return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
        },
        async request() { return null; },
        async generate() {},
        async stop() {},
        async unload() {},
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-optional-packs", "load", downloadPayload()));
    assert.equal((await response(scope, "load-optional-packs")).ok, true);
    assert.equal(loadInput.optionalPacks.length, 1);
    assert.equal(loadInput.optionalPacks[0].pack.id, optionalPack.id);
    assert.deepEqual(
      loadInput.optionalPacks[0].artifacts.map((artifact) => artifact.manifest.sha256),
      [optionalPack.files[0].sha256, optionalPack.files[0].sha256],
    );
    assert.equal(
      loadInput.artifacts.some((artifact) => artifact.manifest.sha256 === optionalPack.files[0].sha256),
      false,
    );
  });

  test("worker fails before backend load when persisted state exposes ambiguous instrument packs", async () => {
    const packCatalog = structuredClone(verifiedCatalog);
    const packVariant = packCatalog.document.models[0].variants[0];
    const instrument = (id, sha256) => ({
      id,
      kind: "sae",
      displayName: id,
      ...optionalPackProvenance,
      bytes: 64,
      required: false,
      runtimeIdentitySha256: packVariant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [packVariant.contextProfiles[0].bindingSha256],
      files: [{
        path: `packs/${id}.safetensors`,
        role: "instrument",
        url: packVariant.files[0].url.replace(
          "weights/params.bin",
          `packs/${id}.safetensors`,
        ),
        revision: packVariant.files[0].revision,
        bytes: 64,
        sha256,
      }],
    });
    const primary = instrument("fixture-sae-primary", "1".repeat(64));
    const secondary = instrument("fixture-sae-secondary", "2".repeat(64));
    packVariant.packs.push(primary, secondary);
    const baseManifests = [
      ...packVariant.files,
      ...packVariant.packs.find((pack) => pack.kind === "core").files,
    ];
    let backendLoads = 0;
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      catalog: packCatalog,
      installs: [
        {
          id: packVariant.id,
          kind: "model",
          objectHashes: baseManifests.map((file) => file.sha256),
          installedAt: 1,
        },
        {
          id: primary.id,
          kind: "pack",
          objectHashes: primary.files.map((file) => file.sha256),
          installedAt: 2,
        },
        {
          id: secondary.id,
          kind: "pack",
          objectHashes: secondary.files.map((file) => file.sha256),
          installedAt: 3,
        },
      ],
      verifiedFile(sha256, state) {
        const manifest = [...state.manifests, ...primary.files, ...secondary.files]
          .find((candidate) => candidate.sha256 === sha256);
        return manifest ? new Blob([new Uint8Array(manifest.bytes)]) : null;
      },
      modelBackend: {
        async load() {
          backendLoads += 1;
          return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
        },
        async request() { return null; },
        async generate() {},
        async stop() {},
        async unload() {},
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-ambiguous-packs", "load", downloadPayload()));
    const rejected = await response(scope, "load-ambiguous-packs");

    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "INSTRUMENT_PACK_AMBIGUOUS");
    assert.equal(backendLoads, 0);
  });

  test("worker loads multiple installed lens sources for in-session J/R swapping", async () => {
    const packCatalog = structuredClone(verifiedCatalog);
    const packVariant = packCatalog.document.models[0].variants[0];
    const instrument = (id, sha256) => ({
      id,
      kind: "jlens",
      displayName: id,
      ...optionalPackProvenance,
      bytes: 64,
      required: false,
      runtimeIdentitySha256: packVariant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [packVariant.contextProfiles[0].bindingSha256],
      files: [{
        path: `packs/${id}.safetensors`,
        role: "instrument",
        url: packVariant.files[0].url.replace("weights/params.bin", `packs/${id}.safetensors`),
        revision: packVariant.files[0].revision,
        bytes: 64,
        sha256,
      }],
    });
    const standard = instrument("fixture-jlens-standard", "1".repeat(64));
    const relp = instrument("fixture-rlens", "2".repeat(64));
    packVariant.packs.push(standard, relp);
    const baseManifests = [
      ...packVariant.files,
      ...packVariant.packs.find((pack) => pack.kind === "core").files,
    ];
    let loadedPacks = [];
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      catalog: packCatalog,
      installs: [
        {
          id: packVariant.id,
          kind: "model",
          objectHashes: baseManifests.map((file) => file.sha256),
          installedAt: 1,
        },
        ...[standard, relp].map((pack, index) => ({
          id: pack.id,
          kind: "pack",
          objectHashes: pack.files.map((file) => file.sha256),
          installedAt: index + 2,
        })),
      ],
      verifiedFile(sha256, state) {
        const manifest = [...state.manifests, ...standard.files, ...relp.files]
          .find((candidate) => candidate.sha256 === sha256);
        return manifest ? new Blob([new Uint8Array(manifest.bytes)]) : null;
      },
      modelBackend: {
        async load(input) {
          loadedPacks = input.optionalPacks.map(({ pack }) => pack.id);
          return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
        },
        async request() { return null; },
        async generate() {},
        async stop() {},
        async unload() {},
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-lens-sources", "load", downloadPayload()));
    assert.equal((await response(scope, "load-lens-sources")).ok, true);
    assert.deepEqual(loadedPacks, [standard.id, relp.id]);
  });

  test("worker enforces the hard browser fitting capability gate", async () => {
    let backendRequests = 0;
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { backendRequests += 1; return null; },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-fitting-gate", "load", downloadPayload()));
    assert.equal((await response(scope, "load-fitting-gate")).ok, true);
    scope.messages.length = 0;

    scope.send(request("gated-fit", "request", {
      service: "manifolds",
      method: "fit",
      args: ["local", "fixture", {}],
    }));
    const rejected = await response(scope, "gated-fit");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "WASM_FITTING_UNAVAILABLE");
    assert.equal(backendRequests, 0);
    assert.deepEqual(
      scope.messages
        .filter((message) => message.kind === "event" && message.event === "status")
        .map((message) => message.payload.fitting.phase),
      ["failed"],
    );
  });

  test("worker fails closed across authoring and excluded preparation RPCs", async () => {
    const backendRequests = [];
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request(input) {
        backendRequests.push(input);
        return { method: input.method };
      },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const blockedCapabilities = compatibleCapabilities();
    blockedCapabilities.operations.manifold_artifacts = {
      available: false,
      reasons: [{
        code: "MANIFOLD_ARTIFACT_BRIDGE_UNAVAILABLE",
        message: "Portable manifold archives are unavailable in this fixture",
        severity: "hard",
      }],
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend, capabilities: blockedCapabilities });
    await prepareDownload(scope);
    scope.send(request("load-policy-gates", "load", downloadPayload()));
    assert.equal((await response(scope, "load-policy-gates")).ok, true);
    scope.messages.length = 0;

    const blocked = [
      ["create", { service: "manifolds", method: "create", args: [{}] }, "MANIFOLD_ARTIFACT_BRIDGE_UNAVAILABLE"],
      ["discover", { service: "manifolds", method: "createDiscover", args: [{}] }, "MANIFOLD_ARTIFACT_BRIDGE_UNAVAILABLE"],
      ["templated", { service: "manifolds", method: "createFromTemplate", args: [{}] }, "MANIFOLD_ARTIFACT_BRIDGE_UNAVAILABLE"],
      ["merge", { service: "manifolds", method: "merge", args: [{}] }, "MANIFOLD_ARTIFACT_BRIDGE_UNAVAILABLE"],
      ["template-create", { service: "templates", method: "create", args: [{}] }, "MANIFOLD_ARTIFACT_BRIDGE_UNAVAILABLE"],
      ["template-score", { service: "templates", method: "score", args: ["local", "fixture", null] }, "WASM_FITTING_UNAVAILABLE"],
      ["lens-fit", { service: "instruments", method: "startPreparation", args: ["lens", { operation: "fit" }] }, "HOSTED_JLENS_FITTING_EXCLUDED"],
      ["sae-train", { service: "instruments", method: "startPreparation", args: ["sae", { operation: "train" }] }, "HOSTED_SAE_TRAINING_EXCLUDED"],
    ];
    for (const [requestId, payload, code] of blocked) {
      scope.send(request(requestId, "request", payload));
      const rejected = await response(scope, requestId);
      assert.equal(rejected.ok, false);
      assert.equal(rejected.error.code, code);
    }
    assert.equal(backendRequests.length, 0);

    const managed = [
      ["lens-fetch", { service: "instruments", method: "startPreparation", args: ["lens", { operation: "fetch" }] }],
      ["sae-fetch", { service: "instruments", method: "startPreparation", args: ["sae", { operation: "fetch" }] }],
    ];
    for (const [requestId, payload] of managed) {
      scope.send(request(requestId, "request", payload));
      const rejected = await response(scope, requestId);
      assert.equal(rejected.ok, false);
      assert.equal(rejected.error.code, "HOSTED_PACK_DOWNLOAD_MANAGED");
      assert.match(rejected.error.message, /Model settings/u);
    }

    const allowed = [
      ["lens-activate", { service: "instruments", method: "activateInstalledPack", args: ["lens", { source: "fixture-jlens" }] }],
      ["sae-activate", { service: "instruments", method: "activateInstalledPack", args: ["sae", { source: "fixture-sae", layer: 1 }] }],
      ["lens-sources", { service: "instruments", method: "sources", args: ["lens"] }],
    ];
    for (const [requestId, payload] of allowed) {
      scope.send(request(requestId, "request", payload));
      assert.deepEqual((await response(scope, requestId)).result, {
        method: payload.method,
      });
    }
    assert.deepEqual(
      backendRequests.map((input) => `${input.service}.${input.method}`),
      [
        "instruments.activateInstalledPack",
        "instruments.activateInstalledPack",
        "instruments.sources",
      ],
    );
  });

  test("worker cannot load an injected backend before the generation gate passes", async () => {
    const gatedCapabilities = compatibleCapabilities();
    gatedCapabilities.operations.generation = {
      available: false,
      reasons: [{
        code: "CUSTOM_MLC_BACKEND_UNAVAILABLE",
        message: "The pinned runtime is still gated",
        severity: "hard",
      }],
    };
    let backendLoads = 0;
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      capabilities: gatedCapabilities,
      modelBackend: {
        async load() {
          backendLoads += 1;
          return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
        },
        async request() { return null; },
        async generate() {},
        async stop() {},
        async unload() {},
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-release-gate", "load", downloadPayload()));
    const rejected = await response(scope, "load-release-gate");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "CUSTOM_MLC_BACKEND_UNAVAILABLE");
    assert.equal(backendLoads, 0);
  });

  test("worker runs every browser authoring computation through the fitting lifecycle", async () => {
    const fittingCapabilities = compatibleCapabilities();
    fittingCapabilities.operations.fitting = { available: true, reasons: [] };
    fittingCapabilities.operations.manifold_artifacts = { available: true, reasons: [] };
    const backendRequests = [];
    const progressEvents = [];
    let sessionPersists = 0;
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request(input, onProgress, signal) {
        backendRequests.push(`${input.service}.${input.method}`);
        assert.ok(signal instanceof AbortSignal);
        assert.equal(signal.aborted, false);
        onProgress({ event: "progress", data: { operation: input.method } });
        return { operation: input.method };
      },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      capabilities: fittingCapabilities,
      modelBackend: backend,
      sessionCoordinatorFactory: () => ({
        async restore() { return { status: "not_found" }; },
        async persist() { sessionPersists += 1; },
        async close() {},
      }),
    });
    await prepareDownload(scope);
    scope.send(request("load-fitting", "load", downloadPayload()));
    assert.equal((await response(scope, "load-fitting")).ok, true);
    scope.messages.length = 0;

    const fittingRequests = [
      ["extract", { service: "profiles", method: "extract", args: [{}] }],
      ["fit", { service: "manifolds", method: "fit", args: ["local", "fixture", {}] }],
      ["generate-manifold", { service: "manifolds", method: "generate", args: [{}] }],
      ["score-template", { service: "templates", method: "score", args: ["local", "fixture", null] }],
    ];
    for (const [requestId, payload] of fittingRequests) {
      scope.send(request(requestId, "request", payload));
      assert.deepEqual((await response(scope, requestId)).result, {
        operation: payload.method,
      });
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    assert.deepEqual(backendRequests, [
      "profiles.extract",
      "manifolds.fit",
      "manifolds.generate",
      "templates.score",
    ]);
    assert.equal(sessionPersists, 0);
    assert.deepEqual(
      scope.messages
        .filter((message) => message.kind === "event" && message.event === "progress")
        .map((message) => message.payload.operation),
      ["extract", "fit", "generate", "score"],
    );
    assert.deepEqual(
      scope.messages
        .filter((message) => message.kind === "event" && message.event === "status")
        .map((message) => message.payload.fitting.phase),
      [
        "running", "complete",
        "running", "complete",
        "running", "complete",
        "running", "complete",
      ],
    );
  });

  test("worker rejects a second fit and linearizes cancellation before acknowledgement", async () => {
    const fittingCapabilities = compatibleCapabilities();
    fittingCapabilities.operations.fitting = { available: true, reasons: [] };
    let signal;
    let stagedArtifact = false;
    let sessionPersists = 0;
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request(input, _onProgress, inputSignal) {
        assert.equal(input.service, "profiles");
        assert.equal(input.method, "extract");
        signal = inputSignal;
        stagedArtifact = true;
        return new Promise((_resolve, reject) => {
          inputSignal.addEventListener("abort", () => {
            stagedArtifact = false;
            reject(new DOMException("cancelled fixture fit", "AbortError"));
          }, { once: true });
        });
      },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      capabilities: fittingCapabilities,
      modelBackend: backend,
      sessionCoordinatorFactory: () => ({
        async restore() { return { status: "not_found" }; },
        async persist() { sessionPersists += 1; },
        async close() {},
      }),
    });
    await prepareDownload(scope);
    scope.send(request("load-cancellable-fit", "load", downloadPayload()));
    assert.equal((await response(scope, "load-cancellable-fit")).ok, true);
    scope.messages.length = 0;
    scope.timeline.length = 0;

    const fitPayload = { service: "profiles", method: "extract", args: [{}] };
    scope.send(request("first-fit", "request", fitPayload));
    await waitFor(() => signal);
    scope.send(request("second-fit", "request", fitPayload));
    const busy = await response(scope, "second-fit");
    assert.equal(busy.ok, false);
    assert.equal(busy.error.code, "FITTING_BUSY");

    scope.send(request("cancel-fit", "cancel_fitting", undefined));
    const cancelled = await response(scope, "first-fit");
    const cancellation = await response(scope, "cancel-fit");
    assert.equal(cancelled.ok, false);
    assert.equal(cancelled.error.code, "FITTING_CANCELLED");
    assert.equal(cancellation.ok, true);
    assert.equal(signal.aborted, true);
    assert.equal(stagedArtifact, false);
    assert.equal(sessionPersists, 0);
    assert.deepEqual(
      scope.messages
        .filter((message) => message.kind === "event" && message.event === "status")
        .map((message) => message.payload.fitting.phase),
      ["running", "cancelling", "complete"],
    );
    assert.ok(
      scope.timeline.indexOf("response:first-fit:error") <
        scope.timeline.indexOf("response:cancel-fit:ok"),
    );
  });

  test("template scoring cancellation settles rollback before acknowledgement and permits retry", async () => {
    const fittingCapabilities = compatibleCapabilities();
    fittingCapabilities.operations.fitting = { available: true, reasons: [] };
    fittingCapabilities.operations.manifold_artifacts = { available: true, reasons: [] };
    let activeSignal;
    let staged = false;
    let rollbackComplete = false;
    let scoringCalls = 0;
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request(input, _onProgress, signal) {
        assert.equal(input.service, "templates");
        assert.equal(input.method, "score");
        scoringCalls += 1;
        if (scoringCalls > 1) return { template: "fixture", contexts: [] };
        activeSignal = signal;
        staged = true;
        return new Promise((_resolve, reject) => {
          signal.addEventListener("abort", () => {
            setTimeout(() => {
              staged = false;
              rollbackComplete = true;
              reject(new DOMException("cancelled template score", "AbortError"));
            }, 5);
          }, { once: true });
        });
      },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { capabilities: fittingCapabilities, modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-template-score", "load", downloadPayload()));
    assert.equal((await response(scope, "load-template-score")).ok, true);
    scope.messages.length = 0;
    scope.timeline.length = 0;

    const scorePayload = {
      service: "templates",
      method: "score",
      args: ["local", "fixture", null],
    };
    scope.send(request("score-to-cancel", "request", scorePayload));
    await waitFor(() => activeSignal);
    scope.send(request("cancel-template-score", "cancel_fitting", undefined));
    const cancelled = await response(scope, "score-to-cancel");
    const cancellation = await response(scope, "cancel-template-score");
    assert.equal(cancelled.ok, false);
    assert.equal(cancelled.error.code, "FITTING_CANCELLED");
    assert.equal(cancellation.ok, true);
    assert.equal(activeSignal.aborted, true);
    assert.equal(staged, false);
    assert.equal(rollbackComplete, true);
    assert.ok(
      scope.timeline.indexOf("response:score-to-cancel:error") <
        scope.timeline.indexOf("response:cancel-template-score:ok"),
    );

    scope.send(request("score-retry", "request", scorePayload));
    assert.deepEqual((await response(scope, "score-retry")).result, {
      template: "fixture",
      contexts: [],
    });
    assert.equal(scoringCalls, 2);
    assert.equal(
      scope.messages.findLast(
        (message) => message.kind === "event" && message.event === "status",
      ).payload.fitting.phase,
      "complete",
    );
  });

  test("fitting cancellation bypasses the ordinary queue and skips a queued template score", async () => {
    const fittingCapabilities = compatibleCapabilities();
    fittingCapabilities.operations.fitting = { available: true, reasons: [] };
    let releaseBlocker;
    let blockerStarted = false;
    let fittingCalls = 0;
    const blocker = new Promise((resolve) => { releaseBlocker = resolve; });
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request(input) {
        if (input.service === "sessions" && input.method === "get") {
          blockerStarted = true;
          await blocker;
          return { id: "fixture-session" };
        }
        fittingCalls += 1;
        return { unexpected: true };
      },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { capabilities: fittingCapabilities, modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-queued-fit", "load", downloadPayload()));
    assert.equal((await response(scope, "load-queued-fit")).ok, true);
    scope.messages.length = 0;

    scope.send(request("blocking-request", "request", {
      service: "sessions",
      method: "get",
      args: [],
    }));
    await waitFor(() => blockerStarted);
    scope.send(request("queued-fit", "request", {
      service: "templates",
      method: "score",
      args: ["local", "fixture", null],
    }));
    scope.send(request("cancel-queued-fit", "cancel_fitting", undefined));
    assert.equal((await response(scope, "cancel-queued-fit")).ok, true);
    assert.equal(fittingCalls, 0);

    releaseBlocker();
    assert.equal((await response(scope, "blocking-request")).ok, true);
    const cancelled = await response(scope, "queued-fit");
    assert.equal(cancelled.ok, false);
    assert.equal(cancelled.error.code, "FITTING_CANCELLED");
    assert.equal(fittingCalls, 0);
  });

  test("worker unload cancels an admitted template score before releasing the backend", async () => {
    const fittingCapabilities = compatibleCapabilities();
    fittingCapabilities.operations.fitting = { available: true, reasons: [] };
    fittingCapabilities.operations.manifold_artifacts = { available: true, reasons: [] };
    const calls = [];
    let fitSignal;
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request(_input, _onProgress, signal) {
        fitSignal = signal;
        calls.push("fit:start");
        return new Promise((_resolve, reject) => {
          signal.addEventListener("abort", () => {
            calls.push("fit:settled");
            reject(new DOMException("cancelled fixture fit", "AbortError"));
          }, { once: true });
        });
      },
      async generate() {},
      async stop() {},
      async unload() { calls.push("unload"); },
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { capabilities: fittingCapabilities, modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-unload-fit", "load", downloadPayload()));
    assert.equal((await response(scope, "load-unload-fit")).ok, true);
    scope.messages.length = 0;

    scope.send(request("fit-before-unload", "request", {
      service: "templates",
      method: "score",
      args: ["local", "fixture", null],
    }));
    await waitFor(() => fitSignal);
    scope.send(request("unload-during-fit", "unload", undefined));
    const fit = await response(scope, "fit-before-unload");
    const unload = await response(scope, "unload-during-fit");
    assert.equal(fit.error.code, "FITTING_CANCELLED");
    assert.equal(unload.ok, true);
    assert.deepEqual(calls, ["fit:start", "fit:settled", "unload"]);
  });

  test("worker device loss settles fitting before backend unload", async () => {
    const fittingCapabilities = compatibleCapabilities();
    fittingCapabilities.operations.fitting = { available: true, reasons: [] };
    fittingCapabilities.operations.manifold_artifacts = { available: true, reasons: [] };
    const calls = [];
    let deviceLost;
    let fitSignal;
    let loads = 0;
    const backend = {
      async load(input) {
        loads += 1;
        deviceLost = input.onDeviceLost;
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request(_input, _onProgress, signal) {
        fitSignal = signal;
        calls.push("fit:start");
        return new Promise((_resolve, reject) => {
          signal.addEventListener("abort", () => {
            calls.push("fit:settled");
            reject(new DOMException("device lost during fit", "AbortError"));
          }, { once: true });
        });
      },
      async generate() {},
      async stop() {},
      async unload() { calls.push("unload"); },
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { capabilities: fittingCapabilities, modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-device-loss-fit", "load", downloadPayload()));
    assert.equal((await response(scope, "load-device-loss-fit")).ok, true);
    scope.messages.length = 0;

    scope.send(request("fit-before-device-loss", "request", {
      service: "manifolds",
      method: "generate",
      args: [{}],
    }));
    await waitFor(() => fitSignal);
    deviceLost({
      code: "GPU_DEVICE_LOST",
      message: "fixture adapter disappeared",
      confirmedOom: false,
    });
    const fit = await response(scope, "fit-before-device-loss");
    assert.equal(fit.error.code, "FITTING_CANCELLED");
    await waitFor(() => scope.messages.find(
      (message) => message.kind === "event" && message.event === "status" &&
        message.payload.lifecycle === "ready",
    ));
    assert.equal(fitSignal.aborted, true);
    assert.deepEqual(calls, ["fit:start", "fit:settled", "unload"]);
    assert.equal(loads, 2);
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" && message.event === "device_lost",
    ), false);
    assert.equal(scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload.lifecycle, "ready");
  });

  test("worker rejects a second admitted generation and stop cancels the queued first", async () => {
    let releaseService;
    let serviceStarted = false;
    let generateCalls = 0;
    let stopCalls = 0;
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() {
        serviceStarted = true;
        return new Promise((resolve) => { releaseService = resolve; });
      },
      async generate() { generateCalls += 1; },
      async stop() { stopCalls += 1; },
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-queued-generation", "load", downloadPayload()));
    assert.equal((await response(scope, "load-queued-generation")).ok, true);
    scope.messages.length = 0;

    scope.send(request("blocking-service", "request", {
      service: "sessions",
      method: "get",
      args: [],
    }));
    await waitFor(() => serviceStarted);
    scope.send(request("first-generation", "generate", { type: "generate" }));
    scope.send(request("second-generation", "generate", { type: "generate" }));
    const busy = await response(scope, "second-generation");
    assert.equal(busy.ok, false);
    assert.equal(busy.error.code, "GENERATION_BUSY");

    scope.send(request("stop-queued-generation", "stop", undefined));
    assert.equal((await response(scope, "stop-queued-generation")).ok, true);
    releaseService({ id: "fixture-session" });
    assert.equal((await response(scope, "blocking-service")).ok, true);
    assert.equal((await response(scope, "first-generation")).ok, true);
    assert.equal(generateCalls, 0);
    assert.equal(stopCalls, 1);
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" &&
        message.payload.generation?.phase === "running",
    ), false);
  });

  test("worker unload cancels an admitted generation before it reaches the backend", async () => {
    let releaseService;
    let serviceStarted = false;
    let generateCalls = 0;
    const calls = [];
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() {
        serviceStarted = true;
        return new Promise((resolve) => { releaseService = resolve; });
      },
      async generate() {
        generateCalls += 1;
      },
      async stop() {
        calls.push("stop");
      },
      async unload() {
        calls.push("unload");
      },
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-before-admitted-unload", "load", downloadPayload()));
    assert.equal((await response(scope, "load-before-admitted-unload")).ok, true);
    scope.messages.length = 0;

    scope.send(request("blocking-service-before-unload", "request", {
      service: "sessions",
      method: "get",
      args: [],
    }));
    await waitFor(() => serviceStarted);
    scope.send(request("generation-before-unload", "generate", { type: "generate" }));
    scope.send(request("unload-with-admitted-generation", "unload", undefined));

    releaseService({ id: "fixture-session" });
    const interruptedService = await response(scope, "blocking-service-before-unload");
    assert.equal(interruptedService.ok, false);
    assert.equal(interruptedService.error.code, "MODEL_REQUEST_CANCELLED");
    assert.equal((await response(scope, "generation-before-unload")).ok, true);
    assert.equal((await response(scope, "unload-with-admitted-generation")).ok, true);
    assert.equal(generateCalls, 0);
    assert.deepEqual(calls, ["unload"]);
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" &&
        message.payload.generation?.phase === "running",
    ), false);
  });

  test("worker never records performance from a stopped generation", async () => {
    let releaseGeneration;
    let generationStarted = false;
    let stopReason = null;
    const backend = {
      async load() {
        return { prefillTokensPerSecond: null, decodeTokensPerSecond: null };
      },
      async request() { return null; },
      async generate(_input, emit) {
        await emit({
          type: "started",
          generation_id: "stopped-generation",
          node_id: null,
          sibling_index: 0,
          sibling_count: 1,
        });
        generationStarted = true;
        await new Promise((resolve) => { releaseGeneration = resolve; });
        await emit({
          type: "token",
          text: "partial",
          thinking: false,
          token_id: 1,
          node_id: null,
        });
        await emit({
          type: "done",
          node_id: null,
          sibling_index: 0,
          sibling_count: 1,
          result: {
            text: "partial",
            tokens: 1,
            finish_reason: "stop",
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          },
        });
        return {
          prefillTokensPerSecond: 20,
          decodeTokensPerSecond: 8,
        };
      },
      async stop(reason) {
        stopReason = reason;
        releaseGeneration();
      },
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-before-stopped-generation", "load", downloadPayload()));
    assert.equal((await response(scope, "load-before-stopped-generation")).ok, true);
    assert.equal(scope.loadRecords.length, 1);

    scope.send(request("stopped-generation", "generate", {
      type: "generate",
      stateless: true,
    }));
    await waitFor(() => generationStarted);
    scope.send(request("stop-active-generation", "stop", undefined));
    assert.equal((await response(scope, "stop-active-generation")).ok, true);
    assert.equal((await response(scope, "stopped-generation")).ok, true);
    assert.equal(stopReason, "user");
    assert.equal(scope.loadRecords.length, 1);
    assert.equal(scope.loadRecords[0].prefillTokensPerSecond, null);
    assert.equal(scope.loadRecords[0].decodeTokensPerSecond, null);
  });

  test("worker never records performance from a failed generation", async () => {
    const backend = {
      async load() {
        return { prefillTokensPerSecond: null, decodeTokensPerSecond: null };
      },
      async request() { return null; },
      async generate(_input, emit) {
        await emit({
          type: "started",
          generation_id: "failed-generation",
          node_id: null,
          sibling_index: 0,
          sibling_count: 1,
        });
        throw new Error("fixture decode failed");
      },
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-before-failed-generation", "load", downloadPayload()));
    assert.equal((await response(scope, "load-before-failed-generation")).ok, true);
    assert.equal(scope.loadRecords.length, 1);

    scope.send(request("failed-generation", "generate", {
      type: "generate",
      stateless: true,
    }));
    const failed = await response(scope, "failed-generation");
    assert.equal(failed.ok, false);
    assert.match(failed.error.message, /fixture decode failed/);
    assert.equal(scope.loadRecords.length, 1);
    assert.equal(scope.loadRecords[0].prefillTokensPerSecond, null);
    assert.equal(scope.loadRecords[0].decodeTokensPerSecond, null);
  });

  test("queued generation admission is not blocked by a later tree mutation", async () => {
    let releaseBlockingRequest;
    let signalBlockingRequestStarted;
    let signalGenerationEntered;
    let signalMutationApplied;
    let revision = 0;
    const backendOrder = [];
    const blockingRequestStarted = new Promise((resolve) => {
      signalBlockingRequestStarted = resolve;
    });
    const generationEntered = new Promise((resolve) => {
      signalGenerationEntered = resolve;
    });
    const mutationApplied = new Promise((resolve) => {
      signalMutationApplied = resolve;
    });
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request(input, onProgress) {
        if (input.service === "sessions" && input.method === "get") {
          backendOrder.push("blocking-request");
          signalBlockingRequestStarted();
          return new Promise((resolve) => { releaseBlockingRequest = resolve; });
        }
        backendOrder.push("tree-mutation");
        assert.equal(input.service, "tree");
        assert.equal(input.method, "navigate");
        revision += 1;
        onProgress({
          event: "tree_mutated",
          data: {
            type: "tree_mutated",
            op: "navigate",
            rev: revision,
            added: [],
            removed: [],
            updated: [],
            active_node_id: String(input.args[0]),
            cast: {},
          },
        });
        signalMutationApplied();
        return { active_node_id: String(input.args[0]) };
      },
      async generate(_input, emit, onAdmitted) {
        backendOrder.push("generation");
        signalGenerationEntered();
        onAdmitted();
        await mutationApplied;
        await emit({
          type: "started",
          generation_id: "queued-generation",
          node_id: null,
          sibling_index: 0,
          sibling_count: 1,
        });
        revision += 1;
        await emit({ type: "tree_mutated", op: "begin_assistant", rev: revision });
        revision += 1;
        await emit({ type: "tree_mutated", op: "finalize_assistant", rev: revision });
        await emit({
          type: "done",
          node_id: "queued-assistant",
          sibling_index: 0,
          sibling_count: 1,
          result: {
            text: "complete",
            tokens: 0,
            finish_reason: "stop",
            usage: { prompt_tokens: 1, completion_tokens: 0, total_tokens: 1 },
          },
        });
      },
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-queued-mutation", "load", downloadPayload()));
    assert.equal((await response(scope, "load-queued-mutation")).ok, true);
    scope.messages.length = 0;

    scope.send(request("blocking-before-generation", "request", {
      service: "sessions",
      method: "get",
      args: [],
    }));
    await blockingRequestStarted;
    scope.send(request("generation-behind-request", "generate", { type: "generate" }));
    scope.send(request("mutation-behind-generation", "request", {
      service: "tree",
      method: "navigate",
      args: ["outside-node"],
    }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.deepEqual(backendOrder, ["blocking-request"]);
    releaseBlockingRequest({ id: "fixture-session" });

    assert.equal((await response(scope, "blocking-before-generation")).ok, true);
    await generationEntered;
    assert.equal((await response(scope, "mutation-behind-generation")).ok, true);
    assert.equal((await response(scope, "generation-behind-request")).ok, true);
    assert.deepEqual(backendOrder, ["blocking-request", "generation", "tree-mutation"]);
    const treeEvents = scope.messages.filter(
      (message) => message.kind === "event" && message.event === "tree_mutated",
    );
    assert.deepEqual(treeEvents.map((message) => message.payload.rev), [1, 2, 3]);
  });

  test("worker executes reservation-aware tree mutations while generation remains live", async () => {
    let releaseAdmission;
    let releaseGeneration;
    let signalGenerationEntered;
    let signalGenerationStarted;
    let revision = 0;
    let persists = 0;
    let reservationReady = false;
    const admissionGate = new Promise((resolve) => {
      releaseAdmission = resolve;
    });
    const generationEntered = new Promise((resolve) => {
      signalGenerationEntered = resolve;
    });
    const generationStarted = new Promise((resolve) => {
      signalGenerationStarted = resolve;
    });
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request(input, onProgress) {
        assert.equal(reservationReady, true);
        if (
          input.method === "reset" || input.method === "restore" ||
          input.method === "transcriptLoad" ||
          ((input.method === "edit" || input.method === "delete") &&
            String(input.args[0]).startsWith("reserved"))
        ) {
          throw Object.assign(new Error("fixture reservation conflict"), {
            code: "MUTATION_DURING_GENERATION",
            status: 409,
            recoverable: true,
          });
        }
        revision += 1;
        onProgress({
          event: "tree_mutated",
          data: {
            type: "tree_mutated",
            op: input.method,
            rev: revision,
            added: input.method === "branch" ? [{ id: "outside-branch" }] : [],
            removed: input.method === "delete" ? [String(input.args[0])] : [],
            updated: ["star", "note", "edit"].includes(input.method)
              ? [{ id: String(input.args[0]) }]
              : [],
            active_node_id: input.method === "navigate" ? String(input.args[0]) : null,
            cast: {},
          },
        });
        return { method: input.method };
      },
      async generate(_input, emit, onAdmitted) {
        signalGenerationEntered();
        await admissionGate;
        reservationReady = true;
        onAdmitted();
        await emit({
          type: "started",
          generation_id: "concurrent-tree-generation",
          node_id: null,
          sibling_index: 0,
          sibling_count: 1,
        });
        revision += 1;
        await emit({
          type: "tree_mutated",
          op: "begin_assistant",
          rev: revision,
        });
        signalGenerationStarted();
        await new Promise((resolve) => { releaseGeneration = resolve; });
        revision += 1;
        await emit({
          type: "tree_mutated",
          op: "finalize_assistant",
          rev: revision,
        });
        await emit({
          type: "done",
          node_id: "reserved-assistant",
          sibling_index: 0,
          sibling_count: 1,
          result: {
            text: "complete",
            tokens: 1,
            finish_reason: "stop",
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          },
        });
      },
      async stop() { releaseGeneration?.(); },
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: backend,
      sessionCoordinatorFactory: () => ({
        async restore() { return { status: "not_found" }; },
        async persist() {
          persists += 1;
          scope.timeline.push(`session:persist:${persists}`);
        },
        async close() {},
      }),
    });
    await prepareDownload(scope);
    scope.send(request("load-concurrent-tree", "load", downloadPayload()));
    assert.equal((await response(scope, "load-concurrent-tree")).ok, true);
    scope.messages.length = 0;
    scope.timeline.length = 0;

    scope.send(request("live-tree-generation", "generate", { type: "generate" }));
    await generationEntered;
    scope.send(request("admission-window-navigate", "request", {
      service: "tree",
      method: "navigate",
      args: ["outside-node"],
    }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(scope.messages.some(
      (message) => message.kind === "response" &&
        message.requestId === "admission-window-navigate",
    ), false);
    releaseAdmission();
    assert.equal((await response(scope, "admission-window-navigate")).ok, true);
    await generationStarted;
    for (const [requestId, method, args] of [
      ["live-navigate", "navigate", ["outside-node"]],
      ["live-branch", "branch", ["reserved-assistant", "safe sibling"]],
      ["live-star", "star", ["reserved-assistant", true]],
      ["live-note", "note", ["reserved-assistant", "keep"]],
      ["live-edit-outside", "edit", ["outside-node", "edited"]],
      ["live-delete-outside", "delete", ["outside-delete"]],
    ]) {
      scope.send(request(requestId, "request", { service: "tree", method, args }));
      assert.equal((await response(scope, requestId)).ok, true);
      assert.equal(scope.messages.some(
        (message) => message.kind === "response" &&
          message.requestId === "live-tree-generation",
      ), false);
      const eventIndex = scope.timeline.lastIndexOf("event:tree_mutated");
      const persistIndex = scope.timeline.lastIndexOf(`session:persist:${persists}`);
      const responseIndex = scope.timeline.lastIndexOf(`response:${requestId}:ok`);
      assert.ok(eventIndex < persistIndex && persistIndex < responseIndex);
    }
    for (const [requestId, method, args] of [
      ["live-edit-reserved", "edit", ["reserved-assistant", "blocked"]],
      ["live-delete-reserved", "delete", ["reserved-root"]],
      ["live-restore", "restore", [{}]],
      ["live-reset", "reset", []],
      ["live-transcript-restore", "transcriptLoad", ["", "default", true]],
    ]) {
      scope.send(request(requestId, "request", { service: "tree", method, args }));
      const rejected = await response(scope, requestId);
      assert.equal(rejected.ok, false);
      assert.equal(rejected.error.code, "MUTATION_DURING_GENERATION");
      assert.equal(rejected.error.status, 409);
    }

    releaseGeneration();
    assert.equal((await response(scope, "live-tree-generation")).ok, true);
    const treeEvents = scope.messages.filter(
      (message) => message.kind === "event" && message.event === "tree_mutated",
    );
    assert.deepEqual(
      treeEvents.map((message) => message.payload.rev),
      treeEvents.map((message) => message.payload.rev).toSorted((left, right) => left - right),
    );
    assert.equal(new Set(treeEvents.map((message) => message.payload.rev)).size, treeEvents.length);
    assert.equal(treeEvents.at(-1).payload.op, "finalize_assistant");
  });

  test("worker rejects tree mutations before started without persisting or emitting them", async () => {
    let stopCalls = 0;
    let persists = 0;
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate(_input, emit) {
        emit({ type: "tree_mutated", op: "begin_assistant", rev: 1 });
      },
      async stop() { stopCalls += 1; },
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: backend,
      sessionCoordinatorFactory: () => ({
        async restore() { return { status: "not_found" }; },
        async persist() { persists += 1; },
        async close() {},
      }),
    });
    await prepareDownload(scope);
    scope.send(request("load-protocol", "load", downloadPayload()));
    assert.equal((await response(scope, "load-protocol")).ok, true);
    scope.messages.length = 0;

    scope.send(request("invalid-generation", "generate", { type: "generate" }));
    const rejected = await response(scope, "invalid-generation");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "GENERATION_PROTOCOL_VIOLATION");
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" && message.event === "tree_mutated",
    ), false);
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" && message.event === "error",
    ), false);
    assert.equal(persists, 0);
    assert.equal(stopCalls, 1);
  });

  test("worker accepts stateless started-token-done streams without tree mutations", async () => {
    let persists = 0;
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate(input, emit) {
        assert.equal(input.stateless, true);
        emit({
          type: "started",
          generation_id: "stateless-generation",
          node_id: null,
          sibling_index: 0,
          sibling_count: 1,
        });
        emit({
          type: "token",
          text: "local",
          thinking: false,
          token_id: 1,
          node_id: "existing-node",
        });
        emit({
          type: "done",
          node_id: "existing-node",
          sibling_index: 0,
          sibling_count: 1,
          result: {
            text: "local",
            tokens: 1,
            finish_reason: "stop",
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          },
        });
      },
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: backend,
      sessionCoordinatorFactory: () => ({
        async restore() { return { status: "not_found" }; },
        async persist() { persists += 1; },
        async close() {},
      }),
    });
    await prepareDownload(scope);
    scope.send(request("load-stateless", "load", downloadPayload()));
    assert.equal((await response(scope, "load-stateless")).ok, true);
    scope.messages.length = 0;

    scope.send(request("stateless-generation", "generate", {
      type: "generate",
      stateless: true,
    }));
    assert.equal((await response(scope, "stateless-generation")).ok, true);
    const events = scope.messages.filter((message) => message.kind === "event");
    assert.deepEqual(events.map((message) => message.event), [
      "status", "started", "token", "done", "status",
    ]);
    assert.deepEqual(events.slice(1, 4).map((message) => message.generationId), [
      "stateless-generation", "stateless-generation", "stateless-generation",
    ]);
    assert.equal(persists, 0);
  });

  test("worker requires a tree mutation before stateful done", async () => {
    let persists = 0;
    let stopCalls = 0;
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate(_input, emit) {
        emit({
          type: "started",
          generation_id: "stateful-without-tree",
          node_id: null,
          sibling_index: 0,
          sibling_count: 1,
        });
        emit({
          type: "token",
          text: "uncommitted",
          thinking: false,
          token_id: 1,
          node_id: "node-1",
        });
        emit({
          type: "done",
          node_id: "node-1",
          sibling_index: 0,
          sibling_count: 1,
          result: {
            text: "uncommitted",
            tokens: 1,
            finish_reason: "stop",
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          },
        });
      },
      async stop() { stopCalls += 1; },
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: backend,
      sessionCoordinatorFactory: () => ({
        async restore() { return { status: "not_found" }; },
        async persist() { persists += 1; },
        async close() {},
      }),
    });
    await prepareDownload(scope);
    scope.send(request("load-stateful-no-tree", "load", downloadPayload()));
    assert.equal((await response(scope, "load-stateful-no-tree")).ok, true);
    scope.messages.length = 0;

    scope.send(request("stateful-no-tree", "generate", { type: "generate" }));
    const rejected = await response(scope, "stateful-no-tree");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "GENERATION_PROTOCOL_VIOLATION");
    assert.deepEqual(
      scope.messages
        .filter((message) => message.kind === "event")
        .map((message) => message.event),
      ["status", "started", "token", "status"],
    );
    assert.equal(persists, 0);
    assert.equal(stopCalls, 1);
  });

  test("worker rejects a done node that differs from the streamed token node", async () => {
    let stopCalls = 0;
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate(_input, emit) {
        emit({
          type: "started",
          generation_id: "mismatched-node",
          node_id: null,
          sibling_index: 0,
          sibling_count: 1,
        });
        emit({
          type: "token",
          text: "local",
          thinking: false,
          token_id: 1,
          node_id: "node-1",
        });
        emit({
          type: "done",
          node_id: "node-2",
          sibling_index: 0,
          sibling_count: 1,
          result: {
            text: "local",
            tokens: 1,
            finish_reason: "stop",
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          },
        });
      },
      async stop() { stopCalls += 1; },
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-node-mismatch", "load", downloadPayload()));
    assert.equal((await response(scope, "load-node-mismatch")).ok, true);
    scope.messages.length = 0;

    scope.send(request("node-mismatch", "generate", {
      type: "generate",
      stateless: true,
    }));
    const rejected = await response(scope, "node-mismatch");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "GENERATION_PROTOCOL_VIOLATION");
    assert.equal(scope.messages.filter(
      (message) => message.kind === "event" && message.event === "token",
    ).length, 1);
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" && message.event === "done",
    ), false);
    assert.equal(stopCalls, 1);
  });

  test("backend generation errors are terminal and delivered only by the response", async () => {
    let stopCalls = 0;
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate(_input, emit) {
        emit({
          type: "started",
          generation_id: "generation-error",
          node_id: null,
          sibling_index: 0,
          sibling_count: 1,
        });
        emit({ type: "tree_mutated", op: "begin_assistant", rev: 1 });
        emit({ type: "error", message: "backend rejected", code: "BACKEND_REJECTED" });
        emit({
          type: "token",
          text: "must not escape",
          thinking: false,
          token_id: 1,
          node_id: "node-error",
        });
      },
      async stop() { stopCalls += 1; },
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: backend,
      sessionCoordinatorFactory: () => ({
        async restore() { return { status: "not_found" }; },
        async persist() {},
        async close() {},
      }),
    });
    await prepareDownload(scope);
    scope.send(request("load-backend-error", "load", downloadPayload()));
    assert.equal((await response(scope, "load-backend-error")).ok, true);
    scope.messages.length = 0;

    scope.send(request("backend-error", "generate", { type: "generate" }));
    const rejected = await response(scope, "backend-error");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "BACKEND_REJECTED");
    assert.deepEqual(
      scope.messages
        .filter((message) => message.kind === "event")
        .map((message) => message.event),
      ["status", "started", "tree_mutated", "status"],
    );
    assert.equal(stopCalls, 1);
  });

  test("worker accepts complete ordered multi-sibling generation sequences", async () => {
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate(_input, emit) {
        for (let index = 0; index < 2; index += 1) {
          const nodeId = `node-${index}`;
          await emit({
            type: "started",
            generation_id: `generation-${index}`,
            node_id: null,
            sibling_index: index,
            sibling_count: 2,
          });
          await emit({ type: "tree_mutated", op: "begin_assistant", rev: index * 2 + 1 });
          await emit({
            type: "token",
            text: String(index),
            thinking: false,
            token_id: index,
            node_id: nodeId,
          });
          await emit({ type: "tree_mutated", op: "finalize", rev: index * 2 + 2 });
          await emit({
            type: "done",
            node_id: nodeId,
            sibling_index: index,
            sibling_count: 2,
            result: {
              text: String(index),
              tokens: 1,
              finish_reason: "stop",
              usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
            },
          });
        }
      },
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-multi", "load", downloadPayload()));
    assert.equal((await response(scope, "load-multi")).ok, true);
    scope.messages.length = 0;

    scope.send(request("multi-generation", "generate", { type: "generate", n: 2 }));
    assert.equal((await response(scope, "multi-generation")).ok, true);
    assert.deepEqual(
      scope.messages
        .filter((message) => message.kind === "event")
        .map((message) => message.event),
      [
        "status",
        "started", "tree_mutated", "token", "tree_mutated", "done",
        "started", "tree_mutated", "token", "tree_mutated", "done",
        "status",
      ],
    );
  });

  test("worker rejects a reused generation id after its terminal done", async () => {
    let stopCalls = 0;
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate(_input, emit) {
        await emit({
          type: "started",
          generation_id: "reused-generation",
          node_id: null,
          sibling_index: 0,
          sibling_count: 2,
        });
        await emit({ type: "tree_mutated", op: "begin_assistant", rev: 1 });
        await emit({ type: "tree_mutated", op: "finalize", rev: 2 });
        await emit({
          type: "done",
          node_id: "node-0",
          sibling_index: 0,
          sibling_count: 2,
          result: {
            text: "",
            tokens: 0,
            finish_reason: "stop",
            usage: { prompt_tokens: 1, completion_tokens: 0, total_tokens: 1 },
          },
        });
        await emit({
          type: "started",
          generation_id: "reused-generation",
          node_id: null,
          sibling_index: 1,
          sibling_count: 2,
        });
        await emit({ type: "tree_mutated", op: "begin_assistant", rev: 2 });
      },
      async stop() { stopCalls += 1; },
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-reused", "load", downloadPayload()));
    assert.equal((await response(scope, "load-reused")).ok, true);
    scope.messages.length = 0;

    scope.send(request("reused-id", "generate", { type: "generate", n: 2 }));
    const rejected = await response(scope, "reused-id");
    assert.equal(rejected.error.code, "GENERATION_PROTOCOL_VIOLATION");
    assert.equal(scope.messages.filter(
      (message) => message.kind === "event" && message.event === "started",
    ).length, 1);
    assert.equal(scope.messages.filter(
      (message) => message.kind === "event" && message.event === "done",
    ).length, 1);
    assert.equal(stopCalls, 1);
  });

  test("worker requires commit-only submits to preserve the full event sequence", async () => {
    let persists = 0;
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate(_input, emit) {
        await emit({
          type: "started",
          generation_id: "commit-generation",
          node_id: null,
          sibling_index: 0,
          sibling_count: 1,
        });
        await emit({ type: "tree_mutated", op: "add_user", rev: 1 });
        await emit({
          type: "done",
          node_id: "commit-node",
          sibling_index: 0,
          sibling_count: 1,
          result: {
            text: "local note",
            tokens: 0,
            finish_reason: "stop",
            usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
          },
        });
      },
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: backend,
      sessionCoordinatorFactory: () => ({
        async restore() { return { status: "not_found" }; },
        async persist() { persists += 1; },
        async close() {},
      }),
    });
    await prepareDownload(scope);
    scope.send(request("load-commit", "load", downloadPayload()));
    assert.equal((await response(scope, "load-commit")).ok, true);
    scope.messages.length = 0;

    scope.send(request("commit-only", "submit", {
      type: "submit",
      text: "local note",
      authored_role: "user",
      generated_role: null,
    }));
    assert.equal((await response(scope, "commit-only")).ok, true);
    assert.equal(persists, 1);
    assert.deepEqual(
      scope.messages
        .filter((message) => message.kind === "event")
        .map((message) => message.event),
      ["status", "started", "tree_mutated", "done", "status"],
    );
  });

  test("worker preserves an incompatible local session and does not blame hardware", async () => {
    let unloads = 0;
    let closes = 0;
    const restoreModes = [];
    const backend = {
      async load() {
        return { prefillTokensPerSecond: 10, decodeTokensPerSecond: 5 };
      },
      async request() { throw new Error("not reached"); },
      async generate() { throw new Error("not reached"); },
      async stop() {},
      async unload() { unloads += 1; },
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: backend,
      sessionCoordinatorFactory: () => ({
        async restore(_identity, mode) {
          restoreModes.push(mode);
          if (mode === "reset") return { status: "not_found" };
          return {
            status: "incompatible",
            reason: "runtime_identity",
            preserved: true,
          };
        },
        async persist() {},
        async close() { closes += 1; },
      }),
    });
    await prepareDownload(scope);
    scope.send(request("load-incompatible", "load", downloadPayload()));
    const rejected = await response(scope, "load-incompatible");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "LOCAL_SESSION_INCOMPATIBLE");
    assert.equal(unloads, 1);
    assert.equal(closes, 1);
    assert.deepEqual(scope.loadRecords, []);

    scope.send(request("load-reset", "load", {
      ...downloadPayload(),
      resetSession: true,
    }));
    assert.equal((await response(scope, "load-reset")).ok, true);
    assert.deepEqual(restoreModes, ["replace", "reset"]);
    assert.deepEqual(scope.loadRecords.map((record) => record.result), ["success"]);
    scope.send(request("unload-reset", "unload", undefined));
    assert.equal((await response(scope, "unload-reset")).ok, true);
    assert.equal(unloads, 2);
    assert.equal(closes, 2);
  });

  test("worker cleanly reloads once after a ready-time non-OOM device loss", async () => {
    const recoveryCatalog = structuredClone(verifiedCatalog);
    const recoveryVariant = recoveryCatalog.document.models[0].variants[0];
    const recoveryPacks = [
      ["recovery-jlens", "jlens", "b".repeat(64)],
      ["recovery-sae", "sae", "c".repeat(64)],
    ].map(([id, kind, sha256]) => ({
      id,
      kind,
      displayName: id,
      ...optionalPackProvenance,
      bytes: 64,
      required: false,
      runtimeIdentitySha256: recoveryVariant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [recoveryVariant.contextProfiles[0].bindingSha256],
      files: [{
        path: `packs/${id}.safetensors`,
        role: "instrument",
        url: recoveryVariant.files[0].url.replace(
          "weights/params.bin",
          `packs/${id}.safetensors`,
        ),
        revision: recoveryVariant.files[0].revision,
        bytes: 64,
        sha256,
      }],
    }));
    recoveryVariant.packs.push(...recoveryPacks);
    const recoveryManifests = [
      ...recoveryVariant.files,
      ...recoveryVariant.packs.find((pack) => pack.kind === "core").files,
    ];
    const recoveryCapabilities = compatibleCapabilities();
    recoveryCapabilities.webGpu.limits.maxBufferSize = 512 * 1024 * 1024;
    recoveryCapabilities.webGpu.limits.maxStorageBufferBindingSize = 512 * 1024 * 1024;
    const deviceLossCallbacks = [];
    const loadedPackIds = [];
    const loadedArtifactFiles = [];
    const verifiedFileCalls = new Map();
    let capabilityChecks = 0;
    let adapterClears = 0;
    let loads = 0;
    let unloads = 0;
    let restores = 0;
    const backend = {
      async load(input) {
        loads += 1;
        deviceLossCallbacks.push(input.onDeviceLost);
        loadedPackIds.push(input.optionalPacks.map(({ pack }) => pack.id));
        loadedArtifactFiles.push([
          ...input.artifacts.map(({ file }) => file),
          ...input.optionalPacks.flatMap(({ artifacts }) =>
            artifacts.map(({ file }) => file)
          ),
        ]);
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate() {},
      async stop() {},
      async unload() { unloads += 1; },
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      catalog: recoveryCatalog,
      capabilities: recoveryCapabilities,
      installs: [
        {
          id: recoveryVariant.id,
          kind: "model",
          objectHashes: recoveryManifests.map((file) => file.sha256),
          installedAt: 1,
        },
        ...recoveryPacks.map((pack, index) => ({
          id: pack.id,
          kind: "pack",
          objectHashes: pack.files.map((file) => file.sha256),
          installedAt: index + 2,
        })),
      ],
      verifiedFile(sha256) {
        verifiedFileCalls.set(sha256, (verifiedFileCalls.get(sha256) ?? 0) + 1);
        const file = [...recoveryManifests, ...recoveryPacks.flatMap((pack) => pack.files)]
          .find((candidate) => candidate.sha256 === sha256);
        return file ? new Blob([new Uint8Array(file.bytes)]) : null;
      },
      modelBackend: backend,
      onCapabilityCheck() { capabilityChecks += 1; },
      onClearAdapter() { adapterClears += 1; },
      sessionCoordinatorFactory: () => ({
        async restore() {
          restores += 1;
          return restores === 1
            ? { status: "not_found" }
            : { status: "restored", session: {}, tree: {}, restoredConfigKeys: [] };
        },
        async persist() {},
        async close() {},
      }),
    });
    await prepareDownload(scope);
    scope.send(request("load-ready-loss", "load", downloadPayload()));
    assert.equal((await response(scope, "load-ready-loss")).ok, true);
    scope.messages.length = 0;

    deviceLossCallbacks[0]({
      code: "GPU_DEVICE_LOST",
      message: "fixture adapter reset",
      confirmedOom: false,
    });
    await waitFor(() => loads === 2 && scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    )?.payload.lifecycle === "ready");

    assert.equal(capabilityChecks, 2);
    assert.equal(adapterClears, 1);
    assert.equal(unloads, 1);
    assert.equal(restores, 2);
    assert.deepEqual(loadedPackIds, [
      ["recovery-jlens", "recovery-sae"],
      ["recovery-jlens", "recovery-sae"],
    ]);
    assert.deepEqual(
      [...verifiedFileCalls.values()],
      Array(verifiedFileCalls.size).fill(2),
    );
    assert.equal(loadedArtifactFiles[0].length, loadedArtifactFiles[1].length);
    for (let index = 0; index < loadedArtifactFiles[0].length; index += 1) {
      assert.notStrictEqual(loadedArtifactFiles[0][index], loadedArtifactFiles[1][index]);
    }
    assert.deepEqual(
      scope.loadRecords.map((record) => record.result),
      ["success", "device_lost", "success"],
    );
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" &&
        (message.event === "device_lost" || message.event === "error"),
    ), false);
    assert.deepEqual(
      scope.messages
        .filter((message) => message.kind === "event" && message.event === "status")
        .map((message) => message.payload.lifecycle),
      ["loading", "loading", "ready"],
    );
  });

  test("device-loss recovery rejects a changed accepted artifact binding", async () => {
    const changedCatalog = structuredClone(verifiedCatalog);
    changedCatalog.document.models[0].variants[0].files[0].sha256 = "d".repeat(64);
    let catalogReads = 0;
    let loads = 0;
    let deviceLoss;
    const backend = {
      async load(input) {
        loads += 1;
        deviceLoss = input.onDeviceLost;
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      catalogGet() {
        catalogReads += 1;
        return catalogReads === 1 ? verifiedCatalog : changedCatalog;
      },
      modelBackend: backend,
    });
    await prepareDownload(scope);
    scope.send(request("load-before-catalog-change", "load", downloadPayload()));
    assert.equal((await response(scope, "load-before-catalog-change")).ok, true);
    scope.send(request("replace-catalog-before-recovery", "catalog", { refresh: true }));
    assert.equal((await response(scope, "replace-catalog-before-recovery")).ok, true);
    scope.messages.length = 0;

    deviceLoss({
      code: "GPU_DEVICE_LOST",
      message: "fixture adapter reset",
      confirmedOom: false,
    });
    const terminal = await waitFor(() => scope.messages.find((message) =>
      message.kind === "event" && message.event === "error" &&
      message.payload?.scope === "lifecycle"
    ));
    assert.equal(terminal.payload.failure.code, "MODEL_RECOVERY_BINDING_CHANGED");
    assert.equal(loads, 1);
  });

  test("device-loss recovery rejects an install removed after initial load", async () => {
    let loads = 0;
    let deviceLoss;
    const backend = {
      async load(input) {
        loads += 1;
        deviceLoss = input.onDeviceLost;
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-before-install-removal", "load", downloadPayload()));
    assert.equal((await response(scope, "load-before-install-removal")).ok, true);
    scope.installs.length = 0;
    scope.messages.length = 0;

    deviceLoss({
      code: "GPU_DEVICE_LOST",
      message: "fixture adapter reset",
      confirmedOom: false,
    });
    const terminal = await waitFor(() => scope.messages.find((message) =>
      message.kind === "event" && message.event === "error" &&
      message.payload?.scope === "lifecycle"
    ));
    assert.equal(terminal.payload.failure.code, "MODEL_NOT_INSTALLED");
    assert.equal(loads, 1);
  });

  test("device-loss recovery fails closed when a verified OPFS object was evicted", async () => {
    const reads = new Map();
    const missingHash = variant.files[0].sha256;
    let loads = 0;
    let deviceLoss;
    const backend = {
      async load(input) {
        loads += 1;
        deviceLoss = input.onDeviceLost;
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: backend,
      verifiedFile(sha256, { manifests }) {
        const count = (reads.get(sha256) ?? 0) + 1;
        reads.set(sha256, count);
        if (sha256 === missingHash && count > 1) return null;
        const manifest = manifests.find((file) => file.sha256 === sha256);
        return manifest ? new Blob([new Uint8Array(manifest.bytes)]) : null;
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-before-object-eviction", "load", downloadPayload()));
    assert.equal((await response(scope, "load-before-object-eviction")).ok, true);
    scope.messages.length = 0;

    deviceLoss({
      code: "GPU_DEVICE_LOST",
      message: "fixture adapter reset",
      confirmedOom: false,
    });
    const terminal = await waitFor(() => scope.messages.find((message) =>
      message.kind === "event" && message.event === "error" &&
      message.payload?.scope === "lifecycle"
    ));
    assert.equal(terminal.payload.failure.code, "VERIFIED_MODEL_OBJECT_MISSING");
    assert.equal(loads, 1);
    assert.equal(reads.get(missingHash), 2);
  });

  test("worker escalates unconfirmed replacement cleanup before publishing device loss", async () => {
    const deviceLossCallbacks = [];
    let loads = 0;
    let unloads = 0;
    let coordinators = 0;
    const backend = {
      async load(input) {
        loads += 1;
        deviceLossCallbacks.push(input.onDeviceLost);
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate() {},
      async stop() {},
      async unload() {
        unloads += 1;
        if (unloads === 2) throw new Error("replacement backend still owns resources");
      },
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: backend,
      sessionCoordinatorFactory: () => {
        coordinators += 1;
        const coordinator = coordinators;
        return {
          async restore() {
            if (coordinator === 2) throw new Error("replacement restore failed");
            return { status: "not_found" };
          },
          async persist() {},
          async close() {
            if (coordinator === 2) throw new Error("replacement session close failed");
          },
        };
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-before-cleanup-fatal-recovery", "load", downloadPayload()));
    assert.equal((await response(scope, "load-before-cleanup-fatal-recovery")).ok, true);
    scope.messages.length = 0;

    deviceLossCallbacks[0]({
      code: "GPU_DEVICE_LOST",
      message: "fixture adapter reset",
      confirmedOom: false,
    });
    const fatal = await waitFor(() => scope.messages.find(
      (message) => message.kind === "event" &&
        message.event === "error" &&
        message.payload.failure?.code === "RUNTIME_CLEANUP_FAILED",
    ));

    assert.equal(loads, 2);
    assert.equal(unloads, 2);
    assert.match(fatal.payload.failure.message, /session cleanup failed/);
    assert.match(fatal.payload.failure.message, /backend disposal failed/);
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" && message.event === "device_lost",
    ), false);
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" &&
        message.event === "status" &&
        message.payload.lifecycle === "failed",
    ), false);
  });

  test("worker aborts and joins an ordinary backend request before device-loss recovery", async () => {
    const deviceLossCallbacks = [];
    let releaseRequest;
    const requestGate = new Promise((resolve) => { releaseRequest = resolve; });
    let requestSignal;
    let requestStarted = false;
    let loads = 0;
    let unloads = 0;
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: {
        async load(input) {
          loads += 1;
          deviceLossCallbacks.push(input.onDeviceLost);
          return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
        },
        async request(_input, _progress, signal) {
          requestStarted = true;
          requestSignal = signal;
          await requestGate;
          return { stale: true };
        },
        async generate() {},
        async stop() {},
        async unload() { unloads += 1; },
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-before-request-loss", "load", downloadPayload()));
    assert.equal((await response(scope, "load-before-request-loss")).ok, true);
    scope.messages.length = 0;

    scope.send(request("request-before-loss", "request", {
      service: "sessions",
      method: "get",
      args: [],
    }));
    await waitFor(() => requestStarted);
    deviceLossCallbacks[0]({
      code: "GPU_DEVICE_LOST",
      message: "fixture loss during request",
      confirmedOom: false,
    });
    await waitFor(() => requestSignal?.aborted);
    assert.equal(unloads, 0);
    assert.equal(loads, 1);

    releaseRequest();
    const staleRequest = await response(scope, "request-before-loss");
    assert.equal(staleRequest.ok, false);
    assert.equal(staleRequest.error.code, "MODEL_REQUEST_CANCELLED");
    await waitFor(() => loads === 2 && scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    )?.payload.lifecycle === "ready");
    assert.equal(unloads, 1);
  });

  test("worker rejects a public recheck while device-loss recovery owns the adapter", async () => {
    const deviceLossCallbacks = [];
    let capabilityChecks = 0;
    let releaseRecoveryCheck;
    let reportRecoveryCheck;
    const recoveryCheckStarted = new Promise((resolve) => {
      reportRecoveryCheck = resolve;
    });
    const recoveryCheckRelease = new Promise((resolve) => {
      releaseRecoveryCheck = resolve;
    });
    const backend = {
      async load(input) {
        deviceLossCallbacks.push(input.onDeviceLost);
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: backend,
      async capabilityCheck() {
        capabilityChecks += 1;
        if (capabilityChecks === 2) {
          reportRecoveryCheck();
          await recoveryCheckRelease;
        }
        return compatibleCapabilities();
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-before-held-recovery", "load", downloadPayload()));
    assert.equal((await response(scope, "load-before-held-recovery")).ok, true);
    scope.messages.length = 0;

    deviceLossCallbacks[0]({
      code: "GPU_DEVICE_LOST",
      message: "fixture adapter reset",
      confirmedOom: false,
    });
    await recoveryCheckStarted;
    scope.send(request("check-during-recovery", "check", undefined));
    const rejected = await response(scope, "check-during-recovery");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "RUNTIME_IN_USE");
    assert.equal(capabilityChecks, 2);

    releaseRecoveryCheck();
    await waitFor(() => scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    )?.payload.lifecycle === "ready");
    assert.equal(capabilityChecks, 2);
  });

  test("worker serializes unload with recovery and rejects a competing model load", async () => {
    const deviceLossCallbacks = [];
    let loadCalls = 0;
    let unloadCalls = 0;
    let releaseRecoveryLoad;
    let reportRecoveryLoad;
    const recoveryLoadStarted = new Promise((resolve) => {
      reportRecoveryLoad = resolve;
    });
    const recoveryLoadRelease = new Promise((resolve) => {
      releaseRecoveryLoad = resolve;
    });
    const backend = {
      async load(input) {
        loadCalls += 1;
        deviceLossCallbacks.push(input.onDeviceLost);
        if (loadCalls === 2) {
          reportRecoveryLoad();
          await recoveryLoadRelease;
        }
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate() {},
      async stop() {},
      async unload() { unloadCalls += 1; },
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-before-recovery-race", "load", downloadPayload()));
    assert.equal((await response(scope, "load-before-recovery-race")).ok, true);
    scope.messages.length = 0;

    deviceLossCallbacks[0]({
      code: "GPU_DEVICE_LOST",
      message: "fixture adapter reset",
      confirmedOom: false,
    });
    await recoveryLoadStarted;
    scope.send(request("load-during-recovery", "load", downloadPayload()));
    const rejected = await response(scope, "load-during-recovery");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "RUNTIME_IN_USE");
    assert.equal(loadCalls, 2);

    scope.send(request("unload-during-recovery", "unload", undefined));
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(scope.messages.some(
      (message) => message.kind === "response" &&
        message.requestId === "unload-during-recovery",
    ), false);
    releaseRecoveryLoad();
    assert.equal((await response(scope, "unload-during-recovery")).ok, true);
    assert.equal(loadCalls, 2);
    assert.equal(unloadCalls, 3);
    assert.equal(scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload.lifecycle, "unloaded");
  });

  test("worker settles a lost generation before clean reload", async () => {
    const deviceLossCallbacks = [];
    let rejectGeneration;
    let loads = 0;
    let stops = 0;
    let unloads = 0;
    const backend = {
      async load(input) {
        loads += 1;
        deviceLossCallbacks.push(input.onDeviceLost);
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate(_input, emit) {
        await emit({
          type: "started",
          generation_id: "lost-generation",
          node_id: null,
          sibling_index: 0,
          sibling_count: 1,
        });
        return new Promise((_resolve, reject) => { rejectGeneration = reject; });
      },
      async stop() {
        stops += 1;
        rejectGeneration(new Error("backend device was lost"));
      },
      async unload() { unloads += 1; },
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-generation-loss", "load", downloadPayload()));
    assert.equal((await response(scope, "load-generation-loss")).ok, true);
    scope.messages.length = 0;

    scope.send(request("generation-device-loss", "generate", {
      type: "generate",
      stateless: true,
    }));
    await waitFor(() => rejectGeneration);
    deviceLossCallbacks[0]({
      code: "GPU_DEVICE_LOST",
      message: "fixture device disappeared during decode",
      confirmedOom: false,
    });
    const generation = await response(scope, "generation-device-loss");
    assert.equal(generation.ok, false);
    assert.equal(generation.error.code, "GPU_DEVICE_LOST");
    await waitFor(() => loads === 2 && scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    )?.payload.lifecycle === "ready");

    assert.equal(stops, 1);
    assert.equal(unloads, 1);
    assert.equal(scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload.generation.phase, "failed");
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" && message.event === "device_lost",
    ), false);
  });

  test("worker never retries a confirmed device-loss OOM", async () => {
    const deviceLossCallbacks = [];
    let capabilityChecks = 0;
    let loads = 0;
    const backend = {
      async load(input) {
        loads += 1;
        deviceLossCallbacks.push(input.onDeviceLost);
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: backend,
      onCapabilityCheck() { capabilityChecks += 1; },
    });
    await prepareDownload(scope);
    scope.send(request("load-before-oom", "load", downloadPayload()));
    assert.equal((await response(scope, "load-before-oom")).ok, true);
    scope.messages.length = 0;

    deviceLossCallbacks[0]({
      code: "WEBGPU_OUT_OF_MEMORY",
      message: "fixture allocation exceeded the device limit",
      confirmedOom: true,
    });
    const deviceLost = await waitFor(() => scope.messages.find(
      (message) => message.kind === "event" && message.event === "device_lost",
    ));

    assert.equal(loads, 1);
    assert.equal(capabilityChecks, 1);
    assert.equal(deviceLost.payload.code, "WEBGPU_OUT_OF_MEMORY");
    assert.deepEqual(
      scope.loadRecords.map((record) => record.result),
      ["success", "oom"],
    );
    assert.equal(scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload.lifecycle, "failed");
    assert.equal(scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "error",
    ).payload.scope, "lifecycle");
  });

  test("worker blocks load and deletion until confirmed-OOM cleanup settles", async () => {
    const deviceLossCallbacks = [];
    const order = [];
    let loads = 0;
    let reportOomRecord;
    let releaseOomRecord;
    const oomRecordStarted = new Promise((resolve) => { reportOomRecord = resolve; });
    const oomRecordRelease = new Promise((resolve) => { releaseOomRecord = resolve; });
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      async recordLoad(record, loadRecords) {
        loadRecords.push(record);
        if (record.result === "oom") {
          reportOomRecord();
          await oomRecordRelease;
        }
      },
      modelBackend: {
        async load(input) {
          loads += 1;
          deviceLossCallbacks.push(input.onDeviceLost);
          return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
        },
        async request() { return null; },
        async generate() {},
        async stop() {},
        async unload() { order.push("backend:unload"); },
      },
      sessionCoordinatorFactory: () => ({
        async restore() { return { status: "not_found" }; },
        async persist() {},
        async close() { order.push("session:close"); },
      }),
      sessionPersistence: {
        async claim() { return { status: "missing" }; },
        async save(input, ownerEpoch) {
          return { ...input, schemaVersion: 2, ownerEpoch, updatedAt: Date.now() };
        },
        async delete() {
          order.push("session:delete");
          return true;
        },
        async clear() {},
        close() {},
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-before-held-oom", "load", downloadPayload()));
    assert.equal((await response(scope, "load-before-held-oom")).ok, true);
    scope.messages.length = 0;

    deviceLossCallbacks[0]({
      code: "WEBGPU_OUT_OF_MEMORY",
      message: "fixture OOM with delayed persistence",
      confirmedOom: true,
    });
    await oomRecordStarted;
    assert.equal(scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload.lifecycle, "loading");
    scope.send(request("load-during-held-oom", "load", {
      ...downloadPayload(),
      explicitUnsafeOverride: true,
    }));
    const rejected = await response(scope, "load-during-held-oom");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "RUNTIME_IN_USE");
    assert.equal(loads, 1);

    scope.send(request("delete-during-held-oom", "delete", {
      id: variant.id,
      kind: "model",
    }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(scope.messages.some(
      (message) => message.kind === "response" &&
        message.requestId === "delete-during-held-oom",
    ), false);
    releaseOomRecord();
    assert.equal((await response(scope, "delete-during-held-oom")).ok, true);
    assert.equal(scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload.lifecycle, "failed");
    assert.deepEqual(order, ["session:close", "backend:unload"]);
  });

  test("worker publishes a confirmed load OOM to same-session recommendations", async () => {
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: {
        async load() {
          throw {
            code: "MODEL_OOM",
            message: "fixture allocation failed",
            recoverable: true,
            status: 507,
          };
        },
        async request() { return null; },
        async generate() {},
        async stop() {},
        async unload() {},
      },
    });
    await prepareDownload(scope);
    scope.send(request("load-confirmed-oom", "load", downloadPayload()));

    const rejected = await response(scope, "load-confirmed-oom");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "MODEL_OOM");
    const failed = scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload;
    assert.equal(failed.lifecycle, "failed");
    assert.deepEqual(failed.loadRecords.map((record) => record.result), ["oom"]);
    assert.deepEqual(scope.loadRecords.map((record) => record.result), ["oom"]);
  });

  test("worker does not loop when the replacement device is lost", async () => {
    const deviceLossCallbacks = [];
    let capabilityChecks = 0;
    let loads = 0;
    const backend = {
      async load(input) {
        loads += 1;
        deviceLossCallbacks.push(input.onDeviceLost);
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: backend,
      onCapabilityCheck() { capabilityChecks += 1; },
    });
    await prepareDownload(scope);
    scope.send(request("load-before-repeated-loss", "load", downloadPayload()));
    assert.equal((await response(scope, "load-before-repeated-loss")).ok, true);
    scope.messages.length = 0;

    deviceLossCallbacks[0]({
      code: "GPU_DEVICE_LOST",
      message: "first fixture loss",
      confirmedOom: false,
    });
    await waitFor(() => loads === 2 && scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    )?.payload.lifecycle === "ready");
    deviceLossCallbacks[1]({
      code: "GPU_DEVICE_LOST",
      message: "replacement fixture loss",
      confirmedOom: false,
    });
    await waitFor(() => scope.messages.find(
      (message) => message.kind === "event" && message.event === "device_lost",
    ));
    await new Promise((resolve) => setTimeout(resolve, 0));

    assert.equal(loads, 2);
    assert.equal(capabilityChecks, 2);
    assert.deepEqual(
      scope.loadRecords.map((record) => record.result),
      ["success", "device_lost", "success", "device_lost"],
    );
    assert.equal(scope.messages.filter(
      (message) => message.kind === "event" && message.event === "device_lost",
    ).length, 1);
    assert.equal(scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload.lifecycle, "failed");
  });

  test("worker records a device loss during load once and disposes partial resources", async () => {
    let unloads = 0;
    const backend = {
      async load(input) {
        input.onDeviceLost({
          code: "GPU_DEVICE_LOST",
          message: "fixture device lost",
          confirmedOom: false,
        });
        throw new Error("load interrupted");
      },
      async request() { throw new Error("not reached"); },
      async generate() { throw new Error("not reached"); },
      async stop() {},
      async unload() { unloads += 1; },
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-loss", "load", {
      modelVariantId: variant.id,
      contextTokens: 2048,
    }));
    assert.equal((await response(scope, "load-loss")).ok, false);
    await waitFor(() => scope.messages.find(
      (message) => message.kind === "event" && message.event === "device_lost",
    ));
    assert.deepEqual(scope.loadRecords.map((record) => record.result), ["device_lost"]);
    assert.equal(unloads, 1);
  });

  test("worker joins an in-flight load before publishing terminal device loss", async () => {
    let reportLoadStarted;
    let releaseLoad;
    let loadSignal;
    let unloads = 0;
    const loadStarted = new Promise((resolve) => { reportLoadStarted = resolve; });
    const loadRelease = new Promise((resolve) => { releaseLoad = resolve; });
    const backend = {
      async load(input) {
        loadSignal = input.signal;
        input.onDeviceLost({
          code: "GPU_DEVICE_LOST",
          message: "fixture loss while model loading",
          confirmedOom: false,
        });
        reportLoadStarted();
        await loadRelease;
        return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
      },
      async request() { return null; },
      async generate() {},
      async stop() {},
      async unload() { unloads += 1; },
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-with-held-device-loss", "load", downloadPayload()));
    await loadStarted;
    await new Promise((resolve) => setTimeout(resolve, 0));

    assert.equal(loadSignal.aborted, true);
    assert.equal(unloads, 0);
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" && message.event === "device_lost",
    ), false);
    assert.notEqual(scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload.lifecycle, "failed");

    releaseLoad();
    assert.equal((await response(scope, "load-with-held-device-loss")).ok, false);
    await waitFor(() => scope.messages.find(
      (message) => message.kind === "event" && message.event === "device_lost",
    ));
    assert.equal(unloads, 1);
    assert.equal(scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload.lifecycle, "failed");
  });

  test("worker cannot publish ready after device loss during session restore", async () => {
    let reportRestoreStarted;
    let releaseRestore;
    let reportDeviceLoss;
    let unloads = 0;
    const restoreStarted = new Promise((resolve) => { reportRestoreStarted = resolve; });
    const restoreRelease = new Promise((resolve) => { releaseRestore = resolve; });
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      modelBackend: {
        async load(input) {
          reportDeviceLoss = input.onDeviceLost;
          return { prefillTokensPerSecond: 30, decodeTokensPerSecond: 12 };
        },
        async request() { return null; },
        async generate() {},
        async stop() {},
        async unload() { unloads += 1; },
      },
      sessionCoordinatorFactory: () => ({
        async restore() {
          reportRestoreStarted();
          await restoreRelease;
          return { status: "not_found" };
        },
        async persist() {},
        async close() {},
      }),
    });
    await prepareDownload(scope);
    scope.send(request("load-with-restore-device-loss", "load", downloadPayload()));
    await restoreStarted;
    reportDeviceLoss({
      code: "GPU_DEVICE_LOST",
      message: "fixture loss during session restore",
      confirmedOom: true,
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(unloads, 0);
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" && message.event === "device_lost",
    ), false);

    releaseRestore();
    assert.equal((await response(scope, "load-with-restore-device-loss")).ok, false);
    await waitFor(() => scope.messages.find(
      (message) => message.kind === "event" && message.event === "device_lost",
    ));
    assert.equal(unloads, 1);
    assert.equal(scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload.lifecycle, "failed");
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" && message.event === "status" &&
        message.payload.lifecycle === "ready",
    ), false);
  });

  test("worker ignores a stale device-loss callback after unload", async () => {
    let reportDeviceLoss;
    const backend = {
      async load(input) {
        reportDeviceLoss = input.onDeviceLost;
        return {
          prefillTokensPerSecond: 30,
          decodeTokensPerSecond: 12,
        };
      },
      async request() { return null; },
      async generate() {},
      async stop() {},
      async unload() {},
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-stale", "load", {
      modelVariantId: variant.id,
      contextTokens: 2048,
    }));
    assert.equal((await response(scope, "load-stale")).ok, true);
    scope.send(request("unload-stale", "unload", undefined));
    assert.equal((await response(scope, "unload-stale")).ok, true);
    reportDeviceLoss({
      code: "GPU_DEVICE_LOST",
      message: "late fixture callback",
      confirmedOom: false,
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.deepEqual(scope.loadRecords.map((record) => record.result), ["success"]);
    assert.equal(scope.messages.some(
      (message) => message.kind === "event" && message.event === "device_lost",
    ), false);
  });

  test("worker protects an active model from deletion and clears it atomically", async () => {
    let unloads = 0;
    const backend = {
      async load() {
        return {
          prefillTokensPerSecond: 30,
          decodeTokensPerSecond: 12,
        };
      },
      async request() { return null; },
      async generate() {},
      async stop() {},
      async unload() { unloads += 1; },
    };
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, { modelBackend: backend });
    await prepareDownload(scope);
    scope.send(request("load-for-clear", "load", {
      modelVariantId: variant.id,
      contextTokens: 2048,
    }));
    assert.equal((await response(scope, "load-for-clear")).ok, true);
    scope.send(request("delete-active", "delete", {
      kind: "model",
      id: variant.id,
    }));
    assert.equal((await response(scope, "delete-active")).error.code, "MODEL_IN_USE");
    scope.send(request("delete-pack-active", "delete", {
      kind: "pack",
      id: "fixture-pack",
    }));
    assert.equal((await response(scope, "delete-pack-active")).error.code, "PACK_IN_USE");
    scope.send(request("clear-active", "clear", undefined));
    assert.equal((await response(scope, "clear-active")).ok, true);
    const cleared = scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload;
    assert.equal(cleared.lifecycle, "unloaded");
    assert.equal(cleared.modelVariantId, null);
    assert.deepEqual(cleared.installedModelVariantIds, []);
    assert.deepEqual(cleared.installedPackIds, []);
    assert.equal(unloads, 1);
  });

  test("worker keeps model and pack install inventories separate", async () => {
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      installs: [
        { id: variant.id, kind: "model", objectHashes: [], installedAt: 1 },
        { id: "fixture-sae", kind: "pack", objectHashes: [], installedAt: 2 },
        { id: "fixture-core", kind: "pack", objectHashes: [], installedAt: 3 },
      ],
    });
    scope.send(request("inventory", "storage", undefined));
    assert.equal((await response(scope, "inventory")).ok, true);
    const inventory = scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload;
    assert.deepEqual(inventory.installedModelVariantIds, [variant.id]);
    assert.deepEqual(inventory.installedPackIds, ["fixture-core", "fixture-sae"]);

    scope.send(request("delete-pack", "delete", {
      kind: "pack",
      id: "fixture-sae",
    }));
    assert.equal((await response(scope, "delete-pack")).ok, true);
    const afterDelete = scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload;
    assert.deepEqual(afterDelete.installedModelVariantIds, [variant.id]);
    assert.deepEqual(afterDelete.installedPackIds, ["fixture-core"]);
  });

  test("model deletion preserves persisted sessions for reinstallation", async () => {
    const deletedSessions = [];
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      sessionPersistence: {
        async claim() { return { status: "missing" }; },
        async save(input) { return input; },
        async delete(modelVariantId) {
          deletedSessions.push(modelVariantId);
          return true;
        },
        async clear() { throw new Error("model deletion must not clear all sessions"); },
        close() {},
      },
    });
    scope.send(request("delete-model-session", "delete", {
      kind: "model",
      id: variant.id,
    }));
    assert.equal((await response(scope, "delete-model-session")).ok, true);
    assert.deepEqual(deletedSessions, []);

    scope.send(request("delete-pack-session", "delete", {
      kind: "pack",
      id: "fixture-pack",
    }));
    assert.equal((await response(scope, "delete-pack-session")).ok, true);
    assert.deepEqual(deletedSessions, []);
  });

  test("model deletion does not invoke session cleanup", async () => {
    const scope = harness({
      async download() { return result; },
      cancel() { return false; },
    }, {
      selectedModelVariantId: variant.id,
      sessionPersistence: {
        async claim() { return { status: "missing" }; },
        async save(input) { return input; },
        async delete() { throw new Error("session cleanup failed"); },
        async clear() {},
        close() {},
      },
    });
    scope.send(request("delete-model-partial-failure", "delete", {
      kind: "model",
      id: variant.id,
    }));
    const deleted = await response(scope, "delete-model-partial-failure");
    assert.equal(deleted.ok, true);
    const finalState = scope.messages.findLast(
      (message) => message.kind === "event" && message.event === "status",
    ).payload;
    assert.deepEqual(finalState.installedModelVariantIds, []);
    assert.equal(finalState.selectedModelVariantId, null);
    assert.equal(scope.installs.length, 0);
  });

  test("worker cancellation linearizes after paused state", async () => {
    let rejectDownload;
    let checkpointed = false;
    const scope = harness({
      download() {
        return new Promise((_resolve, reject) => { rejectDownload = reject; });
      },
      cancel() {
        queueMicrotask(() => {
          checkpointed = true;
          rejectDownload(new downloaderModule.ArtifactDownloadError(
            "DOWNLOAD_CANCELLED",
            "cancelled fixture",
            { status: 499, recoverable: true },
          ));
        });
        return true;
      },
    });
    await prepareDownload(scope);
    scope.send(request("download", "download", downloadPayload()));
    await waitFor(() => scope.messages.some(
      (message) => message.kind === "event" && message.payload.download?.phase === "running",
    ));
    scope.send(request("cancel", "cancel", undefined));
    const [downloadResponse, cancelResponse] = await Promise.all([
      response(scope, "download"),
      response(scope, "cancel"),
    ]);

    assert.equal(checkpointed, true);
    assert.equal(downloadResponse.result.cancelled, true);
    assert.equal(cancelResponse.ok, true);
    assert.deepEqual(
      scope.messages
        .filter((message) => message.kind === "event" && message.event === "status")
        .map((message) => message.payload.download.phase),
      ["running", "cancelling", "paused"],
    );
    const pausedIndex = scope.messages.findIndex(
      (message) => message.kind === "event" && message.payload.download?.phase === "paused",
    );
    const cancelIndex = scope.messages.findIndex(
      (message) => message.kind === "response" && message.requestId === "cancel",
    );
    assert.ok(cancelIndex > pausedIndex);
  });

  test("worker rejects a second admitted download without disturbing the first", async () => {
    let rejectDownload;
    const downloader = {
      download() {
        return new Promise((_resolve, reject) => { rejectDownload = reject; });
      },
      cancel() {
        rejectDownload(new downloaderModule.ArtifactDownloadError(
          "DOWNLOAD_CANCELLED",
          "cancelled fixture",
          { status: 499, recoverable: true },
        ));
        return true;
      },
    };
    const scope = harness(downloader);
    await prepareDownload(scope);
    scope.send(request("first", "download", downloadPayload()));
    await waitFor(() => scope.messages.some(
      (message) => message.kind === "event" && message.payload.download?.phase === "running",
    ));
    scope.send(request("second", "download", downloadPayload()));
    const rejected = await response(scope, "second");
    assert.equal(rejected.ok, false);
    assert.equal(rejected.error.code, "DOWNLOAD_BUSY");
    scope.send(request("cancel", "cancel", undefined));
    await response(scope, "cancel");
  });

  test("idle and late cancellation never overwrite completion", async () => {
    let resolveDownload;
    const scope = harness({
      download() {
        return new Promise((resolve) => { resolveDownload = resolve; });
      },
      cancel() { return false; },
    });
    scope.send(request("idle-cancel", "cancel", undefined));
    assert.equal((await response(scope, "idle-cancel")).ok, true);
    assert.equal(scope.messages.some((message) => message.kind === "event"), false);

    await prepareDownload(scope);
    scope.send(request("download", "download", downloadPayload()));
    await waitFor(() => scope.messages.some(
      (message) => message.kind === "event" && message.payload.download?.phase === "running",
    ));
    scope.send(request("late-cancel", "cancel", undefined));
    await response(scope, "late-cancel");
    resolveDownload(result);
    await response(scope, "download");
    const phases = scope.messages
      .filter((message) => message.kind === "event" && message.event === "status")
      .map((message) => message.payload.download.phase);
    assert.deepEqual(phases, ["running", "complete"]);
  });

  test("worker preserves exact downloader failures in state, event, and response", async () => {
    const failure = new downloaderModule.ArtifactDownloadError(
      "DOWNLOAD_HASH_MISMATCH",
      "fixture digest mismatch",
      { status: 502, recoverable: true, detail: { path: "weights/params.bin" } },
    );
    const scope = harness({
      async download() { throw failure; },
      cancel() { return false; },
    });
    await prepareDownload(scope);
    scope.send(request("download", "download", downloadPayload()));
    const rejected = await response(scope, "download");
    const failedStatus = scope.messages.find(
      (message) => message.kind === "event" && message.payload.download?.phase === "failed",
    );
    const errorEvent = scope.messages.find(
      (message) => message.kind === "event" && message.event === "error",
    );
    assert.equal(rejected.error.code, failure.code);
    assert.deepEqual(failedStatus.payload.download.error, rejected.error);
    assert.deepEqual(errorEvent.payload.failure, rejected.error);
  });

  test("request progress and status survive a generation sequence barrier", async () => {
    const worker = new FakeTransportWorker();
    const transport = new browserModule.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    const events = [];
    const progressEvents = [];
    transport.subscribe((event) => events.push(event));
    transport.setSequenceBarrierActive(true);
    const pending = transport.call("download", downloadPayload(), {
      timeoutMs: 0,
      onProgress: (event) => progressEvents.push(event.data),
    });
    const requestId = worker.posted[0].requestId;
    worker.emit(eventEnvelope(1, "status", {}, null));
    worker.emit(eventEnvelope(3, "token", { type: "token" }, null));
    worker.emit(eventEnvelope(4, "progress", progress, requestId));
    worker.emit(eventEnvelope(5, "status", { lifecycle: "unloaded" }, requestId));
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId,
      ok: true,
      result: { modelVariantId: variant.id, installed: true, cancelled: false },
    });
    await pending;

    assert.deepEqual(progressEvents, [progress]);
    assert.deepEqual(events.map((event) => event.event), [
      "status", "error", "progress", "status",
    ]);
    transport.dispose();
  });

  test("transport terminates immediately on malformed worker messages", async () => {
    const worker = new FakeTransportWorker();
    const transport = new browserModule.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    const pending = transport.call("check", undefined);
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: null,
      sequence: "invalid",
      generationId: null,
      event: "status",
      payload: {},
    });
    await assert.rejects(pending, /message protocol/);
    assert.equal(worker.terminated, true);
  });

  test("transport terminates the worker when backend cleanup is not confirmed", async () => {
    const worker = new FakeTransportWorker();
    const transport = new browserModule.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    const failures = [];
    transport.subscribeFailure((reason) => failures.push(reason));
    const pending = transport.call("load", downloadPayload());
    const requestId = worker.posted[0].requestId;
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId,
      ok: false,
      error: {
        code: "RUNTIME_CLEANUP_FAILED",
        message: "fixture backend could not be disposed",
        recoverable: false,
        status: 500,
      },
    });
    await assert.rejects(pending, /could not be disposed/);
    assert.equal(worker.terminated, true);
    assert.deepEqual(failures, ["fixture backend could not be disposed"]);
  });

  test("transport terminates on asynchronous lifecycle cleanup failure", () => {
    const worker = new FakeTransportWorker();
    const transport = new browserModule.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    const failures = [];
    const delivered = [];
    transport.subscribeFailure((reason) => failures.push(reason));
    transport.subscribe((event) => delivered.push(event));

    worker.emit(eventEnvelope(1, "error", {
      scope: "lifecycle",
      failure: {
        code: "RUNTIME_CLEANUP_FAILED",
        message: "replacement cleanup could not be confirmed",
        recoverable: false,
        status: 500,
      },
    }, null));

    assert.equal(worker.terminated, true);
    assert.deepEqual(failures, ["replacement cleanup could not be confirmed"]);
    assert.deepEqual(delivered, []);
  });

  test("browser runtime delivers worker generation failures exactly once", async () => {
    const worker = new FakeTransportWorker();
    const transport = new browserModule.WorkerRpcTransport(worker, { requestTimeoutMs: 5 });
    const client = new browserModule.BrowserRuntimeClient(transport, { ownsTransport: false });
    const messages = [];
    client.events.subscribe((message) => messages.push(message));
    await client.events.open();
    client.events.send({ type: "generate" });
    const requestId = worker.posted[0].requestId;
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId,
      ok: false,
      error: {
        code: "GENERATION_BUSY",
        message: "fixture generation busy",
        recoverable: true,
        status: 409,
      },
    });
    await waitFor(() => messages.length === 1);
    assert.deepEqual(messages, [{
      type: "error",
      message: "fixture generation busy",
      code: "GENERATION_BUSY",
    }]);
    client.events.close();
    transport.dispose();
  });

  test("browser service RPCs do not expire while the worker may still mutate", async () => {
    const worker = new FakeTransportWorker();
    const transport = new browserModule.WorkerRpcTransport(worker, { requestTimeoutMs: 5 });
    const client = new browserModule.BrowserRuntimeClient(transport, { ownsTransport: false });
    let settled = false;
    const pending = client.tree.navigate("node-1").finally(() => { settled = true; });
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.equal(settled, false);
    const requestId = worker.posted[0].requestId;
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId,
      ok: true,
      result: { active_node_id: "node-1" },
    });
    assert.deepEqual(await pending, { active_node_id: "node-1" });
    transport.dispose();
  });

  test("browser service mutations reach the event channel before the RPC settles", async () => {
    const worker = new FakeTransportWorker();
    const transport = new browserModule.WorkerRpcTransport(worker, { requestTimeoutMs: 5 });
    const client = new browserModule.BrowserRuntimeClient(transport, { ownsTransport: false });
    const messages = [];
    client.events.subscribe((message) => messages.push(message));
    await client.events.open();
    const pending = client.tree.star("node-1", true);
    const requestId = worker.posted[0].requestId;
    const mutation = {
      type: "tree_mutated",
      op: "star",
      rev: 4,
      added: [],
      removed: [],
      updated: [{ id: "node-1", starred: true }],
      active_node_id: null,
      cast: {},
    };
    worker.emit(eventEnvelope(1, "tree_mutated", mutation, requestId));
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId,
      ok: true,
      result: { id: "node-1", starred: true },
    });
    assert.deepEqual(await pending, { id: "node-1", starred: true });
    assert.deepEqual(messages, [mutation]);
    client.events.close();
    transport.dispose();
  });

  test("browser runtime routes replay capability checks through the tree service", async () => {
    const worker = new FakeTransportWorker();
    const transport = new browserModule.WorkerRpcTransport(worker, { requestTimeoutMs: 5 });
    const client = new browserModule.BrowserRuntimeClient(transport, { ownsTransport: false });
    const pending = client.tree.replayCapabilities();
    assert.deepEqual(worker.posted[0].payload, {
      service: "tree",
      method: "replayCapabilities",
      args: [],
    });
    const requestId = worker.posted[0].requestId;
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId,
      ok: true,
      result: {
        jointLogprobs: {
          available: false,
          reason: "fixture replay scoring unavailable",
        },
      },
    });
    assert.deepEqual(await pending, {
      jointLogprobs: {
        available: false,
        reason: "fixture replay scoring unavailable",
      },
    });
    transport.dispose();
  });

  test("fitting cancellation does not expire while backend rollback settles", async () => {
    const worker = new FakeTransportWorker();
    const transport = new browserModule.WorkerRpcTransport(worker, { requestTimeoutMs: 5 });
    let settled = false;
    const pending = transport.call("cancel_fitting", undefined).finally(() => {
      settled = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.equal(settled, false);
    assert.equal(worker.terminated, false);
    const requestId = worker.posted[0].requestId;
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId,
      ok: true,
      result: undefined,
    });
    await pending;
    transport.dispose();
  });

  test("an expiring ordinary RPC terminates the worker instead of leaving it running", async () => {
    const worker = new FakeTransportWorker();
    const transport = new browserModule.WorkerRpcTransport(worker, { requestTimeoutMs: 5 });
    const pending = transport.call("check", undefined);
    await assert.rejects(pending, /request check timed out/);
    assert.equal(worker.terminated, true);
    await assert.rejects(transport.call("catalog", {}), /closed/);
  });

  test("transport cleans up a request when postMessage rejects it synchronously", async () => {
    const worker = new FakeTransportWorker();
    worker.postMessage = () => { throw new DOMException("cannot clone", "DataCloneError"); };
    const transport = new browserModule.WorkerRpcTransport(worker, { requestTimeoutMs: 5 });
    await assert.rejects(transport.call("check", undefined), /cannot clone/);
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.equal(worker.terminated, false);
    transport.dispose();
  });

  function eventEnvelope(sequence, event, payload, requestId) {
    return {
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId,
      sequence,
      generationId: event === "token" ? "generation-fixture" : null,
      event,
      payload,
    };
  }

  const testFilter = process.env.TEST_FILTER?.trim();
  const selectedTests = testFilter
    ? tests.filter(({ name }) => name.includes(testFilter))
    : tests;
  if (selectedTests.length === 0) {
    throw new Error(`No worker orchestration test matched ${JSON.stringify(testFilter)}`);
  }
  for (const { name, run } of selectedTests) {
    await run();
    console.log(`ok - ${name}`);
  }
  console.log(`${selectedTests.length} worker orchestration tests passed`);
} finally {
  await server.close();
}

async function waitFor(read, attempts = 100) {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const value = read();
    if (value !== undefined && value !== false) return value;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error("Timed out waiting for worker fixture");
}
