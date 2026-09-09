import assert from "node:assert/strict";
import { createHash } from "node:crypto";
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

const tests = [];

function test(name, run) {
  tests.push({ name, run });
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

function clone(value) {
  return structuredClone(value);
}

const REVISION = "a".repeat(40);
const FILE_SHA = "3".repeat(64);
const CORE_SHA = "4".repeat(64);
const ABI = "drowse-web-test-v1";

function catalogDocument(overrides = {}) {
  const runtimeIdentity = {
    sourceModel: "HuggingFaceTB/SmolLM2-360M-Instruct",
    sourceRevision: REVISION,
    convertedManifestSha256: "5".repeat(64),
    quantization: "q4f16_1",
    tokenizerSha256: "6".repeat(64),
    chatTemplateSha256: "7".repeat(64),
    modelLibrarySha256: "8".repeat(64),
    runtimeAbi: ABI,
    hookAbi: "post-block-v1",
    hiddenSize: 960,
    layerMap: [0, 1],
  };
  const runtimeSha = digestCanonical(runtimeIdentity);
  const bindingSha = digestCanonical({
    contextTokens: 2048,
    runtimeIdentitySha256: runtimeSha,
  });
  const document = {
    schemaVersion: 1,
    sequence: 7,
    issuedAt: "2026-01-01T00:00:00.000Z",
    expiresAt: "2027-01-01T00:00:00.000Z",
    runtimeAbi: ABI,
    models: [
      {
        id: "smollm2-360m",
        displayName: "SmolLM2 360M",
        description: "Deterministic runtime fixture",
        sourceUrl: "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct",
        license: "Apache-2.0",
        languages: ["en"],
        variants: [
          {
            id: "smollm2-360m-q4f16",
            tier: "fastest",
            structuredHookProfile: "standard-v1",
            thinkingProfile: null,
            contextProfiles: [
              {
                contextTokens: 2048,
                bindingSha256: bindingSha,
                minimumCalibrationScore: 5,
                minimumDeviceMemoryGiB: 4,
                expectedPrefillTokensPerSecond: [20, 40],
                expectedDecodeTokensPerSecond: [8, 16],
                measuredDevices: 2,
              },
            ],
            downloadBytes: 1_200,
            requiredCorePackBytes: 250,
            requirements: {
              features: ["shader-f16"],
              limits: {
              maxBufferSize: 245_760,
              maxStorageBufferBindingSize: 245_760,
              maxStorageBuffersPerShaderStage: 10,
              maxComputeWorkgroupStorageSize: 32_768,
              },
            },
            runtimeIdentity,
            runtimeIdentitySha256: runtimeSha,
            files: [
              {
                path: "weights/params_shard_0.bin",
                role: "weight",
                url: `https://huggingface.co/a9lim/drowse-web-smollm2/resolve/${REVISION}/params_shard_0.bin`,
                revision: REVISION,
                bytes: 700,
                sha256: FILE_SHA,
              },
              {
                path: "tokenizer/tokenizer.json",
                role: "tokenizer",
                url: `https://huggingface.co/a9lim/drowse-web-smollm2/resolve/${REVISION}/tokenizer.json`,
                revision: REVISION,
                bytes: 100,
                sha256: "6".repeat(64),
              },
              {
                path: "config/mlc-chat-config.json",
                role: "configuration",
                url: `https://huggingface.co/a9lim/drowse-web-smollm2/resolve/${REVISION}/mlc-chat-config.json`,
                revision: REVISION,
                bytes: 100,
                sha256: "a".repeat(64),
              },
              {
                path: "config/ndarray-cache.json",
                role: "converted_manifest",
                url: `https://huggingface.co/a9lim/drowse-web-smollm2/resolve/${REVISION}/ndarray-cache.json`,
                revision: REVISION,
                bytes: 100,
                sha256: "5".repeat(64),
              },
              {
                path: "tokenizer/chat-template.json",
                role: "chat_template",
                url: `https://huggingface.co/a9lim/drowse-web-smollm2/resolve/${REVISION}/chat-template.json`,
                revision: REVISION,
                bytes: 100,
                sha256: "7".repeat(64),
              },
              {
                path: "lib/model.wasm",
                role: "model_library",
                url: `https://huggingface.co/a9lim/drowse-web-smollm2/resolve/${REVISION}/model.wasm`,
                revision: REVISION,
                bytes: 100,
                sha256: "8".repeat(64),
              },
            ],
            packs: [
              {
                id: "smollm2-core",
                kind: "core",
                displayName: "Core geometry",
                license: "AGPL-3.0-or-later",
                sourceRepository: "a9lim/drowse-web-smollm2",
                sourceRevision: REVISION,
                bytes: 250,
                required: true,
                runtimeIdentitySha256: runtimeSha,
                compatibleContextBindingSha256: [bindingSha],
                files: [
                  {
                    path: "packs/core.safetensors",
                    role: "core_pack",
                    url: `https://huggingface.co/a9lim/drowse-web-smollm2/resolve/${REVISION}/core.safetensors`,
                    revision: REVISION,
                    bytes: 250,
                    sha256: CORE_SHA,
                  },
                ],
              },
              {
                id: "smollm2-jlens",
                kind: "jlens",
                displayName: "SmolLM2 J-lens",
                license: "Apache-2.0",
                sourceRepository: "a9lim/drowse-web-smollm2-instruments",
                sourceRevision: REVISION,
                bytes: 64,
                required: false,
                runtimeIdentitySha256: runtimeSha,
                compatibleContextBindingSha256: [bindingSha],
                files: [{
                  path: "packs/jlens.safetensors",
                  role: "instrument",
                  url: `https://huggingface.co/a9lim/drowse-web-smollm2-instruments/resolve/${REVISION}/jlens.safetensors`,
                  revision: REVISION,
                  bytes: 64,
                  sha256: "9".repeat(64),
                }],
              },
            ],
          },
        ],
      },
    ],
  };
  return Object.assign(document, overrides);
}

function digestCanonical(value) {
  return createHash("sha256").update(canonicalJson(value)).digest("hex");
}

function runtimeLockModelsFor(document) {
  return document.models.flatMap((model) => model.variants.map((variant) => {
    const artifact = new URL(variant.files[0].url);
    const path = artifact.pathname.split("/").filter(Boolean);
    const resolveIndex = path.indexOf("resolve");
    assert.ok(resolveIndex > 1);
    return {
      runtimeIdentity: structuredClone(variant.runtimeIdentity),
      structuredHookProfile: variant.structuredHookProfile,
      thinkingProfile: structuredClone(variant.thinkingProfile),
      convertedRepository: path.slice(0, resolveIndex).join("/"),
      convertedRevision: variant.files[0].revision,
      contextProfiles: variant.contextProfiles.map((profile) => profile.contextTokens),
    };
  }));
}

function canonicalJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.entries(value)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
    .join(",")}}`;
}

function capabilities(overrides = {}) {
  const facts = {
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
        maxBufferSize: 1_048_576,
        maxStorageBufferBindingSize: 1_048_576,
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
    runtimeVersion: ABI,
    issues: [],
    operations: {
      download: { available: true, reasons: [] },
      generation: { available: true, reasons: [] },
      fitting: { available: true, reasons: [] },
      manifold_artifacts: { available: false, reasons: [] },
      jlens_fitting: {
        available: false,
        reasons: [{
          code: "HOSTED_JLENS_FITTING_EXCLUDED",
          message: "J-lens fitting is available only in the Python runtime",
          severity: "hard",
        }],
      },
      sae_training: {
        available: false,
        reasons: [{
          code: "HOSTED_SAE_TRAINING_EXCLUDED",
          message: "SAE training is available only in the Python runtime",
          severity: "hard",
        }],
      },
      session_admin: {
        available: false,
        reasons: [{
          code: "HOSTED_SESSION_ADMIN_EXCLUDED",
          message: "The browser runtime has no server sessions or API keys",
          severity: "hard",
        }],
      },
      server_endpoints: {
        available: false,
        reasons: [{
          code: "HOSTED_SERVER_ENDPOINTS_EXCLUDED",
          message: "The browser runtime does not expose HTTP inference endpoints",
          severity: "hard",
        }],
      },
    },
  };
  return Object.assign(facts, overrides);
}

function loadRecord(variant, result) {
  return {
    modelVariantId: variant.id,
    runtimeIdentitySha256: variant.runtimeIdentitySha256,
    deviceSignature: "fixture-device",
    contextTokens: 2048,
    result,
    prefillTokensPerSecond: result === "success" ? 30 : null,
    decodeTokensPerSecond: result === "success" ? 12 : null,
    recordedAt: 1,
  };
}

async function signedCatalog(document, keyPair = null) {
  const pair = keyPair ?? await crypto.subtle.generateKey(
    { name: "Ed25519" },
    true,
    ["sign", "verify"],
  );
  const exactBytes = new TextEncoder().encode(JSON.stringify(document));
  const signatureBytes = await crypto.subtle.sign(
    { name: "Ed25519" },
    pair.privateKey,
    exactBytes,
  );
  const publicKey = new Uint8Array(await crypto.subtle.exportKey("raw", pair.publicKey));
  return {
    pair,
    publicKey,
    exactBytes,
    signature: {
      schemaVersion: 1,
      algorithm: "Ed25519",
      keyId: "fixture-key",
      signature: Buffer.from(signatureBytes).toString("base64"),
    },
  };
}

async function rejectsCode(run, code) {
  await assert.rejects(run, (error) => {
    assert.equal(error?.code, code);
    return true;
  });
}

class FakeWorker {
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
    const listeners = this.listeners.get(type);
    assert.ok(listeners);
    listeners.add(listener);
  }

  removeEventListener(type, listener) {
    const listeners = this.listeners.get(type);
    assert.ok(listeners);
    listeners.delete(listener);
  }

  terminate() {
    this.terminated = true;
  }

  emit(data) {
    for (const listener of this.listeners.get("message")) listener({ data });
  }

  emitError(message) {
    for (const listener of this.listeners.get("error")) listener({ message });
  }
}

class FakeContentStore {
  objects = new Map();
  installs = [];
  selectedModelVariantId = null;
  reservations = new Map();
  reservationRecords = [];
  events = [];
  resetCount = 0;
  writeSessionCount = 0;
  checkpointCount = 0;
  garbageCollectionOptions = [];
  collectUnreferencedObjects = false;

  async initialize() {}

  seedPartial(file, bytes, etag = null) {
    const descriptor = {
      sha256: file.sha256,
      expectedBytes: file.bytes,
      url: file.url,
      revision: file.revision,
      etag,
      partialOwners: [{
        id: "fixture-variant",
        kind: "model",
        catalogSequence: 1,
      }],
    };
    this.objects.set(file.sha256, {
      record: this.record(descriptor, "partial", bytes.byteLength),
      bytes: bytes.slice(),
    });
  }

  seedVerified(file, bytes) {
    const descriptor = {
      sha256: file.sha256,
      expectedBytes: file.bytes,
      url: file.url,
      revision: file.revision,
      etag: '"fixture"',
      partialOwners: [],
    };
    this.objects.set(file.sha256, {
      record: this.record(descriptor, "verified", bytes.byteLength),
      bytes: bytes.slice(),
    });
  }

  async beginPartial(descriptor) {
    const existing = this.objects.get(descriptor.sha256);
    if (existing?.record.state === "verified") return clone(existing.record);
    const compatible = existing?.record.state === "partial" &&
      existing.record.expectedBytes === descriptor.expectedBytes &&
      existing.record.url === descriptor.url &&
      existing.record.revision === descriptor.revision &&
      (existing.record.etag === null || descriptor.etag === null ||
        existing.record.etag === descriptor.etag);
    const bytes = compatible ? existing.bytes : new Uint8Array();
    const record = this.record(
      {
        ...descriptor,
        etag: compatible && existing.record.etag
          ? existing.record.etag
          : descriptor.etag,
      },
      "partial",
      bytes.byteLength,
    );
    this.objects.set(descriptor.sha256, { record, bytes });
    return clone(record);
  }

  async resetPartial(descriptor) {
    this.resetCount += 1;
    const record = this.record(descriptor, "partial", 0);
    this.objects.set(descriptor.sha256, { record, bytes: new Uint8Array() });
    return clone(record);
  }

  async openWrite(sha256, offset) {
    const object = this.objects.get(sha256);
    assert.ok(object);
    assert.equal(object.record.state, "partial");
    assert.equal(object.record.contiguousBytes, offset);
    this.writeSessionCount += 1;
    let bytes = object.bytes.slice();
    let closed = false;
    return {
      get offset() { return bytes.byteLength; },
      write: async (chunk) => {
        assert.equal(closed, false);
        const appended = new Uint8Array(bytes.byteLength + chunk.byteLength);
        appended.set(bytes);
        appended.set(chunk, bytes.byteLength);
        bytes = appended;
        return bytes.byteLength;
      },
      checkpoint: async () => {
        if (!closed) {
          closed = true;
          object.bytes = bytes;
          object.record = { ...object.record, contiguousBytes: bytes.byteLength };
          this.checkpointCount += 1;
        }
        return bytes.byteLength;
      },
    };
  }

  async inspectObject(sha256) {
    const object = this.objects.get(sha256);
    return object ? clone(object.record) : null;
  }

  async partialFile(sha256) {
    const object = this.objects.get(sha256);
    if (!object || object.record.state !== "partial") return null;
    return new Blob([object.bytes]);
  }

  async commitVerified(sha256, computedSha256, computedBytes) {
    const object = this.objects.get(sha256);
    assert.ok(object);
    assert.equal(object.record.state, "partial");
    assert.equal(computedSha256, sha256);
    assert.equal(computedBytes, object.record.expectedBytes);
    assert.equal(object.bytes.byteLength, computedBytes);
    object.record = { ...object.record, state: "verified" };
    this.events.push(`commit:${sha256}`);
  }

  async verifiedFile(sha256) {
    const object = this.objects.get(sha256);
    if (!object || object.record.state !== "verified") return null;
    return new Blob([object.bytes]);
  }

  async reserveObjects(record) {
    this.reservations.set(record.id, clone(record));
    this.reservationRecords.push(clone(record));
  }

  async releaseReservation(id) {
    this.reservations.delete(id);
  }

  async registerInstall(record, options = {}) {
    assert.ok(this.reservations.size > 0);
    for (const hash of record.objectHashes) {
      assert.equal(this.objects.get(hash)?.record.state, "verified");
    }
    this.installs.push(clone(record));
    if (options.selectAsCurrentModel === true) {
      assert.equal(record.kind, "model");
      this.selectedModelVariantId = record.id;
    }
    this.events.push(`install:${record.id}`);
  }

  async removeInstall() { return []; }
  async listInstalls() { return clone(this.installs); }
  async collectGarbage(options = {}) {
    this.garbageCollectionOptions.push(clone(options));
    if (!this.collectUnreferencedObjects) return [];
    const retained = new Set([
      ...this.installs.flatMap((install) => install.objectHashes),
      ...[...this.reservations.values()].flatMap((reservation) => reservation.objectHashes),
    ]);
    const removed = [];
    for (const sha256 of this.objects.keys()) {
      if (retained.has(sha256)) continue;
      this.objects.delete(sha256);
      removed.push(sha256);
    }
    return removed;
  }
  async clear() { this.objects.clear(); }

  record(descriptor, state, contiguousBytes) {
    return {
      ...descriptor,
      state,
      contiguousBytes,
      refCount: 0,
      updatedAt: 1,
    };
  }
}

function downloadCatalogFixture({
  base = Uint8Array.of(1, 2, 3, 4, 5, 6),
  core = Uint8Array.of(7, 8, 9),
  baseSha256 = null,
} = {}) {
  const file = (path, role, bytes, sha256 = null) => ({
    path,
    role,
    url: `https://huggingface.co/a9lim/drowse-web-test/resolve/${REVISION}/${path}`,
    revision: REVISION,
    bytes: bytes.byteLength,
    sha256: sha256 ?? createHash("sha256").update(bytes).digest("hex"),
  });
  const baseFile = file("weights/base.bin", "weight", base, baseSha256);
  const coreFile = file("packs/core.bin", "core_pack", core);
  const optionalData = Uint8Array.of(10, 11);
  const optionalFile = file("packs/optional.bin", "instrument", optionalData);
  const variant = {
    id: "fixture-variant",
    downloadBytes: base.byteLength,
    requiredCorePackBytes: core.byteLength,
    files: [baseFile],
    packs: [
      {
        id: "fixture-core",
        kind: "core",
        required: true,
        bytes: core.byteLength,
        files: [coreFile],
      },
      {
        id: "fixture-optional",
        kind: "sae",
        required: false,
        bytes: optionalData.byteLength,
        files: [optionalFile],
      },
    ],
  };
  return {
    catalog: {
      document: { sequence: 1, models: [{ variants: [variant] }] },
      exactBytes: new Uint8Array(),
      signature: {
        schemaVersion: 1,
        algorithm: "Ed25519",
        keyId: "fixture",
        signature: "fixture",
      },
      allowDownloads: true,
      stale: false,
    },
    variant,
    base,
    core,
    optionalData,
    baseFile,
    coreFile,
    optionalFile,
  };
}

function fullResponse(
  bytes,
  etag = '"fixture"',
  url = "https://huggingface.co/download",
) {
  const response = new Response(bytes, {
    status: 200,
    headers: {
      "Content-Length": String(bytes.byteLength),
      ETag: etag,
    },
  });
  Object.defineProperty(response, "url", { value: url });
  return response;
}

function partialResponse(
  bytes,
  start,
  etag = '"fixture"',
  end = bytes.byteLength - 1,
  url = "https://huggingface.co/download",
) {
  const body = bytes.slice(start, end + 1);
  const response = new Response(body, {
    status: 206,
    headers: {
      "Content-Length": String(body.byteLength),
      "Content-Range": `bytes ${start}-${end}/${bytes.byteLength}`,
      ETag: etag,
    },
  });
  Object.defineProperty(response, "url", { value: url });
  return response;
}

const noIntervalScheduler = {
  setInterval() { return 1; },
  clearInterval() {},
};

try {
  const recommendation = await server.ssrLoadModule(
    "/src/lib/runtime/recommendation.ts",
  );
  const eta = await server.ssrLoadModule("/src/lib/runtime/eta.ts");
  const range = await server.ssrLoadModule("/src/lib/runtime/range.ts");
  const catalog = await server.ssrLoadModule("/src/lib/runtime/catalog.ts");
  const contracts = await server.ssrLoadModule("/src/lib/runtime/contracts.ts");
  const browser = await server.ssrLoadModule(
    "/src/hosted/runtime/browserRuntimeClient.ts",
  );
  const downloader = await server.ssrLoadModule(
    "/src/hosted/runtime/downloader.ts",
  );
  const contentStore = await server.ssrLoadModule(
    "/src/hosted/runtime/contentStore.ts",
  );
  const browserCapabilities = await server.ssrLoadModule(
    "/src/hosted/runtime/capabilities.ts",
  );
  const hostedController = await server.ssrLoadModule(
    "/src/hosted/runtime/hostedController.ts",
  );
  const localData = await server.ssrLoadModule("/src/lib/runtime/localData.ts");
  const userErrors = await server.ssrLoadModule("/src/lib/runtime/userFacingError.ts");
  const { cacheOfflineRuntimeAssets } = await server.ssrLoadModule("/src/hosted/runtime/offlineRuntimeAssets.ts");
  test("model setup caches only missing app modules and rejects unsafe or unavailable files", async () => {
    const originalFetch = globalThis.fetch;
    const originalCaches = Object.getOwnPropertyDescriptor(globalThis, "caches");
    const files = new Map();
    const fetched = [];
    let activeFetches = 0;
    let peakFetches = 0;
    let manifest = { assets: ["/assets/browser.worker-abc.js", "/assets/drowse-web-llm-def.js", "/assets/App-ghi.css"] };
    let missing = false;
    let wrongMime = false;
    let networkFailed = false;
    Object.defineProperty(globalThis, "caches", { configurable: true, value: {
      async open(name) {
        assert.equal(name, "drowse-hosted-on-demand-assets-v1");
        return {
          async match(path) { return files.get(path); },
          async put(path, response) { files.set(path, await response.text()); },
        };
      },
    } });
    globalThis.fetch = async path => {
      fetched.push(path);
      if (networkFailed) throw new TypeError("Load failed");
      if (path === "/runtime-assets.json") return Response.json(manifest);
      activeFetches += 1;
      peakFetches = Math.max(peakFetches, activeFetches);
      await new Promise(resolve => setTimeout(resolve, 10));
      activeFetches -= 1;
      return new Response("module", { status: missing ? 404 : 200, headers: {
        "Content-Type": wrongMime ? "text/html" : path.endsWith(".css") ? "text/css" : "text/javascript",
      } });
    };
    try {
      await cacheOfflineRuntimeAssets();
      assert.equal(peakFetches, 3, "independent app files should load concurrently");
      assert.equal(activeFetches, 0);
      assert.deepEqual([...files.keys()], manifest.assets);
      fetched.length = 0;
      await cacheOfflineRuntimeAssets();
      assert.deepEqual(fetched, ["/runtime-assets.json"]);
      for (const path of ["https://outside.test/assets/App-abc.js", "/assets/App-%2fsecret.js", "/assets/App-abc.js?other=1", "/models/weights.bin"]) {
        manifest = { assets: [path] };
        await assert.rejects(cacheOfflineRuntimeAssets(), { code: "APP_MODULE_UNAVAILABLE" });
      }
      manifest = { assets: ["/assets/App-new.js"] };
      missing = true;
      await assert.rejects(cacheOfflineRuntimeAssets(), { code: "APP_MODULE_UNAVAILABLE" });
      missing = false;
      wrongMime = true;
      await assert.rejects(cacheOfflineRuntimeAssets(), { code: "APP_MODULE_UNAVAILABLE" });
      wrongMime = false;
      networkFailed = true;
      await assert.rejects(cacheOfflineRuntimeAssets(), { code: "APP_MODULE_UNAVAILABLE" });
      assert.equal(files.has("/assets/App-new.js"), false);
    } finally {
      globalThis.fetch = originalFetch;
      if (originalCaches) Object.defineProperty(globalThis, "caches", originalCaches);
      else delete globalThis.caches;
    }
  });
  for (const message of [
    "Importing a module script failed.",
    "Failed to fetch dynamically imported module: https://example.test/assets/engine.js",
    "error loading dynamically imported module",
  ]) {
    assert.equal(userErrors.userFacingError({ code: "WORKER_OPERATION_FAILED", message }),
      "Some app files could not load. Reconnect, reload Drowse, then reopen the model. Your downloaded models and saved chats are kept.");
  }

  test("turns runtime failures into actionable interface copy", () => {
    assert.equal(
      userErrors.userFacingError(Object.assign(new Error("QuotaExceededError in OPFS"), {
        code: "STORAGE_QUOTA_INSUFFICIENT",
      })),
      "There is not enough free space on this device. Free some storage or choose a smaller model, then try again.",
    );
    assert.equal(
      userErrors.userFacingError(new Error("TypeError: failed\n    at worker.ts:12")),
      "Drowse could not complete that action. Try again or reopen the model.",
    );
    assert.equal(
      userErrors.userFacingError(new Error("The download was cancelled.")),
      "The download was cancelled.",
    );
    assert.equal(
      userErrors.userFacingError(Object.assign(new Error("catalog request returned HTTP 404"), {
        code: "CATALOG_FETCH_FAILED",
      })),
      "Drowse could not load the model list. Check your connection, refresh the page, and run the device check again.",
    );
    assert.equal(
      userErrors.userFacingError(Object.assign(
        new Error("cannot delete_subtree on a node inside an in-flight generation's reservation"),
        { code: "MUTATION_DURING_GENERATION" },
      )),
      "Finish or stop the current reply before changing that part of the conversation.",
    );
    assert.equal(
      userErrors.userFacingError(Object.assign(
        new Error("file[0].url must be an immutable Drowse Hugging Face URL"),
        { code: "CATALOG_SCHEMA_INVALID" },
      )),
      "Drowse could not verify the model list, so it did not download anything. Refresh the page and run the device check again. If the error continues, try again later.",
    );
    assert.equal(
      userErrors.userFacingError(Object.assign(
        new Error("mlc-chat-config.json must contain exact Drowse capture special-token IDs"),
        { code: "MLC_CAPTURE_TOKEN_IDS_INVALID" },
      )),
      "This model package is not compatible with this version of Drowse. Update Drowse and try again. If it still fails, remove the model and download it again.",
    );
    assert.equal(
      userErrors.userFacingError({
        message: "mlc-chat-config.json must contain exact Drowse capture special-token IDs",
        body: {
          error: {
            code: "MLC_CAPTURE_TOKEN_IDS_INVALID",
          },
        },
      }),
      "This model package is not compatible with this version of Drowse. Update Drowse and try again. If it still fails, remove the model and download it again.",
    );
    assert.equal(
      userErrors.userFacingError(Object.assign(
        new Error("gemma3-1b-instruct-q4f16_1 requires maxStorageBuffersPerShaderStage >= 10; this adapter reports 9"),
        { code: "WEBGPU_LIMIT_TOO_LOW" },
      )),
      "This browser provides 9 of the 10 graphics buffers this model needs. Update your browser and graphics driver, then run the device check again. If it still fails, use another supported browser.",
    );
    assert.equal(
      userErrors.userFacingError(
        "gemma3-1b-instruct-q4f16_1 requires maxStorageBuffersPerShaderStage >= 10; this adapter reports 9",
      ),
      "This browser provides 9 of the 10 graphics buffers this model needs. Update your browser and graphics driver, then run the device check again. If it still fails, use another supported browser.",
    );
    assert.equal(
      userErrors.userFacingError(Object.assign(
        new Error("tensor[0].shape does not match internal ABI 7"),
        { code: "FUTURE_INTERNAL_FAILURE" },
      )),
      "Drowse could not complete that action. Try again or reopen the model.",
    );
  });

  test("generic worker failures preserve actionable stale-sampler recovery", () => {
    for (const code of [undefined, "WORKER_OPERATION_FAILED", "WORKER_REQUEST_FAILED"]) {
      assert.equal(userErrors.userFacingError({
        code,
        message: "Make sure 0 < top_logprobs <= 5. Got 8",
      }), "Drowse is using an older model runtime that cannot accept these sampling settings. Reload Drowse, then reopen the model. Your downloaded models and saved chats do not need to be removed.");
    }
    assert.equal(userErrors.userFacingError({
      code: "WORKER_OPERATION_FAILED",
      message: "TypeError: failure at worker.ts:12",
    }), "Drowse could not complete that action. Try again or reopen the model.");
  });

  test("device loss survives generic worker error wrappers without retry advice", () => {
    for (const code of [undefined, "WORKER_OPERATION_FAILED", "WEBGPU_DEVICE_LOST", "MODEL_DEVICE_LOST_DURING_LOAD"]) {
      const message = userErrors.userFacingError({ code, message: "The WebGPU device was lost while loading" });
      assert.match(message, /graphics device stopped responding/u);
      assert.match(message, /smaller model/u);
      assert.doesNotMatch(message, /try again/iu);
    }
  });

  test("graphics recovery only includes Windows settings on Windows", () => {
    const descriptor = Object.getOwnPropertyDescriptor(globalThis, "navigator");
    try {
      for (const userAgent of ["Macintosh Safari/605.1.15", "iPhone", "Android", "Windows NT 10.0 Chrome/150.0"]) {
        Object.defineProperty(globalThis, "navigator", { configurable: true, value: { userAgent } });
        for (const code of ["WEBGPU_DEVICE_LOST", "REPEATED_DEVICE_LOSS"]) {
          const message = userErrors.userFacingError({ code });
          assert.equal(message.includes("chrome://flags"), userAgent.includes("Windows"));
          assert.equal(message.includes("Windows Settings"), userAgent.includes("Windows"));
        }
      }
    } finally {
      if (descriptor) Object.defineProperty(globalThis, "navigator", descriptor);
      else delete globalThis.navigator;
    }
  });

  test("persistent storage requests protection while the user gesture is active", async () => {
    let requests = 0;
    const order = [];
    assert.equal(await browserCapabilities.requestPersistentStorage({
      async persisted() { order.push("status"); return true; },
      async persist() { order.push("request"); requests += 1; return false; },
    }), true);
    assert.equal(requests, 1);
    assert.deepEqual(order, ["request", "status"]);
  });

  test("persistent storage falls back cleanly when the browser declines", async () => {
    let requests = 0;
    assert.equal(await browserCapabilities.requestPersistentStorage({
      async persisted() { return false; },
      async persist() { requests += 1; return false; },
    }), false);
    assert.equal(requests, 1);
  });

  test("persistent storage still requests protection when its status check fails", async () => {
    let requests = 0;
    assert.equal(await browserCapabilities.requestPersistentStorage({
      async persisted() { throw new Error("status unavailable"); },
      async persist() { requests += 1; return true; },
    }), true);
    assert.equal(requests, 1);
  });

  test("persistent storage requests stop waiting when the browser does not answer", async () => {
    const startedAt = Date.now();
    assert.equal(await browserCapabilities.requestPersistentStorage({
      async persist() { return await new Promise(() => {}); },
      async persisted() { return false; },
    }, 20), false);
    assert.ok(Date.now() - startedAt < 500);
  });

  test("late storage approval is delivered after the bounded wait", async () => {
    let answer;
    let grants = 0;
    assert.equal(await browserCapabilities.requestPersistentStorage({
      persist: () => new Promise(resolve => { answer = resolve; }),
      persisted: async () => false,
    }, 20, () => { grants++; }), false);
    assert.equal(grants, 0);
    answer(true);
    await waitFor(() => grants === 1);
  });

  test("unsupported and rejected storage requests stay nonblocking", async () => {
    for (const storage of [
      {},
      { persisted: async () => true },
      { persist() { throw new TypeError("Storage disabled"); } },
      { persist: async () => { throw new Error("Private browsing"); } },
    ]) {
      assert.equal(await browserCapabilities.requestPersistentStorage(storage), Boolean(storage.persisted));
    }
  });

  test("immediate approval and late denial do not emit late approval", async () => {
    let grants = 0;
    const granted = () => { grants++; };
    assert.equal(await browserCapabilities.requestPersistentStorage({ persist: async () => true }, 20, granted), true);
    let answer;
    assert.equal(await browserCapabilities.requestPersistentStorage({
      persist: () => new Promise(resolve => { answer = resolve; }),
    }, 20, granted), false);
    answer(false);
    await new Promise(resolve => setTimeout(resolve, 0));
    assert.equal(grants, 0);
  });

  test("compatibility checks do not require secure-context UUID support", async () => {
    const descriptor = Object.getOwnPropertyDescriptor(globalThis, "crypto");
    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      writable: true,
      value: undefined,
    });
    try {
      const worker = new FakeWorker();
      const transport = new browser.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
      const checking = transport.call("check", undefined);
      const request = await waitFor(() => worker.posted[0]);
      assert.equal(request.requestId, "rpc-1");
      worker.emit({
        protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
        kind: "response",
        requestId: request.requestId,
        ok: true,
        result: { supported: false },
      });
      assert.deepEqual(await checking, { supported: false });
      transport.dispose();
    } finally {
      if (descriptor) Object.defineProperty(globalThis, "crypto", descriptor);
      else delete globalThis.crypto;
    }
  });

  test("local data clearing removes only Drowse keys and pending writers", async () => {
    const values = new Map([
      ["drowse.chat.v3.fixture", "legacy tree"],
      ["drowse.chat.v4.fixture", "preferences"],
      ["drowse.genui.v1.fixture", "chat mode"],
      ["other.application", "keep"],
    ]);
    const storage = {
      get length() { return values.size; },
      key(index) { return [...values.keys()][index] ?? null; },
      removeItem(key) { values.delete(key); },
    };
    let pendingWriterClears = 0;
    let delayedWriteRan = false;
    const delayedWrite = setTimeout(() => { delayedWriteRan = true; }, 5);
    const unregister = localData.registerDrowseLocalDataClearer(() => {
      pendingWriterClears += 1;
      clearTimeout(delayedWrite);
    });

    localData.clearDrowseLocalData(storage);
    await new Promise((resolve) => setTimeout(resolve, 10));

    assert.equal(pendingWriterClears, 1);
    assert.equal(delayedWriteRan, false);
    assert.deepEqual([...values], [["other.application", "keep"]]);
    unregister();
  });

  test("persisted load-history records are validated before recommendation use", () => {
    const record = loadRecord(catalogDocument().models[0].variants[0], "success");
    assert.doesNotThrow(() => contentStore.validateLoadRecord(record));
    assert.throws(
      () => contentStore.validateLoadRecord({ ...record, runtimeIdentitySha256: "bad" }),
      /SHA-256/,
    );
    assert.throws(
      () => contentStore.validateLoadRecord({ ...record, decodeTokensPerSecond: -1 }),
      /speed/,
    );
    assert.throws(
      () => contentStore.validateLoadRecord({ ...record, result: "timeout" }),
      /result/,
    );
  });

  test("catalog allows swappable lens packs but only one SAE per context", () => {
    const pack = (variant, id, bindingSha256, sha256, kind = "sae") => ({
      id,
      kind,
      displayName: id,
      license: "AGPL-3.0-or-later",
      sourceRepository: "a9lim/drowse-web-smollm2",
      sourceRevision: REVISION,
      bytes: 64,
      required: false,
      runtimeIdentitySha256: variant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [bindingSha256],
      files: [{
        path: `packs/${id}.safetensors`,
        role: "instrument",
        url: `https://huggingface.co/a9lim/drowse-web-smollm2/resolve/${REVISION}/${id}.safetensors`,
        revision: REVISION,
        bytes: 64,
        sha256,
      }],
    });
    const overlapping = catalogDocument();
    const overlappingVariant = overlapping.models[0].variants[0];
    const firstBinding = overlappingVariant.contextProfiles[0].bindingSha256;
    overlappingVariant.packs.push(
      pack(overlappingVariant, "sae-primary", firstBinding, "b".repeat(64)),
      pack(overlappingVariant, "sae-secondary", firstBinding, "c".repeat(64)),
    );
    assert.throws(
      () => catalog.validateCatalogDocument(overlapping),
      /multiple compatible sae packs/,
    );

    const missingJlens = catalogDocument();
    missingJlens.models[0].variants[0].packs =
      missingJlens.models[0].variants[0].packs.filter((candidate) =>
        candidate.kind !== "jlens"
      );
    assert.doesNotThrow(() => catalog.validateCatalogDocument(missingJlens));

    const duplicateJlens = catalogDocument();
    const duplicateJlensVariant = duplicateJlens.models[0].variants[0];
    duplicateJlensVariant.packs.push(
      pack(
        duplicateJlensVariant,
        "jlens-secondary",
        duplicateJlensVariant.contextProfiles[0].bindingSha256,
        "d".repeat(64),
        "jlens",
      ),
    );
    assert.doesNotThrow(() => catalog.validateCatalogDocument(duplicateJlens));

    const disjoint = catalogDocument();
    const disjointVariant = disjoint.models[0].variants[0];
    const secondBinding = digestCanonical({
      contextTokens: 4096,
      runtimeIdentitySha256: disjointVariant.runtimeIdentitySha256,
    });
    disjointVariant.contextProfiles.push({
      ...structuredClone(disjointVariant.contextProfiles[0]),
      contextTokens: 4096,
      bindingSha256: secondBinding,
    });
    disjointVariant.packs[0].compatibleContextBindingSha256.push(secondBinding);
    disjointVariant.packs.find((candidate) => candidate.kind === "jlens")
      .compatibleContextBindingSha256.push(secondBinding);
    disjointVariant.packs.push(
      pack(disjointVariant, "sae-2k", firstBinding, "b".repeat(64)),
      pack(disjointVariant, "sae-4k", secondBinding, "c".repeat(64)),
    );
    assert.doesNotThrow(() => catalog.validateCatalogDocument(disjoint));
    disjoint.models[0].modelType = "base";
    assert.equal(catalog.validateCatalogDocument(disjoint).models[0].modelType, "base");
    for (const kind of ["jlens", "sae"]) {
      const incomplete = clone(disjoint);
      incomplete.models[0].variants[0].packs = incomplete.models[0].variants[0].packs
        .filter((candidate) => candidate.kind !== kind);
      assert.doesNotThrow(() => catalog.validateCatalogDocument(incomplete));
    }
    const missingContext = clone(disjoint);
    missingContext.models[0].variants[0].packs = missingContext.models[0].variants[0].packs
      .filter((candidate) => candidate.id !== "sae-4k");
    assert.doesNotThrow(() => catalog.validateCatalogDocument(missingContext));
    const generationOnly = clone(disjoint);
    generationOnly.models[0].variants[0].packs = generationOnly.models[0].variants[0].packs
      .filter((candidate) => candidate.kind === "core");
    assert.doesNotThrow(() => catalog.validateCatalogDocument(generationOnly));
    disjoint.models[0].modelType = "unknown";
    assert.throws(() => catalog.validateCatalogDocument(disjoint), /modelType/);
  });

  test("recommendation separates hard failures from advisory signals", () => {
    const model = catalogDocument().models[0];
    const variant = model.variants[0];
    const base = {
      capabilities: capabilities(),
      contextTokens: 2048,
      loadRecords: [],
    };
    const eligible = recommendation.assessModelVariant(model, variant, base);
    assert.equal(eligible.eligible, true);
    assert.deepEqual(eligible.hardFailures, []);
    assert.deepEqual(eligible.advisories, []);
    assert.equal(
      eligible.requiredStorageBytes,
      Math.ceil(1_200 + 250 + 256 * 1024 * 1024),
    );

    const unknownStorage = recommendation.assessModelVariant(model, variant, {
      ...base,
      capabilities: capabilities({
        storage: {
          quotaBytes: null,
          usageBytes: null,
          availableBytes: null,
          persisted: null,
        },
      }),
    });
    assert.equal(unknownStorage.eligible, false);
    assert.ok(unknownStorage.hardFailures.some((issue) => issue.code === "STORAGE_UNKNOWN"));

    const compatibleWebGpu = capabilities().webGpu;
    const insufficientBindings = recommendation.assessModelVariant(model, variant, {
      ...base,
      capabilities: capabilities({
        webGpu: {
          ...compatibleWebGpu,
          limits: {
            ...compatibleWebGpu.limits,
            maxStorageBuffersPerShaderStage: 7,
          },
        },
      }),
    });
    assert.equal(insufficientBindings.eligible, false);
    assert.ok(
      insufficientBindings.hardFailures.some(
        (issue) =>
          issue.code === "WEBGPU_LIMIT_TOO_LOW" &&
          issue.message.includes("maxStorageBuffersPerShaderStage"),
      ),
    );

    const weakSignals = recommendation.assessModelVariant(model, variant, {
      ...base,
      capabilities: capabilities({
        signals: {
          deviceMemoryGiB: 2,
          logicalCpuCount: 4,
          mobile: false,
          language: "en-US",
          calibrationScore: 2,
        },
      }),
    });
    assert.equal(weakSignals.eligible, true);
    assert.deepEqual(
      weakSignals.advisories.map((issue) => issue.code).sort(),
      ["CALIBRATION_BELOW_PROFILE", "DEVICE_MEMORY_BELOW_PROFILE"],
    );

    const oom = recommendation.assessModelVariant(model, variant, {
      ...base,
      loadRecords: [loadRecord(variant, "oom")],
    });
    assert.equal(oom.eligible, false);
    assert.ok(oom.hardFailures.some((issue) => issue.code === "CONFIRMED_OOM"));
    const retried = recommendation.assessModelVariant(model, variant, {
      ...base,
      explicitOomRetry: true,
      loadRecords: [loadRecord(variant, "oom")],
    });
    assert.equal(retried.eligible, true);

    const installed = recommendation.assessModelVariant(model, variant, {
      ...base,
      capabilities: capabilities({
        storage: {
          quotaBytes: 2_000,
          usageBytes: 1_999,
          availableBytes: 1,
          persisted: true,
        },
      }),
      installedModelVariantIds: [variant.id],
    });
    assert.equal(installed.installed, true);
    assert.equal(installed.requiredStorageBytes, 0);
    assert.equal(installed.eligible, true);

    const history = [
      { ...loadRecord(variant, "oom"), recordedAt: 1 },
      { ...loadRecord(variant, "success"), recordedAt: 2 },
    ];
    assert.equal(
      recommendation.assessModelVariant(model, variant, {
        ...base,
        loadRecords: history,
      }).proven,
      true,
    );
    const lossesAfterSuccess = [
      ...history,
      { ...loadRecord(variant, "device_lost"), recordedAt: 3 },
      { ...loadRecord(variant, "device_lost"), recordedAt: 4 },
    ];
    const singleLoss = recommendation.assessModelVariant(model, variant, {
      ...base,
      loadRecords: lossesAfterSuccess.slice(0, -1),
    });
    assert.equal(singleLoss.proven, false);
    assert.equal(singleLoss.eligible, true);
    assert.match(singleLoss.advisories.find(({ code }) => code === "PROFILE_UNMEASURED").message,
      /stopped responding on the last attempt/);
    assert.equal(recommendation.assessModelVariant(model, variant, {
      ...base,
      loadRecords: [...lossesAfterSuccess, { ...loadRecord(variant, "success"), recordedAt: 5 }],
    }).proven, true);
    const unstable = recommendation.assessModelVariant(model, variant, {
      ...base,
      loadRecords: lossesAfterSuccess,
    });
    assert.equal(unstable.proven, false);
    assert.ok(
      unstable.hardFailures.some((issue) => issue.code === "REPEATED_DEVICE_LOSS"),
    );
    assert.equal(unstable.eligible, false);
    assert.equal(recommendation.assessModelVariant(model, variant, {
      ...base,
      contextTokens: 4096,
      explicitOomRetry: true,
      loadRecords: lossesAfterSuccess,
    }).hardFailures.some((issue) => issue.code === "REPEATED_DEVICE_LOSS"), true);
    assert.equal(recommendation.assessModelVariant(model, variant, {
      ...base,
      loadRecords: lossesAfterSuccess.map(record => ({ ...record, deviceSignature: "other-gpu" })),
    }).hardFailures.some((issue) => issue.code === "REPEATED_DEVICE_LOSS"), false);

    const gatedCapabilities = capabilities();
    gatedCapabilities.operations.generation = {
      available: false,
      reasons: [{
        code: "CUSTOM_MLC_BACKEND_UNAVAILABLE",
        message: "The custom runtime has not passed feasibility",
        severity: "hard",
      }],
    };
    const gated = recommendation.assessModelVariant(model, variant, {
      ...base,
      capabilities: gatedCapabilities,
    });
    assert.equal(gated.eligible, false);
    assert.ok(
      gated.hardFailures.some(
        (failure) => failure.code === "CUSTOM_MLC_BACKEND_UNAVAILABLE",
      ),
    );
  });

  test("base recommendations retain shorter pinned contexts without relaxing explicit load checks", () => {
    const model = catalogDocument().models[0];
    model.modelType = "base";
    const variant = model.variants[0];
    const profile = variant.contextProfiles[0];
    profile.contextTokens = 1024;
    profile.bindingSha256 = digestCanonical({
      contextTokens: 1024, runtimeIdentitySha256: variant.runtimeIdentitySha256,
    });
    variant.packs[0].compatibleContextBindingSha256 = [profile.bindingSha256];
    const options = { capabilities: capabilities(), contextTokens: 2048,
      preference: "speed", language: "en-US" };
    const [ranked] = recommendation.rankCatalogModels([model], options);
    assert.equal(ranked.eligible, true);
    assert.equal(ranked.recommended, false);
    assert.match(ranked.reason, /available for text completion/);
    assert.equal(ranked.context.contextTokens, 1024);
    assert.equal(ranked.context.bindingSha256, profile.bindingSha256);
    assert.ok(recommendation.assessModelVariant(model, variant, options)
      .hardFailures.some(issue => issue.code === "CONTEXT_UNSUPPORTED"));
    const record = { ...loadRecord(variant, "oom"), contextTokens: 1024 };
    assert.ok(recommendation.rankCatalogModels([model], { ...options, loadRecords: [record] })[0]
      .hardFailures.some(issue => issue.code === "CONFIRMED_OOM"));
    assert.ok(recommendation.rankCatalogModels([model], { ...options, contextTokens: 512 })[0]
      .hardFailures.some(issue => issue.code === "CONTEXT_UNSUPPORTED"));
    model.modelType = "chat";
    assert.ok(recommendation.rankCatalogModels([model], options)[0]
      .hardFailures.some(issue => issue.code === "CONTEXT_UNSUPPORTED"));
  });

  test("recommendation is conservative for unknown and unmeasured devices", () => {
    const base = catalogDocument().models[0];
    const models = ["fastest", "balanced", "quality"].map((tier, index) => {
      const model = clone(base);
      model.id = `model-${tier}`;
      model.variants[0].id = `variant-${tier}`;
      model.variants[0].tier = tier;
      model.variants[0].packs[0].id = `core-${tier}`;
      model.variants[0].runtimeIdentitySha256 = `${index + 5}`.repeat(64);
      model.variants[0].packs[0].runtimeIdentitySha256 =
        model.variants[0].runtimeIdentitySha256;
      return model;
    });
    const ranked = recommendation.rankCatalogModels(models, {
      capabilities: capabilities({
        webGpu: {
          ...capabilities().webGpu,
          fallback: "unknown",
        },
      }),
      contextTokens: 2048,
      preference: "quality",
      language: "en-US",
    });
    assert.equal(ranked[0].variant.tier, "fastest");
    assert.equal(ranked[0].recommended, true);

    const baseOnly = clone(models);
    baseOnly.forEach((model) => { model.modelType = "base"; });
    const options = { capabilities: capabilities(), contextTokens: 2048, preference: "speed", language: "en-US" };
    assert.equal(recommendation.rankCatalogModels(baseOnly, options).some((model) => model.recommended), false);
    const mixed = recommendation.rankCatalogModels([...baseOnly, catalogDocument().models[0]], options);
    assert.equal(mixed.find((model) => model.recommended)?.model.modelType, undefined);

    const unmeasured = clone(models[0]);
    unmeasured.variants[0].contextProfiles[0].measuredDevices = 0;
    const unmeasuredResult = recommendation.rankCatalogModels([unmeasured], {
      capabilities: capabilities(),
      contextTokens: 2048,
      preference: "speed",
      language: "en-US",
    });
    assert.equal(unmeasuredResult[0].eligible, true);
    assert.equal(unmeasuredResult[0].recommended, false);
    assert.ok(
      unmeasuredResult[0].advisories.some((issue) => issue.code === "PROFILE_UNMEASURED"),
    );

    const proven = recommendation.rankCatalogModels([unmeasured], {
      capabilities: capabilities(),
      contextTokens: 2048,
      preference: "speed",
      language: "en-US",
      loadRecords: [loadRecord(unmeasured.variants[0], "success")],
    });
    assert.equal(proven[0].proven, true);
    assert.equal(proven[0].recommended, true);
  });

  test("ETA sampling ignores unusable samples and reports stalls", () => {
    const mib = contracts.MEBIBYTE;
    assert.equal(eta.conservativeDownlinkBytesPerSecond(8), 750_000);
    assert.equal(eta.conservativeDownlinkBytesPerSecond(0), 0);
    assert.equal(eta.nextEwma(100, 200), 120);
    assert.equal(eta.nextEwma(100, 0), 100);

    const estimator = new eta.DownloadEtaEstimator(0);
    const first = estimator.observe(4 * mib, 40 * mib, 1_000);
    assert.equal(first.calculating, true);
    assert.equal(first.throughputBytesPerSecond, 4 * mib);
    const ready = estimator.observe(8 * mib, 40 * mib, 2_000);
    assert.equal(ready.calculating, false);
    assert.ok(ready.etaSeconds);

    const offline = estimator.observe(10 * mib, 40 * mib, 3_000, false);
    assert.equal(offline.throughputBytesPerSecond, 4 * mib);
    assert.equal(offline.etaSeconds, null);
    assert.equal(offline.stalled, false);
    const resumed = estimator.observe(10 * mib, 40 * mib, 12_001);
    assert.equal(resumed.stalled, false);
    const stalled = estimator.observe(10 * mib, 40 * mib, 22_002);
    assert.equal(stalled.stalled, true);
    assert.equal(stalled.etaSeconds, null);
    assert.throws(() => estimator.observe(9 * mib, 40 * mib, 23_000), /monotonic/);

    assert.equal(eta.formatEtaRange([1_080, 1_620]), "18 - 27 minutes");
    assert.equal(eta.formatEtaRange([2_700, 5_400]), "45 minutes - 1.5 hours");
    assert.equal(eta.formatEtaRange([3_600, 7_200]), "1 - 2 hours");
  });

  test("range validation appends only exact contiguous responses", () => {
    assert.deepEqual(range.parseContentRange("bytes 8-15/32"), {
      start: 8,
      end: 15,
      total: 32,
    });
    assert.equal(range.parseContentRange("bytes 8-32/32"), null);
    const facts = {
      status: 206,
      requestedOffset: 8,
      expectedTotalBytes: 32,
      contentRange: "bytes 8-31/32",
      contentLength: 24,
      storedEtag: '"immutable"',
      responseEtag: '"immutable"',
    };
    assert.deepEqual(range.validateRangeResponse(facts), {
      action: "append",
      resumable: true,
      reason: null,
    });
    assert.equal(
      range.validateRangeResponse({ ...facts, status: 200, contentRange: null }).action,
      "restart",
    );
    assert.equal(
      range.validateRangeResponse({ ...facts, contentRange: "bytes 9-31/32" }).action,
      "restart",
    );
    assert.equal(
      range.validateRangeResponse({ ...facts, responseEtag: '"changed"' }).action,
      "restart",
    );
    assert.equal(
      range.validateRangeResponse({ ...facts, status: 500 }).action,
      "reject",
    );
    assert.equal(
      range.validateRangeResponse({
        ...facts,
        status: 200,
        requestedOffset: 0,
        contentRange: null,
        contentLength: 31,
      }).action,
      "reject",
    );
    assert.equal(
      range.validateRangeResponse({
        ...facts,
        status: 200,
        requestedOffset: 0,
        requestedEnd: 15,
        contentRange: null,
        contentLength: 32,
      }).action,
      "restart",
    );
    assert.equal(
      range.validateRangeResponse({
        ...facts,
        requestedOffset: 8,
        requestedEnd: 15,
        contentRange: "bytes 8-16/32",
        contentLength: 9,
      }).action,
      "restart",
    );
  });

  test("artifact downloader verifies base and required core files before registration", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    const calls = [];
    const bodies = new Map([
      [fixture.baseFile.url, fixture.base],
      [fixture.coreFile.url, fixture.core],
    ]);
    const quotaChecks = [];
    const progress = [];
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async (url) => {
        calls.push(url);
        const body = bodies.get(url);
        assert.ok(body);
        return fullResponse(body);
      },
    });
    const result = await instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: (context) => { quotaChecks.push(context.file.path); },
      onProgress: (value) => progress.push(value),
    });

    assert.deepEqual(calls, [fixture.baseFile.url, fixture.coreFile.url]);
    assert.equal(calls.includes(fixture.optionalFile.url), false);
    assert.deepEqual(quotaChecks, [fixture.baseFile.path, fixture.coreFile.path]);
    assert.equal(result.modelVariantId, fixture.variant.id);
    assert.equal(result.totalBytes, fixture.base.byteLength + fixture.core.byteLength);
    assert.equal(result.downloadedBytes, result.totalBytes);
    assert.equal(store.installs.length, 1);
    assert.equal(store.installs[0].id, fixture.variant.id);
    assert.equal(store.selectedModelVariantId, fixture.variant.id);
    assert.deepEqual(store.installs[0].objectHashes, [
      fixture.baseFile.sha256,
      fixture.coreFile.sha256,
    ]);
    assert.deepEqual(store.events, [
      `commit:${fixture.baseFile.sha256}`,
      `commit:${fixture.coreFile.sha256}`,
      `install:${fixture.variant.id}`,
    ]);
    assert.equal(store.reservations.size, 0);
    assert.equal(store.writeSessionCount, 2);
    assert.equal(store.checkpointCount, 2);
    assert.equal(progress.at(-1).calculatingEta, false);
    assert.deepEqual(progress.at(-1).etaSeconds, [0, 0]);
    assert.ok(progress.at(-1).files.every((file) => file.verification === "verified"));
  });

  test("artifact downloader transfers independent files in parallel", async () => {
    const fixture = downloadCatalogFixture();
    const bodies = new Map([
      [fixture.baseFile.url, fixture.base],
      [fixture.coreFile.url, fixture.core],
    ]);
    let activeFetches = 0;
    let peakFetches = 0;
    const instance = new downloader.VerifiedArtifactDownloader(new FakeContentStore(), {
      scheduler: noIntervalScheduler,
      fetch: async (url) => {
        activeFetches += 1;
        peakFetches = Math.max(peakFetches, activeFetches);
        await new Promise((resolve) => setTimeout(resolve, 20));
        activeFetches -= 1;
        return fullResponse(bodies.get(url));
      },
    });

    await instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: () => true,
    });

    assert.equal(peakFetches, 2);
  });

  test("artifact downloader renews and releases its crash-reclaimable reservation", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    const intervals = new Map();
    let nextInterval = 0;
    const scheduler = {
      setInterval(callback, delayMs) {
        const handle = ++nextInterval;
        intervals.set(handle, { callback, delayMs });
        return handle;
      },
      clearInterval(handle) {
        intervals.delete(handle);
      },
    };
    let now = 1_000;
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler,
      wallNow: () => now += 1_000,
      fetch: async (_url, init) => new Promise((_resolve, reject) => {
        init.signal.addEventListener("abort", () => reject(init.signal.reason), {
          once: true,
        });
      }),
    });
    const downloadPromise = instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: () => true,
    });
    const heartbeat = await waitFor(() =>
      [...intervals.values()].find((interval) => interval.delayMs === 30_000)
    );
    const reservationsBeforeHeartbeat = store.reservationRecords.length;
    heartbeat.callback();
    await waitFor(() => store.reservationRecords.length > reservationsBeforeHeartbeat);
    const previous = store.reservationRecords.at(-2);
    const renewed = store.reservationRecords.at(-1);
    assert.equal(renewed.id, previous.id);
    assert.equal(renewed.ownerId, previous.ownerId);
    assert.ok(renewed.updatedAt > previous.updatedAt);
    assert.ok(renewed.expiresAt > previous.expiresAt);

    assert.equal(instance.cancel(), true);
    await rejectsCode(() => downloadPromise, "DOWNLOAD_CANCELLED");
    assert.equal(store.reservations.size, 0);
    assert.equal(intervals.size, 0);
  });

  test("artifact downloader honors cancellation after final verification and before commit", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    const bodies = new Map([
      [fixture.baseFile.url, fixture.base],
      [fixture.coreFile.url, fixture.core],
    ]);
    let cancelResult = null;
    let instance;
    instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async (url) => fullResponse(bodies.get(url)),
    });

    await rejectsCode(
      () => instance.download(fixture.catalog, fixture.variant.id, {
        checkQuota: () => true,
        onProgress: (progress) => {
          if (
            cancelResult === null &&
            progress.files.every((file) => file.verification === "verified")
          ) {
            cancelResult = instance.cancel();
          }
        },
      }),
      "DOWNLOAD_CANCELLED",
    );

    assert.equal(cancelResult, true);
    assert.equal(store.installs.length, 0);
    assert.equal(store.selectedModelVariantId, null);
  });

  test("artifact downloader reports cancellation as too late once commit starts", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    const bodies = new Map([
      [fixture.baseFile.url, fixture.base],
      [fixture.coreFile.url, fixture.core],
    ]);
    const originalRegister = store.registerInstall.bind(store);
    let registrationStarted = false;
    let releaseRegistration;
    const registrationGate = new Promise((resolve) => {
      releaseRegistration = resolve;
    });
    store.registerInstall = async (...args) => {
      registrationStarted = true;
      await registrationGate;
      return originalRegister(...args);
    };
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async (url) => fullResponse(bodies.get(url)),
    });

    const pending = instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: () => true,
    });
    await waitFor(() => registrationStarted);
    assert.equal(instance.cancel(), false);
    releaseRegistration();
    const result = await pending;

    assert.equal(result.modelVariantId, fixture.variant.id);
    assert.equal(store.selectedModelVariantId, fixture.variant.id);
  });

  test("artifact downloader counts duplicate hashes once for quota and transfer", async () => {
    const bytes = Uint8Array.of(1, 2, 3, 4);
    const fixture = downloadCatalogFixture({ base: bytes, core: bytes });
    const store = new FakeContentStore();
    store.seedPartial(fixture.baseFile, bytes.slice(0, 2));
    const quotaChecks = [];
    const fetches = [];
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async (url, init) => {
        fetches.push({ url, range: new Headers(init.headers).get("Range") });
        return partialResponse(bytes, 2);
      },
    });

    const result = await instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: ({ remainingManifestBytes }) => {
        quotaChecks.push(remainingManifestBytes);
        return remainingManifestBytes <= bytes.byteLength;
      },
    });

    assert.deepEqual(quotaChecks, [2]);
    assert.deepEqual(fetches, [{
      url: fixture.baseFile.url,
      range: "bytes=2-",
    }]);
    assert.equal(result.downloadedBytes, 2);
    assert.equal(result.totalBytes, bytes.byteLength * 2);
    assert.deepEqual(store.installs[0].objectHashes, [
      fixture.baseFile.sha256,
      fixture.baseFile.sha256,
    ]);
  });

  test("artifact downloader ETA counts in-flight duplicate hashes once", async () => {
    const bytes = Uint8Array.from({ length: 16 }, (_, index) => index);
    const fixture = downloadCatalogFixture({ base: bytes, core: bytes });
    const progress = [];
    let now = 0;
    const instance = new downloader.VerifiedArtifactDownloader(new FakeContentStore(), {
      scheduler: noIntervalScheduler,
      monotonicNow: () => (now += 1_500),
      fetch: async () => {
        const response = new Response(new ReadableStream({
          start(controller) {
            controller.enqueue(bytes.slice(0, 4));
            controller.enqueue(bytes.slice(4));
            controller.close();
          },
        }), {
          status: 200,
          headers: { "Content-Length": String(bytes.byteLength) },
        });
        Object.defineProperty(response, "url", {
          value: "https://huggingface.co/download",
        });
        return response;
      },
    });

    await instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: () => true,
      onProgress: (value) => progress.push(value),
    });

    const firstEta = progress.find((value) => value.etaSeconds !== null)?.etaSeconds;
    assert.deepEqual(firstEta, [10, 15]);
  });

  test("artifact downloader acquires destructive admission before catalog authorization", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    const held = [];
    const bodies = new Map([
      [fixture.baseFile.url, fixture.base],
      [fixture.coreFile.url, fixture.core],
    ]);
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      runDownloadShared: async (_signal, operation) => {
        held.push("destructive");
        try {
          return await operation();
        } finally {
          assert.equal(held.pop(), "destructive");
        }
      },
      fetch: async (url) => {
        assert.deepEqual(held, ["destructive", "catalog"]);
        return fullResponse(bodies.get(url));
      },
    });
    await instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: () => true,
      withCatalogAuthorization: async (operation) => {
        assert.deepEqual(held, ["destructive"]);
        held.push("catalog");
        try {
          return await operation();
        } finally {
          assert.equal(held.pop(), "catalog");
        }
      },
    });
    assert.deepEqual(held, []);
  });

  test("artifact downloader preserves a superseded-catalog authorization failure", async () => {
    const fixture = downloadCatalogFixture();
    const instance = new downloader.VerifiedArtifactDownloader(new FakeContentStore(), {
      scheduler: noIntervalScheduler,
      runDownloadShared: (_signal, operation) => operation(),
      fetch: async () => { throw new Error("authorization must fail before fetch"); },
    });
    await rejectsCode(
      () => instance.download(fixture.catalog, fixture.variant.id, {
        checkQuota: () => true,
        withCatalogAuthorization: async () => {
          throw new catalog.CatalogValidationError(
            "CATALOG_SUPERSEDED",
            "Refresh the accepted catalog",
          );
        },
      }),
      "CATALOG_SUPERSEDED",
    );
  });

  test("artifact downloader preserves a selection authorization failure", async () => {
    const fixture = downloadCatalogFixture();
    const optionalPack = fixture.variant.packs.find((pack) => !pack.required);
    assert.ok(optionalPack);
    const instance = new downloader.VerifiedArtifactDownloader(new FakeContentStore(), {
      scheduler: noIntervalScheduler,
      runDownloadShared: (_signal, operation) => operation(),
      fetch: async () => { throw new Error("selection authorization must fail before fetch"); },
    });
    await assert.rejects(
      () => instance.downloadPack(
        fixture.catalog,
        fixture.variant.id,
        optionalPack.id,
        {
          checkQuota: () => true,
          authorizeSelection: () => {
            throw {
              code: "PACK_MODEL_NOT_INSTALLED",
              message: "Install the model before its pack",
              recoverable: true,
              status: 409,
            };
          },
        },
      ),
      (error) => {
        assert.equal(error.code, "PACK_MODEL_NOT_INSTALLED");
        assert.equal(error.status, 409);
        return true;
      },
    );
  });

  test("artifact downloader installs an optional pack without fetching model files", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    const calls = [];
    const progress = [];
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async (url) => {
        calls.push(url);
        assert.equal(url, fixture.optionalFile.url);
        return fullResponse(fixture.optionalData);
      },
    });

    const result = await instance.downloadPack(
      fixture.catalog,
      fixture.variant.id,
      "fixture-optional",
      {
        checkQuota: () => true,
        onProgress: (value) => progress.push(value),
      },
    );

    assert.deepEqual(calls, [fixture.optionalFile.url]);
    assert.equal(result.modelVariantId, fixture.variant.id);
    assert.equal(result.packId, "fixture-optional");
    assert.equal(result.totalBytes, fixture.optionalData.byteLength);
    assert.deepEqual(store.installs, [{
      id: "fixture-optional",
      kind: "pack",
      objectHashes: [fixture.optionalFile.sha256],
      installedAt: result.installedAt,
    }]);
    assert.equal(store.selectedModelVariantId, null);
    assert.deepEqual(
      store.objects.get(fixture.optionalFile.sha256).record.partialOwners,
      [{ id: "fixture-optional", kind: "pack", catalogSequence: 1 }],
    );
    assert.ok(progress.length > 0);
    assert.ok(progress.every((value) => value.packId === "fixture-optional"));
    assert.equal(progress.at(-1).files[0].verification, "verified");
  });

  test("optional pack cancellation retains a resumable pack-owned prefix", async () => {
    const fixture = downloadCatalogFixture({
      base: Uint8Array.of(1),
      core: Uint8Array.of(2),
    });
    fixture.optionalData = Uint8Array.of(10, 11, 12, 13);
    fixture.optionalFile.bytes = fixture.optionalData.byteLength;
    fixture.optionalFile.sha256 = createHash("sha256")
      .update(fixture.optionalData)
      .digest("hex");
    fixture.variant.packs.find((pack) => pack.id === "fixture-optional").bytes =
      fixture.optionalData.byteLength;
    const store = new FakeContentStore();
    let instance;
    instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      chunkBytes: 2,
      monotonicNow: (() => {
        let now = 0;
        return () => (now += 1_000);
      })(),
      fetch: async () => {
        const response = new Response(new ReadableStream({
          start(controller) {
            controller.enqueue(fixture.optionalData.slice(0, 2));
            controller.enqueue(fixture.optionalData.slice(2));
            controller.close();
          },
        }), {
          status: 200,
          headers: { "Content-Length": String(fixture.optionalData.byteLength) },
        });
        Object.defineProperty(response, "url", {
          value: "https://huggingface.co/download",
        });
        return response;
      },
    });

    await rejectsCode(
      () => instance.downloadPack(
        fixture.catalog,
        fixture.variant.id,
        "fixture-optional",
        {
          checkQuota: () => true,
          onProgress: (value) => {
            if (value.bytesReceived === 2) instance.cancel();
          },
        },
      ),
      "DOWNLOAD_CANCELLED",
    );

    assert.equal((await store.partialFile(fixture.optionalFile.sha256)).size, 2);
    assert.deepEqual(
      store.objects.get(fixture.optionalFile.sha256).record.partialOwners,
      [{ id: "fixture-optional", kind: "pack", catalogSequence: 1 }],
    );
    assert.equal(store.installs.length, 0);
  });

  test("artifact downloader rejects required or cross-variant pack selection", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async () => assert.fail("invalid pack selection must not fetch"),
    });

    await rejectsCode(
      () => instance.downloadPack(
        fixture.catalog,
        fixture.variant.id,
        "fixture-core",
        { checkQuota: () => true },
      ),
      "OPTIONAL_PACK_INVALID",
    );
    await rejectsCode(
      () => instance.downloadPack(
        fixture.catalog,
        "missing-variant",
        "fixture-optional",
        { checkQuota: () => true },
      ),
      "MODEL_VARIANT_NOT_FOUND",
    );
    await rejectsCode(
      () => instance.downloadPack(
        fixture.catalog,
        fixture.variant.id,
        "missing-pack",
        { checkQuota: () => true },
      ),
      "OPTIONAL_PACK_NOT_FOUND",
    );
  });

  test("artifact downloader mutates partial metadata only under the SHA lock", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    const originalBeginPartial = store.beginPartial.bind(store);
    const heldHashes = new Set();
    store.beginPartial = async (descriptor) => {
      assert.equal(heldHashes.has(descriptor.sha256), true);
      return originalBeginPartial(descriptor);
    };
    const bodies = new Map([
      [fixture.baseFile.url, fixture.base],
      [fixture.coreFile.url, fixture.core],
    ]);
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      runObjectExclusive: async (sha256, _signal, operation) => {
        assert.equal(heldHashes.has(sha256), false);
        heldHashes.add(sha256);
        try {
          return await operation();
        } finally {
          heldHashes.delete(sha256);
        }
      },
      fetch: async (url) => fullResponse(bodies.get(url)),
    });

    await instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: () => true,
    });
    assert.deepEqual([...heldHashes], []);
  });

  test("artifact downloader serializes the same content object across runtimes", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    const tails = new Map();
    const active = new Set();
    const fetches = [];
    const runObjectExclusive = (sha256, signal, operation) => {
      const predecessor = tails.get(sha256) ?? Promise.resolve();
      const current = predecessor.then(async () => {
        assert.equal(signal.aborted, false);
        assert.equal(active.has(sha256), false, `concurrent writer for ${sha256}`);
        active.add(sha256);
        try {
          return await operation();
        } finally {
          active.delete(sha256);
        }
      });
      tails.set(sha256, current.catch(() => undefined));
      return current;
    };
    const dependencies = {
      scheduler: noIntervalScheduler,
      runObjectExclusive,
      fetch: async (url) => {
        fetches.push(url);
        await new Promise((resolve) => setTimeout(resolve, 0));
        return fullResponse(
          url === fixture.baseFile.url ? fixture.base : fixture.core,
        );
      },
    };
    const first = new downloader.VerifiedArtifactDownloader(store, dependencies);
    const second = new downloader.VerifiedArtifactDownloader(store, dependencies);
    await Promise.all([
      first.download(fixture.catalog, fixture.variant.id, { checkQuota: () => true }),
      second.download(fixture.catalog, fixture.variant.id, { checkQuota: () => true }),
    ]);
    assert.equal(fetches.filter((url) => url === fixture.baseFile.url).length, 1);
    assert.equal(fetches.filter((url) => url === fixture.coreFile.url).length, 1);
  });

  test("artifact downloader rejects untrusted and unavailable final response URLs", async () => {
    const fixture = downloadCatalogFixture();
    const untrustedStore = new FakeContentStore();
    const untrusted = new downloader.VerifiedArtifactDownloader(untrustedStore, {
      scheduler: noIntervalScheduler,
      fetch: async () => fullResponse(
        fixture.base,
        '"fixture"',
        "https://untrusted.example/model.bin",
      ),
    });
    await rejectsCode(
      () => untrusted.download(fixture.catalog, fixture.variant.id, {
        checkQuota: () => true,
      }),
      "DOWNLOAD_REDIRECT_REJECTED",
    );
    assert.equal(untrustedStore.writeSessionCount, 0);
    assert.equal(untrustedStore.installs.length, 0);

    const missingUrlStore = new FakeContentStore();
    const missingUrl = new downloader.VerifiedArtifactDownloader(missingUrlStore, {
      scheduler: noIntervalScheduler,
      fetch: async () => new Response(fixture.base, {
        status: 200,
        headers: { "Content-Length": String(fixture.base.byteLength) },
      }),
    });
    await rejectsCode(
      () => missingUrl.download(fixture.catalog, fixture.variant.id, {
        checkQuota: () => true,
      }),
      "DOWNLOAD_REDIRECT_REJECTED",
    );
    assert.throws(
      () => new downloader.VerifiedArtifactDownloader(new FakeContentStore(), {
        allowedRedirectOrigins: ["*"],
      }),
      /origin/i,
    );
  });

  test("artifact downloader accepts an exact configured redirect origin", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      allowedRedirectOrigins: ["https://cdn.example"],
      fetch: async (url) => fullResponse(
        url === fixture.baseFile.url ? fixture.base : fixture.core,
        '"fixture"',
        `https://cdn.example/${url === fixture.baseFile.url ? "base" : "core"}`,
      ),
    });
    await instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: () => true,
    });
    assert.equal(store.installs.length, 1);
  });

  test("artifact downloader rehashes and resumes an exact stored prefix", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    store.seedPartial(fixture.baseFile, fixture.base.slice(0, 3), '"fixture"');
    const calls = [];
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async (url, init) => {
        const headers = new Headers(init.headers);
        calls.push({ url, range: headers.get("Range"), ifRange: headers.get("If-Range") });
        if (url === fixture.baseFile.url) {
          return partialResponse(fixture.base, 3);
        }
        return fullResponse(fixture.core);
      },
    });
    const result = await instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: () => true,
    });

    assert.deepEqual(calls.find((call) => call.url === fixture.baseFile.url), {
      url: fixture.baseFile.url,
      range: "bytes=3-",
      ifRange: '"fixture"',
    });
    assert.equal(result.downloadedBytes, 3 + fixture.core.byteLength);
    assert.equal(store.resetCount, 0);
    assert.equal(store.installs.length, 1);
  });

  test("artifact downloader restarts only the current object when Range is ignored", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    store.seedPartial(fixture.baseFile, fixture.base.slice(0, 2), '"fixture"');
    const baseRanges = [];
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async (url, init) => {
        if (url === fixture.baseFile.url) {
          baseRanges.push(new Headers(init.headers).get("Range"));
          return fullResponse(fixture.base);
        }
        return fullResponse(fixture.core);
      },
    });
    await instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: () => true,
    });

    assert.deepEqual(baseRanges, ["bytes=2-", null]);
    assert.equal(store.resetCount, 1);
    assert.equal(store.installs.length, 1);
  });

  test("artifact downloader discards a partial when its exposed ETag changes", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    store.seedPartial(fixture.baseFile, fixture.base.slice(0, 2), '"old"');
    const baseCalls = [];
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async (url, init) => {
        if (url === fixture.baseFile.url) {
          const headers = new Headers(init.headers);
          baseCalls.push({ range: headers.get("Range"), ifRange: headers.get("If-Range") });
          if (baseCalls.length === 1) return partialResponse(fixture.base, 2, '"new"');
          return fullResponse(fixture.base, '"new"');
        }
        return fullResponse(fixture.core);
      },
    });
    await instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: () => true,
    });

    assert.deepEqual(baseCalls, [
      { range: "bytes=2-", ifRange: '"old"' },
      { range: null, ifRange: null },
    ]);
    assert.equal(store.resetCount, 1);
    assert.equal(store.installs.length, 1);
  });

  test("artifact downloader resets a corrupt object and never registers it", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    const corrupt = fixture.base.slice();
    corrupt[corrupt.length - 1] ^= 0xff;
    let fetches = 0;
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async (url) => {
        fetches += 1;
        return fullResponse(url === fixture.baseFile.url ? corrupt : fixture.core);
      },
    });
    await rejectsCode(
      () => instance.download(fixture.catalog, fixture.variant.id, {
        checkQuota: () => true,
      }),
      "DOWNLOAD_HASH_MISMATCH",
    );

    assert.equal(fetches, 2);
    assert.equal(store.installs.length, 0);
    assert.equal(store.resetCount, 1);
    assert.equal((await store.partialFile(fixture.baseFile.sha256)).size, 0);
    assert.equal(store.reservations.size, 0);
  });

  test("artifact downloader cancellation retains its contiguous partial", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    let bodyChunk = 0;
    let instance;
    instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      chunkBytes: 2,
      monotonicNow: (() => {
        let now = 0;
        return () => (now += 1_000);
      })(),
      fetch: async (url) => {
        assert.equal(url, fixture.baseFile.url);
        const response = new Response(new ReadableStream({
          pull(controller) {
            if (bodyChunk === 0) {
              controller.enqueue(fixture.base.slice(0, 2));
              bodyChunk += 1;
              return;
            }
            controller.enqueue(fixture.base.slice(2));
            controller.close();
          },
        }), {
          status: 200,
          headers: { "Content-Length": String(fixture.base.byteLength) },
        });
        Object.defineProperty(response, "url", {
          value: "https://huggingface.co/download",
        });
        return response;
      },
    });
    await rejectsCode(
      () => instance.download(fixture.catalog, fixture.variant.id, {
        checkQuota: () => true,
        onProgress: (progress) => {
          if (progress.files[0].bytesReceived === 2) instance.cancel();
        },
      }),
      "DOWNLOAD_CANCELLED",
    );

    assert.equal((await store.partialFile(fixture.baseFile.sha256)).size, 2);
    assert.deepEqual(
      store.objects.get(fixture.baseFile.sha256).record.partialOwners,
      [{ id: fixture.variant.id, kind: "model", catalogSequence: 1 }],
    );
    assert.deepEqual(store.garbageCollectionOptions[0].validPartialOwners, [
      {
        id: fixture.variant.id,
        kind: "model",
        objectHashes: [fixture.baseFile.sha256, fixture.coreFile.sha256],
      },
      {
        id: "fixture-optional",
        kind: "pack",
        objectHashes: [fixture.optionalFile.sha256],
      },
    ]);
    assert.equal(store.garbageCollectionOptions[0].activeReservationIds.length, 1);
    assert.match(store.garbageCollectionOptions[0].activeReservationIds[0], /^download:/u);
    assert.equal(store.reservationRecords[0].ownerKind, "download");
    assert.equal(store.reservationRecords[0].targetId, fixture.variant.id);
    assert.equal(store.reservationRecords[0].targetKind, "model");
    assert.ok(store.reservationRecords[0].expiresAt > store.reservationRecords[0].updatedAt);
    assert.equal(store.installs.length, 0);
    assert.equal(store.reservations.size, 0);
    assert.equal(store.writeSessionCount, 1);
    assert.equal(store.checkpointCount, 1);
  });

  test("artifact downloader reserves completed shards before retry garbage collection", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    store.collectUnreferencedObjects = true;
    const calls = [];
    let cancelCore = true;
    let instance;
    instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      maxParallelDownloads: 1,
      chunkBytes: 1,
      monotonicNow: (() => {
        let now = 0;
        return () => (now += 1_000);
      })(),
      fetch: async (url, init) => {
        const rangeHeader = new Headers(init.headers).get("Range");
        calls.push({ url, range: rangeHeader });
        if (url === fixture.baseFile.url) return fullResponse(fixture.base);
        assert.equal(url, fixture.coreFile.url);
        if (rangeHeader === null) return fullResponse(fixture.core);
        assert.equal(rangeHeader, "bytes=1-");
        return partialResponse(fixture.core, 1);
      },
    });

    await rejectsCode(
      () => instance.download(fixture.catalog, fixture.variant.id, {
        checkQuota: () => true,
        onProgress: (progress) => {
          if (
            cancelCore && progress.files[0].verification === "verified" &&
            progress.files[1].bytesReceived === 1
          ) {
            cancelCore = false;
            instance.cancel();
          }
        },
      }),
      "DOWNLOAD_CANCELLED",
    );
    assert.equal(store.reservations.size, 0);

    await instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: () => true,
    });

    assert.deepEqual(calls, [
      { url: fixture.baseFile.url, range: null },
      { url: fixture.coreFile.url, range: null },
      { url: fixture.coreFile.url, range: "bytes=1-" },
    ]);
    assert.equal(store.installs.length, 1);
    assert.equal(store.reservations.size, 0);
  });

  test("artifact downloader checkpoints its contiguous prefix on stream failure", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    let pull = 0;
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      maxParallelDownloads: 1,
      chunkBytes: 2,
      fetch: async () => {
        const response = new Response(new ReadableStream({
          pull(controller) {
            if (pull === 0) {
              pull += 1;
              controller.enqueue(fixture.base.slice(0, 2));
              return;
            }
            controller.error(new Error("fixture stream failed"));
          },
        }), {
          status: 200,
          headers: { "Content-Length": String(fixture.base.byteLength) },
        });
        Object.defineProperty(response, "url", {
          value: "https://huggingface.co/download",
        });
        return response;
      },
    });
    await rejectsCode(
      () => instance.download(fixture.catalog, fixture.variant.id, {
        checkQuota: () => true,
      }),
      "DOWNLOAD_NETWORK_ERROR",
    );

    assert.equal((await store.partialFile(fixture.baseFile.sha256)).size, 2);
    assert.equal(store.writeSessionCount, 1);
    assert.equal(store.checkpointCount, 1);
    assert.equal(store.installs.length, 0);
  });

  test("artifact downloader resumes from a bounded mid-response durable checkpoint", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    const originalOpenWrite = store.openWrite.bind(store);
    let failSecondSession = true;
    store.openWrite = async (sha256, offset) => {
      const writer = await originalOpenWrite(sha256, offset);
      if (
        sha256 !== fixture.baseFile.sha256 || offset !== 2 ||
        !failSecondSession
      ) return writer;
      return {
        get offset() { return writer.offset; },
        write: (chunk) => writer.write(chunk),
        checkpoint: async () => {
          failSecondSession = false;
          throw new Error("fixture worker terminated before its next checkpoint");
        },
      };
    };
    const interrupted = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      maxParallelDownloads: 1,
      chunkBytes: 1,
      durableCheckpointBytes: 2,
      durableCheckpointIntervalMs: Number.MAX_SAFE_INTEGER,
      fetch: async (url) => {
        assert.equal(url, fixture.baseFile.url);
        return fullResponse(fixture.base);
      },
    });
    await rejectsCode(
      () => interrupted.download(fixture.catalog, fixture.variant.id, {
        checkQuota: () => true,
      }),
      "CONTENT_STORE_FAILURE",
    );
    assert.equal((await store.partialFile(fixture.baseFile.sha256)).size, 2);

    store.openWrite = originalOpenWrite;
    const calls = [];
    const resumed = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async (url, init) => {
        calls.push({ url, range: new Headers(init.headers).get("Range") });
        return url === fixture.baseFile.url
          ? partialResponse(fixture.base, 2)
          : fullResponse(fixture.core);
      },
    });
    const result = await resumed.download(fixture.catalog, fixture.variant.id, {
      checkQuota: ({ remainingManifestBytes }) =>
        remainingManifestBytes <= fixture.base.byteLength - 2 + fixture.core.byteLength,
    });

    assert.deepEqual(new Set(calls.map((call) => JSON.stringify(call))), new Set([
      JSON.stringify({ url: fixture.baseFile.url, range: "bytes=2-" }),
      JSON.stringify({ url: fixture.coreFile.url, range: null }),
    ]));
    assert.equal(result.downloadedBytes, fixture.base.byteLength - 2 + fixture.core.byteLength);
    assert.equal(store.installs.length, 1);
  });

  test("artifact downloader pauses an offline fetch and resumes the same object", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    const calls = [];
    const progressEvents = [];
    let online = false;
    let waiting = false;
    let releaseOnline;
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      maxParallelDownloads: 1,
      isOnline: () => online,
      waitUntilOnline: (signal) => new Promise((resolve, reject) => {
        waiting = true;
        releaseOnline = resolve;
        signal.addEventListener("abort", () => reject(signal.reason), { once: true });
      }),
      fetch: async (url, init) => {
        calls.push({ url, range: new Headers(init.headers).get("Range") });
        if (calls.length === 1) throw new TypeError("fixture browser offline");
        return url === fixture.baseFile.url
          ? fullResponse(fixture.base)
          : fullResponse(fixture.core);
      },
    });
    const downloading = instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: () => true,
      onProgress: (value) => progressEvents.push(value),
    });
    await waitFor(() => waiting);
    assert.equal(progressEvents.some((value) => value.offline), true);
    assert.equal(store.installs.length, 0);
    online = true;
    releaseOnline();
    await downloading;

    assert.deepEqual(calls, [
      { url: fixture.baseFile.url, range: null },
      { url: fixture.baseFile.url, range: null },
      { url: fixture.coreFile.url, range: null },
    ]);
    assert.equal(store.installs.length, 1);
  });

  test("artifact downloader checkpoints an offline stream and resumes by range", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    const calls = [];
    let online = true;
    let waiting = false;
    let releaseOnline;
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      maxParallelDownloads: 1,
      chunkBytes: 2,
      isOnline: () => online,
      waitUntilOnline: (signal) => new Promise((resolve, reject) => {
        waiting = true;
        releaseOnline = resolve;
        signal.addEventListener("abort", () => reject(signal.reason), { once: true });
      }),
      fetch: async (url, init) => {
        const range = new Headers(init.headers).get("Range");
        calls.push({ url, range });
        if (calls.length === 1) {
          let pull = 0;
          const response = new Response(new ReadableStream({
            pull(controller) {
              if (pull === 0) {
                pull += 1;
                controller.enqueue(fixture.base.slice(0, 2));
                return;
              }
              online = false;
              controller.error(new Error("fixture connection lost"));
            },
          }), {
            status: 200,
            headers: { "Content-Length": String(fixture.base.byteLength) },
          });
          Object.defineProperty(response, "url", { value: "https://huggingface.co/download" });
          return response;
        }
        if (url === fixture.baseFile.url) return partialResponse(fixture.base, 2);
        return fullResponse(fixture.core);
      },
    });
    const downloading = instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: () => true,
    });
    await waitFor(() => waiting);
    assert.equal((await store.partialFile(fixture.baseFile.sha256)).size, 2);
    assert.equal(store.checkpointCount, 1);
    online = true;
    releaseOnline();
    await downloading;

    assert.deepEqual(calls, [
      { url: fixture.baseFile.url, range: null },
      { url: fixture.baseFile.url, range: "bytes=2-" },
      { url: fixture.coreFile.url, range: null },
    ]);
    assert.equal(store.installs.length, 1);
  });

  test("artifact downloader cancellation interrupts an offline pause", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    let waiting = false;
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      isOnline: () => false,
      waitUntilOnline: (signal) => new Promise((_, reject) => {
        waiting = true;
        signal.addEventListener("abort", () => reject(signal.reason), { once: true });
      }),
      fetch: async () => { throw new TypeError("fixture browser offline"); },
    });
    const downloading = instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: () => true,
    });
    await waitFor(() => waiting);
    assert.equal(instance.cancel(), true);
    await rejectsCode(() => downloading, "DOWNLOAD_CANCELLED");
    assert.equal(store.installs.length, 0);
  });

  test("artifact downloader cancellation interrupts a stalled quota check", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    let quotaStarted = false;
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async () => assert.fail("cancelled quota checks must not fetch"),
    });
    const downloading = instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: () => {
        quotaStarted = true;
        return new Promise(() => {});
      },
    });
    await waitFor(() => quotaStarted);
    assert.equal(instance.cancel(), true);
    const outcome = await Promise.race([
      downloading.then(
        () => ({ code: "resolved" }),
        (error) => ({ code: error?.code }),
      ),
      new Promise((resolve) => setTimeout(() => resolve({ code: "timeout" }), 250)),
    ]);
    assert.equal(outcome.code, "DOWNLOAD_CANCELLED");
    assert.equal(instance.isDownloading, false);
    assert.equal(store.reservations.size, 0);
  });

  test("artifact downloader checks quota before fetching each file", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    let fetches = 0;
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async () => {
        fetches += 1;
        return fullResponse(fixture.base);
      },
    });
    await rejectsCode(
      () => instance.download(fixture.catalog, fixture.variant.id, {
        checkQuota: () => false,
      }),
      "DOWNLOAD_QUOTA_INSUFFICIENT",
    );

    assert.equal(fetches, 0);
    assert.equal(store.installs.length, 0);
  });

  test("artifact downloader rechecks quota after an OPFS quota failure", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    const originalOpenWrite = store.openWrite.bind(store);
    store.openWrite = async (sha256, offset) => {
      const writer = await originalOpenWrite(sha256, offset);
      return {
        get offset() { return writer.offset; },
        async write() {
          throw new DOMException("Browser storage is full", "QuotaExceededError");
        },
        checkpoint: () => writer.checkpoint(),
      };
    };
    const checks = [];
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      maxParallelDownloads: 1,
      fetch: async () => fullResponse(fixture.base),
    });

    await rejectsCode(
      () => instance.download(fixture.catalog, fixture.variant.id, {
        checkQuota: (context) => {
          checks.push(context);
          return true;
        },
      }),
      "DOWNLOAD_QUOTA_INSUFFICIENT",
    );

    assert.equal(checks.length, 2);
    assert.equal(checks[0].remainingManifestBytes, fixture.base.byteLength + fixture.core.byteLength);
    assert.equal(checks[1].remainingManifestBytes, fixture.base.byteLength + fixture.core.byteLength);
    assert.equal(store.installs.length, 0);
    assert.equal(store.reservations.size, 0);
  });

  test("artifact downloader quota subtracts verified and resumable bytes", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    store.seedPartial(fixture.baseFile, fixture.base.slice(0, 2));
    store.seedVerified(fixture.coreFile, fixture.core);
    const checks = [];
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async () => assert.fail("fetch must not run after quota rejection"),
    });
    await rejectsCode(
      () => instance.download(fixture.catalog, fixture.variant.id, {
        checkQuota: (context) => {
          checks.push(context);
          return false;
        },
      }),
      "DOWNLOAD_QUOTA_INSUFFICIENT",
    );
    assert.equal(checks.length, 2);
    const baseCheck = checks.find((context) => context.file.path === fixture.baseFile.path);
    assert.ok(baseCheck);
    assert.equal(baseCheck.remainingFileBytes, fixture.base.byteLength - 2);
    assert.equal(baseCheck.remainingManifestBytes, fixture.base.byteLength - 2);
  });

  test("artifact downloader does not reject a fully cached model on quota", async () => {
    const fixture = downloadCatalogFixture();
    const store = new FakeContentStore();
    store.seedVerified(fixture.baseFile, fixture.base);
    store.seedVerified(fixture.coreFile, fixture.core);
    const checks = [];
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async () => assert.fail("fully cached downloads must not fetch"),
    });
    const result = await instance.download(fixture.catalog, fixture.variant.id, {
      checkQuota: (context) => {
        checks.push(context);
        return false;
      },
    });

    assert.equal(result.downloadedBytes, 0);
    assert.equal(store.installs.length, 1);
    assert.equal(store.writeSessionCount, 0);
    assert.equal(checks.length, 2);
    assert.ok(checks.every((context) => context.remainingManifestBytes === 0));
    assert.ok(checks.every((context) => context.remainingFileBytes === 0));
  });

  test("artifact downloader refuses catalogs that cannot authorize downloads", async () => {
    const fixture = downloadCatalogFixture();
    fixture.catalog.allowDownloads = false;
    const store = new FakeContentStore();
    const instance = new downloader.VerifiedArtifactDownloader(store, {
      scheduler: noIntervalScheduler,
      fetch: async () => assert.fail("fetch must not run"),
    });
    await rejectsCode(
      () => instance.download(fixture.catalog, fixture.variant.id, {
        checkQuota: () => true,
      }),
      "CATALOG_DOWNLOADS_DISABLED",
    );
    assert.equal(store.reservations.size, 0);
  });

  test("catalog admission verifies exact signed bytes and strict schema", async () => {
    const now = Date.parse("2026-06-01T00:00:00.000Z");
    const signed = await signedCatalog(catalogDocument());
    const verifier = new catalog.WebCryptoEd25519Verifier({
      "fixture-key": signed.publicKey,
    });
    const policy = {
      now,
      online: true,
      minimumSequence: 7,
      lastAcceptedSequence: 7,
      lastAcceptedIdentity: {
        exactBytes: signed.exactBytes,
        signature: signed.signature,
      },
      expectedRuntimeAbi: ABI,
      runtimeLockModels: runtimeLockModelsFor(catalogDocument()),
    };
    const verified = await catalog.verifyCatalog(
      signed.exactBytes,
      signed.signature,
      verifier,
      policy,
    );
    assert.equal(verified.document.sequence, 7);
    assert.equal(verified.allowDownloads, true);
    assert.equal(verified.stale, false);
    assert.notStrictEqual(verified.exactBytes, signed.exactBytes);

    const alternateOwner = JSON.parse(
      JSON.stringify(catalogDocument()).replaceAll("a9lim/drowse-web-", "logitsml/drowse-web-"),
    );
    const signedAlternateOwner = await signedCatalog(alternateOwner, signed.pair);
    const verifiedAlternateOwner = await catalog.verifyCatalog(
      signedAlternateOwner.exactBytes,
      signedAlternateOwner.signature,
      verifier,
      {
        ...policy,
        lastAcceptedSequence: null,
        lastAcceptedIdentity: null,
        runtimeLockModels: runtimeLockModelsFor(alternateOwner),
      },
    );
    assert.equal(verifiedAlternateOwner.document.models[0].variants[0].files[0].url.includes("/logitsml/"), true);

    const tampered = signed.exactBytes.slice();
    tampered[tampered.length - 1] ^= 1;
    await rejectsCode(
      () => catalog.verifyCatalog(tampered, signed.signature, verifier, policy),
      "CATALOG_SIGNATURE_INVALID",
    );

    const extraField = catalogDocument({ unexpected: true });
    const signedExtra = await signedCatalog(extraField, signed.pair);
    await rejectsCode(
      () => catalog.verifyCatalog(signedExtra.exactBytes, signedExtra.signature, verifier, policy),
      "CATALOG_SCHEMA_INVALID",
    );

    const wrongTotal = catalogDocument();
    wrongTotal.models[0].variants[0].downloadBytes = 999;
    const signedWrongTotal = await signedCatalog(wrongTotal, signed.pair);
    await rejectsCode(
      () => catalog.verifyCatalog(
        signedWrongTotal.exactBytes,
        signedWrongTotal.signature,
        verifier,
        policy,
      ),
      "CATALOG_SCHEMA_INVALID",
    );

    const duplicatePackIds = catalogDocument();
    const firstVariant = duplicatePackIds.models[0].variants[0];
    firstVariant.packs.push({
      id: "shared-optional-pack",
      kind: "sae",
      displayName: "Shared optional pack",
      license: "AGPL-3.0-or-later",
      sourceRepository: "a9lim/drowse-web-smollm2",
      sourceRevision: REVISION,
      bytes: 64,
      required: false,
      runtimeIdentitySha256: firstVariant.runtimeIdentitySha256,
      compatibleContextBindingSha256: [firstVariant.contextProfiles[0].bindingSha256],
      files: [{
        path: "packs/shared.safetensors",
        role: "instrument",
        url: `https://huggingface.co/a9lim/drowse-web-smollm2/resolve/${REVISION}/shared.safetensors`,
        revision: REVISION,
        bytes: 64,
        sha256: "c".repeat(64),
      }],
    });
    const secondVariant = structuredClone(firstVariant);
    secondVariant.id = "smollm2-360m-q4f16-alt";
    secondVariant.packs[0].id = "smollm2-core-alt";
    duplicatePackIds.models[0].variants.push(secondVariant);
    const signedDuplicatePackIds = await signedCatalog(duplicatePackIds, signed.pair);
    await rejectsCode(
      () => catalog.verifyCatalog(
        signedDuplicatePackIds.exactBytes,
        signedDuplicatePackIds.signature,
        verifier,
        policy,
      ),
      "CATALOG_SCHEMA_INVALID",
    );

    for (const catalogId of ["model", "variant", "pack"]) {
      const oversizedId = catalogDocument();
      const oversizedValue = "x".repeat(201);
      if (catalogId === "model") oversizedId.models[0].id = oversizedValue;
      if (catalogId === "variant") oversizedId.models[0].variants[0].id = oversizedValue;
      if (catalogId === "pack") {
        oversizedId.models[0].variants[0].packs[0].id = oversizedValue;
      }
      const signedOversizedId = await signedCatalog(oversizedId, signed.pair);
      await rejectsCode(
        () => catalog.verifyCatalog(
          signedOversizedId.exactBytes,
          signedOversizedId.signature,
          verifier,
          policy,
        ),
        "CATALOG_SCHEMA_INVALID",
      );

      const controlId = catalogDocument();
      const controlValue = "bad\0id";
      if (catalogId === "model") controlId.models[0].id = controlValue;
      if (catalogId === "variant") controlId.models[0].variants[0].id = controlValue;
      if (catalogId === "pack") controlId.models[0].variants[0].packs[0].id = controlValue;
      const signedControlId = await signedCatalog(controlId, signed.pair);
      await rejectsCode(
        () => catalog.verifyCatalog(
          signedControlId.exactBytes,
          signedControlId.signature,
          verifier,
          policy,
        ),
        "CATALOG_SCHEMA_INVALID",
      );
    }

    const unknownLimit = catalogDocument();
    unknownLimit.models[0].variants[0].requirements.limits.futureLimit = 1;
    const signedUnknownLimit = await signedCatalog(unknownLimit, signed.pair);
    await rejectsCode(
      () => catalog.verifyCatalog(
        signedUnknownLimit.exactBytes,
        signedUnknownLimit.signature,
        verifier,
        policy,
      ),
      "CATALOG_SCHEMA_INVALID",
    );

    const unknownStructuredProfile = catalogDocument();
    unknownStructuredProfile.models[0].variants[0].structuredHookProfile = "adapter-derived";
    const signedUnknownStructuredProfile = await signedCatalog(
      unknownStructuredProfile,
      signed.pair,
    );
    await rejectsCode(
      () => catalog.verifyCatalog(
        signedUnknownStructuredProfile.exactBytes,
        signedUnknownStructuredProfile.signature,
        verifier,
        policy,
      ),
      "CATALOG_SCHEMA_INVALID",
    );

    const exactReadout = catalogDocument();
    const exactReadoutVariant = exactReadout.models[0].variants[0];
    exactReadoutVariant.structuredHookProfile = "standard-v3";
    Object.assign(exactReadoutVariant.requirements.limits, {
      maxBufferSize: 134_217_728,
      maxStorageBufferBindingSize: 134_217_728,
      maxComputeWorkgroupSizeX: 256,
      maxComputeInvocationsPerWorkgroup: 256,
    });
    const signedExactReadout = await signedCatalog(exactReadout, signed.pair);
    const exactReadoutPolicy = {
      ...policy,
      lastAcceptedSequence: null,
      lastAcceptedIdentity: null,
      runtimeLockModels: runtimeLockModelsFor(exactReadout),
    };
    const verifiedExactReadout = await catalog.verifyCatalog(
      signedExactReadout.exactBytes,
      signedExactReadout.signature,
      verifier,
      exactReadoutPolicy,
    );
    assert.equal(
      verifiedExactReadout.document.models[0].variants[0].requirements.limits
        .maxComputeWorkgroupSizeX,
      256,
    );

    for (const name of [
      "maxComputeWorkgroupSizeX",
      "maxComputeInvocationsPerWorkgroup",
    ]) {
      const missingExactReadoutLimit = structuredClone(exactReadout);
      delete missingExactReadoutLimit.models[0].variants[0].requirements.limits[name];
      const signedMissingExactReadoutLimit = await signedCatalog(
        missingExactReadoutLimit,
        signed.pair,
      );
      await rejectsCode(
        () => catalog.verifyCatalog(
          signedMissingExactReadoutLimit.exactBytes,
          signedMissingExactReadoutLimit.signature,
          verifier,
          exactReadoutPolicy,
        ),
        "CATALOG_SCHEMA_INVALID",
      );

      const lowExactReadoutLimit = structuredClone(exactReadout);
      lowExactReadoutLimit.models[0].variants[0].requirements.limits[name] = 255;
      const signedLowExactReadoutLimit = await signedCatalog(
        lowExactReadoutLimit,
        signed.pair,
      );
      await rejectsCode(
        () => catalog.verifyCatalog(
          signedLowExactReadoutLimit.exactBytes,
          signedLowExactReadoutLimit.signature,
          verifier,
          exactReadoutPolicy,
        ),
        "CATALOG_SCHEMA_INVALID",
      );
    }

    const malformedThinkingProfile = catalogDocument();
    malformedThinkingProfile.models[0].variants[0].thinkingProfile = {
      start: "<think>",
      end: "</think>",
      startsInThinking: false,
      startTokenIds: [],
      endTokenIds: [151668],
    };
    const signedMalformedThinkingProfile = await signedCatalog(
      malformedThinkingProfile,
      signed.pair,
    );
    await rejectsCode(
      () => catalog.verifyCatalog(
        signedMalformedThinkingProfile.exactBytes,
        signedMalformedThinkingProfile.signature,
        verifier,
        policy,
      ),
      "CATALOG_SCHEMA_INVALID",
    );

    const unlockedThinkingProfile = catalogDocument();
    unlockedThinkingProfile.models[0].variants[0].thinkingProfile = {
      start: "<think>",
      end: "</think>",
      startsInThinking: false,
      startTokenIds: [151667],
      endTokenIds: [151668],
    };
    const signedUnlockedThinkingProfile = await signedCatalog(
      unlockedThinkingProfile,
      signed.pair,
    );
    await rejectsCode(
      () => catalog.verifyCatalog(
        signedUnlockedThinkingProfile.exactBytes,
        signedUnlockedThinkingProfile.signature,
        verifier,
        policy,
      ),
      "CATALOG_RUNTIME_IDENTITY_UNLOCKED",
    );

    const insufficientStructuredLimit = catalogDocument();
    insufficientStructuredLimit.models[0].variants[0].requirements.limits
      .maxComputeWorkgroupStorageSize = 16_384;
    const signedInsufficientStructuredLimit = await signedCatalog(
      insufficientStructuredLimit,
      signed.pair,
    );
    await rejectsCode(
      () => catalog.verifyCatalog(
        signedInsufficientStructuredLimit.exactBytes,
        signedInsufficientStructuredLimit.signature,
        verifier,
        policy,
      ),
      "CATALOG_SCHEMA_INVALID",
    );

    const insufficientStructuredBindings = catalogDocument();
    insufficientStructuredBindings.models[0].variants[0].requirements.limits
      .maxStorageBuffersPerShaderStage = 7;
    const signedInsufficientStructuredBindings = await signedCatalog(
      insufficientStructuredBindings,
      signed.pair,
    );
    await rejectsCode(
      () => catalog.verifyCatalog(
        signedInsufficientStructuredBindings.exactBytes,
        signedInsufficientStructuredBindings.signature,
        verifier,
        policy,
      ),
      "CATALOG_SCHEMA_INVALID",
    );

    const identityMismatch = catalogDocument();
    identityMismatch.models[0].variants[0].runtimeIdentity.hiddenSize += 1;
    const signedIdentityMismatch = await signedCatalog(identityMismatch, signed.pair);
    await rejectsCode(
      () => catalog.verifyCatalog(
        signedIdentityMismatch.exactBytes,
        signedIdentityMismatch.signature,
        verifier,
        policy,
      ),
      "CATALOG_SCHEMA_INVALID",
    );

    const tokenizerMismatch = catalogDocument();
    tokenizerMismatch.models[0].variants[0].files.find(
      (file) => file.role === "tokenizer",
    ).sha256 = "b".repeat(64);
    const signedTokenizerMismatch = await signedCatalog(tokenizerMismatch, signed.pair);
    await rejectsCode(
      () => catalog.verifyCatalog(
        signedTokenizerMismatch.exactBytes,
        signedTokenizerMismatch.signature,
        verifier,
        policy,
      ),
      "CATALOG_SCHEMA_INVALID",
    );

    for (const field of ["license", "sourceRepository", "sourceRevision"]) {
      const missingPackProvenance = catalogDocument();
      delete missingPackProvenance.models[0].variants[0].packs[0][field];
      const signedMissingPackProvenance = await signedCatalog(
        missingPackProvenance,
        signed.pair,
      );
      await rejectsCode(
        () => catalog.verifyCatalog(
          signedMissingPackProvenance.exactBytes,
          signedMissingPackProvenance.signature,
          verifier,
          policy,
        ),
        "CATALOG_SCHEMA_INVALID",
      );
    }

    const packRevisionMismatch = catalogDocument();
    packRevisionMismatch.models[0].variants[0].packs[0].sourceRevision = "b".repeat(40);
    const signedPackRevisionMismatch = await signedCatalog(packRevisionMismatch, signed.pair);
    await rejectsCode(
      () => catalog.verifyCatalog(
        signedPackRevisionMismatch.exactBytes,
        signedPackRevisionMismatch.signature,
        verifier,
        policy,
      ),
      "CATALOG_SCHEMA_INVALID",
    );

    const packRepositoryMismatch = catalogDocument();
    packRepositoryMismatch.models[0].variants[0].packs[0].sourceRepository =
      "a9lim/drowse-web-other-pack";
    const signedPackRepositoryMismatch = await signedCatalog(
      packRepositoryMismatch,
      signed.pair,
    );
    await rejectsCode(
      () => catalog.verifyCatalog(
        signedPackRepositoryMismatch.exactBytes,
        signedPackRepositoryMismatch.signature,
        verifier,
        policy,
      ),
      "CATALOG_SCHEMA_INVALID",
    );

    const convertedProvenanceMismatch = catalogDocument();
    const mismatchedVariant = convertedProvenanceMismatch.models[0].variants[0];
    for (const file of [
      ...mismatchedVariant.files,
      ...mismatchedVariant.packs.filter((pack) => pack.required)
        .flatMap((pack) => pack.files),
    ]) {
      file.url = file.url.replace(
        "a9lim/drowse-web-smollm2",
        "a9lim/drowse-web-alternate-runtime",
      );
    }
    mismatchedVariant.packs.find((pack) => pack.required).sourceRepository =
      "a9lim/drowse-web-alternate-runtime";
    const signedProvenanceMismatch = await signedCatalog(
      convertedProvenanceMismatch,
      signed.pair,
    );
    await rejectsCode(
      () => catalog.verifyCatalog(
        signedProvenanceMismatch.exactBytes,
        signedProvenanceMismatch.signature,
        verifier,
        policy,
      ),
      "CATALOG_CONVERTED_PROVENANCE_MISMATCH",
    );

    await rejectsCode(
      () => catalog.verifyCatalog(
        signed.exactBytes,
        { ...signed.signature, unexpected: true },
        verifier,
        policy,
      ),
      "CATALOG_SIGNATURE_INVALID",
    );
  });

  test("catalog admission blocks rollback and online expiry", async () => {
    const base = await signedCatalog(catalogDocument());
    const verifier = new catalog.WebCryptoEd25519Verifier({
      "fixture-key": base.publicKey,
    });
    const rollback = await signedCatalog(catalogDocument({ sequence: 6 }), base.pair);
    await rejectsCode(
      () => catalog.verifyCatalog(
        rollback.exactBytes,
        rollback.signature,
        verifier,
        {
          now: Date.parse("2026-06-01T00:00:00.000Z"),
          online: true,
          minimumSequence: 7,
          lastAcceptedSequence: null,
          expectedRuntimeAbi: ABI,
          runtimeLockModels: runtimeLockModelsFor(catalogDocument()),
        },
      ),
      "CATALOG_ROLLBACK",
    );

    const expiredDocument = catalogDocument({
      expiresAt: "2026-05-01T00:00:00.000Z",
    });
    const expired = await signedCatalog(expiredDocument, base.pair);
    const expiredPolicy = {
      now: Date.parse("2026-06-01T00:00:00.000Z"),
      online: true,
      minimumSequence: 1,
      lastAcceptedSequence: null,
      expectedRuntimeAbi: ABI,
      runtimeLockModels: runtimeLockModelsFor(catalogDocument()),
    };
    await rejectsCode(
      () => catalog.verifyCatalog(
        expired.exactBytes,
        expired.signature,
        verifier,
        expiredPolicy,
      ),
      "CATALOG_EXPIRED",
    );
    const offline = await catalog.verifyCatalog(
      expired.exactBytes,
      expired.signature,
      verifier,
      { ...expiredPolicy, online: false },
    );
    assert.equal(offline.stale, true);
    assert.equal(offline.allowDownloads, false);
  });

  test("hosted controller routes optional pack progress and installation", async () => {
    const worker = new FakeWorker();
    const transport = new browser.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    const controller = new hostedController.HostedControllerImpl(transport, {
      ownsTransport: false,
    });
    const progress = {
      modelVariantId: "fixture-variant",
      packId: "fixture-sae",
      files: [],
      bytesReceived: 32,
      bytesTotal: 64,
      throughputBytesPerSecond: 16,
      etaSeconds: [2, 3],
      calculatingEta: false,
      stalled: false,
      offline: false,
      resumable: true,
    };
    const received = [];
    const downloading = controller.downloadPack(
      "fixture-variant",
      "fixture-sae",
      (value) => received.push(clone(value)),
    );
    const request = await waitFor(() => worker.posted[0]);
    assert.deepEqual(
      { command: request.command, payload: request.payload },
      {
        command: "download_pack",
        payload: { modelVariantId: "fixture-variant", packId: "fixture-sae" },
      },
    );
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: request.requestId,
      sequence: 1,
      generationId: null,
      event: "progress",
      payload: progress,
    });
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: request.requestId,
      sequence: 2,
      generationId: null,
      event: "status",
      payload: {
        lifecycle: "unloaded",
        modelVariantId: null,
        contextTokens: null,
        selectedModelVariantId: "fixture-variant",
        installedModelVariantIds: ["fixture-variant"],
        installedPackIds: ["fixture-sae"],
        loadRecords: [],
        download: {
          phase: "complete",
          startedAt: 1,
          finishedAt: 2,
          error: null,
        },
        generation: { phase: "idle", startedAt: null, finishedAt: null, error: null },
        fitting: { phase: "idle", startedAt: null, finishedAt: null, error: null },
        error: null,
      },
    });
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: request.requestId,
      ok: true,
      result: {
        modelVariantId: "fixture-variant",
        packId: "fixture-sae",
        installed: true,
        cancelled: false,
      },
    });

    assert.deepEqual(await downloading, {
      modelVariantId: "fixture-variant",
      packId: "fixture-sae",
      installed: true,
      cancelled: false,
    });
    assert.deepEqual(received, [progress]);
    assert.deepEqual(controller.snapshot.installedPackIds, ["fixture-sae"]);
    transport.dispose();
  });

  test("destructive controller operations publish only committed storage snapshots", async () => {
    const worker = new FakeWorker();
    const transport = new browser.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    const events = [];
    let sequence = 0;
    const runtimeSnapshot = (overrides = {}) => ({
      lifecycle: "ready",
      modelVariantId: "fixture-variant",
      contextTokens: 2048,
      selectedModelVariantId: "fixture-variant",
      installedModelVariantIds: ["fixture-variant"],
      installedPackIds: ["fixture-sae"],
      loadRecords: [],
      download: { phase: "idle", startedAt: null, finishedAt: null, error: null },
      generation: { phase: "idle", startedAt: null, finishedAt: null, error: null },
      fitting: { phase: "idle", startedAt: null, finishedAt: null, error: null },
      error: null,
      ...overrides,
    });
    const emitStatus = (payload) => worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: null,
      sequence: ++sequence,
      generationId: null,
      event: "status",
      payload,
    });
    const ownership = {
      isOwner: false,
      async acquire() {
        events.push("ownership:acquire");
        this.isOwner = true;
        return true;
      },
      release() {
        events.push("ownership:release");
        this.isOwner = false;
      },
      close() {},
    };
    let localDataClears = 0;
    const controller = new hostedController.HostedControllerImpl(transport, {
      ownsTransport: false,
      ownershipFactory: () => ownership,
      clearLocalData: () => { localDataClears += 1; },
      async runDestructiveExclusive(operation) {
        events.push("barrier:start");
        try {
          return await operation();
        } finally {
          events.push("barrier:end");
        }
      },
    });
    emitStatus(runtimeSnapshot());
    const published = [];
    const unsubscribe = controller.subscribe((snapshot) => published.push(clone(snapshot)));
    assert.equal(published.length, 1);

    const deleting = controller.deleteModel("fixture-variant");
    const unload = await waitFor(() => worker.posted[0]);
    assert.equal(unload.command, "unload");
    emitStatus(runtimeSnapshot({
      lifecycle: "unloaded",
      modelVariantId: null,
      contextTokens: null,
    }));
    assert.equal(controller.snapshot.lifecycle, "unloaded");
    assert.equal(published.length, 1);
    const latePublished = [];
    const unsubscribeLate = controller.subscribe((snapshot) => {
      latePublished.push(clone(snapshot));
    });
    assert.equal(latePublished[0].lifecycle, "ready");
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: unload.requestId,
      ok: true,
      result: undefined,
    });
    const remove = await waitFor(() => worker.posted[1]);
    assert.deepEqual(
      { command: remove.command, payload: remove.payload },
      { command: "delete", payload: { kind: "model", id: "fixture-variant" } },
    );
    emitStatus(runtimeSnapshot({
      lifecycle: "unloaded",
      modelVariantId: null,
      contextTokens: null,
      selectedModelVariantId: null,
      installedModelVariantIds: [],
    }));
    assert.equal(published.length, 1);
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: remove.requestId,
      ok: true,
      result: undefined,
    });
    await deleting;
    assert.equal(published.length, 2);
    assert.equal(latePublished.length, 2);
    assert.deepEqual(published.at(-1).installedModelVariantIds, []);
    assert.deepEqual(published.at(-1).installedPackIds, ["fixture-sae"]);
    unsubscribeLate();

    emitStatus(runtimeSnapshot());
    const beforePackDelete = published.length;
    const deletingPack = controller.deletePack("fixture-sae");
    const packUnload = await waitFor(() => worker.posted[2]);
    assert.equal(packUnload.command, "unload");
    emitStatus(runtimeSnapshot({
      lifecycle: "unloaded",
      modelVariantId: null,
      contextTokens: null,
    }));
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: packUnload.requestId,
      ok: true,
      result: undefined,
    });
    const removePack = await waitFor(() => worker.posted[3]);
    assert.deepEqual(
      { command: removePack.command, payload: removePack.payload },
      { command: "delete", payload: { kind: "pack", id: "fixture-sae" } },
    );
    emitStatus(runtimeSnapshot({
      lifecycle: "unloaded",
      modelVariantId: null,
      contextTokens: null,
      installedPackIds: [],
    }));
    assert.equal(published.length, beforePackDelete);
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: removePack.requestId,
      ok: true,
      result: undefined,
    });
    await deletingPack;
    assert.equal(published.length, beforePackDelete + 1);
    assert.deepEqual(published.at(-1).installedPackIds, []);

    emitStatus(runtimeSnapshot());
    const beforeClear = published.length;
    const clearing = controller.clearAll();
    const clearUnload = await waitFor(() => worker.posted[4]);
    assert.equal(clearUnload.command, "unload");
    emitStatus(runtimeSnapshot({
      lifecycle: "unloaded",
      modelVariantId: null,
      contextTokens: null,
    }));
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: clearUnload.requestId,
      ok: true,
      result: undefined,
    });
    const clear = await waitFor(() => worker.posted[5]);
    assert.equal(clear.command, "clear");
    emitStatus(runtimeSnapshot({
      lifecycle: "unloaded",
      modelVariantId: null,
      contextTokens: null,
      selectedModelVariantId: null,
      installedModelVariantIds: [],
      installedPackIds: [],
    }));
    assert.equal(published.length, beforeClear);
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: clear.requestId,
      ok: true,
      result: undefined,
    });
    await clearing;
    assert.equal(localDataClears, 1);
    assert.equal(published.length, beforeClear + 1);
    assert.deepEqual(published.at(-1).installedModelVariantIds, []);
    assert.deepEqual(published.at(-1).installedPackIds, []);

    assert.deepEqual(events, [
      "ownership:acquire",
      "barrier:start",
      "barrier:end",
      "ownership:release",
      "ownership:acquire",
      "barrier:start",
      "barrier:end",
      "ownership:release",
      "ownership:acquire",
      "barrier:start",
      "barrier:end",
      "ownership:release",
    ]);
    unsubscribe();
    transport.dispose();
  });

  test("controller retains GPU ownership when load preflight preserves a ready runtime", async () => {
    const worker = new FakeWorker();
    const transport = new browser.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    let releases = 0;
    const ownership = {
      isOwner: false,
      async acquire() {
        this.isOwner = true;
        return true;
      },
      release() {
        releases += 1;
        this.isOwner = false;
      },
      close() {},
    };
    const controller = new hostedController.HostedControllerImpl(transport, {
      ownsTransport: false,
      ownershipFactory: () => ownership,
    });
    const firstLoad = controller.load("fixture-variant", 2048);
    const firstRequest = await waitFor(() => worker.posted[0]);
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: firstRequest.requestId,
      sequence: 1,
      generationId: null,
      event: "status",
      payload: {
        lifecycle: "ready",
        modelVariantId: "fixture-variant",
        contextTokens: 2048,
        selectedModelVariantId: "fixture-variant",
        installedModelVariantIds: ["fixture-variant"],
        installedPackIds: [],
        loadRecords: [],
        download: { phase: "idle", startedAt: null, finishedAt: null, error: null },
        generation: { phase: "idle", startedAt: null, finishedAt: null, error: null },
        fitting: { phase: "idle", startedAt: null, finishedAt: null, error: null },
        error: null,
      },
    });
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: firstRequest.requestId,
      ok: true,
      result: undefined,
    });
    await firstLoad;

    const failedSwitch = controller.load("fixture-uninstalled", 2048);
    const secondRequest = await waitFor(() => worker.posted[1]);
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: secondRequest.requestId,
      ok: false,
      error: {
        code: "MODEL_NOT_INSTALLED",
        message: "fixture model is not installed",
        recoverable: true,
        status: 409,
      },
    });
    await assert.rejects(failedSwitch, /fixture model is not installed/);
    assert.equal(controller.snapshot.lifecycle, "ready");
    assert.equal(controller.snapshot.modelVariantId, "fixture-variant");
    assert.equal(ownership.isOwner, true);
    assert.equal(releases, 0);
    transport.dispose();
  });

  for (const phase of ["ownership", "worker"]) {
    test(`model load deadline covers stalled ${phase} even with worker progress`, async () => {
      const worker = new FakeWorker();
      let closed = false;
      const ownership = {
        isOwner: false,
        acquire() {
          if (phase === "ownership") return new Promise(() => {});
          this.isOwner = true;
          return Promise.resolve(true);
        },
        release() { this.isOwner = false; },
        close() {
          assert.equal(worker.terminated, true, "terminate GPU work before releasing ownership");
          closed = true;
        },
      };
      const controller = new hostedController.HostedControllerImpl(worker, {
        ownershipFactory: () => ownership,
        requestTimeoutMs: 0,
        longOperationIdleTimeoutMs: 50,
        modelLoadTimeoutMs: 80,
      });
      const loading = controller.load("fixture-variant", 2048);
      const rejected = assert.rejects(loading, /model took too long to load/);
      let progress;
      if (phase === "worker") {
        const request = await waitFor(() => worker.posted[0]);
        let sequence = 0;
        progress = setInterval(() => worker.emit({
          protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
          kind: "event", requestId: request.requestId, sequence: ++sequence,
          generationId: null, event: "progress",
          payload: { kind: "model_load", phase: "webllm_initialization", progress: 0.5 },
        }), 10);
      }
      try {
        await rejected;
      } finally {
        clearInterval(progress);
      }
      assert.equal(worker.terminated, true);
      assert.equal(closed, true);
      assert.equal(controller.snapshot.lifecycle, "failed");
      assert.equal(ownership.isOwner, false);
      assert.equal(controller.snapshot.error.code, "MODEL_LOAD_TIMEOUT");
      assert.match(controller.snapshot.error.message, /saved chats and downloads have not been removed/);
      assert.match(userErrors.userFacingError(controller.snapshot.error), /Reload this page and try a smaller model/);
    });
  }

  test("Safari runtime factory enables both load and idle deadlines", async () => {
    const worker = new FakeWorker();
    const navigatorDescriptor = Object.getOwnPropertyDescriptor(globalThis, "navigator");
    const workerDescriptor = Object.getOwnPropertyDescriptor(globalThis, "Worker");
    const setTimer = globalThis.setTimeout;
    const delays = [];
    let bundle;
    try {
      Object.defineProperty(globalThis, "navigator", { configurable: true, value: {
        userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15",
        platform: "MacIntel", maxTouchPoints: 0,
      } });
      Object.defineProperty(globalThis, "Worker", { configurable: true, value: function () { return worker; } });
      globalThis.setTimeout = (callback, delay, ...args) => {
        delays.push(delay);
        return setTimer(callback, delay, ...args);
      };
      bundle = hostedController.createHostedRuntime({
        ownershipFactory: () => ({ isOwner: true, acquire: async () => true, release() {}, close() {} }),
      });
      const loading = bundle.controller.load("fixture-variant", 2048);
      const rejected = assert.rejects(loading, /fixture worker stopped/);
      await waitFor(() => worker.posted[0]);
      assert.ok(delays.includes(600_000), "Safari has an overall model-load deadline");
      assert.ok(delays.includes(180_000), "Safari retains its no-progress watchdog");
      worker.emitError("fixture worker stopped");
      await rejected;
    } finally {
      globalThis.setTimeout = setTimer;
      if (navigatorDescriptor) Object.defineProperty(globalThis, "navigator", navigatorDescriptor);
      else delete globalThis.navigator;
      if (workerDescriptor) Object.defineProperty(globalThis, "Worker", workerDescriptor);
      else delete globalThis.Worker;
      if (bundle) await bundle.dispose();
    }
  });

  test("successful model load cancels its deadline", async () => {
    const worker = new FakeWorker();
    const transport = new browser.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    const controller = new hostedController.HostedControllerImpl(transport, {
      requestTimeoutMs: 0,
      modelLoadTimeoutMs: 40,
      ownershipFactory: () => ({
        isOwner: true, acquire: async () => true, release() {}, close() {},
      }),
    });
    const loading = controller.load("fixture-variant", 2048);
    const request = await waitFor(() => worker.posted[0]);
    worker.emit({ protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response", requestId: request.requestId, ok: true, result: undefined });
    await loading;
    await new Promise(resolve => setTimeout(resolve, 60));
    assert.equal(worker.terminated, false);
    transport.dispose();
  });

  test("controller releases GPU ownership only after cleanup-fatal worker termination", async () => {
    const worker = new FakeWorker();
    const terminatedAtRelease = [];
    const ownership = {
      isOwner: false,
      async acquire() {
        this.isOwner = true;
        return true;
      },
      release() {
        terminatedAtRelease.push(worker.terminated);
        this.isOwner = false;
      },
      close() {},
    };
    const controller = new hostedController.HostedControllerImpl(worker, {
      ownershipFactory: () => ownership,
      requestTimeoutMs: 0,
    });
    const loading = controller.load("fixture-variant", 2048);
    const load = await waitFor(() => worker.posted[0]);
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: load.requestId,
      sequence: 1,
      generationId: null,
      event: "status",
      payload: {
        lifecycle: "loading",
        modelVariantId: "fixture-variant",
        contextTokens: 2048,
        selectedModelVariantId: "fixture-variant",
        installedModelVariantIds: ["fixture-variant"],
        installedPackIds: [],
        loadRecords: [],
        download: { phase: "idle", startedAt: null, finishedAt: null, error: null },
        generation: { phase: "idle", startedAt: null, finishedAt: null, error: null },
        fitting: { phase: "idle", startedAt: null, finishedAt: null, error: null },
        error: null,
      },
    });
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: load.requestId,
      ok: false,
      error: {
        code: "RUNTIME_CLEANUP_FAILED",
        message: "fixture backend could not be disposed",
        recoverable: false,
        status: 500,
      },
    });
    await assert.rejects(loading, /could not be disposed/);
    assert.equal(worker.terminated, true);
    assert.deepEqual(terminatedAtRelease, [true]);
    assert.equal(controller.snapshot.lifecycle, "failed");
    assert.equal(ownership.isOwner, false);
  });

  test("destructive controller failures publish the final internal state", async () => {
    const worker = new FakeWorker();
    const transport = new browser.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    const ownership = {
      isOwner: false,
      async acquire() {
        this.isOwner = true;
        return true;
      },
      release() { this.isOwner = false; },
      close() {},
    };
    const controller = new hostedController.HostedControllerImpl(transport, {
      ownsTransport: false,
      ownershipFactory: () => ownership,
      runDestructiveExclusive: (operation) => operation(),
    });
    const ready = {
      lifecycle: "ready",
      modelVariantId: "fixture-variant",
      contextTokens: 2048,
      selectedModelVariantId: "fixture-variant",
      installedModelVariantIds: ["fixture-variant"],
      installedPackIds: [],
      loadRecords: [],
      download: { phase: "idle", startedAt: null, finishedAt: null, error: null },
      generation: { phase: "idle", startedAt: null, finishedAt: null, error: null },
      fitting: { phase: "idle", startedAt: null, finishedAt: null, error: null },
      error: null,
    };
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: null,
      sequence: 1,
      generationId: null,
      event: "status",
      payload: ready,
    });
    const published = [];
    controller.subscribe((snapshot) => published.push(clone(snapshot)));

    const deleting = controller.deleteModel("fixture-variant");
    const unload = await waitFor(() => worker.posted[0]);
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: unload.requestId,
      sequence: 2,
      generationId: null,
      event: "status",
      payload: {
        ...ready,
        lifecycle: "unloaded",
        modelVariantId: null,
        contextTokens: null,
      },
    });
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: unload.requestId,
      ok: true,
      result: undefined,
    });
    const remove = await waitFor(() => worker.posted[1]);
    assert.equal(published.length, 1);
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: remove.requestId,
      ok: false,
      error: {
        code: "CONTENT_STORE_FAILURE",
        message: "fixture deletion failed",
        recoverable: true,
        status: 500,
      },
    });
    await assert.rejects(deleting, /fixture deletion failed/);
    assert.equal(published.length, 2);
    assert.equal(published.at(-1).lifecycle, "unloaded");
    assert.deepEqual(published.at(-1).installedModelVariantIds, ["fixture-variant"]);
    transport.dispose();
  });

  test("accepted busy takeover stops generation before unloading", async () => {
    const worker = new FakeWorker();
    const transport = new browser.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    let takeoverOptions;
    const ownership = {
      isOwner: false,
      async acquire() {
        this.isOwner = true;
        return true;
      },
      release() { this.isOwner = false; },
      close() {},
    };
    const controller = new hostedController.HostedControllerImpl(transport, {
      ownsTransport: false,
      confirmBusyTakeover: () => true,
      ownershipFactory(options) {
        takeoverOptions = options;
        return ownership;
      },
    });
    const loading = controller.load("fixture-variant", 2048);
    const load = await waitFor(() => worker.posted[0]);
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: load.requestId,
      ok: true,
      result: undefined,
    });
    await loading;
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: null,
      sequence: 1,
      generationId: "generation-a",
      event: "status",
      payload: {
        lifecycle: "ready",
        modelVariantId: "fixture-variant",
        contextTokens: 2048,
        selectedModelVariantId: "fixture-variant",
        installedModelVariantIds: ["fixture-variant"],
        installedPackIds: [],
        loadRecords: [],
        download: { phase: "idle", startedAt: null, finishedAt: null, error: null },
        generation: { phase: "running", startedAt: 1, finishedAt: null, error: null },
        fitting: { phase: "idle", startedAt: null, finishedAt: null, error: null },
        error: null,
      },
    });

    const takeover = takeoverOptions.onTakeoverRequest();
    const stop = await waitFor(() => worker.posted[1]);
    assert.equal(stop.command, "stop");
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: stop.requestId,
      ok: true,
      result: undefined,
    });
    const unload = await waitFor(() => worker.posted[2]);
    assert.equal(unload.command, "unload");
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: unload.requestId,
      ok: true,
      result: undefined,
    });
    assert.equal(await takeover, true);
    assert.deepEqual(worker.posted.map(({ command }) => command), ["load", "stop", "unload"]);
    transport.dispose();
  });

  test("accepted busy takeover awaits fitting cancellation before unloading", async () => {
    const worker = new FakeWorker();
    const transport = new browser.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    let takeoverOptions;
    const ownership = {
      isOwner: false,
      async acquire() {
        this.isOwner = true;
        return true;
      },
      release() { this.isOwner = false; },
      close() {},
    };
    const controller = new hostedController.HostedControllerImpl(transport, {
      ownsTransport: false,
      confirmBusyTakeover: () => true,
      ownershipFactory(options) {
        takeoverOptions = options;
        return ownership;
      },
    });
    const loading = controller.load("fixture-variant", 2048);
    const load = await waitFor(() => worker.posted[0]);
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: load.requestId,
      ok: true,
      result: undefined,
    });
    await loading;
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: null,
      sequence: 1,
      generationId: null,
      event: "status",
      payload: {
        lifecycle: "ready",
        modelVariantId: "fixture-variant",
        contextTokens: 2048,
        selectedModelVariantId: "fixture-variant",
        installedModelVariantIds: ["fixture-variant"],
        installedPackIds: [],
        loadRecords: [],
        download: { phase: "idle", startedAt: null, finishedAt: null, error: null },
        generation: { phase: "idle", startedAt: null, finishedAt: null, error: null },
        fitting: { phase: "running", startedAt: 1, finishedAt: null, error: null },
        error: null,
      },
    });

    const takeover = takeoverOptions.onTakeoverRequest();
    const cancel = await waitFor(() => worker.posted[1]);
    assert.equal(cancel.command, "cancel_fitting");
    assert.equal(worker.posted.length, 2);
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: cancel.requestId,
      ok: true,
      result: undefined,
    });
    const unload = await waitFor(() => worker.posted[2]);
    assert.equal(unload.command, "unload");
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: unload.requestId,
      ok: true,
      result: undefined,
    });
    assert.equal(await takeover, true);
    assert.deepEqual(worker.posted.map(({ command }) => command), [
      "load", "cancel_fitting", "unload",
    ]);
    transport.dispose();
  });

  test("worker transport orders events, detects gaps, and disposes pending calls", async () => {
    const worker = new FakeWorker();
    const transport = new browser.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    const responsePromise = transport.call("check", undefined);
    const request = worker.posted.at(-1);
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId: request.requestId,
      ok: true,
      result: { supported: true },
    });
    assert.deepEqual(await responsePromise, { supported: true });

    const events = [];
    transport.subscribe((event) => events.push(event));
    transport.setSequenceBarrierActive(true);
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: null,
      sequence: 1,
      generationId: null,
      event: "status",
      payload: {},
    });
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: null,
      sequence: 3,
      generationId: "generation-1",
      event: "token",
      payload: { type: "token", text: "uncertain" },
    });
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: null,
      sequence: 4,
      generationId: "generation-1",
      event: "done",
      payload: { type: "done" },
    });
    assert.equal(events.length, 2);
    assert.equal(events[0].sequence, 1);
    assert.equal(events[1].event, "error");
    assert.equal(events[1].payload.scope, "transport");
    assert.equal(events[1].payload.failure.code, "EVENT_SEQUENCE_GAP");

    transport.resumeEvents();
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: null,
      sequence: 5,
      generationId: null,
      event: "status",
      payload: {},
    });
    assert.equal(events.at(-1).sequence, 5);

    const pending = transport.call("load", {
      modelVariantId: "fixture",
      contextTokens: 2048,
    });
    transport.dispose();
    await assert.rejects(pending, /disposed/);
    await assert.rejects(transport.call("check", undefined), /closed/);
    assert.equal(worker.terminated, true);
    assert.ok([...worker.listeners.values()].every((listeners) => listeners.size === 0));
  });

  test("browser event channel correlates tokens and completion by generation id", async () => {
    const worker = new FakeWorker();
    const transport = new browser.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    const client = new browser.BrowserRuntimeClient(transport, { ownsTransport: false });
    const messages = [];
    client.events.subscribe((message) => messages.push(message));
    await client.events.open();

    const emit = (sequence, generationId, event, payload) => worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: null,
      sequence,
      generationId,
      event,
      payload,
    });
    emit(1, "generation-a", "started", {
      type: "started",
      generation_id: "generation-a",
      node_id: "node-a",
      sibling_index: 0,
      sibling_count: 1,
    });
    emit(2, "generation-b", "token", {
      type: "token",
      node_id: "node-b",
      text: "wrong generation",
      thinking: false,
      token_id: 1,
      logprob: -1,
      perplexity: 1,
      raw_index: 0,
    });
    emit(3, "generation-a", "token", {
      type: "token",
      node_id: "node-a",
      text: "accepted",
      thinking: false,
      token_id: 2,
      logprob: -1,
      perplexity: 1,
      raw_index: 0,
    });
    emit(4, "generation-b", "done", {
      type: "done",
      node_id: "node-b",
      result: {},
      sibling_index: 0,
      sibling_count: 1,
    });
    emit(5, "generation-a", "generation_progress", {
      type: "generation_progress", node_id: "node-a", completed: 1, total: 3,
    });
    emit(6, "generation-b", "generation_progress", {
      type: "generation_progress", node_id: "node-b", completed: 2, total: 3,
    });
    emit(7, "generation-a", "done", {
      type: "done",
      node_id: "node-a",
      result: {},
      sibling_index: 0,
      sibling_count: 1,
    });
    emit(8, "generation-a", "token", {
      type: "token",
      node_id: "node-a",
      text: "after completion",
      thinking: false,
      token_id: 3,
      logprob: -1,
      perplexity: 1,
      raw_index: 1,
    });

    assert.deepEqual(messages.map((message) => message.type), ["started", "token", "generation_progress", "done"]);
    assert.equal(messages[1].text, "accepted");
    client.events.close();
    transport.dispose();
  });

  test("disposing a browser client releases shared-transport subscriptions", async () => {
    const worker = new FakeWorker();
    const transport = new browser.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    const first = new browser.BrowserRuntimeClient(transport, { ownsTransport: false });
    const firstMessages = [];
    first.events.subscribe((message) => firstMessages.push(message));
    await first.events.open();
    await first.dispose();
    await assert.rejects(first.events.open(), /disposed/);

    const replacement = new browser.BrowserRuntimeClient(transport, { ownsTransport: false });
    const replacementMessages = [];
    replacement.events.subscribe((message) => replacementMessages.push(message));
    await replacement.events.open();
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: null,
      sequence: 1,
      generationId: "generation-1",
      event: "started",
      payload: {
        type: "started",
        generation_id: "generation-1",
        node_id: "node-1",
        sibling_index: 0,
        sibling_count: 1,
      },
    });

    assert.deepEqual(firstMessages, []);
    assert.equal(replacementMessages.length, 1);
    await replacement.dispose();
    transport.dispose();
  });

  test("browser event channel pauses on sequence gaps until snapshot acknowledgement", async () => {
    const worker = new FakeWorker();
    const transport = new browser.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    const client = new browser.BrowserRuntimeClient(transport, { ownsTransport: false });
    const messages = [];
    client.events.subscribe((message) => messages.push(message));
    await client.events.open();

    const emit = (sequence, generationId, event, payload) => worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: null,
      sequence,
      generationId,
      event,
      payload,
    });
    emit(1, "generation-a", "started", {
      type: "started",
      generation_id: "generation-a",
      node_id: "node-a",
      sibling_index: 0,
      sibling_count: 1,
    });
    emit(3, "generation-a", "token", {
      type: "token",
      node_id: "node-a",
      text: "gap token",
    });
    emit(4, "generation-a", "token", {
      type: "token",
      node_id: "node-a",
      text: "paused token",
    });
    assert.deepEqual(messages.map((message) => message.type), ["started", "error"]);
    assert.equal(messages[1].code, "EVENT_SEQUENCE_GAP");

    client.events.acknowledgeSnapshot();
    emit(5, "generation-b", "started", {
      type: "started",
      generation_id: "generation-b",
      node_id: "node-b",
      sibling_index: 0,
      sibling_count: 1,
    });
    emit(6, "generation-b", "token", {
      type: "token",
      node_id: "node-b",
      text: "after snapshot",
      thinking: false,
      token_id: 1,
      logprob: -1,
      perplexity: 1,
      raw_index: 0,
    });
    assert.deepEqual(
      messages.map((message) => message.type),
      ["started", "error", "started", "token"],
    );
    assert.equal(messages.at(-1).text, "after snapshot");
    client.events.close();
    transport.dispose();
  });

  test("closed browser channels do not retain a stale sequence barrier", async () => {
    const worker = new FakeWorker();
    const transport = new browser.WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
    const client = new browser.BrowserRuntimeClient(transport, { ownsTransport: false });
    const messages = [];
    client.events.subscribe((message) => messages.push(message));

    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: null,
      sequence: 2,
      generationId: null,
      event: "status",
      payload: {},
    });
    await client.events.open();
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId: null,
      sequence: 3,
      generationId: "generation-a",
      event: "started",
      payload: {
        type: "started",
        generation_id: "generation-a",
        node_id: "node-a",
        sibling_index: 0,
        sibling_count: 1,
      },
    });
    assert.deepEqual(messages.map((message) => message.type), ["started"]);
    client.events.close();
    transport.dispose();
  });

  test("stream commands do not inherit the ordinary RPC timeout", async () => {
    const worker = new FakeWorker();
    const transport = new browser.WorkerRpcTransport(worker, { requestTimeoutMs: 5 });
    const client = new browser.BrowserRuntimeClient(transport, { ownsTransport: false });
    const messages = [];
    client.events.subscribe((message) => messages.push(message));
    await client.events.open();
    client.events.send({ type: "generate" });
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.deepEqual(messages, []);
    client.events.close();
    transport.dispose();
  });

  test("long-operation idle watchdog terminates a stalled runtime worker", async () => {
    const worker = new FakeWorker();
    const transport = new browser.WorkerRpcTransport(worker, {
      requestTimeoutMs: 5,
      longOperationIdleTimeoutMs: 10,
    });
    const pending = transport.call("generate", { type: "generate" });
    await assert.rejects(pending, /request generate made no progress/);
    assert.equal(worker.terminated, true);
  });

  test("worker progress refreshes the long-operation idle watchdog", async () => {
    const worker = new FakeWorker();
    const transport = new browser.WorkerRpcTransport(worker, {
      requestTimeoutMs: 5,
      longOperationIdleTimeoutMs: 30,
    });
    const pending = transport.call("generate", { type: "generate" });
    const requestId = worker.posted[0].requestId;
    await new Promise((resolve) => setTimeout(resolve, 20));
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId,
      sequence: 1,
      generationId: "generation-a",
      event: "token",
      payload: { type: "token" },
    });
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.equal(worker.terminated, false);
    worker.emit({
      protocolVersion: contracts.RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId,
      ok: true,
      result: { done: true },
    });
    assert.deepEqual(await pending, { done: true });
    transport.dispose();
  });

  test("browser event channel distinguishes expected close from worker failure", async () => {
    const expectedWorker = new FakeWorker();
    const expectedTransport = new browser.WorkerRpcTransport(expectedWorker, {
      requestTimeoutMs: 0,
    });
    const expectedClient = new browser.BrowserRuntimeClient(expectedTransport, {
      ownsTransport: false,
    });
    const expectedStates = [];
    expectedClient.events.subscribeState((state) => expectedStates.push(state));
    await expectedClient.events.open();
    expectedClient.events.close();
    assert.deepEqual(expectedStates, [
      { state: "open" },
      { state: "closed", expected: true, reason: null },
    ]);
    assert.equal(expectedClient.events.isOpen, false);
    expectedTransport.dispose();

    const failedWorker = new FakeWorker();
    const failedTransport = new browser.WorkerRpcTransport(failedWorker, {
      requestTimeoutMs: 0,
    });
    const failedClient = new browser.BrowserRuntimeClient(failedTransport, {
      ownsTransport: false,
    });
    const failedStates = [];
    const unsubscribe = failedClient.events.subscribeState((state) => failedStates.push(state));
    await failedClient.events.open();
    failedWorker.emitError("fixture worker crashed");
    assert.deepEqual(failedStates, [
      { state: "open" },
      { state: "closed", expected: false, reason: "fixture worker crashed" },
    ]);
    assert.equal(failedClient.events.isOpen, false);
    assert.equal(failedWorker.terminated, true);
    await assert.rejects(failedClient.events.open(), /fixture worker crashed/);
    unsubscribe();
  });

  for (const { name, run } of tests) {
    await run();
    console.log(`ok - ${name}`);
  }
  console.log(`${tests.length} hosted runtime foundation tests passed`);
} finally {
  await server.close();
}
