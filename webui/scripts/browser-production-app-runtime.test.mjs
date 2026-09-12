import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  classifyInstalledAppState,
  createProductionAppCatalog,
  createProductionAppFailureReport,
  isExpectedOfflineCatalogFailure,
  digestCanonical,
  isOfflineServiceWorkerMaintenanceRequest,
  isPublishedSaeDescriptionUrl,
  isVerifiedLocalArtifactAbort,
  legacyWebLlmPersistentState,
  LEGACY_WEBLLM_CACHE_STORAGE_NAMES,
  LEGACY_WEBLLM_INDEXED_DB_NAMES,
  parseBrowserProductionAppArguments,
  parseSingleByteRange,
  requiredFeaturesForQuantization,
  stripHostedDevelopmentFixture,
  validatorRuntimeLockArguments,
} from "./browser-production-app-runtime-contract.mjs";

test("provider metadata access is limited to the selected model's exact public SAE dictionary", () => {
  const url = "https://www.neuronpedia.org/api/feature/gemma-3-1b-it/13-gemmascope-2-res-16k/396";
  assert.equal(isPublishedSaeDescriptionUrl(url, "gemma3-1b-instruct"), true);
  for (const invalid of [url + "?prompt=hello", url + "/extra", url.replace("/396", "/16384"), url.replace("/396", "/-1"), url.replace("www.neuronpedia.org", "example.com")]) {
    assert.equal(isPublishedSaeDescriptionUrl(invalid, "gemma3-1b-instruct"), false);
  }
  assert.equal(isPublishedSaeDescriptionUrl(url, "gemma3-270m-instruct"), false);
});

test("authoritative installed app state accepts verified onboarding or the exact loaded model", () => {
  assert.equal(classifyInstalledAppState({
    verifiedOnboarding: true,
    loadedModelVariantId: null,
    expectedModelVariantId: "qwen3-1.7b-q4f16_1",
  }), "verified-onboarding");
  assert.equal(classifyInstalledAppState({
    verifiedOnboarding: false,
    loadedModelVariantId: "qwen3-1.7b-q4f16_1",
    expectedModelVariantId: "qwen3-1.7b-q4f16_1",
  }), "loaded-workbench");
  assert.equal(classifyInstalledAppState({
    verifiedOnboarding: false,
    loadedModelVariantId: "qwen3-4b-q4f16_1",
    expectedModelVariantId: "qwen3-1.7b-q4f16_1",
  }), null);
});

test("production requirements select features from compute dtype", () => {
  assert.deepEqual(requiredFeaturesForQuantization("q4f16_1"), ["shader-f16"]);
  assert.deepEqual(requiredFeaturesForQuantization("q4f32_1"), []);
  assert.deepEqual(requiredFeaturesForQuantization("q0f32"), []);
  assert.throws(
    () => requiredFeaturesForQuantization("q8f16"),
    /unsupported browser quantization/,
  );
});

test("production audit rejects only exact pinned WebLLM persistent stores", () => {
  assert.deepEqual(legacyWebLlmPersistentState({
    cacheNames: [
      ...LEGACY_WEBLLM_CACHE_STORAGE_NAMES,
      "webllm/model-other",
      "unrelated-cache",
    ],
    databaseNames: [
      ...LEGACY_WEBLLM_INDEXED_DB_NAMES,
      "webllm/config-backup",
      "unrelated-database",
    ],
  }), {
    cacheNames: [...LEGACY_WEBLLM_CACHE_STORAGE_NAMES],
    databaseNames: [...LEGACY_WEBLLM_INDEXED_DB_NAMES],
  });
});

test("offline audit isolates browser service-worker maintenance", () => {
  assert.equal(isOfflineServiceWorkerMaintenanceRequest("GET /sw.js"), true);
  assert.equal(
    isOfflineServiceWorkerMaintenanceRequest("GET /workbox-7fb1381d.js"),
    true,
  );
  assert.equal(isOfflineServiceWorkerMaintenanceRequest("GET /app"), false);
  assert.equal(isOfflineServiceWorkerMaintenanceRequest("POST /sw.js"), false);
  assert.equal(isOfflineServiceWorkerMaintenanceRequest("GET /model.wasm"), false);
});

test("offline audit permits only failed refreshes of the exact signed catalog pair", () => {
  const urls = {
    catalogUrl: "https://huggingface.co/example/catalog.json",
    signatureUrl: "https://huggingface.co/example/catalog.sig.json",
  };
  const request = {
    url: urls.catalogUrl,
    errorText: "net::ERR_INTERNET_DISCONNECTED",
    phase: "offline",
  };
  assert.equal(isExpectedOfflineCatalogFailure(request, urls), true);
  assert.equal(isExpectedOfflineCatalogFailure({ ...request, phase: "load" }, urls), false);
  assert.equal(isExpectedOfflineCatalogFailure({ ...request, url: "https://example.com/prompt" }, urls), false);
  assert.equal(isExpectedOfflineCatalogFailure({ ...request, errorText: "net::ERR_FAILED" }, urls), false);
});

test("production-app arguments require only the four artifact directories", () => {
  const parsed = parseBrowserProductionAppArguments([
    "--model-directory", "/tmp/model",
    "--core-directory", "/tmp/core",
    "--jlens-directory", "/tmp/jlens",
    "--sae-directory", "/tmp/sae",
    "--runtime-lock", "/tmp/candidate-runtime-lock.json",
  ]);
  assert.equal(parsed.modelLibrary, "/tmp/model/model.wasm");
  assert.equal(parsed.modelId, "qwen3-1.7b");
  assert.equal(parsed.contextTokens, 2048);
  assert.equal(parsed.browserChannel, "chrome");
  assert.equal(parsed.browserFlags, "compatibility");
  assert.equal(parsed.runtimeLock, "/tmp/candidate-runtime-lock.json");
});

test("production-app arguments reject unsafe omissions and unsupported browsers", () => {
  assert.throws(() => parseBrowserProductionAppArguments([]), /--model-directory is required/);
  assert.throws(() => parseBrowserProductionAppArguments([
    "--model-directory", "/tmp/model",
    "--core-directory", "/tmp/core",
    "--jlens-directory", "/tmp/jlens",
    "--sae-directory", "/tmp/sae",
    "--browser-channel", "chromium",
  ]), /chrome or msedge/);
});

test("candidate validators receive the exact runtime-lock override", () => {
  assert.deepEqual(validatorRuntimeLockArguments(null), []);
  assert.deepEqual(
    validatorRuntimeLockArguments("/tmp/candidate-runtime-lock.json"),
    ["--runtime-lock", "/tmp/candidate-runtime-lock.json"],
  );
  assert.throws(() => validatorRuntimeLockArguments(""), /path is invalid/);
});

test("single byte ranges close exactly over the signed object", () => {
  assert.deepEqual(parseSingleByteRange(null, 10), { status: 200, start: 0, end: 9 });
  assert.deepEqual(parseSingleByteRange("bytes=0-3", 10), { status: 206, start: 0, end: 3 });
  assert.deepEqual(parseSingleByteRange("bytes=4-", 10), { status: 206, start: 4, end: 9 });
  assert.throws(() => parseSingleByteRange("bytes=10-", 10), /invalid Range/);
  assert.throws(() => parseSingleByteRange("bytes=-4", 10), /unsupported Range/);
});

test("only verified local artifact consumer aborts are recoverable", () => {
  const sha256 = "a".repeat(64);
  const input = {
    url: `https://127.0.0.1:4321/__artifacts/${sha256}`,
    errorText: "net::ERR_ABORTED",
    artifactOrigin: "https://127.0.0.1:4321",
    expectedHashes: new Set([sha256]),
    downloadedHashes: new Set([sha256]),
  };
  assert.equal(isVerifiedLocalArtifactAbort(input), true);
  assert.equal(isVerifiedLocalArtifactAbort({
    ...input,
    errorText: "net::ERR_FAILED",
  }), false);
  assert.equal(isVerifiedLocalArtifactAbort({
    ...input,
    downloadedHashes: new Set(),
  }), false);
  assert.equal(isVerifiedLocalArtifactAbort({
    ...input,
    url: `https://example.com/__artifacts/${sha256}`,
  }), false);
});

test("production-app catalog closes the runtime lock and pack bindings", () => {
  const revision = "a".repeat(40);
  const lock = {
    id: "qwen3-1.7b",
    sourceRepository: "Qwen/Qwen3-1.7B",
    sourceRevision: "b".repeat(40),
    convertedRepository: "logitsml/drowse-web-qwen3-1.7b",
    manifestSha256: "1".repeat(64),
    librarySha256: "2".repeat(64),
    tokenizerSha256: "3".repeat(64),
    chatTemplateSha256: "4".repeat(64),
    hiddenSize: 2048,
    layerMap: [0, 1],
    quantization: "q4f16_1",
    contextProfiles: [2048, 4096],
    structuredHookProfile: "standard-v3",
    thinkingProfile: {
      start: "<think>",
      end: "</think>",
      startsInThinking: false,
      startTokenIds: [151667],
      endTokenIds: [151668],
    },
  };
  const files = (role, sha256, path = `${role}.bin`) => [{
    path,
    role,
    bytes: 10,
    sha256,
  }];
  const modelFiles = [
    ...files("configuration", "5".repeat(64)),
    ...files("converted_manifest", lock.manifestSha256),
    ...files("tokenizer", lock.tokenizerSha256),
    ...files("chat_template", lock.chatTemplateSha256),
    ...files("weight", "6".repeat(64)),
    ...files("model_library", lock.librarySha256),
  ];
  const result = createProductionAppCatalog({
    runtimeLock: { runtimeAbi: "drowse-web-runtime-v1", hookAbi: "post-block-residual-v4" },
    lock,
    revision,
    modelFiles,
    coreFiles: files("core_pack", "7".repeat(64)),
    jlensFiles: files("instrument", "8".repeat(64)),
    saeFiles: files("instrument", "9".repeat(64)),
    contextTokens: 2048,
    issuedAt: "2026-08-29T00:00:00.000Z",
    expiresAt: "2026-08-30T00:00:00.000Z",
  });
  const variant = result.document.models[0].variants[0];
  assert.equal(variant.runtimeIdentitySha256, digestCanonical(variant.runtimeIdentity));
  assert.deepEqual(variant.contextProfiles.map((profile) => profile.contextTokens), [2048, 4096]);
  assert.deepEqual(variant.packs[0].compatibleContextBindingSha256,
    variant.contextProfiles.map((profile) => profile.bindingSha256));
  assert.deepEqual(variant.packs[1].compatibleContextBindingSha256,
    variant.contextProfiles.map((profile) => profile.bindingSha256));
  assert.match(variant.files[0].url,
    /^https:\/\/huggingface\.co\/logitsml\/drowse-web-qwen3-1\.7b\/resolve\/a{40}\//u);
  assert.equal(variant.packs[1].sourceRepository,
    "logitsml/drowse-web-qwen3-1.7b-instruments");
  assert.equal(variant.requirements.limits.maxComputeWorkgroupSizeX, 256);
  const exactReadout = createProductionAppCatalog({
    runtimeLock: { runtimeAbi: "drowse-web-runtime-v1", hookAbi: "post-block-residual-v4" },
    lock: { ...lock, structuredHookProfile: "standard-v3" },
    revision,
    modelFiles,
    coreFiles: files("core_pack", "7".repeat(64)),
    jlensFiles: files("instrument", "8".repeat(64)),
    saeFiles: files("instrument", "9".repeat(64)),
    contextTokens: 2048,
    issuedAt: "2026-08-29T00:00:00.000Z",
    expiresAt: "2026-08-30T00:00:00.000Z",
  });
  assert.equal(
    exactReadout.document.models[0].variants[0].requirements.limits
      .maxComputeWorkgroupSizeX,
    256,
  );
  assert.equal(
    exactReadout.document.models[0].variants[0].requirements.limits
      .maxComputeInvocationsPerWorkgroup,
    256,
  );
  const qwen4 = createProductionAppCatalog({
    runtimeLock: { runtimeAbi: "drowse-web-runtime-v1", hookAbi: "post-block-residual-v4" },
    lock: {
      ...lock,
      id: "qwen3-4b",
      sourceRepository: "Qwen/Qwen3-4B",
      convertedRepository: "logitsml/drowse-web-qwen3-4b",
      quantization: "q4f16_1",
      structuredHookProfile: "standard-v3",
    },
    revision,
    modelFiles,
    coreFiles: files("core_pack", "7".repeat(64)),
    jlensFiles: files("instrument", "8".repeat(64)),
    saeFiles: files("instrument", "9".repeat(64)),
    contextTokens: 2048,
    issuedAt: "2026-08-29T00:00:00.000Z",
    expiresAt: "2026-08-30T00:00:00.000Z",
  });
  assert.equal(qwen4.document.models[0].displayName, "Qwen3 4B");
  assert.deepEqual(qwen4.document.models[0].languages, ["en", "zh"]);
  assert.equal(qwen4.document.models[0].license, "Apache-2.0");
  assert.equal(qwen4.document.models[0].variants[0].tier, "quality");
  assert.equal(qwen4.document.models[0].variants[0].id, "qwen3-4b-q4f16_1");
  assert.deepEqual(qwen4.document.models[0].variants[0].requirements.features, ["shader-f16"]);
  const unquantizedGemma = createProductionAppCatalog({
    runtimeLock: { runtimeAbi: "drowse-web-runtime-v1", hookAbi: "post-block-residual-v4" },
    lock: { ...lock, id: "gemma3-270m-instruct", quantization: "q0f32", thinkingProfile: null },
    revision,
    modelFiles: modelFiles.map((file) => file.role === "weight" ? { ...file, bytes: 671_088_640 } : file),
    coreFiles: files("core_pack", "7".repeat(64)),
    jlensFiles: files("instrument", "8".repeat(64)),
    saeFiles: files("instrument", "9".repeat(64)),
    contextTokens: 2048,
    issuedAt: "2026-08-29T00:00:00.000Z",
    expiresAt: "2026-08-30T00:00:00.000Z",
  }).document.models[0].variants[0];
  assert.equal(unquantizedGemma.id, "gemma3-270m-instruct-q0f32");
  assert.deepEqual(unquantizedGemma.requirements.features, []);
  assert.equal(unquantizedGemma.requirements.limits.maxBufferSize, 1_073_741_824);
  assert.equal(unquantizedGemma.requirements.limits.maxStorageBufferBindingSize, 1_073_741_824);
});

test("temporary production entry strips only the guarded development fixture", () => {
  const source = `
import { mount } from "svelte";
let controller: import("./types").Controller | undefined;
if (import.meta.env.DEV && new URLSearchParams(location.search).get("fixture") === "1") {
  const { createWorkerFixtureRuntime } = await import("./workerFixtureRuntime");
  controller = createWorkerFixtureRuntime(); // ?fixture
}

let component;
component = HostedRoot;
const app = mount(HostedRoot, { props: { controller } });
`;
  const stripped = stripHostedDevelopmentFixture(source);
  assert.doesNotMatch(stripped, /fixture|workerFixtureRuntime|import\.meta\.env\.DEV/u);
  assert.match(stripped, /const controller = undefined;\n\nlet component;/u);
  assert.throws(
    () => stripHostedDevelopmentFixture("const app = mount(HostedRoot);"),
    /fixture boundary changed/u,
  );
});

test("production entry strips current fixtures without removing bootstrap recovery or routing", async () => {
  const source = await readFile(new URL("../hosted/main.ts", import.meta.url), "utf8");
  const stripped = stripHostedDevelopmentFixture(source);
  assert.doesNotMatch(stripped, /fixtureHostedRuntime|workerFixtureRuntime|layoutFixture|searchParams/u);
  assert.match(stripped, /recoverFromChunkLoadError\(error, "bootstrap"\)/u);
  assert.match(stripped, /component = HostedRoot/u);
  assert.match(stripped, /component = \(await import\("\.\.\/src\/hosted\/ui\/LandingRoot\.svelte"\)\)\.default/u);
});

test("failure reports can never be mistaken for release evidence", () => {
  const error = Object.assign(new Error("generation exceeded 300000 ms"), {
    code: "PRODUCTION_APP_GENERATION_TIMEOUT",
  });
  assert.deepEqual(createProductionAppFailureReport({
    stage: "online generation",
    error,
    measurements: { signedFiles: 56 },
  }), {
    schemaVersion: 1,
    toolVersion: "drowse-production-app-e2e-v1",
    releaseEvidence: false,
    passed: false,
    stage: "online generation",
    error: {
      name: "Error",
      code: "PRODUCTION_APP_GENERATION_TIMEOUT",
      message: "generation exceeded 300000 ms",
    },
    measurements: { signedFiles: 56 },
  });
});
