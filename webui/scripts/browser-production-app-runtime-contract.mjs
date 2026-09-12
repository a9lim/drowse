import { createHash } from "node:crypto";

export const BROWSER_PRODUCTION_APP_TOOL_VERSION = "drowse-production-app-e2e-v1";

export const LEGACY_WEBLLM_CACHE_STORAGE_NAMES = Object.freeze([
  "webllm/model",
  "webllm/config",
  "webllm/wasm",
  "tvmjs",
  "tvmjs-cos-hash-meta",
]);

export const LEGACY_WEBLLM_INDEXED_DB_NAMES = Object.freeze([
  "webllm/model",
  "webllm/config",
  "webllm/wasm",
  "tvmjs",
]);

const PRODUCTION_APP_MODEL_METADATA = Object.freeze({
  "gemma3-270m-instruct": Object.freeze({
    displayName: "Gemma 3 270M",
    tier: "fastest",
    languages: Object.freeze(["en"]),
    license: "Gemma",
  }),
  "gemma3-1b-instruct": Object.freeze({
    displayName: "Gemma 3 1B",
    tier: "balanced",
    languages: Object.freeze(["en"]),
    license: "Gemma",
  }),
  "qwen3-1.7b": Object.freeze({
    displayName: "Qwen3 1.7B",
    tier: "balanced",
    languages: Object.freeze(["en", "zh"]),
    license: "Apache-2.0",
  }),
  "qwen3-4b": Object.freeze({
    displayName: "Qwen3 4B",
    tier: "quality",
    languages: Object.freeze(["en", "zh"]),
    license: "Apache-2.0",
  }),
});

export function legacyWebLlmPersistentState({ cacheNames, databaseNames }) {
  const cacheAllowlist = new Set(LEGACY_WEBLLM_CACHE_STORAGE_NAMES);
  const databaseAllowlist = new Set(LEGACY_WEBLLM_INDEXED_DB_NAMES);
  return {
    cacheNames: cacheNames.filter((name) => cacheAllowlist.has(name)),
    databaseNames: databaseNames.filter((name) => databaseAllowlist.has(name)),
  };
}

export function classifyInstalledAppState({
  verifiedOnboarding,
  loadedModelVariantId,
  expectedModelVariantId,
}) {
  if (typeof expectedModelVariantId !== "string" || !expectedModelVariantId.trim()) {
    throw new Error("expected model variant ID is required");
  }
  if (verifiedOnboarding) return "verified-onboarding";
  return loadedModelVariantId === expectedModelVariantId ? "loaded-workbench" : null;
}

export function requiredFeaturesForQuantization(quantization) {
  if (quantization === "q4f16_1") return ["shader-f16"];
  if (quantization === "q4f32_1" || quantization === "q0f32") return [];
  throw new Error(`unsupported browser quantization ${quantization}`);
}

export function validatorRuntimeLockArguments(runtimeLock) {
  if (runtimeLock === null) return [];
  if (typeof runtimeLock !== "string" || !runtimeLock.trim()) {
    throw new Error("runtime-lock path is invalid");
  }
  return ["--runtime-lock", runtimeLock];
}

export function isOfflineServiceWorkerMaintenanceRequest(request) {
  return request === "GET /sw.js" || /^GET \/workbox-[A-Za-z0-9_-]+\.js$/u.test(request);
}

export function isExpectedCatalogRefreshFailure(
  request,
  { catalogUrl, signatureUrl },
) {
  if (request?.url !== catalogUrl && request?.url !== signatureUrl) return false;
  if (request.intentionalReload === true && request.errorText === "net::ERR_ABORTED") return true;
  return request.phase === "offline" && /^net::ERR_(?:INTERNET_DISCONNECTED|ABORTED)$/u.test(request.errorText ?? "");
}

export function trackIntentionalReload(page) {
  const inFlight = new Set();
  const closingRequests = new WeakSet();
  let closing = false;
  page.on("request", request => {
    inFlight.add(request);
    if (closing) closingRequests.add(request);
  });
  page.on("requestfinished", request => inFlight.delete(request));
  page.on("requestfailed", request => inFlight.delete(request));
  page.on("framenavigated", frame => { if (frame === page.mainFrame()) closing = false; });
  return {
    begin() {
      closing = true;
      for (const request of inFlight) closingRequests.add(request);
      return () => { closing = false; };
    },
    includes: request => closingRequests.has(request),
  };
}

export function isPublishedSaeDescriptionUrl(url, modelId) {
  const dictionary = {
    "gemma3-270m-instruct": "gemma-3-270m-it/12-gemmascope-2-res-16k",
    "gemma3-1b-instruct": "gemma-3-1b-it/13-gemmascope-2-res-16k",
  }[modelId];
  if (!dictionary || typeof url !== "string") return false;
  const prefix = `https://www.neuronpedia.org/api/feature/${dictionary}/`;
  if (!url.startsWith(prefix)) return false;
  const id = url.slice(prefix.length);
  return /^(?:0|[1-9]\d*)$/u.test(id) && Number(id) < 16384;
}

export function createProductionAppFailureReport({ stage, error, measurements = {} }) {
  if (typeof stage !== "string" || !stage.trim()) throw new Error("failure stage is required");
  const failure = error instanceof Error ? error : new Error(String(error));
  const code = typeof failure.code === "string" ? failure.code : null;
  return {
    schemaVersion: 1,
    toolVersion: BROWSER_PRODUCTION_APP_TOOL_VERSION,
    releaseEvidence: false,
    passed: false,
    stage: stage.trim(),
    error: {
      name: failure.name,
      code,
      message: failure.message,
    },
    measurements: structuredClone(measurements),
  };
}

export function parseBrowserProductionAppArguments(args) {
  const result = {
    browserChannel: "chrome",
    browserFlags: "compatibility",
    contextTokens: 2048,
    coreDirectory: null,
    headed: false,
    jlensDirectory: null,
    jlensWord: "ocean",
    maxTokens: 4,
    modelDirectory: null,
    modelId: "qwen3-1.7b",
    modelLibrary: null,
    output: null,
    runtimeLock: null,
    saeDirectory: null,
    saeFeature: 0,
    timeoutMs: 900_000,
  };
  const fields = new Map([
    ["--browser-channel", "browserChannel"],
    ["--browser-flags", "browserFlags"],
    ["--context-tokens", "contextTokens"],
    ["--core-directory", "coreDirectory"],
    ["--jlens-directory", "jlensDirectory"],
    ["--jlens-word", "jlensWord"],
    ["--max-tokens", "maxTokens"],
    ["--model-directory", "modelDirectory"],
    ["--model-id", "modelId"],
    ["--model-library", "modelLibrary"],
    ["--output", "output"],
    ["--runtime-lock", "runtimeLock"],
    ["--sae-directory", "saeDirectory"],
    ["--sae-feature", "saeFeature"],
    ["--timeout-ms", "timeoutMs"],
  ]);
  const numeric = new Set(["contextTokens", "maxTokens", "saeFeature", "timeoutMs"]);
  const seen = new Set();
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--headed") {
      if (seen.has(argument)) throw new Error(`${argument} may be supplied only once`);
      seen.add(argument);
      result.headed = true;
      continue;
    }
    const field = fields.get(argument);
    if (!field) throw new Error(`unexpected argument ${argument}`);
    if (seen.has(argument)) throw new Error(`${argument} may be supplied only once`);
    seen.add(argument);
    const value = args[++index];
    if (!value || value.startsWith("--")) throw new Error(`${argument} requires a value`);
    result[field] = numeric.has(field) ? Number(value) : value;
  }
  for (const field of ["modelDirectory", "coreDirectory", "jlensDirectory", "saeDirectory"]) {
    if (result[field] === null) throw new Error(`${flagFor(field)} is required`);
  }
  if (result.modelLibrary === null) result.modelLibrary = `${result.modelDirectory}/model.wasm`;
  if (!["chrome", "msedge"].includes(result.browserChannel)) {
    throw new Error("--browser-channel must be chrome or msedge");
  }
  if (!["compatibility", "normal"].includes(result.browserFlags)) {
    throw new Error("--browser-flags must be compatibility or normal");
  }
  positiveInteger(result.contextTokens, "--context-tokens");
  positiveInteger(result.maxTokens, "--max-tokens");
  nonnegativeInteger(result.saeFeature, "--sae-feature");
  if (!Number.isSafeInteger(result.timeoutMs) || result.timeoutMs < 60_000) {
    throw new Error("--timeout-ms must be an integer of at least 60000");
  }
  if (!result.jlensWord.trim() || result.jlensWord.length > 128) {
    throw new Error("--jlens-word must be a non-empty single-token candidate");
  }
  return Object.freeze({ ...result, jlensWord: result.jlensWord.trim() });
}

export function createProductionAppCatalog({
  runtimeLock,
  lock,
  revision,
  modelFiles,
  coreFiles,
  jlensFiles,
  saeFiles,
  contextTokens,
  issuedAt,
  expiresAt,
}) {
  if (!/^[a-f0-9]{40}$/u.test(revision)) throw new Error("catalog revision must be a commit hash");
  if (!lock.contextProfiles.includes(contextTokens)) {
    throw new Error(`context ${contextTokens} is not runtime-locked for ${lock.id}`);
  }
  const modelMetadata = PRODUCTION_APP_MODEL_METADATA[lock.id];
  if (!modelMetadata) throw new Error(`production-app metadata is missing for ${lock.id}`);
  const runtimeIdentity = {
    sourceModel: lock.sourceRepository,
    sourceRevision: lock.sourceRevision,
    convertedManifestSha256: lock.manifestSha256,
    quantization: lock.quantization,
    tokenizerSha256: lock.tokenizerSha256,
    chatTemplateSha256: lock.chatTemplateSha256,
    modelLibrarySha256: lock.librarySha256,
    runtimeAbi: runtimeLock.runtimeAbi,
    hookAbi: runtimeLock.hookAbi,
    hiddenSize: lock.hiddenSize,
    layerMap: [...lock.layerMap],
  };
  const runtimeIdentitySha256 = digestCanonical(runtimeIdentity);
  const contextProfiles = lock.contextProfiles.map((tokens) => ({
    contextTokens: tokens,
    bindingSha256: digestCanonical({ contextTokens: tokens, runtimeIdentitySha256 }),
    minimumCalibrationScore: null,
    minimumDeviceMemoryGiB: null,
    expectedPrefillTokensPerSecond: null,
    expectedDecodeTokensPerSecond: null,
    measuredDevices: 1,
  }));
  const bindings = contextProfiles.map((profile) => profile.bindingSha256);
  const selectedBinding = contextProfiles.find((profile) =>
    profile.contextTokens === contextTokens
  )?.bindingSha256;
  if (!selectedBinding) throw new Error("selected context binding is missing");
  const modelRepository = lock.convertedRepository;
  const instrumentRepository = `${lock.convertedRepository}-instruments`;
  const files = modelFiles.map((file) => catalogFile(file, modelRepository, revision));
  const core = catalogPack({
    id: `${lock.id}-core`,
    kind: "core",
    displayName: "Core geometry",
    license: "AGPL-3.0-or-later",
    repository: modelRepository,
    revision,
    required: true,
    runtimeIdentitySha256,
    compatibleContextBindingSha256: bindings,
    files: coreFiles,
  });
  const jlens = catalogPack({
    id: `${lock.id}-jlens`,
    kind: "jlens",
    displayName: "J-lens",
    license: "AGPL-3.0-or-later",
    repository: instrumentRepository,
    revision,
    required: false,
    runtimeIdentitySha256,
    compatibleContextBindingSha256: bindings,
    files: jlensFiles,
  });
  const sae = catalogPack({
    id: `${lock.id}-sae`,
    kind: "sae",
    displayName: "SAE",
    license: "AGPL-3.0-or-later",
    repository: instrumentRepository,
    revision,
    required: false,
    runtimeIdentitySha256,
    compatibleContextBindingSha256: [selectedBinding],
    files: saeFiles,
  });
  const variantId = `${lock.id}-${lock.quantization}`;
  const weightBufferBytes = 2 ** Math.ceil(Math.log2(Math.max(
    134_217_728, ...modelFiles.filter((file) => file.role === "weight").map((file) => file.bytes),
  )));
  const document = {
    schemaVersion: 1,
    sequence: 1,
    issuedAt,
    expiresAt,
    runtimeAbi: runtimeLock.runtimeAbi,
    models: [{
      id: lock.id,
      displayName: modelMetadata.displayName,
      description: "Local production-app integration run",
      sourceUrl: `https://huggingface.co/${lock.sourceRepository}`,
      license: modelMetadata.license,
      languages: [...modelMetadata.languages],
      variants: [{
        id: variantId,
        tier: modelMetadata.tier,
        structuredHookProfile: lock.structuredHookProfile,
        thinkingProfile: lock.thinkingProfile,
        contextProfiles,
        downloadBytes: sumBytes(files),
        requiredCorePackBytes: core.bytes,
        requirements: {
          features: requiredFeaturesForQuantization(lock.quantization),
          limits: {
            maxBufferSize: weightBufferBytes,
            maxStorageBufferBindingSize: weightBufferBytes,
            maxStorageBuffersPerShaderStage: 10,
            maxComputeWorkgroupStorageSize: 32_768,
            ...(lock.structuredHookProfile === "standard-v3"
              ? {
                  maxComputeWorkgroupSizeX: 256,
                  maxComputeInvocationsPerWorkgroup: 256,
                }
              : {}),
          },
        },
        runtimeIdentity,
        runtimeIdentitySha256,
        files,
        packs: [core, jlens, sae],
      }],
    }],
  };
  return Object.freeze({
    document,
    variantId,
    runtimeIdentity,
    runtimeIdentitySha256,
    repositories: Object.freeze({ model: modelRepository, instruments: instrumentRepository }),
  });
}

export function parseSingleByteRange(value, totalBytes) {
  positiveInteger(totalBytes, "artifact byte count");
  if (value === null || value === undefined || value === "") {
    return { status: 200, start: 0, end: totalBytes - 1 };
  }
  const match = /^bytes=([0-9]+)-([0-9]*)$/u.exec(value);
  if (!match) throw new Error(`unsupported Range header ${value}`);
  const start = Number(match[1]);
  const end = match[2] ? Number(match[2]) : totalBytes - 1;
  if (
    !Number.isSafeInteger(start) || !Number.isSafeInteger(end) ||
    start < 0 || start >= totalBytes || end < start || end >= totalBytes
  ) throw new Error(`invalid Range header ${value} for ${totalBytes} bytes`);
  return { status: 206, start, end };
}

export function isVerifiedLocalArtifactAbort({
  url,
  errorText,
  artifactOrigin,
  expectedHashes,
  downloadedHashes,
}) {
  if (errorText !== "net::ERR_ABORTED") return false;
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return false;
  }
  if (parsed.origin !== artifactOrigin) return false;
  const prefix = "/__artifacts/";
  if (!parsed.pathname.startsWith(prefix)) return false;
  const sha256 = parsed.pathname.slice(prefix.length);
  return /^[0-9a-f]{64}$/u.test(sha256) &&
    expectedHashes.has(sha256) && downloadedHashes.has(sha256);
}

export function digestCanonical(value) {
  return createHash("sha256").update(canonicalJson(value)).digest("hex");
}

export function stripHostedDevelopmentFixture(source) {
  const currentStart = source.indexOf("  const searchParams = new URLSearchParams(window.location.search);");
  const currentEndMarker = '\n  } else if (path === "/") {';
  const currentEnd = source.indexOf(currentEndMarker, currentStart);
  if (currentStart >= 0 && currentEnd > currentStart) {
    const fixtureBlock = source.slice(currentStart, currentEnd);
    for (const marker of ["import.meta.env.DEV", "createWorkerFixtureRuntime", "createFixtureHostedRuntime", "if (layoutFixture)"]) {
      if (!fixtureBlock.includes(marker)) throw new Error(`hosted entry fixture block is missing ${marker}`);
    }
    return `${source.slice(0, currentStart)}  if (path === "/") {${source.slice(currentEnd + currentEndMarker.length)}`;
  }
  const start = source.indexOf("let controller:");
  const end = source.indexOf("\n\nlet component;");
  if (start < 0 || end <= start) throw new Error("hosted entry fixture boundary changed");
  const fixtureBlock = source.slice(start, end);
  for (const marker of [
    "import.meta.env.DEV",
    "createWorkerFixtureRuntime",
    "workerFixtureRuntime",
    "new URLSearchParams",
  ]) {
    if (!fixtureBlock.includes(marker)) {
      throw new Error(`hosted entry fixture block is missing ${marker}`);
    }
  }
  return `${source.slice(0, start)}const controller = undefined;${source.slice(end)}`;
}

function catalogPack({
  id,
  kind,
  displayName,
  license,
  repository,
  revision,
  required,
  runtimeIdentitySha256,
  compatibleContextBindingSha256,
  files,
}) {
  const role = kind === "core" ? "core_pack" : "instrument";
  const catalogFiles = files.map((file) =>
    catalogFile({ ...file, role }, repository, revision)
  );
  return {
    id,
    kind,
    displayName,
    license,
    sourceRepository: repository,
    sourceRevision: revision,
    bytes: sumBytes(catalogFiles),
    required,
    runtimeIdentitySha256,
    compatibleContextBindingSha256,
    files: catalogFiles,
  };
}

function catalogFile(file, repository, revision) {
  return {
    path: file.path,
    role: file.role,
    url: `https://huggingface.co/${repository}/resolve/${revision}/${
      file.path.split("/").map(encodeURIComponent).join("/")
    }`,
    revision,
    bytes: file.bytes,
    sha256: file.sha256,
  };
}

function sumBytes(files) {
  const result = files.reduce((sum, file) => sum + file.bytes, 0);
  if (!Number.isSafeInteger(result) || result < 1) throw new Error("artifact byte total is invalid");
  return result;
}

function canonicalJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.entries(value)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
    .join(",")}}`;
}

function flagFor(field) {
  return `--${field.replace(/[A-Z]/gu, (letter) => `-${letter.toLowerCase()}`)}`;
}

function positiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 1) throw new Error(`${label} must be a positive integer`);
}

function nonnegativeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${label} must be a non-negative integer`);
  }
}

export async function clickOpen(page) {
  const choice = await waitForVisibleChoice(page, [
    ["ready", page.locator(".shell")],
    ["loading", page.locator(".runtime-gate.loading-pulse")],
    ["direct", page.getByRole("button", { name: "Open Drowse", exact: true })],
    ["warning", page.getByRole("button", { name: "Review warning", exact: true })],
  ], "model open control");
  if (choice.name === "ready" || choice.name === "loading") return;
  await choice.locator.click();
  if (choice.name === "warning") {
    const override = await waitForVisibleChoice(page, [
      ["override", page.getByRole("button", {
        name: /^(Load anyway|Retry this model)$/u,
      })],
    ], "unsafe model-load confirmation");
    await override.locator.click();
  }
}

export async function waitForVisibleChoice(page, choices, label, timeoutMs = 30_000) {
  try {
    return await Promise.any(choices.map(async ([name, locator]) => {
      await locator.waitFor({ state: "visible", timeout: timeoutMs });
      return { name, locator };
    }));
  } catch (error) {
    const body = (await page.locator("body").innerText().catch(() => ""))
      .replaceAll(/\s+/gu, " ")
      .slice(0, 2_000);
    throw new Error(`${label} did not appear; page text: ${body}`, { cause: error });
  }
}
