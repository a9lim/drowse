export const BROWSER_FULL_RUNTIME_TOOL_VERSION = "drowse-real-browser-e2e-v4";

export const REQUIRED_BROWSER_FULL_RUNTIME_CHECKS = Object.freeze([
  "local_artifacts",
  "webgpu_adapter",
  "load",
  "ordinary_generation",
  "forced_replay",
  "explicit_capture",
  "flat_gated_instruments",
  "curved_fitting",
  "curved_generation",
  "stop",
  "unload_reload",
  "offline_reuse",
]);

export function validatePairedSteeringControls(logprobs) {
  const names = ["baseline", "zero", "positive", "negative"];
  if (!isRecord(logprobs) || Object.keys(logprobs).length !== names.length || names.some((name) => {
    const value = logprobs[name];
    return typeof value !== "number" || Number.isNaN(value) || value > 0;
  }) || !Number.isFinite(logprobs.baseline) || !Number.isFinite(logprobs.zero)) {
    throw new Error("paired steering controls returned invalid sampler logprobs");
  }
  if (Math.abs(logprobs.zero - logprobs.baseline) > 1e-4) {
    throw new Error("zero steering changed the baseline distribution");
  }
  if (Math.max(Math.abs(logprobs.positive - logprobs.baseline), Math.abs(logprobs.negative - logprobs.baseline)) < 1e-4) {
    throw new Error("nonzero steering controls did not change the sampler distribution");
  }
  return {
    controlLogprobs: Object.fromEntries(names.map((name) => [name, logprobs[name] === -Infinity ? "-Infinity" : logprobs[name]])),
    controlProbabilities: Object.fromEntries(names.map((name) => [name, Math.exp(logprobs[name])])),
    zeroSteeringTolerance: 1e-4,
  };
}

export function parseBrowserFullRuntimeArguments(args) {
  const result = {
    browserChannel: "chrome",
    browserFlags: "normal",
    contextTokens: 2048,
    fitLayer: null,
    headed: false,
    jlensWord: "ocean",
    maxTokens: 4,
    modelId: null,
    modelDirectory: null,
    modelLibrary: null,
    output: null,
    runtimeLock: null,
    saeDirectory: null,
    saeFeature: 0,
    stopMaxTokens: 64,
    timeoutMs: 900_000,
    webLlm: null,
    coreDirectory: null,
    jlensDirectory: null,
  };
  const fields = new Map([
    ["--browser-channel", "browserChannel"],
    ["--browser-flags", "browserFlags"],
    ["--context-tokens", "contextTokens"],
    ["--core-directory", "coreDirectory"],
    ["--fit-layer", "fitLayer"],
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
    ["--stop-max-tokens", "stopMaxTokens"],
    ["--timeout-ms", "timeoutMs"],
    ["--webllm", "webLlm"],
  ]);
  const numeric = new Set([
    "contextTokens",
    "fitLayer",
    "maxTokens",
    "saeFeature",
    "stopMaxTokens",
    "timeoutMs",
  ]);
  const seen = new Set();
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--headed") {
      if (seen.has(argument))
        throw new Error(`${argument} may be supplied only once`);
      seen.add(argument);
      result.headed = true;
      continue;
    }
    const field = fields.get(argument);
    if (!field) throw new Error(`unexpected argument ${argument}`);
    if (seen.has(argument))
      throw new Error(`${argument} may be supplied only once`);
    seen.add(argument);
    const value = args[++index];
    if (!value || value.startsWith("--"))
      throw new Error(`${argument} requires a value`);
    result[field] = numeric.has(field) ? Number(value) : value;
  }
  for (const field of [
    "modelId",
    "modelDirectory",
    "modelLibrary",
    "webLlm",
    "coreDirectory",
    "jlensDirectory",
    "saeDirectory",
  ]) {
    if (result[field] === null)
      throw new Error(`${flagFor(field)} is required`);
  }
  if (
    result.browserChannel !== "chrome" &&
    result.browserChannel !== "msedge"
  ) {
    throw new Error("--browser-channel must be chrome or msedge");
  }
  if (
    result.browserFlags !== "compatibility" &&
    result.browserFlags !== "normal"
  ) {
    throw new Error("--browser-flags must be compatibility or normal");
  }
  positiveInteger(result.contextTokens, "--context-tokens");
  if (result.fitLayer !== null)
    nonnegativeInteger(result.fitLayer, "--fit-layer");
  if (!Number.isSafeInteger(result.maxTokens) || result.maxTokens < 2) {
    throw new Error("--max-tokens must be an integer of at least 2");
  }
  nonnegativeInteger(result.saeFeature, "--sae-feature");
  if (!Number.isSafeInteger(result.stopMaxTokens) || result.stopMaxTokens < 8) {
    throw new Error("--stop-max-tokens must be an integer of at least 8");
  }
  if (!Number.isSafeInteger(result.timeoutMs) || result.timeoutMs < 60_000) {
    throw new Error("--timeout-ms must be an integer of at least 60000");
  }
  if (
    typeof result.jlensWord !== "string" ||
    !result.jlensWord.trim() ||
    result.jlensWord.length > 128
  ) {
    throw new Error("--jlens-word must be a non-empty single-token candidate");
  }
  return Object.freeze({ ...result, jlensWord: result.jlensWord.trim() });
}

export function validateBrowserFullRuntimeModelClosure({
  lock,
  runtimeLock,
  manifest,
  files,
  contextTokens,
}) {
  if (!isRecord(lock) || !isRecord(runtimeLock) || !isRecord(manifest)) {
    throw new Error("browser model closure metadata is invalid");
  }
  if (manifest.schemaVersion !== 1)
    throw new Error("browser model closure schema is unsupported");
  const legacyRuntimeAlias = runtimeLock.runtimeAbi === "drowse-web-runtime-v1" &&
    ["saklas-web-runtime-v1", "polythetic-web-runtime-v1"].includes(manifest.runtimeAbi);
  for (const [label, actual, expected] of [
    ["runtime ABI", legacyRuntimeAlias ? runtimeLock.runtimeAbi : manifest.runtimeAbi, runtimeLock.runtimeAbi],
    ["hook ABI", manifest.hookAbi, runtimeLock.hookAbi],
    ["architecture", manifest.architecture, lock.architecture],
    ["quantization", manifest.quantization, lock.quantization],
    ["hidden size", manifest.hiddenSize, lock.hiddenSize],
  ]) {
    if (actual !== expected)
      throw new Error(
        `browser model closure ${label} does not match runtime-lock`,
      );
  }
  if (!isRecord(manifest.source))
    throw new Error("browser model closure source identity is invalid");
  if (
    manifest.source.repository !== lock.sourceRepository ||
    manifest.source.revision !== lock.sourceRevision
  ) {
    throw new Error(
      "browser model closure source identity does not match runtime-lock",
    );
  }
  if (
    !Array.isArray(manifest.layerMap) ||
    !Array.isArray(lock.layerMap) ||
    manifest.layerMap.length !== lock.layerMap.length ||
    manifest.layerMap.some((layer, index) => layer !== lock.layerMap[index])
  ) {
    throw new Error(
      "browser model closure layer map does not match runtime-lock",
    );
  }
  if (
    !Number.isSafeInteger(manifest.contextWindowSize) ||
    manifest.contextWindowSize < contextTokens
  ) {
    throw new Error(
      "browser model closure cannot provide the requested context profile",
    );
  }
  if (!Array.isArray(files))
    throw new Error("browser model closure file list is invalid");
  for (const [role, expected] of [
    ["converted_manifest", lock.manifestSha256],
    ["model_library", lock.librarySha256],
    ["tokenizer", lock.tokenizerSha256],
    ["chat_template", lock.chatTemplateSha256],
  ]) {
    const matching = files.filter(
      (file) => isRecord(file) && file.role === role,
    );
    if (matching.length !== 1 || matching[0].sha256 !== expected) {
      throw new Error(
        `browser model closure ${role} does not match runtime-lock`,
      );
    }
  }
  if (!files.some((file) => isRecord(file) && file.role === "weight")) {
    throw new Error("browser model closure contains no weight shards");
  }
}

export function isBrowserFullRuntimeWebGpuError(message) {
  return (
    typeof message === "string" &&
    /GPU(?:Validation|Pipeline|OutOfMemory)Error|device(?: was| is)? lost|WebGPU error(?: was not captured|\s*#|:)|WebGPU uncaptured error|uncaptured (?:WebGPU )?error|\bWebGPU\b.*\b(?:validation error|invalid|out[ -]?of[ -]?memory|device loss|device lost)\b|\b(?:validation error|out[ -]?of[ -]?memory)\b.*\b(?:WebGPU|GPU)\b/iu.test(
      message,
    )
  );
}

export function validateBrowserFullRuntimeAudit(audit) {
  if (!isRecord(audit))
    throw new Error("real browser E2E error audit is invalid");
  for (const name of [
    "externalRequests",
    "failedRequests",
    "uncaughtPageErrors",
    "errorConsoleMessages",
    "webGpuErrors",
    "offlineServerRequests",
    "deviceLosses",
  ]) {
    const entries = audit[name];
    if (!Array.isArray(entries))
      throw new Error(`real browser E2E audit ${name} is invalid`);
    validateMeasurementValue(entries, `real browser E2E audit ${name}`);
    if (entries.length > 0) {
      throw new Error(
        `real browser E2E audit ${name} is non-empty: ${JSON.stringify(entries)}`,
      );
    }
  }
  return structuredClone(audit);
}

export function validateBrowserFullRuntimeTranscript(events) {
  if (!Array.isArray(events))
    throw new Error("browser check transcript must be an array");
  const checks = {};
  let eventIndex = 0;
  for (const expected of REQUIRED_BROWSER_FULL_RUNTIME_CHECKS) {
    const started = events[eventIndex++];
    if (
      !started ||
      started.type !== "check_started" ||
      started.check !== expected
    ) {
      throw new Error(
        `browser check ${expected} did not start in the required order`,
      );
    }
    const passed = events[eventIndex++];
    if (
      !passed ||
      passed.type !== "check_passed" ||
      passed.check !== expected
    ) {
      throw new Error(
        `browser check ${expected} did not produce a measured pass`,
      );
    }
    if (
      !isRecord(passed.measurements) ||
      Object.keys(passed.measurements).length === 0
    ) {
      throw new Error(`browser check ${expected} has no measurements`);
    }
    validateMeasurementValue(passed.measurements, `browser check ${expected}`);
    checks[expected] = {
      passed: true,
      measurements: structuredClone(passed.measurements),
    };
  }
  if (eventIndex !== events.length)
    throw new Error("browser check transcript contains extra events");
  return checks;
}

export function createBrowserFullRuntimeReport({
  events,
  environment,
  producedAt = new Date(),
}) {
  const checks = validateBrowserFullRuntimeTranscript(events);
  if (!isRecord(environment) || Object.keys(environment).length === 0) {
    throw new Error("browser E2E environment is required");
  }
  validateMeasurementValue(environment, "browser E2E environment");
  const timestamp =
    producedAt instanceof Date ? producedAt.toISOString() : String(producedAt);
  if (!Number.isFinite(Date.parse(timestamp)))
    throw new Error("browser E2E timestamp is invalid");
  return {
    $schema: "drowse-real-browser-e2e-measurement-v1",
    schemaVersion: 1,
    evidenceType: "realBrowserEndToEnd",
    toolVersion: BROWSER_FULL_RUNTIME_TOOL_VERSION,
    producedAt: timestamp,
    passed: true,
    releaseEvidence: false,
    environment: structuredClone(environment),
    checks,
  };
}

function flagFor(field) {
  return `--${field.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`)}`;
}

function positiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 1)
    throw new Error(`${label} must be a positive integer`);
}

function nonnegativeInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 0)
    throw new Error(`${label} must be a non-negative integer`);
}

function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validateMeasurementValue(value, label) {
  if (value === null || typeof value === "string" || typeof value === "boolean")
    return;
  if (typeof value === "number") {
    if (!Number.isFinite(value))
      throw new Error(`${label} contains a non-finite number`);
    return;
  }
  if (Array.isArray(value)) {
    for (const item of value) validateMeasurementValue(item, label);
    return;
  }
  if (!isRecord(value))
    throw new Error(`${label} contains an unsupported value`);
  for (const child of Object.values(value))
    validateMeasurementValue(child, label);
}
