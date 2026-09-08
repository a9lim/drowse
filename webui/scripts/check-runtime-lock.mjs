import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { lstat, readFile, realpath } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { isAbsolute, relative, resolve, sep } from "node:path";
import {
  structuredHookProfile,
  thinkingProfile,
} from "./hosted-model-profiles.mjs";
import {
  releaseEvidenceProducerBinding,
  requireReleaseEvidenceProducer,
} from "./release-evidence-producers.mjs";

const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));

const lockPath = new URL(
  "../../browser-runtime/runtime-lock.json",
  import.meta.url,
);
const distributionPath = new URL(
  "../../browser-runtime/distribution-lock.json",
  import.meta.url,
);
const runtimeLockIdentityFixturePath = new URL(
  "../../browser-runtime/runtime-lock-identity-v1.json",
  import.meta.url,
);
const headersPath = new URL("../public-hosted/_headers", import.meta.url);
const backendFactoryPath = new URL(
  "../src/hosted/runtime/browserModelBackend.ts",
  import.meta.url,
);
const browserWorkerPath = new URL(
  "../src/hosted/runtime/browser.worker.ts",
  import.meta.url,
);
const authoringMarkerPath = new URL(
  "../src/hosted/runtime/browserAuthoring.ts",
  import.meta.url,
);
const browserManifoldFittingPath = new URL(
  "../src/hosted/runtime/browserManifoldFitting.ts",
  import.meta.url,
);
const activationSpoolPath = new URL(
  "../src/hosted/fitting/activationSpool.ts",
  import.meta.url,
);
const fittingWorkerPath = new URL(
  "../src/hosted/fitting/fitting.worker.ts",
  import.meta.url,
);
const capabilitiesPath = new URL(
  "../src/hosted/runtime/capabilities.ts",
  import.meta.url,
);
const packagePath = new URL("../package.json", import.meta.url);
const packageLockPath = new URL("../package-lock.json", import.meta.url);
const installedWebLlmPackagePath = new URL(
  "../node_modules/@drowse/web-llm/package.json",
  import.meta.url,
);
const forkManifestPath = new URL(
  "../../browser-runtime/forks/manifest.json",
  import.meta.url,
);
const compilerRecipePath = new URL(
  "../../browser-runtime/compiler/Dockerfile",
  import.meta.url,
);
const tinyModelToolPath = new URL(
  "../../browser-runtime/forks/create-tiny-webgpu-model.py",
  import.meta.url,
);
const tinyCompilerToolPath = new URL(
  "../../browser-runtime/forks/compile-tiny-webgpu.py",
  import.meta.url,
);
const runtimeLockBytes = await readFile(lockPath);
const lock = JSON.parse(runtimeLockBytes.toString("utf8"));
const distribution = JSON.parse(await readFile(distributionPath, "utf8"));
const runtimeLockIdentityFixture = JSON.parse(
  await readFile(runtimeLockIdentityFixturePath, "utf8"),
);
const hostedHeaders = await readFile(headersPath, "utf8");
const backendFactory = await readFile(backendFactoryPath, "utf8");
const browserWorker = await readFile(browserWorkerPath, "utf8");
const authoringMarker = await readFile(authoringMarkerPath, "utf8");
const browserManifoldFitting = await readFile(
  browserManifoldFittingPath,
  "utf8",
);
const activationSpool = await readFile(activationSpoolPath, "utf8");
const fittingWorker = await readFile(fittingWorkerPath, "utf8");
const capabilitiesSource = await readFile(capabilitiesPath, "utf8");
const packageManifest = JSON.parse(await readFile(packagePath, "utf8"));
const packageLock = JSON.parse(await readFile(packageLockPath, "utf8"));
const installedWebLlmPackage = JSON.parse(
  await readFile(installedWebLlmPackagePath, "utf8"),
);
const forkManifestBytes = await readFile(forkManifestPath);
const forkManifest = JSON.parse(forkManifestBytes.toString("utf8"));
const compilerRecipe = await readFile(compilerRecipePath);
const tinyModelTool = await readFile(tinyModelToolPath, "utf8");
const tinyCompilerTool = await readFile(tinyCompilerToolPath, "utf8");
const COMMIT = /^[0-9a-f]{40}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const IMAGE_DIGEST = /^sha256:[0-9a-f]{64}$/;
const IMAGE_REFERENCE =
  /^[a-z0-9.-]+(?::[1-9][0-9]{0,4})?\/[a-z0-9]+(?:[._/-][a-z0-9]+)*(?::[A-Za-z0-9_][A-Za-z0-9._-]{0,127})?$/;
const BROWSER_VERSION = /^[0-9]+(?:\.[0-9]+){0,3}$/;
const CATALOG_WEBGPU_LIMIT_NAMES = new Set([
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxComputeWorkgroupStorageSize",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupSizeZ",
  "maxComputeWorkgroupsPerDimension",
  "maxComputeInvocationsPerWorkgroup",
  "maxStorageBuffersPerShaderStage",
  "maxBindingsPerBindGroup",
]);
const REQUIRED_BENCHMARK_PLATFORMS = ["macos", "windows", "linux", "android"];
const REQUIRED_BENCHMARK_BROWSERS = ["chrome", "edge"];
const REQUIRED_BENCHMARK_CONTEXTS = [2048, 4096];
const REQUIRED_DROWSE_VM_FUNCTIONS = [
  "drowse_hook_profile",
  "drowse_prefill",
  "drowse_decode",
  "drowse_batch_prefill",
  "drowse_batch_decode",
  "drowse_capture_prefill",
  "drowse_capture_decode",
  "drowse_capture_batch_prefill",
  "drowse_capture_batch_decode",
  "drowse_rank_one_capture_prefill_v1",
  "drowse_rank_one_capture_decode_v1",
  "drowse_rank_one_capture_batch_prefill_v1",
  "drowse_rank_one_capture_batch_decode_v1",
  "drowse_structured_batch_prefill",
  "drowse_structured_batch_decode",
  "drowse_geometry_batch_prefill",
  "drowse_geometry_batch_decode",
  "drowse_curved_batch_prefill",
  "drowse_curved_batch_decode",
  "drowse_jlens_probabilities",
  "drowse_jlens_readout_accumulate",
  "drowse_jlens_readout_topk",
  "drowse_jlens_directions",
  "drowse_sae_readout_accumulate",
  "drowse_sae_jump_relu_readout_accumulate",
];

export { requireReleaseEvidenceProducer };

export function browserRuntimeIdentityBlockers(runtime) {
  const required = [];
  const releaseModels = (runtime.models ?? []).filter(
    (model) => model.convertedRevision !== null && model.convertedRevision !== undefined,
  );
  if (releaseModels.length === 0) {
    return ["models contain no release-ready converted revision"];
  }
  for (const model of releaseModels) {
    for (const field of [
      "convertedRevision",
      "manifestSha256",
      "librarySha256",
      "tokenizerSha256",
      "chatTemplateSha256",
      "hiddenSize",
      "layerMap",
    ]) {
      required.push([`models.${model.id}.${field}`, model[field]]);
    }
  }
  return required
    .filter(([, value]) => value == null)
    .map(([path]) => `${path} is unresolved`);
}

export function browserRuntimeReleaseBlockers(runtime) {
  const blockers = [];
  if (runtime.status !== "verified") {
    blockers.push(
      `status is ${JSON.stringify(runtime.status)} (expected "verified")`,
    );
  }
  return [...blockers, ...browserRuntimeIdentityBlockers(runtime)];
}

async function main() {
  const options = parseArguments(process.argv.slice(2), process.env);
  const release = options.release;

  exactKeys(
    lock,
    [
      "$schema",
      "schemaVersion",
      "status",
      "runtimeAbi",
      "hookAbi",
      "exactReadoutAbi",
      "toolchain",
      "models",
    ],
    "runtime lock",
  );

  if (lock.schemaVersion !== 1)
    throw new Error("unsupported browser runtime lock schema");
  if (!["feasibility-required", "verified"].includes(lock.status)) {
    throw new Error("invalid browser runtime lock status");
  }
  if (!/^drowse-web-runtime-v[0-9]+$/.test(lock.runtimeAbi)) {
    throw new Error("invalid browser runtime ABI");
  }
  if (lock.hookAbi !== "post-block-residual-v4")
    throw new Error("unexpected hook ABI");
  if (lock.exactReadoutAbi !== "exact-readout-v1")
    throw new Error("unexpected exact readout ABI");
  exactKeys(
    runtimeLockIdentityFixture,
    ["schemaVersion", "runtimeLockIdentitySha256"],
    "runtime lock identity fixture",
  );
  if (
    runtimeLockIdentityFixture.schemaVersion !== 1 ||
    runtimeLockIdentityFixture.runtimeLockIdentitySha256 !== runtimeLockIdentitySha256(lock)
  ) {
    throw new Error(
      "runtime lock identity fixture does not match the canonical status-free runtime lock",
    );
  }
  validateFixtureToolPins(forkManifest, tinyModelTool, tinyCompilerTool);
  validateToolchainBindings(
    lock.toolchain,
    forkManifest,
    forkManifestBytes,
    compilerRecipe,
  );
  await validateWebLlmPackageArchive(
    lock.toolchain.webLlmPackage,
    lock.toolchain.webLlmFork,
  );
  if (!Array.isArray(lock.models) || lock.models.length < 1) {
    throw new Error("browser runtime lock must contain at least one model");
  }

  const modelIds = new Set();
  for (const model of lock.models) {
    exactKeys(
      model,
      [
        "id",
        "architecture",
        "sourceRepository",
        "sourceRevision",
        "structuredHookProfile",
        "thinkingProfile",
        "convertedRepository",
        "convertedRevision",
        "manifestSha256",
        "librarySha256",
        "tokenizerSha256",
        "chatTemplateSha256",
        "hiddenSize",
        "layerMap",
        "quantization",
        "contextProfiles",
        ...("modelType" in model ? ["modelType"] : []),
      ],
      `model ${model?.id ?? "unknown"}`,
    );
    if (
      typeof model.id !== "string" ||
      !/^[a-z0-9][a-z0-9._-]{0,63}$/.test(model.id) ||
      modelIds.has(model.id)
    ) {
      throw new Error("browser runtime model IDs must be unique safe slugs");
    }
    modelIds.add(model.id);
    if (
      !["llama", "gemma3_text", "qwen3", "gpt2", "gpt_neox", "qwen3_5"].includes(model.architecture) ||
      (model.modelType !== undefined && !["chat", "base"].includes(model.modelType)) ||
      (["gpt2", "gpt_neox", "qwen3_5"].includes(model.architecture) && model.modelType !== "base") ||
      typeof model.sourceRepository !== "string" ||
      !model.sourceRepository.includes("/") ||
      typeof model.convertedRepository !== "string" ||
      !model.convertedRepository.includes("/") ||
      !["q4f16_1", "q4f32_1", "q0f32"].includes(model.quantization)
    ) {
      throw new Error(
        `browser launch model ${model.id} has an unsupported runtime configuration`,
      );
    }
    if (structuredHookProfile(model.structuredHookProfile) !== "standard-v3") {
      throw new Error(
        `${model.id} does not use the production structured hook profile`,
      );
    }
    const verifiedThinking = thinkingProfile(model.thinkingProfile);
    const expectedThinking =
      model.architecture === "qwen3"
        ? {
            start: "<think>",
            end: "</think>",
            startsInThinking: false,
            startTokenIds: [151667],
            endTokenIds: [151668],
          }
        : null;
    if (JSON.stringify(verifiedThinking) !== JSON.stringify(expectedThinking)) {
      throw new Error(`${model.id} has an unexpected thinking-token profile`);
    }
    if (!COMMIT.test(model.sourceRevision))
      throw new Error(`${model.id} has no immutable source revision`);
    nullableMatch(
      model.convertedRevision,
      COMMIT,
      `${model.id} converted revision`,
    );
    for (const key of [
      "manifestSha256",
      "librarySha256",
      "tokenizerSha256",
      "chatTemplateSha256",
    ]) {
      nullableMatch(model[key], SHA256, `${model.id} ${key}`);
    }
    if (
      model.hiddenSize !== null &&
      (!Number.isSafeInteger(model.hiddenSize) || model.hiddenSize < 1)
    ) {
      throw new Error(`${model.id} hidden size is invalid`);
    }
    if (
      model.layerMap !== null &&
      (!Array.isArray(model.layerMap) ||
        model.layerMap.length < 1 ||
        model.layerMap.some(
          (layer, index) =>
            !Number.isSafeInteger(layer) ||
            layer < 0 ||
            (index > 0 && layer <= model.layerMap[index - 1]),
        ))
    )
      throw new Error(`${model.id} layer map is invalid`);
    if (
      !Array.isArray(model.contextProfiles) ||
      model.contextProfiles.length < 1 ||
      model.contextProfiles.some(
        (profile, index) =>
          !Number.isSafeInteger(profile) ||
          profile < 1 ||
          (model.architecture === "gpt2" && profile > 1024) ||
          (index > 0 && profile <= model.contextProfiles[index - 1]),
      )
    ) {
      throw new Error(`${model.id} context profiles must be unique and sorted`);
    }
  }

  const runtimeReleaseBlockers = browserRuntimeReleaseBlockers(lock);
  const unresolvedRuntimeIdentities = browserRuntimeIdentityBlockers(lock);

  validateDistribution(distribution);
  validateHostedConnectSources(hostedHeaders, distribution);

  if (release && runtimeReleaseBlockers.length > 0) {
    throw new Error(
      `browser runtime is not publishable:\n- ${runtimeReleaseBlockers.join("\n- ")}`,
    );
  }
  if (lock.status === "verified" && unresolvedRuntimeIdentities.length > 0) {
    throw new Error(
      `verified browser runtime lock contains unresolved identities:\n- ${unresolvedRuntimeIdentities.join("\n- ")}`,
    );
  }
  if (release && distribution.status !== "verified") {
    throw new Error("browser artifact distribution is not publishable");
  }
  if (release) validateReleaseRevision(options.drowseRevision);
  const backendIntegrated =
    /export const BROWSER_MODEL_BACKEND_INTEGRATED\s*=\s*true\s*;/.test(
      backendFactory,
    );
  if (backendIntegrated) {
    validateWebLlmDependency(
      packageManifest,
      packageLock,
      lock.toolchain.webLlmFork,
      lock.toolchain.webLlmPackage,
      false,
    );
    validateInstalledWebLlmDependency(
      packageManifest,
      packageLock,
      installedWebLlmPackage,
    );
  }
  if (release && !backendIntegrated) {
    throw new Error("browser model backend is not release-integrated");
  }
  const authoringIntegrated =
    /export const BROWSER_AUTHORING_RUNTIME_INTEGRATED\s*=\s*true\s*;/.test(
      authoringMarker,
    );
  const authoringImplementationIntegrated =
    /export const BROWSER_AUTHORING_IMPLEMENTATION_INTEGRATED\s*=\s*true\s*;/.test(
      authoringMarker,
    );
  const artifactAuthoringIntegrated =
    /export const BROWSER_ARTIFACT_AUTHORING_INTEGRATED\s*=\s*true\s*;/.test(
      authoringMarker,
    );
  if (
    artifactAuthoringIntegrated !==
    /manifold_artifacts:\s*BROWSER_ARTIFACT_AUTHORING_INTEGRATED/.test(
      capabilitiesSource,
    )
  ) {
    throw new Error(
      "browser artifact authoring marker does not match capabilities",
    );
  }
  validateAuthoringCapabilityWiring(capabilitiesSource, authoringIntegrated);
  validateAuthoringIntegrationState({
    implementationIntegrated: authoringImplementationIntegrated,
    capabilityIntegrated: authoringIntegrated,
    release: false,
  });
  validateAuthoringSpoolIntegrationState({
    implementationIntegrated: authoringImplementationIntegrated,
    capabilityIntegrated: authoringIntegrated,
    deprecatedCapabilityMarkerPresent:
      /ACTIVATION_SPOOL_CAPABILITY_ENABLED/.test(activationSpool),
    coordinatorUsesActivationSpool:
      /new BrowserFittingCoordinator\(\s*request\.activationSpool\s*,/.test(
        browserManifoldFitting,
    ),
    fittingWorkerReadsActivationSpool:
      /withBrowserActivationSpool\(/.test(fittingWorker) &&
      /openCommitted\(/.test(fittingWorker),
  });
  if (
    backendIntegrated &&
    (/new UnavailableBrowserModelBackend\s*\(/.test(backendFactory) ||
      !/modelBackend:\s*createLazyBrowserModelBackend\s*\(\s*\(\)\s*=>\s*import\("@drowse\/web-llm"\)\s*,\s*artifacts\s*,?\s*\)/.test(
        browserWorker,
      ))
  ) {
    throw new Error(
      "browser model backend integration marker does not match the worker factory",
    );
  }
  if (distribution.status === "verified" && lock.status !== "verified") {
    throw new Error(
      "verified browser distribution requires a verified runtime",
    );
  }
  console.log(`Browser runtime lock: ${lock.status}`);
  console.log(`Browser distribution lock: ${distribution.status}`);
}

if (
  process.argv[1] &&
  pathToFileURL(process.argv[1]).href === import.meta.url
) {
  await main();
}

export function validateBenchmarkEvidence(value, runtime) {
  exactKeys(
    value,
    [
      "$schema",
      "schemaVersion",
      "status",
      "drowseRevision",
      "requiredMatrix",
      "supportPolicy",
      "runs",
    ],
    "benchmark evidence",
  );
  if (
    value.$schema !== "./benchmark-evidence.schema.json" ||
    value.schemaVersion !== 4
  )
    throw new Error("unsupported benchmark evidence schema");
  if (!["feasibility-required", "verified"].includes(value.status)) {
    throw new Error("invalid benchmark evidence status");
  }
  exactKeys(
    value.requiredMatrix,
    ["modelIds", "contextTokens", "platforms", "browsers"],
    "benchmark required matrix",
  );
  const requiredModelIds = runtime.models.map((model) => model.id);
  if (
    !sameStringSet(value.requiredMatrix.modelIds, requiredModelIds) ||
    !sameNumberSet(
      value.requiredMatrix.contextTokens,
      REQUIRED_BENCHMARK_CONTEXTS,
    ) ||
    !sameStringSet(
      value.requiredMatrix.platforms,
      REQUIRED_BENCHMARK_PLATFORMS,
    ) ||
    !sameStringSet(value.requiredMatrix.browsers, REQUIRED_BENCHMARK_BROWSERS)
  )
    throw new Error(
      "benchmark required matrix does not cover every launch model, context, platform, and browser",
    );
  if (!Array.isArray(value.runs))
    throw new Error("benchmark evidence runs must be an array");
  validateEvidenceSetRevision(
    value.drowseRevision,
    value.runs.length > 0,
    value.status,
    "benchmark evidence",
  );
  exactKeys(
    value.supportPolicy,
    ["minimumChromeMajor", "minimumEdgeMajor", "maximumEvidenceAgeDays"],
    "benchmark support policy",
  );
  for (const field of ["minimumChromeMajor", "minimumEdgeMajor"]) {
    const minimum = value.supportPolicy[field];
    if (minimum !== null && (!Number.isSafeInteger(minimum) || minimum < 100)) {
      throw new Error(`benchmark ${field} is invalid`);
    }
  }
  if (
    !Number.isSafeInteger(value.supportPolicy.maximumEvidenceAgeDays) ||
    value.supportPolicy.maximumEvidenceAgeDays < 1 ||
    value.supportPolicy.maximumEvidenceAgeDays > 180
  )
    throw new Error("benchmark evidence age policy is invalid");
  const models = new Map(runtime.models.map((model) => [model.id, model]));
  const runKeys = new Set();
  for (const run of value.runs) {
    exactKeys(
      run,
      [
        "producerId",
        "producerConfigPath",
        "producerConfigSha256",
        "producerToolVersion",
        "modelId",
        "contextTokens",
        "contextBindingSha256",
        "platform",
        "browser",
        "browserVersion",
        "browserFlags",
        "deviceLabel",
        "drowseRevision",
        "runtimeIdentitySha256",
        "webLlmSha256",
        "modelArtifactManifestSha256",
        "convertedManifestSha256",
        "modelLibrarySha256",
        "coreArtifactManifestSha256",
        "jlensArtifactManifestSha256",
        "saeArtifactManifestSha256",
        "adapterIsFallback",
        "calibrationScore",
        "deviceMemoryGiB",
        "adapterFeatures",
        "adapterLimits",
        "prefillTokensPerSecond",
        "decodeTokensPerSecond",
        "loadSucceeded",
        "measuredAt",
      ],
      "benchmark run",
    );
    const producer = requireReleaseEvidenceProducer("physicalBrowserBenchmark");
    if (
      run.producerId !== producer.id ||
      typeof run.producerConfigPath !== "string" ||
      !/^browser-runtime\/evidence-configs\/[a-z0-9][a-z0-9._-]*\.json$/.test(
        run.producerConfigPath,
      ) ||
      !SHA256.test(run.producerConfigSha256) ||
      typeof run.producerToolVersion !== "string" ||
      !new RegExp(`^${producer.id}\\+sha256\\.[0-9a-f]{64}$`).test(
        run.producerToolVersion,
      )
    ) {
      throw new Error("benchmark producer config binding is invalid");
    }
    const model = models.get(run.modelId);
    if (!model) throw new Error(`benchmark has unknown model ${run.modelId}`);
    if (![2048, 4096].includes(run.contextTokens))
      throw new Error("benchmark context is invalid");
    if (!["macos", "windows", "linux", "android"].includes(run.platform)) {
      throw new Error("benchmark platform is invalid");
    }
    if (!["chrome", "edge"].includes(run.browser))
      throw new Error("benchmark browser is invalid");
    if (run.drowseRevision !== value.drowseRevision) {
      throw new Error(
        "benchmark run source revision does not match its evidence set",
      );
    }
    if (run.adapterIsFallback !== false) {
      throw new Error(
        "benchmark run does not confirm a non-fallback hardware adapter",
      );
    }
    for (const field of ["deviceLabel", "measuredAt"]) {
      if (typeof run[field] !== "string" || !run[field])
        throw new Error(`benchmark ${field} is invalid`);
    }
    if (!BROWSER_VERSION.test(run.browserVersion))
      throw new Error("benchmark browserVersion is invalid");
    if (run.browserFlags !== "normal")
      throw new Error("benchmark browser flags are not release-safe");
    const minimumBrowserMajor =
      run.browser === "chrome"
        ? value.supportPolicy.minimumChromeMajor
        : value.supportPolicy.minimumEdgeMajor;
    if (
      minimumBrowserMajor !== null &&
      Number.parseInt(run.browserVersion.split(".")[0], 10) <
        minimumBrowserMajor
    )
      throw new Error("benchmark browser version predates the support policy");
    const identitySha256 = lockedRuntimeIdentitySha256(runtime, model);
    if (
      identitySha256 === null ||
      run.runtimeIdentitySha256 !== identitySha256
    ) {
      throw new Error(`benchmark runtime identity does not match ${model.id}`);
    }
    if (
      run.convertedManifestSha256 !== model.manifestSha256 ||
      run.modelLibrarySha256 !== model.librarySha256
    ) {
      throw new Error(`benchmark observed artifacts do not match ${model.id}`);
    }
    for (const field of [
      "webLlmSha256",
      "modelArtifactManifestSha256",
      "coreArtifactManifestSha256",
      "jlensArtifactManifestSha256",
      "saeArtifactManifestSha256",
    ]) {
      if (!SHA256.test(run[field] ?? ""))
        throw new Error(`benchmark ${field} is invalid`);
    }
    if (
      run.contextBindingSha256 !==
      digestCanonical({
        contextTokens: run.contextTokens,
        runtimeIdentitySha256: identitySha256,
      })
    )
      throw new Error("benchmark context binding is invalid");
    if (!Number.isFinite(run.calibrationScore) || run.calibrationScore <= 0) {
      throw new Error("benchmark calibration score is invalid");
    }
    if (
      run.deviceMemoryGiB !== null &&
      (!Number.isFinite(run.deviceMemoryGiB) || run.deviceMemoryGiB <= 0)
    ) {
      throw new Error("benchmark device memory is invalid");
    }
    if (
      !Array.isArray(run.adapterFeatures) ||
      new Set(run.adapterFeatures).size !== run.adapterFeatures.length ||
      run.adapterFeatures.some(
        (feature) => typeof feature !== "string" || !feature,
      )
    )
      throw new Error("benchmark adapter features are invalid");
    if (
      !run.adapterLimits ||
      typeof run.adapterLimits !== "object" ||
      Array.isArray(run.adapterLimits)
    ) {
      throw new Error("benchmark adapter limits are invalid");
    }
    for (const [name, limit] of Object.entries(run.adapterLimits)) {
      if (
        !CATALOG_WEBGPU_LIMIT_NAMES.has(name) ||
        !Number.isFinite(limit) ||
        limit < 0
      ) {
        throw new Error("benchmark adapter limits are invalid");
      }
    }
    const measuredAt = Date.parse(run.measuredAt);
    if (
      !Number.isFinite(run.prefillTokensPerSecond) ||
      run.prefillTokensPerSecond <= 0 ||
      !Number.isFinite(run.decodeTokensPerSecond) ||
      run.decodeTokensPerSecond <= 0 ||
      run.loadSucceeded !== true ||
      !Number.isFinite(measuredAt) ||
      measuredAt > Date.now() + 5 * 60 * 1000
    )
      throw new Error("benchmark result is invalid");
    const runKey = [
      run.modelId,
      run.contextTokens,
      run.platform,
      run.browser,
      run.browserVersion,
      run.deviceLabel,
    ].join("\0");
    if (runKeys.has(runKey))
      throw new Error("benchmark evidence contains a duplicate run");
    runKeys.add(runKey);
  }
  if (value.status !== "verified") return;
  if (
    value.supportPolicy.minimumChromeMajor === null ||
    value.supportPolicy.minimumEdgeMajor === null
  )
    throw new Error(
      "verified benchmark evidence requires minimum browser versions",
    );
  const oldestAllowed =
    Date.now() -
    value.supportPolicy.maximumEvidenceAgeDays * 24 * 60 * 60 * 1000;
  if (value.runs.some((run) => Date.parse(run.measuredAt) < oldestAllowed)) {
    throw new Error("physical benchmark evidence is stale");
  }
  for (const [browser, field] of [
    ["chrome", "minimumChromeMajor"],
    ["edge", "minimumEdgeMajor"],
  ]) {
    const observedMinimum = Math.min(
      ...value.runs
        .filter((run) => run.browser === browser)
        .map((run) => Number.parseInt(run.browserVersion.split(".")[0], 10)),
    );
    if (value.supportPolicy[field] !== observedMinimum) {
      throw new Error(
        `benchmark ${field} must equal the oldest physically tested ${browser} version`,
      );
    }
  }
  for (const modelId of value.requiredMatrix.modelIds) {
    for (const contextTokens of value.requiredMatrix.contextTokens) {
      for (const platform of value.requiredMatrix.platforms) {
        for (const browser of value.requiredMatrix.browsers) {
          if (
            !value.runs.some(
              (run) =>
                run.modelId === modelId &&
                run.contextTokens === contextTokens &&
                run.platform === platform &&
                run.browser === browser &&
                run.loadSucceeded &&
                run.adapterIsFallback === false,
            )
          ) {
            throw new Error(
              `${modelId} ${contextTokens} ${platform} ${browser} has no confirmed hardware benchmark evidence`,
            );
          }
        }
      }
    }
  }
}

export async function validateBenchmarkProducerBindings(
  value,
  root = repositoryRoot,
) {
  for (const run of value.runs) {
    const binding = await releaseEvidenceProducerBinding(
      "physicalBrowserBenchmark",
      root,
      run.producerConfigPath,
    );
    if (
      run.producerId !== binding.producerId ||
      run.producerConfigPath !== binding.configPath ||
      run.producerConfigSha256 !== binding.configSha256 ||
      run.producerToolVersion !== binding.toolVersion
    ) {
      throw new Error("benchmark producer config binding does not match checked-in bytes");
    }
  }
}

export async function validateRuntimeGateResults(
  name,
  results,
  fixtureRoot = repositoryRoot,
  runtime = null,
) {
  if (
    [
      "gemmaTinyFp32Parity",
      "llamaTinyFp32Parity",
      "qwenTinyFp32Parity",
    ].includes(name)
  ) {
    exactKeys(
      results,
      [
        "fixturePath",
        "fixtureSha256",
        "architecture",
        "maxAbsoluteLogitError",
        "maxRelativeLogitError",
        "maxAbsoluteCaptureError",
        "maxRelativeCaptureError",
        "probePassed",
        "steeringPassed",
      ],
      `${name} results`,
    );
    await validateFixtureBinding(results, name, fixtureRoot);
    const expectedArchitecture = {
      gemmaTinyFp32Parity: "gemma3_text",
      llamaTinyFp32Parity: "llama",
      qwenTinyFp32Parity: "qwen3",
    }[name];
    if (results.architecture !== expectedArchitecture) {
      throw new Error(`${name} fixture binding is invalid`);
    }
    for (const field of ["maxAbsoluteLogitError", "maxAbsoluteCaptureError"]) {
      if (
        !Number.isFinite(results[field]) ||
        results[field] < 0 ||
        results[field] > 1e-4
      ) {
        throw new Error(`${name} exceeds the fp32 absolute tolerance`);
      }
    }
    for (const field of ["maxRelativeLogitError", "maxRelativeCaptureError"]) {
      if (
        !Number.isFinite(results[field]) ||
        results[field] < 0 ||
        results[field] > 1e-3
      ) {
        throw new Error(`${name} exceeds the fp32 relative tolerance`);
      }
    }
    if (results.probePassed !== true || results.steeringPassed !== true) {
      throw new Error(`${name} did not pass probe and steering checks`);
    }
    return;
  }
  if (
    [
      "gemmaProductionQ4Parity",
      "smolProductionQ4Parity",
      "qwenProductionQ4Parity",
    ].includes(name)
  ) {
    exactKeys(
      results,
      [
        "fixturePath",
        "fixtureSha256",
        "modelId",
        "runtimeIdentitySha256",
        "hookDisabledGreedyMatchesBaseline",
        "maxProbeRelativeError",
        "maxCaptureRelativeError",
        "steeringDirectionCosine",
      ],
      `${name} results`,
    );
    await validateFixtureBinding(results, name, fixtureRoot);
    const expectedModel = {
      gemmaProductionQ4Parity: "gemma3-270m-instruct",
      smolProductionQ4Parity: "smollm2-360m-instruct",
      qwenProductionQ4Parity: "qwen3-1.7b",
    }[name];
    if (results.modelId !== expectedModel) {
      throw new Error(`${name} fixture binding is invalid`);
    }
    if (!SHA256.test(results.runtimeIdentitySha256 ?? "")) {
      throw new Error(`${name} observed runtime identity is invalid`);
    }
    if (runtime !== null) {
      const model = runtime.models?.find((candidate) => candidate.id === expectedModel);
      const expectedIdentity = model
        ? lockedRuntimeIdentitySha256(runtime, model)
        : null;
      if (expectedIdentity === null || results.runtimeIdentitySha256 !== expectedIdentity) {
        throw new Error(`${name} observed runtime identity does not match the locked model`);
      }
    }
    if (
      results.hookDisabledGreedyMatchesBaseline !== true ||
      !Number.isFinite(results.maxProbeRelativeError) ||
      results.maxProbeRelativeError < 0 ||
      results.maxProbeRelativeError > 0.01 ||
      !Number.isFinite(results.maxCaptureRelativeError) ||
      results.maxCaptureRelativeError < 0 ||
      results.maxCaptureRelativeError > 0.01 ||
      !Number.isFinite(results.steeringDirectionCosine) ||
      results.steeringDirectionCosine < 0.99 ||
      results.steeringDirectionCosine > 1
    )
      throw new Error(
        `${name} does not satisfy the production q4 parity thresholds`,
      );
    return;
  }
  if (name === "gemmaLongPrefill") {
    exactKeys(
      results,
      [
        "fixturePath",
        "fixtureSha256",
        "modelId",
        "webLlmBaselineVersion",
        "prompt2048Passed",
        "prompt4096Passed",
        "shapeCacheHangObserved",
      ],
      `${name} results`,
    );
    await validateFixtureBinding(results, name, fixtureRoot);
    if (
      results.modelId !== "gemma3-270m-instruct" ||
      results.webLlmBaselineVersion !== "0.2.84" ||
      results.prompt2048Passed !== true ||
      results.prompt4096Passed !== true ||
      results.shapeCacheHangObserved !== false
    )
      throw new Error(
        "Gemma long-prefill evidence did not rule out the shape-cache hang",
      );
    return;
  }
  if (name !== "runtimeLifecycle")
    throw new Error(`unknown runtime evidence gate ${name}`);
  exactKeys(
    results,
    [
      "fixturePath",
      "fixtureSha256",
      "stopPassed",
      "unloadPassed",
      "repeatedLoadPassed",
      "adapterLossPassed",
      "workerRestartPassed",
    ],
    `${name} results`,
  );
  await validateFixtureBinding(results, name, fixtureRoot);
  if (
    [
      "stopPassed",
      "unloadPassed",
      "repeatedLoadPassed",
      "adapterLossPassed",
      "workerRestartPassed",
    ].some((field) => results[field] !== true)
  )
    throw new Error("browser runtime lifecycle evidence is incomplete");
}

export async function validateAuthoringResults(
  name,
  results,
  fixtureRoot = repositoryRoot,
  runtime = null,
) {
  const common = ["fixturePath", "fixtureSha256", "command"];
  if (typeof results.command !== "string" || !results.command.trim()) {
    throw new Error(`${name} evidence command is invalid`);
  }
  if (name === "activationCapture") {
    exactKeys(
      results,
      [
        ...common,
        "postBlockBoundaryPassed",
        "selectedPromptPositionsPassed",
        "finalDecodePositionPassed",
        "workerIsolationPassed",
        "maxCaptureRelativeError",
      ],
      `${name} results`,
    );
    await validateFixtureBinding(results, name, fixtureRoot);
    requireTrueResults(
      results,
      [
        "postBlockBoundaryPassed",
        "selectedPromptPositionsPassed",
        "finalDecodePositionPassed",
        "workerIsolationPassed",
      ],
      name,
    );
    requireBoundedError(
      results.maxCaptureRelativeError,
      0.01,
      `${name} capture relative error`,
    );
    return;
  }
  if (name === "fittingKernelParity") {
    exactKeys(
      results,
      [
        ...common,
        "neutralCenteringMaxAbsoluteError",
        "whiteningMaxAbsoluteError",
        "pcaProjectionMaxError",
        "spectralProjectionMaxError",
        "rbfMaxAbsoluteError",
        "geometryMaxAbsoluteError",
        "finiteOutputsPassed",
      ],
      `${name} results`,
    );
    await validateFixtureBinding(results, name, fixtureRoot);
    for (const field of [
      "neutralCenteringMaxAbsoluteError",
      "whiteningMaxAbsoluteError",
      "rbfMaxAbsoluteError",
      "geometryMaxAbsoluteError",
    ])
      requireBoundedError(results[field], 1e-4, `${name} ${field}`);
    for (const field of [
      "pcaProjectionMaxError",
      "spectralProjectionMaxError",
    ]) {
      requireBoundedError(results[field], 1e-3, `${name} ${field}`);
    }
    requireTrueResults(results, ["finiteOutputsPassed"], name);
    return;
  }
  if (name === "topologyOrchestration") {
    exactKeys(
      results,
      [
        ...common,
        "flatSelectionPassed",
        "curvedSelectionPassed",
        "periodicSelectionPassed",
        "automaticSelectionPassed",
        "maxEvaluatedOutputError",
      ],
      `${name} results`,
    );
    await validateFixtureBinding(results, name, fixtureRoot);
    requireTrueResults(
      results,
      [
        "flatSelectionPassed",
        "curvedSelectionPassed",
        "periodicSelectionPassed",
        "automaticSelectionPassed",
      ],
      name,
    );
    requireBoundedError(
      results.maxEvaluatedOutputError,
      1e-3,
      `${name} evaluated output error`,
    );
    return;
  }
  if (name === "manifoldSerialization") {
    exactKeys(
      results,
      [
        ...common,
        "manifoldV10RoundTripPassed",
        "safetensorsRoundTripPassed",
        "drowseArchiveRoundTripPassed",
        "pythonBrowserCompatibilityPassed",
        "incompatibleFingerprintRejected",
      ],
      `${name} results`,
    );
    await validateFixtureBinding(results, name, fixtureRoot);
    requireTrueResults(
      results,
      [
        "manifoldV10RoundTripPassed",
        "safetensorsRoundTripPassed",
        "drowseArchiveRoundTripPassed",
        "pythonBrowserCompatibilityPassed",
        "incompatibleFingerprintRejected",
      ],
      name,
    );
    return;
  }
  if (name === "instrumentIntegration") {
    exactKeys(
      results,
      [
        ...common,
        "coreGeometryPassed",
        "jlensV6Passed",
        "saeV1Passed",
        "exactBindingValidationPassed",
        "incompatibleBindingRejected",
        "fp32TensorValidationPassed",
        "runtimeIdentitySha256",
        "webLlmSha256",
        "modelArtifactManifestSha256",
        "convertedManifestSha256",
        "modelLibrarySha256",
        "coreArtifactManifestSha256",
        "jlensArtifactManifestSha256",
        "saeArtifactManifestSha256",
      ],
      `${name} results`,
    );
    await validateFixtureBinding(results, name, fixtureRoot);
    requireTrueResults(
      results,
      [
        "coreGeometryPassed",
        "jlensV6Passed",
        "saeV1Passed",
        "exactBindingValidationPassed",
        "incompatibleBindingRejected",
        "fp32TensorValidationPassed",
      ],
      name,
    );
    validateObservedArtifactResults(results, runtime, name);
    return;
  }
  if (name === "sharedGoldenResults") {
    exactKeys(
      results,
      [
        ...common,
        "expressionParsingPassed",
        "treeMutationsPassed",
        "measurementsPassed",
        "structuredProgramsPassed",
        "samplingPassed",
        "manifoldGeometryPassed",
        "fittingPassed",
        "reciprocalArtifactsPassed",
        "artifactCompatibilityRejectionPassed",
        "maxNumericError",
      ],
      `${name} results`,
    );
    await validateFixtureBinding(results, name, fixtureRoot);
    requireTrueResults(
      results,
      [
        "expressionParsingPassed",
        "treeMutationsPassed",
        "measurementsPassed",
        "structuredProgramsPassed",
        "samplingPassed",
        "manifoldGeometryPassed",
        "fittingPassed",
        "reciprocalArtifactsPassed",
        "artifactCompatibilityRejectionPassed",
      ],
      name,
    );
    requireBoundedError(results.maxNumericError, 1e-3, `${name} numeric error`);
    return;
  }
  if (name === "physicalBrowserFitting") {
    exactKeys(
      results,
      [
        ...common,
        "desktopChromePassed",
        "androidChromePassed",
        "nonFallbackHardwareAdapterConfirmed",
        "opfsSpoolingPassed",
        "uiResponsivePassed",
        "fitCompleted",
        "maxMainThreadTaskMs",
        "desktopMeasuredAt",
        "androidMeasuredAt",
        "runtimeIdentitySha256",
        "webLlmSha256",
        "modelArtifactManifestSha256",
        "convertedManifestSha256",
        "modelLibrarySha256",
        "coreArtifactManifestSha256",
        "jlensArtifactManifestSha256",
        "saeArtifactManifestSha256",
      ],
      `${name} results`,
    );
    await validateFixtureBinding(results, name, fixtureRoot);
    requireTrueResults(
      results,
      [
        "desktopChromePassed",
        "androidChromePassed",
        "nonFallbackHardwareAdapterConfirmed",
        "opfsSpoolingPassed",
        "uiResponsivePassed",
        "fitCompleted",
      ],
      name,
    );
    requireBoundedError(
      results.maxMainThreadTaskMs,
      50,
      `${name} main-thread task duration`,
    );
    for (const field of ["desktopMeasuredAt", "androidMeasuredAt"]) {
      const timestamp = Date.parse(results[field]);
      if (
        !Number.isFinite(timestamp) ||
        timestamp > Date.now() + 5 * 60 * 1000 ||
        timestamp < Date.now() - 30 * 24 * 60 * 60 * 1000
      ) throw new Error(`${name} ${field} is stale or invalid`);
    }
    validateObservedArtifactResults(results, runtime, name);
    return;
  }
  throw new Error(`unknown authoring evidence gate ${name}`);
}

function validateObservedArtifactResults(results, runtime, name) {
  for (const field of [
    "runtimeIdentitySha256",
    "webLlmSha256",
    "modelArtifactManifestSha256",
    "convertedManifestSha256",
    "modelLibrarySha256",
    "coreArtifactManifestSha256",
    "jlensArtifactManifestSha256",
    "saeArtifactManifestSha256",
  ]) {
    if (!SHA256.test(results[field] ?? ""))
      throw new Error(`${name} ${field} is invalid`);
  }
  if (runtime === null) return;
  const model = runtime.models.find(
    (candidate) => lockedRuntimeIdentitySha256(runtime, candidate) === results.runtimeIdentitySha256,
  );
  if (
    !model ||
    model.manifestSha256 !== results.convertedManifestSha256 ||
    model.librarySha256 !== results.modelLibrarySha256
  ) {
    throw new Error(`${name} observed artifacts do not match runtime-lock`);
  }
}

function validateDistribution(value) {
  exactKeys(
    value,
    [
      "$schema",
      "schemaVersion",
      "status",
      "catalogUrl",
      "signatureUrl",
      "minimumAcceptedSequence",
      "publicKeys",
      "allowedCatalogRedirectOrigins",
      "allowedArtifactRedirectOrigins",
    ],
    "distribution lock",
  );
  if (value.schemaVersion !== 1)
    throw new Error("unsupported distribution lock schema");
  if (!["feasibility-required", "verified"].includes(value.status)) {
    throw new Error("invalid browser distribution lock status");
  }
  const expectedBase =
    "https://huggingface.co/logitsml/drowse-web-catalog/resolve/main/";
  if (value.catalogUrl !== `${expectedBase}catalog.json`) {
    throw new Error("unexpected browser catalog URL");
  }
  if (value.signatureUrl !== `${expectedBase}catalog.sig.json`) {
    throw new Error("unexpected browser catalog signature URL");
  }
  if (
    !Number.isSafeInteger(value.minimumAcceptedSequence) ||
    value.minimumAcceptedSequence < 1
  ) {
    throw new Error("invalid minimum accepted catalog sequence");
  }
  if (!Array.isArray(value.publicKeys) || value.publicKeys.length > 2) {
    throw new Error("browser catalog key ring must contain at most two keys");
  }
  const keyIds = new Set();
  for (const key of value.publicKeys) {
    exactKeys(key, ["keyId", "ed25519PublicKeyBase64"], "catalog public key");
    if (typeof key.keyId !== "string" || !key.keyId || keyIds.has(key.keyId)) {
      throw new Error("catalog public key IDs must be unique and non-empty");
    }
    keyIds.add(key.keyId);
    if (typeof key.ed25519PublicKeyBase64 !== "string") {
      throw new Error(`catalog public key ${key.keyId} is invalid`);
    }
    const decoded = Buffer.from(key.ed25519PublicKeyBase64, "base64");
    if (
      decoded.length !== 32 ||
      decoded.toString("base64") !== key.ed25519PublicKeyBase64
    )
      throw new Error(`catalog public key ${key.keyId} is invalid`);
  }
  validateOrigins(value.allowedCatalogRedirectOrigins, "catalog redirect");
  validateOrigins(value.allowedArtifactRedirectOrigins, "artifact redirect");
  if (value.status === "verified") {
    if (value.publicKeys.length !== 2) {
      throw new Error(
        "verified distribution must embed current and next catalog keys",
      );
    }
    if (
      !value.allowedCatalogRedirectOrigins.some((origin) => origin === "https://huggingface.co")
    ) {
      throw new Error("verified distribution must allow the catalog origin");
    }
    if (value.allowedArtifactRedirectOrigins.length === 0) {
      throw new Error(
        "verified distribution must pin preflighted artifact redirect origins",
      );
    }
  }
}

function validateOrigins(values, label) {
  if (!Array.isArray(values) || new Set(values).size !== values.length) {
    throw new Error(`${label} origins must be a unique array`);
  }
  for (const value of values) {
    let url;
    try {
      url = new URL(value);
    } catch {
      throw new Error(`${label} origin is invalid`);
    }
    if (url.protocol !== "https:" || url.origin !== value) {
      throw new Error(`${label} origin must be an HTTPS origin`);
    }
  }
}

export function validateHostedConnectSources(headers, distributionLock) {
  const match = /(?:^|;)\s*connect-src\s+([^;\n]+)/m.exec(headers);
  if (!match) throw new Error("hosted CSP has no connect-src directive");
  const actual = new Set(match[1].trim().split(/\s+/));
  const expected = new Set([
    "'self'",
    "https://www.neuronpedia.org",
    ...distributionLock.allowedCatalogRedirectOrigins,
    ...distributionLock.allowedArtifactRedirectOrigins,
  ]);
  if (
    actual.size !== expected.size ||
    [...actual].some((source) => !expected.has(source))
  ) {
    throw new Error(
      "hosted CSP connect-src does not match distribution and description sources",
    );
  }
  if ([...actual].some((source) => source.includes("*"))) {
    throw new Error("hosted CSP connect-src must not contain wildcard origins");
  }
}

export function validateWebLlmDependency(
  manifest,
  packageLock,
  forkPin,
  packagePin,
  release = false,
) {
  const declarations = {
    ...(manifest.dependencies ?? {}),
    ...(manifest.devDependencies ?? {}),
  };
  if (release && !COMMIT.test(forkPin.commit ?? "")) {
    throw new Error(
      "integrated browser backend requires a pinned WebLLM fork commit",
    );
  }
  if (release && packagePin.sourceCommit !== forkPin.commit) {
    throw new Error(
      "integrated browser backend requires a WebLLM archive bound to its fork commit",
    );
  }
  const expectedSpecifier = `file:../${packagePin.path}`;
  const localSpecifier = declarations["@drowse/web-llm"];
  if (localSpecifier !== expectedSpecifier) {
    throw new Error(
      "browser backend must use the runtime-locked Drowse WebLLM archive",
    );
  }
  const locked = packageLock.packages?.["node_modules/@drowse/web-llm"];
  if (!locked || locked.resolved !== expectedSpecifier) {
    throw new Error(
      "package lock does not resolve the runtime-locked Drowse WebLLM archive",
    );
  }
}

export function validateToolchainBindings(
  toolchain,
  manifest,
  manifestBytes,
  compilerRecipeBytes,
) {
  exactKeys(toolchain, [
    "forkManifestSha256",
    "mlcLlmFork",
    "webLlmFork",
    "webLlmPackage",
    "tvmRepository",
    "tvmCommit",
    "tvmFfiCommit",
    "emsdkCommit",
    "emccVersion",
    "emccRevision",
    "emccPlatform",
    "llvmVersion",
    "llvmArtifacts",
    "compilerImage",
  ], "toolchain");
  if (
    !SHA256.test(toolchain.forkManifestSha256 ?? "") ||
    createHash("sha256").update(manifestBytes).digest("hex") !==
      toolchain.forkManifestSha256
  ) {
    throw new Error("runtime lock does not bind the exact fork overlay manifest");
  }
  const overlays = new Map(
    (manifest?.overlays ?? []).map((overlay) => [overlay?.id, overlay]),
  );
  for (const [name, overlayId] of Object.entries({
    mlcLlmFork: "mlc-llm-drowse",
    webLlmFork: "web-llm-drowse",
  })) {
    const pin = toolchain[name];
    const overlay = overlays.get(overlayId);
    exactKeys(pin, ["repository", "commit"], name);
    if (
      !overlay || pin.repository !== overlay.intendedForkRepository ||
      !COMMIT.test(overlay.baseCommit ?? "")
    ) {
      throw new Error(`${name} does not match the exact fork overlay identity`);
    }
    nullableMatch(pin.commit, COMMIT, `${name} commit`);
    if (pin.commit !== null && pin.commit === overlay.baseCommit) {
      throw new Error(`${name} commit is the unmodified overlay base`);
    }
  }
  const manifestBindings = [
    "tvmRepository",
    "tvmCommit",
    "tvmFfiCommit",
    "emsdkCommit",
    "emccVersion",
    "emccRevision",
    "emccPlatform",
    "llvmVersion",
  ];
  for (const field of manifestBindings) {
    if (toolchain[field] !== manifest?.[field]) {
      throw new Error(`runtime lock ${field} differs from the compiler overlay manifest`);
    }
  }
  nullableMatch(toolchain.tvmCommit, COMMIT, "TVM commit");
  for (const field of ["tvmFfiCommit", "emsdkCommit", "emccRevision"]) {
    if (!COMMIT.test(toolchain[field] ?? "")) {
      throw new Error(`runtime lock ${field} is invalid`);
    }
  }
  if (
    !Array.isArray(toolchain.llvmArtifacts) ||
    JSON.stringify(toolchain.llvmArtifacts) !== JSON.stringify(manifest?.llvmArtifacts)
  ) {
    throw new Error("runtime lock LLVM artifacts differ from the compiler overlay manifest");
  }
  const llvmPlatforms = new Set();
  for (const artifact of toolchain.llvmArtifacts) {
    exactKeys(artifact, ["platform", "archive", "sha256"], "LLVM artifact");
    if (
      !["linux/amd64", "linux/arm64"].includes(artifact.platform) ||
      llvmPlatforms.has(artifact.platform) ||
      typeof artifact.archive !== "string" || !artifact.archive ||
      !SHA256.test(artifact.sha256 ?? "")
    ) {
      throw new Error("runtime lock LLVM artifact identity is invalid");
    }
    llvmPlatforms.add(artifact.platform);
  }
  if (!sameStringSet([...llvmPlatforms], ["linux/amd64", "linux/arm64"])) {
    throw new Error("runtime lock does not close both compiler architectures");
  }
  exactKeys(
    toolchain.compilerImage,
    ["reference", "digest", "recipeSha256"],
    "compiler image",
  );
  if (
    !SHA256.test(toolchain.compilerImage.recipeSha256 ?? "") ||
    createHash("sha256").update(compilerRecipeBytes).digest("hex") !==
      toolchain.compilerImage.recipeSha256
  ) {
    throw new Error("runtime lock does not bind the exact compiler recipe");
  }
  const compilerRecipeSource = Buffer.from(compilerRecipeBytes).toString("utf8");
  const requiredArguments = [
    ["TVM_COMMIT", toolchain.tvmCommit],
    ["TVM_FFI_COMMIT", toolchain.tvmFfiCommit],
    ["MLC_COMMIT", overlays.get("mlc-llm-drowse").baseCommit],
    ["EMSDK_COMMIT", toolchain.emsdkCommit],
    ["EMCC_VERSION", toolchain.emccVersion],
  ];
  for (const [name, value] of requiredArguments) {
    if (!compilerRecipeSource.split("\n").includes(`ARG ${name}=${value}`)) {
      throw new Error(`compiler recipe does not bind ${name}`);
    }
  }
  for (const artifact of toolchain.llvmArtifacts) {
    const suffix = artifact.platform.endsWith("amd64") ? "AMD64" : "ARM64";
    if (
      !compilerRecipeSource.includes(`ARG LLVM_ARCHIVE_${suffix}=${artifact.archive}`) ||
      !compilerRecipeSource.includes(`ARG LLVM_SHA256_${suffix}=${artifact.sha256}`)
    ) {
      throw new Error(`compiler recipe does not bind LLVM ${artifact.platform}`);
    }
  }
  nullableMatch(
    toolchain.compilerImage.digest,
    IMAGE_DIGEST,
    "compiler image digest",
  );
  if (
    (toolchain.compilerImage.reference === null) !==
    (toolchain.compilerImage.digest === null)
  ) {
    throw new Error("compiler image reference and digest must be pinned together");
  }
  if (
    toolchain.compilerImage.reference !== null &&
    !IMAGE_REFERENCE.test(toolchain.compilerImage.reference)
  ) {
    throw new Error(
      "compiler image reference must be a qualified immutable registry path",
    );
  }
  nullableMatch(
    toolchain.webLlmPackage?.sourceCommit,
    COMMIT,
    "WebLLM package source commit",
  );
  if (toolchain.webLlmPackage?.sourceCommit !== toolchain.webLlmFork.commit) {
    throw new Error("WebLLM package source commit differs from the fork commit");
  }
}

async function validateWebLlmPackageArchive(pin, forkPin) {
  exactKeys(
    pin,
    ["path", "sha256", "name", "version", "repository", "sourceCommit"],
    "WebLLM package pin",
  );
  if (
    typeof pin.path !== "string" ||
    !/^browser-runtime\/vendor\/drowse-web-llm-[0-9A-Za-z.-]+\.tgz$/.test(pin.path) ||
    !SHA256.test(pin.sha256) || pin.name !== "@drowse/web-llm" ||
    !/^[0-9]+\.[0-9]+\.[0-9]+-drowse\.[0-9]+$/.test(pin.version ?? "") ||
    pin.repository !== "https://github.com/a9lim/web-llm-polythetic"
  ) {
    throw new Error("WebLLM package pin is invalid");
  }
  const versionMatch = /drowse-web-llm-(.+)\.tgz$/u.exec(pin.path);
  if (versionMatch?.[1] !== pin.version) {
    throw new Error("WebLLM package version differs from its archive filename");
  }
  nullableMatch(pin.sourceCommit, COMMIT, "WebLLM package source commit");
  if (pin.sourceCommit !== forkPin.commit) {
    throw new Error("WebLLM package source commit differs from the fork commit");
  }
  const archive = resolve(repositoryRoot, pin.path);
  const info = await lstat(archive);
  if (!info.isFile() || info.isSymbolicLink()) {
    throw new Error("WebLLM package archive must be a regular file");
  }
  const actual = createHash("sha256").update(await readFile(archive)).digest("hex");
  if (actual !== pin.sha256) {
    throw new Error("WebLLM package archive does not match its runtime-lock digest");
  }
  let packageManifest;
  try {
    packageManifest = JSON.parse(execFileSync(
      "tar",
      ["-xOf", archive, "package/package.json"],
      { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] },
    ));
  } catch (error) {
    throw new Error("WebLLM package archive has no readable package manifest", {
      cause: error,
    });
  }
  const packageRepository = (
    typeof packageManifest.repository === "string"
      ? packageManifest.repository
      : packageManifest.repository?.url
  )?.replace(/^git\+/u, "").replace(/\.git$/u, "");
  if (
    packageManifest.name !== pin.name || packageManifest.version !== pin.version ||
    packageManifest.license !== "Apache-2.0" || packageRepository !== pin.repository ||
    (packageManifest.drowseSourceCommit ?? null) !== pin.sourceCommit
  ) {
    throw new Error("WebLLM package manifest differs from its runtime-lock provenance");
  }
}

export function validateInstalledWebLlmDependency(
  manifest,
  packageLock,
  installed,
) {
  const specifier = manifest.dependencies?.["@drowse/web-llm"] ??
    manifest.devDependencies?.["@drowse/web-llm"];
  const locked = packageLock.packages?.["node_modules/@drowse/web-llm"];
  if (!locked || typeof locked.version !== "string") {
    throw new Error("package lock has no installed Drowse WebLLM version");
  }
  if (
    installed?.name !== "@drowse/web-llm" ||
    installed.version !== locked.version
  ) {
    throw new Error(
      "installed Drowse WebLLM does not match the exact package lock",
    );
  }
  if (typeof specifier === "string" && specifier.startsWith("file:")) {
    const match = /drowse-web-llm-(.+)\.tgz$/u.exec(specifier);
    if (!match || match[1] !== installed.version) {
      throw new Error(
        "installed Drowse WebLLM version does not match its local archive",
      );
    }
  }
}

export function parseArguments(args, environment = process.env) {
  let release = false;
  let cliRevision = null;
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--release") {
      release = true;
      continue;
    }
    if (argument === "--drowse-revision") {
      if (cliRevision !== null || index + 1 >= args.length) {
        throw new Error("--drowse-revision requires exactly one revision");
      }
      cliRevision = args[index + 1];
      index += 1;
      continue;
    }
    if (argument.startsWith("--drowse-revision=")) {
      if (cliRevision !== null) {
        throw new Error("--drowse-revision may be supplied only once");
      }
      cliRevision = argument.slice("--drowse-revision=".length);
      continue;
    }
    throw new Error(`unknown runtime-lock option ${argument}`);
  }
  const candidates = [
    ["--drowse-revision", cliRevision],
    ["DROWSE_RELEASE_REVISION", environment.DROWSE_RELEASE_REVISION ?? null],
    ["CF_PAGES_COMMIT_SHA", environment.CF_PAGES_COMMIT_SHA ?? null],
  ].filter(([, value]) => value !== null && value !== "");
  for (const [source, value] of candidates) {
    if (typeof value !== "string" || !COMMIT.test(value)) {
      throw new Error(
        `${source} must be an exact lowercase 40-character Git revision`,
      );
    }
  }
  const revisions = new Set(candidates.map(([, value]) => value));
  if (revisions.size > 1) {
    throw new Error("supplied Drowse release revisions disagree");
  }
  return { release, drowseRevision: candidates[0]?.[1] ?? null };
}

function validateReleaseRevision(expectedDrowseRevision) {
  if (expectedDrowseRevision === null) {
    throw new Error(
      "release checks require --drowse-revision, DROWSE_RELEASE_REVISION, or CF_PAGES_COMMIT_SHA",
    );
  }
  let currentRevision;
  try {
    currentRevision = execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: repositoryRoot,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    }).trim();
  } catch (error) {
    throw new Error(
      "release checks could not resolve the current Git revision",
      { cause: error },
    );
  }
  if (
    !COMMIT.test(currentRevision) ||
    currentRevision !== expectedDrowseRevision
  ) {
    throw new Error(
      "the supplied Drowse release revision does not match the current Git revision",
    );
  }
}

function validateEvidenceSetRevision(revision, hasRecords, status, label) {
  const requiresRevision = hasRecords || status !== "feasibility-required";
  if (!requiresRevision) {
    if (revision !== null)
      throw new Error(
        `${label} must keep a null revision until evidence exists`,
      );
    return;
  }
  if (typeof revision !== "string" || !COMMIT.test(revision)) {
    throw new Error(`${label} is not bound to an exact tested Drowse revision`);
  }
}

export function validateEvidenceRevisionAgreement(revisions) {
  const populated = revisions.filter((revision) => revision !== null);
  if (
    populated.some(
      (revision) => typeof revision !== "string" || !COMMIT.test(revision),
    )
  ) {
    throw new Error(
      "release evidence contains an invalid tested Drowse revision",
    );
  }
  const unique = new Set(populated);
  if (unique.size > 1) {
    throw new Error(
      "release evidence sets do not agree on the tested Drowse revision",
    );
  }
  return populated[0] ?? null;
}

export function validateReleaseEvidenceProvenance(
  testedDrowseRevision,
  releaseHeadRevision,
  currentRuntimeLockIdentitySha256,
  root = repositoryRoot,
) {
  if (!COMMIT.test(testedDrowseRevision ?? "")) {
    throw new Error(
      "release evidence does not name an exact tested Drowse revision",
    );
  }
  if (!COMMIT.test(releaseHeadRevision ?? "")) {
    throw new Error(
      "release evidence provenance requires an exact release HEAD revision",
    );
  }
  const runGit = (arguments_, options = {}) =>
    execFileSync("git", arguments_, {
      cwd: root,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      ...options,
    });
  let head;
  try {
    head = runGit(["rev-parse", "HEAD"]).trim();
  } catch (error) {
    throw new Error("release evidence provenance could not resolve Git HEAD", {
      cause: error,
    });
  }
  if (head !== releaseHeadRevision) {
    throw new Error(
      "release evidence provenance does not match the current Git revision",
    );
  }
  let dirty;
  try {
    dirty = runGit(["status", "--porcelain", "--untracked-files=all"]).trim();
  } catch (error) {
    throw new Error(
      "release evidence provenance could not inspect the Git worktree",
      {
        cause: error,
      },
    );
  }
  if (dirty) {
    throw new Error(
      "release evidence provenance requires a clean Git worktree",
    );
  }
  try {
    execFileSync(
      "git",
      [
        "merge-base",
        "--is-ancestor",
        testedDrowseRevision,
        releaseHeadRevision,
      ],
      { cwd: root, stdio: "ignore" },
    );
  } catch (error) {
    throw new Error(
      "the tested Drowse revision is not an ancestor of the release HEAD",
      {
        cause: error,
      },
    );
  }
  let changedPaths;
  try {
    changedPaths = runGit([
      "diff",
      "--name-only",
      "--no-renames",
      "-z",
      `${testedDrowseRevision}..${releaseHeadRevision}`,
    ])
      .split("\0")
      .filter(Boolean);
  } catch (error) {
    throw new Error(
      "release evidence provenance could not inspect source drift",
      { cause: error },
    );
  }
  const forbidden = changedPaths.filter(
    (path) => !isAllowedEvidencePromotionPath(path),
  );
  if (forbidden.length > 0) {
    throw new Error(
      `release executable source differs from the tested revision: ${forbidden.join(", ")}`,
    );
  }
  if (changedPaths.includes("webui/src/hosted/runtime/browserAuthoring.ts")) {
    let testedMarker;
    let releaseMarker;
    try {
      testedMarker = runGit([
        "show",
        `${testedDrowseRevision}:webui/src/hosted/runtime/browserAuthoring.ts`,
      ]);
      releaseMarker = runGit([
        "show",
        `${releaseHeadRevision}:webui/src/hosted/runtime/browserAuthoring.ts`,
      ]);
    } catch (error) {
      throw new Error("authoring capability promotion marker is unavailable", {
        cause: error,
      });
    }
    validateAuthoringCapabilityPromotion(testedMarker, releaseMarker);
  }
  let testedRuntimeLock;
  let releaseRuntimeLock;
  try {
    testedRuntimeLock = JSON.parse(
      runGit([
        "show",
        `${testedDrowseRevision}:browser-runtime/runtime-lock.json`,
      ]),
    );
    releaseRuntimeLock = JSON.parse(
      runGit([
        "show",
        `${releaseHeadRevision}:browser-runtime/runtime-lock.json`,
      ]),
    );
  } catch (error) {
    throw new Error(
      "the tested or release revision has no valid browser runtime lock",
      {
        cause: error,
      },
    );
  }
  const releaseRuntimeIdentity = runtimeLockIdentitySha256(releaseRuntimeLock);
  if (
    !SHA256.test(currentRuntimeLockIdentitySha256 ?? "") ||
    releaseRuntimeIdentity !== currentRuntimeLockIdentitySha256 ||
    runtimeLockIdentitySha256(testedRuntimeLock) !== releaseRuntimeIdentity
  ) {
    throw new Error(
      "the canonical browser runtime identity drifted after testing",
    );
  }
}

function isAllowedEvidencePromotionPath(path) {
  return (
    new Set([
      "browser-runtime/benchmark-evidence.json",
      "browser-runtime/runtime-feasibility-evidence.json",
      "browser-runtime/authoring-evidence.json",
      "browser-runtime/runtime-lock.json",
      "browser-runtime/distribution-lock.json",
      "webui/public-hosted/_headers",
      "webui/src/hosted/runtime/browserAuthoring.ts",
    ]).has(path) ||
    /^browser-runtime\/evidence\/[a-z0-9][a-z0-9._/-]*\.json$/.test(path)
  );
}

export function validateAuthoringIntegrationState({
  implementationIntegrated,
  capabilityIntegrated,
  release = false,
}) {
  if (capabilityIntegrated && !implementationIntegrated) {
    throw new Error(
      "browser authoring capability requires an integrated implementation",
    );
  }
  if (release && !capabilityIntegrated) {
    throw new Error(
      "browser fitting and authoring runtime is not release-enabled",
    );
  }
}

export function validateAuthoringCapabilityWiring(
  capabilitiesSource,
  authoringIntegrated,
) {
  if (!authoringIntegrated) return;
  const requiredBindings = [
    /browserAuthoringAvailableForChannel\s*\(/,
    /runtimeIntegrated:\s*BROWSER_AUTHORING_RUNTIME_INTEGRATED/,
    /modelBackendIntegrated:\s*BROWSER_MODEL_BACKEND_INTEGRATED/,
    /fitting:\s*fittingReady/,
  ];
  if (requiredBindings.some((pattern) => !pattern.test(capabilitiesSource))) {
    throw new Error(
      "browser authoring integration marker is not wired to capabilities",
    );
  }
}

export function validateAuthoringSpoolIntegrationState({
  implementationIntegrated,
  capabilityIntegrated,
  deprecatedCapabilityMarkerPresent,
  coordinatorUsesActivationSpool,
  fittingWorkerReadsActivationSpool,
}) {
  if (deprecatedCapabilityMarkerPresent) {
    throw new Error(
      "activation spool must not publish an independent capability marker",
    );
  }
  const spoolIntegrated =
    coordinatorUsesActivationSpool && fittingWorkerReadsActivationSpool;
  if (implementationIntegrated && !spoolIntegrated) {
    throw new Error(
      "integrated browser authoring requires the production activation-spool path",
    );
  }
  if (capabilityIntegrated && !implementationIntegrated) {
    throw new Error(
      "browser authoring capability cannot precede its spool-backed implementation",
    );
  }
}

export function validateAuthoringCapabilityPromotion(
  testedSource,
  releaseSource,
) {
  const implementation =
    /export const BROWSER_AUTHORING_IMPLEMENTATION_INTEGRATED\s*=\s*true\s*;/;
  const integrated =
    /export const BROWSER_AUTHORING_RUNTIME_INTEGRATED\s*=\s*true\s*;/;
  if (
    !implementation.test(testedSource) ||
    !integrated.test(testedSource) ||
    testedSource !== releaseSource
  ) {
    throw new Error(
      "authoring evidence promotion must not change the integrated runtime source",
    );
  }
}

async function validateFixtureBinding(results, evidenceType, fixtureRoot) {
  if (
    typeof results.fixturePath !== "string" ||
    !/^[a-zA-Z0-9][a-zA-Z0-9._/-]*$/.test(results.fixturePath) ||
    isAbsolute(results.fixturePath) ||
    results.fixturePath.split("/").includes("..") ||
    !SHA256.test(results.fixtureSha256)
  )
    throw new Error(`${evidenceType} fixture binding is invalid`);
  const resolved = resolve(fixtureRoot, results.fixturePath);
  await requireTrackedRegularFile(
    resolved,
    results.fixturePath,
    fixtureRoot,
    `${evidenceType} fixture`,
  );
  const bytes = await readFile(resolved);
  if (
    createHash("sha256").update(bytes).digest("hex") !== results.fixtureSha256
  ) {
    throw new Error(
      `${evidenceType} fixture digest does not match its checked-in content`,
    );
  }
}

async function requireTrackedRegularFile(
  filePath,
  repositoryPath,
  root,
  label,
) {
  const relativePath = relative(root, filePath);
  if (
    !relativePath ||
    relativePath === ".." ||
    relativePath.startsWith(`..${sep}`) ||
    isAbsolute(relativePath)
  )
    throw new Error(`${label} escapes the repository`);
  const [rootRealPath, metadata] = await Promise.all([
    realpath(root),
    lstat(filePath).catch((error) => {
      throw new Error(`${label} is unavailable`, { cause: error });
    }),
  ]);
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    throw new Error(`${label} must be a regular checked-in file`);
  }
  const fileRealPath = await realpath(filePath);
  const realRelativePath = relative(rootRealPath, fileRealPath);
  if (
    !realRelativePath ||
    realRelativePath === ".." ||
    realRelativePath.startsWith(`..${sep}`) ||
    isAbsolute(realRelativePath)
  )
    throw new Error(`${label} resolves outside the repository`);
  try {
    execFileSync("git", ["ls-files", "--error-unmatch", "--", repositoryPath], {
      cwd: root,
      stdio: "ignore",
    });
    const committedObject = execFileSync(
      "git",
      ["rev-parse", `HEAD:${repositoryPath}`],
      {
        cwd: root,
        encoding: "utf8",
        stdio: ["ignore", "pipe", "pipe"],
      },
    ).trim();
    const workingObject = execFileSync("git", ["hash-object", "--", filePath], {
      cwd: root,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    }).trim();
    if (!committedObject || committedObject !== workingObject) {
      throw new Error(`${label} differs from its checked-in content`);
    }
  } catch (error) {
    if (
      error instanceof Error &&
      error.message === `${label} differs from its checked-in content`
    ) {
      throw error;
    }
    throw new Error(`${label} is not checked into Git`, { cause: error });
  }
}

function requireTrueResults(results, fields, label) {
  if (fields.some((field) => results[field] !== true)) {
    throw new Error(`${label} has missing or failed required results`);
  }
}

export function validateFixtureToolPins(manifest, modelTool, compilerTool) {
  if (
    manifest?.hookAbi !== "post-block-residual-v4" ||
    manifest?.structuredHookProfile !== "standard-v3" ||
    manifest?.exactReadoutAbi !== "exact-readout-v1"
  ) {
    throw new Error(
      "fork overlay manifest does not attest the production exact-readout profile",
    );
  }
  const overlay = manifest?.overlays?.find(
    (item) => item.id === "mlc-llm-drowse",
  );
  if (
    !overlay ||
    !overlay.resultFiles ||
    typeof overlay.resultFiles !== "object"
  ) {
    throw new Error("MLC overlay manifest is unavailable to fixture tools");
  }
  for (const [path, digest] of Object.entries(overlay.resultFiles)) {
    if (
      typeof digest !== "string" ||
      !SHA256.test(digest) ||
      !modelTool.includes(JSON.stringify(path)) ||
      !modelTool.includes(JSON.stringify(digest)) ||
      !compilerTool.includes(JSON.stringify(path)) ||
      !compilerTool.includes(JSON.stringify(digest))
    ) {
      throw new Error(
        `browser fixture tools do not pin current MLC result ${path}`,
      );
    }
  }
  for (const name of REQUIRED_DROWSE_VM_FUNCTIONS) {
    if (!compilerTool.includes(JSON.stringify(name))) {
      throw new Error(
        `browser fixture compiler does not require VM function ${name}`,
      );
    }
  }
  if (!modelTool.includes(JSON.stringify("drowse_capture_special_token_ids"))) {
    throw new Error("browser fixture model omits exact capture token metadata");
  }
  if (
    !modelTool.includes(JSON.stringify("q4f16_1")) ||
    !compilerTool.includes(JSON.stringify("q4f16_1"))
  ) {
    throw new Error(
      "browser fixture tools cannot exercise the production q4f16_1 graph",
    );
  }
}

function requireBoundedError(value, maximum, label) {
  if (!Number.isFinite(value) || value < 0 || value > maximum) {
    throw new Error(`${label} exceeds ${maximum}`);
  }
}

function sameStringSet(value, expected) {
  return (
    Array.isArray(value) &&
    value.every((item) => typeof item === "string") &&
    new Set(value).size === value.length &&
    value.length === expected.length &&
    expected.every((item) => value.includes(item))
  );
}

function sameNumberSet(value, expected) {
  return (
    Array.isArray(value) &&
    value.every((item) => Number.isSafeInteger(item)) &&
    new Set(value).size === value.length &&
    value.length === expected.length &&
    expected.every((item) => value.includes(item))
  );
}

function exactKeys(value, keys, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (
    actual.length !== expected.length ||
    actual.some((key, index) => key !== expected[index])
  ) {
    throw new Error(`${label} has unknown or missing fields`);
  }
}

function nullableMatch(value, pattern, label) {
  if (value !== null && (typeof value !== "string" || !pattern.test(value))) {
    throw new Error(`${label} is invalid`);
  }
}

export function lockedRuntimeIdentitySha256(runtime, model) {
  const identity = {
    sourceModel: model.sourceRepository,
    sourceRevision: model.sourceRevision,
    convertedManifestSha256: model.manifestSha256,
    quantization: model.quantization,
    tokenizerSha256: model.tokenizerSha256,
    chatTemplateSha256: model.chatTemplateSha256,
    modelLibrarySha256: model.librarySha256,
    runtimeAbi: runtime.runtimeAbi,
    hookAbi: runtime.hookAbi,
    hiddenSize: model.hiddenSize,
    layerMap: model.layerMap,
  };
  if (Object.values(identity).some((value) => value === null)) return null;
  return digestCanonical(identity);
}

export function runtimeLockIdentitySha256(runtime) {
  if (!runtime || typeof runtime !== "object" || Array.isArray(runtime)) {
    throw new Error("browser runtime lock identity must be an object");
  }
  const { status: _status, ...identity } = runtime;
  return digestCanonical(identity);
}

export function digestCanonical(value) {
  return createHash("sha256").update(canonicalJson(value)).digest("hex");
}

function canonicalJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.entries(value)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
    .join(",")}}`;
}
