import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createHash, generateKeyPairSync } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const exec = promisify(execFile);
const dir = await mkdtemp(join(tmpdir(), "drowse-catalog-test-"));
const catalog = join(dir, "catalog.json");
const signature = join(dir, "catalog.sig.json");
const privateKeyPath = join(dir, "private.pem");
const publicKeyPath = join(dir, "public.pem");
const signScript = new URL("./sign-catalog.mjs", import.meta.url);
const verifyScript = new URL("./verify-catalog.mjs", import.meta.url);
const lockScript = new URL("./check-runtime-lock.mjs", import.meta.url);
const manifestScript = new URL("./manifest-artifacts.mjs", import.meta.url);
const repositoryRoot = fileURLToPath(new URL("../..", import.meta.url));
const scriptsDirectory = fileURLToPath(new URL(".", import.meta.url));
const lockEnvironment = {
  ...process.env,
  DROWSE_RELEASE_REVISION: "",
  CF_PAGES_COMMIT_SHA: "",
};

for (const name of await readdir(scriptsDirectory)) {
  if (!name.endsWith(".mjs")) continue;
  const source = await readFile(join(scriptsDirectory, name), "utf8");
  const viteImport = source.match(/import\s*\{([^}]*)\}\s*from\s*"vite";/s);
  const createServerImport = viteImport?.[1].match(
    /\bcreateServer(?:\s+as\s+([A-Za-z_$][\w$]*))?/,
  );
  if (!createServerImport) continue;
  const createServerName = createServerImport[1] ?? "createServer";
  const calls = [...source.matchAll(new RegExp(
    `\\b${createServerName}\\s*\\(\\s*(\\{[\\s\\S]*?\\})\\s*\\)`,
    "g",
  ))];
  assert.ok(calls.length > 0, `${name} imports Vite createServer without calling it`);
  for (const [, options] of calls) {
    assert.match(
      options,
      /\bserver\s*:\s*\{[^}]*\bwatch\s*:\s*null\b[^}]*\}/,
      `${name} must disable file watching for every scripted Vite server`,
    );
  }
}

const { privateKey, publicKey } = generateKeyPairSync("ed25519");
await Promise.all([
  writeFile(catalog, '{"schemaVersion":1,"sequence":7}\n'),
  writeFile(privateKeyPath, privateKey.export({ type: "pkcs8", format: "pem" })),
  writeFile(publicKeyPath, publicKey.export({ type: "spki", format: "pem" })),
]);

await exec(process.execPath, [signScript.pathname, catalog, privateKeyPath, "test-key", signature]);
const verified = await exec(process.execPath, [
  verifyScript.pathname,
  catalog,
  signature,
  publicKeyPath,
]);
assert.match(verified.stdout, /Verified catalog sequence 7/);

await writeFile(catalog, '{"schemaVersion":1,"sequence":8}\n');
await assert.rejects(
  exec(process.execPath, [verifyScript.pathname, catalog, signature, publicKeyPath]),
  /catalog signature verification failed/,
);

const envelope = JSON.parse(await readFile(signature, "utf8"));
assert.equal(envelope.algorithm, "Ed25519");
assert.equal(envelope.keyId, "test-key");

const developmentLock = await exec(process.execPath, [lockScript.pathname], {
  env: lockEnvironment,
});
const repositoryRuntimeLock = JSON.parse(await readFile(new URL("../../browser-runtime/runtime-lock.json", import.meta.url), "utf8"));
assert.ok(developmentLock.stdout.includes(`Browser runtime lock: ${repositoryRuntimeLock.status}`));
const repositoryDistributionLock = JSON.parse(await readFile(new URL("../../browser-runtime/distribution-lock.json", import.meta.url), "utf8"));
assert.ok(developmentLock.stdout.includes(`Browser distribution lock: ${repositoryDistributionLock.status}`));
const releaseReady = repositoryRuntimeLock.status === "verified" && repositoryDistributionLock.status === "verified";
await assert.rejects(
  exec(process.execPath, [lockScript.pathname, "--release"], { env: lockEnvironment }),
  releaseReady ? /release checks require --drowse-revision/ : /not publishable/,
);
const currentRevision = (await exec("git", ["rev-parse", "HEAD"], {
  cwd: repositoryRoot,
})).stdout.trim();
const explicitRelease = exec(process.execPath, [
    lockScript.pathname,
    "--release",
    "--drowse-revision",
    currentRevision,
  ], { env: lockEnvironment });
if (releaseReady) assert.match((await explicitRelease).stdout, /Browser runtime lock: verified/);
else await assert.rejects(explicitRelease, /not publishable/);
const environmentRelease = exec(
  process.execPath,
  [lockScript.pathname, "--release"],
  { env: { ...lockEnvironment, DROWSE_RELEASE_REVISION: currentRevision } },
);
if (releaseReady) assert.match((await environmentRelease).stdout, /Browser runtime lock: verified/);
else await assert.rejects(environmentRelease, /not publishable/);
await assert.rejects(
  exec(process.execPath, [
    lockScript.pathname,
    "--release",
    "--drowse-revision",
    "0".repeat(40),
  ], { env: lockEnvironment }),
  releaseReady ? /does not match the current Git revision/ : /not publishable/,
);
await assert.rejects(
  exec(process.execPath, [
    lockScript.pathname,
    "--drowse-revision",
    "0".repeat(40),
  ], {
    env: { ...lockEnvironment, DROWSE_RELEASE_REVISION: "1".repeat(40) },
  }),
  /supplied Drowse release revisions disagree/,
);

const fixtureRepository = join(dir, "fixture-repository");
const fixtureRelativePath = "fixtures/release-golden.json";
const fixturePath = join(fixtureRepository, fixtureRelativePath);
const fixtureBytes = Buffer.from('{"fixture":"checked-in"}\n');
await mkdir(join(fixtureRepository, "fixtures"), { recursive: true });
await writeFile(fixturePath, fixtureBytes);
await exec("git", ["init", "-q"], { cwd: fixtureRepository });
await exec("git", ["add", fixtureRelativePath], { cwd: fixtureRepository });
await exec("git", [
  "-c", "user.name=Drowse Test", "-c", "user.email=drowse@example.invalid",
  "commit", "-qm", "fixture",
], { cwd: fixtureRepository });
const fixtureSha256 = createHash("sha256").update(fixtureBytes).digest("hex");
const runtimeLockValidators = await import(lockScript.href);
const [runtimeToolchainLock, forkManifestBytes, compilerRecipeBytes] = await Promise.all([
  readFile(join(repositoryRoot, "browser-runtime", "runtime-lock.json"), "utf8").then(JSON.parse),
  readFile(join(repositoryRoot, "browser-runtime", "forks", "manifest.json")),
  readFile(join(repositoryRoot, "browser-runtime", "compiler", "Dockerfile")),
]);
const forkManifest = JSON.parse(forkManifestBytes.toString("utf8"));
assert.deepEqual(
  runtimeLockValidators.browserRuntimeReleaseBlockers({ ...runtimeToolchainLock, status: "verified" }),
  [],
);
assert.deepEqual(
  runtimeLockValidators.browserRuntimeReleaseBlockers({
    ...runtimeToolchainLock,
    status: "verified",
    toolchain: {
      ...runtimeToolchainLock.toolchain,
      mlcLlmFork: {
        ...runtimeToolchainLock.toolchain.mlcLlmFork,
        commit: "1".repeat(40),
      },
      webLlmFork: {
        ...runtimeToolchainLock.toolchain.webLlmFork,
        commit: "2".repeat(40),
      },
      webLlmPackage: {
        ...runtimeToolchainLock.toolchain.webLlmPackage,
        sourceCommit: "2".repeat(40),
      },
      compilerImage: {
        ...runtimeToolchainLock.toolchain.compilerImage,
        reference: "ghcr.io/a9lim/drowse-compiler:release",
        digest: `sha256:${"3".repeat(64)}`,
      },
    },
    models: runtimeToolchainLock.models.map((model, index) => ({
      ...model,
      convertedRevision: String(index + 4).repeat(40),
    })),
  }),
  [],
);
assert.doesNotThrow(() => runtimeLockValidators.validateToolchainBindings(
  runtimeToolchainLock.toolchain,
  forkManifest,
  forkManifestBytes,
  compilerRecipeBytes,
));
assert.throws(
  () => runtimeLockValidators.validateToolchainBindings(
    { ...runtimeToolchainLock.toolchain, forkManifestSha256: "0".repeat(64) },
    forkManifest,
    forkManifestBytes,
    compilerRecipeBytes,
  ),
  /exact fork overlay manifest/,
);
assert.throws(
  () => runtimeLockValidators.validateToolchainBindings(
    {
      ...runtimeToolchainLock.toolchain,
      compilerImage: {
        ...runtimeToolchainLock.toolchain.compilerImage,
        recipeSha256: "0".repeat(64),
      },
    },
    forkManifest,
    forkManifestBytes,
    compilerRecipeBytes,
  ),
  /exact compiler recipe/,
);
assert.throws(
  () => runtimeLockValidators.validateToolchainBindings(
    {
      ...runtimeToolchainLock.toolchain,
      mlcLlmFork: {
        ...runtimeToolchainLock.toolchain.mlcLlmFork,
        commit: forkManifest.overlays.find((overlay) => overlay.id === "mlc-llm-drowse").baseCommit,
      },
    },
    forkManifest,
    forkManifestBytes,
    compilerRecipeBytes,
  ),
  /unmodified overlay base/,
);
assert.throws(
  () => runtimeLockValidators.validateToolchainBindings(
    {
      ...runtimeToolchainLock.toolchain,
      webLlmPackage: {
        ...runtimeToolchainLock.toolchain.webLlmPackage,
        sourceCommit: "1".repeat(40),
      },
    },
    forkManifest,
    forkManifestBytes,
    compilerRecipeBytes,
  ),
  /source commit differs from the fork commit/,
);
assert.doesNotThrow(() => runtimeLockValidators.validateAuthoringIntegrationState({
  implementationIntegrated: false,
  capabilityIntegrated: false,
  evidenceStatus: "feasibility-required",
}));
const integratedSpoolState = {
  implementationIntegrated: true,
  capabilityIntegrated: false,
  deprecatedCapabilityMarkerPresent: false,
  coordinatorUsesActivationSpool: true,
  fittingWorkerReadsActivationSpool: true,
};
assert.doesNotThrow(() =>
  runtimeLockValidators.validateAuthoringSpoolIntegrationState(integratedSpoolState)
);
assert.throws(
  () => runtimeLockValidators.validateAuthoringSpoolIntegrationState({
    ...integratedSpoolState,
    deprecatedCapabilityMarkerPresent: true,
  }),
  /must not publish an independent capability marker/,
);
assert.throws(
  () => runtimeLockValidators.validateAuthoringSpoolIntegrationState({
    ...integratedSpoolState,
    fittingWorkerReadsActivationSpool: false,
  }),
  /requires the production activation-spool path/,
);
assert.doesNotThrow(() => runtimeLockValidators.validateAuthoringIntegrationState({
  implementationIntegrated: true,
  capabilityIntegrated: false,
  evidenceStatus: "feasibility-required",
}));
assert.doesNotThrow(() => runtimeLockValidators.validateAuthoringIntegrationState({
  implementationIntegrated: true,
  capabilityIntegrated: false,
  evidenceStatus: "integrated-unverified",
}));
assert.doesNotThrow(() => runtimeLockValidators.validateAuthoringIntegrationState({
  implementationIntegrated: true,
  capabilityIntegrated: false,
  evidenceStatus: "verified",
}));
assert.doesNotThrow(() => runtimeLockValidators.validateAuthoringIntegrationState({
  implementationIntegrated: true,
  capabilityIntegrated: true,
  evidenceStatus: "verified",
  release: true,
}));
assert.doesNotThrow(() => runtimeLockValidators.validateAuthoringIntegrationState({
  implementationIntegrated: true,
  capabilityIntegrated: true,
  evidenceStatus: "integrated-unverified",
}));
assert.doesNotThrow(
  () => runtimeLockValidators.validateAuthoringIntegrationState({
    implementationIntegrated: true,
    capabilityIntegrated: true,
    evidenceStatus: "integrated-unverified",
    release: true,
  }),
);
assert.throws(
  () => runtimeLockValidators.validateAuthoringIntegrationState({
    implementationIntegrated: false,
    capabilityIntegrated: true,
    evidenceStatus: "feasibility-required",
  }),
  /requires an integrated implementation/,
);
assert.throws(
  () => runtimeLockValidators.validateAuthoringIntegrationState({
    implementationIntegrated: true,
    capabilityIntegrated: false,
    evidenceStatus: "verified",
    release: true,
  }),
  /not release-enabled/,
);
const integratedAuthoringMarker = [
  "export const BROWSER_AUTHORING_IMPLEMENTATION_INTEGRATED = true;",
  "export const BROWSER_AUTHORING_RUNTIME_INTEGRATED = true;",
  "",
].join("\n");
assert.doesNotThrow(() => runtimeLockValidators.validateAuthoringCapabilityPromotion(
  integratedAuthoringMarker,
  integratedAuthoringMarker,
));
assert.throws(
  () => runtimeLockValidators.validateAuthoringCapabilityPromotion(
    integratedAuthoringMarker,
    `${integratedAuthoringMarker}export const UNTESTED_CHANGE = true;\n`,
  ),
  /must not change the integrated runtime source/,
);
const wiredAuthoringCapabilities = `
  const fittingReady = browserAuthoringAvailableForChannel({
    runtimeIntegrated: BROWSER_AUTHORING_RUNTIME_INTEGRATED,
    modelBackendIntegrated: BROWSER_MODEL_BACKEND_INTEGRATED,
  });
  return { fitting: fittingReady };
`;
assert.doesNotThrow(() => runtimeLockValidators.validateAuthoringCapabilityWiring(
  wiredAuthoringCapabilities,
  true,
));
assert.throws(
  () => runtimeLockValidators.validateAuthoringCapabilityWiring(
    wiredAuthoringCapabilities.replace(
      "runtimeIntegrated: BROWSER_AUTHORING_RUNTIME_INTEGRATED,",
      "runtimeIntegrated: true,",
    ),
    true,
  ),
  /not wired to capabilities/,
);
const localWebLlmManifest = {
  dependencies: {
    "@drowse/web-llm":
      "file:../browser-runtime/vendor/drowse-web-llm-0.2.84-drowse.3.tgz",
  },
};
const localWebLlmLock = {
  packages: {
    "node_modules/@drowse/web-llm": {
      resolved: "file:../browser-runtime/vendor/drowse-web-llm-0.2.84-drowse.3.tgz",
    },
  },
};
const localWebLlmPackage = {
  path: "browser-runtime/vendor/drowse-web-llm-0.2.84-drowse.3.tgz",
  sha256: "1".repeat(64),
};
runtimeLockValidators.validateWebLlmDependency(
  localWebLlmManifest,
  localWebLlmLock,
  { commit: null },
  localWebLlmPackage,
);
runtimeLockValidators.validateInstalledWebLlmDependency(
  localWebLlmManifest,
  {
    packages: {
      "node_modules/@drowse/web-llm": { version: "0.2.84-drowse.3" },
    },
  },
  { name: "@drowse/web-llm", version: "0.2.84-drowse.3" },
);
assert.throws(
  () => runtimeLockValidators.validateInstalledWebLlmDependency(
    localWebLlmManifest,
    {
      packages: {
        "node_modules/@drowse/web-llm": { version: "0.2.84-drowse.3" },
      },
    },
    { name: "@drowse/web-llm", version: "0.2.84-drowse.2" },
  ),
  /does not match the exact package lock/,
);
assert.throws(
  () => runtimeLockValidators.validateWebLlmDependency(
    localWebLlmManifest,
    localWebLlmLock,
    { commit: null },
    localWebLlmPackage,
    true,
  ),
  /pinned WebLLM fork commit/,
);
runtimeLockValidators.validateWebLlmDependency(
  localWebLlmManifest,
  localWebLlmLock,
  { commit: "a".repeat(40) },
  { ...localWebLlmPackage, sourceCommit: "a".repeat(40) },
  true,
);
assert.throws(
  () => runtimeLockValidators.validateWebLlmDependency(
    localWebLlmManifest,
    localWebLlmLock,
    { commit: "a".repeat(40) },
    localWebLlmPackage,
    true,
  ),
  /archive bound to its fork commit/,
);
assert.throws(
  () => runtimeLockValidators.validateWebLlmDependency(
    localWebLlmManifest,
    {
      packages: {
        "node_modules/@drowse/web-llm": { resolved: "file:../browser-runtime/vendor/other.tgz" },
      },
    },
    { commit: null },
    localWebLlmPackage,
  ),
  /runtime-locked Drowse WebLLM archive/,
);
const longPrefill = {
  fixturePath: fixtureRelativePath,
  fixtureSha256,
  modelId: "gemma3-270m-instruct",
  webLlmBaselineVersion: "0.2.84",
  prompt2048Passed: true,
  prompt4096Passed: true,
  shapeCacheHangObserved: false,
};
await runtimeLockValidators.validateRuntimeGateResults(
  "gemmaLongPrefill",
  longPrefill,
  fixtureRepository,
);
await assert.rejects(
  runtimeLockValidators.validateRuntimeGateResults(
    "gemmaLongPrefill",
    { ...longPrefill, webLlmBaselineVersion: "0.2.85" },
    fixtureRepository,
  ),
  /long-prefill evidence did not rule out/,
);
const qwenTinyParity = {
  fixturePath: fixtureRelativePath,
  fixtureSha256,
  architecture: "qwen3",
  maxAbsoluteLogitError: 0.0001,
  maxRelativeLogitError: 0.001,
  maxAbsoluteCaptureError: 0.0001,
  maxRelativeCaptureError: 0.001,
  probePassed: true,
  steeringPassed: true,
};
await runtimeLockValidators.validateRuntimeGateResults(
  "qwenTinyFp32Parity",
  qwenTinyParity,
  fixtureRepository,
);
await assert.rejects(
  runtimeLockValidators.validateRuntimeGateResults(
    "qwenTinyFp32Parity",
    { ...qwenTinyParity, architecture: "gemma3_text" },
    fixtureRepository,
  ),
  /fixture binding is invalid/,
);
const qwenProductionParity = {
  fixturePath: fixtureRelativePath,
  fixtureSha256,
  modelId: "qwen3-1.7b",
  runtimeIdentitySha256: "a".repeat(64),
  hookDisabledGreedyMatchesBaseline: true,
  maxProbeRelativeError: 0.01,
  maxCaptureRelativeError: 0.01,
  steeringDirectionCosine: 0.99,
};
await runtimeLockValidators.validateRuntimeGateResults(
  "qwenProductionQ4Parity",
  qwenProductionParity,
  fixtureRepository,
);
await assert.rejects(
  runtimeLockValidators.validateRuntimeGateResults(
    "qwenProductionQ4Parity",
    { ...qwenProductionParity, modelId: "qwen3-0.6b" },
    fixtureRepository,
  ),
  /fixture binding is invalid/,
);
const activationCapture = {
  fixturePath: fixtureRelativePath,
  fixtureSha256,
  command: "node checked-in-capture-golden.mjs",
  postBlockBoundaryPassed: true,
  selectedPromptPositionsPassed: true,
  finalDecodePositionPassed: true,
  workerIsolationPassed: true,
  maxCaptureRelativeError: 0.01,
};
await runtimeLockValidators.validateAuthoringResults(
  "activationCapture",
  activationCapture,
  fixtureRepository,
);
await assert.rejects(
  runtimeLockValidators.validateAuthoringResults(
    "activationCapture",
    { ...activationCapture, maxCaptureRelativeError: 0.010001 },
    fixtureRepository,
  ),
  /capture relative error exceeds 0.01/,
);
await assert.rejects(
  runtimeLockValidators.validateAuthoringResults(
    "activationCapture",
    { ...activationCapture, assertions: { opaque: true } },
    fixtureRepository,
  ),
  /unknown or missing fields/,
);
const releaseRuntime = {
  runtimeAbi: "drowse-web-runtime-v1",
  hookAbi: "post-block-residual-v4",
  models: ["smollm2-360m-instruct", "gemma3-270m-instruct", "qwen3-1.7b"].map(
    (id, index) => ({
      id,
      sourceRepository: `fixture/source-${index}`,
      sourceRevision: String(index + 1).repeat(40),
      manifestSha256: String(index + 1).repeat(64),
      quantization: id === "gemma3-270m-instruct" ? "q4f32_1" : "q4f16_1",
      tokenizerSha256: String(index + 2).repeat(64),
      chatTemplateSha256: String(index + 3).repeat(64),
      librarySha256: String(index + 4).repeat(64),
      hiddenSize: 64,
      layerMap: [0, 1],
      contextProfiles: [2048, 4096],
    }),
  ),
};
const requiredMatrix = {
  modelIds: releaseRuntime.models.map((model) => model.id),
  contextTokens: [2048, 4096],
  platforms: ["macos", "windows", "linux", "android"],
  browsers: ["chrome", "edge"],
};
const measuredAt = new Date().toISOString();
const matrixRuns = requiredMatrix.modelIds.flatMap((modelId) => {
  const model = releaseRuntime.models.find((candidate) => candidate.id === modelId);
  const runtimeIdentitySha256 = runtimeLockValidators.lockedRuntimeIdentitySha256(
    releaseRuntime,
    model,
  );
  return requiredMatrix.contextTokens.flatMap((contextTokens) =>
    requiredMatrix.platforms.flatMap((platform) =>
      requiredMatrix.browsers.map((browser) => ({
        producerId: "physical-browser-benchmark-v1",
        producerConfigPath:
          "browser-runtime/evidence-configs/physical-browser-benchmark.json",
        producerConfigSha256: "a".repeat(64),
        producerToolVersion:
          `physical-browser-benchmark-v1+sha256.${"b".repeat(64)}`,
        modelId,
        contextTokens,
        contextBindingSha256: runtimeLockValidators.digestCanonical({
          contextTokens,
          runtimeIdentitySha256,
        }),
        platform,
        browser,
        browserVersion: "130.0.0.0",
        browserFlags: "normal",
        deviceLabel: `${modelId}-${contextTokens}-${platform}-${browser}`,
        drowseRevision: currentRevision,
        runtimeIdentitySha256,
        webLlmSha256: "1".repeat(64),
        modelArtifactManifestSha256: "2".repeat(64),
        convertedManifestSha256: model.manifestSha256,
        modelLibrarySha256: model.librarySha256,
        coreArtifactManifestSha256: "3".repeat(64),
        jlensArtifactManifestSha256: "4".repeat(64),
        saeArtifactManifestSha256: "5".repeat(64),
        adapterIsFallback: false,
        calibrationScore: 1,
        deviceMemoryGiB: 8,
        adapterFeatures: [],
        adapterLimits: {
          maxBufferSize: 1,
          maxStorageBuffersPerShaderStage: 10,
        },
        prefillTokensPerSecond: 1,
        decodeTokensPerSecond: 1,
        loadSucceeded: true,
        measuredAt,
      })),
    ),
  );
});
const verifiedBenchmarks = {
  $schema: "./benchmark-evidence.schema.json",
  schemaVersion: 4,
  status: "verified",
  drowseRevision: currentRevision,
  requiredMatrix,
  supportPolicy: {
    minimumChromeMajor: 130,
    minimumEdgeMajor: 130,
    maximumEvidenceAgeDays: 180,
  },
  runs: matrixRuns,
};
runtimeLockValidators.validateBenchmarkEvidence(
  verifiedBenchmarks,
  releaseRuntime,
  currentRevision,
);
runtimeLockValidators.validateBenchmarkEvidence(
  {
    ...verifiedBenchmarks,
    runs: [{ ...matrixRuns[0], deviceMemoryGiB: null }, ...matrixRuns.slice(1)],
  },
  releaseRuntime,
  currentRevision,
);
assert.throws(
  () => runtimeLockValidators.validateBenchmarkEvidence(
    { ...verifiedBenchmarks, drowseRevision: "not-a-revision" },
    releaseRuntime,
    "not-a-revision",
  ),
  /not bound to an exact tested Drowse revision/,
);
assert.equal(
  runtimeLockValidators.validateEvidenceRevisionAgreement([
    currentRevision,
    currentRevision,
    null,
  ]),
  currentRevision,
);
assert.throws(
  () => runtimeLockValidators.validateEvidenceRevisionAgreement([
    currentRevision,
    "0".repeat(40),
  ]),
  /do not agree on the tested Drowse revision/,
);
assert.throws(
  () => runtimeLockValidators.validateBenchmarkEvidence(
    { ...verifiedBenchmarks, runs: matrixRuns.slice(0, -1) },
    releaseRuntime,
    currentRevision,
  ),
  /has no confirmed hardware benchmark evidence/,
);
assert.throws(
  () => runtimeLockValidators.validateBenchmarkEvidence(
    {
      ...verifiedBenchmarks,
      runs: [{ ...matrixRuns[0], adapterIsFallback: true }, ...matrixRuns.slice(1)],
    },
    releaseRuntime,
    currentRevision,
  ),
  /does not confirm a non-fallback hardware adapter/,
);
await writeFile(fixturePath, Buffer.from("changed\n"));
await assert.rejects(
  runtimeLockValidators.validateAuthoringResults(
    "activationCapture",
    activationCapture,
    fixtureRepository,
  ),
  /fixture differs from its checked-in content/,
);

const provenanceRepository = join(dir, "provenance-repository");
const provenanceRuntimePath = join(
  provenanceRepository,
  "browser-runtime",
  "runtime-lock.json",
);
const provenanceAuthoringPath = join(
  provenanceRepository,
  "webui",
  "src",
  "hosted",
  "runtime",
  "browserAuthoring.ts",
);
const provenanceRuntime = {
  $schema: "./runtime-lock.schema.json",
  schemaVersion: 1,
  status: "feasibility-required",
  runtimeAbi: "drowse-web-runtime-v1",
  hookAbi: "post-block-residual-v4",
  toolchain: {
    mlcLlmFork: { repository: "fixture/mlc", commit: "1".repeat(40) },
  },
  models: [{ id: "fixture", sourceRevision: "2".repeat(40) }],
};
await mkdir(join(provenanceRepository, "browser-runtime", "evidence"), { recursive: true });
await mkdir(join(provenanceRepository, "webui", "public-hosted"), { recursive: true });
await mkdir(join(provenanceRepository, "webui", "src", "hosted", "runtime"), {
  recursive: true,
});
await writeFile(join(provenanceRepository, "seed.txt"), "seed\n");
await exec("git", ["init", "-q", "-b", "main"], { cwd: provenanceRepository });
await exec("git", ["add", "seed.txt"], { cwd: provenanceRepository });
await exec("git", [
  "-c", "user.name=Drowse Test", "-c", "user.email=drowse@example.invalid",
  "commit", "-qm", "seed",
], { cwd: provenanceRepository });
const provenanceBase = (await exec("git", ["rev-parse", "HEAD"], {
  cwd: provenanceRepository,
})).stdout.trim();
await writeFile(provenanceRuntimePath, `${JSON.stringify(provenanceRuntime)}\n`);
await writeFile(provenanceAuthoringPath, integratedAuthoringMarker);
await exec("git", ["add",
  "browser-runtime/runtime-lock.json",
  "webui/src/hosted/runtime/browserAuthoring.ts",
], {
  cwd: provenanceRepository,
});
await exec("git", [
  "-c", "user.name=Drowse Test", "-c", "user.email=drowse@example.invalid",
  "commit", "-qm", "tested source",
], { cwd: provenanceRepository });
const testedRevision = (await exec("git", ["rev-parse", "HEAD"], {
  cwd: provenanceRepository,
})).stdout.trim();
const promotedRuntime = { ...provenanceRuntime, status: "verified" };
assert.equal(
  runtimeLockValidators.runtimeLockIdentitySha256(provenanceRuntime),
  runtimeLockValidators.runtimeLockIdentitySha256(promotedRuntime),
);
await Promise.all([
  writeFile(provenanceRuntimePath, `${JSON.stringify(promotedRuntime)}\n`),
  writeFile(join(provenanceRepository, "browser-runtime", "benchmark-evidence.json"), "{}\n"),
  writeFile(join(provenanceRepository, "browser-runtime", "distribution-lock.json"), "{}\n"),
  writeFile(join(provenanceRepository, "browser-runtime", "evidence", "result.json"), "{}\n"),
  writeFile(join(provenanceRepository, "webui", "public-hosted", "_headers"), "/*\n"),
  writeFile(provenanceAuthoringPath, integratedAuthoringMarker),
]);
await exec("git", ["add",
  "browser-runtime/runtime-lock.json",
  "browser-runtime/benchmark-evidence.json",
  "browser-runtime/distribution-lock.json",
  "browser-runtime/evidence/result.json",
  "webui/public-hosted/_headers",
  "webui/src/hosted/runtime/browserAuthoring.ts",
], { cwd: provenanceRepository });
await exec("git", [
  "-c", "user.name=Drowse Test", "-c", "user.email=drowse@example.invalid",
  "commit", "-qm", "evidence promotion",
], { cwd: provenanceRepository });
const releaseRevision = (await exec("git", ["rev-parse", "HEAD"], {
  cwd: provenanceRepository,
})).stdout.trim();
runtimeLockValidators.validateReleaseEvidenceProvenance(
  testedRevision,
  releaseRevision,
  runtimeLockValidators.runtimeLockIdentitySha256(promotedRuntime),
  provenanceRepository,
);
await exec("git", ["switch", "-q", "-c", "sibling", provenanceBase], {
  cwd: provenanceRepository,
});
await mkdir(join(provenanceRepository, "browser-runtime"), { recursive: true });
await writeFile(provenanceRuntimePath, `${JSON.stringify(provenanceRuntime)}\n`);
await exec("git", ["add", "browser-runtime/runtime-lock.json"], {
  cwd: provenanceRepository,
});
await exec("git", [
  "-c", "user.name=Drowse Test", "-c", "user.email=drowse@example.invalid",
  "commit", "-qm", "sibling test source",
], { cwd: provenanceRepository });
const siblingRevision = (await exec("git", ["rev-parse", "HEAD"], {
  cwd: provenanceRepository,
})).stdout.trim();
await exec("git", ["switch", "-q", "main"], { cwd: provenanceRepository });
assert.throws(
  () => runtimeLockValidators.validateReleaseEvidenceProvenance(
    siblingRevision,
    releaseRevision,
    runtimeLockValidators.runtimeLockIdentitySha256(promotedRuntime),
    provenanceRepository,
  ),
  /not an ancestor of the release HEAD/,
);
await writeFile(
  join(provenanceRepository, "webui", "src", "hosted", "runtime", "browser.worker.ts"),
  "export const drift = true;\n",
);
await exec("git", ["add", "webui/src/hosted/runtime/browser.worker.ts"], {
  cwd: provenanceRepository,
});
await exec("git", [
  "-c", "user.name=Drowse Test", "-c", "user.email=drowse@example.invalid",
  "commit", "-qm", "runtime source drift",
], { cwd: provenanceRepository });
const driftedReleaseRevision = (await exec("git", ["rev-parse", "HEAD"], {
  cwd: provenanceRepository,
})).stdout.trim();
assert.throws(
  () => runtimeLockValidators.validateReleaseEvidenceProvenance(
    testedRevision,
    driftedReleaseRevision,
    runtimeLockValidators.runtimeLockIdentitySha256(promotedRuntime),
    provenanceRepository,
  ),
  /release executable source differs from the tested revision.*browser\.worker\.ts/,
);
await exec("git", ["switch", "-q", "-c", "identity-drift", releaseRevision], {
  cwd: provenanceRepository,
});
const identityDriftRuntime = {
  ...promotedRuntime,
  toolchain: {
    ...promotedRuntime.toolchain,
    mlcLlmFork: {
      ...promotedRuntime.toolchain.mlcLlmFork,
      commit: "3".repeat(40),
    },
  },
};
await writeFile(provenanceRuntimePath, `${JSON.stringify(identityDriftRuntime)}\n`);
await exec("git", ["add", "browser-runtime/runtime-lock.json"], {
  cwd: provenanceRepository,
});
await exec("git", [
  "-c", "user.name=Drowse Test", "-c", "user.email=drowse@example.invalid",
  "commit", "-qm", "runtime identity drift",
], { cwd: provenanceRepository });
const identityDriftRevision = (await exec("git", ["rev-parse", "HEAD"], {
  cwd: provenanceRepository,
})).stdout.trim();
assert.throws(
  () => runtimeLockValidators.validateReleaseEvidenceProvenance(
    testedRevision,
    identityDriftRevision,
    runtimeLockValidators.runtimeLockIdentitySha256(identityDriftRuntime),
    provenanceRepository,
  ),
  /canonical browser runtime identity drifted after testing/,
);

const artifacts = join(dir, "artifacts");
await mkdir(join(artifacts, "nested"), { recursive: true });
await Promise.all([
  writeFile(join(artifacts, "b.bin"), Buffer.from([2, 3])),
  writeFile(join(artifacts, "nested", "a.bin"), Buffer.from([1])),
]);
const manifested = await exec(process.execPath, [
  manifestScript.pathname,
  artifacts,
  "a9lim/drowse-web-fixture",
  "0123456789abcdef0123456789abcdef01234567",
]);
const manifest = JSON.parse(manifested.stdout);
assert.deepEqual(manifest.files.map((file) => file.path), ["b.bin", "nested/a.bin"]);
assert.deepEqual(manifest.files.map((file) => file.bytes), [2, 1]);
assert.match(manifest.files[0].sha256, /^[0-9a-f]{64}$/);

console.log("Hosted catalog tools passed");
