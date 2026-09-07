#!/usr/bin/env node

import { createHash, randomBytes, randomUUID } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createReadStream } from "node:fs";
import {
  link,
  lstat,
  open,
  readFile,
  realpath,
  unlink,
} from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import {
  digestCanonical,
  lockedRuntimeIdentitySha256,
  runtimeLockIdentitySha256,
  validateAuthoringResults,
  validateBenchmarkEvidence,
  validateRuntimeGateResults,
} from "./check-runtime-lock.mjs";
import {
  assertExecutedProducerReceipt,
  releaseEvidenceProducerBinding,
  requireReleaseEvidenceProducer,
  runReleaseEvidenceProducer,
} from "./release-evidence-producers.mjs";

const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));
const browserRuntimeRoot = resolve(repositoryRoot, "browser-runtime");
const defaultEvidenceRoot = resolve(browserRuntimeRoot, "evidence");
const runtimeLockPath = resolve(browserRuntimeRoot, "runtime-lock.json");
const runtimeLockCheckerPath = fileURLToPath(new URL("./check-runtime-lock.mjs", import.meta.url));
const COMMIT = /^[0-9a-f]{40}$/;
const RUNTIME_EVIDENCE_TYPES = new Set([
  "gemmaTinyFp32Parity",
  "llamaTinyFp32Parity",
  "qwenTinyFp32Parity",
  "gemmaProductionQ4Parity",
  "smolProductionQ4Parity",
  "qwenProductionQ4Parity",
  "gemmaLongPrefill",
  "runtimeLifecycle",
]);
const AUTHORING_EVIDENCE_TYPES = new Set([
  "activationCapture",
  "fittingKernelParity",
  "topologyOrchestration",
  "manifoldSerialization",
  "instrumentIntegration",
  "sharedGoldenResults",
  "physicalBrowserFitting",
]);
const BENCHMARK_RUN_KEYS = [
  "modelId",
  "contextTokens",
  "platform",
  "browser",
  "browserVersion",
  "browserFlags",
  "deviceLabel",
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
];

export function parseArguments(args) {
  const kind = args[0];
  if (!['gate', 'benchmark'].includes(kind)) {
    throw new Error(
      "usage: node record-hosted-evidence.mjs gate|benchmark " +
        "--evidence-type TYPE --producer-config CONFIG.json " +
        "--output browser-runtime/evidence/NAME.json",
    );
  }
  const result = {
    kind,
    evidenceType: null,
    producerConfig: null,
    output: null,
  };
  for (let index = 1; index < args.length; index += 1) {
    const argument = args[index];
    if (!['--evidence-type', '--producer-config', '--output'].includes(argument)) {
      throw new Error(`unexpected argument ${argument}`);
    }
    const value = args[++index];
    if (!value || value.startsWith("--")) {
      throw new Error(`${argument} requires a value`);
    }
    const key = argument
      .slice(2)
      .replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    if (result[key] !== null) throw new Error(`${argument} may be supplied only once`);
    result[key] = key === "evidenceType" ? value : resolve(value);
  }
  if (
    result.evidenceType === null ||
    result.producerConfig === null ||
    result.output === null
  ) {
    throw new Error(
      "--evidence-type, --producer-config, and --output are required",
    );
  }
  const producer = requireReleaseEvidenceProducer(result.evidenceType);
  if (producer.kind !== kind) {
    throw new Error(`${result.evidenceType} is a ${producer.kind} producer`);
  }
  return result;
}

export async function createGateEvidenceRecord({
  receipt,
  runtimeLock,
  fixtureRoot,
  producerRoot = fixtureRoot,
  drowseRevision,
  now = Date.now(),
}) {
  assertExecutedProducerReceipt(receipt);
  const evidenceType = receipt.evidenceType;
  const runtime = RUNTIME_EVIDENCE_TYPES.has(evidenceType);
  const authoring = AUTHORING_EVIDENCE_TYPES.has(evidenceType);
  if (!runtime && !authoring) throw new Error("gate measurement has an unknown evidenceType");
  const producer = requireReleaseEvidenceProducer(evidenceType);
  if (producer.kind !== "gate") throw new Error(`${evidenceType} is not a gate producer`);
  const producerBinding = await validateReceiptProducerBinding(
    receipt,
    producer,
    producerRoot,
  );
  if (receipt.passed !== true) {
    throw new Error("gate measurement must explicitly report passed: true");
  }
  validateRecentDate(receipt.producedAt, now, "gate measurement producedAt");
  if (!COMMIT.test(drowseRevision ?? "")) {
    throw new Error("gate evidence requires an exact tested Drowse revision");
  }
  if (!runtimeLock || typeof runtimeLock !== "object" || Array.isArray(runtimeLock)) {
    throw new Error("runtime lock is unavailable");
  }
  if (
    typeof runtimeLock.runtimeAbi !== "string" ||
    runtimeLock.hookAbi !== "post-block-residual-v4"
  ) {
    throw new Error("runtime lock ABI is invalid");
  }
  if (
    !receipt.results ||
    typeof receipt.results !== "object" ||
    Array.isArray(receipt.results)
  ) {
    throw new Error("gate measurement results must be an object");
  }
  const reserved = new Set(["fixturePath", "fixtureSha256", "command"]);
  if (Object.keys(receipt.results).some((key) => reserved.has(key))) {
    throw new Error("gate measurement results contain recorder-owned fields");
  }
  const fixturePath = validateRepositoryPath(receipt.fixturePath, "fixturePath");
  const fixtureFile = resolve(fixtureRoot, ...fixturePath.split("/"));
  const fixtureSha256 = await sha256File(fixtureFile);
  const results = {
    fixturePath,
    fixtureSha256,
    ...(authoring
      ? {
          command:
            `node ${producer.executable} --evidence-type ${evidenceType}`,
        }
      : {}),
    ...receipt.results,
  };
  if (runtime) {
    await validateRuntimeGateResults(evidenceType, results, fixtureRoot, runtimeLock);
  } else {
    await validateAuthoringResults(evidenceType, results, fixtureRoot, runtimeLock);
  }
  return {
    $schema: "../release-evidence-record.schema.json",
    schemaVersion: 4,
    evidenceType,
    producerId: producer.id,
    producerConfigPath: producerBinding.configPath,
    producerConfigSha256: producerBinding.configSha256,
    runtimeAbi: runtimeLock.runtimeAbi,
    hookAbi: runtimeLock.hookAbi,
    runtimeLockIdentitySha256: runtimeLockIdentitySha256(runtimeLock),
    drowseRevision,
    producedAt: receipt.producedAt,
    toolVersion: producerBinding.toolVersion,
    passed: true,
    results,
  };
}

export async function createBenchmarkRunRecord({
  receipt,
  runtimeLock,
  drowseRevision,
  producerRoot = repositoryRoot,
  now = Date.now(),
}) {
  assertExecutedProducerReceipt(receipt);
  if (receipt.evidenceType !== "physicalBrowserBenchmark") {
    throw new Error("benchmark receipt has the wrong evidenceType");
  }
  if (receipt.passed !== true) {
    throw new Error("benchmark measurement must explicitly report passed: true");
  }
  const producer = requireReleaseEvidenceProducer(receipt.evidenceType);
  const producerBinding = await validateReceiptProducerBinding(
    receipt,
    producer,
    producerRoot,
  );
  exactKeys(receipt.run, BENCHMARK_RUN_KEYS, "benchmark measurement run");
  if (!COMMIT.test(drowseRevision ?? "")) {
    throw new Error("benchmark evidence requires an exact tested Drowse revision");
  }
  const model = runtimeLock.models?.find((candidate) => candidate.id === receipt.run.modelId);
  if (!model) throw new Error(`benchmark names unknown model ${String(receipt.run.modelId)}`);
  const runtimeIdentitySha256 = lockedRuntimeIdentitySha256(runtimeLock, model);
  if (runtimeIdentitySha256 === null) {
    throw new Error(`${model.id} has an incomplete runtime identity`);
  }
  if (
    receipt.run.browserFlags !== "normal" ||
    receipt.run.runtimeIdentitySha256 !== runtimeIdentitySha256 ||
    receipt.run.convertedManifestSha256 !== model.manifestSha256 ||
    receipt.run.modelLibrarySha256 !== model.librarySha256
  ) {
    throw new Error("benchmark did not observe the locked release runtime artifacts");
  }
  for (const field of [
    "webLlmSha256",
    "modelArtifactManifestSha256",
    "coreArtifactManifestSha256",
    "jlensArtifactManifestSha256",
    "saeArtifactManifestSha256",
  ]) {
    if (!/^[0-9a-f]{64}$/.test(receipt.run[field] ?? "")) {
      throw new Error(`benchmark ${field} is invalid`);
    }
  }
  validateRecentDate(receipt.run.measuredAt, now, "benchmark measuredAt");
  const run = {
    producerId: producer.id,
    producerConfigPath: producerBinding.configPath,
    producerConfigSha256: producerBinding.configSha256,
    producerToolVersion: producerBinding.toolVersion,
    modelId: receipt.run.modelId,
    contextTokens: receipt.run.contextTokens,
    contextBindingSha256: digestCanonical({
      contextTokens: receipt.run.contextTokens,
      runtimeIdentitySha256,
    }),
    platform: receipt.run.platform,
    browser: receipt.run.browser,
    browserVersion: receipt.run.browserVersion,
    browserFlags: receipt.run.browserFlags,
    deviceLabel: receipt.run.deviceLabel,
    drowseRevision,
    runtimeIdentitySha256: receipt.run.runtimeIdentitySha256,
    webLlmSha256: receipt.run.webLlmSha256,
    modelArtifactManifestSha256: receipt.run.modelArtifactManifestSha256,
    convertedManifestSha256: receipt.run.convertedManifestSha256,
    modelLibrarySha256: receipt.run.modelLibrarySha256,
    coreArtifactManifestSha256: receipt.run.coreArtifactManifestSha256,
    jlensArtifactManifestSha256: receipt.run.jlensArtifactManifestSha256,
    saeArtifactManifestSha256: receipt.run.saeArtifactManifestSha256,
    adapterIsFallback: receipt.run.adapterIsFallback,
    calibrationScore: receipt.run.calibrationScore,
    deviceMemoryGiB: receipt.run.deviceMemoryGiB,
    adapterFeatures: receipt.run.adapterFeatures,
    adapterLimits: receipt.run.adapterLimits,
    prefillTokensPerSecond: receipt.run.prefillTokensPerSecond,
    decodeTokensPerSecond: receipt.run.decodeTokensPerSecond,
    loadSucceeded: receipt.run.loadSucceeded,
    measuredAt: receipt.run.measuredAt,
  };
  validateBenchmarkEvidence({
    $schema: "./benchmark-evidence.schema.json",
    schemaVersion: 4,
    status: "feasibility-required",
    drowseRevision,
    requiredMatrix: {
      modelIds: runtimeLock.models.map((entry) => entry.id),
      contextTokens: [2048, 4096],
      platforms: ["macos", "windows", "linux", "android"],
      browsers: ["chrome", "edge"],
    },
    supportPolicy: {
      minimumChromeMajor: null,
      minimumEdgeMajor: null,
      maximumEvidenceAgeDays: 180,
    },
    runs: [run],
  }, runtimeLock);
  return run;
}

async function validateReceiptProducerBinding(receipt, producer, producerRoot) {
  if (
    typeof receipt.producerConfigPath !== "string" ||
    !/^browser-runtime\/evidence-configs\/[a-z0-9][a-z0-9._-]*\.json$/.test(
      receipt.producerConfigPath,
    )
  ) {
    throw new Error("release evidence producer config path is invalid");
  }
  const binding = await releaseEvidenceProducerBinding(
    receipt.evidenceType,
    producerRoot,
    receipt.producerConfigPath,
  );
  if (
    receipt.producerId !== producer.id ||
    receipt.producerConfigPath !== binding.configPath ||
    receipt.producerConfigSha256 !== binding.configSha256 ||
    receipt.producerToolVersion !== binding.toolVersion
  ) {
    throw new Error("release evidence producer config binding is invalid");
  }
  return binding;
}

export function resolveEvidenceOutput(value, evidenceRoot = defaultEvidenceRoot) {
  const output = resolve(value);
  const rel = relative(evidenceRoot, output);
  if (
    !rel || rel === ".." || rel.startsWith(`..${sep}`) || isAbsolute(rel) ||
    rel.includes(sep) || !rel.endsWith(".json")
  ) {
    throw new Error(
      "evidence output must be a new JSON file directly under browser-runtime/evidence",
    );
  }
  const parts = rel.split(sep);
  if (parts.some((part) => !/^[a-z0-9][a-z0-9._-]*$/.test(part))) {
    throw new Error("evidence output contains an unsafe path component");
  }
  return output;
}

export async function writeNewJson(output, value, evidenceRoot = defaultEvidenceRoot) {
  const target = resolveEvidenceOutput(output, evidenceRoot);
  const rootInfo = await lstat(evidenceRoot);
  if (!rootInfo.isDirectory() || rootInfo.isSymbolicLink()) {
    throw new Error("browser-runtime/evidence must be a regular directory");
  }
  const [rootRealPath, parentRealPath] = await Promise.all([
    realpath(evidenceRoot),
    realpath(dirname(target)),
  ]);
  const parentRelative = relative(rootRealPath, parentRealPath);
  if (
    parentRelative === ".." || parentRelative.startsWith(`..${sep}`) ||
    isAbsolute(parentRelative)
  ) {
    throw new Error("evidence output parent resolves outside browser-runtime/evidence");
  }
  if (await lstat(target).then(() => true, () => false)) {
    throw new Error("evidence output already exists");
  }
  const bytes = Buffer.from(`${JSON.stringify(sortJson(value), null, 2)}\n`);
  const temporary = `${target}.${process.pid}.${randomUUID()}.tmp`;
  let handle;
  try {
    handle = await open(temporary, "wx", 0o600);
    await handle.writeFile(bytes);
    await handle.sync();
    await handle.close();
    handle = undefined;
    await link(temporary, target);
  } finally {
    await handle?.close().catch(() => undefined);
    await unlink(temporary).catch(() => undefined);
  }
  return {
    bytes: bytes.byteLength,
    sha256: createHash("sha256").update(bytes).digest("hex"),
  };
}

export function unpermittedWorktreeEntries(porcelain) {
  return porcelain.split(/\r?\n/).filter(Boolean).filter((line) =>
    !/^\?\? browser-runtime\/evidence\/[a-z0-9][a-z0-9._/-]*\.json$/.test(line)
  );
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const output = resolveEvidenceOutput(
    isAbsolute(options.output) ? options.output : resolve(repositoryRoot, options.output),
  );
  const status = execFileSync(
    "git",
    ["status", "--porcelain", "--untracked-files=all"],
    { cwd: repositoryRoot, encoding: "utf8" },
  );
  const dirty = unpermittedWorktreeEntries(status);
  if (dirty.length > 0) {
    throw new Error(
      "evidence recording requires committed executable source; unexpected worktree changes: " +
        dirty.slice(0, 5).join(", "),
    );
  }
  execFileSync(process.execPath, [runtimeLockCheckerPath], {
    cwd: repositoryRoot,
    stdio: ["ignore", "pipe", "pipe"],
  });
  const drowseRevision = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repositoryRoot,
    encoding: "utf8",
  }).trim();
  if (!COMMIT.test(drowseRevision)) throw new Error("could not resolve tested Drowse revision");
  const producerConfigPath = relative(
    repositoryRoot,
    resolve(options.producerConfig),
  ).split(sep).join("/");
  if (
    !/^browser-runtime\/evidence-configs\/[a-z0-9][a-z0-9._-]*\.json$/.test(
      producerConfigPath,
    )
  ) {
    throw new Error(
      "producer config must be a JSON file directly under browser-runtime/evidence-configs",
    );
  }
  await assertTrackedRegularFile(options.producerConfig, "producer config");
  const runtimeLock = JSON.parse(await readFile(runtimeLockPath, "utf8"));
  const receipt = await runReleaseEvidenceProducer({
    evidenceType: options.evidenceType,
    configPath: options.producerConfig,
    repositoryRoot,
    nonce: randomBytes(32).toString("hex"),
  });
  const value = options.kind === "gate"
    ? await createGateEvidenceRecord({
        receipt,
        runtimeLock,
        fixtureRoot: repositoryRoot,
        drowseRevision,
      })
    : await createBenchmarkRunRecord({ receipt, runtimeLock, drowseRevision });
  const written = await writeNewJson(output, value);
  const path = relative(browserRuntimeRoot, output).split(sep).join("/");
  process.stdout.write(`${JSON.stringify({
    kind: options.kind,
    path,
    sha256: written.sha256,
    drowseRevision,
    trust: "candidate-unattested",
    attestationRequired: true,
    signerWorkflow: ".github/workflows/ci.yml",
  })}\n`);
}

async function assertTrackedRegularFile(path, label) {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink() || info.size > 4 * 1024 * 1024) {
    throw new Error(`${label} must be a regular file no larger than 4 MiB`);
  }
  const relativePath = relative(repositoryRoot, resolve(path));
  if (
    !relativePath ||
    relativePath === ".." ||
    relativePath.startsWith(`..${sep}`) ||
    isAbsolute(relativePath)
  ) {
    throw new Error(`${label} must be inside the Drowse repository`);
  }
  try {
    execFileSync("git", ["ls-files", "--error-unmatch", relativePath], {
      cwd: repositoryRoot,
      stdio: "ignore",
    });
  } catch (error) {
    throw new Error(`${label} must be checked into Git`, { cause: error });
  }
}

async function sha256File(path) {
  const digest = createHash("sha256");
  try {
    for await (const chunk of createReadStream(path)) digest.update(chunk);
  } catch (error) {
    throw new Error("fixturePath is unavailable", { cause: error });
  }
  return digest.digest("hex");
}

function validateRepositoryPath(value, label) {
  if (
    typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(value) ||
    isAbsolute(value) || value.split("/").some((part) => !part || part === "." || part === "..")
  ) {
    throw new Error(`${label} must be a safe repository-relative path`);
  }
  return value;
}

function validateRecentDate(value, now, label) {
  const timestamp = Date.parse(value);
  if (
    typeof value !== "string" || !Number.isFinite(timestamp) ||
    timestamp > now + 5 * 60 * 1000 ||
    timestamp < now - 180 * 24 * 60 * 60 * 1000
  ) {
    throw new Error(`${label} is stale or invalid`);
  }
}

function exactKeys(value, keys, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new Error(`${label} has unknown or missing fields`);
  }
}

function sortJson(value) {
  if (Array.isArray(value)) return value.map(sortJson);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.keys(value).sort().map((key) => [key, sortJson(value[key])]),
  );
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
