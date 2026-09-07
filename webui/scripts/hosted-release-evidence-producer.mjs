#!/usr/bin/env node

import { execFile } from "node:child_process";
import { lstat, readFile, realpath } from "node:fs/promises";
import { dirname, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import {
  requireReleaseEvidenceProducer,
  validateProducerReceipt,
} from "./release-evidence-producers.mjs";

const execFileAsync = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));
const SHA256 = /^[0-9a-f]{64}$/u;
const MAX_PHYSICAL_REPORT_AGE_MS = 30 * 24 * 60 * 60 * 1000;

export function parseProducerArguments(args) {
  const result = { evidenceType: null, config: null, nonce: null };
  const fields = new Map([
    ["--evidence-type", "evidenceType"],
    ["--config", "config"],
    ["--nonce", "nonce"],
  ]);
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    const field = fields.get(argument);
    if (!field) throw new Error(`unexpected producer argument ${argument}`);
    if (result[field] !== null)
      throw new Error(`${argument} may be supplied only once`);
    const value = args[++index];
    if (!value || value.startsWith("--"))
      throw new Error(`${argument} requires a value`);
    result[field] = value;
  }
  if (Object.values(result).some((value) => value === null)) {
    throw new Error("--evidence-type, --config, and --nonce are required");
  }
  if (!/^[0-9a-f]{64}$/.test(result.nonce)) {
    throw new Error("producer nonce must be 32 lowercase hexadecimal bytes");
  }
  return result;
}

export async function runConfiguredObserver({
  evidenceType,
  configPath,
  nonce,
  root = repositoryRoot,
  execute = execFileAsync,
}) {
  const producer = requireReleaseEvidenceProducer(evidenceType);
  const config = await readConfig(configPath, root);
  exactKeys(
    config,
    ["$schema", "schemaVersion", "evidenceType", "observer", "arguments"],
    "release evidence producer config",
  );
  if (
    config.$schema !== "../evidence-config.schema.json" ||
    config.schemaVersion !== 1 ||
    config.evidenceType !== evidenceType
  ) {
    throw new Error("release evidence producer config identity is invalid");
  }
  if (
    typeof config.observer !== "string" ||
    !producer.closure.includes(config.observer) ||
    config.observer === producer.executable ||
    config.observer === "webui/scripts/release-evidence-producers.mjs"
  ) {
    throw new Error(
      "release evidence observer must be an executable in the registered source closure",
    );
  }
  const argumentSets = normalizeArgumentSets(config.arguments);
  const observerPath = resolve(root, ...config.observer.split("/"));
  await requireRegularContainedFile(observerPath, root, "release evidence observer");
  const command = config.observer.endsWith(".py")
    ? ["python3", [observerPath]]
    : [process.execPath, [observerPath]];
  const reports = [];
  for (const arguments_ of argumentSets) {
    const { stdout, stderr } = await execute(
      command[0],
      [...command[1], ...arguments_],
      {
        cwd: root,
        encoding: "utf8",
        env: observerEnvironment(process.env),
        maxBuffer: 4 * 1024 * 1024,
        timeout: 2 * 60 * 60 * 1000,
        windowsHide: true,
      },
    );
    if (typeof stderr === "string" && stderr.trim()) {
      throw new Error("release evidence observer wrote to stderr");
    }
    try {
      reports.push(JSON.parse(stdout));
    } catch (error) {
      throw new Error("release evidence observer did not return one JSON report", {
        cause: error,
      });
    }
  }
  const normalized = adaptObserverReports({
    evidenceType,
    nonce,
    producer,
    observer: config.observer,
    reports,
  });
  validateProducerReceipt(normalized, producer, evidenceType, nonce);
  return normalized;
}

export function observerEnvironment(environment) {
  const allowed = [
    "PATH",
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "HOME",
    "USERPROFILE",
    "LOCALAPPDATA",
    "APPDATA",
    "XDG_CACHE_HOME",
    "XDG_RUNTIME_DIR",
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "LANG",
    "LC_ALL",
    "TZ",
    "PLAYWRIGHT_BROWSERS_PATH",
  ];
  return Object.fromEntries(
    allowed.flatMap((key) =>
      typeof environment[key] === "string" && environment[key]
        ? [[key, environment[key]]]
        : [],
    ),
  );
}

export function adaptObserverReports({
  evidenceType,
  nonce,
  producer,
  observer,
  reports,
}) {
  if (!Array.isArray(reports) || reports.length === 0) {
    throw new Error(`${evidenceType} observer capability is unavailable: no reports`);
  }
  if (reports.length === 1 && isDirectObservation(reports[0], evidenceType)) {
    return {
      ...reports[0],
      producerId: producer.id,
      nonce,
    };
  }
  if (observer !== "webui/scripts/browser-full-runtime.mjs") {
    throw new Error(
      `${evidenceType} observer capability is unavailable in ${observer}`,
    );
  }
  for (const report of reports) validateFullRuntimeReport(report);
  if (evidenceType === "physicalBrowserBenchmark") {
    if (reports.length !== 1) {
      throw new Error("physicalBrowserBenchmark requires exactly one physical browser report");
    }
    return benchmarkReceipt(reports[0], producer, nonce);
  }
  if (evidenceType === "instrumentIntegration") {
    if (reports.length !== 1) {
      throw new Error("instrumentIntegration requires exactly one full-runtime report");
    }
    return instrumentReceipt(reports[0], producer, nonce);
  }
  if (evidenceType === "physicalBrowserFitting") {
    return physicalFittingReceipt(reports, producer, nonce);
  }
  throw new Error(
    `${evidenceType} observer capability is unavailable in browser-full-runtime-v4`,
  );
}

function benchmarkReceipt(report, producer, nonce) {
  const environment = report.environment;
  const observed = requireReleaseEnvironment(report, "physicalBrowserBenchmark");
  const adapter = report.checks.webgpu_adapter.measurements;
  const generation = report.checks.ordinary_generation.measurements;
  const platform = physicalPlatform(environment);
  const browser = environment.browserChannel.startsWith("msedge")
    ? "edge"
    : "chrome";
  const adapterInfo = environment.adapterInfo ?? {};
  const deviceLabel = [adapterInfo.vendor, adapterInfo.architecture]
    .filter((value) => typeof value === "string" && value.trim())
    .join(" ");
  if (!deviceLabel) {
    throw new Error(
      "physicalBrowserBenchmark observer capability is unavailable: adapter identity",
    );
  }
  for (const [label, value] of [
    ["calibration score", adapter.calibrationScore],
    ["prefill throughput", generation.prefillTokensPerSecond],
    ["decode throughput", generation.decodeTokensPerSecond],
  ]) {
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(`physicalBrowserBenchmark observer has invalid ${label}`);
    }
  }
  const adapterLimits = adapter.adapterLimits;
  const adapterFeatures = adapter.adapterFeatures;
  if (!Array.isArray(adapterFeatures) || !isNumericRecord(adapterLimits)) {
    throw new Error("physicalBrowserBenchmark did not emit adapter features and limits");
  }
  return {
    schemaVersion: 1,
    evidenceType: "physicalBrowserBenchmark",
    producerId: producer.id,
    nonce,
    producedAt: report.producedAt,
    passed: true,
    run: {
      modelId: environment.modelId,
      contextTokens: environment.contextTokens,
      platform,
      browser,
      browserVersion: environment.browserVersion,
      browserFlags: environment.browserFlags,
      deviceLabel,
      runtimeIdentitySha256: observed.runtimeIdentitySha256,
      webLlmSha256: observed.webLlmSha256,
      modelArtifactManifestSha256: observed.modelArtifactManifestSha256,
      convertedManifestSha256: observed.convertedManifestSha256,
      modelLibrarySha256: observed.modelLibrarySha256,
      coreArtifactManifestSha256: observed.coreArtifactManifestSha256,
      jlensArtifactManifestSha256: observed.jlensArtifactManifestSha256,
      saeArtifactManifestSha256: observed.saeArtifactManifestSha256,
      adapterIsFallback: environment.adapterFallback !== "hardware",
      calibrationScore: adapter.calibrationScore,
      deviceMemoryGiB:
        Number.isFinite(environment.deviceMemoryGiB) &&
        environment.deviceMemoryGiB > 0
          ? environment.deviceMemoryGiB
          : null,
      adapterFeatures,
      adapterLimits,
      prefillTokensPerSecond: generation.prefillTokensPerSecond,
      decodeTokensPerSecond: generation.decodeTokensPerSecond,
      loadSucceeded: report.checks.load.passed === true,
      measuredAt: report.producedAt,
    },
  };
}

function instrumentReceipt(report, producer, nonce) {
  const observed = requireReleaseEnvironment(report, "instrumentIntegration");
  const load = report.checks.load.measurements;
  const instruments = report.checks.flat_gated_instruments.measurements;
  if (
    load.jlens !== true ||
    load.sae !== true ||
    load.multiTerm !== true ||
    !Array.isArray(instruments.gatedFamilies) ||
    !["geometry", "lens", "sae"].every((family) =>
      instruments.gatedFamilies.includes(family)
    ) ||
    !Number.isSafeInteger(instruments.exactJlensLayerRows) ||
    instruments.exactJlensLayerRows < 1 ||
    !Number.isSafeInteger(instruments.exactSaeDictionaryFeatureCount) ||
    instruments.exactSaeDictionaryFeatureCount < 1 ||
    instruments.incompatibleBindingRejected !== true ||
    instruments.fp32TensorValidationPassed !== true
  ) {
    throw new Error(
      "instrumentIntegration observer capability is unavailable: exact rejection and fp32 validation assertions are required",
    );
  }
  return {
    schemaVersion: 1,
    evidenceType: "instrumentIntegration",
    producerId: producer.id,
    nonce,
    producedAt: report.producedAt,
    passed: true,
    fixturePath: producer.fixturePath,
    results: {
      coreGeometryPassed: true,
      jlensV6Passed: true,
      saeV1Passed: true,
      exactBindingValidationPassed: true,
      incompatibleBindingRejected: instruments.incompatibleBindingRejected,
      fp32TensorValidationPassed: instruments.fp32TensorValidationPassed,
      ...observed,
    },
  };
}

function physicalFittingReceipt(reports, producer, nonce) {
  const desktop = reports.find((report) => physicalPlatform(report.environment) !== "android");
  const android = reports.find((report) => physicalPlatform(report.environment) === "android");
  if (!desktop || !android) {
    throw new Error(
      "physicalBrowserFitting observer capability is unavailable: desktop and Android reports are both required",
    );
  }
  const observations = reports.map((report) =>
    requireReleaseEnvironment(report, "physicalBrowserFitting"),
  );
  const canonicalObservation = JSON.stringify(observations[0]);
  if (observations.some((observation) => JSON.stringify(observation) !== canonicalObservation)) {
    throw new Error("physicalBrowserFitting reports did not use identical runtime artifacts");
  }
  const maxMainThreadTaskMs = reports.reduce(
    (maximum, report) =>
      Math.max(maximum, report.environment.maxMainThreadTaskMs ?? Number.NaN),
    0,
  );
  if (!Number.isFinite(maxMainThreadTaskMs)) {
    throw new Error(
      "physicalBrowserFitting observer capability is unavailable: maxMainThreadTaskMs was not measured",
    );
  }
  for (const report of reports) {
    if (
      report.environment.adapterFallback !== "hardware" ||
      report.environment.longTaskObserverSupported !== true ||
      report.checks.curved_fitting.measurements.opfsSpoolingObserved !== true
    ) {
      throw new Error("physicalBrowserFitting observer did not pass fitting requirements");
    }
  }
  return {
    schemaVersion: 1,
    evidenceType: "physicalBrowserFitting",
    producerId: producer.id,
    nonce,
    producedAt: reports
      .map((report) => report.producedAt)
      .sort()
      .at(-1),
    passed: true,
    fixturePath: producer.fixturePath,
    results: {
      desktopChromePassed: true,
      androidChromePassed: true,
      nonFallbackHardwareAdapterConfirmed: true,
      opfsSpoolingPassed: true,
      uiResponsivePassed: maxMainThreadTaskMs <= 50,
      fitCompleted: true,
      maxMainThreadTaskMs,
      desktopMeasuredAt: desktop.producedAt,
      androidMeasuredAt: android.producedAt,
      ...observations[0],
    },
  };
}

function validateFullRuntimeReport(report) {
  if (
    !report ||
    typeof report !== "object" ||
    report.$schema !== "drowse-real-browser-e2e-measurement-v1" ||
    report.schemaVersion !== 1 ||
    report.evidenceType !== "realBrowserEndToEnd" ||
    report.toolVersion !== "drowse-real-browser-e2e-v4" ||
    report.passed !== true ||
    report.releaseEvidence !== false ||
    !report.environment ||
    !report.checks ||
    Object.values(report.checks).some((check) => check?.passed !== true)
  ) {
    throw new Error("release evidence observer returned a malformed full-runtime report");
  }
  const producedAt = Date.parse(report.producedAt);
  if (
    !Number.isFinite(producedAt) ||
    producedAt > Date.now() + 5 * 60 * 1000 ||
    producedAt < Date.now() - MAX_PHYSICAL_REPORT_AGE_MS
  ) {
    throw new Error("release evidence observer returned a stale physical report");
  }
}

function requireReleaseEnvironment(report, evidenceType) {
  const environment = report.environment;
  if (environment.browserFlags !== "normal") {
    throw new Error(`${evidenceType} rejects unsafe compatibility browser flags`);
  }
  if (environment.adapterFallback !== "hardware") {
    throw new Error(`${evidenceType} requires a confirmed hardware adapter`);
  }
  const observed = environment.observedArtifacts;
  const keys = [
    "runtimeIdentitySha256",
    "webLlmSha256",
    "modelArtifactManifestSha256",
    "convertedManifestSha256",
    "modelLibrarySha256",
    "coreArtifactManifestSha256",
    "jlensArtifactManifestSha256",
    "saeArtifactManifestSha256",
  ];
  if (
    !observed ||
    typeof observed !== "object" ||
    Array.isArray(observed) ||
    Object.keys(observed).length !== keys.length ||
    keys.some((key) => !SHA256.test(observed[key] ?? "")) ||
    observed.runtimeIdentitySha256 !== environment.runtimeIdentitySha256
  ) {
    throw new Error(`${evidenceType} has incomplete observed artifact identities`);
  }
  const local = report.checks.local_artifacts.measurements;
  if (keys.some((key) => local[key] !== observed[key])) {
    throw new Error(`${evidenceType} observed artifact identities disagree with the harness`);
  }
  return Object.fromEntries(keys.map((key) => [key, observed[key]]));
}

function isNumericRecord(value) {
  return (
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.values(value).every((item) => Number.isFinite(item) && item >= 0)
  );
}

function physicalPlatform(environment) {
  if (/Android/iu.test(environment.userAgent ?? "")) return "android";
  if (environment.platform === "darwin") return "macos";
  if (environment.platform === "win32") return "windows";
  if (environment.platform === "linux") return "linux";
  throw new Error("physical browser report has an unsupported platform");
}

function isDirectObservation(report, evidenceType) {
  if (!report || typeof report !== "object" || Array.isArray(report)) return false;
  const keys = Object.keys(report).sort();
  const expected = evidenceType === "physicalBrowserBenchmark"
    ? ["evidenceType", "passed", "producedAt", "run", "schemaVersion"]
    : [
        "evidenceType",
        "fixturePath",
        "passed",
        "producedAt",
        "results",
        "schemaVersion",
      ];
  return (
    keys.length === expected.length &&
    keys.every((key, index) => key === expected[index]) &&
    report.schemaVersion === 1 &&
    report.evidenceType === evidenceType &&
    report.passed === true
  );
}

function normalizeArgumentSets(value) {
  const sets = Array.isArray(value) && value.every((entry) => typeof entry === "string")
    ? [value]
    : value;
  if (
    !Array.isArray(sets) ||
    sets.length === 0 ||
    sets.length > 8 ||
    sets.some(
      (arguments_) =>
        !Array.isArray(arguments_) ||
        arguments_.length > 128 ||
        arguments_.some(
          (argument) =>
            typeof argument !== "string" ||
            argument.length === 0 ||
            argument.length > 4_096 ||
            argument.includes("\0"),
        ),
    )
  ) {
    throw new Error("release evidence observer arguments are invalid");
  }
  return sets;
}

async function readConfig(path, root) {
  const absolute = resolve(root, path);
  const info = await requireRegularContainedFile(
    absolute,
    root,
    "release evidence producer config",
  );
  if (info.size > 256 * 1024) {
    throw new Error("release evidence producer config is too large");
  }
  try {
    return JSON.parse(await readFile(absolute, "utf8"));
  } catch (error) {
    throw new Error("release evidence producer config is not valid JSON", {
      cause: error,
    });
  }
}

async function requireRegularContainedFile(path, root, label) {
  const [rootRealPath, parentRealPath, info] = await Promise.all([
    realpath(root),
    realpath(dirname(path)),
    lstat(path),
  ]);
  const parentRelative = relative(rootRealPath, parentRealPath);
  if (
    parentRelative === ".." ||
    parentRelative.startsWith(`..${sep}`) ||
    info.isSymbolicLink() ||
    !info.isFile()
  ) {
    throw new Error(`${label} must be a regular file inside the Drowse repository`);
  }
  return info;
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

async function main() {
  const options = parseProducerArguments(process.argv.slice(2));
  const receipt = await runConfiguredObserver({
    evidenceType: options.evidenceType,
    configPath: options.config,
    nonce: options.nonce,
  });
  process.stdout.write(`${JSON.stringify(receipt)}\n`);
}

if (
  process.argv[1] &&
  await realpath(fileURLToPath(import.meta.url)) === await realpath(process.argv[1])
) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
