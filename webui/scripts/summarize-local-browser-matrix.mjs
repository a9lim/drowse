#!/usr/bin/env node

import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { REQUIRED_BROWSER_FULL_RUNTIME_CHECKS } from "./browser-full-runtime-contract.mjs";
import { lockedRuntimeIdentitySha256 } from "./check-runtime-lock.mjs";

const repositoryRoot = resolve(import.meta.dirname, "../..");

export function parseArguments(args) {
  const reports = [];
  let output = null;
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--output") {
      if (output !== null) throw new Error("--output may be supplied only once");
      const value = args[++index];
      if (!value || value.startsWith("--")) throw new Error("--output requires a path");
      output = resolve(value);
      continue;
    }
    if (args[index].startsWith("--")) throw new Error(`unexpected argument ${args[index]}`);
    reports.push(resolve(args[index]));
  }
  if (reports.length === 0) {
    throw new Error("usage: summarize-local-browser-matrix.mjs REPORT.json... [--output SUMMARY.json]");
  }
  return { reports, output };
}

export function summarizeLocalBrowserMatrix(reports, runtimeLock, benchmarkLock) {
  if (!Array.isArray(reports) || reports.length === 0) throw new Error("no browser reports supplied");
  const cells = reports.map((report) => validateReport(report, runtimeLock));
  const keys = cells.map((cell) => [
    cell.modelId,
    cell.contextTokens,
    cell.platform,
    cell.browser,
    cell.browserVersion,
    cell.device,
  ].join("\0"));
  if (new Set(keys).size !== keys.length) throw new Error("local browser reports repeat a matrix cell");
  const required = benchmarkLock.requiredMatrix;
  const requiredCells = required.modelIds.length * required.contextTokens.length *
    required.platforms.length * required.browsers.length;
  const coveredReleaseCoordinates = new Set(cells.map((cell) => [
    cell.modelId,
    cell.contextTokens,
    cell.platform,
    cell.browser,
  ].join("\0"))).size;
  return {
    $schema: "drowse-local-browser-matrix-summary-v1",
    schemaVersion: 1,
    releaseEvidence: false,
    warning: "Local engineering reports are not release evidence; short-prompt throughput is diagnostic and these records cannot be copied into benchmark-evidence.json.",
    coverage: {
      validatedLocalRuns: cells.length,
      coveredReleaseCoordinates,
      requiredReleaseCoordinates: requiredCells,
      percentOfRequiredCoordinates: Number((coveredReleaseCoordinates * 100 / requiredCells).toFixed(2)),
    },
    cells: cells.sort((left, right) =>
      left.modelId.localeCompare(right.modelId) ||
      left.contextTokens - right.contextTokens ||
      left.platform.localeCompare(right.platform) ||
      left.browser.localeCompare(right.browser)
    ),
  };
}

function validateReport(report, runtimeLock) {
  if (
    !record(report) || report.$schema !== "drowse-real-browser-e2e-measurement-v1" ||
    report.schemaVersion !== 1 || report.evidenceType !== "realBrowserEndToEnd" ||
    report.passed !== true || report.releaseEvidence !== false
  ) throw new Error("browser report is not a passing non-release E2E measurement");
  const environment = report.environment;
  const model = runtimeLock.models.find((entry) => entry.id === environment?.modelId);
  if (!model || !model.contextProfiles.includes(environment.contextTokens)) {
    throw new Error("browser report model or context is outside the runtime lock");
  }
  const identity = lockedRuntimeIdentitySha256(runtimeLock, model);
  if (identity === null || environment.runtimeIdentitySha256 !== identity) {
    throw new Error("browser report runtime identity differs from the runtime lock");
  }
  if (environment.adapterFallback !== "hardware" || environment.deviceLosses?.length !== 0) {
    throw new Error("browser report does not confirm a stable hardware adapter");
  }
  const audit = environment.browserAudit;
  for (const name of [
    "externalRequests", "failedRequests", "uncaughtPageErrors", "errorConsoleMessages",
    "webGpuErrors", "offlineServerRequests", "deviceLosses",
  ]) {
    if (!Array.isArray(audit?.[name]) || audit[name].length !== 0) {
      throw new Error(`browser report audit ${name} is not empty`);
    }
  }
  if (
    !record(report.checks) ||
    Object.keys(report.checks).length !== REQUIRED_BROWSER_FULL_RUNTIME_CHECKS.length ||
    REQUIRED_BROWSER_FULL_RUNTIME_CHECKS.some((name) => report.checks[name]?.passed !== true)
  ) throw new Error("browser report does not pass the complete E2E check roster");
  const generation = report.checks.ordinary_generation.measurements;
  if (
    !Number.isFinite(generation.prefillTokensPerSecond) || generation.prefillTokensPerSecond <= 0 ||
    !Number.isFinite(generation.decodeTokensPerSecond) || generation.decodeTokensPerSecond <= 0
  ) throw new Error("browser report generation throughput is invalid");
  const platform = environment.platform === "darwin"
    ? "macos"
    : environment.platform === "win32"
      ? "windows"
      : environment.platform;
  const browser = environment.browserChannel === "msedge" ? "edge" : environment.browserChannel;
  return {
    modelId: environment.modelId,
    contextTokens: environment.contextTokens,
    platform,
    browser,
    browserVersion: environment.browserVersion,
    device: `${environment.adapterInfo?.vendor ?? "unknown"}/${environment.adapterInfo?.architecture ?? "unknown"}`,
    deviceMemoryGiB: environment.deviceMemoryGiB ?? null,
    diagnosticPrefillTokensPerSecond: generation.prefillTokensPerSecond,
    diagnosticDecodeTokensPerSecond: generation.decodeTokensPerSecond,
    producedAt: report.producedAt,
    checksPassed: REQUIRED_BROWSER_FULL_RUNTIME_CHECKS.length,
    offlineNetworkRequests: report.checks.offline_reuse.measurements.networkRequests,
  };
}

function record(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const [reports, runtimeLock, benchmarkLock] = await Promise.all([
    Promise.all(options.reports.map((path) => readFile(path, "utf8").then(JSON.parse))),
    readFile(resolve(repositoryRoot, "browser-runtime/runtime-lock.json"), "utf8").then(JSON.parse),
    readFile(resolve(repositoryRoot, "browser-runtime/benchmark-evidence.json"), "utf8").then(JSON.parse),
  ]);
  const summary = summarizeLocalBrowserMatrix(reports, runtimeLock, benchmarkLock);
  const bytes = `${JSON.stringify(summary, null, 2)}\n`;
  if (options.output !== null) await writeFile(options.output, bytes, { flag: "wx" });
  process.stdout.write(bytes);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
