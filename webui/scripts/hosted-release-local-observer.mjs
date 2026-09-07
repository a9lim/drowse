#!/usr/bin/env node

import { execFile } from "node:child_process";
import { constants } from "node:fs";
import { access, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const repositoryRoot = fileURLToPath(new URL("../../", import.meta.url));
const webuiRoot = resolve(repositoryRoot, "webui");
const pythonExecutable = await resolvePythonExecutable();
const evidenceType = parseArguments(process.argv.slice(2));
const drowseHome = await mkdtemp(join(tmpdir(), "drowse-release-observer-"));

try {
  const results = evidenceType === "manifoldSerialization"
    ? await observeSerialization(drowseHome)
    : await observeSharedGoldens(drowseHome);
  process.stdout.write(`${JSON.stringify({
    schemaVersion: 1,
    evidenceType,
    producedAt: new Date().toISOString(),
    passed: true,
    fixturePath: evidenceType === "manifoldSerialization"
      ? "browser-runtime/fixtures/drowse-interchange-v1.json"
      : "browser-runtime/fixtures/shared-parity-suite-v1.json",
    results,
  })}\n`);
} finally {
  await rm(drowseHome, { recursive: true, force: true });
}

async function observeSerialization(drowseHome) {
  await runNode("webui/scripts/drowse-archive-browser.test.mjs");
  await runNode("webui/scripts/browser-core-pack.test.mjs");
  await runPython(["-m", "pytest", "-q", "tests/test_drowse_archive.py"], drowseHome);
  return {
    manifoldV10RoundTripPassed: true,
    safetensorsRoundTripPassed: true,
    drowseArchiveRoundTripPassed: true,
    pythonBrowserCompatibilityPassed: true,
    incompatibleFingerprintRejected: true,
  };
}

async function observeSharedGoldens(drowseHome) {
  for (const path of [
    "webui/scripts/steering-expression.test.mjs",
    "webui/scripts/browser-loom.test.mjs",
    "webui/scripts/worker-runtime.test.mjs",
    "webui/scripts/browser-core-pack.test.mjs",
    "webui/scripts/structured-hook-parity.test.mjs",
    "webui/scripts/sampling-parity.test.mjs",
    "webui/scripts/web-llm-generation.test.mjs",
    "webui/scripts/drowse-archive-browser.test.mjs",
  ]) {
    await runNode(path);
  }
  await runPython([
    "-m",
    "pytest",
    "-q",
    "tests/test_browser_steering_fixture.py",
    "tests/test_browser_loom_transcript_fixture.py",
    "tests/test_browser_measurements_fixture.py",
    "tests/test_browser_structured_program_fixture.py",
    "tests/test_browser_sampling_fixture.py",
    "tests/test_browser_topology_rbf_fixture.py",
    "tests/test_drowse_archive.py",
    "tests/test_loom_diff.py",
    "tests/test_measurements_envelope.py",
    "tests/test_manifold_math.py",
    "tests/test_manifold_topology.py",
  ], drowseHome);
  const fitting = JSON.parse(await runNode(
    "webui/scripts/fitting-wasm-browser.test.mjs",
    ["--release-evidence", "fittingKernelParity"],
  ));
  const numeric = fitting.results;
  const topology = JSON.parse(await runNode(
    "webui/scripts/fitting-wasm-browser.test.mjs",
    ["--release-evidence", "topologyOrchestration"],
  )).results;
  const maxNumericError = Math.max(
    numeric.neutralCenteringMaxAbsoluteError,
    numeric.whiteningMaxAbsoluteError,
    numeric.pcaProjectionMaxError,
    numeric.spectralProjectionMaxError,
    numeric.rbfMaxAbsoluteError,
    numeric.geometryMaxAbsoluteError,
    topology.maxEvaluatedOutputError,
  );
  if (!Number.isFinite(maxNumericError) || maxNumericError > 0.001) {
    throw new Error("shared golden numeric error exceeds 0.001");
  }
  return {
    expressionParsingPassed: true,
    treeMutationsPassed: true,
    measurementsPassed: true,
    structuredProgramsPassed: true,
    samplingPassed: true,
    manifoldGeometryPassed: true,
    fittingPassed: true,
    reciprocalArtifactsPassed: true,
    artifactCompatibilityRejectionPassed: true,
    maxNumericError,
  };
}

async function runNode(path, args = []) {
  const { stdout, stderr } = await execFileAsync(
    process.execPath,
    [resolve(repositoryRoot, path), ...args],
    executionOptions({}, webuiRoot),
  );
  if (stderr.trim()) throw new Error(`${path} wrote to stderr`);
  return stdout.trim();
}

async function runPython(args, drowseHome) {
  const { stderr } = await execFileAsync(
    pythonExecutable,
    args,
    executionOptions({ DROWSE_HOME: drowseHome }),
  );
  if (stderr.trim()) throw new Error(`python observer wrote to stderr: ${stderr.trim()}`);
}

function executionOptions(environment = {}, cwd = repositoryRoot) {
  return {
    cwd,
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
    timeout: 15 * 60 * 1000,
    windowsHide: true,
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1", ...environment },
  };
}

function parseArguments(args) {
  if (
    args.length !== 2 ||
    args[0] !== "--evidence-type" ||
    !["manifoldSerialization", "sharedGoldenResults"].includes(args[1])
  ) {
    throw new Error(
      "usage: hosted-release-local-observer.mjs --evidence-type manifoldSerialization|sharedGoldenResults",
    );
  }
  return args[1];
}

async function resolvePythonExecutable() {
  for (const candidate of [
    resolve(repositoryRoot, ".venv/bin/python"),
    resolve(repositoryRoot, ".venv/Scripts/python.exe"),
  ]) {
    if (await access(candidate, constants.X_OK).then(() => true, () => false)) {
      return candidate;
    }
  }
  return process.platform === "win32" ? "python" : "python3";
}
