import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  parseArguments,
  summarizeLocalBrowserMatrix,
} from "./summarize-local-browser-matrix.mjs";
import { REQUIRED_BROWSER_FULL_RUNTIME_CHECKS } from "./browser-full-runtime-contract.mjs";
import { lockedRuntimeIdentitySha256 } from "./check-runtime-lock.mjs";

const repositoryRoot = resolve(import.meta.dirname, "../..");
const runtimeLock = JSON.parse(await readFile(
  resolve(repositoryRoot, "browser-runtime/runtime-lock.json"),
  "utf8",
));
const benchmarkLock = {
  requiredMatrix: {
    modelIds: ["gemma3-270m-instruct", "qwen3-1.7b"],
    contextTokens: [2048, 4096],
    platforms: ["macos", "windows"],
    browsers: ["chrome", "edge"],
  },
};
assert.deepEqual(parseArguments(["one.json", "--output", "result.json"]), {
  reports: [resolve("one.json")],
  output: resolve("result.json"),
});
assert.throws(() => parseArguments([]), /usage/);

const first = report("gemma3-270m-instruct", 2048);
const second = report("gemma3-270m-instruct", 4096);
const summary = summarizeLocalBrowserMatrix([first, second], runtimeLock, benchmarkLock);
assert.equal(summary.releaseEvidence, false);
assert.equal(summary.coverage.validatedLocalRuns, 2);
assert.equal(summary.coverage.requiredReleaseCoordinates, 16);
assert.equal(summary.coverage.percentOfRequiredCoordinates, 12.5);
assert.throws(
  () => summarizeLocalBrowserMatrix([first, first], runtimeLock, benchmarkLock),
  /repeat a matrix cell/,
);
const unsafe = structuredClone(first);
unsafe.releaseEvidence = true;
assert.throws(
  () => summarizeLocalBrowserMatrix([unsafe], runtimeLock, benchmarkLock),
  /non-release E2E measurement/,
);
const networked = structuredClone(first);
networked.environment.browserAudit.externalRequests.push("https://example.com/");
assert.throws(
  () => summarizeLocalBrowserMatrix([networked], runtimeLock, benchmarkLock),
  /externalRequests is not empty/,
);

console.log("Local browser-matrix summary checks passed");

function report(modelId, contextTokens) {
  const model = runtimeLock.models.find((entry) => entry.id === modelId);
  return {
    $schema: "drowse-real-browser-e2e-measurement-v1",
    schemaVersion: 1,
    evidenceType: "realBrowserEndToEnd",
    toolVersion: "drowse-real-browser-e2e-v4",
    producedAt: "2026-08-30T00:00:00.000Z",
    passed: true,
    releaseEvidence: false,
    environment: {
      modelId,
      contextTokens,
      runtimeIdentitySha256: lockedRuntimeIdentitySha256(runtimeLock, model),
      adapterFallback: "hardware",
      adapterInfo: { vendor: "fixture", architecture: "gpu" },
      deviceMemoryGiB: 16,
      deviceLosses: [],
      browserChannel: "chrome",
      browserVersion: "151.0.0.0",
      platform: "darwin",
      browserAudit: Object.fromEntries([
        "externalRequests", "failedRequests", "uncaughtPageErrors", "errorConsoleMessages",
        "webGpuErrors", "offlineServerRequests", "deviceLosses",
      ].map((name) => [name, []])),
    },
    checks: Object.fromEntries(REQUIRED_BROWSER_FULL_RUNTIME_CHECKS.map((name) => [
      name,
      {
        passed: true,
        measurements: name === "ordinary_generation"
          ? { prefillTokensPerSecond: 1, decodeTokensPerSecond: 2 }
          : name === "offline_reuse"
            ? { networkRequests: 0 }
            : {},
      },
    ])),
  };
}
