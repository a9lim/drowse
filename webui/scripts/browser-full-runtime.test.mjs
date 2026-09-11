import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import test from "node:test";
import { createServer } from "vite";
import {
  BROWSER_FULL_RUNTIME_TOOL_VERSION,
  REQUIRED_BROWSER_FULL_RUNTIME_CHECKS,
  createBrowserFullRuntimeReport,
  isBrowserFullRuntimeWebGpuError,
  parseBrowserFullRuntimeArguments,
  validateBrowserFullRuntimeAudit,
  validateBrowserFullRuntimeModelClosure,
  validateBrowserFullRuntimeTranscript,
  validatePairedSteeringControls,
} from "./browser-full-runtime-contract.mjs";
import {
  artifactSetReceipt,
  artifactUrl,
  validateBrowserArtifactAudit,
} from "./browser-full-runtime-artifact-audit.mjs";

const requiredArgs = [
  "--model-id",
  "smollm2-360m-instruct",
  "--model-directory",
  "/tmp/model",
  "--model-library",
  "/tmp/model.wasm",
  "--webllm",
  "/tmp/webllm.js",
  "--core-directory",
  "/tmp/core",
  "--jlens-directory",
  "/tmp/jlens",
  "--sae-directory",
  "/tmp/sae",
];

test("paired steering permits a forced token outside the sampler support without losing its zero probability", () => {
  const measured = validatePairedSteeringControls({ baseline: -0.5, zero: -0.5, positive: -2, negative: -Infinity });
  assert.equal(measured.controlProbabilities.negative, 0);
  assert.equal(measured.controlLogprobs.negative, "-Infinity");
  assert.equal(measured.controlProbabilities.baseline, measured.controlProbabilities.zero);
  assert.deepEqual(JSON.parse(JSON.stringify(measured)), measured);
});

test("paired steering rejects missing, invalid, changed-zero, and ineffective controls", () => {
  const valid = { baseline: -0.5, zero: -0.5, positive: -2, negative: -3 };
  for (const value of [NaN, Infinity, 0.1, undefined]) {
    assert.throws(() => validatePairedSteeringControls({ ...valid, negative: value }), /invalid sampler logprobs/);
  }
  assert.throws(() => validatePairedSteeringControls({ ...valid, baseline: -Infinity }), /invalid sampler logprobs/);
  assert.throws(() => validatePairedSteeringControls({ ...valid, zero: -0.6 }), /zero steering changed/);
  assert.throws(() => validatePairedSteeringControls({ baseline: -0.5, zero: -0.5, positive: -0.5, negative: -0.5 }), /did not change/);
});

test("parses the complete real-browser E2E input contract", () => {
  const parsed = parseBrowserFullRuntimeArguments([
    ...requiredArgs,
    "--context-tokens",
    "4096",
    "--fit-layer",
    "12",
    "--max-tokens",
    "5",
    "--stop-max-tokens",
    "32",
    "--timeout-ms",
    "120000",
    "--sae-feature",
    "17",
    "--jlens-word",
    " blue ",
    "--browser-channel",
    "msedge",
    "--browser-flags",
    "normal",
    "--output",
    "/tmp/result.json",
    "--runtime-lock",
    "/tmp/candidate-runtime-lock.json",
    "--headed",
  ]);
  assert.equal(parsed.contextTokens, 4096);
  assert.equal(parsed.fitLayer, 12);
  assert.equal(parsed.maxTokens, 5);
  assert.equal(parsed.stopMaxTokens, 32);
  assert.equal(parsed.timeoutMs, 120000);
  assert.equal(parsed.saeFeature, 17);
  assert.equal(parsed.jlensWord, "blue");
  assert.equal(parsed.browserChannel, "msedge");
  assert.equal(parsed.browserFlags, "normal");
  assert.equal(parsed.headed, true);
  assert.equal(parsed.output, "/tmp/result.json");
  assert.equal(parsed.runtimeLock, "/tmp/candidate-runtime-lock.json");
});

test("requires every physical artifact family and rejects bypass flags", () => {
  assert.throws(
    () => parseBrowserFullRuntimeArguments(requiredArgs.slice(0, -2)),
    /--sae-directory is required/,
  );
  assert.throws(
    () => parseBrowserFullRuntimeArguments([...requiredArgs, "--skip-fitting"]),
    /unexpected argument --skip-fitting/,
  );
  assert.throws(
    () =>
      parseBrowserFullRuntimeArguments([
        ...requiredArgs,
        "--sae-directory",
        "/tmp/other",
      ]),
    /may be supplied only once/,
  );
});

test("rejects invalid browser, token, timeout, and feature controls", () => {
  assert.throws(
    () =>
      parseBrowserFullRuntimeArguments([
        ...requiredArgs,
        "--browser-channel",
        "firefox",
      ]),
    /chrome or msedge/,
  );
  assert.throws(
    () =>
      parseBrowserFullRuntimeArguments([
        ...requiredArgs,
        "--browser-flags",
        "unsafe",
      ]),
    /compatibility or normal/,
  );
  assert.throws(
    () =>
      parseBrowserFullRuntimeArguments([...requiredArgs, "--max-tokens", "1"]),
    /at least 2/,
  );
  assert.throws(
    () =>
      parseBrowserFullRuntimeArguments([
        ...requiredArgs,
        "--stop-max-tokens",
        "7",
      ]),
    /at least 8/,
  );
  assert.throws(
    () =>
      parseBrowserFullRuntimeArguments([
        ...requiredArgs,
        "--timeout-ms",
        "59999",
      ]),
    /at least 60000/,
  );
  assert.throws(
    () =>
      parseBrowserFullRuntimeArguments([
        ...requiredArgs,
        "--sae-feature",
        "-1",
      ]),
    /non-negative integer/,
  );
});

test("requires a model closure that exactly matches the selected runtime lock", () => {
  const fixture = modelClosure();
  assert.doesNotThrow(() => validateBrowserFullRuntimeModelClosure(fixture));
  for (const [label, mutate, expected] of [
    [
      "ABI",
      (value) => {
        value.manifest.hookAbi = "wrong";
      },
      /hook ABI/,
    ],
    [
      "source",
      (value) => {
        value.manifest.source.revision = "b".repeat(40);
      },
      /source identity/,
    ],
    [
      "layer map",
      (value) => {
        value.manifest.layerMap = [1, 0];
      },
      /layer map/,
    ],
    [
      "context",
      (value) => {
        value.manifest.contextWindowSize = 1024;
      },
      /context profile/,
    ],
    [
      "library",
      (value) => {
        value.files.find((file) => file.role === "model_library").sha256 =
          "f".repeat(64);
      },
      /model_library/,
    ],
    [
      "weights",
      (value) => {
        value.files = value.files.filter((file) => file.role !== "weight");
      },
      /no weight shards/,
    ],
  ]) {
    const broken = structuredClone(fixture);
    mutate(broken);
    assert.throws(
      () => validateBrowserFullRuntimeModelClosure(broken),
      expected,
      label,
    );
  }
});

test("accepts only the production loader's exact legacy runtime aliases", () => {
  for (const runtimeAbi of ["saklas-web-runtime-v1", "polythetic-web-runtime-v1"]) {
    const fixture = modelClosure();
    fixture.runtimeLock.runtimeAbi = "drowse-web-runtime-v1";
    fixture.manifest.runtimeAbi = runtimeAbi;
    const original = structuredClone(fixture);
    assert.doesNotThrow(() => validateBrowserFullRuntimeModelClosure(fixture));
    assert.deepEqual(fixture, original);

    fixture.files.find(file => file.role === "model_library").sha256 = "f".repeat(64);
    assert.throws(() => validateBrowserFullRuntimeModelClosure(fixture), /model_library/);
  }
  for (const runtimeAbi of ["saklas-web-runtime-v2", "other-web-runtime-v1"]) {
    const fixture = modelClosure();
    fixture.runtimeLock.runtimeAbi = "drowse-web-runtime-v1";
    fixture.manifest.runtimeAbi = runtimeAbi;
    assert.throws(() => validateBrowserFullRuntimeModelClosure(fixture), /runtime ABI/);
  }
  const future = modelClosure();
  future.runtimeLock.runtimeAbi = "drowse-web-runtime-v2";
  future.manifest.runtimeAbi = "saklas-web-runtime-v1";
  assert.throws(() => validateBrowserFullRuntimeModelClosure(future), /runtime ABI/);
});

test("recognizes uncaptured WebGPU and device-loss console failures", () => {
  for (const message of [
    "WebGPU error was not captured. Validation failed for bind group 0",
    "Uncaught GPUValidationError: binding 8 exceeds the device limit",
    "[WebGPU] Validation error: buffer is too small",
    "GPUOutOfMemoryError while allocating a storage buffer",
    "The GPU device was lost during generation",
    "Error: WebGPU error #1: invalid shader module",
  ]) {
    assert.equal(isBrowserFullRuntimeWebGpuError(message), true, message);
  }
  for (const message of [
    "WebGPU initialized successfully",
    "WebGPU error scopes enabled",
    "GPU adapter limits were measured",
  ]) {
    assert.equal(isBrowserFullRuntimeWebGpuError(message), false, message);
  }
});

test("fails closed when any browser error-audit ledger is non-empty", () => {
  const audit = emptyBrowserAudit();
  assert.deepEqual(validateBrowserFullRuntimeAudit(audit), audit);
  for (const name of Object.keys(audit)) {
    const failed = structuredClone(audit);
    failed[name].push(
      name === "deviceLosses" ? { code: "WEBGPU_DEVICE_LOST" } : "observed",
    );
    assert.throws(
      () => validateBrowserFullRuntimeAudit(failed),
      new RegExp(`audit ${name} is non-empty`),
    );
  }
  assert.throws(
    () => validateBrowserFullRuntimeAudit({ ...audit, webGpuErrors: null }),
    /audit webGpuErrors is invalid/,
  );
});

test("cross-verifies only exact artifact EOF aborts after server and OPFS proof", () => {
  const clean = artifactAuditFixture();
  assert.equal(
    clean.workerReceipt.sets.jlens.entriesSha256,
    "b1901279cd85b354bbb1d581f7b50f91b6ecc48faeab3a9444ec8aa1576312ad",
  );
  assert.deepEqual(validateBrowserArtifactAudit(clean), {
    files: 1,
    bytes: 16,
    crossVerifiedArtifactEofAborts: [],
  });

  const aborted = artifactAuditFixture();
  aborted.observedRequests[0].finished = false;
  aborted.observedRequests[0].failed = true;
  aborted.observedRequests[0].errorText = "net::ERR_ABORTED";
  assert.deepEqual(validateBrowserArtifactAudit(aborted), {
    files: 1,
    bytes: 16,
    crossVerifiedArtifactEofAborts: [
      {
        url: aborted.observedRequests[0].url,
        set: "jlens",
        path: "packs/jlens/layer-1.safetensors",
        bytes: 16,
        sha256: "a".repeat(64),
        errorText: "net::ERR_ABORTED",
        serverResponseFinished: true,
        opfsSizeAndSha256Verified: true,
      },
    ],
  });

  const mutations = [
    (value) => {
      value.observedRequests[0].url = value.observedRequests[0].url.replace(
        "127.0.0.1",
        "localhost",
      );
    },
    (value) => {
      value.observedRequests[0].url = value.observedRequests[0].url.replace(
        "12345",
        "12346",
      );
    },
    (value) => {
      value.observedRequests[0].url += "&extra=1";
    },
    (value) => {
      value.observedRequests.push(structuredClone(value.observedRequests[0]));
    },
    (value) => {
      value.observedRequests.length = 0;
    },
    (value) => {
      value.observedRequests[0].method = "POST";
    },
    (value) => {
      value.observedRequests[0].resourceType = "xhr";
    },
    (value) => {
      value.observedRequests[0].responseStatus = 206;
    },
    (value) => {
      value.observedRequests[0].responseContentLength = 15;
    },
    (value) => {
      value.observedRequests[0].failed = true;
      value.observedRequests[0].errorText = "net::ERR_ABORTED";
    },
    (value) => {
      value.observedRequests[0].finished = false;
    },
    (value) => {
      value.observedRequests[0].finished = false;
      value.observedRequests[0].failed = true;
      value.observedRequests[0].errorText = "net::ERR_FAILED";
    },
    (value) => {
      value.serverLifecycles.length = 0;
    },
    (value) => {
      value.serverLifecycles.push(structuredClone(value.serverLifecycles[0]));
    },
    (value) => {
      value.serverLifecycles[0].method = "POST";
    },
    (value) => {
      value.serverLifecycles[0].statusCode = 206;
    },
    (value) => {
      value.serverLifecycles[0].expectedBytes = 15;
    },
    (value) => {
      value.serverLifecycles[0].finished = false;
    },
    (value) => {
      value.serverLifecycles[0].closed = false;
    },
    (value) => {
      value.serverLifecycles[0].writableFinished = false;
    },
    (value) => {
      value.workerReceipt.schemaVersion = 2;
    },
    (value) => {
      value.workerReceipt.sets.extra = value.workerReceipt.sets.jlens;
    },
    (value) => {
      delete value.workerReceipt.sets.jlens;
    },
    (value) => {
      value.workerReceipt.sets.jlens.fileCount = 2;
    },
    (value) => {
      value.workerReceipt.sets.jlens.totalBytes = 15;
    },
    (value) => {
      value.workerReceipt.sets.jlens.entriesSha256 = "b".repeat(64);
    },
  ];
  for (const mutate of mutations) {
    const broken = artifactAuditFixture();
    mutate(broken);
    assert.throws(() => validateBrowserArtifactAudit(broken));
  }
});

test("accepts only one measured pass for every required check in order", () => {
  const events = transcript();
  const checks = validateBrowserFullRuntimeTranscript(events);
  assert.deepEqual(Object.keys(checks), REQUIRED_BROWSER_FULL_RUNTIME_CHECKS);
  assert.ok(Object.values(checks).every((check) => check.passed === true));

  const report = createBrowserFullRuntimeReport({
    events,
    environment: { browser: "Chrome", version: "140.0.0.0" },
    producedAt: new Date("2026-08-29T12:00:00.000Z"),
  });
  assert.equal(report.toolVersion, BROWSER_FULL_RUNTIME_TOOL_VERSION);
  assert.equal(report.passed, true);
  assert.equal(report.releaseEvidence, false);
  assert.equal(report.evidenceType, "realBrowserEndToEnd");
  assert.equal(report.producedAt, "2026-08-29T12:00:00.000Z");
});

test("fails closed on skipped, reordered, unmeasured, duplicate, or extra checks", () => {
  const missing = transcript();
  missing.splice(5, 1);
  assert.throws(
    () => validateBrowserFullRuntimeTranscript(missing),
    /did not produce a measured pass/,
  );

  const reordered = transcript();
  [reordered[2], reordered[4]] = [reordered[4], reordered[2]];
  assert.throws(
    () => validateBrowserFullRuntimeTranscript(reordered),
    /required order/,
  );

  const unmeasured = transcript();
  unmeasured[1] = { ...unmeasured[1], measurements: {} };
  assert.throws(
    () => validateBrowserFullRuntimeTranscript(unmeasured),
    /has no measurements/,
  );

  const nonfinite = transcript();
  nonfinite[1] = { ...nonfinite[1], measurements: { observed: Number.NaN } };
  assert.throws(
    () => validateBrowserFullRuntimeTranscript(nonfinite),
    /non-finite/,
  );

  const extra = [...transcript(), { type: "check_started", check: "invented" }];
  assert.throws(
    () => validateBrowserFullRuntimeTranscript(extra),
    /extra events/,
  );
});

test("transforms the real-browser orchestration worker through the production module graph", async () => {
  const server = await createServer({
    root: resolve(import.meta.dirname, ".."),
    configFile: false,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true, watch: null },
  });
  try {
    const transformed = await server.transformRequest(
      "/scripts/hosted-full-runtime.worker.ts",
    );
    assert.ok(transformed?.code.includes("browser_e2e_curve"));
    assert.ok(transformed?.code.includes("alternate-token fork"));
    assert.ok(transformed?.code.includes("offline OPFS reuse"));
    assert.ok(transformed?.code.includes("artifactVerification"));
    assert.ok(transformed?.code.includes("setSaeDictionary"));
    assert.ok(
      transformed?.code.includes("exact full-vocabulary J-lens readout"),
    );
    assert.ok(transformed?.code.includes("exact full-dictionary SAE readout"));
    assert.ok(transformed?.code.includes("browser_e2e_jlens_reload"));
    assert.ok(transformed?.code.includes("browser_e2e_sae_offline"));
    assert.match(
      transformed?.code ?? "",
      /REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE\s*=\s*10/u,
    );
    assert.match(
      transformed?.code ?? "",
      /REQUIRED_MAX_COMPUTE_WORKGROUP_SIZE_X\s*=\s*256/u,
    );
    assert.match(
      transformed?.code ?? "",
      /REQUIRED_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP\s*=\s*256/u,
    );
  } finally {
    await server.close();
  }
});

test("streams physical check progress before the terminal report", async () => {
  const source = await readFile(
    resolve(import.meta.dirname, "browser-full-runtime.mjs"),
    "utf8",
  );
  assert.match(
    source,
    /console\.info\(`drowse-e2e-check \$\{message\.check\} \$\{message\.type\}`\)/u,
  );
});

test("the feasibility harness accepts every launch architecture", async () => {
  const source = await readFile(
    resolve(import.meta.dirname, "browser-runtime-feasibility.mjs"),
    "utf8",
  );
  assert.match(
    source,
    /\["qwen3", "llama", "gemma3_text"\]\.includes\(result\.architecture\)/u,
  );
  assert.match(
    source,
    /--architecture must be qwen3, llama, or gemma3_text/u,
  );
});

function transcript() {
  return REQUIRED_BROWSER_FULL_RUNTIME_CHECKS.flatMap((check, index) => [
    { type: "check_started", check },
    { type: "check_passed", check, measurements: { observed: index + 1 } },
  ]);
}

function modelClosure() {
  const digests = {
    converted_manifest: "a".repeat(64),
    model_library: "b".repeat(64),
    tokenizer: "c".repeat(64),
    chat_template: "d".repeat(64),
  };
  return {
    contextTokens: 2048,
    runtimeLock: { runtimeAbi: "runtime-v1", hookAbi: "hook-v1" },
    lock: {
      architecture: "llama",
      quantization: "q4f16_1",
      hiddenSize: 8,
      layerMap: [0, 1],
      sourceRepository: "owner/model",
      sourceRevision: "e".repeat(40),
      manifestSha256: digests.converted_manifest,
      librarySha256: digests.model_library,
      tokenizerSha256: digests.tokenizer,
      chatTemplateSha256: digests.chat_template,
    },
    manifest: {
      schemaVersion: 1,
      runtimeAbi: "runtime-v1",
      hookAbi: "hook-v1",
      architecture: "llama",
      quantization: "q4f16_1",
      hiddenSize: 8,
      layerMap: [0, 1],
      contextWindowSize: 4096,
      source: { repository: "owner/model", revision: "e".repeat(40) },
    },
    files: [
      ...Object.entries(digests).map(([role, sha256]) => ({ role, sha256 })),
      { role: "weight", sha256: "f".repeat(64) },
    ],
  };
}

function emptyBrowserAudit() {
  return {
    externalRequests: [],
    failedRequests: [],
    uncaughtPageErrors: [],
    errorConsoleMessages: [],
    webGpuErrors: [],
    offlineServerRequests: [],
    deviceLosses: [],
  };
}

function artifactAuditFixture() {
  const origin = "http://127.0.0.1:12345";
  const file = {
    path: "packs/jlens/layer-1.safetensors",
    bytes: 16,
    sha256: "a".repeat(64),
  };
  const url = artifactUrl(origin, "jlens", file.path);
  return {
    origin,
    expectedSets: { jlens: [file] },
    observedRequests: [
      {
        url,
        method: "GET",
        resourceType: "fetch",
        responseStatus: 200,
        responseContentLength: 16,
        finished: true,
        failed: false,
        errorText: null,
      },
    ],
    serverLifecycles: [
      {
        set: "jlens",
        path: file.path,
        method: "GET",
        statusCode: 200,
        expectedBytes: 16,
        finished: true,
        closed: true,
        writableFinished: true,
      },
    ],
    workerReceipt: {
      schemaVersion: 1,
      sets: { jlens: artifactSetReceipt("jlens", [file]) },
    },
  };
}
