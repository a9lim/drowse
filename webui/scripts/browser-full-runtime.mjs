#!/usr/bin/env node

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { createServer as createHttpServer } from "node:http";
import { readFile, stat, writeFile } from "node:fs/promises";
import { extname, resolve } from "node:path";
import { pipeline } from "node:stream/promises";
import { createServer as createViteServer } from "vite";
import {
  createBrowserFullRuntimeReport,
  isBrowserFullRuntimeWebGpuError,
  parseBrowserFullRuntimeArguments,
  validateBrowserFullRuntimeAudit,
  validateBrowserFullRuntimeModelClosure,
  validateBrowserFullRuntimeTranscript,
} from "./browser-full-runtime-contract.mjs";
import { validateBrowserArtifactAudit } from "./browser-full-runtime-artifact-audit.mjs";
import { lockedModelExecutionProfiles } from "./hosted-model-profiles.mjs";
import { digestCanonical } from "./check-runtime-lock.mjs";
import {
  grantReleaseToolStorageQuota,
  launchReleaseToolContext,
} from "./release-tool-storage-quota.mjs";

const OFFLINE_NETWORK_QUIET_MS = 250;
const options = parseBrowserFullRuntimeArguments(process.argv.slice(2));
const webuiRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(webuiRoot, "..");
const runtimeLockPath = options.runtimeLock === null
  ? resolve(repositoryRoot, "browser-runtime/runtime-lock.json")
  : resolve(options.runtimeLock);
const runtimeLock = JSON.parse(
  await readFile(runtimeLockPath, "utf8"),
);
const lock = runtimeLock.models.find((entry) => entry.id === options.modelId);
if (!lock) throw new Error(`unknown runtime-lock model ${options.modelId}`);
if (!lock.contextProfiles.includes(options.contextTokens)) {
  throw new Error(
    `context ${options.contextTokens} is not locked for ${options.modelId}; choose ${lock.contextProfiles.join(" or ")}`,
  );
}
if (options.fitLayer !== null && !lock.layerMap.includes(options.fitLayer)) {
  throw new Error(
    `fit layer ${options.fitLayer} is outside the locked model layer map`,
  );
}
const modelLibrary = resolve(options.modelLibrary);
const webLlm = resolve(options.webLlm);
const sets = {
  model: await artifactSet(
    resolve(options.modelDirectory),
    "hosted-artifacts.json",
    {
      modelLibrary,
    },
  ),
  core: await artifactSet(
    resolve(options.coreDirectory),
    "hosted-pack-artifacts.json",
    {
      role: "core_pack",
    },
  ),
  jlens: await artifactSet(
    resolve(options.jlensDirectory),
    "hosted-pack-artifacts.json",
    {
      role: "instrument",
    },
  ),
  sae: await artifactSet(
    resolve(options.saeDirectory),
    "hosted-pack-artifacts.json",
    {
      role: "instrument",
    },
  ),
};
await requireFile(webLlm, "WebLLM bundle");
const webLlmBytes = (await stat(webLlm)).size;
if (webLlmBytes === 0) throw new Error("WebLLM bundle is empty");
const webLlmSha256 = await sha256File(webLlm);
validateBrowserFullRuntimeModelClosure({
  lock,
  runtimeLock,
  manifest: sets.model.manifest,
  files: sets.model.files,
  contextTokens: options.contextTokens,
});
const executionProfiles = lockedModelExecutionProfiles(
  lock,
  sets.model.manifest,
);
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
const localMeasurements = {
  modelFiles: sets.model.files.length,
  modelBytes: sumBytes(sets.model.files),
  coreFiles: sets.core.files.length,
  coreBytes: sumBytes(sets.core.files),
  jlensFiles: sets.jlens.files.length,
  jlensBytes: sumBytes(sets.jlens.files),
  saeFiles: sets.sae.files.length,
  saeBytes: sumBytes(sets.sae.files),
  webLlmBytes,
  webLlmSha256,
  runtimeIdentitySha256: digestCanonical(runtimeIdentity),
  modelArtifactManifestSha256: sets.model.manifestSha256,
  convertedManifestSha256: roleSha256(sets.model.files, "converted_manifest"),
  modelLibrarySha256: roleSha256(sets.model.files, "model_library"),
  coreArtifactManifestSha256: sets.core.manifestSha256,
  jlensArtifactManifestSha256: sets.jlens.manifestSha256,
  saeArtifactManifestSha256: sets.sae.manifestSha256,
  allManifestSizesAndHashesVerified: true,
};
let offlineArmed = false;
const offlineServerRequests = [];
const artifactRequestLifecycle = [];
const vite = await createViteServer({
  root: webuiRoot,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  publicDir: resolve(webuiRoot, "public-hosted"),
  server: { middlewareMode: true, hmr: false, watch: null },
});
const server = createHttpServer(async (request, response) => {
  for (const [name, value] of Object.entries({
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
  }))
    response.setHeader(name, value);
  try {
    const url = new URL(request.url ?? "/", "http://127.0.0.1");
    if (url.pathname === "/offline-arm" && request.method === "POST") {
      response.writeHead(204);
      response.end(() => {
        offlineArmed = true;
      });
      return;
    }
    if (offlineArmed) {
      offlineServerRequests.push(`${request.method ?? "GET"} ${url.pathname}`);
      return send(
        response,
        "offline E2E network cutoff is armed",
        "text/plain",
        503,
      );
    }
    if (url.pathname === "/") {
      return send(
        response,
        "<!doctype html><meta charset=utf-8><link rel=icon href=data:,><title>Drowse real browser E2E</title>",
        "text/html; charset=utf-8",
      );
    }
    if (url.pathname === "/worker.js") {
      const transformed = await vite.transformRequest(
        "/scripts/hosted-full-runtime.worker.ts",
      );
      return send(
        response,
        transformed?.code ?? "worker transform failed",
        "text/javascript; charset=utf-8",
        transformed ? 200 : 500,
      );
    }
    if (url.pathname === "/webllm.js") {
      return sendFile(
        response,
        webLlm,
        "text/javascript; charset=utf-8",
        webLlmBytes,
      );
    }
    if (url.pathname === "/artifact") {
      const setName = url.searchParams.get("set");
      const path = url.searchParams.get("path");
      const set = sets[setName];
      const entry = set?.files.find((candidate) => candidate.path === path);
      if (!entry) return send(response, "not found", "text/plain", 404);
      const lifecycle = {
        url: url.href,
        method: request.method ?? "GET",
        set: setName,
        path,
        statusCode: 200,
        expectedBytes: entry.bytes,
        finished: false,
        closed: false,
        writableFinished: false,
      };
      artifactRequestLifecycle.push(lifecycle);
      response.once("finish", () => {
        lifecycle.finished = true;
        lifecycle.writableFinished = response.writableFinished;
      });
      response.once("close", () => {
        lifecycle.closed = true;
        lifecycle.writableFinished = response.writableFinished;
      });
      const absolute =
        setName === "model" && entry.role === "model_library"
          ? modelLibrary
          : resolve(set.directory, ...entry.path.split("/"));
      return sendFile(response, absolute, contentType(absolute), entry.bytes);
    }
    vite.middlewares(request, response);
  } catch (error) {
    if (response.headersSent)
      response.destroy(error instanceof Error ? error : undefined);
    else {
      send(
        response,
        error instanceof Error ? (error.stack ?? error.message) : String(error),
        "text/plain",
        500,
      );
    }
  }
});
let browser = null;
try {
  await new Promise((resolvePromise, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolvePromise);
  });
  const address = server.address();
  if (!address || typeof address === "string")
    throw new Error("failed to bind real browser E2E server");
  const origin = `http://127.0.0.1:${address.port}`;
  const launchArgs =
    options.browserFlags === "compatibility"
      ? [
          "--enable-unsafe-webgpu",
          "--disable-gpu-sandbox",
          ...(process.platform === "darwin" ? ["--use-angle=metal"] : []),
        ]
      : [];
  browser = await launchReleaseToolContext({
    headless: !options.headed,
    channel: options.browserChannel,
    args: launchArgs,
  });
  const page = browser.page;
  page.setDefaultTimeout(options.timeoutMs);
  const externalRequests = [];
  const gpuErrors = [];
  const failedRequests = [];
  const artifactRequests = new Map();
  const uncaughtPageErrors = [];
  const errorConsoleMessages = [];
  page.on("request", (request) => {
    const url = request.url();
    if (isArtifactRequest(url, origin)) {
      artifactRequests.set(request, {
        url,
        method: request.method(),
        resourceType: request.resourceType(),
        responseStatus: null,
        responseContentLength: null,
        finished: false,
        failed: false,
        errorText: null,
      });
    }
    if (
      !url.startsWith(`${origin}/`) &&
      !url.startsWith(`blob:${origin}/`) &&
      !url.startsWith("data:")
    )
      externalRequests.push(url);
  });
  page.on("response", (response) => {
    const record = artifactRequests.get(response.request());
    if (!record) return;
    record.responseStatus = response.status();
    const value = response.headers()["content-length"];
    record.responseContentLength = /^(?:0|[1-9][0-9]*)$/u.test(value ?? "")
      ? Number(value)
      : null;
  });
  page.on("requestfinished", (request) => {
    const record = artifactRequests.get(request);
    if (record) record.finished = true;
  });
  page.on("requestfailed", (request) => {
    const record = artifactRequests.get(request);
    if (record) {
      record.failed = true;
      record.errorText = request.failure()?.errorText ?? "unknown";
    } else {
      failedRequests.push(
        `${request.url()} (${request.failure()?.errorText ?? "unknown"})`,
      );
    }
  });
  page.on("pageerror", (error) =>
    uncaughtPageErrors.push(error.stack ?? error.message),
  );
  page.on("console", (message) => {
    const value = message.text();
    process.stderr.write(`browser ${message.type()}: ${value}\n`);
    if (message.type() === "error") errorConsoleMessages.push(value);
    if (isBrowserFullRuntimeWebGpuError(value)) gpuErrors.push(value);
  });
  await page.goto(origin, { waitUntil: "load" });
  await grantReleaseToolStorageQuota(
    page,
    origin,
    Object.values(sets).flatMap((set) => set.files),
  );
  const browserResult = await page.evaluate(
    async ({ payload, timeoutMs }) => {
      let maxMainThreadTaskMs = 0;
      let maxMainThreadIntervalDelayMs = 0;
      let expectedTick = performance.now() + 10;
      const tick = setInterval(() => {
        const now = performance.now();
        maxMainThreadIntervalDelayMs = Math.max(
          maxMainThreadIntervalDelayMs,
          Math.max(0, now - expectedTick),
        );
        expectedTick = now + 10;
      }, 10);
      let longTaskObserver = null;
      let longTaskObserverSupported = false;
      if (PerformanceObserver.supportedEntryTypes.includes("longtask")) {
        longTaskObserverSupported = true;
        longTaskObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            maxMainThreadTaskMs = Math.max(maxMainThreadTaskMs, entry.duration);
          }
        });
        longTaskObserver.observe({ entryTypes: ["longtask"] });
      }
      const worker = new Worker("/worker.js", {
        type: "module",
        name: "drowse-real-browser-e2e",
      });
      const events = [];
      try {
        const result = await new Promise((resolvePromise, reject) => {
        const timeout = setTimeout(() => {
          worker.terminate();
          reject(new Error(`real browser E2E timed out after ${timeoutMs} ms`));
        }, timeoutMs);
        worker.onerror = (event) => {
          clearTimeout(timeout);
          worker.terminate();
          reject(new Error(event.message || "real browser E2E worker failed"));
        };
        worker.onmessageerror = () => {
          clearTimeout(timeout);
          worker.terminate();
          reject(
            new Error("real browser E2E worker returned an unreadable message"),
          );
        };
        worker.onmessage = (event) => {
          const message = event.data;
          if (
            message.type === "check_started" ||
            message.type === "check_passed"
          ) {
            events.push(message);
            console.info(`drowse-e2e-check ${message.check} ${message.type}`);
            return;
          }
          clearTimeout(timeout);
          if (message.type === "error") {
            worker.terminate();
            reject(new Error(`${message.check}: ${message.message}`));
          } else if (message.type === "done") {
            resolvePromise({
              events,
              environment: message.environment,
              artifactVerification: message.artifactVerification,
            });
          } else {
            worker.terminate();
            reject(
              new Error(
                "real browser E2E worker returned an invalid terminal message",
              ),
            );
          }
        };
        worker.postMessage(payload);
        });
        await new Promise((resolvePromise) => setTimeout(resolvePromise, 0));
        for (const entry of longTaskObserver?.takeRecords() ?? []) {
          maxMainThreadTaskMs = Math.max(maxMainThreadTaskMs, entry.duration);
        }
        return {
          ...result,
          mainThreadResponsiveness: {
            longTaskObserverSupported,
            maxMainThreadTaskMs,
            maxMainThreadIntervalDelayMs,
          },
        };
      } finally {
        clearInterval(tick);
        longTaskObserver?.disconnect();
      }
    },
    {
      timeoutMs: options.timeoutMs,
      payload: {
        type: "run",
        modelId: options.modelId,
        runtimeIdentity,
        ...executionProfiles,
        contextTokens: options.contextTokens,
        contextProfiles: lock.contextProfiles,
        modelFiles: sets.model.files,
        coreFiles: sets.core.files,
        jlensFiles: sets.jlens.files,
        saeFiles: sets.sae.files,
        maxTokens: options.maxTokens,
        stopMaxTokens: options.stopMaxTokens,
        fitLayer: options.fitLayer,
        jlensWord: options.jlensWord,
        saeFeature: options.saeFeature,
      },
    },
  );
  await page.waitForTimeout(OFFLINE_NETWORK_QUIET_MS);
  if (
    browserResult.environment?.runtimeIdentitySha256 !==
    localMeasurements.runtimeIdentitySha256
  ) {
    throw new Error("browser runtime identity does not match the staged artifact closure");
  }
  const events = [
    { type: "check_started", check: "local_artifacts" },
    {
      type: "check_passed",
      check: "local_artifacts",
      measurements: localMeasurements,
    },
    ...browserResult.events,
  ];
  validateBrowserFullRuntimeTranscript(events);
  const artifactRequestAudit = validateBrowserArtifactAudit({
    origin,
    expectedSets: Object.fromEntries(
      Object.entries(sets).map(([name, set]) => [name, set.files]),
    ),
    observedRequests: [...artifactRequests.values()],
    serverLifecycles: artifactRequestLifecycle,
    workerReceipt: browserResult.artifactVerification,
  });
  const browserAudit = {
    externalRequests,
    failedRequests,
    uncaughtPageErrors,
    errorConsoleMessages,
    webGpuErrors: gpuErrors,
    offlineServerRequests,
    deviceLosses: browserResult.environment?.deviceLosses ?? [],
  };
  let verifiedBrowserAudit;
  try {
    verifiedBrowserAudit = validateBrowserFullRuntimeAudit(browserAudit);
  } catch (error) {
    process.stderr.write(
      `artifact request lifecycle: ${JSON.stringify(artifactRequestLifecycle, null, 2)}\n`,
    );
    throw error;
  }
  const offlinePass = browserResult.events.find(
    (event) => event.type === "check_passed" && event.check === "offline_reuse",
  );
  if (
    offlinePass?.measurements &&
    typeof offlinePass.measurements === "object"
  ) {
    offlinePass.measurements.serverNetworkRequests =
      offlineServerRequests.length;
    offlinePass.measurements.networkQuietWindowMs = OFFLINE_NETWORK_QUIET_MS;
  }
  const report = createBrowserFullRuntimeReport({
    events,
    environment: {
      ...browserResult.environment,
      browserChannel: options.browserChannel,
      browserFlags: options.browserFlags,
      browserVersion: browser.version(),
      headed: options.headed,
      runtimeLockSource: options.runtimeLock === null ? "repository" : "override",
      platform: process.platform,
      architecture: process.arch,
      browserAudit: verifiedBrowserAudit,
      artifactRequestAudit,
      observedArtifacts: {
        runtimeIdentitySha256: localMeasurements.runtimeIdentitySha256,
        webLlmSha256,
        modelArtifactManifestSha256: sets.model.manifestSha256,
        convertedManifestSha256: localMeasurements.convertedManifestSha256,
        modelLibrarySha256: localMeasurements.modelLibrarySha256,
        coreArtifactManifestSha256: sets.core.manifestSha256,
        jlensArtifactManifestSha256: sets.jlens.manifestSha256,
        saeArtifactManifestSha256: sets.sae.manifestSha256,
      },
      ...browserResult.mainThreadResponsiveness,
    },
  });
  const output = `${JSON.stringify(report, null, 2)}\n`;
  if (options.output)
    await writeFile(resolve(options.output), output, { flag: "wx" });
  process.stdout.write(output);
} finally {
  await Promise.all([
    browser?.close() ?? Promise.resolve(),
    server.listening
      ? new Promise((resolvePromise) => server.close(resolvePromise))
      : Promise.resolve(),
    vite.close(),
  ]);
}

async function artifactSet(directory, manifestName, options = {}) {
  const info = await stat(directory);
  if (!info.isDirectory())
    throw new Error(`${manifestName} path is not a directory: ${directory}`);
  const manifestBytes = await readFile(resolve(directory, manifestName));
  const manifest = JSON.parse(manifestBytes.toString("utf8"));
  const manifestSha256 = createHash("sha256").update(manifestBytes).digest("hex");
  if (!Array.isArray(manifest.files) || manifest.files.length === 0) {
    throw new Error(`${manifestName} contains no files`);
  }
  const paths = new Set();
  const files = [];
  for (const raw of manifest.files) {
    if (
      !raw ||
      typeof raw !== "object" ||
      Array.isArray(raw) ||
      typeof raw.path !== "string" ||
      unsafePath(raw.path) ||
      paths.has(raw.path) ||
      !Number.isSafeInteger(raw.bytes) ||
      raw.bytes < 1 ||
      typeof raw.sha256 !== "string" ||
      !/^[a-f0-9]{64}$/u.test(raw.sha256)
    ) {
      throw new Error(`${manifestName} contains an invalid file entry`);
    }
    paths.add(raw.path);
    const entry = { ...raw, ...(options.role ? { role: options.role } : {}) };
    const absolute =
      options.modelLibrary && raw.role === "model_library"
        ? options.modelLibrary
        : resolve(directory, ...raw.path.split("/"));
    await verifyFile(
      absolute,
      entry.bytes,
      entry.sha256,
      `${manifestName}:${entry.path}`,
    );
    files.push(entry);
  }
  if (options.modelLibrary) {
    const libraries = files.filter((entry) => entry.role === "model_library");
    if (libraries.length !== 1)
      throw new Error(`${manifestName} must contain exactly one model library`);
  }
  return { directory, manifest, manifestSha256, files };
}

function roleSha256(files, role) {
  const matches = files.filter((file) => file.role === role);
  if (matches.length !== 1)
    throw new Error(`artifact set must contain exactly one ${role}`);
  return matches[0].sha256;
}

async function verifyFile(path, expectedBytes, expectedSha256, label) {
  const info = await stat(path);
  if (!info.isFile())
    throw new Error(`${label} is not a regular file: ${path}`);
  if (info.size !== expectedBytes) {
    throw new Error(
      `${label} has ${info.size} bytes, expected ${expectedBytes}`,
    );
  }
  const actual = await sha256File(path);
  if (actual !== expectedSha256) {
    throw new Error(
      `${label} has SHA-256 ${actual}, expected ${expectedSha256}`,
    );
  }
}

async function sha256File(path) {
  const hash = createHash("sha256");
  await new Promise((resolvePromise, reject) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", resolvePromise);
  });
  return hash.digest("hex");
}

async function requireFile(path, label) {
  const info = await stat(path);
  if (!info.isFile())
    throw new Error(`${label} is not a regular file: ${path}`);
}

function unsafePath(path) {
  return path
    .split("/")
    .some(
      (part) => !part || part === "." || part === ".." || part.includes("\\"),
    );
}

function isArtifactRequest(value, origin) {
  try {
    const url = new URL(value);
    return url.origin === origin && url.pathname === "/artifact";
  } catch {
    return false;
  }
}

function sumBytes(files) {
  return files.reduce((sum, file) => sum + file.bytes, 0);
}

function contentType(path) {
  if (extname(path) === ".json") return "application/json";
  if (extname(path) === ".wasm") return "application/wasm";
  if (extname(path) === ".safetensors") return "application/octet-stream";
  return "application/octet-stream";
}

function send(response, body, type, status = 200) {
  response.writeHead(status, {
    "Content-Type": type,
    "Content-Length": Buffer.byteLength(body),
  });
  response.end(body);
}

async function sendFile(response, path, type, bytes) {
  response.writeHead(200, { "Content-Type": type, "Content-Length": bytes });
  await pipeline(createReadStream(path), response);
}
