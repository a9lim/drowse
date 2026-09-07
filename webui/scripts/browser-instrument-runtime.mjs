#!/usr/bin/env node

import { createServer as createHttpServer } from "node:http";
import { readFile, stat, writeFile } from "node:fs/promises";
import { extname, resolve } from "node:path";
import { createServer as createViteServer } from "vite";
import { lockedModelExecutionProfiles } from "./hosted-model-profiles.mjs";
import {
  grantReleaseToolStorageQuota,
  launchReleaseToolContext,
} from "./release-tool-storage-quota.mjs";
import { verifyArtifactFile } from "./verify-artifact-file.mjs";

const options = parseArguments(process.argv.slice(2));
const webuiRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(webuiRoot, "..");
const runtimeLock = JSON.parse(
  await readFile(
    resolve(repositoryRoot, "browser-runtime/runtime-lock.json"),
    "utf8",
  ),
);
const lock = runtimeLock.models.find((entry) => entry.id === options.modelId);
if (!lock) throw new Error(`unknown runtime-lock model ${options.modelId}`);
const sets = {
  model: await artifactSet(
    resolve(options.modelDirectory),
    "hosted-artifacts.json",
  ),
  core: await artifactSet(
    resolve(options.coreDirectory),
    "hosted-pack-artifacts.json",
    "core_pack",
  ),
  jlens:
    options.diagnosticFamily === "sae"
      ? emptyArtifactSet()
      : await artifactSet(
          resolve(options.jlensDirectory),
          "hosted-pack-artifacts.json",
          "instrument",
        ),
  sae:
    options.diagnosticFamily === "jlens"
      ? emptyArtifactSet()
      : await artifactSet(
          resolve(options.saeDirectory),
          "hosted-pack-artifacts.json",
          "instrument",
        ),
};
const executionProfiles = lockedModelExecutionProfiles(
  lock,
  sets.model.manifest,
);
const modelLibrary = resolve(options.modelLibrary);
const webLlm = resolve(options.webLlm);
await Promise.all([
  verifyArtifactFile(
    modelLibrary,
    sets.model.files.find((file) => file.role === "model_library"),
    "model library",
  ),
  requireFile(webLlm, "WebLLM bundle"),
]);
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
  }))
    response.setHeader(name, value);
  try {
    const url = new URL(request.url ?? "/", "http://127.0.0.1");
    if (url.pathname === "/") {
      return send(
        response,
        "<!doctype html><meta charset=utf-8><link rel=icon href=data:,>",
        "text/html",
      );
    }
    if (url.pathname === "/worker.js") {
      const transformed = await vite.transformRequest(
        "/scripts/hosted-instrument-runtime.worker.ts",
      );
      return send(
        response,
        transformed?.code ?? "worker transform failed",
        "text/javascript",
        transformed ? 200 : 500,
      );
    }
    if (url.pathname === "/webllm.js") {
      return send(response, await readFile(webLlm), "text/javascript");
    }
    if (url.pathname === "/artifact") {
      const setName = url.searchParams.get("set");
      const path = url.searchParams.get("path");
      const set = sets[setName];
      const entry = set?.files.find((candidate) => candidate.path === path);
      if (!entry) return send(response, "not found", "text/plain", 404);
      const absolute =
        setName === "model" && entry.role === "model_library"
          ? modelLibrary
          : resolve(set.directory, ...entry.path.split("/"));
      return send(response, await readFile(absolute), contentType(absolute));
    }
    vite.middlewares(request, response);
  } catch (error) {
    send(
      response,
      error instanceof Error ? (error.stack ?? error.message) : String(error),
      "text/plain",
      500,
    );
  }
});
await new Promise((resolvePromise, reject) => {
  server.once("error", reject);
  server.listen(0, "127.0.0.1", resolvePromise);
});
const address = server.address();
if (!address || typeof address === "string")
  throw new Error("failed to bind instrument runtime server");
const origin = `http://127.0.0.1:${address.port}`;
const browser = await launchReleaseToolContext({
  headless: true,
  channel: options.browserChannel,
  args: [
    "--enable-unsafe-webgpu",
    "--use-angle=metal",
    "--disable-gpu-sandbox",
  ],
});
try {
  const page = browser.page;
  page.setDefaultTimeout(options.timeoutMs);
  const failedRequests = [];
  const pageErrors = [];
  const consoleErrors = [];
  page.on("console", (message) => {
    process.stderr.write(`browser ${message.type()}: ${message.text()}\n`);
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) =>
    pageErrors.push(error.stack ?? error.message),
  );
  page.on("requestfailed", (request) =>
    failedRequests.push(
      `${request.url()} (${request.failure()?.errorText ?? "unknown"})`,
    ),
  );
  const external = [];
  page.on("request", (request) => {
    const url = request.url();
    if (
      !url.startsWith(`${origin}/`) &&
      !url.startsWith(`blob:${origin}/`) &&
      !url.startsWith("data:")
    ) {
      external.push(url);
    }
  });
  await page.goto(origin, { waitUntil: "load" });
  await grantReleaseToolStorageQuota(
    page,
    origin,
    Object.values(sets).flatMap((set) => set.files),
  );
  const browserResult = await page.evaluate(
    async ({ timeoutMs, payload }) => {
      const worker = new Worker("/worker.js", {
        type: "module",
        name: "drowse-focused-precomputed-instrument-proof",
      });
      const milestones = [];
      return await new Promise((resolvePromise, reject) => {
        const timeout = setTimeout(() => {
          worker.terminate();
          reject(
            new Error(`instrument runtime timed out after ${timeoutMs} ms`),
          );
        }, timeoutMs);
        worker.onerror = (event) => {
          clearTimeout(timeout);
          worker.terminate();
          reject(
            new Error(event.message || "instrument runtime worker failed"),
          );
        };
        worker.onmessageerror = () => {
          clearTimeout(timeout);
          worker.terminate();
          reject(
            new Error(
              "instrument runtime worker returned an unreadable message",
            ),
          );
        };
        worker.onmessage = (event) => {
          const message = event.data;
          if (
            message.type === "milestone" ||
            message.type === "token_milestone"
          ) {
            milestones.push(message);
            console.info(
              message.type === "token_milestone"
                ? `drowse-focused-token ${message.stage} ${message.rawIndex} ${Math.round(message.elapsedMs)}ms`
                : `drowse-focused-stage ${message.stage} ${message.status}${
                    typeof message.elapsedMs === "number"
                      ? ` ${Math.round(message.elapsedMs)}ms`
                      : ""
                  }`,
            );
            return;
          }
          clearTimeout(timeout);
          worker.terminate();
          if (message.type === "error") reject(new Error(message.message));
          else if (message.type === "done") {
            resolvePromise({ result: message.result, milestones });
          } else {
            reject(
              new Error(
                "instrument runtime worker returned an invalid terminal message",
              ),
            );
          }
        };
        worker.postMessage(payload);
      });
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
        jlensLayerLimit: options.jlensLayerLimit,
        diagnosticFamily: options.diagnosticFamily,
        jlensWord: options.jlensWord,
        saeFeature: options.saeFeature,
        stepTimeoutMs: options.stepTimeoutMs,
      },
    },
  );
  const audit = { external, failedRequests, pageErrors, consoleErrors };
  const failures = Object.entries(audit).filter(
    ([, values]) => values.length > 0,
  );
  if (failures.length > 0) {
    throw new Error(
      `instrument runtime browser audit failed: ${JSON.stringify(Object.fromEntries(failures))}`,
    );
  }
  const report = {
    $schema:
      options.diagnosticFamily === "combined"
        ? "drowse-focused-precomputed-instrument-proof-v1"
        : "drowse-focused-precomputed-instrument-diagnostic-v1",
    producedAt: new Date().toISOString(),
    passed: true,
    releaseEvidence: false,
    browser: {
      channel: options.browserChannel,
      version: browser.version(),
      platform: process.platform,
      architecture: process.arch,
    },
    audit,
    milestones: browserResult.milestones,
    ...browserResult.result,
  };
  const output = `${JSON.stringify(report, null, 2)}\n`;
  if (options.output) {
    await writeFile(resolve(options.output), output, { flag: "wx" });
  }
  process.stdout.write(output);
} finally {
  await browser.close();
  await vite.close();
  await new Promise((resolvePromise) => server.close(resolvePromise));
}

async function artifactSet(directory, manifestName, role = null) {
  const manifest = JSON.parse(
    await readFile(resolve(directory, manifestName), "utf8"),
  );
  if (!Array.isArray(manifest.files) || manifest.files.length === 0) {
    throw new Error(`${manifestName} contains no files`);
  }
  const files = manifest.files.map((entry) => ({
    ...entry,
    role: role ?? entry.role,
  }));
  for (const entry of files) {
    if (
      typeof entry.path !== "string" ||
      entry.path
        .split("/")
        .some((part) => !part || part === "." || part === "..")
    ) {
      throw new Error(`${manifestName} contains an unsafe path`);
    }
    const absolute =
      role === null && entry.role === "model_library"
        ? resolve(directory, entry.path)
        : resolve(directory, ...entry.path.split("/"));
    await requireFile(absolute, entry.path);
  }
  return { directory, files, manifest };
}

function emptyArtifactSet() {
  return { directory: null, files: [], manifest: null };
}

function parseArguments(args) {
  const result = {
    browserChannel: "chrome",
    contextTokens: 2048,
    diagnosticFamily: "combined",
    jlensLayerLimit: null,
    jlensWord: "ocean",
    output: null,
    saeFeature: 17,
    stepTimeoutMs: 300_000,
    timeoutMs: 1_200_000,
  };
  const fields = {
    "--model-id": "modelId",
    "--model-directory": "modelDirectory",
    "--model-library": "modelLibrary",
    "--webllm": "webLlm",
    "--core-directory": "coreDirectory",
    "--jlens-directory": "jlensDirectory",
    "--sae-directory": "saeDirectory",
    "--browser-channel": "browserChannel",
    "--context-tokens": "contextTokens",
    "--diagnostic-family": "diagnosticFamily",
    "--jlens-layer-limit": "jlensLayerLimit",
    "--jlens-word": "jlensWord",
    "--output": "output",
    "--sae-feature": "saeFeature",
    "--step-timeout-ms": "stepTimeoutMs",
    "--timeout-ms": "timeoutMs",
  };
  const seen = new Set();
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    const field = fields[argument];
    if (!field) throw new Error(`unexpected argument ${argument}`);
    if (seen.has(argument)) {
      throw new Error(`${argument} may be supplied only once`);
    }
    seen.add(argument);
    const value = args[++index];
    if (!value || value.startsWith("--"))
      throw new Error(`missing value for ${args[index - 1]}`);
    result[field] = [
      "contextTokens",
      "jlensLayerLimit",
      "saeFeature",
      "stepTimeoutMs",
      "timeoutMs",
    ].includes(field)
      ? Number(value)
      : value;
  }
  if (!["combined", "jlens", "sae"].includes(result.diagnosticFamily)) {
    throw new Error("invalid diagnostic family");
  }
  const requiredFields = [
    "modelId",
    "modelDirectory",
    "modelLibrary",
    "webLlm",
    "coreDirectory",
  ];
  if (result.diagnosticFamily !== "sae") requiredFields.push("jlensDirectory");
  if (result.diagnosticFamily !== "jlens") requiredFields.push("saeDirectory");
  for (const field of requiredFields) {
    if (!result[field]) throw new Error(`${field} is required`);
  }
  if (!Number.isSafeInteger(result.contextTokens) || result.contextTokens < 1)
    throw new Error("invalid context tokens");
  if (
    result.jlensLayerLimit !== null &&
    (!Number.isSafeInteger(result.jlensLayerLimit) || result.jlensLayerLimit < 1)
  ) {
    throw new Error("invalid J-lens layer limit");
  }
  if (!Number.isSafeInteger(result.saeFeature) || result.saeFeature < 0)
    throw new Error("invalid SAE feature");
  if (
    !Number.isSafeInteger(result.stepTimeoutMs) ||
    result.stepTimeoutMs < 10_000
  )
    throw new Error("invalid step timeout");
  if (!Number.isSafeInteger(result.timeoutMs) || result.timeoutMs < 10_000)
    throw new Error("invalid timeout");
  if (result.stepTimeoutMs > result.timeoutMs)
    throw new Error("step timeout exceeds total timeout");
  if (
    typeof result.jlensWord !== "string" ||
    !result.jlensWord.trim() ||
    result.jlensWord.length > 128
  ) {
    throw new Error("invalid J-lens word");
  }
  return { ...result, jlensWord: result.jlensWord.trim() };
}

async function requireFile(path, label) {
  const info = await stat(path);
  if (!info.isFile())
    throw new Error(`${label} is not a regular file: ${path}`);
}

function contentType(path) {
  if (extname(path) === ".json") return "application/json";
  if (extname(path) === ".wasm") return "application/wasm";
  return "application/octet-stream";
}

function send(response, body, type, status = 200) {
  response.writeHead(status, { "Content-Type": type });
  response.end(body);
}
