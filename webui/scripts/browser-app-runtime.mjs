#!/usr/bin/env node

import { createServer as createHttpServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { basename, extname, resolve } from "node:path";
import { createServer as createViteServer } from "vite";
import {
  grantReleaseToolStorageQuota,
  launchReleaseToolContext,
} from "./release-tool-storage-quota.mjs";

const options = parseArguments(process.argv.slice(2));
const modelDirectory = resolve(options.model);
const modelLibraryPath = resolve(options.library);
const webLlmPath = resolve(options.webllm);
const artifactsManifest = JSON.parse(
  await readFile(resolve(modelDirectory, "hosted-artifacts.json"), "utf8"),
);
const artifacts = artifactsManifest.files.map((file) => ({ ...file }));
const runtimeIdentity = runtimeIdentityFromManifest(artifactsManifest);
for (const artifact of artifacts) {
  await requireFile(resolve(modelDirectory, artifact.path), `model ${artifact.path}`);
}
await requireFile(webLlmPath, "WebLLM bundle");
await requireFile(modelLibraryPath, "model library");

const vite = await createViteServer({
  root: resolve(import.meta.dirname, ".."),
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, hmr: false, watch: null },
});
const server = createHttpServer(async (request, response) => {
  for (const [name, value] of Object.entries({
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
  })) response.setHeader(name, value);
  try {
    const url = new URL(request.url ?? "/", "http://127.0.0.1");
    if (url.pathname === "/") return send(response, pageHtml(), "text/html; charset=utf-8");
    if (url.pathname === "/worker.js") {
      return send(
        response,
        workerSource(options, artifactsManifest, runtimeIdentity),
        "text/javascript; charset=utf-8",
      );
    }
    if (url.pathname === "/webllm.js") {
      return send(response, await readFile(webLlmPath), "text/javascript; charset=utf-8");
    }
    if (url.pathname.startsWith("/model/")) {
      const name = basename(url.pathname);
      const artifact = artifacts.find((candidate) => candidate.path === name);
      if (!artifact) return send(response, "not found", "text/plain", 404);
      const path = artifact.role === "model_library" ? modelLibraryPath : resolve(modelDirectory, name);
      return send(response, await readFile(path), contentType(name));
    }
    vite.middlewares(request, response);
  } catch (error) {
    console.error(error);
    send(response, "Internal server error", "text/plain", 500);
  }
});

await new Promise((resolvePromise, reject) => {
  server.once("error", reject);
  server.listen(0, "127.0.0.1", resolvePromise);
});
const address = server.address();
if (!address || typeof address === "string") throw new Error("failed to bind test server");
const origin = `http://127.0.0.1:${address.port}`;

const browser = await launchReleaseToolContext({
  headless: true,
  channel: options.browserChannel,
  args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--disable-gpu-sandbox"],
});
try {
  const page = browser.page;
  const browserErrors = [];
  const unexpectedRequests = [];
  page.setDefaultTimeout(options.timeoutMs);
  page.on("console", (message) => {
    const text = message.text();
    process.stderr.write(`browser console: ${text}\n`);
    if (message.type() === "error" || /WebGPU error was not captured/i.test(text)) {
      browserErrors.push(`${message.type()}: ${text}`);
    }
  });
  page.on("pageerror", (error) => browserErrors.push(`pageerror: ${error.message}`));
  page.on("request", (request) => {
    const url = request.url();
    if (
      !url.startsWith(`${origin}/`) &&
      !url.startsWith(`blob:${origin}/`) &&
      !url.startsWith("data:")
    ) unexpectedRequests.push(url);
  });
  await page.goto(origin, { waitUntil: "load" });
  await grantReleaseToolStorageQuota(page, origin, artifacts);
  const result = await page.evaluate(async ({ timeoutMs }) => {
    const worker = new Worker("/worker.js", { type: "module" });
    const events = [];
    return await new Promise((resolvePromise, reject) => {
      const timeout = setTimeout(() => {
        worker.terminate();
        reject(new Error(`app runtime worker timed out after ${timeoutMs} ms: ${JSON.stringify(events)}`));
      }, timeoutMs);
      worker.onerror = (event) => {
        clearTimeout(timeout);
        reject(new Error(event.message || "app runtime worker failed"));
      };
      worker.onmessage = (event) => {
        const message = event.data;
        events.push(message);
        if (message.type === "error") {
          clearTimeout(timeout);
          worker.terminate();
          reject(new Error(`${message.stage}: ${message.message}`));
        } else if (message.type === "done") {
          clearTimeout(timeout);
          worker.terminate();
          resolvePromise({
            ...message.result,
            stages: events
              .filter((entry) => entry.type === "stage" || entry.type === "trace")
              .map((entry) => entry.stage),
          });
        }
      };
      worker.postMessage({ type: "run" });
    });
  }, { timeoutMs: options.timeoutMs });
  if (unexpectedRequests.length !== 0) {
    throw new Error(`app runtime made external requests: ${JSON.stringify(unexpectedRequests)}`);
  }
  if (browserErrors.length !== 0) {
    throw new Error(`app runtime emitted browser errors: ${JSON.stringify(browserErrors)}`);
  }
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} finally {
  await browser.close();
  await vite.close();
  await new Promise((resolvePromise) => server.close(resolvePromise));
}

function workerSource(options, manifest, runtimeIdentity) {
  const revision = "a".repeat(40);
  const repository = `logitsml/drowse-web-${options.id}`;
  return `import * as webllm from "/webllm.js";
import { DrowseWebLlmRuntime } from "/src/hosted/runtime/webLlmEngine.ts";
import { resolveStructuredHookProfile } from "/src/hosted/runtime/structuredHookProfile.ts";
import { compileStructuredHookProgram } from "/src/hosted/runtime/structuredHookProgram.ts";

const manifest = ${JSON.stringify(manifest)};
const revision = ${JSON.stringify(revision)};
const repository = ${JSON.stringify(repository)};
const modelBase = \`https://huggingface.co/\${repository}/resolve/\${revision}/\`;

self.onmessage = async (event) => {
  if (event.data?.type !== "run") return;
  let stage = "start";
  const runtime = new DrowseWebLlmRuntime(webllm);
  try {
    stage = "artifacts";
    postMessage({ type: "stage", stage });
    const root = await navigator.storage.getDirectory();
    const artifactDirectory = await root.getDirectoryHandle("drowse-real-runtime-test", { create: true });
    const artifacts = [];
    for (const entry of manifest.files) {
      const response = await fetch("/model/" + entry.path, { cache: "no-store" });
      if (!response.ok) throw new Error(\`artifact fetch failed for \${entry.path}\`);
      const handle = await artifactDirectory.getFileHandle(entry.path, { create: true });
      const writable = await handle.createWritable();
      await writable.write(await response.blob());
      await writable.close();
      artifacts.push({
        manifest: { ...entry, revision, url: modelBase + entry.path },
        file: await handle.getFile(),
      });
    }
    stage = "adapter";
    postMessage({ type: "stage", stage });
    const calibrationAdapter = await navigator.gpu.requestAdapter();
    if (!calibrationAdapter) throw new Error("no WebGPU calibration adapter");
    const calibrationDevice = await calibrationAdapter.requestDevice();
    calibrationDevice.destroy();
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU load adapter");
    stage = "load";
    postMessage({ type: "stage", stage });
    const loadStarted = performance.now();
    await runtime.load({
      model: { id: ${JSON.stringify(options.id)} },
      variant: {
        id: ${JSON.stringify(options.id + "-q4f16_1-4k")},
        structuredHookProfile: manifest.structuredHookProfile,
        thinkingProfile: manifest.thinkingProfile,
        runtimeIdentity: ${JSON.stringify(runtimeIdentity)},
        requirements: {
          features: ["shader-f16"],
          limits: {
            maxStorageBufferBindingSize: 134217728,
            maxComputeWorkgroupStorageSize: 32768,
            maxComputeWorkgroupSizeX: 256,
            maxComputeInvocationsPerWorkgroup: 256,
          },
        },
      },
      contextTokens: ${options.contextTokens},
      adapter,
      artifacts,
      signal: new AbortController().signal,
      onDeviceLost() {},
    });
    const loadedEngine = runtime.requireEngine();
    if (${options.benchmark}) {
      stage = "benchmark";
      self.postMessage({ type: "stage", stage });
      const { benchmarkGeneration } = await import("/scripts/browser-generation-benchmark.ts");
      const result = await benchmarkGeneration(runtime, ${options.maxTokens});
      await runtime.unload();
      postMessage({ type: "done", result: {
        ...result, loadMs: result.startedAt - loadStarted,
        model: manifest.source, quantization: manifest.quantization,
        userAgent: navigator.userAgent, adapter: {
          vendor: adapter.info?.vendor, architecture: adapter.info?.architecture,
          device: adapter.info?.device, description: adapter.info?.description,
        },
      } });
      return;
    }
    const originalClear = loadedEngine.clearDrowseRankOneProgram.bind(loadedEngine);
    loadedEngine.clearDrowseRankOneProgram = async (...args) => {
      postMessage({ type: "trace", stage: "hook-clear-start" });
      const value = await originalClear(...args);
      postMessage({ type: "trace", stage: "hook-clear-done" });
      return value;
    };
    const controlTransitions = [];
    let controlUpdateIndex = 0;
    if (${options.thinkingControls}) {
      const originalUpdateControls = loadedEngine.updateDrowseStructuredControls.bind(loadedEngine);
      let priorControl = null;
      loadedEngine.updateDrowseStructuredControls = async (affineActive, curveActive) => {
        const active = affineActive[0];
        if (active !== priorControl) {
          controlTransitions.push({ update: controlUpdateIndex, active });
          priorControl = active;
        }
        controlUpdateIndex += 1;
        return originalUpdateControls(affineActive, curveActive);
      };
    }
    const traceCreate = (api) => {
      const originalCreate = api.create.bind(api);
      api.create = async (request) => {
        postMessage({ type: "trace", stage: "stream-create-start", request });
        const stream = await originalCreate(request);
        postMessage({ type: "trace", stage: "stream-create-done" });
        return {
          async *[Symbol.asyncIterator]() {
            let index = 0;
            for await (const chunk of stream) {
              postMessage({ type: "trace", stage: "stream-chunk", index });
              index += 1;
              yield chunk;
            }
            postMessage({ type: "trace", stage: "stream-done", chunks: index });
          },
        };
      };
    };
    traceCreate(loadedEngine.completions);
    traceCreate(loadedEngine.chat.completions);
    stage = "generate";
    postMessage({ type: "stage", stage });
    const tokens = [];
    let thinkingTokenCount = 0;
    let answerTokenCount = 0;
    const phaseTransitions = [];
    let previousPhase = null;
    const hookProgram = ${options.thinkingControls} ? createThinkingHookProgram(manifest) : null;
    const result = await runtime.streamGeneration({
      input: ${options.thinking
        ? '{ kind: "chat", messages: [{ role: "user", content: "Briefly explain why the sky is blue." }] }'
        : '{ kind: "raw", prompt: "hello" }'},
      sampling: { max_tokens: ${options.maxTokens}, seed: 1, temperature: 0 },
      thinking: ${options.thinking},
      steeringExpression: ${options.thinkingControls ? '"0.5 phase@thinking"' : "null"},
      hookProgram,
    }, (token) => {
      tokens.push(token.text);
      if (token.thinking) thinkingTokenCount += 1;
      else answerTokenCount += 1;
      if (previousPhase !== token.thinking) {
        phaseTransitions.push({ index: thinkingTokenCount + answerTokenCount - 1, thinking: token.thinking });
        previousPhase = token.thinking;
      }
      postMessage({ type: "token", text: token.text });
    });
    stage = "unload";
    postMessage({ type: "stage", stage });
    await runtime.unload();
    postMessage({
      type: "done",
      result: {
        ...result,
        emittedText: tokens.join(""),
        thinkingTokenCount,
        answerTokenCount,
        phaseTransitions,
        controlTransitions,
      },
    });
  } catch (error) {
    try { await runtime.unload(); } catch {}
    postMessage({
      type: "error",
      stage,
      message: error instanceof Error ? error.stack ?? error.message : String(error),
    });
  }
};

function createThinkingHookProgram(manifest) {
  const layerCount = manifest.layerMap.length;
  const profile = resolveStructuredHookProfile(manifest.structuredHookProfile);
  const program = compileStructuredHookProgram(
    manifest.hiddenSize,
    Array.from({ length: layerCount }, () => ({})),
    profile,
  );
  program.affineActive[0] = 1;
  const affineControls = Array.from({ length: program.affineActive.length }, () => null);
  affineControls[0] = {
    enabled: true,
    phase: { kind: "thinking_only" },
    gate: null,
  };
  program.controls = {
    affine: affineControls,
    curve: Array.from({ length: program.curveActive.length }, () => null),
  };
  return program;
}`;
}

function parseArguments(args) {
  const result = {
    benchmark: false,
    browserChannel: "chrome",
    contextTokens: 4096,
    id: null,
    library: null,
    maxTokens: 4,
    model: null,
    thinking: false,
    thinkingControls: false,
    timeoutMs: 120_000,
    webllm: null,
  };
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--benchmark") {
      result.benchmark = true;
      continue;
    }
    if (argument === "--thinking" || argument === "--thinking-controls") {
      result[argument === "--thinking" ? "thinking" : "thinkingControls"] = true;
      continue;
    }
    const value = args[index + 1];
    if (!["--browser-channel", "--context-tokens", "--id", "--library", "--max-tokens", "--model", "--timeout-ms", "--webllm"].includes(argument)) {
      throw new Error(`unexpected argument: ${argument}`);
    }
    if (!value || value.startsWith("--")) throw new Error(`missing value for ${argument}`);
    const key = argument.slice(2).replace(/-([a-z])/g, (_match, letter) => letter.toUpperCase());
    result[key] = ["contextTokens", "timeoutMs", "maxTokens"].includes(key)
      ? Number(value)
      : value;
    index += 1;
  }
  for (const key of ["id", "library", "model", "webllm"]) {
    if (!result[key]) throw new Error(`--${key} is required`);
  }
  if (!Number.isSafeInteger(result.timeoutMs) || result.timeoutMs < 1_000) throw new Error("invalid timeout");
  if (!Number.isSafeInteger(result.maxTokens) || result.maxTokens < 1) throw new Error("invalid max token count");
  if (!Number.isSafeInteger(result.contextTokens) || result.contextTokens < 1) {
    throw new Error("invalid context token count");
  }
  if (result.thinkingControls && !result.thinking) throw new Error("--thinking-controls requires --thinking");
  return result;
}

function runtimeIdentityFromManifest(manifest) {
  if (!manifest.source || typeof manifest.source !== "object") {
    throw new Error("hosted artifact manifest is missing source identity");
  }
  if (!Number.isSafeInteger(manifest.hiddenSize) || manifest.hiddenSize < 1) {
    throw new Error("hosted artifact manifest has an invalid hidden size");
  }
  if (
    !Array.isArray(manifest.layerMap) || manifest.layerMap.length === 0 ||
    manifest.layerMap.some((layer, index) => (
      !Number.isSafeInteger(layer) || layer < 0 ||
      (index > 0 && layer <= manifest.layerMap[index - 1])
    ))
  ) {
    throw new Error("hosted artifact manifest has an invalid layer map");
  }
  return {
    sourceModel: requireText(manifest.source.repository, "source repository"),
    sourceRevision: requireSha(manifest.source.revision, "source revision"),
    convertedManifestSha256: artifactForRole(manifest, "converted_manifest").sha256,
    quantization: requireText(manifest.quantization, "quantization"),
    tokenizerSha256: artifactForRole(manifest, "tokenizer").sha256,
    chatTemplateSha256: artifactForRole(manifest, "chat_template").sha256,
    modelLibrarySha256: artifactForRole(manifest, "model_library").sha256,
    runtimeAbi: requireText(manifest.runtimeAbi, "runtime ABI"),
    hookAbi: requireText(manifest.hookAbi, "hook ABI"),
    hiddenSize: manifest.hiddenSize,
    layerMap: [...manifest.layerMap],
  };
}

function artifactForRole(manifest, role) {
  if (!Array.isArray(manifest.files)) {
    throw new Error("hosted artifact manifest is missing files");
  }
  const matches = manifest.files.filter((file) => file?.role === role);
  if (matches.length !== 1) {
    throw new Error(`hosted artifact manifest must contain exactly one ${role} file`);
  }
  requireSha(matches[0].sha256, `${role} SHA-256`);
  return matches[0];
}

function requireText(value, label) {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`hosted artifact manifest has an invalid ${label}`);
  }
  return value;
}

function requireSha(value, label) {
  if (typeof value !== "string" || !/^[a-f0-9]{40}$|^[a-f0-9]{64}$/.test(value)) {
    throw new Error(`hosted artifact manifest has an invalid ${label}`);
  }
  return value;
}

async function requireFile(path, label) {
  const info = await stat(path);
  if (!info.isFile()) throw new Error(`${label} is not a regular file: ${path}`);
}

function pageHtml() {
  return "<!doctype html><meta charset=utf-8><link rel=icon href=data:,><title>Drowse app runtime test</title>";
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
