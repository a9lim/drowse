#!/usr/bin/env node

import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { createServer as createHttpServer } from "node:http";
import {
  access,
  mkdir,
  mkdtemp,
  open,
  readFile,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { dirname, extname, join, resolve } from "node:path";
import { promisify } from "node:util";
import { createServer as createViteServer } from "vite";
import { lockedModelExecutionProfiles } from "./hosted-model-profiles.mjs";
import {
  releaseInstrumentMinimums,
  validateReleaseCorpusInputs,
} from "./hosted-instrument-release-policy.mjs";
import {
  grantReleaseToolStorageQuota,
  launchReleaseToolContext,
} from "./release-tool-storage-quota.mjs";
import { selectHostedSaeLayer } from "./hosted-sae-layer.mjs";
import { verifyArtifactFile } from "./verify-artifact-file.mjs";

const exec = promisify(execFile);
const options = parseArguments(process.argv.slice(2));
if (options.help) {
  process.stdout.write(
    "Usage: npm run build:hosted:sae -- MODEL_ID MODEL_DIR MODEL.wasm WEBLLM.js CORPUS.json OUTPUT_DIR\n" +
    "       [--layer N] [--tokens N] [--features N] [--seq-len N] [--epochs N]\n" +
    "       [--batch-size N] [--device auto] [--browser-channel chrome]\n" +
    "       [--feature-metadata FEATURES.json]\n" +
    "       [--release-corpus RELEASE-CORPUS.json --release-corpus-source APPROVED-SOURCE.json]\n" +
    "       [--release-license-evidence LICENSE-EVIDENCE]\n" +
    "       Default layer: nearest 65% model depth (Python rounding through layerMap).\n" +
    "       [--timeout-ms 3600000]\n",
  );
  process.exit(0);
}

const webuiRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(webuiRoot, "..");
const runtimeLock = JSON.parse(
  await readFile(resolve(repositoryRoot, "browser-runtime/runtime-lock.json"), "utf8"),
);
const lock = runtimeLock.models.find((entry) => entry.id === options.modelId);
if (!lock) throw new Error(`unknown runtime-lock model ${options.modelId}`);
const layer = selectHostedSaeLayer(lock.layerMap, options.layer);
const featureCount = options.features ?? Math.min(lock.hiddenSize * 4, 8192);
const modelDirectory = resolve(options.modelDirectory);
const modelLibraryPath = resolve(options.modelLibrary);
const webLlmPath = resolve(options.webLlm);
const corpusPath = resolve(options.corpusPath);
const outputDirectory = resolve(options.outputDirectory);
await Promise.all([
  requireFile(modelLibraryPath, "model library"),
  requireFile(webLlmPath, "WebLLM bundle"),
  requireFile(corpusPath, "SAE corpus"),
  ...(options.featureMetadata === null
    ? []
    : [requireFile(resolve(options.featureMetadata), "SAE feature metadata")]),
  requireMissing(outputDirectory, "output directory"),
]);

const artifactsManifest = JSON.parse(
  await readFile(resolve(modelDirectory, "hosted-artifacts.json"), "utf8"),
);
const executionProfiles = lockedModelExecutionProfiles(lock, artifactsManifest);
if (
  artifactsManifest.runtimeAbi !== runtimeLock.runtimeAbi ||
  artifactsManifest.hookAbi !== runtimeLock.hookAbi ||
  artifactsManifest.quantization !== lock.quantization ||
  artifactsManifest.hiddenSize !== lock.hiddenSize ||
  JSON.stringify(artifactsManifest.layerMap) !== JSON.stringify(lock.layerMap)
) throw new Error("hosted model artifacts do not match runtime-lock identity");
const artifacts = artifactsManifest.files.map((file) => ({ ...file }));
for (const artifact of artifacts) {
  const path = artifact.role === "model_library"
    ? modelLibraryPath
    : resolve(modelDirectory, artifact.path);
  await verifyArtifactFile(
    path,
    artifact,
    `model ${artifact.path}`,
  );
}
const corpusBytes = await readFile(corpusPath);
const documents = JSON.parse(corpusBytes);
if (
  !Array.isArray(documents) || documents.length === 0 ||
  documents.some((document) => typeof document !== "string" || !document.trim())
) throw new Error("SAE corpus must be a non-empty JSON array of non-empty strings");
let releaseCorpusClosure = null;
if (options.releaseCorpus !== null) {
  const minimumFeatures = Math.min(lock.hiddenSize * 4, 8192);
  if (
    options.tokens * options.epochs < releaseInstrumentMinimums.saeTokens ||
    options.seqLen < releaseInstrumentMinimums.saeSequence ||
    featureCount < minimumFeatures
  ) {
    throw new Error(
      `release SAE requires at least ${releaseInstrumentMinimums.saeTokens} trained tokens, ` +
      `${releaseInstrumentMinimums.saeSequence}-token sequences, and ${minimumFeatures} features`,
    );
  }
  const releaseManifestPath = resolve(options.releaseCorpus);
  const sourceManifestPath = resolve(options.releaseCorpusSource);
  const licenseEvidencePath = resolve(options.releaseLicenseEvidence);
  await Promise.all([
    requireFile(releaseManifestPath, "release corpus manifest"),
    requireFile(sourceManifestPath, "release corpus source manifest"),
    requireFile(licenseEvidencePath, "release license evidence"),
  ]);
  releaseCorpusClosure = await validateReleaseCorpusInputs({
    releaseManifestPath,
    sourceManifestPath,
    licenseEvidencePath,
    kind: "sae",
    corpusBytes,
    runtimeLock,
  });
}
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

const stage = await mkdtemp(join(dirname(outputDirectory), ".drowse-sae-"));
const capturePath = resolve(stage, "activations.f32");
const captureMetadataPath = resolve(stage, "capture.json");
const packDirectory = resolve(stage, "pack");
const captureHandle = await open(capturePath, "wx");
const captureDigest = createHash("sha256");
let capturedBytes = 0;
let result;
let vite;
let server;
let browser;
let captureFailure = null;
try {
  vite = await createViteServer({
    root: webuiRoot,
    configFile: false,
    appType: "custom",
    logLevel: "silent",
    publicDir: resolve(webuiRoot, "public-hosted"),
  server: { middlewareMode: true, watch: null },
  });
  server = createHttpServer(async (request, response) => {
    for (const [name, value] of Object.entries({
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Resource-Policy": "same-origin",
    })) response.setHeader(name, value);
    try {
      const url = new URL(request.url ?? "/", "http://127.0.0.1");
      if (request.method === "GET" && url.pathname === "/") {
        return send(
          response,
          "<!doctype html><meta charset=utf-8><link rel=icon href=data:,><title>Drowse SAE build</title>",
          "text/html",
        );
      }
      if (request.method === "GET" && url.pathname === "/worker.js") {
        const transformed = await vite.transformRequest("/scripts/hosted-sae-capture.worker.ts");
        return send(response, transformed?.code ?? "worker transform failed", "text/javascript", transformed ? 200 : 500);
      }
      if (request.method === "GET" && url.pathname === "/webllm.js") {
        return send(response, await readFile(webLlmPath), "text/javascript");
      }
      if (request.method === "GET" && url.pathname === "/model") {
        const requested = url.searchParams.get("path");
        const artifact = artifacts.find((candidate) => candidate.path === requested);
        if (!artifact) return send(response, "not found", "text/plain", 404);
        const path = artifact.role === "model_library"
          ? modelLibraryPath
          : resolve(modelDirectory, artifact.path);
        return send(response, await readFile(path), contentType(path));
      }
      if (request.method === "POST" && url.pathname === "/capture") {
        const start = Number(url.searchParams.get("start"));
        if (!Number.isSafeInteger(start) || start !== capturedBytes) {
          return send(response, "capture stream offset mismatch", "text/plain", 409);
        }
        const chunk = await readBoundedBody(request, 256 * 1024 * 1024);
        await writeAll(captureHandle, chunk, capturedBytes);
        captureDigest.update(chunk);
        capturedBytes += chunk.length;
        return send(response, "ok", "text/plain");
      }
      vite.middlewares(request, response);
    } catch (error) {
      send(response, error instanceof Error ? error.stack ?? error.message : String(error), "text/plain", 500);
    }
  });
  await new Promise((resolvePromise, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolvePromise);
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("failed to bind SAE capture server");
  const origin = `http://127.0.0.1:${address.port}`;
  browser = await launchReleaseToolContext({
    headless: true,
    channel: options.browserChannel,
    args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--disable-gpu-sandbox"],
  });
  const page = browser.page;
  page.setDefaultTimeout(options.timeoutMs);
  page.on("console", (message) => process.stderr.write(`browser ${message.type()}: ${message.text()}\n`));
  await page.goto(origin, { waitUntil: "load" });
  await grantReleaseToolStorageQuota(page, origin, artifacts);
  result = await page.evaluate(async ({ timeoutMs, payload }) => {
    const worker = new Worker("/worker.js", { type: "module" });
    return await new Promise((resolvePromise, reject) => {
      const timeout = setTimeout(() => {
        worker.terminate();
        reject(new Error(`SAE capture timed out after ${timeoutMs} ms`));
      }, timeoutMs);
      worker.onerror = (event) => {
        clearTimeout(timeout);
        worker.terminate();
        reject(new Error(event.message || "SAE capture worker failed"));
      };
      worker.onmessage = (event) => {
        if (event.data.type === "progress") {
          console.info(`SAE capture ${event.data.completed}/${event.data.total}`);
          return;
        }
        clearTimeout(timeout);
        worker.terminate();
        if (event.data.type === "error") reject(new Error(event.data.message));
        else resolvePromise(event.data.result);
      };
      worker.postMessage(payload);
    });
  }, {
    timeoutMs: options.timeoutMs,
    payload: {
      type: "run",
      modelId: options.modelId,
      artifacts,
      runtimeIdentity,
      ...executionProfiles,
      documents,
      layer,
      tokenLimit: options.tokens,
      sequenceLength: options.seqLen,
      contextTokens: Math.max(...lock.contextProfiles),
    },
  });
} catch (error) {
  captureFailure = error;
} finally {
  await browser?.close();
  await vite?.close();
  if (server) await new Promise((resolvePromise) => server.close(resolvePromise));
  await captureHandle.close();
}
if (captureFailure !== null) {
  await rm(stage, { recursive: true, force: true });
  throw captureFailure;
}

try {
  const localCaptureSha256 = captureDigest.digest("hex");
  if (
    result.bytes !== capturedBytes || result.bytes !== options.tokens * lock.hiddenSize * 4 ||
    result.activationSha256 !== localCaptureSha256
  ) throw new Error("SAE activation capture size or digest changed during transfer");
  await writeFile(captureMetadataPath, `${JSON.stringify({
    schema_version: 1,
    model_id: lock.sourceRepository,
    model_source_fingerprint: lock.sourceRevision,
    runtime_identity_sha256: result.runtimeIdentitySha256,
    layer,
    hidden_size: lock.hiddenSize,
    rows: options.tokens,
    seq_len: options.seqLen,
    corpus_spec: `browser-q4-json-v1:${corpusPath.split("/").at(-1)}`,
    corpus_sha256: sha256(corpusBytes),
    capture_plan_sha256: result.capturePlanSha256,
    activation_sha256: localCaptureSha256,
  }, null, 2)}\n`);
  const trainer = resolve(repositoryRoot, "browser-runtime/train_hosted_sae.py");
  const trained = await exec(options.python, [
    trainer,
    capturePath,
    captureMetadataPath,
    packDirectory,
    "--features", String(featureCount),
    "--epochs", String(options.epochs),
    "--batch-size", String(options.batchSize),
    "--device", options.device,
  ], { maxBuffer: 4 * 1024 * 1024, timeout: options.timeoutMs });
  if (options.featureMetadata !== null) {
    await writeFile(
      resolve(packDirectory, "packs/sae/features.json"),
      await readFile(resolve(options.featureMetadata)),
      { flag: "wx" },
    );
  }
  if (releaseCorpusClosure !== null) {
    await Promise.all([
      writeFile(
        resolve(packDirectory, "packs/sae/release-corpus.json"),
        releaseCorpusClosure.releaseBytes,
        { flag: "wx" },
      ),
      writeFile(
        resolve(packDirectory, "packs/sae/release-corpus-source.json"),
        releaseCorpusClosure.sourceBytes,
        { flag: "wx" },
      ),
      writeFile(
        resolve(packDirectory, "packs/sae/release-license-evidence"),
        releaseCorpusClosure.licenseEvidenceBytes,
        { flag: "wx" },
      ),
    ]);
  }
  await exec(process.execPath, [resolve(import.meta.dirname, "assemble-hosted-pack.mjs"), packDirectory]);
  await exec(process.execPath, [
    resolve(import.meta.dirname, "validate-hosted-instrument-pack.mjs"),
    options.modelId,
    "sae",
    packDirectory,
    ...(releaseCorpusClosure === null ? [] : ["--release"]),
  ]);
  await rename(packDirectory, outputDirectory);
  process.stdout.write(`${JSON.stringify({
    modelId: options.modelId,
    outputDirectory,
    layer,
    rows: options.tokens,
    hiddenSize: lock.hiddenSize,
    features: featureCount,
    releaseCandidate: releaseCorpusClosure !== null,
    runtimeIdentitySha256: result.runtimeIdentitySha256,
    activationSha256: result.activationSha256,
    training: JSON.parse(trained.stdout),
  }, null, 2)}\n`);
} finally {
  await rm(stage, { recursive: true, force: true });
}

function parseArguments(args) {
  if (args.includes("--help")) return { help: true };
  const positional = [];
  const result = {
    layer: null,
    tokens: 50_000,
    features: null,
    seqLen: 256,
    epochs: 1,
    batchSize: 256,
    device: "auto",
    python: "python3",
    browserChannel: "chrome",
    timeoutMs: 3_600_000,
    featureMetadata: null,
    releaseCorpus: null,
    releaseCorpusSource: null,
    releaseLicenseEvidence: null,
  };
  const fields = {
    "--layer": "layer",
    "--tokens": "tokens",
    "--features": "features",
    "--seq-len": "seqLen",
    "--epochs": "epochs",
    "--batch-size": "batchSize",
    "--device": "device",
    "--python": "python",
    "--browser-channel": "browserChannel",
    "--timeout-ms": "timeoutMs",
    "--feature-metadata": "featureMetadata",
    "--release-corpus": "releaseCorpus",
    "--release-corpus-source": "releaseCorpusSource",
    "--release-license-evidence": "releaseLicenseEvidence",
  };
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (!argument.startsWith("--")) {
      positional.push(argument);
      continue;
    }
    const field = fields[argument];
    if (!field) throw new Error(`unexpected argument ${argument}`);
    const value = args[++index];
    if (!value || value.startsWith("--")) throw new Error(`missing value for ${argument}`);
    result[field] = [
      "device", "python", "browserChannel", "featureMetadata",
      "releaseCorpus", "releaseCorpusSource",
      "releaseLicenseEvidence",
    ].includes(field)
      ? value
      : Number(value);
  }
  if (positional.length !== 6) {
    throw new Error("expected MODEL_ID MODEL_DIR MODEL.wasm WEBLLM.js CORPUS.json OUTPUT_DIR");
  }
  for (const field of ["tokens", "seqLen", "epochs", "batchSize", "timeoutMs"]) {
    if (!Number.isSafeInteger(result[field]) || result[field] <= 0) {
      throw new Error(`${field} must be a positive integer`);
    }
  }
  for (const field of ["layer", "features"]) {
    if (result[field] !== null && (!Number.isSafeInteger(result[field]) || result[field] < 0)) {
      throw new Error(`${field} must be a non-negative integer`);
    }
  }
  if (result.features === 0 || result.timeoutMs < 10_000) {
    throw new Error("features must be positive and timeout must be at least 10000 ms");
  }
  const releaseInputs = [
    result.releaseCorpus,
    result.releaseCorpusSource,
    result.releaseLicenseEvidence,
  ];
  if (releaseInputs.some((value) => value !== null) && releaseInputs.some((value) => value === null)) {
    throw new Error(
      "--release-corpus, --release-corpus-source, and --release-license-evidence must be supplied together",
    );
  }
  [
    result.modelId,
    result.modelDirectory,
    result.modelLibrary,
    result.webLlm,
    result.corpusPath,
    result.outputDirectory,
  ] = positional;
  return result;
}

async function requireFile(path, label) {
  const info = await stat(path);
  if (!info.isFile()) throw new Error(`${label} is not a regular file: ${path}`);
}

async function requireMissing(path, label) {
  try {
    await access(path);
  } catch (error) {
    if (error?.code === "ENOENT") return;
    throw error;
  }
  throw new Error(`${label} already exists: ${path}`);
}

async function readBoundedBody(request, limit) {
  const chunks = [];
  let bytes = 0;
  for await (const chunk of request) {
    bytes += chunk.byteLength;
    if (bytes > limit) throw new Error("SAE capture chunk exceeded the release-tool limit");
    chunks.push(chunk);
  }
  return Buffer.concat(chunks, bytes);
}

async function writeAll(handle, bytes, position) {
  let offset = 0;
  while (offset < bytes.length) {
    const result = await handle.write(bytes, offset, bytes.length - offset, position + offset);
    if (result.bytesWritten <= 0) throw new Error("SAE capture file write made no progress");
    offset += result.bytesWritten;
  }
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

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
