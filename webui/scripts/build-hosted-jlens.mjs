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
import { selectHostedJlensLayers } from "./hosted-instrument-layers.mjs";
import {
  releaseInstrumentMinimums,
  validateReleaseCorpusInputs,
} from "./hosted-instrument-release-policy.mjs";
import {
  grantReleaseToolStorageQuota,
  launchReleaseToolContext,
} from "./release-tool-storage-quota.mjs";
import { verifyArtifactFile } from "./verify-artifact-file.mjs";

const exec = promisify(execFile);
const options = parseArguments(process.argv.slice(2));
if (options.help) {
  process.stdout.write(
    "Usage: npm run build:hosted:jlens -- MODEL_ID MODEL_DIR MODEL.wasm WEBLLM.js CORPUS.json WORDS.json OUTPUT_DIR\n" +
    "       [--prompts N] [--seq-len N] [--dim-batch N] [--device mps]\n" +
    "       [--source-layers 0,1] [--python python3] [--browser-channel chrome]\n" +
    "       [--release-corpus RELEASE-CORPUS.json --release-corpus-source APPROVED-SOURCE.json]\n" +
    "       [--release-license-evidence LICENSE-EVIDENCE]\n" +
    "       [--timeout-ms 21600000]\n",
  );
  process.exit(0);
}

const webuiRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(webuiRoot, "..");
const pythonExecutable = options.python.includes("/")
  ? resolve(options.python)
  : options.python;
const runtimeLock = JSON.parse(
  await readFile(resolve(repositoryRoot, "browser-runtime/runtime-lock.json"), "utf8"),
);
const lock = runtimeLock.models.find((entry) => entry.id === options.modelId);
if (!lock) throw new Error(`unknown runtime-lock model ${options.modelId}`);
const modelDirectory = resolve(options.modelDirectory);
const modelLibraryPath = resolve(options.modelLibrary);
const webLlmPath = resolve(options.webLlm);
const corpusPath = resolve(options.corpusPath);
const wordsPath = resolve(options.wordsPath);
const outputDirectory = resolve(options.outputDirectory);
await Promise.all([
  requireFile(modelLibraryPath, "model library"),
  requireFile(webLlmPath, "WebLLM bundle"),
  requireFile(corpusPath, "J-lens corpus"),
  requireFile(wordsPath, "J-lens words"),
  requireMissing(outputDirectory, "output directory"),
]);
const corpusBytes = await readFile(corpusPath);
const documents = JSON.parse(corpusBytes);
if (
  !Array.isArray(documents) || documents.length < options.prompts ||
  documents.some((document) => typeof document !== "string" || !document.trim())
) throw new Error("J-lens corpus must contain enough non-empty strings");
const words = JSON.parse(await readFile(wordsPath, "utf8"));
const wordsBytes = await readFile(wordsPath);
if (
  !Array.isArray(words) || words.length === 0 ||
  words.some((word) => typeof word !== "string" || !word || word.trim() !== word) ||
  new Set(words).size !== words.length
) throw new Error("J-lens words must be a non-empty array of unique trimmed strings");
let releaseCorpusClosure = null;
if (options.releaseCorpus !== null) {
  if (
    options.prompts < releaseInstrumentMinimums.jlensPrompts ||
    options.seqLen < releaseInstrumentMinimums.jlensSequence
  ) {
    throw new Error(
      `release J-lens requires at least ` +
      `${releaseInstrumentMinimums.jlensPrompts} prompts, and ` +
      `${releaseInstrumentMinimums.jlensSequence}-token sequences`,
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
    kind: "jlens",
    corpusBytes,
    wordsBytes,
    runtimeLock,
  });
}
const artifactsManifest = JSON.parse(
  await readFile(resolve(modelDirectory, "hosted-artifacts.json"), "utf8"),
);
const executionProfiles = lockedModelExecutionProfiles(lock, artifactsManifest);
const selectedSourceLayers = options.sourceLayers === null
  ? selectHostedJlensLayers(lock.layerMap, lock.hiddenSize)
  : options.sourceLayers.split(",").map(Number);
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

const stage = await mkdtemp(join(dirname(outputDirectory), ".drowse-jlens-"));
const captureDirectory = resolve(stage, "capture");
const capturePath = resolve(captureDirectory, "residuals.f32");
const packDirectory = resolve(stage, "pack");
await mkdir(captureDirectory);
const residualHandle = await open(capturePath, "wx");
const residualDigest = createHash("sha256");
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
        return send(response, "<!doctype html><meta charset=utf-8><link rel=icon href=data:,>", "text/html");
      }
      if (request.method === "GET" && url.pathname === "/worker.js") {
        const transformed = await vite.transformRequest("/scripts/hosted-jlens-capture.worker.ts");
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
        const chunk = await readBoundedBody(request, 1024 * 1024 * 1024);
        await writeAll(residualHandle, chunk, capturedBytes);
        residualDigest.update(chunk);
        capturedBytes += chunk.length;
        return send(response, "ok", "text/plain");
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
  if (!address || typeof address === "string") throw new Error("failed to bind J-lens capture server");
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
        reject(new Error(`J-lens capture timed out after ${timeoutMs} ms`));
      }, timeoutMs);
      worker.onerror = (event) => {
        clearTimeout(timeout);
        worker.terminate();
        reject(new Error(event.message || "J-lens capture worker failed"));
      };
      worker.onmessage = (event) => {
        if (event.data.type === "progress") {
          console.info(`J-lens capture ${event.data.completed}/${event.data.total}`);
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
      promptLimit: options.prompts,
      sequenceLength: options.seqLen,
      skipFirst: 16,
      contextTokens: Math.max(...lock.contextProfiles),
    },
  });
} catch (error) {
  captureFailure = error;
} finally {
  await browser?.close();
  await vite?.close();
  if (server) await new Promise((resolvePromise) => server.close(resolvePromise));
  await residualHandle.close();
}
if (captureFailure !== null) {
  await rm(stage, { recursive: true, force: true });
  throw captureFailure;
}

try {
  const residualSha256 = residualDigest.digest("hex");
  if (
    result.bytes !== capturedBytes ||
    result.residualSha256 !== residualSha256 ||
    result.rows.length !== options.prompts
  ) throw new Error("J-lens capture size, digest, or row count changed during transfer");
  await writeFile(resolve(captureDirectory, "capture.json"), `${JSON.stringify({
    schema_version: 1,
    model_id: options.modelId,
    source_model_id: lock.sourceRepository,
    model_source_fingerprint: lock.sourceRevision,
    runtime_identity_sha256: result.runtimeIdentitySha256,
    layer_map: lock.layerMap,
    hidden_size: lock.hiddenSize,
    seq_len: options.seqLen,
    skip_first_positions: 16,
    corpus_spec: `browser-q4-corrected-json-v1:${corpusPath.split("/").at(-1)}`,
    raw_corpus_sha256: sha256(corpusBytes),
    capture_plan_sha256: result.capturePlanSha256,
    residual_sha256: result.residualSha256,
    rows: result.rows,
  }, null, 2)}\n`);
  const fitter = resolve(repositoryRoot, "browser-runtime/fit_hosted_jlens.py");
  const fitArguments = [
    fitter,
    modelDirectory,
    captureDirectory,
    wordsPath,
    packDirectory,
    "--device", options.device,
    "--dim-batch", String(options.dimBatch),
  ];
  fitArguments.push("--source-layers", selectedSourceLayers.join(","));
  const fitted = await exec(pythonExecutable, fitArguments, {
    cwd: repositoryRoot,
    env: { ...process.env, PYTHONPATH: repositoryRoot },
    maxBuffer: 4 * 1024 * 1024,
    timeout: options.timeoutMs,
  });
  if (releaseCorpusClosure !== null) {
    await Promise.all([
      writeFile(
        resolve(packDirectory, "packs/jlens/release-corpus.json"),
        releaseCorpusClosure.releaseBytes,
        { flag: "wx" },
      ),
      writeFile(
        resolve(packDirectory, "packs/jlens/release-corpus-source.json"),
        releaseCorpusClosure.sourceBytes,
        { flag: "wx" },
      ),
      writeFile(
        resolve(packDirectory, "packs/jlens/release-license-evidence"),
        releaseCorpusClosure.licenseEvidenceBytes,
        { flag: "wx" },
      ),
    ]);
  }
  await exec(process.execPath, [resolve(import.meta.dirname, "assemble-hosted-pack.mjs"), packDirectory]);
  const validated = await exec(process.execPath, [
    resolve(import.meta.dirname, "validate-hosted-instrument-pack.mjs"),
    options.modelId,
    "jlens",
    packDirectory,
    ...(releaseCorpusClosure === null ? [] : ["--release"]),
  ], { maxBuffer: 4 * 1024 * 1024, timeout: options.timeoutMs });
  await rename(packDirectory, outputDirectory);
  process.stdout.write(`${JSON.stringify({
    modelId: options.modelId,
    outputDirectory,
    runtimeIdentitySha256: result.runtimeIdentitySha256,
    prompts: options.prompts,
    seqLen: options.seqLen,
    releaseCandidate: releaseCorpusClosure !== null,
    residualSha256: result.residualSha256,
    fit: JSON.parse(fitted.stdout),
    validation: JSON.parse(validated.stdout),
  }, null, 2)}\n`);
} finally {
  await rm(stage, { recursive: true, force: true });
}

function parseArguments(args) {
  if (args.includes("--help")) return { help: true };
  const positional = [];
  const result = {
    prompts: 100,
    seqLen: 128,
    dimBatch: 64,
    device: "mps",
    sourceLayers: null,
    python: "python3",
    browserChannel: "chrome",
    timeoutMs: 21_600_000,
    releaseCorpus: null,
    releaseCorpusSource: null,
    releaseLicenseEvidence: null,
  };
  const fields = {
    "--prompts": "prompts",
    "--seq-len": "seqLen",
    "--dim-batch": "dimBatch",
    "--device": "device",
    "--source-layers": "sourceLayers",
    "--python": "python",
    "--browser-channel": "browserChannel",
    "--timeout-ms": "timeoutMs",
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
      "device", "sourceLayers", "python", "browserChannel",
      "releaseCorpus", "releaseCorpusSource",
      "releaseLicenseEvidence",
    ].includes(field)
      ? value
      : Number(value);
  }
  if (positional.length !== 7) {
    throw new Error("expected MODEL_ID MODEL_DIR MODEL.wasm WEBLLM.js CORPUS.json WORDS.json OUTPUT_DIR");
  }
  for (const field of ["prompts", "seqLen", "dimBatch", "timeoutMs"]) {
    if (!Number.isSafeInteger(result[field]) || result[field] <= 0) {
      throw new Error(`${field} must be a positive integer`);
    }
  }
  if (result.seqLen <= 17 || result.timeoutMs < 10_000) {
    throw new Error("seqLen must exceed 17 and timeout must be at least 10000 ms");
  }
  if (result.sourceLayers !== null && !/^\d+(,\d+)*$/.test(result.sourceLayers)) {
    throw new Error("source layers must be a comma-separated integer list");
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
    result.wordsPath,
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
    if (bytes > limit) throw new Error("J-lens capture chunk exceeded the release-tool limit");
    chunks.push(chunk);
  }
  return Buffer.concat(chunks, bytes);
}

async function writeAll(handle, bytes, position) {
  let offset = 0;
  while (offset < bytes.length) {
    const result = await handle.write(bytes, offset, bytes.length - offset, position + offset);
    if (result.bytesWritten <= 0) throw new Error("J-lens capture file write made no progress");
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
