#!/usr/bin/env node

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdtemp, readFile, stat, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { pipeline } from "node:stream/promises";

const [modelPath, libraryPath, portText = "4198", ...options] = process.argv.slice(2);
if (!modelPath || !libraryPath) throw new Error("Usage: base-model-browser-harness.mjs MODEL_DIRECTORY LIBRARY [PORT] [--lifecycle] [--repeat N] [--branch-rounds N] [--diagnostic] [--fp32-replay]");
let lifecycle = false, diagnostic = false, fp32Replay = false, promptRepeats = 0, branchRounds = 1;
for (let index = 0; index < options.length; index++) {
  if (options[index] === "--lifecycle") lifecycle = true;
  else if (options[index] === "--diagnostic") diagnostic = true;
  else if (options[index] === "--fp32-replay") fp32Replay = true;
  else if (options[index] === "--repeat") {
    promptRepeats = Number(options[++index]);
    if (!Number.isSafeInteger(promptRepeats) || promptRepeats < 0 || promptRepeats > 4096) throw new Error("Invalid prompt repetition count");
  } else if (options[index] === "--branch-rounds") {
    branchRounds = Number(options[++index]);
    if (!Number.isSafeInteger(branchRounds) || branchRounds < 1 || branchRounds > 32) throw new Error("Invalid branch round count");
  } else throw new Error("Unknown harness option");
}
const directory = resolve(modelPath);
const library = resolve(libraryPath);
const config = JSON.parse(await readFile(resolve(directory, "mlc-chat-config.json"), "utf8"));
if (fp32Replay && !/f32(?:_|$)/.test(config.quantization)) throw new Error("FP32 replay policy requires a float32-compute model");
const manifestPath = await buildManifestPath(directory);
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
const reports = await mkdtemp(resolve(tmpdir(), "drowse-base-browser-"));
const modelBase = `/model/resolve/${manifest.source.revision}/`;
const routes = new Map([
  ["/model-lib.wasm", library],
  ["/webllm.js", resolve(import.meta.dirname, "../node_modules/@drowse/web-llm/lib/index.js")],
  ["/capture-parity.mjs", resolve(import.meta.dirname, "base-capture-parity.mjs")],
]);
const assets = [];
for (const item of manifest.files) {
  if (!/^[a-zA-Z0-9._-]+$/.test(item.path)) throw new Error("Unsafe model artifact path");
  const path = resolve(directory, item.path);
  if ((await stat(path)).size !== item.bytes || await digest(path) !== item.sha256) {
    throw new Error(`Model closure failed verification: ${item.path}`);
  }
  routes.set(`${modelBase}${item.path}`, path);
  assets.push({ url: `${modelBase}${item.path}`, sha256: item.sha256 });
}
const librarySha256 = await digest(library);
assets.push({ url: "/model-lib.wasm", sha256: librarySha256 });
const evidence = {
  schemaVersion: 1, source: manifest.source, modelBuildSha256: await digest(manifestPath),
  librarySha256, webllmSha256: await digest(routes.get("/webllm.js")),
  architecture: manifest.architecture, quantization: manifest.quantization,
  contextTokens: config.context_window_size,
};
let sequence = 0;
const server = createServer(async (request, response) => {
  response.setHeader("Cache-Control", "no-store");
  response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  response.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  try {
    const pathname = new URL(request.url, "http://127.0.0.1").pathname;
    if (request.method === "POST" && pathname === "/report") {
      const chunks = [];
      let bytes = 0;
      for await (const chunk of request) {
        bytes += chunk.length;
        if (bytes > 64 * 1024 * 1024) throw new Error("Report exceeds limit");
        chunks.push(chunk);
      }
      const report = JSON.parse(Buffer.concat(chunks).toString("utf8"));
      const file = resolve(reports, `run-${++sequence}.json`);
      await writeFile(file, JSON.stringify({ ...report, artifact: evidence }, null, 2) + "\n", { flag: "wx" });
      response.setHeader("Content-Type", "application/json");
      response.end(JSON.stringify({ file }));
      process.stdout.write(`Report: ${file}\n`);
      return;
    }
    if (request.method !== "GET") { response.writeHead(405).end(); return; }
    if (pathname === "/") {
      response.setHeader("Content-Type", "text/html; charset=utf-8");
      response.end(page());
      return;
    }
    if (pathname === "/manifest") {
      response.setHeader("Content-Type", "application/json");
      response.end(JSON.stringify({ assets, evidence, modelBase,
        promptPrefixTokenIds: config.drowse_completion_prefix_token_ids ?? config.polythetic_completion_prefix_token_ids ?? config.saklas_completion_prefix_token_ids ?? [],
        lifecycle, contextTokens: config.context_window_size,
        eosTokenIds: config.conv_template.stop_token_ids }));
      return;
    }
    const path = routes.get(pathname);
    if (!path) { response.writeHead(404).end("Not found"); return; }
    response.setHeader("Content-Type", /\.m?js$/.test(path) ? "text/javascript" : path.endsWith(".wasm") ? "application/wasm" : "application/octet-stream");
    response.setHeader("Content-Length", (await stat(path)).size);
    await pipeline(createReadStream(path), response);
  } catch (error) {
    if (!response.headersSent) response.writeHead(500);
    response.end(String(error));
  }
});
server.listen(Number(portText), "127.0.0.1", () => {
  process.stdout.write(`Local candidate harness: http://127.0.0.1:${server.address().port}\nReports: ${reports}\n`);
});

async function digest(path) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

async function buildManifestPath(directory) {
  for (const slug of ["drowse", "polythetic", "saklas"]) {
    const path = resolve(directory, `${slug}-build.json`);
    try { await stat(path); return path; } catch (error) { if (error.code !== "ENOENT") throw error; }
  }
  throw new Error("Model build manifest is missing");
}

function page() {
  return `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Base-model local candidate test</title>
<style>body{font:16px system-ui;max-width:70rem;margin:2rem auto;padding:1rem;color:#ddd;background:#151515}button,textarea{font:inherit;padding:.6rem}textarea{display:block;width:95%;margin:1rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}button:focus-visible{outline:3px solid #9cf}#output{border:1px solid #888;padding:1rem}</style>
<h1>Base-model local candidate test</h1><p>Real local weights; no release certification. No remote inference or persistent model cache.</p>
<label for="prompt">Test prefix</label><textarea id="prompt" readonly>I love marmots because${" marmot".repeat(promptRepeats)}</textarea>
<button id="run">Run generation and capture</button> <button id="stop" disabled>Stop</button>
<p id="status" role="status">Ready</p><pre id="output" aria-label="Generated completion"></pre>
<pre id="report" aria-label="Test results"></pre>
<script type="module">
import { MLCEngine } from "/webllm.js";
import { compareCaptures } from "/capture-parity.mjs";
const run = document.querySelector("#run"), stop = document.querySelector("#stop");
const status = document.querySelector("#status"), output = document.querySelector("#output"), reportView = document.querySelector("#report");
let engine;
stop.onclick = () => engine?.interruptGenerate();
run.onclick = async () => {
  run.disabled = true; stop.disabled = false; output.textContent = "";
  const report = { startedAt: new Date().toISOString(), userAgent: navigator.userAgent, prompt: document.querySelector("#prompt").value, steps: [] };
  let firstLogits, generatedIds = [], forcedEos = null;
  try {
    if (!navigator.gpu) throw new Error("WebGPU unavailable");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info?.isFallbackAdapter) throw new Error("Hardware WebGPU adapter unavailable");
    report.adapter = { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture, device: adapter.info?.device, description: adapter.info?.description, features: [...adapter.features], maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize };
    status.textContent = "Verifying local model files";
    const { assets, modelBase, promptPrefixTokenIds, lifecycle, contextTokens, eosTokenIds } = await (await fetch("/manifest")).json();
    report.promptPrefixTokenIds = promptPrefixTokenIds;
    const verified = new Map();
    for (const asset of assets) {
      const url = new URL(asset.url, location.origin).href;
      const response = await fetch(url);
      if (!response.ok) throw new Error("Artifact read failed: " + asset.url);
      const bytes = await response.arrayBuffer();
      const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map(v => v.toString(16).padStart(2, "0")).join("");
      if (hash !== asset.sha256) throw new Error("Artifact integrity failure: " + asset.url);
      verified.set(url, bytes);
    }
    const artifactCache = {
      async fetchWithCache(url, type) {
        const data = verified.get(url);
        if (!data) throw new Error("Unverified artifact: " + url);
        return type === "json" ? JSON.parse(new TextDecoder().decode(data)) : data.slice(0);
      },
      async addToCache(url) { if (!verified.has(url)) throw new Error("Unverified artifact: " + url); },
      async hasAllKeys(keys) { return keys.every(key => verified.has(key)); },
      async deleteInCache() { throw new Error("Read-only local artifacts"); }
    };
    const processor = {
      processLogits(logits) {
        if (!firstLogits) firstLogits = Array.from(logits);
        if (forcedEos !== null) { logits.fill(-1e30); logits[forcedEos] = 0; }
        return logits;
      },
      processSampledToken(token) { generatedIds.push(token); },
      resetState() { firstLogits = undefined; generatedIds = []; }
    };
    engine = new MLCEngine({
      appConfig: { artifactCache, gpuAdapter: adapter, model_list: [{ model: new URL(modelBase, location.origin).href, model_id: "local-base", model_lib: new URL("/model-lib.wasm", location.origin).href }] },
      logitProcessorRegistry: new Map([["local-base", processor]]),
      initProgressCallback: progress => { status.textContent = progress.text; }, logLevel: "WARN"
    });
    await engine.reload("local-base");
    report.profile = await engine.getDrowseStructuredHookProfile();
    report.inputIds = [...promptPrefixTokenIds, ...await engine.tokenizeDrowseText(report.prompt)];
    const options = { prompt: report.prompt, max_tokens: 24, temperature: 0, top_p: 1, seed: 11 };
    status.textContent = "Generating";
    const stream = await engine.completions.create({ ...options, stream: true });
    for await (const chunk of stream) output.textContent += chunk.choices[0]?.text ?? "";
    report.completion = output.textContent;
    report.generatedIds = [...generatedIds];
    report.usage = await engine.runtimeStatsText();
    report.firstLogits = firstLogits;
    if (!firstLogits?.length || firstLogits.some(v => !Number.isFinite(v))) throw new Error("Non-finite or missing model logits");
    report.steps.push("finite generation logits");
    status.textContent = "Capturing residuals";
    const positions = capturePositions(report.inputIds.length);
    const capture = await engine.captureDrowseResiduals(report.inputIds, positions);
    report.capture = { ...capture, values: Array.from(capture.values) };
    if (capture.values.some(v => !Number.isFinite(v))) throw new Error("Non-finite residual capture");
    report.steps.push("finite post-block residuals");
    const replay = await engine.captureDrowseResiduals(report.inputIds, positions);
    if (capture.values.length !== replay.values.length || replay.values.some(v => !Number.isFinite(v))) {
      throw new Error("Repeated capture has invalid residuals");
    }
    report.captureReplayMaxAbs = 0;
    for (let index = 0; index < capture.values.length; index++) {
      report.captureReplayMaxAbs = Math.max(report.captureReplayMaxAbs, Math.abs(capture.values[index] - replay.values[index]));
    }
    report.captureReplayComparison = compareCaptures(capture, replay, ${JSON.stringify(fp32Replay ? "fp32" : "strict")});
    if (!report.captureReplayComparison.passed) {
      report.repeatCapture = { ...replay, values: Array.from(replay.values) };
      if (!${diagnostic}) throw new Error("Repeated capture changed residuals");
      (report.diagnosticFailures ??= []).push("repeat capture exceeds declared numerical bounds");
    } else report.steps.push("repeat capture parity");
    status.textContent = "Checking generation after capture";
    const repeated = await engine.completions.create(options);
    report.repeatCompletion = repeated.choices[0].text;
    report.repeatGeneratedIds = [...generatedIds];
    if (report.repeatCompletion !== report.completion || JSON.stringify(report.generatedIds) !== JSON.stringify(report.repeatGeneratedIds)) {
      throw new Error("Greedy generation changed after capture");
    }
    report.steps.push("generation after capture parity");
    if (lifecycle) {
      status.textContent = "Checking cancellation and reset";
      const cancelled = await engine.completions.create({ ...options, max_tokens: 64, stream: true });
      report.cancelledCompletion = "";
      let interrupted = false;
      for await (const chunk of cancelled) {
        report.cancelledCompletion += chunk.choices[0]?.text ?? "";
        if (!interrupted && generatedIds.length >= 3) {
          interrupted = true;
          await engine.interruptGenerate();
        }
      }
      report.cancelledGeneratedIds = [...generatedIds];
      if (!interrupted || generatedIds.length >= 64 || !report.cancelledCompletion) {
        throw new Error("Cancellation did not retain a bounded partial completion");
      }
      const reset = await engine.completions.create(options);
      if (reset.choices[0].text !== report.completion || JSON.stringify(generatedIds) !== JSON.stringify(report.generatedIds)) {
        throw new Error("Generation changed after cancellation");
      }
      report.steps.push("cancellation and fresh generation parity");
      status.textContent = "Checking overlong prompt rejection";
      const overlong = report.prompt + " marmot".repeat(contextTokens);
      report.overlongPromptTokens = promptPrefixTokenIds.length + (await engine.tokenizeDrowseText(overlong)).length;
      if (report.overlongPromptTokens <= contextTokens) throw new Error("Context fixture is not overlong");
      try {
        await engine.completions.create({ ...options, prompt: overlong, max_tokens: 1 });
      } catch (contextError) {
        if (contextError?.constructor?.name !== "ContextWindowSizeExceededError") throw contextError;
        report.contextGuardError = contextError.message;
      }
      if (!report.contextGuardError) throw new Error("Overlong prompt was accepted");
      const afterRejection = await engine.completions.create(options);
      if (afterRejection.choices[0].text !== report.completion || JSON.stringify(generatedIds) !== JSON.stringify(report.generatedIds)) {
        throw new Error("Generation changed after context rejection");
      }
      report.steps.push("overlong context rejection and recovery parity");
      status.textContent = "Checking branch isolation and explicit state reset";
      report.branchRounds = [];
      for (let round = 0; round < ${branchRounds}; round++) {
      status.textContent = "Checking branch " + (round + 1) + "/${branchRounds}";
      const branchIds = [...promptPrefixTokenIds, ...await engine.tokenizeDrowseText(report.prompt + " near the river".repeat(round + 1))];
      await engine.captureDrowseResiduals(branchIds, [branchIds.length - 1]);
      const branchReplay = await engine.captureDrowseResiduals(report.inputIds, positions);
      report.branchReplayMaxAbs = 0;
      for (let index = 0; index < capture.values.length; index++) {
        report.branchReplayMaxAbs = Math.max(report.branchReplayMaxAbs, Math.abs(capture.values[index] - branchReplay.values[index]));
      }
      report.branchRounds.push({round, branchTokens: branchIds.length, maxAbs: report.branchReplayMaxAbs,
        comparison: compareCaptures(capture, branchReplay, ${JSON.stringify(fp32Replay ? "fp32" : "strict")})});
      if (!report.branchRounds.at(-1).comparison.passed) {
        report.branchReplayCapture = { ...branchReplay, values: Array.from(branchReplay.values) };
        const another = await engine.captureDrowseResiduals(report.inputIds, positions);
        report.branchRecoveryCapture = { ...another, values: Array.from(another.values) };
        await engine.resetChat();
        const explicit = await engine.captureDrowseResiduals(report.inputIds, positions);
        report.explicitResetCapture = { ...explicit, values: Array.from(explicit.values) };
        if (!${diagnostic}) throw new Error("A different branch changed the replayed residuals");
        (report.diagnosticFailures ??= []).push("branch " + round + " exceeds declared numerical bounds");
      }
      }
      await engine.resetChat();
      const afterReset = await engine.completions.create(options);
      if (afterReset.choices[0].text !== report.completion || JSON.stringify(generatedIds) !== JSON.stringify(report.generatedIds)) {
        throw new Error("Explicit state reset changed greedy generation");
      }
      report.steps.push("branch isolation and explicit state reset parity");
      status.textContent = "Checking EOS handling";
      report.forcedEosChecks = [];
      for (const eos of eosTokenIds) {
        forcedEos = eos;
        const stopped = await engine.completions.create(options);
        const choice = stopped.choices[0];
        if (choice.finish_reason !== "stop" || choice.text !== "" || generatedIds.length > 1) {
          throw new Error("EOS did not stop a completion cleanly: " + eos);
        }
        report.forcedEosChecks.push({token: eos, finishReason: choice.finish_reason, completion: choice.text});
      }
      forcedEos = null;
      const afterEos = await engine.completions.create(options);
      if (afterEos.choices[0].text !== report.completion || JSON.stringify(generatedIds) !== JSON.stringify(report.generatedIds)) {
        throw new Error("Generation changed after an EOS termination");
      }
      report.steps.push("forced EOS termination and recovery parity");
    }
    report.status = "passed-smoke-only";
  } catch (error) {
    report.status = "failed";
    report.failedStage = status.textContent;
    report.error = error?.stack ?? error?.message ?? String(error);
    report.errorType = error?.constructor?.name;
    report.generatedIds ??= [...generatedIds];
    report.firstLogits ??= firstLogits;
    if (firstLogits) report.invalidFirstLogitCount = firstLogits.filter(value => !Number.isFinite(value)).length;
    if (engine && report.inputIds && !report.capture) {
      try {
        const capture = await engine.captureDrowseResiduals(report.inputIds, capturePositions(report.inputIds.length));
        report.capture = { ...capture, values: Array.from(capture.values) };
        report.invalidCaptureValueCount = report.capture.values.filter(value => !Number.isFinite(value)).length;
      } catch (captureError) { report.diagnosticCaptureError = captureError?.message ?? String(captureError); }
    }
  } finally {
    if (engine) {
      try { await engine.unload(); report.unloaded = true; } catch (error) { report.unloadError = String(error); report.status = "failed"; }
      engine = undefined;
    }
    report.finishedAt = new Date().toISOString();
    if (report.diagnosticFailures?.length && report.status === "passed-smoke-only") report.status = "diagnostic-only";
    const saved = await (await fetch("/report", { method: "POST", body: JSON.stringify(report) })).json();
    const { firstLogits, capture, repeatCapture, branchReplayCapture, branchRecoveryCapture, explicitResetCapture, ...summary } = report;
    reportView.textContent = JSON.stringify({ ...summary, captureShape: capture ? [capture.layerCount, capture.positionCount, capture.hiddenSize] : null, reportFile: saved.file }, null, 2);
    status.textContent = report.status;
    run.disabled = false; stop.disabled = true;
  }
};
function capturePositions(length) {
  return length <= 24 ? Array.from({length}, (_, index) => index) :
    [...new Set([0, 1, 2, 127, 128, 511, 512, 513, 1023, 1024, 1025, length - 1])].filter(position => position < length).sort((a, b) => a - b);
}
</script></html>`;
}
