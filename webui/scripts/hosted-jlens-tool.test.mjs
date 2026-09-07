#!/usr/bin/env node

import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { promisify } from "node:util";
import { createServer } from "vite";
import { selectHostedJlensLayers } from "./hosted-instrument-layers.mjs";

const exec = promisify(execFile);
const webuiRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(webuiRoot, "..");
const builder = resolve(import.meta.dirname, "build-hosted-jlens.mjs");
const fitter = resolve(repositoryRoot, "browser-runtime/fit_hosted_jlens.py");

const help = await exec(process.execPath, [builder, "--help"]);
assert.match(help.stdout, /MODEL_ID MODEL_DIR MODEL\.wasm WEBLLM\.js CORPUS\.json WORDS\.json OUTPUT_DIR/);
assert.match(help.stdout, /--source-layers/);
assert.match(await readFile(builder, "utf8"), /\.\.\.executionProfiles/);
assert.match(await readFile(builder, "utf8"), /resolve\(options\.python\)/);

assert.deepEqual(
  selectHostedJlensLayers(Array.from({ length: 32 }, (_, index) => index), 960),
  [0, 4, 9, 13, 17, 21, 26, 30],
);
assert.deepEqual(
  selectHostedJlensLayers(Array.from({ length: 28 }, (_, index) => index), 2048),
  [0, 4, 7, 11, 15, 19, 22, 26],
);
const qwen4Layers = selectHostedJlensLayers(
  Array.from({ length: 36 }, (_, index) => index),
  2560,
);
assert.equal(qwen4Layers.length, 8);
assert.equal(qwen4Layers[0], 0);
assert.equal(qwen4Layers.at(-1), 34);
assert.equal(new Set(qwen4Layers).size, qwen4Layers.length);
assert.throws(
  () => selectHostedJlensLayers([0], 128),
  /source layer before the final residual/,
);

const vite = await createServer({
  root: webuiRoot,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  optimizeDeps: { noDiscovery: true, include: [] },
  server: { middlewareMode: true, watch: null },
});
try {
  const transformed = await vite.transformRequest("/scripts/hosted-jlens-capture.worker.ts");
  assert.ok(transformed?.code.includes("structuredHookProfile: event.data.structuredHookProfile"));
  assert.ok(transformed.code.includes("thinkingProfile: event.data.thinkingProfile"));
  assert.ok(transformed.code.includes("maxComputeWorkgroupSizeX"));
  assert.ok(transformed.code.includes("maxComputeInvocationsPerWorkgroup"));
} finally {
  await vite.close();
}

await assert.rejects(
  exec(process.execPath, [
    builder,
    "model", "directory", "model.wasm", "webllm.js", "corpus.json", "words.json", "output",
    "--source-layers", "1,nope",
  ]),
  /source layers must be a comma-separated integer list/,
);

const python = process.env.PYTHON ?? "python3";
const pythonHelp = await exec(python, [fitter, "--help"], {
  cwd: repositoryRoot,
  env: { ...process.env, PYTHONPATH: repositoryRoot },
});
assert.match(pythonHelp.stdout, /--dim-batch/);
assert.match(pythonHelp.stdout, /--source-layers/);

console.log("Hosted J-lens production-tool checks passed");
