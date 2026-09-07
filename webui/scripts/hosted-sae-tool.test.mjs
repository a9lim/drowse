import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";
import { createServer } from "vite";
import { selectHostedSaeLayer } from "./hosted-sae-layer.mjs";

const exec = promisify(execFile);
const webuiRoot = resolve(import.meta.dirname, "..");
const buildScript = resolve(import.meta.dirname, "build-hosted-sae.mjs");
await exec(process.execPath, ["--check", buildScript]);
const buildSource = await readFile(buildScript, "utf8");
assert.match(buildSource, /\.\.\.executionProfiles/);
assert.match(buildSource, /selectHostedSaeLayer\(lock\.layerMap, options\.layer\)/);
const help = await exec(process.execPath, [buildScript, "--help"]);
assert.match(help.stdout, /CORPUS\.json OUTPUT_DIR/);
assert.match(help.stdout, /--feature-metadata FEATURES\.json/);
assert.match(help.stdout, /nearest 65% model depth/);

const runtimeLock = JSON.parse(
  await readFile(resolve(webuiRoot, "../browser-runtime/runtime-lock.json"), "utf8"),
);
const qwen4b = runtimeLock.models.find((model) => model.id === "qwen3-4b");
const qwenLarge = runtimeLock.models.find((model) => model.id === "qwen3-1.7b");
assert.equal(selectHostedSaeLayer(qwen4b.layerMap), 23);
assert.equal(selectHostedSaeLayer(qwenLarge.layerMap), 18);
assert.equal(selectHostedSaeLayer([2, 4, 7, 11], 4), 4);
assert.throws(
  () => selectHostedSaeLayer([2, 4, 7, 11], 5),
  /SAE layer 5 is not in the runtime lock/,
);
assert.equal(selectHostedSaeLayer(Array.from({ length: 11 }, (_, index) => index)), 6);
assert.equal(selectHostedSaeLayer(Array.from({ length: 31 }, (_, index) => index)), 20);
const captureScript = resolve(import.meta.dirname, "capture-hosted-residuals.mjs");
await exec(process.execPath, ["--check", captureScript]);
const captureHelp = await exec(process.execPath, [captureScript, "--help"]);
assert.match(captureHelp.stdout, /INPUT\.json OUTPUT_DIR/);

const vite = await createServer({
  root: webuiRoot,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  optimizeDeps: { noDiscovery: true, include: [] },
  server: { middlewareMode: true, watch: null },
});
let decodeFp32Safetensors;
try {
  const transformed = await vite.transformRequest("/scripts/hosted-sae-capture.worker.ts");
  assert.ok(transformed?.code.includes("capturePreparedPositions"));
  assert.ok(transformed.code.includes("structuredHookProfile: event.data.structuredHookProfile"));
  assert.ok(transformed.code.includes("thinkingProfile: event.data.thinkingProfile"));
  assert.ok(transformed.code.includes("maxComputeWorkgroupSizeX"));
  assert.ok(transformed.code.includes("maxComputeInvocationsPerWorkgroup"));
  const residualWorker = await vite.transformRequest(
    "/scripts/hosted-residual-capture.worker.ts",
  );
  assert.ok(residualWorker?.code.includes("capturePreparedPositions"));
  assert.ok(residualWorker.code.includes("structuredHookProfile: event.data.structuredHookProfile"));
  assert.ok(residualWorker.code.includes("thinkingProfile: event.data.thinkingProfile"));
  assert.ok(residualWorker.code.includes("maxComputeWorkgroupSizeX"));
  assert.ok(residualWorker.code.includes("maxComputeInvocationsPerWorkgroup"));
  ({ decodeFp32Safetensors } = await vite.ssrLoadModule(
    "/src/hosted/artifacts/safetensors.ts",
  ));
} finally {
  await vite.close();
}
assert.match(await readFile(captureScript, "utf8"), /\.\.\.executionProfiles/);

const root = await mkdtemp(join(tmpdir(), "drowse-sae-trainer-"));
const activationPath = join(root, "activations.f32");
const metadataPath = join(root, "capture.json");
const outputPath = join(root, "pack");
const values = Float32Array.from([
  1, 0, 0, 1,
  0, 1, 1, 0,
  1, 1, 0, 0,
  0, 0, 1, 1,
  2, 0, 0, 1,
  0, 2, 1, 0,
  1, 1, 1, 0,
  0, 1, 1, 1,
]);
const activationBytes = Buffer.from(values.buffer);
await writeFile(activationPath, activationBytes);
await writeFile(metadataPath, `${JSON.stringify({
  schema_version: 1,
  model_id: "fixture/model",
  model_source_fingerprint: "a".repeat(40),
  runtime_identity_sha256: "b".repeat(64),
  layer: 1,
  hidden_size: 4,
  rows: 8,
  seq_len: 4,
  corpus_spec: "fixture",
  corpus_sha256: "c".repeat(64),
  capture_plan_sha256: "d".repeat(64),
  activation_sha256: sha256(activationBytes),
}, null, 2)}\n`);
const trainer = resolve(webuiRoot, "..", "browser-runtime", "train_hosted_sae.py");
const trained = await exec("python3", [
  trainer,
  activationPath,
  metadataPath,
  outputPath,
  "--features", "3",
  "--batch-size", "4",
  "--device", "cpu",
]);
assert.equal(JSON.parse(trained.stdout).features, 3);
const manifest = JSON.parse(await readFile(join(outputPath, "packs/sae/manifest.json"), "utf8"));
assert.equal(manifest.d_model, 4);
assert.equal(manifest.d_sae, 3);
assert.equal(manifest.tokens_trained, 8);
const tensorBytes = await readFile(join(outputPath, "packs/sae", manifest.tensor_file));
assert.equal(sha256(tensorBytes), manifest.tensor_sha256);
const tensors = decodeFp32Safetensors(new Uint8Array(tensorBytes), manifest.tensor_file);
assert.deepEqual(tensors.keys, ["W_dec", "W_enc", "b_dec", "b_enc"]);
assert.deepEqual(tensors.description.shapes.get("W_enc"), [4, 3]);
assert.deepEqual(tensors.description.shapes.get("W_dec"), [3, 4]);

console.log("Hosted SAE production-tool checks passed");

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
