import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";
import { createServer } from "vite";
import { verifyArtifactFile } from "./verify-artifact-file.mjs";

const exec = promisify(execFile);
const webuiRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(webuiRoot, "..");
const runtimeScript = resolve(
  import.meta.dirname,
  "browser-instrument-runtime.mjs",
);
await exec(process.execPath, ["--check", runtimeScript]);
assert.match(await readFile(runtimeScript, "utf8"), /\.\.\.executionProfiles/);
assert.match(await readFile(runtimeScript, "utf8"), /--jlens-layer-limit/);
const runtimeLock = JSON.parse(
  await readFile(
    resolve(repositoryRoot, "browser-runtime/runtime-lock.json"),
    "utf8",
  ),
);
const lock = runtimeLock.models.find(
  (entry) => entry.id === "qwen3-1.7b",
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
const vite = await createServer({
  root: webuiRoot,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
let encodeFp32Safetensors;
let fingerprint;
try {
  const runtimeWorker = await vite.transformRequest(
    "/scripts/hosted-instrument-runtime.worker.ts",
  );
  assert.ok(
    runtimeWorker?.code.includes(
      "structuredHookProfile: event.data.structuredHookProfile",
    ),
  );
  assert.ok(
    runtimeWorker.code.includes("thinkingProfile: event.data.thinkingProfile"),
  );
  assert.ok(runtimeWorker.code.includes("setSaeDictionary"));
  assert.ok(runtimeWorker.code.includes("standard-v3 exact-readout-v1"));
  assert.ok(
    runtimeWorker.code.includes("exact full-vocabulary J-lens readout"),
  );
  assert.ok(runtimeWorker.code.includes("exact full-dictionary SAE readout"));
  assert.ok(runtimeWorker.code.includes("artifact_staging"));
  assert.ok(runtimeWorker.code.includes("prior_step_gated_generation"));
  assert.ok(runtimeWorker.code.includes("token_milestone"));
  assert.ok(runtimeWorker.code.includes("activationTransitionIndices"));
  assert.ok(runtimeWorker.code.includes("timedStep"));
  assert.ok(runtimeWorker.code.includes("maxComputeWorkgroupSizeX"));
  assert.ok(runtimeWorker.code.includes("maxComputeInvocationsPerWorkgroup"));
  assert.ok(runtimeWorker.code.includes("diagnosticJlensLayerLimit"));
  assert.ok(runtimeWorker.code.includes("limitJlensLayers"));
  assert.ok(runtimeWorker.code.includes("referenceCapture"));
  assert.ok(runtimeWorker.code.includes("inputIds.length !== generation.usage.promptTokens"));
  ({ encodeFp32Safetensors } = await vite.ssrLoadModule(
    "/src/hosted/artifacts/safetensors.ts",
  ));
  const { runtimeIdentitySha256 } = await vite.ssrLoadModule(
    "/src/lib/runtime/catalog.ts",
  );
  fingerprint = runtimeIdentitySha256(runtimeIdentity);
} finally {
  await vite.close();
}

const root = await mkdtemp(join(tmpdir(), "drowse-instrument-validator-"));
const invalidRuntimeLock = join(root, "invalid-runtime-lock.json");
await writeFile(invalidRuntimeLock, JSON.stringify({ ...runtimeLock, runtimeAbi: "unsupported" }));
await assert.rejects(
  exec(process.execPath, [runtimeScript,
    "--model-id", lock.id,
    "--model-directory", root,
    "--model-library", join(root, "model.wasm"),
    "--webllm", join(root, "webllm.js"),
    "--core-directory", root,
    "--jlens-directory", root,
    "--sae-directory", root,
    "--runtime-lock", invalidRuntimeLock,
  ]),
  /runtime lock has an unsupported schema or ABI/,
);
const integrityFixture = join(root, "integrity.bin");
const integrityBytes = Buffer.from("verified instrument artifact");
await writeFile(integrityFixture, integrityBytes);
await verifyArtifactFile(
  integrityFixture,
  { bytes: integrityBytes.byteLength, sha256: sha256(integrityBytes) },
  "integrity fixture",
);
await assert.rejects(
  verifyArtifactFile(
    integrityFixture,
    { bytes: integrityBytes.byteLength + 1, sha256: sha256(integrityBytes) },
    "integrity fixture",
  ),
  /byte count differs/,
);
await assert.rejects(
  verifyArtifactFile(
    integrityFixture,
    { bytes: integrityBytes.byteLength, sha256: "0".repeat(64) },
    "integrity fixture",
  ),
  /SHA-256 differs/,
);
const sae = join(root, "sae-pack");
const saeRoot = join(sae, "packs", "sae");
await mkdir(saeRoot, { recursive: true });
const saeTensor = encodeFp32Safetensors({
  W_enc: {
    shape: [lock.hiddenSize, 2],
    data: new Float32Array(lock.hiddenSize * 2),
  },
  W_dec: {
    shape: [2, lock.hiddenSize],
    data: new Float32Array(lock.hiddenSize * 2),
  },
  b_enc: { shape: [2], data: new Float32Array(2) },
  b_dec: { shape: [lock.hiddenSize], data: new Float32Array(lock.hiddenSize) },
});
const saeTensorName = "packs/sae/layer-16.safetensors";
const saeTensorSha = sha256(saeTensor);
const saeManifest = Buffer.from(
  `${JSON.stringify(
    {
      format_version: 1,
      kind: "local",
      name: "fixture",
      release: "local:fixture",
      model_id: lock.sourceRepository,
      model_fingerprint: fingerprint,
      model_source_fingerprint: lock.sourceRevision,
      layer: 16,
      d_model: lock.hiddenSize,
      d_sae: 2,
      activation: "relu",
      tensor_file: "layer-16.safetensors",
      tensor_sha256: saeTensorSha,
      corpus_spec: "fixture",
      corpus_sha256: "b".repeat(64),
      tokens_trained: 2,
      seq_len: 2,
      batch_size: 1,
      learning_rate: 0.001,
      l1_coefficient: 0.001,
      dead_feature_threshold: 0,
    },
    null,
    2,
  )}\n`,
);
const saeFeatures = Buffer.from(
  `${JSON.stringify(
    {
      format_version: 1,
      model_id: lock.sourceRepository,
      release: "local:fixture",
      features: {
        0: { label: "opening delimiters", max_act: 4.5 },
        1: { label: null, max_act: null },
      },
    },
    null,
    2,
  )}\n`,
);
await Promise.all([
  writeFile(join(sae, saeTensorName), saeTensor),
  writeFile(join(saeRoot, "manifest.json"), saeManifest),
  writeFile(join(saeRoot, "features.json"), saeFeatures),
]);
await writePackManifest(sae, [
  ["packs/sae/manifest.json", saeManifest],
  ["packs/sae/features.json", saeFeatures],
  [saeTensorName, saeTensor],
]);

const script = resolve(
  import.meta.dirname,
  "validate-hosted-instrument-pack.mjs",
);
const valid = await exec(process.execPath, [script, lock.id, "sae", sae]);
const result = JSON.parse(valid.stdout);
assert.equal(result.descriptor.features, 2);
assert.equal(result.descriptor.layer, 16);

const jlens = join(root, "jlens-pack");
const jlensRoot = join(jlens, "packs", "jlens");
await mkdir(jlensRoot, { recursive: true });
const jlensTensor = encodeFp32Safetensors({
  layer_0: {
    shape: [lock.hiddenSize, lock.hiddenSize],
    data: new Float32Array(lock.hiddenSize * lock.hiddenSize),
  },
});
const jlensTensorName = "packs/jlens/layer-0.safetensors";
const jlensTensorSha = sha256(jlensTensor);
const jlensManifest = Buffer.from(
  `${JSON.stringify(
    {
      format_version: 6,
      method: "jlens_cotangent_sum",
      n_prompts: 1,
      checkpoint: false,
      dtype: "float32",
      d_model: lock.hiddenSize,
      source_layers: [0],
      tensor_files: { 0: "layer-0.safetensors" },
      tensor_sha256: { 0: jlensTensorSha },
      corpus_spec: "fixture",
      corpus_sha256: "c".repeat(64),
      corpus_hash_kind: "text_v1",
      seq_len: 4,
      dim_batch: 1,
      skip_first_positions: 0,
      estimator_policy: { method: "jlens_cotangent_sum" },
      raw_corpus_sha256: null,
      raw_prompt_count: null,
      usable_prompt_count: null,
      model_layer_count: lock.layerMap.length,
      model_fingerprint: fingerprint,
      model_source_fingerprint: lock.sourceRevision,
      base_n_prompts: null,
      partial_n_prompts: null,
      consumed_prefix_sha256: null,
    },
    null,
    2,
  )}\n`,
);
const vocabularyTensor = encodeFp32Safetensors({
  unembedding: {
    shape: [1, lock.hiddenSize],
    data: new Float32Array(lock.hiddenSize),
  },
});
const vocabularyTensorName = "packs/jlens/browser-vocabulary.safetensors";
const vocabularyTensorSha = sha256(vocabularyTensor);
const vocabularyManifest = Buffer.from(
  `${JSON.stringify(
    {
      format: "drowse-jlens-vocabulary-v2",
      hidden_size: lock.hiddenSize,
      tensor_file: "browser-vocabulary.safetensors",
      tensor_sha256: vocabularyTensorSha,
      words: [{ word: "hello", row: 0, token_id: 1 }],
    },
    null,
    2,
  )}\n`,
);
await Promise.all([
  writeFile(join(jlens, jlensTensorName), jlensTensor),
  writeFile(join(jlensRoot, "manifest.json"), jlensManifest),
  writeFile(join(jlens, vocabularyTensorName), vocabularyTensor),
  writeFile(join(jlensRoot, "browser-vocabulary.json"), vocabularyManifest),
]);
await writePackManifest(jlens, [
  ["packs/jlens/manifest.json", jlensManifest],
  [jlensTensorName, jlensTensor],
  ["packs/jlens/browser-vocabulary.json", vocabularyManifest],
  [vocabularyTensorName, vocabularyTensor],
]);
const validJlens = await exec(process.execPath, [
  script,
  lock.id,
  "jlens",
  jlens,
]);
const jlensResult = JSON.parse(validJlens.stdout);
assert.deepEqual(jlensResult.descriptor.layers, [0]);
assert.deepEqual(jlensResult.descriptor.words, [{ word: "hello", tokenId: 1 }]);

const invalid = join(root, "invalid-pack");
await mkdir(invalid, { recursive: true });
await writeFile(join(invalid, "manifest.json"), saeManifest);
await writePackManifest(invalid, [["manifest.json", saeManifest]]);
await assert.rejects(
  exec(process.execPath, [script, lock.id, "sae", invalid]),
  /nested sae\/ closure/,
);

console.log("Hosted instrument-pack release-tool checks passed");

async function writePackManifest(directory, entries) {
  const files = entries.map(([path, bytes]) => ({
    path,
    bytes: bytes.byteLength,
    sha256: sha256(bytes),
  }));
  await writeFile(
    join(directory, "hosted-pack-artifacts.json"),
    `${JSON.stringify({ schemaVersion: 1, files }, null, 2)}\n`,
  );
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
