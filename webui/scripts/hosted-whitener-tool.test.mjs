import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { cp, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";
import { createServer } from "vite";
import { readCoreManifoldSource } from "./hosted-core-manifold-source.mjs";
import { lockedModelExecutionProfiles } from "./hosted-model-profiles.mjs";

const exec = promisify(execFile);
const webuiRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(webuiRoot, "..");
for (const script of ["build-hosted-whitener.mjs", "validate-hosted-core-pack.mjs"]) {
  await exec(process.execPath, ["--check", resolve(import.meta.dirname, script)]);
}
const runtimeLock = JSON.parse(await readFile(
  resolve(repositoryRoot, "browser-runtime/runtime-lock.json"),
  "utf8",
));
const qwen4b = runtimeLock.models.find((model) => model.id === "qwen3-4b");
assert.equal(qwen4b.quantization, "q4f16_1");
const executionProfiles = lockedModelExecutionProfiles(qwen4b, {
  structuredHookProfile: qwen4b.structuredHookProfile,
  thinkingProfile: qwen4b.thinkingProfile,
  quantization: qwen4b.quantization,
});
assert.deepEqual(executionProfiles, {
  structuredHookProfile: "standard-v3",
  thinkingProfile: qwen4b.thinkingProfile,
  quantization: "q4f16_1",
  requiredFeatures: ["shader-f16"],
});
assert.throws(
  () => lockedModelExecutionProfiles(qwen4b, {
    structuredHookProfile: "standard-v1",
    thinkingProfile: qwen4b.thinkingProfile,
    quantization: qwen4b.quantization,
  }),
  /runtime-lock execution profile/,
);
const seedSource = resolve(
  repositoryRoot,
  "browser-runtime/release-inputs/core-manifolds/welcoming.detached",
);
const seed = await readCoreManifoldSource(seedSource, repositoryRoot);
assert.equal(seed.name, "welcoming.detached");
assert.equal(seed.fitMode, "pca");
assert.deepEqual(seed.tags, ["starter", "tone"]);
assert.deepEqual(seed.source, { uri: "local", repository: null, revision: null });
assert.deepEqual(seed.nodes.map((node) => node.label), ["welcoming", "detached"]);
assert.deepEqual(seed.nodes.map((node) => node.statements.length), [48, 48]);

const invalidRoot = await mkdtemp(join(tmpdir(), "drowse-core-manifold-source-"));
const invalidSource = join(invalidRoot, "welcoming.detached");
try {
  await cp(seedSource, invalidSource, { recursive: true });
  await writeFile(join(invalidSource, "fitted.safetensors"), "not allowed");
  await assert.rejects(
    readCoreManifoldSource(invalidSource, repositoryRoot),
    /unexpected file closure/,
  );
  await rm(join(invalidSource, "fitted.safetensors"));
  const provenance = JSON.parse(await readFile(join(invalidSource, "provenance.json"), "utf8"));
  provenance.baseline_prompts_sha256 = "0".repeat(64);
  await writeFile(join(invalidSource, "provenance.json"), `${JSON.stringify(provenance, null, 2)}\n`);
  await assert.rejects(
    readCoreManifoldSource(invalidSource, repositoryRoot),
    /source provenance is invalid/,
  );
} finally {
  await rm(invalidRoot, { recursive: true, force: true });
}

const python = process.env.PYTHON ?? "python3";
const loaded = await exec(python, [
  "-c",
  [
    "import json, sys",
    "from pathlib import Path",
    "from drowse.io.manifold_folder import ManifoldFolder",
    "folder = ManifoldFolder.load(Path(sys.argv[1]))",
    "print(json.dumps({'name': folder.name, 'fit_mode': folder.fit_mode, 'nodes': folder.node_labels, 'groups': [len(rows) for _, rows in folder.node_groups()], 'files': folder.files}))",
  ].join("; "),
  seedSource,
], { cwd: repositoryRoot, env: { ...process.env, PYTHONPATH: repositoryRoot } });
assert.deepEqual(JSON.parse(loaded.stdout), {
  name: "welcoming.detached",
  fit_mode: "pca",
  nodes: ["welcoming", "detached"],
  groups: [48, 48],
  files: {},
});
assert.throws(
  () => lockedModelExecutionProfiles(qwen4b, {
    structuredHookProfile: qwen4b.structuredHookProfile,
    thinkingProfile: {
      start: "<think>",
      end: "</think>",
      startsInThinking: false,
      startTokenIds: [1],
      endTokenIds: [2],
    },
    quantization: qwen4b.quantization,
  }),
  /runtime-lock execution profile/,
);

const help = await exec(process.execPath, [
  resolve(import.meta.dirname, "build-hosted-whitener.mjs"),
  "--help",
]);
assert.match(help.stdout, /MODEL_ID MODEL_DIR MODEL\.wasm WEBLLM\.js OUTPUT_DIR/);
assert.match(help.stdout, /--manifold-source DIR/);
assert.match(help.stdout, /--runtime-lock PATH/);
assert.match(help.stdout, /--manual-browser/);
await assert.rejects(
  exec(process.execPath, [
    resolve(import.meta.dirname, "build-hosted-whitener.mjs"),
    "model", "directory", "model.wasm", "webllm.js", "output",
  ]),
  /--manifold-source is required/,
);

const vite = await createServer({
  root: webuiRoot,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  publicDir: resolve(webuiRoot, "public-hosted"),
  server: { middlewareMode: true, watch: null },
});
try {
  const transformed = await vite.transformRequest("/scripts/hosted-whitener.worker.ts");
  assert.ok(transformed?.code.includes("module_or_path"));
  assert.ok(transformed.code.includes("BrowserFittingCoordinator"));
  assert.ok(transformed.code.includes("structuredHookProfile: event.data.structuredHookProfile"));
  assert.ok(transformed.code.includes("thinkingProfile: event.data.thinkingProfile"));
  assert.ok(transformed.code.includes("modelType: event.data.modelType"));
  assert.ok(transformed.code.includes("maxComputeWorkgroupSizeX"));
  assert.ok(transformed.code.includes("maxComputeInvocationsPerWorkgroup"));
  assert.ok(transformed.code.includes("contextBindingSha256"));
  assert.ok(transformed.code.includes("consensusGram: foundation.consensusGram"));
  assert.ok(transformed.code.includes("evaluatedLayers: foundation.layers"));
  const { browserFittedFlatDiscoverPack } = await vite.ssrLoadModule(
    "/src/hosted/artifacts/fittedAuthoring.ts",
  );
  const { validateDrowseArchive } = await vite.ssrLoadModule(
    "/src/hosted/artifacts/drowseArchive.ts",
  );
  const evaluated = {
    operation: "affine_fisher",
    nodeCount: 2,
    columns: 2,
    components: 1,
    centroidMean: new Float64Array(2),
    mean: new Float64Array(2),
    basis: new Float64Array([1, 0]),
    nodeCoordinates: new Float64Array([-1, 1]),
    muCoordinates: new Float64Array([-1, 1]),
    whitenedGram: new Float64Array([1, -1, -1, 1]),
    neutralCrossGram: new Float64Array(2),
    explainedVariance: 1,
    mahalanobisShare: 2,
  };
  const archive = await browserFittedFlatDiscoverPack({
    manifold: seed,
    closure: { source: seed.source, tags: seed.tags, template: null },
    modelId: "fixture/model",
    producerVersion: "browser-test",
    identity: {
      runtimeIdentitySha256: "1".repeat(64),
      contextBindingSha256: "2".repeat(64),
      modelSourceFingerprint: null,
      captureSha256: "3".repeat(64),
      captureVersion: 3,
      captureRenderSha256: "4".repeat(64),
      baselinePromptsSha256: "5".repeat(64),
      fitPolicyVersion: 1,
    },
    topology: {
      operation: "topology",
      winnerName: "flat-pca",
      fitMode: "pca",
      intrinsicDimensions: 1,
      periodicDimensions: 0,
      persistentLoops: 0,
      usedFaintCycle: false,
      coordinates: new Float64Array([-1, 1]),
      embeddedCoordinates: new Float64Array([-1, 1]),
      candidates: [],
      diagnostics: {
        kind: "pca",
        perComponentVariance: new Float64Array([1]),
        cumulativeVariance: new Float64Array([1]),
        pickedDimensions: 1,
        threshold: 0.7,
      },
      winnerPlan: null,
    },
    consensusGram: new Float64Array([1, -1, -1, 1]),
    nodeCoordinates: new Float64Array([-1, 1]),
    evaluatedLayers: new Map([[0, evaluated]]),
    layers: new Map([[0, { ...evaluated, affineMap: null }]]),
  });
  const verified = await validateDrowseArchive(archive);
  assert.deepEqual(verified.primaryIdentity, ["default", "welcoming.detached"]);
  assert.equal(verified.fittedArtifacts.length, 1);
  assert.equal(verified.fittedArtifacts[0].variant, "raw");
  assert.equal(verified.fittedArtifacts[0].modelId, "fixture/model");
  assert.equal(verified.fittedArtifacts[0].contextBindingSha256, "2".repeat(64));
  assert.match(verified.fittedArtifacts[0].tensorPath, /\.safetensors$/);
  assert.equal(verified.manifold.source, "local");
  assert.deepEqual(verified.manifold.tags, ["starter", "tone"]);
} finally {
  await vite.close();
}

const builderSource = await readFile(resolve(import.meta.dirname, "build-hosted-whitener.mjs"), "utf8");
assert.match(builderSource, /\.\.\.executionProfiles/);
assert.match(builderSource, /readRuntimeLock\(runtimeLockPath\)/);
assert.match(builderSource, /options\.manualBrowser \? null : await launchReleaseToolContext/);
assert.match(builderSource, /message\.type !== "done"/);
assert.doesNotMatch(builderSource, /manifold === null|manifold !== null/);
const validatorSource = await readFile(resolve(import.meta.dirname, "validate-hosted-core-pack.mjs"), "utf8");
assert.match(validatorSource, /structuredHookProfile: lock\.structuredHookProfile/);
assert.match(validatorSource, /id: `\$\{modelId\}-\$\{lock\.quantization\}`/);
assert.match(validatorSource, /readRuntimeLock\(runtimeLockPath\)/);
assert.doesNotMatch(validatorSource, /structuredHookProfile: "standard-v1"/);

console.log("Hosted whitener release-tool checks passed");
