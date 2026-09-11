import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createServer } from "vite";
import {
  parseArguments,
  prepareCatalog,
  prepareRequirements,
  validatePackDirectory,
  validatePackSemantics,
  validateWeightBufferRequirements,
} from "./build-hosted-catalog.mjs";
import { expectedBuildToolchain } from "./hosted-model-closure.mjs";

assert.deepEqual(parseArguments(["release.json", "catalog.json"]), {
  specPath: join(process.cwd(), "release.json"),
  outputPath: join(process.cwd(), "catalog.json"),
});
assert.throws(() => parseArguments(["release.json"]), /usage/);
assert.throws(() => parseArguments(["release.json", "catalog.json", "--force"]), /usage/);
const q0Cache = { records: [{ records: [{ dtype: "float32", shape: [262144, 640], format: "raw" }] }] };
const q0Limits = { limits: { maxBufferSize: 671088640, maxStorageBufferBindingSize: 671088640 } };
assert.equal(validateWeightBufferRequirements(q0Limits, q0Cache, "q0 fixture"), 671088640);
for (const name of ["maxBufferSize", "maxStorageBufferBindingSize"]) {
  assert.throws(() => validateWeightBufferRequirements(
    { limits: { ...q0Limits.limits, [name]: 134217728 } }, q0Cache, "q0 fixture",
  ), new RegExp(`${name} must declare at least 671088640`));
}
q0Cache.records[0].records[0].format = "f32-to-bf16";
assert.equal(validateWeightBufferRequirements(q0Limits, q0Cache, "encoded f32 fixture"), 671088640);
for (const cache of [
  { records: [] },
  { records: [{}] },
  { records: [{ records: [{ dtype: "float32", shape: [-1] }] }] },
  { records: [{ records: [{ dtype: "unknown", shape: [1] }] }] },
  { records: [{ records: [{ dtype: "float32", shape: [Number.MAX_SAFE_INTEGER, 2] }] }] },
]) {
  assert.throws(() => validateWeightBufferRequirements(q0Limits, cache, "invalid fixture"), /weight tensor/);
}
const semanticCommands = [];
const semanticRunner = async (...args) => semanticCommands.push(args);
await validatePackSemantics("fixture", "core", "/tmp/core", semanticRunner);
await validatePackSemantics("fixture", "sae", "/tmp/sae", semanticRunner);
assert.deepEqual(
  semanticCommands.map(([, args]) => args.slice(1)),
  [
    ["fixture", "/tmp/core"],
    ["fixture", "sae", "/tmp/sae"],
  ],
);

const root = await mkdtemp(join(tmpdir(), "drowse-catalog-builder-"));
try {
  const modelDirectory = join(root, "model");
  const coreDirectory = join(root, "core");
  const instrumentDirectory = join(root, "instruments");
  const jlensDirectory = join(root, "jlens");
  await Promise.all([
    mkdir(modelDirectory),
    mkdir(join(coreDirectory, "packs"), { recursive: true }),
    mkdir(join(instrumentDirectory, "packs", "sae"), { recursive: true }),
    mkdir(join(jlensDirectory, "packs", "jlens"), { recursive: true }),
  ]);
  const weightContents = "data";
  const tensorCacheContents = JSON.stringify({
    records: [{ dataPath: "params_shard_0.bin", nbytes: Buffer.byteLength(weightContents),
      records: [{ dtype: "float32", shape: [1], byteOffset: 0, nbytes: 4, format: "raw" }],
    }],
  });
  const configContents = JSON.stringify({
    model_type: "qwen3",
    quantization: "q4f16_1",
    context_window_size: 4096,
    prefill_chunk_size: 2048,
    model_config: {
      context_window_size: 4096,
      prefill_chunk_size: 2048,
      hidden_size: 4,
      num_hidden_layers: 2,
    },
  });
  const revision = "b".repeat(40);
  const instrumentRevision = "c".repeat(40);
  const sourceRevision = "a".repeat(40);
  const sourceRepository = "Qwen/Qwen3-0.6B";
  const thinkingProfile = {
    start: "<think>",
    end: "</think>",
    startsInThinking: false,
    startTokenIds: [151667],
    endTokenIds: [151668],
  };
  const licenseText = await readFile(
    new URL("../../browser-runtime/forks/licenses/Apache-2.0.txt", import.meta.url),
    "utf8",
  );
  const forkManifestBytes = await readFile(
    new URL("../../browser-runtime/forks/manifest.json", import.meta.url),
  );
  const buildToolchain = expectedBuildToolchain(
    JSON.parse(forkManifestBytes.toString("utf8")),
  );
  const sourceReadme = "---\nlicense: apache-2.0\n---\n# Fixture source\n";
  const licenseMetadata = JSON.stringify({
    schemaVersion: 1,
    spdx: "apache-2.0",
    declaredIn: "README.md",
    licenseText: "LICENSE.model",
    licenseTextSource: "drowse-standard-text",
  });
  const convertedEntries = [
    ["params_shard_0.bin", "weight", weightContents],
    ["tokenizer.json", "tokenizer", "tokenizer"],
    ["mlc-chat-config.json", "configuration", configContents],
    ["tensor-cache.json", "converted_manifest", tensorCacheContents],
    ["tokenizer_config.json", "chat_template", "template"],
    ["LICENSE.model", "configuration", licenseText],
    ["MODEL-LICENSE.json", "configuration", licenseMetadata],
    ["README.source.md", "configuration", sourceReadme],
  ];
  const buildFiles = convertedEntries.map(([path, _role, contents]) =>
    record(path, undefined, contents)
  ).map(({ path, bytes, sha256 }) => ({ path, bytes, sha256 }));
  const buildContents = JSON.stringify({
    schemaVersion: 1,
    runtimeAbi: "drowse-web-runtime-v1",
    hookAbi: "post-block-residual-v4",
    structuredHookProfile: "standard-v3",
    thinkingProfile,
    architecture: "qwen3",
    quantization: "q4f16_1",
    contextWindowSize: 4096,
    prefillChunkSize: 2048,
    source: {
      repository: sourceRepository,
      revision: sourceRevision,
      chatTemplateSha256: "b".repeat(64),
      files: [record("README.md", undefined, sourceReadme)].map(
        ({ path, bytes, sha256 }) => ({ path, bytes, sha256 }),
      ),
    },
    toolchain: buildToolchain,
    files: buildFiles,
  });
  const base = await artifactSet(modelDirectory, [
    ...convertedEntries,
    ["drowse-build.json", "configuration", buildContents],
    ["model.wasm", "model_library", "\0asm"],
  ]);
  const core = await packArtifactSet(coreDirectory, [
    ["packs/core.drowse", "core pack"],
  ]);
  const sae = await packArtifactSet(instrumentDirectory, [
    ["packs/sae/manifest.json", "sae manifest"],
    ["packs/sae/layer-1.safetensors", "sae tensors"],
  ]);
  const jlens = await packArtifactSet(jlensDirectory, [
    ["packs/jlens/manifest.json", "jlens manifest"],
    ["packs/jlens/layer-1.safetensors", "jlens tensors"],
  ]);
  const model = {
    id: "qwen3-0.6b",
    architecture: "qwen3",
    sourceRepository,
    sourceRevision,
    convertedRepository: "logitsml/drowse-web-qwen3-0.6b",
    convertedRevision: revision,
    manifestSha256: role(base, "converted_manifest").sha256,
    librarySha256: role(base, "model_library").sha256,
    tokenizerSha256: role(base, "tokenizer").sha256,
    chatTemplateSha256: role(base, "chat_template").sha256,
    hiddenSize: 4,
    layerMap: [0, 1],
    quantization: "q4f16_1",
    contextProfiles: [2048, 4096],
    structuredHookProfile: "standard-v3",
    thinkingProfile,
  };
  await writeFile(join(modelDirectory, "hosted-artifacts.json"), JSON.stringify({
    schemaVersion: 1,
    runtimeAbi: "drowse-web-runtime-v1",
    hookAbi: "post-block-residual-v4",
    structuredHookProfile: model.structuredHookProfile,
    thinkingProfile: model.thinkingProfile,
    architecture: model.architecture,
    quantization: model.quantization,
    contextWindowSize: 4096,
    prefillChunkSize: 2048,
    hiddenSize: model.hiddenSize,
    layerMap: model.layerMap,
    source: { repository: model.sourceRepository, revision: sourceRevision },
    files: base,
  }));
  const runtimeLock = {
    runtimeAbi: "drowse-web-runtime-v1",
    hookAbi: "post-block-residual-v4",
    toolchain: {
      forkManifestSha256: createHash("sha256").update(forkManifestBytes).digest("hex"),
    },
    models: [model],
  };
  const spec = {
    schemaVersion: 1,
    sequence: 1,
    issuedAt: "2026-08-29T00:00:00.000Z",
    expiresAt: "2026-09-29T00:00:00.000Z",
    models: [{
      id: model.id,
      variantId: `${model.id}-q4f16`,
      tier: "balanced",
      displayName: "Qwen3 0.6B",
      description: "Catalog builder fixture",
      license: "Apache-2.0",
      languages: ["en", "zh"],
      modelDirectory: "model",
      contextProfiles: [2048, 4096].map((contextTokens) => ({
        contextTokens,
        minimumCalibrationScore: 1,
        minimumDeviceMemoryGiB: 4,
        expectedPrefillTokensPerSecond: [1, 2],
        expectedDecodeTokensPerSecond: [3, 4],
        measuredDevices: 1,
      })),
      requirements: {
        features: ["shader-f16"],
        limits: {
          maxBufferSize: 134_217_728,
          maxStorageBufferBindingSize: 134_217_728,
          maxStorageBuffersPerShaderStage: 10,
          maxComputeWorkgroupStorageSize: 32_768,
          maxComputeWorkgroupSizeX: 256,
          maxComputeInvocationsPerWorkgroup: 256,
        },
      },
      packs: [
        {
          id: `${model.id}-core`,
          kind: "core",
          displayName: "Core geometry",
          license: "AGPL-3.0-or-later",
          directory: "core",
          sourceRepository: model.convertedRepository,
          sourceRevision: revision,
          required: true,
          compatibleContextTokens: [2048, 4096],
        },
        {
          id: `${model.id}-jlens`,
          kind: "jlens",
          displayName: "J-lens",
          license: "Apache-2.0",
          directory: "jlens",
          sourceRepository: `${model.convertedRepository}-instruments`,
          sourceRevision: instrumentRevision,
          required: false,
          compatibleContextTokens: [2048, 4096],
        },
        {
          id: `${model.id}-sae`,
          kind: "sae",
          displayName: "SAE",
          license: "Apache-2.0",
          directory: "instruments",
          sourceRepository: `${model.convertedRepository}-instruments`,
          sourceRevision: instrumentRevision,
          required: false,
          compatibleContextTokens: [2048],
        },
      ],
    }],
  };
  const semanticCalls = [];
  const validateSemantics = async (...args) => semanticCalls.push(args);
  const catalog = await prepareCatalog(spec, runtimeLock, root, validateSemantics);
  const undersizedWeightBuffer = structuredClone(spec);
  undersizedWeightBuffer.models[0].requirements.limits.maxStorageBufferBindingSize = 2;
  await assert.rejects(
    () => prepareCatalog(undersizedWeightBuffer, runtimeLock, root, validateSemantics),
    /maxStorageBufferBindingSize must declare at least 4 bytes/,
  );
  const catalogFiles = catalog.models[0].variants[0].files;
  assert.equal(catalog.models[0].variants[0].downloadBytes, sum(catalogFiles));
  assert.deepEqual(
    catalogFiles
      .filter((file) => file.role === "configuration")
      .map((file) => file.path)
      .filter((path) => [
        "LICENSE.model",
        "MODEL-LICENSE.json",
        "README.source.md",
        "drowse-build.json",
        "hosted-artifacts.json",
      ].includes(path))
      .sort(),
    [
      "LICENSE.model",
      "MODEL-LICENSE.json",
      "README.source.md",
    "drowse-build.json",
    "hosted-artifacts.json",
    ],
  );
  assert.equal(catalog.models[0].variants[0].requiredCorePackBytes, sum(core));
  assert.equal(catalog.models[0].variants[0].structuredHookProfile, "standard-v3");
  assert.deepEqual(catalog.models[0].variants[0].thinkingProfile, model.thinkingProfile);
  assert.equal(catalog.models[0].variants[0].packs[1].bytes, sum(jlens));
  assert.equal(catalog.models[0].variants[0].packs[2].bytes, sum(sae));
  assert.equal(catalog.models[0].variants[0].runtimeIdentity.modelLibrarySha256, model.librarySha256);
  assert.match(catalog.models[0].variants[0].packs[0].files[0].url, new RegExp(revision));
  assert.deepEqual(
    semanticCalls.map(([modelId, kind, directory]) => [modelId, kind, directory]),
    [
      [model.id, "core", coreDirectory],
      [model.id, "jlens", jlensDirectory],
      [model.id, "sae", instrumentDirectory],
    ],
  );
  const noJlensSpec = structuredClone(spec);
  noJlensSpec.models[0].packs = noJlensSpec.models[0].packs.filter(
    (pack) => pack.kind !== "jlens",
  );
  await assert.rejects(
    () => prepareCatalog(noJlensSpec, runtimeLock, root, validateSemantics),
    /must include an existing compatible J-lens pack/u,
  );
  const semanticCallCount = semanticCalls.length;
  const missingExactLimit = structuredClone(spec);
  delete missingExactLimit.models[0].requirements.limits.maxComputeWorkgroupSizeX;
  await assert.rejects(
    () => prepareCatalog(missingExactLimit, runtimeLock, root, validateSemantics),
    /maxComputeWorkgroupSizeX must explicitly declare at least 256/,
  );
  const lowExactLimit = structuredClone(spec);
  lowExactLimit.models[0].requirements.limits.maxComputeInvocationsPerWorkgroup = 255;
  await assert.rejects(
    () => prepareCatalog(lowExactLimit, runtimeLock, root, validateSemantics),
    /maxComputeInvocationsPerWorkgroup must explicitly declare at least 256/,
  );
  assert.equal(
    semanticCalls.length,
    semanticCallCount,
    "exact-readout WebGPU requirements must fail before pack validation",
  );
  assert.deepEqual(
    prepareRequirements({
      features: [],
      limits: { maxStorageBuffersPerShaderStage: 10 },
    }, "legacy", "standard-v2"),
    {
      features: [],
      limits: { maxStorageBuffersPerShaderStage: 10 },
    },
  );

  const server = await createServer({
    root: join(import.meta.dirname, ".."),
    configFile: false,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true, watch: null },
  });
  try {
    const { validateCatalogDocument } = await server.ssrLoadModule("/src/lib/runtime/catalog.ts");
    assert.doesNotThrow(() => validateCatalogDocument(catalog));
    const baseSpec = structuredClone(spec);
    baseSpec.models[0].modelType = "base";
    await assert.rejects(
      () => prepareCatalog(baseSpec, runtimeLock, root, validateSemantics),
      /model type differs from its runtime lock/,
    );
    const originalHosted = await readFile(join(modelDirectory, "hosted-artifacts.json"), "utf8");
    const baseConfig = JSON.parse(configContents);
    baseConfig.drowse_completion_prefix_token_ids = [];
    baseConfig.conv_template = { name: "drowse-base", stop_token_ids: [2] };
    const baseConfigContents = JSON.stringify(baseConfig);
    const baseBuild = JSON.parse(buildContents);
    baseBuild.modelType = "base";
    baseBuild.promptPolicy = { mode: "raw", prefixTokenIds: [], stopTokenIds: [2] };
    baseBuild.files = baseBuild.files.map(file => file.path === "mlc-chat-config.json"
      ? (({path, bytes, sha256}) => ({path, bytes, sha256}))(record(file.path, undefined, baseConfigContents)) : file);
    const baseBuildContents = JSON.stringify(baseBuild);
    const baseHosted = JSON.parse(originalHosted);
    baseHosted.modelType = "base";
    baseHosted.files = baseHosted.files.map(file => file.path === "mlc-chat-config.json"
      ? record(file.path, file.role, baseConfigContents) : file.path === "drowse-build.json"
      ? record(file.path, file.role, baseBuildContents) : file);
    await writeFile(join(modelDirectory, "mlc-chat-config.json"), baseConfigContents);
    await writeFile(join(modelDirectory, "drowse-build.json"), baseBuildContents);
    await writeFile(join(modelDirectory, "hosted-artifacts.json"), JSON.stringify(baseHosted));
    model.modelType = "base";
    const partialInstruments = await prepareCatalog(baseSpec, runtimeLock, root, validateSemantics);
    assert.doesNotThrow(() => validateCatalogDocument(partialInstruments));
    baseSpec.models[0].packs.find((pack) => pack.kind === "sae").compatibleContextTokens = [2048, 4096];
    const baseCatalog = await prepareCatalog(baseSpec, runtimeLock, root, validateSemantics);
    assert.equal(validateCatalogDocument(baseCatalog).models[0].modelType, "base");
    baseSpec.models[0].packs = baseSpec.models[0].packs.filter((pack) => pack.kind !== "sae");
    const withoutSae = await prepareCatalog(baseSpec, runtimeLock, root, validateSemantics);
    assert.doesNotThrow(() => validateCatalogDocument(withoutSae));
    baseSpec.models[0].packs = baseSpec.models[0].packs.filter((pack) => pack.kind === "core");
    const generationOnly = await prepareCatalog(baseSpec, runtimeLock, root, validateSemantics);
    assert.doesNotThrow(() => validateCatalogDocument(generationOnly));
    delete model.modelType;
    await writeFile(join(modelDirectory, "mlc-chat-config.json"), configContents);
    await writeFile(join(modelDirectory, "drowse-build.json"), buildContents);
    await writeFile(join(modelDirectory, "hosted-artifacts.json"), originalHosted);
  } finally {
    await server.close();
  }

  const wrongCore = structuredClone(spec);
  wrongCore.models[0].packs[0].sourceRepository = "logitsml/drowse-web-wrong";
  await assert.rejects(
    () => prepareCatalog(wrongCore, runtimeLock, root, validateSemantics),
    /must share the immutable model revision/,
  );
  const wrongInstrument = structuredClone(spec);
  wrongInstrument.models[0].packs[1].sourceRepository = "logitsml/drowse-web-other-instruments";
  await assert.rejects(
    () => prepareCatalog(wrongInstrument, runtimeLock, root, validateSemantics),
    /must use the provisioned instrument repository/,
  );
  const incompleteContexts = structuredClone(spec);
  incompleteContexts.models[0].contextProfiles.pop();
  await assert.rejects(
    () => prepareCatalog(incompleteContexts, runtimeLock, root, validateSemantics),
    /do not close the runtime lock/,
  );
  await writeFile(join(coreDirectory, "undeclared.bin"), "undeclared");
  await assert.rejects(
    () => validatePackDirectory(coreDirectory),
    /does not exactly close the pack directory/,
  );
  await rm(join(coreDirectory, "undeclared.bin"));
  await writeFile(join(coreDirectory, core[0].path), "changed");
  await assert.rejects(() => validatePackDirectory(coreDirectory), /size differs|digest differs/);
} finally {
  await rm(root, { recursive: true, force: true });
}

console.log("hosted catalog builder checks passed");

async function artifactSet(directory, entries) {
  const files = [];
  for (const [path, role, contents] of entries) {
    await writeFile(join(directory, path), contents);
    files.push(record(path, role, contents));
  }
  return files;
}

async function packArtifactSet(directory, entries) {
  const files = [];
  for (const [path, contents] of entries) {
    await writeFile(join(directory, path), contents);
    files.push(({ path, bytes: Buffer.byteLength(contents), sha256: digest(contents) }));
  }
  await writeFile(join(directory, "hosted-pack-artifacts.json"), JSON.stringify({
    schemaVersion: 1,
    files,
  }));
  return files;
}

function record(path, role, contents) {
  return { path, role, bytes: Buffer.byteLength(contents), sha256: digest(contents) };
}

function role(files, value) {
  return files.find((file) => file.role === value);
}

function sum(files) {
  return files.reduce((total, file) => total + file.bytes, 0);
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}
