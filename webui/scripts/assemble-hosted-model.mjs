import { createHash, randomUUID } from "node:crypto";
import {
  copyFile,
  lstat,
  mkdir,
  readFile,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";
import {
  structuredHookProfile,
  thinkingProfile,
} from "./hosted-model-profiles.mjs";

const STANDARD_V3_VM_FUNCTIONS = [
  "drowse_hook_profile",
  "drowse_prefill",
  "drowse_decode",
  "drowse_batch_prefill",
  "drowse_batch_decode",
  "drowse_capture_prefill",
  "drowse_capture_decode",
  "drowse_capture_batch_prefill",
  "drowse_capture_batch_decode",
  "drowse_rank_one_capture_prefill_v1",
  "drowse_rank_one_capture_decode_v1",
  "drowse_rank_one_capture_batch_prefill_v1",
  "drowse_rank_one_capture_batch_decode_v1",
  "drowse_structured_batch_prefill",
  "drowse_structured_batch_decode",
  "drowse_geometry_batch_prefill",
  "drowse_geometry_batch_decode",
  "drowse_curved_batch_prefill",
  "drowse_curved_batch_decode",
  "drowse_jlens_probabilities",
  "drowse_jlens_readout_accumulate",
  "drowse_jlens_readout_topk",
  "drowse_jlens_directions",
  "drowse_sae_readout_accumulate",
  "drowse_sae_jump_relu_readout_accumulate",
];
const REQUIRED_PROVENANCE_FILES = [
  "LICENSE.model",
  "MODEL-LICENSE.json",
  "README.source.md",
  "drowse-build.json",
];
const OPTIONAL_PROVENANCE_FILES = ["CONVERSION-SOURCE.json"];

const [convertedArg, libraryArg, outputArg] = process.argv.slice(2);
if (!convertedArg || !libraryArg || !outputArg || process.argv.length !== 5) {
  throw new Error(
    "usage: node assemble-hosted-model.mjs CONVERTED_DIR MODEL_WASM OUTPUT_DIR",
  );
}

const converted = resolve(convertedArg);
const library = resolve(libraryArg);
const output = resolve(outputArg);
await requireDirectory(converted, "converted model");
await requireFile(library, "model library");
await requireMissing(output, "output directory");

const build = parseJson(
  await readFile(resolve(converted, "drowse-build.json"), "utf8"),
  "drowse-build.json",
);
const config = parseJson(
  await readFile(resolve(converted, "mlc-chat-config.json"), "utf8"),
  "mlc-chat-config.json",
);
const tensorCache = parseJson(
  await readFile(resolve(converted, "tensor-cache.json"), "utf8"),
  "tensor-cache.json",
);
const modelConfig = config.model_config?.text_config ?? config.model_config;
const buildFiles = validateBuildManifest(build);
await verifyBuildFiles(converted, buildFiles);
const hookProfile = structuredHookProfile(build.structuredHookProfile);
const artifacts = [
  ...await runtimeArtifacts(
    converted,
    library,
    config,
    tensorCache,
    hookProfile,
  ),
  ...await Promise.all(REQUIRED_PROVENANCE_FILES.map(async (path) => {
    const source = resolve(converted, path);
    await requireFile(source, path);
    return artifact(source, path, "configuration");
  })),
  ...await optionalProvenanceArtifacts(converted),
];
const metadata = {
  schemaVersion: 1,
  runtimeAbi: text(build.runtimeAbi, "runtimeAbi"),
  hookAbi: text(build.hookAbi, "hookAbi"),
  structuredHookProfile: hookProfile,
  thinkingProfile: thinkingProfile(build.thinkingProfile),
  architecture: text(build.architecture, "architecture"),
  ...(build.modelType !== undefined ? { modelType: build.modelType } : {}),
  quantization: text(build.quantization, "quantization"),
  contextWindowSize: positiveInteger(
    build.contextWindowSize,
    "contextWindowSize",
  ),
  prefillChunkSize: positiveInteger(build.prefillChunkSize, "prefillChunkSize"),
  source: sourceIdentity(build.source),
  hiddenSize: positiveInteger(
    modelConfig?.hidden_size,
    "model_config.hidden_size",
  ),
  layerMap: Array.from(
    {
      length: positiveInteger(
        modelConfig?.num_hidden_layers,
        "model_config.num_hidden_layers",
      ),
    },
    (_, index) => index,
  ),
  files: artifacts.map(({ source: _source, ...artifact }) => artifact),
};

const stage = `${output}.partial-${randomUUID()}`;
await mkdir(dirname(output), { recursive: true });
await mkdir(stage);
try {
  for (const artifact of artifacts) {
    await copyFile(artifact.source, resolve(stage, artifact.path));
  }
  await writeFile(
    resolve(stage, "hosted-artifacts.json"),
    `${JSON.stringify(metadata, null, 2)}\n`,
    { flag: "wx" },
  );
  await rename(stage, output);
} catch (error) {
  await rm(stage, { recursive: true, force: true });
  throw error;
}

console.log(
  JSON.stringify({
    output,
    files: artifacts.length,
    bytes: artifacts.reduce((sum, file) => sum + file.bytes, 0),
  }),
);

async function runtimeArtifacts(root, wasm, config, tensorCache, hookProfile) {
  const tokenizerFiles = stringArray(config.tokenizer_files, "tokenizer_files");
  if (
    new Set(tokenizerFiles).size !== tokenizerFiles.length ||
    tokenizerFiles.filter((name) => name === "tokenizer.json").length !== 1 ||
    tokenizerFiles.filter((name) => name === "tokenizer_config.json").length !==
      1
  ) {
    throw new Error(
      "tokenizer_files must contain unique tokenizer.json and tokenizer_config.json entries",
    );
  }
  const records = Array.isArray(tensorCache.records) ? tensorCache.records : [];
  if (records.length === 0)
    throw new Error("tensor-cache.json has no weight records");
  const weights = records.map((record) =>
    portableBasename(record?.dataPath, "tensor shard"),
  );
  if (new Set(weights).size !== weights.length)
    throw new Error("tensor-cache.json repeats a weight shard");
  const files = [
    ["mlc-chat-config.json", "configuration"],
    ["tensor-cache.json", "converted_manifest"],
    ...tokenizerFiles.map((name) => [
      portableBasename(name, "tokenizer support file"),
      name === "tokenizer.json"
        ? "tokenizer"
        : name === "tokenizer_config.json"
          ? "chat_template"
          : "configuration",
    ]),
    ...weights.map((name) => [name, "weight"]),
  ];
  const unique = new Map();
  for (const [path, role] of files) {
    if (unique.has(path) && unique.get(path) !== role)
      throw new Error(`${path} has conflicting runtime roles`);
    unique.set(path, role);
  }
  const result = [];
  for (const [path, role] of unique) {
    const source = resolve(root, path);
    await requireFile(source, path);
    result.push(await artifact(source, path, role));
  }
  const wasmBytes = await readFile(wasm);
  if (!wasmBytes.subarray(0, 4).equals(Buffer.from([0, 97, 115, 109]))) {
    throw new Error("model library is not a WebAssembly module");
  }
  if (hookProfile === "standard-v3") {
    for (const name of STANDARD_V3_VM_FUNCTIONS) {
      const encoded = Buffer.from(name);
      const length = Buffer.alloc(8);
      length.writeBigUInt64LE(BigInt(encoded.byteLength));
      if (!wasmBytes.includes(Buffer.concat([length, encoded]))) {
        throw new Error(
          `standard-v3 model library lacks required VM function ${name}`,
        );
      }
    }
  }
  result.push(await artifact(wasm, "model.wasm", "model_library"));
  return result;
}

async function optionalProvenanceArtifacts(root) {
  const artifacts = [];
  for (const path of OPTIONAL_PROVENANCE_FILES) {
    const source = resolve(root, path);
    try {
      await requireFile(source, path);
    } catch (error) {
      if (error?.code === "ENOENT") continue;
      throw error;
    }
    artifacts.push(await artifact(source, path, "configuration"));
  }
  return artifacts;
}

async function artifact(source, path, role) {
  const bytes = await readFile(source);
  return {
    path,
    role,
    bytes: bytes.byteLength,
    sha256: createHash("sha256").update(bytes).digest("hex"),
    source,
  };
}

function validateBuildManifest(value) {
  if (
    value.schemaVersion !== 1 ||
    !Array.isArray(value.files) ||
    value.files.length === 0
  ) {
    throw new Error("drowse-build.json is invalid");
  }
  return value.files.map((file) => ({
    path: portableBasename(file?.path, "build file"),
    bytes: positiveInteger(file?.bytes, "build file bytes"),
    sha256: sha256(file?.sha256, "build file sha256"),
  }));
}

async function verifyBuildFiles(root, files) {
  for (const expected of files) {
    const path = resolve(root, expected.path);
    await requireFile(path, expected.path);
    const bytes = await readFile(path);
    if (
      bytes.byteLength !== expected.bytes ||
      createHash("sha256").update(bytes).digest("hex") !== expected.sha256
    ) {
      throw new Error(
        `converted build file changed after validation: ${expected.path}`,
      );
    }
  }
}

function sourceIdentity(value) {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("build source identity is invalid");
  return {
    repository: text(value.repository, "source.repository"),
    revision: commit(value.revision, "source.revision"),
    chatTemplateSha256: sha256(
      value.chatTemplateSha256,
      "source.chatTemplateSha256",
    ),
  };
}

function portableBasename(value, label) {
  const name = text(value, label);
  if (
    basename(name) !== name ||
    name === "." ||
    name === ".." ||
    name.includes("\\") ||
    name.includes("\0")
  ) {
    throw new Error(`${label} must be a portable basename`);
  }
  return name;
}

function stringArray(value, label) {
  if (
    !Array.isArray(value) ||
    value.length === 0 ||
    value.some((entry) => typeof entry !== "string")
  ) {
    throw new Error(`${label} must be a non-empty string array`);
  }
  return value;
}

function parseJson(value, label) {
  try {
    return JSON.parse(value);
  } catch (error) {
    throw new Error(`${label} is not valid JSON`, { cause: error });
  }
}

function text(value, label) {
  if (typeof value !== "string" || !value.trim())
    throw new Error(`${label} must be a non-empty string`);
  return value;
}

function positiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 1)
    throw new Error(`${label} must be a positive integer`);
  return value;
}

function sha256(value, label) {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value))
    throw new Error(`${label} must be SHA-256`);
  return value;
}

function commit(value, label) {
  if (typeof value !== "string" || !/^[0-9a-f]{40}$/.test(value))
    throw new Error(`${label} must be a commit SHA`);
  return value;
}

async function requireFile(path, label) {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink())
    throw new Error(`${label} is not a regular file`);
}

async function requireDirectory(path, label) {
  const info = await lstat(path);
  if (!info.isDirectory() || info.isSymbolicLink())
    throw new Error(`${label} is not a directory`);
}

async function requireMissing(path, label) {
  try {
    await lstat(path);
  } catch (error) {
    if (error?.code === "ENOENT") return;
    throw error;
  }
  throw new Error(`${label} already exists`);
}
