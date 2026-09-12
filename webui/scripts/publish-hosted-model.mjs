#!/usr/bin/env node

import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { createReadStream } from "node:fs";
import { lstat, opendir, readFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual, promisify } from "node:util";
import {
  sameThinkingProfile,
  structuredHookProfile,
  thinkingProfile,
} from "./hosted-model-profiles.mjs";
import { uploadCommitRevision } from "./hugging-face-upload.mjs";
import { loadExpectedBuildToolchain } from "./hosted-model-closure.mjs";

const exec = promisify(execFile);
const runtimeLockUrl = new URL("../../browser-runtime/runtime-lock.json", import.meta.url);
const allowedSidecars = new Set([
  "hosted-artifacts.json",
]);

export function parseArguments(args) {
  const upload = args.includes("--upload");
  const values = args.filter((value) => value !== "--upload");
  if (values.length !== 2 || args.some((value) => value.startsWith("--") && value !== "--upload")) {
    throw new Error("usage: node publish-hosted-model.mjs MODEL_ID DIRECTORY [--upload]");
  }
  return { modelId: values[0], directory: resolve(values[1]), upload };
}

export async function validatePublishDirectory(directory, model, runtime, {
  allowLegacyBranding = false, allowStaleToolchain = false,
} = {}) {
  await requireDirectory(directory);
  const manifest = JSON.parse(await readFile(resolve(directory, "hosted-artifacts.json"), "utf8"));
  if (
    manifest.schemaVersion !== 1 || !Array.isArray(manifest.files) || manifest.files.length === 0 ||
    !runtime || typeof runtime.runtimeAbi !== "string" || typeof runtime.hookAbi !== "string"
  ) {
    throw new Error("hosted-artifacts.json is invalid");
  }
  const manifestKeys = [
    "architecture", "contextWindowSize", "files", "hiddenSize", "hookAbi", "layerMap",
    "prefillChunkSize", "quantization", "runtimeAbi", "schemaVersion", "source",
    "structuredHookProfile", "thinkingProfile",
    ...("modelType" in manifest ? ["modelType"] : []),
  ];
  const hookProfile = structuredHookProfile(manifest.structuredHookProfile);
  const legacyBrand = allowLegacyBranding && runtime.runtimeAbi === "drowse-web-runtime-v1"
    ? ["saklas", "polythetic"].find((brand) => manifest.runtimeAbi === `${brand}-web-runtime-v1`)
    : undefined;
  const expectedRuntime = legacyBrand === undefined ? runtime : { ...runtime, runtimeAbi: manifest.runtimeAbi };
  thinkingProfile(manifest.thinkingProfile);
  if (
    Object.keys(manifest).sort().join("\0") !== manifestKeys.sort().join("\0") ||
    manifest.runtimeAbi !== expectedRuntime.runtimeAbi || manifest.hookAbi !== runtime.hookAbi ||
    hookProfile !== model.structuredHookProfile ||
    (manifest.modelType ?? "chat") !== (model.modelType ?? "chat") ||
    !sameThinkingProfile(manifest.thinkingProfile, model.thinkingProfile)
  ) {
    throw new Error("hosted model ABI differs from the runtime lock");
  }
  const contextProfiles = model.contextProfiles;
  if (
    !Array.isArray(contextProfiles) || contextProfiles.length === 0 ||
    contextProfiles.some((value) => !Number.isSafeInteger(value) || value < 1) ||
    manifest.contextWindowSize !== Math.max(...contextProfiles) ||
    !Number.isSafeInteger(manifest.prefillChunkSize) || manifest.prefillChunkSize < 1 ||
    manifest.prefillChunkSize > manifest.contextWindowSize
  ) {
    throw new Error("hosted model context configuration differs from the runtime lock");
  }
  if (
    new Set(manifest.files.map((file) => file.path)).size !== manifest.files.length ||
    new Set(manifest.files.map((file) => `${file.role}:${file.path}`)).size !== manifest.files.length
  ) {
    throw new Error("hosted-artifacts.json repeats an artifact identity");
  }
  const identities = new Map(manifest.files.map((file) => [`${file.role}:${file.path}`, file]));
  const expected = [
    ["converted_manifest:tensor-cache.json", model.manifestSha256],
    ["model_library:model.wasm", model.librarySha256],
    ["tokenizer:tokenizer.json", model.tokenizerSha256],
    ["chat_template:tokenizer_config.json", model.chatTemplateSha256],
  ];
  for (const [key, digest] of expected) {
    if (!digest || identities.get(key)?.sha256 !== digest) {
      throw new Error(`runtime lock identity differs for ${key}`);
    }
  }
  if (
    manifest.architecture !== model.architecture ||
    manifest.quantization !== model.quantization ||
    manifest.hiddenSize !== model.hiddenSize ||
    JSON.stringify(manifest.layerMap) !== JSON.stringify(model.layerMap) ||
    manifest.source?.repository !== model.sourceRepository ||
    manifest.source?.revision !== model.sourceRevision
  ) {
    throw new Error("hosted model metadata differs from the runtime lock");
  }
  await validateRuntimeMetadata(directory, manifest, model, expectedRuntime, legacyBrand ?? "drowse", allowStaleToolchain);
  const declared = new Set(manifest.files.map((file) => file.path));
  const entries = [];
  for await (const entry of await opendir(directory)) entries.push(entry);
  for (const entry of entries) {
    const path = resolve(directory, entry.name);
    const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink()) {
      throw new Error(`publish directory contains a non-regular entry: ${entry.name}`);
    }
    if (!declared.has(entry.name) && !allowedSidecars.has(entry.name)) {
      throw new Error(`publish directory contains an undeclared file: ${entry.name}`);
    }
  }
  for (const file of manifest.files) {
    if (basename(file.path) !== file.path || !/^[0-9a-f]{64}$/.test(file.sha256)) {
      throw new Error(`invalid artifact manifest entry: ${file.path}`);
    }
    const path = resolve(directory, file.path);
    const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink() || info.size !== file.bytes) {
      throw new Error(`artifact size differs for ${file.path}`);
    }
    if (await sha256(path) !== file.sha256) throw new Error(`artifact digest differs for ${file.path}`);
  }
  return manifest;
}

async function validateRuntimeMetadata(directory, manifest, model, runtime, brand, allowStaleToolchain) {
  const [config, tensorCache, build] = await Promise.all([
    readFile(resolve(directory, "mlc-chat-config.json"), "utf8").then(JSON.parse),
    readFile(resolve(directory, "tensor-cache.json"), "utf8").then(JSON.parse),
    readFile(resolve(directory, `${brand}-build.json`), "utf8").then(JSON.parse),
  ]);
  const configModel = config.model_config;
  const configModelShape = configModel?.text_config ?? configModel;
  if (
    config.model_type !== model.architecture || config.quantization !== model.quantization ||
    config.context_window_size !== manifest.contextWindowSize ||
    config.prefill_chunk_size !== manifest.prefillChunkSize ||
    configModel?.context_window_size !== manifest.contextWindowSize ||
    configModel?.prefill_chunk_size !== manifest.prefillChunkSize ||
    configModelShape?.hidden_size !== model.hiddenSize ||
    configModelShape?.num_hidden_layers !== model.layerMap.length
  ) {
    throw new Error("MLC runtime configuration differs from the hosted manifest");
  }
  if (
    build.schemaVersion !== 1 || build.runtimeAbi !== runtime.runtimeAbi ||
    build.hookAbi !== runtime.hookAbi || build.architecture !== model.architecture ||
    build.structuredHookProfile !== manifest.structuredHookProfile ||
    !sameThinkingProfile(build.thinkingProfile, manifest.thinkingProfile) ||
    build.quantization !== model.quantization ||
    (build.modelType ?? "chat") !== (model.modelType ?? "chat") ||
    build.contextWindowSize !== manifest.contextWindowSize ||
    build.prefillChunkSize !== manifest.prefillChunkSize ||
    build.source?.repository !== model.sourceRepository ||
    build.source?.revision !== model.sourceRevision
  ) {
    throw new Error("Drowse build metadata differs from the runtime lock");
  }
  if (!allowStaleToolchain) {
    const expected = await loadExpectedBuildToolchain(runtime.toolchain?.forkManifestSha256, model.architecture);
    if (!isDeepStrictEqual(build.toolchain, expected)) {
      throw new Error("Drowse build toolchain is stale or differs from the runtime lock");
    }
  }
  if (model.modelType === "base") {
    const prefix = config.drowse_completion_prefix_token_ids;
    if (
      build.promptPolicy?.mode !== "raw" || !Array.isArray(prefix) ||
      prefix.some((token) => !Number.isSafeInteger(token) || token < 0) ||
      JSON.stringify(prefix) !== JSON.stringify(build.promptPolicy?.prefixTokenIds) ||
      JSON.stringify(config.conv_template?.stop_token_ids) !== JSON.stringify(build.promptPolicy?.stopTokenIds) ||
      config.conv_template?.name !== "drowse-base" ||
      (model.architecture === "qwen3_5" && build.stateAbi !== "kv-rnn-v1")
    ) throw new Error("base model completion or hybrid-state policy differs from its build");
  }
  if (!Array.isArray(tensorCache.records) || tensorCache.records.length === 0) {
    throw new Error("tensor-cache.json has no weight records");
  }
  const weights = new Map(
    manifest.files.filter((file) => file.role === "weight").map((file) => [file.path, file]),
  );
  const weightRecords = tensorCache.records.map((record) => record?.dataPath);
  if (
    weights.size !== tensorCache.records.length ||
    new Set(weightRecords).size !== weightRecords.length ||
    model.layerMap.some((layer, index) => layer !== index) ||
    tensorCache.records.some((record) =>
      typeof record?.dataPath !== "string" || basename(record.dataPath) !== record.dataPath ||
      !weights.has(record.dataPath) ||
      weights.get(record.dataPath).bytes !== record.nbytes
    )
  ) {
    throw new Error("tensor-cache.json does not close the published weight shards");
  }
  if (tensorCache.metadata?.ParamBytes !== undefined) {
    const dtypeBytes = { float16: 2, float32: 4, int32: 4, uint32: 4 };
    const parameterBytes = tensorCache.records.reduce((sum, shard) => sum +
      (shard.records ?? []).reduce((subtotal, tensor) => subtotal +
        (dtypeBytes[tensor.dtype] ?? NaN) *
          (tensor.shape ?? []).reduce((size, dimension) => size * dimension, 1), 0), 0);
    if (tensorCache.metadata.ParamBytes !== parameterBytes) {
      throw new Error("tensor-cache.json parameter byte count is stale");
    }
  }
  for (const record of tensorCache.records) {
    if (record.md5sum !== undefined && (
      !/^[0-9a-f]{32}$/.test(record.md5sum) ||
      await fileDigest(resolve(directory, record.dataPath), "md5") !== record.md5sum
    )) throw new Error(`tensor-cache.json shard checksum differs for ${record.dataPath}`);
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const runtimeLock = JSON.parse(await readFile(runtimeLockUrl, "utf8"));
  const model = runtimeLock.models?.find((candidate) => candidate.id === options.modelId);
  if (!model) throw new Error(`unknown runtime model: ${options.modelId}`);
  const manifest = await validatePublishDirectory(options.directory, model, runtimeLock);
  const info = await modelInfo(model.convertedRepository);
  if (info.private === true || info.gated === true || info.gated === "auto" || info.gated === "manual") {
    throw new Error(`hosted repository must be public and ungated: ${model.convertedRepository}`);
  }
  if (!options.upload) {
    console.log(JSON.stringify({
      checked: model.convertedRepository,
      files: manifest.files.length,
      bytes: manifest.files.reduce((sum, file) => sum + file.bytes, 0),
      currentRevision: info.sha,
    }));
    return;
  }
  const uploaded = await exec("hf", [
    "upload",
    model.convertedRepository,
    options.directory,
    ".",
    "--type",
    "model",
    "--commit-message",
    `publish ${options.modelId} browser runtime`,
    "--format",
    "json",
  ], { encoding: "utf8" });
  const revision = uploadCommitRevision(uploaded.stdout, model.convertedRepository);
  console.log(JSON.stringify({ repository: model.convertedRepository, revision }));
}

async function modelInfo(repository) {
  const response = await exec("hf", [
    "models", "info", repository,
    "--expand", "sha,private,gated", "--format", "json",
  ], { encoding: "utf8" });
  return JSON.parse(response.stdout);
}

async function sha256(path) {
  return fileDigest(path, "sha256");
}

async function fileDigest(path, algorithm) {
  const digest = createHash(algorithm);
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest("hex");
}

async function requireDirectory(path) {
  const info = await lstat(path);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("publish source is not a regular directory");
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
