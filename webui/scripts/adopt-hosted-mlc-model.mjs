#!/usr/bin/env node

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
import { basename, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expectedBuildToolchain } from "./hosted-model-closure.mjs";

const runtimeLockUrl = new URL("../../browser-runtime/runtime-lock.json", import.meta.url);
const forkManifestUrl = new URL("../../browser-runtime/forks/manifest.json", import.meta.url);
const SHA256 = /^[0-9a-f]{64}$/u;
const COMMIT = /^[0-9a-f]{40}$/u;

export function parseArguments(args) {
  if (args.length !== 6 || args.some((value) => value.startsWith("--"))) {
    throw new Error(
      "usage: node adopt-hosted-mlc-model.mjs MODEL_ID UPSTREAM_REPOSITORY " +
        "UPSTREAM_REVISION MLC_DIR SOURCE_METADATA_DIR OUTPUT_DIR",
    );
  }
  return {
    modelId: args[0],
    upstreamRepository: args[1],
    upstreamRevision: args[2],
    mlcDirectory: resolve(args[3]),
    sourceDirectory: resolve(args[4]),
    outputDirectory: resolve(args[5]),
  };
}

export function productionMlcConfig(config, model, tokenizer = null) {
  const result = structuredClone(config);
  const shape = result.model_config?.text_config ?? result.model_config;
  const compatibleArchitecture = result.model_type === model.architecture ||
    (model.architecture === "gemma3_text" && result.model_type === "gemma3");
  if (
    !compatibleArchitecture ||
    result.quantization !== model.quantization ||
    shape?.hidden_size !== model.hiddenSize ||
    shape?.num_hidden_layers !== model.layerMap.length
  ) {
    throw new Error("upstream MLC configuration differs from the runtime lock");
  }
  result.model_type = model.architecture;
  result.context_window_size = Math.max(...model.contextProfiles);
  result.prefill_chunk_size = Math.min(...model.contextProfiles);
  result.model_config.context_window_size = result.context_window_size;
  result.model_config.prefill_chunk_size = result.prefill_chunk_size;
  result.model_config.max_batch_size = 1;
  if (result.model_config.text_config) {
    result.model_config.is_text_model = true;
    result.model_config.text_config.context_window_size = result.context_window_size;
    result.model_config.text_config.prefill_chunk_size = result.prefill_chunk_size;
    result.model_config.text_config.max_batch_size = 1;
  }
  if (tokenizer !== null) {
    result.active_vocab_size = activeVocabSize(tokenizer, result.vocab_size);
  }
  return result;
}

export function activeVocabSize(tokenizer, modelVocabSize) {
  const ids = [];
  const vocabulary = tokenizer?.model?.vocab;
  if (Array.isArray(vocabulary)) {
    for (let id = 0; id < vocabulary.length; id += 1) ids.push(id);
  } else if (vocabulary && typeof vocabulary === "object") {
    for (const id of Object.values(vocabulary)) {
      if (Number.isSafeInteger(id) && id >= 0) ids.push(id);
    }
  }
  for (const token of tokenizer?.added_tokens ?? []) {
    if (Number.isSafeInteger(token?.id) && token.id >= 0) ids.push(token.id);
  }
  if (!Number.isSafeInteger(modelVocabSize) || modelVocabSize < 1 || ids.length === 0) {
    throw new Error("tokenizer metadata cannot determine the active vocabulary size");
  }
  let maximumId = 0;
  for (const id of ids) maximumId = Math.max(maximumId, id);
  return Math.min(modelVocabSize, maximumId + 1);
}

export function captureSpecialTokenIds(tokenizer, tokenizerConfig) {
  const ids = new Set();
  for (const token of tokenizer?.added_tokens ?? []) {
    if (Number.isSafeInteger(token?.id) && token.id >= 0) {
      ids.add(token.id);
    }
  }
  for (const [key, token] of Object.entries(tokenizerConfig?.added_tokens_decoder ?? {})) {
    const id = Number(key);
    if (token && Number.isSafeInteger(id) && id >= 0) ids.add(id);
  }
  for (const key of ["bos_token_id", "eos_token_id", "pad_token_id"]) {
    const id = tokenizerConfig?.[key];
    if (Number.isSafeInteger(id) && id >= 0) ids.add(id);
  }
  if (ids.size === 0) throw new Error("tokenizer metadata has no special token IDs");
  return [...ids].sort((left, right) => left - right);
}

export function adoptedBuildManifest({
  model,
  sourceFiles,
  outputFiles,
  sourceTokenizerConfig,
  toolchain,
  upstreamRepository,
  upstreamRevision,
}) {
  const thinkingProfile = model.thinkingProfile === null
    ? null
    : structuredClone(model.thinkingProfile);
  return {
    schemaVersion: 1,
    runtimeAbi: "drowse-web-runtime-v1",
    hookAbi: "post-block-residual-v4",
    structuredHookProfile: model.structuredHookProfile,
    thinkingProfile,
    architecture: model.architecture,
    quantization: model.quantization,
    contextWindowSize: Math.max(...model.contextProfiles),
    prefillChunkSize: Math.min(...model.contextProfiles),
    source: {
      repository: model.sourceRepository,
      revision: model.sourceRevision,
      files: sourceFiles,
      chatTemplateSha256: digestText(String(sourceTokenizerConfig.chat_template ?? "")),
    },
    toolchain,
    files: outputFiles,
  };
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (!COMMIT.test(options.upstreamRevision)) throw new Error("upstream revision must be immutable");
  await requireDirectory(options.mlcDirectory, "MLC model directory");
  await requireDirectory(options.sourceDirectory, "source metadata directory");
  await requireMissing(options.outputDirectory);
  const [runtime, forkManifest] = await Promise.all([
    readJson(fileURLToPath(runtimeLockUrl)),
    readJson(fileURLToPath(forkManifestUrl)),
  ]);
  const model = runtime.models.find((entry) => entry.id === options.modelId);
  if (!model) throw new Error(`unknown runtime model ${options.modelId}`);
  const sourceConfig = await readJson(resolve(options.sourceDirectory, "config.json"));
  const sourceTokenizerConfig = await readJson(resolve(options.sourceDirectory, "tokenizer_config.json"));
  const upstreamConfig = await readJson(resolve(options.mlcDirectory, "mlc-chat-config.json"));
  const tensorCache = await readJson(resolve(options.mlcDirectory, "tensor-cache.json"));
  const tokenizer = await readJson(resolve(options.mlcDirectory, "tokenizer.json"));
  const sourceShape = sourceConfig.text_config ?? sourceConfig;
  const sourceModelType = sourceShape.model_type ?? sourceConfig.model_type;
  if (
    sourceModelType !== model.architecture ||
    sourceShape.hidden_size !== model.hiddenSize ||
    sourceShape.num_hidden_layers !== model.layerMap.length
  ) {
    throw new Error("source model metadata differs from the runtime lock");
  }
  const sourcePaths = ["README.md", "config.json", "tokenizer_config.json"];
  if (await isRegularFile(resolve(options.sourceDirectory, "LICENSE"))) {
    sourcePaths.push("LICENSE");
  }
  await verifyHuggingFaceRevision(options.sourceDirectory, sourcePaths, model.sourceRevision);
  const inputFiles = inputClosure(upstreamConfig, tensorCache);
  await verifyHuggingFaceRevision(options.mlcDirectory, inputFiles, options.upstreamRevision);
  const stage = `${options.outputDirectory}.partial-${randomUUID()}`;
  await mkdir(stage, { recursive: true });
  try {
    for (const path of inputFiles.filter((path) => path !== "mlc-chat-config.json")) {
      await copyFile(resolve(options.mlcDirectory, path), resolve(stage, path));
    }
    await copyFile(
      resolve(options.sourceDirectory, "tokenizer_config.json"),
      resolve(stage, "tokenizer_config.json"),
    );
    const config = productionMlcConfig(upstreamConfig, model, tokenizer);
    config.drowse_capture_special_token_ids = captureSpecialTokenIds(
      tokenizer,
      sourceTokenizerConfig,
    );
    await writeJson(resolve(stage, "mlc-chat-config.json"), config);
    await copyFile(resolve(options.sourceDirectory, "README.md"), resolve(stage, "README.source.md"));
    const license = await sourceLicense(options.sourceDirectory);
    await copyFile(license.path, resolve(stage, "LICENSE.model"));
    await writeJson(resolve(stage, "MODEL-LICENSE.json"), license.record);
    await writeJson(resolve(stage, "CONVERSION-SOURCE.json"), {
      schemaVersion: 1,
      repository: options.upstreamRepository,
      revision: options.upstreamRevision,
    });
    const sourceFiles = await fileRecords(options.sourceDirectory, sourcePaths);
    const outputPaths = [
      ...inputFiles.filter((path) => path !== "mlc-chat-config.json"),
      "mlc-chat-config.json",
      "README.source.md",
      "LICENSE.model",
      "MODEL-LICENSE.json",
      "CONVERSION-SOURCE.json",
    ].sort();
    const build = adoptedBuildManifest({
      model,
      sourceFiles,
      outputFiles: await fileRecords(stage, outputPaths),
      sourceTokenizerConfig,
      toolchain: expectedBuildToolchain(forkManifest),
      upstreamRepository: options.upstreamRepository,
      upstreamRevision: options.upstreamRevision,
    });
    await writeJson(resolve(stage, "drowse-build.json"), build);
    await rename(stage, options.outputDirectory);
  } catch (error) {
    await rm(stage, { recursive: true, force: true });
    throw error;
  }
  console.log(JSON.stringify({
    modelId: model.id,
    output: options.outputDirectory,
    upstreamRepository: options.upstreamRepository,
    upstreamRevision: options.upstreamRevision,
  }));
}

async function sourceLicense(directory) {
  const readme = resolve(directory, "README.md");
  const licenseId = modelCardLicense(await readFile(readme, "utf8"));
  if (licenseId === "apache-2.0") {
    const path = resolve(directory, "LICENSE");
    if (!await isRegularFile(path)) {
      throw new Error("Apache model source has no LICENSE file");
    }
    return {
      path,
      record: {
        schemaVersion: 1,
        spdx: licenseId,
        declaredIn: "README.md",
        licenseText: "LICENSE.model",
        licenseTextSource: "source-snapshot",
      },
    };
  }
  if (licenseId !== "gemma") {
    throw new Error("source model card declares an unsupported model license");
  }
  return {
    path: readme,
    record: {
      schemaVersion: 1,
      licenseId,
      declaredIn: "README.md",
      licenseText: "LICENSE.model",
      licenseTextSource: "model-card-terms-reference",
      termsUrl: "https://ai.google.dev/gemma/terms",
    },
  };
}

export function modelCardLicense(text) {
  const lines = text.split(/\r?\n/u);
  if (lines[0]?.trim() !== "---") return null;
  for (const line of lines.slice(1)) {
    if (line.trim() === "---") break;
    const match = /^license\s*:\s*['"]?([^'"\s]+)['"]?\s*$/u.exec(line);
    if (match) return match[1].toLowerCase();
  }
  return null;
}

function inputClosure(config, tensorCache) {
  if (!Array.isArray(config.tokenizer_files) || !Array.isArray(tensorCache.records)) {
    throw new Error("upstream MLC model has an invalid closure");
  }
  const paths = [
    "mlc-chat-config.json",
    "tensor-cache.json",
    ...config.tokenizer_files,
    ...tensorCache.records.map((record) => record?.dataPath),
  ];
  if (paths.some((path) => typeof path !== "string" || basename(path) !== path)) {
    throw new Error("upstream MLC model contains an unsafe artifact path");
  }
  return [...new Set(paths)].sort();
}

async function verifyHuggingFaceRevision(directory, paths, expectedRevision) {
  for (const path of paths) {
    const metadata = resolve(directory, ".cache", "huggingface", "download", `${path}.metadata`);
    const revision = (await readFile(metadata, "utf8")).split(/\r?\n/u)[0];
    if (revision !== expectedRevision) {
      throw new Error(`${path} came from ${revision}, expected ${expectedRevision}`);
    }
    const info = await lstat(resolve(directory, path));
    if (!info.isFile() || info.isSymbolicLink()) throw new Error(`${path} is not a regular file`);
  }
}

async function fileRecords(directory, paths) {
  return Promise.all(paths.map(async (path) => {
    const bytes = await readFile(resolve(directory, path));
    return { path, bytes: bytes.byteLength, sha256: digestBytes(bytes) };
  }));
}

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

async function writeJson(path, value) {
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, { flag: "wx" });
}

function digestText(value) {
  return digestBytes(new TextEncoder().encode(value));
}

function digestBytes(value) {
  const digest = createHash("sha256").update(value).digest("hex");
  if (!SHA256.test(digest)) throw new Error("SHA-256 failed");
  return digest;
}

async function requireDirectory(path, label) {
  const info = await lstat(path);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error(`${label} is invalid`);
}

async function isRegularFile(path) {
  try {
    const info = await lstat(path);
    return info.isFile() && !info.isSymbolicLink();
  } catch (error) {
    if (error?.code === "ENOENT") return false;
    throw error;
  }
}

async function requireMissing(path) {
  try {
    await lstat(path);
  } catch (error) {
    if (error?.code === "ENOENT") return;
    throw error;
  }
  throw new Error(`output already exists: ${path}`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
