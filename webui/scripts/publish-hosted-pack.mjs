#!/usr/bin/env node

import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { validatePackDirectory } from "./build-hosted-catalog.mjs";
import { uploadCommitRevision } from "./hugging-face-upload.mjs";

const exec = promisify(execFile);
const runtimeLockUrl = new URL("../../browser-runtime/runtime-lock.json", import.meta.url);
const coreValidatorUrl = new URL("./validate-hosted-core-pack.mjs", import.meta.url);
const instrumentValidatorUrl = new URL("./validate-hosted-instrument-pack.mjs", import.meta.url);

export function parseArguments(args) {
  const upload = args.includes("--upload");
  const values = args.filter((value) => value !== "--upload");
  if (
    values.length !== 3 ||
    !["core", "sae", "jlens"].includes(values[1]) ||
    args.some((value) => value.startsWith("--") && value !== "--upload")
  ) {
    throw new Error("usage: node publish-hosted-pack.mjs MODEL_ID core|sae|jlens DIRECTORY [--upload]");
  }
  return {
    modelId: values[0],
    kind: values[1],
    directory: resolve(values[2]),
    upload,
  };
}

export function targetRepository(model, kind) {
  if (!model?.convertedRepository) throw new Error("runtime model has no converted repository");
  return kind === "core"
    ? model.convertedRepository
    : `${model.convertedRepository}-instruments`;
}

export function semanticValidationArguments(modelId, kind, directory) {
  return kind === "core"
    ? [fileURLToPath(coreValidatorUrl), modelId, directory]
    : [
        fileURLToPath(instrumentValidatorUrl),
        modelId,
        kind,
        directory,
      ];
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const runtimeLock = JSON.parse(await readFile(runtimeLockUrl, "utf8"));
  const model = runtimeLock.models?.find((candidate) => candidate.id === options.modelId);
  if (!model) throw new Error(`unknown runtime model: ${options.modelId}`);
  const manifest = await validatePackDirectory(options.directory);
  await exec(process.execPath, semanticValidationArguments(
    options.modelId,
    options.kind,
    options.directory,
  ));
  const repository = targetRepository(model, options.kind);
  const info = await modelInfo(repository);
  if (info.private === true || info.gated === true || info.gated === "auto" || info.gated === "manual") {
    throw new Error(`hosted repository must be public and ungated: ${repository}`);
  }
  if (!options.upload) {
    console.log(JSON.stringify({
      checked: repository,
      kind: options.kind,
      files: manifest.files.length,
      bytes: manifest.files.reduce((sum, file) => sum + file.bytes, 0),
      currentRevision: info.sha,
    }));
    return;
  }
  const uploaded = await exec("hf", [
    "upload",
    repository,
    options.directory,
    ".",
    "--type",
    "model",
    "--commit-message",
    `publish ${options.modelId} ${options.kind} browser pack`,
    "--format",
    "json",
  ], { encoding: "utf8" });
  const revision = uploadCommitRevision(uploaded.stdout, repository);
  console.log(JSON.stringify({ repository, revision, kind: options.kind }));
}

async function modelInfo(repository) {
  const response = await exec("hf", [
    "models", "info", repository,
    "--expand", "sha,private,gated", "--format", "json",
  ], { encoding: "utf8" });
  return JSON.parse(response.stdout);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
