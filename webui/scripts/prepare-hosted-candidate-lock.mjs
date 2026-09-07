#!/usr/bin/env node

import { randomUUID } from "node:crypto";
import {
  access,
  link,
  mkdir,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { validatePublishDirectory } from "./publish-hosted-model.mjs";
import {
  readRuntimeLock,
  validateRuntimeLock,
} from "./runtime-lock-document.mjs";

const repositoryRuntimeLockPath = fileURLToPath(
  new URL("../../browser-runtime/runtime-lock.json", import.meta.url),
);

export function parseArguments(args) {
  if (args.length !== 3 || args.some((argument) => argument.startsWith("--"))) {
    throw new Error(
      "usage: node prepare-hosted-candidate-lock.mjs MODEL_ID MODEL_DIR OUTPUT.json",
    );
  }
  return {
    modelId: args[0],
    modelDirectory: resolve(args[1]),
    outputPath: resolve(args[2]),
  };
}

export function deriveCandidateRuntimeLock(runtimeLock, modelId, manifest) {
  validateRuntimeLock(runtimeLock);
  const modelIndex = runtimeLock.models.findIndex((model) => model.id === modelId);
  if (modelIndex < 0) throw new Error(`unknown runtime-lock model ${modelId}`);
  if (!manifest || typeof manifest !== "object" || !Array.isArray(manifest.files)) {
    throw new Error("hosted-artifacts.json is invalid");
  }
  const libraries = manifest.files.filter((file) => file?.role === "model_library");
  if (
    libraries.length !== 1 || libraries[0].path !== "model.wasm" ||
    typeof libraries[0].sha256 !== "string" ||
    !/^[0-9a-f]{64}$/.test(libraries[0].sha256)
  ) throw new Error("hosted-artifacts.json has no unique model.wasm library digest");
  const candidate = structuredClone(runtimeLock);
  candidate.models[modelIndex].librarySha256 = libraries[0].sha256;
  validateRuntimeLock(candidate);
  return candidate;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  await requireMissing(options.outputPath);
  const { value: runtimeLock } = await readRuntimeLock(repositoryRuntimeLockPath);
  const manifestPath = resolve(options.modelDirectory, "hosted-artifacts.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  const candidate = deriveCandidateRuntimeLock(
    runtimeLock,
    options.modelId,
    manifest,
  );
  const model = candidate.models.find((entry) => entry.id === options.modelId);
  await validatePublishDirectory(options.modelDirectory, model, candidate);
  const bytes = `${JSON.stringify(candidate, null, 2)}\n`;
  const temporary = `${options.outputPath}.partial-${randomUUID()}`;
  await mkdir(dirname(options.outputPath), { recursive: true });
  try {
    await writeFile(temporary, bytes, { flag: "wx" });
    await link(temporary, options.outputPath);
    await rm(temporary);
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
  const original = runtimeLock.models.find((entry) => entry.id === options.modelId);
  console.log(JSON.stringify({
    modelId: options.modelId,
    modelDirectory: options.modelDirectory,
    outputPath: options.outputPath,
    repositoryLibrarySha256: original.librarySha256,
    candidateLibrarySha256: model.librarySha256,
    changed: original.librarySha256 !== model.librarySha256,
  }));
}

async function requireMissing(path) {
  try {
    await access(path);
  } catch (error) {
    if (error?.code === "ENOENT") return;
    throw error;
  }
  throw new Error(`candidate runtime-lock output already exists: ${path}`);
}

if (
  process.argv[1] &&
  pathToFileURL(process.argv[1]).href === import.meta.url
) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
