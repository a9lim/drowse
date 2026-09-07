#!/usr/bin/env node

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, readdir, stat } from "node:fs/promises";
import { resolve } from "node:path";
import { createServer } from "vite";
import { readRuntimeLock } from "./runtime-lock-document.mjs";

const options = parseArguments(process.argv.slice(2));
if (!options) {
  throw new Error(
    "usage: validate-hosted-core-pack.mjs MODEL_ID PACK_DIR [--runtime-lock PATH]",
  );
}
const { modelId, input } = options;
const webuiRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(webuiRoot, "..");
const directory = resolve(input);
if (!(await stat(directory)).isDirectory()) throw new Error("core pack path is not a directory");
const runtimeLockPath = options.runtimeLock
  ? resolve(options.runtimeLock)
  : resolve(repositoryRoot, "browser-runtime/runtime-lock.json");
const { value: runtimeLock } = await readRuntimeLock(runtimeLockPath);
const runtimeLockSource = options.runtimeLock ? "override" : "repository";
const lock = runtimeLock.models.find((entry) => entry.id === modelId);
if (!lock) throw new Error(`unknown runtime-lock model ${modelId}`);
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
try {
  const [{
    contextBindingSha256,
    runtimeIdentitySha256,
  }, {
    decodeFp32Safetensors,
  }, {
    validateDrowseArchive,
  }, {
    BrowserFeasibilityCorePackCompiler,
  }] = await Promise.all([
    vite.ssrLoadModule("/src/lib/runtime/catalog.ts"),
    vite.ssrLoadModule("/src/hosted/artifacts/safetensors.ts"),
    vite.ssrLoadModule("/src/hosted/artifacts/drowseArchive.ts"),
    vite.ssrLoadModule("/src/hosted/runtime/browserCorePack.ts"),
  ]);
  const identitySha256 = runtimeIdentitySha256(runtimeIdentity);
  let tensorSha256 = null;
  const coreArtifacts = [];
  for (const contextTokens of lock.contextProfiles) {
    const contextDirectory = resolve(directory, String(contextTokens));
    const [tensorBytes, sidecarBytes] = await Promise.all([
      readFile(resolve(contextDirectory, "neutral-whitener.safetensors")),
      readFile(resolve(contextDirectory, "neutral-whitener.json")),
    ]);
    const currentTensorSha256 = sha256(tensorBytes);
    tensorSha256 ??= currentTensorSha256;
    assert.equal(currentTensorSha256, tensorSha256, "context whiteners must share exact tensors");
    const sidecar = JSON.parse(sidecarBytes);
    assert.deepEqual(Object.keys(sidecar).sort(), [
      "context_binding_sha256",
      "format_version",
      "hidden_size",
      "layer_map",
      "ridge_per_layer",
      "runtime_identity_sha256",
      "tensors_sha256",
    ]);
    assert.equal(sidecar.format_version, 1);
    assert.equal(sidecar.runtime_identity_sha256, identitySha256);
    assert.equal(
      sidecar.context_binding_sha256,
      contextBindingSha256(identitySha256, contextTokens),
    );
    assert.equal(sidecar.hidden_size, lock.hiddenSize);
    assert.deepEqual(sidecar.layer_map, lock.layerMap);
    assert.equal(sidecar.tensors_sha256, currentTensorSha256);
    assert.deepEqual(Object.keys(sidecar.ridge_per_layer).map(Number), lock.layerMap);
    assert.ok(
      Object.values(sidecar.ridge_per_layer).every(
        (value) => typeof value === "number" && Number.isFinite(value) && value > 0,
      ),
    );
    const tensors = decodeFp32Safetensors(new Uint8Array(tensorBytes), "neutral-whitener.safetensors");
    const expectedKeys = [];
    for (const layer of lock.layerMap) {
      const prefix = `layer_${layer}`;
      const basisShape = tensors.description.shapes.get(`${prefix}.basis`);
      assert.ok(basisShape && basisShape.length === 2 && basisShape[1] === lock.hiddenSize);
      const rank = basisShape[0];
      expectedKeys.push(
        `${prefix}.basis`,
        `${prefix}.eigenvalues`,
        `${prefix}.inverse_scales`,
        `${prefix}.mean`,
      );
      tensors.tensor(`${prefix}.mean`, [lock.hiddenSize]);
      const eigenvalues = tensors.tensor(`${prefix}.eigenvalues`, [rank]).data;
      const inverseScales = tensors.tensor(`${prefix}.inverse_scales`, [rank]).data;
      assert.ok(eigenvalues.every((value) => value > 0));
      assert.ok(inverseScales.every((value) => value > 0));
    }
    assert.deepEqual(tensors.keys, expectedKeys.sort());
    coreArtifacts.push(
      artifact(`${contextTokens}/neutral-whitener.safetensors`, tensorBytes),
      artifact(`${contextTokens}/neutral-whitener.json`, sidecarBytes),
    );
  }
  const manifoldDirectory = resolve(directory, "manifolds");
  const archives = (await readdir(manifoldDirectory)).filter((name) => name.endsWith(".drowse"));
  assert.equal(archives.length, 1, "core pack must contain exactly one Drowse manifold archive");
  const archiveBytes = await readFile(resolve(manifoldDirectory, archives[0]));
  const verified = await validateDrowseArchive(new Blob([archiveBytes]));
  assert.equal(verified.fittedArtifacts.length, 1);
  assert.equal(verified.fittedArtifacts[0].modelId, lock.sourceRepository);
  assert.equal(verified.fittedArtifacts[0].modelFingerprint, identitySha256);
  assert.equal(verified.fittedArtifacts[0].variant, "raw");
  coreArtifacts.push(artifact(`manifolds/${archives[0]}`, archiveBytes));
  const contextProfiles = lock.contextProfiles.map((contextTokens) => ({
    contextTokens,
    bindingSha256: contextBindingSha256(identitySha256, contextTokens),
  }));
  const requiredCorePack = {
    files: coreArtifacts.map((entry) => entry.manifest),
  };
  for (const contextTokens of lock.contextProfiles) {
    const compiler = await BrowserFeasibilityCorePackCompiler.load({
      model: { id: modelId },
      variant: {
        id: `${modelId}-${lock.quantization}`,
        structuredHookProfile: lock.structuredHookProfile,
        thinkingProfile: lock.thinkingProfile,
        runtimeIdentity,
        runtimeIdentitySha256: identitySha256,
        contextProfiles,
      },
      requiredCorePack,
      contextTokens,
      artifacts: coreArtifacts,
      optionalPacks: [],
      signal: new AbortController().signal,
    });
    assert.equal(compiler.fittingWhiteners().size, lock.layerMap.length);
  }
  process.stdout.write(`${JSON.stringify({
    modelId,
    runtimeLockSource,
    identitySha256,
    tensorSha256,
    manifoldSha256: sha256(archiveBytes),
    manifold: verified.primaryIdentity.join("/"),
    contexts: lock.contextProfiles,
    layers: lock.layerMap.length,
  }, null, 2)}\n`);
} finally {
  await vite.close();
}

function parseArguments(args) {
  const positional = [];
  let runtimeLock = null;
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument !== "--runtime-lock") {
      positional.push(argument);
      continue;
    }
    const value = args[index + 1];
    if (!value || value.startsWith("--")) return null;
    runtimeLock = value;
    index += 1;
  }
  if (positional.length !== 2) return null;
  return { modelId: positional[0], input: positional[1], runtimeLock };
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function artifact(path, bytes) {
  const manifest = {
    path,
    role: "core_pack",
    url: `https://huggingface.co/drowse/local/resolve/${"a".repeat(40)}/${path}`,
    revision: "a".repeat(40),
    bytes: bytes.byteLength,
    sha256: sha256(bytes),
  };
  return {
    manifest,
    file: new File([bytes], path.split("/").at(-1)),
  };
}
