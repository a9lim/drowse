#!/usr/bin/env node

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { createServer } from "vite";
import { validatePackDirectory } from "./build-hosted-catalog.mjs";
import { validateReleaseInstrumentPack } from "./hosted-instrument-release-policy.mjs";

const arguments_ = process.argv.slice(2);
const values = [];
let release = false;
let runtimeLockInput = null;
for (let index = 0; index < arguments_.length; index += 1) {
  const argument = arguments_[index];
  if (argument === "--release") {
    if (release) throw new Error("--release may be supplied only once");
    release = true;
    continue;
  }
  if (argument === "--runtime-lock") {
    if (runtimeLockInput !== null) {
      throw new Error("--runtime-lock may be supplied only once");
    }
    const value = arguments_[++index];
    if (!value || value.startsWith("--")) {
      throw new Error("--runtime-lock requires a path");
    }
    runtimeLockInput = value;
    continue;
  }
  if (argument.startsWith("--")) throw new Error(`unexpected argument ${argument}`);
  values.push(argument);
}
const [modelId, kind, input] = values;
if (!modelId || !["sae", "jlens"].includes(kind) || !input || values.length !== 3) {
  throw new Error(
    "usage: validate-hosted-instrument-pack.mjs MODEL_ID sae|jlens PACK_DIR " +
    "[--release] [--runtime-lock PATH]",
  );
}

const webuiRoot = resolve(import.meta.dirname, "..");
const repositoryRoot = resolve(webuiRoot, "..");
const directory = resolve(input);
const runtimeLockPath = runtimeLockInput === null
  ? resolve(repositoryRoot, "browser-runtime/runtime-lock.json")
  : resolve(runtimeLockInput);
const runtimeLock = JSON.parse(
  await readFile(runtimeLockPath, "utf8"),
);
const lock = runtimeLock.models.find((entry) => entry.id === modelId);
if (!lock) throw new Error(`unknown runtime-lock model ${modelId}`);
const manifest = await validatePackDirectory(directory);
const prefix = `${kind}/`;
if (
  manifest.files.some((file) => !file.path.includes("/") || !file.path.includes(prefix)) ||
  manifest.files.filter((file) => file.path.endsWith(`/${kind}/manifest.json`)).length !== 1
) {
  throw new Error(`${kind} pack files must share a nested ${kind}/ closure with one manifest.json`);
}

const revision = "a".repeat(40);
const files = manifest.files.map((file) => ({
  ...file,
  role: "instrument",
  revision,
  url: `https://huggingface.co/drowse/local/resolve/${revision}/${file.path}`,
}));
const artifacts = await Promise.all(files.map(async (entry) => ({
  manifest: entry,
  file: new File([await readFile(resolve(directory, entry.path))], entry.path),
})));
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
let releaseQuality = null;
if (release) {
  const instrumentManifest = JSON.parse(await readFile(resolve(
    directory,
    manifest.files.find((file) => file.path.endsWith(`/${kind}/manifest.json`)).path,
  ), "utf8"));
  const vocabularyEntry = kind === "jlens"
    ? manifest.files.find((file) => file.path.endsWith("/jlens/browser-vocabulary.json"))
    : null;
  releaseQuality = await validateReleaseInstrumentPack({
    directory,
    kind,
    instrumentManifest,
    vocabularyManifest: vocabularyEntry === null || vocabularyEntry === undefined
      ? null
      : JSON.parse(await readFile(resolve(directory, vocabularyEntry.path), "utf8")),
    runtimeLock,
    modelLock: lock,
  });
}
const vite = await createServer({
  root: webuiRoot,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const [{ BrowserInstrumentRegistry }, { runtimeIdentitySha256 }] = await Promise.all([
    vite.ssrLoadModule("/src/hosted/runtime/browserInstrumentPacks.ts"),
    vite.ssrLoadModule("/src/lib/runtime/catalog.ts"),
  ]);
  const fingerprint = runtimeIdentitySha256(runtimeIdentity);
  const registry = await BrowserInstrumentRegistry.load({
    model: { id: modelId },
    variant: {
      id: `${modelId}-${lock.quantization}`,
      runtimeIdentity,
      runtimeIdentitySha256: fingerprint,
    },
    requiredCorePack: { files: [] },
    contextTokens: lock.contextProfiles[0],
    adapter: {},
    artifacts: [],
    optionalPacks: [{
      pack: {
        id: `${modelId}-${kind}`,
        kind,
        displayName: `${modelId} ${kind}`,
        runtimeIdentitySha256: fingerprint,
        files,
      },
      artifacts,
    }],
    activationSpool: {},
    signal: new AbortController().signal,
    onDeviceLost() {},
  });
  const descriptor = registry.descriptor();
  if (kind === "sae") {
    assert.equal(registry.capabilities().sae, true);
    assert.ok(descriptor.sae?.features > 0);
    registry.saeFeature("0");
    registry.saeFeature(String(descriptor.sae.features - 1));
  } else {
    assert.equal(registry.capabilities().jlens, true);
    assert.ok(descriptor.jlens?.words.length > 0);
    registry.jlensToken(descriptor.jlens.words[0].word);
  }
  process.stdout.write(`${JSON.stringify({
    modelId,
    kind,
    runtimeLockSource: runtimeLockInput === null ? "repository" : "override",
    files: files.length,
    bytes: files.reduce((sum, file) => sum + file.bytes, 0),
    closureSha256: digest(files.map(({ path, bytes, sha256 }) => ({ path, bytes, sha256 }))),
    releaseQuality,
    descriptor: kind === "sae" ? descriptor.sae : descriptor.jlens,
  }, null, 2)}\n`);
} finally {
  await vite.close();
}

function digest(value) {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}
