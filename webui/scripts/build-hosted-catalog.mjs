#!/usr/bin/env node

import { createHash, randomUUID } from "node:crypto";
import { execFile } from "node:child_process";
import { createReadStream } from "node:fs";
import {
  lstat,
  mkdir,
  opendir,
  readFile,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { dirname, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { createServer } from "vite";
import { validatePublishDirectory } from "./publish-hosted-model.mjs";

const runtimeLockUrl = new URL("../../browser-runtime/runtime-lock.json", import.meta.url);
const webuiRoot = fileURLToPath(new URL("..", import.meta.url));
const COMMIT = /^[0-9a-f]{40}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const PACK_MANIFEST = "hosted-pack-artifacts.json";
const EXACT_READOUT_REQUIRED_WEBGPU_LIMITS = Object.freeze({
  maxComputeWorkgroupSizeX: 256,
  maxComputeInvocationsPerWorkgroup: 256,
});
const exec = promisify(execFile);
const coreValidatorUrl = new URL("./validate-hosted-core-pack.mjs", import.meta.url);
const instrumentValidatorUrl = new URL("./validate-hosted-instrument-pack.mjs", import.meta.url);

export function parseArguments(args) {
  if (args.length !== 2 || args.some((value) => value.startsWith("--"))) {
    throw new Error("usage: node build-hosted-catalog.mjs RELEASE_SPEC OUTPUT");
  }
  return { specPath: resolve(args[0]), outputPath: resolve(args[1]) };
}

export async function prepareCatalog(
  spec,
  runtimeLock,
  specDirectory,
  semanticValidator = validatePackSemantics,
) {
  exactKeys(spec, [
    "schemaVersion", "sequence", "issuedAt", "expiresAt", "models",
  ], "release spec");
  if (spec.schemaVersion !== 1 || !positiveInteger(spec.sequence)) {
    throw new Error("release spec metadata is invalid");
  }
  const issuedAt = date(spec.issuedAt, "issuedAt");
  const expiresAt = date(spec.expiresAt, "expiresAt");
  if (Date.parse(expiresAt) <= Date.parse(issuedAt)) {
    throw new Error("release spec expiry must follow its issue time");
  }
  if (!Array.isArray(spec.models) || spec.models.length === 0) {
    throw new Error("release spec must contain models");
  }
  const models = [];
  for (const entry of spec.models) {
    models.push(await prepareModel(entry, runtimeLock, specDirectory, semanticValidator));
  }
  if (new Set(models.map((model) => model.id)).size !== models.length) {
    throw new Error("release spec repeats a model id");
  }
  const document = {
    schemaVersion: 1,
    sequence: spec.sequence,
    issuedAt,
    expiresAt,
    runtimeAbi: nonempty(runtimeLock.runtimeAbi, "runtime ABI"),
    models,
  };
  return document;
}

async function prepareModel(entry, runtimeLock, specDirectory, semanticValidator) {
  exactKeys(entry, [
    "id", "variantId", "tier", "displayName", "description", "license",
    "languages", "modelDirectory", "contextProfiles", "requirements", "packs",
    ...("modelType" in entry ? ["modelType"] : []),
  ], "release model");
  const lock = runtimeLock.models?.find((candidate) => candidate.id === entry.id);
  if (!lock) throw new Error(`unknown runtime model ${String(entry.id)}`);
  if ((entry.modelType ?? "chat") !== (lock.modelType ?? "chat")) {
    throw new Error(`${lock.id} model type differs from its runtime lock`);
  }
  if (!COMMIT.test(lock.convertedRevision ?? "")) {
    throw new Error(`${lock.id} has no immutable converted revision`);
  }
  const modelDirectory = resolvePortable(specDirectory, entry.modelDirectory, `${lock.id} modelDirectory`);
  const modelManifest = await validatePublishDirectory(modelDirectory, lock, runtimeLock);
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
  const runtimeIdentitySha256 = digestCanonical(runtimeIdentity);
  const contextProfiles = prepareContextProfiles(entry.contextProfiles, lock, runtimeIdentitySha256);
  const requirements = prepareRequirements(
    entry.requirements,
    lock.id,
    modelManifest.structuredHookProfile,
  );
  const baseUrl = `https://huggingface.co/${lock.convertedRepository}/resolve/${lock.convertedRevision}/`;
  const hostedManifestPath = resolve(modelDirectory, "hosted-artifacts.json");
  const hostedManifestInfo = await lstat(hostedManifestPath);
  if (!hostedManifestInfo.isFile() || hostedManifestInfo.isSymbolicLink()) {
    throw new Error(`${lock.id} hosted-artifacts.json is not a regular file`);
  }
  const files = [
    ...modelManifest.files,
    {
      path: "hosted-artifacts.json",
      role: "configuration",
      bytes: hostedManifestInfo.size,
      sha256: await sha256(hostedManifestPath),
    },
  ].map((file) => catalogFile(
    file,
    lock.convertedRevision,
    baseUrl,
  ));
  if (!Array.isArray(entry.packs) || entry.packs.length === 0) {
    throw new Error(`${lock.id} has no required core pack`);
  }
  const packs = [];
  for (const pack of entry.packs) {
    packs.push(await preparePack(
      pack,
      specDirectory,
      lock,
      runtimeIdentitySha256,
      contextProfiles,
      semanticValidator,
    ));
  }
  const requiredCore = packs.filter((pack) => pack.kind === "core" && pack.required);
  if (requiredCore.length !== 1) {
    throw new Error(`${lock.id} must have exactly one required core pack`);
  }
  if (entry.modelType !== "base" && !packs.some((pack) => pack.kind === "jlens")) {
    throw new Error(`${lock.id} must include an existing compatible J-lens pack`);
  }
  if (entry.modelType !== undefined && !["chat", "base"].includes(entry.modelType)) {
    throw new Error(`${lock.id} modelType must be chat or base`);
  }
  for (const profile of contextProfiles) {
    const sae = packs.filter((pack) =>
      pack.kind === "sae" &&
      pack.compatibleContextBindingSha256.includes(profile.bindingSha256)
    );
    if (sae.length > 1) {
      throw new Error(
        `${lock.id} context ${profile.contextTokens} has multiple compatible SAE packs`,
      );
    }
  }
  return {
    id: nonempty(entry.id, `${lock.id} id`),
    ...(entry.modelType !== undefined ? { modelType: entry.modelType } : {}),
    displayName: nonempty(entry.displayName, `${lock.id} displayName`),
    description: text(entry.description, `${lock.id} description`),
    sourceUrl: `https://huggingface.co/${lock.sourceRepository}`,
    license: nonempty(entry.license, `${lock.id} license`),
    languages: stringList(entry.languages, `${lock.id} languages`),
    variants: [{
      id: nonempty(entry.variantId, `${lock.id} variantId`),
      tier: tier(entry.tier, lock.id),
      structuredHookProfile: modelManifest.structuredHookProfile,
      thinkingProfile: modelManifest.thinkingProfile,
      contextProfiles,
      downloadBytes: sumBytes(files),
      requiredCorePackBytes: requiredCore[0].bytes,
      requirements,
      runtimeIdentity,
      runtimeIdentitySha256,
      files,
      packs,
    }],
  };
}

function prepareContextProfiles(value, lock, runtimeIdentitySha256) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${lock.id} has no context profiles`);
  }
  const profiles = value.map((profile) => {
    exactKeys(profile, [
      "contextTokens", "minimumCalibrationScore", "minimumDeviceMemoryGiB",
      "expectedPrefillTokensPerSecond", "expectedDecodeTokensPerSecond", "measuredDevices",
    ], `${lock.id} context profile`);
    if (!lock.contextProfiles.includes(profile.contextTokens)) {
      throw new Error(`${lock.id} context ${String(profile.contextTokens)} is not runtime-locked`);
    }
    return {
      ...profile,
      bindingSha256: digestCanonical({
        contextTokens: profile.contextTokens,
        runtimeIdentitySha256,
      }),
    };
  });
  if (
    new Set(profiles.map((profile) => profile.contextTokens)).size !== profiles.length ||
    profiles.length !== lock.contextProfiles.length ||
    lock.contextProfiles.some((tokens) => !profiles.some((profile) => profile.contextTokens === tokens))
  ) {
    throw new Error(`${lock.id} context profiles do not close the runtime lock`);
  }
  return profiles;
}

async function preparePack(
  entry,
  specDirectory,
  lock,
  runtimeIdentitySha256,
  contextProfiles,
  semanticValidator,
) {
  exactKeys(entry, [
    "id", "kind", "displayName", "license", "directory", "sourceRepository",
    "sourceRevision", "required", "compatibleContextTokens",
  ], `${lock.id} pack`);
  if (!["core", "jlens", "sae"].includes(entry.kind)) {
    throw new Error(`${lock.id} pack kind is invalid`);
  }
  if (typeof entry.required !== "boolean" || (entry.kind === "core") !== entry.required) {
    throw new Error(`${lock.id} core packs must be required and instruments optional`);
  }
  if (
    entry.required &&
    (entry.sourceRepository !== lock.convertedRepository || entry.sourceRevision !== lock.convertedRevision)
  ) {
    throw new Error(`${lock.id} required core pack must share the immutable model revision`);
  }
  if (!entry.required && entry.sourceRepository !== `${lock.convertedRepository}-instruments`) {
    throw new Error(`${lock.id} optional instruments must use the provisioned instrument repository`);
  }
  if (!COMMIT.test(entry.sourceRevision ?? "")) {
    throw new Error(`${lock.id} pack has no immutable source revision`);
  }
  const directory = resolvePortable(specDirectory, entry.directory, `${lock.id} pack directory`);
  const manifest = await validatePackDirectory(directory);
  await semanticValidator(lock.id, entry.kind, directory);
  const baseUrl = `https://huggingface.co/${entry.sourceRepository}/resolve/${entry.sourceRevision}/`;
  const files = manifest.files.map((file) => catalogFile(
    { ...file, role: entry.kind === "core" ? "core_pack" : "instrument" },
    entry.sourceRevision,
    baseUrl,
  ));
  const compatibleContextBindingSha256 = contextTokens(entry.compatibleContextTokens, lock.id)
    .map((tokens) => {
      const profile = contextProfiles.find((candidate) => candidate.contextTokens === tokens);
      if (!profile) throw new Error(`${lock.id} pack names an unknown context ${tokens}`);
      return profile.bindingSha256;
    });
  return {
    id: nonempty(entry.id, `${lock.id} pack id`),
    kind: entry.kind,
    displayName: nonempty(entry.displayName, `${lock.id} pack displayName`),
    license: nonempty(entry.license, `${lock.id} pack license`),
    sourceRepository: nonempty(entry.sourceRepository, `${lock.id} pack sourceRepository`),
    sourceRevision: entry.sourceRevision,
    bytes: sumBytes(files),
    required: entry.required,
    runtimeIdentitySha256,
    compatibleContextBindingSha256,
    files,
  };
}

export async function validatePackSemantics(modelId, kind, directory, runner = exec) {
  const args = kind === "core"
    ? [fileURLToPath(coreValidatorUrl), modelId, directory]
    : [fileURLToPath(instrumentValidatorUrl), modelId, kind, directory];
  try {
    await runner(process.execPath, args, { encoding: "utf8" });
  } catch (error) {
    throw new Error(`${modelId} ${kind} pack failed semantic validation`, { cause: error });
  }
}

export async function validatePackDirectory(directory) {
  await requireDirectory(directory, "pack directory");
  const manifestInfo = await lstat(resolve(directory, PACK_MANIFEST));
  if (!manifestInfo.isFile() || manifestInfo.isSymbolicLink()) {
    throw new Error(`${PACK_MANIFEST} is not a regular file`);
  }
  const manifest = JSON.parse(await readFile(resolve(directory, PACK_MANIFEST), "utf8"));
  exactKeys(manifest, ["schemaVersion", "files"], PACK_MANIFEST);
  if (manifest.schemaVersion !== 1 || !Array.isArray(manifest.files) || manifest.files.length === 0) {
    throw new Error(`${PACK_MANIFEST} is invalid`);
  }
  const files = [];
  for (const file of manifest.files) {
    exactKeys(file, ["path", "bytes", "sha256"], "pack artifact");
    const path = portablePath(file.path, "pack artifact path");
    if (!positiveInteger(file.bytes)) throw new Error(`pack artifact byte count is invalid for ${path}`);
    const absolute = resolve(directory, path);
    const info = await lstat(absolute);
    if (!info.isFile() || info.isSymbolicLink() || info.size !== file.bytes) {
      throw new Error(`pack artifact size differs for ${path}`);
    }
    const digest = await sha256(absolute);
    if (!SHA256.test(file.sha256 ?? "") || digest !== file.sha256) {
      throw new Error(`pack artifact digest differs for ${path}`);
    }
    files.push({ path, bytes: file.bytes, sha256: file.sha256 });
  }
  if (new Set(files.map((file) => file.path)).size !== files.length) {
    throw new Error(`${PACK_MANIFEST} repeats a file path`);
  }
  const actual = [];
  await collectRegularFiles(directory, directory, actual);
  actual.sort((left, right) => left.localeCompare(right));
  const declared = files.map((file) => file.path).sort((left, right) => left.localeCompare(right));
  if (
    actual.length !== declared.length ||
    actual.some((path, index) => path !== declared[index])
  ) throw new Error(`${PACK_MANIFEST} does not exactly close the pack directory`);
  return { schemaVersion: 1, files };
}

async function collectRegularFiles(root, directory, result) {
  for await (const entry of await opendir(directory)) {
    if (directory === root && entry.name === PACK_MANIFEST) continue;
    const absolute = resolve(directory, entry.name);
    const info = await lstat(absolute);
    if (info.isSymbolicLink()) throw new Error(`pack directory contains a symbolic link: ${entry.name}`);
    if (info.isDirectory()) {
      await collectRegularFiles(root, absolute, result);
      continue;
    }
    if (!info.isFile()) throw new Error(`pack directory contains a special file: ${entry.name}`);
    result.push(relative(root, absolute).split(sep).join("/"));
  }
}

function catalogFile(file, revision, baseUrl) {
  const path = portablePath(file.path, "catalog file path");
  return {
    path,
    role: file.role,
    url: `${baseUrl}${path.split("/").map(encodeURIComponent).join("/")}`,
    revision,
    bytes: file.bytes,
    sha256: file.sha256,
  };
}

export function prepareRequirements(value, modelId, structuredHookProfile) {
  exactKeys(value, ["features", "limits"], `${modelId} requirements`);
  if (!Array.isArray(value.features) || value.features.some((entry) => typeof entry !== "string")) {
    throw new Error(`${modelId} required features are invalid`);
  }
  if (!value.limits || typeof value.limits !== "object" || Array.isArray(value.limits)) {
    throw new Error(`${modelId} required limits are invalid`);
  }
  if (structuredHookProfile === "standard-v3") {
    for (const [name, minimum] of Object.entries(
      EXACT_READOUT_REQUIRED_WEBGPU_LIMITS,
    )) {
      if (
        !Object.hasOwn(value.limits, name) ||
        !Number.isSafeInteger(value.limits[name]) ||
        value.limits[name] < minimum
      ) {
        throw new Error(
          `${modelId} ${name} must explicitly declare at least ${minimum} ` +
            "for standard-v3 exact GPU readout",
        );
      }
    }
  }
  return { features: [...value.features], limits: { ...value.limits } };
}

async function validateDocument(document) {
  const server = await createServer({
    root: webuiRoot,
    configFile: false,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true, watch: null },
  });
  try {
    const { validateCatalogDocument } = await server.ssrLoadModule("/src/lib/runtime/catalog.ts");
    validateCatalogDocument(document);
  } finally {
    await server.close();
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  await requireMissing(options.outputPath, "catalog output");
  const [runtimeLock, spec] = await Promise.all([
    readFile(runtimeLockUrl, "utf8").then(JSON.parse),
    readFile(options.specPath, "utf8").then(JSON.parse),
  ]);
  const document = await prepareCatalog(spec, runtimeLock, dirname(options.specPath));
  await validateDocument(document);
  const stage = `${options.outputPath}.partial-${randomUUID()}`;
  await mkdir(dirname(options.outputPath), { recursive: true });
  try {
    await writeFile(stage, `${JSON.stringify(document, null, 2)}\n`, { flag: "wx" });
    await rename(stage, options.outputPath);
  } catch (error) {
    await rm(stage, { force: true });
    throw error;
  }
  console.log(JSON.stringify({
    output: options.outputPath,
    sequence: document.sequence,
    models: document.models.length,
    files: document.models.flatMap((model) => model.variants.flatMap((variant) => [
      ...variant.files,
      ...variant.packs.flatMap((pack) => pack.files),
    ])).length,
  }));
}

function digestCanonical(value) {
  return createHash("sha256").update(canonicalJson(value)).digest("hex");
}

function canonicalJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.entries(value)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
    .join(",")}}`;
}

function exactKeys(value, expected, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    throw new Error(`${label} has unknown or missing fields`);
  }
}

function portablePath(value, label) {
  const result = nonempty(value, label);
  const parts = result.split("/");
  if (
    result.startsWith("/") || result.includes("\\") || result.includes("\0") ||
    /^[A-Za-z]:/.test(result) || parts.some((part) => !part || part === "." || part === "..")
  ) throw new Error(`${label} is unsafe`);
  return result;
}

function resolvePortable(root, value, label) {
  const relative = portablePath(value, label);
  const result = resolve(root, relative);
  const prefix = root.endsWith("/") ? root : `${root}/`;
  if (!result.startsWith(prefix)) throw new Error(`${label} escapes the release spec directory`);
  return result;
}

function contextTokens(value, label) {
  if (
    !Array.isArray(value) || value.length === 0 ||
    value.some((entry) => !positiveInteger(entry)) || new Set(value).size !== value.length
  ) throw new Error(`${label} pack context list is invalid`);
  return value;
}

function stringList(value, label) {
  if (!Array.isArray(value) || value.length === 0 || value.some((entry) => !nonempty(entry, label))) {
    throw new Error(`${label} must be a non-empty string list`);
  }
  return [...value];
}

function tier(value, label) {
  if (!["fastest", "balanced", "quality"].includes(value)) {
    throw new Error(`${label} tier is invalid`);
  }
  return value;
}

function sumBytes(files) {
  const total = files.reduce((sum, file) => sum + file.bytes, 0);
  if (!Number.isSafeInteger(total) || total < 1) throw new Error("catalog byte total is invalid");
  return total;
}

function positiveInteger(value) {
  return Number.isSafeInteger(value) && value > 0;
}

function text(value, label) {
  if (typeof value !== "string") throw new Error(`${label} must be a string`);
  return value;
}

function nonempty(value, label) {
  const result = text(value, label);
  if (!result) throw new Error(`${label} must not be empty`);
  return result;
}

function date(value, label) {
  const result = nonempty(value, label);
  if (!Number.isFinite(Date.parse(result))) throw new Error(`${label} is invalid`);
  return result;
}

async function sha256(path) {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest("hex");
}

async function requireDirectory(path, label) {
  const info = await lstat(path);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error(`${label} is not a regular directory`);
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

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
