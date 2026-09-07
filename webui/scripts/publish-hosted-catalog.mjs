#!/usr/bin/env node

import { execFile } from "node:child_process";
import { createPublicKey, verify } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { createServer } from "vite";
import { uploadCommitRevision } from "./hugging-face-upload.mjs";

export { uploadCommitRevision } from "./hugging-face-upload.mjs";

const exec = promisify(execFile);
const webuiRoot = fileURLToPath(new URL("..", import.meta.url));
const runtimeLockUrl = new URL("../../browser-runtime/runtime-lock.json", import.meta.url);
const distributionLockUrl = new URL("../../browser-runtime/distribution-lock.json", import.meta.url);

export function parseArguments(args) {
  const upload = args.includes("--upload");
  const values = args.filter((value) => value !== "--upload");
  if (values.length !== 2 || args.some((value) => value.startsWith("--") && value !== "--upload")) {
    throw new Error("usage: node publish-hosted-catalog.mjs CATALOG SIGNATURE [--upload]");
  }
  return {
    catalogPath: resolve(values[0]),
    signaturePath: resolve(values[1]),
    upload,
  };
}

export function catalogRepository(distribution) {
  const catalog = repositoryFromUrl(distribution.catalogUrl, "catalogUrl", "catalog.json");
  const signature = repositoryFromUrl(
    distribution.signatureUrl,
    "signatureUrl",
    "catalog.sig.json",
  );
  if (catalog !== signature) throw new Error("catalog and signature must publish to one repository");
  return catalog;
}

export async function validateCatalogBundle(
  catalogPath,
  signaturePath,
  distribution,
  runtimeLockModels,
) {
  if (basename(catalogPath) !== "catalog.json" || basename(signaturePath) !== "catalog.sig.json") {
    throw new Error("catalog inputs must use their canonical published filenames");
  }
  const [exactBytes, signatureBytes] = await Promise.all([
    readFile(catalogPath),
    readFile(signaturePath),
  ]);
  let signature;
  try {
    signature = JSON.parse(signatureBytes.toString("utf8"));
  } catch (error) {
    throw new Error("catalog signature is not valid JSON", { cause: error });
  }
  const keys = new Map(distribution.publicKeys.map((entry) => [
    entry.keyId,
    Uint8Array.from(Buffer.from(entry.ed25519PublicKeyBase64, "base64")),
  ]));
  const server = await createServer({
    root: webuiRoot,
    configFile: false,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true, watch: null },
  });
  try {
    const { verifyCatalog, WebCryptoEd25519Verifier, runtimeIdentitySha256 } =
      await server.ssrLoadModule("/src/lib/runtime/catalog.ts");
    const verified = await verifyCatalog(
      new Uint8Array(exactBytes),
      signature,
      new WebCryptoEd25519Verifier(keys),
      {
        now: Date.now(),
        online: true,
        minimumSequence: distribution.minimumAcceptedSequence,
        lastAcceptedSequence: null,
        expectedRuntimeAbi: runtimeLockModels[0]?.runtimeIdentity.runtimeAbi,
        runtimeLockModels,
      },
    );
    if (verified.document.sequence < distribution.minimumAcceptedSequence) {
      throw new Error("catalog sequence is below the embedded rollback floor");
    }
    const publishedIdentities = new Set(verified.document.models.flatMap((model) =>
      model.variants.map((variant) => variant.runtimeIdentitySha256)
    ));
    const missingModels = runtimeLockModels.filter((model) =>
      !publishedIdentities.has(runtimeIdentitySha256(model.runtimeIdentity))
    );
    if (missingModels.length > 0) {
      throw new Error(`catalog is missing locked models: ${missingModels.map((model) =>
        model.runtimeIdentity.sourceModel
      ).join(", ")}`);
    }
    return {
      exactBytes,
      signatureBytes,
      sequence: verified.document.sequence,
      models: verified.document.models.length,
      keyId: signature.keyId,
    };
  } finally {
    await server.close();
  }
}

export function lockedCatalogModels(runtime) {
  if (!Array.isArray(runtime.models) || runtime.models.length === 0) {
    throw new Error("runtime lock has no catalog models");
  }
  return runtime.models.map((model) => {
    const identity = {
      sourceModel: model.sourceRepository,
      sourceRevision: model.sourceRevision,
      convertedManifestSha256: model.manifestSha256,
      quantization: model.quantization,
      tokenizerSha256: model.tokenizerSha256,
      chatTemplateSha256: model.chatTemplateSha256,
      modelLibrarySha256: model.librarySha256,
      runtimeAbi: runtime.runtimeAbi,
      hookAbi: runtime.hookAbi,
      hiddenSize: model.hiddenSize,
      layerMap: model.layerMap,
    };
    if (
      !/^[0-9a-f]{40}$/.test(model.convertedRevision ?? "") ||
      Object.values(identity).some((value) => value == null)
    ) throw new Error(`${model.id} has unresolved catalog publication identities`);
    return {
      ...(model.modelType !== undefined ? { modelType: model.modelType } : {}),
      runtimeIdentity: identity,
      structuredHookProfile: model.structuredHookProfile,
      thinkingProfile: model.thinkingProfile,
      convertedRepository: model.convertedRepository,
      convertedRevision: model.convertedRevision,
      contextProfiles: model.contextProfiles,
    };
  });
}

export function validateCatalogAdvance(nextSequence, currentBytes, signatureBytes, publicKeys) {
  if (currentBytes === null && signatureBytes === null) return null;
  if (currentBytes === null || signatureBytes === null) {
    throw new Error("published catalog and signature must both exist");
  }
  const signature = JSON.parse(signatureBytes.toString("utf8"));
  const trustedKey = publicKeys.find((entry) => entry.keyId === signature.keyId);
  if (!trustedKey || signature.schemaVersion !== 1 || signature.algorithm !== "Ed25519") {
    throw new Error("published catalog signing key or envelope is not trusted");
  }
  const publicKey = createPublicKey({
    key: {
      kty: "OKP",
      crv: "Ed25519",
      x: Buffer.from(trustedKey.ed25519PublicKeyBase64, "base64").toString("base64url"),
    },
    format: "jwk",
  });
  if (!verify(null, currentBytes, publicKey, Buffer.from(signature.signature, "base64"))) {
    throw new Error("published catalog signature is invalid");
  }
  const { sequence } = JSON.parse(currentBytes.toString("utf8"));
  if (!Number.isSafeInteger(sequence) || sequence < 1) {
    throw new Error("published catalog sequence is invalid");
  }
  if (!Number.isSafeInteger(nextSequence) || nextSequence <= sequence) {
    throw new Error(`catalog sequence ${nextSequence} must advance the published sequence ${sequence}`);
  }
  return sequence;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const [runtime, distribution] = await Promise.all([
    readFile(runtimeLockUrl, "utf8").then(JSON.parse),
    readFile(distributionLockUrl, "utf8").then(JSON.parse),
  ]);
  const repository = catalogRepository(distribution);
  const bundle = await validateCatalogBundle(
    options.catalogPath,
    options.signaturePath,
    distribution,
    lockedCatalogModels(runtime),
  );
  const info = await modelInfo(repository);
  if (info.private === true || info.gated === true || info.gated === "auto" || info.gated === "manual") {
    throw new Error(`hosted catalog repository must be public and ungated: ${repository}`);
  }
  if (!/^[0-9a-f]{40}$/.test(info.sha ?? "")) {
    throw new Error("catalog repository has no immutable revision");
  }
  const currentBase = `https://huggingface.co/${repository}/resolve/${info.sha}/`;
  const [currentCatalog, currentSignature] = await Promise.all([
    fetchExact(`${currentBase}catalog.json`, true),
    fetchExact(`${currentBase}catalog.sig.json`, true),
  ]);
  const previousSequence = validateCatalogAdvance(
    bundle.sequence, currentCatalog, currentSignature, distribution.publicKeys,
  );
  if (!options.upload) {
    console.log(JSON.stringify({
      checked: repository,
      sequence: bundle.sequence,
      models: bundle.models,
      keyId: bundle.keyId,
      currentRevision: info.sha,
      previousSequence,
    }));
    return;
  }
  if (runtime.status !== "verified") {
    throw new Error("catalog upload requires a verified browser runtime lock");
  }
  const stage = await mkdtemp(join(tmpdir(), "drowse-catalog-publish-"));
  try {
    await Promise.all([
      writeFile(join(stage, "catalog.json"), bundle.exactBytes, { flag: "wx" }),
      writeFile(join(stage, "catalog.sig.json"), bundle.signatureBytes, { flag: "wx" }),
    ]);
    if ((await modelInfo(repository)).sha !== info.sha) {
      throw new Error("catalog repository changed during preflight; verify the new release before retrying");
    }
    const uploaded = await exec("hf", [
      "upload",
      repository,
      stage,
      ".",
      "--type",
      "model",
      "--commit-message",
      `publish signed browser catalog sequence ${bundle.sequence}`,
      "--format",
      "json",
    ], { encoding: "utf8" });
    const publishedRevision = uploadCommitRevision(uploaded.stdout, repository);
    const base = `https://huggingface.co/${repository}/resolve/${publishedRevision}/`;
    const [publishedCatalog, publishedSignature] = await Promise.all([
      fetchExact(`${base}catalog.json`),
      fetchExact(`${base}catalog.sig.json`),
    ]);
    if (
      !publishedCatalog.equals(bundle.exactBytes) ||
      !publishedSignature.equals(bundle.signatureBytes)
    ) throw new Error("published catalog transaction differs from the signed local pair");
    console.log(JSON.stringify({
      repository,
      revision: publishedRevision,
      sequence: bundle.sequence,
      previousSequence,
    }));
  } finally {
    await rm(stage, { recursive: true, force: true });
  }
}

function repositoryFromUrl(value, label, filename) {
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`${label} is invalid`);
  }
  const parts = url.pathname.split("/").filter(Boolean);
  if (
    url.origin !== "https://huggingface.co" || url.username || url.password ||
    parts.length !== 5 || parts[2] !== "resolve" || parts[3] !== "main" ||
    parts[4] !== filename
  ) {
    throw new Error(`${label} is not a supported Hugging Face main-branch URL`);
  }
  return `${parts[0]}/${parts[1]}`;
}

async function modelInfo(repository) {
  const response = await exec("hf", [
    "models", "info", repository,
    "--expand", "sha,private,gated", "--format", "json",
  ], { encoding: "utf8" });
  return JSON.parse(response.stdout);
}

async function fetchExact(url, allowMissing = false) {
  const response = await fetch(url, { cache: "no-store", credentials: "omit" });
  if (allowMissing && response.status === 404) return null;
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
  return Buffer.from(await response.arrayBuffer());
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
