#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { lstat, readFile, realpath } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const defaultManifestPath = resolve(scriptDirectory, "manifest.json");
const SHA256 = /^[0-9a-f]{64}$/;
const COMMIT = /^[0-9a-f]{40}$/;
const VERSION = /^\d+\.\d+\.\d+$/;

export function parseArguments(args) {
  const options = { apply: false, help: false, id: null, repository: null };
  for (const argument of args) {
    if (argument === "--apply") options.apply = true;
    else if (argument === "--check") options.apply = false;
    else if (argument === "--help" || argument === "-h") options.help = true;
    else if (options.id === null) options.id = argument;
    else if (options.repository === null) options.repository = argument;
    else throw new Error(`unexpected argument: ${argument}`);
  }
  if (!options.help && (options.id === null || options.repository === null)) {
    throw new Error("usage: apply-fork-overlay.mjs ID REPOSITORY [--check|--apply]");
  }
  return options;
}

export async function loadManifest(path = defaultManifestPath) {
  const manifest = JSON.parse(await readFile(path, "utf8"));
  validateManifest(manifest);
  return manifest;
}

export function validateManifest(manifest) {
  if (manifest?.schemaVersion !== 1 || manifest.status !== "feasibility-overlay") {
    throw new Error("unsupported fork overlay manifest");
  }
  if (manifest.hookAbi !== "post-block-residual-v4") {
    throw new Error("unexpected fork overlay hook ABI");
  }
  if (manifest.structuredHookProfile !== "standard-v3") {
    throw new Error("unexpected fork overlay structured-hook profile");
  }
  if (manifest.exactReadoutAbi !== "exact-readout-v1") {
    throw new Error("unexpected fork overlay exact-readout ABI");
  }
  if (manifest.tvmRepository !== "https://github.com/apache/tvm.git") {
    throw new Error("invalid pinned TVM repository");
  }
  for (const key of ["tvmCommit", "tvmFfiCommit", "emsdkCommit", "emccRevision"]) {
    if (!COMMIT.test(manifest[key])) throw new Error(`invalid pinned ${key}`);
  }
  for (const key of ["emccVersion", "llvmVersion"]) {
    if (!VERSION.test(manifest[key])) throw new Error(`invalid pinned ${key}`);
  }
  if (manifest.emccPlatform !== "linux/amd64") {
    throw new Error("invalid pinned Emscripten platform");
  }
  const llvmArtifacts = new Map(
    Array.isArray(manifest.llvmArtifacts)
      ? manifest.llvmArtifacts.map((artifact) => [artifact?.platform, artifact])
      : [],
  );
  const expectedLlvmArchives = {
    "linux/amd64": `clang+llvm-${manifest.llvmVersion}-x86_64-linux-gnu-ubuntu-18.04.tar.xz`,
    "linux/arm64": `clang+llvm-${manifest.llvmVersion}-aarch64-linux-gnu.tar.xz`,
  };
  if (llvmArtifacts.size !== 2) {
    throw new Error("pinned LLVM artifacts must cover linux/amd64 and linux/arm64");
  }
  for (const [platform, archive] of Object.entries(expectedLlvmArchives)) {
    const artifact = llvmArtifacts.get(platform);
    if (artifact?.archive !== archive || !SHA256.test(artifact?.sha256)) {
      throw new Error(`invalid pinned LLVM artifact for ${platform}`);
    }
  }
  if (!Array.isArray(manifest.overlays) || manifest.overlays.length !== 3) {
    throw new Error("fork overlay manifest must contain exactly three overlays");
  }
  const ids = new Set();
  for (const overlay of manifest.overlays) {
    if (!/^[-a-z0-9]+$/.test(overlay?.id) || ids.has(overlay.id)) {
      throw new Error("fork overlay IDs must be unique lowercase slugs");
    }
    ids.add(overlay.id);
    for (const key of ["baseCommit", "baseTree"]) {
      if (!COMMIT.test(overlay[key])) throw new Error(`${overlay.id} has invalid ${key}`);
    }
    if (!/^[-a-z0-9]+\.patch$/.test(overlay.patch)) {
      throw new Error(`${overlay.id} has an invalid patch path`);
    }
    if (!Number.isSafeInteger(overlay.patchBytes) || overlay.patchBytes < 1) {
      throw new Error(`${overlay.id} has an invalid patch size`);
    }
    if (!SHA256.test(overlay.patchSha256)) {
      throw new Error(`${overlay.id} has an invalid patch digest`);
    }
    if (Object.keys(overlay.resultFiles ?? {}).length === 0) {
      throw new Error(`${overlay.id} has no expected result files`);
    }
    for (const [path, digest] of Object.entries(overlay.resultFiles)) {
      assertRelativePath(path, `${overlay.id} result path`);
      if (!SHA256.test(digest)) throw new Error(`${overlay.id} has an invalid result digest`);
    }
  }
}

export async function verifyOverlayArtifacts(overlay, manifestPath = defaultManifestPath) {
  const manifestDirectory = dirname(manifestPath);
  const patchPath = resolve(manifestDirectory, overlay.patch);
  const info = await lstat(patchPath);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error(`${overlay.id} patch is not a regular file`);
  const patchBytes = await readFile(patchPath);
  if (patchBytes.byteLength !== overlay.patchBytes) {
    throw new Error(`${overlay.id} patch size differs from the manifest`);
  }
  const digest = sha256(patchBytes);
  if (digest !== overlay.patchSha256) {
    throw new Error(`${overlay.id} patch digest differs from the manifest`);
  }
  return patchPath;
}

export async function applyOverlay({ overlay, repository, manifestPath = defaultManifestPath, apply = false }) {
  const repositoryPath = await realpath(repository);
  const topLevel = await realpath(git(repositoryPath, ["rev-parse", "--show-toplevel"]));
  if (topLevel !== repositoryPath) throw new Error(`${overlay.id} target must be the checkout root`);
  const head = git(repositoryPath, ["rev-parse", "HEAD"]);
  const tree = git(repositoryPath, ["rev-parse", "HEAD^{tree}"]);
  if (head !== overlay.baseCommit || tree !== overlay.baseTree) {
    throw new Error(`${overlay.id} requires exact base ${overlay.baseCommit}`);
  }
  if (git(repositoryPath, ["status", "--porcelain", "--untracked-files=normal"]) !== "") {
    throw new Error(`${overlay.id} target checkout must be clean`);
  }
  const patchPath = await verifyOverlayArtifacts(overlay, manifestPath);
  verifyResultCoverage(repositoryPath, patchPath, overlay);
  git(repositoryPath, ["apply", "--check", "--binary", patchPath]);
  if (!apply) return { applied: false, head, tree };
  git(repositoryPath, ["apply", "--binary", patchPath]);
  await verifyResultFiles(repositoryPath, overlay);
  return { applied: true, head, tree };
}

function verifyResultCoverage(repository, patchPath, overlay) {
  const output = git(repository, ["apply", "--numstat", "--binary", patchPath]);
  const patchedPaths = output === ""
    ? []
    : output.split("\n").map((line) => {
        const fields = line.split("\t");
        if (fields.length !== 3) throw new Error(`${overlay.id} has an unsupported patch path`);
        assertRelativePath(fields[2], `${overlay.id} patch path`);
        return fields[2];
      });
  const expectedPaths = Object.keys(overlay.resultFiles).sort();
  if (
    patchedPaths.length !== new Set(patchedPaths).size ||
    patchedPaths.sort().join("\n") !== expectedPaths.join("\n")
  ) {
    throw new Error(`${overlay.id} result files do not match its patched files`);
  }
}

export async function verifyResultFiles(repository, overlay) {
  for (const [path, expected] of Object.entries(overlay.resultFiles)) {
    const candidate = resolve(repository, path);
    if (!isWithin(repository, candidate)) throw new Error(`${overlay.id} result escaped the checkout`);
    const info = await lstat(candidate);
    if (!info.isFile() || info.isSymbolicLink()) throw new Error(`${overlay.id} result is not a regular file: ${path}`);
    if (sha256(await readFile(candidate)) !== expected) {
      throw new Error(`${overlay.id} result digest differs for ${path}`);
    }
  }
}

function git(cwd, args) {
  return execFileSync("git", ["-c", "core.hooksPath=/dev/null", ...args], {
    cwd,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  }).trim();
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function assertRelativePath(path, label) {
  if (typeof path !== "string" || path === "" || isAbsolute(path) || path.includes("\\") || path.includes("\0")) {
    throw new Error(`${label} must be a portable relative path`);
  }
  const parts = path.split("/");
  if (parts.some((part) => part === "" || part === "." || part === "..")) {
    throw new Error(`${label} contains an unsafe segment`);
  }
}

function isWithin(root, candidate) {
  const path = relative(resolve(root), resolve(candidate));
  return path === "" || (!path.startsWith(`..${sep}`) && path !== "..");
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.help) {
    console.log("usage: apply-fork-overlay.mjs ID REPOSITORY [--check|--apply]");
    return;
  }
  const manifest = await loadManifest();
  const overlay = manifest.overlays.find((candidate) => candidate.id === options.id);
  if (!overlay) throw new Error(`unknown fork overlay: ${options.id}`);
  const result = await applyOverlay({
    overlay,
    repository: options.repository,
    apply: options.apply,
  });
  console.log(`${overlay.id}: ${result.applied ? "applied and verified" : "base and patch verified"}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
