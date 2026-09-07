#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";

const manifestPath = resolve(import.meta.dirname, "manifest.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
const repositories = new Map([
  ["tvm-webgpu-readonly", process.env.DROWSE_TVM_REPOSITORY],
  ["mlc-llm-drowse", process.env.DROWSE_MLC_REPOSITORY ??
    resolve(homedir(), ".cache/drowse-production-forks/mlc-llm")],
  ["web-llm-drowse", process.env.DROWSE_WEBLLM_REPOSITORY ??
    resolve(homedir(), ".cache/drowse-production-forks/web-llm")],
]);

for (const overlay of manifest.overlays) {
  const repository = repositories.get(overlay.id);
  if (repository === undefined) continue;
  requireOutput(repository, ["rev-parse", "HEAD"], overlay.baseCommit);
  const status = run(repository, ["status", "--porcelain=v1", "-z"]).stdout;
  const entries = status.split("\0").filter(Boolean).map((entry) => ({
    status: entry.slice(0, 2),
    path: entry.slice(3),
  })).sort((left, right) => left.path.localeCompare(right.path));
  if (entries.length === 0) throw new Error(`${overlay.id} has no changes`);
  const chunks = [];
  const resultFiles = {};
  for (const entry of entries) {
    if (entry.status.includes("D")) {
      chunks.push(run(repository, ["diff", "--binary", overlay.baseCommit, "--", entry.path], [0]).stdout);
      continue;
    }
    const untracked = entry.status === "??";
    chunks.push(run(
      repository,
      untracked
        ? ["diff", "--binary", "--no-index", "--", "/dev/null", entry.path]
        : ["diff", "--binary", overlay.baseCommit, "--", entry.path],
      untracked ? [0, 1] : [0],
    ).stdout);
    const bytes = await readFile(resolve(repository, entry.path));
    resultFiles[entry.path] = sha256(bytes);
  }
  const patch = chunks.join("");
  const patchPath = resolve(import.meta.dirname, overlay.patch);
  await writeFile(patchPath, patch);
  overlay.patchBytes = Buffer.byteLength(patch);
  overlay.patchSha256 = sha256(Buffer.from(patch));
  overlay.resultFiles = resultFiles;
}

await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

function run(cwd, args, accepted = [0]) {
  const result = spawnSync("git", args, { cwd, encoding: "utf8" });
  if (!accepted.includes(result.status)) {
    throw new Error(`git ${args.join(" ")} failed in ${cwd}: ${result.stderr}`);
  }
  return result;
}

function requireOutput(cwd, args, expected) {
  const actual = run(cwd, args).stdout.trim();
  if (actual !== expected) {
    throw new Error(`${cwd} must be at ${expected}; found ${actual}`);
  }
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
