import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";

const exec = promisify(execFile);
const repositoryRoot = resolve(import.meta.dirname, "../..");
const script = resolve(import.meta.dirname, "capture-hosted-residuals.mjs");
const runtimeLockPath = resolve(repositoryRoot, "browser-runtime/runtime-lock.json");

const help = await exec(process.execPath, [script, "--help"]);
assert.match(help.stdout, /--runtime-lock PATH/);

const root = await mkdtemp(join(tmpdir(), "drowse-capture-lock-"));
try {
  const runtimeLock = JSON.parse(await readFile(runtimeLockPath, "utf8"));
  const unexpectedPath = join(root, "unexpected.json");
  await writeFile(
    unexpectedPath,
    JSON.stringify({ ...runtimeLock, unexpected: true }),
  );
  await assert.rejects(
    runWithLock(unexpectedPath),
    /runtime lock has unknown or missing fields/,
  );

  const invalidModelPath = join(root, "invalid-model.json");
  runtimeLock.models[0].contextProfiles = [2048, 2048];
  await writeFile(invalidModelPath, JSON.stringify(runtimeLock));
  await assert.rejects(
    runWithLock(invalidModelPath),
    /context profiles are invalid/,
  );
} finally {
  await rm(root, { recursive: true, force: true });
}

console.log("Hosted residual capture runtime-lock checks passed");

async function runWithLock(path) {
  return await exec(process.execPath, [
    script,
    "qwen3-1.7b",
    "missing-model-directory",
    "missing-model.wasm",
    "missing-webllm.js",
    "missing-input.json",
    "missing-output",
    "--runtime-lock",
    path,
  ]);
}
