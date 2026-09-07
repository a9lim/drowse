import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  collectPackArtifacts,
  parseArguments,
} from "./assemble-hosted-pack.mjs";

assert.deepEqual(parseArguments(["./pack"]), {
  directory: join(process.cwd(), "pack"),
});
assert.throws(() => parseArguments([]), /usage/);
assert.throws(() => parseArguments(["./pack", "extra"]), /usage/);

const root = await mkdtemp(join(tmpdir(), "drowse-pack-assembler-"));
try {
  await mkdir(join(root, "packs", "sae"), { recursive: true });
  await writeFile(join(root, "packs", "sae", "manifest.json"), "manifest");
  await writeFile(join(root, "packs", "sae", "weights.safetensors"), "weights");
  const manifest = await collectPackArtifacts(root);
  assert.deepEqual(manifest.files.map((file) => file.path), [
    "packs/sae/manifest.json",
    "packs/sae/weights.safetensors",
  ]);
  assert.equal(manifest.files[0].sha256, digest("manifest"));
  assert.equal(manifest.files[1].bytes, 7);
  await writeFile(join(root, "hosted-pack-artifacts.json"), "ignored");
  assert.equal((await collectPackArtifacts(root)).files.length, 2);
  await symlink(join(root, "packs", "sae", "manifest.json"), join(root, "link"));
  await assert.rejects(() => collectPackArtifacts(root), /symbolic link/);
} finally {
  await rm(root, { recursive: true, force: true });
}

console.log("hosted pack assembler checks passed");

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}
