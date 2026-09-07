#!/usr/bin/env node

import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import {
  lstat,
  opendir,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const MANIFEST = "hosted-pack-artifacts.json";

export function parseArguments(args) {
  if (args.length !== 1 || args[0].startsWith("--")) {
    throw new Error("usage: node assemble-hosted-pack.mjs PACK_DIRECTORY");
  }
  return { directory: resolve(args[0]) };
}

export async function collectPackArtifacts(directory) {
  const root = resolve(directory);
  const rootInfo = await lstat(root);
  if (!rootInfo.isDirectory() || rootInfo.isSymbolicLink()) {
    throw new Error("pack source is not a regular directory");
  }
  const paths = [];
  await walk(root, root, paths);
  paths.sort((left, right) => left.localeCompare(right));
  if (paths.length === 0) throw new Error("pack source contains no artifacts");
  const files = [];
  for (const path of paths) {
    const absolute = resolve(root, ...path.split("/"));
    const info = await lstat(absolute);
    if (!info.isFile() || info.isSymbolicLink() || info.size < 1) {
      throw new Error(`pack artifact is not a non-empty regular file: ${path}`);
    }
    files.push({ path, bytes: info.size, sha256: await sha256(absolute) });
  }
  return { schemaVersion: 1, files };
}

async function walk(root, directory, paths) {
  const entries = [];
  for await (const entry of await opendir(directory)) entries.push(entry);
  entries.sort((left, right) => left.name.localeCompare(right.name));
  for (const entry of entries) {
    if (entry.name === MANIFEST && directory === root) continue;
    const absolute = resolve(directory, entry.name);
    const info = await lstat(absolute);
    if (info.isSymbolicLink()) throw new Error(`pack source contains a symbolic link: ${entry.name}`);
    if (info.isDirectory()) {
      await walk(root, absolute, paths);
      continue;
    }
    if (!info.isFile()) throw new Error(`pack source contains a special file: ${entry.name}`);
    const path = relative(root, absolute).split(sep).join("/");
    if (!portablePath(path)) throw new Error(`pack source contains an unsafe path: ${path}`);
    paths.push(path);
  }
}

async function main() {
  const { directory } = parseArguments(process.argv.slice(2));
  const output = resolve(directory, MANIFEST);
  await requireMissing(output);
  const manifest = await collectPackArtifacts(directory);
  const stage = `${output}.partial-${randomUUID()}`;
  try {
    await writeFile(stage, `${JSON.stringify(manifest, null, 2)}\n`, { flag: "wx" });
    await rename(stage, output);
  } catch (error) {
    await rm(stage, { force: true });
    throw error;
  }
  console.log(JSON.stringify({
    output,
    files: manifest.files.length,
    bytes: manifest.files.reduce((sum, file) => sum + file.bytes, 0),
  }));
}

function portablePath(path) {
  const parts = path.split("/");
  return path.length > 0 && !path.startsWith("/") && !path.includes("\\") &&
    !path.includes("\0") && !/^[A-Za-z]:/.test(path) &&
    parts.every((part) => part && part !== "." && part !== "..");
}

async function sha256(path) {
  const digest = createHash("sha256");
  for await (const chunk of createReadStream(path)) digest.update(chunk);
  return digest.digest("hex");
}

async function requireMissing(path) {
  try {
    await lstat(path);
  } catch (error) {
    if (error?.code === "ENOENT") return;
    throw error;
  }
  throw new Error(`${MANIFEST} already exists`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
