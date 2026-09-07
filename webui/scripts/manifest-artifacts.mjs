import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, opendir } from "node:fs/promises";
import { relative, resolve, sep } from "node:path";

const [rootArg, repository, revision] = process.argv.slice(2);
if (!rootArg || !repository || !/^[0-9a-f]{40}$/.test(revision ?? "")) {
  throw new Error("usage: node manifest-artifacts.mjs ROOT OWNER/REPO COMMIT");
}
if (!/^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/.test(repository)) {
  throw new Error("invalid Hugging Face repository id");
}

const root = resolve(rootArg);
const files = [];

async function walk(directory) {
  const entries = [];
  for await (const entry of await opendir(directory)) entries.push(entry);
  entries.sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    const path = resolve(directory, entry.name);
    const stat = await lstat(path);
    if (stat.isSymbolicLink()) throw new Error(`symbolic link is not publishable: ${path}`);
    if (stat.isDirectory()) {
      await walk(path);
      continue;
    }
    if (!stat.isFile()) throw new Error(`special file is not publishable: ${path}`);
    const name = relative(root, path).split(sep).join("/");
    const digest = createHash("sha256");
    for await (const chunk of createReadStream(path)) digest.update(chunk);
    files.push({
      path: name,
      bytes: stat.size,
      sha256: digest.digest("hex"),
      url: `https://huggingface.co/${repository}/resolve/${revision}/${name
        .split("/")
        .map(encodeURIComponent)
        .join("/")}`,
    });
  }
}

await walk(root);
process.stdout.write(`${JSON.stringify({ repository, revision, files }, null, 2)}\n`);

