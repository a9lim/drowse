// Builds the plain-text third-party license notice for the hosted site from the npm packages
// whose code ships in the build: modules Rollup bundled, plus production dependencies imported
// by worker entries (bundled separately) and the Workbox service-worker runtime.
import { readdir, readFile } from "node:fs/promises";
import { join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const webuiRoot = fileURLToPath(new URL("../", import.meta.url));
const nodeModules = resolve(webuiRoot, "node_modules");
const serviceWorkerPackages = ["workbox-core", "workbox-routing", "workbox-strategies", "workbox-precaching", "workbox-expiration", "workbox-cacheable-response"];
const noticeFile = /^(?:licen[cs]e|copying|notice)(?:[.-].*)?$/i;

/** Maps each `node_modules` package that contributed a bundled module to its directory. */
export function bundledPackageDirectories(moduleIds) {
  const directories = new Map();
  for (const id of moduleIds) {
    const path = id.replace(/^\0/, "").split("?")[0];
    const marker = `${sep}node_modules${sep}`;
    const at = path.lastIndexOf(marker);
    if (at < 0) continue;
    const parts = path.slice(at + marker.length).split(sep);
    const name = parts[0].startsWith("@") ? `${parts[0]}/${parts[1]}` : parts[0];
    if (!name || name.startsWith(".")) continue;
    directories.set(name, path.slice(0, at + marker.length) + name.split("/").join(sep));
  }
  return directories;
}

async function sourceFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await sourceFiles(path));
    else if (/\.(?:ts|svelte|js|mjs)$/.test(entry.name)) files.push(path);
  }
  return files;
}

/** Production dependencies imported anywhere in the hosted source, including worker-only imports. */
export async function importedDependencies() {
  const manifest = JSON.parse(await readFile(resolve(webuiRoot, "package.json"), "utf8"));
  const dependencies = Object.keys(manifest.dependencies ?? {});
  const sources = (await Promise.all(["src", "hosted"].map(root => sourceFiles(resolve(webuiRoot, root)))))
    .flat();
  const text = (await Promise.all(sources.map(file => readFile(file, "utf8")))).join("\n");
  return dependencies.filter(name => new RegExp(`(?:from\\s*|import\\s*\\(\\s*)["']${name.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&")}(?:/[^"']*)?["']`).test(text));
}

export async function thirdPartyLicenseText(moduleIds = []) {
  const directories = bundledPackageDirectories(moduleIds);
  for (const name of [...await importedDependencies(), "svelte", "workbox-window", ...serviceWorkerPackages]) {
    if (!directories.has(name)) directories.set(name, resolve(nodeModules, ...name.split("/")));
  }
  const sections = [];
  for (const [name, directory] of [...directories].sort(([a], [b]) => a.localeCompare(b))) {
    let manifest;
    try {
      manifest = JSON.parse(await readFile(join(directory, "package.json"), "utf8"));
    } catch {
      continue;
    }
    if (manifest.name !== name) continue;
    const license = typeof manifest.license === "string" ? manifest.license : manifest.license?.type ?? "Not declared in package metadata";
    const repository = typeof manifest.repository === "string" ? manifest.repository : manifest.repository?.url;
    const files = (await readdir(directory)).filter(file => noticeFile.test(file)).sort();
    const texts = await Promise.all(files.map(async file => `--- ${file} ---\n${(await readFile(join(directory, file), "utf8")).trim()}`));
    sections.push([
      `${name} ${manifest.version}`,
      `License: ${license}`,
      ...(repository ? [`Source: ${repository.replace(/^git\+/, "")}`] : []),
      "",
      ...(texts.length ? texts : ["No license or notice file is included in this package."]),
    ].join("\n"));
  }
  return [
    "Third-party software in the Drowse website",
    "",
    "Drowse itself is licensed under the GNU Affero General Public License v3 or later; see /LICENSE.",
    "Fonts and icons are listed in /LICENSE-*.txt. Downloaded models and instrument packs carry their own licenses.",
    "This file lists the npm packages whose code ships in this build, with their license and notice files.",
    "",
    ...sections.flatMap(section => ["=".repeat(72), section, ""]),
  ].join("\n");
}
