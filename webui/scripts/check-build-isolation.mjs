import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { requiredDashboardIcons, shipsInDashboard } from "./dashboard-assets.mjs";

// The dashboard bundle ships in the wheel; the hosted site (drowse.ai) builds
// the same app with its own PWA, WebGPU runtime, and assets.  None of those
// belong in the bundle `drowse serve` mounts.
const [rootArg] = process.argv.slice(2);
if (!rootArg) {
  throw new Error("usage: node check-build-isolation.mjs OUTPUT_DIR");
}

const root = resolve(rootArg);
const paths = [];
async function walk(directory, prefix = "") {
  const entries = await readdir(directory, { withFileTypes: true });
  for (const entry of entries) {
    const relative = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) await walk(resolve(directory, entry.name), relative);
    else paths.push(relative);
  }
}
await walk(root);

const forbidden = paths.filter((path) =>
  path === "_headers" ||
  path === "manifest.webmanifest" ||
  path === "sw.js" ||
  !shipsInDashboard(path) ||
  // The browser packs' published SAE description indexes.
  (path.startsWith("assets/") && path.endsWith(".json")) ||
  path.startsWith("wasm/") ||
  path.endsWith(".wasm") ||
  /(?:^|\/)(?:mlc|webllm)/i.test(path)
);
if (forbidden.length) throw new Error(`hosted assets leaked into dashboard build: ${forbidden.join(", ")}`);
const missing = requiredDashboardIcons.filter((icon) => !paths.includes(icon));
if (missing.length) throw new Error(`dashboard icons missing from the build: ${missing.join(", ")}`);
const index = await readFile(resolve(root, "index.html"), "utf8");
if (/manifest\.webmanifest|serviceWorker|registerSW/.test(index)) {
  throw new Error("dashboard bootstraps hosted PWA behavior");
}
for (const match of index.matchAll(/(?:href|src)="\/([^"?#]+)/g)) {
  if (!paths.includes(match[1])) throw new Error(`index.html references a file the build omits: /${match[1]}`);
}

console.log(`dashboard build isolation passed (${paths.length} files)`);
