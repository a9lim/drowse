import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { states, animatedStates, frameCount } from "./tab-icon-artwork.mjs";

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

const sharedIcons = new Set(["drowse-192", "drowse-512", "drowse-maskable-512", "apple-touch-icon"].map((name) => `icons/${name}.png`));
for (const state of states) {
  const frames = animatedStates.has(state) ? frameCount : 1;
  for (let frame = 0; frame < frames; frame++) {
    const suffix = animatedStates.has(state) ? `-${frame}` : "";
    for (const theme of ["light", "dark"]) sharedIcons.add(`icons/tab-${state}-${theme}${suffix}.png`);
    if (frame < 3) sharedIcons.add(`icons/tab-${state}${suffix}.png`);
  }
}
const forbidden = paths.filter((path) =>
  path === "_headers" ||
  path === "manifest.webmanifest" ||
  path === "sw.js" ||
  (path.startsWith("icons/") && !sharedIcons.has(path)) ||
  path.startsWith("wasm/") ||
  path.endsWith(".wasm") ||
  /(?:^|\/)(?:mlc|webllm)/i.test(path)
);
if (forbidden.length) throw new Error(`hosted assets leaked into dashboard build: ${forbidden.join(", ")}`);
const index = await readFile(resolve(root, "index.html"), "utf8");
if (/manifest\.webmanifest|serviceWorker|registerSW/.test(index)) {
  throw new Error("dashboard bootstraps hosted PWA behavior");
}

console.log(`dashboard build isolation passed (${paths.length} files)`);
