import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { states, animatedStates, frameCount } from "./tab-icon-artwork.mjs";

const [mode, rootArg] = process.argv.slice(2);
if (!rootArg || (mode !== "default" && mode !== "hosted")) {
  throw new Error("usage: node check-build-isolation.mjs default|hosted OUTPUT_DIR");
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

const has = (name) => paths.includes(name);
const sharedIcons = new Set(["drowse-192", "drowse-512", "drowse-maskable-512", "apple-touch-icon"].map((name) => `icons/${name}.png`));
for (const state of states) {
  const frames = animatedStates.has(state) ? frameCount : 1;
  for (let frame = 0; frame < frames; frame++) {
    const suffix = animatedStates.has(state) ? `-${frame}` : "";
    for (const theme of ["light", "dark"]) sharedIcons.add(`icons/tab-${state}-${theme}${suffix}.png`);
    if (frame < 3) sharedIcons.add(`icons/tab-${state}${suffix}.png`);
  }
}
if (mode === "default") {
  const forbidden = paths.filter((path) =>
    path === "_headers" ||
    path === "manifest.webmanifest" ||
    path === "sw.js" ||
    (path.startsWith("icons/") && !sharedIcons.has(path)) ||
    path.startsWith("wasm/") ||
    path.endsWith(".wasm") ||
    /(?:^|\/)(?:mlc|webllm)/i.test(path)
  );
  if (forbidden.length) throw new Error(`hosted assets leaked into default build: ${forbidden.join(", ")}`);
  const index = await readFile(resolve(root, "index.html"), "utf8");
  if (/manifest\.webmanifest|serviceWorker|registerSW/.test(index)) {
    throw new Error("default dashboard bootstraps hosted PWA behavior");
  }
} else {
  for (const required of [
    "index.html",
    "_headers",
    "manifest.webmanifest",
    "sw.js",
    "icons/drowse-192.png",
    "icons/drowse-512.png",
    "icons/drowse-maskable-512.png",
    "wasm/fitting-kernel.json",
    "wasm/drowse_fitting_wasm.js",
    "wasm/drowse_fitting_wasm_bg.wasm",
  ]) {
    if (!has(required)) throw new Error(`hosted build is missing ${required}`);
  }
  const headers = await readFile(resolve(root, "_headers"), "utf8");
  for (const header of [
    "Cross-Origin-Opener-Policy: same-origin",
    "Cross-Origin-Embedder-Policy: require-corp",
    "Content-Security-Policy:",
  ]) {
    if (!headers.includes(header)) throw new Error(`hosted headers are missing ${header}`);
  }
  const fixtureAssets = paths.filter((path) => /fixture|\.bin$/i.test(path));
  if (fixtureAssets.length) {
    throw new Error(`development fixture assets leaked into hosted build: ${fixtureAssets.join(", ")}`);
  }
  const scripts = paths.filter((path) => path.endsWith(".js"));
  const source = (await Promise.all(
    scripts.map((path) => readFile(resolve(root, path), "utf8")),
  )).join("\n");
  for (const marker of [
    "development-fixture",
    "DeterministicFakeRuntime",
    "fixtureHostedRuntime",
    "fixture.invalid",
  ]) {
    if (source.includes(marker)) {
      throw new Error(`development fixture marker leaked into hosted build: ${marker}`);
    }
  }
  for (const marker of [
    "SAE training tokens",
    "J-lens corpus prompts",
    "Blocks generation; uses FineWeb-Edu",
    "generation paused during model fitting",
    "Create on this device",
    "drowse lens fit",
    "train_hosted_sae.py",
    "fit_hosted_jlens.py",
  ]) {
    if (source.includes(marker)) {
      throw new Error(`Python-only instrument authoring leaked into hosted build: ${marker}`);
    }
  }
}

console.log(`${mode} build isolation passed (${paths.length} files)`);
