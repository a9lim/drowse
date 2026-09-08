import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readdir, readFile, stat } from "node:fs/promises";
import { resolve } from "node:path";
import { siteDescription, siteAccent, socialImagePath, socialImageAlt } from "./site-metadata.mjs";

const root = resolve("dist-hosted");
const release = process.argv.includes("--release");
const manifest = JSON.parse(await readFile(resolve(root, "manifest.webmanifest"), "utf8"));
const index = await readFile(resolve(root, "index.html"), "utf8");
const notFound = await readFile(resolve(root, "404.html"), "utf8");
const redirects = await readFile(resolve(root, "_redirects"), "utf8");
assert.equal(notFound, index, "The 404 page must render Drowse without an asset SPA fallback");
assert.equal(redirects.trim(), "/app / 200\n/app/ / 200\n/credits / 200\n/credits/ / 200\n/app/* / 200");
const serviceWorker = await readFile(resolve(root, "sw.js"), "utf8");
const headers = await readFile(resolve(root, "_headers"), "utf8");
const robots = await readFile(resolve(root, "robots.txt"), "utf8");
const [builtLicense, sourceLicense] = await Promise.all([
  readFile(resolve(root, "LICENSE"), "utf8"),
  readFile(resolve("..", "LICENSE"), "utf8"),
]);
const distributionLock = JSON.parse(await readFile(
  resolve("..", "browser-runtime", "distribution-lock.json"),
  "utf8",
));
const runtimeLock = JSON.parse(await readFile(
  resolve("..", "browser-runtime", "runtime-lock.json"),
  "utf8",
));
const drowseSource = await readFile(resolve("..", "drowse", "__init__.py"), "utf8");
const drowseVersion = /__version__\s*=\s*"([^"]+)"/.exec(drowseSource)?.[1];
assert.ok(drowseVersion, "could not resolve the Drowse source version");
const fittingManifest = JSON.parse(await readFile(
  resolve(root, "wasm", "fitting-kernel.json"),
  "utf8",
));
const metadata = (name) => {
  const match = new RegExp(`<meta\\s+(?:name|property)="${name}"\\s+content="([^"]+)"\\s*\\/>`).exec(index);
  assert.ok(match, `hosted index is missing ${name}`);
  return match[1];
};
const pageDescription = siteDescription;
const manifestDescription = siteDescription;
const sourceRevision = metadata("drowse-source-revision");
const sourceUrl = metadata("drowse-source-url");
const entryScript = /<script[^>]+src="\/assets\/([^"]+\.js)"/.exec(index)?.[1];
assert.ok(entryScript, "hosted index is missing its entry script");

assert.equal(manifest.id, "/app");
assert.equal(manifest.start_url, "/app");
assert.equal(manifest.scope, "/");
assert.equal(manifest.display, "standalone");
assert.equal(manifest.theme_color, siteAccent);
assert.equal(metadata("theme-color"), siteAccent);
assert.equal(metadata("description"), siteDescription);
assert.equal(metadata("og:description"), siteDescription);
assert.equal(metadata("twitter:description"), siteDescription);
assert.equal(metadata("og:image:alt"), socialImageAlt);
assert.equal(metadata("twitter:image:alt"), socialImageAlt);
assert.ok(metadata("og:image").endsWith(socialImagePath));
assert.ok(metadata("twitter:image").endsWith(socialImagePath));
assert.equal(
  manifest.description,
  manifestDescription,
);
assert.deepEqual(
  manifest.icons.map(({ src, sizes, purpose }) => ({ src, sizes, purpose: purpose ?? "any" })),
  [
    { src: "/icons/drowse-192.png", sizes: "192x192", purpose: "any" },
    { src: "/icons/drowse-512.png", sizes: "512x512", purpose: "any" },
    { src: "/icons/drowse-maskable-512.png", sizes: "512x512", purpose: "maskable" },
  ],
);

assert.match(index, /manifest\.webmanifest/);
assert.doesNotMatch(index, /registerSW\.js/);
assert.doesNotMatch(index, /\/src\/main\.ts|\/drowse\/v1|api-key/);
assert.doesNotMatch(index, /__DROWSE_/);
assert.equal(metadata("drowse-release-channel"), release ? "release" : "preview");
assert.equal(metadata("drowse-version"), drowseVersion);
assert.equal(
  metadata("drowse-artifact-source-revision"),
  release
    ? sourceRevision
    : execFileSync("git", ["rev-parse", "HEAD"], {
        cwd: resolve(".."),
        encoding: "utf8",
      }).trim(),
);
assert.equal(metadata("drowse-runtime-abi"), runtimeLock.runtimeAbi);
assert.equal(metadata("drowse-hook-abi"), runtimeLock.hookAbi);
assert.equal(metadata("drowse-artifact-verification"), release ? "verified" : "unverified");
assert.ok(
  index.includes(`content="${pageDescription}"`),
  "hosted index is missing the on-device description",
);
if (release) {
  assert.match(sourceRevision, /^[0-9a-f]{40}$/);
  assert.equal(sourceUrl, `https://github.com/a9lim/drowse/tree/${sourceRevision}`);
} else {
  assert.equal(sourceRevision, "preview");
  assert.equal(sourceUrl, "https://github.com/a9lim/drowse");
}
assert.equal(builtLicense, sourceLicense, "hosted build has the wrong AGPL license");
assert.match(serviceWorker, /index\.html/);
assert.match(serviceWorker, /drowse-192\.png/);
assert.match(serviceWorker, /LICENSE/);
assert.match(serviceWorker, /drowse-hosted-on-demand-assets-v1/);
assert.match(serviceWorker, /drowse-hosted-on-demand-wasm-v1/);
assert.doesNotMatch(serviceWorker, /huggingface|cdn\.hf\.co|model-.*\.safetensors/i);
if (release) {
  assert.doesNotMatch(headers.split(/\n\s*\n/)[0], /X-Robots-Tag:\s*noindex/i);
  assert.match(headers, /\/app\n  X-Robots-Tag: noindex, follow/);
  assert.match(robots, /^User-agent: \*\nAllow: \/\n\nSitemap: https:\/\/[^\s]+\/sitemap\.xml\n$/);
} else {
  assert.match(headers, /X-Robots-Tag: noindex, nofollow/);
  assert.match(robots, /^User-agent: \*\nDisallow: \/\n$/);
}
for (const required of [
  "Cross-Origin-Opener-Policy: same-origin",
  "Cross-Origin-Embedder-Policy: require-corp",
  "Cross-Origin-Resource-Policy: same-origin",
  "Content-Security-Policy:",
  "frame-ancestors 'none'",
  "object-src 'none'",
  "Permissions-Policy:",
  "Referrer-Policy: no-referrer",
  "X-Content-Type-Options: nosniff",
  "X-Frame-Options: DENY",
]) {
  assert.ok(headers.includes(required), `hosted headers are missing ${required}`);
}

const connectOrigins = [...new Set([
  "https://www.neuronpedia.org",
  new URL(distributionLock.catalogUrl).origin,
  new URL(distributionLock.signatureUrl).origin,
  ...distributionLock.allowedCatalogRedirectOrigins,
  ...distributionLock.allowedArtifactRedirectOrigins,
])].sort();
const expectedConnectSrc = `connect-src 'self' ${connectOrigins.join(" ")};`;
assert.ok(
  headers.includes(expectedConnectSrc),
  `hosted connect-src must match the distribution lock: ${expectedConnectSrc}`,
);
assert.doesNotMatch(headers, /connect-src[^;]*https:\/\/\*/);
assert.match(headers, /\/assets\/\*[\s\S]*Cache-Control: public, max-age=31536000, immutable/);
for (const path of ["/sw.js", "/manifest.webmanifest"]) {
  assert.ok(
    headers.includes(`${path}\n  Cache-Control: no-cache`),
    `${path} must revalidate so PWA updates are discoverable`,
  );
}

for (const icon of ["drowse-192.png", "drowse-512.png", "drowse-maskable-512.png"]) {
  const iconStat = await stat(resolve(root, "icons", icon));
  assert.ok(iconStat.size > 1_000, `${icon} is unexpectedly small`);
}

assert.equal(fittingManifest.schemaVersion, 1);
assert.equal(fittingManifest.crate, "drowse-fitting-wasm");
assert.equal(fittingManifest.crateVersion, "0.1.0");
assert.equal(fittingManifest.wasmBindgenVersion, "0.2.127");
assert.match(fittingManifest.cargoLockSha256, /^[a-f0-9]{64}$/);
assert.deepEqual(
  fittingManifest.files.map(({ name }) => name).sort(),
  ["drowse_fitting_wasm.js", "drowse_fitting_wasm_bg.wasm"],
);
for (const file of fittingManifest.files) {
  assert.match(file.sha256, /^[a-f0-9]{64}$/);
  const bytes = await readFile(resolve(root, "wasm", file.name));
  assert.equal(bytes.byteLength, file.bytes, `${file.name} has the wrong byte count`);
  assert.equal(
    createHash("sha256").update(bytes).digest("hex"),
    file.sha256,
    `${file.name} has the wrong digest`,
  );
}

const assets = await readdir(resolve(root, "assets"));
assert.ok(assets.some((name) => name.endsWith(".woff2")), "hosted build is missing its local font");
assert.ok(assets.some((name) => name.endsWith(".css")), "hosted build is missing extracted CSS");
const runtimeWorkers = assets.filter((name) => /^browser\.worker-[A-Za-z0-9_-]+\.js$/u.test(name));
const webLlmChunks = assets.filter((name) => /^drowse-web-llm-[A-Za-z0-9_-]+\.js$/u.test(name));
const landingChunks = assets.filter((name) => /^LandingRoot-[A-Za-z0-9_-]+\.js$/u.test(name));
const shaderChunks = assets.filter((name) => /^heroShaderRuntime-[A-Za-z0-9_-]+\.js$/u.test(name));
assert.equal(shaderChunks.length, 1, "hosted build must contain the landing renderer");
assert.ok(serviceWorker.includes(`url:"assets/${shaderChunks[0]}"`), "landing renderer must work offline");
assert.ok(serviceWorker.includes('url:"images/ethereal-orb.jpg"'), "landing artwork must work offline");
assert.equal(runtimeWorkers.length, 1, "hosted build must contain one browser runtime worker");
assert.equal(webLlmChunks.length, 1, "hosted build must split the WebLLM engine from onboarding");
assert.equal(landingChunks.length, 1, "hosted build must split the landing route from the app");
assert.ok(
  !serviceWorker.includes(`url:"assets/${webLlmChunks[0]}"`),
  "hosted service worker must not download WebLLM before the user loads a model",
);
assert.ok(
  !serviceWorker.includes(`url:"assets/${runtimeWorkers[0]}"`),
  "hosted service worker must not download the browser worker on the landing page",
);
for (const pattern of [/^App-.*\.(?:css|js)$/u, /^fitting\.worker-.*\.js$/u]) {
  for (const name of assets.filter((asset) => pattern.test(asset))) {
    assert.ok(
      !serviceWorker.includes(`url:"assets/${name}"`),
      `hosted service worker must not precache on-demand asset ${name}`,
    );
  }
}
assert.ok(
  (await stat(resolve(root, "assets", entryScript))).size < 10 * 1024,
  "the route entry eagerly bundles hosted application code",
);
assert.ok(
  (await stat(resolve(root, "assets", landingChunks[0]))).size < 50 * 1024,
  "the landing route eagerly bundles hosted application code",
);
assert.ok(
  (await stat(resolve(root, "assets", webLlmChunks[0]))).size <= 8 * 1024 * 1024,
  "the WebLLM engine exceeds the hosted service worker precache limit",
);
assert.ok(
  (await stat(resolve(root, "assets", runtimeWorkers[0]))).size < 2 * 1024 * 1024,
  "browser onboarding worker eagerly bundled the WebLLM engine",
);
const runtimeWorkerSource = await readFile(resolve(root, "assets", runtimeWorkers[0]), "utf8");
assert.ok(
  [`\"`, "'", "`"].some((quote) =>
    runtimeWorkerSource.includes(`import(${quote}./${webLlmChunks[0]}${quote})`)
  ),
  "browser worker does not lazy-load the pinned WebLLM chunk",
);
let hasUpdatePrompt = false;
let bundledSource = "";
for (const name of assets.filter((asset) => asset.endsWith(".js"))) {
  const source = await readFile(resolve(root, "assets", name), "utf8");
  bundledSource += source;
  if (source.includes("A Drowse update is ready.")) hasUpdatePrompt = true;
  assert.doesNotMatch(source, /\/drowse\/v1|DROWSE_API_KEY/);
  assert.doesNotMatch(source, /@huggingface\/transformers/);
}
assert.doesNotMatch(bundledSource, /browser-preview|["']browser-v1["']/);
for (const identity of [
  drowseVersion,
  metadata("drowse-artifact-source-revision"),
  runtimeLock.runtimeAbi,
  runtimeLock.hookAbi,
]) {
  assert.ok(bundledSource.includes(identity), `hosted artifact provenance is missing ${identity}`);
}
assert.equal(hasUpdatePrompt, true, "hosted build is missing its prompt-style update UI");
assert.ok(bundledSource.includes(sourceUrl), "hosted UI does not expose the exact source URL");
assert.ok(bundledSource.includes("GNU AGPL v3 or later"), "hosted UI is missing its license notice");
assert.ok(
  bundledSource.includes("Open source. Inference and saved work stay on your device."),
  "hosted landing page is missing its local-compute promise",
);

console.log(`Hosted ${release ? "release" : "preview"} shell build passed`);
