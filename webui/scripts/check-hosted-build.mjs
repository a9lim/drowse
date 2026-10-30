import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readdir, readFile, stat } from "node:fs/promises";
import { resolve } from "node:path";
import { siteDescription, siteAccent, themeColors, socialImagePath, socialImageType, socialImageAlt } from "./site-metadata.mjs";
import { guidePages, htmlGuidePaths, sitePages } from "./site-content.mjs";
import { themeInitScript } from "./site-publication.mjs";
import { agentCatalogPaths, agentResources, releaseRobots } from "./agent-discovery.mjs";

const root = resolve("dist-hosted");
const release = process.argv.includes("--release");
const manifest = JSON.parse(await readFile(resolve(root, "manifest.webmanifest"), "utf8"));
const index = await readFile(resolve(root, "index.html"), "utf8");
const notFound = await readFile(resolve(root, "404.html"), "utf8");
const redirects = await readFile(resolve(root, "_redirects"), "utf8");
assert.match(notFound, /name="robots" content="noindex, follow"/);
assert.doesNotMatch(notFound, /rel="canonical"|application\/ld\+json|data-prerendered/);
assert.match(index, /data-prerendered="landing"/);
const redirectRules = redirects.trim().split("\n").map(line => line.split(/\s+/));
assert.deepEqual(redirectRules.slice(0, 2), [["/app", "/app-shell", "200"], ["/app/", "/app-shell", "200"]]);
assert.ok(!redirectRules.some(([from]) => from.includes("*")), "unknown /app/* paths must reach the real 404 page");
const appShell = await readFile(resolve(root, "app-shell.html"), "utf8");
assert.match(appShell, /name="robots" content="noindex, follow"/);
assert.match(appShell, /<title>Drowse workbench<\/title>/);
// Every public page is a real prerendered document with its own metadata and a hydration key.
const guideByPath = new Map(guidePages.map(page => [page.path, page]));
for (const page of [...htmlGuidePaths.map(path => guideByPath.get(path)), ...sitePages]) {
  const html = await readFile(resolve(root, `${page.path.slice(1)}.html`), "utf8");
  const escaped = text => text.replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
  assert.ok(html.includes(`<title>${page.title.replaceAll("&", "&amp;").replaceAll("<", "&lt;")}</title>`), `${page.path} title`);
  assert.ok(html.includes(`name="description"\n      content="${escaped(page.description)}"`) || html.includes(`name="description" content="${escaped(page.description)}"`), `${page.path} description`);
  assert.match(html, /data-prerendered="[^"]+"/, `${page.path} must be prerendered`);
  assert.match(html, /<h1\b/, `${page.path} needs a heading`);
  assert.doesNotMatch(html, /name="robots"|__DROWSE_/, `${page.path} must be indexable and fully resolved`);
  if (/rel="canonical"/.test(index)) assert.match(html, new RegExp(`rel="canonical" href="https://[^"]+${page.path}"`), `${page.path} canonical`);
}
const serviceWorker = await readFile(resolve(root, "sw.js"), "utf8");
assert.match(serviceWorker, /\.clientsClaim\(\)/u, "first-visit model code must be controlled and cached without a reload");
const offlineRuntimeManifest = JSON.parse(await readFile(resolve(root, "runtime-assets.json"), "utf8"));
const offlineRuntimeFiles = (await readdir(resolve(root, "assets"))).filter(name =>
  /^(?:App|browser\.worker|registry|drowse-web-llm)-[^/]+\.(?:css|js)$/u.test(name)
).map(name => `/assets/${name}`).sort();
assert.deepEqual(offlineRuntimeManifest.assets, offlineRuntimeFiles, "offline model setup must include its exact app-code closure");
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
assert.ok(index.includes(`<meta name="theme-color" media="(prefers-color-scheme: dark)" content="${themeColors.dark}" />`));
assert.ok(index.includes(`<meta name="theme-color" media="(prefers-color-scheme: light)" content="${themeColors.light}" />`));
assert.equal(metadata("og:image:type"), socialImageType);
const { source: themeSource, cspSource } = await themeInitScript();
assert.ok(index.includes(`<script>${themeSource}</script>`), "the pre-paint theme script must be inlined");
assert.doesNotMatch(index, /<script src="\/theme-init\.js">/);
assert.match(index, /<link rel="preload" href="\/assets\/WixMadeforText-normal-latin-[^"]+\.woff2" as="font" type="font\/woff2" crossorigin \/>/);
assert.match(index, /<link rel="preload" href="\/assets\/WixMadeforDisplay-latin-[^"]+\.woff2" as="font" type="font\/woff2" crossorigin \/>/);
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
assert.match(serviceWorker, /drowse-hosted-on-demand-wasm-v2/);
const precacheUrls = [...serviceWorker.matchAll(/\burl:"([^"]+)"/g)].map(match => match[1]);
assert.equal(new Set(precacheUrls).size, precacheUrls.length, "service worker precache entries must be unique");
for (const url of precacheUrls) {
  assert.doesNotMatch(url, /^icons\/tab-|^assets\/gemma-.*\.json$|^recordings\//, `${url} must load on demand, not in the precache`);
}
// /app is precached in place of the shell file it rewrites to (the file's own URL is a 404).
const precacheFile = url => url === "app" ? "app-shell.html" : url;
assert.ok(precacheUrls.includes("app") && !precacheUrls.includes("app-shell.html"), "the app shell must be precached as /app");
const precacheBytes = (await Promise.all(precacheUrls.map(url => stat(resolve(root, precacheFile(url)))))).reduce((sum, file) => sum + file.size, 0);
assert.ok(precacheBytes < 2.2 * 1024 * 1024, `service worker precache grew to ${precacheBytes} bytes`);
assert.doesNotMatch(serviceWorker, /url:"assets\/[^"]+",revision:"/, "content-hashed assets must not carry a revision");
for (const cache of ["drowse-hosted-on-demand-assets-v1", "drowse-hosted-landing-shader-v1", "drowse-hosted-on-demand-wasm-v2", "drowse-hosted-tab-icons-v1", "drowse-hosted-sae-descriptions-v1"]) {
  assert.match(serviceWorker, new RegExp(`cacheName:"${cache}",(?:matchOptions:\\{[^}]*\\},)?plugins:\\[new [A-Za-z_$][\\w$]*\\.ExpirationPlugin\\(`), `${cache} must be bounded`);
}
assert.doesNotMatch(serviceWorker, /the-uncertain-loom/);
const bundledWorkers = (await readdir(resolve(root, "assets"))).filter(name => /^fitting\.worker-.*\.js$/.test(name));
for (const name of bundledWorkers) {
  const source = await readFile(resolve(root, "assets", name), "utf8");
  for (const file of fittingManifest.files) {
    assert.ok(source.includes(`/wasm/${file.name}?v=${file.sha256.slice(0, 16)}`), `${name} must load ${file.name} by digest`);
  }
}
assert.doesNotMatch(serviceWorker, /huggingface|cdn\.hf\.co|model-.*\.safetensors/i);
if (release) {
  assert.doesNotMatch(headers.split(/\n\s*\n/)[0], /X-Robots-Tag:\s*noindex/i);
  assert.match(headers, /\/app\n  X-Robots-Tag: noindex, follow/);
  const origin = new URL(/<link rel="canonical" href="([^"]+)"/.exec(index)[1]).origin;
  assert.equal(robots, releaseRobots(origin));
  const resources = agentResources(origin);
  for (const path of agentCatalogPaths) {
    assert.equal(await readFile(resolve(root, path.slice(1)), "utf8"), resources.get(path.slice(1)), `${path} must describe the exact released skill`);
  }
} else {
  assert.match(headers, /X-Robots-Tag: noindex, nofollow/);
  assert.match(robots, /^User-agent: \*\nDisallow: \/\n$/);
}
assert.ok(headers.includes(`script-src 'self' ${cspSource}`), "CSP must allow exactly the inlined theme script");
for (const required of [
  "Strict-Transport-Security: max-age=31536000",
  "Content-Security-Policy-Report-Only: require-trusted-types-for 'script'",
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
for (const path of ["/video/*", "/icons/*", "/images/*"]) {
  assert.ok(headers.includes(`${path}\n  Cache-Control: public, max-age=86400, stale-while-revalidate=604800`), `${path} must revalidate`);
}
assert.ok(headers.includes("/social/*\n  ! Cross-Origin-Resource-Policy\n  Cross-Origin-Resource-Policy: cross-origin"), "share images must be embeddable by other sites");
const thirdPartyLicenses = await readFile(resolve(root, "third-party-licenses.txt"), "utf8");
for (const name of ["@drowse/web-llm", "svelte", "workbox-core"]) {
  assert.ok(thirdPartyLicenses.includes(`\n${name} `), `third-party notices must cover ${name}`);
}
assert.match(thirdPartyLicenses, /@drowse\/web-llm [^\n]+\nLicense: Apache-2\.0/);
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
