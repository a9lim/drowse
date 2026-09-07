import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const usage = `Usage:
  CF_ACCESS_CLIENT_ID=... CF_ACCESS_CLIENT_SECRET=... \\
    npm run verify:hosted:deployment -- https://PREVIEW_HOST --channel preview
  CF_ACCESS_CLIENT_ID=... CF_ACCESS_CLIENT_SECRET=... \\
    npm run verify:hosted:deployment -- https://CANDIDATE_HOST --channel candidate
  npm run verify:hosted:deployment -- https://RELEASE_HOST --channel release

Preview and candidate verification require a Cloudflare Access service token
and prove that shell routes reject unauthenticated requests. Candidate builds
carry release metadata but remain noindex. Release verification forbids Access
credentials and proves anonymous public access without noindex.`;

const permissionsPolicy =
  "accelerometer=(), browsing-topics=(), camera=(), geolocation=(), microphone=(), payment=(), usb=()";
const protectedPaths = ["/", "/app", "/manifest.webmanifest", "/sw.js"];
const requiredManifestIcons = [
  {
    src: "/icons/drowse-192.png",
    sizes: "192x192",
    type: "image/png",
    purpose: "any",
  },
  {
    src: "/icons/drowse-512.png",
    sizes: "512x512",
    type: "image/png",
    purpose: "any",
  },
  {
    src: "/icons/drowse-maskable-512.png",
    sizes: "512x512",
    type: "image/png",
    purpose: "maskable",
  },
];

export function parseArguments(argv) {
  const positional = [];
  let channel = null;
  let revision = null;
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--help" || argument === "-h") return { help: true };
    if (argument === "--channel" || argument === "--revision") {
      const value = argv[index + 1];
      assert.ok(value && !value.startsWith("-"), `${argument} requires a value`);
      if (argument === "--channel") channel = value;
      else revision = value;
      index += 1;
      continue;
    }
    if (argument.startsWith("--channel=")) {
      channel = argument.slice("--channel=".length);
      continue;
    }
    if (argument.startsWith("--revision=")) {
      revision = argument.slice("--revision=".length);
      continue;
    }
    if (argument.startsWith("-")) throw new Error(`Unknown option: ${argument}`);
    positional.push(argument);
  }
  assert.equal(positional.length, 1, "Provide exactly one deployed base URL");
  assert.ok(
    channel === "preview" || channel === "candidate" || channel === "release",
    "--channel must be preview, candidate, or release",
  );
  if (revision !== null) {
    assert.match(
      revision,
      /^[0-9a-f]{40}$/,
      "--revision must be a lowercase 40-character commit SHA",
    );
  }
  if (channel === "preview") {
    assert.equal(revision, null, "Preview verification does not accept --revision");
  }
  return { help: false, baseUrl: positional[0], channel, revision };
}

export function deploymentOrigin(value) {
  const url = new URL(value);
  assert.equal(url.protocol, "https:", "Hosted deployment verification requires HTTPS");
  assert.equal(url.username, "", "The deployment URL must not contain credentials");
  assert.equal(url.password, "", "The deployment URL must not contain credentials");
  assert.equal(url.search, "", "The deployment URL must not contain a query string");
  assert.equal(url.hash, "", "The deployment URL must not contain a fragment");
  assert.ok(url.pathname === "/" || url.pathname === "", "The deployment URL must be an origin");
  return url.origin;
}

export function accessHeaders(channel, environment = process.env) {
  const clientId = environment.CF_ACCESS_CLIENT_ID;
  const clientSecret = environment.CF_ACCESS_CLIENT_SECRET;
  assert.equal(
    Boolean(clientId),
    Boolean(clientSecret),
    "Set both CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET, or neither",
  );
  if (channel === "preview" || channel === "candidate") {
    assert.ok(
      clientId && clientSecret,
      `${channel === "preview" ? "Preview" : "Candidate"} verification requires ` +
        "CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET",
    );
    return {
      "CF-Access-Client-Id": clientId,
      "CF-Access-Client-Secret": clientSecret,
    };
  }
  assert.ok(
    !clientId && !clientSecret,
    "Release verification forbids Cloudflare Access credentials",
  );
  return {};
}

function normalizedHeader(response, name) {
  return response.headers.get(name)?.trim().toLowerCase() ?? "";
}

function assertHeader(response, name, expected) {
  assert.equal(
    normalizedHeader(response, name),
    expected.toLowerCase(),
    `${response.url} has the wrong ${name} header`,
  );
}

export function parseCsp(value) {
  const directives = new Map();
  for (const segment of value.split(";")) {
    const [name, ...tokens] = segment.trim().split(/\s+/);
    if (!name) continue;
    assert.ok(!directives.has(name), `CSP repeats ${name}`);
    directives.set(name, tokens);
  }
  return directives;
}

export function expectedCsp(distributionLock) {
  const connectOrigins = [...new Set([
    new URL(distributionLock.catalogUrl).origin,
    new URL(distributionLock.signatureUrl).origin,
    ...distributionLock.allowedCatalogRedirectOrigins,
    ...distributionLock.allowedArtifactRedirectOrigins,
  ])].sort();
  return new Map([
    ["default-src", ["'self'"]],
    ["base-uri", ["'self'"]],
    ["connect-src", ["'self'", ...connectOrigins]],
    ["font-src", ["'self'"]],
    ["form-action", ["'self'"]],
    ["frame-ancestors", ["'none'"]],
    ["img-src", ["'self'", "data:"]],
    ["manifest-src", ["'self'"]],
    ["object-src", ["'none'"]],
    ["script-src", ["'self'", "'wasm-unsafe-eval'"]],
    ["style-src", ["'self'", "'unsafe-inline'"]],
    ["worker-src", ["'self'", "blob:"]],
  ]);
}

export function assertSecurityHeaders(response, distributionLock) {
  assertHeader(response, "Cross-Origin-Opener-Policy", "same-origin");
  assertHeader(response, "Cross-Origin-Embedder-Policy", "require-corp");
  assertHeader(response, "Cross-Origin-Resource-Policy", "same-origin");
  assertHeader(response, "X-Content-Type-Options", "nosniff");
  assertHeader(response, "Referrer-Policy", "no-referrer");
  assertHeader(response, "X-Frame-Options", "deny");
  assertHeader(response, "Permissions-Policy", permissionsPolicy);

  const csp = normalizedHeader(response, "Content-Security-Policy");
  assert.ok(csp, `${response.url} is missing Content-Security-Policy`);
  const actual = parseCsp(csp);
  const expected = expectedCsp(distributionLock);
  assert.deepEqual(
    [...actual.keys()].sort(),
    [...expected.keys()].sort(),
    "Deployed CSP directive set differs from the hosted policy",
  );
  for (const [name, tokens] of expected) {
    assert.deepEqual(actual.get(name), tokens, `CSP ${name} must be ${tokens.join(" ")}`);
  }
}

export function metadata(html, name) {
  for (const match of html.matchAll(/<meta\b[^>]*>/gi)) {
    const attributes = new Map();
    for (const attribute of match[0].matchAll(/([\w:-]+)\s*=\s*["']([^"']*)["']/g)) {
      attributes.set(attribute[1].toLowerCase(), attribute[2]);
    }
    if (attributes.get("name") === name) return attributes.get("content") ?? "";
  }
  throw new Error(`Hosted index is missing ${name}`);
}

async function fetchRoute(origin, path, headers) {
  const response = await fetch(new URL(path, origin), {
    cache: "no-store",
    credentials: "omit",
    headers,
    redirect: "manual",
    referrerPolicy: "no-referrer",
    signal: AbortSignal.timeout(20_000),
  });
  assert.equal(response.status, 200, `${path} returned HTTP ${response.status}`);
  assert.equal(
    new URL(response.url).origin,
    origin,
    `${path} redirected away from the deployment origin; Cloudflare Access may need credentials`,
  );
  return response;
}

export async function assertAccessProtected(origin, fetcher = fetch) {
  await Promise.all(protectedPaths.map(async (path) => {
    const response = await fetcher(new URL(path, origin), {
      cache: "no-store",
      credentials: "omit",
      redirect: "manual",
      referrerPolicy: "no-referrer",
      signal: AbortSignal.timeout(20_000),
    });
    await response.body?.cancel().catch(() => undefined);
    assert.notEqual(
      response.status,
      200,
      `${path} is publicly reachable without Cloudflare Access credentials`,
    );
  }));
}

export function assertChannelHtml(html, channel, expectedRevision = null) {
  const buildChannel = channel === "preview" ? "preview" : "release";
  assert.equal(
    metadata(html, "drowse-release-channel"),
    buildChannel,
    `Deployed HTML is not a ${buildChannel} build`,
  );
  const revision = metadata(html, "drowse-source-revision");
  const sourceUrl = metadata(html, "drowse-source-url");
  if (buildChannel === "release") {
    assert.match(revision, /^[0-9a-f]{40}$/, "Release HTML has an invalid source revision");
    if (expectedRevision !== null) {
      assert.equal(revision, expectedRevision, "Release HTML revision differs from --revision");
    }
    assert.equal(sourceUrl, "https://github.com/a9lim/polythetic");
  } else {
    assert.equal(revision, "preview", "Preview HTML must not claim a release revision");
    assert.equal(sourceUrl, "https://github.com/a9lim/polythetic");
  }
}

export function assertRobotsHeader(response, channel) {
  const value = normalizedHeader(response, "X-Robots-Tag");
  if (channel === "preview") {
    assert.match(value, /(?:^|[,\s])noindex(?:$|[,\s])/, `${response.url} must be noindex`);
    assert.match(value, /(?:^|[,\s])nofollow(?:$|[,\s])/, `${response.url} must be nofollow`);
  } else if (channel === "candidate") {
    assert.match(
      value,
      /(?:^|[,\s])noindex(?:$|[,\s])/,
      `${response.url} release candidate must be noindex`,
    );
  } else {
    assert.ok(!value.includes("noindex"), `${response.url} release must not be noindex`);
    assert.ok(!value.includes("nofollow"), `${response.url} release must not be nofollow`);
  }
}

export function assertRobotsFile(body, channel) {
  const normalized = body.replace(/\r\n/g, "\n").trim();
  if (channel === "preview") assert.equal(normalized, "User-agent: *\nDisallow: /", "preview robots.txt must block indexing");
  else assert.match(normalized, /^User-agent: \*\nAllow: \/(?:\n\nSitemap: https:\/\/[^\s]+\/sitemap\.xml)?$/, `robots.txt does not match the ${channel} channel`);
}

export function assertHostedManifest(manifest) {
  assert.ok(manifest && typeof manifest === "object", "Deployed manifest is not an object");
  assert.equal(manifest.start_url, "/app", "Deployed manifest has the wrong start_url");
  assert.equal(manifest.scope, "/", "Deployed manifest has the wrong scope");
  const icons = Array.isArray(manifest.icons)
    ? manifest.icons.map((icon) => ({
        src: icon?.src,
        sizes: icon?.sizes,
        type: icon?.type,
        purpose: icon?.purpose ?? "any",
      }))
    : [];
  assert.deepEqual(
    icons,
    requiredManifestIcons,
    "Deployed manifest does not contain the exact required Drowse icons",
  );
}

function assertMime(response, expected) {
  const mime = normalizedHeader(response, "Content-Type").split(";", 1)[0];
  const accepted = Array.isArray(expected) ? expected : [expected];
  assert.ok(accepted.includes(mime), `${response.url} has unexpected MIME type ${mime || "none"}`);
}

function cacheDirectives(response) {
  return normalizedHeader(response, "Cache-Control")
    .split(",")
    .map((directive) => directive.trim())
    .filter(Boolean)
    .sort();
}

function assertCache(response, expected) {
  assert.deepEqual(
    cacheDirectives(response),
    [...expected].sort(),
    `${response.url} has the wrong Cache-Control policy`,
  );
}

async function assertPwaAssets(origin, headers, html) {
  const manifestResponse = await fetchRoute(origin, "/manifest.webmanifest", headers);
  assertMime(manifestResponse, ["application/manifest+json", "application/json"]);
  assertCache(manifestResponse, ["no-cache"]);
  const manifest = await manifestResponse.json();
  assertHostedManifest(manifest);

  const serviceWorkerResponse = await fetchRoute(origin, "/sw.js", headers);
  assertMime(serviceWorkerResponse, ["application/javascript", "text/javascript"]);
  assertCache(serviceWorkerResponse, ["no-cache"]);
  const serviceWorker = await serviceWorkerResponse.text();
  const workboxName = /\b(workbox-[A-Za-z0-9._-]+\.js)\b/.exec(serviceWorker)?.[1];
  assert.ok(workboxName, "Deployed service worker does not reference a Workbox runtime");

  const workboxResponse = await fetchRoute(origin, `/${workboxName}`, headers);
  assertMime(workboxResponse, ["application/javascript", "text/javascript"]);
  assertCache(workboxResponse, ["no-cache"]);
  assert.ok((await workboxResponse.arrayBuffer()).byteLength > 1_000, "Workbox runtime is empty");

  const assetPath = [...html.matchAll(/(?:src|href)=["'](\/assets\/[^"']+)["']/g)]
    .map((match) => match[1])
    .find((path) => /-[A-Za-z0-9_-]{6,}\.(?:css|js|woff2)$/.test(path));
  assert.ok(assetPath, "Deployed HTML has no hashed application asset");
  const assetResponse = await fetchRoute(origin, assetPath, headers);
  assertCache(assetResponse, ["public", "max-age=31536000", "immutable"]);
  assertMime(
    assetResponse,
    assetPath.endsWith(".css")
      ? "text/css"
      : assetPath.endsWith(".woff2")
        ? "font/woff2"
        : ["application/javascript", "text/javascript"],
  );
  assert.ok((await assetResponse.arrayBuffer()).byteLength > 0, `${assetPath} is empty`);

  for (const icon of manifest.icons ?? []) {
    const iconResponse = await fetchRoute(origin, icon.src, headers);
    assertMime(iconResponse, "image/png");
    assert.ok((await iconResponse.arrayBuffer()).byteLength > 1_000, `${icon.src} is empty`);
    assert.ok(!cacheDirectives(iconResponse).includes("no-store"), `${icon.src} must remain cacheable`);
  }

  const licenseResponse = await fetchRoute(origin, "/LICENSE", headers);
  assertMime(licenseResponse, ["text/plain", "application/octet-stream"]);
  assert.ok(
    !cacheDirectives(licenseResponse).includes("no-store"),
    "The project license must remain cacheable",
  );
  const [deployedLicense, sourceLicense] = await Promise.all([
    licenseResponse.text(),
    readFile(new URL("../../LICENSE", import.meta.url), "utf8"),
  ]);
  assert.equal(deployedLicense, sourceLicense, "Deployed AGPL license differs from source");
}

async function assertBrowserBehavior(origin, headers) {
  const browser = await chromium.launch({ headless: true });
  let context;
  try {
    context = await browser.newContext({
      extraHTTPHeaders: headers,
      serviceWorkers: "allow",
    });
    if (Object.keys(headers).length > 0) {
      await context.route("**/*", async (route) => {
        if (new URL(route.request().url()).origin === origin) {
          await route.continue();
          return;
        }
        await route.abort("blockedbyclient");
      });
    }
    const page = await context.newPage();

    await page.goto(`${origin}/`, { waitUntil: "domcontentloaded" });
    await page.locator("#hero-title").waitFor({ state: "visible" });
    assert.equal(await page.evaluate(() => globalThis.crossOriginIsolated), true, "/ is not cross-origin isolated");

    await page.goto(`${origin}/app`, { waitUntil: "domcontentloaded" });
    await page.locator("#device-check").waitFor({ state: "visible" });
    assert.equal(await page.evaluate(() => globalThis.crossOriginIsolated), true, "/app is not cross-origin isolated");

    await page.evaluate(async () => {
      if (!("serviceWorker" in navigator)) throw new Error("Service workers are unavailable");
      await Promise.race([
        navigator.serviceWorker.ready,
        new Promise((_, reject) => {
          setTimeout(() => reject(new Error("Service worker did not become ready")), 20_000);
        }),
      ]);
    });
    if (!(await page.evaluate(() => Boolean(navigator.serviceWorker.controller)))) {
      await page.reload({ waitUntil: "domcontentloaded" });
    }
    await page.waitForFunction(() => Boolean(navigator.serviceWorker.controller));

    const offlineProbe = `${origin}/app?offline_probe=${crypto.randomUUID()}`;
    await context.setOffline(true);
    await page.goto(offlineProbe, { waitUntil: "domcontentloaded" });
    await page.locator("#device-check").waitFor({ state: "visible" });
    assert.equal(
      await page.evaluate(() => globalThis.crossOriginIsolated),
      true,
      "Uncached offline /app navigation lost cross-origin isolation",
    );
  } finally {
    if (context) await context.setOffline(false).catch(() => {});
    await browser.close();
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.help) {
    console.log(usage);
    return;
  }
  const origin = deploymentOrigin(options.baseUrl);
  const headers = accessHeaders(options.channel);
  const distributionLock = JSON.parse(
    await readFile(new URL("../../browser-runtime/distribution-lock.json", import.meta.url), "utf8"),
  );

  if (options.channel !== "release") await assertAccessProtected(origin);
  const [rootResponse, appResponse, robotsResponse] = await Promise.all([
    fetchRoute(origin, "/", headers),
    fetchRoute(origin, "/app", headers),
    fetchRoute(origin, "/robots.txt", headers),
  ]);
  const [rootHtml, appHtml, robots] = await Promise.all([
    rootResponse.text(),
    appResponse.text(),
    robotsResponse.text(),
  ]);

  for (const response of [rootResponse, appResponse]) {
    assertMime(response, "text/html");
    assertSecurityHeaders(response, distributionLock);
    assertRobotsHeader(response, options.channel);
  }
  assertChannelHtml(rootHtml, options.channel, options.revision);
  assertChannelHtml(appHtml, options.channel, options.revision);
  assertRobotsFile(robots, options.channel);
  await assertPwaAssets(origin, headers, rootHtml);
  await assertBrowserBehavior(origin, headers);

  console.log(`Hosted ${options.channel} deployment passed at ${origin}`);
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : "";
if (invokedPath === fileURLToPath(import.meta.url)) {
  await main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
