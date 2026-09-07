import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  accessHeaders,
  assertAccessProtected,
  assertChannelHtml,
  assertHostedManifest,
  assertRobotsHeader,
  assertRobotsFile,
  assertSecurityHeaders,
  deploymentOrigin,
  expectedCsp,
  parseArguments,
  workboxRuntimeName,
} from "./verify-hosted-deployment.mjs";

const revision = "a".repeat(40);
assert.equal(workboxRuntimeName('define(["./workbox-651d168f"], function () { precache([{url:"assets/workbox-window.prod.es5-Bd17z0YL.js"}]); });'), "workbox-651d168f.js");
assert.throws(() => workboxRuntimeName('precache([{url:"assets/workbox-window.prod.es5-Bd17z0YL.js"}]);'));
const distributionLock = {
  catalogUrl: "https://huggingface.co/a9lim/catalog/catalog.json",
  signatureUrl: "https://huggingface.co/a9lim/catalog/catalog.sig.json",
  allowedCatalogRedirectOrigins: ["https://huggingface.co"],
  allowedArtifactRedirectOrigins: ["https://cdn.example.test"],
};

function expectFailure(callback, pattern) {
  assert.throws(callback, pattern);
}

assert.deepEqual(
  parseArguments(["https://preview.example.test", "--channel", "preview"]),
  {
    help: false,
    baseUrl: "https://preview.example.test",
    channel: "preview",
    revision: null,
  },
);
assert.equal(
  parseArguments([
    "https://app.example.test",
    "--channel=release",
    `--revision=${revision}`,
  ]).revision,
  revision,
);
assert.deepEqual(
  parseArguments([
    "https://candidate.example.test",
    "--channel",
    "candidate",
    "--revision",
    revision,
  ]),
  {
    help: false,
    baseUrl: "https://candidate.example.test",
    channel: "candidate",
    revision,
  },
);
assert.equal(
  parseArguments(["https://app.example.test", "--channel", "release"]).revision,
  null,
);
assert.equal(
  parseArguments(["https://candidate.example.test", "--channel", "candidate"]).revision,
  null,
);
expectFailure(
  () => parseArguments(["https://preview.example.test", "--channel", "preview", "--revision", revision]),
  /does not accept --revision/,
);
assert.equal(deploymentOrigin("https://app.example.test/"), "https://app.example.test");
expectFailure(() => deploymentOrigin("http://app.example.test"), /requires HTTPS/);
expectFailure(() => deploymentOrigin("https://app.example.test/path"), /must be an origin/);

const accessEnvironment = {
  CF_ACCESS_CLIENT_ID: "client-id",
  CF_ACCESS_CLIENT_SECRET: "client-secret",
};
assert.deepEqual(accessHeaders("preview", accessEnvironment), {
  "CF-Access-Client-Id": "client-id",
  "CF-Access-Client-Secret": "client-secret",
});
assert.deepEqual(accessHeaders("candidate", accessEnvironment), {
  "CF-Access-Client-Id": "client-id",
  "CF-Access-Client-Secret": "client-secret",
});
expectFailure(() => accessHeaders("preview", {}), /requires CF_ACCESS_CLIENT_ID/);
expectFailure(() => accessHeaders("candidate", {}), /requires CF_ACCESS_CLIENT_ID/);
expectFailure(() => accessHeaders("release", accessEnvironment), /forbids Cloudflare Access/);
assert.deepEqual(accessHeaders("release", {}), {});

const protectedRequests = [];
await assertAccessProtected("https://candidate.example.test", async (url) => {
  protectedRequests.push(url.pathname);
  return new Response(null, { status: 302 });
});
assert.deepEqual(protectedRequests.sort(), ["/", "/app", "/manifest.webmanifest", "/sw.js"]);
await assert.rejects(
  () => assertAccessProtected("https://candidate.example.test", async (url) =>
    new Response(null, { status: url.pathname === "/app" ? 200 : 302 })
  ),
  /\/app is publicly reachable/,
);

const csp = [...expectedCsp(distributionLock)]
  .map(([name, values]) => `${name} ${values.join(" ")}`)
  .join("; ");
const headers = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
  "Content-Security-Policy": csp,
  "Permissions-Policy":
    "accelerometer=(), browsing-topics=(), camera=(), geolocation=(), microphone=(), payment=(), usb=()",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
};
assert.doesNotThrow(() => assertSecurityHeaders(new Response("", { headers }), distributionLock));
expectFailure(
  () => assertSecurityHeaders(
    new Response("", {
      headers: {
        ...headers,
        "Content-Security-Policy": csp.replace("; worker-src 'self' blob:", ""),
      },
    }),
    distributionLock,
  ),
  /directive set differs/,
);

const previewHtml = `
  <meta name="drowse-release-channel" content="preview">
  <meta name="drowse-source-revision" content="preview">
  <meta name="drowse-source-url" content="https://github.com/a9lim/drowse">`;
assert.doesNotThrow(() => assertChannelHtml(previewHtml, "preview"));
const releaseHtml = `
  <meta name="drowse-release-channel" content="release">
  <meta name="drowse-source-revision" content="${revision}">
  <meta name="drowse-source-url" content="https://github.com/a9lim/drowse/tree/${revision}">`;
assert.doesNotThrow(() => assertChannelHtml(releaseHtml, "release"));
assert.doesNotThrow(() => assertChannelHtml(releaseHtml, "candidate"));
assert.throws(() => assertChannelHtml(releaseHtml.replace(`/tree/${revision}`, ""), "release"));
assert.throws(() => assertChannelHtml(releaseHtml.replace(`/tree/${revision}`, `/tree/${"b".repeat(40)}`), "release"));
expectFailure(
  () => assertChannelHtml(releaseHtml, "release", "b".repeat(40)),
  /differs from --revision/,
);

const candidateHeaders = new Response("", {
  headers: { "X-Robots-Tag": "noindex" },
});
assert.doesNotThrow(() => assertRobotsHeader(candidateHeaders, "candidate"));
assert.doesNotThrow(() => assertRobotsFile("User-agent: *\nAllow: /\n", "candidate"));
expectFailure(
  () => assertRobotsHeader(new Response(""), "candidate"),
  /release candidate must be noindex/,
);
expectFailure(
  () => assertRobotsHeader(candidateHeaders, "release"),
  /release must not be noindex/,
);
for (const path of ["/app", "/app/saved-chat"]) {
  const response = new Response("", { headers: { "X-Robots-Tag": "noindex, follow" } });
  Object.defineProperty(response, "url", { value: `https://drowse.ai${path}` });
  assert.doesNotThrow(() => assertRobotsHeader(response, "release"));
  response.headers.set("X-Robots-Tag", "noindex, nofollow");
  assert.throws(() => assertRobotsHeader(response, "release"));
  response.headers.delete("X-Robots-Tag");
  assert.throws(() => assertRobotsHeader(response, "release"));
}

const manifest = {
  start_url: "/app",
  scope: "/",
  icons: [
    {
      src: "/icons/drowse-192.png",
      sizes: "192x192",
      type: "image/png",
    },
    {
      src: "/icons/drowse-512.png",
      sizes: "512x512",
      type: "image/png",
    },
    {
      src: "/icons/drowse-maskable-512.png",
      sizes: "512x512",
      type: "image/png",
      purpose: "maskable",
    },
  ],
};
assert.doesNotThrow(() => assertHostedManifest(manifest));
expectFailure(
  () => assertHostedManifest({ ...manifest, icons: [] }),
  /exact required Drowse icons/,
);
expectFailure(
  () => assertHostedManifest({
    ...manifest,
    icons: manifest.icons.map((icon, index) =>
      index === 2 ? { ...icon, purpose: "any" } : icon
    ),
  }),
  /exact required Drowse icons/,
);

const packageManifest = JSON.parse(
  await readFile(new URL("../package.json", import.meta.url), "utf8"),
);
assert.match(
  packageManifest.scripts["build:hosted:release"],
  /^node \.\/scripts\/check-runtime-lock\.mjs --release && node \.\/scripts\/preflight-hosted-distribution\.mjs/,
  "Hosted release lock validation must run before distribution downloads",
);
assert.doesNotMatch(
  packageManifest.scripts["build:hosted:release"],
  /release-provenance/,
  "Hosted releases must not require a Git provenance attestation",
);

console.log("Hosted release tools passed");
