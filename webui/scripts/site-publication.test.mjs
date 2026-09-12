import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { parse as parseYaml } from "yaml";
import { guidePages, homeSummary } from "./site-content.mjs";
import { discoveryLinks, discoveryTags, guideMarkdown, markdownPath, publicationAssets, publicPaths, sitePublication } from "./site-publication.mjs";
import { prefersMarkdown } from "./discovery-worker.mjs";
import { prerenderLanding } from "./prerender-hosted.mjs";

const origin = "https://drowse.example";
const assets = publicationAssets(origin);
const built = new Map();
await sitePublication(origin).generateBundle.call({ emitFile({ fileName, source }) { built.set(fileName, source); } });
const { default: worker } = await import(`data:text/javascript;base64,${Buffer.from(built.get("_worker.js")).toString("base64")}`);
test("machine-readable documentation preserves its source content without creating visible guide pages", () => {
  assert.deepEqual(publicPaths, ["/"]);
  assert.ok(![...assets.keys()].some(path => path.endsWith(".html")));
  for (const page of guidePages) {
    const markdown = assets.get(markdownPath(page.path).slice(1));
    const content = [page.heading, page.intro, ...page.sections.flatMap(section => [
      section.heading, ...(section.paragraphs ?? []), ...(section.bullets ?? []),
      ...(section.code ? [section.code] : []), ...(section.links ?? []).flatMap(link => [link.label, ...(link.description ? [link.description] : [])]),
    ])];
    for (const sentence of content) {
      assert.ok(markdown.includes(sentence), `${page.path}: Markdown is missing ${sentence}`);
    }
    assert.doesNotMatch(markdown, /__DROWSE_|undefined/);
  }
});

test("Markdown examples preserve literal code and point internal references to machine-readable resources", () => {
  const literal = 'if (score < 0.5 && label === "<script>") { steer("a&b"); }';
  const page = { path: "/learn/escaping", heading: "Code examples", intro: "Preserve the code.",
    sections: [{ heading: "Syntax", code: literal, links: [{ label: "Reference", href: guidePages[0].path }] }] };
  const markdown = guideMarkdown(page, origin);
  assert.ok(markdown.includes(`\`\`\`\n${literal}\n\`\`\``));
  assert.ok(markdown.includes(`](${origin}${markdownPath(guidePages[0].path)})`));
  assert.doesNotMatch(markdown, /&lt;|&amp;|&quot;/);
});

test("only the existing homepage appears in the sitemap and every machine-discovery link resolves", () => {
  const locations = [...assets.get("sitemap.xml").matchAll(/<loc>([^<]+)<\/loc>/g)].map(match => match[1]);
  assert.deepEqual(locations, [origin + "/"]);
  assert.ok(!publicationAssets("").has("sitemap.xml"));
  const existing = new Set(["/", "/app", "/credits", "/contact", "/LICENSE"]);
  function resolves(href, from) {
    const url = new URL(href, origin + from);
    if (url.origin !== origin) return;
    const key = url.pathname.slice(1);
    assert.ok(existing.has(url.pathname) || assets.has(key), `${from} links to unpublished ${href}`);
  }
  for (const [filename, value] of assets) {
    if (filename.endsWith(".md") || filename.endsWith(".txt")) {
      for (const [, href] of value.matchAll(/\]\(([^)]+)\)/g)) resolves(href, "/" + filename);
    }
  }
  for (const [, href] of discoveryTags().matchAll(/href="([^"]+)"/g)) resolves(href, "/");
  for (const [, href] of discoveryLinks(origin, "/").matchAll(/<([^>]+)>/g)) resolves(href, "/");
});

test("agent discovery digest identifies the exact published UTF-8 skill bytes", () => {
  const index = JSON.parse(assets.get(".well-known/agent-skills/index.json"));
  assert.ok(index.skills.length > 0);
  for (const skill of index.skills) {
    const url = new URL(skill.url);
    assert.equal(url.origin, origin);
    assert.equal(skill.type, "skill-md");
    const content = assets.get(url.pathname.slice(1));
    assert.equal(typeof content, "string");
    assert.equal(skill.digest, `sha256:${createHash("sha256").update(Buffer.from(content, "utf8")).digest("hex")}`);
    const metadata = parseYaml(/^---\n([\s\S]*?)\n---\n/.exec(content)[1]);
    assert.equal(metadata.name, skill.name);
    assert.equal(metadata.description, skill.description);
  }
});

test("Markdown negotiation honors explicit exclusions, preferences, and media-range specificity", () => {
  for (const [accept, expected] of [
    ["", false], ["*/*", false], ["text/*", false], ["text/html", false],
    ["text/markdown", true], ["text/html,text/markdown", true],
    ["text/markdown;q=0,text/*;q=1", false],
    ["text/markdown;q=0.8,text/html;q=0.9", false],
    ["text/html;q=0,text/markdown;q=0.5,*/*;q=1", true],
    ["text/html;q=0.6,text/markdown;q=0.6", true],
    ["TEXT/MARKDOWN; Q=0.7, TEXT/HTML; Q=0.5", true],
    ["text/markdown;q=invalid", false], ["text/markdown;q=1.1", false],
  ]) assert.equal(prefersMarkdown(accept), expected, accept);
});

function assetEnvironment({ status = 200, vary = "Origin", contentType = "text/html; charset=utf-8", robots = "noindex, nofollow" } = {}) {
  const requests = [];
  return { requests, env: { ASSETS: { async fetch(request) {
    requests.push(request);
    return new Response(status === 200 ? "asset content" : "missing", { status, headers: {
      "Content-Type": contentType, "Vary": vary, "X-Robots-Tag": robots,
      "Cross-Origin-Opener-Policy": "same-origin", "Cache-Control": "public, max-age=60",
    } });
  } } } };
}

test("generated worker negotiates homepage GET and HEAD while preserving preview and resource indexing controls", async () => {
  for (const [method, robots] of [["GET", "noindex, nofollow"], ["HEAD", "noindex, nofollow"], ["GET", "noindex, follow"], ["HEAD", "noindex, follow"]]) {
    const { env, requests } = assetEnvironment({ robots });
    const response = await worker.fetch(new Request(`${origin}/?ref=agent`, { method, headers: {
      Accept: "text/markdown", "X-Request-Probe": "preserved",
    } }), env);
    assert.equal(requests[0].url, `${origin}/index.md?ref=agent`);
    assert.equal(requests[0].method, method);
    assert.equal(requests[0].headers.get("X-Request-Probe"), "preserved");
    assert.equal(response.status, 200);
    assert.match(response.headers.get("Content-Type"), /^text\/markdown;/);
    assert.equal(response.headers.get("X-Robots-Tag"), robots);
    assert.equal(response.headers.get("Cross-Origin-Opener-Policy"), "same-origin");
    assert.equal(response.headers.get("Cache-Control"), "public, max-age=60");
    assert.equal(response.headers.get("Vary"), "Origin, Accept");
    assert.ok(response.headers.get("Link").includes(`<${origin}/>; rel="canonical"`));
    assert.equal(await response.text(), method === "HEAD" ? "" : "asset content");
  }
});

test("worker serves HTML when Markdown is excluded and preserves existing Vary tokens", async () => {
  for (const vary of ["Accept, Origin", "accept", "*"]) {
    const { env, requests } = assetEnvironment({ vary });
    const response = await worker.fetch(new Request(`${origin}/`, { headers: { Accept: "text/markdown;q=0,*/*;q=1" } }), env);
    assert.equal(requests[0].url, `${origin}/`);
    assert.equal(response.headers.get("Content-Type"), "text/html; charset=utf-8");
    assert.equal(response.headers.get("Vary"), vary);
  }
});

test("worker does not intercept mutations or unlisted paths", async () => {
  for (const [path, method] of [["/", "POST"], ["/", "PUT"], ["/", "DELETE"], ["/app", "GET"], ["/learn", "GET"], ["/developers", "GET"], ["/developers.md", "GET"], ["/missing", "GET"]]) {
    const { env, requests } = assetEnvironment();
    const request = new Request(origin + path, { method, headers: { Accept: "text/markdown" } });
    const response = await worker.fetch(request, env);
    assert.equal(requests[0], request);
    assert.equal(response.headers.get("Vary"), "Origin");
    assert.equal(response.headers.get("Link"), null);
    assert.equal(response.headers.get("Content-Type"), "text/html; charset=utf-8");
  }
});

test("missing negotiated resources remain errors with their original content type", async () => {
  const { env } = assetEnvironment({ status: 404, contentType: "text/plain" });
  const response = await worker.fetch(new Request(`${origin}/`, { headers: { Accept: "text/markdown" } }), env);
  assert.equal(response.status, 404);
  assert.equal(response.headers.get("Content-Type"), "text/plain");
  assert.equal(response.headers.get("Link"), null);
  assert.equal(await response.text(), "missing");
});

test("publication keeps previews blocked, machine resources unindexed, and worker scope limited to the homepage", () => {
  const headers = built.get("_headers");
  assert.match(headers.split(/\n\s*\n/)[0], /X-Robots-Tag: noindex, nofollow/);
  assert.match(headers, /Content-Security-Policy:/);
  for (const path of assets.keys()) {
    if (!path.endsWith(".md") && !path.endsWith(".txt") && !path.endsWith(".json")) continue;
    const block = headers.split(/\n\s*\n/).find(block => block.trimStart().startsWith(`/${path}\n`));
    assert.ok(block, `${path} needs resource headers`);
    if (path === "index.md") {
      assert.match(block, /Link: <https:\/\/drowse.example\/>; rel="canonical"/);
      assert.doesNotMatch(block, /X-Robots-Tag:/);
    } else assert.match(block, /X-Robots-Tag: noindex, follow/);
    if (path.endsWith(".md")) assert.match(block, /Content-Type: text\/markdown; charset=utf-8/);
  }
  const routes = JSON.parse(built.get("_routes.json"));
  assert.deepEqual(routes.include, ["/"]);
});

test("release preparation removes the global preview block while preserving document-specific noindex", async () => {
  const fixture = await mkdtemp(join(tmpdir(), "drowse-publication-release-test-"));
  const webui = join(fixture, "webui");
  const output = join(webui, "dist-hosted");
  const license = await readFile(new URL("../../LICENSE", import.meta.url), "utf8");
  const revision = "1234567890abcdef1234567890abcdef12345678";
  const index = `<html><head>
<meta name="drowse-source-revision" content="${revision}" />
<meta name="drowse-release-channel" content="release" />
<meta name="drowse-source-url" content="https://github.com/a9lim/drowse/tree/${revision}" />
<link rel="canonical" href="${origin}/" />
</head><body>Release fixture</body></html>`;
  try {
    await mkdir(output, { recursive: true });
    await Promise.all([
      writeFile(join(output, "_headers"), built.get("_headers")),
      writeFile(join(output, "index.html"), index),
      writeFile(join(output, "robots.txt"), "User-agent: *\nDisallow: /\n"),
      writeFile(join(output, "LICENSE"), license),
      writeFile(join(fixture, "LICENSE"), license),
    ]);
    execFileSync(process.execPath, [fileURLToPath(new URL("./prepare-hosted-release.mjs", import.meta.url))], {
      cwd: webui, timeout: 10_000, stdio: "pipe",
    });
    const headers = await readFile(join(output, "_headers"), "utf8");
    const blocks = headers.split(/\n\s*\n/);
    assert.doesNotMatch(blocks[0], /X-Robots-Tag:\s*noindex/);
    assert.match(blocks[0], /Content-Security-Policy:/);
    for (const path of ["/app", "/app/*", "/developers.md", "/llms.txt", "/skills/drowse/SKILL.md", "/.well-known/agent-skills/index.json"]) {
      const block = blocks.find(block => block.trimStart().startsWith(`${path}\n`));
      assert.ok(block, `${path} must retain an indexing policy`);
      assert.match(block, /X-Robots-Tag: noindex, follow/);
    }
    const robots = await readFile(join(output, "robots.txt"), "utf8");
    assert.match(robots, /User-agent: \*\nAllow: \//);
    assert.ok(robots.includes(`Sitemap: ${origin}/sitemap.xml`));
    assert.doesNotMatch(robots, /Disallow: \//);
  } finally {
    await rm(fixture, { recursive: true, force: true });
  }
});

test("Vite hooks install middleware without returning a Connect handler as a post-hook", () => {
  const plugin = sitePublication(origin);
  for (const hook of [plugin.configureServer, plugin.configurePreviewServer]) {
    let installed;
    const connect = { use(handler) { installed = handler; return connect; } };
    assert.equal(hook({ middlewares: connect }), undefined);
    assert.equal(typeof installed, "function");
    const headers = new Map();
    let body;
    installed({ url: "/", method: "GET", headers: { accept: "text/markdown" } }, {
      setHeader(name, value) { headers.set(name, value); }, end(value) { body = value; },
    }, () => assert.fail("Negotiated homepage Markdown should be served by middleware"));
    assert.match(headers.get("Content-Type"), /text\/markdown/);
    assert.match(body, /# Drowse/);
    let passed = false;
    installed({ url: "/developers", method: "GET", headers: { accept: "text/html" } }, {
      setHeader() {}, end() { assert.fail("Removed guide pages must not be served"); },
    }, () => { passed = true; });
    assert.ok(passed);
  }
});

test("prerender preserves the actual landing content without visible machine-resource links or hidden keyword copy", async () => {
  const images = [];
  const { body, css } = await prerenderLanding({ assetUrl(filename, source) {
    assert.ok(source.byteLength > 0);
    images.push(filename);
    return `/assets/${basename(filename)}`;
  } });
  assert.match(body, /See inside/);
  assert.match(body, /Inspect the prediction/);
  assert.match(body, /Gemma 3 1B/);
  assert.match(body, /href="\/app"/);
  assert.doesNotMatch(body + css, /\/@fs\/|\/src\/|__DROWSE_/);
  assert.ok(!body.includes(homeSummary.intro));
  const visibleLinks = [...body.matchAll(/<a\b[^>]*href="([^"]+)"/g)].map(match => match[1]);
  for (const href of visibleLinks) {
    const path = new URL(href, origin).pathname;
    assert.ok(!assets.has(path.slice(1)), `Landing must not expose machine resource ${href}`);
    assert.ok(!guidePages.some(page => page.path === path), `Landing must not expose removed guide route ${href}`);
  }
  assert.ok(images.length > 0);
  assert.match(css, /\.landing-shell\.svelte-/);
});
