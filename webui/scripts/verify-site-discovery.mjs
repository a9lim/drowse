import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { guidePages, sitePages } from "./site-content.mjs";
import { publicPaths, sitemapPaths, markdownPath } from "./site-publication.mjs";
import { siteTitle } from "./site-metadata.mjs";
import { agentCatalogPaths, agentResources, releaseRobots } from "./agent-discovery.mjs";

const target = process.argv[2];
if (!target) throw new Error("Usage: node scripts/verify-site-discovery.mjs https://drowse.ai [--preview]");
const origin = new URL(target).origin;
const preview = process.argv.includes("--preview");
const fetchPage = (path, options) => fetch(`${origin}${path}`, { signal: AbortSignal.timeout(30_000), ...options });
const sitemap = await fetchPage("/sitemap.xml");
assert.equal(sitemap.status, 200);
const locations = [...(await sitemap.text()).matchAll(/<loc>([^<]+)<\/loc>/g)].map(match => new URL(match[1]));
assert.deepEqual(locations.map(url => url.pathname), sitemapPaths);
const publicOrigin = locations[0].origin;
const robots = await (await fetchPage("/robots.txt")).text();
assert.match(robots, preview ? /Disallow: \// : /Allow: \//);
if (!preview) assert.equal(robots, releaseRobots(publicOrigin));

const escapeHtml = text => text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
for (const path of sitemapPaths) {
  const page = [...guidePages, ...sitePages].find(page => page.path === path);
  const negotiated = publicPaths.includes(path);
  const response = await fetchPage(path, { headers: { Accept: "text/html" } });
  assert.equal(response.status, 200, path);
  assert.match(response.headers.get("Content-Type"), /text\/html/);
  assert.equal((response.headers.get("X-Robots-Tag") ?? "").includes("noindex"), preview, `${path} indexing policy`);
  const html = await response.text();
  assert.ok(html.includes(`<title>${escapeHtml(page?.title ?? siteTitle)}</title>`), `${path} title`);
  assert.ok(html.includes(`rel="canonical" href="${publicOrigin}${path}"`), `${path} canonical`);
  assert.match(html, /<h1\b/);
  assert.match(html, /application\/ld\+json/);
  assert.match(html, /data-prerendered="[^"]+"/, `${path} must be prerendered`);
  assert.doesNotMatch(html, /name="robots"/, `${path} must not carry a noindex meta tag`);
  if (page?.heading) assert.ok(html.includes(escapeHtml(page.heading)), `${path} heading`);
  const trailing = await fetchPage(`${path}/`, { redirect: "manual" });
  if (path !== "/") assert.ok([301, 308].includes(trailing.status) && new URL(trailing.headers.get("Location"), origin).pathname === path, `${path}/ must redirect to ${path}`);
  if (!negotiated) continue;
  assert.ok(response.headers.get("Link")?.includes(`${publicOrigin}/llms.txt`));
  assert.ok(response.headers.get("Link")?.includes(`<${publicOrigin}/.well-known/ai-catalog.json>; rel="ai-catalog"`));
  assert.match(html, /rel="ai-catalog" href="\/\.well-known\/ai-catalog\.json"/);
  assert.ok(response.headers.get("Link")?.includes(`<${publicOrigin}/.well-known/ard.json>; rel="ard"`));
  assert.match(html, /rel="ard" href="\/\.well-known\/ard\.json"/);
  assert.match(response.headers.get("Vary"), /Accept/i);
  const markdown = await fetchPage(path, { headers: { Accept: "text/markdown" } });
  assert.equal(markdown.status, 200, `${path} Markdown`);
  assert.match(markdown.headers.get("Content-Type"), /text\/markdown/);
  assert.equal(await markdown.text(), await (await fetchPage(markdownPath(path))).text(), `${path} alternate parity`);
  const head = await fetchPage(path, { method: "HEAD", headers: { Accept: "text/markdown" } });
  assert.equal(head.status, 200);
  assert.match(head.headers.get("Content-Type"), /text\/markdown/);
  assert.equal(await head.text(), "");
}
for (const page of guidePages) {
  if (!publicPaths.includes(page.path)) {
    assert.equal((await fetchPage(page.path)).status, 404, `${page.path} is a Markdown-only resource`);
    assert.equal((await fetchPage(`${page.path}.html`)).status, 404);
  }
  const response = await fetchPage(markdownPath(page.path));
  assert.equal(response.status, 200);
  assert.match(response.headers.get("Content-Type"), /text\/markdown/);
  assert.match(response.headers.get("X-Robots-Tag"), /noindex/);
  assert.ok((await response.text()).includes(page.heading));
}
const index = await fetchPage("/.well-known/agent-skills/index.json");
assert.equal(index.status, 200);
assert.match(index.headers.get("Content-Type"), /application\/json/);
const { skills } = await index.json();
for (const skill of skills) {
  const response = await fetchPage(new URL(skill.url, publicOrigin).pathname);
  assert.equal(response.status, 200);
  assert.equal(`sha256:${createHash("sha256").update(Buffer.from(await response.arrayBuffer())).digest("hex")}`, skill.digest);
}
const expectedResources = agentResources(publicOrigin);
for (const path of agentCatalogPaths) {
  for (const method of ["GET", "HEAD"]) {
    const response = await fetchPage(path, { method, headers: { Origin: "https://agent.example" } });
    assert.equal(response.status, 200, `${method} ${path}`);
    assert.match(response.headers.get("Content-Type"), /^application\/json(?:;|$)/);
    assert.equal(response.headers.get("Access-Control-Allow-Origin"), "*");
    assert.equal(await response.text(), method === "HEAD" ? "" : expectedResources.get(path.slice(1)), `${path} published content`);
  }
}
for (const path of ["/missing-discovery-page", "/learn/missing-discovery-page", "/assets/missing-discovery-file.js", "/app-shell", "/app/missing-discovery-page"]) {
  assert.equal((await fetchPage(path)).status, 404, `${path} must remain a real 404`);
}
const app = await fetchPage("/app");
const appHtml = await app.text();
assert.equal(app.status, 200);
assert.match(appHtml, /name="robots" content="noindex, follow"/);
assert.doesNotMatch(appHtml, /data-prerendered|rel="canonical"/);
console.log(`Discovery verification passed: ${sitemapPaths.length} prerendered pages with canonical URLs and sitemap entries, ${guidePages.length} Markdown resources, Markdown GET/HEAD, agent digest, ARD catalogs and CORS, content policy, app shell and real 404s (${preview ? "preview" : "release"}).`);
