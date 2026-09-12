import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { guidePages } from "./site-content.mjs";
import { publicPaths, markdownPath } from "./site-publication.mjs";
import { siteTitle } from "./site-metadata.mjs";

const target = process.argv[2];
if (!target) throw new Error("Usage: node scripts/verify-site-discovery.mjs https://drowse.ai [--preview]");
const origin = new URL(target).origin;
const preview = process.argv.includes("--preview");
const fetchPage = (path, options) => fetch(`${origin}${path}`, { signal: AbortSignal.timeout(30_000), ...options });
const sitemap = await fetchPage("/sitemap.xml");
assert.equal(sitemap.status, 200);
const locations = [...(await sitemap.text()).matchAll(/<loc>([^<]+)<\/loc>/g)].map(match => new URL(match[1]));
assert.deepEqual(locations.map(url => url.pathname), publicPaths);
const publicOrigin = locations[0].origin;
const robots = await (await fetchPage("/robots.txt")).text();
assert.match(robots, preview ? /Disallow: \// : /Allow: \//);
if (!preview) assert.ok(robots.includes(`Sitemap: ${publicOrigin}/sitemap.xml`));

for (const path of publicPaths) {
  const page = guidePages.find(page => page.path === path);
  const response = await fetchPage(path, { headers: { Accept: "text/html" } });
  assert.equal(response.status, 200, path);
  assert.match(response.headers.get("Content-Type"), /text\/html/);
  assert.equal((response.headers.get("X-Robots-Tag") ?? "").includes("noindex"), preview, `${path} indexing policy`);
  const html = await response.text();
  assert.ok(html.includes(`<title>${page?.title ?? siteTitle}</title>`), `${path} title`);
  assert.ok(html.includes(`rel="canonical" href="${publicOrigin}${path}"`), `${path} canonical`);
  assert.match(html, /<h1\b/);
  assert.match(html, /application\/ld\+json/);
  if (!page) assert.match(html, /data-prerendered/);
  else assert.ok(html.includes(page.heading));
  assert.ok(response.headers.get("Link")?.includes(`${publicOrigin}/llms.txt`));
  assert.match(response.headers.get("Vary"), /Accept/i);
  const markdown = await fetchPage(path, { headers: { Accept: "text/markdown" } });
  assert.equal(markdown.status, 200, `${path} Markdown`);
  assert.match(markdown.headers.get("Content-Type"), /text\/markdown/);
  assert.equal(await markdown.text(), await (await fetchPage(markdownPath(path))).text(), `${path} alternate parity`);
  const head = await fetchPage(path, { method: "HEAD", headers: { Accept: "text/markdown" } });
  assert.equal(head.status, 200);
  assert.match(head.headers.get("Content-Type"), /text\/markdown/);
  assert.equal(await head.text(), "");
  assert.doesNotMatch(html, /<a[^>]*href="\/(?:learn|developers|privacy|about)(?:[\/"?.])/);
}
for (const page of guidePages) {
  assert.equal((await fetchPage(page.path)).status, 404, `${page.path} must not expose a guide page`);
  assert.equal((await fetchPage(`${page.path}.html`)).status, 404);
  const response = await fetchPage(markdownPath(page.path));
  assert.equal(response.status, 200);
  assert.match(response.headers.get("Content-Type"), /text\/markdown/);
  assert.match(response.headers.get("X-Robots-Tag"), /noindex/);
  assert.ok((await response.text()).includes(page.heading));
}
const index = await fetchPage("/.well-known/agent-skills/index.json");
assert.match(index.headers.get("Content-Type"), /application\/json/);
const { skills } = await index.json();
for (const skill of skills) {
  const response = await fetchPage(new URL(skill.url, publicOrigin).pathname);
  assert.equal(response.status, 200);
  assert.equal(`sha256:${createHash("sha256").update(Buffer.from(await response.arrayBuffer())).digest("hex")}`, skill.digest);
}
for (const path of ["/missing-discovery-page", "/learn/missing-discovery-page", "/assets/missing-discovery-file.js"]) {
  assert.equal((await fetchPage(path)).status, 404, `${path} must remain a real 404`);
}
const app = await fetchPage("/app");
const appHtml = await app.text();
assert.equal(app.status, 200);
assert.match(appHtml, /name="robots" content="noindex, follow"/);
assert.doesNotMatch(appHtml, /data-prerendered|rel="canonical"/);
console.log(`Discovery verification passed: original homepage, no guide pages or visible links, ${guidePages.length} unindexed Markdown resources, Markdown GET/HEAD, sitemap, agent digest, app shell and real 404s (${preview ? "preview" : "release"}).`);
