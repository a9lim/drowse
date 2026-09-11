import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";
import { siteDescription, siteAccent, socialImagePath, socialImageAlt, publicOrigin, discoveryMetadata } from "./site-metadata.mjs";

test("public metadata requires an explicit, safe HTTPS origin", () => {
  assert.equal(publicOrigin(), "");
  assert.equal(discoveryMetadata(""), "");
  assert.equal(publicOrigin("https://drowse.example/"), "https://drowse.example");
  for (const input of ["http://example.com", "https://user:pass@example.com", "https://example.com/app", "https://example.com/?q=x", "https://example.com/#x"]) assert.throws(() => publicOrigin(input));
});

test("structured data describes only the website and actual application", () => {
  const html = discoveryMetadata("https://drowse.example");
  const scriptStart = '<script type="application/ld+json">';
  assert.ok(html.includes(scriptStart));
  assert.ok(html.endsWith("</script>"));
  const graph = JSON.parse(html.slice(html.indexOf(scriptStart) + scriptStart.length, -"</script>".length))["@graph"];
  assert.deepEqual(graph.map(item => item["@type"]), ["WebSite", "WebApplication"]);
  assert.ok(graph.every(item => item.name === "Drowse" && item.description === siteDescription));
  assert.ok(graph.every(item => !item.aggregateRating && !item.potentialAction));
  assert.equal(graph[1].image, `https://drowse.example${socialImagePath}`);
  assert.match(html, /rel="canonical" href="https:\/\/drowse.example\/"/);
});

test("SEO and social metadata use the requested description and Drowse lavender", async () => {
  assert.equal(siteDescription, "Drowse is an open-source and fully local AI mechanistic interpretability workbench that works fully in your browser. Use Drowse to research large language models, inspect predictions and change an LLMs internal activity.");
  assert.equal(siteAccent, "#c5b3ff");
  assert.ok((await readFile("index.html", "utf8")).includes(`name="description" content="${siteDescription}"`));
  assert.ok((await readFile("src/lib/style/tokens.css", "utf8")).includes(`--accent: ${siteAccent};`));
  assert.ok(socialImagePath.includes("?v=shader-orb"));
  assert.match(socialImageAlt, /shader orb/);
  for (const root of ["public", "public-hosted"]) {
    const artwork = await readFile(`${root}/social/drowse.svg`, "utf8");
    assert.deepEqual([...artwork.matchAll(/<text\b[^>]*>([^<]*)<\/text>/g)].map(match => match[1]), ["Drowse", "Interpretability Workbench"]);
    assert.match(artwork, /fill="#ffffff">Drowse<\/text>/);
    assert.match(artwork, /font-family="DrowseText, sans-serif"[^>]*>Interpretability Workbench<\/text>/);
    assert.ok(artwork.includes('<image href="data:image/png;base64,'));
    assert.ok(artwork.includes("data:font/woff2;base64,"));
  }
  assert.deepEqual(await readFile("public/social/drowse.png"), await readFile("public-hosted/social/drowse.png"));
});

test("social preview and icon assets have the declared PNG dimensions", async () => {
  const dimensions = async path => {
    const bytes = await readFile(path);
    assert.equal(bytes.subarray(1, 4).toString(), "PNG");
    return [bytes.readUInt32BE(16), bytes.readUInt32BE(20)];
  };
  for (const root of ["public", "public-hosted"]) {
    assert.deepEqual(await dimensions(`${root}/social/drowse.png`), [1200, 630]);
    assert.deepEqual(await dimensions(`${root}/icons/tab-home.png`), [96, 96]);
    assert.deepEqual(await dimensions(`${root}/icons/apple-touch-icon.png`), [180, 180]);
  }
});
