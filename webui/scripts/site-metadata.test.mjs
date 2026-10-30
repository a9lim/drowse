import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";
import { siteDescription, siteAccent, socialImagePath, socialImageType, socialImageAlt, publicOrigin, discoveryMetadata, pageDiscoveryMetadata, projectOwner } from "./site-metadata.mjs";

function readJsonLd(html) {
  const scriptStart = '<script type="application/ld+json">';
  assert.ok(html.includes(scriptStart));
  assert.ok(html.endsWith("</script>"));
  return JSON.parse(html.slice(html.indexOf(scriptStart) + scriptStart.length, -"</script>".length));
}

test("public metadata requires an explicit, safe HTTPS origin", () => {
  assert.equal(publicOrigin(), "");
  assert.equal(discoveryMetadata(""), "");
  assert.equal(publicOrigin("https://drowse.example/"), "https://drowse.example");
  for (const input of ["http://example.com", "https://user:pass@example.com", "https://example.com/app", "https://example.com/?q=x", "https://example.com/#x"]) assert.throws(() => publicOrigin(input));
});

test("structured data describes only the website and actual application", () => {
  const html = discoveryMetadata("https://drowse.example", { version: "5.3.0" });
  const graph = readJsonLd(html)["@graph"];
  assert.deepEqual(graph.map(item => item["@type"]), ["WebSite", "WebApplication"]);
  assert.ok(graph.every(item => item.name === "Drowse" && item.description === siteDescription));
  assert.ok(graph.every(item => !item.aggregateRating && !item.potentialAction));
  assert.equal(graph[1].image, `https://drowse.example${socialImagePath}`);
  assert.equal(graph[1].url, "https://drowse.example/");
  assert.equal(graph[1].softwareVersion, "5.3.0");
  assert.deepEqual(graph[1].offers, { "@type": "Offer", price: "0", priceCurrency: "USD" });
  assert.deepEqual(graph[1].author, projectOwner);
  assert.deepEqual(graph[1].publisher, { "@type": "Person", name: "a9lim", url: "https://github.com/a9lim" });
  assert.ok(!("screenshot" in graph[1]), "no product screenshot asset exists to publish");
  assert.ok(!("softwareVersion" in readJsonLd(discoveryMetadata("https://drowse.example"))["@graph"][1]));
  assert.match(html, /rel="canonical" href="https:\/\/drowse.example\/"/);
});

test("prerendered pages get their own canonical URL and WebPage data", () => {
  assert.equal(pageDiscoveryMetadata("", { path: "/privacy", title: "t", description: "d" }), "");
  const html = pageDiscoveryMetadata("https://drowse.example", { path: "/privacy", title: "Privacy <Drowse>", description: "Local data" });
  assert.match(html, /<link rel="canonical" href="https:\/\/drowse.example\/privacy" \/>/);
  assert.match(html, /<meta property="og:url" content="https:\/\/drowse.example\/privacy" \/>/);
  assert.doesNotMatch(html, /<Drowse>/, "JSON-LD must not open markup");
  const page = readJsonLd(html);
  assert.equal(page["@type"], "WebPage");
  assert.equal(page.name, "Privacy <Drowse>");
  assert.deepEqual(page.isPartOf, { "@id": "https://drowse.example/#website" });
});

test("SEO and social metadata use the requested description and Drowse lavender", async () => {
  assert.equal(siteDescription, "Open-source, fully local LLM interpretability workbench in your browser. Inspect token predictions and steer a model's internal activity.");
  assert.ok(siteDescription.length <= 160, "search snippets truncate longer descriptions");
  assert.doesNotMatch(siteDescription, /an LLMs|fully[^.]*fully/);
  assert.equal(siteAccent, "#c5b3ff");
  assert.ok((await readFile("index.html", "utf8")).includes(`name="description" content="${siteDescription}"`), "the Python dashboard's index.html must use the shared description");
  assert.ok((await readFile("hosted/index.html", "utf8")).includes('content="__DROWSE_HOSTED_DESCRIPTION__"'));
  assert.ok((await readFile("src/lib/style/tokens.css", "utf8")).includes(`--accent: ${siteAccent};`));
  assert.ok(socialImagePath.startsWith("/social/drowse.jpg?v=shader-orb"));
  assert.equal(socialImageType, "image/jpeg");
  assert.match(socialImageAlt, /shader orb/);
  // The SVG source of the artwork ships only with the dashboard; the hosted site publishes rasters.
  await assert.rejects(readFile("public-hosted/social/drowse.svg"), { code: "ENOENT" });
  for (const root of ["public"]) {
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
  const jpeg = await readFile("public-hosted/social/drowse.jpg");
  assert.deepEqual([...jpeg.subarray(0, 3)], [0xff, 0xd8, 0xff], "share image must be a JPEG");
  assert.ok(jpeg.byteLength < 200 * 1024, `share image is ${jpeg.byteLength} bytes`);
  let offset = 2;
  while (offset < jpeg.byteLength && !(jpeg[offset + 1] >= 0xc0 && jpeg[offset + 1] <= 0xc3)) offset += 2 + jpeg.readUInt16BE(offset + 2);
  assert.deepEqual([jpeg.readUInt16BE(offset + 7), jpeg.readUInt16BE(offset + 5)], [1200, 630]);
});
