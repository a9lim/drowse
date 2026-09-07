import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";
import { siteDescription, publicOrigin, discoveryMetadata } from "./site-metadata.mjs";

test("public metadata requires an explicit, safe HTTPS origin", () => {
  assert.equal(publicOrigin(), "");
  assert.equal(discoveryMetadata(""), "");
  assert.equal(publicOrigin("https://drowse.example/"), "https://drowse.example");
  for (const input of ["http://example.com", "https://user:pass@example.com", "https://example.com/app", "https://example.com/?q=x", "https://example.com/#x"]) assert.throws(() => publicOrigin(input));
});

test("structured data describes only the website and actual application", () => {
  const html = discoveryMetadata("https://drowse.example");
  const graph = JSON.parse(/<script[^>]*>(.*?)<\/script>/.exec(html)[1])["@graph"];
  assert.deepEqual(graph.map(item => item["@type"]), ["WebSite", "WebApplication"]);
  assert.ok(graph.every(item => item.name === "Drowse" && item.description === siteDescription));
  assert.ok(graph.every(item => !item.aggregateRating && !item.potentialAction));
  assert.match(html, /rel="canonical" href="https:\/\/drowse.example\/"/);
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
