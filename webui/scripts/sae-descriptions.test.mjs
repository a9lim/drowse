import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { verifiedSaeDescriptionSource, parseSaeDescription, loadSaeDescription } from "../src/lib/saeDescriptions.ts";

const headers = await readFile(new URL("../public-hosted/_headers", import.meta.url), "utf8");
assert.match(headers, /connect-src[^;]*https:\/\/www\.neuronpedia\.org(?:\s|;)/);

const manifest = {
  model_id: "google/gemma-3-1b-it", layer: 13, d_sae: 16384, activation: "jump_relu",
  corpus_spec: "provider:google/gemma-scope-2-1b-it",
  corpus_sha256: "6be6fbfed9850588adb1826bcc9fc89551d9eaf573311e95b21307fd4a281cd5",
};
const binding = verifiedSaeDescriptionSource(manifest);
assert.ok(binding);
for (const [size, layer, hash] of [
  ["270m", 12, "f99efac8449faa1e5b715e0298f27d5cf3f5646fce5a5ece14ded90a09efeb65"],
  ["1b", 13, manifest.corpus_sha256],
  ["4b", 17, "f67e2305cfd1703d8a685d99fb25b6f25a2feaaa4eb45ef08a467f400b696d71"],
]) {
  for (const owner of ["google", "unsloth"]) {
    const pack = { ...manifest, model_id: `${owner}/gemma-3-${size}-it`, layer,
      corpus_spec: `provider:google/gemma-scope-2-${size}-it`, corpus_sha256: hash };
    assert.deepEqual(verifiedSaeDescriptionSource(pack), {
      model: `gemma-3-${size}-it`, source: `${layer}-gemmascope-2-res-16k`,
      repository: `google/gemma-scope-2-${size}-it`, folder: `resid_post/layer_${layer}_width_16k_l0_medium`,
    });
    assert.equal(verifiedSaeDescriptionSource({ ...pack, corpus_sha256: "a".repeat(64) }), null);
    assert.equal(verifiedSaeDescriptionSource({ ...pack, layer: layer + 1 }), null);
  }
}
for (const patch of [
  { model_id: "google/gemma-3-1b-pt" }, { layer: 12 }, { d_sae: 65536 },
  { activation: "relu" }, { corpus_sha256: "a".repeat(64) }, { corpus_spec: "local" },
]) assert.equal(verifiedSaeDescriptionSource({ ...manifest, ...patch }), null);
const payload = {
  modelId: binding.model, layer: binding.source, index: "16190",
  source: { hfRepoId: binding.repository, hfFolderId: binding.folder },
  explanations: [{ description: "terms and phrases", explanationModelName: "gemini-2.5-flash-lite" }],
};
assert.equal(parseSaeDescription(payload, binding, 16190).label, "terms and phrases");
assert.equal(parseSaeDescription({ ...payload, explanations: [] }, binding, 16190).label, null);
for (const patch of [
  { modelId: "gemma-3-1b" }, { layer: "12-gemmascope-2-res-16k" }, { index: "396" },
  { source: { ...payload.source, hfFolderId: "resid_post/layer_13_width_16k_l0_small" } },
  { explanations: null },
]) assert.throws(() => parseSaeDescription({ ...payload, ...patch }, binding, 16190));

const originalFetch = globalThis.fetch;
let calls = 0;
try {
  globalThis.fetch = async (url, options) => {
    calls++;
    assert.equal(url, "https://www.neuronpedia.org/api/feature/gemma-3-1b-it/13-gemmascope-2-res-16k/16190");
    assert.equal(options.credentials, "omit");
    assert.equal(options.referrerPolicy, "no-referrer");
    assert.equal(options.body, undefined);
    if (calls === 1) return new Response("offline", { status: 503 });
    return Response.json(payload);
  };
  const signal = new AbortController().signal;
  await assert.rejects(loadSaeDescription(binding, 16190, signal));
  const result = await loadSaeDescription(binding, 16190, signal);
  assert.equal(result.label, "terms and phrases");
  assert.deepEqual(await loadSaeDescription(binding, 16190, signal), result);
  assert.equal(calls, 2);
} finally {
  globalThis.fetch = originalFetch;
}
console.log("SAE descriptions: exact dictionary binding, provenance, missing labels, retry, cache, and request privacy passed");

if (process.argv.includes("--live-catalog")) {
  const catalog = await (await fetch("https://huggingface.co/logitsml/drowse-web-catalog/resolve/main/catalog.json")).json();
  for (const model of catalog.models) for (const variant of model.variants) {
    for (const pack of variant.packs.filter(pack => pack.kind === "sae")) {
      const file = pack.files.find(file => file.path.endsWith("/manifest.json"));
      const manifest = await (await fetch(file.url)).json();
      const binding = verifiedSaeDescriptionSource(manifest);
      assert.ok(binding, `${pack.id} needs a verified published dictionary or bundled descriptions`);
      const files = await (await fetch(`https://huggingface.co/api/models/${binding.repository}/tree/main/${binding.folder}`)).json();
      assert.equal(files.find(file => file.path.endsWith("/params.safetensors"))?.lfs?.oid, manifest.corpus_sha256);
      const description = await loadSaeDescription(binding, 2286, AbortSignal.timeout(20_000));
      assert.ok(description.label, `${pack.id}: known published feature should have a description`);
      console.log(`${pack.id}: verified provider hash and Neuronpedia description (${description.label})`);
    }
  }
}
