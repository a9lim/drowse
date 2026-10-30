import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
// Plain node import, no vite: drowse.ai's metadata test loads this module the
// same way, so it must not grow an extensionless value import.
import { loadSaeDescription } from "../src/lib/saeDescriptions.ts";

// The `description_source` the local server sends for a SAELens gemma-scope-2
// release (drowse/sae/sources.py::list_sae_sources): the saeId form, no folder.
const local = [
  ["270m", 12, "range"],
  ["1b", 13, "Reuters"],
  ["4b", 17, "AI, drones, space, luxury"],
].map(([size, layer, label]) => ({
  binding: {
    model: `gemma-3-${size}-it`, source: `${layer}-gemmascope-2-res-16k`,
    repository: `google/gemma-scope-2-${size}-it`, saeId: `layer_${layer}_width_16k_l0_medium`,
  },
  label,
}));

const originalFetch = globalThis.fetch;
const requested = [];
try {
  globalThis.fetch = async (url, options) => {
    assert.ok(url.includes("/sae-descriptions/"), `local labels resolve offline, not from ${url}`);
    assert.equal(options.credentials, "omit");
    requested.push(url);
    return Response.json(JSON.parse(await readFile(new URL(url), "utf8")));
  };
  for (const { binding, label } of local) {
    const description = await loadSaeDescription(binding, 2286, new AbortController().signal);
    assert.equal(description.label, label);
    assert.equal(description.url, `https://www.neuronpedia.org/${binding.model}/${binding.source}/2286`);
  }
  assert.equal(requested.length, 3, "one index download per published dictionary");
} finally {
  globalThis.fetch = originalFetch;
}
console.log("SAE descriptions: local gemma-scope-2 releases label features from the bundled indexes");
