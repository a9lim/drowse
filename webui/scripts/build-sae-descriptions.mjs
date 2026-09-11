import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import { PUBLISHED_DICTIONARIES, verifiedSaeDescriptionSource } from "../src/lib/saeDescriptions.ts";

const exportOrigin = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com";
const destination = new URL("../src/lib/data/sae-descriptions/", import.meta.url);
const provenance = { retrievedAt: new Date().toISOString(), publisher: "Neuronpedia", dictionaries: [] };
await mkdir(destination, { recursive: true });

async function download(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(30_000) });
  if (!response.ok) throw new Error(`${response.status} downloading ${url}`);
  return Buffer.from(await response.arrayBuffer());
}

for (const dictionary of PUBLISHED_DICTIONARIES) {
  const source = verifiedSaeDescriptionSource({
    model_id: `google/gemma-3-${dictionary.size}-it`, layer: dictionary.layer,
    corpus_spec: `provider:google/gemma-scope-2-${dictionary.size}-it`,
    corpus_sha256: dictionary.sha256, d_sae: 16384, activation: "jump_relu",
  });
  assert.ok(source);
  const example = JSON.parse(await download(`https://www.neuronpedia.org/api/feature/${source.model}/${source.source}/0`));
  assert.equal(example.modelId, source.model);
  assert.equal(example.layer, source.source);
  assert.equal(example.source.hfRepoId, source.repository);
  assert.equal(example.source.hfFolderId, source.folder);
  const files = JSON.parse(await download(`https://huggingface.co/api/models/${source.repository}/tree/main/${source.folder}`));
  assert.equal(files.find(file => file.path.endsWith("/params.safetensors"))?.lfs?.oid, dictionary.sha256);

  const prefix = `v1/${source.model}/${source.source}/explanations/`;
  const listing = (await download(`${exportOrigin}/?list-type=2&prefix=${prefix}`)).toString();
  assert.ok(listing.includes("<IsTruncated>false</IsTruncated>"));
  const batches = [...listing.matchAll(/<Key>([^<]+)<\/Key>/g)].map(match => match[1])
    .filter(key => key !== `${prefix}config.json`);
  assert.ok(batches.length > 0);
  assert.ok(batches.every(key => key.startsWith(prefix) && /^batch-\d+\.jsonl\.gz$/.test(key.slice(prefix.length))));
  batches.sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const exports = [];
  const byFeature = new Map();
  for (let start = 0; start < batches.length; start += 4) {
    const group = await Promise.all(batches.slice(start, start + 4).map(async key => {
      const bytes = await download(`${exportOrigin}/${key}`);
      return {
        key, sha256: createHash("sha256").update(bytes).digest("hex"), bytes: bytes.length,
        rows: gunzipSync(bytes).toString().trim().split("\n").map(line => JSON.parse(line)),
      };
    }));
    for (const { rows, ...receipt } of group) {
      exports.push(receipt);
      for (const row of rows) {
        assert.equal(row.modelId, source.model);
        assert.equal(row.layer, source.source);
        const id = Number(row.index);
        assert.ok(Number.isSafeInteger(id) && id >= 0 && id < 16384);
        if (typeof row.description !== "string" || !row.description.trim()) continue;
        const prior = byFeature.get(id);
        if (!prior || String(row.createdAt) > String(prior.createdAt)) byFeature.set(id, row);
      }
    }
  }
  const explanations = Object.fromEntries([...byFeature.entries()].sort(([a], [b]) => a - b).map(([id, row]) =>
    [id, [row.description.trim(), row.explanationModelName ?? null]]));
  const payload = JSON.stringify({ format_version: 1, source, provider_sha256: dictionary.sha256, feature_count: 16384, explanations });
  const filename = `gemma-3-${dictionary.size}-it.json`;
  await writeFile(new URL(filename, destination), `${payload}\n`);
  provenance.dictionaries.push({
    source, providerSha256: dictionary.sha256, filename, descriptions: byFeature.size,
    sha256: createHash("sha256").update(`${payload}\n`).digest("hex"), exports,
  });
  console.log(`${filename}: ${byFeature.size}/16384 published descriptions, ${Buffer.byteLength(payload)} bytes`);
}
await writeFile(new URL("provenance.json", destination), `${JSON.stringify(provenance, null, 2)}\n`);
