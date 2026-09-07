import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  validateReleaseCorpusInputs,
  validateReleaseInstrumentPack,
} from "./hosted-instrument-release-policy.mjs";
import { selectHostedJlensLayers } from "./hosted-instrument-layers.mjs";

const repositoryRoot = resolve(import.meta.dirname, "../..");
const runtimeLock = JSON.parse(await readFile(
  resolve(repositoryRoot, "browser-runtime/runtime-lock.json"),
  "utf8",
));
const modelLock = runtimeLock.models.find((model) => model.id === "qwen3-1.7b");
const source = JSON.parse(await readFile(
  resolve(repositoryRoot, "browser-runtime/release-corpus/wikitext-103-raw-test.candidate.json"),
  "utf8",
));
const lockedTokenizerClosure = [...runtimeLock.models]
  .sort((left, right) => left.id.localeCompare(right.id))
  .map((model) => ({
    id: model.id,
    sourceRepository: model.sourceRepository,
    sourceRevision: model.sourceRevision,
    tokenizerSha256: model.tokenizerSha256,
    tokenizerConfigSha256: model.chatTemplateSha256,
  }));
assert.deepEqual(source.tokenizers, lockedTokenizerClosure);
for (const manifestName of [
  "fineweb-sample-10bt-train.approved.json",
  "fineweb-sample-10bt-train.candidate.json",
]) {
  const manifest = JSON.parse(await readFile(
    resolve(repositoryRoot, "browser-runtime/release-corpus", manifestName),
    "utf8",
  ));
  assert.deepEqual(manifest.tokenizers, lockedTokenizerClosure);
}
source.status = "approved";
source.blockers = [];
source.source.split = "train";
source.source.path = source.source.path.replace("test-", "train-");
source.source.url = source.source.url.replace("/test-", "/train-");
source.license.reviewStatus = "approved";
source.license.reviewedBy = "release-policy fixture";
source.license.reviewedAt = "2026-08-30T00:00:00.000Z";
source.license.redistributionNotice = "Release-policy test fixture; not a distribution grant.";
const generatorBytes = await readFile(
  resolve(repositoryRoot, "webui/scripts/prepare_hosted_release_corpus.py"),
);
const licenseEvidenceBytes = Buffer.from("release-policy license evidence fixture\n");
source.license.evidence.sha256 = sha256(licenseEvidenceBytes);
source.license.evidence.bytes = licenseEvidenceBytes.byteLength;
const updatedSourceBytes = Buffer.from(`${JSON.stringify(source, null, 2)}\n`);
const seedWords = (await readFile(
  resolve(repositoryRoot, "browser-runtime/release-corpus/jlens-word-seeds.txt"),
  "utf8",
)).trim().split("\n").slice(0, 80);
const saeBytes = Buffer.from(JSON.stringify(
  Array.from({ length: source.selection.saeDocuments }, (_, index) => `sae document ${index}`),
));
const jlensBytes = Buffer.from(JSON.stringify(
  Array.from({ length: source.selection.jlensDocuments }, (_, index) => `jlens document ${index}`),
));
const wordBytes = Buffer.from(JSON.stringify(seedWords));
const release = {
  $schema: "https://polythetic.ai/schemas/release-corpus-v1.json",
  schemaVersion: 1,
  algorithm: "drowse-release-corpus-v1",
  generatorSha256: sha256(generatorBytes),
  sourceManifestSha256: sha256(updatedSourceBytes),
  source: source.source,
  license: source.license,
  wordSeeds: source.wordSeeds,
  selection: source.selection,
  outputs: {
    sae: output("sae-corpus.json", saeBytes, source.selection.saeDocuments),
    jlens: output("jlens-corpus.json", jlensBytes, source.selection.jlensDocuments),
    words: {
      path: "jlens-words.json",
      sha256: sha256(wordBytes),
      bytes: wordBytes.byteLength,
      count: seedWords.length,
      candidateCount: 123,
      tokenIds: Object.fromEntries(runtimeLock.models.map((model) => [
        model.id,
        seedWords.map((_, index) => index + 1),
      ])),
    },
  },
};
const releaseBytes = Buffer.from(`${JSON.stringify(release, null, 2)}\n`);
const root = await mkdtemp(join(tmpdir(), "drowse-instrument-release-policy-"));
const releaseManifestPath = join(root, "release-corpus.json");
const sourceManifestPath = join(root, "approved-source.json");
await Promise.all([
  writeFile(releaseManifestPath, releaseBytes),
  writeFile(sourceManifestPath, updatedSourceBytes),
  writeFile(join(root, "license-evidence"), licenseEvidenceBytes),
]);

const inputClosure = await validateReleaseCorpusInputs({
  releaseManifestPath,
  sourceManifestPath,
  licenseEvidencePath: join(root, "license-evidence"),
  kind: "sae",
  corpusBytes: saeBytes,
  runtimeLock,
});
assert.deepEqual(inputClosure, {
  releaseBytes,
  sourceBytes: updatedSourceBytes,
  licenseEvidenceBytes,
});
await assert.rejects(
  validateReleaseCorpusInputs({
    releaseManifestPath,
    sourceManifestPath,
    licenseEvidencePath: join(root, "license-evidence"),
    kind: "jlens",
    corpusBytes: jlensBytes,
    wordsBytes: Buffer.from("[]"),
    runtimeLock,
  }),
  /word bytes differ/,
);

const saeDirectory = join(root, "sae");
const jlensDirectory = join(root, "jlens");
await Promise.all([
  mkdir(join(saeDirectory, "packs/sae"), { recursive: true }),
  mkdir(join(jlensDirectory, "packs/jlens"), { recursive: true }),
]);
for (const directory of [join(saeDirectory, "packs/sae"), join(jlensDirectory, "packs/jlens")]) {
  await Promise.all([
    writeFile(join(directory, "release-corpus.json"), releaseBytes),
    writeFile(join(directory, "release-corpus-source.json"), updatedSourceBytes),
    writeFile(join(directory, "release-license-evidence"), licenseEvidenceBytes),
  ]);
}
const saeManifest = {
  corpus_sha256: release.outputs.sae.sha256,
  d_model: modelLock.hiddenSize,
  d_sae: Math.min(modelLock.hiddenSize * 4, 8192),
  tokens_trained: 50_000,
  seq_len: 128,
};
assert.equal((await validateReleaseInstrumentPack({
  directory: saeDirectory,
  kind: "sae",
  instrumentManifest: saeManifest,
  runtimeLock,
  modelLock,
})).tokensTrained, 50_000);
await assert.rejects(
  validateReleaseInstrumentPack({
    directory: saeDirectory,
    kind: "sae",
    instrumentManifest: { ...saeManifest, tokens_trained: 49_999 },
    runtimeLock,
    modelLock,
  }),
  /release SAE requires/,
);

const jlensManifest = {
  raw_corpus_sha256: release.outputs.jlens.sha256,
  d_model: modelLock.hiddenSize,
  model_layer_count: modelLock.layerMap.length,
  source_layers: modelLock.layerMap.slice(0, -1),
  n_prompts: 100,
  raw_prompt_count: 100,
  usable_prompt_count: 100,
  seq_len: 128,
};
const vocabularyManifest = {
  words: seedWords.map((word, row) => ({ word, row, token_id: row + 1 })),
};
assert.equal((await validateReleaseInstrumentPack({
  directory: jlensDirectory,
  kind: "jlens",
  instrumentManifest: jlensManifest,
  vocabularyManifest,
  runtimeLock,
  modelLock,
})).words, 80);
await assert.rejects(
  validateReleaseInstrumentPack({
    directory: jlensDirectory,
    kind: "jlens",
    instrumentManifest: { ...jlensManifest, source_layers: [0] },
    vocabularyManifest,
    runtimeLock,
    modelLock,
  }),
  /complete layer map/,
);

assert.deepEqual(await validateReleaseInstrumentPack({
  directory: join(root, "provider-sae"),
  kind: "sae",
  instrumentManifest: {
    activation: "jump_relu",
    corpus_spec: "provider:fixture/sae",
    d_model: modelLock.hiddenSize,
    d_sae: 16_384,
    layer: modelLock.layerMap[1],
  },
  runtimeLock,
  modelLock,
}), {
  source: "provider",
  layer: modelLock.layerMap[1],
  features: 16_384,
  activation: "jump_relu",
});
assert.deepEqual(await validateReleaseInstrumentPack({
  directory: join(root, "provider-jlens"),
  kind: "jlens",
  instrumentManifest: {
    d_model: modelLock.hiddenSize,
    method: "provider_jacobian_lens",
    model_layer_count: modelLock.layerMap.length,
    n_prompts: 278,
    seq_len: 128,
    source_layers: selectHostedJlensLayers(modelLock.layerMap, modelLock.hiddenSize),
  },
  vocabularyManifest: { words: [{ word: "yes", row: 0, token_id: 1 }] },
  runtimeLock,
  modelLock,
}), {
  source: "provider",
  prompts: 278,
  sequenceLength: 128,
  layers: selectHostedJlensLayers(modelLock.layerMap, modelLock.hiddenSize).length,
  words: 1,
});

console.log("Hosted instrument release-policy checks passed");

function output(path, bytes, documents) {
  return {
    path,
    sha256: sha256(bytes),
    bytes: bytes.byteLength,
    documents,
    characters: bytes.toString("utf8").length,
  };
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
