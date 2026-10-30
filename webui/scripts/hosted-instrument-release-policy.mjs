import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { selectHostedJlensLayers } from "./hosted-instrument-layers.mjs";

const repositoryRoot = resolve(import.meta.dirname, "../..");
const SHA256 = /^[0-9a-f]{64}$/;
const COMMIT = /^[0-9a-f]{40}$/;
const RELEASE_CORPUS_SCHEMA = "https://polythetic.ai/schemas/release-corpus-v1.json";
const RELEASE_CORPUS_ALGORITHM = "drowse-release-corpus-v1";
const MINIMUM_JLENS_PROMPTS = 100;
const MINIMUM_JLENS_SEQUENCE = 128;
const MINIMUM_SHARED_WORDS = 80;
const MINIMUM_SAE_TOKENS = 50_000;
const MINIMUM_SAE_SEQUENCE = 128;

export async function validateReleaseInstrumentPack({
  directory,
  kind,
  instrumentManifest,
  vocabularyManifest = null,
  runtimeLock,
  modelLock,
}) {
  if (providerInstrument(kind, instrumentManifest)) {
    return validateProviderInstrument(
      kind,
      instrumentManifest,
      vocabularyManifest,
      modelLock,
    );
  }
  const root = resolve(directory, "packs", kind);
  const [
    releaseBytes,
    sourceBytes,
    licenseEvidenceBytes,
    generatorBytes,
    wordSeedBytes,
  ] = await Promise.all([
    readRequired(resolve(root, "release-corpus.json"), "release corpus manifest"),
    readRequired(resolve(root, "release-corpus-source.json"), "release corpus source manifest"),
    readRequired(resolve(root, "release-license-evidence"), "release license evidence"),
    readFile(resolve(repositoryRoot, "webui/scripts/prepare_hosted_release_corpus.py")),
    readFile(resolve(repositoryRoot, "browser-runtime/release-corpus/jlens-word-seeds.txt")),
  ]);
  const release = parseJson(releaseBytes, "release corpus manifest");
  const source = parseJson(sourceBytes, "release corpus source manifest");
  validateReleaseCorpus({
    release,
    source,
    sourceBytes,
    generatorBytes,
    wordSeedBytes,
    licenseEvidenceBytes,
    runtimeLock,
    modelLock,
  });
  if (kind === "sae") {
    return validateSae(instrumentManifest, release, modelLock);
  }
  if (kind === "jlens") {
    return validateJlens(instrumentManifest, vocabularyManifest, release, modelLock);
  }
  throw new Error(`release policy does not support instrument kind ${String(kind)}`);
}

function providerInstrument(kind, manifest) {
  return kind === "sae"
    ? typeof manifest?.corpus_spec === "string" && manifest.corpus_spec.startsWith("provider:")
    : kind === "jlens" && manifest?.method === "provider_jacobian_lens";
}

function validateProviderInstrument(kind, manifest, vocabulary, modelLock) {
  if (kind === "sae") {
    if (
      manifest.d_model !== modelLock.hiddenSize ||
      !modelLock.layerMap.includes(manifest.layer) ||
      !positiveInteger(manifest.d_sae) ||
      manifest.activation !== "jump_relu"
    ) throw new Error("provider SAE does not match the runtime shape");
    return {
      source: "provider",
      layer: manifest.layer,
      features: manifest.d_sae,
      activation: manifest.activation,
    };
  }
  const expectedLayers = selectHostedJlensLayers(
    modelLock.layerMap,
    modelLock.hiddenSize,
  );
  if (
    manifest.d_model !== modelLock.hiddenSize ||
    manifest.model_layer_count !== modelLock.layerMap.length ||
    canonical(manifest.source_layers) !== canonical(expectedLayers) ||
    !positiveInteger(manifest.n_prompts) ||
    !positiveInteger(manifest.seq_len) ||
    !vocabulary || !Array.isArray(vocabulary.words) || vocabulary.words.length === 0
  ) throw new Error("provider J-lens does not match the runtime shape");
  return {
    source: "provider",
    prompts: manifest.n_prompts,
    sequenceLength: manifest.seq_len,
    layers: expectedLayers.length,
    words: vocabulary.words.length,
  };
}

export async function validateReleaseCorpusInputs({
  releaseManifestPath,
  sourceManifestPath,
  licenseEvidencePath,
  kind,
  corpusBytes,
  wordsBytes = null,
  runtimeLock,
  modelLock,
}) {
  const [
    releaseBytes,
    sourceBytes,
    licenseEvidenceBytes,
    generatorBytes,
    wordSeedBytes,
  ] = await Promise.all([
    readFile(releaseManifestPath),
    readFile(sourceManifestPath),
    readFile(licenseEvidencePath),
    readFile(resolve(repositoryRoot, "webui/scripts/prepare_hosted_release_corpus.py")),
    readFile(resolve(repositoryRoot, "browser-runtime/release-corpus/jlens-word-seeds.txt")),
  ]);
  const release = parseJson(releaseBytes, "release corpus manifest");
  const source = parseJson(sourceBytes, "release corpus source manifest");
  validateReleaseCorpus({
    release,
    source,
    sourceBytes,
    generatorBytes,
    wordSeedBytes,
    licenseEvidenceBytes,
    runtimeLock,
    modelLock,
  });
  const output = release.outputs[kind];
  if (corpusBytes.byteLength !== output.bytes || sha256(corpusBytes) !== output.sha256) {
    throw new Error(`release ${kind} corpus bytes differ from the approved manifest`);
  }
  if (kind === "jlens") {
    if (
      wordsBytes === null ||
      wordsBytes.byteLength !== release.outputs.words.bytes ||
      sha256(wordsBytes) !== release.outputs.words.sha256
    ) throw new Error("release J-lens word bytes differ from the approved manifest");
  }
  return { releaseBytes, sourceBytes, licenseEvidenceBytes };
}

function validateReleaseCorpus({
  release,
  source,
  sourceBytes,
  generatorBytes,
  wordSeedBytes,
  licenseEvidenceBytes,
  runtimeLock,
  modelLock,
}) {
  exactKeys(release, [
    "$schema", "schemaVersion", "algorithm", "generatorSha256",
    "sourceManifestSha256", "source", "license", "wordSeeds", "selection", "outputs",
  ], "release corpus manifest");
  if (
    release.$schema !== RELEASE_CORPUS_SCHEMA ||
    release.schemaVersion !== 1 ||
    release.algorithm !== RELEASE_CORPUS_ALGORITHM
  ) throw new Error("release corpus manifest identity is invalid");
  digest(release.generatorSha256, "release corpus generator digest");
  digest(release.sourceManifestSha256, "release corpus source-manifest digest");
  if (release.generatorSha256 !== sha256(generatorBytes)) {
    throw new Error("release corpus was not produced by the checked-in generator");
  }
  if (release.sourceManifestSha256 !== sha256(sourceBytes)) {
    throw new Error("release corpus source-manifest digest differs");
  }

  exactKeys(source, [
    "$schema", "schemaVersion", "status", "blockers", "source", "license",
    "wordSeeds", "tokenizers", "selection",
  ], "release corpus source manifest");
  if (
    source.schemaVersion !== 1 || source.status !== "approved" ||
    !Array.isArray(source.blockers) || source.blockers.length !== 0
  ) throw new Error("release corpus source is not approved");
  exactKeys(source.source, [
    "repository", "revision", "configuration", "split", "path", "url", "format",
    "textField", "sha256", "bytes", "rows",
  ], "release corpus source");
  if (!COMMIT.test(source.source?.revision ?? "") || !SHA256.test(source.source?.sha256 ?? "")) {
    throw new Error("release corpus source is not immutably identified");
  }
  if (
    !/^https:\/\//.test(source.source.url) ||
    !source.source.url.includes(source.source.revision) ||
    !positiveInteger(source.source.bytes) ||
    !positiveInteger(source.source.rows)
  ) throw new Error("release corpus source URL or extent is invalid");
  if (source.source?.split?.toLowerCase() === "test") {
    throw new Error("release corpus source cannot use a held-out test split");
  }
  exactKeys(source.license, [
    "declaredIdentifiers", "evidence", "attribution", "redistributionNotice",
    "reviewStatus", "reviewedBy", "reviewedAt",
  ], "release corpus license");
  exactKeys(source.license.evidence, ["url", "sha256", "bytes"], "release license evidence");
  if (
    source.license?.reviewStatus !== "approved" ||
    !nonempty(source.license.reviewedBy) ||
    !Number.isFinite(Date.parse(source.license.reviewedAt))
  ) throw new Error("release corpus license review is incomplete");
  if (
    !/^https:\/\//.test(source.license.evidence.url) ||
    !source.license.evidence.url.includes(source.source.revision) ||
    source.license?.evidence?.bytes !== licenseEvidenceBytes.byteLength ||
    source.license?.evidence?.sha256 !== sha256(licenseEvidenceBytes)
  ) throw new Error("release corpus license evidence differs from the approved closure");
  if (
    canonical(source.source) !== canonical(release.source) ||
    canonical(source.license) !== canonical(release.license) ||
    canonical(source.wordSeeds) !== canonical(release.wordSeeds) ||
    canonical(source.selection) !== canonical(release.selection)
  ) throw new Error("release corpus provenance differs from its approved source manifest");

  exactKeys(source.wordSeeds, ["path", "sha256", "bytes", "license"], "word seeds");
  if (
    source.wordSeeds.path !== "browser-runtime/release-corpus/jlens-word-seeds.txt" ||
    source.wordSeeds.bytes !== wordSeedBytes.byteLength ||
    source.wordSeeds.sha256 !== sha256(wordSeedBytes)
  ) throw new Error("release corpus word seeds differ from the checked-in closure");
  validateTokenizerClosure(source.tokenizers, runtimeLock, modelLock);
  exactKeys(release.outputs, ["sae", "jlens", "words"], "release corpus outputs");
  for (const [name, output] of Object.entries(release.outputs)) {
    exactKeys(
      output,
      name === "words"
        ? ["path", "sha256", "bytes", "count", "candidateCount", "tokenIds"]
        : ["path", "sha256", "bytes", "documents", "characters"],
      `release corpus ${name} output`,
    );
    digest(output.sha256, `release corpus ${name} digest`);
  }
  exactKeys(release.outputs.words.tokenIds, source.tokenizers.map(tokenizer => tokenizer.id),
    "release word token closure");
  for (const tokenIds of Object.values(release.outputs.words.tokenIds)) {
    if (!Array.isArray(tokenIds) || tokenIds.length !== release.outputs.words.count ||
        tokenIds.some(id => !Number.isSafeInteger(id) || id < 0) ||
        new Set(tokenIds).size !== tokenIds.length) {
      throw new Error("release word token closure is invalid");
    }
  }
  if (
    release.selection?.algorithm !== RELEASE_CORPUS_ALGORITHM ||
    !positiveInteger(release.selection.saeDocuments) ||
    !positiveInteger(release.selection.jlensDocuments) ||
    release.selection.minimumSharedWords < MINIMUM_SHARED_WORDS ||
    release.outputs.words.count < MINIMUM_SHARED_WORDS
  ) throw new Error("release corpus selection does not meet instrument minimums");
  if (
    release.outputs.sae.path !== "sae-corpus.json" ||
    release.outputs.sae.documents !== release.selection.saeDocuments ||
    release.outputs.jlens.path !== "jlens-corpus.json" ||
    release.outputs.jlens.documents !== release.selection.jlensDocuments ||
    release.outputs.words.path !== "jlens-words.json"
  ) throw new Error("release corpus outputs do not close the approved selection");
}

function validateTokenizerClosure(value, runtimeLock, modelLock) {
  if (!Array.isArray(value) || value.length === 0 ||
      new Set(value.map(tokenizer => tokenizer?.id)).size !== value.length) {
    throw new Error("release corpus tokenizer closure must name unique models");
  }
  if (!modelLock || !value.some(tokenizer => tokenizer?.id === modelLock.id)) {
    throw new Error("release corpus tokenizer closure does not cover the target model");
  }
  for (const tokenizer of value) {
    const model = runtimeLock.models.find(model => model.id === tokenizer?.id);
    if (!model) throw new Error("release corpus tokenizer closure names an unlocked model");
    exactKeys(tokenizer, [
      "id", "sourceRepository", "sourceRevision", "tokenizerSha256", "tokenizerConfigSha256",
    ], `release tokenizer ${model.id}`);
    if (
      tokenizer.id !== model.id ||
      tokenizer.sourceRepository !== model.sourceRepository ||
      tokenizer.sourceRevision !== model.sourceRevision ||
      tokenizer.tokenizerSha256 !== model.tokenizerSha256 ||
      tokenizer.tokenizerConfigSha256 !== model.chatTemplateSha256
    ) throw new Error(`release corpus tokenizer closure differs for ${model.id}`);
  }
}

function validateSae(manifest, release, modelLock) {
  if (
    manifest?.corpus_sha256 !== release.outputs.sae.sha256 ||
    manifest?.d_model !== modelLock.hiddenSize
  ) throw new Error("release SAE does not bind the approved corpus and runtime shape");
  const minimumFeatures = Math.min(modelLock.hiddenSize * 4, 8192);
  if (
    !positiveInteger(manifest.tokens_trained) || manifest.tokens_trained < MINIMUM_SAE_TOKENS ||
    !positiveInteger(manifest.seq_len) || manifest.seq_len < MINIMUM_SAE_SEQUENCE ||
    !positiveInteger(manifest.d_sae) || manifest.d_sae < minimumFeatures
  ) {
    throw new Error(
      `release SAE requires at least ${MINIMUM_SAE_TOKENS} trained tokens, ` +
      `${MINIMUM_SAE_SEQUENCE}-token sequences, and ${minimumFeatures} features`,
    );
  }
  return {
    corpusSha256: release.outputs.sae.sha256,
    sourceRevision: release.source.revision,
    tokensTrained: manifest.tokens_trained,
    sequenceLength: manifest.seq_len,
    features: manifest.d_sae,
  };
}

function validateJlens(manifest, vocabulary, release, modelLock) {
  const expectedLayers = modelLock.layerMap.slice(0, -1);
  if (
    manifest?.raw_corpus_sha256 !== release.outputs.jlens.sha256 ||
    manifest?.d_model !== modelLock.hiddenSize ||
    manifest?.model_layer_count !== modelLock.layerMap.length ||
    canonical(manifest?.source_layers) !== canonical(expectedLayers)
  ) throw new Error("release J-lens does not bind the approved corpus and complete layer map");
  if (
    !positiveInteger(manifest.n_prompts) || manifest.n_prompts < MINIMUM_JLENS_PROMPTS ||
    !positiveInteger(manifest.raw_prompt_count) || manifest.raw_prompt_count < MINIMUM_JLENS_PROMPTS ||
    !positiveInteger(manifest.usable_prompt_count) || manifest.usable_prompt_count < MINIMUM_JLENS_PROMPTS ||
    !positiveInteger(manifest.seq_len) || manifest.seq_len < MINIMUM_JLENS_SEQUENCE
  ) {
    throw new Error(
      `release J-lens requires at least ${MINIMUM_JLENS_PROMPTS} usable prompts ` +
      `and ${MINIMUM_JLENS_SEQUENCE}-token sequences`,
    );
  }
  if (!vocabulary || !Array.isArray(vocabulary.words)) {
    throw new Error("release J-lens vocabulary manifest is unavailable");
  }
  const words = vocabulary.words.map((entry) => entry?.word);
  const wordBytes = Buffer.from(JSON.stringify(words));
  if (
    words.length < MINIMUM_SHARED_WORDS ||
    words.length !== release.outputs.words.count ||
    wordBytes.byteLength !== release.outputs.words.bytes ||
    sha256(wordBytes) !== release.outputs.words.sha256
  ) throw new Error("release J-lens vocabulary does not match the approved shared-word closure");
  const tokenIds = release.outputs.words.tokenIds[modelLock.id];
  if (vocabulary.words.some((entry, index) => entry.row !== index || entry.token_id !== tokenIds[index])) {
    throw new Error("release J-lens vocabulary does not match the approved token closure");
  }
  return {
    corpusSha256: release.outputs.jlens.sha256,
    sourceRevision: release.source.revision,
    prompts: manifest.usable_prompt_count,
    sequenceLength: manifest.seq_len,
    layers: expectedLayers.length,
    words: words.length,
  };
}

function parseJson(bytes, label) {
  try {
    return JSON.parse(bytes.toString("utf8"));
  } catch {
    throw new Error(`${label} is not valid UTF-8 JSON`);
  }
}

async function readRequired(path, label) {
  try {
    return await readFile(path);
  } catch (error) {
    if (error?.code === "ENOENT") throw new Error(`${label} is missing from the instrument pack`);
    throw error;
  }
}

function exactKeys(value, expected, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    throw new Error(`${label} has unknown or missing fields`);
  }
}

function digest(value, label) {
  if (typeof value !== "string" || !SHA256.test(value)) throw new Error(`${label} is invalid`);
}

function positiveInteger(value) {
  return Number.isSafeInteger(value) && value > 0;
}

function nonempty(value) {
  return typeof value === "string" && value.length > 0;
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function canonical(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  return `{${Object.entries(value)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, entry]) => `${JSON.stringify(key)}:${canonical(entry)}`)
    .join(",")}}`;
}

export const releaseInstrumentMinimums = Object.freeze({
  jlensPrompts: MINIMUM_JLENS_PROMPTS,
  jlensSequence: MINIMUM_JLENS_SEQUENCE,
  sharedWords: MINIMUM_SHARED_WORDS,
  saeTokens: MINIMUM_SAE_TOKENS,
  saeSequence: MINIMUM_SAE_SEQUENCE,
});
