import { decodeFp32Safetensors, type Fp32SafetensorsReader } from "../artifacts";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import type { ProbeInfo } from "../../lib/types";
import { runtimeIdentitySha256 } from "../../lib/runtime/catalog";
import { LEGACY_PRODUCT_SLUGS } from "./brandMigration";
import type {
  BrowserInstrumentPackArtifacts,
  BrowserModelArtifact,
  BrowserModelLoadRequest,
} from "./modelBackend";
import type { BrowserSaeFitIdentity } from "../artifacts/fittedAuthoring";
import type {
  BrowserCentroidTransform,
  BrowserFittingLayerCentroids,
} from "../fitting/coordinator";
import { jLensTokenDirection } from "./structuredHookProgram";
import { verifiedSaeDescriptionSource, type SaeDescriptionSource } from "../../lib/saeDescriptions";

const LOCAL_SAE_V1_KEYS = [
  "activation", "batch_size", "corpus_sha256", "corpus_spec", "d_model", "d_sae",
  "dead_feature_threshold", "format_version", "kind", "l1_coefficient", "layer",
  "learning_rate", "model_fingerprint", "model_id", "model_source_fingerprint", "name",
  "release", "seq_len", "tensor_file", "tensor_sha256", "tokens_trained",
].sort();
const LOCAL_JLENS_V6_KEYS = [
  "base_n_prompts", "checkpoint", "consumed_prefix_sha256", "corpus_hash_kind",
  "corpus_sha256", "corpus_spec", "d_model", "dim_batch", "dtype", "estimator_policy",
  "format_version", "method", "model_fingerprint", "model_layer_count",
  "model_source_fingerprint", "n_prompts", "partial_n_prompts", "raw_corpus_sha256",
  "raw_prompt_count", "seq_len", "skip_first_positions", "source_layers", "tensor_files",
  "tensor_sha256", "usable_prompt_count",
].sort();
const JLENS_DIRECTION_CACHE_MAX_ENTRIES = 64;
const JLENS_DIRECTION_CACHE_MAX_BYTES = 32 * 1024 * 1024;

export interface BrowserSaeFeature {
  readonly key: string;
  readonly layer: number;
  readonly label: string | null;
  readonly maxAct: number | null;
  readonly activation: "relu" | "jump_relu";
  readonly threshold: number;
  readonly direction: Float32Array;
  readonly encoderDirection: Float32Array;
  readonly encoderInputBias: Float32Array;
  readonly featureBias: number;
}

export interface BrowserSaeGpuDictionary {
  readonly bindingId: string;
  readonly hiddenSize: number;
  readonly runtimeLayerIndex: number;
  readonly featureCount: number;
  readonly activation: "relu" | "jump_relu";
  readonly encoder: Float32Array;
  readonly encoderBias: Float32Array;
  readonly encoderThreshold: Float32Array;
  readonly decoderBias: Float32Array;
}

export interface BrowserJlensGpuDictionary {
  readonly bindingId: string;
  readonly hiddenSize: number;
  readonly layerIndices: Int32Array;
  readonly matrices: readonly Float32Array[];
}

export interface BrowserJlensToken {
  readonly key: string;
  readonly word: string;
  readonly tokenId: number;
  readonly directions: ReadonlyMap<number, Float32Array>;
}

interface BrowserSae {
  readonly descriptionSource: SaeDescriptionSource | null;
  readonly source: string;
  readonly displayName: string;
  readonly release: string;
  readonly revision: string;
  readonly fingerprint: string;
  readonly layer: number;
  readonly hiddenSize: number;
  readonly featureCount: number;
  readonly activation: "relu" | "jump_relu";
  readonly featureMetadata: ReadonlyMap<number, {
    readonly label: string | null;
    readonly maxAct: number | null;
  }>;
  readonly encoder: Float32Array;
  readonly decoder: Float32Array;
  readonly encoderBias: Float32Array;
  readonly encoderThreshold: Float32Array;
  readonly encoderInputBias: Float32Array;
  readonly decoderBias: Float32Array;
}

export interface BrowserExactSaeFitting {
  readonly layers: readonly number[];
  readonly transformCentroids: BrowserCentroidTransform;
  readonly provenance: BrowserSaeFitIdentity;
}

export interface BrowserJlensRuntimeResolver {
  tokenizeText(text: string): Promise<number[]>;
  decodeTokens(tokenIds: readonly number[]): Promise<string>;
  resolveJlensTokenDirections(
    bindingId: string,
    layerIndices: readonly number[],
    tokenIds: readonly number[],
  ): Promise<Float32Array>;
  setJlensDictionary?(dictionary: BrowserJlensGpuDictionary | null): Promise<void>;
}

export function saeDecoderDirection(
  decoder: ArrayLike<number>,
  hiddenSize: number,
  feature: number,
): Float32Array {
  if (
    !Number.isSafeInteger(hiddenSize) || hiddenSize < 1 ||
    !Number.isSafeInteger(feature) || feature < 0 ||
    decoder.length % hiddenSize !== 0 || feature >= decoder.length / hiddenSize
  ) {
    throw new TypeError("SAE decoder feature selection is invalid");
  }
  const offset = feature * hiddenSize;
  const direction = Float32Array.from(
    { length: hiddenSize },
    (_, index) => decoder[offset + index],
  );
  if (direction.some((value) => !Number.isFinite(value))) {
    throw new TypeError("SAE decoder direction contains a non-finite value");
  }
  return direction;
}

interface BrowserJlens {
  readonly source: string;
  readonly displayName: string;
  readonly hiddenSize: number;
  readonly layerMap: readonly number[];
  readonly runtimeLayerMap: readonly number[];
  readonly bindingId: string;
  readonly jacobians: ReadonlyMap<number, Float32Array>;
  readonly words: Map<string, { row: number; tokenId: number }>;
  readonly unembedding: Float32Array;
  readonly cache: Map<string, BrowserJlensToken>;
  readonly resolvedByTokenId: Map<number, ReadonlyMap<number, Float32Array>>;
  readonly pinnedTokenIds: Set<number>;
  resolvedBytes: number;
}

export class BrowserInstrumentRegistry {
  private constructor(
    private readonly sae: BrowserSae | null,
    private readonly jlensPacks: ReadonlyMap<string, BrowserInstrumentPackArtifacts>,
    private readonly loadRequest: BrowserModelLoadRequest,
    private jlens: BrowserJlens | null,
  ) {}

  static async load(request: BrowserModelLoadRequest): Promise<BrowserInstrumentRegistry> {
    const optionalPacks = request.optionalPacks ?? [];
    for (const { pack } of optionalPacks) {
      if (pack.runtimeIdentitySha256 !== request.variant.runtimeIdentitySha256) {
        throw instrumentError(
          "INSTRUMENT_PACK_RUNTIME_MISMATCH",
          `${pack.displayName} was built for a different browser model runtime`,
        );
      }
    }
    const saePacks = optionalPacks.filter(({ pack }) => pack.kind === "sae");
    const jlensPacks = optionalPacks.filter(({ pack }) => pack.kind === "jlens");
    if (saePacks.length > 1) {
      throw instrumentError(
        "INSTRUMENT_PACK_AMBIGUOUS",
        "Only one active SAE pack may be loaded at a time",
      );
    }
    const jlensBySource = new Map(jlensPacks.map((candidate) => [candidate.pack.id, candidate]));
    const initialJlens = jlensPacks.length === 0 ? null : await loadJlens(jlensPacks[0], request);
    return new BrowserInstrumentRegistry(
      saePacks.length === 1 ? await loadSae(saePacks[0], request) : null,
      jlensBySource,
      request,
      initialJlens,
    );
  }

  jlensSources(): Array<{
    source: string;
    displayName: string;
    active: boolean;
  }> {
    return [...this.jlensPacks].map(([source, candidate]) => ({
      source,
      displayName: candidate.pack.displayName,
      active: source === this.jlens?.source,
    }));
  }

  async activateJlensSource(source: string): Promise<void> {
    if (this.jlens?.source === source) return;
    const candidate = this.jlensPacks.get(source);
    if (candidate === undefined) {
      throw instrumentError(
        "JLENS_SOURCE_UNAVAILABLE",
        `J-lens source ${source} is not installed for this model`,
      );
    }
    this.jlens = await loadJlens(candidate, this.loadRequest);
  }

  capabilities(): { sae: boolean; jlens: boolean } {
    return { sae: this.sae !== null, jlens: this.jlens !== null };
  }

  descriptor(): {
    sae: { source: string; displayName: string; layer: number; features: number; modelLayers: number[]; descriptionSource: SaeDescriptionSource | null } | null;
    jlens: {
      source: string;
      displayName: string;
      layers: number[];
      words: Array<{ word: string; tokenId: number }>;
    } | null;
  } {
    return {
      sae: this.sae === null
        ? null
        : {
            source: this.sae.source,
            displayName: this.sae.displayName,
            layer: this.sae.layer,
            features: this.sae.featureCount,
            modelLayers: [...this.loadRequest.variant.runtimeIdentity.layerMap],
            descriptionSource: this.sae.descriptionSource,
          },
      jlens: this.jlens === null
        ? null
        : {
            source: this.jlens.source,
            displayName: this.jlens.displayName,
            layers: [...this.jlens.layerMap],
            words: [...this.jlens.words].map(([word, value]) => ({
              word,
              tokenId: value.tokenId,
            })),
          },
    };
  }

  saeGpuDictionary(runtimeLayerMap: readonly number[]): BrowserSaeGpuDictionary | null {
    const sae = this.sae;
    if (sae === null) return null;
    const runtimeLayerIndex = runtimeLayerMap.indexOf(sae.layer);
    if (runtimeLayerIndex < 0) {
      throw instrumentError(
        "SAE_PACK_LAYER_MISMATCH",
        `SAE layer ${sae.layer} is outside the loaded runtime layer map`,
      );
    }
    return {
      bindingId: sae.fingerprint,
      hiddenSize: sae.hiddenSize,
      runtimeLayerIndex,
      featureCount: sae.featureCount,
      activation: sae.activation,
      encoder: sae.encoder,
      encoderBias: sae.encoderBias,
      encoderThreshold: sae.encoderThreshold,
      // The GPU ABI calls the encoder input offset decoderBias.
      decoderBias: sae.encoderInputBias,
    };
  }

  jlensGpuDictionary(
    runtimeLayerMap: readonly number[],
  ): BrowserJlensGpuDictionary | null {
    const jlens = this.jlens;
    if (jlens === null) return null;
    const fitted: Array<{ layerIndex: number; matrix: Float32Array }> = [];
    runtimeLayerMap.forEach((layer, layerIndex) => {
      const matrix = jlens.jacobians.get(layer);
      if (matrix !== undefined) fitted.push({ layerIndex, matrix });
    });
    if (fitted.length !== jlens.layerMap.length) {
      throw instrumentError(
        "JLENS_PACK_LAYER_MISMATCH",
        "The installed J-lens layer map does not match the loaded runtime",
      );
    }
    return {
      bindingId: jlens.bindingId,
      hiddenSize: jlens.hiddenSize,
      layerIndices: Int32Array.from(fitted, ({ layerIndex }) => layerIndex),
      matrices: fitted.map(({ matrix }) => matrix),
    };
  }

  setPinnedJlensWords(words: readonly string[]): void {
    const jlens = this.jlens;
    if (jlens === null) return;
    const pinned = new Set<number>();
    for (const value of words) {
      const word = normalizeJlensWord(value);
      const token = jlens.cache.get(word);
      if (token === undefined) {
        throw instrumentError(
          "JLENS_TOKEN_UNAVAILABLE",
          `J-lens token ${JSON.stringify(word)} must be prepared before it is pinned`,
        );
      }
      pinned.add(token.tokenId);
      touchJlensWord(jlens, word, token);
    }
    jlens.pinnedTokenIds.clear();
    for (const tokenId of pinned) jlens.pinnedTokenIds.add(tokenId);
    trimJlensDirectionCache(jlens);
  }

  saeFeatureMetadata(): Record<string, { label: string | null; maxAct: number | null }> {
    const sae = this.sae;
    if (sae === null) return {};
    return Object.fromEntries([...sae.featureMetadata].map(([featureId, metadata]) => [
      String(featureId),
      { ...metadata },
    ]));
  }

  probeHash(
    modelId: string,
    probe: Exclude<ProbeInfo, { family: "geometry" }>,
  ): string {
    if (probe.family === "lens") {
      if (probe.token_id === null) {
        throw instrumentError(
          "JLENS_TOKEN_UNAVAILABLE",
          `J-lens probe ${probe.name} has no resolved token identity`,
        );
      }
      return pythonTupleDigest([
        "jlens-readout-v2",
        modelId,
        probe.word,
        probe.token_id,
        [...probe.layers],
      ]);
    }
    const sae = this.sae;
    if (sae === null || probe.layers.length !== 1 || probe.layers[0] !== sae.layer) {
      throw instrumentError(
        "SAE_PACK_UNAVAILABLE",
        `SAE probe ${probe.name} does not match the active browser SAE`,
      );
    }
    return pythonTupleDigest([
      "sae-readout-v2",
      modelId,
      sae.fingerprint,
      sae.release,
      sae.layer,
      probe.feature_id,
      sae.featureMetadata.get(probe.feature_id)?.maxAct === null ||
          sae.featureMetadata.get(probe.feature_id)?.maxAct === undefined
        ? null
        : { pythonFloat: sae.featureMetadata.get(probe.feature_id)!.maxAct! },
    ]);
  }

  saeFeature(value: string): BrowserSaeFeature {
    const sae = this.sae;
    if (sae === null) {
      throw instrumentError("SAE_PACK_UNAVAILABLE", "Install the compatible SAE pack to use sae/<feature>");
    }
    const feature = Number(value);
    if (!Number.isSafeInteger(feature) || feature < 0 || feature >= sae.featureCount) {
      throw instrumentError(
        "SAE_FEATURE_UNAVAILABLE",
        `SAE feature ${value} is outside the installed pack's 0-${sae.featureCount - 1} range`,
      );
    }
    const encoderDirection = new Float32Array(sae.hiddenSize);
    for (let index = 0; index < sae.hiddenSize; index += 1) {
      encoderDirection[index] = sae.encoder[index * sae.featureCount + feature];
    }
    return {
      key: `sae/${feature}`,
      layer: sae.layer,
      label: sae.featureMetadata.get(feature)?.label ?? null,
      maxAct: sae.featureMetadata.get(feature)?.maxAct ?? null,
      activation: sae.activation,
      threshold: sae.encoderThreshold[feature],
      direction: saeDecoderDirection(sae.decoder, sae.hiddenSize, feature),
      encoderDirection,
      encoderInputBias: sae.encoderInputBias,
      featureBias: sae.encoderBias[feature],
    };
  }

  exactSaeFitting(selector: string): BrowserExactSaeFitting | null {
    const sae = this.sae;
    if (sae === null || selector !== sae.source && selector !== sae.release) return null;
    const idsByLayer = new Map([[sae.layer, `${sae.release}:layer-${sae.layer}`]]);
    return {
      layers: [sae.layer],
      transformCentroids: (layer, centroids) =>
        reconstructSaeCentroids(sae, layer, centroids),
      provenance: {
        release: sae.release,
        revision: sae.revision,
        fingerprint: sae.fingerprint,
        idsByLayer,
        fullCoverage: false,
      },
    };
  }

  async prepareJlensWords(
    words: readonly string[],
    resolver: BrowserJlensRuntimeResolver,
  ): Promise<void> {
    const requested = [...new Set(words.map(normalizeJlensWord))];
    if (requested.length === 0) return;
    if (requested.length > JLENS_DIRECTION_CACHE_MAX_ENTRIES) {
      throw instrumentError(
        "JLENS_PREPARATION_LIMIT",
        `Prepare at most ${JLENS_DIRECTION_CACHE_MAX_ENTRIES} J-lens words at once`,
      );
    }
    const jlens = this.jlens;
    if (jlens === null) {
      throw instrumentError(
        "JLENS_PACK_UNAVAILABLE",
        "Install the compatible J-lens pack to use jlens/<word>",
      );
    }
    const tokenIds = new Map<string, number>();
    for (const word of requested) {
      const tokenId = await resolveJlensWordToken(resolver, word);
      const curated = jlens.words.get(word);
      if (curated !== undefined && curated.tokenId !== tokenId) {
        throw instrumentError(
          "JLENS_VOCABULARY_INVALID",
          `The installed J-lens vocabulary token for ${JSON.stringify(word)} does not match the loaded tokenizer`,
        );
      }
      tokenIds.set(word, tokenId);
    }
    const uniqueIds = [...new Set(tokenIds.values())].filter(
      (tokenId) => !jlens.resolvedByTokenId.has(tokenId),
    );
    if (uniqueIds.length > 0) {
      const layerIndices = jlens.layerMap.map((layer) =>
        jlens.runtimeLayerMap.indexOf(layer)
      );
      for (let start = 0; start < uniqueIds.length; start += 8) {
        const batch = uniqueIds.slice(start, start + 8);
        const values = await resolver.resolveJlensTokenDirections(
          jlens.bindingId,
          layerIndices,
          batch,
        );
        const expected = jlens.layerMap.length * batch.length * jlens.hiddenSize;
        if (
          !(values instanceof Float32Array) || values.length !== expected ||
          values.some((value) => !Number.isFinite(value))
        ) {
          throw instrumentError(
            "JLENS_DIRECTION_INVALID",
            "The loaded model runtime returned invalid J-lens token directions",
          );
        }
        batch.forEach((tokenId, tokenIndex) => {
          const directions = new Map<number, Float32Array>();
          jlens.layerMap.forEach((layer, layerIndex) => {
            const offset = (layerIndex * batch.length + tokenIndex) * jlens.hiddenSize;
            directions.set(layer, values.slice(offset, offset + jlens.hiddenSize));
          });
          storeJlensDirections(jlens, tokenId, directions);
        });
      }
    }
    for (const [word, tokenId] of tokenIds) {
      const directions = touchJlensDirections(jlens, tokenId)!;
      touchJlensWord(jlens, word, {
        key: `jlens/${word}`,
        word,
        tokenId,
        directions,
      });
    }
    trimJlensDirectionCache(jlens);
  }

  jlensToken(word: string): BrowserJlensToken {
    const jlens = this.jlens;
    if (jlens === null) {
      throw instrumentError("JLENS_PACK_UNAVAILABLE", "Install the compatible J-lens pack to use jlens/<word>");
    }
    const normalized = normalizeJlensWord(word);
    const cached = jlens.cache.get(normalized);
    if (cached !== undefined) {
      touchJlensWord(jlens, normalized, cached);
      touchJlensDirections(jlens, cached.tokenId);
      return cached;
    }
    const entry = jlens.words.get(normalized);
    if (entry === undefined) {
      throw instrumentError(
        "JLENS_TOKEN_UNAVAILABLE",
        `J-lens token ${JSON.stringify(normalized)} has not been prepared with the loaded tokenizer`,
      );
    }
    const resolved = touchJlensDirections(jlens, entry.tokenId);
    let directions: ReadonlyMap<number, Float32Array>;
    if (resolved === undefined) {
      const unembedding = jlens.unembedding.subarray(
        entry.row * jlens.hiddenSize,
        (entry.row + 1) * jlens.hiddenSize,
      );
      const fallback = new Map<number, Float32Array>();
      for (const layer of jlens.layerMap) {
        fallback.set(layer, jLensTokenDirection(unembedding, jlens.jacobians.get(layer)!));
      }
      directions = fallback;
      storeJlensDirections(jlens, entry.tokenId, directions);
    } else {
      directions = resolved;
    }
    const token = {
      key: `jlens/${normalized}`,
      word: normalized,
      tokenId: entry.tokenId,
      directions,
    };
    touchJlensWord(jlens, normalized, token);
    trimJlensDirectionCache(jlens);
    return token;
  }

  jlensProgram(
    probeNames: readonly string[],
    runtimeLayerMap: readonly number[],
    probeCount: number,
    includeFullVocabularyReadout = false,
  ): { bindingId: string; layerIndices: Int32Array; tokenIds: Int32Array } | null {
    const names = probeNames.filter((name) => name.startsWith("jlens/"));
    if (names.length === 0 && !includeFullVocabularyReadout) return null;
    const jlens = this.jlens;
    if (jlens === null) {
      throw instrumentError("JLENS_PACK_UNAVAILABLE", "Install the compatible J-lens pack to use probability readouts");
    }
    const fitted: number[] = [];
    runtimeLayerMap.forEach((layer, programIndex) => {
      if (!jlens.jacobians.has(layer)) return;
      fitted.push(programIndex);
    });
    if (fitted.length === 0) {
      throw instrumentError(
        "JLENS_PACK_LAYER_MISMATCH",
        "The installed J-lens has no fitted layer in the loaded runtime",
      );
    }
    const layerIndices = Int32Array.from(fitted);
    const tokenIds = new Int32Array(probeCount);
    probeNames.forEach((name, probeIndex) => {
      if (!name.startsWith("jlens/")) return;
      tokenIds[probeIndex] = this.jlensToken(name.slice("jlens/".length)).tokenId;
    });
    return { bindingId: jlens.bindingId, layerIndices, tokenIds };
  }
}

async function loadSae(
  pack: BrowserInstrumentPackArtifacts,
  request: BrowserModelLoadRequest,
): Promise<BrowserSae> {
  assertArtifactClosure(pack);
  const manifestArtifact = uniqueArtifact(pack.artifacts, (artifact) =>
    artifact.manifest.path.endsWith("/manifest.json"), "SAE manifest");
  const manifest = await jsonArtifact(manifestArtifact, request.signal);
  const hiddenSize = request.variant.runtimeIdentity.hiddenSize;
  if (
    !sameStrings(Object.keys(manifest).filter((key) => key !== "apply_b_dec_to_input").sort(), LOCAL_SAE_V1_KEYS) ||
    ("apply_b_dec_to_input" in manifest && typeof manifest.apply_b_dec_to_input !== "boolean") ||
    !(
      manifest.format_version === 1 && manifest.activation === "relu" ||
      manifest.format_version === 2 && manifest.activation === "jump_relu"
    ) || manifest.kind !== "local" ||
    !nonnegativeInteger(manifest.layer) || !positiveInteger(manifest.d_model) ||
    !positiveInteger(manifest.d_sae) || manifest.d_model !== hiddenSize ||
    manifest.model_id !== request.variant.runtimeIdentity.sourceModel ||
    !matchesRuntimeFingerprint(manifest.model_fingerprint, request) ||
    manifest.model_source_fingerprint !== request.variant.runtimeIdentity.sourceRevision ||
    typeof manifest.release !== "string" ||
    !/^local:[A-Za-z][A-Za-z0-9_-]*$/u.test(manifest.release) ||
    typeof manifest.tensor_file !== "string" || !safeBasename(manifest.tensor_file) ||
    typeof manifest.tensor_sha256 !== "string" || !/^[a-f0-9]{64}$/u.test(manifest.tensor_sha256)
  ) {
    throw instrumentError("SAE_PACK_INVALID", "The SAE pack does not match the local v1 fp32 schema");
  }
  if (!request.variant.runtimeIdentity.layerMap.includes(manifest.layer)) {
    throw instrumentError("SAE_PACK_LAYER_MISMATCH", `SAE layer ${manifest.layer} is outside this model's layer map`);
  }
  const tensorArtifact = artifactAt(pack.artifacts, siblingPath(
    manifestArtifact.manifest.path,
    manifest.tensor_file,
  ));
  if (tensorArtifact.manifest.sha256 !== manifest.tensor_sha256) {
    throw instrumentError("SAE_PACK_INVALID", "The SAE tensor digest does not match its manifest");
  }
  const tensors = await tensorArtifactReader(tensorArtifact, request.signal);
  const expectedKeys = manifest.activation === "jump_relu"
    ? ["W_dec", "W_enc", "b_dec", "b_enc", "threshold"]
    : ["W_dec", "W_enc", "b_dec", "b_enc"];
  if (!sameStrings(tensors.keys, expectedKeys)) {
    throw instrumentError(
      "SAE_PACK_INVALID",
      manifest.activation === "jump_relu"
        ? "The JumpReLU SAE tensor must contain W_enc, W_dec, b_enc, b_dec, and threshold"
        : "The SAE tensor must contain exactly W_enc, W_dec, b_enc, and b_dec",
    );
  }
  const featureMetadataArtifacts = pack.artifacts.filter((artifact) =>
    artifact.manifest.path.endsWith("/features.json")
  );
  if (featureMetadataArtifacts.length > 1) {
    throw instrumentError(
      "SAE_PACK_INVALID",
      "The SAE pack may contain at most one feature metadata file",
    );
  }
  const featureMetadata = featureMetadataArtifacts.length === 0
    ? new Map<number, { label: string | null; maxAct: number | null }>()
    : await loadSaeFeatureMetadata(
        featureMetadataArtifacts[0],
        request.signal,
        manifest.model_id,
        manifest.release,
        manifest.d_sae,
      );
  const gemmaScope2 = manifest.activation === "jump_relu" &&
    typeof manifest.corpus_spec === "string" &&
    /^provider:google\/gemma-scope-2-(270m|1b|4b|12b|27b)-(it|pt)$/u.test(manifest.corpus_spec);
  const applyDecoderBias = manifest.apply_b_dec_to_input ?? (gemmaScope2 ? false : undefined);
  if (applyDecoderBias === undefined && String(manifest.corpus_spec).startsWith("provider:")) {
    throw instrumentError("SAE_PACK_INVALID", "Provider SAE packs must declare apply_b_dec_to_input");
  }
  const decoderBias = tensors.tensor("b_dec", [hiddenSize]).data;
  const centered = applyDecoderBias ?? true;
  return {
    source: pack.pack.id,
    descriptionSource: verifiedSaeDescriptionSource(manifest),
    displayName: pack.pack.displayName,
    release: manifest.release,
    revision: manifest.tensor_sha256,
    fingerprint: centered ? manifest.tensor_sha256 : bytesToHex(sha256(new TextEncoder().encode(
      `sae-uncentered-v1:${manifest.tensor_sha256}`,
    ))),
    layer: manifest.layer,
    hiddenSize,
    featureCount: manifest.d_sae,
    activation: manifest.activation,
    featureMetadata,
    encoder: tensors.tensor("W_enc", [hiddenSize, manifest.d_sae]).data,
    decoder: tensors.tensor("W_dec", [manifest.d_sae, hiddenSize]).data,
    encoderBias: tensors.tensor("b_enc", [manifest.d_sae]).data,
    encoderThreshold: manifest.activation === "jump_relu"
      ? tensors.tensor("threshold", [manifest.d_sae]).data
      : new Float32Array(manifest.d_sae),
    encoderInputBias: centered ? decoderBias : new Float32Array(hiddenSize),
    decoderBias,
  };
}

async function loadSaeFeatureMetadata(
  artifact: BrowserModelArtifact,
  signal: AbortSignal,
  modelId: string,
  release: string,
  featureCount: number,
): Promise<Map<number, { label: string | null; maxAct: number | null }>> {
  const payload = await jsonArtifact(artifact, signal);
  if (
    !sameStrings(Object.keys(payload).sort(), [
      "features", "format_version", "model_id", "release",
    ]) ||
    payload.format_version !== 1 || payload.model_id !== modelId ||
    payload.release !== release || !isRecord(payload.features)
  ) {
    throw instrumentError(
      "SAE_FEATURE_METADATA_INVALID",
      "The SAE feature metadata does not match the browser v1 schema",
    );
  }
  const metadata = new Map<number, { label: string | null; maxAct: number | null }>();
  for (const [key, value] of Object.entries(payload.features)) {
    const featureId = Number(key);
    if (
      !/^(0|[1-9]\d*)$/u.test(key) || !Number.isSafeInteger(featureId) ||
      featureId < 0 || featureId >= featureCount || !isRecord(value) ||
      !sameStrings(Object.keys(value).sort(), ["label", "max_act"])
    ) {
      throw instrumentError(
        "SAE_FEATURE_METADATA_INVALID",
        "The SAE feature metadata contains an invalid feature row",
      );
    }
    const label = value.label;
    const maxAct = value.max_act;
    if (
      label !== null && (typeof label !== "string" || label.trim() === "") ||
      maxAct !== null && (typeof maxAct !== "number" || !Number.isFinite(maxAct) || maxAct <= 0)
    ) {
      throw instrumentError(
        "SAE_FEATURE_METADATA_INVALID",
        `SAE feature ${featureId} has invalid label or max_act metadata`,
      );
    }
    metadata.set(featureId, { label, maxAct });
  }
  return metadata;
}

function reconstructSaeCentroids(
  sae: BrowserSae,
  layer: number,
  centroids: BrowserFittingLayerCentroids,
): BrowserFittingLayerCentroids {
  if (
    layer !== sae.layer || centroids.columns !== sae.hiddenSize ||
    !Number.isSafeInteger(centroids.rows) || centroids.rows < 1 ||
    centroids.values.length !== centroids.rows * centroids.columns ||
    centroids.values.some((value) => !Number.isFinite(value))
  ) {
    throw instrumentError(
      "SAE_FITTING_INPUT_INVALID",
      `The installed SAE cannot transform layer ${layer} centroids`,
    );
  }
  const reconstructed = new Float64Array(centroids.values.length);
  for (let row = 0; row < centroids.rows; row += 1) {
    const offset = row * sae.hiddenSize;
    reconstructed.set(sae.decoderBias, offset);
    for (let feature = 0; feature < sae.featureCount; feature += 1) {
      let activation = sae.encoderBias[feature];
      for (let column = 0; column < sae.hiddenSize; column += 1) {
        activation += (
          centroids.values[offset + column] - sae.encoderInputBias[column]
        ) * sae.encoder[column * sae.featureCount + feature];
      }
      if (activation <= sae.encoderThreshold[feature]) continue;
      const decoderOffset = feature * sae.hiddenSize;
      for (let column = 0; column < sae.hiddenSize; column += 1) {
        reconstructed[offset + column] += activation * sae.decoder[decoderOffset + column];
      }
    }
  }
  if (reconstructed.some((value) => !Number.isFinite(value))) {
    throw instrumentError(
      "SAE_FITTING_RESULT_INVALID",
      "The installed SAE produced non-finite reconstructed centroids",
    );
  }
  return {
    rows: centroids.rows,
    columns: centroids.columns,
    values: reconstructed,
  };
}

async function loadJlens(
  pack: BrowserInstrumentPackArtifacts,
  request: BrowserModelLoadRequest,
): Promise<BrowserJlens> {
  assertArtifactClosure(pack);
  const manifestArtifact = uniqueArtifact(pack.artifacts, (artifact) =>
    artifact.manifest.path.endsWith("/manifest.json"), "J-lens manifest");
  const vocabularyArtifact = uniqueArtifact(pack.artifacts, (artifact) =>
    artifact.manifest.path.endsWith("/browser-vocabulary.json"), "J-lens browser vocabulary");
  const [manifest, vocabulary] = await Promise.all([
    jsonArtifact(manifestArtifact, request.signal),
    jsonArtifact(vocabularyArtifact, request.signal),
  ]);
  const hiddenSize = request.variant.runtimeIdentity.hiddenSize;
  if (
    !sameStrings(Object.keys(manifest).sort(), LOCAL_JLENS_V6_KEYS) ||
    manifest.format_version !== 6 || manifest.checkpoint !== false || manifest.dtype !== "float32" ||
    manifest.d_model !== hiddenSize || !integerList(manifest.source_layers) ||
    !matchesRuntimeFingerprint(manifest.model_fingerprint, request) ||
    manifest.model_source_fingerprint !== request.variant.runtimeIdentity.sourceRevision ||
    manifest.model_layer_count !== request.variant.runtimeIdentity.layerMap.length ||
    !isRecord(manifest.tensor_files) || !isRecord(manifest.tensor_sha256)
  ) {
    throw instrumentError("JLENS_PACK_INVALID", "The J-lens pack does not match the local v6 fp32 schema");
  }
  const layerMap = manifest.source_layers as number[];
  if (layerMap.some((layer) => !request.variant.runtimeIdentity.layerMap.includes(layer))) {
    throw instrumentError("JLENS_PACK_LAYER_MISMATCH", "The J-lens contains layers outside this model's layer map");
  }
  const jacobians = new Map<number, Float32Array>();
  const grouped = new Map<BrowserModelArtifact, number[]>();
  for (const layer of layerMap) {
    const file = manifest.tensor_files[String(layer)];
    const digest = manifest.tensor_sha256[String(layer)];
    if (typeof file !== "string" || !safeBasename(file) || typeof digest !== "string") {
      throw instrumentError("JLENS_PACK_INVALID", `J-lens layer ${layer} has invalid shard metadata`);
    }
    const artifact = artifactAt(pack.artifacts, siblingPath(manifestArtifact.manifest.path, file));
    if (artifact.manifest.sha256 !== digest) {
      throw instrumentError("JLENS_PACK_INVALID", `J-lens layer ${layer} has a mismatched tensor digest`);
    }
    grouped.set(artifact, [...(grouped.get(artifact) ?? []), layer]);
  }
  for (const [artifact, layers] of grouped) {
    const tensors = await tensorArtifactReader(artifact, request.signal);
    const expected = layers.map((layer) => `layer_${layer}`).sort();
    if (!sameStrings(tensors.keys, expected)) {
      throw instrumentError("JLENS_PACK_INVALID", `J-lens shard ${artifact.manifest.path} has unexpected tensor keys`);
    }
    for (const layer of layers) {
      jacobians.set(layer, tensors.tensor(`layer_${layer}`, [hiddenSize, hiddenSize]).data);
    }
  }
  if (
    ![
      "drowse-jlens-vocabulary-v2",
      ...LEGACY_PRODUCT_SLUGS.map((slug) => `${slug}-jlens-vocabulary-v2`),
    ].includes(vocabulary.format as string) ||
    vocabulary.hidden_size !== hiddenSize ||
    typeof vocabulary.tensor_file !== "string" || !safeBasename(vocabulary.tensor_file) ||
    typeof vocabulary.tensor_sha256 !== "string" || !Array.isArray(vocabulary.words) ||
    vocabulary.words.length === 0
  ) {
    throw instrumentError("JLENS_VOCABULARY_INVALID", "The J-lens browser vocabulary manifest is invalid");
  }
  const words = new Map<string, { row: number; tokenId: number }>();
  vocabulary.words.forEach((entry, index) => {
    if (
      !isRecord(entry) || typeof entry.word !== "string" || !entry.word ||
      entry.row !== index || !nonnegativeInteger(entry.token_id) || words.has(entry.word)
    ) {
      throw instrumentError("JLENS_VOCABULARY_INVALID", "J-lens vocabulary words and rows must be unique and contiguous");
    }
    words.set(entry.word, { row: index, tokenId: entry.token_id });
  });
  const tensorArtifact = artifactAt(pack.artifacts, siblingPath(
    vocabularyArtifact.manifest.path,
    vocabulary.tensor_file,
  ));
  if (tensorArtifact.manifest.sha256 !== vocabulary.tensor_sha256) {
    throw instrumentError("JLENS_VOCABULARY_INVALID", "The J-lens vocabulary tensor digest does not match");
  }
  const vocabularyTensors = await tensorArtifactReader(tensorArtifact, request.signal);
  if (!sameStrings(vocabularyTensors.keys, ["unembedding"])) {
    throw instrumentError("JLENS_VOCABULARY_INVALID", "The J-lens vocabulary must contain one unembedding tensor");
  }
  return {
    source: pack.pack.id,
    displayName: pack.pack.displayName,
    hiddenSize,
    layerMap,
    runtimeLayerMap: [...request.variant.runtimeIdentity.layerMap],
    bindingId: manifestArtifact.manifest.sha256,
    jacobians,
    words,
    unembedding: vocabularyTensors.tensor("unembedding", [words.size, hiddenSize]).data,
    cache: new Map(),
    resolvedByTokenId: new Map(),
    pinnedTokenIds: new Set(),
    resolvedBytes: 0,
  };
}

function matchesRuntimeFingerprint(
  fingerprint: unknown,
  request: BrowserModelLoadRequest,
): boolean {
  if (fingerprint === request.variant.runtimeIdentitySha256) return true;
  if (request.variant.runtimeIdentity.runtimeAbi !== "drowse-web-runtime-v1") {
    return false;
  }
  return LEGACY_PRODUCT_SLUGS.some((slug) => fingerprint === runtimeIdentitySha256({
    ...request.variant.runtimeIdentity,
    runtimeAbi: `${slug}-web-runtime-v1`,
  }));
}

function jlensDirectionsByteLength(
  directions: ReadonlyMap<number, Float32Array>,
): number {
  let bytes = 0;
  for (const direction of directions.values()) bytes += direction.byteLength;
  return bytes;
}

function touchJlensDirections(
  jlens: BrowserJlens,
  tokenId: number,
): ReadonlyMap<number, Float32Array> | undefined {
  const directions = jlens.resolvedByTokenId.get(tokenId);
  if (directions === undefined) return undefined;
  jlens.resolvedByTokenId.delete(tokenId);
  jlens.resolvedByTokenId.set(tokenId, directions);
  return directions;
}

function touchJlensWord(
  jlens: BrowserJlens,
  word: string,
  token: BrowserJlensToken,
): void {
  jlens.cache.delete(word);
  jlens.cache.set(word, token);
}

function storeJlensDirections(
  jlens: BrowserJlens,
  tokenId: number,
  directions: ReadonlyMap<number, Float32Array>,
): void {
  const previous = jlens.resolvedByTokenId.get(tokenId);
  if (previous !== undefined) {
    jlens.resolvedBytes -= jlensDirectionsByteLength(previous);
    jlens.resolvedByTokenId.delete(tokenId);
  }
  jlens.resolvedByTokenId.set(tokenId, directions);
  jlens.resolvedBytes += jlensDirectionsByteLength(directions);
  trimJlensDirectionCache(jlens);
}

function trimJlensDirectionCache(jlens: BrowserJlens): void {
  const maximumEntries = JLENS_DIRECTION_CACHE_MAX_ENTRIES + jlens.pinnedTokenIds.size;
  while (
    jlens.resolvedByTokenId.size > maximumEntries ||
    jlens.resolvedBytes > JLENS_DIRECTION_CACHE_MAX_BYTES
  ) {
    const evicted = [...jlens.resolvedByTokenId].find(
      ([tokenId]) => !jlens.pinnedTokenIds.has(tokenId),
    );
    if (evicted === undefined) break;
    const [tokenId, directions] = evicted;
    jlens.resolvedByTokenId.delete(tokenId);
    jlens.resolvedBytes -= jlensDirectionsByteLength(directions);
    for (const [word, token] of jlens.cache) {
      if (token.tokenId === tokenId) jlens.cache.delete(word);
    }
  }
  while (jlens.cache.size > maximumEntries) {
    const evicted = [...jlens.cache].find(
      ([, token]) => !jlens.pinnedTokenIds.has(token.tokenId),
    );
    if (evicted === undefined) break;
    jlens.cache.delete(evicted[0]);
  }
}

async function resolveJlensWordToken(
  resolver: BrowserJlensRuntimeResolver,
  word: string,
): Promise<number> {
  let pieces: number[] = [];
  for (const candidate of [` ${word}`, word]) {
    const ids = await resolver.tokenizeText(candidate);
    if (
      !Array.isArray(ids) ||
      ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id > 0x7fff_ffff)
    ) {
      throw instrumentError(
        "JLENS_TOKENIZATION_INVALID",
        "The loaded tokenizer returned invalid J-lens token IDs",
      );
    }
    if (ids.length === 1 && (await resolver.decodeTokens(ids)).trim() === word) {
      return ids[0];
    }
    if (pieces.length === 0 && ids.length > 1) pieces = [...ids];
  }
  throw instrumentError(
    "MULTI_TOKEN_WORD",
    `${JSON.stringify(word)} is not a single token in this vocabulary` +
      (pieces.length > 0 ? ` (token IDs: ${pieces.join(", ")})` : "") +
      ". The Jacobian lens needs a word that encodes as one token.",
  );
}

function normalizeJlensWord(word: string): string {
  const normalized = word.trim().normalize("NFC");
  if (normalized.length === 0) {
    throw instrumentError("JLENS_TOKEN_UNAVAILABLE", "J-lens words must be non-empty");
  }
  return normalized;
}

type PythonIdentityValue =
  | string
  | number
  | null
  | readonly number[]
  | { readonly pythonFloat: number };

function pythonTupleDigest(values: readonly PythonIdentityValue[]): string {
  const source = `(${values.map(pythonIdentityRepr).join(", ")}${values.length === 1 ? "," : ""})`;
  return bytesToHex(sha256(new TextEncoder().encode(source)));
}

function pythonIdentityRepr(value: PythonIdentityValue): string {
  if (value === null) return "None";
  if (Array.isArray(value)) {
    const entries = value as readonly number[];
    return `(${entries.map((entry) => pythonNumberRepr(entry)).join(", ")}${entries.length === 1 ? "," : ""})`;
  }
  if (typeof value === "number") return pythonNumberRepr(value);
  if (typeof value === "object") {
    return pythonNumberRepr((value as { readonly pythonFloat: number }).pythonFloat, true);
  }
  return pythonStringRepr(value);
}

function pythonStringRepr(value: string): string {
  const quote = value.includes("'") && !value.includes('"') ? '"' : "'";
  let output = quote;
  for (const character of value) {
    const codePoint = character.codePointAt(0)!;
    if (character === "\\" || character === quote) {
      output += `\\${character}`;
    } else if (character === "\t") {
      output += "\\t";
    } else if (character === "\n") {
      output += "\\n";
    } else if (character === "\r") {
      output += "\\r";
    } else if (pythonPrintable(character)) {
      output += character;
    } else if (codePoint <= 0xff) {
      output += `\\x${codePoint.toString(16).padStart(2, "0")}`;
    } else if (codePoint <= 0xffff) {
      output += `\\u${codePoint.toString(16).padStart(4, "0")}`;
    } else {
      output += `\\U${codePoint.toString(16).padStart(8, "0")}`;
    }
  }
  return `${output}${quote}`;
}

function pythonPrintable(value: string): boolean {
  return value === " " || !/[\p{Cc}\p{Cf}\p{Cs}\p{Co}\p{Cn}\p{Zl}\p{Zp}\p{Zs}]/u.test(value);
}

function pythonNumberRepr(value: number, forceFloat = false): string {
  if (!Number.isFinite(value)) {
    throw instrumentError("INSTRUMENT_PROBE_IDENTITY_INVALID", "Probe identity numbers must be finite");
  }
  if (Number.isSafeInteger(value) && !Object.is(value, -0) && !forceFloat) return String(value);
  if (Object.is(value, -0)) return "-0.0";
  const source = String(value);
  const exponentIndex = source.search(/[eE]/u);
  if (exponentIndex >= 0) {
    const coefficient = source.slice(0, exponentIndex);
    const exponent = Number(source.slice(exponentIndex + 1));
    return `${coefficient}e${exponent >= 0 ? "+" : "-"}${Math.abs(exponent).toString().padStart(2, "0")}`;
  }
  const negative = source.startsWith("-");
  const unsigned = negative ? source.slice(1) : source;
  const dot = unsigned.indexOf(".");
  const integer = dot < 0 ? unsigned : unsigned.slice(0, dot);
  const fraction = dot < 0 ? "" : unsigned.slice(dot + 1);
  const exponent = integer !== "0"
    ? integer.length - 1
    : -(fraction.search(/[1-9]/u) + 1);
  if (exponent >= -4 && exponent < 16) {
    return source.includes(".") ? source : `${source}.0`;
  }
  const digits = integer === "0"
    ? fraction.slice(fraction.search(/[1-9]/u)).replace(/0+$/u, "")
    : `${integer}${fraction}`.replace(/0+$/u, "");
  const coefficient = digits.length === 1 ? digits : `${digits[0]}.${digits.slice(1)}`;
  return `${negative ? "-" : ""}${coefficient}e${exponent >= 0 ? "+" : "-"}${Math.abs(exponent).toString().padStart(2, "0")}`;
}

function assertArtifactClosure(pack: BrowserInstrumentPackArtifacts): void {
  const expected = [...pack.pack.files].map((file) => `${file.path}:${file.sha256}`).sort();
  const actual = pack.artifacts.map(({ manifest }) => `${manifest.path}:${manifest.sha256}`).sort();
  if (!sameStrings(actual, expected)) {
    throw instrumentError("INSTRUMENT_PACK_ARTIFACT_MISMATCH", `${pack.pack.displayName} does not match its catalog closure`);
  }
}

function uniqueArtifact(
  artifacts: readonly BrowserModelArtifact[],
  predicate: (artifact: BrowserModelArtifact) => boolean,
  label: string,
): BrowserModelArtifact {
  const matches = artifacts.filter(predicate);
  if (matches.length !== 1) throw instrumentError("INSTRUMENT_PACK_INVALID", `${label} must occur exactly once`);
  return matches[0];
}

function artifactAt(artifacts: readonly BrowserModelArtifact[], path: string): BrowserModelArtifact {
  const matches = artifacts.filter((artifact) => artifact.manifest.path === path);
  if (matches.length !== 1) throw instrumentError("INSTRUMENT_PACK_INVALID", `Instrument artifact ${path} is missing or duplicated`);
  return matches[0];
}

async function jsonArtifact(
  artifact: BrowserModelArtifact,
  signal: AbortSignal,
): Promise<Record<string, unknown>> {
  signal.throwIfAborted();
  let value: unknown;
  try {
    value = JSON.parse(await artifact.file.text());
  } catch {
    throw instrumentError("INSTRUMENT_PACK_INVALID", `${artifact.manifest.path} is not valid JSON`);
  }
  signal.throwIfAborted();
  if (!isRecord(value)) throw instrumentError("INSTRUMENT_PACK_INVALID", `${artifact.manifest.path} must contain an object`);
  return value;
}

async function tensorArtifactReader(
  artifact: BrowserModelArtifact,
  signal: AbortSignal,
): Promise<Fp32SafetensorsReader> {
  signal.throwIfAborted();
  const bytes = new Uint8Array(await artifact.file.arrayBuffer());
  signal.throwIfAborted();
  return decodeFp32Safetensors(bytes, artifact.manifest.path);
}

function siblingPath(path: string, basename: string): string {
  const index = path.lastIndexOf("/");
  return index < 0 ? basename : `${path.slice(0, index + 1)}${basename}`;
}

function safeBasename(value: string): boolean {
  return value.length > 0 && value !== "." && value !== ".." &&
    !value.includes("/") && !value.includes("\\") && !value.includes("\0");
}

function integerList(value: unknown): value is number[] {
  return Array.isArray(value) && value.length > 0 && value.every(nonnegativeInteger) &&
    value.every((item, index) => index === 0 || value[index - 1] < item);
}

function nonnegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0;
}

function positiveInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) > 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function instrumentError(code: string, message: string): Error & {
  code: string;
  status: number;
  recoverable: boolean;
} {
  return Object.assign(new Error(message), { code, status: 409, recoverable: true });
}
