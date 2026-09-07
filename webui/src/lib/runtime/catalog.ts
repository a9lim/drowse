import type {
  CatalogDocument,
  CatalogFile,
  CatalogInstrumentPack,
  CatalogModel,
  DetachedCatalogSignature,
  ModelContextProfile,
  ModelThinkingProfile,
  RuntimeIdentity,
  ModelVariant,
  VerifiedCatalog,
} from "./contracts";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import {
  EXACT_READOUT_REQUIRED_WEBGPU_LIMITS,
  resolveStructuredHookProfile,
  structuredHookMemoryRequirements,
  WEB_LLM_RUNTIME_MIN_STORAGE_BUFFERS_PER_SHADER_STAGE,
} from "../../hosted/runtime/structuredHookProfile";

const SHA256 = /^[a-f0-9]{64}$/;
const MAX_CATALOG_ID_LENGTH = 200;
const COMMIT = /^[a-f0-9]{40}$/;
const CLOCK_SKEW_MS = 5 * 60 * 1000;
const MAX_CATALOG_BYTES = 4 * 1024 * 1024;
export const CATALOG_WEBGPU_LIMIT_NAMES = [
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxComputeWorkgroupStorageSize",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupSizeZ",
  "maxComputeWorkgroupsPerDimension",
  "maxComputeInvocationsPerWorkgroup",
  "maxStorageBuffersPerShaderStage",
  "maxBindingsPerBindGroup",
] as const;
const CATALOG_WEBGPU_LIMIT_SET = new Set<string>(CATALOG_WEBGPU_LIMIT_NAMES);

export interface CatalogSignatureVerifier {
  hasKey(keyId: string): boolean;
  verify(
    exactBytes: Uint8Array,
    signature: DetachedCatalogSignature,
  ): Promise<boolean>;
}

export interface CatalogAdmissionPolicy {
  now: number;
  online: boolean;
  minimumSequence: number;
  lastAcceptedSequence: number | null;
  lastAcceptedIdentity?: {
    exactBytes: Uint8Array;
    signature: DetachedCatalogSignature;
  } | null;
  expectedRuntimeAbi: string;
  runtimeLockModels: readonly CatalogRuntimeLockModel[];
}

export interface CatalogRuntimeLockModel {
  modelType?: "chat" | "base";
  runtimeIdentity: RuntimeIdentity;
  structuredHookProfile: ModelVariant["structuredHookProfile"];
  thinkingProfile: ModelThinkingProfile | null;
  convertedRepository: string;
  convertedRevision: string;
  contextProfiles: readonly number[];
}

export class CatalogValidationError extends Error {
  readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "CatalogValidationError";
    this.code = code;
  }
}

export class WebCryptoEd25519Verifier implements CatalogSignatureVerifier {
  private readonly keys: ReadonlyMap<string, Uint8Array>;

  constructor(keys: ReadonlyMap<string, Uint8Array> | Record<string, Uint8Array>) {
    this.keys = keys instanceof Map ? keys : new Map(Object.entries(keys));
  }

  hasKey(keyId: string): boolean {
    return this.keys.has(keyId);
  }

  async verify(
    exactBytes: Uint8Array,
    signature: DetachedCatalogSignature,
  ): Promise<boolean> {
    const rawKey = this.keys.get(signature.keyId);
    if (
      !rawKey || rawKey.byteLength !== 32 ||
      signature.schemaVersion !== 1 ||
      signature.algorithm !== "Ed25519"
    ) return false;
    if (!globalThis.crypto?.subtle) {
      throw new CatalogValidationError(
        "ED25519_UNAVAILABLE",
        "This browser cannot verify the signed Drowse catalog",
      );
    }
    let key: CryptoKey;
    try {
      key = await globalThis.crypto.subtle.importKey(
        "raw",
        rawKey.slice().buffer,
        { name: "Ed25519" },
        false,
        ["verify"],
      );
    } catch (error) {
      throw new CatalogValidationError(
        "ED25519_UNAVAILABLE",
        error instanceof Error ? error.message : "Ed25519 key import failed",
      );
    }
    const signatureBytes = decodeBase64(signature.signature);
    return globalThis.crypto.subtle.verify(
      { name: "Ed25519" },
      key,
      signatureBytes.slice().buffer,
      exactBytes.slice().buffer,
    );
  }
}

export async function verifyCatalog(
  exactBytes: Uint8Array,
  signature: DetachedCatalogSignature,
  verifier: CatalogSignatureVerifier,
  policy: CatalogAdmissionPolicy,
): Promise<VerifiedCatalog> {
  if (exactBytes.byteLength === 0 || exactBytes.byteLength > MAX_CATALOG_BYTES) {
    throw new CatalogValidationError(
      "CATALOG_SIZE_INVALID",
      `The catalog must be between 1 byte and ${MAX_CATALOG_BYTES} bytes`,
    );
  }
  validateDetachedSignature(signature);
  if (!verifier.hasKey(signature.keyId)) {
    throw new CatalogValidationError(
      "CATALOG_KEY_UNKNOWN",
      `Unknown catalog signing key ${signature.keyId}`,
    );
  }
  if (!(await verifier.verify(exactBytes, signature))) {
    throw new CatalogValidationError(
      "CATALOG_SIGNATURE_INVALID",
      "The catalog signature is invalid",
    );
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(exactBytes));
  } catch {
    throw new CatalogValidationError(
      "CATALOG_JSON_INVALID",
      "The signed catalog is not valid JSON",
    );
  }
  const document = validateCatalogDocument(parsed);
  if (document.runtimeAbi !== policy.expectedRuntimeAbi) {
    throw new CatalogValidationError(
      "CATALOG_RUNTIME_MISMATCH",
      `Catalog runtime ${document.runtimeAbi} is incompatible with ${policy.expectedRuntimeAbi}`,
    );
  }
  validateRuntimeLockBindings(document, policy.runtimeLockModels);
  if (
    document.sequence < policy.minimumSequence ||
    (policy.lastAcceptedSequence !== null &&
      document.sequence < policy.lastAcceptedSequence)
  ) {
    throw new CatalogValidationError(
      "CATALOG_ROLLBACK",
      "The catalog sequence is older than the last accepted release",
    );
  }
  enforceSequenceContinuity(exactBytes, signature, document.sequence, policy);

  const issuedAt = Date.parse(document.issuedAt);
  const expiresAt = Date.parse(document.expiresAt);
  if (!Number.isFinite(issuedAt) || !Number.isFinite(expiresAt) || expiresAt <= issuedAt) {
    throw new CatalogValidationError(
      "CATALOG_TIME_INVALID",
      "The catalog issue or expiry time is invalid",
    );
  }
  if (issuedAt > policy.now + CLOCK_SKEW_MS) {
    throw new CatalogValidationError(
      "CATALOG_NOT_YET_VALID",
      "The catalog issue time is in the future",
    );
  }
  const stale = policy.now >= expiresAt;
  if (stale && policy.online) {
    throw new CatalogValidationError(
      "CATALOG_EXPIRED",
      "The online catalog has expired",
    );
  }
  return {
    document,
    exactBytes: exactBytes.slice(),
    signature,
    allowDownloads: !stale,
    stale,
  };
}

function enforceSequenceContinuity(
  exactBytes: Uint8Array,
  signature: DetachedCatalogSignature,
  sequence: number,
  policy: CatalogAdmissionPolicy,
): void {
  if (policy.lastAcceptedSequence === null) return;
  const accepted = policy.lastAcceptedIdentity;
  if (!accepted) {
    throw new CatalogValidationError(
      "CATALOG_ADMISSION_STATE_INVALID",
      "The accepted catalog sequence has no signed identity record",
    );
  }
  if (sequence !== policy.lastAcceptedSequence) return;
  if (
    !sameExactBytes(exactBytes, accepted.exactBytes) ||
    !sameDetachedSignature(signature, accepted.signature)
  ) {
    throw new CatalogValidationError(
      "CATALOG_EQUIVOCATION",
      "A different signed catalog reused the last accepted sequence",
    );
  }
}

function sameExactBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (!(right instanceof Uint8Array) || left.byteLength !== right.byteLength) return false;
  for (let index = 0; index < left.byteLength; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

function sameDetachedSignature(
  left: DetachedCatalogSignature,
  right: DetachedCatalogSignature,
): boolean {
  return right !== null && typeof right === "object" &&
    left.schemaVersion === right.schemaVersion &&
    left.algorithm === right.algorithm &&
    left.keyId === right.keyId &&
    left.signature === right.signature;
}

function validateRuntimeLockBindings(
  document: CatalogDocument,
  locks: readonly CatalogRuntimeLockModel[],
): void {
  for (const model of document.models) {
    for (const variant of model.variants) {
      const lock = locks.find((candidate) =>
        sameRuntimeIdentity(candidate.runtimeIdentity, variant.runtimeIdentity) &&
        (candidate.modelType ?? "chat") === (model.modelType ?? "chat") &&
        candidate.structuredHookProfile === variant.structuredHookProfile &&
        sameThinkingProfile(candidate.thinkingProfile, variant.thinkingProfile) &&
        sameNumberSet(
          candidate.contextProfiles,
          variant.contextProfiles.map((profile) => profile.contextTokens),
        )
      );
      if (!lock) {
        throw new CatalogValidationError(
          "CATALOG_RUNTIME_IDENTITY_UNLOCKED",
          `Catalog variant ${variant.id} is not present in the embedded browser runtime lock`,
        );
      }
      if (model.sourceUrl !== `https://huggingface.co/${lock.runtimeIdentity.sourceModel}`) {
        throw new CatalogValidationError(
          "CATALOG_SOURCE_PROVENANCE_MISMATCH",
          `Catalog model ${model.id} does not match its locked source repository`,
        );
      }
      const lockedFiles = [
        ...variant.files,
        ...variant.packs.filter((pack) => pack.required).flatMap((pack) => pack.files),
      ];
      const prefix =
        `https://huggingface.co/${lock.convertedRepository}/resolve/${lock.convertedRevision}/`;
      if (lockedFiles.some((file) =>
        file.revision !== lock.convertedRevision || !file.url.startsWith(prefix)
      )) {
        throw new CatalogValidationError(
          "CATALOG_CONVERTED_PROVENANCE_MISMATCH",
          `Catalog variant ${variant.id} does not match its locked converted repository revision`,
        );
      }
    }
  }
}

function sameRuntimeIdentity(left: RuntimeIdentity, right: RuntimeIdentity): boolean {
  return left.sourceModel === right.sourceModel &&
    left.sourceRevision === right.sourceRevision &&
    left.convertedManifestSha256 === right.convertedManifestSha256 &&
    left.quantization === right.quantization &&
    left.tokenizerSha256 === right.tokenizerSha256 &&
    left.chatTemplateSha256 === right.chatTemplateSha256 &&
    left.modelLibrarySha256 === right.modelLibrarySha256 &&
    left.runtimeAbi === right.runtimeAbi &&
    left.hookAbi === right.hookAbi &&
    left.hiddenSize === right.hiddenSize &&
    sameNumberSet(left.layerMap, right.layerMap);
}

function sameNumberSet(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length &&
    new Set(left).size === left.length &&
    new Set(right).size === right.length &&
    left.every((value) => right.includes(value));
}

export function validateCatalogDocument(value: unknown): CatalogDocument {
  const root = record(value, "catalog");
  exactKeys(
    root,
    ["schemaVersion", "sequence", "issuedAt", "expiresAt", "runtimeAbi", "models"],
    "catalog",
  );
  integer(root.schemaVersion, "catalog.schemaVersion");
  if (root.schemaVersion !== 1) {
    throw invalid("catalog.schemaVersion must be 1");
  }
  positiveInteger(root.sequence, "catalog.sequence");
  text(root.issuedAt, "catalog.issuedAt");
  text(root.expiresAt, "catalog.expiresAt");
  text(root.runtimeAbi, "catalog.runtimeAbi");
  const models = array(root.models, "catalog.models").map(validateModel);
  if (!models.length) throw invalid("catalog.models must not be empty");
  const ids = new Map<string, string>();
  const claimId = (id: string, kind: string): void => {
    const previous = ids.get(id);
    if (previous) throw invalid(`${kind} id ${id} collides with ${previous}`);
    ids.set(id, kind);
  };
  for (const model of models) {
    claimId(model.id, "model");
    for (const variant of model.variants) {
      claimId(variant.id, "variant");
      if (variant.runtimeIdentity.runtimeAbi !== root.runtimeAbi) {
        throw invalid(`variant ${variant.id} runtime ABI does not match the catalog`);
      }
      for (const pack of variant.packs) {
        claimId(pack.id, "pack");
      }
    }
  }
  return {
    schemaVersion: 1,
    sequence: root.sequence as number,
    issuedAt: root.issuedAt as string,
    expiresAt: root.expiresAt as string,
    runtimeAbi: root.runtimeAbi as string,
    models,
  };
}

function validateModel(value: unknown, index: number): CatalogModel {
  const path = `catalog.models[${index}]`;
  const item = record(value, path);
  exactKeys(
    item,
    ["id", "displayName", "description", "sourceUrl", "license", "languages", "variants",
      ...("modelType" in item ? ["modelType"] : [])],
    path,
  );
  identifier(item.id, `${path}.id`);
  if ("modelType" in item && item.modelType !== "chat" && item.modelType !== "base") {
    throw invalid(`${path}.modelType must be chat or base`);
  }
  text(item.displayName, `${path}.displayName`);
  text(item.description, `${path}.description`, true);
  httpsUrl(item.sourceUrl, `${path}.sourceUrl`);
  const license = identifier(item.license, `${path}.license`);
  if (license !== license.trim()) {
    throw invalid(`${path}.license must not have leading or trailing whitespace`);
  }
  const languages = array(item.languages, `${path}.languages`).map((entry, i) =>
    text(entry, `${path}.languages[${i}]`),
  );
  const variants = array(item.variants, `${path}.variants`).map(validateVariant);
  if (!variants.length) throw invalid(`${path}.variants must not be empty`);
  return {
    id: item.id as string,
    ...("modelType" in item ? { modelType: item.modelType as "chat" | "base" } : {}),
    displayName: item.displayName as string,
    description: item.description as string,
    sourceUrl: item.sourceUrl as string,
    license: item.license as string,
    languages,
    variants,
  };
}

function validateVariant(value: unknown, index: number): ModelVariant {
  const path = `variant[${index}]`;
  const item = record(value, path);
  exactKeys(
    item,
    [
      "id",
      "tier",
      "structuredHookProfile",
      "thinkingProfile",
      "contextProfiles",
      "downloadBytes",
      "requiredCorePackBytes",
      "requirements",
      "runtimeIdentity",
      "runtimeIdentitySha256",
      "files",
      "packs",
    ],
    path,
  );
  identifier(item.id, `${path}.id`);
  if (item.tier !== "fastest" && item.tier !== "balanced" && item.tier !== "quality") {
    throw invalid(`${path}.tier is invalid`);
  }
  let structuredHookProfile;
  try {
    structuredHookProfile = resolveStructuredHookProfile(item.structuredHookProfile);
  } catch {
    throw invalid(`${path}.structuredHookProfile is not allow-listed`);
  }
  const thinkingProfile = validateThinkingProfile(item.thinkingProfile, `${path}.thinkingProfile`);
  positiveInteger(item.downloadBytes, `${path}.downloadBytes`);
  positiveInteger(item.requiredCorePackBytes, `${path}.requiredCorePackBytes`);
  sha(item.runtimeIdentitySha256, `${path}.runtimeIdentitySha256`);
  const identity = record(item.runtimeIdentity, `${path}.runtimeIdentity`);
  exactKeys(
    identity,
    [
      "sourceModel",
      "sourceRevision",
      "convertedManifestSha256",
      "quantization",
      "tokenizerSha256",
      "chatTemplateSha256",
      "modelLibrarySha256",
      "runtimeAbi",
      "hookAbi",
      "hiddenSize",
      "layerMap",
    ],
    `${path}.runtimeIdentity`,
  );
  for (const key of [
    "sourceModel",
    "sourceRevision",
    "convertedManifestSha256",
    "quantization",
    "tokenizerSha256",
    "chatTemplateSha256",
    "modelLibrarySha256",
    "runtimeAbi",
    "hookAbi",
  ]) {
    text(identity[key], `${path}.runtimeIdentity.${key}`);
  }
  positiveInteger(identity.hiddenSize, `${path}.runtimeIdentity.hiddenSize`);
  if (!COMMIT.test(identity.sourceRevision as string)) {
    throw invalid(`${path}.runtimeIdentity.sourceRevision must be a commit hash`);
  }
  for (const key of [
    "convertedManifestSha256",
    "tokenizerSha256",
    "chatTemplateSha256",
    "modelLibrarySha256",
  ]) {
    sha(identity[key], `${path}.runtimeIdentity.${key}`);
  }
  const layerMap = array(identity.layerMap, `${path}.runtimeIdentity.layerMap`).map(
    (entry, i) => integer(entry, `${path}.runtimeIdentity.layerMap[${i}]`),
  );
  if (!layerMap.length || layerMap.some((layer) => layer < 0)) {
    throw invalid(`${path}.runtimeIdentity.layerMap must contain non-negative layers`);
  }
  if (new Set(layerMap).size !== layerMap.length) {
    throw invalid(`${path}.runtimeIdentity.layerMap contains duplicate layers`);
  }
  if (layerMap.some((layer, index) => index > 0 && layer <= layerMap[index - 1])) {
    throw invalid(`${path}.runtimeIdentity.layerMap must be strictly increasing`);
  }
  const runtimeIdentity: RuntimeIdentity = {
    sourceModel: identity.sourceModel as string,
    sourceRevision: identity.sourceRevision as string,
    convertedManifestSha256: identity.convertedManifestSha256 as string,
    quantization: identity.quantization as string,
    tokenizerSha256: identity.tokenizerSha256 as string,
    chatTemplateSha256: identity.chatTemplateSha256 as string,
    modelLibrarySha256: identity.modelLibrarySha256 as string,
    runtimeAbi: identity.runtimeAbi as string,
    hookAbi: identity.hookAbi as string,
    hiddenSize: identity.hiddenSize as number,
    layerMap,
  };
  if (digestCanonical(runtimeIdentity) !== item.runtimeIdentitySha256) {
    throw invalid(`${path}.runtimeIdentitySha256 does not match runtimeIdentity`);
  }
  const requirements = record(item.requirements, `${path}.requirements`);
  exactKeys(requirements, ["features", "limits"], `${path}.requirements`);
  const features = array(requirements.features, `${path}.requirements.features`).map(
    (entry, i) => text(entry, `${path}.requirements.features[${i}]`),
  );
  const limitsValue = record(requirements.limits, `${path}.requirements.limits`);
  const limits: Record<string, number> = {};
  for (const [name, minimum] of Object.entries(limitsValue)) {
    if (!CATALOG_WEBGPU_LIMIT_SET.has(name)) {
      throw invalid(`${path}.requirements.limits.${name} is not a supported WebGPU limit`);
    }
    limits[name] = positiveInteger(minimum, `${path}.requirements.limits.${name}`, true);
  }
  let structuredRequirements: ReturnType<typeof structuredHookMemoryRequirements>;
  try {
    structuredRequirements = structuredHookMemoryRequirements(
      structuredHookProfile,
      layerMap.length,
      runtimeIdentity.hiddenSize,
    );
  } catch {
    throw invalid(`${path} structured hook requirements exceed safe arithmetic`);
  }
  for (const name of [
    "maxBufferSize",
    "maxStorageBufferBindingSize",
    "maxStorageBuffersPerShaderStage",
    "maxComputeWorkgroupStorageSize",
  ] as const) {
    if ((limits[name] ?? 0) < structuredRequirements[name]) {
      throw invalid(
        `${path}.requirements.limits.${name} must be at least ${structuredRequirements[name]} ` +
          `for structured hook profile ${structuredHookProfile.id}`,
      );
    }
  }
  if (
    (limits.maxStorageBuffersPerShaderStage ?? 0) <
      WEB_LLM_RUNTIME_MIN_STORAGE_BUFFERS_PER_SHADER_STAGE
  ) {
    throw invalid(
      `${path}.requirements.limits.maxStorageBuffersPerShaderStage must be at least ` +
        `${WEB_LLM_RUNTIME_MIN_STORAGE_BUFFERS_PER_SHADER_STAGE} for the WebLLM runtime`,
    );
  }
  if (structuredHookProfile.id === "standard-v3") {
    for (const [name, minimum] of Object.entries(
      EXACT_READOUT_REQUIRED_WEBGPU_LIMITS,
    )) {
      if (!Object.hasOwn(limits, name) || limits[name] < minimum) {
        throw invalid(
          `${path}.requirements.limits.${name} must explicitly declare at least ${minimum} ` +
            "for standard-v3 exact GPU readout",
        );
      }
    }
  }
  const files = array(item.files, `${path}.files`).map(validateFile);
  const packs = array(item.packs, `${path}.packs`).map(validatePack);
  const contextProfiles = array(
    item.contextProfiles,
    `${path}.contextProfiles`,
  ).map(validateContextProfile);
  if (!contextProfiles.length) throw invalid(`${path}.contextProfiles must not be empty`);
  for (const profile of contextProfiles) {
    if (
      profile.bindingSha256 !==
        contextBindingSha256(item.runtimeIdentitySha256 as string, profile.contextTokens)
    ) {
      throw invalid(`${path} context ${profile.contextTokens} binding digest is invalid`);
    }
  }
  unique(
    contextProfiles.map((profile) => profile.contextTokens),
    `${path}.contextProfiles contains duplicate context sizes`,
  );
  unique(files.map((file) => file.path), `${path}.files contains duplicate paths`);
  unique(packs.map((pack) => pack.id), `${path}.packs contains duplicate ids`);
  if (!files.length) throw invalid(`${path}.files must not be empty`);
  if (files.some((file) => file.role === "core_pack" || file.role === "instrument")) {
    throw invalid(`${path}.files may contain only base runtime artifacts`);
  }
  if (sumBytes(files, `${path}.files`) !== item.downloadBytes) {
    throw invalid(`${path}.downloadBytes does not equal its exact file total`);
  }
  const roleCount = (role: CatalogFile["role"]): number =>
    files.filter((file) => file.role === role).length;
  if (
    roleCount("weight") < 1 || roleCount("tokenizer") !== 1 ||
    roleCount("configuration") < 1 || roleCount("converted_manifest") !== 1 ||
    roleCount("chat_template") !== 1 || roleCount("model_library") !== 1
  ) {
    throw invalid(
      `${path}.files must include weights, configuration, and exactly one tokenizer, converted manifest, chat template, and model library`,
    );
  }
  const identityFiles: [CatalogFile["role"], keyof RuntimeIdentity][] = [
    ["tokenizer", "tokenizerSha256"],
    ["converted_manifest", "convertedManifestSha256"],
    ["chat_template", "chatTemplateSha256"],
    ["model_library", "modelLibrarySha256"],
  ];
  for (const [role, identityField] of identityFiles) {
    const artifact = files.find((file) => file.role === role)!;
    if (artifact.sha256 !== runtimeIdentity[identityField]) {
      throw invalid(`${path} ${role} digest does not match runtimeIdentity`);
    }
  }
  const requiredCorePacks = packs.filter(
    (pack) => pack.kind === "core" && pack.required,
  );
  if (requiredCorePacks.length !== 1) {
    throw invalid(`${path} must contain exactly one required core pack`);
  }
  if (packs.some((pack) => (pack.kind === "core") !== pack.required)) {
    throw invalid(`${path} core packs must be required and optional instruments must not be`);
  }
  if (requiredCorePacks[0].bytes !== item.requiredCorePackBytes) {
    throw invalid(`${path}.requiredCorePackBytes does not match its core pack`);
  }
  const contextBindings = new Set(
    contextProfiles.map((profile) => profile.bindingSha256),
  );
  for (const pack of packs) {
    if (pack.runtimeIdentitySha256 !== item.runtimeIdentitySha256) {
      throw invalid(`${path} pack ${pack.id} has a different runtime identity`);
    }
    if (
      pack.required &&
      [...contextBindings].some(
        (binding) => !pack.compatibleContextBindingSha256.includes(binding),
      )
    ) {
      throw invalid(`${path} required core pack does not cover every context profile`);
    }
    if (
      !pack.compatibleContextBindingSha256.length ||
      pack.compatibleContextBindingSha256.some((binding) => !contextBindings.has(binding))
    ) {
      throw invalid(`${path} pack ${pack.id} has an unknown or empty context binding`);
    }
  }
  for (const profile of contextProfiles) {
    const compatible = packs.filter((pack) =>
      pack.kind === "sae" &&
      pack.compatibleContextBindingSha256.includes(profile.bindingSha256)
    );
    if (compatible.length > 1) {
      throw invalid(
        `${path} context ${profile.contextTokens} has multiple compatible sae packs: ${
          compatible.map((pack) => pack.id).join(", ")
        }`,
      );
    }
  }
  return {
    id: item.id as string,
    tier: item.tier,
    structuredHookProfile: structuredHookProfile.id,
    thinkingProfile,
    contextProfiles,
    downloadBytes: item.downloadBytes as number,
    requiredCorePackBytes: item.requiredCorePackBytes as number,
    requirements: { features, limits },
    runtimeIdentity,
    runtimeIdentitySha256: item.runtimeIdentitySha256 as string,
    files,
    packs,
  };
}

function validateThinkingProfile(value: unknown, path: string): ModelThinkingProfile | null {
  if (value === null) return null;
  const item = record(value, path);
  exactKeys(
    item,
    ["start", "end", "startsInThinking", "startTokenIds", "endTokenIds"],
    path,
  );
  const start = boundedDelimiter(item.start, `${path}.start`);
  const end = boundedDelimiter(item.end, `${path}.end`);
  if (start === end) throw invalid(`${path} delimiters must differ`);
  if (typeof item.startsInThinking !== "boolean") {
    throw invalid(`${path}.startsInThinking must be a boolean`);
  }
  const startTokenIds = thinkingTokenIds(item.startTokenIds, `${path}.startTokenIds`);
  const endTokenIds = thinkingTokenIds(item.endTokenIds, `${path}.endTokenIds`);
  if (
    startTokenIds.length === endTokenIds.length &&
    startTokenIds.every((tokenId, index) => tokenId === endTokenIds[index])
  ) {
    throw invalid(`${path} token sequences must differ`);
  }
  return {
    start,
    end,
    startsInThinking: item.startsInThinking,
    startTokenIds,
    endTokenIds,
  };
}

function boundedDelimiter(value: unknown, path: string): string {
  if (
    typeof value !== "string" || value.length === 0 || value.length > 256 ||
    value.includes("\0")
  ) {
    throw invalid(`${path} must be 1-256 characters without NULs`);
  }
  return value;
}

function thinkingTokenIds(value: unknown, path: string): number[] {
  const values = array(value, path);
  if (
    values.length === 0 || values.length > 32 ||
    values.some((tokenId) => !Number.isSafeInteger(tokenId) || (tokenId as number) < 0)
  ) {
    throw invalid(`${path} must contain 1-32 non-negative token IDs`);
  }
  return values as number[];
}

function sameThinkingProfile(
  left: ModelThinkingProfile | null,
  right: ModelThinkingProfile | null,
): boolean {
  if (left === null || right === null) return left === right;
  return left.start === right.start && left.end === right.end &&
    left.startsInThinking === right.startsInThinking &&
    sameNumberSequence(left.startTokenIds, right.startTokenIds) &&
    sameNumberSequence(left.endTokenIds, right.endTokenIds);
}

function sameNumberSequence(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function validateContextProfile(value: unknown, index: number): ModelContextProfile {
  const path = `contextProfile[${index}]`;
  const item = record(value, path);
  exactKeys(
    item,
    [
      "contextTokens",
      "bindingSha256",
      "minimumCalibrationScore",
      "minimumDeviceMemoryGiB",
      "expectedPrefillTokensPerSecond",
      "expectedDecodeTokensPerSecond",
      "measuredDevices",
    ],
    path,
  );
  positiveInteger(item.contextTokens, `${path}.contextTokens`);
  sha(item.bindingSha256, `${path}.bindingSha256`);
  nullableNumber(item.minimumCalibrationScore, `${path}.minimumCalibrationScore`);
  nullableNumber(item.minimumDeviceMemoryGiB, `${path}.minimumDeviceMemoryGiB`);
  nullableRange(item.expectedPrefillTokensPerSecond, `${path}.expectedPrefillTokensPerSecond`);
  nullableRange(item.expectedDecodeTokensPerSecond, `${path}.expectedDecodeTokensPerSecond`);
  positiveInteger(item.measuredDevices, `${path}.measuredDevices`, true);
  for (const field of ["minimumCalibrationScore", "minimumDeviceMemoryGiB"] as const) {
    const number = item[field];
    if (typeof number === "number" && number < 0) {
      throw invalid(`${path}.${field} must be non-negative`);
    }
  }
  return item as unknown as ModelContextProfile;
}

function validateFile(value: unknown, index: number): CatalogFile {
  const path = `file[${index}]`;
  const item = record(value, path);
  exactKeys(item, ["path", "role", "url", "revision", "bytes", "sha256"], path);
  text(item.path, `${path}.path`);
  const filePath = item.path as string;
  const parts = filePath.split("/");
  if (
    filePath.startsWith("/") ||
    filePath.includes("\\") ||
    filePath.includes("\0") ||
    /^[A-Za-z]:/.test(filePath) ||
    parts.some((part) => !part || part === "." || part === "..")
  ) {
    throw invalid(`${path}.path is unsafe`);
  }
  if (![
    "weight",
    "tokenizer",
    "configuration",
    "converted_manifest",
    "chat_template",
    "model_library",
    "core_pack",
    "instrument",
  ].includes(item.role as string)) {
    throw invalid(`${path}.role is invalid`);
  }
  httpsUrl(item.url, `${path}.url`);
  text(item.revision, `${path}.revision`);
  if (!COMMIT.test(item.revision as string)) throw invalid(`${path}.revision must be a commit hash`);
  const url = new URL(item.url as string);
  const urlParts = url.pathname.split("/").filter(Boolean);
  if (
    url.hostname !== "huggingface.co" || url.username || url.password ||
    url.port || url.hash || urlParts.length < 5 ||
    !urlParts[0] || !urlParts[1].startsWith("drowse-web-") ||
    urlParts[2] !== "resolve" || urlParts[3] !== item.revision
  ) {
    throw invalid(`${path}.url must be an immutable Drowse Hugging Face URL`);
  }
  positiveInteger(item.bytes, `${path}.bytes`);
  sha(item.sha256, `${path}.sha256`);
  return item as unknown as CatalogFile;
}

function validatePack(value: unknown, index: number): CatalogInstrumentPack {
  const path = `pack[${index}]`;
  const item = record(value, path);
  exactKeys(
    item,
    [
      "id",
      "kind",
      "displayName",
      "license",
      "sourceRepository",
      "sourceRevision",
      "bytes",
      "required",
      "runtimeIdentitySha256",
      "compatibleContextBindingSha256",
      "files",
    ],
    path,
  );
  identifier(item.id, `${path}.id`);
  if (item.kind !== "core" && item.kind !== "jlens" && item.kind !== "sae") {
    throw invalid(`${path}.kind is invalid`);
  }
  text(item.displayName, `${path}.displayName`);
  text(item.license, `${path}.license`);
  const sourceRepository = hfRepository(
    item.sourceRepository,
    `${path}.sourceRepository`,
  );
  text(item.sourceRevision, `${path}.sourceRevision`);
  if (!COMMIT.test(item.sourceRevision as string)) {
    throw invalid(`${path}.sourceRevision must be a commit hash`);
  }
  positiveInteger(item.bytes, `${path}.bytes`);
  if (typeof item.required !== "boolean") throw invalid(`${path}.required must be boolean`);
  sha(item.runtimeIdentitySha256, `${path}.runtimeIdentitySha256`);
  const compatibleContextBindingSha256 = array(
    item.compatibleContextBindingSha256,
    `${path}.compatibleContextBindingSha256`,
  ).map((entry, i) => {
    sha(entry, `${path}.compatibleContextBindingSha256[${i}]`);
    return entry as string;
  });
  const files = array(item.files, `${path}.files`).map(validateFile);
  if (!files.length) throw invalid(`${path}.files must not be empty`);
  unique(files.map((file) => file.path), `${path}.files contains duplicate paths`);
  if (sumBytes(files, `${path}.files`) !== item.bytes) {
    throw invalid(`${path}.bytes does not equal its exact file total`);
  }
  const expectedRole = item.kind === "core" ? "core_pack" : "instrument";
  if (files.some((file) => file.role !== expectedRole)) {
    throw invalid(`${path}.files have roles incompatible with ${item.kind}`);
  }
  const expectedPrefix =
    `https://huggingface.co/${sourceRepository}/resolve/${item.sourceRevision}/`;
  if (files.some((file) =>
    file.revision !== item.sourceRevision || !file.url.startsWith(expectedPrefix)
  )) {
    throw invalid(
      `${path}.files do not match the pack's immutable source repository and revision`,
    );
  }
  unique(
    compatibleContextBindingSha256,
    `${path}.compatibleContextBindingSha256 contains duplicates`,
  );
  return {
    ...(item as unknown as CatalogInstrumentPack),
    sourceRepository,
    compatibleContextBindingSha256,
    files,
  };
}

function hfRepository(value: unknown, path: string): string {
  const repository = text(value, path);
  const parts = repository.split("/");
  if (
    parts.length !== 2 || parts.some((part) =>
      !/^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$/.test(part)
    ) || repository.length > MAX_CATALOG_ID_LENGTH
  ) {
    throw invalid(`${path} must be a Hugging Face owner/repository name`);
  }
  if (!parts[1].startsWith("drowse-web-")) {
    throw invalid(`${path} must identify a Drowse web repository`);
  }
  return repository;
}

function record(value: unknown, path: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw invalid(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function exactKeys(
  value: Record<string, unknown>,
  expected: string[],
  path: string,
): void {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    throw invalid(`${path} has unknown or missing fields`);
  }
}

function unique<T>(values: T[], message: string): void {
  if (new Set(values).size !== values.length) throw invalid(message);
}

function sumBytes(files: CatalogFile[], path: string): number {
  let total = 0;
  for (const file of files) {
    total += file.bytes;
    if (!Number.isSafeInteger(total)) throw invalid(`${path} byte total is unsafe`);
  }
  return total;
}

function array(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw invalid(`${path} must be an array`);
  return value;
}

function text(value: unknown, path: string, allowEmpty = false): string {
  if (typeof value !== "string" || (!allowEmpty && value.length === 0)) {
    throw invalid(`${path} must be ${allowEmpty ? "a string" : "a non-empty string"}`);
  }
  return value;
}

function identifier(value: unknown, path: string): string {
  const result = text(value, path);
  if (result.length > MAX_CATALOG_ID_LENGTH || /\p{Cc}/u.test(result)) {
    throw invalid(
      `${path} must be at most ${MAX_CATALOG_ID_LENGTH} characters without control characters`,
    );
  }
  return result;
}

function integer(value: unknown, path: string): number {
  if (!Number.isSafeInteger(value)) throw invalid(`${path} must be an integer`);
  return value as number;
}

function positiveInteger(value: unknown, path: string, allowZero = false): number {
  const result = integer(value, path);
  if (result < (allowZero ? 0 : 1)) throw invalid(`${path} must be positive`);
  return result;
}

function nullableNumber(value: unknown, path: string): void {
  if (value !== null && (typeof value !== "number" || !Number.isFinite(value))) {
    throw invalid(`${path} must be a finite number or null`);
  }
}

function nullableRange(value: unknown, path: string): void {
  if (value === null) return;
  const entries = array(value, path);
  if (
    entries.length !== 2 ||
    entries.some(
      (entry) => typeof entry !== "number" || !Number.isFinite(entry) || entry < 0,
    ) ||
    (entries[0] as number) > (entries[1] as number)
  ) {
    throw invalid(`${path} must be an ordered two-number range or null`);
  }
}

function httpsUrl(value: unknown, path: string): void {
  text(value, path);
  let url: URL;
  try {
    url = new URL(value as string);
  } catch {
    throw invalid(`${path} must be a valid URL`);
  }
  if (url.protocol !== "https:") throw invalid(`${path} must use HTTPS`);
}

function sha(value: unknown, path: string): void {
  if (typeof value !== "string" || !SHA256.test(value)) {
    throw invalid(`${path} must be a lowercase SHA-256 digest`);
  }
}

function decodeBase64(value: string): Uint8Array {
  let binary: string;
  try {
    binary = atob(value);
  } catch {
    throw new CatalogValidationError(
      "CATALOG_SIGNATURE_INVALID",
      "The catalog signature is not valid base64",
    );
  }
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

function validateDetachedSignature(signature: DetachedCatalogSignature): void {
  if (!signature || typeof signature !== "object" || Array.isArray(signature)) {
    throw new CatalogValidationError(
      "CATALOG_SIGNATURE_INVALID",
      "The detached catalog signature must be an object",
    );
  }
  const recordValue = signature as unknown as Record<string, unknown>;
  const keys = Object.keys(recordValue).sort();
  const expected = ["algorithm", "keyId", "schemaVersion", "signature"];
  if (keys.length !== expected.length || keys.some((key, index) => key !== expected[index])) {
    throw new CatalogValidationError(
      "CATALOG_SIGNATURE_INVALID",
      "The detached catalog signature has unknown or missing fields",
    );
  }
  if (
    signature.schemaVersion !== 1 || signature.algorithm !== "Ed25519" ||
    !/^[a-z0-9][a-z0-9._-]{0,63}$/.test(signature.keyId)
  ) {
    throw new CatalogValidationError(
      "CATALOG_SIGNATURE_INVALID",
      "The detached catalog signature metadata is invalid",
    );
  }
  if (!/^[A-Za-z0-9+/]{86}==$/.test(signature.signature)) {
    throw new CatalogValidationError(
      "CATALOG_SIGNATURE_INVALID",
      "The catalog signature is not a canonical Ed25519 signature",
    );
  }
  if (decodeBase64(signature.signature).byteLength !== 64) {
    throw new CatalogValidationError(
      "CATALOG_SIGNATURE_INVALID",
      "The catalog signature must be 64 bytes",
    );
  }
}

function digestCanonical(value: unknown): string {
  const bytes = new TextEncoder().encode(canonicalJson(value));
  return bytesToHex(sha256(bytes));
}

export function runtimeIdentitySha256(identity: RuntimeIdentity): string {
  return digestCanonical(identity);
}

export function contextBindingSha256(
  identitySha256: string,
  contextTokens: number,
): string {
  return digestCanonical({ contextTokens, runtimeIdentitySha256: identitySha256 });
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const entries = Object.entries(value as Record<string, unknown>)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`);
  return `{${entries.join(",")}}`;
}

function invalid(message: string): CatalogValidationError {
  return new CatalogValidationError("CATALOG_SCHEMA_INVALID", message);
}
