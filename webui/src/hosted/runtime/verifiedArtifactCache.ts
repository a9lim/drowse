import type { BrowserModelArtifact } from "./modelBackend";

export interface ReadOnlyArtifactCache {
  fetchWithCache(url: string, storeType?: string, signal?: AbortSignal): Promise<unknown>;
  addToCache(url: string, storeType?: string, signal?: AbortSignal): Promise<void>;
  hasAllKeys(keys: string[]): Promise<boolean>;
  deleteInCache(url: string): Promise<void>;
}

export interface VerifiedWebLlmArtifacts {
  artifactCache: ReadOnlyArtifactCache;
  modelUrl: string;
  modelLibraryUrl: string;
  captureSpecialTokenIds: number[];
  isBaseModel: boolean;
  defaultUserRole: string | null;
  defaultAssistantRole: string | null;
}

export async function prepareVerifiedWebLlmArtifacts(
  artifacts: readonly BrowserModelArtifact[],
  signal?: AbortSignal,
): Promise<VerifiedWebLlmArtifacts> {
  throwIfAborted(signal);
  const files = new Map<string, BrowserModelArtifact>();
  for (const artifact of artifacts) {
    if (artifact.file.size !== artifact.manifest.bytes) {
      throw invalid(
        "VERIFIED_ARTIFACT_SIZE_MISMATCH",
        `Verified artifact ${artifact.manifest.path} has an unexpected size`,
      );
    }
    if (files.has(artifact.manifest.url)) {
      throw invalid(
        "VERIFIED_ARTIFACT_URL_DUPLICATE",
        `Verified artifacts contain duplicate URL ${artifact.manifest.url}`,
      );
    }
    files.set(artifact.manifest.url, artifact);
  }

  const chatConfig = exactlyOne(
    artifacts.filter(
      ({ manifest }) =>
        manifest.role === "configuration" &&
        urlBasename(manifest.url) === "mlc-chat-config.json",
    ),
    "MLC_CHAT_CONFIG_INVALID",
    "exactly one mlc-chat-config.json configuration",
  );
  const tensorCache = exactlyOne(
    artifacts.filter(
      ({ manifest }) =>
        manifest.role === "converted_manifest" &&
        urlBasename(manifest.url) === "tensor-cache.json",
    ),
    "MLC_TENSOR_CACHE_INVALID",
    "exactly one tensor-cache.json converted manifest",
  );
  const modelLibrary = exactlyOne(
    artifacts.filter(({ manifest }) => manifest.role === "model_library"),
    "MLC_MODEL_LIBRARY_INVALID",
    "exactly one model library",
  );
  if (!urlBasename(modelLibrary.manifest.url).endsWith(".wasm")) {
    throw invalid("MLC_MODEL_LIBRARY_INVALID", "The model library must be a WebAssembly file");
  }

  const modelUrl = new URL(".", chatConfig.manifest.url).href;
  if (new URL("tensor-cache.json", modelUrl).href !== tensorCache.manifest.url) {
    throw invalid(
      "MLC_TENSOR_CACHE_INVALID",
      "tensor-cache.json must share the model configuration directory",
    );
  }

  const config = await parseJsonFile(chatConfig, signal);
  const tokenizerFiles = stringArray(config.tokenizer_files);
  if (
    tokenizerFiles.length === 0 ||
    new Set(tokenizerFiles).size !== tokenizerFiles.length ||
    tokenizerFiles.filter((path) => path === "tokenizer.json").length !== 1
  ) {
    throw invalid(
      "MLC_TOKENIZER_INVALID",
      "mlc-chat-config.json must select exactly one tokenizer.json file",
    );
  }
  let tokenizer: BrowserModelArtifact | null = null;
  let tokenizerConfig: BrowserModelArtifact | null = null;
  for (const path of tokenizerFiles) {
    const tokenizerUrl = resolveSafeRelative(path, modelUrl);
    const artifact = files.get(tokenizerUrl);
    const expectedRole = path === "tokenizer.json"
      ? "tokenizer"
      : path === "tokenizer_config.json"
        ? "chat_template"
        : "configuration";
    if (!artifact || artifact.manifest.role !== expectedRole) {
      throw invalid(
        "MLC_TOKENIZER_INVALID",
        `Tokenizer support file ${path} is not installed with the required ${expectedRole} role`,
      );
    }
    if (path === "tokenizer.json") tokenizer = artifact;
    if (path === "tokenizer_config.json") tokenizerConfig = artifact;
  }
  if (tokenizer === null) throw invalid("MLC_TOKENIZER_INVALID", "The verified tokenizer is unavailable");
  if (tokenizerConfig === null) {
    throw invalid("MLC_TOKENIZER_INVALID", "The verified tokenizer configuration is unavailable");
  }
  const tokenizerDocument = await parseJsonFile(tokenizer, signal);
  const tokenizerConfiguration = await parseJsonFile(tokenizerConfig, signal);
  const isBaseModel = tokenizerConfiguration.chat_template == null || isRawCompletionConfig(config);
  const defaultRoles = isBaseModel
    ? { user: null, assistant: null }
    : defaultRoleLabels(config);
  const requiredCaptureTokenIds = [...new Set([
    ...tokenizerAddedTokenIds(tokenizerDocument),
    ...tokenizerConfigurationAddedTokenIds(tokenizerConfiguration),
    ...configuredSpecialTokenIds(config),
  ])].sort((left, right) => left - right);
  validateCompletionPrefix(config, tokenizerConfiguration, isBaseModel, requiredCaptureTokenIds);
  const declaredCaptureTokenIds = brandedConfigField(config, "capture_special_token_ids", "MLC_CAPTURE_TOKEN_IDS_INVALID");
  const captureSpecialTokenIds = declaredCaptureTokenIds === undefined
    ? requiredCaptureTokenIds
    : exactTokenIdArray(declaredCaptureTokenIds);
  if (captureSpecialTokenIds.length === 0) {
    throw invalid(
      "MLC_CAPTURE_TOKEN_IDS_INVALID",
      "The verified tokenizer metadata does not define valid capture special-token IDs",
    );
  }
  const captureIds = new Set(captureSpecialTokenIds);
  if (requiredCaptureTokenIds.some((tokenId) => !captureIds.has(tokenId))) {
    throw invalid(
      "MLC_CAPTURE_TOKEN_IDS_INVALID",
      "The Drowse capture token list omits a tokenizer or model special token",
    );
  }

  const tensorDocument = await parseJsonFile(tensorCache, signal);
  const records = Array.isArray(tensorDocument.records) ? tensorDocument.records : null;
  if (!records || records.length === 0) {
    throw invalid("MLC_TENSOR_CACHE_INVALID", "tensor-cache.json has no weight shards");
  }
  const referencedWeights = new Set<string>();
  for (const record of records) {
    if (!isRecord(record) || typeof record.dataPath !== "string") {
      throw invalid("MLC_TENSOR_CACHE_INVALID", "tensor-cache.json contains an invalid shard record");
    }
    const shardUrl = resolveSafeRelative(record.dataPath, modelUrl);
    if (referencedWeights.has(shardUrl)) {
      throw invalid("MLC_TENSOR_CACHE_INVALID", "tensor-cache.json references a shard more than once");
    }
    const shard = files.get(shardUrl);
    if (!shard || shard.manifest.role !== "weight") {
      throw invalid(
        "MLC_WEIGHT_CLOSURE_INVALID",
        `tensor-cache.json references an unverified weight shard ${record.dataPath}`,
      );
    }
    referencedWeights.add(shardUrl);
  }
  const installedWeights = new Set(
    artifacts
      .filter(({ manifest }) => manifest.role === "weight")
      .map(({ manifest }) => manifest.url),
  );
  if (!sameSet(referencedWeights, installedWeights)) {
    throw invalid(
      "MLC_WEIGHT_CLOSURE_INVALID",
      "The installed weight set does not exactly match tensor-cache.json",
    );
  }

  return {
    artifactCache: new VerifiedArtifactCache(files),
    modelUrl,
    modelLibraryUrl: modelLibrary.manifest.url,
    captureSpecialTokenIds,
    isBaseModel,
    defaultUserRole: defaultRoles.user,
    defaultAssistantRole: defaultRoles.assistant,
  };
}

function isRawCompletionConfig(config: Record<string, unknown>): boolean {
  const conversation = config.conv_template;
  return isRecord(conversation) &&
    ["drowse-base", "polythetic-base", "saklas-base"].includes(String(conversation.name)) &&
    brandedConfigField(config, "completion_prefix_token_ids", "MLC_COMPLETION_PREFIX_INVALID") !== undefined &&
    conversation.system_template === "" && conversation.system_message === "" &&
    isRecord(conversation.roles) && conversation.roles.user === "" && conversation.roles.assistant === "" &&
    conversation.role_content_sep === "" && conversation.role_empty_sep === "" &&
    JSON.stringify(conversation.seps) === '[""]';
}

function defaultRoleLabels(config: Record<string, unknown>): {
  user: string;
  assistant: string;
} {
  const conversation = isRecord(config.conv_template) ? config.conv_template : null;
  const roles = conversation && isRecord(conversation.roles) ? conversation.roles : null;
  if (roles === null) {
    throw invalid(
      "MLC_CHAT_ROLES_INVALID",
      "mlc-chat-config.json must declare the chat template's user and assistant role markers",
    );
  }
  return {
    user: roleLabelFromMarker(roles.user, "user"),
    assistant: roleLabelFromMarker(roles.assistant, "assistant"),
  };
}

function roleLabelFromMarker(value: unknown, structuralRole: "user" | "assistant"): string {
  if (typeof value !== "string" || value.length === 0) {
    throw invalid(
      "MLC_CHAT_ROLES_INVALID",
      `mlc-chat-config.json is missing its ${structuralRole} role marker`,
    );
  }
  const visible = value.replace(/<[^>]+>/g, " ").trim();
  if (visible === "") return structuralRole;
  const labels = visible.match(/[A-Za-z][A-Za-z0-9._-]*/g);
  if (labels?.length !== 1) {
    throw invalid(
      "MLC_CHAT_ROLES_INVALID",
      `The ${structuralRole} role marker does not contain one readable role label`,
    );
  }
  return labels[0].toLowerCase();
}

class VerifiedArtifactCache implements ReadOnlyArtifactCache {
  constructor(private readonly files: ReadonlyMap<string, BrowserModelArtifact>) {}

  async fetchWithCache(
    url: string,
    storeType = "arraybuffer",
    signal?: AbortSignal,
  ): Promise<unknown> {
    throwIfAborted(signal);
    const artifact = this.require(url);
    if (storeType === "json") return parseJsonFile(artifact, signal);
    if (storeType !== "arraybuffer") {
      throw invalid(
        "VERIFIED_ARTIFACT_STORE_TYPE_INVALID",
        `Unsupported verified artifact representation ${storeType}`,
      );
    }
    const value = await artifact.file.arrayBuffer();
    throwIfAborted(signal);
    return value;
  }

  async addToCache(url: string, _storeType?: string, signal?: AbortSignal): Promise<void> {
    throwIfAborted(signal);
    this.require(url);
  }

  async hasAllKeys(keys: string[]): Promise<boolean> {
    return keys.every((key) => this.files.has(key));
  }

  async deleteInCache(): Promise<void> {
    throw invalid(
      "VERIFIED_ARTIFACT_READ_ONLY",
      "The model runtime cannot delete caller-owned verified artifacts",
    );
  }

  private require(url: string): BrowserModelArtifact {
    const artifact = this.files.get(url);
    if (!artifact) {
      throw invalid(
        "UNVERIFIED_ARTIFACT_REQUESTED",
        `The model runtime requested an artifact outside the verified installation: ${url}`,
      );
    }
    return artifact;
  }
}

async function parseJsonFile(
  artifact: BrowserModelArtifact,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  throwIfAborted(signal);
  let value: unknown;
  try {
    value = JSON.parse(await artifact.file.text());
  } catch {
    throw invalid(
      "VERIFIED_ARTIFACT_JSON_INVALID",
      `Verified artifact ${artifact.manifest.path} is not valid JSON`,
    );
  }
  throwIfAborted(signal);
  if (!isRecord(value)) {
    throw invalid(
      "VERIFIED_ARTIFACT_JSON_INVALID",
      `Verified artifact ${artifact.manifest.path} must contain a JSON object`,
    );
  }
  return value;
}

function resolveSafeRelative(path: string, base: string): string {
  let decodedParts: string[];
  try {
    decodedParts = path.split("/").map((part) => decodeURIComponent(part));
  } catch {
    throw invalid("MLC_TENSOR_CACHE_INVALID", `Invalid tensor shard path encoding ${path}`);
  }
  if (
    path === "" || path.includes("\\") || path.includes("\0") ||
    path.includes("?") || path.includes("#") || path.startsWith("/") ||
    /^[A-Za-z][A-Za-z\d+.-]*:/.test(path) ||
    decodedParts.some(
      (part) =>
        part === "" || part === "." || part === ".." ||
        part.includes("/") || part.includes("\\") || part.includes("\0"),
    )
  ) {
    throw invalid("MLC_TENSOR_CACHE_INVALID", `Unsafe tensor shard path ${path}`);
  }
  const resolved = new URL(path, base);
  if (resolved.origin !== new URL(base).origin || !resolved.href.startsWith(base)) {
    throw invalid("MLC_TENSOR_CACHE_INVALID", `Tensor shard path escapes its model origin: ${path}`);
  }
  return resolved.href;
}

function urlBasename(url: string): string {
  const parts = new URL(url).pathname.split("/");
  return parts.at(-1) ?? "";
}

function exactlyOne<T>(values: readonly T[], code: string, label: string): T {
  if (values.length !== 1) throw invalid(code, `The verified model must contain ${label}`);
  return values[0];
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) && value.every((entry) => typeof entry === "string")
    ? value
    : [];
}

function exactTokenIdArray(value: unknown): number[] {
  if (
    !Array.isArray(value) || value.length === 0 ||
    value.some((entry) => !Number.isSafeInteger(entry) || entry < 0)
  ) return [];
  const result = value as number[];
  if (
    new Set(result).size !== result.length ||
    result.some((entry, index) => index > 0 && result[index - 1] >= entry)
  ) return [];
  return [...result];
}

function tokenizerAddedTokenIds(tokenizer: Record<string, unknown>): number[] {
  if (tokenizer.added_tokens === undefined) return [];
  if (!Array.isArray(tokenizer.added_tokens)) {
    throw invalid(
      "MLC_TOKENIZER_INVALID",
      "tokenizer.json added_tokens must be an array",
    );
  }
  const ids: number[] = [];
  for (const token of tokenizer.added_tokens) {
    if (!isRecord(token) || !Number.isSafeInteger(token.id) || (token.id as number) < 0) {
      throw invalid(
        "MLC_TOKENIZER_INVALID",
        "tokenizer.json contains an invalid added token ID",
      );
    }
    ids.push(token.id as number);
  }
  return ids;
}

function tokenizerConfigurationAddedTokenIds(
  tokenizerConfiguration: Record<string, unknown>,
): number[] {
  const decoder = tokenizerConfiguration.added_tokens_decoder;
  if (decoder === undefined) return [];
  if (!isRecord(decoder)) {
    throw invalid(
      "MLC_TOKENIZER_INVALID",
      "tokenizer_config.json added_tokens_decoder must be an object",
    );
  }
  const ids: number[] = [];
  for (const [key, token] of Object.entries(decoder)) {
    const id = Number(key);
    if (
      !/^(0|[1-9]\d*)$/.test(key) || !Number.isSafeInteger(id) || id < 0 ||
      !isRecord(token)
    ) {
      throw invalid(
        "MLC_TOKENIZER_INVALID",
        "tokenizer_config.json contains an invalid added token ID",
      );
    }
    ids.push(id);
  }
  return ids;
}

function validateCompletionPrefix(
  config: Record<string, unknown>,
  tokenizer: Record<string, unknown>,
  isBaseModel: boolean,
  specialTokenIds: readonly number[],
): void {
  const prefix = brandedConfigField(config, "completion_prefix_token_ids", "MLC_COMPLETION_PREFIX_INVALID");
  const requiresBos = isBaseModel && tokenizer.add_bos_token === true;
  if (prefix === undefined && !requiresBos) return;
  const capturePrefix = isRecord(config.conv_template)
    ? config.conv_template.system_prefix_token_ids
    : undefined;
  if (
    !isBaseModel || !Array.isArray(prefix) || prefix.length > 32 ||
    !Number.isSafeInteger(config.vocab_size) || (config.vocab_size as number) <= 0 ||
    prefix.some((id) => !Number.isSafeInteger(id) || id < 0 ||
      id >= (config.vocab_size as number) || !specialTokenIds.includes(id)) ||
    !Array.isArray(capturePrefix) || capturePrefix.length !== prefix.length ||
    prefix.some((id, index) => id !== capturePrefix[index]) ||
    (requiresBos && prefix[0] !== config.bos_token_id)
  ) {
    throw invalid(
      "MLC_COMPLETION_PREFIX_INVALID",
      "The base model's completion and capture prefixes must match its tokenizer's special tokens",
    );
  }
}

function brandedConfigField(config: Record<string, unknown>, suffix: string, code: string): unknown {
  const values = ["drowse", "polythetic", "saklas"].map((name) => config[`${name}_${suffix}`])
    .filter((value) => value !== undefined);
  if (values.some((value) => JSON.stringify(value) !== JSON.stringify(values[0]))) {
    throw invalid(code, "The current and legacy capture special-token lists disagree");
  }
  return values[0];
}

function configuredSpecialTokenIds(config: Record<string, unknown>): number[] {
  const ids: number[] = [];
  for (const key of ["bos_token_id", "eos_token_id", "pad_token_id"]) {
    const value = config[key];
    if (value === undefined || value === null) continue;
    const values = Array.isArray(value) ? value : [value];
    if (values.some((entry) => !Number.isSafeInteger(entry) || entry < 0)) {
      throw invalid(
        "MLC_CHAT_CONFIG_INVALID",
        `mlc-chat-config.json ${key} is invalid`,
      );
    }
    ids.push(...values as number[]);
  }
  return ids;
}

function sameSet(left: ReadonlySet<string>, right: ReadonlySet<string>): boolean {
  return left.size === right.size && [...left].every((value) => right.has(value));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw signal.reason ?? new DOMException("Model loading was cancelled", "AbortError");
  }
}

function invalid(code: string, message: string): Error & { code: string } {
  return Object.assign(new Error(message), { code });
}
