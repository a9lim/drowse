import type { RuntimeServiceRequest } from "../../lib/runtime/contracts";
import type { LoomTreeJSON, SessionInfo } from "../../lib/types";
import {
  clampOutputTokenCount,
  DESKTOP_MAX_OUTPUT_TOKENS,
} from "../../lib/runtime/outputTokenPolicy";
import {
  type HostedSessionSettings,
  type HostedSessionBinding,
  type HostedSessionClaimResult,
  type HostedSessionSave,
  type PersistedHostedSession,
  SESSION_SCHEMA_VERSION,
  validatePersistedHostedSession,
} from "./sessionPersistence";

const SHA256 = /^[a-f0-9]{64}$/;
const SESSION_KEYS = [
  "config",
  "created",
  "default_assistant_role",
  "default_steering",
  "default_user_role",
  "device",
  "dtype",
  "history_length",
  "id",
  "instruments",
  "is_base_model",
  "jlens_fitted",
  "model_id",
  "probes",
  "profiles",
  "role_substitution_supported",
  "scene_mode",
  "strips_history_thinking",
  "supports_thinking",
  "thinking_input_supported",
  "thinking_is_optional",
  "user_role_supported",
] as const;
const CONFIG_KEYS = [
  "max_tokens",
  "system_prompt",
  "temperature",
  "thinking",
  "top_k",
  "top_p",
] as const;

export type HostedSessionRuntimeIdentity = HostedSessionBinding;

export type HostedSessionCompatibilityReason =
  | "runtime_identity"
  | "context_tokens"
  | "model_id";

export type HostedSessionRestoreResult =
  | { status: "not_found" }
  | { status: "unsupported"; preserved: true }
  | {
      status: "incompatible";
      reason: HostedSessionCompatibilityReason;
      preserved: true;
    }
  | {
      status: "restored";
      session: SessionInfo;
      tree: LoomTreeJSON;
      restoredConfigKeys: (keyof SessionInfo["config"])[];
    };

export type RuntimeServiceExecutor = (
  request: RuntimeServiceRequest,
) => Promise<unknown>;

export type SessionPersistenceRestoreMode = "replace" | "reset";

export interface BrowserSessionCoordinatorOptions {
  maxOutputTokens?: number;
}

export interface SessionCoordinatorPersistencePort {
  claim(
    binding: HostedSessionRuntimeIdentity,
    mode?: "restore" | "reset",
    compatibleRuntimeIdentitySha256s?: readonly string[],
  ): Promise<HostedSessionClaimResult>;
  save(
    input: HostedSessionSave,
    ownerEpoch: string,
  ): Promise<PersistedHostedSession>;
  delete(modelVariantId: string, ownerEpoch?: string): Promise<boolean>;
  clear(ownerEpoch?: string): Promise<void>;
  close(): void;
}

export class SessionCoordinatorError extends Error {
  constructor(
    readonly code:
      | "INVALID_RUNTIME_IDENTITY"
      | "INVALID_RUNTIME_RESPONSE"
      | "INVALID_PERSISTED_SETTINGS"
      | "RUNTIME_IDENTITY_MISMATCH"
      | "OWNERSHIP_NOT_CLAIMED"
      | "COORDINATOR_CLOSED",
    message: string,
  ) {
    super(message);
    this.name = "SessionCoordinatorError";
  }
}

export const MUTATING_RUNTIME_METHODS = Object.freeze({
  sessions: Object.freeze(["patch"]),
  tree: Object.freeze([
    "branch",
    "castDelete",
    "castPut",
    "delete",
    "edit",
    "navigate",
    "note",
    "reset",
    "restore",
    "star",
    "transcriptLoad",
  ]),
});

export function mutatesPersistedHostedSession(
  request: RuntimeServiceRequest,
): boolean {
  if (request.service === "sessions") {
    return MUTATING_RUNTIME_METHODS.sessions.includes(request.method);
  }
  if (request.service === "tree") {
    return MUTATING_RUNTIME_METHODS.tree.includes(request.method);
  }
  return false;
}

interface PersistAction {
  kind: "persist";
  identity: HostedSessionRuntimeIdentity;
  waiters: Array<{
    resolve(record: PersistedHostedSession): void;
    reject(error: unknown): void;
  }>;
}

interface DeleteAction {
  kind: "delete";
  modelVariantId: string;
  ownerEpoch: string;
  resolve(deleted: boolean): void;
  reject(error: unknown): void;
}

interface ClearAction {
  kind: "clear";
  ownerEpoch: string;
  resolve(): void;
  reject(error: unknown): void;
}

interface CloseAction {
  kind: "close";
  resolve(): void;
}

type CoordinatorAction = PersistAction | DeleteAction | ClearAction | CloseAction;

export class BrowserSessionCoordinator {
  private readonly persistence: SessionCoordinatorPersistencePort;
  private readonly execute: RuntimeServiceExecutor;
  private readonly queue: CoordinatorAction[] = [];
  private readonly claimed = new Map<string, HostedSessionRuntimeIdentity>();
  private draining = false;
  private drainScheduled = false;
  private accepting = true;
  private closePromise: Promise<void> | null = null;
  private readonly maxOutputTokens: number;
  modelDefaults: SessionInfo | undefined;

  constructor(
    persistence: SessionCoordinatorPersistencePort,
    execute: RuntimeServiceExecutor,
    options: BrowserSessionCoordinatorOptions = {},
  ) {
    this.persistence = persistence;
    this.execute = execute;
    this.maxOutputTokens = options.maxOutputTokens ?? DESKTOP_MAX_OUTPUT_TOKENS;
    if (!Number.isSafeInteger(this.maxOutputTokens) || this.maxOutputTokens < 1) {
      throw new SessionCoordinatorError(
        "INVALID_RUNTIME_IDENTITY",
        "The browser output token limit must be a positive integer",
      );
    }
  }

  async restore(
    identity: HostedSessionRuntimeIdentity,
    mode: SessionPersistenceRestoreMode = "replace",
    compatibleRuntimeIdentitySha256s: readonly string[] = [],
  ): Promise<HostedSessionRestoreResult> {
    this.requireOpen();
    validateRuntimeIdentity(identity);
    if (mode !== "replace" && mode !== "reset") {
      throw new SessionCoordinatorError(
        "INVALID_RUNTIME_IDENTITY",
        "Session restore mode is invalid",
      );
    }
    const claim = await this.persistence.claim(
      structuredClone(identity),
      mode === "reset" ? "reset" : "restore",
      compatibleRuntimeIdentitySha256s,
    );
    validateClaimResult(claim);
    if (claim.status === "incompatible") {
      return { status: "incompatible", reason: claim.reason, preserved: true };
    }
    if (claim.status === "unsupported") {
      return { status: "unsupported", preserved: true };
    }
    this.claimed.set(identity.modelVariantId, structuredClone(identity));
    const current = await this.readSession();
    this.modelDefaults = structuredClone(current);
    if (claim.status === "missing") return { status: "not_found" };
    const record = claim.record;
    if (record.modelVariantId !== identity.modelVariantId) {
      throw invalidResponse("Claimed session model variant is invalid");
    }
    if (record.metadata.runtimeIdentitySha256 !== identity.runtimeIdentitySha256) {
      throw invalidResponse("Claimed session runtime identity is invalid");
    }
    if (record.metadata.contextTokens !== identity.contextTokens) {
      throw invalidResponse("Claimed session context binding is invalid");
    }

    if (record.metadata.modelId !== current.model_id) {
      this.claimed.delete(identity.modelVariantId);
      return { status: "incompatible", reason: "model_id", preserved: true };
    }
    const savedConfig = persistedConfig(record.settings);
    const patch = supportedConfigPatch(savedConfig, this.maxOutputTokens);

    const restoreResponse = await this.execute({
      service: "tree",
      method: "restore",
      args: [structuredClone(record.tree)],
    });
    const restoreRevision = validateRestoreResponse(restoreResponse, record.tree);

    let session = current;
    const restoredConfigKeys = Object.keys(patch) as (keyof SessionInfo["config"])[];
    if (restoredConfigKeys.length > 0) {
      const patched = await this.execute({
        service: "sessions",
        method: "patch",
        args: [patch],
      });
      session = validateSessionInfo(patched);
      validateBackendIdentity(session, current);
      for (const key of restoredConfigKeys) {
        if (session.config[key] !== patch[key]) {
          throw invalidResponse(`Session patch did not apply ${key}`);
        }
      }
    }

    const restoredTreeValue = await this.execute({
      service: "tree",
      method: "get",
      args: [],
    });
    const restoredTree = validateTreeSnapshot(
      restoredTreeValue,
      session,
      identity,
    );
    if (
      restoredTree.rev !== restoreRevision ||
      restoredTree.root_id !== record.tree.root_id ||
      restoredTree.active_node_id !== record.tree.active_node_id ||
      restoredTree.nodes.length !== record.tree.nodes.length
    ) {
      throw invalidResponse("Restored tree does not match the backend restore result");
    }

    return {
      status: "restored",
      session: structuredClone(session),
      tree: structuredClone(restoredTree),
      restoredConfigKeys,
    };
  }

  persist(
    identity: HostedSessionRuntimeIdentity,
  ): Promise<PersistedHostedSession> {
    try {
      this.requireOpen();
      validateRuntimeIdentity(identity);
      this.requireClaim(identity);
    } catch (error) {
      return Promise.reject(error);
    }
    return new Promise<PersistedHostedSession>((resolve, reject) => {
      const previous = this.queue.at(-1);
      if (
        previous?.kind === "persist" &&
        sameRuntimeIdentity(previous.identity, identity)
      ) {
        previous.waiters.push({ resolve, reject });
      } else {
        this.queue.push({
          kind: "persist",
          identity: structuredClone(identity),
          waiters: [{ resolve, reject }],
        });
      }
      this.scheduleDrain();
    });
  }

  delete(identity: HostedSessionRuntimeIdentity): Promise<boolean> {
    try {
      this.requireOpen();
      validateRuntimeIdentity(identity);
    } catch (error) {
      return Promise.reject(error);
    }
    this.claimed.delete(identity.modelVariantId);
    return new Promise<boolean>((resolve, reject) => {
      this.queue.push({
        kind: "delete",
        modelVariantId: identity.modelVariantId,
        ownerEpoch: identity.ownerEpoch,
        resolve,
        reject,
      });
      this.scheduleDrain();
    });
  }

  clear(ownerEpoch: string): Promise<void> {
    try {
      this.requireOpen();
      validateOwnerEpoch(ownerEpoch);
    } catch (error) {
      return Promise.reject(error);
    }
    this.claimed.clear();
    return new Promise<void>((resolve, reject) => {
      this.queue.push({ kind: "clear", ownerEpoch, resolve, reject });
      this.scheduleDrain();
    });
  }

  close(): Promise<void> {
    if (this.closePromise !== null) return this.closePromise;
    this.accepting = false;
    this.claimed.clear();
    this.closePromise = new Promise<void>((resolve) => {
      this.queue.push({ kind: "close", resolve });
      this.scheduleDrain();
    });
    return this.closePromise;
  }

  private async captureAndSave(
    identity: HostedSessionRuntimeIdentity,
  ): Promise<PersistedHostedSession> {
    const session = await this.readSession();
    const treeValue = await this.execute({ service: "tree", method: "get", args: [] });
    const tree = validateTreeSnapshot(treeValue, session, identity);
    const input = {
      modelVariantId: identity.modelVariantId,
      metadata: {
        sessionId: session.id,
        modelId: session.model_id,
        runtimeIdentitySha256: identity.runtimeIdentitySha256,
        contextTokens: identity.contextTokens,
        createdAt: session.created,
      },
      settings: configSettings(session.config, this.maxOutputTokens),
      tree,
    };
    return this.persistence.save(input, identity.ownerEpoch);
  }

  private async readSession(): Promise<SessionInfo> {
    const response = await this.execute({
      service: "sessions",
      method: "get",
      args: [],
    });
    return validateSessionInfo(response);
  }

  private scheduleDrain(): void {
    if (this.draining || this.drainScheduled) return;
    this.drainScheduled = true;
    queueMicrotask(() => {
      this.drainScheduled = false;
      void this.drain();
    });
  }

  private async drain(): Promise<void> {
    if (this.draining) return;
    this.draining = true;
    try {
      while (this.queue.length > 0) {
        const action = this.queue.shift()!;
        if (action.kind === "persist") {
          try {
            const record = await this.captureAndSave(action.identity);
            for (const waiter of action.waiters) waiter.resolve(structuredClone(record));
          } catch (error) {
            for (const waiter of action.waiters) waiter.reject(error);
          }
          continue;
        }
        if (action.kind === "delete") {
          try {
            action.resolve(await this.persistence.delete(
              action.modelVariantId,
              action.ownerEpoch,
            ));
          } catch (error) {
            action.reject(error);
          }
          continue;
        }
        if (action.kind === "clear") {
          try {
            await this.persistence.clear(action.ownerEpoch);
            action.resolve();
          } catch (error) {
            action.reject(error);
          }
          continue;
        }
        this.persistence.close();
        action.resolve();
      }
    } finally {
      this.draining = false;
      if (this.queue.length > 0) this.scheduleDrain();
    }
  }

  private requireOpen(): void {
    if (!this.accepting) {
      throw new SessionCoordinatorError(
        "COORDINATOR_CLOSED",
        "The browser session coordinator is closed",
      );
    }
  }

  private requireClaim(identity: HostedSessionRuntimeIdentity): void {
    const claimed = this.claimed.get(identity.modelVariantId);
    if (claimed === undefined || !sameRuntimeIdentity(claimed, identity)) {
      throw new SessionCoordinatorError(
        "OWNERSHIP_NOT_CLAIMED",
        "The current ownership epoch has not claimed this browser session",
      );
    }
  }
}

function validateRuntimeIdentity(identity: HostedSessionRuntimeIdentity): void {
  const value = plainObject(identity, "Runtime identity");
  exactKeys(
    value,
    ["contextTokens", "modelVariantId", "ownerEpoch", "runtimeIdentitySha256"],
    "Runtime identity",
  );
  validateIdentifier(identity.modelVariantId, "model variant", 256);
  if (!SHA256.test(identity.runtimeIdentitySha256)) {
    throw new SessionCoordinatorError(
      "INVALID_RUNTIME_IDENTITY",
      "Runtime identity digest is invalid",
    );
  }
  if (
    !Number.isSafeInteger(identity.contextTokens) ||
    identity.contextTokens < 1 ||
    identity.contextTokens > 1_048_576
  ) {
    throw new SessionCoordinatorError(
      "INVALID_RUNTIME_IDENTITY",
      "Runtime context size is invalid",
    );
  }
  validateOwnerEpoch(identity.ownerEpoch);
}

function sameRuntimeIdentity(
  left: HostedSessionRuntimeIdentity,
  right: HostedSessionRuntimeIdentity,
): boolean {
  return left.modelVariantId === right.modelVariantId &&
    left.runtimeIdentitySha256 === right.runtimeIdentitySha256 &&
    left.contextTokens === right.contextTokens &&
    left.ownerEpoch === right.ownerEpoch;
}

function validateOwnerEpoch(ownerEpoch: unknown): asserts ownerEpoch is string {
  if (
    typeof ownerEpoch !== "string" ||
    ownerEpoch.length === 0 ||
    ownerEpoch.length > 256 ||
    /\p{Cc}/u.test(ownerEpoch)
  ) {
    throw new SessionCoordinatorError(
      "INVALID_RUNTIME_IDENTITY",
      "Runtime ownership epoch is invalid",
    );
  }
}

function validateClaimResult(value: unknown): asserts value is HostedSessionClaimResult {
  const claim = plainObject(value, "Session persistence claim");
  if (claim.status === "missing") {
    exactKeys(claim, ["status"], "Session persistence claim");
    return;
  }
  if (claim.status === "unsupported") {
    exactKeys(claim, ["schemaVersion", "status"], "Session persistence claim");
    if (
      claim.schemaVersion !== null &&
      (!Number.isSafeInteger(claim.schemaVersion) || Number(claim.schemaVersion) < 0)
    ) {
      throw invalidResponse("Unsupported session schema version is invalid");
    }
    return;
  }
  if (claim.status === "incompatible") {
    exactKeys(claim, ["reason", "record", "status"], "Session persistence claim");
    if (!["context_tokens", "runtime_identity"].includes(String(claim.reason))) {
      throw invalidResponse("Session persistence incompatibility reason is invalid");
    }
    validatePersistedHostedSession(claim.record);
    return;
  }
  if (claim.status === "compatible") {
    exactKeys(claim, ["record", "status"], "Session persistence claim");
    validatePersistedHostedSession(claim.record);
    return;
  }
  throw invalidResponse("Session persistence claim status is invalid");
}

function configSettings(
  config: SessionInfo["config"],
  maxOutputTokens = DESKTOP_MAX_OUTPUT_TOKENS,
): HostedSessionSettings {
  validateConfig(config, "Session config", "INVALID_RUNTIME_RESPONSE");
  return structuredClone({
    temperature: config.temperature,
    top_p: config.top_p,
    top_k: config.top_k,
    max_tokens: typeof config.max_tokens === "number"
      ? clampOutputTokenCount(config.max_tokens, maxOutputTokens)
      : config.max_tokens,
    system_prompt: config.system_prompt,
    thinking: config.thinking,
  });
}

function persistedConfig(settings: HostedSessionSettings): SessionInfo["config"] {
  try {
    validateConfig(settings, "Persisted session config", "INVALID_PERSISTED_SETTINGS");
  } catch (error) {
    if (
      error instanceof SessionCoordinatorError &&
      error.code === "INVALID_PERSISTED_SETTINGS"
    ) {
      throw error;
    }
    throw new SessionCoordinatorError(
      "INVALID_PERSISTED_SETTINGS",
      error instanceof Error ? error.message : String(error),
    );
  }
  return structuredClone(settings) as SessionInfo["config"];
}

function supportedConfigPatch(
  config: SessionInfo["config"],
  maxOutputTokens: number,
): Partial<SessionInfo["config"]> {
  const patch: Partial<SessionInfo["config"]> = { top_k: config.top_k };
  if (config.temperature !== null) patch.temperature = config.temperature;
  if (config.top_p !== null) patch.top_p = config.top_p;
  if (config.max_tokens !== null) {
    patch.max_tokens = clampOutputTokenCount(config.max_tokens, maxOutputTokens);
  }
  if (config.system_prompt !== null) patch.system_prompt = config.system_prompt;
  if (config.thinking !== null) patch.thinking = config.thinking;
  return patch;
}

function validateConfig(
  value: unknown,
  label: string,
  code: "INVALID_RUNTIME_RESPONSE" | "INVALID_PERSISTED_SETTINGS",
): asserts value is SessionInfo["config"] {
  const config = plainObject(value, label, code);
  exactKeys(config, CONFIG_KEYS, label, code);
  nullableFinite(config.temperature, `${label} temperature`, code);
  nullableFinite(config.top_p, `${label} top-p`, code);
  nullableInteger(config.top_k, `${label} top-k`, code);
  nullablePositiveInteger(config.max_tokens, `${label} max tokens`, code);
  nullableString(config.system_prompt, `${label} system prompt`, code);
  if (config.thinking !== null && typeof config.thinking !== "boolean") {
    throw coordinatorError(code, `${label} thinking value is invalid`);
  }
}

function validateSessionInfo(value: unknown): SessionInfo {
  const session = plainObject(value, "Session response");
  exactKeys(session, SESSION_KEYS, "Session response");
  nonEmptyString(session.id, "Session id");
  nonEmptyString(session.model_id, "Session model id");
  nonEmptyString(session.device, "Session device");
  nonEmptyString(session.dtype, "Session dtype");
  finiteTimestamp(session.created, "Session creation timestamp");
  validateConfig(session.config, "Session config", "INVALID_RUNTIME_RESPONSE");
  stringArray(session.profiles, "Session profiles");
  stringArray(session.probes, "Session probes");
  nonNegativeInteger(session.history_length, "Session history length");
  for (const key of [
    "supports_thinking",
    "thinking_is_optional",
    "is_base_model",
    "jlens_fitted",
    "role_substitution_supported",
    "user_role_supported",
    "scene_mode",
    "thinking_input_supported",
    "strips_history_thinking",
  ] as const) {
    if (typeof session[key] !== "boolean") {
      throw invalidResponse(`Session ${key} is invalid`);
    }
  }
  nullableString(session.default_steering, "Session default steering");
  nullableString(session.default_assistant_role, "Session default assistant role");
  nullableString(session.default_user_role, "Session default user role");
  if (!Array.isArray(session.instruments)) {
    throw invalidResponse("Session instruments are invalid");
  }
  for (const instrument of session.instruments) validateInstrument(instrument);
  return structuredClone(session) as unknown as SessionInfo;
}

function validateInstrument(value: unknown): void {
  const instrument = plainObject(value, "Session instrument");
  exactKeys(
    instrument,
    ["capabilities", "family", "live", "probes", "source"],
    "Session instrument",
  );
  nonEmptyString(instrument.family, "Instrument family");
  nullableString(instrument.source, "Instrument source");
  stringArray(instrument.probes, "Instrument probes");
  const capabilities = plainObject(instrument.capabilities, "Instrument capabilities");
  exactKeys(
    capabilities,
    ["preparations", "source_switch", "sources", "token_readout"],
    "Instrument capabilities",
  );
  for (const key of ["source_switch", "sources", "token_readout"] as const) {
    if (typeof capabilities[key] !== "boolean") {
      throw invalidResponse(`Instrument capability ${key} is invalid`);
    }
  }
  stringArray(capabilities.preparations, "Instrument preparations");
  jsonValue(instrument.live, "Instrument live state", new Set());
}

function validateRestoreResponse(value: unknown, tree: LoomTreeJSON): number {
  const response = plainObject(value, "Tree restore response");
  exactKeys(
    response,
    ["active_node_id", "nodes", "rev", "root_id"],
    "Tree restore response",
  );
  nonNegativeInteger(response.rev, "Restored tree revision");
  nonEmptyString(response.root_id, "Restored root node id");
  nonEmptyString(response.active_node_id, "Restored active node id");
  nonNegativeInteger(response.nodes, "Restored node count");
  if (
    Number(response.rev) <= tree.rev ||
    response.root_id !== tree.root_id ||
    response.active_node_id !== tree.active_node_id ||
    response.nodes !== tree.nodes.length
  ) {
    throw invalidResponse("Tree restore response does not match the authoritative snapshot");
  }
  return Number(response.rev);
}

function validateTreeSnapshot(
  value: unknown,
  session: SessionInfo,
  identity: HostedSessionRuntimeIdentity,
): LoomTreeJSON {
  const tree = plainObject(value, "Loom tree") as unknown as LoomTreeJSON;
  const candidate = {
    modelVariantId: identity.modelVariantId,
    metadata: {
      sessionId: session.id,
      modelId: session.model_id,
      runtimeIdentitySha256: identity.runtimeIdentitySha256,
      contextTokens: identity.contextTokens,
      createdAt: session.created,
    },
    settings: configSettings(session.config),
    tree,
    schemaVersion: SESSION_SCHEMA_VERSION,
    ownerEpoch: identity.ownerEpoch,
    updatedAt: Date.now(),
  };
  validatePersistedHostedSession(candidate);
  return structuredClone(tree);
}

function validateBackendIdentity(session: SessionInfo, expected: SessionInfo): void {
  if (
    session.id !== expected.id ||
    session.model_id !== expected.model_id ||
    session.created !== expected.created
  ) {
    throw new SessionCoordinatorError(
      "RUNTIME_IDENTITY_MISMATCH",
      "The model backend changed session identity while restoring settings",
    );
  }
}

function plainObject(
  value: unknown,
  label: string,
  code: "INVALID_RUNTIME_RESPONSE" | "INVALID_PERSISTED_SETTINGS" = "INVALID_RUNTIME_RESPONSE",
): Record<string, unknown> {
  if (
    value === null ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null)
  ) {
    throw coordinatorError(code, `${label} is invalid`);
  }
  return value as Record<string, unknown>;
}

function exactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  label: string,
  code: "INVALID_RUNTIME_RESPONSE" | "INVALID_PERSISTED_SETTINGS" = "INVALID_RUNTIME_RESPONSE",
): void {
  const keys = Object.keys(value).sort();
  const target = [...expected].sort();
  if (keys.length !== target.length || keys.some((key, index) => key !== target[index])) {
    throw coordinatorError(code, `${label} fields are invalid`);
  }
}

function validateIdentifier(value: unknown, label: string, maximum: number): void {
  if (
    typeof value !== "string" ||
    value.length === 0 ||
    value.length > maximum ||
    /\p{Cc}/u.test(value)
  ) {
    throw new SessionCoordinatorError(
      "INVALID_RUNTIME_IDENTITY",
      `${label} is invalid`,
    );
  }
}

function nonEmptyString(value: unknown, label: string): asserts value is string {
  if (typeof value !== "string" || value.length === 0) {
    throw invalidResponse(`${label} is invalid`);
  }
}

function nullableString(
  value: unknown,
  label: string,
  code: "INVALID_RUNTIME_RESPONSE" | "INVALID_PERSISTED_SETTINGS" = "INVALID_RUNTIME_RESPONSE",
): void {
  if (value !== null && typeof value !== "string") {
    throw coordinatorError(code, `${label} is invalid`);
  }
}

function finiteTimestamp(value: unknown, label: string): void {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw invalidResponse(`${label} is invalid`);
  }
}

function nullableFinite(
  value: unknown,
  label: string,
  code: "INVALID_RUNTIME_RESPONSE" | "INVALID_PERSISTED_SETTINGS",
): void {
  if (value !== null && (typeof value !== "number" || !Number.isFinite(value))) {
    throw coordinatorError(code, `${label} is invalid`);
  }
}

function nullableInteger(
  value: unknown,
  label: string,
  code: "INVALID_RUNTIME_RESPONSE" | "INVALID_PERSISTED_SETTINGS",
): void {
  if (value !== null && !Number.isSafeInteger(value)) {
    throw coordinatorError(code, `${label} is invalid`);
  }
}

function nullablePositiveInteger(
  value: unknown,
  label: string,
  code: "INVALID_RUNTIME_RESPONSE" | "INVALID_PERSISTED_SETTINGS",
): void {
  if (value !== null && (!Number.isSafeInteger(value) || Number(value) < 1)) {
    throw coordinatorError(code, `${label} is invalid`);
  }
}

function nonNegativeInteger(value: unknown, label: string): void {
  if (!Number.isSafeInteger(value) || Number(value) < 0) {
    throw invalidResponse(`${label} is invalid`);
  }
}

function stringArray(value: unknown, label: string): void {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    throw invalidResponse(`${label} is invalid`);
  }
}

function jsonValue(value: unknown, label: string, seen: Set<object>): void {
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number") {
    if (Number.isFinite(value)) return;
    throw invalidResponse(`${label} contains a non-finite number`);
  }
  if (typeof value !== "object" || seen.has(value)) {
    throw invalidResponse(`${label} is not JSON`);
  }
  seen.add(value);
  if (Array.isArray(value)) {
    for (const item of value) jsonValue(item, label, seen);
    return;
  }
  const object = plainObject(value, label);
  for (const item of Object.values(object)) jsonValue(item, label, seen);
}

function coordinatorError(
  code: "INVALID_RUNTIME_RESPONSE" | "INVALID_PERSISTED_SETTINGS",
  message: string,
): SessionCoordinatorError {
  return new SessionCoordinatorError(code, message);
}

function invalidResponse(message: string): SessionCoordinatorError {
  return coordinatorError("INVALID_RUNTIME_RESPONSE", message);
}
