import type {
  CastMemberJSON,
  LoomNodeJSON,
  LoomTreeJSON,
  RecipeJSON,
  RecipeSamplingJSON,
} from "../../lib/types";
import { migrateLegacyDatabase } from "./brandMigration";
import { isRoleSlug } from "./roleSlug";

const DATABASE_NAME = "drowse-hosted-sessions";
const DATABASE_VERSION = 2;
const SESSIONS_STORE = "sessions";
const SESSION_LOCK = "drowse-hosted-session-state-v1";
export const SESSION_SCHEMA_VERSION = 2 as const;
const MAX_RECORD_BYTES = 64 * 1024 * 1024;
const MAX_SETTINGS_BYTES = 512 * 1024;
const MAX_NODES = 8_192;
const MAX_TOKEN_ROWS = 262_144;
const SHA256 = /^[a-f0-9]{64}$/;

export type HostedJsonValue =
  | null
  | boolean
  | number
  | string
  | HostedJsonValue[]
  | { [key: string]: HostedJsonValue };

export interface HostedSessionSettings {
  temperature: number | null;
  top_p: number | null;
  top_k: number | null;
  max_tokens: number | null;
  system_prompt: string | null;
  thinking: boolean | null;
}

export interface HostedSessionBinding {
  modelVariantId: string;
  runtimeIdentitySha256: string;
  contextTokens: number;
  ownerEpoch: string;
}

export interface HostedSessionMetadata {
  sessionId: string;
  modelId: string;
  runtimeIdentitySha256: string;
  contextTokens: number;
  createdAt: number;
}

export interface HostedSessionSave {
  modelVariantId: string;
  metadata: HostedSessionMetadata;
  settings: HostedSessionSettings;
  tree: LoomTreeJSON;
}

export interface PersistedHostedSession extends HostedSessionSave {
  schemaVersion: typeof SESSION_SCHEMA_VERSION;
  ownerEpoch: string;
  updatedAt: number;
}

export type HostedSessionClaimResult =
  | { status: "missing" }
  | { status: "compatible"; record: PersistedHostedSession }
  | {
      status: "incompatible";
      reason: "runtime_identity" | "context_tokens";
      record: PersistedHostedSession;
    }
  | { status: "unsupported"; schemaVersion: number | null };

export interface HostedSessionStateStore {
  initialize(): Promise<void>;
  read(modelVariantId: string): Promise<unknown | undefined>;
  write(record: PersistedHostedSession): Promise<void>;
  delete(modelVariantId: string): Promise<boolean>;
  clear(): Promise<void>;
  close?(): void;
}

export type SessionExclusiveRunner = <T>(operation: () => Promise<T>) => Promise<T>;

export interface BrowserSessionPersistenceOptions {
  store?: HostedSessionStateStore;
  now?: () => number;
  runExclusive?: SessionExclusiveRunner;
  onCorrupt?: (modelVariantId: string, error: Error) => void;
}

export class SessionPersistenceError extends Error {
  constructor(
    readonly code:
      | "INVALID_SESSION_STATE"
      | "STALE_TREE_REVISION"
      | "STALE_SESSION_OWNER"
      | "UNSUPPORTED_SESSION_SCHEMA"
      | "INDEXEDDB_UNAVAILABLE"
      | "INDEXEDDB_BLOCKED",
    message: string,
  ) {
    super(message);
    this.name = "SessionPersistenceError";
  }
}

let fallbackLockTail: Promise<void> = Promise.resolve();

const fallbackExclusiveRunner: SessionExclusiveRunner = async <T>(
  operation: () => Promise<T>,
): Promise<T> => {
  const predecessor = fallbackLockTail;
  let release!: () => void;
  fallbackLockTail = new Promise<void>((resolve) => {
    release = resolve;
  });
  await predecessor;
  try {
    return await operation();
  } finally {
    release();
  }
};

const defaultExclusiveRunner: SessionExclusiveRunner = <T>(
  operation: () => Promise<T>,
): Promise<T> => {
  if (typeof navigator !== "undefined" && navigator.locks) {
    return navigator.locks.request(
      SESSION_LOCK,
      { mode: "exclusive" },
      operation,
    );
  }
  return fallbackExclusiveRunner(operation);
};

export class BrowserSessionPersistence {
  private readonly store: HostedSessionStateStore;
  private readonly now: () => number;
  private readonly runExclusive: SessionExclusiveRunner;
  private readonly onCorrupt: (modelVariantId: string, error: Error) => void;
  private initialization: Promise<void> | null = null;

  constructor(options: BrowserSessionPersistenceOptions = {}) {
    this.store = options.store ?? new BrowserSessionStateStore();
    this.now = options.now ?? Date.now;
    this.runExclusive = options.runExclusive ?? defaultExclusiveRunner;
    this.onCorrupt = options.onCorrupt ?? (() => undefined);
  }

  initialize(): Promise<void> {
    this.initialization ??= this.store.initialize().catch((error) => {
      this.initialization = null;
      throw error;
    });
    return this.initialization;
  }

  async claim(
    binding: HostedSessionBinding,
    mode: "restore" | "reset" = "restore",
    compatibleRuntimeIdentitySha256s: readonly string[] = [],
  ): Promise<HostedSessionClaimResult> {
    validateSessionBinding(binding);
    compatibleRuntimeIdentitySha256s.forEach((identity) => {
      if (!SHA256.test(identity)) throw invalid("Compatible session runtime identity is invalid");
    });
    await this.initialize();
    return this.runExclusive(async () => {
      const value = await this.store.read(binding.modelVariantId);
      if (value === undefined) return { status: "missing" };
      if (mode === "reset") {
        await this.store.delete(binding.modelVariantId);
        return { status: "missing" };
      }
      const schemaVersion = schemaVersionFrom(value);
      if (schemaVersion !== SESSION_SCHEMA_VERSION) {
        return { status: "unsupported", schemaVersion };
      }
      let record: PersistedHostedSession;
      try {
        validatePersistedHostedSession(value);
        if (value.modelVariantId !== binding.modelVariantId) {
          throw invalid("Stored model variant does not match its key");
        }
        record = value;
      } catch (error) {
        const normalized = normalizeError(error);
        await this.store.delete(binding.modelVariantId);
        this.onCorrupt(binding.modelVariantId, normalized);
        return { status: "missing" };
      }
      if (record.metadata.runtimeIdentitySha256 !== binding.runtimeIdentitySha256) {
        if (compatibleRuntimeIdentitySha256s.includes(record.metadata.runtimeIdentitySha256)) {
          if (record.metadata.contextTokens !== binding.contextTokens) {
            return { status: "incompatible", reason: "context_tokens", record: structuredClone(record) };
          }
          const claimed = structuredClone({
            ...record,
            metadata: {
              ...record.metadata,
              runtimeIdentitySha256: binding.runtimeIdentitySha256,
            },
            ownerEpoch: binding.ownerEpoch,
          });
          validatePersistedHostedSession(claimed);
          await this.store.write(claimed);
          return { status: "compatible", record: claimed };
        }
        return {
          status: "incompatible",
          reason: "runtime_identity",
          record: structuredClone(record),
        };
      }
      if (record.metadata.contextTokens !== binding.contextTokens) {
        return {
          status: "incompatible",
          reason: "context_tokens",
          record: structuredClone(record),
        };
      }
      const claimed = structuredClone({
        ...record,
        ownerEpoch: binding.ownerEpoch,
      });
      await this.store.write(claimed);
      return { status: "compatible", record: claimed };
    });
  }

  async save(
    input: HostedSessionSave,
    ownerEpoch: string,
  ): Promise<PersistedHostedSession> {
    validateIdentifier(ownerEpoch, "session owner epoch", 256);
    const updatedAt = this.now();
    const record: PersistedHostedSession = structuredClone({
      ...input,
      schemaVersion: SESSION_SCHEMA_VERSION,
      ownerEpoch,
      updatedAt,
    });
    validatePersistedHostedSession(record);
    await this.initialize();
    return this.runExclusive(async () => {
      const previousValue = await this.store.read(record.modelVariantId);
      if (previousValue !== undefined) {
        const previousSchema = schemaVersionFrom(previousValue);
        if (previousSchema !== SESSION_SCHEMA_VERSION) {
          throw new SessionPersistenceError(
            "UNSUPPORTED_SESSION_SCHEMA",
            "Refusing to overwrite a session written by an unsupported schema",
          );
        }
        let previous: PersistedHostedSession | null = null;
        try {
          validatePersistedHostedSession(previousValue);
          previous = previousValue;
        } catch (error) {
          await this.store.delete(record.modelVariantId);
          this.onCorrupt(record.modelVariantId, normalizeError(error));
        }
        if (
          previous !== null &&
          previous.ownerEpoch !== ownerEpoch
        ) {
          throw new SessionPersistenceError(
            "STALE_SESSION_OWNER",
            "Refusing a session write from a tab that no longer owns the runtime",
          );
        }
        if (
          previous !== null &&
          previous.tree.rev > record.tree.rev
        ) {
          throw new SessionPersistenceError(
            "STALE_TREE_REVISION",
            `Refusing to replace loom revision ${previous.tree.rev} with ${record.tree.rev}`,
          );
        }
      }
      await this.store.write(record);
      return structuredClone(record);
    });
  }

  async delete(modelVariantId: string): Promise<boolean> {
    validateIdentifier(modelVariantId, "model variant", 256);
    await this.initialize();
    return this.runExclusive(() => this.store.delete(modelVariantId));
  }

  async clear(): Promise<void> {
    await this.initialize();
    await this.runExclusive(() => this.store.clear());
  }

  close(): void {
    this.store.close?.();
    this.initialization = null;
  }
}

export class BrowserSessionStateStore implements HostedSessionStateStore {
  private database: IDBDatabase | null = null;
  private initialization: Promise<void> | null = null;

  initialize(): Promise<void> {
    this.initialization ??= this.initializeOnce().catch((error) => {
      this.initialization = null;
      throw error;
    });
    return this.initialization;
  }

  async read(modelVariantId: string): Promise<unknown | undefined> {
    return this.withDatabase(async (database) => {
      const transaction = database.transaction(SESSIONS_STORE, "readonly");
      const done = transactionDone(transaction);
      const value = await requestValue<unknown | undefined>(
        transaction.objectStore(SESSIONS_STORE).get(modelVariantId),
      );
      await done;
      return value;
    });
  }

  async write(record: PersistedHostedSession): Promise<void> {
    await this.withDatabase(async (database) => {
      const transaction = database.transaction(SESSIONS_STORE, "readwrite");
      const done = transactionDone(transaction);
      transaction.objectStore(SESSIONS_STORE).put(record);
      await done;
    });
  }

  async delete(modelVariantId: string): Promise<boolean> {
    const present = await this.read(modelVariantId) !== undefined;
    if (!present) return false;
    await this.withDatabase(async (database) => {
      const transaction = database.transaction(SESSIONS_STORE, "readwrite");
      const done = transactionDone(transaction);
      transaction.objectStore(SESSIONS_STORE).delete(modelVariantId);
      await done;
    });
    return true;
  }

  async clear(): Promise<void> {
    await this.withDatabase(async (database) => {
      const transaction = database.transaction(SESSIONS_STORE, "readwrite");
      const done = transactionDone(transaction);
      transaction.objectStore(SESSIONS_STORE).clear();
      await done;
    });
  }

  close(): void {
    this.invalidate(this.database);
  }

  private async initializeOnce(): Promise<void> {
    if (typeof indexedDB === "undefined") {
      throw new SessionPersistenceError(
        "INDEXEDDB_UNAVAILABLE",
        "IndexedDB is unavailable",
      );
    }
    const database = await openDatabase();
    this.database = database;
    database.onversionchange = () => this.invalidate(database);
    database.onclose = () => this.invalidate(database, false);
  }

  private async withDatabase<T>(
    operation: (database: IDBDatabase) => Promise<T>,
  ): Promise<T> {
    for (let attempt = 0; attempt < 2; attempt += 1) {
      await this.initialize();
      const database = this.database;
      if (database === null) {
        this.initialization = null;
        continue;
      }
      try {
        return await operation(database);
      } catch (error) {
        if (attempt === 0 && isClosedDatabaseError(error)) {
          this.invalidate(database);
          continue;
        }
        throw error;
      }
    }
    throw new SessionPersistenceError(
      "INDEXEDDB_UNAVAILABLE",
      "IndexedDB closed while accessing session state",
    );
  }

  private invalidate(database: IDBDatabase | null, close = true): void {
    if (database === null) return;
    if (close) database.close();
    if (this.database === database) {
      this.database = null;
      this.initialization = null;
    }
  }
}

export function validatePersistedHostedSession(
  value: unknown,
): asserts value is PersistedHostedSession {
  assertPlainObject(value, "Session record");
  assertExactKeys(value, [
    "metadata",
    "modelVariantId",
    "ownerEpoch",
    "schemaVersion",
    "settings",
    "tree",
    "updatedAt",
  ], "Session record");
  if (value.schemaVersion !== SESSION_SCHEMA_VERSION) {
    throw invalid("Session schema version is unsupported");
  }
  validateIdentifier(value.ownerEpoch, "session owner epoch", 256);
  validateIdentifier(value.modelVariantId, "model variant", 256);
  validateSessionMetadata(value.metadata);
  validateSettings(value.settings);
  validateTree(value.tree);
  validateTimestamp(value.updatedAt, "updated timestamp");
  if (value.tree.model_id !== value.metadata.modelId) {
    throw invalid("Loom model does not match session metadata");
  }
  if (value.tree.session_id !== value.metadata.sessionId) {
    throw invalid("Loom session does not match session metadata");
  }
  assertJsonValue(value, {
    maxDepth: 48,
    maxValues: 2_000_000,
    maxStringLength: 4 * 1024 * 1024,
    maxArrayLength: MAX_TOKEN_ROWS,
  });
  if (jsonByteLength(value) > MAX_RECORD_BYTES) {
    throw invalid("Session record exceeds the 64 MiB limit");
  }
}

export function validateHostedLoomTree(
  value: unknown,
): asserts value is LoomTreeJSON {
  validateTree(value);
}

function validateSessionMetadata(value: unknown): asserts value is HostedSessionMetadata {
  assertPlainObject(value, "Session metadata");
  assertExactKeys(value, [
    "contextTokens",
    "createdAt",
    "modelId",
    "runtimeIdentitySha256",
    "sessionId",
  ], "Session metadata");
  validateIdentifier(value.sessionId, "session id", 256);
  validateIdentifier(value.modelId, "model id", 512);
  if (typeof value.runtimeIdentitySha256 !== "string" || !SHA256.test(value.runtimeIdentitySha256)) {
    throw invalid("Session runtime identity is invalid");
  }
  if (
    typeof value.contextTokens !== "number" ||
    !Number.isSafeInteger(value.contextTokens) ||
    value.contextTokens < 1 ||
    value.contextTokens > 1_048_576
  ) {
    throw invalid("Session context size is invalid");
  }
  validateTimestamp(value.createdAt, "creation timestamp");
}

function validateSettings(value: unknown): asserts value is HostedSessionSettings {
  assertPlainObject(value, "Session settings");
  assertExactKeys(value, [
    "max_tokens",
    "system_prompt",
    "temperature",
    "thinking",
    "top_k",
    "top_p",
  ], "Session settings");
  validateNullableFinite(value.temperature, "session temperature");
  validateNullableFinite(value.top_p, "session top-p");
  validateNullableInteger(value.top_k, "session top-k");
  validateNullablePositiveInteger(value.max_tokens, "session max tokens");
  validateNullableString(value.system_prompt, "session system prompt", 4 * 1024 * 1024);
  if (value.thinking !== null && typeof value.thinking !== "boolean") {
    throw invalid("Session thinking setting is invalid");
  }
  if (jsonByteLength(value) > MAX_SETTINGS_BYTES) {
    throw invalid("Session settings exceed the 512 KiB limit");
  }
}

function validateTree(value: unknown): asserts value is LoomTreeJSON {
  assertPlainObject(value, "Loom tree");
  assertExactKeys(value, [
    "active_node_id",
    "cast",
    "children_of",
    "model_id",
    "name",
    "nodes",
    "rev",
    "root_id",
    "drowse_version",
    "session_id",
    "tree_format",
  ], "Loom tree");
  if (value.tree_format !== 2) throw invalid("Loom tree format is unsupported");
  validateString(value.drowse_version, "Drowse version", 128, false);
  validateNullableString(value.model_id, "Loom model id", 512);
  validateNullableString(value.session_id, "Loom session id", 256);
  validateNullableString(value.name, "Loom name", 256);
  validateNonNegativeInteger(value.rev, "Loom revision");
  validateIdentifier(value.root_id, "root node id", 256);
  validateIdentifier(value.active_node_id, "active node id", 256);
  if (!Array.isArray(value.nodes) || value.nodes.length < 1 || value.nodes.length > MAX_NODES) {
    throw invalid(`Loom tree must contain between 1 and ${MAX_NODES} nodes`);
  }
  assertPlainObject(value.children_of, "Loom children map");
  assertPlainObject(value.cast, "Loom cast");
  if (Object.keys(value.cast).length > 128) throw invalid("Loom cast is too large");
  for (const [label, member] of Object.entries(value.cast)) {
    if (!isRoleSlug(label)) throw invalid("Loom cast label must be a lowercase role slug");
    validateCastMember(member);
  }

  const nodes = new Map<string, LoomNodeJSON>();
  let tokenRows = 0;
  for (const node of value.nodes) {
    validateNode(node);
    if (nodes.has(node.id)) throw invalid(`Duplicate loom node id: ${node.id}`);
    nodes.set(node.id, node);
    tokenRows += (node.tokens?.length ?? 0) + (node.thinking_tokens?.length ?? 0);
    if (tokenRows > MAX_TOKEN_ROWS) throw invalid("Loom tree contains too many token rows");
  }
  const root = nodes.get(value.root_id);
  if (root === undefined) throw invalid("Loom root node does not exist");
  if (root.parent_id !== null) throw invalid("Loom root node must not have a parent");
  if (root.role !== "system") throw invalid("Loom root node must have the system role");
  if (!nodes.has(value.active_node_id)) throw invalid("Loom active node does not exist");

  const childKeys = Object.keys(value.children_of);
  if (childKeys.length !== nodes.size || childKeys.some((id) => !nodes.has(id))) {
    throw invalid("Loom children map must contain exactly one row per node");
  }
  const observedChildren = new Set<string>();
  const childrenByParent = new Map<string, string[]>();
  for (const [parentId, children] of Object.entries(value.children_of)) {
    if (!Array.isArray(children) || children.length > MAX_NODES) {
      throw invalid(`Loom children row is invalid: ${parentId}`);
    }
    const local = new Set<string>();
    for (const childId of children) {
      validateIdentifier(childId, "child node id", 256);
      const child = nodes.get(childId);
      if (child === undefined || child.parent_id !== parentId) {
        throw invalid(`Loom parent relationship is inconsistent: ${childId}`);
      }
      if (local.has(childId) || observedChildren.has(childId)) {
        throw invalid(`Duplicate loom child relationship: ${childId}`);
      }
      local.add(childId);
      observedChildren.add(childId);
    }
    childrenByParent.set(parentId, children);
  }
  for (const node of nodes.values()) {
    if (node.id === value.root_id) continue;
    if (node.parent_id === null || !nodes.has(node.parent_id)) {
      throw invalid(`Loom node has an invalid parent: ${node.id}`);
    }
    if (!observedChildren.has(node.id)) {
      throw invalid(`Loom node is absent from its parent's children row: ${node.id}`);
    }
  }
  if (observedChildren.size !== nodes.size - 1) {
    throw invalid("Loom tree is disconnected or cyclic");
  }
  const visited = new Set<string>();
  const pending = [value.root_id];
  while (pending.length > 0) {
    const id = pending.pop()!;
    if (visited.has(id)) throw invalid("Loom tree contains a cycle");
    visited.add(id);
    pending.push(...(childrenByParent.get(id) ?? []));
  }
  if (visited.size !== nodes.size) throw invalid("Loom tree is disconnected");
}

function validateNode(value: unknown): asserts value is LoomNodeJSON {
  assertPlainObject(value, "Loom node");
  assertExactKeys(value, [
    "aggregate_readings",
    "applied_steering",
    "created_at",
    "edit_count",
    "edited_at",
    "finish_reason",
    "id",
    "mean_logprob",
    "mean_surprise",
    "notes",
    "parent_id",
    "raw_token_ids",
    "recipe",
    "role",
    "role_label",
    "starred",
    "text",
    "thinking_text",
    "thinking_tokens",
    "tokens",
  ], "Loom node");
  validateIdentifier(value.id, "node id", 256);
  validateNullableString(value.parent_id, "parent node id", 256);
  if (!["user", "assistant", "system"].includes(String(value.role))) {
    throw invalid("Loom node role is invalid");
  }
  validateString(value.text, "node text", 4 * 1024 * 1024, true);
  validateNullableRoleSlug(value.role_label, "node role label");
  validateNullableString(value.thinking_text, "node thinking text", 4 * 1024 * 1024);
  validateFiniteMap(value.aggregate_readings, "node aggregate readings", 4_096);
  validateNullableString(value.applied_steering, "node steering expression", 64 * 1024);
  validateNullableString(value.finish_reason, "node finish reason", 1_024);
  if (typeof value.starred !== "boolean") throw invalid("Node starred flag is invalid");
  validateString(value.notes, "node notes", 256 * 1024, true);
  validateTimestamp(value.created_at, "node creation timestamp");
  validateNullableTimestamp(value.edited_at, "node edit timestamp");
  validateNonNegativeInteger(value.edit_count, "node edit count");
  validateNullableFinite(value.mean_logprob, "node mean log probability");
  validateNullableFinite(value.mean_surprise, "node mean surprise");
  if (value.recipe !== null) validateRecipe(value.recipe);
  validateTokenRows(value.tokens, "node tokens");
  validateTokenRows(value.thinking_tokens, "node thinking tokens");
  if (value.raw_token_ids !== null) {
    if (!Array.isArray(value.raw_token_ids) || value.raw_token_ids.length > MAX_TOKEN_ROWS) {
      throw invalid("Node raw token ids are invalid");
    }
    for (const tokenId of value.raw_token_ids) {
      validateNonNegativeInteger(tokenId, "raw token id");
    }
  }
}

function validateCastMember(value: unknown): asserts value is CastMemberJSON {
  assertPlainObject(value, "Cast member");
  assertKeys(value, ["notes", "recipe"], ["origin"], "Cast member");
  validateString(value.notes, "cast member notes", 256 * 1024, true);
  if (value.recipe !== null) validateRecipe(value.recipe);
  if (
    value.origin !== undefined &&
    !["configured", "observed", "structural"].includes(String(value.origin))
  ) {
    throw invalid("Cast member origin is invalid");
  }
}

function validateRecipe(value: unknown): asserts value is RecipeJSON {
  assertPlainObject(value, "Recipe");
  assertKeys(value, [
    "probe_hashes",
    "probes",
    "sampling",
    "seed",
    "steering",
    "thinking",
  ], ["system_prompt"], "Recipe");
  if (value.system_prompt !== undefined) validateNullableString(value.system_prompt, "recipe system prompt", 1024 * 1024);
  validateNullableString(value.steering, "recipe steering", 64 * 1024);
  if (value.sampling !== null) validateRecipeSampling(value.sampling);
  if (value.thinking !== null && typeof value.thinking !== "boolean") {
    throw invalid("Recipe thinking value is invalid");
  }
  validateNullableInteger(value.seed, "recipe seed");
  if (!Array.isArray(value.probes) || value.probes.length > 4_096) {
    throw invalid("Recipe probes are invalid");
  }
  for (const probe of value.probes) validateIdentifier(probe, "recipe probe", 1_024);
  assertPlainObject(value.probe_hashes, "Recipe probe hashes");
  if (Object.keys(value.probe_hashes).length > 4_096) {
    throw invalid("Recipe probe hashes are too large");
  }
  for (const [probe, hash] of Object.entries(value.probe_hashes)) {
    validateIdentifier(probe, "recipe probe hash key", 1_024);
    validateString(hash, "recipe probe hash", 512, false);
  }
}

function validateRecipeSampling(value: unknown): asserts value is RecipeSamplingJSON {
  assertPlainObject(value, "Recipe sampling");
  assertExactKeys(value, [
    "assistant_role",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_tokens",
    "persist_per_layer_scores",
    "persist_subspace_coords",
    "presence_penalty",
    "return_hidden",
    "return_probe_readings",
    "return_top_k",
    "seed",
    "stop",
    "temperature",
    "top_k",
    "top_p",
    "user_role",
  ], "Recipe sampling");
  validateNullableFinite(value.temperature, "sampling temperature");
  validateNullableFinite(value.top_p, "sampling top-p");
  validateNullableInteger(value.top_k, "sampling top-k");
  validateNullableInteger(value.max_tokens, "sampling max tokens");
  validateNullableInteger(value.seed, "sampling seed");
  validateNullableInteger(value.logprobs, "sampling logprobs");
  validateFinite(value.presence_penalty, "sampling presence penalty");
  validateFinite(value.frequency_penalty, "sampling frequency penalty");
  validateNonNegativeInteger(value.return_top_k, "sampling return top-k");
  for (const key of [
    "persist_per_layer_scores",
    "persist_subspace_coords",
    "return_hidden",
    "return_probe_readings",
  ] as const) {
    if (typeof value[key] !== "boolean") throw invalid(`Sampling ${key} is invalid`);
  }
  validateNullableRoleSlug(value.user_role, "sampling user role");
  validateNullableRoleSlug(value.assistant_role, "sampling assistant role");
  if (value.stop !== null) {
    if (!Array.isArray(value.stop) || value.stop.length > 1_024) {
      throw invalid("Sampling stop list is invalid");
    }
    for (const stop of value.stop) validateString(stop, "sampling stop", 64 * 1024, true);
  }
  if (value.logit_bias !== null) {
    assertPlainObject(value.logit_bias, "Sampling logit bias");
    if (Object.keys(value.logit_bias).length > 65_536) {
      throw invalid("Sampling logit bias is too large");
    }
    for (const [tokenId, bias] of Object.entries(value.logit_bias)) {
      if (!/^-?\d+$/.test(tokenId)) throw invalid("Sampling logit bias token id is invalid");
      validateFinite(bias, "sampling logit bias");
    }
  }
}

function validateTokenRows(value: unknown, label: string): void {
  if (value === null) return;
  if (!Array.isArray(value) || value.length > MAX_TOKEN_ROWS) {
    throw invalid(`${label} are invalid`);
  }
  for (const row of value) {
    assertPlainObject(row, "Loom token row");
    if (row.token_id !== undefined) validateNonNegativeInteger(row.token_id, "token id");
    if (row.text !== undefined) validateString(row.text, "token text", 64 * 1024, true);
    if (row.logprob !== undefined) validateNullableFinite(row.logprob, "token log probability");
    if (row.sampler_entropy !== undefined) {
      validateNullableFinite(row.sampler_entropy, "token sampler entropy");
      if (typeof row.sampler_entropy === "number" && row.sampler_entropy < 0) {
        throw invalid("Token sampler entropy is invalid");
      }
    }
    if (row.perplexity !== undefined) validateNullableFinite(row.perplexity, "token perplexity");
    if (row.raw_index !== undefined && row.raw_index !== null) {
      validateNonNegativeInteger(row.raw_index, "raw token index");
    }
    if (row.probes !== undefined) validateFiniteMap(row.probes, "token probes", 4_096);
    if (row.per_layer_scores !== undefined) {
      assertPlainObject(row.per_layer_scores, "token per-layer scores");
      if (Object.keys(row.per_layer_scores).length > 1_024) {
        throw invalid("Token per-layer score map is too large");
      }
      for (const scores of Object.values(row.per_layer_scores)) {
        validateFiniteMap(scores, "token layer scores", 4_096);
      }
    }
    if (row.top_alts !== undefined) {
      if (!Array.isArray(row.top_alts) || row.top_alts.length > MAX_TOKEN_ROWS) {
        throw invalid("Token alternatives are invalid");
      }
      for (const alternative of row.top_alts) {
        assertPlainObject(alternative, "Token alternative");
        assertExactKeys(alternative, ["id", "logprob", "text"], "Token alternative");
        validateNonNegativeInteger(alternative.id, "alternative token id");
        validateString(alternative.text, "alternative token text", 64 * 1024, true);
        validateFinite(alternative.logprob, "alternative log probability");
      }
    }
    if (row.measurements !== undefined) {
      assertPlainObject(row.measurements, "token measurement envelope");
    }
    const { top_alts: _alternatives, ...metadata } = row;
    assertJsonValue(metadata, {
      maxDepth: 24,
      maxValues: 131_072,
      maxStringLength: 4 * 1024 * 1024,
      maxArrayLength: MAX_TOKEN_ROWS,
    });
  }
}

interface JsonLimits {
  maxDepth: number;
  maxValues: number;
  maxStringLength: number;
  maxArrayLength: number;
}

function assertJsonValue(value: unknown, limits: JsonLimits): void {
  const seen = new WeakSet<object>();
  let values = 0;
  const walk = (entry: unknown, depth: number): void => {
    values += 1;
    if (values > limits.maxValues) throw invalid("Session state contains too many values");
    if (depth > limits.maxDepth) throw invalid("Session state is nested too deeply");
    if (entry === null || typeof entry === "boolean") return;
    if (typeof entry === "number") {
      if (!Number.isFinite(entry)) throw invalid("Session state contains a non-finite number");
      return;
    }
    if (typeof entry === "string") {
      if (entry.length > limits.maxStringLength) throw invalid("Session state string is too long");
      return;
    }
    if (typeof entry !== "object") throw invalid("Session state must contain JSON values only");
    if (seen.has(entry)) throw invalid("Session state must not contain repeated object references");
    seen.add(entry);
    if (Array.isArray(entry)) {
      if (entry.length > limits.maxArrayLength) throw invalid("Session state array is too long");
      for (const item of entry) walk(item, depth + 1);
      return;
    }
    assertPlainObject(entry, "Session state value");
    const keys = Object.keys(entry);
    if (keys.length > 16_384) throw invalid("Session state object has too many fields");
    for (const [key, item] of Object.entries(entry)) {
      if (key.length > 1_024 || key.includes("\0")) throw invalid("Session state key is invalid");
      walk(item, depth + 1);
    }
  };
  walk(value, 0);
}

function validateFiniteMap(value: unknown, label: string, maximum: number): void {
  assertPlainObject(value, label);
  const entries = Object.entries(value);
  if (entries.length > maximum) throw invalid(`${label} is too large`);
  for (const [key, item] of entries) {
    if (key.length > 1_024 || key.includes("\0")) throw invalid(`${label} key is invalid`);
    validateFinite(item, label);
  }
}

function assertPlainObject(
  value: unknown,
  label: string,
): asserts value is Record<string, unknown> {
  if (
    value === null ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null)
  ) {
    throw invalid(`${label} is invalid`);
  }
}

function assertExactKeys(
  value: Record<string, unknown>,
  expected: string[],
  label: string,
): void {
  assertKeys(value, expected, [], label);
}

function assertKeys(
  value: Record<string, unknown>,
  required: string[],
  optional: string[],
  label: string,
): void {
  const keys = Object.keys(value).sort();
  const allowedKeys = new Set([...required, ...optional]);
  if (required.some((key) => !keys.includes(key)) || keys.some((key) => !allowedKeys.has(key))) {
    throw invalid(`${label} fields are invalid`);
  }
}

function validateSessionBinding(value: HostedSessionBinding): void {
  assertPlainObject(value, "Session binding");
  assertExactKeys(value, [
    "contextTokens",
    "modelVariantId",
    "ownerEpoch",
    "runtimeIdentitySha256",
  ], "Session binding");
  validateIdentifier(value.modelVariantId, "model variant", 256);
  validateIdentifier(value.ownerEpoch, "session owner epoch", 256);
  if (!SHA256.test(value.runtimeIdentitySha256)) {
    throw invalid("Session runtime identity is invalid");
  }
  if (
    !Number.isSafeInteger(value.contextTokens) ||
    value.contextTokens < 1 ||
    value.contextTokens > 1_048_576
  ) {
    throw invalid("Session context size is invalid");
  }
}

function schemaVersionFrom(value: unknown): number | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const version = (value as Record<string, unknown>).schemaVersion;
  return Number.isSafeInteger(version) && Number(version) >= 0 ? Number(version) : null;
}

function validateIdentifier(value: unknown, label: string, maximum: number): asserts value is string {
  validateString(value, label, maximum, false);
  if (/\p{Cc}/u.test(value)) throw invalid(`${label} contains a control character`);
}

function validateString(
  value: unknown,
  label: string,
  maximum: number,
  allowEmpty: boolean,
): asserts value is string {
  if (
    typeof value !== "string" ||
    (!allowEmpty && value.length === 0) ||
    value.length > maximum
  ) {
    throw invalid(`${label} is invalid`);
  }
}

function validateNullableString(value: unknown, label: string, maximum: number): void {
  if (value !== null) validateString(value, label, maximum, true);
}

function validateNullableRoleSlug(value: unknown, label: string): void {
  if (value !== null && !isRoleSlug(value)) throw invalid(`${label} is not a lowercase role slug`);
}

function validateTimestamp(value: unknown, label: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw invalid(`${label} is invalid`);
  }
}

function validateNullableTimestamp(value: unknown, label: string): void {
  if (value !== null) validateTimestamp(value, label);
}

function validateFinite(value: unknown, label: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw invalid(`${label} is invalid`);
  }
}

function validateNullableFinite(value: unknown, label: string): void {
  if (value !== null) validateFinite(value, label);
}

function validateNullableInteger(value: unknown, label: string): void {
  if (value !== null && !Number.isSafeInteger(value)) throw invalid(`${label} is invalid`);
}

function validateNullablePositiveInteger(value: unknown, label: string): void {
  if (value !== null && (!Number.isSafeInteger(value) || Number(value) < 1)) {
    throw invalid(`${label} is invalid`);
  }
}

function validateNonNegativeInteger(value: unknown, label: string): asserts value is number {
  if (!Number.isSafeInteger(value) || Number(value) < 0) throw invalid(`${label} is invalid`);
}

function jsonByteLength(value: unknown): number {
  return new TextEncoder().encode(JSON.stringify(value)).byteLength;
}

function invalid(message: string): SessionPersistenceError {
  return new SessionPersistenceError("INVALID_SESSION_STATE", message);
}

function normalizeError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function isClosedDatabaseError(error: unknown): boolean {
  return error instanceof DOMException && [
    "AbortError",
    "InvalidStateError",
    "TransactionInactiveError",
  ].includes(error.name);
}

async function openDatabase(): Promise<IDBDatabase> {
  const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
  request.onupgradeneeded = () => {
    const database = request.result;
    if (!database.objectStoreNames.contains(SESSIONS_STORE)) {
      database.createObjectStore(SESSIONS_STORE, { keyPath: "modelVariantId" });
    }
  };
  const database = await new Promise<IDBDatabase>((resolve, reject) => {
    let blocked = false;
    request.onsuccess = () => {
      if (blocked) {
        request.result.close();
        return;
      }
      resolve(request.result);
    };
    request.onerror = () => reject(request.error ?? new Error("IndexedDB open failed"));
    request.onblocked = () => {
      blocked = true;
      reject(new SessionPersistenceError(
        "INDEXEDDB_BLOCKED",
        "Another Drowse tab is blocking the session database upgrade",
      ));
    };
  });
  await migrateLegacyDatabase(database, [SESSIONS_STORE]);
  return database;
}

function requestValue<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(
      transaction.error ?? new Error("IndexedDB transaction aborted"),
    );
    transaction.onerror = () => reject(
      transaction.error ?? new Error("IndexedDB transaction failed"),
    );
  });
}
