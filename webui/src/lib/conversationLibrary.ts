import {
  validateConversationSnapshot,
  type ConversationSnapshotV7,
} from "./conversationSnapshot";
import { migrateLegacyDatabase } from "../hosted/runtime/brandMigration";
import { isChatAccent, type ChatAccent } from "./chatAccent";

const DATABASE_NAME = "drowse-saved-conversations";
const DATABASE_VERSION = 1;
const CONVERSATIONS_STORE = "conversations";
const CONVERSATION_LOCK = "drowse-saved-conversations-v1";
const MAX_RECORD_BYTES = 64 * 1024 * 1024;
const MAX_NAME_LENGTH = 120;
const MAX_AVATAR_SEED_LENGTH = 256;

export const SAVED_CONVERSATION_SCHEMA_VERSION = 1 as const;

export interface SavedConversationRecord {
  schemaVersion: typeof SAVED_CONVERSATION_SCHEMA_VERSION;
  id: string;
  name: string;
  avatarSeed: string;
  accent?: ChatAccent;
  modelId: string;
  modelType?: "chat" | "base";
  createdAt: number;
  updatedAt: number;
  snapshot: ConversationSnapshotV7;
}

export type SavedConversationSummary = Omit<SavedConversationRecord, "snapshot"> & {
  messageCount: number;
  threadCount: number;
};

export interface SavedConversationIssue {
  id: string;
  name: string | null;
  reason: string;
}

export interface SavedConversationList<T = SavedConversationRecord> {
  conversations: T[];
  issues: SavedConversationIssue[];
}

export interface SavedConversationStoreRow {
  key: string;
  value: unknown;
}

export interface SavedConversationStore {
  initialize(): Promise<void>;
  list(): Promise<SavedConversationStoreRow[]>;
  map?<T>(project: (row: SavedConversationStoreRow) => T): Promise<T[]>;
  hasAny?(): Promise<boolean>;
  read(id: string): Promise<unknown | undefined>;
  write(record: SavedConversationRecord): Promise<void>;
  delete(id: string): Promise<boolean>;
  close?(): void;
}

export type ConversationExclusiveRunner = <T>(operation: () => Promise<T>) => Promise<T>;

export interface ConversationLibraryOptions {
  samplingKeys: readonly string[];
  store?: SavedConversationStore;
  now?: () => number;
  randomId?: () => string;
  runExclusive?: ConversationExclusiveRunner;
}

export class ConversationLibraryError extends Error {
  constructor(
    readonly code:
      | "INVALID_RECORD"
      | "NOT_FOUND"
      | "STORAGE_LIMIT"
      | "INDEXEDDB_UNAVAILABLE"
      | "INDEXEDDB_BLOCKED",
    message: string,
  ) {
    super(message);
    this.name = "ConversationLibraryError";
  }
}

let fallbackLockTail: Promise<void> = Promise.resolve();

const fallbackExclusiveRunner: ConversationExclusiveRunner = async <T>(
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

const defaultExclusiveRunner: ConversationExclusiveRunner = <T>(
  operation: () => Promise<T>,
): Promise<T> => {
  if (typeof navigator !== "undefined" && navigator.locks) {
    return navigator.locks.request(CONVERSATION_LOCK, { mode: "exclusive" }, operation);
  }
  return fallbackExclusiveRunner(operation);
};

export class ConversationLibrary {
  private readonly samplingKeys: readonly string[];
  private readonly store: SavedConversationStore;
  private readonly now: () => number;
  private readonly randomId: () => string;
  private readonly runExclusive: ConversationExclusiveRunner;
  private initialization: Promise<void> | null = null;

  constructor(options: ConversationLibraryOptions) {
    this.samplingKeys = [...options.samplingKeys];
    this.store = options.store ?? new BrowserSavedConversationStore();
    this.now = options.now ?? Date.now;
    this.randomId = options.randomId ?? randomUuid;
    this.runExclusive = options.runExclusive ?? defaultExclusiveRunner;
  }

  initialize(): Promise<void> {
    this.initialization ??= this.store.initialize().catch((error) => {
      this.initialization = null;
      throw error;
    });
    return this.initialization;
  }

  list(): Promise<SavedConversationList> {
    return this.readList((record) => structuredClone(record));
  }

  listSummaries(): Promise<SavedConversationList<SavedConversationSummary>> {
    return this.readList(summarizeConversation);
  }

  async findForTree(modelId: string, rootId: string): Promise<SavedConversationRecord | null> {
    const { conversations } = await this.readList((record) => ({
      id: record.id,
      name: record.name,
      updatedAt: record.updatedAt,
      modelId: record.modelId,
      rootId: record.snapshot.tree.root_id,
    }));
    const match = conversations.find((record) => record.modelId === modelId && record.rootId === rootId);
    return match ? this.get(match.id) : null;
  }

  async hasAny(): Promise<boolean> {
    await this.initialize();
    return this.store.hasAny ? this.store.hasAny() : (await this.store.list()).length > 0;
  }

  private async readList<T extends Pick<SavedConversationRecord, "id" | "updatedAt">>(
    project: (record: SavedConversationRecord) => T,
  ): Promise<SavedConversationList<T>> {
    await this.initialize();
    const readRow = (row: SavedConversationStoreRow): { conversation: T } | { issue: SavedConversationIssue } => {
      try {
        validateSavedConversationRecord(row.value, this.samplingKeys);
        if (row.value.id !== row.key) throw invalid("Saved conversation key does not match its id");
        return { conversation: project(row.value) };
      } catch (error) {
        const maybe = isPlainObject(row.value) ? row.value : null;
        return { issue: {
          id: row.key,
          name: typeof maybe?.name === "string" ? maybe.name : null,
          reason: normalizeError(error).message,
        } };
      }
    };
    const rows = this.store.map
      ? await this.store.map(readRow)
      : (await this.store.list()).map(readRow);
    const conversations: T[] = [];
    const issues: SavedConversationIssue[] = [];
    for (const row of rows) {
      if ("conversation" in row) conversations.push(row.conversation);
      else issues.push(row.issue);
    }
    conversations.sort((left, right) => right.updatedAt - left.updatedAt || left.id.localeCompare(right.id));
    return { conversations, issues };
  }

  async get(id: string): Promise<SavedConversationRecord> {
    validateIdentifier(id, "saved conversation id");
    await this.initialize();
    const value = await this.store.read(id);
    if (value === undefined) {
      throw new ConversationLibraryError("NOT_FOUND", "This saved conversation no longer exists");
    }
    validateSavedConversationRecord(value, this.samplingKeys);
    if (value.id !== id) throw invalid("Saved conversation key does not match its id");
    return structuredClone(value);
  }

  async create(input: {
    name: string;
    avatarSeed?: string;
    accent?: ChatAccent;
    modelType?: "chat" | "base";
    snapshot: ConversationSnapshotV7;
    createdAt?: number;
    updatedAt?: number;
  }): Promise<SavedConversationRecord> {
    const name = normalizeName(input.name);
    validateConversationSnapshot(input.snapshot, this.samplingKeys);
    const requestedAvatarSeed = input.avatarSeed === undefined
      ? null
      : normalizeAvatarSeed(input.avatarSeed);
    await this.initialize();
    return this.runExclusive(async () => {
      let id = this.randomId();
      for (let attempt = 0; attempt < 4 && await this.store.read(id) !== undefined; attempt += 1) {
        id = this.randomId();
      }
      if (await this.store.read(id) !== undefined) {
        throw invalid("Unable to allocate a unique saved conversation id");
      }
      const timestamp = this.now();
      const record: SavedConversationRecord = {
        schemaVersion: SAVED_CONVERSATION_SCHEMA_VERSION,
        id,
        name,
        avatarSeed: requestedAvatarSeed ?? normalizeAvatarSeed(this.randomId()),
        ...(input.accent === undefined ? {} : { accent: input.accent }),
        modelId: input.snapshot.model_id,
        ...(input.modelType === undefined ? {} : { modelType: input.modelType }),
        createdAt: input.createdAt ?? timestamp,
        updatedAt: input.updatedAt ?? timestamp,
        snapshot: structuredClone(input.snapshot),
      };
      validateSavedConversationRecord(record, this.samplingKeys);
      await this.store.write(record);
      return structuredClone(record);
    });
  }

  async update(
    id: string,
    changes: Partial<Pick<SavedConversationRecord, "name" | "avatarSeed" | "accent" | "snapshot">>,
  ): Promise<SavedConversationRecord> {
    validateIdentifier(id, "saved conversation id");
    await this.initialize();
    return this.runExclusive(async () => {
      const currentValue = await this.store.read(id);
      if (currentValue === undefined) {
        throw new ConversationLibraryError("NOT_FOUND", "This saved conversation no longer exists");
      }
      validateSavedConversationRecord(currentValue, this.samplingKeys);
      const snapshot = structuredClone(changes.snapshot ?? currentValue.snapshot);
      const record: SavedConversationRecord = {
        ...currentValue,
        ...(changes.accent === undefined ? {} : { accent: changes.accent }),
        name: changes.name === undefined ? currentValue.name : normalizeName(changes.name),
        avatarSeed: changes.avatarSeed === undefined
          ? currentValue.avatarSeed
          : normalizeAvatarSeed(changes.avatarSeed),
        modelId: snapshot.model_id,
        updatedAt: changes.snapshot === undefined ? currentValue.updatedAt : this.now(),
        snapshot,
      };
      validateSavedConversationRecord(record, this.samplingKeys);
      await this.store.write(record);
      return structuredClone(record);
    });
  }

  async duplicate(id: string): Promise<SavedConversationRecord> {
    const source = await this.get(id);
    const snapshot = source.snapshot;
    const prefix = this.randomId();
    const ids = new Map(snapshot.tree.nodes.map((node, index) => [node.id, `${prefix}-${index}`]));
    const remap = (nodeId: string): string => {
      const mapped = ids.get(nodeId);
      if (mapped === undefined) throw invalid("The conversation has a missing tree node");
      return mapped;
    };
    snapshot.tree.root_id = remap(snapshot.tree.root_id);
    snapshot.tree.active_node_id = remap(snapshot.tree.active_node_id);
    snapshot.tree.nodes = snapshot.tree.nodes.map(node => ({
      ...node,
      id: remap(node.id),
      parent_id: node.parent_id === null ? null : remap(node.parent_id),
    }));
    snapshot.tree.children_of = Object.fromEntries(Object.entries(snapshot.tree.children_of)
      .map(([parent, children]) => [remap(parent), children.map(remap)]));
    return this.create({
      name: `${source.name.slice(0, MAX_NAME_LENGTH - 7).trimEnd()} (copy)`,
      avatarSeed: source.avatarSeed,
      accent: source.accent,
      modelType: source.modelType,
      snapshot,
    });
  }

  async autosave(snapshot: ConversationSnapshotV7, activeId: string | null, modelType?: "chat" | "base"): Promise<SavedConversationRecord | null> {
    validateConversationSnapshot(snapshot, this.samplingKeys);
    await this.initialize();
    return this.runExclusive(async () => {
      let current: SavedConversationRecord | undefined;
      const existingNames = new Set<string>();
      let existingCount = 0;
      if (activeId) {
        const value = await this.store.read(activeId);
        if (value === undefined) throw new ConversationLibraryError("NOT_FOUND", "This chat was deleted. Save as new to keep your current work.");
        validateSavedConversationRecord(value, this.samplingKeys);
        if (value.modelId === snapshot.model_id && value.snapshot.tree.root_id === snapshot.tree.root_id) current = value;
      }
      if (!current) {
        const match = (row: SavedConversationStoreRow): SavedConversationRecord | null => {
          try {
            validateSavedConversationRecord(row.value, this.samplingKeys);
            existingNames.add(row.value.name);
            existingCount += 1;
            return row.value.modelId === snapshot.model_id && row.value.snapshot.tree.root_id === snapshot.tree.root_id
              ? row.value : null;
          } catch { return null; }
        };
        const matches = this.store.map ? await this.store.map(match) : (await this.store.list()).map(match);
        current = matches.filter((row): row is SavedConversationRecord => row !== null)
          .sort((a, b) => b.updatedAt - a.updatedAt)[0];
      }
      if (!current && !snapshot.tree.nodes.some(node => node.parent_id !== null)) return null;
      if (current && current.snapshot.tree.rev > snapshot.tree.rev) {
        throw new ConversationLibraryError("INVALID_RECORD", "A newer version of this chat is already saved. Open it from Your chats, or save this version as new.");
      }
      if (current && (modelType === undefined || current.modelType === modelType) && JSON.stringify({ ...current.snapshot, savedAt: "" }) === JSON.stringify({ ...snapshot, savedAt: "" })) {
        return structuredClone(current);
      }
      const timestamp = this.now();
      const id = current?.id ?? this.randomId();
      if (!current && await this.store.read(id) !== undefined) throw invalid("Unable to allocate a unique saved conversation id");
      let chatNumber = existingCount + 1;
      while (existingNames.has(`Chat ${chatNumber}`)) chatNumber += 1;
      const record: SavedConversationRecord = {
        schemaVersion: SAVED_CONVERSATION_SCHEMA_VERSION,
        id,
        name: current?.name ?? `Chat ${chatNumber}`,
        avatarSeed: current?.avatarSeed ?? normalizeAvatarSeed(this.randomId()),
        ...(current?.accent === undefined ? {} : { accent: current.accent }),
        modelId: snapshot.model_id,
        ...((modelType ?? current?.modelType) === undefined ? {} : { modelType: modelType ?? current?.modelType }),
        createdAt: current?.createdAt ?? timestamp,
        updatedAt: timestamp,
        snapshot: structuredClone(snapshot),
      };
      validateSavedConversationRecord(record, this.samplingKeys);
      await this.store.write(record);
      return structuredClone(record);
    });
  }

  async delete(id: string): Promise<boolean> {
    validateIdentifier(id, "saved conversation id");
    await this.initialize();
    return this.runExclusive(() => this.store.delete(id));
  }

  close(): void {
    this.store.close?.();
    this.initialization = null;
  }
}

export class BrowserSavedConversationStore implements SavedConversationStore {
  private database: IDBDatabase | null = null;
  private initialization: Promise<void> | null = null;

  initialize(): Promise<void> {
    this.initialization ??= this.initializeOnce().catch((error) => {
      this.initialization = null;
      throw error;
    });
    return this.initialization;
  }

  async list(): Promise<SavedConversationStoreRow[]> {
    return this.withDatabase(async (database) => {
      const transaction = database.transaction(CONVERSATIONS_STORE, "readonly");
      const done = transactionDone(transaction);
      const store = transaction.objectStore(CONVERSATIONS_STORE);
      const [keys, values] = await Promise.all([
        requestValue<IDBValidKey[]>(store.getAllKeys()),
        requestValue<unknown[]>(store.getAll()),
      ]);
      await done;
      return values.map((value, index) => ({ key: String(keys[index]), value }));
    });
  }

  async map<T>(project: (row: SavedConversationStoreRow) => T): Promise<T[]> {
    return this.withDatabase(async (database) => {
      const transaction = database.transaction(CONVERSATIONS_STORE, "readonly");
      const done = transactionDone(transaction);
      const rows: T[] = [];
      const scan = new Promise<void>((resolve, reject) => {
        const request = transaction.objectStore(CONVERSATIONS_STORE).openCursor();
        request.onerror = () => reject(request.error);
        request.onsuccess = () => {
          const cursor = request.result;
          if (cursor === null) {
            resolve();
            return;
          }
          try {
            rows.push(project({ key: String(cursor.key), value: cursor.value }));
            cursor.continue();
          } catch (error) {
            transaction.abort();
            reject(error);
          }
        };
        transaction.addEventListener("abort", () => reject(transaction.error), { once: true });
      });
      await Promise.all([scan, done]);
      return rows;
    });
  }

  async hasAny(): Promise<boolean> {
    return this.withDatabase(async (database) => {
      const transaction = database.transaction(CONVERSATIONS_STORE, "readonly");
      const done = transactionDone(transaction);
      const [count] = await Promise.all([
        requestValue<number>(transaction.objectStore(CONVERSATIONS_STORE).count()),
        done,
      ]);
      return count > 0;
    });
  }

  async read(id: string): Promise<unknown | undefined> {
    return this.withDatabase(async (database) => {
      const transaction = database.transaction(CONVERSATIONS_STORE, "readonly");
      const done = transactionDone(transaction);
      const value = await requestValue<unknown | undefined>(
        transaction.objectStore(CONVERSATIONS_STORE).get(id),
      );
      await done;
      return value;
    });
  }

  async write(record: SavedConversationRecord): Promise<void> {
    await this.withDatabase(async (database) => {
      const transaction = database.transaction(CONVERSATIONS_STORE, "readwrite");
      const done = transactionDone(transaction);
      transaction.objectStore(CONVERSATIONS_STORE).put(record);
      await done;
    });
  }

  async delete(id: string): Promise<boolean> {
    const present = await this.read(id) !== undefined;
    if (!present) return false;
    await this.withDatabase(async (database) => {
      const transaction = database.transaction(CONVERSATIONS_STORE, "readwrite");
      const done = transactionDone(transaction);
      transaction.objectStore(CONVERSATIONS_STORE).delete(id);
      await done;
    });
    return true;
  }

  close(): void {
    this.invalidate(this.database);
  }

  private async initializeOnce(): Promise<void> {
    if (typeof indexedDB === "undefined") {
      throw new ConversationLibraryError("INDEXEDDB_UNAVAILABLE", "Browser storage is unavailable");
    }
    const database = await openDatabase();
    this.database = database;
    database.onversionchange = () => this.invalidate(database);
    database.onclose = () => this.invalidate(database, false);
  }

  private async withDatabase<T>(operation: (database: IDBDatabase) => Promise<T>): Promise<T> {
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
    throw new ConversationLibraryError(
      "INDEXEDDB_UNAVAILABLE",
      "Browser storage closed while accessing saved conversations",
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

export function summarizeConversation(record: SavedConversationRecord): SavedConversationSummary {
  const { nodes, children_of, root_id } = record.snapshot.tree;
  return {
    schemaVersion: record.schemaVersion,
    id: record.id,
    name: record.name,
    avatarSeed: record.avatarSeed,
    ...(record.accent === undefined ? {} : { accent: record.accent }),
    modelId: record.modelId,
    ...(record.modelType === undefined ? {} : { modelType: record.modelType }),
    createdAt: record.createdAt,
    updatedAt: record.updatedAt,
    messageCount: nodes.filter((node) =>
      (node.role === "user" || node.role === "assistant") &&
      (node.text.length > 0 || (node.raw_token_ids?.length ?? 0) > 0)
    ).length,
    threadCount: nodes.filter((node) =>
      node.id !== root_id && (children_of[node.id]?.length ?? 0) === 0
    ).length,
  };
}

export function validateSavedConversationRecord(
  value: unknown,
  samplingKeys: readonly string[],
): asserts value is SavedConversationRecord {
  if (!isPlainObject(value)) throw invalid("Saved conversation is not an object");
  exactKeys(value, [
    ...("modelType" in value ? ["modelType"] : []),
    ...("accent" in value ? ["accent"] : []),
    "avatarSeed",
    "createdAt",
    "id",
    "modelId",
    "name",
    "schemaVersion",
    "snapshot",
    "updatedAt",
  ]);
  if (value.schemaVersion !== SAVED_CONVERSATION_SCHEMA_VERSION) {
    throw invalid("Saved conversation version is unsupported");
  }
  validateIdentifier(value.id, "saved conversation id");
  normalizeName(value.name);
  normalizeAvatarSeed(value.avatarSeed);
  if ("accent" in value && !isChatAccent(value.accent)) throw invalid("Chat accent color is invalid");
  validateIdentifier(value.modelId, "model id", 512);
  if ("modelType" in value && value.modelType !== "base" && value.modelType !== "chat") throw invalid("Model type is invalid");
  validateTimestamp(value.createdAt, "creation time");
  validateTimestamp(value.updatedAt, "update time");
  if (value.updatedAt < value.createdAt) throw invalid("Saved conversation update time is invalid");
  validateConversationSnapshot(value.snapshot, samplingKeys);
  if (value.snapshot.model_id !== value.modelId) {
    throw invalid("Saved conversation model does not match its snapshot");
  }
  let bytes: number;
  try {
    bytes = new TextEncoder().encode(JSON.stringify(value)).byteLength;
  } catch {
    throw invalid("Saved conversation cannot be serialized");
  }
  if (bytes > MAX_RECORD_BYTES) {
    throw new ConversationLibraryError(
      "STORAGE_LIMIT",
      "This conversation is larger than the 64 MiB save limit",
    );
  }
}

export function defaultConversationName(snapshot: ConversationSnapshotV7): string {
  const firstUserTurn = snapshot.tree.nodes.find(
    (node) => node.parent_id !== null && node.role === "user" && node.text.trim().length > 0,
  );
  if (!firstUserTurn) return "Untitled conversation";
  const text = firstUserTurn.text.replace(/\s+/gu, " ").trim();
  if (text.length <= 54) return text;
  const clipped = text.slice(0, 54).replace(/\s+\S*$/u, "").trim();
  return `${clipped || text.slice(0, 54).trim()}…`;
}

export function displayModelName(modelId: string): string {
  const raw = modelId.split("/").at(-1) ?? modelId;
  return raw
    .replace(/[-_]+/gu, " ")
    .replace(/\b(qwen|gemma|llama|mistral)(\d)/giu, (_, name: string, digit: string) =>
      `${name.charAt(0).toUpperCase()}${name.slice(1).toLowerCase()}${digit}`)
    .replace(/\b(qwen|gemma|llama|mistral)\b/giu, (name) =>
      `${name.charAt(0).toUpperCase()}${name.slice(1).toLowerCase()}`)
    .replace(/\b(\d+(?:\.\d+)?)b\b/giu, "$1B")
    .replace(/\s+/gu, " ")
    .trim();
}

export function randomAvatarSeed(): string {
  return randomUuid();
}

function normalizeName(value: unknown): string {
  if (typeof value !== "string") throw invalid("Conversation name is invalid");
  const name = value.replace(/\s+/gu, " ").trim();
  if (!name) throw invalid("Give this conversation a name");
  if (name.length > MAX_NAME_LENGTH) {
    throw invalid(`Conversation names must be ${MAX_NAME_LENGTH} characters or fewer`);
  }
  return name;
}

function normalizeAvatarSeed(value: unknown): string {
  if (typeof value !== "string") throw invalid("Avatar seed is invalid");
  const seed = value.trim();
  if (!seed || seed.length > MAX_AVATAR_SEED_LENGTH) throw invalid("Avatar seed is invalid");
  return seed;
}

function validateIdentifier(value: unknown, label: string, maximum = 256): asserts value is string {
  if (typeof value !== "string" || !value.trim() || value.length > maximum) {
    throw invalid(`${label} is invalid`);
  }
}

function validateTimestamp(value: unknown, label: string): asserts value is number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw invalid(`${label} is invalid`);
  }
}

function exactKeys(value: Record<string, unknown>, keys: string[]): void {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw invalid("Saved conversation fields are invalid");
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function invalid(message: string): ConversationLibraryError {
  return new ConversationLibraryError("INVALID_RECORD", message);
}

function normalizeError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

function randomUuid(): string {
  const cryptoApi = globalThis.crypto;
  if (!cryptoApi?.getRandomValues) throw invalid("Secure random identifiers are unavailable");
  if (typeof cryptoApi.randomUUID === "function") return cryptoApi.randomUUID();
  const bytes = cryptoApi.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0"));
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
}

async function openDatabase(): Promise<IDBDatabase> {
  const database = await new Promise<IDBDatabase>((resolve, reject) => {
    let settled = false;
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(CONVERSATIONS_STORE)) {
        database.createObjectStore(CONVERSATIONS_STORE, { keyPath: "id" });
      }
    };
    request.onsuccess = () => {
      if (settled) {
        request.result.close();
        return;
      }
      settled = true;
      resolve(request.result);
    };
    request.onerror = () => {
      if (settled) return;
      settled = true;
      reject(request.error ?? new ConversationLibraryError(
        "INDEXEDDB_UNAVAILABLE",
        "Saved conversation storage could not be opened",
      ));
    };
    request.onblocked = () => {
      if (settled) return;
      settled = true;
      reject(new ConversationLibraryError(
        "INDEXEDDB_BLOCKED",
        "Close other Drowse tabs, then try again",
      ));
    };
  });
  await migrateLegacyDatabase(database, [CONVERSATIONS_STORE]);
  return database;
}

function requestValue<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Browser storage request failed"));
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error("Browser storage transaction failed"));
    transaction.onabort = () => reject(transaction.error ?? new Error("Browser storage transaction was aborted"));
  });
}

function isClosedDatabaseError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "InvalidStateError";
}
