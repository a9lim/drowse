import { migrateLegacyStorageItem, removeLegacyStorageItem } from "./brandMigration";

const ENTRY_MEMORY_KEY = "drowse.entry.v1";
const PENDING_CONVERSATION_KEY = "drowse.pending-conversation.v1";

export interface EntryMemory {
  version: 1;
  completedAt: number;
  lastModelVariantId: string;
}

export interface PendingConversationOpen {
  version: 1;
  id: string;
  modelId: string;
}

type WebStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;

export function readEntryMemory(
  storage: WebStorage | undefined = browserStorage("localStorage"),
): EntryMemory | null {
  if (!storage) return null;
  try {
    const raw = migrateLegacyStorageItem(storage, ENTRY_MEMORY_KEY);
    if (!raw) return null;
    const value = JSON.parse(raw) as Partial<EntryMemory>;
    if (
      value.version !== 1 ||
      !Number.isSafeInteger(value.completedAt) ||
      (value.completedAt ?? -1) < 0 ||
      typeof value.lastModelVariantId !== "string" ||
      !value.lastModelVariantId.trim()
    ) return null;
    return value as EntryMemory;
  } catch {
    return null;
  }
}

export function rememberCompletedOnboarding(
  modelVariantId: string,
  storage: WebStorage | undefined = browserStorage("localStorage"),
  now = Date.now(),
): void {
  if (!storage || !modelVariantId.trim()) return;
  try {
    storage.setItem(ENTRY_MEMORY_KEY, JSON.stringify({
      version: 1,
      completedAt: now,
      lastModelVariantId: modelVariantId,
    } satisfies EntryMemory));
  } catch {}
}

export function queueConversationOpen(
  id: string,
  modelId: string,
  storage: WebStorage | undefined = browserStorage("sessionStorage"),
): boolean {
  if (!storage || !id.trim() || !modelId.trim()) return false;
  try {
    storage.setItem(PENDING_CONVERSATION_KEY, JSON.stringify({
      version: 1,
      id,
      modelId,
    } satisfies PendingConversationOpen));
    return true;
  } catch {
    return false;
  }
}

export function peekConversationOpen(
  storage: WebStorage | undefined = browserStorage("sessionStorage"),
): PendingConversationOpen | null {
  if (!storage) return null;
  try {
    const raw = migrateLegacyStorageItem(storage, PENDING_CONVERSATION_KEY);
    if (!raw) return null;
    const value = JSON.parse(raw) as Partial<PendingConversationOpen>;
    if (
      value.version !== 1 ||
      typeof value.id !== "string" ||
      !value.id.trim() ||
      typeof value.modelId !== "string" ||
      !value.modelId.trim()
    ) return null;
    return value as PendingConversationOpen;
  } catch {
    return null;
  }
}

export function clearConversationOpen(
  storage: WebStorage | undefined = browserStorage("sessionStorage"),
): void {
  try {
    if (storage) removeLegacyStorageItem(storage, PENDING_CONVERSATION_KEY);
  } catch {}
}

function browserStorage(name: "localStorage" | "sessionStorage"): Storage | undefined {
  if (typeof window === "undefined") return undefined;
  try {
    return window[name];
  } catch {
    return undefined;
  }
}
