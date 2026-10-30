/** The saved-chat handoff into the workspace: a chats page queues the chat to
 *  open in sessionStorage, and the app opens it on its next boot. */
import { migrateLegacyStorageItem, removeLegacyStorageItem } from "./brandMigration";

const PENDING_CONVERSATION_KEY = "drowse.pending-conversation.v1";

export interface PendingConversationOpen {
  version: 1;
  id: string;
  modelId: string;
}

type WebStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;

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
