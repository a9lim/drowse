import { ConversationLibrary } from "../conversationLibrary";
import type { ChatAccent } from "../chatAccent";
import { CONVERSATION_SAMPLING_KEYS } from "../conversationSnapshot";
import { requestPersistentStorage } from "../runtime/storagePersistence";

export const conversationLibrary = new ConversationLibrary({
  samplingKeys: CONVERSATION_SAMPLING_KEYS,
});

export const savedConversationState: {
  activeId: string | null;
  avatarSeed: string | null;
  accent: ChatAccent;
  status: "idle" | "pending" | "saving" | "saved" | "error";
  error: string | null;
} = $state({
  activeId: null,
  avatarSeed: null,
  accent: "purple",
  status: "idle",
  error: null,
});

let flushHandler: (() => Promise<void>) | null = null;
const libraryListeners = new Set<() => void | Promise<void>>();

export async function notifyConversationLibraryChanged(): Promise<void> {
  await Promise.all([...libraryListeners].map(listener => listener()));
}

export function onConversationLibraryChanged(listener: () => void | Promise<void>): () => void {
  libraryListeners.add(listener);
  return () => libraryListeners.delete(listener);
}

export function registerConversationAutosave(handler: () => Promise<void>): () => void {
  flushHandler = handler;
  return () => { if (flushHandler === handler) flushHandler = null; };
}

export async function flushConversationAutosave(): Promise<void> {
  await flushHandler?.();
}

export async function requestPersistentConversationStorage(): Promise<boolean | null> {
  if (!navigator.storage?.persist) return null;
  return requestPersistentStorage(navigator.storage);
}
