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
