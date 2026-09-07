import { CONVERSATION_SAMPLING_KEYS, validateConversationSnapshot } from "./conversationSnapshot";
import { defaultConversationName, validateSavedConversationRecord, type ConversationLibrary, type SavedConversationRecord } from "./conversationLibrary";
import { validateHostedLoomTree } from "../hosted/runtime/sessionPersistence";
import { migrateLegacyRecord } from "../hosted/runtime/brandMigration";

export const CHAT_BACKUP_FORMAT = "drowse-chat-backup";
export const CHAT_BACKUP_VERSION = 1;
export const MAX_CHAT_BACKUP_BYTES = 65 * 1024 * 1024;

interface ChatBackup {
  format: typeof CHAT_BACKUP_FORMAT;
  version: typeof CHAT_BACKUP_VERSION;
  exportedAt: string;
  sha256: string;
  conversation: SavedConversationRecord;
}

async function checksum(value: unknown): Promise<string> {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, "0")).join("");
}

export async function encodeChatBackup(record: SavedConversationRecord): Promise<string> {
  validateSavedConversationRecord(record, CONVERSATION_SAMPLING_KEYS);
  validateHostedLoomTree(record.snapshot.tree);
  const backup: ChatBackup = {
    format: CHAT_BACKUP_FORMAT,
    version: CHAT_BACKUP_VERSION,
    exportedAt: new Date().toISOString(),
    sha256: await checksum(record),
    conversation: record,
  };
  return JSON.stringify(backup);
}

export async function importChatBackup(
  file: Pick<File, "size" | "text">,
  library: ConversationLibrary,
): Promise<SavedConversationRecord> {
  if (file.size > MAX_CHAT_BACKUP_BYTES) throw new Error("This backup exceeds the 65 MiB import limit.");
  let parsed: unknown;
  try { parsed = JSON.parse(await file.text()); }
  catch { throw new Error("This file is not valid chat backup JSON."); }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Choose a Drowse chat backup or a version 7 conversation file.");
  }
  if ("format" in parsed) {
    if (![CHAT_BACKUP_FORMAT, "polythetic-chat-backup", "saklas-chat-backup"].includes(String(parsed.format)) || !("version" in parsed) || parsed.version !== CHAT_BACKUP_VERSION) {
      throw new Error("This chat backup format or version is not supported.");
    }
    if (!("conversation" in parsed) || !("sha256" in parsed)) throw new Error("This chat backup is incomplete.");
    if (parsed.sha256 !== await checksum(parsed.conversation)) {
      throw new Error("This backup failed its integrity check. The file may be damaged or edited; choose the original download.");
    }
    const record = migrateLegacyRecord(parsed.conversation);
    validateSavedConversationRecord(record, CONVERSATION_SAMPLING_KEYS);
    validateHostedLoomTree(record.snapshot.tree);
    return library.create({
      name: record.name, avatarSeed: record.avatarSeed, accent: record.accent, modelType: record.modelType, snapshot: record.snapshot,
      createdAt: record.createdAt, updatedAt: record.updatedAt,
    });
  }
  const snapshot = migrateLegacyRecord(parsed);
  validateConversationSnapshot(snapshot, CONVERSATION_SAMPLING_KEYS);
  validateHostedLoomTree(snapshot.tree);
  return library.create({ name: defaultConversationName(), snapshot });
}

export async function downloadChatBackup(library: ConversationLibrary, id: string): Promise<void> {
  const record = await library.get(id);
  const blob = new Blob([await encodeChatBackup(record)], { type: "application/json" });
  downloadPreparedChatBackup(blob, record.name);
}

export function chatBackupFilename(name: string): string {
  const stem = name.trim().replace(/(?:\.(?:drowse|polythetic|saklas)chat|\.(?:drowse|polythetic|saklas)-chat\.json|\.json)$/iu, "")
    .replace(/[^a-z0-9._-]+/giu, "-").replace(/^-+|-+$/gu, "").slice(0, 80) || "chat";
  return `${stem}.drowsechat`;
}

export function downloadPreparedChatBackup(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = chatBackupFilename(name);
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}
