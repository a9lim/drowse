import { strFromU8, strToU8, unzipSync, zipSync } from "fflate";
import { CONVERSATION_SAMPLING_KEYS, validateConversationSnapshot, type ConversationSnapshotV7 } from "./conversationSnapshot";
import { defaultConversationName, validateSavedConversationRecord, type ConversationLibrary, type SavedConversationRecord, type SavedConversationSummary } from "./conversationLibrary";
import { validateHostedLoomTree } from "./sessionPersistence";
import { migrateLegacyRecord } from "./brandMigration";

export const CHAT_BACKUP_FORMAT = "drowse-chat-backup";
export const CHAT_BACKUP_VERSION = 1;
export const MAX_CHAT_BACKUP_BYTES = 65 * 1024 * 1024;
export const MAX_CHAT_ARCHIVE_BYTES = 512 * 1024 * 1024;
const MAX_FILENAME_STEM = 80;
const BACKUP_EXTENSION = /(?:\.(?:drowse|polythetic|saklas)chat|\.(?:drowse|polythetic|saklas)-chat\.json|\.json)$/iu;

interface ChatBackup {
  format: typeof CHAT_BACKUP_FORMAT;
  version: typeof CHAT_BACKUP_VERSION;
  exportedAt: string;
  sha256: string;
  conversation: SavedConversationRecord;
}

/** A validated chat backup that has not been written to the library yet. */
export interface ChatImportCandidate {
  /** The id the chat had when it was exported, when the file records one. */
  sourceId: string | null;
  name: string;
  avatarSeed?: string;
  accent?: SavedConversationRecord["accent"];
  modelType?: SavedConversationRecord["modelType"];
  createdAt?: number;
  updatedAt?: number;
  snapshot: ConversationSnapshotV7;
}

type BackupFile = Pick<File, "size" | "text"> & Partial<Pick<File, "name" | "arrayBuffer">>;

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

/** Parses and validates one backup file without changing the library. */
export async function readChatBackup(file: Pick<File, "size" | "text">): Promise<ChatImportCandidate> {
  if (file.size > MAX_CHAT_BACKUP_BYTES) throw new Error("This backup exceeds the 65 MiB import limit.");
  return parseChatBackup(await file.text());
}

function parseChatBackup(text: string): Promise<ChatImportCandidate> | ChatImportCandidate {
  let parsed: unknown;
  try { parsed = JSON.parse(text); }
  catch { throw new Error("This file is not valid chat backup JSON."); }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Choose a Drowse chat backup or a version 7 conversation file.");
  }
  if ("format" in parsed) return parseFormattedBackup(parsed);
  const snapshot = migrateLegacyRecord(parsed);
  validateConversationSnapshot(snapshot, CONVERSATION_SAMPLING_KEYS);
  validateHostedLoomTree(snapshot.tree);
  return { sourceId: null, name: defaultConversationName(), snapshot };
}

async function parseFormattedBackup(parsed: object): Promise<ChatImportCandidate> {
  if (!("format" in parsed) || ![CHAT_BACKUP_FORMAT, "polythetic-chat-backup", "saklas-chat-backup"].includes(String(parsed.format)) || !("version" in parsed) || parsed.version !== CHAT_BACKUP_VERSION) {
    throw new Error("This chat backup format or version is not supported.");
  }
  if (!("conversation" in parsed) || !("sha256" in parsed)) throw new Error("This chat backup is incomplete.");
  if (parsed.sha256 !== await checksum(parsed.conversation)) {
    throw new Error("This backup failed its integrity check. The file may be damaged or incomplete; download the backup again and retry.");
  }
  const record = migrateLegacyRecord(parsed.conversation);
  validateSavedConversationRecord(record, CONVERSATION_SAMPLING_KEYS);
  validateHostedLoomTree(record.snapshot.tree);
  return {
    sourceId: record.id,
    name: record.name, avatarSeed: record.avatarSeed, accent: record.accent, modelType: record.modelType, snapshot: record.snapshot,
    createdAt: record.createdAt, updatedAt: record.updatedAt,
  };
}

/** Writes a validated backup to the library as a new chat. */
export function saveChatImport(candidate: ChatImportCandidate, library: ConversationLibrary): Promise<SavedConversationRecord> {
  const { sourceId: _sourceId, ...input } = candidate;
  if (input.avatarSeed === undefined) delete input.avatarSeed;
  if (input.accent === undefined) delete input.accent;
  if (input.modelType === undefined) delete input.modelType;
  if (input.createdAt === undefined) delete input.createdAt;
  if (input.updatedAt === undefined) delete input.updatedAt;
  return library.create(input);
}

/**
 * Finds a saved chat with the same content as a backup: either the chat it
 * was exported from or an earlier import of the same file. Matches compare
 * the conversation content hash, so an edited chat is not reported.
 */
export async function findImportedChat(
  candidate: ChatImportCandidate,
  library: ConversationLibrary,
  summaries?: SavedConversationSummary[],
): Promise<SavedConversationSummary | null> {
  const rows = summaries ?? (await library.listSummaries()).conversations;
  const matches = rows.filter(row =>
    row.id === candidate.sourceId
    || (row.modelId === candidate.snapshot.model_id && candidate.createdAt !== undefined && row.createdAt === candidate.createdAt));
  if (matches.length === 0) return null;
  const expected = await checksum(candidate.snapshot);
  for (const row of matches) {
    try {
      const existing = await library.get(row.id);
      if (await checksum(existing.snapshot) === expected) return row;
    } catch {}
  }
  return null;
}

export async function importChatBackup(
  file: Pick<File, "size" | "text">,
  library: ConversationLibrary,
): Promise<SavedConversationRecord> {
  return saveChatImport(await readChatBackup(file), library);
}

/**
 * Reads a single backup or a "Back up all chats" archive. Each entry is
 * validated independently; entries that fail are reported by name.
 */
export async function readChatBackupFiles(file: BackupFile): Promise<{
  candidates: Array<{ label: string; candidate: ChatImportCandidate }>;
  failures: Array<{ label: string; error: unknown }>;
}> {
  const label = file.name ?? "chat backup";
  if (!isArchive(file)) return { candidates: [{ label, candidate: await readChatBackup(file) }], failures: [] };
  if (file.size > MAX_CHAT_ARCHIVE_BYTES) throw new Error("This archive exceeds the 512 MiB import limit.");
  let entries: Record<string, Uint8Array>;
  try {
    entries = unzipSync(new Uint8Array(await file.arrayBuffer!()), {
      filter: entry => !entry.name.endsWith("/") && BACKUP_EXTENSION.test(entry.name) && entry.originalSize <= MAX_CHAT_BACKUP_BYTES,
    });
  } catch {
    throw new Error("This archive could not be read. Download the backup again and retry.");
  }
  const names = Object.keys(entries).sort();
  if (names.length === 0) throw new Error("This archive does not contain any chat backups.");
  const candidates: Array<{ label: string; candidate: ChatImportCandidate }> = [];
  const failures: Array<{ label: string; error: unknown }> = [];
  for (const name of names) {
    try {
      candidates.push({ label: name, candidate: await parseChatBackup(strFromU8(entries[name])) });
    } catch (error) {
      failures.push({ label: name, error });
    }
  }
  return { candidates, failures };
}

function isArchive(file: BackupFile): file is BackupFile & Pick<File, "arrayBuffer"> {
  return typeof file.arrayBuffer === "function" && /\.zip$/iu.test(file.name ?? "");
}

export async function downloadChatBackup(library: ConversationLibrary, id: string): Promise<void> {
  const record = await library.get(id);
  const blob = new Blob([await encodeChatBackup(record)], { type: "application/json" });
  downloadPreparedChatBackup(blob, record.name);
}

/** Builds one .zip holding a .drowsechat file per chat. Import reads it back. */
export async function encodeChatArchive(records: SavedConversationRecord[]): Promise<Uint8Array> {
  const files: Record<string, Uint8Array> = {};
  for (const record of records) {
    const filename = chatBackupFilename(record.name);
    const stem = filename.slice(0, -".drowsechat".length);
    let unique = filename;
    for (let index = 2; unique in files; index += 1) unique = `${stem}-${index}.drowsechat`;
    files[unique] = strToU8(await encodeChatBackup(record));
  }
  return zipSync(files, { level: 6 });
}

export async function downloadAllChatBackups(library: ConversationLibrary, ids: string[], now = new Date()): Promise<number> {
  const records: SavedConversationRecord[] = [];
  for (const id of ids) records.push(await library.get(id));
  const bytes = await encodeChatArchive(records);
  const stamp = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
  downloadBlob(new Blob([bytes as BlobPart], { type: "application/zip" }), `drowse-chats-${stamp}.zip`);
  return records.length;
}

/**
 * Keeps letters and numbers from any script so non-Latin names survive.
 * Removes only characters that are unsafe in file names on common systems.
 */
export function chatBackupFilename(name: string): string {
  const cleaned = name.normalize("NFC").trim().replace(BACKUP_EXTENSION, "")
    .replace(/[^\p{L}\p{M}\p{N}._-]+/gu, "-").replace(/-{2,}/gu, "-").replace(/^-+|[-.]+$/gu, "");
  const stem = Array.from(cleaned).slice(0, MAX_FILENAME_STEM).join("").replace(/[-.]+$/gu, "") || "chat";
  return `${stem}.drowsechat`;
}

export function downloadPreparedChatBackup(blob: Blob, name: string): void {
  downloadBlob(blob, chatBackupFilename(name));
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}
