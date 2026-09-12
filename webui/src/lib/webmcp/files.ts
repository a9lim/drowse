import { sha256 } from "@noble/hashes/sha2.js";
import { objectSchema, ToolError, type AppTool } from "./types";

export interface FileDescriptor {
  file_id: string;
  name: string;
  size: number;
  mime_type: string;
  sha256: string | null;
  state: "receiving" | "complete";
  received_bytes: number;
}

const MAX_FILE_BYTES = 1024 * 1024 * 1024;
const CHUNK_BYTES = 512 * 1024;
interface Transfer { descriptor: FileDescriptor; parts: Blob[]; file?: File; expectedHash?: string }

export class FileTransfers {
  private records = new Map<string, Transfer>();

  begin(name: string, size: number, mimeType = "application/octet-stream", expectedHash?: string): FileDescriptor {
    if (!name || name.length > 255 || /[\x00-\x1f/\\]/.test(name)) throw new ToolError("INVALID_FILENAME", "Use a filename without a directory or control characters.");
    if (!Number.isSafeInteger(size) || size < 0 || size > MAX_FILE_BYTES) throw new ToolError("FILE_SIZE", "Transfers support files up to 1 GiB; the import action also enforces its format-specific limit.");
    if (expectedHash !== undefined && !/^[a-f0-9]{64}$/i.test(expectedHash)) throw new ToolError("INVALID_CHECKSUM", "Supply a hexadecimal SHA-256 checksum.");
    if (this.records.size >= 16 || [...this.records.values()].reduce((sum, record) => sum + record.descriptor.size, 0) + size > MAX_FILE_BYTES) {
      throw new ToolError("TRANSFER_CAPACITY", "Delete completed or abandoned transfers before uploading more files.");
    }
    const descriptor: FileDescriptor = { file_id: crypto.randomUUID(), name, size, mime_type: mimeType,
      sha256: null, state: "receiving", received_bytes: 0 };
    this.records.set(descriptor.file_id, { descriptor, parts: [], expectedHash: expectedHash?.toLowerCase() });
    return { ...descriptor };
  }

  async write(id: string, offset: number, base64: string): Promise<FileDescriptor> {
    const record = this.require(id);
    if (!Number.isSafeInteger(offset) || offset < 0) throw new ToolError("INVALID_OFFSET", "Offset must be a nonnegative byte index.");
    if (base64.length > Math.ceil(CHUNK_BYTES / 3) * 4 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(base64)) {
      throw new ToolError("INVALID_CHUNK", "Send canonical base64 containing at most 512 KiB of bytes.");
    }
    const bytes = Uint8Array.from(atob(base64), character => character.charCodeAt(0));
    if (!bytes.length || bytes.length > CHUNK_BYTES || offset + bytes.length > record.descriptor.size) throw new ToolError("INVALID_CHUNK", "The chunk is empty or exceeds the declared file size.");
    if (offset < record.descriptor.received_bytes) {
      const existing = new Uint8Array(await new Blob(record.parts).slice(offset, offset + bytes.length).arrayBuffer());
      if (existing.length === bytes.length && existing.every((value, i) => value === bytes[i])) return { ...record.descriptor };
      throw new ToolError("CHUNK_CONFLICT", "This byte range was already uploaded with different content.");
    }
    if (record.descriptor.state !== "receiving" || offset !== record.descriptor.received_bytes) throw new ToolError("INVALID_OFFSET", `Continue at byte ${record.descriptor.received_bytes}.`);
    record.parts.push(new Blob([bytes]));
    record.descriptor.received_bytes += bytes.length;
    return { ...record.descriptor };
  }

  async finish(id: string): Promise<FileDescriptor> {
    const record = this.require(id);
    if (record.descriptor.state === "complete") return { ...record.descriptor };
    if (record.descriptor.received_bytes !== record.descriptor.size) throw new ToolError("INCOMPLETE_FILE", `Uploaded ${record.descriptor.received_bytes} of ${record.descriptor.size} bytes.`);
    const file = new File(record.parts, record.descriptor.name, { type: record.descriptor.mime_type });
    const hash = sha256.create();
    for await (const bytes of file.stream()) hash.update(bytes);
    const checksum = Array.from(hash.digest(), byte => byte.toString(16).padStart(2, "0")).join("");
    if (record.expectedHash && record.expectedHash !== checksum) throw new ToolError("CHECKSUM_MISMATCH", "The uploaded file does not match its expected SHA-256 checksum.");
    record.file = file;
    record.descriptor.sha256 = checksum;
    record.descriptor.state = "complete";
    return { ...record.descriptor };
  }

  async add(blob: Blob, name: string): Promise<FileDescriptor> {
    const descriptor = this.begin(name, blob.size, blob.type);
    const record = this.require(descriptor.file_id);
    record.parts = [blob];
    record.descriptor.received_bytes = blob.size;
    return this.finish(descriptor.file_id);
  }

  get(id: string): File {
    const record = this.require(id);
    if (!record.file) throw new ToolError("INCOMPLETE_FILE", "Finish and verify the transfer before importing it.");
    return record.file;
  }

  describe(id: string): FileDescriptor { return { ...this.require(id).descriptor }; }
  list(): FileDescriptor[] { return [...this.records.values()].map(record => ({ ...record.descriptor })); }
  delete(id: string): void { this.require(id); this.records.delete(id); }
  clear(): void { this.records.clear(); }

  private require(id: string): Transfer {
    const record = this.records.get(id);
    if (!record) throw new ToolError("FILE_NOT_FOUND", "This file handle expired or belongs to another page. Transfer the file again.");
    return record;
  }
}

const transfers = new FileTransfers();
export const getTransferredFile = (id: string): File => transfers.get(id);
export const addTransferredFile = (blob: Blob, name: string): Promise<FileDescriptor> => transfers.add(blob, name);
export function downloadTransferredFile(id: string): FileDescriptor {
  const url = URL.createObjectURL(transfers.get(id));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = transfers.describe(id).name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
  return transfers.describe(id);
}

export function createFileTools(): AppTool[] {
  const id = { type: "string" };
  const make = (name: string, description: string, inputSchema: AppTool["inputSchema"], execute: AppTool["execute"], readOnlyHint = false): AppTool => ({
    name, title: name.replace("drowse_", "").replaceAll("_", " "), description, inputSchema, execute,
    group: "files", scope: "page", annotations: { readOnlyHint, untrustedContentHint: true },
  });
  return [
    make("drowse_begin_file", "Begin a page-local binary upload. Send base64 chunks with drowse_write_file, then finish to verify its size/checksum. This does not import or execute the file. Max 1 GiB total retained; chat/image/archive imports enforce their own existing limits.", objectSchema({ name: { type: "string", minLength: 1, maxLength: 255 }, size: { type: "integer", minimum: 0, maximum: MAX_FILE_BYTES }, mime_type: { type: "string", maxLength: 128 }, sha256: { type: "string", minLength: 64, maxLength: 64 } }, ["name", "size"]), input => transfers.begin(input.name as string, input.size as number, input.mime_type as string | undefined, input.sha256 as string | undefined)),
    make("drowse_write_file", "Append at the returned byte offset, up to 512 KiB per chunk. Identical repeated chunks are idempotent; conflicting bytes are rejected.", objectSchema({ file_id: id, offset: { type: "integer", minimum: 0 }, base64: { type: "string", maxLength: Math.ceil(CHUNK_BYTES / 3) * 4 } }, ["file_id", "offset", "base64"]), input => transfers.write(input.file_id as string, input.offset as number, input.base64 as string)),
    make("drowse_finish_file", "Verify all declared bytes and optional SHA-256, then make the file_id available to a specific import action. Does not itself change chats, artifacts or wallpaper.", objectSchema({ file_id: id }, ["file_id"]), input => transfers.finish(input.file_id as string)),
    make("drowse_read_file", "Read transfer status or a bounded base64 chunk from a completed file. Offsets and limits are bytes. Omit file_id to list page-local transfers.", objectSchema({ file_id: id, offset: { type: "integer", minimum: 0 }, limit: { type: "integer", minimum: 1, maximum: 4096 } }), async input => {
      if (!input.file_id) return { files: transfers.list() };
      const descriptor = transfers.describe(input.file_id as string);
      if (input.offset === undefined) return descriptor;
      const start = input.offset as number, file = transfers.get(input.file_id as string);
      if (start > file.size) throw new ToolError("INVALID_OFFSET", "Offset exceeds the file size.");
      const bytes = new Uint8Array(await file.slice(start, start + (input.limit as number | undefined ?? 4096)).arrayBuffer());
      return { ...descriptor, offset: start, base64: btoa(String.fromCharCode(...bytes)), next_offset: start + bytes.length < file.size ? start + bytes.length : null };
    }, true),
    make("drowse_download_file", "Start a normal browser download of a completed transfer or exported file. Completion means the download was offered; browser save placement remains user-controlled.", objectSchema({ file_id: id }, ["file_id"]), input => downloadTransferredFile(input.file_id as string)),
    make("drowse_delete_file", "Release a page-local transfer. Does not delete any imported chat, installed artifact, or file on disk.", objectSchema({ file_id: id }, ["file_id"]), input => { transfers.delete(input.file_id as string); return { deleted: input.file_id }; }),
  ];
}
