import { Inflate } from "fflate";
import {
  DROWSE_ARCHIVE_MAX_COMPRESSION_RATIO,
  DROWSE_ARCHIVE_MAX_ENTRIES,
  DROWSE_ARCHIVE_MAX_FILE_BYTES,
  DROWSE_ARCHIVE_MAX_TOTAL_BYTES,
  DrowseArchiveError,
  type DrowseArchiveSource,
} from "./types";

const EOCD = 0x06054b50;
const CENTRAL_FILE = 0x02014b50;
const LOCAL_FILE = 0x04034b50;
const DATA_DESCRIPTOR = 0x08074b50;
const ZIP64_EXTRA = 0x0001;
const UNICODE_PATH_EXTRA = 0x7075;
const AES_EXTRA = 0x9901;
const READ_CHUNK_BYTES = 64 * 1024;
const CENTRAL_DIRECTORY_MAX_BYTES = 32 * 1024 * 1024;

export interface ZipEntry {
  path: string;
  rawPath: Uint8Array;
  flags: number;
  compressionMethod: 0 | 8;
  crc32: number;
  compressedSize: number;
  size: number;
  localHeaderOffset: number;
  dataOffset: number;
  dataEnd: number;
  intervalEnd: number;
}

export interface ZipArchive {
  reader: RandomAccessReader;
  entries: readonly ZipEntry[];
  byPath: ReadonlyMap<string, ZipEntry>;
  expandedBytes: number;
}

export interface EntryConsumer {
  write(chunk: Uint8Array): void | Promise<void>;
}

export async function openZip(source: DrowseArchiveSource): Promise<ZipArchive> {
  const reader = new RandomAccessReader(source);
  if (reader.size < 22) invalid("archive is too short to be a ZIP file");
  const tailStart = Math.max(0, reader.size - 65_557);
  const tail = await reader.read(tailStart, reader.size);
  let eocdOffset = -1;
  for (let offset = tail.length - 22; offset >= 0; offset -= 1) {
    if (u32(tail, offset) !== EOCD) continue;
    const commentLength = u16(tail, offset + 20);
    if (offset + 22 + commentLength === tail.length) {
      eocdOffset = tailStart + offset;
      break;
    }
  }
  if (eocdOffset < 0) invalid("archive has no canonical end-of-central-directory record");
  const eocd = await reader.read(eocdOffset, eocdOffset + 22);
  const disk = u16(eocd, 4);
  const centralDisk = u16(eocd, 6);
  const diskEntries = u16(eocd, 8);
  const entryCount = u16(eocd, 10);
  const centralSize = u32(eocd, 12);
  const centralOffset = u32(eocd, 16);
  if (disk !== 0 || centralDisk !== 0 || diskEntries !== entryCount) {
    invalid("multi-disk ZIP archives are not supported");
  }
  if (
    entryCount === 0xffff
    || centralSize === 0xffffffff
    || centralOffset === 0xffffffff
  ) {
    invalid("ZIP64 metadata is not allowed by the bounded v1 format");
  }
  if (entryCount === 0) invalid("archive is empty");
  if (entryCount > DROWSE_ARCHIVE_MAX_ENTRIES) {
    limit(`archive has ${entryCount} entries; maximum is ${DROWSE_ARCHIVE_MAX_ENTRIES}`);
  }
  if (centralSize > CENTRAL_DIRECTORY_MAX_BYTES) {
    limit("archive central directory is larger than 32 MiB");
  }
  if (centralOffset + centralSize !== eocdOffset) {
    invalid("central directory bounds do not match the archive");
  }
  const central = await reader.read(centralOffset, eocdOffset);
  const entries: ZipEntry[] = [];
  const byPath = new Map<string, ZipEntry>();
  const foldedPaths = new Map<string, string>();
  let cursor = 0;
  let expandedBytes = 0;
  for (let index = 0; index < entryCount; index += 1) {
    if (cursor + 46 > central.length || u32(central, cursor) !== CENTRAL_FILE) {
      invalid(`central directory entry ${index} is truncated or invalid`);
    }
    const madeBy = u16(central, cursor + 4);
    const flags = u16(central, cursor + 8);
    const compressionMethod = u16(central, cursor + 10);
    const crc32 = u32(central, cursor + 16);
    const compressedSize = u32(central, cursor + 20);
    const size = u32(central, cursor + 24);
    const pathLength = u16(central, cursor + 28);
    const extraLength = u16(central, cursor + 30);
    const commentLength = u16(central, cursor + 32);
    const startDisk = u16(central, cursor + 34);
    const externalAttributes = u32(central, cursor + 38);
    const localHeaderOffset = u32(central, cursor + 42);
    const end = cursor + 46 + pathLength + extraLength + commentLength;
    if (end > central.length) invalid(`central directory entry ${index} is truncated`);
    const rawPath = central.slice(cursor + 46, cursor + 46 + pathLength);
    const path = decodePath(rawPath);
    validatePath(path);
    validateFlags(flags, compressionMethod, path);
    validateCompression(compressionMethod, path);
    if (startDisk !== 0) invalid(`archive entry ${JSON.stringify(path)} starts on another disk`);
    validateExtras(
      central.subarray(cursor + 46 + pathLength, cursor + 46 + pathLength + extraLength),
      path,
    );
    validateFileKind(madeBy, externalAttributes, path);
    if (commentLength !== 0) {
      invalid(`archive entry ${JSON.stringify(path)} has an unsupported comment`);
    }
    if (size > DROWSE_ARCHIVE_MAX_FILE_BYTES) {
      limit(`archive entry ${JSON.stringify(path)} is larger than 512 MiB`, path);
    }
    expandedBytes += size;
    if (expandedBytes > DROWSE_ARCHIVE_MAX_TOTAL_BYTES) {
      limit("archive expands beyond the 2 GiB total limit");
    }
    if (
      size > 0
      && (compressedSize === 0 || size / compressedSize > DROWSE_ARCHIVE_MAX_COMPRESSION_RATIO)
    ) {
      limit(`archive entry ${JSON.stringify(path)} exceeds the 100:1 compression limit`, path);
    }
    if (compressionMethod === 0 && compressedSize !== size) {
      invalid(`stored archive entry ${JSON.stringify(path)} has inconsistent sizes`, path);
    }
    if (byPath.has(path)) {
      throw new DrowseArchiveError(
        "ARCHIVE_PATH_COLLISION",
        `archive contains duplicate path ${JSON.stringify(path)}`,
        path,
      );
    }
    const folded = path.toLowerCase();
    const collision = foldedPaths.get(folded);
    if (collision !== undefined) {
      throw new DrowseArchiveError(
        "ARCHIVE_PATH_COLLISION",
        `archive paths ${JSON.stringify(collision)} and ${JSON.stringify(path)} collide case-insensitively`,
        path,
      );
    }
    const entry: ZipEntry = {
      path,
      rawPath,
      flags,
      compressionMethod: compressionMethod as 0 | 8,
      crc32,
      compressedSize,
      size,
      localHeaderOffset,
      dataOffset: 0,
      dataEnd: 0,
      intervalEnd: 0,
    };
    entries.push(entry);
    byPath.set(path, entry);
    foldedPaths.set(folded, path);
    cursor = end;
  }
  if (cursor !== central.length) invalid("central directory has trailing data");

  for (const entry of entries) await validateLocalHeader(reader, entry, centralOffset);
  const intervals = entries
    .map((entry) => ({ start: entry.localHeaderOffset, end: entry.intervalEnd, path: entry.path }))
    .sort((left, right) => left.start - right.start);
  for (let index = 0; index < intervals.length; index += 1) {
    const interval = intervals[index];
    if (interval.start < 0 || interval.end > centralOffset || interval.start >= interval.end) {
      invalid(`archive entry ${JSON.stringify(interval.path)} has invalid local bounds`, interval.path);
    }
    if (index > 0 && interval.start < intervals[index - 1].end) {
      invalid(
        `archive entries ${JSON.stringify(intervals[index - 1].path)} and ${JSON.stringify(interval.path)} overlap`,
        interval.path,
      );
    }
  }
  return { reader, entries, byPath, expandedBytes };
}

export async function readZipEntry(
  archive: ZipArchive,
  entry: ZipEntry,
  consumer?: EntryConsumer,
  signal?: AbortSignal,
): Promise<Uint8Array | null> {
  const collect = consumer === undefined;
  const chunks: Uint8Array[] = [];
  let outputBytes = 0;
  let crc = 0xffffffff;
  const consume = async (chunk: Uint8Array): Promise<void> => {
    if (chunk.length === 0) return;
    signal?.throwIfAborted();
    outputBytes += chunk.length;
    if (outputBytes > entry.size || outputBytes > DROWSE_ARCHIVE_MAX_FILE_BYTES) {
      invalid(`archive entry ${JSON.stringify(entry.path)} exceeds its declared size`, entry.path);
    }
    crc = crc32Update(crc, chunk);
    if (collect) chunks.push(chunk.slice());
    else await consumer.write(chunk);
  };
  if (entry.compressionMethod === 0) {
    for (let offset = entry.dataOffset; offset < entry.dataEnd; offset += READ_CHUNK_BYTES) {
      await consume(await archive.reader.read(offset, Math.min(offset + READ_CHUNK_BYTES, entry.dataEnd)));
    }
  } else {
    const inflated: Uint8Array[] = [];
    let inflateError: unknown;
    const inflate = new Inflate((chunk) => inflated.push(chunk));
    for (let offset = entry.dataOffset; offset < entry.dataEnd; offset += READ_CHUNK_BYTES) {
      signal?.throwIfAborted();
      const final = offset + READ_CHUNK_BYTES >= entry.dataEnd;
      try {
        inflate.push(
          await archive.reader.read(offset, Math.min(offset + READ_CHUNK_BYTES, entry.dataEnd)),
          final,
        );
      } catch (error) {
        inflateError = error;
      }
      if (inflateError !== undefined) {
        invalid(`archive entry ${JSON.stringify(entry.path)} has invalid DEFLATE data`, entry.path);
      }
      for (const chunk of inflated.splice(0)) await consume(chunk);
    }
    if (entry.compressedSize === 0) {
      try {
        inflate.push(new Uint8Array(), true);
      } catch {
        invalid(`archive entry ${JSON.stringify(entry.path)} has invalid DEFLATE data`, entry.path);
      }
      for (const chunk of inflated.splice(0)) await consume(chunk);
    }
  }
  if (outputBytes !== entry.size) {
    invalid(`archive entry ${JSON.stringify(entry.path)} has the wrong size`, entry.path);
  }
  if ((crc ^ 0xffffffff) >>> 0 !== entry.crc32) {
    invalid(`archive entry ${JSON.stringify(entry.path)} failed its ZIP CRC check`, entry.path);
  }
  if (!collect) return null;
  const output = new Uint8Array(outputBytes);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}

export class RandomAccessReader {
  readonly size: number;
  private readonly blob: Blob | null;
  private readonly bytes: Uint8Array | null;

  constructor(source: DrowseArchiveSource) {
    if (source instanceof Blob) {
      this.blob = source;
      this.bytes = null;
      this.size = source.size;
    } else {
      this.blob = null;
      this.bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
      this.size = this.bytes.byteLength;
    }
  }

  async read(start: number, end: number): Promise<Uint8Array> {
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || end > this.size) {
      invalid("archive read is outside the file bounds");
    }
    if (this.blob !== null) return new Uint8Array(await this.blob.slice(start, end).arrayBuffer());
    return this.bytes!.slice(start, end);
  }
}

async function validateLocalHeader(
  reader: RandomAccessReader,
  entry: ZipEntry,
  centralOffset: number,
): Promise<void> {
  if (entry.localHeaderOffset + 30 > centralOffset) {
    invalid(`archive entry ${JSON.stringify(entry.path)} has a truncated local header`, entry.path);
  }
  const header = await reader.read(entry.localHeaderOffset, entry.localHeaderOffset + 30);
  if (u32(header, 0) !== LOCAL_FILE) {
    invalid(`archive entry ${JSON.stringify(entry.path)} has no local header`, entry.path);
  }
  const flags = u16(header, 6);
  const method = u16(header, 8);
  const crc = u32(header, 14);
  const compressedSize = u32(header, 18);
  const size = u32(header, 22);
  const pathLength = u16(header, 26);
  const extraLength = u16(header, 28);
  const variableEnd = entry.localHeaderOffset + 30 + pathLength + extraLength;
  if (variableEnd > centralOffset) {
    invalid(`archive entry ${JSON.stringify(entry.path)} has a truncated local header`, entry.path);
  }
  const variable = await reader.read(entry.localHeaderOffset + 30, variableEnd);
  const rawPath = variable.subarray(0, pathLength);
  if (!equalBytes(rawPath, entry.rawPath)) {
    invalid(`archive entry ${JSON.stringify(entry.path)} has mismatched local and central paths`, entry.path);
  }
  validateExtras(variable.subarray(pathLength), entry.path);
  if (flags !== entry.flags || method !== entry.compressionMethod) {
    invalid(`archive entry ${JSON.stringify(entry.path)} has mismatched local metadata`, entry.path);
  }
  if ((flags & 0x08) === 0) {
    if (crc !== entry.crc32 || compressedSize !== entry.compressedSize || size !== entry.size) {
      invalid(`archive entry ${JSON.stringify(entry.path)} has mismatched local sizes or CRC`, entry.path);
    }
  } else if (
    (crc !== 0 && crc !== entry.crc32)
    || (compressedSize !== 0 && compressedSize !== entry.compressedSize)
    || (size !== 0 && size !== entry.size)
  ) {
    invalid(`archive entry ${JSON.stringify(entry.path)} has an invalid deferred local header`, entry.path);
  }
  entry.dataOffset = variableEnd;
  entry.dataEnd = variableEnd + entry.compressedSize;
  entry.intervalEnd = entry.dataEnd;
  if (entry.dataEnd > centralOffset) {
    invalid(`archive entry ${JSON.stringify(entry.path)} payload exceeds the local data area`, entry.path);
  }
  if ((flags & 0x08) !== 0) {
    const available = Math.min(16, centralOffset - entry.dataEnd);
    if (available < 12) {
      invalid(`archive entry ${JSON.stringify(entry.path)} has no complete data descriptor`, entry.path);
    }
    const descriptor = await reader.read(entry.dataEnd, entry.dataEnd + available);
    let base = 0;
    if (u32(descriptor, 0) === DATA_DESCRIPTOR) base = 4;
    if (descriptor.length < base + 12) {
      invalid(`archive entry ${JSON.stringify(entry.path)} has a truncated data descriptor`, entry.path);
    }
    if (
      u32(descriptor, base) !== entry.crc32
      || u32(descriptor, base + 4) !== entry.compressedSize
      || u32(descriptor, base + 8) !== entry.size
    ) {
      invalid(`archive entry ${JSON.stringify(entry.path)} has a mismatched data descriptor`, entry.path);
    }
    entry.intervalEnd += base + 12;
  }
}

function decodePath(raw: Uint8Array): string {
  if (raw.length === 0 || raw.some((byte) => byte === 0 || byte > 0x7f)) {
    throw new DrowseArchiveError(
      "ARCHIVE_PATH_INVALID",
      "archive paths must be non-empty canonical ASCII without NUL bytes",
    );
  }
  return String.fromCharCode(...raw);
}

function validatePath(path: string): void {
  if (
    path.includes("\\")
    || path.startsWith("/")
    || /^[A-Za-z]:/.test(path)
    || path.split("/").some((part) => part === "" || part === "." || part === "..")
  ) {
    throw new DrowseArchiveError(
      "ARCHIVE_PATH_INVALID",
      `archive path ${JSON.stringify(path)} is not canonical`,
      path,
    );
  }
}

function validateFlags(flags: number, method: number, path: string): void {
  if ((flags & 0x01) !== 0 || (flags & 0x40) !== 0 || (flags & 0x2000) !== 0) {
    throw new DrowseArchiveError(
      "ARCHIVE_ENTRY_UNSAFE",
      `archive entry ${JSON.stringify(path)} is encrypted`,
      path,
    );
  }
  const allowed = 0x0008 | 0x0800 | (method === 8 ? 0x0006 : 0);
  if ((flags & ~allowed) !== 0) {
    throw new DrowseArchiveError(
      "ARCHIVE_ENTRY_UNSAFE",
      `archive entry ${JSON.stringify(path)} uses unsupported ZIP flags`,
      path,
    );
  }
}

function validateCompression(method: number, path: string): void {
  if (method !== 0 && method !== 8) {
    throw new DrowseArchiveError(
      "ARCHIVE_COMPRESSION_UNSUPPORTED",
      `archive entry ${JSON.stringify(path)} uses unsupported compression method ${method}`,
      path,
    );
  }
}

function validateFileKind(madeBy: number, attributes: number, path: string): void {
  if (path.endsWith("/") || (attributes & 0x10) !== 0) {
    throw new DrowseArchiveError(
      "ARCHIVE_ENTRY_UNSAFE",
      `archive entry ${JSON.stringify(path)} must be a regular file`,
      path,
    );
  }
  const host = madeBy >>> 8;
  if (host === 3) {
    const kind = (attributes >>> 16) & 0xf000;
    if (kind !== 0 && kind !== 0x8000) {
      throw new DrowseArchiveError(
        "ARCHIVE_ENTRY_UNSAFE",
        `archive entry ${JSON.stringify(path)} is a link or special file`,
        path,
      );
    }
  }
}

function validateExtras(extra: Uint8Array, path: string): void {
  let offset = 0;
  while (offset < extra.length) {
    if (offset + 4 > extra.length) invalid(`archive entry ${JSON.stringify(path)} has malformed ZIP extras`, path);
    const tag = u16(extra, offset);
    const size = u16(extra, offset + 2);
    offset += 4;
    if (offset + size > extra.length) invalid(`archive entry ${JSON.stringify(path)} has malformed ZIP extras`, path);
    if (tag === ZIP64_EXTRA) invalid(`archive entry ${JSON.stringify(path)} uses unsupported ZIP64 metadata`, path);
    if (tag === AES_EXTRA) {
      throw new DrowseArchiveError("ARCHIVE_ENTRY_UNSAFE", `archive entry ${JSON.stringify(path)} is encrypted`, path);
    }
    if (tag === UNICODE_PATH_EXTRA) {
      invalid(`archive entry ${JSON.stringify(path)} carries an alternate Unicode path`, path);
    }
    offset += size;
  }
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.length !== right.length) return false;
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

function u16(bytes: Uint8Array, offset: number): number {
  return bytes[offset] | (bytes[offset + 1] << 8);
}

function u32(bytes: Uint8Array, offset: number): number {
  return (
    bytes[offset]
    | (bytes[offset + 1] << 8)
    | (bytes[offset + 2] << 16)
    | (bytes[offset + 3] << 24)
  ) >>> 0;
}

const crcTable = makeCrcTable();

function makeCrcTable(): Uint32Array {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value & 1) !== 0 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    }
    table[index] = value >>> 0;
  }
  return table;
}

function crc32Update(crc: number, bytes: Uint8Array): number {
  let value = crc;
  for (const byte of bytes) value = crcTable[(value ^ byte) & 0xff] ^ (value >>> 8);
  return value >>> 0;
}

function invalid(message: string, path?: string): never {
  throw new DrowseArchiveError("ARCHIVE_INVALID", message, path);
}

function limit(message: string, path?: string): never {
  throw new DrowseArchiveError("ARCHIVE_LIMIT_EXCEEDED", message, path);
}
