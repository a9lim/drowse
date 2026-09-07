import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import { zipSync } from "fflate";
import { validateDrowseArchive } from "./drowseArchive";
import {
  DROWSE_ARCHIVE_MANIFEST_MAX_BYTES,
  DROWSE_ARCHIVE_MAX_ENTRIES,
  DROWSE_ARCHIVE_MAX_FILE_BYTES,
  DrowseArchiveError,
  type DrowseArchiveManifest,
} from "./types";

export interface BuildDrowseArchiveInput {
  primary: string;
  template: string | null;
  producerVersion: string;
  source: DrowseArchiveManifest["source"];
  files: Readonly<Record<string, Uint8Array>>;
}

const ZIP_EPOCH = new Date("1980-01-02T12:00:00Z");
export const DROWSE_ARCHIVE_IN_MEMORY_BUILD_MAX_BYTES = 64 * 1024 * 1024;

export async function buildDrowseArchive(input: BuildDrowseArchiveInput): Promise<Blob> {
  const paths = Object.keys(input.files).sort();
  if (paths.length === 0 || paths.length + 1 > DROWSE_ARCHIVE_MAX_ENTRIES) {
    throw new DrowseArchiveError(
      "ARCHIVE_LIMIT_EXCEEDED",
      `Browser-authored Drowse packs need 1-${DROWSE_ARCHIVE_MAX_ENTRIES - 1} payload files`,
    );
  }
  let payloadBytes = 0;
  for (const path of paths) {
    const bytes = input.files[path];
    if (!(bytes instanceof Uint8Array) || bytes.length > DROWSE_ARCHIVE_MAX_FILE_BYTES) {
      throw new DrowseArchiveError(
        "ARCHIVE_LIMIT_EXCEEDED",
        `Browser-authored Drowse pack file ${JSON.stringify(path)} is too large`,
        path,
      );
    }
    payloadBytes += bytes.length;
    if (payloadBytes > DROWSE_ARCHIVE_IN_MEMORY_BUILD_MAX_BYTES) {
      throw new DrowseArchiveError(
        "ARCHIVE_LIMIT_EXCEEDED",
        "Browser-authored Drowse packs cannot exceed 64 MiB before streaming export is available",
      );
    }
  }
  const files = Object.fromEntries(paths.map((path) => {
    const bytes = input.files[path];
    return [path, { size: bytes.length, sha256: bytesToHex(sha256(bytes)) }];
  }));
  const manifest: DrowseArchiveManifest = {
    format_version: 1,
    kind: "drowse-manifold",
    producer: { name: "drowse", version: input.producerVersion },
    primary: input.primary,
    template: input.template,
    source: { ...input.source },
    files,
  };
  const manifestBytes = jsonBytes(manifest);
  if (manifestBytes.length > DROWSE_ARCHIVE_MANIFEST_MAX_BYTES) {
    throw new DrowseArchiveError("ARCHIVE_LIMIT_EXCEEDED", "Browser-authored pack.json is too large");
  }
  const entries = Object.fromEntries([
    ["pack.json", [manifestBytes, { level: 0, mtime: ZIP_EPOCH }]],
    ...paths.map((path) => [
      path,
      [ownedBytes(input.files[path]), { level: 0, mtime: ZIP_EPOCH }],
    ]),
  ]);
  const archive = zipSync(entries, { level: 0, mtime: ZIP_EPOCH });
  const blob = new Blob([ownedBytes(archive)], { type: "application/vnd.drowse.pack+zip" });
  await validateDrowseArchive(blob);
  return blob;
}

export function jsonBytes(value: unknown): Uint8Array {
  return new TextEncoder().encode(`${JSON.stringify(sortJson(value))}\n`);
}

function sortJson(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortJson);
  if (value === null || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, sortJson(item)]),
  );
}

function ownedBytes(bytes: Uint8Array): Uint8Array<ArrayBuffer> {
  const owned = new Uint8Array(bytes.length);
  owned.set(bytes);
  return owned;
}
