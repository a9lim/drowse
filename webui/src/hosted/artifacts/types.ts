export const DROWSE_ARCHIVE_FORMAT_VERSION = 1;
export const DROWSE_ARCHIVE_MAX_ENTRIES = 4_096;
export const DROWSE_ARCHIVE_MAX_FILE_BYTES = 512 * 1024 * 1024;
export const DROWSE_ARCHIVE_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024;
export const DROWSE_ARCHIVE_MAX_COMPRESSION_RATIO = 100;
export const DROWSE_ARCHIVE_MANIFEST_MAX_BYTES = 1024 * 1024;

export type DrowseArchiveErrorCode =
  | "ARCHIVE_INVALID"
  | "ARCHIVE_LIMIT_EXCEEDED"
  | "ARCHIVE_PATH_INVALID"
  | "ARCHIVE_PATH_COLLISION"
  | "ARCHIVE_ENTRY_UNSAFE"
  | "ARCHIVE_COMPRESSION_UNSUPPORTED"
  | "MANIFEST_INVALID"
  | "MANIFEST_MISMATCH"
  | "CHECKSUM_MISMATCH"
  | "CLOSURE_INVALID"
  | "SAFETENSORS_INVALID"
  | "JSON_NOT_UTF8"
  | "JSON_INVALID"
  | "JSON_DUPLICATE_KEY"
  | "STAGE_FAILED";

export class DrowseArchiveError extends Error {
  readonly code: DrowseArchiveErrorCode;
  readonly path?: string;
  readonly status: number;
  readonly recoverable = true;

  constructor(code: DrowseArchiveErrorCode, message: string, path?: string) {
    super(message);
    this.name = "DrowseArchiveError";
    this.code = code;
    this.path = path;
    this.status = code === "STAGE_FAILED" ? 500 : 400;
  }
}

export interface DrowseArchiveFileRecord {
  size: number;
  sha256: string;
}

export interface DrowseArchiveManifest {
  format_version: 1;
  kind: "drowse-manifold";
  producer: { name: "drowse"; version: string };
  primary: string;
  template: string | null;
  source: {
    uri: string;
    repository: string | null;
    revision: string | null;
  };
  files: Record<string, DrowseArchiveFileRecord>;
}

export type DrowseArchiveSource = Blob | ArrayBuffer | Uint8Array;

export interface DrowseArchiveArchiveEntry {
  readonly path: string;
  readonly size: number;
  readonly compressedSize: number;
  readonly compressionMethod: 0 | 8;
}

export interface InspectedDrowseArchive {
  readonly manifest: Readonly<DrowseArchiveManifest>;
  readonly exactManifestBytes: Uint8Array;
  readonly entries: readonly DrowseArchiveArchiveEntry[];
  readonly expandedBytes: number;
  readonly archiveBytes: number;
  readonly primaryIdentity: readonly [namespace: string, name: string];
  readonly templateIdentity: readonly [namespace: string, name: string] | null;
}

export interface DrowseArchiveStage {
  begin(pack: InspectedDrowseArchive): void | Promise<void>;
  write(path: string, chunk: Uint8Array): void | Promise<void>;
  finish(path: string): void | Promise<void>;
  commit(pack: VerifiedDrowseArchive): void | Promise<void>;
  rollback(error: unknown): void | Promise<void>;
}

export interface DrowseArchiveProgress {
  phase: "inspecting" | "verifying" | "validating" | "committing";
  path: string | null;
  verifiedBytes: number;
  totalBytes: number;
}

export interface VerifyDrowseArchiveOptions {
  stage?: DrowseArchiveStage;
  signal?: AbortSignal;
  onProgress?: (progress: DrowseArchiveProgress) => void;
}

export interface DrowseArchiveFittedArtifact {
  readonly tensorPath: string;
  readonly sidecarPath: string;
  readonly modelId: string;
  readonly variant: "raw" | "sae" | "from";
  readonly variantIdentity: string | null;
  readonly modelFingerprint: string | null;
  readonly contextBindingSha256: string | null;
  readonly modelSourceFingerprint: string | null;
}

export interface VerifiedDrowseArchive extends InspectedDrowseArchive {
  readonly verifiedAt: number;
  readonly files: Readonly<Record<string, DrowseArchiveFileRecord>>;
  readonly manifold: Readonly<Record<string, unknown>>;
  readonly template: Readonly<Record<string, unknown>> | null;
  readonly fittedArtifacts: readonly Readonly<DrowseArchiveFittedArtifact>[];
  readonly fittedRuntimeFingerprints: readonly string[];
}
