import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import { DownloadEtaEstimator, type EtaSnapshot } from "../../lib/runtime/eta";
import {
  parseContentRange,
  validateRangeResponse,
} from "../../lib/runtime/range";
import type {
  CatalogFile,
  CatalogInstrumentPack,
  DownloadFileProgress,
  DownloadProgress,
  ModelVariant,
  RuntimeFailure,
  VerifiedCatalog,
} from "../../lib/runtime/contracts";
import { CatalogValidationError } from "../../lib/runtime/catalog";
import type {
  ContentAddressedStore,
  ContentObjectDescriptor,
  ContentReservationRecord,
  ContentWriteSession,
  PartialContentOwner,
  ValidPartialOwner,
} from "./contentStore";
import {
  CONTENT_LIFECYCLE_LOCK,
  CONTENT_OBJECT_LOCK_PREFIX,
} from "./contentStore";
import { HOSTED_DESTRUCTIVE_LOCK } from "./ownership";
import { randomUuid } from "./randomId";

const DEFAULT_CHUNK_BYTES = 256 * 1024;
const MAX_PARALLEL_DOWNLOADS = 4;
const RETAINED_PROBE_BYTES = 8 * 1024 * 1024;
const ETA_TICK_MS = 1_000;
const MAX_CURRENT_OBJECT_RESTARTS = 3;
const RESERVATION_LEASE_MS = 2 * 60 * 1_000;
const RESERVATION_HEARTBEAT_MS = 30_000;
const DEFAULT_DURABLE_CHECKPOINT_BYTES = 4 * 1024 * 1024;
const DEFAULT_DURABLE_CHECKPOINT_INTERVAL_MS = 10_000;

export type ArtifactDownloadErrorCode =
  | "CATALOG_DOWNLOADS_DISABLED"
  | "MODEL_VARIANT_NOT_FOUND"
  | "MODEL_VARIANT_AMBIGUOUS"
  | "REQUIRED_CORE_PACK_INVALID"
  | "OPTIONAL_PACK_NOT_FOUND"
  | "OPTIONAL_PACK_INVALID"
  | "DOWNLOAD_BUSY"
  | "DOWNLOAD_CANCELLED"
  | "DOWNLOAD_QUOTA_INSUFFICIENT"
  | "DOWNLOAD_QUOTA_CHECK_FAILED"
  | "DOWNLOAD_NETWORK_ERROR"
  | "DOWNLOAD_REDIRECT_REJECTED"
  | "DOWNLOAD_HTTP_REJECTED"
  | "DOWNLOAD_RANGE_INVALID"
  | "DOWNLOAD_BODY_MISSING"
  | "DOWNLOAD_SIZE_MISMATCH"
  | "DOWNLOAD_HASH_MISMATCH"
  | "CONTENT_STORE_FAILURE"
  | "DOWNLOAD_FAILED";

export class ArtifactDownloadError extends Error {
  readonly code: ArtifactDownloadErrorCode;
  readonly status: number;
  readonly recoverable: boolean;
  readonly detail?: unknown;

  constructor(
    code: ArtifactDownloadErrorCode,
    message: string,
    options: {
      status: number;
      recoverable: boolean;
      detail?: unknown;
      cause?: unknown;
    },
  ) {
    super(message, { cause: options.cause });
    this.name = "ArtifactDownloadError";
    this.code = code;
    this.status = options.status;
    this.recoverable = options.recoverable;
    this.detail = options.detail;
  }

  toRuntimeFailure(): RuntimeFailure {
    return {
      code: this.code,
      message: this.message,
      recoverable: this.recoverable,
      status: this.status,
      ...(this.detail === undefined ? {} : { detail: this.detail }),
    };
  }
}

interface ArtifactManifestSelection {
  variant: ModelVariant;
  files: CatalogFile[];
  totalBytes: number;
  installId: string;
  installKind: "model" | "pack";
  packId?: string;
  partialOwner: PartialContentOwner;
  validPartialOwners: ValidPartialOwner[];
}

export interface DownloadManifestSelection extends ArtifactManifestSelection {
  requiredCorePack: CatalogInstrumentPack;
  installKind: "model";
}

export interface OptionalPackDownloadManifestSelection extends ArtifactManifestSelection {
  pack: CatalogInstrumentPack;
  packId: string;
  installKind: "pack";
}

export interface DownloadQuotaContext {
  modelVariantId: string;
  packId?: string;
  file: CatalogFile;
  fileIndex: number;
  fileCount: number;
  remainingFileBytes: number;
  remainingManifestBytes: number;
}

export interface ArtifactDownloadOptions {
  checkQuota(
    context: DownloadQuotaContext,
  ): boolean | void | Promise<boolean | void>;
  onProgress?: (progress: DownloadProgress) => void;
  signal?: AbortSignal;
  retainFirstRangeProbe?: boolean;
  authorizeSelection?(signal: AbortSignal): void | Promise<void>;
  withCatalogAuthorization?<T>(
    operation: () => Promise<T>,
    signal: AbortSignal,
  ): Promise<T>;
}

export interface ArtifactDownloadResult {
  modelVariantId: string;
  packId?: string;
  objectHashes: string[];
  downloadedBytes: number;
  totalBytes: number;
  installedAt: number;
}

export type ArtifactFetch = (
  url: string,
  init: RequestInit,
) => Promise<Response>;

export interface DownloadScheduler {
  setInterval(callback: () => void, delayMs: number): unknown;
  clearInterval(handle: unknown): void;
}

export interface ArtifactDownloaderDependencies {
  fetch?: ArtifactFetch;
  monotonicNow?: () => number;
  wallNow?: () => number;
  isOnline?: () => boolean;
  waitUntilOnline?: (signal: AbortSignal) => Promise<void>;
  scheduler?: DownloadScheduler;
  chunkBytes?: number;
  durableCheckpointBytes?: number;
  durableCheckpointIntervalMs?: number;
  maxParallelDownloads?: number;
  allowedRedirectOrigins?: readonly string[];
  runObjectExclusive?: ObjectExclusiveRunner;
  runDownloadShared?: DownloadSharedRunner;
}

export type ObjectExclusiveRunner = <T>(
  sha256: string,
  signal: AbortSignal,
  operation: () => Promise<T>,
) => Promise<T>;

export type DownloadSharedRunner = <T>(
  signal: AbortSignal,
  operation: () => Promise<T>,
) => Promise<T>;

interface ActiveDownload {
  targetId: string;
  controller: AbortController;
  phase: "transferring" | "committing";
}

interface StoredFileState {
  bytes: number;
  verified: boolean;
}

interface ProgressTracker {
  setFile(
    index: number,
    bytesReceived: number,
    resumable: boolean | null,
    verification: DownloadFileProgress["verification"],
  ): void;
  recordChunk(index: number, bytesReceived: number, bytesTransferred: number): void;
  setNetworkActive(active: boolean): void;
  get downloadedBytes(): number;
  complete(): void;
  dispose(): void;
}

export function selectDownloadManifest(
  catalog: VerifiedCatalog,
  modelVariantId: string,
  options: { requireDownloadAuthorization?: boolean } = {},
): DownloadManifestSelection {
  if (options.requireDownloadAuthorization !== false && !catalog.allowDownloads) {
    throw downloadError(
      "CATALOG_DOWNLOADS_DISABLED",
      "This catalog may load installed artifacts but cannot authorize new downloads",
      409,
      false,
    );
  }
  const variants = catalog.document.models.flatMap((model) => model.variants)
    .filter((variant) => variant.id === modelVariantId);
  if (variants.length === 0) {
    throw downloadError(
      "MODEL_VARIANT_NOT_FOUND",
      `Model variant ${modelVariantId} is not present in the verified catalog`,
      404,
      false,
    );
  }
  if (variants.length > 1) {
    throw downloadError(
      "MODEL_VARIANT_AMBIGUOUS",
      `Model variant ${modelVariantId} is not unique in the verified catalog`,
      409,
      false,
    );
  }
  const variant = variants[0];
  const corePacks = variant.packs.filter((pack) => pack.kind === "core");
  if (
    corePacks.length !== 1 || !corePacks[0].required ||
    variant.packs.some((pack) => pack.required && pack !== corePacks[0])
  ) {
    throw downloadError(
      "REQUIRED_CORE_PACK_INVALID",
      `Model variant ${modelVariantId} must have exactly one required core pack`,
      422,
      false,
    );
  }
  const requiredCorePack = corePacks[0];
  const files = [...variant.files, ...requiredCorePack.files];
  const totalBytes = sumBytes(files);
  if (
    sumBytes(variant.files) !== variant.downloadBytes ||
    sumBytes(requiredCorePack.files) !== requiredCorePack.bytes ||
    requiredCorePack.bytes !== variant.requiredCorePackBytes ||
    totalBytes !== variant.downloadBytes + variant.requiredCorePackBytes
  ) {
    throw downloadError(
      "REQUIRED_CORE_PACK_INVALID",
      `Model variant ${modelVariantId} has inconsistent signed byte totals`,
      422,
      false,
    );
  }
  return {
    variant,
    requiredCorePack,
    files,
    totalBytes,
    installId: variant.id,
    installKind: "model",
    partialOwner: {
      id: variant.id,
      kind: "model",
      catalogSequence: catalog.document.sequence,
    },
    validPartialOwners: validPartialOwnersForCatalog(catalog),
  };
}

export function selectOptionalPackManifest(
  catalog: VerifiedCatalog,
  modelVariantId: string,
  packId: string,
): OptionalPackDownloadManifestSelection {
  if (!catalog.allowDownloads) {
    throw downloadError(
      "CATALOG_DOWNLOADS_DISABLED",
      "This catalog may load installed artifacts but cannot authorize new downloads",
      409,
      false,
    );
  }
  const variants = catalog.document.models.flatMap((model) => model.variants)
    .filter((variant) => variant.id === modelVariantId);
  if (variants.length === 0) {
    throw downloadError(
      "MODEL_VARIANT_NOT_FOUND",
      `Model variant ${modelVariantId} is not present in the verified catalog`,
      404,
      false,
    );
  }
  if (variants.length > 1) {
    throw downloadError(
      "MODEL_VARIANT_AMBIGUOUS",
      `Model variant ${modelVariantId} is not unique in the verified catalog`,
      409,
      false,
    );
  }
  const variant = variants[0];
  const pack = variant.packs.find((candidate) => candidate.id === packId);
  if (!pack) {
    throw downloadError(
      "OPTIONAL_PACK_NOT_FOUND",
      `Instrument pack ${packId} does not belong to model variant ${modelVariantId}`,
      404,
      false,
    );
  }
  if (
    pack.required || pack.kind === "core" || pack.files.length === 0 ||
    pack.files.some((file) => file.role !== "instrument") ||
    sumBytes(pack.files) !== pack.bytes
  ) {
    throw downloadError(
      "OPTIONAL_PACK_INVALID",
      `Instrument pack ${packId} is not an optional downloadable pack`,
      422,
      false,
    );
  }
  return {
    variant,
    pack,
    packId: pack.id,
    files: pack.files,
    totalBytes: pack.bytes,
    installId: pack.id,
    installKind: "pack",
    partialOwner: {
      id: pack.id,
      kind: "pack",
      catalogSequence: catalog.document.sequence,
    },
    validPartialOwners: validPartialOwnersForCatalog(catalog),
  };
}

export class VerifiedArtifactDownloader {
  private readonly fetcher: ArtifactFetch;
  private readonly monotonicNow: () => number;
  private readonly wallNow: () => number;
  private readonly isOnline: () => boolean;
  private readonly waitUntilOnline: (signal: AbortSignal) => Promise<void>;
  private readonly scheduler: DownloadScheduler;
  private readonly chunkBytes: number;
  private readonly durableCheckpointBytes: number;
  private readonly durableCheckpointIntervalMs: number;
  private readonly maxParallelDownloads: number;
  private readonly allowedRedirectOrigins: ReadonlySet<string>;
  private readonly runObjectExclusive: ObjectExclusiveRunner;
  private readonly runDownloadShared: DownloadSharedRunner;
  private active: ActiveDownload | null = null;

  constructor(
    private readonly store: ContentAddressedStore,
    dependencies: ArtifactDownloaderDependencies = {},
  ) {
    this.fetcher = dependencies.fetch ?? ((url, init) => globalThis.fetch(url, init));
    this.monotonicNow = dependencies.monotonicNow ?? (() => performance.now());
    this.wallNow = dependencies.wallNow ?? (() => Date.now());
    this.isOnline = dependencies.isOnline ?? (() =>
      typeof navigator === "undefined" || navigator.onLine !== false
    );
    this.waitUntilOnline = dependencies.waitUntilOnline ?? defaultWaitUntilOnline;
    this.scheduler = dependencies.scheduler ?? {
      setInterval: (callback, delayMs) => globalThis.setInterval(callback, delayMs),
      clearInterval: (handle) => globalThis.clearInterval(handle as number),
    };
    this.chunkBytes = dependencies.chunkBytes ?? DEFAULT_CHUNK_BYTES;
    if (!Number.isSafeInteger(this.chunkBytes) || this.chunkBytes <= 0) {
      throw new RangeError("chunkBytes must be a positive integer");
    }
    this.durableCheckpointBytes = dependencies.durableCheckpointBytes ??
      DEFAULT_DURABLE_CHECKPOINT_BYTES;
    this.durableCheckpointIntervalMs = dependencies.durableCheckpointIntervalMs ??
      DEFAULT_DURABLE_CHECKPOINT_INTERVAL_MS;
    this.maxParallelDownloads = dependencies.maxParallelDownloads ?? MAX_PARALLEL_DOWNLOADS;
    if (
      !Number.isSafeInteger(this.durableCheckpointBytes) ||
      this.durableCheckpointBytes <= 0
    ) {
      throw new RangeError("durableCheckpointBytes must be a positive integer");
    }
    if (
      !Number.isFinite(this.durableCheckpointIntervalMs) ||
      this.durableCheckpointIntervalMs <= 0
    ) {
      throw new RangeError("durableCheckpointIntervalMs must be a positive number");
    }
    if (!Number.isSafeInteger(this.maxParallelDownloads) || this.maxParallelDownloads <= 0) {
      throw new RangeError("maxParallelDownloads must be a positive integer");
    }
    this.allowedRedirectOrigins = new Set(
      (dependencies.allowedRedirectOrigins ?? []).map(validateRedirectOrigin),
    );
    this.runObjectExclusive = dependencies.runObjectExclusive ?? defaultObjectExclusive;
    this.runDownloadShared = dependencies.runDownloadShared ?? defaultDownloadShared;
  }

  get isDownloading(): boolean {
    return this.active !== null;
  }

  cancel(): boolean {
    if (!this.active || this.active.phase === "committing") return false;
    this.active.controller.abort();
    return true;
  }

  async download(
    catalog: VerifiedCatalog,
    modelVariantId: string,
    options: ArtifactDownloadOptions,
  ): Promise<ArtifactDownloadResult> {
    const selection = selectDownloadManifest(catalog, modelVariantId);
    return this.downloadSelection(selection, options);
  }

  async downloadPack(
    catalog: VerifiedCatalog,
    modelVariantId: string,
    packId: string,
    options: ArtifactDownloadOptions,
  ): Promise<ArtifactDownloadResult> {
    const selection = selectOptionalPackManifest(catalog, modelVariantId, packId);
    return this.downloadSelection(selection, options);
  }

  private async downloadSelection(
    selection: ArtifactManifestSelection,
    options: ArtifactDownloadOptions,
  ): Promise<ArtifactDownloadResult> {
    if (this.active) {
      throw downloadError(
        "DOWNLOAD_BUSY",
        `${this.active.targetId} is already downloading`,
        409,
        true,
      );
    }
    const controller = new AbortController();
    const abortFromCaller = (): void => controller.abort();
    if (options.signal?.aborted) controller.abort();
    else options.signal?.addEventListener("abort", abortFromCaller, { once: true });
    this.active = {
      targetId: selection.installId,
      controller,
      phase: "transferring",
    };

    try {
      return await this.runDownloadShared(
        controller.signal,
        () => options.withCatalogAuthorization
          ? options.withCatalogAuthorization(
              async () => {
                await options.authorizeSelection?.(controller.signal);
                return this.runDownload(selection, options, controller.signal);
              },
              controller.signal,
            )
          : (async () => {
              await options.authorizeSelection?.(controller.signal);
              return this.runDownload(selection, options, controller.signal);
            })(),
      );
    } catch (error) {
      if (error instanceof ArtifactDownloadError || error instanceof CatalogValidationError) {
        throw error;
      }
      if (isRuntimeFailure(error)) throw error;
      if (controller.signal.aborted || isAbortError(error)) {
        throw cancelledError(selection.installId, error);
      }
      throw downloadError(
        "DOWNLOAD_FAILED",
        errorMessage(error, "The artifact download failed"),
        500,
        true,
        undefined,
        error,
      );
    } finally {
      options.signal?.removeEventListener("abort", abortFromCaller);
      this.active = null;
    }
  }

  private async runDownload(
    selection: ArtifactManifestSelection,
    options: ArtifactDownloadOptions,
    signal: AbortSignal,
  ): Promise<ArtifactDownloadResult> {
    try {
      await this.store.initialize();
    } catch (error) {
      throw downloadError(
        "CONTENT_STORE_FAILURE",
        errorMessage(error, "Browser content storage could not be initialized"),
        500,
        true,
        undefined,
        error,
      );
    }
    throwIfAborted(signal, selection.installId);
    const reservationCreatedAt = this.wallNow();
    const reservation: Omit<ContentReservationRecord, "updatedAt" | "expiresAt"> = {
      id: `download:${randomUuid()}`,
      ownerId: randomUuid(),
      ownerKind: "download",
      targetId: selection.installId,
      targetKind: selection.installKind,
      objectHashes: selection.files.map((file) => file.sha256),
      createdAt: reservationCreatedAt,
    };
    await this.reserveSelection(reservation);
    let reservationHeartbeat = Promise.resolve();
    const reservationHeartbeatHandle = this.scheduler.setInterval(() => {
      reservationHeartbeat = reservationHeartbeat.then(
        () => this.reserveSelection(reservation),
      ).catch((error) => {
        this.active?.controller.abort(error);
      });
    }, RESERVATION_HEARTBEAT_MS);
    try {
      try {
        throwIfAborted(signal, selection.installId);
        await this.store.collectGarbage({
          validPartialOwners: selection.validPartialOwners,
          activeReservationIds: [reservation.id],
        });
      } catch (error) {
        if (error instanceof ArtifactDownloadError) throw error;
        throw downloadError(
          "CONTENT_STORE_FAILURE",
          errorMessage(error, "Browser content storage could not be reconciled"),
          500,
          true,
          undefined,
          error,
        );
      }
      throwIfAborted(signal, selection.installId);
      const storedFiles = await this.inspectStoredFiles(selection, signal);
      const tracker = this.createProgressTracker(
        selection,
        options.onProgress,
        storedFiles,
      );
      try {
        const uniqueIndexes: number[] = [];
        const scheduledHashes = new Set<string>();
        for (let index = 0; index < selection.files.length; index += 1) {
          const sha256Digest = selection.files[index].sha256;
          if (scheduledHashes.has(sha256Digest)) continue;
          scheduledHashes.add(sha256Digest);
          uniqueIndexes.push(index);
        }
        const transferController = new AbortController();
        const abortTransfers = () => transferController.abort(signal.reason);
        signal.addEventListener("abort", abortTransfers, { once: true });
        let cursor = 0;
        let firstError: unknown = null;
        const downloadNext = async (): Promise<void> => {
          while (!transferController.signal.aborted) {
            const queueIndex = cursor;
            cursor += 1;
            if (queueIndex >= uniqueIndexes.length) return;
            const index = uniqueIndexes[queueIndex];
            const file = selection.files[index];
            try {
              throwIfAborted(transferController.signal, selection.installId);
              await this.reserveSelection(reservation);
              await this.checkQuota(
                selection,
                file,
                index,
                storedFiles,
                options.checkQuota,
                transferController.signal,
              );
              try {
                await this.downloadFile(
                  file,
                  index,
                  index === 0 && options.retainFirstRangeProbe === true,
                  selection.installId,
                  selection.partialOwner,
                  () => this.reserveSelection(reservation),
                  tracker,
                  transferController.signal,
                );
                for (let duplicateIndex = 0; duplicateIndex < selection.files.length; duplicateIndex += 1) {
                  if (selection.files[duplicateIndex].sha256 !== file.sha256) continue;
                  storedFiles[duplicateIndex] = {
                    bytes: selection.files[duplicateIndex].bytes,
                    verified: true,
                  };
                  if (duplicateIndex !== index) {
                    tracker.setFile(
                      duplicateIndex,
                      selection.files[duplicateIndex].bytes,
                      false,
                      "verified",
                    );
                  }
                }
              } catch (error) {
                const cancelled = transferController.signal.aborted || isAbortError(error) ||
                  (error instanceof ArtifactDownloadError && error.code === "DOWNLOAD_CANCELLED");
                if (cancelled) {
                  const partial = await this.safePartialSize(file.sha256);
                  tracker.setFile(index, partial, null, "pending");
                  throw cancelledError(selection.installId, error);
                }
                const partial = await this.safePartialSize(file.sha256);
                storedFiles[index] = { bytes: partial, verified: false };
                tracker.setFile(index, partial, null, "failed");
                if (isQuotaExceededError(error)) {
                  await this.checkQuota(
                    selection,
                    file,
                    index,
                    storedFiles,
                    options.checkQuota,
                    transferController.signal,
                  );
                  throw downloadError(
                    "DOWNLOAD_QUOTA_INSUFFICIENT",
                    `Browser storage filled while writing ${artifactLabel(file)}`,
                    507,
                    true,
                    { path: file.path, bytes: partial },
                    error,
                  );
                }
                throw error;
              }
            } catch (error) {
              if (firstError === null) {
                firstError = error;
                transferController.abort(error);
              }
              return;
            }
          }
        };
        try {
          await Promise.all(
            Array.from(
              { length: Math.min(this.maxParallelDownloads, uniqueIndexes.length) },
              () => downloadNext(),
            ),
          );
        } finally {
          signal.removeEventListener("abort", abortTransfers);
        }
        if (firstError !== null) throw firstError;
        throwIfAborted(signal, selection.installId);

        const installedAt = this.wallNow();
        throwIfAborted(signal, selection.installId);
        if (this.active) this.active.phase = "committing";
        try {
          await this.store.registerInstall({
            id: selection.installId,
            kind: selection.installKind,
            objectHashes: selection.files.map((file) => file.sha256),
            installedAt,
          }, {
            selectAsCurrentModel: selection.installKind === "model",
          });
        } catch (error) {
          throw downloadError(
            "CONTENT_STORE_FAILURE",
            errorMessage(error, "The verified artifact could not be registered as installed"),
            500,
            true,
            undefined,
            error,
          );
        }
        tracker.complete();
        return {
          modelVariantId: selection.variant.id,
          ...(selection.packId === undefined ? {} : { packId: selection.packId }),
          objectHashes: selection.files.map((file) => file.sha256),
          downloadedBytes: tracker.downloadedBytes,
          totalBytes: selection.totalBytes,
          installedAt,
        };
      } finally {
        tracker.dispose();
      }
    } finally {
      this.scheduler.clearInterval(reservationHeartbeatHandle);
      await reservationHeartbeat;
      try {
        await this.store.releaseReservation(reservation.id);
      } catch {}
    }
  }

  private async reserveSelection(
    reservation: Omit<ContentReservationRecord, "updatedAt" | "expiresAt">,
  ): Promise<void> {
    try {
      const updatedAt = Math.max(reservation.createdAt, this.wallNow());
      await this.store.reserveObjects({
        ...reservation,
        updatedAt,
        expiresAt: updatedAt + RESERVATION_LEASE_MS,
      });
    } catch (error) {
      throw downloadError(
        "CONTENT_STORE_FAILURE",
        errorMessage(error, "The download objects could not be reserved"),
        500,
        true,
        {
          targetId: reservation.targetId,
          targetKind: reservation.targetKind,
        },
        error,
      );
    }
  }

  private async checkQuota(
    selection: ArtifactManifestSelection,
    file: CatalogFile,
    index: number,
    storedFiles: readonly StoredFileState[],
    check: ArtifactDownloadOptions["checkQuota"],
    signal: AbortSignal,
  ): Promise<void> {
    const remainingFileBytes = Math.max(0, file.bytes - storedFiles[index].bytes);
    const remainingManifestBytes = remainingContentBytes(
      selection.files,
      (manifestIndex) => storedFiles[manifestIndex].bytes,
    );
    let allowed: boolean | void;
    try {
      allowed = await abortable(
        Promise.resolve(check({
          modelVariantId: selection.variant.id,
          ...(selection.packId === undefined ? {} : { packId: selection.packId }),
          file,
          fileIndex: index,
          fileCount: selection.files.length,
          remainingFileBytes,
          remainingManifestBytes,
        })),
        signal,
      );
    } catch (error) {
      if (error instanceof ArtifactDownloadError) throw error;
      if (signal.aborted || isAbortError(error)) {
        throw cancelledError(selection.installId, error);
      }
      throw downloadError(
        "DOWNLOAD_QUOTA_CHECK_FAILED",
        errorMessage(error, "Storage quota could not be checked"),
        500,
        true,
        { path: file.path },
        error,
      );
    }
    if (allowed === false && remainingManifestBytes > 0) {
      throw downloadError(
        "DOWNLOAD_QUOTA_INSUFFICIENT",
        `There is not enough browser storage for ${artifactLabel(file)}`,
        507,
        true,
        { path: file.path, bytes: remainingFileBytes, remainingManifestBytes },
      );
    }
  }

  private async inspectStoredFiles(
    selection: ArtifactManifestSelection,
    signal: AbortSignal,
  ): Promise<StoredFileState[]> {
    const stored: StoredFileState[] = [];
    const storedByHash = new Map<string, StoredFileState & { expectedBytes: number }>();
    try {
      for (const file of selection.files) {
        throwIfAborted(signal, selection.installId);
        const duplicate = storedByHash.get(file.sha256);
        if (duplicate) {
          if (duplicate.expectedBytes !== file.bytes) {
            throw new Error(`Content object ${file.sha256} has conflicting signed sizes`);
          }
          stored.push({ bytes: duplicate.bytes, verified: duplicate.verified });
          continue;
        }
        const record = await this.store.inspectObject(file.sha256);
        throwIfAborted(signal, selection.installId);
        if (record?.state === "verified") {
          const verified = record.expectedBytes === file.bytes
            ? await this.store.verifiedFile(file.sha256)
            : null;
          if (verified?.size === file.bytes) {
            const state = { bytes: file.bytes, verified: true, expectedBytes: file.bytes };
            storedByHash.set(file.sha256, state);
            stored.push({ bytes: state.bytes, verified: state.verified });
            continue;
          }
        }
        const compatiblePartial = record?.state === "partial" &&
          record.expectedBytes === file.bytes &&
          record.url === file.url &&
          record.revision === file.revision;
        const partial = compatiblePartial
          ? await this.store.partialFile(file.sha256)
          : null;
        throwIfAborted(signal, selection.installId);
        const state = {
          bytes: partial?.size ?? 0,
          verified: false,
          expectedBytes: file.bytes,
        };
        storedByHash.set(file.sha256, state);
        stored.push({ bytes: state.bytes, verified: state.verified });
      }
      return stored;
    } catch (error) {
      if (signal.aborted || isAbortError(error)) {
        throw cancelledError(selection.installId, error);
      }
      if (error instanceof ArtifactDownloadError) throw error;
      throw downloadError(
        "CONTENT_STORE_FAILURE",
        errorMessage(error, "Stored download progress could not be inspected"),
        500,
        true,
        {
          modelVariantId: selection.variant.id,
          ...(selection.packId === undefined ? {} : { packId: selection.packId }),
        },
        error,
      );
    }
  }

  private validateResponseOrigin(file: CatalogFile, response: Response): void {
    let finalUrl: URL;
    try {
      if (!response.url) throw new Error("The final response URL is unavailable");
      finalUrl = new URL(response.url);
    } catch (error) {
      throw downloadError(
        "DOWNLOAD_REDIRECT_REJECTED",
        `The final download URL for ${artifactLabel(file)} could not be verified`,
        502,
        false,
        { path: file.path, responseUrl: response.url || null },
        error,
      );
    }
    const signedOrigin = new URL(file.url).origin;
    if (
      finalUrl.protocol !== "https:" || finalUrl.username || finalUrl.password ||
      (finalUrl.origin !== signedOrigin &&
        !this.allowedRedirectOrigins.has(finalUrl.origin))
    ) {
      throw downloadError(
        "DOWNLOAD_REDIRECT_REJECTED",
        `The final download origin for ${artifactLabel(file)} is not allowed`,
        502,
        false,
        { path: file.path, signedOrigin, responseOrigin: finalUrl.origin },
      );
    }
  }

  private async downloadFile(
    file: CatalogFile,
    index: number,
    retainFirstRangeProbe: boolean,
    modelVariantId: string,
    partialOwner: PartialContentOwner,
    refreshReservation: () => Promise<void>,
    tracker: ProgressTracker,
    signal: AbortSignal,
  ): Promise<void> {
    return this.runObjectExclusive(file.sha256, signal, () => this.downloadFileLocked(
      file,
      index,
      retainFirstRangeProbe,
      modelVariantId,
      partialOwner,
      refreshReservation,
      tracker,
      signal,
    ));
  }

  private async downloadFileLocked(
    file: CatalogFile,
    index: number,
    retainFirstRangeProbe: boolean,
    modelVariantId: string,
    partialOwner: PartialContentOwner,
    refreshReservation: () => Promise<void>,
    tracker: ProgressTracker,
    signal: AbortSignal,
  ): Promise<void> {
    throwIfAborted(signal, modelVariantId);
    const verified = await this.store.verifiedFile(file.sha256);
    if (verified) {
      if (verified.size !== file.bytes) {
        throw downloadError(
          "CONTENT_STORE_FAILURE",
          `Verified object ${file.sha256} has the wrong size`,
          500,
          true,
          { path: file.path, expectedBytes: file.bytes, actualBytes: verified.size },
        );
      }
      tracker.setFile(index, file.bytes, false, "verified");
      return;
    }

    const descriptor = descriptorFor(file, null, partialOwner);
    let record = await this.store.beginPartial(descriptor);
    if (record.state === "verified") {
      const racedVerified = await this.store.verifiedFile(file.sha256);
      if (!racedVerified || racedVerified.size !== file.bytes) {
        throw downloadError(
          "CONTENT_STORE_FAILURE",
          `Verified object ${file.sha256} disappeared during download setup`,
          500,
          true,
          { path: file.path },
        );
      }
      tracker.setFile(index, file.bytes, false, "verified");
      return;
    }

    let offset = record.contiguousBytes;
    let hasher = sha256.create();
    tracker.setFile(index, offset, false, "hashing");
    if (offset > 0) {
      const prefix = await this.store.partialFile(file.sha256);
      if (!prefix || prefix.size !== offset) {
        throw downloadError(
          "CONTENT_STORE_FAILURE",
          `Stored partial ${file.sha256} is not contiguous`,
          500,
          true,
          { path: file.path, expectedBytes: offset, actualBytes: prefix?.size ?? null },
        );
      }
      for (let start = 0; start < offset; start += this.chunkBytes) {
        throwIfAborted(signal, modelVariantId);
        const end = Math.min(offset, start + this.chunkBytes);
        const chunk = new Uint8Array(await prefix.slice(start, end).arrayBuffer());
        hasher.update(chunk);
      }
    }

    let probeAttempted = false;
    let restarts = 0;
    while (offset < file.bytes) {
      throwIfAborted(signal, modelVariantId);
      const useProbe = retainFirstRangeProbe && !probeAttempted && offset === 0 &&
        file.bytes > RETAINED_PROBE_BYTES;
      const requestedEnd = useProbe ? RETAINED_PROBE_BYTES - 1 : null;
      if (useProbe) probeAttempted = true;
      const headers = new Headers();
      if (offset > 0 || requestedEnd !== null) {
        headers.set(
          "Range",
          requestedEnd === null
            ? `bytes=${offset}-`
            : `bytes=${offset}-${requestedEnd}`,
        );
      }
      if (offset > 0 && record.etag !== null) headers.set("If-Range", record.etag);

      tracker.setNetworkActive(true);
      let response: Response;
      try {
        response = await this.fetcher(file.url, {
          method: "GET",
          headers,
          cache: "no-store",
          credentials: "omit",
          redirect: "follow",
          signal,
        });
      } catch (error) {
        tracker.setNetworkActive(false);
        if (signal.aborted || isAbortError(error)) throw cancelledError(modelVariantId, error);
        if (!this.isOnline()) {
          await this.waitUntilOnline(signal);
          throwIfAborted(signal, modelVariantId);
          continue;
        }
        throw downloadError(
          "DOWNLOAD_NETWORK_ERROR",
          errorMessage(error, `Downloading ${artifactLabel(file)} failed`),
          503,
          true,
          { path: file.path, offset },
          error,
        );
      }
      try {
        this.validateResponseOrigin(file, response);
      } catch (error) {
        tracker.setNetworkActive(false);
        await cancelBody(response);
        throw error;
      }

      const responseEtag = response.headers.get("ETag");
      const contentLength = integerHeader(response.headers.get("Content-Length"));
      const validation = validateRangeResponse({
        status: response.status,
        requestedOffset: offset,
        requestedEnd,
        expectedTotalBytes: file.bytes,
        contentRange: response.headers.get("Content-Range"),
        contentLength,
        storedEtag: record.etag,
        responseEtag,
      });
      if (validation.action === "restart") {
        tracker.setNetworkActive(false);
        await cancelBody(response);
        restarts += 1;
        if (restarts > MAX_CURRENT_OBJECT_RESTARTS) {
          throw downloadError(
            "DOWNLOAD_RANGE_INVALID",
            validation.reason ?? `The server could not resume ${artifactLabel(file)}`,
            502,
            true,
            { path: file.path, offset, status: response.status },
          );
        }
        record = await this.store.resetPartial(
          descriptorFor(file, responseEtag, partialOwner),
        );
        offset = 0;
        hasher = sha256.create();
        tracker.setFile(index, 0, false, "hashing");
        continue;
      }
      if (validation.action === "reject") {
        tracker.setNetworkActive(false);
        await cancelBody(response);
        throw downloadError(
          "DOWNLOAD_HTTP_REJECTED",
          validation.reason ?? `The server rejected ${artifactLabel(file)}`,
          502,
          true,
          { path: file.path, offset, status: response.status },
        );
      }
      if (!response.body) {
        tracker.setNetworkActive(false);
        throw downloadError(
          "DOWNLOAD_BODY_MISSING",
          `The response for ${artifactLabel(file)} has no body`,
          502,
          true,
          { path: file.path, offset, status: response.status },
        );
      }

      record = await this.store.beginPartial(
        descriptorFor(file, responseEtag ?? record.etag, partialOwner),
      );
      if (record.state !== "partial" || record.contiguousBytes !== offset) {
        tracker.setNetworkActive(false);
        await cancelBody(response);
        throw downloadError(
          "CONTENT_STORE_FAILURE",
          `Stored partial ${file.sha256} changed while downloading`,
          409,
          true,
          { path: file.path, expectedOffset: offset, actualOffset: record.contiguousBytes },
        );
      }

      const parsedRange = response.status === 206
        ? parseContentRange(response.headers.get("Content-Range"))
        : null;
      const expectedResponseBytes = parsedRange
        ? parsedRange.end - parsedRange.start + 1
        : contentLength;
      const reader = response.body.getReader();
      let writer: ContentWriteSession;
      try {
        writer = await this.store.openWrite(file.sha256, offset);
      } catch (error) {
        tracker.setNetworkActive(false);
        try {
          await reader.cancel();
        } catch {}
        reader.releaseLock();
        throw downloadError(
          "CONTENT_STORE_FAILURE",
          errorMessage(error, `${artifactLabel(file)} could not be opened for download`),
          500,
          true,
          { path: file.path, offset },
          error,
        );
      }
      let responseBytes = 0;
      let responseComplete = false;
      let offlineStreamFailure: unknown = null;
      let durableOffset = offset;
      let durableAt = this.wallNow();
      tracker.setFile(index, offset, validation.resumable, "hashing");
      try {
        while (true) {
          throwIfAborted(signal, modelVariantId);
          const { done, value } = await reader.read();
          if (done) {
            responseComplete = true;
            break;
          }
          if (!value || value.byteLength === 0) continue;
          for (let start = 0; start < value.byteLength; start += this.chunkBytes) {
            throwIfAborted(signal, modelVariantId);
            const chunk = value.subarray(
              start,
              Math.min(value.byteLength, start + this.chunkBytes),
            );
            if (
              (expectedResponseBytes !== null &&
                responseBytes + chunk.byteLength > expectedResponseBytes) ||
              offset + chunk.byteLength > file.bytes
            ) {
              throw downloadError(
                "DOWNLOAD_SIZE_MISMATCH",
                `The response for ${artifactLabel(file)} exceeds its signed size`,
                502,
                true,
                { path: file.path, expectedBytes: file.bytes },
              );
            }
            try {
              offset = await writer.write(chunk);
            } catch (error) {
              throw downloadError(
                "CONTENT_STORE_FAILURE",
                errorMessage(error, `${artifactLabel(file)} could not be saved`),
                500,
                true,
                { path: file.path, offset },
                error,
              );
            }
            responseBytes += chunk.byteLength;
            hasher.update(chunk);
            tracker.recordChunk(index, offset, chunk.byteLength);
            const checkpointAt = this.wallNow();
            if (
              offset < file.bytes &&
              (offset - durableOffset >= this.durableCheckpointBytes ||
                checkpointAt - durableAt >= this.durableCheckpointIntervalMs)
            ) {
              try {
                offset = await writer.checkpoint();
                await refreshReservation();
                writer = await this.store.openWrite(file.sha256, offset);
              } catch (error) {
                throw downloadError(
                  "CONTENT_STORE_FAILURE",
                  errorMessage(error, `${artifactLabel(file)} could not save its download progress`),
                  500,
                  true,
                  { path: file.path, offset },
                  error,
                );
              }
              durableOffset = offset;
              durableAt = checkpointAt;
            }
          }
        }
      } catch (error) {
        if (signal.aborted || isAbortError(error)) throw cancelledError(modelVariantId, error);
        if (error instanceof ArtifactDownloadError) throw error;
        if (!this.isOnline()) offlineStreamFailure = error;
        else {
          throw downloadError(
            "DOWNLOAD_NETWORK_ERROR",
            errorMessage(error, `Downloading ${artifactLabel(file)} was interrupted`),
            503,
            true,
            { path: file.path, offset },
            error,
          );
        }
      } finally {
        tracker.setNetworkActive(false);
        if (!responseComplete) {
          try {
            await reader.cancel();
          } catch {}
        }
        reader.releaseLock();
        try {
          offset = await writer.checkpoint();
        } catch (error) {
          throw downloadError(
            "CONTENT_STORE_FAILURE",
            errorMessage(error, `${artifactLabel(file)} could not save its download progress`),
            500,
            true,
            { path: file.path, offset },
            error,
          );
        }
      }
      if (offlineStreamFailure !== null) {
        await this.waitUntilOnline(signal);
        throwIfAborted(signal, modelVariantId);
        tracker.setFile(index, offset, validation.resumable, "hashing");
        continue;
      }
      if (expectedResponseBytes !== null && responseBytes !== expectedResponseBytes) {
        throw downloadError(
          "DOWNLOAD_SIZE_MISMATCH",
          `The download for ${artifactLabel(file)} ended early`,
          502,
          true,
          {
            path: file.path,
            expectedResponseBytes,
            actualResponseBytes: responseBytes,
          },
        );
      }
      tracker.setFile(index, offset, validation.resumable, "hashing");
    }

    if (offset !== file.bytes) {
      throw downloadError(
        "DOWNLOAD_SIZE_MISMATCH",
        `The download for ${artifactLabel(file)} has the wrong size`,
        502,
        true,
        { path: file.path, expectedBytes: file.bytes, actualBytes: offset },
      );
    }
    const computedSha256 = bytesToHex(hasher.digest());
    if (computedSha256 !== file.sha256) {
      await this.store.resetPartial(descriptorFor(file, record.etag, partialOwner));
      tracker.setFile(index, 0, false, "failed");
      throw downloadError(
        "DOWNLOAD_HASH_MISMATCH",
        `The download for ${artifactLabel(file)} failed its integrity check`,
        502,
        true,
        { path: file.path, expectedSha256: file.sha256, computedSha256 },
      );
    }
    try {
      await this.store.commitVerified(file.sha256, computedSha256, offset);
    } catch (error) {
      throw downloadError(
        "CONTENT_STORE_FAILURE",
        errorMessage(error, `${artifactLabel(file)} could not be marked ready`),
        500,
        true,
        { path: file.path },
        error,
      );
    }
    tracker.setFile(index, file.bytes, false, "verified");
  }

  private createProgressTracker(
    selection: ArtifactManifestSelection,
    onProgress: ArtifactDownloadOptions["onProgress"],
    storedFiles: readonly StoredFileState[],
  ): ProgressTracker {
    const files: DownloadFileProgress[] = selection.files.map((file, index) => ({
      path: file.path,
      bytesReceived: storedFiles[index].bytes,
      bytesTotal: file.bytes,
      resumable: false,
      verification: storedFiles[index].verified ? "verified" : "pending",
    }));
    let activeNetworkRequests = 0;
    let downloadedBytes = 0;
    let lastEtaAt = this.monotonicNow();
    const estimator = new DownloadEtaEstimator(lastEtaAt);
    let eta: EtaSnapshot = {
      throughputBytesPerSecond: null,
      etaSeconds: null,
      calculating: true,
      stalled: false,
    };
    let complete = false;

    const emit = (): void => {
      if (!onProgress) return;
      const online = this.isOnline();
      onProgress({
        modelVariantId: selection.variant.id,
        ...(selection.packId === undefined ? {} : { packId: selection.packId }),
        files: files.map((file) => ({ ...file })),
        bytesReceived: files.reduce((total, file) => total + file.bytesReceived, 0),
        bytesTotal: selection.totalBytes,
        throughputBytesPerSecond: eta.throughputBytesPerSecond,
        etaSeconds: eta.etaSeconds,
        calculatingEta: eta.calculating,
        stalled: eta.stalled,
        offline: !online,
        resumable: files.some((file) => file.resumable),
      });
    };
    const updateEta = (): void => {
      const now = this.monotonicNow();
      const remainingBytes = remainingContentBytes(
        selection.files,
        (index) => files[index].bytesReceived,
      );
      eta = estimator.observe(
        downloadedBytes,
        downloadedBytes + remainingBytes,
        now,
        activeNetworkRequests > 0 && this.isOnline(),
      );
      lastEtaAt = now;
      emit();
    };
    const interval = this.scheduler.setInterval(updateEta, ETA_TICK_MS);
    emit();

    return {
      setFile: (index, bytesReceived, resumable, verification) => {
        files[index] = {
          ...files[index],
          bytesReceived,
          resumable: resumable ?? files[index].resumable,
          verification,
        };
        emit();
      },
      recordChunk: (index, bytesReceived, bytesTransferred) => {
        downloadedBytes += bytesTransferred;
        const sha256 = selection.files[index].sha256;
        for (let duplicateIndex = 0; duplicateIndex < selection.files.length; duplicateIndex += 1) {
          if (selection.files[duplicateIndex].sha256 !== sha256) continue;
          files[duplicateIndex] = {
            ...files[duplicateIndex],
            bytesReceived: Math.min(selection.files[duplicateIndex].bytes, bytesReceived),
          };
        }
        if (this.monotonicNow() - lastEtaAt >= ETA_TICK_MS) updateEta();
      },
      setNetworkActive: (active) => {
        activeNetworkRequests = Math.max(0, activeNetworkRequests + (active ? 1 : -1));
      },
      get downloadedBytes() {
        return downloadedBytes;
      },
      complete: () => {
        if (complete) return;
        complete = true;
        eta = {
          ...eta,
          etaSeconds: [0, 0],
          calculating: false,
          stalled: false,
        };
        emit();
      },
      dispose: () => this.scheduler.clearInterval(interval),
    };
  }

  private async safePartialSize(sha256Digest: string): Promise<number> {
    try {
      return (await this.store.partialFile(sha256Digest))?.size ?? 0;
    } catch {
      return 0;
    }
  }
}

function defaultWaitUntilOnline(signal: AbortSignal): Promise<void> {
  if (typeof navigator === "undefined" || navigator.onLine !== false) return Promise.resolve();
  const events = globalThis as unknown as {
    addEventListener(type: "online", listener: () => void): void;
    removeEventListener(type: "online", listener: () => void): void;
  };
  return new Promise<void>((resolve, reject) => {
    const cleanup = (): void => {
      events.removeEventListener("online", onOnline);
      signal.removeEventListener("abort", onAbort);
    };
    const onOnline = (): void => {
      if (navigator.onLine === false) return;
      cleanup();
      resolve();
    };
    const onAbort = (): void => {
      cleanup();
      reject(signal.reason ?? new DOMException("The download was cancelled", "AbortError"));
    };
    events.addEventListener("online", onOnline);
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) onAbort();
    else if (navigator.onLine !== false) onOnline();
  });
}

const fallbackObjectLocks = new Map<string, Promise<void>>();

function remainingContentBytes(
  files: readonly CatalogFile[],
  receivedBytes: (index: number) => number,
): number {
  const remainingByHash = new Map<string, number>();
  for (let index = 0; index < files.length; index += 1) {
    const file = files[index];
    const remaining = Math.max(0, file.bytes - receivedBytes(index));
    remainingByHash.set(
      file.sha256,
      Math.max(remainingByHash.get(file.sha256) ?? 0, remaining),
    );
  }
  return [...remainingByHash.values()].reduce(
    (total, remaining) => total + remaining,
    0,
  );
}

function defaultDownloadShared<T>(
  signal: AbortSignal,
  operation: () => Promise<T>,
): Promise<T> {
  const locks = typeof navigator === "undefined" ? undefined : navigator.locks;
  return locks
    ? locks.request(
        HOSTED_DESTRUCTIVE_LOCK,
        { mode: "shared", signal },
        () => locks.request(
          CONTENT_LIFECYCLE_LOCK,
          { mode: "shared", signal },
          operation,
        ),
      )
    : operation();
}

async function defaultObjectExclusive<T>(
  sha256Digest: string,
  signal: AbortSignal,
  operation: () => Promise<T>,
): Promise<T> {
  const locks = typeof navigator === "undefined" ? undefined : navigator.locks;
  if (locks) {
    return locks.request(
      `${CONTENT_OBJECT_LOCK_PREFIX}${sha256Digest}`,
      { mode: "exclusive", signal },
      operation,
    );
  }
  const predecessor = fallbackObjectLocks.get(sha256Digest) ?? Promise.resolve();
  let release!: () => void;
  const tail = new Promise<void>((resolve) => {
    release = resolve;
  });
  const queued = predecessor.then(() => tail);
  fallbackObjectLocks.set(sha256Digest, queued);
  try {
    await abortable(predecessor, signal);
    return await operation();
  } finally {
    release();
    if (fallbackObjectLocks.get(sha256Digest) === queued) {
      fallbackObjectLocks.delete(sha256Digest);
    }
  }
}

function abortable<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(new DOMException("Aborted", "AbortError"));
  return new Promise<T>((resolve, reject) => {
    const abort = (): void => {
      signal.removeEventListener("abort", abort);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal.addEventListener("abort", abort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener("abort", abort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", abort);
        reject(error);
      },
    );
  });
}

function descriptorFor(
  file: CatalogFile,
  etag: string | null,
  partialOwner: PartialContentOwner,
): ContentObjectDescriptor {
  return {
    sha256: file.sha256,
    expectedBytes: file.bytes,
    url: file.url,
    revision: file.revision,
    etag,
    partialOwners: [{ ...partialOwner }],
  };
}

function validPartialOwnersForCatalog(
  catalog: VerifiedCatalog,
): ValidPartialOwner[] {
  const owners: ValidPartialOwner[] = [];
  for (const model of catalog.document.models) {
    for (const variant of model.variants) {
      owners.push({
        id: variant.id,
        kind: "model",
        objectHashes: [
          ...variant.files.map((file) => file.sha256),
          ...variant.packs
            .filter((pack) => pack.required)
            .flatMap((pack) => pack.files.map((file) => file.sha256)),
        ],
      });
      for (const pack of variant.packs.filter((candidate) => !candidate.required)) {
        owners.push({
          id: pack.id,
          kind: "pack",
          objectHashes: pack.files.map((file) => file.sha256),
        });
      }
    }
  }
  return owners;
}

function sumBytes(files: readonly CatalogFile[]): number {
  return files.reduce((total, file) => total + file.bytes, 0);
}

function artifactLabel(file: CatalogFile): string {
  switch (file.role) {
    case "weight":
      return "a model data file";
    case "tokenizer":
    case "chat_template":
      return "the model language files";
    case "configuration":
    case "converted_manifest":
      return "the model setup files";
    case "model_library":
      return "the local model engine";
    case "core_pack":
      return "the required response controls";
    case "instrument":
      return "the selected model tool";
  }
}

function integerHeader(value: string | null): number | null {
  if (value === null || !/^\d+$/.test(value)) return null;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) ? parsed : null;
}

function validateRedirectOrigin(value: string): string {
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new TypeError(`Invalid download redirect origin ${value}`);
  }
  if (
    value !== parsed.origin || parsed.protocol !== "https:" ||
    parsed.username || parsed.password
  ) {
    throw new TypeError(`Download redirect allowlist entries must be exact HTTPS origins: ${value}`);
  }
  return value;
}

async function cancelBody(response: Response): Promise<void> {
  try {
    await response.body?.cancel();
  } catch {}
}

function throwIfAborted(signal: AbortSignal, modelVariantId: string): void {
  if (signal.aborted) throw cancelledError(modelVariantId);
}

function cancelledError(modelVariantId: string, cause?: unknown): ArtifactDownloadError {
  return downloadError(
    "DOWNLOAD_CANCELLED",
    `Download of ${modelVariantId} was cancelled`,
    499,
    true,
    { modelVariantId },
    cause,
  );
}

function downloadError(
  code: ArtifactDownloadErrorCode,
  message: string,
  status: number,
  recoverable: boolean,
  detail?: unknown,
  cause?: unknown,
): ArtifactDownloadError {
  return new ArtifactDownloadError(code, message, {
    status,
    recoverable,
    detail,
    cause,
  });
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function isRuntimeFailure(value: unknown): value is RuntimeFailure {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const failure = value as Partial<RuntimeFailure>;
  return typeof failure.code === "string" && failure.code.length > 0 &&
    typeof failure.message === "string" && failure.message.length > 0 &&
    typeof failure.recoverable === "boolean" &&
    Number.isSafeInteger(failure.status) && failure.status! >= 400 && failure.status! <= 599;
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function isQuotaExceededError(error: unknown, seen = new Set<unknown>()): boolean {
  if (error === null || (typeof error !== "object" && typeof error !== "function")) {
    return false;
  }
  if (seen.has(error)) return false;
  seen.add(error);
  if ("name" in error && error.name === "QuotaExceededError") return true;
  return "cause" in error && isQuotaExceededError(error.cause, seen);
}
