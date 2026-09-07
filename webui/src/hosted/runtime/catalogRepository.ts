import type {
  CatalogWorkerRequest,
  DetachedCatalogSignature,
  VerifiedCatalog,
} from "../../lib/runtime/contracts";
import {
  CatalogValidationError,
  WebCryptoEd25519Verifier,
  verifyCatalog,
  type CatalogAdmissionPolicy,
  type CatalogSignatureVerifier,
} from "../../lib/runtime/catalog";
import {
  HOSTED_DISTRIBUTION_CONFIG,
  type HostedDistributionConfig,
} from "./distributionConfig";
import { migrateLegacyDatabase } from "./brandMigration";

const DATABASE_NAME = "drowse-hosted-catalog";
const DATABASE_VERSION = 1;
const STATE_STORE = "state";
const STATE_KEY = "catalog";
const CATALOG_LOCK_NAME = "drowse-hosted-catalog-state-v1";
const MAX_CATALOG_BYTES = 4 * 1024 * 1024;
const MAX_SIGNATURE_BYTES = 16 * 1024;
const DEFAULT_FETCH_TIMEOUT_MS = 15_000;

export interface StoredCatalogState {
  id: typeof STATE_KEY;
  exactBytes: Uint8Array | null;
  signature: DetachedCatalogSignature | null;
  highestAcceptedSequence: number | null;
  acceptedAt: number | null;
}

export interface CatalogStateStore {
  initialize(): Promise<void>;
  read(): Promise<StoredCatalogState>;
  write(state: StoredCatalogState): Promise<void>;
  clear(): Promise<void>;
}

interface CatalogFetchResponse {
  readonly ok: boolean;
  readonly status: number;
  readonly url: string;
  readonly body?: ReadableStream<Uint8Array> | null;
  readonly headers?: Headers;
  arrayBuffer(): Promise<ArrayBuffer>;
}

type CatalogFetch = (
  input: string,
  init: RequestInit,
) => Promise<CatalogFetchResponse>;

export type CatalogExclusiveRunner = <T>(
  operation: () => Promise<T>,
  signal?: AbortSignal,
) => Promise<T>;

let fallbackCatalogLockTail: Promise<void> = Promise.resolve();

const fallbackCatalogExclusiveRunner: CatalogExclusiveRunner = <T>(
  operation: () => Promise<T>,
  signal?: AbortSignal,
): Promise<T> => {
  const predecessor = fallbackCatalogLockTail;
  const turn = predecessor.then(async () => {
    throwIfAborted(signal);
    return operation();
  });
  fallbackCatalogLockTail = turn.then(() => undefined, () => undefined);
  return rejectWhenAborted(turn, signal);
};

const defaultCatalogExclusiveRunner: CatalogExclusiveRunner = <T>(
  operation: () => Promise<T>,
  signal?: AbortSignal,
): Promise<T> => {
  if (typeof navigator !== "undefined" && navigator.locks) {
    return navigator.locks.request(
      CATALOG_LOCK_NAME,
      { mode: "exclusive", ...(signal ? { signal } : {}) },
      () => {
        throwIfAborted(signal);
        return operation();
      },
    );
  }
  return fallbackCatalogExclusiveRunner(operation, signal);
};

export interface CatalogRepositoryOptions {
  config?: HostedDistributionConfig;
  store?: CatalogStateStore;
  verifier?: CatalogSignatureVerifier;
  fetch?: CatalogFetch;
  online?: () => boolean;
  now?: () => number;
  fetchTimeoutMs?: number;
  runExclusive?: CatalogExclusiveRunner;
}

export class VerifiedCatalogRepository {
  private readonly config: HostedDistributionConfig;
  private readonly store: CatalogStateStore;
  private readonly verifier: CatalogSignatureVerifier;
  private readonly fetcher: CatalogFetch;
  private readonly online: () => boolean;
  private readonly now: () => number;
  private readonly fetchTimeoutMs: number;
  private readonly runExclusive: CatalogExclusiveRunner;
  private initialization: Promise<void> | null = null;

  constructor(options: CatalogRepositoryOptions = {}) {
    this.config = options.config ?? HOSTED_DISTRIBUTION_CONFIG;
    this.store = options.store ?? new BrowserCatalogStateStore();
    this.verifier = options.verifier ?? new WebCryptoEd25519Verifier(this.config.publicKeys);
    this.fetcher = options.fetch ?? ((input, init) => fetch(input, init));
    this.online = options.online ?? (() => navigator.onLine !== false);
    this.now = options.now ?? Date.now;
    this.fetchTimeoutMs = options.fetchTimeoutMs ?? DEFAULT_FETCH_TIMEOUT_MS;
    if (!Number.isFinite(this.fetchTimeoutMs) || this.fetchTimeoutMs <= 0) {
      throw new TypeError("Catalog fetch timeout must be a positive number");
    }
    this.runExclusive = options.runExclusive ?? defaultCatalogExclusiveRunner;
  }

  async get(request: CatalogWorkerRequest = {}): Promise<VerifiedCatalog> {
    this.assertConfigured();
    return this.runExclusive(async () => {
      await this.initialize();
      const offline = request.offline === true || !this.online();
      let state: StoredCatalogState;
      try {
        state = await this.store.read();
      } catch (error) {
        if (
          offline || !(error instanceof CatalogValidationError) ||
          error.code !== "CATALOG_CACHE_INVALID"
        ) throw error;
        await this.store.clear();
        state = emptyState();
      }
      if (offline) {
        return this.verifyCached(state, false);
      }
      let repairSequence: number | null = null;
      let verifiedCache: VerifiedCatalog | null = null;
      if (state.exactBytes !== null) {
        try {
          verifiedCache = await this.verifyCached(state, true);
        } catch (error) {
          if (isRepairableCachedCatalogError(error)) {
            repairSequence = state.highestAcceptedSequence;
          }
        }
      }
      if (request.preferCached === true && verifiedCache !== null) return verifiedCache;

      try {
        const fetched = await this.fetchAndVerify(state, repairSequence);
        await this.store.write({
          id: STATE_KEY,
          exactBytes: fetched.exactBytes,
          signature: fetched.signature,
          highestAcceptedSequence: fetched.document.sequence,
          acceptedAt: this.now(),
        });
        return fetched;
      } catch (error) {
        if (
          request.refresh ||
          !(error instanceof CatalogFetchError) ||
          !error.cachedFallbackAllowed ||
          state.exactBytes === null
        ) throw error;
        return this.verifyCached(state, false);
      }
    });
  }

  async withDownloadAuthorization<T>(
    catalog: VerifiedCatalog,
    operation: () => Promise<T>,
    signal?: AbortSignal,
  ): Promise<T> {
    this.assertConfigured();
    return this.runExclusive(async () => {
      throwIfAborted(signal);
      await this.initialize();
      throwIfAborted(signal);
      const state = await this.store.read();
      if (
        state.exactBytes === null ||
        state.highestAcceptedSequence !== catalog.document.sequence ||
        !sameBytes(state.exactBytes, catalog.exactBytes)
      ) {
        throw new CatalogValidationError(
          "CATALOG_SUPERSEDED",
          "A newer or different signed catalog has been accepted; refresh before downloading",
        );
      }
      const expiresAt = Date.parse(catalog.document.expiresAt);
      if (!Number.isFinite(expiresAt) || this.now() >= expiresAt) {
        throw new CatalogValidationError(
          "CATALOG_EXPIRED",
          "The accepted catalog expired while waiting to authorize this download",
        );
      }
      throwIfAborted(signal);
      return operation();
    }, signal);
  }

  async clear(): Promise<void> {
    await this.runExclusive(async () => {
      await this.initialize();
      await this.store.clear();
    });
  }

  private initialize(): Promise<void> {
    if (this.initialization === null) {
      const initialization = Promise.resolve().then(() => this.store.initialize());
      this.initialization = initialization;
      void initialization.catch(() => {
        if (this.initialization === initialization) this.initialization = null;
      });
    }
    return this.initialization;
  }

  private assertConfigured(): void {
    if (this.config.publicKeys.size === 0) {
      throw new CatalogValidationError(
        "SIGNED_CATALOG_KEYS_UNCONFIGURED",
        "The signed Drowse catalog key ring has not been provisioned",
      );
    }
    if (this.config.status !== "verified") {
      throw new CatalogValidationError(
        "HOSTED_DISTRIBUTION_NOT_VERIFIED",
        "The hosted Drowse artifact distribution has not been published yet",
      );
    }
  }

  private async fetchAndVerify(
    state: StoredCatalogState,
    repairSequence: number | null = null,
  ): Promise<VerifiedCatalog> {
    const controller = new AbortController();
    let timeout: ReturnType<typeof setTimeout> | undefined;
    const deadline = new Promise<never>((_, reject) => {
      timeout = setTimeout(() => {
        reject(new CatalogFetchError(
          "CATALOG_FETCH_TIMEOUT",
          "The signed catalog download timed out",
          true,
        ));
        controller.abort();
      }, this.fetchTimeoutMs);
    });
    let catalogResponse: { bytes: Uint8Array };
    let signatureResponse: { bytes: Uint8Array };
    try {
      [catalogResponse, signatureResponse] = await Promise.race([
        Promise.all([
          this.fetchResource(
            this.config.catalogUrl,
            MAX_CATALOG_BYTES,
            "catalog",
            controller.signal,
          ),
          this.fetchResource(
            this.config.signatureUrl,
            MAX_SIGNATURE_BYTES,
            "signature",
            controller.signal,
          ),
        ]),
        deadline,
      ]);
    } finally {
      if (timeout !== undefined) clearTimeout(timeout);
      controller.abort();
    }
    const exactBytes = catalogResponse.bytes;
    let signature: DetachedCatalogSignature;
    try {
      signature = JSON.parse(
        new TextDecoder("utf-8", { fatal: true }).decode(signatureResponse.bytes),
      ) as DetachedCatalogSignature;
    } catch {
      throw new CatalogValidationError(
        "CATALOG_SIGNATURE_ENVELOPE_INVALID",
        "The catalog signature envelope is not valid JSON",
      );
    }
    const verified = await verifyCatalog(exactBytes, signature, this.verifier, {
      now: this.now(),
      online: true,
      minimumSequence: this.config.minimumAcceptedSequence,
      lastAcceptedSequence: repairSequence === null ? state.highestAcceptedSequence : null,
      lastAcceptedIdentity: repairSequence === null ? storedIdentity(state) : null,
      expectedRuntimeAbi: this.config.expectedRuntimeAbi,
      runtimeLockModels: this.config.runtimeLockModels,
    });
    if (repairSequence !== null && verified.document.sequence < repairSequence) {
      throw new CatalogValidationError(
        "CATALOG_ROLLBACK",
        "The catalog sequence is older than the last accepted release",
      );
    }
    return verified;
  }

  private async fetchResource(
    url: string,
    maximumBytes: number,
    label: string,
    signal: AbortSignal,
  ): Promise<{ bytes: Uint8Array }> {
    let response: CatalogFetchResponse;
    try {
      response = await this.fetcher(url, {
        cache: "no-store",
        credentials: "omit",
        redirect: "follow",
        referrerPolicy: "no-referrer",
        signal,
      });
    } catch (error) {
      throw new CatalogFetchError(
        "CATALOG_FETCH_UNAVAILABLE",
        error instanceof Error ? error.message : `The ${label} could not be downloaded`,
        true,
      );
    }
    if (!response.ok) {
      throw new CatalogFetchError(
        "CATALOG_FETCH_FAILED",
        `The ${label} request returned HTTP ${response.status}`,
        response.status === 429 || response.status >= 500,
      );
    }
    validateResponseOrigin(
      response.url,
      this.config.allowedCatalogRedirectOrigins,
      label,
    );
    let bytes: Uint8Array;
    try {
      bytes = await readBoundedResponse(
        response,
        maximumBytes,
        label,
        signal,
      );
    } catch (error) {
      if (error instanceof CatalogValidationError || error instanceof CatalogFetchError) {
        throw error;
      }
      throw new CatalogFetchError(
        "CATALOG_FETCH_UNAVAILABLE",
        error instanceof Error ? error.message : `The ${label} response was interrupted`,
        true,
      );
    }
    if (bytes.byteLength === 0) {
      throw new CatalogValidationError(
        "CATALOG_RESPONSE_SIZE_INVALID",
        `The ${label} response has an invalid size`,
      );
    }
    return { bytes };
  }

  private async verifyCached(
    state: StoredCatalogState,
    online: boolean,
  ): Promise<VerifiedCatalog> {
    if (state.exactBytes === null || state.signature === null) {
      throw new CatalogValidationError(
        "CATALOG_OFFLINE_UNAVAILABLE",
        "No verified Drowse catalog is available on this device",
      );
    }
    const verified = await verifyCatalog(state.exactBytes, state.signature, this.verifier, {
      now: this.now(),
      online,
      minimumSequence: this.config.minimumAcceptedSequence,
      lastAcceptedSequence: state.highestAcceptedSequence,
      lastAcceptedIdentity: storedIdentity(state),
      expectedRuntimeAbi: this.config.expectedRuntimeAbi,
      runtimeLockModels: this.config.runtimeLockModels,
    });
    return online ? verified : { ...verified, allowDownloads: false };
  }
}

function storedIdentity(
  state: StoredCatalogState,
): CatalogAdmissionPolicy["lastAcceptedIdentity"] {
  if (state.exactBytes === null || state.signature === null) return null;
  return { exactBytes: state.exactBytes, signature: state.signature };
}

function isRepairableCachedCatalogError(error: unknown): boolean {
  if (!(error instanceof CatalogValidationError)) return false;
  return new Set([
    "CATALOG_SIZE_INVALID",
    "CATALOG_SIGNATURE_INVALID",
    "CATALOG_JSON_INVALID",
    "CATALOG_SCHEMA_INVALID",
    "CATALOG_DOCUMENT_INVALID",
  ]).has(error.code);
}

export class BrowserCatalogStateStore implements CatalogStateStore {
  private database: IDBDatabase | null = null;
  private opening: Promise<IDBDatabase> | null = null;

  async initialize(): Promise<void> {
    if (this.database) return;
    this.opening ??= openDatabase();
    const opening = this.opening;
    try {
      const database = await opening;
      if (this.database) {
        if (this.database !== database) database.close();
        return;
      }
      this.database = database;
      database.onversionchange = () => {
        if (this.database === database) this.database = null;
        database.close();
      };
    } finally {
      if (this.opening === opening) this.opening = null;
    }
  }

  async read(): Promise<StoredCatalogState> {
    await this.initialize();
    const database = this.requireDatabase();
    const transaction = database.transaction(STATE_STORE, "readonly");
    const stored = await requestValue<StoredCatalogState | undefined>(
      transaction.objectStore(STATE_STORE).get(STATE_KEY),
    );
    if (!stored) return emptyState();
    if (
      stored.id !== STATE_KEY ||
      !(stored.exactBytes === null || stored.exactBytes instanceof Uint8Array) ||
      !(stored.highestAcceptedSequence === null ||
        (Number.isSafeInteger(stored.highestAcceptedSequence) &&
          stored.highestAcceptedSequence > 0)) ||
      !(stored.acceptedAt === null ||
        (Number.isFinite(stored.acceptedAt) && stored.acceptedAt >= 0))
    ) {
      throw new CatalogValidationError(
        "CATALOG_CACHE_INVALID",
        "The locally cached catalog metadata is invalid",
      );
    }
    return {
      ...stored,
      exactBytes: stored.exactBytes?.slice() ?? null,
    };
  }

  async write(state: StoredCatalogState): Promise<void> {
    await this.initialize();
    const database = this.requireDatabase();
    const transaction = database.transaction(STATE_STORE, "readwrite");
    transaction.objectStore(STATE_STORE).put({
      ...state,
      exactBytes: state.exactBytes?.slice() ?? null,
    });
    await transactionDone(transaction);
  }

  async clear(): Promise<void> {
    await this.initialize();
    const database = this.requireDatabase();
    const transaction = database.transaction(STATE_STORE, "readwrite");
    transaction.objectStore(STATE_STORE).clear();
    await transactionDone(transaction);
  }

  private requireDatabase(): IDBDatabase {
    if (!this.database) throw new Error("Catalog store has not been initialized");
    return this.database;
  }
}

export class CatalogFetchError extends Error {
  readonly code: string;
  readonly cachedFallbackAllowed: boolean;

  constructor(code: string, message: string, cachedFallbackAllowed: boolean) {
    super(message);
    this.name = "CatalogFetchError";
    this.code = code;
    this.cachedFallbackAllowed = cachedFallbackAllowed;
  }
}

function validateResponseOrigin(
  responseUrl: string,
  allowedOrigins: ReadonlySet<string>,
  label: string,
): void {
  let origin: string;
  try {
    origin = new URL(responseUrl).origin;
  } catch {
    throw new CatalogValidationError(
      "CATALOG_REDIRECT_INVALID",
      `The ${label} response URL is invalid`,
    );
  }
  if (!allowedOrigins.has(origin)) {
    throw new CatalogValidationError(
      "CATALOG_REDIRECT_DISALLOWED",
      `The ${label} response used an unapproved origin`,
    );
  }
}

function emptyState(): StoredCatalogState {
  return {
    id: STATE_KEY,
    exactBytes: null,
    signature: null,
    highestAcceptedSequence: null,
    acceptedAt: null,
  };
}

function sameBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  for (let index = 0; index < left.byteLength; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  throw signal.reason ?? new DOMException("Catalog authorization was cancelled", "AbortError");
}

function rejectWhenAborted<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(
    signal.reason ?? new DOMException("Catalog authorization was cancelled", "AbortError"),
  );
  return new Promise<T>((resolve, reject) => {
    const abort = () => reject(
      signal.reason ?? new DOMException("Catalog authorization was cancelled", "AbortError"),
    );
    signal.addEventListener("abort", abort, { once: true });
    void promise.then(resolve, reject).finally(() => {
      signal.removeEventListener("abort", abort);
    });
  });
}

async function openDatabase(): Promise<IDBDatabase> {
  const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
  request.onupgradeneeded = () => {
    const database = request.result;
    if (!database.objectStoreNames.contains(STATE_STORE)) {
      database.createObjectStore(STATE_STORE, { keyPath: "id" });
    }
  };
  const database = await new Promise<IDBDatabase>((resolve, reject) => {
    let settled = false;
    const rejectOnce = (error: Error) => {
      if (settled) return;
      settled = true;
      reject(error);
    };
    request.onsuccess = () => {
      const database = request.result;
      if (settled) {
        database.close();
        return;
      }
      settled = true;
      resolve(database);
    };
    request.onerror = () => rejectOnce(
      request.error ?? new Error("IndexedDB open failed"),
    );
    request.onblocked = () => rejectOnce(
      new Error("Another Drowse tab is blocking the catalog-store upgrade"),
    );
  });
  await migrateLegacyDatabase(database, [STATE_STORE]);
  return database;
}

async function readBoundedResponse(
  response: CatalogFetchResponse,
  maximumBytes: number,
  label: string,
  signal: AbortSignal,
): Promise<Uint8Array> {
  const declaredLength = response.headers?.get("Content-Length") ?? null;
  if (declaredLength !== null) {
    if (!/^\d+$/.test(declaredLength)) {
      throw new CatalogValidationError(
        "CATALOG_RESPONSE_SIZE_INVALID",
        `The ${label} response has an invalid Content-Length`,
      );
    }
    const parsed = Number(declaredLength);
    if (!Number.isSafeInteger(parsed) || parsed > maximumBytes) {
      throw new CatalogValidationError(
        "CATALOG_RESPONSE_SIZE_INVALID",
        `The ${label} response exceeds its maximum size`,
      );
    }
  }
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength > maximumBytes) {
      throw new CatalogValidationError(
        "CATALOG_RESPONSE_SIZE_INVALID",
        `The ${label} response exceeds its maximum size`,
      );
    }
    return bytes;
  }
  const reader = response.body.getReader();
  const cancelReader = () => {
    void reader.cancel(signal.reason).catch(() => undefined);
  };
  signal.addEventListener("abort", cancelReader, { once: true });
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value?.byteLength) continue;
      total += value.byteLength;
      if (total > maximumBytes) {
        throw new CatalogValidationError(
          "CATALOG_RESPONSE_SIZE_INVALID",
          `The ${label} response exceeds its maximum size`,
        );
      }
      chunks.push(value.slice());
    }
  } catch (error) {
    try {
      await reader.cancel();
    } catch {}
    throw error;
  } finally {
    signal.removeEventListener("abort", cancelReader);
    reader.releaseLock();
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function requestValue<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(
      transaction.error ?? new Error("IndexedDB transaction aborted"),
    );
    transaction.onerror = () => reject(
      transaction.error ?? new Error("IndexedDB transaction failed"),
    );
  });
}
