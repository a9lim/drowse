import { LEGACY_PRODUCT_SLUGS, legacyDatabaseNames } from "./brandMigration";

const PRODUCT_CACHE_SCOPES = ["drowse", ...LEGACY_PRODUCT_SLUGS].map((slug) => `${slug}-hosted`);

export const LEGACY_PRODUCT_INDEXED_DB_NAMES = Object.freeze([
  "runtime", "sessions", "catalog", "artifacts", "authoring", "activation-spool",
].flatMap((suffix) => legacyDatabaseNames(`drowse-hosted-${suffix}`)));

// Exact scopes used by the runtime-locked @drowse/web-llm fork and its pinned TVM runtime.
export const LEGACY_WEBLLM_CACHE_STORAGE_NAMES = Object.freeze([
  "webllm/model",
  "webllm/config",
  "webllm/wasm",
  "tvmjs",
  "tvmjs-cos-hash-meta",
] as const);

export const LEGACY_WEBLLM_INDEXED_DB_NAMES = Object.freeze([
  "webllm/model",
  "webllm/config",
  "webllm/wasm",
  "tvmjs",
] as const);

interface RuntimeStorageCleanupDependencies {
  cacheStorage?: Pick<CacheStorage, "keys" | "delete"> | null;
  indexedDb?: Pick<IDBFactory, "deleteDatabase"> | null;
  storage?: Pick<StorageManager, "getDirectory"> | null;
}

export async function clearOwnedRuntimeStorage(
  dependencies: RuntimeStorageCleanupDependencies = {},
): Promise<void> {
  const cacheStorage = dependencies.cacheStorage === undefined
    ? typeof caches === "undefined" ? null : caches
    : dependencies.cacheStorage;
  const indexedDb = dependencies.indexedDb === undefined
    ? typeof indexedDB === "undefined" ? null : indexedDB
    : dependencies.indexedDb;
  const storage = dependencies.storage === undefined
    ? typeof navigator === "undefined" || typeof navigator.storage?.getDirectory !== "function" ? null : navigator.storage
    : dependencies.storage;

  if (cacheStorage !== null) {
    const legacyNames = new Set<string>(LEGACY_WEBLLM_CACHE_STORAGE_NAMES);
    for (const name of await cacheStorage.keys()) {
      if (
        PRODUCT_CACHE_SCOPES.some((scope) => name === scope || name.startsWith(`${scope}-`)) ||
        legacyNames.has(name)
      ) {
        await cacheStorage.delete(name);
      }
    }
  }

  if (indexedDb !== null) {
    for (const name of [...LEGACY_WEBLLM_INDEXED_DB_NAMES, ...LEGACY_PRODUCT_INDEXED_DB_NAMES]) {
      await deleteIndexedDatabase(indexedDb, name);
    }
  }
  if (storage !== null) {
    const root = await storage.getDirectory();
    for (const slug of LEGACY_PRODUCT_SLUGS) {
      let directory: FileSystemDirectoryHandle;
      try {
        directory = await root.getDirectoryHandle(slug);
      } catch (error) {
        if (error instanceof DOMException && error.name === "NotFoundError") continue;
        throw error;
      }
      await clearDirectoryFiles(directory);
    }
  }
}

async function clearDirectoryFiles(directory: FileSystemDirectoryHandle): Promise<void> {
  // Older OPFS implementations reuse legacy roots; keep their open directory handles valid.
  const iterable = directory as FileSystemDirectoryHandle & { values(): AsyncIterable<FileSystemHandle> };
  for await (const entry of iterable.values()) {
    if (entry.kind === "directory") await clearDirectoryFiles(entry as FileSystemDirectoryHandle);
    else await directory.removeEntry(entry.name);
  }
}

function deleteIndexedDatabase(
  factory: Pick<IDBFactory, "deleteDatabase">,
  name: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = factory.deleteDatabase(name);
    let settled = false;
    const rejectOnce = (error: Error): void => {
      if (settled) return;
      settled = true;
      reject(error);
    };
    request.onsuccess = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    request.onerror = () => rejectOnce(
      request.error ?? new Error(`IndexedDB deletion failed for ${name}`),
    );
    request.onblocked = () => rejectOnce(
      new Error(`Another Drowse tab is blocking deletion of legacy database ${name}`),
    );
  });
}
