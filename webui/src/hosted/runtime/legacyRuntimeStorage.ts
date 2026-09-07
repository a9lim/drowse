const DROWSE_CACHE_PREFIX = "drowse-hosted-";

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

  if (cacheStorage !== null) {
    const legacyNames = new Set<string>(LEGACY_WEBLLM_CACHE_STORAGE_NAMES);
    for (const name of await cacheStorage.keys()) {
      if (
        name === "drowse-hosted" || name.startsWith(DROWSE_CACHE_PREFIX) ||
        legacyNames.has(name)
      ) {
        await cacheStorage.delete(name);
      }
    }
  }

  if (indexedDb !== null) {
    for (const name of LEGACY_WEBLLM_INDEXED_DB_NAMES) {
      await deleteIndexedDatabase(indexedDb, name);
    }
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
