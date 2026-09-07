import assert from "node:assert/strict";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import {
  LEGACY_WEBLLM_CACHE_STORAGE_NAMES as AUDITED_CACHE_NAMES,
  LEGACY_WEBLLM_INDEXED_DB_NAMES as AUDITED_DATABASE_NAMES,
} from "./browser-production-app-runtime-contract.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const {
    LEGACY_WEBLLM_CACHE_STORAGE_NAMES,
    LEGACY_WEBLLM_INDEXED_DB_NAMES,
    clearOwnedRuntimeStorage,
  } = await server.ssrLoadModule("/src/hosted/runtime/legacyRuntimeStorage.ts");

  test("clear-all and physical audit use the same pinned WebLLM store allowlist", () => {
    assert.deepEqual([...LEGACY_WEBLLM_CACHE_STORAGE_NAMES], AUDITED_CACHE_NAMES);
    assert.deepEqual([...LEGACY_WEBLLM_INDEXED_DB_NAMES], AUDITED_DATABASE_NAMES);
  });

  test("clear-all removes only Drowse and exact pinned WebLLM CacheStorage names", async () => {
    const deleted = [];
    await clearOwnedRuntimeStorage({
      cacheStorage: {
        async keys() {
          return [
            "drowse-hosted",
            "drowse-hosted-shell-v1",
            ...LEGACY_WEBLLM_CACHE_STORAGE_NAMES,
            "webllm/model-other",
            "tvmjs-other",
            "unrelated-app-cache",
          ];
        },
        async delete(name) {
          deleted.push(name);
          return true;
        },
      },
      indexedDb: null,
    });
    assert.deepEqual(deleted, [
      "drowse-hosted",
      "drowse-hosted-shell-v1",
      ...LEGACY_WEBLLM_CACHE_STORAGE_NAMES,
    ]);
  });

  test("clear-all deletes only exact pinned WebLLM IndexedDB database names", async () => {
    const deleted = [];
    const indexedDb = {
      deleteDatabase(name) {
        deleted.push(name);
        const request = {};
        queueMicrotask(() => request.onsuccess?.());
        return request;
      },
    };
    await clearOwnedRuntimeStorage({ cacheStorage: null, indexedDb });
    assert.deepEqual(deleted, [...LEGACY_WEBLLM_INDEXED_DB_NAMES]);
  });

  test("blocked legacy database cleanup fails instead of reporting clear-all success", async () => {
    const indexedDb = {
      deleteDatabase(name) {
        const request = {};
        queueMicrotask(() => request.onblocked?.());
        return request;
      },
    };
    await assert.rejects(
      clearOwnedRuntimeStorage({ cacheStorage: null, indexedDb }),
      /blocking deletion of legacy database webllm\/model/u,
    );
  });
} finally {
  await server.close();
}
