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
    LEGACY_PRODUCT_INDEXED_DB_NAMES,
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
            "saklas-hosted",
            "polythetic-hosted-shell-v1",
            "saklas-hostedother",
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
      "saklas-hosted",
      "polythetic-hosted-shell-v1",
      ...LEGACY_WEBLLM_CACHE_STORAGE_NAMES,
    ]);
  });

  test("clear-all deletes only exact owned legacy IndexedDB database names", async () => {
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
    assert.deepEqual(deleted, [...LEGACY_WEBLLM_INDEXED_DB_NAMES, ...LEGACY_PRODUCT_INDEXED_DB_NAMES]);
  });

  test("clear-all removes legacy product databases that can restore cleared conversations", async () => {
    const deleted = [];
    const indexedDb = {
      deleteDatabase(name) {
        deleted.push(name);
        const request = {};
        queueMicrotask(() => request.onsuccess?.());
        return request;
      },
    };
    await clearOwnedRuntimeStorage({ cacheStorage: null, indexedDb, storage: null });
    for (const slug of ["polythetic", "saklas"]) {
      for (const suffix of ["runtime", "sessions", "catalog", "artifacts", "authoring", "activation-spool"]) {
        assert.ok(deleted.includes(`${slug}-hosted-${suffix}`), `${slug}-hosted-${suffix} must not survive clear-all`);
      }
    }
    assert.ok(!deleted.includes("drowse-hosted-sessions"));
    assert.ok(!deleted.includes("unrelated-app"));
  });

  test("clear-all erases legacy OPFS files while keeping active fallback directory handles usable", async () => {
    function directory(entries = []) {
      const children = new Map(entries);
      return {
        kind: "directory", children,
        async *values() { for (const [name, child] of children) yield { ...child, name }; },
        async getDirectoryHandle(name) {
          if (children.has(name)) return children.get(name);
          throw new DOMException("missing", "NotFoundError");
        },
        async removeEntry(name) { children.delete(name); },
      };
    }
    const file = { kind: "file" };
    const fallback = directory([["chat.json", file]]);
    const old = directory([["sessions", fallback], ["model.bin", file]]);
    const unrelated = directory([["private.txt", file]]);
    const current = directory([["active.bin", file]]);
    const root = directory([["polythetic", old], ["saklas", directory([["chat.json", file]])], ["unrelated", unrelated], ["drowse", current]]);
    await clearOwnedRuntimeStorage({ cacheStorage: null, indexedDb: null, storage: { async getDirectory() { return root; } } });
    assert.equal(fallback.children.size, 0);
    assert.equal(old.children.size, 1);
    assert.equal(root.children.get("saklas").children.size, 0);
    assert.equal(await old.getDirectoryHandle("sessions"), fallback);
    assert.equal(unrelated.children.size, 1);
    assert.equal(current.children.size, 1);
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
