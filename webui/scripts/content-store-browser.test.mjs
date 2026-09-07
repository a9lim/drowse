import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "spa",
  logLevel: "silent",
  server: { host: "127.0.0.1", port: 0, watch: null },
});

let browser;
try {
  await server.listen();
  const origin = server.resolvedUrls?.local[0];
  assert.ok(origin);
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();

  const first = await context.newPage();
  await first.goto(origin, { waitUntil: "domcontentloaded" });
  const seeded = await first.evaluate(async () => {
    const { BrowserContentStore } = await import(
      "/src/hosted/runtime/contentStore.ts"
    );
    const bytes = [
      Uint8Array.of(1, 2, 3, 4),
      Uint8Array.of(5, 6, 7, 8),
    ];
    const digests = [];
    const hex = (value) => [...new Uint8Array(value)]
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
    const store = new BrowserContentStore();
    await store.initialize();
    for (let index = 0; index < bytes.length; index += 1) {
      const sha256 = hex(await crypto.subtle.digest("SHA-256", bytes[index]));
      digests.push(sha256);
      const id = `fixture-model-${index}`;
      await store.beginPartial({
        sha256,
        expectedBytes: bytes[index].byteLength,
        url: `https://example.invalid/${sha256}`,
        revision: "a".repeat(40),
        etag: null,
        partialOwners: [{ id, kind: "model", catalogSequence: 1 }],
      });
      const writer = await store.openWrite(sha256, 0);
      await writer.write(bytes[index]);
      await writer.checkpoint();
      await store.commitVerified(sha256, sha256, bytes[index].byteLength);
      await store.registerInstall({
        id,
        kind: "model",
        objectHashes: [sha256],
        installedAt: Date.now(),
      });
      await store.removeInstall(id, "model");
    }
    const ownerId = crypto.randomUUID();
    const now = Date.now();
    await store.reserveObjects({
      id: `download:${ownerId}`,
      ownerId,
      ownerKind: "download",
      targetId: "fixture-model-1",
      targetKind: "model",
      objectHashes: [digests[1]],
      createdAt: now,
      updatedAt: now,
      expiresAt: now + 60_000,
    });
    const staleBytes = Uint8Array.of(9, 10, 11, 12);
    const staleDigest = hex(await crypto.subtle.digest("SHA-256", staleBytes));
    await store.beginPartial({
      sha256: staleDigest,
      expectedBytes: staleBytes.byteLength,
      url: `https://example.invalid/${staleDigest}`,
      revision: "a".repeat(40),
      etag: null,
      partialOwners: [{
        id: "superseded-model",
        kind: "model",
        catalogSequence: 1,
      }],
    });
    const staleWriter = await store.openWrite(staleDigest, 0);
    await staleWriter.write(staleBytes.slice(0, 2));
    await staleWriter.checkpoint();
    const reconciled = await store.collectGarbage({ validPartialOwners: [] });

    const activeBytes = Uint8Array.of(13, 14, 15, 16);
    const activeDigest = hex(await crypto.subtle.digest("SHA-256", activeBytes));
    await store.beginPartial({
      sha256: activeDigest,
      expectedBytes: activeBytes.byteLength,
      url: `https://example.invalid/${activeDigest}`,
      revision: "a".repeat(40),
      etag: null,
      partialOwners: [{
        id: "superseded-active-model",
        kind: "model",
        catalogSequence: 1,
      }],
    });
    let releaseObjectLock;
    let objectLockHeld;
    const held = new Promise((resolve) => { objectLockHeld = resolve; });
    const released = new Promise((resolve) => { releaseObjectLock = resolve; });
    const objectLock = navigator.locks.request(
      `drowse-content-object-v1:${activeDigest}`,
      { mode: "exclusive" },
      async () => {
        objectLockHeld();
        await released;
      },
    );
    await held;
    const whileActive = await store.collectGarbage({ validPartialOwners: [] });
    const activeWriterProtected = !whileActive.includes(activeDigest) &&
      await store.inspectObject(activeDigest) !== null;
    releaseObjectLock();
    await objectLock;
    const afterRelease = await store.collectGarbage({ validPartialOwners: [] });
    return {
      digests,
      staleOwnerReclaimed: reconciled.includes(staleDigest) &&
        await store.inspectObject(staleDigest) === null,
      activeWriterProtected,
      inactiveWriterReclaimed: afterRelease.includes(activeDigest) &&
        await store.inspectObject(activeDigest) === null,
    };
  });
  assert.equal(seeded.staleOwnerReclaimed, true);
  assert.equal(seeded.activeWriterProtected, true);
  assert.equal(seeded.inactiveWriterReclaimed, true);
  await first.close();

  const second = await context.newPage();
  await second.goto(origin, { waitUntil: "domcontentloaded" });
  const afterRestart = await second.evaluate(async (digests) => {
    const { BrowserContentStore } = await import(
      "/src/hosted/runtime/contentStore.ts"
    );
    const store = new BrowserContentStore();
    await store.initialize();
    const objects = await Promise.all(digests.map(async (sha256) => {
      const object = await store.inspectObject(sha256);
      return object ? { state: object.state, refCount: object.refCount } : null;
    }));
    const database = await new Promise((resolve, reject) => {
      const request = indexedDB.open("drowse-hosted-runtime", 5);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    const transaction = database.transaction("reservations", "readwrite");
    const reservations = transaction.objectStore("reservations");
    const records = await new Promise((resolve, reject) => {
      const request = reservations.getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    const expiredAt = Date.now() - 1_000;
    for (const record of records) {
      reservations.put({
        ...record,
        createdAt: expiredAt - 2_000,
        updatedAt: expiredAt - 1_000,
        expiresAt: expiredAt,
      });
    }
    await new Promise((resolve, reject) => {
      transaction.oncomplete = resolve;
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
    database.close();
    return objects;
  }, seeded.digests);
  assert.deepEqual(afterRestart, [null, { state: "verified", refCount: 0 }]);
  await second.close();

  const third = await context.newPage();
  await third.goto(origin, { waitUntil: "domcontentloaded" });
  const crashedReservationReclaimed = await third.evaluate(async (sha256) => {
    const { BrowserContentStore } = await import(
      "/src/hosted/runtime/contentStore.ts"
    );
    const store = new BrowserContentStore();
    await store.initialize();
    return await store.inspectObject(sha256) === null;
  }, seeded.digests[1]);
  assert.equal(crashedReservationReclaimed, true);

  const runtimeStorageCleanup = await third.evaluate(async () => {
    const {
      LEGACY_WEBLLM_CACHE_STORAGE_NAMES,
      LEGACY_WEBLLM_INDEXED_DB_NAMES,
      clearOwnedRuntimeStorage,
    } = await import("/src/hosted/runtime/legacyRuntimeStorage.ts");
    const unrelatedCache = "unrelated-app-cache";
    const unrelatedDatabase = "unrelated-app-database";
    for (const name of [
      ...LEGACY_WEBLLM_CACHE_STORAGE_NAMES,
      "drowse-hosted-shell-v1",
      unrelatedCache,
    ]) {
      await caches.open(name);
    }
    const openDatabase = (name) => new Promise((resolve, reject) => {
      const request = indexedDB.open(name, 1);
      request.onsuccess = () => {
        request.result.close();
        resolve();
      };
      request.onerror = () => reject(request.error);
    });
    for (const name of [...LEGACY_WEBLLM_INDEXED_DB_NAMES, unrelatedDatabase]) {
      await openDatabase(name);
    }
    await clearOwnedRuntimeStorage();
    return {
      cacheNames: await caches.keys(),
      databaseNames: (await indexedDB.databases())
        .map(({ name }) => name)
        .filter((name) => typeof name === "string"),
      legacyCacheNames: [...LEGACY_WEBLLM_CACHE_STORAGE_NAMES],
      legacyDatabaseNames: [...LEGACY_WEBLLM_INDEXED_DB_NAMES],
      unrelatedCache,
      unrelatedDatabase,
    };
  });
  assert.equal(runtimeStorageCleanup.cacheNames.includes(
    runtimeStorageCleanup.unrelatedCache,
  ), true);
  assert.equal(runtimeStorageCleanup.databaseNames.includes(
    runtimeStorageCleanup.unrelatedDatabase,
  ), true);
  assert.deepEqual(
    runtimeStorageCleanup.cacheNames.filter((name) =>
      runtimeStorageCleanup.legacyCacheNames.includes(name) ||
      name === "drowse-hosted-shell-v1"
    ),
    [],
  );
  assert.deepEqual(
    runtimeStorageCleanup.databaseNames.filter((name) =>
      runtimeStorageCleanup.legacyDatabaseNames.includes(name)
    ),
    [],
  );
  await context.close();

  console.log("content-store restart, orphan, reservation, and cleanup regressions passed");
} finally {
  await browser?.close();
  await server.close();
}
