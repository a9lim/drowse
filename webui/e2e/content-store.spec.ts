import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const devUrl = "http://127.0.0.1:4176";
const moduleUrl = `/@fs/${resolve("src/hosted/runtime/contentStore.ts")}`;

test("browser content storage resumes, commits, and recovers an evicted object", async ({ page }) => {
  await page.goto(`${devUrl}/LICENSE-Martian-Mono.txt`);
  const result = await page.evaluate(async ({ moduleUrl }) => {
    const content = await import(moduleUrl);
    const bytes = new TextEncoder().encode("verified browser model fixture");
    const digest = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
      (value) => value.toString(16).padStart(2, "0"),
    ).join("");
    const variantId = "e2e-content-store-model";
    const descriptor = {
      sha256: digest,
      expectedBytes: bytes.byteLength,
      url: "https://huggingface.co/a9lim/fixture/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/model.bin",
      revision: "a".repeat(40),
      etag: null,
      partialOwners: [{ id: variantId, kind: "model" as const, catalogSequence: 1 }],
    };

    const store = new content.BrowserContentStore();
    await store.initialize();
    await store.clear();
    await store.beginPartial(descriptor);
    const first = await store.openWrite(digest, 0);
    await first.write(bytes.slice(0, 9));
    const checkpoint = await first.checkpoint();
    const resumed = await store.beginPartial(descriptor);
    const second = await store.openWrite(digest, resumed.contiguousBytes);
    await second.write(bytes.slice(resumed.contiguousBytes));
    await second.checkpoint();
    const partialText = await (await store.partialFile(digest))?.text();
    await store.commitVerified(digest, digest, bytes.byteLength);
    await store.registerInstall({
      id: variantId,
      kind: "model",
      objectHashes: [digest],
      installedAt: 1,
    }, {
      selectAsCurrentModel: true,
    });

    const reopened = new content.BrowserContentStore();
    await reopened.initialize();
    const beforeEviction = {
      installs: await reopened.listInstalls(),
      selected: await reopened.selectedModelVariantId(),
      text: await (await reopened.verifiedFile(digest))?.text(),
    };
    await reopened.removeInstall(variantId, "model");
    const selectedAfterAtomicRemove = await reopened.selectedModelVariantId();
    await reopened.registerInstall({
      id: variantId,
      kind: "model",
      objectHashes: [digest],
      installedAt: 2,
    }, {
      selectAsCurrentModel: true,
    });

    const root = await navigator.storage.getDirectory();
    const drowse = await root.getDirectoryHandle("drowse");
    const objects = await drowse.getDirectoryHandle("objects");
    await objects.removeEntry(`${digest}.data`);
    const afterLiveEviction = {
      fileMissing: await reopened.verifiedFile(digest) === null,
      installs: await reopened.listInstalls(),
      selected: await reopened.selectedModelVariantId(),
    };
    const orphan = await objects.getFileHandle("orphan.tmp", { create: true });
    const orphanWriter = await orphan.createWritable();
    await orphanWriter.write(new Uint8Array([1, 2, 3]));
    await orphanWriter.close();
    await objects.getDirectoryHandle("orphan-directory", { create: true });

    const recovered = new content.BrowserContentStore();
    await recovered.initialize();
    const afterEviction = {
      installs: await recovered.listInstalls(),
      selected: await recovered.selectedModelVariantId(),
      fileMissing: await recovered.verifiedFile(digest) === null,
    };
    const entriesAfterRecovery: string[] = [];
    for await (const [name] of objects.entries()) entriesAfterRecovery.push(name);
    await recovered.clear();
    const remainingObjectEntries: string[] = [];
    for await (const [name] of objects.entries()) remainingObjectEntries.push(name);
    return {
      checkpoint,
      resumedBytes: resumed.contiguousBytes,
      partialText,
      beforeEviction,
      selectedAfterAtomicRemove,
      afterLiveEviction,
      afterEviction,
      entriesAfterRecovery,
      remainingObjectEntries,
    };
  }, { moduleUrl });

  expect(result.checkpoint).toBe(9);
  expect(result.resumedBytes).toBe(9);
  expect(result.partialText).toBe("verified browser model fixture");
  expect(result.beforeEviction.installs).toHaveLength(1);
  expect(result.beforeEviction.selected).toBe("e2e-content-store-model");
  expect(result.beforeEviction.text).toBe("verified browser model fixture");
  expect(result.selectedAfterAtomicRemove).toBeNull();
  expect(result.afterLiveEviction.fileMissing).toBe(true);
  expect(result.afterLiveEviction.installs).toEqual([]);
  expect(result.afterLiveEviction.selected).toBeNull();
  expect(result.afterEviction.installs).toEqual([]);
  expect(result.afterEviction.selected).toBeNull();
  expect(result.afterEviction.fileMissing).toBe(true);
  expect(result.entriesAfterRecovery).toEqual([]);
  expect(result.remainingObjectEntries).toEqual([]);
});

test("verified content retains a newly pending install owner", async ({ page }) => {
  await page.goto(`${devUrl}/LICENSE-Martian-Mono.txt`);
  const owners = await page.evaluate(async ({ moduleUrl }) => {
    const content = await import(moduleUrl);
    const bytes = new TextEncoder().encode("shared verified shard");
    const digest = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
      (value) => value.toString(16).padStart(2, "0"),
    ).join("");
    const descriptor = {
      sha256: digest,
      expectedBytes: bytes.byteLength,
      url: `https://huggingface.co/a9lim/fixture/resolve/${"a".repeat(40)}/shared.bin`,
      revision: "a".repeat(40),
      etag: null,
      partialOwners: [{ id: "first-model", kind: "model" as const, catalogSequence: 1 }],
    };
    const store = new content.BrowserContentStore();
    await store.initialize();
    await store.clear();
    await store.beginPartial(descriptor);
    const writer = await store.openWrite(digest, 0);
    await writer.write(bytes);
    await writer.checkpoint();
    await store.commitVerified(digest, digest, bytes.byteLength);
    await store.registerInstall({
      id: "first-model",
      kind: "model",
      objectHashes: [digest],
      installedAt: 1,
    });
    await store.beginPartial({
      ...descriptor,
      partialOwners: [{ id: "second-model", kind: "model", catalogSequence: 2 }],
    });
    const record = await store.inspectObject(digest);
    await store.clear();
    return record?.partialOwners ?? [];
  }, { moduleUrl });

  expect(owners).toEqual([{ id: "second-model", kind: "model", catalogSequence: 2 }]);
});

test("startup reconciliation does not truncate an actively reserved checkpoint", async ({ page }) => {
  await page.goto(`${devUrl}/LICENSE-Martian-Mono.txt`);
  const result = await page.evaluate(async ({ moduleUrl }) => {
    const content = await import(moduleUrl);
    const bytes = new TextEncoder().encode("active cross-tab checkpoint");
    const digest = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
      (value) => value.toString(16).padStart(2, "0"),
    ).join("");
    const descriptor = {
      sha256: digest,
      expectedBytes: bytes.byteLength,
      url: `https://huggingface.co/a9lim/fixture/resolve/${"a".repeat(40)}/active.bin`,
      revision: "a".repeat(40),
      etag: null,
      partialOwners: [{ id: "active-model", kind: "model" as const, catalogSequence: 1 }],
    };
    const owner = new content.BrowserContentStore();
    await owner.initialize();
    await owner.clear();
    const reservationId = "download:active-checkpoint";
    const reservedAt = Date.now();
    await owner.reserveObjects({
      id: reservationId,
      ownerId: "active-checkpoint",
      ownerKind: "download",
      targetId: "active-model",
      targetKind: "model",
      objectHashes: [digest],
      createdAt: reservedAt,
      updatedAt: reservedAt,
      expiresAt: reservedAt + 60_000,
    });
    await owner.beginPartial(descriptor);
    const writer = await owner.openWrite(digest, 0);
    await writer.write(bytes);

    let releaseStorageLock!: () => void;
    let storageLockHeld!: () => void;
    const storageLockAcquired = new Promise<void>((resolve) => {
      storageLockHeld = resolve;
    });
    const storageLockGate = new Promise<void>((resolve) => {
      releaseStorageLock = resolve;
    });
    const held = navigator.locks.request("drowse-content-store", async () => {
      storageLockHeld();
      await storageLockGate;
    });
    await storageLockAcquired;

    const reopening = new content.BrowserContentStore();
    const initialization = reopening.initialize();
    const locks = navigator.locks as LockManager & {
      query(): Promise<{ pending: Array<{ name?: string }> }>;
    };
    const waitForPending = async (count: number): Promise<void> => {
      for (let attempt = 0; attempt < 200; attempt += 1) {
        const state = await locks.query();
        if (state.pending.filter((lock) => lock.name === "drowse-content-store").length >= count) {
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      throw new Error(`Timed out waiting for ${count} storage-lock requests`);
    };
    await waitForPending(1);
    const checkpoint = writer.checkpoint();
    await waitForPending(2);
    releaseStorageLock();
    await held;
    await Promise.all([initialization, checkpoint]);

    const record = await owner.inspectObject(digest);
    const partial = await owner.partialFile(digest);
    const text = await partial?.text();
    await owner.releaseReservation(reservationId);
    await owner.clear();
    return {
      contiguousBytes: record?.contiguousBytes ?? null,
      fileBytes: partial?.size ?? null,
      text,
    };
  }, { moduleUrl });

  expect(result.contiguousBytes).toBe("active cross-tab checkpoint".length);
  expect(result.fileBytes).toBe("active cross-tab checkpoint".length);
  expect(result.text).toBe("active cross-tab checkpoint");
});

test("completed shards retain pending owners across garbage collection", async ({ page }) => {
  await page.goto(`${devUrl}/LICENSE-Martian-Mono.txt`);
  const result = await page.evaluate(async ({ moduleUrl }) => {
    const content = await import(moduleUrl);
    const bytes = new TextEncoder().encode("completed shard pending install");
    const digest = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
      (value) => value.toString(16).padStart(2, "0"),
    ).join("");
    const owner = { id: "pending-model", kind: "model" as const };
    const descriptor = {
      sha256: digest,
      expectedBytes: bytes.byteLength,
      url: `https://huggingface.co/a9lim/fixture/resolve/${"a".repeat(40)}/shard.bin`,
      revision: "a".repeat(40),
      etag: null,
      partialOwners: [{ ...owner, catalogSequence: 1 }],
    };
    const validPartialOwners = [{ ...owner, objectHashes: [digest] }];

    const store = new content.BrowserContentStore();
    await store.initialize();
    await store.clear();
    await store.beginPartial(descriptor);
    const writer = await store.openWrite(digest, 0);
    await writer.write(bytes);
    await writer.checkpoint();
    await store.commitVerified(digest, digest, bytes.byteLength);
    const firstGc = await store.collectGarbage({ validPartialOwners });

    const reopened = new content.BrowserContentStore();
    await reopened.initialize();
    const secondGc = await reopened.collectGarbage({ validPartialOwners });
    const retained = await (await reopened.verifiedFile(digest))?.text();
    await reopened.registerInstall({
      id: owner.id,
      kind: owner.kind,
      objectHashes: [digest],
      installedAt: 1,
    });

    const database = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open("drowse-hosted-runtime", 5);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    const stored = await new Promise<any>((resolve, reject) => {
      const request = database.transaction("objects").objectStore("objects").get(digest);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    database.close();

    await reopened.removeInstall(owner.id, owner.kind);
    const finalGc = await reopened.collectGarbage({ validPartialOwners });
    const removed = await reopened.verifiedFile(digest) === null;

    const cancelledBytes = new TextEncoder().encode("cancelled optional pack shard");
    const cancelledDigest = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", cancelledBytes)),
      (value) => value.toString(16).padStart(2, "0"),
    ).join("");
    const cancelledOwner = { id: "cancelled-pack", kind: "pack" as const };
    await reopened.beginPartial({
      ...descriptor,
      sha256: cancelledDigest,
      expectedBytes: cancelledBytes.byteLength,
      partialOwners: [{ ...cancelledOwner, catalogSequence: 1 }],
    });
    const cancelledWriter = await reopened.openWrite(cancelledDigest, 0);
    await cancelledWriter.write(cancelledBytes);
    await cancelledWriter.checkpoint();
    await reopened.commitVerified(
      cancelledDigest,
      cancelledDigest,
      cancelledBytes.byteLength,
    );
    await reopened.removeInstall(cancelledOwner.id, cancelledOwner.kind);
    const cancelledGc = await reopened.collectGarbage({
      validPartialOwners: [{ ...cancelledOwner, objectHashes: [cancelledDigest] }],
    });
    const cancelledRemoved = await reopened.verifiedFile(cancelledDigest) === null;
    await reopened.clear();
    return {
      firstGc,
      secondGc,
      retained,
      ownersAfterInstall: stored.partialOwners,
      finalGc,
      removed,
      cancelledGc,
      cancelledRemoved,
    };
  }, { moduleUrl });

  expect(result.firstGc).toEqual([]);
  expect(result.secondGc).toEqual([]);
  expect(result.retained).toBe("completed shard pending install");
  expect(result.ownersAfterInstall).toEqual([]);
  expect(result.finalGc).toHaveLength(1);
  expect(result.removed).toBe(true);
  expect(result.cancelledGc).toHaveLength(1);
  expect(result.cancelledRemoved).toBe(true);
});

test("startup scrubs malformed content metadata and owned orphan entries", async ({ page }) => {
  await page.goto(`${devUrl}/LICENSE-Martian-Mono.txt`);
  const result = await page.evaluate(async ({ moduleUrl }) => {
    const content = await import(moduleUrl);
    const store = new content.BrowserContentStore();
    await store.initialize();
    await store.clear();

    const validBytes = new TextEncoder().encode("valid row beside corrupt metadata");
    const validDigest = Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", validBytes)),
      (value) => value.toString(16).padStart(2, "0"),
    ).join("");
    await store.beginPartial({
      sha256: validDigest,
      expectedBytes: validBytes.byteLength,
      url: `https://huggingface.co/a9lim/fixture/resolve/${"a".repeat(40)}/valid.bin`,
      revision: "a".repeat(40),
      etag: null,
      partialOwners: [{ id: "valid-model", kind: "model", catalogSequence: 1 }],
    });
    const validWriter = await store.openWrite(validDigest, 0);
    await validWriter.write(validBytes);
    await validWriter.checkpoint();
    await store.commitVerified(validDigest, validDigest, validBytes.byteLength);
    await store.registerInstall({
      id: "valid-model",
      kind: "model",
      objectHashes: [validDigest],
      installedAt: 1,
    });
    let emptyInstallRejected = false;
    try {
      await store.registerInstall({
        id: "ghost",
        kind: "model",
        objectHashes: [],
        installedAt: 1,
      });
    } catch {
      emptyInstallRejected = true;
    }

    const digest = "b".repeat(64);
    const root = await navigator.storage.getDirectory();
    const drowse = await root.getDirectoryHandle("drowse");
    const objects = await drowse.getDirectoryHandle("objects");
    const corruptEntry = await objects.getDirectoryHandle(`${digest}.data`, { create: true });
    const nested = await corruptEntry.getFileHandle("nested.bin", { create: true });
    const nestedWriter = await nested.createWritable();
    await nestedWriter.write(new Uint8Array([1, 2, 3]));
    await nestedWriter.close();

    const database = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open("drowse-hosted-runtime", 5);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    const transaction = database.transaction(
      ["objects", "installs", "reservations", "settings"],
      "readwrite",
    );
    transaction.objectStore("objects").put({
      sha256: digest,
      expectedBytes: "not-a-number",
      state: "verified",
    });
    transaction.objectStore("installs").put({
      id: "corrupt-install",
      kind: "model",
      objectHashes: null,
      installedAt: 1,
    });
    transaction.objectStore("installs").put({
      id: "ghost",
      kind: "model",
      objectHashes: [],
      installedAt: 1,
    });
    transaction.objectStore("reservations").put({
      id: "corrupt-reservation",
      objectHashes: null,
      expiresAt: "never",
    });
    transaction.objectStore("settings").put({
      key: "selected_model_variant_id",
      value: "ghost",
    });
    await new Promise<void>((resolve, reject) => {
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
    database.close();

    const recovered = new content.BrowserContentStore();
    await recovered.initialize();
    const validText = await (await recovered.verifiedFile(validDigest))?.text();
    const selected = await recovered.selectedModelVariantId();
    const inspection = await new Promise<Record<string, number>>((resolve, reject) => {
      const request = indexedDB.open("drowse-hosted-runtime", 5);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const opened = request.result;
        const read = opened.transaction(["objects", "installs", "reservations"]);
        const counts: Record<string, number> = {};
        let remaining = 3;
        for (const name of ["objects", "installs", "reservations"]) {
          const count = read.objectStore(name).count();
          count.onerror = () => reject(count.error);
          count.onsuccess = () => {
            counts[name] = count.result;
            remaining -= 1;
            if (remaining === 0) {
              opened.close();
              resolve(counts);
            }
          };
        }
      };
    });
    const entries: string[] = [];
    for await (const [name] of objects.entries()) entries.push(name);
    await recovered.clear();
    return {
      inspection,
      entries,
      validText,
      validDigest,
      selected,
      emptyInstallRejected,
    };
  }, { moduleUrl });

  expect(result.inspection).toEqual({ objects: 1, installs: 1, reservations: 0 });
  expect(result.entries).toEqual([`${result.validDigest}.data`]);
  expect(result.validText).toBe("valid row beside corrupt metadata");
  expect(result.selected).toBeNull();
  expect(result.emptyInstallRejected).toBe(true);
});
