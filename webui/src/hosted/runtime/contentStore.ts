import type { DeviceLoadRecord } from "../../lib/runtime/contracts";
import { migrateLegacyDatabase, drowseStorageRoot } from "./brandMigration";

const DATABASE_NAME = "drowse-hosted-runtime";
const DATABASE_VERSION = 5;
const OBJECTS_STORE = "objects";
const INSTALLS_STORE = "installs";
const RESERVATIONS_STORE = "reservations";
const LOAD_RECORDS_STORE = "load_records";
const SETTINGS_STORE = "settings";
const MAX_LOAD_RECORDS = 512;
const STORAGE_LOCK = "drowse-content-store";
export const CONTENT_LIFECYCLE_LOCK = "drowse-content-lifecycle-v1";
export const CONTENT_OBJECT_LOCK_PREFIX = "drowse-content-object-v1:";
const OPFS_OBJECTS = "objects";
const SHA256 = /^[a-f0-9]{64}$/;

export interface ContentObjectDescriptor {
  sha256: string;
  expectedBytes: number;
  url: string;
  revision: string;
  etag: string | null;
  partialOwners: PartialContentOwner[];
}

export interface PartialContentOwner {
  id: string;
  kind: "model" | "pack";
  catalogSequence: number;
}

export interface StoredObjectRecord extends ContentObjectDescriptor {
  state: "partial" | "verified";
  contiguousBytes: number;
  refCount: number;
  updatedAt: number;
}

export interface InstalledContentRecord {
  id: string;
  kind: "model" | "pack";
  objectHashes: string[];
  installedAt: number;
}

export interface RegisterInstallOptions {
  selectAsCurrentModel?: boolean;
}

export interface ContentReservationRecord {
  id: string;
  ownerId: string;
  ownerKind: "download";
  targetId: string;
  targetKind: "model" | "pack";
  objectHashes: string[];
  createdAt: number;
  updatedAt: number;
  expiresAt: number;
}

export interface ValidPartialOwner {
  id: string;
  kind: PartialContentOwner["kind"];
  objectHashes: readonly string[];
}

export interface GarbageCollectionOptions {
  validPartialOwners?: readonly ValidPartialOwner[];
  activeReservationIds?: readonly string[];
}

export interface ContentWriteSession {
  readonly offset: number;
  write(chunk: Uint8Array): Promise<number>;
  checkpoint(): Promise<number>;
}

export interface ContentAddressedStore {
  initialize(): Promise<void>;
  beginPartial(descriptor: ContentObjectDescriptor): Promise<StoredObjectRecord>;
  resetPartial(descriptor: ContentObjectDescriptor): Promise<StoredObjectRecord>;
  openWrite(sha256: string, offset: number): Promise<ContentWriteSession>;
  inspectObject(sha256: string): Promise<StoredObjectRecord | null>;
  partialFile(sha256: string): Promise<File | null>;
  commitVerified(sha256: string, computedSha256: string, computedBytes: number): Promise<void>;
  verifiedFile(sha256: string): Promise<File | null>;
  reserveObjects(record: ContentReservationRecord): Promise<void>;
  releaseReservation(id: string): Promise<void>;
  registerInstall(
    record: InstalledContentRecord,
    options?: RegisterInstallOptions,
  ): Promise<void>;
  removeInstall(id: string, expectedKind?: InstalledContentRecord["kind"]): Promise<string[]>;
  listInstalls(): Promise<InstalledContentRecord[]>;
  listLoadRecords(): Promise<DeviceLoadRecord[]>;
  recordLoad(record: DeviceLoadRecord): Promise<void>;
  selectedModelVariantId(): Promise<string | null>;
  setSelectedModelVariantId(modelVariantId: string | null): Promise<void>;
  collectGarbage(options?: GarbageCollectionOptions): Promise<string[]>;
  clear(): Promise<void>;
}

export async function hasSelectedInstalledModel(): Promise<boolean> {
  if (typeof indexedDB === "undefined") return false;
  const database = await openDatabase();
  try {
    const transaction = database.transaction(
      [INSTALLS_STORE, SETTINGS_STORE],
      "readonly",
    );
    const completion = transactionDone(transaction);
    const [installs, selected] = await Promise.all([
      requestValue<InstalledContentRecord[]>(
        transaction.objectStore(INSTALLS_STORE).getAll(),
      ),
      requestValue<{ key: string; value: unknown } | undefined>(
        transaction.objectStore(SETTINGS_STORE).get("selected_model_variant_id"),
      ),
    ]);
    await completion;
    return typeof selected?.value === "string" && installs.some(
      (install) => install.kind === "model" && install.id === selected.value,
    );
  } finally {
    database.close();
  }
}

export class BrowserContentStore implements ContentAddressedStore {
  private database: IDBDatabase | null = null;
  private directory: FileSystemDirectoryHandle | null = null;
  private initialization: Promise<void> | null = null;

  initialize(): Promise<void> {
    if (this.initialization === null) {
      const initialization = this.initializeOnce();
      this.initialization = initialization;
      void initialization.catch(() => {
        if (this.initialization === initialization) this.initialization = null;
      });
    }
    return this.initialization;
  }

  private async initializeOnce(): Promise<void> {
    if (!navigator.storage?.getDirectory) {
      throw new Error("Origin-private file system storage is unavailable");
    }
    const database = await openDatabase();
    database.onversionchange = () => database.close();
    try {
      const root = await navigator.storage.getDirectory();
      const drowseDirectory = await drowseStorageRoot(root);
      const directory = await drowseDirectory.getDirectoryHandle(OPFS_OBJECTS, {
        create: true,
      });
      this.database = database;
      this.directory = directory;
      database.onversionchange = () => {
        database.close();
        if (this.database === database) {
          this.database = null;
          this.directory = null;
          this.initialization = null;
        }
      };
      await this.withStorageLock(() => this.reconcileMissingFiles());
    } catch (error) {
      database.close();
      if (this.database === database) {
        this.database = null;
        this.directory = null;
      }
      throw error;
    }
  }

  async beginPartial(
    descriptor: ContentObjectDescriptor,
  ): Promise<StoredObjectRecord> {
    return this.withStorageLock(async () => {
      validateDescriptor(descriptor);
      const database = this.requireDatabase();
      const directory = this.requireDirectory();
      const existing = await getRecord<StoredObjectRecord>(
        database,
        OBJECTS_STORE,
        descriptor.sha256,
      );
      const handle = await directory.getFileHandle(objectFileName(descriptor.sha256), {
        create: true,
      });
      const file = await handle.getFile();
      if (existing?.state === "verified") {
        if (
          existing.expectedBytes === descriptor.expectedBytes &&
          file.size === existing.expectedBytes
        ) {
          const partialOwners = mergePartialOwners(
            existing.partialOwners,
            descriptor.partialOwners,
          );
          if (samePartialOwners(existing.partialOwners, partialOwners)) return existing;
          const retained = {
            ...existing,
            partialOwners,
            updatedAt: Date.now(),
          } satisfies StoredObjectRecord;
          await putRecord(database, OBJECTS_STORE, retained);
          return retained;
        }
        await invalidateStoredObject(database, descriptor.sha256);
        await truncate(handle);
        const recovered: StoredObjectRecord = {
          ...descriptor,
          partialOwners: normalizePartialOwners(descriptor.partialOwners),
          state: "partial",
          contiguousBytes: 0,
          refCount: 0,
          updatedAt: Date.now(),
        };
        await putRecord(database, OBJECTS_STORE, recovered);
        return recovered;
      }
      const compatible =
        existing?.state === "partial" &&
        existing.expectedBytes === descriptor.expectedBytes &&
        existing.url === descriptor.url &&
        existing.revision === descriptor.revision &&
        (existing.etag === null || descriptor.etag === null || existing.etag === descriptor.etag);
      if (!compatible) {
        await truncate(handle);
        if (existing) {
          await deleteRecord(database, OBJECTS_STORE, descriptor.sha256);
        }
      }

      let current = 0;
      if (compatible && existing) {
        const checkpoint = Math.min(existing.contiguousBytes, descriptor.expectedBytes);
        if (file.size > checkpoint) await truncateTo(handle, checkpoint);
        current = Math.min(file.size, checkpoint);
      }
      const record: StoredObjectRecord = {
        ...descriptor,
        etag: compatible && existing?.etag ? existing.etag : descriptor.etag,
        partialOwners: compatible && existing
          ? mergePartialOwners(existing.partialOwners, descriptor.partialOwners)
          : normalizePartialOwners(descriptor.partialOwners),
        state: "partial",
        contiguousBytes: current,
        refCount: existing?.expectedBytes === descriptor.expectedBytes
          ? existing.refCount
          : 0,
        updatedAt: Date.now(),
      };
      await putRecord(database, OBJECTS_STORE, record);
      return record;
    });
  }

  async resetPartial(
    descriptor: ContentObjectDescriptor,
  ): Promise<StoredObjectRecord> {
    return this.withStorageLock(async () => {
      validateDescriptor(descriptor);
      const database = this.requireDatabase();
      const directory = this.requireDirectory();
      const existing = await getRecord<StoredObjectRecord>(
        database,
        OBJECTS_STORE,
        descriptor.sha256,
      );
      if (existing?.state === "verified") {
        throw new Error(`Verified object ${descriptor.sha256} cannot be reset`);
      }
      const handle = await directory.getFileHandle(objectFileName(descriptor.sha256), {
        create: true,
      });
      await truncate(handle);
      const record: StoredObjectRecord = {
        ...descriptor,
        partialOwners:
          existing?.state === "partial" &&
            existing.expectedBytes === descriptor.expectedBytes &&
            existing.url === descriptor.url &&
            existing.revision === descriptor.revision
            ? mergePartialOwners(existing.partialOwners, descriptor.partialOwners)
            : normalizePartialOwners(descriptor.partialOwners),
        state: "partial",
        contiguousBytes: 0,
        refCount: existing?.expectedBytes === descriptor.expectedBytes
          ? existing.refCount
          : 0,
        updatedAt: Date.now(),
      };
      await putRecord(database, OBJECTS_STORE, record);
      return record;
    });
  }

  async openWrite(sha256: string, offset: number): Promise<ContentWriteSession> {
    validateSha(sha256);
    return this.withStorageLock(async () => {
      const database = this.requireDatabase();
      const record = await getRecord<StoredObjectRecord>(
        database,
        OBJECTS_STORE,
        sha256,
      );
      if (!record || record.state !== "partial") {
        throw new Error(`No resumable partial exists for ${sha256}`);
      }
      if (record.contiguousBytes !== offset) {
        throw new Error(`Expected byte offset ${record.contiguousBytes}, received ${offset}`);
      }
      const handle = await this.requireDirectory().getFileHandle(objectFileName(sha256));
      const file = await handle.getFile();
      if (file.size !== offset) {
        throw new Error(`Stored object size ${file.size} does not match checkpoint ${offset}`);
      }
      const writable = await handle.createWritable({ keepExistingData: true });
      await writable.seek(offset);
      return new BrowserContentWriteSession(
        database,
        record,
        handle,
        writable,
        offset,
        (operation) => this.withStorageLock(operation),
      );
    });
  }

  async inspectObject(sha256: string): Promise<StoredObjectRecord | null> {
    validateSha(sha256);
    const record = await getRecord<StoredObjectRecord>(
      this.requireDatabase(),
      OBJECTS_STORE,
      sha256,
    );
    return record ? { ...record, partialOwners: [...record.partialOwners] } : null;
  }

  async partialFile(sha256: string): Promise<File | null> {
    validateSha(sha256);
    const record = await getRecord<StoredObjectRecord>(
      this.requireDatabase(),
      OBJECTS_STORE,
      sha256,
    );
    if (!record || record.state !== "partial") return null;
    return this.readFile(sha256, record.contiguousBytes);
  }

  async commitVerified(
    sha256: string,
    computedSha256: string,
    computedBytes: number,
  ): Promise<void> {
    await this.withStorageLock(async () => {
      validateSha(sha256);
      validateSha(computedSha256);
      const database = this.requireDatabase();
      const record = await getRecord<StoredObjectRecord>(database, OBJECTS_STORE, sha256);
      if (!record || record.state !== "partial") {
        throw new Error(`No partial object exists for ${sha256}`);
      }
      if (computedSha256 !== record.sha256 || computedBytes !== record.expectedBytes) {
        throw new Error("The downloaded object does not match its signed digest and size");
      }
      const file = await this.readFile(sha256, record.expectedBytes);
      if (!file || file.size !== record.expectedBytes) {
        throw new Error("The downloaded object is incomplete");
      }
      await putRecord(database, OBJECTS_STORE, {
        ...record,
        state: "verified",
        partialOwners: normalizePartialOwners(record.partialOwners),
        contiguousBytes: record.expectedBytes,
        updatedAt: Date.now(),
      } satisfies StoredObjectRecord);
    });
  }

  async verifiedFile(sha256: string): Promise<File | null> {
    return this.withStorageLock(async () => {
      validateSha(sha256);
      const database = this.requireDatabase();
      const record = await getRecord<StoredObjectRecord>(
        database,
        OBJECTS_STORE,
        sha256,
      );
      if (!record || record.state !== "verified") return null;
      const file = await this.readFile(sha256, record.expectedBytes);
      if (file) return file;
      await removeEntry(this.requireDirectory(), objectFileName(sha256));
      await invalidateStoredObject(database, sha256);
      return null;
    });
  }

  async reserveObjects(record: ContentReservationRecord): Promise<void> {
    await this.withStorageLock(async () => {
      if (!isContentReservationRecord(record)) {
        throw new Error("Content reservation metadata is invalid");
      }
      const objectHashes = [...new Set(record.objectHashes)];
      if (!objectHashes.length) throw new Error("Content reservation must not be empty");
      for (const hash of objectHashes) validateSha(hash);
      if (!Number.isSafeInteger(record.expiresAt) || record.expiresAt <= Date.now()) {
        throw new Error("Content reservation expiry must be in the future");
      }
      await putRecord(this.requireDatabase(), RESERVATIONS_STORE, {
        ...record,
        objectHashes,
      } satisfies ContentReservationRecord);
    });
  }

  async releaseReservation(id: string): Promise<void> {
    if (!id) throw new Error("Content reservation ID must not be empty");
    await this.withStorageLock(() =>
      deleteRecord(this.requireDatabase(), RESERVATIONS_STORE, id)
    );
  }

  async registerInstall(
    record: InstalledContentRecord,
    options: RegisterInstallOptions = {},
  ): Promise<void> {
    if (!isInstalledContentRecord(record)) {
      throw new Error("Installed content metadata is invalid");
    }
    if (options.selectAsCurrentModel === true && record.kind !== "model") {
      throw new Error("Only a model install can become the current model");
    }
    await this.withStorageLock(async () => {
      const database = this.requireDatabase();
      const uniqueHashes = [...new Set(record.objectHashes)];
      for (const hash of uniqueHashes) validateSha(hash);
      const transaction = database.transaction(
        options.selectAsCurrentModel === true
          ? [OBJECTS_STORE, INSTALLS_STORE, SETTINGS_STORE]
          : [OBJECTS_STORE, INSTALLS_STORE],
        "readwrite",
      );
      const objects = transaction.objectStore(OBJECTS_STORE);
      const installs = transaction.objectStore(INSTALLS_STORE);
      const previous = await requestValue<InstalledContentRecord | undefined>(
        installs.get(record.id),
      );
      const previousHashes = new Set(previous?.objectHashes ?? []);
      const nextHashes = new Set(uniqueHashes);

      for (const hash of nextHashes) {
        const object = await requestValue<StoredObjectRecord | undefined>(objects.get(hash));
        if (!object || object.state !== "verified") {
          transaction.abort();
          throw new Error(`Install ${record.id} references unverified object ${hash}`);
        }
        const partialOwners = normalizePartialOwners(object.partialOwners).filter(
          (owner) => !matchesPartialOwner(owner, record.id, record.kind),
        );
        objects.put({
          ...object,
          refCount: object.refCount + (previousHashes.has(hash) ? 0 : 1),
          partialOwners,
          updatedAt: samePartialOwners(object.partialOwners, partialOwners)
            ? object.updatedAt
            : Date.now(),
        });
      }
      for (const hash of previousHashes) {
        if (nextHashes.has(hash)) continue;
        const object = await requestValue<StoredObjectRecord | undefined>(objects.get(hash));
        if (object) objects.put({ ...object, refCount: Math.max(0, object.refCount - 1) });
      }
      installs.put({ ...record, objectHashes: uniqueHashes });
      if (options.selectAsCurrentModel === true) {
        transaction.objectStore(SETTINGS_STORE).put({
          key: "selected_model_variant_id",
          value: record.id,
        });
      }
      await transactionDone(transaction);
    });
  }

  async removeInstall(
    id: string,
    expectedKind?: InstalledContentRecord["kind"],
  ): Promise<string[]> {
    return this.withStorageLock(async () => {
      const database = this.requireDatabase();
      const transaction = database.transaction(
        [OBJECTS_STORE, INSTALLS_STORE, SETTINGS_STORE],
        "readwrite",
      );
      const objects = transaction.objectStore(OBJECTS_STORE);
      const installs = transaction.objectStore(INSTALLS_STORE);
      const settings = transaction.objectStore(SETTINGS_STORE);
      const [existing, storedObjects, selected] = await Promise.all([
        requestValue<InstalledContentRecord | undefined>(installs.get(id)),
        requestValue<StoredObjectRecord[]>(objects.getAll()),
        requestValue<{ key: string; value: unknown } | undefined>(
          settings.get("selected_model_variant_id"),
        ),
      ]);
      if (expectedKind && existing && existing.kind !== expectedKind) {
        transaction.abort();
        throw new Error(`Install ${id} is a ${existing.kind}, not a ${expectedKind}`);
      }
      const updated = new Map(storedObjects.map((object) => [object.sha256, object]));
      for (const hash of existing?.objectHashes ?? []) {
        const object = updated.get(hash);
        if (object) {
          updated.set(hash, {
            ...object,
            refCount: Math.max(0, object.refCount - 1),
          });
        }
      }
      const ownerKind = expectedKind ?? existing?.kind;
      for (const [hash, object] of updated) {
        const currentOwners = normalizePartialOwners(object.partialOwners);
        const partialOwners = currentOwners.filter(
          (owner) => !matchesPartialOwner(owner, id, ownerKind),
        );
        if (!samePartialOwners(object.partialOwners, partialOwners)) {
          updated.set(hash, { ...object, partialOwners, updatedAt: Date.now() });
        }
      }
      for (const object of updated.values()) objects.put(object);
      if (existing) installs.delete(id);
      if (
        existing?.kind === "model" &&
        selected?.value === id
      ) {
        settings.delete("selected_model_variant_id");
      }
      await transactionDone(transaction);
      return existing?.objectHashes ?? [];
    });
  }

  async listInstalls(): Promise<InstalledContentRecord[]> {
    return getAllRecords<InstalledContentRecord>(this.requireDatabase(), INSTALLS_STORE);
  }

  async listLoadRecords(): Promise<DeviceLoadRecord[]> {
    return this.withStorageLock(async () => {
      const transaction = this.requireDatabase().transaction(
        LOAD_RECORDS_STORE,
        "readwrite",
      );
      const completion = transactionDone(transaction);
      const records: DeviceLoadRecord[] = [];
      const request = transaction.objectStore(LOAD_RECORDS_STORE).openCursor();
      await new Promise<void>((resolve, reject) => {
        request.onerror = () => reject(
          request.error ?? new Error("IndexedDB load-record cursor failed"),
        );
        request.onsuccess = () => {
          const cursor = request.result;
          if (!cursor) {
            resolve();
            return;
          }
          try {
            validateLoadRecord(cursor.value as DeviceLoadRecord);
            records.push({ ...(cursor.value as DeviceLoadRecord) });
          } catch {
            cursor.delete();
          }
          cursor.continue();
        };
      });
      await completion;
      return records.sort((left, right) => left.recordedAt - right.recordedAt);
    });
  }

  async recordLoad(record: DeviceLoadRecord): Promise<void> {
    validateLoadRecord(record);
    await this.withStorageLock(async () => {
      const database = this.requireDatabase();
      const transaction = database.transaction(LOAD_RECORDS_STORE, "readwrite");
      const store = transaction.objectStore(LOAD_RECORDS_STORE);
      const stored = { ...record };
      while (await requestValue(store.get(loadRecordKey(stored)))) {
        stored.recordedAt += 1;
      }
      await requestValue(store.put(stored));
      const records = await requestValue<DeviceLoadRecord[]>(store.getAll());
      records.sort((left, right) => left.recordedAt - right.recordedAt);
      for (const expired of records.slice(0, -MAX_LOAD_RECORDS)) {
        store.delete(loadRecordKey(expired));
      }
      await transactionDone(transaction);
    });
  }

  async selectedModelVariantId(): Promise<string | null> {
    const record = await getRecord<{ key: string; value: unknown }>(
      this.requireDatabase(),
      SETTINGS_STORE,
      "selected_model_variant_id",
    );
    return typeof record?.value === "string" && record.value.length <= 256
      ? record.value
      : null;
  }

  async setSelectedModelVariantId(modelVariantId: string | null): Promise<void> {
    if (modelVariantId !== null && (!modelVariantId || modelVariantId.length > 256)) {
      throw new Error("Selected model variant is invalid");
    }
    await this.withStorageLock(async () => {
      const database = this.requireDatabase();
      if (modelVariantId === null) {
        await deleteRecord(database, SETTINGS_STORE, "selected_model_variant_id");
      } else {
        await putRecord(database, SETTINGS_STORE, {
          key: "selected_model_variant_id",
          value: modelVariantId,
        });
      }
    });
  }

  async collectGarbage(options: GarbageCollectionOptions = {}): Promise<string[]> {
    return this.withStorageLock(async () => {
      const database = this.requireDatabase();
      const objects = await getAllRecords<StoredObjectRecord>(database, OBJECTS_STORE);
      const reservations = await getAllRecords<ContentReservationRecord>(
        database,
        RESERVATIONS_STORE,
      );
      const now = Date.now();
      const activeReservationIds = new Set(options.activeReservationIds ?? []);
      for (const id of activeReservationIds) {
        if (!id || id.length > 256) throw new Error("Active reservation ID is invalid");
      }
      const reserved = new Set<string>();
      for (const reservation of reservations) {
        if (reservation.expiresAt <= now && !activeReservationIds.has(reservation.id)) {
          await deleteRecord(database, RESERVATIONS_STORE, reservation.id);
          continue;
        }
        for (const hash of reservation.objectHashes) reserved.add(hash);
      }
      for (const hash of await activeContentObjectHashes()) reserved.add(hash);
      const validPartialOwners = options.validPartialOwners === undefined
        ? null
        : partialOwnerManifest(options.validPartialOwners);
      const removed: string[] = [];
      for (const object of objects) {
        const currentOwners = normalizePartialOwners(object.partialOwners);
        const partialOwners = validPartialOwners === null
          ? currentOwners
          : currentOwners.filter((owner) =>
              validPartialOwners.get(partialOwnerKey(owner))?.has(object.sha256)
            );
        if (!samePartialOwners(object.partialOwners, partialOwners)) {
          await putRecord(database, OBJECTS_STORE, {
            ...object,
            partialOwners,
            updatedAt: Date.now(),
          } satisfies StoredObjectRecord);
        }
        if (
          object.refCount !== 0 || reserved.has(object.sha256) ||
          partialOwners.length > 0
        ) continue;
        await removeFile(this.requireDirectory(), objectFileName(object.sha256));
        await deleteRecord(database, OBJECTS_STORE, object.sha256);
        removed.push(object.sha256);
      }
      return removed;
    });
  }

  async clear(): Promise<void> {
    if (!navigator.locks?.request) {
      throw new Error("The Web Locks API is unavailable");
    }
    await navigator.locks.request(
      CONTENT_LIFECYCLE_LOCK,
      { mode: "exclusive" },
      () => this.withStorageLock(async () => {
        const database = this.requireDatabase();
        const directory = this.requireDirectory();
        const transaction = database.transaction(
          [
            OBJECTS_STORE,
            INSTALLS_STORE,
            RESERVATIONS_STORE,
            LOAD_RECORDS_STORE,
            SETTINGS_STORE,
          ],
          "readwrite",
        );
        transaction.objectStore(OBJECTS_STORE).clear();
        transaction.objectStore(INSTALLS_STORE).clear();
        transaction.objectStore(RESERVATIONS_STORE).clear();
        transaction.objectStore(LOAD_RECORDS_STORE).clear();
        transaction.objectStore(SETTINGS_STORE).clear();
        await transactionDone(transaction);
        for (const name of await entryNames(directory)) {
          await removeEntry(directory, name);
        }
      }),
    );
  }

  private async reconcileMissingFiles(): Promise<void> {
    const database = this.requireDatabase();
    const directory = this.requireDirectory();
    const objects = await getValidRecords<StoredObjectRecord>(
      database,
      OBJECTS_STORE,
      isStoredObjectRecord,
    );
    const reservations = await getValidRecords<ContentReservationRecord>(
      database,
      RESERVATIONS_STORE,
      isContentReservationRecord,
    );
    const reserved = new Set<string>();
    const now = Date.now();
    for (const reservation of reservations) {
      if (reservation.expiresAt <= now) {
        await deleteRecord(database, RESERVATIONS_STORE, reservation.id);
      } else {
        for (const hash of reservation.objectHashes) reserved.add(hash);
      }
    }
    for (const hash of await activeContentObjectHashes()) reserved.add(hash);
    const valid = new Map<string, StoredObjectRecord>();
    for (const object of objects) {
      if (reserved.has(object.sha256) && object.state === "partial") {
        valid.set(object.sha256, object);
        continue;
      }
      const file = await this.readFile(object.sha256, null);
      if (
        !file || file.size > object.expectedBytes ||
        (object.state === "verified" && file.size !== object.expectedBytes)
      ) {
        await removeEntry(directory, objectFileName(object.sha256));
        await deleteRecord(database, OBJECTS_STORE, object.sha256);
        continue;
      }
      let reconciled = object;
      if (object.state === "partial") {
        const partialOwners = normalizePartialOwners(object.partialOwners);
        const checkpoint = Math.min(object.contiguousBytes, object.expectedBytes);
        let fileSize = file.size;
        if (file.size > checkpoint) {
          const handle = await directory.getFileHandle(objectFileName(object.sha256));
          await truncateTo(handle, checkpoint);
          fileSize = checkpoint;
        }
        if (
          fileSize !== object.contiguousBytes ||
          object.refCount !== 0 ||
          !samePartialOwners(object.partialOwners, partialOwners)
        ) {
          reconciled = {
            ...object,
            contiguousBytes: fileSize,
            partialOwners,
            refCount: 0,
            updatedAt: Date.now(),
          };
        }
        if (partialOwners.length === 0 && !reserved.has(object.sha256)) {
          await removeFile(directory, objectFileName(object.sha256));
          await deleteRecord(database, OBJECTS_STORE, object.sha256);
          continue;
        }
      } else {
        const partialOwners = normalizePartialOwners(object.partialOwners);
        if (!samePartialOwners(object.partialOwners, partialOwners)) {
          reconciled = { ...object, partialOwners, updatedAt: Date.now() };
        }
      }
      if (reconciled !== object) await putRecord(database, OBJECTS_STORE, reconciled);
      valid.set(object.sha256, reconciled);
    }

    for (const name of await entryNames(directory)) {
      const match = /^([a-f0-9]{64})\.data$/.exec(name);
      if (!match || !valid.has(match[1])) await removeEntry(directory, name);
    }

    const referenceCounts = new Map<string, number>();
    const installedModelIds = new Set<string>();
    const installs = await getValidRecords<InstalledContentRecord>(
      database,
      INSTALLS_STORE,
      isInstalledContentRecord,
    );
    for (const install of installs) {
      const hashes = [...new Set(install.objectHashes)];
      if (hashes.some((hash) => valid.get(hash)?.state !== "verified")) {
        await deleteRecord(database, INSTALLS_STORE, install.id);
        continue;
      }
      if (install.kind === "model") installedModelIds.add(install.id);
      for (const hash of hashes) {
        referenceCounts.set(hash, (referenceCounts.get(hash) ?? 0) + 1);
      }
      if (hashes.length !== install.objectHashes.length) {
        await putRecord(database, INSTALLS_STORE, { ...install, objectHashes: hashes });
      }
    }
    for (const [hash, object] of valid) {
      const refCount = referenceCounts.get(hash) ?? 0;
      if (
        object.state === "verified" && refCount === 0 &&
        normalizePartialOwners(object.partialOwners).length === 0 &&
        !reserved.has(hash)
      ) {
        await removeFile(directory, objectFileName(hash));
        await deleteRecord(database, OBJECTS_STORE, hash);
        valid.delete(hash);
      } else if (object.refCount !== refCount) {
        await putRecord(database, OBJECTS_STORE, { ...object, refCount });
      }
    }
    const selected = await getRecord<{ key: string; value: unknown }>(
      database,
      SETTINGS_STORE,
      "selected_model_variant_id",
    );
    if (
      selected !== undefined &&
      (typeof selected.value !== "string" || !installedModelIds.has(selected.value))
    ) {
      await deleteRecord(database, SETTINGS_STORE, "selected_model_variant_id");
    }
  }

  private async readFile(sha256: string, expectedBytes: number | null): Promise<File | null> {
    try {
      const handle = await this.requireDirectory().getFileHandle(objectFileName(sha256));
      const file = await handle.getFile();
      return expectedBytes === null || file.size === expectedBytes ? file : null;
    } catch (error) {
      if (
        error instanceof DOMException &&
        (error.name === "NotFoundError" || error.name === "TypeMismatchError")
      ) return null;
      throw error;
    }
  }

  private requireDatabase(): IDBDatabase {
    if (!this.database) throw new Error("Content store has not been initialized");
    return this.database;
  }

  private requireDirectory(): FileSystemDirectoryHandle {
    if (!this.directory) throw new Error("Content store has not been initialized");
    return this.directory;
  }

  private withStorageLock<T>(operation: () => Promise<T>): Promise<T> {
    if (!navigator.locks?.request) {
      return Promise.reject(new Error("The Web Locks API is unavailable"));
    }
    return navigator.locks.request(
      STORAGE_LOCK,
      { mode: "exclusive" },
      async (lock) => {
        if (!lock) throw new Error("The Drowse content-store lock was not acquired");
        return operation();
      },
    );
  }
}

async function activeContentObjectHashes(): Promise<Set<string>> {
  const active = new Set<string>();
  if (!navigator.locks?.query) return active;
  const snapshot = await navigator.locks.query();
  for (const lock of snapshot.held ?? []) {
    if (
      lock.mode !== "exclusive" || typeof lock.name !== "string" ||
      !lock.name.startsWith(CONTENT_OBJECT_LOCK_PREFIX)
    ) {
      continue;
    }
    const sha256 = lock.name.slice(CONTENT_OBJECT_LOCK_PREFIX.length);
    if (SHA256.test(sha256)) active.add(sha256);
  }
  return active;
}

class BrowserContentWriteSession implements ContentWriteSession {
  private currentOffset: number;
  private checkpointPromise: Promise<number> | null = null;

  constructor(
    private readonly database: IDBDatabase,
    private readonly record: StoredObjectRecord,
    private readonly handle: FileSystemFileHandle,
    private readonly writable: FileSystemWritableFileStream,
    offset: number,
    private readonly withStorageLock:
      <T>(operation: () => Promise<T>) => Promise<T>,
  ) {
    this.currentOffset = offset;
  }

  get offset(): number {
    return this.currentOffset;
  }

  async write(chunk: Uint8Array): Promise<number> {
    if (this.checkpointPromise) throw new Error("Content write session is closed");
    if (this.currentOffset + chunk.byteLength > this.record.expectedBytes) {
      throw new Error("The chunk exceeds the signed object size");
    }
    if (chunk.byteLength === 0) return this.currentOffset;
    // WebKit writes the entire backing buffer of a typed-array view.
    await this.writable.write(Uint8Array.from(chunk).buffer);
    this.currentOffset += chunk.byteLength;
    return this.currentOffset;
  }

  checkpoint(): Promise<number> {
    if (!this.checkpointPromise) this.checkpointPromise = this.finish();
    return this.checkpointPromise;
  }

  private async finish(): Promise<number> {
    await this.writable.close();
    const file = await this.handle.getFile();
    if (file.size !== this.currentOffset) {
      throw new Error(
        `Stored object size ${file.size} does not match checkpoint ${this.currentOffset}`,
      );
    }
    return this.withStorageLock(async () => {
      const latest = await getRecord<StoredObjectRecord>(
        this.database,
        OBJECTS_STORE,
        this.record.sha256,
      );
      if (
        !latest || latest.state !== "partial" ||
        latest.contiguousBytes !== this.record.contiguousBytes
      ) {
        throw new Error(`Stored partial ${this.record.sha256} changed during the write`);
      }
      await putRecord(this.database, OBJECTS_STORE, {
        ...latest,
        contiguousBytes: this.currentOffset,
        updatedAt: Date.now(),
      } satisfies StoredObjectRecord);
      return this.currentOffset;
    });
  }
}

async function invalidateStoredObject(
  database: IDBDatabase,
  sha256: string,
): Promise<void> {
  const transaction = database.transaction(
    [OBJECTS_STORE, INSTALLS_STORE, SETTINGS_STORE],
    "readwrite",
  );
  const objects = transaction.objectStore(OBJECTS_STORE);
  const installs = transaction.objectStore(INSTALLS_STORE);
  const settings = transaction.objectStore(SETTINGS_STORE);
  const [records, selected] = await Promise.all([
    requestValue<InstalledContentRecord[]>(installs.getAll()),
    requestValue<{ key: string; value: unknown } | undefined>(
      settings.get("selected_model_variant_id"),
    ),
  ]);
  const affected = records.filter((record) => record.objectHashes.includes(sha256));
  const decrements = new Map<string, number>();
  for (const install of affected) {
    for (const hash of new Set(install.objectHashes)) {
      decrements.set(hash, (decrements.get(hash) ?? 0) + 1);
    }
    installs.delete(install.id);
  }
  if (
    typeof selected?.value === "string" &&
    affected.some((install) => install.kind === "model" && install.id === selected.value)
  ) {
    settings.delete("selected_model_variant_id");
  }
  for (const [hash, decrement] of decrements) {
    if (hash === sha256) continue;
    const object = await requestValue<StoredObjectRecord | undefined>(objects.get(hash));
    if (object) {
      objects.put({ ...object, refCount: Math.max(0, object.refCount - decrement) });
    }
  }
  objects.delete(sha256);
  await transactionDone(transaction);
}

async function openDatabase(): Promise<IDBDatabase> {
  const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
  request.onupgradeneeded = () => {
    const database = request.result;
    if (!database.objectStoreNames.contains(OBJECTS_STORE)) {
      database.createObjectStore(OBJECTS_STORE, { keyPath: "sha256" });
    }
    if (!database.objectStoreNames.contains(INSTALLS_STORE)) {
      database.createObjectStore(INSTALLS_STORE, { keyPath: "id" });
    }
    if (!database.objectStoreNames.contains(RESERVATIONS_STORE)) {
      database.createObjectStore(RESERVATIONS_STORE, { keyPath: "id" });
    }
    if (!database.objectStoreNames.contains(LOAD_RECORDS_STORE)) {
      database.createObjectStore(LOAD_RECORDS_STORE, {
        keyPath: [
          "modelVariantId",
          "runtimeIdentitySha256",
          "deviceSignature",
          "contextTokens",
          "recordedAt",
        ],
      });
    }
    if (!database.objectStoreNames.contains(SETTINGS_STORE)) {
      database.createObjectStore(SETTINGS_STORE, { keyPath: "key" });
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
      new Error("Another Drowse tab is blocking the storage upgrade"),
    );
  });
  await migrateLegacyDatabase(database, [
    OBJECTS_STORE,
    INSTALLS_STORE,
    RESERVATIONS_STORE,
    LOAD_RECORDS_STORE,
    SETTINGS_STORE,
  ]);
  return database;
}

function getRecord<T>(
  database: IDBDatabase,
  storeName: string,
  key: IDBValidKey,
): Promise<T | undefined> {
  const transaction = database.transaction(storeName, "readonly");
  return requestValue<T | undefined>(transaction.objectStore(storeName).get(key));
}

function getAllRecords<T>(database: IDBDatabase, storeName: string): Promise<T[]> {
  const transaction = database.transaction(storeName, "readonly");
  return requestValue<T[]>(transaction.objectStore(storeName).getAll());
}

async function getValidRecords<T>(
  database: IDBDatabase,
  storeName: string,
  isValid: (value: unknown) => value is T,
): Promise<T[]> {
  const transaction = database.transaction(storeName, "readwrite");
  const completion = transactionDone(transaction);
  const records: T[] = [];
  const request = transaction.objectStore(storeName).openCursor();
  await new Promise<void>((resolve, reject) => {
    request.onerror = () => reject(
      request.error ?? new Error(`IndexedDB ${storeName} cursor failed`),
    );
    request.onsuccess = () => {
      const cursor = request.result;
      if (!cursor) {
        resolve();
        return;
      }
      let valid = false;
      try {
        valid = isValid(cursor.value);
      } catch {}
      if (valid) records.push(cursor.value as T);
      else cursor.delete();
      cursor.continue();
    };
  });
  await completion;
  return records;
}

async function putRecord(
  database: IDBDatabase,
  storeName: string,
  value: unknown,
): Promise<void> {
  const transaction = database.transaction(storeName, "readwrite");
  transaction.objectStore(storeName).put(value);
  await transactionDone(transaction);
}

async function deleteRecord(
  database: IDBDatabase,
  storeName: string,
  key: IDBValidKey,
): Promise<void> {
  const transaction = database.transaction(storeName, "readwrite");
  transaction.objectStore(storeName).delete(key);
  await transactionDone(transaction);
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
    transaction.onabort = () => reject(transaction.error ?? new Error("IndexedDB transaction aborted"));
    transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB transaction failed"));
  });
}

function validateDescriptor(descriptor: ContentObjectDescriptor): void {
  validateSha(descriptor.sha256);
  if (!Number.isSafeInteger(descriptor.expectedBytes) || descriptor.expectedBytes <= 0) {
    throw new Error("Expected object size must be a positive integer");
  }
  if (!descriptor.url.startsWith("https://")) throw new Error("Object URL must use HTTPS");
  if (!descriptor.revision) throw new Error("Object revision must be immutable");
  if (!Array.isArray(descriptor.partialOwners) || descriptor.partialOwners.length === 0) {
    throw new Error("A partial object must identify its signed manifest owner");
  }
  for (const owner of descriptor.partialOwners) validatePartialOwner(owner);
}

function isStoredObjectRecord(value: unknown): value is StoredObjectRecord {
  if (!value || typeof value !== "object") return false;
  const record = value as Partial<StoredObjectRecord>;
  if (
    typeof record.sha256 !== "string" || !SHA256.test(record.sha256) ||
    !Number.isSafeInteger(record.expectedBytes) || (record.expectedBytes ?? 0) <= 0 ||
    typeof record.url !== "string" || !record.url.startsWith("https://") ||
    typeof record.revision !== "string" || record.revision.length === 0 ||
    !(record.etag === null || typeof record.etag === "string") ||
    (record.state !== "partial" && record.state !== "verified") ||
    !Number.isSafeInteger(record.contiguousBytes) || (record.contiguousBytes ?? -1) < 0 ||
    (record.contiguousBytes ?? 0) > (record.expectedBytes ?? 0) ||
    !Number.isSafeInteger(record.refCount) || (record.refCount ?? -1) < 0 ||
    !Number.isSafeInteger(record.updatedAt) || (record.updatedAt ?? -1) < 0
  ) return false;
  return record.state !== "verified" || record.contiguousBytes === record.expectedBytes;
}

function isInstalledContentRecord(value: unknown): value is InstalledContentRecord {
  if (!value || typeof value !== "object") return false;
  const record = value as Partial<InstalledContentRecord>;
  return typeof record.id === "string" && record.id.length > 0 && record.id.length <= 256 &&
    (record.kind === "model" || record.kind === "pack") &&
    Array.isArray(record.objectHashes) && record.objectHashes.length > 0 &&
    record.objectHashes.every((hash) => typeof hash === "string" && SHA256.test(hash)) &&
    Number.isSafeInteger(record.installedAt) && (record.installedAt ?? -1) >= 0;
}

function isContentReservationRecord(value: unknown): value is ContentReservationRecord {
  if (!value || typeof value !== "object") return false;
  const record = value as Partial<ContentReservationRecord>;
  return typeof record.id === "string" && record.id.length > 0 && record.id.length <= 256 &&
    typeof record.ownerId === "string" && record.ownerId.length > 0 &&
    record.ownerId.length <= 128 && record.ownerKind === "download" &&
    typeof record.targetId === "string" && record.targetId.length > 0 &&
    record.targetId.length <= 256 &&
    (record.targetKind === "model" || record.targetKind === "pack") &&
    Array.isArray(record.objectHashes) && record.objectHashes.length > 0 &&
    record.objectHashes.every((hash) => typeof hash === "string" && SHA256.test(hash)) &&
    Number.isSafeInteger(record.createdAt) && (record.createdAt ?? -1) >= 0 &&
    Number.isSafeInteger(record.updatedAt) && (record.updatedAt ?? -1) >= 0 &&
    (record.updatedAt ?? 0) >= (record.createdAt ?? 0) &&
    Number.isSafeInteger(record.expiresAt) &&
    (record.expiresAt ?? 0) > (record.updatedAt ?? 0);
}

function validatePartialOwner(value: PartialContentOwner): void {
  if (!value || typeof value !== "object") {
    throw new Error("Partial-object ownership is invalid");
  }
  if (typeof value.id !== "string" || !value.id || value.id.length > 256) {
    throw new Error("Partial-object owner ID is invalid");
  }
  if (value.kind !== "model" && value.kind !== "pack") {
    throw new Error("Partial-object owner kind is invalid");
  }
  if (!Number.isSafeInteger(value.catalogSequence) || value.catalogSequence <= 0) {
    throw new Error("Partial-object catalog sequence is invalid");
  }
}

function normalizePartialOwners(value: unknown): PartialContentOwner[] {
  if (!Array.isArray(value)) return [];
  const owners = new Map<string, PartialContentOwner>();
  for (const candidate of value) {
    try {
      validatePartialOwner(candidate as PartialContentOwner);
    } catch {
      continue;
    }
    const owner = candidate as PartialContentOwner;
    const key = partialOwnerKey(owner);
    const previous = owners.get(key);
    if (!previous || owner.catalogSequence > previous.catalogSequence) {
      owners.set(key, { ...owner });
    }
  }
  return [...owners.values()].sort((left, right) =>
    partialOwnerKey(left).localeCompare(partialOwnerKey(right))
  );
}

function mergePartialOwners(
  left: unknown,
  right: unknown,
): PartialContentOwner[] {
  return normalizePartialOwners([
    ...normalizePartialOwners(left),
    ...normalizePartialOwners(right),
  ]);
}

function samePartialOwners(left: unknown, right: unknown): boolean {
  if (!Array.isArray(left) || !Array.isArray(right)) return false;
  const normalizedLeft = normalizePartialOwners(left);
  const normalizedRight = normalizePartialOwners(right);
  return left.length === normalizedLeft.length &&
    right.length === normalizedRight.length &&
    normalizedLeft.length === normalizedRight.length &&
    normalizedLeft.every((owner, index) =>
      owner.id === normalizedRight[index]?.id &&
      owner.kind === normalizedRight[index]?.kind &&
      owner.catalogSequence === normalizedRight[index]?.catalogSequence
    );
}

function partialOwnerManifest(
  owners: readonly ValidPartialOwner[],
): Map<string, Set<string>> {
  const manifest = new Map<string, Set<string>>();
  for (const owner of owners) {
    if (!owner.id || (owner.kind !== "model" && owner.kind !== "pack")) {
      throw new Error("A valid partial owner is invalid");
    }
    const key = partialOwnerKey(owner);
    const hashes = manifest.get(key) ?? new Set<string>();
    for (const hash of owner.objectHashes) {
      validateSha(hash);
      hashes.add(hash);
    }
    manifest.set(key, hashes);
  }
  return manifest;
}

function partialOwnerKey(owner: Pick<PartialContentOwner, "id" | "kind">): string {
  return `${owner.kind}:${owner.id}`;
}

function matchesPartialOwner(
  owner: PartialContentOwner,
  id: string,
  kind?: PartialContentOwner["kind"],
): boolean {
  return owner.id === id && (kind === undefined || owner.kind === kind);
}

export function validateLoadRecord(record: DeviceLoadRecord): void {
  if (!record || typeof record !== "object") throw new Error("Load record is invalid");
  const keys = Object.keys(record).sort();
  const expectedKeys = [
    "contextTokens",
    "decodeTokensPerSecond",
    "deviceSignature",
    "modelVariantId",
    "prefillTokensPerSecond",
    "recordedAt",
    "result",
    "runtimeIdentitySha256",
  ];
  if (
    keys.length !== expectedKeys.length ||
    keys.some((key, index) => key !== expectedKeys[index])
  ) throw new Error("Load record fields are invalid");
  if (!record.modelVariantId || record.modelVariantId.length > 256) {
    throw new Error("Load record model variant is invalid");
  }
  validateSha(record.runtimeIdentitySha256);
  if (!record.deviceSignature || record.deviceSignature.length > 256) {
    throw new Error("Load record device signature is invalid");
  }
  if (!Number.isSafeInteger(record.contextTokens) || record.contextTokens <= 0) {
    throw new Error("Load record context size is invalid");
  }
  if (!["success", "oom", "device_lost", "failed"].includes(record.result)) {
    throw new Error("Load record result is invalid");
  }
  for (const speed of [record.prefillTokensPerSecond, record.decodeTokensPerSecond]) {
    if (speed !== null && (!Number.isFinite(speed) || speed < 0)) {
      throw new Error("Load record speed is invalid");
    }
  }
  if (!Number.isSafeInteger(record.recordedAt) || record.recordedAt < 0) {
    throw new Error("Load record timestamp is invalid");
  }
}

function loadRecordKey(record: DeviceLoadRecord): IDBValidKey {
  return [
    record.modelVariantId,
    record.runtimeIdentitySha256,
    record.deviceSignature,
    record.contextTokens,
    record.recordedAt,
  ];
}

function validateSha(value: string): void {
  if (!SHA256.test(value)) throw new Error("Invalid SHA-256 digest");
}

function objectFileName(sha256: string): string {
  return `${sha256}.data`;
}

async function entryNames(directory: FileSystemDirectoryHandle): Promise<string[]> {
  const iterable = directory as FileSystemDirectoryHandle & {
    entries(): AsyncIterable<[string, FileSystemHandle]>;
  };
  const names: string[] = [];
  for await (const [name] of iterable.entries()) names.push(name);
  return names;
}

async function removeEntry(
  directory: FileSystemDirectoryHandle,
  name: string,
): Promise<void> {
  try {
    await directory.removeEntry(name, { recursive: true });
  } catch (error) {
    if (!(error instanceof DOMException) || error.name !== "NotFoundError") throw error;
  }
}

async function removeFile(
  directory: FileSystemDirectoryHandle,
  name: string,
): Promise<void> {
  try {
    await directory.removeEntry(name);
  } catch (error) {
    if (!(error instanceof DOMException) || error.name !== "NotFoundError") throw error;
  }
}

async function truncate(handle: FileSystemFileHandle): Promise<void> {
  await truncateTo(handle, 0);
}

async function truncateTo(
  handle: FileSystemFileHandle,
  size: number,
): Promise<void> {
  const writable = await handle.createWritable({ keepExistingData: size > 0 });
  await writable.truncate(size);
  await writable.close();
}
