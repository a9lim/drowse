import { Zip, ZipPassThrough } from "fflate";
import { parseExactJson } from "./json";
import { validateDrowseArchive } from "./drowseArchive";
import { randomUuid } from "../runtime/randomId";
import { migrateLegacyDatabase, drowseStorageRoot } from "../runtime/brandMigration";
import {
  DROWSE_ARCHIVE_FORMAT_VERSION,
  DROWSE_ARCHIVE_MANIFEST_MAX_BYTES,
  DROWSE_ARCHIVE_MAX_ENTRIES,
  DROWSE_ARCHIVE_MAX_FILE_BYTES,
  DROWSE_ARCHIVE_MAX_TOTAL_BYTES,
  type InspectedDrowseArchive,
  type DrowseArchiveManifest,
  type DrowseArchiveSource,
  type DrowseArchiveStage,
  type VerifiedDrowseArchive,
  type VerifyDrowseArchiveOptions,
} from "./types";

const DATABASE_NAME = "drowse-hosted-artifacts";
const DATABASE_VERSION = 1;
const TRANSACTIONS_STORE = "transactions";
const INSTALLS_STORE = "installs";
const ARTIFACT_LOCK = "drowse-artifact-repository-v1";
const OPFS_ARTIFACTS = "artifacts";
const OPFS_STAGES = "stages";
const OPFS_EXPORTS = "exports";
const EXPORT_RETENTION_MS = 24 * 60 * 60 * 1_000;
const NAME = /^[a-z][a-z0-9._-]{0,63}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const RESERVED_MANIFOLD_NAMESPACES = new Set(["jlens", "sae"]);

interface ArtifactTransactionRecord {
  id: string;
  state: "staging" | "committed";
  primary: string;
  template: string | null;
  manifest: DrowseArchiveManifest;
  exactManifestBytes: Uint8Array;
  refCount: number;
  createdAt: number;
  committedAt: number | null;
}

interface ArtifactInstallRecord {
  id: string;
  transactionId: string;
  installedAt: number;
}

interface ArtifactLockLease {
  release(): Promise<void>;
}

interface ArtifactDatabaseEntry {
  key: IDBValidKey;
  value: unknown;
}

export interface InstalledDrowseArchive {
  id: string;
  namespace: string;
  name: string;
  template: string | null;
  installedAt: number;
  producerVersion: string;
  source: DrowseArchiveManifest["source"];
}

export interface InstallDrowseArchiveOptions extends Omit<VerifyDrowseArchiveOptions, "stage"> {
  force?: boolean;
  expectedSource?: {
    repository: string;
    revision: string;
  };
}

export class BrowserDrowseArchiveRepository {
  private database: IDBDatabase | null = null;
  private stages: FileSystemDirectoryHandle | null = null;
  private exports: FileSystemDirectoryHandle | null = null;
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

  async install(
    source: DrowseArchiveSource,
    options: InstallDrowseArchiveOptions = {},
  ): Promise<InstalledDrowseArchive> {
    await this.initialize();
    const stage = new BrowserDrowseArchiveStage(
      this,
      options.force === true,
      options.expectedSource ?? null,
    );
    const verified = await validateDrowseArchive(source, {
      stage,
      signal: options.signal,
      onProgress: options.onProgress,
    });
    return this.summary(verified.manifest.primary);
  }

  async list(): Promise<InstalledDrowseArchive[]> {
    await this.initialize();
    return this.withLock(async () => {
      const installs = await getAll<ArtifactInstallRecord>(this.requireDatabase(), INSTALLS_STORE);
      const results = await Promise.all(installs.map((record) => this.summaryUnlocked(record.id)));
      return results.filter((record): record is InstalledDrowseArchive => record !== null)
        .sort((left, right) => left.id.localeCompare(right.id));
    });
  }

  async remove(primary: string): Promise<boolean> {
    await this.initialize();
    return this.withLock(async () => {
      const database = this.requireDatabase();
      const transaction = database.transaction(
        [INSTALLS_STORE, TRANSACTIONS_STORE],
        "readwrite",
      );
      const installs = transaction.objectStore(INSTALLS_STORE);
      const transactions = transaction.objectStore(TRANSACTIONS_STORE);
      const install = await requestResult<ArtifactInstallRecord | undefined>(installs.get(primary));
      if (!install) {
        await transactionDone(transaction);
        return false;
      }
      const stored = await requestResult<ArtifactTransactionRecord | undefined>(
        transactions.get(install.transactionId),
      );
      installs.delete(primary);
      if (stored) {
        stored.refCount = Math.max(0, stored.refCount - 1);
        transactions.put(stored);
      }
      await transactionDone(transaction);
      if (stored?.refCount === 0) {
        await this.removeTransaction(stored.id).catch(() => undefined);
      }
      return true;
    });
  }

  async file(primary: string, path: string): Promise<File | null> {
    await this.initialize();
    return this.withLock(async () => {
      const resolved = await this.resolveCommitted(primary);
      if (!resolved) return null;
      if (path !== "pack.json" && !(path in resolved.transaction.manifest.files)) {
        return null;
      }
      return this.stageFile(resolved.transaction.id, path, false);
    });
  }

  async export(primary: string): Promise<File> {
    await this.initialize();
    return (await this.createExport(primary)).file;
  }

  async withExport<T>(
    primary: string,
    consumer: (file: File) => Promise<T>,
    signal?: AbortSignal,
  ): Promise<T> {
    await this.initialize();
    const leased = await this.createExport(primary, signal);
    const exports = this.requireExports();
    try {
      signal?.throwIfAborted();
      return await consumer(leased.file);
    } finally {
      await exports.removeEntry(leased.id).catch(ignoreMissing);
    }
  }

  async clear(): Promise<void> {
    await this.initialize();
    await this.withLock(async () => {
      const database = this.requireDatabase();
      const transaction = database.transaction(
        [INSTALLS_STORE, TRANSACTIONS_STORE],
        "readwrite",
      );
      transaction.objectStore(INSTALLS_STORE).clear();
      transaction.objectStore(TRANSACTIONS_STORE).clear();
      await transactionDone(transaction);
      await clearDirectory(this.requireStages());
      await clearDirectory(this.requireExports());
    });
  }

  async close(): Promise<void> {
    this.database?.close();
    this.database = null;
    this.stages = null;
    this.exports = null;
    this.initialization = null;
  }

  createStage(force: boolean): DrowseArchiveStage {
    return new BrowserDrowseArchiveStage(this, force);
  }

  async acquireStageLock(): Promise<ArtifactLockLease> {
    if (!navigator.locks) throw new Error("Web Locks are unavailable");
    let releaseLock!: () => void;
    let resolveGranted!: () => void;
    let rejectGranted!: (error: unknown) => void;
    let grantedLock = false;
    const released = new Promise<void>((resolve) => {
      releaseLock = resolve;
    });
    const granted = new Promise<void>((resolve, reject) => {
      resolveGranted = resolve;
      rejectGranted = reject;
    });
    const request = navigator.locks.request(ARTIFACT_LOCK, { mode: "exclusive" }, async () => {
      grantedLock = true;
      resolveGranted();
      await released;
    });
    void request.catch((error) => {
      if (!grantedLock) rejectGranted(error);
    });
    await granted;
    let didRelease = false;
    return {
      release: async () => {
        if (!didRelease) {
          didRelease = true;
          releaseLock();
        }
        await request;
      },
    };
  }

  async beginStageLocked(
    id: string,
    pack: InspectedDrowseArchive,
    force: boolean,
  ): Promise<void> {
    const database = this.requireDatabase();
    const current = await getRecord<ArtifactInstallRecord>(database, INSTALLS_STORE, pack.manifest.primary);
    if (current && !force) {
      throw new Error(`Drowse manifold pack ${pack.manifest.primary} is already installed`);
    }
    await this.requireStages().getDirectoryHandle(id, { create: true });
    const record: ArtifactTransactionRecord = {
      id,
      state: "staging",
      primary: pack.manifest.primary,
      template: pack.manifest.template,
      manifest: structuredClone(pack.manifest) as DrowseArchiveManifest,
      exactManifestBytes: pack.exactManifestBytes.slice(),
      refCount: 0,
      createdAt: Date.now(),
      committedAt: null,
    };
    await putRecord(database, TRANSACTIONS_STORE, record);
    await this.writeWholeStageFile(id, "pack.json", pack.exactManifestBytes);
  }

  async openStageFile(id: string, path: string): Promise<FileSystemWritableFileStream> {
    const handle = await this.stageFileHandle(id, path, true);
    if (!handle) throw new Error(`Could not create staged Drowse file ${path}`);
    return handle.createWritable();
  }

  async commitStageLocked(id: string, verified: VerifiedDrowseArchive, force: boolean): Promise<void> {
    const database = this.requireDatabase();
    for (const [path, expected] of Object.entries(verified.manifest.files)) {
      const file = await this.stageFile(id, path, false);
      if (!file || file.size !== expected.size) {
        throw new Error(`Staged Drowse file ${path} is missing or has the wrong size`);
      }
    }
    const transaction = database.transaction(
      [INSTALLS_STORE, TRANSACTIONS_STORE],
      "readwrite",
    );
    const installs = transaction.objectStore(INSTALLS_STORE);
    const transactions = transaction.objectStore(TRANSACTIONS_STORE);
    const staged = await requestResult<ArtifactTransactionRecord | undefined>(transactions.get(id));
    if (!staged || staged.state !== "staging") {
      transaction.abort();
      throw new Error("Drowse artifact staging transaction is no longer active");
    }
    const previous = await requestResult<ArtifactInstallRecord | undefined>(
      installs.get(verified.manifest.primary),
    );
    if (previous && !force) {
      transaction.abort();
      throw new Error(`Drowse manifold pack ${verified.manifest.primary} is already installed`);
    }
    if (staged.template !== null) {
      const templateFile = `${staged.template}/template.json`;
      const digest = staged.manifest.files[templateFile]?.sha256;
      if (!digest) {
        transaction.abort();
        throw new Error(`Drowse manifold pack ${verified.manifest.primary} has no template digest`);
      }
      const installed = await requestResult<ArtifactInstallRecord[]>(installs.getAll());
      for (const otherInstall of installed) {
        if (otherInstall.id === verified.manifest.primary) continue;
        const other = await requestResult<ArtifactTransactionRecord | undefined>(
          transactions.get(otherInstall.transactionId),
        );
        if (
          other?.state === "committed" && other.template === staged.template &&
          other.manifest.files[templateFile]?.sha256 !== digest
        ) {
          transaction.abort();
          throw new Error(
            `Template ${staged.template} conflicts with an installed Drowse manifold pack`,
          );
        }
      }
    }
    staged.state = "committed";
    staged.refCount = 1;
    staged.committedAt = previous === undefined
      ? Date.now()
      : Math.max(Date.now(), previous.installedAt + 1);
    transactions.put(staged);
    installs.put({
      id: verified.manifest.primary,
      transactionId: id,
      installedAt: staged.committedAt,
    } satisfies ArtifactInstallRecord);
    let priorTransaction: ArtifactTransactionRecord | undefined;
    if (previous && previous.transactionId !== id) {
      priorTransaction = await requestResult<ArtifactTransactionRecord | undefined>(
        transactions.get(previous.transactionId),
      );
      if (priorTransaction) {
        priorTransaction.refCount = Math.max(0, priorTransaction.refCount - 1);
        transactions.put(priorTransaction);
      }
    }
    await transactionDone(transaction);
    if (priorTransaction?.refCount === 0) {
      await this.removeTransaction(priorTransaction.id).catch(() => undefined);
    }
  }

  async rollbackStageLocked(id: string): Promise<void> {
    const record = await getRecord<ArtifactTransactionRecord>(
      this.requireDatabase(),
      TRANSACTIONS_STORE,
      id,
    );
    if (record?.state === "committed") return;
    await deleteRecord(this.requireDatabase(), TRANSACTIONS_STORE, id);
    await this.requireStages().removeEntry(id, { recursive: true }).catch(ignoreMissing);
  }

  private async initializeOnce(): Promise<void> {
    if (!navigator.storage?.getDirectory) {
      throw new Error("Origin-private file storage is unavailable");
    }
    const database = await openDatabase();
    database.onversionchange = () => database.close();
    try {
      const root = await navigator.storage.getDirectory();
      const drowse = await drowseStorageRoot(root);
      const artifacts = await drowse.getDirectoryHandle(OPFS_ARTIFACTS, { create: true });
      const stages = await artifacts.getDirectoryHandle(OPFS_STAGES, { create: true });
      const exports = await artifacts.getDirectoryHandle(OPFS_EXPORTS, { create: true });
      this.database = database;
      this.stages = stages;
      this.exports = exports;
      database.onversionchange = () => {
        database.close();
        if (this.database === database) {
          this.database = null;
          this.stages = null;
          this.exports = null;
          this.initialization = null;
        }
      };
      await this.recover();
    } catch (error) {
      database.close();
      if (this.database === database) {
        this.database = null;
        this.stages = null;
        this.exports = null;
      }
      throw error;
    }
  }

  private async recover(): Promise<void> {
    await this.withLock(async () => {
      const database = this.requireDatabase();
      const installEntries = await getAllEntries(database, INSTALLS_STORE);
      const transactionEntries = await getAllEntries(database, TRANSACTIONS_STORE);
      const installs: ArtifactInstallRecord[] = [];
      const transactions: ArtifactTransactionRecord[] = [];
      const invalidInstallKeys: IDBValidKey[] = [];
      const invalidTransactionKeys: IDBValidKey[] = [];
      for (const entry of installEntries) {
        const record = validateInstallRecord(entry.value, entry.key);
        if (record === null) invalidInstallKeys.push(entry.key);
        else installs.push(record);
      }
      for (const entry of transactionEntries) {
        const record = validateTransactionRecord(entry.value, entry.key);
        if (record === null) invalidTransactionKeys.push(entry.key);
        else transactions.push(record);
      }
      const transactionById = new Map(transactions.map((record) => [record.id, record]));
      const invalidTransactions = new Set<string>();
      for (const transaction of transactions) {
        if (
          transaction.state === "committed" &&
          !(await this.committedStageIsIntact(transaction))
        ) invalidTransactions.add(transaction.id);
      }
      const references = new Map<string, number>();
      const invalidInstallIds: string[] = [];
      const templateDigests = new Map<string, string>();
      for (const install of installs.sort((left, right) =>
        left.installedAt - right.installedAt || left.id.localeCompare(right.id)
      )) {
        const transaction = transactionById.get(install.transactionId);
        if (
          !transaction || transaction.state !== "committed" ||
          transaction.primary !== install.id ||
          transaction.committedAt !== install.installedAt ||
          invalidTransactions.has(transaction.id)
        ) {
          invalidInstallIds.push(install.id);
          continue;
        }
        if (transaction.template !== null) {
          const templatePath = `${transaction.template}/template.json`;
          const digest = transaction.manifest.files[templatePath]?.sha256;
          const accepted = templateDigests.get(transaction.template);
          if (!digest || accepted !== undefined && accepted !== digest) {
            invalidInstallIds.push(install.id);
            continue;
          }
          templateDigests.set(transaction.template, digest);
        }
        references.set(install.transactionId, (references.get(install.transactionId) ?? 0) + 1);
      }
      const removeIds: string[] = [];
      const updateRecords: ArtifactTransactionRecord[] = [];
      for (const transaction of transactions) {
        const refCount = references.get(transaction.id) ?? 0;
        if (
          transaction.state !== "committed" || refCount === 0 ||
          invalidTransactions.has(transaction.id)
        ) {
          removeIds.push(transaction.id);
        } else if (transaction.refCount !== refCount) {
          updateRecords.push({ ...transaction, refCount });
        }
      }
      const write = database.transaction(
        [INSTALLS_STORE, TRANSACTIONS_STORE],
        "readwrite",
      );
      for (const key of invalidInstallKeys) write.objectStore(INSTALLS_STORE).delete(key);
      for (const id of invalidInstallIds) write.objectStore(INSTALLS_STORE).delete(id);
      for (const key of invalidTransactionKeys) write.objectStore(TRANSACTIONS_STORE).delete(key);
      for (const id of removeIds) write.objectStore(TRANSACTIONS_STORE).delete(id);
      for (const record of updateRecords) write.objectStore(TRANSACTIONS_STORE).put(record);
      await transactionDone(write);
      const keep = new Set([...references.keys()].filter((id) => !removeIds.includes(id)));
      for await (const [name, handle] of this.requireStages().entries()) {
        if (handle.kind === "directory" && keep.has(name)) continue;
        await this.requireStages().removeEntry(name, { recursive: true });
      }
      await removeExpiredExports(
        this.requireExports(),
        Date.now() - EXPORT_RETENTION_MS,
      );
    });
  }

  private async committedStageIsIntact(
    transaction: ArtifactTransactionRecord,
  ): Promise<boolean> {
    try {
      const manifest = await this.stageFile(transaction.id, "pack.json", false);
      if (!manifest || manifest.size !== transaction.exactManifestBytes.byteLength) return false;
      const exactBytes = new Uint8Array(await manifest.arrayBuffer());
      if (!sameBytes(exactBytes, transaction.exactManifestBytes)) return false;
      for (const [path, expected] of Object.entries(transaction.manifest.files)) {
        const file = await this.stageFile(transaction.id, path, false);
        if (!file || file.size !== expected.size) return false;
      }
      return true;
    } catch {
      return false;
    }
  }

  private async summary(primary: string): Promise<InstalledDrowseArchive> {
    return this.withLock(async () => {
      const summary = await this.summaryUnlocked(primary);
      if (!summary) throw new Error(`Installed Drowse manifold pack ${primary} disappeared`);
      return summary;
    });
  }

  private createExport(
    primary: string,
    signal?: AbortSignal,
  ): Promise<{ id: string; file: File }> {
    return this.withLock(async () => {
      signal?.throwIfAborted();
      const resolved = await this.resolveCommitted(primary);
      if (!resolved) throw new Error(`No installed Drowse manifold pack named ${primary}`);
      const exportId = `${randomUuid()}.drowse`;
      const exports = this.requireExports();
      const handle = await exports.getFileHandle(exportId, { create: true });
      const writable = await handle.createWritable();
      let writes = Promise.resolve();
      let finalWritten: (() => void) | null = null;
      let finalRejected: ((error: unknown) => void) | null = null;
      const final = new Promise<void>((resolve, reject) => {
        finalWritten = resolve;
        finalRejected = reject;
      });
      const zip = new Zip((error, chunk, isFinal) => {
        if (error) {
          finalRejected?.(error);
          return;
        }
        writes = writes.then(() => writable.write(ownedBytes(chunk)));
        if (isFinal) writes.then(() => finalWritten?.(), finalRejected ?? undefined);
      });
      try {
        const paths = ["pack.json", ...Object.keys(resolved.transaction.manifest.files).sort()];
        for (const path of paths) {
          signal?.throwIfAborted();
          const file = await this.stageFile(resolved.transaction.id, path, false);
          if (!file) throw new Error(`Installed Drowse pack is missing ${path}`);
          const entry = new ZipPassThrough(path);
          entry.os = 3;
          entry.attrs = 0o100644 << 16;
          entry.mtime = new Date("1980-01-02T12:00:00Z");
          zip.add(entry);
          const reader = file.stream().getReader();
          try {
            while (true) {
              signal?.throwIfAborted();
              const { done, value } = await reader.read();
              if (done) break;
              signal?.throwIfAborted();
              entry.push(value, false);
              await writes;
            }
          } finally {
            reader.releaseLock();
          }
          entry.push(new Uint8Array(), true);
          await writes;
        }
        signal?.throwIfAborted();
        zip.end();
        await final;
        signal?.throwIfAborted();
        await writable.close();
        return { id: exportId, file: await handle.getFile() };
      } catch (error) {
        zip.terminate();
        await writable.abort(error).catch(() => undefined);
        await exports.removeEntry(exportId).catch(ignoreMissing);
        throw error;
      }
    });
  }

  private async summaryUnlocked(primary: string): Promise<InstalledDrowseArchive | null> {
    const resolved = await this.resolveCommitted(primary);
    if (!resolved) return null;
    const [, namespace, name] = resolved.transaction.primary.split("/");
    return {
      id: resolved.transaction.primary,
      namespace,
      name,
      template: resolved.transaction.template,
      installedAt: resolved.install.installedAt,
      producerVersion: resolved.transaction.manifest.producer.version,
      source: structuredClone(resolved.transaction.manifest.source),
    };
  }

  private async resolveCommitted(primary: string): Promise<{
    install: ArtifactInstallRecord;
    transaction: ArtifactTransactionRecord;
  } | null> {
    const database = this.requireDatabase();
    const install = await getRecord<ArtifactInstallRecord>(database, INSTALLS_STORE, primary);
    if (!install) return null;
    const transaction = await getRecord<ArtifactTransactionRecord>(
      database,
      TRANSACTIONS_STORE,
      install.transactionId,
    );
    if (!transaction || transaction.state !== "committed" || transaction.primary !== primary) {
      return null;
    }
    return { install, transaction };
  }

  private async writeWholeStageFile(id: string, path: string, bytes: Uint8Array): Promise<void> {
    const handle = await this.stageFileHandle(id, path, true);
    if (!handle) throw new Error(`Could not create staged Drowse file ${path}`);
    const writable = await handle.createWritable();
    await writable.write(ownedBytes(bytes));
    await writable.close();
  }

  private async stageFile(id: string, path: string, create: boolean): Promise<File | null> {
    const handle = await this.stageFileHandle(id, path, create);
    if (!handle) return null;
    try {
      return await handle.getFile();
    } catch (error) {
      if (isMissing(error)) return null;
      throw error;
    }
  }

  private async stageFileHandle(
    id: string,
    path: string,
    create: boolean,
  ): Promise<FileSystemFileHandle | null> {
    try {
      let directory = await this.requireStages().getDirectoryHandle(id, { create });
      const parts = path.split("/");
      const filename = parts.pop();
      if (!filename) throw new Error("Drowse artifact path has no filename");
      for (const part of parts) {
        directory = await directory.getDirectoryHandle(part, { create });
      }
      return await directory.getFileHandle(filename, { create });
    } catch (error) {
      if (!create && isMissing(error)) return null;
      throw error;
    }
  }

  private async removeTransaction(id: string): Promise<void> {
    await this.requireStages().removeEntry(id, { recursive: true }).catch(ignoreMissing);
    await deleteRecord(this.requireDatabase(), TRANSACTIONS_STORE, id);
  }

  private withLock<T>(operation: () => Promise<T>): Promise<T> {
    if (!navigator.locks) throw new Error("Web Locks are unavailable");
    return navigator.locks.request(ARTIFACT_LOCK, { mode: "exclusive" }, operation);
  }

  private requireDatabase(): IDBDatabase {
    if (!this.database) throw new Error("Drowse artifact database is not initialized");
    return this.database;
  }

  private requireStages(): FileSystemDirectoryHandle {
    if (!this.stages) throw new Error("Drowse artifact storage is not initialized");
    return this.stages;
  }

  private requireExports(): FileSystemDirectoryHandle {
    if (!this.exports) throw new Error("Drowse artifact export storage is not initialized");
    return this.exports;
  }
}

class BrowserDrowseArchiveStage implements DrowseArchiveStage {
  private readonly id = randomUuid();
  private readonly repository: BrowserDrowseArchiveRepository;
  private readonly force: boolean;
  private readonly expectedSource: NonNullable<InstallDrowseArchiveOptions["expectedSource"]> | null;
  private lease: ArtifactLockLease | null = null;
  private currentPath: string | null = null;
  private writable: FileSystemWritableFileStream | null = null;
  private began = false;
  private committed = false;

  constructor(
    repository: BrowserDrowseArchiveRepository,
    force: boolean,
    expectedSource: NonNullable<InstallDrowseArchiveOptions["expectedSource"]> | null = null,
  ) {
    this.repository = repository;
    this.force = force;
    this.expectedSource = expectedSource;
  }

  async begin(pack: InspectedDrowseArchive): Promise<void> {
    if (this.began || this.lease !== null) throw new Error("Drowse artifact stage already began");
    if (
      this.expectedSource !== null &&
      (
        pack.manifest.source.repository !== this.expectedSource.repository ||
        pack.manifest.source.revision !== this.expectedSource.revision
      )
    ) {
      throw new Error(
        `Drowse pack provenance does not match ${this.expectedSource.repository}@${this.expectedSource.revision}`,
      );
    }
    await this.repository.initialize();
    this.lease = await this.repository.acquireStageLock();
    try {
      await this.repository.beginStageLocked(this.id, pack, this.force);
      this.began = true;
    } catch (error) {
      await this.repository.rollbackStageLocked(this.id).catch(() => undefined);
      await this.releaseLease().catch(() => undefined);
      throw error;
    }
  }

  async write(path: string, chunk: Uint8Array): Promise<void> {
    this.requireActive();
    if (this.currentPath !== path) {
      if (this.writable) throw new Error(`Drowse artifact stage did not finish ${this.currentPath}`);
      this.currentPath = path;
      this.writable = await this.repository.openStageFile(this.id, path);
    }
    const writable = this.writable;
    if (!writable) throw new Error(`Drowse artifact stage has no open file ${path}`);
    await writable.write(ownedBytes(chunk));
  }

  async finish(path: string): Promise<void> {
    this.requireActive();
    if (this.currentPath !== path || !this.writable) {
      throw new Error(`Drowse artifact stage has no open file ${path}`);
    }
    await this.writable.close();
    this.writable = null;
    this.currentPath = null;
  }

  async commit(pack: VerifiedDrowseArchive): Promise<void> {
    this.requireActive();
    if (this.writable) throw new Error(`Drowse artifact stage did not finish ${this.currentPath}`);
    await this.repository.commitStageLocked(this.id, pack, this.force);
    this.committed = true;
    await this.releaseLease();
  }

  async rollback(): Promise<void> {
    if (this.committed) return;
    if (this.writable) {
      await this.writable.abort().catch(() => undefined);
      this.writable = null;
      this.currentPath = null;
    }
    if (!this.began || this.lease === null) return;
    try {
      await this.repository.rollbackStageLocked(this.id);
    } finally {
      this.began = false;
      await this.releaseLease();
    }
  }

  private requireActive(): void {
    if (!this.began || this.lease === null || this.committed) {
      throw new Error("Drowse artifact stage is not active");
    }
  }

  private async releaseLease(): Promise<void> {
    const lease = this.lease;
    this.lease = null;
    if (lease !== null) await lease.release();
  }
}

function validateTransactionRecord(
  value: unknown,
  key: IDBValidKey,
): ArtifactTransactionRecord | null {
  if (!isRecord(value) || !exactKeys(value, [
    "committedAt", "createdAt", "exactManifestBytes", "id", "manifest",
    "primary", "refCount", "state", "template",
  ])) return null;
  if (
    typeof key !== "string" || typeof value.id !== "string" || key !== value.id ||
    !UUID_V4.test(value.id) ||
    (value.state !== "staging" && value.state !== "committed") ||
    !validIdentity(value.primary, "manifolds") ||
    value.template !== null && !validIdentity(value.template, "templates") ||
    !Number.isSafeInteger(value.refCount) || (value.refCount as number) < 0 ||
    !validTimestamp(value.createdAt) ||
    !(value.exactManifestBytes instanceof Uint8Array) ||
    value.exactManifestBytes.byteLength === 0 ||
    value.exactManifestBytes.byteLength > DROWSE_ARCHIVE_MANIFEST_MAX_BYTES
  ) return null;
  if (
    value.state === "staging" && (value.refCount !== 0 || value.committedAt !== null) ||
    value.state === "committed" && !validTimestamp(value.committedAt)
  ) return null;
  const manifest = validateStoredManifest(value.manifest);
  if (
    manifest === null || manifest.primary !== value.primary || manifest.template !== value.template ||
    manifestPayloadBytes(manifest) + value.exactManifestBytes.byteLength > DROWSE_ARCHIVE_MAX_TOTAL_BYTES
  ) return null;
  let exactManifest: DrowseArchiveManifest | null;
  try {
    exactManifest = validateStoredManifest(parseExactJson(value.exactManifestBytes, "stored pack.json"));
  } catch {
    return null;
  }
  if (exactManifest === null || !sameJson(manifest, exactManifest)) return null;
  return {
    id: value.id,
    state: value.state,
    primary: value.primary as string,
    template: value.template as string | null,
    manifest,
    exactManifestBytes: value.exactManifestBytes.slice(),
    refCount: value.refCount as number,
    createdAt: value.createdAt as number,
    committedAt: value.committedAt as number | null,
  };
}

function validateInstallRecord(value: unknown, key: IDBValidKey): ArtifactInstallRecord | null {
  if (!isRecord(value) || !exactKeys(value, ["id", "installedAt", "transactionId"])) return null;
  if (
    typeof key !== "string" || key !== value.id ||
    !validIdentity(value.id, "manifolds") ||
    typeof value.transactionId !== "string" || !UUID_V4.test(value.transactionId) ||
    !validTimestamp(value.installedAt)
  ) return null;
  return {
    id: value.id as string,
    transactionId: value.transactionId,
    installedAt: value.installedAt as number,
  };
}

function validateStoredManifest(value: unknown): DrowseArchiveManifest | null {
  if (!isRecord(value) || !exactKeys(value, [
    "files", "format_version", "kind", "primary", "producer", "source", "template",
  ])) return null;
  if (
    value.format_version !== DROWSE_ARCHIVE_FORMAT_VERSION ||
    value.kind !== "drowse-manifold" ||
    !validIdentity(value.primary, "manifolds") ||
    value.template !== null && !validIdentity(value.template, "templates") ||
    !isRecord(value.producer) || !exactKeys(value.producer, ["name", "version"]) ||
    value.producer.name !== "drowse" || !shortString(value.producer.version, 1_024) ||
    !validStoredSource(value.source) || !isRecord(value.files)
  ) return null;
  const entries = Object.entries(value.files);
  if (entries.length === 0 || entries.length + 1 > DROWSE_ARCHIVE_MAX_ENTRIES) return null;
  const files: DrowseArchiveManifest["files"] = Object.create(null) as DrowseArchiveManifest["files"];
  const foldedPaths = new Set<string>();
  let totalBytes = 0;
  for (const [path, record] of entries) {
    if (!validManifestPath(path) || foldedPaths.has(path.toLowerCase())) return null;
    foldedPaths.add(path.toLowerCase());
    if (
      !path.startsWith(`${value.primary}/`) &&
      (value.template === null || path !== `${value.template}/template.json`)
    ) return null;
    if (
      !isRecord(record) || !exactKeys(record, ["sha256", "size"]) ||
      typeof record.sha256 !== "string" || !SHA256.test(record.sha256) ||
      !Number.isSafeInteger(record.size) || (record.size as number) < 0 ||
      (record.size as number) > DROWSE_ARCHIVE_MAX_FILE_BYTES
    ) return null;
    totalBytes += record.size as number;
    if (!Number.isSafeInteger(totalBytes) || totalBytes > DROWSE_ARCHIVE_MAX_TOTAL_BYTES) return null;
    files[path] = { sha256: record.sha256, size: record.size as number };
  }
  if (files[`${value.primary}/manifold.json`] === undefined) return null;
  if (value.template !== null && files[`${value.template}/template.json`] === undefined) return null;
  return {
    format_version: DROWSE_ARCHIVE_FORMAT_VERSION,
    kind: "drowse-manifold",
    producer: { name: "drowse", version: value.producer.version as string },
    primary: value.primary as string,
    template: value.template as string | null,
    source: {
      uri: value.source.uri as string,
      repository: value.source.repository as string | null,
      revision: value.source.revision as string | null,
    },
    files,
  };
}

function validStoredSource(value: unknown): value is DrowseArchiveManifest["source"] {
  if (
    !isRecord(value) || !exactKeys(value, ["repository", "revision", "uri"]) ||
    !shortString(value.uri, 4_096) ||
    value.repository !== null && !shortString(value.repository, 1_024) ||
    value.revision !== null && !shortString(value.revision, 1_024)
  ) return false;
  if ((value.uri as string).startsWith("hf://")) {
    return typeof value.repository === "string" && typeof value.revision === "string" &&
      /^[0-9a-f]{40}$/.test(value.revision) &&
      value.uri === `hf://${value.repository}@${value.revision}`;
  }
  return value.repository === null && value.revision === null;
}

function validIdentity(value: unknown, root: "manifolds" | "templates"): value is string {
  if (typeof value !== "string") return false;
  const parts = value.split("/");
  return parts.length === 3 && parts[0] === root && NAME.test(parts[1]) && NAME.test(parts[2]) &&
    (root !== "manifolds" || !RESERVED_MANIFOLD_NAMESPACES.has(parts[1]));
}

function validManifestPath(path: string): boolean {
  return path.length > 0 && /^[\x20-\x7e]+$/.test(path) && !path.includes("\\") &&
    !path.startsWith("/") && !/^[A-Za-z]:/.test(path) &&
    path.split("/").every((part) => part !== "" && part !== "." && part !== "..");
}

function validTimestamp(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function shortString(value: unknown, maximum: number): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= maximum;
}

function manifestPayloadBytes(manifest: DrowseArchiveManifest): number {
  return Object.values(manifest.files).reduce((total, record) => total + record.size, 0);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function exactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const keys = Object.keys(value).sort();
  return keys.length === expected.length && expected.every((key, index) => keys[index] === key);
}

function sameJson(left: unknown, right: unknown): boolean {
  if (left === right) return true;
  if (Array.isArray(left) && Array.isArray(right)) {
    return left.length === right.length && left.every((value, index) => sameJson(value, right[index]));
  }
  if (isRecord(left) && isRecord(right)) {
    const leftKeys = Object.keys(left).sort();
    const rightKeys = Object.keys(right).sort();
    return leftKeys.length === rightKeys.length &&
      leftKeys.every((key, index) => key === rightKeys[index] && sameJson(left[key], right[key]));
  }
  return false;
}

async function openDatabase(): Promise<IDBDatabase> {
  const database = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    let settled = false;
    const rejectOnce = (error: Error) => {
      if (settled) return;
      settled = true;
      reject(error);
    };
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(TRANSACTIONS_STORE)) {
        database.createObjectStore(TRANSACTIONS_STORE, { keyPath: "id" });
      }
      if (!database.objectStoreNames.contains(INSTALLS_STORE)) {
        database.createObjectStore(INSTALLS_STORE, { keyPath: "id" });
      }
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
      request.error ?? new Error("Could not open Drowse artifact database"),
    );
    request.onblocked = () => rejectOnce(
      new Error("Drowse artifact database upgrade is blocked by another tab"),
    );
  });
  await migrateLegacyDatabase(database, [TRANSACTIONS_STORE, INSTALLS_STORE]);
  return database;
}

function getRecord<T>(database: IDBDatabase, storeName: string, key: IDBValidKey): Promise<T | undefined> {
  return requestResult<T | undefined>(database.transaction(storeName).objectStore(storeName).get(key));
}

function getAll<T>(database: IDBDatabase, storeName: string): Promise<T[]> {
  return requestResult<T[]>(database.transaction(storeName).objectStore(storeName).getAll());
}

async function getAllEntries(
  database: IDBDatabase,
  storeName: string,
): Promise<ArtifactDatabaseEntry[]> {
  const transaction = database.transaction(storeName, "readonly");
  const store = transaction.objectStore(storeName);
  const [keys, values] = await Promise.all([
    requestResult<IDBValidKey[]>(store.getAllKeys()),
    requestResult<unknown[]>(store.getAll()),
  ]);
  if (keys.length !== values.length) {
    throw new Error(`Drowse artifact database ${storeName} keys and values are inconsistent`);
  }
  return keys.map((key, index) => ({ key, value: values[index] }));
}

async function putRecord(database: IDBDatabase, storeName: string, value: unknown): Promise<void> {
  const transaction = database.transaction(storeName, "readwrite");
  transaction.objectStore(storeName).put(value);
  await transactionDone(transaction);
}

async function deleteRecord(database: IDBDatabase, storeName: string, key: IDBValidKey): Promise<void> {
  const transaction = database.transaction(storeName, "readwrite");
  transaction.objectStore(storeName).delete(key);
  await transactionDone(transaction);
}

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Drowse artifact database request failed"));
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error("Drowse artifact database transaction failed"));
    transaction.onabort = () => reject(transaction.error ?? new Error("Drowse artifact database transaction was aborted"));
  });
}

async function clearDirectory(directory: FileSystemDirectoryHandle): Promise<void> {
  for await (const [name] of directory.entries()) {
    await directory.removeEntry(name, { recursive: true });
  }
}

async function removeExpiredExports(
  directory: FileSystemDirectoryHandle,
  cutoff: number,
): Promise<void> {
  for await (const [name, handle] of directory.entries()) {
    if (handle.kind !== "file" || !UUID_V4.test(name.replace(/\.drowse$/, ""))) {
      await directory.removeEntry(name, { recursive: handle.kind === "directory" });
      continue;
    }
    const file = await handle.getFile();
    if (file.lastModified < cutoff) await directory.removeEntry(name);
  }
}

function isMissing(error: unknown): boolean {
  return error instanceof DOMException && error.name === "NotFoundError";
}

function ignoreMissing(error: unknown): void {
  if (!isMissing(error)) throw error;
}

function ownedBytes(bytes: Uint8Array): Uint8Array<ArrayBuffer> {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy;
}

function sameBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  for (let index = 0; index < left.byteLength; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}
