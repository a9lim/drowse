import { migrateLegacyDatabase } from "../runtime/brandMigration";

const DATABASE_NAME = "drowse-hosted-authoring";
const DATABASE_VERSION = 1;
const STORE_NAME = "templates";
const ARTIFACT_LOCK = "drowse-artifact-repository-v1";

export interface StoredBrowserTemplate {
  id: string;
  namespace: string;
  name: string;
  payload: Record<string, unknown>;
  installedAt: number;
}

export interface BrowserTemplateRepositoryPort {
  initialize(): Promise<void>;
  list(): Promise<StoredBrowserTemplate[]>;
  get(id: string): Promise<StoredBrowserTemplate | null>;
  put(
    namespace: string,
    name: string,
    payload: Record<string, unknown>,
    force: boolean,
  ): Promise<StoredBrowserTemplate>;
  remove(id: string): Promise<boolean>;
  clear(): Promise<void>;
  close(): Promise<void>;
}

export class BrowserTemplateRepository implements BrowserTemplateRepositoryPort {
  private database: IDBDatabase | null = null;
  private initialization: Promise<void> | null = null;

  initialize(): Promise<void> {
    if (this.initialization === null) {
      const initialization = openDatabase().then((database) => {
        this.database = database;
        database.onversionchange = () => {
          database.close();
          if (this.database === database) {
            this.database = null;
            this.initialization = null;
          }
        };
      });
      this.initialization = initialization;
      void initialization.catch(() => {
        if (this.initialization === initialization) this.initialization = null;
      });
    }
    return this.initialization;
  }

  async list(): Promise<StoredBrowserTemplate[]> {
    await this.initialize();
    return this.withLock(async () => {
      const transaction = this.requireDatabase().transaction(STORE_NAME, "readwrite");
      const store = transaction.objectStore(STORE_NAME);
      const [keys, values] = await Promise.all([
        requestResult<IDBValidKey[]>(store.getAllKeys()),
        requestResult<unknown[]>(store.getAll()),
      ]);
      const records: StoredBrowserTemplate[] = [];
      values.forEach((value, index) => {
        const record = validRecord(value, keys[index]);
        if (record) records.push(record);
        else store.delete(keys[index]);
      });
      await transactionDone(transaction);
      return records.map(cloneRecord).sort((left, right) => left.id.localeCompare(right.id));
    });
  }

  async get(id: string): Promise<StoredBrowserTemplate | null> {
    await this.initialize();
    return this.withLock(async () => {
      const transaction = this.requireDatabase().transaction(STORE_NAME, "readwrite");
      const store = transaction.objectStore(STORE_NAME);
      const value = await requestResult<unknown | undefined>(store.get(id));
      const record = value === undefined ? null : validRecord(value, id);
      if (value !== undefined && record === null) store.delete(id);
      await transactionDone(transaction);
      return record ? cloneRecord(record) : null;
    });
  }

  async put(
    namespace: string,
    name: string,
    payload: Record<string, unknown>,
    force: boolean,
  ): Promise<StoredBrowserTemplate> {
    await this.initialize();
    const id = `${namespace}/${name}`;
    return this.withLock(async () => {
      const database = this.requireDatabase();
      const transaction = database.transaction(STORE_NAME, "readwrite");
      const store = transaction.objectStore(STORE_NAME);
      const current = await requestResult<StoredBrowserTemplate | undefined>(store.get(id));
      if (current && !force) {
        transaction.abort();
        throw new Error(`Template ${id} already exists`);
      }
      const record: StoredBrowserTemplate = {
        id,
        namespace,
        name,
        payload: structuredClone(payload),
        installedAt: Date.now(),
      };
      store.put(record);
      await transactionDone(transaction);
      return cloneRecord(record);
    });
  }

  async remove(id: string): Promise<boolean> {
    await this.initialize();
    return this.withLock(async () => {
      const database = this.requireDatabase();
      const transaction = database.transaction(STORE_NAME, "readwrite");
      const store = transaction.objectStore(STORE_NAME);
      const current = await requestResult<StoredBrowserTemplate | undefined>(store.get(id));
      if (current) store.delete(id);
      await transactionDone(transaction);
      return current !== undefined;
    });
  }

  async clear(): Promise<void> {
    await this.initialize();
    await this.withLock(async () => {
      const transaction = this.requireDatabase().transaction(STORE_NAME, "readwrite");
      transaction.objectStore(STORE_NAME).clear();
      await transactionDone(transaction);
    });
  }

  async close(): Promise<void> {
    this.database?.close();
    this.database = null;
    this.initialization = null;
  }

  private requireDatabase(): IDBDatabase {
    if (!this.database) throw new Error("Browser template repository is not initialized");
    return this.database;
  }

  private async withLock<T>(operation: () => Promise<T>): Promise<T> {
    if (!navigator.locks) throw new Error("Web Locks are unavailable");
    return navigator.locks.request(ARTIFACT_LOCK, { mode: "exclusive" }, operation);
  }
}

async function openDatabase(): Promise<IDBDatabase> {
  const database = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    let settled = false;
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
    request.onsuccess = () => {
      if (settled) {
        request.result.close();
        return;
      }
      settled = true;
      resolve(request.result);
    };
    request.onerror = () => {
      if (settled) return;
      settled = true;
      reject(request.error ?? new Error("Could not open template storage"));
    };
    request.onblocked = () => {
      if (settled) return;
      settled = true;
      reject(new Error("Template storage upgrade is blocked by another tab"));
    };
  });
  await migrateLegacyDatabase(database, [STORE_NAME]);
  return database;
}

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB transaction failed"));
    transaction.onabort = () => reject(transaction.error ?? new Error("IndexedDB transaction aborted"));
  });
}

function cloneRecord(record: StoredBrowserTemplate): StoredBrowserTemplate {
  return {
    ...record,
    payload: structuredClone(record.payload),
  };
}

function validRecord(value: unknown, key: IDBValidKey): StoredBrowserTemplate | null {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const keys = Object.keys(record).sort();
  if (keys.join("\0") !== ["id", "installedAt", "name", "namespace", "payload"].join("\0")) {
    return null;
  }
  if (
    typeof key !== "string" || record.id !== key ||
    typeof record.namespace !== "string" || typeof record.name !== "string" ||
    !/^[a-z][a-z0-9._-]{0,63}$/.test(record.namespace) ||
    !/^[a-z][a-z0-9._-]{0,63}$/.test(record.name) ||
    record.id !== `${record.namespace}/${record.name}` ||
    typeof record.installedAt !== "number" || !Number.isSafeInteger(record.installedAt) ||
    record.installedAt <= 0 || record.payload === null ||
    typeof record.payload !== "object" || Array.isArray(record.payload)
  ) return null;
  return {
    id: record.id,
    namespace: record.namespace,
    name: record.name,
    payload: structuredClone(record.payload as Record<string, unknown>),
    installedAt: record.installedAt,
  };
}
