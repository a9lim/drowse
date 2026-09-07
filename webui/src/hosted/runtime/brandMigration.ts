export const LEGACY_PRODUCT_SLUGS = ["polythetic", "saklas"] as const;
const PRODUCT_SLUG = "drowse";

type DatabaseInfo = { name?: string };
type IndexedDbWithListing = IDBFactory & {
  databases?: () => Promise<DatabaseInfo[]>;
};
type MovableDirectoryHandle = FileSystemDirectoryHandle & {
  move?: (name: string) => Promise<void>;
};

export function legacyDatabaseNames(currentName: string): string[] {
  if (!currentName.startsWith(`${PRODUCT_SLUG}-`)) return [];
  return LEGACY_PRODUCT_SLUGS.map((slug) => currentName.replace(PRODUCT_SLUG, slug));
}

export function migrateLegacyStorageItem(
  storage: Pick<Storage, "getItem" | "setItem" | "removeItem">,
  currentKey: string,
): string | null {
  const current = storage.getItem(currentKey);
  if (current !== null) return current;
  if (!currentKey.startsWith(`${PRODUCT_SLUG}.`) && !currentKey.startsWith(`${PRODUCT_SLUG}:`)) return null;
  for (const slug of LEGACY_PRODUCT_SLUGS) {
    const legacyKey = currentKey.replace(PRODUCT_SLUG, slug);
    const legacy = storage.getItem(legacyKey);
    if (legacy === null) continue;
    storage.setItem(currentKey, legacy);
    return legacy;
  }
  return null;
}

export async function migrateLegacyDatabase(
  current: IDBDatabase,
  stores: readonly string[],
): Promise<void> {
  const factory = indexedDB as IndexedDbWithListing;
  if (typeof factory.databases !== "function") return;
  try {
    const databases = await factory.databases();
    for (const legacyName of legacyDatabaseNames(current.name)) {
      if (!databases.some((database) => database.name === legacyName)) continue;
      const legacy = await openExistingDatabase(factory, legacyName);
      try {
        for (const storeName of stores) {
          if (
            !legacy.objectStoreNames.contains(storeName) ||
            !current.objectStoreNames.contains(storeName) ||
            await storeCount(current, storeName) !== 0
          ) {
            continue;
          }
          await copyStore(legacy, current, storeName);
        }
      } finally {
        legacy.close();
      }
    }
  } catch {
    // Migration is opportunistic. The current store must remain usable if a
    // legacy database is corrupt, blocked, or unavailable in this browser.
  }
}

export function removeLegacyStorageItem(storage: Pick<Storage, "removeItem">, currentKey: string): void {
  storage.removeItem(currentKey);
  if (!currentKey.startsWith(`${PRODUCT_SLUG}.`) && !currentKey.startsWith(`${PRODUCT_SLUG}:`)) return;
  for (const slug of LEGACY_PRODUCT_SLUGS) storage.removeItem(currentKey.replace(PRODUCT_SLUG, slug));
}

export async function drowseStorageRoot(
  root: FileSystemDirectoryHandle,
): Promise<FileSystemDirectoryHandle> {
  const current = await existingDirectory(root, PRODUCT_SLUG);
  if (current !== null) return current;
  for (const slug of LEGACY_PRODUCT_SLUGS) {
    const legacy = await existingDirectory(root, slug);
    if (legacy === null) continue;
    const movable = legacy as MovableDirectoryHandle;
    if (typeof movable.move === "function") {
      try {
        await movable.move(PRODUCT_SLUG);
        return root.getDirectoryHandle(PRODUCT_SLUG);
      } catch {
        // Reuse older OPFS implementations without copying model weights.
      }
    }
    return legacy;
  }
  return root.getDirectoryHandle(PRODUCT_SLUG, { create: true });
}

export function migrateLegacyRecord(value: unknown): unknown {
  return migrateRecord(value);
}

function migrateRecord(value: unknown, parentKey = ""): unknown {
  if (Array.isArray(value)) return value.map((entry) => migrateRecord(entry, parentKey));
  if (value === null || typeof value !== "object") return value;
  if (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null) return value;
  const migrated: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    const currentKey = migrateKey(key);
    if (currentKey !== key && Object.hasOwn(value, currentKey)) continue;
    migrated[currentKey] = typeof entry === "string"
      ? migrateString(key, entry, parentKey)
      : migrateRecord(entry, key);
  }
  return migrated;
}

function migrateKey(key: string): string {
  for (const slug of LEGACY_PRODUCT_SLUGS) {
    if (key === `${slug}_version`) return "drowse_version";
    if (key === `${slug}Version`) return "drowseVersion";
    if (key === `${slug}Kind`) return "drowseKind";
  }
  return key;
}

function migrateString(key: string, value: string, parentKey: string): string {
  for (const slug of LEGACY_PRODUCT_SLUGS) {
    if ((key === "producer" || (parentKey === "producer" && key === "name")) && value === slug) return PRODUCT_SLUG;
    if ((key === "kind" || key === `${slug}Kind`) && value === `${slug}-manifold`) return `${PRODUCT_SLUG}-manifold`;
  }
  return value;
}

async function existingDirectory(
  root: FileSystemDirectoryHandle,
  name: string,
): Promise<FileSystemDirectoryHandle | null> {
  try {
    return await root.getDirectoryHandle(name);
  } catch (error) {
    if (error instanceof DOMException && error.name === "NotFoundError") return null;
    throw error;
  }
}

function openExistingDatabase(factory: IDBFactory, name: string): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = factory.open(name);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Legacy browser storage could not be opened"));
    request.onblocked = () => reject(new Error("Legacy browser storage is blocked by another tab"));
  });
}

function storeCount(database: IDBDatabase, storeName: string): Promise<number> {
  return requestValue(database.transaction(storeName, "readonly").objectStore(storeName).count());
}

async function copyStore(
  source: IDBDatabase,
  destination: IDBDatabase,
  storeName: string,
): Promise<void> {
  const sourceTransaction = source.transaction(storeName, "readonly");
  const sourceStore = sourceTransaction.objectStore(storeName);
  const [keys, values] = await Promise.all([
    requestValue(sourceStore.getAllKeys()),
    requestValue(sourceStore.getAll()),
    transactionDone(sourceTransaction),
  ]);
  if (values.length === 0) return;
  const destinationTransaction = destination.transaction(storeName, "readwrite");
  const destinationStore = destinationTransaction.objectStore(storeName);
  values.forEach((value, index) => {
    const migrated = migrateLegacyRecord(value);
    if (destinationStore.keyPath === null) destinationStore.put(migrated, keys[index]);
    else destinationStore.put(migrated);
  });
  await transactionDone(destinationTransaction);
}

function requestValue<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Browser storage migration failed"));
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error("Browser storage migration failed"));
    transaction.onabort = () => reject(transaction.error ?? new Error("Browser storage migration was aborted"));
  });
}
