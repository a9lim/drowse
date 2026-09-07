export type PersistentStorageManager = Partial<
  Pick<StorageManager, "persist" | "persisted">
>;

export const STORAGE_PERSISTENCE_TIMEOUT_MS = 5_000;

export async function requestPersistentStorage(
  storage: PersistentStorageManager | undefined = navigator.storage,
  timeoutMs = STORAGE_PERSISTENCE_TIMEOUT_MS,
): Promise<boolean> {
  if (!storage) return false;

  const checks: Promise<boolean>[] = [];

  // Start the permission request before any await so it remains attached to
  // the button click in browsers that require transient user activation.
  if (storage.persist) checks.push(startBooleanCheck(() => storage.persist!()));
  if (storage.persisted) checks.push(startBooleanCheck(() => storage.persisted!()));
  if (checks.length === 0) return false;

  try {
    return await withTimeout(
      firstGranted(checks),
      positiveTimeout(timeoutMs),
    );
  } catch {
    return false;
  }
}

function startBooleanCheck(check: () => Promise<boolean>): Promise<boolean> {
  try {
    return Promise.resolve(check()).then((value) => value === true, () => false);
  } catch {
    return Promise.resolve(false);
  }
}

function firstGranted(checks: Promise<boolean>[]): Promise<boolean> {
  return new Promise((resolve) => {
    let remaining = checks.length;
    let settled = false;
    for (const check of checks) {
      check.then((granted) => {
        if (settled) return;
        if (granted) {
          settled = true;
          resolve(true);
          return;
        }
        remaining -= 1;
        if (remaining === 0) {
          settled = true;
          resolve(false);
        }
      });
    }
  });
}

function withTimeout<T>(operation: Promise<T>, timeoutMs: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error("The persistent-storage request timed out")),
      timeoutMs,
    );
    operation.then(
      (value) => {
        clearTimeout(timeout);
        resolve(value);
      },
      (error) => {
        clearTimeout(timeout);
        reject(error);
      },
    );
  });
}

function positiveTimeout(timeoutMs: number): number {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new RangeError("Storage persistence timeouts must be positive numbers");
  }
  return timeoutMs;
}
