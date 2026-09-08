const clearers = new Set<() => void>();

export function registerDrowseLocalDataClearer(clearer: () => void): () => void {
  clearers.add(clearer);
  return () => clearers.delete(clearer);
}

export function clearDrowseLocalData(
  storage: Pick<Storage, "key" | "length" | "removeItem"> | undefined = safeLocalStorage(),
): void {
  let failed = false;
  for (const clear of clearers) {
    try {
      clear();
    } catch { failed = true; }
  }
  if (storage) try {
    const keys: string[] = [];
    for (let index = 0; index < storage.length; index += 1) {
      const key = storage.key(index);
      if (key && /^(?:drowse|polythetic|saklas)[.:]/u.test(key)) keys.push(key);
    }
    for (const key of keys) {
      try {
        storage.removeItem(key);
      } catch { failed = true; }
    }
  } catch { failed = true; }
  if (failed) throw new Error("Could not clear all local data. Close other Drowse tabs and retry.");
}

function safeLocalStorage(): Storage | undefined {
  try {
    return globalThis.localStorage;
  } catch {
    throw new Error("Could not clear all local data. Browser storage is inaccessible; restore storage access and retry.");
  }
}
