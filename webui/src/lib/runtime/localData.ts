const clearers = new Set<() => void>();

export function registerDrowseLocalDataClearer(clearer: () => void): () => void {
  clearers.add(clearer);
  return () => clearers.delete(clearer);
}

export function clearDrowseLocalData(
  storage: Pick<Storage, "key" | "length" | "removeItem"> | undefined = safeLocalStorage(),
): void {
  for (const clear of clearers) {
    try {
      clear();
    } catch {}
  }
  if (!storage) return;
  try {
    const keys: string[] = [];
    for (let index = 0; index < storage.length; index += 1) {
      const key = storage.key(index);
      if (key && ["drowse.", "polythetic.", "saklas."].some((prefix) => key.startsWith(prefix))) keys.push(key);
    }
    for (const key of keys) storage.removeItem(key);
  } catch {}
}

function safeLocalStorage(): Storage | undefined {
  try {
    return globalThis.localStorage;
  } catch {
    return undefined;
  }
}
