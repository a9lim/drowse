const CACHE_NAME = "drowse-hosted-on-demand-assets-v1";
const MODULE_PATH = /^\/assets\/(?:App|browser\.worker|registry|drowse-web-llm)-[A-Za-z0-9_-]+\.(?:css|js)$/u;

export async function cacheOfflineRuntimeAssets(): Promise<void> {
  const response = await fetchAppFile("/runtime-assets.json");
  if (!response.ok) throw unavailable();
  const manifest: unknown = await response.json().catch(() => { throw unavailable(); });
  if (
    !manifest || typeof manifest !== "object" ||
    !("assets" in manifest) || !Array.isArray(manifest.assets) ||
    manifest.assets.length === 0 || manifest.assets.length > 32 ||
    !manifest.assets.every(path => typeof path === "string" && MODULE_PATH.test(path))
  ) throw unavailable();
  const cache = await caches.open(CACHE_NAME);
  for (let offset = 0; offset < manifest.assets.length; offset += 3) {
    const results = await Promise.allSettled(manifest.assets.slice(offset, offset + 3).map(async path => {
      if (await cache.match(path, { ignoreVary: true })) return;
      const asset = await fetchAppFile(path);
      const mime = asset.headers.get("content-type") ?? "";
      if (!asset.ok || !/^(?:text\/css|(?:text|application)\/javascript)\b/iu.test(mime)) {
        throw unavailable();
      }
      await cache.put(path, asset);
    }));
    for (const result of results) {
      if (result.status === "rejected") throw result.reason;
    }
  }
}

function fetchAppFile(path: string): Promise<Response> {
  return fetch(path, { signal: AbortSignal.timeout(60_000) }).catch(() => { throw unavailable(); });
}

function unavailable(): Error & { code: string } {
  return Object.assign(new Error("The model runtime app files could not be prepared for offline use"), {
    code: "APP_MODULE_UNAVAILABLE",
  });
}
