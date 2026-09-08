import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  configFile: false,
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const original = Object.fromEntries(["navigator", "window", "location", "sessionStorage", "caches", "fetch"].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
let cases = 0;

async function setup() {
  const values = new Map();
  const storage = { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key) };
  const navigation = [];
  const requests = [];
  const deleted = [];
  const location = new URL("https://drowse.test/app?choose=1#saved");
  location.replace = url => navigation.push(url);
  const navigator = { onLine: true, platform: "MacIntel", userAgent: "Macintosh", maxTouchPoints: 0 };
  const globals = {
    navigator, location, sessionStorage: storage,
    window: { location, history: { state: null, replaceState: (_state, _title, url) => { location.href = String(url); } } },
    caches: { keys: async () => ["drowse-hosted-precache-v2-https://drowse.test/", "drowse-hosted-on-demand-assets-v1", "saklas-hosted-landing-shader-v1", "drowse-hosted-on-demand-wasm-v1", "webllm/model", "webllm/wasm", "unrelated", "drowse-hosted-models", "drowse-hosted-on-demand-assets-v1-extra"], delete: async name => { deleted.push(name); return true; } },
    fetch: async (url, options) => {
      requests.push({ url: new URL(url), options });
      return new Response('<meta name="drowse-source-revision" content="test">', { headers: { "Content-Type": "text/html" } });
    },
  };
  for (const [key, value] of Object.entries(globals)) Object.defineProperty(globalThis, key, { value, configurable: true, writable: true });
  const recovery = await server.ssrLoadModule(`/src/hosted/runtime/chunkRecovery.ts?test=${++cases}`);
  return { ...recovery, values, storage, navigation, requests, deleted, location, navigator };
}

try {
  {
    const env = await setup();
    for (const message of ["Failed to fetch dynamically imported module", "Importing a module script failed.", "Unable to preload CSS"]) assert.equal(env.isChunkLoadError(new Error(message)), true);
    assert.equal(env.isChunkLoadError(new Error("Model allocation failed")), false);
    await env.reloadHostedApp("bootstrap");
    assert.equal(env.requests.length, 1);
    assert.equal(env.requests[0].options.cache, "no-store");
    assert.equal(env.requests[0].url.pathname, "/index.html");
    assert.ok(env.requests[0].url.searchParams.has("app-recovery"));
    assert.deepEqual(env.deleted, ["drowse-hosted-precache-v2-https://drowse.test/", "drowse-hosted-on-demand-assets-v1", "saklas-hosted-landing-shader-v1"]);
    const destination = new URL(env.navigation[0]);
    assert.equal(destination.pathname, "/app");
    assert.equal(destination.searchParams.get("choose"), "1");
    assert.equal(destination.hash, "#saved");
    assert.match(destination.searchParams.get("app-recovery"), /^bootstrap:\d+$/);
  }
  {
    const env = await setup();
    env.navigator.onLine = false;
    await assert.rejects(env.reloadHostedApp("bootstrap"), /offline/);
    assert.deepEqual(env.requests, []);
    assert.deepEqual(env.deleted, []);
    assert.deepEqual(env.navigation, []);
  }
  for (const response of [new Response("Unavailable", { status: 503 }), new Response("not HTML"), new Response("<html>Captive portal</html>", { headers: { "content-type": "text/html" } })]) {
    const env = await setup();
    globalThis.fetch = async () => response;
    await assert.rejects(env.reloadHostedApp("bootstrap"), /app files are unavailable/);
    assert.deepEqual(env.deleted, []);
    assert.deepEqual(env.navigation, []);
  }
  {
    const env = await setup();
    globalThis.fetch = async () => { throw new TypeError("network failure"); };
    await assert.rejects(env.reloadHostedApp("bootstrap"), /Check your connection/);
    assert.deepEqual(env.deleted, []);
  }
  {
    const env = await setup();
    await assert.rejects(env.reloadHostedApp("workbench", async () => { throw new Error("Save failed"); }), /Save failed/);
    assert.deepEqual(env.deleted, []);
    assert.deepEqual(env.navigation, []);
    await env.reloadHostedApp("workbench");
    assert.equal(env.navigation.length, 1, "A failed preparation leaves the manual retry usable");
  }
  {
    const env = await setup();
    env.location.searchParams.set("app-recovery", "bootstrap:123");
    env.storage.getItem = () => { throw new Error("Storage blocked"); };
    env.storage.setItem = () => { throw new Error("Storage blocked"); };
    assert.equal(await env.recoverFromChunkLoadError(new Error("Failed to fetch dynamically imported module"), "bootstrap"), false);
    assert.deepEqual(env.requests, []);
    await env.reloadHostedApp("bootstrap");
    assert.equal(env.navigation.length, 1, "The manual retry bypasses the automatic loop guard");
  }
  {
    const env = await setup();
    env.location.searchParams.set("app-recovery", "workbench:123");
    env.storage.setItem("drowse:chunk-recovery", JSON.stringify({ stage: "workbench", pathname: "/app", createdAt: Date.now() }));
    env.completeChunkRecovery("bootstrap");
    assert.equal(env.location.searchParams.get("app-recovery"), "workbench:123");
    assert.ok(env.storage.getItem("drowse:chunk-recovery"));
    env.completeChunkRecovery("workbench");
    assert.equal(env.location.searchParams.has("app-recovery"), false);
    assert.equal(env.storage.getItem("drowse:chunk-recovery"), null);
  }
  {
    const env = await setup();
    env.storage.setItem("drowse:chunk-recovery", JSON.stringify({ stage: "bootstrap", pathname: "/app", createdAt: Date.now() }));
    assert.equal(await env.recoverFromChunkLoadError(new Error("Failed to fetch dynamically imported module"), "bootstrap"), false);
    assert.deepEqual(env.navigation, []);
  }
  {
    const env = await setup();
    let unregistered = 0;
    env.navigator.serviceWorker = {
      getRegistration: async () => ({ scope: "https://drowse.test/tools/", active: { scriptURL: "https://drowse.test/tools/sw.js" }, unregister: async () => { unregistered += 1; } }),
    };
    await env.reloadHostedApp("bootstrap");
    assert.equal(unregistered, 0, "Other application workers are never removed");
  }
  {
    const env = await setup();
    let unregistered = 0;
    env.navigator.serviceWorker = {
      getRegistration: async () => ({ scope: "https://drowse.test/", active: { scriptURL: "https://drowse.test/sw.js" }, unregister: async () => { unregistered += 1; } }),
    };
    await env.reloadHostedApp("bootstrap");
    assert.equal(unregistered, 1);
    assert.equal(env.navigation.length, 1);
  }
  {
    const env = await setup();
    globalThis.caches = undefined;
    await env.reloadHostedApp("bootstrap");
    assert.equal(env.navigation.length, 1);
  }
  {
    const env = await setup();
    assert.match(env.reloadInstructions(), /⌘ \+ R/);
    for (const platform of ["Win32", "Linux x86_64"]) {
      env.navigator.platform = platform;
      assert.match(env.reloadInstructions(), /Ctrl \+ R/);
    }
    env.navigator.platform = "MacIntel";
    env.navigator.maxTouchPoints = 5;
    assert.match(env.reloadInstructions(), /browser’s menu/);
    env.navigator.userAgent = "Android";
    env.navigator.platform = "Linux";
    assert.match(env.reloadInstructions(), /browser’s menu/);
  }
  console.log(`chunk recovery: ${cases} cases passed (cache ownership, data safety, network failures, retries, loop guards, OS guidance)`);
} finally {
  for (const [key, descriptor] of Object.entries(original)) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
  await server.close();
}
