import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
const originalSessionStorage = Object.getOwnPropertyDescriptor(globalThis, "sessionStorage");
const originalWindow = Object.getOwnPropertyDescriptor(globalThis, "window");

try {
  let online = true;
  const values = new Map();
  const listeners = new Map([
    ["offline", new Set()],
    ["online", new Set()],
  ]);
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { get onLine() { return online; } },
  });
  Object.defineProperty(globalThis, "sessionStorage", {
    configurable: true,
    value: {
      getItem(key) { return values.get(key) ?? null; },
      setItem(key, value) { values.set(key, String(value)); },
      removeItem(key) { values.delete(key); },
    },
  });
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      addEventListener(type, listener) { listeners.get(type)?.add(listener); },
      removeEventListener(type, listener) { listeners.get(type)?.delete(listener); },
    },
  });

  values.set("drowse.hosted.offline", "1");
  const state = await server.ssrLoadModule("/src/hosted/runtime/networkState.ts");
  const stop = state.startHostedNetworkStateTracking();
  assert.equal(state.hostedNetworkIsOffline(), true);
  assert.equal(values.get("drowse.hosted.offline"), "1");

  for (const listener of listeners.get("online")) listener();
  assert.equal(state.hostedNetworkIsOffline(), false);
  assert.equal(values.has("drowse.hosted.offline"), false);

  online = false;
  for (const listener of listeners.get("offline")) listener();
  assert.equal(state.hostedNetworkIsOffline(), true);

  online = true;
  assert.equal(state.hostedNetworkIsOffline(), true);
  for (const listener of listeners.get("online")) listener();
  assert.equal(state.hostedNetworkIsOffline(), false);

  Object.defineProperty(globalThis, "sessionStorage", {
    configurable: true,
    value: {
      getItem() { throw new DOMException("blocked", "SecurityError"); },
      setItem() { throw new DOMException("blocked", "SecurityError"); },
      removeItem() { throw new DOMException("blocked", "SecurityError"); },
    },
  });
  assert.equal(state.hostedNetworkIsOffline(), false);
  online = false;
  for (const listener of listeners.get("offline")) listener();
  assert.equal(state.hostedNetworkIsOffline(), true);
  online = true;
  for (const listener of listeners.get("online")) listener();
  assert.equal(state.hostedNetworkIsOffline(), false);

  stop();
  assert.equal(listeners.get("offline").size, 0);
  assert.equal(listeners.get("online").size, 0);
  console.log("network-state navigation race regression passed");
} finally {
  await server.close();
  restore("navigator", originalNavigator);
  restore("sessionStorage", originalSessionStorage);
  restore("window", originalWindow);
}

function restore(name, descriptor) {
  if (descriptor) Object.defineProperty(globalThis, name, descriptor);
  else delete globalThis[name];
}
