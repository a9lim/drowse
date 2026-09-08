import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)), configFile: false,
  optimizeDeps: { noDiscovery: true, include: [] }, appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const previousSocket = globalThis.WebSocket;
const previousLocation = globalThis.location;
const sockets = [];
class Socket extends EventTarget {
  static OPEN = 1;
  readyState = 0;
  constructor() { super(); sockets.push(this); }
  connect() { this.readyState = 1; this.dispatchEvent(new Event("open")); }
  close() { this.readyState = 2; }
  finishClose() { this.readyState = 3; this.dispatchEvent(new Event("close")); }
  message(payload) { this.dispatchEvent(new MessageEvent("message", { data: JSON.stringify(payload) })); }
}

try {
  globalThis.WebSocket = Socket;
  globalThis.location = { protocol: "https:", host: "app.example" };
  const { HttpRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/http-client.ts");
  const client = new HttpRuntimeClient();
  const events = [];
  const states = [];
  client.events.subscribe(event => events.push(event));
  client.events.subscribeState(state => states.push(state));

  const first = client.events.open();
  const oldSocket = sockets.at(-1);
  oldSocket.connect();
  await first;
  client.events.close();
  oldSocket.message({ type: "token", token: "closed connection" });
  assert.deepEqual(events, [], "closed sockets must not dispatch queued messages");

  const second = client.events.open();
  const currentSocket = sockets.at(-1);
  currentSocket.connect();
  await second;
  const priorStates = structuredClone(states);
  oldSocket.finishClose();
  assert.deepEqual(states, priorStates, "an old socket must not close the replacement's state");
  oldSocket.message({ type: "token", token: "stale" });
  currentSocket.message({ type: "token", token: "current" });
  assert.deepEqual(events, [{ type: "token", token: "current" }]);
  assert.equal(client.events.isOpen, true);

  client.events.close();
  currentSocket.finishClose();
  assert.deepEqual(states.at(-1), { state: "closed", expected: true, reason: null });
  console.log("HTTP runtime lifecycle: closed and replaced socket events are ignored; active messages and expected shutdown are preserved");
} finally {
  globalThis.WebSocket = previousSocket;
  globalThis.location = previousLocation;
  await server.close();
}
