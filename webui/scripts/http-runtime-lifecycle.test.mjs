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
  // Every SSE operation goes through one POST/stream/done helper.
  const { apiManifoldFitStream, apiExtractStream } = await server.ssrLoadModule("/src/lib/api.ts");
  const previousFetch = globalThis.fetch;
  const posted = [];
  const sse = frames => new Response(new ReadableStream({ start(controller) {
    for (const frame of frames) controller.enqueue(new TextEncoder().encode(frame));
    controller.close();
  } }), { status: 200, headers: { "Content-Type": "text/event-stream" } });
  try {
    globalThis.fetch = async (path, init) => {
      posted.push({ path, body: JSON.parse(init.body), request: init.headers["X-Drowse-Request-Id"] });
      return path.endsWith("/extract")
        ? sse(['event: done\ndata: {"canonical":"calm"}\n\n'])
        : sse(['event: progress\ndata: {"stage":"capture"}\n\n', 'event: done\ndata: {"name":"m","layers_fitted":3}\n\n']);
    };
    const seen = [];
    const fitted = await apiManifoldFitStream("local", "m", { sae: null }, event => seen.push(event.event), "op-1");
    assert.deepEqual(seen, ["progress", "done"]);
    assert.equal(fitted.layers_fitted, 3);
    assert.deepEqual(posted[0], { path: "/drowse/v1/manifolds/local/m/fit", body: { sae: null }, request: "op-1" });
    await assert.rejects(apiExtractStream({ concept: "calm" }, () => {}), /extract: stream ended without done event/,
      "an incomplete done payload is not a result");
    globalThis.fetch = async () => sse(['event: error\ndata: {"message":"not poised","code":"PoisednessError","status":422}\n\n']);
    await assert.rejects(apiManifoldFitStream("local", "m", {}, () => {}), error => error.status === 422 && /not poised/.test(error.message));
  } finally { globalThis.fetch = previousFetch; }
  console.log("HTTP runtime lifecycle: closed and replaced socket events are ignored; active messages and expected shutdown are preserved; SSE operations stream, resolve, and fail through one path");
} finally {
  globalThis.WebSocket = previousSocket;
  globalThis.location = previousLocation;
  await server.close();
}
