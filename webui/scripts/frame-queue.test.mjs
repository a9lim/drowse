import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  configFile: false, appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const { createFrameQueue } = await server.ssrLoadModule("/src/lib/runtime/frameQueue.ts");
  const frames = new Map();
  const received = [];
  let serial = 0;
  const queue = createFrameQueue(
    (value) => received.push(value),
    (callback) => { frames.set(++serial, callback); return serial; },
    (id) => frames.delete(id),
  );
  for (let i = 0; i < 100; i++) queue.push(i);
  assert.equal(frames.size, 1);
  assert.deepEqual(received, []);
  frames.values().next().value();
  assert.deepEqual(received, Array.from({ length: 100 }, (_, i) => i));
  assert.equal(frames.size, 0);
  queue.push(100);
  queue.flush();
  received.push("done");
  assert.deepEqual(received.slice(-2), [100, "done"]);
  assert.equal(frames.size, 0);
  queue.flush();
  assert.equal(received.length, 102);
  for (let i = 0; i < 256; i++) queue.push(i);
  assert.equal(received.length, 358);
  assert.equal(frames.size, 0);
  const { createTokenViewCache } = await server.ssrLoadModule("/src/lib/runtime/tokenViews.ts");
  const visible = createTokenViewCache();
  const tokens = [{ text: " " }, { text: "hello" }];
  const firstViews = visible(tokens);
  assert.equal(firstViews[0].originalIdx, 1);
  tokens.push({ text: " world" });
  assert.equal(visible(tokens)[0], firstViews[0]);
  tokens[1] = { text: "goodbye" };
  assert.equal(visible(tokens)[0].tok.text, "goodbye");
  tokens[0].text = "first";
  assert.equal(visible(tokens)[0].originalIdx, 0);
  tokens.splice(1);
  assert.equal(visible(tokens).length, 1);
  tokens.push({ text: " second" });
  assert.equal(visible(tokens)[1].tok.text, " second");
  console.log("ok - frame delivery preserves every event, terminal order, and bounded pending work");
} finally {
  await server.close();
}
