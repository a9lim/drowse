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

try {
  const { browserTextDiff, browserReadingsDiff, browserPerTokenDiff } =
    await server.ssrLoadModule("/src/hosted/runtime/browserLoomDiff.ts");

  assert.deepEqual(browserTextDiff("the quick brown fox", "the slow brown cat"), [
    { state: "equal", text: "the" },
    { state: "delete", text: "quick" },
    { state: "insert", text: "slow" },
    { state: "equal", text: "brown" },
    { state: "delete", text: "fox" },
    { state: "insert", text: "cat" },
  ]);
  assert.deepEqual(browserTextDiff("a b a b c", "a b c a b"), [
    { state: "delete", text: "a b" },
    { state: "equal", text: "a b c" },
    { state: "insert", text: "a b" },
  ]);
  assert.deepEqual(browserTextDiff("", "new words"), [
    { state: "insert", text: "new words" },
  ]);

  assert.deepEqual(browserReadingsDiff(
    { calm: 0.3, formal: -0.2 },
    { calm: -0.1, honest: 0.25 },
  ), [
    { name: "calm", delta: -0.4, a_value: 0.3, b_value: -0.1 },
    { name: "honest", delta: 0.25, a_value: 0, b_value: 0.25 },
    { name: "formal", delta: 0.2, a_value: -0.2, b_value: 0 },
  ]);

  assert.deepEqual(browserPerTokenDiff(
    [row("a"), row("é"), row("z")],
    [row("a"), row("e"), row("z")],
  ), [
    span(0, 0, "a", "a", true),
    span(1, 1, "é", "e", false),
    span(1, 2, "é", "z", false),
    span(2, 2, "z", "z", false),
    span(2, -1, "z", "", false),
  ]);
} finally {
  await server.close();
}

console.log("browser loom diff checks passed");

function row(text) {
  return { text, logprob: null, perplexity: null };
}

function span(a_index, b_index, a_text, b_text, aligned) {
  return { a_index, b_index, a_text, b_text, aligned, reading_deltas: [] };
}
