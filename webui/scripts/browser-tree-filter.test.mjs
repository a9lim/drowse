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
  const { filterBrowserTree, parseBrowserTreeFilter } = await server.ssrLoadModule(
    "/src/hosted/runtime/browserTreeFilter.ts",
  );
  const rows = [
    node("root", {}, []),
    node("a", { confidence: 0.4, "default/personas[1]": -0.2 }, [
      token({ confidence: 0.1 }),
      token({ confidence: 0.8 }),
    ]),
    node("b", { confidence: 0.7, "default/personas[1]": 0.3 }, [
      token({ confidence: 0.6 }),
      token({ confidence: 0.5 }),
    ]),
    node("c", { confidence: 0.9 }, [
      legacyToken({ confidence: 0.2 }),
      legacyToken({ confidence: 0.4 }),
    ]),
  ];

  assert.deepEqual(filterBrowserTree(rows, ""), []);
  assert.deepEqual(filterBrowserTree(rows, "confidence >= 0.7"), ["b", "c"]);
  assert.deepEqual(filterBrowserTree(rows, "agg:confidence > 0.3,"), ["a", "b", "c"]);
  assert.deepEqual(filterBrowserTree(rows, "any:confidence > 0.7"), ["a"]);
  assert.deepEqual(filterBrowserTree(rows, "any:confidence < 0.3"), ["a", "c"]);
  assert.deepEqual(filterBrowserTree(rows, "last:confidence >= 0.5"), ["a", "b"]);
  assert.deepEqual(
    filterBrowserTree(rows, "agg:confidence > 0.5, last:confidence < 0.5"),
    ["c"],
  );
  assert.deepEqual(
    filterBrowserTree(rows, "default/personas[1] >= 0"),
    ["b"],
  );
  assert.deepEqual(filterBrowserTree(rows, "agg:missing > 0"), []);

  assert.throws(() => parseBrowserTreeFilter("unknown:confidence > 0"), /unknown agg op/);
  assert.throws(() => parseBrowserTreeFilter("agg:confidence"), /missing comparison op/);
  assert.throws(() => parseBrowserTreeFilter("agg:9confidence > 0"), /not a valid identifier/);
  assert.throws(() => parseBrowserTreeFilter("agg:confidence > nope"), /is not a number/);
  assert.throws(() => parseBrowserTreeFilter("agg:confidence > 0x10"), /is not a number/);
  assert.throws(() => parseBrowserTreeFilter("  "), /empty filter expression/);
} finally {
  await server.close();
}

console.log("browser tree filter checks passed");

function node(id, aggregate_readings, tokens) {
  return { id, aggregate_readings, tokens, thinking_tokens: null };
}

function token(scores) {
  return {
    text: "x",
    logprob: null,
    perplexity: null,
    measurements: {
      version: 1,
      scope: "token",
      provenance: "captured",
      instruments: {},
      scores,
    },
  };
}

function legacyToken(probes) {
  return { text: "x", logprob: null, perplexity: null, probes };
}
