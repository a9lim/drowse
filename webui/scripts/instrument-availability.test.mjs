import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

const contextBinding = "a".repeat(64);
const catalog = {
  models: [{
    variants: [{
      id: "with-sae",
      contextProfiles: [{ contextTokens: 2048, bindingSha256: contextBinding }],
      packs: [{
        kind: "sae",
        compatibleContextBindingSha256: [contextBinding],
      }],
    }, {
      id: "without-sae",
      contextProfiles: [{ contextTokens: 2048, bindingSha256: contextBinding }],
      packs: [],
    }, {
      id: "wrong-context",
      contextProfiles: [{ contextTokens: 2048, bindingSha256: contextBinding }],
      packs: [{
        kind: "sae",
        compatibleContextBindingSha256: ["b".repeat(64)],
      }],
    }],
  }],
};

try {
  const { catalogInstrumentAvailability } = await server.ssrLoadModule(
    "/src/lib/runtime/instrumentAvailability.ts",
  );
  assert.equal(
    catalogInstrumentAvailability(catalog, "with-sae", 2048, "sae"),
    "available",
  );
  assert.equal(
    catalogInstrumentAvailability(catalog, "without-sae", 2048, "sae"),
    "unavailable",
  );
  assert.equal(
    catalogInstrumentAvailability(catalog, "wrong-context", 2048, "sae"),
    "context-unavailable",
  );
  assert.equal(
    catalogInstrumentAvailability(catalog, "missing", 2048, "sae"),
    "unknown",
  );
} finally {
  await server.close();
}

console.log("Instrument catalog availability checks passed");
