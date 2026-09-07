import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
const flush = () => new Promise(resolve => setImmediate(resolve));
const envelope = { measurements: { instruments: { geometry: { readings: {} } } } };
try {
  const { availableDrilldownTab } = await server.ssrLoadModule("/src/drawers/token/drilldown.svelte.ts");
  const none = { lens: false, sae: false, geometry: false };
  assert.equal(availableDrilldownTab("lens", none), "logits", "base models without instruments open logits");
  assert.equal(availableDrilldownTab("lens", { ...none, sae: true }), "sae");
  assert.equal(availableDrilldownTab("lens", { ...none, geometry: true }), "geometry");
  assert.equal(availableDrilldownTab("lens", { ...none, lens: true }), "lens");
  for (let mask = 0; mask < 8; mask += 1) {
    const available = { lens: !!(mask & 1), sae: !!(mask & 2), geometry: !!(mask & 4) };
    for (const preferred of ["lens", "sae", "geometry", "logits"]) {
      const selected = availableDrilldownTab(preferred, available);
      assert.ok(selected === "logits" || available[selected], "automatic selection must be usable");
      if (preferred === "logits" || available[preferred]) assert.equal(selected, preferred, "usable preferences stay sticky");
    }
  }
  const requests = [];
  const metadataRequests = [];
  let replay = (...args) => new Promise((resolve, reject) => requests.push({ args, resolve, reject }));
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({ mode: "browser", instruments: {
    tokenReadout: (...args) => replay(...args),
    saeFeaturesMetadata: ids => new Promise((resolve, reject) => metadataRequests.push({ ids, resolve, reject })),
  } });
  const { cachedTokenReadout, invalidateTokenReadoutCache } =
    await server.ssrLoadModule("/src/lib/runtime/tokenReadoutCache.ts");
  const accepted = [];
  for (let index = 0; index < 97; index += 1) {
    accepted.push(cachedTokenReadout("geometry", `queued-${index}`, 0, {}).catch(error => error));
  }
  assert.equal(requests.length, 1, "replays stay serialized");
  for (let index = 0; index < 500; index += 1) {
    await assert.rejects(cachedTokenReadout("geometry", `overflow-${index}`, 0, {}), /queue is full/);
  }
  const duplicate = cachedTokenReadout("geometry", "queued-96", 0, {}).catch(error => error);
  invalidateTokenReadoutCache();
  const cancelled = await Promise.all([...accepted, duplicate]);
  assert.ok(cancelled.every(error => /reading source changed/.test(error.message)));
  requests[0].resolve(envelope);
  await flush();
  assert.equal(requests.length, 1, "invalidated queued work never reaches the model");
  replay = () => { throw new Error("transport unavailable"); };
  await assert.rejects(cachedTokenReadout("geometry", "sync-error", 0, {}), /transport unavailable/);
  replay = async () => envelope;
  assert.deepEqual(await cachedTokenReadout("geometry", "recovered", 0, {}), envelope,
    "synchronous transport failures must release the active slot");
  const { sessionState } = await server.ssrLoadModule("/src/lib/stores/session.svelte.ts");
  const { saeState, recordSaeReadoutFrame, backfillSaeMeta, rehydrateInstrumentsFromSession } =
    await server.ssrLoadModule("/src/lib/stores/instruments.svelte.ts");
  const selectPack = source => {
    sessionState.info = { instruments: [{ family: "sae", source, live: { enabled: true, layer: 13 } }] };
    rehydrateInstrumentsFromSession();
  };
  selectPack("first-dictionary");
  recordSaeReadoutFrame([{ id: 2286, activation: 1, label: null, max_act: 42 }]);
  const firstMetadata = backfillSaeMeta();
  assert.deepEqual(metadataRequests[0].ids, [2286], "a maximum without a label still needs descriptions");
  metadataRequests[0].resolve({ features: { "2286": { label: "Reuters", max_act: 42 } } });
  await firstMetadata;
  recordSaeReadoutFrame([{ id: 2286, activation: 2, label: null, max_act: 42 }]);
  assert.equal(saeState.meta.get(2286).label, "Reuters", "later unlabeled frames preserve fetched descriptions");
  recordSaeReadoutFrame([{ id: 396, activation: 3, label: null, max_act: null }]);
  const staleMetadata = backfillSaeMeta();
  selectPack("second-dictionary");
  metadataRequests[1].resolve({ features: { "396": { label: "wrong dictionary", max_act: null } } });
  await staleMetadata;
  assert.equal(saeState.meta.has(396), false, "late responses cannot label another dictionary's features");
  recordSaeReadoutFrame([{ id: 396, activation: 4, label: null, max_act: null }]);
  const offline = backfillSaeMeta();
  metadataRequests[2].reject(new Error("offline"));
  await offline;
  const retry = backfillSaeMeta();
  metadataRequests[3].resolve({ features: { "396": { label: "correct dictionary", max_act: null } } });
  await retry;
  assert.equal(saeState.meta.get(396).label, "correct dictionary");
  console.log("SAE metadata lifecycle: missing-label backfill, frame preservation, pack-switch isolation, and offline retry passed");
  console.log("Readout lifecycle: bounded pending work, 500 overflow attempts, deduplication, invalidation, and transport recovery passed");
} finally {
  await server.close();
}
