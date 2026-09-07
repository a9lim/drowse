import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const lists = [];
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  const probe = (name, family = "lens") => ({ name, family, word: name, layers: [0] });
  installRuntimeClient({ mode: "browser", probes: {
    list: () => new Promise((resolve, reject) => lists.push({ resolve, reject })),
    attach: async ({ selector }) => probe(selector),
    detach: async () => {},
  } });
  const { refreshProbeList, attachProbe, detachProbe, probeRack } =
    await server.ssrLoadModule("/src/lib/stores/probes.svelte.ts");
  const { tokenReadoutCacheEpoch } = await server.ssrLoadModule("/src/lib/runtime/tokenReadoutCache.ts");

  const older = refreshProbeList();
  const newer = refreshProbeList();
  lists[1].resolve({ probes: [probe("current")] });
  await newer;
  lists[0].resolve({ probes: [probe("obsolete")] });
  await older;
  assert.deepEqual(probeRack.active, ["current"], "old responses cannot replace newer rosters");
  const epoch = tokenReadoutCacheEpoch("lens");
  const unchanged = refreshProbeList();
  lists[2].resolve({ probes: [probe("current")] });
  await unchanged;
  assert.equal(tokenReadoutCacheEpoch("lens"), epoch, "opening the rack does not invalidate unchanged readouts");

  const failed = refreshProbeList();
  lists[3].reject(new Error("temporary disconnect"));
  await failed;
  assert.deepEqual(probeRack.active, ["current"], "a failed refresh preserves the known roster");
  assert.match(probeRack.error, /temporary disconnect/);
  const beforeAttach = refreshProbeList();
  await attachProbe("new");
  lists[4].resolve({ probes: [probe("current")] });
  await beforeAttach;
  assert.deepEqual(probeRack.active, ["current", "new"], "late list responses cannot erase an attachment");
  assert.equal(tokenReadoutCacheEpoch("lens"), epoch + 1, "lens attachments invalidate lens replay readings");
  assert.equal(probeRack.loading, false);
  const beforeDetach = refreshProbeList();
  await detachProbe("new");
  lists[5].resolve({ probes: [probe("current"), probe("new")] });
  await beforeDetach;
  assert.deepEqual(probeRack.active, ["current"], "late lists cannot resurrect a detached probe");
  assert.equal(tokenReadoutCacheEpoch("lens"), epoch + 2);
  assert.equal(probeRack.loading, false);
  console.log("Probe lifecycle: out-of-order refresh, transient failure, attach/detach races, and family cache invalidation passed");
} finally {
  await server.close();
}
