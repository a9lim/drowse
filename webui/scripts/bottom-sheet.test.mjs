import assert from "node:assert/strict";
import { createServer } from "vite";
const server = await createServer({
  configFile: false, appType: "custom", logLevel: "error",
  optimizeDeps: { noDiscovery: true, include: [] },
  server: { middlewareMode: true, watch: null, hmr: false, ws: false },
});
try {
  const { sheetPositions, nearestSheetStop, rubberBandSheet, releaseSheet, sampleSheetVelocity, sheetSpring, sheetSpringVelocity } = await server.ssrLoadModule("/src/lib/bottomSheet.ts");
  for (const height of [240, 390, 844, 1200]) {
    const p = sheetPositions(height, 20);
    assert.ok(p.full < p.half && p.half < p.peek && p.peek < height - 90);
    assert.equal(nearestSheetStop(p.half, p), "half");
    assert.equal(releaseSheet("peek", p.full, -500, -100, p, height), "half", "slow drags cannot skip stops");
    assert.equal(releaseSheet("peek", p.peek - 60, -60, -800, p, height), "full", "fast flick skips the middle");
    assert.equal(releaseSheet("full", p.full + 90, 90, 100, p, height), "half");
    assert.equal(releaseSheet("full", p.full + 60, 60, 900, p, height), "peek", "fast collapse does not dismiss");
    assert.equal(releaseSheet("peek", p.peek + 65, 90, 200, p, height), "dismiss");
    assert.equal(releaseSheet("half", p.half - 5, -5, -900, p, height), "half", "jitter is not a fling");
    assert.equal(releaseSheet("half", p.half - 70, -70, 700, p, height), "peek", "recent reversed velocity determines direction");
    assert.ok(rubberBandSheet(p.full - 100, p) > p.full - 100);
    assert.ok(rubberBandSheet(p.peek + 100, p) < p.peek + 100);
  }
  const samples = [];
  sampleSheetVelocity(samples, 200, 0);
  sampleSheetVelocity(samples, 100, 50);
  assert.equal(sampleSheetVelocity(samples, 100, 250), 0, "paused releases discard stale velocity");
  assert.ok(sampleSheetVelocity(samples, 50, 300) < -600);
  assert.equal(sheetSpring(400, 20, -800, 0), 400);
  assert.ok(sheetSpring(400, 20, -800, 0.1) < sheetSpring(400, 20, 0, 0.1));
  assert.ok(Math.abs(sheetSpring(400, 20, -800, 0.3) - 20) < 1);
  assert.equal(sheetSpringVelocity(400, 20, -800, 0), -800);
  assert.ok(sheetSpringVelocity(400, 20, 0, 0.1) < 0);
  console.log("bottom-sheet detent and spring checks passed");
} finally { await server.close(); }
