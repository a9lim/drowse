import assert from "node:assert/strict";
import { createServer } from "vite";

const server = await createServer({ configFile: false, appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
try {
  const { parseSurfacePoints, exampleSphere, surfaceLabel, quotientDescription, requireBrowserDomain } = await server.ssrLoadModule("/src/lib/manifolds/surfaceGeometry.ts");
  const points = exampleSphere();
  assert.equal(points.length, 96);
  assert.deepEqual(parseSurfacePoints(JSON.stringify(points)), points);
  for (const invalid of ["", "{}", "[]", JSON.stringify(points.slice(0, 31)), JSON.stringify([...points, points[0]]), JSON.stringify(points.map(() => [true, 1])), JSON.stringify([...points.slice(1), [1, 2]])]) {
    assert.throws(() => parseSurfacePoints(invalid));
  }
  assert.equal(surfaceLabel(null), "Unresolved surface");
  assert.equal(surfaceLabel("klein-bottle"), "Klein bottle");
  assert.match(quotientDescription({ type: "klein" }), /reverses v/);
  assert.match(quotientDescription({ type: "projective", dim: 2 }), /same position/);
  for (const domain of [{ type: "klein" }, { type: "projective", dim: 2 }]) assert.throws(() => requireBrowserDomain(domain), /native Python/);
  requireBrowserDomain({ type: "sphere", dim: 2 });
  console.log("Surface input, labels, seam descriptions, and hosted guards passed");
} finally { await server.close(); }
