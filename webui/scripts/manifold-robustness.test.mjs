import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({ root: fileURLToPath(new URL("..", import.meta.url)), configFile: false, appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
try {
  const { compileStructuredHookProgram, initialStructuredHookState, runStructuredHookLayer, geometryCoordinateMean } = await server.ssrLoadModule("/src/hosted/runtime/structuredHookProgram.ts");
  const { STANDARD_STRUCTURED_HOOK_PROFILE_V3: profile } = await server.ssrLoadModule("/src/hosted/runtime/structuredHookProfile.ts");
  const { measurementEnvelope, measurementGateScores } = await server.ssrLoadModule("/src/hosted/runtime/webLlmGeneration.ts");
  const points = [[1, 0], [0, 1], [-1, 0], [0, -1]];
  const curve = {
    active: true, basis: [[1, 0], [0, 1]], neutral: [0, 0],
    nodeParameters: points, rbfWeights: points.map(() => [0, 0]),
    polynomial: [[0, 0], [1, 0], [0, 1]], coordinateOffset: [0, 0], coordinateScale: [1, 1],
    origin: [0], target: [0.25], axes: [{ periodic: true, period: 1, lowerBound: 0, upperBound: 1 }], along: 0, onto: 0,
  };
  const layer = {
    geometryWhitener: { rank: 1, ridge: 1, basis: [[1, 0]], correction: [0] },
    geometryProbes: [{ active: true, mean: [0, 0], inverseMean: [0, 0], basis: [[1, 0], [0, 1]], gramInverse: [1, 0, 0, 1], cholesky: [1, 0, 0, 1], candidateWhite: points, coordinateMap: [[0, 0]], coordinateBias: [0], curve, curveNodeCoordinates: [[0], [0.25], [0.5], [0.75]], curveNodeValues: points }],
  };
  const program = compileStructuredHookProgram(2, [layer, layer], profile);
  program.measurementSchema = {
    layerMap: [3, 13], modelLayerCount: 26, probes: new Array(profile.maxProbes).fill(null), lensSource: null, saeSource: null,
    geometryProbes: [{ name: "local/circle", manifold: "local/circle", scoreKeys: ["local/circle"], labels: ["east", "north", "west", "south"], topN: 4, intrinsicDim: 1, rank: 2, shareWeights: [1, 1], assignBandwidth: [1, 1, 1, 1], assignLogVolumeBias: [0, 0, 0, 0], labelScale: 1 }, ...new Array(profile.maxGeometryProbes - 1).fill(null)],
  };
  function readAt(coordinates) {
    const state = initialStructuredHookState(program);
    const values = new Float32Array(2 * profile.maxGeometryProbes * profile.geometryOutputStride);
    coordinates.forEach((coord, index) => {
      const angle = coord * 2 * Math.PI;
      const result = runStructuredHookLayer(program, state, index, [Math.cos(angle), Math.sin(angle)], { decode: false });
      values.set(result.geometry, index * profile.maxGeometryProbes * profile.geometryOutputStride);
    });
    return { values, envelope: measurementEnvelope(program, new Float32Array(2 * profile.maxProbes), null, values) };
  }
  const { values, envelope } = readAt([0.99, 0.01]);
  const reading = envelope.instruments.geometry.readings["local/circle"];
  assert.ok(Math.abs(reading.coords_per_layer["3"][0] - 0.99) < 1e-4);
  assert.ok(Math.abs(reading.coords_per_layer["13"][0] - 0.01) < 1e-4);
  assert.ok(Math.min(reading.coords[0], 1 - reading.coords[0]) < 1e-4, `Circular mean crossed the opposite pole: ${reading.coords[0]}`);
  assert.ok(Math.min(envelope.scores["local/circle"], 1 - envelope.scores["local/circle"]) < 1e-4);
  assert.equal(measurementGateScores(envelope)["local/circle"], reading.coords[0]);
  for (const rotation of [0.2, 0.6]) {
    const rotated = readAt([rotation - 0.01, rotation + 0.01]).envelope.instruments.geometry.readings["local/circle"];
    assert.ok(Math.abs(rotated.coords[0] - rotation) < 1e-4);
  }
  const slot = profile.maxGeometryProbes;
  assert.equal(geometryCoordinateMean(program, [{ slot: 0, weight: 1, coords: [0] }, { slot, weight: 1, coords: [0.5] }])[0], 0);
  const unequal = geometryCoordinateMean(program, [{ slot: 0, weight: 3, coords: [0.99] }, { slot, weight: 1, coords: [0.01] }])[0];
  assert.ok(unequal > 0.99 && unequal < 1);
  const sphereProgram = { ...program, geometryDomainKind: new Uint32Array([2]) };
  const polarMean = geometryCoordinateMean(sphereProgram, [{ slot: 0, weight: 1, coords: [0.1, 0] }, { slot: 0, weight: 1, coords: [0.1, Math.PI] }]);
  assert.ok(Math.abs(polarMean[0]) < 1e-6, `Spherical mean missed the pole: ${polarMean}`);
  values.fill(0, slot * profile.geometryOutputStride);
  const sparse = measurementEnvelope(program, new Float32Array(2 * profile.maxProbes), null, values).instruments.geometry.readings["local/circle"];
  assert.ok(Math.abs(sparse.coords[0] - 0.99) < 1e-4);

  const warmState = initialStructuredHookState(program);
  for (const phase of [0, 0.01, 0.5, 0.99, 0.25]) {
    const angle = phase * 2 * Math.PI;
    const result = runStructuredHookLayer(program, warmState, 0, [Math.cos(angle), Math.sin(angle)], { decode: phase !== 0 });
    assert.ok(Math.abs(((result.geometry[4] - phase + 1.5) % 1) - 0.5) < 1e-3, `Warm readout lost the circle at ${phase}: ${result.geometry[4]}`);
  }
  for (const phase of [0, 0.13, 0.5, 0.99]) {
    const angle = phase * 2 * Math.PI;
    const input = [Math.cos(angle), Math.sin(angle)];
    const identityProgram = compileStructuredHookProgram(2, [{ curves: [curve] }], profile);
    const identity = runStructuredHookLayer(identityProgram, initialStructuredHookState(identityProgram), 0, input, { decode: false });
    input.forEach((value, axis) => assert.ok(Math.abs(identity.residual[axis] - value) < 1e-5));
  }
  console.log("Manifold robustness checks passed: periodic seams, rotation, sparse layers, weighted/ambiguous means, gate agreement, warm jumps, and zero-strength identity");
} finally {
  await server.close();
}
