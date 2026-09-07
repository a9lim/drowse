import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
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
  const hooks = await server.ssrLoadModule("/src/hosted/runtime/rankOneHookProgram.ts");
  const fixture = JSON.parse(await readFile(
    new URL("../../browser-runtime/fixtures/post-block-rank-one-v2.json", import.meta.url),
    "utf8",
  ));
  const layers = fixture.cases.map((entry) => ({
    enabled: true,
    basis: entry.basis,
    neutral: entry.neutral,
    target: entry.target,
    along: entry.along,
    collapse: entry.collapse,
    probeBasis: entry.probeBasis,
    probeNeutral: entry.probeNeutral,
  }));
  const program = hooks.compileRankOneHookProgram(fixture.hiddenSize, layers);
  assert.equal(program.hookAbi, fixture.hookAbi);
  fixture.cases.forEach((entry, layerIndex) => {
    const result = hooks.runRankOneHookLayer(program, layerIndex, entry.residual);
    assert.ok(Math.abs(result.coordinate - entry.expectedCoordinate) <= 1e-6);
    assert.ok(Math.abs(result.probe - entry.expectedProbe) <= 1e-6);
    Array.from(result.residual).forEach((value, index) => {
      assert.ok(
        Math.abs(value - entry.expectedResidual[index]) <= 1e-6,
        `${entry.name} residual ${index}`,
      );
    });
  });
  console.log("ok 1 - fp32 rank-one hook buffers match the shared Python golden");

  const disabled = hooks.compileRankOneHookProgram(fixture.hiddenSize, [{
    ...layers[0],
    enabled: false,
  }]);
  const input = new Float32Array(fixture.cases[0].residual);
  const output = hooks.runRankOneHookLayer(disabled, 0, input);
  assert.deepEqual(output.residual, input);

  const fullyDisabled = hooks.compileRankOneHookProgram(fixture.hiddenSize, [{
    ...layers[0],
    enabled: false,
    basis: [0, 0, 0, 0],
    probeBasis: [0, 0, 0, 0],
  }]);
  const disabledOutput = hooks.runRankOneHookLayer(fullyDisabled, 0, input);
  assert.deepEqual(disabledOutput.residual, input);
  assert.equal(disabledOutput.coordinate, 0);
  assert.equal(disabledOutput.probe, 0);
  console.log("ok 2 - disabled steering preserves the residual exactly");

  assert.throws(
    () => hooks.compileRankOneHookProgram(4, [{ ...layers[0], basis: [1, 1, 0, 0] }]),
    /unit length/,
  );
  assert.throws(
    () => hooks.compileRankOneHookProgram(4, [{ ...layers[0], target: Infinity }]),
    /finite/,
  );
  assert.throws(
    () => hooks.runRankOneHookLayer({ ...program, target: new Float32Array() }, 0, input),
    /declared dimensions/,
  );
  console.log("ok 3 - malformed hook programs fail before GPU upload");
  console.log("1..3");
} finally {
  await server.close();
}
