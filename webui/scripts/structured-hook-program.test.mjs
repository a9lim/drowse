import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import {
  validateDrowseJlensDictionary,
  validateDrowseStructuredProgram,
} from "@drowse/web-llm";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const [hooks, profiles] = await Promise.all([
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProgram.ts"),
    server.ssrLoadModule("/src/hosted/runtime/structuredHookProfile.ts"),
  ]);
  const standardProfile = profiles.STANDARD_STRUCTURED_HOOK_PROFILE;
  const geometryProfile = profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V2;
  const exactReadoutProfile = profiles.STANDARD_STRUCTURED_HOOK_PROFILE_V3;
  const program = hooks.compileStructuredHookProgram(2, [
    {
      affineGroups: [{
        active: true,
        basis: [[1, 0], [0, 1]],
        neutral: [0, 0],
        target: [1, 0],
        along: 0.5,
        kappa: [0, 1],
      }],
      probes: [
        { kind: "linear", direction: [1, 0], bias: -0.25 },
        { kind: "sae", direction: [0, 1], bias: -0.5 },
      ],
    },
    {
      curves: [{
        active: true,
        basis: [[1, 0], [0, 1]],
        neutral: [0, 0],
        nodeParameters: [[0], [1]],
        rbfWeights: [[0, 0], [0, 0]],
        polynomial: [[0, 0], [1, 0]],
        coordinateOffset: [0],
        coordinateScale: [1],
        origin: [0],
        target: [1],
        axes: [{ periodic: false, period: 1, lowerBound: 0, upperBound: 1 }],
        along: 0.5,
        onto: 0,
      }],
      probes: [{ kind: "linear", direction: [1, 0], bias: 0 }],
    },
  ], standardProfile);
  const state = hooks.initialStructuredHookState(program);
  const affine = hooks.runStructuredHookLayer(program, state, 0, [0.2, 2]);
  assert.ok(Math.abs(affine.residual[0] - 0.7) < 1e-6);
  assert.ok(Math.abs(affine.residual[1] - 1) < 1e-6);
  assert.ok(Math.abs(affine.probes[0] - 0.45) < 1e-6);
  assert.ok(Math.abs(affine.probes[1] - 0.5) < 1e-6);
  console.log("ok 1 - low-rank affine and SAE feature programs share one post-block boundary");

  const curved = hooks.runStructuredHookLayer(
    program,
    state,
    1,
    [0.2, 1],
    { decode: false },
  );
  assert.ok(Math.abs(curved.curveFeet[0] - 0.2) < 1e-5);
  assert.ok(Math.abs(curved.residual[0] - 0.7) < 1e-5);
  assert.ok(Math.abs(curved.residual[1] - 1) < 1e-5);
  assert.ok(Math.abs(curved.probes[0] - 0.7) < 1e-5);
  const next = hooks.runStructuredHookLayer(program, state, 1, [0.3, 1]);
  assert.ok(Math.abs(next.curveFeet[0] - 0.3) < 1e-4);
  assert.ok(Math.abs(next.residual[0] - 0.8) < 1e-4);
  console.log("ok 2 - one-dimensional curved RBF injection carries its prompt foot into decode");

  program.affineActive[0] = 0;
  const gatedOff = hooks.runStructuredHookLayer(program, state, 0, [0.2, 2]);
  assert.deepEqual([...gatedOff.residual], [Math.fround(0.2), 2]);
  assert.throws(
    () => hooks.compileStructuredHookProgram(2, [{
      curves: [{
        active: true,
        basis: [[1, 0]],
        neutral: [0, 0],
        nodeParameters: Array.from({ length: 33 }, (_, index) => [index]),
        rbfWeights: Array.from({ length: 33 }, () => [0]),
        polynomial: [[0], [1]],
        coordinateOffset: [0],
        coordinateScale: [1],
        origin: [0],
        target: [1],
        axes: [{ periodic: false, period: 1, lowerBound: 0, upperBound: 32 }],
        along: 1,
        onto: 0,
      }],
    }], standardProfile),
    /at most 32 nodes|between 2 and 32 nodes/,
  );
  console.log("ok 3 - dynamic gate controls are mutable but malformed geometry fails before upload");

  const saeProbe = hooks.saeFeatureProbe([1, 2], 0.5, [0.25, 0.5]);
  const instrumentProgram = hooks.compileStructuredHookProgram(2, [{
    affineGroups: [hooks.additiveDirectionGroup([3, 4], 0.2)],
    probes: [saeProbe],
  }], standardProfile);
  const instrumentResult = hooks.runStructuredHookLayer(
    instrumentProgram,
    hooks.initialStructuredHookState(instrumentProgram),
    0,
    [1, 2],
  );
  assert.ok(Math.abs(instrumentResult.residual[0] - 1.6) < 1e-6);
  assert.ok(Math.abs(instrumentResult.residual[1] - 2.8) < 1e-6);
  assert.ok(Math.abs(instrumentResult.probes[0] - 6.45) < 1e-6);
  const jumpProbe = hooks.saeFeatureProbe(
    [1, 0],
    0,
    [0, 0],
    2,
    "jump_relu",
    1.5,
  );
  const jumpProgram = hooks.compileStructuredHookProgram(2, [{
    probes: [jumpProbe],
  }], standardProfile);
  const jumpState = hooks.initialStructuredHookState(jumpProgram);
  assert.equal(
    hooks.runStructuredHookLayer(jumpProgram, jumpState, 0, [1, 0]).probes[0],
    0,
  );
  assert.equal(
    hooks.runStructuredHookLayer(jumpProgram, jumpState, 0, [2, 0]).probes[0],
    1,
  );
  assert.deepEqual([...hooks.jLensTokenDirection([2, 3], [1, 2, 4, 5])], [14, 19]);
  const sparseJlens = hooks.compileStructuredHookProgram(2, [
    { probes: [{ kind: "disabled", direction: [0, 0], bias: 0 }] },
    { probes: [{ kind: "jlens", direction: [0, 0], bias: 0 }] },
  ], standardProfile, {
    bindingId: "b".repeat(64),
    layerIndices: new Int32Array([1]),
    tokenIds: new Int32Array(standardProfile.maxProbes),
  });
  assert.equal(sparseJlens.jLensBindingId, "b".repeat(64));
  assert.deepEqual([...sparseJlens.jLensLayerIndices], [1]);
  const wrongSparseMap = structuredClone(sparseJlens);
  wrongSparseMap.jLensLayerIndices = new Int32Array([0]);
  assert.throws(
    () => hooks.validateStructuredHookProgram(wrongSparseMap),
    /layer indices do not match/,
  );
  const duplicateSparseMap = structuredClone(sparseJlens);
  duplicateSparseMap.probeKind[0] = 3;
  duplicateSparseMap.jLensLayerIndices = new Int32Array([1, 1]);
  assert.throws(
    () => hooks.validateStructuredHookProgram(duplicateSparseMap),
    /unique, ordered, and in range/,
  );
  const outOfRangeSparseMap = structuredClone(sparseJlens);
  outOfRangeSparseMap.jLensLayerIndices = new Int32Array([2]);
  assert.throws(
    () => hooks.validateStructuredHookProgram(outOfRangeSparseMap),
    /unique, ordered, and in range/,
  );
  const invalidBinding = structuredClone(sparseJlens);
  invalidBinding.jLensBindingId = "floating";
  assert.throws(
    () => hooks.validateStructuredHookProgram(invalidBinding),
    /wrong types/,
  );
  const productionLayerCount = 28;
  const productionLayerIndices = Int32Array.from(
    { length: productionLayerCount },
    (_, index) => index,
  );
  const productionBindingId = "c".repeat(64);
  const productionJlens = hooks.compileStructuredHookProgram(
    2,
    Array.from({ length: productionLayerCount }, () => ({
      probes: Array.from({ length: exactReadoutProfile.maxProbes }, () => ({
        kind: "jlens",
        direction: [0, 0],
        bias: 0,
      })),
    })),
    exactReadoutProfile,
    {
      bindingId: productionBindingId,
      layerIndices: productionLayerIndices,
      tokenIds: new Int32Array(exactReadoutProfile.maxProbes),
    },
  );
  assert.doesNotThrow(() => hooks.validateStructuredHookProgram(productionJlens));
  assert.doesNotThrow(() => validateDrowseStructuredProgram(
    productionJlens,
    2,
    productionLayerCount,
  ));
  const fullVocabularyJlens = hooks.compileStructuredHookProgram(
    2,
    Array.from({ length: productionLayerCount }, () => ({})),
    exactReadoutProfile,
    {
      bindingId: productionBindingId,
      layerIndices: productionLayerIndices,
      tokenIds: new Int32Array(exactReadoutProfile.maxProbes),
    },
  );
  fullVocabularyJlens.measurementSchema = {
    layerMap: [...productionLayerIndices],
    modelLayerCount: productionLayerCount,
    probes: Array.from({ length: exactReadoutProfile.maxProbes }, () => null),
    lensSource: "fixture",
    saeSource: null,
    lensReadout: true,
  };
  assert.doesNotThrow(() => hooks.validateStructuredHookProgram(fullVocabularyJlens));
  assert.doesNotThrow(() => validateDrowseStructuredProgram(
    fullVocabularyJlens,
    2,
    productionLayerCount,
  ));
  assert.doesNotThrow(() => validateDrowseJlensDictionary({
    hookAbi: "post-block-residual-v4",
    bindingId: productionBindingId,
    hiddenSize: 2,
    layerIndices: productionLayerIndices,
    matrices: Array.from(
      { length: productionLayerCount },
      () => new Float32Array(4),
    ),
  }, 2, productionLayerCount));
  instrumentProgram.controls = {
    affine: [
      {
        enabled: true,
        phase: { kind: "generated_only" },
        gate: { slots: [{ layer: 0, probe: 0 }], operator: ">", threshold: 1 },
      },
      null,
      null,
      null,
    ],
    curve: [null, null, null, null],
  };
  assert.equal(hooks.structuredHookControlsFor(instrumentProgram, {
    prefill: true,
    thinking: false,
    generatedTokens: 0,
  }).affineActive[0], 0);
  assert.equal(hooks.structuredHookControlsFor(instrumentProgram, {
    prefill: false,
    thinking: false,
    generatedTokens: 1,
    priorMeasurements: instrumentResult.probes,
  }).affineActive[0], 1);
  console.log("ok 4 - exact SAE feature and J-lens direction math feed prior-step GPU gates");

  const multiCurve = hooks.compileStructuredHookProgram(4, [{
    curves: [
      {
        active: true,
        basis: [[1, 0, 0, 0], [0, 1, 0, 0]],
        neutral: [0, 0, 0, 0],
        nodeParameters: [[0, 0], [1, 1]],
        rbfWeights: [[0, 0], [0, 0]],
        polynomial: [[0, 0], [1, 0], [0, 1]],
        coordinateOffset: [0, 0],
        coordinateScale: [1, 1],
        origin: [0, 0],
        target: [0.5, 0.25],
        axes: [
          { periodic: false, period: 1, lowerBound: -2, upperBound: 2 },
          { periodic: false, period: 1, lowerBound: -2, upperBound: 2 },
        ],
        along: 1,
        onto: 0,
      },
      {
        active: true,
        basis: [[0, 0, 1, 0], [0, 0, 0, 1]],
        neutral: [0, 0, 0, 0],
        nodeParameters: [[1, 0], [-1, 0]],
        rbfWeights: [[0, 0], [0, 0]],
        polynomial: [[0, 0], [1, 0], [0, 1]],
        coordinateOffset: [0, 0],
        coordinateScale: [1, 1],
        origin: [0.9],
        target: [0.1],
        axes: [{ periodic: true, period: 1, lowerBound: 0, upperBound: 1 }],
        along: 1,
        onto: 0,
      },
    ],
  }], standardProfile);
  const angle = 2 * Math.PI * 0.95;
  const combined = hooks.runStructuredHookLayer(
    multiCurve,
    hooks.initialStructuredHookState(multiCurve),
    0,
    [0.1, 0.2, Math.cos(angle), Math.sin(angle)],
    { decode: false },
  );
  assert.ok(Math.abs(combined.curveFeet[0] - 0.1) < 1e-3);
  assert.ok(Math.abs(combined.curveFeet[1] - 0.2) < 1e-3);
  assert.ok(Math.abs(combined.curveFeet[4] - 0.95) < 1e-3);
  assert.ok(Math.abs(combined.residual[0] - 0.6) < 2e-3);
  assert.ok(Math.abs(combined.residual[1] - 0.45) < 2e-3);
  assert.ok(Math.abs(combined.residual[2] - Math.cos(2 * Math.PI * 0.15)) < 2e-3);
  assert.ok(Math.abs(combined.residual[3] - Math.sin(2 * Math.PI * 0.15)) < 2e-3);
  console.log("ok 5 - multidimensional and periodic curves compose without fixed scalar feet");

  const sphereCurve = {
    active: true,
    domainKind: "sphere",
    basis: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    neutral: [0, 0, 0],
    nodeParameters: [[0, 1, 0], [0, 0, 1]],
    rbfWeights: [[0, 0, 0], [0, 0, 0]],
    polynomial: [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    coordinateOffset: [0, 0, 0],
    coordinateScale: [1, 1, 1],
    origin: [Math.PI / 2, 0],
    target: [Math.PI / 2, Math.PI / 2],
    axes: [
      { periodic: false, period: 2 * Math.PI, lowerBound: 0, upperBound: Math.PI },
      { periodic: true, period: 2 * Math.PI, lowerBound: 0, upperBound: 2 * Math.PI },
    ],
    along: 1,
    onto: 0,
  };
  const sphereProgram = hooks.compileStructuredHookProgram(3, [{
    curves: [sphereCurve],
    geometryWhitener: {
      rank: 1,
      ridge: 1,
      basis: [[1, 0, 0]],
      correction: [0],
    },
    geometryProbes: [{
      active: true,
      mean: [0, 0, 0],
      inverseMean: [0, 0, 0],
      basis: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      gramInverse: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      cholesky: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      candidateWhite: [[0, 1, 0], [0, 0, 1]],
      coordinateMap: [[0, 0, 0], [0, 0, 0]],
      coordinateBias: [0, 0],
      curve: sphereCurve,
      curveNodeCoordinates: [[Math.PI / 2, 0], [Math.PI / 2, Math.PI / 2]],
      curveNodeValues: [[0, 1, 0], [0, 0, 1]],
      foot: [Math.PI / 2, 0],
    }],
  }], geometryProfile);
  assert.equal(sphereProgram.curveDomainKind[0], 2);
  assert.equal(sphereProgram.geometryDomainKind[0], 2);
  assert.equal(sphereProgram.curveEmbedDim[0], 3);
  const geometryCurveStride = profiles.structuredCurveParameterStride(geometryProfile);
  const geometryAxisPeriodOffset = 2 +
    geometryProfile.maxCurveNodes * geometryProfile.maxEmbedDim +
    geometryProfile.maxCurveNodes * geometryProfile.maxRank +
    (geometryProfile.maxEmbedDim + 1) * geometryProfile.maxRank +
    geometryProfile.maxEmbedDim * 2 +
    geometryProfile.maxIntrinsicDim * 2 +
    2 +
    geometryProfile.maxIntrinsicDim * 3;
  const geometryCoordinateScaleOffset = 2 +
    geometryProfile.maxCurveNodes * geometryProfile.maxEmbedDim +
    geometryProfile.maxCurveNodes * geometryProfile.maxRank +
    (geometryProfile.maxEmbedDim + 1) * geometryProfile.maxRank +
    geometryProfile.maxEmbedDim;
  assert.equal(geometryCurveStride, 671);
  const activeGeometryPeriods = sphereProgram.geometryCurveParameters.slice(
    geometryAxisPeriodOffset,
    geometryAxisPeriodOffset + geometryProfile.maxIntrinsicDim,
  );
  assert.ok(Math.abs(activeGeometryPeriods[0] - 2 * Math.PI) < 1e-6);
  assert.ok(Math.abs(activeGeometryPeriods[1] - 2 * Math.PI) < 1e-6);
  assert.deepEqual(Array.from(activeGeometryPeriods.slice(2)), [1, 1]);
  const inactiveGeometrySlot = geometryCurveStride;
  assert.deepEqual(
    Array.from(sphereProgram.geometryCurveParameters.slice(
      inactiveGeometrySlot + geometryAxisPeriodOffset,
      inactiveGeometrySlot + geometryAxisPeriodOffset + geometryProfile.maxIntrinsicDim,
    )),
    Array(geometryProfile.maxIntrinsicDim).fill(1),
  );
  sphereProgram.geometryCurveParameters[geometryAxisPeriodOffset + 3] = 0;
  assert.throws(
    () => hooks.validateStructuredHookProgram(sphereProgram),
    /periods must remain positive/,
  );
  sphereProgram.geometryCurveParameters[geometryAxisPeriodOffset + 3] = 1;
  sphereProgram.geometryCurveParameters[
    geometryCoordinateScaleOffset + geometryProfile.maxEmbedDim - 1
  ] = 0;
  assert.throws(
    () => hooks.validateStructuredHookProgram(sphereProgram),
    /coordinate scales must remain positive/,
  );
  sphereProgram.geometryCurveParameters[
    geometryCoordinateScaleOffset + geometryProfile.maxEmbedDim - 1
  ] = 1;
  hooks.validateStructuredHookProgram(sphereProgram);
  const sphereResult = hooks.runStructuredHookLayer(
    sphereProgram,
    hooks.initialStructuredHookState(sphereProgram),
    0,
    [0, 1, 0],
    { decode: false },
  );
  assert.ok(Math.abs(sphereResult.residual[0]) < 2e-4);
  assert.ok(Math.abs(sphereResult.residual[1]) < 2e-4);
  assert.ok(Math.abs(sphereResult.residual[2] - 1) < 2e-4);
  assert.ok(Math.abs(sphereResult.geometry[4] - Math.PI / 2) < 2e-3);
  assert.ok(Math.abs(sphereResult.geometry[5] - Math.PI / 2) < 2e-3);
  console.log("ok 6 - sphere geometry and fixed-width GPU denominator lanes remain finite");

  const requirements = profiles.structuredHookMemoryRequirements(
    standardProfile,
    2,
    960,
  );
  assert.equal(requirements.curveParameterStride, 671);
  assert.equal(requirements.hostProgramBytes, 636_704);
  assert.equal(requirements.gpuInputBytes, 636_736);
  assert.equal(requirements.maxBufferSize, 245_760);
  assert.equal(requirements.maxStorageBufferBindingSize, 245_760);
  assert.equal(requirements.maxStorageBuffersPerShaderStage, 8);
  assert.equal(requirements.maxComputeWorkgroupStorageSize, 32_768);
  const geometryRequirements = profiles.structuredHookMemoryRequirements(
    geometryProfile,
    2,
    960,
  );
  assert.equal(geometryRequirements.hostProgramBytes, 2_084_784);
  assert.equal(geometryRequirements.gpuInputBytes, 2_084_816);
  assert.equal(geometryRequirements.maxBufferSize, 737_280);
  assert.equal(geometryRequirements.geometryHeaderBytes, 448);
  assert.equal(geometryRequirements.geometryPayloadBytes, 709_312);
  assert.equal(geometryRequirements.maxStorageBuffersPerShaderStage, 8);
  const payloadDominantRequirements = profiles.structuredHookMemoryRequirements(
    geometryProfile,
    1,
    256,
  );
  assert.equal(
    payloadDominantRequirements.maxBufferSize,
    payloadDominantRequirements.geometryPayloadBytes,
  );
  assert.equal(
    payloadDominantRequirements.maxStorageBufferBindingSize,
    payloadDominantRequirements.geometryPayloadBytes,
  );
  const geometryLayout = profiles.structuredGeometryPayloadLayout(
    geometryProfile,
    2,
    960,
  );
  assert.equal(geometryLayout.elements, 177_328);
  assert.deepEqual(geometryLayout.offsets, [
    0,
    15_360,
    30_720,
    153_600,
    154_624,
    155_648,
    159_872,
    160_384,
    160_448,
    171_184,
    173_232,
  ]);
  assert.ok(geometryLayout.offsets.every((offset) => (offset * 4) % 4 === 0));
  assert.ok(geometryRequirements.geometryHeaderBytes <= geometryRequirements.maxBufferSize);
  assert.ok(geometryRequirements.geometryPayloadBytes <= geometryRequirements.maxBufferSize);
  assert.ok(geometryRequirements.hostProgramBytes > requirements.hostProgramBytes);
  const jlensRequirements = profiles.structuredHookMemoryRequirements(
    geometryProfile,
    2,
    960,
    { includeJlens: true },
  );
  assert.equal(jlensRequirements.hostProgramBytes, 9_457_616);
  assert.equal(jlensRequirements.gpuInputBytes, 5_771_252);
  assert.equal(jlensRequirements.maxBufferSize, 3_686_400);
  assert.ok(jlensRequirements.maxBufferSize < 2 * 960 * 960 * 4);
  assert.deepEqual(
    profiles.structuredHookMemoryRequirements(exactReadoutProfile, 2, 960),
    profiles.structuredHookMemoryRequirements(geometryProfile, 2, 960),
  );
  assert.equal(profiles.resolveStructuredHookProfile("standard-v3"), exactReadoutProfile);
  assert.throws(
    () => profiles.validateStructuredHookProfile({ ...standardProfile, maxRank: 9 }),
    /modified/,
  );
  assert.throws(
    () => profiles.checkedStructuredHookProduct("fixture", Number.MAX_SAFE_INTEGER, 2),
    /safe integer arithmetic/,
  );
  console.log("ok 7 - allow-listed profiles calculate checked host and WebGPU requirements");
  console.log("1..7");
} finally {
  await server.close();
}
