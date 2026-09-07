import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import init, {
  MahalanobisWhitener,
  RbfFitPlan,
  fitAffineFisher,
  fitMahalanobisWhitener,
  detectPeriodicTopology,
  selectTopologyFromTargets,
} from "../public-hosted/wasm/drowse_fitting_wasm.js";

const releaseEvidenceType = releaseEvidenceArgument(process.argv.slice(2));
const observed = {
  neutralCenteringMaxAbsoluteError: 0,
  whiteningMaxAbsoluteError: 0,
  pcaProjectionMaxError: 0,
  spectralProjectionMaxError: 0,
  rbfMaxAbsoluteError: 0,
  geometryMaxAbsoluteError: 0,
  maxEvaluatedOutputError: 0,
};

const fixture = JSON.parse(await readFile(
  new URL("../../browser-runtime/fixtures/affine-fisher-fit-v1.json", import.meta.url),
));
const topologyFixture = JSON.parse(await readFile(
  new URL("../../browser-runtime/fixtures/topology-rbf-parity-v1.json", import.meta.url),
));
const wasm = await readFile(new URL(
  "../public-hosted/wasm/drowse_fitting_wasm_bg.wasm",
  import.meta.url,
));
assert.doesNotMatch(wasm.toString("latin1"), /\/(?:Users|home)\/[^/\x00]+\//,
  "shipped WebAssembly must not embed personal home directories");
await init({ module_or_path: wasm });

const neutral = new Float64Array(fixture.neutral);
const centroids = new Float64Array(fixture.centroids);
const whitener = fitMahalanobisWhitener(
  neutral,
  fixture.neutralRows,
  fixture.columns,
  fixture.ridgeScale,
);
try {
  observed.neutralCenteringMaxAbsoluteError = maxAbsoluteDifference(
    whitener.mean(),
    fixture.expected.neutralMean,
  );
  observed.whiteningMaxAbsoluteError = Math.max(
    maxAbsoluteDifference(whitener.mean(), fixture.serializedWhitener.mean),
    maxAbsoluteDifference(whitener.basis(), fixture.serializedWhitener.basis),
    maxAbsoluteDifference(
      whitener.eigenvalues(),
      fixture.serializedWhitener.eigenvalues,
    ),
    maxAbsoluteDifference(
      whitener.inverseScales(),
      fixture.serializedWhitener.inverseScales,
    ),
    Math.abs(whitener.ridge - fixture.serializedWhitener.ridge),
  );
  assert.ok(observed.neutralCenteringMaxAbsoluteError <= 1e-4);
  assert.ok(observed.whiteningMaxAbsoluteError <= 1e-4);
  const serialized = new MahalanobisWhitener(
    whitener.columns,
    whitener.rank,
    whitener.mean(),
    whitener.basis(),
    whitener.eigenvalues(),
    whitener.inverseScales(),
    whitener.ridge,
  );
  try {
    const fit = fitAffineFisher(
      centroids,
      fixture.nodeCount,
      fixture.columns,
      serialized,
      fixture.maxComponents,
      fixture.orientTo,
    );
    try {
      assert.equal(fit.nodeCount, fixture.nodeCount);
      assert.equal(fit.components, fixture.maxComponents);
      close(fit.mean(), fixture.expected.mean, 3e-6);
      close(fit.basis(), fixture.expected.basis, 3e-6);
      close(fit.nodeCoordinates(), fixture.expected.nodeCoordinates, 5e-6);
      close(fit.muCoordinates(), fixture.expected.muCoordinates, 5e-6);
      close(fit.whitenedGram(), fixture.expected.whitenedGram, 5e-5);
      close(fit.neutralCrossGram(), fixture.expected.neutralCrossGram, 5e-6);
      observed.pcaProjectionMaxError = Math.max(
        maxAbsoluteDifference(fit.mean(), fixture.expected.mean),
        maxAbsoluteDifference(fit.basis(), fixture.expected.basis),
        maxAbsoluteDifference(
          fit.nodeCoordinates(),
          fixture.expected.nodeCoordinates,
        ),
        maxAbsoluteDifference(fit.muCoordinates(), fixture.expected.muCoordinates),
      );
      observed.geometryMaxAbsoluteError = Math.max(
        maxAbsoluteDifference(fit.whitenedGram(), fixture.expected.whitenedGram),
        maxAbsoluteDifference(
          fit.neutralCrossGram(),
          fixture.expected.neutralCrossGram,
        ),
      );
      close(
        [fit.explainedVariance, fit.mahalanobisShare],
        [fixture.expected.explainedVariance, fixture.expected.mahalanobisShare],
        5e-6,
      );
      const muCoordinates = fit.muCoordinates();
      const scalarTarget = new Float64Array(fit.nodeCount);
      for (let node = 0; node < fit.nodeCount; node += 1) {
        scalarTarget[node] = muCoordinates[node * fit.components];
      }
      const targets = new Float64Array(muCoordinates.length + scalarTarget.length);
      targets.set(muCoordinates);
      targets.set(scalarTarget, muCoordinates.length);
      const topology = selectTopologyFromTargets(
        fit.whitenedGram(),
        fit.nodeCount,
        targets,
        new Uint32Array([0, muCoordinates.length, targets.length]),
        2,
        0.7,
        "auto",
        undefined,
        undefined,
        undefined,
        0.5,
      );
      try {
        assert.equal(topology.winnerName, "flat-pca");
        assert.equal(topology.diagnosticsKind, "pca");
        assert.equal(topology.diagnosticsPickedDimensions, topology.intrinsicDimensions);
        assert.equal(topology.diagnosticsThreshold, 0.7);
        const variance = topology.diagnosticsPerComponentVariance();
        const cumulative = topology.diagnosticsCumulativeVariance();
        assert.equal(variance.length, 2);
        assert.equal(cumulative.length, variance.length);
        close(cumulative, [variance[0], variance[0] + variance[1]], 5e-8);
      } finally {
        topology.free();
      }
      const forcedPca = selectTopologyFromTargets(
        fit.whitenedGram(),
        fit.nodeCount,
        targets,
        new Uint32Array([0, muCoordinates.length, targets.length]),
        2,
        0.7,
        "pca",
        undefined,
        undefined,
        undefined,
        0.5,
      );
      try {
        assert.equal(forcedPca.winnerName, "flat-pca");
        assert.equal(forcedPca.fitMode, "pca");
      } finally {
        forcedPca.free();
      }
    } finally {
      fit.free();
    }
  } finally {
    serialized.free();
  }
} finally {
  whitener.free();
}

const nodeCount = topologyFixture.nodeCount;
const curveTargets = new Float64Array(topologyFixture.targets);
const curveSelection = topologyFixture.selection;
const curveExpected = topologyFixture.expected;
const curveGram = centeredGram(
  curveTargets,
  nodeCount,
  topologyFixture.outputDimensions,
);
const curved = selectTopologyFromTargets(
  curveGram,
  nodeCount,
  curveTargets,
  new Uint32Array([0, curveTargets.length]),
  curveSelection.maxDimensions,
  curveSelection.varianceThreshold,
  curveSelection.fitMode,
  curveSelection.minDimensions,
  undefined,
  undefined,
  curveSelection.persistenceFraction,
  curveSelection.smoothing,
);
try {
  assert.equal(curved.winnerName, curveExpected.winnerName);
  assert.equal(curved.intrinsicDimensions, curveExpected.intrinsicDimensions);
  assert.equal(curved.diagnosticsKind, "spectral");
  assert.equal(curved.diagnosticsPickedDimensions, curved.intrinsicDimensions);
  assert.equal(curved.diagnosticsKNn, curveExpected.kNN);
  assert.equal(curved.diagnosticsMinDimensions, curveSelection.minDimensions);
  assert.equal(curved.diagnosticsPinned, false);
  close(
    curved.diagnosticsEigenvalues(),
    curveExpected.eigenvalues,
    2e-6,
  );
  assert.ok(
    Math.abs(curved.diagnosticsBandwidth - curveExpected.bandwidth) <= 2e-6,
  );
  assert.ok(
    Math.abs(curved.diagnosticsGapMagnitude - curveExpected.gapMagnitude) <= 2e-6,
  );
  const coordinates = curved.coordinates();
  observed.spectralProjectionMaxError = projectionMaxError(
    coordinates,
    curveExpected.coordinates,
    nodeCount,
    curved.intrinsicDimensions,
  );
  assert.ok(observed.spectralProjectionMaxError <= 2e-6);
  for (const pair of curveExpected.pairDistances) {
    assert.ok(
      Math.abs(
        coordinateDistance(
          coordinates,
          pair.left,
          pair.right,
          curved.intrinsicDimensions,
        ) - pair.distance
      ) <= 2e-6,
    );
  }

  const winnerPlan = curved.winnerPlan();
  assert.ok(winnerPlan);
  try {
    const restoredPlan = new RbfFitPlan(
      winnerPlan.inputDimensions,
      winnerPlan.nodeCount,
      winnerPlan.nodes(),
      winnerPlan.coordinateOffset(),
      winnerPlan.coordinateScale(),
      winnerPlan.kernel(),
      winnerPlan.kernelScale,
      winnerPlan.lambdas(),
      winnerPlan.spectralBasis(),
      winnerPlan.residualRatios(),
      winnerPlan.residualTraces(),
    );
    try {
      for (const [values, outputDimensions, pythonExpected] of [
        [curveTargets, topologyFixture.outputDimensions, curveExpected.evaluatedTargets],
        [Float64Array.from({ length: nodeCount }, (_, row) => curveTargets[2 * row]), 1],
      ]) {
        const original = winnerPlan.fitSmoothed(
          values,
          outputDimensions,
          curveSelection.smoothing,
        );
        const restored = restoredPlan.fitSmoothed(
          values,
          outputDimensions,
          curveSelection.smoothing,
        );
        try {
          assert.equal(original.outputDimensions, outputDimensions);
          const restoredValues = restored.evaluate(
            curved.embeddedCoordinates(),
            nodeCount,
          );
          const originalValues = original.evaluate(
            curved.embeddedCoordinates(),
            nodeCount,
          );
          const restorationError = maxAbsoluteDifference(
            restoredValues,
            originalValues,
          );
          if (pythonExpected !== undefined) {
            const parityError = maxAbsoluteDifference(
              originalValues,
              pythonExpected,
            );
            observed.rbfMaxAbsoluteError = Math.max(
              observed.rbfMaxAbsoluteError,
              parityError,
            );
            observed.maxEvaluatedOutputError = Math.max(
              observed.maxEvaluatedOutputError,
              parityError,
            );
            assert.ok(parityError <= 2e-6);
            assert.ok(
              Math.abs(original.lambda - curveExpected.lambda) <= 2e-6,
            );
            assert.ok(
              Math.abs(
                original.effectiveDegreesOfFreedom -
                curveExpected.effectiveDegreesOfFreedom
              ) <= 2e-5,
            );
          }
          assert.ok(restorationError <= 1e-12);
          close(restoredValues, originalValues, 1e-12);
        } finally {
          original.free();
          restored.free();
        }
      }
    } finally {
      restoredPlan.free();
    }
  } finally {
    winnerPlan.free();
  }
} finally {
  curved.free();
}

const automaticCurveTargets = new Float64Array(40 * 3);
for (let index = 0; index < 40; index += 1) {
  const t = index / 39;
  automaticCurveTargets[3 * index] = Math.sin(5 * t);
  automaticCurveTargets[3 * index + 1] = Math.cos(7 * t);
  automaticCurveTargets[3 * index + 2] = Math.sin(11 * t);
}
const automaticCurve = selectTopologyFromTargets(
  centeredGram(automaticCurveTargets, 40, 3),
  40,
  automaticCurveTargets,
  new Uint32Array([0, automaticCurveTargets.length]),
  6,
  0.7,
  "auto",
  undefined,
  undefined,
  undefined,
  0.5,
);
try {
  assert.equal(automaticCurve.winnerName, "spectral");
} finally {
  automaticCurve.free();
}

const circle = new Float64Array(40 * 2);
for (let index = 0; index < 40; index += 1) {
  const angle = 2 * Math.PI * index / 40;
  circle[2 * index] = Math.cos(angle);
  circle[2 * index + 1] = Math.sin(angle);
}
const periodic = selectTopologyFromTargets(
  centeredGram(circle, 40, 2),
  40,
  circle,
  new Uint32Array([0, circle.length]),
  6,
  0.7,
  "auto",
  undefined,
  undefined,
  undefined,
  0.5,
);
try {
  assert.equal(periodic.winnerName, "torus-T1");
  assert.equal(periodic.periodicDimensions, 1);
} finally {
  periodic.free();
}

const thinTorus = [];
const mobius = [];
const sphere = [];
for (let i = 0; i < 16; i++) {
  for (let j = 0; j < 8; j++) {
    const u = 2 * Math.PI * i / 16, v = 2 * Math.PI * j / 8;
    thinTorus.push([Math.cos(u), Math.sin(u), 0.3 * Math.cos(v), 0.3 * Math.sin(v)]);
  }
}
for (let i = 0; i < 24; i++) {
  for (let j = 0; j < 5; j++) {
    const u = 2 * Math.PI * i / 24, v = 0.1 * (j / 2 - 1);
    mobius.push([(1 + v * Math.cos(u / 2)) * Math.cos(u),
      (1 + v * Math.cos(u / 2)) * Math.sin(u), v * Math.sin(u / 2)]);
  }
}
for (let i = 0; i < 80; i++) {
  const z = 1 - 2 * (i + 0.5) / 80, theta = i * Math.PI * (3 - Math.sqrt(5));
  sphere.push([Math.sqrt(1 - z * z) * Math.cos(theta), Math.sqrt(1 - z * z) * Math.sin(theta), z]);
}
for (const [name, points, expected] of [["thin torus", thinTorus, 2], ["Mobius strip", mobius, 0], ["sphere", sphere, 0]]) {
  for (const rows of [points, [...points].reverse()]) {
    const chart = detectPeriodicTopology(centeredGram(new Float64Array(rows.flat()), rows.length, rows[0].length),
      rows.length, 6, undefined, undefined, 0.5);
    try {
      assert.equal(chart?.dimensions ?? 0, expected, `${name}: wrong periodic chart`);
    } finally {
      chart?.free();
    }
  }
}

const unrelatedTargets = Float64Array.from({ length: 40 }, (_, i) => i % 2 ? -1 : 1);
const unsupportedPeriodicFit = selectTopologyFromTargets(
  centeredGram(circle, 40, 2), 40, unrelatedTargets, new Uint32Array([0, 40]),
  6, 0.7, "auto", undefined, undefined, undefined, 0.5,
);
try {
  assert.notEqual(unsupportedPeriodicFit.winnerName, "torus-T1", "poor periodic predictions must not override the baseline");
} finally {
  unsupportedPeriodicFit.free();
}

if (releaseEvidenceType === null) {
  console.log("browser fitting WASM goldens passed");
} else {
  const results = releaseEvidenceType === "fittingKernelParity"
    ? {
        neutralCenteringMaxAbsoluteError:
          observed.neutralCenteringMaxAbsoluteError,
        whiteningMaxAbsoluteError: observed.whiteningMaxAbsoluteError,
        pcaProjectionMaxError: observed.pcaProjectionMaxError,
        spectralProjectionMaxError: observed.spectralProjectionMaxError,
        rbfMaxAbsoluteError: observed.rbfMaxAbsoluteError,
        geometryMaxAbsoluteError: observed.geometryMaxAbsoluteError,
        finiteOutputsPassed: Object.values(observed).every(Number.isFinite),
      }
    : {
        flatSelectionPassed: true,
        curvedSelectionPassed: true,
        periodicSelectionPassed: true,
        automaticSelectionPassed: true,
        maxEvaluatedOutputError: observed.maxEvaluatedOutputError,
      };
  process.stdout.write(`${JSON.stringify({
    schemaVersion: 1,
    evidenceType: releaseEvidenceType,
    producedAt: new Date().toISOString(),
    passed: true,
    fixturePath: releaseEvidenceType === "fittingKernelParity"
      ? "browser-runtime/fixtures/affine-fisher-fit-v1.json"
      : "browser-runtime/fixtures/topology-rbf-parity-v1.json",
    results,
  })}\n`);
}

function close(actual, expected, tolerance) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(
      Math.abs(actual[index] - expected[index]) <= tolerance,
      `${index}: ${actual[index]} != ${expected[index]} within ${tolerance}`,
    );
  }
}

function maxAbsoluteDifference(actual, expected) {
  assert.equal(actual.length, expected.length);
  let maximum = 0;
  for (let index = 0; index < actual.length; index += 1) {
    maximum = Math.max(maximum, Math.abs(actual[index] - expected[index]));
  }
  return maximum;
}

function projectionMaxError(actual, expected, rows, dimensions) {
  let maximum = 0;
  for (let left = 0; left < rows; left += 1) {
    for (let right = 0; right < rows; right += 1) {
      let actualDot = 0;
      let expectedDot = 0;
      for (let dimension = 0; dimension < dimensions; dimension += 1) {
        actualDot +=
          actual[left * dimensions + dimension] *
          actual[right * dimensions + dimension];
        expectedDot +=
          expected[left * dimensions + dimension] *
          expected[right * dimensions + dimension];
      }
      maximum = Math.max(maximum, Math.abs(actualDot - expectedDot));
    }
  }
  return maximum;
}

function coordinateDistance(values, left, right, dimensions) {
  let squared = 0;
  for (let dimension = 0; dimension < dimensions; dimension += 1) {
    const delta =
      values[left * dimensions + dimension] -
      values[right * dimensions + dimension];
    squared += delta * delta;
  }
  return Math.sqrt(squared);
}

function releaseEvidenceArgument(args) {
  if (args.length === 0) return null;
  if (
    args.length !== 2 ||
    args[0] !== "--release-evidence" ||
    !["fittingKernelParity", "topologyOrchestration"].includes(args[1])
  ) {
    throw new Error(
      "usage: fitting-wasm-browser.test.mjs [--release-evidence fittingKernelParity|topologyOrchestration]",
    );
  }
  return args[1];
}

function centeredGram(values, rows, columns) {
  const means = new Float64Array(columns);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      means[column] += values[row * columns + column] / rows;
    }
  }
  const gram = new Float64Array(rows * rows);
  for (let left = 0; left < rows; left += 1) {
    for (let right = 0; right < rows; right += 1) {
      for (let column = 0; column < columns; column += 1) {
        gram[left * rows + right] +=
          (values[left * columns + column] - means[column]) *
          (values[right * columns + column] - means[column]);
      }
    }
  }
  return gram;
}
