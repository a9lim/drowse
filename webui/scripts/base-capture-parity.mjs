export const CAPTURE_TOLERANCES = Object.freeze({
  strict: Object.freeze({ maxAbsolute: 1e-5, maxRelativeL2: null }),
  fp32: Object.freeze({ maxAbsolute: 1e-4, maxRelativeL2: 1e-5 }),
});

export function compareCaptures(reference, actual, policy = "strict") {
  const tolerance = CAPTURE_TOLERANCES[policy];
  if (!tolerance) throw new Error("Unknown capture tolerance policy");
  for (const field of ["layerCount", "positionCount", "hiddenSize"]) {
    if (!Number.isSafeInteger(reference[field]) || reference[field] <= 0 || actual[field] !== reference[field]) {
      throw new Error("Capture shapes differ or are invalid");
    }
  }
  const width = reference.positionCount * reference.hiddenSize;
  const length = reference.layerCount * width;
  if (reference.values.length !== length || actual.values.length !== length) {
    throw new Error("Capture values do not match their shape");
  }
  const layers = Array.from({ length: reference.layerCount }, (_, layer) => {
    let squaredError = 0, squaredReference = 0, maxAbsolute = 0, maxReference = 0;
    for (let index = layer * width; index < (layer + 1) * width; index++) {
      const expected = reference.values[index], observed = actual.values[index];
      if (!Number.isFinite(expected) || !Number.isFinite(observed)) {
        throw new Error("Capture contains non-finite values");
      }
      const error = observed - expected;
      squaredError += error * error;
      squaredReference += expected * expected;
      maxAbsolute = Math.max(maxAbsolute, Math.abs(error));
      maxReference = Math.max(maxReference, Math.abs(expected));
    }
    const relativeL2 = Math.sqrt(squaredError / Math.max(squaredReference, Number.MIN_VALUE));
    return { layer, maxAbsolute, maxReference, relativeL2 };
  });
  return {
    policy, tolerance, layers,
    maxAbsolute: Math.max(...layers.map(layer => layer.maxAbsolute)),
    maxRelativeL2: Math.max(...layers.map(layer => layer.relativeL2)),
    passed: layers.every(layer => layer.maxAbsolute <= tolerance.maxAbsolute &&
      (tolerance.maxRelativeL2 === null || layer.relativeL2 <= tolerance.maxRelativeL2)),
  };
}
