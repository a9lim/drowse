import type { SerializedMahalanobisWhitener } from "./workerContracts";

const NEAR_ZERO = 1e-12;

export function applySerializedInverse(
  whitener: SerializedMahalanobisWhitener,
  values: ArrayLike<number>,
  rows = 1,
): Float64Array {
  if (values.length !== rows * whitener.columns) {
    throw new TypeError("Mahalanobis input does not match the whitener dimensions");
  }
  const output = Float64Array.from(values, (value) => value / whitener.ridge);
  for (let row = 0; row < rows; row += 1) {
    for (let component = 0; component < whitener.rank; component += 1) {
      let coordinate = 0;
      for (let column = 0; column < whitener.columns; column += 1) {
        coordinate += values[row * whitener.columns + column] *
          whitener.basis[component * whitener.columns + column];
      }
      const correction = coordinate * (
        1 / (whitener.eigenvalues[component] + whitener.ridge) - 1 / whitener.ridge
      );
      for (let column = 0; column < whitener.columns; column += 1) {
        output[row * whitener.columns + column] += correction *
          whitener.basis[component * whitener.columns + column];
      }
    }
  }
  return output;
}

export function mahalanobisDot(
  whitener: SerializedMahalanobisWhitener,
  left: ArrayLike<number>,
  right: ArrayLike<number>,
): number {
  const inverseRight = applySerializedInverse(whitener, right);
  let result = 0;
  for (let index = 0; index < whitener.columns; index += 1) {
    result += left[index] * inverseRight[index];
  }
  return result;
}

export function mahalanobisNorm(
  whitener: SerializedMahalanobisWhitener,
  value: ArrayLike<number>,
): number {
  return Math.sqrt(Math.max(mahalanobisDot(whitener, value, value), 0));
}

export function leaceProject(
  whitener: SerializedMahalanobisWhitener,
  base: ArrayLike<number>,
  onto: ArrayLike<number>,
  operator: "~" | "|",
): Float64Array {
  const denominator = mahalanobisDot(whitener, onto, onto);
  if (denominator < NEAR_ZERO) {
    return operator === "|"
      ? Float64Array.from(base)
      : new Float64Array(whitener.columns);
  }
  const coefficient = mahalanobisDot(whitener, base, onto) / denominator;
  return Float64Array.from(base, (value, index) => {
    const projected = coefficient * onto[index];
    return operator === "~" ? projected : value - projected;
  });
}
