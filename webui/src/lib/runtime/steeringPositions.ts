export interface SteeringPositionSelection {
  position_spans?: readonly (readonly [number, number])[] | null;
  position_indices?: readonly number[] | null;
}

export type SteeringPositionSpan = [number, number];

// Kept in parity with the fork's normalizeDrowsePositionSelection without
// eagerly loading WebLLM in session persistence or the main UI thread.
export function normalizeSteeringPositions(
  selection: SteeringPositionSelection,
): SteeringPositionSpan[] | undefined {
  const { position_spans: spans, position_indices: indices } = selection;
  if (spans != null && indices != null) {
    throw invalid("position_spans and position_indices are mutually exclusive");
  }
  const value = spans ?? indices;
  if (value == null) return undefined;
  if (!Array.isArray(value) || value.length > 4096) {
    throw invalid("Steering positions must be an array of at most 4096 entries");
  }
  const ranges: SteeringPositionSpan[] = [];
  for (const item of value) {
    const pair = spans != null ? item : [item, (item as number) + 1];
    if (!Array.isArray(pair) || pair.length !== 2 ||
      !Number.isSafeInteger(pair[0]) || !Number.isSafeInteger(pair[1]) ||
      pair[0] < 0 || pair[1] <= pair[0]) {
      throw invalid("Steering positions must be non-negative integers with start < end");
    }
    ranges.push([pair[0], pair[1]]);
  }
  ranges.sort((left, right) => left[0] - right[0]);
  const merged: SteeringPositionSpan[] = [];
  for (const [start, end] of ranges) {
    const previous = merged[merged.length - 1];
    if (previous && start < previous[1]) {
      throw invalid("Steering positions must not overlap or repeat");
    }
    if (previous && start === previous[1]) previous[1] = end;
    else merged.push([start, end]);
  }
  return merged;
}

function invalid(message: string): Error {
  return Object.assign(new TypeError(message), { code: "INVALID_STEERING_POSITIONS", status: 400 });
}
