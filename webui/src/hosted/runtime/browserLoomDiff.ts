import type {
  DiffReadingDeltaJSON,
  DiffTextSpanJSON,
  DiffTokenSpanJSON,
  LoomTokenRowJSON,
} from "../../lib/types";

interface Match {
  a: number;
  b: number;
  size: number;
}

export function browserTextDiff(left: string, right: string): DiffTextSpanJSON[] {
  const a = splitWords(left);
  const b = splitWords(right);
  const spans: DiffTextSpanJSON[] = [];
  for (const [tag, aStart, aEnd, bStart, bEnd] of sequenceOpcodes(a, b)) {
    if (tag === "equal" || tag === "delete") {
      appendTextSpan(spans, tag, a.slice(aStart, aEnd));
    } else if (tag === "insert") {
      appendTextSpan(spans, tag, b.slice(bStart, bEnd));
    } else {
      appendTextSpan(spans, "delete", a.slice(aStart, aEnd));
      appendTextSpan(spans, "insert", b.slice(bStart, bEnd));
    }
  }
  return spans;
}

export function browserReadingsDiff(
  left: Readonly<Record<string, number>>,
  right: Readonly<Record<string, number>>,
): DiffReadingDeltaJSON[] {
  const names = new Set([...Object.keys(left), ...Object.keys(right)]);
  return [...names].map((name) => {
    const aValue = left[name] ?? 0;
    const bValue = right[name] ?? 0;
    return {
      name,
      delta: bValue - aValue,
      a_value: aValue,
      b_value: bValue,
    };
  }).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta) || a.name.localeCompare(b.name));
}

export function browserPerTokenDiff(
  left: readonly LoomTokenRowJSON[],
  right: readonly LoomTokenRowJSON[],
): DiffTokenSpanJSON[] {
  const aOffsets = tokenByteOffsets(left);
  const bOffsets = tokenByteOffsets(right);
  const output: DiffTokenSpanJSON[] = [];
  let aIndex = 0;
  let bIndex = 0;
  while (aIndex < left.length && bIndex < right.length) {
    const aEnd = aOffsets[aIndex + 1];
    const bEnd = bOffsets[bIndex + 1];
    const aligned = aEnd === bEnd && left[aIndex].text === right[bIndex].text;
    output.push({
      a_index: aIndex,
      b_index: bIndex,
      a_text: left[aIndex].text,
      b_text: right[bIndex].text,
      aligned,
      reading_deltas: [],
    });
    if (aligned) {
      aIndex += 1;
      bIndex += 1;
    } else if (aEnd <= bEnd) {
      aIndex += 1;
    } else {
      bIndex += 1;
    }
  }
  while (aIndex < left.length) {
    output.push({
      a_index: aIndex,
      b_index: -1,
      a_text: left[aIndex].text,
      b_text: "",
      aligned: false,
      reading_deltas: [],
    });
    aIndex += 1;
  }
  while (bIndex < right.length) {
    output.push({
      a_index: -1,
      b_index: bIndex,
      a_text: "",
      b_text: right[bIndex].text,
      aligned: false,
      reading_deltas: [],
    });
    bIndex += 1;
  }
  return output;
}

type Opcode = [
  tag: "equal" | "delete" | "insert" | "replace",
  aStart: number,
  aEnd: number,
  bStart: number,
  bEnd: number,
];

function sequenceOpcodes(a: readonly string[], b: readonly string[]): Opcode[] {
  let aPosition = 0;
  let bPosition = 0;
  const output: Opcode[] = [];
  for (const match of matchingBlocks(a, b)) {
    const tag = aPosition < match.a && bPosition < match.b
      ? "replace"
      : aPosition < match.a
        ? "delete"
        : bPosition < match.b
          ? "insert"
          : null;
    if (tag) output.push([tag, aPosition, match.a, bPosition, match.b]);
    if (match.size > 0) {
      output.push([
        "equal",
        match.a,
        match.a + match.size,
        match.b,
        match.b + match.size,
      ]);
    }
    aPosition = match.a + match.size;
    bPosition = match.b + match.size;
  }
  return output;
}

function matchingBlocks(a: readonly string[], b: readonly string[]): Match[] {
  const bPositions = new Map<string, number[]>();
  b.forEach((word, index) => {
    const positions = bPositions.get(word) ?? [];
    positions.push(index);
    bPositions.set(word, positions);
  });
  const pending: Array<[number, number, number, number]> = [[0, a.length, 0, b.length]];
  const matches: Match[] = [];
  while (pending.length > 0) {
    const [aLow, aHigh, bLow, bHigh] = pending.pop()!;
    const match = longestMatch(a, b, bPositions, aLow, aHigh, bLow, bHigh);
    if (match.size === 0) continue;
    matches.push(match);
    if (aLow < match.a && bLow < match.b) {
      pending.push([aLow, match.a, bLow, match.b]);
    }
    if (match.a + match.size < aHigh && match.b + match.size < bHigh) {
      pending.push([match.a + match.size, aHigh, match.b + match.size, bHigh]);
    }
  }
  matches.sort((left, right) => left.a - right.a || left.b - right.b);
  const collapsed: Match[] = [];
  for (const match of matches) {
    const previous = collapsed.at(-1);
    if (
      previous && previous.a + previous.size === match.a &&
      previous.b + previous.size === match.b
    ) {
      previous.size += match.size;
    } else {
      collapsed.push({ ...match });
    }
  }
  collapsed.push({ a: a.length, b: b.length, size: 0 });
  return collapsed;
}

function longestMatch(
  a: readonly string[],
  b: readonly string[],
  bPositions: ReadonlyMap<string, readonly number[]>,
  aLow: number,
  aHigh: number,
  bLow: number,
  bHigh: number,
): Match {
  let bestA = aLow;
  let bestB = bLow;
  let bestSize = 0;
  let previousLengths = new Map<number, number>();
  for (let aIndex = aLow; aIndex < aHigh; aIndex += 1) {
    const lengths = new Map<number, number>();
    for (const bIndex of bPositions.get(a[aIndex]) ?? []) {
      if (bIndex < bLow) continue;
      if (bIndex >= bHigh) break;
      const size = (previousLengths.get(bIndex - 1) ?? 0) + 1;
      lengths.set(bIndex, size);
      if (size > bestSize) {
        bestA = aIndex - size + 1;
        bestB = bIndex - size + 1;
        bestSize = size;
      }
    }
    previousLengths = lengths;
  }
  while (bestA > aLow && bestB > bLow && a[bestA - 1] === b[bestB - 1]) {
    bestA -= 1;
    bestB -= 1;
    bestSize += 1;
  }
  while (
    bestA + bestSize < aHigh && bestB + bestSize < bHigh &&
    a[bestA + bestSize] === b[bestB + bestSize]
  ) {
    bestSize += 1;
  }
  return { a: bestA, b: bestB, size: bestSize };
}

function splitWords(value: string): string[] {
  return value.trim() ? value.trim().split(/\s+/u) : [];
}

function appendTextSpan(
  spans: DiffTextSpanJSON[],
  state: "equal" | "delete" | "insert",
  words: readonly string[],
): void {
  const text = words.join(" ");
  if (text) spans.push({ state, text });
}

function tokenByteOffsets(tokens: readonly LoomTokenRowJSON[]): number[] {
  const offsets = [0];
  let position = 0;
  for (const token of tokens) {
    position += new TextEncoder().encode(token.text).byteLength;
    offsets.push(position);
  }
  return offsets;
}
