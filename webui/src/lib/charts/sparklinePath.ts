export function sparklinePaths(
  points: readonly (number | null)[],
  width: number,
  height: number,
  cap: number,
): { line: string; area: string } {
  if (points.length === 0 || !Number.isFinite(cap) || cap <= 0) {
    return { line: "", area: "" };
  }
  const yFor = (value: number) =>
    ((cap - Math.max(-cap, Math.min(cap, value))) / (2 * cap)) * height;
  if (points.length === 1) {
    const value = points[0];
    if (value === null || !Number.isFinite(value)) return { line: "", area: "" };
    const y = yFor(value).toFixed(2);
    return { line: `M 0 ${y} L ${width} ${y}`, area: "" };
  }
  const step = width / (points.length - 1);
  const lines: string[] = [];
  const areas: string[] = [];
  let segment: string[] = [];
  let start = 0;
  let end = 0;
  const finish = () => {
    lines.push(...segment);
    if (segment.length > 1) {
      areas.push(...segment, `L ${end.toFixed(2)} ${height.toFixed(2)}`,
        `L ${start.toFixed(2)} ${height.toFixed(2)} Z`);
    }
    segment = [];
  };
  for (let index = 0; index < points.length; index += 1) {
    const value = points[index];
    if (value === null || !Number.isFinite(value)) {
      finish();
      continue;
    }
    end = index * step;
    if (segment.length === 0) start = end;
    segment.push(`${segment.length === 0 ? "M" : "L"} ${end.toFixed(2)} ${yFor(value).toFixed(2)}`);
  }
  finish();
  return { line: lines.join(" "), area: areas.join(" ") };
}
