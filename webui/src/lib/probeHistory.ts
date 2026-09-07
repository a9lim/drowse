import type { ProbeInfo } from "./types";

export function savedProbeSummary(info: ProbeInfo, scores: Record<string, number>) {
  const scalar = (key: string): number | null => {
    const value = scores[key];
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  };
  const coordinates: number[] = [];
  const fraction = info.family === "geometry" ? scalar(`${info.name}:fraction`) : null;
  if (info.family === "geometry") {
    for (let axis = 0; axis < info.intrinsic_dim; axis += 1) {
      const value = axis === 0 ? scalar(info.name) : scalar(`${info.name}[${axis}]`);
      if (value === null) break;
      coordinates.push(value);
    }
  }
  return {
    value: info.family === "geometry" && !info.is_affine ? fraction : scalar(info.name),
    coordinates,
    fraction,
  };
}
