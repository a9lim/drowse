import type { ManifoldDomain } from "../types";

export function requireBrowserDomain(domain: ManifoldDomain): asserts domain is Exclude<ManifoldDomain, { type: "klein" | "projective" }> {
  if (domain.type === "klein" || domain.type === "projective") {
    throw new TypeError("Klein-bottle and RP² steering require the native Python runtime.");
  }
}

export function surfaceLabel(candidate: string | null): string {
  if (!candidate) return "Unresolved surface";
  const names: Record<string, string> = {
    sphere: "Sphere", torus: "Torus", "klein-bottle": "Klein bottle",
    "projective-plane": "Real projective plane (RP²)",
  };
  return names[candidate] ?? candidate.replaceAll("-", " ");
}

export function quotientDescription(domain: ManifoldDomain): string | null {
  if (domain.type === "klein") return "Klein bottle · two angles in radians. Crossing the u seam reverses v; these are coordinates, not a flat surface.";
  if (domain.type === "projective") return "Real projective plane (RP²) · polar and azimuth angles in radians. Opposite points on the sphere represent the same position.";
  return null;
}

export function parseSurfacePoints(text: string): number[][] {
  let points: unknown;
  try { points = JSON.parse(text); }
  catch { throw new Error("Enter a JSON array of points, such as [[0, 1, 2], …]."); }
  if (!Array.isArray(points) || points.length < 32 || points.length > 384) {
    throw new Error("Use 32–384 points. Larger clouds must be sampled before inspection.");
  }
  const dimension = Array.isArray(points[0]) ? points[0].length : 0;
  if (dimension < 2 || dimension > 1024 || !points.every(row =>
    Array.isArray(row) && row.length === dimension && row.every(value => typeof value === "number" && Number.isFinite(value)))) {
    throw new Error("Each point needs the same 2–1,024 finite numeric coordinates.");
  }
  if (new Set(points.map(row => JSON.stringify(row))).size !== points.length) {
    throw new Error("Remove duplicate points before inspecting the surface.");
  }
  return points as number[][];
}

export function exampleSphere(): number[][] {
  return Array.from({ length: 96 }, (_, i) => {
    const z = 1 - 2 * (i + 0.5) / 96;
    const angle = i * Math.PI * (3 - Math.sqrt(5));
    const radius = Math.sqrt(1 - z * z);
    return [radius * Math.cos(angle), radius * Math.sin(angle), z];
  });
}
