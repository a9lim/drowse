import type { ProbeInfo, ProbeRackEntry } from "./types";

export interface AnalyticsProbeOmission {
  name: string;
  reason: string;
}

/** Direction-cosine analytics require one affine direction. */
export function isDirectionAnalyticsProbe(info: ProbeInfo): boolean {
  return info.family === "geometry" && info.is_affine && info.intrinsic_dim === 1;
}

/**
 * Browser profile lists are authoritative: they already include only probe
 * aliases the runtime can fold to one direction.  The Python profile list
 * predates that union, so preserve its rank-one geometry inference here.
 */
export function comparisonProfileNames(
  profileNames: readonly string[],
  activeProbeNames: readonly string[],
  entries: ReadonlyMap<string, ProbeRackEntry>,
  inferHttpProbeProfiles: boolean,
): string[] {
  const names = new Set(profileNames);
  if (inferHttpProbeProfiles) {
    for (const name of activeProbeNames) {
      const info = entries.get(name)?.info;
      if (info && isDirectionAnalyticsProbe(info)) names.add(name);
    }
  }
  return [...names].sort((left, right) =>
    left.localeCompare(right, undefined, { sensitivity: "base" })
  );
}

/** Active probes omitted from a completed correlation result, with the real
 * reason their readout cannot be treated as a single residual direction. */
export function omittedAnalyticsProbes(
  activeProbeNames: readonly string[],
  entries: ReadonlyMap<string, ProbeRackEntry>,
  correlationNames: readonly string[],
): AnalyticsProbeOmission[] {
  const included = new Set(correlationNames);
  return activeProbeNames
    .filter((name) => {
      const info = entries.get(name)?.info;
      return !info || !isDirectionAnalyticsProbe(info) || !included.has(name);
    })
    .map((name) => ({
      name,
      reason: analyticsOmissionReason(entries.get(name)?.info),
    }))
    .sort((left, right) =>
      left.name.localeCompare(right.name, undefined, { sensitivity: "base" })
    );
}

function analyticsOmissionReason(info: ProbeInfo | undefined): string {
  if (!info) return "not exposed as a direction by this runtime";
  if (info.family === "lens") return "J-lens readout, not a residual direction";
  if (info.family === "sae") return "SAE readout, not a residual direction";
  if (!info.is_affine) return "curved manifold, not one direction";
  if (info.intrinsic_dim !== 1) return "multidimensional subspace, not one direction";
  return "not exposed as a direction by this runtime";
}
