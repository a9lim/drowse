import type { InstrumentFamily, ProbeInfo, SteerEntry } from "../types";

export type OptionalInstrumentPackKind = "jlens" | "sae";

export interface InstrumentPackDependencyInput {
  kind: OptionalInstrumentPackKind | null;
  steeringExpression: string;
  steeringEntries: Iterable<readonly [string, SteerEntry]>;
  probes: Iterable<ProbeInfo>;
}

export function instrumentPackDependencies(
  input: InstrumentPackDependencyInput,
): string[] {
  const kinds: OptionalInstrumentPackKind[] = input.kind === null
    ? ["jlens", "sae"]
    : [input.kind];
  const entries = [...input.steeringEntries];
  const probes = [...input.probes];
  const references: string[] = [];

  for (const kind of kinds) {
    const family: InstrumentFamily = kind === "jlens" ? "lens" : "sae";
    const label = kind === "jlens" ? "J-lens" : "SAE";
    const direct = entries
      .filter(([, entry]) => entry.mode === kind)
      .map(([name]) => name)
      .sort();
    const readings = probes
      .filter((probe) => probe.family === family)
      .map((probe) => probe.name)
      .sort();
    const expressionUsesFamily = kind === "jlens"
      ? /(?:^|[^a-z0-9_-])jlens\//i.test(input.steeringExpression)
      : /(?:^|[^a-z0-9_-])sae\/\d/i.test(input.steeringExpression);

    if (direct.length > 0) references.push(`${label} steering: ${direct.join(", ")}`);
    if (expressionUsesFamily && direct.length === 0) {
      references.push(`${label} steering expression or condition`);
    }
    if (readings.length > 0) references.push(`${label} readings: ${readings.join(", ")}`);
  }

  return references;
}
