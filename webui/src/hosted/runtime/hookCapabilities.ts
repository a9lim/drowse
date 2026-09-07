import type { SteeringExpression, SteeringExpressionTerm } from "./steeringExpression";
import { parseSteeringExpression } from "./steeringExpression";

export type DrowseHookFeature =
  | "rank_one"
  | "multi_term"
  | "projection"
  | "ablation"
  | "curved_manifold"
  | "phase_trigger"
  | "probe_gate"
  | "sae"
  | "jlens";

export type DrowseHookCapabilities = Readonly<Record<DrowseHookFeature, boolean>>;

export const AFFINE_FEASIBILITY_CAPABILITIES: DrowseHookCapabilities = Object.freeze({
  rank_one: true,
  multi_term: true,
  projection: true,
  ablation: true,
  curved_manifold: false,
  phase_trigger: false,
  probe_gate: false,
  sae: false,
  jlens: false,
});

export function requiredHookFeatures(
  expression: string | SteeringExpression,
): ReadonlySet<DrowseHookFeature> {
  const parsed = typeof expression === "string"
    ? parseSteeringExpression(expression)
    : expression;
  const features = new Set<DrowseHookFeature>(["rank_one"]);
  if (parsed.terms.length > 1) features.add("multi_term");
  for (const term of parsed.terms) collectTermFeatures(term, features);
  return features;
}

export function unsupportedHookFeatures(
  required: ReadonlySet<DrowseHookFeature>,
  capabilities: DrowseHookCapabilities,
): DrowseHookFeature[] {
  return [...required].filter((feature) => !capabilities[feature]).sort();
}

export function assertHookFeaturesSupported(
  expression: string | SteeringExpression,
  capabilities: DrowseHookCapabilities,
): void {
  const missing = unsupportedHookFeatures(requiredHookFeatures(expression), capabilities);
  if (missing.length === 0) return;
  throw Object.assign(
    new Error(`The browser model library does not support: ${missing.join(", ")}`),
    {
      code: "DROWSE_HOOK_FEATURE_UNAVAILABLE",
      missingFeatures: missing,
    },
  );
}

function collectTermFeatures(
  term: SteeringExpressionTerm,
  features: Set<DrowseHookFeature>,
): void {
  if (term.selector.projection !== null) features.add("projection");
  if (term.ablation) features.add("ablation");
  if (term.selector.manifoldPosition !== null) features.add("curved_manifold");
  if (term.trigger !== null) {
    if (term.trigger.phase.kind !== "both") features.add("phase_trigger");
    if (term.trigger.gate !== null) features.add("probe_gate");
  }
  for (const atom of [term.selector.base, term.selector.onto]) {
    if (atom === null) continue;
    if (atom.namespace === "sae" || atom.variant === "sae" || atom.variant.startsWith("sae-")) {
      features.add("sae");
    }
    if (atom.namespace === "jlens") features.add("jlens");
  }
}
