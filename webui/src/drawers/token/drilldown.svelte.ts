// Page-session drilldown UI state.
//
// The selected tab is STICKY across token clicks and drawer reopens — a
// research pass compares one instrument across many tokens, and the old
// per-open reset kept bouncing the user off it.  Module scope = page
// session; prefer j-lens when the inspected token can provide it.

export type DrilldownTab = "geometry" | "logits" | "sae" | "lens";

export const drilldownUi: { tab: DrilldownTab } = $state({ tab: "lens" });

export function availableDrilldownTab(
  preferred: DrilldownTab,
  available: Record<Exclude<DrilldownTab, "logits">, boolean>,
): DrilldownTab {
  if (preferred === "logits" || available[preferred]) return preferred;
  if (available.lens) return "lens";
  if (available.sae) return "sae";
  if (available.geometry) return "geometry";
  return "logits";
}
