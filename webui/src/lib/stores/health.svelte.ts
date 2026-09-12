import { genStatus, geometricMeanPpl } from "./chat.svelte";
import { loomTree, refreshLoomTree } from "./loom.svelte";
import { probeRack, refreshProbeList } from "./probes.svelte";
import { refreshSession, sessionState } from "./session.svelte";
import { refreshCorrelation, refreshManifoldList, refreshVectorList, steerRack, vectorsState } from "./steering.svelte";
import { userFacingError } from "../runtime/userFacingError";

export const healthState = $state({ busy: false, lastAudit: null as string | null, error: null as string | null });

export function healthWarnings(): string[] {
  const warnings: string[] = [];
  if (!sessionState.info) warnings.push("session info is not loaded");
  if (sessionState.error) warnings.push(sessionState.error);
  if (loomTree.error) warnings.push(`loom API error: ${loomTree.error}`);
  if (steerRack.error) warnings.push(steerRack.error);
  if (probeRack.error) warnings.push(probeRack.error);
  if (steerRack.catalog.length === 0) warnings.push("no manifold artifacts are available");
  if (probeRack.active.length === 0) warnings.push("no active probes; internal-state views will be sparse");
  return warnings;
}

export function readHealth() {
  const info = sessionState.info;
  const errors = [healthState.error, sessionState.error, loomTree.error, steerRack.error, probeRack.error].filter((error): error is string => !!error);
  return {
    status: !info ? "unavailable" : errors.length ? "degraded" : "ready",
    model_id: info?.model_id ?? null, device: info?.device ?? null, dtype: info?.dtype ?? null,
    refreshing: healthState.busy, refreshed_at: healthState.lastAudit,
    generation: { active: genStatus.active, finish_reason: genStatus.finishReason, tokens: genStatus.tokensSoFar, max_tokens: genStatus.maxTokens, tokens_per_second: genStatus.tokPerSec, perplexity: geometricMeanPpl(genStatus), measured_steps: genStatus.ppl.count },
    tree: { loaded: loomTree.loaded, nodes: loomTree.nodes.size, revision: loomTree.rev, depth: loomTree.activePath.length },
    artifacts: { available: steerRack.catalog.length, racked: steerRack.entries.size, resident: vectorsState.names.length },
    probes: { active: probeRack.active.length, rows: probeRack.entries.size, correlation_available: steerRack.correlation !== null },
    errors, warnings: healthWarnings(),
  };
}

export async function refreshHealth(): Promise<ReturnType<typeof readHealth>> {
  if (healthState.busy) throw new Error("Diagnostics are already refreshing.");
  healthState.busy = true;
  healthState.error = null;
  try {
    const results = await Promise.allSettled([
      refreshSession(), refreshVectorList(), refreshProbeList(), refreshLoomTree(true), refreshManifoldList(), refreshCorrelation(undefined, true),
    ]);
    const failure = results.find(result => result.status === "rejected");
    if (failure?.status === "rejected") healthState.error = userFacingError(failure.reason, "Unable to refresh every diagnostic. Reopen the model and try again.");
    healthState.lastAudit = new Date().toISOString();
  } finally {
    healthState.busy = false;
  }
  return readHealth();
}
