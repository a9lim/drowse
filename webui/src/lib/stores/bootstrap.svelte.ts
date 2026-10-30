// Dashboard bootstrap — call once on App mount.
//
// The only slice that depends on every other one, and nothing depends on
// it: the leaf of the store graph.

import { loadGenUiMode } from "./chat.svelte";
import { refreshLoomTree } from "./loom.svelte";
import { loadPersistedPreferences } from "./persistence.svelte";
import { refreshProbeList } from "./probes.svelte";
import { refreshSession, sessionState } from "./session.svelte";
import {
  refreshCorrelation,
  refreshManifoldList,
  refreshVectorList,
  steerRack,
} from "./steering.svelte";

/** Bootstrap the dashboard — call once on App mount.  Resolves only once
 * every parallel fetch settles so the UI's first paint has a real session
 * shape. */
export async function bootstrap(): Promise<void> {
  // Session info has to land before the localStorage key is known
  // (it's scoped by model_id), so we serialize that step.  The other
  // refreshes parallelize as before.
  await refreshSession();
  if (steerRack.customExpression === null && steerRack.entries.size === 0) {
    steerRack.customExpression = sessionState.info?.default_steering ?? null;
  }
  // Restore presentation preferences before attaching the persist effect so
  // we do not immediately overwrite them.  The tree always comes from the
  // authoritative server fetch below.
  loadPersistedPreferences();
  // Per-model render-mode override (base vs chat) — also model-scoped.
  loadGenUiMode();
  await Promise.allSettled([
    refreshVectorList(),
    // Unified probe roster — every probe shape.
    refreshProbeList(),
    refreshCorrelation(),
    // Current manifold catalog.
    refreshManifoldList(),
    // Server tree wins — fetch and reconcile; failures remain visible.
    refreshLoomTree(),
  ]);
}
