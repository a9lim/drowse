import type { DrawerName } from "../types";
import type {
  RuntimeCapabilities,
  RuntimeCapabilityOperation,
} from "./contracts";
import { getRuntimeCapabilities } from "./registry";
import { userFacingError } from "./userFacingError";

const DRAWER_REQUIREMENTS: Record<
  DrawerName,
  readonly RuntimeCapabilityOperation[]
> = {
  subspace: [],
  manifolds: [],
  manifold_builder: ["fitting", "manifold_artifacts"],
  surface_geometry: [],
  manifold_merge: ["manifold_artifacts"],
  manifold_pack: ["manifold_artifacts"],
  save_conversation: [],
  download_chat: [],
  load_conversation: [],
  compare: [],
  system_prompt: [],
  token_drilldown: [],
  correlation: [],
  probe_inspector: [],
  advanced_sampling: [],
  health: [],
  appearance: [],
  session_admin: ["session_admin"],
  local_runtime: [],
  help: [],
  node_compare: [],
  transcript: [],
  template_lab: ["manifold_artifacts", "fitting"],
  cast: [],
};

export interface UiCapabilityAvailability {
  available: boolean;
  reason: string | null;
}

export function runtimeOperationAvailability(
  operation: RuntimeCapabilityOperation,
): UiCapabilityAvailability {
  return operationAvailability(operation, getRuntimeCapabilities());
}

function operationAvailability(
  operation: RuntimeCapabilityOperation,
  capabilities: RuntimeCapabilities | null,
): UiCapabilityAvailability {
  if (!capabilities) return { available: true, reason: null };
  const availability = capabilities.operations[operation];
  return {
    available: availability.available,
    reason: availability.available
      ? null
      : availability.reasons[0]
        ? userFacingError(
            availability.reasons[0],
            "This action is unavailable in the current browser or model session.",
          )
        : "This action is unavailable in the current browser or model session.",
  };
}

export function drawerAvailability(
  drawer: DrawerName,
  capabilities: RuntimeCapabilities | null = getRuntimeCapabilities(),
): UiCapabilityAvailability {
  for (const requirement of DRAWER_REQUIREMENTS[drawer]) {
    const availability = operationAvailability(requirement, capabilities);
    if (!availability.available) return availability;
  }
  return { available: true, reason: null };
}
