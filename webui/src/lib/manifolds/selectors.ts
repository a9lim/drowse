import type {
  ManifoldInfo,
  ProbeInfo,
  ProbeRackEntry,
  SteerEntry,
  Variant,
} from "../types";

export const PER_NODE_ROLE_HELP =
  "Optional: give each node its own assistant voice. During steering, Drowse uses the role of the nearest node.";

export interface ManifoldSelectorOption {
  selector: string;
  variant: Variant;
  label: string;
  available: boolean;
  unavailableReason: string | null;
}

const VARIANT_RE = /^(?:raw|sae(?:-[a-z0-9._-]+)?|role(?:-[a-z0-9._-]+)?|from(?:-[a-z0-9._-]+)?)$/u;

export function manifoldSelectorOptions(
  manifold: ManifoldInfo,
  modelId: string | null,
  _browserMode: boolean,
): ManifoldSelectorOption[] {
  const key = `${manifold.namespace}/${manifold.name}`;
  const variants = variantsForModel(manifold, modelId);
  const roles = manifold.node_roles ?? [];
  const nonNullRoles = roles.filter((role): role is string => role !== null);
  const uniformRole = nonNullRoles.length === roles.length && roles.length > 0 &&
      roles.every((role) => role === roles[0])
    ? roles[0]
    : null;
  const hasMixedRoles = nonNullRoles.length > 0 && uniformRole === null;
  const out: ManifoldSelectorOption[] = [];

  for (const variant of variants) {
    if (!isVariant(variant)) continue;
    if (variant === "raw" && uniformRole !== null) {
      const roleVariant = `role-${uniformRole}` as Variant;
      out.push({
        selector: `${key}:${roleVariant}`,
        variant: roleVariant,
        label: `shared role · ${uniformRole}`,
        available: true,
        unavailableReason: null,
      });
      continue;
    }
    if (variant === "raw" && hasMixedRoles) {
      out.push({
        selector: key,
        variant: "raw",
        label: "nearest node role",
        available: true,
        unavailableReason: null,
      });
      continue;
    }
    out.push({
      selector: variant === "raw" ? key : `${key}:${variant}`,
      variant,
      label: variantLabel(variant),
      available: true,
      unavailableReason: null,
    });
  }

  if (out.length === 0 && manifold.fitted_for_session) {
    if (uniformRole !== null) {
      const roleVariant = `role-${uniformRole}` as Variant;
      out.push({
        selector: `${key}:${roleVariant}`,
        variant: roleVariant,
        label: `shared role · ${uniformRole}`,
        available: true,
        unavailableReason: null,
      });
    } else {
      out.push({
        selector: key,
        variant: "raw",
        label: hasMixedRoles ? "nearest node role" : "residual",
        available: true,
        unavailableReason: null,
      });
    }
  }

  return dedupeOptions(out).sort((left, right) =>
    variantOrder(left.variant) - variantOrder(right.variant) ||
    left.label.localeCompare(right.label)
  );
}

export function selectedManifoldOption(
  options: readonly ManifoldSelectorOption[],
  selectedSelector: string | undefined,
): ManifoldSelectorOption | null {
  const selected = options.find((option) => option.selector === selectedSelector);
  return selected ?? options.find((option) => option.available) ?? options[0] ?? null;
}

export function selectorReferencesManifold(
  selector: string,
  manifold: Pick<ManifoldInfo, "namespace" | "name">,
): boolean {
  const identity = selectorManifoldIdentity(selector);
  if (identity === null) return false;
  const key = `${manifold.namespace}/${manifold.name}`;
  return identity === key || identity === manifold.name;
}

export function probeReferencesManifold(
  info: ProbeInfo,
  manifold: Pick<ManifoldInfo, "namespace" | "name">,
  requestSelector?: string,
): boolean {
  if (info.family !== "geometry") return false;
  return selectorReferencesManifold(info.manifold, manifold) ||
    (requestSelector !== undefined && selectorReferencesManifold(requestSelector, manifold));
}

export function manifoldUsage(
  manifold: Pick<ManifoldInfo, "namespace" | "name">,
  rack: ReadonlyMap<string, SteerEntry>,
  probes: ReadonlyMap<string, ProbeRackEntry>,
): { rack: string[]; probes: string[] } {
  const rackReferences = [...rack.keys()].filter((selector) =>
    selectorReferencesManifold(selector, manifold)
  );
  const probeReferences = [...probes.values()]
    .filter((entry) => probeReferencesManifold(
      entry.info,
      manifold,
      entry.request.selector,
    ))
    .map((entry) => entry.info.name);
  return {
    rack: [...new Set(rackReferences)].sort(),
    probes: [...new Set(probeReferences)].sort(),
  };
}

export function manifoldUsageMessage(
  manifold: Pick<ManifoldInfo, "namespace" | "name">,
  usage: { rack: readonly string[]; probes: readonly string[] },
): string | null {
  const references: string[] = [];
  if (usage.rack.length > 0) references.push(`steering: ${usage.rack.join(", ")}`);
  if (usage.probes.length > 0) references.push(`readings: ${usage.probes.join(", ")}`);
  if (references.length === 0) return null;
  return `Remove ${manifold.namespace}/${manifold.name} from active tools before deleting it (${references.join("; ")}).`;
}

function variantsForModel(manifold: ManifoldInfo, modelId: string | null): string[] {
  const byModel = manifold.tensor_variants ?? {};
  if (modelId !== null) {
    const direct = byModel[modelId];
    if (direct) return [...direct];
    const safe = safeModelId(modelId);
    const encoded = byModel[safe];
    if (encoded) return [...encoded];
  }
  const values = Object.values(byModel);
  if (values.length === 1) return [...values[0]];
  return manifold.fitted_for_session ? ["raw"] : [];
}

function safeModelId(modelId: string): string {
  const bytes = new TextEncoder().encode(modelId);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return `_z${btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/u, "")}`;
}

function selectorManifoldIdentity(selector: string): string | null {
  let value = selector.trim();
  if (!value || value.startsWith("sae/") || value.startsWith("jlens/")) return null;
  const position = value.indexOf("%");
  if (position >= 0) value = value.slice(0, position);
  const variant = value.lastIndexOf(":");
  if (variant >= 0 && VARIANT_RE.test(value.slice(variant + 1))) {
    value = value.slice(0, variant);
  }
  return value || null;
}

function isVariant(value: string): value is Variant {
  return VARIANT_RE.test(value);
}

function variantLabel(variant: Variant): string {
  if (variant === "raw") return "residual";
  if (variant === "sae") return "SAE";
  if (variant.startsWith("sae-")) return `SAE · ${variant.slice(4)}`;
  if (variant === "from") return "transferred";
  if (variant.startsWith("from-")) return `transferred · ${variant.slice(5)}`;
  if (variant === "role") return "shared role";
  return `shared role · ${variant.slice(5)}`;
}

function variantOrder(variant: Variant): number {
  if (variant === "raw" || variant === "role" || variant.startsWith("role-")) return 0;
  if (variant === "sae" || variant.startsWith("sae-")) return 1;
  return 2;
}

function dedupeOptions(options: readonly ManifoldSelectorOption[]): ManifoldSelectorOption[] {
  const seen = new Set<string>();
  return options.filter((option) => {
    if (seen.has(option.selector)) return false;
    seen.add(option.selector);
    return true;
  });
}
