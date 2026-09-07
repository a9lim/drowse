import type { CatalogDocument } from "./contracts";
import { getHostedController } from "./registry";

export type CatalogInstrumentAvailability =
  | "available"
  | "unavailable"
  | "context-unavailable"
  | "unknown";

export function catalogInstrumentAvailability(
  catalog: Pick<CatalogDocument, "models">,
  modelVariantId: string,
  contextTokens: number | null,
  kind: "jlens" | "sae",
): CatalogInstrumentAvailability {
  const variant = catalog.models
    .flatMap((model) => model.variants)
    .find((candidate) => candidate.id === modelVariantId);
  if (!variant) return "unknown";

  const packs = variant.packs.filter((pack) => pack.kind === kind);
  if (packs.length === 0) return "unavailable";
  if (contextTokens === null) return "available";

  const context = variant.contextProfiles.find(
    (profile) => profile.contextTokens === contextTokens,
  );
  if (!context) return "context-unavailable";
  return packs.some((pack) =>
    pack.compatibleContextBindingSha256.includes(context.bindingSha256)
  )
    ? "available"
    : "context-unavailable";
}

export async function currentCatalogInstrumentAvailability(
  kind: "jlens" | "sae",
): Promise<CatalogInstrumentAvailability> {
  const controller = getHostedController();
  if (!controller) return "unknown";
  const { modelVariantId, contextTokens } = controller.snapshot;
  if (!modelVariantId) return "unknown";

  try {
    const catalog = await controller.catalog({
      preferCached: true,
      offline: typeof navigator !== "undefined" && navigator.onLine === false,
    });
    return catalogInstrumentAvailability(
      catalog.document,
      modelVariantId,
      contextTokens,
      kind,
    );
  } catch {
    return "unknown";
  }
}
