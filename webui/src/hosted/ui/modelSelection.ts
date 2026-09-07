import type { HostedModelOption, HostedShellSnapshot } from "./types";

export function modelsBySize(models: HostedModelOption[]): HostedModelOption[] {
  return models.toSorted((left, right) => left.firstRunBytes - right.firstRunBytes);
}

export function downloadUnavailableReason(
  model: HostedModelOption | null,
  download: HostedShellSnapshot["download"],
): string {
  if (!download.available) return download.reason;
  return model?.setupIssue ?? model?.reason ?? download.reason;
}

export function defaultModelSelection(
  models: HostedModelOption[],
  requestedId: string | null,
  savedId?: string,
): string | null {
  const explicit = models.find((model) => model.id === requestedId) ??
    models.find((model) => model.id === savedId);
  if (explicit) return explicit.id;
  const chatModels = models.filter((model) => model.modelType !== "base");
  return chatModels.find((model) => model.fit === "recommended")?.id ??
    chatModels.find((model) => model.fit === "eligible")?.id ??
    chatModels.find((model) => model.fit === "uncertain")?.id ??
    chatModels[0]?.id ?? null;
}
