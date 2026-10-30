import type { HostedModelOption, HostedShellSnapshot } from "./types";
import { formatBytes } from "../../lib/format/bytes";

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

// Chat models before base models, the device recommendation first, then smallest first.
export function modelsForChooser(models: HostedModelOption[]): HostedModelOption[] {
  return modelsBySize(models).toSorted((left, right) =>
    Number(left.modelType === "base") - Number(right.modelType === "base") ||
    Number(right.fit === "recommended") - Number(left.fit === "recommended")
  );
}

// What the next download for this model contains: the model files (unless already
// on the device) and the selected tool packs that are still missing.
export function remainingDownloadParts(model: HostedModelOption): { modelBytes: number; toolBytes: number } {
  return {
    modelBytes: model.installed ? 0 : model.modelDownloadBytes,
    toolBytes: model.firstRunPacks
      .filter((pack) => pack.selected && !pack.installed)
      .reduce((total, pack) => total + pack.bytes, 0),
  };
}

export function downloadSizeSummary(model: HostedModelOption): string {
  const { modelBytes, toolBytes } = remainingDownloadParts(model);
  if (modelBytes > 0 && toolBytes > 0) {
    return `Model ${formatBytes(modelBytes)} + tools ${formatBytes(toolBytes)}`;
  }
  if (toolBytes > 0) return `Tools ${formatBytes(toolBytes)}`;
  return formatBytes(modelBytes);
}
