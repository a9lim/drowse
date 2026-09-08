import type { RuntimeMode } from "./contracts";

export const SAMPLING_TEMPERATURE_MAX = 3.4028234663852886e38;

// Actual candidate counts are bounded by the loaded vocabulary in the sampler.
export const HTTP_RETURN_TOP_K_MAX = Number.MAX_SAFE_INTEGER;
export const BROWSER_RETURN_TOP_K_MAX = Number.MAX_SAFE_INTEGER;
export const BROWSER_SAMPLING_TOP_K_MAX = Number.MAX_SAFE_INTEGER;
export const BROWSER_RETURN_TOP_K_DEFAULT = 5;
export const HTTP_RETURN_TOP_K_DEFAULT = 8;

export function tokenAlternativeLimit(mode: RuntimeMode): number {
  if (mode === "http") return HTTP_RETURN_TOP_K_MAX;
  if (mode === "browser") return BROWSER_RETURN_TOP_K_MAX;
  return 0;
}

export function tokenAlternativeDefault(mode: RuntimeMode): number {
  if (mode === "http") return HTTP_RETURN_TOP_K_DEFAULT;
  if (mode === "browser") return BROWSER_RETURN_TOP_K_DEFAULT;
  return 0;
}

export function clampTokenAlternativeCount(
  value: number,
  mode: RuntimeMode,
): number {
  return Math.max(0, Math.min(tokenAlternativeLimit(mode), Math.floor(value)));
}
