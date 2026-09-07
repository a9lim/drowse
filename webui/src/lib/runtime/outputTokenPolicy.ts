import type { BrowserRuntimeClass, RuntimeCapabilities } from "./contracts";

export const DESKTOP_MAX_OUTPUT_TOKENS = 8_192;
export const PREVIEW_DESKTOP_MAX_OUTPUT_TOKENS = 1_024;
export const APPLE_MOBILE_MAX_OUTPUT_TOKENS = 256;

type RuntimeSignals = Pick<
  RuntimeCapabilities["signals"],
  "appleMobile" | "runtimeClass"
>;

export function outputTokenLimitForRuntime(
  runtimeClass?: BrowserRuntimeClass,
): number {
  if (runtimeClass === "apple-mobile-webkit") return APPLE_MOBILE_MAX_OUTPUT_TOKENS;
  if (runtimeClass === "desktop-webkit" || runtimeClass === "desktop-gecko") {
    return PREVIEW_DESKTOP_MAX_OUTPUT_TOKENS;
  }
  return DESKTOP_MAX_OUTPUT_TOKENS;
}

export function outputTokenLimitForSignals(
  signals?: Partial<RuntimeSignals> | null,
): number {
  return signals?.appleMobile === true
    ? APPLE_MOBILE_MAX_OUTPUT_TOKENS
    : outputTokenLimitForRuntime(signals?.runtimeClass);
}

export function clampOutputTokenCount(value: number, limit: number): number {
  return Math.min(value, limit);
}
