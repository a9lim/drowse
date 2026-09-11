import type { BrowserRuntimeClass, RuntimeCapabilities } from "./contracts";

export const MAX_OUTPUT_TOKEN_COUNT = Number.MAX_SAFE_INTEGER;

type RuntimeSignals = Pick<
  RuntimeCapabilities["signals"],
  "appleMobile" | "runtimeClass"
>;

export function outputTokenLimitForRuntime(
  runtimeClass?: BrowserRuntimeClass,
  contextTokens?: number,
): number {
  void runtimeClass;
  return contextTokens ?? MAX_OUTPUT_TOKEN_COUNT;
}

export function defaultOutputTokenCount(runtimeClass?: BrowserRuntimeClass): number {
  return runtimeClass === "apple-mobile-webkit" ? 256 : 1024;
}

export function outputTokenLimitForSignals(
  signals?: Partial<RuntimeSignals> | null,
  contextTokens?: number,
): number {
  return outputTokenLimitForRuntime(signals?.runtimeClass, contextTokens);
}

export function clampOutputTokenCount(value: number, limit: number): number {
  return Math.min(value, limit);
}
