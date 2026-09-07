import { ApiError } from "./errors";

export function isFittingCancellation(error: unknown): boolean {
  if (error instanceof ApiError) {
    const body = record(error.body);
    const nested = record(body?.error);
    return nested?.code === "FITTING_CANCELLED";
  }
  return record(error)?.code === "FITTING_CANCELLED";
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}
