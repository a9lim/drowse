import { userFacingError } from "./userFacingError";

export class ApiError extends Error {
  readonly status: number;
  readonly path: string;
  readonly body: unknown;
  readonly rawBody: string;

  constructor(status: number, path: string, rawBody: string, parsed: unknown) {
    const record =
      parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : null;
    const nestedError =
      record?.error && typeof record.error === "object"
        ? (record.error as Record<string, unknown>)
        : null;
    const detail = record?.detail ?? nestedError?.message;
    super(
      typeof detail === "string" && detail.trim()
        ? detail
        : rawBody.trim().slice(0, 200) || `Request failed (${status})`,
    );
    this.name = "ApiError";
    this.status = status;
    this.path = path;
    this.rawBody = rawBody;
    this.body = parsed;
  }
}

export function describeError(error: unknown): string {
  return userFacingError(error);
}
