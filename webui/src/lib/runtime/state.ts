import type {
  RuntimeFailure,
  RuntimeOperationState,
  RuntimeSnapshot,
} from "./contracts";

export function idleOperation(): RuntimeOperationState {
  return {
    phase: "idle",
    startedAt: null,
    finishedAt: null,
    error: null,
  };
}

export function initialRuntimeSnapshot(): RuntimeSnapshot {
  return {
    lifecycle: "uninitialized",
    modelVariantId: null,
    contextTokens: null,
    selectedModelVariantId: null,
    installedModelVariantIds: [],
    installedPackIds: [],
    loadRecords: [],
    download: idleOperation(),
    generation: idleOperation(),
    fitting: idleOperation(),
    error: null,
  };
}

export function runtimeFailure(
  code: string,
  message: string,
  recoverable: boolean,
  status = 500,
  detail?: unknown,
): RuntimeFailure {
  return detail === undefined
    ? { code, message, recoverable, status }
    : { code, message, recoverable, status, detail };
}
