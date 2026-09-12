import type { WSDoneEvent, WSServerMessage } from "../types";

export interface GenerationReceipt {
  requestId: string;
  state: "completed" | "cancelled";
  results: WSDoneEvent[];
}

export function awaitGenerationReceipt(options: {
  requestId: string;
  expectedSiblings: number;
  signal: AbortSignal;
  subscribe: (listener: (event: WSServerMessage) => void) => () => void;
  send: () => void;
  stop: () => void;
  onProgress?: (value: unknown) => void;
  onPartial?: (receipt: GenerationReceipt) => void;
  requestStatus?: () => Promise<import("../runtime/requestJournal").RuntimeRequestStatus>;
  idleTimeoutMs?: number;
}): Promise<GenerationReceipt> {
  return new Promise((resolve, reject) => {
    const results = new Map<number, WSDoneEvent>();
    let failure: Error | null = null;
    let sent = false;
    let finished = false;
    let unsubscribe = () => {};
    let watchdog: ReturnType<typeof setTimeout> | undefined;
    const receipt = (state: "completed" | "cancelled" = "completed"): GenerationReceipt => ({
      requestId: options.requestId, state,
      results: [...results.values()].sort((a, b) => a.sibling_index - b.sibling_index),
    });
    const finish = (error?: Error, state: "completed" | "cancelled" = "completed"): void => {
      if (finished) return;
      finished = true;
      clearTimeout(watchdog);
      unsubscribe();
      options.signal.removeEventListener("abort", abort);
      if (error) reject(Object.assign(error, { partialResult: { ...receipt(), state: "partial" } }));
      else resolve(receipt(state));
    };
    const abort = (): void => {
      if (!sent) { finish(undefined, "cancelled"); return; }
      try { options.stop(); }
      catch { finish(Object.assign(new Error("The cancellation could not reach the runtime; reconcile this request before retrying"), { code: "GENERATION_INTERRUPTED" })); }
    };
    const checkStatus = async () => {
      if (finished) return;
      try {
        const status = await options.requestStatus?.();
        if (finished) return;
        if (status) {
          for (const row of status.results) results.set(row.sibling_index, row);
          options.onPartial?.(receipt());
          if (["completed", "cancelled", "failed"].includes(status.state)) {
            if (status.error) failure = Object.assign(new Error(status.error.message), { code: status.error.code });
            receive({ type: "request_complete", request_id: options.requestId,
              state: status.state as "completed" | "cancelled" | "failed", completed_siblings: status.results.length });
            return;
          }
          if (status.state === "running") { armWatchdog(); return; }
        }
      } catch { /* An unavailable status endpoint cannot establish completion. */ }
      finish(Object.assign(new Error("The runtime has not confirmed completion. Reconcile this request before retrying."), { code: "GENERATION_INTERRUPTED" }));
    };
    const armWatchdog = () => {
      clearTimeout(watchdog);
      watchdog = setTimeout(() => { void checkStatus(); }, options.idleTimeoutMs ?? 120_000);
    };
    const receive = (event: WSServerMessage) => {
      if (event.type === "error" && !event.request_id && [
        "RUNTIME_CHANNEL_CLOSED", "EVENT_SEQUENCE_GAP", "TREE_RESYNC_FAILED", "RUNTIME_WORKER_FAILED",
      ].includes(event.code ?? "")) {
        void checkStatus();
        return;
      }
      if (!("request_id" in event) || event.request_id !== options.requestId) return;
      armWatchdog();
      if (event.type === "done") {
        results.set(event.sibling_index, structuredClone(event));
        options.onPartial?.(receipt());
        options.onProgress?.({ phase: "generation", completedSiblings: results.size, totalSiblings: options.expectedSiblings });
      } else if (event.type === "started" || event.type === "generation_progress") {
        options.onProgress?.(event);
      } else if (event.type === "error") {
        failure = Object.assign(new Error(event.message), { code: event.code ?? "GENERATION_FAILED" });
        if (["REQUEST_RECONCILE_REQUIRED", "REQUEST_ID_CONFLICT", "WORKER_STOP_FAILED"].includes(event.code ?? "")) finish(failure);
      } else if (event.type === "request_complete") {
        if (failure || event.state === "failed") {
          finish(failure ?? Object.assign(new Error("Generation failed"), { code: "GENERATION_FAILED" }));
        } else if (event.completed_siblings !== results.size ||
          (event.state === "completed" && results.size !== options.expectedSiblings)) {
          finish(Object.assign(new Error("The runtime did not finalize every requested result"), {
            code: "GENERATION_RESULT_INCOMPLETE",
          }));
        } else finish(undefined, event.state);
      }
    };
    unsubscribe = options.subscribe(receive);
    options.signal.addEventListener("abort", abort, { once: true });
    if (options.signal.aborted) { abort(); return; }
    try { sent = true; armWatchdog(); options.send(); }
    catch (error) { finish(asError(error)); }
  });
}

function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error("The generation request could not be sent");
}
