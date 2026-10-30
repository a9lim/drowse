// THE client half of the unified background-preparation resource.
//
// The server exposes one `/instruments/{family}/preparations` job per
// family (lens: fetch | fit; sae: load | train).  Each of those four
// operations gets one slice built here, so they share one await / error /
// resume contract instead of four hand-written near-copies that drift:
//
//   * `start` posts, reflects the returned status, then fires the poll
//     loop and returns — never awaits the job to completion.
//   * the poll loop is re-entrancy-guarded, so N panels asking for the
//     same job still run one interval.
//   * `cancel` is a no-op unless the job is running and not already
//     cancelling.
//   * `check` is the mount-time resume probe: reflect the running-or-last
//     job when it is *this* operation (the two operations of a family
//     share one resource), and pick the poll loop back up if it is still
//     running.
//   * settle toasts are uniform — cancelled → info, error → sticky error,
//     otherwise the slice's success line.

import { apiInstruments } from "../runtime/services";
import { userFacingError } from "../runtime/userFacingError";
import type { InstrumentFamily } from "../types";
import type { PreparationOp, PreparationStatusJSON } from "../types";
import { pushToast } from "./toasts.svelte";

/** One preparation's mirror of the server job status. */
export interface PreparationState {
  latest: PreparationStatusJSON | null;
  running: boolean;
  /** Progress in the job's own unit (fit: prompts, train: tokens); both 0
   *  when the job reports none (fetch / load are message-only). */
  current: number;
  total: number;
  message: string | null;
  error: string | null;
  /** Poll-loop guard — one interval regardless of how many panels ask. */
  polling: boolean;
  cancelling: boolean;
}

export interface PreparationSlice {
  readonly state: PreparationState;
  /** Start the job.  Extra body fields ride the operation's request
   *  model (fit: prompts/layers, train: name/layer/tokens/width, …). */
  start(body?: Record<string, unknown>): Promise<void>;
  /** Cooperative cancel; a no-op for a non-cancellable operation. */
  cancel(): Promise<void>;
  /** Mount-time resume probe. */
  check(): Promise<void>;
  startTracked(body: Record<string, unknown>, options?: {
    onStarted?: (status: PreparationStatusJSON) => void;
    onProgress?: (status: PreparationStatusJSON) => void;
  }): Promise<PreparationStatusJSON>;
  cancelOwned(startedAt: number | null): Promise<void>;
  watchTracked(status: PreparationStatusJSON, onProgress?: (status: PreparationStatusJSON) => void): Promise<PreparationStatusJSON>;
}

export interface PreparationSliceOptions {
  /** Human name leading every toast — "J-lens fit", "SAE train", … */
  label: string;
  /** Poll interval in ms. */
  intervalMs: number;
  /** Info toast on a clean finish. */
  successMessage: string;
  /** Run after the job leaves `running` — the session / source / probe
   *  refreshes this operation's result invalidates.  Awaited before the
   *  settle toast so the panel is already consistent when it fires. */
  onSettled?: () => void | Promise<void>;
}

function describe(e: unknown): string {
  return userFacingError(e, "The model tool could not be prepared.");
}

export function createPreparationSlice(
  family: InstrumentFamily,
  operation: PreparationOp,
  { label, intervalMs, successMessage, onSettled }: PreparationSliceOptions,
): PreparationSlice {
  const state: PreparationState = $state({
    latest: null,
    running: false,
    current: 0,
    total: 0,
    message: null,
    error: null,
    polling: false,
    cancelling: false,
  });
  let pollPromise: Promise<PreparationStatusJSON> | null = null;
  const listeners = new Set<(status: PreparationStatusJSON) => void>();

  function apply(st: PreparationStatusJSON): void {
    state.latest = structuredClone(st);
    state.running = st.state === "running";
    state.current = st.progress?.current ?? 0;
    state.total = st.progress?.total ?? 0;
    state.message = st.message;
    state.error = st.error ? describe(st.error) : null;
    if (st.state !== "running") state.cancelling = false;
    for (const listener of listeners) listener(st);
  }

  async function settle(st: PreparationStatusJSON): Promise<void> {
    await onSettled?.();
    if (st.finished_at === null) return;
    if (st.message === "cancelled") {
      pushToast(`${label} cancelled`, { kind: "info" });
    } else if (state.error) {
      pushToast(`${label}: ${state.error}`, { kind: "error", ttlMs: null });
    } else {
      pushToast(successMessage, { kind: "info" });
    }
  }

  function poll(initial: PreparationStatusJSON): Promise<PreparationStatusJSON> {
    if (pollPromise) return pollPromise;
    state.polling = true;
    pollPromise = (async () => {
      try {
        let st = initial;
        for (;;) {
          if (st.state !== "running") {
            await settle(st);
            return st;
          }
          await new Promise((resolve) => setTimeout(resolve, intervalMs));
          st = await apiInstruments.preparationStatus(family);
          if (st.operation !== operation || st.started_at !== initial.started_at) {
            throw Object.assign(new Error("The preparation was replaced; read the current source status before retrying."), { code: "PREPARATION_REPLACED" });
          }
          apply(st);
        }
      } catch (e) {
        state.error = describe(e);
        throw e;
      } finally {
        state.polling = false;
        pollPromise = null;
      }
    })();
    return pollPromise;
  }

  async function begin(body: Record<string, unknown>): Promise<PreparationStatusJSON> {
    if (state.running || state.polling) throw Object.assign(new Error("This preparation is already active."), { code: "BUSY" });
    state.running = true;
    state.error = null;
    try {
      const st = await apiInstruments.startPreparation(family, { ...body, operation });
      if (st.operation !== operation) throw Object.assign(new Error("The backend returned a different preparation operation."), { code: "PREPARATION_REPLACED" });
      apply(st);
      return st;
    } catch (e) {
      state.running = false;
      state.error = describe(e);
      throw e;
    }
  }

  async function cancelOwned(startedAt: number | null): Promise<void> {
    const current = await apiInstruments.preparationStatus(family);
    if (current.operation !== operation || current.started_at !== startedAt) {
      throw Object.assign(new Error("This job no longer owns the active preparation."), { code: "PREPARATION_REPLACED" });
    }
    apply(current);
    if (current.state !== "running") return;
    if (!current.cancellable) throw Object.assign(new Error("This preparation does not support cancellation."), { code: "JOB_NOT_CANCELLABLE" });
    state.cancelling = true;
    try { apply(await apiInstruments.cancelPreparation(family)); }
    catch (error) { state.cancelling = false; throw error; }
  }

  return {
    state,

    async start(body: Record<string, unknown> = {}): Promise<void> {
      if (state.running || state.polling) return;
      try {
        const st = await begin(body);
        void poll(st).catch(() => {});
      } catch (e) {
        state.running = false;
        state.error = describe(e);
        pushToast(`${label}: ${state.error}`, { kind: "error" });
        return;
      }
    },

    async startTracked(body, options = {}): Promise<PreparationStatusJSON> {
      if (options.onProgress) listeners.add(options.onProgress);
      try {
        const st = await begin(body);
        options.onStarted?.(st);
        return await poll(st);
      } finally {
        if (options.onProgress) listeners.delete(options.onProgress);
      }
    },

    cancelOwned,

    async watchTracked(status, onProgress) {
      if (status.operation !== operation) throw Object.assign(new Error("This status belongs to another preparation operation."), { code: "PREPARATION_REPLACED" });
      if (onProgress) listeners.add(onProgress);
      try { apply(status); return await poll(status); }
      finally { if (onProgress) listeners.delete(onProgress); }
    },

    async cancel(): Promise<void> {
      if (!state.running || state.cancelling || !state.latest?.cancellable) return;
      try {
        await cancelOwned(state.latest.started_at);
        pushToast(`${label} cancelling…`, { kind: "info" });
      } catch (e) {
        state.cancelling = false;
        pushToast(`${label} cancel: ${describe(e)}`, { kind: "error" });
      }
    },

    async check(): Promise<void> {
      if (state.polling) return;
      try {
        // The two operations of a family share one preparations resource,
        // so only reflect the running-or-last job when it is this one.
        const st = await apiInstruments.preparationStatus(family);
        if (st.operation !== operation) return;
        apply(st);
        if (st.state === "running") void poll(st).catch(() => {});
      } catch {
        // A mount-time status probe is optional: with no reachable job
        // status the source section still authors and submits normally.
      }
    },
  };
}
