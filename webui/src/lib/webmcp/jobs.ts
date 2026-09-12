export type JobState = "queued" | "running" | "cancelling" | "completed" | "cancelled" | "failed" | "interrupted";

export interface JobError {
  code: string;
  message: string;
  details?: unknown;
  status?: number;
  body?: unknown;
}

export interface JobSnapshot {
  id: string;
  kind: string;
  state: JobState;
  requestId?: string;
  cancellable: boolean;
  result?: unknown;
  error?: JobError;
  progress?: unknown;
  recovery?: unknown;
  persistedDetailOmissions?: string[];
  createdAt: number;
  startedAt: number | null;
  finishedAt: number | null;
}

interface JobRecord {
  snapshot: JobSnapshot;
  controller: AbortController;
  cancel?: () => Promise<void> | void;
  cancelFailed?: boolean;
}

const terminal = (state: JobState): boolean =>
  state === "completed" || state === "cancelled" || state === "failed" || state === "interrupted";

export class JobRegistry {
  private readonly records = new Map<string, JobRecord>();
  private readonly requests = new Map<string, string>();
  private persistence: { storage: Pick<Storage, "getItem" | "setItem">; key: string } | null = null;
  persistenceError: string | null = null;

  constructor(private readonly retention = 100) {}

  start(
    kind: string,
    run: (job: { id: string; signal: AbortSignal; progress: (value: unknown) => void }) => Promise<unknown>,
    options: { requestId?: string; cancel?: () => Promise<void> | void; cancellable?: boolean; recovery?: unknown } = {},
  ): JobSnapshot {
    const priorId = options.requestId === undefined ? undefined : this.requests.get(options.requestId);
    if (priorId !== undefined) {
      const prior = this.require(priorId);
      if (prior.snapshot.kind !== kind) throw new Error("The request ID already belongs to another operation");
      return this.copy(prior);
    }
    const id = `job-${crypto.randomUUID()}`;
    const record: JobRecord = {
      snapshot: {
        id, kind, state: "queued", cancellable: options.cancellable !== false,
        createdAt: Date.now(), startedAt: null, finishedAt: null,
        ...(options.requestId === undefined ? {} : { requestId: options.requestId }),
        ...(options.recovery === undefined ? {} : { recovery: cloneValue(options.recovery) }),
      },
      controller: new AbortController(),
      cancel: options.cancel,
    };
    this.records.set(id, record);
    if (options.requestId !== undefined) this.requests.set(options.requestId, id);
    this.persist();
    void Promise.resolve().then(async () => {
      if (terminal(record.snapshot.state)) return;
      record.snapshot.state = "running";
      record.snapshot.startedAt = Date.now();
      this.persist();
      try {
        const result = await run({ id, signal: record.controller.signal, progress: (value) => {
          if (!terminal(record.snapshot.state)) record.snapshot.progress = cloneValue(value);
        } });
        if (terminal(record.snapshot.state)) return;
        record.snapshot.result = cloneValue(result);
        const completion = result as { state?: unknown } | null;
        record.snapshot.state = completion?.state === "completed" ? "completed"
          : completion?.state === "cancelled" || (record.controller.signal.aborted && !record.cancelFailed) ? "cancelled" : "completed";
      } catch (error) {
        if (terminal(record.snapshot.state)) return;
        const detail = jobError(error);
        const partial = (error as { partialResult?: unknown } | null)?.partialResult;
        if (partial !== undefined) record.snapshot.result = cloneValue(partial);
        if (detail.code === "GENERATION_INTERRUPTED") {
          record.snapshot.state = "interrupted";
          record.snapshot.error = detail;
        } else if (detail.code === "WORKER_STOP_FAILED") {
          record.snapshot.state = "failed";
          record.snapshot.error = detail;
        } else if (!record.cancelFailed && (record.controller.signal.aborted || (error instanceof Error && error.name === "AbortError"))) {
          record.snapshot.state = "cancelled";
        } else {
          record.snapshot.state = "failed";
          record.snapshot.error = detail;
        }
      } finally {
        if (terminal(record.snapshot.state)) record.snapshot.finishedAt ??= Date.now();
        this.prune();
      }
    });
    return this.copy(record);
  }

  get(id: string): JobSnapshot { return this.copy(this.require(id)); }

  list(): JobSnapshot[] { return [...this.records.values()].map((record) => this.copy(record)); }

  checkpoint(id: string, result: unknown): void {
    this.require(id).snapshot.result = cloneValue(result);
    this.persist();
  }

  setCancellable(id: string, cancellable: boolean): void {
    this.require(id).snapshot.cancellable = cancellable;
    this.persist();
  }

  setRecovery(id: string, recovery: unknown): void {
    this.require(id).snapshot.recovery = cloneValue(recovery);
    this.persist();
  }

  reconcile(id: string, update: Pick<JobSnapshot, "state"> & Partial<Pick<JobSnapshot, "result" | "error" | "progress">>): JobSnapshot {
    const record = this.require(id);
    Object.assign(record.snapshot, cloneValue(update));
    if (terminal(update.state)) record.snapshot.finishedAt = Date.now();
    if (update.state === "completed") delete record.snapshot.error;
    this.prune();
    return this.copy(record);
  }

  attachPersistence(storage: Pick<Storage, "getItem" | "setItem">, key: string): void {
    this.persistence = { storage, key };
    try {
      const saved: unknown = JSON.parse(storage.getItem(key) ?? "[]");
      if (!Array.isArray(saved)) throw new Error("Invalid operation history");
      for (const value of saved) {
        if (!value || typeof value.id !== "string" || typeof value.kind !== "string" || typeof value.createdAt !== "number") continue;
        if (this.records.has(value.id)) continue;
        const snapshot = value as JobSnapshot;
        if (!terminal(snapshot.state)) {
          snapshot.state = "interrupted";
          snapshot.finishedAt = Date.now();
          snapshot.error = { code: "PAGE_RELOADED", message: "The page closed before completion was recorded. Reconcile the operation before retrying." };
        }
        this.records.set(snapshot.id, { snapshot, controller: new AbortController() });
        if (snapshot.requestId !== undefined) this.requests.set(snapshot.requestId, snapshot.id);
      }
      this.prune();
    } catch { this.persistenceError = "Operation history could not be restored; previous requests must not be assumed complete."; }
  }

  async cancel(id: string): Promise<JobSnapshot> {
    const record = this.require(id);
    if (terminal(record.snapshot.state) || record.snapshot.state === "cancelling") return this.copy(record);
    if (!record.snapshot.cancellable) {
      throw Object.assign(new Error("This operation does not support scoped cancellation"), { code: "JOB_NOT_CANCELLABLE" });
    }
    const queued = record.snapshot.state === "queued";
    record.snapshot.state = queued ? "cancelled" : "cancelling";
    record.controller.abort(new DOMException("The operation was cancelled", "AbortError"));
    if (queued) record.snapshot.finishedAt = Date.now();
    this.persist();
    if (!queued && record.cancel) {
      try { await record.cancel(); }
      catch (error) {
        record.cancelFailed = true;
        record.snapshot.error = { ...jobError(error), code: "CANCELLATION_FAILED" };
        if (record.snapshot.state === "cancelled") record.snapshot.state = "failed";
        else if (record.snapshot.state === "cancelling") record.snapshot.state = "running";
      }
    }
    this.prune();
    return this.copy(record);
  }

  interrupt(reason: string, matches: (job: JobSnapshot) => boolean = () => true): void {
    for (const record of this.records.values()) {
      if (terminal(record.snapshot.state) || !matches(record.snapshot)) continue;
      record.snapshot.state = "interrupted";
      record.snapshot.error = { code: "WORKSPACE_INTERRUPTED", message: reason };
      record.snapshot.finishedAt = Date.now();
      record.controller.abort(new DOMException(reason, "AbortError"));
    }
    this.prune();
  }

  private require(id: string): JobRecord {
    const record = this.records.get(id);
    if (!record) throw Object.assign(new Error("This job is unknown or no longer retained"), { code: "JOB_NOT_FOUND" });
    return record;
  }

  private copy(record: JobRecord): JobSnapshot { return structuredClone(record.snapshot); }

  private prune(): void {
    const finished = [...this.records.values()].filter((record) => terminal(record.snapshot.state));
    for (const record of finished.slice(0, Math.max(0, finished.length - this.retention))) {
      this.records.delete(record.snapshot.id);
      if (record.snapshot.requestId !== undefined) this.requests.delete(record.snapshot.requestId);
    }
    this.persist();
  }

  private persist(): void {
    if (!this.persistence) return;
    try {
      const snapshots = this.list().map(compactSnapshot);
      let serialized = JSON.stringify(snapshots);
      while (serialized.length > 3 * 1024 * 1024) {
        const oldest = snapshots.findIndex(snapshot => terminal(snapshot.state));
        if (oldest === -1) break;
        snapshots.splice(oldest, 1);
        serialized = JSON.stringify(snapshots);
      }
      if (serialized.length > 4 * 1024 * 1024) throw new Error("Operation ownership records exceed storage budget");
      this.persistence.storage.setItem(this.persistence.key, serialized);
      this.persistenceError = null;
    } catch { this.persistenceError = "Operation history could not be saved. Recovery after closing this page may be incomplete."; }
  }
}

function compactSnapshot(snapshot: JobSnapshot): JobSnapshot {
  const omitted = [...(snapshot.persistedDetailOmissions ?? [])];
  const visit = (value: unknown, path: string, key = ""): unknown => {
    if (["measurements", "hidden_states", "per_layer_scores", "top_alts", "raw_token_ids", "thinking_tokens"].includes(key) || (key === "tokens" && Array.isArray(value))) {
      omitted.push(path);
      return undefined;
    }
    if (typeof value === "string" && value.length > 32768) {
      omitted.push(path);
      return `${value.slice(0, 4096)}\n[Persisted preview truncated; retrieve the complete output from its conversation node or artifact.]`;
    }
    if (Array.isArray(value)) {
      if (value.length > 64) omitted.push(path);
      return value.slice(0, 64).map((row, index) => visit(row, `${path}[${index}]`));
    }
    if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([name, child]) => [name, visit(child, `${path}.${name}`, name)]).filter(([, child]) => child !== undefined));
    return value;
  };
  const compact = visit(snapshot, "job") as JobSnapshot;
  if (omitted.length) compact.persistedDetailOmissions = [...new Set(omitted)].slice(0, 128);
  return compact;
}

function jobError(error: unknown): JobError {
  const value = error as { code?: unknown; message?: unknown; details?: unknown; status?: unknown; body?: unknown } | null;
  return {
    code: typeof value?.code === "string" ? value.code : "OPERATION_FAILED",
    message: typeof value?.message === "string" ? value.message : "The operation failed",
    ...(value?.details === undefined ? {} : { details: cloneValue(value.details) }),
    ...(typeof value?.status === "number" ? { status: value.status } : {}),
    ...(value?.body === undefined ? {} : { body: cloneValue(value.body) }),
  };
}

function cloneValue(value: unknown): unknown {
  try { return structuredClone(value); }
  catch { return JSON.parse(JSON.stringify(value)); }
}
