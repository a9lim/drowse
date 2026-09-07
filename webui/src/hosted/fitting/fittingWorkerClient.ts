import {
  FITTING_WORKER_PROTOCOL_VERSION,
  isFittingWorkerResponse,
  type FittingWorkerJob,
  type FittingWorkerRequest,
  type FittingWorkerResult,
  type FittingWorkerStage,
} from "./workerContracts";
import { randomUuid } from "../runtime/randomId";

interface WorkerPort {
  onmessage: ((event: MessageEvent<unknown>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  onmessageerror: ((event: MessageEvent<unknown>) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
  terminate(): void;
}

export interface BrowserFittingWorkerClientOptions {
  workerFactory?: () => WorkerPort;
  requestTimeoutMs?: number;
  createRequestId?: () => string;
}

export interface FittingWorkerRunOptions {
  signal?: AbortSignal;
  onProgress?: (stage: FittingWorkerStage) => void;
  transferOwnership?: readonly ArrayBufferView[];
}

export class FittingWorkerClientError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly recoverable: boolean,
  ) {
    super(message);
    this.name = "FittingWorkerClientError";
  }
}

export class BrowserFittingWorkerClient {
  private readonly workerFactory: () => WorkerPort;
  private readonly requestTimeoutMs: number;
  private readonly createRequestId: () => string;
  private worker: WorkerPort | null = null;
  private active: PendingRun | null = null;
  private readonly queue: PendingRun[] = [];

  constructor(options: BrowserFittingWorkerClientOptions = {}) {
    this.workerFactory = options.workerFactory ?? defaultWorkerFactory;
    this.requestTimeoutMs = positiveTimeout(options.requestTimeoutMs, 5 * 60_000);
    this.createRequestId = options.createRequestId ?? defaultRequestId;
  }

  run(
    job: FittingWorkerJob,
    options: FittingWorkerRunOptions = {},
  ): Promise<FittingWorkerResult> {
    if (options.signal?.aborted) return Promise.reject(cancelled());
    const requestId = this.createRequestId();
    const request: FittingWorkerRequest = {
      protocolVersion: FITTING_WORKER_PROTOCOL_VERSION,
      requestId,
      kind: "run",
      job,
    };
    let transfer: Transferable[];
    try {
      transfer = fittingTransferList(job, options.transferOwnership);
    } catch (error) {
      return Promise.reject(error);
    }

    return new Promise((resolve, reject) => {
      const pending: PendingRun = {
        request,
        transfer,
        signal: options.signal,
        onProgress: options.onProgress,
        resolve,
        reject,
        onAbort: () => this.abort(pending),
        timeout: null,
        settled: false,
      };
      options.signal?.addEventListener("abort", pending.onAbort, { once: true });
      this.queue.push(pending);
      this.pump();
    });
  }

  dispose(): void {
    const active = this.active;
    if (active !== null) {
      this.finish(active, () => active.reject(cancelled()), true);
    } else {
      this.resetWorker();
    }
    for (const pending of this.queue.splice(0)) {
      this.settle(pending, () => pending.reject(cancelled()));
    }
  }

  private pump(): void {
    if (this.active !== null) return;
    const pending = this.queue.shift();
    if (pending === undefined) return;
    if (pending.signal?.aborted) {
      this.settle(pending, () => pending.reject(cancelled()));
      queueMicrotask(() => this.pump());
      return;
    }
    let worker: WorkerPort;
    try {
      worker = this.worker ?? this.startWorker();
    } catch (error) {
      this.settle(pending, () => pending.reject(clientFailure(
        "FITTING_WORKER_START_FAILED",
        "Unable to start the browser fitting worker",
        error,
      )));
      queueMicrotask(() => this.pump());
      return;
    }
    this.active = pending;
    pending.timeout = setTimeout(() => this.finish(
      pending,
      () => pending.reject(new FittingWorkerClientError(
        "FITTING_WORKER_TIMEOUT",
        "The browser fitting worker did not finish before its safety timeout",
        true,
      )),
      true,
    ), this.requestTimeoutMs);
    try {
      worker.postMessage(pending.request, pending.transfer);
    } catch (error) {
      this.finish(pending, () => pending.reject(clientFailure(
        "FITTING_WORKER_POST_FAILED",
        "Unable to send the fitting job to the browser worker",
        error,
      )), true);
    }
  }

  private startWorker(): WorkerPort {
    const worker = this.workerFactory();
    this.worker = worker;
    worker.onmessage = (event) => this.message(worker, event.data);
    worker.onerror = (event) => this.crash(worker, event.message);
    worker.onmessageerror = () => this.messageError(worker);
    return worker;
  }

  private message(worker: WorkerPort, data: unknown): void {
    if (worker !== this.worker) return;
    const pending = this.active;
    if (
      pending === null ||
      !isFittingWorkerResponse(data) ||
      data.requestId !== pending.request.requestId
    ) {
      if (pending === null) {
        this.resetWorker(worker);
        return;
      }
      this.finish(pending, () => pending.reject(new FittingWorkerClientError(
        "FITTING_WORKER_PROTOCOL_ERROR",
        "The browser fitting worker returned an invalid response",
        false,
      )), true);
      return;
    }
    if (data.kind === "progress") {
      try {
        pending.onProgress?.(data.stage);
      } catch (error) {
        this.finish(pending, () => pending.reject(clientFailure(
          "FITTING_PROGRESS_HANDLER_FAILED",
          "The browser fitting progress handler failed",
          error,
          false,
        )), true);
      }
      return;
    }
    if (data.kind === "error") {
      this.finish(pending, () => pending.reject(new FittingWorkerClientError(
        data.error.code,
        data.error.message,
        data.error.recoverable,
      )));
      return;
    }
    if (data.result.operation !== pending.request.job.operation) {
      this.finish(pending, () => pending.reject(new FittingWorkerClientError(
        "FITTING_WORKER_PROTOCOL_ERROR",
        `The browser fitting worker returned ${data.result.operation} for ${pending.request.job.operation}`,
        false,
      )), true);
      return;
    }
    this.finish(pending, () => pending.resolve(data.result));
  }

  private crash(worker: WorkerPort, message: string): void {
    if (worker !== this.worker) return;
    const pending = this.active;
    if (pending === null) {
      this.resetWorker(worker);
      return;
    }
    this.finish(pending, () => pending.reject(new FittingWorkerClientError(
      "FITTING_WORKER_CRASHED",
      message || "The browser fitting worker stopped unexpectedly",
      true,
    )), true);
  }

  private messageError(worker: WorkerPort): void {
    if (worker !== this.worker) return;
    const pending = this.active;
    if (pending === null) {
      this.resetWorker(worker);
      return;
    }
    this.finish(pending, () => pending.reject(new FittingWorkerClientError(
      "FITTING_WORKER_PROTOCOL_ERROR",
      "The browser fitting worker returned an unreadable message",
      false,
    )), true);
  }

  private abort(pending: PendingRun): void {
    if (pending.settled) return;
    if (this.active === pending) {
      this.finish(pending, () => pending.reject(cancelled()), true);
      return;
    }
    const index = this.queue.indexOf(pending);
    if (index !== -1) this.queue.splice(index, 1);
    this.settle(pending, () => pending.reject(cancelled()));
  }

  private finish(pending: PendingRun, callback: () => void, recycle = false): void {
    if (this.active !== pending || pending.settled) return;
    this.active = null;
    if (recycle) this.resetWorker();
    this.settle(pending, callback);
    queueMicrotask(() => this.pump());
  }

  private settle(pending: PendingRun, callback: () => void): void {
    if (pending.settled) return;
    pending.settled = true;
    if (pending.timeout !== null) clearTimeout(pending.timeout);
    pending.signal?.removeEventListener("abort", pending.onAbort);
    callback();
  }

  private resetWorker(expected?: WorkerPort): void {
    const worker = this.worker;
    if (worker === null || expected !== undefined && worker !== expected) return;
    this.worker = null;
    worker.onmessage = null;
    worker.onerror = null;
    worker.onmessageerror = null;
    worker.terminate();
  }
}

interface PendingRun {
  request: FittingWorkerRequest;
  transfer: Transferable[];
  signal: AbortSignal | undefined;
  onProgress: ((stage: FittingWorkerStage) => void) | undefined;
  resolve: (result: FittingWorkerResult) => void;
  reject: (error: unknown) => void;
  onAbort: () => void;
  timeout: ReturnType<typeof setTimeout> | null;
  settled: boolean;
}

function fittingTransferList(
  job: FittingWorkerJob,
  views: readonly ArrayBufferView[] | undefined,
): Transferable[] {
  if (views === undefined || views.length === 0) return [];
  const jobBuffers = new Set<ArrayBuffer>();
  collectArrayBuffers(job, jobBuffers);
  const transfer = new Set<ArrayBuffer>();
  for (const view of views) {
    if (!ArrayBuffer.isView(view) || !(view.buffer instanceof ArrayBuffer)) {
      throw invalidTransfer("Fitting transfer ownership requires ArrayBuffer-backed views");
    }
    if (view.byteOffset !== 0 || view.byteLength !== view.buffer.byteLength) {
      throw invalidTransfer("Fitting transfer ownership requires a whole-buffer view");
    }
    if (!jobBuffers.has(view.buffer)) {
      throw invalidTransfer("Fitting transfer ownership must reference a buffer in the submitted job");
    }
    transfer.add(view.buffer);
  }
  return [...transfer];
}

function collectArrayBuffers(value: unknown, buffers: Set<ArrayBuffer>): void {
  if (ArrayBuffer.isView(value)) {
    if (value.buffer instanceof ArrayBuffer) buffers.add(value.buffer);
    return;
  }
  if (Array.isArray(value)) {
    for (const item of value) collectArrayBuffers(item, buffers);
    return;
  }
  if (typeof value === "object" && value !== null) {
    for (const item of Object.values(value)) collectArrayBuffers(item, buffers);
  }
}

function invalidTransfer(message: string): FittingWorkerClientError {
  return new FittingWorkerClientError(
    "FITTING_INVALID_TRANSFER_OWNERSHIP",
    message,
    false,
  );
}

function defaultWorkerFactory(): WorkerPort {
  return new Worker(new URL("./fitting.worker.ts", import.meta.url), {
    type: "module",
    name: "drowse-fitting",
  });
}

function defaultRequestId(): string {
  return randomUuid();
}

function positiveTimeout(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) && value! > 0 ? value! : fallback;
}

function cancelled(): FittingWorkerClientError {
  return new FittingWorkerClientError(
    "FITTING_CANCELLED",
    "The browser fitting job was cancelled",
    true,
  );
}

function clientFailure(
  code: string,
  context: string,
  error: unknown,
  recoverable = true,
): FittingWorkerClientError {
  const detail = error instanceof Error ? error.message : String(error);
  return new FittingWorkerClientError(code, `${context}: ${detail}`, recoverable);
}
