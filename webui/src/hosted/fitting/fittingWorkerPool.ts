import {
  BrowserFittingWorkerClient,
  type BrowserFittingWorkerClientOptions,
  type FittingWorkerRunOptions,
} from "./fittingWorkerClient";
import type { FittingWorkerJob, FittingWorkerResult } from "./workerContracts";

const MEMORY_BUDGET = 256 * 1024 * 1024;

export class BrowserFittingWorkerPool {
  readonly parallelism = 2;
  private readonly clients: BrowserFittingWorkerClient[];
  private readonly active = new Map<BrowserFittingWorkerClient, number>();
  private readonly waiters = new Set<() => void>();
  private disposed = false;

  constructor(options: BrowserFittingWorkerClientOptions = {}) {
    this.clients = Array.from({ length: this.parallelism }, () => new BrowserFittingWorkerClient(options));
  }

  async run(job: FittingWorkerJob, options: FittingWorkerRunOptions = {}): Promise<FittingWorkerResult> {
    const bytes = fittingJobWorkingBytes(job);
    while (true) {
      options.signal?.throwIfAborted();
      if (this.disposed) throw new DOMException("Fitting worker pool was disposed", "AbortError");
      const used = [...this.active.values()].reduce((sum, value) => sum + value, 0);
      const client = this.clients.find((candidate) => !this.active.has(candidate));
      if (client && (this.active.size === 0 || used + bytes <= MEMORY_BUDGET)) {
        this.active.set(client, bytes);
        try {
          return await client.run(job, options);
        } finally {
          this.active.delete(client);
          for (const wake of this.waiters) wake();
        }
      }
      await new Promise<void>((resolve) => {
        const wake = () => {
          this.waiters.delete(wake);
          options.signal?.removeEventListener("abort", wake);
          resolve();
        };
        this.waiters.add(wake);
        options.signal?.addEventListener("abort", wake, { once: true });
      });
    }
  }

  dispose(): void {
    this.disposed = true;
    for (const client of this.clients) client.dispose();
    for (const wake of this.waiters) wake();
  }
}

export function fittingJobWorkingBytes(job: FittingWorkerJob): number {
  let payload = 0;
  let dimension = 0;
  const buffers = new Set<ArrayBufferLike>();
  const visit = (value: unknown): void => {
    if (ArrayBuffer.isView(value)) {
      if (!buffers.has(value.buffer)) {
        buffers.add(value.buffer);
        payload += value.buffer.byteLength;
      }
    } else if (value !== null && typeof value === "object") {
      for (const [key, item] of Object.entries(value)) {
        if ((key === "nodeCount" || key === "rows") && typeof item === "number") dimension = Math.max(dimension, Math.min(4096, item));
        visit(item);
      }
    }
  };
  visit(job);
  if (job.operation === "correlations" && job.source.kind === "inline") {
    dimension = Math.max(dimension, Math.min(4096, job.source.columns));
  }
  if ("source" in job && job.source.kind === "activation_spool") {
    // Spool readers may materialize the maximum admitted activation matrix.
    return MEMORY_BUDGET;
  }
  return 8 * 1024 * 1024 + 4 * payload + 8 * dimension * dimension * 12;
}
