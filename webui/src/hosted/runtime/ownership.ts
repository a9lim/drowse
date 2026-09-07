import { randomUuid } from "./randomId";

export interface RuntimeOwnershipOptions {
  lockName?: string;
  channelName?: string;
  takeoverTimeoutMs?: number;
  onTakeoverRequest?: () => boolean | Promise<boolean>;
}

export interface RuntimeOwnership {
  readonly isOwner: boolean;
  acquire(): Promise<boolean>;
  release(): void;
  close(): void;
}

export const HOSTED_DESTRUCTIVE_LOCK = "drowse-hosted-destructive-v1";

export function withHostedDestructiveLock<T>(
  operation: () => Promise<T>,
): Promise<T> {
  if (!navigator.locks?.request) {
    return Promise.reject(new Error("The Web Locks API is unavailable"));
  }
  return navigator.locks.request(
    HOSTED_DESTRUCTIVE_LOCK,
    { mode: "exclusive" },
    async (lock) => {
      if (!lock) throw new Error("The Drowse destructive-operation lock was not acquired");
      return operation();
    },
  );
}

export function withHostedDestructiveSharedLock<T>(
  operation: () => Promise<T>,
): Promise<T> {
  if (typeof navigator === "undefined") return operation();
  if (!navigator.locks?.request) {
    return Promise.reject(new Error("The Web Locks API is unavailable"));
  }
  return navigator.locks.request(
    HOSTED_DESTRUCTIVE_LOCK,
    { mode: "shared" },
    async (lock) => {
      if (!lock) throw new Error("The Drowse destructive-operation lock was not acquired");
      return operation();
    },
  );
}

interface OwnershipMessage {
  type: "takeover";
  requesterId: string;
}

export class GpuRuntimeOwnership implements RuntimeOwnership {
  readonly ownerId = randomUuid();
  private readonly lockName: string;
  private readonly takeoverTimeoutMs: number;
  private readonly onTakeoverRequest: () => boolean | Promise<boolean>;
  private readonly channel: BroadcastChannel;
  private releaseHold: (() => void) | null = null;
  private owned = false;
  private closed = false;
  private acquisition: Promise<boolean> | null = null;
  private acquisitionController: AbortController | null = null;

  constructor(options: RuntimeOwnershipOptions = {}) {
    this.lockName = options.lockName ?? "drowse-webgpu-runtime";
    this.takeoverTimeoutMs = options.takeoverTimeoutMs ?? 10_000;
    this.onTakeoverRequest = options.onTakeoverRequest ?? (() => true);
    this.channel = new BroadcastChannel(options.channelName ?? "drowse-runtime-ownership");
    this.channel.addEventListener("message", (event: MessageEvent<OwnershipMessage>) => {
      void this.handleMessage(event.data).catch(() => {});
    });
  }

  get isOwner(): boolean {
    return this.owned;
  }

  async acquire(): Promise<boolean> {
    if (this.closed) throw new Error("The Drowse GPU ownership coordinator is closed");
    if (this.owned) return true;
    if (this.acquisition) return this.acquisition;
    const controller = new AbortController();
    const acquisition = this.acquireOnce(controller);
    this.acquisitionController = controller;
    this.acquisition = acquisition;
    try {
      return await acquisition;
    } finally {
      if (this.acquisition === acquisition) {
        this.acquisition = null;
        this.acquisitionController = null;
      }
    }
  }

  private async acquireOnce(controller: AbortController): Promise<boolean> {
    if (await this.acquireLock(true, controller.signal)) return !this.closed;
    if (this.closed) return false;
    this.channel.postMessage({ type: "takeover", requesterId: this.ownerId } satisfies OwnershipMessage);
    const timeout = setTimeout(() => controller.abort(), this.takeoverTimeoutMs);
    try {
      const acquired = await this.acquireLock(false, controller.signal);
      if (this.closed && acquired) this.release();
      return acquired && !this.closed;
    } finally {
      clearTimeout(timeout);
    }
  }

  release(): void {
    const release = this.releaseHold;
    this.releaseHold = null;
    this.owned = false;
    release?.();
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.acquisitionController?.abort();
    this.release();
    this.channel.close();
  }

  private acquireLock(ifAvailable: boolean, signal?: AbortSignal): Promise<boolean> {
    return new Promise((resolve, reject) => {
      let settled = false;
      const options: LockOptions = ifAvailable
        ? { mode: "exclusive", ifAvailable: true }
        : { mode: "exclusive", ...(signal ? { signal } : {}) };
      navigator.locks.request(
        this.lockName,
        options,
        async (lock) => {
          if (!lock || this.closed) {
            settled = true;
            resolve(false);
            return;
          }
          this.owned = true;
          let release!: () => void;
          const hold = new Promise<void>((done) => {
            release = done;
          });
          this.releaseHold = release;
          settled = true;
          resolve(true);
          await hold;
          this.releaseHold = null;
          this.owned = false;
        },
      ).catch((error: unknown) => {
        if (settled) return;
        if (error instanceof DOMException && error.name === "AbortError") {
          resolve(false);
        } else {
          reject(error);
        }
      });
    });
  }

  private async handleMessage(message: OwnershipMessage): Promise<void> {
    if (message?.type !== "takeover" || message.requesterId === this.ownerId || !this.owned) {
      return;
    }
    if (await this.onTakeoverRequest()) this.release();
  }
}
