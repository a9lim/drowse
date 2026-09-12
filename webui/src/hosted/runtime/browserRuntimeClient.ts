import { RuntimeRequestJournal, replayRequest } from "../../lib/runtime/requestJournal";
import type {
  RuntimeClient,
  RuntimeEventChannel,
  RuntimeEventChannelState,
  RuntimeFailure,
  RuntimeInstrumentsService,
  RuntimeManifoldsService,
  RuntimeProfilesService,
  RuntimeProbesService,
  RuntimeProgressEvent,
  RuntimeServiceRequest,
  RuntimeSessionsService,
  RuntimeTemplatesService,
  RuntimeTreeService,
  WorkerCommandName,
  WorkerEventEnvelope,
  WorkerScopedFailure,
  WorkerMessage,
  WorkerRequest,
} from "../../lib/runtime/contracts";
import {
  isRuntimeServiceMethod,
  RUNTIME_PROTOCOL_VERSION,
} from "../../lib/runtime/contracts";
import { ApiError } from "../../lib/runtime/errors";
import type { WSClientMessage, WSServerMessage } from "../../lib/types";

export interface WorkerLike {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  addEventListener(type: "message", listener: (event: MessageEvent<unknown>) => void): void;
  addEventListener(type: "error", listener: (event: ErrorEvent) => void): void;
  addEventListener(type: "messageerror", listener: (event: MessageEvent<unknown>) => void): void;
  removeEventListener(type: "message", listener: (event: MessageEvent<unknown>) => void): void;
  removeEventListener(type: "error", listener: (event: ErrorEvent) => void): void;
  removeEventListener(type: "messageerror", listener: (event: MessageEvent<unknown>) => void): void;
  terminate(): void;
}

interface PendingRequest {
  resolve(value: unknown): void;
  reject(error: Error): void;
  timeout: ReturnType<typeof setTimeout> | null;
  timeoutMs: number;
  timeoutMessage: string;
  onProgress?: (event: RuntimeProgressEvent) => void;
}

export interface WorkerRpcOptions {
  requestTimeoutMs?: number;
  longOperationIdleTimeoutMs?: number;
}

export class WorkerRpcTransport {
  private readonly worker: WorkerLike;
  private readonly requestTimeoutMs: number;
  private readonly longOperationIdleTimeoutMs: number;
  private readonly pending = new Map<string, PendingRequest>();
  private readonly eventListeners = new Set<(event: WorkerEventEnvelope) => void>();
  private readonly failureListeners = new Set<(reason: string) => void>();
  private lastSequence = 0;
  private eventsPaused = false;
  private sequenceBarrierActive = false;
  private closed = false;
  private nextRequestId = 1;

  constructor(worker: WorkerLike, options: WorkerRpcOptions = {}) {
    this.worker = worker;
    this.requestTimeoutMs = options.requestTimeoutMs ?? 60_000;
    this.longOperationIdleTimeoutMs = options.longOperationIdleTimeoutMs ?? 0;
    this.worker.addEventListener("message", this.onMessage);
    this.worker.addEventListener("error", this.onWorkerError);
    this.worker.addEventListener("messageerror", this.onMessageError);
  }

  call<T>(
    command: WorkerCommandName,
    payload: unknown,
    options: {
      timeoutMs?: number;
      onProgress?: (event: RuntimeProgressEvent) => void;
    } = {},
  ): Promise<T> {
    if (this.closed) return Promise.reject(new Error("Runtime worker is closed"));
    const requestId = `rpc-${this.nextRequestId++}`;
    const request = {
      protocolVersion: RUNTIME_PROTOCOL_VERSION,
      requestId,
      command,
      payload,
    } as WorkerRequest;
    return new Promise<T>((resolve, reject) => {
      const timeoutMs = options.timeoutMs ?? commandTimeoutMs(
        command,
        this.requestTimeoutMs,
        this.longOperationIdleTimeoutMs,
      );
      const timeoutMessage = NON_EXPIRING_COMMANDS.has(command) && options.timeoutMs === undefined
        ? `Runtime worker request ${command} made no progress for ${Math.ceil(timeoutMs / 1_000)} seconds`
        : `Runtime worker request ${command} timed out`;
      const timeout = this.armTimeout(timeoutMs, timeoutMessage);
      this.pending.set(requestId, {
        resolve: (value) => resolve(value as T),
        reject,
        timeout,
        timeoutMs,
        timeoutMessage,
        onProgress: options.onProgress,
      });
      try {
        this.worker.postMessage(request);
      } catch (error) {
        this.pending.delete(requestId);
        if (timeout) clearTimeout(timeout);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  subscribe(listener: (event: WorkerEventEnvelope) => void): () => void {
    this.eventListeners.add(listener);
    return () => this.eventListeners.delete(listener);
  }

  subscribeFailure(listener: (reason: string) => void): () => void {
    this.failureListeners.add(listener);
    return () => this.failureListeners.delete(listener);
  }

  resumeEvents(): void {
    this.eventsPaused = false;
  }

  setSequenceBarrierActive(active: boolean): void {
    this.sequenceBarrierActive = active;
    if (!active) this.eventsPaused = false;
  }

  dispose(): void {
    if (this.closed) return;
    this.closed = true;
    this.worker.removeEventListener("message", this.onMessage);
    this.worker.removeEventListener("error", this.onWorkerError);
    this.worker.removeEventListener("messageerror", this.onMessageError);
    this.worker.terminate();
    for (const pending of this.pending.values()) {
      if (pending.timeout) clearTimeout(pending.timeout);
      pending.reject(new Error("Runtime worker was disposed"));
    }
    this.pending.clear();
    this.eventListeners.clear();
    this.failureListeners.clear();
  }

  terminateWithFailure(reason: string): void {
    if (this.closed) return;
    this.closed = true;
    this.worker.removeEventListener("message", this.onMessage);
    this.worker.removeEventListener("error", this.onWorkerError);
    this.worker.removeEventListener("messageerror", this.onMessageError);
    this.worker.terminate();
    for (const pending of this.pending.values()) {
      if (pending.timeout) clearTimeout(pending.timeout);
      pending.reject(new Error(reason));
    }
    this.pending.clear();
    for (const listener of this.failureListeners) listener(reason);
    this.eventListeners.clear();
    this.failureListeners.clear();
  }

  private readonly onWorkerError = (event: ErrorEvent): void => {
    this.terminateWithFailure(event.message || "The Drowse runtime worker failed");
  };

  private readonly onMessageError = (): void => {
    this.terminateWithFailure("The Drowse runtime worker sent an unreadable message");
  };

  private readonly onMessage = (event: MessageEvent<unknown>): void => {
    if (!validWorkerMessage(event.data)) {
      this.terminateWithFailure("The Drowse runtime worker violated its message protocol");
      return;
    }
    const message = event.data;
    if (message.kind === "response") {
      const response = message as Extract<WorkerMessage, { kind: "response" }>;
      const pending = this.pending.get(response.requestId);
      if (!pending) return;
      this.pending.delete(response.requestId);
      if (pending.timeout) clearTimeout(pending.timeout);
      if (response.ok) {
        pending.resolve(response.result);
      } else {
        const failure = workerFailure(response.error);
        pending.reject(failure);
        if (response.error.code === "RUNTIME_CLEANUP_FAILED") {
          this.terminateWithFailure(response.error.message);
        }
      }
      return;
    }
    if (message.kind !== "event") return;
    const runtimeEvent = message as WorkerEventEnvelope;
    if (runtimeEvent.requestId) this.refreshTimeout(runtimeEvent.requestId);
    if (
      runtimeEvent.event === "error" &&
      runtimeEvent.payload &&
      typeof runtimeEvent.payload === "object" &&
      (runtimeEvent.payload as Partial<WorkerScopedFailure>).failure?.code ===
        "RUNTIME_CLEANUP_FAILED"
    ) {
      const failure = (runtimeEvent.payload as WorkerScopedFailure).failure;
      this.terminateWithFailure(failure.message);
      return;
    }
    const expectedSequence = this.lastSequence + 1;
    if (
      this.sequenceBarrierActive && !this.eventsPaused &&
      runtimeEvent.sequence !== expectedSequence
    ) {
      this.eventsPaused = true;
      const gap: WorkerEventEnvelope = {
        protocolVersion: RUNTIME_PROTOCOL_VERSION,
        kind: "event",
        requestId: null,
        sequence: runtimeEvent.sequence,
        generationId: runtimeEvent.generationId,
        event: "error",
        payload: {
          scope: "transport",
          failure: {
            code: "EVENT_SEQUENCE_GAP",
            message: `Expected worker event ${expectedSequence}, received ${runtimeEvent.sequence}`,
            recoverable: true,
            status: 409,
          },
        } satisfies WorkerScopedFailure,
      };
      for (const listener of this.eventListeners) listener(gap);
      this.lastSequence = runtimeEvent.sequence;
      if (isSnapshotOrderedEvent(runtimeEvent.event)) return;
    }
    this.lastSequence = runtimeEvent.sequence;
    if (runtimeEvent.requestId) {
      const pending = this.pending.get(runtimeEvent.requestId);
      if (pending?.onProgress && runtimeEvent.event === "progress") {
        try {
          pending.onProgress({ event: "progress", data: runtimeEvent.payload });
        } catch {}
      }
    }
    if (this.eventsPaused && isSnapshotOrderedEvent(runtimeEvent.event)) return;
    for (const listener of this.eventListeners) listener(runtimeEvent);
  };

  private armTimeout(
    timeoutMs: number,
    timeoutMessage: string,
  ): ReturnType<typeof setTimeout> | null {
    return timeoutMs > 0
      ? setTimeout(() => this.terminateWithFailure(timeoutMessage), timeoutMs)
      : null;
  }

  private refreshTimeout(requestId: string): void {
    const pending = this.pending.get(requestId);
    if (!pending || pending.timeoutMs <= 0) return;
    if (pending.timeout) clearTimeout(pending.timeout);
    pending.timeout = this.armTimeout(pending.timeoutMs, pending.timeoutMessage);
  }
}

function isSnapshotOrderedEvent(event: WorkerEventEnvelope["event"]): boolean {
  return event === "started" || event === "generation_progress" || event === "token" || event === "done" ||
    event === "tree_mutated";
}

function validWorkerMessage(value: unknown): value is WorkerMessage {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const record = value as Record<string, unknown>;
  if (record.protocolVersion !== RUNTIME_PROTOCOL_VERSION) return false;
  if (record.kind === "response") {
    if (!validId(record.requestId) || typeof record.ok !== "boolean") return false;
    return record.ok ? "result" in record : validFailure(record.error);
  }
  if (record.kind !== "event") return false;
  return (
    (record.requestId === null || validId(record.requestId)) &&
    Number.isSafeInteger(record.sequence) &&
    (record.sequence as number) > 0 &&
    (record.generationId === null || validId(record.generationId)) &&
    typeof record.event === "string" &&
    WORKER_EVENTS.has(record.event) &&
    "payload" in record
  );
}

function validFailure(value: unknown): value is RuntimeFailure {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const failure = value as Record<string, unknown>;
  return validId(failure.code) && typeof failure.message === "string" &&
    typeof failure.recoverable === "boolean" &&
    Number.isInteger(failure.status) && (failure.status as number) >= 100 &&
    (failure.status as number) <= 599;
}

function validId(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= 128;
}

const WORKER_EVENTS = new Set([
  "status", "progress", "started", "generation_progress", "token", "done", "tree_mutated",
  "device_lost", "error",
]);

export class BrowserRuntimeClient implements RuntimeClient {
  readonly mode = "browser" as const;
  readonly sessions: RuntimeSessionsService;
  readonly profiles: RuntimeProfilesService;
  readonly probes: RuntimeProbesService;
  readonly manifolds: RuntimeManifoldsService;
  readonly templates: RuntimeTemplatesService;
  readonly tree: RuntimeTreeService;
  readonly instruments: RuntimeInstrumentsService;
  readonly events: RuntimeEventChannel;
  private readonly transport: WorkerRpcTransport;
  private readonly ownsTransport: boolean;
  private readonly eventChannel: BrowserRuntimeEventChannel;

  constructor(transport: WorkerRpcTransport, options: { ownsTransport?: boolean } = {}) {
    this.transport = transport;
    this.ownsTransport = options.ownsTransport ?? true;
    const eventChannel = new BrowserRuntimeEventChannel(transport);
    this.eventChannel = eventChannel;
    this.sessions = serviceProxy(transport, "sessions");
    this.profiles = serviceProxy(transport, "profiles");
    this.probes = serviceProxy(transport, "probes");
    this.manifolds = serviceProxy(transport, "manifolds");
    this.templates = serviceProxy(transport, "templates");
    this.tree = serviceProxy(transport, "tree");
    this.instruments = serviceProxy(transport, "instruments");
    this.events = eventChannel;
  }

  async dispose(): Promise<void> {
    if (!this.ownsTransport) {
      this.eventChannel.dispose();
      return;
    }
    try {
      await this.transport.call("unload", undefined);
    } catch {
      // Disposal still has to close a failed or already-terminated worker.
    } finally {
      this.eventChannel.dispose();
      if (this.ownsTransport) this.transport.dispose();
    }
  }
}

class BrowserRuntimeEventChannel implements RuntimeEventChannel {
  private readonly journal = new RuntimeRequestJournal();

  async requestStatus(id: string) { return this.journal.get(id); }
  private readonly transport: WorkerRpcTransport;
  private readonly listeners = new Set<(message: WSServerMessage) => void>();
  private readonly stateListeners = new Set<(state: RuntimeEventChannelState) => void>();
  private openState = false;
  private activeGenerationId: string | null = null;
  private terminalFailure: string | null = null;
  private disposed = false;
  private readonly unsubscribeEvents: () => void;
  private readonly unsubscribeFailures: () => void;

  constructor(transport: WorkerRpcTransport) {
    this.transport = transport;
    this.unsubscribeEvents = transport.subscribe((event) => this.onWorkerEvent(event));
    this.unsubscribeFailures = transport.subscribeFailure((reason) => {
      this.journal.interruptRunning();
      this.terminalFailure = reason;
      if (!this.openState) {
        this.listeners.clear();
        this.stateListeners.clear();
        return;
      }
      this.openState = false;
      this.activeGenerationId = null;
      this.emitState({ state: "closed", expected: false, reason });
      this.listeners.clear();
      this.stateListeners.clear();
    });
  }

  get isOpen(): boolean {
    return this.openState;
  }

  async open(): Promise<void> {
    if (this.disposed) throw new Error("Runtime event channel is disposed");
    if (this.terminalFailure) throw new Error(this.terminalFailure);
    this.transport.setSequenceBarrierActive(true);
    this.openState = true;
    this.emitState({ state: "open" });
  }

  subscribe(listener: (message: WSServerMessage) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  subscribeState(listener: (state: RuntimeEventChannelState) => void): () => void {
    this.stateListeners.add(listener);
    return () => this.stateListeners.delete(listener);
  }

  send(message: WSClientMessage): void {
    if (!this.openState) throw new Error("Runtime event channel is not open");
    if (message.type === "stop") {
      const stop = message.request_id === undefined
        ? this.stop()
        : this.transport.call("stop", { requestId: message.request_id });
      void stop.catch((error) => {
        this.emit({ type: "error", message: error.message, code: "WORKER_STOP_FAILED",
          ...(message.request_id === undefined ? {} : { request_id: message.request_id }) });
      });
      return;
    }
    const admission = this.journal.claim(message);
    if (admission !== "new") {
      if (admission === "conflict") this.emit({ type: "error", request_id: message.request_id,
        code: "REQUEST_ID_CONFLICT", message: "This request ID belongs to different inputs" });
      else replayRequest(this.journal.get(message.request_id!), event => this.emit(event));
      return;
    }
    const command = message.type;
    let completed = 0;
    let cancelled = false;
    const unsubscribe = message.request_id === undefined ? () => {} : this.subscribe((event) => {
      if (event.type === "done" && event.request_id === message.request_id) {
        completed += 1;
        cancelled ||= event.result.finish_reason === "cancelled";
      }
    });
    void this.transport.call(command, message).then(() => {
      if (message.request_id !== undefined) {
        const recorded = this.journal.get(message.request_id);
        completed = recorded.results.length;
        cancelled ||= recorded.results.some(row => row.result.finish_reason === "cancelled");
      }
      if (message.request_id !== undefined) this.emit({
        type: "request_complete", request_id: message.request_id,
        state: cancelled ? "cancelled" : "completed", completed_siblings: completed,
      });
    }).catch((error) => {
      this.emit({ ...wsErrorFrom(error, "WORKER_REQUEST_FAILED"),
        ...(message.request_id === undefined ? {} : { request_id: message.request_id }) });
      if (message.request_id !== undefined && this.journal.get(message.request_id).state !== "interrupted") this.emit({
        type: "request_complete", request_id: message.request_id,
        state: "failed", completed_siblings: completed,
      });
    }).finally(unsubscribe);
  }

  async stop(): Promise<void> {
    if (!this.openState) return;
    await this.transport.call("stop", undefined);
  }

  close(): void {
    if (
      !this.openState && this.listeners.size === 0 &&
      this.stateListeners.size === 0
    ) return;
    const wasOpen = this.openState;
    this.openState = false;
    this.transport.setSequenceBarrierActive(false);
    this.activeGenerationId = null;
    this.listeners.clear();
    if (wasOpen) this.emitState({ state: "closed", expected: true, reason: null });
    this.stateListeners.clear();
  }

  dispose(): void {
    if (this.disposed) return;
    this.journal.interruptRunning();
    this.close();
    this.disposed = true;
    this.unsubscribeEvents();
    this.unsubscribeFailures();
  }

  acknowledgeSnapshot(): void {
    this.activeGenerationId = null;
    this.transport.resumeEvents();
  }

  private onWorkerEvent(event: WorkerEventEnvelope): void {
    if (event.event === "done") this.journal.observe(event.payload as WSServerMessage);
    if (!this.openState) return;
    if (
      event.event === "started" ||
      event.event === "generation_progress" ||
      event.event === "token" ||
      event.event === "done" ||
      event.event === "tree_mutated"
    ) {
      if (event.event === "started") {
        this.activeGenerationId = event.generationId;
      } else if (
        (event.event === "token" || event.event === "done" || event.event === "generation_progress") &&
        event.generationId !== null &&
        event.generationId !== this.activeGenerationId
      ) {
        return;
      }
      this.emit(event.payload as WSServerMessage);
      if (event.event === "done") this.activeGenerationId = null;
    } else if (event.event === "error") {
      const scoped = event.payload as WorkerScopedFailure;
      if (scoped.scope !== "generation" && scoped.scope !== "transport") return;
      const failure = scoped.failure;
      this.emit({ type: "error", message: failure.message, code: failure.code });
      if (event.generationId === this.activeGenerationId) this.activeGenerationId = null;
    } else if (event.event === "device_lost") {
      const failure = event.payload as RuntimeFailure;
      this.emit({ type: "error", message: failure.message, code: failure.code });
      if (event.generationId === this.activeGenerationId) this.activeGenerationId = null;
    }
  }

  private emit(message: WSServerMessage): void {
    this.journal.observe(message);
    for (const listener of this.listeners) listener(message);
  }

  private emitState(state: RuntimeEventChannelState): void {
    for (const listener of this.stateListeners) listener(state);
  }
}

function serviceProxy<T extends object>(
  transport: WorkerRpcTransport,
  service: RuntimeServiceRequest["service"],
): T {
  return new Proxy({}, {
    get(_target, property) {
      if (typeof property !== "string") return undefined;
      if (!isRuntimeServiceMethod(service, property)) return undefined;
      return (...rawArgs: unknown[]) => {
        const callbacks = rawArgs.filter(
          (arg): arg is (event: RuntimeProgressEvent) => void => typeof arg === "function",
        );
        const args = rawArgs.filter((arg) => typeof arg !== "function");
        return transport.call(
          "request",
          { service, method: property, args } satisfies RuntimeServiceRequest,
          {
            ...(callbacks.length ? { onProgress: callbacks[0] } : {}),
          },
        );
      };
    },
  }) as T;
}

function workerFailure(failure: RuntimeFailure): Error {
  return new ApiError(
    Number.isInteger(failure.status) && failure.status >= 400 && failure.status <= 599
      ? failure.status
      : 500,
    `worker:${failure.code}`,
    failure.message,
    { detail: failure.message, error: failure },
  );
}

function commandTimeoutMs(
  command: WorkerCommandName,
  ordinaryTimeoutMs: number,
  longOperationIdleTimeoutMs: number,
): number {
  return NON_EXPIRING_COMMANDS.has(command)
    ? longOperationIdleTimeoutMs
    : ordinaryTimeoutMs;
}

const NON_EXPIRING_COMMANDS = new Set<WorkerCommandName>([
  "download",
  "download_pack",
  "cancel",
  "cancel_fitting",
  "delete",
  "clear",
  "persist",
  "load",
  "unload",
  "takeover",
  "request",
  "submit",
  "generate",
  "stop",
]);

function wsErrorFrom(error: unknown, fallbackCode: string): WSServerMessage {
  const body = error instanceof ApiError && error.body && typeof error.body === "object"
    ? error.body as { error?: { code?: unknown; message?: unknown } }
    : null;
  const nested = body?.error;
  return {
    type: "error",
    message: typeof nested?.message === "string"
      ? nested.message
      : error instanceof Error ? error.message : String(error),
    code: typeof nested?.code === "string" ? nested.code : fallbackCode,
  };
}
