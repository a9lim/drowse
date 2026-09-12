import type {
  CatalogWorkerRequest,
  DeviceLoadRecord,
  DownloadProgress,
  DownloadWorkerResult,
  PackDownloadWorkerResult,
  HostedController,
  ModelPreference,
  ModelRecommendation,
  RuntimeCapabilities,
  RuntimeClient,
  RuntimeSnapshot,
  VerifiedCatalog,
  WorkerEventEnvelope,
  WorkerScopedFailure,
} from "../../lib/runtime/contracts";
import { rankCatalogModels } from "../../lib/runtime/recommendation";
import { ApiError } from "../../lib/runtime/errors";
import { cacheOfflineRuntimeAssets } from "./offlineRuntimeAssets";
import { clearDrowseLocalData } from "../../lib/runtime/localData";
import { initialRuntimeSnapshot, runtimeFailure } from "../../lib/runtime/state";
import {
  BrowserRuntimeClient,
  WorkerRpcTransport,
  type WorkerLike,
  type WorkerRpcOptions,
} from "./browserRuntimeClient";
import {
  GpuRuntimeOwnership,
  withHostedDestructiveLock,
  type RuntimeOwnership,
  type RuntimeOwnershipOptions,
} from "./ownership";
import {
  browserRuntimeClass,
  requestPersistentStorage,
  storageEstimate,
} from "./capabilities";

const CONSERVATIVE_RUNTIME_IDLE_WATCHDOG_MS = 180_000;
const CONSERVATIVE_MODEL_LOAD_TIMEOUT_MS = 600_000;

export interface HostedControllerOptions extends WorkerRpcOptions {
  modelLoadTimeoutMs?: number;
  loadRecords?: DeviceLoadRecord[];
  installedModelVariantIds?: string[];
  installedPackIds?: string[];
  confirmBusyTakeover?: (snapshot: RuntimeSnapshot) => boolean | Promise<boolean>;
  ownershipFactory?: (options: RuntimeOwnershipOptions) => RuntimeOwnership;
  runDestructiveExclusive?: <T>(operation: () => Promise<T>) => Promise<T>;
  clearLocalData?: () => void;
}

export class HostedControllerImpl implements HostedController {
  private readonly transport: WorkerRpcTransport;
  private readonly modelLoadTimeoutMs: number;
  private modelLoadTimedOut = false;
  private readonly listeners = new Set<(snapshot: RuntimeSnapshot) => void>();
  private readonly loadRecords: DeviceLoadRecord[];
  private readonly installedModelVariantIds: Set<string>;
  private readonly installedPackIds: Set<string>;
  private readonly confirmBusyTakeover:
    | ((snapshot: RuntimeSnapshot) => boolean | Promise<boolean>)
    | null;
  private readonly ownershipFactory:
    (options: RuntimeOwnershipOptions) => RuntimeOwnership;
  private readonly runDestructiveExclusive:
    <T>(operation: () => Promise<T>) => Promise<T>;
  private readonly clearLocalData: () => void;
  private readonly unsubscribeEvents: () => void;
  private readonly unsubscribeFailures: () => void;
  private currentSnapshot = initialRuntimeSnapshot();
  private publishedSnapshot = this.currentSnapshot;
  private publicationHoldDepth = 0;
  private publicationPending = false;
  private capabilities: RuntimeCapabilities | null = null;
  private ownership: RuntimeOwnership | null = null;
  private pendingDownloadRequests = 0;
  private readonly ownsTransport: boolean;

  constructor(
    workerOrTransport: WorkerLike | WorkerRpcTransport,
    options: HostedControllerOptions & { ownsTransport?: boolean } = {},
  ) {
    this.transport = workerOrTransport instanceof WorkerRpcTransport
      ? workerOrTransport
      : new WorkerRpcTransport(workerOrTransport, options);
    this.ownsTransport = options.ownsTransport ?? !(workerOrTransport instanceof WorkerRpcTransport);
    this.modelLoadTimeoutMs = options.modelLoadTimeoutMs ?? 0;
    this.loadRecords = options.loadRecords ?? [];
    this.installedModelVariantIds = new Set(options.installedModelVariantIds ?? []);
    this.installedPackIds = new Set(options.installedPackIds ?? []);
    this.confirmBusyTakeover = options.confirmBusyTakeover ?? null;
    this.ownershipFactory = options.ownershipFactory ??
      ((ownershipOptions) => new GpuRuntimeOwnership(ownershipOptions));
    this.runDestructiveExclusive = options.runDestructiveExclusive ??
      withHostedDestructiveLock;
    this.clearLocalData = options.clearLocalData ?? clearDrowseLocalData;
    this.unsubscribeEvents = this.transport.subscribe((event) => this.onWorkerEvent(event));
    this.unsubscribeFailures = this.transport.subscribeFailure((reason) => {
      this.ownership?.release();
      this.fail(this.modelLoadTimedOut ? "MODEL_LOAD_TIMEOUT" : "RUNTIME_WORKER_FAILED", new Error(reason), false);
    });
  }

  get snapshot(): RuntimeSnapshot {
    return this.currentSnapshot;
  }

  subscribe(listener: (snapshot: RuntimeSnapshot) => void): () => void {
    this.listeners.add(listener);
    listener(this.publishedSnapshot);
    return () => this.listeners.delete(listener);
  }

  async check(): Promise<RuntimeCapabilities> {
    try {
      const capabilities = await this.transport.call<RuntimeCapabilities>("check", undefined);
      this.capabilities = capabilities;
      if (capabilities.supported) {
        await this.transport.call("storage", undefined);
      }
      return capabilities;
    } catch (error) {
      this.fail("COMPATIBILITY_CHECK_FAILED", error, true);
      throw error;
    }
  }

  catalog(request: CatalogWorkerRequest = {}): Promise<VerifiedCatalog> {
    return this.transport.call("catalog", request);
  }

  recommend(
    catalog: VerifiedCatalog,
    preference: ModelPreference,
    contextTokens = 2048,
    language?: string,
    options: { explicitOomRetry?: boolean } = {},
  ): ModelRecommendation[] {
    if (!this.capabilities) throw new Error("Run the compatibility check first");
    assertAppleMobileContext(this.capabilities, contextTokens);
    return rankCatalogModels(catalog.document.models, {
      capabilities: this.capabilities,
      contextTokens,
      loadRecords: this.loadRecords,
      preference,
      language: language ?? this.capabilities.signals.language,
      explicitOomRetry: options.explicitOomRetry,
      installedModelVariantIds: [...this.installedModelVariantIds],
    });
  }

  async download(
    modelVariantId: string,
    onProgress?: (progress: DownloadProgress) => void,
    options: { contextTokens?: number; explicitUnsafeOverride?: boolean } = {},
  ): Promise<DownloadWorkerResult> {
    const contextTokens = options.contextTokens ?? 2048;
    if (this.capabilities) assertAppleMobileContext(this.capabilities, contextTokens);
    this.pendingDownloadRequests += 1;
    try {
      const result = await this.transport.call<DownloadWorkerResult>("download", {
        modelVariantId,
        contextTokens,
        ...(options.explicitUnsafeOverride
          ? { explicitUnsafeOverride: true }
          : {}),
      }, {
        timeoutMs: 0,
        onProgress: onProgress
          ? (event) => onProgress(event.data as DownloadProgress)
          : undefined,
      });
      if (result.installed) this.installedModelVariantIds.add(modelVariantId);
      return result;
    } finally {
      this.pendingDownloadRequests -= 1;
    }
  }

  async downloadPack(
    modelVariantId: string,
    packId: string,
    onProgress?: (progress: DownloadProgress) => void,
  ): Promise<PackDownloadWorkerResult> {
    this.pendingDownloadRequests += 1;
    try {
      const result = await this.transport.call<PackDownloadWorkerResult>(
        "download_pack",
        { modelVariantId, packId },
        {
          timeoutMs: 0,
          onProgress: onProgress
            ? (event) => onProgress(event.data as DownloadProgress)
            : undefined,
        },
      );
      if (result.installed) this.installedPackIds.add(packId);
      return result;
    } finally {
      this.pendingDownloadRequests -= 1;
    }
  }

  async cancelDownload(): Promise<void> {
    await this.transport.call("cancel", undefined, { timeoutMs: 0 });
  }

  async cancelFitting(): Promise<void> {
    await this.transport.call("cancel_fitting", undefined, { timeoutMs: 0 });
  }

  async deleteModel(modelVariantId: string): Promise<void> {
    await this.runDestructive(async () => {
      await this.transport.call("delete", { kind: "model", id: modelVariantId });
      this.installedModelVariantIds.delete(modelVariantId);
    });
  }

  async deletePack(packId: string): Promise<void> {
    await this.runDestructive(async () => {
      await this.transport.call("delete", { kind: "pack", id: packId });
      this.installedPackIds.delete(packId);
    });
  }

  async clearAll(): Promise<void> {
    await this.runDestructive(async () => {
      await this.transport.call("clear", undefined);
      this.clearLocalData();
      this.installedModelVariantIds.clear();
      this.installedPackIds.clear();
    });
  }

  requestPersistence(onLateGranted?: () => void): Promise<boolean> {
    return requestPersistentStorage(undefined, undefined, onLateGranted);
  }

  refreshStorage(): Promise<RuntimeCapabilities["storage"]> {
    return storageEstimate();
  }

  async load(
    modelVariantId: string,
    contextTokens: number,
    options: { explicitUnsafeOverride?: boolean; resetSession?: boolean } = {},
  ): Promise<void> {
    if (this.modelLoadTimeoutMs <= 0) return this.loadWithOwnership(modelVariantId, contextTokens, options);
    let timeout: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([
        this.loadWithOwnership(modelVariantId, contextTokens, options),
        new Promise<never>((_, reject) => {
          timeout = setTimeout(() => {
            const error = Object.assign(new Error(
              "The model took too long to load. Reload this page and try a smaller model. Your saved chats and downloads have not been removed.",
            ), { code: "MODEL_LOAD_TIMEOUT" });
            this.modelLoadTimedOut = true;
            this.transport.terminateWithFailure(error.message);
            this.ownership?.close();
            this.ownership = null;
            reject(error);
          }, this.modelLoadTimeoutMs);
        }),
      ]);
    } finally {
      if (timeout !== undefined) clearTimeout(timeout);
    }
  }

  private async loadWithOwnership(
    modelVariantId: string,
    contextTokens: number,
    options: { explicitUnsafeOverride?: boolean; resetSession?: boolean },
  ): Promise<void> {
    if (this.capabilities) assertAppleMobileContext(this.capabilities, contextTokens);
    const ownership = this.runtimeOwnership();
    if (!(await ownership.acquire())) {
      throw new Error("Another tab owns the Drowse GPU runtime. Close it or request takeover there.");
    }
    try {
      if (import.meta.env.PROD) await cacheOfflineRuntimeAssets();
      await this.transport.call("load", {
        modelVariantId,
        contextTokens,
        ...(options.explicitUnsafeOverride
          ? { explicitUnsafeOverride: true }
          : {}),
        ...(options.resetSession ? { resetSession: true } : {}),
      });
    } catch (error) {
      if (
        this.currentSnapshot.lifecycle !== "ready" &&
        this.currentSnapshot.lifecycle !== "loading" &&
        this.currentSnapshot.lifecycle !== "failed"
      ) {
        ownership.release();
      }
      if (
        this.currentSnapshot.lifecycle !== "failed" &&
        this.currentSnapshot.lifecycle !== "ready" &&
        this.currentSnapshot.lifecycle !== "loading"
      ) {
        this.fail("MODEL_LOAD_FAILED", error, true);
      }
      throw error;
    }
  }

  async unload(): Promise<void> {
    await this.unloadWorker();
    this.ownership?.release();
  }

  async dispose(): Promise<void> {
    try {
      await this.unload();
    } catch {
      // A terminal worker failure must not prevent local lock cleanup.
    } finally {
      this.unsubscribeEvents();
      this.unsubscribeFailures();
      this.ownership?.close();
      if (this.ownsTransport) this.transport.dispose();
      this.listeners.clear();
    }
  }

  private onWorkerEvent(event: WorkerEventEnvelope): void {
    if (event.event === "status") {
      this.currentSnapshot = event.payload as RuntimeSnapshot;
      this.installedModelVariantIds.clear();
      for (const modelVariantId of this.currentSnapshot.installedModelVariantIds) {
        this.installedModelVariantIds.add(modelVariantId);
      }
      this.installedPackIds.clear();
      for (const packId of this.currentSnapshot.installedPackIds) {
        this.installedPackIds.add(packId);
      }
      this.loadRecords.splice(
        0,
        this.loadRecords.length,
        ...this.currentSnapshot.loadRecords,
      );
      if (this.currentSnapshot.lifecycle === "failed") this.ownership?.release();
      this.emit();
    } else if (event.event === "device_lost") {
      const failure = event.payload as import("../../lib/runtime/contracts").RuntimeFailure;
      this.ownership?.release();
      this.update({ lifecycle: "failed", error: failure });
    } else if (event.event === "error") {
      const { scope, failure } = event.payload as WorkerScopedFailure;
      if (scope === "lifecycle") {
        if (!failure.recoverable) this.ownership?.release();
        this.update({ lifecycle: "failed", error: failure });
      } else if (scope === "download" || scope === "generation" || scope === "fitting") {
        this.setOperation(scope, "failed", failure);
      }
    }
  }

  private async cancelDownloadIfActive(): Promise<void> {
    const phase = this.currentSnapshot.download.phase;
    if (
      this.pendingDownloadRequests > 0 || phase === "running" || phase === "cancelling"
    ) {
      await this.transport.call("cancel", undefined, { timeoutMs: 0 });
    }
  }

  private runtimeOwnership(): RuntimeOwnership {
    if (!this.ownership) {
      this.ownership = this.ownershipFactory({
        onTakeoverRequest: async () => {
          const busy =
            this.currentSnapshot.lifecycle === "checking" ||
            this.currentSnapshot.lifecycle === "loading" ||
            ["download", "generation", "fitting"].some((operation) => {
              const phase = this.currentSnapshot[
                operation as "download" | "generation" | "fitting"
              ].phase;
              return phase === "running" || phase === "cancelling";
            });
          if (
            busy &&
            (!this.confirmBusyTakeover ||
              !(await this.confirmBusyTakeover(this.currentSnapshot)))
          ) return false;
          await this.unloadWorker();
          return true;
        },
      });
    }
    return this.ownership;
  }

  private async runDestructive<T>(operation: () => Promise<T>): Promise<T> {
    this.publicationHoldDepth += 1;
    try {
      await this.cancelDownloadIfActive();
      const ownership = this.runtimeOwnership();
      if (!(await ownership.acquire())) {
        throw new Error(
          "Another tab owns the Drowse GPU runtime. Close it before changing local runtime data.",
        );
      }
      try {
        await this.unloadWorker();
        return await this.runDestructiveExclusive(operation);
      } finally {
        ownership.release();
      }
    } finally {
      this.publicationPending = true;
      this.publicationHoldDepth -= 1;
      if (this.publicationHoldDepth === 0 && this.publicationPending) {
        this.publicationPending = false;
        this.publishCurrentSnapshot();
      }
    }
  }

  private async unloadWorker(): Promise<void> {
    try {
      await this.cancelDownloadIfActive();
      await this.cancelFittingIfActive();
      await this.stopGenerationIfActive();
      await this.transport.call("unload", undefined);
    } catch (error) {
      this.transport.terminateWithFailure(
        `Model unload failed: ${error instanceof Error ? error.message : String(error)}`,
      );
      throw error;
    }
  }

  private async stopGenerationIfActive(): Promise<void> {
    const phase = this.currentSnapshot.generation.phase;
    if (phase === "running" || phase === "cancelling") {
      await this.transport.call("stop", undefined, { timeoutMs: 0 });
    }
  }

  private async cancelFittingIfActive(): Promise<void> {
    const phase = this.currentSnapshot.fitting.phase;
    if (phase === "running" || phase === "cancelling") {
      await this.cancelFitting();
    }
  }

  private setOperation(
    operation: "download" | "generation" | "fitting",
    phase: RuntimeSnapshot[typeof operation]["phase"],
    error: RuntimeSnapshot[typeof operation]["error"],
  ): void {
    const now = Date.now();
    const previous = this.currentSnapshot[operation];
    this.update({
      [operation]: {
        phase,
        startedAt: phase === "running" ? previous.startedAt ?? now : previous.startedAt,
        finishedAt: phase === "complete" || phase === "failed" ? now : null,
        error,
      },
    });
  }

  private fail(code: string, error: unknown, recoverable: boolean): void {
    this.update({
      lifecycle: "failed",
      error: toFailure(code, error, recoverable),
    });
  }

  private update(patch: Partial<RuntimeSnapshot>): void {
    this.currentSnapshot = { ...this.currentSnapshot, ...patch };
    this.emit();
  }

  private emit(): void {
    if (this.publicationHoldDepth > 0) {
      this.publicationPending = true;
      return;
    }
    this.publishCurrentSnapshot();
  }

  private publishCurrentSnapshot(): void {
    this.publishedSnapshot = this.currentSnapshot;
    for (const listener of this.listeners) listener(this.publishedSnapshot);
  }
}

function assertAppleMobileContext(
  capabilities: RuntimeCapabilities,
  contextTokens: number,
): void {
  if (capabilities.signals.appleMobile === true && contextTokens > 2048) {
    throw new RangeError(
      "iPhone and iPad use the 2,048-token profile to stay within Safari's browser memory limits",
    );
  }
}

function toFailure(
  code: string,
  error: unknown,
  recoverable: boolean,
): import("../../lib/runtime/contracts").RuntimeFailure {
  const nested = error instanceof ApiError && error.body && typeof error.body === "object"
    ? (error.body as { error?: { code?: unknown } }).error
    : null;
  return runtimeFailure(
    typeof nested?.code === "string"
      ? nested.code
      : error instanceof Error && error.name !== "Error"
        ? error.name
        : code,
    error instanceof Error ? error.message : String(error),
    recoverable,
    error instanceof ApiError ? error.status : 500,
  );
}

export interface HostedRuntimeBundle {
  controller: HostedController;
  runtime: RuntimeClient;
  dispose(): Promise<void>;
}

export function createHostedRuntime(
  options: HostedControllerOptions = {},
): HostedRuntimeBundle {
  const worker = new Worker(new URL("./browser.worker.ts", import.meta.url), {
    type: "module",
    name: "drowse-runtime",
  });
  const runtimeOptions = platformTransportOptions(options);
  const transport = new WorkerRpcTransport(worker, runtimeOptions);
  const controller = new HostedControllerImpl(transport, {
    ...runtimeOptions,
    ownsTransport: false,
  });
  const runtime = new BrowserRuntimeClient(transport, { ownsTransport: false });
  let disposed = false;
  return {
    controller,
    runtime,
    async dispose() {
      if (disposed) return;
      disposed = true;
      try {
        await controller.dispose();
        await runtime.dispose();
      } finally {
        transport.dispose();
      }
    },
  };
}

function platformTransportOptions(
  options: HostedControllerOptions,
): HostedControllerOptions {
  const runtimeClass = typeof navigator === "undefined"
    ? null
    : browserRuntimeClass();
  if (
    runtimeClass === null ||
    ![
      "apple-mobile-webkit",
      "desktop-webkit",
      "desktop-gecko",
    ].includes(runtimeClass)
  ) return options;
  return {
    ...options,
    longOperationIdleTimeoutMs: options.longOperationIdleTimeoutMs ?? CONSERVATIVE_RUNTIME_IDLE_WATCHDOG_MS,
    modelLoadTimeoutMs: options.modelLoadTimeoutMs ?? CONSERVATIVE_MODEL_LOAD_TIMEOUT_MS,
  };
}
