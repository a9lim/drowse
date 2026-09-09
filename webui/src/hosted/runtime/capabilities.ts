import type {
  BrowserRuntimeClass,
  CapabilityIssue,
  RuntimeCapabilities,
  RuntimeCapabilityOperation,
} from "../../lib/runtime/contracts";
import {
  BROWSER_RUNTIME_LOCK,
  BROWSER_RUNTIME_RELEASE_VERIFIED,
} from "./runtimeLock";
import { resolveStructuredHookProfile } from "./structuredHookProfile";
import { HOSTED_DISTRIBUTION_CONFIG } from "./distributionConfig";
import { BROWSER_MODEL_BACKEND_INTEGRATED } from "./browserModelBackend";
import {
  requestPersistentStorage as requestBrowserPersistentStorage,
  type PersistentStorageManager,
} from "../../lib/runtime/storagePersistence";
import {
  BROWSER_ARTIFACT_AUTHORING_INTEGRATED,
  BROWSER_AUTHORING_RUNTIME_INTEGRATED,
} from "./browserAuthoring";
import { CATALOG_WEBGPU_LIMIT_NAMES } from "../../lib/runtime/catalog";
import { windowsIntelGpuHint, windowsIntelGen9Blocked, WINDOWS_INTEL_GEN9_BLOCK } from "../../lib/runtime/gpuRecovery";

interface GpuAdapterInfoLike {
  vendor?: string;
  architecture?: string;
  device?: string;
  description?: string;
  isFallbackAdapter?: boolean;
}

interface GpuBufferLike {
  mapAsync(mode: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

interface GpuComputePassLike {
  setPipeline(pipeline: unknown): void;
  setBindGroup(index: number, bindGroup: unknown): void;
  dispatchWorkgroups(count: number): void;
  end(): void;
}

interface GpuCommandEncoderLike {
  beginComputePass(): GpuComputePassLike;
  copyBufferToBuffer(
    source: GpuBufferLike,
    sourceOffset: number,
    destination: GpuBufferLike,
    destinationOffset: number,
    size: number,
  ): void;
  finish(): unknown;
}

interface GpuDeviceLike {
  queue: {
    submit(commands: unknown[]): void;
    onSubmittedWorkDone(): Promise<void>;
  };
  createBuffer(descriptor: Record<string, unknown>): GpuBufferLike;
  createShaderModule(descriptor: Record<string, unknown>): unknown;
  createComputePipeline(descriptor: Record<string, unknown>): {
    getBindGroupLayout(index: number): unknown;
  };
  createBindGroup(descriptor: Record<string, unknown>): unknown;
  createCommandEncoder(): GpuCommandEncoderLike;
  destroy(): void;
}

export interface GpuAdapterLike {
  features: Iterable<string>;
  limits: object;
  info?: GpuAdapterInfoLike;
  isFallbackAdapter?: boolean;
  requestAdapterInfo?(): Promise<GpuAdapterInfoLike>;
  requestDevice(): Promise<GpuDeviceLike>;
}

interface GpuLike {
  requestAdapter(options?: { powerPreference?: "low-power" | "high-performance" }): Promise<GpuAdapterLike | null>;
}

interface NavigatorWithRuntimeSignals {
  gpu?: GpuLike;
  deviceMemory?: number;
  maxTouchPoints?: number;
  platform?: string;
  storage?: StorageManager;
}

interface OpfsWritableLike {
  write(data: BufferSource | Blob | string): Promise<void>;
  seek(position: number): Promise<void>;
  truncate(size: number): Promise<void>;
  close(): Promise<void>;
  abort?(): Promise<void>;
}

interface OpfsFileHandleLike {
  createWritable(options?: { keepExistingData?: boolean }): Promise<OpfsWritableLike>;
  getFile(): Promise<Blob>;
}

interface OpfsDirectoryLike {
  getFileHandle(name: string, options: { create: true }): Promise<OpfsFileHandleLike>;
  getDirectoryHandle(name: string, options: { create: true }): Promise<OpfsDirectoryLike>;
  entries(): AsyncIterableIterator<[string, unknown]>;
  removeEntry(name: string, options?: { recursive?: boolean }): Promise<void>;
}

export interface BrowserCapabilityCheckerOptions {
  adapterTimeoutMs?: number;
  adapterInfoTimeoutMs?: number;
  deviceTimeoutMs?: number;
  calibrationTimeoutMs?: number;
  opfsTimeoutMs?: number;
}

export interface BrowserAuthoringAvailabilityInput {
  runtimeIntegrated: boolean;
  modelBackendIntegrated: boolean;
}

export function browserAuthoringAvailableForChannel({
  runtimeIntegrated,
  modelBackendIntegrated,
}: BrowserAuthoringAvailabilityInput): boolean {
  return runtimeIntegrated && modelBackendIntegrated;
}

export class BrowserCapabilityChecker {
  private adapter: GpuAdapterLike | null = null;
  private checkedAdapterFingerprint: string | null = null;
  private lastCheckSupported = false;
  private readonly adapterTimeoutMs: number;
  private readonly adapterInfoTimeoutMs: number;
  private readonly deviceTimeoutMs: number;
  private readonly calibrationTimeoutMs: number;
  private readonly opfsTimeoutMs: number;

  constructor(options: BrowserCapabilityCheckerOptions = {}) {
    this.adapterTimeoutMs = positiveTimeout(options.adapterTimeoutMs, 10_000);
    this.adapterInfoTimeoutMs = positiveTimeout(options.adapterInfoTimeoutMs, 2_000);
    this.deviceTimeoutMs = positiveTimeout(options.deviceTimeoutMs, 10_000);
    this.calibrationTimeoutMs = positiveTimeout(options.calibrationTimeoutMs, 5_000);
    this.opfsTimeoutMs = positiveTimeout(options.opfsTimeoutMs, 5_000);
  }

  async check(): Promise<RuntimeCapabilities> {
    this.adapter = null;
    this.checkedAdapterFingerprint = null;
    this.lastCheckSupported = false;
    const issues: CapabilityIssue[] = [];
    const secureContext = globalThis.isSecureContext === true;
    const crossOriginIsolated = globalThis.crossOriginIsolated === true;
    const dedicatedWorker = typeof globalThis.Worker === "function";
    const indexedDb = typeof globalThis.indexedDB !== "undefined";
    const opfsApi = typeof navigator.storage?.getDirectory === "function";
    const opfs = opfsApi
      ? await verifyOpfsLifecycle(navigator.storage, this.opfsTimeoutMs)
      : false;
    const webLocks = typeof navigator.locks?.request === "function";
    const broadcastChannel = typeof globalThis.BroadcastChannel === "function";
    const webAssembly = typeof globalThis.WebAssembly === "object";
    const navigatorSignals = navigator as unknown as NavigatorWithRuntimeSignals;
    const browser = browserIdentity(navigator.userAgent);
    const platform = browserPlatform(navigator.userAgent, navigatorSignals);

    hardUnless(secureContext, "INSECURE_CONTEXT", "Drowse requires a secure HTTPS context", issues);
    hardUnless(
      crossOriginIsolated,
      "CROSS_ORIGIN_ISOLATION_REQUIRED",
      "Drowse requires working COOP and COEP cross-origin isolation headers",
      issues,
    );
    hardUnless(dedicatedWorker, "WORKER_UNAVAILABLE", "Dedicated Web Workers are unavailable", issues);
    hardUnless(indexedDb, "INDEXEDDB_UNAVAILABLE", "IndexedDB is unavailable", issues);
    hardUnless(
      opfs,
      opfsApi ? "OPFS_READ_WRITE_FAILED" : "OPFS_UNAVAILABLE",
      opfsApi
        ? "Private local storage could not complete the file create, resume, read, list, and delete check"
        : "Origin-private file storage is unavailable",
      issues,
    );
    hardUnless(webLocks, "WEB_LOCKS_UNAVAILABLE", "The Web Locks API is unavailable", issues);
    hardUnless(
      broadcastChannel,
      "BROADCAST_CHANNEL_UNAVAILABLE",
      "BroadcastChannel is unavailable",
      issues,
    );
    hardUnless(webAssembly, "WASM_UNAVAILABLE", "WebAssembly is unavailable", issues);
    const gpu = navigatorSignals.gpu;
    if (!gpu) {
      issues.push(hard("WEBGPU_UNAVAILABLE", "WebGPU is unavailable"));
    }

    let features: string[] = [];
    let limits: Record<string, number> = {};
    let adapterInfo: RuntimeCapabilities["webGpu"]["adapterInfo"] = null;
    let fallback: RuntimeCapabilities["webGpu"]["fallback"] = "unknown";
    let calibrationScore: number | null = null;
    let calibrationFailure: CapabilityIssue | null = null;
    let deviceCreationFailed = false;

    if (gpu) {
      try {
        this.adapter = await withTimeout(
          gpu.requestAdapter({ powerPreference: "high-performance" }),
          this.adapterTimeoutMs,
          "WEBGPU_ADAPTER_TIMEOUT",
          "Timed out while requesting a WebGPU adapter",
        );
      } catch (error) {
        issues.push(hard(
          error instanceof CapabilityTimeoutError
            ? error.code
            : "WEBGPU_ADAPTER_FAILED",
          message(error),
        ));
      }
      if (!this.adapter) {
        issues.push(hard("WEBGPU_ADAPTER_MISSING", "No WebGPU adapter was available"));
      } else {
        features = [...this.adapter.features].map(String).sort();
        limits = snapshotLimits(this.adapter.limits);
        const info = await adapterInformation(this.adapter, this.adapterInfoTimeoutMs);
        if (info) {
          adapterInfo = adapterInfoSnapshot(info);
          const gpuHint = windowsIntelGpuHint(adapterInfo.vendor, navigator.userAgent);
          if (windowsIntelGen9Blocked(adapterInfo.vendor, adapterInfo.architecture, navigator.userAgent)) {
            issues.push(hard("WEBGPU_WINDOWS_INTEL_GEN9_BLOCKED", WINDOWS_INTEL_GEN9_BLOCK));
          } else if (gpuHint) {
            issues.push(advisory("WINDOWS_INTEL_GPU", gpuHint));
          }
          if (typeof info.isFallbackAdapter === "boolean") {
            fallback = info.isFallbackAdapter ? "fallback" : "hardware";
          }
        }
        if (fallback === "unknown" && typeof this.adapter.isFallbackAdapter === "boolean") {
          fallback = this.adapter.isFallbackAdapter ? "fallback" : "hardware";
        }
        if (fallback === "fallback") {
          issues.push(hard("SOFTWARE_ADAPTER", "The selected WebGPU adapter is a software fallback"));
        }

        let device: GpuDeviceLike | null = null;
        let deviceRequest: Promise<GpuDeviceLike> | null = null;
        try {
          deviceRequest = this.adapter.requestDevice();
          device = await withTimeout(
            deviceRequest,
            this.deviceTimeoutMs,
            "WEBGPU_DEVICE_TIMEOUT",
            "Timed out while creating a WebGPU device",
          );
        } catch (error) {
          deviceCreationFailed = true;
          issues.push(hard(
            error instanceof CapabilityTimeoutError
              ? error.code
              : "WEBGPU_DEVICE_FAILED",
            message(error),
          ));
          if (deviceRequest && error instanceof CapabilityTimeoutError) {
            void deviceRequest.then((lateDevice) => lateDevice.destroy(), () => {});
          }
        }
        if (device) {
          try {
            calibrationScore = await withTimeout(
              calibrateCompute(device, platform.appleMobile),
              this.calibrationTimeoutMs,
              "WEBGPU_CALIBRATION_TIMEOUT",
              "The WebGPU compute calibration timed out",
            );
            if (calibrationScore === null) {
              calibrationFailure = hard(
                "WEBGPU_CALIBRATION_UNAVAILABLE",
                "The browser could not run the WebGPU compute calibration",
              );
            }
          } catch (error) {
            calibrationScore = null;
            calibrationFailure = hard(
              error instanceof CapabilityTimeoutError
                ? error.code
                : "WEBGPU_CALIBRATION_FAILED",
              error instanceof CapabilityTimeoutError
                ? error.message
                : `The WebGPU compute calibration failed: ${message(error)}`,
            );
          } finally {
            device.destroy();
          }
        }
        if (fallback === "unknown") {
          issues.push(calibrationScore === null
            ? hard(
                "ADAPTER_KIND_UNKNOWN",
                "The browser did not identify the WebGPU adapter and its compute check did not complete",
              )
            : advisory(
                "ADAPTER_KIND_UNKNOWN",
                "The browser did not identify the WebGPU adapter, but its WebGPU compute check passed",
              ));
        }
      }
    }

    const storage = await storageEstimate();
    const deviceMemoryGiB = finiteOrNull(navigatorSignals.deviceMemory);
    const logicalCpuCount = finiteOrNull(navigator.hardwareConcurrency);
    const mobile = platform.appleMobile || /Android|Mobile/i.test(navigator.userAgent);
    const language = navigator.language || "en";
    if (deviceMemoryGiB === null) {
      issues.push(advisory("DEVICE_MEMORY_UNKNOWN", "The browser did not expose its coarse device-memory signal"));
    }
    if (calibrationScore === null && !deviceCreationFailed && this.adapter) {
      issues.push(requiresVerifiedCompute(platform.runtimeClass)
        ? calibrationFailure ?? hard(
            "WEBGPU_CALIBRATION_UNAVAILABLE",
            "The browser could not run the WebGPU compute calibration",
          )
        : advisory("CALIBRATION_UNKNOWN", "The WebGPU compute calibration did not complete"));
    }

    const hardFailures = issues.filter((item) => item.severity === "hard");
    const operations = operationAvailability(hardFailures);
    const deviceSignature = coarseSignature({
      fallback,
      adapterInfo,
      features,
      limits,
      deviceMemoryGiB,
      logicalCpuCount,
      mobile,
      runtimeClass: platform.runtimeClass,
      appleMobile: platform.appleMobile,
      browser,
    });
    const result: RuntimeCapabilities = {
      checkedAt: Date.now(),
      supported: hardFailures.length === 0,
      secureContext,
      crossOriginIsolated,
      dedicatedWorker,
      indexedDb,
      opfs,
      webLocks,
      broadcastChannel,
      webAssembly,
      webGpu: {
        available: Boolean(gpu && this.adapter),
        fallback,
        features,
        limits,
        adapterInfo,
      },
      storage,
      signals: {
        deviceMemoryGiB,
        logicalCpuCount,
        mobile,
        runtimeClass: platform.runtimeClass,
        appleMobile: platform.appleMobile,
        language,
        calibrationScore,
      },
      deviceSignature,
      runtimeVersion: BROWSER_RUNTIME_LOCK.runtimeAbi,
      issues,
      operations,
      limits: {
        manifoldFitMaxIntrinsicDim: browserManifoldFitMaxIntrinsicDim(),
      },
    };
    this.lastCheckSupported = result.supported;
    this.checkedAdapterFingerprint = result.supported
      ? adapterFingerprint(fallback, adapterInfo, features, limits)
      : null;
    this.adapter = null;
    return result;
  }

  async adapterForLoad(): Promise<GpuAdapterLike> {
    if (!this.lastCheckSupported || this.checkedAdapterFingerprint === null) {
      throw loadError(
        "COMPATIBILITY_CHECK_REQUIRED",
        "Compatibility must pass before loading a model",
      );
    }
    const gpu = (navigator as unknown as NavigatorWithRuntimeSignals).gpu;
    if (!gpu) {
      throw loadError("WEBGPU_UNAVAILABLE", "WebGPU became unavailable before model loading");
    }
    let adapter: GpuAdapterLike | null;
    try {
      adapter = await withTimeout(
        gpu.requestAdapter({ powerPreference: "high-performance" }),
        this.adapterTimeoutMs,
        "WEBGPU_ADAPTER_TIMEOUT",
        "Timed out while requesting a fresh WebGPU adapter for model loading",
      );
    } catch (error) {
      if (error instanceof CapabilityTimeoutError) throw loadError(error.code, error.message);
      throw loadError("WEBGPU_ADAPTER_FAILED", message(error));
    }
    if (!adapter) {
      throw loadError("WEBGPU_ADAPTER_MISSING", "No WebGPU adapter was available for model loading");
    }
    const features = [...adapter.features].map(String).sort();
    const limits = snapshotLimits(adapter.limits);
    const info = await adapterInformation(adapter, this.adapterInfoTimeoutMs);
    const adapterInfo = info ? adapterInfoSnapshot(info) : null;
    const fallback = fallbackStatus(adapter, info);
    if (fallback === "fallback") {
      throw loadError(
        "SOFTWARE_ADAPTER",
        "The load-time WebGPU adapter is a software fallback",
      );
    }
    if (
      adapterFingerprint(fallback, adapterInfo, features, limits) !==
        this.checkedAdapterFingerprint
    ) {
      throw loadError(
        "WEBGPU_ADAPTER_CHANGED",
        "The WebGPU adapter changed after compatibility testing; run the check again",
      );
    }
    return adapter;
  }

  clearAdapter(): void {
    this.adapter = null;
    this.checkedAdapterFingerprint = null;
    this.lastCheckSupported = false;
  }
}

function browserManifoldFitMaxIntrinsicDim(): number | null {
  const limits = BROWSER_RUNTIME_LOCK.models.map((model) =>
    resolveStructuredHookProfile(model.structuredHookProfile).maxIntrinsicDim
  );
  return limits.length === 0 ? null : Math.min(...limits);
}

export async function storageEstimate(
  timeoutMs = 5_000,
): Promise<RuntimeCapabilities["storage"]> {
  const unknown = {
    quotaBytes: null,
    usageBytes: null,
    availableBytes: null,
    persisted: null,
  };
  if (!navigator.storage?.estimate) {
    return unknown;
  }
  const timeout = positiveTimeout(timeoutMs, 5_000);
  const [estimate, persisted] = await Promise.all([
    withTimeout(
      Promise.resolve().then(() => navigator.storage.estimate()),
      timeout,
      "STORAGE_ESTIMATE_TIMEOUT",
      "The browser storage estimate timed out",
    ).catch(() => null),
    withTimeout(
      Promise.resolve().then(() => navigator.storage.persisted?.() ?? null),
      timeout,
      "STORAGE_PERSISTENCE_TIMEOUT",
      "The browser storage protection check timed out",
    ).catch(() => null),
  ]);
  if (estimate === null) return { ...unknown, persisted };
  const quotaBytes = finiteOrNull(estimate.quota);
  const usageBytes = finiteOrNull(estimate.usage);
  return {
    quotaBytes,
    usageBytes,
    availableBytes:
      quotaBytes !== null && usageBytes !== null
        ? Math.max(0, quotaBytes - usageBytes)
        : null,
    persisted,
  };
}

export async function requestPersistentStorage(
  storage: PersistentStorageManager | undefined = navigator.storage,
  timeoutMs = 5_000,
  onLateGranted?: () => void,
): Promise<boolean> {
  return requestBrowserPersistentStorage(storage, timeoutMs, onLateGranted);
}

async function verifyOpfsLifecycle(
  storage: StorageManager,
  timeoutMs: number,
): Promise<boolean> {
  const directoryName = `.drowse-capability-${canarySuffix()}`;
  const fileName = "lifecycle.bin";
  const canary = async (): Promise<boolean> => {
    const root = await storage.getDirectory() as unknown as OpfsDirectoryLike;
    if (
      typeof root?.getDirectoryHandle !== "function" ||
      typeof root?.removeEntry !== "function"
    ) return false;
    let created = false;
    try {
      const directory = await root.getDirectoryHandle(directoryName, { create: true });
      created = true;
      if (
        typeof directory?.getFileHandle !== "function" ||
        typeof directory?.entries !== "function"
      ) return false;
      const handle = await directory.getFileHandle(fileName, { create: true });
      if (
        typeof handle?.createWritable !== "function" ||
        typeof handle?.getFile !== "function"
      ) return false;
      const writable = await handle.createWritable();
      try {
        await writable.write("dro");
        await writable.close();
      } catch (error) {
        try {
          await writable.abort?.();
        } catch {
          // Deleting the canary below is the authoritative cleanup path.
        }
        throw error;
      }
      const resumed = await handle.createWritable({ keepExistingData: true });
      if (
        typeof resumed.seek !== "function" ||
        typeof resumed.truncate !== "function"
      ) {
        await resumed.abort?.();
        return false;
      }
      try {
        await resumed.seek(3);
        await resumed.write("wse-extra");
        await resumed.truncate(6);
        await resumed.close();
      } catch (error) {
        try {
          await resumed.abort?.();
        } catch {
          // Removing the scoped directory below is the authoritative cleanup path.
        }
        throw error;
      }
      const file = await handle.getFile();
      const contents = new TextDecoder().decode(await file.arrayBuffer());
      let listed = false;
      for await (const [name] of directory.entries()) {
        if (name === fileName) listed = true;
      }
      return contents === "drowse" && listed;
    } finally {
      if (created) await root.removeEntry(directoryName, { recursive: true });
    }
  };
  try {
    return await withTimeout(
      canary(),
      timeoutMs,
      "OPFS_CANARY_TIMEOUT",
      "The private-storage lifecycle check timed out",
    );
  } catch {
    return false;
  }
}

function canarySuffix(): string {
  const bytes = new Uint32Array(2);
  globalThis.crypto?.getRandomValues?.(bytes);
  return `${Date.now().toString(36)}-${bytes[0].toString(36)}${bytes[1].toString(36)}`;
}

function hardUnless(
  condition: boolean,
  code: string,
  messageText: string,
  issues: CapabilityIssue[],
): void {
  if (!condition) issues.push(hard(code, messageText));
}

function hard(code: string, messageText: string): CapabilityIssue {
  return { code, message: messageText, severity: "hard" };
}

function advisory(code: string, messageText: string): CapabilityIssue {
  return { code, message: messageText, severity: "advisory" };
}

function operationAvailability(
  hardFailures: CapabilityIssue[],
): Record<RuntimeCapabilityOperation, { available: boolean; reasons: CapabilityIssue[] }> {
  const unavailable = (code: string, messageText: string) => ({
    available: false,
    reasons: [...hardFailures, hard(code, messageText)],
  });
  const distributionProvisioned =
    BROWSER_RUNTIME_RELEASE_VERIFIED &&
    HOSTED_DISTRIBUTION_CONFIG.status === "verified" &&
    HOSTED_DISTRIBUTION_CONFIG.publicKeys.size === 2 &&
    HOSTED_DISTRIBUTION_CONFIG.allowedCatalogRedirectOrigins.size > 0 &&
    HOSTED_DISTRIBUTION_CONFIG.allowedArtifactRedirectOrigins.size > 0;
  const generationIssue = generationReleaseIssue(
    BROWSER_RUNTIME_RELEASE_VERIFIED,
    BROWSER_MODEL_BACKEND_INTEGRATED,
  );
  const generation = generationIssue === null
    ? { available: hardFailures.length === 0, reasons: [...hardFailures] }
    : { available: false, reasons: [...hardFailures, generationIssue] };
  const geometryProfilesReady = structuredHookProfilesSupportSubspaceTrails(
    BROWSER_RUNTIME_LOCK.models.map((model) => model.structuredHookProfile),
  );
  const probeSubspaceTrails = geometryProfilesReady
    ? { available: generation.available, reasons: [...generation.reasons] }
    : {
        available: false,
        reasons: [
          ...generation.reasons,
          hard(
            "HOSTED_SUBSPACE_COORDINATES_UNAVAILABLE",
            "The locked browser model profiles do not expose the exact geometry distances required for subspace trails",
          ),
        ],
      };
  const fittingReady = browserAuthoringAvailableForChannel({
    runtimeIntegrated: BROWSER_AUTHORING_RUNTIME_INTEGRATED,
    modelBackendIntegrated: BROWSER_MODEL_BACKEND_INTEGRATED,
  });
  return {
    download: distributionProvisioned
      ? { available: hardFailures.length === 0, reasons: [...hardFailures] }
      : unavailable(
          "SIGNED_DISTRIBUTION_UNAVAILABLE",
          "Signed production catalogs and verified model artifacts have not been provisioned",
        ),
    generation,
    fitting: fittingReady
      ? { available: hardFailures.length === 0, reasons: [...hardFailures] }
      : unavailable(
          BROWSER_AUTHORING_RUNTIME_INTEGRATED && BROWSER_MODEL_BACKEND_INTEGRATED
            ? "BROWSER_FITTING_UNAVAILABLE"
            : "BROWSER_FITTING_RUNTIME_UNAVAILABLE",
          BROWSER_AUTHORING_RUNTIME_INTEGRATED && BROWSER_MODEL_BACKEND_INTEGRATED
            ? "Browser manifold fitting is unavailable in this build"
            : "Browser manifold fitting is not connected to the active model runtime",
        ),
    manifold_artifacts: BROWSER_ARTIFACT_AUTHORING_INTEGRATED
      ? { available: hardFailures.length === 0, reasons: [...hardFailures] }
      : unavailable(
          "MANIFOLD_ARTIFACT_BRIDGE_UNAVAILABLE",
          "Portable manifold archives are not connected to browser-local artifact storage",
        ),
    probe_subspace_trails: probeSubspaceTrails,
    jlens_fitting: {
      available: false,
      reasons: [hard("HOSTED_JLENS_FITTING_EXCLUDED", "J-lens fitting is available only in the Python runtime")],
    },
    sae_training: {
      available: false,
      reasons: [hard("HOSTED_SAE_TRAINING_EXCLUDED", "SAE training is available only in the Python runtime")],
    },
    session_admin: {
      available: false,
      reasons: [hard("HOSTED_SESSION_ADMIN_EXCLUDED", "The browser runtime has no server sessions or API keys")],
    },
    server_endpoints: {
      available: false,
      reasons: [hard("HOSTED_SERVER_ENDPOINTS_EXCLUDED", "The browser runtime does not expose HTTP inference endpoints")],
    },
  };
}

export function structuredHookProfilesSupportSubspaceTrails(
  profileIds: readonly unknown[],
): boolean {
  if (profileIds.length === 0) return false;
  return profileIds.every((profileId) => {
    let profile;
    try {
      profile = resolveStructuredHookProfile(profileId);
    } catch {
      return false;
    }
    return profile.maxGeometryProbes > 0 &&
      profile.maxRank > 0 &&
      profile.maxGeometryCandidates >= profile.maxRank + 1 &&
      profile.geometryOutputStride >=
        4 + profile.maxIntrinsicDim + profile.maxGeometryCandidates;
  });
}

export function generationReleaseIssue(
  runtimeVerified: boolean,
  backendIntegrated: boolean,
): CapabilityIssue | null {
  if (!runtimeVerified) {
    return hard(
      "CUSTOM_MLC_BACKEND_UNAVAILABLE",
      "The selected Drowse browser runtime has not been published yet",
    );
  }
  if (!backendIntegrated) {
    return hard(
      "CUSTOM_MLC_BACKEND_UNINTEGRATED",
      "The published Drowse MLC runtime has not been integrated into this browser build",
    );
  }
  return null;
}

function snapshotLimits(value: object): Record<string, number> {
  const limits: Record<string, number> = {};
  const source = value as Record<string, unknown>;
  for (const name of CATALOG_WEBGPU_LIMIT_NAMES) {
    const limit = source[name];
    if (typeof limit === "number" && Number.isFinite(limit)) limits[name] = limit;
  }
  return limits;
}

async function adapterInformation(
  adapter: GpuAdapterLike,
  timeoutMs: number,
): Promise<GpuAdapterInfoLike | null> {
  if (adapter.info) return adapter.info;
  if (!adapter.requestAdapterInfo) return null;
  try {
    return await withTimeout(
      adapter.requestAdapterInfo(),
      timeoutMs,
      "WEBGPU_ADAPTER_INFO_TIMEOUT",
      "Timed out while reading WebGPU adapter information",
    );
  } catch {
    return null;
  }
}

async function calibrateCompute(
  device: GpuDeviceLike,
  repeatReadbacks: boolean,
): Promise<number | null> {
  const usage = (globalThis as unknown as {
    GPUBufferUsage?: {
      STORAGE: number;
      COPY_SRC: number;
      COPY_DST: number;
      MAP_READ: number;
    };
  }).GPUBufferUsage;
  const mapMode = (globalThis as unknown as {
    GPUMapMode?: { READ: number };
  }).GPUMapMode;
  if (
    typeof usage?.STORAGE !== "number" ||
    typeof usage.COPY_SRC !== "number" ||
    typeof usage.COPY_DST !== "number" ||
    typeof usage.MAP_READ !== "number" ||
    typeof mapMode?.READ !== "number"
  ) return null;
  const elementCount = 4096;
  const rounds = repeatReadbacks ? 4 : 1;
  const dispatchesPerRound = 16 / rounds;
  const byteLength = elementCount * Float32Array.BYTES_PER_ELEMENT;
  const buffer = device.createBuffer({
    size: byteLength,
    usage: usage.STORAGE | usage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: byteLength,
    usage: usage.COPY_DST | usage.MAP_READ,
  });
  try {
    const module = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read_write> values: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          if (id.x < ${elementCount}u) {
            var value = values[id.x];
            for (var i = 0u; i < 64u; i = i + 1u) {
              value = value * 1.000001 + f32(i) * 0.000001;
            }
            values[id.x] = value;
          }
        }
      `,
    });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer } }],
    });
    const started = performance.now();
    for (let round = 0; round < rounds; round += 1) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      for (let dispatch = 0; dispatch < dispatchesPerRound; dispatch += 1) {
        pass.dispatchWorkgroups(elementCount / 64);
      }
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();

      const readbackEncoder = device.createCommandEncoder();
      readbackEncoder.copyBufferToBuffer(buffer, 0, readback, 0, byteLength);
      device.queue.submit([readbackEncoder.finish()]);
      await readback.mapAsync(mapMode.READ);
      try {
        const sample = new Float32Array(
          readback.getMappedRange(0, Float32Array.BYTES_PER_ELEMENT),
          0,
          1,
        )[0];
        if (!Number.isFinite(sample)) {
          throw new Error("The WebGPU compute calibration returned an invalid value");
        }
      } finally {
        readback.unmap();
      }
    }
    const elapsed = Math.max(0.1, performance.now() - started);
    return Math.round(
      ((elementCount * rounds * dispatchesPerRound * 64) / elapsed) * 1000,
    );
  } finally {
    readback.destroy();
    buffer.destroy();
  }
}

function coarseSignature(value: {
  fallback: string;
  adapterInfo: RuntimeCapabilities["webGpu"]["adapterInfo"];
  features: string[];
  limits: Record<string, number>;
  deviceMemoryGiB: number | null;
  logicalCpuCount: number | null;
  mobile: boolean;
  runtimeClass: BrowserRuntimeClass;
  appleMobile: boolean;
  browser: string;
}): string {
  const buckets = {
    fallback: value.fallback,
    adapterVendor: value.adapterInfo?.vendor ?? null,
    adapterArchitecture: value.adapterInfo?.architecture ?? null,
    adapterDevice: value.adapterInfo?.device ?? null,
    features: value.features,
    maxBufferSize: bucket(value.limits.maxBufferSize),
    maxStorageBufferBindingSize: bucket(value.limits.maxStorageBufferBindingSize),
    maxStorageBuffersPerShaderStage: bucket(
      value.limits.maxStorageBuffersPerShaderStage,
    ),
    deviceMemoryGiB: value.deviceMemoryGiB,
    logicalCpuCount: value.logicalCpuCount,
    mobile: value.mobile,
    runtimeClass: value.runtimeClass,
    appleMobile: value.appleMobile,
    browser: value.browser,
  };
  let hash = 2166136261;
  for (const character of JSON.stringify(buckets)) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return `device-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function adapterFingerprint(
  fallback: RuntimeCapabilities["webGpu"]["fallback"],
  adapterInfo: RuntimeCapabilities["webGpu"]["adapterInfo"],
  features: string[],
  limits: Record<string, number>,
): string {
  return JSON.stringify({
    fallback,
    vendor: adapterInfo?.vendor ?? null,
    architecture: adapterInfo?.architecture ?? null,
    device: adapterInfo?.device ?? null,
    features,
    limits: Object.fromEntries(
      CATALOG_WEBGPU_LIMIT_NAMES.map((name) => [name, bucket(limits[name])]),
    ),
  });
}

function fallbackStatus(
  adapter: GpuAdapterLike,
  info: GpuAdapterInfoLike | null,
): RuntimeCapabilities["webGpu"]["fallback"] {
  if (typeof info?.isFallbackAdapter === "boolean") {
    return info.isFallbackAdapter ? "fallback" : "hardware";
  }
  if (typeof adapter.isFallbackAdapter === "boolean") {
    return adapter.isFallbackAdapter ? "fallback" : "hardware";
  }
  return "unknown";
}

function browserIdentity(userAgent: string): string {
  for (const [name, pattern] of [
    ["edge", /(?:Edg|EdgA|EdgiOS)\/([0-9]+)/],
    ["chrome", /(?:Chrome|CriOS)\/([0-9]+)/],
    ["firefox", /(?:Firefox|FxiOS)\/([0-9]+)/],
    ["safari", /Version\/([0-9]+).*Safari/],
  ] as const) {
    const match = userAgent.match(pattern);
    if (match) return `${name}-${match[1]}`;
  }
  return "unknown";
}

function browserPlatform(
  userAgent: string,
  navigatorSignals: NavigatorWithRuntimeSignals,
): { runtimeClass: BrowserRuntimeClass; appleMobile: boolean } {
  const appleMobile = isAppleMobileBrowser(
    userAgent,
    navigatorSignals.platform,
    navigatorSignals.maxTouchPoints,
  );
  if (appleMobile) {
    return { runtimeClass: "apple-mobile-webkit", appleMobile: true };
  }
  const browser = browserIdentity(userAgent);
  if (/Android/i.test(userAgent) &&
    (browser.startsWith("chrome-") || browser.startsWith("edge-"))) {
    return { runtimeClass: "android-chromium", appleMobile: false };
  }
  if (browser.startsWith("safari-")) {
    return { runtimeClass: "desktop-webkit", appleMobile: false };
  }
  if (browser.startsWith("firefox-") && !/Android|Mobile/i.test(userAgent)) {
    return { runtimeClass: "desktop-gecko", appleMobile: false };
  }
  if (browser.startsWith("chrome-") || browser.startsWith("edge-")) {
    return { runtimeClass: "desktop-chromium", appleMobile: false };
  }
  return { runtimeClass: "other", appleMobile: false };
}

export function browserRuntimeClass(
  userAgent = navigator.userAgent,
  platform = navigator.platform,
  maxTouchPoints = navigator.maxTouchPoints,
): BrowserRuntimeClass {
  return browserPlatform(userAgent, { platform, maxTouchPoints }).runtimeClass;
}

function requiresVerifiedCompute(runtimeClass: BrowserRuntimeClass): boolean {
  return runtimeClass === "apple-mobile-webkit" ||
    runtimeClass === "desktop-webkit" ||
    runtimeClass === "desktop-gecko";
}

export function isAppleMobileBrowser(
  userAgent = navigator.userAgent,
  platform = navigator.platform,
  maxTouchPoints = navigator.maxTouchPoints,
): boolean {
  return /iPhone|iPad|iPod/i.test(userAgent) || /^(iPhone|iPad|iPod)$/i.test(platform) ||
    (/Macintosh/i.test(userAgent) && /Mobile\//i.test(userAgent)) ||
    (maxTouchPoints > 1 &&
      (platform === "MacIntel" || /Macintosh|Mac OS X/i.test(userAgent)));
}

function bucket(value: number | undefined): number | null {
  if (!value || value <= 0) return null;
  return 2 ** Math.floor(Math.log2(value));
}

function finiteOrNull(value: number | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function optionalText(value: string | undefined): string | undefined {
  return value?.trim() || undefined;
}

function adapterInfoSnapshot(
  info: GpuAdapterInfoLike,
): NonNullable<RuntimeCapabilities["webGpu"]["adapterInfo"]> {
  const result: NonNullable<RuntimeCapabilities["webGpu"]["adapterInfo"]> = {};
  const vendor = optionalText(info.vendor);
  const architecture = optionalText(info.architecture);
  const device = optionalText(info.device);
  const description = optionalText(info.description);
  if (vendor !== undefined) result.vendor = vendor;
  if (architecture !== undefined) result.architecture = architecture;
  if (device !== undefined) result.device = device;
  if (description !== undefined) result.description = description;
  return result;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

class CapabilityTimeoutError extends Error {
  constructor(readonly code: string, messageText: string) {
    super(messageText);
    this.name = "CapabilityTimeoutError";
  }
}

function loadError(code: string, messageText: string): Error & { code: string } {
  return Object.assign(new Error(messageText), { code });
}

function withTimeout<T>(
  operation: Promise<T>,
  timeoutMs: number,
  code: string,
  messageText: string,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new CapabilityTimeoutError(code, messageText)),
      timeoutMs,
    );
    operation.then(
      (value) => {
        clearTimeout(timeout);
        resolve(value);
      },
      (error) => {
        clearTimeout(timeout);
        reject(error);
      },
    );
  });
}

function positiveTimeout(value: number | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError("Capability timeouts must be positive numbers");
  }
  return value;
}
