import type {
  CatalogDocument,
  CatalogModel,
  DeviceLoadRecord,
  DownloadProgress,
  DownloadWorkerResult,
  HostedController,
  ModelPreference,
  ModelRecommendation,
  PackDownloadWorkerResult,
  RuntimeCapabilities,
  RuntimeSnapshot,
  VerifiedCatalog,
} from "../../lib/runtime/contracts";
import { rankCatalogModels } from "../../lib/runtime/recommendation";
import { initialRuntimeSnapshot } from "../../lib/runtime/state";
import { DeterministicFakeRuntime } from "./fakeRuntime";
import type { HostedRuntimeBundle } from "./hostedController";

const SHA = "a".repeat(64);
const REVISION = "b".repeat(40);
const RUNTIME_ABI = "drowse-web-fixture-v1";
const VARIANT_ID = "qwen3-1.7b-fixture";

const runtimeIdentity = {
  sourceModel: "fixture/Qwen3-1.7B",
  sourceRevision: REVISION,
  convertedManifestSha256: SHA,
  quantization: "fixture-fp32",
  tokenizerSha256: SHA,
  chatTemplateSha256: SHA,
  modelLibrarySha256: SHA,
  runtimeAbi: RUNTIME_ABI,
  hookAbi: "post-block-residual-v4",
  hiddenSize: 64,
  layerMap: [0, 1],
};

const fixtureModel: CatalogModel = {
  id: "qwen3-1.7b",
  displayName: "Qwen3 1.7B fixture",
  description: "Deterministic development-only runtime",
  sourceUrl: "https://huggingface.co/Qwen/Qwen3-1.7B",
  license: "Apache-2.0",
  languages: ["en"],
  variants: [{
    id: VARIANT_ID,
    tier: "fastest",
    structuredHookProfile: "standard-v1",
    thinkingProfile: null,
    contextProfiles: [{
      contextTokens: 2048,
      bindingSha256: SHA,
      minimumCalibrationScore: null,
      minimumDeviceMemoryGiB: null,
      expectedPrefillTokensPerSecond: [100, 140],
      expectedDecodeTokensPerSecond: [40, 56],
      measuredDevices: 1,
    }],
    downloadBytes: 1_000_000,
    requiredCorePackBytes: 100_000,
    requirements: {
      features: [],
      limits: {
        maxBufferSize: 21_472,
        maxStorageBufferBindingSize: 21_472,
        maxStorageBuffersPerShaderStage: 10,
        maxComputeWorkgroupStorageSize: 32_768,
      },
    },
    runtimeIdentity,
    runtimeIdentitySha256: SHA,
    files: [],
    packs: [{
      id: "fixture-core",
      kind: "core",
      displayName: "Fixture core geometry",
      license: "AGPL-3.0-or-later",
      sourceRepository: "a9lim/drowse-web-fixture",
      sourceRevision: REVISION,
      bytes: 100_000,
      required: true,
      runtimeIdentitySha256: SHA,
      compatibleContextBindingSha256: [SHA],
      files: [],
    }],
  }],
};

const document: CatalogDocument = {
  schemaVersion: 1,
  sequence: 1,
  issuedAt: "2026-01-01T00:00:00.000Z",
  expiresAt: "2099-01-01T00:00:00.000Z",
  runtimeAbi: RUNTIME_ABI,
  models: [fixtureModel],
};

const catalog: VerifiedCatalog = {
  document,
  exactBytes: new TextEncoder().encode(JSON.stringify(document)),
  signature: {
    schemaVersion: 1,
    algorithm: "Ed25519",
    keyId: "development-fixture",
    signature: "development-only",
  },
  allowDownloads: true,
  stale: false,
};

const capabilities: RuntimeCapabilities = {
  checkedAt: 0,
  supported: true,
  secureContext: true,
  crossOriginIsolated: true,
  dedicatedWorker: true,
  indexedDb: true,
  opfs: true,
  webLocks: true,
  broadcastChannel: true,
  webAssembly: true,
  webGpu: {
    available: true,
    fallback: "hardware",
    features: [],
    limits: {
      maxBufferSize: 21_472,
      maxStorageBufferBindingSize: 21_472,
      maxStorageBuffersPerShaderStage: 10,
      maxComputeWorkgroupStorageSize: 32_768,
    },
    adapterInfo: { vendor: "deterministic fixture" },
  },
  storage: {
    quotaBytes: 2_000_000_000,
    usageBytes: 0,
    availableBytes: 2_000_000_000,
    persisted: true,
  },
  signals: {
    deviceMemoryGiB: 8,
    logicalCpuCount: 8,
    mobile: false,
    language: "en-US",
    calibrationScore: 100,
  },
  deviceSignature: "development-fixture-device",
  runtimeVersion: RUNTIME_ABI,
  issues: [],
  operations: {
    download: { available: true, reasons: [] },
    generation: { available: true, reasons: [] },
    fitting: {
      available: false,
      reasons: [{
        code: "FIXTURE_FITTING_DISABLED",
        message: "The deterministic development fixture does not run fitting jobs",
        severity: "hard",
      }],
    },
    manifold_artifacts: { available: true, reasons: [] },
    probe_subspace_trails: { available: true, reasons: [] },
    jlens_fitting: { available: false, reasons: [] },
    sae_training: { available: false, reasons: [] },
    session_admin: { available: false, reasons: [] },
    server_endpoints: { available: false, reasons: [] },
  },
  limits: { manifoldFitMaxIntrinsicDim: 4 },
};

class FixtureHostedController implements HostedController {
  private listeners = new Set<(snapshot: RuntimeSnapshot) => void>();
  private installed = new Set<string>();
  private installedPacks = new Set<string>();
  private loadRecords: DeviceLoadRecord[] = [];
  private current = initialRuntimeSnapshot();
  private cancelled = false;

  get snapshot(): RuntimeSnapshot {
    return this.current;
  }

  subscribe(listener: (snapshot: RuntimeSnapshot) => void): () => void {
    this.listeners.add(listener);
    listener(this.current);
    return () => this.listeners.delete(listener);
  }

  async check(): Promise<RuntimeCapabilities> {
    this.patch({ lifecycle: "unloaded" });
    return { ...capabilities, checkedAt: Date.now() };
  }

  async catalog(): Promise<VerifiedCatalog> {
    return catalog;
  }

  recommend(
    verified: VerifiedCatalog,
    preference: ModelPreference,
    contextTokens = 2048,
    language = "en-US",
    options: { explicitOomRetry?: boolean } = {},
  ): ModelRecommendation[] {
    return rankCatalogModels(verified.document.models, {
      capabilities,
      contextTokens,
      loadRecords: this.loadRecords,
      preference,
      language,
      explicitOomRetry: options.explicitOomRetry,
      installedModelVariantIds: [...this.installed],
    });
  }

  async download(
    modelVariantId: string,
    onProgress?: (progress: DownloadProgress) => void,
  ): Promise<DownloadWorkerResult> {
    if (modelVariantId !== VARIANT_ID) throw new Error("Unknown fixture model");
    this.cancelled = false;
    const total = 1_100_000;
    for (const received of [200_000, 700_000, total]) {
      await new Promise<void>((resolve) => setTimeout(resolve, 8));
      if (this.cancelled) {
        return { modelVariantId, installed: false, cancelled: true };
      }
      onProgress?.({
        modelVariantId,
        files: [{
          path: "fixture/model.bin",
          bytesReceived: received,
          bytesTotal: total,
          resumable: true,
          verification: received === total ? "verified" : "pending",
        }],
        bytesReceived: received,
        bytesTotal: total,
        throughputBytesPerSecond: 20_000_000,
        etaSeconds: received === total ? [0, 0] : [1, 2],
        calculatingEta: false,
        stalled: false,
        offline: false,
        resumable: true,
      });
    }
    this.installed.add(modelVariantId);
    this.patch({
      selectedModelVariantId: modelVariantId,
      installedModelVariantIds: [...this.installed],
    });
    return { modelVariantId, installed: true, cancelled: false };
  }

  async downloadPack(
    modelVariantId: string,
    packId: string,
  ): Promise<PackDownloadWorkerResult> {
    throw new Error(
      `The development fixture has no optional pack ${packId} for ${modelVariantId}`,
    );
  }

  async cancelDownload(): Promise<void> {
    this.cancelled = true;
  }

  async cancelFitting(): Promise<void> {
    if (
      this.current.fitting.phase !== "running" &&
      this.current.fitting.phase !== "cancelling"
    ) return;
    this.patch({
      fitting: {
        ...this.current.fitting,
        phase: "complete",
        finishedAt: Date.now(),
        error: null,
      },
    });
  }

  async deleteModel(modelVariantId: string): Promise<void> {
    this.installed.delete(modelVariantId);
    this.patch({
      selectedModelVariantId:
        this.current.selectedModelVariantId === modelVariantId
          ? null
          : this.current.selectedModelVariantId,
      installedModelVariantIds: [...this.installed],
    });
  }

  async deletePack(packId: string): Promise<void> {
    this.installedPacks.delete(packId);
    this.patch({ installedPackIds: [...this.installedPacks] });
  }

  async clearAll(): Promise<void> {
    this.installed.clear();
    this.installedPacks.clear();
    this.loadRecords = [];
    this.patch({
      ...initialRuntimeSnapshot(),
      lifecycle: "unloaded",
    });
  }

  async requestPersistence(): Promise<boolean> {
    return true;
  }

  async refreshStorage(): Promise<RuntimeCapabilities["storage"]> {
    return capabilities.storage;
  }

  async load(modelVariantId: string, contextTokens: number): Promise<void> {
    if (!this.installed.has(modelVariantId)) throw new Error("Fixture model is not installed");
    this.patch({ lifecycle: "loading", modelVariantId, contextTokens });
    await new Promise<void>((resolve) => setTimeout(resolve, 8));
    const record: DeviceLoadRecord = {
      modelVariantId,
      runtimeIdentitySha256: SHA,
      deviceSignature: capabilities.deviceSignature,
      contextTokens,
      result: "success",
      prefillTokensPerSecond: 120,
      decodeTokensPerSecond: 48,
      recordedAt: Date.now(),
    };
    this.loadRecords.push(record);
    this.patch({ lifecycle: "ready", loadRecords: [...this.loadRecords] });
  }

  async unload(): Promise<void> {
    this.patch({ lifecycle: "unloaded", modelVariantId: null, contextTokens: null });
  }

  async dispose(): Promise<void> {
    this.listeners.clear();
  }

  private patch(patch: Partial<RuntimeSnapshot>): void {
    this.current = { ...this.current, ...patch };
    for (const listener of this.listeners) listener(this.current);
  }
}

export function createFixtureHostedRuntime(withInstruments = false, baseModel = false): HostedRuntimeBundle {
  const controller = new FixtureHostedController();
  const runtime = new DeterministicFakeRuntime({
    ...(baseModel ? { rootId: crypto.randomUUID(), tokenDelayMs: 150 } : {}),
    modelId: baseModel ? "fixture/pythia-70m-base" : VARIANT_ID,
    ...(baseModel ? { sessionOverrides: { is_base_model: true, supports_thinking: false, scene_mode: false, role_substitution_supported: false, user_role_supported: false, default_user_role: null, default_assistant_role: null } } : {}),
    instrumentPacks: withInstruments ? [
      { id: "fixture-jlens", kind: "jlens", displayName: "Fixture J-lens" },
      { id: "fixture-sae", kind: "sae", displayName: "Fixture SAE" },
    ] : [],
  });
  let disposed = false;
  return {
    controller,
    runtime,
    async dispose() {
      if (disposed) return;
      disposed = true;
      await controller.dispose();
      await runtime.dispose();
    },
  };
}
