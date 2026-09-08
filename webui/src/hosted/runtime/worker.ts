import type {
  CatalogInstrumentPack,
  CatalogModel,
  CatalogWorkerRequest,
  DeviceLoadRecord,
  DownloadWorkerResult,
  PackDownloadWorkerResult,
  RuntimeCapabilities,
  RuntimeCapabilityOperation,
  RuntimeFailure,
  RuntimeOperation,
  RuntimeOperationPhase,
  RuntimeProgressEvent,
  RuntimeServiceRequest,
  RuntimeSnapshot,
  ModelVariant,
  VerifiedCatalog,
  WorkerErrorResponse,
  WorkerEventEnvelope,
  WorkerRequest,
  WorkerSuccessResponse,
} from "../../lib/runtime/contracts";
import {
  isRuntimeServiceMethod,
  RUNTIME_PROTOCOL_VERSION,
} from "../../lib/runtime/contracts";
import { MEBIBYTE } from "../../lib/runtime/contracts";
import { selectorReferencesManifold } from "../../lib/manifolds/selectors";
import { initialRuntimeSnapshot, runtimeFailure } from "../../lib/runtime/state";
import { assessModelVariant } from "../../lib/runtime/recommendation";
import { outputTokenLimitForSignals } from "../../lib/runtime/outputTokenPolicy";
import {
  CatalogValidationError,
  runtimeIdentitySha256,
} from "../../lib/runtime/catalog";
import { BrowserCapabilityChecker, requestPersistentStorage, storageEstimate } from "./capabilities";
import type { GpuAdapterLike } from "./capabilities";
import {
  CatalogFetchError,
  VerifiedCatalogRepository,
} from "./catalogRepository";
import {
  BrowserContentStore,
  type ContentAddressedStore,
} from "./contentStore";
import { HOSTED_DISTRIBUTION_CONFIG } from "./distributionConfig";
import { withHostedDestructiveSharedLock } from "./ownership";
import { randomUuid } from "./randomId";
import {
  ArtifactDownloadError,
  VerifiedArtifactDownloader,
  selectDownloadManifest,
  selectOptionalPackManifest,
  type ArtifactDownloadOptions,
  type ArtifactDownloadResult,
} from "./downloader";
import {
  UnavailableBrowserModelBackend,
  type BrowserGenerationPerformance,
  type BrowserModelArtifact,
  type BrowserModelBackend,
  type BrowserInstrumentPackArtifacts,
  type BrowserModelLoadRequest,
  type BrowserModelLoadResult,
} from "./modelBackend";
import {
  BrowserSessionCoordinator,
  mutatesPersistedHostedSession,
  type HostedSessionRestoreResult,
  type HostedSessionRuntimeIdentity,
  type BrowserSessionCoordinatorOptions,
  type SessionCoordinatorPersistencePort,
} from "./sessionCoordinator";
import { BrowserSessionPersistence } from "./sessionPersistence";
import {
  isFittingLifecycleRequest,
  runtimeServiceOperationPolicy,
  runtimeServiceRoute,
} from "./operationPolicy";
import { BrowserDrowseArchiveService } from "../artifacts/service";
import { optionalPackHardwareBlock } from "./optionalPackCompatibility";
import { clearOwnedRuntimeStorage } from "./legacyRuntimeStorage";
import { LEGACY_PRODUCT_SLUGS } from "./brandMigration";
import {
  ActivationSpool,
  BrowserActivationSpool,
} from "../fitting/activationSpool";
import type {
  WSServerMessage,
  WSGenerateRequest,
  WSSubmitRequest,
  WSTreeMutatedEvent,
} from "../../lib/types";

export interface WorkerScopeLike {
  postMessage(message: unknown): void;
  addEventListener(type: "message", listener: (event: MessageEvent<unknown>) => void): void;
}

interface CapabilityCheckerLike {
  check(): Promise<RuntimeCapabilities>;
  clearAdapter(): void;
  adapterForLoad?(): GpuAdapterLike | Promise<GpuAdapterLike>;
}

interface CatalogRepositoryLike {
  get(request?: CatalogWorkerRequest): Promise<VerifiedCatalog>;
  withDownloadAuthorization<T>(
    catalog: VerifiedCatalog,
    operation: () => Promise<T>,
    signal?: AbortSignal,
  ): Promise<T>;
  clear(): Promise<void>;
}

interface ArtifactDownloaderLike {
  download(
    catalog: VerifiedCatalog,
    modelVariantId: string,
    options: ArtifactDownloadOptions,
  ): Promise<ArtifactDownloadResult>;
  downloadPack(
    catalog: VerifiedCatalog,
    modelVariantId: string,
    packId: string,
    options: ArtifactDownloadOptions,
  ): Promise<ArtifactDownloadResult>;
  cancel(): boolean;
}

export interface ArtifactServicePort {
  handles(request: RuntimeServiceRequest): boolean;
  request(
    request: RuntimeServiceRequest,
    onProgress: (event: RuntimeProgressEvent) => void,
  ): Promise<unknown>;
  clear(): Promise<void>;
}

interface SessionCoordinatorLike {
  readonly modelDefaults?: RuntimeSnapshot["modelDefaults"];
  restore(
    identity: HostedSessionRuntimeIdentity,
    mode?: "replace" | "reset",
    compatibleRuntimeIdentitySha256s?: readonly string[],
  ): Promise<HostedSessionRestoreResult>;
  persist(identity: HostedSessionRuntimeIdentity): Promise<unknown>;
  close(): Promise<void>;
}

type SessionCoordinatorFactory = (
  persistence: SessionCoordinatorPersistencePort,
  execute: (request: RuntimeServiceRequest) => Promise<unknown>,
  options?: BrowserSessionCoordinatorOptions,
) => SessionCoordinatorLike;

interface LoadedModelSelection {
  model: CatalogModel;
  variant: ModelVariant;
  requiredCorePack: CatalogInstrumentPack;
  contextTokens: number;
  artifacts: BrowserModelArtifact[];
  optionalPacks: BrowserInstrumentPackArtifacts[];
}

export interface HostedRuntimeWorkerDependencies {
  capabilityChecker?: CapabilityCheckerLike;
  contentStore?: ContentAddressedStore;
  catalogRepository?: CatalogRepositoryLike;
  downloader?: ArtifactDownloaderLike;
  artifactService?: ArtifactServicePort;
  activationSpool?: ActivationSpool;
  storageEstimator?: () => Promise<RuntimeCapabilities["storage"]>;
  modelBackend?: BrowserModelBackend;
  sessionPersistence?: SessionCoordinatorPersistencePort;
  sessionCoordinatorFactory?: SessionCoordinatorFactory;
  clearRuntimeStorage?: () => Promise<void>;
}

export class HostedRuntimeWorker {
  private readonly scope: WorkerScopeLike;
  private readonly capabilityChecker: CapabilityCheckerLike;
  private readonly contentStore: ContentAddressedStore;
  private readonly catalogRepository: CatalogRepositoryLike;
  private readonly downloader: ArtifactDownloaderLike;
  private readonly artifactService: ArtifactServicePort;
  private readonly activationSpool: ActivationSpool;
  private readonly storageEstimator: () => Promise<RuntimeCapabilities["storage"]>;
  private readonly modelBackend: BrowserModelBackend;
  private readonly sessionPersistence: SessionCoordinatorPersistencePort;
  private readonly sessionCoordinatorFactory: SessionCoordinatorFactory;
  private readonly clearRuntimeStorage: () => Promise<void>;
  private acceptedCatalog: VerifiedCatalog | null = null;
  private checkedCapabilities: RuntimeCapabilities | null = null;
  private admittedDownloadRequestId: string | null = null;
  private cancelAdmittedDownload = false;
  private admittedGenerationRequestId: string | null = null;
  private cancelAdmittedGeneration = false;
  private admittedFittingRequestId: string | null = null;
  private cancelAdmittedFitting = false;
  private activeDownloadSettled: Promise<void> | null = null;
  private resolveActiveDownload: (() => void) | null = null;
  private activeFittingController: AbortController | null = null;
  private activeFittingSettled: Promise<void> | null = null;
  private activeGenerationSettled: Promise<void> | null = null;
  private generationDeviceLoss: {
    backendEpoch: number;
    failure: RuntimeFailure;
  } | null = null;
  private snapshot: RuntimeSnapshot = initialRuntimeSnapshot();
  private sequence = 0;
  private activeGenerationId: string | null = null;
  private backendEpoch = 0;
  private backendLoaded = false;
  private loadedSelection: LoadedModelSelection | null = null;
  private sessionCoordinator: SessionCoordinatorLike | null = null;
  private sessionIdentity: HostedSessionRuntimeIdentity | null = null;
  private deviceLossRetryKey: string | null = null;
  private deviceLossRetryUsed = false;
  private readonly activeDeviceLossTasks = new Set<Promise<void>>();
  private activeBackendLoadController: AbortController | null = null;
  private activeBackendLoadSettled: Promise<void> | null = null;
  private activeBackendRequestController: AbortController | null = null;
  private activeBackendRequestSettled: Promise<void> | null = null;
  private activeModelLoadSettled: Promise<void> | null = null;
  private commandQueue: Promise<void> = Promise.resolve();
  private concurrentTreeMutationQueue: Promise<void> = Promise.resolve();
  private generationTreeEventBarrier: Promise<void> = Promise.resolve();
  private generationMutationAdmission: VoidDeferred | null = null;

  constructor(
    scope: WorkerScopeLike,
    dependencies: HostedRuntimeWorkerDependencies = {},
  ) {
    this.scope = scope;
    this.capabilityChecker = dependencies.capabilityChecker ?? new BrowserCapabilityChecker();
    this.contentStore = dependencies.contentStore ?? new BrowserContentStore();
    this.catalogRepository = dependencies.catalogRepository ?? new VerifiedCatalogRepository();
    this.downloader = dependencies.downloader ?? new VerifiedArtifactDownloader(
      this.contentStore,
      {
        allowedRedirectOrigins: [
          ...HOSTED_DISTRIBUTION_CONFIG.allowedArtifactRedirectOrigins,
        ],
      },
    );
    this.storageEstimator = dependencies.storageEstimator ?? storageEstimate;
    this.artifactService = dependencies.artifactService ?? new BrowserDrowseArchiveService();
    this.activationSpool = dependencies.activationSpool ?? new BrowserActivationSpool();
    this.modelBackend = dependencies.modelBackend ?? new UnavailableBrowserModelBackend();
    this.sessionPersistence = dependencies.sessionPersistence ?? new BrowserSessionPersistence();
    this.sessionCoordinatorFactory = dependencies.sessionCoordinatorFactory ?? (
      (persistence, execute, options) =>
        new BrowserSessionCoordinator(persistence, execute, options)
    );
    this.clearRuntimeStorage = dependencies.clearRuntimeStorage ?? clearOwnedRuntimeStorage;
    this.scope.addEventListener("message", (event) => {
      if (!validRequest(event.data)) {
        const requestId = requestIdFrom(event.data);
        if (requestId) {
          this.error(requestId, runtimeFailure(
            "INVALID_WORKER_REQUEST",
            "The browser runtime received an invalid command envelope",
            false,
            400,
          ));
        }
        return;
      }
      const request = event.data;
      const destructiveRuntimeCommand =
        request.command === "unload" || request.command === "takeover" ||
        request.command === "clear";
      if (destructiveRuntimeCommand) {
        this.activeBackendLoadController?.abort();
        this.activeBackendRequestController?.abort();
        if (this.admittedGenerationRequestId !== null) {
          this.cancelAdmittedGeneration = true;
          if (this.activeGenerationSettled !== null) {
            void this.modelBackend.stop("internal").catch(() => undefined);
          }
        }
      }
      if (
        request.command === "stop" || request.command === "cancel" ||
        request.command === "cancel_fitting"
      ) {
        void this.handle(request);
        return;
      }
      if (destructiveRuntimeCommand && this.admittedFittingRequestId !== null) {
        this.beginFittingCancellation(request.requestId);
      }
      if (request.command === "download" || request.command === "download_pack") {
        if (this.admittedDownloadRequestId !== null) {
          this.error(request.requestId, runtimeFailure(
            "DOWNLOAD_BUSY",
            "Another artifact download is already queued or running",
            true,
            409,
          ));
          return;
        }
        this.admittedDownloadRequestId = request.requestId;
      }
      if (request.command === "submit" || request.command === "generate") {
        if (this.admittedGenerationRequestId !== null) {
          this.error(request.requestId, runtimeFailure(
            "GENERATION_BUSY",
            "Another generation is already queued or running",
            true,
            409,
          ));
          return;
        }
        this.admittedGenerationRequestId = request.requestId;
        this.generationMutationAdmission = deferredVoid();
      }
      const fittingRequest = request.command === "request" &&
        isFittingLifecycleRequest(request.payload);
      if (fittingRequest) {
        if (this.admittedFittingRequestId !== null) {
          this.error(request.requestId, runtimeFailure(
            "FITTING_BUSY",
            "Another browser fitting job is already queued or running",
            true,
            409,
          ));
          return;
        }
        this.admittedFittingRequestId = request.requestId;
      }
      if (
        this.admittedGenerationRequestId !== null &&
        isConcurrentTreeMutationRequest(request)
      ) {
        const admission = this.generationMutationAdmission?.promise ?? Promise.resolve();
        const queued = this.concurrentTreeMutationQueue.then(
          async () => {
            await admission;
            await this.generationTreeEventBarrier;
            await this.handle(request, true);
          },
          async () => {
            await admission;
            await this.generationTreeEventBarrier;
            await this.handle(request, true);
          },
        );
        this.concurrentTreeMutationQueue = queued.catch(() => undefined);
        return;
      }
      const precedingTreeMutations = this.concurrentTreeMutationQueue;
      const queued = this.commandQueue.then(
        async () => {
          await precedingTreeMutations;
          await this.handle(request);
        },
        async () => {
          await precedingTreeMutations;
          await this.handle(request);
        },
      );
      this.commandQueue = request.command === "download" || request.command === "download_pack" ||
          request.command === "submit" || request.command === "generate" || fittingRequest
        ? queued.finally(() => {
            if (this.admittedDownloadRequestId === request.requestId) {
              this.admittedDownloadRequestId = null;
              this.cancelAdmittedDownload = false;
            }
            if (this.admittedGenerationRequestId === request.requestId) {
              this.admittedGenerationRequestId = null;
              this.cancelAdmittedGeneration = false;
            }
            if (this.admittedFittingRequestId === request.requestId) {
              this.admittedFittingRequestId = null;
              this.cancelAdmittedFitting = false;
            }
          })
        : queued;
    });
  }

  private async handle(
    request: WorkerRequest,
    concurrentTreeMutation = false,
  ): Promise<void> {
    try {
      switch (request.command) {
        case "check": {
          if (this.backendLoaded || this.snapshot.lifecycle === "loading") {
            this.error(request.requestId, runtimeFailure(
              "RUNTIME_IN_USE",
              "Unload the active browser model before rerunning compatibility checks",
              true,
              409,
            ));
            return;
          }
          this.patchSnapshot({ lifecycle: "checking", error: null }, request.requestId);
          const capabilities = await this.capabilityChecker.check();
          this.checkedCapabilities = capabilities;
          const hardFailure = capabilities.issues.find((item) => item.severity === "hard");
          this.patchSnapshot(
            {
              lifecycle: capabilities.supported ? "unloaded" : "failed",
              modelVariantId: null,
              contextTokens: null,
              error: hardFailure
                ? runtimeFailure(hardFailure.code, hardFailure.message, false, 412)
                : null,
            },
            request.requestId,
          );
          this.success(request.requestId, capabilities);
          return;
        }
        case "catalog": {
          const catalog = await withHostedDestructiveSharedLock(() =>
            this.catalogRepository.get(request.payload)
          );
          this.acceptedCatalog = catalog;
          this.success(request.requestId, catalog);
          return;
        }
        case "download":
          await this.handleDownload(
            request.requestId,
            request.payload.modelVariantId,
            request.payload.contextTokens,
            request.payload.explicitUnsafeOverride === true,
          );
          return;
        case "download_pack":
          await this.handleOptionalPackDownload(
            request.requestId,
            request.payload.modelVariantId,
            request.payload.packId,
          );
          return;
        case "load":
          await this.handleLoad(
            request.requestId,
            request.payload.modelVariantId,
            request.payload.contextTokens,
            request.payload.explicitUnsafeOverride === true,
            request.payload.resetSession === true,
          );
          return;
        case "request":
          await this.handleServiceRequest(
            request.requestId,
            request.payload,
            concurrentTreeMutation,
          );
          return;
        case "submit":
        case "generate":
          await this.handleGeneration(request.requestId, request.payload);
          return;
        case "cancel":
          await this.handleCancel(request.requestId);
          return;
        case "cancel_fitting":
          await this.handleFittingCancel(request.requestId);
          return;
        case "stop":
          if (this.admittedGenerationRequestId !== null) {
            this.cancelAdmittedGeneration = true;
          }
          await this.modelBackend.stop("user");
          this.success(request.requestId, undefined);
          return;
        case "unload":
        case "takeover":
          this.backendEpoch += 1;
          await this.settleDeviceLossTasks();
          await this.closeSessionCoordinator();
          await this.modelBackend.unload();
          this.backendLoaded = false;
          this.loadedSelection = null;
          this.activeGenerationId = null;
          this.generationDeviceLoss = null;
          this.deviceLossRetryKey = null;
          this.deviceLossRetryUsed = false;
          this.patchSnapshot({
            lifecycle: "unloaded",
            modelVariantId: null,
            contextTokens: null,
            error: null,
          }, request.requestId);
          this.success(request.requestId, undefined);
          return;
        case "delete":
          await this.settleDeviceLossTasks();
          if (
            request.payload.kind === "model" &&
            request.payload.id === this.snapshot.modelVariantId &&
            (this.snapshot.lifecycle === "loading" || this.snapshot.lifecycle === "ready")
          ) {
            throw runtimeFailure(
              "MODEL_IN_USE",
              "Unload the active model before deleting its local files",
              true,
              409,
            );
          }
          if (
            request.payload.kind === "pack" &&
            (this.snapshot.lifecycle === "loading" || this.snapshot.lifecycle === "ready")
          ) {
            throw runtimeFailure(
              "PACK_IN_USE",
              "Unload the active model before deleting an instrument pack",
              true,
              409,
            );
          }
          await this.ensureContentStore();
          await this.contentStore.removeInstall(request.payload.id, request.payload.kind);
          await this.contentStore.collectGarbage();
          await this.patchPersistedState(request.requestId);
          this.success(request.requestId, undefined);
          return;
        case "clear":
          this.backendEpoch += 1;
          await this.settleDeviceLossTasks();
          await this.closeSessionCoordinator();
          await this.modelBackend.unload();
          this.backendLoaded = false;
          this.loadedSelection = null;
          this.activeGenerationId = null;
          this.generationDeviceLoss = null;
          this.deviceLossRetryKey = null;
          this.deviceLossRetryUsed = false;
          this.capabilityChecker.clearAdapter();
          this.patchSnapshot({
            lifecycle: "unloaded",
            modelVariantId: null,
            contextTokens: null,
            error: null,
          }, request.requestId);
          await this.activationSpool.clear();
          await this.ensureContentStore();
          await this.sessionPersistence.clear();
          await this.contentStore.clear();
          await this.artifactService.clear();
          this.patchSnapshot({
            selectedModelVariantId: null,
            installedModelVariantIds: [],
            installedPackIds: [],
            loadRecords: [],
          }, request.requestId);
          await this.catalogRepository.clear();
          this.acceptedCatalog = null;
          await this.clearRuntimeStorage();
          this.success(request.requestId, undefined);
          return;
        case "persist":
          this.success(request.requestId, await requestPersistentStorage());
          return;
        case "storage":
          await this.ensureContentStore();
          await this.patchPersistedState(request.requestId);
          this.success(request.requestId, await storageEstimate());
          return;
      }
    } catch (error) {
      const failure = asFailure(error);
      if (request.command === "delete" || request.command === "clear") {
        await this.patchPersistedState(request.requestId).catch(() => undefined);
      }
      if (request.command === "check") {
        this.checkedCapabilities = null;
        this.patchSnapshot({ lifecycle: "failed", error: failure }, request.requestId);
      } else if (
        request.command === "load" &&
        failure.code !== "RUNTIME_CLEANUP_FAILED" &&
        this.activeDeviceLossTasks.size === 0 &&
        !(this.backendLoaded && this.snapshot.lifecycle === "ready")
      ) {
        this.patchSnapshot({ lifecycle: "failed", error: failure }, request.requestId);
      } else if (request.command === "download" || request.command === "download_pack") {
        this.patchOperation("download", "failed", failure, request.requestId);
        this.event("error", { scope: "download", failure }, request.requestId);
      }
      this.error(request.requestId, failure);
    }
  }

  private async handleDownload(
    requestId: string,
    modelVariantId: string,
    contextTokens: number,
    explicitUnsafeOverride: boolean,
  ): Promise<void> {
    if (this.cancelAdmittedDownload) {
      this.success(requestId, {
        modelVariantId,
        installed: false,
        cancelled: true,
      } satisfies DownloadWorkerResult);
      return;
    }
    const catalog = this.acceptedCatalog;
    if (!catalog) {
      throw unavailable(
        "VERIFIED_CATALOG_REQUIRED",
        "Load and verify the signed Drowse catalog before downloading a model",
      );
    }
    const capabilities = this.checkedCapabilities;
    if (!capabilities?.supported) {
      throw runtimeFailure(
        "COMPATIBILITY_CHECK_REQUIRED",
        "Run and pass the browser compatibility check before downloading a model",
        true,
        412,
      );
    }
    const expiresAt = Date.parse(catalog.document.expiresAt);
    if (!Number.isFinite(expiresAt) || Date.now() >= expiresAt) {
      throw runtimeFailure(
        "CATALOG_EXPIRED",
        "The accepted catalog has expired and cannot authorize a new download",
        true,
        409,
      );
    }
    const selection = selectDownloadManifest(catalog, modelVariantId);
    const model = catalog.document.models.find((candidate) =>
      candidate.variants.some((variant) => variant.id === modelVariantId)
    );
    if (!model) {
      throw runtimeFailure(
        "MODEL_VARIANT_NOT_FOUND",
        `Model variant ${modelVariantId} is not present in the verified catalog`,
        false,
        404,
      );
    }
    const assessment = assessModelVariant(model, selection.variant, {
      capabilities: capabilitiesWithoutStorageAdmission(capabilities),
      contextTokens,
      loadRecords: this.snapshot.loadRecords,
      installedModelVariantIds: this.snapshot.installedModelVariantIds,
      explicitOomRetry: explicitUnsafeOverride,
    });
    const hardFailure = assessment.hardFailures[0];
    if (hardFailure) {
      throw runtimeFailure(
        hardFailure.code,
        hardFailure.message,
        true,
        412,
      );
    }
    const unsafeAdvisory = assessment.advisories.find((issue) =>
      UNSAFE_DOWNLOAD_ADVISORIES.has(issue.code)
    );
    if (unsafeAdvisory && !explicitUnsafeOverride) {
      throw runtimeFailure(
        "UNSAFE_MODEL_OVERRIDE_REQUIRED",
        `${unsafeAdvisory.message}. Confirm the warning to try this model.`,
        true,
        409,
      );
    }
    const headroomBytes = Math.max(
      Math.ceil(selection.variant.downloadBytes * 0.2),
      256 * MEBIBYTE,
    );
    const settled = deferredVoid();
    this.activeDownloadSettled = settled.promise;
    this.resolveActiveDownload = settled.resolve;
    this.patchOperation("download", "running", null, requestId);
    let committed = false;
    try {
      const result = await this.downloader.download(catalog, modelVariantId, {
          retainFirstRangeProbe: shouldRetainFirstRangeProbe(),
          withCatalogAuthorization: (operation, signal) =>
            this.catalogRepository.withDownloadAuthorization(catalog, operation, signal),
          checkQuota: async ({ remainingManifestBytes }) => {
            if (remainingManifestBytes === 0) return true;
            const estimate = await this.storageEstimator();
            if (estimate.availableBytes === null) {
              throw new ArtifactDownloadError(
                "DOWNLOAD_QUOTA_CHECK_FAILED",
                "Browser storage availability is unknown, so Drowse cannot safely start this download",
                { status: 507, recoverable: true },
              );
            }
            return estimate.availableBytes >= remainingManifestBytes + headroomBytes;
          },
          onProgress: (progress) => this.event("progress", progress, requestId),
        });
      committed = true;
      await this.ensureContentStore();
      const [installs, loadRecords, storedSelection] = await Promise.all([
        this.contentStore.listInstalls(),
        this.contentStore.listLoadRecords(),
        this.contentStore.selectedModelVariantId(),
      ]);
      const installedModelVariantIds = modelInstallIds(installs);
      const selectedModelVariantId =
        storedSelection !== null && installedModelVariantIds.includes(storedSelection)
          ? storedSelection
          : null;
      const installed = installedModelVariantIds.includes(modelVariantId);
      this.patchSnapshot({
        selectedModelVariantId,
        installedModelVariantIds,
        installedPackIds: packInstallIds(installs),
        loadRecords,
        download: operationState(this.snapshot.download, "complete", null),
      }, requestId);
      this.success(requestId, {
        modelVariantId: result.modelVariantId,
        installed,
        cancelled: false,
      } satisfies DownloadWorkerResult);
    } catch (error) {
      if (error instanceof ArtifactDownloadError && error.code === "DOWNLOAD_CANCELLED") {
        this.patchOperation("download", "paused", null, requestId);
        this.success(requestId, {
          modelVariantId,
          installed: false,
          cancelled: true,
        } satisfies DownloadWorkerResult);
      } else {
        if (committed) {
          await this.patchPersistedState(requestId).catch(() => undefined);
        }
        const failure = asFailure(error);
        this.patchOperation("download", "failed", failure, requestId);
        this.event("error", { scope: "download", failure }, requestId);
        this.error(requestId, failure);
      }
    } finally {
      settled.resolve();
      if (this.resolveActiveDownload === settled.resolve) {
        this.activeDownloadSettled = null;
        this.resolveActiveDownload = null;
      }
    }
  }

  private async handleOptionalPackDownload(
    requestId: string,
    modelVariantId: string,
    packId: string,
  ): Promise<void> {
    if (this.cancelAdmittedDownload) {
      this.success(requestId, {
        modelVariantId,
        packId,
        installed: false,
        cancelled: true,
      } satisfies PackDownloadWorkerResult);
      return;
    }
    const catalog = this.acceptedCatalog;
    if (!catalog) {
      throw unavailable(
        "VERIFIED_CATALOG_REQUIRED",
        "Load and verify the signed Drowse catalog before downloading an instrument pack",
      );
    }
    if (!this.checkedCapabilities?.supported) {
      throw runtimeFailure(
        "COMPATIBILITY_CHECK_REQUIRED",
        "Run and pass the browser compatibility check before downloading an instrument pack",
        true,
        412,
      );
    }
    const expiresAt = Date.parse(catalog.document.expiresAt);
    if (!Number.isFinite(expiresAt) || Date.now() >= expiresAt) {
      throw runtimeFailure(
        "CATALOG_EXPIRED",
        "The accepted catalog has expired and cannot authorize a new download",
        true,
        409,
      );
    }
    const selection = selectOptionalPackManifest(catalog, modelVariantId, packId);
    const modelSelection = selectDownloadManifest(catalog, modelVariantId, {
      requireDownloadAuthorization: false,
    });
    await this.ensureContentStore();
    const assertPackPrerequisites = async (signal?: AbortSignal): Promise<void> => {
      signal?.throwIfAborted();
      const installs = await this.contentStore.listInstalls();
      signal?.throwIfAborted();
      const modelInstall = installs.find((install) =>
        install.id === modelVariantId && install.kind === "model"
      );
      if (!modelInstall) {
        throw runtimeFailure(
          "PACK_MODEL_NOT_INSTALLED",
          `Install model variant ${modelVariantId} before adding ${selection.pack.displayName}`,
          true,
          409,
        );
      }
      if (!sameObjectHashes(
        modelInstall.objectHashes,
        modelSelection.files.map((file) => file.sha256),
      )) {
        throw runtimeFailure(
          "PACK_MODEL_BINDING_MISMATCH",
          `Reinstall model variant ${modelVariantId} from the current verified catalog before adding ${selection.pack.displayName}`,
          true,
          409,
        );
      }
      if (
        this.snapshot.modelVariantId === modelVariantId &&
        this.snapshot.contextTokens !== null
      ) {
        const context = selection.variant.contextProfiles.find(
          (profile) => profile.contextTokens === this.snapshot.contextTokens,
        );
        if (
          !context ||
          !selection.pack.compatibleContextBindingSha256.includes(context.bindingSha256)
        ) {
          throw runtimeFailure(
            "PACK_CONTEXT_INCOMPATIBLE",
            `${selection.pack.displayName} is not compatible with the active ${this.snapshot.contextTokens}-token profile`,
            true,
            409,
          );
        }
      }
      const activeBindings = this.snapshot.modelVariantId === modelVariantId &&
          this.snapshot.contextTokens !== null
        ? selection.variant.contextProfiles
          .filter((profile) => profile.contextTokens === this.snapshot.contextTokens)
          .map((profile) => profile.bindingSha256)
        : selection.pack.compatibleContextBindingSha256;
      const installedPackIds = new Set(
        installs.filter((install) => install.kind === "pack").map((install) => install.id),
      );
      const concurrentPacks = selection.variant.packs.filter((candidate) =>
        candidate.id !== packId && !candidate.required && candidate.kind !== "core" &&
        installedPackIds.has(candidate.id) &&
        candidate.compatibleContextBindingSha256.some((binding) =>
          activeBindings.includes(binding)
        )
      );
      const hardwareBlock = optionalPackHardwareBlock(
        selection.pack,
        selection.variant,
        this.checkedCapabilities!,
        concurrentPacks,
      );
      if (hardwareBlock !== null) {
        throw runtimeFailure(
          hardwareBlock.code,
          hardwareBlock.message,
          true,
          412,
        );
      }
      const installed = installs.find((install) => install.id === packId);
      if (installed && installed.kind !== "pack") {
        throw runtimeFailure(
          "PACK_INSTALL_ID_CONFLICT",
          `Local install ${packId} is not an instrument pack`,
          false,
          409,
        );
      }
    };
    await assertPackPrerequisites();
    if (this.cancelAdmittedDownload) {
      this.success(requestId, {
        modelVariantId,
        packId,
        installed: false,
        cancelled: true,
      } satisfies PackDownloadWorkerResult);
      return;
    }

    const headroomBytes = Math.max(
      Math.ceil(selection.pack.bytes * 0.2),
      256 * MEBIBYTE,
    );
    const settled = deferredVoid();
    this.activeDownloadSettled = settled.promise;
    this.resolveActiveDownload = settled.resolve;
    this.patchOperation("download", "running", null, requestId);
    let committed = false;
    try {
      const result = await this.downloader.downloadPack(
          catalog,
          modelVariantId,
          packId,
          {
            retainFirstRangeProbe: shouldRetainFirstRangeProbe(),
            authorizeSelection: assertPackPrerequisites,
            withCatalogAuthorization: (operation, signal) =>
              this.catalogRepository.withDownloadAuthorization(catalog, operation, signal),
            checkQuota: async ({ remainingManifestBytes }) => {
              if (remainingManifestBytes === 0) return true;
              const estimate = await this.storageEstimator();
              if (estimate.availableBytes === null) {
                throw new ArtifactDownloadError(
                  "DOWNLOAD_QUOTA_CHECK_FAILED",
                  "Browser storage availability is unknown, so Drowse cannot safely start this download",
                  { status: 507, recoverable: true },
                );
              }
              return estimate.availableBytes >= remainingManifestBytes + headroomBytes;
            },
            onProgress: (progress) => this.event("progress", progress, requestId),
          },
      );
      committed = true;
      const currentInstalls = await this.contentStore.listInstalls();
      const installedPackIds = packInstallIds(currentInstalls);
      this.patchSnapshot({
        installedModelVariantIds: modelInstallIds(currentInstalls),
        installedPackIds,
        download: operationState(this.snapshot.download, "complete", null),
      }, requestId);
      this.success(requestId, {
        modelVariantId,
        packId: result.packId ?? packId,
        installed: installedPackIds.includes(result.packId ?? packId),
        cancelled: false,
      } satisfies PackDownloadWorkerResult);
    } catch (error) {
      if (error instanceof ArtifactDownloadError && error.code === "DOWNLOAD_CANCELLED") {
        this.patchOperation("download", "paused", null, requestId);
        this.success(requestId, {
          modelVariantId,
          packId,
          installed: false,
          cancelled: true,
        } satisfies PackDownloadWorkerResult);
      } else {
        if (committed) {
          await this.patchPersistedState(requestId).catch(() => undefined);
        }
        const failure = asFailure(error);
        this.patchOperation("download", "failed", failure, requestId);
        this.event("error", { scope: "download", failure }, requestId);
        this.error(requestId, failure);
      }
    } finally {
      settled.resolve();
      if (this.resolveActiveDownload === settled.resolve) {
        this.activeDownloadSettled = null;
        this.resolveActiveDownload = null;
      }
    }
  }

  private async handleLoad(
    requestId: string,
    modelVariantId: string,
    contextTokens: number,
    explicitUnsafeOverride: boolean,
    resetSession: boolean,
  ): Promise<void> {
    if (this.activeModelLoadSettled !== null) {
      throw runtimeFailure(
        "RUNTIME_IN_USE",
        "Another browser model load is still settling",
        true,
        409,
      );
    }
    const settled = deferredVoid();
    this.activeModelLoadSettled = settled.promise;
    try {
      await this.performLoad(
        requestId,
        modelVariantId,
        contextTokens,
        explicitUnsafeOverride,
        resetSession,
      );
    } finally {
      settled.resolve();
      if (this.activeModelLoadSettled === settled.promise) {
        this.activeModelLoadSettled = null;
      }
    }
  }

  private async performLoad(
    requestId: string,
    modelVariantId: string,
    contextTokens: number,
    explicitUnsafeOverride: boolean,
    resetSession: boolean,
  ): Promise<void> {
    if (this.snapshot.lifecycle === "loading" || this.activeDeviceLossTasks.size > 0) {
      throw runtimeFailure(
        "RUNTIME_IN_USE",
        "Wait for the current model load or device-loss recovery to finish before loading another model",
        true,
        409,
      );
    }
    const capabilities = this.checkedCapabilities;
    if (!capabilities?.supported) {
      throw runtimeFailure(
        "COMPATIBILITY_CHECK_REQUIRED",
        "Run and pass the browser compatibility check before loading a model",
        true,
        412,
      );
    }
    this.requireOperationAvailable("generation");
    const catalog = this.acceptedCatalog;
    if (!catalog) {
      throw runtimeFailure(
        "VERIFIED_CATALOG_REQUIRED",
        "Load and verify a catalog snapshot before opening an installed model",
        true,
        412,
      );
    }
    const selection = selectDownloadManifest(catalog, modelVariantId, {
      requireDownloadAuthorization: false,
    });
    const model = catalog.document.models.find((candidate) =>
      candidate.variants.some((variant) => variant.id === modelVariantId)
    );
    if (!model) {
      throw runtimeFailure(
        "MODEL_VARIANT_NOT_FOUND",
        `Model variant ${modelVariantId} is not present in the verified catalog`,
        false,
        404,
      );
    }
    await this.ensureContentStore();
    let installs = await this.contentStore.listInstalls();
    const installedModelIds = modelInstallIds(installs);
    const modelInstall = installs.find((install) =>
      install.id === modelVariantId && install.kind === "model"
    );
    if (!modelInstall) {
      throw runtimeFailure(
        "MODEL_NOT_INSTALLED",
        `Model variant ${modelVariantId} is not installed and verified`,
        true,
        409,
      );
    }
    if (!sameObjectHashes(
      modelInstall.objectHashes,
      selection.files.map((file) => file.sha256),
    )) {
      throw runtimeFailure(
        "MODEL_INSTALL_BINDING_MISMATCH",
        `Reinstall model variant ${modelVariantId} from the current verified catalog before loading it`,
        true,
        409,
      );
    }
    const assessment = assessModelVariant(model, selection.variant, {
      capabilities,
      contextTokens,
      loadRecords: this.snapshot.loadRecords,
      installedModelVariantIds: installedModelIds,
      explicitOomRetry: explicitUnsafeOverride,
    });
    const hardFailure = assessment.hardFailures[0];
    if (hardFailure) {
      throw runtimeFailure(
        hardFailure.code,
        hardFailure.message,
        true,
        412,
      );
    }
    const unsafeAdvisory = assessment.advisories.find((issue) =>
      UNSAFE_DOWNLOAD_ADVISORIES.has(issue.code)
    );
    if (unsafeAdvisory && !explicitUnsafeOverride) {
      throw runtimeFailure(
        "UNSAFE_MODEL_OVERRIDE_REQUIRED",
        `${unsafeAdvisory.message}. Confirm the warning to try this model.`,
        true,
        409,
      );
    }
    const artifacts = [];
    for (const manifest of selection.files) {
      const file = await this.contentStore.verifiedFile(manifest.sha256);
      if (!file || file.size !== manifest.bytes) {
        if (!file) {
          await this.patchPersistedState(requestId);
        }
        throw runtimeFailure(
          "VERIFIED_MODEL_OBJECT_MISSING",
          `Verified model object ${manifest.path} is missing from browser storage`,
          true,
          409,
        );
      }
      artifacts.push({ manifest, file });
    }
    const optionalPacks: BrowserInstrumentPackArtifacts[] = [];
    const selectedPackManifests: CatalogInstrumentPack[] = [];
    const installedPacks = new Map(
      installs
        .filter((install) => install.kind === "pack")
        .map((install) => [install.id, install]),
    );
    const contextBindingSha256 = assessment.context!.bindingSha256;
    let optionalPackStateChanged = false;
    for (const pack of selection.variant.packs) {
      if (
        pack.required || pack.kind === "core" ||
        !pack.compatibleContextBindingSha256.includes(contextBindingSha256)
      ) continue;
      const install = installedPacks.get(pack.id);
      if (
        !install ||
        !sameObjectHashes(install.objectHashes, pack.files.map((file) => file.sha256))
      ) {
        continue;
      }
      if (
        optionalPackHardwareBlock(
          pack,
          selection.variant,
          capabilities,
          selectedPackManifests,
        ) !== null
      ) continue;
      const packArtifacts: BrowserModelArtifact[] = [];
      for (const manifest of pack.files) {
        const file = await this.contentStore.verifiedFile(manifest.sha256);
        if (!file || file.size !== manifest.bytes) {
          await this.contentStore.removeInstall(pack.id, "pack");
          optionalPackStateChanged = true;
          break;
        }
        packArtifacts.push({ manifest, file });
      }
      if (packArtifacts.length === pack.files.length) {
        const activeKind = optionalPacks.find(({ pack: active }) => active.kind === pack.kind);
        if (activeKind && pack.kind !== "jlens") {
          throw runtimeFailure(
            "INSTRUMENT_PACK_AMBIGUOUS",
            `The verified catalog selects both ${activeKind.pack.displayName} and ${pack.displayName} for this ${contextTokens}-token profile. Install a catalog with only one ${pack.kind} pack for each context profile.`,
            true,
            409,
          );
        }
        optionalPacks.push({ pack, artifacts: packArtifacts });
        selectedPackManifests.push(pack);
      }
    }
    if (optionalPackStateChanged) {
      await this.contentStore.collectGarbage();
      await this.patchPersistedState(requestId);
      installs = await this.contentStore.listInstalls();
    }

    const loadedSelection: LoadedModelSelection = {
      model,
      variant: selection.variant,
      requiredCorePack: selection.requiredCorePack,
      contextTokens,
      artifacts,
      optionalPacks,
    };
    await this.contentStore.setSelectedModelVariantId(modelVariantId);
    this.patchSnapshot({
      lifecycle: "loading",
      modelVariantId,
      contextTokens,
      error: null,
    }, requestId);
    this.generationDeviceLoss = null;
    this.deviceLossRetryKey = deviceLossSelectionKey(loadedSelection);
    this.deviceLossRetryUsed = false;
    const backendEpoch = ++this.backendEpoch;
    let deviceLossReported = false;
    const assertLoadDevice = (): void => {
      if (deviceLossReported || backendEpoch !== this.backendEpoch) {
        throw runtimeFailure(
          "MODEL_DEVICE_LOST_DURING_LOAD",
          "The WebGPU device was lost while the model was loading",
          true,
          503,
        );
      }
    };
    try {
      if (this.backendLoaded) {
        await this.closeSessionCoordinator();
        await this.modelBackend.unload();
        this.backendLoaded = false;
        this.loadedSelection = null;
      }
      const load = await this.loadModelBackend({
        model,
        variant: selection.variant,
        requiredCorePack: selection.requiredCorePack,
        contextTokens,
        runtimeClass: this.checkedCapabilities?.signals.runtimeClass,
        adapter: this.capabilityChecker.adapterForLoad
          ? await this.capabilityChecker.adapterForLoad()
          : null,
        artifacts,
        optionalPacks,
        activationSpool: this.activationSpool,
        onProgress: (event) => this.emitBackendRequestEvent(event, requestId),
        onDeviceLost: (failure) => {
          if (backendEpoch !== this.backendEpoch || deviceLossReported) return;
          deviceLossReported = true;
          this.scheduleDeviceLost(
            backendEpoch,
            loadedSelection,
            failure,
            this.activeGenerationId,
          );
        },
      });
      assertLoadDevice();
      this.backendLoaded = true;
      await this.restoreSession(loadedSelection, resetSession);
      assertLoadDevice();
      await this.recordLoad(selection.variant.id, selection.variant.runtimeIdentitySha256, {
        result: "success",
        contextTokens,
        prefillTokensPerSecond: load.prefillTokensPerSecond,
        decodeTokensPerSecond: load.decodeTokensPerSecond,
      });
      assertLoadDevice();
      const loadRecords = await this.contentStore.listLoadRecords();
      assertLoadDevice();
      this.loadedSelection = loadedSelection;
      this.patchSnapshot({
        lifecycle: "ready",
        modelVariantId,
        contextTokens,
        installedModelVariantIds: installedModelIds,
        installedPackIds: packInstallIds(installs),
        loadRecords,
        error: null,
      }, requestId);
      this.success(requestId, undefined);
    } catch (error) {
      if (!deviceLossReported) {
        try {
          await this.closeSessionCoordinator();
        } catch {
          // Backend disposal must continue when local-session persistence fails.
        }
        let cleanupError: unknown = null;
        try {
          await this.modelBackend.unload();
        } catch (unloadError) {
          cleanupError = unloadError;
        }
        if (cleanupError !== null) {
          this.backendLoaded = true;
          throw runtimeFailure(
            "RUNTIME_CLEANUP_FAILED",
            `Model load failed (${failureMessage(error)}) and the browser backend could not be disposed (${failureMessage(cleanupError)})`,
            false,
            500,
          );
        }
        this.backendLoaded = false;
        this.loadedSelection = null;
        if (shouldRecordLoadFailure(error)) {
          try {
            const result = loadFailureResult(error);
            await this.recordLoad(selection.variant.id, selection.variant.runtimeIdentitySha256, {
              result,
              contextTokens,
              prefillTokensPerSecond: null,
              decodeTokensPerSecond: null,
            });
            this.patchSnapshot({
              loadRecords: await this.contentStore.listLoadRecords(),
            }, requestId);
          } catch {
            // Preserve the model failure when load-history persistence is unavailable.
          }
        }
      }
      throw error;
    }
  }

  private async handleServiceRequest(
    requestId: string,
    request: RuntimeServiceRequest,
    concurrentTreeMutation = false,
  ): Promise<void> {
    const route = runtimeServiceRoute(request);
    if (route === null) {
      throw runtimeFailure(
        "INVALID_RUNTIME_REQUEST",
        `Unknown browser runtime method ${request.service}.${request.method}`,
        false,
        400,
      );
    }
    const policy = runtimeServiceOperationPolicy(request);
    if (route === "hybrid_manifold_read") {
      if (!this.artifactService.handles(request)) {
        throw runtimeFailure(
          "ARTIFACT_ROUTE_UNAVAILABLE",
          `The browser artifact repository cannot handle ${request.service}.${request.method}`,
          false,
          500,
        );
      }
      const progress = (event: RuntimeProgressEvent) => {
        this.event("progress", event.data as WorkerEventEnvelope["payload"], requestId);
      };
      if (request.method === "get" && this.backendLoaded && this.snapshot.lifecycle === "ready") {
        try {
          this.success(requestId, await this.requestModelBackend(request, progress));
          return;
        } catch (error) {
          if (!isRecord(error) || error.code !== "CORE_PACK_SELECTOR_NOT_FOUND") throw error;
        }
      }
      const local = await withHostedDestructiveSharedLock(() =>
        this.artifactService.request(request, progress)
      );
      if (request.method === "list" && this.backendLoaded && this.snapshot.lifecycle === "ready") {
        const runtime = await this.requestModelBackend(request, progress);
        const localRows = isRecord(local) && Array.isArray(local.manifolds) ? local.manifolds : [];
        const runtimeRows = isRecord(runtime) && Array.isArray(runtime.manifolds)
          ? runtime.manifolds
          : [];
        const merged = new Map<string, unknown>();
        for (const value of [...localRows, ...runtimeRows]) {
          if (!isRecord(value) || typeof value.namespace !== "string" || typeof value.name !== "string") {
            throw runtimeFailure(
              "MANIFOLD_RESPONSE_INVALID",
              "The browser runtime returned an invalid manifold summary",
              false,
              500,
            );
          }
          merged.set(`${value.namespace}/${value.name}`, value);
        }
        this.success(requestId, {
          manifolds: [...merged.values()].sort((left, right) => {
            const a = left as { namespace: string; name: string };
            const b = right as { namespace: string; name: string };
            return `${a.namespace}/${a.name}`.localeCompare(`${b.namespace}/${b.name}`);
          }),
        });
        return;
      }
      this.success(requestId, local);
      return;
    }
    if (route === "artifact") {
      this.requireRequestCapabilities(policy.requiredCapabilities);
      if (!this.artifactService.handles(request)) {
        throw runtimeFailure(
          "ARTIFACT_ROUTE_UNAVAILABLE",
          `The browser artifact repository cannot handle ${request.service}.${request.method}`,
          false,
          500,
        );
      }
      let mutationCommitted = false;
      let committedResult: unknown;
      let result: unknown;
      try {
        result = await withHostedDestructiveSharedLock(async () => {
          await this.assertManifoldDeletionSafe(request);
          const artifactResult = await this.artifactService.request(request, (progress) => {
            this.event(
              "progress",
              progress.data as WorkerEventEnvelope["payload"],
              requestId,
            );
          });
          if (
            refreshesLoadedManifolds(request) && this.backendLoaded &&
            this.snapshot.lifecycle === "ready" &&
            manifoldMutationChanged(request, artifactResult)
          ) {
            mutationCommitted = true;
            committedResult = artifactResult;
            await this.refreshModelBackendManifolds();
          }
          return artifactResult;
        });
      } catch (error) {
        if (mutationCommitted) {
          throw await this.invalidateBackendAfterArtifactRefreshFailure(
            error,
            requestId,
            request,
            committedResult,
          );
        }
        throw error;
      }
      this.success(requestId, result);
      return;
    }
    if (isHostedInstrumentDownloadRequest(request)) {
      throw runtimeFailure(
        "HOSTED_PACK_DOWNLOAD_MANAGED",
        "Browser tool downloads are managed in Model settings. Activate an already loaded compatible pack instead",
        true,
        409,
      );
    }
    if (route === "fitting") {
      await this.handleFittingRequest(requestId, request);
      return;
    }
    this.requireReady();
    this.requireOperationAvailable("generation");
    this.requireRequestCapabilities(policy.requiredCapabilities);
    const result = await this.requestModelBackend(
      request,
      (progress) => {
        this.emitBackendRequestEvent(progress, requestId);
      },
      concurrentTreeMutation ||
          (request.service === "instruments" && request.method === "tokenReadout")
        ? { waitForActiveRequest: true, requireReadyAfterWait: true }
        : undefined,
    );
    if (mutatesPersistedHostedSession(request)) {
      await this.persistSession();
    }
    this.success(requestId, result);
  }

  private emitBackendRequestEvent(
    event: RuntimeProgressEvent,
    requestId: string,
  ): void {
    if (event.event === "tree_mutated") {
      if (!isTreeMutationMessage(event.data)) {
        throw runtimeFailure(
          "TREE_MUTATION_EVENT_INVALID",
          "The browser model backend emitted an invalid conversation mutation",
          false,
          500,
        );
      }
      this.event("tree_mutated", event.data, requestId, null);
      return;
    }
    this.event(
      "progress",
      event.data as WorkerEventEnvelope["payload"],
      requestId,
    );
  }

  private async handleFittingRequest(
    requestId: string,
    request: RuntimeServiceRequest,
  ): Promise<void> {
    if (this.cancelAdmittedFitting) {
      this.patchOperation("fitting", "complete", null, requestId);
      this.error(requestId, fittingCancelledFailure());
      return;
    }

    const controller = new AbortController();
    const settled = deferredVoid();
    this.activeFittingController = controller;
    this.activeFittingSettled = settled.promise;
    try {
      this.requireReady();
      this.requireOperationAvailable("generation");
      this.requireRequestCapabilities(
        runtimeServiceOperationPolicy(request).requiredCapabilities,
      );
      this.patchOperation("fitting", "running", null, requestId);
      const result = await this.modelBackend.request(request, (progress) => {
        this.event(
          "progress",
          progress.data as WorkerEventEnvelope["payload"],
          requestId,
        );
      }, controller.signal);
      if (controller.signal.aborted) throw fittingCancelledFailure();
      this.patchOperation("fitting", "complete", null, requestId);
      this.success(requestId, result);
    } catch (error) {
      if (controller.signal.aborted || isFittingCancellation(error)) {
        this.patchOperation("fitting", "complete", null, requestId);
        this.error(requestId, fittingCancelledFailure());
      } else {
        const failure = asFailure(error);
        this.patchOperation("fitting", "failed", failure, requestId);
        this.error(requestId, failure);
      }
    } finally {
      settled.resolve();
      if (this.activeFittingController === controller) {
        this.activeFittingController = null;
        this.activeFittingSettled = null;
      }
    }
  }

  private async handleGeneration(
    requestId: string,
    request: WSSubmitRequest | WSGenerateRequest,
  ): Promise<void> {
    const mutationAdmission = this.generationMutationAdmission;
    const settled = deferredVoid();
    this.activeGenerationSettled = settled.promise;
    try {
      if (this.cancelAdmittedGeneration) {
        this.success(requestId, undefined);
        return;
      }
      this.requireReady();
      this.requireOperationAvailable("generation");
      this.patchOperation("generation", "running", null, requestId);
      const protocol = newGenerationProtocol(
        request.type === "generate" && request.stateless === true,
      );
      const selection = this.loadedSelection;
      const generationBackendEpoch = this.backendEpoch;
      let queuedFailure: unknown = null;
      let eventQueue = Promise.resolve();
      const performance = await this.modelBackend.generate(request, (message) => {
        const queuedEvent = eventQueue.then(async () => {
          if (queuedFailure !== null) return;
          try {
            await this.emitModelMessage(message, requestId, protocol);
          } catch (error) {
            queuedFailure = error;
            void this.modelBackend.stop().catch(() => undefined);
          }
        });
        eventQueue = queuedEvent;
        if (message.type === "tree_mutated") {
          this.generationTreeEventBarrier = queuedEvent.catch(() => undefined);
        }
        return eventQueue;
      }, () => mutationAdmission?.resolve());
      await eventQueue;
      if (queuedFailure !== null) throw queuedFailure;
      assertGenerationProtocolComplete(protocol);
      let loadRecords: DeviceLoadRecord[] | null = null;
      try {
        loadRecords = await this.persistGenerationPerformance(
          performance,
          selection,
          generationBackendEpoch,
        );
      } catch {
        // Performance history must not turn a completed generation into a failure.
      }
      this.patchSnapshot({
        generation: operationState(this.snapshot.generation, "complete", null),
        ...(loadRecords === null ? {} : { loadRecords }),
      }, requestId);
      this.success(requestId, undefined);
    } catch (error) {
      const deviceLoss = this.generationDeviceLoss;
      const failure = deviceLoss?.backendEpoch === this.backendEpoch
        ? deviceLoss.failure
        : asFailure(error);
      this.patchOperation("generation", "failed", failure, requestId);
      this.activeGenerationId = null;
      this.error(requestId, failure);
    } finally {
      mutationAdmission?.resolve();
      if (this.generationMutationAdmission === mutationAdmission) {
        this.generationMutationAdmission = null;
      }
      settled.resolve();
      if (this.activeGenerationSettled === settled.promise) {
        this.activeGenerationSettled = null;
      }
    }
  }

  private async emitModelMessage(
    message: WSServerMessage,
    requestId: string,
    protocol: GenerationProtocolState,
  ): Promise<void> {
    if (message.type === "error") {
      protocol.terminalError = true;
      throw runtimeFailure(
        message.code ?? "GENERATION_FAILED",
        message.message,
        true,
        500,
      );
    }
    if (protocol.terminalError) {
      throw generationProtocolViolation("The backend emitted an event after a terminal error");
    }
    if (message.type === "started") {
      if (
        typeof message.generation_id !== "string" ||
        message.generation_id.length === 0 || message.generation_id.length > 128
      ) {
        throw generationProtocolViolation("The started event has an invalid generation id");
      }
      if (!validGenerationNodeId(message.node_id)) {
        throw generationProtocolViolation("The started event has an invalid node id");
      }
      if (protocol.phase !== "awaiting_started") {
        throw generationProtocolViolation(
          "A generation started before the previous sibling reached done",
        );
      }
      if (protocol.seenGenerationIds.has(message.generation_id)) {
        throw generationProtocolViolation(
          `Generation id ${message.generation_id} was started more than once`,
        );
      }
      if (
        !Number.isSafeInteger(message.sibling_index) ||
        !Number.isSafeInteger(message.sibling_count) ||
        message.sibling_count < 1 ||
        message.sibling_index !== protocol.completedGenerations ||
        message.sibling_index >= message.sibling_count
      ) {
        throw generationProtocolViolation(
          "The started event has an invalid or out-of-order sibling position",
        );
      }
      if (
        protocol.expectedSiblingCount !== null &&
        message.sibling_count !== protocol.expectedSiblingCount
      ) {
        throw generationProtocolViolation(
          "The backend changed sibling_count during one generation request",
        );
      }
      protocol.phase = "streaming";
      protocol.sawTreeMutation = false;
      protocol.sawStableTreeMutation = false;
      protocol.activeGenerationId = message.generation_id;
      protocol.activeNodeId = message.node_id;
      protocol.siblingIndex = message.sibling_index;
      protocol.siblingCount = message.sibling_count;
      protocol.expectedSiblingCount ??= message.sibling_count;
      protocol.seenGenerationIds.add(message.generation_id);
      this.activeGenerationId = message.generation_id;
      this.event("started", message, requestId, message.generation_id);
      return;
    }
    if (message.type === "tree_mutated") {
      if (protocol.phase !== "streaming" || protocol.activeGenerationId === null) {
        throw generationProtocolViolation(
          "A tree mutation arrived without an active started generation",
        );
      }
      if (protocol.stateless) {
        throw generationProtocolViolation(
          "A stateless generation emitted an unexpected tree mutation",
        );
      }
      protocol.sawTreeMutation = true;
      if (message.op !== "begin_assistant") {
        protocol.sawStableTreeMutation = true;
        await this.persistSession();
      }
      this.event("tree_mutated", message, requestId, protocol.activeGenerationId);
      return;
    }
    if (protocol.phase !== "streaming" || protocol.activeGenerationId === null) {
      throw generationProtocolViolation(
        `${message.type} arrived without an active started generation`,
      );
    }
    if (!validGenerationNodeId(message.node_id)) {
      throw generationProtocolViolation(`${message.type} has an invalid node id`);
    }
    if (protocol.activeNodeId === null && message.node_id !== null) {
      protocol.activeNodeId = message.node_id;
    } else if (
      protocol.activeNodeId !== null && message.node_id !== protocol.activeNodeId
    ) {
      throw generationProtocolViolation(
        `${message.type} node ${message.node_id} does not match started node ${protocol.activeNodeId}`,
      );
    }
    if (message.type === "token") {
      this.event("token", message, requestId, protocol.activeGenerationId);
      return;
    }
    if (message.type === "generation_progress") {
      if (!Number.isSafeInteger(message.completed) || !Number.isSafeInteger(message.total) ||
        message.total < 1 || message.completed < 0 || message.completed > message.total) {
        throw generationProtocolViolation("The continuation progress is invalid");
      }
      this.event("generation_progress", message, requestId, protocol.activeGenerationId);
      return;
    }
    if (!protocol.stateless && !protocol.sawTreeMutation) {
      throw generationProtocolViolation(
        "A stateful generation reached done without an authoritative tree mutation",
      );
    }
    if (!protocol.stateless && !protocol.sawStableTreeMutation) {
      throw generationProtocolViolation(
        "A stateful generation reached done without a stable authoritative tree mutation",
      );
    }
    if (
      message.sibling_index !== protocol.siblingIndex ||
      message.sibling_count !== protocol.siblingCount
    ) {
      throw generationProtocolViolation(
        "The done event does not match the started sibling position",
      );
    }
    this.event("done", message, requestId, protocol.activeGenerationId);
    protocol.completedGenerations += 1;
    protocol.phase = "awaiting_started";
    protocol.activeGenerationId = null;
    protocol.activeNodeId = null;
    protocol.siblingIndex = null;
    protocol.siblingCount = null;
    protocol.sawTreeMutation = false;
    protocol.sawStableTreeMutation = false;
    this.activeGenerationId = null;
  }

  private async handleDeviceLost(
    backendEpoch: number,
    selection: LoadedModelSelection,
    failure: { code: string; message: string; confirmedOom: boolean },
    generationId: string | null,
  ): Promise<void> {
    if (backendEpoch !== this.backendEpoch) return;
    const runtime = runtimeFailure(failure.code, failure.message, true, 503);
    const modelLoadSettled = this.activeModelLoadSettled;
    const selectionKey = deviceLossSelectionKey(selection);
    const retry = !failure.confirmedOom && this.backendLoaded &&
      this.snapshot.lifecycle === "ready" &&
      this.deviceLossRetryKey === selectionKey && !this.deviceLossRetryUsed;
    if (retry) this.deviceLossRetryUsed = true;
    if (this.activeGenerationSettled !== null) {
      this.generationDeviceLoss = { backendEpoch, failure: runtime };
    }
    this.patchSnapshot({ lifecycle: "loading", error: null }, null);

    const backendLoadSettled = this.activeBackendLoadSettled;
    this.activeBackendLoadController?.abort();
    const backendRequestSettled = this.activeBackendRequestSettled;
    this.activeBackendRequestController?.abort();
    const fittingSettled = this.beginFittingCancellation(null);
    const generationSettled = this.activeGenerationSettled;
    if (generationSettled !== null) {
      try {
        await this.modelBackend.stop();
      } catch {
        // The failed device may reject stop while the generation still unwinds.
      }
    }
    if (fittingSettled !== null) await fittingSettled;
    if (generationSettled !== null) await generationSettled;
    if (backendRequestSettled !== null) await backendRequestSettled;
    if (backendLoadSettled !== null) await backendLoadSettled;
    if (modelLoadSettled !== null) await modelLoadSettled;
    if (backendEpoch !== this.backendEpoch) return;
    try {
      await this.recordLoad(selection.variant.id, selection.variant.runtimeIdentitySha256, {
        result: failure.confirmedOom ? "oom" : "device_lost",
        contextTokens: selection.contextTokens,
        prefillTokensPerSecond: null,
        decodeTokensPerSecond: null,
      });
      this.patchSnapshot({
        loadRecords: await this.contentStore.listLoadRecords(),
      }, null);
    } catch {
      // Device-loss recovery must continue even when persistence is unavailable.
    }
    try {
      await this.closeSessionCoordinator();
    } catch {
      // The runtime still has to release the failed device.
    }
    try {
      await this.modelBackend.unload();
    } catch {
      // The device is already unusable; local state still needs to be invalidated.
    }
    this.backendLoaded = false;
    this.loadedSelection = null;
    if (backendEpoch !== this.backendEpoch) return;
    if (this.generationDeviceLoss?.backendEpoch === backendEpoch) {
      this.generationDeviceLoss = null;
    }
    this.activeGenerationId = null;

    let terminalFailure = runtime;
    if (retry) {
      try {
        if (await this.attemptDeviceLossRecovery(selection, backendEpoch)) return;
        return;
      } catch (error) {
        terminalFailure = asFailure(error);
      }
    } else {
      this.capabilityChecker.clearAdapter();
    }

    if (terminalFailure.code === "RUNTIME_CLEANUP_FAILED") {
      this.event("error", {
        scope: "lifecycle",
        failure: terminalFailure,
      }, null, generationId);
      return;
    }
    this.patchSnapshot({ lifecycle: "failed", error: terminalFailure }, null);
    this.event("device_lost", runtime, null, generationId);
    this.event("error", {
      scope: "lifecycle",
      failure: terminalFailure,
    }, null, generationId);
  }

  private scheduleDeviceLost(
    backendEpoch: number,
    selection: LoadedModelSelection,
    failure: { code: string; message: string; confirmedOom: boolean },
    generationId: string | null,
  ): void {
    const task = this.handleDeviceLost(backendEpoch, selection, failure, generationId);
    this.activeDeviceLossTasks.add(task);
    void task.then(
      () => this.activeDeviceLossTasks.delete(task),
      () => this.activeDeviceLossTasks.delete(task),
    );
  }

  private async settleDeviceLossTasks(): Promise<void> {
    while (this.activeDeviceLossTasks.size > 0) {
      await Promise.allSettled([...this.activeDeviceLossTasks]);
    }
  }

  private async attemptDeviceLossRecovery(
    selection: LoadedModelSelection,
    lostBackendEpoch: number,
  ): Promise<boolean> {
    return withHostedDestructiveSharedLock(() =>
      this.attemptDeviceLossRecoveryLocked(selection, lostBackendEpoch)
    );
  }

  private async attemptDeviceLossRecoveryLocked(
    selection: LoadedModelSelection,
    lostBackendEpoch: number,
  ): Promise<boolean> {
    this.capabilityChecker.clearAdapter();
    let capabilities: RuntimeCapabilities;
    try {
      capabilities = await this.capabilityChecker.check();
    } catch (error) {
      if (lostBackendEpoch !== this.backendEpoch) return false;
      throw error;
    }
    if (lostBackendEpoch !== this.backendEpoch) return false;
    this.checkedCapabilities = capabilities;
    const capabilityFailure = capabilities.issues.find((issue) => issue.severity === "hard");
    if (!capabilities.supported) {
      throw runtimeFailure(
        capabilityFailure?.code ?? "WEBGPU_RECOVERY_UNAVAILABLE",
        capabilityFailure?.message ??
          "The browser could not create a replacement WebGPU device",
        true,
        503,
      );
    }
    const assessment = assessModelVariant(selection.model, selection.variant, {
      capabilities,
      contextTokens: selection.contextTokens,
      loadRecords: this.snapshot.loadRecords,
      installedModelVariantIds: this.snapshot.installedModelVariantIds,
      explicitOomRetry: false,
    });
    const hardFailure = assessment.hardFailures[0];
    if (hardFailure) {
      throw runtimeFailure(hardFailure.code, hardFailure.message, true, 412);
    }
    const reauthorizedSelection = await this.reauthorizeLoadedSelection(
      selection,
      capabilities,
    );
    if (lostBackendEpoch !== this.backendEpoch) return false;

    const recoveryEpoch = ++this.backendEpoch;
    let recoveryDeviceLossReported = false;
    const recoverySessionIdentity = sessionIdentityFor(reauthorizedSelection);
    try {
      const load = await this.loadModelBackend({
        model: reauthorizedSelection.model,
        variant: reauthorizedSelection.variant,
        requiredCorePack: reauthorizedSelection.requiredCorePack,
        contextTokens: reauthorizedSelection.contextTokens,
        runtimeClass: capabilities.signals.runtimeClass,
        adapter: this.capabilityChecker.adapterForLoad
          ? await this.capabilityChecker.adapterForLoad()
          : null,
        artifacts: reauthorizedSelection.artifacts,
        optionalPacks: reauthorizedSelection.optionalPacks,
        activationSpool: this.activationSpool,
        onDeviceLost: (recoveryFailure) => {
          if (recoveryEpoch !== this.backendEpoch || recoveryDeviceLossReported) return;
          recoveryDeviceLossReported = true;
          this.scheduleDeviceLost(
            recoveryEpoch,
            reauthorizedSelection,
            recoveryFailure,
            this.activeGenerationId,
          );
        },
      });
      if (recoveryDeviceLossReported || recoveryEpoch !== this.backendEpoch) {
        try {
          await this.modelBackend.unload();
        } catch {}
        this.backendLoaded = false;
        return false;
      }
      this.backendLoaded = true;
      await this.restoreSession(reauthorizedSelection, false, recoverySessionIdentity);
      if (recoveryDeviceLossReported || recoveryEpoch !== this.backendEpoch) {
        await this.closeSessionCoordinator(recoverySessionIdentity);
        return false;
      }
      await this.recordLoad(
        reauthorizedSelection.variant.id,
        reauthorizedSelection.variant.runtimeIdentitySha256,
        {
        result: "success",
        contextTokens: reauthorizedSelection.contextTokens,
        prefillTokensPerSecond: load.prefillTokensPerSecond,
        decodeTokensPerSecond: load.decodeTokensPerSecond,
        },
      );
      const loadRecords = await this.contentStore.listLoadRecords();
      if (recoveryDeviceLossReported || recoveryEpoch !== this.backendEpoch) return false;
      this.loadedSelection = reauthorizedSelection;
      this.patchSnapshot({
        lifecycle: "ready",
        modelVariantId: reauthorizedSelection.variant.id,
        contextTokens: reauthorizedSelection.contextTokens,
        loadRecords,
        error: null,
      }, null);
      return true;
    } catch (error) {
      const recoveryWasSuperseded = recoveryDeviceLossReported ||
        recoveryEpoch !== this.backendEpoch;
      let sessionCleanupError: unknown = null;
      try {
        await this.closeSessionCoordinator(recoverySessionIdentity);
      } catch (closeError) {
        sessionCleanupError = closeError;
      }
      let backendCleanupError: unknown = null;
      try {
        await this.modelBackend.unload();
      } catch (unloadError) {
        backendCleanupError = unloadError;
      }
      if (sessionCleanupError !== null || backendCleanupError !== null) {
        this.backendLoaded = backendCleanupError !== null;
        const cleanupFailures = [
          sessionCleanupError === null
            ? null
            : `session cleanup failed (${failureMessage(sessionCleanupError)})`,
          backendCleanupError === null
            ? null
            : `backend disposal failed (${failureMessage(backendCleanupError)})`,
        ].filter((value): value is string => value !== null).join("; ");
        throw runtimeFailure(
          "RUNTIME_CLEANUP_FAILED",
          `Device-loss recovery failed (${failureMessage(error)}) and cleanup was not confirmed: ${cleanupFailures}`,
          false,
          500,
        );
      }
      this.backendLoaded = false;
      this.loadedSelection = null;
      if (recoveryWasSuperseded) return false;
      if (shouldRecordLoadFailure(error)) {
        await this.recordLoad(selection.variant.id, selection.variant.runtimeIdentitySha256, {
          result: loadFailureResult(error),
          contextTokens: selection.contextTokens,
          prefillTokensPerSecond: null,
          decodeTokensPerSecond: null,
        });
        this.patchSnapshot({
          loadRecords: await this.contentStore.listLoadRecords(),
        }, null);
      }
      throw error;
    }
  }

  private async reauthorizeLoadedSelection(
    previous: LoadedModelSelection,
    capabilities: RuntimeCapabilities,
  ): Promise<LoadedModelSelection> {
    const catalog = this.acceptedCatalog;
    if (!catalog) {
      throw runtimeFailure(
        "VERIFIED_CATALOG_REQUIRED",
        "Reload the verified catalog before recovering the installed model",
        true,
        412,
      );
    }
    const current = selectDownloadManifest(catalog, previous.variant.id, {
      requireDownloadAuthorization: false,
    });
    const model = catalog.document.models.find((candidate) =>
      candidate.id === previous.model.id &&
      candidate.variants.some((variant) => variant.id === previous.variant.id)
    );
    const previousContext = previous.variant.contextProfiles.find(
      (profile) => profile.contextTokens === previous.contextTokens,
    );
    const currentContext = current.variant.contextProfiles.find(
      (profile) => profile.contextTokens === previous.contextTokens,
    );
    if (
      !model || current.variant.runtimeIdentitySha256 !==
        previous.variant.runtimeIdentitySha256 ||
      !previousContext || !currentContext ||
      currentContext.bindingSha256 !== previousContext.bindingSha256 ||
      current.requiredCorePack.id !== previous.requiredCorePack.id ||
      !sameObjectHashes(
        current.files.map((file) => file.sha256),
        previous.artifacts.map(({ manifest }) => manifest.sha256),
      )
    ) {
      throw runtimeFailure(
        "MODEL_RECOVERY_BINDING_CHANGED",
        `The verified catalog binding for ${previous.variant.id} changed; reopen it explicitly after reinstalling if needed`,
        true,
        409,
      );
    }
    await this.ensureContentStore();
    const installs = await this.contentStore.listInstalls();
    const modelInstall = installs.find((install) =>
      install.kind === "model" && install.id === previous.variant.id
    );
    if (!modelInstall) {
      throw runtimeFailure(
        "MODEL_NOT_INSTALLED",
        `Model variant ${previous.variant.id} is no longer installed`,
        true,
        409,
      );
    }
    if (!sameObjectHashes(
      modelInstall.objectHashes,
      current.files.map((file) => file.sha256),
    )) {
      throw runtimeFailure(
        "MODEL_INSTALL_BINDING_MISMATCH",
        `Reinstall model variant ${previous.variant.id} from the current verified catalog before recovering it`,
        true,
        409,
      );
    }
    const artifacts = await this.reopenVerifiedArtifacts(
      current.files,
      "VERIFIED_MODEL_OBJECT_MISSING",
      "model",
    );
    const optionalPacks: BrowserInstrumentPackArtifacts[] = [];
    const selectedPackManifests: CatalogInstrumentPack[] = [];
    for (const previousPack of previous.optionalPacks) {
      const pack = current.variant.packs.find((candidate) =>
        candidate.id === previousPack.pack.id
      );
      const previousHashes = previousPack.artifacts.map(({ manifest }) => manifest.sha256);
      if (
        !pack || pack.kind !== previousPack.pack.kind || pack.required ||
        pack.kind === "core" ||
        pack.runtimeIdentitySha256 !== current.variant.runtimeIdentitySha256 ||
        !pack.compatibleContextBindingSha256.includes(currentContext.bindingSha256) ||
        !sameObjectHashes(pack.files.map((file) => file.sha256), previousHashes)
      ) {
        throw runtimeFailure(
          "INSTRUMENT_PACK_RECOVERY_BINDING_CHANGED",
          `The verified binding for instrument pack ${previousPack.pack.id} changed; reopen the model explicitly after reinstalling the pack if needed`,
          true,
          409,
        );
      }
      const install = installs.find((candidate) =>
        candidate.kind === "pack" && candidate.id === pack.id
      );
      if (!install) {
        throw runtimeFailure(
          "INSTRUMENT_PACK_NOT_INSTALLED",
          `Instrument pack ${pack.displayName} is no longer installed`,
          true,
          409,
        );
      }
      if (!sameObjectHashes(install.objectHashes, pack.files.map((file) => file.sha256))) {
        throw runtimeFailure(
          "INSTRUMENT_PACK_BINDING_MISMATCH",
          `Reinstall ${pack.displayName} from the current verified catalog before recovering it`,
          true,
          409,
        );
      }
      const hardwareBlock = optionalPackHardwareBlock(
        pack,
        current.variant,
        capabilities,
        selectedPackManifests,
      );
      if (hardwareBlock !== null) {
        throw runtimeFailure(hardwareBlock.code, hardwareBlock.message, true, 412);
      }
      optionalPacks.push({
        pack,
        artifacts: await this.reopenVerifiedArtifacts(
          pack.files,
          "VERIFIED_PACK_OBJECT_MISSING",
          `instrument pack ${pack.displayName}`,
        ),
      });
      selectedPackManifests.push(pack);
    }
    if (!sameVerifiedCatalogSnapshot(this.acceptedCatalog, catalog)) {
      throw runtimeFailure(
        "CATALOG_CHANGED_DURING_RECOVERY",
        "The verified catalog changed while the model artifacts were being reopened; try again",
        true,
        409,
      );
    }
    return {
      model,
      variant: current.variant,
      requiredCorePack: current.requiredCorePack,
      contextTokens: previous.contextTokens,
      artifacts,
      optionalPacks,
    };
  }

  private async reopenVerifiedArtifacts(
    manifests: readonly BrowserModelArtifact["manifest"][],
    missingCode: string,
    label: string,
  ): Promise<BrowserModelArtifact[]> {
    const artifacts: BrowserModelArtifact[] = [];
    for (const manifest of manifests) {
      const file = await this.contentStore.verifiedFile(manifest.sha256);
      if (!file || file.size !== manifest.bytes) {
        await this.patchPersistedState(null).catch(() => undefined);
        throw runtimeFailure(
          missingCode,
          `Verified ${label} object ${manifest.path} is missing from browser storage`,
          true,
          409,
        );
      }
      artifacts.push({ manifest, file });
    }
    return artifacts;
  }

  private async restoreSession(
    selection: LoadedModelSelection,
    resetSession: boolean,
    sessionIdentity = sessionIdentityFor(selection),
  ): Promise<void> {
    const sessionCoordinator = this.sessionCoordinatorFactory(
      this.sessionPersistence,
      (serviceRequest) => this.requestModelBackend(
        serviceRequest,
        () => undefined,
        { waitForActiveRequest: true },
      ),
      {
        maxOutputTokens: outputTokenLimitForSignals(
          this.checkedCapabilities?.signals,
          selection.contextTokens,
        ),
      },
    );
    this.sessionCoordinator = sessionCoordinator;
    this.sessionIdentity = sessionIdentity;
    const restore = await sessionCoordinator.restore(
      sessionIdentity,
      resetSession ? "reset" : "replace",
      compatibleSessionRuntimeIdentitySha256s(selection),
    );
    this.snapshot = { ...this.snapshot, modelDefaults: sessionCoordinator.modelDefaults };
    if (restore.status === "incompatible") {
      throw runtimeFailure(
        "LOCAL_SESSION_INCOMPATIBLE",
        `The saved local conversation does not match this ${restore.reason.replaceAll("_", " ")} binding`,
        true,
        409,
      );
    }
    if (restore.status === "unsupported") {
      throw runtimeFailure(
        "LOCAL_SESSION_SCHEMA_UNSUPPORTED",
        "The saved local conversation was written by an unsupported Drowse version and was preserved",
        true,
        409,
      );
    }
  }

  private async loadModelBackend(
    request: Omit<BrowserModelLoadRequest, "signal">,
  ): Promise<BrowserModelLoadResult> {
    if (this.activeBackendLoadController !== null) {
      throw runtimeFailure(
        "RUNTIME_IN_USE",
        "Another browser model load is still settling",
        true,
        409,
      );
    }
    const controller = new AbortController();
    const settled = deferredVoid();
    this.activeBackendLoadController = controller;
    this.activeBackendLoadSettled = settled.promise;
    try {
      return await this.modelBackend.load({ ...request, signal: controller.signal });
    } finally {
      settled.resolve();
      if (this.activeBackendLoadController === controller) {
        this.activeBackendLoadController = null;
        this.activeBackendLoadSettled = null;
      }
    }
  }

  private async requestModelBackend(
    request: RuntimeServiceRequest,
    onProgress: (event: RuntimeProgressEvent) => void,
    options: {
      waitForActiveRequest?: boolean;
      requireReadyAfterWait?: boolean;
    } = {},
  ): Promise<unknown> {
    if (options.waitForActiveRequest) {
      if (
        this.activeBackendRequestSettled !== null &&
        request.service === "instruments" &&
        request.method === "tokenReadout" &&
        typeof request.args[0] === "string" &&
        typeof request.args[1] === "string" &&
        Number.isSafeInteger(request.args[2]) &&
        Number(request.args[2]) >= 0
      ) {
        const rawIndex = Number(request.args[2]);
        onProgress({
          event: "progress",
          data: {
            kind: "token_readout",
            family: request.args[0],
            nodeId: request.args[1],
            rawIndex,
            phase: "queued",
            completed: 0,
            total: rawIndex + 1,
            progress: 0,
            message: "Waiting for the current model task to finish",
          },
        });
      }
      while (this.activeBackendRequestSettled !== null) {
        await this.activeBackendRequestSettled;
      }
      if (options.requireReadyAfterWait) this.requireReady();
    } else if (this.activeBackendRequestController !== null) {
      throw runtimeFailure(
        "RUNTIME_IN_USE",
        "Another browser model request is still settling",
        true,
        409,
      );
    }
    const controller = new AbortController();
    const settled = deferredVoid();
    this.activeBackendRequestController = controller;
    this.activeBackendRequestSettled = settled.promise;
    try {
      const result = await this.modelBackend.request(request, onProgress, controller.signal);
      if (controller.signal.aborted) {
        throw runtimeFailure(
          "MODEL_REQUEST_CANCELLED",
          "The browser model request was cancelled before the runtime changed",
          true,
          499,
        );
      }
      return result;
    } finally {
      settled.resolve();
      if (this.activeBackendRequestController === controller) {
        this.activeBackendRequestController = null;
        this.activeBackendRequestSettled = null;
      }
    }
  }

  private async assertManifoldDeletionSafe(
    request: RuntimeServiceRequest,
  ): Promise<void> {
    const target = deletedManifoldIdentity(request);
    if (
      target === null || !this.backendLoaded || this.snapshot.lifecycle !== "ready"
    ) return;
    const response = await this.requestModelBackend({
      service: "probes",
      method: "list",
      args: [],
    }, () => undefined);
    if (!isRecord(response) || !Array.isArray(response.probes)) {
      throw runtimeFailure(
        "PROBE_RESPONSE_INVALID",
        "The browser runtime returned an invalid probe roster",
        false,
        500,
      );
    }
    const aliases = response.probes.flatMap((value) => {
      if (
        !isRecord(value) || value.family !== "geometry" ||
        typeof value.name !== "string" || typeof value.manifold !== "string"
      ) return [];
      return selectorReferencesManifold(value.manifold, target) ? [value.name] : [];
    });
    if (aliases.length === 0) return;
    throw runtimeFailure(
      "MANIFOLD_IN_USE",
      `Detach ${[...new Set(aliases)].sort().join(", ")} before deleting ${target.namespace}/${target.name}`,
      true,
      409,
      { namespace: target.namespace, name: target.name, probes: aliases },
    );
  }

  private async refreshModelBackendManifolds(): Promise<void> {
    if (this.activeBackendRequestController !== null) {
      throw runtimeFailure(
        "RUNTIME_IN_USE",
        "Another browser model request is still settling",
        true,
        409,
      );
    }
    const controller = new AbortController();
    const settled = deferredVoid();
    this.activeBackendRequestController = controller;
    this.activeBackendRequestSettled = settled.promise;
    try {
      await this.modelBackend.refreshManifolds(controller.signal);
      if (controller.signal.aborted) {
        throw runtimeFailure(
          "MODEL_REQUEST_CANCELLED",
          "The browser model request was cancelled before the runtime changed",
          true,
          499,
        );
      }
    } finally {
      settled.resolve();
      if (this.activeBackendRequestController === controller) {
        this.activeBackendRequestController = null;
        this.activeBackendRequestSettled = null;
      }
    }
  }

  private async invalidateBackendAfterArtifactRefreshFailure(
    error: unknown,
    requestId: string,
    request: RuntimeServiceRequest,
    committedResult: unknown,
  ): Promise<RuntimeFailure> {
    const refreshFailure = asFailure(error);
    this.backendEpoch += 1;
    let sessionCleanupFailure: RuntimeFailure | null = null;
    let backendCleanupFailure: RuntimeFailure | null = null;
    try {
      await this.closeSessionCoordinator();
    } catch (cleanupError) {
      sessionCleanupFailure = asFailure(cleanupError);
    }
    try {
      await this.modelBackend.unload();
    } catch (cleanupError) {
      backendCleanupFailure = asFailure(cleanupError);
    }
    this.backendLoaded = backendCleanupFailure !== null;
    this.loadedSelection = null;
    this.activeGenerationId = null;
    this.generationDeviceLoss = null;
    this.deviceLossRetryKey = null;
    this.deviceLossRetryUsed = false;
    this.capabilityChecker.clearAdapter();
    const failure = runtimeFailure(
      "ARTIFACT_REFRESH_REQUIRED",
      "The manifold change was committed, but the loaded runtime could not adopt it. The active model session was invalidated; reload the model before using model-local operations.",
      true,
      409,
      {
        artifactCommitted: true,
        modelReloadRequired: true,
        committedChange: {
          service: request.service,
          method: request.method,
          result: committedResult,
        },
        refreshFailure,
        ...(sessionCleanupFailure ? { sessionCleanupFailure } : {}),
        ...(backendCleanupFailure ? { backendCleanupFailure } : {}),
      },
    );
    this.patchSnapshot({
      lifecycle: "unloaded",
      modelVariantId: null,
      contextTokens: null,
      error: failure,
    }, requestId);
    return failure;
  }

  private async recordLoad(
    modelVariantId: string,
    runtimeIdentitySha256: string,
    record: Pick<
      DeviceLoadRecord,
      "result" | "contextTokens" | "prefillTokensPerSecond" |
      "decodeTokensPerSecond"
    >,
  ): Promise<void> {
    const capabilities = this.checkedCapabilities;
    if (!capabilities) return;
    await this.contentStore.recordLoad({
      modelVariantId,
      runtimeIdentitySha256,
      deviceSignature: capabilities.deviceSignature,
      recordedAt: Date.now(),
      ...record,
    });
  }

  private async persistGenerationPerformance(
    performance: BrowserGenerationPerformance | void,
    selection: LoadedModelSelection | null,
    backendEpoch: number,
  ): Promise<DeviceLoadRecord[] | null> {
    const prefillTokensPerSecond = positiveRateOrNull(
      performance?.prefillTokensPerSecond,
    );
    const decodeTokensPerSecond = positiveRateOrNull(
      performance?.decodeTokensPerSecond,
    );
    if (
      selection === null ||
      prefillTokensPerSecond === null && decodeTokensPerSecond === null ||
      this.cancelAdmittedGeneration ||
      backendEpoch !== this.backendEpoch ||
      this.loadedSelection !== selection ||
      !this.backendLoaded ||
      this.snapshot.lifecycle !== "ready" ||
      this.generationDeviceLoss !== null
    ) return null;
    await this.recordLoad(selection.variant.id, selection.variant.runtimeIdentitySha256, {
      result: "success",
      contextTokens: selection.contextTokens,
      prefillTokensPerSecond,
      decodeTokensPerSecond,
    });
    return this.contentStore.listLoadRecords();
  }

  private requireReady(): void {
    if (this.snapshot.lifecycle !== "ready") {
      throw runtimeFailure(
        "MODEL_RUNTIME_NOT_READY",
        "Load an installed model before using Drowse model operations",
        true,
        409,
      );
    }
  }

  private requireOperationAvailable(operation: RuntimeCapabilityOperation): void {
    const availability = this.checkedCapabilities?.operations[operation];
    if (availability?.available) return;
    const reason = availability?.reasons.at(-1);
    const fallback = operationUnavailableFallback(operation);
    throw runtimeFailure(
      reason?.code ?? fallback.code,
      reason?.message ?? fallback.message,
      false,
      412,
    );
  }

  private requireRequestCapabilities(
    operations: readonly RuntimeCapabilityOperation[],
  ): void {
    for (const operation of operations) this.requireOperationAvailable(operation);
  }

  private async persistSession(): Promise<void> {
    const coordinator = this.sessionCoordinator;
    const identity = this.sessionIdentity;
    if (coordinator === null || identity === null) {
      throw runtimeFailure(
        "LOCAL_SESSION_NOT_READY",
        "The local conversation store is not bound to the loaded runtime",
        true,
        409,
      );
    }
    await coordinator.persist(identity);
  }

  private async closeSessionCoordinator(
    expectedIdentity?: HostedSessionRuntimeIdentity,
  ): Promise<void> {
    if (expectedIdentity !== undefined && this.sessionIdentity !== expectedIdentity) return;
    const coordinator = this.sessionCoordinator;
    this.sessionCoordinator = null;
    this.sessionIdentity = null;
    if (coordinator !== null) await coordinator.close();
  }

  private async handleCancel(requestId: string): Promise<void> {
    const settled = this.activeDownloadSettled;
    if (!settled) {
      if (this.admittedDownloadRequestId !== null) {
        this.cancelAdmittedDownload = true;
      }
      this.success(requestId, undefined);
      return;
    }
    if (!this.downloader.cancel()) {
      this.success(requestId, undefined);
      return;
    }
    this.patchOperation("download", "cancelling", null, requestId);
    await settled;
    this.success(requestId, undefined);
  }

  private beginFittingCancellation(requestId: string | null): Promise<void> | null {
    if (this.admittedFittingRequestId !== null) {
      this.cancelAdmittedFitting = true;
    }
    const settled = this.activeFittingSettled;
    const controller = this.activeFittingController;
    if (settled === null || controller === null) return null;
    if (!controller.signal.aborted) {
      this.patchOperation("fitting", "cancelling", null, requestId);
      controller.abort();
    }
    return settled;
  }

  private async handleFittingCancel(requestId: string): Promise<void> {
    const settled = this.beginFittingCancellation(requestId);
    if (settled !== null) await settled;
    this.success(requestId, undefined);
  }

  private async ensureContentStore(): Promise<void> {
    await this.contentStore.initialize();
  }

  private async patchPersistedState(requestId: string | null): Promise<void> {
    const [installs, loadRecords, storedSelection] = await Promise.all([
      this.contentStore.listInstalls(),
      this.contentStore.listLoadRecords(),
      this.contentStore.selectedModelVariantId(),
    ]);
    const installedModelVariantIds = modelInstallIds(installs);
    const selectedModelVariantId =
      storedSelection !== null && installedModelVariantIds.includes(storedSelection)
        ? storedSelection
        : null;
    this.patchSnapshot({
      selectedModelVariantId,
      installedModelVariantIds,
      installedPackIds: packInstallIds(installs),
      loadRecords,
    }, requestId);
  }

  private patchSnapshot(
    patch: Partial<RuntimeSnapshot>,
    requestId: string | null,
  ): void {
    this.snapshot = { ...this.snapshot, ...patch };
    this.event("status", this.snapshot, requestId);
  }

  private patchOperation(
    operation: RuntimeOperation,
    phase: RuntimeOperationPhase,
    error: RuntimeFailure | null,
    requestId: string | null,
  ): void {
    this.patchSnapshot({
      [operation]: operationState(this.snapshot[operation], phase, error),
    }, requestId);
  }

  private success(requestId: string, result: unknown): void {
    const response: WorkerSuccessResponse = {
      protocolVersion: RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId,
      ok: true,
      result,
    };
    this.scope.postMessage(response);
  }

  private error(requestId: string, failure: RuntimeFailure): void {
    const response: WorkerErrorResponse = {
      protocolVersion: RUNTIME_PROTOCOL_VERSION,
      kind: "response",
      requestId,
      ok: false,
      error: failure,
    };
    this.scope.postMessage(response);
  }

  private event(
    event: WorkerEventEnvelope["event"],
    payload: WorkerEventEnvelope["payload"],
    requestId: string | null,
    generationId: string | null = null,
  ): void {
    const message: WorkerEventEnvelope = {
      protocolVersion: RUNTIME_PROTOCOL_VERSION,
      kind: "event",
      requestId,
      sequence: ++this.sequence,
      generationId,
      event,
      payload,
    };
    this.scope.postMessage(message);
  }
}

const LOADED_MANIFOLD_MUTATIONS = new Set([
  "create",
  "createDiscover",
  "createFromTemplate",
  "delete",
  "install",
  "merge",
  "drowseArchiveInstall",
  "drowseArchiveDelete",
]);

const CONCURRENT_TREE_MUTATION_METHODS = new Set([
  "branch",
  "delete",
  "edit",
  "navigate",
  "note",
  "reset",
  "restore",
  "star",
  "transcriptLoad",
]);

function isConcurrentTreeMutationRequest(request: WorkerRequest): boolean {
  return request.command === "request" && request.payload.service === "tree" &&
    CONCURRENT_TREE_MUTATION_METHODS.has(request.payload.method);
}

function deletedManifoldIdentity(
  request: RuntimeServiceRequest,
): { namespace: string; name: string } | null {
  if (request.service !== "manifolds") return null;
  if (
    request.method === "delete" && request.args.length === 2 &&
    typeof request.args[0] === "string" && typeof request.args[1] === "string"
  ) {
    return { namespace: request.args[0], name: request.args[1] };
  }
  if (
    request.method === "drowseArchiveDelete" && request.args.length === 1 &&
    typeof request.args[0] === "string"
  ) {
    const parts = request.args[0].split("/");
    if (parts.length === 3 && parts[0] === "manifolds" && parts[1] && parts[2]) {
      return { namespace: parts[1], name: parts[2] };
    }
  }
  return null;
}

function refreshesLoadedManifolds(request: RuntimeServiceRequest): boolean {
  return request.service === "manifolds" && LOADED_MANIFOLD_MUTATIONS.has(request.method);
}

function manifoldMutationChanged(
  request: RuntimeServiceRequest,
  result: unknown,
): boolean {
  if (request.method !== "delete" && request.method !== "drowseArchiveDelete") return true;
  return !isRecord(result) || result.removed !== false;
}

function isHostedInstrumentDownloadRequest(request: RuntimeServiceRequest): boolean {
  if (request.service !== "instruments" || request.method !== "startPreparation") {
    return false;
  }
  const body = request.args[1];
  return isRecord(body) && body.operation === "fetch";
}

function deviceLossSelectionKey(selection: LoadedModelSelection): string {
  return [
    selection.variant.id,
    selection.variant.runtimeIdentitySha256,
    selection.contextTokens,
  ].join(":");
}

function sessionIdentityFor(
  selection: LoadedModelSelection,
): HostedSessionRuntimeIdentity {
  return {
    modelVariantId: selection.variant.id,
    runtimeIdentitySha256: selection.variant.runtimeIdentitySha256,
    contextTokens: selection.contextTokens,
    ownerEpoch: randomUuid(),
  };
}

function compatibleSessionRuntimeIdentitySha256s(
  selection: LoadedModelSelection,
): readonly string[] {
  if (selection.variant.runtimeIdentity.runtimeAbi !== "drowse-web-runtime-v1") {
    return [];
  }
  return LEGACY_PRODUCT_SLUGS.map((slug) => runtimeIdentitySha256({
    ...selection.variant.runtimeIdentity,
    runtimeAbi: `${slug}-web-runtime-v1`,
  }));
}

function validRequest(value: unknown): value is WorkerRequest {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  if (!(
    record.protocolVersion === RUNTIME_PROTOCOL_VERSION &&
    typeof record.requestId === "string" &&
    record.requestId.length > 0 && record.requestId.length <= 128 &&
    typeof record.command === "string" && COMMANDS.has(record.command) &&
    "payload" in record
  )) return false;
  const payload = record.payload;
  switch (record.command) {
    case "check":
    case "cancel":
    case "cancel_fitting":
    case "clear":
    case "persist":
    case "storage":
    case "unload":
    case "takeover":
    case "stop":
      return payload === undefined;
    case "catalog":
      return isRecord(payload) &&
        Object.keys(payload).every((key) =>
          key === "refresh" || key === "offline" || key === "preferCached"
        ) &&
        (payload.refresh === undefined || typeof payload.refresh === "boolean") &&
        (payload.offline === undefined || typeof payload.offline === "boolean") &&
        (payload.preferCached === undefined || typeof payload.preferCached === "boolean");
    case "download":
      return isRecord(payload) &&
        Object.keys(payload).every((key) =>
          key === "modelVariantId" || key === "contextTokens" ||
          key === "explicitUnsafeOverride"
        ) &&
        nonemptyString(payload.modelVariantId) &&
        Number.isSafeInteger(payload.contextTokens) &&
        (payload.contextTokens as number) > 0 &&
        (payload.explicitUnsafeOverride === undefined ||
          typeof payload.explicitUnsafeOverride === "boolean");
    case "download_pack":
      return isRecord(payload) &&
        Object.keys(payload).every((key) =>
          key === "modelVariantId" || key === "packId"
        ) &&
        nonemptyString(payload.modelVariantId) &&
        nonemptyString(payload.packId);
    case "delete":
      return isRecord(payload) &&
        (payload.kind === "model" || payload.kind === "pack") &&
        nonemptyString(payload.id);
    case "load":
      return isRecord(payload) &&
        Object.keys(payload).every((key) =>
          key === "modelVariantId" || key === "contextTokens" ||
          key === "explicitUnsafeOverride" || key === "resetSession"
        ) &&
        nonemptyString(payload.modelVariantId) &&
        Number.isSafeInteger(payload.contextTokens) &&
        (payload.contextTokens as number) > 0 &&
        (payload.explicitUnsafeOverride === undefined ||
          typeof payload.explicitUnsafeOverride === "boolean") &&
        (payload.resetSession === undefined || typeof payload.resetSession === "boolean");
    case "request":
      return isRecord(payload) && isRuntimeServiceName(payload.service) &&
        nonemptyString(payload.method) &&
        isRuntimeServiceMethod(payload.service, payload.method) &&
        Array.isArray(payload.args);
    case "submit":
      return isRecord(payload) && payload.type === "submit";
    case "generate":
      return isRecord(payload) && payload.type === "generate";
  }
  return false;
}

function requestIdFrom(value: unknown): string | null {
  if (!isRecord(value)) return null;
  return nonemptyString(value.requestId) && value.requestId.length <= 128
    ? value.requestId
    : null;
}

const COMMANDS = new Set([
  "check", "catalog", "download", "download_pack", "cancel", "cancel_fitting", "delete", "clear", "persist",
  "storage", "load", "unload", "takeover", "request", "submit", "generate",
  "stop",
]);
const SERVICES = new Set<RuntimeServiceRequest["service"]>([
  "sessions", "profiles", "probes", "manifolds", "templates", "tree",
  "instruments",
]);
const UNSAFE_DOWNLOAD_ADVISORIES = new Set([
  "CALIBRATION_BELOW_PROFILE",
  "DEVICE_MEMORY_BELOW_PROFILE",
  "PROFILE_UNMEASURED",
  "REPEATED_DEVICE_LOSS",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isTreeMutationMessage(value: unknown): value is WSTreeMutatedEvent {
  if (!isRecord(value)) return false;
  if (
    value.type !== "tree_mutated" || !nonemptyString(value.op) ||
    !Number.isSafeInteger(value.rev) || (value.rev as number) < 0
  ) return false;
  if (
    value.active_node_id !== undefined && value.active_node_id !== null &&
    !nonemptyString(value.active_node_id)
  ) return false;
  if (
    value.added !== undefined && (
      !Array.isArray(value.added) ||
      value.added.some((node) => !isRecord(node) || !nonemptyString(node.id))
    )
  ) return false;
  if (
    value.updated !== undefined && (
      !Array.isArray(value.updated) ||
      value.updated.some((node) => !isRecord(node) || !nonemptyString(node.id))
    )
  ) return false;
  if (
    value.removed !== undefined && (
      !Array.isArray(value.removed) || value.removed.some((nodeId) => !nonemptyString(nodeId))
    )
  ) return false;
  return value.cast === undefined || isRecord(value.cast);
}

function nonemptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isRuntimeServiceName(
  value: unknown,
): value is RuntimeServiceRequest["service"] {
  return typeof value === "string" &&
    SERVICES.has(value as RuntimeServiceRequest["service"]);
}

function positiveRateOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : null;
}

function fittingCancelledFailure(): RuntimeFailure {
  return runtimeFailure(
    "FITTING_CANCELLED",
    "The browser fitting job was cancelled",
    true,
    409,
  );
}

function operationUnavailableFallback(
  operation: RuntimeCapabilityOperation,
): { code: string; message: string } {
  switch (operation) {
    case "generation":
      return {
        code: "CUSTOM_MLC_BACKEND_UNAVAILABLE",
        message: "The pinned Drowse MLC runtime has not passed the feasibility gate",
      };
    case "fitting":
      return {
        code: "WASM_FITTING_UNAVAILABLE",
        message: "Browser fitting has not passed Drowse parity gates",
      };
    case "manifold_artifacts":
      return {
        code: "MANIFOLD_ARTIFACT_BRIDGE_UNAVAILABLE",
        message: "Portable manifolds are not connected to the active browser runtime",
      };
    case "jlens_fitting":
      return {
        code: "HOSTED_JLENS_FITTING_EXCLUDED",
        message: "J-lens fitting is available only in the Python runtime",
      };
    case "sae_training":
      return {
        code: "HOSTED_SAE_TRAINING_EXCLUDED",
        message: "SAE training is available only in the Python runtime",
      };
    default:
      return {
        code: "RUNTIME_OPERATION_UNAVAILABLE",
        message: `The browser ${operation.replaceAll("_", " ")} operation is unavailable`,
      };
  }
}

function isFittingCancellation(error: unknown): boolean {
  return isRecord(error) && error.code === "FITTING_CANCELLED";
}

function unavailable(code: string, message: string): RuntimeFailure {
  return runtimeFailure(code, message, false, 503);
}

function asFailure(error: unknown): RuntimeFailure {
  if (error instanceof CatalogFetchError) {
    return runtimeFailure(error.code, error.message, true, 503);
  }
  if (error instanceof CatalogValidationError) {
    const unavailable = new Set([
      "SIGNED_CATALOG_KEYS_UNCONFIGURED",
      "HOSTED_DISTRIBUTION_NOT_VERIFIED",
      "CATALOG_OFFLINE_UNAVAILABLE",
      "ED25519_UNAVAILABLE",
    ]).has(error.code);
    const conflict = error.code === "CATALOG_SUPERSEDED" ||
      error.code === "CATALOG_EXPIRED";
    return runtimeFailure(
      error.code,
      error.message,
      error.code === "CATALOG_OFFLINE_UNAVAILABLE" || conflict,
      unavailable ? 503 : conflict ? 409 : 400,
    );
  }
  if (
    error &&
    typeof error === "object" &&
    typeof (error as Partial<RuntimeFailure>).code === "string" &&
    typeof (error as Partial<RuntimeFailure>).message === "string"
  ) {
    const failure = error as Partial<RuntimeFailure>;
    return {
      code: failure.code!,
      message: failure.message!,
      recoverable: failure.recoverable === true,
      status: typeof failure.status === "number" ? failure.status : 500,
      ...(failure.detail === undefined ? {} : { detail: failure.detail }),
    };
  }
  return runtimeFailure(
    "WORKER_OPERATION_FAILED",
    error instanceof Error ? error.message : String(error),
    true,
    500,
  );
}

function failureMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function loadFailureResult(error: unknown): DeviceLoadRecord["result"] {
  const code = error && typeof error === "object" &&
      typeof (error as { code?: unknown }).code === "string"
    ? (error as { code: string }).code
    : "";
  if (code === "MODEL_OOM" || code === "WEBGPU_OUT_OF_MEMORY") return "oom";
  if (code === "WEBGPU_DEVICE_LOST" || code === "GPU_DEVICE_LOST") {
    return "device_lost";
  }
  return "failed";
}

function shouldRecordLoadFailure(error: unknown): boolean {
  const code = error && typeof error === "object" &&
      typeof (error as { code?: unknown }).code === "string"
    ? (error as { code: string }).code
    : "";
  return code !== "LOCAL_SESSION_INCOMPATIBLE" &&
    code !== "LOCAL_SESSION_SCHEMA_UNSUPPORTED";
}

function shouldRetainFirstRangeProbe(): boolean {
  const connection = (navigator as Navigator & {
    connection?: { downlink?: number; saveData?: boolean };
  }).connection;
  if (connection?.saveData === true) return false;
  return !(
    typeof connection?.downlink === "number" &&
    Number.isFinite(connection.downlink) &&
    connection.downlink > 0
  );
}

interface VoidDeferred {
  promise: Promise<void>;
  resolve(): void;
}

interface GenerationProtocolState {
  phase: "awaiting_started" | "streaming";
  stateless: boolean;
  sawTreeMutation: boolean;
  sawStableTreeMutation: boolean;
  terminalError: boolean;
  activeGenerationId: string | null;
  activeNodeId: string | null;
  siblingIndex: number | null;
  siblingCount: number | null;
  expectedSiblingCount: number | null;
  completedGenerations: number;
  seenGenerationIds: Set<string>;
}

function newGenerationProtocol(stateless: boolean): GenerationProtocolState {
  return {
    phase: "awaiting_started",
    stateless,
    sawTreeMutation: false,
    sawStableTreeMutation: false,
    terminalError: false,
    activeGenerationId: null,
    activeNodeId: null,
    siblingIndex: null,
    siblingCount: null,
    expectedSiblingCount: null,
    completedGenerations: 0,
    seenGenerationIds: new Set(),
  };
}

function assertGenerationProtocolComplete(protocol: GenerationProtocolState): void {
  if (protocol.phase === "streaming") {
    throw generationProtocolViolation(
      `Generation ${protocol.activeGenerationId ?? "unknown"} ended without done`,
    );
  }
  if (protocol.completedGenerations === 0) {
    throw generationProtocolViolation(
      "The backend completed without a started and done sequence",
    );
  }
  if (
    protocol.expectedSiblingCount !== null &&
    protocol.completedGenerations !== protocol.expectedSiblingCount
  ) {
    throw generationProtocolViolation(
      `The backend completed ${protocol.completedGenerations} of ${protocol.expectedSiblingCount} siblings`,
    );
  }
}

function validGenerationNodeId(value: unknown): value is string | null {
  return value === null || (
    typeof value === "string" && value.length > 0
  );
}

function generationProtocolViolation(message: string): RuntimeFailure {
  return runtimeFailure(
    "GENERATION_PROTOCOL_VIOLATION",
    message,
    true,
    502,
  );
}

function deferredVoid(): VoidDeferred {
  let resolve!: () => void;
  const promise = new Promise<void>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

function operationState(
  previous: RuntimeSnapshot[RuntimeOperation],
  phase: RuntimeOperationPhase,
  error: RuntimeFailure | null,
): RuntimeSnapshot[RuntimeOperation] {
  const now = Date.now();
  return {
    phase,
    startedAt: phase === "running" ? now : previous.startedAt,
    finishedAt: phase === "complete" || phase === "failed" ? now : null,
    error,
  };
}

function modelInstallIds(
  installs: import("./contentStore").InstalledContentRecord[],
): string[] {
  return installs
    .filter((install) => install.kind === "model")
    .map((install) => install.id)
    .sort();
}

function packInstallIds(
  installs: import("./contentStore").InstalledContentRecord[],
): string[] {
  return installs
    .filter((install) => install.kind === "pack")
    .map((install) => install.id)
    .sort();
}

function sameObjectHashes(left: readonly string[], right: readonly string[]): boolean {
  const leftSet = new Set(left);
  const rightSet = new Set(right);
  return leftSet.size === rightSet.size &&
    [...leftSet].every((hash) => rightSet.has(hash));
}

function sameVerifiedCatalogSnapshot(
  left: VerifiedCatalog | null,
  right: VerifiedCatalog,
): boolean {
  if (
    left === null || left.signature.schemaVersion !== right.signature.schemaVersion ||
    left.signature.algorithm !== right.signature.algorithm ||
    left.signature.keyId !== right.signature.keyId ||
    left.signature.signature !== right.signature.signature ||
    left.exactBytes.byteLength !== right.exactBytes.byteLength
  ) return false;
  return left.exactBytes.every((byte, index) => byte === right.exactBytes[index]);
}

function capabilitiesWithoutStorageAdmission(
  capabilities: RuntimeCapabilities,
): RuntimeCapabilities {
  return {
    ...capabilities,
    storage: {
      ...capabilities.storage,
      availableBytes: Number.MAX_SAFE_INTEGER,
    },
  };
}
