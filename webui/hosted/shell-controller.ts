import type {
  HostedCheckItem,
  HostedModelOption,
  HostedShellController,
  HostedShellSnapshot,
} from "../src/hosted/ui/types";
import {
  createHostedRuntime,
  type HostedRuntimeBundle,
} from "../src/hosted/runtime/hostedController";
import type {
  CatalogInstrumentPack,
  DownloadProgress,
  ModelRecommendation,
  ModelVariant,
  RuntimeCapabilities,
  RuntimeSnapshot,
} from "../src/lib/runtime/contracts";
import {
  conservativeDownlinkBytesPerSecond,
  roundedEtaRange,
} from "../src/lib/runtime/eta";
import { optionalPackHardwareBlock } from "../src/hosted/runtime/optionalPackCompatibility";
import { hostedNetworkIsOffline } from "../src/hosted/runtime/networkState";
import { userFacingError } from "../src/lib/runtime/userFacingError";

const releaseBuild = typeof __DROWSE_HOSTED_RELEASE__ !== "undefined" &&
  __DROWSE_HOSTED_RELEASE__;

const baseChecks = (state: HostedCheckItem["state"] = "pending"): HostedCheckItem[] => [
  { id: "secure", label: "Private browser connection", detail: "Keeps model work separate and stored locally.", state },
  { id: "webgpu", label: "Graphics support", detail: "The model needs compatible hardware-accelerated graphics.", state },
  { id: "storage", label: "Room on this device", detail: "Models and conversations stay in private browser storage.", state },
  { id: "ownership", label: "One active Drowse tab", detail: "Prevents two tabs from competing for the same graphics hardware.", state },
  {
    id: "runtime",
    label: "Local model engine",
    detail: releaseBuild
      ? "The tested local engine is included in this release."
      : "This development build uses the same published browser artifacts as the release build.",
    state,
  },
];

const models: HostedModelOption[] = [
  {
    id: "gemma3-1b-instruct-q4f16_1",
    modelId: "gemma3-1b-instruct",
    tier: "balanced",
    name: "Gemma 3 1B",
    sourceUrl: "https://huggingface.co/unsloth/gemma-3-1b-it",
    license: "Gemma",
    size: "about 0.9 GB with its tools",
    context: "Short conversations",
    contextTokens: 2048,
    fit: "uncertain",
    installed: false,
    modelDownloadBytes: 625_000_000,
    firstRunBytes: 625_000_000,
    remainingDownloadBytes: 625_000_000,
    firstRunPacks: [],
    setupComplete: false,
    reason: "A compact model with compatible J-Lens and SAE tools.",
    language: "English",
  },
  {
    id: "gemma3-4b-instruct-q4f32_1",
    modelId: "gemma3-4b-instruct",
    tier: "quality",
    name: "Gemma 3 4B",
    sourceUrl: "https://huggingface.co/unsloth/gemma-3-4b-it",
    license: "Gemma",
    size: "about 2.9 GB with its tools",
    context: "Short conversations",
    contextTokens: 2048,
    fit: "uncertain",
    installed: false,
    modelDownloadBytes: 2_320_004_344,
    firstRunBytes: 2_865_904_392,
    remainingDownloadBytes: 2_865_904_392,
    firstRunPacks: [],
    setupComplete: false,
    reason: "A higher-quality Gemma model for devices with more graphics memory.",
    language: "English",
  },
  {
    id: "qwen3-1.7b",
    modelId: "qwen3-1.7b",
    tier: "balanced",
    name: "Qwen3-1.7B",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3-1.7B",
    license: "Apache-2.0",
    size: "about 1.01 GB",
    context: "Short conversations",
    contextTokens: 2048,
    fit: "uncertain",
    installed: false,
    modelDownloadBytes: 1_010_240_829,
    firstRunBytes: 1_010_240_829,
    remainingDownloadBytes: 1_010_240_829,
    firstRunPacks: [],
    setupComplete: false,
    reason: "The smaller Qwen model and the safer choice for most supported computers.",
    language: "Multilingual",
  },
  {
    id: "qwen3-4b",
    modelId: "qwen3-4b",
    tier: "quality",
    name: "Qwen3-4B",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3-4B",
    license: "Apache-2.0",
    size: "about 2.3 GB",
    context: "Short conversations",
    contextTokens: 2048,
    fit: "uncertain",
    installed: false,
    modelDownloadBytes: 2_274_342_846,
    firstRunBytes: 2_274_342_846,
    remainingDownloadBytes: 2_274_342_846,
    firstRunPacks: [],
    setupComplete: false,
    reason: "The highest-quality Qwen model. It needs considerably more graphics memory.",
    language: "Multilingual",
  },
];

const lockedDownload: HostedShellSnapshot["download"] = {
  available: false,
  phase: "locked" as const,
  reason: "The signed public catalog has not been published yet.",
};

const lockedRuntime: HostedShellSnapshot["runtime"] = {
  available: false,
  phase: "locked",
  reason: "The public model catalog has not been published yet.",
};

const idleTakeover: NonNullable<HostedShellSnapshot["takeover"]> = {
  phase: "idle",
  reason: null,
};

interface PendingBusyTakeover {
  decision: Promise<boolean>;
  resolveDecision(value: boolean): void;
  completion: Promise<void>;
  resolveCompletion(): void;
  decisionSettled: boolean;
  completionSettled: boolean;
}

export interface HostedShellControllerOptions {
  runtimeFactory?: typeof createHostedRuntime;
}

export function createShellController(
  hosted?: HostedRuntimeBundle,
  options: HostedShellControllerOptions = {},
): HostedShellController {
  let snapshot: HostedShellSnapshot = {
    phase: "idle",
    headline: "Check this device",
    detail: "We’ll check graphics and storage before showing compatible models.",
    checks: baseChecks(),
    models,
    download: lockedDownload,
    runtime: lockedRuntime,
    takeover: idleTakeover,
  };
  let checkedCapabilities: RuntimeCapabilities | null = null;
  let catalogVariants = new Map<string, ModelVariant>();
  let runtimeSnapshot: RuntimeSnapshot;
  let downloadCancellationRequested = false;
  const listeners = new Set<(next: HostedShellSnapshot) => void>();
  let disposed = false;
  let pendingTakeover: PendingBusyTakeover | null = null;

  const publish = (next: HostedShellSnapshot) => {
    if (disposed) return;
    snapshot = next;
    for (const listener of listeners) listener(snapshot);
  };

  const markStorageProtected = () => publish({
    ...snapshot,
    download: { ...snapshot.download, persistenceDenied: false },
    storage: { ...snapshot.storage, persisted: true },
  });

  const confirmBusyTakeover = (state: RuntimeSnapshot): Promise<boolean> => {
    if (disposed || pendingTakeover !== null) return Promise.resolve(false);
    const pending = createPendingTakeover();
    pendingTakeover = pending;
    publish({
      ...snapshot,
      takeover: {
        phase: "requested",
        reason: busyTakeoverReason(state),
      },
    });
    return pending.decision;
  };

  const ownsTakeoverConfirmation = hosted === undefined;
  const runtimeBundle = hosted ?? (options.runtimeFactory ?? createHostedRuntime)({
    confirmBusyTakeover,
  });
  const runtime = runtimeBundle.controller;
  runtimeSnapshot = runtime.snapshot;

  const unsubscribeRuntime = runtime.subscribe((state) => {
    runtimeSnapshot = state;
    const available = checkedCapabilities?.operations.generation.available ??
      snapshot.runtime.available;
    const phase = state.lifecycle === "ready"
      ? "ready"
      : state.lifecycle === "loading"
        ? "loading"
        : state.lifecycle === "failed"
          ? "failed"
          : available
            ? "unloaded"
            : "locked";
    const finishesAllowedTakeover =
      pendingTakeover?.decisionSettled === true &&
      snapshot.takeover?.phase === "allowing" &&
      (state.lifecycle === "unloaded" || state.lifecycle === "failed");
    const completedTakeover = finishesAllowedTakeover ? pendingTakeover : null;
    if (finishesAllowedTakeover) pendingTakeover = null;
    publish({
      ...snapshot,
      selectedModelVariantId: state.selectedModelVariantId ?? undefined,
      models: snapshot.models.map((model) =>
        refreshModelInstallState(model, state)
      ),
      runtime: {
        available,
        phase,
        reason: state.error
          ? errorMessage(state.error)
          : (phase === "ready"
            ? "The local model is ready."
            : phase === "loading"
              ? "Loading the installed model on this device."
              : snapshot.runtime.reason),
        modelVariantId: state.modelVariantId ?? undefined,
        resetSessionAvailable: state.error !== null && [
          "LOCAL_SESSION_INCOMPATIBLE",
          "LOCAL_SESSION_SCHEMA_UNSUPPORTED",
        ].includes(state.error.code),
      },
      takeover: finishesAllowedTakeover ? idleTakeover : snapshot.takeover,
    });
    if (completedTakeover && !completedTakeover.completionSettled) {
      completedTakeover.completionSettled = true;
      completedTakeover.resolveCompletion();
    }
  });

  return {
    current: () => snapshot,
    subscribe(listener) {
      listeners.add(listener);
      listener(snapshot);
      return () => listeners.delete(listener);
    },
    async check() {
      publish({
        ...snapshot,
        phase: "checking",
        headline: "Checking this device",
        detail: "Checking graphics and storage. This usually takes a few seconds.",
        checks: baseChecks("running"),
      });

      try {
        catalogVariants = new Map();
        const capabilities = await runtime.check();
        checkedCapabilities = capabilities;
        let recommendedModels = models;
        let download: HostedShellSnapshot["download"] = lockedDownload;
        let catalogDetail =
          "Downloads stay locked until the browser runtime and model profiles pass release checks.";
        let firefoxRuntimeIncompatible = false;
        if (capabilities.supported) {
          try {
            const offline = hostedNetworkIsOffline();
            const catalog = await runtime.catalog({
              offline,
              preferCached: offline,
            });
            const recommendations = runtime.recommend(
              catalog,
              capabilities.signals.appleMobile === true ? "speed" : "balanced",
              2048,
            );
            firefoxRuntimeIncompatible =
              capabilities.signals.runtimeClass === "desktop-gecko" &&
              recommendations.length > 0 &&
              recommendations.every((recommendation) =>
                recommendation.hardFailures.some((issue) =>
                  issue.code === "WEBGPU_LIMIT_TOO_LOW"
                )
              );
            catalogVariants = new Map(
              recommendations.map(({ variant }) => [variant.id, variant]),
            );
            recommendedModels = groupModelsByFamily(
              recommendations
                .map((recommendation) =>
                  modelFromRecommendation(
                    recommendation,
                    runtimeSnapshot.installedPackIds,
                    capabilities,
                  )
                ),
            );
            const missingChatModels = models.filter((candidate) =>
              !recommendedModels.some((model) => model.modelId === candidate.modelId)
            );
            recommendedModels.push(...missingChatModels.map((model): HostedModelOption => ({
              ...model,
              catalogAvailable: false,
              fit: "blocked",
              setupIssue: "This chat model is not in the current signed download catalog. You can view its details, but installation is unavailable.",
              reason: "This chat model is not in the current signed download catalog. You can view its details, but installation is unavailable.",
            })));
            const usableModels = recommendedModels.filter((model) => model.fit !== "blocked");
            catalogDetail = usableModels.length === 0
              ? firefoxRuntimeIncompatible
                ? `Firefox is incompatible with the current signed model runtime. ${
                    recommendedModels[0]?.reason ??
                      "Its reported graphics limits are below the runtime requirement."
                  }`
                : recommendedModels[0]?.setupIssue ?? recommendedModels[0]?.reason ??
                  "No signed model in this catalog fits the browser's reported graphics limits."
              : catalog.allowDownloads
                ? capabilities.signals.appleMobile === true
                  ? "Choose a compact model. The first load confirms whether it fits this iPhone or iPad."
                  : isPreviewDesktopRuntime(capabilities)
                    ? "Choose a model to try in this browser preview. Its first successful load becomes the fit evidence for this computer."
                    : "Choose a model to continue."
                : "Your last verified model list can open installed files, but it cannot approve a new download while offline.";
            const downloadCapability = capabilities.operations.download;
            const available = catalog.allowDownloads && downloadCapability.available &&
              usableModels.length > 0;
            download = {
              available,
              phase: available ? "idle" : "locked",
              reason: available
                ? "Downloads are ready."
                : usableModels.length === 0
                  ? firefoxRuntimeIncompatible
                    ? "Firefox is incompatible with the current signed model runtime on this device."
                    : "None of the current signed models fit this browser's reported graphics limits."
                : downloadCapability.reasons[0]
                  ? userFacingError(
                      downloadCapability.reasons[0],
                      "Downloads are not available in this browser. Review the failed device check and try again.",
                    )
                  : "This catalog cannot authorize a new download.",
            };
          } catch (error) {
            catalogDetail = errorMessage(error);
            download = {
              ...lockedDownload,
              reason: errorMessage(error),
            };
          }
        }
        const runtimeAvailability = runtimeStateFromCapabilities(capabilities);
        const usableModelAvailable = recommendedModels.some((model) =>
          model.fit !== "blocked"
        );
        const productReady = capabilities.supported && runtimeAvailability.available &&
          usableModelAvailable &&
          (download.available || recommendedModels.some((model) => model.setupComplete));
        publish({
          ...snapshot,
          phase: capabilities.supported ? "supported" : "unsupported",
          headline: capabilities.supported
              ? productReady
              ? "This device is ready"
              : !usableModelAvailable
                ? firefoxRuntimeIncompatible
                  ? "Firefox is incompatible with the current model runtime"
                  : "The browser works, but the current models do not fit"
              : "This hardware is compatible"
            : "This device is not ready for Drowse",
          detail: capabilities.supported
            ? catalogDetail
            : failureDetail(capabilities),
          checks: checksFromCapabilities(capabilities),
          models: recommendedModels,
          download,
          runtime: runtimeAvailability,
          storage: {
            availableBytes: capabilities.storage.availableBytes ?? undefined,
            persisted: capabilities.storage.persisted ?? undefined,
          },
        });
      } catch (error) {
        publish({
          ...snapshot,
          phase: "failed",
          headline: "The device check could not finish",
          detail: errorMessage(error),
          checks: baseChecks(),
          download: lockedDownload,
          runtime: lockedRuntime,
        });
      }
    },
    setOptionalPackSelected(modelVariantId, packId, selected) {
      if (
        snapshot.download.phase === "requesting_persistence" ||
        snapshot.download.phase === "downloading" ||
        snapshot.download.phase === "cancelling"
      ) {
        throw new Error("Wait for the active download to finish before changing additions");
      }
      const model = snapshot.models.find((candidate) => candidate.id === modelVariantId);
      const variant = catalogVariants.get(modelVariantId);
      if (!model || !variant || !checkedCapabilities) {
        throw new Error("Run the device check before choosing additional tools");
      }
      const pack = model.firstRunPacks.find((candidate) => candidate.id === packId);
      if (!pack) throw new Error("This additional tool is not available for the selected model");
      if (pack.requiredForSetup && !selected) {
        throw new Error(`The compatible ${pack.kind === "sae" ? "SAE" : "J-lens"} is required for this hosted model`);
      }
      if (pack.installed && !selected) {
        throw new Error("Remove an installed addition from Model info");
      }

      let nextPacks = model.firstRunPacks;
      if (selected) {
        const selectedIds = new Set(
          model.firstRunPacks
            .filter((candidate) =>
              candidate.selected &&
              (pack.kind === "jlens" || candidate.kind !== pack.kind)
            )
            .map((candidate) => candidate.id),
        );
        const selectedManifests = variant.packs.filter((candidate) =>
          selectedIds.has(candidate.id)
        );
        const manifest = variant.packs.find((candidate) => candidate.id === pack.id);
        if (!manifest) throw new Error("The selected tool is missing from the verified catalog");
        const hardwareBlock = optionalPackHardwareBlock(
          manifest,
          variant,
          checkedCapabilities,
          selectedManifests,
        );
        if (hardwareBlock !== null) throw new Error(hardwareBlock.message);
        const installedSameKind = model.firstRunPacks.find((candidate) =>
          candidate.kind === pack.kind && candidate.id !== pack.id && candidate.installed
        );
        if (installedSameKind && pack.kind !== "jlens") {
          throw new Error(
            `Remove ${installedSameKind.name} before choosing another feature pack`,
          );
        }
        nextPacks = model.firstRunPacks.map((candidate) => ({
          ...candidate,
          selected: candidate.id === pack.id
            ? true
            : pack.kind !== "jlens" && candidate.kind === pack.kind && !candidate.installed
              ? false
              : candidate.selected,
        }));
      } else {
        nextPacks = model.firstRunPacks.map((candidate) =>
          candidate.id === pack.id ? { ...candidate, selected: false } : candidate
        );
      }
      const nextModel = finalizeModelSetup({ ...model, firstRunPacks: nextPacks });
      publish({
        ...snapshot,
        models: snapshot.models.map((candidate) =>
          candidate.id === modelVariantId ? nextModel : candidate
        ),
        download: snapshot.download.available
          ? {
              ...snapshot.download,
              phase: "idle",
              reason: nextModel.remainingDownloadBytes > 0
                ? "Ready to download."
                : "The selected model and tools are already installed.",
              modelVariantId,
              progress: undefined,
            }
          : snapshot.download,
      });
    },
    async download(modelVariantId, options = {}) {
      const model = snapshot.models.find((candidate) => candidate.id === modelVariantId);
      if (!snapshot.download.available || !model || model.fit === "blocked") {
        throw new Error(model?.setupIssue ?? snapshot.download.reason);
      }
      if (model.setupComplete) return;
      const missingPacks = model.firstRunPacks.filter(
        (pack) => pack.selected && !runtimeSnapshot.installedPackIds.includes(pack.id),
      );
      const remainingBytes = model.remainingDownloadBytes;
      downloadCancellationRequested = false;
      let persistenceGranted = false;
      const persistence = runtime.requestPersistence(() => {
        persistenceGranted = true;
        markStorageProtected();
      });
      publish({
        ...snapshot,
        download: {
          ...snapshot.download,
          phase: "requesting_persistence",
          modelVariantId,
          progress: undefined,
        },
      });
      try {
        persistenceGranted = await persistence || persistenceGranted;
      } catch (error) {
        publish({
          ...snapshot,
          download: {
            ...snapshot.download,
            phase: "failed",
            reason: errorMessage(error),
          },
        });
        throw error;
      }
      let storage: RuntimeCapabilities["storage"];
      try {
        storage = await runtime.refreshStorage();
      } catch (error) {
        publish({
          ...snapshot,
          download: {
            ...snapshot.download,
            phase: "failed",
            reason: errorMessage(error),
            persistenceDenied: !persistenceGranted,
          },
        });
        throw error;
      }
      const persisted = persistenceGranted || storage.persisted === true;
      publish({
        ...snapshot,
        download: {
          ...snapshot.download,
          phase: "downloading",
          persistenceDenied: !persisted,
        },
        storage: {
          availableBytes: storage.availableBytes ?? undefined,
          persisted,
        },
      });
      try {
        let completedBytes = 0;
        let completedFiles: DownloadProgress["files"] = [];
        let lastStepProgress: DownloadProgress | null = null;
        const publishProgress = (progress: DownloadProgress) => {
          lastStepProgress = progress;
          const bytesReceived = Math.min(
            remainingBytes,
            completedBytes + progress.bytesReceived,
          );
          const remaining = Math.max(0, remainingBytes - bytesReceived);
          const throughput = progress.throughputBytesPerSecond;
          const aggregate: DownloadProgress = {
            ...progress,
            files: [...completedFiles, ...progress.files],
            bytesReceived,
            bytesTotal: remainingBytes,
            etaSeconds: remaining === 0
              ? [0, 0]
              : throughput && throughput > 0
                ? roundedEtaRange(remaining / throughput)
                : null,
            calculatingEta: remaining > 0 &&
              (progress.calculatingEta || !throughput || throughput <= 0),
          };
          publish({
            ...snapshot,
            download: {
              ...snapshot.download,
              phase: "downloading",
              progress: aggregate,
            },
          });
        };
        const completeStep = (bytes: number) => {
          completedBytes += bytes;
          if (lastStepProgress) {
            completedFiles = [...completedFiles, ...lastStepProgress.files];
            lastStepProgress = null;
          }
        };
        const pauseDownload = () => {
          publish({
            ...snapshot,
            download: {
              ...snapshot.download,
              phase: "paused",
              reason: "Downloaded files were kept. Resume to finish setup.",
            },
          });
        };

        if (!model.installed) {
          const result = await runtime.download(modelVariantId, publishProgress, {
            contextTokens: model.contextTokens,
            explicitUnsafeOverride: options.explicitUnsafeOverride,
          });
          if (!result.installed || result.cancelled || downloadCancellationRequested) {
            pauseDownload();
            return;
          }
          completeStep(model.modelDownloadBytes);
        }

        for (const pack of missingPacks) {
          if (downloadCancellationRequested) {
            pauseDownload();
            return;
          }
          const result = await runtime.downloadPack(
            modelVariantId,
            pack.id,
            publishProgress,
          );
          if (!result.installed || result.cancelled || downloadCancellationRequested) {
            pauseDownload();
            return;
          }
          completeStep(pack.bytes);
        }

        const completedProgress = snapshot.download.progress
          ? {
              ...snapshot.download.progress,
              bytesReceived: remainingBytes,
              bytesTotal: remainingBytes,
              etaSeconds: [0, 0] as [number, number],
              calculatingEta: false,
              stalled: false,
              offline: false,
            }
          : undefined;
        const completedPackIds = new Set([
          ...runtimeSnapshot.installedPackIds,
          ...missingPacks.map((pack) => pack.id),
        ]);
        publish({
          ...snapshot,
          models: snapshot.models.map((candidate) =>
            candidate.id === modelVariantId
              ? finalizeModelSetup({
                  ...candidate,
                  installed: true,
                  firstRunPacks: candidate.firstRunPacks.map((pack) => ({
                    ...pack,
                    installed: completedPackIds.has(pack.id),
                    selected: completedPackIds.has(pack.id) || pack.selected,
                  })),
                })
              : candidate
          ),
          download: {
            ...snapshot.download,
            phase: "installed",
            reason: missingPacks.length > 0
              ? "The model, response controls, included word insights, and selected additions are verified and installed."
              : "The model and required core geometry are verified and installed.",
            progress: completedProgress,
          },
        });
      } catch (error) {
        publish({
          ...snapshot,
          download: {
            ...snapshot.download,
            phase: "failed",
            reason: errorMessage(error),
          },
        });
        throw error;
      }
    },
    async retryPersistence() {
      let grantedLate = false;
      const persisted = await runtime.requestPersistence(() => {
        grantedLate = true;
        markStorageProtected();
      }) || grantedLate;
      publish({
        ...snapshot,
        download: {
          ...snapshot.download,
          persistenceDenied: !persisted,
        },
        storage: {
          ...snapshot.storage,
          persisted,
        },
      });
      return persisted;
    },
    async cancelDownload() {
      if (snapshot.download.phase !== "downloading") return;
      downloadCancellationRequested = true;
      publish({
        ...snapshot,
        download: { ...snapshot.download, phase: "cancelling" },
      });
      try {
        await runtime.cancelDownload();
      } catch (error) {
        publish({
          ...snapshot,
          download: {
            ...snapshot.download,
            phase: "failed",
            reason: errorMessage(error),
          },
        });
        throw error;
      }
    },
    async open(modelVariantId, options = {}) {
      const model = snapshot.models.find((candidate) => candidate.id === modelVariantId);
      if (!model?.installed) {
        throw new Error("Install and verify this model before opening it");
      }
      if (!snapshot.runtime.available) {
        throw new Error(snapshot.runtime.reason);
      }
      publish({
        ...snapshot,
        runtime: {
          ...snapshot.runtime,
          phase: "loading",
          modelVariantId,
          reason: `Loading ${model.name} on this device.`,
          resetSessionAvailable: false,
        },
      });
      try {
        await runtime.load(modelVariantId, model.contextTokens, {
          explicitUnsafeOverride: model.fit === "uncertain",
          resetSession: options.resetSession === true,
        });
        publish({
          ...snapshot,
          runtime: {
            ...snapshot.runtime,
            phase: "ready",
            modelVariantId,
            reason: `${model.name} is ready on this device.`,
            resetSessionAvailable: false,
          },
        });
      } catch (error) {
        publish({
          ...snapshot,
          runtime: {
            ...snapshot.runtime,
            phase: "failed",
            modelVariantId,
            reason: errorMessage(error),
            resetSessionAvailable: isSessionResetError(error),
          },
        });
        throw error;
      }
    },
    async deleteModel(modelVariantId) {
      if (["requesting_persistence", "downloading", "cancelling"].includes(snapshot.download.phase) || snapshot.runtime.phase === "loading") {
        throw new Error("Wait for the current model operation to finish before deleting a model.");
      }
      await runtime.deleteModel(modelVariantId);
      publish({
        ...snapshot,
        download: {
          available: snapshot.download.available,
          phase: snapshot.download.available ? "idle" : "locked",
          reason: snapshot.download.available ? "Choose a model to download." : snapshot.download.reason,
        },
        runtime: {
          ...snapshot.runtime,
          reason: "Model files removed. Download the model again to continue its chats.",
          resetSessionAvailable: false,
        },
      });
      try {
        const storage = await runtime.refreshStorage();
        publish({ ...snapshot, storage: {
          availableBytes: storage.availableBytes ?? undefined,
          persisted: storage.persisted ?? undefined,
        } });
      } catch { /* A storage estimate is optional after a successful deletion. */ }
    },
    async unload() {
      await runtime.unload();
      publish({
        ...snapshot,
        runtime: {
          ...snapshot.runtime,
          phase: snapshot.runtime.available ? "unloaded" : "locked",
          modelVariantId: undefined,
          reason: snapshot.runtime.available
            ? "The model is installed locally and currently unloaded."
            : snapshot.runtime.reason,
          },
      });
    },
    async prepareForReload() {
      if (runtimeSnapshot.lifecycle !== "failed") {
        await runtime.unload();
      }
      publish({
        ...snapshot,
        runtime: {
          ...snapshot.runtime,
          phase: snapshot.runtime.available ? "unloaded" : "locked",
          modelVariantId: undefined,
          reason: snapshot.runtime.available
            ? "Local work is saved and the model is unloaded."
            : snapshot.runtime.reason,
        },
      });
    },
    async allowBusyTakeover() {
      const pending = pendingTakeover;
      if (!ownsTakeoverConfirmation || pending === null) return;
      if (!pending.decisionSettled) {
        pending.decisionSettled = true;
        publish({
          ...snapshot,
          takeover: {
            phase: "allowing",
            reason: snapshot.takeover?.reason ?? "Releasing the local runtime.",
          },
        });
        pending.resolveDecision(true);
      }
      await pending.completion;
    },
    denyBusyTakeover() {
      if (!ownsTakeoverConfirmation) return;
      const pending = pendingTakeover;
      if (pending === null || pending.decisionSettled) return;
      pendingTakeover = null;
      pending.decisionSettled = true;
      publish({ ...snapshot, takeover: idleTakeover });
      pending.resolveDecision(false);
      if (!pending.completionSettled) {
        pending.completionSettled = true;
        pending.resolveCompletion();
      }
    },
    runtimeClient: () => runtimeBundle.runtime,
    hostedController: () => runtime,
    capabilities: () => checkedCapabilities,
    dispose() {
      if (disposed) return;
      const pending = pendingTakeover;
      pendingTakeover = null;
      if (pending && !pending.decisionSettled) {
        pending.decisionSettled = true;
        pending.resolveDecision(false);
      }
      if (pending && !pending.completionSettled) {
        pending.completionSettled = true;
        pending.resolveCompletion();
      }
      disposed = true;
      unsubscribeRuntime();
      listeners.clear();
      void runtimeBundle.dispose();
    },
  };
}

function createPendingTakeover(): PendingBusyTakeover {
  let resolveDecision!: (value: boolean) => void;
  let resolveCompletion!: () => void;
  const decision = new Promise<boolean>((resolve) => {
    resolveDecision = resolve;
  });
  const completion = new Promise<void>((resolve) => {
    resolveCompletion = resolve;
  });
  return {
    decision,
    resolveDecision,
    completion,
    resolveCompletion,
    decisionSettled: false,
    completionSettled: false,
  };
}

function busyTakeoverReason(state: RuntimeSnapshot): string {
  if (state.generation.phase === "running" || state.generation.phase === "cancelling") {
    return "This tab is generating a response. Releasing the runtime will stop it first.";
  }
  if (state.fitting.phase === "running" || state.fitting.phase === "cancelling") {
    return "This tab is fitting a manifold. Releasing the runtime will stop the active work first.";
  }
  if (state.download.phase === "running" || state.download.phase === "cancelling") {
    return "This tab is downloading model data. Releasing the runtime will preserve the verified partial download.";
  }
  if (state.lifecycle === "loading") {
    return "This tab is loading a model. Releasing it will wait for a clean unload.";
  }
  if (state.lifecycle === "checking") {
    return "This tab is checking the local runtime. Releasing it will wait for that work to settle.";
  }
  return "This tab is using the local GPU runtime. Releasing it will unload the model before the other tab continues.";
}

function modelFromRecommendation(
  recommendation: ModelRecommendation,
  installedPackIds: readonly string[],
  capabilities: RuntimeCapabilities,
): HostedModelOption {
  const { model, variant, context } = recommendation;
  const modelDownloadBytes = variant.downloadBytes + variant.requiredCorePackBytes;
  const compatiblePacks = context === null
    ? []
    : variant.packs.filter((pack) =>
        !pack.required && pack.kind !== "core" &&
        pack.compatibleContextBindingSha256.includes(context.bindingSha256)
      );
  const compatibleJlens = compatiblePacks.find((pack) => pack.kind === "jlens");
  const baseModel = model.modelType === "base";
  const jlensHardwareBlock = compatibleJlens === undefined
    ? null
    : optionalPackHardwareBlock(compatibleJlens, variant, capabilities, []);
  const setupIssue = context === null
    ? "This model has no compatible conversation profile."
    : !baseModel && compatibleJlens === undefined
      ? "This model is not release-ready because it has no compatible precomputed J-lens."
      : !baseModel && jlensHardwareBlock !== null
        ? `This device cannot safely load the J-lens required for this model. ${jlensHardwareBlock.message}`
        : undefined;
  const conservativeRuntime = capabilities.signals.appleMobile === true ||
    isPreviewDesktopRuntime(capabilities);
  const runtimeNotices = conservativeRuntime
    ? [
        capabilities.signals.appleMobile === true
          ? "Live word readouts start off to reduce Safari GPU pressure; turn them on in Model settings when needed."
          : "Live word readouts start off while this browser is in preview; turn them on in Model settings when needed.",
        ...(compatiblePacks.some((pack) => pack.kind === "sae")
          ? ["SAE features remain available as a separate download after the model is set up."]
          : []),
      ]
    : [];
  const toolNotice = runtimeNotices.length > 0
    ? runtimeNotices.join(" ")
    : undefined;
  const selectedPackManifests: CatalogInstrumentPack[] = [];
  const orderedPacks = [...compatiblePacks].sort((left, right) =>
    (left.kind === "jlens" ? 0 : 1) - (right.kind === "jlens" ? 0 : 1)
  );
  const firstRunPacks = orderedPacks.flatMap((pack) => {
    if (
      optionalPackHardwareBlock(
        pack,
        variant,
        capabilities,
        [],
      ) !== null
    ) return [];
    const installed = installedPackIds.includes(pack.id);
    const sameKindSelected = selectedPackManifests.some((candidate) =>
      candidate.kind === pack.kind
    );
    const requiredForSetup = !baseModel && pack.id === compatibleJlens?.id;
    const selected = installed || (
      (requiredForSetup ||
        (!baseModel && !recommendation.installed &&
          !(conservativeRuntime && pack.kind === "sae"))) &&
      !sameKindSelected &&
      optionalPackHardwareBlock(
        pack,
        variant,
        capabilities,
        selectedPackManifests,
      ) === null
    );
    if (selected) selectedPackManifests.push(pack);
    return [{
      id: pack.id,
      kind: pack.kind as "jlens" | "sae",
      name: pack.displayName,
      bytes: pack.bytes,
      requiredForSetup,
      selected,
      installed,
    }];
  });
  const manualOnly = new Set([
    "CALIBRATION_BELOW_PROFILE",
    "DEVICE_MEMORY_BELOW_PROFILE",
    "REPEATED_DEVICE_LOSS",
    "PROFILE_UNMEASURED",
  ]);
  const unknownSignals = new Set([
    "CALIBRATION_UNKNOWN",
    "DEVICE_MEMORY_UNKNOWN",
  ]);
  const uncertain = recommendation.advisories.some((issue) =>
    manualOnly.has(issue.code) ||
    (!recommendation.recommended && unknownSignals.has(issue.code))
  );
  const requiresOomRetry = recommendation.hardFailures.length > 0 &&
    recommendation.hardFailures.every((issue) => issue.code === "CONFIRMED_OOM");
  return finalizeModelSetup({
    id: variant.id,
    modelType: model.modelType ?? "chat",
    modelId: model.id,
    tier: variant.tier,
    name: model.id === "gpt2-base" ? "GPT-2 Base (124M)" : model.displayName,
    sourceUrl: model.sourceUrl,
    license: model.license,
    size: formatModelBytes(modelDownloadBytes),
    context: context
      ? baseModel
        ? `Text completion · ${context.contextTokens.toLocaleString()} tokens`
        : context.contextTokens <= 2048
        ? "Short conversations"
        : "Longer conversations"
      : "Conversation length unavailable",
    fit: setupIssue !== undefined
      ? "blocked"
      : recommendation.recommended
      ? "recommended"
      : requiresOomRetry
        ? "uncertain"
      : !recommendation.eligible
        ? "blocked"
        : uncertain
          ? "uncertain"
          : "eligible",
    installed: recommendation.installed,
    contextTokens: context?.contextTokens ?? 2048,
    reason: setupIssue ?? (
      recommendation.hardFailures[0]
        ? userFacingError(
            recommendation.hardFailures[0],
            "This model is not compatible with this browser or device. Choose another model.",
          )
        : recommendation.reason
    ),
    language: model.languages.includes("multilingual")
      ? "Multilingual"
      : model.languages.join(", ").toUpperCase(),
    estimatedDownloadSeconds: undefined,
    expectedSpeed: context?.expectedDecodeTokensPerSecond
      ? `${formatRate(context.expectedDecodeTokensPerSecond[0])}–${formatRate(context.expectedDecodeTokensPerSecond[1])} tok/s`
      : undefined,
    modelDownloadBytes,
    firstRunBytes: modelDownloadBytes,
    remainingDownloadBytes: recommendation.installed ? 0 : modelDownloadBytes,
    firstRunPacks,
    toolNotice,
    setupIssue,
    setupComplete: recommendation.installed,
    requiresOomRetry,
  });
}

function groupModelsByFamily(models: HostedModelOption[]): HostedModelOption[] {
  const familyOrder = new Map<string, number>();
  const ranked = models.map((model, index) => {
    const family = modelFamily(model.name);
    if (!familyOrder.has(family)) familyOrder.set(family, familyOrder.size);
    return { model, family, index };
  });
  return ranked
    .sort((left, right) =>
      familyOrder.get(left.family)! - familyOrder.get(right.family)! ||
      left.index - right.index
    )
    .map(({ model }) => model);
}

function modelFamily(name: string): string {
  return name
    .replace(/[\s_-]+\d+(?:\.\d+)?\s*[bmk](?:\s.*)?$/iu, "")
    .trim()
    .toLocaleLowerCase();
}

function isPreviewDesktopRuntime(capabilities: RuntimeCapabilities): boolean {
  return capabilities.signals.runtimeClass === "desktop-webkit" ||
    capabilities.signals.runtimeClass === "desktop-gecko";
}

function refreshModelInstallState(
  model: HostedModelOption,
  state: RuntimeSnapshot,
): HostedModelOption {
  const installed = state.installedModelVariantIds.includes(model.id);
  return finalizeModelSetup({
    ...model,
    installed,
    firstRunPacks: model.firstRunPacks.map((pack) => {
      const packInstalled = state.installedPackIds.includes(pack.id);
      return {
        ...pack,
        installed: packInstalled,
        selected: packInstalled || pack.selected,
      };
    }),
  });
}

function finalizeModelSetup(model: HostedModelOption): HostedModelOption {
  const selectedPacks = model.firstRunPacks.filter((pack) => pack.selected);
  const firstRunBytes = model.modelDownloadBytes + selectedPacks.reduce(
    (total, pack) => total + pack.bytes,
    0,
  );
  const remainingDownloadBytes = (model.installed ? 0 : model.modelDownloadBytes) +
    selectedPacks.reduce(
      (total, pack) => total + (pack.installed ? 0 : pack.bytes),
      0,
    );
  return {
    ...model,
    size: formatModelBytes(firstRunBytes),
    firstRunBytes,
    remainingDownloadBytes,
    estimatedDownloadSeconds: remainingDownloadBytes > 0
      ? initialDownloadEta(remainingDownloadBytes)
      : undefined,
    setupComplete: model.installed && model.setupIssue === undefined &&
      remainingDownloadBytes === 0 &&
      model.firstRunPacks
        .filter((pack) => pack.requiredForSetup)
        .every((pack) => pack.installed),
  };
}

function runtimeStateFromCapabilities(
  capabilities: RuntimeCapabilities,
): HostedShellSnapshot["runtime"] {
  const availability = capabilities.operations.generation;
  return {
    available: availability.available,
    phase: availability.available ? "unloaded" : "locked",
    reason: availability.available
      ? "An installed model can be loaded on this device."
      : availability.reasons[0]
        ? userFacingError(
            availability.reasons[0],
            "The local model engine is not available in this browser. Review the failed device check and try again.",
          )
        : "The local model engine is unavailable in this build.",
  };
}

function formatModelBytes(bytes: number): string {
  return bytes >= 1_000_000_000
    ? `${(bytes / 1_000_000_000).toFixed(1)} GB`
    : `${Math.ceil(bytes / 1_000_000)} MB`;
}

function initialDownloadEta(bytes: number): [number, number] | undefined {
  const connection = (navigator as Navigator & {
    connection?: { downlink?: number; saveData?: boolean };
  }).connection;
  if (connection?.saveData === true) return undefined;
  const throughput = conservativeDownlinkBytesPerSecond(
    connection?.downlink ?? 0,
  );
  return throughput > 0 ? roundedEtaRange(bytes / throughput) : undefined;
}

function formatRate(value: number): string {
  return value >= 10 ? Math.round(value).toString() : value.toFixed(1);
}

function errorMessage(error: unknown): string {
  return userFacingError(
    error,
    "Drowse could not complete that setup step. Try again.",
  );
}

function isSessionResetError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const record = error as {
    code?: unknown;
    body?: { error?: { code?: unknown } };
  };
  const code = typeof record.code === "string"
    ? record.code
    : record.body?.error?.code;
  return code === "LOCAL_SESSION_INCOMPATIBLE" ||
    code === "LOCAL_SESSION_SCHEMA_UNSUPPORTED";
}

function checksFromCapabilities(capabilities: RuntimeCapabilities): HostedCheckItem[] {
  const adapter = capabilities.webGpu.adapterInfo;
  const adapterLabel = [adapter?.vendor, adapter?.architecture].filter(Boolean).join(" / ");
  const gpuHint = capabilities.issues.find((issue) => issue.code === "WINDOWS_INTEL_GPU");
  const graphicsFailure = capabilities.issues.find((issue) =>
    issue.severity === "hard" && (
      issue.code.startsWith("WEBGPU_") ||
      issue.code === "SOFTWARE_ADAPTER" ||
      issue.code === "ADAPTER_KIND_UNKNOWN"
    )
  );
  const webGpuState = graphicsFailure || !capabilities.webGpu.available ||
      capabilities.webGpu.fallback === "fallback"
    ? "fail"
    : capabilities.webGpu.fallback === "unknown"
      ? capabilities.signals.calibrationScore === null ? "fail" : "warn"
      : "pass";
  const storageState = !capabilities.opfs || !capabilities.indexedDb
    ? "fail"
    : capabilities.storage.availableBytes === null ? "warn" : "pass";
  const runtimePrerequisites = capabilities.dedicatedWorker && capabilities.webAssembly;
  const runtimeReady = capabilities.operations.generation.available;
  return [
    {
      id: "secure",
      label: "Private browser connection",
      state: capabilities.secureContext && capabilities.crossOriginIsolated ? "pass" : "fail",
      detail: !capabilities.secureContext
        ? "Open Drowse over HTTPS or localhost."
        : !capabilities.crossOriginIsolated
          ? "This site is missing the browser isolation settings needed to keep model work separate."
          : "The private browser features Drowse needs are active.",
    },
    {
      id: "webgpu",
      label: "Graphics support",
      state: webGpuState === "pass" && gpuHint ? "warn" : webGpuState,
      detail: (adapterLabel ? `Selected GPU: ${adapterLabel}. ` : "") + (graphicsFailure
        ? userFacingError(
            graphicsFailure,
            "This browser could not provide the graphics support Drowse needs. Update it and run the device check again.",
          )
        : !capabilities.webGpu.available
        ? "This browser could not find compatible graphics hardware."
        : capabilities.webGpu.fallback === "fallback"
          ? "This browser is using software graphics, which cannot run a Drowse model."
        : capabilities.webGpu.fallback === "unknown"
            ? capabilities.signals.calibrationScore === null
              ? "The browser did not identify its graphics adapter, and the compute check could not finish."
              : "The browser did not name its graphics adapter, but the WebGPU compute check passed."
            : "The basic graphics check passed. This does not guarantee stability under model load.") +
          (gpuHint ? ` ${gpuHint.message}` : ""),
    },
    {
      id: "storage",
      label: "Room on this device",
      state: storageState,
      detail: !capabilities.opfs || !capabilities.indexedDb
        ? "This browser cannot provide the private local storage Drowse needs."
        : capabilities.storage.availableBytes === null
          ? "Local storage is available, but the browser did not report how much room remains."
          : "Private local storage and a space estimate are available.",
    },
    {
      id: "ownership",
      label: "One active Drowse tab",
      state: capabilities.webLocks && capabilities.broadcastChannel ? "pass" : "fail",
      detail: capabilities.webLocks && capabilities.broadcastChannel
        ? "Drowse can prevent two tabs from competing for the same graphics hardware."
        : !capabilities.webLocks && !capabilities.broadcastChannel
          ? "This browser cannot coordinate model use across tabs."
          : !capabilities.webLocks
            ? "This browser cannot reserve the local model for one tab."
            : "This browser cannot ask another Drowse tab to release the local model.",
    },
    {
      id: "runtime",
      label: "Local model engine",
      state: runtimePrerequisites ? runtimeReady ? "pass" : "warn" : "fail",
      detail: runtimePrerequisites
        ? runtimeReady
          ? "The tested local engine is ready to load a model."
          : "This development build can show the interface, but its local model engine has not passed release checks yet."
        : "This browser is missing features needed to run the model away from the main interface.",
    },
  ];
}

function failureDetail(capabilities: RuntimeCapabilities): string {
  const failure = capabilities.issues.find((issue) => issue.severity === "hard");
  const plainFailures: Record<string, string> = {
    INSECURE_CONTEXT: "Open Drowse over a secure HTTPS connection or on localhost",
    CROSS_ORIGIN_ISOLATION_REQUIRED: "This site's browser security setup is incomplete",
    WORKER_UNAVAILABLE: "This browser cannot run model work away from the main interface",
    INDEXEDDB_UNAVAILABLE: "This browser cannot save Drowse settings and conversations locally",
    OPFS_UNAVAILABLE: "This browser cannot store model files privately on this device",
    OPFS_READ_WRITE_FAILED: "Private local storage could not complete the file create, resume, read, list, and delete check",
    WEB_LOCKS_UNAVAILABLE: "This browser cannot reserve the local model for one tab",
    BROADCAST_CHANNEL_UNAVAILABLE: "This browser cannot coordinate model use between Drowse tabs",
    WASM_UNAVAILABLE: "This browser is missing a feature needed by the local model engine",
    WEBGPU_UNAVAILABLE: "This browser does not provide compatible graphics support",
    WEBGPU_ADAPTER_TIMEOUT: "The browser did not finish finding a graphics adapter",
    WEBGPU_ADAPTER_FAILED: "The browser could not open its graphics adapter",
    WEBGPU_ADAPTER_MISSING: "No compatible graphics hardware was found",
    WEBGPU_DEVICE_TIMEOUT: "The browser did not finish creating a WebGPU device",
    WEBGPU_DEVICE_FAILED: "The browser could not create a WebGPU device",
    WEBGPU_CALIBRATION_TIMEOUT: "The graphics compute self-test did not finish",
    WEBGPU_CALIBRATION_FAILED: "The graphics compute self-test failed",
    WEBGPU_CALIBRATION_UNAVAILABLE: "The browser could not run the graphics compute self-test",
    SOFTWARE_ADAPTER: "Hardware-accelerated graphics are not active",
    ADAPTER_KIND_UNKNOWN: "The browser could not confirm hardware-accelerated graphics",
  };
  const detail = failure
    ? plainFailures[failure.code] ?? userFacingError(
        failure,
        "This browser or device is missing a feature Drowse needs",
      )
    : null;
  return detail
    ? `${detail.replace(/\s+$/, "").replace(/([^.!?])$/, "$1.")} Drowse will not switch to cloud processing.`
    : "Resolve the failed checks before downloading a model. Drowse will not switch to cloud processing.";
}
