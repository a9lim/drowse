import runtimeLock from "../../../../browser-runtime/runtime-lock.json";
import type {
  ModelThinkingProfile,
  StructuredHookProfileId,
} from "../../lib/runtime/contracts";

export type BrowserRuntimeReleaseStatus = "feasibility-required" | "verified";

export const BROWSER_RUNTIME_LOCK = Object.freeze({
  status: runtimeLock.status as BrowserRuntimeReleaseStatus,
  runtimeAbi: runtimeLock.runtimeAbi,
  hookAbi: runtimeLock.hookAbi,
  models: runtimeLock.models.map((model) => Object.freeze({
    id: model.id,
    modelType: "modelType" in model && model.modelType === "base" ? "base" as const : "chat" as const,
    sourceRepository: model.sourceRepository,
    sourceRevision: model.sourceRevision,
    convertedRepository: model.convertedRepository,
    convertedRevision: model.convertedRevision,
    manifestSha256: model.manifestSha256,
    librarySha256: model.librarySha256,
    tokenizerSha256: model.tokenizerSha256,
    chatTemplateSha256: model.chatTemplateSha256,
    hiddenSize: model.hiddenSize,
    layerMap: model.layerMap,
    quantization: model.quantization,
    contextProfiles: model.contextProfiles,
    structuredHookProfile: model.structuredHookProfile as StructuredHookProfileId,
    thinkingProfile: model.thinkingProfile as ModelThinkingProfile | null,
  })),
});

export const BROWSER_RUNTIME_RELEASE_VERIFIED =
  BROWSER_RUNTIME_LOCK.status === "verified";
