export type HostedCheckState = "pending" | "running" | "pass" | "warn" | "fail";

export interface HostedCheckItem {
  id: string;
  label: string;
  detail: string;
  state: HostedCheckState;
}

export interface HostedModelOption {
  id: string;
  modelType?: "chat" | "base";
  modelId: string;
  catalogAvailable?: boolean;
  tier: "fastest" | "balanced" | "quality";
  name: string;
  sourceUrl: string;
  license: string;
  size: string;
  context: string;
  contextTokens: number;
  fit: "recommended" | "eligible" | "uncertain" | "blocked";
  installed: boolean;
  reason: string;
  language: string;
  estimatedDownloadSeconds?: [number, number];
  expectedSpeed?: string;
  modelDownloadBytes: number;
  firstRunBytes: number;
  remainingDownloadBytes: number;
  firstRunPacks: Array<{
    id: string;
    kind: "jlens" | "sae";
    name: string;
    bytes: number;
    requiredForSetup: boolean;
    selected: boolean;
    installed: boolean;
  }>;
  toolNotice?: string;
  setupIssue?: string;
  setupComplete: boolean;
  requiresOomRetry?: boolean;
  /** Verified bytes of an unfinished download kept from an earlier visit. */
  partialBytes?: number;
  /** Latest decode speed measured on this device for this exact setup. */
  measuredTokensPerSecond?: number;
}

export interface HostedShellSnapshot {
  phase: "idle" | "checking" | "supported" | "unsupported" | "failed";
  headline: string;
  detail: string;
  checks: HostedCheckItem[];
  models: HostedModelOption[];
  selectedModelVariantId?: string;
  /** Set when the signed model list could not be loaded or verified. */
  catalogError?: string;
  download: {
    available: boolean;
    phase:
      | "locked"
      | "idle"
      | "requesting_persistence"
      | "downloading"
      | "cancelling"
      | "paused"
      | "installed"
      | "failed";
    reason: string;
    modelVariantId?: string;
    progress?: DownloadProgress;
    persistenceDenied?: boolean;
  };
  runtime: {
    available: boolean;
    phase: "locked" | "unloaded" | "loading" | "ready" | "failed";
    reason: string;
    modelVariantId?: string;
    resetSessionAvailable?: boolean;
  };
  storage?: {
    availableBytes?: number;
    quotaBytes?: number;
    usageBytes?: number;
    persisted?: boolean;
  };
  takeover?: {
    phase: "idle" | "requested" | "allowing";
    reason: string | null;
  };
}

export interface HostedShellController {
  current(): HostedShellSnapshot;
  subscribe(listener: (snapshot: HostedShellSnapshot) => void): () => void;
  check(): Promise<void>;
  download(
    modelVariantId: string,
    options?: { explicitUnsafeOverride?: boolean },
  ): Promise<void>;
  retryPersistence(): Promise<boolean>;
  setOptionalPackSelected(
    modelVariantId: string,
    packId: string,
    selected: boolean,
  ): void;
  cancelDownload(): Promise<void>;
  deleteModel(modelVariantId: string): Promise<void>;
  /** Present only when the runtime can remove kept partial download files. */
  discardPartialDownload?(modelVariantId: string): Promise<void>;
  open(
    modelVariantId: string,
    options?: { resetSession?: boolean },
  ): Promise<void>;
  unload(): Promise<void>;
  prepareForReload(): Promise<void>;
  allowBusyTakeover(): Promise<void>;
  denyBusyTakeover(): void;
  runtimeClient(): import("./runtime/contracts").RuntimeClient;
  hostedController(): import("./runtime/contracts").HostedController;
  capabilities(): import("./runtime/contracts").RuntimeCapabilities | null;
  dispose(): void;
}
import type { DownloadProgress } from "./runtime/contracts";
