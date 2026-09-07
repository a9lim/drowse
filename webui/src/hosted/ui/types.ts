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
}

export interface HostedShellSnapshot {
  phase: "idle" | "checking" | "supported" | "unsupported" | "failed";
  headline: string;
  detail: string;
  checks: HostedCheckItem[];
  models: HostedModelOption[];
  selectedModelVariantId?: string;
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
  open(
    modelVariantId: string,
    options?: { resetSession?: boolean },
  ): Promise<void>;
  unload(): Promise<void>;
  prepareForReload(): Promise<void>;
  allowBusyTakeover(): Promise<void>;
  denyBusyTakeover(): void;
  runtimeClient(): import("../../lib/runtime/contracts").RuntimeClient;
  hostedController(): import("../../lib/runtime/contracts").HostedController;
  capabilities(): import("../../lib/runtime/contracts").RuntimeCapabilities | null;
  dispose(): void;
}
import type { DownloadProgress } from "../../lib/runtime/contracts";
