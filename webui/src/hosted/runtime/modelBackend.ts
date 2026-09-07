import type {
  BrowserRuntimeClass,
  CatalogFile,
  CatalogInstrumentPack,
  CatalogModel,
  ModelVariant,
  RuntimeProgressEvent,
  RuntimeServiceRequest,
} from "../../lib/runtime/contracts";
import type {
  WSGenerateRequest,
  WSServerMessage,
  WSSubmitRequest,
} from "../../lib/types";
import type { ActivationSpool } from "../fitting/activationSpool";
import type { GpuAdapterLike } from "./capabilities";

export interface BrowserModelArtifact {
  manifest: CatalogFile;
  file: File;
}

export interface BrowserInstrumentPackArtifacts {
  pack: CatalogInstrumentPack;
  artifacts: BrowserModelArtifact[];
}

export interface BrowserModelLoadRequest {
  model: CatalogModel;
  variant: ModelVariant;
  requiredCorePack: CatalogInstrumentPack;
  contextTokens: number;
  runtimeClass?: BrowserRuntimeClass;
  adapter: GpuAdapterLike | null;
  artifacts: BrowserModelArtifact[];
  optionalPacks: BrowserInstrumentPackArtifacts[];
  activationSpool: ActivationSpool;
  signal: AbortSignal;
  onProgress?: (event: RuntimeProgressEvent) => void;
  onDeviceLost(failure: {
    code: string;
    message: string;
    confirmedOom: boolean;
  }): void;
}

export interface BrowserModelLoadResult {
  prefillTokensPerSecond: number | null;
  decodeTokensPerSecond: number | null;
}

export type BrowserGenerationPerformance = BrowserModelLoadResult;
export type BrowserGenerationStopReason = "user" | "internal";

export interface BrowserModelBackend {
  load(request: BrowserModelLoadRequest): Promise<BrowserModelLoadResult>;
  unload(): Promise<void>;
  refreshManifolds(signal: AbortSignal): Promise<void>;
  // Aborting must roll back staged fitting artifacts and settle before unload can proceed.
  request(
    request: RuntimeServiceRequest,
    onProgress: (event: RuntimeProgressEvent) => void,
    signal: AbortSignal,
  ): Promise<unknown>;
  generate(
    request: WSSubmitRequest | WSGenerateRequest,
    emit: (message: WSServerMessage) => void | Promise<void>,
    onAdmitted?: () => void,
  ): Promise<BrowserGenerationPerformance | void>;
  stop(reason?: BrowserGenerationStopReason): Promise<void>;
}

export class UnavailableBrowserModelBackend implements BrowserModelBackend {
  async load(): Promise<never> {
    throw backendUnavailable();
  }

  async unload(): Promise<void> {}

  async refreshManifolds(): Promise<void> {}

  async request(): Promise<never> {
    throw backendUnavailable();
  }

  async generate(): Promise<never> {
    throw backendUnavailable();
  }

  async stop(): Promise<void> {}
}

function backendUnavailable(): Error & {
  code: string;
  status: number;
  recoverable: boolean;
} {
  return Object.assign(
    new Error(
      "The Drowse MLC fork with post-block steering hooks is not available; stock WebLLM is not used as a fallback",
    ),
    {
      code: "CUSTOM_MLC_BACKEND_UNAVAILABLE",
      status: 503,
      recoverable: false,
    },
  );
}
