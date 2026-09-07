import type {
  RuntimeProgressEvent,
  RuntimeServiceRequest,
} from "../../lib/runtime/contracts";
import type {
  InstrumentFamilyBlock,
  ProbeRequest,
  SessionInfo,
  WSGenerateRequest,
  WSServerMessage,
  WSSubmitRequest,
} from "../../lib/types";
import { BrowserLoomRuntime, type BrowserGenerationPort } from "./browserLoom";
import type {
  BrowserGenerationPerformance,
  BrowserGenerationStopReason,
  BrowserModelBackend,
  BrowserModelLoadRequest,
  BrowserModelLoadResult,
} from "./modelBackend";
import { runtimeServiceRoute } from "./operationPolicy";
import type { RankOneHookProgramBuffers } from "./rankOneHookProgram";
import type { StructuredHookProgramBuffers } from "./structuredHookProgram";
import { DrowseWebLlmRuntime, type DrowseWebLlmModule } from "./webLlmEngine";
import type { WebLlmCaptureRuntimePort } from "./webLlmActivationCapture";
import type { BrowserInstrumentRuntime } from "./browserInstrumentRuntime";
import type { WebLlmRuntimeCapabilities } from "./webLlmGeneration";
import type { WebLlmThinkingProfile } from "./webLlmGeneration";
import { outputTokenLimitForRuntime } from "../../lib/runtime/outputTokenPolicy";

interface DrowseWebLlmRuntimePort extends BrowserGenerationPort, WebLlmCaptureRuntimePort {
  load(request: BrowserModelLoadRequest): Promise<BrowserModelLoadResult>;
  unload(): Promise<void>;
  runtimeCapabilities(): WebLlmRuntimeCapabilities;
  thinkingProfileForSession(): WebLlmThinkingProfile | null;
  tokenizeText(text: string): Promise<number[]>;
  decodeTokens(tokenIds: readonly number[]): Promise<string>;
  resolveJlensTokenDirections(
    bindingId: string,
    layerIndices: readonly number[],
    tokenIds: readonly number[],
  ): Promise<Float32Array>;
  generationPerformance(): BrowserGenerationPerformance;
}

export interface BrowserAuthoringRequestContext {
  request: RuntimeServiceRequest;
  loadRequest: BrowserModelLoadRequest;
  runtime: DrowseWebLlmRuntimePort;
  onProgress: (event: RuntimeProgressEvent) => void;
  signal: AbortSignal;
  compileSteering?: (
    expression: string,
  ) => Promise<RankOneHookProgramBuffers | StructuredHookProgramBuffers>;
}

export interface BrowserSteeringCompileContext {
  expression: string;
  loadRequest: BrowserModelLoadRequest;
  probeRequests?: readonly ProbeRequest[];
}

export interface BrowserSteeringDeltaContext {
  parent: string | null;
  child: string | null;
  loadRequest: BrowserModelLoadRequest;
}

export interface DrowseWebLlmBackendOptions {
  drowseVersion: string;
  runtime?: DrowseWebLlmRuntimePort;
  compileSteering?: (
    context: BrowserSteeringCompileContext,
  ) => Promise<RankOneHookProgramBuffers | StructuredHookProgramBuffers> |
    RankOneHookProgramBuffers | StructuredHookProgramBuffers;
  steeringDelta?: (context: BrowserSteeringDeltaContext) => string;
  prepareLoad?: (
    request: BrowserModelLoadRequest,
  ) => Promise<BrowserInstrumentRuntime> | BrowserInstrumentRuntime;
  authoringRequest?: (context: BrowserAuthoringRequestContext) => Promise<unknown>;
  refreshAfterAuthoring?: (context: BrowserAuthoringRequestContext) => Promise<void>;
  refreshManifolds?: (
    loadRequest: BrowserModelLoadRequest,
    signal: AbortSignal,
  ) => Promise<void>;
  onUnload?: () => void;
  now?: () => number;
  createId?: (kind: "generation" | "node") => string;
}

export class DrowseWebLlmBackend implements BrowserModelBackend {
  private readonly runtime: DrowseWebLlmRuntimePort;
  private readonly options: DrowseWebLlmBackendOptions;
  private loom: BrowserLoomRuntime | null = null;
  private loadRequest: BrowserModelLoadRequest | null = null;

  constructor(module: DrowseWebLlmModule, options: DrowseWebLlmBackendOptions) {
    if (!options.drowseVersion) throw new TypeError("Drowse browser source version is required");
    this.runtime = options.runtime ?? new DrowseWebLlmRuntime(module);
    this.options = options;
  }

  async load(request: BrowserModelLoadRequest): Promise<BrowserModelLoadResult> {
    if (this.loom !== null || this.loadRequest !== null) await this.unload();
    const result = await this.runtime.load(request);
    try {
      if (request.signal.aborted) {
        throw request.signal.reason ?? new DOMException("Model loading was cancelled", "AbortError");
      }
      reportLoadProgress(request, "runtime_ready", "Model runtime ready");
      let instruments: BrowserInstrumentRuntime | undefined;
      if (this.options.prepareLoad !== undefined) {
        reportLoadProgress(
          request,
          "workbench_preparing",
          "Preparing response controls and insights",
        );
        instruments = await this.options.prepareLoad(request);
        request.signal.throwIfAborted();
        reportLoadProgress(request, "workbench_ready", "Workbench ready");
      }
      const loom = new BrowserLoomRuntime({
        session: browserSession(
          request,
          this.options.now?.(),
          instruments,
          this.runtime.runtimeCapabilities(),
          this.runtime.thinkingProfileForSession(),
        ),
        generation: this.runtime,
        drowseVersion: this.options.drowseVersion,
        now: this.options.now,
        createId: this.options.createId,
        maxOutputTokens: outputTokenLimitForRuntime(request.runtimeClass),
        ...(instruments ? { instruments } : {}),
        ...(instruments ? { probeHashes: () => instruments.probeHashes() } : {}),
        ...(this.options.compileSteering
          ? {
              compileSteering: (expression: string, probeRequests?: readonly ProbeRequest[]) => this.options.compileSteering!({
                expression,
                loadRequest: request,
                ...(probeRequests === undefined ? {} : { probeRequests }),
              }),
            }
          : {}),
        ...(this.options.steeringDelta
          ? {
              steeringDelta: (parent: string | null, child: string | null) =>
                this.options.steeringDelta!({ parent, child, loadRequest: request }),
            }
          : {}),
      });
      this.loom = loom;
      this.loadRequest = request;
      return result;
    } catch (error) {
      try {
        await this.runtime.unload();
        this.clearLoadedState();
      } catch (cleanupError) {
        throw backendCleanupError(error, cleanupError);
      }
      throw error;
    }
  }

  async unload(): Promise<void> {
    await this.runtime.unload();
    this.clearLoadedState();
  }

  private clearLoadedState(): void {
    this.loom = null;
    this.loadRequest = null;
    this.options.onUnload?.();
  }

  async refreshManifolds(signal: AbortSignal): Promise<void> {
    throwIfAborted(signal);
    await this.options.refreshManifolds?.(this.requireLoadRequest(), signal);
    throwIfAborted(signal);
    this.requireLoom().refreshInstrumentSession();
  }

  async request(
    request: RuntimeServiceRequest,
    onProgress: (event: RuntimeProgressEvent) => void,
    signal: AbortSignal,
  ): Promise<unknown> {
    throwIfAborted(signal);
    let result: unknown;
    if (isBrowserAuthoringRequest(request) && this.options.authoringRequest) {
      const context: BrowserAuthoringRequestContext = {
        request,
        loadRequest: this.requireLoadRequest(),
        runtime: this.runtime,
        onProgress,
        signal,
        ...(this.options.compileSteering
          ? {
              compileSteering: (expression: string) =>
                Promise.resolve(this.options.compileSteering!({
                  expression,
                  loadRequest: this.requireLoadRequest(),
                })),
            }
          : {}),
      };
      result = await this.options.authoringRequest(context);
      await this.options.refreshAfterAuthoring?.(context);
      this.requireLoom().refreshInstrumentSession();
    } else {
      result = await this.requireLoom().request(request, onProgress);
    }
    throwIfAborted(signal);
    return result;
  }

  async generate(
    request: WSSubmitRequest | WSGenerateRequest,
    emit: (message: WSServerMessage) => void | Promise<void>,
    onAdmitted: () => void = () => undefined,
  ): Promise<BrowserGenerationPerformance | void> {
    const generation = this.requireLoom().generate(request, emit);
    onAdmitted();
    await generation;
    if (
      request.type === "submit" && request.text !== null &&
      request.text !== undefined && request.generated_role == null
    ) return;
    return this.runtime.generationPerformance();
  }

  stop(reason: BrowserGenerationStopReason = "internal"): Promise<void> {
    if (this.loom === null) return this.runtime.stop();
    return this.loom.stop(reason === "user");
  }

  private requireLoom(): BrowserLoomRuntime {
    if (!this.loom) {
      throw backendError("MODEL_NOT_LOADED", "No browser model session is loaded");
    }
    return this.loom;
  }

  private requireLoadRequest(): BrowserModelLoadRequest {
    if (!this.loadRequest) {
      throw backendError("MODEL_NOT_LOADED", "No browser model session is loaded");
    }
    return this.loadRequest;
  }
}

function reportLoadProgress(
  request: BrowserModelLoadRequest,
  phase: string,
  message: string,
): void {
  request.onProgress?.({
    event: "progress",
    data: { kind: "model_load", phase, message },
  });
}

function isBrowserAuthoringRequest(request: RuntimeServiceRequest): boolean {
  return runtimeServiceRoute(request) === "fitting";
}

function browserSession(
  request: BrowserModelLoadRequest,
  created?: number,
  instruments?: BrowserInstrumentRuntime,
  runtimeCapabilities?: WebLlmRuntimeCapabilities,
  thinkingProfile: WebLlmThinkingProfile | null = null,
): SessionInfo {
  const supportsThinking = thinkingProfile !== null;
  return {
    id: "default",
    model_id: request.model.id,
    device: "WebGPU",
    dtype: request.variant.runtimeIdentity.quantization,
    created: created ?? Date.now() / 1_000,
    config: {
      temperature: 1,
      top_p: 0.9,
      top_k: null,
      max_tokens: Math.min(
        1024,
        outputTokenLimitForRuntime(request.runtimeClass),
      ),
      system_prompt: null,
      thinking: null,
    },
    profiles: instruments?.profileNames() ?? [],
    probes: [],
    history_length: 0,
    supports_thinking: supportsThinking,
    thinking_is_optional: supportsThinking,
    is_base_model: runtimeCapabilities?.baseModel ?? false,
    jlens_fitted: instruments?.jlensFitted() ?? false,
    instruments: instruments?.blocks() ?? [
      instrument("geometry"),
      instrument("lens"),
      instrument("sae"),
    ],
    default_steering: null,
    role_substitution_supported:
      runtimeCapabilities?.baseModel === true ? false : runtimeCapabilities?.namedRoles ?? false,
    user_role_supported:
      runtimeCapabilities?.baseModel === true ? false : runtimeCapabilities?.userSeatGeneration ?? false,
    default_assistant_role:
      runtimeCapabilities?.baseModel === true
        ? null
        : runtimeCapabilities?.defaultAssistantRole ?? "assistant",
    default_user_role:
      runtimeCapabilities?.baseModel === true
        ? null
        : runtimeCapabilities?.defaultUserRole ?? "user",
    scene_mode:
      runtimeCapabilities?.baseModel === true ? false : runtimeCapabilities?.sceneStitching === true,
    thinking_input_supported: supportsThinking,
    strips_history_thinking: supportsThinking,
  };
}

function instrument(family: "geometry" | "lens" | "sae"): InstrumentFamilyBlock {
  const live = family === "geometry"
    ? { enabled: false }
    : family === "lens"
      ? { enabled: false, layers: [] }
      : { enabled: false, layer: null, source: null };
  return {
    family,
    live,
    source: null,
    probes: [],
    capabilities: {
      sources: false,
      preparations: [],
      token_readout: false,
      source_switch: false,
    },
  } as InstrumentFamilyBlock;
}

function throwIfAborted(signal: AbortSignal): void {
  if (!signal.aborted) return;
  throw signal.reason ?? new DOMException("Browser model request was cancelled", "AbortError");
}

function backendError(code: string, message: string): Error & {
  code: string;
  status: number;
  recoverable: boolean;
} {
  return Object.assign(new Error(message), { code, status: 409, recoverable: true });
}

function backendCleanupError(loadError: unknown, cleanupError: unknown): Error & {
  code: string;
  status: number;
  recoverable: boolean;
  cause: unknown;
} {
  const message = `Browser model setup failed (${errorMessage(loadError)}) and runtime cleanup failed (${errorMessage(cleanupError)})`;
  return Object.assign(new Error(message), {
    code: "RUNTIME_CLEANUP_FAILED",
    status: 500,
    recoverable: false,
    cause: cleanupError,
  });
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
