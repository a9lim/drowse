import type {
  BrowserModelLoadRequest,
  BrowserModelLoadResult,
} from "./modelBackend";
import type {
  ModelVariantRequirements,
  StructuredHookProfileId,
} from "../../lib/runtime/contracts";
import {
  MAX_OUTPUT_TOKEN_COUNT,
  outputTokenLimitForRuntime,
} from "../../lib/runtime/outputTokenPolicy";
import type { GpuAdapterLike } from "./capabilities";
import type { RankOneHookProgramBuffers } from "./rankOneHookProgram";
import {
  STRUCTURED_HOOK_FORMAT,
  STRUCTURED_HOOK_FORMAT_V2,
  type StructuredHookProgramBuffers,
} from "./structuredHookProgram";
import {
  EXACT_READOUT_REQUIRED_WEBGPU_LIMITS,
  resolveStructuredHookProfile,
  structuredCurveParameterStride,
  structuredGeometryPayloadLayout,
  WEB_LLM_RUNTIME_MIN_STORAGE_BUFFERS_PER_SHADER_STAGE,
} from "./structuredHookProfile";
import { prepareVerifiedWebLlmArtifacts } from "./verifiedArtifactCache";
import {
  assertHookFeaturesSupported,
  AFFINE_FEASIBILITY_CAPABILITIES,
  type DrowseHookCapabilities,
} from "./hookCapabilities";
import {
  BROWSER_EXACT_READOUT_TOP_K_CAPACITY,
  streamWebLlmGeneration,
  readWebLlmRuntimeCapabilities,
  tokenizeWebLlmText,
  type WebLlmJlensTopTokenReadout,
  type WebLlmGeneratedToken,
  type WebLlmGenerationEngine,
  type WebLlmGenerationPlan,
  type WebLlmGenerationResult,
  type WebLlmSaeTopFeatureReadout,
  type WebLlmThinkingProfile,
  type WebLlmRuntimeCapabilities,
} from "./webLlmGeneration";
import type {
  BrowserJlensGpuDictionary,
  BrowserSaeGpuDictionary,
} from "./browserInstrumentPacks";

const DROWSE_READOUT_TOP_K = BROWSER_EXACT_READOUT_TOP_K_CAPACITY;
const DROWSE_EXACT_READOUT_ABI = "exact-readout-v1";
const DROWSE_EXACT_READOUT_ABI_VERSION = 1;
const DROWSE_EXACT_READOUT_MAX_SAE_FEATURES_PER_CHUNK = 16_384;
export const WEB_LLM_DECODED_TOKEN_CACHE_MAX = 4096;

interface DrowseSaeDictionary extends BrowserSaeGpuDictionary {
  hookAbi: string;
}

interface DrowseJlensDictionary extends BrowserJlensGpuDictionary {
  hookAbi: string;
}

interface DrowseJlensTopTokenReadout {
  tokenIds: Int32Array;
  strength: Float32Array;
  centerOfMass: Float32Array;
  spread: Float32Array;
  fittedLayerCount: number;
  layerIndices: Int32Array;
  layerTokenIds: Int32Array;
  layerProbabilities: Float32Array;
}

interface DrowseSaeTopFeatureReadout {
  featureIds: Int32Array;
  activations: Float32Array;
  runtimeLayerIndex: number;
  featureCount: number;
}

interface DrowseMeasurementBundle {
  scalar?: Float32Array;
  geometry?: Float32Array;
  jlensTopTokens?: DrowseJlensTopTokenReadout;
  saeTopFeatures?: DrowseSaeTopFeatureReadout;
}

export interface DrowseWebLlmEngineInstance extends WebLlmGenerationEngine {
  reload(modelId: string): Promise<void>;
  unload(): Promise<void>;
  interruptGenerate(): Promise<void> | void;
  resetChat(): Promise<void>;
  supportsDrowseRankOneHooks(modelId?: string): Promise<boolean>;
  supportsDrowseStructuredHooks(modelId?: string): Promise<boolean>;
  supportsDrowseCurvedHooks(modelId?: string): Promise<boolean>;
  supportsDrowseResidualCapture(modelId?: string): Promise<boolean>;
  supportsDrowseRankOneResidualCaptureV1(modelId?: string): Promise<boolean>;
  prepareDrowseCaptureRows(
    rows: DrowseCaptureRow[],
    specialTokenIds: number[],
    modelId?: string,
  ): Promise<DrowsePreparedCaptureRow[]>;
  captureDrowseResiduals(
    inputIds: number[],
    positions: number[],
    modelId?: string,
  ): Promise<DrowseResidualCapture>;
  captureDrowseRankOneResidualsV1(
    inputIds: number[],
    positions: number[],
    program: RankOneHookProgramBuffers,
    modelId?: string,
  ): Promise<DrowseRankOneResidualCaptureV1>;
  setDrowseRankOneProgram(
    program: RankOneHookProgramBuffers,
    modelId?: string,
  ): Promise<void>;
  setDrowseStructuredProgram(
    program: StructuredHookProgramBuffers,
    modelId?: string,
  ): Promise<void>;
  updateDrowseStructuredControls(
    affineActive: Uint32Array,
    curveActive?: Uint32Array,
    modelId?: string,
  ): Promise<void>;
  clearDrowseRankOneProgram(modelId?: string): Promise<void>;
  setDrowseJlensDictionary(
    dictionary: DrowseJlensDictionary,
    modelId?: string,
  ): Promise<void>;
  clearDrowseJlensDictionary(modelId?: string): Promise<void>;
  setDrowseSaeDictionary(
    dictionary: DrowseSaeDictionary,
    modelId?: string,
  ): Promise<void>;
  clearDrowseSaeDictionary(modelId?: string): Promise<void>;
  readDrowseMeasurementBundle(
    modelId?: string,
  ): Promise<DrowseMeasurementBundle>;
  readDrowseJlensTopTokens(
    modelId?: string,
  ): Promise<DrowseJlensTopTokenReadout | undefined>;
  readDrowseSaeTopFeatures(
    modelId?: string,
  ): Promise<DrowseSaeTopFeatureReadout | undefined>;
  readDrowseMeasurements(modelId?: string): Promise<Float32Array | undefined>;
  readDrowseGeometryMeasurements(
    modelId?: string,
  ): Promise<Float32Array | undefined>;
  resolveDrowseJlensTokenDirections(
    bindingId: string,
    layerIndices: readonly number[],
    tokenIds: readonly number[],
    modelId?: string,
  ): Promise<Float32Array>;
  getDrowseRuntimeCapabilities(
    modelId?: string,
  ): Promise<WebLlmRuntimeCapabilities>;
  getDrowseStructuredHookProfile(modelId?: string): Promise<unknown>;
  tokenizeDrowseText(text: string, modelId?: string): Promise<number[]>;
  decodeDrowseTokens(
    tokenIds: readonly number[],
    modelId?: string,
  ): Promise<string>;
}

interface WebLlmInitProgressReport {
  progress: number;
  timeElapsed: number;
  text: string;
}

export interface DrowseCaptureMessage {
  role: "system" | "user" | "assistant";
  content: string;
  roleName?: string;
}

export interface DrowseCaptureRow {
  system: string;
  messages: DrowseCaptureMessage[];
}

export interface DrowsePreparedCaptureRow {
  inputIds: number[];
  position: number;
}

export interface DrowseResidualCapture {
  layerCount: number;
  positionCount: number;
  hiddenSize: number;
  positions: number[];
  values: Float32Array;
}

export interface DrowseRankOneResidualCaptureV1 extends DrowseResidualCapture {
  abiVersion: 1;
  measurements: Float32Array;
}

function validatePreparedCaptureRequest(
  row: DrowsePreparedCaptureRow,
  positions: number[],
): void {
  if (
    !Number.isSafeInteger(row.position) || row.position < 0 ||
    row.position >= row.inputIds.length
  ) {
    throw runtimeError(
      "INVALID_CAPTURE_ROW",
      "The prepared Drowse capture row has an invalid pooling position",
    );
  }
  if (
    positions.length === 0 || positions.some((position, index) =>
      !Number.isSafeInteger(position) || position < 0 ||
      position >= row.inputIds.length ||
      (index > 0 && position <= positions[index - 1])
    )
  ) {
    throw runtimeError(
      "INVALID_CAPTURE_POSITIONS",
      "Drowse capture positions must be sorted unique token positions in the prepared row",
    );
  }
}

export interface DrowseWebLlmModule {
  DROWSE_HOOK_ABI: string;
  MLCEngine: new (config?: any) => DrowseWebLlmEngineInstance;
}

export interface CompiledDrowseStructuredHookProfile {
  schemaVersion: 2 | 3;
  id: StructuredHookProfileId;
  hookAbi: string;
  format: typeof STRUCTURED_HOOK_FORMAT | typeof STRUCTURED_HOOK_FORMAT_V2;
  layerCount: number;
  hiddenSize: number;
  maxAffineGroups: number;
  maxRank: number;
  maxProbes: number;
  maxCurves: number;
  maxCurveNodes: number;
  maxIntrinsicDim: number;
  maxEmbedDim: number;
  maxGeometryProbes: number;
  maxWhitenerRank: number;
  maxGeometryCandidates: number;
  geometryOutputStride: number;
  geometryFootRestarts: number;
  geometryFootIterations: number;
  geometryWarmRestarts: number;
  geometryWarmIterations: number;
  curveParameterStride: number;
  maxComputeWorkgroupStorageSize: number;
  requiredMaxStorageBuffersPerShaderStage: number;
  geometryKernelStorageBindings: number;
  geometryHeaderStride: number;
  geometryPayloadElements: number;
  exactReadoutAbiVersion?: 1;
  exactReadoutAbi?: typeof DROWSE_EXACT_READOUT_ABI;
  readoutTopK?: typeof DROWSE_READOUT_TOP_K;
  maxSaeFeaturesPerChunk?: typeof DROWSE_EXACT_READOUT_MAX_SAE_FEATURES_PER_CHUNK;
}

export class DrowseWebLlmRuntime {
  private engine: DrowseWebLlmEngineInstance | null = null;
  private captureSpecialTokenIds: number[] | null = null;
  private generating = false;
  private reusablePlainChatPrefix = false;
  private structuredHooks = false;
  private curvedHooks = false;
  private saeHooks = false;
  private jlensHooks = false;
  private thinkingProfile: WebLlmThinkingProfile | null = null;
  private drowseCapabilities: WebLlmRuntimeCapabilities | null = null;
  private structuredHookProfile: CompiledDrowseStructuredHookProfile | null = null;
  private maxOutputTokens = MAX_OUTPUT_TOKEN_COUNT;
  private readonly decodedTokenText = new Map<number, string>();
  private readonly pendingDecodedTokenText = new Map<number, Promise<string>>();
  private decodedTokenCacheEpoch = 0;
  private saeDictionaryIdentity: {
    runtimeLayerIndex: number;
    featureCount: number;
  } | null = null;
  private saeDictionary: DrowseSaeDictionary | null = null;
  private saeDictionaryInstalled = false;
  private jlensDictionary: DrowseJlensDictionary | null = null;
  private jlensDictionaryInstalled = false;
  private performance: Pick<
    WebLlmGenerationResult,
    "prefillTokensPerSecond" | "decodeTokensPerSecond"
  > = { prefillTokensPerSecond: null, decodeTokensPerSecond: null };

  constructor(private readonly module: DrowseWebLlmModule) {}

  async load(request: BrowserModelLoadRequest): Promise<BrowserModelLoadResult> {
    if (request.signal.aborted) throw abortReason(request.signal);
    await this.unload();
    this.maxOutputTokens = outputTokenLimitForRuntime(request.runtimeClass, request.contextTokens);
    assertExactReadoutWebGpuRequirements(
      request.variant.structuredHookProfile,
      request.variant.requirements,
    );
    const adapter = requireAdapter(request.adapter, request.variant.requirements);
    const runtimeStorageBindingLimit = (adapter.limits as Record<string, unknown>)
      .maxStorageBuffersPerShaderStage;
    if (
      typeof runtimeStorageBindingLimit !== "number" ||
      runtimeStorageBindingLimit < WEB_LLM_RUNTIME_MIN_STORAGE_BUFFERS_PER_SHADER_STAGE
    ) {
      throw runtimeError(
        "WEBGPU_LIMIT_TOO_LOW",
        `The selected GPU provides fewer than ${WEB_LLM_RUNTIME_MIN_STORAGE_BUFFERS_PER_SHADER_STAGE} ` +
          "storage buffers per shader stage required by the WebLLM runtime",
      );
    }
    const optionalPacks = request.optionalPacks ?? [];
    if (optionalPacks.some(({ pack }) => pack.kind === "jlens")) {
      assertJlensMatrixBufferSupported(
        adapter,
        request.variant.runtimeIdentity.hiddenSize,
      );
    }
    const prepared = await prepareVerifiedWebLlmArtifacts(request.artifacts, request.signal);
    if (request.model.modelType === "base" && !prepared.isBaseModel) {
      throw runtimeError("BASE_MODEL_TEMPLATE_MISMATCH", "This base-model download contains a chat template and cannot be opened as a completion model");
    }
    if (this.module.DROWSE_HOOK_ABI !== request.variant.runtimeIdentity.hookAbi) {
      throw runtimeError(
        "HOOK_ABI_MISMATCH",
        "The installed model hook ABI does not match the Drowse WebLLM runtime",
      );
    }
    let ownedEngine: DrowseWebLlmEngineInstance | null = null;
    const engine = new this.module.MLCEngine({
      initProgressCallback: (report: WebLlmInitProgressReport) => {
        request.onProgress?.({
          event: "progress",
          data: {
            kind: "model_load",
            phase: "webllm_initialization",
            progress: report.progress,
            elapsedSeconds: report.timeElapsed,
            message: report.text,
          },
        });
      },
      appConfig: {
        artifactCache: prepared.artifactCache,
        gpuAdapter: adapter,
        onDeviceLost: (info: unknown) => {
          if (this.engine === ownedEngine) {
            this.jlensDictionaryInstalled = false;
            this.saeDictionaryInstalled = false;
          }
          const detail = deviceLossDetail(info);
          request.onDeviceLost({
            code: "WEBGPU_DEVICE_LOST",
            message: detail ? `The WebGPU device was lost: ${detail}` : "The WebGPU device was lost",
            confirmedOom: false,
          });
        },
        model_list: [
          {
            model: prepared.modelUrl,
            model_id: request.variant.id,
            model_lib: prepared.modelLibraryUrl,
            overrides: { context_window_size: request.contextTokens },
            required_features: [...request.variant.requirements.features],
            buffer_size_required_bytes:
              request.variant.requirements.limits.maxStorageBufferBindingSize,
          },
        ],
      },
      logLevel: "WARN",
    });
    ownedEngine = engine;
    this.engine = engine;
    let abortCleanup: Promise<void> | null = null;
    const onAbort = (): void => {
      abortCleanup ??= Promise.resolve(engine.unload()).then(() => undefined);
    };
    request.signal.addEventListener("abort", onAbort, { once: true });
    try {
      await engine.reload(request.variant.id);
      if (request.signal.aborted) {
        await abortCleanup;
        throw abortReason(request.signal);
      }
      if (typeof engine.getDrowseStructuredHookProfile !== "function") {
        throw runtimeError(
          "STRUCTURED_HOOK_PROFILE_UNAVAILABLE",
          "The installed model library does not expose its compiled Drowse buffer profile",
        );
      }
      this.structuredHookProfile = attestCompiledStructuredHookProfile(
        await engine.getDrowseStructuredHookProfile(request.variant.id),
        request,
        this.module.DROWSE_HOOK_ABI,
      );
      const storageBindingLimit = (adapter.limits as Record<string, unknown>)
        .maxStorageBuffersPerShaderStage;
      if (
        typeof storageBindingLimit !== "number" ||
        storageBindingLimit <
          this.structuredHookProfile.requiredMaxStorageBuffersPerShaderStage
      ) {
        throw runtimeError(
          "WEBGPU_LIMIT_TOO_LOW",
          "The selected GPU cannot provide enough storage buffers for the compiled Drowse hooks",
        );
      }
      if (request.signal.aborted) {
        await abortCleanup;
        throw abortReason(request.signal);
      }
      this.drowseCapabilities = {
        ...await readWebLlmRuntimeCapabilities(engine),
        baseModel: prepared.isBaseModel,
        defaultUserRole: prepared.defaultUserRole,
        defaultAssistantRole: prepared.defaultAssistantRole,
      };
      if (request.signal.aborted) {
        await abortCleanup;
        throw abortReason(request.signal);
      }
      this.thinkingProfile = await attestBrowserThinkingProfile(
        engine,
        request,
        this.drowseCapabilities,
      );
      if (request.signal.aborted) {
        await abortCleanup;
        throw abortReason(request.signal);
      }
      const supportsHooks = await engine.supportsDrowseRankOneHooks(request.variant.id);
      if (request.signal.aborted) {
        await abortCleanup;
        throw abortReason(request.signal);
      }
      if (!supportsHooks) {
        throw runtimeError(
          "DROWSE_HOOKS_UNAVAILABLE",
          "The installed model library does not expose the required Drowse post-block hooks",
        );
      }
      this.structuredHooks = typeof engine.supportsDrowseStructuredHooks === "function" &&
        await engine.supportsDrowseStructuredHooks(request.variant.id);
      this.curvedHooks = typeof engine.supportsDrowseCurvedHooks === "function" &&
        await engine.supportsDrowseCurvedHooks(request.variant.id);
      if (
        this.structuredHooks &&
        typeof engine.readDrowseMeasurementBundle !== "function"
      ) {
        throw runtimeError(
          "BUNDLED_MEASUREMENT_READBACK_UNAVAILABLE",
          "The installed browser runtime cannot read all token instruments in one GPU synchronization",
        );
      }
      if (
        !this.structuredHooks &&
        optionalPacks.some(({ pack }) => pack.kind === "sae" || pack.kind === "jlens")
      ) {
        throw runtimeError(
          "STRUCTURED_HOOKS_UNAVAILABLE",
          "The installed browser model cannot run the selected precomputed SAE or J-lens pack",
        );
      }
      this.saeHooks = this.structuredHooks &&
        optionalPacks.some(({ pack }) => pack.kind === "sae");
      this.jlensHooks = this.structuredHooks &&
        optionalPacks.some(({ pack }) => pack.kind === "jlens");
      if (this.jlensHooks || this.saeHooks) {
        assertExactReadoutProfile(this.structuredHookProfile);
      }
      if (
        this.jlensHooks &&
        (typeof engine.setDrowseJlensDictionary !== "function" ||
          typeof engine.clearDrowseJlensDictionary !== "function" ||
          typeof engine.resolveDrowseJlensTokenDirections !== "function" ||
          typeof engine.readDrowseJlensTopTokens !== "function")
      ) {
        throw runtimeError(
          "EXACT_JLENS_READOUT_UNAVAILABLE",
          "The installed browser runtime does not expose exact full-vocabulary J-lens readout",
        );
      }
      if (
        this.saeHooks &&
        (typeof engine.setDrowseSaeDictionary !== "function" ||
          typeof engine.clearDrowseSaeDictionary !== "function" ||
          typeof engine.readDrowseSaeTopFeatures !== "function")
      ) {
        throw runtimeError(
          "EXACT_SAE_READOUT_UNAVAILABLE",
          "The installed browser runtime does not expose exact full-dictionary SAE readout",
        );
      }
      const supportsCapture = await engine.supportsDrowseResidualCapture(
        request.variant.id,
      );
      if (request.signal.aborted) {
        await abortCleanup;
        throw abortReason(request.signal);
      }
      if (!supportsCapture) {
        throw runtimeError(
          "DROWSE_CAPTURE_UNAVAILABLE",
          "The installed model library does not expose selective Drowse residual capture",
        );
      }
      this.captureSpecialTokenIds = [...prepared.captureSpecialTokenIds];
      return { prefillTokensPerSecond: null, decodeTokensPerSecond: null };
    } catch (error) {
      this.captureSpecialTokenIds = null;
      this.structuredHooks = false;
      this.curvedHooks = false;
      this.saeHooks = false;
      this.jlensHooks = false;
      this.thinkingProfile = null;
      this.drowseCapabilities = null;
      this.structuredHookProfile = null;
      this.maxOutputTokens = MAX_OUTPUT_TOKEN_COUNT;
      this.resetDecodedTokenCache();
      this.saeDictionaryIdentity = null;
      this.saeDictionary = null;
      this.saeDictionaryInstalled = false;
      this.jlensDictionary = null;
      this.jlensDictionaryInstalled = false;
      try {
        await (abortCleanup ?? Promise.resolve(engine.unload()));
      } catch (cleanupError) {
        throw runtimeCleanupError(error, cleanupError);
      }
      if (this.engine === engine) this.engine = null;
      if (request.signal.aborted) throw abortReason(request.signal);
      throw error;
    } finally {
      request.signal.removeEventListener("abort", onAbort);
    }
  }

  async unload(): Promise<void> {
    const engine = this.engine;
    if (engine) await engine.unload();
    if (this.engine !== engine) return;
    this.engine = null;
    this.reusablePlainChatPrefix = false;
    this.captureSpecialTokenIds = null;
    this.structuredHooks = false;
    this.curvedHooks = false;
    this.saeHooks = false;
    this.jlensHooks = false;
    this.thinkingProfile = null;
    this.drowseCapabilities = null;
    this.structuredHookProfile = null;
    this.maxOutputTokens = MAX_OUTPUT_TOKEN_COUNT;
    this.resetDecodedTokenCache();
    this.saeDictionaryIdentity = null;
    this.saeDictionary = null;
    this.saeDictionaryInstalled = false;
    this.jlensDictionary = null;
    this.jlensDictionaryInstalled = false;
    this.performance = emptyGenerationPerformance();
  }

  async stop(): Promise<void> {
    await this.engine?.interruptGenerate();
  }

  async streamGeneration(
    plan: WebLlmGenerationPlan,
    onToken: (token: WebLlmGeneratedToken) => void | Promise<void>,
  ): Promise<WebLlmGenerationResult> {
    if (this.generating) {
      throw runtimeError(
        "GENERATION_IN_PROGRESS",
        "The loaded browser model is already generating",
      );
    }
    const engine = this.requireEngine();
    this.performance = emptyGenerationPerformance();
    if (this.runtimeCapabilities().baseModel && plan.input.kind !== "raw") {
      throw runtimeError("BASE_MODEL_RAW_INPUT_REQUIRED", "This base model requires a plain text completion prompt");
    }
    const plainChat = plan.input.kind === "chat" && !plan.hookProgram &&
      !plan.steeringExpression?.trim() && !plan.replay && plan.thinking == null &&
      !plan.generationRoleName && (!plan.generationSeat || plan.generationSeat === "assistant");
    const reusePrefix = plainChat && this.reusablePlainChatPrefix;
    this.reusablePlainChatPrefix = false;
    this.generating = true;
    try {
      if (plan.input.kind === "chat" && !reusePrefix) await engine.resetChat();
      const result = await streamWebLlmGeneration(
        engine,
        {
          clear: () => this.clearHookProgram(),
          interrupt: () => this.stop(),
          install: (program) => this.installHookProgram(program),
          updateControls: (affineActive, curveActive) =>
            this.updateStructuredControls(affineActive, curveActive),
          readBundle: (topK) => this.readMeasurementBundle(topK),
          read: () => this.readMeasurements(),
          readGeometry: () => this.readGeometryMeasurements(),
          readJlensTopTokens: (topK) => this.readJlensTopTokens(topK),
          readSaeTopFeatures: (topK) => this.readSaeTopFeatures(topK),
          assertSteeringSupported: (expression) => this.assertSteeringSupported(expression),
        },
        {
          ...plan,
          maxOutputTokens: Math.min(
            plan.maxOutputTokens ?? this.maxOutputTokens,
            this.maxOutputTokens,
          ),
          thinkingProfile: this.thinkingProfile,
          ...(this.captureSpecialTokenIds === null
            ? { measurementSpecialTokenIds: undefined }
            : { measurementSpecialTokenIds: [...this.captureSpecialTokenIds] }),
        },
        onToken,
      );
      this.reusablePlainChatPrefix = plainChat && !plan.signal?.aborted;
      this.performance = {
        prefillTokensPerSecond: result.prefillTokensPerSecond,
        decodeTokensPerSecond: result.decodeTokensPerSecond,
      };
      return result;
    } finally {
      this.generating = false;
    }
  }

  generationPerformance(): Readonly<typeof this.performance> {
    return { ...this.performance };
  }

  runtimeCapabilities(): WebLlmRuntimeCapabilities {
    this.requireEngine();
    if (this.drowseCapabilities === null) {
      throw runtimeError(
        "DROWSE_CAPABILITIES_UNAVAILABLE",
        "The loaded browser model has no Drowse capability record",
      );
    }
    return { ...this.drowseCapabilities };
  }

  thinkingProfileForSession(): WebLlmThinkingProfile | null {
    this.requireEngine();
    return this.thinkingProfile === null
      ? null
      : {
          ...this.thinkingProfile,
          startTokenIds: [...this.thinkingProfile.startTokenIds],
          endTokenIds: [...this.thinkingProfile.endTokenIds],
        };
  }

  async tokenizeText(text: string): Promise<number[]> {
    const engine = this.requireEngine();
    if (!this.runtimeCapabilities().tokenizer) {
      throw runtimeError(
        "DROWSE_TOKENIZER_UNAVAILABLE",
        "The loaded browser model does not expose its tokenizer",
      );
    }
    return tokenizeWebLlmText(engine, text);
  }

  async decodeTokens(tokenIds: readonly number[]): Promise<string> {
    const engine = this.requireEngine();
    if (!this.runtimeCapabilities().tokenizer) {
      throw runtimeError(
        "DROWSE_TOKENIZER_UNAVAILABLE",
        "The loaded browser model does not expose its tokenizer",
      );
    }
    return engine.decodeDrowseTokens(tokenIds);
  }

  hookCapabilities(): DrowseHookCapabilities {
    this.requireEngine();
    return {
      ...AFFINE_FEASIBILITY_CAPABILITIES,
      curved_manifold: this.curvedHooks,
      phase_trigger: this.structuredHooks,
      probe_gate: this.structuredHooks,
      sae: this.saeHooks,
      jlens: this.jlensHooks,
    };
  }

  assertSteeringSupported(expression: string): void {
    assertHookFeaturesSupported(expression, this.hookCapabilities());
  }

  async installRankOneHookProgram(program: RankOneHookProgramBuffers): Promise<void> {
    if (program.hookAbi !== this.module.DROWSE_HOOK_ABI) {
      throw runtimeError(
        "HOOK_ABI_MISMATCH",
        "The steering program does not match the loaded Drowse hook ABI",
      );
    }
    await this.requireEngine().setDrowseRankOneProgram(program);
  }

  async installHookProgram(
    program: RankOneHookProgramBuffers | StructuredHookProgramBuffers,
  ): Promise<void> {
    if ("affineActive" in program) {
      if (this.structuredHookProfile === null) {
        throw runtimeError(
          "STRUCTURED_HOOK_PROFILE_UNAVAILABLE",
          "The loaded model library has no attested Drowse buffer profile",
        );
      }
      if (program.format !== this.structuredHookProfile.format) {
        throw runtimeError(
          "HOOK_FORMAT_MISMATCH",
          "The structured steering program format does not match the loaded model library",
        );
      }
      if (program.hookAbi !== this.module.DROWSE_HOOK_ABI) {
        throw runtimeError(
          "HOOK_ABI_MISMATCH",
          "The steering program does not match the loaded Drowse hook ABI",
        );
      }
      const hasCurve = program.curveRank.some((rank) => rank > 0);
      if (hasCurve && !this.curvedHooks) {
        throw runtimeError(
          "CURVED_HOOKS_UNAVAILABLE",
          "The loaded model library does not expose curved Drowse hooks",
        );
      }
      if (!hasCurve && !this.structuredHooks) {
        throw runtimeError(
          "STRUCTURED_HOOKS_UNAVAILABLE",
          "The loaded model library does not expose structured Drowse hooks",
        );
      }
      if (program.jLensBindingId !== undefined) {
        await this.ensureJlensDictionaryResident(program.jLensBindingId);
      }
      if (program.saeBindingId !== undefined) {
        await this.ensureSaeDictionaryResident(program.saeBindingId);
      }
      await this.requireEngine().setDrowseStructuredProgram(program);
      return;
    }
    await this.installRankOneHookProgram(program);
  }

  async updateStructuredControls(
    affineActive: Uint32Array,
    curveActive: Uint32Array,
  ): Promise<void> {
    await this.requireEngine().updateDrowseStructuredControls(
      affineActive,
      curveActive,
    );
  }

  async clearHookProgram(): Promise<void> {
    await this.requireEngine().clearDrowseRankOneProgram();
  }

  async setSaeDictionary(dictionary: BrowserSaeGpuDictionary | null): Promise<void> {
    const engine = this.requireEngine();
    if (dictionary === null) {
      if (
        this.saeDictionaryInstalled &&
        typeof engine.clearDrowseSaeDictionary === "function"
      ) {
        await engine.clearDrowseSaeDictionary();
      }
      this.saeDictionaryIdentity = null;
      this.saeDictionary = null;
      this.saeDictionaryInstalled = false;
      return;
    }
    if (typeof engine.setDrowseSaeDictionary !== "function") {
      throw runtimeError(
        "EXACT_SAE_READOUT_UNAVAILABLE",
        "The loaded browser runtime cannot install a precomputed SAE dictionary",
      );
    }
    const resident: DrowseSaeDictionary = {
      hookAbi: this.module.DROWSE_HOOK_ABI,
      bindingId: dictionary.bindingId,
      hiddenSize: dictionary.hiddenSize,
      runtimeLayerIndex: dictionary.runtimeLayerIndex,
      featureCount: dictionary.featureCount,
      activation: dictionary.activation,
      encoder: dictionary.encoder,
      encoderBias: dictionary.encoderBias,
      encoderThreshold: dictionary.encoderThreshold,
      decoderBias: dictionary.decoderBias,
    };
    if (this.saeDictionary?.bindingId === resident.bindingId) {
      assertMatchingSaeDictionary(this.saeDictionary, resident);
      this.saeDictionary = resident;
      return;
    }
    if (this.saeDictionaryInstalled) await engine.clearDrowseSaeDictionary();
    this.saeDictionary = resident;
    this.saeDictionaryInstalled = false;
    this.saeDictionaryIdentity = {
      runtimeLayerIndex: dictionary.runtimeLayerIndex,
      featureCount: dictionary.featureCount,
    };
  }

  async setJlensDictionary(dictionary: BrowserJlensGpuDictionary | null): Promise<void> {
    const engine = this.requireEngine();
    if (dictionary === null) {
      if (
        this.jlensDictionaryInstalled &&
        typeof engine.clearDrowseJlensDictionary === "function"
      ) {
        await engine.clearDrowseJlensDictionary();
      }
      this.jlensDictionary = null;
      this.jlensDictionaryInstalled = false;
      return;
    }
    if (typeof engine.setDrowseJlensDictionary !== "function") {
      throw runtimeError(
        "EXACT_JLENS_READOUT_UNAVAILABLE",
        "The loaded browser runtime cannot install a precomputed J-lens dictionary",
      );
    }
    const resident: DrowseJlensDictionary = {
      hookAbi: this.module.DROWSE_HOOK_ABI,
      bindingId: dictionary.bindingId,
      hiddenSize: dictionary.hiddenSize,
      layerIndices: dictionary.layerIndices,
      matrices: dictionary.matrices,
    };
    if (this.jlensDictionary?.bindingId === resident.bindingId) {
      assertMatchingJlensDictionary(this.jlensDictionary, resident);
      this.jlensDictionary = resident;
      return;
    }
    if (this.jlensDictionaryInstalled) await engine.clearDrowseJlensDictionary();
    this.jlensDictionary = resident;
    this.jlensDictionaryInstalled = false;
  }

  async warmJlensDictionary(): Promise<void> {
    const bindingId = this.jlensDictionary?.bindingId;
    if (bindingId === undefined) return;
    await this.ensureJlensDictionaryResident(bindingId);
  }

  async warmSaeDictionary(): Promise<void> {
    const bindingId = this.saeDictionary?.bindingId;
    if (bindingId === undefined) return;
    await this.ensureSaeDictionaryResident(bindingId);
  }

  private async ensureSaeDictionaryResident(bindingId: string): Promise<void> {
    if (this.saeDictionary?.bindingId !== bindingId) {
      throw runtimeError(
        "SAE_DICTIONARY_BINDING_MISMATCH",
        "The structured program does not match the configured SAE dictionary",
      );
    }
    if (this.saeDictionaryInstalled) return;
    await this.requireEngine().setDrowseSaeDictionary(this.saeDictionary);
    this.saeDictionaryInstalled = true;
  }

  private async ensureJlensDictionaryResident(bindingId: string): Promise<void> {
    if (this.jlensDictionary?.bindingId !== bindingId) {
      throw runtimeError(
        "JLENS_DICTIONARY_BINDING_MISMATCH",
        "The structured program does not match the configured J-lens dictionary",
      );
    }
    if (this.jlensDictionaryInstalled) return;
    await this.requireEngine().setDrowseJlensDictionary(this.jlensDictionary);
    this.jlensDictionaryInstalled = true;
  }

  async readJlensTopTokens(
    topK = DROWSE_READOUT_TOP_K,
  ): Promise<WebLlmJlensTopTokenReadout | undefined> {
    assertExactReadoutTopK(topK);
    const read = this.requireEngine().readDrowseJlensTopTokens;
    if (typeof read !== "function") return undefined;
    const value = await read.call(this.requireEngine());
    if (value === undefined) return undefined;
    return this.formatJlensTopTokens(value, topK);
  }

  private async formatJlensTopTokens(
    value: DrowseJlensTopTokenReadout,
    topK: number,
  ): Promise<WebLlmJlensTopTokenReadout> {
    validateJlensTopTokenReadout(value);
    const tokenIds = value.tokenIds.slice(0, topK);
    const layerTokenIds = firstColumns(
      value.layerTokenIds,
      value.fittedLayerCount,
      DROWSE_READOUT_TOP_K,
      topK,
    );
    const layerProbabilities = firstColumns(
      value.layerProbabilities,
      value.fittedLayerCount,
      DROWSE_READOUT_TOP_K,
      topK,
    );
    const tokens = await Promise.all([...tokenIds].map((tokenId) =>
      this.decodeSingleToken(tokenId)
    ));
    const layerTokens = await Promise.all([...layerTokenIds].map((tokenId) =>
      this.decodeSingleToken(tokenId)
    ));
    return {
      tokenIds,
      tokens,
      strength: value.strength.slice(0, topK),
      centerOfMass: value.centerOfMass.slice(0, topK),
      spread: value.spread.slice(0, topK),
      fittedLayerCount: value.fittedLayerCount,
      layerIndices: new Int32Array(value.layerIndices),
      layerTokenIds,
      layerTokens,
      layerProbabilities,
    };
  }

  async readSaeTopFeatures(
    topK = DROWSE_READOUT_TOP_K,
  ): Promise<WebLlmSaeTopFeatureReadout | undefined> {
    assertExactReadoutTopK(topK);
    const read = this.requireEngine().readDrowseSaeTopFeatures;
    if (typeof read !== "function") return undefined;
    const value = await read.call(this.requireEngine());
    if (value === undefined) return undefined;
    return this.formatSaeTopFeatures(value, topK);
  }

  private formatSaeTopFeatures(
    value: DrowseSaeTopFeatureReadout,
    topK: number,
  ): WebLlmSaeTopFeatureReadout {
    validateSaeTopFeatureReadout(
      value,
      this.saeDictionaryIdentity,
      this.structuredHookProfile?.layerCount ?? null,
    );
    return {
      featureIds: value.featureIds.slice(0, topK),
      activations: value.activations.slice(0, topK),
      runtimeLayerIndex: value.runtimeLayerIndex,
      featureCount: value.featureCount,
    };
  }

  async readMeasurementBundle(topK = DROWSE_READOUT_TOP_K): Promise<{
    scalar: Float32Array | undefined;
    geometry: Float32Array | undefined;
    jlensTopTokens: WebLlmJlensTopTokenReadout | undefined;
    saeTopFeatures: WebLlmSaeTopFeatureReadout | undefined;
  }> {
    assertExactReadoutTopK(topK);
    const value = await this.requireEngine().readDrowseMeasurementBundle();
    const [jlensTopTokens, saeTopFeatures] = await Promise.all([
      value.jlensTopTokens === undefined
        ? undefined
        : this.formatJlensTopTokens(value.jlensTopTokens, topK),
      value.saeTopFeatures === undefined
        ? undefined
        : this.formatSaeTopFeatures(value.saeTopFeatures, topK),
    ]);
    return {
      scalar: value.scalar === undefined
        ? undefined
        : new Float32Array(value.scalar),
      geometry: value.geometry === undefined
        ? undefined
        : new Float32Array(value.geometry),
      jlensTopTokens,
      saeTopFeatures,
    };
  }

  async readMeasurements(): Promise<Float32Array | undefined> {
    const values = await this.requireEngine().readDrowseMeasurements();
    return values === undefined ? undefined : new Float32Array(values);
  }

  async readGeometryMeasurements(): Promise<Float32Array | undefined> {
    const values = await this.requireEngine().readDrowseGeometryMeasurements();
    return values === undefined ? undefined : new Float32Array(values);
  }

  async resolveJlensTokenDirections(
    bindingId: string,
    layerIndices: readonly number[],
    tokenIds: readonly number[],
  ): Promise<Float32Array> {
    await this.ensureJlensDictionaryResident(bindingId);
    return new Float32Array(
      await this.requireEngine().resolveDrowseJlensTokenDirections(
        bindingId,
        layerIndices,
        tokenIds,
      ),
    );
  }

  async prepareCaptureRows(
    rows: DrowseCaptureRow[],
  ): Promise<DrowsePreparedCaptureRow[]> {
    const specialTokenIds = this.captureSpecialTokenIds;
    if (specialTokenIds === null) {
      throw runtimeError("MODEL_NOT_LOADED", "No browser model runtime is loaded");
    }
    return this.requireEngine().prepareDrowseCaptureRows(rows, specialTokenIds);
  }

  async capturePreparedRow(
    row: DrowsePreparedCaptureRow,
  ): Promise<DrowseResidualCapture> {
    return this.capturePreparedPositions(row, [row.position]);
  }

  async capturePreparedPositions(
    row: DrowsePreparedCaptureRow,
    positions: number[],
  ): Promise<DrowseResidualCapture> {
    if (
      !Number.isSafeInteger(row.position) || row.position < 0 ||
      row.position >= row.inputIds.length
    ) {
      throw runtimeError(
        "INVALID_CAPTURE_ROW",
        "The prepared Drowse capture row has an invalid pooling position",
      );
    }
    if (
      positions.length === 0 || positions.some((position, index) =>
        !Number.isSafeInteger(position) || position < 0 ||
        position >= row.inputIds.length ||
        (index > 0 && position <= positions[index - 1])
      )
    ) {
      throw runtimeError(
        "INVALID_CAPTURE_POSITIONS",
        "Drowse capture positions must be sorted unique token positions in the prepared row",
      );
    }
    this.reusablePlainChatPrefix = false;
    const capture = await this.requireEngine().captureDrowseResiduals(
      row.inputIds,
      positions,
    );
    if (
      capture.positionCount !== positions.length ||
      capture.positions.length !== positions.length ||
      capture.positions.some((position, index) => position !== positions[index]) ||
      capture.values.length !== capture.layerCount * positions.length * capture.hiddenSize
    ) {
      throw runtimeError(
        "INVALID_CAPTURE_RESULT",
        "The Drowse WebLLM runtime returned a malformed residual capture",
      );
    }
    return {
      ...capture,
      positions: [...capture.positions],
      values: new Float32Array(capture.values),
    };
  }

  async capturePreparedRankOnePositionsV1(
    row: DrowsePreparedCaptureRow,
    positions: number[],
    program: RankOneHookProgramBuffers,
  ): Promise<DrowseRankOneResidualCaptureV1> {
    validatePreparedCaptureRequest(row, positions);
    const engine = this.requireEngine();
    if (
      typeof engine.supportsDrowseRankOneResidualCaptureV1 !== "function" ||
      !await engine.supportsDrowseRankOneResidualCaptureV1()
    ) {
      throw runtimeError(
        "RANK_ONE_CAPTURE_UNAVAILABLE",
        "The loaded model library does not expose the Drowse rank-one residual capture v1 ABI",
      );
    }
    this.reusablePlainChatPrefix = false;
    const capture = await engine.captureDrowseRankOneResidualsV1(
      row.inputIds,
      positions,
      program,
    );
    if (
      capture.abiVersion !== 1 ||
      capture.positionCount !== positions.length ||
      capture.positions.length !== positions.length ||
      capture.positions.some((position, index) => position !== positions[index]) ||
      capture.values.length !== capture.layerCount * positions.length * capture.hiddenSize ||
      capture.measurements.length !== capture.layerCount
    ) {
      throw runtimeError(
        "INVALID_RANK_ONE_CAPTURE_RESULT",
        "The Drowse WebLLM runtime returned a malformed rank-one residual capture",
      );
    }
    return {
      ...capture,
      positions: [...capture.positions],
      values: new Float32Array(capture.values),
      measurements: new Float32Array(capture.measurements),
    };
  }

  requireEngine(): DrowseWebLlmEngineInstance {
    if (!this.engine) {
      throw runtimeError("MODEL_NOT_LOADED", "No browser model runtime is loaded");
    }
    return this.engine;
  }

  private async decodeSingleToken(tokenId: number): Promise<string> {
    const cached = this.decodedTokenText.get(tokenId);
    if (cached !== undefined) {
      this.decodedTokenText.delete(tokenId);
      this.decodedTokenText.set(tokenId, cached);
      return cached;
    }
    const pending = this.pendingDecodedTokenText.get(tokenId);
    if (pending !== undefined) return pending;
    const epoch = this.decodedTokenCacheEpoch;
    const decode = this.decodeUncachedToken(tokenId, epoch);
    this.pendingDecodedTokenText.set(tokenId, decode);
    return decode;
  }

  private async decodeUncachedToken(tokenId: number, epoch: number): Promise<string> {
    try {
      const decoded = await this.requireEngine().decodeDrowseTokens([tokenId]);
      if (this.decodedTokenCacheEpoch === epoch) {
        while (this.decodedTokenText.size >= WEB_LLM_DECODED_TOKEN_CACHE_MAX) {
          const oldest = this.decodedTokenText.keys().next().value;
          if (oldest === undefined) break;
          this.decodedTokenText.delete(oldest);
        }
        this.decodedTokenText.set(tokenId, decoded);
      }
      return decoded;
    } finally {
      if (this.decodedTokenCacheEpoch === epoch) {
        this.pendingDecodedTokenText.delete(tokenId);
      }
    }
  }

  private resetDecodedTokenCache(): void {
    this.decodedTokenCacheEpoch += 1;
    this.decodedTokenText.clear();
    this.pendingDecodedTokenText.clear();
  }
}

function validateJlensTopTokenReadout(value: DrowseJlensTopTokenReadout): void {
  if (
    !(value.tokenIds instanceof Int32Array) ||
    !(value.strength instanceof Float32Array) ||
    !(value.centerOfMass instanceof Float32Array) ||
    !(value.spread instanceof Float32Array) ||
    value.tokenIds.length !== DROWSE_READOUT_TOP_K ||
    value.strength.length !== DROWSE_READOUT_TOP_K ||
    value.centerOfMass.length !== DROWSE_READOUT_TOP_K ||
    value.spread.length !== DROWSE_READOUT_TOP_K ||
    !Number.isSafeInteger(value.fittedLayerCount) || value.fittedLayerCount < 1 ||
    !(value.layerIndices instanceof Int32Array) ||
    !(value.layerTokenIds instanceof Int32Array) ||
    !(value.layerProbabilities instanceof Float32Array) ||
    value.layerIndices.length !== value.fittedLayerCount ||
    value.layerTokenIds.length !== value.fittedLayerCount * DROWSE_READOUT_TOP_K ||
    value.layerProbabilities.length !== value.fittedLayerCount * DROWSE_READOUT_TOP_K ||
    value.layerIndices.some((layer, index) =>
      layer < 0 || index > 0 && layer <= value.layerIndices[index - 1]
    ) ||
    value.tokenIds.some((tokenId) => tokenId < 0) ||
    value.layerTokenIds.some((tokenId) => tokenId < 0) ||
    value.strength.some((strength) => !Number.isFinite(strength) || strength < 0 || strength > 1) ||
    value.layerProbabilities.some((probability) =>
      !Number.isFinite(probability) || probability < 0 || probability > 1
    ) ||
    value.centerOfMass.some((center) => !Number.isFinite(center) || center < 0 || center > 1) ||
    value.spread.some((spread) => !Number.isFinite(spread) || spread < 0 || spread > 1)
  ) {
    throw runtimeError(
      "INVALID_JLENS_TOP_READOUT",
      "The browser model returned an invalid full-vocabulary J-lens readout",
    );
  }
}

function assertExactReadoutTopK(topK: number): void {
  if (!Number.isSafeInteger(topK) || topK < 1) {
    throw runtimeError(
      "INVALID_EXACT_READOUT_TOP_K",
      "The exact browser readout width must be a positive integer",
    );
  }
  if (topK > DROWSE_READOUT_TOP_K) {
    throw runtimeError(
      "EXACT_READOUT_TOP_K_EXCEEDS_CAPACITY",
      `Requested exact browser readout top-${topK}, but the verified GPU ABI supports at most top-${DROWSE_READOUT_TOP_K}`,
    );
  }
}

function firstColumns(
  value: Int32Array,
  rows: number,
  sourceColumns: number,
  selectedColumns: number,
): Int32Array;
function firstColumns(
  value: Float32Array,
  rows: number,
  sourceColumns: number,
  selectedColumns: number,
): Float32Array;
function firstColumns(
  value: Int32Array | Float32Array,
  rows: number,
  sourceColumns: number,
  selectedColumns: number,
): Int32Array | Float32Array {
  const output = value instanceof Int32Array
    ? new Int32Array(rows * selectedColumns)
    : new Float32Array(rows * selectedColumns);
  for (let row = 0; row < rows; row += 1) {
    output.set(
      value.subarray(
        row * sourceColumns,
        row * sourceColumns + selectedColumns,
      ),
      row * selectedColumns,
    );
  }
  return output;
}

function validateSaeTopFeatureReadout(
  value: DrowseSaeTopFeatureReadout,
  expected: { runtimeLayerIndex: number; featureCount: number } | null,
  runtimeLayerCount: number | null,
): void {
  if (
    expected === null || runtimeLayerCount === null ||
    !(value.featureIds instanceof Int32Array) ||
    !(value.activations instanceof Float32Array) ||
    value.featureIds.length !== value.activations.length ||
    value.featureIds.length > DROWSE_READOUT_TOP_K ||
    !Number.isSafeInteger(value.runtimeLayerIndex) || value.runtimeLayerIndex < 0 ||
    value.runtimeLayerIndex >= runtimeLayerCount ||
    value.runtimeLayerIndex !== expected.runtimeLayerIndex ||
    !Number.isSafeInteger(value.featureCount) || value.featureCount < 1 ||
    value.featureCount !== expected.featureCount ||
    value.featureIds.some((featureId) => featureId < 0 || featureId >= value.featureCount) ||
    value.activations.some((activation) => !Number.isFinite(activation) || activation < 0)
  ) {
    throw runtimeError(
      "INVALID_SAE_TOP_READOUT",
      "The browser model returned an invalid full-dictionary SAE readout",
    );
  }
}

export function browserThinkingProfile(
  request: Pick<BrowserModelLoadRequest, "variant">,
): WebLlmThinkingProfile | null {
  const profile = request.variant.thinkingProfile;
  return profile === null
      ? null
      : {
          start: profile.start,
          end: profile.end,
          startTokenIds: [...profile.startTokenIds],
          endTokenIds: [...profile.endTokenIds],
          startsInThinking: profile.startsInThinking,
        };
}

export async function attestBrowserThinkingProfile(
  engine: WebLlmGenerationEngine,
  request: Pick<BrowserModelLoadRequest, "variant">,
  capabilities: WebLlmRuntimeCapabilities,
): Promise<WebLlmThinkingProfile | null> {
  const declared = request.variant.thinkingProfile;
  if (declared === null) return null;
  if (!capabilities.tokenizer) {
    throw runtimeError(
      "THINKING_PROFILE_UNATTESTED",
      "The selected model declares thinking delimiters but its verified tokenizer is unavailable",
    );
  }
  const startTokenIds = await tokenizeWebLlmText(engine, declared.start);
  const endTokenIds = await tokenizeWebLlmText(engine, declared.end);
  if (
    !sameTokenIds(startTokenIds, declared.startTokenIds) ||
    !sameTokenIds(endTokenIds, declared.endTokenIds)
  ) {
    throw runtimeError(
      "THINKING_PROFILE_MISMATCH",
      "The loaded tokenizer does not match the model's signed thinking-delimiter profile",
    );
  }
  return Object.freeze({
    start: declared.start,
    end: declared.end,
    startTokenIds: Object.freeze([...declared.startTokenIds]),
    endTokenIds: Object.freeze([...declared.endTokenIds]),
    startsInThinking: declared.startsInThinking,
  });
}

function sameTokenIds(actual: readonly number[], expected: readonly number[]): boolean {
  return actual.length === expected.length &&
    actual.every((tokenId, index) => tokenId === expected[index]);
}

export function attestCompiledStructuredHookProfile(
  value: unknown,
  request: Pick<BrowserModelLoadRequest, "variant">,
  moduleHookAbi: string,
): CompiledDrowseStructuredHookProfile {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw runtimeError(
      "STRUCTURED_HOOK_PROFILE_MISMATCH",
      "The compiled model returned an invalid Drowse buffer profile",
    );
  }
  const record = value as Record<string, unknown>;
  const profile = resolveStructuredHookProfile(request.variant.structuredHookProfile);
  const expected: CompiledDrowseStructuredHookProfile = {
    schemaVersion: profile.id === "standard-v3" ? 3 : 2,
    id: profile.id,
    hookAbi: moduleHookAbi,
    format: profile.id === "standard-v1"
      ? STRUCTURED_HOOK_FORMAT_V2
      : STRUCTURED_HOOK_FORMAT,
    layerCount: request.variant.runtimeIdentity.layerMap.length,
    hiddenSize: request.variant.runtimeIdentity.hiddenSize,
    maxAffineGroups: profile.maxAffineGroups,
    maxRank: profile.maxRank,
    maxProbes: profile.maxProbes,
    maxCurves: profile.maxCurves,
    maxCurveNodes: profile.maxCurveNodes,
    maxIntrinsicDim: profile.maxIntrinsicDim,
    maxEmbedDim: profile.maxEmbedDim,
    maxGeometryProbes: profile.maxGeometryProbes,
    maxWhitenerRank: profile.maxWhitenerRank,
    maxGeometryCandidates: profile.maxGeometryCandidates,
    geometryOutputStride: profile.geometryOutputStride,
    geometryFootRestarts: profile.geometryFootRestarts,
    geometryFootIterations: profile.geometryFootIterations,
    geometryWarmRestarts: profile.geometryWarmRestarts,
    geometryWarmIterations: profile.geometryWarmIterations,
    curveParameterStride: structuredCurveParameterStride(profile),
    maxComputeWorkgroupStorageSize: profile.maxComputeWorkgroupStorageSize,
    requiredMaxStorageBuffersPerShaderStage:
      profile.requiredMaxStorageBuffersPerShaderStage,
    geometryKernelStorageBindings: profile.geometryKernelStorageBindings,
    geometryHeaderStride: profile.geometryHeaderStride,
    geometryPayloadElements: structuredGeometryPayloadLayout(
      profile,
      request.variant.runtimeIdentity.layerMap.length,
      request.variant.runtimeIdentity.hiddenSize,
    ).elements,
    ...(profile.id === "standard-v3"
      ? {
          exactReadoutAbiVersion: DROWSE_EXACT_READOUT_ABI_VERSION,
          exactReadoutAbi: DROWSE_EXACT_READOUT_ABI,
          readoutTopK: DROWSE_READOUT_TOP_K,
          maxSaeFeaturesPerChunk: DROWSE_EXACT_READOUT_MAX_SAE_FEATURES_PER_CHUNK,
        } as const
      : {}),
  };
  const expectedKeys = Object.keys(expected).sort();
  const actualKeys = Object.keys(record).sort();
  if (
    actualKeys.length !== expectedKeys.length ||
    actualKeys.some((key, index) => key !== expectedKeys[index])
  ) {
    throw runtimeError(
      "STRUCTURED_HOOK_PROFILE_MISMATCH",
      "The compiled model returned an unsupported Drowse buffer profile shape",
    );
  }
  for (const key of Object.keys(expected) as (keyof CompiledDrowseStructuredHookProfile)[]) {
    if (record[key] !== expected[key]) {
      throw runtimeError(
        "STRUCTURED_HOOK_PROFILE_MISMATCH",
        `The compiled model Drowse buffer profile differs at ${key}`,
      );
    }
  }
  return Object.freeze({ ...expected });
}

function assertExactReadoutProfile(
  profile: CompiledDrowseStructuredHookProfile,
): void {
  if (
    profile.schemaVersion !== 3 ||
    profile.id !== "standard-v3" ||
    profile.exactReadoutAbiVersion !== DROWSE_EXACT_READOUT_ABI_VERSION ||
    profile.exactReadoutAbi !== DROWSE_EXACT_READOUT_ABI ||
    profile.readoutTopK !== DROWSE_READOUT_TOP_K ||
    profile.maxSaeFeaturesPerChunk !==
      DROWSE_EXACT_READOUT_MAX_SAE_FEATURES_PER_CHUNK
  ) {
    throw runtimeError(
      "EXACT_READOUT_ABI_MISMATCH",
      "The compiled model does not attest the exact GPU readout ABI required by precomputed J-lens and SAE packs",
    );
  }
}

function assertExactReadoutWebGpuRequirements(
  structuredHookProfile: StructuredHookProfileId,
  requirements: ModelVariantRequirements,
): void {
  if (structuredHookProfile !== "standard-v3") return;
  for (const [name, minimum] of Object.entries(
    EXACT_READOUT_REQUIRED_WEBGPU_LIMITS,
  )) {
    const declared = requirements.limits[name];
    if (
      typeof declared !== "number" ||
      !Number.isSafeInteger(declared) ||
      declared < minimum
    ) {
      throw runtimeError(
        "EXACT_READOUT_WEBGPU_REQUIREMENTS_INVALID",
        `The signed model requirements must declare ${name} of at least ${minimum} ` +
          "for exact GPU readout",
      );
    }
  }
}

function requireAdapter(
  value: unknown,
  requirements: ModelVariantRequirements,
): GpuAdapterLike {
  if (
    typeof value !== "object" || value === null ||
    typeof (value as { requestDevice?: unknown }).requestDevice !== "function" ||
    !isIterable((value as { features?: unknown }).features) ||
    typeof (value as { limits?: unknown }).limits !== "object" ||
    (value as { limits?: unknown }).limits === null
  ) {
    throw runtimeError(
      "VERIFIED_GPU_ADAPTER_REQUIRED",
      "The compatibility gate did not retain a usable WebGPU adapter",
    );
  }
  const adapter = value as GpuAdapterLike;
  if (adapter.isFallbackAdapter === true || adapter.info?.isFallbackAdapter === true) {
    throw runtimeError(
      "SOFTWARE_ADAPTER",
      "The retained WebGPU adapter is a software fallback",
    );
  }
  const availableFeatures = new Set(Array.from(adapter.features, String));
  const missingFeatures = requirements.features.filter(
    (feature) => !availableFeatures.has(feature),
  );
  if (missingFeatures.length > 0) {
    throw runtimeError(
      "WEBGPU_FEATURE_UNAVAILABLE",
      `The retained WebGPU adapter is missing ${missingFeatures.join(", ")}`,
    );
  }
  const limits = adapter.limits as Record<string, unknown>;
  for (const [name, minimum] of Object.entries(requirements.limits)) {
    const available = limits[name];
    if (typeof available !== "number" || !Number.isFinite(available)) {
      throw runtimeError(
        "WEBGPU_LIMIT_UNAVAILABLE",
        `The retained WebGPU adapter did not report ${name}`,
      );
    }
    if (available < minimum) {
      throw runtimeError(
        "WEBGPU_LIMIT_TOO_LOW",
        `The retained WebGPU adapter reports ${name}=${available}, below the required ${minimum}`,
      );
    }
  }
  return adapter;
}

function assertJlensMatrixBufferSupported(
  adapter: GpuAdapterLike,
  hiddenSize: number,
): void {
  const bytes = hiddenSize * hiddenSize * Float32Array.BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(bytes)) {
    throw runtimeError(
      "JLENS_GPU_BUFFER_TOO_LARGE",
      "The selected model's J-lens matrix exceeds safe browser buffer arithmetic",
    );
  }
  const limits = adapter.limits as Record<string, unknown>;
  for (const name of ["maxBufferSize", "maxStorageBufferBindingSize"]) {
    const available = limits[name];
    if (
      typeof available !== "number" ||
      !Number.isFinite(available) ||
      available < bytes
    ) {
      throw runtimeError(
        "JLENS_GPU_BUFFER_TOO_LARGE",
        `The selected model needs a ${bytes}-byte J-lens matrix buffer, but WebGPU ${name} is insufficient`,
      );
    }
  }
}

function assertMatchingSaeDictionary(
  current: DrowseSaeDictionary,
  next: DrowseSaeDictionary,
): void {
  if (
    current.hiddenSize !== next.hiddenSize ||
    current.runtimeLayerIndex !== next.runtimeLayerIndex ||
    current.featureCount !== next.featureCount ||
    current.activation !== next.activation ||
    current.encoder.length !== next.encoder.length ||
    current.encoderBias.length !== next.encoderBias.length ||
    current.encoderThreshold.length !== next.encoderThreshold.length ||
    current.decoderBias.length !== next.decoderBias.length
  ) {
    throw runtimeError(
      "SAE_DICTIONARY_BINDING_COLLISION",
      "The same SAE binding identity was configured with a different tensor shape",
    );
  }
}

function assertMatchingJlensDictionary(
  current: DrowseJlensDictionary,
  next: DrowseJlensDictionary,
): void {
  if (
    current.hiddenSize !== next.hiddenSize ||
    current.layerIndices.length !== next.layerIndices.length ||
    current.matrices.length !== next.matrices.length ||
    current.layerIndices.some((value, index) => value !== next.layerIndices[index]) ||
    current.matrices.some((matrix, index) =>
      matrix.length !== next.matrices[index]?.length
    )
  ) {
    throw runtimeError(
      "JLENS_DICTIONARY_BINDING_COLLISION",
      "The same J-lens binding identity was configured with a different tensor shape",
    );
  }
}

function isIterable(value: unknown): value is Iterable<unknown> {
  return typeof value === "object" && value !== null &&
    typeof (value as { [Symbol.iterator]?: unknown })[Symbol.iterator] === "function";
}

function deviceLossDetail(value: unknown): string {
  if (typeof value !== "object" || value === null) return "";
  const record = value as Record<string, unknown>;
  if (typeof record.message === "string" && record.message.trim()) return record.message.trim();
  if (typeof record.reason === "string" && record.reason.trim()) return record.reason.trim();
  return "";
}

function abortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException("Model loading was cancelled", "AbortError");
}

function emptyGenerationPerformance(): Pick<
  WebLlmGenerationResult,
  "prefillTokensPerSecond" | "decodeTokensPerSecond"
> {
  return { prefillTokensPerSecond: null, decodeTokensPerSecond: null };
}

function runtimeError(code: string, message: string): Error & { code: string } {
  return Object.assign(new Error(message), { code });
}

function runtimeCleanupError(loadError: unknown, cleanupError: unknown): Error & {
  code: string;
  cause: unknown;
} {
  return Object.assign(
    new Error(
      `Model loading failed (${errorMessage(loadError)}) and WebLLM cleanup failed (${errorMessage(cleanupError)})`,
    ),
    { code: "RUNTIME_CLEANUP_FAILED", cause: cleanupError },
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
