import type {
  DepthSummaryJSON,
  LensReadoutBlockJSON,
  MeasurementsEnvelopeJSON,
  ProbeReadingJSON,
  SaeFeatureJSON,
  ScalarReadingJSON,
  TokenAltJSON,
  WSSampling,
} from "../../lib/types";
import type { RankOneHookProgramBuffers } from "./rankOneHookProgram";
import { BROWSER_RETURN_TOP_K_MAX, BROWSER_SAMPLING_TOP_K_MAX } from "../../lib/runtime/samplingCapabilities";
import { MAX_OUTPUT_TOKEN_COUNT } from "../../lib/runtime/outputTokenPolicy";
import {
  geometryCoordinateMean,
  createStructuredHookControlEvaluator,
  type StructuredGeometryMeasurementProbe,
  type StructuredHookProgramBuffers,
} from "./structuredHookProgram";
import { isRoleSlug } from "./roleSlug";

export const BROWSER_EXACT_READOUT_TOP_K_CAPACITY = 8;

export type WebLlmHookProgram = RankOneHookProgramBuffers | StructuredHookProgramBuffers;

export type WebLlmChatRole = "system" | "user" | "assistant";

export type WebLlmGenerationInput =
  | {
      kind: "chat";
      messages: readonly {
        role: WebLlmChatRole;
        content: string;
        name?: string;
      }[];
    }
  | { kind: "raw"; prompt: string };

export interface WebLlmGenerationPlan {
  input: WebLlmGenerationInput;
  sampling?: WSSampling | null;
  thinking?: boolean | null;
  thinkingProfile?: WebLlmThinkingProfile | null;
  generationSeat?: "user" | "assistant" | null;
  generationRoleName?: string | null;
  replay?: WebLlmReplayRequest | null;
  readoutTopK?: number;
  steeringExpression?: string | null;
  hookProgram?: WebLlmHookProgram | null;
  measurementSpecialTokenIds?: readonly number[];
  measurementTargetRawIndex?: number;
  measurementStartRawIndex?: number;
  maxOutputTokens?: number;
  onRawTokenStart?: (rawIndex: number) => void | Promise<void>;
  onRawToken?: (token: WebLlmRawToken) => void | Promise<void>;
  signal?: AbortSignal;
}

export interface WebLlmRawToken {
  text: string;
  tokenId: number | null;
  logprob: number;
  samplerEntropy: number;
  perplexity: number;
  rawIndex: number;
  topAlts: TokenAltJSON[] | null;
  replayScore?: WebLlmReplayScore;
  measurements?: MeasurementsEnvelopeJSON;
  hookMeasurements?: Float32Array;
}

export interface WebLlmGeneratedToken {
  text: string;
  thinking: boolean;
  tokenId: number | null;
  logprob: number | null;
  samplerEntropy: number | null;
  perplexity: number | null;
  rawIndex: number | null;
  topAlts?: TokenAltJSON[] | null;
  replayScore?: WebLlmReplayScore;
  measurements?: MeasurementsEnvelopeJSON;
  hookMeasurements?: Float32Array;
}

export interface WebLlmGenerationUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export interface WebLlmReplayRequest {
  forcedPrefixTokenIds?: readonly number[] | null;
  scoreTokenIds?: readonly number[] | null;
}

export interface WebLlmSamplerLogprob {
  tokenId: number;
  logprob: number;
}

export interface WebLlmReplayScore {
  emittedTokenId: number;
  sampledTokenId: number;
  forcedTokenId: number | null;
  requestedLogprobs: WebLlmSamplerLogprob[];
  argmax: WebLlmSamplerLogprob;
  topLogprobs: WebLlmSamplerLogprob[];
}

export interface WebLlmRuntimeCapabilities {
  topK: boolean;
  forcedReplay: boolean;
  replayScoring: boolean;
  tokenizer: boolean;
  namedRoles: boolean;
  userSeatGeneration: boolean;
  sceneStitching: boolean;
  /** Added by Drowse after validating the installed tokenizer configuration. */
  baseModel?: boolean;
  /** Visible labels emitted by the installed model's verified chat template. */
  defaultUserRole?: string | null;
  defaultAssistantRole?: string | null;
}

export interface WebLlmGenerationResult {
  text: string;
  thinkingText: string | null;
  tokens: number;
  finishReason: string;
  terminalReason: WebLlmTerminalReason;
  usage: WebLlmGenerationUsage;
  meanLogprob: number | null;
  meanSurprise: number | null;
  prefillTokensPerSecond: number | null;
  decodeTokensPerSecond: number | null;
  measurements?: MeasurementsEnvelopeJSON;
}

export interface WebLlmThinkingProfile {
  start: string;
  end: string;
  startTokenIds: readonly number[];
  endTokenIds: readonly number[];
  startsInThinking: boolean;
}

export type WebLlmTerminalReason =
  | "eos"
  | "stop_sequence"
  | "external_stop"
  | "length";

export interface WebLlmJlensTopTokenReadout {
  tokenIds: Int32Array;
  tokens: string[];
  strength: Float32Array;
  centerOfMass: Float32Array;
  spread: Float32Array;
  fittedLayerCount: number;
  layerIndices: Int32Array;
  layerTokenIds: Int32Array;
  layerTokens: string[];
  layerProbabilities: Float32Array;
}

export interface WebLlmSaeTopFeatureReadout {
  featureIds: Int32Array;
  activations: Float32Array;
  runtimeLayerIndex: number;
  featureCount: number;
}

export interface WebLlmGenerationEngine {
  chat: {
    completions: {
      create(request: any): Promise<any>;
    };
  };
  completions: {
    create(request: any): Promise<any>;
  };
  getDrowseRuntimeCapabilities(modelId?: string): Promise<WebLlmRuntimeCapabilities>;
  tokenizeDrowseText(text: string, modelId?: string): Promise<number[]>;
  decodeDrowseTokens(tokenIds: readonly number[], modelId?: string): Promise<string>;
}

export interface WebLlmGenerationHooks {
  clear(): Promise<void>;
  interrupt(): Promise<void>;
  install(program: WebLlmHookProgram): Promise<void>;
  updateControls?(affineActive: Uint32Array, curveActive: Uint32Array): Promise<void>;
  readBundle?(
    topK: number,
  ): Promise<WebLlmHookMeasurementBuffers>;
  read(): Promise<Float32Array | undefined>;
  readGeometry?(): Promise<Float32Array | undefined>;
  readJlensTopTokens?(topK: number): Promise<WebLlmJlensTopTokenReadout | undefined>;
  readSaeTopFeatures?(topK: number): Promise<WebLlmSaeTopFeatureReadout | undefined>;
  assertSteeringSupported(expression: string): void;
}

export interface WebLlmHookMeasurementBuffers {
  scalar: Float32Array | undefined;
  geometry: Float32Array | undefined;
  jlensTopTokens: WebLlmJlensTopTokenReadout | undefined;
  saeTopFeatures: WebLlmSaeTopFeatureReadout | undefined;
}

interface CapturedGenerationRow {
  text: string;
  tokenId: number | null;
  logprob: number;
  samplerEntropy: number;
  perplexity: number;
  rawIndex: number;
  topAlts: TokenAltJSON[] | null;
  replayScore?: WebLlmReplayScore;
  measurements?: MeasurementsEnvelopeJSON;
  hookMeasurements?: Float32Array;
}

interface ClassifiedGenerationRow extends CapturedGenerationRow {
  thinking: boolean;
}

const generationErrorUsage = new WeakMap<object, WebLlmGenerationUsage>();

export function webLlmGenerationErrorUsage(
  error: unknown,
): WebLlmGenerationUsage | null {
  if ((typeof error !== "object" || error === null) && typeof error !== "function") {
    return null;
  }
  const usage = generationErrorUsage.get(error);
  return usage === undefined ? null : { ...usage };
}

export async function streamWebLlmGeneration(
  engine: WebLlmGenerationEngine,
  hooks: WebLlmGenerationHooks,
  plan: WebLlmGenerationPlan,
  onToken: (token: WebLlmGeneratedToken) => void | Promise<void>,
): Promise<WebLlmGenerationResult> {
  if (plan.hookProgram && isStructuredProgram(plan.hookProgram)) {
    plan = { ...plan, hookProgram: {
      ...plan.hookProgram,
      affineActive: new Uint32Array(plan.hookProgram.affineActive),
      curveActive: new Uint32Array(plan.hookProgram.curveActive),
    } };
  }
  const request = buildRequest(plan);
  const readoutTopK = exactReadoutTopK(plan, plan.sampling ?? {});
  if (
    plan.measurementTargetRawIndex !== undefined &&
    (!Number.isInteger(plan.measurementTargetRawIndex) || plan.measurementTargetRawIndex < 0)
  ) {
    throw generationError(
      "INVALID_MEASUREMENT_TARGET",
      "The browser measurement target must be a non-negative raw token index",
    );
  }
  if (plan.measurementStartRawIndex !== undefined && (
    !Number.isSafeInteger(plan.measurementStartRawIndex) ||
    plan.measurementStartRawIndex < 0 ||
    plan.measurementStartRawIndex > (plan.replay?.forcedPrefixTokenIds?.length ?? 0)
  )) {
    throw generationError("INVALID_MEASUREMENT_START", "The saved response prefix is invalid.");
  }
  await assertWebLlmGenerationCapabilities(engine, plan);
  if (plan.thinking && (plan.input.kind !== "chat" || plan.thinkingProfile == null)) {
    throw generationError(
      "THINKING_STREAM_UNAVAILABLE",
      "The selected browser model does not expose verified thinking delimiters",
    );
  }
  const thinkingProfile = plan.thinking ? plan.thinkingProfile ?? null : null;
  const thinking = new ThinkingStreamState(thinkingProfile);
  const stopBuffer = new ResponseStopBuffer(plan.sampling?.stop ?? []);
  const steering = plan.steeringExpression?.trim() ?? "";
  if (steering) {
    hooks.assertSteeringSupported(steering);
    if (!plan.hookProgram) {
      throw generationError(
        "STEERING_PROGRAM_REQUIRED",
        "Browser steering requires a compiled hook program for this generation",
      );
    }
  }
  throwIfAborted(plan.signal);
  const controlEvaluator = plan.hookProgram && isStructuredProgram(plan.hookProgram)
    ? createStructuredHookControlEvaluator(plan.hookProgram)
    : null;
  const perTokenControlUpdates = requiresPerTokenControlUpdates(plan.hookProgram);
  const needsGateReadings = hasProbeGates(plan.hookProgram);
  let observedUsage: WebLlmGenerationUsage | null = null;

  await hooks.clear();
  try {
    if (plan.hookProgram) {
      if (isStructuredProgram(plan.hookProgram)) {
        const controls = controlEvaluator!({
          prefill: true,
          thinking: thinking.active,
          generatedTokens: 0,
        });
        plan.hookProgram.affineActive.set(controls.affineActive);
        plan.hookProgram.curveActive.set(controls.curveActive);
      }
      await hooks.install(plan.hookProgram);
    }
    throwIfAborted(plan.signal);
    const stream = plan.input.kind === "chat"
      ? await engine.chat.completions.create(request)
      : await engine.completions.create(request);
    if (!isAsyncIterable(stream)) {
      throw generationError(
        "INVALID_GENERATION_STREAM",
        "The browser model runtime did not return a streaming response",
      );
    }

    const text: string[] = [];
    const thinkingText: string[] = [];
    const responseLogprobRows: ResponseLogprobRow[] = [];
    let tokenCount = 0;
    let finishReason: string | null = null;
    let runtimeTerminalReason: WebLlmTerminalReason | null = null;
    let localStopRequested = false;
    let prefillTokensPerSecond: number | null = null;
    let decodeTokensPerSecond: number | null = null;
    const aggregateMeasurements = new AggregateMeasurementSelector(
      plan.sampling?.return_probe_readings !== false
        ? plan.measurementSpecialTokenIds
        : null,
    );
    let processingError: unknown = null;
    for await (const value of stream) {
      try {
        const chunk = requireRecord(value, "generation chunk");
        const chunkUsage = readUsage(chunk.usage);
        if (chunkUsage) {
          observedUsage = mergeObservedUsage(observedUsage, chunkUsage.usage);
          prefillTokensPerSecond = chunkUsage.prefillTokensPerSecond;
          decodeTokensPerSecond = chunkUsage.decodeTokensPerSecond;
        }
        if (processingError !== null) continue;
        throwIfAborted(plan.signal);
        const choices = requireArray(chunk.choices, "generation chunk choices");
        if (choices.length === 0) {
          if (!chunkUsage) {
            throw generationError(
              "INVALID_GENERATION_CHUNK",
              "A generation chunk without choices must contain usage",
            );
          }
          continue;
        }
        if (choices.length !== 1) {
          throw generationError(
            "MULTIPLE_GENERATION_CHOICES_UNSUPPORTED",
            "The browser runtime returned multiple generation choices",
          );
        }
        const choice = requireRecord(choices[0], "generation choice");
        if (choice.index !== 0) {
          throw generationError(
            "INVALID_GENERATION_CHOICE",
            "The browser runtime returned an out-of-order generation choice",
          );
        }
        const terminalSeenBeforeChunk = runtimeTerminalReason !== null;
        if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
          const nextFinishReason = readFinishReason(choice.finish_reason);
          if (finishReason !== null && finishReason !== nextFinishReason) {
            throw generationError(
              "INCONSISTENT_GENERATION_TERMINATION",
              "The browser model runtime returned conflicting finish reasons",
            );
          }
          finishReason = nextFinishReason;
        }
        if (
          choice.drowse_finish_reason !== null &&
          choice.drowse_finish_reason !== undefined
        ) {
          const nextTerminalReason = readTerminalReason(
            choice.drowse_finish_reason,
          );
          if (
            runtimeTerminalReason !== null &&
            runtimeTerminalReason !== nextTerminalReason
          ) {
            throw generationError(
              "INCONSISTENT_GENERATION_TERMINATION",
              "The browser model runtime returned conflicting exact terminal reasons",
            );
          }
          if (
            choice.finish_reason !== null &&
            choice.finish_reason !== undefined &&
            !terminalReasonMatchesFinishReason(
              nextTerminalReason,
              readFinishReason(choice.finish_reason),
            )
          ) {
            throw generationError(
              "INCONSISTENT_GENERATION_TERMINATION",
              "The browser model runtime returned incompatible terminal metadata",
            );
          }
          runtimeTerminalReason = nextTerminalReason;
        }
        const content = plan.input.kind === "chat"
          ? readChatContent(choice)
          : readRawContent(choice);
        const rows = readLogprobRows(choice.logprobs);
        if (rows !== null) {
          if (terminalSeenBeforeChunk && rows.length > 0) {
            throw generationError(
              "TOKEN_AFTER_GENERATION_TERMINATION",
              "The browser model runtime emitted a token after generation terminated",
            );
          }
          if (rows.length > 1) {
            if (perTokenControlUpdates) {
              throw generationError(
                "BATCHED_TOKEN_CONTROLS_UNAVAILABLE",
                "The model runtime batched multiple streamed tokens, so phase and probe gates cannot be updated at the required token boundary",
              );
            }
            if (requiresPerTokenMeasurementReads(plan.hookProgram)) {
              throw generationError(
                "BATCHED_TOKEN_MEASUREMENTS_UNAVAILABLE",
                "The model runtime batched multiple streamed tokens, so an exact measurement cannot be matched to each token",
              );
            }
          }
          const rowText = rows.map((row) => row.token).join("");
          if (content !== "" && content !== rowText) {
            throw generationError(
              "GENERATION_TOKEN_TEXT_MISMATCH",
              "The browser runtime's token rows do not match its streamed text " +
                `(stream ${generationTextDiagnostic(content)}; rows ${generationTextDiagnostic(rowText)})`,
            );
          }
          if (runtimeTerminalReason === "eos" && rows.length > 0) {
            if (rows.length !== 1 || content !== "" || rowText !== "") {
              throw generationError(
                "EOS_TOKEN_SURFACED",
                "The browser model runtime surfaced text for its terminal EOS token",
              );
            }
            continue;
          }
          if (localStopRequested && rows.length > 0) {
            throw generationError(
              "TOKEN_AFTER_STOP_SEQUENCE",
              "The browser model runtime emitted a token after Drowse matched a stop sequence",
            );
          }
          for (const row of rows) {
            throwIfAborted(plan.signal);
            const rawIndex = tokenCount;
            if (plan.onRawTokenStart) await plan.onRawTokenStart(rawIndex);
            const preservingPrefix = rawIndex < (plan.measurementStartRawIndex ?? 0);
            const measurement = await readHookMeasurements(
              preservingPrefix && !needsGateReadings ? null : plan.hookProgram,
              hooks,
              readoutTopK,
              !preservingPrefix && (plan.measurementTargetRawIndex === undefined ||
                rawIndex === plan.measurementTargetRawIndex),
            );
            throwIfAborted(plan.signal);
            tokenCount += 1;
            const capturedEnvelope = measurementEnvelope(
              plan.hookProgram,
              measurement.scalar,
              plan.steeringExpression ?? null,
              measurement.geometry,
              measurement.jlensTopTokens,
              measurement.saeTopFeatures,
              readoutTopK,
              plan.sampling?.persist_subspace_coords === true,
            );
            const gateScores = capturedEnvelope === undefined
              ? undefined
              : measurementGateScores(capturedEnvelope);
            const envelope = persistedMeasurementEnvelope(capturedEnvelope, plan.sampling);
            const captured: CapturedGenerationRow = {
              text: row.token,
              tokenId: row.tokenId,
              logprob: row.logprob,
              samplerEntropy: row.samplerEntropy,
              perplexity: row.perplexity,
              rawIndex,
              topAlts: row.topAlts ?? null,
              ...(row.replayScore === undefined
                ? {}
                : { replayScore: row.replayScore }),
              ...(envelope === undefined ? {} : { measurements: envelope }),
              ...(measurement.scalar === undefined
                ? {}
                : { hookMeasurements: new Float32Array(measurement.scalar) }),
            };
            if (plan.onRawToken) {
              await plan.onRawToken(captured);
            }
            const classified = thinking.consume(captured);
            for (const piece of classified.flatMap((item) =>
              stopBuffer.consume(item)
            )) {
              throwIfAborted(plan.signal);
              const token = emittedToken(
                piece.text,
                piece.thinking,
                piece.tokenId,
                piece.logprob,
                piece.samplerEntropy,
                piece.perplexity,
                piece.rawIndex,
                (plan.sampling?.return_top_k ?? 0) > 0
                  ? piece.topAlts?.slice(0, plan.sampling?.return_top_k ?? 0) ?? null
                  : null,
                piece.measurements,
                piece.hookMeasurements,
                piece.replayScore,
              );
              (token.thinking ? thinkingText : text).push(token.text);
              if (token.text !== "") {
                if (!token.thinking) {
                  responseLogprobRows.push({
                    text: token.text,
                    tokenId: token.tokenId,
                    logprob: piece.logprob,
                  });
                }
                aggregateMeasurements.observe(token.tokenId, token.measurements);
                await onToken(token);
              }
            }
            if (stopBuffer.matched && !localStopRequested) {
              localStopRequested = true;
              await hooks.interrupt();
            }
            if (perTokenControlUpdates) {
              await updateStructuredControls(
                plan.hookProgram,
                hooks,
                controlEvaluator!,
                tokenCount,
                thinking.active,
                measurement.scalar,
                gateScores,
              );
            }
          }
        } else if (content !== "") {
          throw generationError(
            "RAW_TOKEN_BOUNDARY_UNAVAILABLE",
            "The browser model runtime omitted exact token and sampler metadata",
          );
        }
      } catch (error) {
        processingError ??= error;
        await hooks.interrupt();
      }
    }
    if (processingError !== null) throw processingError;
    const trailing = thinking.flush().flatMap((item) => stopBuffer.consume(item));
    const terminalReason = resolveTerminalReason(
      runtimeTerminalReason,
      finishReason,
      stopBuffer.matched,
    );
    const visibleTrailing = stopBuffer.flush(terminalReason);
    for (const piece of [...trailing, ...visibleTrailing]) {
      const token = emittedToken(
        piece.text,
        piece.thinking,
        piece.tokenId,
        piece.logprob,
        piece.samplerEntropy,
        piece.perplexity,
        piece.rawIndex,
        (plan.sampling?.return_top_k ?? 0) > 0
          ? piece.topAlts?.slice(0, plan.sampling?.return_top_k ?? 0) ?? null
          : null,
        piece.measurements,
        piece.hookMeasurements,
        piece.replayScore,
      );
      (token.thinking ? thinkingText : text).push(token.text);
      if (token.text !== "") {
        if (!token.thinking) {
          responseLogprobRows.push({
            text: token.text,
            tokenId: token.tokenId,
            logprob: piece.logprob,
          });
        }
        aggregateMeasurements.observe(token.tokenId, token.measurements);
        await onToken(token);
      }
    }

    const completedUsage = requireCompletedUsage(observedUsage, tokenCount);

    const logprobs = responseContentLogprobs(
      responseLogprobRows,
      plan.measurementSpecialTokenIds,
    );
    const meanLogprob = logprobs.length === 0 || logprobs.some((value) => !Number.isFinite(value))
      ? null
      : logprobs.reduce((sum, value) => sum + value, 0) / logprobs.length;
    return {
      text: text.join(""),
      thinkingText: thinkingText.length === 0 ? null : thinkingText.join(""),
      tokens: tokenCount,
      finishReason: legacyFinishReason(terminalReason),
      terminalReason,
      usage: completedUsage,
      meanLogprob,
      meanSurprise: meanLogprob === null ? null : -meanLogprob,
      prefillTokensPerSecond,
      decodeTokensPerSecond,
      ...(aggregateMeasurements.value
        ? { measurements: { ...aggregateMeasurements.value, scope: "aggregate" as const } }
        : {}),
    };
  } catch (error) {
    rememberGenerationErrorUsage(error, observedUsage);
    throw error;
  } finally {
    await hooks.clear();
  }
}

export async function readWebLlmRuntimeCapabilities(
  engine: WebLlmGenerationEngine,
): Promise<WebLlmRuntimeCapabilities> {
  if (typeof engine.getDrowseRuntimeCapabilities !== "function") {
    throw generationError(
      "DROWSE_CAPABILITIES_UNAVAILABLE",
      "The browser model runtime does not expose explicit Drowse capabilities",
    );
  }
  const capabilities = await engine.getDrowseRuntimeCapabilities();
  for (const key of [
    "topK",
    "forcedReplay",
    "replayScoring",
    "tokenizer",
    "namedRoles",
    "userSeatGeneration",
  ] as const) {
    if (typeof capabilities?.[key] !== "boolean") {
      throw generationError(
        "INVALID_DROWSE_CAPABILITIES",
        "The browser model runtime returned invalid Drowse capabilities",
      );
    }
  }
  if (
    capabilities.sceneStitching !== undefined &&
    typeof capabilities.sceneStitching !== "boolean"
  ) {
    throw generationError(
      "INVALID_DROWSE_CAPABILITIES",
      "The browser model runtime returned invalid Drowse capabilities",
    );
  }
  return {
    ...capabilities,
    sceneStitching: capabilities.sceneStitching === true,
  };
}

export async function tokenizeWebLlmText(
  engine: WebLlmGenerationEngine,
  text: string,
): Promise<number[]> {
  if (typeof text !== "string") {
    throw generationError(
      "INVALID_TOKENIZER_INPUT",
      "Browser tokenization requires a string",
    );
  }
  const capabilities = await readWebLlmRuntimeCapabilities(engine);
  if (!capabilities.tokenizer) {
    throw generationError(
      "DROWSE_TOKENIZER_UNAVAILABLE",
      "The loaded browser model does not expose its tokenizer",
    );
  }
  const tokenIds = await engine.tokenizeDrowseText(text);
  validateTokenIds(tokenIds, "tokenizer result");
  return [...tokenIds];
}

async function assertWebLlmGenerationCapabilities(
  engine: WebLlmGenerationEngine,
  plan: WebLlmGenerationPlan,
): Promise<void> {
  const needsTopK = (plan.sampling?.top_k ?? 0) > 0;
  const needsForcedReplay = plan.replay?.forcedPrefixTokenIds != null;
  const needsReplayScoring = plan.replay?.scoreTokenIds != null;
  const needsNamedRoles = plan.generationRoleName != null ||
    plan.input.kind === "chat" &&
      plan.input.messages.some((message) => message.name != null);
  const needsUserSeat = plan.generationSeat === "user";
  const needsSceneStitching = requiresSceneStitching(plan);
  if (
    !needsTopK &&
    !needsForcedReplay &&
    !needsReplayScoring &&
    !needsNamedRoles &&
    !needsUserSeat &&
    !needsSceneStitching
  ) {
    return;
  }
  const capabilities = await readWebLlmRuntimeCapabilities(engine);
  const required: Array<{
    needed: boolean;
    supported: boolean;
    code: string;
    message: string;
  }> = [
    {
      needed: needsTopK,
      supported: capabilities.topK,
      code: "TOP_K_SAMPLING_UNAVAILABLE",
      message: "The loaded browser model runtime does not support exact top-k sampling",
    },
    {
      needed: needsForcedReplay,
      supported: capabilities.forcedReplay,
      code: "FORCED_REPLAY_UNAVAILABLE",
      message: "The loaded browser model runtime does not support forced token replay",
    },
    {
      needed: needsReplayScoring,
      supported: capabilities.replayScoring,
      code: "REPLAY_SCORING_UNAVAILABLE",
      message: "The loaded browser model runtime does not support compact replay scoring",
    },
    {
      needed: needsNamedRoles,
      supported: capabilities.namedRoles,
      code: "NAMED_ROLE_UNAVAILABLE",
      message: "The loaded conversation template does not support structural named roles",
    },
    {
      needed: needsUserSeat,
      supported: capabilities.userSeatGeneration,
      code: "USER_SEAT_GENERATION_UNAVAILABLE",
      message: "The loaded conversation template does not support user-seat generation",
    },
    {
      needed: needsSceneStitching,
      supported: capabilities.sceneStitching,
      code: "SCENE_STITCHING_UNAVAILABLE",
      message: "The loaded conversation template cannot render this participant sequence exactly",
    },
  ];
  const missing = required.find((item) => item.needed && !item.supported);
  if (missing) throw generationError(missing.code, missing.message);
}

function buildRequest(plan: WebLlmGenerationPlan): Record<string, unknown> {
  validateInput(plan.input);
  validateGenerationRole(plan);
  validateGenerationSeat(plan);
  validateReplay(plan.replay);
  const sampling = applyOutputTokenLimit(
    plan,
    validateWebLlmGenerationSettings(plan.sampling, plan.thinking),
  );
  validateMeasurementPlan(plan, sampling);
  const request: Record<string, unknown> = {
    stream: true,
    stream_options: { include_usage: true },
    logprobs: true,
    top_logprobs: Math.max(1, sampling.return_top_k ?? 0),
  };
  copySampling(request, sampling);
  const replayExtraBody = {
    ...replayRequestExtraBody(plan.replay),
    ...(plan.measurementStartRawIndex === undefined ? {} : { drowse_readout_start_index: plan.measurementStartRawIndex }),
    ...(plan.measurementTargetRawIndex === undefined ? {} : { drowse_readout_target_index: plan.measurementTargetRawIndex }),
  };
  if (plan.input.kind === "chat") {
    request.messages = plan.input.messages.map((message) => ({ ...message }));
    const extraBody = {
      ...replayExtraBody,
      ...(plan.generationSeat === null || plan.generationSeat === undefined
        ? {}
        : { drowse_generation_seat: plan.generationSeat }),
      ...(plan.thinkingProfile == null || plan.thinking === null || plan.thinking === undefined
        ? {}
        : { enable_thinking: plan.thinking }),
      ...(plan.generationRoleName === null || plan.generationRoleName === undefined
        ? {}
        : { drowse_generation_role: plan.generationRoleName }),
    };
    if (Object.keys(extraBody).length > 0) {
      request.extra_body = extraBody;
    }
  } else {
    request.prompt = plan.input.prompt;
    if (Object.keys(replayExtraBody).length > 0) {
      request.extra_body = replayExtraBody;
    }
  }
  return request;
}

function replayRequestExtraBody(
  replay: WebLlmReplayRequest | null | undefined,
): Record<string, unknown> {
  return {
    ...(replay?.forcedPrefixTokenIds === null ||
    replay?.forcedPrefixTokenIds === undefined
      ? {}
      : {
          drowse_forced_prefix_token_ids: [
            ...replay.forcedPrefixTokenIds,
          ],
        }),
    ...(replay?.scoreTokenIds === null || replay?.scoreTokenIds === undefined
      ? {}
      : { drowse_score_token_ids: [...replay.scoreTokenIds] }),
  };
}

export function validateWebLlmGenerationSettings(
  value: WSSampling | null | undefined,
  thinking: boolean | null | undefined,
): WSSampling {
  const sampling = validateSampling(value);
  void thinking;
  return sampling;
}

function applyOutputTokenLimit(
  plan: WebLlmGenerationPlan,
  sampling: WSSampling,
): WSSampling {
  const limit = plan.maxOutputTokens;
  if (limit === undefined) return sampling;
  if (
    !Number.isSafeInteger(limit) || limit < 1 ||
    limit > MAX_OUTPUT_TOKEN_COUNT
  ) {
    throw generationError(
      "INVALID_OUTPUT_TOKEN_LIMIT",
      `The browser output token limit must be between 1 and ${MAX_OUTPUT_TOKEN_COUNT}`,
    );
  }
  const forcedTokens = plan.replay?.forcedPrefixTokenIds?.length ?? 0;
  const maximumCompletionTokens = Math.min(Number.MAX_SAFE_INTEGER, forcedTokens + limit);
  return {
    ...sampling,
    max_tokens: Math.min(
      sampling.max_tokens ?? maximumCompletionTokens,
      maximumCompletionTokens,
    ),
  };
}

function validateInput(input: WebLlmGenerationInput): void {
  if (input.kind === "chat") {
    if (!Array.isArray(input.messages) || input.messages.length === 0) {
      throw generationError("INVALID_GENERATION_INPUT", "Chat generation requires messages");
    }
    for (const message of input.messages) {
      if (
        !message ||
        !["system", "user", "assistant"].includes(message.role) ||
        typeof message.content !== "string" ||
        message.name !== undefined &&
          (message.role === "system" || !isRoleSlug(message.name))
      ) {
        throw generationError("INVALID_GENERATION_INPUT", "Invalid chat message");
      }
    }
    return;
  }
  if (input.kind !== "raw" || typeof input.prompt !== "string") {
    throw generationError("INVALID_GENERATION_INPUT", "Invalid raw generation prompt");
  }
}

function validateGenerationRole(plan: WebLlmGenerationPlan): void {
  const role = plan.generationRoleName;
  if (role === null || role === undefined) return;
  if (plan.input.kind !== "chat") {
    throw generationError(
      "GENERATION_ROLE_UNAVAILABLE",
      "A custom generated role is available only for chat generation",
    );
  }
  if (!isRoleSlug(role)) {
    throw generationError(
      "INVALID_GENERATION_ROLE",
      "The generated role must be a lowercase role slug",
    );
  }
}

function validateGenerationSeat(plan: WebLlmGenerationPlan): void {
  const seat = plan.generationSeat;
  if (seat === null || seat === undefined) return;
  if (plan.input.kind !== "chat") {
    throw generationError(
      "GENERATION_SEAT_UNAVAILABLE",
      "A generated seat is available only for chat generation",
    );
  }
  if (seat !== "user" && seat !== "assistant") {
    throw generationError(
      "INVALID_GENERATION_SEAT",
      "The generated seat must be user or assistant",
    );
  }
}

function requiresSceneStitching(plan: WebLlmGenerationPlan): boolean {
  if (plan.input.kind !== "chat") return false;
  if ((plan.generationSeat ?? "assistant") === "user") return true;
  let expected: "user" | "assistant" = "user";
  let turns = 0;
  for (let index = 0; index < plan.input.messages.length; index += 1) {
    const role = plan.input.messages[index].role;
    if (role === "system") {
      if (index !== 0) return true;
      continue;
    }
    turns += 1;
    if (role !== expected) return true;
    expected = expected === "user" ? "assistant" : "user";
  }
  return turns === 0 || expected !== "assistant";
}

function validateReplay(replay: WebLlmReplayRequest | null | undefined): void {
  if (replay === null || replay === undefined) return;
  validateTokenIds(replay.forcedPrefixTokenIds, "forcedPrefixTokenIds");
  validateTokenIds(replay.scoreTokenIds, "scoreTokenIds");
  if (
    replay.forcedPrefixTokenIds == null &&
    replay.scoreTokenIds == null
  ) {
    throw generationError(
      "INVALID_REPLAY_REQUEST",
      "Replay requires forced-prefix token IDs or score token IDs",
    );
  }
}

function validateMeasurementPlan(
  plan: WebLlmGenerationPlan,
  sampling: WSSampling,
): void {
  exactReadoutTopK(plan, sampling);
  const specialTokenIds = plan.measurementSpecialTokenIds;
  if (
    specialTokenIds !== undefined &&
    (!Array.isArray(specialTokenIds) || specialTokenIds.some((tokenId) =>
      !Number.isSafeInteger(tokenId) || tokenId < 0
    ) || new Set(specialTokenIds).size !== specialTokenIds.length)
  ) {
    throw generationError(
      "INVALID_MEASUREMENT_SPECIAL_TOKEN_IDS",
      "Measurement special-token IDs must be a unique list of nonnegative integers",
    );
  }
  const program = plan.hookProgram;
  if (!program || !isStructuredProgram(program) || program.measurementSchema === undefined) {
    return;
  }
  const schema = program.measurementSchema;
  const scalarProbes = schema.probes
    .map((probe, probeIndex) => ({ probe, probeIndex }))
    .filter((entry) => entry.probe !== null);
  for (const { probe, probeIndex } of scalarProbes) {
    const active = Array.from({ length: program.layerCount }, (_, layerIndex) =>
      program.probeKind[layerIndex * program.profile.maxProbes + probeIndex]
    ).some((kind) => kind !== 0);
    if (!active) {
      throw generationError(
        "INERT_MEASUREMENT_PROBE",
        `Browser probe ${probe!.name} has no active GPU measurement slot`,
      );
    }
  }
  const geometryProbes = schema.geometryProbes ?? [];
  for (let probeIndex = 0; probeIndex < geometryProbes.length; probeIndex += 1) {
    const probe = geometryProbes[probeIndex];
    if (probe === null) continue;
    const active = program.geometryActive !== undefined &&
      Array.from({ length: program.layerCount }, (_, layerIndex) =>
        program.geometryActive![
          layerIndex * program.profile.maxGeometryProbes + probeIndex
        ]
      ).some((flag) => flag === 1);
    if (!active) {
      throw generationError(
        "INERT_MEASUREMENT_PROBE",
        `Browser probe ${probe.name} has no active GPU geometry slot`,
      );
    }
  }
  if (sampling.persist_subspace_coords === true) {
    prepareSubspaceTrailPlans(program);
  }
  const hasMeasurements = scalarProbes.length > 0 ||
    geometryProbes.some((probe) => probe !== null) ||
    schema.lensReadout === true || schema.saeReadout === true;
  if (
    hasMeasurements && sampling.return_probe_readings !== false &&
    specialTokenIds === undefined
  ) {
    throw generationError(
      "MEASUREMENT_SPECIAL_TOKEN_IDS_UNAVAILABLE",
      "Exact aggregate measurements require the model's verified special-token IDs",
    );
  }
}

function exactReadoutTopK(
  plan: Pick<WebLlmGenerationPlan, "readoutTopK">,
  sampling: WSSampling,
): number {
  const requested = plan.readoutTopK ?? (
    typeof sampling.return_top_k === "number" && sampling.return_top_k > 0
      ? Math.min(sampling.return_top_k, BROWSER_EXACT_READOUT_TOP_K_CAPACITY)
      : BROWSER_EXACT_READOUT_TOP_K_CAPACITY
  );
  if (!Number.isSafeInteger(requested) || requested < 1) {
    throw generationError(
      "INVALID_EXACT_READOUT_TOP_K",
      "The exact browser readout width must be a positive integer",
    );
  }
  if (requested > BROWSER_EXACT_READOUT_TOP_K_CAPACITY) {
    throw generationError(
      "EXACT_READOUT_TOP_K_EXCEEDS_CAPACITY",
      `Requested exact browser readout top-${requested}, but the verified GPU ABI supports at most top-${BROWSER_EXACT_READOUT_TOP_K_CAPACITY}`,
    );
  }
  return requested;
}

function validateTokenIds(
  value: readonly number[] | null | undefined,
  label: string,
): void {
  if (value === null || value === undefined) return;
  if (!Array.isArray(value) || value.some((tokenId) =>
    !Number.isSafeInteger(tokenId) || tokenId < 0
  )) {
    throw generationError(
      "INVALID_REPLAY_REQUEST",
      `${label} must be a list of nonnegative token IDs`,
    );
  }
}

function validateSampling(value: WSSampling | null | undefined): WSSampling {
  const sampling = value ?? {};
  optionalFinite(sampling.temperature, "temperature");
  optionalFiniteRange(sampling.top_p, "top_p", 0, 1);
  optionalInteger(sampling.top_k, "top_k", 0, BROWSER_SAMPLING_TOP_K_MAX);
  optionalInteger(sampling.max_tokens, "max_tokens", 1);
  optionalInteger(sampling.seed, "seed", 0, Number.MAX_SAFE_INTEGER);
  optionalInteger(
    sampling.return_top_k,
    "return_top_k",
    0,
    BROWSER_RETURN_TOP_K_MAX,
  );
  optionalFiniteRange(sampling.presence_penalty, "presence_penalty", -2, 2);
  optionalFiniteRange(sampling.frequency_penalty, "frequency_penalty", -2, 2);
  optionalBoolean(sampling.return_probe_readings, "return_probe_readings");
  optionalBoolean(sampling.persist_per_layer_scores, "persist_per_layer_scores");
  optionalBoolean(sampling.persist_subspace_coords, "persist_subspace_coords");
  if (
    sampling.stop !== null && sampling.stop !== undefined &&
    (!Array.isArray(sampling.stop) ||
      sampling.stop.some((item) => typeof item !== "string" || item.length === 0))
  ) {
    throw generationError("INVALID_SAMPLING", "stop must be a list of nonempty strings");
  }
  if (sampling.logit_bias !== null && sampling.logit_bias !== undefined) {
    if (!isPlainRecord(sampling.logit_bias)) {
      throw generationError("INVALID_SAMPLING", "logit_bias must be an object");
    }
    for (const bias of Object.values(sampling.logit_bias)) {
      if (typeof bias !== "number" || !Number.isFinite(bias)) {
        throw generationError("INVALID_SAMPLING", "logit_bias values must be finite");
      }
    }
  }
  return sampling;
}

function copySampling(target: Record<string, unknown>, sampling: WSSampling): void {
  for (const key of [
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "seed",
    "logit_bias",
    "presence_penalty",
    "frequency_penalty",
  ] as const) {
    const value = sampling[key];
    if (value !== null && value !== undefined) target[key] = value;
  }
}

function emittedToken(
  text: string,
  thinking: boolean,
  tokenId: number | null,
  logprob: number | null,
  samplerEntropy: number | null,
  perplexity: number | null,
  rawIndex: number | null,
  topAlts: TokenAltJSON[] | null,
  measurements: MeasurementsEnvelopeJSON | undefined,
  hookMeasurements: Float32Array | undefined,
  replayScore: WebLlmReplayScore | undefined,
): WebLlmGeneratedToken {
  return {
    text,
    thinking,
    tokenId,
    // Forced replay can have zero sampler probability; raw callbacks retain -Infinity.
    logprob: Number.isFinite(logprob) ? logprob : null,
    samplerEntropy,
    perplexity,
    rawIndex,
    ...(topAlts === null ? {} : {
      topAlts: topAlts.filter((alternative) => Number.isFinite(alternative.logprob)),
    }),
    ...(replayScore === undefined ? {} : { replayScore }),
    ...(measurements === undefined ? {} : { measurements }),
    ...(hookMeasurements === undefined
      ? {}
      : { hookMeasurements: new Float32Array(hookMeasurements) }),
  };
}

function persistedMeasurementEnvelope(
  envelope: MeasurementsEnvelopeJSON | undefined,
  sampling: WSSampling | null | undefined,
): MeasurementsEnvelopeJSON | undefined {
  if (envelope === undefined || sampling?.persist_per_layer_scores === true) {
    return envelope;
  }
  const persisted = { ...envelope };
  delete persisted.per_layer_scores;
  return persisted;
}

class AggregateMeasurementSelector {
  private readonly specialTokenIds: ReadonlySet<number> | null | undefined;
  private first: MeasurementsEnvelopeJSON | undefined;
  private lastContent: MeasurementsEnvelopeJSON | undefined;

  constructor(specialTokenIds: readonly number[] | null | undefined) {
    this.specialTokenIds = specialTokenIds === null
      ? null
      : specialTokenIds === undefined
        ? undefined
        : new Set(specialTokenIds);
  }

  observe(
    tokenId: number | null,
    envelope: MeasurementsEnvelopeJSON | undefined,
  ): void {
    if (envelope === undefined || this.specialTokenIds === null) return;
    if (this.specialTokenIds === undefined) {
      throw generationError(
        "MEASUREMENT_SPECIAL_TOKEN_IDS_UNAVAILABLE",
        "Exact aggregate measurements require the model's verified special-token IDs",
      );
    }
    if (tokenId === null) {
      throw generationError(
        "MEASUREMENT_TOKEN_ID_UNAVAILABLE",
        "The browser model omitted a token ID required for exact aggregate measurement pooling",
      );
    }
    this.first ??= envelope;
    if (!this.specialTokenIds.has(tokenId)) this.lastContent = envelope;
  }

  get value(): MeasurementsEnvelopeJSON | undefined {
    return this.lastContent ?? this.first;
  }
}

async function readHookMeasurements(
  hookProgram: WebLlmHookProgram | null | undefined,
  hooks: WebLlmGenerationHooks,
  readoutTopK: number,
  includeDiscoveryReadouts = true,
): Promise<WebLlmHookMeasurementBuffers> {
  if (!hookProgram) {
    return {
      scalar: undefined,
      geometry: undefined,
      jlensTopTokens: undefined,
      saeTopFeatures: undefined,
    };
  }
  const schema = isStructuredProgram(hookProgram)
    ? hookProgram.measurementSchema
    : undefined;
  const hasScalarMeasurements = schema?.probes.some((probe) => probe !== null) === true;
  const hasGeometryMeasurements = hasActiveGeometryMeasurements(hookProgram);
  const hasJlensReadout = includeDiscoveryReadouts && schema?.lensReadout === true;
  const hasSaeReadout = includeDiscoveryReadouts && schema?.saeReadout === true;
  const bundled = !includeDiscoveryReadouts || hooks.readBundle === undefined ||
      (!hasScalarMeasurements && !hasGeometryMeasurements &&
        !hasJlensReadout && !hasSaeReadout)
    ? undefined
    : await hooks.readBundle(readoutTopK);
  const scalar = bundled?.scalar ??
    (hasScalarMeasurements ? await hooks.read() : undefined);
  const geometry = bundled?.geometry ??
    (hasGeometryMeasurements ? await hooks.readGeometry?.() : undefined);
  if (hasScalarMeasurements && scalar === undefined) {
    throw generationError(
      "SCALAR_MEASUREMENTS_UNAVAILABLE",
      "The loaded browser model did not return its declared scalar probe buffer",
    );
  }
  const jlensTopTokens = bundled?.jlensTopTokens ?? (hasJlensReadout
    ? await hooks.readJlensTopTokens?.(readoutTopK)
    : undefined);
  const saeTopFeatures = bundled?.saeTopFeatures ?? (hasSaeReadout
    ? await hooks.readSaeTopFeatures?.(readoutTopK)
    : undefined);
  if (hasJlensReadout && jlensTopTokens === undefined) {
    throw generationError(
      "JLENS_FULL_READOUT_UNAVAILABLE",
      "The loaded browser model did not return its exact full-vocabulary J-lens readout",
    );
  }
  if (hasSaeReadout && saeTopFeatures === undefined) {
    throw generationError(
      "SAE_FULL_READOUT_UNAVAILABLE",
      "The loaded browser model did not return its exact full-dictionary SAE readout",
    );
  }
  return {
    scalar: scalar === undefined ? undefined : new Float32Array(scalar),
    geometry: geometry === undefined ? undefined : new Float32Array(geometry),
    jlensTopTokens,
    saeTopFeatures,
  };
}

export function measurementEnvelope(
  hookProgram: WebLlmHookProgram | null | undefined,
  values: Float32Array | undefined,
  steering: string | null,
  geometryValues?: Float32Array,
  jlensTopTokens?: WebLlmJlensTopTokenReadout,
  saeTopFeatures?: WebLlmSaeTopFeatureReadout,
  readoutTopK = BROWSER_EXACT_READOUT_TOP_K_CAPACITY,
  persistSubspaceCoords = false,
): MeasurementsEnvelopeJSON | undefined {
  if (
    !hookProgram || !isStructuredProgram(hookProgram) ||
    hookProgram.measurementSchema === undefined ||
    values === undefined && geometryValues === undefined &&
      jlensTopTokens === undefined && saeTopFeatures === undefined
  ) {
    return undefined;
  }
  const schema = hookProgram.measurementSchema;
  const probeCount = schema.probes.length;
  if (values !== undefined && values.length !== hookProgram.layerCount * probeCount) {
    throw generationError(
      "INVALID_HOOK_MEASUREMENTS",
      "The browser model returned an unexpected measurement buffer shape",
    );
  }
  const scores: Record<string, number> = {};
  const perLayerScores: Record<string, Record<string, number>> = {};
  const gateScores: Record<string, number> = {};
  const lensReadings: Record<string, ScalarReadingJSON> = {};
  const saeReadings: Record<string, ScalarReadingJSON> = {};
  const geometryReadings: Record<string, ProbeReadingJSON> = {};
  let hasGeometry = false;
  schema.probes.forEach((probe, probeIndex) => {
    if (probe === null) return;
    if (values === undefined) return;
    const readings: Array<{ layer: number; depth: number; value: number }> = [];
    for (let layerIndex = 0; layerIndex < hookProgram.layerCount; layerIndex += 1) {
      const slot = layerIndex * probeCount + probeIndex;
      if (hookProgram.probeKind[slot] === 0) continue;
      const value = values[slot];
      if (!Number.isFinite(value)) continue;
      readings.push({
        layer: schema.layerMap[layerIndex],
        depth: normalizedLayerDepth(
          schema.layerMap[layerIndex],
          schema.modelLayerCount,
        ),
        value,
      });
    }
    if (readings.length === 0) return;
    const aggregateValue = readings.reduce((sum, item) => sum + item.value, 0) /
      readings.length;
    scores[probe.name] = round6(aggregateValue);
    gateScores[probe.name] = aggregateValue;
    const perLayer = Object.fromEntries(
      readings.map((item) => [String(item.layer), round6(item.value)]),
    );
    for (const reading of readings) {
      const layer = String(reading.layer);
      const row = perLayerScores[layer] ?? {};
      row[probe.name] = round6(reading.value);
      perLayerScores[layer] = row;
    }
    if (probe.family === "geometry") {
      hasGeometry = true;
      return;
    }
    if (probe.family === "sae" && probe.featureId !== undefined) {
      const maxAct = probe.maxAct ?? null;
      saeReadings[probe.name] = {
        value: round6(aggregateValue),
        unit: maxAct === null ? "raw_activation" : "activation_over_max",
        per_layer: perLayer,
        depth: scalarDepthSummary(
          readings,
          (reading) => Math.max(0, reading.value),
          "single_layer",
        ),
      };
      return;
    }
    if (probe.family === "lens" && probe.tokenId !== undefined) {
      lensReadings[probe.name] = {
        value: round6(aggregateValue),
        unit: "mean_token_probability",
        per_layer: perLayer,
        depth: scalarDepthSummary(
          readings,
          (reading) => Math.max(0, reading.value),
          "readout_probability_mass",
        ),
      };
    }
  });
  const geometrySchemas = schema.geometryProbes ?? [];
  if (geometrySchemas.some((probe) => probe !== null)) {
    if (
      hookProgram.format !== "drowse-structured-v3" ||
      hookProgram.profile.maxGeometryProbes === 0
    ) {
      throw generationError(
        "INVALID_GEOMETRY_MEASUREMENT_SCHEMA",
        "The browser model returned geometry readout metadata for an incompatible hook program",
      );
    }
    if (geometryValues === undefined) {
      throw generationError(
        "GEOMETRY_MEASUREMENTS_UNAVAILABLE",
        "The browser model runtime did not return its exact geometry readout buffer",
      );
    }
    const expected = hookProgram.layerCount * hookProgram.profile.maxGeometryProbes *
      hookProgram.profile.geometryOutputStride;
    if (geometryValues.length !== expected) {
      throw generationError(
        "INVALID_GEOMETRY_MEASUREMENTS",
        "The browser model returned an unexpected geometry measurement buffer shape",
      );
    }
    geometrySchemas.forEach((probe, probeIndex) => {
      if (probe === null) return;
      const reading = aggregateGeometryProbe(
        hookProgram,
        geometryValues,
        probe,
        probeIndex,
        persistSubspaceCoords,
      );
      if (reading === null) return;
      geometryReadings[probe.name] = reading.reading;
      const primary = reading.scores[probe.name];
      if (primary !== undefined) scores[probe.name] = round6(primary);
      Object.assign(gateScores, reading.scores);
      for (const [layer, layerScores] of Object.entries(reading.perLayerScores)) {
        const primaryPerLayer = layerScores[probe.name];
        if (primaryPerLayer !== undefined) {
          (perLayerScores[layer] ??= {})[probe.name] = round6(primaryPerLayer);
        }
      }
      hasGeometry = true;
    });
  }

  const instruments: MeasurementsEnvelopeJSON["instruments"] = {};
  if (hasGeometry) instruments.geometry = { readings: geometryReadings };
  const exactLens = jlensTopTokens === undefined
    ? null
    : exactLensReadout(hookProgram, schema, jlensTopTokens, readoutTopK);
  if (Object.keys(lensReadings).length > 0 || exactLens !== null) {
    instruments.lens = {
      binding: { source: schema.lensSource, steering },
      ...(Object.keys(lensReadings).length > 0 ? { readings: lensReadings } : {}),
      ...(exactLens === null ? {} : { readout: exactLens }),
    };
  }
  const exactSae = saeTopFeatures === undefined
    ? null
    : exactSaeReadout(hookProgram, schema, saeTopFeatures, readoutTopK);
  if (Object.keys(saeReadings).length > 0 || exactSae !== null) {
    instruments.sae = {
      binding: {
        source: schema.saeSource,
        steering,
        layer: schema.saeLayer ?? null,
      },
      ...(Object.keys(saeReadings).length > 0 ? { readings: saeReadings } : {}),
      ...(exactSae === null ? {} : { readout: { features: exactSae } }),
    };
  }
  const envelope: MeasurementsEnvelopeJSON = {
    version: 1,
    scope: "token",
    provenance: "captured",
    instruments,
    ...(Object.keys(scores).length > 0 ? { scores } : {}),
    ...(Object.keys(perLayerScores).length > 0 ? { per_layer_scores: perLayerScores } : {}),
  };
  measurementGateScoreCache.set(envelope, Object.freeze({ ...gateScores }));
  return envelope;
}

const measurementGateScoreCache = new WeakMap<
  MeasurementsEnvelopeJSON,
  Readonly<Record<string, number>>
>();

export function measurementGateScores(
  envelope: MeasurementsEnvelopeJSON,
): Readonly<Record<string, number>> {
  return measurementGateScoreCache.get(envelope) ?? envelope.scores ?? {};
}

function exactLensReadout(
  program: StructuredHookProgramBuffers,
  schema: NonNullable<StructuredHookProgramBuffers["measurementSchema"]>,
  value: WebLlmJlensTopTokenReadout,
  width: number,
): LensReadoutBlockJSON {
  if (
    schema.lensReadout !== true ||
    !(value.tokenIds instanceof Int32Array) ||
    !(value.strength instanceof Float32Array) ||
    !(value.centerOfMass instanceof Float32Array) ||
    !(value.spread instanceof Float32Array) ||
    !(value.layerIndices instanceof Int32Array) ||
    !(value.layerTokenIds instanceof Int32Array) ||
    !(value.layerProbabilities instanceof Float32Array) ||
    value.tokenIds.length !== width || value.tokens.length !== width ||
    value.strength.length !== width || value.centerOfMass.length !== width ||
    value.spread.length !== width ||
    value.layerIndices.length !== value.fittedLayerCount ||
    value.layerTokenIds.length !== value.fittedLayerCount * width ||
    value.layerTokens.length !== value.fittedLayerCount * width ||
    value.layerProbabilities.length !== value.fittedLayerCount * width ||
    !Number.isSafeInteger(value.fittedLayerCount) || value.fittedLayerCount < 1 ||
    value.tokenIds.some((tokenId) => tokenId < 0) ||
    value.layerTokenIds.some((tokenId) => tokenId < 0) ||
    value.strength.some((strength) => !unitInterval(strength)) ||
    value.centerOfMass.some((center) => !unitInterval(center)) ||
    value.spread.some((spread) => !unitInterval(spread)) ||
    value.layerProbabilities.some((probability) => !unitInterval(probability)) ||
    value.tokens.some((token) => typeof token !== "string") ||
    value.layerTokens.some((token) => typeof token !== "string") ||
    value.layerIndices.some((layerIndex, index) =>
      layerIndex < 0 || layerIndex >= program.layerCount ||
      index > 0 && layerIndex <= value.layerIndices[index - 1]
    ) ||
    new Set(value.tokenIds).size !== value.tokenIds.length ||
    !descending(value.strength)
  ) {
    throw generationError(
      "INVALID_JLENS_TOP_READOUT",
      "The browser model returned a malformed exact J-lens readout",
    );
  }
  const layers = [...value.layerIndices].map((runtimeLayerIndex, row) => {
    const start = row * width;
    const ids = value.layerTokenIds.subarray(start, start + width);
    const probabilities = value.layerProbabilities.subarray(start, start + width);
    if (new Set(ids).size !== ids.length || !descending(probabilities)) {
      throw generationError(
        "INVALID_JLENS_TOP_READOUT",
        "The browser model returned an invalid per-layer J-lens ranking",
      );
    }
    return {
      layer: schema.layerMap[runtimeLayerIndex],
      tokens: [...ids].map((id, index) => ({
        token: value.layerTokens[start + index],
        id,
        logprob: Math.log(Math.max(1e-45, probabilities[index])),
      })),
    };
  });
  return {
    layers,
    aggregate: [...value.tokenIds].map((tokenId, index) => ({
      token: value.tokens[index],
      strength: value.strength[index],
      com: value.centerOfMass[index],
      spread: value.spread[index],
    })),
  };
}

function exactSaeReadout(
  program: StructuredHookProgramBuffers,
  schema: NonNullable<StructuredHookProgramBuffers["measurementSchema"]>,
  value: WebLlmSaeTopFeatureReadout,
  width: number,
): SaeFeatureJSON[] {
  const actualLayer = schema.layerMap[value.runtimeLayerIndex];
  if (
    schema.saeReadout !== true ||
    !(value.featureIds instanceof Int32Array) ||
    !(value.activations instanceof Float32Array) ||
    value.featureIds.length !== value.activations.length ||
    value.featureIds.length > width ||
    value.runtimeLayerIndex < 0 || value.runtimeLayerIndex >= program.layerCount ||
    actualLayer !== schema.saeLayer ||
    value.featureCount !== schema.saeFeatureCount ||
    value.featureIds.some((featureId) =>
      featureId < 0 || featureId >= value.featureCount
    ) ||
    value.activations.some((activation) => !Number.isFinite(activation) || activation < 0) ||
    new Set(value.featureIds).size !== value.featureIds.length ||
    !descending(value.activations)
  ) {
    throw generationError(
      "INVALID_SAE_TOP_READOUT",
      "The browser model returned a malformed exact SAE dictionary readout",
    );
  }
  return [...value.featureIds].map((id, index) => {
    const metadata = schema.saeFeatureMetadata?.[String(id)];
    return {
      id,
      activation: value.activations[index],
      label: metadata?.label ?? null,
      max_act: metadata?.maxAct ?? null,
    };
  });
}

function scalarDepthSummary<T extends { depth: number }>(
  readings: readonly T[],
  mass: (reading: T) => number,
  basis: "readout_probability_mass" | "single_layer",
): DepthSummaryJSON | null {
  const weights = readings.map((reading) => Math.max(0, mass(reading)));
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (!(total > 0)) return null;
  const center = readings.reduce(
    (sum, reading, index) => sum + reading.depth * weights[index],
    0,
  ) / total;
  const spread = Math.sqrt(readings.reduce(
    (sum, reading, index) => sum + weights[index] * (reading.depth - center) ** 2,
    0,
  ) / total);
  return { center: [round6(center)], spread: [round6(spread)], basis };
}

function round6(value: number): number {
  return Math.round(value * 1e6) / 1e6;
}

function normalizedLayerDepth(layer: number, modelLayerCount: number): number {
  return modelLayerCount <= 1 ? 0 : layer / (modelLayerCount - 1);
}

function unitInterval(value: number): boolean {
  return Number.isFinite(value) && value >= 0 && value <= 1;
}

function descending(values: ArrayLike<number>): boolean {
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[index - 1]) return false;
  }
  return true;
}

function aggregateGeometryProbe(
  program: StructuredHookProgramBuffers,
  values: Float32Array,
  probe: StructuredGeometryMeasurementProbe,
  probeIndex: number,
  persistSubspaceCoords: boolean,
): {
  reading: ProbeReadingJSON;
  scores: Record<string, number>;
  perLayerScores: Record<string, Record<string, number>>;
} | null {
  const profile = program.profile;
  if (
    probeIndex < 0 || probeIndex >= profile.maxGeometryProbes ||
    probe.intrinsicDim < 1 || probe.intrinsicDim > profile.maxIntrinsicDim ||
    probe.labels.length < 1 || probe.labels.length > profile.maxGeometryCandidates ||
    probe.shareWeights.length !== program.layerCount ||
    probe.assignBandwidth.length !== probe.labels.length ||
    probe.assignLogVolumeBias.length !== probe.labels.length ||
    !(probe.labelScale > 0) || !Number.isFinite(probe.labelScale)
  ) {
    throw generationError(
      "INVALID_GEOMETRY_MEASUREMENT_SCHEMA",
      `Geometry probe ${probe.name} has incompatible aggregation metadata`,
    );
  }
  const rows: Array<{
    slot: number;
    layer: number;
    weight: number;
    fraction: number;
    residual: number;
    membership: number;
    coords: number[];
    distances: number[];
    subspaceCoords: number[] | null;
  }> = [];
  for (let layerIndex = 0; layerIndex < program.layerCount; layerIndex += 1) {
    const slot = layerIndex * profile.maxGeometryProbes + probeIndex;
    const offset = slot * profile.geometryOutputStride;
    if (values[offset] < 0.5) continue;
    const fraction = values[offset + 1];
    const residual = values[offset + 2];
    const membership = values[offset + 3];
    const coords = Array.from(
      values.subarray(offset + 4, offset + 4 + probe.intrinsicDim),
    );
    const distanceOffset = offset + 4 + profile.maxIntrinsicDim;
    const distances = Array.from(
      values.subarray(distanceOffset, distanceOffset + probe.labels.length),
    );
    if (
      ![fraction, residual, membership, ...coords, ...distances].every(Number.isFinite)
    ) {
      if (persistSubspaceCoords) {
        throw subspaceTrailError(
          probe.name,
          program.measurementSchema!.layerMap[layerIndex],
          "the geometry readout contains non-finite values",
        );
      }
      continue;
    }
    const subspaceCoords = persistSubspaceCoords
      ? reconstructSubspaceCoordinates(
          program,
          slot,
          distances,
          probe.name,
          program.measurementSchema!.layerMap[layerIndex],
        )
      : null;
    rows.push({
      slot,
      layer: program.measurementSchema!.layerMap[layerIndex],
      weight: Math.max(0, probe.shareWeights[layerIndex] ?? 0),
      fraction,
      residual,
      membership,
      coords,
      distances,
      subspaceCoords,
    });
  }
  if (rows.length === 0) return null;
  let weightTotal = rows.reduce((sum, row) => sum + row.weight, 0);
  if (!(weightTotal > 0)) {
    rows.forEach((row) => { row.weight = 1; });
    weightTotal = rows.length;
  }
  const coordinates = geometryCoordinateMean(program, rows);
  const distances = new Array(probe.labels.length).fill(0) as number[];
  let fraction = 0;
  let residual = 0;
  let membership = 0;
  for (const row of rows) {
    const weight = row.weight / weightTotal;
    fraction += weight * row.fraction;
    residual += weight * row.residual;
    membership += weight * row.membership;
    row.distances.forEach((value, candidate) => {
      distances[candidate] += weight * value;
    });
  }
  const probabilities = geometryAssignmentProbabilities(
    distances,
    probe.assignBandwidth,
    probe.assignLogVolumeBias,
  );
  const topN = Math.min(Math.max(0, probe.topN), probe.labels.length);
  const nearest = distances
    .map((distance, index) => ({ index, distance }))
    .sort((left, right) => left.distance - right.distance)
    .slice(0, topN)
    .map(({ index, distance }): [string, number] => [
      probe.labels[index],
      distance / probe.labelScale,
    ]);
  const assignment = probabilities
    .map((probability, index) => ({ index, probability }))
    .sort((left, right) => right.probability - left.probability)
    .slice(0, topN)
    .map(({ index, probability }): [string, number] => [probe.labels[index], probability]);
  const fractionPerLayer = Object.fromEntries(rows.map((row) => [String(row.layer), row.fraction]));
  const residualPerLayer = Object.fromEntries(rows.map((row) => [String(row.layer), row.residual]));
  const coordsPerLayer = Object.fromEntries(rows.map((row) => [String(row.layer), row.coords]));
  const depth = geometryDepthStats(
    rows,
    probe.intrinsicDim,
    program.measurementSchema!.modelLayerCount,
  );
  const reading: ProbeReadingJSON = {
    fraction,
    nearest,
    coords: coordinates,
    residual,
    fraction_per_layer: fractionPerLayer,
    coords_per_layer: coordsPerLayer,
    residual_per_layer: residualPerLayer,
    assignment,
    membership,
    depth_com: depth.com,
    depth_spread: depth.spread,
    ...(persistSubspaceCoords
      ? {
          subspace_coords_per_layer: Object.fromEntries(rows.map((row) => [
            String(row.layer),
            row.subspaceCoords!,
          ])),
        }
      : {}),
  };
  const scores = geometryScores(
    probe.name,
    coordinates,
    fraction,
    membership,
    probe.labels,
    distances,
    probabilities,
    probe.labelScale,
  );
  for (const scoreKey of probe.scoreKeys) {
    if (!(scoreKey in scores)) {
      throw generationError(
        "INVALID_GEOMETRY_MEASUREMENT_SCHEMA",
        `Geometry probe ${probe.name} does not produce requested channel ${scoreKey}`,
      );
    }
  }
  const perLayerScores = Object.fromEntries(rows.map((row) => [
    String(row.layer),
    geometryScores(
      probe.name,
      row.coords,
      row.fraction,
      row.membership,
      probe.labels,
      row.distances,
      geometryAssignmentProbabilities(
        row.distances,
        probe.assignBandwidth,
        probe.assignLogVolumeBias,
      ),
      probe.labelScale,
    ),
  ]));
  return { reading, scores, perLayerScores };
}

function geometryAssignmentProbabilities(
  distances: readonly number[],
  bandwidths: readonly number[],
  logVolumeBias: readonly number[],
): number[] {
  const logits = distances.map((distance, index) => {
    const bandwidth = Math.max(bandwidths[index], 1e-8);
    return -(distance ** 2) / (2 * bandwidth ** 2) + logVolumeBias[index];
  });
  const maximum = Math.max(...logits);
  const exponentials = logits.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return total > 0 && Number.isFinite(total)
    ? exponentials.map((value) => value / total)
    : exponentials.map(() => 1 / exponentials.length);
}

function geometryScores(
  name: string,
  coordinates: readonly number[],
  fraction: number,
  membership: number,
  labels: readonly string[],
  distances: readonly number[],
  probabilities: readonly number[],
  labelScale: number,
): Record<string, number> {
  const scores: Record<string, number> = {};
  if (coordinates.length > 0) scores[name] = coordinates[0];
  coordinates.forEach((value, axis) => { scores[`${name}[${axis}]`] = value; });
  scores[`${name}:fraction`] = fraction;
  scores[`${name}:membership`] = membership;
  labels.forEach((label, index) => {
    scores[`${name}@${label}`] = -distances[index] / labelScale;
    scores[`${name}~${label}`] = probabilities[index];
  });
  return scores;
}

function geometryDepthStats(
  rows: readonly { layer: number; weight: number; coords: readonly number[] }[],
  intrinsicDim: number,
  modelLayerCount: number,
): { com: number[]; spread: number[] } {
  const denominator = modelLayerCount - 1;
  if (!(denominator > 0)) return { com: [], spread: [] };
  const com = new Array(intrinsicDim).fill(0) as number[];
  const spread = new Array(intrinsicDim).fill(0) as number[];
  for (let axis = 0; axis < intrinsicDim; axis += 1) {
    const masses = rows.map((row) => ({
      depth: row.layer / denominator,
      mass: row.weight * Math.abs(row.coords[axis]),
    }));
    const total = masses.reduce((sum, item) => sum + item.mass, 0);
    if (!(total > 1e-12)) continue;
    com[axis] = masses.reduce((sum, item) => sum + item.mass * item.depth, 0) / total;
    spread[axis] = Math.sqrt(masses.reduce(
      (sum, item) => sum + item.mass * (item.depth - com[axis]) ** 2,
      0,
    ) / total);
  }
  return { com, spread };
}

interface SubspaceTrailSolvePlan {
  rank: number;
  candidateCount: number;
  rowCount: number;
  anchors: Float64Array;
  anchorNormSquared: Float64Array;
  orthogonalColumns: Float64Array;
  upper: Float64Array;
}

const subspaceTrailSolvePlans = new WeakMap<
  StructuredHookProgramBuffers,
  Map<number, SubspaceTrailSolvePlan>
>();

function prepareSubspaceTrailPlans(program: StructuredHookProgramBuffers): void {
  const geometry = program.measurementSchema?.geometryProbes;
  if (geometry === undefined || !geometry.some((probe) => probe !== null)) return;
  if (
    program.format !== "drowse-structured-v3" ||
    program.geometryActive === undefined ||
    program.geometryRank === undefined ||
    program.geometryCandidateCount === undefined ||
    program.geometryNodeWhite === undefined
  ) {
    throw subspaceTrailError(
      "browser geometry",
      null,
      "the structured program does not expose exact geometry anchors",
    );
  }
  geometry.forEach((probe, probeIndex) => {
    if (probe === null) return;
    for (let layerIndex = 0; layerIndex < program.layerCount; layerIndex += 1) {
      const slot = layerIndex * program.profile.maxGeometryProbes + probeIndex;
      if (program.geometryActive![slot] !== 1) continue;
      subspaceTrailSolvePlan(
        program,
        slot,
        probe.name,
        program.measurementSchema!.layerMap[layerIndex],
      );
    }
  });
}

function subspaceTrailSolvePlan(
  program: StructuredHookProgramBuffers,
  slot: number,
  probeName: string,
  modelLayer: number,
): SubspaceTrailSolvePlan {
  let programPlans = subspaceTrailSolvePlans.get(program);
  if (programPlans === undefined) {
    programPlans = new Map();
    subspaceTrailSolvePlans.set(program, programPlans);
  }
  const cached = programPlans.get(slot);
  if (cached !== undefined) return cached;

  const rank = program.geometryRank?.[slot];
  const candidateCount = program.geometryCandidateCount?.[slot];
  const packed = program.geometryNodeWhite;
  if (
    !Number.isSafeInteger(rank) || rank! < 1 || rank! > program.profile.maxRank ||
    !Number.isSafeInteger(candidateCount) || candidateCount! < rank! + 1 ||
    candidateCount! > program.profile.maxGeometryCandidates || packed === undefined
  ) {
    throw subspaceTrailError(
      probeName,
      modelLayer,
      `trilateration needs at least rank + 1 anchors (rank ${String(rank)}, anchors ${String(candidateCount)})`,
    );
  }

  const actualRank = rank as number;
  const actualCandidateCount = candidateCount as number;
  const anchors = new Float64Array(actualCandidateCount * actualRank);
  for (let candidate = 0; candidate < actualCandidateCount; candidate += 1) {
    for (let axis = 0; axis < actualRank; axis += 1) {
      const value = packed[
        (slot * program.profile.maxGeometryCandidates + candidate) *
          program.profile.maxRank + axis
      ];
      if (!Number.isFinite(value)) {
        throw subspaceTrailError(
          probeName,
          modelLayer,
          "the whitened geometry anchors contain non-finite values",
        );
      }
      anchors[candidate * actualRank + axis] = value;
    }
  }

  const rowCount = actualCandidateCount - 1;
  const differences = new Float64Array(rowCount * actualRank);
  const anchorNormSquared = new Float64Array(actualCandidateCount);
  for (let candidate = 0; candidate < actualCandidateCount; candidate += 1) {
    let normSquared = 0;
    for (let axis = 0; axis < actualRank; axis += 1) {
      const value = anchors[candidate * actualRank + axis];
      normSquared += value * value;
      if (candidate > 0) {
        differences[(candidate - 1) * actualRank + axis] = 2 * (
          value - anchors[axis]
        );
      }
    }
    anchorNormSquared[candidate] = normSquared;
  }

  const columnNorms = Array.from({ length: actualRank }, (_, column) => {
    let squared = 0;
    for (let row = 0; row < rowCount; row += 1) {
      squared += differences[row * actualRank + column] ** 2;
    }
    return Math.sqrt(squared);
  });
  const rankTolerance = Math.max(1e-12, Math.max(...columnNorms) * 1e-6);
  const orthogonalColumns = new Float64Array(rowCount * actualRank);
  const upper = new Float64Array(actualRank * actualRank);
  for (let column = 0; column < actualRank; column += 1) {
    const residual = Float64Array.from(
      { length: rowCount },
      (_, row) => differences[row * actualRank + column],
    );
    for (let pass = 0; pass < 2; pass += 1) {
      for (let prior = 0; prior < column; prior += 1) {
        let projection = 0;
        for (let row = 0; row < rowCount; row += 1) {
          projection += orthogonalColumns[prior * rowCount + row] * residual[row];
        }
        upper[prior * actualRank + column] += projection;
        for (let row = 0; row < rowCount; row += 1) {
          residual[row] -= projection * orthogonalColumns[prior * rowCount + row];
        }
      }
    }
    const norm = Math.sqrt(residual.reduce((sum, value) => sum + value * value, 0));
    if (!Number.isFinite(norm) || norm <= rankTolerance) {
      throw subspaceTrailError(
        probeName,
        modelLayer,
        `the whitened anchors have affine rank below ${actualRank}`,
      );
    }
    upper[column * actualRank + column] = norm;
    for (let row = 0; row < rowCount; row += 1) {
      orthogonalColumns[column * rowCount + row] = residual[row] / norm;
    }
  }

  const plan = {
    rank: actualRank,
    candidateCount: actualCandidateCount,
    rowCount,
    anchors,
    anchorNormSquared,
    orthogonalColumns,
    upper,
  };
  programPlans.set(slot, plan);
  return plan;
}

function reconstructSubspaceCoordinates(
  program: StructuredHookProgramBuffers,
  slot: number,
  distances: readonly number[],
  probeName: string,
  modelLayer: number,
): number[] {
  const plan = subspaceTrailSolvePlan(program, slot, probeName, modelLayer);
  if (
    distances.length !== plan.candidateCount ||
    distances.some((distance) => !Number.isFinite(distance) || distance < 0)
  ) {
    throw subspaceTrailError(
      probeName,
      modelLayer,
      "the geometry candidate distances are malformed",
    );
  }

  const right = new Float64Array(plan.rowCount);
  const referenceDistanceSquared = distances[0] ** 2;
  for (let row = 0; row < plan.rowCount; row += 1) {
    const candidate = row + 1;
    right[row] = plan.anchorNormSquared[candidate] - plan.anchorNormSquared[0] -
      (distances[candidate] ** 2 - referenceDistanceSquared);
  }
  if (right.some((value) => !Number.isFinite(value))) {
    throw subspaceTrailError(
      probeName,
      modelLayer,
      "the geometry distance equations are non-finite",
    );
  }

  const transformed = new Float64Array(plan.rank);
  for (let column = 0; column < plan.rank; column += 1) {
    for (let row = 0; row < plan.rowCount; row += 1) {
      transformed[column] += plan.orthogonalColumns[column * plan.rowCount + row] * right[row];
    }
  }
  const coordinates = new Float64Array(plan.rank);
  for (let row = plan.rank - 1; row >= 0; row -= 1) {
    let value = transformed[row];
    for (let column = row + 1; column < plan.rank; column += 1) {
      value -= plan.upper[row * plan.rank + column] * coordinates[column];
    }
    coordinates[row] = value / plan.upper[row * plan.rank + row];
  }
  if (coordinates.some((value) => !Number.isFinite(value))) {
    throw subspaceTrailError(
      probeName,
      modelLayer,
      "the reconstructed whitened coordinate is non-finite",
    );
  }

  let maximumResidual = 0;
  let distanceScale = 1;
  for (let candidate = 0; candidate < plan.candidateCount; candidate += 1) {
    let squared = 0;
    for (let axis = 0; axis < plan.rank; axis += 1) {
      const difference = coordinates[axis] - plan.anchors[candidate * plan.rank + axis];
      squared += difference * difference;
    }
    const reconstructed = Math.sqrt(squared);
    maximumResidual = Math.max(maximumResidual, Math.abs(reconstructed - distances[candidate]));
    distanceScale = Math.max(distanceScale, reconstructed, distances[candidate]);
  }
  if (!Number.isFinite(maximumResidual) || maximumResidual > 2e-4 * distanceScale) {
    throw subspaceTrailError(
      probeName,
      modelLayer,
      "the candidate distances do not resolve to one exact whitened coordinate",
    );
  }
  return Array.from(coordinates);
}

function subspaceTrailError(
  probeName: string,
  modelLayer: number | null,
  detail: string,
): Error & { code: string } {
  const layer = modelLayer === null ? "" : ` at model layer ${modelLayer}`;
  return generationError(
    "SUBSPACE_COORD_RECONSTRUCTION_FAILED",
    `Cannot persist exact subspace coordinates for ${probeName}${layer}: ${detail}`,
  );
}

function isStructuredProgram(
  program: WebLlmHookProgram,
): program is StructuredHookProgramBuffers {
  return "format" in program && (
    program.format === "drowse-structured-v2" ||
    program.format === "drowse-structured-v3"
  );
}

function hasActiveGeometryMeasurements(
  program: WebLlmHookProgram,
): program is StructuredHookProgramBuffers {
  if (!isStructuredProgram(program) || program.format !== "drowse-structured-v3") return false;
  if (!program.measurementSchema?.geometryProbes?.some((probe) => probe !== null)) return false;
  return program.geometryActive?.some((active) => active === 1) ?? false;
}

function requiresPerTokenControlUpdates(
  program: WebLlmHookProgram | null | undefined,
): boolean {
  if (!program || !isStructuredProgram(program) || program.controls === undefined) return false;
  return [...program.controls.affine, ...program.controls.curve].some((control) =>
    control !== null && control.enabled &&
      (control.gate !== null || control.phase.kind !== "both")
  );
}

function hasProbeGates(program: WebLlmHookProgram | null | undefined): boolean {
  if (!program || !isStructuredProgram(program) || !program.controls) return false;
  return [...program.controls.affine, ...program.controls.curve].some((control) =>
    control !== null && control.enabled && control.gate !== null
  );
}

function requiresPerTokenMeasurementReads(
  program: WebLlmHookProgram | null | undefined,
): boolean {
  if (!program || !isStructuredProgram(program) || !program.measurementSchema) {
    return false;
  }
  const schema = program.measurementSchema;
  return schema.probes.some((probe) => probe !== null) ||
    hasActiveGeometryMeasurements(program) ||
    schema.lensReadout === true || schema.saeReadout === true;
}

async function updateStructuredControls(
  program: WebLlmHookProgram | null | undefined,
  hooks: WebLlmGenerationHooks,
  evaluateControls: ReturnType<typeof createStructuredHookControlEvaluator>,
  generatedTokens: number,
  thinking: boolean,
  priorMeasurements: Float32Array | undefined,
  priorScores: Readonly<Record<string, number>> | undefined,
): Promise<void> {
  if (!program || !isStructuredProgram(program) || program.controls === undefined) return;
  if (!hooks.updateControls) {
    throw generationError(
      "STRUCTURED_CONTROL_UPDATE_UNAVAILABLE",
      "The browser model runtime cannot update gated steering controls",
    );
  }
  const controls = evaluateControls({
    prefill: false,
    thinking,
    generatedTokens,
    priorMeasurements,
    priorScores,
  });
  if (
    equalUint32(program.affineActive, controls.affineActive) &&
    equalUint32(program.curveActive, controls.curveActive)
  ) return;
  await hooks.updateControls(controls.affineActive, controls.curveActive);
  program.affineActive.set(controls.affineActive);
  program.curveActive.set(controls.curveActive);
}

function equalUint32(left: Uint32Array, right: Uint32Array): boolean {
  if (left.length !== right.length) return false;
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

interface ResponseLogprobRow {
  text: string;
  tokenId: number | null;
  logprob: number;
}

function responseContentLogprobs(
  rows: readonly ResponseLogprobRow[],
  specialTokenIds: readonly number[] | undefined,
): number[] {
  const special = new Set(specialTokenIds ?? []);
  return rows
    .filter((row) =>
      row.text !== "" && (row.tokenId === null || !special.has(row.tokenId))
    )
    .map((row) => row.logprob);
}

export class ThinkingStreamState {
  private readonly pending: CapturedGenerationRow[] = [];
  private inside: boolean;

  constructor(private readonly profile: WebLlmThinkingProfile | null) {
    if (
      profile !== null &&
      (!profile.start ||
        !profile.end ||
        profile.start === profile.end ||
        !validDelimiterIds(profile.startTokenIds) ||
        !validDelimiterIds(profile.endTokenIds) ||
        sameTokenSequence(profile.startTokenIds, profile.endTokenIds))
    ) {
      throw generationError(
        "THINKING_PROFILE_INVALID",
        "The model thinking delimiters are invalid",
      );
    }
    this.inside = profile?.startsInThinking ?? false;
  }

  get active(): boolean {
    return this.inside;
  }

  consume(row: CapturedGenerationRow): ClassifiedGenerationRow[] {
    if (this.profile === null) return [{ ...row, thinking: false }];
    if (row.tokenId === null) {
      throw generationError(
        "THINKING_TOKEN_ID_UNAVAILABLE",
        "The browser runtime omitted a token ID required for verified thinking classification",
      );
    }
    this.pending.push(row);
    const output: ClassifiedGenerationRow[] = [];
    while (this.pending.length > 0) {
      const delimiter = this.inside
        ? this.profile.endTokenIds
        : this.profile.startTokenIds;
      const pendingIds = this.pending.map((item) => item.tokenId!);
      if (isTokenSequencePrefix(pendingIds, delimiter)) {
        if (pendingIds.length === delimiter.length) {
          this.pending.length = 0;
          this.inside = !this.inside;
          continue;
        }
        break;
      }
      output.push({ ...this.pending.shift()!, thinking: this.inside });
    }
    return output;
  }

  flush(): ClassifiedGenerationRow[] {
    const output = this.pending.map((row) => ({
      ...row,
      thinking: this.inside,
    }));
    this.pending.length = 0;
    return output;
  }
}

class ResponseStopBuffer {
  private readonly pending: ClassifiedGenerationRow[] = [];
  private didMatch = false;

  constructor(private readonly stopStrings: readonly string[]) {}

  get matched(): boolean {
    return this.didMatch;
  }

  consume(row: ClassifiedGenerationRow): ClassifiedGenerationRow[] {
    if (row.text === "") return [];
    if (this.didMatch) {
      throw generationError(
        "TOKEN_AFTER_STOP_SEQUENCE",
        "The browser model runtime emitted content after Drowse matched a stop sequence",
      );
    }
    if (row.thinking) {
      return [...this.pending.splice(0), row];
    }
    if (this.stopStrings.length === 0) {
      return [row];
    }
    this.pending.push(row);
    const combined = this.pending.map((item) => item.text).join("");
    const matchIndex = earliestStopIndex(combined, this.stopStrings);
    if (matchIndex >= 0) {
      const visible = prefixRows(this.pending, matchIndex);
      this.pending.length = 0;
      this.didMatch = true;
      return visible;
    }
    const retainedCharacters = longestStopPrefixSuffix(
      combined,
      this.stopStrings,
    );
    const stableCharacters = combined.length - retainedCharacters;
    let stableRows = 0;
    let stableEnd = 0;
    for (const item of this.pending) {
      if (stableEnd + item.text.length > stableCharacters) break;
      stableEnd += item.text.length;
      stableRows += 1;
    }
    return this.pending.splice(0, stableRows);
  }

  flush(terminalReason: WebLlmTerminalReason): ClassifiedGenerationRow[] {
    if (terminalReason === "stop_sequence") {
      if (!this.didMatch || this.pending.length > 0) {
        throw generationError(
          "STOP_SEQUENCE_METADATA_MISMATCH",
          "The browser runtime reported a stop sequence without an exact visible-text match",
        );
      }
      return [];
    }
    if (this.didMatch) {
      throw generationError(
        "STOP_SEQUENCE_METADATA_MISMATCH",
        "Drowse matched a stop sequence but the terminal metadata disagreed",
      );
    }
    return this.pending.splice(0);
  }
}

function validDelimiterIds(value: readonly number[]): boolean {
  return Array.isArray(value) &&
    value.length > 0 &&
    value.every((tokenId) => Number.isSafeInteger(tokenId) && tokenId >= 0);
}

function sameTokenSequence(
  left: readonly number[],
  right: readonly number[],
): boolean {
  return left.length === right.length &&
    left.every((tokenId, index) => tokenId === right[index]);
}

function isTokenSequencePrefix(
  value: readonly number[],
  sequence: readonly number[],
): boolean {
  if (value.length > sequence.length) return false;
  return value.every((tokenId, index) => tokenId === sequence[index]);
}

function earliestStopIndex(
  value: string,
  stopStrings: readonly string[],
): number {
  let earliest = -1;
  for (const stop of stopStrings) {
    const index = value.indexOf(stop);
    if (index >= 0 && (earliest < 0 || index < earliest)) earliest = index;
  }
  return earliest;
}

function longestStopPrefixSuffix(
  value: string,
  stopStrings: readonly string[],
): number {
  let longest = 0;
  for (const stop of stopStrings) {
    const maximum = Math.min(value.length, stop.length - 1);
    for (let length = maximum; length > longest; length -= 1) {
      if (value.endsWith(stop.slice(0, length))) {
        longest = length;
        break;
      }
    }
  }
  return longest;
}

function prefixRows(
  rows: readonly ClassifiedGenerationRow[],
  characterCount: number,
): ClassifiedGenerationRow[] {
  const output: ClassifiedGenerationRow[] = [];
  let remaining = characterCount;
  for (const row of rows) {
    if (remaining <= 0) break;
    if (row.text.length <= remaining) {
      output.push(row);
      remaining -= row.text.length;
      continue;
    }
    output.push({ ...row, text: row.text.slice(0, remaining) });
    remaining = 0;
  }
  return output;
}

function readChatContent(choice: Record<string, unknown>): string {
  const delta = requireRecord(choice.delta, "chat completion delta");
  const content = delta.content;
  if (content === null || content === undefined) return "";
  if (typeof content !== "string") {
    throw generationError("INVALID_GENERATION_CHUNK", "Invalid chat completion content");
  }
  return content;
}

function readRawContent(choice: Record<string, unknown>): string {
  if (choice.text === null || choice.text === undefined) return "";
  if (typeof choice.text !== "string") {
    throw generationError("INVALID_GENERATION_CHUNK", "Invalid raw completion content");
  }
  return choice.text;
}

function readLogprobRows(value: unknown): Array<{
  token: string;
  tokenId: number | null;
  logprob: number;
  samplerEntropy: number;
  perplexity: number;
  topAlts: TokenAltJSON[] | null;
  replayScore?: WebLlmReplayScore;
}> | null {
  if (value === null || value === undefined) return null;
  const logprobs = requireRecord(value, "completion logprobs");
  if (logprobs.content === null || logprobs.content === undefined) return null;
  const content = requireArray(logprobs.content, "completion logprob content");
  return content.map((value) => {
    const row = requireRecord(value, "completion logprob row");
    if (typeof row.token !== "string" || !isLogprob(row.logprob)) {
      throw generationError("INVALID_GENERATION_CHUNK", "Invalid completion token logprob");
    }
    const rowTokenId = row.token_id === undefined || row.token_id === null
      ? null
      : nonnegativeInteger(row.token_id, "token_id");
    const replayScore = readReplayScore(row.drowse_replay, rowTokenId);
    const tokenId = rowTokenId ?? replayScore?.emittedTokenId ?? null;
    const sampler = requireRecord(
      row.drowse_sampler,
      "Drowse sampler metadata",
    );
    const samplerEntropy = nonnegativeFinite(
      sampler.entropy_nats,
      "sampler entropy_nats",
    );
    const perplexity = finiteAtLeast(
      sampler.perplexity,
      "sampler perplexity",
      1,
    );
    if (
      Math.abs(Math.log(perplexity) - samplerEntropy) >
        1e-6 * Math.max(1, samplerEntropy)
    ) {
      throw generationError(
        "INVALID_GENERATION_CHUNK",
        "The browser runtime returned inconsistent sampler entropy and perplexity",
      );
    }
    const alternatives = row.top_logprobs === undefined || row.top_logprobs === null
      ? null
      : requireArray(row.top_logprobs, "completion top logprobs").map((value) => {
          const alternative = requireRecord(value, "completion top logprob");
          if (
            typeof alternative.token !== "string" ||
            !isLogprob(alternative.logprob)
          ) {
            throw generationError(
              "INVALID_GENERATION_CHUNK",
              "Invalid completion top logprob",
            );
          }
          return {
            id: nonnegativeInteger(alternative.token_id, "top token_id"),
            text: alternative.token,
            logprob: alternative.logprob,
          };
        });
    return {
      token: row.token,
      tokenId,
      logprob: row.logprob,
      samplerEntropy,
      perplexity,
      topAlts: alternatives,
      ...(replayScore === undefined ? {} : { replayScore }),
    };
  });
}

function readReplayScore(
  value: unknown,
  rowTokenId: number | null,
): WebLlmReplayScore | undefined {
  if (value === null || value === undefined) return undefined;
  const replay = requireRecord(value, "Drowse replay score");
  const emittedTokenId = nonnegativeInteger(
    replay.emitted_token_id,
    "replay emitted_token_id",
  );
  const sampledTokenId = nonnegativeInteger(
    replay.sampled_token_id,
    "replay sampled_token_id",
  );
  const forcedTokenId = replay.forced_token_id === null
    ? null
    : nonnegativeInteger(replay.forced_token_id, "replay forced_token_id");
  const requestedLogprobs = readSamplerLogprobs(
    replay.selected_logprobs,
    "replay selected_logprobs",
  );
  const argmax = readSamplerLogprob(replay.argmax, "replay argmax");
  const topLogprobs = readSamplerLogprobs(
    replay.top_logprobs,
    "replay top_logprobs",
  );
  if (
    topLogprobs.length === 0 ||
    topLogprobs.length > 32 ||
    topLogprobs[0].tokenId !== argmax.tokenId ||
    topLogprobs[0].logprob !== argmax.logprob ||
    (rowTokenId !== null && emittedTokenId !== rowTokenId) ||
    (forcedTokenId === null
      ? emittedTokenId !== sampledTokenId
      : emittedTokenId !== forcedTokenId)
  ) {
    throw generationError(
      "INVALID_GENERATION_CHUNK",
      "Invalid Drowse replay score",
    );
  }
  return {
    emittedTokenId,
    sampledTokenId,
    forcedTokenId,
    requestedLogprobs,
    argmax,
    topLogprobs,
  };
}

function readSamplerLogprobs(
  value: unknown,
  label: string,
): WebLlmSamplerLogprob[] {
  return requireArray(value, label).map((row) => readSamplerLogprob(row, label));
}

function readSamplerLogprob(
  value: unknown,
  label: string,
): WebLlmSamplerLogprob {
  const row = requireRecord(value, label);
  if (!isLogprob(row.logprob)) {
    throw generationError("INVALID_GENERATION_CHUNK", `Invalid ${label}`);
  }
  return {
    tokenId: nonnegativeInteger(row.token_id, `${label} token_id`),
    logprob: row.logprob,
  };
}

function isLogprob(value: unknown): value is number {
  return typeof value === "number" &&
    (Number.isFinite(value) || value === Number.NEGATIVE_INFINITY);
}

function readUsage(value: unknown): {
  usage: WebLlmGenerationUsage;
  prefillTokensPerSecond: number | null;
  decodeTokensPerSecond: number | null;
} | null {
  if (value === null || value === undefined) return null;
  const usage = requireRecord(value, "completion usage");
  const promptTokens = nonnegativeInteger(usage.prompt_tokens, "prompt_tokens");
  const completionTokens = nonnegativeInteger(usage.completion_tokens, "completion_tokens");
  const totalTokens = nonnegativeInteger(usage.total_tokens, "total_tokens");
  if (totalTokens !== promptTokens + completionTokens) {
    throw generationError(
      "INCONSISTENT_GENERATION_USAGE",
      "The browser model runtime returned usage totals that do not add up",
    );
  }
  const extra = usage.extra === null || usage.extra === undefined
    ? null
    : requireRecord(usage.extra, "completion usage extra");
  return {
    usage: { promptTokens, completionTokens, totalTokens },
    prefillTokensPerSecond: optionalPositiveRate(extra?.prefill_tokens_per_s),
    decodeTokensPerSecond: optionalPositiveRate(extra?.decode_tokens_per_s),
  };
}

function mergeObservedUsage(
  current: WebLlmGenerationUsage | null,
  next: WebLlmGenerationUsage,
): WebLlmGenerationUsage {
  if (
    current !== null &&
    (current.promptTokens !== next.promptTokens ||
      current.completionTokens !== next.completionTokens ||
      current.totalTokens !== next.totalTokens)
  ) {
    throw generationError(
      "INCONSISTENT_GENERATION_USAGE",
      "The browser model runtime returned conflicting usage records",
    );
  }
  return { ...next };
}

function requireCompletedUsage(
  usage: WebLlmGenerationUsage | null,
  observedCompletionTokens: number,
): WebLlmGenerationUsage {
  if (usage === null) {
    throw generationError(
      "MISSING_GENERATION_USAGE",
      "The browser model runtime completed without token usage",
    );
  }
  if (usage.completionTokens !== observedCompletionTokens) {
    throw generationError(
      "INCONSISTENT_GENERATION_USAGE",
      "The browser model runtime's completion usage does not match the streamed tokens",
    );
  }
  return { ...usage };
}

function readFinishReason(value: unknown): "stop" | "length" | "abort" {
  if (
    typeof value !== "string" ||
    !new Set(["stop", "length", "abort"]).has(value)
  ) {
    throw generationError(
      "INVALID_GENERATION_TERMINATION",
      "The browser model runtime returned an invalid finish reason",
    );
  }
  return value as "stop" | "length" | "abort";
}

function readTerminalReason(value: unknown): WebLlmTerminalReason {
  if (
    typeof value !== "string" ||
    !new Set<WebLlmTerminalReason>([
      "eos",
      "stop_sequence",
      "external_stop",
      "length",
    ]).has(value as WebLlmTerminalReason)
  ) {
    throw generationError(
      "INVALID_GENERATION_TERMINATION",
      "The browser model runtime returned an invalid exact terminal reason",
    );
  }
  return value as WebLlmTerminalReason;
}

function terminalReasonMatchesFinishReason(
  terminalReason: WebLlmTerminalReason,
  finishReason: "stop" | "length" | "abort",
): boolean {
  return legacyFinishReason(terminalReason) === finishReason;
}

function legacyFinishReason(
  terminalReason: WebLlmTerminalReason,
): "stop" | "length" | "abort" {
  if (terminalReason === "length") return "length";
  if (terminalReason === "external_stop") return "abort";
  return "stop";
}

function resolveTerminalReason(
  runtimeTerminalReason: WebLlmTerminalReason | null,
  finishReason: string | null,
  localStopMatched: boolean,
): WebLlmTerminalReason {
  if (runtimeTerminalReason === null || finishReason === null) {
    throw generationError(
      "MISSING_GENERATION_TERMINATION",
      "The browser model runtime completed without exact terminal metadata",
    );
  }
  const parsedFinishReason = readFinishReason(finishReason);
  if (!terminalReasonMatchesFinishReason(runtimeTerminalReason, parsedFinishReason)) {
    throw generationError(
      "INCONSISTENT_GENERATION_TERMINATION",
      "The browser model runtime returned incompatible terminal metadata",
    );
  }
  return localStopMatched ? "stop_sequence" : runtimeTerminalReason;
}

function rememberGenerationErrorUsage(
  error: unknown,
  usage: WebLlmGenerationUsage | null,
): void {
  if (
    usage === null ||
    ((typeof error !== "object" || error === null) && typeof error !== "function")
  ) return;
  generationErrorUsage.set(error, { ...usage });
}

function optionalFiniteRange(
  value: number | null | undefined,
  label: string,
  minimum: number,
  maximum: number,
): void {
  if (value === null || value === undefined) return;
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw generationError("INVALID_SAMPLING", `${label} must be between ${minimum} and ${maximum}`);
  }
}

function optionalFinite(
  value: number | null | undefined,
  label: string,
): void {
  if (value === null || value === undefined) return;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw generationError("INVALID_SAMPLING", `${label} must be finite`);
  }
}

function nonnegativeFinite(value: unknown, label: string): number {
  return finiteAtLeast(value, label, 0);
}

function finiteAtLeast(value: unknown, label: string, minimum: number): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum) {
    throw generationError("INVALID_GENERATION_CHUNK", `Invalid ${label}`);
  }
  return value;
}

function optionalInteger(
  value: number | null | undefined,
  label: string,
  minimum: number,
  maximum = Number.MAX_SAFE_INTEGER,
): void {
  if (value === null || value === undefined) return;
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw generationError("INVALID_SAMPLING", `${label} is outside its supported range`);
  }
}

function optionalBoolean(
  value: boolean | null | undefined,
  label: string,
): void {
  if (value === null || value === undefined) return;
  if (typeof value !== "boolean") {
    throw generationError("INVALID_SAMPLING", `${label} must be a boolean`);
  }
}

function nonnegativeInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw generationError("INVALID_GENERATION_CHUNK", `Invalid ${label}`);
  }
  return value as number;
}

function optionalPositiveRate(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw generationError("INVALID_GENERATION_CHUNK", `Invalid ${label}`);
  }
  return value as Record<string, unknown>;
}

function requireArray(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) {
    throw generationError("INVALID_GENERATION_CHUNK", `Invalid ${label}`);
  }
  return value;
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return typeof value === "object" && value !== null && Symbol.asyncIterator in value;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return;
  throw signal.reason ?? new DOMException("Generation was cancelled", "AbortError");
}

function generationError(code: string, message: string): Error & { code: string } {
  return Object.assign(new Error(message), { code });
}

function generationTextDiagnostic(value: string): string {
  const limit = 96;
  const excerpt = value.length > limit ? `${value.slice(0, limit)}…` : value;
  return `${JSON.stringify(excerpt)} [${value.length} UTF-16 units]`;
}
