import type { RuntimeProgressEvent, RuntimeServiceRequest } from "../../lib/runtime/contracts";
import {
  clampOutputTokenCount,
  MAX_OUTPUT_TOKEN_COUNT,
} from "../../lib/runtime/outputTokenPolicy";
import { blake2b } from "@noble/hashes/blake2.js";
import { SAMPLING_TEMPERATURE_MAX } from "../../lib/runtime/samplingCapabilities";
import type {
  ChatRole,
  LoomNodeJSON,
  LoomTreeJSON,
  ProbeRequest,
  RecipeJSON,
  RecipeSamplingJSON,
  SessionInfo,
  WSGenerateRequest,
  WSInputMessage,
  WSSampling,
  WSServerMessage,
  WSSubmitRequest,
  WSTreeMutatedEvent,
} from "../../lib/types";
import type { RankOneHookProgramBuffers } from "./rankOneHookProgram";
import type { StructuredHookProgramBuffers } from "./structuredHookProgram";
import type {
  BrowserInstrumentRuntime,
  BrowserTokenReplayRequest,
} from "./browserInstrumentRuntime";
import { parseSteeringExpression } from "./steeringExpression";
import { validateHostedLoomTree } from "./sessionPersistence";
import { filterBrowserTree } from "./browserTreeFilter";
import {
  exportBrowserTranscript,
  parseBrowserTranscript,
  type BrowserTranscript,
} from "./browserTranscript";
import {
  browserPerTokenDiff,
  browserReadingsDiff,
  browserTextDiff,
} from "./browserLoomDiff";
import type {
  WebLlmGeneratedToken,
  WebLlmGenerationPlan,
  WebLlmGenerationResult,
  WebLlmReplayScore,
  WebLlmRuntimeCapabilities,
} from "./webLlmGeneration";
import {
  validateWebLlmGenerationSettings,
  webLlmGenerationErrorUsage,
} from "./webLlmGeneration";
import { randomUuid } from "./randomId";
import { isRoleSlug } from "./roleSlug";

type BrowserHookProgram = RankOneHookProgramBuffers | StructuredHookProgramBuffers;

interface GenerationReservation {
  anchorId: string;
  rootId: string | null;
}

interface ContinuationContext {
  node: LoomNodeJSON;
  original: LoomNodeJSON;
  parentId: string;
  forcedPrefix: number[];
}

export interface BrowserGenerationPort {
  streamGeneration(
    plan: WebLlmGenerationPlan,
    onToken: (token: WebLlmGeneratedToken) => void | Promise<void>,
  ): Promise<WebLlmGenerationResult>;
  stop(): Promise<void>;
  runtimeCapabilities?(): WebLlmRuntimeCapabilities;
  tokenizeText?(text: string): Promise<number[]>;
  decodeTokens?(tokenIds: readonly number[]): Promise<string>;
}

export interface BrowserLoomOptions {
  session: SessionInfo;
  generation: BrowserGenerationPort;
  initialTree?: LoomTreeJSON;
  drowseVersion?: string;
  now?: () => number;
  createId?: (kind: "generation" | "node") => string;
  compileSteering?: (
    expression: string,
    probeRequests?: readonly ProbeRequest[],
  ) => Promise<BrowserHookProgram> | BrowserHookProgram;
  steeringDelta?: (parent: string | null, child: string | null) => string;
  probeHashes?: () =>
    | Readonly<Record<string, string>>
    | Promise<Readonly<Record<string, string>>>;
  instruments?: BrowserInstrumentRuntime;
  maxOutputTokens?: number;
}

export class BrowserLoomRuntime {
  private session: SessionInfo;
  private tree: LoomTreeJSON;
  private readonly generation: BrowserGenerationPort;
  private readonly now: () => number;
  private readonly createId: (kind: "generation" | "node") => string;
  private readonly compileSteering: BrowserLoomOptions["compileSteering"];
  private readonly steeringDelta: BrowserLoomOptions["steeringDelta"];
  private readonly probeHashes: BrowserLoomOptions["probeHashes"];
  private readonly instruments: BrowserInstrumentRuntime | undefined;
  private readonly runtimeCapabilities: WebLlmRuntimeCapabilities | null;
  private readonly maxOutputTokens: number;
  private generating = false;
  private stopRequested = false;
  private generationController: AbortController | null = null;
  private generationReservation: GenerationReservation | null = null;

  constructor(options: BrowserLoomOptions) {
    this.maxOutputTokens = options.maxOutputTokens ?? MAX_OUTPUT_TOKEN_COUNT;
    if (!Number.isSafeInteger(this.maxOutputTokens) || this.maxOutputTokens < 1) {
      throw loomError(
        "INVALID_OUTPUT_TOKEN_LIMIT",
        "The browser output token limit must be a positive integer",
      );
    }
    const session = clone(options.session);
    this.session = {
      ...session,
      config: clampSessionOutputTokens(session.config, this.maxOutputTokens),
    };
    this.generation = options.generation;
    this.now = options.now ?? (() => Date.now() / 1_000);
    this.createId = options.createId ?? ((kind) => `${kind}-${randomUuid()}`);
    this.compileSteering = options.compileSteering;
    this.steeringDelta = options.steeringDelta;
    this.probeHashes = options.probeHashes;
    this.instruments = options.instruments;
    this.runtimeCapabilities = this.generation.runtimeCapabilities?.() ?? null;
    this.tree = options.initialTree
      ? clone(options.initialTree)
      : emptyTree(this.session, options.drowseVersion ?? "browser");
    validateHostedLoomTree(this.tree);
    if (this.tree.model_id !== this.session.model_id || this.tree.session_id !== this.session.id) {
      throw loomError(
        "LOOM_SESSION_MISMATCH",
        "The browser conversation does not match the loaded model session",
      );
    }
    this.instruments?.configureTokenReplay(
      this.compileSteering && this.runtimeCapabilities?.forcedReplay
        ? (request, onProgress) => this.replayTokenReadout(request, onProgress)
        : null,
    );
    this.syncHistoryLength();
    this.syncInstrumentSession();
  }

  async request(
    request: RuntimeServiceRequest,
    onProgress: (event: RuntimeProgressEvent) => void = () => undefined,
  ): Promise<unknown> {
    if (request.service === "sessions") return this.sessionRequest(request.method, request.args);
    if (request.service === "tree") {
      const before = clone(this.tree);
      const result = await this.treeRequest(request.method, request.args);
      const op = serviceMutationOp(request.method);
      if (op !== null && this.tree.rev !== before.rev) {
        onProgress({
          event: "tree_mutated",
          data: this.serviceMutation(op, before),
        });
      }
      return result;
    }
    if (request.service === "profiles" && this.instruments) {
      const result = await this.instruments.request(request, (nodeId, rawIndex) =>
        this.tokenMeasurements(nodeId, rawIndex)
      );
      this.syncInstrumentSession();
      return result;
    }
    if (request.service === "profiles") {
      if (request.method === "list") return { profiles: [] };
      if (request.method === "correlation") {
        return { names: [], matrix: {}, layers_shared: {} };
      }
    }
    if (request.service === "probes" && this.instruments) {
      const result = await this.instruments.request(request, (nodeId, rawIndex) =>
        this.tokenMeasurements(nodeId, rawIndex)
      );
      this.syncInstrumentSession();
      return result;
    }
    if (request.service === "probes" && request.method === "list") {
      return { probes: [] };
    }
    if (request.service === "manifolds" && this.instruments) {
      return await this.instruments.request(request, (nodeId, rawIndex) =>
        this.tokenMeasurements(nodeId, rawIndex)
      );
    }
    if (request.service === "instruments" && this.instruments) {
      const result = await this.instruments.request(
        request,
        (nodeId, rawIndex) => this.tokenMeasurements(nodeId, rawIndex),
        onProgress,
      );
      this.syncInstrumentSession();
      return result;
    }
    if (request.service === "instruments" && request.method === "sources") {
      return { sources: [] };
    }
    throw loomError(
      "BROWSER_SERVICE_UNAVAILABLE",
      `Browser runtime service ${request.service}.${request.method} is not implemented`,
    );
  }

  async generate(
    request: WSSubmitRequest | WSGenerateRequest,
    emit: (message: WSServerMessage) => void | Promise<void>,
  ): Promise<void> {
    if (this.generating) {
      throw loomError("GENERATION_BUSY", "The browser session is already generating");
    }
    validateGenerationRequest(request);
    validateInputRoleCapabilities(request, this.session);
    if (this.session.is_base_model) {
      if (request.type === "generate" && Array.isArray(request.input)) {
        throw loomError("BASE_MODEL_RAW_INPUT_REQUIRED", "Use a plain text prompt for this base model, not chat messages");
      }
      request = { ...request, raw: true };
    }
    this.stopRequested = false;
    this.generating = true;
    this.generationController = new AbortController();
    try {
      await this.runGeneration(request, emit);
    } finally {
      this.generationReservation = null;
      this.generationController = null;
      this.stopRequested = false;
      this.generating = false;
    }
  }

  stop(userInitiated = true): Promise<void> {
    if (userInitiated && this.generating) this.stopRequested = true;
    this.generationController?.abort();
    return this.generation.stop();
  }

  snapshot(): { session: SessionInfo; tree: LoomTreeJSON } {
    return { session: clone(this.session), tree: clone(this.tree) };
  }

  refreshInstrumentSession(): void {
    this.syncInstrumentSession();
  }

  private async runGeneration(
    request: WSSubmitRequest | WSGenerateRequest,
    emit: (message: WSServerMessage) => void | Promise<void>,
  ): Promise<void> {
    const fork = request.type === "generate" ? await this.resolveTokenFork(request) : null;
    const stateless = fork === null && request.type === "generate" && request.stateless !== false;
    const rawCompletion = usesRawCompletionInput(request);
    const selectedParentId = fork?.parentId ??
      this.requireNode(request.parent_node_id ?? this.tree.active_node_id).id;
    if (!stateless) {
      const addsAuthoredNode = request.type === "submit" &&
        request.text !== null && request.text !== undefined;
      const addsStringInputNode = request.type === "generate" &&
        typeof request.input === "string" &&
        (!request.raw || request.input !== "");
      this.generationReservation = {
        anchorId: selectedParentId,
        rootId: addsAuthoredNode || addsStringInputNode ? null : selectedParentId,
      };
    }
    const commitOnly = request.type === "submit" &&
      request.text !== null && request.text !== undefined &&
      (request.generated_role === null || request.generated_role === undefined);
    if (commitOnly) {
      const commitSampling = effectiveSampling(
        this.session,
        request.sampling,
        this.maxOutputTokens,
      );
      validateResolvedSamplingCapabilities(this.session, commitSampling, rawCompletion);
      const generationId = this.uniqueId("generation");
      await emit({
        type: "started",
        generation_id: generationId,
        node_id: null,
        sibling_index: 0,
        sibling_count: 1,
      });
      const authored = await this.appendAuthoredTurn({
        parentId: selectedParentId,
        role: request.authored_role!,
        text: request.text!,
        roleLabel: request.raw ? null : roleLabel(request.authored_role!, commitSampling),
        thinkingText: request.raw ? null : request.authored_thinking ?? null,
      });
      this.setReservationRoot(authored.node.id);
      await emit(authored.mutation);
      await emit({
        type: "done",
        result: {
          text: request.text!,
          tokens: 0,
          finish_reason: "stop",
          usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
          mean_logprob: null,
          mean_surprise: null,
        },
        node_id: authored.node.id,
        sibling_index: 0,
        sibling_count: 1,
      });
      return;
    }

    const settings = fork?.settings ?? resolveGenerationSettings(
      request, this.session, this.tree, selectedParentId, this.maxOutputTokens,
    );
    const effectiveSteering = settings.steering;
    const effectiveSystemPrompt = rawCompletion ? null : settings.systemPrompt;
    const sampling = settings.sampling;
    validateResolvedSamplingCapabilities(this.session, sampling, rawCompletion);
    const hookProgram = effectiveSteering || this.instruments?.hasLiveReadout()
      ? await this.preflightGenerationSteering(effectiveSteering ?? "")
      : null;
    const activeRole = activeAssistantRole(hookProgram);
    const currentProbeHashes = await this.resolveProbeHashes();
    const siblingCount = fork === null ? request.n ?? 1 : 1;
    const siblingSeeds = deriveSeedSchedule(sampling.seed ?? null, siblingCount);
    let generatedParentId = selectedParentId;
    const role: ChatRole = fork
      ? fork.source.role as ChatRole
      : request.type === "submit"
        ? request.generated_role ?? "assistant"
        : request.generate_seat ?? "assistant";
    const continuationRoleLabel = fork !== null
      ? fork.source.role_label
      : request.raw
        ? null
        : roleLabel(role, sampling);
    const continuationCandidate = request.type === "submit" &&
      request.text !== null && request.text !== undefined
      ? {
          role: request.authored_role!,
          roleLabel: request.raw ? null : roleLabel(request.authored_role!, sampling),
        }
      : this.requireNode(selectedParentId);
    const appendSameRole = request.type === "submit" || request.append_same_role !== false;
    const requestWillContinue = appendSameRole && fork === null && !stateless && siblingCount === 1 &&
      (request.type === "submit" || request.input === null || request.input === undefined ||
        (request.raw === true && request.input === "")) &&
      continuationCandidate.role === role &&
      ("roleLabel" in continuationCandidate
        ? continuationCandidate.roleLabel
        : continuationCandidate.role_label) === continuationRoleLabel;
    if (requestWillContinue) {
      if (this.runtimeCapabilities?.forcedReplay !== true) {
        throw loomError(
          "CONTINUATION_REPLAY_UNAVAILABLE",
          "The loaded browser model cannot replay an existing message as an exact continuation",
        );
      }
      const canTokenize = this.runtimeCapabilities.tokenizer === true &&
        this.generation.tokenizeText !== undefined;
      const changesAuthoredText = request.type === "submit" &&
        request.text !== null && request.text !== undefined;
      const canReuseRawRecord = !changesAuthoredText &&
        !("roleLabel" in continuationCandidate) &&
        continuationCandidate.thinking_text === null &&
        continuationCandidate.raw_token_ids !== null;
      if (!canTokenize && !canReuseRawRecord) {
        throw loomError(
          "CONTINUATION_TOKENIZER_UNAVAILABLE",
          "The loaded browser model cannot tokenize the existing message for exact continuation",
        );
      }
    }
    const requestThinking = requestWillContinue ? false : settings.thinking;
    if (requestThinking && !this.session.supports_thinking) {
      throw loomError(
        "THINKING_STREAM_UNAVAILABLE",
        "The selected browser model does not expose verified thinking tokens",
      );
    }
    validateWebLlmGenerationSettings(sampling, requestThinking);
    for (let siblingIndex = 0; siblingIndex < siblingCount; siblingIndex += 1) {
      const generationId = this.uniqueId("generation");
      await emit({
        type: "started",
        generation_id: generationId,
        node_id: null,
        sibling_index: siblingIndex,
        sibling_count: siblingCount,
      });
      if (
        siblingIndex === 0 && request.type === "submit" &&
        request.text !== null && request.text !== undefined
      ) {
        const authored = await this.appendAuthoredTurn({
          parentId: selectedParentId,
          role: request.authored_role!,
          text: request.text,
          roleLabel: request.raw ? null : roleLabel(request.authored_role!, sampling),
          thinkingText: request.raw ? null : request.authored_thinking ?? null,
        });
        generatedParentId = authored.node.id;
        this.setReservationRoot(authored.node.id);
        await emit(authored.mutation);
      } else if (
        siblingIndex === 0 && !stateless && request.type === "generate" &&
        typeof request.input === "string" &&
        (!request.raw || request.input !== "")
      ) {
        const authored = this.newNode({
          parentId: selectedParentId,
          role: "user",
          text: request.input,
          roleLabel: request.raw ? null : roleLabel("user", sampling),
          thinkingText: null,
          steering: null,
          recipe: null,
        });
        this.addNode(authored);
        generatedParentId = authored.id;
        this.setReservationRoot(authored.id);
        await emit(this.addedMutation("add_user", authored));
      }
      let siblingSampling = { ...sampling, seed: siblingSeeds[siblingIndex] };
      const generationHeaderRoleLabel = fork !== null
        ? fork.source.role_label
        : rawCompletion
          ? null
          : role === "assistant"
            ? activeRole ?? roleLabel(role, siblingSampling)
            : roleLabel(role, siblingSampling);
      const stampedRoleLabel = fork !== null
        ? fork.source.role_label
        : request.raw
          ? null
          : generationHeaderRoleLabel;
      const continuation = await this.prepareContinuation({
        eligible: appendSameRole && fork === null && !stateless && siblingCount === 1 &&
          (request.type === "submit" || request.input === null || request.input === undefined ||
            (request.raw === true && request.input === "")),
        nodeId: generatedParentId,
        role,
        roleLabel: request.raw ? null : roleLabel(role, siblingSampling),
      });
      const effectiveThinking = continuation === null ? requestThinking : false;
      if (continuation !== null && continuation.forcedPrefix.length > 0) {
        siblingSampling = {
          ...siblingSampling,
          max_tokens: continuation.forcedPrefix.length +
            (siblingSampling.max_tokens ?? this.session.config.max_tokens ?? 512),
        };
      }
      validateWebLlmGenerationSettings(siblingSampling, effectiveThinking);
      const nodeRecipe = recipe(
        effectiveSteering,
        siblingSampling,
        effectiveThinking,
        this.session.probes,
        currentProbeHashes,
        effectiveSystemPrompt,
      );
      const prefix = fork !== null
        ? savedTokenPrefix(fork.source, fork.forcedPrefix)
        : continuation !== null
          ? { ...savedTokenPrefix(continuation.node, continuation.forcedPrefix),
              text: continuation.node.text, thinking_text: null,
              thinking_tokens: [], length: continuation.forcedPrefix.length }
          : null;
      const node = continuation === null
        ? this.newNode({
            parentId: generatedParentId,
            role,
            text: "",
            roleLabel: stampedRoleLabel,
            thinkingText: null,
            steering: effectiveSteering,
            recipe: nodeRecipe,
          })
        : {
            ...clone(continuation.node),
            text: "",
            thinking_text: null,
            aggregate_readings: {},
            applied_steering: null,
            finish_reason: null,
            mean_logprob: null,
            mean_surprise: null,
            recipe: nodeRecipe,
            tokens: [],
            thinking_tokens: [],
            raw_token_ids: null,
          };
      if (prefix !== null) {
        node.text = prefix.text;
        node.thinking_text = prefix.thinking_text;
        node.tokens = clone(prefix.tokens);
        node.thinking_tokens = clone(prefix.thinking_tokens);
        node.raw_token_ids = (fork?.forcedPrefix ?? continuation!.forcedPrefix).slice(0, prefix.length);
      }
      const wireNodeId = stateless ? generatedParentId : node.id;
      if (!stateless) {
        if (continuation === null) {
          this.addNode(node);
          await emit(this.addedMutation("begin_assistant", node));
        } else {
          this.tree.active_node_id = node.id;
          this.replaceNodeWithoutActivation(node);
          await emit(this.updatedMutation("begin_assistant", node, node.id));
        }
      }

      const rawTokenIds: number[] = [...(node.raw_token_ids ?? [])];
      const prefixTokenRows = new Map((node.tokens ?? []).map((row, index) => [row.raw_index, index]));
      let rawTokenCount = 0;
      let sawRawToken = rawTokenIds.length > 0;
      let rawTokenIdentityComplete = true;
      const replayLength = fork?.forcedPrefix.length ?? continuation?.forcedPrefix.length ?? 0;
      let lastReplayProgress = 0;
      const emitReplayProgress = async (completed: number): Promise<void> => {
        if (replayLength === 0) return;
        const now = performance.now();
        if (completed !== 0 && completed !== replayLength && now - lastReplayProgress < 100) return;
        lastReplayProgress = now;
        await emit({ type: "generation_progress", node_id: wireNodeId,
          completed, total: replayLength });
      };
      await emitReplayProgress(0);
      if (this.stopRequested) {
        await this.finalizeGeneration(
          node,
          cancelledGenerationResult(node, rawTokenCount),
          "cancelled",
          rawTokenIds,
          sawRawToken,
          rawTokenIdentityComplete,
          stateless,
          wireNodeId,
          siblingIndex,
          siblingCount,
          emit,
        );
        break;
      }
      let result: WebLlmGenerationResult;
      try {
        const plan: WebLlmGenerationPlan = {
          signal: this.generationController!.signal,
          input: generationInput(
            request,
            this.tree,
            continuation?.parentId ?? generatedParentId,
            effectiveSystemPrompt,
            stateless,
            roleLabel("user", siblingSampling),
          ),
          sampling: siblingSampling,
          thinking: effectiveThinking,
          generationSeat: rawCompletion ? null : role,
          generationRoleName: rawCompletion ? null : generationHeaderRoleLabel,
          steeringExpression: effectiveSteering,
          hookProgram,
          maxOutputTokens: this.maxOutputTokens,
          ...(fork === null ? {} : { measurementStartRawIndex: prefix!.length }),
          ...(fork !== null
            ? { replay: { forcedPrefixTokenIds: fork.forcedPrefix } }
            : continuation !== null
              ? { replay: { forcedPrefixTokenIds: continuation.forcedPrefix } }
              : {}),
          onRawToken: async (token) => {
            rawTokenCount += 1;
            sawRawToken = true;
            if (token.tokenId === null) {
              rawTokenIdentityComplete = false;
              return;
            }
            if (token.rawIndex >= (prefix?.length ?? 0)) rawTokenIds.push(token.tokenId);
            if (token.rawIndex < replayLength) await emitReplayProgress(token.rawIndex + 1);
          },
        };
        result = await this.generation.streamGeneration(plan, async (token) => {
          const row = {
            text: token.text,
            logprob: token.logprob,
            ...(token.samplerEntropy === undefined
              ? {}
              : { sampler_entropy: token.samplerEntropy }),
            perplexity: token.perplexity,
            ...(token.tokenId !== null ? { token_id: token.tokenId } : {}),
            ...(token.rawIndex !== null ? { raw_index: token.rawIndex } : {}),
            ...(token.topAlts ? { top_alts: token.topAlts } : {}),
            ...(token.measurements ? {
              measurements: clone(token.measurements),
              probes: clone(token.measurements.scores ?? {}),
              per_layer_scores: clone(token.measurements.per_layer_scores ?? {}),
            } : {}),
          };
          if (token.rawIndex !== null && token.rawIndex < (prefix?.length ?? 0)) {
            if (continuation !== null) {
              const existingIndex = prefixTokenRows.get(token.rawIndex);
              if (existingIndex === undefined) {
                prefixTokenRows.set(token.rawIndex, node.tokens!.length);
                node.tokens!.push(row);
              } else node.tokens![existingIndex] = row;
            }
            return;
          }
          if (token.thinking) {
            node.thinking_text = (node.thinking_text ?? "") + token.text;
            node.thinking_tokens ??= [];
          } else {
            node.text += token.text;
            node.tokens ??= [];
          }
          (token.thinking ? node.thinking_tokens! : node.tokens!).push(row);
          await emit({
            type: "token",
            text: token.text,
            thinking: token.thinking,
            token_id: token.tokenId,
            logprob: token.logprob,
            sampler_entropy: token.samplerEntropy,
            perplexity: token.perplexity,
            top_alts: token.topAlts ?? null,
            raw_index: token.rawIndex,
            node_id: wireNodeId,
            ...(token.measurements ? { measurements: clone(token.measurements) } : {}),
          });
        });
      } catch (error) {
        if (this.stopRequested) {
          const partialUsage = webLlmGenerationErrorUsage(error);
          await this.finalizeGeneration(
            node,
            cancelledGenerationResult(
              node,
              rawTokenCount,
              partialUsage?.promptTokens ?? 0,
            ),
            "cancelled",
            rawTokenIds,
            sawRawToken,
            rawTokenIdentityComplete,
            stateless,
            wireNodeId,
            siblingIndex,
            siblingCount,
            emit,
          );
          break;
        }
        if (!stateless && this.tree.nodes.some((candidate) => candidate.id === node.id)) {
          if (continuation !== null) {
            this.tree.active_node_id = continuation.original.id;
            this.replaceNodeWithoutActivation(continuation.original);
            await emit(this.updatedMutation(
              "finalize_assistant",
              continuation.original,
              continuation.original.id,
            ));
          } else {
            this.removeLeaf(node.id);
            await emit({
              type: "tree_mutated",
              op: "delete",
              added: [],
              removed: [node.id],
              updated: [],
              active_node_id: this.tree.active_node_id,
              rev: this.tree.rev,
              cast: effectiveCast(this.tree),
            });
          }
        }
        throw error;
      }
      const externallyStopped = this.stopRequested || result.terminalReason === "external_stop";
      const finishReason = externallyStopped ? "cancelled" : result.finishReason;
      if (externallyStopped && prefix !== null && rawTokenCount < prefix.length) {
        result = cancelledGenerationResult(node, rawTokenCount, result.usage.promptTokens);
      }
      await this.finalizeGeneration(
        node,
        result,
        finishReason,
        rawTokenIds,
        sawRawToken,
        rawTokenIdentityComplete,
        stateless,
        wireNodeId,
        siblingIndex,
        siblingCount,
        emit,
      );
      if (this.stopRequested) break;
    }
  }

  private async finalizeGeneration(
    node: LoomNodeJSON,
    result: WebLlmGenerationResult,
    finishReason: string,
    rawTokenIds: readonly number[],
    sawRawToken: boolean,
    rawTokenIdentityComplete: boolean,
    stateless: boolean,
    wireNodeId: string,
    siblingIndex: number,
    siblingCount: number,
    emit: (message: WSServerMessage) => void | Promise<void>,
  ): Promise<void> {
    node.raw_token_ids = sawRawToken && rawTokenIdentityComplete
      ? [...rawTokenIds]
      : null;
    node.applied_steering = node.recipe?.steering ?? null;
    node.finish_reason = finishReason;
    node.mean_logprob = result.meanLogprob;
    node.mean_surprise = result.meanSurprise;
    node.aggregate_readings = clone(result.measurements?.scores ?? {});
    if (!stateless) {
      const finalized = this.replaceGeneratedNode(node);
      await emit(this.updatedMutation("finalize_assistant", finalized));
    }
    await emit({
      type: "done",
      result: {
        text: result.text,
        tokens: result.tokens,
        finish_reason: finishReason,
        applied_steering: node.applied_steering,
        terminal_reason: this.stopRequested
          ? "external_stop"
          : result.terminalReason,
        usage: {
          prompt_tokens: result.usage.promptTokens,
          completion_tokens: result.usage.completionTokens,
          total_tokens: result.usage.totalTokens,
        },
        mean_logprob: result.meanLogprob,
        mean_surprise: result.meanSurprise,
        ...(result.measurements ? { measurements: clone(result.measurements) } : {}),
      },
      node_id: wireNodeId,
      sibling_index: siblingIndex,
      sibling_count: siblingCount,
    });
  }

  private async sessionRequest(method: string, args: unknown[]): Promise<unknown> {
    if (method === "list") return { sessions: [clone(this.session)] };
    if (method === "get") return clone(this.session);
    if (method === "patch") {
      const patch = requireRecord(args[0], "session settings");
      const allowed = new Set(Object.keys(this.session.config));
      if (Object.keys(patch).some((key) => !allowed.has(key))) {
        throw loomError("INVALID_SESSION_SETTINGS", "Session settings contain unknown fields");
      }
      const next = clampSessionOutputTokens(
        { ...this.session.config, ...patch },
        this.maxOutputTokens,
      );
      validateSessionConfig(next);
      this.session = { ...this.session, config: next };
      return clone(this.session);
    }
    if (method === "validateSteering") {
      const expression = args[0];
      if (typeof expression !== "string") {
        throw loomError("INVALID_STEERING", "Steering expression must be a string");
      }
      try {
        const options = validateSteeringOptions(args[2]);
        if (options.tree !== undefined) {
          validateHostedLoomTree(options.tree);
          if (options.tree.model_id !== this.session.model_id) {
            throw loomError(
              "LOOM_SESSION_MISMATCH",
              "Restored conversation belongs to another model",
            );
          }
        }
        const probeRequests = options.probeRequests;
        const parsed = expression.trim() ? parseSteeringExpression(expression) : null;
        if (expression.trim() || probeRequests !== undefined) {
          await this.preflightGenerationSteering(expression, probeRequests);
        }
        return {
          valid: true,
          expression: parsed === null ? "" : formatParsedSteering(parsed),
          error: null,
        };
      } catch (error) {
        return {
          valid: false,
          expression,
          error: error instanceof Error ? error.message : String(error),
        };
      }
    }
    throw loomError("BROWSER_SERVICE_UNAVAILABLE", `sessions.${method} is not implemented`);
  }

  private async treeRequest(method: string, args: unknown[]): Promise<unknown> {
    if (method === "replayCapabilities") {
      const available = this.runtimeCapabilities?.forcedReplay === true &&
        this.runtimeCapabilities.replayScoring === true;
      return {
        jointLogprobs: {
          available,
          reason: available
            ? null
            : "This model runtime cannot replay exact token likelihoods for branch comparison.",
        },
      };
    }
    if (method === "get") return clone(this.tree);
    if (method === "active") return activePath(this.tree);
    if (method === "navigate") {
      const nodeId = requireIdentifier(args[0], "node id");
      this.requireNode(nodeId);
      if (this.tree.active_node_id === nodeId) return activePath(this.tree);
      this.tree.active_node_id = nodeId;
      this.bumpRevision();
      return activePath(this.tree);
    }
    if (method === "reset") {
      this.assertWholeTreeMutationAllowed("reset");
      this.resetTree();
      return undefined;
    }
    if (method === "restore") {
      this.assertWholeTreeMutationAllowed("restore");
      const restored = clone(args[0]);
      validateHostedLoomTree(restored);
      if (restored.model_id !== this.session.model_id) {
        throw loomError("LOOM_SESSION_MISMATCH", "Restored conversation belongs to another model");
      }
      this.tree = {
        ...restored,
        session_id: this.session.id,
        rev: Math.max(this.tree.rev, restored.rev) + 1,
      };
      this.syncHistoryLength();
      return {
        rev: this.tree.rev,
        root_id: this.tree.root_id,
        active_node_id: this.tree.active_node_id,
        nodes: this.tree.nodes.length,
      };
    }
    if (method === "edit") {
      const node = this.requireEditableNode(args[0]);
      this.assertReservedNodeMutationAllowed(node.id, "edit");
      const text = requireText(args[1], "node text", 4 * 1024 * 1024);
      node.text = text;
      node.edit_count += 1;
      node.edited_at = this.now();
      this.replaceNodeWithoutActivation(node);
      return clone(node);
    }
    if (method === "branch") {
      const sibling = this.requireEditableNode(args[0]);
      const text = requireText(args[1], "branch text", 4 * 1024 * 1024);
      const requestedRole = args[3];
      if (
        requestedRole !== undefined && requestedRole !== null &&
        requestedRole !== "user" && requestedRole !== "assistant"
      ) {
        throw loomError("INVALID_RUNTIME_REQUEST", "Branch role is invalid");
      }
      const branch = this.newNode({
        parentId: sibling.parent_id!,
        role: requestedRole ?? sibling.role as ChatRole,
        text,
        roleLabel: sibling.role_label,
        thinkingText: text === sibling.text ? sibling.thinking_text : null,
        steering: null,
        recipe: null,
      });
      branch.tokens = null;
      branch.thinking_tokens = null;
      this.addNode(branch);
      return {
        node_id: branch.id,
        node: clone(branch),
        active_path: activePath(this.tree),
      };
    }
    if (method === "delete") {
      const nodeId = requireIdentifier(args[0], "node id");
      return { removed: this.removeSubtree(nodeId) };
    }
    if (method === "star") {
      const node = this.requireNode(requireIdentifier(args[0], "node id"));
      if (typeof args[1] !== "boolean") {
        throw loomError("INVALID_RUNTIME_REQUEST", "Star state must be a boolean");
      }
      if (node.starred !== args[1]) {
        node.starred = args[1];
        this.replaceNodeWithoutActivation(node);
      }
      return clone(node);
    }
    if (method === "note") {
      const node = this.requireNode(requireIdentifier(args[0], "node id"));
      node.notes = requireText(args[1], "node note", 256 * 1024);
      this.replaceNodeWithoutActivation(node);
      return clone(node);
    }
    if (method === "filter") {
      const expression = requireText(args[0], "tree filter", 16 * 1024).trim();
      return {
        expr: expression,
        matching_node_ids: filterBrowserTree(this.tree.nodes, expression),
      };
    }
    if (method === "transcriptExport") {
      const requested = args[0];
      const nodeId = requested === null || requested === undefined
        ? this.tree.active_node_id
        : requireIdentifier(requested, "node id");
      this.requireNode(nodeId);
      const probeHashes = await this.resolveProbeHashes();
      return {
        yaml: exportBrowserTranscript(this.tree, this.session, nodeId, probeHashes),
        node_id: nodeId,
      };
    }
    if (method === "transcriptLoad") {
      this.assertWholeTreeMutationAllowed("restore");
      const source = requireText(args[0], "transcript YAML", 16 * 1024 * 1024);
      const mode = args[1];
      if (mode !== "default" && mode !== "here" && mode !== "merge") {
        throw loomError("INVALID_TRANSCRIPT", "Transcript import mode is invalid");
      }
      if (typeof args[2] !== "boolean") {
        throw loomError("INVALID_TRANSCRIPT", "Transcript strict mode must be a boolean");
      }
      return this.importTranscript(parseBrowserTranscript(source), mode, args[2]);
    }
    if (method === "edgeLabel") {
      const parent = this.requireNode(requireIdentifier(args[0], "parent node id"));
      const child = this.requireNode(requireIdentifier(args[1], "child node id"));
      return { label: this.requireSteeringDelta(nodeSteering(parent), nodeSteering(child)) };
    }
    if (method === "diff") {
      const a = this.requireNode(requireIdentifier(args[0], "first node id"));
      const b = this.requireNode(requireIdentifier(args[1], "second node id"));
      const parentId = a.parent_id === b.parent_id ? a.parent_id : null;
      const parent = parentId === null ? null : this.requireNode(parentId);
      const parentSteering = parent === null ? null : nodeSteering(parent);
      const aSteering = nodeSteering(a);
      const bSteering = nodeSteering(b);
      return {
        a_id: a.id,
        b_id: b.id,
        parent_id: parentId,
        a_text: a.text,
        b_text: b.text,
        a_applied_steering: aSteering,
        b_applied_steering: bSteering,
        parent_applied_steering: parentSteering,
        steering_delta: this.requireSteeringDelta(aSteering, bSteering),
        parent_to_a_delta: parentSteering !== null || aSteering !== null
          ? this.requireSteeringDelta(parentSteering, aSteering)
          : "",
        parent_to_b_delta: parentSteering !== null || bSteering !== null
          ? this.requireSteeringDelta(parentSteering, bSteering)
          : "",
        text: browserTextDiff(a.text, b.text),
        readings: browserReadingsDiff(a.aggregate_readings, b.aggregate_readings)
          .map(roundReading),
        per_token: a.tokens?.length && b.tokens?.length
          ? browserPerTokenDiff(a.tokens, b.tokens)
          : [],
      };
    }
    if (method === "jointLogprobs") {
      return this.jointLogprobs(
        requireIdentifier(args[0], "first node id"),
        requireIdentifier(args[1], "second node id"),
      );
    }
    if (method === "cast") return { cast: effectiveCast(this.tree) };
    if (method === "castPut") {
      const label = requireCastLabel(args[0]);
      const body = requireRecord(args[1], "cast member");
      const allowed = new Set(["steering", "thinking", "seed", "notes"]);
      if (Object.keys(body).some((key) => !allowed.has(key))) {
        throw loomError("INVALID_CAST_MEMBER", "Cast member contains unknown fields");
      }
      const steering = nullableText(body.steering, "cast steering", 16 * 1024);
      if (steering) {
        try {
          parseSteeringExpression(steering);
        } catch (error) {
          throw loomError(
            "INVALID_STEERING",
            error instanceof Error ? error.message : String(error),
          );
        }
      }
      const thinking = nullableBoolean(body.thinking, "cast thinking");
      const seed = nullableSafeInteger(body.seed, "cast seed");
      const notes = body.notes === undefined
        ? ""
        : requireText(body.notes, "cast notes", 256 * 1024);
      const hasRecipe = steering !== null || thinking !== null || seed !== null;
      const member = {
        recipe: hasRecipe
          ? {
              steering,
              sampling: null,
              thinking,
              seed,
              probes: [],
              probe_hashes: {},
            }
          : null,
        notes,
      };
      const current = this.tree.cast[label];
      if (JSON.stringify(current) !== JSON.stringify(member)) {
        this.tree.cast[label] = member;
        this.bumpRevision();
      }
      return { label, member: { ...clone(member), origin: "configured" } };
    }
    if (method === "castDelete") {
      const label = requireCastLabel(args[0]);
      if (this.tree.cast[label] !== undefined) {
        delete this.tree.cast[label];
        this.bumpRevision();
      }
      return undefined;
    }
    throw loomError("BROWSER_SERVICE_UNAVAILABLE", `tree.${method} is not implemented`);
  }

  private async resolveTokenFork(request: WSGenerateRequest): Promise<TokenForkContext | null> {
    const fields = [
      request.fork_node_id,
      request.fork_raw_index,
      request.fork_alt_token_id,
      request.fork_replacement_text,
    ];
    if (fields.every((value) => value === null || value === undefined)) return null;
    if (!this.runtimeCapabilities?.forcedReplay) {
      throw loomError(
        "TOKEN_FORK_UNAVAILABLE",
        "The loaded browser model runtime does not support exact forced-token replay",
      );
    }
    const source = this.requireNode(requireIdentifier(request.fork_node_id, "fork node id"));
    const rawIndex = requireNonnegativeInteger(request.fork_raw_index, "fork raw token index");
    if (!source.raw_token_ids || source.raw_token_ids.length === 0) {
      throw loomError(
        "TOKEN_FORK_UNAVAILABLE",
        `Conversation node ${source.id} has no exact raw token record`,
      );
    }
    if (rawIndex >= source.raw_token_ids.length) {
      throw loomError(
        "INVALID_NODE_OPERATION",
        `Raw token index ${rawIndex} is outside node ${source.id}`,
      );
    }
    if (source.parent_id === null) {
      throw loomError(
        "INVALID_NODE_OPERATION",
        `Conversation node ${source.id} has no parent for a sibling fork`,
      );
    }
    if (source.role !== "user" && source.role !== "assistant") {
      throw loomError(
        "INVALID_NODE_OPERATION",
        `Conversation node ${source.id} is not a generated user or assistant turn`,
      );
    }
    if (source.recipe === null || source.recipe.sampling === null) {
      throw loomError(
        "TOKEN_FORK_UNAVAILABLE",
        `Conversation node ${source.id} has no reproducible generation recipe`,
      );
    }
    let replacementIds: number[];
    if (request.fork_replacement_text !== null && request.fork_replacement_text !== undefined) {
      if (this.runtimeCapabilities.tokenizer !== true || !this.generation.tokenizeText) {
        throw loomError(
          "TOKEN_FORK_TOKENIZER_UNAVAILABLE",
          "The loaded browser model cannot tokenize a typed replacement",
        );
      }
      replacementIds = await this.generation.tokenizeText(request.fork_replacement_text);
      validateRuntimeTokenIds(replacementIds, "replacement tokenizer result");
      if (replacementIds.length === 0) {
        throw loomError(
          "INVALID_GENERATION_REQUEST",
          "Replacement text must produce at least one model token",
        );
      }
    } else {
      replacementIds = [requireNonnegativeInteger(
        request.fork_alt_token_id,
        "fork alternative token id",
      )];
    }
    const forcedPrefix = [
      ...source.raw_token_ids.slice(0, rawIndex),
      ...replacementIds,
    ];
    let sampling = effectiveSampling(
      this.session,
      recipeSampling(source.recipe.sampling),
      this.maxOutputTokens,
    );
    if (source.recipe.seed !== null && sampling.seed === null) {
      sampling = { ...sampling, seed: source.recipe.seed };
    }
    sampling = {
      ...sampling,
      ...(request.fork_seed == null ? {} : { seed: request.fork_seed }),
      max_tokens: forcedPrefix.length +
        (sampling.max_tokens ?? this.session.config.max_tokens ?? 512),
      ...(source.role === "user"
        ? { user_role: source.role_label }
        : { assistant_role: source.role_label }),
    };
    return {
      source,
      parentId: source.parent_id,
      forcedPrefix,
      settings: {
        steering: source.recipe.steering?.trim() || null,
        sampling,
        thinking: source.recipe.thinking ?? false,
        systemPrompt: source.recipe.system_prompt === undefined ? this.session.config.system_prompt : source.recipe.system_prompt,
      },
    };
  }

  private async replayTokenReadout(
    request: BrowserTokenReplayRequest,
    onProgress: (event: RuntimeProgressEvent) => void,
  ): Promise<import("../../lib/types").MeasurementsEnvelopeJSON> {
    return this.withReplayLock(async () => {
      const node = this.replayableNode(request.nodeId);
      if (request.rawIndex >= node.raw_token_ids!.length) {
        throw loomError(
          "INVALID_NODE_OPERATION",
          `Raw token index ${request.rawIndex} is outside node ${node.id}`,
        );
      }
      const forcedPrefix = node.raw_token_ids!.slice(0, request.rawIndex + 1);
      const settings = replaySettings(
        this.session,
        node,
        forcedPrefix.length,
        request.options.steered !== false,
        this.maxOutputTokens,
      );
      const total = forcedPrefix.length;
      const readoutLabel = request.family === "lens"
        ? "J-lens"
        : request.family === "sae"
          ? "model features"
          : "geometry";
      const emitProgress = (
        phase: "context" | "readout" | "complete",
        completed: number,
        message: string,
      ): void => {
        onProgress({
          event: "progress",
          data: {
            kind: "token_readout",
            family: request.family,
            nodeId: request.nodeId,
            rawIndex: request.rawIndex,
            phase,
            completed,
            total,
            progress: phase === "complete"
              ? 1
              : phase === "readout"
                ? 0.9
                : total === 0
                  ? 0
                  : 0.85 * completed / total,
            message,
          },
        });
      };
      emitProgress("context", 0, "Preparing the conversation context");
      const hookProgram = await this.compileRequiredSteering(settings.steering ?? "");
      let measurement: import("../../lib/types").MeasurementsEnvelopeJSON | null = null;
      await this.generation.streamGeneration({
        input: generationInput(
          { type: "generate", input: null, raw: this.session.is_base_model || request.options.raw || false },
          this.tree,
          node.parent_id!,
          settings.systemPrompt,
        ),
        sampling: settings.sampling,
        readoutTopK: request.options.topK,
        thinking: settings.thinking,
        generationSeat: this.session.is_base_model || request.options.raw ? null : node.role as ChatRole,
        generationRoleName: this.session.is_base_model || request.options.raw ? null : node.role_label,
        steeringExpression: settings.steering,
        hookProgram,
        maxOutputTokens: this.maxOutputTokens,
        replay: { forcedPrefixTokenIds: forcedPrefix },
        measurementTargetRawIndex: request.rawIndex,
        onRawTokenStart: (rawIndex) => {
          if (rawIndex === request.rawIndex) {
            emitProgress("readout", rawIndex, `Reading ${readoutLabel}`);
          }
        },
        onRawToken: (token) => {
          if (token.rawIndex === request.rawIndex && token.measurements !== undefined) {
            measurement = clone(token.measurements);
          }
          if (token.rawIndex < request.rawIndex) {
            emitProgress(
              "context",
              token.rawIndex + 1,
              `Preparing context · ${token.rawIndex + 1} of ${total}`,
            );
          }
        },
      }, () => undefined);
      if (measurement === null) {
        throw loomError(
          "TOKEN_REPLAY_UNAVAILABLE",
          `The browser replay did not return a measurement for raw token ${request.rawIndex}`,
        );
      }
      emitProgress("complete", total, `${readoutLabel} ready`);
      return measurement;
    });
  }

  private async jointLogprobs(
    aId: string,
    bId: string,
  ): Promise<import("../../lib/types").JointLogprobsJSON> {
    if (aId === bId) {
      throw loomError("INVALID_RUNTIME_REQUEST", "Compared conversation nodes must differ");
    }
    if (
      !this.runtimeCapabilities?.forcedReplay ||
      !this.runtimeCapabilities.replayScoring
    ) {
      throw loomError(
        "JOINT_LOGPROBS_UNAVAILABLE",
        "The loaded browser model runtime does not support exact compact replay scoring",
      );
    }
    return this.withReplayLock(async () => {
      const a = this.replayableNode(aId);
      const b = this.replayableNode(bId);
      if (a.parent_id !== b.parent_id) {
        throw loomError(
          "JOINT_LOGPROB_PARENT_MISMATCH",
          "Joint token probabilities require generated sibling branches",
        );
      }
      const aTokens = a.tokens ?? [];
      const bTokens = b.tokens ?? [];
      if (aTokens.length === 0 && bTokens.length === 0) {
        return { a_id: aId, b_id: bId, parent_id: a.parent_id, rows: [], n_rank1_changed: 0 };
      }
      const spans = browserPerTokenDiff(aTokens, bTokens);
      const chosenIds = new Set<number>();
      for (const node of [a, b]) {
        for (const row of node.tokens ?? []) {
          if (row.token_id !== undefined && row.token_id !== null) chosenIds.add(row.token_id);
        }
      }
      const firstIds = [...chosenIds].sort((left, right) => left - right);
      const firstA = await this.replayNodeScores(a, firstIds);
      const firstB = await this.replayNodeScores(b, firstIds);
      const scoreIdsA = new Set(firstIds);
      const scoreIdsB = new Set(firstIds);
      for (const span of spans) {
        if (!span.aligned || span.a_index < 0 || span.b_index < 0) continue;
        const rawA = aTokens[span.a_index]?.raw_index;
        const rawB = bTokens[span.b_index]?.raw_index;
        if (rawA === undefined || rawA === null || rawB === undefined || rawB === null) continue;
        for (const row of firstB.get(rawB)?.topLogprobs ?? []) scoreIdsA.add(row.tokenId);
        for (const row of firstA.get(rawA)?.topLogprobs ?? []) scoreIdsB.add(row.tokenId);
      }
      const scoresA = await this.replayNodeScores(
        a,
        [...scoreIdsA].sort((left, right) => left - right),
      );
      const scoresB = await this.replayNodeScores(
        b,
        [...scoreIdsB].sort((left, right) => left - right),
      );
      const rows = spans.map((span) => {
        const aToken = span.a_index < 0 ? undefined : aTokens[span.a_index];
        const bToken = span.b_index < 0 ? undefined : bTokens[span.b_index];
        const scoreA = aToken?.raw_index === undefined || aToken.raw_index === null
          ? undefined
          : scoresA.get(aToken.raw_index);
        const scoreB = bToken?.raw_index === undefined || bToken.raw_index === null
          ? undefined
          : scoresB.get(bToken.raw_index);
        const aTokenId = aToken?.token_id;
        const bTokenId = bToken?.token_id;
        let rankChanged = false;
        let approxKl: number | null = null;
        let lpAInB: number | null = null;
        let lpBInA: number | null = null;
        if (
          span.aligned && scoreA !== undefined && scoreB !== undefined &&
          aTokenId !== undefined && aTokenId !== null &&
          bTokenId !== undefined && bTokenId !== null
        ) {
          rankChanged = scoreA.argmax.tokenId !== scoreB.argmax.tokenId;
          lpAInB = replayLogprob(scoreB, aTokenId);
          lpBInA = replayLogprob(scoreA, bTokenId);
          approxKl = approximateReplayKl(scoreA, scoreB);
        }
        return {
          a_index: span.a_index,
          b_index: span.b_index,
          a_text: span.a_text,
          b_text: span.b_text,
          aligned: span.aligned,
          lp_a_in_a: scoreA === undefined || aTokenId === undefined || aTokenId === null
            ? null
            : replayLogprob(scoreA, aTokenId),
          lp_b_in_b: scoreB === undefined || bTokenId === undefined || bTokenId === null
            ? null
            : replayLogprob(scoreB, bTokenId),
          lp_a_in_b: lpAInB,
          lp_b_in_a: lpBInA,
          rank_changed: rankChanged,
          approx_kl: approxKl,
        };
      });
      return {
        a_id: aId,
        b_id: bId,
        parent_id: a.parent_id,
        rows,
        n_rank1_changed: rows.filter((row) => row.aligned && row.rank_changed).length,
      };
    });
  }

  private async replayNodeScores(
    node: LoomNodeJSON,
    scoreTokenIds: readonly number[],
  ): Promise<Map<number, WebLlmReplayScore>> {
    const forcedPrefix = [...node.raw_token_ids!];
    const settings = replaySettings(
      this.session,
      node,
      forcedPrefix.length,
      true,
      this.maxOutputTokens,
    );
    const hookProgram = settings.steering
      ? await this.compileRequiredSteering(settings.steering)
      : null;
    const scores = new Map<number, WebLlmReplayScore>();
    await this.generation.streamGeneration({
      input: generationInput(
        { type: "generate", input: null, raw: this.session.is_base_model === true },
        this.tree,
        node.parent_id!,
        settings.systemPrompt,
      ),
      sampling: settings.sampling,
      thinking: settings.thinking,
      generationSeat: this.session.is_base_model ? null : node.role as ChatRole,
      generationRoleName: this.session.is_base_model ? null : node.role_label,
      steeringExpression: settings.steering,
      hookProgram,
      maxOutputTokens: this.maxOutputTokens,
      replay: { forcedPrefixTokenIds: forcedPrefix, scoreTokenIds },
      onRawToken: (token) => {
        if (token.replayScore !== undefined) scores.set(token.rawIndex, token.replayScore);
      },
    }, () => undefined);
    if (scores.size !== forcedPrefix.length) {
      throw loomError(
        "JOINT_LOGPROBS_UNAVAILABLE",
        `Compact replay returned ${scores.size} of ${forcedPrefix.length} token score rows`,
      );
    }
    return scores;
  }

  private replayableNode(nodeId: string): LoomNodeJSON {
    const node = this.requireNode(nodeId);
    if (node.parent_id === null || !node.raw_token_ids || node.raw_token_ids.length === 0) {
      throw loomError(
        "TOKEN_REPLAY_UNAVAILABLE",
        `Conversation node ${node.id} has no exact raw token record`,
      );
    }
    if (node.role !== "user" && node.role !== "assistant") {
      throw loomError(
        "TOKEN_REPLAY_UNAVAILABLE",
        `Conversation node ${node.id} is not a generated user or assistant turn`,
      );
    }
    if (node.recipe === null || node.recipe.sampling === null) {
      throw loomError(
        "TOKEN_REPLAY_UNAVAILABLE",
        `Conversation node ${node.id} has no reproducible generation recipe`,
      );
    }
    return node;
  }

  private async withReplayLock<T>(operation: () => Promise<T>): Promise<T> {
    if (this.generating) {
      throw loomError("GENERATION_BUSY", "The browser model is already generating or replaying");
    }
    this.generating = true;
    try {
      return await operation();
    } finally {
      this.generating = false;
    }
  }

  private compileRequiredSteering(
    expression: string,
    probeRequests?: readonly ProbeRequest[],
  ): Promise<BrowserHookProgram> {
    if (!this.compileSteering) {
      throw loomError(
        "STEERING_COMPILER_UNAVAILABLE",
        "The selected browser model has no installed compiler for this steering expression",
      );
    }
    return Promise.resolve(this.compileSteering(expression, probeRequests));
  }

  private async preflightGenerationSteering(
    expression: string,
    probeRequests?: readonly ProbeRequest[],
  ): Promise<BrowserHookProgram> {
    const program = await this.compileRequiredSteering(expression, probeRequests);
    const activeRole = activeAssistantRole(program);
    if (activeRole !== null && !this.session.role_substitution_supported) {
      throw loomError(
        "ROLE_SUBSTITUTION_UNAVAILABLE",
        `Steering requires assistant role ${JSON.stringify(activeRole)}, but this browser model does not support named assistant roles`,
      );
    }
    return program;
  }

  private tokenMeasurements(
    nodeId: string,
    rawIndex: number,
  ): import("../../lib/types").MeasurementsEnvelopeJSON | null {
    const node = this.requireNode(nodeId);
    for (const row of [...(node.thinking_tokens ?? []), ...(node.tokens ?? [])]) {
      if (row.raw_index === rawIndex && row.measurements !== undefined) {
        return clone(row.measurements);
      }
    }
    return null;
  }

  private async importTranscript(
    transcript: BrowserTranscript,
    mode: "default" | "here" | "merge",
    strict: boolean,
  ): Promise<{ leaf_id: string; rev: number; guards: string[] }> {
    const original = this.tree;
    this.tree = clone(this.tree);
    try {
      const guards = transcriptGuards(
        transcript,
        this.session,
        mode,
        strict,
        await this.resolveProbeHashes(),
      );
      const castConflicts: string[] = [];
      for (const [label, member] of Object.entries(transcript.cast)) {
        const existing = this.tree.cast[label];
        if (existing === undefined) {
          this.tree.cast[label] = clone(member);
          this.bumpRevision();
        } else if (JSON.stringify(existing) !== JSON.stringify(member)) {
          castConflicts.push(label);
        }
      }
      if (castConflicts.length > 0) {
        guards.push(
          `cast_conflict: ${JSON.stringify(castConflicts.sort())}`,
        );
      }
      const attachParent = mode === "default"
        ? this.tree.root_id
        : mode === "here"
          ? this.tree.active_node_id
          : transcriptMergeAnchor(this.tree, transcript);
      const skipUsers = attachParent === this.tree.root_id
        ? 0
        : matchingTranscriptUsers(this.tree, attachParent, transcript);
      let usersSeen = 0;
      let currentParent = attachParent;
      let leafId = attachParent;
      let firstImportedId: string | null = null;
      for (const turn of transcript.turns) {
        if (turn.role === "system") continue;
        if (turn.role === "user") {
          usersSeen += 1;
          if (usersSeen <= skipUsers) continue;
        } else if (usersSeen < skipUsers) {
          continue;
        }
        const node = this.newNode({
          parentId: currentParent,
          role: turn.role,
          text: turn.text,
          roleLabel: turn.speaker,
          thinkingText: turn.thinking,
          steering: turn.recipe?.steering ?? null,
          recipe: clone(turn.recipe),
        });
        node.aggregate_readings = clone(turn.readings);
        node.tokens = null;
        node.thinking_tokens = null;
        if (firstImportedId === null && guards.length > 0) {
          node.notes = guards.join("\n");
        }
        this.addNode(node);
        firstImportedId ??= node.id;
        currentParent = node.id;
        leafId = node.id;
      }
      if (firstImportedId === null && guards.length > 0) {
        const target = this.requireNode(leafId);
        target.notes = guards.join("\n");
        this.replaceNodeWithoutActivation(target);
      }
      validateHostedLoomTree(this.tree);
      this.syncHistoryLength();
      return { leaf_id: leafId, rev: this.tree.rev, guards };
    } catch (error) {
      this.tree = original;
      this.syncHistoryLength();
      throw error;
    }
  }

  private requireSteeringDelta(parent: string | null, child: string | null): string {
    const normalizedParent = parent?.trim() || null;
    const normalizedChild = child?.trim() || null;
    if (normalizedParent === normalizedChild) return "";
    if (this.steeringDelta) return this.steeringDelta(normalizedParent, normalizedChild);
    throw loomError(
      "STEERING_DELTA_UNAVAILABLE",
      "This comparison needs the installed core pack to resolve steering aliases",
    );
  }

  private async resolveProbeHashes(): Promise<Record<string, string>> {
    if (this.probeHashes === undefined || this.session.probes.length === 0) return {};
    const value = await this.probeHashes();
    if (!isPlainRecord(value)) {
      throw loomError(
        "PROBE_HASH_PROVIDER_INVALID",
        "The browser probe identity provider returned an invalid hash roster",
      );
    }
    const installed = new Set(this.session.probes);
    const hashes: Record<string, string> = {};
    for (const [name, digest] of Object.entries(value)) {
      if (!installed.has(name)) continue;
      if (typeof digest !== "string" || digest.length === 0 || digest.length > 512) {
        throw loomError(
          "PROBE_HASH_PROVIDER_INVALID",
          `The browser probe identity for ${JSON.stringify(name)} is invalid`,
        );
      }
      hashes[name] = digest;
    }
    return hashes;
  }

  private async appendAuthoredTurn(input: {
    parentId: string;
    role: ChatRole;
    text: string;
    roleLabel: string | null;
    thinkingText: string | null;
  }): Promise<{ node: LoomNodeJSON; mutation: WSServerMessage }> {
    const parent = this.requireNode(input.parentId);
    if (parent.role !== input.role || parent.role_label !== input.roleLabel ||
        (this.tree.children_of[parent.id]?.length ?? 0) > 0) {
      const node = this.newNode({
        parentId: input.parentId,
        role: input.role,
        text: input.text,
        roleLabel: input.roleLabel,
        thinkingText: input.thinkingText,
        steering: null,
        recipe: null,
      });
      this.addNode(node);
      return { node, mutation: this.addedMutation("add_user", node) };
    }

    const node = clone(parent);
    node.text += input.text;
    if (input.thinkingText !== null) {
      node.thinking_text = (node.thinking_text ?? "") + input.thinkingText;
    }
    node.tokens = null;
    node.thinking_tokens = null;
    node.recipe = null;
    node.aggregate_readings = {};
    node.applied_steering = null;
    node.finish_reason = "stop";
    node.mean_logprob = null;
    node.mean_surprise = null;
    node.raw_token_ids = await this.authoredTokenIds(node.text);
    node.edit_count += 1;
    node.edited_at = this.now();
    this.tree.active_node_id = node.id;
    this.replaceNodeWithoutActivation(node);
    return {
      node,
      mutation: this.updatedMutation("edit", node, node.id),
    };
  }

  private async authoredTokenIds(text: string): Promise<number[] | null> {
    if (this.runtimeCapabilities?.tokenizer !== true || !this.generation.tokenizeText) {
      return null;
    }
    const tokenIds = await this.generation.tokenizeText(text);
    validateRuntimeTokenIds(tokenIds, "authored message tokenizer result");
    return [...tokenIds];
  }

  private async prepareContinuation(input: {
    eligible: boolean;
    nodeId: string;
    role: ChatRole;
    roleLabel: string | null;
  }): Promise<ContinuationContext | null> {
    if (!input.eligible) return null;
    const candidate = this.requireNode(input.nodeId);
    if (candidate.role !== input.role || candidate.role_label !== input.roleLabel) return null;
    if (this.runtimeCapabilities?.forcedReplay !== true) {
      throw loomError(
        "CONTINUATION_REPLAY_UNAVAILABLE",
        "The loaded browser model cannot replay an existing message as an exact continuation",
      );
    }
    let forcedPrefix: number[];
    const exactSavedText = candidate.thinking_text === null && candidate.raw_token_ids !== null &&
      this.generation.decodeTokens !== undefined &&
      await this.generation.decodeTokens(candidate.raw_token_ids) === candidate.text;
    if (exactSavedText) {
      forcedPrefix = [...candidate.raw_token_ids!];
      validateRuntimeTokenIds(forcedPrefix, "continuation raw token record");
    } else if (this.runtimeCapabilities.tokenizer && this.generation.tokenizeText) {
      forcedPrefix = await this.generation.tokenizeText(candidate.text);
      validateRuntimeTokenIds(forcedPrefix, "continuation tokenizer result");
    } else if (candidate.thinking_text === null && candidate.raw_token_ids !== null) {
      forcedPrefix = [...candidate.raw_token_ids];
      validateRuntimeTokenIds(forcedPrefix, "continuation raw token record");
    } else {
      throw loomError(
        "CONTINUATION_TOKENIZER_UNAVAILABLE",
        "The loaded browser model cannot tokenize the existing message for exact continuation",
      );
    }
    if (candidate.parent_id === null) {
      throw loomError(
        "INVALID_NODE_OPERATION",
        "The conversation root cannot be continued",
      );
    }
    return {
      node: candidate,
      original: clone(candidate),
      parentId: candidate.parent_id,
      forcedPrefix: [...forcedPrefix],
    };
  }

  private newNode(input: {
    parentId: string;
    role: ChatRole;
    text: string;
    roleLabel: string | null;
    thinkingText: string | null;
    steering: string | null;
    recipe: RecipeJSON | null;
  }): LoomNodeJSON {
    return {
      id: this.uniqueId("node"),
      parent_id: input.parentId,
      role: input.role,
      text: input.text,
      role_label: input.roleLabel,
      thinking_text: input.thinkingText,
      aggregate_readings: {},
      applied_steering: input.steering,
      finish_reason: null,
      starred: false,
      notes: "",
      created_at: this.now(),
      edited_at: null,
      edit_count: 0,
      mean_logprob: null,
      mean_surprise: null,
      recipe: input.recipe,
      tokens: [],
      thinking_tokens: [],
      raw_token_ids: null,
    };
  }

  private addNode(node: LoomNodeJSON): void {
    this.requireNode(node.parent_id!);
    this.tree.nodes.push(clone(node));
    this.tree.children_of[node.id] = [];
    this.tree.children_of[node.parent_id!]!.push(node.id);
    this.tree.active_node_id = node.id;
    this.bumpRevision();
  }

  private replaceGeneratedNode(node: LoomNodeJSON): LoomNodeJSON {
    const index = this.tree.nodes.findIndex((candidate) => candidate.id === node.id);
    if (index < 0) throw loomError("LOOM_NODE_NOT_FOUND", `Conversation node ${node.id} does not exist`);
    const current = this.tree.nodes[index]!;
    const finalized = {
      ...clone(node),
      starred: current.starred,
      notes: current.notes,
    };
    this.tree.nodes[index] = finalized;
    this.bumpRevision();
    return clone(finalized);
  }

  private replaceNodeWithoutActivation(node: LoomNodeJSON): void {
    const index = this.tree.nodes.findIndex((candidate) => candidate.id === node.id);
    if (index < 0) throw loomError("LOOM_NODE_NOT_FOUND", `Conversation node ${node.id} does not exist`);
    this.tree.nodes[index] = clone(node);
    this.bumpRevision();
  }

  private removeLeaf(nodeId: string): void {
    const node = this.requireNode(nodeId);
    if (this.tree.children_of[nodeId]!.length > 0 || node.parent_id === null) {
      throw loomError("LOOM_DELETE_CONFLICT", "Only an unfinished leaf can be rolled back");
    }
    this.tree.nodes = this.tree.nodes.filter((candidate) => candidate.id !== nodeId);
    delete this.tree.children_of[nodeId];
    this.tree.children_of[node.parent_id] = this.tree.children_of[node.parent_id]!.filter(
      (childId) => childId !== nodeId,
    );
    if (this.tree.active_node_id === nodeId) this.tree.active_node_id = node.parent_id;
    this.bumpRevision();
  }

  private removeSubtree(nodeId: string): number {
    if (nodeId === this.tree.root_id) {
      throw loomError("LOOM_DELETE_CONFLICT", "The conversation root cannot be deleted");
    }
    const root = this.requireNode(nodeId);
    this.assertReservedNodeMutationAllowed(nodeId, "delete_subtree");
    const removed = new Set<string>();
    const pending = [nodeId];
    while (pending.length > 0) {
      const current = pending.pop()!;
      if (removed.has(current)) continue;
      removed.add(current);
      pending.push(...this.tree.children_of[current]!);
    }
    if (removed.has(this.tree.active_node_id)) this.tree.active_node_id = root.parent_id!;
    this.tree.children_of[root.parent_id!] = this.tree.children_of[root.parent_id!]!.filter(
      (childId) => childId !== nodeId,
    );
    this.tree.nodes = this.tree.nodes.filter((node) => !removed.has(node.id));
    for (const removedId of removed) delete this.tree.children_of[removedId];
    this.bumpRevision();
    return removed.size;
  }

  private resetTree(): void {
    const root = rootNode(this.uniqueId("node"), this.now());
    this.tree = {
      ...this.tree,
      rev: this.tree.rev + 1,
      root_id: root.id,
      active_node_id: root.id,
      nodes: [root],
      children_of: { [root.id]: [] },
    };
    this.syncHistoryLength();
  }

  private setReservationRoot(nodeId: string): void {
    if (this.generationReservation !== null) {
      this.generationReservation.rootId = nodeId;
    }
  }

  private assertWholeTreeMutationAllowed(op: "reset" | "restore"): void {
    if (!this.generating) return;
    throw loomError(
      "MUTATION_DURING_GENERATION",
      `cannot ${op} tree while a generation is in flight`,
    );
  }

  private assertReservedNodeMutationAllowed(nodeId: string, op: string): void {
    const reservation = this.generationReservation;
    if (reservation === null) return;
    const rootId = reservation.rootId;
    const intersects = rootId === null
      ? nodeId === reservation.anchorId || this.isAncestor(nodeId, reservation.anchorId)
      : nodeId === rootId || this.isAncestor(rootId, nodeId) || this.isAncestor(nodeId, rootId);
    if (!intersects) return;
    throw loomError(
      "MUTATION_DURING_GENERATION",
      `cannot ${op} on a node inside an in-flight generation's reservation ` +
        `(reservation root: ${rootId ?? reservation.anchorId})`,
    );
  }

  private isAncestor(ancestorId: string, nodeId: string): boolean {
    let current = this.requireNode(nodeId);
    while (current.parent_id !== null) {
      if (current.parent_id === ancestorId) return true;
      current = this.requireNode(current.parent_id);
    }
    return false;
  }

  private addedMutation(op: string, node: LoomNodeJSON): WSServerMessage {
    return {
      type: "tree_mutated",
      op,
      added: [clone(node)],
      removed: [],
      updated: [],
      active_node_id: this.tree.active_node_id,
      rev: this.tree.rev,
      cast: effectiveCast(this.tree),
    };
  }

  private updatedMutation(
    op: string,
    node: LoomNodeJSON,
    activeNodeId: string | null = null,
  ): WSServerMessage {
    return {
      type: "tree_mutated",
      op,
      added: [],
      removed: [],
      updated: [clone(node)],
      active_node_id: activeNodeId,
      rev: this.tree.rev,
      cast: effectiveCast(this.tree),
    };
  }

  private serviceMutation(
    op: WSTreeMutatedEvent["op"],
    before: LoomTreeJSON,
  ): WSTreeMutatedEvent {
    const prior = new Map(before.nodes.map((node) => [node.id, node]));
    const current = new Map(this.tree.nodes.map((node) => [node.id, node]));
    const added = this.tree.nodes.filter((node) => !prior.has(node.id)).map(clone);
    const removed = before.nodes.filter((node) => !current.has(node.id)).map((node) => node.id);
    const updated = this.tree.nodes.filter((node) => {
      const previous = prior.get(node.id);
      return previous !== undefined && JSON.stringify(previous) !== JSON.stringify(node);
    }).map(clone);
    const replacement = op === "restore";
    return {
      type: "tree_mutated",
      op,
      added,
      removed,
      updated,
      active_node_id: replacement || before.active_node_id !== this.tree.active_node_id
        ? this.tree.active_node_id
        : null,
      rev: this.tree.rev,
      cast: effectiveCast(this.tree),
    };
  }

  private requireNode(nodeId: string): LoomNodeJSON {
    const node = this.tree.nodes.find((candidate) => candidate.id === nodeId);
    if (!node) throw loomError("LOOM_NODE_NOT_FOUND", `Conversation node ${nodeId} does not exist`);
    return node;
  }

  private requireEditableNode(value: unknown): LoomNodeJSON {
    const node = this.requireNode(requireIdentifier(value, "node id"));
    if (node.id === this.tree.root_id) {
      throw loomError("LOOM_EDIT_CONFLICT", "The conversation root cannot be edited or branched");
    }
    return node;
  }

  private bumpRevision(): void {
    this.tree.rev += 1;
    this.syncHistoryLength();
  }

  private syncHistoryLength(): void {
    const historyLength = pathNodes(this.tree, this.tree.active_node_id)
      .filter((node) => node.id !== this.tree.root_id).length;
    this.session = { ...this.session, history_length: historyLength };
  }

  private syncInstrumentSession(): void {
    if (!this.instruments) return;
    const blocks = this.instruments.blocks();
    this.session = {
      ...this.session,
      profiles: this.instruments.profileNames(),
      probes: blocks.flatMap((block) => block.probes),
      jlens_fitted: this.instruments.jlensFitted(),
      instruments: blocks,
    };
  }

  private uniqueId(kind: "generation" | "node"): string {
    for (let attempt = 0; attempt < 32; attempt += 1) {
      const id = this.createId(kind);
      if (
        typeof id === "string" && id.length > 0 && id.length <= 128 &&
        (kind === "generation" || !this.tree.nodes.some((node) => node.id === id))
      ) return id;
    }
    throw loomError("LOOM_ID_COLLISION", `Could not allocate a unique ${kind} id`);
  }
}

function savedTokenPrefix(source: LoomNodeJSON, forced: readonly number[]) {
  let length = 0;
  const original = source.raw_token_ids ?? [];
  while (length < forced.length && original[length] === forced[length]) length += 1;
  const rows = [...(source.tokens ?? []), ...(source.thinking_tokens ?? [])];
  if (rows.length === 0 || rows.some((row) => row.raw_index == null)) length = 0;
  const keep = (row: NonNullable<LoomNodeJSON["tokens"]>[number]) =>
    row.raw_index != null && row.raw_index < length;
  const tokens = (source.tokens ?? []).filter(keep);
  const thinking_tokens = (source.thinking_tokens ?? []).filter(keep);
  return {
    length,
    text: tokens.map((row) => row.text).join(""),
    thinking_text: thinking_tokens.length === 0 ? null : thinking_tokens.map((row) => row.text).join(""),
    tokens,
    thinking_tokens,
  };
}

function generationInput(
  request: WSSubmitRequest | WSGenerateRequest,
  tree: LoomTreeJSON,
  parentId: string,
  systemPrompt: string | null,
  stateless = false,
  stringRoleLabel: string | null = null,
): WebLlmGenerationPlan["input"] {
  if (usesRawCompletionInput(request)) {
    if (request.type === "generate" && stateless) {
      return {
        kind: "raw",
        prompt: typeof request.input === "string" ? request.input : "",
      };
    }
    return { kind: "raw", prompt: pathNodes(tree, parentId).map((node) => node.text).join("") };
  }
  let messages: WSInputMessage[];
  if (request.type === "generate" && Array.isArray(request.input)) {
    messages = request.input.map((message) => ({ ...message }));
  } else if (request.type === "generate" && stateless) {
    messages = typeof request.input === "string"
      ? [{ role: "user", content: request.input, label: stringRoleLabel }]
      : [];
  } else {
    messages = pathNodes(tree, parentId)
      .filter((node) => node.id !== tree.root_id)
      .map((node) => ({ role: node.role, content: node.text, label: node.role_label }));
  }
  if (systemPrompt) {
    messages.unshift({ role: "system", content: systemPrompt });
  }
  if (messages.length === 0) {
    throw loomError("INVALID_GENERATION_INPUT", "Chat generation requires conversation text");
  }
  return {
    kind: "chat",
    messages: messages.map(({ role, content, label }) => ({
      role,
      content,
      ...(label?.trim() ? { name: label.trim() } : {}),
    })),
  };
}

function cancelledGenerationResult(
  node: LoomNodeJSON,
  rawTokenCount: number,
  promptTokens = 0,
): WebLlmGenerationResult {
  const responseRows = node.tokens ?? [];
  const logprobs = responseRows
    .map((row) => row.logprob)
    .filter((value): value is number => value !== null && Number.isFinite(value));
  const meanLogprob = logprobs.length === 0 || logprobs.length !== responseRows.length
    ? null
    : logprobs.reduce((sum, value) => sum + value, 0) / logprobs.length;
  const allRows = [...(node.thinking_tokens ?? []), ...responseRows];
  const latestMeasurements = allRows.findLast((row) => row.measurements !== undefined)
    ?.measurements;
  const tokens = rawTokenCount > 0 ? rawTokenCount : allRows.length;
  return {
    text: node.text,
    thinkingText: node.thinking_text,
    tokens,
    finishReason: "cancelled",
    terminalReason: "external_stop",
    usage: {
      promptTokens,
      completionTokens: tokens,
      totalTokens: promptTokens + tokens,
    },
    meanLogprob,
    meanSurprise: meanLogprob === null ? null : -meanLogprob,
    prefillTokensPerSecond: null,
    decodeTokensPerSecond: null,
    ...(latestMeasurements === undefined
      ? {}
      : {
          measurements: {
            ...clone(latestMeasurements),
            scope: "aggregate" as const,
          },
        }),
  };
}

function pathNodes(tree: LoomTreeJSON, leafId: string): LoomNodeJSON[] {
  const byId = new Map(tree.nodes.map((node) => [node.id, node]));
  const path: LoomNodeJSON[] = [];
  let current = byId.get(leafId);
  while (current) {
    path.push(current);
    current = current.parent_id === null ? undefined : byId.get(current.parent_id);
  }
  path.reverse();
  return path;
}

function serviceMutationOp(method: string): WSTreeMutatedEvent["op"] | null {
  if (method === "transcriptLoad") return "restore";
  if (method === "castPut" || method === "castDelete") return "cast";
  if (
    method === "branch" || method === "delete" || method === "edit" ||
    method === "navigate" || method === "note" || method === "restore" ||
    method === "reset" || method === "star"
  ) return method;
  return null;
}

function transcriptGuards(
  transcript: BrowserTranscript,
  session: SessionInfo,
  mode: "default" | "here" | "merge",
  strict: boolean,
  currentProbeHashes: Readonly<Record<string, string>>,
): string[] {
  const guards: string[] = [];
  if (transcript.modelId && transcript.modelId !== session.model_id) {
    const message = `transcript model ${JSON.stringify(transcript.modelId)} differs from session model ${JSON.stringify(session.model_id)}`;
    if (mode === "merge") {
      throw loomError(
        "TRANSCRIPT_MODEL_MISMATCH",
        `${message}; refusing to merge under semantic mismatch`,
      );
    }
    guards.push(`model_mismatch: ${message}`);
  }
  if (
    transcript.systemPrompt !== null && session.config.system_prompt !== null &&
    transcript.systemPrompt !== session.config.system_prompt
  ) {
    guards.push(
      `system_prompt_mismatch: original was ${JSON.stringify(transcript.systemPrompt)}`,
    );
  }
  const installed = new Set(session.probes);
  const missing = transcript.probes
    .map((probe) => probe.name)
    .filter((name) => !installed.has(name))
    .sort();
  if (missing.length > 0) guards.push(`probes_missing: ${JSON.stringify(missing)}`);
  const drift = transcript.probes
    .filter((probe) =>
      installed.has(probe.name) && probe.sha256.length > 0 &&
      currentProbeHashes[probe.name] !== undefined &&
      currentProbeHashes[probe.name] !== probe.sha256
    )
    .map((probe) => probe.name)
    .sort();
  if (drift.length > 0) {
    const detail = `probe_drift: ${JSON.stringify(drift)}`;
    if (strict) {
      throw loomError(
        "TRANSCRIPT_PROBE_DRIFT",
        `Probe content drift between transcript and browser session: ${drift.join(", ")}`,
      );
    }
    guards.push(detail);
  }
  const unverifiable = transcript.probes
    .filter((probe) =>
      installed.has(probe.name) && probe.sha256.length > 0 &&
      currentProbeHashes[probe.name] === undefined
    )
    .map((probe) => probe.name)
    .sort();
  if (unverifiable.length > 0) {
    const detail = `probe_hash_unavailable: ${JSON.stringify(unverifiable)}`;
    if (strict) {
      throw loomError(
        "TRANSCRIPT_PROBE_HASH_UNAVAILABLE",
        `Strict import cannot verify browser probe hashes for: ${unverifiable.join(", ")}`,
      );
    }
    guards.push(detail);
  }
  return guards;
}

function transcriptMergeAnchor(tree: LoomTreeJSON, transcript: BrowserTranscript): string {
  const activeUsers = pathNodes(tree, tree.active_node_id).filter((node) => node.role === "user");
  const transcriptUsers = transcript.turns.filter((turn) => turn.role === "user");
  let anchor = tree.root_id;
  const length = Math.min(activeUsers.length, transcriptUsers.length);
  for (let index = 0; index < length; index += 1) {
    if (activeUsers[index].text !== transcriptUsers[index].text) break;
    anchor = activeUsers[index].id;
  }
  return anchor;
}

function matchingTranscriptUsers(
  tree: LoomTreeJSON,
  anchorId: string,
  transcript: BrowserTranscript,
): number {
  const anchorUsers = pathNodes(tree, anchorId).filter((node) => node.role === "user");
  const transcriptUsers = transcript.turns.filter((turn) => turn.role === "user");
  const length = Math.min(anchorUsers.length, transcriptUsers.length);
  let count = 0;
  for (let index = 0; index < length; index += 1) {
    if (anchorUsers[index].text !== transcriptUsers[index].text) break;
    count += 1;
  }
  return count;
}

function activePath(tree: LoomTreeJSON): {
  active_node_id: string;
  rev: number;
  messages: { role: string; content: string }[];
  node_ids: string[];
} {
  const nodes = pathNodes(tree, tree.active_node_id)
    .filter((node) => node.id !== tree.root_id);
  return {
    active_node_id: tree.active_node_id,
    rev: tree.rev,
    messages: nodes.map((node) => ({ role: node.role, content: node.text })),
    node_ids: nodes.map((node) => node.id),
  };
}

function emptyTree(session: SessionInfo, drowseVersion: string): LoomTreeJSON {
  const root = rootNode(`node-${randomUuid()}`, session.created);
  return {
    tree_format: 2,
    drowse_version: drowseVersion,
    model_id: session.model_id,
    session_id: session.id,
    name: null,
    rev: 0,
    root_id: root.id,
    active_node_id: root.id,
    nodes: [root],
    children_of: { [root.id]: [] },
    cast: {},
  };
}

function rootNode(id: string, createdAt: number): LoomNodeJSON {
  return {
    id,
    parent_id: null,
    role: "system",
    text: "",
    role_label: null,
    thinking_text: null,
    aggregate_readings: {},
    applied_steering: null,
    finish_reason: null,
    starred: false,
    notes: "",
    created_at: createdAt,
    edited_at: null,
    edit_count: 0,
    mean_logprob: null,
    mean_surprise: null,
    recipe: null,
    tokens: null,
    thinking_tokens: null,
    raw_token_ids: null,
  };
}

function recipe(
  steering: string | null,
  sampling: WSSampling,
  thinking: boolean,
  probes: readonly string[],
  probeHashes: Readonly<Record<string, string>>,
  systemPrompt?: string | null,
): RecipeJSON {
  const recipeSampling: RecipeSamplingJSON = {
    temperature: sampling.temperature ?? null,
    top_p: sampling.top_p ?? null,
    top_k: sampling.top_k ?? null,
    max_tokens: sampling.max_tokens ?? null,
    seed: sampling.seed ?? null,
    stop: sampling.stop ?? null,
    logit_bias: sampling.logit_bias ?? null,
    presence_penalty: sampling.presence_penalty ?? 0,
    frequency_penalty: sampling.frequency_penalty ?? 0,
    logprobs: 1,
    return_hidden: false,
    return_top_k: sampling.return_top_k ?? 0,
    user_role: sampling.user_role ?? null,
    assistant_role: sampling.assistant_role ?? null,
    persist_per_layer_scores: sampling.persist_per_layer_scores ?? false,
    persist_subspace_coords: sampling.persist_subspace_coords ?? false,
    return_probe_readings: sampling.return_probe_readings ?? true,
  };
  return {
    steering,
    sampling: recipeSampling,
    ...(systemPrompt === undefined ? {} : { system_prompt: systemPrompt }),
    thinking,
    seed: recipeSampling.seed,
    probes: [...probes],
    probe_hashes: Object.fromEntries(
      probes.flatMap((name) => probeHashes[name] === undefined
        ? []
        : [[name, probeHashes[name]]]),
    ),
  };
}

function replaySettings(
  session: SessionInfo,
  node: LoomNodeJSON,
  forcedTokenCount: number,
  steered: boolean,
  maxOutputTokens: number,
): ResolvedGenerationSettings {
  const stamped = node.recipe!;
  let sampling = effectiveSampling(
    session,
    recipeSampling(stamped.sampling!),
    maxOutputTokens,
  );
  if (stamped.seed !== null && sampling.seed === null) {
    sampling = { ...sampling, seed: stamped.seed };
  }
  sampling = {
    ...sampling,
    max_tokens: forcedTokenCount,
    ...(node.role === "user"
      ? { user_role: node.role_label }
      : { assistant_role: node.role_label }),
  };
  return {
    steering: steered ? stamped.steering?.trim() || null : null,
    sampling,
    thinking: stamped.thinking ?? false,
    systemPrompt: stamped.system_prompt === undefined ? session.config.system_prompt : stamped.system_prompt,
  };
}

function effectiveSampling(
  session: SessionInfo,
  value: WSSampling | null | undefined,
  maxOutputTokens: number,
): WSSampling {
  const requestedMaxTokens = value?.max_tokens ?? session.config.max_tokens;
  return {
    temperature: value?.temperature ?? session.config.temperature,
    top_p: value?.top_p ?? session.config.top_p,
    top_k: value?.top_k ?? session.config.top_k,
    max_tokens: typeof requestedMaxTokens === "number"
      ? clampOutputTokenCount(requestedMaxTokens, maxOutputTokens)
      : requestedMaxTokens,
    seed: value?.seed ?? null,
    stop: value?.stop ?? null,
    logit_bias: value?.logit_bias ?? null,
    presence_penalty: value?.presence_penalty ?? 0,
    frequency_penalty: value?.frequency_penalty ?? 0,
    return_top_k: value?.return_top_k ?? 0,
    return_probe_readings: value?.return_probe_readings ?? true,
    persist_per_layer_scores: value?.persist_per_layer_scores ?? false,
    persist_subspace_coords: value?.persist_subspace_coords ?? false,
    user_role: value?.user_role ?? null,
    assistant_role: value?.assistant_role ?? null,
  };
}

export function deriveSeedSchedule(baseSeed: number | null, count: number): Array<number | null> {
  if (!Number.isSafeInteger(count) || count < 1 || count > 32) {
    throw loomError("INVALID_GENERATION_REQUEST", "Generation sibling count is outside its supported range");
  }
  if (count === 1) return [baseSeed];
  const resolvedBase = baseSeed ?? (crypto.getRandomValues(new Uint32Array(1))[0] & 0x7fffffff);
  const maskedBase = BigInt(resolvedBase) & 0x7fffffffffffffffn;
  return Array.from({ length: count }, (_, index) => {
    const payload = new Uint8Array(16);
    const view = new DataView(payload.buffer);
    view.setBigInt64(0, maskedBase, true);
    view.setBigInt64(8, BigInt(index), true);
    const digest = blake2b(payload, { dkLen: 8 });
    const value = new DataView(
      digest.buffer,
      digest.byteOffset,
      digest.byteLength,
    ).getBigUint64(0, true);
    return Number(value & 0x7fffffffn);
  });
}

function roleLabel(role: ChatRole, sampling: WSSampling | null | undefined): string | null {
  if (role === "user") return sampling?.user_role?.trim() || null;
  return sampling?.assistant_role?.trim() || null;
}

function activeAssistantRole(program: BrowserHookProgram | null): string | null {
  const role = program?.activeRole;
  return typeof role === "string" ? role.trim() || null : null;
}

function resolveSteering(request: string | null | undefined, fallback: string | null): string | null {
  if (request === "") return null;
  const resolved = request ?? fallback;
  return resolved?.trim() || null;
}

interface ResolvedGenerationSettings {
  systemPrompt: string | null;
  steering: string | null;
  sampling: WSSampling;
  thinking: boolean;
}

interface TokenForkContext {
  source: LoomNodeJSON;
  parentId: string;
  forcedPrefix: number[];
  settings: ResolvedGenerationSettings;
}

interface PartialRecipe {
  system_prompt?: string | null;
  steering: string | null;
  sampling: RecipeSamplingJSON | null;
  thinking: boolean | null;
  seed: number | null;
}

function resolveGenerationSettings(
  request: WSSubmitRequest | WSGenerateRequest,
  session: SessionInfo,
  tree: LoomTreeJSON,
  parentId: string,
  maxOutputTokens: number,
): ResolvedGenerationSettings {
  let systemPrompt = request.system_prompt === undefined ? session.config.system_prompt : request.system_prompt;
  const castRecipe = request.raw ? null : generationCastRecipe(request, tree);
  const castSampling = castRecipe?.sampling === null || castRecipe?.sampling === undefined
    ? null
    : recipeSampling(castRecipe.sampling);
  const requestedSteering = request.steering === null || request.steering === undefined
    ? castRecipe?.steering
    : request.steering;
  let steering = resolveSteering(requestedSteering, session.default_steering);
  let sampling = effectiveSampling(
    session,
    mergeCastSampling(castSampling, request.sampling),
    maxOutputTokens,
  );
  if (
    castRecipe?.seed !== null && castRecipe?.seed !== undefined &&
    (request.sampling?.seed === null || request.sampling?.seed === undefined)
  ) {
    sampling = { ...sampling, seed: castRecipe.seed };
  }
  let thinking = request.thinking ?? castRecipe?.thinking ??
    session.config.thinking ?? session.supports_thinking;
  const requestedOverride = request.recipe_override;
  if (requestedOverride === null || requestedOverride === undefined) {
    return { steering, sampling, thinking, systemPrompt };
  }

  const anchor = anchorRecipe(tree, parentId);
  const modifier = recipeModifier(requestedOverride, anchor);
  const overlaid: PartialRecipe = {
    steering: modifier.steering !== null ? modifier.steering : anchor.steering,
    sampling: modifier.sampling !== null ? modifier.sampling : anchor.sampling,
    thinking: modifier.thinking !== null ? modifier.thinking : anchor.thinking,
    seed: modifier.seed !== null ? modifier.seed : anchor.seed,
  };
  if (anchor.system_prompt !== undefined) systemPrompt = anchor.system_prompt;
  if (modifier.system_prompt !== undefined) systemPrompt = modifier.system_prompt;
  if (overlaid.steering !== null) steering = overlaid.steering.trim() || null;
  if (overlaid.sampling !== null) {
    sampling = effectiveSampling(
      session,
      recipeSampling(overlaid.sampling),
      maxOutputTokens,
    );
  }
  if (overlaid.thinking !== null) thinking = overlaid.thinking;
  if (overlaid.seed !== null) sampling = { ...sampling, seed: overlaid.seed };
  return { steering, sampling, thinking, systemPrompt };
}

function generationCastRecipe(
  request: WSSubmitRequest | WSGenerateRequest,
  tree: LoomTreeJSON,
): RecipeJSON | null {
  const role = request.type === "submit"
    ? request.generated_role ?? "assistant"
    : request.generate_seat ?? "assistant";
  const explicitLabel = role === "user"
    ? request.sampling?.user_role?.trim()
    : request.sampling?.assistant_role?.trim();
  return tree.cast[explicitLabel || role]?.recipe ?? null;
}

function mergeCastSampling(
  base: WSSampling | null,
  request: WSSampling | null | undefined,
): WSSampling | null | undefined {
  if (base === null) return request;
  if (request === null || request === undefined) return base;
  const merged: WSSampling = { ...base };
  for (const key of [
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "seed",
    "stop",
    "logit_bias",
    "user_role",
    "assistant_role",
  ] as const) {
    if (request[key] !== null && request[key] !== undefined) {
      Object.assign(merged, { [key]: request[key] });
    }
  }
  for (const key of ["presence_penalty", "frequency_penalty"] as const) {
    if (request[key] !== null && request[key] !== undefined && request[key] !== 0) {
      merged[key] = request[key];
    }
  }
  if ((request.return_top_k ?? 0) > 0) merged.return_top_k = request.return_top_k;
  if (request.persist_per_layer_scores === true) merged.persist_per_layer_scores = true;
  if (request.persist_subspace_coords === true) merged.persist_subspace_coords = true;
  if (request.return_probe_readings === false) merged.return_probe_readings = false;
  return merged;
}

function anchorRecipe(tree: LoomTreeJSON, parentId: string): PartialRecipe {
  const byId = new Map(tree.nodes.map((node) => [node.id, node]));
  let node = byId.get(parentId);
  while (node) {
    if (node.recipe !== null) {
      return {
        steering: node.recipe.steering,
        sampling: node.recipe.sampling,
        thinking: node.recipe.thinking,
        seed: node.recipe.seed,
        ...(node.recipe.system_prompt === undefined ? {} : { system_prompt: node.recipe.system_prompt }),
      };
    }
    node = node.parent_id === null ? undefined : byId.get(node.parent_id);
  }
  return { steering: null, sampling: null, thinking: null, seed: null };
}

function recipeModifier(
  value: NonNullable<
    WSGenerateRequest["recipe_override"] | WSSubmitRequest["recipe_override"]
  >,
  anchor: PartialRecipe,
): PartialRecipe {
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed === "unsteered") return partialRecipe({ steering: "" });
    if (trimmed === "inverted") {
      return partialRecipe({ steering: invertSteering(anchor.steering) });
    }
    if (trimmed === "reseed") return partialRecipe({ seed: randomSeed() });
    if (trimmed === "cool") {
      return partialRecipe({ sampling: recipeSamplingBlock({ temperature: 0.3 }) });
    }
    if (trimmed === "hot") {
      return partialRecipe({ sampling: recipeSamplingBlock({ temperature: 1.2 }) });
    }
    return parsePartialRecipeExpression(trimmed);
  }
  if (!isPlainRecord(value)) {
    throw loomError("INVALID_RECIPE_OVERRIDE", "Recipe override must be a mode or partial recipe");
  }
  return partialRecipeRecord(value);
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function partialRecipe(
  value: Partial<PartialRecipe>,
): PartialRecipe {
  return {
    steering: value.steering ?? null,
    sampling: value.sampling ?? null,
    thinking: value.thinking ?? null,
    seed: value.seed ?? null,
  };
}

function partialRecipeRecord(value: Record<string, unknown>): PartialRecipe {
  const allowed = new Set(["steering", "sampling", "thinking", "seed"]);
  if (Object.keys(value).some((key) => !allowed.has(key))) {
    throw loomError("INVALID_RECIPE_OVERRIDE", "Partial recipe contains unknown fields");
  }
  const steering = nullableText(value.steering, "recipe steering", 16 * 1024);
  if (steering) parseRecipeSteering(steering);
  const thinking = nullableBoolean(value.thinking, "recipe thinking");
  const seed = nullableSafeInteger(value.seed, "recipe seed");
  let sampling: RecipeSamplingJSON | null = null;
  if (value.sampling !== null && value.sampling !== undefined) {
    sampling = parseRecipeSamplingRecord(requireRecord(value.sampling, "recipe sampling"));
  }
  return { steering, sampling, thinking, seed };
}

function parsePartialRecipeExpression(value: string): PartialRecipe {
  if (!value) throw loomError("INVALID_RECIPE_OVERRIDE", "Recipe override cannot be empty");
  const fields = splitPartialRecipeFields(value);
  const allowed = new Set([
    "steering",
    "thinking",
    "seed",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
  ]);
  for (const key of fields.keys()) {
    if (!allowed.has(key)) {
      throw loomError("INVALID_RECIPE_OVERRIDE", `Unknown custom recipe field ${key}`);
    }
  }
  const steering = fields.has("steering") ? fields.get("steering")! : null;
  if (steering !== null) parseRecipeSteering(steering);
  const thinking = fields.has("thinking")
    ? parseRecipeBoolean(fields.get("thinking")!, "thinking")
    : null;
  const seed = fields.has("seed") ? parseRecipeInteger(fields.get("seed")!, "seed") : null;
  const samplingValues: Partial<RecipeSamplingJSON> = {};
  for (const key of [
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
  ] as const) {
    if (fields.has(key)) samplingValues[key] = parseRecipeNumber(fields.get(key)!, key);
  }
  for (const key of ["top_k", "max_tokens"] as const) {
    if (fields.has(key)) samplingValues[key] = parseRecipeInteger(fields.get(key)!, key);
  }
  const sampling = Object.keys(samplingValues).length > 0
    ? recipeSamplingBlock(samplingValues)
    : null;
  validateRecipeSampling(sampling);
  return { steering, sampling, thinking, seed };
}

function splitPartialRecipeFields(value: string): Map<string, string> {
  const pattern = /(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*=/g;
  const matches = [...value.matchAll(pattern)];
  if (matches.length === 0 || matches[0].index !== 0) {
    throw loomError(
      "INVALID_RECIPE_OVERRIDE",
      "Custom recipe fields must use name=value and be separated by commas",
    );
  }
  const fields = new Map<string, string>();
  for (let index = 0; index < matches.length; index += 1) {
    const match = matches[index];
    const key = match[1];
    const start = match.index! + match[0].length;
    const end = index + 1 < matches.length ? matches[index + 1].index! : value.length;
    const raw = value.slice(start, end).trim();
    if ((key !== "steering" && !raw) || fields.has(key)) {
      throw loomError("INVALID_RECIPE_OVERRIDE", `Custom recipe field ${key} is invalid`);
    }
    fields.set(key, raw);
  }
  return fields;
}

function parseRecipeSamplingRecord(value: Record<string, unknown>): RecipeSamplingJSON {
  const allowed = new Set([
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "seed",
    "stop",
    "logit_bias",
    "presence_penalty",
    "frequency_penalty",
    "return_top_k",
    "user_role",
    "assistant_role",
    "persist_per_layer_scores",
    "persist_subspace_coords",
    "return_probe_readings",
  ]);
  if (Object.keys(value).some((key) => !allowed.has(key))) {
    throw loomError("INVALID_RECIPE_OVERRIDE", "Recipe sampling contains unknown fields");
  }
  for (const key of [
    "persist_per_layer_scores",
    "persist_subspace_coords",
    "return_probe_readings",
  ] as const) {
    if (value[key] !== null && value[key] !== undefined && typeof value[key] !== "boolean") {
      throw loomError("INVALID_RECIPE_OVERRIDE", `Recipe sampling ${key} must be a boolean`);
    }
  }
  for (const key of ["user_role", "assistant_role"] as const) {
    const role = value[key];
    if (
      role !== null && role !== undefined &&
      (typeof role !== "string" || role.length > 256 || !isRoleSlug(role))
    ) {
      throw loomError(
        "INVALID_RECIPE_OVERRIDE",
        `Recipe sampling ${key} must be a lowercase role slug`,
      );
    }
  }
  const sampling = recipeSamplingBlock(value as Partial<RecipeSamplingJSON>);
  validateRecipeSampling(sampling);
  return sampling;
}

function recipeSamplingBlock(
  value: Partial<RecipeSamplingJSON>,
): RecipeSamplingJSON {
  return {
    temperature: value.temperature ?? null,
    top_p: value.top_p ?? null,
    top_k: value.top_k ?? null,
    max_tokens: value.max_tokens ?? null,
    seed: value.seed ?? null,
    stop: value.stop ?? null,
    logit_bias: value.logit_bias ?? null,
    presence_penalty: value.presence_penalty ?? 0,
    frequency_penalty: value.frequency_penalty ?? 0,
    logprobs: null,
    return_hidden: false,
    return_top_k: value.return_top_k ?? 0,
    user_role: value.user_role ?? null,
    assistant_role: value.assistant_role ?? null,
    persist_per_layer_scores: value.persist_per_layer_scores ?? false,
    persist_subspace_coords: value.persist_subspace_coords ?? false,
    return_probe_readings: value.return_probe_readings ?? true,
  };
}

function recipeSampling(value: RecipeSamplingJSON): WSSampling {
  return {
    temperature: value.temperature,
    top_p: value.top_p,
    top_k: value.top_k,
    max_tokens: value.max_tokens,
    seed: value.seed,
    stop: value.stop,
    logit_bias: value.logit_bias,
    presence_penalty: value.presence_penalty,
    frequency_penalty: value.frequency_penalty,
    return_top_k: value.return_top_k,
    user_role: value.user_role,
    assistant_role: value.assistant_role,
    persist_per_layer_scores: value.persist_per_layer_scores,
    persist_subspace_coords: value.persist_subspace_coords,
    return_probe_readings: value.return_probe_readings,
  };
}

function validateRecipeSampling(value: RecipeSamplingJSON | null): void {
  if (value === null) return;
  validateWebLlmGenerationSettings(recipeSampling(value), false);
}

function parseRecipeSteering(value: string): void {
  if (!value.trim()) return;
  try {
    parseSteeringExpression(value);
  } catch (error) {
    throw loomError("INVALID_RECIPE_OVERRIDE", error instanceof Error ? error.message : String(error));
  }
}

function parseRecipeNumber(value: string, field: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    throw loomError("INVALID_RECIPE_OVERRIDE", `Custom recipe ${field} must be finite`);
  }
  return parsed;
}

function parseRecipeInteger(value: string, field: string): number {
  const parsed = parseRecipeNumber(value, field);
  if (!Number.isSafeInteger(parsed)) {
    throw loomError("INVALID_RECIPE_OVERRIDE", `Custom recipe ${field} must be an integer`);
  }
  return parsed;
}

function parseRecipeBoolean(value: string, field: string): boolean {
  if (value === "true") return true;
  if (value === "false") return false;
  throw loomError("INVALID_RECIPE_OVERRIDE", `Custom recipe ${field} must be true or false`);
}

function randomSeed(): number {
  return crypto.getRandomValues(new Uint32Array(1))[0] & 0x7fffffff;
}

function invertSteering(value: string | null): string {
  if (!value?.trim()) return "";
  const parsed = parseSteeringExpression(value);
  return formatParsedSteering({
    ...parsed,
    terms: parsed.terms.map((term) => ({
    ...term,
    coefficients: term.coefficients.map((coefficient, index) =>
      index === 0 ? -coefficient : coefficient
    ),
    })),
  });
}

function formatParsedSteering(
  parsed: ReturnType<typeof parseSteeringExpression>,
): string {
  return parsed.terms.map(formatSteeringTerm).join(" + ").replaceAll("+ -", "- ");
}

function formatSteeringTerm(term: ReturnType<typeof parseSteeringExpression>["terms"][number]): string {
  const leadingSign = term.coefficients[0] < 0 ? -1 : 1;
  const coefficients = term.coefficients
    .map((coefficient, index) =>
      formatSteeringNumber(index === 0 ? coefficient : coefficient * leadingSign)
    )
    .join(",");
  const selector = term.selector;
  let output = `${coefficients} ${term.ablation ? "!" : ""}${formatSteeringAtom(selector.base)}`;
  if (selector.manifoldPosition !== null) {
    output += `%${Array.isArray(selector.manifoldPosition)
      ? selector.manifoldPosition.map(formatSteeringNumber).join(",")
      : selector.manifoldPosition}`;
  } else if (selector.projection !== null && selector.onto !== null) {
    output += `${selector.projection}${formatSteeringAtom(selector.onto)}`;
  }
  if (term.trigger !== null) output += formatSteeringTrigger(term.trigger);
  return output;
}

function formatSteeringAtom(atom: ReturnType<typeof parseSteeringExpression>["terms"][number]["selector"]["base"]): string {
  return `${atom.namespace === null ? "" : `${atom.namespace}/`}${atom.concept}${
    atom.variant === "raw" ? "" : `:${atom.variant}`
  }`;
}

function formatSteeringTrigger(trigger: NonNullable<ReturnType<typeof parseSteeringExpression>["terms"][number]["trigger"]>): string {
  const gate = trigger.gate === null
    ? ""
    : `when:${trigger.gate.probe}${trigger.gate.operator}${formatSteeringNumber(trigger.gate.threshold)}`;
  if (gate && trigger.phase.kind === "generated_only") return `@${gate}`;
  const phase = trigger.phase.kind === "both" ? "both"
    : trigger.phase.kind === "after_thinking" ? "after"
    : trigger.phase.kind === "prompt_only" ? "prompt"
    : trigger.phase.kind === "thinking_only" ? "thinking"
    : trigger.phase.kind === "generated_only" ? "response"
    : `${trigger.phase.kind}:${trigger.phase.tokens}`;
  return `@${phase}${gate ? `&${gate}` : ""}`;
}

function formatSteeringNumber(value: number): string {
  return Object.is(value, -0) ? "0" : String(value);
}

function nodeSteering(node: LoomNodeJSON): string | null {
  return node.applied_steering ?? node.recipe?.steering ?? null;
}

function effectiveCast(tree: LoomTreeJSON): LoomTreeJSON["cast"] {
  const configured = clone(tree.cast);
  const output: Record<string, LoomTreeJSON["cast"][string] & { origin: string }> = {};
  for (const label of ["user", "assistant"]) {
    output[label] = configured[label]
      ? { ...configured[label], origin: "configured" }
      : { recipe: null, notes: "", origin: "structural" };
  }
  for (const node of tree.nodes) {
    const label = node.role_label;
    if (!label || output[label] !== undefined) continue;
    output[label] = configured[label]
      ? { ...configured[label], origin: "configured" }
      : { recipe: null, notes: "", origin: "observed" };
  }
  for (const [label, member] of Object.entries(configured)) {
    output[label] = { ...member, origin: "configured" };
  }
  return output;
}

function roundReading(reading: {
  name: string;
  delta: number;
  a_value: number;
  b_value: number;
}): typeof reading {
  return {
    name: reading.name,
    delta: roundSix(reading.delta),
    a_value: roundSix(reading.a_value),
    b_value: roundSix(reading.b_value),
  };
}

function roundSix(value: number): number {
  return Math.round((value + Number.EPSILON) * 1_000_000) / 1_000_000;
}

function replayLogprob(score: WebLlmReplayScore, tokenId: number): number | null {
  const value = replayRawLogprob(score, tokenId);
  return value !== undefined && Number.isFinite(value) ? value : null;
}

function replayRawLogprob(score: WebLlmReplayScore, tokenId: number): number | undefined {
  if (score.argmax.tokenId === tokenId) return score.argmax.logprob;
  return score.requestedLogprobs.find((row) => row.tokenId === tokenId)?.logprob ??
    score.topLogprobs.find((row) => row.tokenId === tokenId)?.logprob;
}

function approximateReplayKl(
  scoreA: WebLlmReplayScore,
  scoreB: WebLlmReplayScore,
): number | null {
  const tokenIds = new Set([
    ...scoreA.topLogprobs.map((row) => row.tokenId),
    ...scoreB.topLogprobs.map((row) => row.tokenId),
  ]);
  let total = 0;
  for (const tokenId of tokenIds) {
    const logpA = replayRawLogprob(scoreA, tokenId);
    const logpB = replayRawLogprob(scoreB, tokenId);
    if (logpA === undefined || logpB === undefined) {
      throw loomError(
        "JOINT_LOGPROBS_UNAVAILABLE",
        "Compact replay omitted a requested top-token probability",
      );
    }
    if (!Number.isFinite(logpA)) continue;
    if (!Number.isFinite(logpB)) return null;
    total += Math.exp(logpA) * (logpA - logpB);
  }
  return Number.isFinite(total) ? total : null;
}

function validateGenerationRequest(request: WSSubmitRequest | WSGenerateRequest): void {
  for (const [label, role] of [
    ["user", request.sampling?.user_role],
    ["assistant", request.sampling?.assistant_role],
  ] as const) {
    if (role !== null && role !== undefined && !isRoleSlug(role)) {
      throw loomError(
        "INVALID_GENERATION_ROLE",
        `The ${label} role must be a lowercase role slug`,
      );
    }
  }
  if (request.type === "generate") {
    if (request.fork_seed != null && (
      !Number.isSafeInteger(request.fork_seed) || request.fork_seed < 0 ||
      request.fork_seed > 0x7fffffff || request.fork_node_id == null
    )) throw loomError("INVALID_GENERATION_REQUEST", "The continuation seed is invalid.");
    const forkFields = [
      request.fork_node_id,
      request.fork_raw_index,
      request.fork_alt_token_id,
      request.fork_replacement_text,
    ];
    const supplied = forkFields.filter((value) => value !== null && value !== undefined).length;
    if (supplied > 0) {
      const hasAlt = request.fork_alt_token_id !== null && request.fork_alt_token_id !== undefined;
      const hasText = request.fork_replacement_text !== null &&
        request.fork_replacement_text !== undefined &&
        request.fork_replacement_text.length > 0;
      if (
        request.fork_node_id === null || request.fork_node_id === undefined ||
        request.fork_raw_index === null || request.fork_raw_index === undefined ||
        hasAlt === hasText
      ) {
        throw loomError(
          "INVALID_GENERATION_REQUEST",
          "A token fork requires a node, raw token index, and exactly one replacement",
        );
      }
      requireIdentifier(request.fork_node_id, "fork node id");
      requireNonnegativeInteger(request.fork_raw_index, "fork raw token index");
      if (hasAlt) {
        requireNonnegativeInteger(request.fork_alt_token_id, "fork alternative token id");
      }
      return;
    }
  }
  const n = request.n ?? 1;
  if (!Number.isSafeInteger(n) || n < 1 || n > 32) {
    throw loomError("INVALID_GENERATION_REQUEST", "Generation sibling count is outside its supported range");
  }
  if (
    usesRawCompletionInput(request) &&
    (request.sampling?.user_role?.trim() || request.sampling?.assistant_role?.trim())
  ) {
    throw loomError(
      "ROLE_SUBSTITUTION_UNAVAILABLE",
      "Raw generation has no OpenAI message-name channel for participant labels",
    );
  }
  if (request.type === "submit") {
    const hasText = request.text !== null && request.text !== undefined;
    if (hasText && (!request.text || !request.authored_role)) {
      throw loomError("INVALID_GENERATION_REQUEST", "Submitted text requires an authored role");
    }
    if (!hasText && request.authored_role) {
      throw loomError("INVALID_GENERATION_REQUEST", "An authored role requires submitted text");
    }
    if (!hasText && !request.generated_role) {
      throw loomError("INVALID_GENERATION_REQUEST", "Submit requires text or a generated role");
    }
    return;
  }
  if (request.stateless !== undefined && typeof request.stateless !== "boolean") {
    throw loomError("INVALID_GENERATION_REQUEST", "Generation stateless must be a boolean");
  }
  if (
    request.input !== null && request.input !== undefined &&
    typeof request.input !== "string" && !Array.isArray(request.input)
  ) {
    throw loomError(
      "INVALID_GENERATION_INPUT",
      "Generation input must be text, a message list, or null",
    );
  }
  if (Array.isArray(request.input)) {
    if (request.input.length === 0) {
      throw loomError("INVALID_GENERATION_INPUT", "Explicit replay requires at least one message");
    }
    for (const message of request.input) {
      if (
        !message || !["system", "user", "assistant"].includes(message.role) ||
        typeof message.content !== "string"
      ) {
        throw loomError("INVALID_GENERATION_INPUT", "Explicit replay contains an invalid message");
      }
      if (
        message.label !== null && message.label !== undefined &&
        !isRoleSlug(message.label)
      ) {
        throw loomError("INVALID_GENERATION_INPUT", "Explicit replay contains an invalid role label");
      }
      if (message.role === "system" && message.label?.trim()) {
        throw loomError(
          "ROLE_SUBSTITUTION_UNAVAILABLE",
          "OpenAI-compatible message names are unavailable for system messages",
        );
      }
      if (usesRawCompletionInput(request) && message.label?.trim()) {
        throw loomError(
          "ROLE_SUBSTITUTION_UNAVAILABLE",
          "Raw replay has no OpenAI message-name channel for participant labels",
        );
      }
    }
  }
}

function usesRawCompletionInput(request: WSSubmitRequest | WSGenerateRequest): boolean {
  return request.raw === true && !(
    request.type === "generate" && Array.isArray(request.input)
  );
}

function validateRuntimeTokenIds(value: readonly number[], label: string): void {
  if (!Array.isArray(value) || value.some((tokenId) =>
    !Number.isSafeInteger(tokenId) || tokenId < 0
  )) {
    throw loomError(
      "INVALID_TOKENIZER_RESULT",
      `The ${label} contains an invalid token id`,
    );
  }
}

function validateInputRoleCapabilities(
  request: WSSubmitRequest | WSGenerateRequest,
  session: SessionInfo,
): void {
  if (request.type !== "generate" || !Array.isArray(request.input)) return;
  for (const message of request.input) {
    if (!message.label?.trim()) continue;
    if (message.role === "user" && !session.user_role_supported) {
      throw loomError(
        "ROLE_SUBSTITUTION_UNAVAILABLE",
        "The loaded browser model cannot replay named user roles",
      );
    }
    if (message.role === "assistant" && !session.role_substitution_supported) {
      throw loomError(
        "ROLE_SUBSTITUTION_UNAVAILABLE",
        "The loaded browser model cannot replay named assistant roles",
      );
    }
  }
}

function validateResolvedSamplingCapabilities(
  session: SessionInfo,
  sampling: WSSampling,
  raw: boolean,
): void {
  const userRole = sampling.user_role?.trim() || null;
  const assistantRole = sampling.assistant_role?.trim() || null;
  for (const [label, role] of [["user", userRole], ["assistant", assistantRole]] as const) {
    if (role !== null && !isRoleSlug(role)) {
      throw loomError(
        "INVALID_GENERATION_ROLE",
        `The ${label} role must be a lowercase role slug`,
      );
    }
  }
  if (raw && (userRole !== null || assistantRole !== null)) {
    throw loomError(
      "ROLE_SUBSTITUTION_UNAVAILABLE",
      "Raw generation has no role-label channel",
    );
  }
  if (userRole !== null && !session.user_role_supported) {
    throw loomError(
      "ROLE_SUBSTITUTION_UNAVAILABLE",
      `The loaded browser model cannot render user role ${JSON.stringify(userRole)}`,
    );
  }
  if (assistantRole !== null && !session.role_substitution_supported) {
    throw loomError(
      "ROLE_SUBSTITUTION_UNAVAILABLE",
      `The loaded browser model cannot render assistant role ${JSON.stringify(assistantRole)}`,
    );
  }
}

function validateSessionConfig(value: SessionInfo["config"]): void {
  finiteRange(value.temperature, "temperature", 0, SAMPLING_TEMPERATURE_MAX);
  finiteRange(value.top_p, "top_p", 0, 1);
  nullableInteger(value.top_k, "top_k", 0);
  nullableInteger(value.max_tokens, "max_tokens", 1);
  if (value.system_prompt !== null && typeof value.system_prompt !== "string") {
    throw loomError("INVALID_SESSION_SETTINGS", "system_prompt must be a string or null");
  }
  if (value.thinking !== null && typeof value.thinking !== "boolean") {
    throw loomError("INVALID_SESSION_SETTINGS", "thinking must be a boolean or null");
  }
}

function clampSessionOutputTokens(
  config: SessionInfo["config"],
  limit: number,
): SessionInfo["config"] {
  return {
    ...config,
    max_tokens: typeof config.max_tokens === "number"
      ? clampOutputTokenCount(config.max_tokens, limit)
      : config.max_tokens,
  };
}

function validateSteeringOptions(value: unknown): {
  probeRequests?: ProbeRequest[];
  tree?: LoomTreeJSON;
} {
  if (value === undefined) return {};
  const options = requireRecord(value, "steering validation options");
  const probeRequests = validateProbeRequests(options.probeRequests);
  return {
    ...(probeRequests === undefined ? {} : { probeRequests }),
    ...(options.tree === undefined ? {} : { tree: clone(options.tree) as LoomTreeJSON }),
  };
}

function validateProbeRequests(value: unknown): ProbeRequest[] | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value)) {
    throw loomError("INVALID_STEERING", "Steering probe preflight must be a list");
  }
  const names = new Set<string>();
  return value.map((raw) => {
    const request = requireRecord(raw, "probe request");
    const selector = requireIdentifier(request.selector, "probe selector");
    const name = request.name === undefined
      ? undefined
      : requireIdentifier(request.name, "probe name");
    const registeredName = name ?? selector;
    if (names.has(registeredName)) {
      throw loomError("INVALID_STEERING", `Probe name ${JSON.stringify(registeredName)} is duplicated`);
    }
    names.add(registeredName);
    const topN = request.top_n;
    if (topN !== undefined && (!Number.isSafeInteger(topN) || (topN as number) < 1)) {
      throw loomError("INVALID_STEERING", "Probe nearest-node count must be a positive integer");
    }
    return {
      selector,
      ...(name === undefined ? {} : { name }),
      ...(topN === undefined ? {} : { top_n: topN as number }),
    };
  });
}

function finiteRange(value: number | null, label: string, min: number, max: number): void {
  if (value === null) return;
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) {
    throw loomError("INVALID_SESSION_SETTINGS", `${label} is outside its supported range`);
  }
}

function nullableInteger(value: number | null, label: string, min: number): void {
  if (value === null) return;
  if (!Number.isSafeInteger(value) || value < min) {
    throw loomError("INVALID_SESSION_SETTINGS", `${label} is outside its supported range`);
  }
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw loomError("INVALID_RUNTIME_REQUEST", `${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireIdentifier(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0 || value.length > 256) {
    throw loomError("INVALID_RUNTIME_REQUEST", `${label} is invalid`);
  }
  return value;
}

function requireNonnegativeInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw loomError("INVALID_RUNTIME_REQUEST", `${label} must be a non-negative integer`);
  }
  return value as number;
}

function requireCastLabel(value: unknown): string {
  if (typeof value !== "string" || !/^[a-z0-9._-]+$/u.test(value)) {
    throw loomError(
      "INVALID_CAST_LABEL",
      "Cast labels use lowercase letters, numbers, dots, underscores, and dashes",
    );
  }
  return value;
}

function nullableText(value: unknown, label: string, maxLength: number): string | null {
  if (value === undefined || value === null) return null;
  return requireText(value, label, maxLength);
}

function nullableBoolean(value: unknown, label: string): boolean | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "boolean") {
    throw loomError("INVALID_CAST_MEMBER", `${label} must be a boolean or null`);
  }
  return value;
}

function nullableSafeInteger(value: unknown, label: string): number | null {
  if (value === undefined || value === null) return null;
  if (!Number.isSafeInteger(value)) {
    throw loomError("INVALID_CAST_MEMBER", `${label} must be an integer or null`);
  }
  return value as number;
}

function requireText(value: unknown, label: string, maximum: number): string {
  if (typeof value !== "string" || value.length > maximum || value.includes("\0")) {
    throw loomError("INVALID_RUNTIME_REQUEST", `${label} is invalid`);
  }
  return value;
}

function clone<T>(value: T): T {
  return structuredClone(value);
}

function loomError(code: string, message: string): Error & {
  code: string;
  status: number;
  recoverable: boolean;
} {
  return Object.assign(new Error(message), { code, status: 409, recoverable: true });
}
