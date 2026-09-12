import { getComposerController, getRawBufferController } from "../workspaceController";
import { SAMPLING_SEED_MAX, SAMPLING_TEMPERATURE_MAX, samplingSeedMinimum } from "../runtime/samplingCapabilities";
import { buildSamplingPayload, samplingState, sessionDefaultsPending } from "../stores/sampling.svelte";
import { manifoldJobs } from "../stores/manifoldJobs.svelte";
import { lensFetch, saeLoad } from "../stores/instruments.svelte";
import { lensFit, saeTrain } from "../stores/instrumentAuthoring.svelte";
import { comparisonPending } from "../stores/ab.svelte";
import { sessionState } from "../stores/session.svelte";
import { castState, loomTree } from "../stores/loom.svelte";
import { currentSteeringExpression } from "../stores/steering.svelte";
import { ensureRuntimeChannel, onWsMessage } from "../stores/ws.svelte";
import { cancelPendingAction, drainNextPendingAction, isPendingBusy, pendingActions, reservePendingGeneration } from "../stores/pending.svelte";
import type { ChatRole, SessionInfo, WSGenerateRequest, WSSampling, WSSubmitRequest } from "../types";
import { awaitGenerationReceipt, type GenerationReceipt } from "./generationReceipt";
import { objectSchema, ToolError, type AppTool, type InputSchema, type ToolContext } from "./types";

interface ComparisonCondition {
  label: string;
  steering: string;
  system_prompt?: string | null;
  text?: string;
  sampling?: WSSampling;
  thinking?: boolean;
}

interface GenerationRecovery {
  version: 1;
  model: string;
  modelId: string;
  rootId: string | null;
  payloads: (WSSubmitRequest | WSGenerateRequest)[];
  labels?: string[];
  systemPrompts: (string | null)[];
  originalSystemPrompt: string | null;
  restoreNeeded: boolean;
  activeSystemPrompt?: string | null;
  dispatched: number[];
}

const samplingSchema = objectSchema({
  temperature: { type: "number", minimum: 0, maximum: SAMPLING_TEMPERATURE_MAX, description: "Higher values broaden sampling; zero is greedy. This is not steering strength." },
  top_p: { type: "number", minimum: 0, maximum: 1, description: "Keep a cumulative probability mass of candidate tokens." },
  top_k: { type: "integer", minimum: 0, description: "Restrict sampling to this many candidates; zero disables top-k." },
  max_tokens: { type: "integer", minimum: 1, description: "Maximum new tokens, including thinking; context/device limits still apply." },
  seed: { type: "integer", minimum: Number.MIN_SAFE_INTEGER, maximum: SAMPLING_SEED_MAX, description: "HTTP accepts signed safe integers; browser requires nonnegative safe integers. One reply preserves the seed; n>1 derives 31-bit sibling seeds. Equal seeds do not guarantee cross-device identity." },
  stop: { type: "array", items: { type: "string", minLength: 1 }, maxItems: 32 },
  logit_bias: { type: "object", additionalProperties: { type: "number" }, description: "Vocabulary token ID to logit adjustment." },
  presence_penalty: { type: "number", minimum: -2, maximum: 2 },
  frequency_penalty: { type: "number", minimum: -2, maximum: 2 },
  return_top_k: { type: "integer", minimum: 0, description: "Alternatives to record, not the sampling top-k. Larger values cost memory." },
  assistant_role: { type: "string", minLength: 1, description: "Actual lowercase chat-header role slug, not a display name. Role-baselined steering may override it." },
  user_role: { type: "string", minLength: 1 },
});

const commonFields: Record<string, InputSchema> = {
  request_id: { type: "string", minLength: 1, maxLength: 128, description: "Reuse this ID only to reconcile the same operation without generating twice." },
  parent_node_id: { type: "string", minLength: 1, description: "Fixed conversation anchor. Omitted freezes the current active node now." },
  steering: { type: "string", description: "Full validated Drowse expression; empty string explicitly disables steering. Omitted snapshots the visible rack/expression." },
  sampling: samplingSchema,
  n: { type: "integer", minimum: 1, maximum: 32, description: "Number of sibling generations; multiplies generation work." },
};

export function createGenerationTools(): AppTool[] {
  return [
    generationTool("drowse_start_generation", "Generate a chat turn",
      "Append text or continue from a fixed conversation node and return a job. Chat instructions are distinct from latent steering and role-header substitution. Settings are frozen when admitted; the job completes only after every sibling finalizes. Omit text to continue, or use_composer to submit the visible draft.",
      objectSchema({ ...commonFields,
        text: { type: "string", minLength: 1 },
        system_prompt: { type: ["string", "null"], description: "Instruction for this generation only; omitted uses the visible session default. Null clears it for this generation. The saved recipe preserves it for replay." },
        use_composer: { type: "boolean", description: "Use the actual visible draft and its authored/generated seats." },
        authored_role: { type: "string", enum: ["user", "assistant"] },
        generated_role: { type: ["string", "null"], enum: ["user", "assistant", null], description: "Null appends authored text without model generation." },
        thinking: { type: "boolean" },
      }), (input, context) => start(input, context, false)),
    generationTool("drowse_raw_continue", "Continue raw text",
      "Continue the flat raw buffer, optionally appending text first, and return a job. Raw completion ignores system prompts, cast recipes and chat-role labels. Save dirty raw edits before continuing. For a base model use a raw text scaffold or compatible steering.",
      objectSchema({ ...commonFields, text: { type: "string", minLength: 1 },
        from_selection: { type: "boolean", description: "Continue from the visible raw selection/caret using the same buffer action." },
        recomplete: { type: "boolean", description: "With from_selection, replace following text from the selection anchor." },
      }), async (input, context) => {
        const prior = previousJob(input, context, "raw_generation");
        if (prior) return prior;
        if (input.recomplete && !input.from_selection) throw new ToolError("INVALID_SELECTION", "Recomplete requires from_selection");
        if (input.from_selection) {
          if (input.text !== undefined || input.parent_node_id !== undefined) {
            throw new ToolError("AMBIGUOUS_SELECTION", "Use the visible selection or explicit text and parent, not both");
          }
          if (isPendingBusy()) throw new ToolError("GENERATION_BUSY", "Wait for pending work before changing the raw selection anchor");
          const buffer = getRawBufferController();
          if (!buffer) throw new ToolError("RAW_BUFFER_UNAVAILABLE", "The raw buffer is not mounted");
          context.signal.throwIfAborted();
          const anchor = await buffer.prepareSelection(input.recomplete === true);
          return start({ ...input, parent_node_id: anchor.parent_node_id }, context, true);
        }
        return start(input, context, true);
      }),
    generationTool("drowse_compare_generations", "Compare generation conditions",
      "Generate matched visible branches from the same prompt and fixed parent, with the same sampling seed and model. Omit text to compare replies to the existing conversation; supply text to add the same prompt below an assistant/root anchor. Each condition can override steering, system_prompt, chat role labels and sampling, or text for a raw scaffold. Omitted settings stay matched; shared seed and model stay fixed unless explicitly overridden. System instructions are per-request overrides that leave session settings unchanged; completed conditions survive failure or interruption. Include an empty expression baseline. Coefficients are artifact- and model-dependent: start with small changes, inspect coherence and reduce strength if repetition appears. Returns one cancellable comparison job.",
      objectSchema({ ...commonFields,
        text: { type: "string", minLength: 1 },
        conditions: { type: "array", minItems: 2, maxItems: 8,
          items: objectSchema({ label: { type: "string", minLength: 1, maxLength: 120 },
            steering: { type: "string" },
            system_prompt: { type: ["string", "null"], description: "Chat instruction condition; raw completion rejects this ignored setting. Null clears it for this condition." },
            text: { type: "string", minLength: 1, description: "Optional prompt/scaffold condition. Omitted uses the shared text." },
            sampling: samplingSchema, thinking: { type: "boolean" },
          }, ["label", "steering"]) },
        thinking: { type: "boolean" },
      }, ["conditions"]),
      (input, context) => start(input, context, requireWorkspace(context).is_base_model)),
    generationTool("drowse_fork_token", "Fork at a token",
      "Create a sibling from a stored generated token boundary, using the source node's saved recipe. Supply exactly one captured alternative token ID or replacement text; raw_index is a raw decode index, not a character index.",
      objectSchema({
        request_id: commonFields.request_id,
        node_id: { type: "string", minLength: 1 },
        raw_index: { type: "integer", minimum: 0 },
        alternative_token_id: { type: "integer", minimum: 0 },
        replacement_text: { type: "string", minLength: 1 },
        seed: { type: "integer", minimum: 0, maximum: 2147483647, description: "Optional continuation seed. Token-fork overrides require 0..2147483647 on both runtimes; omit to reuse the source recipe's seed." },
      }, ["node_id", "raw_index"]), (input, context) => {
        const prior = previousJob(input, context, "token_fork");
        if (prior) return prior;
        const hasToken = input.alternative_token_id !== undefined;
        if (hasToken === (input.replacement_text !== undefined)) {
          throw new ToolError("INVALID_TOKEN_FORK", "Supply exactly one alternative token or replacement text");
        }
        const info = requireWorkspace(context);
        const payload: WSGenerateRequest = {
          type: "generate", fork_node_id: input.node_id as string,
          fork_raw_index: input.raw_index as number,
          ...(hasToken ? { fork_alt_token_id: input.alternative_token_id as number }
            : { fork_replacement_text: input.replacement_text as string }),
          ...(input.seed === undefined ? {} : { fork_seed: input.seed as number }),
        };
        return admit(input, context, "token_fork", payload, info, 1);
      }),
  ];
}

function generationTool(name: string, title: string, description: string, inputSchema: InputSchema,
  execute: AppTool["execute"]): AppTool {
  return {
    name, title, description, inputSchema, execute, scope: "workspace", group: "generation",
    annotations: { readOnlyHint: false, untrustedContentHint: true },
    surfaces: ["conversation", "raw_buffer", "token_drilldown"],
    services: ["events.submit", "events.generate"],
    available: (context) => context.runtime === null ? "Load a model first" :
      context.capabilities?.operations.generation.available === false ? "Generation is unavailable on this runtime" : null,
  };
}

function start(input: Record<string, unknown>, context: ToolContext, raw: boolean) {
  const conditions = input.conditions as ComparisonCondition[] | undefined;
  const kind = conditions ? "comparison" : raw ? "raw_generation" : "chat_generation";
  const prior = previousJob(input, context, kind);
  if (prior) return prior;
  const info = requireWorkspace(context);
  if (!raw && info.is_base_model) {
    throw new ToolError("RAW_MODEL", "This is a base model. Use drowse_raw_continue; system prompts and chat roles do not apply.");
  }
  if (raw && getRawBufferController()?.read().dirty) {
    throw new ToolError("UNSAVED_RAW_DRAFT", "Save or revert the visible raw buffer edits before continuing");
  }
  if (input.use_composer && (input.text !== undefined || input.authored_role !== undefined || input.generated_role !== undefined)) {
    throw new ToolError("AMBIGUOUS_COMPOSER", "Use the visible composer or explicit text and roles, not both");
  }
  const draft = input.use_composer ? getComposerController()?.read() : null;
  if (input.use_composer && !draft) throw new ToolError("COMPOSER_UNAVAILABLE", "The chat composer is not mounted");
  const text = draft ? draft.text || null : input.text as string | undefined ?? null;
  const authoredRole = text === null ? null : draft?.authoredRole ?? input.authored_role as ChatRole | undefined ?? "user";
  const generatedRole = draft ? draft.generatedRole : input.generated_role === null ? null : input.generated_role as ChatRole | undefined ?? "assistant";
  if (text === null && generatedRole === null) throw new ToolError("EMPTY_SUBMISSION", "Provide authored text or a generated role");
  const overrides = input.sampling as WSSampling | undefined;
  if (raw && (overrides?.assistant_role !== undefined || overrides?.user_role !== undefined)) {
    throw new ToolError("RAW_ROLE_UNAVAILABLE", "Raw completion has no chat role headers");
  }
  const sampling: WSSampling = {
    ...buildSamplingPayload(), ...overrides,
  };
  if (sampling.seed !== undefined && sampling.seed !== null && sampling.seed < samplingSeedMinimum(context.runtime!.mode)) {
    throw new ToolError("INVALID_SEED", "Browser generation seeds must be nonnegative safe integers");
  }
  if (raw) { delete sampling.assistant_role; delete sampling.user_role; }
  validateSampling(sampling, info, raw);
  sampling.temperature ??= info.config.temperature;
  sampling.top_p ??= info.config.top_p;
  sampling.top_k = sampling.top_k ?? info.config.top_k ?? 0;
  sampling.max_tokens ??= info.config.max_tokens;
  if (raw) { delete sampling.assistant_role; delete sampling.user_role; }
  const payload: WSSubmitRequest = {
    type: "submit", text, authored_role: authoredRole, generated_role: generatedRole,
    parent_node_id: input.parent_node_id as string | undefined ?? loomTree.active_node_id,
    steering: input.steering as string | undefined ?? currentSteeringExpression(),
    sampling, thinking: raw ? false : input.thinking as boolean | undefined ?? samplingState.thinking ?? false,
    raw, n: generatedRole === null ? 1 : input.n as number | undefined ?? 1,
    ...(draft?.authoredThinking ? { authored_thinking: draft.authoredThinking } : {}),
  };
  if (conditions) {
    if (input.n !== undefined || input.steering !== undefined) {
      throw new ToolError("AMBIGUOUS_COMPARISON", "Comparisons use one generation and the steering expression of each condition");
    }
    payload.sampling!.seed ??= crypto.getRandomValues(new Uint32Array(1))[0]! & 0x7fffffff;
    payload.n = 1;
    if (!raw && text !== null && payload.parent_node_id && loomTree.nodes.get(payload.parent_node_id)?.role === "user") {
      throw new ToolError("COMPARISON_PARENT_ROLE", "Omit text to compare replies to this existing user turn, or choose an assistant/root parent for a new prompt");
    }
    const requests: WSGenerateRequest[] = conditions.map((condition) => {
      if (raw && condition.system_prompt !== undefined) throw new ToolError("RAW_SYSTEM_UNAVAILABLE", "Raw completion ignores system prompts. Compare explicit text scaffolds instead.");
      const conditionSampling = { ...payload.sampling, ...condition.sampling };
      validateSampling(conditionSampling, info, raw);
      if (conditionSampling.seed !== undefined && conditionSampling.seed !== null && conditionSampling.seed < samplingSeedMinimum(context.runtime!.mode)) {
        throw new ToolError("INVALID_SEED", "Browser generation seeds must be nonnegative safe integers");
      }
      if (!raw && condition.text !== undefined && payload.parent_node_id && loomTree.nodes.get(payload.parent_node_id)?.role === "user") {
        throw new ToolError("COMPARISON_PARENT_ROLE", "A condition with new text requires an assistant/root anchor.");
      }
      return { type: "generate", input: condition.text ?? text, parent_node_id: payload.parent_node_id,
        steering: condition.steering, sampling: conditionSampling, thinking: condition.thinking ?? payload.thinking,
        raw, n: 1, stateless: false, append_same_role: false };
    });
    return admit(input, context, kind, requests, info, 1);
  }
  return admit(input, context, kind, payload, info, payload.n ?? 1);
}

function requireWorkspace(context: ToolContext): SessionInfo {
  if (!context.runtime || !sessionState.info || !loomTree.loaded) {
    throw new ToolError("WORKSPACE_NOT_READY", "Wait for the loaded model and conversation to become ready");
  }
  const busyJob = context.jobs.list().find(job => ["running", "queued", "cancelling"].includes(job.state) && !["chat_generation", "raw_generation", "comparison", "token_fork"].includes(job.kind));
  const preparing = [lensFetch, saeLoad, lensFit, saeTrain].some(slice => slice.state.running || slice.state.cancelling);
  const fitting = manifoldJobs.current && ["running", "waiting", "cancelling"].includes(manifoldJobs.current.status);
  if (busyJob || preparing || fitting || comparisonPending() || context.hosted?.snapshot.fitting.phase === "running") {
    throw new ToolError("WORKSPACE_BUSY", "Wait for active model, analysis, fitting or comparison work before generating");
  }
  if (sessionDefaultsPending()) throw new ToolError("SETTINGS_PENDING", "Wait for the visible session settings to finish saving before generating");
  return JSON.parse(JSON.stringify(sessionState.info)) as SessionInfo;
}

function previousJob(input: Record<string, unknown>, context: ToolContext, kind: string) {
  if (typeof input.request_id !== "string") return undefined;
  const prior = context.jobs.list().find((job) => job.requestId === input.request_id);
  if (prior && prior.kind !== kind) throw new ToolError("REQUEST_ID_CONFLICT", "This request ID belongs to another operation");
  return prior;
}

function admit(input: Record<string, unknown>, context: ToolContext, kind: string,
  request: WSSubmitRequest | WSGenerateRequest | (WSSubmitRequest | WSGenerateRequest)[], info: SessionInfo, expectedSiblings: number) {
  const payloads = structuredClone(Array.isArray(request) ? request : [request]);
  const model = modelIdentity(info, context);
  const rootId = loomTree.root_id;
  const cast = castRecipes();
  const conditions = input.conditions as ComparisonCondition[] | undefined;
  const recovery: GenerationRecovery = {
    version: 1, model, modelId: info.model_id, rootId, payloads,
    ...(conditions ? { labels: conditions.map(row => row.label) } : {}),
    systemPrompts: payloads.map((payload, i) => payload.raw ? null : conditions?.[i].system_prompt === undefined
      ? input.system_prompt === undefined ? info.config.system_prompt : input.system_prompt as string | null
      : conditions[i].system_prompt!),
    originalSystemPrompt: info.config.system_prompt, restoreNeeded: false, dispatched: [],
  };
  const anchors = payloads.map((payload) => {
    const id = payload.type === "generate" && payload.fork_node_id ? payload.fork_node_id : payload.parent_node_id;
    return { id: id ?? loomTree.active_node_id, signature: promptAncestry(id ?? loomTree.active_node_id) };
  });
  return context.jobs.start(kind, async (job) => {
    const comparisons: Record<string, unknown>[] = [];
    const checkpoint = (partial?: Record<string, unknown>) => context.jobs.checkpoint(job.id,
      payloads.length === 1 ? partial : comparisonResult(recovery, partial ? [...comparisons, partial] : comparisons));
    try {
      for (const [index, payload] of payloads.entries()) {
        job.signal.throwIfAborted();
        const requestId = payloads.length === 1 ? job.id : `${job.id}-${index}`;
        payload.request_id = requestId;
        let partialOutput: Record<string, unknown> | undefined;
        try {
          const result = await enqueueGeneration(requestId, job.signal, async () => {
            requireWorkspace(context);
            const current = await context.runtime!.sessions.get();
            if (modelIdentity(current, context) !== model || loomTree.root_id !== rootId ||
              castRecipes() !== cast || promptAncestry(anchors[index].id) !== anchors[index].signature ||
              (!payload.raw && current.config.system_prompt !== info.config.system_prompt)) {
              throw new ToolError("WORKSPACE_CHANGED", "The model, system prompt, cast or conversation changed while this request waited. Inspect the workspace before submitting a new request.");
            }
            const channel = await ensureRuntimeChannel();
            const systemPrompt = recovery.systemPrompts[index];
            try {
              if (!payload.raw && !(payload.type === "generate" && payload.fork_node_id)) payload.system_prompt = systemPrompt;
              recovery.dispatched.push(index);
              context.jobs.setRecovery(job.id, recovery);
              return await awaitGenerationReceipt({
                requestId, expectedSiblings, signal: job.signal,
                subscribe: (listener) => {
                  const unsubscribe = onWsMessage(listener);
                  const unsubscribeState = channel.subscribeState((state) => {
                    if (state.state === "closed") listener({ type: "error", code: "RUNTIME_CHANNEL_CLOSED",
                      message: state.reason ?? "The runtime connection closed" });
                  });
                  return () => { unsubscribe(); unsubscribeState(); };
                },
                send: () => channel.send(payload), stop: () => channel.send({ type: "stop", request_id: requestId }),
                requestStatus: channel.requestStatus ? () => channel.requestStatus!(requestId) : undefined,
                onProgress: job.progress,
                onPartial: receipt => {
                  partialOutput = generationOutput(receipt, payload, info.model_id, systemPrompt, recovery.labels?.[index]);
                  partialOutput.state = "partial";
                  checkpoint(partialOutput);
                },
              });
            } finally { await restoreSystemPrompt(job.id, recovery, context); }
          }, job.progress);
          const output = generationOutput(result, payload, info.model_id, recovery.systemPrompts[index], recovery.labels?.[index]);
          if (payloads.length === 1) { checkpoint(output); return output; }
          comparisons.push(output);
          checkpoint();
          job.progress({ phase: "comparison", completed: comparisons.length, total: payloads.length });
          if (result.state === "cancelled") break;
        } catch (error) {
          if (partialOutput && payloads.length > 1) comparisons.push(partialOutput);
          const partialResult = payloads.length === 1 ? partialOutput ?? context.jobs.get(job.id).result : comparisonResult(recovery, comparisons);
          throw Object.assign(error instanceof Error ? error : new Error("Generation failed"), { partialResult });
        }
      }
      return comparisonResult(recovery, comparisons);
    } catch (error) {
      if (!(error as { partialResult?: unknown })?.partialResult) Object.assign(error as object, { partialResult: comparisonResult(recovery, comparisons) });
      throw error;
    }
  }, { requestId: input.request_id as string | undefined, recovery });
}

function generationOutput(result: GenerationReceipt, payload: WSSubmitRequest | WSGenerateRequest,
  model: string, systemPrompt: string | null, label?: string): Record<string, unknown> {
  return { ...result, ...(label === undefined ? {} : { label }), model, requested_settings: payload,
    system_prompt: result.results[0]?.node_id && loomTree.nodes.get(result.results[0].node_id!)?.recipe?.system_prompt !== undefined
      ? loomTree.nodes.get(result.results[0].node_id!)!.recipe!.system_prompt : systemPrompt,
    effective: result.results.map(row => {
      const node = row.node_id ? loomTree.nodes.get(row.node_id) : undefined;
      return { node_id: row.node_id, sibling_index: row.sibling_index, source: node ? "finalized_node" : "completion_event",
        applied_steering: node?.applied_steering ?? row.result.applied_steering ?? null,
        role: node?.role ?? null, role_label: node?.role_label ?? null, recipe: node?.recipe ?? null };
    }), node_ids: result.results.map(row => row.node_id).filter(Boolean) };
}

function comparisonResult(recovery: GenerationRecovery, comparisons: Record<string, unknown>[]) {
  return { comparisons, model: recovery.modelId, completed: comparisons.filter(row => row.state === "completed").length,
    total: recovery.payloads.length,
    state: comparisons.length === recovery.payloads.length && comparisons.every(row => row.state === "completed") ? "completed"
      : comparisons.some(row => row.state === "cancelled") ? "cancelled" : "partial" };
}

function publishSession(info: SessionInfo): void {
  sessionState.info = info;
  samplingState.system_prompt = info.config.system_prompt ?? "";
}

async function restoreSystemPrompt(id: string, recovery: GenerationRecovery, context: ToolContext): Promise<void> {
  if (!recovery.restoreNeeded) return;
  if (context.jobs.get(id).persistedDetailOmissions?.some(path => path.startsWith("job.recovery"))) {
    throw new ToolError("RESTORATION_CONTEXT_UNAVAILABLE", "The saved prompt was too large to retain; inspect the session before restoring it manually.");
  }
  const current = await context.runtime!.sessions.get();
  if (modelIdentity(current, context) !== recovery.model) throw new ToolError("RESTORATION_MODEL_CHANGED", "The model changed before the original system prompt could be restored.");
  if (current.config.system_prompt !== recovery.originalSystemPrompt) {
    if (current.config.system_prompt !== recovery.activeSystemPrompt) throw new ToolError("RESTORATION_CONFLICT", "The system prompt was edited outside this comparison; the newer prompt was preserved.");
    publishSession(await context.runtime!.sessions.patch({ system_prompt: recovery.originalSystemPrompt }));
  }
  recovery.restoreNeeded = false;
  context.jobs.setRecovery(id, recovery);
}

export async function reconcileGenerationJob(id: string, context: ToolContext) {
  const snapshot = context.jobs.get(id);
  const recovery = snapshot.recovery as GenerationRecovery | undefined;
  if (!recovery || recovery.version !== 1 || !context.runtime || snapshot.state !== "interrupted") return snapshot;
  const current = await context.runtime.sessions.get();
  if (modelIdentity(current, context) !== recovery.model) return snapshot;
  const channel = context.runtime.events;
  if (!channel.requestStatus) return snapshot;
  const previous = snapshot.result as { comparisons?: Record<string, unknown>[] } | undefined;
  const comparisons: Record<string, unknown>[] = recovery.payloads.length === 1 ? [] : [...(previous?.comparisons ?? [])];
  let state: "completed" | "cancelled" | "failed" | "interrupted" = "completed";
  let error = snapshot.error;
  for (const index of recovery.dispatched) {
    const payload = recovery.payloads[index];
    const requestId = recovery.payloads.length === 1 ? id : `${id}-${index}`;
    const status = await channel.requestStatus(requestId);
    if (status.state === "running") return context.jobs.reconcile(id, { state: "interrupted", progress: { phase: "runtime_running", request_id: requestId } });
    if (status.state === "unknown" || status.state === "interrupted") { if (state !== "failed") state = "interrupted"; }
    else if (status.state === "failed") { state = "failed"; error = { code: status.error?.code ?? "GENERATION_FAILED", message: status.error?.message ?? "The runtime recorded a generation failure" }; }
    else if (status.state === "cancelled" && state === "completed") state = "cancelled";
    if (status.results.length || status.state === "completed") {
      const output = generationOutput({ requestId, state: status.state === "completed" ? "completed" : "cancelled", results: status.results },
        payload, recovery.modelId, recovery.systemPrompts[index], recovery.labels?.[index]);
      if (status.state === "unknown" || status.state === "interrupted" || status.state === "failed") output.state = "partial";
      const priorIndex = comparisons.findIndex(row => row.requestId === requestId);
      if (priorIndex === -1) comparisons.push(output);
      else comparisons[priorIndex] = output;
      if (status.state === "completed" && status.results.length !== (payload.n ?? 1)) state = "interrupted";
    }
  }
  await restoreSystemPrompt(id, recovery, context);
  if (recovery.dispatched.length < recovery.payloads.length && state === "completed") state = "interrupted";
  const result = recovery.payloads.length === 1 ? comparisons[0] ?? snapshot.result : comparisonResult(recovery, comparisons);
  return context.jobs.reconcile(id, { state, result, ...(error ? { error } : {}), progress: { phase: "reconciled", dispatched: recovery.dispatched.length } });
}

function validateSampling(sampling: WSSampling, info: SessionInfo, raw: boolean): void {
  for (const [key, supported, normal] of [
    ["assistant_role", info.role_substitution_supported, info.default_assistant_role ?? "assistant"],
    ["user_role", info.user_role_supported, info.default_user_role ?? "user"],
  ] as const) {
    const label = sampling[key];
    if (label === undefined || label === null) continue;
    if (raw) throw new ToolError("RAW_ROLE_UNAVAILABLE", "Raw completion has no chat role headers");
    if (!/^[a-z0-9._-]+$/.test(label)) throw new ToolError("INVALID_ROLE", "Role labels use lowercase letters, digits, dots, underscores, and hyphens");
    if (!supported) {
      if (label !== normal) throw new ToolError("ROLE_UNAVAILABLE", `This model does not support ${key} substitution`);
      delete sampling[key];
    }
  }
}

function modelIdentity(info: SessionInfo, context: ToolContext): string {
  return JSON.stringify([info.model_id, info.created, context.hosted?.snapshot.modelVariantId,
    context.hosted?.snapshot.contextTokens]);
}

function castRecipes(): string {
  return JSON.stringify(Object.entries(castState.roster)
    .filter(([, member]) => member.recipe !== null)
    .map(([label, member]) => [label, member.recipe]).sort(([a], [b]) => String(a).localeCompare(String(b))));
}

function promptAncestry(id: string | null): string {
  const rows: unknown[] = [];
  const seen = new Set<string>();
  while (id !== null) {
    const node = loomTree.nodes.get(id);
    if (!node || seen.has(id)) throw new ToolError("INVALID_PARENT", "The selected conversation anchor is unavailable. Refresh the conversation before generating.");
    seen.add(id);
    rows.push([node.id, node.parent_id, node.role, node.role_label, node.text, node.thinking_text, node.recipe]);
    id = node.parent_id;
  }
  return JSON.stringify(rows);
}

export function enqueueGeneration(id: string, signal: AbortSignal, run: () => Promise<GenerationReceipt>,
  progress: (value: unknown) => void): Promise<GenerationReceipt> {
  return new Promise((resolve, reject) => {
    let queued = true;
    const cancelled = () => {
      if (!queued) return;
      queued = false;
      signal.removeEventListener("abort", cancelQueued);
      reject(new DOMException("The queued generation was cancelled", "AbortError"));
    };
    const cancelQueued = () => {
      if (!queued) return;
      cancelPendingAction(id);
      cancelled();
    };
    const apply = async () => {
      if (!queued) return;
      queued = false;
      signal.removeEventListener("abort", cancelQueued);
      if (signal.aborted) { reject(signal.reason); return; }
      const release = reservePendingGeneration();
      try { resolve(await run()); }
      catch (error) { reject(error); }
      finally {
        release();
        setTimeout(() => { void drainNextPendingAction(); }, 0);
      }
    };
    signal.addEventListener("abort", cancelQueued, { once: true });
    if (signal.aborted) { cancelled(); return; }
    if (isPendingBusy()) {
      if (pendingActions.queue.length >= 32) {
        signal.removeEventListener("abort", cancelQueued);
        reject(new ToolError("GENERATION_QUEUE_FULL", "The pending action queue is full"));
        return;
      }
      pendingActions.queue.push({ id, label: "agent generation", text: null, apply,
        awaitsGen: true, rebuild: null, createdAt: Date.now(), onCancel: cancelled });
      progress({ phase: "queued", position: pendingActions.queue.length });
    } else void apply();
  });
}
