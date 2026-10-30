import { IDENTITY_HELP, SAMPLING_HELP } from "../parameterHelp";
import { outputTokenLimitForSignals } from "../runtime/outputTokenPolicy";
import { SAMPLING_SEED_MAX, SAMPLING_TEMPERATURE_MAX, samplingSeedMinimum, tokenAlternativeLimit } from "../runtime/samplingCapabilities";
import { EXAMPLES_PER_ROUND } from "../exampleBudget";
import { PER_NODE_ROLE_HELP } from "../manifolds/selectors";
import type { ToolContext } from "./types";

export interface ControlReference {
  id: string;
  title: string;
  effect: string;
  when: string;
  scope: string;
  unit: string;
  defaults: string;
  support: string;
  interactions: string[];
}

const generationScope = "Subsequent generations; committed text and recorded measurements remain unchanged.";
const sampling = (id: string, title: string, effect: string, unit: string, defaults: string, interactions: string[] = []): ControlReference => ({
  id, title, effect, unit, defaults, interactions, scope: generationScope,
  when: "Adjust token selection or generation length for the user's requested outcome; read current effective settings before changing them.",
  support: "Use the loaded runtime's parameter support and limits; model defaults can differ.",
});

export const CONTROL_REFERENCE: Record<string, ControlReference> = {
  pirate: {
    id: "pirate", title: "Choose how to make a pirate voice",
    effect: "A speech-style request is usually served by a system instruction such as: Speak in a playful pirate voice while preserving factual accuracy and answering the user's question. A chat name changes metadata. A role label changes supported template tokens. Steering injects fitted activation directions during generation.",
    when: "For ordinary voice/style, start with prompting. For an adjustable learned persona, composing traits, or a steering experiment, inspect installed persona nodes and fitted support first. Use a custom extraction only if a suitable artifact is absent and the user wants the compute involved.",
    scope: generationScope, unit: "Choice of intervention; effects are model- and prompt-dependent.",
    defaults: "Keep the current model, sampling and template role labels. Avoid unnecessary model training or downloads.",
    support: "System instructions require a chat model. Base models use a raw textual prompt. Steering requires a compatible fitted artifact. Role substitution requires supported template metadata.",
    interactions: [
      "A saved chat called Captain has no behavioral effect. Renaming the assistant's actual template role to pirate may affect generation but is not a reliable substitute for a style instruction.",
      "When using steering, resolve an exact namespace/name and node label from the current catalog. Use the returned fit metadata rather than inventing pirate selectors.",
      "Compare an unsteered baseline and small artifact-specific increments with the same prompt and seed. Values 0.25 and 0.5 are exploratory, not calibrated: they collapsed into repetition in a small-model smoke test. Reduce strength promptly when quality degrades.",
      "Prefer one intervention at a time for attribution. Check pirate language, answer quality, repetition and factual correctness. Lower strength when repetition or loss of meaning appears.",
      "Changing the system prompt, template role and steering together can make style stronger but prevents identifying which intervention caused the change.",
    ],
  },
  chat_identity: { id: "chat_identity", title: "Chat name and avatar", effect: IDENTITY_HELP.chat, when: "Organize and identify saved conversations.", scope: "Saved conversation metadata and presentation.", unit: "Name, avatar and accent.", defaults: "Current saved chat identity.", support: "Browser conversation library.", interactions: ["Use system instructions or steering for a voice or behavior request."] },
  system_prompt: { id: "system_prompt", title: "System instructions", effect: IDENTITY_HELP.systemPrompt, when: "Specify speaking style, task behavior and response format in natural language.", scope: generationScope, unit: "Text instruction.", defaults: "Loaded session instructions; an empty string explicitly clears them.", support: "Chat-template models; raw text for base models.", interactions: ["Prompting is not weight training and does not guarantee instruction adherence.", "Preserve existing task instructions when adding a style request."] },
  role_labels: { id: "role_labels", title: "Chat-template role labels", effect: IDENTITY_HELP.role, when: "Explicitly experiment with the speaker identity encoded in supported model templates.", scope: "Future rendered turns and their stored recipes.", unit: "Template role label.", defaults: "Use the loaded model's default_user_role and default_assistant_role.", support: "Check supported template role substitution on the current model.", interactions: ["Keep structural user/assistant roles distinct from their textual labels.", "Role-augmented artifacts carry matching extraction baseline requirements.", "Base models use raw prompt text rather than chat-template role controls."] },
  cast: { id: "cast", title: "Standing role recipes", effect: IDENTITY_HELP.cast, when: "Configure fallback steering, thinking or seed settings for a named speaker.", scope: "Future matching-speaker calls that omit the corresponding setting.", unit: "Steering expression, thinking override and seed.", defaults: "Explicit per-reply settings take precedence over standing recipes.", support: "Runtime cast service.", interactions: ["Ordinary workspace generation explicitly sends the visible steering expression, including an empty rack, and the visible thinking setting. These suppress cast steering/thinking fallbacks; setting a cast recipe alone does not change those visible controls.", "A cast seed can apply when the generation supplies no seed. Read the finalized node's effective recipe and role label to verify what was used.", "A note is metadata, not a system instruction.", "A steering field takes the Drowse expression grammar, not natural-language prose."] },
  temperature: sampling("temperature", "Temperature", SAMPLING_HELP.advancedTemperature, "Nonnegative logit temperature; 0 is greedy.", "Read current model defaults; null follows runtime default.", ["Higher temperature increases variation, not a specific persona.", "Top P, Top K, penalties, bias and steering jointly affect candidate selection.", "Zero selects an argmax; seed normally has no effect on that selection."]),
  top_p: sampling("top_p", "Top P", SAMPLING_HELP.topP, "Cumulative probability mass in (0, 1].", "Read loaded model defaults.", ["Lower values restrict the candidate set; 1 removes this probability-mass cutoff.", "Combine cautiously with Top K when comparing interventions."]),
  top_k: sampling("top_k", "Top K", SAMPLING_HELP.topK, "Candidate count, bounded by the loaded vocabulary.", "Cleared input uses Drowse's effective runtime default; inspect the sampling snapshot.", ["Top K controls the sampling candidate pool; return_top_k controls saved alternatives.", "Large candidate pools can cost more sampling work."]),
  max_tokens: sampling("max_tokens", "Maximum output tokens", SAMPLING_HELP.maxTokens, "Positive token count within runtime/device/context limits.", "Loaded session setting; UI pre-hydration placeholder is not an authoritative default.", ["This is a ceiling, not a required answer length or word count.", "Thinking tokens can consume generation budget.", "Context limits include prompt and output; increasing a limit can increase memory and time."]),
  seed: sampling("seed", "Random seed", SAMPLING_HELP.seed, "Integer seed; null leaves it unspecified.", "Unspecified unless the current reply or cast recipe supplies one.", ["Equal seeds improve controlled comparisons but do not guarantee cross-model, cross-device or cross-version identity.", "Keep prompt, history, sampling and model fixed when comparing steering."]),
  thinking: sampling("thinking", "Thinking", SAMPLING_HELP.thinking, "true, false, or automatic where supported.", "Use the current UI state and model capability; do not infer support from the label.", ["Separate reasoning consumes tokens and time; it does not guarantee better correctness.", "Thinking/response steering triggers depend on a supported phase boundary."]),
  presence_penalty: sampling("presence_penalty", "Presence penalty", SAMPLING_HELP.presencePenalty, "Additive logit penalty; zero is off.", "0 unless overridden.", ["Positive values discourage revisiting tokens; negative values favor repetition.", "This is not a direct topicality or truthfulness control."]),
  frequency_penalty: sampling("frequency_penalty", "Frequency penalty", SAMPLING_HELP.frequencyPenalty, "Additive logit penalty weighted by token count; zero is off.", "0 unless overridden.", ["Positive values penalize repeatedly used tokens more strongly.", "Large penalties can hurt coherence; diagnose excessive steering before compensating with penalties."]),
  return_top_k: sampling("return_top_k", "Recorded token alternatives", SAMPLING_HELP.returnTopK, "Nonnegative number of alternatives; 0 retains chosen-only evidence.", "Read current runtime setting; defaults vary by runtime.", ["This is evidence capture, not the sampling Top K cutoff.", "Captured alternatives permit token-choice inspection and branching; absent alternatives cannot be recovered as original capture."]),
  stop_sequences: sampling("stop_sequences", "Stop sequences", "End generation when an exact configured text sequence is produced.", "List of exact strings.", "Empty list uses no additional user stops.", ["A stop can truncate a response before its intended ending.", "Use exact desired delimiters rather than punctuation likely to occur naturally."]),
  logit_bias: sampling("logit_bias", "Token logit bias", "Add a bias to exact vocabulary token logits before sampling.", "Map of tokenizer-specific token IDs to finite numeric biases.", "Empty map.", ["Token IDs depend on the current tokenizer and are not words shared across models.", "Strong positive bias may force repetitive tokens; a phrase can contain multiple tokens."]),
  steering: {
    id: "steering", title: "Activation steering", effect: "Steering expressions lower to a unified per-layer injection without changing model weights. Positive coefficients move along the fitted target-minus-neutral direction; negative coefficients reverse that displacement.",
    when: "Adjust fitted concepts, combine learned traits, test causal effects, or choose manifold positions with controlled coefficients.", scope: generationScope, unit: "Model-specific dimensionless coefficients applied to fitted geometry.",
    defaults: "Read the current expression. Ordinary omitted coefficients parse as 0.5; bare ! ablation is 1.0. An empty expression explicitly disables steering.", support: "Compatible per-model fits and required geometry/instruments must be available.",
    interactions: ["There is no universal coefficient-to-behavior scale. Compare outputs and reduce coefficients if quality degrades.", "! removes the centered mean-direction component; it is not a semantic ban. ~ keeps a shared projection and | removes it. These operators have distinct effects.", "The visual rack's flat subspaces share an along master; a custom expression is authoritative and may not have a faithful rack rendering.", "Adding one expression should preserve existing desired terms unless the user requested replacement."] },
  manifold_position: { id: "manifold_position", title: "Manifold position, along and onto", effect: "The % operator chooses a named node or domain coordinate. Along translates the in-subspace foot toward the target direction; curved onto reduces the off-surface residual.", when: "Select a learned persona node or position on a fitted continuous trait geometry.", scope: generationScope, unit: "Coordinates in the artifact's authoring domain; along and onto coefficients.", defaults: "Read the artifact's dimensions, node coordinates and current rack values; no universal coordinate range.", support: "Flat fits use along; onto is meaningful for curved geometry and its fitted thickness.", interactions: ["A flat manifold's onto is vacuous and should not be presented as another strength knob.", "Two curved manifolds may conflict when their fitted spans overlap; the runtime validates this.", "Label-form positions use the exact installed node label. Coordinate arity is validated at load."] },
  triggers: { id: "triggers", title: "Steering phases and probe gates", effect: "Triggers select which prompt/decode positions receive a steering term. Probe gates apply it when a measured channel passes a threshold.", when: "Restrict intervention to response, thinking, counted decode windows, or a measurable activation condition.", scope: "Per-term injection eligibility during generation.", unit: "Phase, token count or threshold in the chosen probe channel's units.", defaults: "Read current term triggers and loaded model phase support.", support: "Probe gates require an attached instrument that can emit the requested channel.", interactions: ["Gate direction scores, probabilities and distances are different units; inspect probe metadata before choosing thresholds.", "Gates can require per-step computation even with live display capture disabled.", "A channel unsupported by that instrument is an error, not a gate that should silently remain inactive."] },
  probes: { id: "probes", title: "Probe measurements", effect: "Probes measure activation evidence. Attaching a probe does not itself steer the model.", when: "Observe the effect of a prompt or steering experiment.", scope: "Future capture; stored historical evidence preserves its original provenance.", unit: "Geometry coordinates, fractions, membership, distances, posterior assignments, or native instrument strength.", defaults: "Read the attached roster and channel metadata.", support: "Geometry needs its fitted metric; lens and SAE channels require resident sources.", interactions: ["Geometry fraction and fuzzy membership are in [0,1] but measure different things.", "Label-similarity gates are negative distance in typical-label-spacing units; larger means closer.", "Lens strength is mean fitted-layer vocabulary probability; SAE strength uses its source normalization when available.", "Nearest-label and probe readings are evidence within a fitted representation, not verified facts about intent or truth."] },
  lens: { id: "lens", title: "Lens readout and steering", effect: "A fitted Jacobian or R-lens maps intermediate residuals toward final readout. Token probes report mean fitted-layer probability; lens atoms steer token-specific directions.", when: "Inspect what intermediate layers favor saying, or test a specific single-token direction.", scope: "Current active source controls future reads and lens atoms; historical captures retain original source metadata.", unit: "Probability [0,1] for readouts; model-specific coefficients for steering.", defaults: "Read available sources and active source; local lens fitting defaults to R-lens on supported Python models.", support: "Exact tokenizer token validation and compatible fitted lens are required. Browser fitting may be unavailable even when reading is supported.", interactions: ["Sharp token directions can oversteer at smaller coefficients than broad concept directions.", "Top entries and depth summaries are derived from fitted-layer probabilities, not raw logits.", "Fetching an external source and fitting locally are different operations with different compute costs."] },
  sae: { id: "sae", title: "Sparse autoencoder features", effect: "Read or steer a feature from the active sparse autoencoder source and layer.", when: "Inspect or intervene on a validated sparse feature rather than a broad named manifold node.", scope: generationScope, unit: "Feature activation, normalized strength when cached maxActApprox is available, and steering coefficient.", defaults: "Read active source, feature ID, layer and normalization metadata.", support: "Resident compatible SAE; feature IDs and meanings are source-specific. Browser read/activation support does not imply browser training.", interactions: ["External descriptions are hypotheses about a feature, not guarantees of semantic behavior.", "SAE probes emit strength; geometry membership or fraction channels are not SAE channels."] },
  extraction: { id: "extraction", title: "Corpus generation and manifold fitting", effect: "Generate or author labeled corpora, then fit their activation geometry to a model. A two-node flat fit is difference-of-means; one-node fits are neutral-anchored rays.", when: "A needed fitted control does not exist or the user is authoring an experiment.", scope: "Persistent artifact and per-model fitted evidence; generation uses it only after steering selects it.", unit: "Concepts, samples per baseline prompt, layers and geometry fit settings.", defaults: "Reuse matching cached fits; force explicitly regenerates or refits.", support: "Read model/runtime fitting capabilities and artifact prerequisites first.", interactions: ["More examples increase compute and may improve coverage; they do not guarantee a stronger or better concept.", "Abstract framing suits traits; concrete framing suits entities/personas; custom framing takes an explicit {c} template.", "Templates suit referenced categories such as weekdays; an in-character weekday persona is not a meaningful extraction design.", "PCA, spectral and auto refer to manifold fitting; they are not separate concept extraction methods."] },
  fit_geometry: { id: "fit_geometry", title: "Fit method and geometry parameters", effect: "PCA fits a flat layout; spectral uses a neighbor graph and curved geometry; auto selects supported topology from the evidence. These choices change the artifact, not the current sampling distribution until the artifact is used.", when: "Author or refit a manifold and inspect its diagnostics.", scope: "Saved manifold and its per-model fit.", unit: "Intrinsic dimensions, variance fraction, graph neighbors, kernel width and smoothing.", defaults: "The builder uses auto on Python and PCA in the browser, max_dim up to 8 subject to runtime limit, and PCA variance 0.7; omitted graph values use data-derived defaults.", support: "Browser fit support and intrinsic-dimension limits can differ from Python. Read runtime capabilities before choosing a method.", interactions: ["max_dim caps coordinates; increasing var_threshold can retain more PCA axes up to that cap.", "Spectral min_dim floors coordinates. Setting it equal to max_dim pins their count. max_subspace_dim separately limits curved activation-space rank; it does not apply to PCA.", "Increasing k_nn joins more neighbors and may connect components. Bandwidth changes graph affinity scale. Neither is a behavioral-strength knob.", "Curved smoothing 0 interpolates nodes; positive smoothing regularizes the surface; auto selects a penalty through generalized cross-validation.", "The auto persistence_frac threshold controls the strength of loop evidence needed for periodic topology.", "Refitting can change directions and probe units; preserve fit provenance and do not compare old and new measurements as if the geometry were identical."] },
  example_budget: { id: "example_budget", title: "Generated example budget", effect: `Each samples_per_prompt round generates ${EXAMPLES_PER_ROUND} aligned responses per concept. Increasing rounds increases corpus generation work approximately in proportion to the number of responses.`, when: "Choose a corpus budget before starting concept generation.", scope: "Persistent node corpora used by later fitting.", unit: `Responses = concepts × ${EXAMPLES_PER_ROUND} baseline prompts × samples_per_prompt.`, defaults: "The builder starts at one round. Standard, More examples and Thorough select one, two and four rounds.", support: "Runtime generation/fitting capability and available memory.", interactions: ["More examples may improve coverage but do not guarantee behavioral quality.", "Keep framing and shared prompts matched when comparing concepts.", "Inspect saved corpora and fit diagnostics before increasing the budget."] },
  role_per_node: { id: "role_per_node", title: "Per-node role baselines", effect: PER_NODE_ROLE_HELP, when: "Explicitly author a manifold whose fitted persona directions also carry matching assistant-role baselines.", scope: "Corpus/fit provenance and future generations using the artifact.", unit: "Boolean authoring choice and exact node role labels.", defaults: "Off in the builder; ordinary artifact fitting preserves the standard assistant baseline.", support: "Model template role substitution and compatible role-aware fitting.", interactions: ["Nearest-node role selection changes template headers as well as activation steering.", "An explicit role-variant selector carries its own matching role requirement.", "This differs from cosmetic chat naming and should not be silently enabled for a simple speech-style request."] },
  template_scoring: { id: "template_scoring", title: "Restricted-choice completion scoring", effect: "Score each candidate at a template slot under the raw model distribution, optionally steered, with separate summed and length-normalized logprob distributions.", when: "Test whether steering changes a defined choice distribution rather than inspecting one generated argmax.", scope: "An evaluation result; does not alter model weights or session sampling.", unit: "Candidate logprob and softmax restricted to the supplied candidate set.", defaults: "Keep sum and mean views separate.", support: "Valid saved template with slot exactly once in the final assistant content and absent from history.", interactions: ["Restricted-choice probabilities do not represent confidence over all possible answers.", "Sampling temperature/top-k/top-p do not truncate the teacher-forced raw scoring distribution."] },
};

export function explainControl(control: string): ControlReference | null {
  return CONTROL_REFERENCE[control.trim().toLowerCase()] ?? null;
}

export function listControlTopics(): { id: string; title: string }[] {
  return Object.values(CONTROL_REFERENCE).map(({ id, title }) => ({ id, title }));
}

export async function explainCurrentControl(control: string, context: ToolContext) {
  const reference = explainControl(control);
  if (!reference) return null;
  if (!context.runtime) return { ...reference, current: null, source: "No loaded workspace", persistence: "Read this reference before opening a model." };
  const info = await context.runtime.sessions.get();
  const { samplingState, modelDefaultsState, buildSamplingPayload } = await import("../stores/sampling.svelte");
  const { currentSteeringExpression, steerRack } = await import("../stores/steering.svelte");
  const { effectiveRawMode } = await import("../stores/chat.svelte");
  const { readRegisteredForms, getCastController, getSystemPromptController } = await import("../workspaceController");
  const raw = effectiveRawMode();
  const overrides = buildSamplingPayload() ?? {};
  const cfg = info.config;
  const original = modelDefaultsState.info?.model_id === info.model_id ? modelDefaultsState.info.config : null;
  const effective: Record<string, unknown> = {
    temperature: overrides.temperature ?? cfg.temperature,
    top_p: overrides.top_p ?? cfg.top_p,
    top_k: overrides.top_k ?? cfg.top_k,
    max_tokens: overrides.max_tokens ?? cfg.max_tokens,
    system_prompt: samplingState.system_prompt,
    seed: samplingState.seed,
    thinking: samplingState.thinking,
    presence_penalty: samplingState.presence_penalty,
    frequency_penalty: samplingState.frequency_penalty,
    return_top_k: samplingState.return_top_k,
    stop_sequences: overrides.stop ?? [],
    logit_bias: overrides.logit_bias ?? {},
    role_labels: { user: samplingState.user_role || info.default_user_role, assistant: samplingState.assistant_role || info.default_assistant_role },
    steering: currentSteeringExpression(),
    lens: info.instruments.find((item) => item.family === "lens") ?? null,
    sae: info.instruments.find((item) => item.family === "sae") ?? null,
  };
  const formTopics: Record<string, string[]> = {
    extraction: ["manifold_builder", "manifold_discover", "manifold_authored", "manifold_templated"],
    fit_geometry: ["manifold_discover", "manifold_authored", "manifold_templated", "manifold_merge"],
    example_budget: ["manifold_discover"], role_per_node: ["manifold_discover", "manifold_authored"],
    template_scoring: ["template_lab"],
  };
  if (reference.id in formTopics) {
    const forms = readRegisteredForms();
    effective[reference.id] = { forms: Object.fromEntries(formTopics[reference.id].filter(id => forms[id]).map(id => [id, forms[id]])), status: formTopics[reference.id].some(id => forms[id]) ? "visible_draft" : "form_not_open", applied: false };
  } else if (reference.id === "cast") {
    const { castState } = await import("../stores/loom.svelte");
    effective.cast = { applied: castState.roster, draft: getCastController()?.read() ?? null };
  } else if (reference.id === "chat_identity") {
    const { conversationLibrary, savedConversationState } = await import("../stores/savedConversations.svelte");
    const chats = savedConversationState.activeId ? (await conversationLibrary.list()).conversations : [];
    const chat = chats.find(item => item.id === savedConversationState.activeId);
    effective.chat_identity = { id: savedConversationState.activeId, name: chat?.name ?? null, avatar_seed: savedConversationState.avatarSeed, accent: savedConversationState.accent, draft: readRegisteredForms().chat_identity ?? null };
  } else if (reference.id === "probes") {
    const { probeRack } = await import("../stores/probes.svelte");
    effective.probes = { attached: probeRack.active.map(name => ({ name, info: probeRack.entries.get(name)?.info ?? null })), instruments: info.instruments, error: probeRack.error };
  } else if (["manifold_position", "triggers"].includes(reference.id)) {
    effective[reference.id] = { expression: currentSteeringExpression(), custom_expression: steerRack.customExpression, subspace_along: steerRack.subspaceAlong, entries: [...steerRack.entries].map(([selector, entry]) => ({ selector, ...entry })), catalogue: steerRack.catalog, catalogue_error: steerRack.error };
  } else if (reference.id === "pirate") {
    effective.pirate = { conversation_format: raw ? "raw" : "chat", system_prompt: samplingState.system_prompt, role_labels: effective.role_labels, steering: currentSteeringExpression() };
  }
  const ranges: Record<string, unknown> = {
    temperature: { minimum: 0, maximum: SAMPLING_TEMPERATURE_MAX },
    top_p: { exclusive_minimum: 0, maximum: 1 },
    top_k: { minimum: 0, maximum: Number.MAX_SAFE_INTEGER, effective_cap: "Loaded vocabulary size" },
    max_tokens: { minimum: 1, maximum: outputTokenLimitForSignals(context.capabilities?.signals, context.hosted?.snapshot.contextTokens ?? undefined) },
    seed: { minimum: samplingSeedMinimum(context.runtime.mode), maximum: SAMPLING_SEED_MAX, nullable: true },
    presence_penalty: { minimum: -2, maximum: 2 },
    frequency_penalty: { minimum: -2, maximum: 2 },
    return_top_k: { minimum: 0, maximum: tokenAlternativeLimit(context.runtime.mode) },
    manifold_position: "Read the selected manifold's domain and coordinate arity.",
    steering: "Finite coefficients; inspect fitted model behavior. Larger values can degrade output.",
  };
  const sessionDefaults = ["temperature", "top_p", "top_k", "max_tokens", "system_prompt", "thinking"];
  return {
    ...reference,
    current: effective[reference.id] ?? null,
    range: ranges[reference.id] ?? reference.unit,
    model_id: info.model_id,
    runtime: context.runtime.mode,
    is_base_model: info.is_base_model,
    source: reference.id in formTopics ? "Registered visible authoring forms. Draft values have no artifact or output effect until submitted." : ["lens", "sae"].includes(reference.id) ? "Fresh loaded-session instrument descriptor including active source, state, capability and provenance." : "Visible workspace settings, falling back to loaded session config for omitted numeric overrides; explicit per-generation arguments and cast precedence still apply.",
    persistence: reference.id in formTopics ? "Unsaved visible form state; submit the corresponding authoring action to persist an artifact or compute a scoring result."
      : reference.id === "chat_identity" ? "Saved conversation metadata; no generation effect."
      : ["lens", "sae"].includes(reference.id) ? "Active source selection and live capture state; historical token evidence retains its captured provenance."
      : sessionDefaults.includes(reference.id)
      ? "Changing this session default affects subsequent calls. Saving a conversation preserves the visible configuration in its backup."
      : "Workspace settings are recorded in generation recipes and saved conversation snapshots; historical generations are unchanged.",
    model_defaults: original && Object.hasOwn(original, reference.id) ? { [reference.id]: original[reference.id as keyof typeof original] } : null,
    session_defaults: Object.hasOwn(cfg, reference.id) ? { [reference.id]: cfg[reference.id as keyof typeof cfg] } : null,
    current_support: {
      conversation_format: raw ? "raw" : "chat",
      system_prompt: !raw,
      assistant_role: !raw && info.role_substitution_supported,
      user_role: !raw && info.user_role_supported,
      thinking: info.supports_thinking,
      thinking_optional: info.thinking_is_optional,
      operations: context.capabilities?.operations ?? null,
    },
    ...(reference.id === "system_prompt" ? { draft: getSystemPromptController()?.read() ?? null } : {}),
  };
}
