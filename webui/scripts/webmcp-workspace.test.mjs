import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});
try {
  const registry = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  let validate = async expression => ({ valid: !expression.includes("invalid"), expression, error: "Unknown control" });
  const patches = [];
  const castWrites = [];
  const runtime = { mode: "http", sessions: {
    patch: async patch => { patches.push(patch); Object.assign(sessionState.info.config, patch); return sessionState.info; },
    validateSteering: expression => validate(expression),
  }, tree: {
    castPut: async (label, recipe) => { castWrites.push({ label, recipe }); return { member: { recipe, notes: recipe.notes ?? "", origin: "configured" } }; },
    castDelete: async () => {},
  }, probes: {
    attach: async ({ selector, name }) => ({ family: "lens", name: name ?? selector, word: selector, layers: [0] }),
    detach: async () => {},
  } };
  registry.installRuntimeClient(runtime);
  const { sessionState } = await server.ssrLoadModule("/src/lib/stores/session.svelte.ts");
  const { loomTree, castState } = await server.ssrLoadModule("/src/lib/stores/loom.svelte.ts");
  const { samplingState } = await server.ssrLoadModule("/src/lib/stores/sampling.svelte.ts");
  const { chatLog, genStatus, setGenUiMode } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const { steerRack } = await server.ssrLoadModule("/src/lib/stores/steering.svelte.ts");
  const { probeRack, highlightState } = await server.ssrLoadModule("/src/lib/stores/probes.svelte.ts");
  const { reservePendingGeneration } = await server.ssrLoadModule("/src/lib/stores/pending.svelte.ts");
  const { registerComposerController, registerRawBufferController, getComposerController, registerWorkspaceController, registerSystemPromptController, registerCastController } = await server.ssrLoadModule("/src/lib/workspaceController.ts");
  const { createWorkspaceTools, workspaceRevision } = await server.ssrLoadModule("/src/lib/webmcp/workspaceTools.ts");
  const { validateInput } = await server.ssrLoadModule("/src/lib/webmcp/validation.ts");
  const tools = new Map(createWorkspaceTools().map(tool => [tool.name, tool]));
  const context = { signal: new AbortController().signal, runtime, jobs: { list: () => [] }, capabilities: null, hosted: null, shell: null };
  const invoke = async (name, input) => {
    const tool = tools.get(name);
    assert.ok(tool, name);
    validateInput(tool.inputSchema, input);
    return tool.execute(input, context);
  };
  sessionState.info = {
    model_id: "fixture", is_base_model: false, scene_mode: true,
    role_substitution_supported: true, user_role_supported: false,
    default_user_role: "user", default_assistant_role: "assistant",
    supports_thinking: true, thinking_is_optional: true, thinking_input_supported: true,
    config: { system_prompt: "", temperature: 1, top_p: 1, top_k: null, max_tokens: 64, thinking: false },
  };
  loomTree.loaded = true; loomTree.rev = 1; loomTree.active_node_id = "node-1";
  setGenUiMode("chat");

  await invoke("drowse_set_system_prompt", { text: "Speak like a pirate." });
  assert.equal(samplingState.system_prompt, "Speak like a pirate.");
  assert.equal(samplingState.assistant_role, "assistant", "instructions preserve role labels");
  await invoke("drowse_set_role_labels", { assistant: "pirate" });
  assert.equal(samplingState.assistant_role, "pirate");
  assert.equal(samplingState.system_prompt, "Speak like a pirate.");
  await assert.rejects(invoke("drowse_set_role_labels", { assistant: "captain", user: "pirate" }), /does not support user/);
  assert.equal(samplingState.assistant_role, "pirate", "unsupported compound role changes are atomic");
  await assert.rejects(invoke("drowse_set_role_labels", { assistant: "Captain Jack" }), /lowercase/);
  await invoke("drowse_set_cast_recipe", { label: "pirate", steering: "0.3 default/personas%pirate", notes: "Private metadata" });
  assert.equal(castWrites[0].recipe.steering, "0.3 default/personas%pirate");
  assert.equal(castState.roster.pirate.notes, "Private metadata");
  await invoke("drowse_set_cast_recipe", { label: "pirate", thinking: true, seed: 42 });
  assert.equal(castState.roster.pirate.recipe.seed, 42);
  assert.equal(castState.roster.pirate.recipe.thinking, true);
  await assert.rejects(invoke("drowse_set_cast_recipe", { label: "pirate", clear: true, seed: 42 }), /Clear cannot/);
  await invoke("drowse_set_cast_recipe", { label: "pirate", seed: Number.MIN_SAFE_INTEGER });
  assert.equal(castState.roster.pirate.recipe.seed, Number.MIN_SAFE_INTEGER);
  sessionState.info.thinking_is_optional = false;
  await assert.rejects(invoke("drowse_set_cast_recipe", { label: "pirate", thinking: true }), /no optional thinking/);
  await invoke("drowse_set_cast_recipe", { label: "pirate", thinking: null, seed: null });
  sessionState.info.thinking_is_optional = true;
  const closeCastForm = registerCastController({ read: () => ({ busy: true, dirty: false, values: {} }), sync: () => {} });
  await assert.rejects(invoke("drowse_set_cast_recipe", { label: "pirate", seed: 42 }), /open role form/);
  closeCastForm();
  const closeSystemForm = registerSystemPromptController({ read: () => ({ busy: false, dirty: true, values: { text: "human draft" } }), sync: () => {} });
  await assert.rejects(invoke("drowse_set_system_prompt", { text: "replace" }), /unsaved edit/);
  closeSystemForm();

  const beforeSampling = workspaceRevision();
  await invoke("drowse_set_sampling", { temperature: 0.4, return_top_k: 12, stop_sequences: ["DONE"], logit_bias: { 42: -3 } });
  assert.equal(samplingState.temperature, 0.4);
  assert.equal(samplingState.return_top_k, 12);
  assert.equal(samplingState.top_k, null, "recorded alternatives do not change sampling top-k");
  assert.equal(samplingState.stop_sequences, "DONE");
  assert.notEqual(workspaceRevision(), beforeSampling);
  await assert.rejects(invoke("drowse_set_sampling", { logit_bias: { word: 2 }, temperature: 0.9 }), /token IDs/);
  assert.equal(samplingState.temperature, 0.4);
  await assert.rejects(invoke("drowse_set_sampling", { temperature: Infinity }), /expected number/);
  await assert.rejects(invoke("drowse_set_sampling", {}), /at least one/);
  await assert.rejects(invoke("drowse_set_sampling", { seed: Number.MAX_SAFE_INTEGER + 1 }), /accepted input shape/);
  await assert.rejects(invoke("drowse_set_sampling", { stop_sequences: ["line\nline"] }), /single-line/);
  await invoke("drowse_set_sampling", { top_k: 0, seed: Number.MAX_SAFE_INTEGER });
  assert.equal(samplingState.seed, Number.MAX_SAFE_INTEGER, "large supported seeds are retained without a UI clamp");
  assert.equal(samplingState.top_k, 0, "sampling top-k zero disables filtering");
  assert.equal(samplingState.return_top_k, 12, "disabling filtering preserves recorded alternative count");

  const release = reservePendingGeneration();
  await assert.rejects(invoke("drowse_set_system_prompt", { text: "race" }), /Wait for or cancel/);
  release();
  genStatus.active = true;
  await assert.rejects(invoke("drowse_set_steering", { expression: "0.2 demo" }), /Wait for or cancel/);
  genStatus.active = false;
  await invoke("drowse_set_steering", { expression: "0.3 default/personas%pirate" });
  await assert.rejects(invoke("drowse_set_steering", { expression: "invalid" }), /Unknown control/);
  assert.equal(steerRack.customExpression, "0.3 default/personas%pirate", "failed validation preserves visible steering");
  let finishValidation;
  validate = expression => new Promise(resolve => { finishValidation = () => resolve({ valid: true, expression, error: null }); });
  const stale = invoke("drowse_set_steering", { expression: "0.6 demo" });
  samplingState.temperature = 0.8;
  finishValidation();
  await assert.rejects(stale, /changed while/);
  assert.equal(steerRack.customExpression, "0.3 default/personas%pirate");
  validate = async expression => ({ valid: true, expression, error: null });

  await invoke("drowse_edit_steer_rack", { action: "clear" });
  steerRack.catalog = [{ namespace: "default", name: "personas", fitted_for_session: true, fit_mode: "pca", intrinsic_dim: 1, node_count: 2, node_labels: ["pirate", "assistant"], node_coords: [[1], [-1]], domain: { type: "box", axes: [{ lo: -1, hi: 1 }] } }];
  await invoke("drowse_edit_steer_rack", { action: "upsert", selector: "default/personas", label: "pirate", subspace_along: 0.3 });
  assert.equal(steerRack.entries.get("default/personas").label, "pirate");
  assert.equal(steerRack.subspaceAlong, 0.3);
  await assert.rejects(invoke("drowse_edit_steer_rack", { action: "upsert", selector: "default/personas", label: "missing" }), /no node/);
  assert.equal(steerRack.entries.get("default/personas").label, "pirate");
  await invoke("drowse_attach_probe", { selector: "jlens/pirate" });
  assert.ok(probeRack.entries.has("jlens/pirate"));
  await invoke("drowse_set_highlighting", { target: "jlens/pirate" });
  assert.equal(highlightState.target, "jlens/pirate");
  await invoke("drowse_detach_probe", { name: "jlens/pirate" });
  assert.equal(highlightState.target, null);

  let draft = { text: "human text", authoredRole: "user", generatedRole: "assistant", authoredThinking: "" };
  const unregister = registerComposerController({ read: () => draft, update: changes => { draft = { ...draft, ...changes }; } });
  const humanComposerRevision = workspaceRevision();
  draft.text = "new human draft";
  assert.notEqual(workspaceRevision(), humanComposerRevision, "human composer changes invalidate stale state");
  await invoke("drowse_set_composer", { text: "Ahoy", generated_role: null });
  assert.equal(draft.text, "Ahoy"); assert.equal(draft.generatedRole, null);
  unregister(); assert.equal(getComposerController(), null, "unmount removes stale component callbacks");
  sessionState.info.is_base_model = true;
  await assert.rejects(invoke("drowse_set_system_prompt", { text: "ignored" }), /Raw completion/);
  await assert.rejects(invoke("drowse_set_sampling", { format: "chat" }), /Base models/);
  let raw = { text: "once", dirty: false, busy: false, selection: null };
  registerRawBufferController({ read: () => raw, update: change => { raw = { ...raw, ...change, dirty: true }; }, save: async () => { raw.dirty = false; }, revert: () => {} });
  const humanRawRevision = workspaceRevision();
  raw.text = "human raw draft";
  assert.notEqual(workspaceRevision(), humanRawRevision, "human raw changes invalidate stale state");
  raw.text = "once";
  await assert.rejects(invoke("drowse_set_raw_buffer", { action: "edit", selection: { start: 0, end: 12 } }), /within the buffer/);
  await invoke("drowse_set_raw_buffer", { action: "edit", text: "once upon", selection: { start: 0, end: 4 } });
  assert.equal(raw.text, "once upon");
  await invoke("drowse_set_raw_buffer", { action: "save" });
  assert.equal(raw.dirty, false);
  let navigation = { view: "conversation", section: "response", leftSidebar: true, headersVisible: true };
  registerWorkspaceController({ read: () => navigation, navigate: change => { navigation = { ...navigation, ...change }; } });
  await invoke("drowse_navigate", { view: "controls", section: "chat", left_sidebar: false });
  assert.deepEqual(navigation, { view: "controls", section: "chat", leftSidebar: false, headersVisible: true });
  await assert.rejects(invoke("drowse_navigate", { view: "conversation", drawer: "token_drilldown", params: { turnIdx: 0, tokenIdx: 0 } }), /token is not present/);
  await assert.rejects(invoke("drowse_navigate", { view: "conversation", drawer: "node_compare", params: { node_ids: ["missing-1", "missing-2"] } }), /No node named/);
  await assert.rejects(invoke("drowse_navigate", { view: "conversation", drawer: "help", params: { unknown: "ignored" } }), /unknown property/);
  await assert.rejects(invoke("drowse_navigate", { view: "conversation", token_details: "dock" }), /Open a token/);
  assert.equal(navigation.view, "controls", "failed context validation does not partially navigate");
  chatLog.turns = [{ tokens: [{ text: "Ahoy" }] }];
  globalThis.document = { activeElement: null };
  globalThis.HTMLElement = class {};
  await invoke("drowse_navigate", { drawer: "token_drilldown", params: { turnIdx: 0, tokenIdx: 0 } });
  sessionState.info.api_key = "credential-never-returned";
  const health = await invoke("drowse_health", {});
  assert.equal(health.model_id, "fixture");
  assert.ok(!JSON.stringify(health).includes("credential-never-returned"));
  const refreshedHealth = await invoke("drowse_health", { refresh: true });
  assert.equal(refreshedHealth.refreshing, false);
  assert.equal(refreshedHealth.status, "degraded", "failed diagnostic reads never report ready");
  assert.ok(refreshedHealth.errors.length > 0);
  assert.ok(refreshedHealth.refreshed_at);
  assert.equal(patches.length, 3, "metadata/role changes do not issue system prompt patches");
  sessionState.info.is_base_model = false;
  runtime.sessions.patch = async () => { throw new Error("save denied"); };
  const originalPrompt = sessionState.info.config.system_prompt;
  await assert.rejects(invoke("drowse_set_system_prompt", { text: "failed change" }), /save denied/);
  assert.equal(samplingState.system_prompt, originalPrompt, "a rejected server update restores the authoritative visible setting");
  console.log("WebMCP workspace: identity, sampling, reservations, validation, stale state, rack/probe synchronization, draft controllers and base-model boundaries passed");
} finally {
  await server.close();
}
