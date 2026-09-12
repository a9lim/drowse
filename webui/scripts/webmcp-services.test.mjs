import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import ts from "typescript";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({ root, appType: "custom", logLevel: "silent", optimizeDeps: { noDiscovery: true, include: [] }, server: { middlewareMode: true, watch: null } });

try {
  const { createServiceTools } = await server.ssrLoadModule("/src/lib/webmcp/serviceTools.ts");
  const { INTERFACE_CATALOGUE, INTERFACE_FAMILIES, RUNTIME_SERVICE_CATALOGUE, catalogueServices } = await server.ssrLoadModule("/src/lib/webmcp/catalogue.ts");
  const { RUNTIME_SERVICE_METHODS } = await server.ssrLoadModule("/src/lib/runtime/contracts.ts");
  const { CONTROL_REFERENCE, explainControl, listControlTopics, explainCurrentControl } = await server.ssrLoadModule("/src/lib/webmcp/controls.ts");
  const { validateInput } = await server.ssrLoadModule("/src/lib/webmcp/validation.ts");
  const { JobRegistry } = await server.ssrLoadModule("/src/lib/webmcp/jobs.ts");
  const { onArtifactUpdate, notifyArtifactUpdate } = await server.ssrLoadModule("/src/lib/artifactUpdates.ts");
  const tools = createServiceTools();
  const byName = new Map(tools.map((tool) => [tool.name, tool]));
  assert.equal(byName.size, tools.length, "service names must be unique");

  const registrySource = await readFile(new URL("../src/drawers/index.ts", import.meta.url), "utf8");
  const drawerNames = [...registrySource.slice(registrySource.indexOf("export const DRAWERS:")).matchAll(/^  ([a-z_]+): \{/gm)].map((match) => match[1]);
  assert.equal(drawerNames.length, 25);
  assert.deepEqual(Object.keys(INTERFACE_CATALOGUE).sort(), drawerNames.sort(), "every drawer needs a conscious coverage decision");
  assert.deepEqual(Object.keys(RUNTIME_SERVICE_CATALOGUE).sort(), Object.keys(RUNTIME_SERVICE_METHODS).sort());
  for (const [service, methods] of Object.entries(RUNTIME_SERVICE_METHODS)) {
    assert.deepEqual(Object.keys(RUNTIME_SERVICE_CATALOGUE[service]).sort(), Object.keys(methods).sort(), `${service} must be exhaustively classified`);
  }
  const actualToolNames = new Set();
  for (const file of await readdir(new URL("../src/lib/webmcp/", import.meta.url))) {
    if (!file.endsWith(".ts") || file === "catalogue.ts") continue;
    const content = await readFile(new URL(`../src/lib/webmcp/${file}`, import.meta.url), "utf8");
    const ast = ts.createSourceFile(file, content, ts.ScriptTarget.Latest, true);
    function visit(node) {
      if (ts.isPropertyAssignment(node) && node.name.getText(ast) === "name" && ts.isStringLiteral(node.initializer) && node.initializer.text.startsWith("drowse_")) actualToolNames.add(node.initializer.text);
      if (ts.isCallExpression(node) && ["tool", "generationTool", "read", "make"].includes(node.expression.getText(ast)) && node.arguments.length && ts.isStringLiteral(node.arguments[0]) && node.arguments[0].text.startsWith("drowse_")) actualToolNames.add(node.arguments[0].text);
      ts.forEachChild(node, visit);
    }
    visit(ast);
  }
  for (const entry of [...Object.values(INTERFACE_CATALOGUE), ...Object.values(INTERFACE_FAMILIES), ...catalogueServices()]) {
    assert.ok(["tool", "unavailable", "internal", "interaction"].includes(entry.kind));
    if (entry.kind !== "tool") assert.ok(entry.reason, "non-tool coverage must explain its disposition");
    for (const name of entry.tools) assert.ok(actualToolNames.has(name), `catalogue references an unknown tool: ${name}`);
  }
  for (const tool of tools) {
    for (const method of tool.services) assert.ok(catalogueServices().some((entry) => entry.id === method && entry.tools.includes(tool.name)), `missing service ownership ${tool.name}: ${method}`);
    assert.equal(tool.annotations.untrustedContentHint, true);
    assert.ok(tool.available({ runtime: null, capabilities: null }));
  }

  const context = { signal: new AbortController().signal, jobs: new JobRegistry(), runtime: { mode: "http", sessions: {}, profiles: {}, probes: {}, manifolds: {}, templates: {}, tree: {}, instruments: {} }, hosted: null, shell: null, capabilities: null };
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient(context.runtime);
  const { sessionState } = await server.ssrLoadModule("/src/lib/stores/session.svelte.ts");
  const { loomTree } = await server.ssrLoadModule("/src/lib/stores/loom.svelte.ts");
  const { genStatus } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const { reservePendingGeneration } = await server.ssrLoadModule("/src/lib/stores/pending.svelte.ts");
  const instrument = (family, source, capabilities = {}) => ({ family, source, live: { enabled: false }, state: source ? "ready" : "unavailable", capabilities: { sources: true, token_readout: !!source, source_switch: !!source, preparations: family === "lens" ? ["fit", "fetch"] : family === "sae" ? ["fetch", "train"] : [], ...capabilities } });
  sessionState.info = { model_id: "test-model", config: {}, instruments: [instrument("geometry", "geometry"), instrument("lens", "local:relp"), instrument("sae", null)], default_user_role: "user", default_assistant_role: "assistant", role_substitution_supported: true, user_role_supported: true };
  loomTree.loaded = true;
  const call = (name, input = {}, ctx = context) => {
    const tool = byName.get(name);
    validateInput(tool.inputSchema, input);
    return tool.execute(input, ctx);
  };
  Object.assign(context.runtime.sessions, { list: async () => ({ sessions: [{ id: "default", model_id: "test-model", device: "cpu", dtype: "float32", created: 10, profiles: ["persona"], probes: [], history_length: 2, config: { system_prompt: "private instructions" } }] }) });
  context.runtime.sessions.get = async () => sessionState.info;
  const inventory = await call("drowse_list_sessions");
  assert.equal(inventory.items[0].profiles, 1);
  assert.equal(inventory.items[0].config, undefined, "inventory omits unrelated prompt/configuration content");
  Object.assign(context.runtime.profiles, { list: async () => ({ profiles: Array.from({ length: 30 }, (_, index) => ({ name: `profile-${index}` })) }) });
  const profiles = await call("drowse_list_profiles", { offset: 1, limit: 2 });
  assert.deepEqual(profiles, { items: [{ name: "profile-1" }, { name: "profile-2" }], total: 30, next_offset: 3 });
  Object.assign(context.runtime.manifolds, { list: async () => ({ manifolds: [{ namespace: "default", name: "personas", description: "Personas", node_labels: ["pirate", "scholar"] }, { namespace: "default", name: "emotions", node_labels: ["happy"] }] }) });
  assert.equal((await call("drowse_list_manifolds", { query: "PIRATE" })).items[0].name, "personas");
  assert.equal((await call("drowse_list_manifolds", { query: "missing" })).total, 0);

  const captured = { provenance: "captured", instruments: { lens: { source: "local:relp" } } };
  const node = { id: "a", text: "x".repeat(1800), thinking_text: null, tokens: [{ measurements: captured }], raw_token_ids: [7], thinking_tokens: [], role: "assistant", recipe: { steering: "0.25 default/personas%pirate" } };
  Object.assign(context.runtime.tree, { get: async () => ({ rev: 2, model_id: "model", root_id: "root", active_node_id: "a", nodes: [node] }), replayCapabilities: async () => ({ jointLogprobs: { available: false, reason: "No replay" } }) });
  assert.equal((await call("drowse_read_tree")).items[0].text.length, 1200);
  assert.deepEqual((await call("drowse_read_tree", { node_id: "a", include_tokens: true })).node.tokens[0].measurements, captured);
  assert.equal((await call("drowse_read_tree", { node_id: "a" })).node.text.length, 1800);
  await assert.rejects(() => call("drowse_read_tree", { node_id: "missing" }), /not in the current tree/);
  await assert.rejects(() => call("drowse_joint_logprobs", { a_id: "a", b_id: "b" }), /No replay/);

  assert.throws(() => validateInput(byName.get("drowse_create_template").inputSchema, { name: "weekdays", slot: "SLOT", values: ["Mon", "Tue"], contexts: [{ turns: [{ role: "user", content: "Day?" }] }] }), /missing assistant/);
  assert.throws(() => validateInput(byName.get("drowse_prepare_instrument").inputSchema, { family: "sae", operation: "fit" }), /accepted input shape/);
  assert.throws(() => validateInput(byName.get("drowse_prepare_instrument").inputSchema, { family: "lens", operation: "fit", prompts: 0 }), /accepted input shape/);
  assert.throws(() => validateInput(byName.get("drowse_prepare_instrument").inputSchema, { family: "lens", operation: "fetch", service: "delete" }), /accepted input shape/);
  assert.throws(() => validateInput(byName.get("drowse_create_manifold").inputSchema, { name: "test", description: "test", domain: { type: "box", axes: [{ name: "a", periodic: false, period: 1, lo: 0 }] }, nodes: [] }), /accepted input shape/);

  let scoreArguments;
  let settle;
  const pending = new Promise((resolve) => { settle = resolve; });
  Object.assign(context.runtime.templates, { score: async (...args) => { scoreArguments = args; return pending; } });
  genStatus.active = true;
  await assert.rejects(() => call("drowse_score_template", { namespace: "local", name: "days" }), { code: "BUSY" });
  genStatus.active = false;
  const release = reservePendingGeneration();
  await assert.rejects(() => call("drowse_score_template", { namespace: "local", name: "days" }), { code: "BUSY" });
  release();
  for (const operation of ["download", "generation", "fitting", "loading"]) {
    const snapshot = { lifecycle: "ready", download: { phase: "idle" }, generation: { phase: "idle" }, fitting: { phase: "idle" } };
    if (operation === "loading") snapshot.lifecycle = "loading";
    else snapshot[operation].phase = "running";
    await assert.rejects(() => call("drowse_score_template", { namespace: "local", name: "days" }, { ...context, hosted: { snapshot } }), { code: "BUSY" });
  }
  const originalInput = { namespace: "local", name: "days", steering: "" };
  const starting = call("drowse_score_template", originalInput);
  originalInput.name = "changed-after-admission";
  const scoreJob = await starting;
  assert.equal(scoreJob.cancellable, false);
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(scoreArguments.slice(0, 3), ["local", "days", ""]);
  assert.equal(scoreArguments[3], scoreJob.id, "the operation receipt uses this exact application job id");
  await assert.rejects(() => call("drowse_score_template", { namespace: "local", name: "days" }), { code: "BUSY" });
  await assert.rejects(() => context.jobs.cancel(scoreJob.id), { code: "JOB_NOT_CANCELLABLE" });
  assert.equal(context.jobs.get(scoreJob.id).state, "running");
  const probabilities = { contexts: [{ choices: [{ value: "Mon", prob_sum: 0.2, prob_mean: 0.6 }] }] };
  settle(probabilities);
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(context.jobs.get(scoreJob.id).state, "completed");
  assert.deepEqual(context.jobs.get(scoreJob.id).result, { ...probabilities, state: "completed" });
  let finishInterrupted;
  context.runtime.templates.score = async () => new Promise(resolve => { finishInterrupted = resolve; });
  const interrupted = await call("drowse_score_template", { namespace: "local", name: "days" });
  await new Promise(resolve => setTimeout(resolve, 0));
  context.jobs.interrupt("The model changed.");
  finishInterrupted(probabilities);
  await new Promise(resolve => setTimeout(resolve, 0));
  assert.equal(context.jobs.get(interrupted.id).state, "interrupted", "late backend success must not overwrite a workspace interruption");
  assert.equal(context.jobs.get(interrupted.id).result, undefined);

  const blocked = { ...context, capabilities: { operations: { fitting: { available: false, reasons: [{ message: "Insufficient memory" }] } } } };
  assert.equal(byName.get("drowse_fit_manifold").available(blocked), "Insufficient memory");
  assert.ok(byName.get("drowse_prepare_instrument").available({ ...context, runtime: { mode: "browser" } }));
  let cancellationCalls = 0;
  const status = (patch = {}) => ({ state: "running", operation: "fit", started_at: 100, finished_at: null, progress: { current: 1, total: 3 }, message: "fitting", error: null, cancellable: false, ...patch });
  Object.assign(context.runtime.instruments, { preparationStatus: async () => status(), cancelPreparation: async () => { cancellationCalls++; } });
  await assert.rejects(() => call("drowse_cancel_preparation", { family: "lens", operation: "fit", started_at: 100 }), /does not support cancellation/);
  assert.equal(cancellationCalls, 0);
  await assert.rejects(() => call("drowse_cancel_preparation", { family: "lens", operation: "fit", started_at: 99 }), { code: "PREPARATION_REPLACED" });
  assert.equal(byName.get("drowse_list_manifold_packs").available(context).includes("browser runtime"), true);
  assert.equal(byName.get("drowse_delete_manifold_pack").available(context).includes("browser runtime"), true);
  assert.ok(byName.get("drowse_validate_sae_feature").available({ ...context, session: sessionState.info }));
  await assert.rejects(() => call("drowse_validate_sae_feature", { feature_id: 0 }), { code: "INSTRUMENT_UNAVAILABLE" });
  let validatedWord;
  context.runtime.instruments.validateLensToken = async word => { validatedWord = word; return { token_id: 1 }; };
  assert.deepEqual(await call("drowse_validate_lens_token", { word: "sail" }), { token_id: 1 });
  assert.equal(validatedWord, "sail");

  assert.equal(explainControl("unknown"), null);
  assert.equal(listControlTopics().length, Object.keys(CONTROL_REFERENCE).length);
  assert.match(explainControl("pirate").defaults, /Keep the current model/);
  assert.match(explainControl("pirate").effect, /system instruction/);
  assert.match(explainControl("role_labels").effect, /not display names/);
  assert.match(explainControl("manifold_position").interactions.join(" "), /onto is vacuous/);
  assert.match(explainControl("probes").effect, /does not itself steer/);
  for (const reference of Object.values(CONTROL_REFERENCE)) for (const key of ["id", "title", "effect", "when", "scope", "unit", "defaults", "support"]) assert.ok(reference[key], `missing ${key}: ${reference.id}`);

  const { registerInterfaceController } = await server.ssrLoadModule("/src/lib/workspaceController.ts");
  const closeDraft = registerInterfaceController("manifold_discover", { schema: { type: "object" }, read: () => ({ dirty: true, values: { samples_per_prompt: 4, tuning: { fitMode: "pca", maxDim: 2 } } }), update: () => {} });
  const currentBudget = await explainCurrentControl("example_budget", context);
  assert.equal(currentBudget.current.forms.manifold_discover.values.samples_per_prompt, 4);
  assert.equal(currentBudget.current.applied, false);
  assert.match(currentBudget.persistence, /Unsaved visible form/);
  closeDraft();
  assert.equal((await explainCurrentControl("example_budget", context)).current.status, "form_not_open");
  assert.equal((await explainCurrentControl("seed", context)).range.minimum, Number.MIN_SAFE_INTEGER);
  assert.equal((await explainCurrentControl("seed", { ...context, runtime: { ...context.runtime, mode: "browser" } })).range.maximum, Number.MAX_SAFE_INTEGER);

  const { runServiceActionCases } = await import("./webmcp-service-action-cases.mjs");
  await runServiceActionCases({ server, tools, validateInput, JobRegistry, installRuntimeClient, sessionState, instrument, context });

  let refreshCount = 0;
  const unsubscribe = onArtifactUpdate(async (area) => { await Promise.resolve(); if (area === "templates") refreshCount++; });
  await notifyArtifactUpdate("templates");
  assert.equal(refreshCount, 1, "mutation completion awaits open UI reconciliation");
  unsubscribe();
  await notifyArtifactUpdate("templates");
  assert.equal(refreshCount, 1, "unmounted surfaces unsubscribe");
  console.log(`webmcp service contracts: ${tools.length} tools, ${drawerNames.length} drawers, ${catalogueServices().length} runtime methods, ${Object.keys(INTERFACE_FAMILIES).length} interface families`);
} finally {
  await server.close();
}
