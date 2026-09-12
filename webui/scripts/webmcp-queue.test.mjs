import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({ root: fileURLToPath(new URL("..", import.meta.url)),
  appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
const flush = () => new Promise(resolve => setTimeout(resolve, 1));
try {
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  const { DeterministicFakeRuntime } = await server.ssrLoadModule("/src/hosted/runtime/fakeRuntime.ts");
  const runtime = new DeterministicFakeRuntime({ response: "fixture answer", tokenDelayMs: 1 });
  installRuntimeClient(runtime);
  const { enqueueGeneration, createGenerationTools, reconcileGenerationJob } = await server.ssrLoadModule("/src/lib/webmcp/generation.ts");
  const pending = await server.ssrLoadModule("/src/lib/stores/pending.svelte.ts");
  const { genStatus } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const calls = [];
  const first = new AbortController();
  const second = new AbortController();
  let finishFirst;
  genStatus.active = true;
  const a = enqueueGeneration("first", first.signal, () => {
    calls.push("first");
    return new Promise(resolve => { finishFirst = resolve; });
  }, () => {});
  const b = enqueueGeneration("second", second.signal, async () => {
    calls.push("second");
    return { requestId: "second", state: "completed", results: [] };
  }, () => {});
  const bRejected = assert.rejects(b, { name: "AbortError" });
  second.abort();
  await bRejected;
  assert.deepEqual(calls, [], "queued cancellation never reaches the runtime or a global stop");
  assert.deepEqual(pending.pendingActions.queue.map(row => row.id), ["first"]);
  genStatus.active = false;
  const drain = pending.drainNextPendingAction();
  await flush();
  assert.deepEqual(calls, ["first"]);
  assert.equal(pending.isPendingBusy(), true, "dispatch is reserved before a started event exists");
  pending.enqueueOrApply("human adjustment", () => calls.push("human"));
  assert.deepEqual(calls, ["first"]);
  finishFirst({ requestId: "first", state: "completed", results: [] });
  await a;
  await drain;
  await flush();
  assert.deepEqual(calls, ["first", "human"], "human and agent mutations share FIFO admission");
  assert.equal(pending.isPendingBusy(), false);
  genStatus.active = true;
  const uiCancelled = enqueueGeneration("cancel-in-ui", new AbortController().signal, async () => {
    throw new Error("must not run");
  }, () => {});
  const rejected = assert.rejects(uiCancelled, { name: "AbortError" });
  pending.cancelPendingAction("cancel-in-ui");
  await rejected;
  genStatus.active = false;
  const { JobRegistry } = await server.ssrLoadModule("/src/lib/webmcp/jobs.ts");
  const { validateInput } = await server.ssrLoadModule("/src/lib/webmcp/validation.ts");
  const { sessionState } = await server.ssrLoadModule("/src/lib/stores/session.svelte.ts");
  const { applyTreeSnapshot, loomTree } = await server.ssrLoadModule("/src/lib/stores/loom.svelte.ts");
  const { disconnectRuntimeChannel } = await server.ssrLoadModule("/src/lib/stores/ws.svelte.ts");
  sessionState.info = await runtime.sessions.get();
  applyTreeSnapshot(await runtime.tree.get());
  const jobs = new JobRegistry();
  const context = { jobs, runtime, signal: new AbortController().signal, hosted: null, shell: null, capabilities: null };
  const tools = createGenerationTools();
  const complete = async (snapshot) => {
    for (let attempt = 0; attempt < 5000; attempt += 1) {
      const current = jobs.get(snapshot.id);
      if (["completed", "failed", "cancelled", "interrupted"].includes(current.state)) return current;
      await flush();
    }
    throw new Error("fixture job did not settle");
  };
  const generationTool = tools.find(row => row.name === "drowse_start_generation");
  validateInput(generationTool.inputSchema, { sampling: { seed: Number.MIN_SAFE_INTEGER } });
  validateInput(generationTool.inputSchema, { sampling: { seed: Number.MAX_SAFE_INTEGER } });
  assert.throws(() => generationTool.execute({ sampling: { seed: -1 } }, context), error => error.code === "INVALID_SEED");
  const forkTool = tools.find(row => row.name === "drowse_fork_token");
  assert.throws(() => validateInput(forkTool.inputSchema, { node_id: "source", raw_index: 0, alternative_token_id: 1,
    seed: Number.MAX_SAFE_INTEGER }), error => error.code === "invalid_input", "fork overrides retain their actual 31-bit protocol limit");
  const generation = await complete(generationTool.execute({ text: "fixture prompt", n: 2,
    sampling: { seed: Number.MAX_SAFE_INTEGER } }, context));
  assert.equal(generation.state, "completed", JSON.stringify(generation));
  assert.equal(generation.result.results.length, 2);
  assert.equal(generation.result.requested_settings.steering, "");
  assert.equal(generation.result.requested_settings.sampling.seed, Number.MAX_SAFE_INTEGER);
  assert.ok(generation.result.effective.every(row => row.recipe.seed >= 0 && row.recipe.seed <= 2147483647), "fans derive a 31-bit seed per sibling without changing the requested seed receipt");
  assert.ok(generation.result.effective.every(row => row.source === "finalized_node" && row.applied_steering === null));
  const parent = loomTree.nodes.get(generation.result.node_ids[0]).parent_id;
  const parentBefore = JSON.stringify(loomTree.nodes.get(parent));
  const compared = await complete(tools.find(row => row.name === "drowse_compare_generations").execute({ parent_node_id: parent, sampling: { seed: Number.MAX_SAFE_INTEGER },
    conditions: [{ label: "baseline", steering: "" }, { label: "repeat", steering: "" }] }, context));
  assert.equal(compared.state, "completed", JSON.stringify(compared));
  assert.equal(JSON.stringify(loomTree.nodes.get(parent)), parentBefore, "matched comparisons preserve the shared user prompt");
  assert.deepEqual(compared.result.comparisons.map(row => row.requested_settings.parent_node_id), [parent, parent]);
  assert.equal(compared.result.comparisons[0].effective[0].recipe.seed, compared.result.comparisons[1].effective[0].recipe.seed);
  assert.equal(compared.result.comparisons[0].effective[0].recipe.seed, Number.MAX_SAFE_INTEGER, "single generations retain large seeds exactly");
  const compareTool = tools.find(row => row.name === "drowse_compare_generations");
  const originalSystem = (await runtime.sessions.get()).config.system_prompt;
  const varied = await complete(compareTool.execute({ parent_node_id: parent, conditions: [
    { label: "baseline", steering: "", system_prompt: null, sampling: { temperature: 0 } },
    { label: "instruction", steering: "", system_prompt: "Speak like a pirate.", sampling: { temperature: 0.8, assistant_role: "assistant" } },
  ] }, context));
  assert.equal(varied.state, "completed", JSON.stringify(varied));
  assert.equal((await runtime.sessions.get()).config.system_prompt, originalSystem, "comparison restores the exact original null system prompt");
  assert.deepEqual(varied.result.comparisons.map(row => row.system_prompt), [null, "Speak like a pirate."]);
  assert.deepEqual(varied.result.comparisons.map(row => row.effective[0].recipe.system_prompt), [null, "Speak like a pirate."]);
  assert.deepEqual(varied.result.comparisons.map(row => row.effective[0].recipe.sampling.temperature), [0, 0.8]);
  const originalSend = runtime.events.send.bind(runtime.events);
  runtime.events.send = payload => {
    if (payload.system_prompt === "fail-condition") throw new Error("fixture condition failed");
    return originalSend(payload);
  };
  const partial = await complete(compareTool.execute({ parent_node_id: parent, conditions: [
    { label: "completed baseline", steering: "" }, { label: "failed instruction", steering: "", system_prompt: "fail-condition" },
  ] }, context));
  runtime.events.send = originalSend;
  assert.equal(partial.state, "failed");
  assert.equal(partial.result.comparisons[0].label, "completed baseline");
  assert.equal(partial.result.comparisons[0].state, "completed");
  assert.equal(partial.result.completed, 1, "finished comparison conditions survive a later failure");
  assert.equal((await runtime.sessions.get()).config.system_prompt, originalSystem);
  assert.throws(() => compareTool.execute({ parent_node_id: parent, conditions: [
    { label: "baseline", steering: "" }, { label: "role", steering: "", sampling: { assistant_role: "pirate" } },
  ] }, context), error => error.code === "ROLE_UNAVAILABLE");
  let saved;
  const persistence = { getItem: () => saved ?? null, setItem: (_key, value) => { saved = value; } };
  jobs.attachPersistence(persistence, "recover");
  const reconciledSource = await complete(generationTool.execute({ parent_node_id: parent, request_id: "durable-generation" }, context));
  jobs.reconcile(reconciledSource.id, { state: "interrupted", error: { code: "PAGE_RELOADED", message: "fixture page interrupted after runtime completion" } });
  const restoredJobs = new JobRegistry();
  restoredJobs.attachPersistence(persistence, "recover");
  const reconnected = await reconcileGenerationJob(reconciledSource.id, { ...context, jobs: restoredJobs });
  assert.equal(reconnected.state, "completed");
  assert.deepEqual(reconnected.result.node_ids, reconciledSource.result.node_ids);
  assert.equal(restoredJobs.start("chat_generation", async () => { throw new Error("duplicate generation"); }, { requestId: "durable-generation" }).id, reconciledSource.id);
  const fitting = jobs.start("manifold_fit", () => new Promise(() => {}), { cancellable: false });
  await flush();
  assert.throws(() => generationTool.execute({ parent_node_id: parent }, context), error => error.code === "WORKSPACE_BUSY");
  jobs.interrupt("fixture cleanup", job => job.id === fitting.id);
  genStatus.active = true;
  const changed = generationTool.execute({ parent_node_id: parent }, context);
  await flush();
  await runtime.tree.edit(parent, "edited fixture prompt");
  applyTreeSnapshot(await runtime.tree.get());
  genStatus.active = false;
  await pending.drainNextPendingAction();
  const refused = await complete(changed);
  assert.equal(refused.state, "failed");
  assert.equal(refused.error.code, "WORKSPACE_CHANGED");
  disconnectRuntimeChannel();
  await runtime.dispose();
  console.log("WebMCP shared queue, correlated generation, provenance and matched comparison checks passed");
} finally { await server.close(); }
