import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({ root: fileURLToPath(new URL("..", import.meta.url)), appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
const originalFetch = globalThis.fetch;
globalThis.fetch = async () => { throw Error("Network calls are forbidden in this unit test"); };
try {
  const { createHostedTools } = await server.ssrLoadModule("/src/lib/webmcp/hostedTools.ts");
  const { createPublicTools } = await server.ssrLoadModule("/src/lib/webmcp/publicTools.ts");
  const { registerModelPicker } = await server.ssrLoadModule("/src/lib/webmcp/lifecycle.ts");
  const { JobRegistry } = await server.ssrLoadModule("/src/lib/webmcp/jobs.ts");
  const { validateInput } = await server.ssrLoadModule("/src/lib/webmcp/validation.ts");
  const state = { phase: "supported", detail: "Ready", models: [{ id: "chat", name: "Chat", fit: "fits", reason: "Fits", setupComplete: true, firstRunPacks: [{ id: "core", requiredForSetup: true, selected: true }] }], download: { phase: "idle" }, runtime: { phase: "unloaded" } };
  const hosted = { snapshot: { lifecycle: "ready", modelVariantId: "chat", generation: { phase: "idle" }, fitting: { phase: "idle" }, download: { phase: "idle" } } };
  let downloaded = 0, opened = 0, chosen = null, downloadState = "installed";
  const shell = {
    current: () => state, hostedController: () => hosted,
    check: async () => {},
    download: async () => { downloaded++; state.download = { phase: downloadState }; },
    open: async () => { opened++; state.runtime.phase = "ready"; },
    subscribe: listener => { listener(state); return () => {}; },
    setOptionalPackSelected: () => { throw Error("Required pack should be rejected before mutation"); },
  };
  const ctx = { signal: new AbortController().signal, jobs: new JobRegistry(), runtime: null, hosted, shell, capabilities: null };
  const byName = new Map([...createHostedTools(), ...createPublicTools(true)].map(tool => [tool.name, tool]));
  function call(name, input = {}) { const tool = byName.get(name); validateInput(tool.inputSchema, input); return tool.execute(input, ctx); }
  async function finish(id) {
    for (let count = 0; count < 100; count++) {
      await new Promise(resolve => setTimeout(resolve, 1));
      const result = ctx.jobs.get(id);
      if (["completed", "failed", "cancelled", "interrupted"].includes(result.state)) return result;
    }
    throw Error("Test job did not settle");
  }
  const unregister = registerModelPicker(id => { chosen = id; });
  ctx.assertWorkspaceIdle = () => { throw Error("Human generation is queued"); };
  assert.throws(() => call("drowse_load_model", { model_id: "chat" }), /Human generation/);
  delete ctx.assertWorkspaceIdle;
  await call("drowse_select_model", { model_id: "chat" }); assert.equal(chosen, "chat");
  await assert.rejects(() => call("drowse_select_model", { model_id: "chat", packs: [{ pack_id: "core", selected: false }] }), /Required/);
  assert.throws(() => call("drowse_download_model", { model_id: "absent" }), /catalog/);
  const download = call("drowse_download_model", { model_id: "chat" });
  assert.throws(() => call("drowse_load_model", { model_id: "chat" }), /current application job/);
  assert.equal((await finish(download.id)).state, "completed"); assert.equal(downloaded, 1);
  downloadState = "paused";
  assert.equal((await finish(call("drowse_download_model", { model_id: "chat" }).id)).state, "cancelled");
  const load = call("drowse_load_model", { model_id: "chat" });
  assert.equal(load.cancellable, false);
  assert.equal((await finish(load.id)).state, "completed"); assert.equal(opened, 1);
  state.phase = "failed"; state.detail = "Device check failed";
  const checked = await finish(call("drowse_check_device").id);
  assert.equal(checked.state, "failed"); assert.equal(checked.error.code, "device_check_failed");
  const draft = call("drowse_contact_draft", { reason: "research", title: "Study", body: "A test draft" });
  assert.ok(draft.topics.length);
  assert.equal(call("drowse_read_contact").state, "idle");
  call("drowse_contact_draft", { body: "" });
  await assert.rejects(() => call("drowse_send_contact"), /Correct the contact draft/);
  assert.equal(byName.get("drowse_send_contact").annotations.consequentialHint, true);
  unregister();
  console.log("WebMCP hosted/public: picker selection, prerequisites, job exclusion, truthful download/check outcomes and contact validation without sending passed");
} finally { globalThis.fetch = originalFetch; await server.close(); }
