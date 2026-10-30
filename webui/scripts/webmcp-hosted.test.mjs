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
  const globals = new Map(["window", "document", "location"].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  const sessionRows = new Map([
    ["drowse.webmcp.jobs.v1", JSON.stringify([{ id: "old-private-job", kind: "generate", state: "completed", cancellable: false, createdAt: 1, startedAt: 1, finishedAt: 2, result: { text: "private result" } }])],
    ["saved-chat-library-fixture", "untouched"],
    ["other-app", "untouched"],
  ]);
  let rejectJobStorage = false, rejectJobRemoval = false;
  const removedKeys = [];
  const sessionStorage = {
    getItem: key => sessionRows.get(key) ?? null,
    setItem: (key, value) => { if (rejectJobStorage && key === "drowse.webmcp.jobs.v1") throw Error("storage denied"); sessionRows.set(key, value); },
    removeItem: key => { removedKeys.push(key); if (rejectJobRemoval && key === "drowse.webmcp.jobs.v1") throw Error("removal denied"); sessionRows.delete(key); },
  };
  const registered = new Map();
  const pageEvents = new EventTarget();
  try {
    Object.defineProperty(globalThis, "window", { configurable: true, value: { sessionStorage, addEventListener: pageEvents.addEventListener.bind(pageEvents) } });
    Object.defineProperty(globalThis, "location", { configurable: true, value: { pathname: "/app" } });
    Object.defineProperty(globalThis, "document", { configurable: true, value: { modelContext: { registerTool: (tool, { signal }) => {
      registered.set(tool.name, tool);
      signal.addEventListener("abort", () => registered.delete(tool.name), { once: true });
    } } } });
    const { startWebMcp } = await server.ssrLoadModule("/src/lib/webmcp/index.ts");
    const { clearDrowseLocalData } = await server.ssrLoadModule("/src/lib/runtime/localData.ts");
    const localStorage = { length: 0, key: () => null, removeItem: () => {} };
    const provider = document.modelContext;
    delete document.modelContext;
    startWebMcp(true);
    const savedJobHistory = sessionRows.get("drowse.webmcp.jobs.v1");
    sessionRows.set("drowse.webmcp.requests.v1", "stale private receipts");
    clearDrowseLocalData(localStorage);
    assert.equal(sessionRows.has("drowse.webmcp.jobs.v1"), false);
    assert.equal(sessionRows.has("drowse.webmcp.requests.v1"), false, "clear removes old receipts even when native agent support is unavailable");
    rejectJobRemoval = true;
    sessionRows.set("drowse.webmcp.jobs.v1", savedJobHistory);
    sessionRows.set("drowse.webmcp.requests.v1", "stale private receipts");
    assert.throws(() => clearDrowseLocalData(localStorage), /Could not clear all local data/);
    assert.equal(sessionRows.has("drowse.webmcp.requests.v1"), false, "a failed first removal must not skip the second key");
    assert.deepEqual(removedKeys.slice(-2), ["drowse.webmcp.jobs.v1", "drowse.webmcp.requests.v1"]);
    rejectJobRemoval = false;
    document.modelContext = provider;
    startWebMcp(true);
    for (let attempt = 0; attempt < 100 && !registered.has("drowse_contact_draft"); attempt++) await new Promise(resolve => setImmediate(resolve));
    const nativeCall = (name, input = {}) => registered.get(name).execute(input, { signal: new AbortController().signal });
    assert.equal((await nativeCall("drowse_get_job")).data.jobs[0].id, "old-private-job");
    await nativeCall("drowse_contact_draft", { request_id: "old-private-request", title: "private draft" });
    assert.ok(sessionRows.get("drowse.webmcp.requests.v1").includes("private draft"));
    clearDrowseLocalData(localStorage);
    assert.deepEqual((await nativeCall("drowse_get_job")).data.jobs, []);
    assert.equal(sessionRows.has("drowse.webmcp.jobs.v1"), false);
    assert.equal(sessionRows.has("drowse.webmcp.requests.v1"), false);
    await nativeCall("drowse_contact_draft", { request_id: "old-private-request", title: "fresh draft" });
    assert.ok(sessionRows.get("drowse.webmcp.requests.v1").includes("fresh draft"), "the live adapter forgets old request ownership after clearing");
    rejectJobStorage = true;
    assert.throws(() => clearDrowseLocalData(localStorage), /Could not clear all local data/);
    assert.equal(sessionRows.has("drowse.webmcp.requests.v1"), false, "receipt cleanup still runs after job history storage fails");
    assert.equal(sessionRows.get("saved-chat-library-fixture"), "untouched");
    assert.equal(sessionRows.get("other-app"), "untouched");
  } finally {
    pageEvents.dispatchEvent(new Event("pagehide"));
    for (const [key, descriptor] of globals) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
  console.log("WebMCP hosted/public: picker selection, prerequisites, job exclusion, truthful download/check outcomes and contact validation without sending passed");
} finally { globalThis.fetch = originalFetch; await server.close(); }
