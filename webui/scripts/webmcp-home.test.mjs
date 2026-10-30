import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({ root: fileURLToPath(new URL("..", import.meta.url)), appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
try {
  const { createHomeTools, homeRevision } = await server.ssrLoadModule("/src/lib/webmcp/homeTools.ts");
  const { getRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  assert.throws(() => getRuntimeClient(), /has not been installed/, "home tools import without installing or loading a runtime");
  const { conversationLibrary, onConversationLibraryChanged } = await server.ssrLoadModule("/src/lib/stores/savedConversations.svelte.ts");
  const { ConversationLibrary } = await server.ssrLoadModule("/src/lib/conversationLibrary.ts");
  const { CONVERSATION_SAMPLING_KEYS } = await server.ssrLoadModule("/src/lib/conversationSnapshot.ts");
  const { registerPageActions } = await server.ssrLoadModule("/src/lib/webmcp/lifecycle.ts");
  const { registerHomeController } = await server.ssrLoadModule("/src/lib/homeController.ts");
  const { JobRegistry } = await server.ssrLoadModule("/src/lib/webmcp/jobs.ts");
  const { validateInput } = await server.ssrLoadModule("/src/lib/webmcp/validation.ts");
  const rows = new Map(); let seq = 0; let failWrites = false;
  const library = new ConversationLibrary({ samplingKeys: CONVERSATION_SAMPLING_KEYS, randomId: () => `chat-${++seq}`, store: {
    initialize: async () => {}, list: async () => [...rows].map(([key, value]) => ({ key, value })),
    read: async id => structuredClone(rows.get(id)), write: async record => { if (failWrites) throw new Error("Storage full"); rows.set(record.id, structuredClone(record)); }, delete: async id => rows.delete(id),
  } });
  for (const method of ["listSummaries", "get", "create", "update", "duplicate", "delete"]) conversationLibrary[method] = library[method].bind(library);
  const original = await library.create({ name: "Pirate experiment", avatarSeed: "avatar", snapshot: snapshot() });
  const tools = new Map(createHomeTools().map(tool => [tool.name, tool]));
  const jobs = new JobRegistry();
  const context = { jobs, shell: {}, runtime: null, hosted: null, capabilities: null, signal: new AbortController().signal };
  const invoke = async (name, input) => { const tool = tools.get(name); validateInput(tool.inputSchema, input); return tool.execute(input, context); };
  let refreshed = 0;
  const unsubscribe = onConversationLibraryChanged(() => { refreshed++; });
  assert.equal((await invoke("drowse_list_chats", {})).chats[0].id, original.id);
  await invoke("drowse_update_chat", { id: original.id, name: "Captain", accent: "mint" });
  const renamed = await library.get(original.id);
  assert.equal(renamed.name, "Captain"); assert.equal(renamed.accent, "mint");
  assert.equal(renamed.snapshot.samplingState.assistant_role, "assistant", "renaming a chat does not alter model role labels");
  assert.equal(renamed.snapshot.samplingState.system_prompt, "Be helpful", "renaming a chat does not alter model instructions");
  const staleMetadataVersion = (await invoke("drowse_list_chats", {})).chats.find(chat => chat.id === original.id).metadataVersion;
  const originalUpdatedAt = renamed.updatedAt;
  await library.update(original.id, { name: "Human correction" });
  await assert.rejects(invoke("drowse_update_chat", { id: original.id, name: "stale agent edit", expected_metadata_version: staleMetadataVersion }), /name or avatar changed/);
  await assert.rejects(invoke("drowse_delete_chat", { id: original.id, expected_metadata_version: staleMetadataVersion }), /name or avatar changed/);
  assert.equal((await library.get(original.id)).name, "Human correction");
  assert.equal((await library.get(original.id)).updatedAt, originalUpdatedAt, "metadata checks preserve chat ordering timestamps");
  const freshMetadataVersion = (await library.listSummaries()).conversations.find(chat => chat.id === original.id).metadataVersion;
  const competingUpdates = await Promise.allSettled([
    library.update(original.id, { name: "Concurrent writer one" }, freshMetadataVersion),
    library.update(original.id, { name: "Concurrent writer two" }, freshMetadataVersion),
  ]);
  assert.equal(competingUpdates.filter(result => result.status === "fulfilled").length, 1, "metadata checks and writes share the exclusive lock");
  assert.equal(competingUpdates.filter(result => result.status === "rejected")[0].reason.code, "STALE_RECORD");
  await library.update(original.id, { name: "Captain" });
  assert.equal(refreshed, 1, "visible home refresh listeners run after metadata changes");
  const duplicate = await invoke("drowse_duplicate_chat", { id: original.id });
  assert.notEqual(duplicate.chat.id, original.id);
  const backup = await invoke("drowse_export_chat", { id: original.id, format: "text" });
  const fileBackup = await invoke("drowse_export_chat", { id: original.id, format: "file" });
  assert.equal(fileBackup.file.state, "complete");
  const fileImport = await invoke("drowse_import_chat", { file_id: fileBackup.file.file_id });
  assert.equal(fileImport.opened, false);
  await assert.rejects(invoke("drowse_import_chat", { text: backup.text, file_id: fileBackup.file.file_id }), /exactly one/);
  const imported = await invoke("drowse_import_chat", { text: backup.text });
  assert.notEqual(imported.chat.id, original.id); assert.equal(imported.opened, false);
  const changedBackup = JSON.parse(backup.text); changedBackup.conversation.name = "Untrusted edit";
  await assert.rejects(invoke("drowse_import_chat", { text: JSON.stringify(changedBackup) }), /integrity check/);
  failWrites = true;
  await assert.rejects(invoke("drowse_update_chat", { id: original.id, name: "not saved" }), /Storage full/);
  assert.equal((await library.get(original.id)).name, "Captain");
  failWrites = false;
  let homeNavigations = 0;
  registerPageActions({ home: () => { homeNavigations++; }, state: () => ({ route: "home", recovery: null }), models() {}, retry: async () => {}, allowTakeover: async () => {}, denyTakeover() {} });
  let busy = false;
  let metadataDraft = null;
  registerHomeController({ busy: () => busy, metadataDraft: () => metadataDraft, open: async id => ({ state: "model_required", chat_id: id, model_id: "fixture/model", model_variant_id: "fixture-variant" }) });
  const beforeHumanRename = homeRevision();
  metadataDraft = { id: original.id, name: "Human inline rename", dirty: true };
  assert.notEqual(homeRevision(), beforeHumanRename, "home revision includes unsaved human rename fields");
  await assert.rejects(invoke("drowse_update_chat", { id: original.id, name: "agent overwrite" }), /open chat-name edit/);
  metadataDraft = null;
  const opening = await invoke("drowse_open_chat", { id: original.id });
  for (let attempt = 0; jobs.get(opening.id).state !== "completed" && attempt < 10; attempt++) await new Promise(resolve => setImmediate(resolve));
  assert.equal(homeNavigations, 1);
  assert.equal(jobs.get(opening.id).result.state, "model_required", "missing model routes to setup without claiming the chat opened");
  busy = true;
  await assert.rejects(invoke("drowse_delete_chat", { id: original.id }), /Finish the current/);
  busy = false;
  await invoke("drowse_delete_chat", { id: duplicate.chat.id });
  assert.equal(rows.has(duplicate.chat.id), false);
  await assert.rejects(invoke("drowse_delete_chat", { id: duplicate.chat.id }), /no longer exists/);
  unsubscribe();
  console.log("WebMCP home: pre-runtime imports, atomic metadata conflicts, human draft protection, copies, text/file backups, failed writes, busy state, and model-required routing passed");
} finally { await server.close(); }

function snapshot() {
  const node = { id: "root", parent_id: null, role: "system", text: "Be helpful", role_label: null, thinking_text: null, aggregate_readings: {}, applied_steering: null, finish_reason: null, starred: false, notes: "", created_at: 100, edited_at: null, edit_count: 0, mean_logprob: null, mean_surprise: null, recipe: null, tokens: null, thinking_tokens: null, raw_token_ids: null };
  return {
    version: 7, savedAt: "2026-09-12T12:00:00.000Z", model_id: "fixture/model", session_id: "default",
    tree: { tree_format: 2, drowse_version: "test", model_id: "fixture/model", session_id: "default", name: null, rev: 1, root_id: "root", active_node_id: "root", nodes: [node], children_of: { root: [] }, cast: {} },
    steerRack: [], subspaceAlong: 0.5, customSteeringExpression: null,
    probeRack: { sortMode: "name", active: [], entries: [] }, highlightState: { target: null, compareTarget: null, compareTwo: false, smoothBlend: false },
    samplingState: { temperature: 0.7, top_p: 0.95, top_k: null, max_tokens: 256, seed: 42, system_prompt: "Be helpful", stop_sequences: "", logit_bias_text: "", presence_penalty: 0, frequency_penalty: 0, thinking: false, return_top_k: 8, user_role: "user", assistant_role: "assistant" },
  };
}
