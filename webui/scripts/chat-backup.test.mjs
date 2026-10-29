import assert from "node:assert/strict";
import { createServer } from "vite";

const server = await createServer({ configFile: false, appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
try {
  const { ConversationLibrary } = await server.ssrLoadModule("/src/lib/conversationLibrary.ts");
  const { CONVERSATION_SAMPLING_KEYS } = await server.ssrLoadModule("/src/lib/conversationSnapshot.ts");
  const { encodeChatBackup, encodeChatArchive, importChatBackup, readChatBackup, readChatBackupFiles, findImportedChat, saveChatImport, chatBackupFilename, MAX_CHAT_BACKUP_BYTES } = await server.ssrLoadModule("/src/lib/chatBackup.ts");
  assert.equal(chatBackupFilename(" My backup.drowse-chat.json "), "My-backup.drowsechat");
  assert.equal(chatBackupFilename("My backup.drowsechat"), "My-backup.drowsechat");
  assert.equal(chatBackupFilename("chat.json"), "chat.drowsechat");
  assert.equal(chatBackupFilename("../../study / copy"), "..-..-study-copy.drowsechat");
  assert.equal(chatBackupFilename("  "), "chat.drowsechat");
  assert.equal(chatBackupFilename("x".repeat(200)), `${"x".repeat(80)}.drowsechat`);
  assert.equal(chatBackupFilename("マーモットの研究"), "マーモットの研究.drowsechat", "Japanese names survive");
  assert.equal(chatBackupFilename("Исследование сурков"), "Исследование-сурков.drowsechat", "Cyrillic names survive");
  assert.equal(chatBackupFilename("بحث المرموط"), "بحث-المرموط.drowsechat", "Arabic names survive");
  assert.equal(chatBackupFilename("Cafe\u0301 notes"), "Café-notes.drowsechat", "names are NFC-normalized");
  assert.equal(chatBackupFilename('a<b>:c"d|e?f*g\\h\u0000i'), "a-b-c-d-e-f-g-h-i.drowsechat", "unsafe and control characters are replaced");
  assert.equal(chatBackupFilename("trailing dots..."), "trailing-dots.drowsechat");
  assert.equal(chatBackupFilename("研".repeat(200)), `${"研".repeat(80)}.drowsechat`, "the length cap counts characters");
  const rows = new Map();
  let sequence = 0;
  let failWrites = false;
  const library = new ConversationLibrary({
    samplingKeys: CONVERSATION_SAMPLING_KEYS, now: () => 2000, randomId: () => `copy-${++sequence}`,
    store: {
      async initialize() {},
      async read(id) { return structuredClone(rows.get(id)); },
      async write(record) { if (failWrites) throw new Error("Storage full"); rows.set(record.id, structuredClone(record)); },
      async list() { return [...rows].map(([key, value]) => ({ key, value })); },
      async delete(id) { return rows.delete(id); },
    },
  });
  const original = await library.create({ name: "Marmots / research", avatarSeed: "original-avatar", accent: "mint", modelType: "base", snapshot: snapshot(), createdAt: 1000, updatedAt: 1500 });
  const encoded = await encodeChatBackup(original);
  const backup = JSON.parse(encoded);
  assert.equal(backup.format, "drowse-chat-backup");
  assert.equal(backup.version, 1);
  assert.match(backup.sha256, /^[a-f0-9]{64}$/);
  assert.deepEqual(backup.conversation, original);
  assert.equal(backup.conversation.modelType, "base");

  for (let i = 0; i < 2; i++) {
    const imported = await importChatBackup(file(encoded), library);
    assert.notEqual(imported.id, original.id);
    assert.deepEqual({ ...imported, id: original.id }, original, "preserve every field except the new library id");
  }
  assert.equal(rows.size, 3, "duplicate imports never overwrite an existing chat");
  const candidate = await readChatBackup(file(encoded));
  assert.equal((await findImportedChat(candidate, library))?.id, original.id, "a backup of an existing chat is detected by id");
  const elsewhere = { ...candidate, sourceId: "another-browser" };
  assert.ok(await findImportedChat(elsewhere, library), "an earlier import of the same backup is detected by content");
  const edited = structuredClone(elsewhere);
  edited.snapshot.tree.nodes[2].text = "Marmots whistle.";
  assert.equal(await findImportedChat(edited, library), null, "an edited chat is not reported as already imported");
  const archive = await encodeChatArchive([original, await library.get(original.id)]);
  const unpacked = await readChatBackupFiles({
    name: "drowse-chats.zip", size: archive.byteLength,
    async text() { throw new Error("archives are read as bytes"); },
    async arrayBuffer() { return archive.buffer.slice(archive.byteOffset, archive.byteOffset + archive.byteLength); },
  });
  assert.equal(unpacked.failures.length, 0);
  assert.deepEqual(unpacked.candidates.map(entry => entry.label), ["Marmots-research-2.drowsechat", "Marmots-research.drowsechat"], "archive entries get unique names");
  assert.deepEqual(unpacked.candidates[0].candidate.snapshot, original.snapshot, "archives round-trip every chat");
  const single = await readChatBackupFiles({ name: "one.drowsechat", ...file(encoded) });
  assert.equal(single.candidates.length, 1);
  const copy = await saveChatImport(single.candidates[0].candidate, library);
  assert.notEqual(copy.id, original.id);
  await library.delete(copy.id);
  assert.deepEqual(await library.get(original.id), original);
  const legacy = await importChatBackup(file(JSON.stringify(original.snapshot)), library);
  assert.match(legacy.name, /^New Chat - [A-Z][a-z]{2} \d{1,2} - \d{1,2}:\d{2} \S+$/);
  assert.deepEqual(legacy.snapshot, original.snapshot, "legacy version 7 exports remain importable");
  assert.equal(legacy.accent, undefined, "legacy chats use the default purple accent");
  assert.equal(legacy.modelType, undefined, "legacy imports must not guess a model's type");

  const before = structuredClone([...rows]);
  const tampered = structuredClone(backup);
  tampered.conversation.snapshot.tree.nodes[2].tokens[0].top_alts[0].logprob = -99;
  await assert.rejects(importChatBackup(file(JSON.stringify(tampered)), library), /integrity check/);
  await assert.rejects(importChatBackup(file(JSON.stringify(tampered)), library), /damaged or incomplete/, "the checksum message claims only what it detects");
  await assert.rejects(importChatBackup(file("{broken"), library), /not valid/);
  await assert.rejects(importChatBackup(file("null"), library), /Choose a Drowse/);
  await assert.rejects(importChatBackup(file(JSON.stringify({ ...backup, version: 99 })), library), /not supported/);
  await assert.rejects(importChatBackup({ size: MAX_CHAT_BACKUP_BYTES + 1, text() { throw new Error("must not read oversized files"); } }, library), /65 MiB/);
  for (const mutate of [
    tree => { tree.nodes.push(structuredClone(tree.nodes[1])); },
    tree => { tree.active_node_id = "missing"; },
    tree => { tree.children_of.root.push("missing"); },
    tree => { tree.nodes[1].parent_id = "alternate"; },
    tree => { tree.nodes[2].tokens[0].top_alts[0].logprob = "invalid"; },
    tree => { tree.tree_format = 999; },
  ]) {
    const malformed = structuredClone(original.snapshot);
    mutate(malformed.tree);
    await assert.rejects(importChatBackup(file(JSON.stringify(malformed)), library));
  }
  failWrites = true;
  await assert.rejects(importChatBackup(file(encoded), library), /Storage full/);
  assert.deepEqual([...rows], before, "all rejected imports leave existing records unchanged");
  console.log("chat backups: full-fidelity round trips, metadata, duplicate copies, legacy imports, integrity, tree validation, limits, and write failures passed");
} finally { await server.close(); }

function file(text) { return { size: Buffer.byteLength(text), async text() { return text; } }; }

function node(id, parent_id, role, text) {
  return { id, parent_id, role, text, role_label: null, thinking_text: null, aggregate_readings: {}, applied_steering: null, finish_reason: null, starred: false, notes: "", created_at: 100, edited_at: null, edit_count: 0, mean_logprob: null, mean_surprise: null, recipe: null, tokens: null, thinking_tokens: null, raw_token_ids: null };
}

function snapshot() {
  const root = node("root", null, "system", "Be precise");
  const user = node("user", "root", "user", "Tell me about marmots");
  const reply = node("reply", "user", "assistant", "Marmots hibernate.");
  const alternate = node("alternate", "user", "assistant", "They live in burrows.");
  reply.notes = "Keep this branch";
  reply.starred = true;
  reply.thinking_text = "Consider winter behavior";
  reply.thinking_tokens = [{ text: "Consider", token_id: 4, raw_index: 0, logprob: -0.5 }];
  reply.raw_token_ids = [4, 5];
  reply.tokens = [{
    text: "Marmots", token_id: 5, raw_index: 1, logprob: -0.2, sampler_entropy: 1.1,
    top_alts: [{ id: 5, text: "Marmots", logprob: -0.2 }, { id: 6, text: "They", logprob: -2.4 }],
    measurements: {
      version: 1, scope: "token", provenance: { source: "recorded" },
      instruments: {
        lens: { readout: { layers: [{ layer: 0, tokens: [{ id: 5, token: "Marmots", logprob: -1.2 }] }], aggregate: [{ token: "Marmots", strength: 0.3 }] } },
        sae: { readout: { layer: 13, features: [{ id: 396, activation: 975.08 }] } },
        geometry: { readings: { calm: { coords: [0.2], per_layer: { 13: [0.3] } } } },
      },
    },
  }];
  reply.recipe = { steering: "0.3 jlens/calm", sampling: null, thinking: true, seed: 42, probes: ["jlens/calm"], probe_hashes: { "jlens/calm": "source-hash" } };
  alternate.edited_at = 110;
  alternate.edit_count = 1;
  return {
    version: 7, savedAt: "2026-09-05T12:00:00.000Z", model_id: "fixture/model", session_id: "default",
    tree: { tree_format: 2, drowse_version: "0.1", model_id: "fixture/model", session_id: "default", name: "Research", rev: 4, root_id: "root", active_node_id: "reply", nodes: [root, user, reply, alternate], children_of: { root: ["user"], user: ["reply", "alternate"], reply: [], alternate: [] }, cast: {} },
    steerRack: [{ name: "jlens/calm", mode: "jlens", alpha: 0.3, trigger: "RESPONSE", enabled: true }], subspaceAlong: 0.5, customSteeringExpression: null,
    probeRack: { sortMode: "name", active: ["jlens/calm"], entries: [{ name: "jlens/calm", request: { selector: "jlens/calm", name: "jlens/calm" }, sparkline: [0.1, 0.2], current: 0.2, previous: 0.1 }] },
    highlightState: { target: "jlens/calm", compareTarget: null, compareTwo: false, smoothBlend: true },
    samplingState: { temperature: 0.7, top_p: 0.95, top_k: null, max_tokens: 256, seed: 42, system_prompt: "Be precise", stop_sequences: "", logit_bias_text: "", presence_penalty: 0, frequency_penalty: 0, thinking: true, return_top_k: 8, user_role: "user", assistant_role: "assistant" },
  };
}
