import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

const samplingState = {
  temperature: 0.7,
  top_p: 0.95,
  top_k: null,
  max_tokens: 256,
  seed: null,
  system_prompt: "",
  stop_sequences: "",
  logit_bias_text: "",
  presence_penalty: 0,
  frequency_penalty: 0,
  thinking: false,
  return_top_k: 8,
  user_role: "user",
  assistant_role: "assistant",
};

try {
  const {
    ConversationLibrary,
    ConversationLibraryError,
    defaultConversationName,
    displayModelName,
    summarizeConversation,
  } = await server.ssrLoadModule("/src/lib/conversationLibrary.ts");

  const clock = sequence([1_000, 2_000, 3_000, 4_000, 5_000]);
  const ids = sequence(["chat-a", "avatar-a", "chat-b", "avatar-b"]);
  const store = memoryStore();
  const library = new ConversationLibrary({
    samplingKeys: Object.keys(samplingState),
    store,
    now: clock,
    randomId: ids,
    runExclusive: serialRunner(),
  });

  const first = await library.create({
    name: "  Pirate   ideas  ",
    snapshot: snapshot("fixture/qwen3-1.7b", "Tell me a pirate story"),
  });
  assert.equal(first.id, "chat-a");
  assert.equal(first.name, "Pirate ideas");
  assert.equal(first.avatarSeed, "avatar-a");
  assert.equal(first.createdAt, 1_000);
  first.name = "mutated outside the repository";
  assert.equal((await library.get("chat-a")).name, "Pirate ideas");

  const second = await library.create({
    name: "Marmot notes",
    snapshot: snapshot("google/gemma-3-1b-it", "What do marmots do?"),
  });
  assert.equal(second.id, "chat-b");
  assert.equal(second.avatarSeed, "avatar-b");
  assert.deepEqual((await library.list()).conversations.map((row) => row.id), ["chat-b", "chat-a"]);
  assert.equal(await library.hasAny(), true);
  assert.deepEqual((await library.listSummaries()).conversations,
    (await library.list()).conversations.map(summarizeConversation));
  const summariesOnlyStore = {
    ...store,
    async list() { throw new Error("full transcript listing must not run"); },
    async map(project) {
      return [...this.rows].map(([key, value]) => project({ key, value }));
    },
    async hasAny() { return this.rows.size > 0; },
  };
  const summaryLibrary = new ConversationLibrary({
    samplingKeys: Object.keys(samplingState), store: summariesOnlyStore,
  });
  assert.equal(await summaryLibrary.hasAny(), true);
  const summaries = await summaryLibrary.listSummaries();
  const reopened = await summaryLibrary.findForTree(second.modelId, second.snapshot.tree.root_id);
  assert.deepEqual(reopened, second);
  reopened.snapshot.steerRack.length = 0;
  assert.deepEqual((await library.get(second.id)).snapshot.steerRack, second.snapshot.steerRack);
  assert.equal(await summaryLibrary.findForTree(second.modelId, "unrelated-root"), null);
  assert.equal(await summaryLibrary.findForTree("unrelated-model", second.snapshot.tree.root_id), null);
  assert.equal(summaries.conversations[0].messageCount, 1);
  assert.equal(summaries.conversations[0].threadCount, 1);
  const branched = structuredClone(second);
  branched.snapshot.tree.nodes.push(
    { id: "reply-a", parent_id: "user-1", role: "assistant", text: "First reply" },
    { id: "reply-b", parent_id: "user-1", role: "assistant", text: "Other reply" },
    { id: "pending", parent_id: "reply-b", role: "assistant", text: "" },
  );
  branched.snapshot.tree.children_of["user-1"] = ["reply-a", "reply-b"];
  branched.snapshot.tree.children_of["reply-b"] = ["pending"];
  assert.equal(summarizeConversation(branched).messageCount, 3, "count messages across branches, excluding empty continuations and the system root");
  assert.equal(summarizeConversation(branched).threadCount, 2, "count leaf paths, not every node or shared prefix");
  const rootOnly = structuredClone(second);
  rootOnly.snapshot.tree.nodes = rootOnly.snapshot.tree.nodes.slice(0, 1);
  rootOnly.snapshot.tree.children_of = { root: [] };
  assert.equal(summarizeConversation(rootOnly).messageCount, 0);
  assert.equal(summarizeConversation(rootOnly).threadCount, 0);
  assert.ok(summaries.conversations.every((record) => !("snapshot" in record)));
  summaries.conversations[0].name = "changed card";
  assert.equal((await library.get("chat-b")).name, "Marmot notes");

  const updated = await library.update("chat-a", {
    name: "Captain's log",
    avatarSeed: "new-face",
    snapshot: snapshot("fixture/qwen3-1.7b", "Tell me a pirate story", 2),
  });
  assert.equal(updated.createdAt, 1_000);
  assert.equal(updated.updatedAt, 3_000);
  assert.equal(updated.avatarSeed, "new-face");
  assert.equal(updated.snapshot.tree.rev, 2);
  assert.deepEqual((await library.list()).conversations.map((row) => row.id), ["chat-a", "chat-b"]);

  const renamed = await library.update("chat-b", { name: "A renamed chat" });
  assert.equal(renamed.updatedAt, second.updatedAt, "renaming does not change chat activity time");
  assert.deepEqual((await library.list()).conversations.map((row) => row.id), ["chat-a", "chat-b"]);

  const avatarOnly = await library.update("chat-b", { avatarSeed: "another-face" });
  assert.equal(avatarOnly.updatedAt, second.updatedAt, "avatar changes do not change chat activity time");
  assert.equal(avatarOnly.avatarSeed, "another-face");
  assert.deepEqual((await library.list()).conversations.map((row) => row.id), ["chat-a", "chat-b"]);
  const colored = await library.update("chat-b", { accent: "mint" });
  assert.equal(colored.accent, "mint");
  assert.equal(colored.updatedAt, second.updatedAt);
  assert.equal(summarizeConversation(colored).accent, "mint");
  assert.equal((await library.autosave(colored.snapshot, colored.id)).accent, "mint");
  const revised = structuredClone(colored.snapshot);
  revised.tree.rev += 1;
  assert.equal((await library.autosave(revised, colored.id)).accent, "mint");
  await assert.rejects(library.update("chat-b", { accent: "not-a-color" }), /accent color is invalid/);

  assert.equal(await library.delete("chat-a"), true);
  assert.equal(await library.delete("chat-a"), false);
  await assert.rejects(
    library.get("chat-a"),
    (error) => error instanceof ConversationLibraryError && error.code === "NOT_FOUND",
  );

  store.rows.set("future-chat", { id: "future-chat", schemaVersion: 99, name: "Future" });
  const withIssue = await library.list();
  assert.equal(withIssue.issues.length, 1);
  assert.equal(withIssue.issues[0].id, "future-chat");
  assert.equal(store.rows.has("future-chat"), true, "invalid data must never be deleted during reads");
  assert.deepEqual((await summaryLibrary.listSummaries()).issues, withIssue.issues);
  assert.equal(await library.delete("future-chat"), true, "explicit deletion remains available");
  const onlyInvalid = new ConversationLibrary({
    samplingKeys: Object.keys(samplingState), store: memoryStore(new Map([["invalid", null]])),
  });
  assert.equal(await onlyInvalid.hasAny(), true, "damaged chats still lead to the library, not onboarding");
  assert.equal((await onlyInvalid.listSummaries()).issues.length, 1);
  assert.equal(await new ConversationLibrary({ samplingKeys: [], store: memoryStore() }).hasAny(), false);

  await assert.rejects(
    library.create({ name: " ", snapshot: snapshot("fixture/model", "Hello") }),
    (error) => error instanceof ConversationLibraryError && error.code === "INVALID_RECORD",
  );
  await assert.rejects(
    library.create({ name: "x".repeat(121), snapshot: snapshot("fixture/model", "Hello") }),
    /120 characters or fewer/,
  );

  const named = snapshot("fixture/model", "  A question about   systems and policy  ");
  assert.equal(defaultConversationName(named), "A question about systems and policy");
  assert.equal(displayModelName("Qwen/Qwen3-1.7B"), "Qwen3 1.7B");
  assert.equal(displayModelName("google/gemma-3-4b-it"), "Gemma 3 4B it");

  let nextAutoId = 0;
  const autoLibrary = new ConversationLibrary({ samplingKeys: Object.keys(samplingState), store: memoryStore(),
    randomId: () => `auto-${++nextAutoId}`, runExclusive: serialRunner() });
  const initial = snapshot("fixture/auto", "First automatic chat");
  const [autoFirst, autoConcurrent] = await Promise.all([
    autoLibrary.autosave(initial, null), autoLibrary.autosave(initial, null),
  ]);
  assert.equal(autoFirst.id, autoConcurrent.id, "concurrent first saves must not duplicate chats");
  assert.equal(autoFirst.name, "Chat 1");
  assert.ok(autoFirst.avatarSeed);
  await autoLibrary.update(autoFirst.id, { name: "My name", avatarSeed: "chosen-face" });
  const autoUpdated = await autoLibrary.autosave(snapshot("fixture/auto", "Second message", 2), autoFirst.id);
  assert.equal(autoUpdated.name, "My name");
  assert.equal(autoUpdated.avatarSeed, "chosen-face");
  assert.equal(autoUpdated.createdAt, autoFirst.createdAt);
  const resumed = await autoLibrary.autosave(snapshot("fixture/auto", "Second message", 2), null);
  assert.equal(resumed.id, autoFirst.id, "reopening a model must reuse the root's saved chat");
  await assert.rejects(autoLibrary.autosave(initial, autoFirst.id), /newer version/);
  await autoLibrary.delete(autoFirst.id);
  await assert.rejects(autoLibrary.autosave(initial, autoFirst.id), /deleted/);
  assert.equal((await autoLibrary.list()).conversations.length, 0, "autosave must not resurrect a deleted active chat");
  const empty = snapshot("fixture/auto", "");
  empty.tree.nodes = empty.tree.nodes.slice(0, 1);
  empty.tree.children_of = { root: [] };
  empty.tree.active_node_id = "root";
  assert.equal(await autoLibrary.autosave(empty, null), null, "empty visits must not create chats");
  const [numberedFirst, numberedSecond] = await Promise.all([
    autoLibrary.autosave(snapshot("fixture/one", "One"), null),
    autoLibrary.autosave(snapshot("fixture/two", "Two"), null),
  ]);
  assert.deepEqual([numberedFirst.name, numberedSecond.name], ["Chat 1", "Chat 2"]);
  await autoLibrary.update(numberedFirst.id, { name: "Research notes" });
  const third = await autoLibrary.autosave(snapshot("fixture/three", "Three"), null);
  assert.equal(third.name, "Chat 3");
  await autoLibrary.update(numberedSecond.id, { name: "Chat 4" });
  const fourth = await autoLibrary.autosave(snapshot("fixture/four", "Four"), null);
  assert.equal(fourth.name, "Chat 5", "automatic names skip existing user-chosen names");

  let copyId = 0;
  const copyStore = memoryStore();
  const copies = new ConversationLibrary({
    samplingKeys: Object.keys(samplingState), store: copyStore,
    randomId: () => `copy-${++copyId}`, runExclusive: serialRunner(),
  });
  const tiedStore = memoryStore();
  let tiedId = 0;
  const tied = new ConversationLibrary({ samplingKeys: Object.keys(samplingState), store: tiedStore, now: () => 5_000, randomId: () => `tied-${++tiedId}` });
  const tiedFirst = await tied.create({ name: "Zulu", snapshot: snapshot("fixture/tied", "One") });
  await tied.create({ name: "Middle", snapshot: snapshot("fixture/tied", "Two") });
  const tiedOrder = (await tied.listSummaries()).conversations.map(row => row.id);
  await tied.update(tiedFirst.id, { name: "Alpha" });
  const reloaded = new ConversationLibrary({ samplingKeys: Object.keys(samplingState), store: tiedStore });
  assert.deepEqual((await reloaded.listSummaries()).conversations.map(row => row.id), tiedOrder, "equal-timestamp chats keep their order after renaming and reloading");
  const originalSnapshot = snapshot("fixture/base", "The opening line. ");
  originalSnapshot.tree.nodes.push({
    id: "answer-1", parent_id: "user-1", role: "assistant", text: "One path.",
    recipe: { steering: "0.2 jlens/bright", seed: 12 },
    tokens: [{ text: "One", id: 17, logprob: -0.2, raw_index: 0 }],
    raw_token_ids: [17], starred: true, notes: "Keep this path",
  }, {
    id: "answer-2", parent_id: "user-1", role: "assistant", text: "Another path.", recipe: null,
  });
  originalSnapshot.tree.children_of["user-1"] = ["answer-1", "answer-2"];
  originalSnapshot.tree.children_of["answer-1"] = [];
  originalSnapshot.tree.children_of["answer-2"] = [];
  originalSnapshot.tree.active_node_id = "answer-1";
  originalSnapshot.samplingState.temperature = .85;
  const original = await copies.create({ name: "A branching completion", avatarSeed: "same-face", accent: "mint", modelType: "base", snapshot: originalSnapshot });
  const copy = await copies.duplicate(original.id);
  assert.equal(copy.name, "A branching completion (copy)");
  assert.equal(copy.avatarSeed, original.avatarSeed);
  assert.equal(copy.accent, "mint");
  assert.equal(copy.modelType, "base");
  assert.notEqual(copy.id, original.id);
  assert.notEqual(copy.snapshot.tree.root_id, original.snapshot.tree.root_id);
  assert.equal(copy.snapshot.tree.nodes.length, 4);
  assert.equal(summarizeConversation(copy).threadCount, 2);
  assert.deepEqual(copy.snapshot.samplingState, original.snapshot.samplingState);
  const oldIds = new Set(original.snapshot.tree.nodes.map(node => node.id));
  const newIds = new Set(copy.snapshot.tree.nodes.map(node => node.id));
  for (const [index, node] of copy.snapshot.tree.nodes.entries()) {
    assert.equal(oldIds.has(node.id), false);
    assert.equal(node.parent_id === null || newIds.has(node.parent_id), true);
    const { id, parent_id, ...content } = node;
    const { id: oldId, parent_id: oldParent, ...oldContent } = original.snapshot.tree.nodes[index];
    assert.deepEqual(content, oldContent, "duplication preserves every token, recipe, note, and flag");
  }
  const selectedCopy = copy.snapshot.tree.nodes.find(node => node.id === copy.snapshot.tree.active_node_id);
  assert.equal(selectedCopy.text, "One path.");
  selectedCopy.text = "Edited copy only";
  copy.snapshot.tree.rev += 1;
  await copies.autosave(copy.snapshot, null, "base");
  assert.equal((await copies.findForTree(copy.modelId, copy.snapshot.tree.root_id)).id, copy.id);
  assert.deepEqual(await copies.get(original.id), original, "copy autosave never touches the original");
  await copies.autosave(original.snapshot, null, "base");
  assert.equal((await copies.findForTree(original.modelId, original.snapshot.tree.root_id)).id, original.id);
  const anotherCopy = await copies.duplicate(original.id);
  assert.notEqual(anotherCopy.snapshot.tree.root_id, copy.snapshot.tree.root_id);
  const longName = await copies.update(original.id, { name: "x".repeat(120) });
  assert.equal((await copies.duplicate(longName.id)).name.length, 120);
  const beforeFailure = structuredClone([...copyStore.rows]);
  copyStore.write = async () => { throw new Error("Storage quota exceeded"); };
  await assert.rejects(copies.duplicate(original.id), /Storage quota exceeded/);
  assert.deepEqual([...copyStore.rows], beforeFailure, "failed copies never alter stored chats");
  await assert.rejects(copies.duplicate("missing-chat"), /no longer exists/);

  process.stdout.write("conversation library tests passed\n");
} finally {
  await server.close();
}

function snapshot(modelId, prompt, rev = 1) {
  return {
    version: 7,
    savedAt: "2026-09-03T12:00:00.000Z",
    model_id: modelId,
    session_id: "default",
    tree: {
      tree_format: 2,
      drowse_version: "fixture",
      model_id: modelId,
      session_id: "default",
      name: null,
      root_id: "root",
      active_node_id: "user-1",
      rev,
      nodes: [
        { id: "root", parent_id: null, role: "system", text: "", recipe: null },
        { id: "user-1", parent_id: "root", role: "user", text: prompt, recipe: null },
      ],
      children_of: { root: ["user-1"], "user-1": [] },
      cast: {},
    },
    steerRack: [],
    subspaceAlong: 0.5,
    customSteeringExpression: null,
    probeRack: { sortMode: "name", active: [], entries: [] },
    highlightState: {
      target: null,
      compareTarget: null,
      compareTwo: false,
      smoothBlend: false,
    },
    samplingState: { ...samplingState },
  };
}

function memoryStore(initial = new Map()) {
  return {
    rows: new Map(initial),
    async initialize() {},
    async list() {
      return [...this.rows].map(([key, value]) => ({ key, value: structuredClone(value) }));
    },
    async read(id) {
      const value = this.rows.get(id);
      return value === undefined ? undefined : structuredClone(value);
    },
    async write(record) {
      this.rows.set(record.id, structuredClone(record));
    },
    async delete(id) {
      return this.rows.delete(id);
    },
  };
}

function sequence(values) {
  let index = 0;
  return () => {
    if (index >= values.length) throw new Error("sequence exhausted");
    return values[index++];
  };
}

function serialRunner() {
  let tail = Promise.resolve();
  return async (operation) => {
    const previous = tail;
    let release;
    tail = new Promise((resolve) => { release = resolve; });
    await previous;
    try {
      return await operation();
    } finally {
      release();
    }
  };
}
