import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const requests = [];
  const filters = [];
  const navigations = [];
  let treeReads = 0;
  let fetchedSnapshot;
  const { installRuntimeClient } = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
  installRuntimeClient({ mode: "browser", tree: {
    get: async () => { treeReads += 1; return fetchedSnapshot; },
    edgeLabel: (parent, child) => new Promise((resolve, reject) => requests.push({ parent, child, resolve, reject })),
    filter: expr => new Promise((resolve, reject) => filters.push({ expr, resolve, reject })),
    navigate: nodeId => new Promise(resolve => navigations.push({ nodeId, resolve })),
  } });
  const { applyTreeSnapshot, applyTreeDelta, loomTree, syncChatLogFromTree, refreshLoomTree } =
    await server.ssrLoadModule("/src/lib/stores/loom.svelte.ts");
  const { fetchEdgeLabel, edgeLabelCache, invalidateEdgeLabels, applyTreeFilter, clearTreeFilter, filterState, loomUiState } =
    await server.ssrLoadModule("/src/lib/stores/loomUi.svelte.ts");
  const { chatLog, genStatus, clearChat } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
  const { enqueuePending, drainNextPendingAction, pendingActions } =
    await server.ssrLoadModule("/src/lib/stores/pending.svelte.ts");
  const node = (id, parent_id, steering = null) => ({
    id, parent_id, role: parent_id === null ? "system" : "assistant", text: id,
    role_label: null, recipe: { steering }, applied_steering: steering,
    finish_reason: "stop", aggregate_readings: {}, tokens: [], thinking_tokens: [],
  });
  const snapshot = {
    tree_format: 2, drowse_version: "test", session_id: "default", model_id: "fixture",
    name: null, root_id: "root", active_node_id: "b", rev: 1,
    nodes: [node("root", null), node("a", "root"), node("b", "a", "0.3 calm"), node("c", "root")],
    children_of: { root: ["a", "c"], a: ["b"], b: [], c: [] }, cast: {},
  };
  applyTreeSnapshot(snapshot);
  fetchEdgeLabel("a", "b");
  fetchEdgeLabel("a", "b");
  assert.equal(requests.length, 1);
  requests[0].resolve({ label: "calm" });
  await new Promise((resolve) => setImmediate(resolve));
  applyTreeDelta({ rev: 2, active_node_id: "c" });
  applyTreeDelta({ rev: 3, updated: [{ ...loomTree.nodes.get("b"), starred: true }] });
  fetchEdgeLabel("a", "b");
  assert.equal(requests.length, 1, "navigation and stars reuse edge labels");
  edgeLabelCache.set("root|c", "unrelated");
  applyTreeDelta({ rev: 4, updated: [{ ...loomTree.nodes.get("a"), applied_steering: "0.2 calm" }] });
  assert.equal(edgeLabelCache.has("a|b"), false, "parent steering changes invalidate outgoing edges");
  assert.equal(edgeLabelCache.get("root|c"), "unrelated");
  fetchEdgeLabel("a", "b");
  applyTreeDelta({ rev: 5, updated: [{ ...loomTree.nodes.get("b"), applied_steering: "0.4 calm" }] });
  fetchEdgeLabel("a", "b");
  requests[1].resolve({ label: "stale" });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(edgeLabelCache.has("a|b"), false);
  fetchEdgeLabel("a", "b");
  assert.equal(requests.length, 3, "stale completions cannot remove a newer in-flight request");
  requests[2].resolve({ label: "fresh" });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(edgeLabelCache.get("a|b"), "fresh");
  applyTreeDelta({ rev: 6, removed: ["b"] });
  assert.equal(edgeLabelCache.has("a|b"), false);
  applyTreeSnapshot({ ...snapshot, rev: 7 });
  assert.equal(edgeLabelCache.size, 0);
  fetchEdgeLabel("a", "b");
  applyTreeSnapshot({ ...snapshot, rev: 8 });
  requests[3].resolve({ label: "previous chat" });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(edgeLabelCache.size, 0, "restored chats cannot receive old edge responses");

  genStatus.active = true;
  loomTree.pendingNodeId = "b";
  applyTreeDelta({ rev: 9, updated: [{ ...loomTree.nodes.get("b"), text: "", finish_reason: null }] });
  const pending = chatLog.turns.at(-1);
  pending.text = "A reply is streaming";
  pending.tokens = [{ text: pending.text, rawIndex: 0 }];
  applyTreeDelta({ rev: 10, updated: [{ ...loomTree.nodes.get("c"), starred: true }] });
  assert.equal(chatLog.turns.at(-1).text, "A reply is streaming");
  assert.equal(chatLog.turns.at(-1).tokens.length, 1);
  chatLog.turns.find = () => { throw new Error("quadratic turn search"); };
  syncChatLogFromTree();
  applyTreeDelta({ rev: 11, updated: [{ ...loomTree.nodes.get("b"), text: "Final reply", finish_reason: "stop" }] });
  assert.equal(chatLog.turns.at(-1).text, "Final reply");
  genStatus.active = false;
  edgeLabelCache.set("a|b", "cached");
  edgeLabelCache.set("root|c", "unchanged");
  const currentSnapshot = { ...snapshot, rev: 12, nodes: [...loomTree.nodes.values()] };
  applyTreeSnapshot(currentSnapshot, { reconcileEdgeLabels: true });
  assert.equal(edgeLabelCache.get("a|b"), "cached", "same-tree refreshes preserve valid labels");
  applyTreeSnapshot({ ...currentSnapshot, rev: 13, nodes: currentSnapshot.nodes.map((n) =>
    n.id === "b" ? { ...n, applied_steering: "0.8 calm" } : n,
  ) }, { reconcileEdgeLabels: true });
  assert.equal(edgeLabelCache.has("a|b"), false, "refreshes invalidate changed steering");
  assert.equal(edgeLabelCache.get("root|c"), "unchanged");
  applyTreeSnapshot({ ...currentSnapshot, rev: 14, model_id: "another-model" }, { reconcileEdgeLabels: true });
  assert.equal(edgeLabelCache.size, 0, "changing models invalidates all labels");
  genStatus.startedAt = 1;
  genStatus.tokensSoFar = 24;
  genStatus.tokPerSec = 7.3;
  assert.equal(applyTreeSnapshot({ ...currentSnapshot, rev: 1 }), true,
    "a different model sharing the default session id can have a lower revision");
  assert.equal(loomTree.modelId, "fixture");
  assert.equal(loomTree.rev, 1);
  assert.equal(genStatus.startedAt, null, "another model cannot inherit the previous run's status");
  assert.equal(genStatus.tokensSoFar, 0);
  assert.equal(genStatus.tokPerSec, 0);
  assert.equal(applyTreeSnapshot({ ...currentSnapshot, rev: 0 }), false,
    "stale revisions are still rejected within the same model and session");
  assert.equal(loomTree.rev, 1);
  genStatus.active = true;
  genStatus.tokensSoFar = 17;
  loomTree.pendingNodeId = "b";
  const freshTree = { ...snapshot, root_id: "fresh", active_node_id: "fresh", rev: 0,
    nodes: [{ ...node("fresh", null), text: "" }], children_of: { fresh: [] } };
  assert.equal(applyTreeSnapshot(freshTree), true,
    "a new chat in the same model and default session starts at revision zero");
  assert.equal(loomTree.root_id, "fresh");
  assert.equal(loomTree.nodes.has("b"), false);
  assert.equal(chatLog.turns.length, 0);
  assert.equal(loomTree.pendingNodeId, null);
  assert.equal(genStatus.active, false);
  assert.equal(genStatus.tokensSoFar, 0);
  applyTreeSnapshot({ ...currentSnapshot, rev: 1 });
  fetchEdgeLabel("a", "b");
  requests.at(-1).reject(new Error("temporary connection failure"));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(edgeLabelCache.has("a|b"), false, "failed labels must not become permanent empty results");
  const failedRequestCount = requests.length;
  fetchEdgeLabel("a", "b");
  assert.equal(requests.length, failedRequestCount + 1);
  requests.at(-1).resolve({ label: "retried" });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(edgeLabelCache.get("a|b"), "retried");
  invalidateEdgeLabels();
  fetchedSnapshot = { ...currentSnapshot, rev: 15 };
  genStatus.active = true;
  await refreshLoomTree();
  assert.equal(treeReads, 0, "a live map does not queue a read behind generation");
  genStatus.active = false;
  await refreshLoomTree();
  assert.equal(treeReads, 1, "refresh works again after generation");
  assert.equal(loomTree.rev, 15);
  genStatus.active = true;
  loomTree.loaded = false;
  await refreshLoomTree();
  assert.equal(treeReads, 2, "an initial snapshot is never suppressed");
  genStatus.active = false;
  applyTreeSnapshot({ ...snapshot, rev: 20, nodes: snapshot.nodes.map(n => ({ ...n,
    text: n.id === "a" ? "Hello,\n WORLD" : n.id === "b" ? "hello world fox" : "elsewhere",
    starred: n.id === "b",
  })) });
  await applyTreeFilter("hello, world", "text");
  assert.deepEqual([...filterState.matchingIds], ["a"]);
  assert.equal(filters.length, 0, "ordinary text never reaches the measurement parser");
  await applyTreeFilter("text:hello, starred", "advanced");
  assert.deepEqual([...filterState.matchingIds], ["b"]);
  const advanced = applyTreeFilter("text:hello, starred, agg:calm>0.5", "advanced");
  assert.equal(filters.at(-1).expr, "agg:calm>0.5");
  filters.at(-1).resolve({ matching_node_ids: ["a", "b", "c"] });
  await advanced;
  assert.deepEqual([...filterState.matchingIds], ["b"], "local and measurement filters intersect");
  const stale = applyTreeFilter("agg:calm>0.1", "advanced");
  await applyTreeFilter("elsewhere", "text");
  filters.at(-1).resolve({ matching_node_ids: ["b"] });
  await stale;
  assert.deepEqual([...filterState.matchingIds], ["root", "c"]);
  const cleared = applyTreeFilter("agg:calm>0.1", "advanced");
  clearTreeFilter();
  filters.at(-1).reject(new Error("stale failure"));
  await cleared;
  assert.equal(filterState.error, null);
  assert.equal(filterState.matchingIds, null);
  assert.equal(filterState.loading, false);
  await applyTreeFilter("text:", "advanced");
  assert.ok(filterState.error);
  await applyTreeFilter("sort:confidence", "advanced");
  assert.equal(loomUiState.siblingSort, "confidence");
  assert.equal(filterState.matchingIds, null);
  await applyTreeFilter(" ", "text");
  assert.equal(loomUiState.siblingSort, "default");
  assert.equal(filterState.error, null);
  genStatus.active = true;
  clearChat();
  let nextActionNode = null;
  enqueuePending({ label: "next message", text: null, awaitsGen: false, rebuild: null,
    apply: () => { nextActionNode = loomTree.active_node_id; } });
  assert.equal(navigations.length, 0, "clearing waits for the active generation");
  genStatus.active = false;
  const drained = drainNextPendingAction();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(navigations.length, 1);
  assert.equal(navigations[0].nodeId, "root");
  assert.equal(nextActionNode, null, "the next action must wait for clearing to finish");
  fetchedSnapshot = { ...snapshot, rev: 21, active_node_id: "root" };
  navigations[0].resolve();
  await drained;
  assert.equal(nextActionNode, "root", "a queued message starts from the cleared conversation");
  assert.equal(chatLog.turns.some(turn => turn.role !== "system"), false);
  assert.equal(loomTree.nodes.size, snapshot.nodes.length, "clearing preserves existing branches");
  assert.equal(pendingActions.queue.length, 0);
  console.log("Loom store invalidation, streaming, and search checks passed");
} finally {
  await server.close();
}
