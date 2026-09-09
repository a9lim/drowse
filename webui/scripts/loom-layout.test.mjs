import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const { segmentLoom } = await server.ssrLoadModule("/src/panels/loom/loomSegments.ts");
  const makeReply = (id, words, parent = "prompt") => ({
    id, parent_id: parent, role: "assistant", role_label: null, recipe: {},
    text: words.join(""), raw_token_ids: words.map(word => word.charCodeAt(0)),
    tokens: words.map((text, raw_index) => ({ text, raw_index, token_id: text.charCodeAt(0), logprob: -0.2 })),
  });
  const prompt = { ...makeReply("prompt", ["Prompt"], null), role: "user", recipe: null };
  const { loomLiveNode } = await server.ssrLoadModule("/src/panels/loom/loomLiveNode.ts");
  const pending = { ...makeReply("pending", ["A"]), finish_reason: null };
  const pendingSaved = JSON.stringify(pending);
  const stream = { nodeId: "pending", text: "AB", tokens: [
    { text: "A", rawIndex: 0, tokenId: 65 },
    { text: "B", rawIndex: 1, tokenId: 66, logprob: -0.4 },
  ] };
  const live = loomLiveNode(pending, stream);
  assert.equal(live.text, "AB");
  assert.deepEqual(live.raw_token_ids, [65, 66]);
  assert.equal(live.tokens[1].logprob, -0.4);
  assert.equal(JSON.stringify(pending), pendingSaved, "live display cannot mutate the saved tree");
  assert.equal(loomLiveNode(pending, { ...stream, nodeId: "elsewhere" }), pending);
  const settled = { ...pending, finish_reason: "stop" };
  assert.equal(loomLiveNode(settled, stream), settled, "settled nodes always use the authoritative record");
  assert.equal(loomLiveNode(pending, { ...stream, tokens: [stream.tokens[0], { ...stream.tokens[1], rawIndex: 3 }] }).raw_token_ids, null,
    "hidden control-token gaps must not invent raw token identities");
  const source = [prompt, makeReply("a", ["A", "B", "C"]), makeReply("b", ["A", "B", "D"]),
    makeReply("c", ["A", "E"]), makeReply("next", ["Next"], "b")];
  const saved = JSON.stringify(source);
  const segmented = segmentLoom(source, new Set(["prompt", "b", "next"]));
  const shared = segmented.filter(part => part.shared);
  assert.deepEqual(shared.map(part => [part.start, part.end, part.memberIds.length]), [[0, 1, 3], [1, 2, 2]]);
  assert.equal(shared[0].node.id, "b", "shared token tools follow the active reply");
  assert.equal(segmented.find(part => part.id === "next").parentId, "b");
  assert.equal(segmented.find(part => part.id === "a").start, 2);
  assert.equal(segmented.find(part => part.id === "c").start, 1);
  const segmentIndex = new Map(segmented.map(part => [part.id, part]));
  for (const original of source) {
    const parts = [];
    let part = segmentIndex.get(original.id);
    while (part && (part.id === original.id || part.shared)) {
      parts.unshift(part.node.tokens.slice(part.start, part.end).map(token => token.token_id));
      part = segmentIndex.get(part.parentId);
    }
    assert.deepEqual(parts.flat(), original.raw_token_ids, "every visible path reconstructs the exact original reply");
  }
  assert.equal(JSON.stringify(source), saved, "projection never edits saved replies");
  const identical = segmentLoom([makeReply("a", ["A"]), makeReply("b", ["A"])], new Set());
  assert.equal(identical.filter(part => part.shared).length, 1);
  assert.equal(identical.filter(part => !part.shared && part.start === part.end).length, 2, "identical replies retain separate endpoints");
  const shorter = segmentLoom([makeReply("a", ["A"]), makeReply("b", ["A", "B"])], new Set());
  assert.equal(shorter.find(part => part.id === "a").start, 1);
  assert.equal(shorter.find(part => part.id === "b").start, 1);
  for (const altered of [
    { ...makeReply("b", ["A"]), raw_token_ids: null },
    { ...makeReply("b", ["A"]), recipe: null },
    { ...makeReply("b", ["A"]), role_label: "pirate" },
    { ...makeReply("b", ["A"]), parent_id: "different-prompt" },
    { ...makeReply("b", ["A"]), tokens: [{ text: "A", token_id: 65, raw_index: 1 }], raw_token_ids: [99, 65] },
  ]) {
    assert.equal(segmentLoom([prompt, makeReply("different-prompt", ["Other"], null), makeReply("a", ["A"]), altered], new Set()).filter(part => part.shared).length, 0);
  }
  const {
    layoutLoom,
    loomMessageWidth,
    loomBranchJunctions,
    LOOM_NODE_HEIGHT,
    LOOM_NODE_WIDTH,
  } = await server.ssrLoadModule("/src/panels/loom/loomLayout.ts");
  const {
    centerLoomRect,
    fitLoomCamera,
    resizeLoomCamera,
    LOOM_MAX_ZOOM,
    LOOM_MIN_ZOOM,
    zoomLoomAt,
  } = await server.ssrLoadModule("/src/panels/loom/loomViewport.ts");
  const { splitTokenSentences } = await server.ssrLoadModule(
    "/src/panels/loom/loomSentences.ts",
  );
  const { pinchMetrics, scaleFromPinch } = await server.ssrLoadModule(
    "/src/lib/pointerGesture.ts",
  );

  assert.deepEqual(
    pinchMetrics({ x: 0, y: 0 }, { x: 6, y: 8 }),
    { center: { x: 3, y: 4 }, distance: 10 },
  );
  assert.equal(scaleFromPinch(1, 100, 150, 0.5, 2), 1.5);
  assert.equal(scaleFromPinch(1, 100, 10, 0.5, 2), 0.5);
  assert.equal(scaleFromPinch(1, 100, 300, 0.5, 2), 2);
  assert.equal(scaleFromPinch(1, 0, 100, 0.5, 2), 1);

  assert.deepEqual(
    splitTokenSentences([
      { text: "Hello" },
      { text: " world. " },
      { text: "How" },
      { text: " are" },
      { text: " you?" },
      { text: " Fine" },
    ]),
    [
      { start: 0, end: 1, text: "Hello world. " },
      { start: 2, end: 4, text: "How are you?" },
      { start: 5, end: 5, text: " Fine" },
    ],
    "sentence groups retain exact token indexes and bytes",
  );
  assert.deepEqual(
    splitTokenSentences([
      { text: "First." },
      { text: "\n\n" },
      { text: "Second." },
    ]),
    [
      { start: 0, end: 0, text: "First." },
      { start: 1, end: 2, text: "\n\nSecond." },
    ],
    "blank line tokens do not become empty sentence branches",
  );

  const layout = layoutLoom([
    { id: "user", parentId: null, depth: 0 },
    { id: "reply-a", parentId: "user", depth: 1 },
    { id: "follow-a", parentId: "reply-a", depth: 2 },
    { id: "reply-b", parentId: "user", depth: 1 },
    { id: "follow-b", parentId: "reply-b", depth: 2 },
  ]);

  assert.equal(layout.nodes.length, 5);
  assert.equal(layout.edges.length, 4);
  const byId = new Map(layout.nodes.map((node) => [node.id, node]));
  const user = byId.get("user");
  const replyA = byId.get("reply-a");
  const replyB = byId.get("reply-b");
  assert.ok(user && replyA && replyB);
  assert.equal(replyA.x, replyB.x, "siblings share a conversation depth");
  assert.ok(replyA.y + LOOM_NODE_HEIGHT < replyB.y, "sibling cards never overlap");
  assert.ok(user.x + LOOM_NODE_WIDTH < replyA.x, "turns advance left to right");
  assert.equal(
    user.y + LOOM_NODE_HEIGHT / 2,
    ((replyA.y + LOOM_NODE_HEIGHT / 2) + (replyB.y + LOOM_NODE_HEIGHT / 2)) / 2,
    "a branch parent is centered between its continuations",
  );
  assert.ok(layout.edges.every((edge) => edge.path.includes(" C ")));
  assert.ok(layout.width > 0 && layout.height > 0);

  assert.equal(loomMessageWidth("Short"), 276);
  assert.equal(loomMessageWidth("x".repeat(181)), 414);
  assert.equal(loomMessageWidth("x".repeat(451)), 552);
  assert.equal(loomMessageWidth("x".repeat(901)), 680);
  assert.equal(loomMessageWidth("x".repeat(100_000)), 680);
  const wide = layoutLoom([
    { id: "wide", parentId: null, depth: 0, width: 680 },
    { id: "small", parentId: "wide", depth: 1, width: 276 },
    { id: "large", parentId: "wide", depth: 1, width: 552 },
    { id: "next", parentId: "small", depth: 2 },
  ]);
  assert.equal(wide.nodes[1].x, 28 + 680 + 112);
  assert.equal(wide.nodes[3].x, wide.nodes[1].x + 552 + 112);
  assert.ok(wide.edges[0].path.startsWith(`M ${28 + 680} `));
  assert.equal(loomBranchJunctions(wide)[0].x, 28 + 680 + 112 * 0.52);
  assert.equal(layoutLoom([{ id: "invalid", parentId: null, depth: 0, width: NaN }]).nodes[0].width, 276);

  const { loomTextMatches, loomSearchExcerpt } = await server.ssrLoadModule("/src/lib/loomSearch.ts");
  assert.equal(loomTextMatches("Hello,\n  WORLD", "hello, world"), true);
  assert.equal(loomTextMatches("hello world", "hello, world"), false);
  assert.equal(loomTextMatches("[literal].*", ".*"), true);
  assert.equal(loomTextMatches("anything", "  "), false);
  assert.equal(loomSearchExcerpt("x".repeat(200) + " HELLO world", "hello").match, "HELLO");
  assert.ok(loomSearchExcerpt("x".repeat(200) + " HELLO world", "hello").before.startsWith("…"));

  const varied = layoutLoom([
    { id: "short", parentId: null, depth: 0, height: 136 },
    { id: "long-a", parentId: "short", depth: 1, height: 236 },
    { id: "long-b", parentId: "short", depth: 1, height: 196 },
  ]);
  const variedById = new Map(varied.nodes.map((node) => [node.id, node]));
  const short = variedById.get("short");
  const longA = variedById.get("long-a");
  const longB = variedById.get("long-b");
  assert.ok(short && longA && longB);
  assert.equal(short.height, 136, "each card keeps its requested content height");
  assert.equal(longA.height, 236);
  assert.ok(longA.y + longA.height < longB.y, "different-height siblings never overlap");
  assert.equal(
    short.y + short.height / 2,
    ((longA.y + longA.height / 2) + (longB.y + longB.height / 2)) / 2,
    "connectors stay centered for different-height cards",
  );
  assert.ok(
    varied.height > Math.max(...varied.nodes.map((node) => node.y + node.height)),
    "the canvas includes every card's full height",
  );

  const tallRoot = layoutLoom([
    { id: "tall-root", parentId: null, depth: 0, height: 236 },
    { id: "short-child", parentId: "tall-root", depth: 1, height: 116 },
  ]);
  assert.ok(
    Math.min(...tallRoot.nodes.map((node) => node.y)) >= 28,
    "a tall root centered on a short child stays inside the canvas",
  );

  const forest = layoutLoom([
    { id: "a", parentId: "hidden-root", depth: 0 },
    { id: "b", parentId: "hidden-root", depth: 0 },
  ]);
  assert.equal(forest.edges.length, 0);
  assert.ok(forest.nodes[0].y + LOOM_NODE_HEIGHT < forest.nodes[1].y);

  assert.deepEqual(layoutLoom([]), { nodes: [], edges: [], width: 0, height: 0 });
  assert.deepEqual(loomBranchJunctions(layout).map((junction) => junction.id), ["user"]);
  const deep = layoutLoom(Array.from({ length: 12_000 }, (_, index) => ({
    id: `deep-${index}`, parentId: index === 0 ? null : `deep-${index - 1}`, depth: index,
  })));
  assert.equal(deep.nodes.length, 12_000);
  assert.equal(deep.edges.length, 11_999);
  assert.ok(deep.nodes.every((node) => node.y === 28));
  assert.ok(Number.isFinite(deep.width));
  assert.deepEqual(loomBranchJunctions(deep), []);
  const cyclic = layoutLoom([
    { id: "cycle-a", parentId: "cycle-b", depth: 0 },
    { id: "cycle-b", parentId: "cycle-a", depth: 1 },
  ]);
  assert.ok(cyclic.nodes.every((node) => Number.isFinite(node.y)));
  let edgeVisits = 0;
  const countedGraph = { ...deep, edges: deep.edges.map((edge) => ({
    ...edge,
    get parentId() { edgeVisits += 1; return edge.parentId; },
  })) };
  loomBranchJunctions(countedGraph);
  assert.equal(edgeVisits, 2 * deep.edges.length, "junction counts visit edges linearly");

  const shortScreen = fitLoomCamera({ width: 440, height: 96 }, { width: 1100, height: 3000 });
  assert.ok(shortScreen.y >= 0 && shortScreen.y + 3000 * shortScreen.zoom <= 96,
    "short phone viewports fit without a fixed padding consuming the entire map");
  const fitted = fitLoomCamera(
    { width: 1200, height: 720 },
    { width: layout.width, height: layout.height },
  );
  assert.ok(fitted.zoom >= LOOM_MIN_ZOOM && fitted.zoom <= 1);
  assert.ok(fitted.x >= 0 && fitted.y >= 0, "fit keeps a small loom inside the viewport");

  const oversized = fitLoomCamera(
    { width: 800, height: 500 },
    { width: 100_000, height: 80_000 },
  );
  assert.equal(oversized.zoom, LOOM_MIN_ZOOM);
  assert.equal(oversized.x, 48, "an oversized overview remains reachable from the left edge");
  assert.equal(oversized.y, 48, "an oversized overview remains reachable from the top edge");

  const centered = centerLoomRect(
    { width: 1200, height: 720 },
    { x: replyA.x, y: replyA.y, width: LOOM_NODE_WIDTH, height: LOOM_NODE_HEIGHT },
    1,
  );
  assert.equal(
    centered.x + (replyA.x + LOOM_NODE_WIDTH / 2) * centered.zoom,
    600,
  );
  assert.equal(
    centered.y + (replyA.y + LOOM_NODE_HEIGHT / 2) * centered.zoom,
    360,
  );

  const anchor = { x: 420, y: 240 };
  const originalCamera = { x: -180, y: 64, zoom: 0.9 };
  const oldViewport = { width: 1100, height: 440 };
  const fullViewport = { width: 1100, height: 680 };
  const resizedCamera = resizeLoomCamera(originalCamera, oldViewport, fullViewport);
  assert.equal(resizedCamera.zoom, originalCamera.zoom, "resizing preserves manual zoom");
  for (const [coordinate, dimension] of [["x", "width"], ["y", "height"]]) {
    assert.equal(
      (fullViewport[dimension] / 2 - resizedCamera[coordinate]) / resizedCamera.zoom,
      (oldViewport[dimension] / 2 - originalCamera[coordinate]) / originalCamera.zoom,
      "resizing preserves the world point at the viewport center",
    );
  }
  const phoneViewport = { width: 390, height: 320 };
  assert.deepEqual(
    resizeLoomCamera(resizeLoomCamera(resizedCamera, fullViewport, phoneViewport), phoneViewport, oldViewport),
    originalCamera,
    "desktop, phone, and fullscreen resizing round-trip without drift",
  );
  const zoomed = zoomLoomAt({ x: 100, y: 60, zoom: 1 }, LOOM_MAX_ZOOM + 1, anchor);
  assert.equal(zoomed.zoom, LOOM_MAX_ZOOM);
  assert.equal((anchor.x - zoomed.x) / zoomed.zoom, anchor.x - 100);
  assert.equal((anchor.y - zoomed.y) / zoomed.zoom, anchor.y - 60);
  console.log("loom layout passed");
} finally {
  await server.close();
}
