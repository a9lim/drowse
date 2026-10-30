// Streaming cost of the loom map projection: a tree of finished replies plus
// one pending reply that grows by one token per animation frame.  Reports the
// mean per-frame cost of the projection the LoomSidebar recomputes for each
// streamed frame (live overlay -> segmentation -> segment sizes).
//
//   node ./scripts/bench-loom-stream.mjs [finishedNodes] [tokensPerNode] [frames]
import { createServer } from "vite";

const [nodes = 40, tokensPerNode = 600, frames = 400] = process.argv.slice(2).map(Number);
const server = await createServer({ configFile: false, appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
try {
  const model = await server.ssrLoadModule("/src/panels/loom/loomMapModel.ts");
  const { segmentLoom } = await server.ssrLoadModule("/src/panels/loom/loomSegments.ts");
  const words = ["The", " loom", " keeps", " every", " branch", ".", " It", " grows", "!"];
  // Replies share a three-token opening and then diverge, like sampled siblings.
  const reply = (id, parent, count, finished = true, seed = 0) => {
    const tokens = Array.from({ length: count }, (_, i) => {
      const word = i < 3 ? i : Math.imul(i + 1, 2654435761 ^ Math.imul(seed + 1, 40503)) >>> 0 >> 7 & 7;
      return { text: words[word], raw_index: i, token_id: 100 + word, logprob: -0.3 };
    });
    return {
      id, parent_id: parent, role: "assistant", role_label: null, recipe: {}, text: tokens.map(t => t.text).join(""),
      raw_token_ids: tokens.map(t => t.token_id), tokens, finish_reason: finished ? "stop" : null,
    };
  };
  const prompt = { id: "prompt", parent_id: null, role: "user", role_label: null, recipe: null, text: "Hi", tokens: null, raw_token_ids: null, finish_reason: null };
  const tree = [prompt];
  for (let i = 0; i < nodes; i++) tree.push(reply(`r${i}`, "prompt", tokensPerNode, true, i + 1));
  const pendingSaved = reply("pending", "prompt", 0, false);
  tree.push(pendingSaved);
  const active = new Set(["prompt", "pending"]);
  const turn = { nodeId: "pending", text: "", tokens: [] };
  const live = model.createLoomLiveNode();
  let sink = 0;
  let started = performance.now();
  const warmup = 10;
  for (let frame = 0; frame < warmup + frames; frame++) {
    if (frame === warmup) started = performance.now();
    const word = frame < 3 ? frame : (frame * 7) % words.length;
    turn.text += words[word];
    turn.tokens.push({ text: words[word], rawIndex: frame, tokenId: 100 + word, logprob: -0.2 });
    const liveNode = live(pendingSaved, turn);
    const projected = tree.map(node => node.id === "pending" ? liveNode : node);
    const segments = segmentLoom(projected, active);
    for (const segment of segments) {
      const width = model.segmentWidth(segment);
      sink += width + model.segmentHeightEstimate(segment, width, false);
    }
  }
  const elapsed = performance.now() - started;
  console.log(JSON.stringify({ nodes, tokensPerNode, frames, totalTokens: nodes * tokensPerNode, msPerFrame: +(elapsed / frames).toFixed(3), sink }));
} finally { await server.close(); }
