import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const input = process.argv[2];
if (!input) throw new Error("Usage: node scripts/prepare-not-found.mjs <recording.json>");
const root = resolve(import.meta.dirname, "..");
const raw = await readFile(input, "utf8");
const recording = JSON.parse(raw);
assert.equal(recording.modelId, "gemma3-4b-instruct");
assert.equal(recording.runs.length, 50);
assert.equal(new Set(recording.runs.map(run => run.seed)).size, 50);
const messages = new Map();
for (const run of recording.runs) {
  assert.equal(run.result.terminalReason, "eos", `Run ${run.index} did not finish naturally`);
  assert.equal(run.text, run.tokens.map(token => token.text).join(""));
  const text = run.text.trim();
  assert.match(text, /^Error 404: [^\n]+[.!?]$/);
  assert.ok(text.length <= 180, `Run ${run.index} is too long`);
  for (const token of run.tokens) {
    assert.ok(Number.isInteger(token.tokenId));
    assert.ok(Number.isFinite(token.logprob) && token.logprob <= 0);
    assert.ok(token.topAlts.length > 0);
    for (const alternative of token.topAlts) {
      assert.ok(Number.isFinite(alternative.logprob) && alternative.logprob <= 0);
    }
    const chosen = token.topAlts.find(alt => alt.id === token.tokenId);
    if (chosen) assert.ok(Math.abs(chosen.logprob - token.logprob) < 1e-9);
    assert.ok(token.topAlts.reduce((sum, alt) => sum + Math.exp(alt.logprob), 0) <= 1 + 1e-6);
  }
  const previous = messages.get(text);
  if (previous) previous.runIndices.push(run.index);
  else messages.set(text, {
    id: createHash("sha256").update(text).digest("hex").slice(0, 16),
    text, runIndices: [run.index],
    tokens: run.tokens.map(({ text, thinking, tokenId, rawIndex, logprob, topAlts }) => ({ text, thinking, tokenId, rawIndex, logprob, topAlts })),
  });
}
const dataset = {
  schemaVersion: 1, modelId: recording.modelId, recordedAt: recording.recordedAt,
  prompt: recording.prompt, sampling: recording.sampling,
  totalRuns: recording.runs.length, uniqueMessages: messages.size,
  recordingSha256: createHash("sha256").update(raw).digest("hex"),
  messages: [...messages.values()],
};
await mkdir(resolve(root, "public-hosted/recordings"), { recursive: true });
await writeFile(resolve(root, "public-hosted/recordings/404-gemma3-4b.json"), raw);
await writeFile(resolve(root, "src/hosted/data/not-found-messages.json"), JSON.stringify(dataset, null, 2) + "\n");
console.log(`${dataset.totalRuns} verified runs, ${dataset.uniqueMessages} unique messages`);
