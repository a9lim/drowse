import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { createServer } from "vite";
import { digestCanonical } from "./check-runtime-lock.mjs";

const raw = await readFile(new URL("../public-hosted/recordings/404-gemma3-4b.json", import.meta.url), "utf8");
const recording = JSON.parse(raw);
const dataset = JSON.parse(await readFile(new URL("../src/hosted/data/not-found-messages.json", import.meta.url), "utf8"));
assert.equal(dataset.recordingSha256, createHash("sha256").update(raw).digest("hex"));
assert.equal(recording.modelId, "gemma3-4b-instruct");
assert.equal(recording.runs.length, 50);
assert.equal(new Set(recording.runs.map(run => run.seed)).size, 50);
assert.equal(recording.runtimeIdentitySha256, digestCanonical(recording.runtimeIdentity));
assert.equal(dataset.totalRuns, 50);
assert.equal(dataset.messages.length, dataset.uniqueMessages);
assert.equal(dataset.uniqueMessages, new Set(recording.runs.map(run => run.text.trim())).size);
assert.deepEqual(dataset.sampling, recording.sampling);
assert.equal(dataset.prompt, recording.prompt);
const covered = [];
for (const message of dataset.messages) {
  assert.equal(message.id, createHash("sha256").update(message.text).digest("hex").slice(0, 16));
  const first = recording.runs[message.runIndices[0]];
  assert.equal(message.text, first.text.trim());
  assert.deepEqual(message.tokens, first.tokens.map(({ text, thinking, tokenId, rawIndex, logprob, topAlts }) => ({ text, thinking, tokenId, rawIndex, logprob, topAlts })));
  for (const index of message.runIndices) {
    covered.push(index);
    assert.equal(recording.runs[index].text.trim(), message.text);
  }
}
assert.deepEqual(covered.sort((a, b) => a - b), Array.from({ length: 50 }, (_, i) => i));
for (const run of recording.runs) {
  assert.equal(run.text, run.result.text);
  assert.equal(run.text, run.tokens.map(token => token.text).join(""));
  assert.equal(run.result.terminalReason, "eos");
  assert.ok(run.tokens.length < recording.sampling.max_tokens);
  for (const token of run.tokens) {
    assert.ok(Number.isInteger(token.tokenId));
    assert.ok(Number.isFinite(token.logprob) && token.logprob <= 0);
    assert.ok(token.topAlts.length > 0 && token.topAlts.length <= 8);
    assert.equal(new Set(token.topAlts.map(alt => alt.id)).size, token.topAlts.length);
    for (const alt of token.topAlts) assert.ok(Number.isFinite(alt.logprob) && alt.logprob <= 0);
    const selected = token.topAlts.find(alt => alt.id === token.tokenId);
    if (selected) assert.equal(selected.logprob, token.logprob);
    const mass = token.topAlts.reduce((sum, alt) => sum + Math.exp(alt.logprob), 0) + (selected ? 0 : Math.exp(token.logprob));
    assert.ok(mass > 0 && mass <= 1 + 1e-6);
  }
}
for (const [role, key] of [["model_library", "modelLibrarySha256"], ["converted_manifest", "convertedManifestSha256"], ["tokenizer", "tokenizerSha256"], ["chat_template", "chatTemplateSha256"]]) {
  assert.equal(recording.artifacts.model.find(file => file.role === role).sha256, recording.runtimeIdentity[key]);
}
const vite = await createServer({ configFile: false, logLevel: "silent", server: { middlewareMode: true, watch: null } });
try {
  const { chooseMessage } = await vite.ssrLoadModule("/src/hosted/ui/notFoundMessages.ts");
  for (const previous of [null, ...dataset.messages.map(message => message.id)]) {
    const count = dataset.messages.length - (previous ? 1 : 0);
    const choices = Array.from({ length: count }, (_, index) => chooseMessage(previous, (index + 0.5) / count));
    assert.equal(new Set(choices).size, count, "Every eligible unique message has the same sampling interval");
    if (previous) assert.ok(choices.every(index => dataset.messages[index].id !== previous));
  }
} finally { await vite.close(); }
console.log(`404 recording: 50 complete runs, ${dataset.uniqueMessages} unique replies, exact token provenance, probability mass, and random selection passed`);
