import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { digestCanonical } from "./check-runtime-lock.mjs";

const recording = JSON.parse(await readFile(new URL("../src/hosted/data/neighbor-example.json", import.meta.url), "utf8"));
assert.equal(recording.schemaVersion, 1);
assert.equal(recording.modelId, "gemma3-4b-instruct");
assert.equal(digestCanonical(recording.runtimeIdentity), recording.runtimeIdentitySha256);
assert.equal(recording.sampling.seed, 42);
assert.equal(recording.sampling.top_k, 0);
assert.equal(recording.sampling.top_p, 1);
assert.deepEqual(recording.runs.map(run => run.alpha), [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15]);
for (const run of recording.runs) {
  assert.equal(run.expression, run.alpha === 0 ? null : `${run.alpha} default/welcoming.detached`);
  assert.equal(run.result.finishReason, "stop");
  assert.equal(run.result.terminalReason, "eos");
  assert.equal(run.text, run.result.text);
  assert.equal(run.text, run.tokens.map(token => token.text).join(""));
  assert.ok(run.tokens.length > 15 && run.tokens.length < recording.sampling.max_tokens);
  for (const token of run.tokens) {
    assert.ok(Number.isSafeInteger(token.tokenId));
    assert.ok(Number.isFinite(token.logprob) && token.logprob <= 0);
    assert.ok(token.topAlts.length > 0 && token.topAlts.length <= 5);
    assert.equal(new Set(token.topAlts.map(row => row.id)).size, token.topAlts.length);
    for (const row of token.topAlts) assert.ok(Number.isFinite(row.logprob) && row.logprob <= 0);
    const chosen = token.topAlts.find(row => row.id === token.tokenId);
    if (chosen) assert.equal(chosen.logprob, token.logprob);
    const mass = token.topAlts.reduce((sum, row) => sum + Math.exp(row.logprob), 0) + (chosen ? 0 : Math.exp(token.logprob));
    assert.ok(mass <= 1 + 1e-6 && mass > 0);
  }
}
const files = recording.artifacts.model;
for (const [role, expected] of [
  ["model_library", recording.runtimeIdentity.modelLibrarySha256],
  ["converted_manifest", recording.runtimeIdentity.convertedManifestSha256],
  ["tokenizer", recording.runtimeIdentity.tokenizerSha256],
]) assert.equal(files.find(file => file.role === role).sha256, expected);
assert.ok(recording.artifacts.core.some(file => file.path.endsWith("default-welcoming.detached.drowse")));
console.log("Recorded preset: seven complete real runs, exact token text, model identities, and unscaled probability mass passed");
