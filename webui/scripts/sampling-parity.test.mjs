import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {
  effectiveDrowseTopK,
  sampleDrowseTopKTopP,
} from "@drowse/web-llm";


const fixture = JSON.parse(await readFile(
  new URL("../../browser-runtime/fixtures/sampling-parity-v1.json", import.meta.url),
  "utf8",
));

for (const testCase of fixture.cases) {
  const probabilities = Float32Array.from(testCase.sortedProbabilities);
  const tokenIds = Int32Array.from(testCase.sortedTokenIds);
  const expected = testCase.expected;
  const effectiveTopK = effectiveDrowseTopK(testCase.topK, tokenIds.length);
  assert.equal(
    effectiveTopK,
    Math.min(testCase.topK || 1024, tokenIds.length),
    `${testCase.name}: effective top-k`,
  );
  for (const sample of expected.samples) {
    const result = sampleDrowseTopKTopP(
      probabilities,
      tokenIds,
      testCase.topP,
      testCase.topK,
      sample.uniform,
      testCase.requestedTokenIds,
    );
    assert.equal(result.sampledTokenId, sample.tokenId, `${testCase.name}: sampled token`);
    close(result.sampledLogprob, sample.logprob, 2e-6, `${testCase.name}: sampled logprob`);
    assert.deepEqual(
      result.topLogprobs.map((entry) => entry.token_id),
      expected.supportTokenIds,
      `${testCase.name}: support token IDs`,
    );
    result.topLogprobs.forEach((entry, index) => {
      close(
        Math.exp(entry.logprob),
        expected.supportProbabilities[index],
        2e-6,
        `${testCase.name}: support probability ${index}`,
      );
    });
    close(result.entropyNats, expected.entropyNats, 2e-6, `${testCase.name}: entropy`);
    close(result.perplexity, expected.perplexity, 2e-6, `${testCase.name}: perplexity`);
    assert.equal(result.argmax.token_id, expected.supportTokenIds[0]);
    for (const entry of result.selectedLogprobs) {
      const supportIndex = expected.supportTokenIds.indexOf(entry.token_id);
      if (supportIndex < 0) {
        assert.equal(entry.logprob, Number.NEGATIVE_INFINITY);
      } else {
        close(
          Math.exp(entry.logprob),
          expected.supportProbabilities[supportIndex],
          2e-6,
          `${testCase.name}: requested token ${entry.token_id}`,
        );
      }
    }
  }
}

console.log("Shared Python/WebLLM sampling parity checks passed");


function close(actual, expected, tolerance, label) {
  assert.ok(
    Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `${label}: ${actual} != ${expected} within ${tolerance}`,
  );
}
