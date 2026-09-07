import assert from "node:assert/strict";
import {
  activeVocabSize,
  adoptedBuildManifest,
  captureSpecialTokenIds,
  modelCardLicense,
  parseArguments,
  productionMlcConfig,
} from "./adopt-hosted-mlc-model.mjs";

const model = {
  id: "qwen3-4b",
  architecture: "qwen3",
  structuredHookProfile: "standard-v3",
  thinkingProfile: {
    start: "<think>",
    end: "</think>",
    startsInThinking: false,
    startTokenIds: [151667],
    endTokenIds: [151668],
  },
  sourceRepository: "Qwen/Qwen3-4B",
  sourceRevision: "1".repeat(40),
  hiddenSize: 2560,
  layerMap: Array.from({ length: 36 }, (_, index) => index),
  quantization: "q4f16_1",
  contextProfiles: [2048, 4096],
};

assert.deepEqual(parseArguments([
  "qwen3-4b",
  "mlc-ai/Qwen3-4B-q4f16_1-MLC",
  "2".repeat(40),
  "/tmp/mlc",
  "/tmp/source",
  "/tmp/output",
]), {
  modelId: "qwen3-4b",
  upstreamRepository: "mlc-ai/Qwen3-4B-q4f16_1-MLC",
  upstreamRevision: "2".repeat(40),
  mlcDirectory: "/tmp/mlc",
  sourceDirectory: "/tmp/source",
  outputDirectory: "/tmp/output",
});

const safeConfig = productionMlcConfig({
  model_type: "qwen3",
  quantization: "q4f16_1",
  context_window_size: 40_960,
  prefill_chunk_size: 2048,
  model_config: {
    hidden_size: 2560,
    num_hidden_layers: 36,
    context_window_size: 40_960,
    prefill_chunk_size: 2048,
    max_batch_size: 128,
  },
}, model);
assert.equal(safeConfig.context_window_size, 4096);
assert.equal(safeConfig.prefill_chunk_size, 2048);
assert.equal(safeConfig.model_config.context_window_size, 4096);
assert.equal(safeConfig.model_config.max_batch_size, 1);
assert.throws(
  () => productionMlcConfig({ ...safeConfig, quantization: "q4f32_1" }, model),
  /differs from the runtime lock/u,
);

const gemmaModel = {
  ...model,
  id: "gemma3-4b-instruct",
  architecture: "gemma3_text",
  thinkingProfile: null,
  quantization: "q4f16_1",
  layerMap: Array.from({ length: 34 }, (_, index) => index),
};
const gemmaConfig = productionMlcConfig({
  model_type: "gemma3",
  quantization: "q4f16_1",
  context_window_size: 8192,
  prefill_chunk_size: 8192,
  model_config: {
    text_config: { hidden_size: 2560, num_hidden_layers: 34 },
    is_text_model: false,
    context_window_size: 8192,
    prefill_chunk_size: 8192,
    max_batch_size: 128,
  },
}, gemmaModel);
assert.equal(gemmaConfig.model_type, "gemma3_text");
assert.equal(gemmaConfig.model_config.is_text_model, true);
assert.equal(gemmaConfig.model_config.text_config.context_window_size, 4096);
assert.equal(activeVocabSize({
  model: { vocab: [["a", 0], ["b", 0]] },
  added_tokens: [{ id: 262144 }],
}, 262208), 262145);
assert.equal(activeVocabSize({
  model: { vocab: { a: 0, b: 1 } },
  added_tokens: [{ id: 7 }],
}, 6), 6);
assert.throws(
  () => activeVocabSize({}, 262208),
  /cannot determine the active vocabulary size/u,
);
assert.equal(modelCardLicense("---\nlicense: gemma\n---\n"), "gemma");
assert.equal(modelCardLicense("no card"), null);

assert.deepEqual(captureSpecialTokenIds(
  { added_tokens: [{ id: 151667, special: true }, { id: 42, special: false }] },
  {
    added_tokens_decoder: {
      151668: { special: true },
      151669: { special: true },
    },
    eos_token_id: 151645,
  },
), [42, 151645, 151667, 151668, 151669]);

const manifest = adoptedBuildManifest({
  model,
  sourceFiles: [{ path: "README.md", bytes: 1, sha256: "a".repeat(64) }],
  outputFiles: [{ path: "tensor-cache.json", bytes: 1, sha256: "b".repeat(64) }],
  sourceTokenizerConfig: { chat_template: "template" },
  toolchain: { pinned: true },
  upstreamRepository: "mlc-ai/Qwen3-4B-q4f16_1-MLC",
  upstreamRevision: "2".repeat(40),
});
assert.deepEqual(Object.keys(manifest).sort(), [
  "architecture",
  "contextWindowSize",
  "files",
  "hookAbi",
  "prefillChunkSize",
  "quantization",
  "runtimeAbi",
  "schemaVersion",
  "source",
  "structuredHookProfile",
  "thinkingProfile",
  "toolchain",
].sort());
assert.equal(manifest.contextWindowSize, 4096);
assert.equal(manifest.prefillChunkSize, 2048);

console.log("13 hosted MLC adoption tests passed");
