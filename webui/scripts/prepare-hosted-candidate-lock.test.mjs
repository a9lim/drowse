import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  deriveCandidateRuntimeLock,
  parseArguments,
} from "./prepare-hosted-candidate-lock.mjs";
import { validateRuntimeLock } from "./runtime-lock-document.mjs";
import { lockedModelExecutionProfiles } from "./hosted-model-profiles.mjs";

const repositoryRoot = resolve(import.meta.dirname, "../..");
const runtimeLock = JSON.parse(
  await readFile(resolve(repositoryRoot, "browser-runtime/runtime-lock.json"), "utf8"),
);
const modelId = "qwen3-1.7b";
const original = runtimeLock.models.find((model) => model.id === modelId);
const candidateDigest = "b".repeat(64);
const candidate = deriveCandidateRuntimeLock(runtimeLock, modelId, {
  files: [{ path: "model.wasm", role: "model_library", sha256: candidateDigest }],
});
assert.equal(candidate.models.find((model) => model.id === modelId).librarySha256, candidateDigest);
assert.equal(original.librarySha256, runtimeLock.models.find((model) => model.id === modelId).librarySha256);
const normalized = structuredClone(candidate);
normalized.models.find((model) => model.id === modelId).librarySha256 = original.librarySha256;
assert.deepEqual(normalized, runtimeLock);
for (const architecture of ["gpt2", "gpt_neox", "qwen3_5", "gemma3_text"]) {
  const base = structuredClone(runtimeLock);
  const pin = base.models[0];
  Object.assign(pin, {
    architecture, modelType: "base", thinkingProfile: null,
    quantization: architecture === "qwen3_5" ? "q4f16_1" : "q0f32",
    contextProfiles: architecture === "gpt2" ? [1024] : [2048],
  });
  assert.doesNotThrow(() => validateRuntimeLock(base));
  assert.equal(lockedModelExecutionProfiles(pin, pin).requiredFeatures.length,
    pin.quantization === "q0f32" ? 0 : 1);
  if (architecture !== "gemma3_text") {
    delete pin.modelType;
    assert.throws(() => validateRuntimeLock(base), /invalid/);
    pin.modelType = "base";
  }
  if (architecture === "gpt2") {
    pin.contextProfiles = [2048];
    assert.throws(() => validateRuntimeLock(base), /context profiles/);
  }
}

assert.throws(
  () => deriveCandidateRuntimeLock(runtimeLock, "missing", { files: [] }),
  /unknown runtime-lock model/,
);
assert.throws(
  () => deriveCandidateRuntimeLock(runtimeLock, modelId, {
    files: [
      { path: "model.wasm", role: "model_library", sha256: candidateDigest },
      { path: "other.wasm", role: "model_library", sha256: candidateDigest },
    ],
  }),
  /no unique model\.wasm library digest/,
);
assert.throws(
  () => parseArguments([modelId, "model"]),
  /MODEL_ID MODEL_DIR OUTPUT\.json/,
);
assert.deepEqual(
  parseArguments([modelId, "model", "candidate.json"]),
  {
    modelId,
    modelDirectory: resolve("model"),
    outputPath: resolve("candidate.json"),
  },
);

console.log("Hosted candidate runtime-lock checks passed");
