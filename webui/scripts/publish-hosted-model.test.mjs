import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  parseArguments,
  validatePublishDirectory,
} from "./publish-hosted-model.mjs";
import { structuredHookProfile, thinkingProfile } from "./hosted-model-profiles.mjs";

assert.equal(structuredHookProfile("standard-v1"), "standard-v1");
assert.equal(structuredHookProfile("standard-v2"), "standard-v2");
assert.equal(structuredHookProfile("standard-v3"), "standard-v3");
assert.throws(() => structuredHookProfile("adapter-derived"), /not allow-listed/);
assert.equal(thinkingProfile(null), null);
const verifiedThinkingProfile = {
  start: "<think>",
  end: "</think>",
  startsInThinking: false,
  startTokenIds: [151667],
  endTokenIds: [151668],
};
assert.deepEqual(thinkingProfile(verifiedThinkingProfile), verifiedThinkingProfile);
assert.throws(
  () => thinkingProfile({ ...verifiedThinkingProfile, startTokenIds: [] }),
  /1-32 non-negative token IDs/,
);

assert.deepEqual(parseArguments(["qwen3-0.6b", "./candidate"]), {
  modelId: "qwen3-0.6b",
  directory: join(process.cwd(), "candidate"),
  upload: false,
});
assert.equal(parseArguments(["qwen3-0.6b", "./candidate", "--upload"]).upload, true);
assert.throws(() => parseArguments(["qwen3-0.6b"]), /usage/);
assert.throws(() => parseArguments(["qwen3-0.6b", "./candidate", "--force"]), /usage/);

const directory = await mkdtemp(join(tmpdir(), "drowse-publish-test-"));
try {
  const contextWindowSize = 4096;
  const prefillChunkSize = 2048;
  const weight = "weight";
  const tensorCache = JSON.stringify({
    records: [{ dataPath: "params_shard_0.bin", nbytes: Buffer.byteLength(weight) }],
  });
  const config = JSON.stringify({
    model_type: "qwen3",
    quantization: "q4f16_1",
    context_window_size: contextWindowSize,
    prefill_chunk_size: prefillChunkSize,
    model_config: {
      context_window_size: contextWindowSize,
      prefill_chunk_size: prefillChunkSize,
      text_config: {
        hidden_size: 4,
        num_hidden_layers: 2,
      },
    },
  });
  const licenseText = await readFile(
    new URL("../../browser-runtime/forks/licenses/Apache-2.0.txt", import.meta.url),
    "utf8",
  );
  const buildToolchain = { fixture: true };
  const sourceReadme = "---\nlicense: apache-2.0\n---\n# Fixture source\n";
  const licenseMetadata = JSON.stringify({
    schemaVersion: 1,
    spdx: "apache-2.0",
    declaredIn: "README.md",
    licenseText: "LICENSE.model",
    licenseTextSource: "drowse-standard-text",
  });
  const convertedContents = [
    ["tensor-cache.json", "converted_manifest", tensorCache],
    ["tokenizer.json", "tokenizer", "tokenizer"],
    ["tokenizer_config.json", "chat_template", "template"],
    ["mlc-chat-config.json", "configuration", config],
    ["params_shard_0.bin", "weight", weight],
    ["LICENSE.model", "configuration", licenseText],
    ["MODEL-LICENSE.json", "configuration", licenseMetadata],
    ["README.source.md", "configuration", sourceReadme],
  ];
  const buildFileRecords = convertedContents.map(([path, _role, value]) => ({
    path,
    bytes: Buffer.byteLength(value),
    sha256: createHash("sha256").update(value).digest("hex"),
  }));
  const build = JSON.stringify({
    schemaVersion: 1,
    runtimeAbi: "drowse-web-runtime-v1",
    hookAbi: "post-block-residual-v4",
    structuredHookProfile: "standard-v2",
    thinkingProfile: verifiedThinkingProfile,
    architecture: "qwen3",
    quantization: "q4f16_1",
    contextWindowSize,
    prefillChunkSize,
    source: {
      repository: "fixture/source",
      revision: "a".repeat(40),
      chatTemplateSha256: "b".repeat(64),
      files: [{
        path: "README.md",
        bytes: Buffer.byteLength(sourceReadme),
        sha256: createHash("sha256").update(sourceReadme).digest("hex"),
      }],
    },
    toolchain: buildToolchain,
    files: buildFileRecords,
  });
  const contents = [
    ...convertedContents,
    ["drowse-build.json", "configuration", build],
    ["model.wasm", "model_library", "\0asm"],
  ];
  const files = [];
  for (const [path, role, value] of contents) {
    await writeFile(join(directory, path), value);
    files.push({
      path,
      role,
      bytes: Buffer.byteLength(value),
      sha256: createHash("sha256").update(value).digest("hex"),
    });
  }
  const manifest = {
    schemaVersion: 1,
    runtimeAbi: "drowse-web-runtime-v1",
    hookAbi: "post-block-residual-v4",
    structuredHookProfile: "standard-v2",
    thinkingProfile: verifiedThinkingProfile,
    architecture: "qwen3",
    quantization: "q4f16_1",
    contextWindowSize,
    prefillChunkSize,
    hiddenSize: 4,
    layerMap: [0, 1],
    source: { repository: "fixture/source", revision: "a".repeat(40) },
    files,
  };
  await writeFile(join(directory, "hosted-artifacts.json"), JSON.stringify(manifest));
  const model = {
    architecture: manifest.architecture,
    quantization: manifest.quantization,
    hiddenSize: manifest.hiddenSize,
    layerMap: manifest.layerMap,
    sourceRepository: manifest.source.repository,
    sourceRevision: manifest.source.revision,
    structuredHookProfile: manifest.structuredHookProfile,
    thinkingProfile: manifest.thinkingProfile,
    manifestSha256: files.find((file) => file.role === "converted_manifest").sha256,
    librarySha256: files.find((file) => file.role === "model_library").sha256,
    tokenizerSha256: files.find((file) => file.role === "tokenizer").sha256,
    chatTemplateSha256: files.find((file) => file.role === "chat_template").sha256,
    contextProfiles: [2048, 4096],
  };
  const runtime = {
    runtimeAbi: manifest.runtimeAbi,
    hookAbi: manifest.hookAbi,
  };
  assert.equal((await validatePublishDirectory(directory, model, runtime)).files.length, 10);
  manifest.thinkingProfile = {
    endTokenIds: [...verifiedThinkingProfile.endTokenIds],
    start: verifiedThinkingProfile.start,
    startsInThinking: verifiedThinkingProfile.startsInThinking,
    end: verifiedThinkingProfile.end,
    startTokenIds: [...verifiedThinkingProfile.startTokenIds],
  };
  await writeFile(join(directory, "hosted-artifacts.json"), JSON.stringify(manifest));
  assert.equal((await validatePublishDirectory(directory, model, runtime)).files.length, 10);
  manifest.thinkingProfile = verifiedThinkingProfile;
  await writeFile(join(directory, "hosted-artifacts.json"), JSON.stringify(manifest));
  await assert.rejects(
    () => validatePublishDirectory(directory, model, { ...runtime, hookAbi: "wrong" }),
    /ABI differs/,
  );
  await assert.rejects(
    () => validatePublishDirectory(directory, { ...model, contextProfiles: [2048] }, runtime),
    /context configuration differs/,
  );
  await writeFile(join(directory, "tensor-cache.json"), JSON.stringify({ records: [] }));
  await assert.rejects(
    () => validatePublishDirectory(directory, model, runtime),
    /no weight records/,
  );
  await writeFile(join(directory, "tensor-cache.json"), tensorCache);
  await writeFile(join(directory, "tokenizer.json"), "changed");
  await assert.rejects(
    () => validatePublishDirectory(directory, model, runtime),
    /size differs|digest differs/,
  );
} finally {
  await rm(directory, { recursive: true, force: true });
}

console.log("hosted model publisher checks passed");
