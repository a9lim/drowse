import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rename, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  parseArguments,
  validatePublishDirectory,
} from "./publish-hosted-model.mjs";
import { structuredHookProfile, thinkingProfile } from "./hosted-model-profiles.mjs";
import { expectedBuildToolchain } from "./hosted-model-closure.mjs";

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
    metadata: { ParamBytes: 8 },
    records: [{
      dataPath: "params_shard_0.bin", nbytes: Buffer.byteLength(weight),
      md5sum: createHash("md5").update(weight).digest("hex"),
      records: [
        { name: "scale", dtype: "float32", shape: [1], format: "f32-to-bf16", byteOffset: 0, nbytes: 2 },
        { name: "packed", dtype: "uint32", shape: [1], format: "raw", byteOffset: 2, nbytes: 4 },
      ],
    }],
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
  const forkBytes = await readFile(new URL("../../browser-runtime/forks/manifest.json", import.meta.url));
  const buildToolchain = expectedBuildToolchain(JSON.parse(forkBytes), "qwen3");
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
    toolchain: { forkManifestSha256: createHash("sha256").update(forkBytes).digest("hex") },
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
  for (const stale of [
    { records: [{ dataPath: "params_shard_0.bin", nbytes: Buffer.byteLength(weight), md5sum: "0".repeat(32) }] },
    { ...JSON.parse(tensorCache), metadata: { ParamBytes: 1 } },
  ]) {
    await writeFile(join(directory, "tensor-cache.json"), JSON.stringify(stale));
    await assert.rejects(() => validatePublishDirectory(directory, model, runtime), /shard checksum differs|parameter byte count is stale/);
  }
  await writeFile(join(directory, "tensor-cache.json"), tensorCache);
  const buildEntry = files.find((file) => file.path === "drowse-build.json");
  const originalBuildEntry = { ...buildEntry };
  const staleBuild = JSON.parse(build);
  staleBuild.toolchain.mlcOverlayFiles["python/mlc_llm/model/gemma3/gemma3_model.py"] = "0".repeat(64);
  const staleBytes = JSON.stringify(staleBuild);
  await writeFile(join(directory, "drowse-build.json"), staleBytes);
  Object.assign(buildEntry, { bytes: Buffer.byteLength(staleBytes),
    sha256: createHash("sha256").update(staleBytes).digest("hex") });
  await writeFile(join(directory, "hosted-artifacts.json"), JSON.stringify(manifest));
  await assert.rejects(() => validatePublishDirectory(directory, model, runtime), /build toolchain is stale/);
  assert.equal((await validatePublishDirectory(directory, model, runtime,
    { allowStaleToolchain: true })).files.length, 10);
  await writeFile(join(directory, "drowse-build.json"), build);
  Object.assign(buildEntry, originalBuildEntry);
  await writeFile(join(directory, "hosted-artifacts.json"), JSON.stringify(manifest));
  for (const brand of ["saklas", "polythetic"]) {
    manifest.runtimeAbi = `${brand}-web-runtime-v1`;
    const legacyBuild = JSON.stringify({ ...JSON.parse(build), runtimeAbi: manifest.runtimeAbi });
    await rename(join(directory, "drowse-build.json"), join(directory, `${brand}-build.json`));
    await writeFile(join(directory, `${brand}-build.json`), legacyBuild);
    Object.assign(buildEntry, {
      path: `${brand}-build.json`, bytes: Buffer.byteLength(legacyBuild),
      sha256: createHash("sha256").update(legacyBuild).digest("hex"),
    });
    await writeFile(join(directory, "hosted-artifacts.json"), JSON.stringify(manifest));
    await assert.rejects(() => validatePublishDirectory(directory, model, runtime), /ABI differs/);
    assert.equal((await validatePublishDirectory(directory, model, runtime, { allowLegacyBranding: true })).files.length, 10);
    await assert.rejects(
      () => validatePublishDirectory(directory, model, { ...runtime, hookAbi: "wrong" }, { allowLegacyBranding: true }),
      /ABI differs/,
    );
    await writeFile(join(directory, `${brand}-build.json`), `${legacyBuild} `);
    await assert.rejects(
      () => validatePublishDirectory(directory, model, runtime, { allowLegacyBranding: true }),
      /size differs|digest differs/,
    );
    await rename(join(directory, `${brand}-build.json`), join(directory, "drowse-build.json"));
    await writeFile(join(directory, "drowse-build.json"), build);
    Object.assign(buildEntry, originalBuildEntry);
  }
  manifest.runtimeAbi = "unknown-web-runtime-v1";
  await writeFile(join(directory, "hosted-artifacts.json"), JSON.stringify(manifest));
  await assert.rejects(
    () => validatePublishDirectory(directory, model, runtime, { allowLegacyBranding: true }), /ABI differs/,
  );
  manifest.runtimeAbi = runtime.runtimeAbi;
  await writeFile(join(directory, "hosted-artifacts.json"), JSON.stringify(manifest));
  await writeFile(join(directory, "tokenizer.json"), "changed");
  await assert.rejects(
    () => validatePublishDirectory(directory, model, runtime),
    /size differs|digest differs/,
  );
} finally {
  await rm(directory, { recursive: true, force: true });
}

console.log("hosted model publisher checks passed");
