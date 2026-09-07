import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { promisify } from "node:util";

const exec = promisify(execFile);
const root = await mkdtemp(join(tmpdir(), "drowse-assemble-model-"));
try {
  const converted = join(root, "converted");
  const library = join(root, "model.wasm");
  const output = join(root, "published");
  await mkdir(converted);
  const files = new Map([
    [
      "mlc-chat-config.json",
      JSON.stringify({
        tokenizer_files: ["tokenizer.json", "tokenizer_config.json"],
        model_config: {
          text_config: { hidden_size: 4, num_hidden_layers: 2 },
        },
      }),
    ],
    [
      "tensor-cache.json",
      JSON.stringify({
        records: [{ dataPath: "params_shard_0.bin", nbytes: 7 }],
      }),
    ],
    ["tokenizer.json", "tokenizer"],
    ["tokenizer_config.json", "template"],
    ["params_shard_0.bin", "weights"],
    ["LICENSE.model", "Apache License\nVersion 2.0\n"],
    [
      "MODEL-LICENSE.json",
      JSON.stringify({
        schemaVersion: 1,
        spdx: "apache-2.0",
        declaredIn: "README.md",
        licenseText: "LICENSE.model",
        licenseTextSource: "drowse-standard-text",
      }),
    ],
    ["README.source.md", "---\nlicense: apache-2.0\n---\n# Fixture\n"],
  ]);
  for (const [path, bytes] of files)
    await writeFile(join(converted, path), bytes);
  await writeFile(library, Buffer.from([0, 97, 115, 109]));
  const thinkingProfile = {
    start: "<think>",
    end: "</think>",
    startsInThinking: false,
    startTokenIds: [151667],
    endTokenIds: [151668],
  };
  await writeFile(
    join(converted, "drowse-build.json"),
    JSON.stringify({
      schemaVersion: 1,
      runtimeAbi: "drowse-web-runtime-v1",
      hookAbi: "post-block-residual-v4",
      structuredHookProfile: "standard-v3",
      thinkingProfile,
      architecture: "qwen3",
      quantization: "q4f16_1",
      contextWindowSize: 4096,
      prefillChunkSize: 2048,
      source: {
        repository: "Qwen/Qwen3-0.6B",
        revision: "a".repeat(40),
        chatTemplateSha256: "b".repeat(64),
      },
      files: [...files].map(([path, bytes]) => ({
        path,
        bytes: Buffer.byteLength(bytes),
        sha256: createHash("sha256").update(bytes).digest("hex"),
      })),
    }),
  );
  await assert.rejects(
    exec(process.execPath, [
      join(import.meta.dirname, "assemble-hosted-model.mjs"),
      converted,
      library,
      output,
    ]),
    /standard-v3 model library lacks required VM function drowse_hook_profile/,
  );
  const requiredFunctions = [
    "drowse_hook_profile",
    "drowse_prefill",
    "drowse_decode",
    "drowse_batch_prefill",
    "drowse_batch_decode",
    "drowse_capture_prefill",
    "drowse_capture_decode",
    "drowse_capture_batch_prefill",
    "drowse_capture_batch_decode",
    "drowse_rank_one_capture_prefill_v1",
    "drowse_rank_one_capture_decode_v1",
    "drowse_rank_one_capture_batch_prefill_v1",
    "drowse_rank_one_capture_batch_decode_v1",
    "drowse_structured_batch_prefill",
    "drowse_structured_batch_decode",
    "drowse_geometry_batch_prefill",
    "drowse_geometry_batch_decode",
    "drowse_curved_batch_prefill",
    "drowse_curved_batch_decode",
    "drowse_jlens_probabilities",
    "drowse_jlens_readout_accumulate",
    "drowse_jlens_readout_topk",
    "drowse_jlens_directions",
    "drowse_sae_readout_accumulate",
    "drowse_sae_jump_relu_readout_accumulate",
  ];
  const markers = requiredFunctions.flatMap((name) => {
    const encoded = Buffer.from(name);
    const length = Buffer.alloc(8);
    length.writeBigUInt64LE(BigInt(encoded.byteLength));
    return [length, encoded];
  });
  await writeFile(
    library,
    Buffer.concat([Buffer.from([0, 97, 115, 109]), ...markers]),
  );
  await exec(process.execPath, [
    join(import.meta.dirname, "assemble-hosted-model.mjs"),
    converted,
    library,
    output,
  ]);
  const manifest = JSON.parse(
    await readFile(join(output, "hosted-artifacts.json"), "utf8"),
  );
  assert.equal(manifest.structuredHookProfile, "standard-v3");
  assert.equal(manifest.hiddenSize, 4);
  assert.deepEqual(manifest.layerMap, [0, 1]);
  assert.deepEqual(manifest.thinkingProfile, thinkingProfile);
  assert.equal(
    manifest.files.find((file) => file.role === "model_library").path,
    "model.wasm",
  );
  for (const path of [
    "LICENSE.model",
    "MODEL-LICENSE.json",
    "README.source.md",
    "drowse-build.json",
  ]) {
    assert.equal(
      manifest.files.filter((file) => file.path === path && file.role === "configuration").length,
      1,
    );
  }
} finally {
  await rm(root, { recursive: true, force: true });
}

console.log("hosted model assembler checks passed");
