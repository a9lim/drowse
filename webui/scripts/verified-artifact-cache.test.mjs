import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const { prepareVerifiedWebLlmArtifacts } = await server.ssrLoadModule(
    "/src/hosted/runtime/verifiedArtifactCache.ts",
  );
  const { DrowseWebLlmRuntime } = await server.ssrLoadModule(
    "/src/hosted/runtime/webLlmEngine.ts",
  );
  const revision = "a".repeat(40);
  const base = `https://huggingface.co/a9lim/drowse-web-fixture/resolve/${revision}/`;
  const config = JSON.stringify({
    tokenizer_files: [
      "tokenizer.json",
      "vocab.json",
      "merges.txt",
      "special_tokens_map.json",
      "tokenizer_config.json",
    ],
    bos_token_id: 1,
    eos_token_id: 2,
    drowse_capture_special_token_ids: [1, 2, 9],
    conv_template: {
      roles: {
        user: "<start_of_turn>user",
        assistant: "<start_of_turn>model",
      },
    },
  });
  const tensorCache = JSON.stringify({
    records: [{ dataPath: "params_shard_0.bin", nbytes: 3, records: [] }],
  });
  const artifact = (path, role, content, url = `${base}${path}`) => ({
    manifest: {
      path,
      role,
      url,
      revision,
      bytes: new Blob([content]).size,
      sha256: "b".repeat(64),
    },
    file: new File([content], path),
  });
  const fixture = () => [
    artifact("mlc-chat-config.json", "configuration", config),
    artifact("tokenizer.json", "tokenizer", JSON.stringify({
      added_tokens: [{ id: 9, content: "<turn>", special: false }],
    })),
    artifact("tensor-cache.json", "converted_manifest", tensorCache),
    artifact("params_shard_0.bin", "weight", new Uint8Array([1, 2, 3])),
    artifact("tokenizer_config.json", "chat_template", JSON.stringify({
      chat_template: "{{ messages }}",
    })),
    artifact("model-lib.wasm", "model_library", new Uint8Array([0, 97, 115, 109])),
    artifact("vocab.json", "configuration", "{}"),
    artifact("merges.txt", "configuration", ""),
    artifact("special_tokens_map.json", "configuration", "{}"),
  ];

  const prepared = await prepareVerifiedWebLlmArtifacts(fixture());
  assert.equal(prepared.modelUrl, base);
  assert.equal(prepared.modelLibraryUrl, `${base}model-lib.wasm`);
  assert.deepEqual(prepared.captureSpecialTokenIds, [1, 2, 9]);
  assert.equal(prepared.isBaseModel, false);
  assert.equal(prepared.defaultUserRole, "user");
  assert.equal(prepared.defaultAssistantRole, "model");
  const baseModelFixture = fixture().map((entry) =>
    entry.manifest.path === "tokenizer_config.json"
      ? artifact("tokenizer_config.json", "chat_template", "{}")
      : entry
  );
  const preparedBaseModel = await prepareVerifiedWebLlmArtifacts(baseModelFixture);
  assert.equal(preparedBaseModel.isBaseModel, true);
  assert.equal(preparedBaseModel.defaultUserRole, null);
  assert.equal(preparedBaseModel.defaultAssistantRole, null);
  const baseWithSourceChatTemplate = fixture().map(entry => {
    if (entry.manifest.path !== "mlc-chat-config.json") return entry;
    const values = JSON.parse(config);
    values.vocab_size = 16;
    values.drowse_completion_prefix_token_ids = [];
    values.conv_template = {
      name: "drowse-base", system_template: "", system_message: "",
      roles: {user: "", assistant: ""}, seps: [""], role_content_sep: "", role_empty_sep: "",
      system_prefix_token_ids: [], stop_token_ids: [2],
    };
    return artifact(entry.manifest.path, "configuration", JSON.stringify(values));
  });
  const rawBase = await prepareVerifiedWebLlmArtifacts(baseWithSourceChatTemplate);
  assert.equal(rawBase.isBaseModel, true);
  assert.equal(rawBase.defaultUserRole, null);
  const mislabeledChat = baseWithSourceChatTemplate.map(entry => {
    if (entry.manifest.path !== "mlc-chat-config.json") return entry;
    const values = JSON.parse(config);
    values.vocab_size = 16;
    values.drowse_completion_prefix_token_ids = [];
    values.conv_template.name = "drowse-base";
    values.conv_template.system_prefix_token_ids = [];
    return artifact(entry.manifest.path, "configuration", JSON.stringify(values));
  });
  await assert.rejects(prepareVerifiedWebLlmArtifacts(mislabeledChat),
    error => error.code === "MLC_COMPLETION_PREFIX_INVALID");

  const prefixedBase = (prefix = [1], capturePrefix = prefix) => baseModelFixture.map((entry) => {
    if (entry.manifest.path === "tokenizer_config.json") {
      return artifact("tokenizer_config.json", "chat_template", JSON.stringify({add_bos_token: true}));
    }
    if (entry.manifest.path === "mlc-chat-config.json") {
      const values = JSON.parse(config);
      values.vocab_size = 16;
      values.drowse_completion_prefix_token_ids = prefix;
      values.conv_template.system_prefix_token_ids = capturePrefix;
      return artifact("mlc-chat-config.json", "configuration", JSON.stringify(values));
    }
    return entry;
  });
  assert.equal((await prepareVerifiedWebLlmArtifacts(prefixedBase())).isBaseModel, true);
  for (const prefix of [null, [], [-1], [16], [1.5], ["1"], [2], [1, 7]]) {
    await assert.rejects(prepareVerifiedWebLlmArtifacts(prefixedBase(prefix)),
      (error) => error.code === "MLC_COMPLETION_PREFIX_INVALID");
  }
  await assert.rejects(prepareVerifiedWebLlmArtifacts(prefixedBase([1], [])),
    (error) => error.code === "MLC_COMPLETION_PREFIX_INVALID");
  const missingBosPrefix = baseModelFixture.map((entry) => entry.manifest.path === "tokenizer_config.json"
    ? artifact("tokenizer_config.json", "chat_template", JSON.stringify({add_bos_token: true})) : entry);
  await assert.rejects(prepareVerifiedWebLlmArtifacts(missingBosPrefix),
    (error) => error.code === "MLC_COMPLETION_PREFIX_INVALID");

  const legacyConfig = JSON.stringify({
    tokenizer_files: [
      "tokenizer.json",
      "vocab.json",
      "merges.txt",
      "special_tokens_map.json",
      "tokenizer_config.json",
    ],
    bos_token_id: 1,
    eos_token_id: 2,
    saklas_capture_special_token_ids: [1, 2, 9, 10],
    conv_template: {
      roles: {
        user: "<start_of_turn>user",
        assistant: "<start_of_turn>model",
      },
    },
  });
  const legacyFixture = fixture().map((entry) => {
    if (entry.manifest.path === "mlc-chat-config.json") {
      return artifact("mlc-chat-config.json", "configuration", legacyConfig);
    }
    if (entry.manifest.path === "tokenizer_config.json") {
      return artifact("tokenizer_config.json", "chat_template", JSON.stringify({
        chat_template: "{{ messages }}",
        added_tokens_decoder: {
          10: { content: "<legacy-control>", special: true },
        },
      }));
    }
    return entry;
  });
  const preparedLegacy = await prepareVerifiedWebLlmArtifacts(legacyFixture);
  assert.deepEqual(preparedLegacy.captureSpecialTokenIds, [1, 2, 9, 10]);

  const metadataFreeFixture = legacyFixture.map((entry) => {
    if (entry.manifest.path !== "mlc-chat-config.json") return entry;
    const metadataFreeConfig = JSON.parse(legacyConfig);
    delete metadataFreeConfig.saklas_capture_special_token_ids;
    return artifact(
      "mlc-chat-config.json",
      "configuration",
      JSON.stringify(metadataFreeConfig),
    );
  });
  const preparedMetadataFree = await prepareVerifiedWebLlmArtifacts(metadataFreeFixture);
  assert.deepEqual(preparedMetadataFree.captureSpecialTokenIds, [1, 2, 9, 10]);

  const malformedCaptureIds = fixture();
  malformedCaptureIds[0] = artifact(
    "mlc-chat-config.json",
    "configuration",
    JSON.stringify({
      tokenizer_files: ["tokenizer.json", "tokenizer_config.json"],
      bos_token_id: 1,
      eos_token_id: 2,
      drowse_capture_special_token_ids: null,
      conv_template: {
        roles: {
          user: "<start_of_turn>user",
          assistant: "<start_of_turn>model",
        },
      },
    }),
  );
  await assert.rejects(
    prepareVerifiedWebLlmArtifacts(malformedCaptureIds),
    (error) => error.code === "MLC_CAPTURE_TOKEN_IDS_INVALID",
  );
  assert.deepEqual(
    await prepared.artifactCache.fetchWithCache(`${base}tensor-cache.json`, "json"),
    JSON.parse(tensorCache),
  );
  assert.deepEqual(
    new Uint8Array(
      await prepared.artifactCache.fetchWithCache(`${base}params_shard_0.bin`, "arraybuffer"),
    ),
    new Uint8Array([1, 2, 3]),
  );
  assert.equal(
    await prepared.artifactCache.hasAllKeys([
      `${base}mlc-chat-config.json`,
      `${base}params_shard_0.bin`,
    ]),
    true,
  );
  assert.equal(
    await prepared.artifactCache.hasAllKeys([`${base}missing.bin`]),
    false,
  );
  await assert.rejects(
    prepared.artifactCache.fetchWithCache(`${base}missing.bin`, "arraybuffer"),
    (error) => error.code === "UNVERIFIED_ARTIFACT_REQUESTED",
  );
  await assert.rejects(
    prepared.artifactCache.addToCache(`${base}missing.bin`, "arraybuffer"),
    (error) => error.code === "UNVERIFIED_ARTIFACT_REQUESTED",
  );
  await assert.rejects(
    prepared.artifactCache.deleteInCache(`${base}tokenizer.json`),
    (error) => error.code === "VERIFIED_ARTIFACT_READ_ONLY",
  );

  const missingWeight = fixture().filter(({ manifest }) => manifest.role !== "weight");
  await assert.rejects(
    prepareVerifiedWebLlmArtifacts(missingWeight),
    (error) => error.code === "MLC_WEIGHT_CLOSURE_INVALID",
  );

  const extraWeight = fixture();
  extraWeight.push(artifact("params_shard_1.bin", "weight", new Uint8Array([4])));
  await assert.rejects(
    prepareVerifiedWebLlmArtifacts(extraWeight),
    (error) => error.code === "MLC_WEIGHT_CLOSURE_INVALID",
  );

  const missingTokenizerSupport = fixture().filter(
    ({ manifest }) => manifest.path !== "merges.txt",
  );
  await assert.rejects(
    prepareVerifiedWebLlmArtifacts(missingTokenizerSupport),
    (error) => error.code === "MLC_TOKENIZER_INVALID",
  );

  const traversal = fixture();
  const traversalManifest = JSON.stringify({
    records: [{ dataPath: "../params_shard_0.bin", nbytes: 3, records: [] }],
  });
  traversal[2] = artifact("tensor-cache.json", "converted_manifest", traversalManifest);
  await assert.rejects(
    prepareVerifiedWebLlmArtifacts(traversal),
    (error) => error.code === "MLC_TENSOR_CACHE_INVALID",
  );

  const encodedTraversal = fixture();
  const encodedTraversalManifest = JSON.stringify({
    records: [{ dataPath: "%2e%2e/params_shard_0.bin", nbytes: 3, records: [] }],
  });
  encodedTraversal[2] = artifact(
    "tensor-cache.json",
    "converted_manifest",
    encodedTraversalManifest,
  );
  await assert.rejects(
    prepareVerifiedWebLlmArtifacts(encodedTraversal),
    (error) => error.code === "MLC_TENSOR_CACHE_INVALID",
  );

  const missingCaptureToken = fixture();
  missingCaptureToken[0] = artifact(
    "mlc-chat-config.json",
    "configuration",
    JSON.stringify({
      tokenizer_files: ["tokenizer.json", "tokenizer_config.json"],
      bos_token_id: 1,
      eos_token_id: 2,
      drowse_capture_special_token_ids: [1, 2],
      conv_template: {
        roles: {
          user: "<start_of_turn>user",
          assistant: "<start_of_turn>model",
        },
      },
    }),
  );
  await assert.rejects(
    prepareVerifiedWebLlmArtifacts(missingCaptureToken),
    (error) => error.code === "MLC_CAPTURE_TOKEN_IDS_INVALID",
  );

  const queriedShard = fixture();
  const queriedManifest = JSON.stringify({
    records: [{ dataPath: "params_shard_0.bin?raw=1", nbytes: 3, records: [] }],
  });
  queriedShard[2] = artifact("tensor-cache.json", "converted_manifest", queriedManifest);
  await assert.rejects(
    prepareVerifiedWebLlmArtifacts(queriedShard),
    (error) => error.code === "MLC_TENSOR_CACHE_INVALID",
  );

  const controller = new AbortController();
  controller.abort(new DOMException("cancelled", "AbortError"));
  await assert.rejects(
    prepareVerifiedWebLlmArtifacts(fixture(), controller.signal),
    (error) => error.name === "AbortError",
  );

  let constructedEngine;
  class ConstructionProbeEngine {
    constructor(engineConfig) {
      this.engineConfig = engineConfig;
      constructedEngine = this;
    }
    async reload() { throw new Error("stop after production engine construction"); }
    async unload() {}
  }
  const runtime = new DrowseWebLlmRuntime({
    DROWSE_HOOK_ABI: "post-block-residual-v4",
    MLCEngine: ConstructionProbeEngine,
  });
  await assert.rejects(runtime.load({
    model: { id: "fixture-model" },
    variant: {
      id: "fixture-q4",
      structuredHookProfile: "standard-v1",
      thinkingProfile: null,
      runtimeIdentity: {
        hookAbi: "post-block-residual-v4",
        hiddenSize: 2,
        layerMap: [0, 1],
      },
      requirements: {
        features: ["shader-f16"],
        limits: {
          maxStorageBufferBindingSize: 134_217_728,
          maxStorageBuffersPerShaderStage: 10,
        },
      },
    },
    requiredCorePack: {},
    contextTokens: 2048,
    adapter: {
      features: new Set(["shader-f16"]),
      limits: {
        maxBufferSize: 268_435_456,
        maxStorageBufferBindingSize: 134_217_728,
        maxStorageBuffersPerShaderStage: 10,
      },
      requestDevice() {},
    },
    artifacts: fixture(),
    optionalPacks: [],
    activationSpool: {},
    signal: new AbortController().signal,
    onDeviceLost() {},
  }), /stop after production engine construction/u);
  assert.ok(constructedEngine);
  assert.ok(constructedEngine.engineConfig.appConfig.artifactCache);
  assert.equal(Object.hasOwn(constructedEngine.engineConfig, "cacheBackend"), false);
  assert.equal(Object.hasOwn(constructedEngine.engineConfig.appConfig, "cacheBackend"), false);
  assert.equal(Object.hasOwn(constructedEngine.engineConfig, "opfsAccessMode"), false);
  assert.equal(Object.hasOwn(constructedEngine.engineConfig.appConfig, "opfsAccessMode"), false);
  await assert.rejects(
    constructedEngine.engineConfig.appConfig.artifactCache.fetchWithCache(
      `${base}outside-install.bin`,
      "arraybuffer",
    ),
    (error) => error.code === "UNVERIFIED_ARTIFACT_REQUESTED",
  );

  console.log("verified artifact cache checks passed");
} finally {
  await server.close();
}
