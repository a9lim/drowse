import { createHash } from "node:crypto";

export const FIXTURE_ABI = "drowse-web-test-v1";
export const FIXTURE_REVISION = "a".repeat(40);
export const FIXTURE_SIGNATURE = {
  schemaVersion: 1,
  algorithm: "Ed25519",
  keyId: "fixture-key",
  signature: Buffer.alloc(64).toString("base64"),
};

export function catalogFixture(overrides = {}) {
  const runtimeIdentity = {
    sourceModel: "HuggingFaceTB/SmolLM2-360M-Instruct",
    sourceRevision: FIXTURE_REVISION,
    convertedManifestSha256: "5".repeat(64),
    quantization: "q4f16_1",
    tokenizerSha256: "6".repeat(64),
    chatTemplateSha256: "7".repeat(64),
    modelLibrarySha256: "8".repeat(64),
    runtimeAbi: FIXTURE_ABI,
    hookAbi: "post-block-residual-v4",
    hiddenSize: 960,
    layerMap: [0, 1],
  };
  const runtimeIdentitySha256 = digestCanonical(runtimeIdentity);
  const bindingSha256 = digestCanonical({
    contextTokens: 2048,
    runtimeIdentitySha256,
  });
  const file = (path, role, bytes, sha256) => ({
    path,
    role,
    url: `https://huggingface.co/a9lim/drowse-web-fixture/resolve/${FIXTURE_REVISION}/${path}`,
    revision: FIXTURE_REVISION,
    bytes,
    sha256,
  });
  const files = [
    file("weights/params.bin", "weight", 700, "3".repeat(64)),
    file("tokenizer/tokenizer.json", "tokenizer", 100, "6".repeat(64)),
    file("config/mlc-chat-config.json", "configuration", 100, "a".repeat(64)),
    file("config/ndarray-cache.json", "converted_manifest", 100, "5".repeat(64)),
    file("tokenizer/chat-template.json", "chat_template", 100, "7".repeat(64)),
    file("lib/model.wasm", "model_library", 100, "8".repeat(64)),
    file("LICENSE.model", "configuration", 100, "b1".repeat(32)),
    file("MODEL-LICENSE.json", "configuration", 100, "c1".repeat(32)),
    file("README.source.md", "configuration", 100, "d1".repeat(32)),
    file("drowse-build.json", "configuration", 100, "e1".repeat(32)),
    file("hosted-artifacts.json", "configuration", 100, "f1".repeat(32)),
  ];
  const document = {
    schemaVersion: 1,
    sequence: 7,
    issuedAt: "2026-01-01T00:00:00.000Z",
    expiresAt: "2027-01-01T00:00:00.000Z",
    runtimeAbi: FIXTURE_ABI,
    models: [{
      id: "smollm2-360m",
      displayName: "SmolLM2 360M",
      description: "Deterministic catalog fixture",
      sourceUrl: "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct",
      license: "Apache-2.0",
      languages: ["en"],
      variants: [{
        id: "smollm2-360m-q4f16",
        tier: "fastest",
        structuredHookProfile: "standard-v1",
        thinkingProfile: null,
        contextProfiles: [{
          contextTokens: 2048,
          bindingSha256,
          minimumCalibrationScore: 5,
          minimumDeviceMemoryGiB: 4,
          expectedPrefillTokensPerSecond: [20, 40],
          expectedDecodeTokensPerSecond: [8, 16],
          measuredDevices: 2,
        }],
        downloadBytes: files.reduce((sum, artifact) => sum + artifact.bytes, 0),
        requiredCorePackBytes: 250,
        requirements: {
          features: ["shader-f16"],
          limits: {
            maxBufferSize: 245_760,
            maxStorageBufferBindingSize: 245_760,
            maxStorageBuffersPerShaderStage: 10,
            maxComputeWorkgroupStorageSize: 32_768,
          },
        },
        runtimeIdentity,
        runtimeIdentitySha256,
        files,
        packs: [{
          id: "smollm2-core",
          kind: "core",
          displayName: "Core geometry",
          license: "AGPL-3.0-or-later",
          sourceRepository: "a9lim/drowse-web-fixture",
          sourceRevision: FIXTURE_REVISION,
          bytes: 250,
          required: true,
          runtimeIdentitySha256,
          compatibleContextBindingSha256: [bindingSha256],
          files: [file("packs/core.safetensors", "core_pack", 250, "4".repeat(64))],
        }, {
          id: "smollm2-jlens",
          kind: "jlens",
          displayName: "SmolLM2 J-lens",
          license: "Apache-2.0",
          sourceRepository: "a9lim/drowse-web-fixture-instruments",
          sourceRevision: FIXTURE_REVISION,
          bytes: 40,
          required: false,
          runtimeIdentitySha256,
          compatibleContextBindingSha256: [bindingSha256],
          files: [{
            ...file("packs/jlens.safetensors", "instrument", 40, "9".repeat(64)),
            url: `https://huggingface.co/a9lim/drowse-web-fixture-instruments/resolve/${FIXTURE_REVISION}/packs/jlens.safetensors`,
          }],
        }],
      }],
    }],
  };
  return Object.assign(document, overrides);
}

export function catalogBytes(overrides = {}) {
  return new TextEncoder().encode(JSON.stringify(catalogFixture(overrides)));
}

export function catalogRuntimeLockModels(document = catalogFixture()) {
  return document.models.flatMap((model) => model.variants.map((variant) => ({
    runtimeIdentity: structuredClone(variant.runtimeIdentity),
    structuredHookProfile: variant.structuredHookProfile,
    thinkingProfile: structuredClone(variant.thinkingProfile),
    convertedRepository: "a9lim/drowse-web-fixture",
    convertedRevision: FIXTURE_REVISION,
    contextProfiles: variant.contextProfiles.map((profile) => profile.contextTokens),
  })));
}

function digestCanonical(value) {
  return createHash("sha256").update(canonicalJson(value)).digest("hex");
}

function canonicalJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.entries(value)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
    .join(",")}}`;
}
