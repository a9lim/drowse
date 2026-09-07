import type {
  CatalogDocument,
  CatalogInstrumentPack,
  RuntimeCapabilities,
  VerifiedCatalog,
} from "../../lib/runtime/contracts";
import { DeterministicBrowserModelBackend } from "./fakeModelBackend";
import { BrowserContentStore } from "./contentStore";
import { VerifiedArtifactDownloader } from "./downloader";
import { HostedRuntimeWorker } from "./worker";

const revision = "b".repeat(40);
const runtimeIdentitySha256 = "a".repeat(64);
const contextBindingSha256 = "c".repeat(64);
const modelSha256 = "7867f3a41523c10847863e0dc1e7de68e99aa6a486e48bd86fbac122540f3fd2";
const coreSha256 = "2ae33cdf163ca521afdb50969cc456c94493e120a9a792971039afe0f0b80e95";
const jlensSha256 = "bd202dd90b332ce80c5ca3062eac16128c9388c2d36d06140ad2fe25b1a8da4f";
const saeSha256 = "4a275f923eb563e6cfa1dec9af87a4a61e0de6e18b3c2b3751b143934cb4df57";
const runtimeAbi = "drowse-web-fixture-v1";
const fixtureModelUrl = new URL("./fixtures/model.fixture.bin", import.meta.url).href;
const fixtureCoreUrl = new URL("./fixtures/core.fixture.bin", import.meta.url).href;
const fixtureJlensUrl = new URL("./fixtures/jlens.fixture.bin", import.meta.url).href;
const fixtureSaeUrl = new URL("./fixtures/sae.fixture.bin", import.meta.url).href;
const fixtureModelSource = "https://fixture.invalid/model.fixture.bin";
const fixtureCoreSource = "https://fixture.invalid/core.fixture.bin";
const fixtureJlensSource = "https://fixture.invalid/jlens.fixture.bin";
const fixtureSaeSource = "https://fixture.invalid/sae.fixture.bin";
const saeAvailable = !self.name.includes("-without-sae");
const generationDelayMs = self.name.includes("-slow-generation") ? 2_000 : 35;
const fixtureSaePack: CatalogInstrumentPack = {
  id: "fixture-sae",
  kind: "sae",
  displayName: "SAE features",
  license: "AGPL-3.0-or-later",
  sourceRepository: "a9lim/drowse-web-fixture",
  sourceRevision: revision,
  bytes: 58,
  required: false,
  runtimeIdentitySha256,
  compatibleContextBindingSha256: [contextBindingSha256],
  files: [{
    path: "fixture/sae.bin",
    role: "instrument",
    url: fixtureSaeSource,
    revision,
    bytes: 58,
    sha256: saeSha256,
  }],
};

const document: CatalogDocument = {
  schemaVersion: 1,
  sequence: 1,
  issuedAt: "2026-01-01T00:00:00.000Z",
  expiresAt: "2099-01-01T00:00:00.000Z",
  runtimeAbi,
  models: [{
    id: "qwen3-1.7b",
    displayName: "Qwen3 1.7B fixture",
    description: "Deterministic worker-backed development runtime",
    sourceUrl: "https://huggingface.co/Qwen/Qwen3-1.7B",
    license: "Apache-2.0",
    languages: ["en"],
    variants: [{
      id: "qwen3-1.7b-fixture",
      tier: "fastest",
      structuredHookProfile: "standard-v1",
      thinkingProfile: null,
      contextProfiles: [{
        contextTokens: 2048,
        bindingSha256: contextBindingSha256,
        minimumCalibrationScore: null,
        minimumDeviceMemoryGiB: null,
        expectedPrefillTokensPerSecond: [100, 140],
        expectedDecodeTokensPerSecond: [40, 56],
        measuredDevices: 1,
      }],
      downloadBytes: 49,
      requiredCorePackBytes: 57,
      requirements: {
        features: [],
        limits: {
          maxBufferSize: 21_472,
          maxStorageBufferBindingSize: 21_472,
          maxStorageBuffersPerShaderStage: 10,
          maxComputeWorkgroupStorageSize: 32_768,
        },
      },
      runtimeIdentity: {
        sourceModel: "fixture/Qwen3-1.7B",
        sourceRevision: revision,
        convertedManifestSha256: runtimeIdentitySha256,
        quantization: "fixture-fp32",
        tokenizerSha256: runtimeIdentitySha256,
        chatTemplateSha256: runtimeIdentitySha256,
        modelLibrarySha256: runtimeIdentitySha256,
        runtimeAbi,
        hookAbi: "post-block-residual-v4",
        hiddenSize: 64,
        layerMap: [0, 1],
      },
      runtimeIdentitySha256,
      files: [{
        path: "fixture/model.bin",
        role: "weight",
        url: fixtureModelSource,
        revision,
        bytes: 49,
        sha256: modelSha256,
      }],
      packs: [{
        id: "fixture-core",
        kind: "core",
        displayName: "Fixture core geometry",
        license: "AGPL-3.0-or-later",
        sourceRepository: "a9lim/drowse-web-fixture",
        sourceRevision: revision,
        bytes: 57,
        required: true,
        runtimeIdentitySha256,
        compatibleContextBindingSha256: [contextBindingSha256],
        files: [{
          path: "fixture/core.bin",
          role: "core_pack",
          url: fixtureCoreSource,
          revision,
          bytes: 57,
          sha256: coreSha256,
        }],
      }, {
        id: "fixture-jlens",
        kind: "jlens",
        displayName: "J-lens readouts",
        license: "AGPL-3.0-or-later",
        sourceRepository: "a9lim/drowse-web-fixture",
        sourceRevision: revision,
        bytes: 61,
        required: false,
        runtimeIdentitySha256,
        compatibleContextBindingSha256: [contextBindingSha256],
        files: [{
          path: "fixture/jlens.bin",
          role: "instrument",
          url: fixtureJlensSource,
          revision,
          bytes: 61,
          sha256: jlensSha256,
        }],
      }, ...(saeAvailable ? [fixtureSaePack] : []), {
        id: "fixture-rlens",
        kind: "jlens",
        displayName: "R-lens readouts",
        license: "AGPL-3.0-or-later",
        sourceRepository: "a9lim/drowse-web-fixture",
        sourceRevision: revision,
        bytes: 61,
        required: false,
        runtimeIdentitySha256,
        compatibleContextBindingSha256: [contextBindingSha256],
        files: [{
          path: "fixture/rlens.bin",
          role: "instrument",
          url: fixtureJlensSource,
          revision,
          bytes: 61,
          sha256: jlensSha256,
        }],
      }],
    }],
  }],
};

const catalog: VerifiedCatalog = {
  document,
  exactBytes: new TextEncoder().encode(JSON.stringify(document)),
  signature: {
    schemaVersion: 1,
    algorithm: "Ed25519",
    keyId: "development-fixture",
    signature: "development-only",
  },
  allowDownloads: true,
  stale: false,
};

const capabilities: RuntimeCapabilities = {
  checkedAt: 0,
  supported: true,
  secureContext: true,
  crossOriginIsolated: true,
  dedicatedWorker: true,
  indexedDb: true,
  opfs: true,
  webLocks: true,
  broadcastChannel: true,
  webAssembly: true,
  webGpu: {
    available: true,
    fallback: "hardware",
    features: [],
    limits: {
      maxBufferSize: 21_472,
      maxStorageBufferBindingSize: 21_472,
      maxStorageBuffersPerShaderStage: 10,
      maxComputeWorkgroupStorageSize: 32_768,
    },
    adapterInfo: { vendor: "deterministic worker fixture" },
  },
  storage: {
    quotaBytes: 2_000_000_000,
    usageBytes: 0,
    availableBytes: 2_000_000_000,
    persisted: true,
  },
  signals: {
    deviceMemoryGiB: 8,
    logicalCpuCount: 8,
    mobile: false,
    language: "en-US",
    calibrationScore: 100,
  },
  deviceSignature: "development-fixture-device",
  runtimeVersion: runtimeAbi,
  issues: [],
  operations: {
    download: { available: true, reasons: [] },
    generation: { available: true, reasons: [] },
    fitting: {
      available: false,
      reasons: [{
        code: "FIXTURE_FITTING_DISABLED",
        message: "The deterministic development fixture does not run fitting jobs",
        severity: "hard",
      }],
    },
    manifold_artifacts: { available: true, reasons: [] },
    probe_subspace_trails: { available: true, reasons: [] },
    jlens_fitting: { available: false, reasons: [] },
    sae_training: { available: false, reasons: [] },
    session_admin: { available: false, reasons: [] },
    server_endpoints: { available: false, reasons: [] },
  },
  limits: { manifoldFitMaxIntrinsicDim: 4 },
};

async function fixtureFetch(url: string, init: RequestInit): Promise<Response> {
  const assetUrl = url === fixtureModelSource
    ? fixtureModelUrl
    : url === fixtureCoreSource
      ? fixtureCoreUrl
      : url === fixtureJlensSource
        ? fixtureJlensUrl
        : url === fixtureSaeSource
          ? fixtureSaeUrl
      : null;
  if (assetUrl === null) throw new TypeError(`Unknown fixture artifact ${url}`);
  const response = await fetch(assetUrl, init);
  Object.defineProperty(response, "url", { value: url });
  return response;
}

const contentStore = new BrowserContentStore();

new HostedRuntimeWorker(
  globalThis as unknown as import("./worker").WorkerScopeLike,
  {
    capabilityChecker: {
      async check() {
        return { ...capabilities, checkedAt: Date.now() };
      },
      clearAdapter() {},
    },
    catalogRepository: {
      async get() {
        return catalog;
      },
      async withDownloadAuthorization(_catalog, operation) {
        return operation();
      },
      async clear() {},
    },
    contentStore,
    downloader: new VerifiedArtifactDownloader(contentStore, { fetch: fixtureFetch }),
    modelBackend: new DeterministicBrowserModelBackend(generationDelayMs, {
      role_substitution_supported: true,
      user_role_supported: true,
      scene_mode: true,
    }),
  },
);
