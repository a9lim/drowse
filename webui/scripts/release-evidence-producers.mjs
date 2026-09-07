import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { lstat, readFile, realpath } from "node:fs/promises";
import { dirname, relative, resolve, sep } from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const locallyExecutedReceiptCandidates = new WeakSet();

const PRODUCER_EXECUTABLE = "webui/scripts/hosted-release-evidence-producer.mjs";
const COMMON_CLOSURE = Object.freeze([
  PRODUCER_EXECUTABLE,
  "webui/scripts/release-evidence-producers.mjs",
  "webui/scripts/record-hosted-evidence.mjs",
  "webui/scripts/check-runtime-lock.mjs",
  "browser-runtime/evidence-config.schema.json",
  "browser-runtime/release-evidence-record.schema.json",
]);

const runtimeFeasibilityClosure = Object.freeze([
  ...COMMON_CLOSURE,
  "webui/scripts/browser-runtime-feasibility.mjs",
  "webui/scripts/browser-full-runtime.mjs",
  "webui/scripts/browser-full-runtime-contract.mjs",
  "webui/scripts/browser-full-runtime-artifact-audit.mjs",
  "webui/scripts/hosted-model-profiles.mjs",
  "webui/scripts/runtime-lock-document.mjs",
  "webui/scripts/publish-hosted-model.mjs",
  "webui/scripts/hosted-model-closure.mjs",
  "webui/scripts/hugging-face-upload.mjs",
  "webui/scripts/hosted-full-runtime.worker.ts",
  "webui/scripts/capture-hosted-residuals.mjs",
  "webui/scripts/hosted-residual-capture.worker.ts",
  "browser-runtime/forks/create-tiny-webgpu-model.py",
  "browser-runtime/forks/compile-tiny-webgpu.py",
  "browser-runtime/forks/manifest.json",
  "browser-runtime/fixtures/post-block-rank-one-v2.json",
  "browser-runtime/fixtures/structured-program-parity-v1.json",
  "browser-runtime/fixtures/sampling-parity-v1.json",
  "browser-runtime/fixtures/production-q4-parity-input-v1.json",
  "browser-runtime/fixtures/smollm2-q4-parity-input-v3.json",
  "browser-runtime/mlc_q4_torch.py",
  "browser-runtime/validate_mlc_q4_parity.py",
]);

const authoringClosure = Object.freeze([
  ...COMMON_CLOSURE,
  "webui/scripts/browser-full-runtime.mjs",
  "webui/scripts/browser-full-runtime-contract.mjs",
  "webui/scripts/browser-full-runtime-artifact-audit.mjs",
  "webui/scripts/hosted-model-profiles.mjs",
  "webui/scripts/hosted-full-runtime.worker.ts",
  "webui/scripts/browser-manifold-fitting.test.mjs",
  "webui/scripts/fitting-wasm-browser.test.mjs",
  "webui/scripts/hosted-release-local-observer.mjs",
  "webui/scripts/drowse-archive-browser.test.mjs",
  "webui/scripts/browser-core-pack.test.mjs",
  "webui/scripts/steering-expression.test.mjs",
  "webui/scripts/browser-loom.test.mjs",
  "webui/scripts/worker-runtime.test.mjs",
  "webui/scripts/structured-hook-parity.test.mjs",
  "webui/scripts/sampling-parity.test.mjs",
  "webui/scripts/web-llm-generation.test.mjs",
  "webui/scripts/build-fitting-wasm.mjs",
  "webui/src/hosted/runtime/browserCorePack.ts",
  "webui/src/hosted/runtime/browserInstrumentPacks.ts",
  "webui/src/hosted/runtime/browserLoom.ts",
  "webui/src/hosted/runtime/roleSlug.ts",
  "webui/src/hosted/runtime/steeringExpression.ts",
  "webui/src/hosted/runtime/structuredHookProgram.ts",
  "webui/src/hosted/runtime/webLlmGeneration.ts",
  "webui/src/hosted/artifacts/drowseArchive.ts",
  "webui/src/hosted/artifacts/safetensors.ts",
  "webui/src/hosted/artifacts/zip.ts",
  "webui/src/hosted/fitting/coordinator.ts",
  "webui/public-hosted/wasm/fitting-kernel.json",
  "webui/public-hosted/wasm/drowse_fitting_wasm.js",
  "webui/public-hosted/wasm/drowse_fitting_wasm_bg.wasm",
  "browser-runtime/fitting-wasm/Cargo.lock",
  "browser-runtime/fitting-wasm/Cargo.toml",
  "browser-runtime/fitting-wasm/src/error.rs",
  "browser-runtime/fitting-wasm/src/kernel.rs",
  "browser-runtime/fitting-wasm/src/lib.rs",
  "browser-runtime/fitting-wasm/src/linalg.rs",
  "browser-runtime/fitting-wasm/tests/golden.rs",
  "browser-runtime/fixtures/affine-fisher-fit-v1.json",
  "browser-runtime/fixtures/post-block-rank-one-v2.json",
  "browser-runtime/fixtures/structured-program-parity-v1.json",
  "browser-runtime/fixtures/measurement-envelope-v1.json",
  "browser-runtime/fixtures/loom-operation-transcript-v1.json",
  "browser-runtime/fixtures/topology-rbf-parity-v1.json",
  "browser-runtime/fixtures/sampling-parity-v1.json",
  "browser-runtime/fixtures/drowse-interchange-v1.json",
  "browser-runtime/fixtures/shared-parity-suite-v1.json",
  "tests/browser_drowse_archive_bridge.py",
  "tests/test_browser_loom_transcript_fixture.py",
  "tests/test_browser_measurements_fixture.py",
  "tests/test_browser_sampling_fixture.py",
  "tests/test_browser_steering_fixture.py",
  "tests/test_browser_structured_program_fixture.py",
  "tests/test_browser_topology_rbf_fixture.py",
  "tests/test_drowse_archive.py",
  "drowse/core/generation.py",
  "drowse/core/jlens.py",
  "drowse/core/loom.py",
  "drowse/core/manifold.py",
  "drowse/core/measurements.py",
  "drowse/core/sae.py",
  "drowse/core/steering_expr.py",
  "drowse/core/topology.py",
  "drowse/io/drowse_archive.py",
]);

const benchmarkClosure = Object.freeze([
  ...COMMON_CLOSURE,
  "webui/scripts/browser-full-runtime.mjs",
  "webui/scripts/browser-full-runtime-contract.mjs",
  "webui/scripts/browser-full-runtime-artifact-audit.mjs",
  "webui/scripts/hosted-model-profiles.mjs",
  "webui/scripts/hosted-full-runtime.worker.ts",
]);

function descriptor(id, kind, closure, fixturePath) {
  return Object.freeze({
    id,
    kind,
    executable: PRODUCER_EXECUTABLE,
    closure,
    fixturePath,
  });
}

const RELEASE_EVIDENCE_PRODUCERS = new Map([
  [
    "gemmaTinyFp32Parity",
    descriptor(
      "gemma-tiny-fp32-parity-v1",
      "gate",
      runtimeFeasibilityClosure,
      "browser-runtime/fixtures/post-block-rank-one-v2.json",
    ),
  ],
  [
    "llamaTinyFp32Parity",
    descriptor(
      "llama-tiny-fp32-parity-v1",
      "gate",
      runtimeFeasibilityClosure,
      "browser-runtime/fixtures/post-block-rank-one-v2.json",
    ),
  ],
  [
    "qwenTinyFp32Parity",
    descriptor(
      "qwen-tiny-fp32-parity-v1",
      "gate",
      runtimeFeasibilityClosure,
      "browser-runtime/fixtures/post-block-rank-one-v2.json",
    ),
  ],
  [
    "gemmaProductionQ4Parity",
    descriptor(
      "gemma-production-q4-parity-v1",
      "gate",
      runtimeFeasibilityClosure,
      "browser-runtime/fixtures/production-q4-parity-input-v1.json",
    ),
  ],
  [
    "smolProductionQ4Parity",
    descriptor(
      "smol-production-q4-parity-v1",
      "gate",
      runtimeFeasibilityClosure,
      "browser-runtime/fixtures/smollm2-q4-parity-input-v3.json",
    ),
  ],
  [
    "qwenProductionQ4Parity",
    descriptor(
      "qwen-production-q4-parity-v1",
      "gate",
      runtimeFeasibilityClosure,
      "browser-runtime/fixtures/production-q4-parity-input-v1.json",
    ),
  ],
  [
    "gemmaLongPrefill",
    descriptor(
      "gemma-long-prefill-v1",
      "gate",
      runtimeFeasibilityClosure,
      "browser-runtime/runtime-lock-identity-v1.json",
    ),
  ],
  [
    "runtimeLifecycle",
    descriptor(
      "browser-runtime-lifecycle-v1",
      "gate",
      runtimeFeasibilityClosure,
      "browser-runtime/runtime-lock-identity-v1.json",
    ),
  ],
  [
    "activationCapture",
    descriptor(
      "browser-activation-capture-v1",
      "gate",
      authoringClosure,
      "browser-runtime/runtime-lock-identity-v1.json",
    ),
  ],
  [
    "fittingKernelParity",
    descriptor(
      "browser-fitting-kernel-parity-v1",
      "gate",
      authoringClosure,
      "browser-runtime/fixtures/affine-fisher-fit-v1.json",
    ),
  ],
  [
    "topologyOrchestration",
    descriptor(
      "browser-topology-orchestration-v1",
      "gate",
      authoringClosure,
      "browser-runtime/fixtures/topology-rbf-parity-v1.json",
    ),
  ],
  [
    "manifoldSerialization",
    descriptor(
      "browser-manifold-serialization-v1",
      "gate",
      authoringClosure,
      "browser-runtime/fixtures/drowse-interchange-v1.json",
    ),
  ],
  [
    "instrumentIntegration",
    descriptor(
      "browser-instrument-integration-v1",
      "gate",
      authoringClosure,
      "browser-runtime/runtime-lock-identity-v1.json",
    ),
  ],
  [
    "sharedGoldenResults",
    descriptor(
      "browser-shared-goldens-v1",
      "gate",
      authoringClosure,
      "browser-runtime/fixtures/shared-parity-suite-v1.json",
    ),
  ],
  [
    "physicalBrowserFitting",
    descriptor(
      "physical-browser-fitting-v1",
      "gate",
      authoringClosure,
      "browser-runtime/runtime-lock-identity-v1.json",
    ),
  ],
  [
    "physicalBrowserBenchmark",
    descriptor(
      "physical-browser-benchmark-v1",
      "benchmark",
      benchmarkClosure,
      null,
    ),
  ],
]);

export function requireReleaseEvidenceProducer(evidenceType) {
  const producer = RELEASE_EVIDENCE_PRODUCERS.get(evidenceType);
  if (!producer) {
    throw new Error(
      `${evidenceType} has no registered executable release evidence producer`,
    );
  }
  return producer;
}

export function releaseEvidenceProducerTypes() {
  return [...RELEASE_EVIDENCE_PRODUCERS.keys()];
}

export async function releaseEvidenceProducerToolVersion(
  evidenceType,
  repositoryRoot,
  configPath,
) {
  return (await releaseEvidenceProducerBinding(
    evidenceType,
    repositoryRoot,
    configPath,
  )).toolVersion;
}

export async function releaseEvidenceProducerBinding(
  evidenceType,
  repositoryRoot,
  configPath,
) {
  const producer = requireReleaseEvidenceProducer(evidenceType);
  const relativeConfig = repositoryRelativePath(
    repositoryRoot,
    configPath,
    "producer config",
  );
  const configBytes = await readRegularContainedFile(
    resolve(repositoryRoot, ...relativeConfig.split("/")),
    repositoryRoot,
    "producer config",
  );
  const digest = createHash("sha256");
  for (const path of producer.closure) {
    const bytes = await readRegularContainedFile(
      resolve(repositoryRoot, ...path.split("/")),
      repositoryRoot,
      "release evidence producer source",
    );
    digest.update(`${path}\0${bytes.byteLength}\0`);
    digest.update(bytes);
    digest.update("\0");
  }
  digest.update(`${relativeConfig}\0${configBytes.byteLength}\0`);
  digest.update(configBytes);
  digest.update("\0");
  return Object.freeze({
    producerId: producer.id,
    configPath: relativeConfig,
    configSha256: createHash("sha256").update(configBytes).digest("hex"),
    toolVersion: `${producer.id}+sha256.${digest.digest("hex")}`,
  });
}

export async function runReleaseEvidenceProducer({
  evidenceType,
  configPath,
  repositoryRoot,
  nonce,
  execute = execFileAsync,
}) {
  const producer = requireReleaseEvidenceProducer(evidenceType);
  if (typeof nonce !== "string" || !/^[0-9a-f]{64}$/.test(nonce)) {
    throw new Error("release evidence producer nonce is invalid");
  }
  const relativeConfig = repositoryRelativePath(
    repositoryRoot,
    configPath,
    "producer config",
  );
  const binding = await releaseEvidenceProducerBinding(
    evidenceType,
    repositoryRoot,
    configPath,
  );
  const executable = resolve(
    repositoryRoot,
    ...producer.executable.split("/"),
  );
  const { stdout, stderr } = await execute(
    process.execPath,
    [
      executable,
      "--evidence-type",
      evidenceType,
      "--config",
      relativeConfig,
      "--nonce",
      nonce,
    ],
    {
      cwd: repositoryRoot,
      encoding: "utf8",
      maxBuffer: 4 * 1024 * 1024,
      timeout: 2 * 60 * 60 * 1000,
      windowsHide: true,
    },
  );
  if (typeof stderr === "string" && stderr.trim()) {
    throw new Error("release evidence producer wrote to stderr");
  }
  let receipt;
  try {
    receipt = JSON.parse(stdout);
  } catch (error) {
    throw new Error("release evidence producer did not return one JSON receipt", {
      cause: error,
    });
  }
  validateProducerReceipt(receipt, producer, evidenceType, nonce);
  const boundReceipt = {
    ...receipt,
    producerConfigPath: binding.configPath,
    producerConfigSha256: binding.configSha256,
    producerToolVersion: binding.toolVersion,
  };
  deepFreeze(boundReceipt);
  locallyExecutedReceiptCandidates.add(boundReceipt);
  return boundReceipt;
}

export function assertExecutedProducerReceipt(receipt) {
  if (
    !receipt ||
    typeof receipt !== "object" ||
    !locallyExecutedReceiptCandidates.has(receipt)
  ) {
    throw new Error(
      "release evidence must come from an executed registered producer",
    );
  }
}

export function validateProducerReceipt(receipt, producer, evidenceType, nonce) {
  if (!receipt || typeof receipt !== "object" || Array.isArray(receipt)) {
    throw new Error("release evidence producer receipt must be an object");
  }
  const expected = producer.kind === "benchmark"
    ? [
        "schemaVersion",
        "evidenceType",
        "producerId",
        "nonce",
        "producedAt",
        "passed",
        "run",
      ]
    : [
        "schemaVersion",
        "evidenceType",
        "producerId",
        "nonce",
        "producedAt",
        "passed",
        "fixturePath",
        "results",
      ];
  exactKeys(receipt, expected, "release evidence producer receipt");
  if (
    receipt.schemaVersion !== 1 ||
    receipt.evidenceType !== evidenceType ||
    receipt.producerId !== producer.id ||
    receipt.nonce !== nonce ||
    receipt.passed !== true
  ) {
    throw new Error("release evidence producer receipt identity is invalid");
  }
  const producedAt = Date.parse(receipt.producedAt);
  if (!Number.isFinite(producedAt)) {
    throw new Error("release evidence producer receipt timestamp is invalid");
  }
  if (producer.kind === "gate" && receipt.fixturePath !== producer.fixturePath) {
    throw new Error("release evidence producer receipt fixture is not pinned");
  }
}

function repositoryRelativePath(repositoryRoot, path, label) {
  const value = relative(repositoryRoot, resolve(repositoryRoot, path));
  if (
    !value ||
    value === ".." ||
    value.startsWith(`..${sep}`) ||
    value.includes("\\") ||
    value.split(sep).some((part) => !part || part === "." || part === "..")
  ) {
    throw new Error(`${label} must be inside the Drowse repository`);
  }
  return value.split(sep).join("/");
}

async function readRegularContainedFile(path, root, label) {
  const [rootRealPath, parentRealPath, info] = await Promise.all([
    realpath(root),
    realpath(dirname(path)),
    lstat(path),
  ]);
  const parentRelative = relative(rootRealPath, parentRealPath);
  if (
    parentRelative === ".." ||
    parentRelative.startsWith(`..${sep}`) ||
    info.isSymbolicLink() ||
    !info.isFile() ||
    info.size > 4 * 1024 * 1024
  ) {
    throw new Error(`${label} must be a regular file inside the Drowse repository`);
  }
  return readFile(path);
}

function exactKeys(value, keys, label) {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (
    actual.length !== expected.length ||
    actual.some((key, index) => key !== expected[index])
  ) {
    throw new Error(`${label} has unknown or missing fields`);
  }
}

function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  for (const child of Object.values(value)) deepFreeze(child);
  return Object.freeze(value);
}
