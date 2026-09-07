import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, rm, unlink, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { test } from "node:test";
import runtimeLock from "../../browser-runtime/runtime-lock.json" with { type: "json" };
import {
  createBenchmarkRunRecord,
  createGateEvidenceRecord,
  parseArguments,
  resolveEvidenceOutput,
  unpermittedWorktreeEntries,
  writeNewJson,
} from "./record-hosted-evidence.mjs";
import {
  releaseEvidenceProducerToolVersion,
  releaseEvidenceProducerTypes,
  requireReleaseEvidenceProducer,
  runReleaseEvidenceProducer,
  validateProducerReceipt,
} from "./release-evidence-producers.mjs";
import {
  adaptObserverReports,
  observerEnvironment,
  parseProducerArguments,
  runConfiguredObserver,
} from "./hosted-release-evidence-producer.mjs";
import {
  lockedRuntimeIdentitySha256,
  validateBenchmarkProducerBindings,
} from "./check-runtime-lock.mjs";
import {
  collectEvidenceAttestationSubjects,
  parseAttestationArguments,
  verifyEvidenceAttestations,
} from "./verify-hosted-evidence-attestations.mjs";

const repositoryRoot = resolve(import.meta.dirname, "../..");
const revision = "a".repeat(40);
const producedAt = "2026-08-29T10:00:00.000Z";
const now = Date.parse("2026-08-29T10:01:00.000Z");
const nonce = "b".repeat(64);
const producerConfig = resolve(
  repositoryRoot,
  "browser-runtime/evidence-configs/fitting-kernel-parity.json",
);
const evidenceTypes = [
  "gemmaTinyFp32Parity",
  "llamaTinyFp32Parity",
  "qwenTinyFp32Parity",
  "gemmaProductionQ4Parity",
  "smolProductionQ4Parity",
  "qwenProductionQ4Parity",
  "gemmaLongPrefill",
  "runtimeLifecycle",
  "activationCapture",
  "fittingKernelParity",
  "topologyOrchestration",
  "manifoldSerialization",
  "instrumentIntegration",
  "sharedGoldenResults",
  "physicalBrowserFitting",
  "physicalBrowserBenchmark",
];

test("every release gate and benchmark has a pinned executable producer", () => {
  assert.deepEqual(releaseEvidenceProducerTypes(), evidenceTypes);
  for (const evidenceType of evidenceTypes) {
    const producer = requireReleaseEvidenceProducer(evidenceType);
    assert.equal(
      producer.executable,
      "webui/scripts/hosted-release-evidence-producer.mjs",
    );
    assert.ok(producer.closure.includes(producer.executable));
    assert.match(producer.id, /^[a-z0-9-]+-v[0-9]+$/);
  }
  assert.throws(
    () => requireReleaseEvidenceProducer("inventedGate"),
    /no registered executable release evidence producer/,
  );
});

test("producer tool identity covers its exact checked-in source closure", async () => {
  const first = await releaseEvidenceProducerToolVersion(
    "gemmaTinyFp32Parity",
    repositoryRoot,
    producerConfig,
  );
  const second = await releaseEvidenceProducerToolVersion(
    "gemmaTinyFp32Parity",
    repositoryRoot,
    producerConfig,
  );
  assert.equal(first, second);
  assert.match(
    first,
    /^gemma-tiny-fp32-parity-v1\+sha256\.[0-9a-f]{64}$/,
  );

  const temporaryConfig = resolve(
    repositoryRoot,
    "browser-runtime/evidence-configs",
    `binding-test-${process.pid}-${Date.now()}.json`,
  );
  try {
    await writeFile(temporaryConfig, '{"fixture":1}\n', { flag: "wx" });
    const before = await releaseEvidenceProducerToolVersion(
      "gemmaTinyFp32Parity",
      repositoryRoot,
      temporaryConfig,
    );
    await writeFile(temporaryConfig, '{"fixture":2}\n');
    const after = await releaseEvidenceProducerToolVersion(
      "gemmaTinyFp32Parity",
      repositoryRoot,
      temporaryConfig,
    );
    assert.notEqual(before, after);
  } finally {
    await unlink(temporaryConfig).catch(() => undefined);
  }
});

test("an executed receipt is invalidated when its producer config changes", async () => {
  const temporaryConfig = resolve(
    repositoryRoot,
    "browser-runtime/evidence-configs",
    `binding-recheck-${process.pid}-${Date.now()}.json`,
  );
  const fixture = await trackedFixture(
    "browser-runtime/fixtures/post-block-rank-one-v2.json",
  );
  try {
    await writeFile(temporaryConfig, '{"fixture":1}\n', { flag: "wx" });
    const receipt = await runReleaseEvidenceProducer({
      evidenceType: "gemmaTinyFp32Parity",
      configPath: temporaryConfig,
      repositoryRoot,
      nonce,
      async execute() {
        return { stdout: JSON.stringify(gemmaReceipt()), stderr: "" };
      },
    });
    await writeFile(temporaryConfig, '{"fixture":2}\n');
    await assert.rejects(
      createGateEvidenceRecord({
        receipt,
        runtimeLock,
        fixtureRoot: fixture.root,
        producerRoot: repositoryRoot,
        drowseRevision: revision,
        now,
      }),
      /config binding is invalid/,
    );
  } finally {
    await unlink(temporaryConfig).catch(() => undefined);
    await fixture.remove();
  }
});

test("recorder executes the registered producer with a fresh nonce", async () => {
  const fixture = await trackedFixture(
    "browser-runtime/fixtures/post-block-rank-one-v2.json",
  );
  let invocation;
  try {
    const receipt = await runReleaseEvidenceProducer({
      evidenceType: "gemmaTinyFp32Parity",
      configPath: producerConfig,
      repositoryRoot,
      nonce,
      async execute(command, args, options) {
        invocation = { command, args, options };
        return { stdout: JSON.stringify(gemmaReceipt()), stderr: "" };
      },
    });
    assert.equal(invocation.command, process.execPath);
    assert.deepEqual(invocation.args.slice(-2), ["--nonce", nonce]);
    assert.equal(invocation.options.cwd, repositoryRoot);
    assert.equal(Object.isFrozen(receipt), true);
    assert.equal(Object.isFrozen(receipt.results), true);
    const record = await createGateEvidenceRecord({
      receipt,
      runtimeLock,
      fixtureRoot: fixture.root,
      producerRoot: repositoryRoot,
      drowseRevision: revision,
      now,
    });
    assert.equal(record.evidenceType, "gemmaTinyFp32Parity");
    assert.equal(record.schemaVersion, 4);
    assert.equal(record.producerId, "gemma-tiny-fp32-parity-v1");
    assert.equal(
      record.producerConfigPath,
      "browser-runtime/evidence-configs/fitting-kernel-parity.json",
    );
    assert.match(record.producerConfigSha256, /^[0-9a-f]{64}$/);
    assert.equal(record.producedAt, producedAt);
    assert.match(record.toolVersion, /^gemma-tiny-fp32-parity-v1\+sha256\./);
    assert.equal(
      record.results.fixturePath,
      "browser-runtime/fixtures/post-block-rank-one-v2.json",
    );
    assert.match(record.results.fixtureSha256, /^[0-9a-f]{64}$/);
  } finally {
    await fixture.remove();
  }
});

test("caller-authored gate and benchmark objects cannot cross the recorder boundary", async () => {
  await assert.rejects(
    createGateEvidenceRecord({
      receipt: gemmaReceipt(),
      runtimeLock,
      fixtureRoot: repositoryRoot,
      drowseRevision: revision,
      now,
    }),
    /must come from an executed registered producer/,
  );
  await assert.rejects(
    createBenchmarkRunRecord({
      receipt: benchmarkReceipt(),
      runtimeLock,
      drowseRevision: revision,
      now,
    }),
    /must come from an executed registered producer/,
  );
});

test("producer receipts are exact, nonce-bound, and pinned to their fixture", () => {
  const producer = requireReleaseEvidenceProducer("gemmaTinyFp32Parity");
  assert.throws(
    () => validateProducerReceipt(
      { ...gemmaReceipt(), nonce: "c".repeat(64) },
      producer,
      "gemmaTinyFp32Parity",
      nonce,
    ),
    /identity is invalid/,
  );
  assert.throws(
    () => validateProducerReceipt(
      { ...gemmaReceipt(), fixturePath: "LICENSE" },
      producer,
      "gemmaTinyFp32Parity",
      nonce,
    ),
    /fixture is not pinned/,
  );
  assert.throws(
    () => validateProducerReceipt(
      { ...gemmaReceipt(), callerClaim: true },
      producer,
      "gemmaTinyFp32Parity",
      nonce,
    ),
    /unknown or missing fields/,
  );
});

test("benchmark producer output is re-bound to the locked runtime identity", async () => {
  const receipt = await executedReceipt(
    "physicalBrowserBenchmark",
    benchmarkReceipt(),
  );
  const run = await createBenchmarkRunRecord({
    receipt,
    runtimeLock,
    drowseRevision: revision,
    now,
  });
  assert.equal(run.modelId, "gemma3-270m-instruct");
  assert.equal(run.drowseRevision, revision);
  assert.match(run.runtimeIdentitySha256, /^[0-9a-f]{64}$/);
  assert.match(run.contextBindingSha256, /^[0-9a-f]{64}$/);
  assert.equal(run.producerId, "physical-browser-benchmark-v1");
  assert.match(run.producerConfigSha256, /^[0-9a-f]{64}$/);
  assert.match(run.producerToolVersion, /^physical-browser-benchmark-v1\+sha256\./);
  await validateBenchmarkProducerBindings({ runs: [run] }, repositoryRoot);
  await assert.rejects(
    validateBenchmarkProducerBindings({
      runs: [{ ...run, producerConfigSha256: "0".repeat(64) }],
    }, repositoryRoot),
    /does not match checked-in bytes/,
  );

  const fallbackReceipt = await executedReceipt(
    "physicalBrowserBenchmark",
    benchmarkReceipt({ adapterIsFallback: true }),
  );
  await assert.rejects(
    createBenchmarkRunRecord({
      receipt: fallbackReceipt,
      runtimeLock,
      drowseRevision: revision,
      now,
    }),
    /confirmed hardware|adapterIsFallback|benchmark/i,
  );
});

test("configured producer can execute only a registered observer closure", async () => {
  const root = await mkdtemp(join(tmpdir(), "drowse-evidence-producer-"));
  const observer = "webui/scripts/browser-runtime-feasibility.mjs";
  const configPath = join(root, "config.json");
  await mkdir(join(root, "webui", "scripts"), { recursive: true });
  await writeFile(join(root, observer), "// fixture observer\n");
  await writeFile(configPath, JSON.stringify({
    $schema: "../evidence-config.schema.json",
    schemaVersion: 1,
    evidenceType: "gemmaTinyFp32Parity",
    observer,
    arguments: ["--fixture-mode"],
  }));
  try {
    let invocation;
    const receipt = await runConfiguredObserver({
      evidenceType: "gemmaTinyFp32Parity",
      configPath,
      nonce,
      root,
      async execute(command, args) {
        invocation = { command, args };
        const value = gemmaReceipt();
        delete value.producerId;
        delete value.nonce;
        return { stdout: JSON.stringify(value), stderr: "" };
      },
    });
    assert.equal(invocation.command, process.execPath);
    assert.equal(invocation.args.at(-1), "--fixture-mode");
    assert.equal(receipt.producerId, "gemma-tiny-fp32-parity-v1");

    await writeFile(configPath, JSON.stringify({
      $schema: "../evidence-config.schema.json",
      schemaVersion: 1,
      evidenceType: "gemmaTinyFp32Parity",
      observer: "webui/scripts/unregistered.mjs",
      arguments: [],
    }));
    await assert.rejects(
      runConfiguredObserver({
        evidenceType: "gemmaTinyFp32Parity",
        configPath,
        nonce,
        root,
        async execute() {
          throw new Error("must not execute");
        },
      }),
      /registered source closure/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("configured observers receive only the explicit execution environment", () => {
  assert.deepEqual(
    observerEnvironment({
      PATH: "/tools/bin",
      HOME: "/tmp/home",
      DISPLAY: ":1",
      NODE_OPTIONS: "--import=/tmp/inject.mjs",
      PYTHONPATH: "/tmp/inject",
      DROWSE_HOME: "/tmp/poisoned",
      HF_TOKEN: "secret",
    }),
    {
      PATH: "/tools/bin",
      HOME: "/tmp/home",
      DISPLAY: ":1",
    },
  );
});

test("the real producer executable fails closed on an observer without structured output", async () => {
  const temporary = await mkdtemp(join(repositoryRoot, ".evidence-contract-"));
  const config = join(temporary, "config.json");
  await writeFile(config, JSON.stringify({
    $schema: "../evidence-config.schema.json",
    schemaVersion: 1,
    evidenceType: "physicalBrowserBenchmark",
    observer: "webui/scripts/browser-full-runtime-contract.mjs",
    arguments: [],
  }));
  try {
    assert.throws(
      () => execFileSync(
        process.execPath,
        [
          resolve(repositoryRoot, "webui/scripts/hosted-release-evidence-producer.mjs"),
          "--evidence-type",
          "physicalBrowserBenchmark",
          "--config",
          config,
          "--nonce",
          nonce,
        ],
        { cwd: repositoryRoot, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] },
      ),
      (error) => /did not return one JSON report/.test(error.stderr),
    );
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

test("the real producer executable derives a benchmark receipt from full-runtime output", async () => {
  const root = await mkdtemp(join(tmpdir(), "drowse-evidence-full-runtime-"));
  const scripts = join(root, "webui", "scripts");
  const config = join(root, "config.json");
  await mkdir(scripts, { recursive: true });
  await Promise.all([
    writeFile(
      join(scripts, "hosted-release-evidence-producer.mjs"),
      await readFile(resolve(repositoryRoot, "webui/scripts/hosted-release-evidence-producer.mjs")),
    ),
    writeFile(
      join(scripts, "release-evidence-producers.mjs"),
      await readFile(resolve(repositoryRoot, "webui/scripts/release-evidence-producers.mjs")),
    ),
    writeFile(
      join(scripts, "browser-full-runtime.mjs"),
      `process.stdout.write(${JSON.stringify(`${JSON.stringify(fullRuntimeReport())}\n`)});\n`,
    ),
    writeFile(config, JSON.stringify({
      $schema: "../evidence-config.schema.json",
      schemaVersion: 1,
      evidenceType: "physicalBrowserBenchmark",
      observer: "webui/scripts/browser-full-runtime.mjs",
      arguments: ["--physical-fixture"],
    })),
  ]);
  try {
    const stdout = execFileSync(
      process.execPath,
      [
        join(scripts, "hosted-release-evidence-producer.mjs"),
        "--evidence-type",
        "physicalBrowserBenchmark",
        "--config",
        config,
        "--nonce",
        nonce,
      ],
      { cwd: root, encoding: "utf8" },
    );
    const receipt = JSON.parse(stdout);
    assert.equal(receipt.producerId, "physical-browser-benchmark-v1");
    assert.equal(receipt.nonce, nonce);
    assert.equal(receipt.run.modelId, "gemma3-270m-instruct");
    assert.equal(receipt.run.platform, "macos");
    assert.equal(receipt.run.browser, "chrome");
    assert.equal(receipt.run.deviceLabel, "apple metal-3");
    assert.equal(receipt.run.prefillTokensPerSecond, 13.8);
    assert.deepEqual(receipt.run.adapterFeatures, ["shader-f16"]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("full-runtime benchmark adapter rejects malformed reports separately", () => {
  const producer = requireReleaseEvidenceProducer("physicalBrowserBenchmark");
  const receipt = adaptObserverReports({
    evidenceType: "physicalBrowserBenchmark",
    nonce,
    producer,
    observer: "webui/scripts/browser-full-runtime.mjs",
    reports: [fullRuntimeReport()],
  });
  assert.equal(receipt.run.modelId, "gemma3-270m-instruct");
  assert.equal(receipt.run.platform, "macos");
  assert.equal(receipt.run.browser, "chrome");
  assert.equal(receipt.run.deviceLabel, "apple metal-3");
  assert.equal(receipt.run.prefillTokensPerSecond, 13.8);
  assert.deepEqual(receipt.run.adapterFeatures, ["shader-f16"]);
  assert.throws(
    () => adaptObserverReports({
      evidenceType: "physicalBrowserBenchmark",
      nonce,
      producer,
      observer: "webui/scripts/browser-full-runtime.mjs",
      reports: [{
        ...fullRuntimeReport(),
        environment: {
          ...fullRuntimeReport().environment,
          browserFlags: "compatibility",
        },
      }],
    }),
    /unsafe compatibility browser flags/,
  );
  const inconsistent = fullRuntimeReport();
  inconsistent.checks.local_artifacts.measurements.webLlmSha256 = "9".repeat(64);
  assert.throws(
    () => adaptObserverReports({
      evidenceType: "physicalBrowserBenchmark",
      nonce,
      producer,
      observer: "webui/scripts/browser-full-runtime.mjs",
      reports: [inconsistent],
    }),
    /disagree with the harness/,
  );
  assert.throws(
    () => adaptObserverReports({
      evidenceType: "physicalBrowserBenchmark",
      nonce,
      producer,
      observer: "webui/scripts/browser-full-runtime.mjs",
      reports: [{ ...fullRuntimeReport(), checks: null }],
    }),
    /malformed full-runtime report/,
  );
});

test("producer CLI parser is exact", () => {
  assert.deepEqual(
    parseProducerArguments([
      "--evidence-type",
      "runtimeLifecycle",
      "--config",
      "browser-runtime/evidence-configs/runtime.json",
      "--nonce",
      nonce,
    ]),
    {
      evidenceType: "runtimeLifecycle",
      config: "browser-runtime/evidence-configs/runtime.json",
      nonce,
    },
  );
  assert.throws(
    () => parseProducerArguments([
      "--evidence-type",
      "runtimeLifecycle",
      "--config",
      "config.json",
      "--nonce",
      "bad",
    ]),
    /nonce/,
  );
});

test("writes deterministic JSON once and rejects paths outside the evidence root", async () => {
  const root = await mkdtemp(join(tmpdir(), "drowse-evidence-output-"));
  const evidence = join(root, "evidence");
  await mkdir(evidence);
  try {
    const output = join(evidence, "runtime-qwen.json");
    const first = await writeNewJson(output, { z: 1, a: { d: 2, b: 1 } }, evidence);
    assert.match(first.sha256, /^[0-9a-f]{64}$/);
    assert.equal(
      await readFile(output, "utf8"),
      '{\n  "a": {\n    "b": 1,\n    "d": 2\n  },\n  "z": 1\n}\n',
    );
    await assert.rejects(writeNewJson(output, { a: 1 }, evidence), /already exists/);
    assert.throws(
      () => resolveEvidenceOutput(join(root, "outside.json"), evidence),
      /under browser-runtime\/evidence/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("recorder CLI removes arbitrary measurement input and keeps worktree policy fail-closed", () => {
  const parsed = parseArguments([
    "gate",
    "--evidence-type",
    "qwenTinyFp32Parity",
    "--producer-config",
    "browser-runtime/evidence-configs/qwen-tiny.json",
    "--output",
    "browser-runtime/evidence/qwen.json",
  ]);
  assert.equal(parsed.kind, "gate");
  assert.equal(parsed.evidenceType, "qwenTinyFp32Parity");
  assert.throws(
    () => parseArguments([
      "gate",
      "--measurement",
      "measurement.json",
      "--output",
      "x.json",
    ]),
    /unexpected argument --measurement/,
  );
  assert.throws(
    () => parseArguments([
      "benchmark",
      "--evidence-type",
      "qwenTinyFp32Parity",
      "--producer-config",
      "config.json",
      "--output",
      "x.json",
    ]),
    /is a gate producer/,
  );
  assert.deepEqual(
    unpermittedWorktreeEntries(
      "?? browser-runtime/evidence/qwen.json\n M webui/src/App.svelte\n?? notes.txt\n",
    ),
    [" M webui/src/App.svelte", "?? notes.txt"],
  );
});

test("attestation verifier accepts only this repository's CI signer", () => {
  assert.deepEqual(
    parseAttestationArguments([
      "--repository",
      "a9lim/drowse",
      "--signer-workflow",
      "a9lim/drowse/.github/workflows/ci.yml",
    ]),
    {
      repository: "a9lim/drowse",
      signerWorkflow: "a9lim/drowse/.github/workflows/ci.yml",
    },
  );
  assert.throws(
    () => parseAttestationArguments([
      "--repository",
      "a9lim/drowse",
      "--signer-workflow",
      "attacker/fork/.github/workflows/ci.yml",
    ]),
    /must be a9lim\/drowse\/\.github\/workflows\/ci\.yml/,
  );
});

test("CI attests producer output and verifies every referenced evidence subject", async () => {
  const workflow = await readFile(resolve(repositoryRoot, ".github/workflows/ci.yml"), "utf8");
  assert.match(workflow, /release-evidence-trust:/);
  assert.match(
    workflow,
    /release-evidence-trust:[\s\S]*?permissions:\s+attestations: read\s+contents: read/,
  );
  assert.match(workflow, /verify-hosted-evidence-attestations\.mjs/);
  assert.match(workflow, /--signer-workflow \"\$DROWSE_ATTESTATION_SIGNER\"/);
  assert.match(workflow, /attest-local-release-evidence:/);
  assert.match(workflow, /environment: release-evidence/);
  assert.match(workflow, /attestations: write/);
  assert.match(workflow, /id-token: write/);
  assert.match(
    workflow,
    /actions\/attest@508db95dd578ae2727ebd6217d5ba78e4fbda05d/,
  );
  assert.match(
    workflow,
    /actions\/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02/,
  );
  assert.match(workflow, /subject-path: \$\{\{ steps\.evidence\.outputs\.path \}\}/);
  assert.match(workflow, /\$\{\{ steps\.attest\.outputs\.bundle-path \}\}/);
});

test("durable evidence verification covers referenced gates and canonical benchmark rows", async () => {
  const root = await attestationFixture();
  const calls = [];
  try {
    const subjects = await collectEvidenceAttestationSubjects(root.path);
    assert.deepEqual(
      subjects.map(({ kind, evidenceType, sourceDigest }) => ({
        kind,
        evidenceType,
        sourceDigest,
      })),
      [
        {
          kind: "gate",
          evidenceType: "fittingKernelParity",
          sourceDigest: revision,
        },
        {
          kind: "benchmark",
          evidenceType: "physicalBrowserBenchmark",
          sourceDigest: revision,
        },
      ],
    );
    const verified = await verifyEvidenceAttestations({
      repository: "a9lim/drowse",
      signerWorkflow: "a9lim/drowse/.github/workflows/ci.yml",
      repositoryRoot: root.path,
      async execute(command, args) {
        calls.push({ command, args });
        return { stdout: '[{"verificationResult":{"verifiedTimestamps":[{}]}}]', stderr: "" };
      },
    });
    assert.equal(verified.length, 2);
    assert.equal(calls.length, 2);
    for (const call of calls) {
      assert.equal(call.command, "gh");
      assert.deepEqual(call.args.slice(0, 2), ["attestation", "verify"]);
      assert.ok(call.args.includes("a9lim/drowse/.github/workflows/ci.yml"));
      assert.ok(call.args.includes(revision));
      assert.ok(call.args.includes("https://slsa.dev/provenance/v1"));
    }
  } finally {
    await root.remove();
  }
});

test("durable evidence verification fails closed on digest drift or a missing attestation", async () => {
  const root = await attestationFixture();
  try {
    await writeFile(root.gatePath, '{"tampered":true}\n');
    await assert.rejects(
      collectEvidenceAttestationSubjects(root.path),
      /digest does not match/,
    );
  } finally {
    await root.remove();
  }

  const missing = await attestationFixture();
  try {
    await assert.rejects(
      verifyEvidenceAttestations({
        repository: "a9lim/drowse",
        signerWorkflow: "a9lim/drowse/.github/workflows/ci.yml",
        repositoryRoot: missing.path,
        async execute() {
          throw new Error("not found");
        },
      }),
      /has no valid CI release-evidence attestation/,
    );
  } finally {
    await missing.remove();
  }
});

async function executedReceipt(evidenceType, value) {
  return runReleaseEvidenceProducer({
    evidenceType,
    configPath: producerConfig,
    repositoryRoot,
    nonce,
    async execute() {
      return { stdout: JSON.stringify(value), stderr: "" };
    },
  });
}

async function attestationFixture() {
  const path = await mkdtemp(join(tmpdir(), "drowse-evidence-attestation-"));
  const evidenceRoot = join(path, "browser-runtime", "evidence");
  const gatePath = join(evidenceRoot, "fitting.json");
  await mkdir(evidenceRoot, { recursive: true });
  const gate = {
    evidenceType: "fittingKernelParity",
    drowseRevision: revision,
  };
  const gateBytes = `${JSON.stringify(gate)}\n`;
  await writeFile(gatePath, gateBytes);
  await writeFile(
    join(path, "browser-runtime", "runtime-feasibility-evidence.json"),
    JSON.stringify({ drowseRevision: null, gates: { runtimeLifecycle: null } }),
  );
  await writeFile(
    join(path, "browser-runtime", "authoring-evidence.json"),
    JSON.stringify({
      drowseRevision: revision,
      evidence: {
        fittingKernelParity: {
          path: "evidence/fitting.json",
          sha256: createHash("sha256").update(gateBytes).digest("hex"),
        },
      },
    }),
  );
  await writeFile(
    join(path, "browser-runtime", "benchmark-evidence.json"),
    JSON.stringify({
      drowseRevision: revision,
      runs: [{
        browser: "chrome",
        contextTokens: 2048,
        modelId: "gemma3-270m-instruct",
        drowseRevision: revision,
      }],
    }),
  );
  return {
    path,
    gatePath,
    remove: () => rm(path, { recursive: true, force: true }),
  };
}

function gemmaReceipt() {
  return {
    schemaVersion: 1,
    evidenceType: "gemmaTinyFp32Parity",
    producerId: "gemma-tiny-fp32-parity-v1",
    nonce,
    producedAt,
    passed: true,
    fixturePath: "browser-runtime/fixtures/post-block-rank-one-v2.json",
    results: {
      architecture: "gemma3_text",
      maxAbsoluteLogitError: 0.00001,
      maxRelativeLogitError: 0.0001,
      maxAbsoluteCaptureError: 0.00001,
      maxRelativeCaptureError: 0.0001,
      probePassed: true,
      steeringPassed: true,
    },
  };
}

function benchmarkReceipt(overrides = {}) {
  const model = runtimeLock.models.find((entry) => entry.id === "gemma3-270m-instruct");
  return {
    schemaVersion: 1,
    evidenceType: "physicalBrowserBenchmark",
    producerId: "physical-browser-benchmark-v1",
    nonce,
    producedAt,
    passed: true,
    run: {
      modelId: "gemma3-270m-instruct",
      contextTokens: 2048,
      platform: "macos",
      browser: "chrome",
      browserVersion: "140.0.0.0",
      browserFlags: "normal",
      deviceLabel: "M1 Max 32 GB",
      runtimeIdentitySha256: lockedRuntimeIdentitySha256(runtimeLock, model),
      webLlmSha256: "1".repeat(64),
      modelArtifactManifestSha256: "2".repeat(64),
      convertedManifestSha256: model.manifestSha256,
      modelLibrarySha256: model.librarySha256,
      coreArtifactManifestSha256: "3".repeat(64),
      jlensArtifactManifestSha256: "4".repeat(64),
      saeArtifactManifestSha256: "5".repeat(64),
      adapterIsFallback: false,
      calibrationScore: 10.5,
      deviceMemoryGiB: 8,
      adapterFeatures: ["shader-f16"],
      adapterLimits: {
        maxBufferSize: 268435456,
        maxStorageBufferBindingSize: 134217728,
        maxStorageBuffersPerShaderStage: 10,
      },
      prefillTokensPerSecond: 24.5,
      decodeTokensPerSecond: 8.25,
      loadSucceeded: true,
      measuredAt: producedAt,
      ...overrides,
    },
  };
}

function fullRuntimeReport() {
  const model = runtimeLock.models.find((entry) => entry.id === "gemma3-270m-instruct");
  const observedArtifacts = {
    runtimeIdentitySha256: lockedRuntimeIdentitySha256(runtimeLock, model),
    webLlmSha256: "1".repeat(64),
    modelArtifactManifestSha256: "2".repeat(64),
    convertedManifestSha256: model.manifestSha256,
    modelLibrarySha256: model.librarySha256,
    coreArtifactManifestSha256: "3".repeat(64),
    jlensArtifactManifestSha256: "4".repeat(64),
    saeArtifactManifestSha256: "5".repeat(64),
  };
  const checks = Object.fromEntries(
    [
      "local_artifacts",
      "webgpu_adapter",
      "load",
      "ordinary_generation",
      "forced_replay",
      "explicit_capture",
      "flat_gated_instruments",
      "curved_fitting",
      "curved_generation",
      "stop",
      "unload_reload",
      "offline_reuse",
    ].map((name) => [name, { passed: true, measurements: {} }]),
  );
  checks.webgpu_adapter.measurements = {
    calibrationScore: 100,
    adapterFeatures: ["shader-f16"],
    adapterLimits: {
      maxStorageBufferBindingSize: 134217728,
      maxStorageBuffersPerShaderStage: 10,
      maxComputeWorkgroupSizeX: 256,
      maxComputeInvocationsPerWorkgroup: 256,
    },
    maxStorageBufferBindingSize: 134217728,
    maxStorageBuffersPerShaderStage: 10,
    maxComputeWorkgroupSizeX: 256,
    maxComputeInvocationsPerWorkgroup: 256,
  };
  checks.ordinary_generation.measurements = {
    prefillTokensPerSecond: 13.8,
    decodeTokensPerSecond: 42.1,
  };
  checks.local_artifacts.measurements = { ...observedArtifacts };
  return {
    $schema: "drowse-real-browser-e2e-measurement-v1",
    schemaVersion: 1,
    evidenceType: "realBrowserEndToEnd",
    toolVersion: "drowse-real-browser-e2e-v4",
    producedAt,
    passed: true,
    releaseEvidence: false,
    environment: {
      modelId: "gemma3-270m-instruct",
      contextTokens: 2048,
      userAgent: "Chrome",
      adapterFallback: "hardware",
      adapterInfo: { vendor: "apple", architecture: "metal-3" },
      deviceMemoryGiB: 32,
      browserChannel: "chrome",
      browserFlags: "normal",
      browserVersion: "151.0.0.0",
      platform: "darwin",
      runtimeIdentitySha256: observedArtifacts.runtimeIdentitySha256,
      observedArtifacts,
    },
    checks,
  };
}

async function trackedFixture(path) {
  const root = await mkdtemp(join(tmpdir(), "drowse-evidence-fixture-"));
  const absolute = join(root, ...path.split("/"));
  await mkdir(resolve(absolute, ".."), { recursive: true });
  await writeFile(absolute, '{"fixture":true}\n');
  execFileSync("git", ["init", "-q"], { cwd: root });
  execFileSync("git", ["config", "user.email", "tests@drowse.invalid"], {
    cwd: root,
  });
  execFileSync("git", ["config", "user.name", "Drowse tests"], { cwd: root });
  execFileSync("git", ["add", path], { cwd: root });
  execFileSync("git", ["commit", "-qm", "fixture"], { cwd: root });
  return {
    root,
    remove: () => rm(root, { recursive: true, force: true }),
  };
}
