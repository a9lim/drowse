import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { deflateSync } from "fflate";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

const tests = [];
const test = (name, run) => tests.push({ name, run });
const text = (value) => new TextEncoder().encode(value);
const digest = (value) => createHash("sha256").update(value).digest("hex");

try {
  const {
    browserFittedCurvedDiscoverPack,
    browserFittedFlatDiscoverPack,
    buildDrowseArchive,
    decodeFp32Safetensors,
    encodeFp32Safetensors,
    inspectDrowseArchive,
    parseDrowseTensorFilename,
    readVerifiedDrowseArchiveFile,
    DROWSE_ARCHIVE_IN_MEMORY_BUILD_MAX_BYTES,
    DrowseArchiveError,
    validateDrowseArchive,
    verifyDrowseArchive,
    writeFp32Safetensors,
  } = await server.ssrLoadModule("/src/hosted/artifacts/index.ts");

  test("publishes a self-validated browser flat auto fit with its DLS selection map", async () => {
    const evaluated = {
      operation: "affine_fisher",
      nodeCount: 3,
      columns: 2,
      components: 1,
      centroidMean: new Float64Array([0, 0]),
      mean: new Float64Array([0, 0]),
      basis: new Float64Array([1, 0]),
      nodeCoordinates: new Float64Array([-1, 0, 1]),
      muCoordinates: new Float64Array([-1, 0, 1]),
      whitenedGram: new Float64Array([
        1, 0, 0,
        0, 4, 0,
        0, 0, 9,
      ]),
      neutralCrossGram: new Float64Array(3),
      explainedVariance: 1,
      mahalanobisShare: 2,
    };
    const pack = await browserFittedFlatDiscoverPack({
      manifold: {
        namespace: "local",
        name: "browser_fit",
        description: "browser fitted fixture",
        fitMode: "auto",
        hyperparams: { max_dim: 2 },
        nodes: ["calm", "alert", "focused"].map((label) => ({
          label,
          statements: [label],
          role: null,
          kind: "abstract",
        })),
      },
      closure: {
        source: { uri: "local", repository: null, revision: null },
        tags: ["browser-fit"],
        template: null,
      },
      modelId: "test/model",
      producerVersion: "browser-test",
      identity: {
        runtimeIdentitySha256: "a".repeat(64),
        contextBindingSha256: "9".repeat(64),
        modelSourceFingerprint: "b".repeat(64),
        captureSha256: "c".repeat(64),
        captureVersion: 1,
        fitPolicyVersion: 1,
      },
      topology: {
        operation: "topology",
        winnerName: "flat-pca",
        fitMode: "pca",
        intrinsicDimensions: 2,
        periodicDimensions: 0,
        persistentLoops: 0,
        usedFaintCycle: false,
        coordinates: new Float64Array([-1, 0, 0, 1, 1, 0]),
        embeddedCoordinates: new Float64Array([-1, 0, 0, 1, 1, 0]),
        candidates: [{
          name: "flat-pca",
          fitMode: "pca",
          dimensions: 2,
          score: 0.5,
          viable: true,
          reason: "",
        }, {
          name: "spectral",
          fitMode: "spectral",
          dimensions: 2,
          score: null,
          viable: false,
          reason: "poisedness floor exceeds node count",
        }],
        diagnostics: {
          kind: "pca",
          perComponentVariance: new Float64Array([0.6, 0.4]),
          cumulativeVariance: new Float64Array([0.6, 1]),
          pickedDimensions: 2,
          threshold: 0.7,
        },
        winnerPlan: null,
      },
      consensusGram: new Float64Array([
        1, 0, -1,
        0, 1, 0,
        -1, 0, 1,
      ]),
      nodeCoordinates: new Float64Array([-1, 0, 0, 1, 1, 0]),
      evaluatedLayers: new Map([[0, evaluated]]),
      layers: new Map([[0, { ...evaluated, affineMap: new Float64Array([1, 0]) }]]),
    });
    const verified = await validateDrowseArchive(pack);
    assert.equal(verified.manifold.name, "browser_fit");
    assert.equal(verified.fittedArtifacts[0].modelFingerprint, "a".repeat(64));
    assert.equal(verified.fittedArtifacts[0].contextBindingSha256, "9".repeat(64));
    const sidecarPath = verified.fittedArtifacts[0].sidecarPath;
    const sidecar = JSON.parse(new TextDecoder().decode(
      await readVerifiedDrowseArchiveFile(verified, sidecarPath),
    ));
    assert.equal(sidecar.resolved_fit_mode, "pca");
    assert.equal(sidecar.topology_candidates[1].score, null);
    assert.deepEqual(sidecar.node_spread_per_layer, { 0: 14 });
    assert.equal(sidecar.diagnostics.picked_k, 2);
    assert.equal(sidecar.context_binding_sha256, "9".repeat(64));
    assert.deepEqual(verified.manifold.tags, ["browser-fit"]);
    const tensorPath = verified.fittedArtifacts[0].tensorPath;
    const tensor = decodeFp32Safetensors(
      await readVerifiedDrowseArchiveFile(verified, tensorPath),
      tensorPath,
    );
    assert.deepEqual(tensor.tensor("layer_0.affine_map", [2, 1]).data, new Float32Array([1, 0]));
  });

  test("publishes a self-validated browser curved fit with sigma and origin geometry", async () => {
    const surface = {
      inputDimensions: 1,
      outputDimensions: 2,
      nodeCount: 3,
      lambda: 0,
      effectiveDegreesOfFreedom: 3,
      gcv: -1,
      nodes: new Float64Array([-1, 0, 1]),
      coordinateOffset: new Float64Array([0]),
      coordinateScale: new Float64Array([1]),
      weights: new Float64Array(6),
      polynomial: new Float64Array([0, 0, 1, 0]),
    };
    const evaluated = {
      operation: "affine_fisher",
      nodeCount: 3,
      columns: 2,
      components: 2,
      centroidMean: new Float64Array([0, 0]),
      mean: new Float64Array([0, 0]),
      basis: new Float64Array([1, 0, 0, 1]),
      nodeCoordinates: new Float64Array([-1, 0, 0, 0, 1, 0]),
      muCoordinates: new Float64Array([-1, 0, 0, 0, 1, 0]),
      whitenedGram: new Float64Array([
        1, 0, 0,
        0, 4, 0,
        0, 0, 9,
      ]),
      neutralCrossGram: new Float64Array(3),
      explainedVariance: 1,
      mahalanobisShare: 2,
    };
    const pack = await browserFittedCurvedDiscoverPack({
      manifold: {
        namespace: "local",
        name: "browser_curve",
        description: "browser curved fixture",
        fitMode: "auto",
        hyperparams: { max_dim: 1 },
        nodes: ["calm", "alert", "focused"].map((label) => ({
          label,
          statements: [label],
          role: null,
          kind: "abstract",
        })),
      },
      closure: {
        source: { uri: "local", repository: null, revision: null },
        tags: [],
        template: null,
      },
      modelId: "test/model",
      producerVersion: "browser-test",
      identity: {
        runtimeIdentitySha256: "d".repeat(64),
        contextBindingSha256: "8".repeat(64),
        modelSourceFingerprint: "e".repeat(64),
        captureSha256: "f".repeat(64),
        captureVersion: 1,
        fitPolicyVersion: 1,
      },
      topology: {
        operation: "topology",
        winnerName: "spectral",
        fitMode: "spectral",
        intrinsicDimensions: 1,
        periodicDimensions: 0,
        persistentLoops: 0,
        usedFaintCycle: false,
        coordinates: new Float64Array([-1, 0, 1]),
        embeddedCoordinates: new Float64Array([-1, 0, 1]),
        candidates: [{
          name: "spectral",
          fitMode: "spectral",
          dimensions: 1,
          score: 0.1,
          viable: true,
          reason: "",
        }],
        diagnostics: {
          kind: "spectral",
          eigenvalues: new Float64Array([0.1, 0.7]),
          pickedDimensions: 1,
          gapMagnitude: 0.6,
          bandwidth: 1,
          kNn: 2,
          componentCount: 1,
          heuristicDimensions: 1,
          minDimensions: null,
          pinned: false,
        },
        winnerPlan: null,
      },
      consensusGram: new Float64Array([
        1, 0, -1,
        0, 1, 0,
        -1, 0, 1,
      ]),
      evaluatedLayers: new Map([[2, evaluated]]),
      layers: new Map([[2, {
        ...evaluated,
        surface,
        sigmaSurface: {
          ...surface,
          outputDimensions: 1,
          weights: new Float64Array(3),
          polynomial: new Float64Array([0, 0]),
        },
        sigmaSummary: { mean: 1, min: 1, max: 1, lambda: 0 },
        origin: new Float64Array([0]),
        originDistance: 0,
      }]]),
    });
    const verified = await validateDrowseArchive(pack);
    const fitted = verified.fittedArtifacts[0];
    const sidecar = JSON.parse(new TextDecoder().decode(
      await readVerifiedDrowseArchiveFile(verified, fitted.sidecarPath),
    ));
    assert.equal(sidecar.resolved_fit_mode, "spectral");
    assert.equal(sidecar.context_binding_sha256, "8".repeat(64));
    assert.deepEqual(sidecar.node_spread_per_layer, { 2: 14 });
    assert.deepEqual(sidecar.rbf_smoothing_per_layer[2], {
      lambda: 0,
      edf: 3,
      gcv: -1,
    });
    assert.deepEqual(sidecar.origin_per_layer, { 2: [0] });
    assert.deepEqual(sidecar.sigma_field_per_layer[2], {
      sigma_mean: 1,
      sigma_min: 1,
      sigma_max: 1,
      lambda: 0,
    });
    const tensor = decodeFp32Safetensors(
      await readVerifiedDrowseArchiveFile(verified, fitted.tensorPath),
      fitted.tensorPath,
    );
    assert.deepEqual(tensor.tensor("layer_2.sigma_rbf_weights", [3, 1]).data, new Float32Array(3));
  });

  test("preserves immutable source, tags, and the exact referenced template closure", async () => {
    const revision = "7".repeat(40);
    const source = {
      uri: `hf://owner/browser-manifold@${revision}`,
      repository: "owner/browser-manifold",
      revision,
    };
    const template = {
      format_version: 2,
      name: "weekday",
      slot: "[DAY]",
      values: ["Monday", "Tuesday"],
      contexts: [{
        turns: [{ role: "user", content: "What day is it?" }],
        assistant: "Today is [DAY]",
      }],
      description: "weekday fixture",
      source: source.uri,
      tags: ["calendar"],
    };
    const evaluated = {
      operation: "affine_fisher",
      nodeCount: 2,
      columns: 2,
      components: 1,
      centroidMean: new Float64Array(2),
      mean: new Float64Array(2),
      basis: new Float64Array([1, 0]),
      nodeCoordinates: new Float64Array([-1, 1]),
      muCoordinates: new Float64Array([-1, 1]),
      whitenedGram: new Float64Array([1, -1, -1, 1]),
      neutralCrossGram: new Float64Array(2),
      explainedVariance: 1,
      mahalanobisShare: 2,
    };
    const pack = await browserFittedFlatDiscoverPack({
      manifold: {
        namespace: "local",
        name: "weekday_fit",
        description: "templated browser fit",
        fitMode: "pca",
        hyperparams: { max_dim: 1 },
        nodes: [
          { label: "monday", statements: ["Today is Monday"], role: null, kind: "custom" },
          { label: "tuesday", statements: ["Today is Tuesday"], role: null, kind: "custom" },
        ],
      },
      closure: {
        source,
        tags: ["calendar", "browser"],
        template: { reference: "local/weekday", payload: template },
      },
      modelId: "test/model",
      producerVersion: "browser-test",
      identity: {
        runtimeIdentitySha256: "1".repeat(64),
        contextBindingSha256: "2".repeat(64),
        modelSourceFingerprint: "3".repeat(64),
        captureSha256: "4".repeat(64),
        captureVersion: 3,
        captureRenderSha256: "5".repeat(64),
        baselinePromptsSha256: "6".repeat(64),
        fitPolicyVersion: 1,
      },
      topology: {
        operation: "topology",
        winnerName: "flat-pca",
        fitMode: "pca",
        intrinsicDimensions: 1,
        periodicDimensions: 0,
        persistentLoops: 0,
        usedFaintCycle: false,
        coordinates: new Float64Array([-1, 1]),
        embeddedCoordinates: new Float64Array([-1, 1]),
        candidates: [],
        diagnostics: {
          kind: "pca",
          perComponentVariance: new Float64Array([1]),
          cumulativeVariance: new Float64Array([1]),
          pickedDimensions: 1,
          threshold: 0.7,
        },
        winnerPlan: null,
      },
      consensusGram: new Float64Array([1, -1, -1, 1]),
      nodeCoordinates: new Float64Array([-1, 1]),
      evaluatedLayers: new Map([[0, evaluated]]),
      layers: new Map([[0, { ...evaluated, affineMap: null }]]),
    });
    const verified = await validateDrowseArchive(pack);
    assert.deepEqual(verified.manifest.source, source);
    assert.equal(verified.manifold.source, source.uri);
    assert.deepEqual(verified.manifold.tags, ["calendar", "browser"]);
    assert.equal(verified.manifold.template_ref, "local/weekday");
    assert.equal(verified.template.name, "weekday");
    assert.deepEqual(verified.template.tags, ["calendar"]);
    assert.equal(verified.fittedArtifacts[0].contextBindingSha256, "2".repeat(64));
  });

  test("builds deterministic self-validated transport envelopes", async () => {
    const fixture = manifoldFixture({ template: true });
    const input = {
      primary: fixture.manifest.primary,
      template: fixture.manifest.template,
      producerVersion: "browser-test",
      source: fixture.manifest.source,
      files: fixture.payloads,
    };
    const first = await buildDrowseArchive(input);
    const second = await buildDrowseArchive(input);
    assert.deepEqual(
      new Uint8Array(await first.arrayBuffer()),
      new Uint8Array(await second.arrayBuffer()),
    );
    const verified = await validateDrowseArchive(first);
    assert.equal(verified.manifest.producer.version, "browser-test");
    assert.equal(verified.template.name, "weekday");

    await assert.rejects(
      buildDrowseArchive({ ...input, primary: "manifolds/local/wrong" }),
      (error) => error.code === "MANIFEST_MISMATCH" && /outside/.test(error.message),
    );
  });

  test("round-trips the shared manifold closure across Python and browser codecs", async () => {
    const fixtureUrl = new URL(
      "../../browser-runtime/fixtures/drowse-interchange-v1.json",
      import.meta.url,
    );
    const fixture = JSON.parse(await readFile(fixtureUrl, "utf8"));
    const temporary = await mkdtemp(join(tmpdir(), "drowse-archive-interchange-"));
    try {
      const pythonArchive = join(temporary, "python.drowse");
      const pythonSummary = runPythonPackBridge(
        "export",
        fileURLToPath(fixtureUrl),
        join(temporary, "python-export-home"),
        pythonArchive,
      );
      assert.deepEqual(pythonSummary, fixture.expected);

      const pythonPack = await validateDrowseArchive(
        new Uint8Array(await readFile(pythonArchive)),
      );
      const pythonNodeGroups = {};
      const pythonNodeFiles = Object.keys(pythonPack.files)
        .filter((path) => path.startsWith(`${pythonPack.manifest.primary}/nodes/`))
        .sort();
      for (const [index, path] of pythonNodeFiles.entries()) {
        pythonNodeGroups[fixture.nodes[index].label] = JSON.parse(
          new TextDecoder().decode(await readVerifiedDrowseArchiveFile(pythonPack, path)),
        );
      }
      assert.deepEqual({
        primary: pythonPack.manifest.primary,
        template: pythonPack.manifest.template,
        name: pythonPack.manifold.name,
        fit_mode: pythonPack.manifold.fit_mode,
        node_labels: pythonPack.manifold.nodes.map((node) => node.label),
        node_groups: pythonNodeGroups,
        hyperparams: { ...pythonPack.manifold.hyperparams },
        source: pythonPack.manifold.source,
        tags: pythonPack.manifold.tags,
        template_ref: pythonPack.manifold.template_ref,
      }, fixture.expected);

      const primary = fixture.expected.primary;
      const browserPayloads = {
        [`${primary}/manifold.json`]: json({
          format_version: 10,
          name: fixture.name,
          description: fixture.description,
          fit_mode: fixture.fitMode,
          hyperparams: fixture.hyperparams,
          nodes: fixture.nodes.map(({ label, role, kind }) => ({ label, role, kind })),
          files: {},
          source: fixture.source.uri,
          tags: fixture.expected.tags,
          template_ref: fixture.expected.template_ref,
        }),
      };
      fixture.nodes.forEach((node, index) => {
        browserPayloads[
          `${primary}/nodes/${String(index).padStart(2, "0")}_${node.label}.json`
        ] = json(node.corpus);
      });
      const browserPack = await buildDrowseArchive({
        primary,
        template: null,
        producerVersion: fixture.producerVersion,
        source: fixture.source,
        files: browserPayloads,
      });
      const browserArchive = join(temporary, "browser.drowse");
      const browserBytes = new Uint8Array(await browserPack.arrayBuffer());
      await writeFile(browserArchive, browserBytes);
      const installedSummary = runPythonPackBridge(
        "install",
        fileURLToPath(fixtureUrl),
        join(temporary, "python-install-home"),
        browserArchive,
      );
      assert.deepEqual(installedSummary, fixture.expected);
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  });

  test("decodes canonical fitted tensor identities for browser inventories", () => {
    assert.deepEqual(
      parseDrowseTensorFilename("_zdGVzdC9tb2RlbA.safetensors"),
      {
        modelId: "test/model",
        safeModelId: "_zdGVzdC9tb2RlbA",
        variant: "raw",
        variantIdentity: null,
        safeVariantIdentity: null,
      },
    );
    assert.equal(parseDrowseTensorFilename("not-canonical.safetensors"), null);
  });

  test("bounds in-memory authoring before allocating a ZIP archive", async () => {
    const fixture = manifoldFixture();
    await assert.rejects(
      buildDrowseArchive({
        primary: fixture.manifest.primary,
        template: null,
        producerVersion: "browser-test",
        source: fixture.manifest.source,
        files: {
          ...fixture.payloads,
          "manifolds/local/demo/oversized.bin": new Uint8Array(
            DROWSE_ARCHIVE_IN_MEMORY_BUILD_MAX_BYTES + 1,
          ),
        },
      }),
      (error) => error.code === "ARCHIVE_LIMIT_EXCEEDED" && /64 MiB/.test(error.message),
    );
  });

  test("validates stored and deflated manifold closures without rewriting bytes", async () => {
    for (const method of [0, 8]) {
      const fixture = manifoldFixture({ method });
      const writes = new Map();
      let committed = false;
      const verified = await validateDrowseArchive(fixture.archive, {
        stage: memoryStage(writes, () => { committed = true; }),
      });
      assert.equal(verified.manifest.primary, "manifolds/local/demo");
      assert.equal(verified.manifold.name, "demo");
      assert.equal(committed, true);
      for (const [path, original] of Object.entries(fixture.payloads)) {
        assert.deepEqual(writes.get(path), original, `${path} changed during staging`);
      }
    }
  });

  test("validates and normalizes a legacy-branded fitted archive", async () => {
    const fixture = fittedFixture({ legacyBrand: true });
    const verified = await validateDrowseArchive(fixture.archive);
    assert.equal(verified.manifest.kind, "drowse-manifold");
    assert.equal(verified.manifest.producer.name, "drowse");
    assert.equal(verified.fittedArtifacts.length, 1);
  });

  test("inspection is bounded and verification requires its opaque handle", async () => {
    const fixture = manifoldFixture();
    const inspected = await inspectDrowseArchive(new Blob([fixture.archive]));
    assert.equal(inspected.entries.length, 4);
    assert.equal(inspected.primaryIdentity.join("/"), "local/demo");
    await assert.rejects(
      verifyDrowseArchive(structuredClone(inspected)),
      (error) => error instanceof DrowseArchiveError && error.code === "ARCHIVE_INVALID",
    );
  });

  test("reads only checksum-bound files from a verified pack handle", async () => {
    const fixture = fittedFixture();
    const verified = await validateDrowseArchive(fixture.archive);
    const tensorPath = verified.fittedArtifacts[0].tensorPath;
    assert.deepEqual(
      await readVerifiedDrowseArchiveFile(verified, tensorPath),
      fixture.payloads[tensorPath],
    );
    await assert.rejects(
      readVerifiedDrowseArchiveFile(verified, `${verified.manifest.primary}/missing.bin`),
      (error) => error.code === "MANIFEST_MISMATCH",
    );
    await assert.rejects(
      readVerifiedDrowseArchiveFile({ ...verified }, tensorPath),
      (error) => error.code === "ARCHIVE_INVALID",
    );
  });

  test("validates the optional single template closure and derived corpora", async () => {
    const fixture = manifoldFixture({ template: true });
    const verified = await validateDrowseArchive(fixture.archive);
    assert.equal(verified.template.name, "weekday");
    assert.deepEqual(verified.templateIdentity, ["local", "weekday"]);

    const bad = manifoldFixture({ template: true, nodeCorpora: [["wrong"], ["today is Tuesday"]] });
    await assert.rejects(
      validateDrowseArchive(bad.archive),
      (error) => error.code === "CLOSURE_INVALID" && /does not match/.test(error.message),
    );
  });

  test("rejects duplicate JSON keys and any manifest schema drift", async () => {
    const fixture = manifoldFixture();
    const raw = new TextDecoder().decode(fixture.entries[0].data);
    const duplicate = raw.replace(
      '"kind":"drowse-manifold"',
      '"kind":"drowse-manifold","kind":"drowse-manifold"',
    );
    const duplicateArchive = makeZip([
      { name: "pack.json", data: text(duplicate) },
      ...fixture.entries.slice(1),
    ]);
    await assert.rejects(
      inspectDrowseArchive(duplicateArchive),
      (error) => error.code === "JSON_DUPLICATE_KEY",
    );
    const manifest = JSON.parse(raw);
    manifest.extra = true;
    const drift = rebuildWithManifest(fixture, manifest);
    await assert.rejects(
      inspectDrowseArchive(drift),
      (error) => error.code === "MANIFEST_INVALID",
    );
  });

  test("rejects traversal, absolute, drive, backslash, NUL, and alternate path names", async () => {
    for (const name of ["../escape", "/absolute", "C:/drive", "a\\b", new Uint8Array([0x61, 0, 0x62])]) {
      await assert.rejects(
        inspectDrowseArchive(makeZip([{ name, data: new Uint8Array() }])),
        (error) => error.code === "ARCHIVE_PATH_INVALID",
      );
    }
    await assert.rejects(
      inspectDrowseArchive(makeZip([{ name: "pack.json", localName: "other.json", data: text("{}") }])),
      (error) => error.code === "ARCHIVE_INVALID" && /mismatched/.test(error.message),
    );
    await assert.rejects(
      inspectDrowseArchive(makeZip([{ name: "pack.json", data: text("{}"), extra: extraField(0x7075, new Uint8Array([1])) }])),
      (error) => error.code === "ARCHIVE_INVALID" && /Unicode path/.test(error.message),
    );
  });

  test("rejects duplicates, case-fold collisions, unknown roots, and non-exact closures", async () => {
    await assert.rejects(
      inspectDrowseArchive(makeZip([
        { name: "pack.json", data: text("{}") },
        { name: "pack.json", data: text("{}") },
      ])),
      (error) => error.code === "ARCHIVE_PATH_COLLISION",
    );
    await assert.rejects(
      inspectDrowseArchive(makeZip([
        { name: "pack.json", data: text("{}") },
        { name: "PACK.JSON", data: text("{}") },
      ])),
      (error) => error.code === "ARCHIVE_PATH_COLLISION",
    );
    const fixture = manifoldFixture();
    const payloads = { ...fixture.payloads, "models/local/demo.bin": text("bad root") };
    const manifest = makeManifest(payloads);
    const archive = makeZip([
      { name: "pack.json", data: json(manifest) },
      ...Object.entries(payloads).map(([name, data]) => ({ name, data })),
    ]);
    await assert.rejects(
      inspectDrowseArchive(archive),
      (error) => error.code === "MANIFEST_MISMATCH" && /outside/.test(error.message),
    );
    const extraInner = manifoldFixture({ extraPayload: ["manifolds/local/demo/notes.txt", text("no") ] });
    await assert.rejects(
      validateDrowseArchive(extraInner.archive),
      (error) => error.code === "CLOSURE_INVALID" && /not exact/.test(error.message),
    );
  });

  test("rejects encrypted, linked, special, directory, and unsupported entries", async () => {
    const cases = [
      { flags: 1 },
      { flags: 0x2000 },
      { externalAttributes: 0xa1ff0000 },
      { externalAttributes: 0x21a40000 },
      { name: "pack.json/", externalAttributes: 0x41ed0010 },
      { method: 12 },
      { extra: extraField(0x9901, new Uint8Array([1])) },
    ];
    for (const overrides of cases) {
      await assert.rejects(
        inspectDrowseArchive(makeZip([{ name: "pack.json", data: text("{}"), ...overrides }])),
        (error) => [
          "ARCHIVE_ENTRY_UNSAFE",
          "ARCHIVE_COMPRESSION_UNSUPPORTED",
          "ARCHIVE_PATH_INVALID",
        ].includes(error.code),
      );
    }
  });

  test("enforces entry count, file, total expansion, and compression-ratio limits before extraction", async () => {
    const many = Array.from({ length: 4_097 }, (_, index) => ({ name: `f${index}`, data: new Uint8Array() }));
    await assert.rejects(inspectDrowseArchive(makeZip(many)), (error) => error.code === "ARCHIVE_LIMIT_EXCEEDED");
    await assert.rejects(
      inspectDrowseArchive(makeZip([{ name: "pack.json", data: new Uint8Array(), declaredSize: 512 * 1024 * 1024 + 1 }])),
      (error) => error.code === "ARCHIVE_LIMIT_EXCEEDED" && /512 MiB/.test(error.message),
    );
    const huge = Array.from({ length: 5 }, (_, index) => ({
      name: `f${index}`,
      data: new Uint8Array(),
      declaredSize: 512 * 1024 * 1024,
      declaredCompressedSize: 512 * 1024 * 1024,
    }));
    await assert.rejects(inspectDrowseArchive(makeZip(huge)), (error) => error.code === "ARCHIVE_LIMIT_EXCEEDED" && /2 GiB/.test(error.message));
    await assert.rejects(
      inspectDrowseArchive(makeZip([{ name: "pack.json", data: new Uint8Array([0]), declaredSize: 101 }])),
      (error) => error.code === "ARCHIVE_LIMIT_EXCEEDED" && /100:1/.test(error.message),
    );
  });

  test("checks exact payload sizes and SHA-256 before committing staged data", async () => {
    const fixture = manifoldFixture();
    const manifest = JSON.parse(new TextDecoder().decode(fixture.entries[0].data));
    const path = "manifolds/local/demo/nodes/00_calm.json";
    const changed = text('["evil"]\n');
    const entries = fixture.entries.map((entry) => entry.name === path ? { ...entry, data: changed } : entry);
    const events = [];
    await assert.rejects(
      validateDrowseArchive(makeZip(entries), {
        stage: {
          begin: () => events.push("begin"),
          write: () => {},
          finish: (name) => events.push(`finish:${name}`),
          commit: () => events.push("commit"),
          rollback: () => events.push("rollback"),
        },
      }),
      (error) => error.code === "CHECKSUM_MISMATCH",
    );
    assert.equal(manifest.files[path].sha256, digest(fixture.payloads[path]));
    assert.equal(events.includes("commit"), false);
    assert.equal(events.at(-1), "rollback");
  });

  test("rolls staging back when begin, semantic validation, or cancellation fails", async () => {
    const fixture = manifoldFixture();
    let rollbacks = 0;
    await assert.rejects(
      validateDrowseArchive(fixture.archive, {
        stage: {
          begin: () => { throw new Error("begin failed"); },
          write: () => {},
          finish: () => {},
          commit: () => {},
          rollback: () => { rollbacks += 1; },
        },
      }),
      /begin failed/,
    );
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(validateDrowseArchive(fixture.archive, { signal: controller.signal }), (error) => error.name === "AbortError");
    assert.equal(rollbacks, 1);
  });

  test("validates safetensors keys, fp32 shapes, layer map, hidden dimension, and finiteness", async () => {
    const valid = fittedFixture();
    assert.equal(
      JSON.parse(new TextDecoder().decode(valid.payloads["manifolds/local/demo/_zdGVzdC9tb2RlbA.json"])).nodes_sha256,
      "91da8d73195b00ca382dc4b6b7535dbd6eee33409ee16b4d41fc332631483938",
    );
    const verified = await validateDrowseArchive(valid.archive);
    assert.deepEqual(verified.fittedRuntimeFingerprints, ["runtime-fixture"]);
    assert.deepEqual(verified.fittedArtifacts, [{
      tensorPath: "manifolds/local/demo/_zdGVzdC9tb2RlbA.safetensors",
      sidecarPath: "manifolds/local/demo/_zdGVzdC9tb2RlbA.json",
      modelId: "test/model",
      variant: "raw",
      variantIdentity: null,
      modelFingerprint: "runtime-fixture",
      contextBindingSha256: null,
      modelSourceFingerprint: null,
    }]);

    const badKey = fittedFixture({ tensor: makeTensor({ "layer_9.mean": [2] }) });
    await assert.rejects(
      validateDrowseArchive(badKey.archive),
      (error) => error.code === "SAFETENSORS_INVALID" && /layer keys/.test(error.message),
    );
    const badHidden = fittedFixture({
      layers: [0, 1],
      tensor: makeTensor({
        "layer_1.mean": [3],
        "layer_1.basis": [1, 3],
        "layer_1.node_coords": [2, 1],
      }),
    });
    await assert.rejects(
      validateDrowseArchive(badHidden.archive),
      (error) => error.code === "SAFETENSORS_INVALID" && /hidden dimensions/.test(error.message),
    );
    const nonFinite = fittedFixture({ tensor: makeTensor({}, { nan: true }) });
    await assert.rejects(
      validateDrowseArchive(nonFinite.archive),
      (error) => error.code === "SAFETENSORS_INVALID" && /non-finite/.test(error.message),
    );
  });

  test("round-trips exact named fp32 tensors deterministically and streams identical bytes", async () => {
    const alpha = { shape: [2], data: new Float32Array([1, -2.5]) };
    const zeta = { shape: [2, 2], data: new Float32Array([0, 3.25, -4, 5.5]) };
    const metadata = { version: "1", source: "python" };
    const encoded = encodeFp32Safetensors({ zeta, alpha }, { metadata, path: "fixture.safetensors" });
    assert.deepEqual(
      encoded,
      encodeFp32Safetensors({ alpha, zeta }, {
        metadata: { source: "python", version: "1" },
        path: "fixture.safetensors",
      }),
    );

    const decoded = decodeFp32Safetensors(encoded, "fixture.safetensors");
    assert.deepEqual(decoded.keys, ["alpha", "zeta"]);
    assert.deepEqual(decoded.description.metadata, { source: "python", version: "1" });
    assert.deepEqual(decoded.tensor("alpha", [2]).shape, [2]);
    assert.deepEqual([...decoded.tensor("alpha").data], [1, -2.5]);
    assert.deepEqual([...decoded.tensor("zeta", [2, 2]).data], [0, 3.25, -4, 5.5]);
    const detached = decoded.tensor("alpha").data;
    detached[0] = 99;
    assert.equal(decoded.tensor("alpha").data[0], 1);
    assert.throws(
      () => decoded.tensor("Alpha"),
      (error) => error.code === "SAFETENSORS_INVALID" && /no tensor named/.test(error.message),
    );
    assert.throws(
      () => decoded.tensor("alpha", [1, 2]),
      (error) => error.code === "SAFETENSORS_INVALID" && /instead of/.test(error.message),
    );

    const large = { shape: [20_000], data: new Float32Array(20_000).fill(0.25) };
    const streamedChunks = [];
    const streamedDescription = await writeFp32Safetensors(
      { large },
      (chunk) => streamedChunks.push(chunk.slice()),
      { path: "streamed.safetensors" },
    );
    assert.deepEqual(concat(streamedChunks), encodeFp32Safetensors({ large }, { path: "streamed.safetensors" }));
    assert.deepEqual(streamedDescription.shapes.get("large"), [20_000]);
    assert.equal(streamedChunks.slice(1).every((chunk) => chunk.length <= 64 * 1024), true);
  });

  test("matches Python safetensors 0.7.0 golden bytes and header layout", () => {
    const golden = Uint8Array.from(Buffer.from(
      "a8000000000000007b225f5f6d657461646174615f5f223a7b22736f75726365223a22707974686f6e222c2276657273696f6e223a2231227d2c22616c706861223a7b226474797065223a22463332222c227368617065223a5b325d2c22646174615f6f666673657473223a5b302c385d7d2c227a657461223a7b226474797065223a22463332222c227368617065223a5b322c325d2c22646174615f6f666673657473223a5b382c32345d7d7d20200000803f000020c00000000000005040000080c00000b040",
      "hex",
    ));
    const encoded = encodeFp32Safetensors({
      zeta: { shape: [2, 2], data: new Float32Array([0, 3.25, -4, 5.5]) },
      alpha: { shape: [2], data: new Float32Array([1, -2.5]) },
    }, { metadata: { version: "1", source: "python" } });
    assert.deepEqual(encoded, golden);
    const headerBytes = new DataView(encoded.buffer, encoded.byteOffset, 8).getBigUint64(0, true);
    assert.equal(headerBytes, 168n);
    assert.equal(
      new TextDecoder().decode(encoded.subarray(8, 8 + Number(headerBytes))),
      '{"__metadata__":{"source":"python","version":"1"},"alpha":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"zeta":{"dtype":"F32","shape":[2,2],"data_offsets":[8,24]}}  ',
    );
  });

  test("rejects malformed fp32 codec inputs and enforces header and key ceilings", () => {
    const invalidInputs = [
      {},
      { x: { shape: [2], data: new Float32Array([1]) } },
      { x: { shape: [1, 1, 1], data: new Float32Array([1]) } },
      { x: { shape: [1], data: new Float32Array([Number.NaN]) } },
      { __metadata__: { shape: [1], data: new Float32Array([1]) } },
    ];
    for (const tensors of invalidInputs) {
      assert.throws(
        () => encodeFp32Safetensors(tensors),
        (error) => error.code === "SAFETENSORS_INVALID",
      );
    }
    assert.throws(
      () => encodeFp32Safetensors({ x: { shape: [1], data: new Float32Array([1]) } }, {
        metadata: { invalid: 1 },
      }),
      (error) => error.code === "SAFETENSORS_INVALID" && /metadata/.test(error.message),
    );
    const tooMany = Object.fromEntries(Array.from({ length: 4_097 }, (_, index) => [
      `tensor_${index}`,
      { shape: [1], data: new Float32Array([index]) },
    ]));
    assert.throws(
      () => encodeFp32Safetensors(tooMany),
      (error) => error.code === "SAFETENSORS_INVALID" && /4096/.test(error.message),
    );
    const oversizedHeader = {
      ["x".repeat(1024 * 1024)]: { shape: [1], data: new Float32Array([1]) },
    };
    assert.throws(
      () => encodeFp32Safetensors(oversizedHeader),
      (error) => error.code === "SAFETENSORS_INVALID" && /oversized header/.test(error.message),
    );

    const gap = rawSafetensors({
      x: { dtype: "F32", shape: [1], data_offsets: [4, 8] },
    }, new Uint8Array(8));
    assert.throws(
      () => decodeFp32Safetensors(gap, "gap.safetensors"),
      (error) => error.code === "SAFETENSORS_INVALID" && /gaps/.test(error.message),
    );
    const wrongDtype = rawSafetensors({
      x: { dtype: "F64", shape: [1], data_offsets: [0, 4] },
    }, new Uint8Array(4));
    assert.throws(
      () => decodeFp32Safetensors(wrongDtype, "dtype.safetensors"),
      (error) => error.code === "SAFETENSORS_INVALID" && /fp32/.test(error.message),
    );
    const nonFinite = encodeFp32Safetensors({ x: { shape: [1], data: new Float32Array([1]) } });
    const dataOffset = 8 + new DataView(nonFinite.buffer, nonFinite.byteOffset, 8).getUint32(0, true);
    new DataView(nonFinite.buffer, nonFinite.byteOffset).setUint32(dataOffset, 0x7f800000, true);
    assert.throws(
      () => decodeFp32Safetensors(nonFinite, "infinite.safetensors"),
      (error) => error.code === "SAFETENSORS_INVALID" && /non-finite/.test(error.message),
    );
  });

  test("binds fitted tensors to node bytes, fit semantics, and filename variants", async () => {
    const stale = fittedFixture({ sidecar: { nodes_sha256: "0".repeat(64) } });
    await assert.rejects(
      validateDrowseArchive(stale.archive),
      (error) => error.code === "CLOSURE_INVALID" && /node hash/.test(error.message),
    );
    const wrongMethod = fittedFixture({ sidecar: { method: "manifold_discover_spectral" } });
    await assert.rejects(
      validateDrowseArchive(wrongMethod.archive),
      (error) => error.code === "CLOSURE_INVALID" && /method\/fit_mode/.test(error.message),
    );
    const rawNamedSae = fittedFixture({ sidecar: {
      method: "manifold_discover_sae",
      feature_space: "sae-release-a",
      sae_release: "release-a",
      sae_ids_by_layer: { "0": "feature-0" },
    } });
    await assert.rejects(
      validateDrowseArchive(rawNamedSae.archive),
      (error) => error.code === "CLOSURE_INVALID" && /filename variant/.test(error.message),
    );
  });

  test("returns runtime fingerprints bound to each exact SAE variant", async () => {
    const tensorName = "_zdGVzdC9tb2RlbA_sae-_zojswyzlbonss2yi.safetensors";
    const fixture = fittedFixture({
      tensorName,
      sidecar: {
        method: "manifold_discover_sae",
        feature_space: "sae-release-a",
        sae_release: "release-a",
        sae_ids_by_layer: { "0": "feature-0" },
      },
    });
    const verified = await validateDrowseArchive(fixture.archive);
    assert.deepEqual(verified.fittedArtifacts, [{
      tensorPath: `manifolds/local/demo/${tensorName}`,
      sidecarPath: `manifolds/local/demo/${tensorName.replace(".safetensors", ".json")}`,
      modelId: "test/model",
      variant: "sae",
      variantIdentity: "release-a",
      modelFingerprint: "runtime-fixture",
      contextBindingSha256: null,
      modelSourceFingerprint: null,
    }]);
  });

  test("binds transferred tensors to their exact source-model variant", async () => {
    const tensorName = "_zdGVzdC9tb2RlbA_from-_zonxxk4tdmuxw233emvwa.safetensors";
    const fixture = fittedFixture({
      tensorName,
      sidecar: {
        method: "manifold_procrustes_transfer",
        source_model_id: "source/model",
        source_model_fingerprint: "source-runtime",
        transfer_quality_estimate: 0.9,
      },
    });
    const verified = await validateDrowseArchive(fixture.archive);
    assert.equal(verified.fittedArtifacts[0].variant, "from");
    assert.equal(verified.fittedArtifacts[0].variantIdentity, "source/model");

    const incomplete = fittedFixture({
      tensorName,
      sidecar: {
        method: "manifold_procrustes_transfer",
        source_model_id: "source/model",
      },
    });
    await assert.rejects(
      validateDrowseArchive(incomplete.archive),
      (error) => error.code === "CLOSURE_INVALID" && /incomplete transfer/.test(error.message),
    );
  });

  test("requires immutable HF provenance and non-reserved manifold namespaces", async () => {
    const mutable = manifoldFixture({ source: "hf://owner/repo@main" });
    const mutableManifest = structuredClone(mutable.manifest);
    mutableManifest.source = { uri: "hf://owner/repo@main", repository: "owner/repo", revision: "main" };
    await assert.rejects(
      inspectDrowseArchive(rebuildWithManifest(mutable, mutableManifest)),
      (error) => error.code === "MANIFEST_INVALID" && /immutable/.test(error.message),
    );

    const fixture = manifoldFixture();
    const renamedPayloads = Object.fromEntries(Object.entries(fixture.payloads).map(([path, data]) => [
      path.replace("manifolds/local/demo", "manifolds/jlens/demo"),
      data,
    ]));
    const reservedManifest = makeManifest(renamedPayloads, { primary: "manifolds/jlens/demo" });
    await assert.rejects(
      inspectDrowseArchive(makeZip([
        { name: "pack.json", data: json(reservedManifest) },
        ...Object.entries(renamedPayloads).map(([name, data]) => ({ name, data })),
      ])),
      (error) => error.code === "MANIFEST_INVALID" && /reserved/.test(error.message),
    );
  });

  test("enforces authored node floors and discover hyperparameter contracts", async () => {
    await assert.rejects(
      validateDrowseArchive(authoredFixture().archive),
      (error) => error.code === "CLOSURE_INVALID" && /at least 3 nodes/.test(error.message),
    );
    await assert.rejects(
      validateDrowseArchive(manifoldFixture({ hyperparams: { k_nn: 3 } }).archive),
      (error) => error.code === "CLOSURE_INVALID" && /does not accept/.test(error.message),
    );
    await assert.rejects(
      validateDrowseArchive(manifoldFixture({ hyperparams: { max_dim: true } }).archive),
      (error) => error.code === "CLOSURE_INVALID" && /invalid value/.test(error.message),
    );
  });

  test("validates affine maps against embedded rather than intrinsic dimension", async () => {
    const valid = fittedFixture({
      tensor: makeTensor({ "layer_0.affine_map": [2, 1] }),
      sidecar: { domain: { type: "sphere", dim: 1 } },
    });
    await validateDrowseArchive(valid.archive);
    const invalid = fittedFixture({
      tensor: makeTensor({ "layer_0.affine_map": [1, 1] }),
      sidecar: { domain: { type: "sphere", dim: 1 } },
    });
    await assert.rejects(
      validateDrowseArchive(invalid.archive),
      (error) => error.code === "SAFETENSORS_INVALID" && /affine_map/.test(error.message),
    );
  });

  let passed = 0;
  for (const { name, run } of tests) {
    await run();
    passed += 1;
    process.stdout.write(`ok ${passed} - ${name}\n`);
  }
  process.stdout.write(`1..${passed}\n`);
} finally {
  await server.close();
}

function manifoldFixture(options = {}) {
  const labels = options.template ? ["monday", "tuesday"] : ["calm", "alert"];
  const corpora = options.nodeCorpora ?? (options.template
    ? [["today is Monday"], ["today is Tuesday"]]
    : [["calm"], ["alert"]]);
  const templateRef = options.template ? "local/weekday" : null;
  const manifold = {
    format_version: 10,
    name: "demo",
    description: "fixture",
    fit_mode: "pca",
    hyperparams: options.hyperparams ?? {},
    nodes: labels.map((label) => ({ label, role: null, kind: "abstract" })),
    files: {},
    source: options.source ?? "local",
    tags: [],
    template_ref: templateRef,
  };
  const payloads = {
    "manifolds/local/demo/manifold.json": json(manifold),
  };
  labels.forEach((label, index) => {
    payloads[`manifolds/local/demo/nodes/${String(index).padStart(2, "0")}_${label}.json`] = json(corpora[index]);
  });
  if (options.template) {
    payloads["templates/local/weekday/template.json"] = json({
      format_version: 2,
      name: "weekday",
      slot: "[DAY]",
      values: ["Monday", "Tuesday"],
      contexts: [{
        turns: [{ role: "user", content: "what day is it?" }],
        assistant: "today is [DAY]",
      }],
      description: "fixture",
      source: "local",
      tags: [],
    });
  }
  if (options.extraPayload) payloads[options.extraPayload[0]] = options.extraPayload[1];
  const manifest = makeManifest(payloads, { template: options.template ? "templates/local/weekday" : null });
  const entries = [
    { name: "pack.json", data: json(manifest), method: options.method ?? 0 },
    ...Object.entries(payloads).map(([name, data]) => ({ name, data, method: options.method ?? 0 })),
  ];
  return { archive: makeZip(entries), entries, payloads, manifest };
}

function runPythonPackBridge(action, fixturePath, homePath, archivePath) {
  const repository = fileURLToPath(new URL("../..", import.meta.url));
  const bridge = fileURLToPath(
    new URL("../../tests/browser_drowse_archive_bridge.py", import.meta.url),
  );
  const result = spawnSync(
    process.env.PYTHON ?? "python3",
    [bridge, action, fixturePath, homePath, archivePath],
    {
      cwd: repository,
      encoding: "utf8",
      env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" },
    },
  );
  assert.equal(
    result.status,
    0,
    `Python drowseArchive bridge failed:\n${result.stderr || result.stdout}`,
  );
  return JSON.parse(result.stdout);
}

function fittedFixture(options = {}) {
  const tensorName = options.tensorName ?? "_zdGVzdC9tb2RlbA.safetensors";
  const sidecarName = tensorName.replace(".safetensors", ".json");
  const tensor = options.tensor ?? makeTensor();
  const layers = options.layers ?? [0];
  const nodePayloads = {
    "manifolds/local/demo/nodes/00_calm.json": json(["calm"]),
    "manifolds/local/demo/nodes/01_alert.json": json(["alert"]),
  };
  const sidecar = {
    format_version: 10,
    name: "demo",
    method: "manifold_discover_pca",
    ...(options.legacyBrand
      ? { saklas_version: "test" }
      : { drowse_version: "test" }),
    fit_mode: "pca",
    hyperparams: {},
    diagnostics: {},
    node_count: 2,
    node_labels: ["calm", "alert"],
    node_roles: [null, null],
    node_kinds: ["abstract", "abstract"],
    domain: { type: "custom", embed_dim: 1, bounds: null },
    node_spread_per_layer: {},
    fitted_layers: layers,
    mahalanobis_share_per_layer: Object.fromEntries(layers.map((layer) => [String(layer), 1])),
    origin_per_layer: {},
    feature_space: "raw",
    nodes_sha256: discoverNodesHash(
      ["calm", "alert"],
      Object.values(nodePayloads),
      "pca",
      {},
      [null, null],
      ["abstract", "abstract"],
    ),
    sae_release: null,
    sae_revision: null,
    sae_fingerprint: null,
    sae_ids_by_layer: {},
    sae_full_coverage: false,
    model_fingerprint: "runtime-fixture",
    context_binding_sha256: null,
    model_source_fingerprint: null,
    capture_sha256: null,
    capture_version: null,
    capture_render_sha256: null,
    baseline_prompts_sha256: null,
    fit_policy_version: 1,
    share_metric: "mahalanobis",
    subspace_metric: "mahalanobis",
    rbf_smoothing_per_layer: {},
    sigma_field_per_layer: {},
    resolved_fit_mode: null,
    topology_winner: null,
    topology_candidates: [],
    components: null,
    bake_policy: null,
    source_model_id: null,
    source_model_fingerprint: null,
    transfer_quality_estimate: null,
    ...options.sidecar,
  };
  const sidecarBytes = json(sidecar);
  const manifold = {
    format_version: 10,
    name: "demo",
    description: "fixture",
    fit_mode: "pca",
    hyperparams: {},
    nodes: ["calm", "alert"].map((label) => ({ label, role: null, kind: "abstract" })),
    files: {
      [tensorName]: digest(tensor),
      [sidecarName]: digest(sidecarBytes),
    },
    source: "local",
    tags: [],
    template_ref: null,
  };
  const payloads = {
    "manifolds/local/demo/manifold.json": json(manifold),
    ...nodePayloads,
    [`manifolds/local/demo/${tensorName}`]: tensor,
    [`manifolds/local/demo/${sidecarName}`]: sidecarBytes,
  };
  const manifest = makeManifest(payloads);
  if (options.legacyBrand) {
    manifest.kind = "saklas-manifold";
    manifest.producer.name = "saklas";
  }
  const entries = [{ name: "pack.json", data: json(manifest) }, ...Object.entries(payloads).map(([name, data]) => ({ name, data }))];
  return { archive: makeZip(entries), entries, payloads, manifest };
}

function authoredFixture() {
  const manifold = {
    format_version: 10,
    name: "demo",
    description: "fixture",
    fit_mode: "authored",
    domain: {
      type: "box",
      axes: [{ name: "x", periodic: false, lo: 0.0, hi: 1.0, period: 1.0 }],
    },
    nodes: [
      { label: "calm", coords: [0.0], role: null, kind: "abstract" },
      { label: "alert", coords: [1.0], role: null, kind: "abstract" },
    ],
    files: {},
    source: "local",
    tags: [],
    template_ref: null,
  };
  const payloads = {
    "manifolds/local/demo/manifold.json": json(manifold),
    "manifolds/local/demo/nodes/00_calm.json": json(["calm"]),
    "manifolds/local/demo/nodes/01_alert.json": json(["alert"]),
  };
  const manifest = makeManifest(payloads);
  const entries = [{ name: "pack.json", data: json(manifest) }, ...Object.entries(payloads).map(([name, data]) => ({ name, data }))];
  return { archive: makeZip(entries), entries, payloads, manifest };
}

function discoverNodesHash(labels, nodePayloads, fitMode, hyperparams, roles, kinds) {
  const hash = createHash("sha256");
  hash.update(canonicalJson(labels));
  for (const payload of nodePayloads) hash.update(payload);
  hash.update(canonicalJson({ fit_mode: fitMode, hyperparams }));
  hash.update(canonicalJson(roles));
  hash.update(canonicalJson(kinds));
  return hash.digest("hex");
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function makeTensor(extraShapes = {}, options = {}) {
  const shapes = {
    node_coords: [2, 1],
    "layer_0.mean": [2],
    "layer_0.basis": [1, 2],
    "layer_0.node_coords": [2, 1],
    ...extraShapes,
  };
  let offset = 0;
  const header = {};
  for (const [key, shape] of Object.entries(shapes)) {
    const elements = shape.reduce((total, value) => total * value, 1);
    header[key] = { dtype: "F32", shape, data_offsets: [offset, offset + elements * 4] };
    offset += elements * 4;
  }
  let headerBytes = text(JSON.stringify(header));
  const padding = (8 - (headerBytes.length % 8)) % 8;
  if (padding) {
    const padded = new Uint8Array(headerBytes.length + padding);
    padded.set(headerBytes);
    padded.fill(0x20, headerBytes.length);
    headerBytes = padded;
  }
  const output = new Uint8Array(8 + headerBytes.length + offset);
  new DataView(output.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  output.set(headerBytes, 8);
  if (options.nan) new DataView(output.buffer).setUint32(8 + headerBytes.length, 0x7fc00000, true);
  return output;
}

function rawSafetensors(header, payload) {
  let headerBytes = text(JSON.stringify(header));
  const padding = (8 - headerBytes.length % 8) % 8;
  if (padding > 0) {
    const padded = new Uint8Array(headerBytes.length + padding);
    padded.set(headerBytes);
    padded.fill(0x20, headerBytes.length);
    headerBytes = padded;
  }
  const output = new Uint8Array(8 + headerBytes.length + payload.length);
  new DataView(output.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
  output.set(headerBytes, 8);
  output.set(payload, 8 + headerBytes.length);
  return output;
}

function makeManifest(payloads, overrides = {}) {
  return {
    format_version: 1,
    kind: "drowse-manifold",
    producer: { name: "drowse", version: "test" },
    primary: "manifolds/local/demo",
    template: null,
    source: { uri: "local", repository: null, revision: null },
    files: Object.fromEntries(Object.entries(payloads).map(([path, data]) => [path, { size: data.length, sha256: digest(data) }])),
    ...overrides,
  };
}

function rebuildWithManifest(fixture, manifest) {
  return makeZip([{ name: "pack.json", data: json(manifest) }, ...fixture.entries.slice(1)]);
}

function json(value) {
  return text(`${JSON.stringify(value)}\n`);
}

function memoryStage(output, committed) {
  const chunks = new Map();
  return {
    begin() {},
    write(path, chunk) {
      const list = chunks.get(path) ?? [];
      list.push(chunk.slice());
      chunks.set(path, list);
    },
    finish(path) {
      const list = chunks.get(path) ?? [];
      const bytes = new Uint8Array(list.reduce((total, chunk) => total + chunk.length, 0));
      let offset = 0;
      for (const chunk of list) {
        bytes.set(chunk, offset);
        offset += chunk.length;
      }
      output.set(path, bytes);
    },
    commit: committed,
    rollback() {},
  };
}

function makeZip(rawEntries) {
  const entries = [];
  const localParts = [];
  let localOffset = 0;
  for (const raw of rawEntries) {
    const path = rawBytes(raw.name);
    const localPath = rawBytes(raw.localName ?? raw.name);
    const data = raw.data ?? new Uint8Array();
    const method = raw.method ?? 0;
    const compressed = method === 8 ? deflateSync(data) : data;
    const flags = raw.flags ?? 0;
    const extra = raw.extra ?? new Uint8Array();
    const checksum = crc32(data);
    const declaredSize = raw.declaredSize ?? data.length;
    const declaredCompressedSize = raw.declaredCompressedSize ?? compressed.length;
    const local = new Uint8Array(30 + localPath.length + extra.length + compressed.length);
    const view = new DataView(local.buffer);
    view.setUint32(0, 0x04034b50, true);
    view.setUint16(4, 20, true);
    view.setUint16(6, flags, true);
    view.setUint16(8, method, true);
    view.setUint32(14, checksum, true);
    view.setUint32(18, declaredCompressedSize, true);
    view.setUint32(22, declaredSize, true);
    view.setUint16(26, localPath.length, true);
    view.setUint16(28, extra.length, true);
    local.set(localPath, 30);
    local.set(extra, 30 + localPath.length);
    local.set(compressed, 30 + localPath.length + extra.length);
    localParts.push(local);
    entries.push({
      path,
      data,
      method,
      compressed,
      flags,
      extra,
      checksum,
      declaredSize,
      declaredCompressedSize,
      localOffset,
      madeBy: raw.madeBy ?? 0x0314,
      externalAttributes: raw.externalAttributes ?? 0x81a40000,
    });
    localOffset += local.length;
  }
  const centralParts = [];
  for (const entry of entries) {
    const central = new Uint8Array(46 + entry.path.length + entry.extra.length);
    const view = new DataView(central.buffer);
    view.setUint32(0, 0x02014b50, true);
    view.setUint16(4, entry.madeBy, true);
    view.setUint16(6, 20, true);
    view.setUint16(8, entry.flags, true);
    view.setUint16(10, entry.method, true);
    view.setUint32(16, entry.checksum, true);
    view.setUint32(20, entry.declaredCompressedSize, true);
    view.setUint32(24, entry.declaredSize, true);
    view.setUint16(28, entry.path.length, true);
    view.setUint16(30, entry.extra.length, true);
    view.setUint32(38, entry.externalAttributes >>> 0, true);
    view.setUint32(42, entry.localOffset, true);
    central.set(entry.path, 46);
    central.set(entry.extra, 46 + entry.path.length);
    centralParts.push(central);
  }
  const centralOffset = localParts.reduce((total, part) => total + part.length, 0);
  const centralSize = centralParts.reduce((total, part) => total + part.length, 0);
  const eocd = new Uint8Array(22);
  const eocdView = new DataView(eocd.buffer);
  eocdView.setUint32(0, 0x06054b50, true);
  eocdView.setUint16(8, entries.length, true);
  eocdView.setUint16(10, entries.length, true);
  eocdView.setUint32(12, centralSize, true);
  eocdView.setUint32(16, centralOffset, true);
  return concat([...localParts, ...centralParts, eocd]);
}

function rawBytes(value) {
  return value instanceof Uint8Array ? value : text(value);
}

function extraField(tag, data) {
  const output = new Uint8Array(4 + data.length);
  const view = new DataView(output.buffer);
  view.setUint16(0, tag, true);
  view.setUint16(2, data.length, true);
  output.set(data, 4);
  return output;
}

function concat(parts) {
  const output = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }
  return output;
}

function crc32(bytes) {
  const table = crc32.table ??= Array.from({ length: 256 }, (_, index) => {
    let entry = index;
    for (let bit = 0; bit < 8; bit += 1) entry = (entry & 1) ? 0xedb88320 ^ (entry >>> 1) : entry >>> 1;
    return entry >>> 0;
  });
  let value = 0xffffffff;
  for (const byte of bytes) value = table[(value ^ byte) & 0xff] ^ (value >>> 8);
  return (value ^ 0xffffffff) >>> 0;
}
