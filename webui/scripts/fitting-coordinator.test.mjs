import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
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

const tests = [];
const test = (name, run) => tests.push({ name, run });

try {
  const {
    BrowserFittingCoordinator,
    anchorBrowserNeutralLayout,
    embedBrowserAuthoredCoordinates,
    finalizeBrowserAffineLayers,
    fitBrowserCurvedSurfaceLayers,
    selectBrowserDlsAxes,
  } = await server.ssrLoadModule(
    "/src/hosted/fitting/coordinator.ts",
  );

  test("captures, commits, pools, and removes activation rows in order", async () => {
    const harness = coordinatorHarness(BrowserFittingCoordinator);
    const events = [];
    const result = await harness.coordinator.captureNodeCentroids(
      plan(),
      captureSource([
        [0, new Float32Array([1, 10, 3, 14, 8, 20, 12, 24])],
        [1, new Float32Array([2, 4, 4, 8, 10, 20, 14, 28])],
      ]),
      { onProgress: (event) => events.push(event) },
    );
    assert.deepEqual([...result.layers.get(0).values], [2, 12, 10, 22]);
    assert.deepEqual([...result.layers.get(1).values], [3, 6, 12, 24]);
    assert.equal(result.captureRetained, false);
    assert.deepEqual(harness.lifecycle, [
      "begin", "append:0:0", "append:1:0", "checkpoint:0", "seal:0",
      "checkpoint:1", "seal:1", "commit", "run:0", "run:1", "remove",
    ]);
    assert.equal(events.some((event) => event.stage === "capturing"), true);
    assert.equal(events.filter((event) => event.stage === "pooling").at(-1).completed, 2);
  });

  test("fits pooled layers against verified serialized whiteners", async () => {
    const harness = coordinatorHarness(BrowserFittingCoordinator);
    const progress = [];
    const sourceWhitener = identityWhitener(2);
    const result = await harness.coordinator.captureAffineLayers(
      {
        ...plan(),
        whiteners: new Map([
          [0, sourceWhitener],
          [1, identityWhitener(2)],
        ]),
        maxComponents: 1,
      },
      captureSource([
        [0, new Float32Array([1, 10, 3, 14, 8, 20, 12, 24])],
        [1, new Float32Array([2, 4, 4, 8, 10, 20, 14, 28])],
      ]),
      { onProgress: (event) => progress.push(event) },
    );
    assert.deepEqual([...result.layers.keys()], [0, 1]);
    assert.deepEqual([...result.layers.get(0).centroidMean], [6, 17]);
    assert.deepEqual([...result.layers.get(1).centroidMean], [7.5, 15]);
    assert.deepEqual(harness.lifecycle.slice(-3), ["remove", "fit", "fit"]);
    assert.equal(progress.filter((event) => event.stage === "fitting").at(-1).completed, 2);
    assert.equal(sourceWhitener.mean.byteLength, 16);
    const fitTransfer = harness.transferRuns.find(({ job }) => job.operation === "affine_fisher");
    assert.ok(fitTransfer);
    assert.equal(fitTransfer.views.includes(fitTransfer.job.source.values), true);
    assert.equal(fitTransfer.views.includes(fitTransfer.job.whitener.mean), true);
    assert.equal(fitTransfer.views.includes(sourceWhitener.mean), false);
  });

  test("parallel affine fitting preserves layer order and drains failed siblings", async () => {
    for (const fail of [false, true]) {
      const harness = coordinatorHarness(BrowserFittingCoordinator);
      harness.worker.parallelism = 2;
      const original = harness.worker.run.bind(harness.worker);
      const pending = [];
      let bothStarted;
      const started = new Promise(resolve => { bothStarted = resolve; });
      let aborted = false;
      harness.worker.run = async (job, options) => {
        if (job.operation !== "affine_fisher") return original(job, options);
        return new Promise((resolve, reject) => {
          const onAbort = () => { aborted = true; reject(options.signal.reason); };
          options.signal.addEventListener("abort", onAbort, { once: true });
          pending.push({
            finish: async () => {
              options.signal.removeEventListener("abort", onAbort);
              resolve(await original(job, options));
            },
            fail: () => reject(new Error("parallel fit failed")),
          });
          if (pending.length === 2) bothStarted();
        });
      };
      const result = harness.coordinator.captureAffineLayers({
        ...plan(), maxComponents: 1,
        whiteners: new Map([[0, identityWhitener(2)], [1, identityWhitener(2)]]),
      }, captureSource([
        [0, new Float32Array([1, 10, 3, 14, 8, 20, 12, 24])],
        [1, new Float32Array([2, 4, 4, 8, 10, 20, 14, 28])],
      ]));
      await started;
      if (fail) {
        const rejected = assert.rejects(result, /parallel fit failed/);
        pending[1].fail();
        await rejected;
        assert.equal(aborted, true);
        assert.equal(harness.lifecycle.at(-1), "remove");
      } else {
        await pending[1].finish();
        await pending[0].finish();
        assert.deepEqual([...(await result).layers.keys()], [0, 1]);
      }
    }
  });

  test("builds a consensus Gram and variable-width topology target envelope", async () => {
    const harness = coordinatorHarness(BrowserFittingCoordinator);
    const result = await harness.coordinator.captureTopologyFoundation(
      {
        ...plan(),
        whiteners: new Map([
          [0, identityWhitener(2, 5)],
          [1, identityWhitener(2, 100)],
        ]),
        maxComponents: 1,
        fitMode: "pca",
        maxDimensions: 2,
        varianceThreshold: 0.7,
        persistenceFraction: 0.5,
      },
      captureSource([
        [0, new Float32Array([1, 10, 3, 14, 8, 20, 12, 24])],
        [1, new Float32Array([2, 4, 4, 8, 10, 20, 14, 28])],
      ]),
    );
    assert.equal(result.topology.winnerName, "flat-pca");
    assert.deepEqual([...result.neutralLayoutCoordinate], [0]);
    assert.deepEqual([...result.anchoredNodeCoordinates], [-1, 1]);
    assert.deepEqual([...result.dlsKept.get(0)], [0]);
    assert.equal(result.dlsKept.has(1), false);
    assert.deepEqual([...result.finalAffineLayers.keys()], [0]);
    assert.equal(result.curvedSurfaceLayers, null);
    assert.deepEqual([...result.finalAffineLayers.get(0).mean], [5, 0]);
    assert.equal(result.finalAffineLayers.get(0).affineMap, null);
    assert.ok(Math.abs(result.finalAffineLayers.get(0).mahalanobisShare - Math.sqrt(32)) < 1e-12);
    assert.deepEqual([...result.consensusGram], [1.5, 1.5, 1.5, 1.5]);
    assert.deepEqual([...harness.topologyJobs[0].targetOffsets], [0, 2, 4]);
    const topologyTransfer = harness.transferRuns.find(({ job }) => job.operation === "topology");
    assert.deepEqual(topologyTransfer.views, [
      topologyTransfer.job.targets,
      topologyTransfer.job.targetOffsets,
    ]);
    assert.equal(topologyTransfer.views.includes(topologyTransfer.job.consensusGram), false);
    assert.deepEqual(harness.lifecycle.slice(-3), ["fit", "topology", "remove"]);
  });

  test("fits every retained layer through the selected curved topology", async () => {
    const harness = coordinatorHarness(BrowserFittingCoordinator, {
      topologyFitMode: "spectral",
    });
    const result = await harness.coordinator.captureTopologyFoundation(
      {
        ...plan(),
        whiteners: new Map([
          [0, identityWhitener(2)],
          [1, identityWhitener(2)],
        ]),
        maxComponents: 1,
        fitMode: "spectral",
        maxDimensions: 1,
        varianceThreshold: 0.7,
        persistenceFraction: 0.5,
        smoothing: 0.25,
      },
      captureSource([
        [0, new Float32Array([1, 10, 3, 14, 8, 20, 12, 24])],
        [1, new Float32Array([2, 4, 4, 8, 10, 20, 14, 28])],
      ]),
    );
    assert.equal(result.finalAffineLayers, null);
    assert.equal(result.dlsKept, null);
    assert.deepEqual([...result.curvedSurfaceLayers.keys()], [0, 1]);
    assert.deepEqual([...result.curvedSurfaceLayers.get(0).surface.nodes], [-1, 1]);
    assert.equal(result.curvedSurfaceLayers.get(0).sigmaSummary.mean, 1);
    assert.deepEqual([...result.curvedSurfaceLayers.get(0).origin], [0]);
    assert.equal(harness.topologyJobs[0].smoothing, 0.25);
    assert.deepEqual(harness.rbfJobs.map((job) => job.smoothing), [0.25, 0.25]);
    assert.equal(harness.rbfJobs.every((job) => job.plan !== undefined), true);
    assert.equal(harness.sigmaJobs.every((job) => job.plan !== undefined), true);
    assert.deepEqual(
      harness.rbfJobs.map((job) => [...job.plan.nodes]),
      [[-1, 1], [-1, 1]],
    );
    assert.deepEqual(
      harness.sigmaJobs.map((job) => [...job.plan.nodes]),
      [[-1, 1], [-1, 1]],
    );
    assert.deepEqual(
      harness.lifecycle.slice(-7),
      ["covariance", "sigma", "origin", "covariance", "sigma", "origin", "remove"],
    );
  });

  test("reuses one curved winner plan across per-layer output widths", async () => {
    const harness = coordinatorHarness(BrowserFittingCoordinator);
    const topology = spectralTopology();
    const layers = new Map([
      [0, affineFit([-2, -1, 1, 2], 2)],
      [1, affineFit([-3, 3], 1)],
    ]);
    const result = await fitBrowserCurvedSurfaceLayers(
      layers,
      topology,
      harness.worker,
      { smoothing: 0.25 },
    );
    assert.deepEqual([...result].map(([layer, fit]) => [layer, fit.surface.outputDimensions]), [
      [0, 2],
      [1, 1],
    ]);
    assert.deepEqual(harness.rbfJobs.map((job) => job.outputDimensions), [2, 1]);
    assert.deepEqual(harness.rbfJobs.map((job) => job.smoothing), [0.25, 0.25]);
    for (const job of harness.rbfJobs) {
      assert.deepEqual([...job.plan.nodes], [...topology.winnerPlan.nodes]);
      assert.deepEqual([...job.plan.kernel], [...topology.winnerPlan.kernel]);
      assert.deepEqual([...job.plan.lambdas], [...topology.winnerPlan.lambdas]);
    }
  });

  test("fits authored coordinates as exact curved surfaces with raw sigma fields", async () => {
    const harness = coordinatorHarness(BrowserFittingCoordinator);
    const domain = {
      type: "box",
      axes: [{ name: "phase", periodic: true, period: 4, lo: 0, hi: 4 }],
    };
    const coordinates = new Float64Array([0, 1, 2, 3]);
    const embeddedCoordinates = embedBrowserAuthoredCoordinates(domain, coordinates, 4);
    close(embeddedCoordinates, [1, 0, 0, 1, -1, 0, 0, -1], 1e-12);
    const authoredPlan = {
      descriptor: {
        ...plan().descriptor,
        layers: [
          { layer: 0, rows: 4, width: 2, expectedBytes: 32 },
          { layer: 1, rows: 4, width: 2, expectedBytes: 32 },
        ],
      },
      groupOffsets: new Uint32Array([0, 1, 2, 3, 4]),
      whiteners: new Map([
        [0, identityWhitener(2)],
        [1, identityWhitener(2)],
      ]),
      maxComponents: 2,
      domain,
      coordinates,
      embeddedCoordinates,
      intrinsicDimensions: 1,
      fitSigma: true,
    };
    const result = await harness.coordinator.captureAuthoredFoundation(
      authoredPlan,
      captureSource([
        [0, new Float32Array([1, 10, 3, 14, 8, 20, 12, 24])],
        [1, new Float32Array([2, 4, 4, 8, 10, 20, 14, 28])],
      ]),
    );
    assert.deepEqual([...result.curvedSurfaceLayers.keys()], [0, 1]);
    assert.deepEqual(harness.rbfJobs.slice(0, 2).map((job) => job.smoothing), [0, 0]);
    assert.deepEqual(harness.rbfJobs.slice(0, 2).map((job) => [...job.nodes]), [
      [...embeddedCoordinates],
      [...embeddedCoordinates],
    ]);
    assert.equal(result.curvedSurfaceLayers.get(0).surface.lambda, 0);
    assert.ok(result.curvedSurfaceLayers.get(0).sigmaSummary.mean > 0);
    assert.deepEqual([...result.curvedSurfaceLayers.get(0).origin], [0]);
    assert.equal(harness.lifecycle.filter((entry) => entry === "covariance").length, 2);
    assert.equal(harness.lifecycle.at(-1), "remove");
  });

  test("applies an exact centroid reconstruction port and omits raw sigma for SAE fits", async () => {
    const harness = coordinatorHarness(BrowserFittingCoordinator);
    const coordinates = new Float64Array([-1, 0, 1]);
    const domain = { type: "custom", embed_dim: 1, bounds: null };
    const result = await harness.coordinator.captureAuthoredFoundation(
      {
        descriptor: {
          runtimeIdentitySha256: "1".repeat(64),
          contextBindingSha256: "2".repeat(64),
          captureSha256: "3".repeat(64),
          layers: [{ layer: 0, rows: 3, width: 2, expectedBytes: 24 }],
        },
        groupOffsets: new Uint32Array([0, 1, 2, 3]),
        whiteners: new Map([[0, identityWhitener(2)]]),
        maxComponents: 2,
        transformCentroids(layer, centroids) {
          assert.equal(layer, 0);
          return {
            ...centroids,
            values: Float64Array.from(centroids.values, (value) => value + 5),
          };
        },
        domain,
        coordinates,
        embeddedCoordinates: coordinates.slice(),
        intrinsicDimensions: 1,
        fitSigma: false,
      },
      captureSource([[0, new Float32Array([1, 2, 3, 4, 5, 6])]]),
    );
    assert.deepEqual([...result.layers.get(0).centroidMean], [8, 9]);
    assert.equal(result.curvedSurfaceLayers.get(0).sigmaSurface, null);
    assert.equal(result.curvedSurfaceLayers.get(0).sigmaSummary, null);
    assert.equal(harness.lifecycle.includes("covariance"), false);
    assert.equal(harness.lifecycle.includes("remove"), true);
  });

  test("matches the shared Python DLS keep and all-fail fallback fixtures", async () => {
    const fixture = JSON.parse(await readFile(new URL(
      "../../browser-runtime/fixtures/dls-axes-v1.json",
      import.meta.url,
    )));
    const layers = new Map(fixture.layers.map((layer) => [layer.layer, {
      operation: "affine_fisher",
      nodeCount: layer.nodeCount,
      columns: layer.components,
      components: layer.components,
      centroidMean: new Float64Array(layer.components),
      mean: new Float64Array(layer.components),
      basis: new Float64Array(layer.components * layer.components),
      nodeCoordinates: new Float64Array(layer.nodeCoordinates),
      muCoordinates: new Float64Array(layer.nodeCoordinates.length),
      whitenedGram: new Float64Array(layer.nodeCount * layer.nodeCount),
      neutralCrossGram: new Float64Array(layer.nodeCount),
      explainedVariance: 1,
      mahalanobisShare: 1,
    }]));
    const selected = selectBrowserDlsAxes(layers, fixture.intrinsicDimensions);
    assert.deepEqual(mapAxes(selected), fixture.expected);
    for (const layer of layers.values()) {
      layer.nodeCoordinates = layer.nodeCoordinates.map((value) => Math.abs(value) + 1);
    }
    assert.deepEqual(
      mapAxes(selectBrowserDlsAxes(layers, fixture.intrinsicDimensions)),
      fixture.allFailExpected,
    );
  });

  test("matches Python neutral anchoring in the shared consensus layout", async () => {
    const fixture = JSON.parse(await readFile(new URL(
      "../../browser-runtime/fixtures/affine-fisher-fit-v1.json",
      import.meta.url,
    )));
    const expected = fixture.expected;
    const layers = new Map([[0, {
      operation: "affine_fisher",
      nodeCount: fixture.nodeCount,
      columns: fixture.columns,
      components: fixture.maxComponents,
      centroidMean: new Float64Array(expected.centroidMean),
      mean: new Float64Array(expected.mean),
      basis: new Float64Array(expected.basis),
      nodeCoordinates: new Float64Array(expected.nodeCoordinates),
      muCoordinates: new Float64Array(expected.muCoordinates),
      whitenedGram: new Float64Array(expected.whitenedGram),
      neutralCrossGram: new Float64Array(expected.neutralCrossGram),
      explainedVariance: expected.explainedVariance,
      mahalanobisShare: expected.mahalanobisShare,
    }]]);
    const anchored = anchorBrowserNeutralLayout(layers, {
      operation: "topology",
      winnerName: "flat-pca",
      fitMode: "pca",
      intrinsicDimensions: 2,
      periodicDimensions: 0,
      persistentLoops: 0,
      usedFaintCycle: false,
      coordinates: new Float64Array(expected.layoutCoordinates),
      embeddedCoordinates: new Float64Array(expected.layoutCoordinates),
      candidates: [],
      diagnostics: {
        kind: "pca",
        perComponentVariance: new Float64Array([0.75, 0.25]),
        cumulativeVariance: new Float64Array([0.75, 1]),
        pickedDimensions: 2,
        threshold: 0.7,
      },
      winnerPlan: null,
    });
    close(anchored.neutral, expected.neutralLayoutCoordinate, 2e-6);
    close(anchored.coordinates, expected.anchoredLayoutCoordinates, 2e-6);

    const serialized = fixture.serializedWhitener;
    const finalized = finalizeBrowserAffineLayers(
      layers,
      new Map([[0, {
        ...serialized,
        mean: new Float64Array(serialized.mean),
        basis: new Float64Array(serialized.basis),
        eigenvalues: new Float64Array(serialized.eigenvalues),
        inverseScales: new Float64Array(serialized.inverseScales),
      }]]),
      new Map([[0, new Uint32Array([0])]]),
      2,
    ).get(0);
    close(finalized.mean, expected.axisZeroFinal.mean, 2e-6);
    close(finalized.basis, expected.axisZeroFinal.basis, 2e-6);
    close(finalized.nodeCoordinates, expected.axisZeroFinal.nodeCoordinates, 2e-6);
    close(finalized.muCoordinates, expected.axisZeroFinal.muCoordinates, 2e-6);
    close(finalized.affineMap, [1, 0], 0);
    assert.ok(
      Math.abs(finalized.mahalanobisShare - expected.axisZeroFinal.mahalanobisShare) <= 2e-6,
    );

    const reordered = finalizeBrowserAffineLayers(
      layers,
      new Map([[0, {
        ...serialized,
        mean: new Float64Array(serialized.mean),
        basis: new Float64Array(serialized.basis),
        eigenvalues: new Float64Array(serialized.eigenvalues),
        inverseScales: new Float64Array(serialized.inverseScales),
      }]]),
      new Map([[0, new Uint32Array([1, 0])]]),
      2,
    ).get(0);
    close(reordered.affineMap, [0, 1, 1, 0], 0);
  });

  test("rolls back incomplete or cancelled captures before fitting", async () => {
    const incomplete = coordinatorHarness(BrowserFittingCoordinator);
    await assert.rejects(
      incomplete.coordinator.captureNodeCentroids(plan(), captureSource([
        [0, new Float32Array([1, 2])],
      ])),
      /wrote 1 of 4 rows/,
    );
    assert.equal(incomplete.lifecycle.at(-1), "rollback");
    assert.equal(incomplete.lifecycle.some((entry) => entry.startsWith("run:")), false);

    const cancelled = coordinatorHarness(BrowserFittingCoordinator);
    const controller = new AbortController();
    await assert.rejects(
      cancelled.coordinator.captureNodeCentroids(plan(), {
        async capture(sink) {
          await sink.appendRows(0, 0, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
          controller.abort();
        },
      }, { signal: controller.signal }),
      (error) => error.name === "AbortError",
    );
    assert.equal(cancelled.lifecycle.at(-1), "rollback");
  });

  test("removes committed captures when pooling fails and retains only by request", async () => {
    const failed = coordinatorHarness(BrowserFittingCoordinator, { failLayer: 1 });
    await assert.rejects(
      failed.coordinator.captureNodeCentroids(plan(), captureSource([
        [0, new Float32Array([1, 10, 3, 14, 8, 20, 12, 24])],
        [1, new Float32Array([2, 4, 4, 8, 10, 20, 14, 28])],
      ])),
      /pool failed/,
    );
    assert.deepEqual(failed.lifecycle.slice(-2), ["run:1", "remove"]);

    const retained = coordinatorHarness(BrowserFittingCoordinator);
    const result = await retained.coordinator.captureNodeCentroids(
      { ...plan(), retainCapture: true },
      captureSource([
        [0, new Float32Array([1, 10, 3, 14, 8, 20, 12, 24])],
        [1, new Float32Array([2, 4, 4, 8, 10, 20, 14, 28])],
      ]),
    );
    assert.equal(result.captureRetained, true);
    assert.equal(retained.lifecycle.includes("remove"), false);
  });

  for (const method of ["captureAffineLayers", "captureTopologyFoundation", "captureAuthoredFoundation"]) {
    test(`${method} removes retained capture when the affine worker fails`, async () => {
      const harness = coordinatorHarness(BrowserFittingCoordinator);
      const run = harness.worker.run;
      harness.worker.run = async (job, options) => {
        if (job.operation === "affine_fisher") throw new Error("affine worker failed");
        return run(job, options);
      };
      await assert.rejects(harness.coordinator[method]({
        ...plan(),
        retainCapture: method === "captureAffineLayers",
        whiteners: new Map([[0, identityWhitener(2)], [1, identityWhitener(2)]]),
        maxComponents: 1,
        fitMode: "pca", maxDimensions: 1, varianceThreshold: 0.7, persistenceFraction: 0.5,
        domain: { type: "custom", embed_dim: 1, bounds: null },
        coordinates: new Float64Array([-1, 1]),
        embeddedCoordinates: new Float64Array([-1, 1]),
        intrinsicDimensions: 1, fitSigma: true,
      }, captureSource([
        [0, new Float32Array([1, 10, 3, 14, 8, 20, 12, 24])],
        [1, new Float32Array([2, 4, 4, 8, 10, 20, 14, 28])],
      ])), /affine worker failed/);
      assert.equal(harness.lifecycle.at(-1), "remove");
      assert.equal(harness.lifecycle.filter(event => event === "remove").length, 1);
    });
  }

  test("cancellation after the final affine layer removes the retained capture", async () => {
    const harness = coordinatorHarness(BrowserFittingCoordinator);
    const controller = new AbortController();
    await assert.rejects(harness.coordinator.captureAffineLayers({
      ...plan(), retainCapture: true,
      whiteners: new Map([[0, identityWhitener(2)], [1, identityWhitener(2)]]),
      maxComponents: 1,
    }, captureSource([
      [0, new Float32Array([1, 10, 3, 14, 8, 20, 12, 24])],
      [1, new Float32Array([2, 4, 4, 8, 10, 20, 14, 28])],
    ]), {
      signal: controller.signal,
      onProgress(event) {
        if (event.stage === "fitting" && event.completed === event.total) controller.abort();
      },
    }), { name: "AbortError" });
    assert.equal(harness.lifecycle.at(-1), "remove");
  });

  test("authored fits honor explicit capture retention without sigma fitting", async () => {
    const harness = coordinatorHarness(BrowserFittingCoordinator);
    const result = await harness.coordinator.captureAuthoredFoundation({
      ...plan(), retainCapture: true,
      whiteners: new Map([[0, identityWhitener(2)], [1, identityWhitener(2)]]),
      maxComponents: 1,
      domain: { type: "custom", embed_dim: 1, bounds: null },
      coordinates: new Float64Array([-1, 1]),
      embeddedCoordinates: new Float64Array([-1, 1]),
      intrinsicDimensions: 1, fitSigma: false,
    }, captureSource([
      [0, new Float32Array([1, 10, 3, 14, 8, 20, 12, 24])],
      [1, new Float32Array([2, 4, 4, 8, 10, 20, 14, 28])],
    ]));
    assert.equal(result.captureRetained, true);
    assert.equal(harness.lifecycle.includes("remove"), false);
  });

  test("rejects invalid group boundaries before opening storage", async () => {
    const harness = coordinatorHarness(BrowserFittingCoordinator);
    await assert.rejects(
      harness.coordinator.captureNodeCentroids(
        { ...plan(), groupOffsets: new Uint32Array([0, 3]) },
        captureSource([]),
      ),
      /row count does not match/,
    );
    assert.deepEqual(harness.lifecycle, []);
  });

  test("rejects incomplete affine whitener coverage before opening storage", async () => {
    const harness = coordinatorHarness(BrowserFittingCoordinator);
    await assert.rejects(
      harness.coordinator.captureAffineLayers(
        {
          ...plan(),
          whiteners: new Map([[0, identityWhitener(2)]]),
          maxComponents: 1,
        },
        captureSource([]),
      ),
      /missing a whitener for layer 1/,
    );
    assert.deepEqual(harness.lifecycle, []);
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

function plan() {
  return {
    descriptor: {
      runtimeIdentitySha256: "1".repeat(64),
      contextBindingSha256: "2".repeat(64),
      captureSha256: "3".repeat(64),
      layers: [
        { layer: 0, rows: 4, width: 2, expectedBytes: 32 },
        { layer: 1, rows: 4, width: 2, expectedBytes: 32 },
      ],
    },
    groupOffsets: new Uint32Array([0, 2, 4]),
  };
}

function captureSource(rows) {
  return {
    async capture(sink, _signal, onProgress) {
      let complete = 0;
      const total = rows.reduce((sum, [, values]) => sum + values.length / 2, 0);
      for (const [layer, values] of rows) {
        await sink.appendRows(layer, 0, values);
        complete += values.length / 2;
        onProgress(complete, total);
      }
    },
  };
}

function identityWhitener(columns, firstMean = 0) {
  const mean = new Float64Array(columns);
  mean[0] = firstMean;
  return {
    columns,
    rank: 0,
    ridge: 1,
    mean,
    basis: new Float64Array(),
    eigenvalues: new Float64Array(),
    inverseScales: new Float64Array(),
  };
}

function mapAxes(value) {
  return Object.fromEntries(
    [...value].map(([layer, axes]) => [String(layer), [...axes]]),
  );
}

function close(actual, expected, tolerance) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => {
    assert.ok(Math.abs(value - expected[index]) <= tolerance);
  });
}

function spectralTopology() {
  return {
    operation: "topology",
    winnerName: "spectral",
    fitMode: "spectral",
    intrinsicDimensions: 1,
    periodicDimensions: 0,
    persistentLoops: 0,
    usedFaintCycle: false,
    coordinates: new Float64Array([-1, 1]),
    embeddedCoordinates: new Float64Array([-1, 1]),
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
      eigenvalues: new Float64Array([0.1, 0.8]),
      pickedDimensions: 1,
      gapMagnitude: 0.7,
      bandwidth: 1,
      kNn: 1,
      componentCount: 1,
      heuristicDimensions: 1,
      minDimensions: null,
      pinned: false,
    },
    winnerPlan: rbfPlan([-1, 1], 2, 1),
  };
}

function affineFit(nodeCoordinates, components) {
  const nodeCount = nodeCoordinates.length / components;
  return {
    operation: "affine_fisher",
    nodeCount,
    columns: components,
    components,
    centroidMean: new Float64Array(components),
    mean: new Float64Array(components),
    basis: Float64Array.from({ length: components * components }, (_, index) =>
      index % (components + 1) === 0 ? 1 : 0),
    nodeCoordinates: new Float64Array(nodeCoordinates),
    muCoordinates: new Float64Array(nodeCount * components),
    whitenedGram: new Float64Array(nodeCount * nodeCount),
    neutralCrossGram: new Float64Array(nodeCount),
    explainedVariance: 1,
    mahalanobisShare: 1,
  };
}

function coordinatorHarness(BrowserFittingCoordinator, options = {}) {
  const lifecycle = [];
  const topologyJobs = [];
  const rbfJobs = [];
  const sigmaJobs = [];
  const transferRuns = [];
  const matrices = new Map();
  const widths = new Map();
  const spool = {
    async begin(descriptor) {
      lifecycle.push("begin");
      const progress = descriptor.layers.map((layer) => ({
        ...layer,
        writtenRows: 0,
        writtenBytes: 0,
        checkpointRows: 0,
        checkpointBytes: 0,
        sealed: false,
      }));
      for (const layer of descriptor.layers) {
        matrices.set(layer.layer, []);
        widths.set(layer.layer, layer.width);
      }
      return {
        get progress() { return progress.map((entry) => ({ ...entry })); },
        async appendRows(layer, startRow, values) {
          lifecycle.push(`append:${layer}:${startRow}`);
          const entry = progress.find((item) => item.layer === layer);
          const rowCount = values.length / entry.width;
          matrices.get(layer).push(...values);
          entry.writtenRows += rowCount;
          entry.writtenBytes += values.byteLength;
        },
        async checkpoint(layer) {
          lifecycle.push(`checkpoint:${layer}`);
          const entry = progress.find((item) => item.layer === layer);
          entry.checkpointRows = entry.writtenRows;
          entry.checkpointBytes = entry.writtenBytes;
        },
        async sealLayer(layer) {
          lifecycle.push(`seal:${layer}`);
          progress.find((item) => item.layer === layer).sealed = true;
        },
        async commit() { lifecycle.push("commit"); },
        async rollback() { lifecycle.push("rollback"); },
      };
    },
    async removeCommitted() { lifecycle.push("remove"); return true; },
  };
  const worker = {
    async run(job, runOptions) {
      transferRuns.push({ job, views: [...(runOptions?.transferOwnership ?? [])] });
      if (job.operation === "topology") {
        lifecycle.push("topology");
        topologyJobs.push(job);
        runOptions?.onProgress?.("computing");
        const fitMode = options.topologyFitMode ?? "pca";
        return {
          operation: "topology",
          winnerName: fitMode === "pca" ? "flat-pca" : "spectral",
          fitMode,
          intrinsicDimensions: 1,
          periodicDimensions: 0,
          persistentLoops: 0,
          usedFaintCycle: false,
          coordinates: new Float64Array([-1, 1]),
          embeddedCoordinates: new Float64Array([-1, 1]),
          candidates: [{
            name: fitMode === "pca" ? "flat-pca" : "spectral",
            fitMode,
            dimensions: 1,
            score: 0,
            viable: true,
            reason: "",
          }],
          diagnostics: fitMode === "pca"
            ? {
                kind: "pca",
                perComponentVariance: new Float64Array([1]),
                cumulativeVariance: new Float64Array([1]),
                pickedDimensions: 1,
                threshold: 0.7,
              }
            : {
                kind: "spectral",
                eigenvalues: new Float64Array([0.1, 0.8]),
                pickedDimensions: 1,
                gapMagnitude: 0.7,
                bandwidth: 1,
                kNn: 1,
                componentCount: 1,
                heuristicDimensions: 1,
                minDimensions: null,
                pinned: false,
              },
          winnerPlan: fitMode === "pca" ? null : rbfPlan([-1, 1], 2, 1),
        };
      }
      if (job.operation === "rbf") {
        lifecycle.push("rbf");
        rbfJobs.push(job);
        runOptions?.onProgress?.("computing");
        return {
          operation: "rbf",
          model: {
            inputDimensions: job.inputDimensions,
            outputDimensions: job.outputDimensions,
            nodeCount: job.nodeCount,
            lambda: job.smoothing ?? 0.1,
            effectiveDegreesOfFreedom: job.nodeCount,
            gcv: 0,
            nodes: job.nodes.slice(),
            coordinateOffset: new Float64Array(job.inputDimensions),
            coordinateScale: new Float64Array(job.inputDimensions).fill(1),
            weights: new Float64Array(job.nodeCount * job.outputDimensions),
            polynomial: new Float64Array((job.inputDimensions + 1) * job.outputDimensions),
          },
          evaluated: null,
          queryRows: null,
        };
      }
      if (job.operation === "reduced_covariances") {
        lifecycle.push("covariance");
        runOptions?.onProgress?.("reading");
        return {
          operation: "reduced_covariances",
          nodeCount: job.offsets.length - 1,
          components: job.components,
          covariances: new Float64Array(
            (job.offsets.length - 1) * job.components * job.components,
          ),
        };
      }
      if (job.operation === "sigma_field") {
        lifecycle.push("sigma");
        sigmaJobs.push(job);
        runOptions?.onProgress?.("computing");
        return {
          operation: "sigma_field",
          model: {
            ...job.surface,
            outputDimensions: 1,
            weights: new Float64Array(job.surface.nodeCount),
            polynomial: new Float64Array(job.surface.inputDimensions + 1),
          },
          sigmaMean: 1,
          sigmaMin: 1,
          sigmaMax: 1,
        };
      }
      if (job.operation === "rbf_origin") {
        lifecycle.push("origin");
        runOptions?.onProgress?.("computing");
        return {
          operation: "rbf_origin",
          coordinates: new Float64Array(job.intrinsicDimensions),
          distance: 0,
        };
      }
      if (job.operation === "affine_fisher") {
        lifecycle.push("fit");
        runOptions?.onProgress?.("computing");
        const nodeCount = job.source.rows;
        const columns = job.source.columns;
        const centroidMean = new Float64Array(columns);
        for (let row = 0; row < nodeCount; row += 1) {
          for (let column = 0; column < columns; column += 1) {
            centroidMean[column] += job.source.values[row * columns + column] / nodeCount;
          }
        }
        const nodeCoordinates = new Float64Array(nodeCount);
        const muCoordinates = new Float64Array(nodeCount);
        for (let row = 0; row < nodeCount; row += 1) {
          nodeCoordinates[row] = job.source.values[row * columns] - job.whitener.mean[0];
          muCoordinates[row] = job.source.values[row * columns] - centroidMean[0];
        }
        return {
          operation: "affine_fisher",
          nodeCount,
          columns,
          components: 1,
          centroidMean,
          mean: new Float64Array(columns),
          basis: new Float64Array([1, ...new Array(columns - 1).fill(0)]),
          nodeCoordinates,
          muCoordinates,
          whitenedGram: new Float64Array(nodeCount * nodeCount).fill(
            lifecycle.filter((event) => event === "fit").length,
          ),
          neutralCrossGram: new Float64Array(nodeCount),
          explainedVariance: 1,
          mahalanobisShare: 1,
        };
      }
      lifecycle.push(`run:${job.source.layer}`);
      runOptions?.onProgress?.("reading");
      if (options.failLayer === job.source.layer) throw new Error("pool failed");
      const values = matrices.get(job.source.layer);
      const width = widths.get(job.source.layer);
      const pooled = [];
      for (let group = 0; group < job.offsets.length - 1; group += 1) {
        const start = job.offsets[group];
        const end = job.offsets[group + 1];
        for (let column = 0; column < width; column += 1) {
          let sum = 0;
          for (let row = start; row < end; row += 1) sum += values[row * width + column];
          pooled.push(sum / (end - start));
        }
      }
      return {
        operation: "group_means",
        rows: job.offsets.length - 1,
        columns: width,
        values: new Float64Array(pooled),
      };
    },
  };
  return {
    coordinator: new BrowserFittingCoordinator(spool, worker),
    worker,
    lifecycle,
    topologyJobs,
    rbfJobs,
    sigmaJobs,
    transferRuns,
  };
}

function rbfPlan(rawNodes, nodeCount, inputDimensions) {
  const nodes = new Float64Array(rawNodes);
  return {
    inputDimensions,
    nodeCount,
    nodes,
    coordinateOffset: new Float64Array(inputDimensions),
    coordinateScale: new Float64Array(inputDimensions).fill(1),
    kernel: new Float64Array(nodeCount * nodeCount),
    kernelScale: 1,
    lambdas: Float64Array.from({ length: 40 }, (_, index) => 10 ** (-6 + 9 * index / 39)),
    spectralBasis: new Float64Array(),
    residualRatios: new Float64Array(),
    residualTraces: new Float64Array(40),
  };
}
