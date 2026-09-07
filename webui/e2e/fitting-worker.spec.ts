import { expect, test } from "@playwright/test";
import { resolve } from "node:path";

const devUrl = "http://127.0.0.1:4176";
const fittingClientUrl = `/@fs/${resolve("src/hosted/fitting/fittingWorkerClient.ts")}`;
const activationSpoolUrl = `/@fs/${resolve("src/hosted/fitting/activationSpool.ts")}`;

test("runs the pinned Rust kernel in a stateless browser worker", async ({ page }) => {
  await page.goto(devUrl);
  const result = await page.evaluate(async (clientUrl) => {
    const { BrowserFittingWorkerClient } = await import(clientUrl);
    const stages: string[] = [];
    const client = new BrowserFittingWorkerClient();
    const output = await client.run({
      operation: "center",
      source: {
        kind: "inline",
        values: new Float64Array([1, 2, 3, 4]),
        rows: 2,
        columns: 2,
      },
    }, { onProgress: (stage: string) => stages.push(stage) });
    if (output.operation !== "center") throw new Error("unexpected fitting result");
    return {
      stages,
      rows: output.rows,
      columns: output.columns,
      mean: [...output.mean],
      centered: [...output.centered],
    };
  }, fittingClientUrl);

  expect(result).toEqual({
    stages: ["loading", "computing"],
    rows: 2,
    columns: 2,
    mean: [2, 3],
    centered: [-1, -1, 1, 1],
  });
});

test("reads a committed activation layer inside the fitting worker", async ({ page }) => {
  await page.goto(devUrl);
  const result = await page.evaluate(async ({ spoolUrl, clientUrl }) => {
    const [{ BrowserActivationSpool }, { BrowserFittingWorkerClient }] = await Promise.all([
      import(spoolUrl),
      import(clientUrl),
    ]);
    const spool = new BrowserActivationSpool();
    await spool.clear();
    const identity = {
      runtimeIdentitySha256: "1".repeat(64),
      contextBindingSha256: "2".repeat(64),
      captureSha256: "3".repeat(64),
    };
    const writer = await spool.begin({
      ...identity,
      layers: [{ layer: 0, rows: 4, width: 2, expectedBytes: 32 }],
    });
    await writer.appendRows(0, 0, new Float32Array([1, 2, 2, 4, 3, 6, 4, 8]));
    await writer.checkpoint(0);
    await writer.sealLayer(0);
    await writer.commit();

    const stages: string[] = [];
    try {
      const output = await new BrowserFittingWorkerClient().run({
        operation: "pca",
        source: { kind: "activation_spool", identity, layer: 0 },
        maxComponents: 2,
        varianceThreshold: 0.99,
      }, { onProgress: (stage: string) => stages.push(stage) });
      if (output.operation !== "pca") throw new Error("unexpected fitting result");
      return {
        stages,
        rows: output.rows,
        columns: output.columns,
        components: output.components,
        mean: [...output.mean],
        scoreCount: output.scores.length,
      };
    } finally {
      await spool.clear();
    }
  }, { spoolUrl: activationSpoolUrl, clientUrl: fittingClientUrl });

  expect(result).toEqual({
    stages: ["loading", "reading", "computing"],
    rows: 4,
    columns: 2,
    components: 1,
    mean: [2.5, 5],
    scoreCount: 4,
  });
});

test("pools contiguous activation rows into exact node centroids", async ({ page }) => {
  await page.goto(devUrl);
  const result = await page.evaluate(async ({ spoolUrl, clientUrl }) => {
    const [{ BrowserActivationSpool }, { BrowserFittingWorkerClient }] = await Promise.all([
      import(spoolUrl),
      import(clientUrl),
    ]);
    const spool = new BrowserActivationSpool();
    await spool.clear();
    const identity = {
      runtimeIdentitySha256: "4".repeat(64),
      contextBindingSha256: "5".repeat(64),
      captureSha256: "6".repeat(64),
    };
    const writer = await spool.begin({
      ...identity,
      layers: [{ layer: 2, rows: 5, width: 2, expectedBytes: 40 }],
    });
    await writer.appendRows(2, 0, new Float32Array([
      1, 10,
      3, 14,
      6, 20,
      9, 23,
      12, 26,
    ]));
    await writer.checkpoint(2);
    await writer.sealLayer(2);
    await writer.commit();
    try {
      const output = await new BrowserFittingWorkerClient().run({
        operation: "group_means",
        source: { kind: "activation_spool", identity, layer: 2 },
        offsets: new Uint32Array([0, 2, 5]),
      });
      if (output.operation !== "group_means") throw new Error("unexpected fitting result");
      return { rows: output.rows, columns: output.columns, values: [...output.values] };
    } finally {
      await spool.clear();
    }
  }, { spoolUrl: activationSpoolUrl, clientUrl: fittingClientUrl });

  expect(result).toEqual({ rows: 2, columns: 2, values: [2, 12, 9, 23] });
});

test("streams reduced covariances and fits a curved sigma field", async ({ page }) => {
  await page.goto(devUrl);
  const result = await page.evaluate(async ({ spoolUrl, clientUrl }) => {
    const [{ BrowserActivationSpool }, { BrowserFittingWorkerClient }] = await Promise.all([
      import(spoolUrl),
      import(clientUrl),
    ]);
    const spool = new BrowserActivationSpool();
    await spool.clear();
    const identity = {
      runtimeIdentitySha256: "7".repeat(64),
      contextBindingSha256: "8".repeat(64),
      captureSha256: "9".repeat(64),
    };
    const writer = await spool.begin({
      ...identity,
      layers: [{ layer: 3, rows: 6, width: 2, expectedBytes: 48 }],
    });
    await writer.appendRows(3, 0, new Float32Array([
      -2, -1, 0, 1,
      -1, -2, 1, 2,
      0, -3, 2, 3,
    ]));
    await writer.checkpoint(3);
    await writer.sealLayer(3);
    await writer.commit();
    const client = new BrowserFittingWorkerClient();
    try {
      const covariance = await client.run({
        operation: "reduced_covariances",
        source: { kind: "activation_spool", identity, layer: 3 },
        offsets: new Uint32Array([0, 2, 4, 6]),
        mean: new Float64Array([0, 0]),
        basis: new Float64Array([1, 0, 0, 1]),
        components: 2,
      });
      if (covariance.operation !== "reduced_covariances") {
        throw new Error("unexpected covariance result");
      }
      const surface = await client.run({
        operation: "rbf",
        nodes: new Float64Array([-1, 0, 1]),
        nodeCount: 3,
        inputDimensions: 1,
        values: new Float64Array([-1, 0, 0, 0, 1, 0]),
        outputDimensions: 2,
        smoothing: 0,
      });
      if (surface.operation !== "rbf") throw new Error("unexpected RBF result");
      const sigma = await client.run({
        operation: "sigma_field",
        surface: surface.model,
        covariances: covariance.covariances,
        coordinates: new Float64Array([-1, 0, 1]),
        embeddedCoordinates: new Float64Array([-1, 0, 1]),
        intrinsicDimensions: 1,
        periodicDimensions: 0,
        smoothing: 0,
        floorFraction: 1e-3,
      });
      if (sigma.operation !== "sigma_field") throw new Error("unexpected sigma result");
      const origin = await client.run({
        operation: "rbf_origin",
        surface: surface.model,
        coordinates: new Float64Array([-1, 0, 1]),
        embeddedCoordinates: new Float64Array([-1, 0, 1]),
        intrinsicDimensions: 1,
        periodicDimensions: 0,
        maxIterations: 12,
        restartCount: 3,
        damping: 1e-3,
      });
      if (origin.operation !== "rbf_origin") throw new Error("unexpected origin result");
      return {
        covariances: [...covariance.covariances],
        sigma: [sigma.sigmaMean, sigma.sigmaMin, sigma.sigmaMax],
        modelShape: [sigma.model.nodeCount, sigma.model.inputDimensions, sigma.model.outputDimensions],
        origin: [...origin.coordinates],
      };
    } finally {
      await spool.clear();
    }
  }, { spoolUrl: activationSpoolUrl, clientUrl: fittingClientUrl });

  expect(result.covariances).toEqual([
    2, 2, 2, 2,
    2, 4, 4, 8,
    2, 6, 6, 18,
  ]);
  expect(result.sigma[0]).toBeGreaterThan(0);
  expect(result.sigma[1]).toBeGreaterThan(0);
  expect(result.sigma[2]).toBeGreaterThanOrEqual(result.sigma[1]);
  expect(result.modelShape).toEqual([3, 1, 1]);
  expect(result.origin[0]).toBeCloseTo(0, 6);
});

test("serializes an unviable topology candidate without breaking the worker protocol", async ({ page }) => {
  await page.goto(devUrl);
  const candidate = await page.evaluate(async (clientUrl) => {
    const { BrowserFittingWorkerClient } = await import(clientUrl);
    const output = await new BrowserFittingWorkerClient().run({
      operation: "topology",
      consensusGram: new Float64Array([1, -1, -1, 1]),
      nodeCount: 2,
      targets: new Float64Array([-1, 1]),
      targetOffsets: new Uint32Array([0, 2]),
      maxDimensions: 2,
      varianceThreshold: 0.7,
      requestedFitMode: "auto",
      persistenceFraction: 0.5,
    });
    if (output.operation !== "topology") throw new Error("unexpected fitting result");
    return output.candidates.find((item) => item.name === "spectral");
  }, fittingClientUrl);

  expect(candidate).toMatchObject({
    name: "spectral",
    dimensions: 1,
    score: null,
    viable: false,
  });
});

test("reuses a fixed-smoothed topology plan across variable output widths", async ({ page }) => {
  await page.goto(devUrl);
  const result = await page.evaluate(async (clientUrl) => {
    const { BrowserFittingWorkerClient } = await import(clientUrl);
    const client = new BrowserFittingWorkerClient();
    const nodeCount = 16;
    const targets = new Float64Array(nodeCount * 2);
    for (let index = 0; index < nodeCount; index += 1) {
      const t = index / (nodeCount - 1);
      targets[2 * index] = Math.sin(5 * t);
      targets[2 * index + 1] = Math.cos(7 * t);
    }
    const means = [0, 1].map((column) => {
      let sum = 0;
      for (let row = 0; row < nodeCount; row += 1) sum += targets[2 * row + column];
      return sum / nodeCount;
    });
    const gram = new Float64Array(nodeCount * nodeCount);
    for (let left = 0; left < nodeCount; left += 1) {
      for (let right = 0; right < nodeCount; right += 1) {
        gram[left * nodeCount + right] =
          (targets[2 * left] - means[0]) * (targets[2 * right] - means[0]) +
          (targets[2 * left + 1] - means[1]) * (targets[2 * right + 1] - means[1]);
      }
    }
    const topology = await client.run({
      operation: "topology",
      consensusGram: gram,
      nodeCount,
      targets,
      targetOffsets: new Uint32Array([0, targets.length]),
      maxDimensions: 2,
      varianceThreshold: 0.7,
      requestedFitMode: "spectral",
      minDimensions: 1,
      persistenceFraction: 0.5,
      smoothing: 0.25,
    });
    if (topology.operation !== "topology" || topology.winnerPlan === null) {
      throw new Error("spectral topology returned no reusable plan");
    }
    const nodes = topology.embeddedCoordinates;
    const widths = [];
    let maxDifference = 0;
    for (const [values, outputDimensions] of [
      [targets, 2],
      [Float64Array.from({ length: nodeCount }, (_, index) => targets[2 * index]), 1],
    ] as const) {
      const planned = await client.run({
        operation: "rbf",
        nodes,
        nodeCount,
        inputDimensions: topology.winnerPlan.inputDimensions,
        values,
        outputDimensions,
        smoothing: 0.25,
        plan: topology.winnerPlan,
        queries: nodes,
        queryRows: nodeCount,
      });
      const standalone = await client.run({
        operation: "rbf",
        nodes,
        nodeCount,
        inputDimensions: topology.winnerPlan.inputDimensions,
        values,
        outputDimensions,
        smoothing: 0.25,
        queries: nodes,
        queryRows: nodeCount,
      });
      if (
        planned.operation !== "rbf" || standalone.operation !== "rbf" ||
        planned.evaluated === null || standalone.evaluated === null
      ) throw new Error("RBF evaluation is missing");
      widths.push(planned.model.outputDimensions);
      for (let index = 0; index < planned.evaluated.length; index += 1) {
        maxDifference = Math.max(
          maxDifference,
          Math.abs(planned.evaluated[index] - standalone.evaluated[index]),
        );
      }
    }
    return {
      widths,
      maxDifference,
      diagnostics: topology.diagnostics,
      planDimensions: [topology.winnerPlan.nodeCount, topology.winnerPlan.inputDimensions],
      score: topology.candidates.find((candidate) => candidate.name === "spectral")?.score,
    };
  }, fittingClientUrl);

  expect(result.widths).toEqual([2, 1]);
  expect(result.maxDifference).toBeLessThanOrEqual(1e-12);
  expect(result.diagnostics.kind).toBe("spectral");
  expect(result.diagnostics).toMatchObject({
    componentCount: 1,
    kNn: 5,
    minDimensions: 1,
  });
  expect(result.planDimensions[0]).toBe(16);
  expect(result.score).toBeGreaterThanOrEqual(0);
});

test("rejects a topology job without an explicit fit mode", async ({ page }) => {
  await page.goto(devUrl);
  const failure = await page.evaluate(async (clientUrl) => {
    const { BrowserFittingWorkerClient } = await import(clientUrl);
    try {
      await new BrowserFittingWorkerClient().run({
        operation: "topology",
        consensusGram: new Float64Array([1, -1, -1, 1]),
        nodeCount: 2,
        targets: new Float64Array([-1, 1]),
        targetOffsets: new Uint32Array([0, 2]),
        maxDimensions: 2,
        varianceThreshold: 0.7,
        persistenceFraction: 0.5,
      } as never);
      return null;
    } catch (error) {
      return {
        code: (error as { code?: string }).code,
        message: error instanceof Error ? error.message : String(error),
      };
    }
  }, fittingClientUrl);

  expect(failure?.code).toBe("FITTING_INVALID_REQUEST");
  expect(failure?.message).toContain("pca, spectral, or auto");
});

test("rejects malformed fitting matrices without crashing the page", async ({ page }) => {
  await page.goto(devUrl);
  const failure = await page.evaluate(async (clientUrl) => {
    const { BrowserFittingWorkerClient } = await import(clientUrl);
    try {
      await new BrowserFittingWorkerClient().run({
        operation: "center",
        source: {
          kind: "inline",
          values: new Float64Array([1, 2, 3]),
          rows: 2,
          columns: 2,
        },
      });
      return null;
    } catch (error) {
      return {
        code: (error as { code?: string }).code,
        message: error instanceof Error ? error.message : String(error),
      };
    }
  }, fittingClientUrl);

  expect(failure?.code).toBe("FITTING_INVALID_MATRIX");
  expect(failure?.message).toContain("expected 4");
});
