#!/usr/bin/env node

import { createServer } from "node:http";
import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { basename, extname, resolve } from "node:path";
import { chromium } from "playwright";

const options = parseArguments(process.argv.slice(2));
const webllmPath = resolve(options.webllm);
const modelDirectory = resolve(options.model);
const modelLibraryPath = resolve(options.library);
await requireFile(webllmPath, "WebLLM bundle");
await requireFile(modelLibraryPath, "model library");
const modelConfig = JSON.parse(
  await readFile(resolve(modelDirectory, "mlc-chat-config.json"), "utf8"),
);
const modelShape = modelConfig.model_config?.text_config ?? modelConfig.model_config;
if (
  options.jLensLayers !== null &&
  options.jLensLayers > modelShape.num_hidden_layers
) {
  throw new Error(
    `--jlens-layers cannot exceed ${modelShape.num_hidden_layers}`,
  );
}
const tensorCache = JSON.parse(
  await readFile(resolve(modelDirectory, "tensor-cache.json"), "utf8"),
);
const modelArtifactNames = [
  "mlc-chat-config.json",
  "tensor-cache.json",
  ...modelConfig.tokenizer_files,
  ...tensorCache.records.map((record) => record.dataPath),
].filter((value, index, values) => values.indexOf(value) === index);
for (const file of modelArtifactNames) {
  await requireFile(resolve(modelDirectory, file), `model ${file}`);
}
const modelLibrary = await readFile(modelLibraryPath);
if (!modelLibrary.subarray(0, 4).equals(Buffer.from([0, 97, 115, 109]))) {
  throw new Error("model library is not WebAssembly");
}

const requestCounts = new Map();

const server = createServer(async (request, response) => {
  try {
    const url = new URL(request.url ?? "/", "http://127.0.0.1");
    requestCounts.set(url.pathname, (requestCounts.get(url.pathname) ?? 0) + 1);
    if (url.pathname === "/")
      return send(response, pageHtml(), "text/html; charset=utf-8");
    if (url.pathname === "/webllm.js") {
      return send(
        response,
        await readFile(webllmPath),
        "text/javascript; charset=utf-8",
      );
    }
    if (url.pathname === "/worker.js") {
      return send(
        response,
        workerSource(
          options.verifiedArtifacts,
          options.selectedAdapter,
          modelArtifactNames,
        ),
        "text/javascript; charset=utf-8",
      );
    }
    if (url.pathname === "/model-lib.wasm") {
      return send(response, modelLibrary, "application/wasm");
    }
    if (url.pathname.startsWith("/model/")) {
      const name = basename(url.pathname);
      if (!modelArtifactNames.includes(name)) {
        return send(response, "not found", "text/plain", 404);
      }
      return send(
        response,
        await readFile(resolve(modelDirectory, name)),
        contentType(name),
      );
    }
    return send(response, "not found", "text/plain", 404);
  } catch (error) {
    return send(
      response,
      error instanceof Error ? error.message : String(error),
      "text/plain",
      500,
    );
  }
});

await new Promise((resolvePromise, reject) => {
  server.once("error", reject);
  server.listen(0, "127.0.0.1", resolvePromise);
});
const address = server.address();
if (address === null || typeof address === "string")
  throw new Error("failed to bind test server");
const origin = `http://127.0.0.1:${address.port}`;

const browser = await chromium.launch({
  headless: !options.headed,
  ...(options.browserChannel ? { channel: options.browserChannel } : {}),
  args: [
    "--enable-unsafe-webgpu",
    "--use-angle=metal",
    "--disable-gpu-sandbox",
  ],
});
try {
  const page = await browser.newPage();
  const webGpuErrors = [];
  page.setDefaultTimeout(120_000);
  page.on("console", (message) => {
    const value = message.text();
    if (value.startsWith("drowse feasibility:")) {
      process.stderr.write(`${value}\n`);
      return;
    }
    if (message.type() !== "error") return;
    if (value.includes("WebGPU error was not captured"))
      webGpuErrors.push(value);
    process.stderr.write(`browser console: ${value}\n`);
  });
  await page.goto(origin, { waitUntil: "load" });
  const result = await page.evaluate(
    async ({
      architecture,
      promptTokens,
      quantization,
      reloads,
      useWorker,
      verifiedArtifacts,
      selectedAdapter,
      hiddenSize,
      layerCount,
      jLensLayerCount,
      vocabSize,
      captureSpecialTokenIds,
    }) => {
      if (!navigator.gpu)
        throw new Error("WebGPU is unavailable in the test browser");
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter === null) throw new Error("no WebGPU adapter was returned");
      const module = await import("/webllm.js");
      const engineConfig = {
        logLevel: "WARN",
      };
      if (!verifiedArtifacts) {
        engineConfig.appConfig = {
          cacheBackend: "indexeddb",
          model_list: [
            {
              model: `${location.origin}/model/`,
              model_id: "drowse-tiny",
              model_lib: `${location.origin}/model-lib.wasm`,
            },
          ],
        };
      }
      const runtimeWorker = useWorker
        ? new Worker("/worker.js", { type: "module" })
        : undefined;
      const engine = runtimeWorker
        ? new module.WebWorkerMLCEngine(runtimeWorker, engineConfig)
        : new module.MLCEngine(engineConfig);
      console.info("drowse feasibility: loading runtime");
      await engine.reload("drowse-tiny");
      console.info("drowse feasibility: runtime loaded");
      console.info("drowse feasibility: rank-one capability");
      const supported = await engine.supportsDrowseRankOneHooks("drowse-tiny");
      if (!supported)
        throw new Error(
          "compiled library did not expose Drowse hooks to WebLLM",
        );
      console.info("drowse feasibility: structured capability");
      const structuredSupported =
        await engine.supportsDrowseStructuredHooks("drowse-tiny");
      console.info("drowse feasibility: curved capability");
      const curvedSupported =
        await engine.supportsDrowseCurvedHooks("drowse-tiny");
      if (!structuredSupported || !curvedSupported) {
        throw new Error(
          `compiled library did not expose structured and curved Drowse hooks to WebLLM: structured=${structuredSupported}, curved=${curvedSupported}`,
        );
      }
      console.info("drowse feasibility: capture capability");
      const captureSupported =
        await engine.supportsDrowseResidualCapture("drowse-tiny");
      if (!captureSupported) {
        throw new Error(
          "compiled library did not expose Drowse residual capture to WebLLM",
        );
      }
      const prompt = Array.from({ length: promptTokens }, () => "hello").join(
        " ",
      );
      console.info("drowse feasibility: pre-capture baseline generation");
      await engine.completions.create({
        model: "drowse-tiny",
        prompt,
        max_tokens: 1,
        seed: 1,
        temperature: 0,
      });
      console.info("drowse feasibility: pre-capture baseline returned");
      if ((await engine.readDrowseMeasurements("drowse-tiny")) !== undefined) {
        throw new Error(
          "ordinary pre-capture generation unexpectedly produced Drowse measurements",
        );
      }
      console.info("drowse feasibility: prepare capture rows");
      const preparedCaptureRows = await engine.prepareDrowseCaptureRows(
        [
          {
            system: "",
            messages: [
              { role: "user", content: "hello" },
              { role: "assistant", content: "world" },
            ],
          },
        ],
        captureSpecialTokenIds,
        "drowse-tiny",
      );
      if (
        preparedCaptureRows.length !== 1 ||
        preparedCaptureRows[0].inputIds.length === 0 ||
        preparedCaptureRows[0].position < 0 ||
        preparedCaptureRows[0].position >=
          preparedCaptureRows[0].inputIds.length ||
        captureSpecialTokenIds.includes(
          preparedCaptureRows[0].inputIds[preparedCaptureRows[0].position],
        )
      ) {
        throw new Error(
          "Drowse capture row preparation returned an invalid pool position",
        );
      }
      const captureInput = preparedCaptureRows[0].inputIds;
      const capturePositions = [preparedCaptureRows[0].position];
      console.info("drowse feasibility: capture residuals");
      const captureBeforeSteering = await engine.captureDrowseResiduals(
        captureInput,
        capturePositions,
        "drowse-tiny",
      );
      if (
        captureBeforeSteering.layerCount !== layerCount ||
        captureBeforeSteering.positionCount !== capturePositions.length ||
        captureBeforeSteering.hiddenSize !== hiddenSize ||
        captureBeforeSteering.values.length !==
          layerCount * capturePositions.length * hiddenSize ||
        captureBeforeSteering.values.some((value) => !Number.isFinite(value))
      ) {
        const firstNonFinite = captureBeforeSteering.values.findIndex(
          (value) => !Number.isFinite(value),
        );
        throw new Error(
          "Drowse residual capture returned an invalid tensor envelope: " +
            JSON.stringify({
              layerCount: captureBeforeSteering.layerCount,
              positionCount: captureBeforeSteering.positionCount,
              hiddenSize: captureBeforeSteering.hiddenSize,
              values: captureBeforeSteering.values.length,
              firstNonFinite,
              firstNonFiniteValue: firstNonFinite < 0
                ? null
                : String(captureBeforeSteering.values[firstNonFinite]),
            }),
        );
      }
      if (layerCount === 2 && hiddenSize === 64) {
        for (
          let layer = 0;
          layer < captureBeforeSteering.layerCount;
          layer += 1
        ) {
          const coordinate =
            captureBeforeSteering.values[
              layer *
                captureBeforeSteering.positionCount *
                captureBeforeSteering.hiddenSize
            ];
          if (Math.abs(coordinate - 0.5) > 1e-3) {
            throw new Error(
              `unexpected fixture residual coordinate: ${coordinate}`,
            );
          }
        }
      }
      console.info("drowse feasibility: post-capture baseline generation");
      const baseline = await engine.completions.create({
        model: "drowse-tiny",
        prompt,
        max_tokens: 1,
        seed: 1,
        temperature: 0,
      });
      console.info("drowse feasibility: post-capture baseline returned");
      if ((await engine.readDrowseMeasurements("drowse-tiny")) !== undefined) {
        throw new Error(
          "ordinary generation unexpectedly produced Drowse measurements",
        );
      }

      const basis = new Float32Array(hiddenSize * layerCount);
      const probeBasis = new Float32Array(hiddenSize * layerCount);
      for (let layer = 0; layer < layerCount; layer += 1) {
        basis[layer * hiddenSize] = 1;
        probeBasis[layer * hiddenSize] = 1;
      }
      const disabled = new Uint32Array(layerCount);
      await engine.setDrowseRankOneProgram(
        {
          hookAbi: module.DROWSE_HOOK_ABI,
          hiddenSize,
          layerCount,
          enabled: disabled,
          basis,
          neutral: new Float32Array(hiddenSize * layerCount),
          target: new Float32Array(layerCount),
          along: new Float32Array(layerCount),
          collapse: new Float32Array(layerCount),
          probeBasis,
          probeNeutral: new Float32Array(hiddenSize * layerCount),
        },
        "drowse-tiny",
      );
      await engine.completions.create({
        model: "drowse-tiny",
        prompt,
        max_tokens: 1,
        seed: 1,
        temperature: 0,
      });
      const probeBaseline = await engine.readDrowseMeasurements("drowse-tiny");
      if (
        probeBaseline === undefined ||
        probeBaseline.length !== layerCount ||
        probeBaseline.some((value) => !Number.isFinite(value))
      ) {
        throw new Error("probe-only generation returned invalid measurements");
      }
      const enabled = new Uint32Array(layerCount);
      const target = new Float32Array(layerCount);
      const along = new Float32Array(layerCount);
      const collapse = new Float32Array(layerCount);
      enabled[0] = 1;
      target[0] = probeBaseline[0] + 10;
      along[0] = 1;
      collapse[0] = 1;
      await engine.setDrowseRankOneProgram(
        {
          hookAbi: module.DROWSE_HOOK_ABI,
          hiddenSize,
          layerCount,
          enabled,
          basis,
          neutral: new Float32Array(hiddenSize * layerCount),
          target,
          along,
          collapse,
          probeBasis,
          probeNeutral: new Float32Array(hiddenSize * layerCount),
        },
        "drowse-tiny",
      );
      const steered = await engine.completions.create({
        model: "drowse-tiny",
        prompt,
        max_tokens: 1,
        seed: 1,
        temperature: 0,
      });
      const measurements = await engine.readDrowseMeasurements("drowse-tiny");
      if (measurements === undefined || measurements.length !== layerCount) {
        throw new Error("Drowse probe measurements were not returned");
      }
      const measurementTolerance =
        layerCount === 2 && hiddenSize === 64
          ? 1e-3
          : Math.max(1e-2, Math.abs(target[0]) * 1e-3);
      if (
        measurements.some((measurement) => !Number.isFinite(measurement)) ||
        Math.abs(measurements[0] - target[0]) > measurementTolerance
      ) {
        throw new Error(
          `unexpected steered Drowse probe measurement: actual=${measurements[0]}, ` +
            `baseline=${probeBaseline[0]}, target=${target[0]}`,
        );
      }
      if (Math.abs(measurements[0] - probeBaseline[0]) < 5) {
        throw new Error(
          "Drowse rank-one program did not materially change the residual coordinate",
        );
      }
      const captureDuringSteering = await engine.captureDrowseResiduals(
        captureInput,
        capturePositions,
        "drowse-tiny",
      );
      if (
        captureDuringSteering.values.length !==
          captureBeforeSteering.values.length ||
        captureDuringSteering.values.some(
          (value, index) => value !== captureBeforeSteering.values[index],
        )
      ) {
        throw new Error(
          "Drowse fitting capture was contaminated by live steering",
        );
      }
      await engine.setDrowseRankOneProgram(
        {
          hookAbi: module.DROWSE_HOOK_ABI,
          hiddenSize,
          layerCount,
          enabled: disabled,
          basis,
          neutral: new Float32Array(hiddenSize * layerCount),
          target: new Float32Array(layerCount),
          along: new Float32Array(layerCount),
          collapse: new Float32Array(layerCount),
          probeBasis,
          probeNeutral: new Float32Array(hiddenSize * layerCount),
        },
        "drowse-tiny",
      );
      await engine.completions.create({
        model: "drowse-tiny",
        prompt: "hello",
        max_tokens: 1,
        seed: 1,
        temperature: 0,
      });
      const affineBaselineMeasurements =
        await engine.readDrowseMeasurements("drowse-tiny");
      if (affineBaselineMeasurements === undefined) {
        throw new Error(
          "probe-only affine baseline did not return measurements",
        );
      }
      await engine.setDrowseRankOneProgram(
        {
          hookAbi: module.DROWSE_HOOK_ABI,
          hiddenSize,
          layerCount,
          enabled,
          basis,
          neutral: new Float32Array(hiddenSize * layerCount),
          target: Float32Array.from({ length: layerCount }, (_, index) =>
            index === 0 ? 100 : 0,
          ),
          along,
          collapse: new Float32Array(layerCount),
          probeBasis,
          probeNeutral: new Float32Array(hiddenSize * layerCount),
        },
        "drowse-tiny",
      );
      await engine.completions.create({
        model: "drowse-tiny",
        prompt: "hello",
        max_tokens: 1,
        seed: 1,
        temperature: 0,
      });
      const uncappedMeasurements =
        await engine.readDrowseMeasurements("drowse-tiny");
      const uncappedTolerance =
        layerCount === 2 && hiddenSize === 64 ? 1e-3 : 0.1;
      if (
        uncappedMeasurements === undefined ||
        Math.abs(
          uncappedMeasurements[0] - affineBaselineMeasurements[0] - 100,
        ) > uncappedTolerance
      ) {
        throw new Error(
          `affine steering was unexpectedly norm-capped: ${uncappedMeasurements?.[0]}`,
        );
      }

      const groups = 4;
      const rank = 8;
      const probes = 8;
      const curves = 4;
      const curveNodes = 32;
      const intrinsic = 4;
      const embedded = 8;
      const structuredProgram = () => {
        const curveSlots = layerCount * curves;
        const geometryProbes = module.DROWSE_STRUCTURED_MAX_GEOMETRY_PROBES;
        const geometrySlots = layerCount * geometryProbes;
        const program = {
          hookAbi: module.DROWSE_HOOK_ABI,
          format: module.DROWSE_STRUCTURED_HOOK_FORMAT,
          hiddenSize,
          layerCount,
          affineActive: new Uint32Array(layerCount * groups),
          affineBasis: new Float32Array(
            layerCount * groups * rank * hiddenSize,
          ),
          affineNeutral: new Float32Array(layerCount * groups * hiddenSize),
          affineTarget: new Float32Array(layerCount * groups * rank),
          affineAlong: new Float32Array(layerCount * groups),
          affineKappa: new Float32Array(layerCount * groups * rank),
          probeKind: new Uint32Array(layerCount * probes),
          probeDirection: new Float32Array(layerCount * probes * hiddenSize),
          probeBias: new Float32Array(layerCount * probes),
          probeThreshold: new Float32Array(layerCount * probes),
          curveActive: new Uint32Array(curveSlots),
          curveDomainKind: new Uint32Array(curveSlots),
          curveRank: new Uint32Array(curveSlots),
          curveIntrinsicDim: new Uint32Array(curveSlots),
          curveEmbedDim: new Uint32Array(curveSlots),
          curveNodeCount: new Uint32Array(curveSlots),
          curveBasis: new Float32Array(curveSlots * rank * hiddenSize),
          curveNeutral: new Float32Array(curveSlots * hiddenSize),
          curveNodeParameters: new Float32Array(
            curveSlots * curveNodes * embedded,
          ),
          curveRbfWeights: new Float32Array(curveSlots * curveNodes * rank),
          curvePolynomial: new Float32Array(curveSlots * (embedded + 1) * rank),
          curveCoordinateOffset: new Float32Array(curveSlots * embedded),
          curveCoordinateScale: new Float32Array(curveSlots * embedded),
          curveOrigin: new Float32Array(curveSlots * intrinsic),
          curveTarget: new Float32Array(curveSlots * intrinsic),
          curveAlong: new Float32Array(curveSlots),
          curveOnto: new Float32Array(curveSlots),
          curveBounds: new Float32Array(curveSlots * intrinsic * 2),
          curveAxisPeriodic: new Uint32Array(curveSlots * intrinsic),
          curveAxisPeriod: new Float32Array(curveSlots * intrinsic),
          curveSigmaPresent: new Uint32Array(curveSlots),
          curveSigmaRbfWeights: new Float32Array(curveSlots * curveNodes),
          curveSigmaPolynomial: new Float32Array(curveSlots * (embedded + 1)),
          curveDamping: new Float32Array(curveSlots),
          whitenerRank: new Uint32Array(layerCount),
          whitenerRidge: new Float32Array(layerCount),
          whitenerBasis: new Float32Array(
            layerCount *
              module.DROWSE_STRUCTURED_MAX_WHITENER_RANK *
              hiddenSize,
          ),
          whitenerCorrection: new Float32Array(
            layerCount * module.DROWSE_STRUCTURED_MAX_WHITENER_RANK,
          ),
          geometryActive: new Uint32Array(geometrySlots),
          geometryKind: new Uint32Array(geometrySlots),
          geometryRank: new Uint32Array(geometrySlots),
          geometryIntrinsicDim: new Uint32Array(geometrySlots),
          geometryCandidateCount: new Uint32Array(geometrySlots),
          geometryCurveNodeCount: new Uint32Array(geometrySlots),
          geometryDomainKind: new Uint32Array(geometrySlots),
          geometryMean: new Float32Array(geometrySlots * hiddenSize),
          geometryInverseMean: new Float32Array(geometrySlots * hiddenSize),
          geometryBasis: new Float32Array(geometrySlots * rank * hiddenSize),
          geometryGramInverse: new Float32Array(geometrySlots * rank * rank),
          geometryCholesky: new Float32Array(geometrySlots * rank * rank),
          geometryNodeWhite: new Float32Array(
            geometrySlots *
              rank *
              module.DROWSE_STRUCTURED_MAX_GEOMETRY_CANDIDATES,
          ),
          geometryCoordMap: new Float32Array(geometrySlots * intrinsic * rank),
          geometryCoordBias: new Float32Array(geometrySlots * intrinsic),
          geometryCurveParameters: new Float32Array(
            geometrySlots * module.DROWSE_STRUCTURED_CURVE_PARAMETER_STRIDE,
          ),
          geometryCurveNodeCoords: new Float32Array(
            geometrySlots * curveNodes * intrinsic,
          ),
          geometryCurveNodeValues: new Float32Array(
            geometrySlots * curveNodes * rank,
          ),
          geometryFeet: new Float32Array(geometrySlots * intrinsic),
        };
        program.curveCoordinateScale.fill(1);
        program.curveAxisPeriod.fill(1);
        program.curveDamping.fill(1e-3);
        for (let layer = 0; layer < layerCount; layer += 1) {
          program.probeKind[layer * probes] = 1;
          program.probeDirection[layer * probes * hiddenSize] = 1;
          program.probeKind[layer * probes + 1] = 2;
          program.probeDirection[(layer * probes + 1) * hiddenSize] = 1;
          program.probeBias[layer * probes + 1] = -0.25;
          program.probeKind[layer * probes + 2] = 1;
          program.probeDirection[(layer * probes + 2) * hiddenSize + 1] = 1;
        }
        for (let slot = 0; slot < curveSlots; slot += 1) {
          for (let axis = 0; axis < intrinsic; axis += 1) {
            const offset = (slot * intrinsic + axis) * 2;
            program.curveBounds[offset] = -100;
            program.curveBounds[offset + 1] = 100;
          }
        }
        return program;
      };
      const runOne = () =>
        engine.completions.create({
          model: "drowse-tiny",
          prompt: "hello",
          max_tokens: 1,
          seed: 1,
          temperature: 0,
        });
      const probeOnlyProgram = structuredProgram();
      console.info("drowse feasibility: structured probes");
      await engine.setDrowseStructuredProgram(probeOnlyProgram, "drowse-tiny");
      await runOne();
      const structuredBaseline =
        await engine.readDrowseMeasurements("drowse-tiny");
      if (
        structuredBaseline === undefined ||
        structuredBaseline.length !== layerCount * probes ||
        structuredBaseline.some((value) => !Number.isFinite(value))
      ) {
        throw new Error(
          "structured probe-only generation returned invalid measurements",
        );
      }
      const saeBaseline = Math.max(0, structuredBaseline[0] - 0.25);
      if (Math.abs(structuredBaseline[1] - saeBaseline) > 1e-4) {
        throw new Error(
          "SAE feature probe did not apply its exact ReLU encoder equation",
        );
      }
      const multiaxisProgram = structuredProgram();
      multiaxisProgram.affineActive[0] = 1;
      multiaxisProgram.affineBasis[0] = 1;
      multiaxisProgram.affineBasis[hiddenSize + 1] = 1;
      multiaxisProgram.affineTarget[0] = 2;
      multiaxisProgram.affineTarget[1] = 3;
      multiaxisProgram.affineAlong[0] = 0.5;
      console.info("drowse feasibility: structured affine");
      await engine.setDrowseStructuredProgram(multiaxisProgram, "drowse-tiny");
      await engine.updateDrowseStructuredControls(
        multiaxisProgram.affineActive,
        new Uint32Array(layerCount * curves),
        "drowse-tiny",
      );
      await runOne();
      const multiaxisMeasurements =
        await engine.readDrowseMeasurements("drowse-tiny");
      const structuredDelta =
        (multiaxisMeasurements?.[0] ?? Number.NaN) - structuredBaseline[0];
      if (
        multiaxisMeasurements === undefined ||
        Math.abs(structuredDelta - 1) > 1e-3 ||
        Math.abs(
          multiaxisMeasurements[1] -
            Math.max(0, multiaxisMeasurements[0] - 0.25),
        ) > 1e-4
      ) {
        throw new Error(
          `multiaxis structured steering or SAE readout is incorrect: baseline=${structuredBaseline[0]}, linear=${multiaxisMeasurements?.[0]}, sae=${multiaxisMeasurements?.[1]}`,
        );
      }
      const inactiveAffine = new Uint32Array(layerCount * groups);
      console.info("drowse feasibility: affine gate off");
      await engine.updateDrowseStructuredControls(
        inactiveAffine,
        new Uint32Array(layerCount * curves),
        "drowse-tiny",
      );
      await runOne();
      const gatedOffMeasurements =
        await engine.readDrowseMeasurements("drowse-tiny");
      if (
        gatedOffMeasurements === undefined ||
        Math.abs(gatedOffMeasurements[0] - structuredBaseline[0]) > 1e-3
      ) {
        throw new Error(
          `dynamic structured controls did not disable the affine program: baseline=${structuredBaseline[0]}, actual=${gatedOffMeasurements?.[0]}`,
        );
      }

      const gateRequest = {
        model: "drowse-tiny",
        prompt: "hello",
        max_tokens: 4,
        seed: 1,
        stream: true,
        logprobs: true,
        logit_bias: { 2: -100, 15: 100 },
        temperature: 0,
      };
      const gateBaselineStream = await engine.completions.create(gateRequest);
      const gateBaselineMeasurements = [];
      for await (const chunk of gateBaselineStream) {
        if (!chunk.choices?.[0] || chunk.choices[0].finish_reason !== null)
          continue;
        const values = await engine.readDrowseMeasurements("drowse-tiny");
        if (values === undefined)
          throw new Error("gate baseline measurement is unavailable");
        gateBaselineMeasurements.push(values[0]);
      }
      const gateStream = await engine.completions.create(gateRequest);
      const gateMeasurements = [];
      console.info("drowse feasibility: streamed gate transition");
      for await (const chunk of gateStream) {
        if (!chunk.choices?.[0] || chunk.choices[0].finish_reason !== null)
          continue;
        console.info(
          `drowse feasibility: gate token ${gateMeasurements.length + 1}`,
        );
        const values = await engine.readDrowseMeasurements("drowse-tiny");
        console.info(
          `drowse feasibility: gate measurement ${gateMeasurements.length + 1}`,
        );
        if (values === undefined)
          throw new Error("streamed gate measurement is unavailable");
        gateMeasurements.push(values[0]);
        if (gateMeasurements.length === 1) {
          await engine.updateDrowseStructuredControls(
            multiaxisProgram.affineActive,
            new Uint32Array(layerCount * curves),
            "drowse-tiny",
          );
          console.info("drowse feasibility: gate enabled");
        }
      }
      if (
        gateMeasurements.length !== gateBaselineMeasurements.length ||
        gateMeasurements.length < 2 ||
        Math.abs(gateMeasurements[0] - gateBaselineMeasurements[0]) > 1e-3 ||
        gateMeasurements
          .slice(1)
          .some(
            (value, index) =>
              Math.abs(
                value - gateBaselineMeasurements[index + 1] - structuredDelta,
              ) > 2e-3,
          )
      ) {
        throw new Error(
          `streamed gate controls were not applied between decode steps: baseline=${gateBaselineMeasurements}, gated=${gateMeasurements}`,
        );
      }

      const curvedProgram = structuredProgram();
      curvedProgram.curveActive[0] = 1;
      curvedProgram.curveDomainKind[0] = 1;
      curvedProgram.curveRank[0] = 2;
      curvedProgram.curveIntrinsicDim[0] = 1;
      curvedProgram.curveEmbedDim[0] = 2;
      curvedProgram.curveNodeCount[0] = 2;
      curvedProgram.curveBasis[0] = 1;
      curvedProgram.curveBasis[hiddenSize + 1] = 1;
      curvedProgram.curvePolynomial[rank] = 1;
      curvedProgram.curveTarget[0] = 0.3;
      curvedProgram.curveAlong[0] = 1;
      console.info("drowse feasibility: curved manifold");
      await engine.setDrowseStructuredProgram(curvedProgram, "drowse-tiny");
      await engine.updateDrowseStructuredControls(
        curvedProgram.affineActive,
        curvedProgram.curveActive,
        "drowse-tiny",
      );
      await runOne();
      const curvedMeasurements =
        await engine.readDrowseMeasurements("drowse-tiny");
      if (
        curvedMeasurements === undefined ||
        curvedMeasurements.some((value) => !Number.isFinite(value)) ||
        Math.abs(curvedMeasurements[0] - structuredBaseline[0] - 0.3) > 2e-3
      ) {
        throw new Error(
          `curved manifold program returned unexpected measurements: ${curvedMeasurements?.[0]}`,
        );
      }

      const multiCurveProgram = structuredProgram();
      for (let curve = 0; curve < 2; curve += 1) {
        const slot = curve;
        multiCurveProgram.curveActive[slot] = 1;
        multiCurveProgram.curveDomainKind[slot] = 1;
        multiCurveProgram.curveRank[slot] = 1;
        multiCurveProgram.curveIntrinsicDim[slot] = 1;
        multiCurveProgram.curveEmbedDim[slot] = 2;
        multiCurveProgram.curveNodeCount[slot] = 2;
        multiCurveProgram.curveBasis[slot * rank * hiddenSize + curve] = 1;
        multiCurveProgram.curvePolynomial[slot * (embedded + 1) * rank + rank] =
          1;
        multiCurveProgram.curveTarget[slot * intrinsic] =
          curve === 0 ? 0.1 : -0.2;
        multiCurveProgram.curveAlong[slot] = 1;
      }
      console.info("drowse feasibility: simultaneous curved manifolds");
      await engine.setDrowseStructuredProgram(multiCurveProgram, "drowse-tiny");
      await runOne();
      const multiCurveMeasurements =
        await engine.readDrowseMeasurements("drowse-tiny");
      if (
        multiCurveMeasurements === undefined ||
        Math.abs(multiCurveMeasurements[0] - structuredBaseline[0] - 0.1) >
          2e-3 ||
        Math.abs(multiCurveMeasurements[2] - structuredBaseline[2] + 0.2) > 2e-3
      ) {
        throw new Error(
          `simultaneous curved programs did not compose: ${multiCurveMeasurements}`,
        );
      }

      const multidimensionalCurveProgram = structuredProgram();
      multidimensionalCurveProgram.curveActive[0] = 1;
      multidimensionalCurveProgram.curveDomainKind[0] = 1;
      multidimensionalCurveProgram.curveRank[0] = 2;
      multidimensionalCurveProgram.curveIntrinsicDim[0] = 2;
      multidimensionalCurveProgram.curveEmbedDim[0] = 4;
      multidimensionalCurveProgram.curveNodeCount[0] = 2;
      multidimensionalCurveProgram.curveBasis[0] = 1;
      multidimensionalCurveProgram.curveBasis[hiddenSize + 1] = 1;
      multidimensionalCurveProgram.curvePolynomial[rank] = 1;
      multidimensionalCurveProgram.curvePolynomial[3 * rank + 1] = 1;
      multidimensionalCurveProgram.curveTarget[0] = 0.1;
      multidimensionalCurveProgram.curveTarget[1] = -0.1;
      multidimensionalCurveProgram.curveAlong[0] = 1;
      console.info("drowse feasibility: multidimensional curved manifold");
      await engine.setDrowseStructuredProgram(
        multidimensionalCurveProgram,
        "drowse-tiny",
      );
      await runOne();
      const multidimensionalCurveMeasurements =
        await engine.readDrowseMeasurements("drowse-tiny");
      if (
        multidimensionalCurveMeasurements === undefined ||
        Math.abs(
          multidimensionalCurveMeasurements[0] - structuredBaseline[0] - 0.1,
        ) > 2e-3 ||
        Math.abs(
          multidimensionalCurveMeasurements[2] - structuredBaseline[2] + 0.1,
        ) > 2e-3
      ) {
        throw new Error(
          `multidimensional curved program returned unexpected measurements: ${multidimensionalCurveMeasurements}`,
        );
      }

      const periodicCurveProgram = structuredProgram();
      periodicCurveProgram.curveActive[0] = 1;
      periodicCurveProgram.curveDomainKind[0] = 1;
      periodicCurveProgram.curveRank[0] = 2;
      periodicCurveProgram.curveIntrinsicDim[0] = 1;
      periodicCurveProgram.curveEmbedDim[0] = 2;
      periodicCurveProgram.curveNodeCount[0] = 2;
      periodicCurveProgram.curveBasis[0] = 1;
      periodicCurveProgram.curveBasis[hiddenSize + 1] = 1;
      periodicCurveProgram.curvePolynomial[rank] = 1;
      periodicCurveProgram.curvePolynomial[2 * rank + 1] = 1;
      periodicCurveProgram.curveTarget[0] = 0.25;
      periodicCurveProgram.curveAlong[0] = 1;
      periodicCurveProgram.curveOnto[0] = 1;
      periodicCurveProgram.curveBounds[0] = 0;
      periodicCurveProgram.curveBounds[1] = 1;
      periodicCurveProgram.curveAxisPeriodic[0] = 1;
      periodicCurveProgram.curveAxisPeriod[0] = 1;
      console.info("drowse feasibility: periodic curved manifold");
      await engine.setDrowseStructuredProgram(
        periodicCurveProgram,
        "drowse-tiny",
      );
      await runOne();
      const periodicCurveMeasurements =
        await engine.readDrowseMeasurements("drowse-tiny");
      let expectedPeriodicFoot = 0;
      const periodicQ = [structuredBaseline[0], structuredBaseline[2]];
      for (let iteration = 0; iteration < 1; iteration += 1) {
        const angle = 2 * Math.PI * expectedPeriodicFoot;
        const surface = [Math.cos(angle), Math.sin(angle)];
        const jacobian = [
          -2 * Math.PI * Math.sin(angle),
          2 * Math.PI * Math.cos(angle),
        ];
        const rhs =
          jacobian[0] * (periodicQ[0] - surface[0]) +
          jacobian[1] * (periodicQ[1] - surface[1]);
        const diagonal = jacobian[0] ** 2 + jacobian[1] ** 2;
        expectedPeriodicFoot += rhs / (diagonal * (1 + 1e-3) + 1e-9);
        expectedPeriodicFoot -= Math.floor(expectedPeriodicFoot);
      }
      expectedPeriodicFoot += 0.25;
      expectedPeriodicFoot -= Math.floor(expectedPeriodicFoot);
      const expectedPeriodicCoordinates = [
        Math.cos(2 * Math.PI * expectedPeriodicFoot),
        Math.sin(2 * Math.PI * expectedPeriodicFoot),
      ];
      if (
        periodicCurveMeasurements === undefined ||
        Math.abs(
          periodicCurveMeasurements[0] - expectedPeriodicCoordinates[0],
        ) > 2e-3 ||
        Math.abs(
          periodicCurveMeasurements[2] - expectedPeriodicCoordinates[1],
        ) > 2e-3
      ) {
        throw new Error(
          `periodic curved program returned unexpected measurements: actual=${periodicCurveMeasurements}, expected=${expectedPeriodicCoordinates}`,
        );
      }

      const jLensProgram = structuredProgram();
      jLensProgram.probeKind.fill(0);
      for (let layer = 0; layer < jLensLayerCount; layer += 1) {
        for (let probe = 0; probe < probes; probe += 1) {
          jLensProgram.probeKind[layer * probes + probe] = 3;
        }
      }
      const jLensBindingId = "c".repeat(64);
      const jLensLayerIndices = Int32Array.from(
        { length: jLensLayerCount },
        (_, index) => index,
      );
      const jLensMatrices = Array.from(
        { length: jLensLayerCount },
        () => new Float32Array(hiddenSize * hiddenSize),
      );
      for (let layer = 0; layer < jLensLayerCount; layer += 1) {
        for (let coordinate = 0; coordinate < hiddenSize; coordinate += 1) {
          jLensMatrices[layer][coordinate * hiddenSize + coordinate] = 1;
        }
      }
      await engine.setDrowseJlensDictionary(
        {
          hookAbi: module.DROWSE_HOOK_ABI,
          bindingId: jLensBindingId,
          hiddenSize,
          layerIndices: jLensLayerIndices,
          matrices: jLensMatrices,
        },
        "drowse-tiny",
      );
      jLensProgram.jLensBindingId = jLensBindingId;
      jLensProgram.jLensTokenIds = Int32Array.from(
        { length: probes },
        (_, index) => index,
      );
      jLensProgram.jLensLayerIndices = jLensLayerIndices;
      console.info(
        `drowse feasibility: exact full-vocabulary J-lens (${jLensLayerCount} fitted layers)`,
      );
      const jLensInstallStarted = performance.now();
      let jLensInstallMs;
      let jLensGenerateMs;
      try {
        await engine.setDrowseStructuredProgram(jLensProgram, "drowse-tiny");
        jLensInstallMs = performance.now() - jLensInstallStarted;
        console.info("drowse feasibility: exact J-lens installed");
        const jLensGenerateStarted = performance.now();
        await runOne();
        jLensGenerateMs = performance.now() - jLensGenerateStarted;
        console.info("drowse feasibility: exact J-lens generated");
      } catch (error) {
        const value = error ?? {};
        throw new Error(
          `exact J-lens stage failed: name=${value.name ?? "unknown"}, ` +
            `message=${value.message ?? String(value)}, status=${value.status ?? "unknown"}, ` +
            `stack=${value.stack ?? "unavailable"}`,
        );
      }
      const jLensTopTokens =
        await engine.readDrowseJlensTopTokens("drowse-tiny");
      if (
        jLensTopTokens === undefined ||
        jLensTopTokens.fittedLayerCount !== jLensLayerCount ||
        jLensTopTokens.tokenIds.length !== 8 ||
        jLensTopTokens.strength.length !== 8 ||
        jLensTopTokens.centerOfMass.length !== 8 ||
        jLensTopTokens.spread.length !== 8 ||
        jLensTopTokens.layerIndices.length !== jLensLayerCount ||
        jLensTopTokens.layerTokenIds.length !== jLensLayerCount * 8 ||
        jLensTopTokens.layerProbabilities.length !== jLensLayerCount * 8
      ) {
        throw new Error(
          "exact J-lens returned an incomplete top-token payload",
        );
      }
      for (let index = 0; index < 8; index += 1) {
        if (
          jLensTopTokens.tokenIds[index] < 0 ||
          jLensTopTokens.tokenIds[index] >= vocabSize ||
          !Number.isFinite(jLensTopTokens.strength[index]) ||
          jLensTopTokens.strength[index] < 0 ||
          jLensTopTokens.strength[index] > 1 ||
          !Number.isFinite(jLensTopTokens.centerOfMass[index]) ||
          jLensTopTokens.centerOfMass[index] < 0 ||
          jLensTopTokens.centerOfMass[index] > 1 ||
          !Number.isFinite(jLensTopTokens.spread[index]) ||
          jLensTopTokens.spread[index] < 0 ||
          jLensTopTokens.spread[index] > 1 ||
          (index > 0 &&
            jLensTopTokens.strength[index] > jLensTopTokens.strength[index - 1])
        ) {
          throw new Error(
            "exact J-lens returned invalid aggregate top-token data",
          );
        }
      }
      if (new Set(jLensTopTokens.tokenIds).size !== 8) {
        throw new Error("exact J-lens aggregate top-token IDs are not unique");
      }
      for (let layer = 0; layer < jLensLayerCount; layer += 1) {
        if (jLensTopTokens.layerIndices[layer] !== layer) {
          throw new Error("exact J-lens layer indices changed during readout");
        }
        const rowStart = layer * 8;
        const rowIds = jLensTopTokens.layerTokenIds.subarray(
          rowStart,
          rowStart + 8,
        );
        if (new Set(rowIds).size !== 8) {
          throw new Error(
            "exact J-lens per-layer top-token IDs are not unique",
          );
        }
        for (let index = 0; index < 8; index += 1) {
          const tokenId = rowIds[index];
          const probability =
            jLensTopTokens.layerProbabilities[rowStart + index];
          if (
            tokenId < 0 ||
            tokenId >= vocabSize ||
            !Number.isFinite(probability) ||
            probability < 0 ||
            probability > 1 ||
            (index > 0 &&
              probability >
                jLensTopTokens.layerProbabilities[rowStart + index - 1])
          ) {
            throw new Error(
              "exact J-lens returned invalid per-layer top-token data",
            );
          }
        }
      }
      const jLensMeasurements =
        await engine.readDrowseMeasurements("drowse-tiny");
      if (
        jLensMeasurements === undefined ||
        jLensMeasurements.some(
          (value) => !Number.isFinite(value) || value < 0 || value > 1,
        )
      ) {
        throw new Error(
          `exact J-lens returned invalid probabilities: ${jLensMeasurements}`,
        );
      }
      const jLensRms = Math.sqrt(
        (structuredBaseline[0] ** 2 + structuredBaseline[2] ** 2) / hiddenSize +
          1e-6,
      );
      const jLensLogits = Array.from({ length: 16 }, (_, tokenId) => {
        if (quantization === "q4f16_1") {
          return ((tokenId - 7) * 0.125 * structuredBaseline[0]) / jLensRms;
        }
        const fraction = tokenId / 15;
        return (
          (structuredBaseline[0] * (-0.5 + fraction)) / jLensRms +
          (structuredBaseline[2] * (0.25 - 0.5 * fraction)) / jLensRms
        );
      });
      const maxJLensLogit = Math.max(...jLensLogits);
      const jLensDenominator = jLensLogits.reduce(
        (total, value) => total + Math.exp(value - maxJLensLogit),
        0,
      );
      const expectedJLensProbabilities = jLensLogits.map(
        (value) => Math.exp(value - maxJLensLogit) / jLensDenominator,
      );
      for (let layer = 0; layer < jLensLayerCount; layer += 1) {
        const selectedProbability = jLensMeasurements
          .subarray(layer * probes, (layer + 1) * probes)
          .reduce((total, value) => total + value, 0);
        if (selectedProbability <= 0 || selectedProbability >= 1) {
          throw new Error(
            `exact J-lens probabilities did not come from a full-vocabulary softmax: ${jLensMeasurements}`,
          );
        }
        if (
          layerCount === 2 &&
          hiddenSize === 64 &&
          expectedJLensProbabilities
            .slice(0, probes)
            .some(
              (expected, tokenId) =>
                Math.abs(
                  jLensMeasurements[layer * probes + tokenId] - expected,
                ) > Math.max(2e-6, expected * 1e-2),
            )
        ) {
          throw new Error(
            `exact J-lens probabilities disagree with the fp32 CPU golden: actual=${jLensMeasurements}, expected=${expectedJLensProbabilities.slice(0, probes)}`,
          );
        }
      }
      await engine.clearDrowseRankOneProgram("drowse-tiny");
      const cleared = await engine.completions.create({
        model: "drowse-tiny",
        prompt,
        max_tokens: 1,
        seed: 1,
        temperature: 0,
      });
      if (baseline.choices[0]?.text !== cleared.choices[0]?.text) {
        throw new Error(
          "clearing Drowse hooks did not restore deterministic ordinary generation",
        );
      }
      const stream = await engine.completions.create({
        model: "drowse-tiny",
        prompt: "hello",
        max_tokens: 16,
        seed: 1,
        stream: true,
        stream_options: { include_usage: true },
        logprobs: true,
        temperature: 0,
      });
      let stopChunks = 0;
      for await (const _chunk of stream) {
        stopChunks += 1;
        if (stopChunks === 1) await engine.interruptGenerate();
      }
      if (stopChunks < 1 || stopChunks >= 16) {
        throw new Error(
          `interrupt did not stop streamed generation: ${stopChunks} chunks`,
        );
      }
      await engine.unload();
      for (let index = 1; index < reloads; index += 1) {
        await engine.reload("drowse-tiny");
        await engine.completions.create({
          model: "drowse-tiny",
          prompt: "hello",
          max_tokens: 1,
          seed: 1,
          temperature: 0,
        });
        await engine.unload();
      }
      runtimeWorker?.terminate();
      const info = adapter.info ?? {};
      return {
        adapter: {
          architecture: info.architecture ?? "",
          description: info.description ?? "",
          device: info.device ?? "",
          isFallbackAdapter: info.isFallbackAdapter ?? null,
          vendor: info.vendor ?? "",
        },
        architecture,
        baseline: baseline.choices[0]?.text ?? "",
        cleared: cleared.choices[0]?.text ?? "",
        steered: steered.choices[0]?.text ?? "",
        measurements: Array.from(measurements),
        uncappedMeasurement: uncappedMeasurements[0],
        promptTokens,
        quantization,
        reloads,
        runtime: useWorker ? "worker" : "main-thread",
        artifactSource: verifiedArtifacts ? "caller-owned" : "webllm-cache",
        selectedAdapter,
        stopChunks,
        supported,
        captureSupported,
        structuredSupported,
        curvedSupported,
        structuredMeasurement: multiaxisMeasurements[0],
        saeMeasurement: multiaxisMeasurements[1],
        gateMeasurements,
        gateBaselineMeasurements,
        curvedMeasurement: curvedMeasurements[0],
        multiCurveMeasurements: [
          multiCurveMeasurements[0],
          multiCurveMeasurements[2],
        ],
        multidimensionalCurveMeasurements: [
          multidimensionalCurveMeasurements[0],
          multidimensionalCurveMeasurements[2],
        ],
        periodicCurveMeasurements: [
          periodicCurveMeasurements[0],
          periodicCurveMeasurements[2],
        ],
        jLensLayerCount,
        jLensInstallMs,
        jLensGenerateMs,
        jLensMeasurements: Array.from(jLensMeasurements),
        jLensTopTokenIds: Array.from(jLensTopTokens.tokenIds),
        jLensTopStrength: Array.from(jLensTopTokens.strength),
        capturedPositions: captureBeforeSteering.positions,
      };
    },
    {
      architecture: options.architecture,
      promptTokens: options.promptTokens,
      quantization: options.quantization,
      reloads: options.reloads,
      useWorker: options.useWorker,
      verifiedArtifacts: options.verifiedArtifacts,
      selectedAdapter: options.selectedAdapter,
      hiddenSize: modelShape.hidden_size,
      layerCount: modelShape.num_hidden_layers,
      jLensLayerCount:
        options.jLensLayers ?? modelShape.num_hidden_layers,
      vocabSize: modelConfig.vocab_size,
      captureSpecialTokenIds: modelConfig.drowse_capture_special_token_ids ?? [
        0, 1, 2, 3,
      ],
    },
  );
  if (webGpuErrors.length !== 0) {
    throw new Error(
      `browser reported ${webGpuErrors.length} uncaptured WebGPU error(s)`,
    );
  }
  const artifactRequests = Object.fromEntries(
    [
      ...modelArtifactNames.map((name) => `/model/${name}`),
      "/model-lib.wasm",
    ].map((path) => [path, requestCounts.get(path) ?? 0]),
  );
  if (
    options.verifiedArtifacts &&
    Object.values(artifactRequests).some((count) => count !== 1)
  ) {
    throw new Error(
      `caller-owned artifacts were fetched an unexpected number of times: ${JSON.stringify(artifactRequests)}`,
    );
  }
  console.log(
    JSON.stringify(
      {
        ...result,
        libraryBytes: modelLibrary.byteLength,
        librarySha256: createHash("sha256").update(modelLibrary).digest("hex"),
        artifactRequests,
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
  await new Promise((resolvePromise) => server.close(resolvePromise));
}

function parseArguments(args) {
  const result = {
    architecture: null,
    browserChannel: null,
    headed: false,
    library: null,
    jLensLayers: null,
    model: null,
    promptTokens: 1,
    quantization: "q0f32",
    reloads: 1,
    useWorker: true,
    verifiedArtifacts: false,
    selectedAdapter: false,
    webllm: null,
  };
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--headed") result.headed = true;
    else if (argument === "--main-thread") result.useWorker = false;
    else if (argument === "--verified-artifacts")
      result.verifiedArtifacts = true;
    else if (argument === "--selected-adapter") result.selectedAdapter = true;
    else if (
      [
        "--architecture",
        "--browser-channel",
        "--library",
        "--jlens-layers",
        "--model",
        "--prompt-tokens",
        "--quantization",
        "--reloads",
        "--webllm",
      ].includes(argument)
    ) {
      const value = args[index + 1];
      if (!value || value.startsWith("--"))
        throw new Error(`missing value for ${argument}`);
      const key =
        argument === "--jlens-layers"
          ? "jLensLayers"
          : argument
              .slice(2)
              .replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
      result[key] = ["jLensLayers", "promptTokens", "reloads"].includes(key)
        ? Number(value)
        : value;
      index += 1;
    } else throw new Error(`unexpected argument: ${argument}`);
  }
  if (!["qwen3", "llama", "gemma3_text"].includes(result.architecture)) {
    throw new Error("--architecture must be qwen3, llama, or gemma3_text");
  }
  if (
    result.browserChannel !== null &&
    ![
      "chrome",
      "chrome-beta",
      "chrome-canary",
      "msedge",
      "msedge-beta",
      "msedge-dev",
      "msedge-canary",
    ].includes(result.browserChannel)
  ) {
    throw new Error(
      "--browser-channel must name a Playwright Chrome or Edge channel",
    );
  }
  if (!["q0f32", "q4f32_1", "q4f16_1"].includes(result.quantization)) {
    throw new Error("--quantization must be q0f32, q4f32_1, or q4f16_1");
  }
  if (!Number.isSafeInteger(result.promptTokens) || result.promptTokens < 1) {
    throw new Error("--prompt-tokens must be a positive integer");
  }
  if (
    result.jLensLayers !== null &&
    (!Number.isSafeInteger(result.jLensLayers) || result.jLensLayers < 1)
  ) {
    throw new Error("--jlens-layers must be a positive integer");
  }
  if (result.verifiedArtifacts && !result.useWorker) {
    throw new Error("--verified-artifacts requires the worker runtime");
  }
  if (
    result.selectedAdapter &&
    (!result.verifiedArtifacts || result.reloads !== 1)
  ) {
    throw new Error(
      "--selected-adapter requires --verified-artifacts and --reloads 1",
    );
  }
  if (
    !Number.isSafeInteger(result.reloads) ||
    result.reloads < 1 ||
    result.reloads > 10
  ) {
    throw new Error("--reloads must be an integer from 1 through 10");
  }
  for (const key of ["library", "model", "webllm"]) {
    if (result[key] === null) throw new Error(`--${key} is required`);
  }
  return result;
}

async function requireFile(path, label) {
  const info = await stat(path);
  if (!info.isFile())
    throw new Error(`${label} is not a regular file: ${path}`);
}

function pageHtml() {
  return "<!doctype html><meta charset=utf-8><link rel=icon href=data:,><title>Drowse browser feasibility</title>";
}

function workerSource(verifiedArtifacts, selectedAdapter, artifactNames) {
  if (!verifiedArtifacts) {
    return `import { WebWorkerMLCEngineHandler } from "/webllm.js";
const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (message) => handler.onmessage(message);`;
  }
  return `import { MLCEngine, WebWorkerMLCEngineHandler } from "/webllm.js";

const originalConsoleError = console.error.bind(console);
console.error = (...values) => originalConsoleError(...values.map((value) =>
  value?.error?.message ?? value?.message ?? value
));

const pending = [];
self.onmessage = (message) => pending.push(message);

const modelBase = new URL("/model/resolve/main/", self.location.origin);
const sources = ${JSON.stringify(artifactNames)}.map((name) => [
  new URL(name, modelBase).href,
  \`/model/\${name}\`,
]);
sources.push([new URL("/model-lib.wasm", self.location.origin).href, "/model-lib.wasm"]);
const entries = await Promise.all(
  sources.map(async ([artifactUrl, sourcePath]) => [
    artifactUrl,
    await (await fetch(sourcePath)).arrayBuffer(),
  ]),
);
const artifacts = new Map(entries);
let gpuAdapter;
if (${selectedAdapter}) {
  const calibrationAdapter = await navigator.gpu.requestAdapter();
  if (calibrationAdapter === null) throw new Error("no calibration adapter was returned");
  const calibrationDevice = await calibrationAdapter.requestDevice();
  calibrationDevice.destroy();
  gpuAdapter = await navigator.gpu.requestAdapter();
  if (gpuAdapter === null) throw new Error("no load-time WebGPU adapter was returned");
  const checkedInfo = calibrationAdapter.info ?? {};
  const loadInfo = gpuAdapter.info ?? {};
  for (const field of ["vendor", "architecture", "device", "isFallbackAdapter"]) {
    if ((checkedInfo[field] ?? null) !== (loadInfo[field] ?? null)) {
      throw new Error(\`load-time WebGPU adapter changed \${field}\`);
    }
  }
  const checkedFeatures = [...calibrationAdapter.features].sort().join(",");
  const loadFeatures = [...gpuAdapter.features].sort().join(",");
  if (checkedFeatures !== loadFeatures) throw new Error("load-time WebGPU features changed");
}
const artifactCache = {
  async fetchWithCache(url, storeType) {
    const data = artifacts.get(url);
    if (data === undefined) throw new Error(\`unverified artifact requested: \${url}\`);
    if (storeType === "json") {
      return JSON.parse(new TextDecoder().decode(data));
    }
    return data.slice(0);
  },
  async addToCache(url) {
    if (!artifacts.has(url)) throw new Error(\`unverified artifact requested: \${url}\`);
  },
  async hasAllKeys(keys) {
    return keys.every((key) => artifacts.has(key));
  },
  async deleteInCache() {
    throw new Error("caller-owned artifacts are read-only");
  },
};
const engine = new MLCEngine({
  appConfig: {
    artifactCache,
    ...(gpuAdapter ? { gpuAdapter } : {}),
    model_list: [
      {
        model: modelBase.href,
        model_id: "drowse-tiny",
        model_lib: new URL("/model-lib.wasm", self.location.origin).href,
      },
    ],
  },
  logLevel: "WARN",
});
const handler = new WebWorkerMLCEngineHandler(engine);
self.onmessage = (message) => handler.onmessage(message);
for (const message of pending) handler.onmessage(message);`;
}

function contentType(path) {
  if (extname(path) === ".json") return "application/json";
  if (extname(path) === ".bin") return "application/octet-stream";
  return "application/octet-stream";
}

function send(response, body, type, status = 200) {
  response.writeHead(status, {
    "Cache-Control": "no-store",
    "Content-Type": type,
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Opener-Policy": "same-origin",
  });
  response.end(body);
}
