import { compileRankOneHookProgram } from "../src/hosted/runtime/rankOneHookProgram";
import { compileStructuredHookProgram } from "../src/hosted/runtime/structuredHookProgram";
import { resolveStructuredHookProfile } from "../src/hosted/runtime/structuredHookProfile";

export async function verifySteeringPositions(runtime: any, manifest: any, options: { arithmeticOnly?: boolean } = {}) {
  const engine = runtime.requireEngine();
  const pipeline: any = [...engine.loadedModelIdToPipeline.values()][0];
  const originalForward = pipeline.embedAndForward.bind(pipeline);
  const hidden = manifest.hiddenSize;
  const layers = manifest.layerMap.length;
  const direction = Array.from({ length: hidden }, (_, i) => Math.sin(i + 1) / Math.sqrt(hidden / 2));
  const norm = Math.hypot(...direction);
  for (let i = 0; i < direction.length; i++) direction[i] /= norm;
  const zeros = new Array(hidden).fill(0);
  const layer = Math.floor(layers / 3);
  const structured = compileStructuredHookProgram(hidden,
    Array.from({ length: layers }, (_, i) => i === layer ? {
      affineGroups: [{ active: true, basis: [direction], neutral: zeros, target: [2], along: 1, kappa: [0] }],
    } : {}), resolveStructuredHookProfile(manifest.structuredHookProfile));
  const rankOne = compileRankOneHookProgram(hidden,
    Array.from({ length: layers }, (_, i) => ({
      enabled: i === layer, basis: i === layer ? direction : zeros,
      neutral: zeros, target: 2, along: 1, collapse: 0,
      probeBasis: i === layer ? direction : zeros, probeNeutral: zeros,
    })));
  const forced = (await runtime.tokenizeText(" clear skies and calm water beside the shore.")).slice(0, 7);
  const prompt = "The weather near the quiet lake was pleasant and the sky was";
  const run = async (program: any, positions?: any, reference?: Set<number>, referenceChunkSize = 1) => {
    const trace: { position: number; length: number; active: boolean }[] = [];
    if (reference !== undefined) {
      pipeline.embedAndForward = async (parts: number[][], length: number, useProgram = true) => {
        if (!useProgram) return originalForward(parts, length, useProgram);
        const tokens = parts.flat();
        let logits: any;
        for (let offset = 0; offset < tokens.length;) {
          const position = pipeline.filledKVCacheLength;
          const active = reference.has(position);
          let count = 1;
          while (count < referenceChunkSize && offset + count < tokens.length &&
            reference.has(position + count) === active) count++;
          if (program.format) {
            await pipeline.updateDrowseStructuredControls(
              active ? program.affineActive : new Uint32Array(program.affineActive.length),
              active ? program.curveActive : new Uint32Array(program.curveActive.length),
            );
          } else {
            pipeline.drowseProgram.enabled.copyFrom(active
              ? new Int32Array(program.enabled) : new Int32Array(program.enabled.length));
          }
          logits?.dispose();
          logits = await originalForward([tokens.slice(offset, offset + count)], count, useProgram);
          trace.push({ position, length: count, active });
          offset += count;
        }
        return logits;
      };
    }
    const rows: { tokenId: number; logprob: number }[] = [];
    try {
      const result = await runtime.streamGeneration({
        input: { kind: "raw", prompt },
        sampling: { max_tokens: forced.length, temperature: 1, top_p: 1, top_k: 2147483647, seed: 1, return_top_k: 0 },
        replay: { forcedPrefixTokenIds: forced, scoreTokenIds: forced },
        hookProgram: program,
        ...positions,
        onRawToken(token: any) {
          if (program.controls?.affine.some((control: any) => control?.gate != null) &&
            token.hookMeasurements?.[layer * program.profile.maxProbes] !== 1) {
            throw new Error("The gate oracle requires its declared constant probe to read exactly 1");
          }
          const score = token.replayScore?.requestedLogprobs.find((row: any) => row.tokenId === token.tokenId);
          if (!score) throw new Error("Replay omitted the requested raw-model score");
          rows.push({ tokenId: token.tokenId, logprob: score.logprob });
        },
      }, () => {});
      if (rows.length !== forced.length) throw new Error(`Expected ${forced.length} forced tokens, got ${rows.length}`);
      return { rows, promptTokens: result.usage.promptTokens, trace };
    } finally {
      pipeline.embedAndForward = originalForward;
    }
  };
  const compare = (label: string, actual: any, expected: any, tolerance: number) => {
    let maxLogprobError = 0;
    for (let i = 0; i < expected.rows.length; i++) {
      if (actual.rows[i].tokenId !== expected.rows[i].tokenId) throw new Error(`${label}: token mismatch at ${i}`);
      const error = Math.abs(actual.rows[i].logprob - expected.rows[i].logprob);
      if (!Number.isFinite(error)) throw new Error(`${label}: non-finite log probabilities at ${i}: ${actual.rows[i].logprob}, ${expected.rows[i].logprob}`);
      maxLogprobError = Math.max(maxLogprobError, error);
    }
    if (maxLogprobError > tolerance) throw new Error(`${label}: max logprob error ${maxLogprobError} > ${tolerance}`);
    return { label, maxLogprobError, tolerance, tokens: actual.rows.length };
  };
  const checks = [];
  const batchingDiagnostics = [];
  const originalChunkSize = pipeline.prefillChunkSize;
  // Exercise cross-chunk boundaries with the real compiled prefill/decode kernels.
  pipeline.prefillChunkSize = Math.min(4, originalChunkSize);
  try {
    if (!options.arithmeticOnly) {
    for (const [name, program] of [["rank-one", rankOne], ["structured", structured]] as [string, any][]) {
      const control = program.format ? { ...program, affineActive: new Uint32Array(program.affineActive.length), curveActive: new Uint32Array(program.curveActive.length) }
        : { ...program, enabled: new Uint32Array(program.enabled.length) };
      const baseline = await run(control);
      const all = await run(program);
      const p = baseline.promptTokens;
      const spans = [[1, 3], [5, 8], [p, p + 2]];
      const indices = spans.flatMap(([start, end]) => Array.from({ length: end - start }, (_, i) => start + i));
      const expected = await run(program, undefined, new Set(indices));
      const matched = await run(program, undefined, new Set(indices), pipeline.prefillChunkSize);
      const matchedControl = await run(control, undefined, new Set(indices), pipeline.prefillChunkSize);
      const actual = await run(program, { position_spans: spans });
      checks.push(compare(`${name}: spans vs independent matched-batch reference`, actual, matched, 0));
      pipeline.prefillChunkSize = 1;
      try {
        checks.push(compare(`${name}: single-token spans vs independent per-token reference`,
          await run(program, { position_spans: spans }), expected, 0));
      } finally {
        pipeline.prefillChunkSize = Math.min(4, originalChunkSize);
      }
      // Different prefill/decode batch shapes can round differently, especially
      // in FP16. Record that separately; position parity uses matched shapes.
      const unsteeredPerToken = await run(control, undefined, new Set());
      const steeredBatching = compare(`${name}: batched vs per-token steering`, actual, expected, Infinity);
      const unsteeredBatching = compare(`${name}: batched vs per-token no injection`, baseline, unsteeredPerToken, Infinity);
      batchingDiagnostics.push({ name, steeredMaxLogprobError: steeredBatching.maxLogprobError,
        unsteeredMaxLogprobError: unsteeredBatching.maxLogprobError,
        meetsFp32ReferenceTolerance: steeredBatching.maxLogprobError <= 5e-4,
        matchedReferenceTrace: matched.trace });
      checks.push(compare(`${name}: indices vs spans`, await run(program, { position_indices: indices }), actual, 0));
      checks.push(compare(`${name}: all positions vs existing steering`, await run(program, { position_spans: [[0, 100000]] }), all, 0));
      checks.push(compare(`${name}: empty selection vs no injection`, await run(program, { position_spans: [] }), baseline, 0));
      checks.push(compare(`${name}: future selection vs no injection`, await run(program, { position_spans: [[10000, 10001]] }), baseline, 0));
      checks.push(compare(`${name}: cleared selection restores existing steering`, await run(program), all, 0));
      const effect = Math.max(...actual.rows.map((row, i) => Math.abs(row.logprob - matchedControl.rows[i].logprob)));
      if (effect < 1e-4) throw new Error(`${name}: nonzero steering had no measurable effect`);
      checks.push({ label: `${name}: measurable steering`, maxLogprobChange: effect, promptTokens: p, spans });
      console.info(`position spans: ${name} hardware checks passed`);
    }
    const chatInput = { kind: "chat", messages: [{ role: "user", content: prompt }] };
    const chat = await run(structured, { input: chatInput });
    const chatSpans = [[0, 1], [chat.promptTokens - 1, chat.promptTokens + 2]];
    const chatIndices = new Set([0, chat.promptTokens - 1, chat.promptTokens, chat.promptTokens + 1]);
    checks.push(compare("chat template/BOS positions vs independent control",
      await run(structured, { input: chatInput, position_spans: chatSpans }),
      await run(structured, { input: chatInput }, chatIndices, pipeline.prefillChunkSize), 0));
    pipeline.prefillChunkSize = 1;
    const probeProgram = compileStructuredHookProgram(hidden,
      Array.from({ length: layers }, (_, i) => i === layer ? {
        affineGroups: [{ active: true, basis: [direction], neutral: zeros, target: [2], along: 1, kappa: [0] }],
        probes: [{ kind: "linear" as const, direction: zeros, bias: 1 }],
      } : {}), resolveStructuredHookProfile(manifest.structuredHookProfile));
    probeProgram.measurementSchema = {
      layerMap: Array.from({ length: layers }, (_, i) => i), modelLayerCount: layers,
      probes: Array.from({ length: probeProgram.profile.maxProbes }, (_, i) =>
        i === 0 ? { name: "test.constant", family: "geometry" as const } : null),
      lensSource: null, saeSource: null,
    };
    const promptTokens = (await run(probeProgram)).promptTokens;
    const controlSpans = [[0, 3], [promptTokens - 1, promptTokens + 4]];
    for (const [label, phase, threshold, matches] of [
      ["prompt phase", "prompt_only", null, (p: number) => p < promptTokens],
      ["generated phase", "generated_only", null, (p: number) => p >= promptTokens],
      ["open probe gate", "both", 0.5, (p: number) => p >= promptTokens],
      ["closed probe gate", "both", 1.5, (_p: number) => false],
    ] as const) {
      const affine = Array(probeProgram.affineActive.length).fill(null);
      affine[layer * probeProgram.profile.maxAffineGroups] = { enabled: true, phase: { kind: phase },
        gate: threshold === null ? null : { slots: [{ layer, probe: 0 }], operator: ">", threshold } };
      const controlled = { ...probeProgram, controls: { affine, curve: Array(probeProgram.curveActive.length).fill(null) } };
      const expectedPositions = new Set(controlSpans.flatMap(([start, end]) =>
        Array.from({ length: end - start }, (_, i) => start + i).filter(matches)));
      checks.push(compare(`spans intersect ${label}`,
        await run(controlled, { position_spans: controlSpans }),
        await run(probeProgram, undefined, expectedPositions), 0));
    }
    }
    const captureTokens = await runtime.tokenizeText(prompt);
    const capturePositions = captureTokens.map((_: number, index: number) => index);
    const captureSpans: [number, number][] = [[1, 3], [5, 8]];
    const captureIndices = new Set([1, 2, 5, 6, 7]);
    const originalCapture = pipeline.captureDrowseRankOneTokenChunkV1.bind(pipeline);
    pipeline.prefillChunkSize = 1;
    pipeline.captureDrowseRankOneTokenChunkV1 = async (tokens: number[], positions: number[]) => {
      pipeline.drowseProgram.enabled.copyFrom(captureIndices.has(pipeline.filledKVCacheLength)
        ? new Int32Array(rankOne.enabled) : new Int32Array(rankOne.enabled.length));
      return originalCapture(tokens, positions);
    };
    let expectedCapture: any;
    try {
      expectedCapture = await engine.captureDrowseRankOneResidualsV1(captureTokens, capturePositions, rankOne);
    } finally {
      pipeline.captureDrowseRankOneTokenChunkV1 = originalCapture;
    }
    const actualCapture = await engine.captureDrowseRankOneResidualsV1(captureTokens, capturePositions,
      { ...rankOne, position_spans: captureSpans });
    if (actualCapture.values.length !== expectedCapture.values.length) throw new Error("Capture size mismatch");
    let maxAbsResidualError = 0;
    for (let i = 0; i < actualCapture.values.length; i++) {
      const error = Math.abs(actualCapture.values[i] - expectedCapture.values[i]);
      if (!Number.isFinite(error)) throw new Error(`Capture non-finite residual at ${i}`);
      maxAbsResidualError = Math.max(maxAbsResidualError, error);
    }
    if (maxAbsResidualError !== 0) throw new Error(`Position capture mismatch: ${maxAbsResidualError}`);
    checks.push({ label: "rank-one capture vs independent per-token reference", maxAbsResidualError,
      positions: capturePositions.length, layers, values: actualCapture.values.length });
    const baselineCapture = await engine.captureDrowseRankOneResidualsV1(captureTokens, capturePositions,
      { ...rankOne, position_spans: [] });
    const isFp16 = manifest.quantization.includes("f16");
    const roundResidual = isFp16 ? (Math as any).f16round : Math.fround;
    if (typeof roundResidual !== "function") throw new Error("The FP16 arithmetic oracle requires Math.f16round");
    const halfView = new DataView(new ArrayBuffer(2)) as any;
    const halfNeighbor = (value: number, up: boolean) => {
      if (value === 0) return up ? 2 ** -24 : -(2 ** -24);
      halfView.setFloat16(0, value, true);
      halfView.setUint16(0, halfView.getUint16(0, true) + ((value > 0) === up ? 1 : -1), true);
      return halfView.getFloat16(0, true) as number;
    };
    const halfBound = (value: number, upper: boolean) => {
      const nearest = roundResidual(value);
      return upper ? nearest < value ? halfNeighbor(nearest, true) : nearest
        : nearest > value ? halfNeighbor(nearest, false) : nearest;
    };
    for (const sign of [1, -1]) {
      const signedProgram = { ...rankOne, target: Float32Array.from(rankOne.target, value => sign * value),
        position_spans: captureSpans };
      const captured = sign === 1 ? actualCapture
        : await engine.captureDrowseRankOneResidualsV1(captureTokens, capturePositions, signedProgram);
      let maxArithmeticError = 0;
      let maxArithmeticLocation: unknown;
      let offTargetMismatches = 0;
      let mismatchCount = 0;
      let outsideRoundingBounds = 0;
      let firstOutsideRoundingBounds: unknown;
      let outsideQuantizationBounds = 0;
      for (let l = 0; l <= layer; l++) for (let p = 0; p < capturePositions.length; p++) {
        for (let d = 0; d < hidden; d++) {
          const index = (l * capturePositions.length + p) * hidden + d;
          const delta = l === layer && captureIndices.has(p) ? sign * 2 * rankOne.basis[layer * hidden + d] : 0;
          const expected = roundResidual(Math.fround(baselineCapture.values[index] + delta));
          if (isFp16 && (roundResidual(baselineCapture.values[index]) !== baselineCapture.values[index] ||
            roundResidual(captured.values[index]) !== captured.values[index])) {
            throw new Error("The quantized capture does not lie on the expected FP16 grid");
          }
          const error = Math.abs(captured.values[index] - expected);
          if (!Number.isFinite(error)) throw new Error(`Non-finite arithmetic check at ${l}/${p}/${d}`);
          if (error !== 0) {
            mismatchCount++;
            if (delta === 0) offTargetMismatches++;
            const exact = Math.fround(baselineCapture.values[index] + delta);
            const neighbor = isFp16 && exact !== expected ? halfNeighbor(expected, expected < exact) : expected;
            if (captured.values[index] !== neighbor) {
              outsideRoundingBounds++;
              firstOutsideRoundingBounds ??= { layer: l, position: p, dimension: d,
                baseline: baselineCapture.values[index], delta, expected, neighbor, actual: captured.values[index] };
            }
            // The compiled hook adds to an FP32 intermediate, then stores FP16.
            // Its zero-injection capture is already rounded: do not equate
            // half(h + delta) with half(half(h) + delta). Bound both conversions
            // using adjacent representable values, as required by WGSL 15.7.
            const lower = isFp16 && delta !== 0
              ? halfBound(halfNeighbor(baselineCapture.values[index], false) + delta, false) : expected;
            const upper = isFp16 && delta !== 0
              ? halfBound(halfNeighbor(baselineCapture.values[index], true) + delta, true) : expected;
            if (captured.values[index] < lower || captured.values[index] > upper) outsideQuantizationBounds++;
          }
          if (error > maxArithmeticError) maxArithmeticLocation = { layer: l, position: p, dimension: d,
            baseline: baselineCapture.values[index], delta, unrounded: baselineCapture.values[index] + delta,
            expected, actual: captured.values[index] };
          maxArithmeticError = Math.max(maxArithmeticError, error);
        }
      }
      if (offTargetMismatches !== 0 || outsideQuantizationBounds !== 0) throw new Error(`Position injection arithmetic mismatch: ${JSON.stringify({
        sign, maxArithmeticError, maxArithmeticLocation, mismatchCount, offTargetMismatches,
        outsideRoundingBounds, firstOutsideRoundingBounds, outsideQuantizationBounds,
      })}`);
      checks.push({ label: `rank-one ${sign > 0 ? "positive" : "negative"} injection ${isFp16 ? "within input/output quantization bounds" : "vs direct FP32 arithmetic"}`,
        maxRoundedCaptureDifference: maxArithmeticError, roundedCaptureMismatches: mismatchCount,
        offTargetMismatches, outsideQuantizationBounds,
        checkedLayers: layer + 1, positions: capturePositions.length,
        residualPrecision: isFp16 ? "fp32 intermediate, fp16 capture" : "fp32" });
    }
    const sparsePositions = [0, 3, 8, capturePositions.length - 1];
    const sparseCapture = await engine.captureDrowseRankOneResidualsV1(captureTokens, sparsePositions,
      { ...rankOne, position_spans: captureSpans });
    for (let l = 0; l < layers; l++) for (let p = 0; p < sparsePositions.length; p++) {
      for (let d = 0; d < hidden; d++) {
        const sparse = sparseCapture.values[(l * sparsePositions.length + p) * hidden + d];
        const full = actualCapture.values[(l * capturePositions.length + sparsePositions[p]) * hidden + d];
        if (sparse !== full) throw new Error(`Sparse capture changed residual at ${l}/${sparsePositions[p]}/${d}`);
      }
    }
    checks.push({ label: "sparse captures preserve steering on uncaptured tokens", maxAbsResidualError: 0,
      positions: sparsePositions, layers });
  } finally {
    pipeline.prefillChunkSize = originalChunkSize;
    pipeline.embedAndForward = originalForward;
    await engine.clearDrowseRankOneProgram();
  }
  return { passed: true, model: manifest.source, quantization: manifest.quantization, checks, batchingDiagnostics };
}
