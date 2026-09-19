import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { fileURLToPath } from "node:url";
import ts from "typescript";

// Runs the real consumer methods with fake device objects; this is not GPU inference.
const sourcePath = process.argv[2] ?? fileURLToPath(
  new URL("../node_modules/@drowse/web-llm/lib/index.js", import.meta.url));
const source = await readFile(sourcePath, "utf8");
const parsed = ts.createSourceFile(sourcePath, source, ts.ScriptTarget.Latest, true);
const pipeline = parsed.statements.find((node) =>
  ts.isClassDeclaration(node) && node.name?.text === "LLMChatPipeline");
assert.ok(pipeline, "Missing LLMChatPipeline in the supplied source");
const names = [
  "requireKVCache", "requireRNNState", "getSingleStateForABI", "getActiveKVStates",
  "drowseStateArguments", "drowseOutputOffset",
  "invokePrefill", "invokeDecode", "invokeDrowseCapture",
  "invokeDrowseRankOneCaptureV1", "captureDrowseTokenChunk",
  "captureDrowseRankOneTokenChunkV1", "resetDrowseCaptureState",
  "embedAndForward", "embedAndForwardChunk", "applyDrowsePositionControls",
  "updateDrowseStructuredControls",
];
const methods = names.map((name) => {
  const method = pipeline.members.find((node) => node.name?.getText(parsed) === name);
  assert.ok(method && ts.isMethodDeclaration(method), `Missing method ${name}`);
  return method.getText(parsed);
});
const asyncHelper = parsed.statements.find((node) =>
  ts.isFunctionDeclaration(node) && node.name?.text === "__awaiter");
const compiled = ts.transpileModule(`${asyncHelper?.getText(parsed) ?? ""}\nclass Consumer { ${methods.join("\n")} }`, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
}).outputText;
const Consumer = new Function(`${compiled}; return Consumer;`)();
console.log(JSON.stringify({ sourceSha256: createHash("sha256").update(source).digest("hex"),
  sourcePath, validation: "consumer-contract-with-fake-device", gpuInference: false }));

test("all 256 position masks preserve absolute boundaries and intersect dynamic controls at every batch size", async () => {
  let cases = 0;
  for (let mask = 0; mask < 256; mask++) for (const chunkSize of [1, 2, 3, 8]) {
    for (const structured of [false, true]) {
      const start = 13 + mask % 5;
      const spans = [];
      for (let position = 0; position < 8; position++) {
        if (!(mask & (1 << position))) continue;
        const left = position;
        while (position < 7 && mask & (1 << (position + 1))) position++;
        spans.push([start + left, start + position + 1]);
      }
      const consumer = new Consumer();
      let affine = [1, 0], curve = [0, 1];
      let baseAffine = [1, 0], baseCurve = [0, 1];
      const observed = [];
      Object.assign(consumer, {
        filledKVCacheLength: start, prefillChunkSize: chunkSize,
        drowsePositionSpans: spans, drowsePositionActive: true,
        drowsePositionEnabled: Uint32Array.of(1, 0),
        drowsePositionAffine: Uint32Array.of(1, 0), drowsePositionCurve: Uint32Array.of(0, 1),
        drowseProgram: structured ? undefined : { enabled: { copyFrom(value) { affine = [...value]; } } },
        drowseStructuredProgram: structured ? {} : undefined,
        async writeDrowseStructuredControls(a, c) { affine = [...a]; if (c !== undefined) curve = [...c]; },
        async embedAndForwardChunk(parts, length) {
          for (const token of parts.flat()) {
            const offset = this.filledKVCacheLength++ - start;
            const selected = !!(mask & (1 << offset));
            assert.deepEqual(affine, selected ? baseAffine : [0, 0], `affine mask=${mask} position=${offset}`);
            if (structured) assert.deepEqual(curve, selected ? baseCurve : [0, 0], `curve mask=${mask} position=${offset}`);
            observed.push(token);
          }
          assert.equal(length, parts.flat().length);
          return { dispose() {} };
        },
      });
      for (let offset = 0; offset < 8;) {
        const count = offset < 4 ? Math.min(4 - offset, chunkSize) : 1;
        if (structured) {
          baseAffine = [offset % 2, offset % 3 === 0 ? 1 : 0];
          baseCurve = [offset % 3 === 1 ? 1 : 0, offset % 2 === 0 ? 1 : 0];
          await consumer.updateDrowseStructuredControls(Uint32Array.from(baseAffine), Uint32Array.from(baseCurve));
        }
        await consumer.embedAndForward([Array.from({ length: count }, (_, i) => offset + i)], count);
        offset += count;
      }
      assert.deepEqual(observed, [0, 1, 2, 3, 4, 5, 6, 7]);
      assert.equal(consumer.filledKVCacheLength, start + 8);
      cases++;
    }
  }
  assert.equal(cases, 2048);
});

function fixture(kind, abi = "batch") {
  const consumer = new Consumer();
  const events = [];
  const kv = { name: "kv" };
  const rnn = { name: "rnn" };
  const embeddings = { name: "embeddings", shape: [2, 4], view() { return this; } };
  const positions = { name: "positions", copyFrom() { return this; } };
  const program = { name: "program" };
  const params = { name: "params" };
  const calls = [];
  const forward = (...args) => { calls.push(args); return args; };
  Object.assign(consumer, {
    resolvedModelABI: { prefillABI: abi, decodeABI: abi,
      needsKVCache: kind !== "rnn", needsRNNState: kind !== "kv" },
    kvCache: kind === "rnn" ? undefined : kv,
    rnnState: kind === "kv" ? undefined : rnn,
    params, prefill: forward, decoding: forward,
    drowsePrefill: forward, drowseDecoding: forward,
    drowseCapturePrefill: forward, drowseCaptureDecoding: forward,
    drowseRankOneCapturePrefillV1: forward, drowseRankOneCaptureDecodingV1: forward,
    prefillLogitPositions: positions, prefillLogitPositionHost: new Int32Array(1),
    getDrowseForwardArguments: () => consumer.drowseProgram ? [program] : [],
    filledKVCacheLength: 0,
    prefillChunkSize: 64,
    resetDrowseReadbackState: () => {},
    device: { async sync() {} },
    tvm: {
      beginScope: () => events.push("scope+"), endScope: () => events.push("scope-"),
      makeShapeTuple: (value) => value, cpu: () => "cpu",
      detachFromCurrentScope: (value) => value,
      attachToCurrentScope: (value) => value,
      empty: (shape, dtype) => ({ shape, dtype, copyFrom(value) { this.value = value; return this; },
        toArray() { return this.value.values; } }),
    },
    fKVCacheBeginForward: (state) => events.push(`begin:${state.name}`),
    fKVCacheEndForward: (state) => events.push(`end:${state.name}`),
    getTokensEmbeddings: () => embeddings,
    requireModelDimension: (name) => name === "num_hidden_layers" ? 2 : 4,
    resetKVCache: () => events.push("reset"),
  });
  return { consumer, events, kv, rnn, embeddings, positions, program, params, calls };
}

const variants = ["kv", "rnn", "hybrid"].flatMap((kind) =>
  (kind === "hybrid" ? ["batch"] : ["single", "batch"]).map((abi) => ({ kind, abi })));
for (const { kind, abi } of variants) {
  for (const phase of ["prefill", "decode"]) {
    for (const steered of [false, true]) {
      test(`${kind}/${abi}: ${phase} forwards states in order (${steered ? "steered" : "plain"})`, () => {
        const f = fixture(kind, abi);
        if (steered) f.consumer.drowseProgram = {};
        if (phase === "prefill") f.consumer.invokePrefill(f.embeddings, 2);
        else f.consumer.invokeDecode(f.embeddings);
        assert.deepEqual(f.calls[0], [f.embeddings,
          ...(phase === "prefill" && abi === "batch" ? [f.positions] : []),
          ...(kind !== "rnn" ? [f.kv] : []), ...(kind !== "kv" ? [f.rnn] : []),
          ...(steered ? [f.program] : []), f.params]);
        if (phase === "prefill" && abi === "batch") assert.equal(f.consumer.prefillLogitPositionHost[0], 1);
      });
    }
    for (const rankOne of [false, true]) {
      test(`${kind}/${abi}: ${phase} ${rankOne ? "rank-one " : ""}capture forwards every state`, () => {
        const f = fixture(kind, abi);
        if (rankOne) f.consumer.drowseProgram = {};
        const name = rankOne ? "invokeDrowseRankOneCaptureV1" : "invokeDrowseCapture";
        f.consumer[name](f.embeddings, phase === "prefill" ? 2 : 1, f.positions);
        assert.deepEqual(f.calls[0], [f.embeddings,
          ...(phase === "prefill" && abi === "batch" ? [f.positions] : []),
          ...(kind !== "rnn" ? [f.kv] : []), ...(kind !== "kv" ? [f.rnn] : []),
          f.positions, ...(rankOne ? [f.program] : []), f.params]);
      });
    }
  }
}

for (const kind of ["kv", "hybrid"]) {
  for (const rankOne of [false, true]) {
    test(`${kind}: ${rankOne ? "rank-one " : ""}capture reads tensors after returned states`, async () => {
      const f = fixture(kind);
      const values = Float32Array.from({ length: 8 }, (_, i) => i / 8);
      const captures = { dtype: "float32", shape: [2, 1, 4], values };
      const measurements = { name: "measurements", dispose() {} };
      const returned = [{ name: "logits" }, f.kv, ...(kind === "hybrid" ? [f.rnn] : []),
        ...(rankOne ? [measurements] : []), captures];
      const invoke = rankOne ? "invokeDrowseRankOneCaptureV1" : "invokeDrowseCapture";
      f.consumer[invoke] = () => ({ get: (index) => returned[index] });
      const capture = rankOne ? "captureDrowseRankOneTokenChunkV1" : "captureDrowseTokenChunk";
      try {
        assert.deepEqual(await f.consumer[capture]([4, 5], [1]), values);
        if (rankOne) assert.equal(f.consumer.drowseMeasurements, measurements);
        assert.equal(f.consumer.filledKVCacheLength, 2);
      } finally {
        assert.deepEqual(f.events, ["scope+", "begin:kv",
          ...(kind === "hybrid" ? ["begin:rnn", "end:rnn"] : []), "end:kv", "scope-"]);
      }
    });
  }
}

for (const rankOne of [false, true]) {
  test(`hybrid: second state begin failure closes only begun states (${rankOne ? "rank-one" : "plain"})`, async () => {
    const f = fixture("hybrid");
    f.consumer.fKVCacheBeginForward = (state) => {
      if (state === f.rnn) throw new Error("begin failed");
      f.events.push(`begin:${state.name}`);
    };
    const capture = rankOne ? "captureDrowseRankOneTokenChunkV1" : "captureDrowseTokenChunk";
    await assert.rejects(f.consumer[capture]([4, 5], [1]), /begin failed/);
    assert.deepEqual(f.events, ["scope+", "begin:kv", "end:kv", "scope-"]);
  });
  test(`hybrid: failed ${rankOne ? "rank-one " : ""}capture balances scopes and both states`, async () => {
    const f = fixture("hybrid");
    f.consumer.drowseProgram = {};
    const invoke = rankOne ? "invokeDrowseRankOneCaptureV1" : "invokeDrowseCapture";
    f.consumer[invoke] = () => { throw new Error("capture failed"); };
    const capture = rankOne ? "captureDrowseRankOneTokenChunkV1" : "captureDrowseTokenChunk";
    await assert.rejects(f.consumer[capture]([4, 5], [1]), /capture failed/);
    assert.equal(f.consumer.filledKVCacheLength, 0);
    assert.deepEqual(f.events, ["scope+", "begin:kv", "begin:rnn", "end:rnn", "end:kv", "scope-"]);
  });
}

for (const kind of ["kv", "hybrid"]) {
  test(`${kind}: generation reads measurements after all returned states`, async () => {
    const f = fixture(kind);
    const logits = { name: "logits" };
    const measurements = { name: "measurements" };
    f.consumer.drowseProgram = {};
    const returned = [logits, f.kv, ...(kind === "hybrid" ? [f.rnn] : []), measurements];
    f.consumer.invokePrefill = () => ({ get: (index) => returned[index] });
    assert.equal(await f.consumer.embedAndForward([[4, 5]], 2), logits);
    assert.equal(f.consumer.drowseMeasurements, measurements);
    assert.equal(f.consumer.filledKVCacheLength, 2);
  });
}

for (const failure of ["embedding", "second state", "forward"]) {
  test(`hybrid: generation cleans up after ${failure} failure`, async () => {
    const f = fixture("hybrid");
    if (failure === "embedding") f.consumer.getTokensEmbeddings = () => { throw new Error("failed"); };
    if (failure === "second state") f.consumer.fKVCacheBeginForward = (state) => {
      if (state === f.rnn) throw new Error("failed");
      f.events.push(`begin:${state.name}`);
    };
    if (failure === "forward") f.consumer.invokePrefill = () => { throw new Error("failed"); };
    await assert.rejects(f.consumer.embedAndForward([[4, 5]], 2), /failed/);
    assert.equal(f.consumer.filledKVCacheLength, 0);
    assert.deepEqual(f.events, ["scope+",
      ...(failure === "embedding" ? [] : ["begin:kv",
        ...(failure === "forward" ? ["begin:rnn", "end:rnn"] : []), "end:kv"]), "scope-"]);
  });
}
