import assert from "node:assert/strict";
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
const {
  BrowserFittingWorkerClient,
  FittingWorkerClientError,
} = await server.ssrLoadModule("/src/hosted/fitting/fittingWorkerClient.ts");

const { BrowserFittingWorkerPool, fittingJobWorkingBytes } = await server.ssrLoadModule("/src/hosted/fitting/fittingWorkerPool.ts");

const tests = [];
const test = (name, run) => tests.push({ name, run });
const centerJob = {
  operation: "center",
  source: {
    kind: "inline",
    values: new Float64Array([1, 2, 3, 4]),
    rows: 2,
    columns: 2,
  },
};
const centerResult = () => ({
  operation: "center",
  rows: 2,
  columns: 2,
  mean: new Float64Array([2, 3]),
  centered: new Float64Array([-1, -1, 1, 1]),
});

test("pool runs two independent jobs and preserves queued cancellation", async () => {
  const started = [];
  const pool = new BrowserFittingWorkerPool({
    workerFactory: () => new FakeWorker((request, port) => started.push({ request, port })),
  });
  const first = pool.run(centerJob);
  const second = pool.run(centerJob);
  const abort = new AbortController();
  const cancelled = pool.run(centerJob, { signal: abort.signal });
  const cancelledCheck = assert.rejects(cancelled, (error) => error.name === "AbortError");
  await Promise.resolve();
  assert.equal(started.length, 2);
  abort.abort();
  await cancelledCheck;
  for (const { request, port } of started) port.emit({ protocolVersion: 1, requestId: request.requestId, kind: "result", result: centerResult() });
  await Promise.all([first, second]);
  assert.equal(started.length, 2);
  pool.dispose();
});

test("pool serializes large spool jobs within the memory budget", async () => {
  const started = [];
  const pool = new BrowserFittingWorkerPool({
    workerFactory: () => new FakeWorker((request, port) => started.push({ request, port })),
  });
  const job = { ...centerJob, source: { kind: "activation_spool", identity: {}, layer: 0 } };
  const first = pool.run(job);
  const second = pool.run(job);
  await Promise.resolve();
  assert.equal(started.length, 1);
  started[0].port.emit({ protocolVersion: 1, requestId: started[0].request.requestId, kind: "result", result: centerResult() });
  await first;
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(started.length, 2);
  started[1].port.emit({ protocolVersion: 1, requestId: started[1].request.requestId, kind: "result", result: centerResult() });
  await second;
  pool.dispose();
});

test("pool admission includes quadratic outputs for tall and wide matrices", () => {
  for (const operation of ["pairwise", "correlations"]) {
    const rows = operation === "pairwise" ? 4096 : 2;
    const columns = operation === "correlations" ? 4096 : 2;
    assert.ok(fittingJobWorkingBytes({operation, source: {kind: "inline", rows, columns,
      values: new Float64Array(rows * columns)}}) >= 256 * 1024 * 1024);
  }
});

test("forwards progress and resolves a validated result", async () => {
  const stages = [];
  const worker = new FakeWorker((request, port) => {
    port.emit({
      protocolVersion: 1,
      requestId: request.requestId,
      kind: "progress",
      stage: "loading",
    });
    port.emit({
      protocolVersion: 1,
      requestId: request.requestId,
      kind: "result",
      result: centerResult(),
    });
  });
  const client = new BrowserFittingWorkerClient({
    workerFactory: () => worker,
    createRequestId: () => "fit-1",
  });
  const result = await client.run(centerJob, { onProgress: (stage) => stages.push(stage) });
  assert.deepEqual(stages, ["loading"]);
  assert.equal(result.operation, "center");
  assert.deepEqual([...result.mean], [2, 3]);
  assert.equal(worker.terminations, 0);
  client.dispose();
  assert.equal(worker.terminations, 1);
});

test("transfers only explicitly owned whole job buffers", async () => {
  const values = new Float64Array([1, 2, 3, 4]);
  const worker = new FakeWorker((request, port) => port.emit({
    protocolVersion: 1,
    requestId: request.requestId,
    kind: "result",
    result: centerResult(),
  }), { cloneMessages: true });
  const client = new BrowserFittingWorkerClient({ workerFactory: () => worker });
  const result = await client.run({
    operation: "center",
    source: { kind: "inline", values, rows: 2, columns: 2 },
  }, { transferOwnership: [values] });
  assert.equal(result.operation, "center");
  assert.equal(values.byteLength, 0);
  assert.equal(worker.transfers.length, 1);
  assert.equal(worker.transfers[0].length, 1);
  client.dispose();
});

test("keeps caller-owned and reused buffers attached unless ownership is explicit", async () => {
  const values = new Float64Array([1, 2, 3, 4]);
  const worker = new FakeWorker((request, port) => port.emit({
    protocolVersion: 1,
    requestId: request.requestId,
    kind: "result",
    result: centerResult(),
  }), { cloneMessages: true });
  const client = new BrowserFittingWorkerClient({ workerFactory: () => worker });
  await client.run({
    operation: "center",
    source: { kind: "inline", values, rows: 2, columns: 2 },
  });
  assert.equal(values.byteLength, 32);
  assert.deepEqual([...values], [1, 2, 3, 4]);
  assert.deepEqual(worker.transfers[0], []);
  client.dispose();
});

test("rejects partial or unrelated transfer ownership before spawning a worker", async () => {
  const values = new Float64Array([1, 2, 3, 4]);
  let spawned = 0;
  const client = new BrowserFittingWorkerClient({
    workerFactory: () => {
      spawned += 1;
      return new FakeWorker(() => {});
    },
  });
  const job = {
    operation: "center",
    source: { kind: "inline", values, rows: 2, columns: 2 },
  };
  await assert.rejects(
    client.run(job, { transferOwnership: [values.subarray(1)] }),
    { code: "FITTING_INVALID_TRANSFER_OWNERSHIP" },
  );
  await assert.rejects(
    client.run(job, { transferOwnership: [new Float64Array(4)] }),
    { code: "FITTING_INVALID_TRANSFER_OWNERSHIP" },
  );
  assert.equal(spawned, 0);
  assert.equal(values.byteLength, 32);
});

test("surfaces typed worker failures", async () => {
  const worker = new FakeWorker((request, port) => port.emit({
    protocolVersion: 1,
    requestId: request.requestId,
    kind: "error",
    error: {
      code: "FITTING_INVALID_MATRIX",
      message: "matrix is invalid",
      recoverable: false,
    },
  }));
  const client = new BrowserFittingWorkerClient({ workerFactory: () => worker });
  await assert.rejects(client.run(centerJob), (error) => {
    assert.ok(error instanceof FittingWorkerClientError);
    assert.equal(error.code, "FITTING_INVALID_MATRIX");
    assert.equal(error.recoverable, false);
    return true;
  });
  assert.equal(worker.terminations, 0);
  client.dispose();
  assert.equal(worker.terminations, 1);
});

test("reuses one worker and serializes sequential jobs", async () => {
  const requests = [];
  let spawned = 0;
  let nextRequestId = 0;
  const worker = new FakeWorker((request, port) => {
    requests.push(request.requestId);
    port.emit({
      protocolVersion: 1,
      requestId: request.requestId,
      kind: "result",
      result: centerResult(),
    });
  });
  const client = new BrowserFittingWorkerClient({
    workerFactory: () => {
      spawned += 1;
      return worker;
    },
    createRequestId: () => `fit-${++nextRequestId}`,
  });

  const first = client.run(centerJob);
  const second = client.run(centerJob);
  await Promise.resolve();
  assert.deepEqual(requests, ["fit-1"]);
  await first;
  await second;
  assert.deepEqual(requests, ["fit-1", "fit-2"]);
  assert.equal(spawned, 1);
  assert.equal(worker.terminations, 0);
  client.dispose();
  assert.equal(worker.terminations, 1);
});

test("rejects invalid or cross-request responses", async () => {
  const worker = new FakeWorker((_request, port) => port.emit({
    protocolVersion: 1,
    requestId: "wrong-request",
    kind: "result",
    result: { operation: "center" },
  }));
  const client = new BrowserFittingWorkerClient({ workerFactory: () => worker });
  await assert.rejects(client.run(centerJob), { code: "FITTING_WORKER_PROTOCOL_ERROR" });
  assert.equal(worker.terminations, 1);
});

test("rejects malformed and wrong-operation results", async () => {
  for (const result of [
    { operation: "center" },
    {
      operation: "pca",
      rows: 1,
      columns: 1,
      components: 1,
      mean: new Float64Array(1),
      basis: new Float64Array(1),
      eigenvalues: new Float64Array(1),
      explainedVariance: new Float64Array(1),
      cumulativeVariance: new Float64Array(1),
      scores: new Float64Array(1),
    },
  ]) {
    const worker = new FakeWorker((request, port) => port.emit({
      protocolVersion: 1,
      requestId: request.requestId,
      kind: "result",
      result,
    }));
    const client = new BrowserFittingWorkerClient({ workerFactory: () => worker });
    await assert.rejects(client.run(centerJob), { code: "FITTING_WORKER_PROTOCOL_ERROR" });
    assert.equal(worker.terminations, 1);
  }
});

test("terminates when sending the job fails synchronously", async () => {
  const worker = new FakeWorker(() => {});
  worker.postMessage = () => { throw new DOMException("cannot clone", "DataCloneError"); };
  const client = new BrowserFittingWorkerClient({ workerFactory: () => worker });
  await assert.rejects(client.run(centerJob), { code: "FITTING_WORKER_POST_FAILED" });
  assert.equal(worker.terminations, 1);
});

test("terminates when a progress callback throws", async () => {
  const worker = new FakeWorker((request, port) => port.emit({
    protocolVersion: 1,
    requestId: request.requestId,
    kind: "progress",
    stage: "loading",
  }));
  const client = new BrowserFittingWorkerClient({ workerFactory: () => worker });
  await assert.rejects(client.run(centerJob, {
    onProgress: () => { throw new Error("progress failed"); },
  }), { code: "FITTING_PROGRESS_HANDLER_FAILED" });
  assert.equal(worker.terminations, 1);
});

test("does not spawn a worker for an already-cancelled job", async () => {
  const controller = new AbortController();
  controller.abort();
  let spawned = false;
  const client = new BrowserFittingWorkerClient({
    workerFactory: () => {
      spawned = true;
      return new FakeWorker(() => {});
    },
  });
  await assert.rejects(client.run(centerJob, { signal: controller.signal }), {
    code: "FITTING_CANCELLED",
  });
  assert.equal(spawned, false);
});

test("terminates an active worker on cancellation", async () => {
  const worker = new FakeWorker(() => {});
  const controller = new AbortController();
  const client = new BrowserFittingWorkerClient({ workerFactory: () => worker });
  const running = client.run(centerJob, { signal: controller.signal });
  controller.abort();
  await assert.rejects(running, { code: "FITTING_CANCELLED" });
  assert.equal(worker.terminations, 1);
});

test("an aborted job recreates the worker for the next job", async () => {
  const firstWorker = new FakeWorker(() => {});
  const secondWorker = new FakeWorker((request, port) => port.emit({
    protocolVersion: 1,
    requestId: request.requestId,
    kind: "result",
    result: centerResult(),
  }));
  const workers = [firstWorker, secondWorker];
  const controller = new AbortController();
  const client = new BrowserFittingWorkerClient({
    workerFactory: () => workers.shift(),
  });

  const cancelledRun = client.run(centerJob, { signal: controller.signal });
  controller.abort();
  await assert.rejects(cancelledRun, { code: "FITTING_CANCELLED" });
  const result = await client.run(centerJob);
  assert.equal(result.operation, "center");
  assert.equal(firstWorker.terminations, 1);
  assert.equal(secondWorker.terminations, 0);
  client.dispose();
  assert.equal(secondWorker.terminations, 1);
});

test("a worker crash rejects its job and recreates for queued work", async () => {
  const firstWorker = new FakeWorker(() => {});
  const secondWorker = new FakeWorker((request, port) => port.emit({
    protocolVersion: 1,
    requestId: request.requestId,
    kind: "result",
    result: centerResult(),
  }));
  const workers = [firstWorker, secondWorker];
  const client = new BrowserFittingWorkerClient({
    workerFactory: () => workers.shift(),
  });

  const crashedRun = client.run(centerJob);
  const queuedRun = client.run(centerJob);
  firstWorker.crash("wasm worker crashed");
  await assert.rejects(crashedRun, (error) => {
    assert.equal(error.code, "FITTING_WORKER_CRASHED");
    assert.equal(error.message, "wasm worker crashed");
    return true;
  });
  assert.equal((await queuedRun).operation, "center");
  assert.equal(firstWorker.terminations, 1);
  assert.equal(secondWorker.terminations, 0);
  client.dispose();
  assert.equal(secondWorker.terminations, 1);
});

test("cancelling queued work leaves the active worker running", async () => {
  const worker = new FakeWorker(() => {});
  const controller = new AbortController();
  let nextRequestId = 0;
  const client = new BrowserFittingWorkerClient({
    workerFactory: () => worker,
    createRequestId: () => `fit-${++nextRequestId}`,
  });
  const active = client.run(centerJob);
  const queued = client.run(centerJob, { signal: controller.signal });
  controller.abort();
  await assert.rejects(queued, { code: "FITTING_CANCELLED" });
  assert.equal(worker.terminations, 0);
  worker.emit({
    protocolVersion: 1,
    requestId: "fit-1",
    kind: "result",
    result: centerResult(),
  });
  assert.equal((await active).operation, "center");
  assert.equal(worker.messages.length, 1);
  client.dispose();
});

test("dispose releases the retained worker and a later run recreates it", async () => {
  const workers = [
    new FakeWorker((request, port) => port.emit({
      protocolVersion: 1,
      requestId: request.requestId,
      kind: "result",
      result: centerResult(),
    })),
    new FakeWorker((request, port) => port.emit({
      protocolVersion: 1,
      requestId: request.requestId,
      kind: "result",
      result: centerResult(),
    })),
  ];
  let nextWorker = 0;
  const client = new BrowserFittingWorkerClient({
    workerFactory: () => workers[nextWorker++],
  });
  await client.run(centerJob);
  client.dispose();
  await client.run(centerJob);
  assert.equal(nextWorker, 2);
  assert.equal(workers[0].terminations, 1);
  assert.equal(workers[1].terminations, 0);
  client.dispose();
  assert.equal(workers[1].terminations, 1);
});

test("terminates a stalled worker at the configured safety timeout", async () => {
  const worker = new FakeWorker(() => {});
  const client = new BrowserFittingWorkerClient({
    workerFactory: () => worker,
    requestTimeoutMs: 5,
  });
  await assert.rejects(client.run(centerJob), { code: "FITTING_WORKER_TIMEOUT" });
  assert.equal(worker.terminations, 1);
});

class FakeWorker {
  onmessage = null;
  onerror = null;
  onmessageerror = null;
  terminations = 0;
  messages = [];
  transfers = [];

  constructor(handler, { cloneMessages = false } = {}) {
    this.handler = handler;
    this.cloneMessages = cloneMessages;
  }

  postMessage(message, transfer = []) {
    this.transfers.push(transfer);
    const delivered = this.cloneMessages
      ? structuredClone(message, { transfer })
      : message;
    this.messages.push(delivered);
    queueMicrotask(() => this.handler(delivered, this));
  }

  terminate() {
    this.terminations += 1;
  }

  emit(data) {
    this.onmessage?.({ data });
  }

  crash(message) {
    this.onerror?.({ message });
  }
}

let failed = 0;
try {
  for (const { name, run } of tests) {
    try {
      await run();
      console.log(`ok - ${name}`);
    } catch (error) {
      failed += 1;
      console.error(`not ok - ${name}`);
      console.error(error);
    }
  }
} finally {
  await server.close();
}
if (failed) process.exitCode = 1;
else console.log(`${tests.length} fitting-worker client tests passed`);
