import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const server = await createServer({ root: fileURLToPath(new URL("..", import.meta.url)),
  configFile: false, appType: "custom", logLevel: "silent",
  server: { middlewareMode: true, watch: null } });
const flush = () => new Promise(resolve => setImmediate(resolve));

try {
  const { JobRegistry } = await server.ssrLoadModule("/src/lib/webmcp/jobs.ts");
  const { awaitGenerationReceipt } = await server.ssrLoadModule("/src/lib/webmcp/generationReceipt.ts");
  const jobs = new JobRegistry(2);
  let calls = 0;
  const queued = jobs.start("generate", async () => { calls += 1; }, { requestId: "same" });
  assert.equal(jobs.start("generate", async () => { throw new Error("duplicate"); }, { requestId: "same" }).id, queued.id);
  await jobs.cancel(queued.id);
  await flush();
  assert.equal(calls, 0);
  assert.equal(jobs.get(queued.id).state, "cancelled");
  const active = jobs.start("generate", ({ signal }) => new Promise(resolve => {
    signal.addEventListener("abort", () => setTimeout(() => resolve({ partial: true }), 5), { once: true });
  }));
  await flush();
  assert.equal((await jobs.cancel(active.id)).state, "cancelling");
  await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(jobs.get(active.id).state, "cancelled");
  assert.deepEqual(jobs.get(active.id).result, { partial: true });
  const uncancellable = jobs.start("server_fit", async () => {}, { cancellable: false });
  await assert.rejects(jobs.cancel(uncancellable.id), error => error.code === "JOB_NOT_CANCELLABLE");
  await flush();
  assert.equal(jobs.list().length, 2, "only a bounded number of terminal results are retained");
  const interrupted = jobs.start("generate", () => new Promise(() => {}));
  await flush();
  jobs.interrupt("model unloaded");
  assert.equal(jobs.get(interrupted.id).state, "interrupted");
  const failed = jobs.start("generate", async () => { throw Object.assign(new Error("bad selector"), { code: "BAD_SELECTOR" }); });
  await flush();
  assert.equal(jobs.get(failed.id).error.code, "BAD_SELECTOR");
  let finishLate;
  const late = jobs.start("late_cancel", () => new Promise(resolve => { finishLate = resolve; }));
  await flush();
  await jobs.cancel(late.id);
  finishLate({ state: "completed" });
  await flush();
  assert.equal(jobs.get(late.id).state, "completed", "late cancellation cannot relabel successful backend completion");
  const cancellationFailure = jobs.start("stop_failure", ({ signal }) => new Promise((resolve, reject) => {
    signal.addEventListener("abort", () => reject(Object.assign(new Error("stop failed"), { code: "WORKER_STOP_FAILED" })));
  }));
  await flush();
  await jobs.cancel(cancellationFailure.id);
  await flush();
  assert.equal(jobs.get(cancellationFailure.id).state, "failed", "failed cancellation is not reported as cancelled compute");

  const stored = new Map();
  const storage = { getItem: key => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, value) };
  const durable = new JobRegistry();
  durable.attachPersistence(storage, "test");
  const durableJob = durable.start("generate", () => new Promise(() => {}), { requestId: "reload-safe", recovery: { request: "immutable" } });
  await flush();
  durable.checkpoint(durableJob.id, { node_ids: ["completed-sibling"] });
  const reloaded = new JobRegistry();
  reloaded.attachPersistence(storage, "test");
  assert.equal(reloaded.get(durableJob.id).state, "interrupted");
  assert.deepEqual(reloaded.get(durableJob.id).result.node_ids, ["completed-sibling"]);
  let duplicateStarted = false;
  assert.equal(reloaded.start("generate", async () => { duplicateStarted = true; }, { requestId: "reload-safe" }).id, durableJob.id);
  await flush();
  assert.equal(duplicateStarted, false, "reload never silently resends unfinished work");
  reloaded.reconcile(durableJob.id, { state: "completed", result: { confirmed: true } });
  const restoredTerminal = new JobRegistry();
  restoredTerminal.attachPersistence(storage, "test");
  assert.equal(restoredTerminal.get(durableJob.id).state, "completed");
  const large = durable.start("generate", () => new Promise(() => {}), { requestId: "large-receipt" });
  await flush();
  durable.checkpoint(large.id, { node_ids: ["retrieve-full-node"], results: [{ node_id: "retrieve-full-node", result: { text: "a".repeat(5 * 1024 * 1024), measurements: { heavy: "x".repeat(5 * 1024 * 1024) } } }] });
  assert.equal(durable.persistenceError, null);
  const recoveredLarge = new JobRegistry();
  recoveredLarge.attachPersistence(storage, "test");
  assert.deepEqual(recoveredLarge.get(large.id).result.node_ids, ["retrieve-full-node"]);
  assert.ok(recoveredLarge.get(large.id).persistedDetailOmissions.length > 0, "large receipts retain ownership and retrieval pointers with explicit omissions");
  const quota = new JobRegistry();
  quota.attachPersistence({ getItem: () => null, setItem: () => { throw new Error("quota"); } }, "quota");
  assert.match(quota.persistenceError, /could not be saved/);
  let finishDownload;
  const cancelFailure = jobs.start("download", () => new Promise(resolve => { finishDownload = resolve; }), {
    cancel: () => { throw new Error("cannot stop"); },
  });
  await flush();
  assert.equal((await jobs.cancel(cancelFailure.id)).error.code, "CANCELLATION_FAILED");
  finishDownload({ installed: true, state: "completed" });
  await flush();
  assert.equal(jobs.get(cancelFailure.id).state, "completed");

  const fixture = (id = "ours", n = 2) => {
    const listeners = new Set();
    const controller = new AbortController();
    let sent = 0;
    let stopped = 0;
    const promise = awaitGenerationReceipt({ requestId: id, expectedSiblings: n, signal: controller.signal,
      subscribe: callback => { listeners.add(callback); return () => listeners.delete(callback); },
      send: () => { sent += 1; }, stop: () => { stopped += 1; } });
    return { promise, controller, emit: event => { for (const listener of listeners) listener(event); },
      counts: () => ({ sent, stopped, listeners: listeners.size }) };
  };
  const done = (request_id, sibling_index, finish_reason = "stop") => ({ type: "done", request_id,
    node_id: `node-${sibling_index}`, sibling_index, sibling_count: 2,
    result: { text: `answer-${sibling_index}`, tokens: 1, finish_reason,
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } } });
  const receipt = fixture();
  let settled = false;
  receipt.promise.then(() => { settled = true; });
  receipt.emit({ type: "error", request_id: "another", message: "unrelated failure" });
  receipt.emit(done("ours", 0));
  await flush();
  assert.equal(settled, false, "one done is not completion of the fan");
  receipt.emit(done("ours", 1));
  await flush();
  assert.equal(settled, false, "wait for transport completion after final sibling persistence");
  receipt.emit({ type: "request_complete", request_id: "ours", state: "completed", completed_siblings: 2 });
  assert.deepEqual((await receipt.promise).results.map(row => row.node_id), ["node-0", "node-1"]);
  assert.equal(receipt.counts().listeners, 0);
  const cancel = fixture();
  cancel.controller.abort();
  assert.equal(cancel.counts().stopped, 1);
  cancel.emit(done("ours", 0, "cancelled"));
  cancel.emit({ type: "request_complete", request_id: "ours", state: "cancelled", completed_siblings: 1 });
  assert.equal((await cancel.promise).state, "cancelled");
  const incomplete = fixture();
  incomplete.emit(done("ours", 0));
  incomplete.emit({ type: "request_complete", request_id: "ours", state: "completed", completed_siblings: 1 });
  await assert.rejects(incomplete.promise, error => error.code === "GENERATION_RESULT_INCOMPLETE" && error.partialResult.results.length === 1);
  const lost = fixture();
  lost.emit({ type: "error", code: "RUNTIME_CHANNEL_CLOSED", message: "gone" });
  await assert.rejects(lost.promise, error => error.code === "GENERATION_INTERRUPTED");

  const recoveredReceipt = awaitGenerationReceipt({ requestId: "lost-terminal", expectedSiblings: 1, signal: new AbortController().signal,
    subscribe: () => () => {}, send: () => {}, stop: () => {}, idleTimeoutMs: 5,
    requestStatus: async () => ({ request_id: "lost-terminal", state: "completed", results: [done("lost-terminal", 0)] }),
  });
  assert.equal((await recoveredReceipt).state, "completed", "missing terminal is recovered only from authoritative status");

  const { RuntimeRequestJournal } = await server.ssrLoadModule("/src/lib/runtime/requestJournal.ts");
  globalThis.sessionStorage = storage;
  const journal = new RuntimeRequestJournal();
  const commandPayload = { type: "generate", request_id: "persisted", input: "hello" };
  assert.equal(journal.claim(commandPayload), "new");
  journal.observe(done("persisted", 0));
  journal.observe({ type: "request_complete", request_id: "persisted", state: "completed", completed_siblings: 1 });
  const journalReload = new RuntimeRequestJournal();
  assert.equal(journalReload.claim(commandPayload), "existing");
  assert.equal(journalReload.claim({ ...commandPayload, input: "different" }), "conflict");
  assert.equal(journalReload.get("persisted").results.length, 1);
  delete globalThis.sessionStorage;

  const { WorkerRpcTransport, BrowserRuntimeClient } = await server.ssrLoadModule("/src/hosted/runtime/browserRuntimeClient.ts");
  const listeners = new Set();
  const posted = [];
  const worker = { postMessage: message => posted.push(message),
    addEventListener: (type, listener) => { if (type === "message") listeners.add(listener); },
    removeEventListener: (type, listener) => listeners.delete(listener), terminate() {} };
  const transport = new WorkerRpcTransport(worker, { requestTimeoutMs: 0 });
  const client = new BrowserRuntimeClient(transport, { ownsTransport: false });
  const events = [];
  client.events.subscribe(event => events.push(event));
  await client.events.open();
  client.events.send({ type: "submit", request_id: "our-job", text: "hi", authored_role: "user", generated_role: "assistant" });
  const request = posted[0];
  let sequence = 0;
  const emit = message => { for (const listener of listeners) listener({ data: message }); };
  const runtimeEvent = payload => emit({ protocolVersion: 1, kind: "event", requestId: request.requestId,
    sequence: ++sequence, generationId: "gen-1", event: payload.type, payload });
  runtimeEvent({ type: "started", request_id: "our-job", generation_id: "gen-1", node_id: "node-0", sibling_index: 0, sibling_count: 1 });
  runtimeEvent({ ...done("our-job", 0), sibling_count: 1 });
  assert.equal(events.some(event => event.type === "request_complete"), false);
  emit({ protocolVersion: 1, kind: "response", requestId: request.requestId, ok: true, result: undefined });
  await flush();
  assert.deepEqual(events.at(-1), { type: "request_complete", request_id: "our-job", state: "completed", completed_siblings: 1 });
  const admittedCount = posted.length;
  client.events.send({ type: "submit", request_id: "our-job", text: "hi", authored_role: "user", generated_role: "assistant" });
  assert.equal(posted.length, admittedCount, "repeated IDs replay a receipt without another worker command");
  assert.equal((await client.events.requestStatus("our-job")).state, "completed");
  client.events.send({ type: "stop", request_id: "our-job" });
  assert.deepEqual(posted.at(-1).payload, { requestId: "our-job" }, "cancellation retains ownership identity");
  emit({ protocolVersion: 1, kind: "response", requestId: posted.at(-1).requestId, ok: true, result: undefined });
  await flush();
  transport.dispose();
  console.log("WebMCP job, generation receipt and browser correlation checks passed");
} finally { await server.close(); }
