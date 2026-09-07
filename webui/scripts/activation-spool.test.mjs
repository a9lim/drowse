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

const tests = [];
const test = (name, run) => tests.push({ name, run });
let ActivationSpoolClass;

try {
  const {
    ACTIVATION_SPOOL_MAX_CHUNK_BYTES,
    ActivationSpool,
    ActivationSpoolError,
    BrowserActivationSpoolFilePort,
    activationSpoolFileName,
    withBrowserActivationSpool,
  } = await server.ssrLoadModule("/src/hosted/fitting/activationSpool.ts");
  ActivationSpoolClass = ActivationSpool;

  test("maps an OPFS directory at a layer filename to recoverable corruption", async () => {
    const previous = Object.getOwnPropertyDescriptor(globalThis, "navigator");
    const activationDirectory = {
      async getFileHandle() {
        throw new DOMException("entry is a directory", "TypeMismatchError");
      },
    };
    const fittingDirectory = {
      async getDirectoryHandle(name) {
        assert.equal(name, "activation-spool");
        return activationDirectory;
      },
    };
    const drowseDirectory = {
      async getDirectoryHandle(name) {
        assert.equal(name, "fitting");
        return fittingDirectory;
      },
    };
    const rootDirectory = {
      async getDirectoryHandle(name) {
        assert.equal(name, "drowse");
        return drowseDirectory;
      },
    };
    Object.defineProperty(globalThis, "navigator", {
      configurable: true,
      value: { storage: { async getDirectory() { return rootDirectory; } } },
    });
    try {
      const port = new BrowserActivationSpoolFilePort();
      await port.initialize();
      await assert.rejects(
        port.size(activationSpoolFileName("a".repeat(64), 0)),
        errorCode(ActivationSpoolError, "STORAGE_MISMATCH"),
      );
    } finally {
      if (previous) Object.defineProperty(globalThis, "navigator", previous);
      else delete globalThis.navigator;
    }
  });

  test("closes every temporary browser spool after success and failure", async () => {
    let opens = 0;
    let closes = 0;
    const createSpool = () => {
      opens += 1;
      return { close() { closes += 1; } };
    };
    for (let index = 0; index < 4; index += 1) {
      assert.equal(await withBrowserActivationSpool(async () => index, createSpool), index);
      await assert.rejects(
        withBrowserActivationSpool(async () => { throw new Error(`failure-${index}`); }, createSpool),
        new RegExp(`failure-${index}`),
      );
    }
    assert.equal(opens, 8);
    assert.equal(closes, opens);
  });

  test("passes the exact Uint8Array view to OPFS without copying it", async () => {
    const previous = Object.getOwnPropertyDescriptor(globalThis, "navigator");
    let written = null;
    const writable = {
      async write(value) { written = value; },
      async close() {},
      async abort() {},
    };
    const fileHandle = {
      async getFile() { return { size: 0 }; },
      async createWritable() { return writable; },
    };
    const activationDirectory = {
      async getFileHandle() { return fileHandle; },
    };
    const fittingDirectory = {
      async getDirectoryHandle() { return activationDirectory; },
    };
    const drowseDirectory = {
      async getDirectoryHandle() { return fittingDirectory; },
    };
    const rootDirectory = {
      async getDirectoryHandle() { return drowseDirectory; },
    };
    Object.defineProperty(globalThis, "navigator", {
      configurable: true,
      value: { storage: { async getDirectory() { return rootDirectory; } } },
    });
    try {
      const port = new BrowserActivationSpoolFilePort();
      await port.initialize();
      const backing = new Uint8Array([9, 1, 2, 3, 8]);
      const view = backing.subarray(1, 4);
      await port.append(activationSpoolFileName("a".repeat(64), 0), 0, view);
      assert.equal(written, view);
      assert.deepEqual([...written], [1, 2, 3]);
    } finally {
      if (previous) Object.defineProperty(globalThis, "navigator", previous);
      else delete globalThis.navigator;
    }
  });

  test("writes contiguous float32 rows and commits only exact sealed layers", async () => {
    const { spool, metadata, files } = harness(1_000);
    const capture = descriptor("a", [
      { layer: 1, rows: 3, width: 2, expectedBytes: 24, ignoredProducerField: true },
      { layer: 4, rows: 2, width: 3, expectedBytes: 24 },
    ]);
    const writer = await spool.begin(capture);

    await assert.rejects(
      writer.appendRows(1, 1, new Float32Array([1, 2])),
      errorCode(ActivationSpoolError, "OUT_OF_ORDER_WRITE"),
    );
    await assert.rejects(
      writer.appendRows(1, 0, new Float32Array([1, Number.NaN])),
      errorCode(ActivationSpoolError, "NONFINITE_VALUES"),
    );
    await assert.rejects(
      writer.appendRows(1, 0, new Float32Array([1, 2, 3])),
      errorCode(ActivationSpoolError, "OUT_OF_ORDER_WRITE"),
    );

    const first = new Float32Array([1, 2, 3, 4]);
    await writer.appendRows(1, 0, first);
    first.fill(99);
    await writer.checkpoint(1);
    await assert.rejects(
      writer.appendRows(1, 0, new Float32Array([1, 2])),
      errorCode(ActivationSpoolError, "OUT_OF_ORDER_WRITE"),
    );
    await assert.rejects(
      writer.sealLayer(1),
      errorCode(ActivationSpoolError, "INCOMPLETE_LAYER"),
    );
    await writer.appendRows(1, 2, new Float32Array([5, 6]));
    const sealedOne = await writer.sealLayer(1);
    assert.deepEqual(
      pickProgress(sealedOne),
      { writtenRows: 3, writtenBytes: 24, checkpointRows: 3, checkpointBytes: 24, sealed: true },
    );
    await assert.rejects(
      writer.appendRows(1, 3, new Float32Array([7, 8])),
      errorCode(ActivationSpoolError, "OUT_OF_ORDER_WRITE"),
    );
    await assert.rejects(
      writer.commit(),
      errorCode(ActivationSpoolError, "INCOMPLETE_LAYER"),
    );

    await writer.appendRows(4, 0, new Float32Array([7, 8, 9, 10, 11, 12]));
    await writer.sealLayer(4);
    const reader = await writer.commit();
    assert.equal(metadata.rows.get(capture.captureSha256).state, "committed");
    assert.equal(metadata.rows.get(capture.captureSha256).writeEpoch, null);
    assert.equal("ignoredProducerField" in metadata.rows.get(capture.captureSha256).layers[0], false);
    assert.equal(files.rows.get(activationSpoolFileName(capture.captureSha256, 1)).byteLength, 24);

    const chunks = await collect(reader.readChunks(1, { maxChunkBytes: 8 }));
    assert.deepEqual(chunks.map(({ startRow, rows }) => ({ startRow, rows })), [
      { startRow: 0, rows: 1 },
      { startRow: 1, rows: 1 },
      { startRow: 2, rows: 1 },
    ]);
    assert.deepEqual(chunks.flatMap((chunk) => [...chunk.values]), [1, 2, 3, 4, 5, 6]);
    assert.ok(files.maximumReadBytes <= 8);
    await assert.rejects(
      collect(reader.readChunks(1, { maxChunkBytes: 4 })),
      errorCode(ActivationSpoolError, "INVALID_DESCRIPTOR"),
    );

    chunks[0].values.fill(88);
    assert.deepEqual(
      [...(await collect(reader.readChunks(1, { startRow: 0, rowCount: 1 })))[0].values],
      [1, 2],
    );
    await assert.rejects(
      async () => writer.checkpoint(1),
      errorCode(ActivationSpoolError, "STALE_WRITER"),
    );
  });

  test("binds resume and committed reads to all three identity digests and exact shapes", async () => {
    const { spool } = harness(2_000);
    const capture = descriptor("b", [
      { layer: 0, rows: 1, width: 2, expectedBytes: 8 },
    ]);
    const writer = await spool.begin(capture);

    await assert.rejects(
      spool.begin(capture),
      errorCode(ActivationSpoolError, "CAPTURE_EXISTS"),
    );
    await assert.rejects(
      spool.resume({ ...capture, runtimeIdentitySha256: "9".repeat(64) }),
      errorCode(ActivationSpoolError, "IDENTITY_MISMATCH"),
    );
    await assert.rejects(
      spool.resume({
        ...capture,
        layers: [{ layer: 0, rows: 2, width: 2, expectedBytes: 16 }],
      }),
      errorCode(ActivationSpoolError, "LAYER_MISMATCH"),
    );
    await assert.rejects(
      spool.openCommitted(capture),
      errorCode(ActivationSpoolError, "NOT_COMMITTED"),
    );

    await writer.appendRows(0, 0, new Float32Array([1, 2]));
    await writer.sealLayer(0);
    await writer.commit();
    await assert.rejects(
      spool.openCommitted({
        runtimeIdentitySha256: capture.runtimeIdentitySha256,
        contextBindingSha256: "8".repeat(64),
        captureSha256: capture.captureSha256,
      }),
      errorCode(ActivationSpoolError, "IDENTITY_MISMATCH"),
    );
    const reader = await spool.openCommitted(capture);
    assert.deepEqual([...(await collect(reader.readChunks(0)))[0].values], [1, 2]);
  });

  test("snapshots mutable descriptors before storage or lease awaits", async () => {
    const { spool, metadata } = harness(2_500);
    const capture = descriptor("6", [
      { layer: 1, rows: 1, width: 2, expectedBytes: 8 },
    ]);
    const originalSha = capture.captureSha256;
    const pending = spool.begin(capture);
    capture.captureSha256 = "f".repeat(64);
    capture.layers[0].rows = 9;
    capture.layers[0].expectedBytes = 72;
    const writer = await pending;
    assert.equal(writer.identity.captureSha256, originalSha);
    assert.equal(writer.layers[0].rows, 1);
    assert.equal(metadata.rows.has(originalSha), true);
    assert.equal(metadata.rows.has(capture.captureSha256), false);
    await writer.rollback();
  });

  test("bounds every append before encoding or file I/O", async () => {
    const { spool, files } = harness(2_750);
    const rows = ACTIVATION_SPOOL_MAX_CHUNK_BYTES / 4 + 1;
    const capture = descriptor("7", [
      { layer: 0, rows, width: 1, expectedBytes: rows * 4 },
    ]);
    const writer = await spool.begin(capture);
    await assert.rejects(
      writer.appendRows(0, 0, new Float32Array(rows)),
      errorCode(ActivationSpoolError, "INVALID_DESCRIPTOR"),
    );
    assert.equal(files.rows.get(activationSpoolFileName(capture.captureSha256, 0)).byteLength, 0);
    await writer.rollback();
  });

  test("recovers only checkpointed rows and invalidates the crashed writer", async () => {
    const shared = sharedPorts();
    const first = harness(3_000, shared);
    const capture = descriptor("c", [
      { layer: 2, rows: 3, width: 2, expectedBytes: 24 },
    ]);
    const crashed = await first.spool.begin(capture);
    await crashed.appendRows(2, 0, new Float32Array([1, 2]));
    await crashed.checkpoint(2);
    await crashed.appendRows(2, 1, new Float32Array([3, 4]));
    assert.equal(shared.files.rows.get(activationSpoolFileName(capture.captureSha256, 2)).byteLength, 16);

    const observer = harness(3_500, shared);
    await observer.spool.initialize();
    assert.equal(shared.files.rows.get(activationSpoolFileName(capture.captureSha256, 2)).byteLength, 16);
    assert.notEqual(shared.metadata.rows.get(capture.captureSha256).writeEpoch, null);
    await assert.rejects(
      observer.spool.resume(capture),
      errorCode(ActivationSpoolError, "STALE_WRITER"),
    );

    shared.leases.crash(capture.captureSha256);
    const recovered = harness(4_000, shared);
    await recovered.spool.initialize();
    assert.equal(shared.files.rows.get(activationSpoolFileName(capture.captureSha256, 2)).byteLength, 8);
    const record = shared.metadata.rows.get(capture.captureSha256);
    assert.deepEqual(pickProgress(record.layers[0]), {
      writtenRows: 1,
      writtenBytes: 8,
      checkpointRows: 1,
      checkpointBytes: 8,
      sealed: false,
    });
    assert.equal(record.writeEpoch, null);
    await assert.rejects(
      crashed.appendRows(2, 2, new Float32Array([5, 6])),
      errorCode(ActivationSpoolError, "STALE_WRITER"),
    );

    const resumed = await recovered.spool.resume(capture);
    assert.equal(resumed.progress[0].writtenRows, 1);
    assert.deepEqual(
      [...(await collect(resumed.readChunks(2)))[0].values],
      [1, 2],
    );
    await resumed.appendRows(2, 1, new Float32Array([3, 4, 5, 6]));
    await resumed.sealLayer(2);
    const reader = await resumed.commit();
    assert.deepEqual(
      (await collect(reader.readChunks(2))).flatMap((chunk) => [...chunk.values]),
      [1, 2, 3, 4, 5, 6],
    );
  });

  test("removes short, malformed, and orphaned stages during crash recovery", async () => {
    const shared = sharedPorts();
    const short = descriptor("d", [
      { layer: 0, rows: 2, width: 2, expectedBytes: 16 },
    ]);
    const started = harness(5_000, shared);
    const writer = await started.spool.begin(short);
    await writer.appendRows(0, 0, new Float32Array([1, 2]));
    await writer.checkpoint(0);
    shared.files.rows.set(activationSpoolFileName(short.captureSha256, 0), new Uint8Array(4));
    shared.leases.crash(short.captureSha256);

    const malformedSha = "e".repeat(64);
    shared.metadata.rows.set(malformedSha, {
      schemaVersion: 1,
      captureSha256: malformedSha,
      state: "staged",
    });
    shared.files.rows.set(activationSpoolFileName(malformedSha, 9), new Uint8Array(8));
    const orphanSha = "f".repeat(64);
    shared.files.rows.set(activationSpoolFileName(orphanSha, 7), new Uint8Array(8));
    shared.files.rows.set("malformed-orphan.tmp", new Uint8Array(8));

    const recovered = harness(6_000, shared);
    await recovered.spool.initialize();
    assert.equal(shared.metadata.rows.has(short.captureSha256), false);
    assert.equal(shared.metadata.rows.has(malformedSha), false);
    assert.equal(shared.files.rows.size, 0);
  });

  test("rollback removes a stage without exposing it as a committed capture", async () => {
    const { spool, metadata, files } = harness(7_000);
    const capture = descriptor("1", [
      { layer: 3, rows: 2, width: 1, expectedBytes: 8 },
    ]);
    const writer = await spool.begin(capture);
    await writer.appendRows(3, 0, new Float32Array([4]));
    await writer.checkpoint(3);
    await writer.rollback();
    await writer.rollback();
    assert.equal(metadata.rows.has(capture.captureSha256), false);
    assert.equal(files.rows.size, 0);
    await assert.rejects(
      spool.openCommitted(capture),
      errorCode(ActivationSpoolError, "CAPTURE_NOT_FOUND"),
    );
  });

  test("removes only an exact committed capture identity", async () => {
    const { spool, metadata, files } = harness(7_250);
    const capture = descriptor("3", [
      { layer: 2, rows: 2, width: 1, expectedBytes: 8 },
    ]);
    const writer = await spool.begin(capture);
    await writer.appendRows(2, 0, new Float32Array([4, 8]));
    await writer.sealLayer(2);
    await writer.commit();
    await assert.rejects(
      spool.removeCommitted({ ...capture, runtimeIdentitySha256: "f".repeat(64) }),
      errorCode(ActivationSpoolError, "IDENTITY_MISMATCH"),
    );
    assert.equal(metadata.rows.has(capture.captureSha256), true);
    assert.equal(await spool.removeCommitted(capture), true);
    assert.equal(await spool.removeCommitted(capture), false);
    assert.equal(metadata.rows.size, 0);
    assert.equal(files.rows.size, 0);
  });

  test("clear refuses active capture writers and removes every owned record and file", async () => {
    const { spool, metadata, files } = harness(7_500);
    const capture = descriptor("4", [
      { layer: 0, rows: 1, width: 1, expectedBytes: 4 },
    ]);
    const active = await spool.begin(capture);
    await assert.rejects(
      spool.clear(),
      errorCode(ActivationSpoolError, "STALE_WRITER"),
    );
    assert.equal(metadata.rows.has(capture.captureSha256), true);
    assert.equal(files.rows.size, 1);
    await active.rollback();

    const writer = await spool.begin(capture);
    await writer.appendRows(0, 0, new Float32Array([3]));
    await writer.sealLayer(0);
    await writer.commit();
    const orphanSha = "9".repeat(64);
    files.rows.set(activationSpoolFileName(orphanSha, 9), new Uint8Array(4));
    files.rows.set("unknown-entry", new Uint8Array(4));
    await spool.clear();
    assert.equal(metadata.rows.size, 0);
    assert.equal(files.rows.size, 0);
  });

  test("checks AbortSignal between bounded owned read chunks", async () => {
    const { spool } = harness(8_000);
    const capture = descriptor("2", [
      { layer: 0, rows: 4, width: 2, expectedBytes: 32 },
    ]);
    const writer = await spool.begin(capture);
    await writer.appendRows(0, 0, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
    await writer.sealLayer(0);
    const reader = await writer.commit();
    const controller = new AbortController();
    const iterator = reader.readChunks(0, {
      maxChunkBytes: 8,
      signal: controller.signal,
    })[Symbol.asyncIterator]();
    assert.deepEqual([...(await iterator.next()).value.values], [1, 2]);
    controller.abort(new Error("stop bounded transfer"));
    await assert.rejects(iterator.next(), /stop bounded transfer/);
  });

} catch (error) {
  await server.close();
  throw error;
}

function descriptor(seed, layers) {
  return {
    runtimeIdentitySha256: seed.repeat(64),
    contextBindingSha256: nextHex(seed).repeat(64),
    captureSha256: nextHex(nextHex(seed)).repeat(64),
    layers,
  };
}

function nextHex(value) {
  const index = "0123456789abcdef".indexOf(value);
  return "0123456789abcdef"[(index + 1) % 16];
}

function sharedPorts() {
  return {
    metadata: new MemoryMetadataPort(),
    files: new MemoryFilePort(),
    leases: new MemoryLeasePort(),
  };
}

function harness(now, ports = sharedPorts()) {
  let epoch = 0;
  return {
    ...ports,
    spool: new ActivationSpoolClass({
      ...ports,
      now: () => now,
      createEpoch: () => `epoch-${now}-${++epoch}`,
      runExclusive: serialRunner(),
    }),
  };
}

class MemoryMetadataPort {
  rows = new Map();

  async initialize() {}
  async read(key) { return clone(this.rows.get(key)); }
  async write(record) { this.rows.set(record.captureSha256, clone(record)); }
  async remove(key) { this.rows.delete(key); }
  async list() {
    return [...this.rows].map(([key, value]) => ({ key, value: clone(value) }));
  }
}

class MemoryFilePort {
  rows = new Map();
  maximumReadBytes = 0;

  async initialize() {}
  async list() { return [...this.rows.keys()].sort(); }
  async size(path) { return this.rows.get(path)?.byteLength ?? null; }
  async append(path, offset, bytes) {
    const current = this.rows.get(path) ?? new Uint8Array();
    assert.equal(current.byteLength, offset);
    const next = new Uint8Array(offset + bytes.byteLength);
    next.set(current);
    next.set(bytes.slice(), offset);
    this.rows.set(path, next);
  }
  async truncate(path, size) {
    const current = this.rows.get(path) ?? new Uint8Array();
    const next = new Uint8Array(size);
    next.set(current.subarray(0, Math.min(current.byteLength, size)));
    this.rows.set(path, next);
  }
  async read(path, offset, length) {
    this.maximumReadBytes = Math.max(this.maximumReadBytes, length);
    const current = this.rows.get(path);
    if (!current) throw new Error(`missing ${path}`);
    return current.subarray(offset, offset + length);
  }
  async remove(path) { this.rows.delete(path); }
  async clear() { this.rows.clear(); }
}

class MemoryLeasePort {
  held = new Set();

  async tryAcquire(captureSha256) {
    if (this.held.has(captureSha256)) return null;
    this.held.add(captureSha256);
    let released = false;
    return {
      release: async () => {
        if (released) return;
        released = true;
        this.held.delete(captureSha256);
      },
    };
  }

  crash(captureSha256) {
    this.held.delete(captureSha256);
  }
}

function serialRunner() {
  let tail = Promise.resolve();
  return async (operation) => {
    const predecessor = tail;
    let release;
    tail = new Promise((resolve) => { release = resolve; });
    await predecessor;
    try {
      return await operation();
    } finally {
      release();
    }
  };
}

function clone(value) {
  return value === undefined ? undefined : structuredClone(value);
}

function errorCode(ErrorClass, code) {
  return (error) => error instanceof ErrorClass && error.code === code;
}

function pickProgress(layer) {
  const { writtenRows, writtenBytes, checkpointRows, checkpointBytes, sealed } = layer;
  return { writtenRows, writtenBytes, checkpointRows, checkpointBytes, sealed };
}

async function collect(iterable) {
  const rows = [];
  for await (const chunk of iterable) rows.push(chunk);
  return rows;
}

try {
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
