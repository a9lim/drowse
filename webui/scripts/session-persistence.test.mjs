import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

const tests = [];

function test(name, run) {
  tests.push({ name, run });
}

try {
  const {
    BrowserSessionPersistence,
    BrowserSessionStateStore,
    SessionPersistenceError,
    validatePersistedHostedSession,
  } = await server.ssrLoadModule("/src/hosted/runtime/sessionPersistence.ts");

  test("round-trips one authoritative session per model variant", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({
      store,
      now: () => 1_000,
      runExclusive: serialRunner(),
    });
    const fastest = sessionInput("fastest", "session-a", 2);
    const balanced = sessionInput("balanced", "session-b", 4);

    const saved = await persistence.save(fastest, "owner-a");
    await persistence.save(balanced, "owner-a");
    saved.tree.rev = 99;
    saved.settings.temperature = 99;

    const restored = await persistence.claim(binding("fastest", "owner-a"));
    assert.equal(restored.status, "compatible");
    assert.equal(restored.record.tree.rev, 2);
    assert.equal(restored.record.settings.temperature, 0.7);
    const balancedRestored = await persistence.claim(binding("balanced", "owner-a"));
    assert.equal(balancedRestored.status, "compatible");
    assert.equal(balancedRestored.record.metadata.sessionId, "session-b");
    assert.equal(await persistence.delete("fastest"), true);
    assert.equal(await persistence.delete("fastest"), false);
    assert.equal((await persistence.claim(binding("fastest", "owner-a"))).status, "missing");
    assert.equal((await persistence.claim(binding("balanced", "owner-a"))).status, "compatible");
    await persistence.clear();
    assert.equal((await persistence.claim(binding("balanced", "owner-a"))).status, "missing");
  });

  test("serializes writes and rejects stale revisions and relinquished owners", async () => {
    const store = memoryStore();
    const runner = trackedSerialRunner();
    const persistence = new BrowserSessionPersistence({
      store,
      now: () => 2_000,
      runExclusive: runner.run,
    });
    await persistence.save(sessionInput("fastest", "session-a", 5), "owner-a");

    const newer = persistence.save(sessionInput("fastest", "session-a", 6), "owner-a");
    const stale = persistence.save(sessionInput("fastest", "session-a", 4), "owner-a");
    await newer;
    await assert.rejects(
      stale,
      (error) => error instanceof SessionPersistenceError &&
        error.code === "STALE_TREE_REVISION",
    );

    assert.equal(runner.maximumActive(), 1);
    const claimed = await persistence.claim(binding("fastest", "owner-b"));
    assert.equal(claimed.status, "compatible");
    assert.equal(claimed.record.tree.rev, 6);

    await assert.rejects(
      persistence.save(sessionInput("fastest", "session-a", 7), "owner-a"),
      (error) => error instanceof SessionPersistenceError &&
        error.code === "STALE_SESSION_OWNER",
    );
    await persistence.save(sessionInput("fastest", "session-a", 7), "owner-b");
    assert.equal(
      (await persistence.claim(binding("fastest", "owner-b"))).record.tree.rev,
      7,
    );
  });

  test("uses the named Web Lock for default cross-tab serialization", async () => {
    const requests = [];
    const previous = Object.getOwnPropertyDescriptor(globalThis, "navigator");
    Object.defineProperty(globalThis, "navigator", {
      configurable: true,
      value: {
        locks: {
          async request(name, options, operation) {
            requests.push({ name, options });
            return operation();
          },
        },
      },
    });
    try {
      const persistence = new BrowserSessionPersistence({
        store: memoryStore(),
        now: () => 2_500,
      });
      await persistence.save(sessionInput("fastest", "session-a", 1), "owner-a");
      await persistence.claim(binding("fastest", "owner-a"));
      assert.deepEqual(requests, [
        { name: "drowse-hosted-session-state-v1", options: { mode: "exclusive" } },
        { name: "drowse-hosted-session-state-v1", options: { mode: "exclusive" } },
      ]);
    } finally {
      if (previous) Object.defineProperty(globalThis, "navigator", previous);
      else delete globalThis.navigator;
    }
  });

  test("preserves unsupported rows but removes corrupt current-schema rows", async () => {
    const unsupported = { schemaVersion: 99, future: true };
    const store = memoryStore(new Map([["fastest", unsupported]]));
    const recovered = [];
    const persistence = new BrowserSessionPersistence({
      store,
      runExclusive: serialRunner(),
      onCorrupt: (modelVariantId, error) => recovered.push({ modelVariantId, error }),
    });

    assert.deepEqual(await persistence.claim(binding("fastest", "owner-a")), {
      status: "unsupported",
      schemaVersion: 99,
    });
    assert.deepEqual(store.rows.get("fastest"), unsupported);
    assert.equal(recovered.length, 0);

    store.rows.set("fastest", { schemaVersion: 2 });
    assert.equal((await persistence.claim(binding("fastest", "owner-a"))).status, "missing");
    assert.equal(store.rows.has("fastest"), false);
    assert.equal(recovered.length, 1);
    assert.equal(recovered[0].modelVariantId, "fastest");
    assert.match(recovered[0].error.message, /fields are invalid/);

    await persistence.save(sessionInput("fastest", "session-a", 1), "owner-a");
    assert.equal(
      (await persistence.claim(binding("fastest", "owner-a"))).record.tree.rev,
      1,
    );
  });

  test("explicit new-chat reset clears compatible autosaves without touching other models", async () => {
    for (const migrate of [false, true]) {
      const store = memoryStore();
      const persistence = new BrowserSessionPersistence({ store, runExclusive: serialRunner() });
      await persistence.save(sessionInput("fastest", "session-a", 3), "owner-a");
      await persistence.save(sessionInput("balanced", "session-b", 4), "owner-a");
      const target = binding("fastest", "owner-b");
      if (migrate) target.runtimeIdentitySha256 = "b".repeat(64);
      const result = await persistence.claim(target, "reset", migrate ? ["a".repeat(64)] : []);
      assert.equal(result.status, "missing", "reset must not restore a compatible session");
      assert.equal(store.rows.has("fastest"), false);
      assert.equal(store.rows.get("balanced").tree.rev, 4);
    }
  });

  test("keeps incompatible runtime data until an explicit reset", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({
      store,
      runExclusive: serialRunner(),
    });
    await persistence.save(sessionInput("fastest", "session-a", 3), "owner-a");

    const runtimeMismatch = await persistence.claim({
      ...binding("fastest", "owner-b"),
      runtimeIdentitySha256: "b".repeat(64),
    });
    assert.equal(runtimeMismatch.status, "incompatible");
    assert.equal(runtimeMismatch.reason, "runtime_identity");
    assert.equal(store.rows.get("fastest").tree.rev, 3);

    const reset = await persistence.claim({
      ...binding("fastest", "owner-b"),
      runtimeIdentitySha256: "b".repeat(64),
    }, "reset");
    assert.equal(reset.status, "missing");
    assert.equal(store.rows.has("fastest"), false);
  });

  test("migrates an explicitly compatible runtime identity without losing the conversation", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({
      store,
      runExclusive: serialRunner(),
    });
    await persistence.save(sessionInput("fastest", "session-a", 3), "owner-a");
    const currentIdentity = "b".repeat(64);
    const claimed = await persistence.claim({
      ...binding("fastest", "owner-b"),
      runtimeIdentitySha256: currentIdentity,
    }, "restore", ["a".repeat(64)]);

    assert.equal(claimed.status, "compatible");
    assert.equal(claimed.record.tree.rev, 3);
    assert.equal(claimed.record.ownerEpoch, "owner-b");
    assert.equal(claimed.record.metadata.runtimeIdentitySha256, currentIdentity);
    assert.equal(store.rows.get("fastest").metadata.runtimeIdentitySha256, currentIdentity);
  });

  test("rejects malformed topology, non-JSON settings, and metadata mismatches", () => {
    const valid = persistedRecord("fastest", "session-a", 1);
    validatePersistedHostedSession(valid);

    const disconnected = structuredClone(valid);
    disconnected.tree.children_of.root = [];
    assert.throws(
      () => validatePersistedHostedSession(disconnected),
      (error) => error.code === "INVALID_SESSION_STATE" &&
        /absent from its parent's children row/.test(error.message),
    );

    const tensorSetting = structuredClone(valid);
    tensorSetting.settings.temperature = new Float32Array([1, 2]);
    assert.throws(
      () => validatePersistedHostedSession(tensorSetting),
      (error) => error.code === "INVALID_SESSION_STATE" &&
        /temperature is invalid/.test(error.message),
    );

    const mismatched = structuredClone(valid);
    mismatched.tree.model_id = "different/model";
    assert.throws(
      () => validatePersistedHostedSession(mismatched),
      (error) => error.code === "INVALID_SESSION_STATE" &&
        /does not match session metadata/.test(error.message),
    );

    const wrongFormat = structuredClone(valid);
    wrongFormat.tree.tree_format = 3;
    assert.throws(
      () => validatePersistedHostedSession(wrongFormat),
      (error) => error.code === "INVALID_SESSION_STATE" &&
        /format is unsupported/.test(error.message),
    );

    const wrongRoot = structuredClone(valid);
    wrongRoot.tree.nodes[0].role = "user";
    assert.throws(
      () => validatePersistedHostedSession(wrongRoot),
      (error) => error.code === "INVALID_SESSION_STATE" &&
        /system role/.test(error.message),
    );
  });

  test("accepts open current token rows while validating their known fields", () => {
    const valid = persistedRecord("fastest", "session-a", 1);
    valid.tree.nodes[1].tokens = [{
      token_id: 7,
      text: "Hello",
      logprob: -0.25,
      producer_extension: { finite: 1 },
    }];
    validatePersistedHostedSession(valid);

    const invalid = structuredClone(valid);
    invalid.tree.nodes[1].tokens[0].producer_extension.finite = Infinity;
    assert.throws(
      () => validatePersistedHostedSession(invalid),
      (error) => error.code === "INVALID_SESSION_STATE" &&
        /non-finite/.test(error.message),
    );
  });

  test("reopens the IndexedDB state store after versionchange", async () => {
    const fake = fakeIndexedDb();
    const previous = Object.getOwnPropertyDescriptor(globalThis, "indexedDB");
    Object.defineProperty(globalThis, "indexedDB", {
      configurable: true,
      value: fake,
    });
    try {
      const store = new BrowserSessionStateStore();
      const record = persistedRecord("fastest", "session-a", 1);
      await Promise.all([store.initialize(), store.initialize()]);
      assert.equal(fake.openCalls, 1);
      await store.write(record);
      assert.equal((await store.read("fastest")).tree.rev, 1);

      fake.handles[0].onversionchange();
      assert.equal(fake.handles[0].closed, true);
      assert.equal((await store.read("fastest")).tree.rev, 1);
      assert.equal(fake.openCalls, 2);
      assert.equal(fake.handles[1].closed, false);
    } finally {
      if (previous) Object.defineProperty(globalThis, "indexedDB", previous);
      else delete globalThis.indexedDB;
    }
  });

  for (const { name, run } of tests) {
    await run();
    console.log(`ok - ${name}`);
  }
  console.log(`${tests.length} session persistence tests passed`);
} finally {
  await server.close();
}

function sessionInput(modelVariantId, sessionId, revision) {
  return {
    modelVariantId,
    metadata: {
      sessionId,
      modelId: "HuggingFaceTB/SmolLM2-360M-Instruct",
      runtimeIdentitySha256: "a".repeat(64),
      contextTokens: 2_048,
      createdAt: 100,
    },
    settings: {
      temperature: 0.7,
      top_p: 0.9,
      top_k: 40,
      max_tokens: 256,
      system_prompt: "You are helpful.",
      thinking: false,
    },
    tree: tree(sessionId, revision),
  };
}

function persistedRecord(modelVariantId, sessionId, revision) {
  return {
    ...sessionInput(modelVariantId, sessionId, revision),
    schemaVersion: 2,
    ownerEpoch: "owner-a",
    updatedAt: 1_000,
  };
}

function binding(modelVariantId, ownerEpoch) {
  return {
    modelVariantId,
    runtimeIdentitySha256: "a".repeat(64),
    contextTokens: 2_048,
    ownerEpoch,
  };
}

function tree(sessionId, revision) {
  const root = node("root", null, "system", "");
  const user = node("user-1", "root", "user", "Hello");
  return {
    tree_format: 2,
    drowse_version: "fixture",
    model_id: "HuggingFaceTB/SmolLM2-360M-Instruct",
    session_id: sessionId,
    name: "Local conversation",
    rev: revision,
    root_id: root.id,
    active_node_id: user.id,
    nodes: [root, user],
    children_of: { root: [user.id], [user.id]: [] },
    cast: {},
  };
}

function node(id, parentId, role, text) {
  return {
    id,
    parent_id: parentId,
    role,
    text,
    role_label: null,
    thinking_text: null,
    aggregate_readings: {},
    applied_steering: null,
    finish_reason: null,
    starred: false,
    notes: "",
    created_at: 100,
    edited_at: null,
    edit_count: 0,
    mean_logprob: null,
    mean_surprise: null,
    recipe: null,
    tokens: null,
    thinking_tokens: null,
    raw_token_ids: null,
  };
}

function memoryStore(initial = new Map()) {
  const rows = initial;
  return {
    rows,
    async initialize() {},
    async read(key) {
      const value = rows.get(key);
      return value === undefined ? undefined : structuredClone(value);
    },
    async write(record) {
      rows.set(record.modelVariantId, structuredClone(record));
    },
    async delete(key) {
      return rows.delete(key);
    },
    async clear() {
      rows.clear();
    },
  };
}

function serialRunner() {
  let tail = Promise.resolve();
  return async (operation) => {
    const predecessor = tail;
    let release;
    tail = new Promise((resolve) => {
      release = resolve;
    });
    await predecessor;
    try {
      return await operation();
    } finally {
      release();
    }
  };
}

function trackedSerialRunner() {
  let active = 0;
  let maximum = 0;
  const serial = serialRunner();
  return {
    run: (operation) => serial(async () => {
      active += 1;
      maximum = Math.max(maximum, active);
      await Promise.resolve();
      try {
        return await operation();
      } finally {
        active -= 1;
      }
    }),
    maximumActive: () => maximum,
  };
}

function fakeIndexedDb() {
  const rows = new Map();
  let storeCreated = false;
  return {
    openCalls: 0,
    handles: [],
    open() {
      this.openCalls += 1;
      const handle = fakeDatabaseHandle(rows, () => storeCreated, () => {
        storeCreated = true;
      });
      this.handles.push(handle);
      const request = { result: handle, error: null };
      queueMicrotask(() => {
        if (!storeCreated) request.onupgradeneeded?.();
        request.onsuccess?.();
      });
      return request;
    },
  };
}

function fakeDatabaseHandle(rows, hasStore, createStore) {
  const handle = {
    closed: false,
    onversionchange: null,
    onclose: null,
    objectStoreNames: { contains: () => hasStore() },
    createObjectStore() {
      createStore();
    },
    close() {
      this.closed = true;
    },
    transaction() {
      if (this.closed) throw new DOMException("Database is closed", "InvalidStateError");
      const transaction = {
        error: null,
        objectStore() {
          return {
            get(key) {
              const request = { result: undefined, error: null };
              queueMicrotask(() => {
                const value = rows.get(key);
                request.result = value === undefined ? undefined : structuredClone(value);
                request.onsuccess?.();
                queueMicrotask(() => transaction.oncomplete?.());
              });
              return request;
            },
            put(record) {
              rows.set(record.modelVariantId, structuredClone(record));
              queueMicrotask(() => transaction.oncomplete?.());
            },
            delete(key) {
              rows.delete(key);
              queueMicrotask(() => transaction.oncomplete?.());
            },
            clear() {
              rows.clear();
              queueMicrotask(() => transaction.oncomplete?.());
            },
          };
        },
      };
      return transaction;
    },
  };
  return handle;
}
