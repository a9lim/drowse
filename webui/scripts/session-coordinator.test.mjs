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

function test(name, run) {
  tests.push({ name, run });
}

try {
  const {
    BrowserSessionCoordinator,
    SessionCoordinatorError,
    mutatesPersistedHostedSession,
  } = await server.ssrLoadModule("/src/hosted/runtime/sessionCoordinator.ts");
  const { BrowserSessionPersistence } = await server.ssrLoadModule(
    "/src/hosted/runtime/sessionPersistence.ts",
  );

  test("restores only an exactly compatible session and applies tree before config", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({
      store,
      now: () => 1_000,
      runExclusive: serialRunner(),
    });
    const savedSession = session({
      id: "saved-session",
      config: {
        temperature: 0.6,
        top_p: 0.91,
        top_k: null,
        max_tokens: 384,
        system_prompt: "Saved prompt",
        thinking: true,
      },
    });
    const savedTree = tree(savedSession.id, savedSession.model_id, 7);
    await persistence.save(sessionSave(savedSession, savedTree), identity().ownerEpoch);

    let current = session({
      id: "current-session",
      config: {
        temperature: 1,
        top_p: 1,
        top_k: 40,
        max_tokens: 128,
        system_prompt: "Current prompt",
        thinking: false,
      },
    });
    let restoredTree;
    const calls = [];
    const coordinator = new BrowserSessionCoordinator(persistencePort(persistence), async (request) => {
      calls.push(`${request.service}.${request.method}`);
      if (request.service === "sessions" && request.method === "get") {
        return structuredClone(current);
      }
      if (request.service === "tree" && request.method === "restore") {
        assert.deepEqual(request.args, [savedTree]);
        restoredTree = {
          ...structuredClone(savedTree),
          session_id: current.id,
          rev: savedTree.rev + 1,
        };
        return restoreAck(restoredTree);
      }
      if (request.service === "sessions" && request.method === "patch") {
        current = {
          ...current,
          config: { ...current.config, ...request.args[0] },
        };
        return structuredClone(current);
      }
      if (request.service === "tree" && request.method === "get") {
        return structuredClone(restoredTree);
      }
      throw new Error(`unexpected request ${request.service}.${request.method}`);
    });

    const startingConfig = structuredClone(current.config);
    const result = await coordinator.restore(identity());
    assert.deepEqual(coordinator.modelDefaults.config, startingConfig);
    assert.equal(result.status, "restored");
    assert.deepEqual(calls, [
      "sessions.get",
      "tree.restore",
      "sessions.patch",
      "tree.get",
    ]);
    assert.deepEqual(result.tree, restoredTree);
    assert.deepEqual(result.session.config, savedSession.config);
    assert.deepEqual(result.restoredConfigKeys.sort(), [
      "max_tokens",
      "system_prompt",
      "temperature",
      "thinking",
      "top_k",
      "top_p",
    ]);
  });

  test("preserves incompatible records without activating them", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({
      store,
      runExclusive: serialRunner(),
    });
    const savedSession = session();
    await persistence.save(sessionSave(savedSession, tree(
      savedSession.id,
      savedSession.model_id,
      3,
    )), identity().ownerEpoch);
    const calls = [];
    const coordinator = new BrowserSessionCoordinator(persistencePort(persistence), async (request) => {
      calls.push(`${request.service}.${request.method}`);
      return session({ model_id: "different/model" });
    });

    const wrongRuntime = await coordinator.restore({
      ...identity(),
      runtimeIdentitySha256: "b".repeat(64),
    });
    assert.deepEqual(wrongRuntime, {
      status: "incompatible",
      reason: "runtime_identity",
      preserved: true,
    });
    const wrongContext = await coordinator.restore({
      ...identity(),
      contextTokens: 4_096,
    });
    assert.deepEqual(wrongContext, {
      status: "incompatible",
      reason: "context_tokens",
      preserved: true,
    });
    const wrongModel = await coordinator.restore(identity());
    assert.deepEqual(wrongModel, {
      status: "incompatible",
      reason: "model_id",
      preserved: true,
    });
    const otherVariant = await coordinator.restore({
      ...identity(),
      modelVariantId: "other-variant",
    });
    assert.deepEqual(otherVariant, { status: "not_found" });
    assert.deepEqual(calls, ["sessions.get", "sessions.get"]);
    assert.notEqual(storedRecord(store, identity().modelVariantId), null);
  });

  test("restores only config values supported by the current session patch contract", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({
      store,
      runExclusive: serialRunner(),
    });
    const savedSession = session({
      config: {
        temperature: null,
        top_p: null,
        top_k: null,
        max_tokens: null,
        system_prompt: null,
        thinking: null,
      },
    });
    const savedTree = tree(savedSession.id, savedSession.model_id, 2);
    await persistence.save(sessionSave(savedSession, savedTree), identity().ownerEpoch);
    let current = session();
    let patch;
    let restoredTree;
    const coordinator = new BrowserSessionCoordinator(persistencePort(persistence), async (request) => {
      if (request.service === "sessions" && request.method === "get") return current;
      if (request.service === "tree" && request.method === "restore") {
        restoredTree = { ...structuredClone(savedTree), rev: savedTree.rev + 1 };
        return restoreAck(restoredTree);
      }
      if (request.service === "sessions" && request.method === "patch") {
        patch = structuredClone(request.args[0]);
        current = { ...current, config: { ...current.config, ...patch } };
        return current;
      }
      if (request.service === "tree" && request.method === "get") return restoredTree;
      throw new Error("unexpected request");
    });

    const result = await coordinator.restore(identity());
    assert.equal(result.status, "restored");
    assert.deepEqual(patch, { top_k: null });
    assert.deepEqual(result.restoredConfigKeys, ["top_k"]);
  });

  test("snapshots exact session metadata, config, and authoritative tree", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({
      store,
      now: () => 2_000,
      runExclusive: serialRunner(),
    });
    const currentSession = session({ created: 321 });
    const currentTree = tree(currentSession.id, currentSession.model_id, 11);
    const calls = [];
    const coordinator = new BrowserSessionCoordinator(persistencePort(persistence), async (request) => {
      calls.push(`${request.service}.${request.method}`);
      if (request.service === "sessions") return structuredClone(currentSession);
      if (request.service === "tree") return structuredClone(currentTree);
      throw new Error("unexpected request");
    });
    assert.deepEqual(await coordinator.restore(identity()), { status: "not_found" });

    const [first, second, third] = await Promise.all([
      coordinator.persist(identity()),
      coordinator.persist(identity()),
      coordinator.persist(identity()),
    ]);
    assert.deepEqual(calls, ["sessions.get", "sessions.get", "tree.get"]);
    assert.deepEqual(first, second);
    assert.deepEqual(second, third);
    const stored = storedRecord(store, identity().modelVariantId);
    assert.deepEqual(stored.metadata, {
      sessionId: currentSession.id,
      modelId: currentSession.model_id,
      runtimeIdentitySha256: identity().runtimeIdentitySha256,
      contextTokens: identity().contextTokens,
      createdAt: 321,
    });
    assert.deepEqual(stored.settings, currentSession.config);
    assert.deepEqual(stored.tree, currentTree);
  });

  test("serializes a follow-up snapshot so a later tree revision wins", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({
      store,
      runExclusive: serialRunner(),
    });
    const currentSession = session();
    let revision = 5;
    let treeReads = 0;
    let releaseFirst;
    const firstReadBlocked = new Promise((resolve) => {
      releaseFirst = resolve;
    });
    let firstReadStarted;
    const started = new Promise((resolve) => {
      firstReadStarted = resolve;
    });
    const coordinator = new BrowserSessionCoordinator(persistencePort(persistence), async (request) => {
      if (request.service === "sessions") return structuredClone(currentSession);
      if (request.service === "tree") {
        treeReads += 1;
        const captured = tree(currentSession.id, currentSession.model_id, revision);
        if (treeReads === 1) {
          firstReadStarted();
          await firstReadBlocked;
        }
        return captured;
      }
      throw new Error("unexpected request");
    });
    assert.deepEqual(await coordinator.restore(identity()), { status: "not_found" });

    const older = coordinator.persist(identity());
    await started;
    revision = 6;
    const newer = coordinator.persist(identity());
    releaseFirst();
    const [oldRecord, newRecord] = await Promise.all([older, newer]);
    assert.equal(oldRecord.tree.rev, 5);
    assert.equal(newRecord.tree.rev, 6);
    assert.equal(treeReads, 2);
    assert.equal(storedRecord(store, identity().modelVariantId).tree.rev, 6);
  });

  test("rejects invalid backend shapes and tensor-bearing trees without persisting", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({
      store,
      runExclusive: serialRunner(),
    });
    const invalidSessionCoordinator = new BrowserSessionCoordinator(
      persistencePort(persistence),
      async () => ({ id: "partial" }),
    );
    await assert.rejects(
      invalidSessionCoordinator.restore(identity()),
      (error) => error instanceof SessionCoordinatorError &&
        error.code === "INVALID_RUNTIME_RESPONSE",
    );
    await assert.rejects(
      invalidSessionCoordinator.persist(identity()),
      (error) => error instanceof SessionCoordinatorError &&
        error.code === "INVALID_RUNTIME_RESPONSE",
    );

    const currentSession = session();
    const tensorTree = tree(currentSession.id, currentSession.model_id, 1);
    tensorTree.nodes[1].tokens = new Float32Array([1, 2]);
    const tensorCoordinator = new BrowserSessionCoordinator(
      persistencePort(persistence),
      async (request) => request.service === "sessions" ? currentSession : tensorTree,
    );
    assert.deepEqual(await tensorCoordinator.restore(identity()), {
      status: "not_found",
    });
    await assert.rejects(tensorCoordinator.persist(identity()));
    assert.equal(storedRecord(store, identity().modelVariantId), null);
  });

  test("fails closed on a lying tree restore response before patching settings", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({
      store,
      runExclusive: serialRunner(),
    });
    const currentSession = session();
    const savedTree = tree(currentSession.id, currentSession.model_id, 4);
    await persistence.save(
      sessionSave(currentSession, savedTree),
      identity().ownerEpoch,
    );
    const calls = [];
    const coordinator = new BrowserSessionCoordinator(persistencePort(persistence), async (request) => {
      calls.push(`${request.service}.${request.method}`);
      if (request.service === "sessions") return currentSession;
      return { ...restoreAck(savedTree), rev: savedTree.rev };
    });

    await assert.rejects(
      coordinator.restore(identity()),
      (error) => error instanceof SessionCoordinatorError &&
        error.code === "INVALID_RUNTIME_RESPONSE",
    );
    assert.deepEqual(calls, ["sessions.get", "tree.restore"]);
    assert.notEqual(storedRecord(store, identity().modelVariantId), null);
  });

  test("preserves an unsupported persistence schema without touching the backend", async () => {
    let executions = 0;
    const port = {
      async claim() { return { status: "unsupported", schemaVersion: 99 }; },
      async save() { throw new Error("unexpected save"); },
      async delete() { throw new Error("unexpected delete"); },
      async clear() { throw new Error("unexpected clear"); },
      close() {},
    };
    const coordinator = new BrowserSessionCoordinator(port, async () => {
      executions += 1;
      throw new Error("unexpected backend request");
    });

    assert.deepEqual(await coordinator.restore(identity()), {
      status: "unsupported",
      preserved: true,
    });
    assert.equal(executions, 0);
    await assert.rejects(
      coordinator.persist(identity()),
      (error) => error instanceof SessionCoordinatorError &&
        error.code === "OWNERSHIP_NOT_CLAIMED",
    );
  });

  test("passes owner epochs and replace/reset intent through persistence CAS", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({
      store,
      runExclusive: serialRunner(),
    });
    const port = persistencePort(persistence);
    let currentSession = session();
    let currentTree = tree(currentSession.id, currentSession.model_id, 1);
    await persistence.save(
      sessionSave(currentSession, currentTree),
      identity().ownerEpoch,
    );
    const executor = async (request) => {
      if (request.service === "sessions" && request.method === "get") {
        return structuredClone(currentSession);
      }
      if (request.service === "sessions" && request.method === "patch") {
        currentSession = {
          ...currentSession,
          config: { ...currentSession.config, ...request.args[0] },
        };
        return structuredClone(currentSession);
      }
      if (request.service === "tree" && request.method === "restore") {
        const saved = request.args[0];
        currentTree = {
          ...structuredClone(saved),
          session_id: currentSession.id,
          rev: Math.max(currentTree.rev, saved.rev) + 1,
        };
        return restoreAck(currentTree);
      }
      if (request.service === "tree" && request.method === "get") {
        return structuredClone(currentTree);
      }
      throw new Error("unexpected request");
    };
    const first = new BrowserSessionCoordinator(port, executor);
    const second = new BrowserSessionCoordinator(port, executor);
    const firstIdentity = identity();
    const secondIdentity = { ...identity(), ownerEpoch: "owner-b" };
    assert.equal((await first.restore(firstIdentity)).status, "restored");
    assert.equal((await second.restore(secondIdentity)).status, "restored");

    await assert.rejects(
      first.persist(firstIdentity),
      (error) => error.code === "STALE_SESSION_OWNER",
    );
    await second.persist(secondIdentity);
    assert.deepEqual(port.claimModes, ["restore", "restore"]);
    assert.equal(
      storedRecord(store, identity().modelVariantId).metadata.contextTokens,
      identity().contextTokens,
    );
    const resetIdentity = {
      ...secondIdentity,
      runtimeIdentitySha256: "b".repeat(64),
      ownerEpoch: "owner-c",
    };
    assert.deepEqual(await second.restore(resetIdentity, "reset"), {
      status: "not_found",
    });
    assert.deepEqual(port.claimModes, ["restore", "restore", "reset"]);
    assert.equal(storedRecord(store, identity().modelVariantId), null);
  });

  test("classifies only mutating session and tree service methods", () => {
    for (const method of [
      "reset",
      "restore",
      "navigate",
      "edit",
      "branch",
      "delete",
      "star",
      "note",
      "transcriptLoad",
      "castPut",
      "castDelete",
    ]) {
      assert.equal(mutatesPersistedHostedSession({
        service: "tree",
        method,
        args: [],
      }), true, method);
    }
    assert.equal(mutatesPersistedHostedSession({
      service: "sessions",
      method: "patch",
      args: [],
    }), true);
    for (const [service, method] of [
      ["sessions", "get"],
      ["tree", "get"],
      ["tree", "active"],
      ["tree", "edgeLabel"],
      ["tree", "filter"],
      ["tree", "diff"],
      ["tree", "transcriptExport"],
      ["tree", "jointLogprobs"],
      ["tree", "replayCapabilities"],
      ["profiles", "extract"],
    ]) {
      assert.equal(mutatesPersistedHostedSession({ service, method, args: [] }), false);
    }
  });

  test("caps restored and persisted output settings for Apple mobile", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({
      store,
      runExclusive: serialRunner(),
    });
    const savedSession = session({ config: { max_tokens: 8_192 } });
    const savedTree = tree(savedSession.id, savedSession.model_id, 3);
    await persistence.save(sessionSave(savedSession, savedTree), identity().ownerEpoch);

    let current = session({ config: { max_tokens: 128 } });
    let restoredTree;
    let appliedPatch;
    const coordinator = new BrowserSessionCoordinator(
      persistencePort(persistence),
      async (request) => {
        if (request.service === "sessions" && request.method === "get") {
          return structuredClone(current);
        }
        if (request.service === "tree" && request.method === "restore") {
          restoredTree = {
            ...structuredClone(savedTree),
            session_id: current.id,
            rev: savedTree.rev + 1,
          };
          return restoreAck(restoredTree);
        }
        if (request.service === "sessions" && request.method === "patch") {
          appliedPatch = structuredClone(request.args[0]);
          current = {
            ...current,
            config: { ...current.config, ...appliedPatch },
          };
          return structuredClone(current);
        }
        if (request.service === "tree" && request.method === "get") {
          return structuredClone(restoredTree);
        }
        throw new Error(`unexpected request ${request.service}.${request.method}`);
      },
      { maxOutputTokens: 256 },
    );

    const restored = await coordinator.restore(identity());
    assert.equal(restored.status, "restored");
    assert.equal(appliedPatch.max_tokens, 256);
    assert.equal(restored.session.config.max_tokens, 256);
    await coordinator.persist(identity());
    assert.equal(
      storedRecord(store, identity().modelVariantId).settings.max_tokens,
      256,
    );
  });

  test("orders delete and clear after queued snapshots and closes once", async () => {
    const store = memoryStore();
    const persistence = new BrowserSessionPersistence({
      store,
      runExclusive: serialRunner(),
    });
    const currentSession = session();
    const currentTree = tree(currentSession.id, currentSession.model_id, 1);
    const coordinator = new BrowserSessionCoordinator(persistencePort(persistence), async (request) => (
      request.service === "sessions" ? currentSession : currentTree
    ));
    assert.deepEqual(await coordinator.restore(identity()), { status: "not_found" });

    const saving = coordinator.persist(identity());
    const deleting = coordinator.delete(identity());
    await Promise.all([saving, deleting]);
    assert.equal(storedRecord(store, identity().modelVariantId), null);

    assert.deepEqual(await coordinator.restore(identity()), { status: "not_found" });
    await coordinator.persist(identity());
    await coordinator.clear(identity().ownerEpoch);
    assert.equal(storedRecord(store, identity().modelVariantId), null);
    await Promise.all([coordinator.close(), coordinator.close()]);
    assert.equal(store.closes, 1);
    await assert.rejects(
      coordinator.persist(identity()),
      (error) => error instanceof SessionCoordinatorError &&
        error.code === "COORDINATOR_CLOSED",
    );
  });

  for (const { name, run } of tests) {
    await run();
    console.log(`ok - ${name}`);
  }
  console.log(`${tests.length} session coordinator tests passed`);
} finally {
  await server.close();
}

function identity() {
  return {
    modelVariantId: "smollm2-q4f16",
    runtimeIdentitySha256: "a".repeat(64),
    contextTokens: 2_048,
    ownerEpoch: "owner-a",
  };
}

function session(overrides = {}) {
  const base = {
    id: "default",
    model_id: "HuggingFaceTB/SmolLM2-360M-Instruct",
    device: "webgpu",
    dtype: "float16",
    created: 100,
    config: {
      temperature: 0.7,
      top_p: 0.9,
      top_k: 40,
      max_tokens: 256,
      system_prompt: "Be concise",
      thinking: false,
    },
    profiles: [],
    probes: [],
    history_length: 1,
    supports_thinking: false,
    thinking_is_optional: false,
    is_base_model: false,
    jlens_fitted: false,
    instruments: [],
    default_steering: null,
    role_substitution_supported: true,
    user_role_supported: true,
    default_assistant_role: "assistant",
    default_user_role: "user",
    scene_mode: false,
    thinking_input_supported: false,
    strips_history_thinking: false,
  };
  return {
    ...base,
    ...overrides,
    config: { ...base.config, ...(overrides.config ?? {}) },
  };
}

function sessionSave(currentSession, currentTree) {
  return {
    modelVariantId: identity().modelVariantId,
    metadata: {
      sessionId: currentSession.id,
      modelId: currentSession.model_id,
      runtimeIdentitySha256: identity().runtimeIdentitySha256,
      contextTokens: identity().contextTokens,
      createdAt: currentSession.created,
    },
    settings: structuredClone(currentSession.config),
    tree: structuredClone(currentTree),
  };
}

function tree(sessionId, modelId, revision) {
  const root = node("root", null, "system", "");
  const user = node("user-1", "root", "user", "Hello");
  return {
    tree_format: 2,
    drowse_version: "fixture",
    model_id: modelId,
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

function restoreAck(currentTree) {
  return {
    rev: currentTree.rev,
    root_id: currentTree.root_id,
    active_node_id: currentTree.active_node_id,
    nodes: currentTree.nodes.length,
  };
}

function memoryStore(initial = new Map()) {
  const rows = initial;
  return {
    rows,
    closes: 0,
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
    close() {
      this.closes += 1;
    },
  };
}

function persistencePort(persistence) {
  return {
    claimModes: [],
    async claim(binding, mode = "restore", compatibleRuntimeIdentitySha256s = []) {
      this.claimModes.push(mode);
      return persistence.claim(binding, mode, compatibleRuntimeIdentitySha256s);
    },
    async save(input, ownerEpoch) {
      return persistence.save(input, ownerEpoch);
    },
    async delete(modelVariantId) {
      return persistence.delete(modelVariantId);
    },
    async clear() {
      await persistence.clear();
    },
    close() {
      persistence.close();
    },
  };
}

function storedRecord(store, modelVariantId) {
  const value = store.rows.get(modelVariantId);
  return value === undefined ? null : structuredClone(value);
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
