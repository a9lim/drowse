import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import {
  FIXTURE_ABI,
  FIXTURE_SIGNATURE,
  catalogBytes,
  catalogRuntimeLockModels,
} from "./catalog-fixture.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const {
    BrowserCatalogStateStore,
    CatalogFetchError,
    VerifiedCatalogRepository,
  } = await server.ssrLoadModule("/src/hosted/runtime/catalogRepository.ts");
  const { CatalogValidationError } = await server.ssrLoadModule(
    "/src/lib/runtime/catalog.ts",
  );
  const { BrowserContentStore } = await server.ssrLoadModule(
    "/src/hosted/runtime/contentStore.ts",
  );
  const { BrowserDrowseArchiveRepository } = await server.ssrLoadModule(
    "/src/hosted/artifacts/repository.ts",
  );

  const tests = [];
  const test = (name, run) => tests.push({ name, run });
  const now = Date.parse("2026-06-01T00:00:00.000Z");

  test("catalog repository accepts an exact same-sequence signed replay", async () => {
    const bytes = catalogBytes();
    const store = memoryStore(cachedState(bytes, 7));
    const repository = new VerifiedCatalogRepository({
      config: config(), store, fetch: fixtureFetch(bytes),
      verifier: acceptingVerifier(), now: () => now,
    });

    const catalog = await repository.get();
    assert.equal(catalog.document.sequence, 7);
    assert.deepEqual(store.state.exactBytes, bytes);
    assert.deepEqual(store.state.signature, FIXTURE_SIGNATURE);
  });

  test("catalog upgrade preserves a retired-key sequence floor without clearing installed state", async () => {
    for (const sequence of [3, 7]) {
      const store = memoryStore(cachedState(catalogBytes({ sequence: 6 }), 6));
      store.state.signature.keyId = "retired-brand-key";
      store.clear = async () => { throw new Error("upgrade must not clear catalog state"); };
      const verifier = acceptingVerifier();
      verifier.hasKey = (keyId) => keyId !== "retired-brand-key";
      const repository = new VerifiedCatalogRepository({
        config: config(), store, fetch: fixtureFetch(catalogBytes({ sequence })),
        verifier, now: () => now,
      });
      if (sequence === 3) {
        await assert.rejects(repository.get(), (error) => error.code === "CATALOG_ROLLBACK");
        assert.equal(store.state.highestAcceptedSequence, 6);
      } else {
        assert.equal((await repository.get()).document.sequence, 7);
        assert.equal(store.state.highestAcceptedSequence, 7);
      }
    }
  });

  test("catalog repository rejects same-sequence signed-byte equivocation", async () => {
    const acceptedBytes = catalogBytes();
    const store = memoryStore(cachedState(acceptedBytes, 7));
    const alternateBytes = catalogBytes({ expiresAt: "2027-02-01T00:00:00.000Z" });
    const alternateSignature = {
      ...FIXTURE_SIGNATURE,
      signature: Buffer.alloc(64, 1).toString("base64"),
    };
    const repository = new VerifiedCatalogRepository({
      config: config(), store,
      fetch: fixtureFetch(alternateBytes, alternateSignature),
      verifier: acceptingVerifier(), now: () => now,
    });

    await assert.rejects(
      repository.get(),
      (error) => error.code === "CATALOG_EQUIVOCATION",
    );
    assert.deepEqual(store.state.exactBytes, acceptedBytes);
    assert.deepEqual(store.state.signature, FIXTURE_SIGNATURE);
    assert.equal(store.state.highestAcceptedSequence, 7);
  });

  test("catalog repository rejects a new signature or key at the accepted sequence", async () => {
    const bytes = catalogBytes();
    for (const signature of [
      { ...FIXTURE_SIGNATURE, signature: Buffer.alloc(64, 2).toString("base64") },
      { ...FIXTURE_SIGNATURE, keyId: "next-key", signature: Buffer.alloc(64, 3).toString("base64") },
    ]) {
      const store = memoryStore(cachedState(bytes, 7));
      const repository = new VerifiedCatalogRepository({
        config: config(), store, fetch: fixtureFetch(bytes, signature),
        verifier: acceptingVerifier(), now: () => now,
      });
      await assert.rejects(
        repository.get(),
        (error) => error.code === "CATALOG_EQUIVOCATION",
      );
      assert.deepEqual(store.state.signature, FIXTURE_SIGNATURE);
    }
  });

  test("catalog repository accepts a higher sequence signed by the next key", async () => {
    const store = memoryStore(cachedState(catalogBytes(), 7));
    const nextSignature = {
      ...FIXTURE_SIGNATURE,
      keyId: "next-key",
      signature: Buffer.alloc(64, 4).toString("base64"),
    };
    const repository = new VerifiedCatalogRepository({
      config: config(), store,
      fetch: fixtureFetch(catalogBytes({ sequence: 8 }), nextSignature),
      verifier: acceptingVerifier(), now: () => now,
    });

    assert.equal((await repository.get()).document.sequence, 8);
    assert.equal(store.state.highestAcceptedSequence, 8);
    assert.deepEqual(store.state.signature, nextSignature);
  });

  test("catalog repository verifies exact network bytes before committing", async () => {
    const store = memoryStore();
    const fetch = fixtureFetch(catalogBytes());
    const repository = new VerifiedCatalogRepository({
      config: config(), store, fetch, verifier: acceptingVerifier(), now: () => now,
    });
    const catalog = await repository.get();
    assert.equal(catalog.document.sequence, 7);
    assert.equal(catalog.allowDownloads, true);
    assert.equal(store.state.highestAcceptedSequence, 7);
    assert.deepEqual(store.state.exactBytes, catalog.exactBytes);
    assert.equal(fetch.calls.length, 2);
    assert.equal(fetch.calls.every((call) => call.init.cache === "no-store"), true);
  });

  test("catalog repository retries initialization after a transient failure", async () => {
    const store = memoryStore();
    let initializeCalls = 0;
    let clearCalls = 0;
    store.initialize = async () => {
      initializeCalls += 1;
      if (initializeCalls === 1) throw new Error("transient initialization failure");
    };
    store.clear = async () => {
      clearCalls += 1;
      store.state = emptyState();
    };
    const repository = new VerifiedCatalogRepository({
      config: config(),
      store,
      fetch: fixtureFetch(catalogBytes()),
      verifier: acceptingVerifier(),
      now: () => now,
    });

    await assert.rejects(repository.get(), /transient initialization failure/);
    await repository.clear();
    assert.equal((await repository.get()).document.sequence, 7);
    assert.equal(initializeCalls, 2);
    assert.equal(clearCalls, 1);
  });

  test("catalog repository rejects rollback without replacing last-known-good", async () => {
    const store = memoryStore(cachedState(catalogBytes(), 7));
    const repository = new VerifiedCatalogRepository({
      config: config(),
      store,
      fetch: fixtureFetch(catalogBytes({ sequence: 6 })),
      verifier: acceptingVerifier(),
      now: () => now,
    });
    await assert.rejects(repository.get(), (error) => error.code === "CATALOG_ROLLBACK");
    assert.equal(store.state.highestAcceptedSequence, 7);
    assert.equal(JSON.parse(new TextDecoder().decode(store.state.exactBytes)).sequence, 7);
  });

  test("offline expired catalog authorizes installed artifacts but not downloads", async () => {
    const expired = catalogBytes({ expiresAt: "2026-05-01T00:00:00.000Z" });
    const repository = new VerifiedCatalogRepository({
      config: config(),
      store: memoryStore(cachedState(expired, 7)),
      verifier: acceptingVerifier(),
      online: () => false,
      now: () => now,
    });
    const catalog = await repository.get();
    assert.equal(catalog.stale, true);
    assert.equal(catalog.allowDownloads, false);
  });

  test("an explicit offline request never fetches when worker network state is stale", async () => {
    const now = Date.parse("2026-01-01T00:00:00Z");
    const store = memoryStore(cachedState(catalogBytes(), 7));
    let fetched = false;
    const repository = new VerifiedCatalogRepository({
      config: config(),
      store,
      fetch: async () => {
        fetched = true;
        throw new Error("must not fetch");
      },
      online: () => true,
      verifier: acceptingVerifier(),
      now: () => now,
    });

    const catalog = await repository.get({ offline: true });

    assert.equal(fetched, false);
    assert.equal(catalog.allowDownloads, false);
    assert.equal(catalog.stale, false);
  });

  test("a cache-preferred startup verifies current signed bytes without fetching", async () => {
    const store = memoryStore(cachedState(catalogBytes(), 7));
    let fetched = false;
    const repository = new VerifiedCatalogRepository({
      config: config(),
      store,
      fetch: async () => {
        fetched = true;
        throw new Error("must not fetch");
      },
      online: () => true,
      verifier: acceptingVerifier(),
      now: () => now,
    });

    const catalog = await repository.get({ preferCached: true });

    assert.equal(fetched, false);
    assert.equal(catalog.document.sequence, 7);
    assert.equal(catalog.allowDownloads, true);
  });

  test("transient online failure falls back only to a current verified cache", async () => {
    const unavailable = async () => {
      throw new TypeError("network unavailable");
    };
    const repository = new VerifiedCatalogRepository({
      config: config(),
      store: memoryStore(cachedState(catalogBytes(), 7)),
      fetch: unavailable,
      verifier: acceptingVerifier(),
      now: () => now,
    });
    const cached = await repository.get();
    assert.equal(cached.document.sequence, 7);
    assert.equal(cached.allowDownloads, false);
    await assert.rejects(
      repository.get({ refresh: true }),
      (error) => error instanceof CatalogFetchError && error.code === "CATALOG_FETCH_UNAVAILABLE",
    );
  });

  test("online fetch failure opens an expired last-known-good without authorizing downloads", async () => {
    const expired = catalogBytes({ expiresAt: "2026-05-01T00:00:00.000Z" });
    const repository = new VerifiedCatalogRepository({
      config: config(),
      store: memoryStore(cachedState(expired, 7)),
      fetch: async () => { throw new TypeError("network unavailable"); },
      verifier: acceptingVerifier(),
      now: () => now,
    });

    const cached = await repository.get();
    assert.equal(cached.document.sequence, 7);
    assert.equal(cached.stale, true);
    assert.equal(cached.allowDownloads, false);
  });

  test("online startup repairs malformed cached metadata from the signed source", async () => {
    const store = memoryStore();
    let malformed = true;
    let clearCalls = 0;
    store.read = async () => {
      if (malformed) {
        throw new CatalogValidationError(
          "CATALOG_CACHE_INVALID",
          "The locally cached catalog metadata is invalid",
        );
      }
      return cloneState(store.state);
    };
    store.clear = async () => {
      clearCalls += 1;
      malformed = false;
      store.state = emptyState();
    };
    const repository = new VerifiedCatalogRepository({
      config: config(),
      store,
      fetch: fixtureFetch(catalogBytes()),
      verifier: acceptingVerifier(),
      now: () => now,
    });

    const catalog = await repository.get({ preferCached: true });
    assert.equal(catalog.document.sequence, 7);
    assert.equal(catalog.allowDownloads, true);
    assert.equal(clearCalls, 1);
    assert.equal(store.state.highestAcceptedSequence, 7);
  });

  test("online startup repairs malformed cached bytes at the accepted sequence", async () => {
    const malformed = new TextEncoder().encode("{not-json");
    const store = memoryStore(cachedState(malformed, 7));
    const repository = new VerifiedCatalogRepository({
      config: config(),
      store,
      fetch: fixtureFetch(catalogBytes()),
      verifier: acceptingVerifier(),
      now: () => now,
    });

    const catalog = await repository.get({ preferCached: true });
    assert.equal(catalog.document.sequence, 7);
    assert.equal(catalog.allowDownloads, true);
    assert.deepEqual(store.state.exactBytes, catalogBytes());
  });

  test("interrupted catalog response bodies use the verified cache", async () => {
    const interruptedFetch = async (url) => {
      const body = url.endsWith("catalog.json")
        ? catalogBytes()
        : new TextEncoder().encode(JSON.stringify(FIXTURE_SIGNATURE));
      if (!url.endsWith("catalog.json")) return response(url, body);
      return {
        ok: true,
        status: 200,
        url,
        body: new ReadableStream({
          start(controller) {
            controller.enqueue(body.slice(0, 8));
            queueMicrotask(() => controller.error(new TypeError("connection reset")));
          },
        }),
        arrayBuffer: async () => body.slice().buffer,
      };
    };
    const repository = new VerifiedCatalogRepository({
      config: config(),
      store: memoryStore(cachedState(catalogBytes(), 7)),
      fetch: interruptedFetch,
      verifier: acceptingVerifier(),
      now: () => now,
    });
    assert.equal((await repository.get()).document.sequence, 7);
    await assert.rejects(
      repository.get({ refresh: true }),
      (error) =>
        error instanceof CatalogFetchError && error.code === "CATALOG_FETCH_UNAVAILABLE",
    );
  });

  test("invalid online signature never falls back to cached content", async () => {
    const invalidSignature = { ...FIXTURE_SIGNATURE, keyId: "unknown-key" };
    const repository = new VerifiedCatalogRepository({
      config: config(),
      store: memoryStore(cachedState(catalogBytes(), 7)),
      fetch: fixtureFetch(catalogBytes({ sequence: 8 }), invalidSignature),
      verifier: acceptingVerifier(),
      now: () => now,
    });
    await assert.rejects(repository.get(), (error) => error.code === "CATALOG_KEY_UNKNOWN");
  });

  test("an empty signing key ring fails before network access", async () => {
    let fetched = false;
    const repository = new VerifiedCatalogRepository({
      config: { ...config(), status: "feasibility-required", publicKeys: new Map() },
      store: memoryStore(),
      fetch: async () => {
        fetched = true;
        throw new Error("must not fetch");
      },
      verifier: acceptingVerifier(),
      now: () => now,
    });
    await assert.rejects(
      repository.get(),
      (error) =>
        error.code === "SIGNED_CATALOG_KEYS_UNCONFIGURED" &&
        error.message === "The signed Drowse catalog key ring has not been provisioned",
    );
    assert.equal(fetched, false);
  });

  test("an unverified distribution with configured keys fails before network access", async () => {
    let fetched = false;
    const repository = new VerifiedCatalogRepository({
      config: { ...config(), status: "feasibility-required" },
      store: memoryStore(),
      fetch: async () => {
        fetched = true;
        throw new Error("must not fetch");
      },
      verifier: acceptingVerifier(),
      now: () => now,
    });
    await assert.rejects(
      repository.get(),
      (error) =>
        error.code === "HOSTED_DISTRIBUTION_NOT_VERIFIED" &&
        error.message === "The hosted Drowse artifact distribution has not been published yet",
    );
    assert.equal(fetched, false);
  });

  test("catalog responses are bounded while streaming", async () => {
    const repository = new VerifiedCatalogRepository({
      config: config(),
      store: memoryStore(),
      verifier: acceptingVerifier(),
      now: () => now,
      fetch: async (url) => {
        const body = url.endsWith("catalog.json")
          ? catalogBytes()
          : new Uint8Array(16 * 1024 + 1);
        return {
          ok: true,
          status: 200,
          url,
          body: new ReadableStream({
            start(controller) {
              controller.enqueue(body);
              controller.close();
            },
          }),
          arrayBuffer: async () => body.slice().buffer,
        };
      },
    });
    await assert.rejects(
      repository.get(),
      (error) => error.code === "CATALOG_RESPONSE_SIZE_INVALID",
    );
  });

  test("concurrent repositories serialize sequence admission and preserve the newest catalog", async () => {
    const store = memoryStore(cachedState(catalogBytes({ sequence: 6 }), 6));
    const lock = serialLock();
    const gate = deferred();
    const started = deferred();
    const newerFetch = fixtureFetch(catalogBytes({ sequence: 8 }));
    const delayedNewerFetch = async (url, init) => {
      if (url.endsWith("catalog.json")) {
        started.resolve();
        await gate.promise;
      }
      return newerFetch(url, init);
    };
    const olderFetch = fixtureFetch(catalogBytes({ sequence: 7 }));
    const newerRepository = new VerifiedCatalogRepository({
      config: config(), store, fetch: delayedNewerFetch,
      verifier: acceptingVerifier(), now: () => now, runExclusive: lock,
    });
    const olderRepository = new VerifiedCatalogRepository({
      config: config(), store, fetch: olderFetch,
      verifier: acceptingVerifier(), now: () => now, runExclusive: lock,
    });

    const newer = newerRepository.get();
    await started.promise;
    const older = olderRepository.get();
    await Promise.resolve();
    assert.equal(olderFetch.calls.length, 0);
    gate.resolve();

    assert.equal((await newer).document.sequence, 8);
    await assert.rejects(older, (error) => error.code === "CATALOG_ROLLBACK");
    assert.equal(store.state.highestAcceptedSequence, 8);
    assert.equal(JSON.parse(new TextDecoder().decode(store.state.exactBytes)).sequence, 8);
  });

  test("download authorization holds catalog admission and rejects a superseded snapshot", async () => {
    const store = memoryStore(cachedState(catalogBytes(), 7));
    const lock = serialLock();
    const currentRepository = new VerifiedCatalogRepository({
      config: config(), store, fetch: fixtureFetch(catalogBytes()),
      verifier: acceptingVerifier(), now: () => now, runExclusive: lock,
    });
    const newerFetch = fixtureFetch(catalogBytes({ sequence: 8 }));
    const newerRepository = new VerifiedCatalogRepository({
      config: config(), store, fetch: newerFetch,
      verifier: acceptingVerifier(), now: () => now, runExclusive: lock,
    });
    const catalog = await currentRepository.get();
    const operationStarted = deferred();
    const releaseOperation = deferred();
    const authorized = currentRepository.withDownloadAuthorization(catalog, async () => {
      operationStarted.resolve();
      await releaseOperation.promise;
      return "installed";
    });
    await operationStarted.promise;
    const newer = newerRepository.get();
    await Promise.resolve();
    assert.equal(newerFetch.calls.length, 0);
    releaseOperation.resolve();
    assert.equal(await authorized, "installed");
    assert.equal((await newer).document.sequence, 8);
    await assert.rejects(
      currentRepository.withDownloadAuthorization(catalog, async () => "stale"),
      (error) => error.code === "CATALOG_SUPERSEDED",
    );
  });

  test("queued catalog download authorization is cancelled without waiting for another download", async () => {
    const store = memoryStore(cachedState(catalogBytes(), 7));
    const lock = serialLock();
    const firstRepository = new VerifiedCatalogRepository({
      config: config(), store, fetch: fixtureFetch(catalogBytes()),
      verifier: acceptingVerifier(), now: () => now, runExclusive: lock,
    });
    const secondRepository = new VerifiedCatalogRepository({
      config: config(), store, fetch: fixtureFetch(catalogBytes()),
      verifier: acceptingVerifier(), now: () => now, runExclusive: lock,
    });
    const catalog = await firstRepository.get();
    const firstStarted = deferred();
    const releaseFirst = deferred();
    const first = firstRepository.withDownloadAuthorization(catalog, async () => {
      firstStarted.resolve();
      await releaseFirst.promise;
    });
    await firstStarted.promise;

    let secondRan = false;
    const controller = new AbortController();
    const second = secondRepository.withDownloadAuthorization(catalog, async () => {
      secondRan = true;
    }, controller.signal);
    controller.abort(new DOMException("fixture cancelled", "AbortError"));
    await assert.rejects(second, (error) => error?.name === "AbortError");
    assert.equal(secondRan, false);
    releaseFirst.resolve();
    await first;
    await Promise.resolve();
    assert.equal(secondRan, false);
  });

  test("queued download authorization rechecks catalog expiry when admission begins", async () => {
    const store = memoryStore(cachedState(catalogBytes(), 7));
    const lock = serialLock();
    let currentNow = Date.parse("2026-12-31T23:59:59.000Z");
    const repository = new VerifiedCatalogRepository({
      config: config(),
      store,
      fetch: fixtureFetch(catalogBytes()),
      verifier: acceptingVerifier(),
      now: () => currentNow,
      runExclusive: lock,
    });
    const catalog = await repository.get();
    const firstStarted = deferred();
    const releaseFirst = deferred();
    const first = repository.withDownloadAuthorization(catalog, async () => {
      firstStarted.resolve();
      await releaseFirst.promise;
      return "installed";
    });
    await firstStarted.promise;

    let secondRan = false;
    const second = repository.withDownloadAuthorization(catalog, async () => {
      secondRan = true;
    });
    currentNow = Date.parse(catalog.document.expiresAt);
    releaseFirst.resolve();

    assert.equal(await first, "installed");
    await assert.rejects(second, (error) => error.code === "CATALOG_EXPIRED");
    assert.equal(secondRan, false);
  });

  test("clear waits for an in-flight admission and remains the final committed state", async () => {
    const store = memoryStore();
    const lock = serialLock();
    const gate = deferred();
    const started = deferred();
    const baseFetch = fixtureFetch(catalogBytes());
    const delayedFetch = async (url, init) => {
      if (url.endsWith("catalog.json")) {
        started.resolve();
        await gate.promise;
      }
      return baseFetch(url, init);
    };
    const fetchingRepository = new VerifiedCatalogRepository({
      config: config(), store, fetch: delayedFetch,
      verifier: acceptingVerifier(), now: () => now, runExclusive: lock,
    });
    const clearingRepository = new VerifiedCatalogRepository({
      config: config(), store, verifier: acceptingVerifier(),
      now: () => now, runExclusive: lock,
    });

    const fetching = fetchingRepository.get();
    await started.promise;
    const clearing = clearingRepository.clear();
    gate.resolve();
    await fetching;
    await clearing;

    assert.equal(store.state.highestAcceptedSequence, null);
    assert.equal(store.state.exactBytes, null);
  });

  test("catalog fetch deadline aborts both requests even when fetch ignores abort", async () => {
    const signals = [];
    const store = memoryStore();
    const lock = serialLock();
    const repository = new VerifiedCatalogRepository({
      config: config(),
      store,
      fetch: async (_url, init) => {
        signals.push(init.signal);
        return new Promise(() => {});
      },
      verifier: acceptingVerifier(),
      now: () => now,
      fetchTimeoutMs: 10,
      runExclusive: lock,
    });

    const fetching = repository.get();
    const clearing = repository.clear();
    await assert.rejects(
      fetching,
      (error) => error instanceof CatalogFetchError && error.code === "CATALOG_FETCH_TIMEOUT",
    );
    await clearing;
    assert.equal(signals.length, 2);
    assert.equal(signals.every((signal) => signal.aborted), true);
    assert.deepEqual(store.state, emptyState());
  });

  test("catalog fetch deadline cancels a stalled response body", async () => {
    let bodyCancelled = false;
    const repository = new VerifiedCatalogRepository({
      config: config(),
      store: memoryStore(),
      fetch: async (url) => {
        const signature = new TextEncoder().encode(JSON.stringify(FIXTURE_SIGNATURE));
        if (!url.endsWith("catalog.json")) {
          return response(url, signature);
        }
        return {
          ok: true,
          status: 200,
          url,
          body: new ReadableStream({
            cancel() {
              bodyCancelled = true;
            },
          }),
          arrayBuffer: async () => new ArrayBuffer(0),
        };
      },
      verifier: acceptingVerifier(),
      now: () => now,
      fetchTimeoutMs: 10,
    });

    await assert.rejects(
      repository.get(),
      (error) => error instanceof CatalogFetchError && error.code === "CATALOG_FETCH_TIMEOUT",
    );
    await Promise.resolve();
    assert.equal(bodyCancelled, true);
  });

  test("browser catalog store reopens after a versionchange closes its handle", async () => {
    const fake = fakeIndexedDb();
    const previous = Object.getOwnPropertyDescriptor(globalThis, "indexedDB");
    Object.defineProperty(globalThis, "indexedDB", {
      configurable: true,
      value: fake,
    });
    try {
      const store = new BrowserCatalogStateStore();
      await Promise.all([store.initialize(), store.initialize()]);
      assert.equal(fake.openCalls, 1);
      assert.equal(fake.handles[0].closed, false);

      fake.handles[0].onversionchange();
      assert.equal(fake.handles[0].closed, true);
      assert.deepEqual(await store.read(), emptyState());
      assert.equal(fake.openCalls, 2);
      assert.equal(fake.handles[1].closed, false);
    } finally {
      if (previous) Object.defineProperty(globalThis, "indexedDB", previous);
      else delete globalThis.indexedDB;
    }
  });

  test("blocked IndexedDB opens close handles that succeed after rejection", async () => {
    const restoreNavigator = replaceGlobal("navigator", {
      storage: { async getDirectory() { return {}; } },
      locks: { async request(_name, _options, operation) { return operation({}); } },
    });
    try {
      for (const create of [
        () => new BrowserCatalogStateStore(),
        () => new BrowserContentStore(),
        () => new BrowserDrowseArchiveRepository(),
      ]) {
        const fake = lateSuccessfulIndexedDb(true);
        const restoreIndexedDb = replaceGlobal("indexedDB", fake);
        try {
          await assert.rejects(create().initialize(), /block/i);
          await new Promise((resolve) => setTimeout(resolve, 0));
          assert.equal(fake.database.closeCalls, 1);
        } finally {
          restoreIndexedDb();
        }
      }
    } finally {
      restoreNavigator();
    }
  });

  test("OPFS acquisition failures close databases opened during initialization", async () => {
    const restoreNavigator = replaceGlobal("navigator", {
      storage: { async getDirectory() { throw new Error("OPFS unavailable"); } },
      locks: { async request(_name, _options, operation) { return operation({}); } },
    });
    try {
      for (const create of [
        () => new BrowserContentStore(),
        () => new BrowserDrowseArchiveRepository(),
      ]) {
        const fake = lateSuccessfulIndexedDb(false);
        const restoreIndexedDb = replaceGlobal("indexedDB", fake);
        try {
          await assert.rejects(create().initialize(), /OPFS unavailable/);
          assert.equal(fake.database.closeCalls, 1);
        } finally {
          restoreIndexedDb();
        }
      }
    } finally {
      restoreNavigator();
    }
  });

  test("stale initialization failures cannot tear down a replacement store", async () => {
    for (const create of [
      () => new BrowserContentStore(),
      () => new BrowserDrowseArchiveRepository(),
    ]) {
      const fake = successfulIndexedDb();
      const locks = controlledLocks();
      const directory = fakeDirectory();
      const restoreIndexedDb = replaceGlobal("indexedDB", fake);
      const restoreNavigator = replaceGlobal("navigator", {
        storage: { async getDirectory() { return directory; } },
        locks,
      });
      try {
        const repository = create();
        const first = repository.initialize();
        await waitForCalls(locks.calls, 1);
        fake.handles[0].onversionchange?.();

        const replacement = repository.initialize();
        await waitForCalls(locks.calls, 2);
        const firstRejected = assert.rejects(first, /old initialization failed/);
        locks.calls[0].reject(new Error("old initialization failed"));
        await firstRejected;

        assert.strictEqual(repository.initialize(), replacement);
        assert.equal(fake.handles[1].closeCalls, 0);

        const replacementRejected = assert.rejects(
          replacement,
          /replacement initialization stopped/,
        );
        locks.calls[1].reject(new Error("replacement initialization stopped"));
        await replacementRejected;
      } finally {
        restoreNavigator();
        restoreIndexedDb();
      }
    }
  });

  for (const { name, run } of tests) {
    await run();
    console.log(`ok - ${name}`);
  }
  console.log(`${tests.length} catalog repository tests passed`);
} finally {
  await server.close();
}

function config() {
  return {
    status: "verified",
    catalogUrl: "https://catalog.example/catalog.json",
    signatureUrl: "https://catalog.example/catalog.sig.json",
    minimumAcceptedSequence: 1,
    expectedRuntimeAbi: FIXTURE_ABI,
    runtimeLockModels: catalogRuntimeLockModels(),
    publicKeys: new Map([
      ["fixture-key", new Uint8Array(32)],
      ["next-key", new Uint8Array(32)],
    ]),
    allowedCatalogRedirectOrigins: new Set(["https://catalog.example"]),
    allowedArtifactRedirectOrigins: new Set(["https://artifacts.example"]),
  };
}

function acceptingVerifier() {
  return {
    hasKey: (keyId) => keyId === "fixture-key" || keyId === "next-key",
    verify: async () => true,
  };
}

function fixtureFetch(bytes, signature = FIXTURE_SIGNATURE) {
  const calls = [];
  const fetch = async (url, init) => {
    calls.push({ url, init });
    const body = url.endsWith("catalog.json")
      ? bytes
      : new TextEncoder().encode(JSON.stringify(signature));
    return {
      ok: true,
      status: 200,
      url,
      arrayBuffer: async () => body.slice().buffer,
    };
  };
  fetch.calls = calls;
  return fetch;
}

function response(url, body) {
  return {
    ok: true,
    status: 200,
    url,
    arrayBuffer: async () => body.slice().buffer,
  };
}

function cachedState(exactBytes, sequence) {
  return {
    id: "catalog",
    exactBytes,
    signature: structuredClone(FIXTURE_SIGNATURE),
    highestAcceptedSequence: sequence,
    acceptedAt: 1,
  };
}

function memoryStore(initial = null) {
  const store = {
    state: initial ?? {
      id: "catalog",
      exactBytes: null,
      signature: null,
      highestAcceptedSequence: null,
      acceptedAt: null,
    },
    async initialize() {},
    async read() {
      return cloneState(this.state);
    },
    async write(state) {
      this.state = cloneState(state);
    },
    async clear() {
      this.state = {
        id: "catalog",
        exactBytes: null,
        signature: null,
        highestAcceptedSequence: null,
        acceptedAt: null,
      };
    },
  };
  return store;
}

function emptyState() {
  return {
    id: "catalog",
    exactBytes: null,
    signature: null,
    highestAcceptedSequence: null,
    acceptedAt: null,
  };
}

function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function serialLock() {
  let tail = Promise.resolve();
  return (operation, signal) => {
    const predecessor = tail;
    const turn = predecessor.then(async () => {
      if (signal?.aborted) throw signal.reason;
      return operation();
    });
    tail = turn.then(() => undefined, () => undefined);
    if (!signal) return turn;
    return Promise.race([
      turn,
      new Promise((_, reject) => {
        if (signal.aborted) reject(signal.reason);
        else signal.addEventListener("abort", () => reject(signal.reason), { once: true });
      }),
    ]);
  };
}

function fakeIndexedDb() {
  let value;
  const fake = {
    openCalls: 0,
    handles: [],
    open() {
      this.openCalls += 1;
      const handle = fakeDatabaseHandle(() => value, (next) => {
        value = next;
      });
      this.handles.push(handle);
      const request = { result: handle, error: null };
      queueMicrotask(() => {
        if (this.openCalls === 1) request.onupgradeneeded?.();
        request.onsuccess?.();
      });
      return request;
    },
  };
  return fake;
}

function fakeDatabaseHandle(readValue, writeValue) {
  return {
    closed: false,
    onversionchange: null,
    objectStoreNames: { contains: () => false },
    createObjectStore() {},
    close() {
      this.closed = true;
    },
    transaction() {
      if (this.closed) throw new DOMException("Database is closed", "InvalidStateError");
      const transaction = {
        error: null,
        objectStore() {
          return {
            get() {
              const request = { result: undefined, error: null };
              queueMicrotask(() => {
                request.result = readValue();
                request.onsuccess?.();
              });
              return request;
            },
            put(next) {
              writeValue(structuredClone(next));
              queueMicrotask(() => transaction.oncomplete?.());
            },
            clear() {
              writeValue(undefined);
              queueMicrotask(() => transaction.oncomplete?.());
            },
          };
        },
      };
      return transaction;
    },
  };
}

function cloneState(state) {
  return {
    ...state,
    exactBytes: state.exactBytes?.slice() ?? null,
    signature: state.signature ? structuredClone(state.signature) : null,
  };
}

function replaceGlobal(name, value) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, {
    configurable: true,
    writable: true,
    value,
  });
  return () => {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  };
}

function lateSuccessfulIndexedDb(blocked) {
  const database = {
    closeCalls: 0,
    objectStoreNames: { contains: () => true },
    createObjectStore() {},
    close() { this.closeCalls += 1; },
  };
  return {
    database,
    open() {
      const request = { result: database, error: null };
      queueMicrotask(() => {
        if (blocked) request.onblocked?.();
        queueMicrotask(() => request.onsuccess?.());
      });
      return request;
    },
  };
}

function successfulIndexedDb() {
  const handles = [];
  return {
    handles,
    open() {
      const database = {
        closeCalls: 0,
        onversionchange: null,
        objectStoreNames: { contains: () => true },
        createObjectStore() {},
        close() { this.closeCalls += 1; },
      };
      handles.push(database);
      const request = { result: database, error: null };
      queueMicrotask(() => request.onsuccess?.());
      return request;
    },
  };
}

function controlledLocks() {
  const calls = [];
  return {
    calls,
    request() {
      let reject;
      const promise = new Promise((_resolve, rejectPromise) => {
        reject = rejectPromise;
      });
      calls.push({ promise, reject });
      return promise;
    },
  };
}

function fakeDirectory() {
  return {
    async getDirectoryHandle() { return this; },
  };
}

async function waitForCalls(calls, count) {
  for (let attempt = 0; attempt < 100 && calls.length < count; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  assert.equal(calls.length, count);
}
