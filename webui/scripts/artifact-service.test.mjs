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

try {
  const { BrowserDrowseArchiveService, validateDrowseArchive } = await server.ssrLoadModule(
    "/src/hosted/artifacts/index.ts",
  );

  function harness() {
    const calls = [];
    const pack = {
      id: "manifolds/local/demo",
      namespace: "local",
      name: "demo",
      template: null,
      installedAt: 1,
      producerVersion: "test",
      source: { uri: "local", repository: null, revision: null },
    };
    const repository = {
      async initialize() { calls.push(["initialize"]); },
      async install(source, options) {
        calls.push(["install", source, options.force]);
        options.onProgress?.({
          phase: "verifying",
          path: "manifolds/local/demo/manifold.json",
          verifiedBytes: 10,
          totalBytes: 20,
        });
        return pack;
      },
      async list() { calls.push(["list"]); return [pack]; },
      async remove(id) { calls.push(["remove", id]); return true; },
      async file() { return null; },
      async export(id) { calls.push(["export", id]); return new Blob([id]); },
      async withExport(id, consumer, signal) {
        calls.push(["lease", id]);
        try {
          signal?.throwIfAborted();
          return await consumer(new Blob([id]));
        } finally {
          calls.push(["release", id]);
        }
      },
      async clear() { calls.push(["clear"]); },
      async close() { calls.push(["close"]); },
    };
    const templates = {
      async initialize() {},
      async list() { return []; },
      async get() { return null; },
      async put() { throw new Error("unused"); },
      async remove() { return false; },
      async clear() {},
      async close() {},
    };
    const huggingFace = {
      async search() { throw new Error("unused"); },
      async withArchive() { throw new Error("unused"); },
      async clear() {},
    };
    return {
      service: new BrowserDrowseArchiveService(repository, templates, "test", huggingFace),
      repository,
      calls,
      pack,
    };
  }

  test("routes only the reserved manifold artifact methods", () => {
    const { service } = harness();
    assert.equal(service.handles({ service: "manifolds", method: "drowseArchiveList", args: [] }), true);
    assert.equal(service.handles({ service: "manifolds", method: "list", args: [] }), true);
    assert.equal(service.handles({ service: "templates", method: "drowseArchiveList", args: [] }), false);
  });

  test("lists, exports, and deletes exact canonical identities", async () => {
    const { service, calls, pack } = harness();
    assert.deepEqual(
      await service.request(
        { service: "manifolds", method: "drowseArchiveList", args: [] },
        () => {},
      ),
      { packs: [pack] },
    );
    const archive = await service.request(
      { service: "manifolds", method: "drowseArchiveExport", args: [pack.id] },
      () => {},
    );
    assert.equal(await archive.text(), pack.id);
    assert.deepEqual(
      await service.request(
        { service: "manifolds", method: "drowseArchiveDelete", args: [pack.id] },
        () => {},
      ),
      { id: pack.id, removed: true },
    );
    assert.deepEqual(calls.map((call) => call[0]), [
      "initialize", "list", "initialize", "export", "initialize", "remove",
    ]);
  });

  test("leases compiler archives only until successful consumption", async () => {
    const { service, calls } = harness();
    const signal = new AbortController().signal;
    const ids = ["manifolds/local/demo", "manifolds/local/second"];
    const result = await service.withDrowseArchiveArchives(ids, signal, async (archives) => {
      calls.push(["consume", archives.length]);
      return Promise.all(archives.map((archive) => archive.text()));
    });
    assert.deepEqual(result, ids);
    assert.deepEqual(calls, [
      ["initialize"],
      ["lease", ids[0]],
      ["lease", ids[1]],
      ["consume", 2],
      ["release", ids[1]],
      ["release", ids[0]],
    ]);
  });

  test("releases every compiler archive when consumption fails", async () => {
    const { service, calls } = harness();
    const ids = ["manifolds/local/demo", "manifolds/local/second"];
    await assert.rejects(
      service.withDrowseArchiveArchives(
        ids,
        new AbortController().signal,
        async () => { throw new Error("compiler failed"); },
      ),
      /compiler failed/,
    );
    assert.deepEqual(calls, [
      ["initialize"],
      ["lease", ids[0]],
      ["lease", ids[1]],
      ["release", ids[1]],
      ["release", ids[0]],
    ]);
  });

  test("releases every compiler archive when consumption is cancelled", async () => {
    const { service, calls } = harness();
    const ids = ["manifolds/local/demo", "manifolds/local/second"];
    const controller = new AbortController();
    await assert.rejects(
      service.withDrowseArchiveArchives(ids, controller.signal, async () => {
        controller.abort(new DOMException("cancel compile", "AbortError"));
        controller.signal.throwIfAborted();
      }),
      (error) => error instanceof DOMException && error.name === "AbortError",
    );
    assert.deepEqual(calls, [
      ["initialize"],
      ["lease", ids[0]],
      ["lease", ids[1]],
      ["release", ids[1]],
      ["release", ids[0]],
    ]);
  });

  test("installs Blob input with force and progress intact", async () => {
    const { service, calls, pack } = harness();
    const events = [];
    const source = new Blob(["archive"]);
    assert.equal(
      await service.request(
        {
          service: "manifolds",
          method: "drowseArchiveInstall",
          args: [source, { force: true }],
        },
        (event) => events.push(event),
      ),
      pack,
    );
    assert.equal(calls[1][1], source);
    assert.equal(calls[1][2], true);
    assert.deepEqual(events, [{
      event: "progress",
      data: {
        phase: "verifying",
        path: "manifolds/local/demo/manifold.json",
        verifiedBytes: 10,
        totalBytes: 20,
      },
    }]);
  });

  test("rejects malformed argument counts, identities, options, and non-Blob input", async () => {
    const { service } = harness();
    const request = (method, args) => service.request({ service: "manifolds", method, args }, () => {});
    await assert.rejects(request("drowseArchiveList", [1]), /expected 0 arguments/);
    await assert.rejects(request("drowseArchiveExport", ["../escape"]), /must be manifolds/);
    await assert.rejects(request("drowseArchiveInstall", [new Blob(), { force: "yes" }]), /options are invalid/);
    await assert.rejects(request("drowseArchiveInstall", [new Uint8Array(), {}]), /requires a Blob/);
  });

  test("clear and close are explicit repository lifecycle operations", async () => {
    const { service, calls } = harness();
    await service.clear();
    await service.close();
    assert.deepEqual(calls, [["clear"], ["close"]]);
  });

  test("authors verified manifolds and templates through the runtime service", async () => {
    const { service } = authoringHarness(validateDrowseArchive, BrowserDrowseArchiveService);
    const call = (serviceName, method, args = []) => service.request(
      { service: serviceName, method, args },
      () => {},
    );

    const template = await call("templates", "create", [{
      namespace: "local",
      name: "weekday",
      slot: "[DAY]",
      values: ["Monday", "Tuesday"],
      contexts: [{
        turns: [{ role: "user", content: "What day is it?" }],
        assistant: "Today is [DAY].",
      }],
      description: "weekday fixture",
    }]);
    assert.deepEqual(template.labels, ["monday", "tuesday"]);
    assert.equal((await call("templates", "list")).templates.length, 1);

    const templated = await call("manifolds", "createFromTemplate", [{
      namespace: "local",
      name: "weekday_axis",
      template_ref: "local/weekday",
      fit_mode: "pca",
    }]);
    assert.equal(templated.template_ref, "local/weekday");
    assert.deepEqual(templated.nodes.map((node) => node.statements), [
      ["Today is Monday."],
      ["Today is Tuesday."],
    ]);

    const authored = await call("manifolds", "create", [{
      namespace: "local",
      name: "tone",
      description: "tone fixture",
      domain: {
        type: "box",
        axes: [{ name: "tone", periodic: false, period: 1, lo: -1, hi: 1 }],
      },
      nodes: [
        { label: "calm", coords: [-1], statements: ["calm"] },
        { label: "neutral", coords: [0], statements: ["neutral"] },
        { label: "alert", coords: [1], statements: ["alert"] },
      ],
    }]);
    assert.equal(authored.domain_label, "box(1d)");
    assert.equal(authored.advisories.length, 0);

    const unpoised = await call("manifolds", "create", [{
      namespace: "local",
      name: "flat_plane",
      description: "advisory fixture",
      domain: {
        type: "box",
        axes: [
          { name: "x", periodic: false, period: 1, lo: -1, hi: 1 },
          { name: "y", periodic: false, period: 1, lo: -1, hi: 1 },
        ],
      },
      nodes: [
        { label: "a", coords: [-1, 0], statements: ["a"] },
        { label: "b", coords: [-0.5, 0], statements: ["b"] },
        { label: "c", coords: [0, 0], statements: ["c"] },
        { label: "d", coords: [0.5, 0], statements: ["d"] },
        { label: "e", coords: [1, 0], statements: ["e"] },
      ],
    }]);
    assert.equal(unpoised.advisories.length, 2);
    assert.match(unpoised.advisories[0], /affine rank 1.*embedding has 2 dimensions/);
    assert.match(unpoised.advisories[1], /axis 'y' has only 1 distinct/);

    const discover = await call("manifolds", "createDiscover", [{
      namespace: "local",
      name: "energy",
      fit_mode: "pca",
      nodes: [
        { label: "low", statements: ["low energy"] },
        { label: "high", statements: ["high energy"] },
      ],
    }]);
    assert.equal(discover.domain_label, "discover-pca");
    assert.equal(discover.intrinsic_dim, 0);

    const listed = await call("manifolds", "list");
    assert.deepEqual(listed.manifolds.map((item) => item.name), [
      "energy", "flat_plane", "tone", "weekday_axis",
    ]);
    const detail = await call("manifolds", "get", ["local", "tone"]);
    assert.deepEqual(detail.archive_source, {
      uri: "local",
      repository: null,
      revision: null,
    });
    assert.deepEqual(detail.nodes[2], {
      label: "alert",
      coords: [1],
      statements: ["alert"],
      role: null,
    });
    assert.deepEqual(
      await call("manifolds", "delete", ["local", "energy"]),
      { namespace: "local", name: "energy", removed: true },
    );
  });

  test("rejects invalid authoring before commit and preserves force semantics", async () => {
    const { service, repository } = authoringHarness(validateDrowseArchive, BrowserDrowseArchiveService);
    const call = (serviceName, method, args = []) => service.request(
      { service: serviceName, method, args },
      () => {},
    );
    const invalid = {
      namespace: "local",
      name: "bad",
      description: "too few nodes",
      domain: {
        type: "box",
        axes: [{ name: "x", periodic: false, period: 1, lo: 0, hi: 1 }],
      },
      nodes: [{ label: "only", coords: [0], statements: ["only"] }],
    };
    await assert.rejects(call("manifolds", "create", [invalid]), /needs at least 3 nodes/);
    assert.equal(repository.installs.size, 0);

    const template = {
      namespace: "local",
      name: "choice",
      slot: "[X]",
      values: ["A", "B"],
      contexts: [{
        turns: [{ role: "user", content: "choose" }],
        assistant: "[X]",
      }],
    };
    await call("templates", "create", [template]);
    await assert.rejects(call("templates", "create", [template]), /already exists/);
    const replaced = await call("templates", "create", [{
      ...template,
      force: true,
      description: "replacement",
    }]);
    assert.equal(replaced.description, "replacement");
  });

  test("merges discover closures without invoking the model backend", async () => {
    const { service } = authoringHarness(validateDrowseArchive, BrowserDrowseArchiveService);
    const call = (method, args = []) => service.request(
      { service: "manifolds", method, args },
      () => {},
    );
    await call("createDiscover", [{
      namespace: "local",
      name: "temperatures",
      fit_mode: "pca",
      hyperparams: { max_dim: 2 },
      nodes: [
        { label: "cold", statements: ["cold"], role: "critic" },
        { label: "hot", statements: ["hot"] },
      ],
    }]);
    await call("createDiscover", [{
      namespace: "local",
      name: "pressures",
      fit_mode: "pca",
      nodes: [
        { label: "low_pressure", statements: ["low pressure"] },
        { label: "high_pressure", statements: ["high pressure"] },
      ],
    }]);
    const merged = await call("merge", [{
      namespace: "local",
      name: "weather",
      description: "merged fixture",
      sources: [
        { namespace: "local", name: "temperatures" },
        { namespace: "local", name: "pressures" },
      ],
    }]);
    assert.equal(merged.fit_mode, "pca");
    assert.equal(merged.description, "merged fixture");
    assert.deepEqual(merged.hyperparams, { max_dim: 2 });
    assert.deepEqual(merged.node_labels, [
      "cold", "hot", "low_pressure", "high_pressure",
    ]);
    assert.equal(merged.nodes[0].role, "critic");

    await assert.rejects(call("merge", [{
      name: "weather",
      sources: [
        { namespace: "local", name: "temperatures" },
        { namespace: "local", name: "pressures" },
      ],
    }]), /already installed/);
    const replaced = await call("merge", [{
      name: "weather",
      description: "replacement",
      force: true,
      sources: [
        { namespace: "local", name: "temperatures" },
        { namespace: "local", name: "pressures" },
      ],
    }]);
    assert.equal(replaced.description, "replacement");

    await call("createDiscover", [{
      namespace: "local",
      name: "collision",
      fit_mode: "pca",
      nodes: [
        { label: "cold", statements: ["duplicate"] },
        { label: "mild", statements: ["mild"] },
      ],
    }]);
    await assert.rejects(call("merge", [{
      name: "collision_target",
      sources: [
        { namespace: "local", name: "temperatures" },
        { namespace: "local", name: "collision" },
      ],
    }]), /label "cold" appears/);

    await call("createDiscover", [{
      namespace: "local",
      name: "automatic",
      fit_mode: "auto",
      nodes: [
        { label: "dry", statements: ["dry"] },
        { label: "wet", statements: ["wet"] },
      ],
    }]);
    await assert.rejects(call("merge", [{
      name: "mixed_target",
      sources: [
        { namespace: "local", name: "temperatures" },
        { namespace: "local", name: "automatic" },
      ],
    }]), /mixed fit modes/);

    const automaticMerge = await call("merge", [{
      name: "automatic_target",
      fit_mode: "auto",
      sources: [
        { namespace: "local", name: "temperatures" },
        { namespace: "local", name: "automatic" },
      ],
    }]);
    assert.equal(automaticMerge.fit_mode, "auto");
    assert.equal(automaticMerge.domain_label, "discover-auto");
  });

  test("routes immutable Hugging Face discovery through provenance-bound install", async () => {
    const archive = new File(["fixture"], "portable.drowse");
    const calls = [];
    const revision = "a".repeat(40);
    const primary = "manifolds/local/portable";
    const source = {
      uri: `hf://fixture/portable@${revision}`,
      repository: "fixture/portable",
      revision,
    };
    const manifold = {
      format_version: 10,
      name: "portable",
      description: "Portable fixture manifold",
      source: source.uri,
      tags: ["fixture", "portable"],
      template_ref: null,
      fit_mode: "pca",
      hyperparams: { max_dim: 1 },
      nodes: [
        { label: "warm", role: null, kind: "abstract" },
        { label: "cool", role: "assistant", kind: "concrete" },
      ],
      files: {
        [`${primary}/nodes/00_warm.json`]: "0".repeat(64),
        [`${primary}/nodes/01_cool.json`]: "1".repeat(64),
      },
    };
    const files = new Map([
      [`${primary}/manifold.json`, new Blob([JSON.stringify(manifold)])],
      [`${primary}/nodes/00_warm.json`, new Blob([JSON.stringify(["a warm response"])])],
      [`${primary}/nodes/01_cool.json`, new Blob([JSON.stringify(["a cool response"])])],
    ]);
    const repository = {
      async initialize() {},
      async install(input, options) {
        calls.push(["install", input, options]);
        return {
          id: primary,
          namespace: "local",
          name: "portable",
          template: null,
          installedAt: 1,
          producerVersion: "test",
          source,
        };
      },
      async list() { return []; },
      async remove() { return false; },
      async file(_primary, path) { return files.get(path) ?? null; },
      async export() { throw new Error("unused"); },
      async clear() {},
      async close() {},
    };
    const templates = {
      async initialize() {},
      async list() { return []; },
      async get() { return null; },
      async put() { throw new Error("unused"); },
      async remove() { return false; },
      async clear() {},
      async close() {},
    };
    const huggingFace = {
      async search(query, limit) {
        calls.push(["search", query, limit]);
        return { query, results: [] };
      },
      async withArchive(target, options, consume) {
        calls.push(["download", target]);
        options.onProgress({
          phase: "downloading",
          path: "portable.drowse",
          downloadedBytes: 7,
          totalBytes: 7,
        });
        return consume({
          file: archive,
          repository: "fixture/portable",
          revision,
          filename: "portable.drowse",
        });
      },
      async clear() {},
    };
    const service = new BrowserDrowseArchiveService(repository, templates, "test", huggingFace);
    assert.deepEqual(
      await service.request(
        { service: "manifolds", method: "search", args: ["portable", 4] },
        () => {},
      ),
      { query: "portable", results: [] },
    );
    const progress = [];
    const installed = await service.request(
      {
        service: "manifolds",
        method: "install",
        args: [{ target: `fixture/portable@${revision}`, force: true }],
      },
      (event) => progress.push(event),
    );
    assert.deepEqual(installed, {
      namespace: "local",
      name: "portable",
      description: "Portable fixture manifold",
      source: source.uri,
      tags: ["fixture", "portable"],
      template_ref: null,
      fit_mode: "pca",
      is_discover: true,
      domain: {},
      domain_label: "discover-pca",
      intrinsic_dim: 0,
      min_nodes: null,
      node_count: 2,
      node_labels: ["warm", "cool"],
      node_coords: [],
      node_roles: [null, "assistant"],
      node_kinds: ["abstract", "concrete"],
      hyperparams: { max_dim: 1 },
      fitted_models: [],
      tensor_variants: {},
      fitted_for_session: false,
      stale: false,
      resolved_fit_mode: null,
      nodes: [
        { label: "warm", coords: null, statements: ["a warm response"], role: null },
        { label: "cool", coords: null, statements: ["a cool response"], role: "assistant" },
      ],
      fitted: [],
      archive_source: source,
    });
    assert.equal(`installed ${installed.namespace}/${installed.name}`, "installed local/portable");
    assert.equal("id" in installed, false);
    assert.deepEqual(calls.slice(0, 2), [
      ["search", "portable", 4],
      ["download", `fixture/portable@${revision}`],
    ]);
    assert.equal(calls[2][0], "install");
    assert.equal(calls[2][1], archive);
    assert.deepEqual(calls[2][2].expectedSource, {
      repository: "fixture/portable",
      revision,
    });
    assert.equal(calls[2][2].force, true);
    assert.equal(progress[0].data.phase, "downloading");
    const callsBeforeRejectedRename = calls.length;
    await assert.rejects(
      service.request(
        { service: "manifolds", method: "install", args: [{ target: "fixture/portable", as_: "local/renamed" }] },
        () => {},
      ),
      /preserve the archive identity.*omit as_/,
    );
    assert.equal(calls.length, callsBeforeRejectedRename);
  });

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

function authoringHarness(validateDrowseArchive, BrowserDrowseArchiveService) {
  const installs = new Map();
  const repository = {
    installs,
    async initialize() {},
    async install(source, options = {}) {
      const chunks = new Map();
      let inspected;
      const verified = await validateDrowseArchive(source, {
        signal: options.signal,
        onProgress: options.onProgress,
        stage: {
          begin(pack) {
            inspected = pack;
            chunks.set("pack.json", [pack.exactManifestBytes.slice()]);
          },
          write(path, chunk) {
            const existing = chunks.get(path) ?? [];
            existing.push(chunk.slice());
            chunks.set(path, existing);
          },
          finish() {},
          commit() {},
          rollback() {},
        },
      });
      if (installs.has(verified.manifest.primary) && options.force !== true) {
        throw new Error(`Drowse manifold pack ${verified.manifest.primary} is already installed`);
      }
      const files = new Map([...chunks].map(([path, parts]) => [path, join(parts)]));
      installs.set(verified.manifest.primary, {
        manifest: verified.manifest,
        files,
        archive: source,
        installedAt: Date.now(),
      });
      const [, namespace, name] = inspected.manifest.primary.split("/");
      return {
        id: inspected.manifest.primary,
        namespace,
        name,
        template: inspected.manifest.template,
        installedAt: installs.get(inspected.manifest.primary).installedAt,
        producerVersion: inspected.manifest.producer.version,
        source: inspected.manifest.source,
      };
    },
    async list() {
      return [...installs].sort(([left], [right]) => left.localeCompare(right)).map(([id, item]) => {
        const [, namespace, name] = id.split("/");
        return {
          id,
          namespace,
          name,
          template: item.manifest.template,
          installedAt: item.installedAt,
          producerVersion: item.manifest.producer.version,
          source: item.manifest.source,
        };
      });
    },
    async remove(id) { return installs.delete(id); },
    async file(id, path) {
      const bytes = installs.get(id)?.files.get(path);
      return bytes ? new Blob([bytes]) : null;
    },
    async export(id) { return installs.get(id)?.archive ?? null; },
    async clear() { installs.clear(); },
    async close() {},
  };
  const records = new Map();
  const templates = {
    async initialize() {},
    async list() { return [...records.values()].map((record) => structuredClone(record)); },
    async get(id) { return records.has(id) ? structuredClone(records.get(id)) : null; },
    async put(namespace, name, payload, force) {
      const id = `${namespace}/${name}`;
      if (records.has(id) && !force) throw new Error(`Template ${id} already exists`);
      const record = { id, namespace, name, payload: structuredClone(payload), installedAt: Date.now() };
      records.set(id, record);
      return structuredClone(record);
    },
    async remove(id) { return records.delete(id); },
    async clear() { records.clear(); },
    async close() {},
  };
  const huggingFace = {
    async search() { throw new Error("unused"); },
    async withArchive() { throw new Error("unused"); },
    async clear() {},
  };
  return {
    service: new BrowserDrowseArchiveService(repository, templates, "browser-test", huggingFace),
    repository,
    templates,
  };
}

function join(parts) {
  const result = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result;
}
