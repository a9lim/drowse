import { expect, test } from "@playwright/test";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { zipSync } from "fflate";

const devUrl = "http://127.0.0.1:4176";
const moduleUrl = `/@fs/${resolve("src/hosted/artifacts/index.ts")}`;

test("browser manifold packs commit, survive reopen, export, roll back, and delete", async ({ page }) => {
  const validArchive = archiveFixture([["calm"], ["alert"]]);
  const invalidReplacement = archiveFixture([[], ["alert"]]);
  await page.goto(`${devUrl}/app?fixture=1`);

  const result = await page.evaluate(async ({ moduleUrl, valid, invalid }) => {
    let step = "import module";
    try {
    const artifacts = await import(moduleUrl);
    const decode = (base64: string) => Uint8Array.from(atob(base64), (character) => character.charCodeAt(0));
    const repository = new artifacts.BrowserDrowseArchiveRepository();
    step = "install";
    const installed = await repository.install(new Blob([decode(valid)]));
    step = "list";
    const initial = await repository.list();
    step = "read original";
    const original = await repository.file(
      installed.id,
      "manifolds/local/demo/nodes/00_calm.json",
    );
    let replacementCode: string | null = null;
    step = "replace invalid";
    try {
      await repository.install(new Blob([decode(invalid)]), { force: true });
    } catch (error) {
      replacementCode = error instanceof artifacts.DrowseArchiveError
        ? error.code
        : error instanceof Error ? error.message : String(error);
    }
    const afterFailedReplacement = await repository.file(
      installed.id,
      "manifolds/local/demo/nodes/00_calm.json",
    );
    const originalText = await original?.text();
    const replacementText = await afterFailedReplacement?.text();
    const realNow = Date.now;
    Date.now = () => installed.installedAt;
    let replaced: { installedAt: number };
    try {
      replaced = await repository.install(new Blob([decode(valid)]), { force: true });
    } finally {
      Date.now = realNow;
    }
    step = "export";
    const exported = await repository.export(installed.id);
    const exportedName = exported.name;
    const exportedType = exported.type;
    step = "replace export stage";
    const replacementExport = await repository.export(installed.id);
    await artifacts.validateDrowseArchive(replacementExport);
    step = "verify first export after second export";
    const verifiedExport = await artifacts.validateDrowseArchive(exported);
    const root = await navigator.storage.getDirectory();
    const drowse = await root.getDirectoryHandle("drowse");
    const artifactsDirectory = await drowse.getDirectoryHandle("artifacts");
    const exportsDirectory = await artifactsDirectory.getDirectoryHandle("exports");
    const countExports = async () => {
      let count = 0;
      for await (const _entry of exportsDirectory.entries()) count += 1;
      return count;
    };
    let stagedExports = 0;
    for await (const _entry of exportsDirectory.entries()) stagedExports += 1;
    let leasedDuringSuccess = 0;
    await repository.withExport(installed.id, async (archive) => {
      leasedDuringSuccess = await countExports();
      await artifacts.validateDrowseArchive(archive);
    });
    const stagedAfterLeaseSuccess = await countExports();
    let leasedDuringFailure = 0;
    try {
      await repository.withExport(installed.id, async () => {
        leasedDuringFailure = await countExports();
        throw new Error("compiler fixture failed");
      });
    } catch (error) {
      if (!(error instanceof Error) || error.message !== "compiler fixture failed") throw error;
    }
    const stagedAfterLeaseFailure = await countExports();
    const leaseController = new AbortController();
    try {
      await repository.withExport(installed.id, async () => {
        leaseController.abort(new DOMException("cancel compile", "AbortError"));
        leaseController.signal.throwIfAborted();
      }, leaseController.signal);
    } catch (error) {
      if (!(error instanceof DOMException) || error.name !== "AbortError") throw error;
    }
    const stagedAfterLeaseCancellation = await countExports();
    step = "close first";
    await repository.close();

    step = "reopen";
    const reopened = new artifacts.BrowserDrowseArchiveRepository();
    const persisted = await reopened.list();
    const firstExportBytesAfterReopen = (await exported.arrayBuffer()).byteLength;
    let stagedExportsAfterReopen = 0;
    for await (const _entry of exportsDirectory.entries()) stagedExportsAfterReopen += 1;
    step = "remove";
    const removed = await reopened.remove(installed.id);
    const afterDelete = await reopened.list();
    const inaccessible = await reopened.file(
      installed.id,
      "manifolds/local/demo/manifold.json",
    );
    await reopened.close();
    return {
      installed,
      initial,
      original: originalText,
      replacementCode,
      afterFailedReplacement: replacementText,
      replacementTimestampAdvanced: replaced.installedAt > installed.installedAt,
      exportedName,
      exportedType,
      exportedPrimary: verifiedExport.manifest.primary,
      stagedExports,
      leasedDuringSuccess,
      stagedAfterLeaseSuccess,
      leasedDuringFailure,
      stagedAfterLeaseFailure,
      stagedAfterLeaseCancellation,
      stagedExportsAfterReopen,
      firstExportBytesAfterReopen,
      persisted,
      removed,
      afterDelete,
      inaccessible: inaccessible === null,
    };
    } catch (error) {
      throw new Error(`${step}: ${error instanceof Error ? `${error.name}: ${error.message}` : String(error)}`);
    }
  }, {
    moduleUrl,
    valid: Buffer.from(validArchive).toString("base64"),
    invalid: Buffer.from(invalidReplacement).toString("base64"),
  });

  expect(result.installed.id).toBe("manifolds/local/demo");
  expect(result.initial).toHaveLength(1);
  expect(result.original).toBe('["calm"]\n');
  expect(result.replacementCode).toBe("CLOSURE_INVALID");
  expect(result.afterFailedReplacement).toBe('["calm"]\n');
  expect(result.replacementTimestampAdvanced).toBe(true);
  expect(result.exportedName).toMatch(/^[0-9a-f-]+\.drowse$/);
  expect(result.exportedType).toBe("");
  expect(result.exportedPrimary).toBe("manifolds/local/demo");
  expect(result.stagedExports).toBe(2);
  expect(result.leasedDuringSuccess).toBe(3);
  expect(result.stagedAfterLeaseSuccess).toBe(2);
  expect(result.leasedDuringFailure).toBe(3);
  expect(result.stagedAfterLeaseFailure).toBe(2);
  expect(result.stagedAfterLeaseCancellation).toBe(2);
  expect(result.stagedExportsAfterReopen).toBe(2);
  expect(result.firstExportBytesAfterReopen).toBeGreaterThan(0);
  expect(result.persisted).toHaveLength(1);
  expect(result.removed).toBe(true);
  expect(result.afterDelete).toEqual([]);
  expect(result.inaccessible).toBe(true);
});

test("startup removes uncommitted OPFS artifact directories", async ({ page }) => {
  await page.goto(`${devUrl}/app?fixture=1`);
  const recovered = await page.evaluate(async ({ moduleUrl }) => {
    const root = await navigator.storage.getDirectory();
    const drowse = await root.getDirectoryHandle("drowse", { create: true });
    const artifactsDirectory = await drowse.getDirectoryHandle("artifacts", { create: true });
    const stages = await artifactsDirectory.getDirectoryHandle("stages", { create: true });
    const orphan = await stages.getDirectoryHandle("orphaned-stage", { create: true });
    const handle = await orphan.getFileHandle("unverified.bin", { create: true });
    const writable = await handle.createWritable();
    await writable.write(new Uint8Array([1, 2, 3]));
    await writable.close();

    const artifacts = await import(moduleUrl);
    const repository = new artifacts.BrowserDrowseArchiveRepository();
    await repository.initialize();
    let missing = false;
    try {
      await stages.getDirectoryHandle("orphaned-stage");
    } catch (error) {
      missing = error instanceof DOMException && error.name === "NotFoundError";
    }
    await repository.close();
    return missing;
  }, { moduleUrl });
  expect(recovered).toBe(true);
});

test("startup removes committed metadata when its OPFS stage is missing", async ({ page }) => {
  const validArchive = archiveFixture([["calm"], ["alert"]]);
  await page.goto(`${devUrl}/app?fixture=1`);
  const recovered = await page.evaluate(async ({ moduleUrl, valid }) => {
    const decode = (base64: string) =>
      Uint8Array.from(atob(base64), (character) => character.charCodeAt(0));
    const artifacts = await import(moduleUrl);
    const repository = new artifacts.BrowserDrowseArchiveRepository();
    const installed = await repository.install(new Blob([decode(valid)]));
    const root = await navigator.storage.getDirectory();
    const drowse = await root.getDirectoryHandle("drowse");
    const artifactsDirectory = await drowse.getDirectoryHandle("artifacts");
    const stages = await artifactsDirectory.getDirectoryHandle("stages");
    let stageId: string | null = null;
    for await (const [name, handle] of stages.entries()) {
      if (handle.kind === "directory") stageId = name;
    }
    if (!stageId) throw new Error("Committed fixture stage was not created");
    await repository.close();
    await stages.removeEntry(stageId, { recursive: true });

    const reopened = new artifacts.BrowserDrowseArchiveRepository();
    const installedAfterRecovery = await reopened.list();
    const file = await reopened.file(
      installed.id,
      "manifolds/local/demo/manifold.json",
    );
    let stageMissing = false;
    try {
      await stages.getDirectoryHandle(stageId);
    } catch (error) {
      stageMissing = error instanceof DOMException && error.name === "NotFoundError";
    }
    await reopened.close();
    return {
      installedAfterRecovery,
      fileMissing: file === null,
      stageMissing,
    };
  }, {
    moduleUrl,
    valid: Buffer.from(validArchive).toString("base64"),
  });
  expect(recovered.installedAfterRecovery).toEqual([]);
  expect(recovered.fileMissing).toBe(true);
  expect(recovered.stageMissing).toBe(true);
});

test("startup scrubs malformed committed transaction and install records", async ({ page }) => {
  const firstArchive = archiveFixture([["calm"], ["alert"]], "first");
  const secondArchive = archiveFixture([["quiet"], ["loud"]], "second");
  await page.goto(`${devUrl}/app?fixture=1`);
  const recovered = await page.evaluate(async ({ moduleUrl, first, second }) => {
    const decode = (base64: string) =>
      Uint8Array.from(atob(base64), (character) => character.charCodeAt(0));
    const requestValue = <T,>(request: IDBRequest<T>) => new Promise<T>((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    const transactionDone = (transaction: IDBTransaction) => new Promise<void>((resolve, reject) => {
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
    const openDatabase = () => new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open("drowse-hosted-artifacts", 1);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });

    const artifacts = await import(moduleUrl);
    const repository = new artifacts.BrowserDrowseArchiveRepository();
    await repository.install(new Blob([decode(first)]));
    await repository.install(new Blob([decode(second)]));
    await repository.close();

    const database = await openDatabase();
    const write = database.transaction(["transactions", "installs"], "readwrite");
    const transactions = write.objectStore("transactions");
    const installs = write.objectStore("installs");
    const transactionRows = await requestValue<any[]>(transactions.getAll());
    const installRows = await requestValue<any[]>(installs.getAll());
    const malformedTransaction = transactionRows.find(
      (record) => record.primary === "manifolds/local/first",
    );
    const malformedInstall = installRows.find(
      (record) => record.id === "manifolds/local/second",
    );
    if (!malformedTransaction || !malformedInstall) throw new Error("Artifact fixtures are missing");
    transactions.put({ ...malformedTransaction, primary: 5 });
    installs.delete("manifolds/local/first");
    installs.put({
      id: 5,
      transactionId: malformedTransaction.id,
      installedAt: malformedTransaction.committedAt,
    });
    installs.put({ ...malformedInstall, installedAt: "yesterday" });
    await transactionDone(write);
    database.close();

    const reopened = new artifacts.BrowserDrowseArchiveRepository();
    const installed = await reopened.list();
    await reopened.close();

    const scrubbedDatabase = await openDatabase();
    const read = scrubbedDatabase.transaction(["transactions", "installs"], "readonly");
    const transactionCount = await requestValue(read.objectStore("transactions").count());
    const installCount = await requestValue(read.objectStore("installs").count());
    scrubbedDatabase.close();
    const root = await navigator.storage.getDirectory();
    const drowse = await root.getDirectoryHandle("drowse");
    const artifactsDirectory = await drowse.getDirectoryHandle("artifacts");
    const stages = await artifactsDirectory.getDirectoryHandle("stages");
    let stageCount = 0;
    for await (const _entry of stages.entries()) stageCount += 1;
    return { installed, transactionCount, installCount, stageCount };
  }, {
    moduleUrl,
    first: Buffer.from(firstArchive).toString("base64"),
    second: Buffer.from(secondArchive).toString("base64"),
  });
  expect(recovered).toEqual({
    installed: [],
    transactionCount: 0,
    installCount: 0,
    stageCount: 0,
  });
});

test("a second repository waits for a live streamed install", async ({ page }) => {
  const validArchive = archiveFixture([["calm"], ["alert"]]);
  await page.goto(`${devUrl}/app?fixture=1`);
  const result = await page.evaluate(async ({ moduleUrl, valid }) => {
    const decode = (base64: string) =>
      Uint8Array.from(atob(base64), (character) => character.charCodeAt(0));
    const artifacts = await import(moduleUrl);
    const firstRepository = new artifacts.BrowserDrowseArchiveRepository();
    const stage = firstRepository.createStage(false);
    let unblock!: () => void;
    const blocked = new Promise<void>((resolve) => {
      unblock = resolve;
    });
    let reachedBarrier!: () => void;
    const barrier = new Promise<void>((resolve) => {
      reachedBarrier = resolve;
    });
    let firstFinish = true;
    const heldStage = {
      begin: (pack: any) => stage.begin(pack),
      write: (path: string, chunk: Uint8Array) => stage.write(path, chunk),
      finish: async (path: string) => {
        await stage.finish(path);
        if (firstFinish) {
          firstFinish = false;
          reachedBarrier();
          await blocked;
        }
      },
      commit: (pack: any) => stage.commit(pack),
      rollback: (error: unknown) => stage.rollback(error),
    };
    const installation = artifacts.validateDrowseArchive(new Blob([decode(valid)]), {
      stage: heldStage,
    });
    await barrier;

    const secondRepository = new artifacts.BrowserDrowseArchiveRepository();
    let secondState = "pending";
    const secondInitialization = secondRepository.initialize().then(
      () => { secondState = "ready"; },
      (error: unknown) => {
        secondState = error instanceof Error ? `failed: ${error.message}` : "failed";
      },
    );
    let pendingLock = false;
    for (let attempt = 0; attempt < 100; attempt += 1) {
      const snapshot = await navigator.locks.query();
      pendingLock = snapshot.pending?.some((lock) => lock.name === "drowse-artifact-repository-v1") ?? false;
      if (pendingLock || secondState !== "pending") break;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const stateWhileHeld = secondState;
    unblock();
    await installation;
    await secondInitialization;
    const firstInstalled = await firstRepository.list();
    const secondInstalled = await secondRepository.list();
    await firstRepository.close();
    await secondRepository.close();
    return { pendingLock, stateWhileHeld, secondState, firstInstalled, secondInstalled };
  }, {
    moduleUrl,
    valid: Buffer.from(validArchive).toString("base64"),
  });
  expect(result.pendingLock).toBe(true);
  expect(result.stateWhileHeld).toBe("pending");
  expect(result.secondState).toBe("ready");
  expect(result.firstInstalled).toHaveLength(1);
  expect(result.secondInstalled).toEqual(result.firstInstalled);
});

test("browser authoring service persists exact template-backed closures", async ({ page }) => {
  await page.goto(`${devUrl}/app?fixture=1`);
  const result = await page.evaluate(async ({ moduleUrl }) => {
    const artifacts = await import(moduleUrl);
    const call = (service: any, serviceName: string, method: string, args: unknown[] = []) =>
      service.request({ service: serviceName, method, args }, () => {});
    const service = new artifacts.BrowserDrowseArchiveService();
    await service.clear();
    await call(service, "templates", "create", [{
      namespace: "local",
      name: "weekday",
      slot: "[DAY]",
      values: ["Monday", "Tuesday"],
      contexts: [{
        turns: [{ role: "user", content: "What day is it?" }],
        assistant: "Today is [DAY].",
      }],
    }]);
    await call(service, "manifolds", "createFromTemplate", [{
      namespace: "local",
      name: "weekday_axis",
      template_ref: "local/weekday",
      fit_mode: "pca",
    }]);
    await service.close();

    const reopened = new artifacts.BrowserDrowseArchiveService();
    const templates = await call(reopened, "templates", "list");
    const manifolds = await call(reopened, "manifolds", "list");
    const detail = await call(reopened, "manifolds", "get", ["local", "weekday_axis"]);
    const exported = await call(reopened, "manifolds", "drowseArchiveExport", [
      "manifolds/local/weekday_axis",
    ]);
    const verified = await artifacts.validateDrowseArchive(exported);
    await reopened.clear();
    const afterClear = {
      templates: await call(reopened, "templates", "list"),
      manifolds: await call(reopened, "manifolds", "list"),
    };
    await reopened.close();
    return {
      templates,
      manifolds,
      detail,
      verifiedTemplate: verified.manifest.template,
      verifiedCorpora: detail.nodes.map((node: any) => node.statements),
      afterClear,
    };
  }, { moduleUrl });

  expect(result.templates.templates).toHaveLength(1);
  expect(result.manifolds.manifolds).toHaveLength(1);
  expect(result.detail.template_ref).toBe("local/weekday");
  expect(result.verifiedTemplate).toBe("templates/local/weekday");
  expect(result.verifiedCorpora).toEqual([
    ["Today is Monday."],
    ["Today is Tuesday."],
  ]);
  expect(result.afterClear).toEqual({ templates: { templates: [] }, manifolds: { manifolds: [] } });
});

test("browser discover merges commit as one verified local artifact", async ({ page }) => {
  await page.goto(`${devUrl}/app?fixture=1`);
  const result = await page.evaluate(async ({ moduleUrl }) => {
    const artifacts = await import(moduleUrl);
    const call = (service: any, method: string, args: unknown[] = []) =>
      service.request({ service: "manifolds", method, args }, () => {});
    const service = new artifacts.BrowserDrowseArchiveService();
    await service.clear();
    await call(service, "createDiscover", [{
      namespace: "local",
      name: "left",
      fit_mode: "pca",
      hyperparams: { max_dim: 2 },
      nodes: [
        { label: "cold", statements: ["cold"], role: "critic" },
        { label: "hot", statements: ["hot"] },
      ],
    }]);
    await call(service, "createDiscover", [{
      namespace: "local",
      name: "right",
      fit_mode: "pca",
      nodes: [
        { label: "dry", statements: ["dry"] },
        { label: "wet", statements: ["wet"] },
      ],
    }]);
    const merged = await call(service, "merge", [{
      name: "combined",
      sources: [
        { namespace: "local", name: "left" },
        { namespace: "local", name: "right" },
      ],
    }]);
    await service.close();

    const reopened = new artifacts.BrowserDrowseArchiveService();
    const detail = await call(reopened, "get", ["local", "combined"]);
    const exported = await call(reopened, "drowseArchiveExport", [
      "manifolds/local/combined",
    ]);
    const verified = await artifacts.validateDrowseArchive(exported);
    await reopened.clear();
    await reopened.close();
    return {
      mergedLabels: merged.node_labels,
      mergedHyperparams: merged.hyperparams,
      detailLabels: detail.node_labels,
      firstRole: detail.nodes[0].role,
      primary: verified.manifest.primary,
      template: verified.manifest.template,
      fittedCount: verified.fittedArtifacts.length,
    };
  }, { moduleUrl });

  expect(result).toEqual({
    mergedLabels: ["cold", "hot", "dry", "wet"],
    mergedHyperparams: { max_dim: 2 },
    detailLabels: ["cold", "hot", "dry", "wet"],
    firstRole: "critic",
    primary: "manifolds/local/combined",
    template: null,
    fittedCount: 0,
  });
});

test("Hugging Face packs stream through OPFS and bind immutable provenance", async ({ page }) => {
  await page.goto(`${devUrl}/app?fixture=1`);
  const result = await page.evaluate(async ({ moduleUrl }) => {
    const artifacts = await import(moduleUrl);
    const repository = "fixture/portable";
    const revision = "b".repeat(40);
    const primary = "manifolds/local/portable";
    const provenance = {
      uri: `hf://${repository}@${revision}`,
      repository,
      revision,
    };
    const manifold = {
      format_version: 10,
      name: "portable",
      description: "HF browser fixture",
      fit_mode: "pca",
      hyperparams: { max_dim: 1 },
      nodes: [
        { label: "cold", role: null, kind: null },
        { label: "hot", role: null, kind: null },
      ],
      files: {},
      source: provenance.uri,
      tags: ["fixture"],
      template_ref: null,
    };
    const archive = await artifacts.buildDrowseArchive({
      primary,
      template: null,
      producerVersion: "browser-e2e",
      source: provenance,
      files: {
        [`${primary}/manifold.json`]: artifacts.jsonBytes(manifold),
        [`${primary}/nodes/00_cold.json`]: artifacts.jsonBytes(["cold"]),
        [`${primary}/nodes/01_hot.json`]: artifacts.jsonBytes(["hot"]),
      },
    });
    const archiveBytes = new Uint8Array(await archive.arrayBuffer());
    const jsonResponse = (value: unknown, url: string) => {
      const response = new Response(JSON.stringify(value), {
        headers: { "Content-Type": "application/json" },
      });
      Object.defineProperty(response, "url", { value: url });
      return response;
    };
    const fakeFetch = async (input: RequestInfo | URL) => {
      const url = new URL(String(input));
      if (url.pathname === "/api/models") {
        return jsonResponse([{
          id: repository,
          sha: revision,
          tags: ["drowse-manifold"],
          siblings: [
            { rfilename: "manifold.json", size: JSON.stringify(manifold).length },
            { rfilename: "portable.drowse", size: archiveBytes.byteLength },
          ],
        }], url.href);
      }
      if (url.pathname.startsWith("/api/models/")) {
        return jsonResponse({
          id: repository,
          sha: revision,
          tags: ["drowse-manifold"],
          siblings: [
            { rfilename: "manifold.json", size: JSON.stringify(manifold).length },
            { rfilename: "portable.drowse", size: archiveBytes.byteLength },
          ],
        }, url.href);
      }
      if (url.pathname.endsWith("/manifold.json")) {
        return jsonResponse(manifold, url.href);
      }
      if (url.pathname.endsWith("/portable.drowse")) {
        const response = new Response(archiveBytes.slice(), {
          headers: { "Content-Length": String(archiveBytes.byteLength) },
        });
        Object.defineProperty(response, "url", { value: url.href });
        return response;
      }
      throw new Error(`Unexpected fixture request ${url}`);
    };
    const huggingFace = new artifacts.BrowserHfManifoldClient({ fetchImpl: fakeFetch });
    const service = new artifacts.BrowserDrowseArchiveService(
      new artifacts.BrowserDrowseArchiveRepository(),
      new artifacts.BrowserTemplateRepository(),
      "browser-e2e",
      huggingFace,
    );
    await service.clear();
    const call = (method: string, args: unknown[] = [], onProgress = () => {}) =>
      service.request({ service: "manifolds", method, args }, onProgress);
    const search = await call("search", ["portable", 4]);
    const progress: unknown[] = [];
    const installed = await call(
      "install",
      [{ target: `${repository}@${revision}` }],
      (event: unknown) => progress.push(event),
    );
    const detail = await call("get", ["local", "portable"]);
    const root = await navigator.storage.getDirectory();
    const drowse = await root.getDirectoryHandle("drowse");
    const artifactRoot = await drowse.getDirectoryHandle("artifacts");
    const downloads = await artifactRoot.getDirectoryHandle("hf-downloads");
    const temporaryFiles: string[] = [];
    for await (const name of downloads.keys()) temporaryFiles.push(name);

    const localArchive = await artifacts.buildDrowseArchive({
      primary: "manifolds/local/wrong_source",
      template: null,
      producerVersion: "browser-e2e",
      source: { uri: "local", repository: null, revision: null },
      files: {
        "manifolds/local/wrong_source/manifold.json": artifacts.jsonBytes({
          ...manifold,
          name: "wrong_source",
          source: "local",
        }),
        "manifolds/local/wrong_source/nodes/00_cold.json": artifacts.jsonBytes(["cold"]),
        "manifolds/local/wrong_source/nodes/01_hot.json": artifacts.jsonBytes(["hot"]),
      },
    });
    let mismatch = "";
    try {
      await new artifacts.BrowserDrowseArchiveRepository().install(localArchive, {
        expectedSource: { repository, revision },
      });
    } catch (error) {
      mismatch = error instanceof Error ? error.message : String(error);
    }
    await service.clear();
    await service.close();
    return {
      search,
      installedArchiveSource: installed.archive_source,
      labels: detail.node_labels,
      phases: progress.map((entry: any) => entry.data.phase),
      temporaryFiles,
      mismatch,
    };
  }, { moduleUrl });

  expect(result.search.results).toHaveLength(1);
  expect(result.search.results[0]).toMatchObject({
    repository: "fixture/portable",
    revision: "b".repeat(40),
    browser_compatible: true,
  });
  expect(result.installedArchiveSource).toEqual({
    uri: `hf://fixture/portable@${"b".repeat(40)}`,
    repository: "fixture/portable",
    revision: "b".repeat(40),
  });
  expect(result.labels).toEqual(["cold", "hot"]);
  expect(result.phases).toContain("downloading");
  expect(result.phases).toContain("committing");
  expect(result.temporaryFiles).toEqual([]);
  expect(result.mismatch).toContain("provenance does not match");
});

test("shared template identities cannot resolve to conflicting installed bytes", async ({ page }) => {
  await page.goto(`${devUrl}/app?fixture=1`);
  const result = await page.evaluate(async ({ moduleUrl }) => {
    const artifacts = await import(moduleUrl);
    const makePack = async (name: string, assistant: string) => {
      const primary = `manifolds/local/${name}`;
      const template = "templates/local/shared";
      const values = ["Calm", "Alert"];
      const labels = ["calm", "alert"];
      const templatePayload = {
        format_version: 2,
        name: "shared",
        slot: "[TONE]",
        values,
        contexts: [{
          turns: [{ role: "user", content: "How should I answer?" }],
          assistant,
        }],
        description: "shared fixture",
        source: "local",
        tags: [],
      };
      const manifold = {
        format_version: 10,
        name,
        description: "conflict fixture",
        fit_mode: "pca",
        hyperparams: {},
        nodes: labels.map((label) => ({ label, role: null, kind: null })),
        files: {},
        source: "local",
        tags: [],
        template_ref: "local/shared",
      };
      return artifacts.buildDrowseArchive({
        primary,
        template,
        producerVersion: "browser-test",
        source: { uri: "local", repository: null, revision: null },
        files: {
          [`${primary}/manifold.json`]: artifacts.jsonBytes(manifold),
          [`${primary}/nodes/00_calm.json`]: artifacts.jsonBytes([
            assistant.replace("[TONE]", values[0]),
          ]),
          [`${primary}/nodes/01_alert.json`]: artifacts.jsonBytes([
            assistant.replace("[TONE]", values[1]),
          ]),
          [`${template}/template.json`]: artifacts.jsonBytes(templatePayload),
        },
      });
    };
    const repository = new artifacts.BrowserDrowseArchiveRepository();
    await repository.clear();
    await repository.install(await makePack("first", "Answer [TONE]."));
    let conflict: string | null = null;
    try {
      await repository.install(await makePack("second", "Use a [TONE] tone."));
    } catch (error) {
      conflict = error instanceof Error ? error.message : String(error);
    }
    const installed = await repository.list();
    await repository.clear();
    await repository.close();
    return { conflict, installed: installed.map((item) => item.name) };
  }, { moduleUrl });

  expect(result.conflict).toContain("conflicts with an installed Drowse manifold pack");
  expect(result.installed).toEqual(["first"]);
});

test("template storage scrubs malformed IndexedDB records on reopen", async ({ page }) => {
  await page.goto(`${devUrl}/app?fixture=1`);
  const result = await page.evaluate(async ({ moduleUrl }) => {
    const artifacts = await import(moduleUrl);
    const initial = new artifacts.BrowserTemplateRepository();
    await initial.clear();
    await initial.close();
    const database = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open("drowse-hosted-authoring", 1);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    const write = database.transaction("templates", "readwrite");
    write.objectStore("templates").put({
      id: "local/corrupt",
      namespace: "local",
      name: "corrupt",
      payload: {},
      installedAt: 1,
      unexpected: true,
    });
    await new Promise<void>((resolve, reject) => {
      write.oncomplete = () => resolve();
      write.onerror = () => reject(write.error);
    });
    database.close();

    const reopened = new artifacts.BrowserTemplateRepository();
    const listed = await reopened.list();
    await reopened.close();
    const check = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open("drowse-hosted-authoring", 1);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    const read = check.transaction("templates", "readonly").objectStore("templates").getAllKeys();
    const keys = await new Promise<IDBValidKey[]>((resolve, reject) => {
      read.onsuccess = () => resolve(read.result);
      read.onerror = () => reject(read.error);
    });
    check.close();
    return { listed, keys };
  }, { moduleUrl });

  expect(result).toEqual({ listed: [], keys: [] });
});

function archiveFixture(corpora: string[][], name = "demo"): Uint8Array {
  const encode = (value: unknown) => new TextEncoder().encode(`${JSON.stringify(value)}\n`);
  const manifold = {
    format_version: 10,
    name,
    description: "browser repository fixture",
    fit_mode: "pca",
    hyperparams: {},
    nodes: ["calm", "alert"].map((label) => ({ label, role: null, kind: "abstract" })),
    files: {},
    source: "local",
    tags: [],
    template_ref: null,
  };
  const primary = `manifolds/local/${name}`;
  const payloads: Record<string, Uint8Array> = {
    [`${primary}/manifold.json`]: encode(manifold),
    [`${primary}/nodes/00_calm.json`]: encode(corpora[0]),
    [`${primary}/nodes/01_alert.json`]: encode(corpora[1]),
  };
  const files = Object.fromEntries(Object.entries(payloads).map(([path, bytes]) => [path, {
    size: bytes.byteLength,
    sha256: createHash("sha256").update(bytes).digest("hex"),
  }]));
  const manifest = encode({
    format_version: 1,
    kind: "drowse-manifold",
    producer: { name: "drowse", version: "test" },
    primary,
    template: null,
    source: { uri: "local", repository: null, revision: null },
    files,
  });
  return zipSync({ "pack.json": manifest, ...payloads }, { level: 0 });
}
