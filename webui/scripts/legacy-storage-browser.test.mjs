import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root, configFile: false, appType: "custom", logLevel: "error",
  optimizeDeps: { noDiscovery: true },
  server: { host: "127.0.0.1", port: 0, watch: null },
  plugins: [{ name: "legacy-storage-fixture", configureServer(server) {
    server.middlewares.use((req, res, next) => {
      if (req.url !== "/") return next();
      res.setHeader("Content-Type", "text/html");
      res.end("<!doctype html><title>Storage regression</title>");
    });
  } }],
});
let browser;
try {
  await server.listen();
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.goto(server.resolvedUrls.local[0]);
  const initial = await page.evaluate(async () => {
    const { BrowserSessionStateStore } = await import("/src/hosted/runtime/sessionPersistence.ts");
    const { clearOwnedRuntimeStorage } = await import("/src/hosted/runtime/legacyRuntimeStorage.ts");
    const { drowseStorageRoot } = await import("/src/hosted/runtime/brandMigration.ts");
    async function seed(name, value) {
      const database = await new Promise((resolve, reject) => {
        const request = indexedDB.open(name, 2);
        request.onupgradeneeded = () => request.result.createObjectStore("sessions", { keyPath: "modelVariantId" });
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });
      await new Promise((resolve, reject) => {
        const transaction = database.transaction("sessions", "readwrite");
        transaction.objectStore("sessions").put({ modelVariantId: "private-fixture", text: value });
        transaction.oncomplete = resolve;
        transaction.onerror = () => reject(transaction.error);
      });
      database.close();
    }
    await seed("polythetic-hosted-sessions", "private legacy conversation");
    await seed("saklas-hosted-sessions", "second legacy conversation");
    await seed("unrelated-app", "preserve me");
    const root = await navigator.storage.getDirectory();
    for (const name of ["polythetic", "saklas", "unrelated"]) {
      const directory = await root.getDirectoryHandle(name, { create: true });
      const file = await directory.getFileHandle("private.txt", { create: true });
      const writer = await file.createWritable();
      await writer.write(name);
      await writer.close();
    }
    // On browsers without directory move, the app can still own a legacy handle.
    const active = await drowseStorageRoot(root);
    const store = new BrowserSessionStateStore();
    await store.initialize();
    const restored = await store.read("private-fixture");
    await store.clear();
    await clearOwnedRuntimeStorage();
    const writable = await active.getFileHandle("after-clear.txt", { create: true });
    const writer = await writable.createWritable();
    await writer.write("still usable");
    await writer.close();
    await active.removeEntry("after-clear.txt");
    store.close();
    return restored;
  });
  assert.equal(initial.text, "private legacy conversation");
  await page.reload();
  const final = await page.evaluate(async () => {
    const { BrowserSessionStateStore } = await import("/src/hosted/runtime/sessionPersistence.ts");
    const store = new BrowserSessionStateStore();
    await store.initialize();
    const restored = await store.read("private-fixture");
    store.close();
    const databases = (await indexedDB.databases()).map(row => row.name);
    const root = await navigator.storage.getDirectory();
    const files = {};
    for (const name of ["polythetic", "saklas", "unrelated"]) {
      try {
        const directory = await root.getDirectoryHandle(name);
        files[name] = [];
        for await (const entry of directory.values()) files[name].push(entry.name);
      } catch (error) {
        if (error.name !== "NotFoundError") throw error;
        files[name] = [];
      }
    }
    return { restored, databases, files };
  });
  assert.equal(final.restored, undefined);
  assert.ok(!final.databases.includes("polythetic-hosted-sessions"));
  assert.ok(!final.databases.includes("saklas-hosted-sessions"));
  assert.ok(final.databases.includes("unrelated-app"));
  assert.deepEqual(final.files.polythetic, []);
  assert.deepEqual(final.files.saklas, []);
  assert.deepEqual(final.files.unrelated, ["private.txt"]);
  console.log("legacy conversation clear/reload and OPFS reuse passed in Chromium");
} finally {
  await browser?.close();
  await server.close();
}
