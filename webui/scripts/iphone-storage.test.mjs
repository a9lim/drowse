import assert from "node:assert/strict";
import { createHash, createPublicKey } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, webkit, devices } from "playwright";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const modelBytes = Buffer.alloc(8 * 1024 * 1024 + 37);
for (let i = 0; i < modelBytes.length; i++) modelBytes[i] = i % 251;
const coreBytes = Buffer.from("iphone-storage-core");
const requests = [];
const storageBrowsers = (process.env.DROWSE_STORAGE_BROWSERS ?? "webkit-iphone,chromium").split(",");
assert.ok(storageBrowsers.length > 0 && storageBrowsers.every(name => ["webkit-iphone", "chromium"].includes(name)));
const tlsDirectory = await mkdtemp(join(tmpdir(), "drowse-iphone-tls-"));
const keyPath = join(tlsDirectory, "key.pem");
const certPath = join(tlsDirectory, "cert.pem");
execFileSync("openssl", ["req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", keyPath, "-out", certPath, "-subj", "/CN=127.0.0.1", "-addext", "subjectAltName=IP:127.0.0.1", "-days", "1"], { stdio: "ignore" });
const cert = await readFile(certPath);
const spki = createHash("sha256").update(createPublicKey(cert).export({ type: "spki", format: "der" })).digest("base64");
const server = await createServer({
  root, configFile: false, appType: "custom", logLevel: "error",
  optimizeDeps: { noDiscovery: true, include: ["@noble/hashes/sha2.js", "@noble/hashes/utils.js"] },
  server: { host: "127.0.0.1", port: 0, watch: null, https: { key: await readFile(keyPath), cert } },
  plugins: [{ name: "iphone-storage-fixtures", configureServer(server) {
    server.middlewares.use((req, res, next) => {
      if (req.url === "/") { res.setHeader("Content-Type", "text/html"); res.end("<!doctype html><title>iPhone storage</title>"); return; }
      const data = req.url === "/model.bin" ? modelBytes : req.url === "/core.bin" ? coreBytes : null;
      if (!data) return next();
      const offset = Number(/^bytes=(\d+)-$/.exec(req.headers.range ?? "")?.[1] ?? 0);
      requests.push({ path: req.url, offset });
      res.statusCode = offset ? 206 : 200;
      res.setHeader("Content-Type", "application/octet-stream");
      res.setHeader("Content-Length", data.length - offset);
      res.setHeader("ETag", '"iphone-fixture-v1"');
      if (offset) res.setHeader("Content-Range", `bytes ${offset}-${data.length - 1}/${data.length}`);
      res.end(data.subarray(offset));
    });
  } }],
});

async function writeSlices() {
  const seed = typeof document === "undefined" ? 10 : 0;
  const { BrowserContentStore } = await import("/src/hosted/runtime/contentStore.ts");
  const store = new BrowserContentStore();
  await store.initialize();
  await store.clear();
  const expected = Uint8Array.of(1, 2, 3, 4, 5, 6).map(value => value + seed);
  const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", expected))]
    .map(byte => byte.toString(16).padStart(2, "0")).join("");
  const descriptor = { sha256: hash, expectedBytes: 6, url: "https://example.test/slices", revision: "a".repeat(40), etag: null, partialOwners: [{ id: "slice-test", kind: "model", catalogSequence: 1 }] };
  await store.beginPartial(descriptor);
  if (!(await store.verifiedFile(hash))) {
    const writer = await store.openWrite(hash, 0);
    await writer.write(Uint8Array.of(91, ...expected.slice(0, 3), 92).subarray(1, 4));
    const checkpoint = await writer.checkpoint();
    if (checkpoint !== 3) throw new Error(`Wrong checkpoint: ${checkpoint}`);
    const resumed = await store.openWrite(hash, checkpoint);
    await resumed.write(Uint8Array.of(...expected.slice(3), 93, 94).subarray(0, 3));
    await resumed.checkpoint();
    await store.commitVerified(hash, hash, expected.length);
  }
  const actual = [...new Uint8Array(await (await store.verifiedFile(hash)).arrayBuffer())];
  const { BrowserActivationSpoolFilePort } = await import("/src/hosted/fitting/activationSpool.ts");
  const spool = new BrowserActivationSpoolFilePort();
  await spool.initialize();
  const path = `${hash}.layer-0.f32`;
  await spool.remove(path);
  await spool.append(path, 0, Uint8Array.of(91, ...expected.slice(0, 3), 92).subarray(1, 4));
  await spool.append(path, 3, Uint8Array.of(...expected.slice(3), 93).subarray(0, 3));
  const spooled = [...await spool.read(path, 0, 6)];
  if (JSON.stringify(spooled) !== JSON.stringify(actual)) throw new Error("Activation spool wrote bytes outside the view");
  spool.close();
  return actual;
}

async function transfer({ catalog, pause }) {
  const { BrowserContentStore } = await import("/src/hosted/runtime/contentStore.ts");
  const { VerifiedArtifactDownloader } = await import("/src/hosted/runtime/downloader.ts");
  const store = new BrowserContentStore();
  await store.initialize();
  const downloader = new VerifiedArtifactDownloader(store, { chunkBytes: 64 * 1024, durableCheckpointBytes: 256 * 1024, monotonicNow: (() => { let now = 0; return () => now += 1000; })() });
  let cancelled = false;
  try {
    await downloader.download(catalog, "iphone-fixture", {
      checkQuota: () => true,
      onProgress: progress => {
        if (pause && progress.bytesReceived > 512 * 1024) downloader.cancel();
      },
    });
  } catch (error) {
    if (error.code !== "DOWNLOAD_CANCELLED") throw error;
    cancelled = true;
  }
  const file = catalog.document.models[0].variants[0].files[0];
  const record = await store.inspectObject(file.sha256);
  const installed = await store.listInstalls();
  let digest = null;
  if (!pause) {
    const blob = await store.verifiedFile(file.sha256);
    digest = [...new Uint8Array(await crypto.subtle.digest("SHA-256", await blob.arrayBuffer()))]
      .map(byte => byte.toString(16).padStart(2, "0")).join("");
  }
  return { cancelled, bytes: record.contiguousBytes, installed: installed.map(item => item.id), digest };
}

let context;
let profile;
try {
  await server.listen();
  const origin = server.resolvedUrls.local[0].replace(/\/$/, "");
  const file = (path, bytes, role) => ({ path, url: `${origin}/${path}`, bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex"), revision: "a".repeat(40), role });
  const modelFile = file("model.bin", modelBytes, "weight");
  const coreFile = file("core.bin", coreBytes, "core_pack");
  const catalog = { allowDownloads: true, document: { sequence: 1, models: [{ variants: [{
    id: "iphone-fixture", downloadBytes: modelBytes.length, requiredCorePackBytes: coreBytes.length,
    files: [modelFile], packs: [{ id: "iphone-core", kind: "core", required: true, bytes: coreBytes.length, files: [coreFile] }],
  }] }] } };
  for (const [name, browserType, options] of [["webkit-iphone", webkit, devices["iPhone 13"]], ["chromium", chromium, {}]]) {
    if (!storageBrowsers.includes(name)) continue;
    requests.length = 0;
    profile = await mkdtemp(join(tmpdir(), "drowse-iphone-storage-"));
    context = await browserType.launchPersistentContext(profile, { ...options, headless: true, ignoreHTTPSErrors: true, ...(name === "chromium" ? { args: [`--ignore-certificate-errors-spki-list=${spki}`] } : {}) });
    let page = await context.newPage();
    await page.goto(origin);
    assert.deepEqual(await page.evaluate(writeSlices), [1, 2, 3, 4, 5, 6]);
    const workerResult = await page.evaluate(source => new Promise((resolve, reject) => {
      const url = URL.createObjectURL(new Blob([`(${source})().then(value => postMessage({value}), error => postMessage({error: error.message}))`], { type: "text/javascript" }));
      const worker = new Worker(url, { type: "module" });
      worker.onmessage = ({ data }) => { worker.terminate(); URL.revokeObjectURL(url); data.error ? reject(new Error(data.error)) : resolve(data.value); };
      worker.onerror = event => { worker.terminate(); URL.revokeObjectURL(url); reject(new Error(event.message)); };
    }), writeSlices.toString().replaceAll('import("/', `import("${origin}/`));
    assert.deepEqual(workerResult, [11, 12, 13, 14, 15, 16]);
    const paused = await page.evaluate(transfer, { catalog, pause: true });
    assert.equal(paused.cancelled, true);
    assert.ok(paused.bytes > 0 && paused.bytes < modelBytes.length);
    assert.deepEqual(paused.installed, []);
    await page.evaluate(async hash => {
      const root = await navigator.storage.getDirectory();
      const drowse = await root.getDirectoryHandle("drowse");
      const objects = await drowse.getDirectoryHandle("objects");
      const handle = await objects.getFileHandle(`${hash}.data`);
      const writer = await handle.createWritable({ keepExistingData: true });
      await writer.seek((await handle.getFile()).size);
      await writer.write(Uint8Array.of(251, 252, 253).buffer);
      await writer.close();
    }, modelFile.sha256);
    await context.close();
    context = await browserType.launchPersistentContext(profile, { ...options, headless: true, ignoreHTTPSErrors: true, ...(name === "chromium" ? { args: [`--ignore-certificate-errors-spki-list=${spki}`] } : {}) });
    page = await context.newPage();
    await page.goto(origin);
    const finished = await page.evaluate(transfer, { catalog, pause: false });
    assert.equal(finished.cancelled, false);
    assert.equal(finished.bytes, modelBytes.length);
    assert.equal(finished.digest, modelFile.sha256);
    assert.deepEqual(finished.installed, ["iphone-fixture"]);
    assert.ok(requests.some(request => request.path === "/model.bin" && request.offset === paused.bytes));
    console.log(`${name}: sliced writes, worker access, checkpoints, pause, browser restart, HTTP range resume, and verified installation passed`);
    await context.close(); context = null;
    await rm(profile, { recursive: true, force: true }); profile = null;
  }
} finally {
  await context?.close();
  if (profile) await rm(profile, { recursive: true, force: true });
  await server.close();
  await rm(tlsDirectory, { recursive: true, force: true });
}
