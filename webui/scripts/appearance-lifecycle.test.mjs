import assert from "node:assert/strict";
import { createServer } from "vite";

const server = await createServer({ appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
const originals = Object.fromEntries(["indexedDB", "document", "createImageBitmap"].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
let saved, failWrite = false, closedBitmaps = 0, dimensions;
globalThis.indexedDB = { open() {
  const request = {};
  queueMicrotask(() => {
    request.result = { close() {}, transaction() {
      const transaction = { objectStore() { return {
        get() {
          const read = {};
          queueMicrotask(() => { read.result = saved; read.onsuccess(); transaction.oncomplete(); });
          return read;
        },
        put(value) {
          queueMicrotask(() => {
            if (failWrite) { transaction.error = new Error("quota"); transaction.onabort(); }
            else { saved = structuredClone(value); transaction.oncomplete(); }
          });
        },
      }; } };
      return transaction;
    } };
    request.onsuccess();
  });
  return request;
} };
globalThis.createImageBitmap = async () => ({ width: 4096, height: 2048, close() { closedBitmaps++; } });
globalThis.document = { createElement() {
  const canvas = { getContext: () => ({ drawImage() { dimensions = [canvas.width, canvas.height]; } }), toBlob(callback) { callback(new Blob(["processed image"], { type: "image/webp" })); } };
  return canvas;
} };

try {
  const store = await server.ssrLoadModule("/src/lib/stores/appearance.svelte.ts");
  await store.loadAppearance();
  assert.equal(store.appearanceState.ready, true);
  assert.equal(store.appearanceState.url, "");
  await store.uploadBackground(new File(["source image"], "test.png", { type: "image/png" }));
  assert.equal(store.appearanceState.name, "test.png");
  assert.match(store.appearanceState.url, /^blob:/);
  assert.deepEqual(dimensions, [2048, 1024]);
  assert.equal(closedBitmaps, 1);
  const imageUrl = store.appearanceState.url;
  await store.updateBackground({ effect: "pixel", pixelSize: 8 });
  assert.equal(store.appearanceState.url, imageUrl, "Settings do not allocate another image URL");
  assert.equal(saved.effect, "pixel");
  await store.uploadBackground(new File(["bad"], "test.svg", { type: "image/svg+xml" }));
  assert.match(store.appearanceState.error, /Choose a PNG/);
  assert.equal(store.appearanceState.url, imageUrl, "Invalid files preserve the current image");
  failWrite = true;
  await store.updateBackground({ effect: "original" });
  assert.match(store.appearanceState.error, /could not be saved/);
  assert.equal(store.appearanceState.effect, "pixel", "Failed writes do not change the visible setting");
  assert.equal(store.appearanceState.busy, false);
  await store.removeBackground();
  assert.equal(store.appearanceState.url, imageUrl, "Failed removal preserves the current image");
  failWrite = false;
  await store.removeBackground();
  assert.equal(store.appearanceState.url, "");
  assert.equal(saved.image, null);
  assert.equal(store.appearanceState.busy, false);
  console.log("appearance lifecycle: local save, image resizing, rejected files, settings reuse, failed writes, removal, and bitmap cleanup passed");
} finally {
  for (const [key, descriptor] of Object.entries(originals)) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
  await server.close();
}
