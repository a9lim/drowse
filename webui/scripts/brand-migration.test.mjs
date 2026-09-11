import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const fixtureWorker = await readFile(new URL("../src/hosted/runtime/fixtureBrowser.worker.ts", import.meta.url), "utf8");
const fixtureTests = await readFile(new URL("../e2e/hosted.spec.ts", import.meta.url), "utf8");
for (const name of ["model", "core", "jlens", "sae"]) {
  const bytes = await readFile(new URL(`../src/hosted/runtime/fixtures/${name}.fixture.bin`, import.meta.url));
  const digest = createHash("sha256").update(bytes).digest("hex");
  const declared = fixtureWorker.match(new RegExp(`const ${name}Sha256 = "([a-f0-9]+)"`))?.[1];
  assert.equal(declared, digest, `${name} fixture catalog hash must match its bytes`);
  assert.ok(fixtureTests.includes(`"${digest}"`), `${name} fixture browser expectations must match its bytes`);
}
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const {
    legacyDatabaseNames,
    migrateLegacyRecord,
    migrateLegacyStorageItem,
    removeLegacyStorageItem,
    drowseStorageRoot,
  } = await server.ssrLoadModule("/src/hosted/runtime/brandMigration.ts");

  const values = new Map([["saklas.hosted.theme", "dark"]]);
  const storage = {
    getItem(key) { return values.get(key) ?? null; },
    setItem(key, value) { values.set(key, String(value)); },
    removeItem(key) { values.delete(key); },
  };
  assert.equal(migrateLegacyStorageItem(storage, "drowse.hosted.theme"), "dark");
  assert.equal(values.get("drowse.hosted.theme"), "dark");
  assert.equal(values.get("saklas.hosted.theme"), "dark");

  values.set("drowse.hosted.theme", "light");
  values.set("saklas.hosted.theme", "dark");
  assert.equal(migrateLegacyStorageItem(storage, "drowse.hosted.theme"), "light");
  assert.equal(values.get("saklas.hosted.theme"), "dark");

  values.delete("drowse.hosted.theme");
  values.set("polythetic.hosted.theme", "system");
  assert.equal(migrateLegacyStorageItem(storage, "drowse.hosted.theme"), "system");
  assert.equal(migrateLegacyStorageItem(storage, "unrelated.theme"), null);
  values.set("polythetic:chunk-recovery", "record");
  assert.equal(migrateLegacyStorageItem(storage, "drowse:chunk-recovery"), "record");
  removeLegacyStorageItem(storage, "drowse:chunk-recovery");
  assert.equal(migrateLegacyStorageItem(storage, "drowse:chunk-recovery"), null);
  removeLegacyStorageItem(storage, "drowse.hosted.theme");
  assert.equal(migrateLegacyStorageItem(storage, "drowse.hosted.theme"), null);
  const binary = new Uint8Array([0, 255, 17]);
  const blob = new Blob([binary]);
  const stored = migrateLegacyRecord({ binary, buffer: binary.buffer, blob });
  assert.equal(stored.binary, binary);
  assert.equal(stored.buffer, binary.buffer);
  assert.equal(stored.blob, blob);
  assert.deepEqual(new Uint8Array(await stored.blob.arrayBuffer()), binary);
  assert.deepEqual(legacyDatabaseNames("drowse-conversations"), ["polythetic-conversations", "saklas-conversations"]);
  assert.deepEqual(legacyDatabaseNames("unrelated"), []);
  for (const slug of ["polythetic", "saklas"]) {
    const original = {
      [`${slug}_version`]: "1.0", producer: { name: slug }, kind: `${slug}-manifold`,
      text: slug, name: slug, prompt: `${slug}-manifold`,
      source: `https://example.com/${slug}`, archive: `demo.${slug}`,
      nested: [{ [`${slug}_version`]: "2.0", content: slug }],
    };
    assert.deepEqual(migrateLegacyRecord(original), {
      drowse_version: "1.0", producer: { name: "drowse" }, kind: "drowse-manifold",
      text: slug, name: slug, prompt: `${slug}-manifold`,
      source: `https://example.com/${slug}`, archive: `demo.${slug}`,
      nested: [{ drowse_version: "2.0", content: slug }],
    });
    assert.equal(original.producer.name, slug);
    assert.deepEqual(migrateLegacyRecord({ [`${slug}_version`]: "old", drowse_version: "new" }), { drowse_version: "new" });
  }

  const legacy = {
    async move(name) { this.movedTo = name; },
  };
  const directories = new Map([["saklas", legacy]]);
  const storageRoot = {
    async getDirectoryHandle(name, options = {}) {
      if (directories.has(name)) return directories.get(name);
      if (name === "drowse" && legacy.movedTo === name) {
        directories.set(name, legacy);
        return legacy;
      }
      if (options.create) {
        const created = {};
        directories.set(name, created);
        return created;
      }
      throw new DOMException("missing", "NotFoundError");
    },
  };
  assert.equal(await drowseStorageRoot(storageRoot), legacy);
  assert.equal(legacy.movedTo, "drowse");
  const current = {};
  directories.set("drowse", current);
  assert.equal(await drowseStorageRoot(storageRoot), current);
  directories.delete("drowse");
  legacy.movedTo = undefined;
  directories.delete("saklas");
  directories.set("polythetic", current);
  assert.equal(await drowseStorageRoot(storageRoot), current);

  console.log("brand migration regression tests passed");
} finally {
  await server.close();
}
