import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import ts from "typescript";

const source = await readFile("src/lib/stores/persistence.svelte.ts", "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
});
const roots = new Set();
const timers = new Map();
const writes = [];
const storage = new Map();
let clearer;
let currentRoot;
let nextTimer = 1;
const effect = callback => { currentRoot.callbacks.push(callback); callback(); };
effect.root = callback => {
  const root = { callbacks: [] };
  roots.add(root);
  currentRoot = root;
  callback();
  currentRoot = null;
  return () => roots.delete(root);
};
const highlightState = { target: null, compareTarget: null, compareTwo: false };
const sessionState = { info: { model_id: "first-model" } };
const imports = {
  "./loom.svelte": { loomTree: {} },
  "./probes.svelte": { highlightState },
  "./session.svelte": { sessionState },
  "../../hosted/runtime/brandMigration": {
    migrateLegacyStorageItem: (store, key) => store.getItem(key),
    removeLegacyStorageItem: (store, key) => store.removeItem(key),
  },
  "../runtime/localData": { registerDrowseLocalDataClearer: callback => { clearer = callback; } },
  "../tokens": { PROBABILITY_TARGET: "probability", SURPRISE_TARGET: "surprise" },
};
const exports = {};
const localStorage = {
  getItem: key => storage.get(key) ?? null,
  setItem: (key, value) => { storage.set(key, value); writes.push([key, JSON.parse(value)]); },
  removeItem: key => storage.delete(key),
};
new Function("require", "exports", "$effect", "globalThis", "setTimeout", "clearTimeout", outputText)(
  name => { assert.ok(imports[name], `unexpected dependency: ${name}`); return imports[name]; },
  exports, effect, { localStorage },
  callback => { const id = nextTimer++; timers.set(id, callback); return id; },
  id => timers.delete(id),
);
const change = () => { for (const root of roots) for (const callback of root.callbacks) callback(); };
const flush = () => {
  const callbacks = [...timers.values()];
  timers.clear();
  for (const callback of callbacks) callback();
};

for (let index = 0; index < 100; index += 1) {
  const dispose = exports.attachPersistence();
  assert.equal(roots.size, 1);
  assert.equal(timers.size, 0, "mount must not overwrite restored preferences");
  highlightState.target = `probe-${index}`;
  change();
  assert.equal(timers.size, 1);
  dispose();
  assert.equal(roots.size, 0, "unmount releases its effect root");
  assert.equal(timers.size, 0, "unmount clears pending writes");
}
const oldDispose = exports.attachPersistence();
change();
const currentDispose = exports.attachPersistence();
oldDispose();
assert.equal(roots.size, 1, "stale cleanup cannot stop the current owner");
assert.equal(timers.size, 0, "reattachment discards the old model's pending write");
sessionState.info = { model_id: "second-model" };
highlightState.target = "current-probe";
change();
flush();
assert.equal(writes.length, 1);
assert.equal(writes[0][0], "drowse.chat.v4.second-model");
assert.equal(writes[0][1].highlight.target, "current-probe");
change();
clearer();
flush();
assert.equal(writes.length, 1, "clearing local data cancels pending writers");
currentDispose();
currentDispose();
assert.equal(roots.size, 0);
assert.equal(timers.size, 0);

const app = await readFile("src/App.svelte", "utf8");
const bootstrap = await readFile("src/lib/stores/bootstrap.svelte.ts", "utf8");
assert.match(app, /\$effect\(\(\) => \{\s*if \(bootStatus === "ready"\) return attachPersistence\(\);/);
assert.doesNotMatch(bootstrap, /attachPersistence\(/, "async bootstrap cannot create an orphan after unmount");
console.log("Persistence lifecycle: 100 mount/unmount cycles, single-owner replacement, stale cleanup, model keys, and pending-write cleanup passed");
