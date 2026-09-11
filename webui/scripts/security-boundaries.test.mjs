import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import ts from "typescript";
import { clearDrowseLocalData, registerDrowseLocalDataClearer } from "../src/lib/runtime/localData.ts";
import * as auth from "../src/lib/runtime/http-auth.ts";

const { outputText } = ts.transpileModule(await readFile("src/lib/api.ts", "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
});
const api = {};
const imports = { "./runtime/http-auth": auth, "./runtime/errors": {} };
new Function("require", "exports", outputText)(name => {
  assert.ok(imports[name], `unexpected dependency: ${name}`);
  return imports[name];
}, api);
const originalSocket = globalThis.WebSocket;
const originalLocation = globalThis.location;
const calls = [];
try {
  globalThis.location = { protocol: "https:", host: "app.example" };
  globalThis.WebSocket = class {
    constructor(url, protocols) { calls.push({ url, protocols }); }
  };
  for (const key of [null, "synthetic +/ secret", "synthetic-unicode-\u03bb"]) {
    api.setApiKey(key);
    api.connectWs();
    const { url, protocols } = calls.at(-1);
    assert.equal(url, "wss://app.example/drowse/v1/sessions/default/stream");
    assert.equal(protocols[0], "drowse.v1");
    assert.equal(protocols.length, key ? 2 : 1);
    if (key) {
      const encoded = protocols[1].slice("drowse.auth.".length);
      assert.match(encoded, /^[A-Za-z0-9_-]+$/);
      assert.equal(Buffer.from(encoded, "base64url").toString("utf8"), key);
    }
  }
} finally {
  api.setApiKey(null);
  globalThis.WebSocket = originalSocket;
  globalThis.location = originalLocation;
}

const data = new Map([["drowse.chat", "synthetic-private-text"], ["saklas.chat", "legacy"], ["polythetic:chunk-recovery", "old"], ["drowse:chunk-recovery", "current"], ["unrelated", "keep"]]);
const storage = {
  get length() { return data.size; },
  key(index) { return [...data.keys()][index] ?? null; },
  removeItem(key) {
    if (key === "drowse.chat") throw new Error("storage write denied");
    data.delete(key);
  },
};
let stopped = false;
const unregister = registerDrowseLocalDataClearer(() => { stopped = true; });
try {
  assert.throws(() => clearDrowseLocalData(storage), /could not clear/i);
  assert.ok(stopped);
  assert.ok(!data.has("saklas.chat"));
  assert.ok(!data.has("polythetic:chunk-recovery"));
  assert.ok(!data.has("drowse:chunk-recovery"));
  assert.equal(data.get("unrelated"), "keep");
  assert.equal(data.get("drowse.chat"), "synthetic-private-text");
  storage.removeItem = key => { data.delete(key); };
  clearDrowseLocalData(storage);
  assert.deepEqual([...data.keys()], ["unrelated"]);
} finally {
  unregister();
}
console.log("Security boundaries: URL-free browser credentials and explicit, retryable local-data deletion failures passed");
