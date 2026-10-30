import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";

const { scripts } = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
const visited = new Set();
const scheduled = new Set();
function visit(name) {
  if (visited.has(name)) return;
  assert.equal(typeof scripts[name], "string", `Missing npm script: ${name}`);
  visited.add(name);
  for (const match of scripts[name].matchAll(/\bnpm run ([\w:-]+)/g)) visit(match[1]);
  for (const match of scripts[name].matchAll(/\bnode\s+(?:--test\s+)?\.\/scripts\/([\w.-]+\.test\.mjs)\b/g)) {
    scheduled.add(match[1]);
  }
}
for (const name of ["check", "check:hosted", "test:hosted"]) visit(name);
const tests = (await readdir(new URL(".", import.meta.url))).filter(name => name.endsWith(".test.mjs"));
const missing = tests.filter(name => !scheduled.has(name));
assert.deepEqual(missing, [], `Tests disconnected from the standard CI checks: ${missing.join(", ")}`);
console.log(`Test wiring: all ${tests.length} script test files are reachable from standard CI checks`);
