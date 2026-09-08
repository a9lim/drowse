import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

test("browser harness keeps failed-request diagnostics local and continues serving", { timeout: 15_000 }, async (t) => {
  const directory = await mkdtemp(join(tmpdir(), "drowse-http-errors-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  await writeFile(join(directory, "mlc-chat-config.json"), JSON.stringify({
    context_window_size: 128,
    conv_template: { stop_token_ids: [2] },
  }));
  await writeFile(join(directory, "drowse-build.json"), JSON.stringify({
    source: { revision: "a".repeat(40) },
    files: [],
    architecture: "llama",
    quantization: "q4f16_1",
  }));
  const library = join(directory, "model.wasm");
  await writeFile(library, Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]));
  const child = spawn(process.execPath, [
    new URL("./base-model-browser-harness.mjs", import.meta.url).pathname,
    directory, library, "0",
  ], { stdio: ["ignore", "pipe", "pipe"], signal: t.signal });
  let stdout = "", stderr = "";
  child.stdout.on("data", (chunk) => { stdout += chunk; });
  child.stderr.on("data", (chunk) => { stderr += chunk; });
  t.after(async () => {
    if (child.exitCode === null && child.signalCode === null) {
      const exited = once(child, "exit");
      child.kill();
      await exited;
    }
    const reports = /Reports: (.+)\n/.exec(stdout)?.[1];
    if (reports) await rm(reports, { recursive: true, force: true });
  });
  const origin = await new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", () => reject(new Error(`Harness exited before listening: ${stderr}`)));
    child.stdout.on("data", () => {
      const match = /Local candidate harness: (http:\/\/127\.0\.0\.1:\d+)/.exec(stdout);
      if (match) resolve(match[1]);
    });
  });
  const manifest = await fetch(`${origin}/manifest`);
  assert.equal(manifest.status, 200);
  assert.equal((await manifest.json()).contextTokens, 128);

  const malformed = await fetch(`${origin}/report`, { method: "POST", body: "private-report-content" });
  assert.equal(malformed.status, 500);
  assert.equal(await malformed.text(), "Internal server error");

  await rm(library);
  const missing = await fetch(`${origin}/model-lib.wasm`);
  assert.equal(missing.status, 500);
  assert.equal(await missing.text(), "Internal server error");
  assert.match(stderr, /SyntaxError/);
  assert.ok(stderr.includes(library));

  const saved = await fetch(`${origin}/report`, {
    method: "POST", body: JSON.stringify({ steps: ["still working"] }),
  });
  assert.equal(saved.status, 200);
  const { file } = await saved.json();
  assert.deepEqual(JSON.parse(await readFile(file, "utf8")).steps, ["still working"]);
});
