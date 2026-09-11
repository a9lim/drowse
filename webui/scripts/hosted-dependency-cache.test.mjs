import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { createServer } from "vite";

const root = resolve(import.meta.dirname, "..");
const cacheDir = await mkdtemp(resolve(tmpdir(), "drowse-dependency-cache-"));
const server = await createServer({
  configFile: resolve(root, "vite.hosted.config.ts"),
  cacheDir,
  logLevel: "silent",
  server: { host: "127.0.0.1", port: 0, hmr: false, watch: null },
});
try {
  await server.listen();
  const origin = server.resolvedUrls.local[0];
  const workerUrl = `${origin}@fs${root}/src/hosted/runtime/browser.worker.ts?worker_file&type=module`;
  const response = await fetch(workerUrl);
  assert.equal(response.status, 200);
  const source = await response.text();
  const runtimeImport = source.match(/import\("([^"\n]+\/node_modules\/@drowse\/web-llm\/lib\/index\.js[^"\n]*)"\)/u);
  assert.ok(runtimeImport, "the worker must import the current vendored bundle, not a stale optimized copy");
  const runtimeResponse = await fetch(new URL(runtimeImport[1], origin));
  assert.equal(runtimeResponse.status, 200);
  const runtime = await runtimeResponse.text();
  assert.match(runtime, /Number\.isSafeInteger\(config\.top_logprobs\)/u);
  assert.doesNotMatch(runtime, /config\.top_logprobs > 5/u);
  process.stdout.write("hosted worker serves the current full-vocabulary sampler without an optimized dependency cache\n");
} finally {
  await server.close();
  await rm(cacheDir, { recursive: true, force: true });
}
