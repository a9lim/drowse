import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { basename, resolve } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "playwright";

const { values } = parseArgs({ options: {
  model: { type: "string" },
  webllm: { type: "string", default: "webui/node_modules/@drowse/web-llm/lib/index.js" },
} });
if (!values.model) throw new Error("Use --model DIRECTORY containing model.wasm and mlc-chat-config.json");
const model = resolve(values.model);
const bundle = resolve(values.webllm);
const configBytes = await readFile(resolve(model, "mlc-chat-config.json"));
const config = JSON.parse(configBytes);
const failures = [];
const server = createServer(async (request, response) => {
  response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  response.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  const path = new URL(request.url, "http://localhost").pathname;
  if (path === "/") {
    response.setHeader("Content-Type", "text/html");
    response.end("<!doctype html><title>Sampling limits test</title>");
    return;
  }
  const file = path === "/webllm.js" ? bundle
    : (path.startsWith("/model/") && basename(path) === path.slice(7)) ||
      (path.startsWith("/model/resolve/main/") && basename(path) === path.slice(20))
      ? resolve(model, basename(path)) : null;
  if (!file) { process.stderr.write(`Unexpected artifact request: ${path}\n`); response.writeHead(404).end(); return; }
  try {
    const info = await stat(file);
    response.setHeader("Content-Length", info.size);
    response.setHeader("Content-Type", file.endsWith(".wasm") ? "application/wasm"
      : file.endsWith(".js") ? "text/javascript" : "application/octet-stream");
    createReadStream(file).pipe(response);
  } catch (error) {
    failures.push(String(error));
    process.stderr.write(`${error}\n`);
    response.writeHead(404).end();
  }
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const browser = await chromium.launch({ args: [
  "--enable-unsafe-webgpu", "--use-angle=metal", "--disable-gpu-sandbox",
] });
try {
  const page = await browser.newPage();
  page.on("console", (message) => {
    if (message.type() === "error") failures.push(message.text());
    if (message.text().startsWith("sampling limits:")) process.stderr.write(`${message.text()}\n`);
  });
  page.on("pageerror", (error) => failures.push(String(error)));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  const result = await page.evaluate(async ({ vocabSize }) => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || adapter.info.isFallbackAdapter) throw new Error("A hardware WebGPU adapter is required");
    const { MLCEngine } = await import("/webllm.js");
    const engine = new MLCEngine({
      logLevel: "WARN",
      appConfig: { artifactCache: {
        async fetchWithCache(url, storeType) {
          const response = await fetch(url);
          if (!response.ok) throw new Error(`Artifact request failed: ${url}`);
          return storeType === "json" ? response.json() : response.arrayBuffer();
        },
        async addToCache() {},
        async hasAllKeys() { return false; },
        async deleteInCache() { throw new Error("Test artifacts are read-only"); },
      }, model_list: [{ model_id: "sampling-test",
        model: `${location.origin}/model/`, model_lib: `${location.origin}/model/model.wasm`,
        overrides: { context_window_size: 2048 },
      }] },
    });
    await engine.reload("sampling-test");
    console.info("sampling limits: model loaded");
    const rows = [];
    try {
      for (const topK of [1024, 1025, 4096, vocabSize, vocabSize + 1]) {
        const started = performance.now();
        const response = await engine.completions.create({
          model: "sampling-test", prompt: "The sky is", max_tokens: 1,
          temperature: 3, top_p: 1, top_k: topK, seed: 42,
          logprobs: true, top_logprobs: vocabSize,
        });
        const entries = response.choices[0].logprobs.content[0].top_logprobs;
        rows.push({ topK, count: entries.length,
          probabilityMass: entries.reduce((sum, row) => sum + Math.exp(row.logprob), 0),
          elapsedMs: performance.now() - started });
        console.info(`sampling limits: top K ${topK} returned ${entries.length} alternatives`);
      }
      const long = await engine.completions.create({
        model: "sampling-test", prompt: "Count upwards: 1, 2, 3,", max_tokens: 300,
        temperature: 0, ignore_eos: true,
      });
      console.info(`sampling limits: generated ${long.usage.completion_tokens} tokens`);
      const extremes = [];
      for (const seed of [0, Number.MAX_SAFE_INTEGER]) {
        const sampled = [];
        for (let repeat = 0; repeat < 2; repeat++) {
          const response = await engine.completions.create({
            model: "sampling-test", prompt: "The sky is", max_tokens: 1,
            temperature: 3.4028234663852886e38, top_p: 1, top_k: Number.MAX_SAFE_INTEGER,
            seed, logprobs: true, top_logprobs: 0,
          });
          const token = response.choices[0].logprobs.content[0];
          sampled.push({ tokenId: token.token_id, logprob: token.logprob });
        }
        extremes.push({ seed, sampled });
      }
      const fullContext = await engine.completions.create({
        model: "sampling-test", prompt: " sky".repeat(2024), max_tokens: 4096,
        temperature: 0, ignore_eos: true,
      });
      return { adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture }, rows, extremes,
        long: { usage: long.usage, finishReason: long.choices[0].finish_reason },
        context: { usage: fullContext.usage, finishReason: fullContext.choices[0].finish_reason } };
    } finally { await engine.unload(); }
  }, { vocabSize: config.vocab_size });
  assert.deepEqual(failures, []);
  for (const row of result.rows) {
    assert.equal(row.count, Math.min(row.topK, config.vocab_size), JSON.stringify(row));
    assert.ok(Math.abs(row.probabilityMass - 1) < 0.00001, JSON.stringify(row));
  }
  assert.equal(result.long.usage.completion_tokens, 300);
  assert.equal(result.long.finishReason, "length");
  for (const row of result.extremes) {
    assert.deepEqual(row.sampled[0], row.sampled[1]);
    assert.ok(Number.isFinite(row.sampled[0].logprob));
  }
  // The final sampled token does not require another KV-cache position.
  assert.equal(result.context.usage.total_tokens, 2049);
  assert.ok(result.context.usage.completion_tokens < 4096);
  assert.equal(result.context.finishReason, "length");
  process.stdout.write(`${JSON.stringify({
    configSha256: createHash("sha256").update(configBytes).digest("hex"),
    bundleSha256: createHash("sha256").update(await readFile(bundle)).digest("hex"), ...result,
  }, null, 2)}\n`);
} finally {
  await browser.close();
  await new Promise((done) => server.close(done));
}
