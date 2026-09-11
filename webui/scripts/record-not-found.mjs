import { createHash, randomInt } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { resolve } from "node:path";
import { pipeline } from "node:stream/promises";
import { createServer as createViteServer } from "vite";
import { grantReleaseToolStorageQuota, launchReleaseToolContext } from "./release-tool-storage-quota.mjs";

const [directory, output] = process.argv.slice(2);
if (!directory || !output) throw new Error("Usage: node scripts/record-not-found.mjs <artifact closure directory> <output.json>");
const root = resolve(import.meta.dirname, "..");
const manifest = JSON.parse(await readFile(resolve(directory, "hosted-artifacts.json"), "utf8"));
if (manifest.source.repository !== "unsloth/gemma-3-4b-it") throw new Error("The recording requires Gemma 3 4B");
const seeds = Array.from({ length: 50 }, () => randomInt(0, 0x7fffffff));
const sets = {};
for (const set of ["model"]) {
  sets[set] = manifest.files.map(file => ({ ...file, ...(set === "core" ? { role: "core_pack" } : {}),
    revision: manifest.source.revision,
    url: `https://huggingface.co/${manifest.source.repository}/resolve/${manifest.source.revision}/${file.path}`,
  }));
  for (const file of sets[set]) {
    if (file.path.split("/").some(part => !part || part === "." || part === ".." || part.includes("\\"))) throw new Error("Unsafe artifact path");
    const hash = createHash("sha256");
    let bytes = 0;
    for await (const chunk of createReadStream(resolve(directory, file.path))) { hash.update(chunk); bytes += chunk.length; }
    if (bytes !== file.bytes || hash.digest("hex") !== file.sha256) throw new Error(`Invalid artifact ${set}/${file.path}`);
  }
}
const artifactHash = role => {
  const files = manifest.files.filter(file => file.role === role);
  if (files.length !== 1) throw new Error(`Expected one ${role} artifact`);
  return files[0].sha256;
};
const identity = {
  sourceModel: manifest.source.repository, sourceRevision: manifest.source.revision,
  convertedManifestSha256: artifactHash("converted_manifest"), quantization: manifest.quantization,
  tokenizerSha256: artifactHash("tokenizer"), chatTemplateSha256: artifactHash("chat_template"),
  modelLibrarySha256: artifactHash("model_library"), runtimeAbi: manifest.runtimeAbi, hookAbi: manifest.hookAbi,
  hiddenSize: manifest.hiddenSize, layerMap: manifest.layerMap,
};
const webllmPath = resolve(root, "node_modules/@drowse/web-llm/lib/index.js");
const webllmSha256 = createHash("sha256").update(await readFile(webllmPath)).digest("hex");
const vite = await createViteServer({ root, configFile: false, appType: "custom", logLevel: "error", server: { middlewareMode: true, hmr: false, watch: null } });
const server = createServer(async (request, response) => {
  response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  response.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  try {
    const url = new URL(request.url, "http://127.0.0.1");
    if (url.pathname === "/") { response.setHeader("Content-Type", "text/html"); response.end("<!doctype html><title>Record Drowse 404 messages</title>"); return; }
    if (url.pathname === "/worker.js") {
      const result = await vite.transformRequest("/scripts/record-not-found.worker.ts");
      response.setHeader("Content-Type", "text/javascript"); response.end(result.code); return;
    }
    if (url.pathname === "/webllm.js") {
      response.setHeader("Content-Type", "text/javascript"); await pipeline(createReadStream(webllmPath), response); return;
    }
    if (url.pathname === "/artifact") {
      const set = url.searchParams.get("set");
      const file = sets[set]?.find(entry => entry.path === url.searchParams.get("path"));
      if (!file) { response.writeHead(404); response.end(); return; }
      response.setHeader("Content-Length", file.bytes);
      await pipeline(createReadStream(resolve(directory, file.path)), response); return;
    }
    vite.middlewares(request, response);
  } catch (error) {
    console.error(error);
    if (response.headersSent) response.destroy();
    else { response.writeHead(500); response.end("Internal server error"); }
  }
});
let browser;
try {
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  browser = await launchReleaseToolContext({ headless: true, channel: "chrome", args: ["--enable-unsafe-webgpu", "--disable-gpu-sandbox", "--use-angle=metal"] });
  const page = browser.page;
  page.setDefaultTimeout(900_000);
  page.on("console", message => console.log(message.text()));
  await page.goto(origin);
  await grantReleaseToolStorageQuota(page, origin, Object.values(sets).flat());
  const result = await page.evaluate(payload => new Promise((resolve, reject) => {
    const worker = new Worker("/worker.js", { type: "module" });
    worker.onerror = event => reject(new Error(event.message));
    worker.onmessage = ({ data }) => {
      if (data.type === "error") reject(new Error(data.message));
      if (data.type === "progress") console.log(JSON.stringify(data));
      if (data.type === "result") resolve(data.value);
    };
    worker.postMessage(payload);
  }), { modelId: "gemma3-4b-instruct", identity, modelFiles: sets.model, seeds });
  const artifacts = Object.fromEntries(Object.entries(sets).map(([name, files]) => [name,
    files.map(({ path, role, bytes, sha256 }) => ({ path, role, bytes, sha256 })),
  ]));
  await writeFile(resolve(output), JSON.stringify({ ...result, webllmSha256,
    manifestSha256: createHash("sha256").update(await readFile(resolve(directory, "hosted-artifacts.json"))).digest("hex"),
    artifactSource: "Locally verified artifact closure; hashes identify this recording, not a hosted-release attestation.",
    artifacts,
  }, null, 2) + "\n");
  console.log(`Saved ${output}`);
} finally {
  await browser?.close();
  await vite.close();
  await new Promise(resolve => server.close(resolve));
}
