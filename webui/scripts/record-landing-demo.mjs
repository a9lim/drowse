import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { resolve } from "node:path";
import { pipeline } from "node:stream/promises";
import { createServer as createViteServer } from "vite";
import { grantReleaseToolStorageQuota, launchReleaseToolContext } from "./release-tool-storage-quota.mjs";

const [directory, output] = process.argv.slice(2);
if (!directory || !output) throw new Error("Usage: node scripts/record-landing-demo.mjs <artifact closure directory> <output.json>");
const root = resolve(import.meta.dirname, "..");
const lock = JSON.parse(await readFile(resolve(directory, "runtime-lock.json"), "utf8"));
const model = lock.models.find(entry => entry.id === "gemma3-4b-instruct");
if (!model) throw new Error("The recording requires the Gemma 3 4B closure");
const sets = {};
for (const set of ["model", "core"]) {
  const manifest = JSON.parse(await readFile(resolve(directory, set, set === "model" ? "hosted-artifacts.json" : "hosted-pack-artifacts.json"), "utf8"));
  sets[set] = manifest.files.map(file => ({ ...file, ...(set === "core" ? { role: "core_pack" } : {}),
    revision: model.convertedRevision,
    url: `https://huggingface.co/${model.convertedRepository}/resolve/${model.convertedRevision}/${file.path}`,
  }));
  for (const file of sets[set]) {
    if (file.path.split("/").some(part => !part || part === "." || part === ".." || part.includes("\\"))) throw new Error("Unsafe artifact path");
    const hash = createHash("sha256");
    let bytes = 0;
    for await (const chunk of createReadStream(resolve(directory, set, file.path))) { hash.update(chunk); bytes += chunk.length; }
    if (bytes !== file.bytes || hash.digest("hex") !== file.sha256) throw new Error(`Invalid artifact ${set}/${file.path}`);
  }
}
const identity = {
  sourceModel: model.sourceRepository, sourceRevision: model.sourceRevision,
  convertedManifestSha256: model.manifestSha256, quantization: model.quantization,
  tokenizerSha256: model.tokenizerSha256, chatTemplateSha256: model.chatTemplateSha256,
  modelLibrarySha256: model.librarySha256, runtimeAbi: lock.runtimeAbi, hookAbi: lock.hookAbi,
  hiddenSize: model.hiddenSize, layerMap: model.layerMap,
};
const webllmPath = resolve(root, "node_modules/@drowse/web-llm/lib/index.js");
const webllmSha256 = createHash("sha256").update(await readFile(webllmPath)).digest("hex");
const vite = await createViteServer({ root, configFile: false, appType: "custom", logLevel: "error", server: { middlewareMode: true, hmr: false, watch: null } });
const server = createServer(async (request, response) => {
  response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  response.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  try {
    const url = new URL(request.url, "http://127.0.0.1");
    if (url.pathname === "/") { response.setHeader("Content-Type", "text/html"); response.end("<!doctype html><title>Record Drowse example</title>"); return; }
    if (url.pathname === "/worker.js") {
      const result = await vite.transformRequest("/scripts/record-landing-demo.worker.ts");
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
      await pipeline(createReadStream(resolve(directory, set, file.path)), response); return;
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
  }), { modelId: model.id, identity, modelFiles: sets.model, coreFiles: sets.core });
  const artifacts = Object.fromEntries(Object.entries(sets).map(([name, files]) => [name,
    files.map(({ path, role, bytes, sha256 }) => ({ path, role, bytes, sha256 })),
  ]));
  await writeFile(resolve(output), JSON.stringify({ ...result, webllmSha256,
    artifactSource: "Locally verified artifact closure; hashes identify this recording, not a hosted-release attestation.",
    artifacts,
  }, null, 2) + "\n");
  console.log(`Saved ${output}`);
} finally {
  await browser?.close();
  await vite.close();
  await new Promise(resolve => server.close(resolve));
}
