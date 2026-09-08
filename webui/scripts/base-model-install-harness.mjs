#!/usr/bin/env node

import { createReadStream } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { resolve } from "node:path";
import { pipeline } from "node:stream/promises";
import { createServer as createViteServer } from "vite";
import { validatePublishDirectory } from "./publish-hosted-model.mjs";
import { validateLocalHostedModelClosure } from "./hosted-model-closure.mjs";
import { validatePackDirectory } from "./build-hosted-catalog.mjs";

const [modelId, modelArgument, coreArgument, lockArgument, outputArgument, portArgument] = process.argv.slice(2);
if (!modelId || !modelArgument || !coreArgument || !lockArgument || !outputArgument || !/^\d+$/.test(portArgument ?? "")) {
  throw new Error("usage: node base-model-install-harness.mjs MODEL_ID MODEL_DIR CORE_DIR RUNTIME_LOCK REPORT PORT");
}
const modelDirectory = resolve(modelArgument);
const coreDirectory = resolve(coreArgument);
const output = resolve(outputArgument);
const lock = JSON.parse(await readFile(resolve(lockArgument), "utf8"));
const pin = lock.models.find(model => model.id === modelId);
if (pin?.modelType !== "base") throw new Error("This harness requires an explicitly classified base model");
const manifest = await validatePublishDirectory(modelDirectory, pin, lock);
await validateLocalHostedModelClosure(modelDirectory, manifest, pin, lock);
const core = await validatePackDirectory(coreDirectory);
const runtimeIdentity = {
  sourceModel: pin.sourceRepository,
  sourceRevision: pin.sourceRevision,
  convertedManifestSha256: pin.manifestSha256,
  quantization: pin.quantization,
  tokenizerSha256: pin.tokenizerSha256,
  chatTemplateSha256: pin.chatTemplateSha256,
  modelLibrarySha256: pin.librarySha256,
  runtimeAbi: lock.runtimeAbi,
  hookAbi: lock.hookAbi,
  hiddenSize: pin.hiddenSize,
  layerMap: pin.layerMap,
};
const payload = {
  type: "run", modelId, runtimeIdentity,
  structuredHookProfile: manifest.structuredHookProfile,
  thinkingProfile: manifest.thinkingProfile,
  contextTokens: manifest.contextWindowSize,
  requiredFeatures: pin.quantization === "q4f16_1" ? ["shader-f16"] : [],
  modelFiles: manifest.files,
  coreFiles: core.files.map(file => ({...file, role: "core_pack"})),
};
const webuiRoot = resolve(import.meta.dirname, "..");
const webLlmPath = resolve(webuiRoot, "node_modules/@drowse/web-llm/lib/index.js");
const vite = await createViteServer({
  root: webuiRoot, configFile: false, appType: "custom", logLevel: "silent",
  server: {middlewareMode: true, hmr: false, watch: null},
});
let cutoff = false;
const blockedRequests = [];
const server = createServer(async (request, response) => {
  for (const [name, value] of Object.entries({
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
  })) response.setHeader(name, value);
  const send = (value, type, status = 200) => {
    response.writeHead(status, {"Content-Type": type}); response.end(value);
  };
  try {
    const url = new URL(request.url ?? "/", "http://127.0.0.1");
    if (url.pathname === "/result" && request.method === "POST") {
      let body = "";
      for await (const chunk of request) {
        body += chunk;
        if (body.length > 1024 * 1024) throw new Error("Report exceeds its size limit");
      }
      const report = {...JSON.parse(body), blockedServerRequests: blockedRequests,
        modelId, runtimeIdentity, recordedAt: new Date().toISOString()};
      if (report.status === "passed" && blockedRequests.length) throw new Error("Offline reload attempted HTTP access");
      await writeFile(output, JSON.stringify(report, null, 2) + "\n", {flag: "wx"});
      console.log(JSON.stringify({modelId, status: report.status, report: output}));
      send("saved", "text/plain");
      return;
    }
    if (url.pathname === "/offline-arm" && request.method === "POST") {
      cutoff = true; send("armed", "text/plain"); return;
    }
    if (cutoff) {
      blockedRequests.push(`${request.method} ${url.pathname}`);
      send("Model-reload network cutoff is armed", "text/plain", 503); return;
    }
    if (url.pathname === "/") {
      send(`<!doctype html><meta charset="utf-8"><link rel="icon" href="data:,"><title>Base model install checks</title>
<h1>Base model core-only install checks</h1><p>Real production runtime, local artifact staging, steering and network-blocked OPFS model reload. This does not test the signed catalog or offline page navigation.</p>
<button id="run">Run install checks</button><p role="status">Ready</p><pre aria-label="Test results"></pre>
<script type="module">
const button=document.querySelector('#run'),status=document.querySelector('[role=status]'),output=document.querySelector('pre');
button.addEventListener('click',async()=>{
button.disabled=true; const request=await (await fetch('/request')).json();const worker=new Worker('/worker.js',{type:'module'});const checks=[];
worker.onmessage=async event=>{const message=event.data;
if(message.type==='progress'){status.textContent=message.message;return;}
if(message.type==='check'){checks.push(message);status.textContent=message.name;output.textContent=JSON.stringify(checks,null,2);return;}
if(!['done','error'].includes(message.type))return;
const report={...message.result,status:message.type==='done'?'passed':'failed',error:message.message,checks};
worker.terminate();const response=await fetch('/result',{method:'POST',body:JSON.stringify(report)});
status.textContent=response.ok?report.status:'Report save failed';output.textContent=JSON.stringify(report,null,2);
};worker.onerror=event=>{status.textContent=event.message;worker.terminate();};worker.postMessage(request);
});</script>`, "text/html; charset=utf-8");
      return;
    }
    if (url.pathname === "/request") {send(JSON.stringify(payload), "application/json"); return;}
    if (url.pathname === "/worker.js") {
      const transformed = await vite.transformRequest("/scripts/hosted-base-install.worker.ts");
      if (!transformed) throw new Error("Worker transform failed");
      send(transformed.code, "text/javascript"); return;
    }
    if (url.pathname === "/webllm.js") {
      response.writeHead(200, {"Content-Type": "text/javascript"});
      await pipeline(createReadStream(webLlmPath), response); return;
    }
    if (url.pathname === "/artifact") {
      const set = url.searchParams.get("set");
      const files = set === "model" ? payload.modelFiles : set === "core" ? payload.coreFiles : [];
      const file = files.find(file => file.path === url.searchParams.get("path"));
      if (!file) {send("not found", "text/plain", 404); return;}
      response.writeHead(200, {"Content-Type": "application/octet-stream", "Content-Length": file.bytes});
      await pipeline(createReadStream(resolve(set === "model" ? modelDirectory : coreDirectory, file.path)), response);
      return;
    }
    vite.middlewares(request, response);
  } catch (error) {
    console.error(error);
    if (response.headersSent) response.destroy(error);
    else send("Internal server error", "text/plain", 500);
  }
});
await new Promise((accept, reject) => {
  server.once("error", reject); server.listen(Number(portArgument), "127.0.0.1", accept);
});
console.log(`Base install validation: http://127.0.0.1:${portArgument}`);
