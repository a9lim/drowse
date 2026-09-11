import { contextBindingSha256, runtimeIdentitySha256 } from "../src/lib/runtime/catalog";
import type { CatalogFile, ModelVariant, RuntimeIdentity } from "../src/lib/runtime/contracts";
import { BrowserActivationSpool } from "../src/hosted/fitting/activationSpool";
import { BrowserCapabilityChecker } from "../src/hosted/runtime/capabilities";
import type { BrowserModelArtifact, BrowserModelLoadRequest } from "../src/hosted/runtime/modelBackend";
import { DrowseWebLlmRuntime, type DrowseWebLlmModule } from "../src/hosted/runtime/webLlmEngine";
import type { WebLlmGeneratedToken } from "../src/hosted/runtime/webLlmGeneration";

self.onmessage = async ({ data }: MessageEvent<{
  modelId: string;
  identity: RuntimeIdentity;
  modelFiles: CatalogFile[];
  seeds: number[];
}>) => {
  let runtime: DrowseWebLlmRuntime | undefined;
  try {
    const url = "/webllm.js";
    const webllm = await import(/* @vite-ignore */ url) as DrowseWebLlmModule;
    const checker = new BrowserCapabilityChecker();
    const capabilities = await checker.check();
    if (!capabilities.supported || capabilities.webGpu.fallback !== "hardware") {
      throw new Error("Recording requires a supported hardware WebGPU adapter");
    }
    const fingerprint = runtimeIdentitySha256(data.identity);
    const contextTokens = 2048;
    const modelArtifacts = await stage("model", data.modelFiles);
    const request = {
      model: { id: data.modelId },
      variant: {
        id: `${data.modelId}-recording`, structuredHookProfile: "standard-v3", thinkingProfile: null,
        runtimeIdentity: data.identity, runtimeIdentitySha256: fingerprint,
        contextProfiles: [2048, 4096].map(contextTokens => ({ contextTokens, bindingSha256: contextBindingSha256(fingerprint, contextTokens) })),
        requirements: { features: [], limits: {
          maxStorageBufferBindingSize: 134_217_728, maxComputeWorkgroupStorageSize: 32_768,
          maxComputeWorkgroupSizeX: 256, maxComputeInvocationsPerWorkgroup: 256, maxStorageBuffersPerShaderStage: 10,
        } },
      } as ModelVariant,
      contextTokens, adapter: await checker.adapterForLoad(), artifacts: modelArtifacts,
      optionalPacks: [], activationSpool: new BrowserActivationSpool(), signal: new AbortController().signal,
      onDeviceLost(failure: { message: string }) { throw new Error(failure.message); },
    } as BrowserModelLoadRequest;
    runtime = new DrowseWebLlmRuntime(webllm);
    await runtime.load(request);
    const prompt = "Generate a creative and unique 404 error message that is one short sentence. Return only that single sentence: no introduction, explanations, lists, alternatives, or quotation marks.\nBegin exactly with Error 404:";
    const sampling = { temperature: 1, top_p: 0.9, top_k: 0, max_tokens: 80, return_top_k: 8 };
    const runs = [];
    for (const [index, seed] of data.seeds.entries()) {
      const tokens: WebLlmGeneratedToken[] = [];
      const result = await runtime.streamGeneration({
        input: { kind: "chat", messages: [{ role: "user", content: prompt }] },
        sampling: { ...sampling, seed }, thinking: false, steeringExpression: null, hookProgram: null,
      }, token => { tokens.push(token); });
      const text = tokens.map(token => token.text).join("");
      runs.push({ index, seed, text, tokens, result });
      self.postMessage({ type: "progress", index, seed, text });
    }
    self.postMessage({ type: "result", value: { schemaVersion: 1, recordedAt: new Date().toISOString(), modelId: data.modelId,
      runtimeIdentity: data.identity, runtimeIdentitySha256: fingerprint, contextTokens, prompt, sampling, runs } });
  } catch (error) {
    self.postMessage({ type: "error", message: error instanceof Error ? error.stack : String(error) });
  } finally {
    await runtime?.unload();
  }
};

async function stage(set: string, files: CatalogFile[]): Promise<BrowserModelArtifact[]> {
  const root = await navigator.storage.getDirectory();
  const directory = await root.getDirectoryHandle(set, { create: true });
  const artifacts = [];
  for (const entry of files) {
    const response = await fetch(`/artifact?set=${set}&path=${encodeURIComponent(entry.path)}`);
    if (!response.ok || !response.body) throw new Error(`Unable to stage ${entry.path}`);
    let parent = directory;
    const parts = entry.path.split("/");
    for (const part of parts.slice(0, -1)) parent = await parent.getDirectoryHandle(part, { create: true });
    const handle = await parent.getFileHandle(parts.at(-1)!, { create: true });
    await response.body.pipeTo(await handle.createWritable());
    artifacts.push({ manifest: entry, file: await handle.getFile() });
  }
  return artifacts;
}
