import {sha256} from "@noble/hashes/sha2.js";
import {bytesToHex} from "@noble/hashes/utils.js";
import {contextBindingSha256, runtimeIdentitySha256} from "../src/lib/runtime/catalog";
import {BrowserActivationSpool} from "../src/hosted/fitting/activationSpool";
import {BrowserFeasibilityCorePackCompiler} from "../src/hosted/runtime/browserCorePack";
import {DrowseWebLlmRuntime} from "../src/hosted/runtime/webLlmEngine";
import type {BrowserModelArtifact, BrowserModelLoadRequest} from "../src/hosted/runtime/modelBackend";
import type {CatalogFile, RuntimeIdentity} from "../src/lib/runtime/contracts";

interface RunRequest {
  type: "run";
  modelId: string;
  runtimeIdentity: RuntimeIdentity;
  structuredHookProfile: string;
  thinkingProfile: unknown;
  contextTokens: number;
  requiredFeatures: string[];
  modelFiles: CatalogFile[];
  coreFiles: CatalogFile[];
}

const nativeFetch = self.fetch.bind(self);
let offline = false;
const networkAttempts: string[] = [];
self.fetch = ((input, init) => {
  const value = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  const url = new URL(value, self.location.href);
  if (url.protocol === "http:" || url.protocol === "https:") {
    if (offline || url.origin !== self.location.origin) {
      networkAttempts.push(url.href);
      return Promise.reject(new TypeError(`Blocked test network request: ${url.href}`));
    }
  }
  return nativeFetch(input, init);
}) as typeof self.fetch;

self.onmessage = async (event: MessageEvent<RunRequest>) => {
  if (event.data?.type !== "run") return;
  const data = event.data;
  let runtime: DrowseWebLlmRuntime | null = null;
  const deviceLosses: unknown[] = [];
  const check = (name: string, values: Record<string, unknown>) => self.postMessage({type: "check", name, values});
  try {
    const moduleUrl = "/webllm.js";
    const webllm = await import(/* @vite-ignore */ moduleUrl);
    const root = await navigator.storage.getDirectory();
    const directory = await root.getDirectoryHandle(`drowse-base-install-${data.modelId}`, {create: true});
    const identity = runtimeIdentitySha256(data.runtimeIdentity);
    const revision = "a".repeat(40);
    const artifacts = async (set: "model" | "core", download: boolean): Promise<BrowserModelArtifact[]> => {
      const result: BrowserModelArtifact[] = [];
      const container = await directory.getDirectoryHandle(set, {create: download});
      for (const entry of data[set === "model" ? "modelFiles" : "coreFiles"]) {
        let parent = container;
        const parts = entry.path.split("/");
        for (const part of parts.slice(0, -1)) parent = await parent.getDirectoryHandle(part, {create: download});
        const handle = await parent.getFileHandle(parts.at(-1)!, {create: download});
        if (download) {
          self.postMessage({type: "progress", message: `Verifying ${set}/${entry.path}`});
          const response = await fetch(`/artifact?set=${set}&path=${encodeURIComponent(entry.path)}`);
          if (!response.ok || !response.body) throw new Error(`Artifact fetch failed: ${entry.path}`);
          await response.body.pipeTo(await handle.createWritable());
        }
        const file = await handle.getFile();
        const hasher = sha256.create();
        for await (const chunk of file.stream()) hasher.update(chunk);
        if (file.size !== entry.bytes || bytesToHex(hasher.digest()) !== entry.sha256) {
          throw new Error(`Persisted artifact integrity failed: ${entry.path}`);
        }
        result.push({file, manifest: {...entry, revision,
          url: `https://huggingface.co/drowse/local-${set}/resolve/${revision}/${entry.path}`}});
      }
      return result;
    };
    const makeRequest = async (download: boolean): Promise<BrowserModelLoadRequest> => {
      const model = await artifacts("model", download);
      const core = await artifacts("core", download);
      const variant = {
        id: `${data.modelId}-core-only-test`, runtimeIdentity: data.runtimeIdentity,
        runtimeIdentitySha256: identity, structuredHookProfile: data.structuredHookProfile,
        thinkingProfile: data.thinkingProfile, files: model.map(file => file.manifest),
        contextProfiles: [{contextTokens: data.contextTokens,
          bindingSha256: contextBindingSha256(identity, data.contextTokens)}],
        requirements: {features: data.requiredFeatures, limits: {
          maxBufferSize: 1073741824, maxStorageBufferBindingSize: 1073741824,
          maxComputeWorkgroupStorageSize: 32768, maxStorageBuffersPerShaderStage: 10,
          maxComputeWorkgroupSizeX: 256, maxComputeInvocationsPerWorkgroup: 256,
        }},
      };
      return {
        model: {id: data.modelId, modelType: "base"}, variant,
        requiredCorePack: {id: `${data.modelId}-core`, kind: "core", runtimeIdentitySha256: identity,
          files: core.map(file => file.manifest)},
        contextTokens: data.contextTokens, adapter: await navigator.gpu.requestAdapter(),
        artifacts: [...model, ...core], optionalPacks: [], activationSpool: new BrowserActivationSpool(),
        signal: new AbortController().signal, onDeviceLost: failure => deviceLosses.push(failure),
        onProgress: event => self.postMessage({type: "progress", message: JSON.stringify(event)}),
      } as unknown as BrowserModelLoadRequest;
    };
    const prompt = "I love marmots because";
    const sampling = {max_tokens: 24, seed: 1, temperature: 0, top_p: 1, top_k: 0,
      presence_penalty: 0, frequency_penalty: 0};
    const generate = async (expression: string | null, program: unknown, forced?: number) => {
      const ids: number[] = [];
      let score: number | null = null;
      const result = await runtime!.streamGeneration({
        input: {kind: "raw", prompt}, sampling: {...sampling,
          temperature: forced === undefined ? 0 : 1, max_tokens: forced === undefined ? 24 : 1},
        thinking: false, steeringExpression: expression, hookProgram: program,
        ...(forced === undefined ? {} : {replay: {forcedPrefixTokenIds: [forced], scoreTokenIds: [forced]}}),
        onRawToken: token => {
          if (token.tokenId !== null) ids.push(token.tokenId);
          const value = token.replayScore?.requestedLogprobs.find(entry => entry.tokenId === forced)?.logprob;
          if (value !== undefined) score = value;
        },
      }, () => undefined);
      if (result.tokens < 1 || ids.length !== result.tokens) throw new Error("Generation returned no exact token stream");
      return {ids, score, result};
    };
    let request = await makeRequest(true);
    runtime = new DrowseWebLlmRuntime(webllm);
    await runtime.load(request);
    self.postMessage({type: "progress", message: "Loading the required core compiler"});
    let compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    if (compiler.jlensGpuDictionary() !== null || compiler.saeGpuDictionary() !== null) throw new Error("Core-only test unexpectedly installed optional tools");
    check("core-only load", {identity, optionalPacks: 0, layers: data.runtimeIdentity.layerMap.length});
    const baseline = await generate(null, null);
    check("base generation", {ids: baseline.ids, performance: baseline.result});
    const expression = "0.5 default/welcoming.detached";
    const program = compiler.compile(expression);
    const active = "affineActive" in program ? program.affineActive : program.enabled;
    if (!active.some(value => value !== 0)) {
      throw new Error("Core steering did not produce an active affine GPU program");
    }
    const ordinary = await generate(null, null, baseline.ids[0]);
    const steeredScore = await generate(expression, program, baseline.ids[0]);
    if (!Number.isFinite(ordinary.score) || !Number.isFinite(steeredScore.score) ||
        Math.abs(ordinary.score! - steeredScore.score!) < 1e-6) throw new Error("Steering did not change the same token's finite conditional logprob");
    const steered = await generate(expression, program);
    check("physical core steering", {expression, ordinaryLogprob: ordinary.score,
      steeredLogprob: steeredScore.score, ids: steered.ids});
    const after = await generate(null, null);
    if (JSON.stringify(after.ids) !== JSON.stringify(baseline.ids)) throw new Error("Steering leaked into the next baseline generation");
    await runtime.unload();
    runtime = null;
    const response = await fetch("/offline-arm", {method: "POST"});
    if (!response.ok) throw new Error("Could not arm the HTTP cutoff");
    offline = true;
    request = await makeRequest(false);
    runtime = new DrowseWebLlmRuntime(webllm);
    await runtime.load(request);
    compiler = await BrowserFeasibilityCorePackCompiler.load(request);
    const reused = await generate(null, null);
    if (JSON.stringify(reused.ids) !== JSON.stringify(baseline.ids)) throw new Error("Offline model reload changed baseline generation");
    const reusedSteer = await generate(expression, compiler.compile(expression));
    if (JSON.stringify(reusedSteer.ids) !== JSON.stringify(steered.ids)) throw new Error("Offline model reload changed steered generation");
    if (networkAttempts.length || deviceLosses.length) throw new Error("Offline reload attempted network access or lost the GPU");
    check("network-blocked OPFS model reload", {networkAttempts: 0, baselineIds: reused.ids, steeredIds: reusedSteer.ids});
    await runtime.unload();
    runtime = null;
    self.postMessage({type: "done", result: {scope: "production runtime core-only load and offline model reload; not signed installation or offline page navigation",
      optionalPacks: 0, networkAttempts, deviceLosses, userAgent: navigator.userAgent}});
  } catch (error) {
    await runtime?.unload();
    self.postMessage({type: "error", message: error instanceof Error ? error.stack ?? error.message : String(error)});
  }
};
