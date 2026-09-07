import { runtimeIdentitySha256 } from "../src/lib/runtime/catalog";
import type { ModelVariant, RuntimeIdentity } from "../src/lib/runtime/contracts";
import {
  DrowseWebLlmRuntime,
  type DrowseWebLlmModule,
} from "../src/hosted/runtime/webLlmEngine";
import { compileRankOneHookProgram } from "../src/hosted/runtime/rankOneHookProgram";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";

interface CaptureRequest {
  type: "run";
  modelId: string;
  artifacts: Array<{ path: string; role: string; bytes: number; sha256: string }>;
  runtimeIdentity: RuntimeIdentity;
  structuredHookProfile: ModelVariant["structuredHookProfile"];
  thinkingProfile: ModelVariant["thinkingProfile"];
  quantization: "q4f16_1" | "q4f32_1";
  requiredFeatures: GPUFeatureName[];
  inputIds: number[];
  positions: number[];
  greedyPrompt: string;
  greedyMaxTokens: number;
  namedRoleCheck?: {
    row: {
      system: string;
      messages: Array<{
        role: "system" | "user" | "assistant";
        content: string;
        roleName?: string;
      }>;
    };
  };
  contextTokens: number;
  dumpShaders: boolean;
}

self.onmessage = async (event: MessageEvent<CaptureRequest>) => {
  if (event.data?.type !== "run") return;
  let runtime: DrowseWebLlmRuntime | null = null;
  try {
    const webLlmUrl = "/webllm.js";
    const webllm = await import(/* @vite-ignore */ webLlmUrl) as unknown as DrowseWebLlmModule;
    runtime = new DrowseWebLlmRuntime(webllm);
    const root = await navigator.storage.getDirectory();
    const directory = await root.getDirectoryHandle("drowse-residual-capture", { create: true });
    const artifacts = [];
    const revision = "a".repeat(40);
    for (const entry of event.data.artifacts) {
      const response = await fetch(
        `/model?path=${encodeURIComponent(entry.path)}`,
        { cache: "no-store" },
      );
      if (!response.ok) throw new Error(`artifact fetch failed for ${entry.path}`);
      const handle = await directory.getFileHandle(entry.path, { create: true });
      const writable = await handle.createWritable();
      await writable.write(await response.blob());
      await writable.close();
      artifacts.push({
        manifest: {
          ...entry,
          revision,
          url: `https://huggingface.co/polythetic/local/resolve/${revision}/${entry.path}`,
        },
        file: await handle.getFile(),
      });
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter was returned");
    const shaderPosts = event.data.dumpShaders ? installShaderCapture() : [];
    const fingerprint = runtimeIdentitySha256(event.data.runtimeIdentity);
    await runtime.load({
      model: { id: event.data.modelId },
      variant: {
        id: `${event.data.modelId}-${event.data.quantization}`,
        structuredHookProfile: event.data.structuredHookProfile,
        thinkingProfile: event.data.thinkingProfile,
        runtimeIdentity: event.data.runtimeIdentity,
        runtimeIdentitySha256: fingerprint,
        contextProfiles: [{ contextTokens: event.data.contextTokens, bindingSha256: "0".repeat(64) }],
        requirements: {
          features: event.data.requiredFeatures,
          limits: {
            maxStorageBufferBindingSize: 134217728,
            maxComputeWorkgroupSizeX: 256,
            maxComputeInvocationsPerWorkgroup: 256,
          },
        },
      },
      contextTokens: event.data.contextTokens,
      adapter,
      artifacts,
      optionalPacks: [],
      signal: new AbortController().signal,
      onDeviceLost() {},
    } as never);
    const capture = await runtime.capturePreparedPositions({
      inputIds: event.data.inputIds,
      position: event.data.positions[event.data.positions.length - 1],
    }, event.data.positions);
    const rankOne = deterministicRankOneProgram(
      capture.hiddenSize,
      capture.layerCount,
    );
    const rankOneControl = await runtime.capturePreparedRankOnePositionsV1({
      inputIds: event.data.inputIds,
      position: event.data.positions[event.data.positions.length - 1],
    }, event.data.positions, rankOne.controlProgram);
    const steered = await runtime.capturePreparedRankOnePositionsV1({
      inputIds: event.data.inputIds,
      position: event.data.positions[event.data.positions.length - 1],
    }, event.data.positions, rankOne.program);
    const greedyInputIds = await runtime.tokenizeText(event.data.greedyPrompt);
    const greedyTokenIds: number[] = [];
    await runtime.streamGeneration(
      {
        input: { kind: "raw", prompt: event.data.greedyPrompt },
        sampling: deterministicSampling(event.data.greedyMaxTokens),
        thinking: false,
        steeringExpression: null,
        hookProgram: null,
        onRawToken(token) {
          if (token.tokenId === null) {
            throw new Error("greedy generation returned a token without its vocabulary ID");
          }
          greedyTokenIds.push(token.tokenId);
        },
      },
      () => undefined,
    );
    const namedRolePrepared = event.data.namedRoleCheck === undefined
      ? undefined
      : (await runtime.prepareCaptureRows([event.data.namedRoleCheck.row]))[0];
    const bytes = new Uint8Array(
      capture.values.buffer,
      capture.values.byteOffset,
      capture.values.byteLength,
    );
    const steeredBytes = new Uint8Array(
      steered.values.buffer,
      steered.values.byteOffset,
      steered.values.byteLength,
    );
    const rankOneControlBytes = new Uint8Array(
      rankOneControl.values.buffer,
      rankOneControl.values.byteOffset,
      rankOneControl.values.byteLength,
    );
    const response = await fetch("/capture?kind=unsteered", {
      method: "POST",
      body: bytes,
    });
    if (!response.ok) throw new Error(await response.text());
    const rankOneControlResponse = await fetch("/capture?kind=rank-one-control", {
      method: "POST",
      body: rankOneControlBytes,
    });
    if (!rankOneControlResponse.ok) {
      throw new Error(await rankOneControlResponse.text());
    }
    const steeredResponse = await fetch("/capture?kind=steered", {
      method: "POST",
      body: steeredBytes,
    });
    if (!steeredResponse.ok) throw new Error(await steeredResponse.text());
    await Promise.all(shaderPosts);
    await runtime.unload();
    runtime = null;
    self.postMessage({
      type: "done",
      result: {
        runtimeIdentitySha256: fingerprint,
        layerCount: capture.layerCount,
        positionCount: capture.positionCount,
        hiddenSize: capture.hiddenSize,
        positions: capture.positions,
        residualSha256: bytesToHex(sha256(bytes)),
        bytes: bytes.byteLength,
        rankOneControlResidualSha256: bytesToHex(sha256(rankOneControlBytes)),
        rankOneControlBytes: rankOneControlBytes.byteLength,
        steeredResidualSha256: bytesToHex(sha256(steeredBytes)),
        steeredBytes: steeredBytes.byteLength,
        probeMeasurements: [...steered.measurements],
        rankOneProgram: rankOne.descriptor,
        greedyPrompt: event.data.greedyPrompt,
        greedyInputIds,
        greedyTokenIds,
        ...(namedRolePrepared === undefined
          ? {}
          : { namedRolePrepared }),
      },
    });
  } catch (error) {
    try { await runtime?.unload(); } catch {}
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.stack ?? error.message : String(error),
    });
  }
};

function deterministicRankOneProgram(hiddenSize: number, layerCount: number) {
  const enabledLayer = Math.floor(layerCount / 2);
  const zero = new Float32Array(hiddenSize);
  const direction = new Float32Array(hiddenSize);
  const component = Math.fround(1 / Math.sqrt(hiddenSize));
  for (let index = 0; index < hiddenSize; index += 1) {
    direction[index] = index % 2 === 0 ? component : -component;
  }
  const target = Math.fround(1.25);
  const along = Math.fround(0.4);
  const collapse = Math.fround(0.3);
  const program = compileRankOneHookProgram(
    hiddenSize,
    Array.from({ length: layerCount }, (_, layer) => ({
      enabled: layer === enabledLayer,
      basis: layer === enabledLayer ? direction : zero,
      neutral: zero,
      target: layer === enabledLayer ? target : 0,
      along: layer === enabledLayer ? along : 0,
      collapse: layer === enabledLayer ? collapse : 0,
      probeBasis: direction,
      probeNeutral: zero,
    })),
  );
  return {
    program,
    controlProgram: {
      ...program,
      enabled: new Uint32Array(program.layerCount),
    },
    descriptor: {
      abiVersion: 1,
      enabledLayer,
      direction: [...direction],
      target,
      along,
      collapse,
    },
  };
}

function deterministicSampling(maxTokens: number) {
  return {
    max_tokens: maxTokens,
    seed: 1,
    temperature: 0,
    top_p: 1,
    top_k: 0,
    presence_penalty: 0,
    frequency_penalty: 0,
  };
}

function installShaderCapture(): Promise<void>[] {
  const posts: Promise<void>[] = [];
  const sources = new WeakMap<GPUShaderModule, string>();
  const prototype = GPUDevice.prototype;
  const createShaderModule = prototype.createShaderModule;
  const createComputePipeline = prototype.createComputePipeline;
  const createComputePipelineAsync = prototype.createComputePipelineAsync;
  Object.defineProperty(prototype, "createShaderModule", {
    configurable: true,
    value(this: GPUDevice, descriptor: GPUShaderModuleDescriptor) {
      const module = createShaderModule.call(this, descriptor);
      sources.set(module, descriptor.code);
      return module;
    },
  });
  const record = (descriptor: GPUComputePipelineDescriptor) => {
    const code = sources.get(descriptor.compute.module);
    if (!code) return;
    const entryPoint = descriptor.compute.entryPoint ?? "main";
    posts.push(fetch(`/shader?entry=${encodeURIComponent(entryPoint)}`, {
      method: "POST",
      body: code,
    }).then((response) => {
      if (!response.ok) throw new Error(`shader capture failed for ${entryPoint}`);
    }));
  };
  Object.defineProperty(prototype, "createComputePipeline", {
    configurable: true,
    value(this: GPUDevice, descriptor: GPUComputePipelineDescriptor) {
      record(descriptor);
      return createComputePipeline.call(this, descriptor);
    },
  });
  Object.defineProperty(prototype, "createComputePipelineAsync", {
    configurable: true,
    value(this: GPUDevice, descriptor: GPUComputePipelineDescriptor) {
      record(descriptor);
      return createComputePipelineAsync.call(this, descriptor);
    },
  });
  return posts;
}
