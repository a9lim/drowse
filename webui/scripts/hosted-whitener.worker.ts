import { encodeFp32Safetensors } from "../src/hosted/artifacts/safetensors";
import {
  browserFittedFlatDiscoverPack,
  type BrowserFittedFlatDiscoverPackInput,
} from "../src/hosted/artifacts/fittedAuthoring";
import { BrowserActivationSpool } from "../src/hosted/fitting/activationSpool";
import { BrowserFittingCoordinator } from "../src/hosted/fitting/coordinator";
import { BrowserFittingWorkerClient } from "../src/hosted/fitting/fittingWorkerClient";
import type { SerializedMahalanobisWhitener } from "../src/hosted/fitting/workerContracts";
import {
  contextBindingSha256,
  runtimeIdentitySha256,
} from "../src/lib/runtime/catalog";
import type { ModelVariant, RuntimeIdentity } from "../src/lib/runtime/contracts";
import {
  DrowseWebLlmRuntime,
  type DrowseCaptureRow,
  type DrowseWebLlmModule,
} from "../src/hosted/runtime/webLlmEngine";
import {
  prepareWebLlmActivationCaptureRows,
  WebLlmActivationCaptureSource,
} from "../src/hosted/runtime/webLlmActivationCapture";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";

interface ArtifactEntry {
  path: string;
  role: string;
  bytes: number;
  sha256: string;
}

interface BuildRequest {
  type: "run";
  modelId: string;
  modelType: "chat" | "base";
  artifacts: ArtifactEntry[];
  baselinePrompts: string[];
  contexts: number[];
  rows: DrowseCaptureRow[];
  runtimeIdentity: RuntimeIdentity;
  quantization: "q4f16_1" | "q4f32_1" | "q0f32";
  requiredFeatures: WebGPUFeatureName[];
  structuredHookProfile: ModelVariant["structuredHookProfile"];
  thinkingProfile: ModelVariant["thinkingProfile"];
  artifactProducer: {
    drowseVersion: string;
    producerVersion: string;
  };
  manifold: {
    namespace: string;
    name: string;
    description: string;
    fitMode: "pca";
    hyperparams: Record<string, number>;
    source: { uri: string; repository: string | null; revision: string | null };
    tags: string[];
    nodes: Array<{
      label: string;
      role: string | null;
      kind: string | null;
      statements: string[];
    }>;
  };
}

self.onmessage = async (event: MessageEvent<BuildRequest>) => {
  if (event.data?.type !== "run") return;
  let runtime: DrowseWebLlmRuntime | null = null;
  try {
    const webLlmUrl = "/webllm.js";
    const fittingUrl = "/fitting.js";
    const webllm = await import(/* @vite-ignore */ webLlmUrl) as unknown as DrowseWebLlmModule;
    const fitting = await import(/* @vite-ignore */ fittingUrl);
    runtime = new DrowseWebLlmRuntime(webllm);
    await fitting.default({ module_or_path: "/fitting.wasm" });
    const root = await navigator.storage.getDirectory();
    const directory = await root.getDirectoryHandle("drowse-whitener-build", { create: true });
    const artifacts = [];
    const revision = "a".repeat(40);
    let stagedBytes = 0;
    for (const entry of event.data.artifacts) {
      try {
        const response = await fetch(
          `/model?path=${encodeURIComponent(entry.path)}`,
          { cache: "no-store" },
        );
        if (!response.ok) throw new Error(`artifact fetch failed for ${entry.path}`);
        const handle = await directory.getFileHandle(entry.path, { create: true });
        const writable = await handle.createWritable();
        await writable.write(await response.blob());
        await writable.close();
        stagedBytes += entry.bytes;
        artifacts.push({
          manifest: {
            ...entry,
            revision,
            url: `https://huggingface.co/drowse/local/resolve/${revision}/${entry.path}`,
          },
          file: await handle.getFile(),
        });
      } catch (error) {
        const estimate = await navigator.storage.estimate();
        throw new Error(
          `Could not stage ${entry.path} after ${stagedBytes} bytes ` +
          `(usage ${estimate.usage ?? "unknown"}, quota ${estimate.quota ?? "unknown"}): ` +
          `${error instanceof Error ? error.message : String(error)}`,
          { cause: error },
        );
      }
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter was returned");
    const identitySha256 = runtimeIdentitySha256(event.data.runtimeIdentity);
    const contexts = event.data.contexts.map((contextTokens) => ({
      contextTokens,
      bindingSha256: contextBindingSha256(identitySha256, contextTokens),
    }));
    await runtime.load({
      model: { id: event.data.modelId, modelType: event.data.modelType },
      variant: {
        id: `${event.data.modelId}-${event.data.quantization}`,
        structuredHookProfile: event.data.structuredHookProfile,
        thinkingProfile: event.data.thinkingProfile,
        runtimeIdentity: event.data.runtimeIdentity,
        runtimeIdentitySha256: identitySha256,
        contextProfiles: contexts,
        requirements: {
          features: event.data.requiredFeatures,
          limits: {
            maxStorageBufferBindingSize: 134217728,
            maxComputeWorkgroupSizeX: 256,
            maxComputeInvocationsPerWorkgroup: 256,
          },
        },
      },
      contextTokens: Math.max(...event.data.contexts),
      adapter,
      artifacts,
      signal: new AbortController().signal,
      onDeviceLost() {},
    } as never);
    const prepared = await runtime.prepareCaptureRows(event.data.rows);
    if (prepared.length !== event.data.rows.length) {
      throw new Error("capture renderer returned the wrong neutral row count");
    }
    const rows = event.data.rows.length;
    const columns = event.data.runtimeIdentity.hiddenSize;
    const layerMap = event.data.runtimeIdentity.layerMap;
    const matrices = layerMap.map(() => new Float64Array(rows * columns));
    for (let row = 0; row < prepared.length; row += 1) {
      const capture = await runtime.capturePreparedRow(prepared[row]);
      if (
        capture.layerCount !== layerMap.length || capture.hiddenSize !== columns ||
        capture.values.length !== layerMap.length * columns
      ) throw new Error("runtime returned an incompatible neutral activation row");
      for (let slot = 0; slot < layerMap.length; slot += 1) {
        const start = slot * columns;
        matrices[slot].set(capture.values.subarray(start, start + columns), row * columns);
      }
      self.postMessage({ type: "progress", completed: row + 1, total: rows });
    }
    const tensors: Record<string, { shape: number[]; data: Float32Array }> = {};
    const ridgePerLayer: Record<string, number> = {};
    const whiteners = new Map<number, SerializedMahalanobisWhitener>();
    for (let slot = 0; slot < layerMap.length; slot += 1) {
      const layer = layerMap[slot];
      const fit = fitting.fitMahalanobisWhitener(matrices[slot], rows, columns, 1);
      try {
        const rank = fit.rank;
        const mean = fit.mean();
        const basis = fit.basis();
        const eigenvalues = fit.eigenvalues();
        const inverseScales = fit.inverseScales();
        tensors[`layer_${layer}.mean`] = {
          shape: [columns], data: Float32Array.from(mean),
        };
        tensors[`layer_${layer}.basis`] = {
          shape: [rank, columns], data: Float32Array.from(basis),
        };
        tensors[`layer_${layer}.eigenvalues`] = {
          shape: [rank], data: Float32Array.from(eigenvalues),
        };
        tensors[`layer_${layer}.inverse_scales`] = {
          shape: [rank], data: Float32Array.from(inverseScales),
        };
        ridgePerLayer[String(layer)] = fit.ridge;
        whiteners.set(layer, {
          columns,
          rank,
          ridge: fit.ridge,
          mean,
          basis,
          eigenvalues,
          inverseScales,
        });
      } finally {
        fit.free();
      }
    }
    const tensorBytes = encodeFp32Safetensors(tensors, {
      path: "neutral-whitener.safetensors",
    });
    const tensorPayload = tensorBytes as Uint8Array<ArrayBuffer>;
    const tensorsSha256 = [...new Uint8Array(
      await crypto.subtle.digest("SHA-256", tensorPayload),
    )].map((byte) => byte.toString(16).padStart(2, "0")).join("");
    const response = await fetch("/output", { method: "POST", body: tensorPayload });
    if (!response.ok) throw new Error("failed to return the fitted neutral whitener");
    const manifoldSha256 = await fitCoreManifold(
      runtime,
      event.data.manifold,
      event.data.runtimeIdentity,
      identitySha256,
      contexts.find(
        (context) => context.contextTokens === Math.max(...event.data.contexts),
      )!.bindingSha256,
      whiteners,
      event.data.baselinePrompts,
      event.data.artifactProducer,
    );
    await runtime.unload();
    runtime = null;
    self.postMessage({
      type: "done",
      result: {
        runtimeIdentitySha256: identitySha256,
        contexts,
        ridgePerLayer,
        tensorsSha256,
        manifoldSha256,
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

async function fitCoreManifold(
  runtime: DrowseWebLlmRuntime,
  manifold: BuildRequest["manifold"],
  runtimeIdentity: RuntimeIdentity,
  identitySha256: string,
  contextBindingSha256: string,
  whiteners: ReadonlyMap<number, SerializedMahalanobisWhitener>,
  baselinePrompts: string[],
  artifactProducer: BuildRequest["artifactProducer"],
): Promise<string> {
  const rows: DrowseCaptureRow[] = [];
  const groupOffsets = new Uint32Array(manifold.nodes.length + 1);
  for (let nodeIndex = 0; nodeIndex < manifold.nodes.length; nodeIndex += 1) {
    const node = manifold.nodes[nodeIndex];
    for (let row = 0; row < node.statements.length; row += 1) {
      rows.push({
        system: "Answer in one short paragraph.",
        messages: [
          { role: "user", content: baselinePrompts[row % baselinePrompts.length] },
          {
            role: "assistant",
            content: node.statements[row],
            ...(node.role === null ? {} : { roleName: node.role }),
          },
        ],
      });
    }
    groupOffsets[nodeIndex + 1] = rows.length;
  }
  const rendered = await prepareWebLlmActivationCaptureRows(runtime, rows, groupOffsets);
  const captureSha256 = digest({
    captureRenderSha256: rendered.captureRenderSha256,
    captureVersion: 3,
    contextBindingSha256,
    runtimeIdentitySha256: identitySha256,
  });
  const descriptor = {
    runtimeIdentitySha256: identitySha256,
    contextBindingSha256,
    captureSha256,
    layers: runtimeIdentity.layerMap.map((layer) => ({
      layer,
      rows: rows.length,
      width: runtimeIdentity.hiddenSize,
      expectedBytes: rows.length * runtimeIdentity.hiddenSize * 4,
    })),
  };
  const source = new WebLlmActivationCaptureSource(runtime, {
    descriptor,
    rows,
    preparedRows: rendered.preparedRows,
    layerMap: runtimeIdentity.layerMap,
  });
  const coordinator = new BrowserFittingCoordinator(
    new BrowserActivationSpool(),
    new BrowserFittingWorkerClient({ requestTimeoutMs: 30 * 60_000 }),
  );
  const foundation = await coordinator.captureTopologyFoundation({
    descriptor,
    groupOffsets,
    whiteners,
    fitMode: "pca",
    maxComponents: Math.min(64, runtimeIdentity.hiddenSize),
    maxDimensions: Number(manifold.hyperparams.max_dim ?? 1),
    varianceThreshold: Number(manifold.hyperparams.var_threshold ?? 0.7),
    persistenceFraction: 0.5,
  }, source, {
    onProgress(progress) {
      self.postMessage({
        type: "manifold-progress",
        stage: progress.stage,
        layer: progress.layer,
        completed: progress.completed,
        total: progress.total,
      });
    },
  });
  if (foundation.finalAffineLayers === null || foundation.anchoredNodeCoordinates === null) {
    throw new Error("core PCA fitting returned incomplete geometry");
  }
  const packInput: BrowserFittedFlatDiscoverPackInput = {
    manifold,
    closure: {
      source: manifold.source,
      tags: manifold.tags,
      template: null,
    },
    modelId: runtimeIdentity.sourceModel,
    producerVersion: artifactProducer.producerVersion,
    drowseVersion: artifactProducer.drowseVersion,
    identity: {
      runtimeIdentitySha256: identitySha256,
      contextBindingSha256,
      modelSourceFingerprint: null,
      captureSha256,
      captureVersion: 3,
      captureRenderSha256: rendered.captureRenderSha256,
      baselinePromptsSha256: digest(baselinePrompts),
      fitPolicyVersion: 1,
    },
    topology: foundation.topology,
    consensusGram: foundation.consensusGram,
    nodeCoordinates: foundation.anchoredNodeCoordinates,
    evaluatedLayers: foundation.layers,
    layers: foundation.finalAffineLayers,
  };
  const archive = await browserFittedFlatDiscoverPack(packInput);
  const bytes = new Uint8Array(await archive.arrayBuffer());
  const archiveSha256 = bytesToHex(sha256(bytes));
  const response = await fetch("/manifold-output", { method: "POST", body: bytes });
  if (!response.ok) throw new Error("failed to return the fitted core manifold");
  return archiveSha256;
}

function digest(value: unknown): string {
  return bytesToHex(sha256(new TextEncoder().encode(JSON.stringify(value))));
}
