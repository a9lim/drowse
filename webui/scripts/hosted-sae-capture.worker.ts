import { runtimeIdentitySha256 } from "../src/lib/runtime/catalog";
import type { ModelVariant, RuntimeIdentity } from "../src/lib/runtime/contracts";
import {
  DrowseWebLlmRuntime,
  type DrowseWebLlmModule,
} from "../src/hosted/runtime/webLlmEngine";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";

interface ArtifactEntry {
  path: string;
  role: string;
  bytes: number;
  sha256: string;
}

interface CaptureRequest {
  type: "run";
  modelId: string;
  artifacts: ArtifactEntry[];
  runtimeIdentity: RuntimeIdentity;
  structuredHookProfile: ModelVariant["structuredHookProfile"];
  thinkingProfile: ModelVariant["thinkingProfile"];
  quantization: "q4f16_1" | "q4f32_1";
  requiredFeatures: GPUFeatureName[];
  documents: string[];
  layer: number;
  tokenLimit: number;
  sequenceLength: number;
  contextTokens: number;
}

self.onmessage = async (event: MessageEvent<CaptureRequest>) => {
  if (event.data?.type !== "run") return;
  let runtime: DrowseWebLlmRuntime | null = null;
  try {
    const webLlmUrl = "/webllm.js";
    const webllm = await import(/* @vite-ignore */ webLlmUrl) as unknown as DrowseWebLlmModule;
    runtime = new DrowseWebLlmRuntime(webllm);
    const root = await navigator.storage.getDirectory();
    const directory = await root.getDirectoryHandle("drowse-sae-build", { create: true });
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
    const rows = event.data.documents.map((document) => ({
      system: "Read the following text.",
      messages: [{ role: "assistant" as const, content: document }],
    }));
    const prepared = await runtime.prepareCaptureRows(rows);
    const layerSlot = event.data.runtimeIdentity.layerMap.indexOf(event.data.layer);
    if (layerSlot < 0) throw new Error(`SAE layer ${event.data.layer} is not compiled`);
    const activationDigest = sha256.create();
    const planDigest = sha256.create();
    let capturedTokens = 0;
    let capturedBytes = 0;
    for (let row = 0; row < prepared.length && capturedTokens < event.data.tokenLimit; row += 1) {
      const item = prepared[row];
      const end = Math.min(item.position, item.inputIds.length - 1);
      const start = Math.max(0, end - event.data.sequenceLength + 1);
      const count = Math.min(end - start + 1, event.data.tokenLimit - capturedTokens);
      if (count <= 0) continue;
      const positions = Array.from({ length: count }, (_value, index) => start + index);
      const capture = await runtime.capturePreparedPositions(item, positions);
      const width = event.data.runtimeIdentity.hiddenSize;
      if (
        capture.layerCount !== event.data.runtimeIdentity.layerMap.length ||
        capture.positionCount !== count || capture.hiddenSize !== width
      ) throw new Error("runtime returned an incompatible SAE activation capture");
      const source = layerSlot * count * width;
      const selected = capture.values.slice(source, source + count * width);
      const bytes = new Uint8Array(selected.buffer, selected.byteOffset, selected.byteLength);
      activationDigest.update(bytes);
      planDigest.update(new TextEncoder().encode(JSON.stringify({
        inputIds: item.inputIds,
        positions,
      })));
      const response = await fetch(`/capture?start=${capturedBytes}`, {
        method: "POST",
        body: bytes,
      });
      if (!response.ok) throw new Error(await response.text());
      capturedTokens += count;
      capturedBytes += bytes.byteLength;
      self.postMessage({
        type: "progress",
        completed: capturedTokens,
        total: event.data.tokenLimit,
      });
    }
    if (capturedTokens !== event.data.tokenLimit) {
      throw new Error(
        `corpus yielded ${capturedTokens} capture tokens, fewer than the requested ${event.data.tokenLimit}`,
      );
    }
    await runtime.unload();
    runtime = null;
    self.postMessage({
      type: "done",
      result: {
        tokens: capturedTokens,
        bytes: capturedBytes,
        activationSha256: bytesToHex(activationDigest.digest()),
        capturePlanSha256: bytesToHex(planDigest.digest()),
        runtimeIdentitySha256: fingerprint,
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
