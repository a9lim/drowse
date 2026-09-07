import { bytesToHex } from "@noble/hashes/utils.js";
import { sha256 } from "@noble/hashes/sha2.js";
import { runtimeIdentitySha256 } from "../src/lib/runtime/catalog";
import type { ModelVariant, RuntimeIdentity } from "../src/lib/runtime/contracts";
import {
  DrowseWebLlmRuntime,
  type DrowseWebLlmModule,
} from "../src/hosted/runtime/webLlmEngine";

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
  promptLimit: number;
  sequenceLength: number;
  skipFirst: number;
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
    const directory = await root.getDirectoryHandle("drowse-jlens-build", { create: true });
    const artifacts = [];
    const revision = "a".repeat(40);
    for (const entry of event.data.artifacts) {
      const response = await fetch(
        `/model?path=${encodeURIComponent(entry.path)}`,
        { cache: "no-store" },
      );
      if (!response.ok) throw new Error(`artifact fetch failed for ${entry.path}`);
      const handle = await fileHandle(directory, entry.path);
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
    const prepared = await runtime.prepareCaptureRows(event.data.documents.map((document) => ({
      system: "Read the following text.",
      messages: [{ role: "assistant" as const, content: document }],
    })));
    const captureDigest = sha256.create();
    const planDigest = sha256.create();
    const rows = [];
    let offset = 0;
    for (const item of prepared) {
      if (rows.length >= event.data.promptLimit) break;
      const count = Math.min(item.position + 1, event.data.sequenceLength);
      if (count <= event.data.skipFirst + 1) continue;
      const inputIds = item.inputIds.slice(0, count);
      const positions = Array.from({ length: count }, (_value, index) => index);
      const capture = await runtime.capturePreparedPositions(
        { inputIds, position: count - 1 },
        positions,
      );
      if (
        capture.layerCount !== event.data.runtimeIdentity.layerMap.length ||
        capture.positionCount !== count ||
        capture.hiddenSize !== event.data.runtimeIdentity.hiddenSize
      ) throw new Error("runtime returned an incompatible J-lens residual capture");
      const bytes = new Uint8Array(
        capture.values.buffer,
        capture.values.byteOffset,
        capture.values.byteLength,
      );
      captureDigest.update(bytes);
      planDigest.update(new TextEncoder().encode(JSON.stringify({ inputIds })));
      const response = await fetch(`/capture?start=${offset}`, {
        method: "POST",
        body: bytes,
      });
      if (!response.ok) throw new Error(await response.text());
      rows.push({ input_ids: inputIds, byte_offset: offset, byte_length: bytes.byteLength });
      offset += bytes.byteLength;
      self.postMessage({
        type: "progress",
        completed: rows.length,
        total: event.data.promptLimit,
      });
    }
    if (rows.length !== event.data.promptLimit) {
      throw new Error(
        `corpus yielded ${rows.length} usable prompts, fewer than the requested ${event.data.promptLimit}`,
      );
    }
    await runtime.unload();
    runtime = null;
    self.postMessage({
      type: "done",
      result: {
        runtimeIdentitySha256: fingerprint,
        rows,
        bytes: offset,
        capturePlanSha256: bytesToHex(planDigest.digest()),
        residualSha256: bytesToHex(captureDigest.digest()),
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

async function fileHandle(root: FileSystemDirectoryHandle, path: string): Promise<FileSystemFileHandle> {
  const parts = path.split("/");
  if (
    parts.length === 0 ||
    parts.some((part) => !part || part === "." || part === ".." || part.includes("\\"))
  ) throw new Error(`artifact has an unsafe path: ${path}`);
  let directory = root;
  for (const part of parts.slice(0, -1)) {
    directory = await directory.getDirectoryHandle(part, { create: true });
  }
  return directory.getFileHandle(parts.at(-1)!, { create: true });
}
