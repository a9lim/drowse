import type {
  ActivationCaptureSink,
  BrowserActivationCaptureSource,
} from "../fitting/coordinator";
import type { ActivationSpoolDescriptor } from "../fitting/activationSpool";
import type {
  DrowseCaptureRow,
  DrowsePreparedCaptureRow,
  DrowseResidualCapture,
} from "./webLlmEngine";
import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";

const DEFAULT_CAPTURE_BATCH_ROWS = 16;

export interface WebLlmCaptureRuntimePort {
  prepareCaptureRows(rows: DrowseCaptureRow[]): Promise<DrowsePreparedCaptureRow[]>;
  capturePreparedRow(row: DrowsePreparedCaptureRow): Promise<DrowseResidualCapture>;
}

export interface WebLlmActivationCapturePlan {
  descriptor: ActivationSpoolDescriptor;
  rows: DrowseCaptureRow[];
  preparedRows?: DrowsePreparedCaptureRow[];
  layerMap: number[];
  batchRows?: number;
}

export class WebLlmActivationCaptureSource implements BrowserActivationCaptureSource {
  private readonly plan: WebLlmActivationCapturePlan;
  private readonly captureSlots = new Map<number, number>();

  constructor(
    private readonly runtime: WebLlmCaptureRuntimePort,
    plan: WebLlmActivationCapturePlan,
  ) {
    this.plan = validatePlan(plan);
    for (let slot = 0; slot < this.plan.layerMap.length; slot += 1) {
      this.captureSlots.set(this.plan.layerMap[slot], slot);
    }
  }

  async capture(
    sink: ActivationCaptureSink,
    signal: AbortSignal,
    onProgress: (completedRows: number, totalRows: number) => void,
  ): Promise<void> {
    signal.throwIfAborted();
    const prepared = this.plan.preparedRows ??
      await this.runtime.prepareCaptureRows(this.plan.rows);
    if (prepared.length !== this.plan.rows.length) {
      throw new Error("The Drowse capture renderer returned the wrong row count");
    }
    const batchRows = this.plan.batchRows ?? DEFAULT_CAPTURE_BATCH_ROWS;
    for (let start = 0; start < prepared.length; start += batchRows) {
      signal.throwIfAborted();
      const end = Math.min(start + batchRows, prepared.length);
      const buffers = new Map(
        this.plan.descriptor.layers.map((layer) => [
          layer.layer,
          new Float32Array((end - start) * layer.width),
        ]),
      );
      for (let row = start; row < end; row += 1) {
        signal.throwIfAborted();
        const capture = await this.runtime.capturePreparedRow(prepared[row]);
        this.validateCapture(capture);
        for (const layer of this.plan.descriptor.layers) {
          const slot = this.captureSlots.get(layer.layer)!;
          const source = slot * layer.width;
          buffers.get(layer.layer)!.set(
            capture.values.subarray(source, source + layer.width),
            (row - start) * layer.width,
          );
        }
        onProgress(row + 1, prepared.length);
      }
      for (const layer of this.plan.descriptor.layers) {
        signal.throwIfAborted();
        await sink.appendRows(layer.layer, start, buffers.get(layer.layer)!);
      }
    }
  }

  private validateCapture(capture: DrowseResidualCapture): void {
    const width = this.plan.descriptor.layers[0].width;
    if (
      capture.layerCount !== this.plan.layerMap.length ||
      capture.positionCount !== 1 || capture.hiddenSize !== width ||
      capture.values.length !== capture.layerCount * width
    ) {
      throw new Error("The Drowse runtime returned an incompatible activation row");
    }
  }
}

function validatePlan(plan: WebLlmActivationCapturePlan): WebLlmActivationCapturePlan {
  const rows = plan.descriptor.layers[0]?.rows;
  const width = plan.descriptor.layers[0]?.width;
  if (
    rows === undefined || width === undefined || plan.rows.length !== rows ||
    plan.descriptor.layers.some((layer) => layer.rows !== rows || layer.width !== width) ||
    (plan.preparedRows !== undefined && (
      plan.preparedRows.length !== rows ||
      plan.preparedRows.some((row) =>
        !Array.isArray(row.inputIds) || row.inputIds.length === 0 ||
        !Number.isSafeInteger(row.position) || row.position < 0 ||
        row.position >= row.inputIds.length
      )
    ))
  ) {
    throw new TypeError("The Drowse capture plan does not match its activation descriptor");
  }
  if (
    plan.layerMap.length === 0 ||
    new Set(plan.layerMap).size !== plan.layerMap.length ||
    plan.layerMap.some((layer) => !Number.isSafeInteger(layer) || layer < 0) ||
    plan.descriptor.layers.some((layer) => !plan.layerMap.includes(layer.layer))
  ) {
    throw new TypeError("The Drowse capture plan has an invalid runtime layer map");
  }
  const batchRows = plan.batchRows ?? DEFAULT_CAPTURE_BATCH_ROWS;
  if (!Number.isSafeInteger(batchRows) || batchRows <= 0 || batchRows > 256) {
    throw new TypeError("The Drowse capture batch size must be between 1 and 256 rows");
  }
  return {
    ...plan,
    descriptor: {
      ...plan.descriptor,
      layers: plan.descriptor.layers.map((layer) => ({ ...layer })),
    },
    rows: plan.rows.map((row) => ({
      system: row.system,
      messages: row.messages.map((message) => ({ ...message })),
    })),
    preparedRows: plan.preparedRows?.map((row) => ({
      inputIds: [...row.inputIds],
      position: row.position,
    })),
    layerMap: [...plan.layerMap],
  };
}

export async function prepareWebLlmActivationCaptureRows(
  runtime: WebLlmCaptureRuntimePort,
  rows: DrowseCaptureRow[],
  groupOffsets: Uint32Array,
  signal?: AbortSignal,
): Promise<{
  preparedRows: DrowsePreparedCaptureRow[];
  captureRenderSha256: string;
}> {
  signal?.throwIfAborted();
  const preparedRows = await runtime.prepareCaptureRows(rows);
  signal?.throwIfAborted();
  if (
    preparedRows.length !== rows.length || groupOffsets.length < 2 ||
    groupOffsets[0] !== 0 || groupOffsets[groupOffsets.length - 1] !== rows.length ||
    [...groupOffsets].some((offset, index) =>
      index > 0 && offset <= groupOffsets[index - 1]
    )
  ) {
    throw new TypeError("The Drowse capture render does not match its node partition");
  }
  const encoded = new TextEncoder().encode(JSON.stringify({
    schemaVersion: 1,
    groupOffsets: [...groupOffsets],
    rows: preparedRows.map((row) => ({
      inputIds: row.inputIds,
      position: row.position,
    })),
  }));
  return {
    preparedRows: preparedRows.map((row) => ({
      inputIds: [...row.inputIds],
      position: row.position,
    })),
    captureRenderSha256: bytesToHex(sha256(encoded)),
  };
}
