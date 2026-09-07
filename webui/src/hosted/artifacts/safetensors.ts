import { parseExactJson } from "./json";
import { DROWSE_ARCHIVE_MAX_FILE_BYTES, DrowseArchiveError } from "./types";

const HEADER_MAX_BYTES = 1024 * 1024;
const MAX_KEYS = 4_096;
const MAX_ELEMENTS = DROWSE_ARCHIVE_MAX_FILE_BYTES / 4;
const WRITE_CHUNK_ELEMENTS = 16_384;

export type SafetensorsShape =
  | readonly [number]
  | readonly [number, number];

export interface SafetensorsTensorDescription {
  readonly shape: SafetensorsShape;
  readonly dataOffsets: readonly [start: number, end: number];
}

export interface SafetensorsDescription {
  readonly shapes: ReadonlyMap<string, SafetensorsShape>;
  readonly tensors: ReadonlyMap<string, SafetensorsTensorDescription>;
  readonly metadata: Readonly<Record<string, string>>;
  readonly dataOffset: number;
}

export interface Fp32TensorInput {
  readonly shape: readonly number[];
  readonly data: Float32Array;
}

export interface DecodedFp32Tensor {
  readonly name: string;
  readonly shape: SafetensorsShape;
  readonly data: Float32Array;
}

export interface Fp32SafetensorsReader {
  readonly description: SafetensorsDescription;
  readonly keys: readonly string[];
  tensor(name: string, expectedShape?: readonly number[]): DecodedFp32Tensor;
}

export interface EncodeFp32SafetensorsOptions {
  readonly metadata?: Readonly<Record<string, string>>;
  readonly path?: string;
}

export type SafetensorsWrite = (chunk: Uint8Array) => void | Promise<void>;

export class SafetensorsValidator {
  private readonly prefixChunks: Uint8Array[] = [];
  private prefixBytes = 0;
  private headerBytes: number | null = null;
  private description: SafetensorsDescription | null = null;
  private dataBytes = 0;
  private finiteCarry = new Uint8Array();

  constructor(
    private readonly path: string,
    private readonly fileBytes: number,
  ) {}

  push(chunk: Uint8Array): void {
    if (this.description !== null) {
      this.scanFinite(chunk);
      return;
    }
    let offset = 0;
    while (offset < chunk.length && this.description === null) {
      const targetBytes = this.headerBytes === null ? 8 : this.headerBytes + 8;
      const take = Math.min(chunk.length - offset, targetBytes - this.prefixBytes);
      if (take > 0) {
        this.prefixChunks.push(chunk.slice(offset, offset + take));
        this.prefixBytes += take;
        offset += take;
      }
      if (this.headerBytes === null && this.prefixBytes === 8) {
        const first = this.joinPrefix(8);
        const low = readU32(first, 0);
        const high = readU32(first, 4);
        if (high !== 0 || low === 0 || low > HEADER_MAX_BYTES || low + 8 > this.fileBytes) {
          this.fail("has an invalid or oversized header");
        }
        this.headerBytes = low;
      }
      if (this.headerBytes !== null && this.prefixBytes === this.headerBytes + 8) {
        const headerEnd = this.headerBytes + 8;
        const prefix = this.joinPrefix(headerEnd);
        this.description = validateHeader(
          parseExactJson(prefix.subarray(8), `${this.path} safetensors header`),
          this.path,
          headerEnd,
          this.fileBytes,
        );
        this.prefixChunks.length = 0;
        this.prefixBytes = 0;
      }
    }
    if (this.description !== null && offset < chunk.length) this.scanFinite(chunk.subarray(offset));
  }

  finish(): SafetensorsDescription {
    if (this.description === null) this.fail("has a truncated header");
    if (this.dataBytes !== this.fileBytes - this.description.dataOffset) {
      this.fail("has a truncated tensor payload");
    }
    if (this.finiteCarry.length !== 0) this.fail("has an unaligned tensor payload");
    return this.description;
  }

  private scanFinite(chunk: Uint8Array): void {
    this.dataBytes += chunk.length;
    let bytes = chunk;
    if (this.finiteCarry.length !== 0) {
      const merged = new Uint8Array(this.finiteCarry.length + chunk.length);
      merged.set(this.finiteCarry);
      merged.set(chunk, this.finiteCarry.length);
      bytes = merged;
      this.finiteCarry = new Uint8Array();
    }
    const complete = bytes.length - (bytes.length % 4);
    const view = new DataView(bytes.buffer, bytes.byteOffset, complete);
    for (let offset = 0; offset < complete; offset += 4) {
      if ((view.getUint32(offset, true) & 0x7f800000) === 0x7f800000) {
        this.fail("contains a non-finite fp32 tensor value");
      }
    }
    if (complete !== bytes.length) this.finiteCarry = bytes.slice(complete);
  }

  private joinPrefix(length: number): Uint8Array {
    const output = new Uint8Array(length);
    let offset = 0;
    for (const chunk of this.prefixChunks) {
      const take = Math.min(chunk.length, length - offset);
      output.set(chunk.subarray(0, take), offset);
      offset += take;
      if (offset === length) break;
    }
    return output;
  }

  private fail(message: string): never {
    throw new DrowseArchiveError(
      "SAFETENSORS_INVALID",
      `fitted tensor ${JSON.stringify(this.path)} ${message}`,
      this.path,
    );
  }
}

export function decodeFp32Safetensors(
  bytes: Uint8Array,
  path = "safetensors",
): Fp32SafetensorsReader {
  if (bytes.byteLength > DROWSE_ARCHIVE_MAX_FILE_BYTES) {
    fail(path, "is larger than 512 MiB");
  }
  const validator = new SafetensorsValidator(path, bytes.byteLength);
  validator.push(bytes);
  const description = validator.finish();
  return new Reader(bytes, description, path);
}

export function encodeFp32Safetensors(
  tensors: Readonly<Record<string, Fp32TensorInput>>,
  options: EncodeFp32SafetensorsOptions = {},
): Uint8Array {
  const plan = planEncoding(tensors, options);
  const output = new Uint8Array(plan.fileBytes);
  output.set(plan.prefix);
  let outputOffset = plan.prefix.length;
  for (const tensor of plan.tensors) {
    writeValues(output, outputOffset, tensor.data, 0, tensor.data.length);
    outputOffset += tensor.data.byteLength;
  }
  return output;
}

export async function writeFp32Safetensors(
  tensors: Readonly<Record<string, Fp32TensorInput>>,
  write: SafetensorsWrite,
  options: EncodeFp32SafetensorsOptions = {},
): Promise<SafetensorsDescription> {
  const plan = planEncoding(tensors, options);
  await write(plan.prefix);
  for (const tensor of plan.tensors) {
    for (let start = 0; start < tensor.data.length; start += WRITE_CHUNK_ELEMENTS) {
      const end = Math.min(start + WRITE_CHUNK_ELEMENTS, tensor.data.length);
      const chunk = new Uint8Array((end - start) * 4);
      writeValues(chunk, 0, tensor.data, start, end);
      await write(chunk);
    }
  }
  return plan.description;
}

class Reader implements Fp32SafetensorsReader {
  readonly description: SafetensorsDescription;
  readonly keys: readonly string[];
  private readonly tensors: ReadonlyMap<string, SafetensorsTensorDescription>;

  constructor(
    private readonly bytes: Uint8Array,
    description: SafetensorsDescription,
    private readonly path: string,
  ) {
    this.tensors = new Map(description.tensors);
    this.keys = Object.freeze([...this.tensors.keys()].sort());
    this.description = Object.freeze({
      shapes: new Map(description.shapes),
      tensors: new Map(description.tensors),
      metadata: Object.freeze({ ...description.metadata }),
      dataOffset: description.dataOffset,
    });
  }

  tensor(name: string, expectedShape?: readonly number[]): DecodedFp32Tensor {
    const tensor = this.tensors.get(name);
    if (tensor === undefined) fail(this.path, `has no tensor named ${JSON.stringify(name)}`);
    if (expectedShape !== undefined) {
      const expected = validateShape(expectedShape, this.path, name);
      if (!sameShape(tensor.shape, expected)) {
        fail(
          this.path,
          `key ${JSON.stringify(name)} has shape [${tensor.shape.join(",")}] instead of [${expected.join(",")}]`,
        );
      }
    }
    const [start, end] = tensor.dataOffsets;
    const elements = (end - start) / 4;
    const data = new Float32Array(elements);
    const source = new DataView(
      this.bytes.buffer,
      this.bytes.byteOffset + this.description.dataOffset + start,
      end - start,
    );
    for (let index = 0; index < elements; index += 1) {
      const value = source.getFloat32(index * 4, true);
      if (!Number.isFinite(value)) fail(this.path, "contains a non-finite fp32 tensor value");
      data[index] = value;
    }
    return Object.freeze({ name, shape: tensor.shape, data });
  }
}

interface PlannedTensor extends SafetensorsTensorDescription {
  readonly name: string;
  readonly data: Float32Array;
}

interface EncodingPlan {
  readonly tensors: readonly PlannedTensor[];
  readonly prefix: Uint8Array;
  readonly fileBytes: number;
  readonly description: SafetensorsDescription;
}

function planEncoding(
  inputs: Readonly<Record<string, Fp32TensorInput>>,
  options: EncodeFp32SafetensorsOptions,
): EncodingPlan {
  const path = options.path ?? "safetensors";
  if (!isRecord(inputs)) fail(path, "tensor mapping must be an object");
  const names: string[] = [];
  for (const name in inputs) {
    if (!Object.hasOwn(inputs, name)) continue;
    if (names.length === MAX_KEYS) fail(path, "must contain between 1 and 4096 tensor keys");
    names.push(name);
  }
  names.sort();
  if (names.length === 0) fail(path, "must contain between 1 and 4096 tensor keys");
  if (names.includes("__metadata__")) fail(path, "cannot use the reserved __metadata__ tensor key");
  if (names.some((name) => jsonStringByteLength(name) > HEADER_MAX_BYTES)) {
    fail(path, "has an invalid or oversized header");
  }

  const planned: PlannedTensor[] = [];
  let dataBytes = 0;
  for (const name of names) {
    const input = inputs[name];
    if (!isRecord(input) || !hasExactKeys(input, ["data", "shape"]) || !(input.data instanceof Float32Array)) {
      fail(path, `key ${JSON.stringify(name)} must contain only a shape and Float32Array data`);
    }
    const shape = validateShape(input.shape, path, name);
    const elements = shape.reduce((total, dimension) => total * dimension, 1);
    if (input.data.length !== elements) {
      fail(path, `key ${JSON.stringify(name)} data length does not match its shape`);
    }
    for (const value of input.data) {
      if (!Number.isFinite(value)) fail(path, "contains a non-finite fp32 tensor value");
    }
    const tensorBytes = elements * 4;
    if (dataBytes > DROWSE_ARCHIVE_MAX_FILE_BYTES - tensorBytes) fail(path, "is larger than 512 MiB");
    const dataOffsets = Object.freeze([dataBytes, dataBytes + tensorBytes]) as readonly [number, number];
    planned.push(Object.freeze({ name, shape, data: input.data, dataOffsets }));
    dataBytes += tensorBytes;
  }

  const metadata = validateMetadata(options.metadata, path);
  if (headerJsonByteLength(planned, metadata, options.metadata !== undefined) > HEADER_MAX_BYTES) {
    fail(path, "has an invalid or oversized header");
  }
  const header = Object.create(null) as Record<string, unknown>;
  if (options.metadata !== undefined) header.__metadata__ = metadata;
  for (const tensor of planned) {
    header[tensor.name] = {
      dtype: "F32",
      shape: tensor.shape,
      data_offsets: tensor.dataOffsets,
    };
  }
  const rawHeader = new TextEncoder().encode(JSON.stringify(header));
  const padding = (8 - rawHeader.length % 8) % 8;
  const headerBytes = rawHeader.length + padding;
  if (headerBytes === 0 || headerBytes > HEADER_MAX_BYTES) fail(path, "has an invalid or oversized header");
  const fileBytes = 8 + headerBytes + dataBytes;
  if (fileBytes > DROWSE_ARCHIVE_MAX_FILE_BYTES) fail(path, "is larger than 512 MiB");
  const prefix = new Uint8Array(8 + headerBytes);
  const prefixView = new DataView(prefix.buffer);
  prefixView.setUint32(0, headerBytes, true);
  prefixView.setUint32(4, 0, true);
  prefix.set(rawHeader, 8);
  prefix.fill(0x20, 8 + rawHeader.length);
  const tensors = new Map<string, SafetensorsTensorDescription>();
  for (const tensor of planned) {
    tensors.set(tensor.name, Object.freeze({ shape: tensor.shape, dataOffsets: tensor.dataOffsets }));
  }
  const description = Object.freeze({
    shapes: new Map([...tensors].map(([name, tensor]) => [name, tensor.shape])),
    tensors,
    metadata,
    dataOffset: prefix.length,
  });
  return Object.freeze({ tensors: planned, prefix, fileBytes, description });
}

function validateMetadata(
  value: Readonly<Record<string, string>> | undefined,
  path: string,
): Readonly<Record<string, string>> {
  if (value === undefined) return Object.freeze({});
  if (!isRecord(value)) fail(path, "metadata must contain only string values");
  const entries: Array<[string, string]> = [];
  let bytes = 2;
  for (const key in value) {
    if (!Object.hasOwn(value, key)) continue;
    const entry = value[key];
    if (typeof entry !== "string") fail(path, "metadata must contain only string values");
    bytes += (entries.length > 0 ? 1 : 0)
      + jsonStringByteLength(key)
      + 1
      + jsonStringByteLength(entry);
    if (bytes > HEADER_MAX_BYTES) fail(path, "has an invalid or oversized header");
    entries.push([key, entry]);
  }
  entries.sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0);
  return Object.freeze(Object.fromEntries(entries));
}

function headerJsonByteLength(
  tensors: readonly PlannedTensor[],
  metadata: Readonly<Record<string, string>>,
  includeMetadata: boolean,
): number {
  let bytes = 2;
  let entries = 0;
  const addEntry = (keyBytes: number, valueBytes: number) => {
    bytes += (entries > 0 ? 1 : 0) + keyBytes + 1 + valueBytes;
    entries += 1;
  };
  if (includeMetadata) {
    let metadataBytes = 2;
    let metadataEntries = 0;
    for (const [key, value] of Object.entries(metadata)) {
      metadataBytes += (metadataEntries > 0 ? 1 : 0)
        + jsonStringByteLength(key)
        + 1
        + jsonStringByteLength(value);
      metadataEntries += 1;
      if (metadataBytes > HEADER_MAX_BYTES) return metadataBytes;
    }
    addEntry(jsonStringByteLength("__metadata__"), metadataBytes);
  }
  for (const tensor of tensors) {
    const descriptorBytes = JSON.stringify({
      dtype: "F32",
      shape: tensor.shape,
      data_offsets: tensor.dataOffsets,
    }).length;
    addEntry(jsonStringByteLength(tensor.name), descriptorBytes);
    if (bytes > HEADER_MAX_BYTES) return bytes;
  }
  return bytes;
}

function jsonStringByteLength(value: string): number {
  let bytes = 2;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code === 0x22 || code === 0x5c || code === 0x08 || code === 0x09 || code === 0x0a || code === 0x0c || code === 0x0d) {
      bytes += 2;
    } else if (code < 0x20) {
      bytes += 6;
    } else if (code < 0x80) {
      bytes += 1;
    } else if (code < 0x800) {
      bytes += 2;
    } else if (code >= 0xd800 && code <= 0xdbff && index + 1 < value.length) {
      const next = value.charCodeAt(index + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        bytes += 4;
        index += 1;
      } else {
        bytes += 6;
      }
    } else if (code >= 0xd800 && code <= 0xdfff) {
      bytes += 6;
    } else {
      bytes += 3;
    }
  }
  return bytes;
}

function validateShape(value: readonly number[], path: string, key: string): SafetensorsShape {
  if (
    !Array.isArray(value)
    || value.length < 1
    || value.length > 2
    || !value.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0)
  ) {
    fail(path, `key ${JSON.stringify(key)} must have a positive rank-1 or rank-2 shape`);
  }
  let elements = 1;
  for (const dimension of value) {
    if (dimension > MAX_ELEMENTS / elements) fail(path, `key ${JSON.stringify(key)} is too large`);
    elements *= dimension;
  }
  return value.length === 1
    ? Object.freeze([value[0]]) as readonly [number]
    : Object.freeze([value[0], value[1]]) as readonly [number, number];
}

function writeValues(
  output: Uint8Array,
  outputOffset: number,
  data: Float32Array,
  start: number,
  end: number,
): void {
  const view = new DataView(output.buffer, output.byteOffset, output.byteLength);
  for (let index = start; index < end; index += 1) {
    view.setFloat32(outputOffset + (index - start) * 4, data[index], true);
  }
}

function validateHeader(
  value: unknown,
  path: string,
  dataOffset: number,
  fileBytes: number,
): SafetensorsDescription {
  if (!isRecord(value)) fail(path, "header must be a JSON object");
  const metadata: Record<string, string> = Object.create(null) as Record<string, string>;
  const records: Array<{
    key: string;
    start: number;
    end: number;
    shape: SafetensorsShape;
  }> = [];
  for (const [key, raw] of Object.entries(value)) {
    if (key === "__metadata__") {
      if (!isRecord(raw) || !Object.values(raw).every((entry) => typeof entry === "string")) {
        fail(path, "metadata must contain only string values");
      }
      for (const metadataKey of Object.keys(raw).sort()) metadata[metadataKey] = raw[metadataKey] as string;
      continue;
    }
    if (!isRecord(raw) || !hasExactKeys(raw, ["data_offsets", "dtype", "shape"])) {
      fail(path, `key ${JSON.stringify(key)} has an invalid record`);
    }
    if (raw.dtype !== "F32") fail(path, `key ${JSON.stringify(key)} must be fp32`);
    const shape = validateShape(raw.shape as readonly number[], path, key);
    const elements = shape.reduce((total, dimension) => total * dimension, 1);
    if (
      !Array.isArray(raw.data_offsets)
      || raw.data_offsets.length !== 2
      || !raw.data_offsets.every((offset) => Number.isSafeInteger(offset) && offset >= 0)
    ) {
      fail(path, `key ${JSON.stringify(key)} has invalid data offsets`);
    }
    const [start, end] = raw.data_offsets as number[];
    if (end < start || end - start !== elements * 4) {
      fail(path, `key ${JSON.stringify(key)} byte range does not match its shape`);
    }
    records.push({ key, start, end, shape });
  }
  if (records.length === 0 || records.length > MAX_KEYS) {
    fail(path, "must contain between 1 and 4096 tensor keys");
  }
  records.sort((left, right) => left.start - right.start || left.end - right.end);
  let expectedOffset = 0;
  for (const record of records) {
    if (record.start !== expectedOffset) fail(path, "tensor byte ranges overlap or contain gaps");
    expectedOffset = record.end;
  }
  if (dataOffset + expectedOffset !== fileBytes) {
    fail(path, "tensor byte ranges do not cover the file exactly");
  }
  const tensors = new Map(records.map((record) => [
    record.key,
    Object.freeze({
      shape: record.shape,
      dataOffsets: Object.freeze([record.start, record.end]) as readonly [number, number],
    }),
  ]));
  return {
    shapes: new Map(records.map((record) => [record.key, record.shape])),
    tensors,
    metadata: Object.freeze({ ...metadata }),
    dataOffset,
  };
}

function readU32(bytes: Uint8Array, offset: number): number {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(offset, true);
}

function hasExactKeys(record: Record<string, unknown>, expected: readonly string[]): boolean {
  const remaining = new Set(expected);
  let count = 0;
  for (const key in record) {
    if (!Object.hasOwn(record, key)) continue;
    count += 1;
    if (count > expected.length || !remaining.delete(key)) return false;
  }
  return count === expected.length;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((dimension, index) => dimension === right[index]);
}

function fail(path: string, message: string): never {
  throw new DrowseArchiveError(
    "SAFETENSORS_INVALID",
    `fitted tensor ${JSON.stringify(path)} ${message}`,
    path,
  );
}
