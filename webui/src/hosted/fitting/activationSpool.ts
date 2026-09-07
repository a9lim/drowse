import { migrateLegacyDatabase, drowseStorageRoot } from "../runtime/brandMigration";
import { randomUuid } from "../runtime/randomId";

export const ACTIVATION_SPOOL_SCHEMA_VERSION = 1 as const;

const SHA256 = /^[a-f0-9]{64}$/;
const DATABASE_NAME = "drowse-hosted-activation-spool";
const DATABASE_VERSION = 1;
const CAPTURES_STORE = "captures";
const STORAGE_LOCK = "drowse-activation-spool-v1";
const OPFS_FITTING = "fitting";
const OPFS_ACTIVATIONS = "activation-spool";
const FLOAT32_BYTES = 4;
const DEFAULT_CHUNK_BYTES = 1024 * 1024;
export const ACTIVATION_SPOOL_MAX_CHUNK_BYTES = 8 * 1024 * 1024;
const MAX_LAYERS = 1_024;

export interface ActivationSpoolIdentity {
  runtimeIdentitySha256: string;
  contextBindingSha256: string;
  captureSha256: string;
}

export interface ActivationLayerSpec {
  layer: number;
  rows: number;
  width: number;
  expectedBytes: number;
}

export interface ActivationSpoolDescriptor extends ActivationSpoolIdentity {
  layers: ActivationLayerSpec[];
}

export interface ActivationLayerRecord extends ActivationLayerSpec {
  writtenRows: number;
  writtenBytes: number;
  checkpointRows: number;
  checkpointBytes: number;
  sealed: boolean;
}

export interface ActivationSpoolRecord extends ActivationSpoolIdentity {
  schemaVersion: typeof ACTIVATION_SPOOL_SCHEMA_VERSION;
  state: "staged" | "committed";
  layers: ActivationLayerRecord[];
  writeEpoch: string | null;
  createdAt: number;
  updatedAt: number;
  committedAt: number | null;
}

export interface ActivationSpoolMetadataEntry {
  key: string;
  value: unknown;
}

export interface ActivationSpoolMetadataPort {
  initialize(): Promise<void>;
  read(captureSha256: string): Promise<unknown | undefined>;
  write(record: ActivationSpoolRecord): Promise<void>;
  remove(key: string): Promise<void>;
  list(): Promise<ActivationSpoolMetadataEntry[]>;
  close?(): void;
}

export interface ActivationSpoolFilePort {
  initialize(): Promise<void>;
  list(): Promise<string[]>;
  size(path: string): Promise<number | null>;
  append(path: string, offset: number, bytes: Uint8Array): Promise<void>;
  truncate(path: string, size: number): Promise<void>;
  read(path: string, offset: number, length: number): Promise<Uint8Array>;
  remove(path: string): Promise<void>;
  clear(): Promise<void>;
  close?(): void;
}

export interface ActivationSpoolLease {
  release(): Promise<void>;
}

export interface ActivationSpoolLeasePort {
  tryAcquire(captureSha256: string): Promise<ActivationSpoolLease | null>;
}

export type ActivationSpoolExclusiveRunner =
  <T>(operation: () => Promise<T>) => Promise<T>;

export interface ActivationSpoolOptions {
  metadata: ActivationSpoolMetadataPort;
  files: ActivationSpoolFilePort;
  leases?: ActivationSpoolLeasePort;
  now?: () => number;
  createEpoch?: () => string;
  runExclusive?: ActivationSpoolExclusiveRunner;
}

export interface ActivationSpoolReadOptions {
  startRow?: number;
  rowCount?: number;
  maxChunkBytes?: number;
  signal?: AbortSignal;
}

export interface ActivationRowChunk {
  layer: number;
  startRow: number;
  rows: number;
  width: number;
  values: Float32Array;
}

export class ActivationSpoolError extends Error {
  constructor(
    readonly code:
      | "INVALID_DESCRIPTOR"
      | "CAPTURE_EXISTS"
      | "CAPTURE_NOT_FOUND"
      | "IDENTITY_MISMATCH"
      | "LAYER_MISMATCH"
      | "NOT_STAGED"
      | "NOT_COMMITTED"
      | "STALE_WRITER"
      | "OUT_OF_ORDER_WRITE"
      | "NONFINITE_VALUES"
      | "INCOMPLETE_LAYER"
      | "STORAGE_MISMATCH"
      | "CORRUPT_RECORD",
    message: string,
  ) {
    super(message);
    this.name = "ActivationSpoolError";
  }
}

export class ActivationSpool {
  private readonly metadata: ActivationSpoolMetadataPort;
  private readonly files: ActivationSpoolFilePort;
  private readonly leases: ActivationSpoolLeasePort;
  private readonly now: () => number;
  private readonly createEpoch: () => string;
  private readonly runExclusive: ActivationSpoolExclusiveRunner;
  private initialization: Promise<void> | null = null;

  constructor(options: ActivationSpoolOptions) {
    this.metadata = options.metadata;
    this.files = options.files;
    this.leases = options.leases ?? fallbackLeasePort;
    this.now = options.now ?? Date.now;
    this.createEpoch = options.createEpoch ?? defaultEpoch;
    this.runExclusive = options.runExclusive ?? defaultExclusiveRunner;
  }

  initialize(): Promise<void> {
    this.initialization ??= Promise.all([
      this.metadata.initialize(),
      this.files.initialize(),
    ]).then(() => this.runExclusive(() => this.recoverUnlocked())).catch((error) => {
      this.initialization = null;
      throw error;
    });
    return this.initialization;
  }

  async begin(descriptor: ActivationSpoolDescriptor): Promise<ActivationSpoolWriter> {
    validateActivationSpoolDescriptor(descriptor);
    const bound = cloneDescriptor(descriptor);
    await this.initialize();
    await this.runExclusive(async () => {
      const existing = await this.metadata.read(bound.captureSha256);
      if (existing !== undefined) {
        let record: ActivationSpoolRecord;
        try {
          validateActivationSpoolRecord(existing);
          record = existing;
        } catch {
          await this.removeCaptureUnlocked(bound.captureSha256);
          throw spoolError(
            "CORRUPT_RECORD",
            `Stored activation capture ${bound.captureSha256} is corrupt`,
          );
        }
        assertIdentity(record, bound);
        throw spoolError(
          "CAPTURE_EXISTS",
          `Activation capture ${bound.captureSha256} already exists`,
        );
      }
    });
    const lease = await this.leases.tryAcquire(bound.captureSha256);
    if (!lease) {
      throw spoolError(
        "CAPTURE_EXISTS",
        `Activation capture ${bound.captureSha256} already has an active writer`,
      );
    }
    try {
      const snapshot = await this.runExclusive(async () => {
        const existing = await this.metadata.read(bound.captureSha256);
        if (existing !== undefined) {
          let record: ActivationSpoolRecord;
          try {
            validateActivationSpoolRecord(existing);
            record = existing;
          } catch {
            await this.removeCaptureUnlocked(bound.captureSha256);
            throw spoolError(
              "CORRUPT_RECORD",
              `Stored activation capture ${bound.captureSha256} is corrupt`,
            );
          }
          assertIdentity(record, bound);
          throw spoolError(
            "CAPTURE_EXISTS",
            `Activation capture ${bound.captureSha256} already exists`,
          );
        }

        await this.removeCaptureFilesUnlocked(bound.captureSha256);
        const created: string[] = [];
        try {
          for (const layer of bound.layers) {
            const path = activationSpoolFileName(bound.captureSha256, layer.layer);
            await this.files.truncate(path, 0);
            created.push(path);
          }
          const timestamp = this.timestamp();
          const record: ActivationSpoolRecord = {
            schemaVersion: ACTIVATION_SPOOL_SCHEMA_VERSION,
            state: "staged",
            runtimeIdentitySha256: bound.runtimeIdentitySha256,
            contextBindingSha256: bound.contextBindingSha256,
            captureSha256: bound.captureSha256,
            layers: bound.layers.map((layer) => ({
              ...cloneLayerSpec(layer),
              writtenRows: 0,
              writtenBytes: 0,
              checkpointRows: 0,
              checkpointBytes: 0,
              sealed: false,
            })),
            writeEpoch: this.newEpoch(),
            createdAt: timestamp,
            updatedAt: timestamp,
            committedAt: null,
          };
          await this.metadata.write(record);
          return cloneRecord(record);
        } catch (error) {
          for (const path of created) await this.files.remove(path).catch(() => undefined);
          throw error;
        }
      });
      return new ActivationSpoolWriter(this, snapshot, lease);
    } catch (error) {
      await lease.release().catch(() => undefined);
      throw error;
    }
  }

  async resume(descriptor: ActivationSpoolDescriptor): Promise<ActivationSpoolWriter> {
    validateActivationSpoolDescriptor(descriptor);
    const bound = cloneDescriptor(descriptor);
    await this.initialize();
    await this.runExclusive(async () => {
      const record = await this.requireRecordUnlocked(bound.captureSha256);
      assertIdentity(record, bound);
      assertLayerSpecs(record.layers, bound.layers);
      if (record.state !== "staged") {
        throw spoolError(
          "NOT_STAGED",
          `Activation capture ${bound.captureSha256} is already committed`,
        );
      }
    });
    const lease = await this.leases.tryAcquire(bound.captureSha256);
    if (!lease) {
      throw spoolError(
        "STALE_WRITER",
        `Activation capture ${bound.captureSha256} already has an active writer`,
      );
    }
    try {
      const snapshot = await this.runExclusive(async () => {
        const record = await this.requireRecordUnlocked(bound.captureSha256);
        assertIdentity(record, bound);
        assertLayerSpecs(record.layers, bound.layers);
        if (record.state !== "staged") {
          throw spoolError(
            "NOT_STAGED",
            `Activation capture ${bound.captureSha256} is already committed`,
          );
        }
        const recovered = await this.recoverStagedRecordUnlocked(record);
        recovered.writeEpoch = this.newEpoch();
        recovered.updatedAt = this.timestamp();
        await this.metadata.write(recovered);
        return cloneRecord(recovered);
      });
      return new ActivationSpoolWriter(this, snapshot, lease);
    } catch (error) {
      await lease.release().catch(() => undefined);
      throw error;
    }
  }

  async openCommitted(
    identity: ActivationSpoolIdentity | ActivationSpoolDescriptor,
  ): Promise<ActivationSpoolReader> {
    validateActivationSpoolIdentity(identity);
    const boundIdentity = cloneIdentity(identity);
    let boundLayers: ActivationLayerSpec[] | null = null;
    if ("layers" in identity) {
      validateActivationSpoolDescriptor(identity);
      boundLayers = identity.layers.map(cloneLayerSpec);
    }
    await this.initialize();
    const snapshot = await this.runExclusive(async () => {
      const record = await this.requireRecordUnlocked(boundIdentity.captureSha256);
      assertIdentity(record, boundIdentity);
      if (boundLayers) assertLayerSpecs(record.layers, boundLayers);
      if (record.state !== "committed") {
        throw spoolError(
          "NOT_COMMITTED",
          `Activation capture ${boundIdentity.captureSha256} has not been committed`,
        );
      }
      await this.assertCommittedFilesUnlocked(record);
      return cloneRecord(record);
    });
    return new ActivationSpoolReader(this, snapshot);
  }

  async recover(): Promise<void> {
    await this.initialize();
    await this.runExclusive(() => this.recoverUnlocked());
  }

  async removeCommitted(identity: ActivationSpoolIdentity): Promise<boolean> {
    validateActivationSpoolIdentity(identity);
    const bound = cloneIdentity(identity);
    await this.initialize();
    const lease = await this.leases.tryAcquire(bound.captureSha256);
    if (!lease) {
      throw spoolError(
        "STALE_WRITER",
        `Activation capture ${bound.captureSha256} still has an active writer`,
      );
    }
    try {
      return await this.runExclusive(async () => {
        const raw = await this.metadata.read(bound.captureSha256);
        if (raw === undefined) return false;
        validateActivationSpoolRecord(raw);
        assertIdentity(raw, bound);
        if (raw.state !== "committed") {
          throw spoolError(
            "NOT_COMMITTED",
            `Activation capture ${bound.captureSha256} has not been committed`,
          );
        }
        await this.removeCaptureUnlocked(bound.captureSha256);
        return true;
      });
    } finally {
      await lease.release();
    }
  }

  async clear(): Promise<void> {
    await this.initialize();
    await this.runExclusive(async () => {
      const entries = await this.metadata.list();
      const leases: ActivationSpoolLease[] = [];
      try {
        for (const captureSha256 of new Set(
          entries.map((entry) => entry.key).filter((key) => SHA256.test(key)),
        )) {
          const lease = await this.leases.tryAcquire(captureSha256);
          if (!lease) {
            throw spoolError(
              "STALE_WRITER",
              `Activation capture ${captureSha256} still has an active writer`,
            );
          }
          leases.push(lease);
        }
        for (const entry of entries) await this.metadata.remove(entry.key);
        await this.files.clear();
      } finally {
        for (const lease of leases.reverse()) await lease.release();
      }
    });
  }

  close(): void {
    this.metadata.close?.();
    this.files.close?.();
    this.initialization = null;
  }

  async writerAppend(
    identity: ActivationSpoolIdentity,
    writeEpoch: string,
    layerIndex: number,
    startRow: number,
    values: Float32Array,
  ): Promise<ActivationLayerRecord> {
    validateRowIndex(startRow, "start row");
    if (!(values instanceof Float32Array)) {
      throw spoolError("INVALID_DESCRIPTOR", "Activation rows must be a Float32Array");
    }
    if (values.byteLength > ACTIVATION_SPOOL_MAX_CHUNK_BYTES) {
      throw spoolError(
        "INVALID_DESCRIPTOR",
        `Activation writes must not exceed ${ACTIVATION_SPOOL_MAX_CHUNK_BYTES} bytes`,
      );
    }
    const bytes = encodeFiniteFloat32(values);
    await this.initialize();
    return this.runExclusive(async () => {
      const record = await this.requireWriterRecordUnlocked(identity, writeEpoch);
      const layer = requireLayer(record, layerIndex);
      if (layer.sealed) {
        throw spoolError("OUT_OF_ORDER_WRITE", `Activation layer ${layerIndex} is sealed`);
      }
      if (values.length === 0 || values.length % layer.width !== 0) {
        throw spoolError(
          "OUT_OF_ORDER_WRITE",
          `Activation write for layer ${layerIndex} must contain one or more complete rows`,
        );
      }
      const rowCount = values.length / layer.width;
      if (startRow !== layer.writtenRows) {
        throw spoolError(
          "OUT_OF_ORDER_WRITE",
          `Activation layer ${layerIndex} expected row ${layer.writtenRows}, received ${startRow}`,
        );
      }
      if (startRow + rowCount > layer.rows) {
        throw spoolError(
          "OUT_OF_ORDER_WRITE",
          `Activation write exceeds the ${layer.rows} rows declared for layer ${layerIndex}`,
        );
      }
      const path = activationSpoolFileName(record.captureSha256, layer.layer);
      await this.assertFileSizeUnlocked(path, layer.writtenBytes);
      await this.files.append(path, layer.writtenBytes, bytes);
      const next = cloneRecord(record);
      const nextLayer = requireLayer(next, layerIndex);
      nextLayer.writtenRows += rowCount;
      nextLayer.writtenBytes += bytes.byteLength;
      next.updatedAt = this.timestamp();
      await this.metadata.write(next);
      return cloneLayer(nextLayer);
    });
  }

  async writerCheckpoint(
    identity: ActivationSpoolIdentity,
    writeEpoch: string,
    layerIndex: number,
  ): Promise<ActivationLayerRecord> {
    await this.initialize();
    return this.runExclusive(async () => {
      const record = await this.requireWriterRecordUnlocked(identity, writeEpoch);
      const layer = requireLayer(record, layerIndex);
      const path = activationSpoolFileName(record.captureSha256, layer.layer);
      await this.assertFileSizeUnlocked(path, layer.writtenBytes);
      const next = cloneRecord(record);
      const nextLayer = requireLayer(next, layerIndex);
      nextLayer.checkpointRows = nextLayer.writtenRows;
      nextLayer.checkpointBytes = nextLayer.writtenBytes;
      next.updatedAt = this.timestamp();
      await this.metadata.write(next);
      return cloneLayer(nextLayer);
    });
  }

  async writerSeal(
    identity: ActivationSpoolIdentity,
    writeEpoch: string,
    layerIndex: number,
  ): Promise<ActivationLayerRecord> {
    await this.initialize();
    return this.runExclusive(async () => {
      const record = await this.requireWriterRecordUnlocked(identity, writeEpoch);
      const layer = requireLayer(record, layerIndex);
      if (layer.writtenRows !== layer.rows || layer.writtenBytes !== layer.expectedBytes) {
        throw spoolError(
          "INCOMPLETE_LAYER",
          `Activation layer ${layerIndex} has ${layer.writtenRows}/${layer.rows} rows and cannot be sealed`,
        );
      }
      const path = activationSpoolFileName(record.captureSha256, layer.layer);
      await this.assertFileSizeUnlocked(path, layer.expectedBytes);
      const next = cloneRecord(record);
      const nextLayer = requireLayer(next, layerIndex);
      nextLayer.checkpointRows = nextLayer.rows;
      nextLayer.checkpointBytes = nextLayer.expectedBytes;
      nextLayer.sealed = true;
      next.updatedAt = this.timestamp();
      await this.metadata.write(next);
      return cloneLayer(nextLayer);
    });
  }

  async writerCommit(
    identity: ActivationSpoolIdentity,
    writeEpoch: string,
  ): Promise<ActivationSpoolReader> {
    await this.initialize();
    const snapshot = await this.runExclusive(async () => {
      const record = await this.requireWriterRecordUnlocked(identity, writeEpoch);
      const incomplete = record.layers.find((layer) => !layer.sealed);
      if (incomplete) {
        throw spoolError(
          "INCOMPLETE_LAYER",
          `Activation layer ${incomplete.layer} must be sealed before capture commit`,
        );
      }
      for (const layer of record.layers) {
        const path = activationSpoolFileName(record.captureSha256, layer.layer);
        await this.assertFileSizeUnlocked(path, layer.expectedBytes);
      }
      const timestamp = this.timestamp();
      const committed: ActivationSpoolRecord = {
        ...cloneRecord(record),
        state: "committed",
        writeEpoch: null,
        updatedAt: timestamp,
        committedAt: timestamp,
      };
      await this.metadata.write(committed);
      return committed;
    });
    return new ActivationSpoolReader(this, snapshot);
  }

  async writerRollback(
    identity: ActivationSpoolIdentity,
    writeEpoch: string,
  ): Promise<void> {
    await this.initialize();
    await this.runExclusive(async () => {
      validateActivationSpoolIdentity(identity);
      validateEpoch(writeEpoch);
      const value = await this.metadata.read(identity.captureSha256);
      if (value !== undefined) {
        let record: ActivationSpoolRecord;
        try {
          validateActivationSpoolRecord(value);
          record = value;
        } catch {
          throw spoolError(
            "CORRUPT_RECORD",
            `Stored activation capture ${identity.captureSha256} is corrupt`,
          );
        }
        assertIdentity(record, identity);
        if (record.state !== "staged") {
          throw spoolError(
            "NOT_STAGED",
            `Activation capture ${identity.captureSha256} is not staged`,
          );
        }
        if (record.writeEpoch !== writeEpoch) {
          throw spoolError(
            "STALE_WRITER",
            `Activation capture ${identity.captureSha256} is owned by another write transaction`,
          );
        }
      }
      await this.metadata.remove(identity.captureSha256);
      await this.removeCaptureFilesUnlocked(identity.captureSha256);
    });
  }

  async *writerReadChunks(
    identity: ActivationSpoolIdentity,
    writeEpoch: string,
    layerIndex: number,
    options: ActivationSpoolReadOptions,
  ): AsyncGenerator<ActivationRowChunk> {
    yield* this.readChunks(
      identity,
      layerIndex,
      options,
      async () => {
        const record = await this.requireWriterRecordUnlocked(identity, writeEpoch);
        return { record, readableRows: requireLayer(record, layerIndex).writtenRows };
      },
    );
  }

  async *committedReadChunks(
    identity: ActivationSpoolIdentity,
    layerIndex: number,
    options: ActivationSpoolReadOptions,
  ): AsyncGenerator<ActivationRowChunk> {
    yield* this.readChunks(
      identity,
      layerIndex,
      options,
      async () => {
        const record = await this.requireRecordUnlocked(identity.captureSha256);
        assertIdentity(record, identity);
        if (record.state !== "committed") {
          throw spoolError(
            "NOT_COMMITTED",
            `Activation capture ${identity.captureSha256} has not been committed`,
          );
        }
        return { record, readableRows: requireLayer(record, layerIndex).rows };
      },
    );
  }

  private async *readChunks(
    identity: ActivationSpoolIdentity,
    layerIndex: number,
    options: ActivationSpoolReadOptions,
    resolveRecord: () => Promise<{
      record: ActivationSpoolRecord;
      readableRows: number;
    }>,
  ): AsyncGenerator<ActivationRowChunk> {
    validateActivationSpoolIdentity(identity);
    validateLayerIndex(layerIndex);
    const startRow = options.startRow ?? 0;
    validateRowIndex(startRow, "read start row");
    const maxChunkBytes = options.maxChunkBytes ?? DEFAULT_CHUNK_BYTES;
    if (
      !Number.isSafeInteger(maxChunkBytes) || maxChunkBytes <= 0 ||
      maxChunkBytes > ACTIVATION_SPOOL_MAX_CHUNK_BYTES
    ) {
      throw spoolError(
        "INVALID_DESCRIPTOR",
        `Activation read chunk size must be between 1 and ${ACTIVATION_SPOOL_MAX_CHUNK_BYTES} bytes`,
      );
    }
    if (
      options.rowCount !== undefined &&
      (!Number.isSafeInteger(options.rowCount) || options.rowCount < 0)
    ) {
      throw spoolError("INVALID_DESCRIPTOR", "Activation read row count is invalid");
    }
    throwIfAborted(options.signal);
    await this.initialize();

    let cursor = startRow;
    let endRow: number | null = null;
    while (endRow === null || cursor < endRow) {
      throwIfAborted(options.signal);
      const result = await this.runExclusive(async () => {
        const { record, readableRows } = await resolveRecord();
        const layer = requireLayer(record, layerIndex);
        if (startRow > readableRows) {
          throw spoolError(
            "OUT_OF_ORDER_WRITE",
            `Activation read starts after the ${readableRows} readable rows in layer ${layerIndex}`,
          );
        }
        const requestedEnd = options.rowCount === undefined
          ? readableRows
          : checkedSum(startRow, options.rowCount, "activation read range");
        if (requestedEnd > readableRows) {
          throw spoolError(
            "OUT_OF_ORDER_WRITE",
            `Activation read ends after the ${readableRows} readable rows in layer ${layerIndex}`,
          );
        }
        if (endRow !== null && requestedEnd !== endRow) {
          throw spoolError(
            "STORAGE_MISMATCH",
            `Activation layer ${layerIndex} changed while it was being read`,
          );
        }
        endRow = requestedEnd;
        if (cursor >= requestedEnd) return null;
        const rowBytes = layer.width * FLOAT32_BYTES;
        if (rowBytes > maxChunkBytes) {
          throw spoolError(
            "INVALID_DESCRIPTOR",
            `Activation read chunk bound ${maxChunkBytes} is smaller than one ${rowBytes}-byte row`,
          );
        }
        const rowsPerChunk = Math.floor(maxChunkBytes / rowBytes);
        const rows = Math.min(rowsPerChunk, requestedEnd - cursor);
        const byteOffset = cursor * rowBytes;
        const byteLength = rows * rowBytes;
        const path = activationSpoolFileName(record.captureSha256, layer.layer);
        const expectedFileBytes = record.state === "committed"
          ? layer.expectedBytes
          : layer.writtenBytes;
        await this.assertFileSizeUnlocked(path, expectedFileBytes);
        throwIfAborted(options.signal);
        const stored = await this.files.read(path, byteOffset, byteLength);
        throwIfAborted(options.signal);
        if (stored.byteLength !== byteLength) {
          throw spoolError(
            "STORAGE_MISMATCH",
            `Activation layer ${layerIndex} returned ${stored.byteLength} bytes, expected ${byteLength}`,
          );
        }
        const values = decodeFiniteFloat32(stored.slice(), layerIndex, cursor);
        return { layer: layer.layer, startRow: cursor, rows, width: layer.width, values };
      });
      if (!result) break;
      throwIfAborted(options.signal);
      cursor += result.rows;
      yield result;
    }
  }

  private async recoverUnlocked(): Promise<void> {
    const entries = await this.metadata.list();
    const referenced = new Set<string>();
    for (const entry of entries) {
      let record: ActivationSpoolRecord;
      try {
        if (!SHA256.test(entry.key)) throw new Error("invalid metadata key");
        validateActivationSpoolRecord(entry.value);
        if (entry.value.captureSha256 !== entry.key) {
          throw new Error("capture identity does not match metadata key");
        }
        record = entry.value;
      } catch {
        await this.metadata.remove(entry.key);
        if (SHA256.test(entry.key)) await this.removeCaptureFilesUnlocked(entry.key);
        continue;
      }

      let lease: ActivationSpoolLease | null = null;
      try {
        if (record.state === "staged") {
          lease = await this.leases.tryAcquire(record.captureSha256);
          if (lease) {
            record = await this.recoverStagedRecordUnlocked(record);
            record.writeEpoch = null;
            record.updatedAt = this.timestamp();
            await this.metadata.write(record);
          } else {
            await this.assertActiveStagedFilesUnlocked(record);
          }
        } else {
          await this.assertCommittedFilesUnlocked(record);
        }
        for (const layer of record.layers) {
          referenced.add(activationSpoolFileName(record.captureSha256, layer.layer));
        }
      } catch (error) {
        if (!isRecoverableCorruption(error)) throw error;
        await this.metadata.remove(record.captureSha256);
        await this.removeCaptureFilesUnlocked(record.captureSha256);
      } finally {
        await lease?.release();
      }
    }

    for (const path of await this.files.list()) {
      if (!referenced.has(path)) await this.files.remove(path);
    }
  }

  private async recoverStagedRecordUnlocked(
    record: ActivationSpoolRecord,
  ): Promise<ActivationSpoolRecord> {
    const recovered = cloneRecord(record);
    for (const layer of recovered.layers) {
      const path = activationSpoolFileName(recovered.captureSha256, layer.layer);
      const size = await this.files.size(path);
      if (size === null || size < layer.checkpointBytes) {
        throw spoolError(
          "STORAGE_MISMATCH",
          `Activation layer ${layer.layer} is shorter than its checkpoint`,
        );
      }
      if (size !== layer.checkpointBytes) {
        await this.files.truncate(path, layer.checkpointBytes);
      }
      layer.writtenRows = layer.checkpointRows;
      layer.writtenBytes = layer.checkpointBytes;
    }
    return recovered;
  }

  private async assertActiveStagedFilesUnlocked(record: ActivationSpoolRecord): Promise<void> {
    for (const layer of record.layers) {
      const path = activationSpoolFileName(record.captureSha256, layer.layer);
      const size = await this.files.size(path);
      if (size === null || size < layer.writtenBytes || size > layer.expectedBytes) {
        throw spoolError(
          "STORAGE_MISMATCH",
          `Active activation layer ${layer.layer} has an invalid byte tail`,
        );
      }
    }
  }

  private async assertCommittedFilesUnlocked(record: ActivationSpoolRecord): Promise<void> {
    for (const layer of record.layers) {
      if (!layer.sealed) {
        throw spoolError(
          "CORRUPT_RECORD",
          `Committed activation layer ${layer.layer} is not sealed`,
        );
      }
      await this.assertFileSizeUnlocked(
        activationSpoolFileName(record.captureSha256, layer.layer),
        layer.expectedBytes,
      );
    }
  }

  private async requireWriterRecordUnlocked(
    identity: ActivationSpoolIdentity,
    writeEpoch: string,
  ): Promise<ActivationSpoolRecord> {
    validateActivationSpoolIdentity(identity);
    validateEpoch(writeEpoch);
    const record = await this.requireRecordUnlocked(identity.captureSha256);
    assertIdentity(record, identity);
    if (record.state !== "staged") {
      throw spoolError(
        "NOT_STAGED",
        `Activation capture ${identity.captureSha256} is not staged`,
      );
    }
    if (record.writeEpoch !== writeEpoch) {
      throw spoolError(
        "STALE_WRITER",
        `Activation capture ${identity.captureSha256} is owned by another write transaction`,
      );
    }
    return record;
  }

  private async requireRecordUnlocked(captureSha256: string): Promise<ActivationSpoolRecord> {
    const value = await this.metadata.read(captureSha256);
    if (value === undefined) {
      throw spoolError(
        "CAPTURE_NOT_FOUND",
        `Activation capture ${captureSha256} does not exist`,
      );
    }
    try {
      validateActivationSpoolRecord(value);
      if (value.captureSha256 !== captureSha256) throw new Error("metadata key mismatch");
      return value;
    } catch {
      throw spoolError(
        "CORRUPT_RECORD",
        `Stored activation capture ${captureSha256} is corrupt`,
      );
    }
  }

  private async assertFileSizeUnlocked(path: string, expectedBytes: number): Promise<void> {
    const size = await this.files.size(path);
    if (size !== expectedBytes) {
      throw spoolError(
        "STORAGE_MISMATCH",
        `Activation spool ${path} has ${size ?? "no"} bytes, expected ${expectedBytes}`,
      );
    }
  }

  private async removeCaptureUnlocked(captureSha256: string): Promise<void> {
    await this.metadata.remove(captureSha256);
    await this.removeCaptureFilesUnlocked(captureSha256);
  }

  private async removeCaptureFilesUnlocked(captureSha256: string): Promise<void> {
    const prefix = `${captureSha256}.layer-`;
    for (const path of await this.files.list()) {
      if (path.startsWith(prefix) && isActivationSpoolFileName(path)) {
        await this.files.remove(path);
      }
    }
  }

  private timestamp(): number {
    const value = this.now();
    if (!Number.isSafeInteger(value) || value < 0) {
      throw spoolError("INVALID_DESCRIPTOR", "Activation spool timestamp is invalid");
    }
    return value;
  }

  private newEpoch(): string {
    const epoch = this.createEpoch();
    validateEpoch(epoch);
    return epoch;
  }
}

export class ActivationSpoolWriter {
  readonly identity: Readonly<ActivationSpoolIdentity>;
  readonly layers: readonly Readonly<ActivationLayerSpec>[];
  private readonly boundIdentity: ActivationSpoolIdentity;
  private readonly writeEpoch: string;
  private readonly progressByLayer = new Map<number, ActivationLayerRecord>();
  private state: "open" | "committed" | "rolled_back" = "open";

  constructor(
    private readonly spool: ActivationSpool,
    record: ActivationSpoolRecord,
    private readonly lease: ActivationSpoolLease,
  ) {
    if (record.state !== "staged" || record.writeEpoch === null) {
      throw spoolError("NOT_STAGED", "An activation writer requires an owned staged record");
    }
    this.boundIdentity = Object.freeze(cloneIdentity(record));
    this.identity = this.boundIdentity;
    this.layers = Object.freeze(
      record.layers.map((layer) => Object.freeze(cloneLayerSpec(layer))),
    );
    this.writeEpoch = record.writeEpoch;
    for (const layer of record.layers) this.progressByLayer.set(layer.layer, cloneLayer(layer));
  }

  get progress(): ActivationLayerRecord[] {
    return this.layers.map((layer) => cloneLayer(this.progressByLayer.get(layer.layer)!));
  }

  async appendRows(
    layer: number,
    startRow: number,
    values: Float32Array,
  ): Promise<ActivationLayerRecord> {
    this.assertOpen();
    const progress = await this.spool.writerAppend(
      this.boundIdentity,
      this.writeEpoch,
      layer,
      startRow,
      values,
    );
    this.progressByLayer.set(layer, cloneLayer(progress));
    return progress;
  }

  async checkpoint(layer: number): Promise<ActivationLayerRecord> {
    this.assertOpen();
    const progress = await this.spool.writerCheckpoint(
      this.boundIdentity,
      this.writeEpoch,
      layer,
    );
    this.progressByLayer.set(layer, cloneLayer(progress));
    return progress;
  }

  async sealLayer(layer: number): Promise<ActivationLayerRecord> {
    this.assertOpen();
    const progress = await this.spool.writerSeal(this.boundIdentity, this.writeEpoch, layer);
    this.progressByLayer.set(layer, cloneLayer(progress));
    return progress;
  }

  readChunks(
    layer: number,
    options: ActivationSpoolReadOptions = {},
  ): AsyncIterable<ActivationRowChunk> {
    this.assertOpen();
    return this.spool.writerReadChunks(
      this.boundIdentity,
      this.writeEpoch,
      layer,
      cloneReadOptions(options),
    );
  }

  async commit(): Promise<ActivationSpoolReader> {
    this.assertOpen();
    const reader = await this.spool.writerCommit(this.boundIdentity, this.writeEpoch);
    this.state = "committed";
    await this.lease.release();
    return reader;
  }

  async rollback(): Promise<void> {
    if (this.state === "rolled_back") return;
    this.assertOpen();
    await this.spool.writerRollback(this.boundIdentity, this.writeEpoch);
    this.state = "rolled_back";
    await this.lease.release();
  }

  private assertOpen(): void {
    if (this.state !== "open") {
      throw spoolError("STALE_WRITER", "Activation spool write transaction is closed");
    }
  }
}

export class ActivationSpoolReader {
  readonly identity: Readonly<ActivationSpoolIdentity>;
  readonly layers: readonly Readonly<ActivationLayerSpec>[];
  private readonly boundIdentity: ActivationSpoolIdentity;

  constructor(
    private readonly spool: ActivationSpool,
    record: ActivationSpoolRecord,
  ) {
    if (record.state !== "committed") {
      throw spoolError("NOT_COMMITTED", "An activation reader requires a committed record");
    }
    this.boundIdentity = Object.freeze(cloneIdentity(record));
    this.identity = this.boundIdentity;
    this.layers = Object.freeze(
      record.layers.map((layer) => Object.freeze(cloneLayerSpec(layer))),
    );
  }

  readChunks(
    layer: number,
    options: ActivationSpoolReadOptions = {},
  ): AsyncIterable<ActivationRowChunk> {
    return this.spool.committedReadChunks(
      this.boundIdentity,
      layer,
      cloneReadOptions(options),
    );
  }
}

export class BrowserActivationSpoolMetadataPort implements ActivationSpoolMetadataPort {
  private database: IDBDatabase | null = null;
  private initialization: Promise<void> | null = null;

  initialize(): Promise<void> {
    this.initialization ??= openActivationSpoolDatabase().then((database) => {
      this.database = database;
      database.onversionchange = () => {
        database.close();
        if (this.database === database) {
          this.database = null;
          this.initialization = null;
        }
      };
    }).catch((error) => {
      this.initialization = null;
      throw error;
    });
    return this.initialization;
  }

  async read(captureSha256: string): Promise<unknown | undefined> {
    await this.initialize();
    const transaction = this.requireDatabase().transaction(CAPTURES_STORE, "readonly");
    return requestResult(transaction.objectStore(CAPTURES_STORE).get(captureSha256));
  }

  async write(record: ActivationSpoolRecord): Promise<void> {
    await this.initialize();
    validateActivationSpoolRecord(record);
    const transaction = this.requireDatabase().transaction(CAPTURES_STORE, "readwrite");
    transaction.objectStore(CAPTURES_STORE).put(record);
    await transactionDone(transaction);
  }

  async remove(key: string): Promise<void> {
    await this.initialize();
    const transaction = this.requireDatabase().transaction(CAPTURES_STORE, "readwrite");
    transaction.objectStore(CAPTURES_STORE).delete(key);
    await transactionDone(transaction);
  }

  async list(): Promise<ActivationSpoolMetadataEntry[]> {
    await this.initialize();
    const transaction = this.requireDatabase().transaction(CAPTURES_STORE, "readonly");
    const store = transaction.objectStore(CAPTURES_STORE);
    const [keys, values] = await Promise.all([
      requestResult<IDBValidKey[]>(store.getAllKeys()),
      requestResult<unknown[]>(store.getAll()),
    ]);
    return keys.map((key, index) => ({ key: String(key), value: values[index] }));
  }

  close(): void {
    this.database?.close();
    this.database = null;
    this.initialization = null;
  }

  private requireDatabase(): IDBDatabase {
    if (!this.database) throw new Error("Activation spool metadata is not initialized");
    return this.database;
  }
}

export class BrowserActivationSpoolFilePort implements ActivationSpoolFilePort {
  private directory: FileSystemDirectoryHandle | null = null;
  private initialization: Promise<void> | null = null;

  initialize(): Promise<void> {
    this.initialization ??= this.initializeOnce().catch((error) => {
      this.initialization = null;
      this.directory = null;
      throw error;
    });
    return this.initialization;
  }

  private async initializeOnce(): Promise<void> {
    if (typeof navigator === "undefined" || !navigator.storage?.getDirectory) {
      throw new Error("Origin-private file system storage is unavailable");
    }
    const root = await navigator.storage.getDirectory();
    const drowse = await drowseStorageRoot(root);
    const fitting = await drowse.getDirectoryHandle(OPFS_FITTING, { create: true });
    this.directory = await fitting.getDirectoryHandle(OPFS_ACTIVATIONS, { create: true });
  }

  async list(): Promise<string[]> {
    const directory = this.requireDirectory() as FileSystemDirectoryHandle & {
      entries(): AsyncIterable<[string, FileSystemHandle]>;
    };
    const names: string[] = [];
    for await (const [name] of directory.entries()) names.push(name);
    return names.sort();
  }

  async size(path: string): Promise<number | null> {
    validateActivationSpoolPath(path);
    try {
      const handle = await this.requireDirectory().getFileHandle(path);
      return (await handle.getFile()).size;
    } catch (error) {
      if (isNotFoundError(error)) return null;
      if (isTypeMismatchError(error)) {
        throw spoolError(
          "STORAGE_MISMATCH",
          `Activation spool ${path} is not a file`,
        );
      }
      throw error;
    }
  }

  async append(path: string, offset: number, bytes: Uint8Array): Promise<void> {
    validateActivationSpoolPath(path);
    validateByteOffset(offset, "activation append offset");
    const handle = await this.requireDirectory().getFileHandle(path, { create: true });
    const file = await handle.getFile();
    if (file.size !== offset) {
      throw spoolError(
        "STORAGE_MISMATCH",
        `Activation spool ${path} has ${file.size} bytes, expected append offset ${offset}`,
      );
    }
    const writable = await handle.createWritable({ keepExistingData: true });
    try {
      if (offset > 0) await writable.seek(offset);
      if (!(bytes.buffer instanceof ArrayBuffer)) {
        throw new TypeError("Activation spool writes require an ArrayBuffer-backed view");
      }
      await writable.write(bytes as Uint8Array<ArrayBuffer>);
      await writable.close();
    } catch (error) {
      await writable.abort(error).catch(() => undefined);
      throw error;
    }
  }

  async truncate(path: string, size: number): Promise<void> {
    validateActivationSpoolPath(path);
    validateByteOffset(size, "activation truncate size");
    const handle = await this.requireDirectory().getFileHandle(path, { create: true });
    const writable = await handle.createWritable({ keepExistingData: true });
    try {
      await writable.truncate(size);
      await writable.close();
    } catch (error) {
      await writable.abort(error).catch(() => undefined);
      throw error;
    }
  }

  async read(path: string, offset: number, length: number): Promise<Uint8Array> {
    validateActivationSpoolPath(path);
    validateByteOffset(offset, "activation read offset");
    validateByteOffset(length, "activation read length");
    const end = checkedSum(offset, length, "activation read byte range");
    const handle = await this.requireDirectory().getFileHandle(path);
    const file = await handle.getFile();
    if (end > file.size) {
      throw spoolError(
        "STORAGE_MISMATCH",
        `Activation read ends at ${end}, beyond ${file.size} bytes in ${path}`,
      );
    }
    return new Uint8Array(await file.slice(offset, end).arrayBuffer());
  }

  async remove(path: string): Promise<void> {
    validateOwnedEntryName(path);
    try {
      await this.requireDirectory().removeEntry(path, { recursive: true });
    } catch (error) {
      if (!isNotFoundError(error)) throw error;
    }
  }

  async clear(): Promise<void> {
    for (const path of await this.list()) await this.remove(path);
  }

  close(): void {
    this.directory = null;
    this.initialization = null;
  }

  private requireDirectory(): FileSystemDirectoryHandle {
    if (!this.directory) throw new Error("Activation spool files are not initialized");
    return this.directory;
  }
}

export class BrowserActivationSpoolLeasePort implements ActivationSpoolLeasePort {
  async tryAcquire(captureSha256: string): Promise<ActivationSpoolLease | null> {
    validateSha(captureSha256, "capture identity");
    if (typeof navigator === "undefined" || !navigator.locks?.request) {
      throw new Error("The Web Locks API is unavailable");
    }
    let releaseHold!: () => void;
    const hold = new Promise<void>((resolve) => {
      releaseHold = resolve;
    });
    let resolveAcquired!: (acquired: boolean) => void;
    let rejectAcquired!: (error: unknown) => void;
    const acquired = new Promise<boolean>((resolve, reject) => {
      resolveAcquired = resolve;
      rejectAcquired = reject;
    });
    const request = navigator.locks.request(
      `${STORAGE_LOCK}:writer:${captureSha256}`,
      { mode: "exclusive", ifAvailable: true },
      async (lock) => {
        resolveAcquired(lock !== null);
        if (lock) await hold;
      },
    );
    request.catch(rejectAcquired);
    if (!await acquired) {
      await request;
      return null;
    }
    let released = false;
    return {
      release: async () => {
        if (!released) {
          released = true;
          releaseHold();
        }
        await request.catch(() => undefined);
      },
    };
  }
}

export class BrowserActivationSpool extends ActivationSpool {
  constructor(
    options: Omit<ActivationSpoolOptions, "metadata" | "files" | "leases"> & {
      metadata?: ActivationSpoolMetadataPort;
      files?: ActivationSpoolFilePort;
      leases?: ActivationSpoolLeasePort;
    } = {},
  ) {
    super({
      ...options,
      metadata: options.metadata ?? new BrowserActivationSpoolMetadataPort(),
      files: options.files ?? new BrowserActivationSpoolFilePort(),
      leases: options.leases ?? new BrowserActivationSpoolLeasePort(),
    });
  }
}

export async function withBrowserActivationSpool<T>(
  operation: (spool: BrowserActivationSpool) => Promise<T>,
  createSpool: () => BrowserActivationSpool = () => new BrowserActivationSpool(),
): Promise<T> {
  const spool = createSpool();
  try {
    return await operation(spool);
  } finally {
    spool.close();
  }
}

export function validateActivationSpoolDescriptor(
  descriptor: ActivationSpoolDescriptor,
): void {
  validateActivationSpoolIdentity(descriptor);
  if (!Array.isArray(descriptor.layers) || descriptor.layers.length === 0) {
    throw spoolError("INVALID_DESCRIPTOR", "Activation capture must declare at least one layer");
  }
  if (descriptor.layers.length > MAX_LAYERS) {
    throw spoolError(
      "INVALID_DESCRIPTOR",
      `Activation capture declares more than ${MAX_LAYERS} layers`,
    );
  }
  let previous = -1;
  for (const layer of descriptor.layers) {
    validateActivationLayerSpec(layer);
    if (layer.layer <= previous) {
      throw spoolError(
        "INVALID_DESCRIPTOR",
        "Activation layers must be unique and sorted in ascending layer-major order",
      );
    }
    previous = layer.layer;
  }
}

export function validateActivationSpoolIdentity(identity: ActivationSpoolIdentity): void {
  if (!identity || typeof identity !== "object") {
    throw spoolError("INVALID_DESCRIPTOR", "Activation capture identity is invalid");
  }
  validateSha(identity.runtimeIdentitySha256, "runtime identity");
  validateSha(identity.contextBindingSha256, "context binding");
  validateSha(identity.captureSha256, "capture identity");
}

export function validateActivationSpoolRecord(value: unknown): asserts value is ActivationSpoolRecord {
  if (!isRecord(value)) throw new Error("Activation spool record is invalid");
  assertExactKeys(value, [
    "captureSha256",
    "committedAt",
    "contextBindingSha256",
    "createdAt",
    "layers",
    "runtimeIdentitySha256",
    "schemaVersion",
    "state",
    "updatedAt",
    "writeEpoch",
  ]);
  if (value.schemaVersion !== ACTIVATION_SPOOL_SCHEMA_VERSION) {
    throw new Error("Activation spool schema is unsupported");
  }
  validateActivationSpoolIdentity(value as unknown as ActivationSpoolIdentity);
  if (value.state !== "staged" && value.state !== "committed") {
    throw new Error("Activation spool state is invalid");
  }
  if (!Array.isArray(value.layers) || value.layers.length === 0 || value.layers.length > MAX_LAYERS) {
    throw new Error("Activation spool layer roster is invalid");
  }
  let previous = -1;
  for (const candidate of value.layers) {
    validateActivationLayerRecord(candidate);
    if (candidate.layer <= previous) throw new Error("Activation spool layers are not ordered");
    previous = candidate.layer;
  }
  validateStoredTimestamp(value.createdAt, "created timestamp");
  validateStoredTimestamp(value.updatedAt, "updated timestamp");
  if (value.updatedAt < value.createdAt) throw new Error("Activation spool timestamps are invalid");
  if (value.committedAt !== null) {
    validateStoredTimestamp(value.committedAt, "commit timestamp");
    if (value.committedAt < value.createdAt || value.updatedAt < value.committedAt) {
      throw new Error("Activation spool commit timestamp is invalid");
    }
  }
  if (value.writeEpoch !== null) validateEpoch(value.writeEpoch);
  if (value.state === "committed") {
    if (value.writeEpoch !== null || value.committedAt === null) {
      throw new Error("Committed activation spool ownership is invalid");
    }
    if (value.layers.some((layer) => !layer.sealed)) {
      throw new Error("Committed activation spool contains an unsealed layer");
    }
  } else if (value.committedAt !== null) {
    throw new Error("Staged activation spool has a commit timestamp");
  }
}

export function activationSpoolFileName(captureSha256: string, layer: number): string {
  validateSha(captureSha256, "capture identity");
  validateLayerIndex(layer);
  return `${captureSha256}.layer-${layer}.f32`;
}

let fallbackLockTail: Promise<void> = Promise.resolve();
const fallbackWriterLeases = new Set<string>();

const fallbackLeasePort: ActivationSpoolLeasePort = {
  async tryAcquire(captureSha256) {
    validateSha(captureSha256, "capture identity");
    if (fallbackWriterLeases.has(captureSha256)) return null;
    fallbackWriterLeases.add(captureSha256);
    let released = false;
    return {
      async release() {
        if (released) return;
        released = true;
        fallbackWriterLeases.delete(captureSha256);
      },
    };
  },
};

const fallbackExclusiveRunner: ActivationSpoolExclusiveRunner = async <T>(
  operation: () => Promise<T>,
): Promise<T> => {
  const predecessor = fallbackLockTail;
  let release!: () => void;
  fallbackLockTail = new Promise<void>((resolve) => {
    release = resolve;
  });
  await predecessor;
  try {
    return await operation();
  } finally {
    release();
  }
};

const defaultExclusiveRunner: ActivationSpoolExclusiveRunner = <T>(
  operation: () => Promise<T>,
): Promise<T> => {
  if (typeof navigator !== "undefined") {
    if (!navigator.locks?.request) {
      return Promise.reject(new Error("The Web Locks API is unavailable"));
    }
    return navigator.locks.request(
      STORAGE_LOCK,
      { mode: "exclusive" },
      async (lock) => {
        if (!lock) throw new Error("The activation-spool lock was not acquired");
        return operation();
      },
    );
  }
  return fallbackExclusiveRunner(operation);
};

function validateActivationLayerSpec(layer: ActivationLayerSpec): void {
  if (!isRecord(layer)) {
    throw spoolError("INVALID_DESCRIPTOR", "Activation layer descriptor is invalid");
  }
  validateLayerIndex(layer.layer);
  if (!Number.isSafeInteger(layer.rows) || layer.rows <= 0) {
    throw spoolError("INVALID_DESCRIPTOR", `Activation layer ${layer.layer} row count is invalid`);
  }
  if (!Number.isSafeInteger(layer.width) || layer.width <= 0) {
    throw spoolError("INVALID_DESCRIPTOR", `Activation layer ${layer.layer} width is invalid`);
  }
  const rowBytes = checkedProduct(layer.width, FLOAT32_BYTES, "activation row byte size");
  if (rowBytes > ACTIVATION_SPOOL_MAX_CHUNK_BYTES) {
    throw spoolError(
      "INVALID_DESCRIPTOR",
      `Activation layer ${layer.layer} rows exceed the ${ACTIVATION_SPOOL_MAX_CHUNK_BYTES}-byte reader bound`,
    );
  }
  const expected = checkedProduct(layer.rows, rowBytes, "activation layer byte size");
  if (layer.expectedBytes !== expected) {
    throw spoolError(
      "INVALID_DESCRIPTOR",
      `Activation layer ${layer.layer} declares ${layer.expectedBytes} bytes, expected ${expected}`,
    );
  }
}

function validateActivationLayerRecord(value: unknown): asserts value is ActivationLayerRecord {
  if (!isRecord(value)) throw new Error("Activation layer record is invalid");
  assertExactKeys(value, [
    "checkpointBytes",
    "checkpointRows",
    "expectedBytes",
    "layer",
    "rows",
    "sealed",
    "width",
    "writtenBytes",
    "writtenRows",
  ]);
  validateActivationLayerSpec(value as unknown as ActivationLayerSpec);
  const rowBytes = value.width * FLOAT32_BYTES;
  for (const [name, rows] of [
    ["written", value.writtenRows],
    ["checkpoint", value.checkpointRows],
  ] as const) {
    if (!Number.isSafeInteger(rows) || rows < 0 || rows > value.rows) {
      throw new Error(`Activation layer ${value.layer} ${name} row count is invalid`);
    }
  }
  if (value.checkpointRows > value.writtenRows) {
    throw new Error(`Activation layer ${value.layer} checkpoint is ahead of its write tail`);
  }
  if (value.writtenBytes !== value.writtenRows * rowBytes) {
    throw new Error(`Activation layer ${value.layer} written byte count is invalid`);
  }
  if (value.checkpointBytes !== value.checkpointRows * rowBytes) {
    throw new Error(`Activation layer ${value.layer} checkpoint byte count is invalid`);
  }
  if (typeof value.sealed !== "boolean") {
    throw new Error(`Activation layer ${value.layer} seal state is invalid`);
  }
  if (
    value.sealed &&
    (
      value.writtenRows !== value.rows ||
      value.checkpointRows !== value.rows ||
      value.writtenBytes !== value.expectedBytes ||
      value.checkpointBytes !== value.expectedBytes
    )
  ) {
    throw new Error(`Activation layer ${value.layer} is sealed at the wrong size`);
  }
}

function assertIdentity(
  record: ActivationSpoolIdentity,
  expected: ActivationSpoolIdentity,
): void {
  if (
    record.runtimeIdentitySha256 !== expected.runtimeIdentitySha256 ||
    record.contextBindingSha256 !== expected.contextBindingSha256 ||
    record.captureSha256 !== expected.captureSha256
  ) {
    throw spoolError(
      "IDENTITY_MISMATCH",
      `Activation capture ${expected.captureSha256} does not match the exact runtime and context identity`,
    );
  }
}

function assertLayerSpecs(
  actual: readonly ActivationLayerRecord[],
  expected: readonly ActivationLayerSpec[],
): void {
  if (
    actual.length !== expected.length ||
    actual.some((layer, index) => {
      const candidate = expected[index];
      return !candidate ||
        layer.layer !== candidate.layer ||
        layer.rows !== candidate.rows ||
        layer.width !== candidate.width ||
        layer.expectedBytes !== candidate.expectedBytes;
    })
  ) {
    throw spoolError(
      "LAYER_MISMATCH",
      "Activation capture layer shape does not match the staged record",
    );
  }
}

function requireLayer(
  record: ActivationSpoolRecord,
  layerIndex: number,
): ActivationLayerRecord {
  validateLayerIndex(layerIndex);
  const layer = record.layers.find((candidate) => candidate.layer === layerIndex);
  if (!layer) {
    throw spoolError(
      "LAYER_MISMATCH",
      `Activation capture does not declare layer ${layerIndex}`,
    );
  }
  return layer;
}

function encodeFiniteFloat32(values: Float32Array): Uint8Array {
  const bytes = new Uint8Array(values.length * FLOAT32_BYTES);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) {
      throw spoolError(
        "NONFINITE_VALUES",
        `Activation rows contain a non-finite value at index ${index}`,
      );
    }
    view.setFloat32(index * FLOAT32_BYTES, value, true);
  }
  return bytes;
}

function decodeFiniteFloat32(
  bytes: Uint8Array,
  layer: number,
  startRow: number,
): Float32Array {
  if (bytes.byteLength % FLOAT32_BYTES !== 0) {
    throw spoolError("STORAGE_MISMATCH", `Activation layer ${layer} has a partial float32 value`);
  }
  const values = new Float32Array(bytes.byteLength / FLOAT32_BYTES);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < values.length; index += 1) {
    const value = view.getFloat32(index * FLOAT32_BYTES, true);
    if (!Number.isFinite(value)) {
      throw spoolError(
        "NONFINITE_VALUES",
        `Stored activation layer ${layer} has a non-finite value near row ${startRow}`,
      );
    }
    values[index] = value;
  }
  return values;
}

function cloneIdentity(record: ActivationSpoolIdentity): ActivationSpoolIdentity {
  return {
    runtimeIdentitySha256: record.runtimeIdentitySha256,
    contextBindingSha256: record.contextBindingSha256,
    captureSha256: record.captureSha256,
  };
}

function cloneLayerSpec(layer: ActivationLayerSpec): ActivationLayerSpec {
  return {
    layer: layer.layer,
    rows: layer.rows,
    width: layer.width,
    expectedBytes: layer.expectedBytes,
  };
}

function cloneDescriptor(descriptor: ActivationSpoolDescriptor): ActivationSpoolDescriptor {
  return {
    ...cloneIdentity(descriptor),
    layers: descriptor.layers.map(cloneLayerSpec),
  };
}

function cloneReadOptions(options: ActivationSpoolReadOptions): ActivationSpoolReadOptions {
  return {
    startRow: options.startRow,
    rowCount: options.rowCount,
    maxChunkBytes: options.maxChunkBytes,
    signal: options.signal,
  };
}

function cloneLayer(layer: ActivationLayerRecord): ActivationLayerRecord {
  return {
    ...cloneLayerSpec(layer),
    writtenRows: layer.writtenRows,
    writtenBytes: layer.writtenBytes,
    checkpointRows: layer.checkpointRows,
    checkpointBytes: layer.checkpointBytes,
    sealed: layer.sealed,
  };
}

function cloneRecord(record: ActivationSpoolRecord): ActivationSpoolRecord {
  return {
    ...cloneIdentity(record),
    schemaVersion: ACTIVATION_SPOOL_SCHEMA_VERSION,
    state: record.state,
    layers: record.layers.map(cloneLayer),
    writeEpoch: record.writeEpoch,
    createdAt: record.createdAt,
    updatedAt: record.updatedAt,
    committedAt: record.committedAt,
  };
}

function validateSha(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string" || !SHA256.test(value)) {
    throw spoolError("INVALID_DESCRIPTOR", `Activation ${name} SHA-256 is invalid`);
  }
}

function validateLayerIndex(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw spoolError("INVALID_DESCRIPTOR", "Activation layer index is invalid");
  }
}

function validateRowIndex(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw spoolError("INVALID_DESCRIPTOR", `Activation ${name} is invalid`);
  }
}

function validateByteOffset(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw spoolError("INVALID_DESCRIPTOR", `${name} is invalid`);
  }
}

function validateEpoch(value: unknown): asserts value is string {
  if (typeof value !== "string" || value.length === 0 || value.length > 256) {
    throw spoolError("INVALID_DESCRIPTOR", "Activation spool write epoch is invalid");
  }
}

function validateStoredTimestamp(value: unknown, name: string): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`Activation spool ${name} is invalid`);
  }
}

function checkedProduct(left: number, right: number, name: string): number {
  const product = left * right;
  if (!Number.isSafeInteger(product)) {
    throw spoolError("INVALID_DESCRIPTOR", `${name} exceeds the safe integer range`);
  }
  return product;
}

function checkedSum(left: number, right: number, name: string): number {
  const sum = left + right;
  if (!Number.isSafeInteger(sum)) {
    throw spoolError("INVALID_DESCRIPTOR", `${name} exceeds the safe integer range`);
  }
  return sum;
}

function assertExactKeys(value: Record<string, unknown>, expected: string[]): void {
  const keys = Object.keys(value).sort();
  const sorted = [...expected].sort();
  if (keys.length !== sorted.length || keys.some((key, index) => key !== sorted[index])) {
    throw new Error("Activation spool record fields are invalid");
  }
}

function isRecord(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isActivationSpoolFileName(path: string): boolean {
  return /^[a-f0-9]{64}\.layer-[0-9]+\.f32$/.test(path);
}

function validateActivationSpoolPath(path: string): void {
  if (!isActivationSpoolFileName(path)) {
    throw spoolError("INVALID_DESCRIPTOR", "Activation spool path is invalid");
  }
}

function validateOwnedEntryName(path: string): void {
  if (
    typeof path !== "string" || path.length === 0 || path.length > 255 ||
    path === "." || path === ".." || path.includes("/") || path.includes("\\") ||
    path.includes("\0")
  ) {
    throw spoolError("INVALID_DESCRIPTOR", "Activation spool entry name is invalid");
  }
}

function isRecoverableCorruption(error: unknown): boolean {
  return error instanceof ActivationSpoolError &&
    (error.code === "CORRUPT_RECORD" || error.code === "STORAGE_MISMATCH");
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return;
  throw signal.reason ?? new DOMException("The activation read was aborted", "AbortError");
}

function defaultEpoch(): string {
  return randomUuid();
}

async function openActivationSpoolDatabase(): Promise<IDBDatabase> {
  if (typeof indexedDB === "undefined") throw new Error("IndexedDB is unavailable");
  const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
  request.onupgradeneeded = () => {
    const database = request.result;
    if (!database.objectStoreNames.contains(CAPTURES_STORE)) {
      database.createObjectStore(CAPTURES_STORE, { keyPath: "captureSha256" });
    }
  };
  const database = await new Promise<IDBDatabase>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Activation spool database open failed"));
    request.onblocked = () => reject(
      new Error("Another Drowse tab is blocking the activation spool upgrade"),
    );
  });
  await migrateLegacyDatabase(database, [CAPTURES_STORE]);
  return database;
}

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB request failed"));
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(transaction.error ?? new Error("IndexedDB transaction aborted"));
    transaction.onerror = () => reject(transaction.error ?? new Error("IndexedDB transaction failed"));
  });
}

function isNotFoundError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "NotFoundError";
}

function isTypeMismatchError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "TypeMismatchError";
}

function spoolError(
  code: ConstructorParameters<typeof ActivationSpoolError>[0],
  message: string,
): ActivationSpoolError {
  return new ActivationSpoolError(code, message);
}
