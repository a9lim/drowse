import type {
  CatalogInstrumentPack,
  ModelVariant,
  RuntimeCapabilities,
} from "../../lib/runtime/contracts";
import { MEBIBYTE } from "../../lib/runtime/contracts";

const FP32_BYTES = 4;
const EXACT_READOUT_TOP_K = 8;
const MAX_RUNTIME_CHUNK_BYTES = 32 * MEBIBYTE;

export const OPTIONAL_PACK_RESIDENT_BUDGET_BYTES = 1024 * MEBIBYTE;

export interface OptionalPackHardwareBlock {
  code:
    | "JLENS_DEVICE_BUFFER_LIMIT"
    | "SAE_DEVICE_BUFFER_LIMIT"
    | "OPTIONAL_PACK_RESIDENT_BUDGET_EXCEEDED";
  message: string;
  requiredBytes: number;
  limitBytes: number;
  packIds: string[];
}

export function optionalPackHardwareBlock(
  pack: CatalogInstrumentPack,
  variant: ModelVariant,
  capabilities: RuntimeCapabilities,
  concurrentlySelected: readonly CatalogInstrumentPack[] = [],
): OptionalPackHardwareBlock | null {
  const available = uniqueOptionalPacks([...concurrentlySelected, pack]);
  for (const candidate of available) {
    const block = perBufferBlock(candidate, variant, capabilities);
    if (block !== null) return block;
  }
  const selected = simultaneouslyResidentPacks(available, variant);

  const requiredBytes = selected.reduce(
    (total, candidate) => total + conservativeResidentBytes(candidate, variant),
    0,
  );
  if (
    Number.isSafeInteger(requiredBytes) && requiredBytes > 0 &&
    requiredBytes <= OPTIONAL_PACK_RESIDENT_BUDGET_BYTES
  ) return null;

  const packIds = selected.map((candidate) => candidate.id);
  return {
    code: "OPTIONAL_PACK_RESIDENT_BUDGET_EXCEEDED",
    message:
      "These model tools need more graphics memory than Drowse can safely reserve at once. Remove one installed tool, or choose a smaller model or tool.",
    requiredBytes,
    limitBytes: OPTIONAL_PACK_RESIDENT_BUDGET_BYTES,
    packIds,
  };
}

function perBufferBlock(
  pack: CatalogInstrumentPack,
  variant: ModelVariant,
  capabilities: RuntimeCapabilities,
): OptionalPackHardwareBlock | null {
  if (pack.kind !== "jlens" && pack.kind !== "sae") return null;
  const hiddenSize = variant.runtimeIdentity.hiddenSize;
  const requiredBytes = pack.kind === "jlens"
    ? hiddenSize * hiddenSize * FP32_BYTES
    : Math.max(
        hiddenSize * EXACT_READOUT_TOP_K * FP32_BYTES,
        Math.min(largestFileBytes(pack), MAX_RUNTIME_CHUNK_BYTES),
      );
  const limitBytes = Math.min(adapterBufferLimit(capabilities), MAX_RUNTIME_CHUNK_BYTES);
  if (
    Number.isSafeInteger(requiredBytes) &&
    requiredBytes > 0 &&
    limitBytes > 0 &&
    limitBytes >= requiredBytes
  ) return null;
  return {
    code: pack.kind === "jlens"
      ? "JLENS_DEVICE_BUFFER_LIMIT"
      : "SAE_DEVICE_BUFFER_LIMIT",
    message:
      `${pack.displayName} needs a larger graphics buffer than this browser can provide. Use the model without this tool, or choose a smaller compatible model or tool.`,
    requiredBytes,
    limitBytes,
    packIds: [pack.id],
  };
}

function largestFileBytes(pack: CatalogInstrumentPack): number {
  return pack.files.reduce((largest, file) => Math.max(largest, file.bytes), 0);
}

function adapterBufferLimit(capabilities: RuntimeCapabilities): number {
  const limits = [
    capabilities.webGpu.limits.maxBufferSize,
    capabilities.webGpu.limits.maxStorageBufferBindingSize,
  ];
  if (limits.some((value) => !Number.isSafeInteger(value) || value <= 0)) return 0;
  return Math.min(...limits);
}

function conservativeResidentBytes(
  pack: CatalogInstrumentPack,
  variant: ModelVariant,
): number {
  if (pack.kind === "jlens") {
    const oneMatrixBytes =
      variant.runtimeIdentity.hiddenSize * variant.runtimeIdentity.hiddenSize * FP32_BYTES;
    return Math.max(pack.bytes, oneMatrixBytes);
  }
  if (pack.kind === "sae") {
    const minimumEncoderChunkBytes =
      variant.runtimeIdentity.hiddenSize * EXACT_READOUT_TOP_K * FP32_BYTES;
    return Math.max(pack.bytes, minimumEncoderChunkBytes);
  }
  return 0;
}

function uniqueOptionalPacks(
  packs: readonly CatalogInstrumentPack[],
): CatalogInstrumentPack[] {
  const selected = new Map<string, CatalogInstrumentPack>();
  for (const pack of packs) {
    if (pack.kind === "jlens" || pack.kind === "sae") selected.set(pack.id, pack);
  }
  return [...selected.values()];
}

function simultaneouslyResidentPacks(
  packs: readonly CatalogInstrumentPack[],
  variant: ModelVariant,
): CatalogInstrumentPack[] {
  const selected = new Map<CatalogInstrumentPack["kind"], CatalogInstrumentPack>();
  for (const pack of packs) {
    const current = selected.get(pack.kind);
    if (
      current === undefined ||
      conservativeResidentBytes(pack, variant) > conservativeResidentBytes(current, variant)
    ) {
      selected.set(pack.kind, pack);
    }
  }
  return [...selected.values()];
}
