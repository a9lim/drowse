export const DROWSE_HOOK_ABI = "post-block-residual-v4" as const;

export interface RankOneHookLayer {
  enabled: boolean;
  basis: ArrayLike<number>;
  neutral: ArrayLike<number>;
  target: number;
  along: number;
  collapse: number;
  probeBasis: ArrayLike<number>;
  probeNeutral: ArrayLike<number>;
}

export interface RankOneHookProgramBuffers {
  hookAbi: typeof DROWSE_HOOK_ABI;
  activeRole?: string | null;
  hiddenSize: number;
  layerCount: number;
  enabled: Uint32Array;
  basis: Float32Array;
  neutral: Float32Array;
  target: Float32Array;
  along: Float32Array;
  collapse: Float32Array;
  probeBasis: Float32Array;
  probeNeutral: Float32Array;
}

export interface RankOneHookResult {
  coordinate: number;
  residual: Float32Array;
  probe: number;
}

export function compileRankOneHookProgram(
  hiddenSize: number,
  layers: readonly RankOneHookLayer[],
): RankOneHookProgramBuffers {
  if (!Number.isSafeInteger(hiddenSize) || hiddenSize <= 0) {
    throw new TypeError("Hook hidden size must be a positive integer");
  }
  if (layers.length === 0) {
    throw new TypeError("Hook program must contain at least one layer");
  }
  const layerCount = layers.length;
  const enabled = new Uint32Array(layerCount);
  const basis = new Float32Array(layerCount * hiddenSize);
  const neutral = new Float32Array(layerCount * hiddenSize);
  const target = new Float32Array(layerCount);
  const along = new Float32Array(layerCount);
  const collapse = new Float32Array(layerCount);
  const probeBasis = new Float32Array(layerCount * hiddenSize);
  const probeNeutral = new Float32Array(layerCount * hiddenSize);

  layers.forEach((layer, layerIndex) => {
    enabled[layerIndex] = layer.enabled ? 1 : 0;
    copyVector(layer.basis, basis, layerIndex * hiddenSize, hiddenSize, "basis");
    copyVector(layer.neutral, neutral, layerIndex * hiddenSize, hiddenSize, "neutral");
    copyVector(
      layer.probeBasis,
      probeBasis,
      layerIndex * hiddenSize,
      hiddenSize,
      "probe basis",
    );
    copyVector(
      layer.probeNeutral,
      probeNeutral,
      layerIndex * hiddenSize,
      hiddenSize,
      "probe neutral",
    );
    target[layerIndex] = finiteFloat(layer.target, "target");
    along[layerIndex] = finiteFloat(layer.along, "along");
    collapse[layerIndex] = finiteFloat(layer.collapse, "collapse");
    if (layer.enabled) {
      requireUnitVector(basis, layerIndex * hiddenSize, hiddenSize, "basis");
    }
    requireUnitVector(
      probeBasis,
      layerIndex * hiddenSize,
      hiddenSize,
      "probe basis",
      true,
    );
  });

  return {
    hookAbi: DROWSE_HOOK_ABI,
    hiddenSize,
    layerCount,
    enabled,
    basis,
    neutral,
    target,
    along,
    collapse,
    probeBasis,
    probeNeutral,
  };
}

export function runRankOneHookLayer(
  program: RankOneHookProgramBuffers,
  layerIndex: number,
  residual: ArrayLike<number>,
): RankOneHookResult {
  validateProgramShape(program);
  if (!Number.isSafeInteger(layerIndex) || layerIndex < 0 || layerIndex >= program.layerCount) {
    throw new RangeError(`Hook layer ${layerIndex} is outside the compiled program`);
  }
  if (residual.length !== program.hiddenSize) {
    throw new TypeError(
      `Residual has ${residual.length} values; expected ${program.hiddenSize}`,
    );
  }
  const values = new Float32Array(program.hiddenSize);
  for (let index = 0; index < values.length; index += 1) {
    values[index] = finiteFloat(residual[index], "residual");
  }
  const offset = layerIndex * program.hiddenSize;
  let coordinate = 0;
  for (let index = 0; index < program.hiddenSize; index += 1) {
    coordinate = Math.fround(
      coordinate + Math.fround(
        Math.fround(values[index] - program.neutral[offset + index]) *
          program.basis[offset + index],
      ),
    );
  }
  if (program.enabled[layerIndex] === 1) {
    const delta = Math.fround(
      program.along[layerIndex] * Math.fround(
        program.target[layerIndex] - Math.fround(program.collapse[layerIndex] * coordinate),
      ),
    );
    for (let index = 0; index < program.hiddenSize; index += 1) {
      values[index] = Math.fround(
        values[index] + Math.fround(delta * program.basis[offset + index]),
      );
    }
  }
  let probe = 0;
  for (let index = 0; index < program.hiddenSize; index += 1) {
    probe = Math.fround(
      probe + Math.fround(
        Math.fround(values[index] - program.probeNeutral[offset + index]) *
          program.probeBasis[offset + index],
      ),
    );
  }
  return { coordinate, residual: values, probe };
}

function validateProgramShape(program: RankOneHookProgramBuffers): void {
  if (program.hookAbi !== DROWSE_HOOK_ABI) {
    throw new TypeError(`Unsupported hook ABI ${String(program.hookAbi)}`);
  }
  const vectorValues = program.layerCount * program.hiddenSize;
  if (
    program.enabled.length !== program.layerCount ||
    program.target.length !== program.layerCount ||
    program.along.length !== program.layerCount ||
    program.collapse.length !== program.layerCount ||
    program.basis.length !== vectorValues ||
    program.neutral.length !== vectorValues ||
    program.probeBasis.length !== vectorValues ||
    program.probeNeutral.length !== vectorValues
  ) {
    throw new TypeError("Hook program buffers do not match their declared dimensions");
  }
}

function copyVector(
  source: ArrayLike<number>,
  target: Float32Array,
  offset: number,
  size: number,
  label: string,
): void {
  if (source.length !== size) {
    throw new TypeError(`${label} has ${source.length} values; expected ${size}`);
  }
  for (let index = 0; index < size; index += 1) {
    target[offset + index] = finiteFloat(source[index], label);
  }
}

function finiteFloat(value: number, label: string): number {
  if (!Number.isFinite(value)) throw new TypeError(`${label} values must be finite`);
  const narrowed = Math.fround(value);
  if (!Number.isFinite(narrowed)) throw new TypeError(`${label} exceeds fp32 range`);
  return narrowed;
}

function requireUnitVector(
  values: Float32Array,
  offset: number,
  size: number,
  label: string,
  allowZero = false,
): void {
  let squaredNorm = 0;
  for (let index = 0; index < size; index += 1) {
    squaredNorm += values[offset + index] ** 2;
  }
  if (allowZero && squaredNorm === 0) return;
  if (Math.abs(squaredNorm - 1) > 1e-4) {
    throw new TypeError(`${label} must be unit length`);
  }
}
