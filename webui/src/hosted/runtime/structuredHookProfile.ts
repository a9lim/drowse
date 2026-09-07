import type { StructuredHookProfileId } from "../../lib/runtime/contracts";

export const WEB_LLM_RUNTIME_MIN_STORAGE_BUFFERS_PER_SHADER_STAGE = 10;
export const EXACT_READOUT_MIN_COMPUTE_WORKGROUP_SIZE_X = 256;
export const EXACT_READOUT_MIN_COMPUTE_INVOCATIONS_PER_WORKGROUP = 256;
export const EXACT_READOUT_REQUIRED_WEBGPU_LIMITS = Object.freeze({
  maxComputeWorkgroupSizeX: EXACT_READOUT_MIN_COMPUTE_WORKGROUP_SIZE_X,
  maxComputeInvocationsPerWorkgroup:
    EXACT_READOUT_MIN_COMPUTE_INVOCATIONS_PER_WORKGROUP,
} as const);

export interface StructuredHookCapacityProfile {
  readonly id: StructuredHookProfileId;
  readonly maxAffineGroups: number;
  readonly maxRank: number;
  readonly maxProbes: number;
  readonly maxCurves: number;
  readonly maxCurveNodes: number;
  readonly maxIntrinsicDim: number;
  readonly maxEmbedDim: number;
  readonly maxGeometryProbes: number;
  readonly maxWhitenerRank: number;
  readonly maxGeometryCandidates: number;
  readonly geometryOutputStride: number;
  readonly geometryFootRestarts: number;
  readonly geometryFootIterations: number;
  readonly geometryWarmRestarts: number;
  readonly geometryWarmIterations: number;
  readonly requiredMaxStorageBuffersPerShaderStage: number;
  readonly geometryKernelStorageBindings: number;
  readonly geometryHeaderStride: number;
  readonly maxComputeWorkgroupStorageSize: number;
}

export interface StructuredHookMemoryRequirements {
  readonly curveParameterStride: number;
  readonly hostProgramBytes: number;
  readonly gpuInputBytes: number;
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly maxStorageBuffersPerShaderStage: number;
  readonly maxComputeWorkgroupStorageSize: number;
  readonly geometryHeaderBytes: number;
  readonly geometryPayloadBytes: number;
}

export const STANDARD_STRUCTURED_HOOK_PROFILE: StructuredHookCapacityProfile =
  Object.freeze({
    id: "standard-v1",
    maxAffineGroups: 4,
    maxRank: 8,
    maxProbes: 8,
    maxCurves: 4,
    maxCurveNodes: 32,
    maxIntrinsicDim: 4,
    maxEmbedDim: 8,
    maxGeometryProbes: 0,
    maxWhitenerRank: 0,
    maxGeometryCandidates: 0,
    geometryOutputStride: 0,
    geometryFootRestarts: 0,
    geometryFootIterations: 0,
    geometryWarmRestarts: 0,
    geometryWarmIterations: 0,
    requiredMaxStorageBuffersPerShaderStage: 8,
    geometryKernelStorageBindings: 0,
    geometryHeaderStride: 0,
    maxComputeWorkgroupStorageSize: 32 * 1024,
  });

export const STANDARD_STRUCTURED_HOOK_PROFILE_V2: StructuredHookCapacityProfile =
  Object.freeze({
    id: "standard-v2",
    maxAffineGroups: 4,
    maxRank: 8,
    maxProbes: 8,
    maxCurves: 4,
    maxCurveNodes: 32,
    maxIntrinsicDim: 4,
    maxEmbedDim: 8,
    maxGeometryProbes: 8,
    maxWhitenerRank: 96,
    maxGeometryCandidates: 33,
    geometryOutputStride: 41,
    geometryFootRestarts: 3,
    geometryFootIterations: 12,
    geometryWarmRestarts: 2,
    geometryWarmIterations: 4,
    requiredMaxStorageBuffersPerShaderStage: 8,
    geometryKernelStorageBindings: 8,
    geometryHeaderStride: 7,
    maxComputeWorkgroupStorageSize: 32 * 1024,
  });

export const STANDARD_STRUCTURED_HOOK_PROFILE_V3: StructuredHookCapacityProfile =
  Object.freeze({
    ...STANDARD_STRUCTURED_HOOK_PROFILE_V2,
    id: "standard-v3",
  });

const PROFILE_KEYS = Object.freeze([
  "id",
  "maxAffineGroups",
  "maxRank",
  "maxProbes",
  "maxCurves",
  "maxCurveNodes",
  "maxIntrinsicDim",
  "maxEmbedDim",
  "maxGeometryProbes",
  "maxWhitenerRank",
  "maxGeometryCandidates",
  "geometryOutputStride",
  "geometryFootRestarts",
  "geometryFootIterations",
  "geometryWarmRestarts",
  "geometryWarmIterations",
  "requiredMaxStorageBuffersPerShaderStage",
  "geometryKernelStorageBindings",
  "geometryHeaderStride",
  "maxComputeWorkgroupStorageSize",
] as const);

const PROFILES: ReadonlyMap<StructuredHookProfileId, StructuredHookCapacityProfile> =
  new Map([
    [STANDARD_STRUCTURED_HOOK_PROFILE.id, STANDARD_STRUCTURED_HOOK_PROFILE],
    [STANDARD_STRUCTURED_HOOK_PROFILE_V2.id, STANDARD_STRUCTURED_HOOK_PROFILE_V2],
    [STANDARD_STRUCTURED_HOOK_PROFILE_V3.id, STANDARD_STRUCTURED_HOOK_PROFILE_V3],
  ]);

export function resolveStructuredHookProfile(
  id: unknown,
): StructuredHookCapacityProfile {
  if (typeof id !== "string" || !PROFILES.has(id as StructuredHookProfileId)) {
    throw new TypeError(`Unsupported structured hook capacity profile ${String(id)}`);
  }
  return PROFILES.get(id as StructuredHookProfileId)!;
}

export function validateStructuredHookProfile(
  value: unknown,
): StructuredHookCapacityProfile {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError("Structured hook capacity profile must be an object");
  }
  const candidate = value as Record<string, unknown>;
  const keys = Object.keys(candidate).sort();
  const expectedKeys = [...PROFILE_KEYS].sort();
  if (
    keys.length !== expectedKeys.length ||
    keys.some((key, index) => key !== expectedKeys[index])
  ) {
    throw new TypeError("Structured hook capacity profile has an unsupported shape");
  }
  const profile = resolveStructuredHookProfile(candidate.id);
  for (const key of PROFILE_KEYS) {
    if (candidate[key] !== profile[key]) {
      throw new TypeError(`Structured hook capacity profile ${profile.id} was modified`);
    }
  }
  validateProfileInvariants(profile);
  return profile;
}

export function structuredCurveParameterStride(
  input: StructuredHookCapacityProfile,
): number {
  const profile = validateStructuredHookProfile(input);
  return checkedStructuredHookSum("curve parameter stride", [
    checkedStructuredHookProduct(
      "curve node parameter fields",
      profile.maxCurveNodes,
      profile.maxEmbedDim,
    ),
    checkedStructuredHookProduct(
      "curve RBF fields",
      profile.maxCurveNodes,
      profile.maxRank,
    ),
    checkedStructuredHookProduct(
      "curve polynomial fields",
      profile.maxEmbedDim + 1,
      profile.maxRank,
    ),
    profile.maxEmbedDim,
    profile.maxEmbedDim,
    profile.maxIntrinsicDim,
    profile.maxIntrinsicDim,
    checkedStructuredHookProduct(
      "curve bound fields",
      profile.maxIntrinsicDim,
      2,
    ),
    profile.maxIntrinsicDim,
    profile.maxIntrinsicDim,
    profile.maxCurveNodes,
    profile.maxEmbedDim + 1,
    6,
  ]);
}

export interface StructuredGeometryPayloadLayout {
  readonly offsets: readonly number[];
  readonly elements: number;
}

export function structuredGeometryPayloadLayout(
  input: StructuredHookCapacityProfile,
  layerCount: number,
  hiddenSize: number,
): StructuredGeometryPayloadLayout {
  const profile = validateStructuredHookProfile(input);
  positiveInteger(layerCount, "structured hook layer count");
  positiveInteger(hiddenSize, "structured hook hidden size");
  if (profile.maxGeometryProbes === 0) {
    return Object.freeze({ offsets: Object.freeze([]), elements: 0 });
  }
  const slots = checkedStructuredHookProduct(
    "geometry payload slots",
    layerCount,
    profile.maxGeometryProbes,
  );
  const fieldElements = [
    checkedStructuredHookProduct("geometry payload means", slots, hiddenSize),
    checkedStructuredHookProduct("geometry payload inverse means", slots, hiddenSize),
    checkedStructuredHookProduct("geometry payload basis", slots, profile.maxRank, hiddenSize),
    checkedStructuredHookProduct("geometry payload Gram inverse", slots, profile.maxRank, profile.maxRank),
    checkedStructuredHookProduct("geometry payload Cholesky", slots, profile.maxRank, profile.maxRank),
    checkedStructuredHookProduct(
      "geometry payload candidate coordinates",
      slots,
      profile.maxGeometryCandidates,
      profile.maxRank,
    ),
    checkedStructuredHookProduct(
      "geometry payload coordinate map",
      slots,
      profile.maxIntrinsicDim,
      profile.maxRank,
    ),
    checkedStructuredHookProduct("geometry payload coordinate bias", slots, profile.maxIntrinsicDim),
    checkedStructuredHookProduct(
      "geometry payload curve parameters",
      slots,
      structuredCurveParameterStride(profile),
    ),
    checkedStructuredHookProduct(
      "geometry payload curve node coordinates",
      slots,
      profile.maxCurveNodes,
      profile.maxIntrinsicDim,
    ),
    checkedStructuredHookProduct(
      "geometry payload curve node values",
      slots,
      profile.maxCurveNodes,
      profile.maxRank,
    ),
  ];
  const offsets: number[] = [];
  let elements = 0;
  for (const field of fieldElements) {
    const byteOffset = checkedStructuredHookProduct(
      "geometry payload byte offset",
      elements,
      Float32Array.BYTES_PER_ELEMENT,
    );
    if (byteOffset % Float32Array.BYTES_PER_ELEMENT !== 0) {
      throw new RangeError("geometry payload fields are not float32-aligned");
    }
    offsets.push(elements);
    elements = checkedStructuredHookSum("geometry payload elements", [elements, field]);
  }
  return Object.freeze({ offsets: Object.freeze(offsets), elements });
}

export function structuredHookMemoryRequirements(
  input: StructuredHookCapacityProfile,
  layerCount: number,
  hiddenSize: number,
  options: { includeJlens?: boolean } = {},
): StructuredHookMemoryRequirements {
  const profile = validateStructuredHookProfile(input);
  positiveInteger(layerCount, "structured hook layer count");
  positiveInteger(hiddenSize, "structured hook hidden size");

  const l = layerCount;
  const h = hiddenSize;
  const g = profile.maxAffineGroups;
  const r = profile.maxRank;
  const p = profile.maxProbes;
  const c = profile.maxCurves;
  const n = profile.maxCurveNodes;
  const i = profile.maxIntrinsicDim;
  const e = profile.maxEmbedDim;
  const geometryProbes = profile.maxGeometryProbes;
  const whitenerRank = profile.maxWhitenerRank;
  const geometryCandidates = profile.maxGeometryCandidates;
  const stride = structuredCurveParameterStride(profile);
  const lg = checkedStructuredHookProduct("affine slots", l, g);
  const lp = checkedStructuredHookProduct("probe slots", l, p);
  const lc = checkedStructuredHookProduct("curve slots", l, c);
  const geometrySlots = checkedStructuredHookProduct(
    "geometry slots",
    l,
    geometryProbes,
  );
  const geometryPayload = structuredGeometryPayloadLayout(profile, l, h);
  const geometryHeaderElements = checkedStructuredHookProduct(
    "geometry header elements",
    geometrySlots,
    profile.geometryHeaderStride,
  );

  const hostElements = [
    lg,
    checkedStructuredHookProduct("affine basis elements", l, g, r, h),
    checkedStructuredHookProduct("affine neutral elements", l, g, h),
    checkedStructuredHookProduct("affine target elements", l, g, r),
    lg,
    checkedStructuredHookProduct("affine kappa elements", l, g, r),
    lp,
    checkedStructuredHookProduct("probe direction elements", l, p, h),
    lp,
    checkedStructuredHookProduct("curve scalar metadata elements", lc, 5),
    checkedStructuredHookProduct("curve basis elements", l, c, r, h),
    checkedStructuredHookProduct("curve neutral elements", l, c, h),
    checkedStructuredHookProduct("curve domain kind elements", l, c),
    checkedStructuredHookProduct("curve node parameter elements", l, c, n, e),
    checkedStructuredHookProduct("curve RBF elements", l, c, n, r),
    checkedStructuredHookProduct("curve polynomial elements", l, c, e + 1, r),
    checkedStructuredHookProduct("curve coordinate elements", l, c, e, 2),
    checkedStructuredHookProduct("curve point elements", l, c, i, 2),
    checkedStructuredHookProduct("curve along and onto elements", lc, 2),
    checkedStructuredHookProduct("curve bound elements", l, c, i, 2),
    checkedStructuredHookProduct("curve axis metadata elements", l, c, i, 2),
    lc,
    checkedStructuredHookProduct("curve sigma RBF elements", l, c, n),
    checkedStructuredHookProduct("curve sigma polynomial elements", l, c, e + 1),
    lc,
  ];
  if (geometryProbes > 0) {
    hostElements.push(
      l,
      l,
      checkedStructuredHookProduct(
        "geometry whitener basis elements",
        l,
        whitenerRank,
        h,
      ),
      checkedStructuredHookProduct(
        "geometry whitener correction elements",
        l,
        whitenerRank,
      ),
      checkedStructuredHookProduct("geometry scalar metadata elements", geometrySlots, 7),
      checkedStructuredHookProduct("geometry mean elements", geometrySlots, h),
      checkedStructuredHookProduct("geometry inverse mean elements", geometrySlots, h),
      checkedStructuredHookProduct("geometry basis elements", geometrySlots, r, h),
      checkedStructuredHookProduct("geometry Gram inverse elements", geometrySlots, r, r),
      checkedStructuredHookProduct("geometry Cholesky elements", geometrySlots, r, r),
      checkedStructuredHookProduct(
        "geometry node white elements",
        geometrySlots,
        geometryCandidates,
        r,
      ),
      checkedStructuredHookProduct("geometry coordinate map elements", geometrySlots, i, r),
      checkedStructuredHookProduct("geometry coordinate bias elements", geometrySlots, i),
      checkedStructuredHookProduct("geometry curve parameter elements", geometrySlots, stride),
      checkedStructuredHookProduct("geometry curve node coordinate elements", geometrySlots, n, i),
      checkedStructuredHookProduct("geometry curve node value elements", geometrySlots, n, r),
      checkedStructuredHookProduct("geometry foot elements", geometrySlots, i),
    );
  }
  if (options.includeJlens === true) {
    hostElements.push(
      checkedStructuredHookProduct("J-lens Jacobian elements", l, h, h),
      p,
    );
  }

  const gpuInputElements = [
    ...hostElements.slice(0, 9),
    checkedStructuredHookProduct("GPU curve basis elements", l, c, r, h),
    checkedStructuredHookProduct("GPU curve neutral elements", l, c, h),
    checkedStructuredHookProduct("GPU curve domain kind elements", l, c),
    checkedStructuredHookProduct("GPU curve parameter elements", l, c, stride),
    checkedStructuredHookProduct("GPU curve foot elements", l, c, i),
  ];
  if (geometryProbes > 0) {
    gpuInputElements.push(
      l,
      l,
      checkedStructuredHookProduct(
        "GPU geometry whitener basis elements",
        l,
        whitenerRank,
        h,
      ),
      checkedStructuredHookProduct(
        "GPU geometry whitener correction elements",
        l,
        whitenerRank,
      ),
      geometryHeaderElements,
      geometryPayload.elements,
      checkedStructuredHookProduct("GPU geometry foot elements", geometrySlots, i),
    );
  }
  if (options.includeJlens === true) {
    gpuInputElements.push(
      checkedStructuredHookProduct("GPU J-lens Jacobian elements", h, h),
      p,
      1,
    );
  }
  const outputElements = [
    lp,
    ...(geometryProbes > 0
      ? [
          checkedStructuredHookProduct(
            "GPU geometry output elements",
            geometrySlots,
            profile.geometryOutputStride,
          ),
        ]
      : []),
    ...(options.includeJlens === true
      ? [
          checkedStructuredHookProduct("GPU J-lens hidden elements", l, h),
          p,
          checkedStructuredHookProduct("GPU J-lens direction elements", p, h),
        ]
      : []),
  ];
  const largestElements = Math.max(...gpuInputElements, ...outputElements);
  const largestBytes = checkedStructuredHookProduct(
    "largest structured hook buffer bytes",
    largestElements,
    Float32Array.BYTES_PER_ELEMENT,
  );
  const geometryHeaderBytes = checkedStructuredHookProduct(
    "geometry header bytes",
    geometryHeaderElements,
    Uint32Array.BYTES_PER_ELEMENT,
  );
  const geometryPayloadBytes = checkedStructuredHookProduct(
    "geometry payload bytes",
    geometryPayload.elements,
    Float32Array.BYTES_PER_ELEMENT,
  );
  return {
    curveParameterStride: stride,
    hostProgramBytes: checkedStructuredHookProduct(
      "structured hook host program bytes",
      checkedStructuredHookSum("structured hook host program elements", hostElements),
      Float32Array.BYTES_PER_ELEMENT,
    ),
    gpuInputBytes: checkedStructuredHookProduct(
      "structured hook GPU input bytes",
      checkedStructuredHookSum("structured hook GPU input elements", gpuInputElements),
      Float32Array.BYTES_PER_ELEMENT,
    ),
    maxBufferSize: largestBytes,
    maxStorageBufferBindingSize: largestBytes,
    maxStorageBuffersPerShaderStage: profile.requiredMaxStorageBuffersPerShaderStage,
    maxComputeWorkgroupStorageSize: profile.maxComputeWorkgroupStorageSize,
    geometryHeaderBytes,
    geometryPayloadBytes,
  };
}

export function checkedStructuredHookProduct(
  label: string,
  ...factors: number[]
): number {
  let value = 1;
  for (const factor of factors) {
    if (!Number.isSafeInteger(factor) || factor < 0) {
      throw new RangeError(`${label} has an invalid factor`);
    }
    value *= factor;
    if (!Number.isSafeInteger(value)) {
      throw new RangeError(`${label} exceeds safe integer arithmetic`);
    }
  }
  return value;
}

function checkedStructuredHookSum(label: string, terms: readonly number[]): number {
  let value = 0;
  for (const term of terms) {
    if (!Number.isSafeInteger(term) || term < 0) {
      throw new RangeError(`${label} has an invalid term`);
    }
    value += term;
    if (!Number.isSafeInteger(value)) {
      throw new RangeError(`${label} exceeds safe integer arithmetic`);
    }
  }
  return value;
}

function validateProfileInvariants(profile: StructuredHookCapacityProfile): void {
  const geometryKeys = new Set<keyof StructuredHookCapacityProfile>([
    "maxGeometryProbes",
    "maxWhitenerRank",
    "maxGeometryCandidates",
    "geometryOutputStride",
    "geometryFootRestarts",
    "geometryFootIterations",
    "geometryWarmRestarts",
    "geometryWarmIterations",
    "geometryKernelStorageBindings",
    "geometryHeaderStride",
  ]);
  for (const key of PROFILE_KEYS.slice(1)) {
    if (geometryKeys.has(key) && profile.id === "standard-v1") {
      if (profile[key] !== 0) {
        throw new TypeError("Structured hook profile standard-v1 cannot declare geometry buffers");
      }
      continue;
    }
    positiveInteger(profile[key], `structured hook profile ${key}`);
  }
  if (profile.maxCurveNodes < 2) {
    throw new TypeError("Structured hook profiles require at least two curve nodes");
  }
  if (profile.maxRank < profile.maxIntrinsicDim) {
    throw new TypeError("Structured hook profiles require rank to cover intrinsic dimensions");
  }
  if (profile.maxEmbedDim < profile.maxIntrinsicDim * 2) {
    throw new TypeError("Structured hook profiles require two embedding slots per intrinsic axis");
  }
  if (profile.requiredMaxStorageBuffersPerShaderStage !== 8) {
    throw new TypeError("Structured hook profiles must fit WebGPU's guaranteed storage binding limit");
  }
  if (profile.id === "standard-v2" || profile.id === "standard-v3") {
    if (profile.maxGeometryCandidates !== profile.maxCurveNodes + 1) {
      throw new TypeError("Structured hook geometry requires one neutral candidate slot");
    }
    if (
      profile.geometryOutputStride !==
        4 + profile.maxIntrinsicDim + profile.maxGeometryCandidates
    ) {
      throw new TypeError("Structured hook geometry output stride is inconsistent");
    }
    if (
      profile.geometryHeaderStride !== 7 ||
      profile.geometryKernelStorageBindings !== 8 ||
      profile.geometryKernelStorageBindings > profile.requiredMaxStorageBuffersPerShaderStage
    ) {
      throw new TypeError("Structured hook geometry exceeds the WebGPU storage binding budget");
    }
  }
}

function positiveInteger(value: unknown, label: string): asserts value is number {
  if (!Number.isSafeInteger(value) || (value as number) < 1) {
    throw new TypeError(`${label} must be a positive integer`);
  }
}
