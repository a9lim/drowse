import { DROWSE_HOOK_ABI } from "./rankOneHookProgram";
import type {
  SteeringComparisonOperator,
  SteeringPhase,
} from "./steeringExpression";
import {
  checkedStructuredHookProduct,
  structuredCurveParameterStride,
  validateStructuredHookProfile,
  type StructuredHookCapacityProfile,
} from "./structuredHookProfile";

export const STRUCTURED_HOOK_FORMAT_V2 = "drowse-structured-v2" as const;
export const STRUCTURED_HOOK_FORMAT = "drowse-structured-v3" as const;
export type StructuredHookFormat =
  | typeof STRUCTURED_HOOK_FORMAT_V2
  | typeof STRUCTURED_HOOK_FORMAT;

export type StructuredProbeKind =
  | "disabled"
  | "linear"
  | "sae"
  | "sae_jump_relu"
  | "jlens";

export interface StructuredAffineGroup {
  active: boolean;
  basis: readonly ArrayLike<number>[];
  neutral: ArrayLike<number>;
  target: ArrayLike<number>;
  along: number;
  kappa: ArrayLike<number>;
}

export interface StructuredProbe {
  kind: StructuredProbeKind;
  direction: ArrayLike<number>;
  bias: number;
  threshold?: number;
}

export interface StructuredCurve {
  active: boolean;
  domainKind?: "box" | "custom" | "sphere";
  basis: readonly ArrayLike<number>[];
  neutral: ArrayLike<number>;
  nodeParameters: readonly ArrayLike<number>[];
  rbfWeights: readonly ArrayLike<number>[];
  polynomial: readonly ArrayLike<number>[];
  coordinateOffset: ArrayLike<number>;
  coordinateScale: ArrayLike<number>;
  origin: ArrayLike<number>;
  target: ArrayLike<number>;
  axes: readonly StructuredCurveAxis[];
  along: number;
  onto: number;
  sigmaRbfWeights?: ArrayLike<number> | null;
  sigmaPolynomial?: ArrayLike<number> | null;
  damping?: number;
}

export interface StructuredCurveAxis {
  periodic: boolean;
  period: number;
  lowerBound: number;
  upperBound: number;
}

export interface StructuredHookLayer {
  affineGroups?: readonly StructuredAffineGroup[];
  probes?: readonly StructuredProbe[];
  curves?: readonly StructuredCurve[];
  geometryWhitener?: StructuredGeometryWhitener | null;
  geometryProbes?: readonly StructuredGeometryProbe[];
}

export interface StructuredGeometryWhitener {
  rank: number;
  ridge: number;
  basis: readonly ArrayLike<number>[];
  correction: ArrayLike<number>;
}

export interface StructuredGeometryProbe {
  active: boolean;
  mean: ArrayLike<number>;
  inverseMean: ArrayLike<number>;
  basis: readonly ArrayLike<number>[];
  gramInverse: ArrayLike<number>;
  cholesky: ArrayLike<number>;
  candidateWhite: readonly ArrayLike<number>[];
  coordinateMap: readonly ArrayLike<number>[];
  coordinateBias: ArrayLike<number>;
  curve?: StructuredCurve | null;
  curveNodeCoordinates?: readonly ArrayLike<number>[];
  curveNodeValues?: readonly ArrayLike<number>[];
  foot?: ArrayLike<number> | null;
}

export interface StructuredJlensProgram {
  bindingId: string;
  layerIndices: Int32Array;
  tokenIds: Int32Array;
}

export interface StructuredMeasurementProbe {
  name: string;
  family: "geometry" | "lens" | "sae";
  tokenId?: number;
  featureId?: number;
  label?: string | null;
  maxAct?: number | null;
}

export interface StructuredGeometryMeasurementProbe {
  name: string;
  scoreKeys: string[];
  manifold: string;
  labels: string[];
  topN: number;
  intrinsicDim: number;
  rank: number;
  shareWeights: number[];
  assignBandwidth: number[];
  assignLogVolumeBias: number[];
  labelScale: number;
}

export interface StructuredMeasurementSchema {
  layerMap: number[];
  modelLayerCount: number;
  probes: Array<StructuredMeasurementProbe | null>;
  geometryProbes?: Array<StructuredGeometryMeasurementProbe | null>;
  lensSource: string | null;
  saeSource: string | null;
  saeLayer?: number | null;
  saeFeatureCount?: number | null;
  lensReadout?: boolean;
  saeReadout?: boolean;
  saeFeatureMetadata?: Record<string, { label: string | null; maxAct: number | null }>;
}

export interface StructuredHookProgramBuffers {
  hookAbi: typeof DROWSE_HOOK_ABI;
  format: StructuredHookFormat;
  activeRole?: string | null;
  profile: StructuredHookCapacityProfile;
  hiddenSize: number;
  layerCount: number;
  affineActive: Uint32Array;
  affineBasis: Float32Array;
  affineNeutral: Float32Array;
  affineTarget: Float32Array;
  affineAlong: Float32Array;
  affineKappa: Float32Array;
  probeKind: Uint32Array;
  probeDirection: Float32Array;
  probeBias: Float32Array;
  probeThreshold: Float32Array;
  curveActive: Uint32Array;
  curveRank: Uint32Array;
  curveIntrinsicDim: Uint32Array;
  curveEmbedDim: Uint32Array;
  curveNodeCount: Uint32Array;
  curveBasis: Float32Array;
  curveNeutral: Float32Array;
  curveNodeParameters: Float32Array;
  curveRbfWeights: Float32Array;
  curvePolynomial: Float32Array;
  curveCoordinateOffset: Float32Array;
  curveCoordinateScale: Float32Array;
  curveOrigin: Float32Array;
  curveTarget: Float32Array;
  curveAlong: Float32Array;
  curveOnto: Float32Array;
  curveBounds: Float32Array;
  curveAxisPeriodic: Uint32Array;
  curveAxisPeriod: Float32Array;
  curveDomainKind?: Uint32Array;
  curveSigmaPresent: Uint32Array;
  curveSigmaRbfWeights: Float32Array;
  curveSigmaPolynomial: Float32Array;
  curveDamping: Float32Array;
  whitenerRank?: Uint32Array;
  whitenerRidge?: Float32Array;
  whitenerBasis?: Float32Array;
  whitenerCorrection?: Float32Array;
  geometryActive?: Uint32Array;
  geometryKind?: Uint32Array;
  geometryRank?: Uint32Array;
  geometryIntrinsicDim?: Uint32Array;
  geometryCandidateCount?: Uint32Array;
  geometryCurveNodeCount?: Uint32Array;
  geometryMean?: Float32Array;
  geometryInverseMean?: Float32Array;
  geometryBasis?: Float32Array;
  geometryGramInverse?: Float32Array;
  geometryCholesky?: Float32Array;
  geometryNodeWhite?: Float32Array;
  geometryCoordMap?: Float32Array;
  geometryCoordBias?: Float32Array;
  geometryCurveParameters?: Float32Array;
  geometryCurveNodeCoords?: Float32Array;
  geometryCurveNodeValues?: Float32Array;
  geometryFeet?: Float32Array;
  geometryDomainKind?: Uint32Array;
  jLensBindingId?: string;
  jLensLayerIndices?: Int32Array;
  jLensReadoutLayerIndices?: Int32Array;
  jLensTokenIds?: Int32Array;
  saeBindingId?: string;
  measurementSchema?: StructuredMeasurementSchema;
  controls?: StructuredHookControls;
}

export interface StructuredMeasurementSlot {
  layer: number;
  probe: number;
}

export interface StructuredHookGate {
  slots: readonly StructuredMeasurementSlot[];
  scoreKey?: string | null;
  operator: SteeringComparisonOperator;
  threshold: number;
}

export interface StructuredHookControl {
  enabled: boolean;
  phase: SteeringPhase;
  gate: StructuredHookGate | null;
}

export interface StructuredHookControls {
  affine: readonly (StructuredHookControl | null)[];
  curve: readonly (StructuredHookControl | null)[];
}

export interface StructuredHookControlContext {
  prefill: boolean;
  thinking: boolean;
  generatedTokens: number;
  priorMeasurements?: ArrayLike<number> | null;
  priorScores?: Readonly<Record<string, number>> | null;
}

export interface StructuredHookControlState {
  affineActive: Uint32Array;
  curveActive: Uint32Array;
}

export interface StructuredHookState {
  curveFeet: Float32Array;
  geometryFeet?: Float32Array;
}

export interface StructuredHookLayerResult {
  residual: Float32Array;
  probes: Float32Array;
  curveFeet: Float32Array;
  geometry?: Float32Array;
}

export function additiveDirectionGroup(
  direction: ArrayLike<number>,
  coefficient: number,
  active = true,
): StructuredAffineGroup {
  const values = copyFiniteVector(direction, direction.length, "additive direction");
  const norm = vectorNorm(values);
  if (!(norm > 1e-9)) throw new TypeError("Additive direction must be nonzero");
  return {
    active,
    basis: [Float32Array.from(values, (value) => value / norm)],
    neutral: new Float32Array(values.length),
    target: [norm],
    along: finiteFloat(coefficient, "additive coefficient"),
    kappa: [0],
  };
}

export function saeFeatureProbe(
  encoderDirection: ArrayLike<number>,
  encoderBias: number,
  decoderBias: ArrayLike<number>,
  maxAct: number | null = null,
  activation: "relu" | "jump_relu" = "relu",
  threshold = 0,
): StructuredProbe {
  if (encoderDirection.length !== decoderBias.length) {
    throw new TypeError("SAE encoder and decoder bias dimensions do not match");
  }
  let direction = copyFiniteVector(
    encoderDirection,
    encoderDirection.length,
    "SAE encoder direction",
  );
  if (maxAct !== null) {
    if (!Number.isFinite(maxAct) || maxAct <= 0) {
      throw new TypeError("SAE max activation must be positive and finite");
    }
    direction = Float32Array.from(direction, (value) => value / maxAct);
  }
  return {
    kind: activation === "jump_relu" ? "sae_jump_relu" : "sae",
    direction,
    bias: finiteFloat(
      encoderBias / (maxAct ?? 1) - dotProduct(decoderBias, direction),
      "SAE feature bias",
    ),
    threshold: finiteFloat(threshold / (maxAct ?? 1), "SAE feature threshold"),
  };
}

export function jLensTokenDirection(
  unembeddingRow: ArrayLike<number>,
  jacobian: ArrayLike<number>,
): Float32Array {
  const hiddenSize = unembeddingRow.length;
  if (hiddenSize === 0 || jacobian.length !== hiddenSize * hiddenSize) {
    throw new TypeError("J-lens direction requires one square Jacobian");
  }
  const output = new Float32Array(hiddenSize);
  for (let column = 0; column < hiddenSize; column += 1) {
    let value = 0;
    for (let row = 0; row < hiddenSize; row += 1) {
      value += finiteFloat(unembeddingRow[row], "J-lens unembedding") *
        finiteFloat(jacobian[row * hiddenSize + column], "J-lens Jacobian");
    }
    output[column] = finiteFloat(value, "J-lens direction");
  }
  return output;
}

export function compileStructuredHookProgram(
  hiddenSize: number,
  layers: readonly StructuredHookLayer[],
  inputProfile: StructuredHookCapacityProfile,
  jLens: StructuredJlensProgram | null = null,
): StructuredHookProgramBuffers {
  requirePositiveInteger(hiddenSize, "Hook hidden size");
  if (layers.length === 0) throw new TypeError("Hook program requires at least one layer");
  const profile = validateStructuredHookProfile(inputProfile);
  const layerCount = layers.length;
  const affineSlots = checkedStructuredHookProduct(
    "affine slots",
    layerCount,
    profile.maxAffineGroups,
  );
  const affineAxes = checkedStructuredHookProduct(
    "affine axes",
    affineSlots,
    profile.maxRank,
  );
  const probeSlots = checkedStructuredHookProduct(
    "probe slots",
    layerCount,
    profile.maxProbes,
  );
  const curveSlots = checkedStructuredHookProduct(
    "curve slots",
    layerCount,
    profile.maxCurves,
  );
  const curveAxes = checkedStructuredHookProduct(
    "curve axes",
    curveSlots,
    profile.maxRank,
  );
  const curveNodes = checkedStructuredHookProduct(
    "curve nodes",
    curveSlots,
    profile.maxCurveNodes,
  );
  const program: StructuredHookProgramBuffers = {
    hookAbi: DROWSE_HOOK_ABI,
    format: profile.maxGeometryProbes > 0
      ? STRUCTURED_HOOK_FORMAT
      : STRUCTURED_HOOK_FORMAT_V2,
    profile,
    hiddenSize,
    layerCount,
    affineActive: new Uint32Array(affineSlots),
    affineBasis: new Float32Array(checkedStructuredHookProduct("affine basis", affineAxes, hiddenSize)),
    affineNeutral: new Float32Array(checkedStructuredHookProduct("affine neutral", affineSlots, hiddenSize)),
    affineTarget: new Float32Array(affineAxes),
    affineAlong: new Float32Array(affineSlots),
    affineKappa: new Float32Array(affineAxes),
    probeKind: new Uint32Array(probeSlots),
    probeDirection: new Float32Array(checkedStructuredHookProduct("probe direction", probeSlots, hiddenSize)),
    probeBias: new Float32Array(probeSlots),
    probeThreshold: new Float32Array(probeSlots),
    curveActive: new Uint32Array(curveSlots),
    curveRank: new Uint32Array(curveSlots),
    curveIntrinsicDim: new Uint32Array(curveSlots),
    curveEmbedDim: new Uint32Array(curveSlots),
    curveNodeCount: new Uint32Array(curveSlots),
    curveBasis: new Float32Array(checkedStructuredHookProduct("curve basis", curveAxes, hiddenSize)),
    curveNeutral: new Float32Array(checkedStructuredHookProduct("curve neutral", curveSlots, hiddenSize)),
    curveNodeParameters: new Float32Array(checkedStructuredHookProduct("curve node parameters", curveNodes, profile.maxEmbedDim)),
    curveRbfWeights: new Float32Array(checkedStructuredHookProduct("curve RBF weights", curveNodes, profile.maxRank)),
    curvePolynomial: new Float32Array(
      checkedStructuredHookProduct(
        "curve polynomial",
        curveSlots,
        profile.maxEmbedDim + 1,
        profile.maxRank,
      ),
    ),
    curveCoordinateOffset: new Float32Array(checkedStructuredHookProduct("curve coordinate offset", curveSlots, profile.maxEmbedDim)),
    curveCoordinateScale: new Float32Array(checkedStructuredHookProduct("curve coordinate scale", curveSlots, profile.maxEmbedDim)),
    curveOrigin: new Float32Array(checkedStructuredHookProduct("curve origin", curveSlots, profile.maxIntrinsicDim)),
    curveTarget: new Float32Array(checkedStructuredHookProduct("curve target", curveSlots, profile.maxIntrinsicDim)),
    curveAlong: new Float32Array(curveSlots),
    curveOnto: new Float32Array(curveSlots),
    curveBounds: new Float32Array(checkedStructuredHookProduct("curve bounds", curveSlots, profile.maxIntrinsicDim, 2)),
    curveAxisPeriodic: new Uint32Array(checkedStructuredHookProduct("curve periodic axes", curveSlots, profile.maxIntrinsicDim)),
    curveAxisPeriod: new Float32Array(checkedStructuredHookProduct("curve axis periods", curveSlots, profile.maxIntrinsicDim)),
    curveSigmaPresent: new Uint32Array(curveSlots),
    curveSigmaRbfWeights: new Float32Array(curveNodes),
    curveSigmaPolynomial: new Float32Array(checkedStructuredHookProduct("curve sigma polynomial", curveSlots, profile.maxEmbedDim + 1)),
    curveDamping: new Float32Array(curveSlots),
  };
  if (profile.maxGeometryProbes > 0) initializeGeometryBuffers(program);
  program.curveCoordinateScale.fill(1);
  program.curveAxisPeriod.fill(1);
  for (let slot = 0; slot < curveSlots; slot += 1) {
    for (let axis = 0; axis < profile.maxIntrinsicDim; axis += 1) {
      const bound = (slot * profile.maxIntrinsicDim + axis) * 2;
      program.curveBounds[bound] = -3.4028234663852886e38;
      program.curveBounds[bound + 1] = 3.4028234663852886e38;
    }
  }
  for (let layer = 0; layer < layerCount; layer += 1) {
    compileLayer(program, layer, layers[layer]);
  }
  if (jLens !== null) {
    program.jLensBindingId = jLens.bindingId;
    program.jLensLayerIndices = new Int32Array(jLens.layerIndices);
    program.jLensTokenIds = new Int32Array(jLens.tokenIds);
  }
  validateStructuredHookProgram(program);
  return program;
}

export function initialStructuredHookState(
  program: StructuredHookProgramBuffers,
): StructuredHookState {
  validateStructuredHookProgram(program);
  return {
    curveFeet: new Float32Array(program.curveOrigin),
    ...(program.geometryFeet === undefined
      ? {}
      : { geometryFeet: new Float32Array(program.geometryFeet) }),
  };
}

function initializeGeometryBuffers(program: StructuredHookProgramBuffers): void {
  const { profile, layerCount: l, hiddenSize: d } = program;
  const g = profile.maxGeometryProbes;
  const w = profile.maxWhitenerRank;
  const r = profile.maxRank;
  const q = profile.maxGeometryCandidates;
  const i = profile.maxIntrinsicDim;
  const n = profile.maxCurveNodes;
  const stride = structuredCurveParameterStride(profile);
  const slots = checkedStructuredHookProduct("geometry slots", l, g);
  program.curveDomainKind = new Uint32Array(
    checkedStructuredHookProduct("curve domain kinds", l, profile.maxCurves),
  );
  program.whitenerRank = new Uint32Array(l);
  program.whitenerRidge = new Float32Array(l);
  program.whitenerBasis = new Float32Array(
    checkedStructuredHookProduct("geometry whitener basis", l, w, d),
  );
  program.whitenerCorrection = new Float32Array(
    checkedStructuredHookProduct("geometry whitener correction", l, w),
  );
  program.geometryActive = new Uint32Array(slots);
  program.geometryKind = new Uint32Array(slots);
  program.geometryRank = new Uint32Array(slots);
  program.geometryIntrinsicDim = new Uint32Array(slots);
  program.geometryCandidateCount = new Uint32Array(slots);
  program.geometryCurveNodeCount = new Uint32Array(slots);
  program.geometryMean = new Float32Array(
    checkedStructuredHookProduct("geometry mean", slots, d),
  );
  program.geometryInverseMean = new Float32Array(
    checkedStructuredHookProduct("geometry inverse mean", slots, d),
  );
  program.geometryBasis = new Float32Array(
    checkedStructuredHookProduct("geometry basis", slots, r, d),
  );
  program.geometryGramInverse = new Float32Array(
    checkedStructuredHookProduct("geometry gram inverse", slots, r, r),
  );
  program.geometryCholesky = new Float32Array(
    checkedStructuredHookProduct("geometry cholesky", slots, r, r),
  );
  program.geometryNodeWhite = new Float32Array(
    checkedStructuredHookProduct("geometry node white", slots, q, r),
  );
  program.geometryCoordMap = new Float32Array(
    checkedStructuredHookProduct("geometry coordinate map", slots, i, r),
  );
  program.geometryCoordBias = new Float32Array(
    checkedStructuredHookProduct("geometry coordinate bias", slots, i),
  );
  program.geometryCurveParameters = new Float32Array(
    checkedStructuredHookProduct("geometry curve parameters", slots, stride),
  );
  const curveOffsets = curveParameterOffsets(profile);
  for (let slot = 0; slot < slots; slot += 1) {
    const base = slot * stride;
    for (let embed = 0; embed < profile.maxEmbedDim; embed += 1) {
      program.geometryCurveParameters[base + curveOffsets.coordinateScale + embed] = 1;
    }
    for (let axis = 0; axis < profile.maxIntrinsicDim; axis += 1) {
      program.geometryCurveParameters[base + curveOffsets.axisPeriod + axis] = 1;
    }
  }
  program.geometryCurveNodeCoords = new Float32Array(
    checkedStructuredHookProduct("geometry curve node coordinates", slots, n, i),
  );
  program.geometryCurveNodeValues = new Float32Array(
    checkedStructuredHookProduct("geometry curve node values", slots, n, r),
  );
  program.geometryFeet = new Float32Array(
    checkedStructuredHookProduct("geometry feet", slots, i),
  );
  program.geometryDomainKind = new Uint32Array(slots);
}

export function structuredHookControlsFor(
  program: StructuredHookProgramBuffers,
  context: StructuredHookControlContext,
): StructuredHookControlState {
  validateStructuredHookProgram(program);
  return evaluateStructuredHookControls(program, context);
}

export function createStructuredHookControlEvaluator(
  program: StructuredHookProgramBuffers,
): (context: StructuredHookControlContext) => StructuredHookControlState {
  validateStructuredHookProgram(program);
  const snapshot = {
    layerCount: program.layerCount,
    profile: { maxProbes: program.profile.maxProbes },
    affineActive: new Uint32Array(program.affineActive),
    curveActive: new Uint32Array(program.curveActive),
    controls: program.controls === undefined ? undefined : structuredClone(program.controls),
  };
  return (context) => evaluateStructuredHookControls(snapshot, context);
}

type StructuredControlProgram = Pick<
  StructuredHookProgramBuffers, "layerCount" | "affineActive" | "curveActive" | "controls"
> & { profile: Pick<StructuredHookCapacityProfile, "maxProbes"> };

function evaluateStructuredHookControls(
  program: StructuredControlProgram,
  context: StructuredHookControlContext,
): StructuredHookControlState {
  const affineActive = new Uint32Array(program.affineActive);
  const curveActive = new Uint32Array(program.curveActive);
  const controls = program.controls;
  if (controls === undefined) return { affineActive, curveActive };
  if (
    controls.affine.length !== affineActive.length ||
    controls.curve.length !== curveActive.length
  ) {
    throw new TypeError("Structured hook controls do not match the program shape");
  }
  controls.affine.forEach((control, index) => {
    if (control !== null) affineActive[index] = controlActive(program, control, context) ? 1 : 0;
  });
  controls.curve.forEach((control, index) => {
    if (control !== null) curveActive[index] = controlActive(program, control, context) ? 1 : 0;
  });
  return { affineActive, curveActive };
}

function controlActive(
  program: StructuredControlProgram,
  control: StructuredHookControl,
  context: StructuredHookControlContext,
): boolean {
  if (!control.enabled || !phaseActive(control.phase, context)) return false;
  if (control.gate === null) return true;
  if (context.prefill) return false;
  let score: number;
  if (control.gate.scoreKey !== null && control.gate.scoreKey !== undefined) {
    score = context.priorScores?.[control.gate.scoreKey] ?? Number.NaN;
    if (!Number.isFinite(score)) return false;
  } else {
    if (context.priorMeasurements == null || control.gate.slots.length === 0) return false;
    score = 0;
    for (const slot of control.gate.slots) {
      if (
        !Number.isSafeInteger(slot.layer) || slot.layer < 0 || slot.layer >= program.layerCount ||
        !Number.isSafeInteger(slot.probe) || slot.probe < 0 || slot.probe >= program.profile.maxProbes
      ) {
        throw new TypeError("Structured hook gate references an invalid measurement slot");
      }
      const value = context.priorMeasurements[slot.layer * program.profile.maxProbes + slot.probe];
      if (!Number.isFinite(value)) return false;
      score += value;
    }
    score /= control.gate.slots.length;
  }
  switch (control.gate.operator) {
    case ">": return score > control.gate.threshold;
    case ">=": return score >= control.gate.threshold;
    case "<": return score < control.gate.threshold;
    case "<=": return score <= control.gate.threshold;
  }
}

function phaseActive(
  phase: SteeringPhase,
  context: StructuredHookControlContext,
): boolean {
  if (context.prefill) return phase.kind === "both" || phase.kind === "prompt_only";
  switch (phase.kind) {
    case "both":
    case "generated_only":
      return true;
    case "prompt_only":
      return false;
    case "thinking_only":
      return context.thinking;
    case "after_thinking":
      return !context.thinking;
    case "first":
      return context.generatedTokens < phase.tokens;
    case "after":
      return context.generatedTokens >= phase.tokens;
  }
}

export function runStructuredHookLayer(
  program: StructuredHookProgramBuffers,
  state: StructuredHookState,
  layer: number,
  residual: ArrayLike<number>,
  options: { decode: boolean } = { decode: true },
): StructuredHookLayerResult {
  validateStructuredHookProgram(program);
  requireLayer(program, layer);
  if (
    state.curveFeet.length !==
    checkedStructuredHookProduct(
      "structured hook feet",
      program.layerCount,
      program.profile.maxCurves,
      program.profile.maxIntrinsicDim,
    )
  ) {
    throw new TypeError("Structured hook feet do not match the program layer count");
  }
  if (
    program.geometryFeet !== undefined &&
    state.geometryFeet?.length !== program.geometryFeet.length
  ) {
    throw new TypeError("Structured geometry feet do not match the program shape");
  }
  const values = copyFiniteVector(residual, program.hiddenSize, "residual");
  runAffineGroups(program, layer, values);
  for (let curve = 0; curve < program.profile.maxCurves; curve += 1) {
    const slot = layer * program.profile.maxCurves + curve;
    if (program.curveActive[slot] !== 1) continue;
    const footOffset = slot * program.profile.maxIntrinsicDim;
    const seed = options.decode
      ? state.curveFeet.slice(footOffset, footOffset + program.profile.maxIntrinsicDim)
      : program.curveOrigin.slice(footOffset, footOffset + program.profile.maxIntrinsicDim);
    const next = runCurve(program, slot, values, seed, options.decode ? 1 : 4);
    state.curveFeet.set(next, footOffset);
  }
  const geometry = program.geometryActive === undefined
    ? undefined
    : runGeometryProbes(program, state, layer, values, options.decode);
  return {
    residual: values,
    probes: runProbes(program, layer, values),
    curveFeet: state.curveFeet.slice(
      layer * program.profile.maxCurves * program.profile.maxIntrinsicDim,
      (layer + 1) * program.profile.maxCurves * program.profile.maxIntrinsicDim,
    ),
    ...(geometry === undefined ? {} : { geometry }),
  };
}

export function validateStructuredHookProgram(program: StructuredHookProgramBuffers): void {
  if (program.hookAbi !== DROWSE_HOOK_ABI) {
    throw new TypeError("Unsupported structured hook program identity");
  }
  requirePositiveInteger(program.hiddenSize, "Hook hidden size");
  requirePositiveInteger(program.layerCount, "Hook layer count");
  const profile = validateStructuredHookProfile(program.profile);
  const expectedFormat = profile.maxGeometryProbes > 0
    ? STRUCTURED_HOOK_FORMAT
    : STRUCTURED_HOOK_FORMAT_V2;
  if (program.format !== expectedFormat) {
    throw new TypeError("Unsupported structured hook program identity");
  }
  const l = program.layerCount;
  const d = program.hiddenSize;
  const g = profile.maxAffineGroups;
  const r = profile.maxRank;
  const p = profile.maxProbes;
  const n = profile.maxCurveNodes;
  const c = profile.maxCurves;
  const i = profile.maxIntrinsicDim;
  const m = profile.maxEmbedDim;
  const curveSlots = checkedStructuredHookProduct("curve slots", l, c);
  const size = (label: string, ...factors: number[]): number =>
    checkedStructuredHookProduct(label, ...factors);
  const expected: Array<[ArrayLike<number>, number, string]> = [
    [program.affineActive, size("affine active", l, g), "affine active"],
    [program.affineBasis, size("affine basis", l, g, r, d), "affine basis"],
    [program.affineNeutral, size("affine neutral", l, g, d), "affine neutral"],
    [program.affineTarget, size("affine target", l, g, r), "affine target"],
    [program.affineAlong, size("affine along", l, g), "affine along"],
    [program.affineKappa, size("affine kappa", l, g, r), "affine kappa"],
    [program.probeKind, size("probe kind", l, p), "probe kind"],
    [program.probeDirection, size("probe direction", l, p, d), "probe direction"],
    [program.probeBias, size("probe bias", l, p), "probe bias"],
    [program.probeThreshold, size("probe threshold", l, p), "probe threshold"],
    [program.curveActive, curveSlots, "curve active"],
    [program.curveRank, curveSlots, "curve rank"],
    [program.curveIntrinsicDim, curveSlots, "curve intrinsic dimension"],
    [program.curveEmbedDim, curveSlots, "curve embedding dimension"],
    [program.curveNodeCount, curveSlots, "curve node count"],
    [program.curveBasis, size("curve basis", curveSlots, r, d), "curve basis"],
    [program.curveNeutral, size("curve neutral", curveSlots, d), "curve neutral"],
    [program.curveNodeParameters, size("curve node parameters", curveSlots, n, m), "curve node parameters"],
    [program.curveRbfWeights, size("curve RBF weights", curveSlots, n, r), "curve RBF weights"],
    [program.curvePolynomial, size("curve polynomial", curveSlots, m + 1, r), "curve polynomial"],
    [program.curveCoordinateOffset, size("curve coordinate offset", curveSlots, m), "curve coordinate offset"],
    [program.curveCoordinateScale, size("curve coordinate scale", curveSlots, m), "curve coordinate scale"],
    [program.curveOrigin, size("curve origin", curveSlots, i), "curve origin"],
    [program.curveTarget, size("curve target", curveSlots, i), "curve target"],
    [program.curveAlong, curveSlots, "curve along"],
    [program.curveOnto, curveSlots, "curve onto"],
    [program.curveBounds, size("curve bounds", curveSlots, i, 2), "curve bounds"],
    [program.curveAxisPeriodic, size("curve periodic axes", curveSlots, i), "curve periodic axes"],
    [program.curveAxisPeriod, size("curve axis periods", curveSlots, i), "curve axis periods"],
    [program.curveSigmaPresent, curveSlots, "curve sigma present"],
    [program.curveSigmaRbfWeights, size("curve sigma weights", curveSlots, n), "curve sigma weights"],
    [program.curveSigmaPolynomial, size("curve sigma polynomial", curveSlots, m + 1), "curve sigma polynomial"],
    [program.curveDamping, curveSlots, "curve damping"],
  ];
  for (const [values, length, label] of expected) {
    if (values.length !== length) throw new TypeError(`${label} has the wrong length`);
    if (!(values instanceof Uint32Array)) {
      for (let index = 0; index < values.length; index += 1) {
        finiteFloat(values[index], label, values === program.curveBounds);
      }
    }
  }
  validateGeometryBuffers(program, profile);
  for (const value of [
    ...program.affineActive,
    ...program.curveActive,
    ...program.curveSigmaPresent,
    ...program.curveAxisPeriodic,
  ]) {
    if (value !== 0 && value !== 1) throw new TypeError("Hook active values must be zero or one");
  }
  for (const value of program.probeKind) {
    if (value > 4) throw new TypeError("Hook probe kind is invalid");
  }
  if (program.jLensReadoutLayerIndices !== undefined && (
    !(program.jLensReadoutLayerIndices instanceof Int32Array) ||
    program.jLensLayerIndices === undefined ||
    program.jLensReadoutLayerIndices.some((layer, index) =>
      !program.jLensLayerIndices!.includes(layer) ||
      index > 0 && layer <= program.jLensReadoutLayerIndices![index - 1]
    )
  )) throw new TypeError("J-lens readout layers must be an ordered subset of probability layers");
  const hasJlensProbe = program.probeKind.some((value) => value === 3);
  const hasJlensProgram = program.jLensBindingId !== undefined;
  if (
    hasJlensProgram !== (program.jLensLayerIndices !== undefined) ||
    hasJlensProgram !== (program.jLensTokenIds !== undefined) ||
    hasJlensProbe && !hasJlensProgram
  ) {
    throw new TypeError("J-lens probes and readouts require one complete probability program");
  }
  if (hasJlensProgram) {
    if (
      !validInstrumentBindingId(program.jLensBindingId) ||
      !(program.jLensLayerIndices instanceof Int32Array) ||
      !(program.jLensTokenIds instanceof Int32Array)
    ) {
      throw new TypeError("J-lens probability buffers have the wrong types");
    }
    const layerIndices = program.jLensLayerIndices!;
    if (
      layerIndices.some((layer, index) =>
        layer < 0 || layer >= l || (index > 0 && layer <= layerIndices[index - 1])
      )
    ) {
      throw new TypeError("J-lens layer indices must be unique, ordered, and in range");
    }
    if (hasJlensProbe) {
      const activeLayers: number[] = [];
      for (let layer = 0; layer < l; layer += 1) {
        const offset = layer * p;
        if (program.probeKind.subarray(offset, offset + p).some((value) => value === 3)) {
          activeLayers.push(layer);
        }
      }
      if (
        layerIndices.length !== activeLayers.length ||
        layerIndices.some((layer, index) => layer !== activeLayers[index])
      ) {
        throw new TypeError("J-lens layer indices do not match the active probability probes");
      }
    }
    if (program.jLensTokenIds!.length !== p) {
      throw new TypeError("J-lens token IDs have the wrong length");
    }
    if (program.jLensTokenIds!.some((tokenId) => tokenId < 0)) {
      throw new TypeError("J-lens token IDs are invalid");
    }
  }
  if (
    program.saeBindingId !== undefined &&
    !validInstrumentBindingId(program.saeBindingId)
  ) {
    throw new TypeError("SAE readout binding is invalid");
  }
  for (let slot = 0; slot < curveSlots; slot += 1) {
    if (
      program.curveRank[slot] > r || program.curveNodeCount[slot] > n ||
      program.curveIntrinsicDim[slot] > i || program.curveEmbedDim[slot] > m
    ) {
      throw new TypeError("Curved hook dimensions exceed the structured program limits");
    }
    if (program.curveActive[slot] === 1) {
      if (
        program.curveRank[slot] === 0 || program.curveNodeCount[slot] < 2 ||
        program.curveIntrinsicDim[slot] === 0 || program.curveEmbedDim[slot] === 0
      ) {
        throw new TypeError("An active curved hook requires a rank and at least two nodes");
      }
      const domainKind = program.curveDomainKind?.[slot] ?? 1;
      if (
        (domainKind !== 1 && domainKind !== 2) ||
        (domainKind === 2 &&
          program.curveEmbedDim[slot] !== program.curveIntrinsicDim[slot] + 1)
      ) {
        throw new TypeError("An active curved hook has an invalid domain kind");
      }
      if (!(program.curveDamping[slot] > 0)) throw new TypeError("Curved damping must be positive");
      for (let embed = 0; embed < program.curveEmbedDim[slot]; embed += 1) {
        if (!(program.curveCoordinateScale[slot * m + embed] > 0)) {
          throw new TypeError("Curved coordinate scales must be positive");
        }
      }
      for (let axis = 0; axis < program.curveIntrinsicDim[slot]; axis += 1) {
        const axisSlot = slot * i + axis;
        const bound = axisSlot * 2;
        if (!(program.curveBounds[bound] <= program.curveBounds[bound + 1])) {
          throw new TypeError("Curved hook bounds are inverted");
        }
        if (program.curveAxisPeriodic[axisSlot] === 1 && !(program.curveAxisPeriod[axisSlot] > 0)) {
          throw new TypeError("Periodic curve axes require a positive period");
        }
      }
    }
  }
  if (program.controls !== undefined) {
    if (
      program.controls.affine.length !== l * g ||
      program.controls.curve.length !== curveSlots
    ) {
      throw new TypeError("Structured hook controls have the wrong length");
    }
    for (const control of [...program.controls.affine, ...program.controls.curve]) {
      if (control === null) continue;
      if (typeof control.enabled !== "boolean") {
        throw new TypeError("Structured hook control enabled state must be boolean");
      }
      if (!Number.isFinite(control.gate?.threshold ?? 0)) {
        throw new TypeError("Structured hook gate threshold must be finite");
      }
      if (control.gate !== null) validateStructuredHookGate(program, control.gate);
    }
  }
  if (program.measurementSchema !== undefined) {
    const schema = program.measurementSchema;
    if (
      schema.layerMap.length !== l ||
      !Number.isSafeInteger(schema.modelLayerCount) || schema.modelLayerCount < 1 ||
      schema.layerMap.some((layer, index) =>
        !Number.isSafeInteger(layer) || layer < 0 ||
        layer >= schema.modelLayerCount ||
        index > 0 && layer <= schema.layerMap[index - 1]
      ) ||
      schema.probes.length !== p ||
      !nullableString(schema.lensSource) ||
      !nullableString(schema.saeSource) ||
      schema.saeLayer !== undefined && schema.saeLayer !== null &&
        (!Number.isSafeInteger(schema.saeLayer) || schema.saeLayer < 0) ||
      schema.saeFeatureCount !== undefined && schema.saeFeatureCount !== null &&
        (!Number.isSafeInteger(schema.saeFeatureCount) || schema.saeFeatureCount < 1) ||
      schema.lensReadout !== undefined && typeof schema.lensReadout !== "boolean" ||
      schema.saeReadout !== undefined && typeof schema.saeReadout !== "boolean"
    ) {
      throw new TypeError("Structured measurement schema has the wrong shape");
    }
    if (schema.lensReadout === true && !hasJlensProgram) {
      throw new TypeError("J-lens live readout requires the full-vocabulary probability program");
    }
    if (schema.saeFeatureMetadata !== undefined) {
      for (const [featureId, metadata] of Object.entries(schema.saeFeatureMetadata)) {
        if (
          !/^(0|[1-9]\d*)$/u.test(featureId) ||
          metadata === null || typeof metadata !== "object" ||
          metadata.label !== null &&
            (typeof metadata.label !== "string" || metadata.label.trim() === "") ||
          metadata.maxAct !== null &&
            (!Number.isFinite(metadata.maxAct) || metadata.maxAct <= 0)
        ) {
          throw new TypeError("Structured SAE feature metadata is invalid");
        }
      }
    }
    for (const probe of schema.probes) {
      if (probe === null) continue;
      if (
        probe.name.length === 0 ||
        !["geometry", "lens", "sae"].includes(probe.family) ||
        (probe.tokenId !== undefined &&
          (!Number.isSafeInteger(probe.tokenId) || probe.tokenId < 0)) ||
        (probe.featureId !== undefined &&
          (!Number.isSafeInteger(probe.featureId) || probe.featureId < 0)) ||
        (probe.label !== undefined && probe.label !== null &&
          (typeof probe.label !== "string" || probe.label.trim() === "")) ||
        (probe.maxAct !== undefined && probe.maxAct !== null &&
          (!Number.isFinite(probe.maxAct) || probe.maxAct <= 0))
      ) {
        throw new TypeError("Structured measurement schema contains an invalid probe");
      }
    }
    validateGeometryMeasurementSchema(program, schema);
  }
}

function validateStructuredHookGate(
  program: StructuredHookProgramBuffers,
  gate: StructuredHookGate,
): void {
  if (![">", ">=", "<", "<="].includes(gate.operator)) {
    throw new TypeError("Structured hook gate comparison is invalid");
  }
  const hasScoreKey = gate.scoreKey !== null && gate.scoreKey !== undefined;
  if (hasScoreKey) {
    if (typeof gate.scoreKey !== "string" || gate.scoreKey.length === 0 || gate.slots.length !== 0) {
      throw new TypeError("A score-key gate cannot also reference scalar measurement slots");
    }
    return;
  }
  if (gate.slots.length === 0) {
    throw new TypeError("A scalar gate requires at least one measurement slot");
  }
  for (const slot of gate.slots) {
    if (
      !Number.isSafeInteger(slot.layer) || slot.layer < 0 || slot.layer >= program.layerCount ||
      !Number.isSafeInteger(slot.probe) || slot.probe < 0 ||
      slot.probe >= program.profile.maxProbes
    ) {
      throw new TypeError("Structured hook gate references an invalid measurement slot");
    }
  }
}

function validateGeometryMeasurementSchema(
  program: StructuredHookProgramBuffers,
  schema: StructuredMeasurementSchema,
): void {
  const geometry = schema.geometryProbes;
  if (geometry === undefined) return;
  if (
    program.format !== STRUCTURED_HOOK_FORMAT ||
    geometry.length !== program.profile.maxGeometryProbes
  ) {
    throw new TypeError("Structured geometry measurement schema has the wrong shape");
  }
  geometry.forEach((probe, probeIndex) => {
    if (probe === null) return;
    if (
      schema.probes[probeIndex] !== null ||
      probe.name.length === 0 || probe.manifold.length === 0 ||
      probe.scoreKeys.length === 0 ||
      probe.scoreKeys.some((key) => typeof key !== "string" || key.length === 0) ||
      new Set(probe.scoreKeys).size !== probe.scoreKeys.length ||
      probe.labels.length < 1 ||
      probe.labels.length > program.profile.maxGeometryCandidates ||
      probe.labels.some((label) => typeof label !== "string" || label.length === 0) ||
      new Set(probe.labels).size !== probe.labels.length ||
      !Number.isSafeInteger(probe.topN) || probe.topN < 0 || probe.topN > probe.labels.length ||
      !Number.isSafeInteger(probe.intrinsicDim) || probe.intrinsicDim < 1 ||
      probe.intrinsicDim > program.profile.maxIntrinsicDim ||
      !Number.isSafeInteger(probe.rank) || probe.rank < 1 ||
      probe.rank > program.profile.maxRank ||
      probe.shareWeights.length !== program.layerCount ||
      probe.shareWeights.some((value) => !Number.isFinite(value) || value < 0) ||
      !probe.shareWeights.some((value) => value > 0) ||
      probe.assignBandwidth.length !== probe.labels.length ||
      probe.assignBandwidth.some((value) => !Number.isFinite(value) || value <= 0) ||
      probe.assignLogVolumeBias.length !== probe.labels.length ||
      probe.assignLogVolumeBias.some((value) => !Number.isFinite(value)) ||
      !Number.isFinite(probe.labelScale) || probe.labelScale <= 0
    ) {
      throw new TypeError("Structured geometry measurement schema contains an invalid probe");
    }
    let active = false;
    for (let layer = 0; layer < program.layerCount; layer += 1) {
      const slot = layer * program.profile.maxGeometryProbes + probeIndex;
      if (program.geometryActive?.[slot] === 1) {
        active = true;
        if (
          program.geometryIntrinsicDim?.[slot] !== probe.intrinsicDim ||
          program.geometryCandidateCount?.[slot] !== probe.labels.length
        ) {
          throw new TypeError("Structured geometry measurement metadata does not match its buffers");
        }
      }
    }
    if (!active) {
      throw new TypeError("Structured geometry measurement schema references an inactive probe");
    }
  });
}

function validateGeometryBuffers(
  program: StructuredHookProgramBuffers,
  profile: StructuredHookCapacityProfile,
): void {
  const names = [
    "curveDomainKind",
    "whitenerRank",
    "whitenerRidge",
    "whitenerBasis",
    "whitenerCorrection",
    "geometryActive",
    "geometryKind",
    "geometryRank",
    "geometryIntrinsicDim",
    "geometryCandidateCount",
    "geometryCurveNodeCount",
    "geometryMean",
    "geometryInverseMean",
    "geometryBasis",
    "geometryGramInverse",
    "geometryCholesky",
    "geometryNodeWhite",
    "geometryCoordMap",
    "geometryCoordBias",
    "geometryCurveParameters",
    "geometryCurveNodeCoords",
    "geometryCurveNodeValues",
    "geometryFeet",
    "geometryDomainKind",
  ] as const;
  if (profile.maxGeometryProbes === 0) {
    if (names.some((name) => program[name] !== undefined)) {
      throw new TypeError("drowse-structured-v2 cannot contain geometry buffers");
    }
    return;
  }
  if (names.some((name) => program[name] === undefined)) {
    throw new TypeError("drowse-structured-v3 requires complete geometry buffers");
  }
  const l = program.layerCount;
  const d = program.hiddenSize;
  const g = profile.maxGeometryProbes;
  const w = profile.maxWhitenerRank;
  const r = profile.maxRank;
  const q = profile.maxGeometryCandidates;
  const i = profile.maxIntrinsicDim;
  const n = profile.maxCurveNodes;
  const slots = checkedStructuredHookProduct("geometry slots", l, g);
  const curveStride = structuredCurveParameterStride(profile);
  const curveOffsets = curveParameterOffsets(profile);
  const expected: Array<[ArrayLike<number>, number, string]> = [
    [program.curveDomainKind!, l * profile.maxCurves, "curve domain kind"],
    [program.whitenerRank!, l, "geometry whitener rank"],
    [program.whitenerRidge!, l, "geometry whitener ridge"],
    [program.whitenerBasis!, l * w * d, "geometry whitener basis"],
    [program.whitenerCorrection!, l * w, "geometry whitener correction"],
    [program.geometryActive!, slots, "geometry active"],
    [program.geometryKind!, slots, "geometry kind"],
    [program.geometryRank!, slots, "geometry rank"],
    [program.geometryIntrinsicDim!, slots, "geometry intrinsic dimension"],
    [program.geometryCandidateCount!, slots, "geometry candidate count"],
    [program.geometryCurveNodeCount!, slots, "geometry curve node count"],
    [program.geometryMean!, slots * d, "geometry mean"],
    [program.geometryInverseMean!, slots * d, "geometry inverse mean"],
    [program.geometryBasis!, slots * r * d, "geometry basis"],
    [program.geometryGramInverse!, slots * r * r, "geometry gram inverse"],
    [program.geometryCholesky!, slots * r * r, "geometry cholesky"],
    [program.geometryNodeWhite!, slots * q * r, "geometry node white"],
    [program.geometryCoordMap!, slots * i * r, "geometry coordinate map"],
    [program.geometryCoordBias!, slots * i, "geometry coordinate bias"],
    [
      program.geometryCurveParameters!,
      slots * structuredCurveParameterStride(profile),
      "geometry curve parameters",
    ],
    [program.geometryCurveNodeCoords!, slots * n * i, "geometry curve node coordinates"],
    [program.geometryCurveNodeValues!, slots * n * r, "geometry curve node values"],
    [program.geometryFeet!, slots * i, "geometry feet"],
    [program.geometryDomainKind!, slots, "geometry domain kind"],
  ];
  for (const [values, length, label] of expected) {
    if (values.length !== length) throw new TypeError(`${label} has the wrong length`);
    if (!(values instanceof Uint32Array)) {
      for (let index = 0; index < values.length; index += 1) finiteFloat(values[index], label);
    }
  }
  if (
    program.curveDomainKind!.some((value) => value > 2) ||
    program.geometryDomainKind!.some((value) => value > 2)
  ) {
    throw new TypeError("Structured geometry domain tags must be inactive, box/custom, or sphere");
  }
  for (let layer = 0; layer < l; layer += 1) {
    const rank = program.whitenerRank![layer];
    if (rank > w) throw new TypeError("Geometry whitener rank exceeds the profile");
    const layerActive = program.geometryActive!.subarray(layer * g, (layer + 1) * g)
      .some((value) => value === 1);
    if (layerActive && (rank === 0 || !(program.whitenerRidge![layer] > 0))) {
      throw new TypeError("An active geometry layer requires one positive-definite whitener");
    }
  }
  for (let slot = 0; slot < slots; slot += 1) {
    const active = program.geometryActive![slot];
    if (active !== 0 && active !== 1) throw new TypeError("Geometry active values must be zero or one");
    if (active === 0) continue;
    const kind = program.geometryKind![slot];
    const rank = program.geometryRank![slot];
    const intrinsicDim = program.geometryIntrinsicDim![slot];
    const candidateCount = program.geometryCandidateCount![slot];
    const curveNodes = program.geometryCurveNodeCount![slot];
    const domainKind = program.geometryDomainKind![slot];
    if (
      (kind !== 1 && kind !== 2) || rank < 1 || rank > r ||
      intrinsicDim < 1 || intrinsicDim > i ||
      candidateCount < 1 || candidateCount > q ||
      (kind === 1 && curveNodes !== 0) ||
      (kind === 2 && (curveNodes < 2 || curveNodes > n)) ||
      (kind === 1 && domainKind !== 0) ||
      (kind === 2 && domainKind !== 1 && domainKind !== 2)
    ) {
      throw new TypeError("An active geometry probe has invalid dimensions");
    }
    if (kind === 2) {
      const base = slot * curveStride;
      for (let embed = 0; embed < profile.maxEmbedDim; embed += 1) {
        if (!(program.geometryCurveParameters![base + curveOffsets.coordinateScale + embed] > 0)) {
          throw new TypeError("Geometry curve coordinate scales must remain positive in every GPU lane");
        }
      }
      for (let axis = 0; axis < profile.maxIntrinsicDim; axis += 1) {
        if (!(program.geometryCurveParameters![base + curveOffsets.axisPeriod + axis] > 0)) {
          throw new TypeError("Geometry curve periods must remain positive in every GPU lane");
        }
      }
    }
  }
}

function nullableString(value: unknown): boolean {
  return value === null || typeof value === "string";
}

function compileLayer(
  program: StructuredHookProgramBuffers,
  layer: number,
  input: StructuredHookLayer,
): void {
  const groups = input.affineGroups ?? [];
  const probes = input.probes ?? [];
  const curves = input.curves ?? [];
  const geometryProbes = input.geometryProbes ?? [];
  if (groups.length > program.profile.maxAffineGroups) {
    throw new TypeError(`A layer supports at most ${program.profile.maxAffineGroups} affine groups`);
  }
  if (probes.length > program.profile.maxProbes) {
    throw new TypeError(`A layer supports at most ${program.profile.maxProbes} probes`);
  }
  if (curves.length > program.profile.maxCurves) {
    throw new TypeError(`A layer supports at most ${program.profile.maxCurves} curved groups`);
  }
  if (geometryProbes.length > program.profile.maxGeometryProbes) {
    throw new TypeError(
      `A layer supports at most ${program.profile.maxGeometryProbes} geometry probes`,
    );
  }
  if (
    (input.geometryWhitener !== undefined && input.geometryWhitener !== null) ||
    geometryProbes.length > 0
  ) {
    if (program.format !== STRUCTURED_HOOK_FORMAT) {
      throw new TypeError("Geometry probes require the drowse-structured-v3 program");
    }
    if (input.geometryWhitener === undefined || input.geometryWhitener === null) {
      throw new TypeError("Geometry probes require one exact layer whitener");
    }
    compileGeometryWhitener(program, layer, input.geometryWhitener);
  }
  groups.forEach((group, groupIndex) => compileAffineGroup(program, layer, groupIndex, group));
  probes.forEach((probe, probeIndex) => compileProbe(program, layer, probeIndex, probe));
  curves.forEach((curve, curveIndex) => compileCurve(program, layer, curveIndex, curve));
  geometryProbes.forEach((probe, probeIndex) =>
    compileGeometryProbe(program, layer, probeIndex, probe)
  );
}

function compileGeometryWhitener(
  program: StructuredHookProgramBuffers,
  layer: number,
  input: StructuredGeometryWhitener,
): void {
  const rank = input.rank;
  if (
    !Number.isSafeInteger(rank) || rank < 1 ||
    rank > program.profile.maxWhitenerRank || input.basis.length !== rank ||
    input.correction.length !== rank || !(input.ridge > 0)
  ) {
    throw new TypeError("Geometry whitener exceeds the structured profile");
  }
  program.whitenerRank![layer] = rank;
  program.whitenerRidge![layer] = finiteFloat(input.ridge, "geometry whitener ridge");
  for (let component = 0; component < rank; component += 1) {
    copyVector(
      input.basis[component],
      program.whitenerBasis!,
      (layer * program.profile.maxWhitenerRank + component) * program.hiddenSize,
      program.hiddenSize,
      "geometry whitener basis",
    );
    program.whitenerCorrection![layer * program.profile.maxWhitenerRank + component] =
      finiteFloat(input.correction[component], "geometry whitener correction");
  }
}

function compileGeometryProbe(
  program: StructuredHookProgramBuffers,
  layer: number,
  probe: number,
  input: StructuredGeometryProbe,
): void {
  const rank = input.basis.length;
  const intrinsicDim = input.coordinateBias.length;
  const candidateCount = input.candidateWhite.length;
  const slot = layer * program.profile.maxGeometryProbes + probe;
  if (
    rank < 1 || rank > program.profile.maxRank ||
    intrinsicDim < 1 || intrinsicDim > program.profile.maxIntrinsicDim ||
    candidateCount < 1 || candidateCount > program.profile.maxGeometryCandidates ||
    input.gramInverse.length !== rank * rank || input.cholesky.length !== rank * rank
  ) {
    throw new TypeError("Geometry probe dimensions exceed the structured profile");
  }
  if (
    input.candidateWhite.some((row) => row.length !== rank) ||
    input.coordinateMap.length !== intrinsicDim ||
    input.coordinateMap.some((row) => row.length !== rank)
  ) {
    throw new TypeError("Geometry probe matrices have inconsistent dimensions");
  }
  program.geometryActive![slot] = input.active ? 1 : 0;
  program.geometryKind![slot] = input.curve == null ? 1 : 2;
  program.geometryRank![slot] = rank;
  program.geometryIntrinsicDim![slot] = intrinsicDim;
  program.geometryCandidateCount![slot] = candidateCount;
  copyVector(
    input.mean,
    program.geometryMean!,
    slot * program.hiddenSize,
    program.hiddenSize,
    "geometry mean",
  );
  copyVector(
    input.inverseMean,
    program.geometryInverseMean!,
    slot * program.hiddenSize,
    program.hiddenSize,
    "geometry inverse mean",
  );
  for (let axis = 0; axis < rank; axis += 1) {
    copyVector(
      input.basis[axis],
      program.geometryBasis!,
      (slot * program.profile.maxRank + axis) * program.hiddenSize,
      program.hiddenSize,
      "geometry basis",
    );
    for (let column = 0; column < rank; column += 1) {
      const target = (slot * program.profile.maxRank + axis) * program.profile.maxRank + column;
      program.geometryGramInverse![target] = finiteFloat(
        input.gramInverse[axis * rank + column],
        "geometry gram inverse",
      );
      program.geometryCholesky![target] = finiteFloat(
        input.cholesky[axis * rank + column],
        "geometry cholesky",
      );
    }
  }
  input.candidateWhite.forEach((row, candidate) => {
    for (let axis = 0; axis < rank; axis += 1) {
      program.geometryNodeWhite![
        (slot * program.profile.maxGeometryCandidates + candidate) *
          program.profile.maxRank + axis
      ] = finiteFloat(row[axis], "geometry candidate");
    }
  });
  input.coordinateMap.forEach((row, coordinate) => {
    program.geometryCoordBias![slot * program.profile.maxIntrinsicDim + coordinate] =
      finiteFloat(input.coordinateBias[coordinate], "geometry coordinate bias");
    for (let axis = 0; axis < rank; axis += 1) {
      program.geometryCoordMap![
        (slot * program.profile.maxIntrinsicDim + coordinate) *
          program.profile.maxRank + axis
      ] = finiteFloat(row[axis], "geometry coordinate map");
    }
  });
  if (input.curve != null) compileGeometryCurve(program, slot, input);
}

function compileGeometryCurve(
  program: StructuredHookProgramBuffers,
  slot: number,
  input: StructuredGeometryProbe,
): void {
  const curve = input.curve!;
  const rank = input.basis.length;
  const nodes = curve.nodeParameters.length;
  const intrinsicDim = curve.axes.length;
  const sourceEmbedDim = curve.coordinateOffset.length;
  if (
    curve.domainKind !== undefined && curve.domainKind !== "box" &&
    curve.domainKind !== "custom" && curve.domainKind !== "sphere"
  ) {
    throw new TypeError("Geometry curve domain kind is invalid");
  }
  const sphere = curve.domainKind === "sphere";
  const embedDim = sphere ? intrinsicDim + 1 : intrinsicDim * 2;
  const nodeCoordinates = input.curveNodeCoordinates ?? [];
  const nodeValues = input.curveNodeValues ?? [];
  if (
    nodes < 2 || nodes > program.profile.maxCurveNodes ||
    intrinsicDim !== input.coordinateBias.length ||
    intrinsicDim > program.profile.maxIntrinsicDim ||
    embedDim > program.profile.maxEmbedDim ||
    nodeCoordinates.length !== nodes || nodeValues.length !== nodes ||
    nodeCoordinates.some((row) => row.length !== intrinsicDim) ||
    nodeValues.some((row) => row.length !== rank)
  ) {
    throw new TypeError("Geometry curve dimensions exceed the structured profile");
  }
  const expectedSourceEmbedDim = sphere
    ? intrinsicDim + 1
    : curve.axes.reduce((sum, axis) => sum + (axis.periodic ? 2 : 1), 0);
  if (
    sourceEmbedDim !== expectedSourceEmbedDim ||
    curve.coordinateScale.length !== sourceEmbedDim ||
    curve.origin.length !== intrinsicDim || curve.target.length !== intrinsicDim ||
    curve.rbfWeights.length !== nodes || curve.polynomial.length !== sourceEmbedDim + 1 ||
    curve.nodeParameters.some((row) => row.length !== sourceEmbedDim) ||
    curve.rbfWeights.some((row) => row.length !== rank) ||
    curve.polynomial.some((row) => row.length !== rank)
  ) {
    throw new TypeError("Geometry curve RBF arrays have inconsistent dimensions");
  }
  const offsets = curveParameterOffsets(program.profile);
  const packed = program.geometryCurveParameters!;
  const base = slot * structuredCurveParameterStride(program.profile);
  const put = (offset: number, value: number, label: string): void => {
    packed[base + offset] = finiteFloat(value, label, true);
  };
  put(offsets.active, curve.active ? 1 : 0, "geometry curve active");
  put(offsets.intrinsicDim, intrinsicDim, "geometry curve intrinsic dimension");
  program.geometryDomainKind![slot] = sphere ? 2 : 1;
  const embedTargets: number[] = [];
  if (sphere) {
    for (let embed = 0; embed < sourceEmbedDim; embed += 1) embedTargets.push(embed);
  } else {
    for (let intrinsic = 0; intrinsic < intrinsicDim; intrinsic += 1) {
      const width = curve.axes[intrinsic].periodic ? 2 : 1;
      for (let component = 0; component < width; component += 1) {
        embedTargets.push(intrinsic * 2 + component);
      }
    }
  }
  embedTargets.forEach((targetEmbed, sourceEmbed) => {
      put(
        offsets.coordinateOffset + targetEmbed,
        curve.coordinateOffset[sourceEmbed],
        "geometry curve coordinate offset",
      );
      put(
        offsets.coordinateScale + targetEmbed,
        curve.coordinateScale[sourceEmbed],
        "geometry curve coordinate scale",
      );
      for (let node = 0; node < nodes; node += 1) {
        put(
          offsets.nodeParameters + node * program.profile.maxEmbedDim + targetEmbed,
          curve.nodeParameters[node][sourceEmbed],
          "geometry curve node parameter",
        );
      }
      const polynomial = sourceEmbed + 1;
      for (let axis = 0; axis < rank; axis += 1) {
        put(
          offsets.polynomial + (targetEmbed + 1) * program.profile.maxRank + axis,
          curve.polynomial[polynomial][axis],
          "geometry curve polynomial",
        );
      }
  });
  for (let embed = 0; embed < program.profile.maxEmbedDim; embed += 1) {
    if (!embedTargets.includes(embed)) {
      put(offsets.coordinateScale + embed, 1, "geometry curve coordinate scale");
    }
  }
  for (let axis = 0; axis < rank; axis += 1) {
    put(
      offsets.polynomial + axis,
      curve.polynomial[0][axis],
      "geometry curve polynomial",
    );
  }
  for (let node = 0; node < nodes; node += 1) {
    for (let axis = 0; axis < rank; axis += 1) {
      put(
        offsets.rbfWeights + node * program.profile.maxRank + axis,
        curve.rbfWeights[node][axis],
        "geometry curve RBF weight",
      );
      program.geometryCurveNodeValues![
        (slot * program.profile.maxCurveNodes + node) * program.profile.maxRank + axis
      ] = finiteFloat(nodeValues[node][axis], "geometry curve node value");
    }
    for (let coordinate = 0; coordinate < intrinsicDim; coordinate += 1) {
      program.geometryCurveNodeCoords![
        (slot * program.profile.maxCurveNodes + node) *
          program.profile.maxIntrinsicDim + coordinate
      ] = finiteFloat(nodeCoordinates[node][coordinate], "geometry curve node coordinate");
    }
  }
  for (let coordinate = 0; coordinate < intrinsicDim; coordinate += 1) {
    const spec = curve.axes[coordinate];
    if (!(spec.lowerBound <= spec.upperBound) || !(spec.period > 0)) {
      throw new TypeError("Geometry curve coordinate bounds or period are invalid");
    }
    put(offsets.origin + coordinate, curve.origin[coordinate], "geometry curve origin");
    put(offsets.target + coordinate, curve.target[coordinate], "geometry curve target");
    put(offsets.bounds + coordinate * 2, spec.lowerBound, "geometry curve lower bound");
    put(offsets.bounds + coordinate * 2 + 1, spec.upperBound, "geometry curve upper bound");
    put(offsets.axisPeriodic + coordinate, spec.periodic ? 1 : 0, "geometry periodic axis");
    put(offsets.axisPeriod + coordinate, spec.period, "geometry curve period");
  }
  put(offsets.along, curve.along, "geometry curve along");
  put(offsets.onto, curve.onto, "geometry curve onto");
  put(offsets.damping, curve.damping ?? 1e-3, "geometry curve damping");
  const sigmaWeights = curve.sigmaRbfWeights ?? [];
  const sigmaPolynomial = curve.sigmaPolynomial ?? [];
  if ((sigmaWeights.length === 0) !== (sigmaPolynomial.length === 0)) {
    throw new TypeError("Geometry curve sigma arrays must be supplied together");
  }
  if (sigmaWeights.length > 0) {
    if (sigmaWeights.length !== nodes || sigmaPolynomial.length !== sourceEmbedDim + 1) {
      throw new TypeError("Geometry curve sigma arrays have inconsistent dimensions");
    }
    put(offsets.sigmaPresent, 1, "geometry curve sigma present");
    for (let node = 0; node < nodes; node += 1) {
      put(offsets.sigmaRbfWeights + node, sigmaWeights[node], "geometry curve sigma weight");
    }
    put(offsets.sigmaPolynomial, sigmaPolynomial[0], "geometry curve sigma polynomial");
    embedTargets.forEach((targetEmbed, sourceEmbed) => {
      put(
        offsets.sigmaPolynomial + targetEmbed + 1,
        sigmaPolynomial[sourceEmbed + 1],
        "geometry curve sigma polynomial",
      );
    });
  }
  program.geometryCurveNodeCount![slot] = nodes;
  const foot = input.foot ?? curve.origin;
  copyVector(
    foot,
    program.geometryFeet!,
    slot * program.profile.maxIntrinsicDim,
    intrinsicDim,
    "geometry curve foot",
  );
}

export function geometryCoordinateMean(
  program: StructuredHookProgramBuffers,
  rows: readonly { slot: number; weight: number; coords: number[] }[],
): number[] {
  const dimensions = rows[0].coords.length;
  const slot = rows[0].slot;
  const total = rows.reduce((sum, row) => sum + row.weight, 0);
  const reference = rows.reduce((best, row) => row.weight > best.weight ? row : best).coords;
  if (program.geometryDomainKind?.[slot] === 2) {
    const embedded = new Float64Array(dimensions + 1);
    for (const row of rows) {
      const point = embedSpherePoint(row.coords, dimensions);
      point.forEach((value, axis) => { embedded[axis] += value * row.weight / total; });
    }
    return vectorNorm(embedded) < 1e-6 ? [...reference] : Array.from(unembedSpherePoint(embedded, dimensions));
  }
  const offsets = curveParameterOffsets(program.profile);
  const base = slot * structuredCurveParameterStride(program.profile);
  const parameters = program.geometryCurveParameters;
  return Array.from({ length: dimensions }, (_, axis) => {
    const periodic = program.geometryKind?.[slot] === 2 && parameters?.[base + offsets.axisPeriodic + axis] === 1;
    if (!periodic) return rows.reduce((sum, row) => sum + row.coords[axis] * row.weight / total, 0);
    const period = parameters![base + offsets.axisPeriod + axis];
    let sine = 0;
    let cosine = 0;
    for (const row of rows) {
      const angle = row.coords[axis] * 2 * Math.PI / period;
      sine += Math.sin(angle) * row.weight / total;
      cosine += Math.cos(angle) * row.weight / total;
    }
    // Opposing readings have no unique circular mean; retain the strongest
    // observed layer instead of manufacturing an unrelated midpoint.
    return Math.hypot(sine, cosine) < 1e-6
      ? modulo(reference[axis], period)
      : modulo(Math.atan2(sine, cosine) * period / (2 * Math.PI), period);
  });
}

function curveParameterOffsets(profile: StructuredHookCapacityProfile): {
  active: number;
  intrinsicDim: number;
  nodeParameters: number;
  rbfWeights: number;
  polynomial: number;
  coordinateOffset: number;
  coordinateScale: number;
  origin: number;
  target: number;
  along: number;
  onto: number;
  bounds: number;
  axisPeriodic: number;
  axisPeriod: number;
  sigmaPresent: number;
  sigmaRbfWeights: number;
  sigmaPolynomial: number;
  damping: number;
} {
  const offsets = {
    active: 0,
    intrinsicDim: 1,
    nodeParameters: 2,
    rbfWeights: 0,
    polynomial: 0,
    coordinateOffset: 0,
    coordinateScale: 0,
    origin: 0,
    target: 0,
    along: 0,
    onto: 0,
    bounds: 0,
    axisPeriodic: 0,
    axisPeriod: 0,
    sigmaPresent: 0,
    sigmaRbfWeights: 0,
    sigmaPolynomial: 0,
    damping: 0,
  };
  offsets.rbfWeights = offsets.nodeParameters +
    profile.maxCurveNodes * profile.maxEmbedDim;
  offsets.polynomial = offsets.rbfWeights +
    profile.maxCurveNodes * profile.maxRank;
  offsets.coordinateOffset = offsets.polynomial +
    (profile.maxEmbedDim + 1) * profile.maxRank;
  offsets.coordinateScale = offsets.coordinateOffset + profile.maxEmbedDim;
  offsets.origin = offsets.coordinateScale + profile.maxEmbedDim;
  offsets.target = offsets.origin + profile.maxIntrinsicDim;
  offsets.along = offsets.target + profile.maxIntrinsicDim;
  offsets.onto = offsets.along + 1;
  offsets.bounds = offsets.onto + 1;
  offsets.axisPeriodic = offsets.bounds + profile.maxIntrinsicDim * 2;
  offsets.axisPeriod = offsets.axisPeriodic + profile.maxIntrinsicDim;
  offsets.sigmaPresent = offsets.axisPeriod + profile.maxIntrinsicDim;
  offsets.sigmaRbfWeights = offsets.sigmaPresent + 1;
  offsets.sigmaPolynomial = offsets.sigmaRbfWeights + profile.maxCurveNodes;
  offsets.damping = offsets.sigmaPolynomial + profile.maxEmbedDim + 1;
  if (offsets.damping + 1 !== structuredCurveParameterStride(profile)) {
    throw new TypeError("Geometry curve parameter stride is inconsistent");
  }
  return offsets;
}

function compileAffineGroup(
  program: StructuredHookProgramBuffers,
  layer: number,
  group: number,
  input: StructuredAffineGroup,
): void {
  const rank = input.basis.length;
  if (rank === 0 || rank > program.profile.maxRank) {
    throw new TypeError(`Affine group rank must be between 1 and ${program.profile.maxRank}`);
  }
  if (input.target.length !== rank || input.kappa.length !== rank) {
    throw new TypeError("Affine target and kappa must match the group rank");
  }
  const slot = layer * program.profile.maxAffineGroups + group;
  program.affineActive[slot] = input.active ? 1 : 0;
  program.affineAlong[slot] = finiteFloat(input.along, "affine along");
  copyVector(input.neutral, program.affineNeutral, slot * program.hiddenSize, program.hiddenSize, "affine neutral");
  for (let axis = 0; axis < rank; axis += 1) {
    const axisSlot = slot * program.profile.maxRank + axis;
    copyVector(input.basis[axis], program.affineBasis, axisSlot * program.hiddenSize, program.hiddenSize, "affine basis");
    requireUnitVector(program.affineBasis, axisSlot * program.hiddenSize, program.hiddenSize, "affine basis");
    program.affineTarget[axisSlot] = finiteFloat(input.target[axis], "affine target");
    program.affineKappa[axisSlot] = finiteFloat(input.kappa[axis], "affine kappa");
    for (let previous = 0; previous < axis; previous += 1) {
      const previousSlot = slot * program.profile.maxRank + previous;
      const overlap = dotProductSlices(
        program.affineBasis,
        previousSlot * program.hiddenSize,
        axisSlot * program.hiddenSize,
        program.hiddenSize,
      );
      if (Math.abs(overlap) > 1e-4) {
        throw new TypeError("Affine group basis rows must be orthonormal");
      }
    }
  }
}

function compileProbe(
  program: StructuredHookProgramBuffers,
  layer: number,
  probe: number,
  input: StructuredProbe,
): void {
  const slot = layer * program.profile.maxProbes + probe;
  if (input.kind === "disabled") {
    if (
      input.direction.length !== program.hiddenSize ||
      input.bias !== 0 ||
      (input.threshold ?? 0) !== 0
    ) {
      throw new TypeError("Disabled probe slots must contain zero-valued placeholders");
    }
    return;
  }
  program.probeKind[slot] = input.kind === "jlens"
    ? 3
    : input.kind === "sae_jump_relu" ? 4 : input.kind === "sae" ? 2 : 1;
  program.probeBias[slot] = finiteFloat(input.bias, "probe bias");
  program.probeThreshold[slot] = finiteFloat(input.threshold ?? 0, "probe threshold");
  copyVector(input.direction, program.probeDirection, slot * program.hiddenSize, program.hiddenSize, "probe direction");
}

function compileCurve(
  program: StructuredHookProgramBuffers,
  layer: number,
  curve: number,
  input: StructuredCurve,
): void {
  const rank = input.basis.length;
  const nodes = input.nodeParameters.length;
  const intrinsicDim = input.axes.length;
  const sourceEmbedDim = input.coordinateOffset.length;
  const domainKind = input.domainKind ?? "box";
  if (domainKind !== "box" && domainKind !== "custom" && domainKind !== "sphere") {
    throw new TypeError("Curved hook domain kind is invalid");
  }
  const sphere = domainKind === "sphere";
  const embedDim = sphere ? intrinsicDim + 1 : intrinsicDim * 2;
  const slot = layer * program.profile.maxCurves + curve;
  if (rank === 0 || rank > program.profile.maxRank) {
    throw new TypeError(`Curved hook rank must be between 1 and ${program.profile.maxRank}`);
  }
  if (nodes < 2 || nodes > program.profile.maxCurveNodes) {
    throw new TypeError(`Curved hooks support between 2 and ${program.profile.maxCurveNodes} nodes`);
  }
  if (
    intrinsicDim === 0 || intrinsicDim > program.profile.maxIntrinsicDim ||
    sourceEmbedDim === 0 || embedDim > program.profile.maxEmbedDim
  ) {
    throw new TypeError("Curved intrinsic or embedding dimension exceeds the structured limits");
  }
  const expectedEmbedDim = sphere
    ? intrinsicDim + 1
    : input.axes.reduce((sum, axis) => sum + (axis.periodic ? 2 : 1), 0);
  if (sourceEmbedDim !== expectedEmbedDim) {
    throw new TypeError("Curved coordinate arrays do not match the domain embedding");
  }
  if (
    input.coordinateScale.length !== sourceEmbedDim || input.origin.length !== intrinsicDim ||
    input.target.length !== intrinsicDim || input.rbfWeights.length !== nodes ||
    input.polynomial.length !== sourceEmbedDim + 1 ||
    input.nodeParameters.some((row) => row.length !== sourceEmbedDim)
  ) {
    throw new TypeError("Curved RBF arrays do not match the domain shape");
  }
  if (input.onto < 0 || input.onto > 1) throw new TypeError("Curved onto must be between zero and one");
  program.curveActive[slot] = input.active ? 1 : 0;
  program.curveRank[slot] = rank;
  program.curveIntrinsicDim[slot] = intrinsicDim;
  program.curveEmbedDim[slot] = embedDim;
  program.curveNodeCount[slot] = nodes;
  if (sphere && program.curveDomainKind === undefined) {
    throw new TypeError("Sphere curves require drowse-structured-v3");
  }
  if (program.curveDomainKind !== undefined) {
    program.curveDomainKind[slot] = sphere ? 2 : 1;
  }
  copyVector(input.neutral, program.curveNeutral, slot * program.hiddenSize, program.hiddenSize, "curve neutral");
  for (let axis = 0; axis < rank; axis += 1) {
    const axisOffset = (slot * program.profile.maxRank + axis) * program.hiddenSize;
    copyVector(input.basis[axis], program.curveBasis, axisOffset, program.hiddenSize, "curve basis");
    requireUnitVector(program.curveBasis, axisOffset, program.hiddenSize, "curve basis");
    for (let previous = 0; previous < axis; previous += 1) {
      const previousOffset = (slot * program.profile.maxRank + previous) * program.hiddenSize;
      if (Math.abs(dotProductSlices(
        program.curveBasis,
        previousOffset,
        axisOffset,
        program.hiddenSize,
      )) > 1e-4) {
        throw new TypeError("Curve basis rows must be orthonormal");
      }
    }
  }
  for (let node = 0; node < nodes; node += 1) {
    if (input.rbfWeights[node].length !== rank) throw new TypeError("Curve RBF weight rank is inconsistent");
    for (let axis = 0; axis < rank; axis += 1) {
      const weightSlot = (slot * program.profile.maxCurveNodes + node) * program.profile.maxRank + axis;
      program.curveRbfWeights[weightSlot] = finiteFloat(input.rbfWeights[node][axis], "curve RBF weight");
    }
  }
  if (input.polynomial[0].length !== rank) throw new TypeError("Curve polynomial rank is inconsistent");
  for (let axis = 0; axis < rank; axis += 1) {
    program.curvePolynomial[
      (slot * (program.profile.maxEmbedDim + 1)) * program.profile.maxRank + axis
    ] = finiteFloat(input.polynomial[0][axis], "curve polynomial");
  }
  const embedTargets: number[] = [];
  if (sphere) {
    for (let embed = 0; embed < sourceEmbedDim; embed += 1) embedTargets.push(embed);
  } else {
    for (let intrinsic = 0; intrinsic < intrinsicDim; intrinsic += 1) {
      const width = input.axes[intrinsic].periodic ? 2 : 1;
      for (let component = 0; component < width; component += 1) {
        embedTargets.push(intrinsic * 2 + component);
      }
    }
  }
  embedTargets.forEach((targetEmbed, sourceEmbed) => {
      program.curveCoordinateOffset[slot * program.profile.maxEmbedDim + targetEmbed] =
        finiteFloat(input.coordinateOffset[sourceEmbed], "curve coordinate offset");
      program.curveCoordinateScale[slot * program.profile.maxEmbedDim + targetEmbed] =
        finiteFloat(input.coordinateScale[sourceEmbed], "curve coordinate scale");
      for (let node = 0; node < nodes; node += 1) {
        program.curveNodeParameters[
          (slot * program.profile.maxCurveNodes + node) * program.profile.maxEmbedDim + targetEmbed
        ] = finiteFloat(input.nodeParameters[node][sourceEmbed], "curve node parameter");
      }
      const polynomial = sourceEmbed + 1;
      if (input.polynomial[polynomial].length !== rank) {
        throw new TypeError("Curve polynomial rank is inconsistent");
      }
      for (let axis = 0; axis < rank; axis += 1) {
        program.curvePolynomial[
          (slot * (program.profile.maxEmbedDim + 1) + targetEmbed + 1) * program.profile.maxRank + axis
          ] = finiteFloat(input.polynomial[polynomial][axis], "curve polynomial");
      }
  });
  for (let polynomial = 0; polynomial < sourceEmbedDim + 1; polynomial += 1) {
    if (input.polynomial[polynomial].length !== rank) throw new TypeError("Curve polynomial rank is inconsistent");
    if (polynomial === 0) continue;
    for (let axis = 0; axis < rank; axis += 1) {
      finiteFloat(input.polynomial[polynomial][axis], "curve polynomial");
    }
  }
  const sigmaWeights = input.sigmaRbfWeights ?? [];
  const sigmaPolynomial = input.sigmaPolynomial ?? [];
  if ((sigmaWeights.length === 0) !== (sigmaPolynomial.length === 0)) {
    throw new TypeError("Curve sigma weights and polynomial must be supplied together");
  }
  if (sigmaWeights.length > 0) {
    if (sigmaWeights.length !== nodes || sigmaPolynomial.length !== sourceEmbedDim + 1) {
      throw new TypeError("Curve sigma arrays do not match the RBF shape");
    }
    program.curveSigmaPresent[slot] = 1;
    for (let node = 0; node < nodes; node += 1) {
      program.curveSigmaRbfWeights[slot * program.profile.maxCurveNodes + node] = finiteFloat(sigmaWeights[node], "curve sigma weight");
    }
    program.curveSigmaPolynomial[slot * (program.profile.maxEmbedDim + 1)] =
      finiteFloat(sigmaPolynomial[0], "curve sigma polynomial");
    embedTargets.forEach((targetEmbed, sourceEmbed) => {
      program.curveSigmaPolynomial[
        slot * (program.profile.maxEmbedDim + 1) + targetEmbed + 1
      ] = finiteFloat(sigmaPolynomial[sourceEmbed + 1], "curve sigma polynomial");
    });
  }
  for (let axis = 0; axis < intrinsicDim; axis += 1) {
    const axisSlot = slot * program.profile.maxIntrinsicDim + axis;
    const spec = input.axes[axis];
    if (!(spec.lowerBound <= spec.upperBound) || !(spec.period > 0)) {
      throw new TypeError("Curved coordinate bounds or period are invalid");
    }
    program.curveOrigin[axisSlot] = finiteFloat(input.origin[axis], "curve origin");
    program.curveTarget[axisSlot] = finiteFloat(input.target[axis], "curve target");
    program.curveAxisPeriodic[axisSlot] = spec.periodic ? 1 : 0;
    program.curveAxisPeriod[axisSlot] = finiteFloat(spec.period, "curve period");
    program.curveBounds[axisSlot * 2] = finiteFloat(spec.lowerBound, "curve lower bound", true);
    program.curveBounds[axisSlot * 2 + 1] = finiteFloat(spec.upperBound, "curve upper bound", true);
  }
  program.curveAlong[slot] = finiteFloat(input.along, "curve along");
  program.curveOnto[slot] = finiteFloat(input.onto, "curve onto");
  program.curveDamping[slot] = finiteFloat(input.damping ?? 1e-3, "curve damping");
}

function runAffineGroups(
  program: StructuredHookProgramBuffers,
  layer: number,
  values: Float32Array,
): void {
  const d = program.hiddenSize;
  for (let group = 0; group < program.profile.maxAffineGroups; group += 1) {
    const slot = layer * program.profile.maxAffineGroups + group;
    if (program.affineActive[slot] !== 1) continue;
    const deltas = new Float32Array(program.profile.maxRank);
    for (let axis = 0; axis < program.profile.maxRank; axis += 1) {
      const axisSlot = slot * program.profile.maxRank + axis;
      const vector = axisSlot * d;
      let coordinate = 0;
      for (let hidden = 0; hidden < d; hidden += 1) {
        coordinate = Math.fround(coordinate + Math.fround(
          Math.fround(values[hidden] - program.affineNeutral[slot * d + hidden]) * program.affineBasis[vector + hidden],
        ));
      }
      deltas[axis] = Math.fround(program.affineAlong[slot] * Math.fround(
        program.affineTarget[axisSlot] - Math.fround(program.affineKappa[axisSlot] * coordinate),
      ));
    }
    for (let hidden = 0; hidden < d; hidden += 1) {
      let next = values[hidden];
      for (let axis = 0; axis < program.profile.maxRank; axis += 1) {
        const axisSlot = slot * program.profile.maxRank + axis;
        next = Math.fround(next + Math.fround(deltas[axis] * program.affineBasis[axisSlot * d + hidden]));
      }
      values[hidden] = next;
    }
  }
}

function runProbes(
  program: StructuredHookProgramBuffers,
  layer: number,
  values: Float32Array,
): Float32Array {
  const output = new Float32Array(program.profile.maxProbes);
  for (let probe = 0; probe < program.profile.maxProbes; probe += 1) {
    const slot = layer * program.profile.maxProbes + probe;
    if (program.probeKind[slot] === 0) continue;
    let value = program.probeBias[slot];
    for (let hidden = 0; hidden < program.hiddenSize; hidden += 1) {
      value = Math.fround(value + Math.fround(
        values[hidden] * program.probeDirection[slot * program.hiddenSize + hidden],
      ));
    }
    output[probe] = program.probeKind[slot] === 2
      ? Math.max(0, value)
      : program.probeKind[slot] === 3
        ? 0
        : program.probeKind[slot] === 4
          ? value > program.probeThreshold[slot] ? value : 0
          : value;
  }
  return output;
}

function runGeometryProbes(
  program: StructuredHookProgramBuffers,
  state: StructuredHookState,
  layer: number,
  values: Float32Array,
  decode: boolean,
): Float32Array {
  const profile = program.profile;
  const output = new Float32Array(
    profile.maxGeometryProbes * profile.geometryOutputStride,
  );
  const whitenerRank = program.whitenerRank![layer];
  if (whitenerRank === 0) return output;
  const inverseHidden = new Float32Array(program.hiddenSize);
  const ridge = program.whitenerRidge![layer];
  for (let hidden = 0; hidden < program.hiddenSize; hidden += 1) {
    inverseHidden[hidden] = values[hidden] / ridge;
  }
  for (let component = 0; component < whitenerRank; component += 1) {
    const basisOffset =
      (layer * profile.maxWhitenerRank + component) * program.hiddenSize;
    let coordinate = 0;
    for (let hidden = 0; hidden < program.hiddenSize; hidden += 1) {
      coordinate += values[hidden] * program.whitenerBasis![basisOffset + hidden];
    }
    const correction = coordinate * program.whitenerCorrection![
      layer * profile.maxWhitenerRank + component
    ];
    for (let hidden = 0; hidden < program.hiddenSize; hidden += 1) {
      inverseHidden[hidden] += correction * program.whitenerBasis![basisOffset + hidden];
    }
  }
  for (let probe = 0; probe < profile.maxGeometryProbes; probe += 1) {
    const slot = layer * profile.maxGeometryProbes + probe;
    if (program.geometryActive![slot] !== 1) continue;
    const rank = program.geometryRank![slot];
    const intrinsicDim = program.geometryIntrinsicDim![slot];
    const candidateCount = program.geometryCandidateCount![slot];
    const outputOffset = probe * profile.geometryOutputStride;
    const inverseCentered = new Float32Array(program.hiddenSize);
    const centered = new Float32Array(program.hiddenSize);
    let totalSquared = 0;
    for (let hidden = 0; hidden < program.hiddenSize; hidden += 1) {
      inverseCentered[hidden] = inverseHidden[hidden] -
        program.geometryInverseMean![slot * program.hiddenSize + hidden];
      centered[hidden] = values[hidden] -
        program.geometryMean![slot * program.hiddenSize + hidden];
      totalSquared += centered[hidden] * inverseCentered[hidden];
    }
    const gradient = new Float32Array(rank);
    for (let axis = 0; axis < rank; axis += 1) {
      const basisOffset = (slot * profile.maxRank + axis) * program.hiddenSize;
      for (let hidden = 0; hidden < program.hiddenSize; hidden += 1) {
        gradient[axis] += program.geometryBasis![basisOffset + hidden] *
          inverseCentered[hidden];
      }
    }
    const reduced = new Float32Array(rank);
    for (let row = 0; row < rank; row += 1) {
      for (let column = 0; column < rank; column += 1) {
        reduced[row] += program.geometryGramInverse![
          (slot * profile.maxRank + row) * profile.maxRank + column
        ] * gradient[column];
      }
    }
    let parallelSquared = 0;
    for (let axis = 0; axis < rank; axis += 1) {
      parallelSquared += gradient[axis] * reduced[axis];
    }
    const fraction = Math.max(0, Math.min(1,
      Math.sqrt(Math.max(0, parallelSquared)) /
        Math.max(Math.sqrt(Math.max(0, totalSquared)), 1e-8),
    ));
    const white = new Float32Array(rank);
    for (let column = 0; column < rank; column += 1) {
      for (let axis = 0; axis < rank; axis += 1) {
        white[column] += reduced[axis] * program.geometryCholesky![
          (slot * profile.maxRank + axis) * profile.maxRank + column
        ];
      }
    }
    output[outputOffset] = 1;
    output[outputOffset + 1] = fraction;
    let residual = 0;
    let membership = 1;
    let coordinates: Float32Array;
    if (program.geometryKind![slot] === 2) {
      const result = solveGeometryCurveFoot(program, state, slot, reduced, decode);
      coordinates = result.foot;
      residual = result.norm / Math.max(vectorNorm(reduced), 1e-8);
      const offsets = curveParameterOffsets(profile);
      const packedOffset = slot * structuredCurveParameterStride(profile);
      if (program.geometryCurveParameters![packedOffset + offsets.sigmaPresent] === 1) {
        const sigma = Math.exp(evaluateGeometrySigma(program, slot, result.foot));
        membership = Math.exp(
          -(result.norm ** 2) / (2 * Math.max(sigma ** 2, 1e-8)),
        );
      }
    } else {
      coordinates = new Float32Array(intrinsicDim);
      for (let coordinate = 0; coordinate < intrinsicDim; coordinate += 1) {
        coordinates[coordinate] = program.geometryCoordBias![
          slot * profile.maxIntrinsicDim + coordinate
        ];
        for (let axis = 0; axis < rank; axis += 1) {
          coordinates[coordinate] += program.geometryCoordMap![
            (slot * profile.maxIntrinsicDim + coordinate) * profile.maxRank + axis
          ] * reduced[axis];
        }
      }
    }
    output[outputOffset + 2] = residual;
    output[outputOffset + 3] = membership;
    for (let coordinate = 0; coordinate < intrinsicDim; coordinate += 1) {
      output[outputOffset + 4 + coordinate] = coordinates[coordinate];
    }
    const distanceOffset = outputOffset + 4 + profile.maxIntrinsicDim;
    for (let candidate = 0; candidate < candidateCount; candidate += 1) {
      let squared = 0;
      for (let axis = 0; axis < rank; axis += 1) {
        const difference = program.geometryNodeWhite![
          (slot * profile.maxGeometryCandidates + candidate) * profile.maxRank + axis
        ] - white[axis];
        squared += difference * difference;
      }
      output[distanceOffset + candidate] = Math.sqrt(squared);
    }
  }
  return output;
}

function solveGeometryCurveFoot(
  program: StructuredHookProgramBuffers,
  state: StructuredHookState,
  slot: number,
  query: Float32Array,
  decode: boolean,
): { foot: Float32Array; norm: number } {
  const profile = program.profile;
  const intrinsicDim = program.geometryIntrinsicDim![slot];
  const nodeCount = program.geometryCurveNodeCount![slot];
  const restartCount = decode ? profile.geometryWarmRestarts : profile.geometryFootRestarts;
  const iterationCount = decode ? profile.geometryWarmIterations : profile.geometryFootIterations;
  const candidates = Array.from({ length: nodeCount }, (_, node) => {
    let squared = 0;
    for (let axis = 0; axis < query.length; axis += 1) {
      const difference = query[axis] - program.geometryCurveNodeValues![
        (slot * profile.maxCurveNodes + node) * profile.maxRank + axis
      ];
      squared += difference * difference;
    }
    return { node, squared };
  }).sort((left, right) => left.squared - right.squared);
  const seeds: Float32Array[] = [];
  if (decode && state.geometryFeet !== undefined) {
    seeds.push(state.geometryFeet.slice(
      slot * profile.maxIntrinsicDim,
      slot * profile.maxIntrinsicDim + intrinsicDim,
    ));
  }
  for (const candidate of candidates) {
    if (seeds.length >= restartCount) break;
    seeds.push(program.geometryCurveNodeCoords!.slice(
      (slot * profile.maxCurveNodes + candidate.node) * profile.maxIntrinsicDim,
      (slot * profile.maxCurveNodes + candidate.node) * profile.maxIntrinsicDim + intrinsicDim,
    ));
  }
  if (seeds.length === 0) {
    seeds.push(program.geometryFeet!.slice(
      slot * profile.maxIntrinsicDim,
      slot * profile.maxIntrinsicDim + intrinsicDim,
    ));
  }
  let bestFoot = seeds[0];
  let bestNorm = Number.POSITIVE_INFINITY;
  for (const seed of seeds) {
    let foot = clampGeometryCurvePoint(program, slot, seed);
    for (let iteration = 0; iteration < iterationCount; iteration += 1) {
      const evaluated = evaluateGeometryCurve(program, slot, foot, query.length);
      const normal = new Float32Array(intrinsicDim * intrinsicDim);
      const rhs = new Float32Array(intrinsicDim);
      for (let row = 0; row < intrinsicDim; row += 1) {
        for (let axis = 0; axis < query.length; axis += 1) {
          rhs[row] += evaluated.jacobian[axis * intrinsicDim + row] *
            (query[axis] - evaluated.value[axis]);
        }
        for (let column = 0; column < intrinsicDim; column += 1) {
          for (let axis = 0; axis < query.length; axis += 1) {
            normal[row * intrinsicDim + column] +=
              evaluated.jacobian[axis * intrinsicDim + row] *
              evaluated.jacobian[axis * intrinsicDim + column];
          }
        }
      }
      const offsets = curveParameterOffsets(profile);
      const base = slot * structuredCurveParameterStride(profile);
      const damping = program.geometryCurveParameters![base + offsets.damping];
      for (let axis = 0; axis < intrinsicDim; axis += 1) {
        const diagonal = axis * intrinsicDim + axis;
        normal[diagonal] += damping * Math.max(normal[diagonal], 1e-9) + 1e-9;
      }
      const delta = solveLinearSystem(normal, rhs, intrinsicDim);
      foot = clampGeometryCurvePoint(
        program,
        slot,
        Float32Array.from(foot, (value, axis) => value + delta[axis]),
      );
    }
    const value = evaluateGeometryCurve(program, slot, foot, query.length).value;
    let squared = 0;
    for (let axis = 0; axis < query.length; axis += 1) {
      squared += (query[axis] - value[axis]) ** 2;
    }
    const norm = Math.sqrt(squared);
    if (norm < bestNorm) {
      bestNorm = norm;
      bestFoot = foot;
    }
  }
  state.geometryFeet!.set(bestFoot, slot * profile.maxIntrinsicDim);
  return { foot: bestFoot, norm: bestNorm };
}

function evaluateGeometryCurve(
  program: StructuredHookProgramBuffers,
  slot: number,
  foot: Float32Array,
  rank: number,
): { value: Float32Array; jacobian: Float32Array } {
  const profile = program.profile;
  const offsets = curveParameterOffsets(profile);
  const base = slot * structuredCurveParameterStride(profile);
  const intrinsicDim = program.geometryIntrinsicDim![slot];
  const embedDim = program.geometryDomainKind![slot] === 2
    ? intrinsicDim + 1
    : intrinsicDim * 2;
  const embedded = embedGeometryCurvePoint(program, slot, foot);
  const embedJacobian = embedGeometryCurveJacobian(program, slot, foot);
  const normalized = new Float32Array(embedDim);
  for (let embed = 0; embed < embedDim; embed += 1) {
    normalized[embed] = (
      embedded[embed] -
      program.geometryCurveParameters![base + offsets.coordinateOffset + embed]
    ) / program.geometryCurveParameters![base + offsets.coordinateScale + embed];
  }
  const value = new Float32Array(rank);
  const normalizedJacobian = new Float32Array(rank * embedDim);
  const jacobian = new Float32Array(rank * intrinsicDim);
  const nodes = program.geometryCurveNodeCount![slot];
  for (let axis = 0; axis < rank; axis += 1) {
    value[axis] = program.geometryCurveParameters![base + offsets.polynomial + axis];
    for (let embed = 0; embed < embedDim; embed += 1) {
      const coefficient = program.geometryCurveParameters![
        base + offsets.polynomial + (embed + 1) * profile.maxRank + axis
      ];
      value[axis] += normalized[embed] * coefficient;
      normalizedJacobian[axis * embedDim + embed] = coefficient;
    }
    for (let node = 0; node < nodes; node += 1) {
      let squaredRadius = 0;
      const differences = new Float32Array(embedDim);
      for (let embed = 0; embed < embedDim; embed += 1) {
        const difference = normalized[embed] - program.geometryCurveParameters![
          base + offsets.nodeParameters + node * profile.maxEmbedDim + embed
        ];
        differences[embed] = difference;
        squaredRadius += difference * difference;
      }
      const radius = Math.sqrt(squaredRadius);
      const weight = program.geometryCurveParameters![
        base + offsets.rbfWeights + node * profile.maxRank + axis
      ];
      value[axis] += radius ** 3 * weight;
      for (let embed = 0; embed < embedDim; embed += 1) {
        normalizedJacobian[axis * embedDim + embed] +=
          3 * radius * differences[embed] * weight;
      }
    }
    for (let intrinsic = 0; intrinsic < intrinsicDim; intrinsic += 1) {
      for (let embed = 0; embed < embedDim; embed += 1) {
        jacobian[axis * intrinsicDim + intrinsic] +=
          normalizedJacobian[axis * embedDim + embed] /
          program.geometryCurveParameters![base + offsets.coordinateScale + embed] *
          embedJacobian[embed * intrinsicDim + intrinsic];
      }
    }
  }
  return { value, jacobian };
}

function evaluateGeometrySigma(
  program: StructuredHookProgramBuffers,
  slot: number,
  foot: Float32Array,
): number {
  const profile = program.profile;
  const offsets = curveParameterOffsets(profile);
  const base = slot * structuredCurveParameterStride(profile);
  const intrinsicDim = program.geometryIntrinsicDim![slot];
  const embedDim = program.geometryDomainKind![slot] === 2
    ? intrinsicDim + 1
    : intrinsicDim * 2;
  const embedded = embedGeometryCurvePoint(program, slot, foot);
  const normalized = new Float32Array(embedDim);
  let value = program.geometryCurveParameters![base + offsets.sigmaPolynomial];
  for (let embed = 0; embed < embedDim; embed += 1) {
    normalized[embed] = (
      embedded[embed] -
      program.geometryCurveParameters![base + offsets.coordinateOffset + embed]
    ) / program.geometryCurveParameters![base + offsets.coordinateScale + embed];
    value += normalized[embed] * program.geometryCurveParameters![
      base + offsets.sigmaPolynomial + embed + 1
    ];
  }
  for (let node = 0; node < program.geometryCurveNodeCount![slot]; node += 1) {
    let squaredRadius = 0;
    for (let embed = 0; embed < embedDim; embed += 1) {
      const difference = normalized[embed] - program.geometryCurveParameters![
        base + offsets.nodeParameters + node * profile.maxEmbedDim + embed
      ];
      squaredRadius += difference * difference;
    }
    value += Math.sqrt(squaredRadius) ** 3 * program.geometryCurveParameters![
      base + offsets.sigmaRbfWeights + node
    ];
  }
  return value;
}

function embedGeometryCurvePoint(
  program: StructuredHookProgramBuffers,
  slot: number,
  point: ArrayLike<number>,
): Float32Array {
  const profile = program.profile;
  const offsets = curveParameterOffsets(profile);
  const base = slot * structuredCurveParameterStride(profile);
  const intrinsicDim = program.geometryIntrinsicDim![slot];
  if (program.geometryDomainKind![slot] === 2) {
    return embedSpherePoint(point, intrinsicDim);
  }
  const output = new Float32Array(intrinsicDim * 2);
  for (let axis = 0; axis < intrinsicDim; axis += 1) {
    const embed = axis * 2;
    if (program.geometryCurveParameters![base + offsets.axisPeriodic + axis] === 1) {
      const period = program.geometryCurveParameters![base + offsets.axisPeriod + axis];
      const angle = 2 * Math.PI * point[axis] / period;
      output[embed] = Math.cos(angle);
      output[embed + 1] = Math.sin(angle);
    } else {
      output[embed] = point[axis];
    }
  }
  return output;
}

function embedGeometryCurveJacobian(
  program: StructuredHookProgramBuffers,
  slot: number,
  point: ArrayLike<number>,
): Float32Array {
  const profile = program.profile;
  const offsets = curveParameterOffsets(profile);
  const base = slot * structuredCurveParameterStride(profile);
  const intrinsicDim = program.geometryIntrinsicDim![slot];
  if (program.geometryDomainKind![slot] === 2) {
    return embedSphereJacobian(point, intrinsicDim);
  }
  const output = new Float32Array(intrinsicDim * 2 * intrinsicDim);
  for (let axis = 0; axis < intrinsicDim; axis += 1) {
    const embed = axis * 2;
    if (program.geometryCurveParameters![base + offsets.axisPeriodic + axis] === 1) {
      const period = program.geometryCurveParameters![base + offsets.axisPeriod + axis];
      const frequency = 2 * Math.PI / period;
      const angle = frequency * point[axis];
      output[embed * intrinsicDim + axis] = -frequency * Math.sin(angle);
      output[(embed + 1) * intrinsicDim + axis] = frequency * Math.cos(angle);
    } else {
      output[embed * intrinsicDim + axis] = 1;
    }
  }
  return output;
}

function clampGeometryCurvePoint(
  program: StructuredHookProgramBuffers,
  slot: number,
  point: ArrayLike<number>,
): Float32Array {
  const profile = program.profile;
  const offsets = curveParameterOffsets(profile);
  const base = slot * structuredCurveParameterStride(profile);
  const intrinsicDim = program.geometryIntrinsicDim![slot];
  if (program.geometryDomainKind![slot] === 2) {
    return clampSpherePoint(point, intrinsicDim);
  }
  const output = new Float32Array(intrinsicDim);
  for (let axis = 0; axis < intrinsicDim; axis += 1) {
    const lower = program.geometryCurveParameters![base + offsets.bounds + axis * 2];
    const upper = program.geometryCurveParameters![base + offsets.bounds + axis * 2 + 1];
    output[axis] = program.geometryCurveParameters![base + offsets.axisPeriodic + axis] === 1
      ? lower + modulo(
          point[axis] - lower,
          program.geometryCurveParameters![base + offsets.axisPeriod + axis],
        )
      : clamp(point[axis], lower, upper);
  }
  return output;
}

function runCurve(
  program: StructuredHookProgramBuffers,
  slot: number,
  values: Float32Array,
  seed: Float32Array,
  steps: number,
): Float32Array {
  const d = program.hiddenSize;
  const rank = program.curveRank[slot];
  const intrinsicDim = program.curveIntrinsicDim[slot];
  const centered = new Float32Array(d);
  const q = new Float32Array(rank);
  for (let hidden = 0; hidden < d; hidden += 1) {
    centered[hidden] = Math.fround(values[hidden] - program.curveNeutral[slot * d + hidden]);
  }
  for (let axis = 0; axis < rank; axis += 1) {
    for (let hidden = 0; hidden < d; hidden += 1) {
      q[axis] = Math.fround(q[axis] + Math.fround(
        centered[hidden] * program.curveBasis[(slot * program.profile.maxRank + axis) * d + hidden],
      ));
    }
  }
  let foot = clampCurvePoint(program, slot, seed);
  for (let step = 0; step < steps; step += 1) {
    const evaluated = evaluateCurve(program, slot, foot, rank);
    const normal = new Float32Array(intrinsicDim * intrinsicDim);
    const rhs = new Float32Array(intrinsicDim);
    for (let row = 0; row < intrinsicDim; row += 1) {
      for (let axis = 0; axis < rank; axis += 1) {
        rhs[row] += evaluated.jacobian[axis * intrinsicDim + row] *
          (q[axis] - evaluated.value[axis]);
      }
      for (let column = 0; column < intrinsicDim; column += 1) {
        for (let axis = 0; axis < rank; axis += 1) {
          normal[row * intrinsicDim + column] +=
            evaluated.jacobian[axis * intrinsicDim + row] *
            evaluated.jacobian[axis * intrinsicDim + column];
        }
      }
    }
    for (let axis = 0; axis < intrinsicDim; axis += 1) {
      const diagonal = axis * intrinsicDim + axis;
      normal[diagonal] += program.curveDamping[slot] * Math.max(normal[diagonal], 1e-9) + 1e-9;
    }
    const delta = solveLinearSystem(normal, rhs, intrinsicDim);
    foot = clampCurvePoint(
      program,
      slot,
      Float32Array.from(foot, (value, axis) => value + delta[axis]),
    );
  }
  const oldSurface = evaluateCurve(program, slot, foot, rank);
  const newFoot = program.curveDomainKind?.[slot] === 2
    ? sphereCurveTranslation(program, slot, foot)
    : (() => {
        const targetDelta = curveTranslation(program, slot);
        return clampCurvePoint(
          program,
          slot,
          Float32Array.from(foot, (value, axis) =>
            value + program.curveAlong[slot] * targetDelta[axis]),
        );
      })();
  const newSurface = evaluateCurve(program, slot, newFoot, rank);
  const residual = new Float32Array(rank);
  for (let axis = 0; axis < rank; axis += 1) residual[axis] = q[axis] - oldSurface.value[axis];
  const transported = rotateTangentFrames(
    residual,
    oldSurface.jacobian,
    newSurface.jacobian,
    intrinsicDim,
  );
  const norm = Math.sqrt(transported.reduce((sum, value) => sum + value * value, 0));
  let sigma = 0;
  if (hasSigma(program, slot)) sigma = Math.exp(evaluateSigma(program, slot, newFoot));
  const shrink = Math.max(0, 1 - sigma / Math.max(norm, 1e-6));
  const ontoScale = 1 - program.curveOnto[slot] * shrink;
  const nextReduced = new Float32Array(rank);
  for (let axis = 0; axis < rank; axis += 1) {
    nextReduced[axis] = newSurface.value[axis] + ontoScale * transported[axis];
  }
  const hPerp = new Float32Array(centered);
  for (let hidden = 0; hidden < d; hidden += 1) {
    for (let axis = 0; axis < rank; axis += 1) {
      hPerp[hidden] -= q[axis] * program.curveBasis[(slot * program.profile.maxRank + axis) * d + hidden];
    }
  }
  let oldNorm = 0;
  let newNorm = 0;
  for (let hidden = 0; hidden < d; hidden += 1) {
    let next = program.curveNeutral[slot * d + hidden] + hPerp[hidden];
    for (let axis = 0; axis < rank; axis += 1) {
      next += nextReduced[axis] * program.curveBasis[(slot * program.profile.maxRank + axis) * d + hidden];
    }
    oldNorm += values[hidden] ** 2;
    newNorm += next ** 2;
    values[hidden] = Math.fround(next);
  }
  const cap = Math.min(1, 3 * Math.sqrt(oldNorm) / Math.max(Math.sqrt(newNorm), 1e-6));
  for (let hidden = 0; hidden < d; hidden += 1) values[hidden] = Math.fround(values[hidden] * cap);
  return foot;
}

function evaluateCurve(
  program: StructuredHookProgramBuffers,
  slot: number,
  foot: Float32Array,
  rank: number,
): { value: Float32Array; jacobian: Float32Array } {
  const intrinsicDim = program.curveIntrinsicDim[slot];
  const embedDim = program.curveEmbedDim[slot];
  const embedded = embedCurvePoint(program, slot, foot);
  const embedJacobian = embedCurveJacobian(program, slot, foot);
  const normalized = new Float32Array(embedDim);
  for (let embed = 0; embed < embedDim; embed += 1) {
    normalized[embed] =
      (embedded[embed] - program.curveCoordinateOffset[slot * program.profile.maxEmbedDim + embed]) /
      program.curveCoordinateScale[slot * program.profile.maxEmbedDim + embed];
  }
  const value = new Float32Array(rank);
  const normalizedJacobian = new Float32Array(rank * embedDim);
  const jacobian = new Float32Array(rank * intrinsicDim);
  const nodes = program.curveNodeCount[slot];
  for (let axis = 0; axis < rank; axis += 1) {
    value[axis] = program.curvePolynomial[
      (slot * (program.profile.maxEmbedDim + 1)) * program.profile.maxRank + axis
    ];
    for (let embed = 0; embed < embedDim; embed += 1) {
      const coefficient = program.curvePolynomial[
        (slot * (program.profile.maxEmbedDim + 1) + embed + 1) * program.profile.maxRank + axis
      ];
      value[axis] += normalized[embed] * coefficient;
      normalizedJacobian[axis * embedDim + embed] = coefficient;
    }
    for (let node = 0; node < nodes; node += 1) {
      let squaredRadius = 0;
      const differences = new Float32Array(embedDim);
      for (let embed = 0; embed < embedDim; embed += 1) {
        const difference = normalized[embed] - program.curveNodeParameters[
          (slot * program.profile.maxCurveNodes + node) * program.profile.maxEmbedDim + embed
        ];
        differences[embed] = difference;
        squaredRadius += difference * difference;
      }
      const radius = Math.sqrt(squaredRadius);
      const weight = program.curveRbfWeights[
        (slot * program.profile.maxCurveNodes + node) * program.profile.maxRank + axis
      ];
      value[axis] += radius ** 3 * weight;
      for (let embed = 0; embed < embedDim; embed += 1) {
        normalizedJacobian[axis * embedDim + embed] +=
          3 * radius * differences[embed] * weight;
      }
    }
    for (let intrinsic = 0; intrinsic < intrinsicDim; intrinsic += 1) {
      for (let embed = 0; embed < embedDim; embed += 1) {
        jacobian[axis * intrinsicDim + intrinsic] +=
          normalizedJacobian[axis * embedDim + embed] /
          program.curveCoordinateScale[slot * program.profile.maxEmbedDim + embed] *
          embedJacobian[embed * intrinsicDim + intrinsic];
      }
    }
  }
  return { value, jacobian };
}

function evaluateSigma(
  program: StructuredHookProgramBuffers,
  slot: number,
  foot: Float32Array,
): number {
  const embedDim = program.curveEmbedDim[slot];
  const embedded = embedCurvePoint(program, slot, foot);
  const normalized = new Float32Array(embedDim);
  let value = program.curveSigmaPolynomial[slot * (program.profile.maxEmbedDim + 1)];
  for (let embed = 0; embed < embedDim; embed += 1) {
    normalized[embed] =
      (embedded[embed] - program.curveCoordinateOffset[slot * program.profile.maxEmbedDim + embed]) /
      program.curveCoordinateScale[slot * program.profile.maxEmbedDim + embed];
    value += normalized[embed] * program.curveSigmaPolynomial[
      slot * (program.profile.maxEmbedDim + 1) + embed + 1
    ];
  }
  for (let node = 0; node < program.curveNodeCount[slot]; node += 1) {
    let squaredRadius = 0;
    for (let embed = 0; embed < embedDim; embed += 1) {
      const difference = normalized[embed] - program.curveNodeParameters[
        (slot * program.profile.maxCurveNodes + node) * program.profile.maxEmbedDim + embed
      ];
      squaredRadius += difference * difference;
    }
    value += Math.sqrt(squaredRadius) ** 3 *
      program.curveSigmaRbfWeights[slot * program.profile.maxCurveNodes + node];
  }
  return value;
}

function hasSigma(program: StructuredHookProgramBuffers, slot: number): boolean {
  return program.curveSigmaPresent[slot] === 1;
}

function rotateTangentFrames(
  residual: Float32Array,
  oldJacobian: Float32Array,
  newJacobian: Float32Array,
  intrinsicDim: number,
): Float32Array {
  const oldFrame = orthonormalJacobianColumns(oldJacobian, residual.length, intrinsicDim);
  const newFrame = orthonormalJacobianColumns(newJacobian, residual.length, intrinsicDim);
  let transported = new Float32Array(residual);
  for (let tangent = 0; tangent < intrinsicDim; tangent += 1) {
    const oldAxis = oldFrame[tangent];
    const desiredAxis = newFrame[tangent];
    if (vectorNorm(oldAxis) <= 1e-6 || vectorNorm(desiredAxis) <= 1e-6) continue;
    let overlap = dotProduct(oldAxis, desiredAxis);
    const signedTarget = overlap < 0
      ? Float32Array.from(desiredAxis, (value) => -value)
      : desiredAxis;
    overlap = Math.min(1, Math.max(-1, Math.abs(overlap)));
    const perpendicular = Float32Array.from(
      signedTarget,
      (value, axis) => value - overlap * oldAxis[axis],
    );
    const perpendicularNorm = vectorNorm(perpendicular);
    if (perpendicularNorm <= 1e-6) continue;
    const normal = Float32Array.from(perpendicular, (value) => value / perpendicularNorm);
    const sine = Math.sqrt(Math.max(0, 1 - overlap * overlap));
    transported = new Float32Array(
      applyPlaneRotation(transported, oldAxis, normal, overlap, sine),
    );
    for (let future = tangent + 1; future < intrinsicDim; future += 1) {
      oldFrame[future] = applyPlaneRotation(
        oldFrame[future],
        oldAxis,
        normal,
        overlap,
        sine,
      );
    }
  }
  return transported;
}

function orthonormalJacobianColumns(
  jacobian: Float32Array,
  rank: number,
  intrinsicDim: number,
): Float32Array[] {
  const output: Float32Array[] = [];
  for (let column = 0; column < intrinsicDim; column += 1) {
    const axis = new Float32Array(rank);
    for (let row = 0; row < rank; row += 1) axis[row] = jacobian[row * intrinsicDim + column];
    for (const prior of output) {
      const overlap = dotProduct(axis, prior);
      for (let row = 0; row < rank; row += 1) axis[row] -= overlap * prior[row];
    }
    const norm = vectorNorm(axis);
    output.push(norm > 1e-6
      ? Float32Array.from(axis, (value) => value / norm)
      : new Float32Array(rank));
  }
  return output;
}

function applyPlaneRotation(
  values: Float32Array,
  axis: Float32Array,
  normal: Float32Array,
  cosine: number,
  sine: number,
): Float32Array {
  const alpha = dotProduct(axis, values);
  const beta = dotProduct(normal, values);
  const axisDelta = (cosine - 1) * alpha - sine * beta;
  const normalDelta = sine * alpha + (cosine - 1) * beta;
  return Float32Array.from(
    values,
    (value, index) => value + axisDelta * axis[index] + normalDelta * normal[index],
  );
}

function embedCurvePoint(
  program: StructuredHookProgramBuffers,
  slot: number,
  point: ArrayLike<number>,
): Float32Array {
  if (program.curveDomainKind?.[slot] === 2) {
    return embedSpherePoint(point, program.curveIntrinsicDim[slot]);
  }
  const output = new Float32Array(program.curveEmbedDim[slot]);
  for (let axis = 0; axis < program.curveIntrinsicDim[slot]; axis += 1) {
    const axisSlot = slot * program.profile.maxIntrinsicDim + axis;
    const embed = axis * 2;
    if (program.curveAxisPeriodic[axisSlot] === 1) {
      const angle = 2 * Math.PI * point[axis] / program.curveAxisPeriod[axisSlot];
      output[embed] = Math.cos(angle);
      output[embed + 1] = Math.sin(angle);
    } else {
      output[embed] = point[axis];
    }
  }
  return output;
}

function embedCurveJacobian(
  program: StructuredHookProgramBuffers,
  slot: number,
  point: ArrayLike<number>,
): Float32Array {
  const intrinsicDim = program.curveIntrinsicDim[slot];
  if (program.curveDomainKind?.[slot] === 2) {
    return embedSphereJacobian(point, intrinsicDim);
  }
  const output = new Float32Array(program.curveEmbedDim[slot] * intrinsicDim);
  for (let axis = 0; axis < intrinsicDim; axis += 1) {
    const axisSlot = slot * program.profile.maxIntrinsicDim + axis;
    const embed = axis * 2;
    if (program.curveAxisPeriodic[axisSlot] === 1) {
      const frequency = 2 * Math.PI / program.curveAxisPeriod[axisSlot];
      const angle = frequency * point[axis];
      output[embed * intrinsicDim + axis] = -frequency * Math.sin(angle);
      output[(embed + 1) * intrinsicDim + axis] = frequency * Math.cos(angle);
    } else {
      output[embed * intrinsicDim + axis] = 1;
    }
  }
  return output;
}

function embedSpherePoint(
  point: ArrayLike<number>,
  intrinsicDim: number,
): Float32Array {
  const output = new Float32Array(intrinsicDim + 1);
  let running = 1;
  for (let axis = 0; axis < intrinsicDim; axis += 1) {
    output[axis] = running * Math.cos(point[axis]);
    running *= Math.sin(point[axis]);
  }
  output[intrinsicDim] = running;
  return output;
}

function embedSphereJacobian(
  point: ArrayLike<number>,
  intrinsicDim: number,
): Float32Array {
  const output = new Float32Array((intrinsicDim + 1) * intrinsicDim);
  for (let component = 0; component <= intrinsicDim; component += 1) {
    for (let derivative = 0; derivative < intrinsicDim; derivative += 1) {
      if (component < intrinsicDim && derivative > component) continue;
      let value = 1;
      if (component < intrinsicDim && derivative === component) {
        for (let axis = 0; axis < component; axis += 1) {
          value *= Math.sin(point[axis]);
        }
        value *= -Math.sin(point[component]);
      } else {
        const limit = component < intrinsicDim ? component : intrinsicDim;
        for (let axis = 0; axis < limit; axis += 1) {
          value *= axis === derivative
            ? Math.cos(point[axis])
            : Math.sin(point[axis]);
        }
        if (component < intrinsicDim) value *= Math.cos(point[component]);
      }
      output[component * intrinsicDim + derivative] = value;
    }
  }
  return output;
}

function unembedSpherePoint(
  embedded: ArrayLike<number>,
  intrinsicDim: number,
): Float32Array {
  const output = new Float32Array(intrinsicDim);
  for (let axis = 0; axis < intrinsicDim; axis += 1) {
    if (axis < intrinsicDim - 1) {
      let tailSquared = 0;
      for (let component = axis + 1; component <= intrinsicDim; component += 1) {
        tailSquared += embedded[component] ** 2;
      }
      output[axis] = Math.atan2(Math.sqrt(tailSquared), embedded[axis]);
    } else {
      output[axis] = modulo(
        Math.atan2(embedded[intrinsicDim], embedded[intrinsicDim - 1]),
        2 * Math.PI,
      );
    }
  }
  return output;
}

function sphereCurveTranslation(
  program: StructuredHookProgramBuffers,
  slot: number,
  point: ArrayLike<number>,
): Float32Array {
  const intrinsicDim = program.curveIntrinsicDim[slot];
  const pointEmbedded = embedSpherePoint(point, intrinsicDim);
  const origin = program.curveOrigin.subarray(
    slot * program.profile.maxIntrinsicDim,
    slot * program.profile.maxIntrinsicDim + intrinsicDim,
  );
  const target = program.curveTarget.subarray(
    slot * program.profile.maxIntrinsicDim,
    slot * program.profile.maxIntrinsicDim + intrinsicDim,
  );
  const originEmbedded = embedSpherePoint(origin, intrinsicDim);
  const targetEmbedded = embedSpherePoint(target, intrinsicDim);
  const originTargetDot = clamp(
    dotProduct(originEmbedded, targetEmbedded),
    -1,
    1,
  );
  const omega = Math.acos(originTargetDot);
  const tangent = Float32Array.from(
    targetEmbedded,
    (value, index) => value - originTargetDot * originEmbedded[index],
  );
  const tangentNorm = vectorNorm(tangent);
  if (tangentNorm > 1e-9) {
    for (let index = 0; index < tangent.length; index += 1) {
      tangent[index] *= omega / tangentNorm;
    }
  } else {
    tangent.fill(0);
  }
  const originPointDot = dotProduct(originEmbedded, pointEmbedded);
  const coefficient = dotProduct(tangent, pointEmbedded) /
    Math.max(1 + originPointDot, 1e-9);
  const transported = Float32Array.from(
    tangent,
    (value, index) => value - coefficient *
      (originEmbedded[index] + pointEmbedded[index]),
  );
  const scaled = Float32Array.from(
    transported,
    (value) => program.curveAlong[slot] * value,
  );
  const theta = vectorNorm(scaled);
  if (theta < 1e-9) return clampCurvePoint(program, slot, point);
  const nextEmbedded = Float32Array.from(
    pointEmbedded,
    (value, index) => Math.cos(theta) * value +
      Math.sin(theta) * scaled[index] / theta,
  );
  return clampCurvePoint(
    program,
    slot,
    unembedSpherePoint(nextEmbedded, intrinsicDim),
  );
}

function curveTranslation(
  program: StructuredHookProgramBuffers,
  slot: number,
): Float32Array {
  const output = new Float32Array(program.curveIntrinsicDim[slot]);
  for (let axis = 0; axis < output.length; axis += 1) {
    const axisSlot = slot * program.profile.maxIntrinsicDim + axis;
    let delta = program.curveTarget[axisSlot] - program.curveOrigin[axisSlot];
    if (program.curveAxisPeriodic[axisSlot] === 1) {
      const period = program.curveAxisPeriod[axisSlot];
      delta = modulo(delta + period / 2, period) - period / 2;
    }
    output[axis] = delta;
  }
  return output;
}

function clampCurvePoint(
  program: StructuredHookProgramBuffers,
  slot: number,
  point: ArrayLike<number>,
): Float32Array {
  if (program.curveDomainKind?.[slot] === 2) {
    return clampSpherePoint(point, program.curveIntrinsicDim[slot]);
  }
  const output = new Float32Array(program.curveIntrinsicDim[slot]);
  for (let axis = 0; axis < output.length; axis += 1) {
    const axisSlot = slot * program.profile.maxIntrinsicDim + axis;
    const lower = program.curveBounds[axisSlot * 2];
    const upper = program.curveBounds[axisSlot * 2 + 1];
    output[axis] = program.curveAxisPeriodic[axisSlot] === 1
      ? lower + modulo(point[axis] - lower, program.curveAxisPeriod[axisSlot])
      : clamp(point[axis], lower, upper);
  }
  return output;
}

function clampSpherePoint(
  point: ArrayLike<number>,
  intrinsicDim: number,
): Float32Array {
  const output = new Float32Array(intrinsicDim);
  for (let axis = 0; axis < intrinsicDim; axis += 1) {
    output[axis] = axis === intrinsicDim - 1
      ? modulo(point[axis], 2 * Math.PI)
      : clamp(point[axis], 0, Math.PI);
  }
  return output;
}

function solveLinearSystem(
  matrix: Float32Array,
  right: Float32Array,
  size: number,
): Float32Array {
  const augmented = Array.from({ length: size }, (_, row) => [
    ...Array.from(matrix.slice(row * size, (row + 1) * size)),
    right[row],
  ]);
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    if (Math.abs(divisor) <= 1e-12) continue;
    for (let value = column; value <= size; value += 1) augmented[column][value] /= divisor;
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let value = column; value <= size; value += 1) {
        augmented[row][value] -= factor * augmented[column][value];
      }
    }
  }
  return Float32Array.from(augmented, (row) => Number.isFinite(row[size]) ? row[size] : 0);
}

function modulo(value: number, modulus: number): number {
  return ((value % modulus) + modulus) % modulus;
}

function requireLayer(program: StructuredHookProgramBuffers, layer: number): void {
  if (!Number.isSafeInteger(layer) || layer < 0 || layer >= program.layerCount) {
    throw new RangeError(`Hook layer ${layer} is outside the compiled program`);
  }
}

function requirePositiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) throw new TypeError(`${label} must be a positive integer`);
}

function validInstrumentBindingId(value: unknown): value is string {
  return typeof value === "string" && /^[a-f0-9]{64}$/u.test(value);
}

function copyFiniteVector(source: ArrayLike<number>, size: number, label: string): Float32Array {
  const output = new Float32Array(size);
  copyVector(source, output, 0, size, label);
  return output;
}

function dotProduct(left: ArrayLike<number>, right: ArrayLike<number>): number {
  if (left.length !== right.length) throw new TypeError("Vector dimensions do not match");
  let value = 0;
  for (let index = 0; index < left.length; index += 1) value += left[index] * right[index];
  return value;
}

function dotProductSlices(
  values: Float32Array,
  leftOffset: number,
  rightOffset: number,
  size: number,
): number {
  let result = 0;
  for (let index = 0; index < size; index += 1) {
    result += values[leftOffset + index] * values[rightOffset + index];
  }
  return result;
}

function vectorNorm(values: ArrayLike<number>): number {
  return Math.sqrt(dotProduct(values, values));
}

function copyVector(
  source: ArrayLike<number>,
  target: Float32Array,
  offset: number,
  size: number,
  label: string,
): void {
  if (source.length !== size) throw new TypeError(`${label} has ${source.length} values; expected ${size}`);
  for (let index = 0; index < size; index += 1) target[offset + index] = finiteFloat(source[index], label);
}

function finiteFloat(value: number, label: string, allowInfinity = false): number {
  if (allowInfinity && value === Number.POSITIVE_INFINITY) return 3.4028234663852886e38;
  if (allowInfinity && value === Number.NEGATIVE_INFINITY) return -3.4028234663852886e38;
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
): void {
  let squaredNorm = 0;
  for (let index = 0; index < size; index += 1) squaredNorm += values[offset + index] ** 2;
  if (Math.abs(squaredNorm - 1) > 1e-4) throw new TypeError(`${label} must be unit length`);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
