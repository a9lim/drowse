import type { ActivationSpoolIdentity } from "./activationSpool";

export const FITTING_WORKER_PROTOCOL_VERSION = 1 as const;
export const FITTING_WORKER_MAX_MATRIX_ELEMENTS = 16 * 1024 * 1024;

export type FittingMatrixSource =
  | {
      kind: "inline";
      values: Float64Array;
      rows: number;
      columns: number;
    }
  | {
      kind: "activation_spool";
      identity: ActivationSpoolIdentity;
      layer: number;
    };

export type FittingWorkerJob =
  | { operation: "center"; source: FittingMatrixSource }
  | {
      operation: "group_means";
      source: FittingMatrixSource;
      offsets: Uint32Array;
    }
  | {
      operation: "whiten";
      source: FittingMatrixSource;
      ridgeScale: number;
      transform?: FittingMatrixSource;
    }
  | {
      operation: "pca";
      source: FittingMatrixSource;
      maxComponents: number;
      varianceThreshold: number;
    }
  | {
      operation: "affine_fisher";
      source: FittingMatrixSource;
      whitener: SerializedMahalanobisWhitener;
      maxComponents: number;
      orientTo: number;
    }
  | {
      operation: "reduced_covariances";
      source: Extract<FittingMatrixSource, { kind: "activation_spool" }>;
      offsets: Uint32Array;
      mean: Float64Array;
      basis: Float64Array;
      components: number;
    }
  | {
      operation: "position";
      mean: Float64Array;
      basis: Float64Array;
      components: number;
      coordinates: Float64Array;
    }
  | {
      operation: "project";
      source: FittingMatrixSource;
      mean: Float64Array;
      basis: Float64Array;
      components: number;
    }
  | {
      operation: "ablate";
      source: FittingMatrixSource;
      mean: Float64Array;
      basis: Float64Array;
      components: number;
      coefficients: Float64Array;
    }
  | { operation: "correlations"; source: FittingMatrixSource }
  | { operation: "pairwise"; source: FittingMatrixSource }
  | {
      operation: "spectral";
      gram: Float64Array;
      nodeCount: number;
      maxDimensions: number;
      minDimensions?: number;
      kNn?: number;
      bandwidth?: number;
    }
  | {
      operation: "topology";
      consensusGram: Float64Array;
      nodeCount: number;
      targets: Float64Array;
      targetOffsets: Uint32Array;
      maxDimensions: number;
      varianceThreshold: number;
      requestedFitMode: "pca" | "spectral" | "auto";
      minDimensions?: number;
      kNn?: number;
      bandwidth?: number;
      persistenceFraction: number;
      smoothing?: number;
    }
  | {
      operation: "rbf";
      nodes: Float64Array;
      nodeCount: number;
      inputDimensions: number;
      values: Float64Array;
      outputDimensions: number;
      smoothing?: number;
      plan?: SerializedRbfFitPlan;
      queries?: Float64Array;
      queryRows?: number;
    }
  | {
      operation: "sigma_field";
      surface: SerializedRbfModel;
      covariances: Float64Array;
      coordinates: Float64Array;
      embeddedCoordinates: Float64Array;
      intrinsicDimensions: number;
      periodicDimensions: number;
      smoothing?: number;
      plan?: SerializedRbfFitPlan;
      floorFraction: number;
    }
  | {
      operation: "rbf_origin";
      surface: SerializedRbfModel;
      coordinates: Float64Array;
      embeddedCoordinates: Float64Array;
      intrinsicDimensions: number;
      periodicDimensions: number;
      maxIterations: number;
      restartCount: number;
      damping: number;
    }
  | {
      operation: "template_scores";
      sumLogProbabilities: Float64Array;
      tokenCounts: Uint32Array;
    };

export interface FittingWorkerRequest {
  protocolVersion: typeof FITTING_WORKER_PROTOCOL_VERSION;
  requestId: string;
  kind: "run";
  job: FittingWorkerJob;
}

export interface FittingWorkerFailure {
  code: string;
  message: string;
  recoverable: boolean;
}

export type FittingWorkerStage = "loading" | "reading" | "computing";

export type FittingWorkerResponse =
  | {
      protocolVersion: typeof FITTING_WORKER_PROTOCOL_VERSION;
      requestId: string;
      kind: "progress";
      stage: FittingWorkerStage;
    }
  | {
      protocolVersion: typeof FITTING_WORKER_PROTOCOL_VERSION;
      requestId: string;
      kind: "result";
      result: FittingWorkerResult;
    }
  | {
      protocolVersion: typeof FITTING_WORKER_PROTOCOL_VERSION;
      requestId: string;
      kind: "error";
      error: FittingWorkerFailure;
    };

export type FittingWorkerResult =
  | CenteringResult
  | WhiteningResult
  | AffineFisherResult
  | ReducedCovarianceResult
  | PcaResult
  | MatrixResult
  | SpectralResult
  | TopologyResult
  | RbfResult
  | SigmaFieldResult
  | RbfOriginResult
  | TemplateScoreResult;

export interface CenteringResult {
  operation: "center";
  rows: number;
  columns: number;
  mean: Float64Array;
  centered: Float64Array;
}

export interface WhiteningResult {
  operation: "whiten";
  columns: number;
  rank: number;
  ridge: number;
  mean: Float64Array;
  basis: Float64Array;
  eigenvalues: Float64Array;
  inverseScales: Float64Array;
  transformed: Float64Array | null;
  transformedRows: number | null;
}

export interface SerializedMahalanobisWhitener {
  columns: number;
  rank: number;
  ridge: number;
  mean: Float64Array;
  basis: Float64Array;
  eigenvalues: Float64Array;
  inverseScales: Float64Array;
}

export interface AffineFisherResult {
  operation: "affine_fisher";
  nodeCount: number;
  columns: number;
  components: number;
  centroidMean: Float64Array;
  mean: Float64Array;
  basis: Float64Array;
  nodeCoordinates: Float64Array;
  muCoordinates: Float64Array;
  whitenedGram: Float64Array;
  neutralCrossGram: Float64Array;
  explainedVariance: number;
  mahalanobisShare: number;
}

export interface ReducedCovarianceResult {
  operation: "reduced_covariances";
  nodeCount: number;
  components: number;
  covariances: Float64Array;
}

export interface PcaResult {
  operation: "pca";
  rows: number;
  columns: number;
  components: number;
  mean: Float64Array;
  basis: Float64Array;
  eigenvalues: Float64Array;
  explainedVariance: Float64Array;
  cumulativeVariance: Float64Array;
  scores: Float64Array;
}

export interface MatrixResult {
  operation: "group_means" | "position" | "project" | "ablate" | "correlations" | "pairwise";
  rows: number;
  columns: number;
  values: Float64Array;
}

export interface SpectralResult {
  operation: "spectral";
  nodeCount: number;
  dimensions: number;
  heuristicDimensions: number;
  pinned: boolean;
  kNn: number;
  bandwidth: number;
  gapMagnitude: number;
  eigenvalues: Float64Array;
  coordinates: Float64Array;
}

export interface TopologyCandidateResult {
  name: string;
  fitMode: string;
  dimensions: number;
  score: number | null;
  viable: boolean;
  reason: string;
}

export interface TopologyResult {
  operation: "topology";
  winnerName: string;
  fitMode: string;
  intrinsicDimensions: number;
  periodicDimensions: number;
  persistentLoops: number;
  usedFaintCycle: boolean;
  coordinates: Float64Array;
  embeddedCoordinates: Float64Array;
  candidates: TopologyCandidateResult[];
  diagnostics: TopologyDiagnostics;
  winnerPlan: SerializedRbfFitPlan | null;
}

export type TopologyDiagnostics =
  | {
      kind: "pca";
      perComponentVariance: Float64Array;
      cumulativeVariance: Float64Array;
      pickedDimensions: number;
      threshold: number;
    }
  | {
      kind: "spectral";
      eigenvalues: Float64Array;
      pickedDimensions: number;
      gapMagnitude: number;
      bandwidth: number;
      kNn: number;
      componentCount: 1;
      heuristicDimensions: number;
      minDimensions: number | null;
      pinned: boolean;
    };

export interface SerializedRbfFitPlan {
  inputDimensions: number;
  nodeCount: number;
  nodes: Float64Array;
  coordinateOffset: Float64Array;
  coordinateScale: Float64Array;
  kernel: Float64Array;
  kernelScale: number;
  lambdas: Float64Array;
  spectralBasis: Float64Array;
  residualRatios: Float64Array;
  residualTraces: Float64Array;
}

export interface SerializedRbfModel {
  inputDimensions: number;
  outputDimensions: number;
  nodeCount: number;
  lambda: number;
  effectiveDegreesOfFreedom: number;
  gcv: number;
  nodes: Float64Array;
  coordinateOffset: Float64Array;
  coordinateScale: Float64Array;
  weights: Float64Array;
  polynomial: Float64Array;
}

export interface RbfResult {
  operation: "rbf";
  model: SerializedRbfModel;
  evaluated: Float64Array | null;
  queryRows: number | null;
}

export interface SigmaFieldResult {
  operation: "sigma_field";
  model: SerializedRbfModel;
  sigmaMean: number;
  sigmaMin: number;
  sigmaMax: number;
}

export interface RbfOriginResult {
  operation: "rbf_origin";
  coordinates: Float64Array;
  distance: number;
}

export interface TemplateScoreResult {
  operation: "template_scores";
  sumProbabilities: Float64Array;
  meanLogProbabilities: Float64Array;
  meanProbabilities: Float64Array;
}

export function isFittingWorkerResponse(value: unknown): value is FittingWorkerResponse {
  if (!isRecord(value)) return false;
  if (value.protocolVersion !== FITTING_WORKER_PROTOCOL_VERSION) return false;
  if (typeof value.requestId !== "string" || value.requestId.length === 0) return false;
  if (value.kind === "progress") {
    return value.stage === "loading" || value.stage === "reading" || value.stage === "computing";
  }
  if (value.kind === "result") return isFittingWorkerResult(value.result);
  return value.kind === "error" && isFailure(value.error);
}

function isFittingWorkerResult(value: unknown): value is FittingWorkerResult {
  if (!isRecord(value) || typeof value.operation !== "string") return false;
  switch (value.operation) {
    case "center":
      return isPositiveInteger(value.rows) && isPositiveInteger(value.columns) &&
        isFloat64(value.mean) && isFloat64(value.centered) &&
        value.mean.length === value.columns &&
        value.centered.length === value.rows * value.columns;
    case "whiten":
      return isPositiveInteger(value.columns) && isNonNegativeInteger(value.rank) &&
        isFiniteNumber(value.ridge) && isFloat64(value.mean) &&
        isFloat64(value.basis) && isFloat64(value.eigenvalues) &&
        isFloat64(value.inverseScales) && value.mean.length === value.columns &&
        value.basis.length === value.columns * value.rank &&
        value.eigenvalues.length === value.rank && value.inverseScales.length === value.rank &&
        ((value.transformed === null && value.transformedRows === null) ||
          (isFloat64(value.transformed) && isPositiveInteger(value.transformedRows) &&
            value.transformed.length === value.transformedRows * value.columns));
    case "pca":
      return isPositiveInteger(value.rows) && isPositiveInteger(value.columns) &&
        isPositiveInteger(value.components) && isFloat64(value.mean) &&
        isFloat64(value.basis) && isFloat64(value.eigenvalues) &&
        isFloat64(value.explainedVariance) && isFloat64(value.cumulativeVariance) &&
        isFloat64(value.scores) && value.mean.length === value.columns &&
        value.basis.length === value.columns * value.components &&
        value.eigenvalues.length === value.components &&
        value.explainedVariance.length === value.components &&
        value.cumulativeVariance.length === value.components &&
        value.scores.length === value.rows * value.components;
    case "affine_fisher":
      return isPositiveInteger(value.nodeCount) && isPositiveInteger(value.columns) &&
        isPositiveInteger(value.components) && value.components < value.nodeCount &&
        isFloat64(value.centroidMean) && value.centroidMean.length === value.columns &&
        isFloat64(value.mean) && value.mean.length === value.columns &&
        isFloat64(value.basis) && value.basis.length === value.components * value.columns &&
        isFloat64(value.nodeCoordinates) &&
        value.nodeCoordinates.length === value.nodeCount * value.components &&
        isFloat64(value.muCoordinates) &&
        value.muCoordinates.length === value.nodeCount * value.components &&
        isFloat64(value.whitenedGram) &&
        value.whitenedGram.length === value.nodeCount * value.nodeCount &&
        isFloat64(value.neutralCrossGram) && value.neutralCrossGram.length === value.nodeCount &&
        isFiniteNumber(value.explainedVariance) &&
        value.explainedVariance >= 0 && value.explainedVariance <= 1 &&
        isFiniteNumber(value.mahalanobisShare) && value.mahalanobisShare >= 0;
    case "reduced_covariances":
      return isPositiveInteger(value.nodeCount) && isPositiveInteger(value.components) &&
        isFloat64(value.covariances) &&
        value.covariances.length === value.nodeCount * value.components * value.components;
    case "position":
    case "group_means":
    case "project":
    case "ablate":
    case "correlations":
    case "pairwise":
      return isPositiveInteger(value.rows) && isPositiveInteger(value.columns) &&
        isFloat64(value.values) && value.values.length === value.rows * value.columns;
    case "spectral":
      return isPositiveInteger(value.nodeCount) && isPositiveInteger(value.dimensions) &&
        isNonNegativeInteger(value.heuristicDimensions) && typeof value.pinned === "boolean" &&
        isPositiveInteger(value.kNn) && isFiniteNumber(value.bandwidth) &&
        isFiniteNumber(value.gapMagnitude) && isFloat64(value.eigenvalues) &&
        isFloat64(value.coordinates) && value.coordinates.length === value.nodeCount * value.dimensions;
    case "topology":
      return typeof value.winnerName === "string" && typeof value.fitMode === "string" &&
        isPositiveInteger(value.intrinsicDimensions) &&
        isNonNegativeInteger(value.periodicDimensions) &&
        isNonNegativeInteger(value.persistentLoops) && typeof value.usedFaintCycle === "boolean" &&
        isFloat64(value.coordinates) && isFloat64(value.embeddedCoordinates) &&
        Array.isArray(value.candidates) && value.candidates.every(isTopologyCandidate) &&
        isTopologyDiagnostics(value.diagnostics) &&
        (value.winnerPlan === null || isSerializedRbfPlan(value.winnerPlan));
    case "rbf":
      return isSerializedRbf(value.model) &&
        ((value.evaluated === null && value.queryRows === null) ||
          (isFloat64(value.evaluated) && isPositiveInteger(value.queryRows) &&
            value.evaluated.length === value.queryRows * value.model.outputDimensions));
    case "sigma_field":
      return isSerializedRbf(value.model) && value.model.outputDimensions === 1 &&
        isFiniteNumber(value.sigmaMean) && value.sigmaMean > 0 &&
        isFiniteNumber(value.sigmaMin) && value.sigmaMin > 0 &&
        isFiniteNumber(value.sigmaMax) && value.sigmaMax >= value.sigmaMin;
    case "rbf_origin":
      return isFloat64(value.coordinates) && value.coordinates.length > 0 &&
        isFiniteNumber(value.distance) && value.distance >= 0;
    case "template_scores":
      return isFloat64(value.sumProbabilities) && isFloat64(value.meanLogProbabilities) &&
        isFloat64(value.meanProbabilities) &&
        value.sumProbabilities.length === value.meanLogProbabilities.length &&
        value.sumProbabilities.length === value.meanProbabilities.length;
    default:
      return false;
  }
}

function isTopologyCandidate(value: unknown): value is TopologyCandidateResult {
  return isRecord(value) && typeof value.name === "string" &&
    typeof value.fitMode === "string" && isPositiveInteger(value.dimensions) &&
    (value.score === null || isFiniteNumber(value.score)) &&
    typeof value.viable === "boolean" &&
    typeof value.reason === "string";
}

function isTopologyDiagnostics(value: unknown): value is TopologyDiagnostics {
  if (!isRecord(value) || !isPositiveInteger(value.pickedDimensions)) return false;
  if (value.kind === "pca") {
    return isFloat64(value.perComponentVariance) && value.perComponentVariance.length > 0 &&
      isFloat64(value.cumulativeVariance) &&
      value.cumulativeVariance.length === value.perComponentVariance.length &&
      value.pickedDimensions <= value.perComponentVariance.length &&
      isFiniteNumber(value.threshold) && value.threshold > 0 && value.threshold <= 1;
  }
  return value.kind === "spectral" && isFloat64(value.eigenvalues) &&
    value.eigenvalues.length > 0 && isFiniteNumber(value.gapMagnitude) &&
    isFiniteNumber(value.bandwidth) && value.bandwidth > 0 &&
    isPositiveInteger(value.kNn) && value.componentCount === 1 &&
    isPositiveInteger(value.heuristicDimensions) &&
    (value.minDimensions === null || isPositiveInteger(value.minDimensions)) &&
    typeof value.pinned === "boolean";
}

function isSerializedRbfPlan(value: unknown): value is SerializedRbfFitPlan {
  if (!isRecord(value) || !isPositiveInteger(value.inputDimensions) ||
    !isPositiveInteger(value.nodeCount) || value.nodeCount < value.inputDimensions + 1 ||
    !isFloat64(value.nodes) || !isFloat64(value.coordinateOffset) ||
    !isFloat64(value.coordinateScale) || !isFloat64(value.kernel) ||
    !isFiniteNumber(value.kernelScale) || value.kernelScale <= 0 ||
    !isFloat64(value.lambdas) || value.lambdas.length !== 40 ||
    !isFloat64(value.spectralBasis) || !isFloat64(value.residualRatios) ||
    !isFloat64(value.residualTraces) || value.residualTraces.length !== 40) return false;
  const nullDimensions = value.nodeCount - value.inputDimensions - 1;
  return value.nodes.length === value.nodeCount * value.inputDimensions &&
    value.coordinateOffset.length === value.inputDimensions &&
    value.coordinateScale.length === value.inputDimensions &&
    value.kernel.length === value.nodeCount * value.nodeCount &&
    value.spectralBasis.length === value.nodeCount * nullDimensions &&
    value.residualRatios.length === 40 * nullDimensions;
}

function isSerializedRbf(value: unknown): value is SerializedRbfModel {
  if (!isRecord(value) || !isPositiveInteger(value.inputDimensions) ||
    !isPositiveInteger(value.outputDimensions) || !isPositiveInteger(value.nodeCount) ||
    !isFiniteNumber(value.lambda) || !isFiniteNumber(value.effectiveDegreesOfFreedom) ||
    !isFiniteNumber(value.gcv) || !isFloat64(value.nodes) ||
    !isFloat64(value.coordinateOffset) || !isFloat64(value.coordinateScale) ||
    !isFloat64(value.weights) || !isFloat64(value.polynomial)) return false;
  return value.nodes.length === value.nodeCount * value.inputDimensions &&
    value.coordinateOffset.length === value.inputDimensions &&
    value.coordinateScale.length === value.inputDimensions &&
    value.weights.length === value.nodeCount * value.outputDimensions &&
    value.polynomial.length === (value.inputDimensions + 1) * value.outputDimensions;
}

function isFloat64(value: unknown): value is Float64Array {
  return value instanceof Float64Array;
}

function isPositiveInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function isNonNegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isFailure(value: unknown): value is FittingWorkerFailure {
  return isRecord(value) && typeof value.code === "string" &&
    typeof value.message === "string" && typeof value.recoverable === "boolean";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
