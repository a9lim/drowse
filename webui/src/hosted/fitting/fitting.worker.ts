import {
  ActivationSpoolError,
  withBrowserActivationSpool,
  type ActivationSpoolIdentity,
} from "./activationSpool";
import {
  FITTING_WORKER_MAX_MATRIX_ELEMENTS,
  FITTING_WORKER_PROTOCOL_VERSION,
  type CenteringResult,
  type AffineFisherResult,
  type FittingMatrixSource,
  type FittingWorkerJob,
  type FittingWorkerRequest,
  type FittingWorkerResponse,
  type MatrixResult,
  type PcaResult,
  type ReducedCovarianceResult,
  type RbfResult,
  type RbfOriginResult,
  type SerializedRbfFitPlan,
  type SerializedRbfModel,
  type SigmaFieldResult,
  type SpectralResult,
  type TemplateScoreResult,
  type TopologyResult,
  type WhiteningResult,
} from "./workerContracts";

interface WorkerScope {
  location: Location;
  addEventListener(type: "message", listener: (event: MessageEvent<unknown>) => void): void;
  postMessage(message: FittingWorkerResponse, transfer?: Transferable[]): void;
}

interface DisposableWasmValue {
  free(): void;
}

interface CenteringWasm extends DisposableWasmValue {
  readonly rows: number;
  readonly columns: number;
  mean(): Float64Array;
  centered(): Float64Array;
}

interface WhitenerWasm extends DisposableWasmValue {
  readonly columns: number;
  readonly rank: number;
  readonly ridge: number;
  mean(): Float64Array;
  basis(): Float64Array;
  eigenvalues(): Float64Array;
  inverseScales(): Float64Array;
  transform(values: Float64Array, rows: number): Float64Array;
}

interface PcaWasm extends DisposableWasmValue {
  readonly rows: number;
  readonly columns: number;
  readonly components: number;
  mean(): Float64Array;
  basis(): Float64Array;
  eigenvalues(): Float64Array;
  explainedVariance(): Float64Array;
  cumulativeVariance(): Float64Array;
  scores(): Float64Array;
}

interface AffineFisherWasm extends DisposableWasmValue {
  readonly nodeCount: number;
  readonly columns: number;
  readonly components: number;
  readonly explainedVariance: number;
  readonly mahalanobisShare: number;
  centroidMean(): Float64Array;
  mean(): Float64Array;
  basis(): Float64Array;
  nodeCoordinates(): Float64Array;
  muCoordinates(): Float64Array;
  whitenedGram(): Float64Array;
  neutralCrossGram(): Float64Array;
}

interface GroupedReducedCovarianceAccumulatorWasm extends DisposableWasmValue {
  readonly nodeCount: number;
  readonly components: number;
  append(values: Float32Array, startRow: number, rows: number): void;
  finalize(): Float64Array;
}

interface SpectralWasm extends DisposableWasmValue {
  readonly nodeCount: number;
  readonly dimensions: number;
  readonly heuristicDimensions: number;
  readonly pinned: boolean;
  readonly kNn: number;
  readonly bandwidth: number;
  readonly gapMagnitude: number;
  eigenvalues(): Float64Array;
  coordinates(): Float64Array;
}

interface RbfWasm extends DisposableWasmValue {
  readonly inputDimensions: number;
  readonly outputDimensions: number;
  readonly nodeCount: number;
  readonly lambda: number;
  readonly effectiveDegreesOfFreedom: number;
  readonly gcv: number;
  nodes(): Float64Array;
  coordinateOffset(): Float64Array;
  coordinateScale(): Float64Array;
  weights(): Float64Array;
  polynomial(): Float64Array;
  evaluate(queries: Float64Array, rows: number): Float64Array;
}

interface RbfFitPlanWasm extends DisposableWasmValue {
  readonly inputDimensions: number;
  readonly nodeCount: number;
  readonly kernelScale: number;
  nodes(): Float64Array;
  coordinateOffset(): Float64Array;
  coordinateScale(): Float64Array;
  kernel(): Float64Array;
  lambdas(): Float64Array;
  spectralBasis(): Float64Array;
  residualRatios(): Float64Array;
  residualTraces(): Float64Array;
  fitAutoSmoothed(values: Float64Array, outputDimensions: number): RbfWasm;
  fitSmoothed(values: Float64Array, outputDimensions: number, smoothing: number): RbfWasm;
}

interface SigmaFieldWasm extends DisposableWasmValue {
  readonly sigmaMean: number;
  readonly sigmaMin: number;
  readonly sigmaMax: number;
  model(): RbfWasm;
}

interface RbfOriginWasm extends DisposableWasmValue {
  readonly distance: number;
  coordinates(): Float64Array;
}

interface TopologyWasm extends DisposableWasmValue {
  readonly winnerName: string;
  readonly fitMode: string;
  readonly intrinsicDimensions: number;
  readonly periodicDimensions: number;
  readonly persistentLoops: number;
  readonly usedFaintCycle: boolean;
  readonly candidateCount: number;
  readonly hasWinnerPlan: boolean;
  readonly diagnosticsKind: string;
  readonly diagnosticsPickedDimensions: number;
  readonly diagnosticsThreshold: number | undefined;
  readonly diagnosticsGapMagnitude: number | undefined;
  readonly diagnosticsBandwidth: number | undefined;
  readonly diagnosticsKNn: number | undefined;
  readonly diagnosticsHeuristicDimensions: number | undefined;
  readonly diagnosticsMinDimensions: number | undefined;
  readonly diagnosticsPinned: boolean | undefined;
  coordinates(): Float64Array;
  embeddedCoordinates(): Float64Array;
  winnerPlan(): RbfFitPlanWasm | undefined;
  diagnosticsPerComponentVariance(): Float64Array;
  diagnosticsCumulativeVariance(): Float64Array;
  diagnosticsEigenvalues(): Float64Array;
  candidateName(index: number): string | undefined;
  candidateFitMode(index: number): string | undefined;
  candidateDimensions(index: number): number | undefined;
  candidateScore(index: number): number | undefined;
  candidateViable(index: number): boolean | undefined;
  candidateReason(index: number): string | undefined;
}

interface TemplateScoresWasm extends DisposableWasmValue {
  sumProbabilities(): Float64Array;
  meanLogProbabilities(): Float64Array;
  meanProbabilities(): Float64Array;
}

interface FittingWasmBindings {
  default(): Promise<unknown>;
  MahalanobisWhitener: new (
    columns: number,
    rank: number,
    mean: Float64Array,
    basis: Float64Array,
    eigenvalues: Float64Array,
    inverseScales: Float64Array,
    ridge: number,
  ) => WhitenerWasm;
  GroupedReducedCovarianceAccumulator: new (
    columns: number,
    center: Float64Array,
    basis: Float64Array,
    components: number,
    offsets: Uint32Array,
  ) => GroupedReducedCovarianceAccumulatorWasm;
  RbfModel: new (
    inputDimensions: number,
    outputDimensions: number,
    nodeCount: number,
    nodes: Float64Array,
    coordinateOffset: Float64Array,
    coordinateScale: Float64Array,
    weights: Float64Array,
    polynomial: Float64Array,
    lambda: number,
    effectiveDegreesOfFreedom: number,
    gcv: number,
  ) => RbfWasm;
  RbfFitPlan: new (
    inputDimensions: number,
    nodeCount: number,
    nodes: Float64Array,
    coordinateOffset: Float64Array,
    coordinateScale: Float64Array,
    kernel: Float64Array,
    kernelScale: number,
    lambdas: Float64Array,
    spectralBasis: Float64Array,
    residualRatios: Float64Array,
    residualTraces: Float64Array,
  ) => RbfFitPlanWasm;
  fitNeutralCentering(values: Float64Array, rows: number, columns: number): CenteringWasm;
  groupRowMeans(
    values: Float64Array,
    rows: number,
    columns: number,
    offsets: Uint32Array,
  ): Float64Array;
  fitMahalanobisWhitener(
    values: Float64Array,
    rows: number,
    columns: number,
    ridgeScale: number,
  ): WhitenerWasm;
  fitPca(
    values: Float64Array,
    rows: number,
    columns: number,
    maxComponents: number,
    varianceThreshold: number,
  ): PcaWasm;
  fitAffineFisher(
    centroids: Float64Array,
    nodeCount: number,
    columns: number,
    whitener: WhitenerWasm,
    maxComponents: number,
    orientTo: number,
  ): AffineFisherWasm;
  manifoldPosition(
    mean: Float64Array,
    basis: Float64Array,
    components: number,
    coordinates: Float64Array,
  ): Float64Array;
  euclideanProjectRows(
    values: Float64Array,
    rows: number,
    mean: Float64Array,
    basis: Float64Array,
    components: number,
  ): Float64Array;
  euclideanAblateRows(
    values: Float64Array,
    rows: number,
    mean: Float64Array,
    basis: Float64Array,
    components: number,
    coefficients: Float64Array,
  ): Float64Array;
  euclideanPearsonCorrelations(
    values: Float64Array,
    rows: number,
    columns: number,
  ): Float64Array;
  euclideanPairwiseDistances(
    values: Float64Array,
    rows: number,
    columns: number,
  ): Float64Array;
  deriveSpectralEmbedding(
    gram: Float64Array,
    nodeCount: number,
    maxDimensions: number,
    minDimensions?: number,
    kNn?: number,
    bandwidth?: number,
  ): SpectralWasm;
  selectTopologyFromTargets(
    consensusGram: Float64Array,
    nodeCount: number,
    targets: Float64Array,
    targetOffsets: Uint32Array,
    maxDimensions: number,
    varianceThreshold: number,
    requestedFitMode: string,
    minDimensions: number | undefined,
    kNn: number | undefined,
    bandwidth: number | undefined,
    persistenceFraction: number,
    smoothing: number | undefined,
  ): TopologyWasm;
  fitRbfAutoSmoothed(
    nodes: Float64Array,
    nodeCount: number,
    inputDimensions: number,
    values: Float64Array,
    outputDimensions: number,
  ): RbfWasm;
  fitRbfSmoothed(
    nodes: Float64Array,
    nodeCount: number,
    inputDimensions: number,
    values: Float64Array,
    outputDimensions: number,
    smoothing: number,
  ): RbfWasm;
  fitSigmaFieldAutoSmoothed(
    surface: RbfWasm,
    covariances: Float64Array,
    coordinates: Float64Array,
    embeddedCoordinates: Float64Array,
    intrinsicDimensions: number,
    periodicDimensions: number,
    floorFraction: number,
    plan: RbfFitPlanWasm | undefined,
  ): SigmaFieldWasm;
  fitSigmaFieldSmoothed(
    surface: RbfWasm,
    covariances: Float64Array,
    coordinates: Float64Array,
    embeddedCoordinates: Float64Array,
    intrinsicDimensions: number,
    periodicDimensions: number,
    smoothing: number,
    floorFraction: number,
    plan: RbfFitPlanWasm | undefined,
  ): SigmaFieldWasm;
  invertRbfOrigin(
    surface: RbfWasm,
    coordinates: Float64Array,
    embeddedCoordinates: Float64Array,
    intrinsicDimensions: number,
    periodicDimensions: number,
    maxIterations: number,
    restartCount: number,
    damping: number,
  ): RbfOriginWasm;
  normalizeTemplateScores(
    sumLogProbabilities: Float64Array,
    tokenCounts: Uint32Array,
  ): TemplateScoresWasm;
}

interface ResolvedMatrix {
  values: Float64Array;
  rows: number;
  columns: number;
}

const scope = globalThis as unknown as WorkerScope;
let bindingsPromise: Promise<FittingWasmBindings> | null = null;

scope.addEventListener("message", (event) => {
  if (!isRequest(event.data)) return;
  void handle(event.data);
});

async function handle(request: FittingWorkerRequest): Promise<void> {
  try {
    progress(request.requestId, "loading");
    const bindings = await loadBindings();
    if (jobUsesSpool(request.job)) progress(request.requestId, "reading");
    const result = await execute(bindings, request.job, () => {
      progress(request.requestId, "computing");
    });
    respond({
      protocolVersion: FITTING_WORKER_PROTOCOL_VERSION,
      requestId: request.requestId,
      kind: "result",
      result,
    });
  } catch (error) {
    const failure = fittingFailure(error);
    respond({
      protocolVersion: FITTING_WORKER_PROTOCOL_VERSION,
      requestId: request.requestId,
      kind: "error",
      error: failure,
    });
  }
}

async function execute(
  bindings: FittingWasmBindings,
  job: FittingWorkerJob,
  onCompute: () => void,
): Promise<CenteringResult | WhiteningResult | AffineFisherResult | ReducedCovarianceResult | PcaResult | MatrixResult | SpectralResult | TopologyResult | RbfResult | SigmaFieldResult | RbfOriginResult | TemplateScoreResult> {
  switch (job.operation) {
    case "center": {
      const source = await resolveMatrix(job.source);
      onCompute();
      const value = bindings.fitNeutralCentering(source.values, source.rows, source.columns);
      try {
        return {
          operation: "center",
          rows: value.rows,
          columns: value.columns,
          mean: value.mean(),
          centered: value.centered(),
        };
      } finally {
        value.free();
      }
    }
    case "group_means": {
      const source = await resolveMatrix(job.source);
      onCompute();
      const values = bindings.groupRowMeans(
        source.values,
        source.rows,
        source.columns,
        job.offsets,
      );
      return {
        operation: "group_means",
        rows: job.offsets.length - 1,
        columns: source.columns,
        values,
      };
    }
    case "whiten": {
      const source = await resolveMatrix(job.source);
      const transform = job.transform ? await resolveMatrix(job.transform) : null;
      onCompute();
      const value = bindings.fitMahalanobisWhitener(
        source.values,
        source.rows,
        source.columns,
        job.ridgeScale,
      );
      try {
        return {
          operation: "whiten",
          columns: value.columns,
          rank: value.rank,
          ridge: value.ridge,
          mean: value.mean(),
          basis: value.basis(),
          eigenvalues: value.eigenvalues(),
          inverseScales: value.inverseScales(),
          transformed: transform ? value.transform(transform.values, transform.rows) : null,
          transformedRows: transform?.rows ?? null,
        };
      } finally {
        value.free();
      }
    }
    case "pca": {
      const source = await resolveMatrix(job.source);
      onCompute();
      const value = bindings.fitPca(
        source.values,
        source.rows,
        source.columns,
        job.maxComponents,
        job.varianceThreshold,
      );
      try {
        return {
          operation: "pca",
          rows: value.rows,
          columns: value.columns,
          components: value.components,
          mean: value.mean(),
          basis: value.basis(),
          eigenvalues: value.eigenvalues(),
          explainedVariance: value.explainedVariance(),
          cumulativeVariance: value.cumulativeVariance(),
          scores: value.scores(),
        };
      } finally {
        value.free();
      }
    }
    case "affine_fisher": {
      const source = await resolveMatrix(job.source);
      onCompute();
      const whitener = new bindings.MahalanobisWhitener(
        job.whitener.columns,
        job.whitener.rank,
        job.whitener.mean,
        job.whitener.basis,
        job.whitener.eigenvalues,
        job.whitener.inverseScales,
        job.whitener.ridge,
      );
      try {
        const value = bindings.fitAffineFisher(
          source.values,
          source.rows,
          source.columns,
          whitener,
          job.maxComponents,
          job.orientTo,
        );
        try {
          return {
            operation: "affine_fisher",
            nodeCount: value.nodeCount,
            columns: value.columns,
            components: value.components,
            centroidMean: value.centroidMean(),
            mean: value.mean(),
            basis: value.basis(),
            nodeCoordinates: value.nodeCoordinates(),
            muCoordinates: value.muCoordinates(),
            whitenedGram: value.whitenedGram(),
            neutralCrossGram: value.neutralCrossGram(),
            explainedVariance: value.explainedVariance,
            mahalanobisShare: value.mahalanobisShare,
          } satisfies AffineFisherResult;
        } finally {
          value.free();
        }
      } finally {
        whitener.free();
      }
    }
    case "reduced_covariances": {
      return withBrowserActivationSpool(async (spool) => {
        const reader = await spool.openCommitted(job.source.identity);
        const spec = reader.layers.find((candidate) => candidate.layer === job.source.layer);
        if (!spec) throw new FittingExecutionError(
          "FITTING_LAYER_NOT_FOUND",
          `Activation capture ${job.source.identity.captureSha256} has no layer ${job.source.layer}`,
          false,
        );
        const accumulator = new bindings.GroupedReducedCovarianceAccumulator(
          spec.width,
          job.mean,
          job.basis,
          job.components,
          job.offsets,
        );
        try {
          let nextRow = 0;
          for await (const chunk of reader.readChunks(job.source.layer)) {
            if (chunk.startRow !== nextRow || chunk.width !== spec.width) {
              throw new FittingExecutionError(
                "FITTING_SPOOL_DISCONTIGUOUS",
                `Activation layer ${job.source.layer} is not a contiguous row matrix`,
                false,
              );
            }
            accumulator.append(chunk.values, chunk.startRow, chunk.rows);
            nextRow += chunk.rows;
          }
          if (nextRow !== spec.rows) {
            throw new FittingExecutionError(
              "FITTING_SPOOL_INCOMPLETE",
              `Activation layer ${job.source.layer} contains ${nextRow} of ${spec.rows} rows`,
              false,
            );
          }
          onCompute();
          return {
            operation: "reduced_covariances",
            nodeCount: accumulator.nodeCount,
            components: accumulator.components,
            covariances: accumulator.finalize(),
          };
        } finally {
          accumulator.free();
        }
      });
    }
    case "position": {
      onCompute();
      const values = bindings.manifoldPosition(
        job.mean,
        job.basis,
        job.components,
        job.coordinates,
      );
      return { operation: "position", rows: 1, columns: values.length, values };
    }
    case "project": {
      const source = await resolveMatrix(job.source);
      onCompute();
      const values = bindings.euclideanProjectRows(
        source.values,
        source.rows,
        job.mean,
        job.basis,
        job.components,
      );
      return {
        operation: "project",
        rows: source.rows,
        columns: job.components,
        values,
      };
    }
    case "ablate": {
      const source = await resolveMatrix(job.source);
      onCompute();
      const values = bindings.euclideanAblateRows(
        source.values,
        source.rows,
        job.mean,
        job.basis,
        job.components,
        job.coefficients,
      );
      return { operation: "ablate", rows: source.rows, columns: source.columns, values };
    }
    case "correlations": {
      const source = await resolveMatrix(job.source);
      onCompute();
      const values = bindings.euclideanPearsonCorrelations(
        source.values,
        source.rows,
        source.columns,
      );
      return {
        operation: "correlations",
        rows: source.columns,
        columns: source.columns,
        values,
      };
    }
    case "pairwise": {
      const source = await resolveMatrix(job.source);
      onCompute();
      const values = bindings.euclideanPairwiseDistances(
        source.values,
        source.rows,
        source.columns,
      );
      return { operation: "pairwise", rows: source.rows, columns: source.rows, values };
    }
    case "spectral": {
      onCompute();
      const value = bindings.deriveSpectralEmbedding(
        job.gram,
        job.nodeCount,
        job.maxDimensions,
        job.minDimensions,
        job.kNn,
        job.bandwidth,
      );
      try {
        return {
          operation: "spectral",
          nodeCount: value.nodeCount,
          dimensions: value.dimensions,
          heuristicDimensions: value.heuristicDimensions,
          pinned: value.pinned,
          kNn: value.kNn,
          bandwidth: value.bandwidth,
          gapMagnitude: value.gapMagnitude,
          eigenvalues: value.eigenvalues(),
          coordinates: value.coordinates(),
        };
      } finally {
        value.free();
      }
    }
    case "topology": {
      assertTopologyFitMode(job.requestedFitMode);
      onCompute();
      const value = bindings.selectTopologyFromTargets(
        job.consensusGram,
        job.nodeCount,
        job.targets,
        job.targetOffsets,
        job.maxDimensions,
        job.varianceThreshold,
        job.requestedFitMode,
        job.minDimensions,
        job.kNn,
        job.bandwidth,
        job.persistenceFraction,
        job.smoothing,
      );
      try {
        const winnerPlanValue = value.winnerPlan();
        const winnerPlan = winnerPlanValue === undefined
          ? null
          : (() => {
              try {
                return serializeRbfPlan(winnerPlanValue);
              } finally {
                winnerPlanValue.free();
              }
            })();
        const candidates = Array.from({ length: value.candidateCount }, (_, index) => ({
          name: required(value.candidateName(index), "topology candidate name"),
          fitMode: required(value.candidateFitMode(index), "topology candidate fit mode"),
          dimensions: required(value.candidateDimensions(index), "topology candidate dimensions"),
          score: finiteOrNull(required(
            value.candidateScore(index),
            "topology candidate score",
          )),
          viable: required(value.candidateViable(index), "topology candidate viability"),
          reason: required(value.candidateReason(index), "topology candidate reason"),
        }));
        return {
          operation: "topology",
          winnerName: value.winnerName,
          fitMode: value.fitMode,
          intrinsicDimensions: value.intrinsicDimensions,
          periodicDimensions: value.periodicDimensions,
          persistentLoops: value.persistentLoops,
          usedFaintCycle: value.usedFaintCycle,
          coordinates: value.coordinates(),
          embeddedCoordinates: value.embeddedCoordinates(),
          candidates,
          diagnostics: value.diagnosticsKind === "pca"
            ? {
                kind: "pca",
                perComponentVariance: value.diagnosticsPerComponentVariance(),
                cumulativeVariance: value.diagnosticsCumulativeVariance(),
                pickedDimensions: value.diagnosticsPickedDimensions,
                threshold: required(value.diagnosticsThreshold, "PCA variance threshold"),
              }
            : {
                kind: "spectral",
                eigenvalues: value.diagnosticsEigenvalues(),
                pickedDimensions: value.diagnosticsPickedDimensions,
                gapMagnitude: required(value.diagnosticsGapMagnitude, "spectral gap"),
                bandwidth: required(value.diagnosticsBandwidth, "spectral bandwidth"),
                kNn: required(value.diagnosticsKNn, "spectral neighbor count"),
                componentCount: 1,
                heuristicDimensions: required(
                  value.diagnosticsHeuristicDimensions,
                  "spectral heuristic dimensions",
                ),
                minDimensions: value.diagnosticsMinDimensions ?? null,
                pinned: required(value.diagnosticsPinned, "spectral pinned state"),
              },
          winnerPlan,
        };
      } finally {
        value.free();
      }
    }
    case "rbf": {
      assertOptionalPair(job.queries, job.queryRows, "RBF queries");
      onCompute();
      const plan = job.plan ? deserializeRbfPlan(bindings, job.plan) : null;
      let value: RbfWasm;
      try {
        value = plan === null
          ? job.smoothing === undefined
            ? bindings.fitRbfAutoSmoothed(
                job.nodes,
                job.nodeCount,
                job.inputDimensions,
                job.values,
                job.outputDimensions,
              )
            : bindings.fitRbfSmoothed(
                job.nodes,
                job.nodeCount,
                job.inputDimensions,
                job.values,
                job.outputDimensions,
                job.smoothing,
              )
          : job.smoothing === undefined
            ? plan.fitAutoSmoothed(job.values, job.outputDimensions)
            : plan.fitSmoothed(job.values, job.outputDimensions, job.smoothing);
      } finally {
        plan?.free();
      }
      try {
        return {
          operation: "rbf",
          model: serializeRbf(value),
          evaluated: job.queries && job.queryRows !== undefined
            ? value.evaluate(job.queries, job.queryRows)
            : null,
          queryRows: job.queryRows ?? null,
        };
      } finally {
        value.free();
      }
    }
    case "sigma_field": {
      onCompute();
      const surface = deserializeRbf(bindings, job.surface);
      try {
        const plan = job.plan ? deserializeRbfPlan(bindings, job.plan) : undefined;
        const value = job.smoothing === undefined
          ? bindings.fitSigmaFieldAutoSmoothed(
              surface,
              job.covariances,
              job.coordinates,
              job.embeddedCoordinates,
              job.intrinsicDimensions,
              job.periodicDimensions,
              job.floorFraction,
              plan,
            )
          : bindings.fitSigmaFieldSmoothed(
              surface,
              job.covariances,
              job.coordinates,
              job.embeddedCoordinates,
              job.intrinsicDimensions,
              job.periodicDimensions,
              job.smoothing,
              job.floorFraction,
              plan,
            );
        try {
          const model = value.model();
          try {
            return {
              operation: "sigma_field",
              model: serializeRbf(model),
              sigmaMean: value.sigmaMean,
              sigmaMin: value.sigmaMin,
              sigmaMax: value.sigmaMax,
            };
          } finally {
            model.free();
          }
        } finally {
          value.free();
        }
      } finally {
        surface.free();
      }
    }
    case "rbf_origin": {
      onCompute();
      const surface = deserializeRbf(bindings, job.surface);
      try {
        const value = bindings.invertRbfOrigin(
          surface,
          job.coordinates,
          job.embeddedCoordinates,
          job.intrinsicDimensions,
          job.periodicDimensions,
          job.maxIterations,
          job.restartCount,
          job.damping,
        );
        try {
          return {
            operation: "rbf_origin",
            coordinates: value.coordinates(),
            distance: value.distance,
          };
        } finally {
          value.free();
        }
      } finally {
        surface.free();
      }
    }
    case "template_scores": {
      onCompute();
      const value = bindings.normalizeTemplateScores(
        job.sumLogProbabilities,
        job.tokenCounts,
      );
      try {
        return {
          operation: "template_scores",
          sumProbabilities: value.sumProbabilities(),
          meanLogProbabilities: value.meanLogProbabilities(),
          meanProbabilities: value.meanProbabilities(),
        };
      } finally {
        value.free();
      }
    }
  }
}

async function resolveMatrix(source: FittingMatrixSource): Promise<ResolvedMatrix> {
  if (source.kind === "inline") {
    validateMatrix(source.values, source.rows, source.columns);
    return source;
  }
  return readActivationLayer(source.identity, source.layer);
}

async function readActivationLayer(
  identity: ActivationSpoolIdentity,
  layer: number,
): Promise<ResolvedMatrix> {
  return withBrowserActivationSpool(async (spool) => {
    const reader = await spool.openCommitted(identity);
    const spec = reader.layers.find((candidate) => candidate.layer === layer);
    if (!spec) throw new FittingExecutionError(
      "FITTING_LAYER_NOT_FOUND",
      `Activation capture ${identity.captureSha256} has no layer ${layer}`,
      false,
    );
    const elements = checkedElements(spec.rows, spec.width);
    const values = new Float64Array(elements);
    let nextRow = 0;
    for await (const chunk of reader.readChunks(layer)) {
      if (chunk.startRow !== nextRow || chunk.width !== spec.width) {
        throw new FittingExecutionError(
          "FITTING_SPOOL_DISCONTIGUOUS",
          `Activation layer ${layer} is not a contiguous row matrix`,
          false,
        );
      }
      values.set(chunk.values, chunk.startRow * spec.width);
      nextRow += chunk.rows;
    }
    if (nextRow !== spec.rows) {
      throw new FittingExecutionError(
        "FITTING_SPOOL_INCOMPLETE",
        `Activation layer ${layer} contains ${nextRow} of ${spec.rows} rows`,
        false,
      );
    }
    return { values, rows: spec.rows, columns: spec.width };
  });
}

function serializeRbf(value: RbfWasm): SerializedRbfModel {
  return {
    inputDimensions: value.inputDimensions,
    outputDimensions: value.outputDimensions,
    nodeCount: value.nodeCount,
    lambda: value.lambda,
    effectiveDegreesOfFreedom: value.effectiveDegreesOfFreedom,
    gcv: value.gcv,
    nodes: value.nodes(),
    coordinateOffset: value.coordinateOffset(),
    coordinateScale: value.coordinateScale(),
    weights: value.weights(),
    polynomial: value.polynomial(),
  };
}

function serializeRbfPlan(value: RbfFitPlanWasm): SerializedRbfFitPlan {
  return {
    inputDimensions: value.inputDimensions,
    nodeCount: value.nodeCount,
    nodes: value.nodes(),
    coordinateOffset: value.coordinateOffset(),
    coordinateScale: value.coordinateScale(),
    kernel: value.kernel(),
    kernelScale: value.kernelScale,
    lambdas: value.lambdas(),
    spectralBasis: value.spectralBasis(),
    residualRatios: value.residualRatios(),
    residualTraces: value.residualTraces(),
  };
}

function deserializeRbfPlan(
  bindings: FittingWasmBindings,
  value: SerializedRbfFitPlan,
): RbfFitPlanWasm {
  return new bindings.RbfFitPlan(
    value.inputDimensions,
    value.nodeCount,
    value.nodes,
    value.coordinateOffset,
    value.coordinateScale,
    value.kernel,
    value.kernelScale,
    value.lambdas,
    value.spectralBasis,
    value.residualRatios,
    value.residualTraces,
  );
}

function deserializeRbf(
  bindings: FittingWasmBindings,
  value: SerializedRbfModel,
): RbfWasm {
  return new bindings.RbfModel(
    value.inputDimensions,
    value.outputDimensions,
    value.nodeCount,
    value.nodes,
    value.coordinateOffset,
    value.coordinateScale,
    value.weights,
    value.polynomial,
    value.lambda,
    value.effectiveDegreesOfFreedom,
    value.gcv,
  );
}

function loadBindings(): Promise<FittingWasmBindings> {
  bindingsPromise ??= (async () => {
    const moduleUrl = new URL("/wasm/drowse_fitting_wasm.js", scope.location.origin).href;
    const bindings = await import(/* @vite-ignore */ moduleUrl) as FittingWasmBindings;
    await bindings.default();
    return bindings;
  })().catch((error) => {
    bindingsPromise = null;
    throw new FittingExecutionError(
      "WASM_FITTING_MODULE_UNAVAILABLE",
      `Unable to load the Drowse fitting module: ${message(error)}`,
      true,
    );
  });
  return bindingsPromise;
}

function isRequest(value: unknown): value is FittingWorkerRequest {
  if (!isRecord(value)) return false;
  if (value.protocolVersion !== FITTING_WORKER_PROTOCOL_VERSION || value.kind !== "run") {
    return false;
  }
  return typeof value.requestId === "string" && value.requestId.length > 0 &&
    isRecord(value.job) && typeof value.job.operation === "string" &&
    OPERATIONS.has(value.job.operation);
}

const OPERATIONS = new Set([
  "center",
  "group_means",
  "whiten",
  "pca",
  "affine_fisher",
  "reduced_covariances",
  "position",
  "project",
  "ablate",
  "correlations",
  "pairwise",
  "spectral",
  "topology",
  "rbf",
  "sigma_field",
  "rbf_origin",
  "template_scores",
]);

function jobUsesSpool(job: FittingWorkerJob): boolean {
  if ("source" in job && job.source.kind === "activation_spool") return true;
  return job.operation === "whiten" && job.transform?.kind === "activation_spool";
}

function validateMatrix(values: Float64Array, rows: number, columns: number): void {
  if (!(values instanceof Float64Array)) {
    throw new FittingExecutionError(
      "FITTING_INVALID_MATRIX",
      "Fitting matrix values must be a Float64Array",
      false,
    );
  }
  const elements = checkedElements(rows, columns);
  if (values.length !== elements) {
    throw new FittingExecutionError(
      "FITTING_INVALID_MATRIX",
      `Fitting matrix has ${values.length} values, expected ${elements}`,
      false,
    );
  }
}

function checkedElements(rows: number, columns: number): number {
  if (!Number.isSafeInteger(rows) || rows <= 0 || !Number.isSafeInteger(columns) || columns <= 0) {
    throw new FittingExecutionError(
      "FITTING_INVALID_MATRIX",
      "Fitting matrix dimensions must be positive safe integers",
      false,
    );
  }
  const elements = rows * columns;
  if (!Number.isSafeInteger(elements) || elements > FITTING_WORKER_MAX_MATRIX_ELEMENTS) {
    throw new FittingExecutionError(
      "FITTING_MATRIX_TOO_LARGE",
      `Fitting matrix exceeds the ${FITTING_WORKER_MAX_MATRIX_ELEMENTS}-element safety ceiling`,
      false,
    );
  }
  return elements;
}

function assertOptionalPair(
  left: unknown,
  right: unknown,
  label: string,
): void {
  if ((left === undefined) !== (right === undefined)) {
    throw new FittingExecutionError(
      "FITTING_INVALID_REQUEST",
      `${label} must provide both values or neither value`,
      false,
    );
  }
}

function assertTopologyFitMode(value: unknown): asserts value is "pca" | "spectral" | "auto" {
  if (value !== "pca" && value !== "spectral" && value !== "auto") {
    throw new FittingExecutionError(
      "FITTING_INVALID_REQUEST",
      "Topology fitting requires pca, spectral, or auto mode",
      false,
    );
  }
}

function finiteOrNull(value: number): number | null {
  return Number.isFinite(value) ? value : null;
}

function required<T>(value: T | undefined, label: string): T {
  if (value === undefined) {
    throw new FittingExecutionError(
      "FITTING_KERNEL_PROTOCOL_ERROR",
      `The fitting kernel omitted ${label}`,
      false,
    );
  }
  return value;
}

function progress(requestId: string, stage: "loading" | "reading" | "computing"): void {
  respond({
    protocolVersion: FITTING_WORKER_PROTOCOL_VERSION,
    requestId,
    kind: "progress",
    stage,
  });
}

function respond(response: FittingWorkerResponse): void {
  scope.postMessage(response, response.kind === "result" ? transferableBuffers(response.result) : []);
}

function transferableBuffers(value: unknown): Transferable[] {
  const buffers = new Set<ArrayBuffer>();
  collectBuffers(value, buffers);
  return [...buffers];
}

function collectBuffers(value: unknown, buffers: Set<ArrayBuffer>): void {
  if (ArrayBuffer.isView(value)) {
    if (value.buffer instanceof ArrayBuffer) buffers.add(value.buffer);
    return;
  }
  if (Array.isArray(value)) {
    for (const item of value) collectBuffers(item, buffers);
    return;
  }
  if (isRecord(value)) {
    for (const item of Object.values(value)) collectBuffers(item, buffers);
  }
}

class FittingExecutionError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly recoverable: boolean,
  ) {
    super(message);
  }
}

function fittingFailure(error: unknown): {
  code: string;
  message: string;
  recoverable: boolean;
} {
  if (error instanceof FittingExecutionError) {
    return { code: error.code, message: error.message, recoverable: error.recoverable };
  }
  if (error instanceof ActivationSpoolError) {
    return { code: `ACTIVATION_${error.code}`, message: error.message, recoverable: false };
  }
  return {
    code: "FITTING_KERNEL_FAILED",
    message: message(error),
    recoverable: false,
  };
}

function message(error: unknown): string {
  if (error instanceof Error) return error.message;
  return typeof error === "string" ? error : "The browser fitting kernel failed";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
