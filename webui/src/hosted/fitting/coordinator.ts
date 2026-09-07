import type {
  ActivationLayerRecord,
  ActivationSpoolDescriptor,
  ActivationSpoolIdentity,
} from "./activationSpool";
import type {
  AffineFisherResult,
  FittingWorkerJob,
  FittingWorkerResult,
  FittingWorkerStage,
  MatrixResult,
  SerializedRbfFitPlan,
  SerializedRbfModel,
  SerializedMahalanobisWhitener,
  TopologyResult,
} from "./workerContracts";
import { applySerializedInverse } from "./mahalanobis";
import type { ManifoldDomain } from "../../lib/types";

export interface ActivationCaptureSink {
  appendRows(layer: number, startRow: number, values: Float32Array): Promise<void>;
}

export interface BrowserActivationCaptureSource {
  capture(
    sink: ActivationCaptureSink,
    signal: AbortSignal,
    onProgress: (completedRows: number, totalRows: number) => void,
  ): Promise<void>;
}

export interface BrowserFittingCapturePlan {
  descriptor: ActivationSpoolDescriptor;
  groupOffsets: Uint32Array;
  retainCapture?: boolean;
}

export interface BrowserAffineFittingPlan extends BrowserFittingCapturePlan {
  whiteners: ReadonlyMap<number, SerializedMahalanobisWhitener>;
  maxComponents: number;
  orientTo?: number;
  transformCentroids?: BrowserCentroidTransform;
}

export type BrowserCentroidTransform = (
  layer: number,
  centroids: BrowserFittingLayerCentroids,
) => BrowserFittingLayerCentroids;

export interface BrowserTopologyFittingPlan extends BrowserAffineFittingPlan {
  fitMode: "pca" | "spectral" | "auto";
  maxDimensions: number;
  minDimensions?: number;
  varianceThreshold: number;
  kNn?: number;
  bandwidth?: number;
  persistenceFraction: number;
  smoothing?: number | "auto";
  fitSigma?: boolean;
}

export interface BrowserAuthoredFittingPlan extends BrowserAffineFittingPlan {
  domain: ManifoldDomain;
  coordinates: Float64Array;
  embeddedCoordinates: Float64Array;
  intrinsicDimensions: number;
  fitSigma: boolean;
}

export interface BrowserFittingCoordinatorProgress {
  stage: "capturing" | "committing" | "pooling" | "fitting" | "topology" | "surface" | "covariance" | "sigma" | "origin";
  layer: number | null;
  completed: number;
  total: number;
  workerStage?: FittingWorkerStage;
}

export interface BrowserFittingAffineLayers {
  identity: ActivationSpoolIdentity;
  layers: ReadonlyMap<number, AffineFisherResult>;
  captureRetained: boolean;
}

export interface BrowserFittingTopologyFoundation extends BrowserFittingAffineLayers {
  consensusGram: Float64Array;
  topology: TopologyResult;
  dlsKept: ReadonlyMap<number, Uint32Array> | null;
  neutralLayoutCoordinate: Float64Array | null;
  anchoredNodeCoordinates: Float64Array | null;
  finalAffineLayers: ReadonlyMap<number, FinalizedBrowserAffineLayer> | null;
  curvedSurfaceLayers: ReadonlyMap<number, FinalizedBrowserCurvedLayer> | null;
}

export interface BrowserFittingAuthoredFoundation extends BrowserFittingAffineLayers {
  curvedSurfaceLayers: ReadonlyMap<number, FinalizedBrowserCurvedLayer>;
}

export interface FinalizedBrowserAffineLayer extends AffineFisherResult {
  affineMap: Float64Array | null;
}

export interface BrowserCurvedMeanLayer extends AffineFisherResult {
  surface: SerializedRbfModel;
}

export interface FinalizedBrowserCurvedLayer extends BrowserCurvedMeanLayer {
  sigmaSurface: SerializedRbfModel | null;
  sigmaSummary: BrowserSigmaFieldSummary | null;
  origin: Float64Array;
  originDistance: number;
}

export interface BrowserSigmaFieldSummary {
  mean: number;
  min: number;
  max: number;
  lambda: number;
}

export interface BrowserFittingCentroids {
  identity: ActivationSpoolIdentity;
  layers: ReadonlyMap<number, BrowserFittingLayerCentroids>;
  captureRetained: boolean;
}

export interface BrowserFittingLayerCentroids {
  rows: number;
  columns: number;
  values: Float64Array;
}

interface CaptureWriterPort {
  readonly progress: ActivationLayerRecord[];
  appendRows(layer: number, startRow: number, values: Float32Array): Promise<unknown>;
  checkpoint(layer: number): Promise<unknown>;
  sealLayer(layer: number): Promise<unknown>;
  commit(): Promise<unknown>;
  rollback(): Promise<void>;
}

export interface FittingCoordinatorSpoolPort {
  begin(descriptor: ActivationSpoolDescriptor): Promise<CaptureWriterPort>;
  removeCommitted(identity: ActivationSpoolIdentity): Promise<boolean>;
}

export interface FittingCoordinatorWorkerPort {
  run(
    job: FittingWorkerJob,
    options?: {
      signal?: AbortSignal;
      onProgress?: (stage: FittingWorkerStage) => void;
      transferOwnership?: readonly ArrayBufferView[];
    },
  ): Promise<FittingWorkerResult>;
}

export class BrowserFittingCoordinator {
  constructor(
    private readonly spool: FittingCoordinatorSpoolPort,
    private readonly worker: FittingCoordinatorWorkerPort,
  ) {}

  async captureAffineLayers(
    plan: BrowserAffineFittingPlan,
    source: BrowserActivationCaptureSource,
    options: {
      signal?: AbortSignal;
      onProgress?: (progress: BrowserFittingCoordinatorProgress) => void;
    } = {},
  ): Promise<BrowserFittingAffineLayers> {
    validateAffinePlan(plan);
    const centroids = await this.captureNodeCentroids(plan, source, options);
    const signal = options.signal ?? new AbortController().signal;
    const layers = new Map<number, AffineFisherResult>();
    let index = 0;
    try {
      for (const [layer, pooled] of centroids.layers) {
        signal.throwIfAborted();
        const whitener = plan.whiteners.get(layer)!;
        const fitCentroids = plan.transformCentroids?.(layer, {
          rows: pooled.rows,
          columns: pooled.columns,
          values: pooled.values.slice(),
        }) ?? pooled;
        if (
          fitCentroids.rows !== pooled.rows || fitCentroids.columns !== pooled.columns ||
          !(fitCentroids.values instanceof Float64Array) ||
          fitCentroids.values.length !== pooled.values.length ||
          fitCentroids.values.some((value) => !Number.isFinite(value))
        ) {
          throw new Error(`Browser centroid transform returned invalid layer ${layer}`);
        }
        const sourceValues = fitCentroids.values.slice();
        const serializedWhitener = cloneWhitener(whitener);
        const result = await this.worker.run({
          operation: "affine_fisher",
          source: {
            kind: "inline",
            values: sourceValues,
            rows: fitCentroids.rows,
            columns: fitCentroids.columns,
          },
          whitener: serializedWhitener,
          maxComponents: plan.maxComponents,
          orientTo: plan.orientTo ?? 0,
        }, {
          signal,
          transferOwnership: [
            sourceValues,
            ...whitenerTransferViews(serializedWhitener),
          ],
          onProgress: (workerStage) => options.onProgress?.({
            stage: "fitting",
            layer,
            completed: index,
            total: centroids.layers.size,
            workerStage,
          }),
        });
        layers.set(layer, requireAffineFisher(result, fitCentroids.rows, fitCentroids.columns));
        index += 1;
        options.onProgress?.({
          stage: "fitting",
          layer,
          completed: index,
          total: centroids.layers.size,
        });
      }
      signal.throwIfAborted();
      return {
        identity: centroids.identity,
        layers,
        captureRetained: centroids.captureRetained,
      };
    } catch (error) {
      if (centroids.captureRetained) {
        await this.spool.removeCommitted(centroids.identity).catch(() => undefined);
      }
      throw error;
    }
  }

  async captureAuthoredFoundation(
    plan: BrowserAuthoredFittingPlan,
    source: BrowserActivationCaptureSource,
    options: {
      signal?: AbortSignal;
      onProgress?: (progress: BrowserFittingCoordinatorProgress) => void;
    } = {},
  ): Promise<BrowserFittingAuthoredFoundation> {
    validateAuthoredPlan(plan);
    const retainCapture = plan.retainCapture === true;
    const affine = await this.captureAffineLayers(
      { ...plan, retainCapture: plan.fitSigma || retainCapture },
      source,
      options,
    );
    let removeTransientCapture = plan.fitSigma && !retainCapture;
    try {
      const signal = options.signal ?? new AbortController().signal;
      const meanLayers = await fitBrowserAuthoredSurfaceLayers(
        affine.layers,
        plan.embeddedCoordinates,
        this.worker,
        {
          signal,
          onProgress: (layer, completed, total, workerStage) =>
            options.onProgress?.({
              stage: "surface",
              layer,
              completed,
              total,
              workerStage,
            }),
        },
      );
      const curvedSurfaceLayers = await fitBrowserAuthoredFields(
        affine.identity,
        plan.groupOffsets,
        meanLayers,
        plan.domain,
        plan.coordinates,
        plan.embeddedCoordinates,
        this.worker,
        {
          signal,
          fitSigma: plan.fitSigma,
          onProgress: (stage, layer, completed, total, workerStage) =>
            options.onProgress?.({
              stage,
              layer,
              completed,
              total,
              workerStage,
            }),
        },
      );
      if (removeTransientCapture) {
        await this.spool.removeCommitted(affine.identity);
        removeTransientCapture = false;
      }
      return {
        ...affine,
        captureRetained: retainCapture,
        curvedSurfaceLayers,
      };
    } catch (error) {
      if (removeTransientCapture) {
        await this.spool.removeCommitted(affine.identity).catch(() => undefined);
      }
      throw error;
    }
  }

  async captureTopologyFoundation(
    plan: BrowserTopologyFittingPlan,
    source: BrowserActivationCaptureSource,
    options: {
      signal?: AbortSignal;
      onProgress?: (progress: BrowserFittingCoordinatorProgress) => void;
    } = {},
  ): Promise<BrowserFittingTopologyFoundation> {
    validateTopologyPlan(plan);
    const retainCapture = plan.retainCapture === true;
    const fitSigma = plan.fitSigma !== false;
    const affine = await this.captureAffineLayers(
      { ...plan, retainCapture: fitSigma || retainCapture },
      source,
      options,
    );
    let removeTransientCapture = fitSigma && !retainCapture;
    try {
    const signal = options.signal ?? new AbortController().signal;
    signal.throwIfAborted();
    const nodeCount = plan.groupOffsets.length - 1;
    const consensusGram = new Float64Array(nodeCount * nodeCount);
    const targetOffsets = new Uint32Array(affine.layers.size + 1);
    let targetElements = 0;
    let layerIndex = 0;
    for (const layer of affine.layers.values()) {
      for (let index = 0; index < consensusGram.length; index += 1) {
        consensusGram[index] += layer.whitenedGram[index] / affine.layers.size;
      }
      targetElements += layer.muCoordinates.length;
      if (targetElements > 0xffff_ffff) {
        throw new RangeError("Browser topology targets exceed the worker offset range");
      }
      targetOffsets[++layerIndex] = targetElements;
    }
    const targets = new Float64Array(targetElements);
    let targetOffset = 0;
    for (const layer of affine.layers.values()) {
      targets.set(layer.muCoordinates, targetOffset);
      targetOffset += layer.muCoordinates.length;
    }
    const result = await this.worker.run({
      operation: "topology",
      consensusGram,
      nodeCount,
      targets,
      targetOffsets,
      maxDimensions: plan.maxDimensions,
      varianceThreshold: plan.varianceThreshold,
      requestedFitMode: plan.fitMode,
      minDimensions: plan.minDimensions,
      kNn: plan.kNn,
      bandwidth: plan.bandwidth,
      persistenceFraction: plan.persistenceFraction,
      ...(typeof plan.smoothing === "number" ? { smoothing: plan.smoothing } : {}),
    }, {
      signal,
      transferOwnership: [targets, targetOffsets],
      onProgress: (workerStage) => options.onProgress?.({
        stage: "topology",
        layer: null,
        completed: 0,
        total: 1,
        workerStage,
      }),
    });
    const topology = requireTopology(result, nodeCount);
    const dlsKept = topology.fitMode === "pca"
      ? selectBrowserDlsAxes(affine.layers, topology.intrinsicDimensions)
      : null;
    const neutralLayout = topology.fitMode === "pca"
      ? anchorBrowserNeutralLayout(affine.layers, topology)
      : null;
    const finalAffineLayers = dlsKept === null
      ? null
      : finalizeBrowserAffineLayers(
          affine.layers,
          plan.whiteners,
          dlsKept,
          topology.intrinsicDimensions,
        );
    const curvedMeanSurfaces = topology.fitMode !== "spectral"
      ? null
      : await fitBrowserCurvedSurfaceLayers(
          affine.layers,
          topology,
          this.worker,
          {
            signal,
            smoothing: plan.smoothing,
            onProgress: (layer, completed, total, workerStage) =>
              options.onProgress?.({
                stage: "surface",
                layer,
                completed,
                total,
                workerStage,
              }),
          },
        );
    const curvedSurfaceLayers = curvedMeanSurfaces === null
      ? null
      : await fitBrowserCurvedSigmaFields(
          affine.identity,
          plan.groupOffsets,
          curvedMeanSurfaces,
          topology,
          this.worker,
          {
            signal,
            smoothing: plan.smoothing,
            fitSigma,
            onProgress: (stage, layer, completed, total, workerStage) =>
              options.onProgress?.({
                stage,
                layer,
                completed,
                total,
                workerStage,
              }),
          },
        );
    options.onProgress?.({
      stage: "topology",
      layer: null,
      completed: 1,
      total: 1,
    });
    if (removeTransientCapture) {
      await this.spool.removeCommitted(affine.identity);
      removeTransientCapture = false;
    }
    return {
      ...affine,
      captureRetained: retainCapture,
      consensusGram,
      topology,
      dlsKept,
      neutralLayoutCoordinate: neutralLayout?.neutral ?? null,
      anchoredNodeCoordinates: neutralLayout?.coordinates ?? null,
      finalAffineLayers,
      curvedSurfaceLayers,
    };
    } catch (error) {
      if (removeTransientCapture) {
        await this.spool.removeCommitted(affine.identity).catch(() => undefined);
      }
      throw error;
    }
  }

  async captureNodeCentroids(
    plan: BrowserFittingCapturePlan,
    source: BrowserActivationCaptureSource,
    options: {
      signal?: AbortSignal;
      onProgress?: (progress: BrowserFittingCoordinatorProgress) => void;
    } = {},
  ): Promise<BrowserFittingCentroids> {
    validatePlan(plan);
    const signal = options.signal ?? new AbortController().signal;
    signal.throwIfAborted();
    const descriptor = cloneDescriptor(plan.descriptor);
    const offsets = plan.groupOffsets.slice();
    let writer: CaptureWriterPort | null = await this.spool.begin(descriptor);
    let committed = false;
    try {
      await source.capture({
        appendRows: async (layer, startRow, values) => {
          signal.throwIfAborted();
          await writer!.appendRows(layer, startRow, values);
        },
      }, signal, (completed, total) => {
        options.onProgress?.({
          stage: "capturing",
          layer: null,
          completed,
          total,
        });
      });
      signal.throwIfAborted();
      for (const layer of descriptor.layers) {
        const progress = writer.progress.find((entry) => entry.layer === layer.layer);
        if (!progress || progress.writtenRows !== layer.rows) {
          throw new Error(
            `Activation capture layer ${layer.layer} wrote ${progress?.writtenRows ?? 0} of ${layer.rows} rows`,
          );
        }
        options.onProgress?.({
          stage: "committing",
          layer: layer.layer,
          completed: progress.writtenRows,
          total: layer.rows,
        });
        await writer.checkpoint(layer.layer);
        await writer.sealLayer(layer.layer);
      }
      await writer.commit();
      committed = true;
      writer = null;

      const layers = new Map<number, BrowserFittingLayerCentroids>();
      for (let index = 0; index < descriptor.layers.length; index += 1) {
        signal.throwIfAborted();
        const layer = descriptor.layers[index];
        const result = await this.worker.run({
          operation: "group_means",
          source: {
            kind: "activation_spool",
            identity: descriptor,
            layer: layer.layer,
          },
          offsets,
        }, {
          signal,
          onProgress: (workerStage) => options.onProgress?.({
            stage: "pooling",
            layer: layer.layer,
            completed: index,
            total: descriptor.layers.length,
            workerStage,
          }),
        });
        const pooled = requireGroupMeans(result, offsets.length - 1, layer.width);
        layers.set(layer.layer, {
          rows: pooled.rows,
          columns: pooled.columns,
          values: pooled.values.slice(),
        });
        options.onProgress?.({
          stage: "pooling",
          layer: layer.layer,
          completed: index + 1,
          total: descriptor.layers.length,
        });
      }
      signal.throwIfAborted();
      if (plan.retainCapture !== true) {
        await this.spool.removeCommitted(descriptor);
        committed = false;
      }
      return {
        identity: Object.freeze(identity(descriptor)),
        layers,
        captureRetained: plan.retainCapture === true,
      };
    } catch (error) {
      if (writer !== null) await writer.rollback().catch(() => undefined);
      else if (committed) await this.spool.removeCommitted(descriptor).catch(() => undefined);
      throw error;
    }
  }
}

export async function fitBrowserCurvedSurfaceLayers(
  layers: ReadonlyMap<number, AffineFisherResult>,
  topology: TopologyResult,
  worker: FittingCoordinatorWorkerPort,
  options: {
    signal?: AbortSignal;
    smoothing?: number | "auto";
    onProgress?: (
      layer: number,
      completed: number,
      total: number,
      workerStage?: FittingWorkerStage,
    ) => void;
  } = {},
): Promise<ReadonlyMap<number, BrowserCurvedMeanLayer>> {
  if (topology.fitMode !== "spectral" || layers.size === 0) {
    throw new TypeError("Browser curved fitting requires a spectral topology and fitted layers");
  }
  const signal = options.signal ?? new AbortController().signal;
  const first = layers.values().next().value as AffineFisherResult;
  const nodeCount = first.nodeCount;
  if (topology.embeddedCoordinates.length % nodeCount !== 0) {
    throw new Error("The browser topology worker returned invalid embedded coordinates");
  }
  const inputDimensions = topology.embeddedCoordinates.length / nodeCount;
  if (inputDimensions <= 0) {
    throw new Error("The browser topology worker returned empty embedded coordinates");
  }
  const winnerPlan = topology.winnerPlan;
  if (
    winnerPlan === null || winnerPlan.nodeCount !== nodeCount ||
    winnerPlan.inputDimensions !== inputDimensions
  ) {
    throw new Error("The browser topology worker returned no reusable winner plan");
  }
  const finalized = new Map<number, BrowserCurvedMeanLayer>();
  let completed = 0;
  for (const [layer, fit] of layers) {
    signal.throwIfAborted();
    if (fit.nodeCount !== nodeCount) {
      throw new Error("Browser curved fitting layers disagree on node count");
    }
    const nodes = topology.embeddedCoordinates.slice();
    const values = fit.nodeCoordinates.slice();
    const plan = cloneRbfPlan(winnerPlan);
    const result = await worker.run({
      operation: "rbf",
      nodes,
      nodeCount,
      inputDimensions,
      values,
      outputDimensions: fit.components,
      plan,
      ...(typeof options.smoothing === "number"
        ? { smoothing: options.smoothing }
        : {}),
    }, {
      signal,
      transferOwnership: [nodes, values, ...rbfPlanTransferViews(plan)],
      onProgress: (workerStage) => options.onProgress?.(
        layer,
        completed,
        layers.size,
        workerStage,
      ),
    });
    finalized.set(layer, {
      ...fit,
      surface: requireRbf(result, nodeCount, inputDimensions, fit.components),
    });
    completed += 1;
    options.onProgress?.(layer, completed, layers.size);
  }
  return finalized;
}

export async function fitBrowserAuthoredSurfaceLayers(
  layers: ReadonlyMap<number, AffineFisherResult>,
  embeddedCoordinates: Float64Array,
  worker: FittingCoordinatorWorkerPort,
  options: {
    signal?: AbortSignal;
    onProgress?: (
      layer: number,
      completed: number,
      total: number,
      workerStage?: FittingWorkerStage,
    ) => void;
  } = {},
): Promise<ReadonlyMap<number, BrowserCurvedMeanLayer>> {
  if (layers.size === 0) {
    throw new TypeError("Browser authored fitting requires fitted layers");
  }
  const signal = options.signal ?? new AbortController().signal;
  const first = layers.values().next().value as AffineFisherResult;
  const nodeCount = first.nodeCount;
  const inputDimensions = embeddedCoordinates.length / nodeCount;
  if (!Number.isSafeInteger(inputDimensions) || inputDimensions <= 0) {
    throw new TypeError("Browser authored fitting has invalid embedded coordinates");
  }
  const finalized = new Map<number, BrowserCurvedMeanLayer>();
  let completed = 0;
  for (const [layer, fit] of layers) {
    signal.throwIfAborted();
    if (fit.nodeCount !== nodeCount) {
      throw new Error("Browser authored fitting layers disagree on node count");
    }
    const nodes = embeddedCoordinates.slice();
    const values = fit.nodeCoordinates.slice();
    const result = await worker.run({
      operation: "rbf",
      nodes,
      nodeCount,
      inputDimensions,
      values,
      outputDimensions: fit.components,
      smoothing: 0,
    }, {
      signal,
      transferOwnership: [nodes, values],
      onProgress: (workerStage) => options.onProgress?.(
        layer,
        completed,
        layers.size,
        workerStage,
      ),
    });
    finalized.set(layer, {
      ...fit,
      surface: requireRbf(result, nodeCount, inputDimensions, fit.components),
    });
    completed += 1;
    options.onProgress?.(layer, completed, layers.size);
  }
  return finalized;
}

export async function fitBrowserAuthoredFields(
  identity: ActivationSpoolIdentity,
  groupOffsets: Uint32Array,
  layers: ReadonlyMap<number, BrowserCurvedMeanLayer>,
  domain: ManifoldDomain,
  coordinates: Float64Array,
  embeddedCoordinates: Float64Array,
  worker: FittingCoordinatorWorkerPort,
  options: {
    signal?: AbortSignal;
    fitSigma: boolean;
    floorFraction?: number;
    onProgress?: (
      stage: "covariance" | "sigma" | "origin",
      layer: number,
      completed: number,
      total: number,
      workerStage?: FittingWorkerStage,
    ) => void;
  },
): Promise<ReadonlyMap<number, FinalizedBrowserCurvedLayer>> {
  if (layers.size === 0) throw new TypeError("Browser authored fitting requires curved layers");
  const signal = options.signal ?? new AbortController().signal;
  const nodeCount = groupOffsets.length - 1;
  const intrinsicDimensions = domainIntrinsicDimensions(domain);
  if (
    nodeCount <= 0 || coordinates.length !== nodeCount * intrinsicDimensions ||
    embeddedCoordinates.length % nodeCount !== 0
  ) {
    throw new TypeError("Browser authored fitting has invalid domain coordinates");
  }
  const finalized = new Map<number, FinalizedBrowserCurvedLayer>();
  let completed = 0;
  for (const [layer, fit] of layers) {
    signal.throwIfAborted();
    let sigmaSurface: SerializedRbfModel | null = null;
    let sigmaSummary: BrowserSigmaFieldSummary | null = null;
    if (options.fitSigma) {
      const offsets = groupOffsets.slice();
      const mean = fit.mean.slice();
      const basis = fit.basis.slice();
      const covarianceResult = await worker.run({
        operation: "reduced_covariances",
        source: { kind: "activation_spool", identity, layer },
        offsets,
        mean,
        basis,
        components: fit.components,
      }, {
        signal,
        transferOwnership: [offsets, mean, basis],
        onProgress: (workerStage) => options.onProgress?.(
          "covariance", layer, completed, layers.size, workerStage,
        ),
      });
      if (
        covarianceResult.operation !== "reduced_covariances" ||
        covarianceResult.nodeCount !== nodeCount ||
        covarianceResult.components !== fit.components
      ) {
        throw new Error("The browser fitting worker returned invalid authored covariances");
      }
      options.onProgress?.("covariance", layer, completed + 1, layers.size);
      const sigma = authoredSigmaValues(
        fit.surface,
        covarianceResult.covariances,
        fit.components,
        domain,
        coordinates,
        options.floorFraction ?? 1e-3,
      );
      const sigmaNodes = embeddedCoordinates.slice();
      const sigmaValues = Float64Array.from(sigma, (value) => Math.fround(Math.log(value)));
      const sigmaResult = await worker.run({
        operation: "rbf",
        nodes: sigmaNodes,
        nodeCount,
        inputDimensions: fit.surface.inputDimensions,
        values: sigmaValues,
        outputDimensions: 1,
      }, {
        signal,
        transferOwnership: [sigmaNodes, sigmaValues],
        onProgress: (workerStage) => options.onProgress?.(
          "sigma", layer, completed, layers.size, workerStage,
        ),
      });
      sigmaSurface = requireRbf(
        sigmaResult,
        nodeCount,
        fit.surface.inputDimensions,
        1,
      );
      sigmaSummary = {
        mean: sigma.reduce((sum, value) => sum + value, 0) / sigma.length,
        min: Math.min(...sigma),
        max: Math.max(...sigma),
        lambda: sigmaSurface.lambda,
      };
      options.onProgress?.("sigma", layer, completed + 1, layers.size);
    }
    const origin = invertAuthoredOrigin(
      fit.surface,
      domain,
      coordinates,
      12,
      3,
      1e-3,
    );
    finalized.set(layer, {
      ...fit,
      sigmaSurface,
      sigmaSummary,
      origin: origin.coordinates,
      originDistance: origin.distance,
    });
    completed += 1;
    options.onProgress?.("origin", layer, completed, layers.size);
  }
  return finalized;
}

export async function fitBrowserCurvedSigmaFields(
  identity: ActivationSpoolIdentity,
  groupOffsets: Uint32Array,
  layers: ReadonlyMap<number, BrowserCurvedMeanLayer>,
  topology: TopologyResult,
  worker: FittingCoordinatorWorkerPort,
  options: {
    signal?: AbortSignal;
    smoothing?: number | "auto";
    fitSigma?: boolean;
    floorFraction?: number;
    onProgress?: (
      stage: "covariance" | "sigma" | "origin",
      layer: number,
      completed: number,
      total: number,
      workerStage?: FittingWorkerStage,
    ) => void;
  } = {},
): Promise<ReadonlyMap<number, FinalizedBrowserCurvedLayer>> {
  if (topology.fitMode !== "spectral" || layers.size === 0) {
    throw new TypeError("Browser sigma fitting requires curved fitted layers");
  }
  const signal = options.signal ?? new AbortController().signal;
  const nodeCount = groupOffsets.length - 1;
  if (nodeCount <= 0 || groupOffsets[0] !== 0) {
    throw new TypeError("Browser sigma fitting requires non-empty capture groups");
  }
  const finalized = new Map<number, FinalizedBrowserCurvedLayer>();
  let completed = 0;
  for (const [layer, fit] of layers) {
    signal.throwIfAborted();
    if (fit.nodeCount !== nodeCount) {
      throw new Error(`Browser sigma fitting layer ${layer} disagrees on node count`);
    }
    let sigmaSurface: SerializedRbfModel | null = null;
    let sigmaSummary: BrowserSigmaFieldSummary | null = null;
    if (options.fitSigma !== false) {
      const offsets = groupOffsets.slice();
      const mean = fit.mean.slice();
      const basis = fit.basis.slice();
      const covarianceResult = await worker.run({
        operation: "reduced_covariances",
        source: { kind: "activation_spool", identity, layer },
        offsets,
        mean,
        basis,
        components: fit.components,
      }, {
        signal,
        transferOwnership: [offsets, mean, basis],
        onProgress: (workerStage) => options.onProgress?.(
          "covariance",
          layer,
          completed,
          layers.size,
          workerStage,
        ),
      });
      if (
        covarianceResult.operation !== "reduced_covariances" ||
        covarianceResult.nodeCount !== nodeCount ||
        covarianceResult.components !== fit.components
      ) {
        throw new Error("The browser fitting worker returned invalid reduced covariances");
      }
      options.onProgress?.("covariance", layer, completed + 1, layers.size);

      const sigmaSurfaceInput = cloneRbf(fit.surface);
      const sigmaCoordinates = topology.coordinates.slice();
      const sigmaEmbeddedCoordinates = topology.embeddedCoordinates.slice();
      const sigmaPlan = topology.winnerPlan === null ? undefined : cloneRbfPlan(topology.winnerPlan);
      const sigmaResult = await worker.run({
        operation: "sigma_field",
        surface: sigmaSurfaceInput,
        covariances: covarianceResult.covariances,
        coordinates: sigmaCoordinates,
        embeddedCoordinates: sigmaEmbeddedCoordinates,
        intrinsicDimensions: topology.intrinsicDimensions,
        periodicDimensions: topology.periodicDimensions,
        ...(typeof options.smoothing === "number"
          ? { smoothing: options.smoothing }
          : {}),
        plan: sigmaPlan,
        floorFraction: options.floorFraction ?? 1e-3,
      }, {
        signal,
        transferOwnership: [
          ...rbfTransferViews(sigmaSurfaceInput),
          covarianceResult.covariances,
          sigmaCoordinates,
          sigmaEmbeddedCoordinates,
          ...(sigmaPlan === undefined ? [] : rbfPlanTransferViews(sigmaPlan)),
        ],
        onProgress: (workerStage) => options.onProgress?.(
          "sigma",
          layer,
          completed,
          layers.size,
          workerStage,
        ),
      });
      if (
        sigmaResult.operation !== "sigma_field" ||
        sigmaResult.model.nodeCount !== nodeCount ||
        sigmaResult.model.inputDimensions !== fit.surface.inputDimensions
      ) {
        throw new Error("The browser fitting worker returned an invalid sigma field");
      }
      sigmaSurface = sigmaResult.model;
      sigmaSummary = {
        mean: sigmaResult.sigmaMean,
        min: sigmaResult.sigmaMin,
        max: sigmaResult.sigmaMax,
        lambda: sigmaResult.model.lambda,
      };
    }
    const originSurface = cloneRbf(fit.surface);
    const originCoordinates = topology.coordinates.slice();
    const originEmbeddedCoordinates = topology.embeddedCoordinates.slice();
    const originResult = await worker.run({
      operation: "rbf_origin",
      surface: originSurface,
      coordinates: originCoordinates,
      embeddedCoordinates: originEmbeddedCoordinates,
      intrinsicDimensions: topology.intrinsicDimensions,
      periodicDimensions: topology.periodicDimensions,
      maxIterations: 12,
      restartCount: 3,
      damping: 1e-3,
    }, {
      signal,
      transferOwnership: [
        ...rbfTransferViews(originSurface),
        originCoordinates,
        originEmbeddedCoordinates,
      ],
      onProgress: (workerStage) => options.onProgress?.(
        "origin",
        layer,
        completed,
        layers.size,
        workerStage,
      ),
    });
    if (
      originResult.operation !== "rbf_origin" ||
      originResult.coordinates.length !== topology.intrinsicDimensions
    ) {
      throw new Error("The browser fitting worker returned an invalid curved origin");
    }
    finalized.set(layer, {
      ...fit,
      sigmaSurface,
      sigmaSummary,
      origin: originResult.coordinates,
      originDistance: originResult.distance,
    });
    completed += 1;
    options.onProgress?.("origin", layer, completed, layers.size);
  }
  return finalized;
}

function authoredSigmaValues(
  surface: SerializedRbfModel,
  covariances: Float64Array,
  components: number,
  domain: ManifoldDomain,
  coordinates: Float64Array,
  floorFraction: number,
): Float64Array {
  const intrinsicDimensions = domainIntrinsicDimensions(domain);
  const nodeCount = surface.nodeCount;
  if (
    !Number.isFinite(floorFraction) || floorFraction <= 0 ||
    covariances.length !== nodeCount * components * components ||
    coordinates.length !== nodeCount * intrinsicDimensions
  ) {
    throw new TypeError("Browser authored sigma fitting received invalid geometry");
  }
  const variances = new Float64Array(nodeCount);
  for (let node = 0; node < nodeCount; node += 1) {
    const coordinate = coordinates.slice(
      node * intrinsicDimensions,
      (node + 1) * intrinsicDimensions,
    );
    const embedded = embedDomainPoint(domain, coordinate);
    const embeddedJacobian = evaluateRbfJacobian(surface, embedded);
    const domainJacobian = domainEmbedJacobian(domain, coordinate);
    const tangent = multiply(
      embeddedJacobian,
      components,
      surface.inputDimensions,
      domainJacobian,
      intrinsicDimensions,
    );
    const covariance = covariances.subarray(
      node * components * components,
      (node + 1) * components * components,
    );
    variances[node] = offSurfaceVariance(
      covariance,
      tangent,
      components,
      intrinsicDimensions,
    );
  }
  const ordered = [...variances].sort((left, right) => left - right);
  const floor = floorFraction * Math.max(ordered[Math.floor((nodeCount - 1) / 2)], 1e-12);
  return Float64Array.from(variances, (variance) =>
    Math.fround(Math.sqrt(Math.max(variance, floor)))
  );
}

function offSurfaceVariance(
  covariance: Float64Array,
  tangent: Float64Array,
  components: number,
  intrinsicDimensions: number,
): number {
  const gram = new Float64Array(intrinsicDimensions * intrinsicDimensions);
  const covarianceTangent = new Float64Array(components * intrinsicDimensions);
  for (let row = 0; row < components; row += 1) {
    for (let dimension = 0; dimension < intrinsicDimensions; dimension += 1) {
      let value = 0;
      for (let column = 0; column < components; column += 1) {
        value += covariance[row * components + column] *
          tangent[column * intrinsicDimensions + dimension];
      }
      covarianceTangent[row * intrinsicDimensions + dimension] = value;
    }
  }
  for (let left = 0; left < intrinsicDimensions; left += 1) {
    for (let right = left; right < intrinsicDimensions; right += 1) {
      let value = 0;
      for (let row = 0; row < components; row += 1) {
        value += tangent[row * intrinsicDimensions + left] *
          tangent[row * intrinsicDimensions + right];
      }
      gram[left * intrinsicDimensions + right] = value;
      gram[right * intrinsicDimensions + left] = value;
    }
  }
  const { values, vectors } = symmetricEigen(gram, intrinsicDimensions);
  const largestSingular = Math.sqrt(Math.max(values[0] ?? 0, 0));
  const tolerance = largestSingular * Math.max(components, intrinsicDimensions) * Number.EPSILON;
  let tangentTrace = 0;
  let rank = 0;
  for (let component = 0; component < intrinsicDimensions; component += 1) {
    const eigenvalue = Math.max(values[component], 0);
    if (Math.sqrt(eigenvalue) <= tolerance) continue;
    rank += 1;
    let projected = 0;
    for (let row = 0; row < components; row += 1) {
      let tangentVector = 0;
      let covarianceVector = 0;
      for (let dimension = 0; dimension < intrinsicDimensions; dimension += 1) {
        const eigenvector = vectors[dimension * intrinsicDimensions + component];
        tangentVector += tangent[row * intrinsicDimensions + dimension] * eigenvector;
        covarianceVector += covarianceTangent[row * intrinsicDimensions + dimension] * eigenvector;
      }
      projected += tangentVector * covarianceVector;
    }
    tangentTrace += projected / eigenvalue;
  }
  let totalTrace = 0;
  for (let index = 0; index < components; index += 1) {
    totalTrace += covariance[index * components + index];
  }
  const normalDimensions = components - rank;
  const variance = normalDimensions > 0
    ? (totalTrace - tangentTrace) / normalDimensions
    : totalTrace / components;
  if (!Number.isFinite(variance)) {
    throw new Error("Browser authored sigma fitting produced a non-finite variance");
  }
  return Math.max(variance, 0);
}

function invertAuthoredOrigin(
  surface: SerializedRbfModel,
  domain: ManifoldDomain,
  coordinates: Float64Array,
  maxIterations: number,
  restartCount: number,
  damping: number,
): { coordinates: Float64Array; distance: number } {
  const intrinsicDimensions = domainIntrinsicDimensions(domain);
  const nodeCount = surface.nodeCount;
  const seeds = Array.from({ length: nodeCount }, (_, node) => {
    const coordinate = coordinates.slice(
      node * intrinsicDimensions,
      (node + 1) * intrinsicDimensions,
    );
    const values = evaluateRbf(surface, embedDomainPoint(domain, coordinate));
    return { node, norm: values.reduce((sum, value) => sum + value * value, 0) };
  }).sort((left, right) => left.norm - right.norm || left.node - right.node);
  let bestCoordinates = new Float64Array();
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const { node } of seeds.slice(0, Math.min(restartCount, nodeCount))) {
    let position = coordinates.slice(
      node * intrinsicDimensions,
      (node + 1) * intrinsicDimensions,
    );
    for (let iteration = 0; iteration < maxIterations; iteration += 1) {
      const embedded = embedDomainPoint(domain, position);
      const residual = evaluateRbf(surface, embedded);
      const embeddedJacobian = evaluateRbfJacobian(surface, embedded);
      const tangent = multiply(
        embeddedJacobian,
        surface.outputDimensions,
        surface.inputDimensions,
        domainEmbedJacobian(domain, position),
        intrinsicDimensions,
      );
      const normal = new Float64Array(intrinsicDimensions * intrinsicDimensions);
      const gradient = new Float64Array(intrinsicDimensions);
      for (let left = 0; left < intrinsicDimensions; left += 1) {
        for (let output = 0; output < surface.outputDimensions; output += 1) {
          gradient[left] += tangent[output * intrinsicDimensions + left] * residual[output];
        }
        for (let right = 0; right < intrinsicDimensions; right += 1) {
          for (let output = 0; output < surface.outputDimensions; output += 1) {
            normal[left * intrinsicDimensions + right] +=
              tangent[output * intrinsicDimensions + left] *
              tangent[output * intrinsicDimensions + right];
          }
        }
        const diagonal = normal[left * intrinsicDimensions + left];
        normal[left * intrinsicDimensions + left] += damping * Math.max(diagonal, 1e-9) + 1e-9;
      }
      const step = solveLinear(normal, intrinsicDimensions, gradient);
      for (let dimension = 0; dimension < intrinsicDimensions; dimension += 1) {
        position[dimension] -= step[dimension];
      }
      position.set(clampDomainPoint(domain, position));
    }
    const residual = evaluateRbf(surface, embedDomainPoint(domain, position));
    const distance = Math.sqrt(residual.reduce((sum, value) => sum + value * value, 0));
    if (distance < bestDistance) {
      bestDistance = distance;
      bestCoordinates = position;
    }
  }
  if (!Number.isFinite(bestDistance) || bestCoordinates.length !== intrinsicDimensions) {
    throw new Error("Browser authored origin inversion failed");
  }
  return { coordinates: bestCoordinates, distance: bestDistance };
}

function evaluateRbf(surface: SerializedRbfModel, embedded: Float64Array): Float64Array {
  const normalized = Float64Array.from(embedded, (value, dimension) =>
    (value - surface.coordinateOffset[dimension]) / surface.coordinateScale[dimension]
  );
  const output = new Float64Array(surface.outputDimensions);
  for (let target = 0; target < surface.outputDimensions; target += 1) {
    let value = surface.polynomial[target];
    for (let dimension = 0; dimension < surface.inputDimensions; dimension += 1) {
      value += normalized[dimension] *
        surface.polynomial[(dimension + 1) * surface.outputDimensions + target];
    }
    for (let node = 0; node < surface.nodeCount; node += 1) {
      let squared = 0;
      for (let dimension = 0; dimension < surface.inputDimensions; dimension += 1) {
        const difference = normalized[dimension] -
          surface.nodes[node * surface.inputDimensions + dimension];
        squared += difference * difference;
      }
      value += Math.pow(Math.sqrt(squared), 3) *
        surface.weights[node * surface.outputDimensions + target];
    }
    output[target] = value;
  }
  return output;
}

function evaluateRbfJacobian(
  surface: SerializedRbfModel,
  embedded: Float64Array,
): Float64Array {
  const normalized = Float64Array.from(embedded, (value, dimension) =>
    (value - surface.coordinateOffset[dimension]) / surface.coordinateScale[dimension]
  );
  const output = new Float64Array(surface.outputDimensions * surface.inputDimensions);
  for (let target = 0; target < surface.outputDimensions; target += 1) {
    for (let dimension = 0; dimension < surface.inputDimensions; dimension += 1) {
      let derivative = surface.polynomial[
        (dimension + 1) * surface.outputDimensions + target
      ];
      for (let node = 0; node < surface.nodeCount; node += 1) {
        let squared = 0;
        for (let axis = 0; axis < surface.inputDimensions; axis += 1) {
          const difference = normalized[axis] -
            surface.nodes[node * surface.inputDimensions + axis];
          squared += difference * difference;
        }
        derivative += 3 * Math.sqrt(squared) *
          (normalized[dimension] - surface.nodes[node * surface.inputDimensions + dimension]) *
          surface.weights[node * surface.outputDimensions + target];
      }
      output[target * surface.inputDimensions + dimension] =
        derivative / surface.coordinateScale[dimension];
    }
  }
  return output;
}

function domainIntrinsicDimensions(domain: ManifoldDomain): number {
  requireBrowserDomain(domain);
  if (domain.type === "box") return domain.axes.length;
  if (domain.type === "sphere") return domain.dim;
  const dimensions = domain.embed_dim;
  if (!Number.isSafeInteger(dimensions) || (dimensions as number) <= 0) {
    throw new TypeError("Browser authored custom domain has an invalid embed_dim");
  }
  return dimensions as number;
}

export function embedBrowserAuthoredCoordinates(
  domain: ManifoldDomain,
  coordinates: Float64Array,
  nodeCount: number,
): Float64Array {
  const intrinsicDimensions = domainIntrinsicDimensions(domain);
  if (
    !Number.isSafeInteger(nodeCount) || nodeCount <= 0 ||
    coordinates.length !== nodeCount * intrinsicDimensions ||
    coordinates.some((value) => !Number.isFinite(value))
  ) {
    throw new TypeError("Browser authored coordinates do not match the domain");
  }
  const first = embedDomainPoint(domain, coordinates.slice(0, intrinsicDimensions));
  const output = new Float64Array(nodeCount * first.length);
  output.set(first);
  for (let node = 1; node < nodeCount; node += 1) {
    const embedded = embedDomainPoint(domain, coordinates.slice(
      node * intrinsicDimensions,
      (node + 1) * intrinsicDimensions,
    ));
    if (embedded.length !== first.length) {
      throw new TypeError("Browser authored domain embedding width is inconsistent");
    }
    output.set(embedded, node * first.length);
  }
  return output;
}

function embedDomainPoint(domain: ManifoldDomain, coordinates: Float64Array): Float64Array {
  requireBrowserDomain(domain);
  if (domain.type === "custom") return coordinates.slice();
  if (domain.type === "box") {
    const output: number[] = [];
    domain.axes.forEach((axis, dimension) => {
      if (axis.periodic) {
        const angle = 2 * Math.PI * coordinates[dimension] / axis.period;
        output.push(Math.cos(angle), Math.sin(angle));
      } else {
        output.push(coordinates[dimension]);
      }
    });
    return Float64Array.from(output);
  }
  const output = new Float64Array(domain.dim + 1);
  let running = 1;
  for (let dimension = 0; dimension < domain.dim; dimension += 1) {
    output[dimension] = running * Math.cos(coordinates[dimension]);
    running *= Math.sin(coordinates[dimension]);
  }
  output[domain.dim] = running;
  return output;
}

function domainEmbedJacobian(
  domain: ManifoldDomain,
  coordinates: Float64Array,
): Float64Array {
  requireBrowserDomain(domain);
  const intrinsicDimensions = domainIntrinsicDimensions(domain);
  if (domain.type === "custom") {
    const output = new Float64Array(intrinsicDimensions * intrinsicDimensions);
    for (let index = 0; index < intrinsicDimensions; index += 1) {
      output[index * intrinsicDimensions + index] = 1;
    }
    return output;
  }
  if (domain.type === "box") {
    const embeddedDimensions = domain.axes.reduce(
      (total, axis) => total + (axis.periodic ? 2 : 1),
      0,
    );
    const output = new Float64Array(embeddedDimensions * intrinsicDimensions);
    let embedded = 0;
    domain.axes.forEach((axis, dimension) => {
      if (axis.periodic) {
        const frequency = 2 * Math.PI / axis.period;
        const angle = frequency * coordinates[dimension];
        output[embedded * intrinsicDimensions + dimension] = -frequency * Math.sin(angle);
        output[(embedded + 1) * intrinsicDimensions + dimension] = frequency * Math.cos(angle);
        embedded += 2;
      } else {
        output[embedded * intrinsicDimensions + dimension] = 1;
        embedded += 1;
      }
    });
    return output;
  }
  const output = new Float64Array((domain.dim + 1) * domain.dim);
  const sins = Array.from(coordinates, Math.sin);
  const cosines = Array.from(coordinates, Math.cos);
  for (let embedded = 0; embedded <= domain.dim; embedded += 1) {
    for (let dimension = 0; dimension < domain.dim; dimension += 1) {
      let value = 0;
      if (embedded < domain.dim) {
        if (dimension > embedded) continue;
        if (dimension === embedded) {
          value = -sins[embedded];
          for (let index = 0; index < embedded; index += 1) value *= sins[index];
        } else {
          value = cosines[embedded];
          for (let index = 0; index < embedded; index += 1) {
            value *= index === dimension ? cosines[index] : sins[index];
          }
        }
      } else {
        value = 1;
        for (let index = 0; index < domain.dim; index += 1) {
          value *= index === dimension ? cosines[index] : sins[index];
        }
      }
      output[embedded * domain.dim + dimension] = value;
    }
  }
  return output;
}

function clampDomainPoint(domain: ManifoldDomain, coordinates: Float64Array): Float64Array {
  requireBrowserDomain(domain);
  const output = coordinates.slice();
  if (domain.type === "custom") return output;
  if (domain.type === "box") {
    domain.axes.forEach((axis, dimension) => {
      output[dimension] = axis.periodic
        ? modulo(output[dimension], axis.period)
        : Math.min(axis.hi, Math.max(axis.lo, output[dimension]));
    });
    return output;
  }
  for (let dimension = 0; dimension < domain.dim - 1; dimension += 1) {
    output[dimension] = Math.min(Math.PI, Math.max(0, output[dimension]));
  }
  output[domain.dim - 1] = modulo(output[domain.dim - 1], 2 * Math.PI);
  return output;
}

function modulo(value: number, period: number): number {
  return (value % period + period) % period;
}

function multiply(
  left: Float64Array,
  rows: number,
  shared: number,
  right: Float64Array,
  columns: number,
): Float64Array {
  const output = new Float64Array(rows * columns);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      for (let inner = 0; inner < shared; inner += 1) {
        output[row * columns + column] +=
          left[row * shared + inner] * right[inner * columns + column];
      }
    }
  }
  return output;
}

function solveLinear(
  matrix: Float64Array,
  dimensions: number,
  rightHandSide: Float64Array,
): Float64Array {
  const coefficients = matrix.slice();
  const solution = rightHandSide.slice();
  for (let pivot = 0; pivot < dimensions; pivot += 1) {
    let best = pivot;
    for (let row = pivot + 1; row < dimensions; row += 1) {
      if (Math.abs(coefficients[row * dimensions + pivot]) >
        Math.abs(coefficients[best * dimensions + pivot])) best = row;
    }
    if (!Number.isFinite(coefficients[best * dimensions + pivot]) ||
      Math.abs(coefficients[best * dimensions + pivot]) <= Number.MIN_VALUE
    ) {
      throw new Error("Browser authored origin inversion produced a singular system");
    }
    if (best !== pivot) {
      for (let column = pivot; column < dimensions; column += 1) {
        const index = pivot * dimensions + column;
        const other = best * dimensions + column;
        [coefficients[index], coefficients[other]] = [coefficients[other], coefficients[index]];
      }
      [solution[pivot], solution[best]] = [solution[best], solution[pivot]];
    }
    const diagonal = coefficients[pivot * dimensions + pivot];
    for (let row = pivot + 1; row < dimensions; row += 1) {
      const factor = coefficients[row * dimensions + pivot] / diagonal;
      for (let column = pivot; column < dimensions; column += 1) {
        coefficients[row * dimensions + column] -=
          factor * coefficients[pivot * dimensions + column];
      }
      solution[row] -= factor * solution[pivot];
    }
  }
  for (let row = dimensions - 1; row >= 0; row -= 1) {
    for (let column = row + 1; column < dimensions; column += 1) {
      solution[row] -= coefficients[row * dimensions + column] * solution[column];
    }
    solution[row] /= coefficients[row * dimensions + row];
  }
  return solution;
}

function symmetricEigen(
  matrix: Float64Array,
  dimensions: number,
): { values: Float64Array; vectors: Float64Array } {
  const values = matrix.slice();
  const vectors = new Float64Array(dimensions * dimensions);
  for (let index = 0; index < dimensions; index += 1) {
    vectors[index * dimensions + index] = 1;
  }
  const tolerance = Math.max(...values.map(Math.abs), 0) * Number.EPSILON * dimensions;
  for (let iteration = 0; iteration < Math.max(32, dimensions * dimensions * 64); iteration += 1) {
    let left = 0;
    let right = 0;
    let largest = 0;
    for (let row = 0; row < dimensions; row += 1) {
      for (let column = row + 1; column < dimensions; column += 1) {
        const magnitude = Math.abs(values[row * dimensions + column]);
        if (magnitude > largest) {
          largest = magnitude;
          left = row;
          right = column;
        }
      }
    }
    if (largest <= tolerance) break;
    const app = values[left * dimensions + left];
    const aqq = values[right * dimensions + right];
    const apq = values[left * dimensions + right];
    const angle = 0.5 * Math.atan2(2 * apq, aqq - app);
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    for (let index = 0; index < dimensions; index += 1) {
      if (index === left || index === right) continue;
      const aip = values[index * dimensions + left];
      const aiq = values[index * dimensions + right];
      const nextLeft = cosine * aip - sine * aiq;
      const nextRight = sine * aip + cosine * aiq;
      values[index * dimensions + left] = nextLeft;
      values[left * dimensions + index] = nextLeft;
      values[index * dimensions + right] = nextRight;
      values[right * dimensions + index] = nextRight;
    }
    values[left * dimensions + left] =
      cosine * cosine * app - 2 * sine * cosine * apq + sine * sine * aqq;
    values[right * dimensions + right] =
      sine * sine * app + 2 * sine * cosine * apq + cosine * cosine * aqq;
    values[left * dimensions + right] = 0;
    values[right * dimensions + left] = 0;
    for (let index = 0; index < dimensions; index += 1) {
      const vip = vectors[index * dimensions + left];
      const viq = vectors[index * dimensions + right];
      vectors[index * dimensions + left] = cosine * vip - sine * viq;
      vectors[index * dimensions + right] = sine * vip + cosine * viq;
    }
  }
  const order = Array.from({ length: dimensions }, (_, index) => index)
    .sort((left, right) =>
      values[right * dimensions + right] - values[left * dimensions + left]
    );
  const eigenvalues = Float64Array.from(order, (index) => values[index * dimensions + index]);
  const eigenvectors = new Float64Array(dimensions * dimensions);
  order.forEach((source, column) => {
    for (let row = 0; row < dimensions; row += 1) {
      eigenvectors[row * dimensions + column] = vectors[row * dimensions + source];
    }
  });
  return { values: eigenvalues, vectors: eigenvectors };
}

function cloneRbf(model: SerializedRbfModel): SerializedRbfModel {
  return {
    ...model,
    nodes: model.nodes.slice(),
    coordinateOffset: model.coordinateOffset.slice(),
    coordinateScale: model.coordinateScale.slice(),
    weights: model.weights.slice(),
    polynomial: model.polynomial.slice(),
  };
}

function cloneRbfPlan(plan: SerializedRbfFitPlan): SerializedRbfFitPlan {
  return {
    ...plan,
    nodes: plan.nodes.slice(),
    coordinateOffset: plan.coordinateOffset.slice(),
    coordinateScale: plan.coordinateScale.slice(),
    kernel: plan.kernel.slice(),
    lambdas: plan.lambdas.slice(),
    spectralBasis: plan.spectralBasis.slice(),
    residualRatios: plan.residualRatios.slice(),
    residualTraces: plan.residualTraces.slice(),
  };
}

function rbfTransferViews(model: SerializedRbfModel): ArrayBufferView[] {
  return [
    model.nodes,
    model.coordinateOffset,
    model.coordinateScale,
    model.weights,
    model.polynomial,
  ];
}

function rbfPlanTransferViews(plan: SerializedRbfFitPlan): ArrayBufferView[] {
  return [
    plan.nodes,
    plan.coordinateOffset,
    plan.coordinateScale,
    plan.kernel,
    plan.lambdas,
    plan.spectralBasis,
    plan.residualRatios,
    plan.residualTraces,
  ];
}

function requireAffineFisher(
  result: FittingWorkerResult,
  nodeCount: number,
  columns: number,
): AffineFisherResult {
  if (
    result.operation !== "affine_fisher" || result.nodeCount !== nodeCount ||
    result.columns !== columns
  ) {
    throw new Error("The browser fitting worker returned an invalid affine-Fisher result");
  }
  return result;
}

function requireTopology(
  result: FittingWorkerResult,
  nodeCount: number,
): TopologyResult {
  if (
    result.operation !== "topology" ||
    result.coordinates.length !== nodeCount * result.intrinsicDimensions
  ) {
    throw new Error("The browser fitting worker returned an invalid topology result");
  }
  return result;
}

function requireRbf(
  result: FittingWorkerResult,
  nodeCount: number,
  inputDimensions: number,
  outputDimensions: number,
): SerializedRbfModel {
  if (
    result.operation !== "rbf" || result.model.nodeCount !== nodeCount ||
    result.model.inputDimensions !== inputDimensions ||
    result.model.outputDimensions !== outputDimensions
  ) {
    throw new Error("The browser fitting worker returned an invalid curved-surface result");
  }
  return result.model;
}

export function selectBrowserDlsAxes(
  layers: ReadonlyMap<number, AffineFisherResult>,
  intrinsicDimensions: number,
): ReadonlyMap<number, Uint32Array> {
  const checkable = new Map<number, Uint32Array>();
  const passing = new Map<number, Uint32Array>();
  let anyPass = false;
  for (const [layer, fit] of layers) {
    const components = Math.min(intrinsicDimensions, fit.components);
    const all = Uint32Array.from({ length: components }, (_, index) => index);
    checkable.set(layer, all);
    const kept: number[] = [];
    for (let component = 0; component < components; component += 1) {
      let minimum = Number.POSITIVE_INFINITY;
      let maximum = Number.NEGATIVE_INFINITY;
      for (let node = 0; node < fit.nodeCount; node += 1) {
        const coordinate = fit.nodeCoordinates[node * fit.components + component];
        minimum = Math.min(minimum, coordinate);
        maximum = Math.max(maximum, coordinate);
      }
      if (minimum < 0 && maximum > 0) kept.push(component);
    }
    if (kept.length > 0) {
      passing.set(layer, Uint32Array.from(kept));
      anyPass = true;
    }
  }
  return anyPass ? passing : checkable;
}

export function anchorBrowserNeutralLayout(
  layers: ReadonlyMap<number, AffineFisherResult>,
  topology: TopologyResult,
): { neutral: Float64Array; coordinates: Float64Array } {
  const nodeCount = topology.coordinates.length / topology.intrinsicDimensions;
  const crossGram = new Float64Array(nodeCount);
  for (const layer of layers.values()) {
    for (let node = 0; node < nodeCount; node += 1) {
      crossGram[node] += layer.neutralCrossGram[node] / layers.size;
    }
  }
  const neutral = new Float64Array(topology.intrinsicDimensions);
  for (let component = 0; component < topology.intrinsicDimensions; component += 1) {
    let numerator = 0;
    let denominator = 0;
    for (let node = 0; node < nodeCount; node += 1) {
      const coordinate = topology.coordinates[node * topology.intrinsicDimensions + component];
      numerator += coordinate * crossGram[node];
      denominator += coordinate * coordinate;
    }
    if (!Number.isFinite(denominator) || denominator <= Number.EPSILON) {
      throw new Error("The browser topology worker returned a degenerate flat layout");
    }
    neutral[component] = numerator / denominator;
  }
  const coordinates = topology.coordinates.slice();
  for (let node = 0; node < nodeCount; node += 1) {
    for (let component = 0; component < topology.intrinsicDimensions; component += 1) {
      coordinates[node * topology.intrinsicDimensions + component] -= neutral[component];
    }
  }
  return { neutral, coordinates };
}

export function finalizeBrowserAffineLayers(
  layers: ReadonlyMap<number, AffineFisherResult>,
  whiteners: ReadonlyMap<number, SerializedMahalanobisWhitener>,
  keptByLayer: ReadonlyMap<number, Uint32Array>,
  intrinsicDimensions: number,
): ReadonlyMap<number, FinalizedBrowserAffineLayer> {
  const finalized = new Map<number, FinalizedBrowserAffineLayer>();
  for (const [layer, axes] of keptByLayer) {
    const fit = layers.get(layer);
    const whitener = whiteners.get(layer);
    if (!fit || !whitener || axes.length === 0) {
      throw new Error(`Browser affine finalization is missing fitted layer ${layer}`);
    }
    validateWhitener(whitener, fit.columns, layer);
    const seen = new Set<number>();
    for (const axis of axes) {
      if (axis >= intrinsicDimensions || axis >= fit.components || seen.has(axis)) {
        throw new Error(`Browser affine finalization has an invalid axis for layer ${layer}`);
      }
      seen.add(axis);
    }
    const components = axes.length;
    const affineMap = components === intrinsicDimensions &&
      axes.every((axis, index) => axis === index)
      ? null
      : affineSelectionMap(intrinsicDimensions, axes);
    const basis = selectRows(fit.basis, fit.columns, axes);
    const nodeCoordinates = selectColumns(
      fit.nodeCoordinates,
      fit.nodeCount,
      fit.components,
      axes,
    );
    const muCoordinates = selectColumns(
      fit.muCoordinates,
      fit.nodeCount,
      fit.components,
      axes,
    );
    const mean = new Float64Array(fit.columns);
    for (let component = 0; component < components; component += 1) {
      let coordinate = 0;
      for (let column = 0; column < fit.columns; column += 1) {
        coordinate += whitener.mean[column] * basis[component * fit.columns + column];
      }
      for (let column = 0; column < fit.columns; column += 1) {
        mean[column] += coordinate * basis[component * fit.columns + column];
      }
    }
    const inverseBasis = applySerializedInverse(whitener, basis, components);
    let shareSquared = 0;
    for (let node = 0; node < fit.nodeCount; node += 1) {
      for (let left = 0; left < components; left += 1) {
        for (let right = 0; right < components; right += 1) {
          let metric = 0;
          for (let column = 0; column < fit.columns; column += 1) {
            metric += basis[left * fit.columns + column] *
              inverseBasis[right * fit.columns + column];
          }
          shareSquared += muCoordinates[node * components + left] * metric *
            muCoordinates[node * components + right];
        }
      }
    }
    const mahalanobisShare = Math.sqrt(Math.max(shareSquared, 0));
    if (!Number.isFinite(mahalanobisShare) || mahalanobisShare <= 0) {
      throw new Error(`Browser affine finalization produced a degenerate layer ${layer}`);
    }
    finalized.set(layer, {
      ...fit,
      components,
      mean,
      basis,
      nodeCoordinates,
      muCoordinates,
      mahalanobisShare,
      affineMap,
    });
  }
  return finalized;
}

function affineSelectionMap(
  intrinsicDimensions: number,
  axes: Uint32Array,
): Float64Array {
  const output = new Float64Array(intrinsicDimensions * axes.length);
  axes.forEach((axis, column) => {
    output[axis * axes.length + column] = 1;
  });
  return output;
}

function selectRows(
  values: Float64Array,
  columns: number,
  selected: Uint32Array,
): Float64Array {
  const output = new Float64Array(selected.length * columns);
  selected.forEach((row, outputRow) => {
    output.set(values.subarray(row * columns, (row + 1) * columns), outputRow * columns);
  });
  return output;
}

function selectColumns(
  values: Float64Array,
  rows: number,
  columns: number,
  selected: Uint32Array,
): Float64Array {
  const output = new Float64Array(rows * selected.length);
  for (let row = 0; row < rows; row += 1) {
    selected.forEach((column, outputColumn) => {
      output[row * selected.length + outputColumn] = values[row * columns + column];
    });
  }
  return output;
}

function requireGroupMeans(
  result: FittingWorkerResult,
  rows: number,
  columns: number,
): MatrixResult & { operation: "group_means" } {
  if (
    result.operation !== "group_means" || result.rows !== rows ||
    result.columns !== columns || result.values.length !== rows * columns
  ) {
    throw new Error("The browser fitting worker returned an invalid group-means result");
  }
  return result as MatrixResult & { operation: "group_means" };
}

function validatePlan(plan: BrowserFittingCapturePlan): void {
  if (!(plan.groupOffsets instanceof Uint32Array) || plan.groupOffsets.length < 2) {
    throw new TypeError("Browser fitting group offsets must contain at least one group");
  }
  if (plan.groupOffsets[0] !== 0) {
    throw new TypeError("Browser fitting group offsets must start at zero");
  }
  for (let index = 1; index < plan.groupOffsets.length; index += 1) {
    if (plan.groupOffsets[index] <= plan.groupOffsets[index - 1]) {
      throw new TypeError("Browser fitting group offsets must be strictly increasing");
    }
  }
  if (plan.descriptor.layers.length === 0) {
    throw new TypeError("Browser fitting capture needs at least one layer");
  }
  const expectedRows = plan.groupOffsets[plan.groupOffsets.length - 1];
  for (const layer of plan.descriptor.layers) {
    if (layer.rows !== expectedRows) {
      throw new TypeError(
        `Browser fitting layer ${layer.layer} row count does not match the group offsets`,
      );
    }
  }
}

function validateAffinePlan(plan: BrowserAffineFittingPlan): void {
  validatePlan(plan);
  if (!Number.isSafeInteger(plan.maxComponents) || plan.maxComponents <= 0) {
    throw new TypeError("Browser affine fitting max components must be a positive integer");
  }
  const orientTo = plan.orientTo ?? 0;
  const nodeCount = plan.groupOffsets.length - 1;
  if (!Number.isSafeInteger(orientTo) || orientTo < 0 || orientTo >= nodeCount) {
    throw new TypeError("Browser affine fitting orientation node is outside the node roster");
  }
  for (const layer of plan.descriptor.layers) {
    const whitener = plan.whiteners.get(layer.layer);
    if (!whitener) {
      throw new TypeError(`Browser affine fitting is missing a whitener for layer ${layer.layer}`);
    }
    validateWhitener(whitener, layer.width, layer.layer);
  }
}

function validateTopologyPlan(plan: BrowserTopologyFittingPlan): void {
  validateAffinePlan(plan);
  if (plan.fitMode !== "pca" && plan.fitMode !== "spectral" && plan.fitMode !== "auto") {
    throw new TypeError("Browser topology fit mode must be pca, spectral, or auto");
  }
  if (!Number.isSafeInteger(plan.maxDimensions) || plan.maxDimensions <= 0) {
    throw new TypeError("Browser topology max dimensions must be a positive integer");
  }
  if (plan.minDimensions !== undefined && (
    !Number.isSafeInteger(plan.minDimensions) || plan.minDimensions <= 0
  )) {
    throw new TypeError("Browser topology minimum dimensions must be a positive integer");
  }
  if (!Number.isFinite(plan.varianceThreshold) || plan.varianceThreshold <= 0 ||
    plan.varianceThreshold > 1
  ) {
    throw new TypeError("Browser topology variance threshold must be in (0, 1]");
  }
  if (!Number.isFinite(plan.persistenceFraction) || plan.persistenceFraction < 0) {
    throw new TypeError("Browser topology persistence fraction must be non-negative");
  }
  if (plan.kNn !== undefined && (!Number.isSafeInteger(plan.kNn) || plan.kNn <= 0)) {
    throw new TypeError("Browser topology neighbor count must be a positive integer");
  }
  if (plan.bandwidth !== undefined && (!Number.isFinite(plan.bandwidth) || plan.bandwidth <= 0)) {
    throw new TypeError("Browser topology bandwidth must be finite and positive");
  }
  if (plan.smoothing !== undefined && plan.smoothing !== "auto" &&
    (!Number.isFinite(plan.smoothing) || plan.smoothing < 0)
  ) {
    throw new TypeError("Browser topology smoothing must be auto or finite and non-negative");
  }
}

function validateAuthoredPlan(plan: BrowserAuthoredFittingPlan): void {
  validateAffinePlan(plan);
  const nodeCount = plan.groupOffsets.length - 1;
  const intrinsicDimensions = domainIntrinsicDimensions(plan.domain);
  if (
    plan.intrinsicDimensions !== intrinsicDimensions ||
    plan.coordinates.length !== nodeCount * intrinsicDimensions ||
    plan.embeddedCoordinates.length % nodeCount !== 0 ||
    plan.embeddedCoordinates.length === 0 ||
    plan.coordinates.some((value) => !Number.isFinite(value)) ||
    plan.embeddedCoordinates.some((value) => !Number.isFinite(value))
  ) {
    throw new TypeError("Browser authored fitting has invalid domain geometry");
  }
  if (plan.domain.type === "box") {
    if (plan.domain.axes.length === 0 || plan.domain.axes.some((axis) =>
      !axis.name || typeof axis.periodic !== "boolean" ||
      !Number.isFinite(axis.period) || axis.period <= 0 ||
      !Number.isFinite(axis.lo) || !Number.isFinite(axis.hi) || axis.hi <= axis.lo
    )) {
      throw new TypeError("Browser authored fitting has an invalid box domain");
    }
  } else if (plan.domain.type === "sphere") {
    if (!Number.isSafeInteger(plan.domain.dim) || plan.domain.dim <= 0) {
      throw new TypeError("Browser authored fitting has an invalid sphere domain");
    }
  }
  const expectedEmbedded = new Float64Array(plan.embeddedCoordinates.length);
  const embeddedDimensions = plan.embeddedCoordinates.length / nodeCount;
  for (let node = 0; node < nodeCount; node += 1) {
    const embedded = embedDomainPoint(plan.domain, plan.coordinates.slice(
      node * intrinsicDimensions,
      (node + 1) * intrinsicDimensions,
    ));
    if (embedded.length !== embeddedDimensions) {
      throw new TypeError("Browser authored fitting domain embedding has the wrong width");
    }
    expectedEmbedded.set(embedded, node * embeddedDimensions);
  }
  if (expectedEmbedded.some((value, index) =>
    Math.abs(value - plan.embeddedCoordinates[index]) > 1e-12
  )) {
    throw new TypeError("Browser authored fitting embedded coordinates do not match the domain");
  }
}

function validateWhitener(
  whitener: SerializedMahalanobisWhitener,
  columns: number,
  layer: number,
): void {
  if (whitener.columns !== columns || !Number.isSafeInteger(whitener.rank) || whitener.rank < 0 ||
    !Number.isFinite(whitener.ridge) || whitener.ridge <= 0 ||
    whitener.mean.length !== columns || whitener.basis.length !== whitener.rank * columns ||
    whitener.eigenvalues.length !== whitener.rank ||
    whitener.inverseScales.length !== whitener.rank
  ) {
    throw new TypeError(`Browser affine fitting whitener for layer ${layer} is invalid`);
  }
  for (const values of [
    whitener.mean,
    whitener.basis,
    whitener.eigenvalues,
    whitener.inverseScales,
  ]) {
    if (!(values instanceof Float64Array) || values.some((value) => !Number.isFinite(value))) {
      throw new TypeError(`Browser affine fitting whitener for layer ${layer} is invalid`);
    }
  }
}

function cloneWhitener(
  whitener: SerializedMahalanobisWhitener,
): SerializedMahalanobisWhitener {
  return {
    columns: whitener.columns,
    rank: whitener.rank,
    ridge: whitener.ridge,
    mean: whitener.mean.slice(),
    basis: whitener.basis.slice(),
    eigenvalues: whitener.eigenvalues.slice(),
    inverseScales: whitener.inverseScales.slice(),
  };
}

function whitenerTransferViews(
  whitener: SerializedMahalanobisWhitener,
): ArrayBufferView[] {
  return [
    whitener.mean,
    whitener.basis,
    whitener.eigenvalues,
    whitener.inverseScales,
  ];
}

function cloneDescriptor(descriptor: ActivationSpoolDescriptor): ActivationSpoolDescriptor {
  return {
    ...identity(descriptor),
    layers: descriptor.layers.map((layer) => ({ ...layer })),
  };
}

function identity(value: ActivationSpoolIdentity): ActivationSpoolIdentity {
  return {
    runtimeIdentitySha256: value.runtimeIdentitySha256,
    contextBindingSha256: value.contextBindingSha256,
    captureSha256: value.captureSha256,
  };
}
import { requireBrowserDomain } from "../../lib/manifolds/surfaceGeometry";
