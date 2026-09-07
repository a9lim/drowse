import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import baselinePrompts from "../../../../drowse/data/baseline_prompts.json";
import type { RuntimeProgressEvent, RuntimeServiceRequest } from "../../lib/runtime/contracts";
import type { ExtractRequest, FitManifoldRequest, GenerateManifoldRequest } from "../../lib/types";
import type {
  ChoiceScore,
  ChoiceScores,
  ManifoldInfo,
  ScoreTemplateResponse,
  TemplateDetail,
  VectorInfo,
} from "../../lib/types.gen";
import {
  browserFittedCurvedDiscoverPack,
  browserFittedFlatDiscoverPack,
  readVerifiedDrowseArchiveFile,
  validateDrowseArchive,
  type BrowserMergedDiscoverManifold,
} from "../artifacts";
import {
  browserFittedAuthoredPack,
  type BrowserAuthoredFittableManifold,
  type BrowserFittedSourceClosure,
  type BrowserSaeFitIdentity,
} from "../artifacts/fittedAuthoring";
import {
  mergedDiscoverManifoldPack,
  templatePayloadFromDetail,
} from "../artifacts/authoring";
import {
  BrowserFittingCoordinator,
  embedBrowserAuthoredCoordinates,
  type BrowserAuthoredFittingPlan,
  type BrowserCentroidTransform,
  type BrowserFittingAuthoredFoundation,
  type BrowserFittingCentroids,
  type BrowserFittingCoordinatorProgress,
  type BrowserFittingLayerCentroids,
  type BrowserFittingTopologyFoundation,
  type BrowserTopologyFittingPlan,
  type FinalizedBrowserAffineLayer,
} from "../fitting/coordinator";
import { BrowserFittingWorkerClient } from "../fitting/fittingWorkerClient";
import { mahalanobisNorm } from "../fitting/mahalanobis";
import type { SerializedMahalanobisWhitener, TopologyResult } from "../fitting/workerContracts";
import type { BrowserModelLoadRequest } from "./modelBackend";
import { BrowserFeasibilityCorePackCompiler } from "./browserCorePack";
import {
  resolveStructuredHookProfile,
  type StructuredHookCapacityProfile,
} from "./structuredHookProfile";
import {
  browserArtifactProducer,
  type BrowserArtifactProducer,
} from "./buildProvenance";
import {
  prepareWebLlmActivationCaptureRows,
  WebLlmActivationCaptureSource,
  type WebLlmCaptureRuntimePort,
} from "./webLlmActivationCapture";
import type { DrowseCaptureMessage, DrowseCaptureRow } from "./webLlmEngine";
import type {
  WebLlmGeneratedToken,
  WebLlmGenerationPlan,
  WebLlmGenerationResult,
  WebLlmHookProgram,
} from "./webLlmGeneration";

const CAPTURE_VERSION = 3;
const FIT_POLICY_VERSION = 1;
const LENGTH_DIRECTIVE = "Answer in one short paragraph.";
const RESPONSE_MAX_TOKENS = 256;
const DEFAULT_NAMESPACE = "local";
const NAME = /^[a-z][a-z0-9._-]{0,63}$/;
const LABEL = /^[a-z][a-z0-9_-]{0,63}$/;
const ROLE = /^[a-z0-9._-]+$/;

export interface BrowserAuthoringRuntimePort extends WebLlmCaptureRuntimePort {
  streamGeneration(
    plan: WebLlmGenerationPlan,
    onToken: (token: WebLlmGeneratedToken) => void | Promise<void>,
  ): Promise<WebLlmGenerationResult>;
  tokenizeText?(text: string): Promise<number[]>;
  exactSaeFitting?(selector: string): Promise<BrowserExactSaeFittingPort | null>;
}

export interface BrowserManifoldArtifactPort {
  request(
    request: RuntimeServiceRequest,
    onProgress: (event: RuntimeProgressEvent) => void,
  ): Promise<unknown>;
  withDrowseArchiveArchives?<T>(
    ids: readonly string[],
    signal: AbortSignal,
    consumer: (archives: readonly Blob[]) => Promise<T>,
  ): Promise<T>;
}

export interface BrowserManifoldFittingContext {
  request: RuntimeServiceRequest;
  loadRequest: BrowserModelLoadRequest;
  runtime: BrowserAuthoringRuntimePort;
  onProgress: (event: RuntimeProgressEvent) => void;
  signal: AbortSignal;
  compileSteering?: (expression: string) => Promise<WebLlmHookProgram>;
}

interface BrowserFittingCoordinatorPort {
  captureNodeCentroids(
    plan: Pick<BrowserTopologyFittingPlan, "descriptor" | "groupOffsets">,
    source: WebLlmActivationCaptureSource,
    options: {
      signal?: AbortSignal;
      onProgress?: (progress: BrowserFittingCoordinatorProgress) => void;
    },
  ): Promise<BrowserFittingCentroids>;
  captureTopologyFoundation(
    plan: BrowserTopologyFittingPlan,
    source: WebLlmActivationCaptureSource,
    options: {
      signal?: AbortSignal;
      onProgress?: (progress: BrowserFittingCoordinatorProgress) => void;
    },
  ): Promise<BrowserFittingTopologyFoundation>;
  captureAuthoredFoundation(
    plan: BrowserAuthoredFittingPlan,
    source: WebLlmActivationCaptureSource,
    options: {
      signal?: AbortSignal;
      onProgress?: (progress: BrowserFittingCoordinatorProgress) => void;
    },
  ): Promise<BrowserFittingAuthoredFoundation>;
}

export interface BrowserExactSaeFittingPort {
  layers: readonly number[];
  transformCentroids: BrowserCentroidTransform;
  provenance: BrowserSaeFitIdentity;
}

type BrowserFittableManifold =
  | BrowserMergedDiscoverManifold
  | BrowserAuthoredFittableManifold;

export interface BrowserManifoldFittingDependencies {
  allowAutomaticTopologyDiscovery: boolean;
  loadWhiteners(
    request: BrowserModelLoadRequest,
  ): Promise<ReadonlyMap<number, SerializedMahalanobisWhitener>>;
  createCoordinator(request: BrowserModelLoadRequest): BrowserFittingCoordinatorPort;
  loadExactSae(
    request: BrowserModelLoadRequest,
    selector: string,
    runtime: BrowserAuthoringRuntimePort,
  ): Promise<BrowserExactSaeFittingPort | null>;
  buildFlatPack: typeof browserFittedFlatDiscoverPack;
  buildCurvedPack: typeof browserFittedCurvedDiscoverPack;
  buildAuthoredPack: typeof browserFittedAuthoredPack;
  buildDiscoverPack: typeof mergedDiscoverManifoldPack;
  validateFittedPack(request: BrowserModelLoadRequest, archive: Blob): Promise<void>;
}

const DEFAULT_DEPENDENCIES: BrowserManifoldFittingDependencies = {
  allowAutomaticTopologyDiscovery: true,
  async loadWhiteners(request) {
    return (await BrowserFeasibilityCorePackCompiler.load(request)).fittingWhiteners();
  },
  createCoordinator(request) {
    return new BrowserFittingCoordinator(
      request.activationSpool,
      new BrowserFittingWorkerClient(),
    );
  },
  async loadExactSae(_request, selector, runtime) {
    return runtime.exactSaeFitting?.(selector) ?? null;
  },
  buildFlatPack: browserFittedFlatDiscoverPack,
  buildCurvedPack: browserFittedCurvedDiscoverPack,
  buildAuthoredPack: browserFittedAuthoredPack,
  buildDiscoverPack: mergedDiscoverManifoldPack,
  async validateFittedPack(request, archive) {
    await BrowserFeasibilityCorePackCompiler.load(request, {
      installedManifoldArchives: [archive],
    });
  },
};

export class BrowserManifoldFitting {
  private readonly dependencies: BrowserManifoldFittingDependencies;
  private readonly fittingWorker: BrowserFittingWorkerClient | null;
  private readonly producerVersion: string;
  private readonly drowseVersion: string;

  constructor(
    private readonly artifacts: BrowserManifoldArtifactPort,
    producer: BrowserArtifactProducer | string = browserArtifactProducer(),
    dependencies: Partial<BrowserManifoldFittingDependencies> = {},
  ) {
    this.producerVersion = typeof producer === "string" ? producer : producer.producerVersion;
    this.drowseVersion = typeof producer === "string" ? producer : producer.drowseVersion;
    this.fittingWorker = dependencies.createCoordinator === undefined
      ? new BrowserFittingWorkerClient()
      : null;
    this.dependencies = {
      ...DEFAULT_DEPENDENCIES,
      ...(this.fittingWorker === null
        ? {}
        : {
            createCoordinator: (request: BrowserModelLoadRequest) =>
              new BrowserFittingCoordinator(request.activationSpool, this.fittingWorker!),
          }),
      ...dependencies,
    };
  }

  dispose(): void {
    this.fittingWorker?.dispose();
  }

  async request(context: BrowserManifoldFittingContext): Promise<unknown> {
    if (context.request.service === "manifolds" && context.request.method === "fit") {
      return this.fit(context);
    }
    if (context.request.service === "manifolds" && context.request.method === "generate") {
      return this.generate(context);
    }
    if (context.request.service === "profiles" && context.request.method === "extract") {
      return this.extract(context);
    }
    if (context.request.service === "templates" && context.request.method === "score") {
      return this.scoreTemplate(context);
    }
    throw fittingError(
      "BROWSER_AUTHORING_OPERATION_UNAVAILABLE",
      `Browser authoring operation ${context.request.service}.${context.request.method} is not implemented`,
      501,
    );
  }

  private async scoreTemplate(
    context: BrowserManifoldFittingContext,
  ): Promise<ScoreTemplateResponse> {
    if (context.request.args.length !== 3) {
      throw fittingError(
        "INVALID_TEMPLATE_SCORE_REQUEST",
        "Browser template scoring expects namespace, name, and steering",
      );
    }
    const [namespace, name, rawSteering] = context.request.args;
    if (
      typeof namespace !== "string" ||
      typeof name !== "string" ||
      rawSteering !== null && typeof rawSteering !== "string"
    ) {
      throw fittingError(
        "INVALID_TEMPLATE_SCORE_REQUEST",
        "Browser template scoring requires a canonical template identity",
      );
    }
    const steering = rawSteering?.trim() || null;
    const hookProgram = steering === null
      ? null
      : context.compileSteering
        ? await context.compileSteering(steering)
        : (() => {
            throw fittingError(
              "STEERING_COMPILER_UNAVAILABLE",
              "Browser template scoring cannot apply steering without the verified compiler",
            );
          })();
    context.signal.throwIfAborted();
    const template = requireTemplate(await this.artifacts.request(
      { service: "templates", method: "get", args: [namespace, name] },
      context.onProgress,
    ));
    validateTemplateForScoring(template, namespace, name);
    const tokenizeText = context.runtime.tokenizeText?.bind(context.runtime);
    if (tokenizeText === undefined) {
      throw fittingError(
        "BROWSER_TEMPLATE_SCORING_UNAVAILABLE",
        "Exact browser template scoring requires the runtime tokenizer port",
        501,
      );
    }
    const contexts: ChoiceScores[] = [];
    for (let contextIndex = 0; contextIndex < template.contexts.length; contextIndex += 1) {
      context.signal.throwIfAborted();
      const templateContext = template.contexts[contextIndex];
      const slotIndex = templateContext.assistant.indexOf(template.slot);
      const assistantPrefix = templateContext.assistant.slice(0, slotIndex);
      const prefixTokenIds = await tokenizeText(assistantPrefix);
      const choices: ChoiceScore[] = [];
      for (let choiceIndex = 0; choiceIndex < template.values.length; choiceIndex += 1) {
        context.signal.throwIfAborted();
        const value = template.values[choiceIndex];
        const fullTokenIds = await tokenizeText(
          assistantPrefix + value,
        );
        const scoreStart = sharedTokenPrefix(prefixTokenIds, fullTokenIds);
        const scoredTokenIds = fullTokenIds.slice(scoreStart);
        let sumLogprob = 0;
        if (scoredTokenIds.length > 0) {
          const rows: Array<{ tokenId: number | null; replayScore?: {
            requestedLogprobs: Array<{ tokenId: number; logprob: number }>;
          } }> = [];
          const uniqueScoreTokenIds = [...new Set(scoredTokenIds)];
          await context.runtime.streamGeneration({
            input: {
              kind: "chat",
              messages: templateContext.turns.map((turn) => ({
                role: requireChatRole(turn.role),
                content: turn.content,
              })),
            },
            sampling: {
              temperature: 1,
              top_p: 1,
              top_k: 0,
              max_tokens: fullTokenIds.length,
              presence_penalty: 0,
              frequency_penalty: 0,
            },
            generationSeat: "assistant",
            replay: {
              forcedPrefixTokenIds: fullTokenIds,
              scoreTokenIds: uniqueScoreTokenIds,
            },
            steeringExpression: steering,
            hookProgram,
            signal: context.signal,
            onRawToken(token) {
              rows.push(token);
            },
          }, () => undefined);
          if (rows.length !== fullTokenIds.length) {
            throw fittingError(
              "INCOMPLETE_TEMPLATE_REPLAY",
              "The browser model stopped before the forced template candidate was complete",
              500,
            );
          }
          for (let index = scoreStart; index < fullTokenIds.length; index += 1) {
            const expectedTokenId = fullTokenIds[index];
            const row = rows[index];
            if (row.tokenId !== expectedTokenId || row.replayScore === undefined) {
              throw fittingError(
                "INVALID_TEMPLATE_REPLAY",
                "The browser model did not return exact forced-replay metadata",
                500,
              );
            }
            const scored = row.replayScore.requestedLogprobs.find(
              (entry) => entry.tokenId === expectedTokenId,
            );
            if (scored === undefined || Number.isNaN(scored.logprob)) {
              throw fittingError(
                "INVALID_TEMPLATE_REPLAY",
                "The browser model omitted a forced token log-probability",
                500,
              );
            }
            sumLogprob += scored.logprob;
          }
        }
        choices.push({
          text: value,
          label: template.labels[choiceIndex],
          n_tokens: scoredTokenIds.length,
          sum_logprob: sumLogprob,
          mean_logprob: scoredTokenIds.length === 0
            ? sumLogprob
            : sumLogprob / scoredTokenIds.length,
          prob_sum: 0,
          prob_mean: 0,
        });
      }
      applyRestrictedChoiceProbabilities(choices);
      contexts.push({ steering, choices });
    }
    return { template: name, namespace, steering, contexts };
  }

  private async fit(context: BrowserManifoldFittingContext): Promise<unknown> {
    if (context.request.args.length !== 3) {
      throw fittingError("INVALID_FIT_REQUEST", "Browser manifold fitting expects namespace, name, and options");
    }
    const [namespace, name, rawOptions] = context.request.args;
    if (typeof namespace !== "string" || typeof name !== "string") {
      throw fittingError("INVALID_FIT_REQUEST", "Browser manifold fitting requires a canonical identity");
    }
    const options = fitOptions(rawOptions);
    const hookProfile = resolveStructuredHookProfile(
      context.loadRequest.variant.structuredHookProfile,
    );
    context.signal.throwIfAborted();
    const detail = requireManifold(await this.artifacts.request(
      { service: "manifolds", method: "get", args: [namespace, name] },
      context.onProgress,
    ));
    const manifold = await this.resolveManifold(detail, options, context.onProgress);
    assertFitWithinProfile(manifold, hookProfile);
    const rowsAndOffsets = await this.captureRows(manifold, detail.template_ref, context.onProgress);
    const closure = fittedSourceClosure(detail, rowsAndOffsets.template);
    const sae = options.sae === undefined || options.sae === null
      ? null
      : await this.dependencies.loadExactSae(context.loadRequest, options.sae, context.runtime);
    if (options.sae !== undefined && options.sae !== null && sae === null) {
      throw fittingError(
        "BROWSER_SAE_FITTING_PORT_UNAVAILABLE",
        "The installed SAE runtime does not expose the exact centroid encode/decode port required for manifold fitting",
        501,
      );
    }
    const allWhiteners = await this.dependencies.loadWhiteners(context.loadRequest);
    const layers = sae === null
      ? resolveBrowserFitLayerIndices(
          options.layers,
          context.loadRequest.variant.runtimeIdentity.layerMap,
        )
      : saeFitLayers(options.layers, sae.layers, context.loadRequest.variant.runtimeIdentity.layerMap);
    const saeProvenance = sae === null
      ? null
      : selectedSaeProvenance(sae.provenance, layers, options.layers === undefined || options.layers === null);
    const whiteners = new Map(layers.map((layer) => {
      const whitener = allWhiteners.get(layer);
      if (!whitener) {
        throw fittingError(
          "CORE_PACK_WHITENER_UNAVAILABLE",
          `The required core pack has no neutral whitener for layer ${layer}`,
        );
      }
      return [layer, whitener] as const;
    }));
    const rendered = await prepareWebLlmActivationCaptureRows(
      context.runtime,
      rowsAndOffsets.rows,
      rowsAndOffsets.groupOffsets,
      context.signal,
    );
    const runtimeIdentitySha256 = context.loadRequest.variant.runtimeIdentitySha256;
    const contextBindingSha256 = requireContextBinding(context.loadRequest);
    const captureSha256 = digest({
      captureRenderSha256: rendered.captureRenderSha256,
      captureVersion: CAPTURE_VERSION,
      contextBindingSha256,
      runtimeIdentitySha256,
    });
    const hiddenSize = context.loadRequest.variant.runtimeIdentity.hiddenSize;
    const descriptor = {
      runtimeIdentitySha256,
      contextBindingSha256,
      captureSha256,
      layers: layers.map((layer) => ({
        layer,
        rows: rowsAndOffsets.rows.length,
        width: hiddenSize,
        expectedBytes: rowsAndOffsets.rows.length * hiddenSize * 4,
      })),
    };
    const source = new WebLlmActivationCaptureSource(context.runtime, {
      descriptor,
      rows: rowsAndOffsets.rows,
      preparedRows: rendered.preparedRows,
      layerMap: context.loadRequest.variant.runtimeIdentity.layerMap,
    });
    const coordinator = this.dependencies.createCoordinator(context.loadRequest);
    const progressOptions = {
      signal: context.signal,
      onProgress: (progress: BrowserFittingCoordinatorProgress) =>
        context.onProgress({ event: "progress", data: progress }),
    };
    const identity = {
      runtimeIdentitySha256,
      contextBindingSha256,
      modelSourceFingerprint: browserModelSourceFingerprint(context.loadRequest),
      captureSha256,
      captureVersion: CAPTURE_VERSION,
      captureRenderSha256: rendered.captureRenderSha256,
      baselinePromptsSha256: rowsAndOffsets.baselinePromptsSha256,
      fitPolicyVersion: FIT_POLICY_VERSION,
    };
    let archive: Blob;
    if (manifold.fitMode === "authored") {
      const coordinates = Float64Array.from(
        manifold.nodes.flatMap((node) => [...node.coords]),
      );
      const intrinsicDimensions = manifold.nodes[0].coords.length;
      const embeddedCoordinates = embedBrowserAuthoredCoordinates(
        manifold.domain,
        coordinates,
        manifold.nodes.length,
      );
      const authored = await coordinator.captureAuthoredFoundation({
        descriptor,
        groupOffsets: rowsAndOffsets.groupOffsets,
        whiteners,
        maxComponents: Math.min(hookProfile.maxRank, hiddenSize),
        transformCentroids: sae?.transformCentroids,
        domain: manifold.domain,
        coordinates,
        embeddedCoordinates,
        intrinsicDimensions,
        fitSigma: sae === null,
      }, source, progressOptions);
      archive = await this.dependencies.buildAuthoredPack({
        manifold,
        closure,
        modelId: context.loadRequest.variant.runtimeIdentity.sourceModel,
        producerVersion: this.producerVersion,
        drowseVersion: this.drowseVersion,
        identity,
        evaluatedLayers: authored.layers,
        layers: authored.curvedSurfaceLayers,
        sae: saeProvenance,
      });
    } else {
      const topology = manifold.nodes.length === 1
        ? monopolarFoundation(
            transformCapturedCentroids(
              await coordinator.captureNodeCentroids({
                descriptor,
                groupOffsets: rowsAndOffsets.groupOffsets,
              }, source, progressOptions),
              sae?.transformCentroids,
            ),
            whiteners,
          )
        : await coordinator.captureTopologyFoundation({
            descriptor,
            groupOffsets: rowsAndOffsets.groupOffsets,
            whiteners,
            transformCentroids: sae?.transformCentroids,
            fitSigma: sae === null,
            fitMode: manifold.fitMode,
            maxComponents: integerHyperparameter(
              manifold.hyperparams.max_subspace_dim,
              Math.min(hookProfile.maxRank, hiddenSize),
              hookProfile.maxRank,
            ),
            maxDimensions: integerHyperparameter(
              manifold.hyperparams.max_dim,
              hookProfile.maxIntrinsicDim,
              hookProfile.maxIntrinsicDim,
            ),
            minDimensions: optionalInteger(manifold.hyperparams.min_dim),
            varianceThreshold: numberHyperparameter(manifold.hyperparams.var_threshold, 0.7),
            kNn: optionalInteger(manifold.hyperparams.k_nn),
            bandwidth: optionalNumber(manifold.hyperparams.bandwidth),
            persistenceFraction: numberHyperparameter(manifold.hyperparams.persistence_frac, 0.5),
            smoothing: smoothingHyperparameter(manifold.hyperparams.smoothing),
          }, source, progressOptions);
      const common = {
        manifold,
        closure,
        modelId: context.loadRequest.variant.runtimeIdentity.sourceModel,
        producerVersion: this.producerVersion,
        drowseVersion: this.drowseVersion,
        identity,
        topology: topology.topology,
        consensusGram: topology.consensusGram,
        evaluatedLayers: topology.layers,
        sae: saeProvenance,
      };
      if (topology.topology.fitMode === "pca") {
        if (topology.finalAffineLayers === null || topology.anchoredNodeCoordinates === null) {
          throw fittingError("BROWSER_FIT_RESULT_INVALID", "Browser PCA fitting returned incomplete geometry");
        }
        archive = await this.dependencies.buildFlatPack({
          ...common,
          nodeCoordinates: topology.anchoredNodeCoordinates,
          layers: topology.finalAffineLayers,
        });
      } else {
        if (topology.curvedSurfaceLayers === null) {
          throw fittingError("BROWSER_FIT_RESULT_INVALID", "Browser spectral fitting returned incomplete geometry");
        }
        archive = await this.dependencies.buildCurvedPack({
          ...common,
          layers: topology.curvedSurfaceLayers,
        });
      }
    }
    context.signal.throwIfAborted();
    await this.dependencies.validateFittedPack(context.loadRequest, archive);
    context.signal.throwIfAborted();
    await this.artifacts.request(
      {
        service: "manifolds",
        method: "drowseArchiveInstall",
        args: [archive, { force: true }],
      },
      context.onProgress,
    );
    context.signal.throwIfAborted();
    return this.artifacts.request(
      { service: "manifolds", method: "get", args: [namespace, name] },
      context.onProgress,
    );
  }

  private async generate(context: BrowserManifoldFittingContext): Promise<unknown> {
    if (context.request.args.length !== 1) {
      throw fittingError("INVALID_GENERATE_REQUEST", "Browser manifold generation expects one request body");
    }
    const request = generateOptions(context.request.args[0]);
    this.assertDiscoverModeAllowed(request.fitMode);
    const hookProfile = resolveStructuredHookProfile(
      context.loadRequest.variant.structuredHookProfile,
    );
    const existing = request.force
      ? null
      : await this.installedManifold(request.namespace, request.name, context.onProgress);
    if (existing && (!existing.is_discover || !existing.nodes || existing.template_ref !== null)) {
      throw fittingError(
        "BROWSER_GENERATE_TARGET_INVALID",
        `Manifold ${request.namespace}/${request.name} is not a resumable generated manifold`,
      );
    }
    const nodes = existing
      ? existing.nodes!.map((node, index) => ({
          label: node.label,
          statements: [...node.statements],
          role: node.role,
          kind: existing.node_kinds[index] ?? null,
        }))
      : [];
    const existingLabels = new Set(nodes.map((node) => node.label));
    const fitMode = existing ? requireDiscoverFitMode(existing.fit_mode) : request.fitMode;
    const hyperparams = existing
      ? sanitizeHyperparameters(fitMode, existing.hyperparams)
      : sanitizeHyperparameters(fitMode, request.hyperparams);
    assertDiscoverHyperparametersWithinProfile(hyperparams, hookProfile);
    let published = false;
    for (const concept of request.concepts) {
      if (existingLabels.has(concept)) continue;
      context.signal.throwIfAborted();
      const role = request.rolePerNode ? concept : null;
      const statements = await this.generateCorpus(
        concept,
        request.kind,
        request.customSystem,
        request.rolePerNode ? concept : undefined,
        request.samplesPerPrompt,
        context,
      );
      nodes.push({ label: concept, statements, role, kind: request.kind });
      existingLabels.add(concept);
      await this.publishDiscover({
        namespace: request.namespace,
        name: request.name,
        description: request.description,
        fitMode,
        hyperparams,
        nodes,
      }, context);
      published = true;
    }
    if (!published && (!existing || existing.description !== request.description)) {
      await this.publishDiscover({
        namespace: request.namespace,
        name: request.name,
        description: request.description,
        fitMode,
        hyperparams,
        nodes,
      }, context);
    }
    context.signal.throwIfAborted();
    return this.artifacts.request(
      { service: "manifolds", method: "get", args: [request.namespace, request.name] },
      context.onProgress,
    );
  }

  private async extract(context: BrowserManifoldFittingContext): Promise<unknown> {
    if (
      context.request.args.length < 1 || context.request.args.length > 2 ||
      context.request.args[1] !== undefined && context.request.args[1] !== "default"
    ) {
      throw fittingError("INVALID_EXTRACT_REQUEST", "Browser extraction uses the default local session");
    }
    const request = extractOptions(context.request.args[0]);
    if (request.sae !== null && request.role !== null) {
      throw fittingError(
        "INVALID_EXTRACT_REQUEST",
        "SAE-backed and role-baselined extraction are mutually exclusive",
        400,
      );
    }
    const exactSae = request.sae === null
      ? null
      : await this.dependencies.loadExactSae(context.loadRequest, request.sae, context.runtime);
    if (request.sae !== null && exactSae === null) {
      throw fittingError(
        "BROWSER_SAE_FITTING_PORT_UNAVAILABLE",
        "The selected installed SAE does not expose the exact centroid reconstruction required for extraction",
        501,
      );
    }
    const [positive, negative] = splitCompositeSource(request.concept, request.baseline);
    const positiveLabel = conceptSlug(positive);
    const negativeLabel = negative === null ? null : conceptSlug(negative);
    if (!LABEL.test(positiveLabel) || negativeLabel !== null && !LABEL.test(negativeLabel)) {
      throw fittingError(
        "INVALID_EXTRACT_REQUEST",
        "Extracted concept poles must produce grammar-addressable lowercase labels",
        400,
      );
    }
    const canonical = negativeLabel === null
      ? positiveLabel
      : `${positiveLabel}.${negativeLabel}`;
    const storedName = exactSae !== null
      ? `${canonical}:sae-${exactSae.provenance.release}`
      : request.role === null ? canonical : `${canonical}:role-${request.role}`;
    const registryName = request.explicitNamespace
      ? `${request.namespace}/${storedName}`
      : storedName;
    const existing = request.force
      ? null
      : await this.installedManifold(request.namespace, canonical, context.onProgress);
    if (existing) {
      if (
        !existing.is_discover || existing.fit_mode !== "pca" || !existing.nodes ||
        existing.node_roles.length === 0 ||
        existing.node_roles.some((role) => role !== request.role)
      ) {
        throw fittingError(
          "EXTRACT_BASELINE_MISMATCH",
          `Manifold ${request.namespace}/${canonical} does not use the requested role baseline; retry with force`,
        );
      }
      const cached = await this.profileForInstalled(
        request.namespace,
        canonical,
        storedName,
        registryName,
        context,
      );
      if (cached !== null) return { done: true, canonical: registryName, profile: cached };
    } else {
      const concepts = negative === null ? [positive] : [positive, negative];
      const labels = negativeLabel === null ? [positiveLabel] : [positiveLabel, negativeLabel];
      const nodes: Array<BrowserMergedDiscoverManifold["nodes"][number]> = [];
      for (let index = 0; index < concepts.length; index += 1) {
        const concept = concepts[index];
        const statements = await this.generateCorpus(
          concept,
          request.kind,
          request.customSystem,
          request.role ?? undefined,
          1,
          context,
        );
        nodes.push({
          label: labels[index],
          statements,
          role: request.role,
          kind: request.kind,
        });
      }
      await this.publishDiscover({
        namespace: request.namespace,
        name: canonical,
        description: negative === null
          ? `Monopolar axis: ${positive} (+) vs neutral baseline (-).`
          : `Bipolar axis: ${positive} (+) vs ${negative} (-).`,
        fitMode: "pca",
        hyperparams: { max_dim: 1, var_threshold: 0.7 },
        nodes,
      }, context);
    }
    await this.fit({
      ...context,
      request: {
        service: "manifolds",
        method: "fit",
        args: [request.namespace, canonical, {
          sae: request.sae,
          layers: request.sae === null ? "all" : null,
          force: request.force,
          fit_mode: "pca",
          hyperparams: { max_dim: 1, var_threshold: 0.7 },
        }],
      },
    });
    const profile = await this.profileForInstalled(
      request.namespace,
      canonical,
      storedName,
      registryName,
      context,
    );
    if (profile === null) {
      throw fittingError(
        "BROWSER_FIT_RESULT_INVALID",
        "Browser extraction did not publish a compatible fitted profile",
        500,
      );
    }
    return { done: true, canonical: registryName, profile };
  }

  private async generateCorpus(
    concept: string,
    kind: "abstract" | "concrete" | "custom",
    customSystem: string | null,
    explicitRole: string | undefined,
    samplesPerPrompt: number,
    context: BrowserManifoldFittingContext,
  ): Promise<string[]> {
    const conceptHuman = concept.replaceAll("_", " ");
    const system = `${LENGTH_DIRECTIVE} ${conceptSystem(conceptHuman, kind, customSystem)}`;
    const generationRoleName = explicitRole ?? elicitationRole(concept, kind);
    const prompts = requireBaselinePrompts();
    const total = samplesPerPrompt * prompts.length;
    const responses: string[] = [];
    for (let sample = 0; sample < samplesPerPrompt; sample += 1) {
      for (const prompt of prompts) {
        context.signal.throwIfAborted();
        const position = responses.length + 1;
        context.onProgress({
          event: "progress",
          data: { message: `Generating ${JSON.stringify(concept)} response ${position}/${total}...` },
        });
        const result = await context.runtime.streamGeneration({
          input: {
            kind: "chat",
            messages: [
              { role: "system", content: system },
              { role: "user", content: prompt },
            ],
          },
          sampling: {
            temperature: 1,
            top_p: 0.9,
            max_tokens: RESPONSE_MAX_TOKENS,
          },
          thinking: false,
          generationRoleName,
          signal: context.signal,
        }, () => undefined);
        responses.push(result.text.trim());
      }
    }
    return responses;
  }

  private async publishDiscover(
    manifold: BrowserMergedDiscoverManifold,
    context: BrowserManifoldFittingContext,
  ): Promise<void> {
    context.signal.throwIfAborted();
    const archive = await this.dependencies.buildDiscoverPack(manifold, this.producerVersion);
    context.signal.throwIfAborted();
    await this.artifacts.request({
      service: "manifolds",
      method: "drowseArchiveInstall",
      args: [archive, { force: true }],
    }, context.onProgress);
  }

  private async installedManifold(
    namespace: string,
    name: string,
    onProgress: (event: RuntimeProgressEvent) => void,
  ): Promise<ManifoldInfo | null> {
    const value = await this.artifacts.request(
      { service: "manifolds", method: "list", args: [] },
      onProgress,
    );
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw fittingError("BROWSER_ARTIFACT_INDEX_INVALID", "The browser manifold index is invalid", 500);
    }
    const manifolds = (value as { manifolds?: unknown }).manifolds;
    if (!Array.isArray(manifolds)) {
      throw fittingError("BROWSER_ARTIFACT_INDEX_INVALID", "The browser manifold index is invalid", 500);
    }
    if (!manifolds.some((item) =>
      item && typeof item === "object" && !Array.isArray(item) &&
      (item as Record<string, unknown>).namespace === namespace &&
      (item as Record<string, unknown>).name === name
    )) return null;
    return requireManifold(await this.artifacts.request(
      { service: "manifolds", method: "get", args: [namespace, name] },
      onProgress,
    ));
  }

  private async profileForInstalled(
    namespace: string,
    name: string,
    storedName: string,
    registryName: string,
    context: BrowserManifoldFittingContext,
  ): Promise<VectorInfo | null> {
    const archive = await this.artifacts.request({
      service: "manifolds",
      method: "drowseArchiveExport",
      args: [`manifolds/${namespace}/${name}`],
    }, context.onProgress);
    if (!(archive instanceof Blob)) {
      throw fittingError("BROWSER_ARTIFACT_INVALID", "The browser manifold archive is invalid", 500);
    }
    const verified = await validateDrowseArchive(archive, { signal: context.signal });
    const saeRelease = storedName.startsWith(`${name}:sae-`)
      ? storedName.slice(`${name}:sae-`.length)
      : null;
    const fitted = verified.fittedArtifacts.find((artifact) =>
      (saeRelease === null
        ? artifact.variant === "raw" && artifact.variantIdentity === null
        : artifact.variant === "sae" && artifact.variantIdentity === saeRelease) &&
      artifact.modelId === context.loadRequest.variant.runtimeIdentity.sourceModel &&
      artifact.modelFingerprint === context.loadRequest.variant.runtimeIdentitySha256
    );
    if (!fitted) return null;
    const sidecarBytes = await readVerifiedDrowseArchiveFile(
      verified,
      fitted.sidecarPath,
      context.signal,
    );
    const sidecar = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(sidecarBytes)) as {
      fitted_layers?: unknown;
      share_metric?: unknown;
    };
    if (
      !Array.isArray(sidecar.fitted_layers) || sidecar.fitted_layers.length === 0 ||
      sidecar.fitted_layers.some((layer) => !Number.isSafeInteger(layer) || layer < 0) ||
      sidecar.share_metric !== "mahalanobis"
    ) {
      throw fittingError("BROWSER_FIT_RESULT_INVALID", "The fitted browser profile sidecar is invalid", 500);
    }
    return {
      name: registryName,
      layers: [...sidecar.fitted_layers as number[]],
      metadata: {
        method: "manifold_pca",
        name: storedName,
        share_metric: "mahalanobis",
      },
    };
  }

  private async resolveManifold(
    detail: ManifoldInfo,
    options: FitManifoldRequest,
    _onProgress: (event: RuntimeProgressEvent) => void,
  ): Promise<BrowserFittableManifold> {
    if (!detail.nodes || detail.nodes.length === 0) {
      throw fittingError(
        "BROWSER_FIT_MODE_UNAVAILABLE",
        "Hosted fitting requires a non-empty manifold",
      );
    }
    if (!detail.is_discover) {
      if (
        detail.fit_mode !== "authored" || options.fit_mode != null ||
        options.hyperparams != null && Object.keys(options.hyperparams).length > 0 ||
        detail.nodes.length < 3 || detail.nodes.some((node) =>
          node.coords === null || node.coords.length !== detail.intrinsic_dim ||
          node.coords.some((value) => !Number.isFinite(value))
        )
      ) {
        throw fittingError(
          "INVALID_FIT_REQUEST",
          "Browser authored fitting consumes its stored domain and coordinates without fit-mode or hyperparameter overrides",
        );
      }
      return {
        namespace: detail.namespace,
        name: detail.name,
        description: detail.description,
        fitMode: "authored",
        domain: structuredClone(detail.domain),
        nodes: detail.nodes.map((node, index) => ({
          label: node.label,
          coords: [...node.coords!],
          statements: [...node.statements],
          role: node.role,
          kind: detail.node_kinds[index] ?? null,
        })),
      };
    }
    const fitMode = options.fit_mode ?? detail.fit_mode;
    if (fitMode !== "pca" && fitMode !== "spectral" && fitMode !== "auto") {
      throw fittingError("INVALID_FIT_REQUEST", "Browser discover fitting requires pca, spectral, or auto mode");
    }
    if (detail.nodes.length === 1 && fitMode !== "pca") {
      throw fittingError(
        "BROWSER_FIT_MODE_UNAVAILABLE",
        "A one-node browser manifold is a monopolar PCA axis against the neutral baseline",
      );
    }
    this.assertDiscoverModeAllowed(fitMode);
    return {
      namespace: detail.namespace,
      name: detail.name,
      description: detail.description,
      fitMode,
      hyperparams: sanitizeHyperparameters(
        fitMode,
        { ...detail.hyperparams, ...(options.hyperparams ?? {}) },
      ),
      nodes: detail.nodes.map((node, index) => ({
        label: node.label,
        statements: [...node.statements],
        role: node.role,
        kind: detail.node_kinds[index] ?? null,
      })),
    };
  }

  private assertDiscoverModeAllowed(fitMode: "pca" | "spectral" | "auto"): void {
    if (fitMode === "pca" || this.dependencies.allowAutomaticTopologyDiscovery) return;
    throw fittingError(
      "BROWSER_AUTOMATIC_TOPOLOGY_UNAVAILABLE",
      "Hosted production supports linear PCA fits and explicitly authored geometries. Automatic curved-topology detection is not release-enabled.",
      501,
    );
  }

  private async captureRows(
    manifold: BrowserFittableManifold,
    templateRef: string | null,
    onProgress: (event: RuntimeProgressEvent) => void,
  ): Promise<{
    rows: DrowseCaptureRow[];
    groupOffsets: Uint32Array;
    baselinePromptsSha256: string;
    template: TemplateDetail | null;
  }> {
    const template = templateRef === null
      ? null
      : requireTemplate(await this.artifacts.request(
          { service: "templates", method: "get", args: templateRef.split("/") },
          onProgress,
        ));
    const prompts: unknown[] = template === null
      ? baselinePrompts
      : template.contexts.map((context) => context.turns);
    if (prompts.length === 0) {
      throw fittingError("BROWSER_CAPTURE_INPUT_INVALID", "The manifold has no baseline capture prompts");
    }
    const rows: DrowseCaptureRow[] = [];
    const groupOffsets = new Uint32Array(manifold.nodes.length + 1);
    manifold.nodes.forEach((node, nodeIndex) => {
      if (node.statements.length === 0) {
        throw fittingError(
          "BROWSER_CAPTURE_INPUT_INVALID",
          `Manifold node ${node.label} has no capture statements`,
        );
      }
      node.statements.forEach((response, rowIndex) => {
        const prompt = prompts[rowIndex % prompts.length];
        rows.push(captureRow(prompt, response, node.role));
      });
      groupOffsets[nodeIndex + 1] = rows.length;
    });
    return {
      rows,
      groupOffsets,
      baselinePromptsSha256: digest(prompts),
      template,
    };
  }
}

interface BrowserGenerateOptions {
  namespace: string;
  name: string;
  description: string;
  concepts: string[];
  kind: "abstract" | "concrete" | "custom";
  customSystem: string | null;
  samplesPerPrompt: number;
  fitMode: "pca" | "spectral" | "auto";
  hyperparams: Record<string, number | string>;
  force: boolean;
  rolePerNode: boolean;
}

interface BrowserExtractOptions {
  concept: string;
  baseline: string | null;
  kind: "abstract" | "concrete" | "custom";
  customSystem: string | null;
  sae: string | null;
  role: string | null;
  namespace: string;
  explicitNamespace: boolean;
  force: boolean;
}

function generateOptions(value: unknown): BrowserGenerateOptions {
  const body = objectRequest(value, "generate") as unknown as GenerateManifoldRequest;
  const allowed = new Set([
    "namespace", "name", "description", "concepts", "kind", "custom_system",
    "samples_per_prompt", "fit_mode", "hyperparams", "force", "role_per_node",
  ]);
  rejectUnknown(body, allowed, "generate");
  const namespace = body.namespace ?? DEFAULT_NAMESPACE;
  requireIdentity(namespace, body.name, "generate");
  if (
    !Array.isArray(body.concepts) || body.concepts.length < 2 ||
    body.concepts.some((concept) => typeof concept !== "string" || !LABEL.test(concept)) ||
    new Set(body.concepts).size !== body.concepts.length
  ) {
    throw fittingError(
      "INVALID_GENERATE_REQUEST",
      "Browser manifold generation requires at least two unique grammar-addressable concept labels",
      400,
    );
  }
  const kind = body.kind ?? "abstract";
  if (kind !== "abstract" && kind !== "concrete" && kind !== "custom") {
    throw fittingError("INVALID_GENERATE_REQUEST", "Browser manifold generation kind is invalid", 400);
  }
  const customSystem = body.custom_system ?? null;
  if (customSystem !== null && typeof customSystem !== "string") {
    throw fittingError("INVALID_GENERATE_REQUEST", "Browser custom system prompt is invalid", 400);
  }
  if (kind === "custom" && !customSystem) {
    throw fittingError(
      "INVALID_GENERATE_REQUEST",
      "kind='custom' requires custom_system (a system template with a {c} placeholder)",
      400,
    );
  }
  if (
    body.samples_per_prompt !== undefined &&
    !Number.isSafeInteger(body.samples_per_prompt)
  ) {
    throw fittingError("INVALID_GENERATE_REQUEST", "samples_per_prompt must be an integer", 400);
  }
  const fitMode = body.fit_mode ?? "pca";
  if (fitMode !== "pca" && fitMode !== "spectral" && fitMode !== "auto") {
    throw fittingError("INVALID_GENERATE_REQUEST", "Browser manifold generation fit mode is invalid", 400);
  }
  if (body.description !== undefined && typeof body.description !== "string") {
    throw fittingError("INVALID_GENERATE_REQUEST", "Browser manifold description is invalid", 400);
  }
  if (body.force !== undefined && typeof body.force !== "boolean") {
    throw fittingError("INVALID_GENERATE_REQUEST", "Browser manifold force flag is invalid", 400);
  }
  if (body.role_per_node !== undefined && typeof body.role_per_node !== "boolean") {
    throw fittingError("INVALID_GENERATE_REQUEST", "Browser role-per-node flag is invalid", 400);
  }
  const rawHyperparams = body.hyperparams ?? {};
  if (!rawHyperparams || typeof rawHyperparams !== "object" || Array.isArray(rawHyperparams)) {
    throw fittingError("INVALID_GENERATE_REQUEST", "Browser manifold hyperparameters are invalid", 400);
  }
  return {
    namespace,
    name: body.name,
    description: body.description ?? "",
    concepts: [...body.concepts],
    kind,
    customSystem,
    samplesPerPrompt: Math.max(1, body.samples_per_prompt ?? 1),
    fitMode,
    hyperparams: sanitizeHyperparameters(
      fitMode,
      rawHyperparams as Record<string, number | string>,
    ),
    force: body.force === true,
    rolePerNode: body.role_per_node === true,
  };
}

function extractOptions(value: unknown): BrowserExtractOptions {
  const body = objectRequest(value, "extract") as unknown as ExtractRequest;
  const allowed = new Set([
    "concept", "baseline", "kind", "custom_system", "sae", "role", "namespace", "force",
  ]);
  rejectUnknown(body, allowed, "extract");
  if (typeof body.concept !== "string" || !body.concept.trim()) {
    throw fittingError("INVALID_EXTRACT_REQUEST", "Browser extraction requires a concept", 400);
  }
  if (body.baseline !== undefined && body.baseline !== null && typeof body.baseline !== "string") {
    throw fittingError("INVALID_EXTRACT_REQUEST", "Browser extraction baseline is invalid", 400);
  }
  const kind = body.kind ?? "abstract";
  if (kind !== "abstract" && kind !== "concrete" && kind !== "custom") {
    throw fittingError("INVALID_EXTRACT_REQUEST", "Browser extraction kind is invalid", 400);
  }
  const customSystem = body.custom_system ?? null;
  if (customSystem !== null && typeof customSystem !== "string") {
    throw fittingError("INVALID_EXTRACT_REQUEST", "Browser custom system prompt is invalid", 400);
  }
  if (kind === "custom" && !customSystem) {
    throw fittingError(
      "INVALID_EXTRACT_REQUEST",
      "kind='custom' requires custom_system (a system template with a {c} placeholder)",
      400,
    );
  }
  const sae = body.sae ?? null;
  if (sae !== null && typeof sae !== "string") {
    throw fittingError("INVALID_EXTRACT_REQUEST", "Browser extraction SAE selector is invalid", 400);
  }
  const role = body.role ?? null;
  if (role !== null && (typeof role !== "string" || !ROLE.test(role))) {
    throw fittingError("INVALID_EXTRACT_REQUEST", "Browser extraction role is invalid", 400);
  }
  const explicitNamespace = body.namespace !== undefined && body.namespace !== null;
  const namespace = body.namespace ?? DEFAULT_NAMESPACE;
  requireIdentity(namespace, "placeholder", "extract", false);
  if (body.force !== undefined && typeof body.force !== "boolean") {
    throw fittingError("INVALID_EXTRACT_REQUEST", "Browser extraction force flag is invalid", 400);
  }
  return {
    concept: body.concept,
    baseline: body.baseline ?? null,
    kind,
    customSystem,
    sae,
    role,
    namespace,
    explicitNamespace,
    force: body.force === true,
  };
}

function objectRequest(value: unknown, operation: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw fittingError(
      `INVALID_${operation.toUpperCase()}_REQUEST`,
      `Browser ${operation} request body is invalid`,
      400,
    );
  }
  return value as Record<string, unknown>;
}

function rejectUnknown(
  value: object,
  allowed: ReadonlySet<string>,
  operation: string,
): void {
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length > 0) {
    throw fittingError(
      `INVALID_${operation.toUpperCase()}_REQUEST`,
      `Browser ${operation} request contains unknown field(s): ${unknown.sort().join(", ")}`,
      400,
    );
  }
}

function requireIdentity(
  namespace: unknown,
  name: unknown,
  operation: string,
  validateName = true,
): void {
  if (
    typeof namespace !== "string" || !NAME.test(namespace) ||
    validateName && (typeof name !== "string" || !NAME.test(name)) ||
    namespace === "jlens" || namespace === "sae"
  ) {
    throw fittingError(
      `INVALID_${operation.toUpperCase()}_REQUEST`,
      `Browser ${operation} requires canonical, non-reserved manifold identity slugs`,
      400,
    );
  }
}

function requireDiscoverFitMode(value: unknown): "pca" | "spectral" | "auto" {
  if (value !== "pca" && value !== "spectral" && value !== "auto") {
    throw fittingError("BROWSER_GENERATE_TARGET_INVALID", "Existing manifold is not discover-mode");
  }
  return value;
}

function splitCompositeSource(concept: string, baseline: string | null): [string, string | null] {
  if (baseline === null && concept.includes(".")) {
    const separator = concept.indexOf(".");
    return [concept.slice(0, separator).trim(), concept.slice(separator + 1).trim()];
  }
  return [concept, baseline];
}

function conceptSlug(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

function conceptSystem(
  concept: string,
  kind: "abstract" | "concrete" | "custom",
  customSystem: string | null,
): string {
  const article = /^[aeiou]/i.test(concept) ? "an" : "a";
  if (kind === "abstract") {
    return `You are someone ${concept}. Respond exactly as someone ${concept} would.`;
  }
  if (kind === "concrete") {
    return `You are ${article} ${concept}. Respond exactly as ${article} ${concept} would.`;
  }
  if (!customSystem) {
    throw fittingError("INVALID_GENERATE_REQUEST", "Custom generation requires a system template", 400);
  }
  return formatCustomSystem(customSystem, { c: concept, art: article });
}

function formatCustomSystem(
  template: string,
  values: { c: string; art: string },
): string {
  let output = "";
  for (let index = 0; index < template.length;) {
    const char = template[index];
    if (char === "{") {
      if (template[index + 1] === "{") {
        output += "{";
        index += 2;
        continue;
      }
      const end = template.indexOf("}", index + 1);
      if (end < 0) throw fittingError("INVALID_GENERATE_REQUEST", "Custom system template has an unmatched '{'", 400);
      const field = template.slice(index + 1, end);
      if (field !== "c" && field !== "art") {
        throw fittingError(
          "INVALID_GENERATE_REQUEST",
          `Custom system template contains unsupported field {${field}}`,
          400,
        );
      }
      output += values[field];
      index = end + 1;
      continue;
    }
    if (char === "}") {
      if (template[index + 1] === "}") {
        output += "}";
        index += 2;
        continue;
      }
      throw fittingError("INVALID_GENERATE_REQUEST", "Custom system template has an unmatched '}'", 400);
    }
    output += char;
    index += 1;
  }
  return output;
}

function elicitationRole(
  concept: string,
  kind: "abstract" | "concrete" | "custom",
): string | undefined {
  if (kind === "custom") return undefined;
  const slug = conceptSlug(concept);
  return kind === "abstract" ? `someone_${slug}` : slug;
}

function requireBaselinePrompts(): string[] {
  if (!Array.isArray(baselinePrompts) || baselinePrompts.length === 0 ||
    baselinePrompts.some((prompt) => typeof prompt !== "string" || !prompt)) {
    throw fittingError("BROWSER_CAPTURE_INPUT_INVALID", "The shared baseline prompt corpus is invalid", 500);
  }
  return [...baselinePrompts];
}

function monopolarFoundation(
  centroids: BrowserFittingCentroids,
  whiteners: ReadonlyMap<number, SerializedMahalanobisWhitener>,
): BrowserFittingTopologyFoundation {
  const layers = new Map<number, FinalizedBrowserAffineLayer>();
  for (const [layer, pooled] of centroids.layers) {
    const whitener = whiteners.get(layer);
    if (!whitener || pooled.rows !== 1 || pooled.columns !== whitener.columns) {
      throw fittingError(
        "BROWSER_FIT_RESULT_INVALID",
        `Browser monopolar centroid for layer ${layer} does not match its neutral whitener`,
        500,
      );
    }
    const direction = Float64Array.from(
      pooled.values,
      (value, index) => value - whitener.mean[index],
    );
    const norm = Math.hypot(...direction);
    if (norm <= 1e-12) continue;
    const basis = Float64Array.from(direction, (value) => value / norm);
    let neutralCoordinate = 0;
    for (let index = 0; index < pooled.columns; index += 1) {
      neutralCoordinate += whitener.mean[index] * basis[index];
    }
    const mean = Float64Array.from(basis, (value) => value * neutralCoordinate);
    const share = mahalanobisNorm(whitener, direction);
    if (!Number.isFinite(share) || share <= 0) continue;
    layers.set(layer, {
      operation: "affine_fisher",
      nodeCount: 1,
      columns: pooled.columns,
      components: 1,
      centroidMean: pooled.values.slice(),
      mean,
      basis,
      nodeCoordinates: new Float64Array([norm]),
      muCoordinates: new Float64Array([0]),
      whitenedGram: new Float64Array([share * share]),
      neutralCrossGram: new Float64Array([0]),
      explainedVariance: 1,
      mahalanobisShare: share,
      affineMap: null,
    });
  }
  if (layers.size === 0) {
    throw fittingError(
      "BROWSER_MONOPOLAR_DEGENERATE",
      "The concept activation is indistinguishable from the neutral baseline on every fitted layer",
    );
  }
  const topology: TopologyResult = {
    operation: "topology",
    winnerName: "monopolar-neutral-ray",
    fitMode: "pca",
    intrinsicDimensions: 1,
    periodicDimensions: 0,
    persistentLoops: 0,
    usedFaintCycle: false,
    coordinates: new Float64Array([1]),
    embeddedCoordinates: new Float64Array([1]),
    candidates: [],
    diagnostics: {
      kind: "pca",
      perComponentVariance: new Float64Array([1]),
      cumulativeVariance: new Float64Array([1]),
      pickedDimensions: 1,
      threshold: 0.7,
    },
    winnerPlan: null,
  };
  return {
    identity: centroids.identity,
    layers,
    captureRetained: centroids.captureRetained,
    consensusGram: new Float64Array([0]),
    topology,
    dlsKept: null,
    neutralLayoutCoordinate: new Float64Array([0]),
    anchoredNodeCoordinates: new Float64Array([1]),
    finalAffineLayers: layers,
    curvedSurfaceLayers: null,
  };
}

function captureRow(prompt: unknown, response: string, roleName: string | null): DrowseCaptureRow {
  const messages: DrowseCaptureMessage[] = [];
  if (typeof prompt === "string") {
    messages.push({ role: "user", content: prompt });
  } else if (Array.isArray(prompt)) {
    for (const turn of prompt) {
      if (!turn || typeof turn !== "object" || Array.isArray(turn)) {
        throw fittingError("BROWSER_CAPTURE_INPUT_INVALID", "A template capture turn is invalid");
      }
      const value = turn as Record<string, unknown>;
      if (typeof value.content !== "string") {
        throw fittingError("BROWSER_CAPTURE_INPUT_INVALID", "A template capture turn has invalid content");
      }
      if (value.role === "system" || value.role === "user" || value.role === "assistant") {
        messages.push({ role: value.role, content: value.content });
      } else {
        throw fittingError("BROWSER_CAPTURE_INPUT_INVALID", "A template capture turn has an unsupported role");
      }
    }
  } else {
    throw fittingError("BROWSER_CAPTURE_INPUT_INVALID", "A manifold capture prompt is invalid");
  }
  messages.push({
    role: "assistant",
    content: response,
    ...(roleName === null ? {} : { roleName }),
  });
  return { system: LENGTH_DIRECTIVE, messages };
}

function fitOptions(value: unknown): FitManifoldRequest {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw fittingError("INVALID_FIT_REQUEST", "Browser manifold fit options are invalid");
  }
  const options = value as Record<string, unknown>;
  const allowed = new Set(["sae", "layers", "force", "fit_mode", "hyperparams"]);
  const unknown = Object.keys(options).filter((key) => !allowed.has(key));
  if (unknown.length > 0) {
    throw fittingError(
      "INVALID_FIT_REQUEST",
      `Browser manifold fit options contain unknown field(s): ${unknown.sort().join(", ")}`,
    );
  }
  if (options.force !== undefined && typeof options.force !== "boolean") {
    throw fittingError("INVALID_FIT_REQUEST", "Browser manifold fit force must be boolean");
  }
  if (
    options.sae !== undefined && options.sae !== null &&
    (typeof options.sae !== "string" || options.sae.trim().length === 0)
  ) {
    throw fittingError("INVALID_FIT_REQUEST", "Browser manifold fit SAE selector is invalid");
  }
  if (
    options.fit_mode !== undefined && options.fit_mode !== null &&
    options.fit_mode !== "pca" && options.fit_mode !== "spectral" && options.fit_mode !== "auto"
  ) {
    throw fittingError("INVALID_FIT_REQUEST", "Browser manifold fit mode is invalid");
  }
  if (
    options.hyperparams !== undefined && options.hyperparams !== null &&
    (typeof options.hyperparams !== "object" || Array.isArray(options.hyperparams))
  ) {
    throw fittingError("INVALID_FIT_REQUEST", "Browser manifold fit hyperparameters are invalid");
  }
  return structuredClone(options as FitManifoldRequest);
}

function sanitizeHyperparameters(
  fitMode: "pca" | "spectral" | "auto",
  hyperparameters: Record<string, number | string>,
): Record<string, number | string> {
  const allowedByMode = {
    pca: new Set(["max_dim", "var_threshold"]),
    spectral: new Set(["max_dim", "min_dim", "k_nn", "bandwidth", "max_subspace_dim", "smoothing"]),
    auto: new Set([
      "max_dim", "var_threshold", "min_dim", "k_nn", "bandwidth",
      "max_subspace_dim", "smoothing", "persistence_frac",
    ]),
  } as const;
  const invalid = Object.keys(hyperparameters).filter((key) => !allowedByMode[fitMode].has(key));
  if (invalid.length > 0) {
    throw fittingError(
      "INVALID_FIT_REQUEST",
      `Browser ${fitMode} fitting does not accept hyperparameter(s): ${invalid.sort().join(", ")}`,
    );
  }
  const integerKeys = new Set(["max_dim", "min_dim", "k_nn", "max_subspace_dim"]);
  const numberKeys = new Set(["var_threshold", "bandwidth", "persistence_frac"]);
  for (const [key, value] of Object.entries(hyperparameters)) {
    const valid = integerKeys.has(key)
      ? Number.isSafeInteger(value) && (value as number) > 0
      : numberKeys.has(key)
        ? typeof value === "number" && Number.isFinite(value)
        : key === "smoothing" && (
            value === "auto" || typeof value === "number" && Number.isFinite(value)
          );
    if (!valid) {
      throw fittingError(
        "INVALID_FIT_REQUEST",
        `Browser ${fitMode} fitting hyperparameter ${key} is invalid`,
      );
    }
  }
  return { ...hyperparameters };
}

export function resolveBrowserFitLayerIndices(
  selection: FitManifoldRequest["layers"],
  layerMap: readonly number[],
): number[] {
  if (selection === undefined || selection === null || selection === "all") return [...layerMap];
  if (selection === "workspace") {
    const layerCount = Math.max(...layerMap) + 1;
    const denominator = Math.max(layerCount - 1, 1);
    const workspace = layerMap.filter((layer) => {
      const depth = layer / denominator;
      return depth >= 0.40 && depth <= 0.90;
    });
    return workspace.length > 0 ? workspace : [...layerMap];
  }
  if (
    !Array.isArray(selection) || selection.length === 0 ||
    new Set(selection).size !== selection.length ||
    selection.some((layer) => !layerMap.includes(layer))
  ) {
    throw fittingError("INVALID_FIT_REQUEST", "Browser fit layers are outside the compiled layer map");
  }
  return [...selection].sort((left, right) => left - right);
}

function saeFitLayers(
  requested: FitManifoldRequest["layers"],
  saeLayers: readonly number[],
  runtimeLayers: readonly number[],
): number[] {
  if (
    saeLayers.length === 0 || new Set(saeLayers).size !== saeLayers.length ||
    saeLayers.some((layer) => !Number.isSafeInteger(layer) || !runtimeLayers.includes(layer))
  ) {
    throw fittingError(
      "BROWSER_SAE_FITTING_PORT_INVALID",
      "The exact SAE fitting port returned invalid layer coverage",
      500,
    );
  }
  if (requested === undefined || requested === null) return [...saeLayers].sort((a, b) => a - b);
  const selected = resolveBrowserFitLayerIndices(requested, runtimeLayers);
  const missing = selected.filter((layer) => !saeLayers.includes(layer));
  if (missing.length > 0) {
    throw fittingError(
      "SAE_LAYER_COVERAGE_INCOMPLETE",
      `The installed SAE does not cover requested layer(s): ${missing.join(", ")}`,
    );
  }
  return selected;
}

function transformCapturedCentroids(
  centroids: BrowserFittingCentroids,
  transform: BrowserCentroidTransform | undefined,
): BrowserFittingCentroids {
  if (transform === undefined) return centroids;
  const layers = new Map<number, BrowserFittingLayerCentroids>();
  for (const [layer, pooled] of centroids.layers) {
    const transformed = transform(layer, {
      rows: pooled.rows,
      columns: pooled.columns,
      values: pooled.values.slice(),
    });
    if (
      transformed.rows !== pooled.rows || transformed.columns !== pooled.columns ||
      !(transformed.values instanceof Float64Array) ||
      transformed.values.length !== pooled.values.length ||
      transformed.values.some((value) => !Number.isFinite(value))
    ) {
      throw fittingError(
        "BROWSER_SAE_FITTING_PORT_INVALID",
        `The exact SAE fitting port returned invalid centroids for layer ${layer}`,
        500,
      );
    }
    layers.set(layer, transformed);
  }
  return { ...centroids, layers };
}

function selectedSaeProvenance(
  provenance: BrowserSaeFitIdentity,
  layers: readonly number[],
  fullCoverage: boolean,
): BrowserSaeFitIdentity {
  const idsByLayer = new Map<number, string>();
  for (const layer of layers) {
    const id = provenance.idsByLayer.get(layer);
    if (!id) {
      throw fittingError(
        "BROWSER_SAE_FITTING_PORT_INVALID",
        `The exact SAE fitting port returned no provenance ID for layer ${layer}`,
        500,
      );
    }
    idsByLayer.set(layer, id);
  }
  return { ...provenance, idsByLayer, fullCoverage };
}

function browserModelSourceFingerprint(request: BrowserModelLoadRequest): string {
  const fingerprint = request.variant.runtimeIdentity.convertedManifestSha256;
  if (!/^[a-f0-9]{64}$/.test(fingerprint)) {
    throw fittingError(
      "BROWSER_MODEL_SOURCE_IDENTITY_INVALID",
      "The browser model runtime has no exact converted-manifest fingerprint",
      500,
    );
  }
  return fingerprint;
}

function assertFitWithinProfile(
  manifold: BrowserFittableManifold,
  profile: StructuredHookCapacityProfile,
): void {
  if (manifold.fitMode === "authored") {
    const intrinsicDimensions = manifold.nodes[0].coords.length;
    const embeddedDimensions = embedBrowserAuthoredCoordinates(
      manifold.domain,
      Float64Array.from(manifold.nodes[0].coords),
      1,
    ).length;
    if (
      intrinsicDimensions > profile.maxIntrinsicDim ||
      embeddedDimensions > profile.maxEmbedDim ||
      manifold.nodes.length > profile.maxCurveNodes
    ) {
      throw fittingError(
        "BROWSER_FIT_PROFILE_LIMIT",
        `This model supports at most ${profile.maxIntrinsicDim} authored dimensions, ${profile.maxEmbedDim} embedded dimensions, and ${profile.maxCurveNodes} curved nodes`,
      );
    }
    return;
  }
  assertDiscoverHyperparametersWithinProfile(manifold.hyperparams, profile);
  if (manifold.fitMode === "spectral" && manifold.nodes.length > profile.maxCurveNodes) {
    throw fittingError(
      "BROWSER_FIT_PROFILE_LIMIT",
      `This model supports at most ${profile.maxCurveNodes} nodes in a curved manifold`,
    );
  }
}

function assertDiscoverHyperparametersWithinProfile(
  hyperparams: Readonly<Record<string, number | string>>,
  profile: StructuredHookCapacityProfile,
): void {
  const maxDimensions = hyperparams.max_dim;
  const maxSubspaceDimensions = hyperparams.max_subspace_dim;
  if (
    typeof maxDimensions === "number" && maxDimensions > profile.maxIntrinsicDim ||
    typeof maxSubspaceDimensions === "number" && maxSubspaceDimensions > profile.maxRank
  ) {
    throw fittingError(
      "BROWSER_FIT_PROFILE_LIMIT",
      `This model supports fitted layouts up to ${profile.maxIntrinsicDim} dimensions and subspaces up to rank ${profile.maxRank}`,
    );
  }
}

function requireContextBinding(request: BrowserModelLoadRequest): string {
  const context = request.variant.contextProfiles.find(
    (profile) => profile.contextTokens === request.contextTokens,
  );
  if (!context) {
    throw fittingError("BROWSER_CONTEXT_BINDING_MISSING", "The selected model context profile is unavailable");
  }
  return context.bindingSha256;
}

function integerHyperparameter(
  value: unknown,
  fallback: number,
  maximum?: number,
): number {
  const resolved = Number.isSafeInteger(value) && (value as number) > 0
    ? value as number
    : fallback;
  if (maximum !== undefined && resolved > maximum) {
    throw fittingError(
      "BROWSER_FIT_PROFILE_LIMIT",
      `This model supports fitting dimensions up to ${maximum}`,
    );
  }
  return resolved;
}

function numberHyperparameter(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function optionalInteger(value: unknown): number | undefined {
  return Number.isSafeInteger(value) && (value as number) > 0 ? value as number : undefined;
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined;
}

function smoothingHyperparameter(value: unknown): number | "auto" {
  return value === "auto" || value === undefined
    ? "auto"
    : typeof value === "number" && Number.isFinite(value) && value >= 0
      ? value
      : "auto";
}

interface BrowserInstalledManifoldInfo extends ManifoldInfo {
  archive_source?: unknown;
}

function fittedSourceClosure(
  detail: BrowserInstalledManifoldInfo,
  template: TemplateDetail | null,
): BrowserFittedSourceClosure {
  let source: BrowserFittedSourceClosure["source"];
  const raw = detail.archive_source;
  if (raw === undefined && detail.source === "local") {
    source = { uri: "local", repository: null, revision: null };
  } else if (
    raw && typeof raw === "object" && !Array.isArray(raw) &&
    Object.keys(raw).sort().join(",") === "repository,revision,uri"
  ) {
    const candidate = raw as Record<string, unknown>;
    const hf = typeof candidate.repository === "string" &&
      typeof candidate.revision === "string";
    if (
      typeof candidate.uri !== "string" || !candidate.uri ||
      (hf
        ? candidate.uri !== `hf://${candidate.repository}@${candidate.revision}`
        : candidate.repository !== null || candidate.revision !== null ||
          candidate.uri.startsWith("hf://")) ||
      candidate.uri !== detail.source
    ) {
      throw fittingError(
        "BROWSER_MANIFOLD_PROVENANCE_INVALID",
        "The installed manifold has inconsistent source provenance",
      );
    }
    source = {
      uri: candidate.uri,
      repository: candidate.repository as string | null,
      revision: candidate.revision as string | null,
    };
  } else {
    throw fittingError(
      "BROWSER_MANIFOLD_PROVENANCE_INVALID",
      "The installed manifold has no exact source repository and revision provenance",
    );
  }
  if (detail.template_ref === null) {
    if (template !== null) {
      throw fittingError("BROWSER_TEMPLATE_INVALID", "An untemplated manifold returned a template closure");
    }
    return { source, tags: [...detail.tags], template: null };
  }
  if (template === null) {
    throw fittingError("BROWSER_TEMPLATE_INVALID", "The fitted manifold is missing its template closure");
  }
  const [namespace, name, extra] = detail.template_ref.split("/");
  if (extra !== undefined || !namespace || !name || template.namespace !== namespace || template.name !== name) {
    throw fittingError("BROWSER_TEMPLATE_INVALID", "The fitted manifold template identity is inconsistent");
  }
  return {
    source,
    tags: [...detail.tags],
    template: {
      reference: detail.template_ref,
      payload: templatePayloadFromDetail(template),
    },
  };
}

function requireManifold(value: unknown): BrowserInstalledManifoldInfo {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw fittingError("BROWSER_MANIFOLD_INVALID", "The installed manifold could not be read");
  }
  return value as BrowserInstalledManifoldInfo;
}

function requireTemplate(value: unknown): TemplateDetail {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw fittingError("BROWSER_TEMPLATE_INVALID", "The installed template could not be read");
  }
  return value as TemplateDetail;
}

function validateTemplateForScoring(
  template: TemplateDetail,
  namespace: string,
  name: string,
): void {
  if (
    template.namespace !== namespace ||
    template.name !== name ||
    typeof template.slot !== "string" ||
    template.slot.length === 0 ||
    !Array.isArray(template.values) ||
    template.values.length === 0 ||
    template.values.some((value) => typeof value !== "string") ||
    !Array.isArray(template.labels) ||
    template.labels.length !== template.values.length ||
    template.labels.some((label) => typeof label !== "string") ||
    !Array.isArray(template.contexts) ||
    template.contexts.length === 0
  ) {
    throw fittingError(
      "BROWSER_TEMPLATE_INVALID",
      "The installed template is not valid for browser scoring",
    );
  }
  for (const context of template.contexts) {
    if (
      !context ||
      typeof context !== "object" ||
      typeof context.assistant !== "string" ||
      context.assistant.split(template.slot).length !== 2 ||
      !Array.isArray(context.turns) ||
      context.turns.length === 0 ||
      context.turns.at(-1)?.role !== "user" ||
      context.turns.some((turn) =>
        !turn ||
        !["system", "user", "assistant"].includes(turn.role) ||
        typeof turn.content !== "string" ||
        turn.content.includes(template.slot)
      )
    ) {
      throw fittingError(
        "BROWSER_TEMPLATE_INVALID",
        "The installed template context is not valid for browser scoring",
      );
    }
  }
}

function sharedTokenPrefix(left: readonly number[], right: readonly number[]): number {
  const length = Math.min(left.length, right.length);
  let index = 0;
  while (index < length && left[index] === right[index]) index += 1;
  return index;
}

function requireChatRole(value: string): "system" | "user" | "assistant" {
  if (value !== "system" && value !== "user" && value !== "assistant") {
    throw fittingError(
      "INVALID_TEMPLATE_SCORE_REQUEST",
      `Browser template scoring does not support chat role ${JSON.stringify(value)}`,
    );
  }
  return value;
}

function applyRestrictedChoiceProbabilities(choices: ChoiceScore[]): void {
  const assign = (
    key: "sum_logprob" | "mean_logprob",
    output: "prob_sum" | "prob_mean",
  ): void => {
    const eligible = choices.map((choice) =>
      choice.n_tokens === 0 ? Number.NEGATIVE_INFINITY : choice[key]
    );
    if (eligible.every((value) => value === Number.NEGATIVE_INFINITY)) {
      if (choices.every((choice) => choice.n_tokens === 0)) return;
      throw fittingError(
        "TEMPLATE_SCORE_UNDERFLOW",
        "Every template candidate had zero probability in the browser sampler distribution",
        500,
      );
    }
    const maximum = Math.max(...eligible);
    const weights = eligible.map((value) =>
      value === Number.NEGATIVE_INFINITY ? 0 : Math.exp(value - maximum)
    );
    const total = weights.reduce((sum, value) => sum + value, 0);
    for (let index = 0; index < choices.length; index += 1) {
      choices[index][output] = weights[index] / total;
    }
  };
  assign("sum_logprob", "prob_sum");
  assign("mean_logprob", "prob_mean");
}

function digest(value: unknown): string {
  return bytesToHex(sha256(new TextEncoder().encode(JSON.stringify(value))));
}

function fittingError(code: string, message: string, status = 409): Error & {
  code: string;
  status: number;
  recoverable: boolean;
} {
  return Object.assign(new Error(message), { code, status, recoverable: status < 500 });
}
