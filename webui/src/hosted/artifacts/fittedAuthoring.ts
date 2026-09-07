import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import type {
  FinalizedBrowserCurvedLayer,
  FinalizedBrowserAffineLayer,
} from "../fitting/coordinator";
import type { AffineFisherResult, TopologyResult } from "../fitting/workerContracts";
import type { BrowserMergedDiscoverManifold } from "./authoring";
import type { ManifoldDomain } from "../../lib/types";
import type { DrowseArchiveManifest } from "./types";
import { buildDrowseArchive, jsonBytes } from "./builder";
import { encodeFp32Safetensors, type Fp32TensorInput } from "./safetensors";
import {
  discoverManifoldNodesSha256,
  drowseTensorFilename,
  templateClosureSha256,
} from "./drowseArchive";

const SHA256 = /^[a-f0-9]{64}$/;

export interface BrowserFlatFitIdentity {
  runtimeIdentitySha256: string;
  contextBindingSha256: string;
  modelSourceFingerprint: string | null;
  captureSha256: string;
  captureVersion: number;
  captureRenderSha256?: string | null;
  baselinePromptsSha256?: string | null;
  fitPolicyVersion: number;
}

export interface BrowserFittedSourceClosure {
  source: DrowseArchiveManifest["source"];
  tags: readonly string[];
  template: {
    reference: string;
    payload: Readonly<Record<string, unknown>>;
  } | null;
}

export interface BrowserSaeFitIdentity {
  release: string;
  revision: string;
  fingerprint: string;
  idsByLayer: ReadonlyMap<number, string>;
  fullCoverage: boolean;
}

export interface BrowserAuthoredFittableManifold {
  namespace: string;
  name: string;
  description: string;
  fitMode: "authored";
  domain: ManifoldDomain;
  nodes: ReadonlyArray<{
    label: string;
    coords: readonly number[];
    statements: readonly string[];
    role: string | null;
    kind: string | null;
  }>;
}

export interface BrowserFittedFlatDiscoverPackInput {
  manifold: BrowserMergedDiscoverManifold;
  closure: BrowserFittedSourceClosure;
  modelId: string;
  producerVersion: string;
  drowseVersion?: string;
  identity: BrowserFlatFitIdentity;
  topology: TopologyResult;
  consensusGram: Float64Array;
  nodeCoordinates: Float64Array;
  evaluatedLayers: ReadonlyMap<number, AffineFisherResult>;
  layers: ReadonlyMap<number, FinalizedBrowserAffineLayer>;
  sae?: BrowserSaeFitIdentity | null;
}

export interface BrowserFittedCurvedDiscoverPackInput {
  manifold: BrowserMergedDiscoverManifold;
  closure: BrowserFittedSourceClosure;
  modelId: string;
  producerVersion: string;
  drowseVersion?: string;
  identity: BrowserFlatFitIdentity;
  topology: TopologyResult;
  consensusGram: Float64Array;
  evaluatedLayers: ReadonlyMap<number, AffineFisherResult>;
  layers: ReadonlyMap<number, FinalizedBrowserCurvedLayer>;
  sae?: BrowserSaeFitIdentity | null;
}

export interface BrowserFittedAuthoredPackInput {
  manifold: BrowserAuthoredFittableManifold;
  closure: BrowserFittedSourceClosure;
  modelId: string;
  producerVersion: string;
  drowseVersion?: string;
  identity: BrowserFlatFitIdentity;
  evaluatedLayers: ReadonlyMap<number, AffineFisherResult>;
  layers: ReadonlyMap<number, FinalizedBrowserCurvedLayer>;
  sae?: BrowserSaeFitIdentity | null;
}

export async function browserFittedFlatDiscoverPack(
  input: BrowserFittedFlatDiscoverPackInput,
): Promise<Blob> {
  validateInput(input);
  const { manifold, identity, topology } = input;
  const closure = prepareClosure(input.closure, true);
  const primary = `manifolds/${manifold.namespace}/${manifold.name}`;
  const nodeBytes = manifold.nodes.map((node) => jsonBytes(node.statements));
  const roles = manifold.nodes.map((node) => node.role);
  const kinds = manifold.nodes.map((node) => node.kind);
  const nodeLabels = manifold.nodes.map((node) => node.label);
  const fittedLayers = [...input.layers.keys()].sort((left, right) => left - right);
  const provenance = fitProvenance(input.modelId, fittedLayers, input.sae);
  const domain = {
    type: "custom",
    embed_dim: topology.intrinsicDimensions,
    bounds: null,
  };
  const tensors: Record<string, Fp32TensorInput> = {
    node_coords: {
      shape: [manifold.nodes.length, topology.intrinsicDimensions],
      data: fp32(input.nodeCoordinates, "node coordinates"),
    },
  };
  const shares: Record<string, number> = {};
  for (const layer of fittedLayers) {
    const fit = input.layers.get(layer)!;
    tensors[`layer_${layer}.mean`] = {
      shape: [fit.columns],
      data: fp32(fit.mean, `layer ${layer} mean`),
    };
    tensors[`layer_${layer}.basis`] = {
      shape: [fit.components, fit.columns],
      data: fp32(fit.basis, `layer ${layer} basis`),
    };
    tensors[`layer_${layer}.node_coords`] = {
      shape: [fit.nodeCount, fit.components],
      data: fp32(fit.nodeCoordinates, `layer ${layer} node coordinates`),
    };
    if (fit.affineMap !== null) {
      tensors[`layer_${layer}.affine_map`] = {
        shape: [topology.intrinsicDimensions, fit.components],
        data: fp32(fit.affineMap, `layer ${layer} affine map`),
      };
    }
    shares[String(layer)] = fit.mahalanobisShare;
  }
  const tensorName = provenance.tensorName;
  const sidecarName = tensorName.replace(/\.safetensors$/, ".json");
  const tensorBytes = encodeFp32Safetensors(tensors, {
    path: `${primary}/${tensorName}`,
  });
  const isAuto = manifold.fitMode === "auto";
  const isMonopolar = manifold.nodes.length === 1;
  const sidecar = {
    format_version: 10,
    name: manifold.name,
    method: provenance.sae !== null
      ? isMonopolar ? "manifold_monopolar_sae" : "manifold_discover_sae"
      : isMonopolar
        ? "manifold_monopolar"
      : isAuto
        ? "manifold_discover_auto"
        : "manifold_discover_pca",
    drowse_version: input.drowseVersion ?? input.producerVersion,
    fit_mode: manifold.fitMode,
    hyperparams: manifold.hyperparams,
    diagnostics: topologyDiagnostics(topology),
    node_count: manifold.nodes.length,
    node_labels: nodeLabels,
    node_roles: roles,
    node_kinds: kinds,
    domain,
    node_spread_per_layer: nodeSpreadPerLayer(input.evaluatedLayers),
    fitted_layers: fittedLayers,
    mahalanobis_share_per_layer: shares,
    origin_per_layer: {},
    feature_space: provenance.featureSpace,
    nodes_sha256: discoverManifoldNodesSha256({
      labels: nodeLabels,
      nodeBytes,
      fitMode: manifold.fitMode,
      hyperparams: manifold.hyperparams,
      roles,
      kinds,
      templateSha256: closure.templateSha256,
    }),
    sae_release: provenance.sae?.release ?? null,
    sae_revision: provenance.sae?.revision ?? null,
    sae_fingerprint: provenance.sae?.fingerprint ?? null,
    sae_ids_by_layer: provenance.saeIds,
    sae_full_coverage: provenance.sae?.fullCoverage ?? false,
    model_fingerprint: identity.runtimeIdentitySha256,
    context_binding_sha256: identity.contextBindingSha256,
    model_source_fingerprint: identity.modelSourceFingerprint,
    capture_sha256: identity.captureSha256,
    capture_version: identity.captureVersion,
    capture_render_sha256: identity.captureRenderSha256 ?? null,
    baseline_prompts_sha256: identity.baselinePromptsSha256 ?? null,
    fit_policy_version: identity.fitPolicyVersion,
    share_metric: "mahalanobis",
    subspace_metric: isMonopolar ? "euclidean" : "mahalanobis",
    rbf_smoothing_per_layer: {},
    sigma_field_per_layer: {},
    resolved_fit_mode: isAuto ? "pca" : null,
    topology_winner: isAuto ? topology.winnerName : null,
    topology_candidates: isAuto
      ? topology.candidates.map((candidate) => ({
          name: candidate.name,
          fit_mode: candidate.fitMode,
          intrinsic_dim: candidate.dimensions,
          score: candidate.score,
          viable: candidate.viable,
          reason: candidate.reason,
        }))
      : [],
    components: null,
    bake_policy: null,
    source_model_id: null,
    source_model_fingerprint: null,
    transfer_quality_estimate: null,
  };
  const sidecarBytes = jsonBytes(sidecar);
  const manifoldPayload = {
    format_version: 10,
    name: manifold.name,
    description: manifold.description,
    fit_mode: manifold.fitMode,
    hyperparams: manifold.hyperparams,
    nodes: manifold.nodes.map((node) => ({
      label: node.label,
      role: node.role,
      kind: node.kind,
    })),
    files: {
      [tensorName]: bytesToHex(sha256(tensorBytes)),
      [sidecarName]: bytesToHex(sha256(sidecarBytes)),
    },
    source: closure.source.uri,
    tags: closure.tags,
    template_ref: closure.templateReference,
  };
  const files: Record<string, Uint8Array> = {
    [`${primary}/manifold.json`]: jsonBytes(manifoldPayload),
    [`${primary}/${tensorName}`]: tensorBytes,
    [`${primary}/${sidecarName}`]: sidecarBytes,
  };
  manifold.nodes.forEach((node, index) => {
    files[`${primary}/nodes/${String(index).padStart(2, "0")}_${node.label}.json`] =
      nodeBytes[index];
  });
  if (closure.templatePath !== null && closure.templateBytes !== null) {
    files[`${closure.templatePath}/template.json`] = closure.templateBytes;
  }
  return buildDrowseArchive({
    primary,
    template: closure.templatePath,
    producerVersion: input.producerVersion,
    source: closure.source,
    files,
  });
}

export async function browserFittedCurvedDiscoverPack(
  input: BrowserFittedCurvedDiscoverPackInput,
): Promise<Blob> {
  validateCurvedInput(input);
  const { manifold, identity, topology } = input;
  const closure = prepareClosure(input.closure, true);
  const primary = `manifolds/${manifold.namespace}/${manifold.name}`;
  const nodeBytes = manifold.nodes.map((node) => jsonBytes(node.statements));
  const roles = manifold.nodes.map((node) => node.role);
  const kinds = manifold.nodes.map((node) => node.kind);
  const nodeLabels = manifold.nodes.map((node) => node.label);
  const fittedLayers = [...input.layers.keys()].sort((left, right) => left - right);
  const provenance = fitProvenance(input.modelId, fittedLayers, input.sae);
  const domain = topology.periodicDimensions > 0
    ? {
        type: "box",
        axes: Array.from({ length: topology.intrinsicDimensions }, (_, index) => ({
          name: `theta${index}`,
          periodic: true,
          period: 2 * Math.PI,
          lo: 0,
          hi: 1,
        })),
      }
    : {
        type: "custom",
        embed_dim: topology.embeddedCoordinates.length / manifold.nodes.length,
        bounds: null,
      };
  const tensors: Record<string, Fp32TensorInput> = {
    node_coords: {
      shape: [manifold.nodes.length, topology.intrinsicDimensions],
      data: fp32(topology.coordinates, "node coordinates"),
    },
  };
  const shares: Record<string, number> = {};
  const origins: Record<string, number[]> = {};
  const smoothing: Record<string, { lambda: number; edf: number; gcv: number }> = {};
  const sigma: Record<string, { sigma_mean: number; sigma_min: number; sigma_max: number; lambda: number }> = {};
  for (const layer of fittedLayers) {
    const fit = input.layers.get(layer)!;
    const prefix = `layer_${layer}`;
    tensors[`${prefix}.mean`] = { shape: [fit.columns], data: fp32(fit.mean, `${prefix} mean`) };
    tensors[`${prefix}.basis`] = {
      shape: [fit.components, fit.columns],
      data: fp32(fit.basis, `${prefix} basis`),
    };
    tensors[`${prefix}.node_params`] = {
      shape: [fit.nodeCount, fit.surface.inputDimensions],
      data: fp32(fit.surface.nodes, `${prefix} node parameters`),
    };
    tensors[`${prefix}.rbf_weights`] = {
      shape: [fit.nodeCount, fit.components],
      data: fp32(fit.surface.weights, `${prefix} RBF weights`),
    };
    tensors[`${prefix}.poly_coeffs`] = {
      shape: [fit.surface.inputDimensions + 1, fit.components],
      data: fp32(fit.surface.polynomial, `${prefix} polynomial coefficients`),
    };
    tensors[`${prefix}.coord_offset`] = {
      shape: [fit.surface.inputDimensions],
      data: fp32(fit.surface.coordinateOffset, `${prefix} coordinate offset`),
    };
    tensors[`${prefix}.coord_scale`] = {
      shape: [fit.surface.inputDimensions],
      data: fp32(fit.surface.coordinateScale, `${prefix} coordinate scale`),
    };
    if (fit.sigmaSurface !== null) {
      tensors[`${prefix}.sigma_rbf_weights`] = {
        shape: [fit.nodeCount, 1],
        data: fp32(fit.sigmaSurface.weights, `${prefix} sigma weights`),
      };
      tensors[`${prefix}.sigma_poly_coeffs`] = {
        shape: [fit.surface.inputDimensions + 1, 1],
        data: fp32(fit.sigmaSurface.polynomial, `${prefix} sigma polynomial coefficients`),
      };
    }
    shares[String(layer)] = fit.mahalanobisShare;
    origins[String(layer)] = [...fit.origin];
    smoothing[String(layer)] = {
      lambda: fit.surface.lambda,
      edf: fit.surface.effectiveDegreesOfFreedom,
      gcv: fit.surface.gcv,
    };
    if (fit.sigmaSummary !== null) {
      sigma[String(layer)] = {
        sigma_mean: fit.sigmaSummary.mean,
        sigma_min: fit.sigmaSummary.min,
        sigma_max: fit.sigmaSummary.max,
        lambda: fit.sigmaSummary.lambda,
      };
    }
  }
  const tensorName = provenance.tensorName;
  const sidecarName = tensorName.replace(/\.safetensors$/, ".json");
  const tensorBytes = encodeFp32Safetensors(tensors, { path: `${primary}/${tensorName}` });
  const isAuto = manifold.fitMode === "auto";
  const sidecar = {
    format_version: 10,
    name: manifold.name,
    method: provenance.sae !== null
      ? "manifold_discover_sae"
      : isAuto ? "manifold_discover_auto" : "manifold_discover_spectral",
    drowse_version: input.drowseVersion ?? input.producerVersion,
    fit_mode: manifold.fitMode,
    hyperparams: manifold.hyperparams,
    diagnostics: topologyDiagnostics(topology),
    node_count: manifold.nodes.length,
    node_labels: nodeLabels,
    node_roles: roles,
    node_kinds: kinds,
    domain,
    node_spread_per_layer: nodeSpreadPerLayer(input.evaluatedLayers),
    fitted_layers: fittedLayers,
    mahalanobis_share_per_layer: shares,
    origin_per_layer: origins,
    feature_space: provenance.featureSpace,
    nodes_sha256: discoverManifoldNodesSha256({
      labels: nodeLabels,
      nodeBytes,
      fitMode: manifold.fitMode,
      hyperparams: manifold.hyperparams,
      roles,
      kinds,
      templateSha256: closure.templateSha256,
    }),
    sae_release: provenance.sae?.release ?? null,
    sae_revision: provenance.sae?.revision ?? null,
    sae_fingerprint: provenance.sae?.fingerprint ?? null,
    sae_ids_by_layer: provenance.saeIds,
    sae_full_coverage: provenance.sae?.fullCoverage ?? false,
    model_fingerprint: identity.runtimeIdentitySha256,
    context_binding_sha256: identity.contextBindingSha256,
    model_source_fingerprint: identity.modelSourceFingerprint,
    capture_sha256: identity.captureSha256,
    capture_version: identity.captureVersion,
    capture_render_sha256: identity.captureRenderSha256 ?? null,
    baseline_prompts_sha256: identity.baselinePromptsSha256 ?? null,
    fit_policy_version: identity.fitPolicyVersion,
    share_metric: "mahalanobis",
    subspace_metric: "mahalanobis",
    rbf_smoothing_per_layer: smoothing,
    sigma_field_per_layer: sigma,
    resolved_fit_mode: isAuto ? "spectral" : null,
    topology_winner: isAuto ? topology.winnerName : null,
    topology_candidates: isAuto
      ? topology.candidates.map((candidate) => ({
          name: candidate.name,
          fit_mode: candidate.fitMode,
          intrinsic_dim: candidate.dimensions,
          score: candidate.score,
          viable: candidate.viable,
          reason: candidate.reason,
        }))
      : [],
    components: null,
    bake_policy: null,
    source_model_id: null,
    source_model_fingerprint: null,
    transfer_quality_estimate: null,
  };
  const sidecarBytes = jsonBytes(sidecar);
  const manifoldPayload = {
    format_version: 10,
    name: manifold.name,
    description: manifold.description,
    fit_mode: manifold.fitMode,
    hyperparams: manifold.hyperparams,
    nodes: manifold.nodes.map((node) => ({ label: node.label, role: node.role, kind: node.kind })),
    files: {
      [tensorName]: bytesToHex(sha256(tensorBytes)),
      [sidecarName]: bytesToHex(sha256(sidecarBytes)),
    },
    source: closure.source.uri,
    tags: closure.tags,
    template_ref: closure.templateReference,
  };
  const files: Record<string, Uint8Array> = {
    [`${primary}/manifold.json`]: jsonBytes(manifoldPayload),
    [`${primary}/${tensorName}`]: tensorBytes,
    [`${primary}/${sidecarName}`]: sidecarBytes,
  };
  manifold.nodes.forEach((node, index) => {
    files[`${primary}/nodes/${String(index).padStart(2, "0")}_${node.label}.json`] = nodeBytes[index];
  });
  if (closure.templatePath !== null && closure.templateBytes !== null) {
    files[`${closure.templatePath}/template.json`] = closure.templateBytes;
  }
  return buildDrowseArchive({
    primary,
    template: closure.templatePath,
    producerVersion: input.producerVersion,
    source: closure.source,
    files,
  });
}

export async function browserFittedAuthoredPack(
  input: BrowserFittedAuthoredPackInput,
): Promise<Blob> {
  validateAuthoredInput(input);
  const { manifold, identity } = input;
  const closure = prepareClosure(input.closure, false);
  const primary = `manifolds/${manifold.namespace}/${manifold.name}`;
  const nodeBytes = manifold.nodes.map((node) => jsonBytes(node.statements));
  const roles = manifold.nodes.map((node) => node.role);
  const kinds = manifold.nodes.map((node) => node.kind);
  const nodeLabels = manifold.nodes.map((node) => node.label);
  const fittedLayers = [...input.layers.keys()].sort((left, right) => left - right);
  const provenance = fitProvenance(input.modelId, fittedLayers, input.sae);
  const intrinsicDimensions = domainDimensions(manifold.domain);
  const nodeCoordinates = Float64Array.from(manifold.nodes.flatMap((node) => [...node.coords]));
  const tensors: Record<string, Fp32TensorInput> = {
    node_coords: {
      shape: [manifold.nodes.length, intrinsicDimensions],
      data: fp32(nodeCoordinates, "node coordinates"),
    },
  };
  const shares: Record<string, number> = {};
  const origins: Record<string, number[]> = {};
  const smoothing: Record<string, { lambda: number; edf: number; gcv: number }> = {};
  const sigma: Record<string, {
    sigma_mean: number;
    sigma_min: number;
    sigma_max: number;
    lambda: number;
  }> = {};
  for (const layer of fittedLayers) {
    const fit = input.layers.get(layer)!;
    const prefix = `layer_${layer}`;
    tensors[`${prefix}.mean`] = {
      shape: [fit.columns],
      data: fp32(fit.mean, `${prefix} mean`),
    };
    tensors[`${prefix}.basis`] = {
      shape: [fit.components, fit.columns],
      data: fp32(fit.basis, `${prefix} basis`),
    };
    tensors[`${prefix}.node_params`] = {
      shape: [fit.nodeCount, fit.surface.inputDimensions],
      data: fp32(fit.surface.nodes, `${prefix} node parameters`),
    };
    tensors[`${prefix}.rbf_weights`] = {
      shape: [fit.nodeCount, fit.components],
      data: fp32(fit.surface.weights, `${prefix} RBF weights`),
    };
    tensors[`${prefix}.poly_coeffs`] = {
      shape: [fit.surface.inputDimensions + 1, fit.components],
      data: fp32(fit.surface.polynomial, `${prefix} polynomial coefficients`),
    };
    tensors[`${prefix}.coord_offset`] = {
      shape: [fit.surface.inputDimensions],
      data: fp32(fit.surface.coordinateOffset, `${prefix} coordinate offset`),
    };
    tensors[`${prefix}.coord_scale`] = {
      shape: [fit.surface.inputDimensions],
      data: fp32(fit.surface.coordinateScale, `${prefix} coordinate scale`),
    };
    if (fit.sigmaSurface !== null) {
      tensors[`${prefix}.sigma_rbf_weights`] = {
        shape: [fit.nodeCount, 1],
        data: fp32(fit.sigmaSurface.weights, `${prefix} sigma weights`),
      };
      tensors[`${prefix}.sigma_poly_coeffs`] = {
        shape: [fit.surface.inputDimensions + 1, 1],
        data: fp32(fit.sigmaSurface.polynomial, `${prefix} sigma polynomial coefficients`),
      };
    }
    shares[String(layer)] = fit.mahalanobisShare;
    origins[String(layer)] = [...fit.origin];
    smoothing[String(layer)] = {
      lambda: fit.surface.lambda,
      edf: fit.surface.effectiveDegreesOfFreedom,
      gcv: fit.surface.gcv,
    };
    if (fit.sigmaSummary !== null) {
      sigma[String(layer)] = {
        sigma_mean: fit.sigmaSummary.mean,
        sigma_min: fit.sigmaSummary.min,
        sigma_max: fit.sigmaSummary.max,
        lambda: fit.sigmaSummary.lambda,
      };
    }
  }
  const tensorName = provenance.tensorName;
  const sidecarName = tensorName.replace(/\.safetensors$/, ".json");
  const tensorBytes = encodeFp32Safetensors(tensors, { path: `${primary}/${tensorName}` });
  const sidecar = {
    format_version: 10,
    name: manifold.name,
    method: provenance.sae === null ? "manifold_pca" : "manifold_sae",
    drowse_version: input.drowseVersion ?? input.producerVersion,
    fit_mode: "authored",
    hyperparams: {},
    diagnostics: {},
    node_count: manifold.nodes.length,
    node_labels: nodeLabels,
    node_roles: roles,
    node_kinds: kinds,
    domain: manifold.domain,
    node_spread_per_layer: nodeSpreadPerLayer(input.evaluatedLayers),
    fitted_layers: fittedLayers,
    mahalanobis_share_per_layer: shares,
    origin_per_layer: origins,
    feature_space: provenance.featureSpace,
    nodes_sha256: authoredNodesSha256(manifold, nodeBytes),
    sae_release: provenance.sae?.release ?? null,
    sae_revision: provenance.sae?.revision ?? null,
    sae_fingerprint: provenance.sae?.fingerprint ?? null,
    sae_ids_by_layer: provenance.saeIds,
    sae_full_coverage: provenance.sae?.fullCoverage ?? false,
    model_fingerprint: identity.runtimeIdentitySha256,
    context_binding_sha256: identity.contextBindingSha256,
    model_source_fingerprint: identity.modelSourceFingerprint,
    capture_sha256: identity.captureSha256,
    capture_version: identity.captureVersion,
    capture_render_sha256: identity.captureRenderSha256 ?? null,
    baseline_prompts_sha256: identity.baselinePromptsSha256 ?? null,
    fit_policy_version: identity.fitPolicyVersion,
    share_metric: "mahalanobis",
    subspace_metric: "mahalanobis",
    rbf_smoothing_per_layer: smoothing,
    sigma_field_per_layer: sigma,
    resolved_fit_mode: null,
    topology_winner: null,
    topology_candidates: [],
    components: null,
    bake_policy: null,
    source_model_id: null,
    source_model_fingerprint: null,
    transfer_quality_estimate: null,
  };
  const sidecarBytes = jsonBytes(sidecar);
  const manifoldPayload = {
    format_version: 10,
    name: manifold.name,
    description: manifold.description,
    fit_mode: "authored",
    domain: manifold.domain,
    nodes: manifold.nodes.map((node) => ({
      label: node.label,
      coords: [...node.coords],
      role: node.role,
      kind: node.kind,
    })),
    files: {
      [tensorName]: bytesToHex(sha256(tensorBytes)),
      [sidecarName]: bytesToHex(sha256(sidecarBytes)),
    },
    source: closure.source.uri,
    tags: closure.tags,
    template_ref: null,
  };
  const files: Record<string, Uint8Array> = {
    [`${primary}/manifold.json`]: jsonBytes(manifoldPayload),
    [`${primary}/${tensorName}`]: tensorBytes,
    [`${primary}/${sidecarName}`]: sidecarBytes,
  };
  manifold.nodes.forEach((node, index) => {
    files[`${primary}/nodes/${String(index).padStart(2, "0")}_${node.label}.json`] = nodeBytes[index];
  });
  return buildDrowseArchive({
    primary,
    template: null,
    producerVersion: input.producerVersion,
    source: closure.source,
    files,
  });
}

interface PreparedBrowserFittedSourceClosure {
  source: DrowseArchiveManifest["source"];
  tags: string[];
  templateReference: string | null;
  templatePath: string | null;
  templateBytes: Uint8Array | null;
  templateSha256: string | null;
}

function prepareClosure(
  closure: BrowserFittedSourceClosure,
  allowTemplate: boolean,
): PreparedBrowserFittedSourceClosure {
  const source = closure?.source;
  if (
    !source || typeof source !== "object" || Array.isArray(source) ||
    Object.keys(source).sort().join(",") !== "repository,revision,uri" ||
    typeof source.uri !== "string" || source.uri.length === 0
  ) {
    throw new TypeError("Browser fitted publication source provenance is invalid");
  }
  const hasHfIdentity = typeof source.repository === "string" &&
    typeof source.revision === "string";
  if (
    hasHfIdentity
      ? source.uri !== `hf://${source.repository}@${source.revision}`
      : source.repository !== null || source.revision !== null || source.uri.startsWith("hf://")
  ) {
    throw new TypeError("Browser fitted publication source provenance is inconsistent");
  }
  if (!Array.isArray(closure.tags) || closure.tags.some((tag) => typeof tag !== "string")) {
    throw new TypeError("Browser fitted publication tags are invalid");
  }
  if (closure.template === null) {
    return {
      source: { ...source },
      tags: [...closure.tags],
      templateReference: null,
      templatePath: null,
      templateBytes: null,
      templateSha256: null,
    };
  }
  if (!allowTemplate) {
    throw new TypeError("Browser authored fitted publication cannot carry a template closure");
  }
  const match = /^([a-z][a-z0-9._-]{0,63})\/([a-z][a-z0-9._-]{0,63})$/.exec(
    closure.template.reference,
  );
  if (match === null || !closure.template.payload ||
      typeof closure.template.payload !== "object" || Array.isArray(closure.template.payload)) {
    throw new TypeError("Browser fitted publication template closure is invalid");
  }
  const templateBytes = jsonBytes(closure.template.payload);
  return {
    source: { ...source },
    tags: [...closure.tags],
    templateReference: closure.template.reference,
    templatePath: `templates/${closure.template.reference}`,
    templateBytes,
    templateSha256: templateClosureSha256(closure.template.payload),
  };
}

function nodeSpreadPerLayer(
  layers: ReadonlyMap<number, AffineFisherResult>,
): Record<string, number> {
  const result: Record<string, number> = {};
  for (const [layer, fit] of [...layers].sort(([left], [right]) => left - right)) {
    let spread = 0;
    for (let node = 0; node < fit.nodeCount; node += 1) {
      spread += fit.whitenedGram[node * fit.nodeCount + node];
    }
    if (!Number.isFinite(spread)) {
      throw new TypeError(`Browser fitted publication has invalid node spread at layer ${layer}`);
    }
    result[String(layer)] = Math.max(0, spread);
  }
  return result;
}

function topologyDiagnostics(
  topology: TopologyResult,
): Record<string, unknown> {
  const diagnostics = topology.diagnostics;
  if (diagnostics.kind === "pca") {
    return {
      per_component_variance: [...diagnostics.perComponentVariance],
      cumulative_variance: [...diagnostics.cumulativeVariance],
      picked_k: diagnostics.pickedDimensions,
      threshold: diagnostics.threshold,
    };
  }
  return {
    eigenvalues: [...diagnostics.eigenvalues],
    picked_k: diagnostics.pickedDimensions,
    gap_index: diagnostics.pickedDimensions,
    gap_magnitude: diagnostics.gapMagnitude,
    bandwidth: diagnostics.bandwidth,
    k_nn: diagnostics.kNn,
    component_count: diagnostics.componentCount,
    heuristic_k: diagnostics.heuristicDimensions,
    pinned: diagnostics.pinned,
    ...(diagnostics.minDimensions === null ? {} : { min_dim: diagnostics.minDimensions }),
  };
}

function validateEvaluatedLayers(
  layers: ReadonlyMap<number, AffineFisherResult>,
  nodeCount: number,
  fittedLayers: ReadonlyMap<number, AffineFisherResult>,
): void {
  if (!(layers instanceof Map) || layers.size === 0) {
    throw new TypeError("Browser fitted publication has no evaluated layer diagnostics");
  }
  for (const layer of fittedLayers.keys()) {
    if (!layers.has(layer)) {
      throw new TypeError(`Browser fitted publication is missing evaluated layer ${layer}`);
    }
  }
  for (const [layer, fit] of layers) {
    if (
      !Number.isSafeInteger(layer) || layer < 0 || fit.nodeCount !== nodeCount ||
      fit.whitenedGram.length !== nodeCount * nodeCount || !finiteValues(fit.whitenedGram)
    ) {
      throw new TypeError(`Browser fitted publication has invalid evaluated layer ${layer}`);
    }
  }
}

function validateInput(input: BrowserFittedFlatDiscoverPackInput): void {
  if (input.manifold.fitMode !== "pca" && input.manifold.fitMode !== "auto") {
    throw new TypeError("Browser flat fitted publication requires a pca or auto manifold");
  }
  if (input.topology.fitMode !== "pca" || input.topology.intrinsicDimensions <= 0) {
    throw new TypeError("Browser flat fitted publication requires a flat PCA topology winner");
  }
  if (input.manifold.fitMode === "auto" && input.topology.candidates.length === 0) {
    throw new TypeError("Browser automatic fitted publication requires topology candidates");
  }
  if (!input.modelId || !input.producerVersion) {
    throw new TypeError("Browser flat fitted publication requires model and producer identities");
  }
  const { identity } = input;
  for (const [label, value] of [
    ["runtime identity", identity.runtimeIdentitySha256],
    ["context binding", identity.contextBindingSha256],
    ["capture identity", identity.captureSha256],
  ] as const) {
    if (!SHA256.test(value)) throw new TypeError(`Browser flat fit ${label} is invalid`);
  }
  for (const [label, value] of [
    ["model source fingerprint", identity.modelSourceFingerprint],
    ["capture render digest", identity.captureRenderSha256],
    ["baseline prompt digest", identity.baselinePromptsSha256],
  ] as const) {
    if (value !== null && value !== undefined && !SHA256.test(value)) {
      throw new TypeError(`Browser flat fit ${label} is invalid`);
    }
  }
  for (const [label, value] of [
    ["capture version", identity.captureVersion],
    ["fit policy version", identity.fitPolicyVersion],
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new TypeError(`Browser flat fit ${label} is invalid`);
    }
  }
  const nodeCount = input.manifold.nodes.length;
  const dimensions = input.topology.intrinsicDimensions;
  if (
    input.nodeCoordinates.length !== nodeCount * dimensions || input.layers.size === 0 ||
    input.consensusGram.length !== nodeCount * nodeCount || !finiteValues(input.consensusGram)
  ) {
    throw new TypeError("Browser flat fitted publication has incomplete fitted geometry");
  }
  validateEvaluatedLayers(input.evaluatedLayers, nodeCount, input.layers);
  for (const [layer, fit] of input.layers) {
    if (!Number.isSafeInteger(layer) || layer < 0 || fit.nodeCount !== nodeCount ||
      fit.components <= 0 || fit.components > dimensions || fit.columns < fit.components ||
      fit.mean.length !== fit.columns || fit.basis.length !== fit.components * fit.columns ||
      fit.nodeCoordinates.length !== nodeCount * fit.components ||
      !Number.isFinite(fit.mahalanobisShare) || fit.mahalanobisShare <= 0 ||
      (fit.affineMap === null
        ? fit.components !== dimensions
        : fit.affineMap.length !== dimensions * fit.components)
    ) {
      throw new TypeError(`Browser flat fitted publication has invalid layer ${layer}`);
    }
  }
}

function validateCurvedInput(input: BrowserFittedCurvedDiscoverPackInput): void {
  if (input.manifold.fitMode !== "spectral" && input.manifold.fitMode !== "auto") {
    throw new TypeError("Browser curved fitted publication requires a spectral or auto manifold");
  }
  if (input.topology.fitMode !== "spectral" || input.topology.intrinsicDimensions <= 0) {
    throw new TypeError("Browser curved fitted publication requires a curved topology winner");
  }
  if (input.manifold.fitMode === "auto" && input.topology.candidates.length === 0) {
    throw new TypeError("Browser automatic curved publication requires topology candidates");
  }
  if (!input.modelId || !input.producerVersion) {
    throw new TypeError("Browser curved fitted publication requires model and producer identities");
  }
  const { identity } = input;
  for (const [label, value] of [
    ["runtime identity", identity.runtimeIdentitySha256],
    ["context binding", identity.contextBindingSha256],
    ["capture identity", identity.captureSha256],
  ] as const) {
    if (!SHA256.test(value)) throw new TypeError(`Browser curved fit ${label} is invalid`);
  }
  for (const [label, value] of [
    ["model source fingerprint", identity.modelSourceFingerprint],
    ["capture render digest", identity.captureRenderSha256],
    ["baseline prompt digest", identity.baselinePromptsSha256],
  ] as const) {
    if (value !== null && value !== undefined && !SHA256.test(value)) {
      throw new TypeError(`Browser curved fit ${label} is invalid`);
    }
  }
  for (const [label, value] of [
    ["capture version", identity.captureVersion],
    ["fit policy version", identity.fitPolicyVersion],
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new TypeError(`Browser curved fit ${label} is invalid`);
    }
  }
  const nodeCount = input.manifold.nodes.length;
  const intrinsicDimensions = input.topology.intrinsicDimensions;
  const embeddedDimensions = input.topology.embeddedCoordinates.length / nodeCount;
  if (
    !Number.isSafeInteger(intrinsicDimensions) || intrinsicDimensions <= 0 ||
    input.topology.coordinates.length !== nodeCount * intrinsicDimensions ||
    input.consensusGram.length !== nodeCount * nodeCount ||
    !finiteValues(input.consensusGram) ||
    !Number.isSafeInteger(embeddedDimensions) || embeddedDimensions <= 0 ||
    input.layers.size === 0 ||
    (input.topology.periodicDimensions === 0
      ? embeddedDimensions !== intrinsicDimensions
      : input.topology.periodicDimensions !== intrinsicDimensions ||
        embeddedDimensions !== 2 * intrinsicDimensions)
  ) {
    throw new TypeError("Browser curved fitted publication has incomplete topology geometry");
  }
  validateEvaluatedLayers(input.evaluatedLayers, nodeCount, input.layers);
  for (const [layer, fit] of input.layers) {
    const surface = fit.surface;
    const sigma = fit.sigmaSurface;
    const sigmaSummary = fit.sigmaSummary;
    const sigmaValid = sigma !== null && sigmaSummary !== null &&
      sigma.nodeCount === nodeCount && sigma.inputDimensions === embeddedDimensions &&
      sigma.outputDimensions === 1 && sigma.weights.length === nodeCount &&
      sigma.polynomial.length === embeddedDimensions + 1 &&
      sameValues(surface.nodes, sigma.nodes) &&
      sameValues(surface.coordinateOffset, sigma.coordinateOffset) &&
      sameValues(surface.coordinateScale, sigma.coordinateScale) &&
      [sigma.weights, sigma.polynomial].every(finiteValues) &&
      [sigma.lambda, sigma.effectiveDegreesOfFreedom, sigma.gcv,
        sigmaSummary.mean, sigmaSummary.min, sigmaSummary.max,
        sigmaSummary.lambda].every(Number.isFinite) &&
      sigmaSummary.min > 0 && sigmaSummary.mean > 0 &&
      sigmaSummary.max >= sigmaSummary.min && sigmaSummary.lambda === sigma.lambda;
    if (
      !Number.isSafeInteger(layer) || layer < 0 || fit.nodeCount !== nodeCount ||
      fit.components <= 0 || fit.columns < fit.components ||
      fit.mean.length !== fit.columns || fit.basis.length !== fit.components * fit.columns ||
      !Number.isFinite(fit.mahalanobisShare) || fit.mahalanobisShare <= 0 ||
      surface.nodeCount !== nodeCount || surface.inputDimensions !== embeddedDimensions ||
      surface.outputDimensions !== fit.components ||
      surface.nodes.length !== nodeCount * embeddedDimensions ||
      surface.weights.length !== nodeCount * fit.components ||
      surface.polynomial.length !== (embeddedDimensions + 1) * fit.components ||
      surface.coordinateOffset.length !== embeddedDimensions ||
      surface.coordinateScale.length !== embeddedDimensions ||
      (input.sae == null ? !sigmaValid : sigma !== null || sigmaSummary !== null) ||
      fit.origin.length !== intrinsicDimensions || !Number.isFinite(fit.originDistance) ||
      fit.originDistance < 0 ||
      ![fit.origin, fit.mean, fit.basis, surface.nodes, surface.weights,
        surface.polynomial, surface.coordinateOffset, surface.coordinateScale].every(finiteValues) ||
      !surface.coordinateScale.every((value) => value > 0) ||
      ![surface.lambda, surface.effectiveDegreesOfFreedom, surface.gcv].every(Number.isFinite)
    ) {
      throw new TypeError(`Browser curved fitted publication has invalid layer ${layer}`);
    }
  }
}

function validateAuthoredInput(input: BrowserFittedAuthoredPackInput): void {
  if (!input.modelId || !input.producerVersion || input.manifold.nodes.length < 3) {
    throw new TypeError("Browser authored fitted publication is incomplete");
  }
  validateIdentity(input.identity, "authored");
  const intrinsicDimensions = domainDimensions(input.manifold.domain);
  const embeddedDimensions = embeddedDomainDimensions(input.manifold.domain);
  if (input.layers.size === 0 || input.manifold.nodes.some((node) =>
    node.coords.length !== intrinsicDimensions || node.coords.some((value) => !Number.isFinite(value))
  )) {
    throw new TypeError("Browser authored fitted publication has invalid node geometry");
  }
  validateEvaluatedLayers(input.evaluatedLayers, input.manifold.nodes.length, input.layers);
  for (const [layer, fit] of input.layers) {
    const surface = fit.surface;
    const sigma = fit.sigmaSurface;
    const sigmaSummary = fit.sigmaSummary;
    const sigmaValid = sigma !== null && sigmaSummary !== null &&
      sigma.nodeCount === fit.nodeCount && sigma.inputDimensions === embeddedDimensions &&
      sigma.outputDimensions === 1 && sigma.weights.length === fit.nodeCount &&
      sigma.polynomial.length === embeddedDimensions + 1 &&
      sameValues(surface.nodes, sigma.nodes) &&
      sameValues(surface.coordinateOffset, sigma.coordinateOffset) &&
      sameValues(surface.coordinateScale, sigma.coordinateScale) &&
      sigmaSummary.lambda === sigma.lambda && sigmaSummary.min > 0 &&
      sigmaSummary.mean > 0 && sigmaSummary.max >= sigmaSummary.min;
    if (
      !Number.isSafeInteger(layer) || layer < 0 || fit.nodeCount !== input.manifold.nodes.length ||
      fit.components <= 0 || fit.columns < fit.components ||
      fit.mean.length !== fit.columns || fit.basis.length !== fit.components * fit.columns ||
      !Number.isFinite(fit.mahalanobisShare) || fit.mahalanobisShare <= 0 ||
      surface.nodeCount !== fit.nodeCount || surface.inputDimensions !== embeddedDimensions ||
      surface.outputDimensions !== fit.components ||
      surface.nodes.length !== fit.nodeCount * embeddedDimensions ||
      surface.weights.length !== fit.nodeCount * fit.components ||
      surface.polynomial.length !== (embeddedDimensions + 1) * fit.components ||
      surface.coordinateOffset.length !== embeddedDimensions ||
      surface.coordinateScale.length !== embeddedDimensions ||
      fit.origin.length !== intrinsicDimensions || !Number.isFinite(fit.originDistance) ||
      fit.originDistance < 0 ||
      (input.sae == null ? !sigmaValid : sigma !== null || sigmaSummary !== null) ||
      ![fit.origin, fit.mean, fit.basis, surface.nodes, surface.weights,
        surface.polynomial, surface.coordinateOffset, surface.coordinateScale].every(finiteValues) ||
      !surface.coordinateScale.every((value) => value > 0)
    ) {
      throw new TypeError(`Browser authored fitted publication has invalid layer ${layer}`);
    }
  }
}

function validateIdentity(identity: BrowserFlatFitIdentity, kind: string): void {
  for (const [label, value] of [
    ["runtime identity", identity.runtimeIdentitySha256],
    ["context binding", identity.contextBindingSha256],
    ["capture identity", identity.captureSha256],
  ] as const) {
    if (!SHA256.test(value)) throw new TypeError(`Browser ${kind} fit ${label} is invalid`);
  }
  for (const [label, value] of [
    ["model source fingerprint", identity.modelSourceFingerprint],
    ["capture render digest", identity.captureRenderSha256],
    ["baseline prompt digest", identity.baselinePromptsSha256],
  ] as const) {
    if (value !== null && value !== undefined && !SHA256.test(value)) {
      throw new TypeError(`Browser ${kind} fit ${label} is invalid`);
    }
  }
  for (const [label, value] of [
    ["capture version", identity.captureVersion],
    ["fit policy version", identity.fitPolicyVersion],
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new TypeError(`Browser ${kind} fit ${label} is invalid`);
    }
  }
}

function fitProvenance(
  modelId: string,
  fittedLayers: readonly number[],
  sae: BrowserSaeFitIdentity | null | undefined,
): {
  tensorName: string;
  featureSpace: string;
  sae: BrowserSaeFitIdentity | null;
  saeIds: Record<string, string>;
} {
  const rawTensorName = drowseTensorFilename(modelId);
  if (sae == null) {
    return { tensorName: rawTensorName, featureSpace: "raw", sae: null, saeIds: {} };
  }
  if (
    !sae.release || !sae.revision || !sae.fingerprint ||
    typeof sae.fullCoverage !== "boolean" || !SHA256.test(sae.fingerprint)
  ) {
    throw new TypeError("Browser SAE fit provenance is invalid");
  }
  const saeIds: Record<string, string> = {};
  for (const layer of fittedLayers) {
    const id = sae.idsByLayer.get(layer);
    if (!id) throw new TypeError(`Browser SAE fit has no provenance ID for layer ${layer}`);
    saeIds[String(layer)] = id;
  }
  if ([...sae.idsByLayer.keys()].some((layer) => !fittedLayers.includes(layer))) {
    throw new TypeError("Browser SAE fit provenance contains an unfitted layer");
  }
  const variant = encodeReleaseId(sae.release);
  return {
    tensorName: rawTensorName.replace(/\.safetensors$/, `_sae-${variant}.safetensors`),
    featureSpace: `sae-${sae.release}`,
    sae,
    saeIds,
  };
}

function encodeReleaseId(value: string): string {
  const alphabet = "abcdefghijklmnopqrstuvwxyz234567";
  let accumulator = 0;
  let bits = 0;
  let output = "";
  for (const byte of new TextEncoder().encode(value)) {
    accumulator = (accumulator << 8) | byte;
    bits += 8;
    while (bits >= 5) {
      bits -= 5;
      output += alphabet[(accumulator >>> bits) & 31];
      accumulator &= (1 << bits) - 1;
    }
  }
  if (bits > 0) output += alphabet[(accumulator << (5 - bits)) & 31];
  return `_z${output}`;
}

function domainDimensions(domain: ManifoldDomain): number {
  requireBrowserDomain(domain);
  if (domain.type === "box") return domain.axes.length;
  if (domain.type === "sphere") return domain.dim;
  const dimensions = domain.embed_dim;
  if (!Number.isSafeInteger(dimensions) || (dimensions as number) <= 0) {
    throw new TypeError("Browser authored custom domain has an invalid embed_dim");
  }
  return dimensions as number;
}

function embeddedDomainDimensions(domain: ManifoldDomain): number {
  if (domain.type === "box") {
    return domain.axes.reduce((total, axis) => total + (axis.periodic ? 2 : 1), 0);
  }
  if (domain.type === "sphere") return domain.dim + 1;
  return domainDimensions(domain);
}

function authoredNodesSha256(
  manifold: BrowserAuthoredFittableManifold,
  nodeBytes: readonly Uint8Array[],
): string {
  const roles = manifold.nodes.map((node) => node.role);
  const kinds = manifold.nodes.map((node) => node.kind);
  const labels = manifold.nodes.map((node) => node.label);
  const coords = manifold.nodes.map((node) =>
    node.coords.map((value) => new ForcedPythonFloat(value))
  );
  return hashParts([
    pythonCanonicalJsonBytes(labels),
    ...nodeBytes,
    pythonCanonicalJsonBytes(pythonFloatDomain(manifold.domain)),
    pythonCanonicalJsonBytes(coords),
    pythonCanonicalJsonBytes(roles),
    pythonCanonicalJsonBytes(kinds),
  ]);
}

class ForcedPythonFloat {
  constructor(readonly value: number) {}
}

function pythonFloatDomain(value: ManifoldDomain): unknown {
  if (value.type === "box") {
    return {
      ...value,
      axes: value.axes.map((axis) => ({
        ...axis,
        hi: new ForcedPythonFloat(axis.hi),
        lo: new ForcedPythonFloat(axis.lo),
        period: new ForcedPythonFloat(axis.period),
      })),
    };
  }
  if (value.type === "custom" && Array.isArray(value.bounds)) {
    return {
      ...value,
      bounds: value.bounds.map((row) => Array.isArray(row)
        ? row.map((coordinate) => new ForcedPythonFloat(coordinate as number))
        : row),
    };
  }
  return value;
}

function hashParts(parts: readonly Uint8Array[]): string {
  const hasher = sha256.create();
  for (const part of parts) hasher.update(part);
  return bytesToHex(hasher.digest());
}

function pythonCanonicalJsonBytes(value: unknown): Uint8Array {
  return new TextEncoder().encode(pythonCanonicalJson(value));
}

function pythonCanonicalJson(value: unknown): string {
  if (value instanceof ForcedPythonFloat) return pythonFloatRepr(value.value);
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return pythonJsonString(value);
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new TypeError("canonical JSON requires finite numbers");
    return Number.isInteger(value) ? String(value) : pythonFloatRepr(value);
  }
  if (Array.isArray(value)) return `[${value.map(pythonCanonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) =>
      `${pythonJsonString(key)}:${pythonCanonicalJson(record[key])}`
    ).join(",")}}`;
  }
  throw new TypeError("canonical JSON contains an unsupported value");
}

function pythonJsonString(value: string): string {
  let output = '"';
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code === 0x22) output += '\\"';
    else if (code === 0x5c) output += "\\\\";
    else if (code === 0x08) output += "\\b";
    else if (code === 0x0c) output += "\\f";
    else if (code === 0x0a) output += "\\n";
    else if (code === 0x0d) output += "\\r";
    else if (code === 0x09) output += "\\t";
    else if (code < 0x20 || code > 0x7e) output += `\\u${code.toString(16).padStart(4, "0")}`;
    else output += value[index];
  }
  return `${output}"`;
}

function pythonFloatRepr(value: number): string {
  if (!Number.isFinite(value)) throw new TypeError("Python float repr requires a finite value");
  if (value === 0) return Object.is(value, -0) ? "-0.0" : "0.0";
  const sign = value < 0 ? "-" : "";
  const raw = Math.abs(value).toString().toLowerCase();
  let digits: string;
  let exponent: number;
  if (raw.includes("e")) {
    const [mantissa, rawExponent] = raw.split("e");
    digits = mantissa.replace(".", "");
    exponent = Number(rawExponent);
  } else if (raw.includes(".")) {
    const [integer, fraction] = raw.split(".");
    if (integer !== "0") {
      digits = integer + fraction;
      exponent = integer.length - 1;
    } else {
      const leading = /^0*/.exec(fraction)![0].length;
      digits = fraction.slice(leading);
      exponent = -(leading + 1);
    }
  } else {
    digits = raw;
    exponent = raw.length - 1;
  }
  digits = digits.replace(/^0+/, "").replace(/0+$/, "") || "0";
  if (exponent >= -4 && exponent < 16) {
    const point = exponent + 1;
    if (point <= 0) return `${sign}0.${"0".repeat(-point)}${digits}`;
    if (point >= digits.length) return `${sign}${digits}${"0".repeat(point - digits.length)}.0`;
    return `${sign}${digits.slice(0, point)}.${digits.slice(point)}`;
  }
  const mantissa = digits.length === 1 ? digits : `${digits[0]}.${digits.slice(1)}`;
  return `${sign}${mantissa}e${exponent >= 0 ? "+" : "-"}${Math.abs(exponent).toString().padStart(2, "0")}`;
}

function sameValues(left: Float64Array, right: Float64Array): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function finiteValues(values: Float64Array): boolean {
  return values.every(Number.isFinite);
}

function fp32(values: Float64Array, label: string): Float32Array {
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    output[index] = values[index];
    if (!Number.isFinite(values[index]) || !Number.isFinite(output[index])) {
      throw new TypeError(`Browser fitted ${label} contains a non-finite fp32 value`);
    }
  }
  return output;
}
import { requireBrowserDomain } from "../../lib/manifolds/surfaceGeometry";
