import type {
  CreateDiscoverManifoldRequest,
  CreateManifoldFromTemplateRequest,
  CreateManifoldRequest,
  CreateTemplateRequest,
  ManifoldDomain,
} from "../../lib/types";
import type {
  ManifoldInfo,
  ManifoldFitInfo,
  ManifoldNodeDetail,
  TemplateDetail,
  TemplateSummary,
} from "../../lib/types.gen";
import { buildDrowseArchive, jsonBytes } from "./builder";
import { parseDrowseTensorFilename } from "./drowseArchive";

const DEFAULT_NAMESPACE = "local";

interface StoredClosure {
  manifold: Record<string, unknown>;
  nodeCorpora: string[][];
  fitted: ManifoldFitInfo[];
}

export async function authoredManifoldPack(
  request: CreateManifoldRequest,
  producerVersion: string,
): Promise<Blob> {
  const namespace = request.namespace ?? DEFAULT_NAMESPACE;
  const primary = `manifolds/${namespace}/${request.name}`;
  const manifold = {
    format_version: 10,
    name: request.name,
    description: request.description,
    fit_mode: "authored",
    domain: request.domain,
    nodes: request.nodes.map((node) => ({
      label: node.label,
      coords: node.coords,
      role: node.role ?? null,
      kind: null,
    })),
    files: {},
    source: "local",
    tags: [],
    template_ref: null,
  };
  return buildDrowseArchive({
    primary,
    template: null,
    producerVersion,
    source: localSource(),
    files: closureFiles(primary, manifold, request.nodes.map((node) => ({
      label: node.label,
      statements: node.statements,
    }))),
  });
}

export function authoredManifoldAdvisories(request: CreateManifoldRequest): string[] {
  const embedded = embedCoordinates(request.domain, request.nodes.map((node) => node.coords));
  const rank = affineRank(embedded);
  const embedDimensions = embedded[0]?.length ?? 0;
  const advisories: string[] = [];
  if (rank !== embedDimensions) {
    advisories.push(
      `manifold '${request.name}': node coordinates have affine rank ${rank} ` +
      `but the embedding has ${embedDimensions} dimensions. These points cannot define the fit; ` +
      "spread the nodes across every axis",
    );
  }
  if (request.domain.type === "box") {
    request.domain.axes.forEach((axis, index) => {
      if (axis.periodic) return;
      const distinct = new Set(request.nodes.map((node) =>
        Math.round(node.coords[index] * 1_000_000_000) / 1_000_000_000
      ));
      if (distinct.size < 3) {
        advisories.push(
          `manifold '${request.name}': axis '${axis.name}' has only ${distinct.size} ` +
          "distinct coordinate value(s). There aren't enough to fit curvature along this axis",
        );
      }
    });
  }
  return advisories;
}

export async function discoverManifoldPack(
  request: CreateDiscoverManifoldRequest,
  producerVersion: string,
): Promise<Blob> {
  const namespace = request.namespace ?? DEFAULT_NAMESPACE;
  const primary = `manifolds/${namespace}/${request.name}`;
  const manifold = {
    format_version: 10,
    name: request.name,
    description: request.description ?? "",
    fit_mode: request.fit_mode,
    hyperparams: request.hyperparams ?? {},
    nodes: request.nodes.map((node) => ({
      label: node.label,
      role: node.role ?? null,
      kind: null,
    })),
    files: {},
    source: "local",
    tags: [],
    template_ref: null,
  };
  return buildDrowseArchive({
    primary,
    template: null,
    producerVersion,
    source: localSource(),
    files: closureFiles(primary, manifold, request.nodes),
  });
}

export interface BrowserMergedDiscoverManifold {
  namespace: string;
  name: string;
  description: string;
  fitMode: "pca" | "spectral" | "auto";
  hyperparams: Record<string, number | string>;
  nodes: ReadonlyArray<{
    label: string;
    statements: readonly string[];
    role: string | null;
    kind: string | null;
  }>;
}

export async function mergedDiscoverManifoldPack(
  request: BrowserMergedDiscoverManifold,
  producerVersion: string,
): Promise<Blob> {
  const primary = `manifolds/${request.namespace}/${request.name}`;
  const manifold = {
    format_version: 10,
    name: request.name,
    description: request.description,
    fit_mode: request.fitMode,
    hyperparams: request.hyperparams,
    nodes: request.nodes.map((node) => ({
      label: node.label,
      role: node.role,
      kind: node.kind,
    })),
    files: {},
    source: "local",
    tags: [],
    template_ref: null,
  };
  return buildDrowseArchive({
    primary,
    template: null,
    producerVersion,
    source: localSource(),
    files: closureFiles(primary, manifold, request.nodes),
  });
}

export async function templateManifoldPack(
  request: CreateManifoldFromTemplateRequest,
  templateNamespace: string,
  template: TemplateDetail,
  producerVersion: string,
): Promise<Blob> {
  const namespace = request.namespace ?? DEFAULT_NAMESPACE;
  const primary = `manifolds/${namespace}/${request.name}`;
  const templatePath = `templates/${templateNamespace}/${template.name}`;
  const labels = template.labels;
  const corpora = labels.map((_label, index) => template.contexts.map((context) =>
    context.assistant.replace(template.slot, template.values[index])
  ));
  const manifold = {
    format_version: 10,
    name: request.name,
    description: request.description ?? "",
    fit_mode: request.fit_mode,
    hyperparams: request.hyperparams ?? {},
    nodes: labels.map((label) => ({ label, role: null, kind: null })),
    files: {},
    source: "local",
    tags: [],
    template_ref: `${templateNamespace}/${template.name}`,
  };
  const templatePayload = templatePayloadFromDetail(template);
  return buildDrowseArchive({
    primary,
    template: templatePath,
    producerVersion,
    source: localSource(),
    files: {
      ...closureFiles(primary, manifold, labels.map((label, index) => ({
        label,
        statements: corpora[index],
      }))),
      [`${templatePath}/template.json`]: jsonBytes(templatePayload),
    },
  });
}

export function templatePayload(request: CreateTemplateRequest): Record<string, unknown> {
  return {
    format_version: 2,
    name: request.name,
    slot: request.slot,
    values: request.values,
    contexts: request.contexts,
    description: request.description ?? "",
    source: "local",
    tags: request.tags ?? [],
  };
}

export function templateDetail(
  namespace: string,
  value: Record<string, unknown>,
): TemplateDetail {
  const values = value.values as string[];
  const contexts = value.contexts as TemplateDetail["contexts"];
  return {
    namespace,
    name: value.name as string,
    slot: value.slot as string,
    n_values: values.length,
    n_contexts: contexts.length,
    values: [...values],
    labels: values.map(slugValue),
    description: value.description as string,
    tags: [...value.tags as string[]],
    contexts: structuredClone(contexts),
    source: value.source,
  };
}

export function templateSummary(detail: TemplateDetail): TemplateSummary {
  const { contexts: _contexts, ...summary } = detail;
  return summary;
}

export function manifoldInfo(
  namespace: string,
  closure: StoredClosure,
  full: boolean,
): ManifoldInfo {
  const value = closure.manifold;
  const fitMode = value.fit_mode as string;
  const isDiscover = fitMode === "pca" || fitMode === "spectral" || fitMode === "auto";
  const hasAuthoredGeometry = fitMode === "authored";
  const nodes = value.nodes as Array<{
    label: string;
    coords?: number[];
    role: string | null;
    kind: string | null;
  }>;
  const domain = (hasAuthoredGeometry ? value.domain : {}) as ManifoldDomain;
  const intrinsicDim = hasAuthoredGeometry ? domainDimension(domain) : 0;
  const fittedModels: string[] = [];
  const tensorVariants: Record<string, string[]> = {};
  for (const filename of Object.keys(value.files as Record<string, string>)) {
    if (!filename.endsWith(".safetensors")) continue;
    const parsed = parseDrowseTensorFilename(filename);
    if (!parsed) continue;
    if (!fittedModels.includes(parsed.safeModelId)) fittedModels.push(parsed.safeModelId);
    const variant = parsed.variant === "raw"
      ? "raw"
      : `${parsed.variant}-${parsed.safeVariantIdentity}`;
    (tensorVariants[parsed.safeModelId] ??= []).push(variant);
  }
  const info: ManifoldInfo = {
    namespace,
    name: value.name as string,
    description: value.description as string,
    source: value.source as string,
    tags: [...value.tags as string[]],
    template_ref: value.template_ref as string | null,
    fit_mode: fitMode,
    is_discover: isDiscover,
    domain,
    domain_label: hasAuthoredGeometry ? domainLabel(domain) : `discover-${fitMode}`,
    intrinsic_dim: intrinsicDim,
    min_nodes: intrinsicDim > 0 ? 2 * intrinsicDim + 1 : null,
    node_count: nodes.length,
    node_labels: nodes.map((node) => node.label),
    node_coords: hasAuthoredGeometry ? nodes.map((node) => [...node.coords!]) : [],
    node_roles: nodes.map((node) => node.role),
    node_kinds: nodes.map((node) => node.kind),
    hyperparams: structuredClone((value.hyperparams ?? {}) as Record<string, number | string>),
    fitted_models: fittedModels,
    tensor_variants: tensorVariants,
    fitted_for_session: false,
    stale: false,
    resolved_fit_mode: isDiscover ? null : fitMode,
  };
  if (full) {
    info.nodes = nodes.map((node, index): ManifoldNodeDetail => ({
      label: node.label,
      coords: hasAuthoredGeometry && node.coords ? [...node.coords] : null,
      statements: [...(closure.nodeCorpora[index] ?? [])],
      role: node.role,
    }));
    info.fitted = structuredClone(closure.fitted);
  }
  return info;
}

export function parseJsonFile(bytes: Blob, path: string): Promise<Record<string, unknown>> {
  return bytes.text().then((text) => {
    const value: unknown = JSON.parse(text);
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
      throw new Error(`${path} is not a JSON object`);
    }
    return value as Record<string, unknown>;
  });
}

export function nodePath(primary: string, index: number, label: string): string {
  return `${primary}/nodes/${String(index).padStart(2, "0")}_${label}.json`;
}

function closureFiles(
  primary: string,
  manifold: Record<string, unknown>,
  nodes: ReadonlyArray<{ label: string; statements: readonly string[] }>,
): Record<string, Uint8Array> {
  const files: Record<string, Uint8Array> = {
    [`${primary}/manifold.json`]: jsonBytes(manifold),
  };
  nodes.forEach((node, index) => {
    files[nodePath(primary, index, node.label)] = jsonBytes(node.statements);
  });
  return files;
}

function localSource() {
  return { uri: "local", repository: null, revision: null } as const;
}

export function templatePayloadFromDetail(detail: TemplateDetail): Record<string, unknown> {
  return {
    format_version: 2,
    name: detail.name,
    slot: detail.slot,
    values: detail.values,
    contexts: detail.contexts,
    description: detail.description,
    source: typeof detail.source === "string" ? detail.source : "local",
    tags: detail.tags,
  };
}

function slugValue(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

function domainDimension(domain: ManifoldDomain): number {
  requireBrowserDomain(domain);
  if (domain.type === "box") return domain.axes.length;
  if (domain.type === "sphere") return domain.dim;
  return Number(domain.embed_dim ?? 0);
}

function domainLabel(domain: ManifoldDomain): string {
  return `${domain.type}(${domainDimension(domain)}d)`;
}

function embedCoordinates(domain: ManifoldDomain, coordinates: number[][]): number[][] {
  requireBrowserDomain(domain);
  if (domain.type === "custom") return coordinates.map((row) => row.map(Math.fround));
  if (domain.type === "box") {
    return coordinates.map((row) => {
      const embedded: number[] = [];
      domain.axes.forEach((axis, index) => {
        const value = Math.fround(row[index]);
        if (axis.periodic) {
          const angle = Math.fround(2 * Math.PI / axis.period * value);
          embedded.push(Math.fround(Math.cos(angle)), Math.fround(Math.sin(angle)));
        } else {
          embedded.push(value);
        }
      });
      return embedded;
    });
  }
  return coordinates.map((row) => {
    let running = 1;
    const embedded: number[] = [];
    for (let index = 0; index < domain.dim; index += 1) {
      embedded.push(Math.fround(running * Math.cos(row[index])));
      running = Math.fround(running * Math.sin(row[index]));
    }
    embedded.push(Math.fround(running));
    return embedded;
  });
}

function affineRank(values: number[][]): number {
  if (values.length === 0 || values[0].length === 0) return 0;
  const columns = values[0].length;
  const means = Array.from({ length: columns }, (_, column) => Math.fround(
    values.reduce((total, row) => total + row[column], 0) / values.length,
  ));
  const matrix = values.map((row) => row.map((value, column) =>
    Math.fround(value - means[column])
  ));
  const maximum = Math.max(...matrix.flat().map(Math.abs), 0);
  const tolerance = maximum * Math.max(matrix.length, columns) * 1.1920928955078125e-7;
  let rank = 0;
  for (let column = 0; column < columns && rank < matrix.length; column += 1) {
    let pivot = rank;
    for (let row = rank + 1; row < matrix.length; row += 1) {
      if (Math.abs(matrix[row][column]) > Math.abs(matrix[pivot][column])) pivot = row;
    }
    if (Math.abs(matrix[pivot][column]) <= tolerance) continue;
    [matrix[rank], matrix[pivot]] = [matrix[pivot], matrix[rank]];
    for (let row = rank + 1; row < matrix.length; row += 1) {
      const factor = matrix[row][column] / matrix[rank][column];
      for (let next = column; next < columns; next += 1) {
        matrix[row][next] = Math.fround(matrix[row][next] - factor * matrix[rank][next]);
      }
    }
    rank += 1;
  }
  return rank;
}
import { requireBrowserDomain } from "../../lib/manifolds/surfaceGeometry";
