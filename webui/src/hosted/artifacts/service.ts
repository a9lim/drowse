import type { RuntimeProgressEvent, RuntimeServiceRequest } from "../../lib/runtime/contracts";
import type {
  CreateDiscoverManifoldRequest,
  CreateManifoldFromTemplateRequest,
  CreateManifoldRequest,
  CreateTemplateRequest,
  InstallManifoldRequest,
  ManifoldInfo,
  MergeManifoldRequest,
} from "../../lib/types";
import type { ManifoldFitInfo } from "../../lib/types.gen";
import {
  BrowserDrowseArchiveRepository,
  type InstalledDrowseArchive,
} from "./repository";
import {
  authoredManifoldPack,
  authoredManifoldAdvisories,
  discoverManifoldPack,
  manifoldInfo,
  mergedDiscoverManifoldPack,
  nodePath,
  parseJsonFile,
  templateDetail,
  templateManifoldPack,
  templatePayload,
  templateSummary,
} from "./authoring";
import {
  BrowserTemplateRepository,
  type BrowserTemplateRepositoryPort,
  type StoredBrowserTemplate,
} from "./templateRepository";
import {
  BrowserHfManifoldClient,
  type BrowserHfManifoldClientPort,
} from "./hfManifolds";
import {
  HOSTED_DISTRIBUTION_CONFIG,
  type HostedDistributionConfig,
} from "../runtime/distributionConfig";
import { browserArtifactProducer } from "../runtime/buildProvenance";

export const DROWSE_ARCHIVE_SERVICE_METHODS = new Set([
  "drowseArchiveList",
  "drowseArchiveInstall",
  "drowseArchiveExport",
  "drowseArchiveDelete",
]);

const LOCAL_MANIFOLD_METHODS = new Set([
  "list",
  "get",
  "create",
  "createDiscover",
  "createFromTemplate",
  "delete",
  "install",
  "merge",
  "search",
]);

const LOCAL_TEMPLATE_METHODS = new Set(["list", "get", "create", "delete"]);

export interface DrowseArchiveRepositoryPort {
  initialize(): Promise<void>;
  install(
    source: Blob | ArrayBuffer | Uint8Array,
    options?: {
      force?: boolean;
      expectedSource?: { repository: string; revision: string };
      signal?: AbortSignal;
      onProgress?: (progress: {
        phase: "inspecting" | "verifying" | "validating" | "committing";
        path: string | null;
        verifiedBytes: number;
        totalBytes: number;
      }) => void;
    },
  ): Promise<InstalledDrowseArchive>;
  list(): Promise<InstalledDrowseArchive[]>;
  remove(primary: string): Promise<boolean>;
  file(primary: string, path: string): Promise<Blob | null>;
  export(primary: string): Promise<File>;
  withExport<T>(
    primary: string,
    consumer: (file: File) => Promise<T>,
    signal?: AbortSignal,
  ): Promise<T>;
  clear(): Promise<void>;
  close(): Promise<void>;
}

export function hostedHfManifoldDownloadOrigins(
  config: Pick<
    HostedDistributionConfig,
    | "catalogUrl"
    | "signatureUrl"
    | "allowedCatalogRedirectOrigins"
    | "allowedArtifactRedirectOrigins"
  > = HOSTED_DISTRIBUTION_CONFIG,
): ReadonlySet<string> {
  return new Set([
    new URL(config.catalogUrl).origin,
    new URL(config.signatureUrl).origin,
    ...config.allowedCatalogRedirectOrigins,
    ...config.allowedArtifactRedirectOrigins,
  ]);
}

export class BrowserDrowseArchiveService {
  private readonly repository: DrowseArchiveRepositoryPort;
  private readonly templates: BrowserTemplateRepositoryPort;
  private readonly huggingFace: BrowserHfManifoldClientPort;
  private readonly producerVersion: string;

  constructor(
    repository: DrowseArchiveRepositoryPort = new BrowserDrowseArchiveRepository(),
    templates: BrowserTemplateRepositoryPort = new BrowserTemplateRepository(),
    producerVersion = browserArtifactProducer().producerVersion,
    huggingFace: BrowserHfManifoldClientPort = new BrowserHfManifoldClient({
      allowedDownloadOrigins: hostedHfManifoldDownloadOrigins(),
    }),
  ) {
    this.repository = repository;
    this.templates = templates;
    this.producerVersion = producerVersion;
    this.huggingFace = huggingFace;
  }

  handles(request: RuntimeServiceRequest): boolean {
    return request.service === "manifolds" &&
        (DROWSE_ARCHIVE_SERVICE_METHODS.has(request.method) || LOCAL_MANIFOLD_METHODS.has(request.method)) ||
      request.service === "templates" && LOCAL_TEMPLATE_METHODS.has(request.method);
  }

  async withDrowseArchiveArchives<T>(
    ids: readonly string[],
    signal: AbortSignal,
    consumer: (archives: readonly Blob[]) => Promise<T>,
  ): Promise<T> {
    await this.repository.initialize();
    const archives: Blob[] = [];
    const visit = async (index: number): Promise<T> => {
      signal.throwIfAborted();
      if (index === ids.length) return consumer(archives);
      const id = primary(ids[index]);
      return this.repository.withExport(id, async (archive) => {
        signal.throwIfAborted();
        archives.push(archive);
        try {
          return await visit(index + 1);
        } finally {
          archives.pop();
        }
      }, signal);
    };
    return visit(0);
  }

  async request(
    request: RuntimeServiceRequest,
    onProgress: (event: RuntimeProgressEvent) => void,
  ): Promise<unknown> {
    if (!this.handles(request)) {
      throw new Error(`Unsupported Drowse artifact request ${request.service}.${request.method}`);
    }
    await this.repository.initialize();
    if (request.service === "templates") {
      await this.templates.initialize();
      return this.templateRequest(request);
    }
    switch (request.method) {
      case "list":
        requireArgs(request.args, 0);
        {
          const packs = await this.repository.list();
          return {
            manifolds: await Promise.all(packs.map((pack) =>
              this.readManifold(pack.id, false, pack)
            )),
          };
        }
      case "get": {
        requireArgs(request.args, 2);
        return this.readManifold(identity(request.args[0], request.args[1]), true);
      }
      case "create": {
        requireArgs(request.args, 1);
        const body = request.args[0] as CreateManifoldRequest;
        const advisories = authoredManifoldAdvisories(body);
        const archive = await authoredManifoldPack(body, this.producerVersion);
        const installed = await this.repository.install(archive);
        const result = await this.readManifold(installed.id, true);
        return { ...result, advisories };
      }
      case "createDiscover": {
        requireArgs(request.args, 1);
        const archive = await discoverManifoldPack(
          request.args[0] as CreateDiscoverManifoldRequest,
          this.producerVersion,
        );
        const installed = await this.repository.install(archive);
        return this.readManifold(installed.id, true);
      }
      case "createFromTemplate": {
        requireArgs(request.args, 1);
        await this.templates.initialize();
        const body = request.args[0] as CreateManifoldFromTemplateRequest;
        const stored = await this.resolveTemplate(body.template_ref);
        const detail = templateDetail(stored.namespace, stored.payload);
        const archive = await templateManifoldPack(
          body,
          stored.namespace,
          detail,
          this.producerVersion,
        );
        const installed = await this.repository.install(archive, { force: body.force === true });
        return this.readManifold(installed.id, true);
      }
      case "merge": {
        requireArgs(request.args, 1);
        const body = mergeRequest(request.args[0]);
        const sources = await Promise.all(body.sources.map(async (source) => {
          const detail = await this.readManifold(
            identity(source.namespace, source.name),
            true,
          );
          if (!isDiscoverMode(detail.fit_mode)) {
            throw new Error(
              `Merge source ${source.namespace}/${source.name} is authored; only discover manifolds can be merged`,
            );
          }
          return detail;
        }));
        const labels = new Map<string, string>();
        for (const source of sources) {
          for (const label of source.node_labels) {
            const owner = labels.get(label);
            if (owner) {
              throw new Error(
                `Merge label ${JSON.stringify(label)} appears in ${owner} and ${source.namespace}/${source.name}`,
              );
            }
            labels.set(label, `${source.namespace}/${source.name}`);
          }
        }
        const modes = [...new Set(sources.map((source) => source.fit_mode))].sort();
        if (!body.fit_mode && modes.length !== 1) {
          throw new Error(
            `Merge sources use mixed fit modes (${modes.join(", ")}); choose a target fit mode`,
          );
        }
        const fitMode = body.fit_mode ?? modes[0];
        if (!isDiscoverMode(fitMode)) throw new TypeError("Merge target fit mode is invalid");
        const nodes = sources.flatMap((source) => source.nodes!.map((node, index) => ({
          label: node.label,
          statements: node.statements,
          role: node.role,
          kind: source.node_kinds[index] ?? null,
        })));
        const archive = await mergedDiscoverManifoldPack({
          namespace: body.namespace,
          name: body.name,
          description: body.description,
          fitMode,
          hyperparams: fitHyperparams(body.hyperparams ?? sources[0].hyperparams),
          nodes,
        }, this.producerVersion);
        const installed = await this.repository.install(archive, { force: body.force });
        return this.readManifold(installed.id, true);
      }
      case "search": {
        if (request.args.length < 1 || request.args.length > 2) {
          throw new TypeError("Drowse Hugging Face search expects a query and optional limit");
        }
        return this.huggingFace.search(
          request.args[0] as string,
          request.args[1] === undefined ? undefined : request.args[1] as number,
        );
      }
      case "install": {
        requireArgs(request.args, 1);
        const body = hfInstallRequest(request.args[0]);
        return this.huggingFace.withArchive(body.target, {
          onProgress: (progress) => onProgress({ event: "progress", data: progress }),
        }, async (archive) => {
          const installed = await this.repository.install(archive.file, {
            force: body.force,
            expectedSource: {
              repository: archive.repository,
              revision: archive.revision,
            },
            onProgress: (progress) => onProgress({ event: "progress", data: progress }),
          });
          return this.readManifold(installed.id, true, installed);
        });
      }
      case "delete": {
        requireArgs(request.args, 2);
        const [namespace, name] = identityParts(request.args[0], request.args[1]);
        return {
          namespace,
          name,
          removed: await this.repository.remove(`manifolds/${namespace}/${name}`),
        };
      }
      case "drowseArchiveList":
        requireArgs(request.args, 0);
        return { packs: await this.repository.list() };
      case "drowseArchiveInstall": {
        requireArgs(request.args, 2);
        const [source, rawOptions] = request.args;
        if (!(source instanceof Blob)) {
          throw new TypeError("A browser Drowse pack install requires a Blob");
        }
        const options = installOptions(rawOptions);
        return this.repository.install(source, {
          force: options.force,
          onProgress: (progress) => onProgress({ event: "progress", data: progress }),
        });
      }
      case "drowseArchiveExport": {
        requireArgs(request.args, 1);
        return this.repository.export(primary(request.args[0]));
      }
      case "drowseArchiveDelete": {
        requireArgs(request.args, 1);
        const id = primary(request.args[0]);
        return { id, removed: await this.repository.remove(id) };
      }
    }
    throw new Error(`Unsupported Drowse artifact method ${request.method}`);
  }

  clear(): Promise<void> {
    return Promise.all([
      this.repository.clear(),
      this.templates.clear(),
      this.huggingFace.clear(),
    ]).then(() => undefined);
  }

  async close(): Promise<void> {
    await Promise.all([this.repository.close(), this.templates.close()]);
  }

  private async templateRequest(request: RuntimeServiceRequest): Promise<unknown> {
    switch (request.method) {
      case "list":
        requireArgs(request.args, 0);
        return {
          templates: (await this.allTemplates()).map((stored) =>
            templateSummary(templateDetail(stored.namespace, stored.payload))
          ),
        };
      case "get": {
        requireArgs(request.args, 2);
        const [namespace, name] = identityParts(request.args[0], request.args[1]);
        const stored = await this.resolveExactTemplate(`${namespace}/${name}`);
        if (!stored) throw new Error(`Template ${namespace}/${name} not found`);
        return templateDetail(namespace, stored.payload);
      }
      case "create": {
        requireArgs(request.args, 1);
        const body = request.args[0] as CreateTemplateRequest;
        const namespace = body.namespace ?? "local";
        const payload = templatePayload(body);
        const detail = templateDetail(namespace, payload);
        await templateManifoldPack({
          namespace: "local",
          name: "template_validation",
          fit_mode: "pca",
          template_ref: `${namespace}/${body.name}`,
        }, namespace, detail, this.producerVersion);
        await this.templates.put(namespace, body.name, payload, body.force === true);
        return detail;
      }
      case "delete": {
        requireArgs(request.args, 2);
        const [namespace, name] = identityParts(request.args[0], request.args[1]);
        return {
          namespace,
          name,
          removed: await this.templates.remove(`${namespace}/${name}`),
        };
      }
    }
    throw new Error(`Unsupported Drowse template method ${request.method}`);
  }

  private async resolveTemplate(reference: string): Promise<StoredBrowserTemplate> {
    const parts = reference.split("/");
    if (parts.length === 2) {
      const stored = await this.resolveExactTemplate(`${parts[0]}/${parts[1]}`);
      if (!stored) throw new Error(`Template ${reference} not found`);
      return stored;
    }
    if (parts.length !== 1) throw new TypeError("Template reference must be name or namespace/name");
    const matches = (await this.allTemplates()).filter((stored) => stored.name === reference);
    if (matches.length === 0) throw new Error(`Template ${reference} not found`);
    if (matches.length > 1) throw new Error(`Template ${reference} is ambiguous; use namespace/name`);
    return matches[0];
  }

  private async resolveExactTemplate(id: string): Promise<StoredBrowserTemplate | null> {
    return await this.templates.get(id) ??
      (await this.embeddedTemplates()).find((stored) => stored.id === id) ?? null;
  }

  private async allTemplates(): Promise<StoredBrowserTemplate[]> {
    const merged = new Map((await this.embeddedTemplates()).map((stored) => [stored.id, stored]));
    for (const stored of await this.templates.list()) merged.set(stored.id, stored);
    return [...merged.values()].sort((left, right) => left.id.localeCompare(right.id));
  }

  private async embeddedTemplates(): Promise<StoredBrowserTemplate[]> {
    const templates = new Map<string, StoredBrowserTemplate>();
    for (const pack of await this.repository.list()) {
      if (pack.template === null) continue;
      const path = `${pack.template}/template.json`;
      const file = await this.repository.file(pack.id, path);
      if (!file) throw new Error(`Installed Drowse pack ${pack.id} is missing ${path}`);
      const payload = await parseJsonFile(file, path);
      const [, namespace, name] = pack.template.split("/");
      const id = `${namespace}/${name}`;
      const current = templates.get(id);
      if (current && JSON.stringify(current.payload) !== JSON.stringify(payload)) {
        throw new Error(`Template ${id} has conflicting installed closures`);
      }
      templates.set(id, {
        id,
        namespace,
        name,
        payload,
        installedAt: pack.installedAt,
      });
    }
    return [...templates.values()].sort((left, right) => left.id.localeCompare(right.id));
  }

  private async readManifold(
    primary: string,
    full: boolean,
    installed?: InstalledDrowseArchive,
  ): Promise<ManifoldInfo & { archive_source: InstalledDrowseArchive["source"] }> {
    const pack = installed ?? (await this.repository.list()).find((candidate) =>
      candidate.id === primary
    );
    if (!pack) throw new Error(`Installed Drowse pack ${primary} is unavailable`);
    const [, namespace] = primary.split("/");
    const metadataFile = await this.repository.file(primary, `${primary}/manifold.json`);
    if (!metadataFile) throw new Error(`Installed Drowse pack ${primary} is missing manifold.json`);
    const manifold = await parseJsonFile(metadataFile, `${primary}/manifold.json`);
    const nodes = manifold.nodes as Array<{ label: string }>;
    const nodeCorpora = full && manifold.fit_mode !== "baked"
      ? await Promise.all(nodes.map(async (node, index) => {
          const path = nodePath(primary, index, node.label);
          const file = await this.repository.file(primary, path);
          if (!file) throw new Error(`Installed Drowse pack ${primary} is missing ${path}`);
          const value: unknown = JSON.parse(await file.text());
          if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) {
            throw new Error(`Installed Drowse pack ${primary} has an invalid node corpus`);
          }
          return value as string[];
        }))
      : nodes.map(() => []);
    const fitted: ManifoldFitInfo[] = [];
    if (full) {
      for (const filename of Object.keys(manifold.files as Record<string, string>)) {
        if (!filename.endsWith(".safetensors")) continue;
        const sidecarName = filename.replace(/\.safetensors$/, ".json");
        const path = `${primary}/${sidecarName}`;
        const file = await this.repository.file(primary, path);
        if (!file) throw new Error(`Installed Drowse pack ${primary} is missing ${path}`);
        const sidecar = await parseJsonFile(file, path);
        const entry: ManifoldFitInfo = {
          stem: filename.replace(/\.safetensors$/, ""),
          method: String(sidecar.method),
          feature_space: String(sidecar.feature_space),
          node_count: Number(sidecar.node_count),
          nodes_sha256: typeof sidecar.nodes_sha256 === "string" ? sidecar.nodes_sha256 : null,
          fit_mode: String(sidecar.fit_mode),
        };
        if (sidecar.hyperparams && typeof sidecar.hyperparams === "object") {
          entry.hyperparams = structuredClone(sidecar.hyperparams as Record<string, number | string>);
        }
        if (sidecar.diagnostics && typeof sidecar.diagnostics === "object") {
          entry.diagnostics = structuredClone(sidecar.diagnostics) as ManifoldFitInfo["diagnostics"];
        }
        fitted.push(entry);
      }
    }
    return {
      ...manifoldInfo(namespace, { manifold, nodeCorpora, fitted }, full),
      archive_source: structuredClone(pack.source),
    };
  }
}

function hfInstallRequest(value: unknown): {
  target: string;
  force: boolean;
} {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError("Drowse Hugging Face install request is invalid");
  }
  const body = value as InstallManifoldRequest;
  if (Object.keys(value).some((key) => key !== "target" && key !== "as_" && key !== "force")) {
    throw new TypeError("Drowse Hugging Face install request has unknown fields");
  }
  if (typeof body.target !== "string" || !body.target) {
    throw new TypeError("Drowse Hugging Face install target is required");
  }
  if (body.as_ !== undefined) {
    throw new TypeError(
      "Hosted installs preserve the archive identity; omit as_ or publish a .drowse with the desired namespace/name",
    );
  }
  if (body.force !== undefined && typeof body.force !== "boolean") {
    throw new TypeError("Drowse Hugging Face install force flag must be boolean");
  }
  return { target: body.target, force: body.force === true };
}

function mergeRequest(value: unknown): {
  namespace: string;
  name: string;
  description: string;
  sources: MergeManifoldRequest["sources"];
  fit_mode: MergeManifoldRequest["fit_mode"];
  hyperparams: MergeManifoldRequest["hyperparams"];
  force: boolean;
} {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError("Drowse manifold merge request is invalid");
  }
  const allowed = new Set([
    "namespace", "name", "description", "sources", "fit_mode", "hyperparams", "force",
  ]);
  if (Object.keys(value).some((key) => !allowed.has(key))) {
    throw new TypeError("Drowse manifold merge request has unknown fields");
  }
  const body = value as MergeManifoldRequest;
  const namespace = body.namespace ?? "local";
  identityParts(namespace, body.name);
  if (!Array.isArray(body.sources) || body.sources.length < 2) {
    throw new TypeError("Drowse manifold merge requires at least two sources");
  }
  for (const source of body.sources) {
    if (
      !source || typeof source !== "object" || Array.isArray(source) ||
      Object.keys(source).some((key) => key !== "namespace" && key !== "name")
    ) {
      throw new TypeError("Drowse manifold merge source is invalid");
    }
    identityParts(source.namespace, source.name);
  }
  if (body.description !== undefined && typeof body.description !== "string") {
    throw new TypeError("Drowse manifold merge description must be a string");
  }
  if (
    body.fit_mode !== undefined &&
    body.fit_mode !== "pca" && body.fit_mode !== "spectral" && body.fit_mode !== "auto"
  ) {
    throw new TypeError("Drowse manifold merge fit mode must be pca, spectral, or auto");
  }
  if (
    body.hyperparams !== undefined &&
    (!body.hyperparams || typeof body.hyperparams !== "object" || Array.isArray(body.hyperparams))
  ) {
    throw new TypeError("Drowse manifold merge hyperparameters must be an object");
  }
  if (body.force !== undefined && typeof body.force !== "boolean") {
    throw new TypeError("Drowse manifold merge force flag must be boolean");
  }
  return {
    namespace,
    name: body.name,
    description: body.description ?? "",
    sources: body.sources.map((source) => ({ ...source })),
    fit_mode: body.fit_mode,
    hyperparams: body.hyperparams ? structuredClone(body.hyperparams) : undefined,
    force: body.force === true,
  };
}

function isDiscoverMode(value: unknown): value is "pca" | "spectral" | "auto" {
  return value === "pca" || value === "spectral" || value === "auto";
}

function fitHyperparams(value: Record<string, unknown>): Record<string, number | string> {
  const result: Record<string, number | string> = {};
  for (const [key, item] of Object.entries(value)) {
    if ((typeof item !== "number" || !Number.isFinite(item)) && typeof item !== "string") {
      throw new TypeError(`Drowse manifold hyperparameter ${JSON.stringify(key)} is invalid`);
    }
    result[key] = item;
  }
  return result;
}

function requireArgs(args: unknown[], expected: number): void {
  if (args.length !== expected) {
    throw new TypeError(`Drowse artifact request expected ${expected} arguments`);
  }
}

function installOptions(value: unknown): { force: boolean } {
  if (value === undefined || value === null) return { force: false };
  if (
    typeof value !== "object" || Array.isArray(value) ||
    Object.keys(value).some((key) => key !== "force") ||
    (value as { force?: unknown }).force !== undefined &&
      typeof (value as { force?: unknown }).force !== "boolean"
  ) {
    throw new TypeError("Drowse artifact install options are invalid");
  }
  return { force: (value as { force?: boolean }).force === true };
}

function primary(value: unknown): string {
  if (
    typeof value !== "string" ||
    !/^manifolds\/[a-z][a-z0-9._-]{0,63}\/[a-z][a-z0-9._-]{0,63}$/.test(value)
  ) {
    throw new TypeError("Drowse artifact identity must be manifolds/<namespace>/<name>");
  }
  return value;
}

function identity(namespace: unknown, name: unknown): string {
  const [validNamespace, validName] = identityParts(namespace, name);
  return `manifolds/${validNamespace}/${validName}`;
}

function identityParts(namespace: unknown, name: unknown): [string, string] {
  if (
    typeof namespace !== "string" || typeof name !== "string" ||
    !/^[a-z][a-z0-9._-]{0,63}$/.test(namespace) ||
    !/^[a-z][a-z0-9._-]{0,63}$/.test(name)
  ) {
    throw new TypeError("Drowse artifact identity must use canonical namespace and name slugs");
  }
  return [namespace, name];
}
