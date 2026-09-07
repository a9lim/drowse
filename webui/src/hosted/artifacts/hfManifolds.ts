import type { RemoteManifoldInfo } from "../../lib/types";
import { DROWSE_ARCHIVE_MAX_TOTAL_BYTES } from "./types";
import { drowseStorageRoot } from "../runtime/brandMigration";
import { randomUuid } from "../runtime/randomId";

const HF_ORIGIN = "https://huggingface.co";
const SEARCH_LIMIT = 20;
const API_MAX_BYTES = 2 * 1024 * 1024;
const MANIFEST_MAX_BYTES = 1024 * 1024;
const DOWNLOAD_LOCK = "drowse-hf-manifold-download-v1";
const REPOSITORY = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;
const REVISION = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$/;
const SHA256 = /^[0-9a-f]{40}$/;
const ARCHIVE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.drowse$/i;
const NAME = /^[a-z][a-z0-9._-]{0,63}$/;

export interface BrowserHfManifoldRow extends RemoteManifoldInfo {
  repository: string;
  revision: string;
  archive_filename: string;
  archive_bytes: number | null;
  browser_compatible: true;
}

export interface BrowserHfArchive {
  file: File;
  repository: string;
  revision: string;
  filename: string;
}

export interface BrowserHfDownloadProgress {
  phase: "downloading";
  path: string;
  downloadedBytes: number;
  totalBytes: number | null;
}

export interface BrowserHfManifoldClientPort {
  search(query: string, limit?: number): Promise<{
    query: string;
    results: BrowserHfManifoldRow[];
  }>;
  withArchive<T>(
    target: string,
    options: {
      signal?: AbortSignal;
      onProgress?: (progress: BrowserHfDownloadProgress) => void;
    },
    consume: (archive: BrowserHfArchive) => Promise<T>,
  ): Promise<T>;
  clear(): Promise<void>;
}

interface HfSibling {
  rfilename: string;
  size: number | null;
}

interface HfModel {
  id: string;
  sha: string;
  tags: string[];
  siblings: HfSibling[];
}

export class BrowserHfManifoldClient implements BrowserHfManifoldClientPort {
  private readonly fetchImpl: typeof fetch;
  private readonly apiOrigin: string;
  private readonly allowedDownloadOrigins: ReadonlySet<string>;

  constructor(options: {
    fetchImpl?: typeof fetch;
    apiOrigin?: string;
    allowedDownloadOrigins?: ReadonlySet<string>;
  } = {}) {
    this.fetchImpl = options.fetchImpl ?? fetch;
    this.apiOrigin = canonicalOrigin(options.apiOrigin ?? HF_ORIGIN);
    this.allowedDownloadOrigins = options.allowedDownloadOrigins ?? new Set([this.apiOrigin]);
  }

  async search(query: string, limit = SEARCH_LIMIT): Promise<{
    query: string;
    results: BrowserHfManifoldRow[];
  }> {
    const normalizedQuery = searchQuery(query);
    const normalizedLimit = searchLimit(limit);
    const url = new URL("/api/models", this.apiOrigin);
    url.searchParams.set("filter", "drowse-manifold");
    url.searchParams.set("limit", String(normalizedLimit));
    url.searchParams.set("full", "true");
    if (normalizedQuery) url.searchParams.set("search", normalizedQuery);
    const response = await this.fetchJson(url, API_MAX_BYTES);
    if (!Array.isArray(response)) throw new Error("Hugging Face returned an invalid model search response");
    const models = response.slice(0, normalizedLimit).map(parseModel).filter(
      (model): model is HfModel => model !== null && model.tags.includes("drowse-manifold"),
    );
    const results: BrowserHfManifoldRow[] = [];
    for (const model of models) {
      const archives = rootArchives(model.siblings);
      if (archives.length !== 1) continue;
      try {
        const summary = await this.fetchManifoldSummary(model);
        results.push({
          ...summary,
          repository: model.id,
          revision: model.sha,
          archive_filename: archives[0].rfilename,
          archive_bytes: archives[0].size,
          browser_compatible: true,
        });
      } catch {
        // Search remains useful when one third-party repository is malformed.
      }
    }
    return { query: normalizedQuery, results };
  }

  async withArchive<T>(
    target: string,
    options: {
      signal?: AbortSignal;
      onProgress?: (progress: BrowserHfDownloadProgress) => void;
    },
    consume: (archive: BrowserHfArchive) => Promise<T>,
  ): Promise<T> {
    const requested = parseTarget(target);
    options.signal?.throwIfAborted();
    const model = await this.resolveModel(requested.repository, requested.revision, options.signal);
    if (!model.tags.includes("drowse-manifold")) {
      throw new Error(`${model.id} is not tagged as a Drowse manifold repository`);
    }
    const archives = rootArchives(model.siblings);
    if (archives.length !== 1) {
      throw new Error(`${model.id}@${model.sha} must contain exactly one root .drowse archive`);
    }
    return withNamedLock(DOWNLOAD_LOCK, "shared", async () => {
      const directory = await hfDownloadDirectory();
      const temporaryName = `${randomUuid()}.drowse`;
      const handle = await directory.getFileHandle(temporaryName, { create: true });
      const writable = await handle.createWritable();
      try {
        const archive = archives[0];
        const url = resolveUrl(this.apiOrigin, model.id, model.sha, archive.rfilename);
        const response = await this.fetchImpl(url, requestInit(options.signal));
        requireOk(response, `${model.id}@${model.sha}`);
        requireAllowedResponseOrigin(response, this.allowedDownloadOrigins);
        const declared = responseLength(response);
        if (declared !== null && declared > DROWSE_ARCHIVE_MAX_TOTAL_BYTES) {
          throw new Error("The Hugging Face Drowse pack exceeds the 2 GiB archive limit");
        }
        if (!response.body) throw new Error("The Hugging Face Drowse pack response has no body");
        const reader = response.body.getReader();
        let downloaded = 0;
        while (true) {
          options.signal?.throwIfAborted();
          const { done, value } = await reader.read();
          if (done) break;
          downloaded += value.byteLength;
          if (downloaded > DROWSE_ARCHIVE_MAX_TOTAL_BYTES) {
            await reader.cancel();
            throw new Error("The Hugging Face Drowse pack exceeds the 2 GiB archive limit");
          }
          await writable.write(ownedBytes(value));
          options.onProgress?.({
            phase: "downloading",
            path: archive.rfilename,
            downloadedBytes: downloaded,
            totalBytes: declared,
          });
        }
        if (declared !== null && downloaded !== declared) {
          throw new Error(`The Hugging Face Drowse pack stopped at ${downloaded} of ${declared} bytes`);
        }
        await writable.close();
        return await consume({
          file: await handle.getFile(),
          repository: model.id,
          revision: model.sha,
          filename: archive.rfilename,
        });
      } catch (error) {
        await writable.abort(error).catch(() => undefined);
        throw error;
      } finally {
        await directory.removeEntry(temporaryName).catch(ignoreMissing);
      }
    });
  }

  clear(): Promise<void> {
    return withNamedLock(DOWNLOAD_LOCK, "exclusive", async () => {
      const directory = await hfDownloadDirectory();
      for await (const name of directory.keys()) await directory.removeEntry(name, { recursive: true });
    });
  }

  private async resolveModel(
    repository: string,
    revision: string | null,
    signal?: AbortSignal,
  ): Promise<HfModel> {
    const path = revision === null
      ? `/api/models/${repositoryPath(repository)}`
      : `/api/models/${repositoryPath(repository)}/revision/${encodeURIComponent(revision)}`;
    const value = await this.fetchJson(new URL(path, this.apiOrigin), API_MAX_BYTES, signal);
    const model = parseModel(value);
    if (!model || model.id !== repository) {
      throw new Error(`Hugging Face returned invalid metadata for ${repository}`);
    }
    return model;
  }

  private async fetchManifoldSummary(model: HfModel): Promise<RemoteManifoldInfo> {
    const url = resolveUrl(this.apiOrigin, model.id, model.sha, "manifold.json");
    const value = await this.fetchJson(url, MANIFEST_MAX_BYTES, undefined, this.allowedDownloadOrigins);
    if (!isRecord(value) || value.format_version !== 10 || !NAME.test(String(value.name ?? ""))) {
      throw new Error(`${model.id}@${model.sha} has an invalid manifold.json`);
    }
    const fitMode = value.fit_mode;
    if (
      fitMode !== "authored" && fitMode !== "baked" && fitMode !== "pca" &&
      fitMode !== "spectral" && fitMode !== "auto"
    ) throw new Error(`${model.id}@${model.sha} has an invalid fit mode`);
    if (!Array.isArray(value.nodes)) throw new Error(`${model.id}@${model.sha} has invalid nodes`);
    const tags = Array.isArray(value.tags)
      ? value.tags.filter((tag): tag is string => typeof tag === "string")
      : [];
    return {
      name: value.name as string,
      namespace: model.id.split("/")[0],
      description: typeof value.description === "string" ? value.description : "",
      tags,
      node_count: value.nodes.length,
      domain_label: domainLabel(value.domain, fitMode),
      fit_mode: fitMode,
      tensor_models: model.siblings
        .map((item) => item.rfilename)
        .filter((name) => !name.includes("/") && name.endsWith(".safetensors"))
        .map((name) => name.slice(0, -".safetensors".length))
        .sort(),
    };
  }

  private async fetchJson(
    url: URL,
    maximumBytes: number,
    signal?: AbortSignal,
    allowedOrigins: ReadonlySet<string> = new Set([this.apiOrigin]),
  ): Promise<unknown> {
    const response = await this.fetchImpl(url, requestInit(signal));
    requireOk(response, url.pathname);
    requireAllowedResponseOrigin(response, allowedOrigins);
    const bytes = await readBounded(response, maximumBytes, signal);
    try {
      return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
    } catch {
      throw new Error(`Hugging Face returned invalid JSON for ${url.pathname}`);
    }
  }
}

function parseModel(value: unknown): HfModel | null {
  if (!isRecord(value) || typeof value.id !== "string" || !REPOSITORY.test(value.id)) return null;
  if (typeof value.sha !== "string" || !SHA256.test(value.sha)) return null;
  if (!Array.isArray(value.tags) || !Array.isArray(value.siblings)) return null;
  const tags = value.tags.filter((tag): tag is string => typeof tag === "string");
  const siblings: HfSibling[] = [];
  for (const sibling of value.siblings) {
    if (!isRecord(sibling) || typeof sibling.rfilename !== "string") continue;
    const size = sibling.size;
    siblings.push({
      rfilename: sibling.rfilename,
      size: Number.isSafeInteger(size) && (size as number) >= 0 ? size as number : null,
    });
  }
  return { id: value.id, sha: value.sha, tags, siblings };
}

function rootArchives(siblings: readonly HfSibling[]): HfSibling[] {
  return siblings.filter((item) => ARCHIVE.test(item.rfilename));
}

function parseTarget(target: string): { repository: string; revision: string | null } {
  if (typeof target !== "string" || !target) throw new TypeError("Hugging Face target is required");
  const separator = target.lastIndexOf("@");
  const repository = separator < 0 ? target : target.slice(0, separator);
  const revision = separator < 0 ? null : target.slice(separator + 1);
  if (!REPOSITORY.test(repository)) throw new TypeError("Hugging Face target must be owner/repository");
  if (revision !== null && !REVISION.test(revision)) throw new TypeError("Hugging Face revision is invalid");
  return { repository, revision };
}

function searchQuery(value: string): string {
  if (typeof value !== "string") throw new TypeError("Hugging Face search query must be a string");
  const normalized = value.trim();
  if (normalized.length > 200) throw new TypeError("Hugging Face search query is too long");
  return normalized;
}

function searchLimit(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > SEARCH_LIMIT) {
    throw new TypeError(`Hugging Face search limit must be between 1 and ${SEARCH_LIMIT}`);
  }
  return value;
}

function repositoryPath(repository: string): string {
  return repository.split("/").map(encodeURIComponent).join("/");
}

function resolveUrl(origin: string, repository: string, revision: string, filename: string): URL {
  return new URL(
    `/${repositoryPath(repository)}/resolve/${encodeURIComponent(revision)}/${encodeURIComponent(filename)}`,
    origin,
  );
}

function requestInit(signal?: AbortSignal): RequestInit {
  return {
    method: "GET",
    credentials: "omit",
    redirect: "follow",
    referrerPolicy: "no-referrer",
    headers: { Accept: "application/json, application/octet-stream" },
    ...(signal ? { signal } : {}),
  };
}

async function readBounded(
  response: Response,
  maximumBytes: number,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const declared = responseLength(response);
  if (declared !== null && declared > maximumBytes) throw new Error("Hugging Face response is too large");
  if (!response.body) throw new Error("Hugging Face response has no body");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    signal?.throwIfAborted();
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maximumBytes) {
      await reader.cancel();
      throw new Error("Hugging Face response is too large");
    }
    chunks.push(ownedBytes(value));
  }
  const result = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

function responseLength(response: Response): number | null {
  const value = response.headers.get("Content-Length");
  if (value === null || !/^\d+$/.test(value)) return null;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) ? parsed : null;
}

function requireOk(response: Response, label: string): void {
  if (!response.ok) throw new Error(`Hugging Face request for ${label} failed with HTTP ${response.status}`);
}

function requireAllowedResponseOrigin(response: Response, allowed: ReadonlySet<string>): void {
  let origin: string;
  try {
    origin = new URL(response.url).origin;
  } catch {
    throw new Error("Hugging Face returned an invalid final response URL");
  }
  if (!allowed.has(origin)) throw new Error(`Hugging Face redirected to untrusted origin ${origin}`);
}

function domainLabel(value: unknown, fitMode: string): string {
  if (isRecord(value)) {
    if (value.type === "box" && Array.isArray(value.axes)) return `box(${value.axes.length}d)`;
    if (value.type === "sphere" && Number.isSafeInteger(value.dim)) return `sphere(${value.dim}d)`;
    if (value.type === "custom" && Number.isSafeInteger(value.embed_dim)) {
      return `custom(${value.embed_dim}d)`;
    }
  }
  return fitMode === "pca" || fitMode === "spectral" || fitMode === "auto"
    ? `discover-${fitMode}`
    : "?";
}

async function hfDownloadDirectory(): Promise<FileSystemDirectoryHandle> {
  if (!navigator.storage?.getDirectory) throw new Error("Origin-private file storage is unavailable");
  const root = await navigator.storage.getDirectory();
  const drowse = await drowseStorageRoot(root);
  const artifacts = await drowse.getDirectoryHandle("artifacts", { create: true });
  return artifacts.getDirectoryHandle("hf-downloads", { create: true });
}

function withNamedLock<T>(
  name: string,
  mode: "shared" | "exclusive",
  operation: () => Promise<T>,
): Promise<T> {
  if (!navigator.locks?.request) return Promise.reject(new Error("The Web Locks API is unavailable"));
  return navigator.locks.request(name, { mode }, async (lock) => {
    if (!lock) throw new Error("The Hugging Face download lock was not acquired");
    return operation();
  });
}

function canonicalOrigin(value: string): string {
  const url = new URL(value);
  if (url.protocol !== "https:" || url.pathname !== "/" || url.search || url.hash) {
    throw new TypeError("Hugging Face API origin must be an HTTPS origin");
  }
  return url.origin;
}

function ownedBytes(value: Uint8Array): Uint8Array<ArrayBuffer> {
  return new Uint8Array(value);
}

function ignoreMissing(error: unknown): void {
  if (error instanceof DOMException && error.name === "NotFoundError") return;
  throw error;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
