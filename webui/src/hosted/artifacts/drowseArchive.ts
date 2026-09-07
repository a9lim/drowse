import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import { parseExactJson } from "./json";
import { SafetensorsValidator, type SafetensorsDescription } from "./safetensors";
import {
  DROWSE_ARCHIVE_FORMAT_VERSION,
  DROWSE_ARCHIVE_MANIFEST_MAX_BYTES,
  DROWSE_ARCHIVE_MAX_FILE_BYTES,
  DrowseArchiveError,
  type InspectedDrowseArchive,
  type DrowseArchiveArchiveEntry,
  type DrowseArchiveFileRecord,
  type DrowseArchiveFittedArtifact,
  type DrowseArchiveManifest,
  type DrowseArchiveSource,
  type VerifiedDrowseArchive,
  type VerifyDrowseArchiveOptions,
} from "./types";
import { openZip, readZipEntry, type ZipArchive, type ZipEntry } from "./zip";

const NAME = /^[a-z][a-z0-9._-]{0,63}$/;
const LABEL = /^[a-z][a-z0-9_-]{0,63}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const METADATA_MAX_BYTES = 4 * 1024 * 1024;
const NODE_MAX_BYTES = 64 * 1024 * 1024;
const MANIFOLD_FORMAT_VERSION = 10;
const TEMPLATE_FORMAT_VERSION = 2;
const RESERVED_MANIFOLD_NAMESPACES = new Set(["jlens", "sae"]);
const MERGE_BAKE_POLICY = "additive_union_v1";

type FitMode = "authored" | "baked" | "pca" | "spectral" | "auto";

const HYPERPARAMS_BY_MODE: Readonly<Record<"pca" | "spectral" | "auto", ReadonlySet<string>>> = {
  pca: new Set(["max_dim", "var_threshold"]),
  spectral: new Set(["max_dim", "min_dim", "k_nn", "bandwidth", "max_subspace_dim", "smoothing"]),
  auto: new Set([
    "max_dim", "var_threshold", "min_dim", "k_nn", "bandwidth",
    "max_subspace_dim", "smoothing", "persistence_frac",
  ]),
};

const METHOD_FIT_MODES: Readonly<Record<string, ReadonlySet<FitMode>>> = {
  manifold_pca: new Set(["authored"]),
  manifold_sae: new Set(["authored"]),
  manifold_monopolar: new Set(["authored", "pca"]),
  manifold_monopolar_sae: new Set(["authored", "pca"]),
  manifold_discover_auto: new Set(["auto"]),
  manifold_discover_pca: new Set(["pca"]),
  manifold_discover_spectral: new Set(["spectral"]),
  manifold_discover_sae: new Set(["pca", "spectral", "auto"]),
  manifold_procrustes_transfer: new Set(["authored", "baked", "pca", "spectral", "auto"]),
  merge: new Set(["baked"]),
  folded_vector: new Set(["baked"]),
};

interface InternalInspection {
  archive: ZipArchive;
  manifest: DrowseArchiveManifest;
  publicPack: InspectedDrowseArchive;
}

interface ManifoldClosure {
  value: Record<string, unknown>;
  fitMode: FitMode;
  name: string;
  labels: string[];
  roles: Array<string | null>;
  kinds: Array<string | null>;
  coords: number[][];
  domain: DomainShape | null;
  hyperparams: Record<string, unknown>;
  templateRef: string | null;
  expectedPaths: Set<string>;
  tensorPaths: string[];
}

interface TemplateClosure {
  value: Record<string, unknown>;
  labels: string[];
  corpora: Map<string, string[]>;
  sha256: string;
}

interface DomainShape {
  intrinsicDim: number;
  embedDim: number;
}

export interface ParsedDrowseTensorFilename {
  modelId: string;
  safeModelId: string;
  variant: "raw" | "sae" | "from";
  variantIdentity: string | null;
  safeVariantIdentity: string | null;
}

const internals = new WeakMap<InspectedDrowseArchive, InternalInspection>();
const verifiedInternals = new WeakMap<VerifiedDrowseArchive, InternalInspection>();

export async function inspectDrowseArchive(source: DrowseArchiveSource): Promise<InspectedDrowseArchive> {
  const archive = await openZip(source);
  const packEntry = archive.byPath.get("pack.json");
  if (packEntry === undefined) manifestError("archive has no pack.json");
  if (packEntry.size > DROWSE_ARCHIVE_MANIFEST_MAX_BYTES) {
    throw new DrowseArchiveError("ARCHIVE_LIMIT_EXCEEDED", "pack.json is larger than 1 MiB", "pack.json");
  }
  const exactManifestBytes = await readZipEntry(archive, packEntry);
  if (exactManifestBytes === null) manifestError("pack.json could not be read");
  const manifest = validateManifest(parseExactJson(exactManifestBytes, "pack.json"), archive);
  const primaryIdentity = identity(manifest.primary, "manifolds");
  const templateIdentity = manifest.template === null
    ? null
    : identity(manifest.template, "templates");
  const entries: DrowseArchiveArchiveEntry[] = archive.entries.map((entry) => ({
    path: entry.path,
    size: entry.size,
    compressedSize: entry.compressedSize,
    compressionMethod: entry.compressionMethod,
  }));
  const publicPack: InspectedDrowseArchive = Object.freeze({
    manifest: deepFreeze(manifest),
    exactManifestBytes: exactManifestBytes.slice(),
    entries: Object.freeze(entries),
    expandedBytes: archive.expandedBytes,
    archiveBytes: archive.reader.size,
    primaryIdentity: Object.freeze(primaryIdentity),
    templateIdentity: templateIdentity === null ? null : Object.freeze(templateIdentity),
  });
  internals.set(publicPack, { archive, manifest, publicPack });
  return publicPack;
}

export async function verifyDrowseArchive(
  inspected: InspectedDrowseArchive,
  options: VerifyDrowseArchiveOptions = {},
): Promise<VerifiedDrowseArchive> {
  const internal = internals.get(inspected);
  if (internal === undefined) {
    throw new DrowseArchiveError(
      "ARCHIVE_INVALID",
      "inspection handle was not created by inspectDrowseArchive in this Worker",
    );
  }
  const { archive, manifest } = internal;
  const stage = options.stage;
  let began = false;
  let verifiedBytes = 0;
  const report = (phase: "verifying" | "validating" | "committing", path: string | null) => {
    options.onProgress?.({ phase, path, verifiedBytes, totalBytes: totalPayloadBytes(manifest) });
  };
  try {
    options.signal?.throwIfAborted();
    if (stage !== undefined) {
      began = true;
      await stage.begin(inspected);
    }
    const primaryPath = `${manifest.primary}/manifold.json`;
    const manifoldBytes = await verifyEntry(primaryPath, capture(METADATA_MAX_BYTES));
    const manifold = validateManifold(
      parseExactJson(manifoldBytes, primaryPath),
      internal,
    );
    let template: TemplateClosure | null = null;
    if (manifest.template !== null) {
      const templatePath = `${manifest.template}/template.json`;
      const templateBytes = await verifyEntry(templatePath, capture(METADATA_MAX_BYTES));
      template = validateTemplate(parseExactJson(templateBytes, templatePath), internal);
    }
    validateExactClosure(manifold, template, internal);

    const nodeContents = new Map<string, string[]>();
    const nodeBytes: Uint8Array[] = [];
    for (let index = 0; index < manifold.labels.length && manifold.fitMode !== "baked"; index += 1) {
      const label = manifold.labels[index];
      const path = `${manifest.primary}/nodes/${nodeFilename(index, label)}`;
      const bytes = await verifyEntry(path, capture(NODE_MAX_BYTES));
      const corpus = validateNodeCorpus(parseExactJson(bytes, path), path);
      nodeContents.set(label, corpus);
      nodeBytes.push(bytes);
    }
    if (template !== null) {
      for (const label of manifold.labels) {
        if (!sameStrings(nodeContents.get(label), template.corpora.get(label))) {
          closureError("templated manifold corpus does not match its template closure");
        }
      }
    }

    const expectedNodesSha256 = nodesSha256Candidates(manifold, nodeBytes, template?.sha256 ?? null);
    const runtimeFingerprints = new Set<string>();
    const fittedArtifacts: DrowseArchiveFittedArtifact[] = [];
    for (const tensorPath of manifold.tensorPaths) {
      const tensorIdentity = parseDrowseTensorFilename(tensorPath.split("/").at(-1)!);
      if (tensorIdentity === null) closureError("fitted tensor filename is invalid", tensorPath);
      const sidecarPath = tensorPath.replace(/\.safetensors$/, ".json");
      const sidecarBytes = await verifyEntry(sidecarPath, capture(METADATA_MAX_BYTES));
      const sidecar = validateSidecar(
        parseExactJson(sidecarBytes, sidecarPath),
        sidecarPath,
        manifold,
        expectedNodesSha256,
        tensorIdentity,
      );
      const tensorEntry = requiredEntry(archive, tensorPath);
      const validator = new SafetensorsValidator(tensorPath, tensorEntry.size);
      await verifyEntry(tensorPath, {
        push: (chunk) => validator.push(chunk),
        finish: () => validateTensorDescription(validator.finish(), tensorPath, sidecar),
      });
      if (typeof sidecar.model_fingerprint === "string" && sidecar.model_fingerprint) {
        runtimeFingerprints.add(sidecar.model_fingerprint);
      }
      fittedArtifacts.push(Object.freeze({
        tensorPath,
        sidecarPath,
        modelId: tensorIdentity.modelId,
        variant: tensorIdentity.variant,
        variantIdentity: tensorIdentity.variantIdentity,
        modelFingerprint: sidecar.model_fingerprint as string | null,
        contextBindingSha256: sidecar.context_binding_sha256 as string | null,
        modelSourceFingerprint: sidecar.model_source_fingerprint as string | null,
      }));
    }

    report("validating", null);
    options.signal?.throwIfAborted();
    const files = Object.freeze(
      Object.fromEntries(Object.entries(manifest.files).map(([path, record]) => [path, Object.freeze({ ...record })])),
    );
    const verified: VerifiedDrowseArchive = Object.freeze({
      ...inspected,
      verifiedAt: Date.now(),
      files,
      manifold: deepFreeze(manifold.value),
      template: template === null ? null : deepFreeze(template.value),
      fittedArtifacts: Object.freeze(fittedArtifacts),
      fittedRuntimeFingerprints: Object.freeze([...runtimeFingerprints].sort()),
    });
    verifiedInternals.set(verified, internal);
    if (stage !== undefined) {
      report("committing", null);
      await stage.commit(verified);
    }
    return verified;

    async function verifyEntry(
      path: string,
      validator?: ByteValidator,
    ): Promise<Uint8Array> {
      options.signal?.throwIfAborted();
      const entry = requiredEntry(archive, path);
      const expected = manifest.files[path];
      if (expected === undefined) manifestMismatch(`pack.json has no record for ${JSON.stringify(path)}`, path);
      const hasher = sha256.create();
      const captured: Uint8Array[] = [];
      report("verifying", path);
      await readZipEntry(archive, entry, {
        write: async (chunk) => {
          options.signal?.throwIfAborted();
          hasher.update(chunk);
          validator?.push(chunk);
          if (validator?.capture === true) captured.push(chunk.slice());
          if (stage !== undefined) await stage.write(path, chunk);
          verifiedBytes += chunk.length;
          report("verifying", path);
        },
      }, options.signal);
      validator?.finish?.();
      const actual = bytesToHex(hasher.digest());
      if (actual !== expected.sha256) {
        throw new DrowseArchiveError(
          "CHECKSUM_MISMATCH",
          `archive entry ${JSON.stringify(path)} failed SHA-256 verification`,
          path,
        );
      }
      if (stage !== undefined) await stage.finish(path);
      if (validator?.capture !== true) return new Uint8Array();
      const output = new Uint8Array(expected.size);
      let offset = 0;
      for (const chunk of captured) {
        output.set(chunk, offset);
        offset += chunk.length;
      }
      return output;
    }
  } catch (error) {
    if (began && stage !== undefined) {
      try {
        await stage.rollback(error);
      } catch (rollbackError) {
        throw new DrowseArchiveError(
          "STAGE_FAILED",
          `drowseArchive validation failed and staging rollback also failed: ${describe(rollbackError)}`,
        );
      }
    }
    throw error;
  }
}

export async function readVerifiedDrowseArchiveFile(
  verified: VerifiedDrowseArchive,
  path: string,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const internal = verifiedInternals.get(verified);
  if (internal === undefined) {
    throw new DrowseArchiveError(
      "ARCHIVE_INVALID",
      "verified pack handle was not created by this Worker",
    );
  }
  signal?.throwIfAborted();
  const expected = verified.files[path];
  if (expected === undefined) manifestMismatch(`pack.json has no record for ${JSON.stringify(path)}`, path);
  const bytes = await readZipEntry(internal.archive, requiredEntry(internal.archive, path), undefined, signal);
  if (bytes === null || bytes.length !== expected.size) {
    manifestMismatch(`archive entry ${JSON.stringify(path)} has an unexpected size`, path);
  }
  const actual = bytesToHex(sha256(bytes));
  if (actual !== expected.sha256) {
    throw new DrowseArchiveError(
      "CHECKSUM_MISMATCH",
      `archive entry ${JSON.stringify(path)} does not match pack.json`,
      path,
    );
  }
  return bytes;
}

export async function validateDrowseArchive(
  source: DrowseArchiveSource,
  options: VerifyDrowseArchiveOptions = {},
): Promise<VerifiedDrowseArchive> {
  options.onProgress?.({
    phase: "inspecting",
    path: null,
    verifiedBytes: 0,
    totalBytes: source instanceof Blob ? source.size : source.byteLength,
  });
  return verifyDrowseArchive(await inspectDrowseArchive(source), options);
}

interface ByteValidator {
  capture?: boolean;
  push(chunk: Uint8Array): void;
  finish?(): void;
}

function capture(maximum: number): ByteValidator {
  let bytes = 0;
  return {
    capture: true,
    push(chunk) {
      bytes += chunk.length;
      if (bytes > maximum) {
        throw new DrowseArchiveError("ARCHIVE_LIMIT_EXCEEDED", `JSON payload is larger than ${maximum} bytes`);
      }
    },
  };
}

function validateManifest(value: unknown, archive: ZipArchive): DrowseArchiveManifest {
  if (!isRecord(value) || !exactKeys(value, [
    "files", "format_version", "kind", "primary", "producer", "source", "template",
  ])) {
    manifestError("pack.json does not match the exact v1 schema");
  }
  if (value.format_version !== DROWSE_ARCHIVE_FORMAT_VERSION) {
    manifestError(`pack.json format_version must be ${DROWSE_ARCHIVE_FORMAT_VERSION}`);
  }
  const producerName = ["drowse", "polythetic", "saklas"].find((name) => value.kind === `${name}-manifold`);
  if (producerName === undefined) {
    manifestError("pack.json kind must identify a Drowse manifold");
  }
  if (
    !isRecord(value.producer)
    || !exactKeys(value.producer, ["name", "version"])
    || value.producer.name !== producerName
    || !shortString(value.producer.version, 1_024)
  ) {
    manifestError("pack.json producer must name a drowse version");
  }
  const primary = archivePath(value.primary, "manifolds");
  const template = value.template === null ? null : archivePath(value.template, "templates");
  const source = validateSource(value.source);
  if (!isRecord(value.files) || Object.keys(value.files).length === 0) {
    manifestError("pack.json files must be a non-empty object");
  }
  const files: Record<string, DrowseArchiveFileRecord> = Object.create(null) as Record<string, DrowseArchiveFileRecord>;
  for (const [path, raw] of Object.entries(value.files)) {
    validateManifestPath(path);
    if (!isRecord(raw) || !exactKeys(raw, ["sha256", "size"])) {
      manifestError(`pack.json file record for ${JSON.stringify(path)} is invalid`);
    }
    if (
      typeof raw.size !== "number"
      || !Number.isSafeInteger(raw.size)
      || raw.size < 0
      || raw.size > DROWSE_ARCHIVE_MAX_FILE_BYTES
    ) {
      manifestError(`pack.json size for ${JSON.stringify(path)} is invalid`);
    }
    if (typeof raw.sha256 !== "string" || !SHA256.test(raw.sha256)) {
      manifestError(`pack.json sha256 for ${JSON.stringify(path)} is invalid`);
    }
    const entry = archive.byPath.get(path);
    if (entry === undefined || entry.size !== raw.size) {
      manifestMismatch(`pack.json size for ${JSON.stringify(path)} does not match ZIP`, path);
    }
    const templateFile = template === null ? null : `${template}/template.json`;
    if (!path.startsWith(`${primary}/`) && path !== templateFile) {
      manifestMismatch(`archive entry ${JSON.stringify(path)} is outside its declared closure`, path);
    }
    files[path] = { size: raw.size, sha256: raw.sha256 };
  }
  const archivePaths = new Set(archive.entries.map((entry) => entry.path));
  archivePaths.delete("pack.json");
  if (!sameSet(new Set(Object.keys(files)), archivePaths)) {
    manifestMismatch("pack.json files must describe every non-manifest archive entry exactly");
  }
  if (files[`${primary}/manifold.json`] === undefined) {
    manifestMismatch("archive closure has no manifold.json");
  }
  if (template !== null && files[`${template}/template.json`] === undefined) {
    manifestMismatch("archive template closure has no template.json");
  }
  return {
    format_version: 1,
    kind: "drowse-manifold",
    producer: { name: "drowse", version: value.producer.version as string },
    primary,
    template,
    source,
    files,
  };
}

function validateManifold(value: unknown, internal: InternalInspection): ManifoldClosure {
  const path = `${internal.manifest.primary}/manifold.json`;
  if (!isRecord(value)) closureError("manifold.json must be a JSON object", path);
  const fitMode = value.fit_mode;
  if (fitMode !== "authored" && fitMode !== "baked" && fitMode !== "pca" && fitMode !== "spectral" && fitMode !== "auto") {
    closureError("manifold.json has an invalid fit_mode", path);
  }
  const common = [
    "artifact_id", "description", "files", "fit_epochs", "fit_mode", "format_version",
    "name", "nodes", "source", "tags", "template_ref",
  ];
  const allowed = new Set([...common, fitMode === "authored" || fitMode === "baked" ? "domain" : "hyperparams"]);
  if (Object.keys(value).some((key) => !allowed.has(key))) closureError("manifold.json has unknown fields", path);
  for (const required of ["description", "files", "fit_mode", "format_version", "name", "nodes", "source", "tags", "template_ref"]) {
    if (!(required in value)) closureError(`manifold.json is missing ${required}`, path);
  }
  if (value.format_version !== MANIFOLD_FORMAT_VERSION) closureError("manifold.json must use format_version 10", path);
  const [, expectedName] = internal.publicPack.primaryIdentity;
  if (typeof value.name !== "string" || value.name !== expectedName || !NAME.test(value.name)) {
    closureError("manifold.json identity does not match pack.json primary", path);
  }
  if (typeof value.description !== "string" || !shortString(value.source, 4_096)) {
    closureError("manifold.json description/source is invalid", path);
  }
  if (!Array.isArray(value.tags) || !value.tags.every((tag) => typeof tag === "string")) {
    closureError("manifold.json tags must be strings", path);
  }
  if (value.artifact_id !== undefined && !shortString(value.artifact_id, 4_096)) {
    closureError("manifold.json artifact_id is invalid", path);
  }
  if (value.fit_epochs !== undefined && !isRecord(value.fit_epochs)) {
    closureError("manifold.json fit_epochs is invalid", path);
  }
  if (!Array.isArray(value.nodes) || value.nodes.length === 0) closureError("manifold.json needs nodes", path);
  const labels: string[] = [];
  const roles: Array<string | null> = [];
  const kinds: Array<string | null> = [];
  const coords: number[][] = [];
  let domain: DomainShape | null = null;
  let hyperparams: Record<string, unknown> = {};
  if (fitMode === "authored" || fitMode === "baked") domain = validateDomain(value.domain, path);
  else {
    if (!isRecord(value.hyperparams)) closureError("discover manifold hyperparams must be an object", path);
    hyperparams = validateHyperparams(fitMode, value.hyperparams, path);
  }
  if (fitMode === "authored" && value.nodes.length < 2 * domain!.intrinsicDim + 1) {
    closureError(`authored manifold needs at least ${2 * domain!.intrinsicDim + 1} nodes`, path);
  }
  for (const rawNode of value.nodes) {
    const expected = fitMode === "authored" ? ["coords", "kind", "label", "role"] : ["kind", "label", "role"];
    if (!isRecord(rawNode) || !exactKeys(rawNode, expected)) closureError("manifold.json node has an invalid schema", path);
    if (typeof rawNode.label !== "string" || !LABEL.test(rawNode.label)) closureError("manifold.json node label is invalid", path);
    if (rawNode.role !== null && (typeof rawNode.role !== "string" || !/^[a-z0-9._-]+$/.test(rawNode.role))) {
      closureError("manifold.json node role is invalid", path);
    }
    if (rawNode.kind !== null && rawNode.kind !== "abstract" && rawNode.kind !== "concrete" && rawNode.kind !== "custom") {
      closureError("manifold.json node kind is invalid", path);
    }
    if (fitMode === "authored") {
      if (!numericVector(rawNode.coords, domain!.intrinsicDim)) closureError("manifold.json node coordinates are invalid", path);
      coords.push([...(rawNode.coords as number[])]);
    }
    labels.push(rawNode.label);
    roles.push(rawNode.role as string | null);
    kinds.push(rawNode.kind as string | null);
  }
  if (new Set(labels).size !== labels.length) closureError("manifold.json has duplicate node labels", path);
  const templateRef = value.template_ref;
  if (templateRef !== null && (typeof templateRef !== "string" || !/^([a-z][a-z0-9._-]{0,63})\/([a-z][a-z0-9._-]{0,63})$/.test(templateRef))) {
    closureError("manifold.json template_ref must be namespace-qualified", path);
  }
  if (templateRef !== null && (fitMode === "authored" || fitMode === "baked")) {
    closureError("only discover manifolds may reference a template", path);
  }
  if (!isRecord(value.files)) closureError("manifold.json files must be an object", path);
  const tensorPaths: string[] = [];
  const expectedPairs = new Set<string>();
  for (const [name, digest] of Object.entries(value.files)) {
    if (!/^[A-Za-z0-9%_-]+\.(?:json|safetensors)$/.test(name) || name.includes("/")) {
      closureError(`manifold.json files entry ${JSON.stringify(name)} is not canonical`, path);
    }
    if (typeof digest !== "string" || !SHA256.test(digest)) closureError("manifold.json contains an invalid file digest", path);
    const fullPath = `${internal.manifest.primary}/${name}`;
    if (internal.manifest.files[fullPath]?.sha256 !== digest) {
      closureError(`manifold.json digest for ${JSON.stringify(name)} does not match pack.json`, path);
    }
    const tensorName = name.replace(/\.json$/, ".safetensors");
    if (!isCanonicalTensorFilename(tensorName)) closureError(`manifold.json file ${JSON.stringify(name)} is not a fitted tensor pair`, path);
    expectedPairs.add(tensorName);
    expectedPairs.add(tensorName.replace(/\.safetensors$/, ".json"));
    if (name.endsWith(".safetensors")) tensorPaths.push(fullPath);
  }
  if (!sameSet(new Set(Object.keys(value.files)), expectedPairs)) {
    closureError("manifold.json files must contain complete tensor/sidecar pairs", path);
  }
  if (fitMode === "baked" && tensorPaths.length === 0) closureError("baked manifold has no fitted tensor", path);
  const expectedPaths = new Set<string>([path]);
  if (fitMode !== "baked") {
    labels.forEach((label, index) => expectedPaths.add(`${internal.manifest.primary}/nodes/${nodeFilename(index, label)}`));
  }
  for (const name of Object.keys(value.files)) expectedPaths.add(`${internal.manifest.primary}/${name}`);
  validatePackSourceMatchesManifold(internal.manifest.source, value.source as string);
  return {
    value,
    fitMode,
    name: value.name,
    labels,
    roles,
    kinds,
    coords,
    domain,
    hyperparams,
    templateRef: templateRef as string | null,
    expectedPaths,
    tensorPaths: tensorPaths.sort(),
  };
}

function validateTemplate(value: unknown, internal: InternalInspection): TemplateClosure {
  const path = `${internal.manifest.template}/template.json`;
  if (!isRecord(value) || !exactKeys(value, [
    "contexts", "description", "format_version", "name", "slot", "source", "tags", "values",
  ])) closureError("template.json does not match the exact v2 schema", path);
  const identity = internal.publicPack.templateIdentity!;
  if (value.format_version !== TEMPLATE_FORMAT_VERSION || value.name !== identity[1] || typeof value.name !== "string" || !NAME.test(value.name)) {
    closureError("template.json identity or format does not match its closure", path);
  }
  if (!shortString(value.slot, 4_096) || typeof value.description !== "string" || !shortString(value.source, 4_096)) {
    closureError("template.json metadata is invalid", path);
  }
  if (!Array.isArray(value.tags) || !value.tags.every((tag) => typeof tag === "string")) closureError("template.json tags are invalid", path);
  if (!Array.isArray(value.values) || value.values.length < 2 || !value.values.every((item) => typeof item === "string" && item.trim())) {
    closureError("template.json values must contain at least two non-blank strings", path);
  }
  const labels = (value.values as string[]).map(slugValue);
  if (labels.some((label) => !LABEL.test(label)) || new Set(labels).size !== labels.length) {
    closureError("template.json values do not produce unique valid node labels", path);
  }
  if (!Array.isArray(value.contexts) || value.contexts.length === 0) closureError("template.json contexts must be non-empty", path);
  const contexts: Array<{ assistant: string }> = [];
  for (const rawContext of value.contexts) {
    if (!isRecord(rawContext) || !exactKeys(rawContext, ["assistant", "turns"]) || !Array.isArray(rawContext.turns) || rawContext.turns.length === 0) {
      closureError("template.json context has an invalid schema", path);
    }
    for (const turn of rawContext.turns) {
      if (!isRecord(turn) || !exactKeys(turn, ["content", "role"]) || !["system", "user", "assistant"].includes(String(turn.role)) || typeof turn.content !== "string" || !turn.content.trim()) {
        closureError("template.json context turn is invalid", path);
      }
      if (turn.content.includes(value.slot as string)) closureError("template slot appears in a history turn", path);
    }
    const finalTurn = rawContext.turns[rawContext.turns.length - 1] as Record<string, unknown>;
    if (finalTurn.role !== "user") closureError("template context must end with a user turn", path);
    if (typeof rawContext.assistant !== "string" || !rawContext.assistant.trim() || occurrences(rawContext.assistant, value.slot as string) !== 1) {
      closureError("template assistant text must contain the slot exactly once", path);
    }
    contexts.push({ assistant: rawContext.assistant });
  }
  const corpora = new Map<string, string[]>();
  labels.forEach((label, index) => {
    const replacement = (value.values as string[])[index];
    corpora.set(label, contexts.map((context) => context.assistant.replace(value.slot as string, replacement)));
  });
  const core = {
    format_version: TEMPLATE_FORMAT_VERSION,
    slot: value.slot,
    values: value.values,
    contexts: value.contexts,
  };
  return {
    value,
    labels,
    corpora,
    sha256: templateClosureSha256(core),
  };
}

function validateExactClosure(
  manifold: ManifoldClosure,
  template: TemplateClosure | null,
  internal: InternalInspection,
): void {
  const expected = new Set(manifold.expectedPaths);
  if (internal.manifest.template === null) {
    if (manifold.templateRef !== null) closureError("templated manifold is missing its template closure");
  } else {
    expected.add(`${internal.manifest.template}/template.json`);
    const expectedRef = internal.publicPack.templateIdentity!.join("/");
    if (manifold.templateRef !== expectedRef) closureError("manifold template_ref does not match its closure");
    if (template === null || !sameStrings(manifold.labels, template.labels)) {
      closureError("manifold node labels do not match its template closure");
    }
  }
  if (!sameSet(expected, new Set(Object.keys(internal.manifest.files)))) {
    closureError("archive manifold/template closure is not exact");
  }
}

function validateNodeCorpus(value: unknown, path: string): string[] {
  if (!Array.isArray(value) || value.length === 0 || !value.every((row) => typeof row === "string")) {
    closureError("node corpus must be a non-empty JSON list of strings", path);
  }
  return value as string[];
}

function nodesSha256Candidates(
  manifold: ManifoldClosure,
  nodeBytes: readonly Uint8Array[],
  templateSha256: string | null,
): ReadonlySet<string> {
  if (manifold.fitMode === "baked") {
    return new Set([hashParts([
      pythonCanonicalJsonBytes({ fit_mode: "baked", node_labels: manifold.labels }),
    ])]);
  }
  const prefix: Uint8Array[] = [pythonCanonicalJsonBytes(manifold.labels), ...nodeBytes];
  const suffix = [
    pythonCanonicalJsonBytes(manifold.roles),
    pythonCanonicalJsonBytes(manifold.kinds),
  ];
  if (templateSha256 !== null) suffix.push(new TextEncoder().encode(templateSha256));
  if (manifold.fitMode === "authored") {
    const coords = manifold.coords.map((row) => row.map((value) => new PythonFloat(value)));
    const domains = [
      manifold.value.domain,
      pythonFloatDomain(manifold.value.domain),
    ];
    return new Set(domains.map((domain) => hashParts([
      ...prefix,
      pythonCanonicalJsonBytes(domain),
      pythonCanonicalJsonBytes(coords),
      ...suffix,
    ])));
  }
  const hyperparams = [manifold.hyperparams, pythonFloatHyperparams(manifold.hyperparams)];
  return new Set(hyperparams.map((params) => hashParts([
    ...prefix,
    pythonCanonicalJsonBytes({ fit_mode: manifold.fitMode, hyperparams: params }),
    ...suffix,
  ])));
}

export function discoverManifoldNodesSha256(input: {
  labels: readonly string[];
  nodeBytes: readonly Uint8Array[];
  fitMode: "pca" | "spectral" | "auto";
  hyperparams: Readonly<Record<string, unknown>>;
  roles: readonly (string | null)[];
  kinds: readonly (string | null)[];
  templateSha256?: string | null;
}): string {
  const parts = [
    pythonCanonicalJsonBytes(input.labels),
    ...input.nodeBytes,
    pythonCanonicalJsonBytes({
      fit_mode: input.fitMode,
      hyperparams: input.hyperparams,
    }),
    pythonCanonicalJsonBytes(input.roles),
    pythonCanonicalJsonBytes(input.kinds),
  ];
  if (input.templateSha256 !== null && input.templateSha256 !== undefined) {
    if (!SHA256.test(input.templateSha256)) {
      throw new TypeError("Template closure fingerprint must be a SHA-256 digest");
    }
    parts.push(new TextEncoder().encode(input.templateSha256));
  }
  return hashParts(parts);
}

export function templateClosureSha256(
  value: Readonly<Record<string, unknown>>,
): string {
  return hashParts([pythonCanonicalJsonBytes({
    format_version: value.format_version,
    slot: value.slot,
    values: value.values,
    contexts: value.contexts,
  })]);
}

function hashParts(parts: readonly Uint8Array[]): string {
  const hasher = sha256.create();
  for (const part of parts) hasher.update(part);
  return bytesToHex(hasher.digest());
}

function validateHyperparams(
  fitMode: FitMode,
  value: unknown,
  path: string,
): Record<string, unknown> {
  if (!isRecord(value)) closureError("manifold hyperparams must be an object", path);
  if (fitMode === "authored" || fitMode === "baked") {
    if (Object.keys(value).length !== 0) closureError(`${fitMode} fit cannot carry hyperparams`, path);
    return value;
  }
  const allowed = HYPERPARAMS_BY_MODE[fitMode];
  const intKeys = new Set(["max_dim", "min_dim", "k_nn", "max_subspace_dim"]);
  const floatKeys = new Set(["var_threshold", "bandwidth", "persistence_frac"]);
  for (const [key, item] of Object.entries(value)) {
    if (!allowed.has(key)) closureError(`fit_mode ${fitMode} does not accept hyperparameter ${JSON.stringify(key)}`, path);
    const valid = intKeys.has(key)
      ? positiveInteger(item)
      : floatKeys.has(key)
        ? finite(item)
        : key === "smoothing" && (item === "auto" || finite(item));
    if (!valid) closureError(`hyperparameter ${JSON.stringify(key)} has an invalid value`, path);
  }
  return value;
}

function validateDiagnostics(value: unknown, path: string): void {
  if (!isRecord(value)) closureError("fitted sidecar diagnostics must be an object", path);
  const allowed = new Set([
    "per_component_variance", "cumulative_variance", "eigenvalues", "picked_k",
    "gap_index", "k_nn", "component_count", "heuristic_k", "threshold",
    "gap_magnitude", "bandwidth", "pinned", "min_dim",
  ]);
  const listKeys = new Set(["per_component_variance", "cumulative_variance", "eigenvalues"]);
  const intKeys = new Set(["picked_k", "gap_index", "k_nn", "component_count", "heuristic_k", "min_dim"]);
  for (const [key, item] of Object.entries(value)) {
    if (!allowed.has(key)) closureError("fitted sidecar diagnostics has unknown fields", path);
    const valid = listKeys.has(key)
      ? Array.isArray(item) && item.length > 0 && item.every(finite)
      : intKeys.has(key)
        ? nonNegativeInteger(item)
        : key === "pinned"
          ? typeof item === "boolean"
          : finite(item);
    if (!valid) closureError(`fitted sidecar diagnostics field ${JSON.stringify(key)} is invalid`, path);
  }
}

function validateLayerMap(
  value: unknown,
  fittedLayers: ReadonlySet<string>,
  requireFitted: boolean,
  validateValue: (item: unknown) => boolean,
  field: string,
  path: string,
): void {
  if (!isRecord(value)) closureError(`fitted sidecar ${field} must be an object`, path);
  for (const [layer, item] of Object.entries(value)) {
    if (!/^(?:0|[1-9]\d*)$/.test(layer) || !Number.isSafeInteger(Number(layer)) || (requireFitted && !fittedLayers.has(layer)) || !validateValue(item)) {
      closureError(`fitted sidecar ${field} has an invalid layer entry`, path);
    }
  }
}

function validateTopologyCandidates(value: unknown, path: string): void {
  if (!Array.isArray(value)) closureError("fitted sidecar topology_candidates must be an array", path);
  for (const candidate of value) {
    if (!isRecord(candidate) || !exactKeys(candidate, ["fit_mode", "intrinsic_dim", "name", "reason", "score", "viable"])) {
      closureError("fitted sidecar has an invalid topology candidate schema", path);
    }
    if (
      !shortString(candidate.name, 4_096)
      || (candidate.fit_mode !== "pca" && candidate.fit_mode !== "spectral")
      || !positiveInteger(candidate.intrinsic_dim)
      || (candidate.score !== null && !finite(candidate.score))
      || typeof candidate.viable !== "boolean"
      || (candidate.reason !== null && typeof candidate.reason !== "string")
    ) closureError("fitted sidecar has invalid topology candidate values", path);
  }
}

function validateComponents(value: unknown, path: string): void {
  if (value === null) return;
  if (!isRecord(value) || Object.keys(value).length === 0) {
    closureError("fitted sidecar components must be null or a non-empty object", path);
  }
  for (const [coordinate, item] of Object.entries(value)) {
    if (
      !coordinate
      || !isRecord(item)
      || !exactKeys(item, ["alpha", "selector", "tensor_sha256"])
      || !shortString(item.selector, 4_096)
      || !finite(item.alpha)
      || typeof item.tensor_sha256 !== "string"
      || !SHA256.test(item.tensor_sha256)
    ) closureError("fitted sidecar has invalid component provenance", path);
  }
}

interface ValidatedSidecar extends Record<string, unknown> {
  domain: Record<string, unknown>;
  fitted_layers: number[];
  mahalanobis_share_per_layer: Record<string, unknown>;
  origin_per_layer: Record<string, unknown>;
  feature_space: string;
  model_fingerprint: string | null;
  context_binding_sha256: string | null;
  model_source_fingerprint: string | null;
}

function validateSidecar(
  input: unknown,
  path: string,
  manifold: ManifoldClosure,
  expectedNodesSha256: ReadonlySet<string>,
  tensorIdentity: ParsedDrowseTensorFilename,
): ValidatedSidecar {
  if (!isRecord(input)) closureError("fitted sidecar must be a JSON object", path);
  const versionKey = Object.hasOwn(input, "drowse_version")
    ? "drowse_version"
    : Object.hasOwn(input, "polythetic_version") ? "polythetic_version" : "saklas_version";
  if (!exactKeys(input, [
    "bake_policy", "baseline_prompts_sha256", "capture_render_sha256",
    "capture_sha256", "capture_version", "components", "context_binding_sha256",
    "diagnostics", "domain",
    "feature_space", "fit_mode", "fit_policy_version", "fitted_layers",
    "format_version", "hyperparams", "mahalanobis_share_per_layer", "method",
    "model_fingerprint", "model_source_fingerprint", "name", "node_count",
    "node_kinds", "node_labels", "node_roles", "node_spread_per_layer",
    "nodes_sha256", "origin_per_layer", "rbf_smoothing_per_layer",
    "resolved_fit_mode", "sae_fingerprint", "sae_full_coverage", "sae_ids_by_layer",
    "sae_release", "sae_revision", versionKey, "share_metric",
    "sigma_field_per_layer", "source_model_fingerprint", "source_model_id",
    "subspace_metric", "topology_candidates", "topology_winner",
    "transfer_quality_estimate",
  ])) closureError("fitted sidecar does not match the exact manifold v10 schema", path);
  const value = versionKey !== "drowse_version"
    ? { ...input, drowse_version: input[versionKey] }
    : input;
  if (versionKey !== "drowse_version") delete value[versionKey];
  if (
    value.format_version !== MANIFOLD_FORMAT_VERSION
    || value.name !== manifold.name
    || value.fit_mode !== manifold.fitMode
    || value.node_count !== manifold.labels.length
    || !sameStrings(value.node_labels, manifold.labels)
    || !sameNullableStrings(value.node_roles, manifold.roles)
    || !sameNullableStrings(value.node_kinds, manifold.kinds)
  ) closureError("fitted sidecar identity does not match manifold.json", path);
  const methodModes = typeof value.method === "string" ? METHOD_FIT_MODES[value.method] : undefined;
  if (methodModes === undefined || !methodModes.has(manifold.fitMode)) {
    closureError("fitted sidecar has an invalid method/fit_mode combination", path);
  }
  if (
    !shortString(value.method, 1_024)
    || !shortString(value.drowse_version, 1_024)
    || !shortString(value.feature_space, 4_096)
    || typeof value.sae_full_coverage !== "boolean"
  ) closureError("fitted sidecar metadata types are invalid", path);
  validateHyperparams(manifold.fitMode, value.hyperparams, path);
  validateDiagnostics(value.diagnostics, path);
  for (const field of [
    "nodes_sha256", "sae_release", "sae_revision", "sae_fingerprint",
    "model_fingerprint", "model_source_fingerprint", "capture_sha256", "share_metric",
    "subspace_metric", "resolved_fit_mode", "topology_winner", "bake_policy",
    "source_model_id", "source_model_fingerprint",
  ]) {
    const item = value[field];
    if (item !== null && (typeof item !== "string" || !item)) {
      closureError(`fitted sidecar ${field} must be a non-empty string or null`, path);
    }
  }
  for (const field of [
    "context_binding_sha256", "capture_render_sha256", "baseline_prompts_sha256",
  ]) {
    const item = value[field];
    if (item !== null && (typeof item !== "string" || !SHA256.test(item))) {
      closureError(`fitted sidecar ${field} must be a SHA-256 digest or null`, path);
    }
  }
  if (value.capture_version !== null && !nonNegativeInteger(value.capture_version)) {
    closureError("fitted sidecar capture_version is invalid", path);
  }
  if (value.fit_policy_version !== null && !nonNegativeInteger(value.fit_policy_version)) {
    closureError("fitted sidecar fit_policy_version is invalid", path);
  }
  if (value.transfer_quality_estimate !== null && !finite(value.transfer_quality_estimate)) {
    closureError("fitted sidecar transfer quality is invalid", path);
  }
  if (typeof value.nodes_sha256 !== "string" || !expectedNodesSha256.has(value.nodes_sha256)) {
    closureError("fitted sidecar node hash does not match its closure", path);
  }
  if (!isRecord(value.domain)) closureError("fitted sidecar domain is invalid", path);
  validateDomain(value.domain, path);
  if ((manifold.fitMode === "authored" || manifold.fitMode === "baked") && !deepEqual(value.domain, manifold.value.domain)) {
    closureError("fitted sidecar domain does not match manifold.json", path);
  }
  if (!Array.isArray(value.fitted_layers) || value.fitted_layers.length === 0 || !value.fitted_layers.every(nonNegativeInteger) || !sameNumbers(value.fitted_layers, [...new Set(value.fitted_layers)].sort((left, right) => left - right))) {
    closureError("fitted sidecar layer map is invalid", path);
  }
  const layerKeys = new Set((value.fitted_layers as number[]).map(String));
  validateLayerMap(value.node_spread_per_layer, layerKeys, false, (item) => finite(item) && item >= 0, "node_spread_per_layer", path);
  validateLayerMap(value.mahalanobis_share_per_layer, layerKeys, true, positiveFinite, "mahalanobis_share_per_layer", path);
  if (!isRecord(value.mahalanobis_share_per_layer) || !sameSet(new Set(Object.keys(value.mahalanobis_share_per_layer)), layerKeys)) {
    closureError("fitted sidecar Mahalanobis shares do not match its layer map", path);
  }
  validateLayerMap(value.origin_per_layer, layerKeys, true, (item) => Array.isArray(item) && item.length > 0 && item.every(finite), "origin_per_layer", path);
  validateLayerMap(value.sae_ids_by_layer, layerKeys, false, (item) => typeof item === "string" && item.length > 0, "sae_ids_by_layer", path);
  validateLayerMap(value.rbf_smoothing_per_layer, layerKeys, true, (item) => (
    isRecord(item)
    && exactKeys(item, ["edf", "gcv", "lambda"])
    && Object.values(item).every(finite)
  ), "rbf_smoothing_per_layer", path);
  validateLayerMap(value.sigma_field_per_layer, layerKeys, true, (item) => (
    isRecord(item)
    && exactKeys(item, ["lambda", "sigma_max", "sigma_mean", "sigma_min"])
    && Object.values(item).every((entry) => finite(entry) && entry >= 0)
  ), "sigma_field_per_layer", path);
  validateTopologyCandidates(value.topology_candidates, path);
  validateComponents(value.components, path);

  const isAuto = manifold.fitMode === "auto";
  const hasAutoProvenance = (
    value.resolved_fit_mode !== null
    && value.topology_winner !== null
    && Array.isArray(value.topology_candidates)
    && value.topology_candidates.length > 0
  );
  if (isAuto !== hasAutoProvenance) {
    closureError("fitted sidecar has inconsistent auto-topology provenance", path);
  }
  if (!isAuto && (
    value.resolved_fit_mode !== null
    || value.topology_winner !== null
    || (value.topology_candidates as unknown[]).length > 0
  )) closureError("fitted sidecar non-auto fit carries topology provenance", path);

  const isMerge = value.method === "merge";
  if (isMerge) {
    if (manifold.fitMode !== "baked" || value.components === null || value.bake_policy !== MERGE_BAKE_POLICY) {
      closureError("fitted sidecar has incomplete merge provenance", path);
    }
  } else if (value.components !== null || value.bake_policy !== null) {
    closureError("fitted sidecar non-merge fit carries merge provenance", path);
  }

  const isTransfer = value.method === "manifold_procrustes_transfer";
  const transferValues = [value.source_model_id, value.source_model_fingerprint, value.transfer_quality_estimate];
  if (isTransfer) {
    if (value.source_model_id === null || value.source_model_fingerprint === null) {
      closureError("fitted sidecar has incomplete transfer provenance", path);
    }
  } else if (transferValues.some((item) => item !== null)) {
    closureError("fitted sidecar non-transfer fit carries transfer provenance", path);
  }

  const saeValues = [value.sae_release, value.sae_revision, value.sae_fingerprint];
  const saeIds = value.sae_ids_by_layer as Record<string, unknown>;
  if (value.feature_space === "raw") {
    if (saeValues.some((item) => item !== null) || Object.keys(saeIds).length > 0 || value.sae_full_coverage !== false) {
      closureError("fitted sidecar raw fit carries SAE provenance", path);
    }
  } else if (
    !(value.feature_space as string).startsWith("sae-")
    || value.sae_release === null
    || Object.keys(saeIds).length === 0
  ) {
    closureError("fitted sidecar has incomplete SAE provenance", path);
  }

  const matchesVariant = tensorIdentity.variant === "raw"
    ? value.feature_space === "raw" && !isTransfer
    : tensorIdentity.variant === "sae"
      ? value.feature_space === `sae-${tensorIdentity.variantIdentity}`
        && value.sae_release === tensorIdentity.variantIdentity
        && !isTransfer
      : value.feature_space === "raw"
        && isTransfer
        && value.source_model_id === tensorIdentity.variantIdentity;
  if (!matchesVariant) {
    closureError("fitted tensor filename variant does not match its feature-space/source provenance", path);
  }
  return value as ValidatedSidecar;
}

function validateTensorDescription(
  description: SafetensorsDescription,
  path: string,
  sidecar: ValidatedSidecar,
): void {
  const shapes = new Map(description.shapes);
  const domain = validateDomain(sidecar.domain, path);
  const nodeCount = sidecar.node_count as number;
  if (!sameShape(shapes.get("node_coords"), [nodeCount, domain.intrinsicDim])) {
    tensorError(path, "has invalid node_coords shape");
  }
  shapes.delete("node_coords");
  const allowed = new Set([
    "mean", "basis", "node_coords", "affine_map", "node_params", "rbf_weights",
    "poly_coeffs", "coord_offset", "coord_scale", "sigma_rbf_weights", "sigma_poly_coeffs",
  ]);
  const byLayer = new Map<number, Map<string, readonly number[]>>();
  for (const [key, shape] of shapes) {
    const match = /^layer_(0|[1-9]\d{0,9})\.([a-z_]+)$/.exec(key);
    if (match === null || !allowed.has(match[2])) tensorError(path, `has invalid tensor key ${JSON.stringify(key)}`);
    const layer = Number(match[1]);
    const fields = byLayer.get(layer) ?? new Map<string, readonly number[]>();
    if (fields.has(match[2])) tensorError(path, `has duplicate tensor field ${JSON.stringify(key)}`);
    fields.set(match[2], shape);
    byLayer.set(layer, fields);
  }
  if (!sameSet(new Set([...byLayer.keys()].map(String)), new Set(sidecar.fitted_layers.map(String)))) {
    tensorError(path, "layer keys do not match the sidecar layer map");
  }
  let hiddenDim: number | null = null;
  let curved: boolean | null = null;
  const curvedLayers = new Set<string>();
  for (const [layer, fields] of byLayer) {
    const mean = fields.get("mean");
    const basis = fields.get("basis");
    if (mean === undefined || basis === undefined || mean.length !== 1 || basis.length !== 2 || basis[0] < 1 || basis[0] > basis[1] || mean[0] !== basis[1]) {
      tensorError(path, `layer ${layer} has invalid mean/basis shapes`);
    }
    const [rank, dimension] = basis;
    if (hiddenDim === null) hiddenDim = dimension;
    else if (hiddenDim !== dimension) tensorError(path, "has inconsistent hidden dimensions");
    const isCurved = fields.has("node_params");
    if (curved === null) curved = isCurved;
    else if (curved !== isCurved) tensorError(path, "mixes affine and curved layers");
    if (isCurved) {
      curvedLayers.add(String(layer));
      const required = ["mean", "basis", "node_params", "rbf_weights", "poly_coeffs", "coord_offset", "coord_scale"];
      const sigma = ["sigma_rbf_weights", "sigma_poly_coeffs"];
      if (!required.every((field) => fields.has(field)) || [...fields.keys()].some((field) => !required.includes(field) && !sigma.includes(field))) {
        tensorError(path, `layer ${layer} has invalid curved tensor keys`);
      }
      if (fields.has(sigma[0]) !== fields.has(sigma[1])) tensorError(path, `layer ${layer} has incomplete sigma fields`);
      if (
        !sameShape(fields.get("node_params"), [nodeCount, domain.embedDim])
        || !sameShape(fields.get("rbf_weights"), [nodeCount, rank])
        || !sameShape(fields.get("poly_coeffs"), [domain.embedDim + 1, rank])
        || !sameShape(fields.get("coord_offset"), [domain.embedDim])
        || !sameShape(fields.get("coord_scale"), [domain.embedDim])
      ) tensorError(path, `layer ${layer} has invalid curved tensor shapes`);
      if (sidecar.feature_space === "raw" && !fields.has("sigma_rbf_weights")) tensorError(path, `raw curved layer ${layer} lacks sigma fields`);
      if (fields.has("sigma_rbf_weights") && (
        !sameShape(fields.get("sigma_rbf_weights"), [nodeCount, 1])
        || !sameShape(fields.get("sigma_poly_coeffs"), [domain.embedDim + 1, 1])
      )) tensorError(path, `layer ${layer} has invalid sigma shapes`);
    } else {
      const allowedAffine = new Set(["mean", "basis", "node_coords", "affine_map"]);
      if ([...fields.keys()].some((field) => !allowedAffine.has(field)) || !sameShape(fields.get("node_coords"), [nodeCount, rank])) {
        tensorError(path, `layer ${layer} has invalid affine tensor keys or shapes`);
      }
      const affineMap = fields.get("affine_map");
      if (affineMap === undefined && domain.embedDim !== rank) tensorError(path, `layer ${layer} needs affine_map`);
      if (affineMap !== undefined && !sameShape(affineMap, [domain.embedDim, rank])) tensorError(path, `layer ${layer} has invalid affine_map`);
    }
  }
  if (!sameSet(new Set(Object.keys(sidecar.origin_per_layer)), curvedLayers)) {
    tensorError(path, "origins do not match curved layers");
  }
  if (!Object.values(sidecar.origin_per_layer).every((origin) => numericVector(origin, domain.intrinsicDim))) {
    tensorError(path, "has invalid origin coordinates");
  }
}

function validateDomain(value: unknown, path: string): DomainShape {
  if (!isRecord(value)) closureError("manifold domain must be an object", path);
  if (value.type === "box") {
    if (!exactKeys(value, ["axes", "type"]) || !Array.isArray(value.axes) || value.axes.length === 0) closureError("box domain has an invalid schema", path);
    let embedDim = 0;
    for (const axis of value.axes) {
      if (!isRecord(axis) || !exactKeys(axis, ["hi", "lo", "name", "period", "periodic"]) || !shortString(axis.name, 1_024) || typeof axis.periodic !== "boolean" || !positiveFinite(axis.period) || !finite(axis.lo) || !finite(axis.hi) || (axis.hi as number) <= (axis.lo as number)) {
        closureError("box domain has an invalid axis", path);
      }
      embedDim += axis.periodic ? 2 : 1;
    }
    return { intrinsicDim: value.axes.length, embedDim };
  }
  if (value.type === "sphere") {
    if (!exactKeys(value, ["dim", "type"]) || !positiveInteger(value.dim)) closureError("sphere domain has an invalid schema", path);
    return { intrinsicDim: value.dim as number, embedDim: (value.dim as number) + 1 };
  }
  if (value.type === "custom") {
    if (!exactKeys(value, ["bounds", "embed_dim", "type"]) || !positiveInteger(value.embed_dim)) closureError("custom domain has an invalid schema", path);
    const dimension = value.embed_dim as number;
    if (value.bounds !== null && (!Array.isArray(value.bounds) || value.bounds.length !== dimension || !value.bounds.every((row) => Array.isArray(row) && row.length === 2 && finite(row[0]) && finite(row[1]) && row[1] > row[0]))) {
      closureError("custom domain bounds are invalid", path);
    }
    return { intrinsicDim: dimension, embedDim: dimension };
  }
  closureError("manifold domain type is invalid", path);
}

function validateSource(value: unknown): DrowseArchiveManifest["source"] {
  if (!isRecord(value) || !exactKeys(value, ["repository", "revision", "uri"]) || !shortString(value.uri, 4_096)) {
    manifestError("pack.json source does not match the exact v1 schema");
  }
  for (const field of ["repository", "revision"] as const) {
    if (value[field] !== null && !shortString(value[field], 1_024)) manifestError(`pack.json source.${field} is invalid`);
  }
  if ((value.uri as string).startsWith("hf://")) {
    if (
      typeof value.repository !== "string"
      || typeof value.revision !== "string"
      || !/^[0-9a-f]{40}$/.test(value.revision)
      || value.uri !== `hf://${value.repository}@${value.revision}`
    ) {
      manifestError("Hugging Face source provenance requires matching immutable fields");
    }
  } else if (value.repository !== null || value.revision !== null) {
    manifestError("non-Hugging Face source provenance cannot name a repository or revision");
  }
  return { uri: value.uri as string, repository: value.repository as string | null, revision: value.revision as string | null };
}

function validatePackSourceMatchesManifold(source: DrowseArchiveManifest["source"], manifoldSource: string): void {
  if (source.uri !== manifoldSource) closureError("pack.json source provenance does not match manifold.json");
  if (manifoldSource.startsWith("hf://") && source.uri !== `hf://${source.repository}@${source.revision}`) {
    closureError("pack.json Hugging Face provenance does not match manifold.json");
  }
}

function archivePath(value: unknown, root: "manifolds" | "templates"): string {
  if (typeof value !== "string") manifestError(`pack.json ${root} path must be a string`);
  validateManifestPath(value);
  const parts = value.split("/");
  if (parts.length !== 3 || parts[0] !== root || !NAME.test(parts[1]) || !NAME.test(parts[2])) {
    manifestError(`pack.json path ${JSON.stringify(value)} must be ${root}/<namespace>/<name>`);
  }
  if (root === "manifolds" && RESERVED_MANIFOLD_NAMESPACES.has(parts[1])) {
    manifestError(`pack.json manifold namespace ${JSON.stringify(parts[1])} is reserved`);
  }
  return value;
}

function identity(value: string, root: "manifolds" | "templates"): [string, string] {
  const parts = value.split("/");
  if (parts.length !== 3 || parts[0] !== root) manifestError(`invalid ${root} identity`);
  return [parts[1], parts[2]];
}

function validateManifestPath(path: string): void {
  if (!path || !/^[\x20-\x7e]+$/.test(path) || path.includes("\\") || path.startsWith("/") || /^[A-Za-z]:/.test(path) || path.split("/").some((part) => part === "" || part === "." || part === "..")) {
    throw new DrowseArchiveError("ARCHIVE_PATH_INVALID", `archive path ${JSON.stringify(path)} is not canonical`, path);
  }
}

function requiredEntry(archive: ZipArchive, path: string): ZipEntry {
  const entry = archive.byPath.get(path);
  if (entry === undefined) manifestMismatch(`archive has no entry ${JSON.stringify(path)}`, path);
  return entry;
}

function nodeFilename(index: number, label: string): string {
  return `${String(index).padStart(2, "0")}_${label}.json`;
}

function isCanonicalTensorFilename(filename: string): boolean {
  return parseDrowseTensorFilename(filename) !== null;
}

export function parseDrowseTensorFilename(filename: string): ParsedDrowseTensorFilename | null {
  if (!filename.endsWith(".safetensors")) return null;
  const stem = filename.slice(0, -".safetensors".length);
  const separators = ["_sae-", "_from-"];
  const matches = separators
    .map((separator) => ({ separator, index: stem.indexOf(separator) }))
    .filter((match) => match.index >= 0)
    .sort((left, right) => left.index - right.index);
  if (matches.length > 1) return null;
  const match = matches[0];
  const modelPart = match === undefined ? stem : stem.slice(0, match.index);
  const variantPart = match === undefined ? null : stem.slice(match.index + match.separator.length);
  const decodedModelPart = decodeTensorComponent(modelPart);
  if (decodedModelPart === null || encodeTensorComponent(decodedModelPart) !== modelPart) return null;
  const modelId = decodeSafeModel(decodedModelPart);
  if (modelId === null) return null;
  let variantIdentity: string | null = null;
  let safeVariantIdentity: string | null = null;
  if (variantPart !== null) {
    const decodedVariantPart = decodeTensorComponent(variantPart);
    if (decodedVariantPart === null || encodeTensorComponent(decodedVariantPart) !== variantPart) return null;
    variantIdentity = decodeBase32Identifier(decodedVariantPart);
    if (variantIdentity === null) return null;
    safeVariantIdentity = decodedVariantPart;
  }
  return {
    modelId,
    safeModelId: decodedModelPart,
    variant: match === undefined ? "raw" : match.separator === "_sae-" ? "sae" : "from",
    variantIdentity,
    safeVariantIdentity,
  };
}

export function drowseTensorFilename(modelId: string): string {
  if (!modelId) throw new TypeError("Drowse fitted model ID must not be empty");
  const bytes = new TextEncoder().encode(modelId);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  const safeModel = `_z${btoa(binary)
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replace(/=+$/, "")}`;
  return `${encodeTensorComponent(safeModel)}.safetensors`;
}

function decodeTensorComponent(value: string): string | null {
  if (/%(?!25|5F(?:sae|from)-)/.test(value)) return null;
  return value
    .replaceAll("%5Fsae-", "_sae-")
    .replaceAll("%5Ffrom-", "_from-")
    .replaceAll("%25", "%");
}

function encodeTensorComponent(value: string): string {
  return value
    .replaceAll("%", "%25")
    .replaceAll("_sae-", "%5Fsae-")
    .replaceAll("_from-", "%5Ffrom-");
}

function decodeSafeModel(value: string): string | null {
  if (!/^_z[A-Za-z0-9_-]+$/.test(value)) return null;
  const encoded = value.slice(2);
  if (encoded.length % 4 === 1) return null;
  try {
    const binary = atob(encoded.replaceAll("-", "+").replaceAll("_", "/") + "=".repeat((4 - encoded.length % 4) % 4));
    const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
    const decoded = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    if (!decoded) return null;
    const canonical = btoa(String.fromCharCode(...new TextEncoder().encode(decoded)))
      .replaceAll("+", "-")
      .replaceAll("/", "_")
      .replace(/=+$/, "");
    return `_z${canonical}` === value ? decoded : null;
  } catch {
    return null;
  }
}

function decodeBase32Identifier(value: string): string | null {
  if (!/^_z[a-z2-7]+$/.test(value)) return null;
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
  const encoded = value.slice(2).toUpperCase();
  let bits = 0;
  let accumulator = 0;
  const bytes: number[] = [];
  for (const char of encoded) {
    const digit = alphabet.indexOf(char);
    if (digit < 0) return null;
    accumulator = (accumulator << 5) | digit;
    bits += 5;
    if (bits >= 8) {
      bits -= 8;
      bytes.push((accumulator >>> bits) & 0xff);
      accumulator &= (1 << bits) - 1;
    }
  }
  if (bits > 0 && accumulator !== 0) return null;
  try {
    const decoded = new TextDecoder("utf-8", { fatal: true }).decode(Uint8Array.from(bytes));
    if (!decoded) return null;
    let output = "";
    bits = 0;
    accumulator = 0;
    for (const byte of new TextEncoder().encode(decoded)) {
      accumulator = (accumulator << 8) | byte;
      bits += 8;
      while (bits >= 5) {
        bits -= 5;
        output += alphabet[(accumulator >>> bits) & 31];
        accumulator &= (1 << bits) - 1;
      }
    }
    if (bits > 0) output += alphabet[(accumulator << (5 - bits)) & 31];
    return `_z${output.toLowerCase()}` === value ? decoded : null;
  } catch {
    return null;
  }
}

function slugValue(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

function occurrences(value: string, needle: string): number {
  if (!needle) return 0;
  let count = 0;
  let offset = 0;
  while ((offset = value.indexOf(needle, offset)) !== -1) {
    count += 1;
    offset += needle.length;
  }
  return count;
}

function totalPayloadBytes(manifest: DrowseArchiveManifest): number {
  return Object.values(manifest.files).reduce((total, file) => total + file.size, 0);
}

function exactKeys(record: Record<string, unknown>, expected: readonly string[]): boolean {
  const keys = Object.keys(record).sort();
  const sortedExpected = [...expected].sort();
  return keys.length === sortedExpected.length && keys.every((key, index) => key === sortedExpected[index]);
}

function sameSet<T>(left: Set<T>, right: Set<T>): boolean {
  return left.size === right.size && [...left].every((value) => right.has(value));
}

function sameStrings(left: unknown, right: unknown): boolean {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => typeof value === "string" && value === right[index]);
}

function sameNullableStrings(left: unknown, right: unknown): boolean {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => (value === null || typeof value === "string") && value === right[index]);
}

function sameShape(left: readonly number[] | undefined, right: readonly number[]): boolean {
  return left !== undefined && left.length === right.length && left.every((value, index) => value === right[index]);
}

function sameNumbers(left: unknown, right: unknown): boolean {
  return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => typeof value === "number" && value === right[index]);
}

function numericVector(value: unknown, length: number): boolean {
  return Array.isArray(value) && value.length === length && value.every(finite);
}

function deepEqual(left: unknown, right: unknown): boolean {
  if (left === right) return true;
  if (Array.isArray(left) && Array.isArray(right)) return left.length === right.length && left.every((value, index) => deepEqual(value, right[index]));
  if (isRecord(left) && isRecord(right)) {
    const keys = Object.keys(left);
    return keys.length === Object.keys(right).length && keys.every((key) => key in right && deepEqual(left[key], right[key]));
  }
  return false;
}

class PythonFloat {
  constructor(readonly value: number) {}
}

function pythonFloatDomain(value: unknown): unknown {
  if (!isRecord(value)) return value;
  if (value.type === "box" && Array.isArray(value.axes)) {
    return {
      ...value,
      axes: value.axes.map((axis) => isRecord(axis) ? {
        ...axis,
        hi: new PythonFloat(axis.hi as number),
        lo: new PythonFloat(axis.lo as number),
        period: new PythonFloat(axis.period as number),
      } : axis),
    };
  }
  if (value.type === "custom" && Array.isArray(value.bounds)) {
    return {
      ...value,
      bounds: value.bounds.map((row) => Array.isArray(row)
        ? row.map((coordinate) => new PythonFloat(coordinate as number))
        : row),
    };
  }
  return value;
}

function pythonFloatHyperparams(value: Record<string, unknown>): Record<string, unknown> {
  const floatKeys = new Set(["var_threshold", "bandwidth", "persistence_frac", "smoothing"]);
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [
    key,
    floatKeys.has(key) && typeof item === "number" ? new PythonFloat(item) : item,
  ]));
}

function pythonCanonicalJsonBytes(value: unknown): Uint8Array {
  return new TextEncoder().encode(pythonCanonicalJson(value));
}

function pythonCanonicalJson(value: unknown): string {
  if (value instanceof PythonFloat) return pythonFloatRepr(value.value);
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return pythonJsonString(value);
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new TypeError("canonical JSON requires finite numbers");
    return Number.isInteger(value) ? String(value) : pythonFloatRepr(value);
  }
  if (Array.isArray(value)) return `[${value.map(pythonCanonicalJson).join(",")}]`;
  if (isRecord(value)) {
    return `{${Object.keys(value).sort().map((key) => (
      `${pythonJsonString(key)}:${pythonCanonicalJson(value[key])}`
    )).join(",")}}`;
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
  const exponentSign = exponent >= 0 ? "+" : "-";
  return `${sign}${mantissa}e${exponentSign}${Math.abs(exponent).toString().padStart(2, "0")}`;
}

function shortString(value: unknown, maximumBytes: number): value is string {
  return typeof value === "string" && value.length > 0 && !value.includes("\0") && new TextEncoder().encode(value).length <= maximumBytes;
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function positiveFinite(value: unknown): value is number {
  return finite(value) && value > 0;
}

function positiveInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function nonNegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function deepFreeze<T>(value: T): T {
  if (value !== null && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child);
  }
  return value;
}

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function manifestError(message: string): never {
  throw new DrowseArchiveError("MANIFEST_INVALID", message, "pack.json");
}

function manifestMismatch(message: string, path?: string): never {
  throw new DrowseArchiveError("MANIFEST_MISMATCH", message, path);
}

function closureError(message: string, path?: string): never {
  throw new DrowseArchiveError("CLOSURE_INVALID", message, path);
}

function tensorError(path: string, message: string): never {
  throw new DrowseArchiveError("SAFETENSORS_INVALID", `fitted tensor ${JSON.stringify(path)} ${message}`, path);
}
