import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const COMMIT = /^[0-9a-f]{40}$/;
const SPDX = /^[A-Za-z0-9][A-Za-z0-9.+-]{0,127}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const GPT2_LICENSE_SHA256 = "0dbeda4bc78823b1d67ec0ee414d8690ee4cad826bc9c147b215adc3af202436";

export const HOSTED_MODEL_PROVENANCE_PATHS = Object.freeze([
  "LICENSE.model",
  "MODEL-LICENSE.json",
  "README.source.md",
  "drowse-build.json",
  "hosted-artifacts.json",
]);
const FORK_MANIFEST_URL = new URL(
  "../../browser-runtime/forks/manifest.json",
  import.meta.url,
);

export function assertCatalogLicenseProvenanceClosure(document) {
  if (
    !document || typeof document !== "object" || !Array.isArray(document.models) ||
    document.models.length === 0
  ) {
    throw new Error("catalog has no model license/provenance closure");
  }
  for (const model of document.models) {
    if (!["Apache-2.0", "Gemma", "LicenseRef-GPT2-Modified-MIT"].includes(model.license)) {
      throw new Error(`${String(model.id)} declares an unsupported model license`);
    }
    if (!Array.isArray(model.variants) || model.variants.length === 0) {
      throw new Error(`${String(model.id)} has no variant license/provenance closure`);
    }
    for (const variant of model.variants) {
      for (const path of HOSTED_MODEL_PROVENANCE_PATHS) {
        const matches = (variant.files ?? []).filter((file) => file.path === path);
        if (matches.length !== 1 || matches[0].role !== "configuration") {
          throw new Error(
            `${String(variant.id)} must include exactly one configuration artifact ${path}`,
          );
        }
      }
      for (const pack of variant.packs ?? []) {
        if (
          typeof pack.license !== "string" || !SPDX.test(pack.license) ||
          typeof pack.sourceRepository !== "string" || !pack.sourceRepository ||
          typeof pack.sourceRevision !== "string" || !COMMIT.test(pack.sourceRevision)
        ) {
          throw new Error(`${String(pack.id)} has incomplete license or source provenance`);
        }
      }
    }
  }
}

export async function validateLocalHostedModelClosure(directory, manifest, model, runtime) {
  const expectedBuildToolchain = await loadExpectedBuildToolchain(
    runtime.toolchain?.forkManifestSha256,
    model.architecture,
  );
  const hostedManifestBytes = await readFile(resolve(directory, "hosted-artifacts.json"));
  const declared = [
    ...manifest.files.map((file) => ({ ...file, url: file.path })),
    {
      path: "hosted-artifacts.json",
      role: "configuration",
      url: "hosted-artifacts.json",
      bytes: hostedManifestBytes.byteLength,
      sha256: createHash("sha256").update(hostedManifestBytes).digest("hex"),
    },
  ];
  const bytesByUrl = new Map(await Promise.all(declared.map(async (file) => [
    file.url,
    new Uint8Array(await readFile(resolve(directory, file.path))),
  ])));
  const sourceReadme = new TextDecoder("utf-8", { fatal: true }).decode(
    bytesByUrl.get("README.source.md"),
  );
  validateDownloadedLicenseProvenanceClosure(
    {
      id: model.id,
      modelType: model.modelType,
      license: modelCardLicense(sourceReadme) === "gemma" ? "Gemma" :
        modelCardLicense(sourceReadme) === "mit" ? "LicenseRef-GPT2-Modified-MIT" : "Apache-2.0",
    },
    {
      id: model.id,
      structuredHookProfile: manifest.structuredHookProfile,
      runtimeIdentity: {
        sourceModel: model.sourceRepository,
        sourceRevision: model.sourceRevision,
        convertedManifestSha256: model.manifestSha256,
        quantization: model.quantization,
        tokenizerSha256: model.tokenizerSha256,
        chatTemplateSha256: model.chatTemplateSha256,
        modelLibrarySha256: model.librarySha256,
        runtimeAbi: runtime.runtimeAbi,
        hookAbi: runtime.hookAbi,
        hiddenSize: model.hiddenSize,
        layerMap: model.layerMap,
      },
      files: declared,
      packs: [],
    },
    bytesByUrl,
    expectedBuildToolchain,
  );
}

export async function loadExpectedBuildToolchain(expectedManifestSha256, architecture) {
  if (!SHA256.test(expectedManifestSha256 ?? "")) {
    throw new Error("runtime lock has no exact fork-manifest binding");
  }
  const bytes = await readFile(FORK_MANIFEST_URL);
  if (createHash("sha256").update(bytes).digest("hex") !== expectedManifestSha256) {
    throw new Error("runtime lock fork-manifest binding differs from the checked-in manifest");
  }
  let manifest;
  try {
    manifest = JSON.parse(bytes.toString("utf8"));
  } catch (error) {
    throw new Error("fork overlay manifest is not valid JSON", { cause: error });
  }
  return expectedBuildToolchain(manifest, architecture);
}

export function expectedBuildToolchain(manifest, architecture) {
  const overlays = new Map(
    (manifest?.overlays ?? []).map((overlay) => [overlay?.id, overlay]),
  );
  const mlc = overlays.get("mlc-llm-drowse");
  const tvm = overlays.get("tvm-webgpu-readonly");
  if (
    !mlc || !tvm || !COMMIT.test(mlc.baseCommit ?? "") ||
    !COMMIT.test(manifest?.tvmCommit ?? "") ||
    !COMMIT.test(manifest?.tvmFfiCommit ?? "")
  ) {
    throw new Error("fork overlay manifest has incomplete build-toolchain identity");
  }
  validateDigestMap(mlc.resultFiles, "MLC overlay result files");
  validateDigestMap(tvm.resultFiles, "TVM overlay result files");
  const result = {
    mlcLlmBaseCommit: mlc.baseCommit,
    mlcOverlayFiles: structuredClone(mlc.resultFiles),
    tvmCommit: manifest.tvmCommit,
    tvmFfiCommit: manifest.tvmFfiCommit,
    tvmOverlayFiles: structuredClone(tvm.resultFiles),
  };
  if (["gpt2", "gpt_neox", "qwen3_5"].includes(architecture)) {
    const base = manifest.baseModels?.[architecture];
    validateDigestMap(base?.modelAdapterFiles, `${architecture} adapter files`);
    validateDigestMap(base?.modelSourceFiles, `${architecture} source files`);
    Object.assign(result, structuredClone(base));
  }
  return result;
}

export function validateDownloadedLicenseProvenanceClosure(
  model,
  variant,
  bytesByUrl,
  expectedBuildToolchain,
) {
  assertCatalogLicenseProvenanceClosure({ models: [{ ...model, variants: [variant] }] });
  const read = (path) => {
    const file = variant.files.find((candidate) => candidate.path === path);
    const bytes = file ? bytesByUrl.get(file.url) : null;
    if (!(bytes instanceof Uint8Array)) {
      throw new Error(`${String(variant.id)} preflight did not retain ${path}`);
    }
    if (
      bytes.byteLength !== file.bytes ||
      createHash("sha256").update(bytes).digest("hex") !== file.sha256
    ) {
      throw new Error(`${String(variant.id)} retained ${path} differs from the catalog`);
    }
    return bytes;
  };
  const text = (path) => new TextDecoder("utf-8", { fatal: true }).decode(read(path));
  const json = (path) => {
    try {
      return JSON.parse(text(path));
    } catch (error) {
      throw new Error(`${String(variant.id)} ${path} is not valid UTF-8 JSON`, { cause: error });
    }
  };

  const licenseText = text("LICENSE.model");
  const sourceReadme = text("README.source.md");
  const declaredLicense = modelCardLicense(sourceReadme);
  const license = json("MODEL-LICENSE.json");
  if (model.license === "Apache-2.0") {
    if (!isCompleteApacheLicense(licenseText) || declaredLicense !== "apache-2.0") {
      throw new Error(`${String(variant.id)} does not include complete Apache-2.0 license text`);
    }
    exactKeys(
      license,
      ["schemaVersion", "spdx", "declaredIn", "licenseText", "licenseTextSource"],
      `${String(variant.id)} model license metadata`,
    );
    if (
      license.schemaVersion !== 1 || license.spdx !== "apache-2.0" ||
      license.declaredIn !== "README.md" || license.licenseText !== "LICENSE.model" ||
      !["source-snapshot", "drowse-standard-text"].includes(license.licenseTextSource)
    ) {
      throw new Error(`${String(variant.id)} model license metadata is inconsistent`);
    }
  } else if (model.license === "LicenseRef-GPT2-Modified-MIT") {
    exactKeys(license, [
      "schemaVersion", "licenseId", "modelCardLicense", "declaredIn",
      "licenseText", "licenseTextSource", "sourceUrl",
    ], `${String(variant.id)} model license metadata`);
    if (
      declaredLicense !== "mit" || variant.runtimeIdentity.sourceModel !== "openai-community/gpt2" ||
      license.schemaVersion !== 1 || license.licenseId !== model.license ||
      license.modelCardLicense !== "mit" || license.declaredIn !== "README.md" ||
      license.licenseText !== "LICENSE.model" || license.licenseTextSource !== "upstream-repository" ||
      license.sourceUrl !== "https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/LICENSE" ||
      createHash("sha256").update(licenseText).digest("hex") !== GPT2_LICENSE_SHA256
    ) throw new Error(`${String(variant.id)} GPT-2 license provenance is inconsistent`);
  } else {
    if (declaredLicense !== "gemma" || licenseText !== sourceReadme) {
      throw new Error(`${String(variant.id)} does not include its Gemma terms notice`);
    }
    exactKeys(
      license,
      [
        "schemaVersion",
        "licenseId",
        "declaredIn",
        "licenseText",
        "licenseTextSource",
        "termsUrl",
      ],
      `${String(variant.id)} model license metadata`,
    );
    if (
      license.schemaVersion !== 1 || license.licenseId !== "gemma" ||
      license.declaredIn !== "README.md" || license.licenseText !== "LICENSE.model" ||
      license.licenseTextSource !== "model-card-terms-reference" ||
      license.termsUrl !== "https://ai.google.dev/gemma/terms"
    ) {
      throw new Error(`${String(variant.id)} model license metadata is inconsistent`);
    }
  }

  const build = json("drowse-build.json");
  exactKeys(build, [
    "schemaVersion",
    "runtimeAbi",
    "hookAbi",
    "structuredHookProfile",
    "thinkingProfile",
    "architecture",
    "quantization",
    "contextWindowSize",
    "prefillChunkSize",
    "source",
    "toolchain",
    "files",
    ...("modelType" in build ? ["modelType", "promptPolicy"] : []),
    ...("stateAbi" in build ? ["stateAbi"] : []),
    ...("weightEncoding" in build ? ["weightEncoding"] : []),
  ], `${String(variant.id)} build provenance`);
  exactKeys(build.source, [
    "repository",
    "revision",
    "files",
    "chatTemplateSha256",
  ], `${String(variant.id)} build source provenance`);
  if (
    build.schemaVersion !== 1 || build.runtimeAbi !== variant.runtimeIdentity.runtimeAbi ||
    build.hookAbi !== variant.runtimeIdentity.hookAbi ||
    build.structuredHookProfile !== variant.structuredHookProfile ||
    !["qwen3", "llama", "gemma3_text", "gpt2", "gpt_neox", "qwen3_5"].includes(build.architecture) ||
    (build.modelType ?? "chat") !== (model.modelType ?? "chat") ||
    (build.architecture === "qwen3_5" && build.stateAbi !== "kv-rnn-v1") ||
    (build.quantization === "q0f32" && build.weightEncoding !== "raw") ||
    build.quantization !== variant.runtimeIdentity.quantization ||
    build.source?.repository !== variant.runtimeIdentity.sourceModel ||
    build.source?.revision !== variant.runtimeIdentity.sourceRevision ||
    !SHA256.test(build.source?.chatTemplateSha256 ?? "") ||
    !Array.isArray(build.source?.files) || build.source.files.length === 0 ||
    !Array.isArray(build.files) || build.files.length === 0
  ) {
    throw new Error(`${String(variant.id)} build provenance differs from its runtime identity`);
  }
  exactKeys(build.toolchain, [
    "mlcLlmBaseCommit",
    "mlcOverlayFiles",
    "tvmCommit",
    "tvmFfiCommit",
    "tvmOverlayFiles",
    ...(["gpt2", "gpt_neox", "qwen3_5"].includes(build.architecture)
      ? ["modelAdapterFiles", "modelSourceFiles"] : []),
  ], `${String(variant.id)} build toolchain provenance`);
  if (canonicalJson(build.toolchain) !== canonicalJson(expectedBuildToolchain)) {
    throw new Error(`${String(variant.id)} build toolchain differs from the runtime lock`);
  }
  const sourceFiles = new Map(build.source.files.map((file) => [file?.path, file]));
  if (!sourceFiles.has("README.md")) {
    throw new Error(`${String(variant.id)} build provenance omits the source model card`);
  }
  if (license.licenseTextSource === "source-snapshot" && !sourceFiles.has("LICENSE")) {
    throw new Error(`${String(variant.id)} build provenance omits the source license file`);
  }
  validateFileRecords(build.source.files, `${String(variant.id)} source provenance`);
  validateFileRecords(build.files, `${String(variant.id)} build provenance`);
  const sourceReadmeRecord = sourceFiles.get("README.md");
  const sourceReadmeCatalog = variant.files.find((file) => file.path === "README.source.md");
  if (
    sourceReadmeRecord.bytes !== sourceReadmeCatalog.bytes ||
    sourceReadmeRecord.sha256 !== sourceReadmeCatalog.sha256
  ) {
    throw new Error(`${String(variant.id)} source model card snapshot differs from build provenance`);
  }
  if (license.licenseTextSource === "source-snapshot") {
    const sourceLicenseRecord = sourceFiles.get("LICENSE");
    const sourceLicenseCatalog = variant.files.find((file) => file.path === "LICENSE.model");
    if (
      sourceLicenseRecord.bytes !== sourceLicenseCatalog.bytes ||
      sourceLicenseRecord.sha256 !== sourceLicenseCatalog.sha256
    ) {
      throw new Error(`${String(variant.id)} source license snapshot differs from build provenance`);
    }
  }

  const hosted = json("hosted-artifacts.json");
  exactKeys(hosted, [
    "schemaVersion", "runtimeAbi", "hookAbi", "structuredHookProfile", "thinkingProfile",
    "architecture", "quantization", "contextWindowSize", "prefillChunkSize", "source",
    "hiddenSize", "layerMap", "files",
    ...("modelType" in hosted ? ["modelType"] : []),
  ], `${String(variant.id)} hosted artifact manifest`);
  if (
    hosted.schemaVersion !== 1 || hosted.runtimeAbi !== variant.runtimeIdentity.runtimeAbi ||
    hosted.hookAbi !== variant.runtimeIdentity.hookAbi ||
    hosted.structuredHookProfile !== variant.structuredHookProfile ||
    hosted.architecture !== build.architecture ||
    (hosted.modelType ?? "chat") !== (build.modelType ?? "chat") ||
    canonicalJson(hosted.thinkingProfile) !== canonicalJson(build.thinkingProfile) ||
    hosted.quantization !== variant.runtimeIdentity.quantization ||
    hosted.contextWindowSize !== build.contextWindowSize ||
    hosted.prefillChunkSize !== build.prefillChunkSize ||
    hosted.source?.repository !== variant.runtimeIdentity.sourceModel ||
    hosted.source?.revision !== variant.runtimeIdentity.sourceRevision ||
    hosted.hiddenSize !== variant.runtimeIdentity.hiddenSize ||
    JSON.stringify(hosted.layerMap) !== JSON.stringify(variant.runtimeIdentity.layerMap) ||
    !Array.isArray(hosted.files)
  ) {
    throw new Error(`${String(variant.id)} hosted manifest differs from its runtime identity`);
  }
  const expected = variant.files
    .filter((file) => file.path !== "hosted-artifacts.json")
    .map(({ path, role, bytes, sha256 }) => ({ path, role, bytes, sha256 }))
    .sort(compareFiles);
  const actual = hosted.files.map(({ path, role, bytes, sha256 }) => ({
    path,
    role,
    bytes,
    sha256,
  })).sort(compareFiles);
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${String(variant.id)} hosted manifest does not close its catalog files`);
  }
  const builtFiles = new Map(build.files.map((file) => [file.path, file]));
  for (const catalogFile of expected.filter(
    (file) => file.path !== "drowse-build.json" && file.role !== "model_library",
  )) {
    const built = builtFiles.get(catalogFile.path);
    if (
      !built || built.bytes !== catalogFile.bytes ||
      built.sha256 !== catalogFile.sha256
    ) {
      throw new Error(
        `${String(variant.id)} build manifest does not bind ${catalogFile.path}`,
      );
    }
  }
}

function validateFileRecords(files, label) {
  const paths = new Set();
  for (const file of files) {
    exactKeys(file, ["path", "bytes", "sha256"], label);
    if (
      !file || typeof file !== "object" || typeof file.path !== "string" || !file.path ||
      paths.has(file.path) || !Number.isSafeInteger(file.bytes) || file.bytes < 1 ||
      typeof file.sha256 !== "string" || !SHA256.test(file.sha256)
    ) {
      throw new Error(`${label} has an invalid file closure`);
    }
    paths.add(file.path);
  }
}

function validateDigestMap(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} is invalid`);
  }
  const entries = Object.entries(value);
  if (
    entries.length === 0 ||
    entries.some(([path, digest]) => !path || !SHA256.test(digest))
  ) {
    throw new Error(`${label} is invalid`);
  }
}

function modelCardLicense(markdown) {
  const lines = markdown.split(/\r?\n/u);
  if (lines[0]?.trim() !== "---") return null;
  for (const line of lines.slice(1)) {
    if (line.trim() === "---") return null;
    const match = /^license\s*:\s*["']?([^"']+?)["']?\s*$/iu.exec(line);
    if (match) return match[1].trim().toLowerCase();
  }
  return null;
}

function isCompleteApacheLicense(value) {
  return (
    value.length >= 8_000 &&
    /Apache License/iu.test(value) &&
    /Version 2\.0, January 2004/iu.test(value) &&
    /TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION/iu.test(value) &&
    /1\.\s+Definitions\./u.test(value) &&
    /9\.\s+Accepting Warranty or Additional Liability\./u.test(value) &&
    /END OF TERMS AND CONDITIONS/iu.test(value)
  );
}

function compareFiles(left, right) {
  return left.path.localeCompare(right.path) || left.role.localeCompare(right.role);
}

function exactKeys(value, expected, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (
    actual.length !== wanted.length ||
    actual.some((key, index) => key !== wanted[index])
  ) {
    throw new Error(`${label} has unknown or missing fields`);
  }
}

function canonicalJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.entries(value)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
    .join(",")}}`;
}
