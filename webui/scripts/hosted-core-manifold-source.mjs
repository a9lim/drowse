import { createHash } from "node:crypto";
import { readFile, readdir } from "node:fs/promises";
import { resolve } from "node:path";

const BASELINE_PROMPTS_PATH = "drowse/data/baseline_prompts.json";
const ROOT_ENTRIES = ["LICENSE.txt", "manifold.json", "nodes", "provenance.json"];
const MANIFEST_KEYS = [
  "description",
  "files",
  "fit_mode",
  "format_version",
  "hyperparams",
  "name",
  "nodes",
  "source",
  "tags",
  "template_ref",
];
const PROVENANCE_KEYS = [
  "artifact",
  "authorship",
  "baseline_prompts_path",
  "baseline_prompts_sha256",
  "license",
  "responses_per_node",
  "schema_version",
  "source",
];
const NODE_KEYS = ["kind", "label", "role"];
const LABEL = /^[a-z][a-z0-9_-]{0,63}$/;
const KINDS = new Set([null, "abstract", "concrete", "custom"]);

export async function readCoreManifoldSource(directory, repositoryRoot) {
  const root = resolve(directory);
  const rootEntries = await readdir(root, { withFileTypes: true });
  if (
    rootEntries.map((entry) => entry.name).sort().join("\0") !== ROOT_ENTRIES.join("\0") ||
    rootEntries.some((entry) => entry.name === "nodes" ? !entry.isDirectory() : !entry.isFile())
  ) {
    throw new Error("core manifold source has an unexpected file closure");
  }
  const [manifest, provenance, licenseText, baselineBytes] = await Promise.all([
    readJson(resolve(root, "manifold.json"), "manifold.json"),
    readJson(resolve(root, "provenance.json"), "provenance.json"),
    readFile(resolve(root, "LICENSE.txt"), "utf8"),
    readFile(resolve(repositoryRoot, BASELINE_PROMPTS_PATH)),
  ]);
  const baselinePrompts = parseJson(baselineBytes, BASELINE_PROMPTS_PATH);
  if (
    !Array.isArray(baselinePrompts) || baselinePrompts.length === 0 ||
    baselinePrompts.some((prompt) => typeof prompt !== "string" || !prompt.trim())
  ) {
    throw new Error("canonical baseline prompts are invalid");
  }
  exactKeys(manifest, MANIFEST_KEYS, "manifold.json");
  if (
    manifest.format_version !== 10 || manifest.fit_mode !== "pca" ||
    typeof manifest.name !== "string" || !LABEL.test(manifest.name.split(".")[0]) ||
    !/^[a-z][a-z0-9._-]{0,63}$/.test(manifest.name) ||
    typeof manifest.description !== "string" || !manifest.description.trim() ||
    manifest.source !== "local" || manifest.template_ref !== null ||
    !plainObject(manifest.files) || Object.keys(manifest.files).length !== 0
  ) {
    throw new Error("core manifold source is not an unfitted local v10 PCA manifold");
  }
  exactKeys(manifest.hyperparams, ["max_dim", "var_threshold"], "manifold hyperparams");
  if (
    !Number.isSafeInteger(manifest.hyperparams.max_dim) || manifest.hyperparams.max_dim < 1 ||
    manifest.hyperparams.max_dim > 64 ||
    typeof manifest.hyperparams.var_threshold !== "number" ||
    !Number.isFinite(manifest.hyperparams.var_threshold) ||
    manifest.hyperparams.var_threshold <= 0 || manifest.hyperparams.var_threshold > 1
  ) {
    throw new Error("core manifold PCA hyperparameters are invalid");
  }
  if (
    !Array.isArray(manifest.tags) || manifest.tags.length === 0 ||
    manifest.tags.some((tag) => typeof tag !== "string" || !LABEL.test(tag)) ||
    new Set(manifest.tags).size !== manifest.tags.length
  ) {
    throw new Error("core manifold tags are invalid");
  }
  if (!Array.isArray(manifest.nodes) || manifest.nodes.length < 2) {
    throw new Error("core manifold needs at least two nodes");
  }
  const labels = new Set();
  for (const [index, node] of manifest.nodes.entries()) {
    exactKeys(node, NODE_KEYS, `core manifold node ${index}`);
    if (
      typeof node.label !== "string" || !LABEL.test(node.label) || labels.has(node.label) ||
      (node.role !== null && (typeof node.role !== "string" || !LABEL.test(node.role))) ||
      !KINDS.has(node.kind)
    ) {
      throw new Error(`core manifold node ${index} is invalid`);
    }
    labels.add(node.label);
  }
  exactKeys(provenance, PROVENANCE_KEYS, "provenance.json");
  if (
    provenance.schema_version !== 1 || provenance.artifact !== `default/${manifest.name}` ||
    typeof provenance.authorship !== "string" || !provenance.authorship.trim() ||
    typeof provenance.license !== "string" || !/^[A-Za-z0-9.-]+$/.test(provenance.license) ||
    provenance.source !== manifest.source || provenance.baseline_prompts_path !== BASELINE_PROMPTS_PATH ||
    provenance.baseline_prompts_sha256 !== sha256(baselineBytes) ||
    provenance.responses_per_node !== baselinePrompts.length ||
    !licenseText.startsWith(`SPDX-License-Identifier: ${provenance.license}\n`)
  ) {
    throw new Error("core manifold source provenance is invalid");
  }
  const expectedNodeFiles = manifest.nodes.map((node, index) =>
    `${String(index).padStart(2, "0")}_${node.label}.json`
  );
  const nodeEntries = await readdir(resolve(root, "nodes"), { withFileTypes: true });
  if (
    nodeEntries.some((entry) => !entry.isFile()) ||
    nodeEntries.map((entry) => entry.name).sort().join("\0") !== [...expectedNodeFiles].sort().join("\0")
  ) {
    throw new Error("core manifold node closure does not match manifold.json");
  }
  const nodes = await Promise.all(manifest.nodes.map(async (node, index) => {
    const statements = await readJson(
      resolve(root, "nodes", expectedNodeFiles[index]),
      expectedNodeFiles[index],
    );
    if (
      !Array.isArray(statements) || statements.length !== provenance.responses_per_node ||
      statements.some((statement) =>
        typeof statement !== "string" || !statement.trim() || statement.includes("\0") ||
        statement.length > 4_096
      ) || new Set(statements).size !== statements.length
    ) {
      throw new Error(`core manifold node ${node.label} corpus is invalid`);
    }
    return { label: node.label, role: node.role, kind: node.kind, statements };
  }));
  for (let index = 0; index < provenance.responses_per_node; index += 1) {
    if (new Set(nodes.map((node) => node.statements[index])).size !== nodes.length) {
      throw new Error(`core manifold node responses collide at baseline prompt ${index}`);
    }
  }
  return {
    namespace: "default",
    name: manifest.name,
    description: manifest.description,
    fitMode: "pca",
    hyperparams: { ...manifest.hyperparams },
    nodes,
    source: { uri: manifest.source, repository: null, revision: null },
    tags: [...manifest.tags],
  };
}

async function readJson(path, label) {
  return parseJson(await readFile(path), label);
}

function parseJson(bytes, label) {
  try {
    return JSON.parse(bytes.toString("utf8"));
  } catch (error) {
    throw new Error(`${label} is not valid JSON`, { cause: error });
  }
}

function exactKeys(value, expected, label) {
  if (!plainObject(value) || Object.keys(value).sort().join("\0") !== [...expected].sort().join("\0")) {
    throw new Error(`${label} does not match its exact schema`);
  }
}

function plainObject(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
