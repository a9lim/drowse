#!/usr/bin/env node

import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { lstat, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const defaultRepositoryRoot = fileURLToPath(new URL("../../", import.meta.url));
const COMMIT = /^[0-9a-f]{40}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const REPOSITORY = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
const EVIDENCE_PATH = /^evidence\/[a-z0-9][a-z0-9._/-]*\.json$/;

export function parseAttestationArguments(args) {
  const result = { repository: null, signerWorkflow: null };
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (!["--repository", "--signer-workflow"].includes(argument)) {
      throw new Error(`unexpected argument ${argument}`);
    }
    const value = args[++index];
    if (!value || value.startsWith("--")) {
      throw new Error(`${argument} requires a value`);
    }
    const key = argument === "--repository" ? "repository" : "signerWorkflow";
    if (result[key] !== null) throw new Error(`${argument} may be supplied only once`);
    result[key] = value;
  }
  if (!REPOSITORY.test(result.repository ?? "")) {
    throw new Error("--repository must be OWNER/REPOSITORY");
  }
  const expectedWorkflow = `${result.repository}/.github/workflows/ci.yml`;
  if (result.signerWorkflow !== expectedWorkflow) {
    throw new Error(`--signer-workflow must be ${expectedWorkflow}`);
  }
  return result;
}

export async function collectEvidenceAttestationSubjects(
  repositoryRoot = defaultRepositoryRoot,
) {
  const browserRuntimeRoot = resolve(repositoryRoot, "browser-runtime");
  const [runtime, authoring, benchmark] = await Promise.all([
    readJson(resolve(browserRuntimeRoot, "runtime-feasibility-evidence.json")),
    readJson(resolve(browserRuntimeRoot, "authoring-evidence.json")),
    readJson(resolve(browserRuntimeRoot, "benchmark-evidence.json")),
  ]);
  const subjects = [];
  for (const [kind, revision, entries] of [
    ["runtime", runtime.drowseRevision, runtime.gates],
    ["authoring", authoring.drowseRevision, authoring.evidence],
  ]) {
    if (!entries || typeof entries !== "object" || Array.isArray(entries)) {
      throw new Error(`${kind} evidence references are invalid`);
    }
    for (const [evidenceType, reference] of Object.entries(entries)) {
      if (reference === null) continue;
      requireCommit(revision, `${kind} evidence revision`);
      exactKeys(reference, ["path", "sha256"], `${evidenceType} evidence reference`);
      if (
        !EVIDENCE_PATH.test(reference.path) ||
        reference.path.split("/").some((part) => part === "..") ||
        !SHA256.test(reference.sha256)
      ) {
        throw new Error(`${evidenceType} evidence reference is invalid`);
      }
      const path = resolve(browserRuntimeRoot, ...reference.path.split("/"));
      requireContainedPath(path, resolve(browserRuntimeRoot, "evidence"));
      const bytes = await readRegularContainedFile(
        path,
        resolve(browserRuntimeRoot, "evidence"),
        `${evidenceType} evidence record`,
      );
      const digest = sha256(bytes);
      if (digest !== reference.sha256) {
        throw new Error(`${evidenceType} evidence digest does not match its reference`);
      }
      const record = parseJson(bytes, `${evidenceType} evidence record`);
      if (
        record.evidenceType !== evidenceType ||
        record.drowseRevision !== revision
      ) {
        throw new Error(`${evidenceType} evidence source binding is invalid`);
      }
      subjects.push({
        kind: "gate",
        evidenceType,
        path,
        displayPath: relative(repositoryRoot, path).split(sep).join("/"),
        sha256: digest,
        sourceDigest: revision,
      });
    }
  }
  if (!Array.isArray(benchmark.runs)) {
    throw new Error("benchmark evidence runs are invalid");
  }
  for (const [index, run] of benchmark.runs.entries()) {
    if (!run || typeof run !== "object" || Array.isArray(run)) {
      throw new Error(`benchmark run ${index} is invalid`);
    }
    requireCommit(run.drowseRevision, `benchmark run ${index} revision`);
    if (benchmark.drowseRevision !== run.drowseRevision) {
      throw new Error(`benchmark run ${index} source binding is invalid`);
    }
    const bytes = canonicalJsonBytes(run);
    subjects.push({
      kind: "benchmark",
      evidenceType: "physicalBrowserBenchmark",
      bytes,
      displayPath: `browser-runtime/benchmark-evidence.json#runs/${index}`,
      sha256: sha256(bytes),
      sourceDigest: run.drowseRevision,
    });
  }
  const seen = new Set();
  return subjects.filter((subject) => {
    const key = `${subject.sha256}:${subject.sourceDigest}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export async function verifyEvidenceAttestations({
  repository,
  signerWorkflow,
  repositoryRoot = defaultRepositoryRoot,
  execute = execFileAsync,
}) {
  if (!REPOSITORY.test(repository ?? "")) {
    throw new Error("attestation repository must be OWNER/REPOSITORY");
  }
  if (signerWorkflow !== `${repository}/.github/workflows/ci.yml`) {
    throw new Error("release evidence must be signed by this repository's CI workflow");
  }
  const subjects = await collectEvidenceAttestationSubjects(repositoryRoot);
  if (subjects.length === 0) return [];
  const temporary = await mkdtemp(join(tmpdir(), "drowse-evidence-attestations-"));
  try {
    const verified = [];
    for (const [index, subject] of subjects.entries()) {
      let path = subject.path;
      if (subject.kind === "benchmark") {
        path = join(temporary, `benchmark-run-${index}-${subject.sha256}.json`);
        await writeFile(path, subject.bytes, { flag: "wx", mode: 0o600 });
      }
      const args = [
        "attestation",
        "verify",
        path,
        "--repo",
        repository,
        "--signer-workflow",
        signerWorkflow,
        "--signer-digest",
        subject.sourceDigest,
        "--source-digest",
        subject.sourceDigest,
        "--predicate-type",
        "https://slsa.dev/provenance/v1",
        "--format",
        "json",
      ];
      let stdout;
      try {
        ({ stdout } = await execute("gh", args, {
          cwd: repositoryRoot,
          encoding: "utf8",
          maxBuffer: 4 * 1024 * 1024,
          timeout: 2 * 60 * 1000,
          windowsHide: true,
        }));
      } catch (error) {
        throw new Error(
          `${subject.displayPath} has no valid CI release-evidence attestation`,
          { cause: error },
        );
      }
      let result;
      try {
        result = JSON.parse(stdout);
      } catch (error) {
        throw new Error(
          `${subject.displayPath} attestation verification returned invalid JSON`,
          { cause: error },
        );
      }
      if (!Array.isArray(result) || result.length === 0) {
        throw new Error(`${subject.displayPath} has no verified CI attestation`);
      }
      verified.push({
        evidenceType: subject.evidenceType,
        path: subject.displayPath,
        sha256: subject.sha256,
        sourceDigest: subject.sourceDigest,
      });
    }
    return verified;
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

function canonicalJsonBytes(value) {
  return Buffer.from(`${JSON.stringify(sortJson(value), null, 2)}\n`);
}

function sortJson(value) {
  if (Array.isArray(value)) return value.map(sortJson);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.keys(value).sort().map((key) => [key, sortJson(value[key])]),
  );
}

async function readJson(path) {
  return parseJson(await readFile(path), basename(path));
}

async function readRegularContainedFile(path, root, label) {
  const [rootRealPath, parentRealPath, info] = await Promise.all([
    realpath(root),
    realpath(resolve(path, "..")),
    lstat(path),
  ]);
  const parentRelative = relative(rootRealPath, parentRealPath);
  if (
    parentRelative === ".." ||
    parentRelative.startsWith(`..${sep}`) ||
    isAbsolute(parentRelative) ||
    info.isSymbolicLink() ||
    !info.isFile() ||
    info.size > 4 * 1024 * 1024
  ) {
    throw new Error(`${label} must be a regular file inside browser-runtime/evidence`);
  }
  return readFile(path);
}

function parseJson(bytes, label) {
  try {
    return JSON.parse(bytes.toString("utf8"));
  } catch (error) {
    throw new Error(`${label} is not valid JSON`, { cause: error });
  }
}

function requireCommit(value, label) {
  if (!COMMIT.test(value ?? "")) throw new Error(`${label} is invalid`);
}

function requireContainedPath(path, root) {
  const value = relative(root, path);
  if (
    !value ||
    value === ".." ||
    value.startsWith(`..${sep}`) ||
    isAbsolute(value)
  ) {
    throw new Error("release evidence path escapes browser-runtime/evidence");
  }
}

function exactKeys(value, keys, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (
    actual.length !== expected.length ||
    actual.some((key, index) => key !== expected[index])
  ) {
    throw new Error(`${label} has unknown or missing fields`);
  }
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function main() {
  const options = parseAttestationArguments(process.argv.slice(2));
  const verified = await verifyEvidenceAttestations(options);
  process.stdout.write(`${JSON.stringify({ verified: verified.length, subjects: verified })}\n`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
