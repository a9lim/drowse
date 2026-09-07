import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const exec = promisify(execFile);
const runtimeLockUrl = new URL("../../browser-runtime/runtime-lock.json", import.meta.url);
const distributionLockUrl = new URL("../../browser-runtime/distribution-lock.json", import.meta.url);
const REPOSITORY = /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/;

export function parseArguments(args) {
  if (args.length > 1 || (args[0] !== undefined && !["--check", "--create"].includes(args[0]))) {
    throw new Error("usage: node provision-hosted-repositories.mjs [--check|--create]");
  }
  return { create: args[0] === "--create" };
}

export function repositoryPlan(runtimeLock, distributionLock) {
  const catalog = catalogRepository(distributionLock.catalogUrl, "catalogUrl");
  const signature = catalogRepository(distributionLock.signatureUrl, "signatureUrl");
  if (catalog !== signature) {
    throw new Error("catalog and signature URLs must use the same Hugging Face repository");
  }
  const models = runtimeLock.models?.map((model) => model.convertedRepository) ?? [];
  const instruments = models.map((repository) => `${repository}-instruments`);
  const repositories = [...new Set([catalog, ...models, ...instruments])];
  if (repositories.some((repository) => !REPOSITORY.test(repository))) {
    throw new Error("runtime locks contain an invalid Hugging Face repository ID");
  }
  const owners = new Set(repositories.map((repository) => repository.split("/", 1)[0]));
  if (owners.size !== 1) {
    throw new Error("catalog and model repositories must share one Hugging Face owner");
  }
  return { owner: [...owners][0], repositories };
}

export async function provisionRepositories({
  create = false,
  runtimeLock,
  distributionLock,
  run = runCommand,
}) {
  const plan = repositoryPlan(runtimeLock, distributionLock);
  if (create) await assertOwnerAccess(plan.owner, run);
  const results = [];
  for (const repository of plan.repositories) {
    let info = await modelInfo(repository, run);
    if (info === null && create) {
      await run("hf", ["repos", "create", repository, "--type", "model", "--exist-ok"]);
      info = await modelInfo(repository, run);
    }
    if (info === null) {
      throw new Error(`missing Hugging Face repository: ${repository}`);
    }
    if (info.private === true || info.gated === true || info.gated === "auto" || info.gated === "manual") {
      throw new Error(`hosted repository must be public and ungated: ${repository}`);
    }
    results.push({ repository, private: info.private === true, gated: info.gated ?? false });
  }
  return results;
}

async function assertOwnerAccess(owner, run) {
  const response = await run("hf", ["auth", "whoami", "--format", "json"]);
  const identity = parseJson(response.stdout, "Hugging Face identity");
  const organizations = Array.isArray(identity.orgs)
    ? identity.orgs.map((org) => typeof org === "string" ? org : org?.name).filter(Boolean)
    : [];
  if (identity.user !== owner && !organizations.includes(owner)) {
    throw new Error(`authenticated Hugging Face account cannot provision ${owner} repositories`);
  }
}

async function modelInfo(repository, run) {
  try {
    const response = await run("hf", [
      "models", "info", repository,
      "--expand", "private,gated", "--format", "json",
    ]);
    return parseJson(response.stdout, repository);
  } catch (error) {
    if (error && typeof error === "object" && "exitCode" in error && error.exitCode === 1) {
      return null;
    }
    throw error;
  }
}

function catalogRepository(value, label) {
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`${label} is not a valid URL`);
  }
  if (url.origin !== "https://huggingface.co" || url.username || url.password) {
    throw new Error(`${label} must use the public Hugging Face origin`);
  }
  const match = /^\/([^/]+)\/([^/]+)\/resolve\/main\/catalog(?:\.sig)?\.json$/.exec(url.pathname);
  if (!match) throw new Error(`${label} has an unsupported catalog path`);
  return `${decodeURIComponent(match[1])}/${decodeURIComponent(match[2])}`;
}

function parseJson(value, label) {
  try {
    return JSON.parse(value);
  } catch (error) {
    throw new Error(`${label} returned invalid JSON`, { cause: error });
  }
}

async function runCommand(command, args) {
  try {
    return await exec(command, args, { encoding: "utf8" });
  } catch (error) {
    if (error && typeof error === "object") {
      error.exitCode = error.code;
    }
    throw error;
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const [runtimeLock, distributionLock] = await Promise.all([
    readFile(runtimeLockUrl, "utf8").then(JSON.parse),
    readFile(distributionLockUrl, "utf8").then(JSON.parse),
  ]);
  const results = await provisionRepositories({ ...options, runtimeLock, distributionLock });
  for (const result of results) console.log(`verified ${result.repository}`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
