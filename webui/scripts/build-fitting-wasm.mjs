import { createHash } from "node:crypto";
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const WASM_BINDGEN_VERSION = "0.2.127";
const OUTPUT_NAMES = [
  "drowse_fitting_wasm.js",
  "drowse_fitting_wasm_bg.wasm",
];
const check = process.argv.includes("--check");
const webuiRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = resolve(webuiRoot, "..");
const crateRoot = resolve(repositoryRoot, "browser-runtime", "fitting-wasm");
const outputRoot = resolve(webuiRoot, "public-hosted", "wasm");
const wasmBindgen = process.env.DROWSE_WASM_BINDGEN || "wasm-bindgen";
const rustFlags = process.env.CARGO_ENCODED_RUSTFLAGS !== undefined
  ? process.env.CARGO_ENCODED_RUSTFLAGS.split("\x1f").filter(Boolean)
  : (process.env.RUSTFLAGS ?? "").split(/\s+/).filter(Boolean);
rustFlags.push(
  `--remap-path-prefix=${homedir()}=/build-user`,
  `--remap-path-prefix=${resolve(process.env.CARGO_HOME || join(homedir(), ".cargo"))}=/cargo`,
  `--remap-path-prefix=${repositoryRoot}=/drowse`,
);

const version = run(wasmBindgen, ["--version"], { capture: true }).trim();
if (version !== `wasm-bindgen ${WASM_BINDGEN_VERSION}`) {
  throw new Error(
    `expected wasm-bindgen ${WASM_BINDGEN_VERSION}, got ${JSON.stringify(version)}`,
  );
}

run("cargo", [
  "build",
  "--locked",
  "--release",
  "--target",
  "wasm32-unknown-unknown",
  "--manifest-path",
  resolve(crateRoot, "Cargo.toml"),
], { env: { ...process.env, CARGO_ENCODED_RUSTFLAGS: rustFlags.join("\x1f") } });

const temporary = await mkdtemp(join(tmpdir(), "drowse-fitting-wasm-"));
try {
  run(wasmBindgen, [
    resolve(
      crateRoot,
      "target",
      "wasm32-unknown-unknown",
      "release",
      "drowse_fitting_wasm.wasm",
    ),
    "--target",
    "web",
    "--typescript",
    "--out-dir",
    temporary,
    "--out-name",
    "drowse_fitting_wasm",
  ]);

  const cargoLock = await readFile(resolve(crateRoot, "Cargo.lock"));
  const files = [];
  for (const name of OUTPUT_NAMES) {
    const bytes = await readFile(resolve(temporary, name));
    if (bytes.includes(Buffer.from(homedir())) || bytes.includes(Buffer.from(repositoryRoot))) {
      throw new Error(`hosted fitting asset contains a machine-specific path: ${name}`);
    }
    files.push({ name, bytes: bytes.byteLength, sha256: sha256(bytes) });
  }
  const manifest = `${JSON.stringify({
    schemaVersion: 1,
    crate: "drowse-fitting-wasm",
    crateVersion: "0.1.0",
    wasmBindgenVersion: WASM_BINDGEN_VERSION,
    cargoLockSha256: sha256(cargoLock),
    files,
  }, null, 2)}\n`;
  await writeFile(resolve(temporary, "fitting-kernel.json"), manifest);

  if (check) {
    for (const name of [...OUTPUT_NAMES, "fitting-kernel.json"]) {
      const [generated, committed] = await Promise.all([
        readFile(resolve(temporary, name)),
        readFile(resolve(outputRoot, name)),
      ]);
      if (!generated.equals(committed)) {
        throw new Error(`hosted fitting asset is stale: ${name}`);
      }
    }
    console.log("Hosted fitting WebAssembly assets are reproducible");
  } else {
    await mkdir(outputRoot, { recursive: true });
    for (const name of [...OUTPUT_NAMES, "fitting-kernel.json"]) {
      await copyFile(resolve(temporary, name), resolve(outputRoot, name));
    }
    console.log(`Wrote ${OUTPUT_NAMES.length + 1} hosted fitting assets`);
  }
} finally {
  await rm(temporary, { recursive: true, force: true });
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: repositoryRoot,
    env: options.env ?? process.env,
    encoding: options.capture ? "utf8" : undefined,
    stdio: options.capture ? ["ignore", "pipe", "pipe"] : "inherit",
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    const detail = options.capture ? `: ${(result.stderr || result.stdout).trim()}` : "";
    throw new Error(`${basename(command)} exited with status ${result.status}${detail}`);
  }
  return options.capture ? result.stdout : "";
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}
