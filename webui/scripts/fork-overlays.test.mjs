import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  applyOverlay,
  loadManifest,
  parseArguments,
  verifyOverlayArtifacts,
} from "../../browser-runtime/forks/apply-fork-overlay.mjs";

const root = resolve(import.meta.dirname, "../..");
const manifestPath = join(root, "browser-runtime/forks/manifest.json");
const manifest = await loadManifest(manifestPath);
const webLlmPatch = await readFile(
  join(root, "browser-runtime/forks/web-llm-drowse.patch"),
  "utf8",
);
const compilerDockerfile = await readFile(
  join(root, "browser-runtime/compiler/Dockerfile"),
  "utf8",
);

assert.deepEqual(parseArguments(["mlc-llm-drowse", "/tmp/repo"]), {
  apply: false,
  help: false,
  id: "mlc-llm-drowse",
  repository: "/tmp/repo",
});
assert.equal(manifest.overlays.length, 3);
assert.equal(manifest.hookAbi, "post-block-residual-v4");
assert.equal(manifest.structuredHookProfile, "standard-v3");
assert.equal(manifest.exactReadoutAbi, "exact-readout-v1");
assert.ok(
  webLlmPatch.includes('"drowse_sae_jump_relu_readout_accumulate"'),
  "the exact JumpReLU VM function must be loaded into the WebLLM registry",
);
assert.ok(
  webLlmPatch.includes('"<start_of_turn>"'),
  "the role-header ABI must preserve Gemma turn templates",
);
assert.equal(manifest.tvmCommit, "68ba2b31c2a6d202fcd44e5780ef10ce08721dd5");
assert.equal(manifest.tvmFfiCommit, "12dbf053b3d9ba4ebd9da3123b1aeca79cf74229");
assert.equal(manifest.emccVersion, "3.1.56");
assert.equal(manifest.emccPlatform, "linux/amd64");
assert.equal(manifest.llvmVersion, "18.1.8");
assert.deepEqual(manifest.llvmArtifacts, [
  {
    platform: "linux/amd64",
    archive: "clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz",
    sha256: "54ec30358afcc9fb8aa74307db3046f5187f9fb89fb37064cdde906e062ebf36",
  },
  {
    platform: "linux/arm64",
    archive: "clang+llvm-18.1.8-aarch64-linux-gnu.tar.xz",
    sha256: "dcaa1bebbfbb86953fdfbdc7f938800229f75ad26c5c9375ef242edad737d999",
  },
]);
const compilerArguments = new Set(compilerDockerfile.split("\n"));
for (const [name, value] of [
  ["TVM_COMMIT", manifest.tvmCommit],
  ["TVM_FFI_COMMIT", manifest.tvmFfiCommit],
  ["EMSDK_COMMIT", manifest.emsdkCommit],
  ["EMCC_VERSION", manifest.emccVersion],
]) {
  assert.ok(compilerArguments.has(`ARG ${name}=${value}`));
}
for (const artifact of manifest.llvmArtifacts) {
  const suffix = artifact.platform.endsWith("amd64") ? "AMD64" : "ARM64";
  assert.ok(compilerArguments.has(`ARG LLVM_ARCHIVE_${suffix}=${artifact.archive}`));
  assert.ok(compilerArguments.has(`ARG LLVM_SHA256_${suffix}=${artifact.sha256}`));
}
assert.ok(
  compilerArguments.has(`ARG MLC_COMMIT=${manifest.overlays[1].baseCommit}`),
);
assert.ok(compilerArguments.has("ENV EMSDK_ARCH=x86_64"));
assert.ok(compilerArguments.has("COPY forks/verify-tvm-attention.py /opt/drowse/verify-tvm-attention.py"));
for (const adapter of ["base_model_adapters.py", "qwen35_adapter.py", "hybrid_state_kernels.py"]) {
  assert.ok(compilerArguments.has(`COPY forks/${adapter} /opt/drowse/${adapter}`));
}
assert.match(compilerDockerfile, /RUN python \/opt\/drowse\/verify-tvm-attention\.py/);
assert.match(compilerDockerfile, /libc6:amd64 libstdc\+\+6:amd64 libtinfo5:amd64 zlib1g:amd64/);
assert.equal(
  compilerDockerfile.match(/--quantization q4f16_1/g)?.length,
  3,
  "compiler image must gate every q4f16 architecture",
);
assert.equal(
  compilerDockerfile.match(/--quantization q4f32_1/g)?.length,
  1,
  "compiler image must gate Gemma's overflow-safe q4f32 build",
);
const bindingVerifier = String.raw`
import importlib.util
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("drowse_compiler", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.verify_geometry_shader_bindings(sys.stdin.buffer.read())
`;
const shader = (bindings) =>
  Buffer.from(
    `// Function: drowse_geometry_measurements_kernel\n${Array.from(
      { length: bindings },
      (_, index) => `@group(0) @binding(${index}) var<storage, read> b${index} : array<f32>;`,
    ).join("\n")}\n@compute\n`,
  );
for (const compilerScript of [
  "compile-tiny-webgpu.py",
  "build-production-webgpu.py",
]) {
  const path = join(root, "browser-runtime/forks", compilerScript);
  const accepted = spawnSync("python3", ["-c", bindingVerifier, path], {
    input: shader(8),
    encoding: "utf8",
  });
  assert.equal(accepted.status, 0, accepted.stderr);
  const rejected = spawnSync("python3", ["-c", bindingVerifier, path], {
    input: shader(9),
    encoding: "utf8",
  });
  assert.notEqual(rejected.status, 0);
  assert.match(rejected.stderr, /uses 9 storage bindings; expected 8/);
}
for (const overlay of manifest.overlays) {
  await verifyOverlayArtifacts(overlay, manifestPath);
}

const directory = await mkdtemp(join(tmpdir(), "drowse-overlay-test-"));
const artifactDirectory = await mkdtemp(join(tmpdir(), "drowse-overlay-artifacts-"));
try {
  git(directory, ["init", "-q"]);
  git(directory, ["config", "user.email", "test@drowse.local"]);
  git(directory, ["config", "user.name", "Drowse Test"]);
  await writeFile(join(directory, "fixture.txt"), "base\n");
  git(directory, ["add", "fixture.txt"]);
  git(directory, ["commit", "-qm", "fixture"]);
  const fixtureOverlay = {
    ...manifest.overlays[0],
    baseCommit: "0".repeat(40),
    baseTree: git(directory, ["rev-parse", "HEAD^{tree}"]),
  };
  await assert.rejects(
    () => applyOverlay({ overlay: fixtureOverlay, repository: directory, manifestPath }),
    /requires exact base/,
  );
  await writeFile(join(directory, "untracked.txt"), "dirty\n");
  const actualBase = {
    ...fixtureOverlay,
    baseCommit: git(directory, ["rev-parse", "HEAD"]),
  };
  await assert.rejects(
    () => applyOverlay({ overlay: actualBase, repository: directory, manifestPath }),
    /must be clean/,
  );
  await rm(join(directory, "untracked.txt"));

  await writeFile(join(directory, "fixture.txt"), "patched\n");
  const fixturePatch = execFileSync("git", ["diff", "--binary", "HEAD"], {
    cwd: directory,
    encoding: "utf8",
  });
  git(directory, ["checkout", "--", "fixture.txt"]);
  const fixturePatchPath = join(artifactDirectory, "fixture.patch");
  await writeFile(fixturePatchPath, fixturePatch);
  const validOverlay = {
    ...actualBase,
    patch: "fixture.patch",
    patchBytes: Buffer.byteLength(fixturePatch),
    patchSha256: sha256(fixturePatch),
    resultFiles: { "fixture.txt": sha256("patched\n") },
  };
  await assert.rejects(
    () =>
      applyOverlay({
        overlay: { ...validOverlay, resultFiles: {} },
        repository: directory,
        manifestPath: join(artifactDirectory, "manifest.json"),
      }),
    /result files do not match/,
  );
  const result = await applyOverlay({
    overlay: validOverlay,
    repository: directory,
    manifestPath: join(artifactDirectory, "manifest.json"),
    apply: true,
  });
  assert.equal(result.applied, true);
  assert.equal(await readFile(join(directory, "fixture.txt"), "utf8"), "patched\n");
} finally {
  await rm(directory, { recursive: true, force: true });
  await rm(artifactDirectory, { recursive: true, force: true });
}

const storedManifest = JSON.parse(await readFile(manifestPath, "utf8"));
assert.equal(storedManifest.status, "feasibility-overlay");
console.log("fork overlay checks passed");

function git(cwd, args) {
  return execFileSync("git", args, { cwd, encoding: "utf8" }).trim();
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}
