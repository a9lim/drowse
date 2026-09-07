import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

const realRuntimeHarness = await readFile(
  fileURLToPath(new URL("./browser-app-runtime.mjs", import.meta.url)),
  "utf8",
);
assert.match(realRuntimeHarness, /maxComputeWorkgroupSizeX: 256/);
assert.match(realRuntimeHarness, /maxComputeInvocationsPerWorkgroup: 256/);
assert.match(realRuntimeHarness, /compileStructuredHookProgram/);
assert.match(realRuntimeHarness, /resolveStructuredHookProfile/);
assert.doesNotMatch(realRuntimeHarness, /profile:\s*\{\s*id:\s*manifest\.structuredHookProfile/);

try {
  const {
    browserArtifactProducer,
    browserBuildProvenance,
    validateBrowserBuildProvenance,
  } = await server.ssrLoadModule("/src/hosted/runtime/buildProvenance.ts");
  const preview = {
    schemaVersion: 1,
    drowseVersion: "5.3.0",
    sourceRevision: "a".repeat(40),
    runtimeAbi: "drowse-web-runtime-v1",
    hookAbi: "post-block-residual-v4",
    channel: "preview",
    verified: false,
  };
  assert.deepEqual(validateBrowserBuildProvenance(preview), preview);
  const producer = browserArtifactProducer(preview);
  assert.equal(producer.drowseVersion, preview.drowseVersion);
  assert.deepEqual(JSON.parse(producer.producerVersion), preview);
  assert.deepEqual(
    validateBrowserBuildProvenance({ ...preview, channel: "release", verified: true }),
    { ...preview, channel: "release", verified: true },
  );
  for (const invalid of [
    { ...preview, sourceRevision: "preview" },
    { ...preview, runtimeAbi: "browser-v1" },
    { ...preview, channel: "preview", verified: true },
    { ...preview, channel: "release", verified: false },
  ]) {
    assert.throws(
      () => validateBrowserBuildProvenance(invalid),
      /invalid or contradictory/,
    );
  }
  assert.throws(
    () => browserBuildProvenance(),
    /did not inject the exact Drowse version/,
  );
} finally {
  await server.close();
}

console.log("browser build provenance tests passed");
