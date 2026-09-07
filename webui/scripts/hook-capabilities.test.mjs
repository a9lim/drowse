import assert from "node:assert/strict";
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

try {
  const {
    assertHookFeaturesSupported,
    requiredHookFeatures,
    AFFINE_FEASIBILITY_CAPABILITIES,
  } = await server.ssrLoadModule("/src/hosted/runtime/hookCapabilities.ts");

  assert.deepEqual([...requiredHookFeatures("0.5 honest")], ["rank_one"]);
  assert.doesNotThrow(() =>
    assertHookFeaturesSupported("0.5 honest", AFFINE_FEASIBILITY_CAPABILITIES)
  );
  assert.doesNotThrow(() =>
    assertHookFeaturesSupported(
      "0.5 honest~sycophantic + 0.25 focused",
      AFFINE_FEASIBILITY_CAPABILITIES,
    )
  );
  assert.doesNotThrow(() =>
    assertHookFeaturesSupported("0.5 !honest", AFFINE_FEASIBILITY_CAPABILITIES)
  );

  const required = requiredHookFeatures(
    "0.4 honest~sycophantic@response&when:mood[1]>0.2 + !sae/42 + persona%pirate + jlens/fake",
  );
  assert.deepEqual([...required].sort(), [
    "ablation",
    "curved_manifold",
    "jlens",
    "multi_term",
    "phase_trigger",
    "probe_gate",
    "projection",
    "rank_one",
    "sae",
  ]);
  assert.throws(
    () => assertHookFeaturesSupported(
      "honest + !sae/42@response",
      AFFINE_FEASIBILITY_CAPABILITIES,
    ),
    (error) => {
      assert.equal(error.code, "DROWSE_HOOK_FEATURE_UNAVAILABLE");
      assert.deepEqual(error.missingFeatures, [
        "phase_trigger",
        "sae",
      ]);
      return true;
    },
  );

  console.log("Hook capability checks passed");
} finally {
  await server.close();
}
