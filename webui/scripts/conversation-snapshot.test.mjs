import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({
  root,
  configFile: false,
  optimizeDeps: { noDiscovery: true, include: [] },
  appType: "custom",
  logLevel: "silent",
  server: { middlewareMode: true, watch: null },
});

try {
  const {
    buildConversationImportPlan,
    isConversationSnapshot,
    runConversationImportTransaction,
  } = await server.ssrLoadModule("/src/lib/conversationSnapshot.ts");

  const samplingState = {
    temperature: 0.7,
    top_p: 0.95,
    top_k: null,
    max_tokens: 256,
    seed: null,
    system_prompt: "",
    stop_sequences: "",
    logit_bias_text: "",
    presence_penalty: 0,
    frequency_penalty: 0,
    thinking: false,
    return_top_k: 8,
    user_role: "user",
    assistant_role: "assistant",
  };
  const snapshot = {
    version: 7,
    savedAt: "2026-08-29T12:00:00.000Z",
    model_id: "fixture/model",
    session_id: "default",
    tree: {
      tree_format: 2,
      drowse_version: "fixture",
      model_id: "fixture/model",
      session_id: "default",
      name: null,
      root_id: "root",
      active_node_id: "root",
      rev: 0,
      nodes: [],
      children_of: {},
      cast: {},
    },
    steerRack: [{
      name: "local/calm",
      mode: "subspace",
      ablate: true,
      coords: [1],
      label: "calm",
      variant: "raw",
      trigger: "BOTH",
      enabled: true,
    }, {
      name: "jlens/fixture",
      mode: "jlens",
      ablate: true,
      alpha: 0.3,
      trigger: "RESPONSE",
      enabled: true,
    }, {
      name: "sae/17",
      mode: "sae",
      ablate: false,
      alpha: 0.4,
      trigger: "BOTH",
      enabled: true,
    }, {
      name: "local/emotions",
      mode: "manifold",
      blend: 0.4,
      onto: 0.7,
      coords: [0.2, 0.8],
      label: "happy",
      variant: "sae-release-a",
      trigger: "RESPONSE",
      enabled: true,
    }],
    subspaceAlong: 0.5,
    customSteeringExpression: null,
    probeRack: {
      sortMode: "change",
      active: ["mood alias", "latent alias"],
      entries: [{
        name: "mood alias",
        request: {
          selector: "local/emotions:sae-release-a",
          name: "mood alias",
          top_n: 4,
        },
        sparkline: [0.1, 0.2],
        current: 0.2,
        previous: 0.1,
      }, {
        name: "latent alias",
        request: { selector: "sae/17", name: "latent alias" },
        sparkline: [0.3],
        current: 0.3,
        previous: 0.25,
      }],
    },
    highlightState: {
      target: "mood alias",
      compareTarget: "latent alias",
      compareTwo: true,
      smoothBlend: false,
    },
    samplingState,
  };

  assert.equal(isConversationSnapshot(snapshot, Object.keys(samplingState)), true);

  const pythonSnapshot = structuredClone(snapshot);
  pythonSnapshot.tree = JSON.parse(readFileSync(new URL(
    "../../tests/fixtures/saklas_v5_3/legacy_tree.json", import.meta.url,
  ), "utf8"));
  pythonSnapshot.tree.drowse_version = pythonSnapshot.tree.saklas_version;
  delete pythonSnapshot.tree.saklas_version;
  pythonSnapshot.tree.model_id = pythonSnapshot.model_id;
  assert.equal(pythonSnapshot.tree.session_id, null);
  assert.equal(isConversationSnapshot(pythonSnapshot, Object.keys(samplingState)), true);
  assert.equal(buildConversationImportPlan(pythonSnapshot).snapshot.tree.session_id, null);
  for (const sessionId of [undefined, "unrelated-session"]) {
    const invalidIdentity = structuredClone(pythonSnapshot);
    invalidIdentity.tree.session_id = sessionId;
    assert.equal(isConversationSnapshot(invalidIdentity, Object.keys(samplingState)), false);
  }
  const wrongModel = structuredClone(pythonSnapshot);
  wrongModel.tree.model_id = "different/model";
  assert.equal(isConversationSnapshot(wrongModel, Object.keys(samplingState)), false);
  const plan = buildConversationImportPlan(snapshot);
  assert.equal(
    plan.steeringExpression,
    "0.5 !local/calm + 0.3 !jlens/fixture@response + 0.4 sae/17 + 0.4,0.7 local/emotions:sae-release-a%happy@response",
  );
  assert.deepEqual(plan.probeRequests, [
    {
      selector: "local/emotions:sae-release-a",
      name: "mood alias",
      top_n: 4,
    },
    { selector: "sae/17", name: "latent alias" },
  ]);
  assert.equal(plan.steerEntries.get("local/emotions").variant, "sae-release-a");
  assert.equal(plan.steerEntries.get("jlens/fixture").ablate, true);

  const legacyAblationRows = structuredClone(snapshot);
  delete legacyAblationRows.steerRack[0].ablate;
  delete legacyAblationRows.steerRack[1].ablate;
  delete legacyAblationRows.steerRack[2].ablate;
  const legacyPlan = buildConversationImportPlan(legacyAblationRows);
  assert.equal(legacyPlan.steerEntries.get("local/calm").ablate, false);
  assert.equal(legacyPlan.steerEntries.get("jlens/fixture").ablate, false);
  assert.equal(legacyPlan.steerEntries.get("sae/17").ablate, false);

  const oldVersion = structuredClone(snapshot);
  oldVersion.version = 6;
  assert.equal(isConversationSnapshot(oldVersion, Object.keys(samplingState)), false);
  const lostAlias = structuredClone(snapshot);
  delete lostAlias.probeRack.entries[0].request.name;
  assert.equal(isConversationSnapshot(lostAlias, Object.keys(samplingState)), false);
  const lostCurveVariant = structuredClone(snapshot);
  delete lostCurveVariant.steerRack[3].variant;
  assert.equal(isConversationSnapshot(lostCurveVariant, Object.keys(samplingState)), false);

  const preflightOrder = [];
  await assert.rejects(
    runConversationImportTransaction({
      capture() {
        preflightOrder.push("capture");
        return "old";
      },
      async preflight() {
        preflightOrder.push("preflight");
        throw new Error("missing instrument");
      },
      async apply() { preflightOrder.push("apply"); },
      async rollback() { preflightOrder.push("rollback"); },
    }),
    /missing instrument/,
  );
  assert.deepEqual(preflightOrder, ["capture", "preflight"]);

  let workspace = "original";
  const rollbackOrder = [];
  await assert.rejects(
    runConversationImportTransaction({
      capture() {
        rollbackOrder.push("capture");
        return workspace;
      },
      async preflight() { rollbackOrder.push("preflight"); },
      async apply() {
        rollbackOrder.push("apply");
        workspace = "partial import";
        throw new Error("attach failed");
      },
      async rollback(previous) {
        rollbackOrder.push("rollback");
        workspace = previous;
      },
    }),
    /attach failed/,
  );
  assert.equal(workspace, "original");
  assert.deepEqual(rollbackOrder, ["capture", "preflight", "apply", "rollback"]);

  process.stdout.write("conversation snapshot tests passed\n");
} finally {
  await server.close();
}
