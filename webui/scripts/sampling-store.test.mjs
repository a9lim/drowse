import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));

async function samplingFor(
  mode,
  returnTopK,
  {
    activeProbe = false,
    probeSubspaceTrails,
    defaultUserRole = "user",
    defaultAssistantRole = "assistant",
    assistantRoleOverride,
    runtimeSignals,
    sessionMaxTokens = 64,
  } = {},
) {
  const server = await createServer({
    root,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true, watch: null },
  });
  try {
    const {
      installRuntimeCapabilities,
      installRuntimeClient,
    } = await server.ssrLoadModule(
      "/src/lib/runtime/registry.ts",
    );
    installRuntimeClient({ mode });
    if (probeSubspaceTrails !== undefined || runtimeSignals !== undefined) {
      installRuntimeCapabilities({
        ...(runtimeSignals === undefined ? {} : { signals: runtimeSignals }),
        operations: {
          probe_subspace_trails: {
            available: probeSubspaceTrails,
            reasons: probeSubspaceTrails
              ? []
              : [{
                  code: "HOSTED_SUBSPACE_COORDINATES_UNAVAILABLE",
                  message: "Exact browser subspace coordinates are unavailable",
                  severity: "hard",
                }],
          },
        },
      });
    }
    const { sessionState } = await server.ssrLoadModule(
      "/src/lib/stores/session.svelte.ts",
    );
    const {
      buildSamplingPayload,
      hydrateSamplingFromInfo,
      samplingState,
    } = await server.ssrLoadModule(
      "/src/lib/stores/sampling.svelte.ts",
    );
    const { probeRack } = await server.ssrLoadModule(
      "/src/lib/stores/probes.svelte.ts",
    );
    sessionState.info = {
      model_id: `fixture/${mode}`,
      default_user_role: defaultUserRole,
      default_assistant_role: defaultAssistantRole,
      config: {
        max_tokens: sessionMaxTokens,
        temperature: 1,
        top_p: 1,
        top_k: null,
        system_prompt: null,
        thinking: false,
      },
    };
    probeRack.active = activeProbe ? ["local/demo"] : [];
    samplingState.return_top_k = returnTopK;
    hydrateSamplingFromInfo();
    if (assistantRoleOverride !== undefined) {
      samplingState.assistant_role = assistantRoleOverride;
    }
    return {
      retained: samplingState.return_top_k,
      userRole: samplingState.user_role,
      assistantRole: samplingState.assistant_role,
      maxTokens: samplingState.max_tokens,
      payload: buildSamplingPayload(),
    };
  } finally {
    await server.close();
  }
}

async function refreshedGemmaRoles() {
  const server = await createServer({
    root,
    appType: "custom",
    logLevel: "silent",
    server: { middlewareMode: true, watch: null },
  });
  try {
    const { installRuntimeClient } = await server.ssrLoadModule(
      "/src/lib/runtime/registry.ts",
    );
    installRuntimeClient({ mode: "browser" });
    const { sessionState } = await server.ssrLoadModule(
      "/src/lib/stores/session.svelte.ts",
    );
    const { hydrateSamplingFromInfo, samplingState } = await server.ssrLoadModule(
      "/src/lib/stores/sampling.svelte.ts",
    );
    const session = {
      model_id: "gemma3-270m-instruct",
      default_user_role: "user",
      default_assistant_role: "assistant",
      config: {
        max_tokens: 64,
        temperature: 1,
        top_p: 1,
        top_k: null,
        system_prompt: null,
        thinking: false,
      },
    };
    sessionState.info = session;
    hydrateSamplingFromInfo();
    sessionState.info = { ...session, default_assistant_role: "model" };
    hydrateSamplingFromInfo();
    return samplingState.assistant_role;
  } finally {
    await server.close();
  }
}

const browser = await samplingFor("browser", 8);
assert.equal(browser.retained, 5);
assert.equal(browser.payload.return_top_k, 5);

const http = await samplingFor("http", 8);
assert.equal(http.retained, 8);
assert.equal(http.payload.return_top_k, 8);

const preview = await samplingFor("fake", 8);
assert.equal(preview.retained, 0);
assert.equal("return_top_k" in preview.payload, false);

const appleMobile = await samplingFor("browser", 8, {
  runtimeSignals: {
    runtimeClass: "apple-mobile-webkit",
    appleMobile: true,
  },
  sessionMaxTokens: 8_192,
});
assert.equal(appleMobile.maxTokens, 256);
assert.equal(appleMobile.payload.max_tokens, 256);

const desktop = await samplingFor("browser", 8, {
  runtimeSignals: {
    runtimeClass: "desktop-chromium",
    appleMobile: false,
  },
  sessionMaxTokens: 8_192,
});
assert.equal(desktop.maxTokens, 8_192);
assert.equal(desktop.payload.max_tokens, 8_192);

const hostedProbe = await samplingFor("browser", 8, {
  activeProbe: true,
  probeSubspaceTrails: false,
});
assert.equal(hostedProbe.payload.persist_per_layer_scores, true);
assert.equal("persist_subspace_coords" in hostedProbe.payload, false);

const hostedProbeWithExactTrails = await samplingFor("browser", 8, {
  activeProbe: true,
  probeSubspaceTrails: true,
});
assert.equal(hostedProbeWithExactTrails.payload.persist_subspace_coords, true);

const httpProbe = await samplingFor("http", 8, { activeProbe: true });
assert.equal(httpProbe.payload.persist_subspace_coords, true);

const gemma = await samplingFor("browser", 8, {
  defaultAssistantRole: "model",
});
assert.equal(gemma.userRole, "user");
assert.equal(gemma.assistantRole, "model");
assert.equal("user_role" in gemma.payload, false);
assert.equal("assistant_role" in gemma.payload, false);

const gemmaAssistant = await samplingFor("browser", 8, {
  defaultAssistantRole: "model",
  assistantRoleOverride: "assistant",
});
assert.equal(gemmaAssistant.assistantRole, "assistant");
assert.equal(gemmaAssistant.payload.assistant_role, "assistant");

assert.equal(await refreshedGemmaRoles(), "model");

async function queuedSamplingAndReset() {
  const server = await createServer({ root, appType: "custom", logLevel: "silent", server: { middlewareMode: true, watch: null } });
  try {
    const registry = await server.ssrLoadModule("/src/lib/runtime/registry.ts");
    const original = {
      model_id: "fixture/reset", default_user_role: "user", default_assistant_role: "model",
      default_steering: null, is_base_model: false, thinking_is_optional: true,
      instruments: [
        { family: "geometry", source: null, live: { enabled: true }, capabilities: {} },
        { family: "lens", source: "original-lens", live: { enabled: true, layers: [0, 2] }, capabilities: { source_switch: true } },
      ],
      config: { temperature: 0.9, top_p: 0.95, top_k: null, max_tokens: 1024, system_prompt: "", thinking: false },
    };
    let current = structuredClone(original);
    const calls = [];
    let release;
    let failNext = false;
    const detached = [];
    registry.installRuntimeClient({ mode: "browser", probes: { detach: async name => { detached.push(name); } }, instruments: {
      setLive: async (family, live) => {
        current.instruments.find(row => row.family === family).live = live;
        return live;
      },
      setLensSource: async source => {
        current.instruments.find(row => row.family === "lens").source = source;
        return { source, live_layers: [0, 2] };
      },
    }, sessions: {
      get: async () => structuredClone(current),
      patch: async patch => {
        calls.push(patch);
        if (calls.length === 1) await new Promise(resolve => { release = resolve; });
        if (failNext) { failNext = false; throw new Error("fixture persistence failure"); }
        Object.assign(current.config, patch);
        return structuredClone(current);
      },
    } });
    registry.installHostedController({ snapshot: { modelDefaults: original } });
    const { sessionState } = await server.ssrLoadModule("/src/lib/stores/session.svelte.ts");
    const sampling = await server.ssrLoadModule("/src/lib/stores/sampling.svelte.ts");
    const { resetSettings } = await server.ssrLoadModule("/src/lib/stores/settingsReset.svelte.ts");
    const { steerRack } = await server.ssrLoadModule("/src/lib/stores/steering.svelte.ts");
    const { probeRack } = await server.ssrLoadModule("/src/lib/stores/probes.svelte.ts");
    const { genStatus } = await server.ssrLoadModule("/src/lib/stores/chat.svelte.ts");
    sessionState.info = current;
    sampling.hydrateSamplingFromInfo();
    const first = sampling.patchSessionDefaults({ temperature: 0.2 });
    const second = sampling.patchSessionDefaults({ temperature: 0.6, top_p: 0.7 });
    const last = sampling.patchSessionDefaults({ temperature: 1.4 });
    assert.equal(sampling.samplingState.temperature, 1.4);
    release();
    await Promise.all([first, second, last]);
    assert.equal(calls.length, 2, "drag updates coalesce while a request is pending");
    assert.equal(sampling.samplingState.temperature, 1.4);
    assert.equal(current.config.top_p, 0.7);
    assert.equal(sampling.modelDefaultsState.info.config.temperature, 0.9);
    Object.assign(sampling.samplingState, { seed: 42, stop_sequences: "STOP", logit_bias_text: "123: 2", presence_penalty: 1, frequency_penalty: 1, assistant_role: "pirate" });
    steerRack.customExpression = "0.3 jlens/orange";
    await resetSettings();
    assert.deepEqual(current.config, original.config);
    assert.equal(sampling.samplingState.seed, null);
    assert.equal(sampling.samplingState.stop_sequences, "");
    assert.equal(sampling.samplingState.logit_bias_text, "");
    assert.equal(sampling.samplingState.presence_penalty, 0);
    assert.equal(sampling.samplingState.frequency_penalty, 0);
    assert.equal(sampling.samplingState.assistant_role, "pirate", "generation-only reset preserves role changes");
    assert.equal(steerRack.customExpression, "0.3 jlens/orange");
    probeRack.active = ["jlens/orange"];
    current.instruments[0].live.enabled = false;
    current.instruments[1].source = "changed-lens";
    current.instruments[1].live = { enabled: false, layers: [2] };
    sessionState.info = structuredClone(current);
    await resetSettings(true);
    assert.deepEqual(detached, ["jlens/orange"]);
    assert.deepEqual(probeRack.active, []);
    assert.deepEqual(current.instruments, original.instruments);
    assert.equal(steerRack.customExpression, null);
    assert.equal(sampling.samplingState.assistant_role, "model");
    genStatus.active = true;
    const before = calls.length;
    await resetSettings(true);
    assert.equal(calls.length, before);
    genStatus.active = false;
    failNext = true;
    await assert.rejects(resetSettings(), /fixture persistence failure/);
    await resetSettings();
    assert.deepEqual(current.config, original.config);
  } finally { await server.close(); }
}

await queuedSamplingAndReset();
console.log("Sampling store runtime-capability, queued edits, and reset checks passed");
