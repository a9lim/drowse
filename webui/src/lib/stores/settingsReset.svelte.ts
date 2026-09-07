import { apiInstruments } from "../runtime/services";
import { getRuntimeClient } from "../runtime/registry";
import { clampTokenAlternativeCount } from "../runtime/samplingCapabilities";
import { genStatus, setGenUiMode } from "./chat.svelte";
import { modelDefaultsState, patchSessionDefaults, samplingState } from "./sampling.svelte";
import { refreshSession, sessionState } from "./session.svelte";
import { steerRack } from "./steering.svelte";
import { detachProbe, probeRack } from "./probes.svelte";
import { autoRegenState } from "./ab.svelte";

export const settingsResetState = $state({ busy: false });

export async function resetSettings(full = false): Promise<void> {
  const original = modelDefaultsState.info;
  if (settingsResetState.busy || genStatus.active) return;
  if (!original || original.model_id !== sessionState.info?.model_id) {
    throw new Error("Open a model before resetting its settings.");
  }
  settingsResetState.busy = true;
  try {
    const config = original.config;
    await patchSessionDefaults({
      temperature: config.temperature ?? 1,
      top_p: config.top_p ?? 1,
      top_k: config.top_k,
      max_tokens: config.max_tokens ?? 1024,
      system_prompt: config.system_prompt ?? "",
      ...(original.thinking_is_optional ? { thinking: config.thinking ?? false } : {}),
    });
    Object.assign(samplingState, {
      seed: null,
      stop_sequences: "",
      logit_bias_text: "",
      presence_penalty: 0,
      frequency_penalty: 0,
      return_top_k: clampTokenAlternativeCount(8, getRuntimeClient().mode),
    });
    if (!full) return;
    steerRack.entries.clear();
    steerRack.customExpression = original.default_steering;
    steerRack.subspaceAlong = 0.5;
    samplingState.user_role = original.default_user_role ?? "user";
    samplingState.assistant_role = original.default_assistant_role ?? "assistant";
    Object.assign(autoRegenState, { enabled: false, mode: "unsteered", custom: "" });
    setGenUiMode(original.is_base_model ? "raw" : "chat");
    for (const name of [...probeRack.active]) await detachProbe(name);
    for (const instrument of original.instruments) {
      if (instrument.family !== "geometry" && instrument.family !== "lens" && instrument.family !== "sae") continue;
      const current = sessionState.info?.instruments.find(row => row.family === instrument.family);
      if (!current) continue;
      if (instrument.source && instrument.source !== current.source && instrument.family !== "geometry") {
        if (instrument.family === "lens" && current.capabilities.source_switch) {
          await apiInstruments.setLensSource(instrument.source);
        } else if (getRuntimeClient().mode === "browser") {
          await apiInstruments.activateInstalledPack(instrument.family, {
            source: instrument.source,
            ...("layer" in instrument.live ? { layer: instrument.live.layer } : {}),
          });
        }
      }
      if (JSON.stringify(current.live) !== JSON.stringify(instrument.live)) {
        await apiInstruments.setLive(instrument.family, {
          enabled: instrument.live.enabled,
          ...("layers" in instrument.live ? { layers: instrument.live.layers } : {}),
        });
      }
    }
    await refreshSession();
    if (sessionState.error) throw new Error(sessionState.error);
  } finally {
    settingsResetState.busy = false;
  }
}
