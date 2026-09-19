import { attachProbe, chatLog, openDrawer } from "../../src/lib/stores.svelte";
import { getRuntimeClient } from "../../src/lib/runtime/registry";
import { drilldownUi } from "../../src/drawers/token/drilldown.svelte";
import type { InstrumentFamily, MeasurementsEnvelopeJSON } from "../../src/lib/types";

const pending: Array<{
  family: InstrumentFamily;
  rawIndex: number;
  steered: boolean;
  resolve: (value: { measurements: MeasurementsEnvelopeJSON }) => void;
  reject: (error: Error) => void;
}> = [];
let calls = 0;

function measurement(family: InstrumentFamily, count: number, steered = true): MeasurementsEnvelopeJSON {
  const binding = { source: "badge-fixture", steering: steered ? "0.1 fixture/calm.focused" : null };
  const envelope: MeasurementsEnvelopeJSON = {
    version: 1, scope: "replay", provenance: "replayed", instruments: {},
  };
  if (family === "lens") {
    envelope.instruments.lens = { binding, readout: {
      layers: Array.from({ length: count }, (_, layer) => ({
        layer, tokens: [{ id: 7, token: " fixture", logprob: -0.5 }],
      })),
      aggregate: [],
    } };
  } else if (family === "sae") {
    envelope.instruments.sae = { binding: { ...binding, layer: 1 }, readout: {
      features: Array.from({ length: count }, (_, id) => ({
        id, activation: 0.5, max_act: 1, label: `Feature ${id}`,
      })),
    } };
  } else {
    envelope.instruments.geometry = { binding, readings: Object.fromEntries(
      Array.from({ length: count }, (_, index) => [`probe${index}`, {
        fraction: 0.5, nearest: [], coords: [0.2], residual: 0.1,
        fraction_per_layer: { "0": 0.5 }, coords_per_layer: { "0": [0.2] },
        residual_per_layer: { "0": 0.1 },
      }]),
    ) };
  }
  return envelope;
}

export async function install() {
  await attachProbe("fixture/calm.focused");
  const captured: MeasurementsEnvelopeJSON = {
    version: 1, scope: "token", provenance: "captured", instruments: {
      ...measurement("lens", 2).instruments,
      ...measurement("sae", 3).instruments,
      ...measurement("geometry", 1).instruments,
    },
  };
  getRuntimeClient().instruments.tokenReadout = (family, _nodeId, rawIndex, options) => {
    calls++;
    return new Promise((resolve, reject) => pending.push({
      family, rawIndex, steered: options?.steered ?? true, resolve, reject,
    }));
  };
  chatLog.turns = [{
    role: "assistant", nodeId: "badge-node", text: "one two three four", generated: true,
    tokens: [
      { text: "one", tokenId: 1, rawIndex: 0, topAlts: [{ id: 1, text: "one", logprob: -0.1 }] },
      { text: " two", tokenId: 2, rawIndex: 1, measurements: captured, topAlts: [] },
      { text: " three", tokenId: 3, rawIndex: 2 },
      { text: " four", tokenId: 4 },
    ],
  }];
  drilldownUi.tab = "lens";
  openDrawer("token_drilldown", { turnIdx: 0, tokenIdx: 0 });
}

export function status() {
  return { calls, pending: pending.map(({ family, rawIndex, steered }) => ({ family, rawIndex, steered })) };
}

export function complete(count: number, fail = false) {
  const request = pending.shift()!;
  if (fail) request.reject(new Error("Fixture replay failed"));
  else request.resolve({ measurements: measurement(request.family, count, request.steered) });
}

export function capturedText() {
  return JSON.stringify(chatLog.turns[0].tokens![1].measurements);
}
