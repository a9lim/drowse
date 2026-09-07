// THE per-token transcript-highlight implementation.
//
// Every surface that tints tokens with the selected highlight probe —
// the chat transcript, the raw-buffer mirror, the drilldown's context
// ribbon — renders through here, so they agree on the score lookup, the
// per-probe scale, the hue family, and compare-two.  ``lib/tokens.ts``
// owns the pure ramp math (no store dependency); this module is the thin
// layer that reads ``highlightState`` and applies it to a token.

import { highlightState, highlightScale, probeRack } from "./stores.svelte";
import {
  ENTROPY_TARGET,
  PROBABILITY_TARGET,
  SURPRISE_TARGET,
  entropyScore,
  highlightHue,
  parseProbeTarget,
  probabilityScore,
  probeScoreForTarget,
  surpriseScore,
  tokenBackgroundStyle,
} from "./tokens";
import type { TokenScore } from "./types";

/** The deepest recorded layer, for the explicitly layer-labeled tooltip. */
export function latestLayerScores(
  t: TokenScore,
): Record<string, number> | undefined {
  const pls = t.perLayerScores;
  if (!pls) return undefined;
  const layers = Object.keys(pls).sort((a, b) => Number(a) - Number(b));
  const last = layers[layers.length - 1];
  return last === undefined ? undefined : pls[last];
}

/** Score lookup for one token against one highlight target.
 *
 *  ``SURPRISE_TARGET`` routes to the logit-pass surprise value.  A real
 *  probe target reads the live per-axis coords first (the full rank-R
 *  reading), then the collapsed axis-0 ``probes`` row (what ``done`` and a
 *  tree reload restore). A layer-specific or unnamed legacy score cannot
 *  stand in for the selected probe's aggregate. Returns
 *  ``undefined`` when nothing carries the target — the caller renders
 *  transparent. */
export function highlightScoreFor(
  t: TokenScore,
  target: string | null,
): number | undefined {
  if (!target) return undefined;
  if (target === SURPRISE_TARGET) return surpriseScore(t.logprob);
  if (target === PROBABILITY_TARGET) return probabilityScore(t.logprob);
  if (target === ENTROPY_TARGET) return entropyScore(t.samplerEntropy);
  return probeScoreForTarget(t, target);
}

/** Background style for one token under the current highlight selection.
 *
 *  Single-probe mode paints a ``background-color``; compare-two paints the
 *  two-stripe (or smooth-blend) gradient, each half on its own scale and
 *  hue — ``SURPRISE_TARGET`` works in either slot and reads in the
 *  chat accent regardless of what the other slot holds. Falls back
 *  to single-probe rendering when compare-two is on but no second target
 *  is picked. */
export function highlightStyleFor(
  t: TokenScore,
): { backgroundColor?: string; backgroundImage?: string } {
  const a = highlightState.target;
  if (!a) return {};
  const compare =
    highlightState.compareTwo && highlightState.compareTarget
      ? highlightState.compareTarget
      : null;
  return tokenBackgroundStyle(
    highlightScoreFor(t, a),
    compare === null ? null : highlightScoreFor(t, compare),
    highlightState.smoothBlend,
    highlightScale(a),
    compare === null ? undefined : highlightScale(compare),
    probeHue(a),
    compare === null ? undefined : probeHue(compare),
  );
}

function probeHue(target: string) {
  const info = probeRack.entries.get(parseProbeTarget(target).base)?.info;
  if (info?.family === "lens") return "surprise";
  if (info?.family === "sae") return "sae";
  return highlightHue(target);
}

/** ``highlightStyleFor`` flattened to an inline ``style`` attribute. */
export function highlightStyleString(t: TokenScore): string {
  const style = highlightStyleFor(t);
  const parts: string[] = [];
  if (style.backgroundColor) {
    parts.push(`background-color: ${style.backgroundColor}`);
  }
  if (style.backgroundImage) {
    parts.push(`background-image: ${style.backgroundImage}`);
  }
  return parts.join(";");
}
