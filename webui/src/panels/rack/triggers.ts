// Trigger-pill vocabulary shared by every steer card (subspace / manifold
// / j-lens).  One cycle order, one display word, one tooltip per preset —
// extracted from SteerCard so the J-lens steer card's pill stays in
// lock-step with the concept cards'.

import type { Trigger } from "../../lib/types";

export const TRIGGER_ORDER: Trigger[] = [
  "BOTH",
  "BEFORE",
  "AFTER",
  "THINKING",
  "RESPONSE",
];

export const TRIGGER_WORD: Record<Trigger, string> = {
  BOTH: "both",
  BEFORE: "before",
  AFTER: "after",
  THINKING: "thinking",
  RESPONSE: "response",
};

// Phases follow the engine presets (steering/triggers.py): ``before`` is
// PROMPT_ONLY (the prompt prefill, no generated token), ``response`` is
// GENERATED_ONLY (every generated token, thinking included), ``after`` is
// AFTER_THINKING (the reply once thinking has closed).
export const TRIGGER_LABEL: Record<Trigger, string> = {
  BOTH: "both: steer the prompt and every generated token (default)",
  BEFORE: "before: steer the prompt prefill only; generated tokens are unsteered",
  AFTER: "after: steer the reply after thinking ends, not the prompt or thinking",
  THINKING: "thinking: steer the thinking tokens only",
  RESPONSE: "response: steer every generated token, thinking included, not the prompt",
};

/** The next trigger in the cycle order — the pill's click step. */
export function nextTrigger(current: Trigger): Trigger {
  const idx = TRIGGER_ORDER.indexOf(current);
  return TRIGGER_ORDER[(idx + 1) % TRIGGER_ORDER.length];
}
