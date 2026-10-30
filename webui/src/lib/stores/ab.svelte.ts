// The A/B comparison: an auto-regen of the just-finished turn, shown in
// the chat's right column.
//
// ``autoRegenState`` is the whole control surface (there is no standalone
// A/B toggle). Every mode runs here as a stateless *shadow* generation:
// ``abState.processingAb`` routes the next
// ``started``/``token``/``done`` stream into
// ``chatLog.turns[pendingTurnIdx].abPair`` instead of appending a fresh
// top-level turn.
//
// The shadow's prompt is reconstructed from ``chatLog.turns`` at fire
// time, so the comparison works for any turn, not only the just-sent one.

import type { WSClientMessage } from "../types";
import { chatLog, genStatus } from "./chat.svelte";
import { buildSamplingPayload, samplingState } from "./sampling.svelte";
import { userFacingError } from "../runtime/userFacingError";
import { ensureRuntimeChannel } from "./ws.svelte";

/** Transient routing state for the automatic shadow generation.
 *
 *  ``processingAb`` / ``pendingTurnIdx`` are
 *  load-bearing for the WS dispatcher — while ``processingAb`` is set the
 *  next ``started``/``token``/``done`` stream routes into
 *  ``chatLog.turns[pendingTurnIdx].abPair`` instead of appending a fresh
 *  top-level turn.  ``pendingTurnIdx`` is set when the shadow gen is
 *  dispatched and cleared on its ``done`` / ``error``.
 *
 *  The shadow's prompt is reconstructed from ``chatLog.turns`` at fire
 *  time (see ``_buildShadowMessages``), so the comparison works for any
 *  turn, not only the just-sent one.  Turning auto-regen off mid-flight
 *  lets the in-flight shadow finish writing into ``abPair`` — the turn is
 *  harmless when not rendered, and tearing the WS state down mid-stream
 *  is more error-prone than letting it complete; it only prevents the
 *  *next* steered gen from spawning a shadow. */
export interface AbState {
  pendingTurnIdx: number | null;
  processingAb: boolean;
  pendingRole: "user" | "assistant" | null;
  pendingRoleLabel: string | null;
}

export const abState: AbState = $state({
  pendingTurnIdx: null,
  processingAb: false,
  pendingRole: null,
  pendingRoleLabel: null,
});

/** Build the conversation as a messages list to replay through the
 * unsteered shadow.  Walks ``chatLog.turns[0..steeredIdx-1]`` (excluding
 * ``steeredIdx`` itself, which is the generated response we
 * don't want the shadow to inherit), filtering out system / error turns
 * that aren't real conversation context.
 *
 * The unsteered model sees prior steered turns as if they
 * happened naturally — that's the user's "play the conversation back"
 * contract. Scene rendering permits arbitrary seat sequences, so the target
 * turn itself—not the final history role—selects the prompt that follows. */
function _buildShadowMessages(
  steeredIdx: number,
): Array<{
  role: "user" | "assistant";
  content: string;
  label?: string | null;
}> {
  const out: Array<{
    role: "user" | "assistant";
    content: string;
    label?: string | null;
  }> = [];
  for (let i = 0; i < steeredIdx; i++) {
    const t = chatLog.turns[i];
    if (!t) continue;
    if (t.role !== "user" && t.role !== "assistant") continue; // skip system / errors
    // Use the accumulated text — generated turns already exclude their
    // thinking content (only response tokens land in ``turn.text``), so
    // replaying them through ``enable_thinking=False`` is well-formed.
    out.push({ role: t.role, content: t.text ?? "", label: t.roleLabel });
  }
  return out;
}

/** Internal: dispatch the unsteered shadow generate that pairs with the
 * just-finished steered turn at index ``steeredIdx``.  Sends the full
 * conversation as a ``messages`` list instead of a bare input string +
 * server-side history — the shadow runs ``stateless: true`` so the
 * server doesn't append to history (the steered branch already did) and
 * the messages list is the *only* context the unsteered model sees.
 * That makes the comparison work for any turn, not just the first. */
export async function sendShadowGenerate(
  steeredIdx: number,
  recipeOverride = "unsteered",
): Promise<void> {
  const target = chatLog.turns[steeredIdx];
  if (!target?.generated || target.role === "system") return;
  const messages = _buildShadowMessages(steeredIdx);
  // Reserve the one generation slot before awaiting the channel. Without the
  // reservation, a fast composer send can slip into the connection-open gap
  // and make both requests fail each other's busy guard.
  abState.pendingTurnIdx = steeredIdx;
  abState.processingAb = true;
  abState.pendingRole = target.role;
  abState.pendingRoleLabel = target.roleLabel ?? null;
  target.abPair = undefined;
  genStatus.active = true;
  try {
    const channel = await ensureRuntimeChannel();
    // Shadow path mirrors ``sendGenerate``'s sampling-payload build so the
    // ``return_top_k`` opt-in rides shadow / auto-regen runs too (matches
    // the steered turn's wire-shape, keeps logit captures comparable across
    // siblings).
    const sampling = buildSamplingPayload() ?? {};
    if (target.roleLabel) {
      if (target.role === "user") sampling.user_role = target.roleLabel;
      else sampling.assistant_role = target.roleLabel;
    }
    const payload: WSClientMessage = {
      type: "generate",
      // ``input`` accepts ``Any`` server-side; a list goes straight through
      // to ``session._prepare_input`` which dispatches on isinstance(list).
      input: messages,
      // Every comparison is stateless. The recipe modifier selects the
      // alternate behavior without adding a throwaway sibling to the loom.
      steering: "",
      sampling,
      thinking: samplingState.thinking ?? false,
      stateless: true,
      raw: false,
      generate_seat: target.role,
      recipe_override: recipeOverride,
      // The parent is used only to resolve the original turn's recipe.
      parent_node_id: target.nodeId ?? undefined,
    };
    channel.send(payload);
  } catch (error) {
    genStatus.active = false;
    abState.pendingTurnIdx = null;
    abState.processingAb = false;
    abState.pendingRole = null;
    abState.pendingRoleLabel = null;
    target.abPair = {
      role: "system",
      text: `Comparison unavailable: ${userFacingError(error, "Try again after the model is ready.")}`,
    };
  }
}

let scheduledComparison: ReturnType<typeof setTimeout> | null = null;

export function comparisonPending(): boolean {
  return scheduledComparison !== null || abState.processingAb;
}

/** Start a comparison after the just-finished generation has completely
 * released the runtime. Worker and fixture transports acknowledge a request
 * just after emitting ``done``; sending synchronously from that event races
 * their busy guard and used to leave the UI stuck at ``pending``. */
export function scheduleAutoComparison(steeredIdx: number): boolean {
  const override = currentRecipeOverride();
  if (override === null) return false;
  if (scheduledComparison !== null) clearTimeout(scheduledComparison);
  scheduledComparison = setTimeout(() => {
    scheduledComparison = null;
    if (!autoRegenState.enabled || genStatus.active || abState.processingAb) return;
    void sendShadowGenerate(steeredIdx, override);
  }, 0);
  return true;
}

// --------------------------------- auto-regen recipe-override -------

/** Built-in auto-regen modes from the engine. */
export type AutoRegenMode =
  | "unsteered"
  | "inverted"
  | "reseed"
  | "cool"
  | "hot"
  | "custom";

export interface AutoRegenState {
  /** Master toggle (replaces the old A/B toggle one-for-one).  Default
   *  off — the previous A/B behaviour resumed by toggling on with mode
   *  ``"unsteered"``. */
  enabled: boolean;
  mode: AutoRegenMode;
  /** Custom-mode body — a partial-recipe expression (e.g. ``"seed=42,
   *  temperature=1.5"``).  Ignored when ``mode != "custom"``. */
  custom: string;
}

export const autoRegenState: AutoRegenState = $state({
  enabled: false,
  mode: "unsteered",
  custom: "",
});

export function toggleAutoRegen(): void {
  const wasOff = !autoRegenState.enabled;
  autoRegenState.enabled = !autoRegenState.enabled;
  // Off → on: replay the conversation for the most recent generated turn
  // with the selected recipe when it does not already carry an ``abPair``,
  // so users who flip the toggle on
  // after-the-fact see the right column populate immediately rather
  // than waiting for the next send.
  if (!wasOff) {
    if (scheduledComparison !== null) {
      clearTimeout(scheduledComparison);
      scheduledComparison = null;
    }
    return;
  }
  if (genStatus.active) return; // ``done`` handler will fire its own
  for (let i = chatLog.turns.length - 1; i >= 0; i--) {
    const t = chatLog.turns[i];
    if (!t) continue;
    if (!t.generated || t.role === "system") continue;
    if (t.abPair) break;
    scheduleAutoComparison(i);
    break;
  }
}

export function disableAutoRegen(): void {
  autoRegenState.enabled = false;
  if (scheduledComparison !== null) {
    clearTimeout(scheduledComparison);
    scheduledComparison = null;
  }
}

export function setAutoRegenMode(mode: AutoRegenMode): void {
  autoRegenState.mode = mode;
  if (!autoRegenState.enabled || genStatus.active || abState.processingAb) return;
  for (let i = chatLog.turns.length - 1; i >= 0; i--) {
    const turn = chatLog.turns[i];
    if (turn?.generated && turn.role !== "system") {
      scheduleAutoComparison(i);
      return;
    }
  }
}

export function setAutoRegenCustom(text: string): void {
  autoRegenState.custom = text;
}

/** Render the configured recipe-override the engine consumes.  Returns
 *  ``null`` when auto-regen is off — callers shouldn't dispatch a
 *  shadow regen in that case. */
export function currentRecipeOverride(): string | null {
  if (!autoRegenState.enabled) return null;
  if (autoRegenState.mode === "custom") {
    const v = autoRegenState.custom.trim();
    return v || null;
  }
  return autoRegenState.mode;
}
