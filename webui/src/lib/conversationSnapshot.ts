import { serializeExpression } from "./expression";
import type {
  LoomTreeJSON,
  ProbeRequest,
  ProbeSortMode,
  SteerEntry,
  Trigger,
  Variant,
} from "./types";
import type { SamplingState } from "./stores/sampling.svelte";

export const CONVERSATION_SNAPSHOT_VERSION = 7 as const;
export const CONVERSATION_SAMPLING_KEYS = [
  "temperature",
  "top_p",
  "top_k",
  "max_tokens",
  "seed",
  "system_prompt",
  "stop_sequences",
  "logit_bias_text",
  "presence_penalty",
  "frequency_penalty",
  "thinking",
  "return_top_k",
  "user_role",
  "assistant_role",
] as const satisfies readonly (keyof SamplingState)[];

export type ConversationSteerRow = { name: string } & SteerEntry;

export interface ConversationProbeRow {
  name: string;
  request: ProbeRequest;
  sparkline: number[];
  current: number;
  previous: number;
}

export interface ConversationSnapshotV7 {
  version: typeof CONVERSATION_SNAPSHOT_VERSION;
  savedAt: string;
  model_id: string;
  session_id: string;
  tree: LoomTreeJSON;
  steerRack: ConversationSteerRow[];
  subspaceAlong: number;
  customSteeringExpression: string | null;
  probeRack: {
    sortMode: ProbeSortMode;
    active: string[];
    entries: ConversationProbeRow[];
  };
  highlightState: {
    target: string | null;
    compareTarget: string | null;
    compareTwo: boolean;
    smoothBlend: boolean;
  };
  samplingState: SamplingState;
}

export interface ConversationImportPlan {
  snapshot: ConversationSnapshotV7;
  steerEntries: Map<string, SteerEntry>;
  steeringExpression: string;
  probeRequests: ProbeRequest[];
  probeRows: Map<string, ConversationProbeRow>;
}

const TRIGGERS = new Set<Trigger>([
  "BOTH", "BEFORE", "AFTER", "THINKING", "RESPONSE", "PROMPT", "GENERATED",
]);
const VARIANT = /^(?:raw|sae(?:-.+)?|role(?:-.+)?|from(?:-.+)?)$/u;

export function isConversationSnapshot(
  value: unknown,
  samplingKeys: readonly string[],
): value is ConversationSnapshotV7 {
  try {
    validateConversationSnapshot(value, samplingKeys);
    return true;
  } catch {
    return false;
  }
}

export function validateConversationSnapshot(
  value: unknown,
  samplingKeys: readonly string[],
): asserts value is ConversationSnapshotV7 {
  const snapshot = record(value, "conversation snapshot");
  exactKeys(snapshot, [
    "customSteeringExpression",
    "highlightState",
    "model_id",
    "probeRack",
    "samplingState",
    "savedAt",
    "session_id",
    "steerRack",
    "subspaceAlong",
    "tree",
    "version",
  ], "conversation snapshot");
  if (snapshot.version !== CONVERSATION_SNAPSHOT_VERSION) invalid("snapshot version");
  nonempty(snapshot.savedAt, "saved timestamp");
  nonempty(snapshot.model_id, "model id");
  nonempty(snapshot.session_id, "session id");
  const tree = record(snapshot.tree, "conversation tree");
  finiteInteger(tree.tree_format, "tree format");
  nonempty(tree.drowse_version, "tree Drowse version");
  nonempty(tree.root_id, "tree root id");
  nonempty(tree.active_node_id, "tree active node id");
  finiteInteger(tree.rev, "tree revision");
  if (
    tree.model_id !== snapshot.model_id ||
    (tree.session_id !== null && tree.session_id !== snapshot.session_id)
  ) {
    invalid("tree identity");
  }
  if (!Array.isArray(tree.nodes)) invalid("tree nodes");
  record(tree.children_of, "tree children map");
  record(tree.cast, "tree cast");

  finite(snapshot.subspaceAlong, "subspace strength");
  if (
    typeof snapshot.customSteeringExpression !== "string" &&
    snapshot.customSteeringExpression !== null
  ) invalid("custom steering expression");

  const steerRows = array(snapshot.steerRack, "steering rack");
  const steerNames = new Set<string>();
  for (const raw of steerRows) {
    const row = record(raw, "steering row");
    const name = nonempty(row.name, "steering row name");
    if (steerNames.has(name)) invalid(`duplicate steering row ${name}`);
    steerNames.add(name);
    if (!TRIGGERS.has(row.trigger as Trigger) || typeof row.enabled !== "boolean") {
      invalid(`steering row ${name}`);
    }
    if (row.mode === "jlens" || row.mode === "sae") {
      finite(row.alpha, `steering row ${name} alpha`);
      if (row.ablate !== undefined && typeof row.ablate !== "boolean") {
        invalid(`steering row ${name} ablation`);
      }
      if (row.mode === "jlens" && !name.startsWith("jlens/")) invalid(`J-lens row ${name}`);
      if (row.mode === "sae" && !/^sae\/(?:0|[1-9]\d*)$/u.test(name)) invalid(`SAE row ${name}`);
      continue;
    }
    if (row.mode !== "subspace" && row.mode !== "manifold") invalid(`steering row ${name} mode`);
    if (row.mode === "subspace" && row.ablate !== undefined && typeof row.ablate !== "boolean") {
      invalid(`steering row ${name} ablation`);
    }
    finiteArray(row.coords, `steering row ${name} coordinates`);
    nullableString(row.label, `steering row ${name} label`);
    if (typeof row.variant !== "string" || !VARIANT.test(row.variant)) {
      invalid(`steering row ${name} variant`);
    }
    if (row.mode === "manifold") {
      finite(row.blend, `steering row ${name} blend`);
      finite(row.onto, `steering row ${name} onto`);
    }
  }

  const probeRack = record(snapshot.probeRack, "probe rack");
  if (probeRack.sortMode !== "name" && probeRack.sortMode !== "value" && probeRack.sortMode !== "change") {
    invalid("probe sort mode");
  }
  const active = stringArray(probeRack.active, "active probes");
  if (new Set(active).size !== active.length) invalid("duplicate active probe");
  const probeNames = new Set<string>();
  for (const raw of array(probeRack.entries, "probe entries")) {
    const row = record(raw, "probe row");
    const name = nonempty(row.name, "probe row name");
    if (probeNames.has(name)) invalid(`duplicate probe row ${name}`);
    probeNames.add(name);
    const request = record(row.request, `probe ${name} request`);
    nonempty(request.selector, `probe ${name} selector`);
    if (request.name !== name) invalid(`probe ${name} alias`);
    if (request.top_n !== undefined) {
      finiteInteger(request.top_n, `probe ${name} nearest count`);
      if ((request.top_n as number) < 1) invalid(`probe ${name} nearest count`);
    }
    finiteArray(row.sparkline, `probe ${name} sparkline`);
    finite(row.current, `probe ${name} current value`);
    finite(row.previous, `probe ${name} previous value`);
  }
  if (active.some((name) => !probeNames.has(name)) || probeNames.size !== active.length) {
    invalid("probe roster");
  }

  const highlight = record(snapshot.highlightState, "highlight state");
  nullableString(highlight.target, "highlight target");
  nullableString(highlight.compareTarget, "highlight compare target");
  if (typeof highlight.compareTwo !== "boolean" || typeof highlight.smoothBlend !== "boolean") {
    invalid("highlight state");
  }

  const sampling = record(snapshot.samplingState, "sampling state");
  if (Object.keys(sampling).sort().join("\0") !== [...samplingKeys].sort().join("\0")) {
    invalid("sampling fields");
  }
  nullableFinite(sampling.temperature, "sampling temperature");
  nullableFinite(sampling.top_p, "sampling top-p");
  nullableFinite(sampling.top_k, "sampling top-k");
  finite(sampling.max_tokens, "sampling max tokens");
  nullableFinite(sampling.seed, "sampling seed");
  nonemptyOrEmpty(sampling.system_prompt, "sampling system prompt");
  nonemptyOrEmpty(sampling.stop_sequences, "sampling stop sequences");
  nonemptyOrEmpty(sampling.logit_bias_text, "sampling logit bias");
  finite(sampling.presence_penalty, "sampling presence penalty");
  finite(sampling.frequency_penalty, "sampling frequency penalty");
  if (sampling.thinking !== null && typeof sampling.thinking !== "boolean") invalid("sampling thinking");
  finite(sampling.return_top_k, "sampling alternatives");
  nonemptyOrEmpty(sampling.user_role, "sampling user role");
  nonemptyOrEmpty(sampling.assistant_role, "sampling assistant role");
}

export function buildConversationImportPlan(
  snapshot: ConversationSnapshotV7,
): ConversationImportPlan {
  const steerEntries = new Map<string, SteerEntry>();
  for (const { name, ...entry } of snapshot.steerRack) {
    const cloned = structuredClone(entry) as SteerEntry;
    if (cloned.mode === "subspace" || cloned.mode === "jlens" || cloned.mode === "sae") {
      cloned.ablate = cloned.ablate === true;
    }
    steerEntries.set(name, cloned);
  }
  const probeRows = new Map(
    snapshot.probeRack.entries.map((row) => [row.name, structuredClone(row)]),
  );
  return {
    snapshot,
    steerEntries,
    steeringExpression: snapshot.customSteeringExpression
      ?? serializeExpression(steerEntries, snapshot.subspaceAlong),
    probeRequests: snapshot.probeRack.active.map((name) =>
      structuredClone(probeRows.get(name)!.request)
    ),
    probeRows,
  };
}

export async function runConversationImportTransaction<T>(operations: {
  capture(): Promise<T> | T;
  preflight(): Promise<void>;
  apply(): Promise<void>;
  rollback(previous: T): Promise<void>;
}): Promise<void> {
  const previous = await operations.capture();
  await operations.preflight();
  try {
    await operations.apply();
  } catch (applyError) {
    try {
      await operations.rollback(previous);
    } catch (rollbackError) {
      throw new AggregateError(
        [applyError, rollbackError],
        "Conversation restore failed and the previous workspace could not be fully restored",
      );
    }
    throw applyError;
  }
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) invalid(label);
  return value as Record<string, unknown>;
}

function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) invalid(label);
  return value;
}

function stringArray(value: unknown, label: string): string[] {
  const values = array(value, label);
  if (!values.every((item) => typeof item === "string" && item.length > 0)) invalid(label);
  return values as string[];
}

function finite(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) invalid(label);
  return value;
}

function finiteInteger(value: unknown, label: string): number {
  const number = finite(value, label);
  if (!Number.isSafeInteger(number)) invalid(label);
  return number;
}

function nullableFinite(value: unknown, label: string): void {
  if (value !== null) finite(value, label);
}

function finiteArray(value: unknown, label: string): number[] {
  const values = array(value, label);
  if (!values.every((item) => typeof item === "number" && Number.isFinite(item))) invalid(label);
  return values as number[];
}

function exactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  label: string,
): void {
  if (Object.keys(value).sort().join("\0") !== [...expected].sort().join("\0")) invalid(label);
}

function nonempty(value: unknown, label: string): string {
  if (typeof value !== "string" || value.trim().length === 0) invalid(label);
  return value;
}

function nonemptyOrEmpty(value: unknown, label: string): void {
  if (typeof value !== "string") invalid(label);
}

function nullableString(value: unknown, label: string): void {
  if (value !== null && typeof value !== "string") invalid(label);
}

function invalid(label: string): never {
  throw new TypeError(`Invalid ${label}`);
}
