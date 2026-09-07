// Presentation-preference persistence — the localStorage half of the
// dashboard's client state.
//
// The server is the sole authority for the Loom tree.  In particular, do
// not first-paint a cached tree: after a server restart its node ids are
// invalid, and rendering it briefly triggers requests against nonexistent
// edges before the authoritative fetch arrives.  Durable conversation
// persistence is the explicit v4 whole-tree Save/Load flow; localStorage
// retains only lightweight highlight preferences.
//
// Preference changes persist through a short debounce so rapid highlight
// switching does not issue redundant synchronous localStorage writes.
//
// ``safeLocalStorage*`` are the shared guarded accessors — every other
// slice that touches storage (the per-model gen-UI render mode) goes
// through them, so quota / private-mode / SSR failures stay non-fatal in
// one place.

import { loomTree } from "./loom.svelte";
import { highlightState } from "./probes.svelte";
import { sessionState } from "./session.svelte";
import { migrateLegacyStorageItem, removeLegacyStorageItem } from "../../hosted/runtime/brandMigration";
import { registerDrowseLocalDataClearer } from "../runtime/localData";
import { PROBABILITY_TARGET, SURPRISE_TARGET } from "../tokens";

const PERSIST_VERSION = 4;
const PERSIST_KEY_PREFIX = "drowse.chat.v" + PERSIST_VERSION + ".";
const LEGACY_TREE_PERSIST_KEY_PREFIX = "drowse.chat.v3.";

function persistKey(): string | null {
  const id = sessionState.info?.model_id;
  return id ? PERSIST_KEY_PREFIX + id : null;
}

interface PersistedSnapshot {
  version: 4;
  model_id: string;
  saved_at: number;
  highlight: {
    target: string | null;
    compareTarget: string | null;
    compareTwo: boolean;
  };
}

function isPersistedSnapshot(value: unknown): value is PersistedSnapshot {
  if (!value || typeof value !== "object") return false;
  const snap = value as Record<string, unknown>;
  if (snap.version !== PERSIST_VERSION || typeof snap.model_id !== "string") return false;
  if (typeof snap.saved_at !== "number") return false;
  if (!snap.highlight || typeof snap.highlight !== "object") return false;
  const highlight = snap.highlight as Record<string, unknown>;
  if (!(typeof highlight.target === "string" || highlight.target === null)) return false;
  if (!(typeof highlight.compareTarget === "string" || highlight.compareTarget === null)) return false;
  if (typeof highlight.compareTwo !== "boolean") return false;
  return true;
}

export function safeLocalStorageGet(key: string): string | null {
  try {
    return globalThis.localStorage ? migrateLegacyStorageItem(globalThis.localStorage, key) : null;
  } catch {
    return null;
  }
}

export function safeLocalStorageSet(key: string, value: string): void {
  try {
    globalThis.localStorage?.setItem(key, value);
  } catch {
    // Quota exceeded / private-mode / SSR — silently drop.  Persistence
    // is a UX nicety, not a correctness requirement.
  }
}

function safeLocalStorageRemove(key: string): void {
  try {
    if (globalThis.localStorage) removeLegacyStorageItem(globalThis.localStorage, key);
  } catch {
    /* ignore */
  }
}

export function loadPersistedPreferences(): void {
  const key = persistKey();
  if (!key) return;
  // v3 embedded the complete tree and could occupy most of the origin quota.
  // It is unsafe after a backend restart and superseded by explicit v4
  // save/load, so reclaim it during the one model-scoped bootstrap read.
  const modelId = sessionState.info?.model_id;
  if (modelId) safeLocalStorageRemove(LEGACY_TREE_PERSIST_KEY_PREFIX + modelId);
  const raw = safeLocalStorageGet(key);
  if (!raw) return;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isPersistedSnapshot(parsed)) {
      safeLocalStorageRemove(key);
      return;
    }
    if (parsed.model_id !== sessionState.info?.model_id) return;
    loomTree.pendingNodeId = null;
    const target = parsed.highlight.target === PROBABILITY_TARGET
      ? SURPRISE_TARGET
      : parsed.highlight.target;
    const compareTarget = parsed.highlight.compareTarget === PROBABILITY_TARGET
      ? SURPRISE_TARGET
      : parsed.highlight.compareTarget;
    highlightState.target = target;
    highlightState.compareTarget = compareTarget === target ? null : compareTarget;
    highlightState.compareTwo = parsed.highlight.compareTwo && highlightState.compareTarget !== null;
  } catch {
    safeLocalStorageRemove(key);
  }
}

let _persistTimer: ReturnType<typeof setTimeout> | null = null;
function schedulePersist(): void {
  if (_persistTimer) return;
  _persistTimer = setTimeout(() => {
    _persistTimer = null;
    const key = persistKey();
    if (!key) return;
    const snapshot: PersistedSnapshot = {
      version: 4,
      model_id: sessionState.info!.model_id,
      saved_at: Date.now(),
      highlight: {
        target: highlightState.target,
        compareTarget: highlightState.compareTarget,
        compareTwo: highlightState.compareTwo,
      },
    };
    safeLocalStorageSet(key, JSON.stringify(snapshot));
  }, 250);
}

/** Watch preferences while the workbench is mounted; return its cleanup. */
export function attachPersistence(): () => void {
  _disposePersistence?.();
  const disposeRoot = $effect.root(() => {
    $effect(() => {
      // Touch every reactive field we want to persist so the effect
      // re-runs whenever any of them change.
      void highlightState.target;
      void highlightState.compareTarget;
      void highlightState.compareTwo;
      // Skip the initial call (right after restore) — saves cycles and
      // avoids overwriting the snapshot before the user has done
      // anything.  Detect via the sentinel below.
      if (!_persistArmed) {
        _persistArmed = true;
        return;
      }
      schedulePersist();
    });
  });
  const dispose = () => {
    if (_disposePersistence !== dispose) return;
    _disposePersistence = null;
    disposeRoot();
    if (_persistTimer !== null) clearTimeout(_persistTimer);
    _persistTimer = null;
    _persistArmed = false;
  };
  _disposePersistence = dispose;
  return dispose;
}

let _persistArmed = false;
let _disposePersistence: (() => void) | null = null;

registerDrowseLocalDataClearer(() => {
  if (_persistTimer !== null) clearTimeout(_persistTimer);
  _persistTimer = null;
  _persistArmed = false;
});
