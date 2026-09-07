// Shared captured-or-replay readout resource for the drilldown's
// instrument tabs.
//
// Every replay-capable family (geometry / sae / lens) reads the same
// way: prefer the token's loom-owned ``measurements`` envelope (original
// capture), otherwise hit the family's ``token-readout`` replay endpoint.
// Replays are keyed and shared so pending and completed token results survive
// navigation and duplicate hover/drilldown requests reuse the same work.
// This class is the one implementation the three tabs share.

import { describeError } from "../../lib/runtime/services";
import type { RuntimeProgressEvent } from "../../lib/runtime/contracts";
import {
  cachedTokenReadout,
  tokenReadoutCacheEpoch,
} from "../../lib/runtime/tokenReadoutCache";
import type { InstrumentFamily } from "../../lib/types";
import type {
  MeasurementsEnvelopeJSON,
  ProbeReadingJSON,
} from "../../lib/types";

export type ReadoutOrigin = "captured" | "replayed" | null;

export interface ReadoutProgress {
  phase: "queued" | "context" | "readout" | "complete";
  completed: number;
  total: number;
  progress: number;
  message: string;
}

interface ReplayEntry<T> {
  state: "loading" | "ready" | "failed";
  data: T | null;
  source: string | null;
  error: string | null;
  progress: ReadoutProgress | null;
}

const MAX_REPLAY_CACHE_ENTRIES = 48;

/** The geometry tab's render shape — the Monitor-roster readings plus
 *  the steering the read ran under (captured: the recipe; replayed: the
 *  binding the endpoint reports). */
export interface GeometryTokenReadout {
  steering: string | null;
  readings: Record<string, ProbeReadingJSON>;
}

export class ReplayReadout<T> {
  data: T | null = $state(null);
  loading: boolean = $state(false);
  error: string | null = $state(null);
  origin: ReadoutOrigin = $state(null);
  source: string | null = $state(null);
  progress: ReadoutProgress | null = $state(null);
  #selectedKey: string | null = null;
  #entries = new Map<string, ReplayEntry<T>>();
  #disposed = false;

  /** Adopt the loom-captured envelope view directly (no fetch). */
  adopt(data: T, source: string | null): void {
    this.#selectedKey = null;
    this.data = data;
    this.origin = "captured";
    this.source = source;
    this.loading = false;
    this.error = null;
    this.progress = null;
  }

  /** Drop the current view without cancelling or forgetting other token jobs. */
  clear(): void {
    this.#selectedKey = null;
    this.data = null;
    this.origin = null;
    this.source = null;
    this.loading = false;
    this.error = null;
    this.progress = null;
  }

  dispose(): void {
    this.#disposed = true;
    this.clear();
    this.#entries.clear();
  }

  /** Fetch the family's token-readout replay and map the returned
   *  measurements envelope into the tab's render shape.  ``map`` may
   *  return a null source; the caller-supplied ``fallbackSource`` covers
   *  bindings from older servers. */
  replay(
    family: InstrumentFamily,
    nodeId: string,
    rawIndex: number,
    opts: { topK?: number; steered?: boolean; raw?: boolean; layers?: string },
    map: (env: MeasurementsEnvelopeJSON) => {
      data: T;
      source: string | null;
    },
  ): void {
    const key = replayKey(family, nodeId, rawIndex, opts);
    this.#selectedKey = key;
    const cached = this.#entries.get(key);
    if (cached && cached.state !== "failed") {
      this.#entries.delete(key);
      this.#entries.set(key, cached);
      this.#show(key, cached);
      return;
    }
    const entry: ReplayEntry<T> = {
      state: "loading",
      data: null,
      source: null,
      error: null,
      progress: {
        phase: "queued",
        completed: 0,
        total: rawIndex + 1,
        progress: 0,
        message: "Waiting for the current model task to finish",
      },
    };
    this.#entries.set(key, entry);
    this.#show(key, entry);
    cachedTokenReadout(family, nodeId, rawIndex, opts, (event) => {
        const progress = readoutProgress(event);
        if (progress === null) return;
        entry.progress = progress;
        if (this.#selectedKey === key) this.#show(key, entry);
      })
      .then((res) => {
        const { data, source } = map(res.measurements);
        entry.state = "ready";
        entry.data = data;
        entry.source = source;
        entry.error = null;
        entry.progress = null;
      })
      .catch((e) => {
        entry.state = "failed";
        entry.error = describeError(e);
        entry.progress = null;
      })
      .finally(() => {
        this.#trimCache();
        if (this.#selectedKey === key) this.#show(key, entry);
      });
  }

  #show(key: string, entry: ReplayEntry<T>): void {
    if (this.#disposed || this.#selectedKey !== key) return;
    this.loading = entry.state === "loading";
    this.data = entry.data;
    this.error = entry.error;
    this.origin = entry.state === "ready" ? "replayed" : null;
    this.source = entry.source;
    this.progress = entry.progress;
  }

  #trimCache(): void {
    if (this.#entries.size <= MAX_REPLAY_CACHE_ENTRIES) return;
    for (const [key, entry] of this.#entries) {
      if (this.#entries.size <= MAX_REPLAY_CACHE_ENTRIES) break;
      if (entry.state === "loading" || key === this.#selectedKey) continue;
      this.#entries.delete(key);
    }
  }
}

function replayKey(
  family: InstrumentFamily,
  nodeId: string,
  rawIndex: number,
  opts: { topK?: number; steered?: boolean; raw?: boolean; layers?: string },
): string {
  return JSON.stringify([
    family,
    tokenReadoutCacheEpoch(family),
    nodeId,
    rawIndex,
    opts.topK ?? null,
    opts.steered ?? true,
    opts.raw ?? false,
    opts.layers ?? null,
  ]);
}

function readoutProgress(event: RuntimeProgressEvent): ReadoutProgress | null {
  if (event.event !== "progress" || event.data === null || typeof event.data !== "object") {
    return null;
  }
  const value = event.data as Record<string, unknown>;
  if (
    value.kind !== "token_readout" ||
    !["queued", "context", "readout", "complete"].includes(String(value.phase)) ||
    !Number.isInteger(value.completed) ||
    !Number.isInteger(value.total) ||
    typeof value.progress !== "number" ||
    !Number.isFinite(value.progress) ||
    typeof value.message !== "string"
  ) {
    return null;
  }
  return {
    phase: value.phase as ReadoutProgress["phase"],
    completed: Math.max(0, Number(value.completed)),
    total: Math.max(0, Number(value.total)),
    progress: Math.max(0, Math.min(1, value.progress)),
    message: value.message,
  };
}
