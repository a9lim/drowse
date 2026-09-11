// The three read-side instrument families' client state.
//
// One slice per the server's own split: the live J-lens readout, the live
// SAE feature readout, and the geometry (CAA) live toggle — plus the two
// things every family shares, the source registry (list / switch / fetch)
// and the browser-safe fetch jobs. Python-only SAE training and J-lens
// fitting live in instrumentAuthoring.svelte.ts so the hosted bundle never
// imports their stores.
//
// ``tokenHoverState`` is the non-destructive token-anchored overlay across
// all three: hovering a transcript token shows THAT token's readings, and
// the panels fall back to their ordinary live/settled state on leave.
// ``mergedReadings`` / ``lensReadoutSnapshot`` are the envelope readers
// every consumer of a measurements envelope goes through, so the wire
// shape is decoded in exactly one place.

import { SvelteMap } from "svelte/reactivity";
import { apiInstruments } from "../runtime/services";
import {
  cachedTokenReadout,
  invalidateTokenReadoutCache,
} from "../runtime/tokenReadoutCache";
import { userFacingError } from "../runtime/userFacingError";
import type {
  AnyReadingJSON,
  InstrumentSourceJSON,
  LensReadoutBlockJSON,
  MeasurementsEnvelopeJSON,
  SaeFeatureJSON,
  TokenScore,
} from "../types";
import { resolveReadoutTopK } from "../readouts";
import { pushToast } from "./toasts.svelte";
import { createPreparationSlice } from "./preparations.svelte";
import { effectiveRawMode } from "./chat.svelte";
import { samplingState } from "./sampling.svelte";
import { MAX_SPARKLINE, refreshProbeList } from "./probes.svelte";
import {
  instrumentFamily,
  refreshSession,
  saeLoaded,
  sessionState,
} from "./session.svelte";

// ------------------------------------------------------ live lens ---

export interface LensState {
  /** Resolved fitted-layer list while the live J-lens readout is
   * enabled; ``null`` while off.  Mirrors the server's
   * lens family block's ``live.layers`` — the panel toggle reads this, not
   * a local boolean, so reloads and multi-tab stay honest. */
  layers: number[] | null;
  /** Latest decode step's readout: layer-index string → descending
   * ``[token, p]`` top-k (per-layer softmax probability — the one
   * strength unit every lens surface reports).  Overwritten per token
   * frame; kept after ``done`` so the settled matrix stays readable. */
  readout: Record<string, [string, number][]> | null;
  /** Layer-aggregated chip list riding the same step — ``[token,
   * strength, com, spread]`` strength-descending (mean fitted-layer probability
   * + probability-mass-weighted depth center of mass).  Same lifecycle as
   * ``readout``. */
  aggregate: [string, number, number, number][] | null;
  /** Rolling buffer of recent aggregate frames (``[token, strength]``
   * pairs per step, newest last, capped like the probe sparklines) —
   * backs the workspace token cards' sparklines.  Carries across
   * generations like probe sparklines; cleared on live-lens disable. */
  aggHistory: [string, number][][];
  /** Presentation order for the aggregate workspace cards.  Kept in the
   * shared lens state so switching inspector tabs does not reset it. */
  workspaceSortMode: LensWorkspaceSortMode;
  /** In-flight toggle guard (the enable moves J_l device-resident and
   * waits on the session lock, so it can lag behind a long stream). */
  busy: boolean;
}

export type LensWorkspaceSortMode = "strength" | "name" | "depth";

export const lensState: LensState = $state({
  layers: null,
  readout: null,
  aggregate: null,
  aggHistory: [],
  workspaceSortMode: "strength",
  busy: false,
});

export const lensSourceState: {
  sources: InstrumentSourceJSON[];
  loading: boolean;
  busy: boolean;
  error: string | null;
} = $state({ sources: [], loading: false, busy: false, error: null });

export async function refreshLensSources(): Promise<void> {
  if (lensSourceState.loading) return;
  lensSourceState.loading = true;
  try {
    lensSourceState.sources = (await apiInstruments.sources("lens")).sources;
    lensSourceState.error = null;
  } catch (e) {
    lensSourceState.error = userFacingError(e, "Word-likelihood sources could not be refreshed.");
  } finally {
    lensSourceState.loading = false;
  }
}

export async function useLensSource(source: string): Promise<void> {
  if (lensSourceState.busy || !source) return;
  lensSourceState.busy = true;
  lensSourceState.error = null;
  try {
    const block = instrumentFamily("lens");
    if (block?.capabilities.source_switch === true) {
      const out = await apiInstruments.setLensSource(source);
      lensState.layers = out.live_layers;
    } else {
      const out = await apiInstruments.activateInstalledPack("lens", { source });
      lensState.layers = out.live.enabled && "layers" in out.live
        ? out.live.layers ?? []
        : null;
    }
    invalidateTokenReadoutCache("lens");
    await refreshSession();
    await refreshLensSources();
    pushToast(`J-lens · ${source}`, { kind: "info" });
  } catch (e) {
    lensSourceState.error = userFacingError(
      e,
      "Word insights could not start. Close and reopen the model after changing its tools.",
    );
    pushToast(
      `J-lens source: ${lensSourceState.error}`,
      { kind: "error" },
    );
  } finally {
    lensSourceState.busy = false;
  }
}

export function setLensWorkspaceSortMode(mode: LensWorkspaceSortMode): void {
  lensState.workspaceSortMode = mode;
}

// ------------------------------------------------------- live SAE ---

export interface SaeState {
  live: boolean;
  readout: SaeFeatureJSON[];
  /** Raw activation history per feature id (drives the sparklines; the
   *  bars derive their scale from ``meta`` instead). */
  history: Map<number, number[]>;
  /** Session-side Neuronpedia metadata per feature id — merged from the
   *  token frames' cached values and the between-generation backfill.
   *  ``max_act`` is the strength unit: bars render
   *  ``activation / max_act`` on the absolute 0..1 scale (the lens-card
   *  convention); features without it fall back to the panel-shared raw
   *  scale. */
  meta: Map<number, { label: string | null; max_act: number | null }>;
  /** Resident release the discovery state belongs to (reset key). */
  release: string | null;
  /** Resident hook layer; changing it invalidates feature ids/history too. */
  layer: number | null;
  /** Presentation order shared across tab switches, mirroring the lens
   *  workspace sorter. */
  sortMode: SaeSortMode;
  busy: boolean;
}

export const saeState: SaeState = $state({
  live: false,
  readout: [],
  history: new SvelteMap<number, number[]>(),
  meta: new SvelteMap<number, { label: string | null; max_act: number | null }>(),
  release: null,
  layer: null,
  sortMode: "strength",
  busy: false,
});

// --------------------------------------------- token hover readout --

/** A non-destructive, token-anchored view of the three read channels shown in
 * the inspector.  Panels read this overlay while a transcript token is under
 * the pointer, then fall back to their ordinary live/settled state on leave. */
export interface TokenHoverState {
  active: boolean;
  key: string | null;
  tokenText: string;
  probeReadings: Record<string, AnyReadingJSON> | null;
  probes: Record<string, number> | null;
  coordsByProbe: Record<string, number[]> | null;
  perLayerScores: Record<string, Record<string, number>> | null;
  lensReadout: Record<string, [string, number][]> | null;
  lensAggregate: [string, number, number, number][] | null;
  saeReadout: SaeFeatureJSON[] | null;
  lensLoading: boolean;
  saeLoading: boolean;
  lensError: string | null;
  saeError: string | null;
}

export const tokenHoverState: TokenHoverState = $state({
  active: false,
  key: null,
  tokenText: "",
  probeReadings: null,
  probes: null,
  coordsByProbe: null,
  perLayerScores: null,
  lensReadout: null,
  lensAggregate: null,
  saeReadout: null,
  lensLoading: false,
  saeLoading: false,
  lensError: null,
  saeError: null,
});

interface LensHoverSnapshot {
  readout: Record<string, [string, number][]>;
  aggregate: [string, number, number, number][];
}

const TOKEN_HOVER_DELAY_MS = 140;
let _hoverFetchTimer: ReturnType<typeof setTimeout> | null = null;
let _hoverClearTimer: ReturnType<typeof setTimeout> | null = null;

function _hoverTokenKey(nodeId: string, rawIndex: number, raw: boolean): string {
  const modelKey = sessionState.info?.model_id ?? "unknown-model";
  return `${modelKey}:${nodeId}:${rawIndex}:${raw ? 1 : 0}`;
}

/** The lens native readout block → the display shape the workspace cards and
 *  hover consume: per-layer ``[token, probability]`` (the wire carries
 *  logprob — ``exp()`` restores the displayed per-layer softmax probability)
 *  plus the ``[token, strength, com, spread]`` aggregate chips. */
export function lensReadoutSnapshot(
  block: LensReadoutBlockJSON | undefined,
): LensHoverSnapshot | null {
  if (!block) return null;
  return {
    readout: Object.fromEntries(block.layers.map((row) => [
      String(row.layer),
      row.tokens.map((token) => [token.token, Math.exp(token.logprob)]),
    ])),
    aggregate: block.aggregate.map((token) => [
      token.token,
      token.strength,
      token.com,
      token.spread,
    ]),
  };
}

/** Merge the three families' attached-probe ``readings`` from a measurement
 *  envelope — the probe rack keys by name across families, so one dict drives
 *  every probe strip. */
export function mergedReadings(
  m: MeasurementsEnvelopeJSON | undefined | null,
): Record<string, AnyReadingJSON> | undefined {
  if (!m) return undefined;
  const g = m.instruments.geometry?.readings;
  const l = m.instruments.lens?.readings;
  const s = m.instruments.sae?.readings;
  if (!g && !l && !s) return undefined;
  return { ...(g ?? {}), ...(l ?? {}), ...(s ?? {}) };
}

function _fetchLensHover(
  nodeId: string,
  rawIndex: number,
  raw: boolean,
): Promise<LensHoverSnapshot | null> {
  return cachedTokenReadout("lens", nodeId, rawIndex, {
    topK: resolveReadoutTopK(samplingState.return_top_k),
    steered: true,
    raw,
    layers: "all",
  }, undefined, "background").then((res) => (
    lensReadoutSnapshot(res.measurements.instruments.lens?.readout)
  ));
}

function _fetchSaeHover(
  nodeId: string,
  rawIndex: number,
  raw: boolean,
): Promise<SaeFeatureJSON[]> {
  return cachedTokenReadout("sae", nodeId, rawIndex, {
    topK: resolveReadoutTopK(samplingState.return_top_k),
    steered: true,
    raw,
  }, undefined, "background").then(
    (res) => res.measurements.instruments.sae?.readout?.features ?? [],
  );
}

/** Begin showing one transcript token in the inspector. Loom-owned captures
 * land synchronously; a channel absent from the original generation may be
 * replayed after a short dwell, but replay results are not substituted for or
 * retained as original capture data. */
export function beginTokenHover(
  token: TokenScore,
  nodeId: string | null | undefined,
): void {
  if (_hoverClearTimer !== null) clearTimeout(_hoverClearTimer);
  if (_hoverFetchTimer !== null) clearTimeout(_hoverFetchTimer);
  _hoverClearTimer = null;
  _hoverFetchTimer = null;

  const raw = effectiveRawMode();
  const rawIndex = token.rawIndex ?? null;
  const modelKey = sessionState.info?.model_id ?? "unknown-model";
  const key = nodeId && rawIndex !== null
    ? _hoverTokenKey(nodeId, rawIndex, raw)
    : `live:${modelKey}:${nodeId ?? "none"}:${rawIndex ?? "none"}:${token.tokenId ?? "none"}`;
  const m = token.measurements;
  const capturedLens = m?.instruments.lens?.readout;
  const capturedSae = m?.instruments.sae?.readout;
  tokenHoverState.active = true;
  tokenHoverState.key = key;
  tokenHoverState.tokenText = token.text;
  tokenHoverState.probeReadings = mergedReadings(m) ?? null;
  tokenHoverState.probes = m?.scores ?? token.probes ?? null;
  tokenHoverState.coordsByProbe = token.coordsByProbe ?? null;
  tokenHoverState.perLayerScores =
    m?.per_layer_scores ?? token.perLayerScores ?? null;
  tokenHoverState.lensReadout = null;
  tokenHoverState.lensAggregate = null;
  tokenHoverState.saeReadout = capturedSae?.features ?? null;
  tokenHoverState.lensLoading = false;
  tokenHoverState.saeLoading = false;
  tokenHoverState.lensError = null;
  tokenHoverState.saeError = null;

  const lensSnapshot = lensReadoutSnapshot(capturedLens);
  if (lensSnapshot) {
    tokenHoverState.lensReadout = lensSnapshot.readout;
    tokenHoverState.lensAggregate = lensSnapshot.aggregate;
  }

  if (!nodeId || rawIndex === null) return;
  const needsLens = capturedLens === undefined &&
    sessionState.info?.jlens_fitted === true &&
    instrumentFamily("lens")?.capabilities.token_readout === true;
  const needsSae = capturedSae === undefined && saeLoaded() &&
    instrumentFamily("sae")?.capabilities.token_readout === true;
  tokenHoverState.lensLoading = needsLens;
  tokenHoverState.saeLoading = needsSae;
  if (!needsLens && !needsSae) return;

  _hoverFetchTimer = setTimeout(() => {
    _hoverFetchTimer = null;
    if (needsLens) {
      void _fetchLensHover(nodeId, rawIndex, raw)
        .then((snapshot) => {
          if (!tokenHoverState.active || tokenHoverState.key !== key) return;
          if (!snapshot) return;
          tokenHoverState.lensReadout = snapshot.readout;
          tokenHoverState.lensAggregate = snapshot.aggregate;
        })
        .catch((error) => {
          if (!tokenHoverState.active || tokenHoverState.key !== key) return;
          tokenHoverState.lensError = userFacingError(
            error,
            "This token's word-likelihood reading could not be rebuilt. Try again, or close and reopen the model.",
          );
        })
        .finally(() => {
          if (tokenHoverState.active && tokenHoverState.key === key) {
            tokenHoverState.lensLoading = false;
          }
        });
    }
    if (needsSae) {
      void _fetchSaeHover(nodeId, rawIndex, raw)
        .then((features) => {
          if (!tokenHoverState.active || tokenHoverState.key !== key) return;
          tokenHoverState.saeReadout = features;
        })
        .catch((error) => {
          if (!tokenHoverState.active || tokenHoverState.key !== key) return;
          tokenHoverState.saeError = userFacingError(
            error,
            "This token's feature reading could not be rebuilt. Try again, or close and reopen the model.",
          );
        })
        .finally(() => {
          if (tokenHoverState.active && tokenHoverState.key === key) {
            tokenHoverState.saeLoading = false;
          }
        });
    }
  }, TOKEN_HOVER_DELAY_MS);
}

/** Delay the clear just enough to cross the whitespace between adjacent token
 * spans without flashing the inspector back to the settled generation. */
export function endTokenHover(): void {
  if (_hoverFetchTimer !== null) clearTimeout(_hoverFetchTimer);
  if (_hoverClearTimer !== null) clearTimeout(_hoverClearTimer);
  _hoverFetchTimer = null;
  _hoverClearTimer = setTimeout(() => {
    tokenHoverState.active = false;
    tokenHoverState.key = null;
    tokenHoverState.tokenText = "";
    tokenHoverState.lensLoading = false;
    tokenHoverState.saeLoading = false;
    tokenHoverState.lensError = null;
    tokenHoverState.saeError = null;
    _hoverClearTimer = null;
  }, 45);
}

export function lensReadoutForDisplay(): Record<string, [string, number][]> | null {
  return tokenHoverState.active ? tokenHoverState.lensReadout : lensState.readout;
}

export function lensAggregateForDisplay(): [string, number, number, number][] | null {
  return tokenHoverState.active ? tokenHoverState.lensAggregate : lensState.aggregate;
}

export function saeReadoutForDisplay(): SaeFeatureJSON[] {
  return tokenHoverState.active ? (tokenHoverState.saeReadout ?? []) : saeState.readout;
}

export const saeSourceState: {
  sources: InstrumentSourceJSON[];
  loading: boolean;
  busy: boolean;
  error: string | null;
} = $state({ sources: [], loading: false, busy: false, error: null });

export async function refreshSaeSources(): Promise<void> {
  if (saeSourceState.loading) return;
  saeSourceState.loading = true;
  try {
    saeSourceState.sources = (await apiInstruments.sources("sae")).sources;
    saeSourceState.error = null;
  } catch (e) {
    saeSourceState.error = userFacingError(e, "Learned-feature sources could not be refreshed.");
  } finally {
    saeSourceState.loading = false;
  }
}

export type SaeSortMode = "strength" | "name";

export function setSaeSortMode(mode: SaeSortMode): void {
  saeState.sortMode = mode;
}

/** Ids already sent to the metadata backfill this session — a miss on
 *  Neuronpedia stays a miss, so don't re-ask every generation.  Not
 *  reactive state (never rendered). */
const _saeMetaRequested = new Set<number>();
let saeMetadataEpoch = 0;
export const MAX_SAE_HISTORY_FEATURES = 512;

export function recordSaeReadoutFrame(features: SaeFeatureJSON[]): void {
  saeState.readout = features;
  for (const feature of features) {
    const prior = saeState.history.get(feature.id) ?? [];
    const next = [...prior, feature.activation].slice(-MAX_SPARKLINE);
    saeState.history.delete(feature.id);
    saeState.history.set(feature.id, next);
    if (feature.max_act != null || feature.label != null) {
      saeState.meta.set(feature.id, {
        label: feature.label ?? saeState.meta.get(feature.id)?.label ?? null,
        max_act: feature.max_act ?? saeState.meta.get(feature.id)?.max_act ?? null,
      });
    }
  }
  while (saeState.history.size > MAX_SAE_HISTORY_FEATURES) {
    const oldest = saeState.history.keys().next().value;
    if (oldest === undefined) break;
    saeState.history.delete(oldest);
    saeState.meta.delete(oldest);
    _saeMetaRequested.delete(oldest);
  }
}

/** Between-generation discovery backfill: fetch-and-cache Neuronpedia
 *  metadata (label + maxActApprox) for every feature the live top-k
 *  surfaced that has none yet.  Fire-and-forget from the ``done``
 *  handler — never per token. */
export async function backfillSaeMeta(): Promise<void> {
  if (!saeLoaded()) return;
  const wanted: number[] = [];
  for (const id of saeState.history.keys()) {
    if (saeState.meta.get(id)?.max_act != null && saeState.meta.get(id)?.label?.trim()) continue;
    if (_saeMetaRequested.has(id)) continue;
    wanted.push(id);
    if (wanted.length >= 64) break;
  }
  if (wanted.length === 0) return;
  for (const id of wanted) _saeMetaRequested.add(id);
  const epoch = saeMetadataEpoch;
  try {
    const out = await apiInstruments.saeFeaturesMetadata(wanted);
    if (epoch !== saeMetadataEpoch) return;
    for (const id of wanted) {
      if (!Object.hasOwn(out.features, String(id))) _saeMetaRequested.delete(id);
    }
    for (const [key, entry] of Object.entries(out.features)) {
      const prior = saeState.meta.get(Number(key));
      saeState.meta.set(Number(key), {
        label: entry.label ?? prior?.label ?? null,
        max_act: entry.max_act ?? prior?.max_act ?? null,
      });
    }
  } catch {
    if (epoch !== saeMetadataEpoch) return;
    // Best-effort — allow a retry on the next generation.
    for (const id of wanted) _saeMetaRequested.delete(id);
  }
}

export async function setLiveSae(enabled: boolean): Promise<void> {
  if (saeState.busy) return;
  saeState.busy = true;
  try {
    const out = await apiInstruments.setLive("sae", { enabled });
    saeState.live = out.enabled;
    if (!out.enabled) {
      saeMetadataEpoch++;
      saeState.readout = [];
      saeState.history.clear();
      saeState.meta.clear();
      _saeMetaRequested.clear();
    }
  } catch (e) {
    pushToast(
      `SAE live: ` +
        userFacingError(e, "That learned-feature reading is not available."),
      { kind: "error" },
    );
  } finally {
    saeState.busy = false;
  }
}

/** Browser-safe source preparation — see
 *  ``lib/stores/preparations.svelte.ts`` for the shared contract. */
// The SAE source-binding preparation.  The HTTP operation is ``fetch``,
// matching the CLI verb and the lens family (it used to be spelled
// ``load`` over HTTP alone).
export const saeLoad = createPreparationSlice("sae", "fetch", {
  label: "SAE fetch",
  intervalMs: 1000,
  successMessage: "SAE loaded",
  onSettled: async () => {
    await refreshSession();
    await refreshSaeSources();
    await refreshProbeList();
  },
});

/** Load a resident SAE — ``local:<name>`` or ``saelens:<release>``, with an
 *  optional hook layer.  Thin wrapper over the slice: the release is the
 *  only field that needs trimming + an empty check. */
export async function loadSae(release: string, layer: number | null = null): Promise<void> {
  const trimmed = release.trim();
  if (!trimmed) return;
  const block = instrumentFamily("sae");
  if (block?.capabilities.preparations.includes("fetch") === true) {
    await saeLoad.start({ release: trimmed, layer });
    return;
  }
  if (saeSourceState.busy) return;
  saeSourceState.busy = true;
  saeSourceState.error = null;
  try {
    await apiInstruments.activateInstalledPack("sae", {
      source: trimmed,
      layer,
    });
    invalidateTokenReadoutCache("sae");
    await refreshSession();
    await refreshSaeSources();
    await refreshProbeList();
    pushToast(`SAE · ${trimmed}`, { kind: "info" });
  } catch (error) {
    saeSourceState.error = userFacingError(
      error,
      "Model features could not start. Close and reopen the model after changing its tools.",
    );
    pushToast(`SAE source: ${saeSourceState.error}`, { kind: "error" });
  } finally {
    saeSourceState.busy = false;
  }
}

// ------------------------------------ live toggles + preparations ----

/** CAA PROBE-section live toggle — whether per-token monitor scoring
 *  feeds live consumers.  The J-lens sibling is ``lensState.layers``
 *  (the live lens); with both off a compute-constrained session pays no
 *  per-token scoring at all, and every probe still reports its
 *  end-of-gen aggregate. */
export const probesLiveState: { enabled: boolean; busy: boolean } = $state({
  enabled: true,
  busy: false,
});

export async function setLiveProbes(enabled: boolean): Promise<void> {
  if (probesLiveState.busy) return;
  probesLiveState.busy = true;
  try {
    const out = await apiInstruments.setLive("geometry", { enabled });
    probesLiveState.enabled = out.enabled;
  } catch (e) {
    pushToast(
      `probe live: ` +
        userFacingError(e, "That concept reading is not available."),
      { kind: "error" },
    );
  } finally {
    probesLiveState.busy = false;
  }
}

/** Inspector-column mode — the four instrument pillars.  All four tabs are
 *  views over the ONE steering expression / probe roster; the split is
 *  presentational (each tab shows its own term/probe family):
 *    subspace — flat/affine fits (concept axes, personas)
 *    manifold — curved fits (emotions, months)
 *    sae      — resident sparse-autoencoder feature space
 *    lens     — the Jacobian-lens surface (JLensPanel) */
export type InspectorTab = "subspace" | "manifold" | "sae" | "lens";

export const inspectorState: { tab: InspectorTab } = $state({
  tab: "subspace",
});

export function setInspectorTab(tab: InspectorTab): void {
  inspectorState.tab = tab;
}

/** Toggle the live J-lens readout server-side. J-lens and SAE both follow the
 * same per-generation ``return_top_k`` value as the logit alternatives. */
export async function setLiveLens(enabled: boolean): Promise<void> {
  if (lensState.busy) return;
  lensState.busy = true;
  try {
    const out = await apiInstruments.setLive("lens", { enabled });
    lensState.layers = out.enabled && "layers" in out ? (out.layers ?? []) : null;
    if (!out.enabled) {
      lensState.readout = null;
      lensState.aggregate = null;
      lensState.aggHistory = [];
    }
  } catch (e) {
    pushToast(
      `lens live: ` +
        userFacingError(e, "Word-likelihood details could not be loaded."),
      { kind: "error" },
    );
  } finally {
    lensState.busy = false;
  }
}

export const lensFetch = createPreparationSlice("lens", "fetch", {
  label: "J-lens fetch",
  intervalMs: 1000,
  successMessage: "J-lens active · live",
  onSettled: async () => {
    await refreshSession();
    await refreshLensSources();
  },
});

// ------------------------------------ session-info rehydration -------

let replayLensIdentity: string | null = null;
let replaySaeIdentity: string | null = null;

/** Rehydrate every family's live/source state from ONE server
 *  representation — the same per-family blocks ``GET .../instruments``
 *  lists.  Called from ``refreshSession``; a page reload must reflect the
 *  server exactly. */
export function rehydrateInstrumentsFromSession(): void {
  const lens = instrumentFamily("lens");
  const sae = instrumentFamily("sae");
  const geometry = instrumentFamily("geometry");
  const lensIdentity = lens?.source ?? null;
  const saeIdentity = sae === undefined
    ? null
    : `${sae.source ?? ""}:${"layer" in sae.live ? sae.live.layer ?? "" : ""}`;
  if (lensIdentity !== replayLensIdentity) invalidateTokenReadoutCache("lens");
  if (saeIdentity !== replaySaeIdentity) invalidateTokenReadoutCache("sae");
  replayLensIdentity = lensIdentity;
  replaySaeIdentity = saeIdentity;
  lensState.layers = lens?.live.enabled
    ? ("layers" in lens.live ? lens.live.layers : null)
    : null;
  saeState.live = sae?.live.enabled === true;
  // Feature ids (and their metadata) belong to the resident release —
  // reset the discovery/metadata state when it changes.
  const release = sae?.source ?? null;
  const layer = sae && "layer" in sae.live ? sae.live.layer : null;
  if (release !== saeState.release || layer !== saeState.layer) {
    saeMetadataEpoch++;
    saeState.release = release;
    saeState.layer = layer;
    saeState.readout = [];
    saeState.history.clear();
    saeState.meta.clear();
    _saeMetaRequested.clear();
  }
  probesLiveState.enabled = geometry?.live.enabled !== false;
}
