// One unified steer rack.
//
// A steering vector is the K=2 flat case of a manifold, so every term is a
// position on a fitted geometry — one ``entries`` map of tagged
// ``SteerEntry`` (``mode: "subspace" | "manifold" | "jlens" | "sae"``), one
// card, and one serializer.  Subspace (flat) terms share the rack-level
// ``subspaceAlong`` master (the merged affine subspace slides once);
// manifold (curved) terms keep their per-card along/onto; the two
// single-direction atom families share one mutator set.
//
// The sidecars live here too: ``profiles`` + ``correlation`` (vector
// metadata) and ``catalog`` + ``loading`` / ``error`` (the manifold HTTP
// surface).

import { SvelteMap } from "svelte/reactivity";
import { apiManifolds, apiProfiles } from "../runtime/services";
import { userFacingError } from "../runtime/userFacingError";
import type { CorrelationData, ManifoldInfo, VectorInfo } from "../types";
import type {
  AtomMode,
  AtomSteerEntry,
  ManifoldSteerEntry,
  SteerEntry,
  SubspaceSteerEntry,
  Trigger,
  Variant,
} from "../types";
import { serializeExpression } from "../expression";
import { enqueueOrApply } from "./pending.svelte";

/** Default shared subspace-along master — the ~0.5 coherent sweet spot
 *  (matches the engine's ``_SUBSPACE_GAIN`` calibration so a freshly
 *  racked concept at its pole lands at a usable strength). */
const DEFAULT_SUBSPACE_ALONG = 0.5;

export interface SteerRack {
  /** Rack key = atom display form (``honest``, ``ns/foo``, ``happy.sad``,
   *  ``personas``).  One entry per name; ``mode`` discriminates subspace
   *  (flat) vs manifold (curved).  Variant lives on the entry, not the key —
   *  matching the drowse parser's Steering.alphas semantics. */
  entries: Map<string, SteerEntry>;
  /** Advanced full-grammar expression. ``null`` means serialize the visual
   * rack; a string (including empty = explicitly unsteered) is authoritative. */
  customExpression: string | null;
  /** Shared "subspace along" master — the single slide magnitude every
   *  subspace (flat) term serializes with (the merged affine subspace has one
   *  slide).  Unclamped (a high-share layer is meant to overshoot; the engine
   *  bounds it via ``norm_cap``).  Defaults to the ~0.5 coherent sweet spot. */
  subspaceAlong: number;
  /** Per-profile metadata fetched from GET /profiles/{name}.
   * Populated lazily; absent until the user opens a strip's expander. */
  profiles: Map<string, VectorInfo>;
  /** Cosine matrix from GET /correlation; refreshed after each generation. */
  correlation: CorrelationData | null;
  /** Server-side catalog of available manifolds. */
  catalog: ManifoldInfo[];
  loading: boolean;
  error: string | null;
}

// SvelteMap from svelte/reactivity — plain Map mutations don't trigger
// Svelte 5 rune reactivity, so any rack add/remove or profile cache
// update wouldn't re-render the strips list.  SvelteMap.set/.delete is
// rune-tracked.  Inner-object property writes still aren't tracked, so
// callers that mutate an entry must reassign via .set(name, {...e, …}).
export const steerRack: SteerRack = $state({
  entries: new SvelteMap(),
  customExpression: null,
  subspaceAlong: DEFAULT_SUBSPACE_ALONG,
  profiles: new SvelteMap(),
  correlation: null,
  catalog: [],
  loading: false,
  error: null,
});

/** Server-derived list of registered vectors — names only.  Mirrors
 * sessionState.info?.profiles but kept as its own slice so panels that
 * only care about the list don't re-render when other session fields
 * change. */
export const vectorsState: { names: string[] } = $state({ names: [] });

let correlationEpoch = 0;

/** Drop the cached matrix and prevent an older in-flight request from
 * repopulating it after the direction/probe roster changes. */
export function invalidateCorrelation(): void {
  correlationEpoch += 1;
  steerRack.correlation = null;
}

export async function refreshVectorList(): Promise<void> {
  const r = await apiProfiles.list();
  vectorsState.names = r.profiles.map((v) => v.name);
  // Cache profile metadata — cheap, server already serialized.
  steerRack.profiles.clear();
  for (const v of r.profiles) {
    steerRack.profiles.set(v.name, v);
  }
}

export async function refreshCorrelation(
  names?: string[] | null,
  strict = false,
): Promise<void> {
  const requestedAt = correlationEpoch;
  try {
    const data = await apiProfiles.correlation(names);
    if (requestedAt === correlationEpoch) steerRack.correlation = data;
  } catch (error) {
    if (requestedAt === correlationEpoch) steerRack.correlation = null;
    if (strict) throw error;
  }
}

// -------------------------------------------- subspace-mode mutators -

function defaultSubspaceEntry(
  coords: number[] = [],
  label: string | null = null,
  variant: Variant = "raw",
): SubspaceSteerEntry {
  return {
    mode: "subspace",
    ablate: false,
    coords,
    label,
    variant,
    trigger: "BOTH",
    enabled: true,
  };
}

// SvelteMap tracks .set/.delete, not mutations of stored objects, so every
// rack edit reassigns its entry with a fresh spread through ``patchSteerEntry``.

type EntryOf<M extends SteerEntry["mode"]> = Extract<SteerEntry, { mode: M }>;
type SteerPatch = { [K in keyof SteerEntry]?: SteerEntry[K] }
  & Partial<Pick<SubspaceSteerEntry, "ablate" | "coords" | "label">>
  & Partial<Pick<ManifoldSteerEntry, "blend" | "onto">>
  & Partial<Pick<AtomSteerEntry, "alpha">>;

/** Describe a rack edit for the pending-queue bubble. */
function describePatch(name: string, patch: SteerPatch): string {
  if (patch.enabled !== undefined) return `${patch.enabled ? "enable" : "disable"} ${name}`;
  if (patch.ablate !== undefined) return `${patch.ablate ? "ablate" : "push"} ${name}`;
  const [field, value] = Object.entries(patch)[0] ?? ["edit", ""];
  return `${field} ${name}${typeof value === "number" ? ` ${value.toFixed(3)}` : typeof value === "string" ? ` ${value}` : ""}`;
}

/** Apply ``patch`` to one rack entry (queued behind a running generation).
 *  ``modes`` restricts it to entries of those modes; an absent entry or one
 *  of another mode is left alone. */
export function patchSteerEntry<M extends SteerEntry["mode"]>(
  name: string,
  patch: Partial<EntryOf<M>> & SteerPatch,
  modes?: readonly M[],
): void {
  enqueueOrApply(describePatch(name, patch), () => {
    const entry = steerRack.entries.get(name);
    if (!entry || (modes && !modes.includes(entry.mode as M))) return;
    steerRack.entries.set(name, { ...entry, ...patch } as SteerEntry);
  });
}

/** Move a geometric (subspace or manifold) term to free authoring coords —
 *  an XYPad or slider drag — dropping any node-label binding so it
 *  serializes as a coord list. */
export function setSteerCoords(name: string, coords: number[]): void {
  patchSteerEntry(name, { coords: [...coords], label: null }, ["subspace", "manifold"]);
}

/** Bind a geometric term to a node label (``<name>%<label>``), mirroring the
 *  node's coords so the XYPad still renders the position; ``null`` reverts
 *  to coord form.  Without catalog metadata the label is kept as-is and
 *  resolves server-side. */
export function setSteerLabel(name: string, label: string | null): void {
  const info = label === null ? null : manifoldByName(name);
  const index = info ? info.node_labels.indexOf(label!) : -1;
  const coords = index >= 0 && info!.node_coords[index] ? [...info!.node_coords[index]] : undefined;
  patchSteerEntry(name, coords ? { label, coords } : { label }, ["subspace", "manifold"]);
}

/** The shared "subspace along" master — one slide magnitude for every
 *  subspace (flat) term (the merged affine subspace slides once).  Adjusting
 *  it scales every flat term uniformly. */
export function setSubspaceAlong(along: number): void {
  enqueueOrApply(`subspace along ${along.toFixed(3)}`, () => {
    steerRack.subspaceAlong = along;
  });
}

/** Add a flat (subspace) term.  A 2-node concept defaults to its positive
 *  pole (label form); a higher-rank flat (personas) to the domain centroid;
 *  an uncatalogued typed name to its positive pole label.  Magnitude is the
 *  shared ``subspaceAlong`` master, not per-card. */
export function addSubspaceToRack(name: string, variant: Variant = "raw"): void {
  if (steerRack.entries.has(name)) return;
  steerRack.customExpression = null;
  const info = manifoldByName(name);
  let coords: number[] = [];
  let label: string | null = null;
  if (info && info.node_count === 2 && info.node_labels.length > 0) {
    label = info.node_labels[0];
    coords = info.node_coords?.[0] ? [...info.node_coords[0]] : [];
  } else if (info) {
    coords = manifoldCentroid(info);
  } else {
    const bare = name.includes("/") ? name.slice(name.indexOf("/") + 1) : name;
    label = bare.split(".")[0];
  }
  steerRack.entries.set(name, defaultSubspaceEntry(coords, label, variant));
}

/** Take any term off the rack.  Never deletes the server-side artifact. */
export function removeSteerEntry(name: string): void {
  steerRack.entries.delete(name);
}

/** The canonical expression string the rack would send to the server.
 * Recomputed on demand; cheap.  Subspace terms first (at the shared
 * ``subspaceAlong`` master), then manifold (curved) terms. */
export function currentSteeringExpression(): string {
  return steerRack.customExpression
    ?? serializeExpression(steerRack.entries, steerRack.subspaceAlong);
}

/** Make a validated full-grammar expression authoritative.  The visual rack
 * cannot faithfully represent every binary projection/gate form, so mixing
 * the two would lie about what generation uses; switching modes clears it. */
export function applyCustomSteeringExpression(expression: string): void {
  enqueueOrApply("apply custom steering expression", () => {
    steerRack.entries.clear();
    steerRack.customExpression = expression;
  });
}

// ------------------------------------------------ manifold catalog --

/** Fetch the manifold catalog. */
export async function refreshManifoldList(): Promise<void> {
  steerRack.loading = true;
  try {
    const r = await apiManifolds.list();
    steerRack.catalog = r.manifolds;
    steerRack.error = null;
  } catch (e) {
    steerRack.catalog = [];
    steerRack.error = userFacingError(e, "Saved directions could not be refreshed.");
  } finally {
    steerRack.loading = false;
  }
}

/** Look up a catalog row by display name (``ns/name`` or bare name). */
export function manifoldByName(name: string): ManifoldInfo | null {
  for (const m of steerRack.catalog) {
    if (`${m.namespace}/${m.name}` === name || m.name === name) return m;
  }
  return null;
}

/** Domain-centroid coordinates for a manifold — the default rack
 *  position.  Box: midpoint of each axis.  Sphere: the north pole
 *  ``[0,…,0,1]`` in R^(dim+1) embedding (here we just author with
 *  ``dim`` intrinsic coords, all zero, which the domain maps to a valid
 *  point). */
export function manifoldCentroid(m: ManifoldInfo): number[] {
  if (m.domain.type === "box") {
    return m.domain.axes.map((a) => (a.lo + a.hi) / 2);
  }
  // Sphere / custom — intrinsic_dim zeros is a safe authoring default.
  return new Array(m.intrinsic_dim).fill(0);
}

// -------------------------------------------- manifold-mode mutators -

/** Add a curved manifold to the rack at its domain centroid, along 0.5. */
export function addManifoldToRack(name: string, variant: Variant = "raw"): void {
  if (steerRack.entries.has(name)) return;
  steerRack.customExpression = null;
  const info = manifoldByName(name);
  const coords = info ? manifoldCentroid(info) : [];
  steerRack.entries.set(name, {
    mode: "manifold",
    blend: 0.5,
    onto: 0,
    coords,
    label: null,
    variant,
    trigger: "BOTH",
    enabled: true,
  });
}

// ------------------------------------------------------ atom mutators
//
// The two single-direction atom families (``jlens/<word>``, ``sae/<id>``)
// rack identically — one α, one trigger, one enable, no geometry — and are
// edited through ``patchSteerEntry`` like every other term.  Only the two
// adds stay family-specific: they differ in key construction and validation.

/** Default α for a fresh atom chip.  Atoms run hotter than concept vectors
 *  (a single sharp direction, not a distributed contrast): ≈0.3 is the
 *  coherent sweet spot, ≥0.5 over-steers into repetition. */
export const ATOM_DEFAULT_ALPHA = 0.3;

/** Rack-key prefix per family — the atom's namespace segment in the
 *  steering grammar. */
export const ATOM_PREFIX: Record<AtomMode, string> = {
  jlens: "jlens/",
  sae: "sae/",
};

function addAtomToRack(mode: AtomMode, id: string): void {
  const name = `${ATOM_PREFIX[mode]}${id}`;
  if (steerRack.entries.has(name)) return;
  steerRack.customExpression = null;
  steerRack.entries.set(name, {
    mode,
    ablate: false,
    alpha: ATOM_DEFAULT_ALPHA,
    trigger: "BOTH",
    enabled: true,
  } as AtomSteerEntry);
}

/** Add a J-lens token steering chip (``α jlens/<word>``).  Accepts a bare
 *  word or a full ``jlens/…`` atom; the rack key is the full atom.
 *  Dashboard callers validate through ``apiInstruments.validateLensToken``
 *  before this local mutation; the engine revalidates when it resolves the
 *  atom. */
export function addJLensToRack(word: string): void {
  const bare = word.trim().replace(/^jlens\//, "");
  if (!bare) return;
  addAtomToRack("jlens", bare);
}

/** Add a resident-SAE decoder-row steering chip (``α sae/<id>``). */
export function addSaeToRack(featureId: number): void {
  addAtomToRack("sae", String(featureId));
}
