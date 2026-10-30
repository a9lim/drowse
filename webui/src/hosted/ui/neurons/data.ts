// Data for the /neurons explorer. The files are prebuilt by scripts/neurons/build-neuron-catalog.py
// from the neuron-lens catalog and served as static JSON next to the page.

export const DATA_BASE = "/neurons-data";

export type Band = "embed" | "early" | "crossover" | "concept" | "dark" | "predict" | "suppress";
export type Trust = "primary" | "trusted" | "weak" | "compare";
export type ReadColumn =
  | "read_S" | "read_S_z" | "read_R_dclm" | "read_R_dclm_z"
  | "write_S" | "write_S_z" | "write_R_dclm" | "write_R_dclm_z"
  | "write_J" | "write_logit";

/** [token, variants merged into it, rank of its first variant, flags: 1 shared by another lens, 2 confirmed by measurement]. */
export type LensToken = [string, number, number, number];

export interface Head {
  h: number; kv: number; sink: number; prev: number; self: number; induction: number; duplicate: number;
  copy: number; next: boolean; copies: boolean; dark: boolean; reads: string;
}

export interface LayerMeta {
  layer: number;
  band: Band;
  missing: boolean;
  ins?: ReadColumn[];
  outs?: ReadColumn[];
  stats?: { silent: number; detector: number; legible: number; dark: number } | null;
  hubs?: string[];
  count?: number;
  partial?: string[];
  heads?: Head[];
}

/** [layer, neuron, title of the + pole, title of the − pole, score, flags]. */
export type IndexRow = [number, number, string, string, number, number];
export const FLAG_LABELLED = 1;
export const FLAG_DETECTOR = 2;
export const FLAG_PREDICTS = 4;
export const FLAG_SUPPRESSES = 8;

export interface CatalogIndex {
  model: string;
  layers: number;
  neuronsPerLayer: number;
  total: number;
  layerMeta: LayerMeta[];
  neurons: IndexRow[];
}

export interface LensRead {
  col: ReadColumn;
  trust: Trust;
  /** True when this is the raw or z twin of a read shown above it: only its extra tokens are listed. */
  twin: boolean;
  /** Mean top-8 score as a percentile of the layer's 24,576 poles. */
  pct: number;
  tokens: LensToken[];
  hubs: number;
}

export interface Label {
  text: string;
  confidence: number;
  relation?: string | null;
  note?: string;
  source: string;
}

export interface Pole {
  pole: 1 | -1;
  reads: LensRead[];
  shared: string[];
  inout: string[];
  measuredAgree: string[];
  w_norm: number;
  null_share: number;
  act_top?: number;
  act_max?: number;
  rate?: number;
  /** Firing rate on web, Chinese, code and chat text. */
  slices?: [number, number, number, number];
  peaks?: [string, number][];
  contexts?: [string, string, string][];
  silent: boolean;
  drivers?: [string, number][];
  driversSum?: [string, number][];
  s_hits?: number;
  legible: Record<string, number>;
  detector: "token" | "next" | null;
  modes: [number, number];
  shape?: { kurtPct: number; skew: number; kind: "predicts" | "suppresses" | null };
  label?: Label;
  score: number;
}

export interface NeuronDetail {
  n: number;
  titles: [string, string];
  poles: [Pole, Pole];
}

export interface LayerFile { layer: number; neurons: NeuronDetail[] }
export interface SearchFile { forms: Record<string, [number, 0 | 1, number][]>; display: Record<string, string> }
export type Glossary = Record<string, [string, string]>;

const cache = new Map<string, Promise<unknown>>();

function load<T>(name: string): Promise<T> {
  let pending = cache.get(name) as Promise<T> | undefined;
  if (!pending) {
    pending = fetch(`${DATA_BASE}/${name}`).then((response) => {
      if (!response.ok) throw new Error(`${name}: ${response.status}`);
      return response.json() as Promise<T>;
    });
    // A failed request can be retried later instead of failing forever.
    pending.catch(() => cache.delete(name));
    cache.set(name, pending);
  }
  return pending;
}

export const loadIndex = () => load<CatalogIndex>("index.json");
export const loadLayer = (layer: number) => load<LayerFile>(`L${String(layer).padStart(2, "0")}.json`);
export const loadSearch = () => load<SearchFile>("search.json");
/** Chinese glosses for the tokens shown in one layer's file. */
export const loadGlossary = (layer: number) => load<Glossary>(`L${String(layer).padStart(2, "0")}.zh.json`);

// ---------- depth bands: what the catalog's checks say to trust at each depth

export interface BandInfo { name: string; range: string; summary: string; reading: string }

export const BANDS: Record<Band, BandInfo> = {
  embed: {
    name: "Token layer", range: "L0",
    summary: "Neurons here mostly respond to one literal input token.",
    reading: "The input-token read (in·S) names the token for 82% of poles. Writes are mostly noise this early.",
  },
  early: {
    name: "Early layers", range: "L1–L7",
    summary: "Neurons respond to input tokens a few positions back; many are still silent.",
    reading: "Trust the input-token reads (in·S). From L5, the write read predicts what pushing the neuron does.",
  },
  crossover: {
    name: "Crossover", range: "L8",
    summary: "The layer where the input-token reads and the concept reads both carry signal.",
    reading: "All three input reads are informative here. Tokens they share are the strongest evidence.",
  },
  concept: {
    name: "Concept layers", range: "L9–L12",
    summary: "Neurons respond to concepts more than to literal tokens.",
    reading: "The concept read (in·R) takes over from in·S, which has gone to noise.",
  },
  dark: {
    name: "Middle layers", range: "L13–L20",
    summary: "Most neurons here aren't read by any weight-only lens. They respond to context.",
    reading: "Read the measured contexts and drivers first. Lens reads are only a cross-check here.",
  },
  predict: {
    name: "Prediction layers", range: "L21–L31",
    summary: "Neurons fire just before a predictable word, and many write that word.",
    reading: "in·R is often the word about to come next. The write read names what the neuron predicts.",
  },
  suppress: {
    name: "Suppression layers", range: "L32–L35",
    summary: "Many neurons push down the word the model is about to say, to calibrate it.",
    reading: "Read both poles' writes: a pole whose bottom tokens are the next word suppresses it.",
  },
};

/** Shown beside layers 13–20, where every weight-only lens is at its weakest (findings §15). */
export const MIDDLE_CAUTION = "Lens readings are least reliable at layers 13–20. The S-lens and R-lens read a neuron from its weights, but most neurons here respond to context built up over many tokens, not to one token or word. Trust the real-text examples first.";

export const BAND_ORDER: Band[] = ["embed", "early", "crossover", "concept", "dark", "predict", "suppress"];

export function bandOf(layer: number): Band {
  if (layer === 0) return "embed";
  if (layer <= 7) return "early";
  if (layer === 8) return "crossover";
  if (layer <= 12) return "concept";
  if (layer <= 20) return "dark";
  if (layer <= 31) return "predict";
  return "suppress";
}

// ---------- lenses

export interface LensInfo { short: string; name: string; side: "input" | "output"; lens: "S" | "R" | "J" | "logit"; z: boolean; describe: string }

export const LENSES: Record<ReadColumn, LensInfo> = {
  read_S: { short: "in·S", name: "Input tokens", side: "input", lens: "S", z: false, describe: "Input tokens, here or a few positions back, whose presence drives the neuron." },
  read_S_z: { short: "in·S z", name: "Input tokens, specific", side: "input", lens: "S", z: true, describe: "The same, scored against the layer's other neurons so that common tokens drop out." },
  read_R_dclm: { short: "in·R", name: "Input concept", side: "input", lens: "R", z: false, describe: "What the model is getting ready to say when this neuron is driven: the concept it responds to." },
  read_R_dclm_z: { short: "in·R z", name: "Input concept, specific", side: "input", lens: "R", z: true, describe: "The same, scored against the layer's other neurons so that common tokens drop out." },
  write_S: { short: "out·S", name: "Writes like these tokens", side: "output", lens: "S", z: false, describe: "Its push looks to later layers as if these input tokens were present." },
  write_S_z: { short: "out·S z", name: "Writes like these tokens, specific", side: "output", lens: "S", z: true, describe: "The same, scored against the layer's other neurons." },
  write_R_dclm: { short: "out·R", name: "Promotes", side: "output", lens: "R", z: false, describe: "What its push makes the model say." },
  write_R_dclm_z: { short: "out·R z", name: "Promotes, specific", side: "output", lens: "R", z: true, describe: "What's specific to this neuron's push, with tokens every neuron promotes removed." },
  write_J: { short: "out·J", name: "Jacobian lens", side: "output", lens: "J", z: false, describe: "Neuronpedia's Jacobian lens on the write, for comparison. It agrees with out·R from L22." },
  write_logit: { short: "logit", name: "Logit lens", side: "output", lens: "logit", z: false, describe: "The unembedding applied directly, for comparison. It is weak before L22." },
};

/** How to phrase the output side at this depth (the brief: steering before L21, prediction after). */
export function writeVerb(layer: number): string {
  if (layer < 5) return "Writes (mostly noise this early)";
  if (layer < 21) return "When pushed, it promotes";
  if (layer < 32) return "It predicts the next word";
  return "It predicts or suppresses the next word";
}

export function trustText(trust: Trust): string {
  return trust === "primary" ? "Best read at this depth" : trust === "trusted" ? "Informative here" : trust === "compare" ? "For comparison" : "Weak at this depth";
}

// ---------- tokens

export interface TokenParts { lead: boolean; text: string; trail: boolean; byte: boolean; empty: boolean }

/** Splits a token for display: a leading space marks a word start, byte pieces are marked, whitespace is made visible. */
export function tokenParts(raw: string): TokenParts {
  if (raw.startsWith("\u0001")) return { lead: false, text: raw.slice(1), trail: false, byte: true, empty: false };
  const lead = raw.startsWith(" ");
  const trail = raw.length > 1 && raw.endsWith(" ") && raw.trim() !== "";
  let text = raw.trim() === "" ? raw : raw.slice(lead ? 1 : 0, trail ? -1 : undefined);
  text = text.replace(/\n/g, "⏎").replace(/\r/g, "␍").replace(/\t/g, "⇥");
  if (text.trim() === "" && text.length) text = text.replace(/ /g, "␣");
  return { lead, text, trail, byte: false, empty: text === "" };
}

const CJK = /[㐀-鿿豈-﫿]/;
export const hasChinese = (text: string) => CJK.test(text);

/** Variant key, matching the builder's stem(): case, leading punctuation and common English suffixes removed. */
export function stem(token: string): string {
  if (token.startsWith("\u0001")) return token;
  let key = token.trim().toLowerCase().replace(/^[_\-.]+/, "").replace(/^[/([]+/, "");
  for (const suffix of ["ing", "ed", "es", "s", "ly"]) {
    if (key.endsWith(suffix) && key.length - suffix.length >= 4) { key = key.slice(0, -suffix.length); break; }
  }
  return key.endsWith("e") && key.length > 4 ? key.slice(0, -1) : key;
}

export const neuronId = (layer: number, neuron: number) => `L${layer}·n${neuron}`;
export const poleWord = (pole: 1 | -1) => (pole > 0 ? "+ pole" : "− pole");

/** "L8 344", "l8/n344", "8.344", "L8·n344" -> [8, 344]. */
export function parseNeuronRef(query: string): [number, number] | null {
  const match = query.trim().match(/^l?\s*(\d{1,2})\s*[\s./·:,_-]\s*n?\s*(\d{1,5})$/i);
  if (!match) return null;
  const layer = Number(match[1]);
  const neuron = Number(match[2]);
  return layer < 36 && neuron < 12288 ? [layer, neuron] : null;
}

/** Layer colour on the violet → cyan depth ramp, as used in the neuron-lens animation. */
export function depthColor(layer: number): string {
  return `color-mix(in oklch, var(--neuron-deep) ${Math.round(100 - (layer / 35) * 100)}%, var(--neuron-late))`;
}

export function percent(value: number | undefined, digits = 0): string {
  if (value == null || !Number.isFinite(value)) return "–";
  const pct = value * 100;
  if (pct === 0) return "0%";
  if (pct < 0.001) return "<0.001%";
  // Firing rates are often hundredths of a percent: keep two significant figures there.
  if (pct < 1) return `${Number(pct.toPrecision(2))}%`;
  return `${pct.toFixed(digits)}%`;
}

/** The reads to trust at each depth, primary first; mirrors plan() in the catalog builder. */
export function planFor(layer: number): { ins: ReadColumn[]; outs: ReadColumn[] } {
  const ins: ReadColumn[] = layer <= 4 ? ["read_S", "read_S_z"]
    : layer <= 7 ? ["read_S_z", "read_S"]
    : layer === 8 ? ["read_S", "read_S_z", "read_R_dclm", "read_R_dclm_z"]
    : layer <= 12 ? ["read_R_dclm", "read_R_dclm_z"]
    : layer <= 20 ? ["read_R_dclm_z", "read_R_dclm"]
    : ["read_R_dclm", "read_R_dclm_z", "read_S_z"];
  const outs: ReadColumn[] = layer <= 4 ? ["write_S_z", "write_R_dclm_z"]
    : layer <= 8 ? ["write_R_dclm", "write_R_dclm_z", "write_S_z"]
    : ["write_R_dclm", "write_R_dclm_z"];
  return { ins, outs };
}
