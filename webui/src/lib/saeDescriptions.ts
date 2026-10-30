import type { SaeDescriptionSourceJSON } from "./types";

export type SaeDescriptionSource = SaeDescriptionSourceJSON;

export interface SaeDescription {
  label: string | null;
  explanationModel: string | null;
  url: string;
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Original provider params.safetensors hashes, verified against the exact
// Neuronpedia hfRepoId/hfFolderId dictionaries (2026-09-07).
export const PUBLISHED_DICTIONARIES = [
  { size: "270m", layer: 12, sha256: "f99efac8449faa1e5b715e0298f27d5cf3f5646fce5a5ece14ded90a09efeb65" },
  { size: "1b", layer: 13, sha256: "6be6fbfed9850588adb1826bcc9fc89551d9eaf573311e95b21307fd4a281cd5" },
  { size: "4b", layer: 17, sha256: "f67e2305cfd1703d8a685d99fb25b6f25a2feaaa4eb45ef08a467f400b696d71" },
];

export function verifiedSaeDescriptionSource(manifest: Record<string, unknown>): SaeDescriptionSource | null {
  const dictionary = PUBLISHED_DICTIONARIES.find(entry =>
    manifest.corpus_sha256 === entry.sha256 && manifest.layer === entry.layer &&
    manifest.corpus_spec === `provider:google/gemma-scope-2-${entry.size}-it` &&
    ["google", "unsloth"].some(owner => manifest.model_id === `${owner}/gemma-3-${entry.size}-it`));
  if (!dictionary || manifest.d_sae !== 16384 || manifest.activation !== "jump_relu") return null;
  const { size, layer } = dictionary;
  return {
    model: `gemma-3-${size}-it`,
    source: `${layer}-gemmascope-2-res-16k`,
    repository: `google/gemma-scope-2-${size}-it`,
    folder: `resid_post/layer_${layer}_width_16k_l0_medium`,
  };
}

export function parseSaeDescription(value: unknown, binding: SaeDescriptionSource, id: number): SaeDescription {
  const data = value;
  if (!record(data) || data.modelId !== binding.model || data.layer !== binding.source ||
      String(data.index) !== String(id) || !record(data.source) || data.source.hfRepoId !== binding.repository ||
      (binding.folder !== undefined && data.source.hfFolderId !== binding.folder) ||
      (binding.saeId !== undefined && data.source.saelensSaeId !== binding.saeId) ||
      (!binding.folder && !binding.saeId) || !Array.isArray(data.explanations)) {
    throw new Error("The description belongs to a different SAE dictionary.");
  }
  const explanation = data.explanations.find((entry): entry is Record<string, unknown> & { description: string } =>
    record(entry) && typeof entry.description === "string" && Boolean(entry.description.trim()));
  return {
    label: explanation?.description.trim() ?? null,
    explanationModel: typeof explanation?.explanationModelName === "string" ? explanation.explanationModelName : null,
    url: `https://www.neuronpedia.org/${encodeURIComponent(binding.model)}/${encodeURIComponent(binding.source)}/${id}`,
  };
}

const descriptions = new Map<string, SaeDescription>();
type ExplanationIndex = Record<string, [string, string | null]>;
const bundledDescriptions = new Map<string, Promise<ExplanationIndex>>();

function publishedDictionary(binding: SaeDescriptionSource) {
  return PUBLISHED_DICTIONARIES.find(entry =>
    binding.model === `gemma-3-${entry.size}-it` &&
    binding.source === `${entry.layer}-gemmascope-2-res-16k` &&
    binding.repository === `google/gemma-scope-2-${entry.size}-it` &&
    Boolean(binding.folder || binding.saeId) &&
    (binding.folder === undefined || binding.folder === `resid_post/layer_${entry.layer}_width_16k_l0_medium`) &&
    (binding.saeId === undefined || binding.saeId === `layer_${entry.layer}_width_16k_l0_medium`));
}

async function bundledDescription(binding: SaeDescriptionSource, id: number): Promise<SaeDescription | null> {
  const dictionary = publishedDictionary(binding);
  if (!dictionary) return null;
  let loading = bundledDescriptions.get(dictionary.size);
  if (!loading) {
    loading = (async () => {
      const url = new URL(`./data/sae-descriptions/gemma-3-${dictionary.size}-it.json`, import.meta.url);
      const response = await fetch(url.href, {
        signal: AbortSignal.timeout(5000), credentials: "omit", referrerPolicy: "no-referrer",
      });
      if (!response.ok) throw new Error("Bundled SAE descriptions are unavailable.");
      const payload: unknown = await response.json();
      if (!record(payload) || payload.format_version !== 1 || payload.provider_sha256 !== dictionary.sha256 ||
          payload.feature_count !== 16384 || !record(payload.source) ||
          payload.source.model !== binding.model || payload.source.source !== binding.source ||
          payload.source.repository !== binding.repository ||
          payload.source.folder !== `resid_post/layer_${dictionary.layer}_width_16k_l0_medium` ||
          !record(payload.explanations)) throw new Error("Bundled descriptions do not match this SAE.");
      for (const [key, row] of Object.entries(payload.explanations)) {
        if (!/^(0|[1-9]\d*)$/.test(key) || Number(key) >= 16384 || !Array.isArray(row) || row.length !== 2 ||
            typeof row[0] !== "string" || !row[0].trim() || (row[1] !== null && typeof row[1] !== "string")) {
          throw new Error("Bundled SAE descriptions are invalid.");
        }
      }
      return payload.explanations as ExplanationIndex;
    })();
    bundledDescriptions.set(dictionary.size, loading);
  }
  let index: ExplanationIndex;
  try {
    index = await loading;
  } catch {
    if (bundledDescriptions.get(dictionary.size) === loading) bundledDescriptions.delete(dictionary.size);
    return null;
  }
  const row = index[String(id)];
  return row ? {
    label: row[0], explanationModel: row[1],
    url: `https://www.neuronpedia.org/${encodeURIComponent(binding.model)}/${encodeURIComponent(binding.source)}/${id}`,
  } : null;
}

export async function loadSaeDescription(binding: SaeDescriptionSource, id: number, signal: AbortSignal): Promise<SaeDescription> {
  if (!Number.isSafeInteger(id) || id < 0) throw new Error("Invalid SAE feature ID.");
  signal.throwIfAborted();
  const key = JSON.stringify([binding, id]);
  const cached = descriptions.get(key);
  if (cached) return cached;
  let result = await bundledDescription(binding, id);
  signal.throwIfAborted();
  if (!result) {
    const response = await fetch(`https://www.neuronpedia.org/api/feature/${encodeURIComponent(binding.model)}/${encodeURIComponent(binding.source)}/${id}`, {
      signal, credentials: "omit", referrerPolicy: "no-referrer",
    });
    if (!response.ok) throw new Error("Descriptions could not be loaded. Try again when you are online.");
    result = parseSaeDescription(await response.json(), binding, id);
  }
  descriptions.set(key, result);
  if (descriptions.size > 512) descriptions.delete(descriptions.keys().next().value!);
  return result;
}
