// Chinese → English glosses (CC-CEDICT), shared by the hover tooltip and the inline translation toggle.
// Each layer file has its own small glossary, fetched the first time a Chinese token there is needed.
import { loadGlossary, type Glossary } from "./data";

export const gloss = $state<{ dict: Glossary; inline: boolean; loaded: number[] }>({
  // The legend's example token, so its tooltip works before any layer's glossary loads.
  dict: { "圣经": ["shèng jīng", "the Bible; the holy scriptures"] },
  inline: false,
  loaded: [],
});

const requests = new Map<number, Promise<void>>();

export function ensureGlossary(layer: number): Promise<void> {
  let pending = requests.get(layer);
  if (!pending) {
    pending = loadGlossary(layer)
      .then((dict) => {
        gloss.dict = { ...gloss.dict, ...dict };
        gloss.loaded = [...gloss.loaded, layer];
      })
      .catch(() => { requests.delete(layer); });
    requests.set(layer, pending);
  }
  return pending;
}

export const hasGlossary = (layer: number) => gloss.loaded.includes(layer);

/** First sense only, for inline display. Compound glosses ("违法 (illegal) + 行为 (behaviour)") join their parts' English. */
export function shortGloss(entry: [string, string]): string {
  const meaning = entry[1];
  const text = meaning.includes(" + ")
    ? meaning.split(" + ").map((part) => part.match(/\(([^)]*)\)/)?.[1] ?? "").filter(Boolean).join(" ")
    : meaning.split(";")[0].replace(/\s*\(.*?\)/g, "").trim();
  const shown = text || meaning;
  return shown.length > 28 ? `${shown.slice(0, 27)}…` : shown;
}
