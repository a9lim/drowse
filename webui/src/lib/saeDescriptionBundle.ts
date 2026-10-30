// Location of the published-description index for one of the browser packs'
// Gemma Scope 2 dictionaries.  The local dashboard build swaps this module
// for ``saeDescriptionBundle.none.ts`` (see vite.config.ts) so the wheel does
// not carry the ~3 MB of indexes; lookups there go to Neuronpedia instead.
export function bundledDescriptionUrl(size: string): string | null {
  return new URL(`./data/sae-descriptions/gemma-3-${size}-it.json`, import.meta.url).href;
}
