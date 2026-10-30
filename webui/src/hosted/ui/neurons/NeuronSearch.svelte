<script lang="ts">
  import FluentIcon from "../../../lib/ui/FluentIcon.svelte";
  import MorphText from "../../../lib/ui/MorphText.svelte";
  import { loadSearch, neuronId, parseNeuronRef, stem, type CatalogIndex, type SearchFile } from "./data";

  interface Result { layer: number; neuron: number; side: 0 | 1; title: string; matched: string; score: number }

  let { index, onopen, onmatches, examples = [] }: {
    index: CatalogIndex | null;
    onopen: (layer: number, neuron: number, pole: 1 | -1) => void;
    /** Every matching neuron, as "layer:neuron" keys, so the ribbon can light them up. */
    onmatches: (keys: Set<string>, query: string) => void;
    examples?: string[];
  } = $props();

  let query = $state("");
  let results = $state<Result[]>([]);
  let total = $state(0);
  let active = $state(-1);
  let open = $state(false);
  let loading = $state(false);
  let searchFailed = $state(false);
  let data: SearchFile | null = null;
  let input: HTMLInputElement | undefined = $state();
  let timer: ReturnType<typeof setTimeout> | undefined;

  async function ensure(): Promise<SearchFile | null> {
    if (data) return data;
    loading = true;
    searchFailed = false;
    try { data = await loadSearch(); } catch { data = null; searchFailed = true; } finally { loading = false; }
    return data;
  }

  function run(text: string) {
    const trimmed = text.trim();
    if (!trimmed || !index) { results = []; total = 0; onmatches(new Set(), ""); return; }
    const ref = parseNeuronRef(trimmed);
    if (ref) {
      const row = index.neurons.find((r) => r[0] === ref[0] && r[1] === ref[1]);
      results = row ? [{ layer: row[0], neuron: row[1], side: 0, title: row[2], matched: "", score: 1 }] : [];
      total = results.length;
      onmatches(new Set(results.map((r) => `${r.layer}:${r.neuron}`)), trimmed);
      active = results.length ? 0 : -1;
      return;
    }
    if (!data) {
      if (searchFailed) return;
      void ensure().then(() => { if (data) run(query); });
      return;
    }
    const key = stem(trimmed);
    const lower = trimmed.toLowerCase();
    const scores = new Map<number, { score: number; matched: string; side: 0 | 1 }>();
    for (const [form, refs] of Object.entries(data.forms)) {
      let factor = 0;
      if (form === key || form === lower) factor = 3;
      else if (key.length >= 2 && form.startsWith(key)) factor = 1.4;
      else if (key.length >= 3 && form.includes(key)) factor = 0.7;
      else if (/[㐀-鿿]/.test(lower) && form.includes(lower)) factor = 1.2;
      if (!factor) continue;
      for (const [row, side, weight] of refs) {
        const id = row * 2 + side;
        const value = weight * factor;
        const previous = scores.get(id);
        if (!previous) scores.set(id, { score: value, matched: data.display[form] ?? form, side });
        else { previous.score += value * 0.35; if (value > previous.score) previous.matched = data.display[form] ?? form; }
      }
    }
    const ranked = [...scores.entries()].sort((a, b) => b[1].score - a[1].score);
    const keys = new Set<string>();
    for (const [id] of ranked) { const row = index.neurons[id >> 1]; keys.add(`${row[0]}:${row[1]}`); }
    total = keys.size;
    const seen = new Set<string>();
    const list: Result[] = [];
    for (const [id, hit] of ranked) {
      const row = index.neurons[id >> 1];
      const k = `${row[0]}:${row[1]}`;
      if (seen.has(k)) continue;
      seen.add(k);
      list.push({ layer: row[0], neuron: row[1], side: hit.side, title: hit.side === 0 ? row[2] : row[3], matched: hit.matched, score: hit.score });
      if (list.length >= 12) break;
    }
    results = list;
    active = list.length ? 0 : -1;
    onmatches(keys, trimmed);
  }

  function changed() {
    open = true;
    clearTimeout(timer);
    timer = setTimeout(() => run(query), 70);
  }

  function choose(result: Result) {
    open = false;
    onopen(result.layer, result.neuron, result.side === 0 ? 1 : -1);
  }

  function key(event: KeyboardEvent) {
    if (event.key === "ArrowDown") { event.preventDefault(); open = true; active = Math.min(results.length - 1, active + 1); }
    else if (event.key === "ArrowUp") { event.preventDefault(); active = Math.max(0, active - 1); }
    else if (event.key === "Enter" && active >= 0 && results[active]) { event.preventDefault(); choose(results[active]); }
    else if (event.key === "Escape") {
      if (open && results.length) open = false;
      else { query = ""; run(""); }
    }
  }

  function tryExample(text: string) {
    query = text;
    open = true;
    input?.focus();
    void ensure().then(() => run(text));
  }

  // "/" focuses search from anywhere on the page, as on most explorers.
  $effect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "/" || event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable='true']")) return;
      event.preventDefault();
      input?.focus();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  });

  const status = $derived(
    !query.trim() ? ""
    : searchFailed ? "Unable to load search. Check your connection and try again."
    : loading ? "Loading search…"
    : !results.length ? `No neurons match “${query.trim()}”. Try another word, or an ID like L8 344.`
    : total === 1 ? "1 neuron matches."
    : `${total} neurons match. Showing the top ${results.length}.`,
  );
</script>

<div class="search" role="search">
  <div class="field" class:open={open && results.length > 0}>
    <FluentIcon name="search" size={20} />
    <input
      bind:this={input}
      bind:value={query}
      type="search"
      role="combobox"
      aria-label="Search neurons by concept or ID"
      aria-autocomplete="list"
      aria-expanded={open && results.length > 0}
      aria-controls="neuron-results"
      aria-activedescendant={open && active >= 0 ? `neuron-result-${active}` : undefined}
      placeholder="owl, 圣经, or L8 344"
      autocomplete="off"
      spellcheck="false"
      oninput={changed}
      onfocus={() => { open = true; searchFailed = false; void ensure().then(() => { if (data && query.trim()) run(query); }); }}
      onblur={() => setTimeout(() => (open = false), 120)}
      onkeydown={key}
    />
    <kbd aria-hidden="true">/</kbd>
  </div>
  <p class="status" aria-live="polite">{#if status}<MorphText text={status} duration={200} />{/if}</p>
  {#if open && results.length}
    <ul class="results" id="neuron-results" role="listbox" aria-label="Matching neurons">
      {#each results as result, i (`${result.layer}:${result.neuron}`)}
        <li
          id={`neuron-result-${i}`}
          role="option"
          aria-selected={i === active}
          class:active={i === active}
          onpointerdown={(event) => { event.preventDefault(); choose(result); }}
          onpointerenter={() => (active = i)}
          style:--L={result.layer}
        >
          <span class="r-id">{neuronId(result.layer, result.neuron)}</span>
          <span class="r-sign">{result.side === 0 ? "+" : "−"}</span>
          <span class="r-title">{result.title || "unclear"}</span>
          {#if result.matched}<span class="r-match">{result.matched}</span>{/if}
        </li>
      {/each}
    </ul>
  {/if}
  {#if examples.length && !query}
    <p class="examples"><span>Try</span>{#each examples as example}<button type="button" onclick={() => tryExample(example)}>{example}</button>{/each}</p>
  {/if}
</div>

<style>
  .search { position: relative; display: grid; gap: var(--space-3); width: min(100%, 40rem); }
  .field {
    display: flex; align-items: center; gap: var(--space-3); min-height: 56px; padding-inline: var(--space-5);
    border-radius: var(--radius-lg); background: var(--input-well, var(--glass)); box-shadow: var(--shadow-well, inset 0 0 0 1px var(--glass-line));
    color: var(--fg-muted); transition: box-shadow var(--dur-fast) var(--ease-out);
  }
  .field:focus-within { box-shadow: var(--shadow-well, none), 0 0 0 2px var(--focus-ring); color: var(--fg); }
  input { flex: 1; min-width: 0; height: 54px; border: 0; background: transparent; color: var(--fg); font: inherit; font-size: var(--text-md); outline: none; }
  input::placeholder { color: var(--fg-muted); }
  input::-webkit-search-cancel-button { filter: grayscale(1); }
  kbd { display: grid; place-items: center; min-width: 24px; height: 24px; border-radius: var(--radius-sm); box-shadow: inset 0 0 0 1px var(--glass-line); color: var(--fg-muted); font-family: var(--font-mono); font-size: var(--text-xs); }
  .field:focus-within kbd { display: none; }
  .status { min-height: 1.2em; margin: 0; color: var(--fg-muted); font-size: var(--text-xs); }
  /* Stays rendered while empty, so screen readers announce its first message. */
  .status:empty { min-height: 0; margin-block-end: calc(var(--space-3) * -1); }
  .results {
    position: absolute; inset: 60px 0 auto; z-index: 6; display: grid; max-height: min(26rem, 60vh); margin: 0; padding: var(--space-2); overflow-y: auto;
    border-radius: var(--radius-lg); background: var(--bg-elev); box-shadow: var(--shadow-card); list-style: none;
    animation: drop var(--dur) var(--ease-enter) backwards; transform-origin: top;
  }
  @keyframes drop { from { opacity: 0; transform: translateY(-4px) scale(0.99); } }
  .results li { display: grid; grid-template-columns: 5.5rem 1.5ch minmax(0, 1fr) auto; align-items: center; gap: var(--space-3); min-height: 44px; padding: var(--space-2) var(--space-3); border-radius: var(--radius); cursor: pointer; }
  .results li.active { background: var(--bg-hover); }
  .r-id { color: var(--neuron-tint-ink); font-family: var(--font-mono); font-size: var(--text-xs); font-variant-numeric: tabular-nums; }
  .r-sign { color: var(--fg-muted); font-weight: var(--weight-display); }
  .r-title { overflow: hidden; color: var(--fg); font-size: var(--text-sm); text-overflow: ellipsis; white-space: nowrap; }
  .r-match { max-width: 10ch; overflow: hidden; padding: 0 var(--space-xs); border-radius: var(--radius-sm); background: var(--glass-strong); color: var(--fg-dim); font-family: var(--font-mono); font-size: var(--text-xs); text-overflow: ellipsis; white-space: nowrap; }
  .examples { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2); margin: 0; color: var(--fg-muted); font-size: var(--text-sm); }
  .examples span { margin-inline-end: var(--space-1); }
  .examples button { min-height: var(--control-target); padding-inline: var(--space-4); border: 0; border-radius: var(--radius-pill); background: var(--glass); box-shadow: inset 0 0 0 1px var(--glass-line); color: var(--fg-dim); font: inherit; font-size: var(--text-sm); cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out); }
  .examples button:hover { background: var(--glass-strong); color: var(--fg); }
  .examples button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (max-width: 520px) { .results li { grid-template-columns: 5.5rem 1.5ch minmax(0, 1fr); } .r-match { display: none; } }
  @media (prefers-reduced-motion: reduce) { .results { animation: none; } .field, .examples button { transition: none; } }
</style>
