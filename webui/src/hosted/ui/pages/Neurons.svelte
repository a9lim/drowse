<script lang="ts">
  import { onMount, tick } from "svelte";
  import TabIdentity from "../../../lib/ui/TabIdentity.svelte";
  import MorphText from "../../../lib/ui/MorphText.svelte";
  import PageHeader from "../PageHeader.svelte";
  import PageFooter from "../PageFooter.svelte";
  import PwaUpdatePrompt from "../PwaUpdatePrompt.svelte";
  import DepthRibbon from "../neurons/DepthRibbon.svelte";
  import LayerStrip from "../neurons/LayerStrip.svelte";
  import NeuronDetail from "../neurons/NeuronDetail.svelte";
  import NeuronSearch from "../neurons/NeuronSearch.svelte";
  import { tokenHover } from "../neurons/tokenHover";
  import { ensureAllGlossary, ensureGlossary, gloss } from "../neurons/gloss.svelte";
  import {
    ALL_CHUNK, asIndexRow, decodeScores, depthColor, loadAllChunk, loadAllIndex, loadAllScores, loadIndex, loadLayer, neuronId,
    type AllChunk, type AllLayerIndex, type CatalogIndex, type IndexRow, type LayerFile, type NeuronDetail as Detail,
  } from "../neurons/data";

  const TOTAL = 442368;
  const format = (value: number) => value.toLocaleString("en-US");
  let index = $state<CatalogIndex | null>(null);
  let failed = $state(false);
  let layer = $state(8);
  /** The neuron named in the URL. Without one, the layer's clearest neuron is shown. */
  let neuron = $state<number | null>(null);
  let layerFile = $state<LayerFile | null>(null);
  let highlight = $state<Set<string>>(new Set());
  let mounted = $state(false);
  let explorer: HTMLElement | undefined = $state();
  let stage: HTMLElement | undefined = $state();
  // Sticky panes sit just under the site header, whose height changes with width and zoom.
  let headerHeight = $state(92);
  // The headline figure counts up once the catalog arrives (Torph morphs each step).
  let readCount = $state(0);
  // "All" shows every one of the 442,368 neurons; the default shows the easiest to read.
  let everyNeuron = $state(false);
  let allScores = $state<Uint8Array[] | null>(null);
  let allIndex = $state<AllLayerIndex | null>(null);
  let allChunk = $state<AllChunk & { chunk: number } | null>(null);

  const curatedRows = $derived((index?.neurons ?? []).filter((row) => row[0] === layer).sort((a, b) => b[4] - a[4]));
  const allRows = $derived<IndexRow[]>(allIndex?.layer === layer ? [...allIndex.neurons].sort((a, b) => b[3] - a[3]).map((row) => asIndexRow(layer, row)) : []);
  const rows = $derived(everyNeuron ? allRows : curatedRows);
  const selected = $derived(neuron ?? rows[0]?.[1] ?? null);
  const fileReady = $derived(everyNeuron ? allChunk?.layer === layer && selected !== null && allChunk.chunk === Math.floor(selected / ALL_CHUNK) : layerFile?.layer === layer);
  const detail = $derived<Detail | null>(
    selected === null || !fileReady ? null
    : everyNeuron ? allChunk!.neurons.find((n) => n.n === selected) ?? null
    : layerFile!.neurons.find((n) => n.n === selected) ?? null,
  );
  const position = $derived(selected === null ? -1 : rows.findIndex((row) => row[1] === selected));

  // ---------- URL state: /neurons?layer=8&neuron=344

  function readUrl(): { layer: number; neuron: number | null; all: boolean } {
    const params = new URLSearchParams(window.location.search);
    const L = Number(params.get("layer"));
    const n = params.get("neuron");
    return {
      layer: params.has("layer") && Number.isInteger(L) && L >= 0 && L < 36 ? L : 8,
      neuron: n !== null && /^\d{1,5}$/.test(n) && Number(n) < 12288 ? Number(n) : null,
      all: params.get("all") === "1",
    };
  }

  function urlFor(L: number, n: number | null): string {
    const params = new URLSearchParams({ layer: String(L) });
    if (n !== null) params.set("neuron", String(n));
    if (everyNeuron) params.set("all", "1");
    return `/neurons?${params}`;
  }

  function writeUrl(push: boolean) {
    const url = urlFor(layer, neuron);
    if (url === window.location.pathname + window.location.search) return;
    history[push ? "pushState" : "replaceState"]({ neurons: true }, "", url);
  }

  async function ensureLayer(L: number) {
    if (gloss.inline) void ensureGlossary(L);
    if (everyNeuron) return ensureAll(L);
    if (layerFile?.layer === L) return;
    failed = false;
    try {
      const file = await loadLayer(L);
      if (layer === L) layerFile = file;
    } catch {
      if (layer === L) failed = true;
    }
  }

  async function ensureAll(L: number) {
    failed = false;
    try {
      if (!allScores) loadAllScores().then((file) => { allScores = file.layers.map(decodeScores); }).catch(() => undefined);
      if (allIndex?.layer !== L) {
        const file = await loadAllIndex(L);
        if (layer === L) allIndex = file;
      }
      const n = neuron ?? allRows[0]?.[1];
      if (n === undefined || n === null) return;
      const chunk = Math.floor(n / ALL_CHUNK);
      if (allChunk?.layer === L && allChunk.chunk === chunk) return;
      const file = await loadAllChunk(L, chunk);
      if (layer === L) allChunk = { ...file, chunk };
    } catch {
      if (layer === L) failed = true;
    }
  }

  // The detail's chunk follows the selection (stepping can cross a 256-neuron boundary).
  $effect(() => {
    if (!everyNeuron || selected === null) return;
    if (allChunk?.layer === layer && allChunk.chunk === Math.floor(selected / ALL_CHUNK)) return;
    void ensureAll(layer);
  });

  // The explorer's hover tip in All mode reads from per-layer indexes, loaded on first hover.
  let peekCache = $state<Record<number, IndexRow[]>>({});
  const peekRequested = new Set<number>();
  function peek(L: number, n: number): IndexRow | null {
    const rows = peekCache[L];
    if (rows) return rows[n] ?? null;
    if (!peekRequested.has(L)) {
      peekRequested.add(L);
      loadAllIndex(L).then((file) => {
        const byNeuron: IndexRow[] = [];
        for (const row of file.neurons) byNeuron[row[0]] = asIndexRow(L, row);
        peekCache = { ...peekCache, [L]: byNeuron };
      }).catch(() => peekRequested.delete(L));
    }
    return null;
  }

  function setEveryNeuron(on: boolean) {
    if (everyNeuron === on) return;
    everyNeuron = on;
    writeUrl(true);
    if (on && gloss.inline) void ensureAllGlossary();
    void ensureLayer(layer);
  }

  /** Scrubbing replaces the history entry; a deliberate choice adds one. */
  function setLayer(L: number, commit: boolean) {
    if (L === layer && neuron === null) return;
    layer = L;
    neuron = null;
    writeUrl(commit);
    void ensureLayer(L);
  }

  async function openNeuron(L: number, n: number, reveal = false, focus = true) {
    layer = L;
    neuron = n;
    writeUrl(true);
    void ensureLayer(L);
    await tick();
    // Move focus to the neuron, so the next Tab continues in it rather than back in the list.
    if (focus) document.getElementById("neuron-title")?.focus({ preventScroll: true });
    const target = reveal ? explorer : stage;
    if (target) {
      const top = target.getBoundingClientRect().top;
      if (top < 0 || top > window.innerHeight * 0.7) target.scrollIntoView({ block: "start", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    }
  }

  function step(delta: -1 | 1) {
    if (!rows.length || position < 0) return;
    neuron = rows[(position + delta + rows.length) % rows.length][1];
    writeUrl(false);
  }

  const stepHref = (delta: -1 | 1) => rows.length && position >= 0
    ? urlFor(layer, rows[(position + delta + rows.length) % rows.length][1])
    : urlFor(layer, selected);

  function pageKeys(event: KeyboardEvent) {
    const target = event.target as HTMLElement | null;
    if (event.metaKey || event.ctrlKey || event.altKey || target?.closest("input, textarea, select, [role='slider']")) return;
    if (event.key === "j" || event.key === "k") { event.preventDefault(); step(event.key === "j" ? 1 : -1); }
  }

  function countUp() {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { readCount = TOTAL; return; }
    const steps = [0.02, 0.09, 0.22, 0.41, 0.63, 0.82, 0.94, 1];
    steps.forEach((share, i) => setTimeout(() => (readCount = Math.round(TOTAL * share)), 160 * (i + 1)));
  }

  onMount(() => {
    mounted = true;
    const start = readUrl();
    layer = start.layer;
    neuron = start.neuron;
    everyNeuron = start.all;
    loadIndex().then((loaded) => { index = loaded; countUp(); }).catch(() => { failed = true; });
    void ensureLayer(layer);
    const pop = () => {
      const state = readUrl();
      layer = state.layer;
      neuron = state.neuron;
      everyNeuron = state.all;
      void ensureLayer(state.layer);
    };
    window.addEventListener("popstate", pop);
    document.addEventListener("keydown", pageKeys);
    const header = document.querySelector<HTMLElement>(".page-header");
    const resize = header ? new ResizeObserver(() => (headerHeight = header.offsetHeight)) : null;
    if (header) resize!.observe(header);
    return () => {
      resize?.disconnect();
      window.removeEventListener("popstate", pop);
      document.removeEventListener("keydown", pageKeys);
    };
  });

  const pageTitle = $derived(neuron !== null ? `${neuronId(layer, neuron)} · Neuronlens Explorer · Drowse` : "Neuronlens Explorer · Drowse");
</script>

<TabIdentity title={pageTitle} />
{#if mounted}<PwaUpdatePrompt />{/if}

<div class="neurons-page" style:--neuron-tint={depthColor(layer)} style:--neuron-sticky-top={`${headerHeight + 8}px`}>
  <a class="skip-link" href="#explore">Skip to the explorer</a>
  <PageHeader current="neurons" />
  <main id="neurons-main" data-layer={layer} use:tokenHover>
    <section class="intro" aria-labelledby="neurons-title">
      <h1 id="neurons-title">Neuronlens Explorer</h1>
      <p class="lede">See what the neurons inside Qwen3-8B respond to and what they make the model say.</p>
      <dl class="figures">
        <div><dt>Neurons read</dt><dd><MorphText text={format(readCount)} duration={200} /></dd></div>
        <div><dt>Layers</dt><dd>36</dd></div>
        <div><dt>Easiest to read</dt><dd>{index ? format(index.neurons.length) : "–"}</dd></div>
      </dl>
      <NeuronSearch {index} anyNeuron={everyNeuron} onopen={(L, n) => void openNeuron(L, n, true)} onmatches={(keys) => (highlight = keys)} examples={["owl", "Bible", "money", "rain", "圣经"]} />
    </section>

    <section class="explorer" id="explore" aria-label="Neuron explorer" bind:this={explorer} tabindex="-1">
      <div class="panel ribbon-panel">
        <div class="ribbon-bar">
          <div class="mode" role="group" aria-label="Neurons shown">
            <button type="button" aria-pressed={!everyNeuron} onclick={() => setEveryNeuron(false)}>Easiest <span>{index ? format(index.neurons.length) : ""}</span></button>
            <button type="button" aria-pressed={everyNeuron} onclick={() => setEveryNeuron(true)}>All <span>{format(TOTAL)}</span></button>
          </div>
          <label class="translate">
            <input type="checkbox" bind:checked={gloss.inline} onchange={() => { if (gloss.inline) { void ensureGlossary(layer); if (everyNeuron) void ensureAllGlossary(); } }} />
            <span class="switch" aria-hidden="true"></span>
            <span>Translate Chinese</span>
          </label>
        </div>
        <DepthRibbon {index} {layer} neuron={selected} {highlight} scores={everyNeuron ? allScores : null} {peek} onlayer={setLayer} onneuron={(L, n) => void openNeuron(L, n)} />
        <p class="ribbon-caption">Early layers <span aria-hidden="true">→</span> late layers{everyNeuron ? " · brighter cells are easier to read" : ""}</p>
      </div>

      {#if failed && !index}
        <p class="panel notice" role="alert">Unable to load the neuron catalog. Check your connection and reload the page.</p>
      {:else}
        <LayerStrip {layer} {rows} {selected} onneuron={(n) => void openNeuron(layer, n)} hrefFor={(n) => urlFor(layer, n)} />

        <div class="panel stage" bind:this={stage}>
          {#if detail}
            <NeuronDetail neuron={detail} {layer} onstep={step} position={Math.max(0, position)} total={rows.length} hrefFor={stepHref} />
          {:else if failed}
            <div class="notice" role="alert">
              <p>Unable to load layer {layer}. Check your connection and try again.</p>
              <button type="button" class="text-button" onclick={() => void ensureLayer(layer)}>Try again</button>
            </div>
          {:else if fileReady && neuron !== null}
            <div class="notice">
              <p>{neuronId(layer, neuron)} isn't one of the neurons shown for this layer.</p>
              <button type="button" class="text-button" onclick={() => setLayer(layer, true)}>Show the easiest neuron in this layer</button>
            </div>
          {:else}
            <div class="skeleton" aria-live="polite" aria-busy="true">
              <span class="sr-only">Loading layer {layer}…</span>
              <span class="bone wide"></span><span class="bone"></span><span class="bone"></span>
              <span class="bone block"></span>
            </div>
          {/if}
        </div>
      {/if}
    </section>
  </main>
  <PageFooter current="neurons" />
</div>

<style>
  .neurons-page {
    --neuron-tint-ink: color-mix(in oklch, var(--neuron-tint) 62%, var(--fg));
    --neuron-translated-ink: var(--accent-amber);
    /* Surfaces: hairline-edged panels in dark; white cards on soft shadows in light. */
    --neuron-panel: var(--bg-alt);
    --neuron-panel-edge: inset 0 0 0 1px var(--glass-line);
    --neuron-panel-lift: var(--shadow-card);
    --neuron-chip-bg: var(--glass);
    --neuron-well: var(--bg);
    display: flex; flex-direction: column; min-height: 100svh; overflow-x: clip; background: var(--bg); color: var(--fg);
    -webkit-font-smoothing: antialiased;
  }
  :global(:root[data-theme="light"]) .neurons-page {
    --neuron-tint-ink: color-mix(in oklch, var(--neuron-tint) 65%, var(--fg));
    --neuron-panel-edge: var(--shadow-card);
    --neuron-panel-lift: var(--shadow-card-active);
    --neuron-chip-bg: var(--bg-alt);
    --neuron-well: var(--bg);
  }
  .skip-link { position: fixed; inset: 8px auto auto 8px; z-index: 10; padding: var(--space-sm); background: var(--accent); color: var(--text-on-accent); border-radius: var(--radius); text-decoration: none; transform: translateY(-160%); }
  .skip-link:focus { transform: none; }
  /* Flat panels, as in the workbench: no sheen gradients on this page's content. */
  main {
    --surface-sheen: none;
    --control-sheen: none;
    flex: 1; display: grid; gap: var(--space-xl); width: min(100%, var(--page-max)); margin-inline: auto;
    padding-inline: var(--page-gutter); padding-block-end: var(--section-space);
  }
  .panel { border-radius: var(--popup-radius); background: var(--neuron-panel); box-shadow: var(--neuron-panel-edge); }

  .intro { display: grid; justify-items: start; gap: var(--space-5); padding-block-start: var(--space-2xl); }
  h1 { margin: 0; color: var(--fg); font-family: var(--font-structure); font-weight: var(--weight-display); font-size: var(--text-explorer-title); line-height: 1.02; letter-spacing: -0.045em; text-wrap: balance; animation: rise-in 600ms var(--ease-enter) backwards; }
  .lede { max-width: 48ch; margin: 0; color: var(--fg-dim); font-size: var(--text-lg); line-height: 1.5; text-wrap: pretty; animation: rise-in 600ms var(--ease-enter) 80ms backwards; }
  @keyframes rise-in { from { opacity: 0; translate: 0 10px; } }
  .figures { display: flex; flex-wrap: wrap; gap: var(--space-4) var(--space-xl); margin: 0; animation: rise-in 600ms var(--ease-enter) 160ms backwards; }
  .figures div { display: grid; gap: 0; }
  .figures dt { color: var(--fg-muted); font-size: var(--text-sm); }
  .figures dd { margin: 0; color: var(--fg); font-family: var(--font-structure); font-size: var(--text-feature-title); font-weight: var(--weight-display); font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }

  .explorer { display: grid; gap: var(--space-6); min-width: 0; scroll-margin-top: var(--neuron-sticky-top); }
  .explorer:focus { outline: none; }
  .ribbon-panel { display: grid; gap: var(--space-4); padding: var(--space-5) var(--space-6) var(--space-4); }
  .ribbon-bar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--space-3); }
  .ribbon-caption { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); }
  .mode { display: inline-flex; gap: 0; padding: 0; border-radius: var(--radius-pill); background: var(--neuron-chip-bg); box-shadow: inset 0 0 0 1px var(--glass-line); }
  .mode button { display: inline-flex; align-items: center; gap: var(--space-2); min-height: var(--control-target); padding-inline: var(--space-5); border: 0; border-radius: var(--radius-pill); background: transparent; color: var(--fg-dim); font: inherit; font-size: var(--text-sm); font-weight: var(--weight-structure); cursor: pointer; transition: background-color var(--dur) var(--ease-out), color var(--dur) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
  .mode button span { color: var(--fg-muted); font-family: var(--font-mono); font-size: var(--text-xs); font-variant-numeric: tabular-nums; }
  .mode button[aria-pressed="true"] { background: var(--fg); color: var(--bg); }
  .mode button[aria-pressed="true"] span { color: color-mix(in srgb, var(--bg) 70%, var(--fg)); }
  .mode button:active { scale: var(--press-scale); }
  .mode button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .translate { position: relative; display: inline-flex; align-items: center; gap: var(--space-3); min-height: var(--control-target); color: var(--fg-dim); font-size: var(--text-sm); cursor: pointer; }
  .translate input { position: absolute; opacity: 0; width: 1px; height: 1px; }
  .switch { position: relative; flex: none; width: 36px; height: 20px; border-radius: var(--radius-pill); background: var(--glass-strong); box-shadow: inset 0 0 0 1px var(--glass-line); transition: background-color var(--dur) var(--ease-out); }
  .switch::after { content: ""; position: absolute; top: 3px; left: 3px; width: 14px; height: 14px; border-radius: 50%; background: var(--fg-dim); transition: translate var(--dur) var(--ease-out), background-color var(--dur) var(--ease-out); }
  .translate input:checked + .switch { background: color-mix(in srgb, var(--accent-amber) 60%, transparent); }
  .translate input:checked + .switch::after { translate: 16px 0; background: var(--fg); }
  .translate input:focus-visible + .switch { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .stage { min-width: 0; padding: var(--space-6) var(--space-xl); scroll-margin-top: var(--neuron-sticky-top); }
  .notice { display: grid; justify-items: start; gap: var(--space-3); margin: 0; padding: var(--space-5); color: var(--fg-dim); font-size: var(--text-sm); }
  .notice p { margin: 0; }
  .text-button { min-height: var(--control-target); padding-inline: var(--space-4); border: 0; border-radius: var(--radius); background: var(--glass-strong); color: var(--fg); font: inherit; cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
  .text-button:active { scale: var(--press-scale); }
  .text-button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (hover: hover) { .text-button:hover { background: var(--glass-bright); } }
  .skeleton { display: grid; gap: var(--space-3); }
  .bone { display: block; height: 14px; width: 40%; border-radius: var(--radius-sm); background: var(--glass-strong); animation: pulse 1.4s ease-in-out infinite; }
  .bone.wide { width: 24%; height: 32px; }
  .bone.block { width: 100%; height: 240px; border-radius: var(--radius-lg); }
  @keyframes pulse { 50% { opacity: 0.5; } }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }

  @media (max-width: 760px) {
    .ribbon-panel { padding: var(--space-4) var(--space-3) var(--space-3); }
    .stage { padding: var(--space-5) var(--space-4); }
  }
  @media (prefers-reduced-motion: reduce) {
    h1, .lede, .figures, .bone { animation: none; }
    .switch, .switch::after, .text-button { transition: none; }
    .text-button:active { scale: none; }
  }

  :global(.zh-gloss) {
    position: absolute; left: 0; top: 0; z-index: 60; display: grid; gap: 0; width: max-content;
    padding: var(--space-3) var(--space-4); border-radius: var(--radius); background: var(--bg-elev); box-shadow: var(--shadow-card);
    color: var(--fg); font-size: var(--text-sm); line-height: 1.4;
    opacity: 0; translate: 0 3px; transition: opacity var(--dur-fast) var(--ease-out), translate var(--dur-fast) var(--ease-out);
  }
  :global(.zh-gloss[hidden]) { display: none; }
  :global(.zh-gloss.shown) { opacity: 1; translate: 0 0; }
  :global(.zh-gloss .zh-word) { font-size: var(--text-md); font-weight: var(--weight-structure); }
  :global(.zh-gloss .zh-pinyin) { color: var(--fg-dim); font-family: var(--font-mono); font-size: var(--text-xs); }
  :global(.zh-gloss .zh-meaning) { color: var(--fg-dim); }
  :global(.zh-gloss .muted) { color: var(--fg-muted); font-size: var(--text-xs); }
  @media (prefers-reduced-motion: reduce) { :global(.zh-gloss) { transition: none; } }
</style>
