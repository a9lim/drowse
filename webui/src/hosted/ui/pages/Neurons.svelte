<script lang="ts">
  import { onMount, tick } from "svelte";
  import TabIdentity from "../../../lib/ui/TabIdentity.svelte";
  import MorphText from "../../../lib/ui/MorphText.svelte";
  import PageHeader from "../PageHeader.svelte";
  import PageFooter from "../PageFooter.svelte";
  import PwaUpdatePrompt from "../PwaUpdatePrompt.svelte";
  import DepthRibbon from "../neurons/DepthRibbon.svelte";
  import LayerPanel from "../neurons/LayerPanel.svelte";
  import NeuronDetail from "../neurons/NeuronDetail.svelte";
  import NeuronSearch from "../neurons/NeuronSearch.svelte";
  import LensMap from "../neurons/LensMap.svelte";
  import Token from "../neurons/Token.svelte";
  import TitleText from "../neurons/TitleText.svelte";
  import { tokenHover } from "../neurons/tokenHover";
  import { ensureGlossary, gloss } from "../neurons/gloss.svelte";
  import {
    BANDS, BAND_ORDER, FLAG_LABELLED, bandOf, depthColor, loadIndex, loadLayer, neuronId,
    type CatalogIndex, type LayerFile, type NeuronDetail as Detail,
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

  const shownCount = $derived(index?.neurons.length ?? 0);
  const rows = $derived((index?.neurons ?? []).filter((row) => row[0] === layer).sort((a, b) => b[4] - a[4]));
  const meta = $derived(index?.layerMeta.find((m) => m.layer === layer));
  const selected = $derived(neuron ?? rows[0]?.[1] ?? null);
  const fileReady = $derived(layerFile?.layer === layer);
  const detail = $derived<Detail | null>(selected === null || !fileReady ? null : layerFile!.neurons.find((n) => n.n === selected) ?? null);
  const position = $derived(selected === null ? -1 : rows.findIndex((row) => row[1] === selected));
  const featured = $derived.by(() => {
    if (!index) return [];
    return BAND_ORDER.map((band) => {
      const inBand = index!.neurons.filter((row) => bandOf(row[0]) === band).sort((a, b) => b[4] - a[4]);
      return inBand.find((row) => row[5] & FLAG_LABELLED) ?? inBand[0];
    }).filter(Boolean);
  });

  // ---------- URL state: /neurons?layer=8&neuron=344

  function readUrl(): { layer: number; neuron: number | null } {
    const params = new URLSearchParams(window.location.search);
    const L = Number(params.get("layer"));
    const n = params.get("neuron");
    return {
      layer: params.has("layer") && Number.isInteger(L) && L >= 0 && L < 36 ? L : 8,
      neuron: n !== null && /^\d{1,5}$/.test(n) ? Number(n) : null,
    };
  }

  function urlFor(L: number, n: number | null): string {
    const params = new URLSearchParams({ layer: String(L) });
    if (n !== null) params.set("neuron", String(n));
    return `/neurons?${params}`;
  }

  function writeUrl(push: boolean) {
    const url = urlFor(layer, neuron);
    if (url === window.location.pathname + window.location.search) return;
    history[push ? "pushState" : "replaceState"]({ neurons: true }, "", url);
  }

  async function ensureLayer(L: number) {
    if (gloss.inline) void ensureGlossary(L);
    if (layerFile?.layer === L) return;
    failed = false;
    try {
      const file = await loadLayer(L);
      if (layer === L) layerFile = file;
    } catch {
      if (layer === L) failed = true;
    }
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
    // On one-column layouts the detail sits below the list, so bring it into view.
    const narrow = window.matchMedia("(max-width: 1000px)").matches;
    const target = narrow ? stage : reveal ? explorer : null;
    if (target) {
      const top = target.getBoundingClientRect().top;
      if (top < 0 || top > window.innerHeight * 0.5) target.scrollIntoView({ block: "start", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
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

  onMount(() => {
    mounted = true;
    const start = readUrl();
    layer = start.layer;
    neuron = start.neuron;
    loadIndex().then((loaded) => { index = loaded; }).catch(() => { failed = true; });
    void ensureLayer(layer);
    const pop = () => {
      const state = readUrl();
      layer = state.layer;
      neuron = state.neuron;
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

  const pageTitle = $derived(neuron !== null ? `${neuronId(layer, neuron)} · Neurons · Drowse` : "Neurons of Qwen3-8B · Drowse");

</script>

<TabIdentity title={pageTitle} />
{#if mounted}<PwaUpdatePrompt />{/if}

<div class="neurons-page" style:--neuron-tint={depthColor(layer)} style:--neuron-sticky-top={`${headerHeight + 8}px`}>
  <a class="skip-link" href="#explore">Skip to the explorer</a>
  <PageHeader current="neurons" />
  <main id="neurons-main" data-layer={layer} use:tokenHover>
    <section class="intro" aria-labelledby="neurons-title">
      <p class="eyebrow">Qwen3-8B · neuron-lens</p>
      <h1 id="neurons-title">Every neuron, read.</h1>
      <p class="lede">The {shownCount ? format(shownCount) : "1,490"} clearest of Qwen3-8B's {format(TOTAL)} MLP neurons, read through four lenses.</p>
      <NeuronSearch {index} onopen={(L, n) => void openNeuron(L, n, true)} onmatches={(keys) => (highlight = keys)} examples={["owl", "Bible", "money", "rain", "圣经", "L8 344"]} />
      <dl class="figures">
        <div><dt>Layers</dt><dd>36</dd></div>
        <div><dt>Neurons read</dt><dd>{format(TOTAL)}</dd></div>
        <div><dt>Poles</dt><dd>{format(TOTAL * 2)}</dd></div>
        <div><dt>Shown here</dt><dd><MorphText text={shownCount ? format(shownCount) : "–"} duration={420} /></dd></div>
      </dl>
    </section>

    <section class="explorer" id="explore" aria-labelledby="explore-title" bind:this={explorer} tabindex="-1">
      <div class="explorer-bar">
        <h2 id="explore-title">Explore by depth</h2>
        <label class="translate">
          <input type="checkbox" bind:checked={gloss.inline} onchange={() => { if (gloss.inline) void ensureGlossary(layer); }} />
          <span class="switch" aria-hidden="true"></span>
          <span>Translate Chinese tokens</span>
        </label>
      </div>

      <div class="panel ribbon-panel">
        <DepthRibbon {index} {layer} neuron={selected} {highlight} onlayer={setLayer} onneuron={(L, n) => void openNeuron(L, n)} />
      </div>

      {#if failed && !index}
        <p class="panel notice" role="alert">Unable to load the neuron catalog. Check your connection and reload the page.</p>
      {:else}
        <div class="workspace">
          <aside class="panel rail-panel" aria-label={`Layer ${layer}`}>
            <LayerPanel {layer} {meta} {rows} {selected} onneuron={(n) => void openNeuron(layer, n)} hrefFor={(n) => urlFor(layer, n)} />
          </aside>
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
                <p>{neuronId(layer, neuron)} isn't one of the {rows.length} neurons shown for this layer.</p>
                <button type="button" class="text-button" onclick={() => setLayer(layer, true)}>Show the layer's clearest neuron</button>
              </div>
            {:else}
              <div class="skeleton" aria-live="polite" aria-busy="true">
                <span class="sr-only">Loading layer {layer}…</span>
                <span class="bone wide"></span><span class="bone"></span><span class="bone"></span>
                <span class="bone block"></span><span class="bone block"></span>
              </div>
            {/if}
          </div>
        </div>
      {/if}
    </section>

    {#if featured.length}
      <section class="featured" aria-labelledby="featured-title">
        <h2 id="featured-title">One neuron from each depth</h2>
        <ol class="featured-list">
          {#each featured as row (row[0] * 100000 + row[1])}
            <li style:--c={depthColor(row[0])}>
              <a href={urlFor(row[0], row[1])} onclick={(event) => { if (event.button || event.metaKey || event.ctrlKey || event.shiftKey) return; event.preventDefault(); void openNeuron(row[0], row[1], true); }}>
                <span class="f-meta"><span class="f-dot" aria-hidden="true"></span>{BANDS[bandOf(row[0])].name} · {neuronId(row[0], row[1])}</span>
                <span class="f-title"><TitleText text={row[2] || row[3]} /></span>
              </a>
            </li>
          {/each}
        </ol>
      </section>
    {/if}

    <section class="key" aria-labelledby="key-title">
      <h2 id="key-title">Key</h2>
      <ul class="legend">
        <li><Token token=" owl" /><span>Word start</span></li>
        <li><Token token="release" count={9} /><span>Variants merged</span></li>
        <li><Token token=" mass" flags={1} /><span>Lenses agree</span></li>
        <li><Token token=" mass" flags={2} /><span>Confirmed on real text</span></li>
        <li><Token token="圣经" /><span>Hover or tap to translate</span></li>
        <li><span class="tok-translated">the Bible</span><span>Translated</span></li>
        <li><Token token={"\u0001<0xE5>"} /><span>Byte piece</span></li>
      </ul>
      <LensMap {layer} onlayer={(L) => { setLayer(L, true); explorer?.scrollIntoView({ block: "start", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }); }} />
    </section>

    <footer class="notes">
    <details class="about">
      <summary>About these readings</summary>
      <ul class="caveat-list">
        <li>Reads are directions, not effect sizes.</li>
        <li>Before layer 21, a write describes pushing the neuron by hand, not a natural firing.</li>
        <li>Input reads from the weights are estimates. The real-text examples are the ground truth.</li>
        <li>No automatic test reads most poles at layers 13–20.</li>
        <li>Activations were measured in bf16; tokens seen fewer than 5 times are masked.</li>
        <li>Descriptions were written by Claude Sonnet and can be wrong.</li>
      </ul>
    </details>
    <p class="sources">Data: <strong>neuron-lens</strong> by a9 (stdlogits), CC BY-SA 4.0. Translations: <a href="https://cc-cedict.org/" rel="noreferrer">CC-CEDICT</a>, CC BY-SA 4.0.</p>
    </footer>
  </main>
  <PageFooter current="neurons" />
</div>

<style>
  .neurons-page {
    --neuron-tint-ink: color-mix(in oklch, var(--neuron-tint) 62%, var(--fg));
    --neuron-translated-ink: var(--accent-amber);
    display: flex; flex-direction: column; min-height: 100svh; overflow-x: clip; background: var(--bg); color: var(--fg);
    -webkit-font-smoothing: antialiased;
  }
  :global(:root[data-theme="light"]) .neurons-page { --neuron-tint-ink: color-mix(in oklch, var(--neuron-tint) 65%, var(--fg)); }
  .skip-link { position: fixed; inset: 8px auto auto 8px; z-index: 10; padding: var(--space-sm); background: var(--accent); color: var(--text-on-accent); border-radius: var(--radius); text-decoration: none; transform: translateY(-160%); }
  .skip-link:focus { transform: none; }
  /* Flat panels, as in the workbench: no sheen gradients on this page's content. */
  main {
    --surface-sheen: none;
    --control-sheen: none;
    flex: 1; display: grid; gap: var(--space-xl); width: min(100%, var(--page-max)); margin-inline: auto;
    padding-inline: var(--page-gutter); padding-block-end: var(--section-space);
  }
  .panel { border-radius: var(--radius-lg); background: var(--bg-alt); box-shadow: inset 0 0 0 1px var(--glass-line); }
  h2 { margin: 0; color: var(--fg); font-family: var(--font-structure); font-size: var(--text-lg); font-weight: var(--weight-display); line-height: 1.2; letter-spacing: -0.02em; text-wrap: balance; }

  .intro { display: grid; gap: var(--space-5); padding-block-start: var(--space-2xl); }
  .eyebrow { margin: 0; color: var(--neuron-tint-ink); font-family: var(--font-mono); font-size: var(--text-xs); }
  h1 { margin: 0; color: var(--fg); font-family: var(--font-structure); font-weight: var(--weight-display); font-size: var(--text-explorer-title); line-height: 1.02; letter-spacing: -0.045em; text-wrap: balance; }
  .lede { max-width: 60ch; margin: 0; color: var(--fg-dim); font-size: var(--text-md); line-height: 1.6; text-wrap: pretty; }
  .figures { display: flex; flex-wrap: wrap; gap: var(--space-3) var(--space-8); margin: 0; }
  .figures div { display: grid; gap: 0; }
  .figures dt { color: var(--fg-muted); font-size: var(--text-xs); }
  .figures dd { margin: 0; color: var(--fg); font-family: var(--font-mono); font-size: var(--text-md); font-variant-numeric: tabular-nums; }

  .explorer { display: grid; gap: var(--space-4); min-width: 0; scroll-margin-top: var(--neuron-sticky-top); }
  .explorer:focus { outline: none; }
  .explorer-bar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--space-3); }
  .translate { position: relative; display: inline-flex; align-items: center; gap: var(--space-3); min-height: var(--control-target); color: var(--fg-dim); font-size: var(--text-sm); cursor: pointer; }
  .translate input { position: absolute; opacity: 0; width: 1px; height: 1px; }
  .switch { position: relative; flex: none; width: 32px; height: 18px; border-radius: var(--radius-pill); background: var(--glass-strong); box-shadow: inset 0 0 0 1px var(--glass-line); transition: background-color var(--dur) var(--ease-out); }
  .switch::after { content: ""; position: absolute; top: 3px; left: 3px; width: 12px; height: 12px; border-radius: 50%; background: var(--fg-dim); transition: transform var(--dur) var(--ease-out), background-color var(--dur) var(--ease-out); }
  .translate input:checked + .switch { background: color-mix(in srgb, var(--accent-amber) 55%, transparent); }
  .translate input:checked + .switch::after { transform: translateX(14px); background: var(--fg); }
  .translate input:focus-visible + .switch { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .ribbon-panel { padding: var(--space-4) var(--space-5) var(--space-3); }
  .workspace { display: grid; grid-template-columns: minmax(16rem, 19rem) minmax(0, 1fr); gap: var(--space-4); align-items: start; }
  /* The rail stays in reach while the neuron scrolls; it scrolls itself when the list is long. */
  .rail-panel { position: sticky; top: var(--neuron-sticky-top); max-height: calc(100svh - var(--neuron-sticky-top) - var(--space-4)); overflow-y: auto; overscroll-behavior: contain; padding: var(--space-4); scrollbar-width: thin; }
  .stage { min-width: 0; padding: var(--space-5); scroll-margin-top: var(--neuron-sticky-top); }
  .notice { display: grid; justify-items: start; gap: var(--space-3); margin: 0; padding: var(--space-5); color: var(--fg-dim); font-size: var(--text-sm); }
  .notice p { margin: 0; }
  .text-button { min-height: var(--control-target); padding-inline: var(--space-4); border: 0; border-radius: var(--radius); background: var(--glass-strong); color: var(--fg); font: inherit; cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
  .text-button:hover { background: var(--glass-bright); }
  .text-button:active { scale: var(--press-scale); }
  .text-button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .skeleton { display: grid; gap: var(--space-3); }
  .bone { display: block; height: 14px; width: 40%; border-radius: var(--radius-sm); background: var(--glass-strong); animation: pulse 1.4s ease-in-out infinite; }
  .bone.wide { width: 24%; height: 28px; }
  .bone.block { width: 100%; height: 160px; border-radius: var(--radius-lg); }
  @keyframes pulse { 50% { opacity: 0.5; } }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }

  .featured { display: grid; gap: var(--space-4); }
  .featured-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 13rem), 1fr)); gap: var(--space-3); margin: 0; padding: 0; list-style: none; }
  .featured-list a {
    display: grid; align-content: start; gap: var(--space-2); height: 100%; padding: var(--space-4); border-radius: var(--radius-lg);
    background: var(--bg-alt); box-shadow: inset 0 0 0 1px var(--glass-line); color: var(--fg); text-decoration: none;
    transition: background-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out);
  }
  .featured-list a:hover { background: color-mix(in srgb, var(--bg-alt) 94%, var(--fg)); box-shadow: inset 0 0 0 1px color-mix(in oklch, var(--c) 50%, transparent); }
  .featured-list a:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .f-meta { display: flex; align-items: center; gap: var(--space-2); color: var(--fg-muted); font-size: var(--text-xs); }
  .f-dot { flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--c); }
  .f-title { font-size: var(--text-sm); line-height: 1.45; text-wrap: pretty; overflow-wrap: anywhere; }

  .key { display: grid; gap: var(--space-4); min-width: 0; }
  .legend { display: flex; flex-wrap: wrap; gap: var(--space-3) var(--space-6); margin: 0; padding: 0; list-style: none; }
  .legend li { display: inline-flex; align-items: center; gap: var(--space-2); color: var(--fg-muted); font-size: var(--text-xs); }
  .tok-translated { display: inline-flex; align-items: center; min-height: 22px; padding: 0 var(--space-xs); border-radius: var(--radius-sm); background: color-mix(in srgb, var(--accent-amber) 16%, var(--glass)); box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent-amber) 55%, transparent); color: var(--neuron-translated-ink); font-family: var(--font-mono); font-size: var(--text-sm); font-style: italic; white-space: nowrap; }
  .about { border-top: 1px solid var(--grid-line); }
  .about summary { display: flex; align-items: center; min-height: var(--control-target); color: var(--fg-dim); font-size: var(--text-sm); cursor: pointer; }
  .about summary:hover { color: var(--fg); }
  .about summary:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; border-radius: var(--radius-sm); }
  .caveat-list { display: grid; gap: var(--space-2); margin: 0 0 var(--space-4); padding-inline-start: var(--space-5); color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.55; }
  .notes { display: grid; gap: var(--space-3); }
  .sources { margin: 0; color: var(--fg-muted); font-size: var(--text-xs); }
  .sources a { color: var(--fg-dim); text-underline-offset: 3px; }
  .sources a:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; border-radius: var(--radius-sm); }

  @media (max-width: 1000px) {
    .workspace { grid-template-columns: minmax(0, 1fr); }
    .rail-panel { position: static; max-height: none; overflow: visible; }
    /* One column: keep the list short so the neuron below stays in reach; the next item peeks as a cue. */
    .rail-panel :global(.list) { max-height: 22rem; overflow-y: auto; overscroll-behavior: contain; scrollbar-width: thin; }
  }
  @media (max-width: 640px) {
    .ribbon-panel { padding: var(--space-3) var(--space-3) var(--space-2); }
    .stage { padding: var(--space-4); }
  }
  @media (prefers-reduced-motion: reduce) {
    .bone { animation: none; }
    .switch, .switch::after, .featured-list a, .text-button { transition: none; }
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
