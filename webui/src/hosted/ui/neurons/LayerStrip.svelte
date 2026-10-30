<script lang="ts">
  import MorphText from "../../../lib/ui/MorphText.svelte";
  import InfoTip from "../../../lib/ui/InfoTip.svelte";
  import FluentIcon from "../../../lib/ui/FluentIcon.svelte";
  import TitleText from "./TitleText.svelte";
  import { BANDS, FLAG_DETECTOR, FLAG_LABELLED, FLAG_PREDICTS, FLAG_SUPPRESSES, MIDDLE_CAUTION, bandOf, neuronId, type IndexRow } from "./data";

  // The selected layer: its name, a filter, and its neurons as a row of cards.
  let { layer, rows, selected, onneuron, hrefFor }: {
    layer: number;
    rows: IndexRow[];
    selected: number | null;
    onneuron: (neuron: number) => void;
    hrefFor: (neuron: number) => string;
  } = $props();

  // Each layer opens on its easiest neurons: the highest-scoring dozen, already ranked best first.
  const EASIEST = 12;
  type Filter = "easiest" | "all" | "labelled" | "detector" | "predicts" | "suppresses";
  let filter = $state<Filter>("easiest");
  const flagOf: Record<Filter, number> = { easiest: 0, all: 0, labelled: FLAG_LABELLED, detector: FLAG_DETECTOR, predicts: FLAG_PREDICTS, suppresses: FLAG_SUPPRESSES };
  const filters = $derived(([
    ["easiest", "Easiest"], ["all", `All ${rows.length.toLocaleString("en-US")}`], ["labelled", "Described"], ["predicts", "Predictors"], ["suppresses", "Suppressors"],
  ] as [Filter, string][]).filter(([value]) => value === "easiest" || value === "all" || rows.some((row) => (row[5] & flagOf[value]) !== 0)));
  const shown = $derived(
    filter === "easiest" ? rows.slice(0, EASIEST)
    : filter === "all" ? rows
    : rows.filter((row) => (row[5] & flagOf[filter]) !== 0),
  );
  // Long lists (every neuron of a layer) render 48 cards at a time.
  const PAGE = 48;
  let visible = $state(PAGE);
  $effect.pre(() => { void layer; void filter; visible = PAGE; });
  const rendered = $derived(shown.slice(0, visible));
  // A neuron picked elsewhere (search, the explorer) that isn't in the current view switches to All.
  $effect.pre(() => { void layer; filter = "easiest"; });
  $effect(() => {
    if (selected === null || !rows.length) return;
    if (!shown.some((row) => row[1] === selected)) { filter = "all"; return; }
    const at = shown.findIndex((row) => row[1] === selected);
    if (at >= visible) visible = Math.ceil((at + 1) / PAGE) * PAGE;
  });

  const band = $derived(BANDS[bandOf(layer)]);
  let track: HTMLOListElement | undefined = $state();

  // Keep the selected card in view as the selection moves (j/k, the ribbon, search).
  $effect(() => {
    void selected;
    const card = track?.querySelector<HTMLElement>('[aria-current="true"]');
    if (!card || !track) return;
    const left = card.offsetLeft - track.clientWidth / 2 + card.clientWidth / 2;
    track.scrollTo({ left, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  });

  function page(delta: -1 | 1) {
    track?.scrollBy({ left: delta * track.clientWidth * 0.8, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }

  function open(event: MouseEvent, neuron: number) {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    onneuron(neuron);
  }
</script>

<section class="strip" aria-labelledby="layer-title">
  <header class="strip-head">
    <div class="title">
      <h2 id="layer-title" tabindex="-1"><MorphText text={`Layer ${layer}`} duration={320} /></h2>
      <span class="band"><MorphText text={band.name} numbers={false} duration={260} /></span>
      {#if bandOf(layer) === "dark"}<span class="caution"><InfoTip icon="error" label="Why layers 13–20 are less reliable" text={MIDDLE_CAUTION} /></span>{/if}
    </div>
    <div class="controls">
      {#if filters.length > 1}
        <div class="filters" role="group" aria-label="Filter neurons">
          {#each filters as [value, label] (value)}
            <button type="button" aria-pressed={filter === value} onclick={() => (filter = value)}>{label}</button>
          {/each}
        </div>
      {/if}
      <div class="pagers">
        <button type="button" class="pager" aria-label="Scroll neurons back" onclick={() => page(-1)}><FluentIcon name="back" size={18} /></button>
        <button type="button" class="pager" aria-label="Scroll neurons forward" onclick={() => page(1)}><FluentIcon name="next" size={18} /></button>
      </div>
    </div>
  </header>

  {#if !rows.length}
    <p class="empty">Loading neurons…</p>
  {:else}
    <div class="track-wrap">
      <ol class="track" bind:this={track}>
        {#each rendered as row, i (row[0] * 100000 + row[1])}
          <li style:--i={Math.min(i, 10)}>
            <a class="card" href={hrefFor(row[1])} aria-current={selected === row[1] ? "true" : undefined} onclick={(event) => open(event, row[1])}>
              <span class="card-top">
                <span class="card-id">{neuronId(row[0], row[1])}</span>
                {#if row[5] & FLAG_LABELLED}<span class="flag">Described</span>{/if}
                {#if row[5] & FLAG_PREDICTS}<span class="flag">Predicts</span>{/if}
                {#if row[5] & FLAG_SUPPRESSES}<span class="flag">Suppresses</span>{/if}
              </span>
              <span class="card-title"><span class="sign" aria-hidden="true">+</span><span class="sr-only">Plus pole: </span><span class="t"><TitleText text={row[2]} /></span></span>
              <span class="card-title dim"><span class="sign" aria-hidden="true">−</span><span class="sr-only">Minus pole: </span><span class="t"><TitleText text={row[3]} /></span></span>
            </a>
          </li>
        {/each}
        {#if shown.length > visible}
          <li class="more-cell"><button type="button" class="more-card" onclick={() => (visible += PAGE)}>Show {Math.min(PAGE, shown.length - visible)} more<span>{visible.toLocaleString("en-US")} of {shown.length.toLocaleString("en-US")} shown</span></button></li>
        {/if}
      </ol>
    </div>
  {/if}
</section>

<style>
  .strip { display: grid; gap: var(--space-4); min-width: 0; }
  .strip-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--space-3) var(--space-5); }
  .title { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2) var(--space-4); }
  h2 { margin: 0; color: var(--fg); font-family: var(--font-structure); font-weight: var(--weight-display); font-size: var(--text-feature-title); line-height: 1.1; letter-spacing: -0.03em; font-variant-numeric: tabular-nums; }
  h2:focus { outline: none; }
  .band { color: var(--neuron-tint-ink); font-size: var(--text-md); font-weight: var(--weight-structure); }
  .caution { display: inline-flex; }
  .caution :global(.info-trigger) { color: var(--accent-yellow); }

  .filters { display: flex; flex-wrap: wrap; gap: var(--space-2); }
  .filters button { min-height: var(--control-target); padding-inline: var(--space-4); border: 0; border-radius: var(--radius-pill); background: var(--neuron-chip-bg); box-shadow: inset 0 0 0 1px var(--glass-line); color: var(--fg-dim); font: inherit; font-size: var(--text-sm); cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
  .filters button[aria-pressed="true"] { background: var(--fg); box-shadow: none; color: var(--bg); }
  .filters button:active { scale: var(--press-scale); }
  .filters button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (hover: hover) { .filters button:not([aria-pressed="true"]):hover { background: var(--glass-strong); color: var(--fg); } }

  .track-wrap { position: relative; min-width: 0; }
  .track {
    display: grid; grid-auto-flow: column; grid-auto-columns: minmax(15rem, 17rem); gap: var(--space-3);
    margin: 0; padding: var(--space-1) var(--space-1) var(--space-3); overflow-x: auto; scroll-snap-type: x proximity;
    scroll-padding-inline: var(--space-1); list-style: none; scrollbar-width: thin; overscroll-behavior-inline: contain;
  }
  .track li { scroll-snap-align: start; animation: card-in 360ms var(--ease-enter) backwards; animation-delay: calc(var(--i) * 30ms); }
  @keyframes card-in { from { opacity: 0; translate: 12px 0; } }
  .card {
    position: relative; display: grid; align-content: start; gap: var(--space-2); height: 100%; padding: var(--space-4) var(--space-5);
    border-radius: var(--radius-lg); background: var(--neuron-panel); box-shadow: var(--neuron-panel-edge); color: var(--fg); text-decoration: none;
    transition: box-shadow var(--dur) var(--ease-out), translate var(--dur) var(--ease-out), background-color var(--dur) var(--ease-out);
  }
  .card[aria-current="true"] { box-shadow: inset 0 0 0 2px var(--neuron-tint), var(--neuron-panel-lift); background: color-mix(in oklch, var(--neuron-tint) 10%, var(--neuron-panel)); }
  .card[aria-current="true"]::after { content: ""; position: absolute; inset: auto var(--space-5) 0; height: 3px; border-radius: var(--radius-pill) var(--radius-pill) 0 0; background: var(--neuron-tint); }
  .card:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (hover: hover) { .card:hover { translate: 0 -3px; box-shadow: var(--neuron-panel-edge), var(--neuron-panel-lift); } }
  .card-top { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2); }
  .card-id { color: var(--neuron-tint-ink); font-family: var(--font-mono); font-size: var(--text-sm); font-variant-numeric: tabular-nums; }
  .flag { padding: 0 var(--space-2); border-radius: var(--radius-pill); background: var(--glass-strong); color: var(--fg-dim); font-size: var(--text-xs); line-height: 1.6; }
  .card-title { display: flex; gap: var(--space-2); min-width: 0; font-size: var(--text-md); line-height: 1.4; }
  .card-title.dim { color: var(--fg-dim); font-size: var(--text-sm); }
  .t { display: -webkit-box; -webkit-line-clamp: 2; line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; overflow-wrap: anywhere; }
  .sign { flex: none; width: 1ch; color: var(--fg-muted); }

  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-3); }
  .pagers { display: flex; gap: var(--space-2); }
  .pager {
    display: grid; place-items: center; width: var(--control-target); height: var(--control-target);
    border: 0; border-radius: 50%; background: var(--neuron-chip-bg); box-shadow: inset 0 0 0 1px var(--glass-line); color: var(--fg); cursor: pointer;
    transition: scale var(--dur-fast) var(--ease-out), background-color var(--dur-fast) var(--ease-out);
  }
  @media (hover: hover) { .pager:hover { background: var(--glass-strong); } }
  .pager:active { scale: var(--press-scale); }
  .pager:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .empty { margin: 0; color: var(--fg-muted); }
  .more-card { display: grid; place-content: center; gap: var(--space-1); width: 100%; height: 100%; min-height: 120px; border: 0; border-radius: var(--radius-lg); background: transparent; box-shadow: inset 0 0 0 1px var(--glass-line); color: var(--fg); font: inherit; font-size: var(--text-md); cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out); }
  .more-card span { color: var(--fg-muted); font-size: var(--text-xs); }
  .more-card:active { scale: var(--press-scale); }
  .more-card:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (hover: hover) { .more-card:hover { background: var(--bg-hover); } }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  @media (max-width: 640px) { .pager { display: none; } .track { grid-auto-columns: minmax(78%, 1fr); } }
  @media (prefers-reduced-motion: reduce) {
    .track li { animation: none; }
    .card, .filters button, .pager { transition: none; }
    .card:hover { translate: none; }
    .filters button:active, .pager:active { scale: none; }
  }
</style>
