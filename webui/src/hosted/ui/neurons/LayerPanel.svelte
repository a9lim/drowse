<script lang="ts">
  import MorphText from "../../../lib/ui/MorphText.svelte";
  import InfoTip from "../../../lib/ui/InfoTip.svelte";
  import TitleText from "./TitleText.svelte";
  import { BANDS, FLAG_DETECTOR, FLAG_LABELLED, FLAG_PREDICTS, FLAG_SUPPRESSES, LENSES, MIDDLE_CAUTION, bandOf, neuronId, percent, type Head, type IndexRow, type LayerMeta } from "./data";

  // The explorer's left rail: the layer's guidance and statistics, then its neurons as a list.
  let { layer, meta, rows, selected, onneuron, hrefFor }: {
    layer: number;
    meta: LayerMeta | undefined;
    rows: IndexRow[];
    selected: number | null;
    onneuron: (neuron: number) => void;
    hrefFor: (neuron: number) => string;
  } = $props();

  type Filter = "all" | "labelled" | "detector" | "predicts" | "suppresses";
  let filter = $state<Filter>("all");
  const flagOf: Record<Filter, number> = { all: 0, labelled: FLAG_LABELLED, detector: FLAG_DETECTOR, predicts: FLAG_PREDICTS, suppresses: FLAG_SUPPRESSES };
  const filters = $derived(([
    ["all", "All"], ["labelled", "Described"], ["detector", "Detectors"], ["predicts", "Predictors"], ["suppresses", "Suppressors"],
  ] as [Filter, string][]).filter(([value]) => value === "all" || rows.some((row) => (row[5] & flagOf[value]) !== 0)));
  const shown = $derived(filter === "all" ? rows : rows.filter((row) => (row[5] & flagOf[filter]) !== 0));
  $effect.pre(() => { void layer; filter = "all"; });

  const band = $derived(BANDS[bandOf(layer)]);
  const stats = $derived(meta?.stats);
  const heads = $derived(meta?.heads ?? []);

  function headKind(head: Head): { kind: string; label: string } {
    if (head.sink > 0.9) return { kind: "sink", label: "Attention sink" };
    if (head.prev > 0.5) return { kind: "pattern", label: "Previous-token head" };
    if (head.induction > 0.3) return { kind: "pattern", label: "Induction head" };
    if (head.duplicate > 0.3) return { kind: "pattern", label: "Duplicate-token head" };
    if (head.self > 0.5) return { kind: "pattern", label: "Self-attending head" };
    if (head.copy < -0.3) return { kind: "suppress", label: "Suppresses what it reads" };
    if (head.copies || head.copy > 0.3) return { kind: "copy", label: "Copies what it reads" };
    if (head.next) return { kind: "copy", label: "Promotes the next token" };
    return { kind: "other", label: head.dark ? "No test reads it" : "Mixed" };
  }
  const headKinds = $derived(heads.map(headKind));
  // The list staggers in once; later layer changes swap it instantly so scrubbing stays smooth.
  let entered = $state(false);
  $effect(() => { if (rows.length && !entered) setTimeout(() => (entered = true), 600); });

  function open(event: MouseEvent, neuron: number) {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    onneuron(neuron);
  }
</script>

<section class="rail" aria-labelledby="layer-title">
  <header class="layer-head">
    <p class="kicker"><MorphText text={`${band.name} · ${band.range}`} numbers={false} duration={240} /></p>
    <div class="title-row">
      <h2 id="layer-title" tabindex="-1"><MorphText text={`Layer ${layer}`} duration={300} /></h2>
      {#if bandOf(layer) === "dark"}<span class="caution"><InfoTip icon="error" label="Why layers 13–20 are less reliable" text={MIDDLE_CAUTION} /></span>{/if}
    </div>
    {#key band}
      <p class="summary swap">{band.summary}</p>
    {/key}
    {#if meta?.ins && meta?.outs}
      <p class="trusted">
        <span class="trusted-key">Best reads</span>
        {#each [meta.ins[0], meta.outs[0]] as col, i (i)}<span class="lens-chip"><MorphText text={LENSES[col].short} numbers={false} duration={240} /></span>{/each}
      </p>
    {/if}
  </header>

  {#if stats}
    <dl class="stats" aria-label="Across all 24,576 poles of this layer">
      {#each [["Silent", stats.silent], ["Detectors", stats.detector], ["Legible", stats.legible], ["Dark", stats.dark]] as [name, value] (name)}
        <div class="stat">
          <dt>{name}</dt>
          <dd><span class="track"><span style:--v={Math.max(0.015, value as number)}></span></span><span class="num"><MorphText text={percent(value as number)} duration={300} /></span></dd>
        </div>
      {/each}
    </dl>
  {/if}

  {#if heads.length}
    <details class="heads">
      <summary><span>Attention heads <span class="head-key">Bright: copies · Mid: position pattern · Dim: sink</span></span><span class="heads-strip" aria-hidden="true">{#each headKinds as k}<span class={`head ${k.kind}`}></span>{/each}</span></summary>
      <ul class="head-list">
        {#each heads as head, i (head.h)}
          <li><span class="head-id">H{head.h}</span><span class="head-kind">{headKinds[i].label}</span>{#if head.reads}<span class="head-reads">{head.reads}</span>{/if}</li>
        {/each}
      </ul>
    </details>
  {/if}

  <div class="list-head">
    <h3><MorphText text={`${shown.length} neurons`} duration={260} /></h3>
    {#if filters.length > 1}
      <div class="filters" role="group" aria-label="Filter neurons">
        {#each filters as [value, label] (value)}
          <button type="button" aria-pressed={filter === value} onclick={() => (filter = value)}>{label}</button>
        {/each}
      </div>
    {/if}
  </div>

  {#if meta?.missing}
    <p class="empty">This layer isn't in the catalog.</p>
  {:else if !rows.length}
    <p class="empty">Loading neurons…</p>
  {:else}
    <ol class="list" class:entered>
      {#each shown as row, i (row[0] * 100000 + row[1])}
        <li style:--i={Math.min(i, 12)}>
          <a class="item" href={hrefFor(row[1])} aria-current={selected === row[1] ? "true" : undefined} onclick={(event) => open(event, row[1])}>
            <span class="item-top">
              <span class="item-id">{neuronId(row[0], row[1])}</span>
              {#if row[5] & FLAG_LABELLED}<span class="flag">Described</span>{/if}
              {#if row[5] & FLAG_PREDICTS}<span class="flag">Predicts</span>{/if}
              {#if row[5] & FLAG_SUPPRESSES}<span class="flag warn">Suppresses</span>{/if}
            </span>
            <span class="item-title"><span class="sign" aria-hidden="true">+</span><span class="sr-only">Plus pole: </span><span class="t"><TitleText text={row[2]} /></span></span>
            <span class="item-title dim"><span class="sign" aria-hidden="true">−</span><span class="sr-only">Minus pole: </span><span class="t"><TitleText text={row[3]} /></span></span>
          </a>
        </li>
      {/each}
    </ol>
  {/if}
</section>

<style>
  .rail { display: grid; align-content: start; gap: var(--space-5); min-width: 0; }
  .layer-head { display: grid; gap: var(--space-2); }
  .kicker { margin: 0; color: var(--neuron-tint-ink); font-size: var(--text-xs); font-weight: var(--weight-structure-bold); }
  h2 { margin: 0; color: var(--fg); font-family: var(--font-structure); font-weight: var(--weight-display); font-size: var(--text-md); line-height: 1.2; letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }
  h2:focus { outline: none; }
  .title-row { display: flex; align-items: center; gap: var(--space-2); }
  .caution { display: inline-flex; animation: caution-in var(--dur-slow) var(--ease-enter) backwards; }
  .caution :global(.info-trigger) { color: var(--accent-yellow); }
  @keyframes caution-in { from { opacity: 0; transform: scale(0.25); filter: blur(4px); } }
  .summary { margin: 0; color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.5; text-wrap: pretty; }
  .swap { animation: guide-in var(--dur-slow) var(--ease-enter) backwards; }
  @keyframes guide-in { from { opacity: 0; transform: translateY(3px); } }
  .trusted { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2); margin: var(--space-1) 0 0; }
  .trusted-key { color: var(--fg-muted); font-size: var(--text-xs); }
  .lens-chip { display: inline-flex; align-items: center; min-height: 22px; padding-inline: var(--space-2); border-radius: var(--radius-sm); background: color-mix(in oklch, var(--neuron-tint) 16%, transparent); color: var(--fg); font-family: var(--font-mono); font-size: var(--text-xs); cursor: help; }

  .stats { display: grid; gap: var(--space-2); margin: 0; }
  .stat { display: grid; grid-template-columns: 5.5rem minmax(0, 1fr); align-items: center; gap: var(--space-3); }
  dt { color: var(--fg-dim); font-size: var(--text-xs); }
  dd { display: grid; grid-template-columns: minmax(0, 1fr) 3.6rem; align-items: center; gap: var(--space-3); margin: 0; }
  .track { position: relative; height: 4px; border-radius: var(--radius-pill); background: var(--glass-strong); overflow: hidden; }
  .track span { position: absolute; inset: 0; border-radius: inherit; background: var(--neuron-tint); transform-origin: left; scale: var(--v) 1; transition: scale var(--dur-slow) var(--ease-out); }
  .num { color: var(--fg); font-family: var(--font-mono); font-size: var(--text-xs); font-variant-numeric: tabular-nums; text-align: end; }

  .heads { border-radius: var(--radius-sm); background: var(--glass); }
  .heads summary { display: grid; gap: var(--space-2); min-height: var(--control-target); padding: var(--space-3); color: var(--fg-dim); font-size: var(--text-xs); cursor: pointer; list-style: none; border-radius: var(--radius-sm); }
  .heads summary::-webkit-details-marker { display: none; }
  .head-key { display: block; color: var(--fg-muted); }
  .heads summary:hover { color: var(--fg); }
  .heads summary:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .heads-strip { display: grid; grid-template-columns: repeat(16, minmax(0, 1fr)); row-gap: var(--space-xs); }
  .head { justify-self: center; width: 80%; height: 8px; border-radius: var(--data-mark-radius); background: var(--glass-strong); }
  /* Neutral steps: the violet-cyan ramp means depth everywhere else on the page. */
  .head.sink { background: var(--fg-muted); opacity: 0.5; }
  .head.pattern { background: var(--fg-dim); }
  .head.copy { background: var(--fg); }
  .head.suppress { background: var(--fg-muted); }
  .head-list { display: grid; margin: 0; padding: 0 var(--space-3) var(--space-3); list-style: none; }
  .head-list li { display: grid; grid-template-columns: 3.2ch minmax(0, 1fr); gap: 0 var(--space-2); padding-block: var(--space-1); border-top: 1px solid var(--grid-line); font-size: var(--text-xs); }
  .head-id { grid-row: span 2; color: var(--fg-muted); font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
  .head-kind { color: var(--fg-dim); }
  .head-reads { color: var(--fg-muted); font-family: var(--font-mono); overflow-wrap: anywhere; }

  .list-head { display: grid; gap: var(--space-2); padding-top: var(--space-2); }
  h3 { margin: 0; color: var(--fg); font-size: var(--text-sm); font-weight: var(--weight-structure); font-variant-numeric: tabular-nums; }
  .filters { display: flex; flex-wrap: wrap; gap: var(--space-1); }
  .filters button { position: relative; min-height: 28px; padding-inline: var(--space-3); border: 0; border-radius: var(--radius-pill); background: transparent; box-shadow: inset 0 0 0 1px var(--glass-line); color: var(--fg-dim); font: inherit; font-size: var(--text-xs); cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out); }
  .filters button::after { content: ""; position: absolute; inset: calc(var(--space-xs) * -0.75) 0; }
  .filters button:hover { background: var(--bg-hover); color: var(--fg); }
  .filters button:active { scale: var(--press-scale); }
  .filters button[aria-pressed="true"] { background: var(--glass-bright); box-shadow: none; color: var(--fg); }
  .filters button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }

  .list { display: grid; gap: 0; margin: 0; padding: 0; list-style: none; }
  .list:not(.entered) li { animation: item-in 240ms var(--ease-enter) backwards; animation-delay: calc(var(--i) * 14ms); }
  @keyframes item-in { from { opacity: 0; transform: translateY(3px); } }
  .item {
    position: relative; display: grid; gap: 0; padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm);
    color: var(--fg); text-decoration: none; transition: background-color var(--dur-fast) var(--ease-out);
  }
  @media (hover: hover) { .item:hover { background: var(--bg-hover); } }
  .item[aria-current="true"] { background: var(--glass-bright); }
  .item[aria-current="true"]::before { content: ""; position: absolute; inset: var(--space-2) auto var(--space-2) 0; width: 2px; border-radius: var(--radius-pill); background: var(--neuron-tint); }
  .item:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: -2px; }
  .item-top { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2); }
  .item-id { color: var(--neuron-tint-ink); font-family: var(--font-mono); font-size: var(--text-xs); font-variant-numeric: tabular-nums; }
  .flag { padding: 0 var(--space-2); border-radius: var(--radius-pill); background: var(--glass-strong); color: var(--fg-dim); font-size: var(--text-xs); line-height: 1.6; }
  .flag.warn { color: var(--fg-dim); }
  .item-title { display: flex; gap: var(--space-2); min-width: 0; font-size: var(--text-sm); line-height: 1.4; }
  .item-title.dim { color: var(--fg-dim); font-size: var(--text-xs); }
  .t { display: -webkit-box; -webkit-line-clamp: 2; line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  .sign { flex: none; width: 1ch; color: var(--fg-muted); }
  .empty { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  @media (prefers-reduced-motion: reduce) {
    .list li, .swap, .caution { animation: none; }
    .item, .filters button, .track span { transition: none; }
    .filters button:active { scale: none; }
  }
</style>
