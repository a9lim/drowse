<script lang="ts">
  import { BANDS, BAND_ORDER, bandOf, depthColor, neuronId, type CatalogIndex, type IndexRow } from "./data";

  let { index, layer, neuron, highlight, onlayer, onneuron }: {
    index: CatalogIndex | null;
    layer: number;
    neuron: number | null;
    /** "layer:neuron" keys to light up, such as search matches. */
    highlight: Set<string>;
    onlayer: (layer: number, commit: boolean) => void;
    onneuron: (layer: number, neuron: number) => void;
  } = $props();

  const COL = 28;
  const DOT = 3.4;
  const TOP = 14;
  const PROFILE = 30;
  const W = COL * 36;

  const columns = $derived.by(() => {
    const byLayer: IndexRow[][] = Array.from({ length: 36 }, () => []);
    for (const row of index?.neurons ?? []) byLayer[row[0]]?.push(row);
    for (const rows of byLayer) rows.sort((a, b) => b[4] - a[4]);
    return byLayer;
  });
  const ROWS = $derived(Math.max(40, ...columns.map((rows) => rows.length)));
  const PLOT = $derived(TOP + ROWS * DOT + 6);
  const H = $derived(PLOT + PROFILE + 62);
  // Dots are sized and lit by their score's rank across the whole catalog, so the clearest stand out.
  const rankOf = $derived.by(() => {
    const scores = (index?.neurons ?? []).map((row) => row[4]).sort((a, b) => a - b);
    return (score: number) => {
      let lo = 0, hi = scores.length;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (scores[mid] < score) lo = mid + 1; else hi = mid; }
      return scores.length ? lo / scores.length : 0;
    };
  });
  // Share of the layer's poles that a weight-only lens reads, or that fire on one token.
  const readable = $derived(Array.from({ length: 36 }, (_, L) => {
    const stats = index?.layerMeta.find((meta) => meta.layer === L)?.stats;
    return stats ? Math.min(1, stats.legible + stats.detector) : 0;
  }));
  const readableMax = $derived(Math.max(0.01, ...readable));
  const missing = $derived(new Set((index?.layerMeta ?? []).filter((meta) => meta.missing).map((meta) => meta.layer)));
  const bands = $derived(BAND_ORDER.map((band) => {
    const layers = Array.from({ length: 36 }, (_, L) => L).filter((L) => bandOf(L) === band);
    return { band, from: layers[0], to: layers[layers.length - 1] };
  }));

  let svg: SVGSVGElement | undefined = $state();
  let drawn = $state(1008);
  // The viewBox scales with the pane; scale the labels back so they stay 10–11px on screen.
  const k = $derived(W / Math.max(drawn, 1));
  let hover = $state<{ layer: number; row: IndexRow | null; x: number; y: number } | null>(null);
  let dragging = $state(false);
  let moved = false;

  function locate(event: PointerEvent): { layer: number; row: IndexRow | null; x: number; y: number } | null {
    if (!svg) return null;
    const box = svg.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width) * W;
    const y = ((event.clientY - box.top) / box.height) * H;
    const L = Math.max(0, Math.min(35, Math.floor(x / COL)));
    const slot = Math.round((y - TOP - DOT / 2) / DOT);
    const rows = columns[L];
    // Pointer targets are generous: the nearest dot within two rows counts.
    let row: IndexRow | null = null;
    if (y < PLOT && rows.length) {
      const clamped = Math.max(0, Math.min(rows.length - 1, slot));
      if (Math.abs(clamped - slot) <= 2) row = rows[clamped];
    }
    // The tooltip is up to 18rem (288px) wide; keep it inside the ribbon.
    const left = Math.max(0, Math.min(box.width - 288, event.clientX - box.left - 144));
    return { layer: L, row, x: left, y: event.clientY - box.top };
  }

  function move(event: PointerEvent) {
    const spot = locate(event);
    hover = spot;
    if (dragging && spot && spot.layer !== layer) {
      moved = true;
      onlayer(spot.layer, false);
    }
  }

  function down(event: PointerEvent) {
    if (event.button !== 0) return;
    dragging = true;
    moved = false;
    svg?.setPointerCapture(event.pointerId);
    move(event);
  }

  function up(event: PointerEvent) {
    if (!dragging) return;
    dragging = false;
    const spot = locate(event);
    if (!spot) return;
    if (!moved && spot.row) onneuron(spot.row[0], spot.row[1]);
    else onlayer(spot.layer, true);
    if (event.pointerType !== "mouse") hover = null;
  }

  function key(event: KeyboardEvent) {
    const steps: Record<string, number> = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1, PageUp: 4, PageDown: -4 };
    let next = layer;
    if (event.key in steps) next = layer + steps[event.key];
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = 35;
    else if (event.key === "Enter" && columns[layer][0]) { event.preventDefault(); onneuron(layer, columns[layer][0][1]); return; }
    else return;
    event.preventDefault();
    onlayer(Math.max(0, Math.min(35, next)), true);
  }

  const valueText = $derived(`Layer ${layer}, ${BANDS[bandOf(layer)].name}, ${columns[layer]?.length ?? 0} neurons shown`);
  const tipRow = $derived(hover?.row ?? null);
</script>

<div class="ribbon" bind:clientWidth={drawn} class:ready={!!index} class:scrubbing={dragging}>
  <svg
    bind:this={svg}
    viewBox={`0 0 ${W} ${H}`}
    role="slider"
    tabindex="0"
    aria-label="Layer"
    aria-valuemin={0}
    aria-valuemax={35}
    aria-valuenow={layer}
    aria-valuetext={valueText}
    aria-describedby="ribbon-help"
    onpointermove={move}
    onpointerdown={down}
    onpointerup={up}
    onpointercancel={() => { dragging = false; hover = null; }}
    onpointerleave={() => { if (!dragging) hover = null; }}
    onkeydown={key}
  >
    <defs>
      <pattern id="ribbon-field" width="3.4" height="3.4" patternUnits="userSpaceOnUse"><circle cx="1.7" cy="1.7" r="0.45" class="field-dot" /></pattern>
      <pattern id="ribbon-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" class="hatch" /></pattern>
    </defs>

    <rect x="0" y={TOP - 4} width={W} height={ROWS * DOT + 8} fill="url(#ribbon-field)" class="field" />
    <rect class="lens" x={layer * COL + 2} y={TOP - 8} width={COL - 4} height={ROWS * DOT + 16} rx="8" style:--c={depthColor(layer)} />
    {#if hover && hover.layer !== layer}
      <rect class="hover-col" x={hover.layer * COL + 2} y={TOP - 8} width={COL - 4} height={ROWS * DOT + 16} rx="8" />
    {/if}

    {#each columns as rows, L}
      <g class="col" class:active={L === layer} style:--c={depthColor(L)} style:--L={L}>
        {#if missing.has(L)}
          <rect x={L * COL + 6} y={TOP} width={COL - 12} height={ROWS * DOT} rx="4" fill="url(#ribbon-hatch)" />
        {/if}
        {#each rows as row, slot (row[1])}
          {@const key = `${L}:${row[1]}`}
          {@const rank = rankOf(row[4])}
          {@const r = 0.9 + 0.9 * rank * rank}
          <circle
            class="dot"
            class:lit={highlight.has(key)}
            class:dim={highlight.size > 0 && !highlight.has(key)}
            class:selected={neuron === row[1] && L === layer}
            class:labelled={(row[5] & 1) !== 0}
            cx={L * COL + COL / 2}
            cy={TOP + slot * DOT + DOT / 2}
            {r}
            style:--slot={slot}
            style:--o={0.28 + 0.6 * rank}
          />
        {/each}
      </g>
    {/each}

    <g class="profile">
      <text class="profile-label" x="2" y={PLOT + 10} font-size={10 * k}>Readable by the lenses</text>
      {#each readable as share, L}
        {@const h = Math.max(1.5, (share / readableMax) * (PROFILE - 12))}
        <rect class="bar" class:active={L === layer} x={L * COL + 8} y={PLOT + PROFILE - h} width={COL - 16} height={h} rx="2" style:--c={depthColor(L)}></rect>
      {/each}
    </g>
    {#each Array.from({ length: 36 }, (_, L) => L) as L}
      <text class="layer-num" class:active={L === layer} class:minor={L % 4 !== 0 && L !== 35} x={L * COL + COL / 2} y={PLOT + PROFILE + 16} font-size={10 * k}>{L}</text>
    {/each}
    {#each bands as b}
      <g class="band" class:active={bandOf(layer) === b.band}>
        <line x1={b.from * COL + 4} x2={(b.to + 1) * COL - 4} y1={PLOT + PROFILE + 28} y2={PLOT + PROFILE + 28} />
        <text x={((b.from + b.to + 1) * COL) / 2} y={PLOT + PROFILE + 46} font-size={11 * k}>{b.to - b.from >= 2 ? BANDS[b.band].name.replace(" layers", "") : BANDS[b.band].range}</text>
      </g>
    {/each}
  </svg>

  {#if hover}
    <div class="tip" style:--x={`${hover.x}px`} style:--y={`${hover.y}px`} aria-hidden="true">
      <strong>{tipRow ? neuronId(tipRow[0], tipRow[1]) : `Layer ${hover.layer}`}</strong>
      {#if tipRow}
        <span><b>+</b> {tipRow[2] || "unclear"}</span>
        <span><b>−</b> {tipRow[3] || "unclear"}</span>
      {:else}
        <span>{BANDS[bandOf(hover.layer)].name} · {columns[hover.layer]?.length ?? 0} shown</span>
      {/if}
    </div>
  {/if}
  <p id="ribbon-help" class="sr-only">Use the arrow keys to change layer and Enter to open its clearest neuron.</p>
</div>

<style>
  .ribbon { position: relative; display: grid; gap: var(--space-3); min-width: 0; user-select: none; -webkit-user-select: none; }
  svg { display: block; width: 100%; height: auto; overflow: visible; cursor: pointer; touch-action: pan-y; border-radius: var(--radius-lg); }
  svg:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 6px; }
  .field-dot { fill: var(--fg-muted); opacity: 0.22; }
  .hatch { stroke: var(--grid-line); stroke-width: 2; }
  .lens {
    fill: color-mix(in oklch, var(--c) 16%, transparent);
    stroke: color-mix(in oklch, var(--c) 55%, transparent);
    stroke-width: 1;
    transform-box: fill-box;
    transition: x var(--dur-fast) var(--ease-out), fill var(--dur) var(--ease-out), stroke var(--dur) var(--ease-out);
  }
  .scrubbing .lens { transition: none; }
  .hover-col { fill: var(--bg-hover); pointer-events: none; }
  .dot {
    fill: var(--c);
    opacity: var(--o, 0.6);
    transform-box: fill-box;
    transform-origin: center;
    transition: opacity var(--dur) var(--ease-out), r var(--dur) var(--ease-out);
  }
  .ready .dot { animation: dot-in 520ms var(--ease-enter) backwards; animation-delay: calc(var(--L) * 16ms + var(--slot) * 4ms); }
  @keyframes dot-in { from { opacity: 0; transform: scale(0.2); } }
  .col.active .dot { opacity: calc(var(--o, 0.6) + 0.3); }
  .bar { fill: color-mix(in oklch, var(--c) 45%, transparent); transition: fill var(--dur) var(--ease-out); }
  .bar.active { fill: var(--c); }
  .profile-label { fill: var(--fg-muted); font-family: var(--font-reading); }
  .dot.labelled { stroke: var(--fg); stroke-width: 0.7; stroke-opacity: 0.6; }
  .dot.dim { opacity: 0.16; }
  .dot.lit { opacity: 1; r: 2.2; }
  .dot.selected { opacity: 1; r: 2.4; stroke: var(--fg); stroke-width: 1.4; stroke-opacity: 1; }
  .layer-num { fill: var(--fg-muted); font-family: var(--font-mono); text-anchor: middle; font-variant-numeric: tabular-nums; transition: fill var(--dur-fast) var(--ease-out); }
  .layer-num.active { fill: var(--fg); font-weight: var(--weight-data-bold); }
  .band line { stroke: var(--grid-line); stroke-width: 2; stroke-linecap: round; transition: stroke var(--dur) var(--ease-out); }
  .band text { fill: var(--fg-muted); font-family: var(--font-reading); text-anchor: middle; transition: fill var(--dur) var(--ease-out); }
  .band.active line { stroke: var(--neuron-tint); }
  .band.active text { fill: var(--fg); }
  .tip {
    position: absolute; left: 0; top: 0; z-index: 3; transition: transform 90ms var(--ease-out); display: grid; gap: 0; width: max-content; max-width: min(18rem, 80vw);
    padding: var(--space-3) var(--space-4); border-radius: var(--radius); background: var(--bg-elev); box-shadow: var(--shadow-card);
    color: var(--fg-dim); font-size: var(--text-xs); line-height: 1.45; pointer-events: none;
    transform: translate(var(--x), calc(var(--y) - 100% - 14px));
  }
  .tip strong { color: var(--fg); font-family: var(--font-mono); font-size: var(--text-sm); }
  .tip b { color: var(--fg-muted); }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  @media (max-width: 640px) {
    .layer-num.minor { display: none; }
    .profile-label { display: none; }
    .band text { display: none; }
    .band.active text { display: block; }
  }
  @media (prefers-reduced-motion: reduce) {
    .ready .dot { animation: none; }
    .lens, .dot, .layer-num, .band line, .band text, .bar, .tip { transition: none; }
  }
  @media (forced-colors: active) { .dot { fill: CanvasText; } .lens { stroke: Highlight; } }
</style>
