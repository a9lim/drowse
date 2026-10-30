<script lang="ts">
  import { BANDS, BAND_ORDER, bandOf, depthColor, neuronId, type CatalogIndex, type IndexRow } from "./data";

  let { index, layer, neuron, highlight, onlayer, onneuron, scores = null, peek }: {
    index: CatalogIndex | null;
    /** All mode: a neuron's row if its layer's index has loaded (asking loads it). */
    peek?: (layer: number, neuron: number) => IndexRow | null;
    /** With the "All" toggle on: each layer's 12,288 readability bytes, drawn as a texture instead of dots. */
    scores?: Uint8Array[] | null;
    layer: number;
    neuron: number | null;
    /** "layer:neuron" keys to light up, such as search matches. */
    highlight: Set<string>;
    onlayer: (layer: number, commit: boolean) => void;
    onneuron: (layer: number, neuron: number) => void;
  } = $props();

  const COL = 32;
  const GAP = 10;
  const TOP = 20;
  const W = COL * 36;

  const columns = $derived.by(() => {
    const byLayer: IndexRow[][] = Array.from({ length: 36 }, () => []);
    for (const row of index?.neurons ?? []) byLayer[row[0]]?.push(row);
    for (const rows of byLayer) rows.sort((a, b) => b[4] - a[4]);
    return byLayer;
  });
  const ROWS = $derived(Math.max(40, ...columns.map((rows) => rows.length)));
  const PLOT = $derived(TOP + ROWS * GAP);
  const H = $derived(PLOT + 64);
  // Dots are sized and lit by their score's rank across the whole catalog, so the clearest stand out.
  const rankOf = $derived.by(() => {
    const scores = (index?.neurons ?? []).map((row) => row[4]).sort((a, b) => a - b);
    return (score: number) => {
      let lo = 0, hi = scores.length;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (scores[mid] < score) lo = mid + 1; else hi = mid; }
      return scores.length ? lo / scores.length : 0;
    };
  });
  const bands = $derived(BAND_ORDER.map((band) => {
    const layers = Array.from({ length: 36 }, (_, L) => L).filter((L) => bandOf(L) === band);
    return { band, from: layers[0], to: layers[layers.length - 1] };
  }));

  // ---------- all-neurons texture: each column is a 24 × 512 grid, one cell per neuron in index order,
  // so a cell's position is its neuron number and the easiest neurons show as bright points.
  const GRID_W = 24;
  const GRID_H = 512;
  const everyNeuron = $derived(!!scores);
  const order = $derived(scores ? scores.map((layerScores) => Array.from({ length: layerScores.length }, (_, n) => n)) : null);
  let texture = $state("");
  let themeTick = $state(0);

  function channels(hex: string): [number, number, number] {
    const value = hex.trim().replace("#", "");
    const n = parseInt(value.length === 3 ? value.split("").map((c) => c + c).join("") : value, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  $effect(() => {
    void themeTick;
    if (!scores || !order) { texture = ""; return; }
    const style = getComputedStyle(document.documentElement);
    const deep = channels(style.getPropertyValue("--neuron-deep"));
    const late = channels(style.getPropertyValue("--neuron-late"));
    const canvas = document.createElement("canvas");
    canvas.width = W;
    canvas.height = GRID_H;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const image = ctx.createImageData(W, GRID_H);
    for (let L = 0; L < 36; L += 1) {
      const t = L / 35;
      const color = deep.map((c, i) => Math.round(c + (late[i] - c) * t));
      const x0 = L * COL + (COL - GRID_W) / 2;
      const layerOrder = order[L];
      const layerScores = scores[L];
      for (let rank = 0; rank < layerOrder.length; rank += 1) {
        const x = x0 + (rank % GRID_W);
        const y = Math.floor(rank / GRID_W);
        // Four brightness steps keep the cells crisp rather than a smooth wash.
        const v = Math.round((layerScores[layerOrder[rank]] / 255) * 4) / 4;
        const at = (y * W + x) * 4;
        image.data[at] = color[0];
        image.data[at + 1] = color[1];
        image.data[at + 2] = color[2];
        image.data[at + 3] = Math.round(255 * (0.1 + 0.9 * v ** 1.6));
      }
    }
    ctx.putImageData(image, 0, 0);
    texture = canvas.toDataURL("image/png");
  });

  // Redraw the texture when the theme changes, since the depth colours differ per theme.
  $effect(() => {
    const observer = new MutationObserver(() => (themeTick += 1));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  });

  function cellOf(L: number, n: number): { x: number; y: number } | null {
    if (!order) return null;
    const rank = n;
    if (rank < 0 || rank >= order[L].length) return null;
    return { x: L * COL + (COL - GRID_W) / 2 + (rank % GRID_W) + 0.5, y: TOP + ((Math.floor(rank / GRID_W) + 0.5) / GRID_H) * (ROWS * GAP) };
  }
  const selectedCell = $derived(everyNeuron && neuron !== null ? cellOf(layer, neuron) : null);
  let hoverNeuron = $state<number | null>(null);

  let svg: SVGSVGElement | undefined = $state();
  let scroller: HTMLDivElement | undefined = $state();
  let drawn = $state(W);
  // The viewBox scales with the drawn width; scale labels back so they keep a fixed on-screen size.
  const k = $derived(W / Math.max(drawn, 1));
  let hover = $state<{ layer: number; row: IndexRow | null; x: number; y: number; flip: boolean } | null>(null);
  let dragging = $state(false);
  let moved = false;

  // Fisheye: the hovered column and its neighbours grow, so dots are easy to aim at.
  function lift(L: number): number {
    if (!hover) return 1;
    const d = L - hover.layer;
    return 1 + 0.75 * Math.exp(-(d * d) / 2.2);
  }

  function locate(event: PointerEvent) {
    if (!svg || !scroller) return null;
    const box = svg.getBoundingClientRect();
    const x = ((event.clientX - box.left) / box.width) * W;
    const y = ((event.clientY - box.top) / box.height) * H;
    const L = Math.max(0, Math.min(35, Math.floor(x / COL)));
    const slot = Math.round((y - TOP - GAP / 2) / GAP);
    const rows = columns[L];
    let row: IndexRow | null = null;
    if (y < PLOT + GAP && rows.length) {
      const clamped = Math.max(0, Math.min(rows.length - 1, slot));
      if (Math.abs(clamped - slot) <= 1) row = rows[clamped];
    }
    if (order) {
      row = null;
      const cx = Math.floor(x - (L * COL + (COL - GRID_W) / 2));
      const cy = Math.floor(((y - TOP) / (ROWS * GAP)) * GRID_H);
      const rank = cx >= 0 && cx < GRID_W && cy >= 0 && cy < GRID_H ? cy * GRID_W + cx : -1;
      hoverNeuron = rank >= 0 && rank < order[L].length ? order[L][rank] : null;
    }
    const outer = scroller.getBoundingClientRect();
    const px = event.clientX - outer.left;
    // The tip is centred on the pointer; keep it inside the panel, which clips anything outside.
    const py = Math.max(90, Math.min(outer.height - 90, event.clientY - outer.top));
    return { layer: L, row, x: px, y: py, flip: px > outer.width - 300 };
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
    if (event.pointerType === "mouse") svg?.setPointerCapture(event.pointerId);
    move(event);
  }

  function up(event: PointerEvent) {
    if (!dragging) return;
    dragging = false;
    const spot = locate(event);
    if (!spot) return;
    if (!moved && order && hoverNeuron !== null) onneuron(spot.layer, hoverNeuron);
    else if (!moved && spot.row) onneuron(spot.row[0], spot.row[1]);
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

  // On narrow screens the explorer scrolls sideways; keep the selected layer in view.
  $effect(() => {
    void layer;
    if (!scroller || scroller.scrollWidth <= scroller.clientWidth + 1) return;
    const colWidth = scroller.scrollWidth / 36;
    const target = layer * colWidth - scroller.clientWidth / 2 + colWidth / 2;
    scroller.scrollTo({ left: target, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  });

  const valueText = $derived(`Layer ${layer}, ${BANDS[bandOf(layer)].name}, ${columns[layer]?.length ?? 0} neurons`);
  const tipRow = $derived(hover?.row ?? null);
  const peeked = $derived(everyNeuron && hoverNeuron !== null && hover ? peek?.(hover.layer, hoverNeuron) ?? null : tipRow);
  const selectedSlot = $derived(neuron === null ? -1 : columns[layer]?.findIndex((row) => row[1] === neuron) ?? -1);
</script>

<div class="ribbon" class:ready={!!index} class:scrubbing={dragging} class:hovering={!!hover}>
  <div class="scroller" bind:this={scroller}>
    <svg
      bind:this={svg}
      bind:clientWidth={drawn}
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
      {#each Array.from({ length: 36 }, (_, L) => L) as L}
        <rect class="col-bg" class:active={L === layer} class:hot={hover?.layer === L && L !== layer} x={L * COL + 3} y={TOP - 12} width={COL - 6} height={ROWS * GAP + 16} rx="10" style:--c={depthColor(L)} />
      {/each}

      {#if index}<rect class="scan" x="0" y={TOP - 12} width="6" height={ROWS * GAP + 16} rx="3" />{/if}

      {#if everyNeuron && texture}
        <image class="texture" href={texture} x="0" y={TOP} width={W} height={ROWS * GAP} preserveAspectRatio="none" />
        {#if selectedCell}
          {#key `${layer}:${neuron}`}<circle class="ripple" cx={selectedCell.x} cy={selectedCell.y} r="6" style:--c={depthColor(layer)} />{/key}
          <circle class="cell-ring" cx={selectedCell.x} cy={selectedCell.y} r="4" />
        {/if}
      {/if}

      {#each everyNeuron ? [] : columns as rows, L}
        <g class="col" class:active={L === layer} style:--c={depthColor(L)} style:--L={L} style:--s={lift(L)}>
          {#each rows as row, slot (row[1])}
            {@const key = `${L}:${row[1]}`}
            {@const rank = rankOf(row[4])}
            <circle
              class="dot"
              class:lit={highlight.has(key)}
              class:dim={highlight.size > 0 && !highlight.has(key)}
              class:selected={neuron === row[1] && L === layer}
              class:labelled={(row[5] & 1) !== 0}
              class:pointed={tipRow === row}
              cx={L * COL + COL / 2}
              cy={TOP + slot * GAP + GAP / 2}
              r={1.8 + 2 * rank}
              style:--slot={slot}
              style:--o={0.35 + 0.65 * rank}
            />
          {/each}
        </g>
      {/each}

      {#if selectedSlot >= 0 && !everyNeuron}
        {#key `${layer}:${neuron}`}
          <circle class="ripple" cx={layer * COL + COL / 2} cy={TOP + selectedSlot * GAP + GAP / 2} r="6" style:--c={depthColor(layer)} />
        {/key}
      {/if}

      {#each Array.from({ length: 36 }, (_, L) => L) as L}
        <text class="layer-num" class:active={L === layer} x={L * COL + COL / 2} y={PLOT + 18} font-size={11 * Math.min(k, 1.25)}>{L}</text>
      {/each}
      {#each bands as b}
        <g class="band" class:active={bandOf(layer) === b.band}>
          <line x1={b.from * COL + 6} x2={(b.to + 1) * COL - 6} y1={PLOT + 32} y2={PLOT + 32} />
          <text x={((b.from + b.to + 1) * COL) / 2} y={PLOT + 52} font-size={12 * Math.min(k, 1.25)}>{b.to - b.from >= 2 ? BANDS[b.band].name.replace(" layers", "") : BANDS[b.band].range}</text>
        </g>
      {/each}
    </svg>

    {#if hover}
      <div class="tip" class:flip={hover.flip} style:--x={`${hover.x}px`} style:--y={`${hover.y}px`} aria-hidden="true">
        {#if everyNeuron && hoverNeuron !== null}
          <strong>{neuronId(hover.layer, hoverNeuron)}</strong>
          {#if peeked?.[6] || peeked?.[7]}
            <span class="tip-over"><em>Triggered by:</em>{peeked[6] || "unclear"}</span>
            <span class="tip-over"><em>Steers outputs:</em>{peeked[7] || "unclear"}</span>
          {:else}
            <span class="tip-line">{BANDS[bandOf(hover.layer)].name}</span>
          {/if}
          <span class="tip-hint">Click to open</span>
        {:else if tipRow}
          <strong>{neuronId(tipRow[0], tipRow[1])}</strong>
          <span class="tip-over"><em>Triggered by:</em>{tipRow[6] || "unclear"}</span>
          <span class="tip-over"><em>Steers outputs:</em>{tipRow[7] || "unclear"}</span>
          <span class="tip-hint">Click to open</span>
        {:else}
          <strong>Layer {hover.layer}</strong>
          <span class="tip-line">{BANDS[bandOf(hover.layer)].name}</span>
        {/if}
      </div>
    {/if}
  </div>
  <p id="ribbon-help" class="sr-only">Use the arrow keys to change layer and Enter to open its clearest neuron.</p>
</div>

<style>
  .ribbon { position: relative; min-width: 0; user-select: none; -webkit-user-select: none; }
  .scroller { position: relative; overflow-x: auto; overflow-y: visible; scrollbar-width: thin; overscroll-behavior-inline: contain; }
  svg { display: block; width: 100%; min-width: 56rem; height: auto; overflow: visible; cursor: pointer; touch-action: pan-x pan-y; border-radius: var(--radius-lg); }
  svg:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 4px; }

  .col-bg { fill: transparent; transition: fill var(--dur) var(--ease-out); }
  .col-bg.hot { fill: var(--bg-hover); }
  .col-bg.active { fill: color-mix(in oklch, var(--c) 14%, transparent); stroke: color-mix(in oklch, var(--c) 50%, transparent); stroke-width: 1; }

  .scan { fill: var(--neuron-tint); opacity: 0; pointer-events: none; }
  .ready .scan { animation: scan 1.3s cubic-bezier(0.45, 0, 0.2, 1) backwards; }
  @keyframes scan {
    0% { opacity: 0.9; transform: translateX(0); }
    90% { opacity: 0.9; }
    100% { opacity: 0; transform: translateX(1146px); }
  }

  .dot {
    fill: var(--c);
    opacity: var(--o, 0.6);
    transform-box: fill-box;
    transform-origin: center;
    scale: var(--s, 1);
    transition: scale 180ms var(--ease-out), opacity var(--dur) var(--ease-out);
  }
  .ready .dot { animation: dot-in 600ms var(--ease-enter) backwards; animation-delay: calc(var(--L) * 32ms + var(--slot) * 3ms); }
  @keyframes dot-in { from { opacity: 0; scale: 0; } }
  .col.active .dot { opacity: 1; }
  .dot.labelled { stroke: var(--fg); stroke-width: 1; stroke-opacity: 0.55; }
  .dot.dim { opacity: 0.14; }
  .dot.lit { opacity: 1; scale: 1.6; }
  .dot.pointed { scale: 2.1; opacity: 1; }
  .dot.selected { opacity: 1; stroke: var(--fg); stroke-width: 1.6; stroke-opacity: 1; scale: 1.7; }
  .texture { image-rendering: pixelated; animation: texture-in 900ms var(--ease-enter) backwards; }
  @keyframes texture-in { from { opacity: 0; clip-path: inset(0 100% 0 0); } }
  .cell-ring { fill: none; stroke: var(--fg); stroke-width: 1.5; pointer-events: none; }
  .ripple { fill: none; stroke: var(--c); stroke-width: 2; opacity: 0; pointer-events: none; transform-box: fill-box; transform-origin: center; animation: ripple 900ms var(--ease-out); }
  @keyframes ripple { from { opacity: 0.9; scale: 0.6; } to { opacity: 0; scale: 3.2; } }

  .layer-num { fill: var(--fg-muted); font-family: var(--font-mono); text-anchor: middle; font-variant-numeric: tabular-nums; transition: fill var(--dur-fast) var(--ease-out); }
  .layer-num.active { fill: var(--fg); font-weight: var(--weight-data-bold); }
  .band line { stroke: var(--grid-line); stroke-width: 3; stroke-linecap: round; transition: stroke var(--dur) var(--ease-out); }
  .band text { fill: var(--fg-muted); font-family: var(--font-reading); text-anchor: middle; transition: fill var(--dur) var(--ease-out); }
  .band.active line { stroke: var(--neuron-tint); }
  .band.active text { fill: var(--fg); }

  .tip {
    position: absolute; left: 0; top: 0; z-index: 3; display: grid; gap: var(--space-1); width: max-content; max-width: 18rem;
    padding: var(--space-3) var(--space-4); border-radius: var(--radius); background: var(--bg-elev); box-shadow: var(--shadow-card);
    color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.4; pointer-events: none;
    transform: translate(calc(var(--x) + 16px), calc(var(--y) - 50%));
    animation: tip-in 140ms var(--ease-out) backwards;
  }
  .tip.flip { transform: translate(calc(var(--x) - 100% - 16px), calc(var(--y) - 50%)); }
  @keyframes tip-in { from { opacity: 0; } }
  .tip strong { color: var(--fg); font-family: var(--font-mono); font-size: var(--text-md); }
  .tip-line { display: flex; gap: var(--space-2); }
  .tip-over { display: grid; gap: 0; color: var(--fg); }
  .tip-over em { color: var(--fg-muted); font-size: var(--text-xs); font-style: normal; }
  .tip-hint { color: var(--fg-muted); font-size: var(--text-xs); }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }

  @media (prefers-reduced-motion: reduce) {
    .ready .dot, .ready .scan, .ripple, .tip, .texture { animation: none; }
    .dot, .col-bg, .layer-num, .band line, .band text { transition: none; }
    .dot { scale: 1; }
  }
  @media (forced-colors: active) { .dot { fill: CanvasText; } .col-bg.active { stroke: Highlight; } }
</style>
