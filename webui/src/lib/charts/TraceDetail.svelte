<script lang="ts">
  import Slider from "../Slider.svelte";
  import MorphText from "../ui/MorphText.svelte";
  import Sparkline from "./Sparkline.svelte";
  import { chartValue } from "./chartValues";
  let { points, percentage = false }: { points: (number | null)[]; percentage?: boolean } = $props();
  let index = $state<number | null>(null);
  let paused = $state<(number | null)[] | null>(null);
  const displayPoints = $derived(paused ?? points);
  const selected = $derived(Math.min(index ?? displayPoints.length - 1, displayPoints.length - 1));
  const value = $derived(displayPoints[selected]);
</script>

{#if displayPoints.length > 1}
  <details class="trace-detail">
    <summary>Inspect trace</summary>
    <div class="trace-plot">
      <Sparkline points={displayPoints} width={280} height={72} cap={Infinity} {percentage} />
      <span class="trace-cursor" style:left={`${selected / (displayPoints.length - 1) * 100}%`} aria-hidden="true"></span>
    </div>
    <div class="trace-readout"><span class="trace-mode">{paused ? "Paused trace" : "Live trace"}</span> <MorphText text={`Sample ${selected + 1} / ${displayPoints.length}`} /> · <MorphText text={value == null ? "Unavailable" : chartValue(value, percentage)} /></div>
    <Slider value={selected} min={0} max={displayPoints.length - 1} step={1} ariaLabel="Trace sample" displayValue={`Sample ${selected + 1}`} oninput={v => { paused ??= points.slice(); index = v; }} />
    <button type="button" onclick={() => { index = null; paused = null; }}>Follow latest</button>
  </details>
{/if}

<style>
  .trace-mode { display: block; color: var(--fg-muted); font-family: var(--font-ui); }
  .trace-detail { min-width: 0; color: var(--fg-dim); font-size: var(--text-xs); }
  summary { cursor: pointer; min-height: 44px; align-content: center; }
  .trace-plot { position: relative; margin: var(--space-2) var(--space-xs); }
  .trace-plot :global(svg) { width: 100%; height: 72px; }
  .trace-cursor { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--accent); }
  .trace-readout { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
  button { min-height: 44px; border: 0; background: transparent; color: var(--accent); cursor: pointer; }
</style>
