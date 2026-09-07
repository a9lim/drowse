<script lang="ts">
  import Select from "../../lib/Select.svelte";
  import Bar from "../../lib/charts/Bar.svelte";
  import { probeAxisScale } from "../../lib/stores.svelte";
  import type { ProbeReadingJSON } from "../../lib/types";

  let { name, reading, axisLabels }: {
    name: string;
    reading: ProbeReadingJSON;
    axisLabels: string[];
  } = $props();
  let requestedLayer = $state("");
  const layers = $derived(Object.keys(reading.coords_per_layer ?? {})
    .filter(layer => Number.isSafeInteger(Number(layer)) && Number(layer) >= 0 && reading.coords_per_layer[layer]?.length > 0)
    .sort((a, b) => Number(a) - Number(b)));
  const selectedLayer = $derived(layers.includes(requestedLayer) ? requestedLayer : layers[0]);
  const coords = $derived(reading.coords_per_layer?.[selectedLayer] ?? []);
  const fraction = $derived(reading.fraction_per_layer?.[selectedLayer]);
</script>

{#if layers.length > 0}
  <section class="layer-readings" aria-label={`${name} layer readings`}>
    <header>
      <span>Layer readings</span>
      <Select
        value={selectedLayer}
        options={layers.map(layer => ({ value: layer, label: `Layer ${layer}` }))}
        onchange={layer => { requestedLayer = layer; }}
        ariaLabel={`${name} layer`}
      />
    </header>
    <div class="layer-values" aria-live="polite" aria-atomic="true">
      {#each coords as coord, axis}
        <div class="layer-coordinate" role="group" aria-label={`${name} layer ${selectedLayer} axis ${axis}`}>
          <span>{axisLabels[axis] ?? `c${axis}`}</span>
          <Bar value={coord} max={probeAxisScale(name, axis)} bipolar />
          <output>{coord.toFixed(3)}</output>
        </div>
      {/each}
      {#if fraction != null}<p>Subspace fraction <output>{fraction.toFixed(3)}</output></p>{/if}
    </div>
  </section>
{/if}

<style>
  .layer-readings { padding: var(--surface-padding); margin-top: var(--space-4); background: var(--input-well); border-radius: var(--radius-sm); }
  header { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: var(--space-3); margin-bottom: var(--space-4); }
  header > span { color: var(--fg); font-size: var(--text-sm); }
  .layer-values { display: grid; gap: var(--space-3); }
  .layer-coordinate { display: grid; grid-template-columns: minmax(0, 1fr) minmax(32px, 2fr) auto; align-items: center; gap: var(--space-3); }
  .layer-coordinate > span { overflow-wrap: anywhere; }
  .layer-coordinate :global(.bar) { width: 100%; min-width: 0; }
  .layer-coordinate, p { color: var(--fg-muted); font-size: var(--text-xs); }
  p { display: flex; justify-content: space-between; gap: var(--space-3); margin: 0; }
  output { color: var(--fg); font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
</style>
