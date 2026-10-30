<script lang="ts">
  import { LENSES, depthColor, planFor, type ReadColumn } from "./data";

  let { layer, onlayer }: { layer: number; onlayer: (layer: number) => void } = $props();

  const rows: ReadColumn[] = ["read_S", "read_S_z", "read_R_dclm", "read_R_dclm_z", "write_S_z", "write_R_dclm", "write_R_dclm_z"];
  const layers = Array.from({ length: 36 }, (_, L) => L);
  const plans = layers.map(planFor);
  function level(col: ReadColumn, L: number): 0 | 1 | 2 {
    const plan = plans[L];
    if (plan.ins[0] === col || plan.outs[0] === col) return 2;
    return plan.ins.includes(col) || plan.outs.includes(col) ? 1 : 0;
  }
</script>

<div class="map-wrap">
  <table class="map">
    <caption>Which lens to trust at each depth: <span class="c-key c2"></span> best <span class="c-key c1"></span> informative <span class="c-key c0"></span> weak</caption>
    <thead>
      <tr>
        <th scope="col"><span class="sr">Lens</span></th>
        {#each layers as L}
          <!-- Only the labelled columns are buttons, so each target stays large enough; the ribbon reaches every layer. -->
          <th scope="col" class:active={L === layer} class:minor={L % 4 !== 0 && L !== 35}>{#if L % 4 === 0 || L === 35}<button type="button" aria-label={`Layer ${L}`} aria-pressed={L === layer} onclick={() => onlayer(L)}>{L}</button>{:else}<span class="col-num">{L}</span>{/if}</th>
        {/each}
      </tr>
    </thead>
    <tbody>
      {#each rows as col}
        <tr class:output={col.startsWith("write")}>
          <th scope="row"><span class="short">{LENSES[col].short}</span><span class="long">{LENSES[col].name}</span></th>
          {#each layers as L}
            {@const v = level(col, L)}
            <td class={`c${v}`} class:active={L === layer} style:--c={depthColor(L)}>
              <span class="cell" aria-hidden="true"></span><span class="sr">{v === 2 ? "best" : v === 1 ? "informative" : "weak"}</span>
            </td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .map-wrap { position: relative; min-width: 0; overflow-x: auto; scrollbar-width: thin; }
  .map { width: 100%; min-width: 44rem; border-collapse: separate; border-spacing: 2px; table-layout: fixed; }
  caption { margin-bottom: var(--space-3); color: var(--fg-muted); font-size: var(--text-xs); text-align: start; }
  .c-key { display: inline-block; width: 10px; height: 10px; margin-inline: var(--space-2) 0; border-radius: var(--data-mark-radius); vertical-align: -1px; }
  .c-key.c2 { background: var(--neuron-tint); }
  .c-key.c1 { background: transparent; box-shadow: inset 0 0 0 2px var(--neuron-tint); }
  .c-key.c0 { width: 4px; height: 4px; border-radius: 50%; background: var(--fg-muted); vertical-align: 1px; }
  thead th { padding: 0; font-weight: normal; }
  thead th:first-child { width: 15rem; }
  thead button { width: 100%; min-height: var(--control-target); padding: 0; border: 0; border-radius: var(--radius-sm); background: transparent; color: var(--fg-muted); font-family: var(--font-mono); font-size: var(--text-2xs); font-variant-numeric: tabular-nums; cursor: pointer; }
  thead th { position: relative; }
  thead th:not(.minor) button { position: absolute; inset: 0 auto 0 50%; width: max(28px, 100%); translate: -50% 0; }
  .col-num { display: grid; place-items: center; min-height: var(--control-target); color: var(--fg-muted); font-family: var(--font-mono); font-size: var(--text-2xs); }
  thead th.active .col-num { color: var(--fg); }
  thead button:hover { background: var(--bg-hover); color: var(--fg); }
  thead button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 1px; }
  thead th.active button { color: var(--fg); font-weight: var(--weight-data-bold); background: var(--glass-strong); }
  tbody th { display: flex; align-items: baseline; gap: var(--space-3); padding: 0 var(--space-3) 0 0; text-align: start; white-space: nowrap; font-weight: normal; }
  tbody tr:nth-child(5) > * { padding-top: var(--space-3); }
  .short { min-width: 6.5ch; color: var(--fg); font-family: var(--font-mono); font-size: var(--text-xs); }
  .long { color: var(--fg-muted); font-size: var(--text-xs); }
  td { height: 22px; padding: 0; text-align: center; vertical-align: middle; }
  .cell { display: block; height: 18px; margin-inline: auto; border-radius: var(--data-mark-radius); transition: transform var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out); }
  .c2 .cell { background: var(--c); }
  .c1 .cell { background: transparent; box-shadow: inset 0 0 0 2px var(--c); }
  .c0 .cell { width: 4px; height: 4px; border-radius: 50%; background: var(--fg-muted); opacity: 0.6; }
  td.active .cell { transform: scaleY(1.15); }
  .sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
  @media (prefers-reduced-motion: reduce) { .cell { transition: none; } }
  @media (forced-colors: active) { .c2 .cell { background: CanvasText; } .c1 .cell { border: 1px solid CanvasText; } }
</style>
