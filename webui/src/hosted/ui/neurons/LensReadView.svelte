<script lang="ts">
  import MorphText from "../../../lib/ui/MorphText.svelte";
  import Token from "./Token.svelte";
  import { LENSES, type LensRead } from "./data";

  let { read, twinOf, compact = false }: {
    read: LensRead;
    /** The read whose twin this is, when only extra tokens are listed. */
    twinOf?: string;
    /** The main view: the strongest tokens large, the rest behind "more", and strength as a word. */
    compact?: boolean;
  } = $props();

  const info = $derived(LENSES[read.col]);
  const top = $derived(read.tokens.filter((token) => token[2] < 8));
  const middle = $derived(read.tokens.filter((token) => token[2] >= 8 && token[2] < 24));
  const deep = $derived(read.tokens.filter((token) => token[2] >= 24));
  // Strength is a percentile of the layer's poles: under about the 20th it is usually the noise floor.
  const strength = $derived(read.pct >= 90 ? "strong" : read.pct >= 60 ? "clear" : read.pct >= 20 ? "faint" : "noise");
  const strengthWord = $derived({ strong: "Strong signal", clear: "Clear signal", faint: "Faint signal", noise: "Mostly noise" }[strength]);
  const hidden = $derived(compact ? [...middle, ...deep] : deep);
  let open = $state(false);
  const ordinal = new Intl.PluralRules("en", { type: "ordinal" });
  const suffixes: Record<string, string> = { one: "st", two: "nd", few: "rd", other: "th" };
  const suffix = $derived(suffixes[ordinal.select(read.pct)] ?? "th");
</script>

<div class={`read ${read.trust}`}>
  <div class="read-head">
    <span class="lens">{info.short}</span>
    {#if !compact}<span class="name">{read.twin && twinOf ? `${info.short} adds` : info.name}</span>{/if}
    <span class="meter" role="img" aria-label={`${strengthWord}: ${read.pct}${suffix} percentile of the layer`}>
      <span class="bar"><span class="fill" style:--v={Math.max(read.pct, 3) / 100}></span></span>
      {#if compact}
        <span class={`strength word ${strength}`}><MorphText text={strengthWord} numbers={false} duration={260} /></span>
      {:else}
        <span class={`strength ${strength}`}><MorphText text={read.pct} duration={300} /><span class="th">{suffix}</span></span>
      {/if}
    </span>
  </div>
  {#if read.tokens.length === 0}
    <p class="empty">{read.twin ? `Nothing beyond ${twinOf ?? "the read above"}` : "Only common tokens"}</p>
  {:else}
    {#if top.length}
      <div class="tier tier-top">{#each top as [token, count, , flags]}<Token {token} {count} {flags} size={compact || read.trust === "primary" ? "lg" : "md"} />{/each}</div>
    {/if}
    {#if middle.length && !compact}
      <div class="tier">{#each middle as [token, count, , flags]}<Token {token} {count} {flags} size="sm" />{/each}</div>
    {/if}
    {#if hidden.length}
      {#if open}
        <div class="tier deep">{#each hidden as [token, count, , flags]}<Token {token} {count} {flags} size="sm" />{/each}</div>
      {/if}
      <button type="button" class="more" aria-expanded={open} onclick={() => (open = !open)}>
        {open ? "Show fewer" : `${hidden.length} more`}
      </button>
    {/if}
  {/if}
</div>

<style>
  .read { display: grid; gap: var(--space-2); min-width: 0; padding-block: var(--space-3); }
  .read:first-child { padding-top: 0; }
  .read-head { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2) var(--space-3); }
  .lens {
    display: inline-flex; align-items: center; min-height: 24px; padding-inline: var(--space-2);
    border-radius: var(--radius-sm); background: color-mix(in oklch, var(--neuron-tint) 18%, transparent);
    color: var(--fg); font-family: var(--font-mono); font-size: var(--text-xs); white-space: nowrap; 
  }
  .weak .lens, .compare .lens { background: var(--glass); color: var(--fg-dim); }
  .name { color: var(--fg-dim); font-size: var(--text-xs); }
  .meter { display: inline-flex; align-items: center; gap: var(--space-2); margin-inline-start: auto; }
  .bar { position: relative; width: 56px; height: 4px; border-radius: var(--radius-pill); background: var(--glass-strong); overflow: hidden; }
  .fill { position: absolute; inset: 0; border-radius: inherit; background: var(--neuron-tint); transform-origin: left; scale: var(--v) 1; transition: scale var(--dur-slow) var(--ease-out); }
  .strength { min-width: 3.2ch; color: var(--fg-dim); font-family: var(--font-mono); font-size: var(--text-xs); font-variant-numeric: tabular-nums; text-align: end; }
  .strength.strong { color: var(--fg); }
  .strength.noise { color: var(--fg-muted); }
  .th { font-size: var(--text-2xs); }
  .strength.word { min-width: 0; font-family: var(--font-reading); }
  .tier-top { animation: tier-in 360ms var(--ease-enter) backwards; }
  @keyframes tier-in { from { opacity: 0; translate: 0 4px; } }
  .tier { display: flex; flex-wrap: wrap; gap: var(--space-1); min-width: 0; }
  .tier-top { gap: var(--space-1); }
  .deep { animation: reveal var(--dur-slow) var(--ease-enter) backwards; }
  @keyframes reveal { from { opacity: 0; transform: translateY(-4px); } }
  .empty { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); }
  .more {
    position: relative; justify-self: start; min-height: 32px; margin-inline-start: calc(var(--space-3) * -1); padding: 0 var(--space-3); border: 0; border-radius: var(--radius-sm);
    background: transparent; color: var(--fg-dim); font: inherit; font-size: var(--text-xs); cursor: pointer;
    transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out);
  }
  .more::after { content: ""; position: absolute; inset: calc(var(--space-xs) * -0.5) 0; }
  .more:hover { background: var(--bg-hover); color: var(--fg); }
  .more:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (prefers-reduced-motion: reduce) { .deep, .tier-top { animation: none; } .fill { transition: none; } .more { transition: none; } }
</style>
