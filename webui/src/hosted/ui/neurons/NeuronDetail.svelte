<script lang="ts">
  import FluentIcon from "../../../lib/ui/FluentIcon.svelte";
  import MorphText from "../../../lib/ui/MorphText.svelte";
  import InfoTip from "../../../lib/ui/InfoTip.svelte";
  import PoleColumn from "./PoleColumn.svelte";
  import TitleText from "./TitleText.svelte";
  import { MIDDLE_CAUTION, bandOf, neuronId, type NeuronDetail } from "./data";

  let { neuron, layer, onstep, position, total, hrefFor }: {
    neuron: NeuronDetail;
    layer: number;
    /** Move to the previous (-1) or next (+1) shown neuron in the layer. */
    onstep: (delta: -1 | 1) => void;
    position: number;
    total: number;
    hrefFor: (delta: -1 | 1) => string;
  } = $props();

  const clearer = $derived(neuron.poles[0].score >= neuron.poles[1].score ? 0 : 1);
  // On phones the two poles share one column; start on the clearer one.
  let shown = $state<0 | 1>(0);
  let lastNeuron = -1;
  $effect.pre(() => {
    if (neuron.n !== lastNeuron) { lastNeuron = neuron.n; shown = clearer as 0 | 1; }
  });

  // Tabs only on phones, where one pole shows at a time; on wider screens both are plain columns.
  // Measured on the pane itself, since the rail beside it takes a varying share of the window.
  let width = $state(1000);
  const narrow = $derived(width < 720);

  // Arrow keys move between the two pole tabs (APG tabs pattern).
  function tabKey(event: KeyboardEvent) {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    shown = event.key === "Home" ? 0 : event.key === "End" ? 1 : (shown === 0 ? 1 : 0);
    document.getElementById(`pole-tab-${shown}`)?.focus();
  }

  // The one-line overview: the clearer side's top input and output words.
  const overview = $derived.by((): [string, string] => {
    if (neuron.overview) return neuron.overview;
    const pole = neuron.poles[clearer];
    const words = (prefix: string) => {
      const read = pole.reads.find((r) => r.trust === "primary" && r.col.startsWith(prefix));
      return (read?.tokens ?? []).map(([token]) => token.trim()).filter((word) => word && !word.startsWith("\u0001") && /[\p{L}\p{N}]/u.test(word)).slice(0, 4).join(" · ");
    };
    return [words("read"), words("write")];
  });

  let copied = $state(false);
  async function copyLink() {
    try {
      await navigator.clipboard.writeText(window.location.href);
      copied = true;
      setTimeout(() => (copied = false), 1600);
    } catch { /* clipboard blocked: the address bar still has the link */ }
  }

  function step(event: MouseEvent, delta: -1 | 1) {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    onstep(delta);
  }
</script>

<section class="detail" class:narrow aria-labelledby="neuron-title" data-link-scope bind:clientWidth={width}>
  <div class="detail-bar">
    <div class="id-row">
      <h2 id="neuron-title" tabindex="-1"><MorphText text={neuronId(layer, neuron.n)} duration={340} /></h2>
      {#if bandOf(layer) === "dark"}<span class="caution"><InfoTip icon="error" label="Why layers 13–20 are less reliable" text={MIDDLE_CAUTION} /></span>{/if}
    </div>
    <div class="stepper">
      <span class="position" aria-live="polite"><MorphText text={`${(position + 1).toLocaleString("en-US")} / ${total.toLocaleString("en-US")}`} duration={240} /></span>
      <a class="icon-button" href={hrefFor(-1)} aria-label="Previous neuron" onclick={(event) => step(event, -1)}><FluentIcon name="up" size={18} /></a>
      <a class="icon-button" href={hrefFor(1)} aria-label="Next neuron" onclick={(event) => step(event, 1)}><FluentIcon name="down" size={18} /></a>
      <button type="button" class="icon-button" aria-label={copied ? "Link copied" : "Copy link to this neuron"} onclick={copyLink}>
        <span class="swap-icon" class:copied><span class="i-copy"><FluentIcon name="copy" size={18} /></span><span class="i-check"><FluentIcon name="check" size={18} /></span></span>
      </button>
    </div>
  </div>

  <header class="detail-head">
    {#if overview[0] || overview[1]}
      <dl class="overview">
        <div><dt>Triggered by:</dt><dd><TitleText text={overview[0]} fallback="unclear" interactive /></dd></div>
        <div><dt>Steers outputs:</dt><dd><TitleText text={overview[1]} fallback="unclear" interactive /></dd></div>
      </dl>
    {/if}
    <p class="titles">
      <span><span class="sign">+</span><TitleText text={neuron.titles[0]} /></span>
      <span><span class="sign">−</span><TitleText text={neuron.titles[1]} /></span>
    </p>
  </header>

  <div class="pole-switch" role="tablist" aria-label="Side" hidden={!narrow}>
    {#each [0, 1] as index}
      <button type="button" role="tab" id={`pole-tab-${index}`} aria-selected={shown === index} aria-controls={`pole-panel-${index}`} tabindex={shown === index ? 0 : -1} onclick={() => (shown = index as 0 | 1)} onkeydown={tabKey}>
        {index === 0 ? "+ Positive side" : "− Negative side"}{#if clearer === index}<span class="clear-dot" aria-hidden="true"></span><span class="sr-only">, clearer</span>{/if}
      </button>
    {/each}
  </div>

  {#key neuron.n}
  <div class="poles">
    {#each neuron.poles as pole, index (pole.pole)}
      <div class="pole-panel" class:hidden-phone={shown !== index} id={`pole-panel-${index}`} role={narrow ? "tabpanel" : undefined} aria-labelledby={narrow ? `pole-tab-${index}` : undefined}>
        <PoleColumn {pole} {layer} clearer={clearer === index} />
      </div>
    {/each}
  </div>
  {/key}
</section>

<style>
  .detail { display: grid; gap: var(--space-6); min-width: 0; }
  /* Stepping to another neuron slides the new one in; it plays once per change, never on hover. */
  .poles { animation: poles-in 380ms var(--ease-enter) backwards; }
  @keyframes poles-in { from { opacity: 0; translate: 0 10px; } }
  .detail-bar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--space-2) var(--space-3); }
  .icon-button {
    display: inline-grid; place-items: center; width: 36px; height: 36px; border: 0; border-radius: var(--radius);
    background: var(--glass); color: var(--fg-dim); font: inherit; text-decoration: none; cursor: pointer;
    transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out);
  }
  .icon-button::after { content: ""; position: absolute; inset: -4px; }
  .icon-button { position: relative; }
  .icon-button:hover { background: var(--glass-strong); color: var(--fg); }
  .icon-button:active { scale: var(--press-scale); }
  .swap-icon { display: grid; }
  .swap-icon > span { grid-area: 1 / 1; display: grid; transition: opacity var(--dur-slow) cubic-bezier(0.2, 0, 0, 1), scale var(--dur-slow) cubic-bezier(0.2, 0, 0, 1), filter var(--dur-slow) cubic-bezier(0.2, 0, 0, 1); }
  .i-check, .copied .i-copy { opacity: 0; scale: 0.25; filter: blur(4px); }
  .copied .i-check { opacity: 1; scale: 1; filter: blur(0); }
  .icon-button:focus-visible, .pole-switch button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .stepper { display: flex; align-items: center; gap: var(--space-2); }
  .position { margin-inline-end: var(--space-2); color: var(--fg-muted); font-size: var(--text-xs); font-variant-numeric: tabular-nums; }

  .overview { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-3); margin: 0; }
  .overview > div { display: grid; gap: var(--space-1); padding: var(--space-4) var(--space-5); border-radius: var(--radius-lg); background: color-mix(in oklch, var(--neuron-tint) 10%, var(--neuron-chip-bg)); box-shadow: inset 0 0 0 1px color-mix(in oklch, var(--neuron-tint) 35%, transparent); }
  .overview dt { color: var(--neuron-tint-ink); font-size: var(--text-sm); font-weight: var(--weight-structure-bold); }
  .overview dd { margin: 0; color: var(--fg); font-family: var(--font-mono); font-size: var(--text-md); overflow-wrap: anywhere; }
  .narrow .overview { grid-template-columns: minmax(0, 1fr); }
  .detail-head { display: grid; gap: var(--space-3); }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  .id-row { display: flex; align-items: center; gap: var(--space-1); }
  .caution { display: inline-flex; }
  .caution :global(.info-trigger) { color: var(--accent-yellow); }
  h2 { margin: 0; color: var(--fg); font-family: var(--font-mono); font-weight: var(--weight-data-medium); font-size: var(--text-feature-title); line-height: 1.15; letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }
  h2:focus { outline: none; }
  .titles { display: grid; gap: var(--space-1); margin: 0; color: var(--fg); font-size: var(--text-lg); line-height: 1.4; }
  .titles > span { display: flex; gap: var(--space-2); min-width: 0; }
  .titles > span + span { color: var(--fg-dim); }
  .sign { flex: none; width: 1ch; color: var(--fg-muted); font-weight: var(--weight-structure-bold); }

  .poles { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-8); align-items: start; }
  .pole-switch { display: none; }
  .narrow {
    .poles { grid-template-columns: minmax(0, 1fr); }
    .pole-switch:not([hidden]) { position: sticky; top: var(--neuron-sticky-top); z-index: 2; display: grid; grid-template-columns: 1fr 1fr; gap: 0; padding: 0; border-radius: var(--radius); background: var(--bg-elev); }
    .pole-switch button { display: inline-flex; align-items: center; justify-content: center; gap: var(--space-2); min-height: var(--control-target); border: 0; border-radius: var(--radius); background: transparent; color: var(--fg-dim); font: inherit; font-size: var(--text-sm); cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out); }
    .pole-switch button[aria-selected="true"] { background: var(--glass-bright); color: var(--fg); }
    .clear-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--neuron-tint); }
    .hidden-phone { display: none; }
  }
  @media (prefers-reduced-motion: reduce) { .poles { animation: none; } .icon-button, .pole-switch button { transition: none; } .icon-button:active { scale: none; } .swap-icon > span { transition: opacity var(--dur-fast) linear; } .i-check, .copied .i-copy, .copied .i-check { scale: 1; filter: none; } }
</style>
