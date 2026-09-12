<script lang="ts">
  import { pageTransition } from "../../lib/motion";
  import MorphText from "../../lib/ui/MorphText.svelte";
  import DemoReply from "./DemoReply.svelte";
  import Slider from "../../lib/Slider.svelte";
  import SegmentedTabs from "../../lib/ui/SegmentedTabs.svelte";
  import RecordedTokenText from "./RecordedTokenText.svelte";
  import recording from "../data/neighbor-example.json";

  let mode = $state("steering");
  let setting = $state(3);
  let coloring = $state("surprisal");
  const promptPreview = recording.prompt.split(" Write your reply")[0];
  const options = [{ value: "steering", label: "Steer" }, { value: "prediction", label: "Inspect" }];
  const colorOptions = [{ value: "surprisal", label: "Surprisal" }, { value: "probability", label: "Probability" }];
  const run = $derived(recording.runs[setting]);
  const tokens = $derived(run.tokens.filter(token => token.text.length > 0));
  const alphaLabel = (alpha: number) => alpha === 0 ? "No steering" : alpha < 0 ? "Toward detached" : "Toward welcoming";

  function fitContent(node: HTMLElement) {
    const panel = node.parentElement!;
    const measure = () => { panel.style.height = `${node.getBoundingClientRect().height}px`; };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return { destroy() { observer.disconnect(); } };
  }

</script>

<section class="capability-demo" aria-label="Recorded Drowse example">
  <div class="demo-content" use:fitContent>
  <header>
    <SegmentedTabs items={options} bind:value={mode} ariaLabel="Example capability" />
  </header>
  <p class="context">{promptPreview}</p>
    <div class="alpha"><span>Welcoming influence</span><span><MorphText text={`${alphaLabel(run.alpha)} · α ${run.alpha.toFixed(2)}`} /></span></div>
    <Slider value={setting} min={0} max={recording.runs.length - 1} step={1} oninput={value => setting = value}
      ariaLabel="Recorded welcoming setting" title={`${alphaLabel(run.alpha)}, alpha ${run.alpha.toFixed(2)}`} displayValue={`α ${run.alpha.toFixed(2)}`} />
    <div class="scale-labels" aria-hidden="true"><span>Detached</span><span>Unsteered</span><span>Welcoming</span></div>
  <div class="mode-stage">
  {#key mode}
  <div class="mode-content" transition:pageTransition={{}}>
    {#if mode === "steering"}
      <p class="result"><DemoReply text={run.text} /></p>
    {:else}
      <div class="inspect-toolbar">
        <p class="inspect-hint">Hover or click a token to inspect.</p>
        <SegmentedTabs items={colorOptions} bind:value={coloring} ariaLabel="Token coloring" />
      </div>
      <div class="result token-text">
        <RecordedTokenText {tokens} layout="inline" {coloring} />
      </div>
    {/if}
  </div>
  {/key}
  </div>
  </div>
</section>

<style>
  .capability-demo { min-width: 0; font: var(--weight-reading) var(--text-md)/1.6 var(--font-reading); text-shadow: none; transition: height 280ms cubic-bezier(0.2, 0, 0, 1); }
  .demo-content { display: flow-root; }
  .mode-stage { position: relative; }
  .mode-content { display: flow-root; }
  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 24px; }
  .inspect-hint, .scale-labels { color: var(--fg-muted); font-size: var(--text-sm); line-height: 1.6; }
  .context { max-width: 70ch; font: inherit; margin: 0 0 24px; text-wrap: pretty; }
  .alpha, .scale-labels { display: flex; justify-content: space-between; gap: 16px; font-variant-numeric: tabular-nums; }
  .alpha, .result,
  .alpha :global(.morph-text), .result :global(.morph-text),
  .alpha :global(.morph-source), .result :global(.morph-source) { color: var(--fg); }
  .alpha { flex-wrap: wrap; align-items: baseline; row-gap: var(--space-2); font-size: var(--text-sm); margin-bottom: calc(var(--space-8) + var(--space-2)); line-height: 1.6; -webkit-user-select: none; user-select: none; }
  .alpha > span { min-width: 0; max-width: 100%; overflow-wrap: anywhere; }
  .alpha :global(.morph-paint) { overflow: visible; }
  .capability-demo :global(.slider-bubble) { font-family: var(--font-reading); white-space: nowrap; }
  .capability-demo :global(.sk-tabs .tab) { min-width: 80px; min-height: 48px; padding: 8px 16px; font-family: var(--font-reading) !important; font-size: var(--text-md); font-weight: var(--weight-reading-medium); line-height: 1.5; letter-spacing: normal; }
  .capability-demo :global(.recorded-token-panel) { text-shadow: none; }
  .scale-labels { -webkit-user-select: none; user-select: none; }
  .result { max-width: 70ch; font: inherit; font-weight: var(--weight-bold); text-wrap: pretty; overflow-wrap: anywhere; margin: 24px 0 0; white-space: pre-wrap; }
  .inspect-toolbar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: var(--space-2); margin-top: var(--space-6); }
  .inspect-hint { margin: 0; }
  .token-text { margin-block: var(--space-4) var(--space-6); white-space: normal; }
  @media (prefers-reduced-motion: reduce) {
    .capability-demo { transition: none; }
  }
  @media (max-width: 700px) {
    .capability-demo :global(.sk-tabs) { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); width: 100%; }
    .capability-demo :global(.sk-tabs .tab) { padding-inline: 16px; }
  }
</style>
