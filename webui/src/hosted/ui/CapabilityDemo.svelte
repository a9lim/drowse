<script lang="ts">
  import MorphText from "../../lib/ui/MorphText.svelte";
  import Slider from "../../lib/Slider.svelte";
  import SegmentedTabs from "../../lib/ui/SegmentedTabs.svelte";
  let mode = $state("steering");
  let alpha = $state(0);
  let branch = $state("a");
  const options = [{value: "steering", label: "Steer"}, {value: "prediction", label: "Inspect"}, {value: "branches", label: "Compare"}];
  const phrase = $derived(alpha < .34 ? "The sea is calm." : alpha < .67 ? "The sea glows in the evening light." : "The sea wears a silver veil of moonlight.");
  const probability = $derived(Math.round(35 + alpha * 45));
</script>

<section class="capability-demo" aria-label="Illustrative Drowse example">
  <header><span class="demo-label">Interactive example · illustrative data</span><SegmentedTabs items={options} bind:value={mode} ariaLabel="Example capability" /></header>
  {#if mode === "branches"}
    <p class="context">The door opened onto</p>
    <div class="branch-options" role="group" aria-label="Example continuation">
      <button type="button" aria-pressed={branch === "a"} onclick={() => branch = "a"}>A · a quiet garden.</button>
      <button type="button" aria-pressed={branch === "b"} onclick={() => branch = "b"}>B · a crowded station.</button>
    </div>
    <p class="result"><span class="branch-indicator" aria-hidden="true">↳</span> <MorphText text={branch === "a" ? "a quiet garden." : "a crowded station."} numbers={false} duration={240} /></p>
    <p class="demo-note">A shared prefix, two saved continuations. Select a branch to compare.</p>
  {:else}
    <label class="alpha"><span>Poetic influence</span><span>α <MorphText text={alpha.toFixed(2)} /></span></label>
    <Slider value={alpha} min={0} max={1} step={.01} oninput={value => alpha = value} ariaLabel="Example poetic influence" />
    {#if mode === "steering"}
      <p class="result"><MorphText text={phrase} numbers={false} duration={240} /></p>
    {:else}
      <div class="distribution" aria-label={`Illustrative probability: glows ${probability}%, is ${100 - probability}%`}><span style:width={`${probability}%`}></span><span style:width={`${100-probability}%`}></span></div>
      <p class="probabilities">glows <MorphText text={`${probability}%`} /> <span>is <MorphText text={`${100 - probability}%`} /></span></p>
    {/if}
    <p class="demo-note">This example explains the controls. Model results depend on the prompt, model, and steering direction.</p>
  {/if}
</section>

<style>
  .capability-demo { margin-top: 32px; padding: clamp(16px, 4vw, 32px); background: var(--surface-sheen), var(--bg-alt); border-radius: var(--radius-lg); border: 1px solid var(--glass-line); min-width: 0; color: var(--fg); }
  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 24px; }
  .demo-label, .demo-note { color: var(--fg-muted); font-size: var(--text-sm); line-height: 1.6; }
  .alpha, .probabilities { display: flex; justify-content: space-between; gap: 16px; font-variant-numeric: tabular-nums; }
  .result { min-height: 3.6em; font-size: var(--text-page-title); line-height: 1.6; overflow-wrap: anywhere; margin-block: 24px; }
  .context { font-size: var(--text-lg); }
  .branch-options { display: flex; flex-wrap: wrap; gap: 8px; }
  button { min-height: 44px; background: var(--input-well); color: var(--fg); padding: var(--space-2) var(--space-3); border-radius: var(--radius); border: 1px solid transparent; cursor: pointer; text-align: start; }
  button[aria-pressed=true] { background: var(--accent-subtle); border-color: var(--accent); }
  .branch-indicator { color: var(--accent); }
  .distribution { display: flex; height: 18px; overflow: hidden; border-radius: var(--radius-pill); margin-top: 24px; }
  .distribution span:first-child { background: var(--accent); }
  .distribution span:last-child { background: var(--fg-muted); }
</style>
