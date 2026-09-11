<script lang="ts">
  import { pageTransition } from "../../lib/motion";
  import MorphText from "../../lib/ui/MorphText.svelte";
  import DemoReply from "./DemoReply.svelte";
  import Slider from "../../lib/Slider.svelte";
  import SegmentedTabs from "../../lib/ui/SegmentedTabs.svelte";
  import recording from "../data/neighbor-example.json";
  import recordingUrl from "../data/neighbor-example.json?url";

  let mode = $state("steering");
  let setting = $state(3);
  let tokenIndex = $state(9);
  const promptPreview = recording.prompt.split(" Write your reply")[0];
  const options = [{ value: "steering", label: "Steer" }, { value: "prediction", label: "Inspect" }, { value: "branches", label: "Compare" }];
  const run = $derived(recording.runs[setting]);
  const tokens = $derived(run.tokens.filter(token => token.text.length > 0));
  const step = $derived(Math.min(tokenIndex, tokens.length - 1));
  const token = $derived(tokens[step]);
  const alternatives = $derived.by(() => {
    const rows = [...(token.topAlts ?? [])];
    if (!rows.some(row => row.id === token.tokenId)) {
      rows.push({ id: token.tokenId, text: token.text, logprob: token.logprob });
    }
    return rows.sort((a, b) => b.logprob - a.logprob);
  });
  const remaining = $derived(Math.max(0, 1 - alternatives.reduce((sum, row) => sum + Math.exp(row.logprob), 0)));
  const comparison = recording.runs.filter(run => [-0.1, 0, 0.1].includes(run.alpha));
  const alphaLabel = (alpha: number) => alpha === 0 ? "No steering" : alpha < 0 ? "Toward detached" : "Toward welcoming";
  const percent = (probability: number) => probability > 0 && probability < 0.001 ? "<0.1%" : `${(probability * 100).toFixed(1)}%`;
  const tokenLabel = (text: string) => text.replaceAll("\n", "↵").replaceAll(" ", "·");

  const tokenTone = (logprob: number) => {
    const probability = Math.exp(logprob);
    return probability >= 0.8 ? "likely" : probability >= 0.4 ? "possible" : "unlikely";
  };

  function fitContent(node: HTMLElement) {
    const panel = node.parentElement!;
    const measure = () => { panel.style.height = `${node.getBoundingClientRect().height}px`; };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return { destroy() { observer.disconnect(); } };
  }

  function moveToken(event: KeyboardEvent, index: number) {
    const next = event.key === "ArrowRight" ? Math.min(index + 1, tokens.length - 1)
      : event.key === "ArrowLeft" ? Math.max(index - 1, 0)
      : event.key === "Home" ? 0 : event.key === "End" ? tokens.length - 1 : null;
    if (next === null) return;
    event.preventDefault();
    tokenIndex = next;
    const buttons = (event.currentTarget as HTMLElement).parentElement!.querySelectorAll<HTMLButtonElement>("button");
    buttons[next].focus();
  }
</script>

<section class="capability-demo" aria-label="Recorded Drowse example">
  <div class="demo-content" use:fitContent>
  <header>
    <a class="demo-label" href={recordingUrl} download="drowse-neighbor-recording.json">Recorded preset: a new neighbor</a>
    <SegmentedTabs items={options} bind:value={mode} ariaLabel="Example capability" />
  </header>
  <p class="context">{promptPreview}</p>
  <div class="mode-stage">
  {#key mode}
  <div class="mode-content" transition:pageTransition={{}}>
  {#if mode !== "branches"}
    <div class="alpha"><span>Welcoming influence</span><span><MorphText text={`${alphaLabel(run.alpha)} · α ${run.alpha.toFixed(2)}`} /></span></div>
    <Slider value={setting} min={0} max={recording.runs.length - 1} step={1} oninput={value => setting = value}
      ariaLabel="Recorded welcoming setting" title={`${alphaLabel(run.alpha)}, alpha ${run.alpha.toFixed(2)}`} displayValue={`α ${run.alpha.toFixed(2)}`} />
    <div class="scale-labels" aria-hidden="true"><span>Detached</span><span>Unsteered</span><span>Welcoming</span></div>
    {#if mode === "steering"}
      <p class="result"><DemoReply text={run.text} /></p>
    {:else}
      <p class="inspect-hint">Select a token to inspect.</p>
      <div class="result token-text" role="group" aria-label="Recorded reply tokens">
        {#each tokens as piece, index}
          <button class="token" type="button" data-tone={tokenTone(piece.logprob)} class:whitespace={piece.text.trim().length === 0} aria-pressed={index === step}
            tabindex={index === step ? 0 : -1} aria-label={`Inspect token ${index + 1}: ${tokenLabel(piece.text)}, ${percent(Math.exp(piece.logprob))} probability`}
            onclick={() => tokenIndex = index} onkeydown={event => moveToken(event, index)}><MorphText text={piece.text} numbers={false} duration={240} /></button>
        {/each}
      </div>
      <div class="step-controls">
        <button type="button" disabled={step === 0} onclick={() => tokenIndex = step - 1}>Previous token</button>
        <span aria-live="polite"><MorphText text={`Token ${step + 1} of ${tokens.length}`} /></span>
        <button type="button" disabled={step === tokens.length - 1} onclick={() => tokenIndex = step + 1}>Next token</button>
      </div>
      <table class="distribution">
        <caption><MorphText text={`Before generating “${tokenLabel(token.text)}”`} numbers={false} /></caption>
        <thead><tr><th scope="col">Possible next token</th><th scope="col">Probability</th></tr></thead>
        <tbody>
          {#each alternatives as alternative}
            <tr class:chosen={alternative.id === token.tokenId} data-tone={tokenTone(alternative.logprob)}>
              <td><span class="token-name"><MorphText text={tokenLabel(alternative.text)} numbers={false} /></span>{#if alternative.id === token.tokenId}<span class="chosen-label">Generated</span>{/if}</td>
              <td><span class="probability-bar" aria-hidden="true" style:width={`${Math.exp(alternative.logprob) * 100}%`}></span><span class="probability-value"><MorphText text={percent(Math.exp(alternative.logprob))} /></span></td>
            </tr>
          {/each}
          <tr class="remaining"><td>All other tokens</td><td><MorphText text={percent(remaining)} /></td></tr>
        </tbody>
      </table>
    {/if}
  {:else}
    <div class="comparison">
      {#each comparison as branch}
        <article>
          <h3>{alphaLabel(branch.alpha)} <span>α {branch.alpha.toFixed(2)}</span></h3>
          <p class="branch-reply">{branch.text}</p>
          <button type="button" aria-label={`Inspect ${alphaLabel(branch.alpha).toLowerCase()} reply`} onclick={() => {
            setting = recording.runs.findIndex(run => run.alpha === branch.alpha);
            mode = "prediction";
            tokenIndex = 9;
          }}>Inspect this reply</button>
        </article>
      {/each}
    </div>
  {/if}
  </div>
  {/key}
  </div>
  </div>
</section>

<style>
  .capability-demo { min-width: 0; transition: height 280ms cubic-bezier(0.2, 0, 0, 1); }
  .demo-content { display: flow-root; }
  .mode-stage { position: relative; }
  .mode-content { display: flow-root; }
  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 24px; }
  .demo-label, .inspect-hint, .scale-labels { color: var(--fg-muted); font-size: var(--text-sm); line-height: 1.6; }
  .demo-label { text-decoration: none; min-height: 44px; display: inline-flex; align-items: center; }
  .demo-label:hover { text-decoration: underline; text-underline-offset: 4px; }
  .context { max-width: 70ch; font-size: var(--text-lg); line-height: 1.6; margin: 0 0 24px; text-wrap: pretty; }
  .alpha, .scale-labels { display: flex; justify-content: space-between; gap: 16px; font-variant-numeric: tabular-nums; }
  .alpha { flex-wrap: wrap; -webkit-user-select: none; user-select: none; }
  .scale-labels { -webkit-user-select: none; user-select: none; }
  .result { max-width: 70ch; font-family: var(--font-reading); font-size: var(--text-lg); font-weight: var(--weight-bold); line-height: 1.85; text-wrap: pretty; overflow-wrap: anywhere; margin: 24px 0 0; white-space: pre-wrap; }
  button { min-height: 44px; background: var(--input-well); color: var(--fg); padding: var(--space-2) var(--space-3); border-radius: var(--radius); border: 1px solid transparent; cursor: pointer; text-align: start; transition: background-color 160ms ease, color 160ms ease, scale 160ms ease; }
  button:active:not(:disabled) { scale: 0.96; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .inspect-hint { margin: 24px 0 0; }
  .token-text { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px; margin-block: 16px 24px; }
  [data-tone="likely"] { --token-ink: var(--demo-token-likely); }
  [data-tone="possible"] { --token-ink: var(--demo-token-possible); }
  [data-tone="unlikely"] { --token-ink: var(--demo-token-unlikely); }
  .token { --control-sheen: none; display: inline-block; min-width: 40px; min-height: 44px; max-width: 100%; padding: 0 8px; margin: 0; border: 0; border-radius: var(--radius-sm); background: color-mix(in srgb, var(--token-ink) 12%, transparent); color: var(--token-ink); box-shadow: inset 0 -2px color-mix(in srgb, var(--token-ink) 40%, transparent); font: inherit; font-family: var(--font-reading) !important; font-weight: var(--weight-bold) !important; line-height: 1.6; white-space: pre-wrap; overflow-wrap: anywhere; vertical-align: baseline; text-align: center; transition: background-color 160ms ease, color 160ms ease, box-shadow 160ms ease; }
  .token.whitespace { min-width: 40px; }
  .token:active:not(:disabled) { scale: 1; }
  .token:hover { background: color-mix(in srgb, var(--token-ink) 22%, transparent); }
  .token[aria-pressed=true] { background: color-mix(in srgb, var(--token-ink) 26%, transparent); box-shadow: inset 0 0 0 2px var(--token-ink); text-decoration: underline; text-underline-offset: 5px; text-decoration-thickness: 2px; }
  .step-controls { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 24px; font-size: var(--text-sm); font-variant-numeric: tabular-nums; }
  .distribution { width: 100%; table-layout: fixed; border-collapse: collapse; font-size: var(--text-sm); }
  caption { text-align: start; font-weight: var(--weight-structure); padding-block: 8px 16px; }
  th { text-align: start; font-weight: normal; color: var(--fg-muted); padding: 8px 0; }
  td { position: relative; padding: 8px 0; overflow-wrap: anywhere; }
  th:last-child, td:last-child { width: 35%; text-align: end; font-variant-numeric: tabular-nums; }
  .token-name { white-space: pre-wrap; }
  .chosen-label { display: inline-block; margin-inline-start: 8px; color: var(--fg-muted); font-size: var(--text-xs); }
  .chosen .token-name { font-weight: var(--weight-bold); color: var(--token-ink); }
  .probability-bar { position: absolute; inset-inline-start: 0; top: 8px; bottom: 8px; background: color-mix(in srgb, var(--token-ink) 22%, transparent); border-radius: var(--radius-sm); transition: width 240ms cubic-bezier(0.2, 0, 0, 1), background-color 160ms ease; }
  .probability-value { position: relative; }
  .remaining { color: var(--fg-muted); }
  .comparison { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 32px; margin-block: 8px 0; }
  article { display: flex; flex-direction: column; align-items: start; min-width: 0; }
  h3 { text-wrap: balance; font-size: var(--text); font-weight: var(--weight-structure); margin: 0; }
  h3 span { display: block; color: var(--fg-muted); font-size: var(--text-sm); font-weight: normal; margin-top: 8px; }
  .branch-reply { font-family: var(--font-reading); font-weight: var(--weight-bold); line-height: 1.85; text-wrap: pretty; margin-block: 24px; white-space: pre-wrap; overflow-wrap: anywhere; flex: 1; }
  @media (prefers-reduced-motion: reduce) {
    .capability-demo, button, .token, .probability-bar { transition: none; }
    button:active:not(:disabled) { scale: 1; }
  }
  @media (max-width: 700px) {
    header :global(.sk-tabs) { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); width: 100%; }
    header :global(.tab) { padding-inline: 8px; letter-spacing: 0; }
    .comparison { grid-template-columns: 1fr; gap: 24px; }
    .alpha span:last-child { font-size: var(--text-sm); }
    .step-controls { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .step-controls > span { grid-column: 1 / -1; grid-row: 2; text-align: center; padding-top: 8px; }
    .step-controls > button { text-align: center; }
  }
</style>
