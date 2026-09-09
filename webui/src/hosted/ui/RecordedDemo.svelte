<script lang="ts">
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
  <header>
    <span class="demo-label">Recorded preset: a new neighbor</span>
    <SegmentedTabs items={options} bind:value={mode} ariaLabel="Example capability" />
  </header>
  <p class="context">{promptPreview}</p>
  <p class="prompt-note">The model was asked to reply in two short sentences.</p>
  {#if mode !== "branches"}
    <div class="alpha"><span>Welcoming influence</span><span>{alphaLabel(run.alpha)} · α {run.alpha.toFixed(2)}</span></div>
    <Slider value={setting} min={0} max={recording.runs.length - 1} step={1} oninput={value => setting = value}
      ariaLabel="Recorded welcoming setting" title={`${alphaLabel(run.alpha)}, alpha ${run.alpha.toFixed(2)}`} displayValue={`α ${run.alpha.toFixed(2)}`} />
    <div class="scale-labels" aria-hidden="true"><span>Detached</span><span>Unsteered</span><span>Welcoming</span></div>
    {#if mode === "steering"}
      <p class="result">{run.text}</p>
      <p class="demo-note">Seven saved settings, one prompt and seed. Each stop shows a complete recorded reply, not an interpolated prediction.</p>
    {:else}
      <p class="inspect-hint">Select a token to see what could have come next at that point.</p>
      <div class="result token-text" role="group" aria-label="Recorded reply tokens">
        {#each tokens as piece, index}
          <button class="token" type="button" class:future={index > step} class:whitespace={piece.text.trim().length === 0} aria-pressed={index === step}
            tabindex={index === step ? 0 : -1} aria-label={`Inspect token ${index + 1}: ${tokenLabel(piece.text)}`}
            onclick={() => tokenIndex = index} onkeydown={event => moveToken(event, index)}>{piece.text}</button>
        {/each}
      </div>
      <div class="step-controls">
        <button type="button" disabled={step === 0} onclick={() => tokenIndex = step - 1}>Previous token</button>
        <span aria-live="polite">Token {step + 1} of {tokens.length}</span>
        <button type="button" disabled={step === tokens.length - 1} onclick={() => tokenIndex = step + 1}>Next token</button>
      </div>
      <table class="distribution">
        <caption>Before generating “{tokenLabel(token.text)}”</caption>
        <thead><tr><th scope="col">Possible next token</th><th scope="col">Probability</th></tr></thead>
        <tbody>
          {#each alternatives as alternative}
            <tr class:chosen={alternative.id === token.tokenId}>
              <td><span class="token-name">{tokenLabel(alternative.text)}</span>{#if alternative.id === token.tokenId}<span class="chosen-label">Generated</span>{/if}</td>
              <td><span class="probability-bar" aria-hidden="true" style:width={`${Math.exp(alternative.logprob) * 100}%`}></span><span class="probability-value">{percent(Math.exp(alternative.logprob))}</span></td>
            </tr>
          {/each}
          <tr class="remaining"><td>All other tokens</td><td>{percent(remaining)}</td></tr>
        </tbody>
      </table>
      <p class="demo-note">Probabilities use the prompt and every preceding token, after temperature. The leading alternatives and generated token are shown without rescaling. Tokens can be words, word pieces, or punctuation.</p>
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
    <p class="demo-note">Three saved paths from the same prompt, model, and seed. Only steering changes. Once the replies diverge, later tokens also have different preceding text.</p>
  {/if}
  <details class="provenance">
    <summary>About this recording</summary>
    <p>Gemma 3 4B Instruct, recorded with Drowse WebGPU on {recording.recordedAt.slice(0, 10)}. No model is loaded by this preset.</p>
    <p>Full prompt: {recording.prompt}</p>
    <p>Temperature {recording.sampling.temperature}, seed {recording.sampling.seed}, top-p {recording.sampling.top_p}, no top-k cutoff. Context: {recording.contextTokens} tokens. Direction: <code>default/welcoming.detached</code>.</p>
    <p>This is one curated example, not a guarantee of a particular tone. Stronger steering is not always better. Wording, punctuation, and probabilities are preserved from the recordings.</p>
    <a href={recordingUrl} download="drowse-neighbor-recording.json">Download the recorded tokens and model provenance</a>
  </details>
</section>

<style>
  .capability-demo { min-width: 0; }
  header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 24px; }
  .demo-label, .demo-note, .inspect-hint, .provenance, .scale-labels, .prompt-note { color: var(--fg-muted); font-size: var(--text-sm); line-height: 1.6; }
  .context { max-width: 70ch; font-size: var(--text-lg); line-height: 1.6; margin: 0 0 8px; }
  .prompt-note { margin: 0 0 24px; }
  .alpha, .scale-labels { display: flex; justify-content: space-between; gap: 16px; font-variant-numeric: tabular-nums; }
  .alpha { flex-wrap: wrap; -webkit-user-select: none; user-select: none; }
  .scale-labels { -webkit-user-select: none; user-select: none; }
  .result { min-height: 4.8em; max-width: 70ch; font-family: var(--font-reading); font-size: var(--text-lg); line-height: 1.8; overflow-wrap: anywhere; margin-block: 24px; white-space: pre-wrap; }
  button { min-height: 44px; background: var(--input-well); color: var(--fg); padding: var(--space-2) var(--space-3); border-radius: var(--radius); border: 1px solid transparent; cursor: pointer; text-align: start; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .token-text { display: block; }
  .token { --control-sheen: none; display: inline; min-height: 0; padding: 0; margin: 0; border: 0; border-radius: var(--radius-sm); background: transparent; box-shadow: none; font: inherit; white-space: pre-wrap; overflow-wrap: anywhere; }
  .token.whitespace { display: inline-block; min-width: 1ch; min-height: 1em; }
  .token.future { color: var(--fg-muted); }
  .token:hover, .token[aria-pressed=true] { background: var(--accent-subtle); color: var(--fg); }
  .token[aria-pressed=true] { text-decoration: underline; text-decoration-color: var(--accent); text-underline-offset: 5px; text-decoration-thickness: 2px; }
  .step-controls { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 16px; font-size: var(--text-sm); font-variant-numeric: tabular-nums; }
  .distribution { width: 100%; table-layout: fixed; border-collapse: collapse; font-size: var(--text-sm); }
  caption { text-align: start; font-weight: var(--weight-structure); padding-block: 8px; }
  th { text-align: start; font-weight: normal; color: var(--fg-muted); padding: 8px 0; }
  td { position: relative; padding: 8px 0; overflow-wrap: anywhere; }
  th:last-child, td:last-child { width: 35%; text-align: end; font-variant-numeric: tabular-nums; }
  .token-name { white-space: pre-wrap; }
  .chosen-label { display: inline-block; margin-inline-start: 8px; color: var(--fg-muted); font-size: var(--text-xs); }
  .chosen .token-name { font-weight: var(--weight-structure); }
  .probability-bar { position: absolute; inset-inline-start: 0; top: 8px; bottom: 8px; background: var(--accent-subtle); border-radius: var(--radius-sm); }
  .probability-value { position: relative; }
  .remaining { color: var(--fg-muted); }
  .comparison { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 24px; margin-block: 24px; }
  article { display: flex; flex-direction: column; align-items: start; min-width: 0; }
  h3 { font-size: var(--text); font-weight: var(--weight-structure); margin: 0; }
  h3 span { display: block; color: var(--fg-muted); font-size: var(--text-sm); font-weight: normal; margin-top: 8px; }
  .branch-reply { font-family: var(--font-reading); line-height: 1.8; white-space: pre-wrap; overflow-wrap: anywhere; flex: 1; }
  .provenance { margin-top: 24px; }
  summary { width: fit-content; cursor: pointer; min-height: 44px; align-content: center; }
  .provenance p { max-width: 75ch; }
  code { overflow-wrap: anywhere; }
  @media (max-width: 700px) { .comparison { grid-template-columns: 1fr; gap: 24px; } .alpha span:last-child { font-size: var(--text-sm); } }
</style>
