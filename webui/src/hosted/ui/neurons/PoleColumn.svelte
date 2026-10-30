<script lang="ts">
  import LensReadView from "./LensReadView.svelte";
  import Token from "./Token.svelte";
  import Expander from "./Expander.svelte";
  import { LENSES, bandOf, percent, type Pole } from "./data";

  let { pole, layer, clearer }: { pole: Pole; layer: number; clearer: boolean } = $props();

  const trusted = (side: "input" | "output") => pole.reads.filter((read) => LENSES[read.col].side === side && (read.trust === "primary" || read.trust === "trusted"));
  const inputs = $derived(trusted("input"));
  const outputs = $derived(trusted("output"));
  const others = $derived(pole.reads.filter((read) => read.trust === "weak" || read.trust === "compare"));
  // In the middle layers the examples describe a neuron better than any lens, so they come first.
  const examplesFirst = $derived(bandOf(layer) === "dark");
  const contexts = $derived(pole.contexts ?? []);
  // All-neurons data carries the lens reads only; measured sections render when present.
  const measured = $derived(pole.contexts !== undefined);
  const slices = $derived(pole.slices ?? [0, 0, 0, 0]);
  const sliceMax = $derived(Math.max(...slices, 1e-6));
  const sliceNames = ["Web", "Chinese", "Code", "Chat"];
  const confidence = $derived(pole.label?.confidence ?? 0);
  const saysVerb = $derived(layer >= 21 ? "Predicts the next word" : "Pushes the model to say");
  const clip = (text: string, length: number, fromEnd: boolean) =>
    text.length <= length ? text : fromEnd ? `…${text.slice(-length)}` : `${text.slice(0, length)}…`;

  function twinName(read: Pole["reads"][number]): string | undefined {
    if (!read.twin) return undefined;
    const family = read.col.replace(/_z$/, "");
    const first = pole.reads.find((other) => other !== read && other.col.replace(/_z$/, "") === family);
    return first ? LENSES[first.col].short : undefined;
  }
</script>

<article class="pole" class:silent={pole.silent} aria-label={pole.pole > 0 ? "Positive side" : "Negative side"}>
  <header class="pole-head">
    <span class="sign" aria-hidden="true">{pole.pole > 0 ? "+" : "−"}</span>
    <h3>{pole.pole > 0 ? "Positive side" : "Negative side"}</h3>
    {#if clearer}<span class="badge clear">Clearer side</span>{/if}
    {#if pole.detector}<span class="badge">Fires on one token</span>{/if}
    {#if pole.shape?.kind === "predicts"}<span class="badge">Predicts the next word</span>{/if}
    {#if pole.shape?.kind === "suppresses"}<span class="badge">Holds back the next word</span>{/if}
  </header>

  {#if pole.label}
    <div class="label" class:low={confidence < 0.5}>
      <p class="label-text">{pole.label.text}</p>
      <p class="label-meta"><span class="conf" aria-hidden="true"><span style:--c={confidence}></span></span>{Math.round(confidence * 100)}% sure</p>
    </div>
  {/if}

  {#snippet examples()}
    <section class="part">
      <h4>Examples</h4>
      {#if contexts.length}
        <ol class="contexts">
          {#each contexts.slice(0, 3) as [before, hit, after], i}
            <li style:--i={i}><span class="ctx-pre">{clip(before, 60, true)}</span><mark>{hit || "␣"}</mark><span>{clip(after, 36, false)}</span></li>
          {/each}
        </ol>
      {:else}
        <p class="empty">No examples for this side.</p>
      {/if}
    </section>
  {/snippet}

  {#if examplesFirst && measured}{@render examples()}{/if}

  {#if inputs[0]}
    <section class="part">
      <h4>What triggers it</h4>
      <LensReadView read={inputs[0]} compact />
    </section>
  {/if}

  {#if outputs[0]}
    <section class="part">
      <h4>{saysVerb}</h4>
      <LensReadView read={outputs[0]} compact />
    </section>
  {/if}

  {#if !examplesFirst && measured}{@render examples()}{/if}

  <Expander label="More detail" openLabel="Less detail">
    <div class="more">
      {#if pole.measuredAgree.length || pole.shared.length}
        <div class="agree">
          {#if pole.measuredAgree.length}
            <p class="agree-row"><span class="agree-key"><span class="dot measured" aria-hidden="true"></span>Confirmed on real text</span>
              <span class="chips">{#each pole.measuredAgree.slice(0, 10) as token}<Token token={token} flags={2} size="sm" />{/each}</span></p>
          {/if}
          {#if pole.shared.length}
            <p class="agree-row"><span class="agree-key"><span class="dot shared" aria-hidden="true"></span>Lenses agree</span>
              <span class="chips">{#each pole.shared.slice(0, 10) as token}<Token token={token} flags={1} size="sm" />{/each}</span></p>
          {/if}
        </div>
      {/if}

      {#if measured}
      <dl class="facts">
        <div><dt>Typical peak</dt><dd>{pole.act_top?.toFixed(2) ?? "–"}</dd></div>
        <div><dt>Fires on</dt><dd>{percent(pole.rate)} of tokens</dd></div>
        <div class="slices">
          <dt>Where</dt>
          <dd>
            <span class="slice-bars" role="img" aria-label={sliceNames.map((name, index) => `${name} ${percent(slices[index])}`).join(", ")}>
              {#each slices as value, index}
                <span class="slice"><span class="slice-bar"><span style:--h={Math.max(0.06, value / sliceMax)}></span></span><span class="slice-name">{sliceNames[index]}</span></span>
              {/each}
            </span>
          </dd>
        </div>
      </dl>
      {/if}

      {#each [...inputs.slice(1), ...outputs.slice(1)] as read (read.col)}<LensReadView {read} twinOf={twinName(read)} />{/each}

      {#if pole.drivers?.length}
        <div class="drivers">
          <h5>Words that set it off in real text</h5>
          <div class="chips">{#each pole.drivers.slice(0, 12) as [token, value]}<Token {token} {value} size="sm" />{/each}</div>
        </div>
      {/if}

      {#if contexts.length > 3}
        <ol class="contexts" start="4">
          {#each contexts.slice(3) as [before, hit, after]}
            <li><span class="ctx-pre">{clip(before, 60, true)}</span><mark>{hit || "␣"}</mark><span>{clip(after, 36, false)}</span></li>
          {/each}
        </ol>
      {/if}

      {#if pole.label?.note}<p class="note">{pole.label.note}</p>{/if}

      {#if others.length}
        <Expander label={`${others.length} weaker lenses`} openLabel="Hide weaker lenses">
          <div class="others">{#each others as read (read.col)}<LensReadView {read} twinOf={twinName(read)} />{/each}</div>
        </Expander>
      {/if}
    </div>
  </Expander>
</article>

<style>
  .pole { display: grid; align-content: start; gap: var(--space-5); min-width: 0; }
  .silent { opacity: 0.6; }
  .pole-head { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2) var(--space-3); }
  .sign {
    display: grid; place-items: center; flex: none; width: 32px; height: 32px; border-radius: var(--radius);
    background: color-mix(in oklch, var(--neuron-tint) 20%, var(--neuron-chip-bg)); color: var(--fg);
    font-family: var(--font-structure); font-size: var(--text-lg); font-weight: var(--weight-display); line-height: 1;
  }
  h3 { margin: 0; color: var(--fg); font-size: var(--text-lg); font-weight: var(--weight-structure-bold); letter-spacing: -0.01em; }
  .badge { display: inline-flex; align-items: center; min-height: 24px; padding-inline: var(--space-3); border-radius: var(--radius-pill); background: var(--glass-strong); color: var(--fg-dim); font-size: var(--text-xs); font-weight: var(--weight-structure); white-space: nowrap; }
  .badge.clear { background: color-mix(in oklch, var(--neuron-tint) 18%, var(--glass)); color: var(--fg); }

  .label { display: grid; gap: var(--space-2); padding: var(--space-4) var(--space-5); border-radius: var(--radius); background: var(--neuron-chip-bg); box-shadow: inset 0 0 0 1px var(--glass-line); }
  .label.low .label-text { color: var(--fg-dim); }
  .label-text { margin: 0; color: var(--fg); font-family: var(--font-structure); font-size: var(--text-md); font-weight: var(--weight-structure); line-height: 1.4; text-wrap: pretty; }
  .label-meta { display: flex; align-items: center; gap: var(--space-2); margin: 0; color: var(--fg-muted); font-size: var(--text-xs); }
  .conf { position: relative; display: inline-block; width: 48px; height: 4px; border-radius: var(--radius-pill); background: var(--glass-strong); overflow: hidden; }
  .conf span { position: absolute; inset: 0; background: var(--neuron-tint); border-radius: inherit; transform-origin: left; scale: var(--c) 1; }

  .part { display: grid; gap: var(--space-3); min-width: 0; }
  h4 { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); font-weight: var(--weight-structure); }
  h5 { margin: 0; color: var(--fg-muted); font-size: var(--text-xs); font-weight: var(--weight-structure); }
  .contexts { display: grid; gap: var(--space-2); margin: 0; padding: 0; list-style: none; }
  .contexts li { padding: var(--space-3) var(--space-4); border-radius: var(--radius); background: var(--neuron-well); color: var(--fg-dim); font-family: var(--font-mono); font-size: var(--text-sm); line-height: 1.7; overflow-wrap: anywhere; white-space: pre-wrap; animation: ctx-in 320ms var(--ease-enter) backwards; animation-delay: calc(var(--i, 0) * 60ms); }
  @keyframes ctx-in { from { opacity: 0; translate: 0 4px; } }
  /* The highlight grows with a shadow, not padding, so the token stays joined to its word. */
  mark { border-radius: var(--radius-sm); background: color-mix(in oklch, var(--neuron-tint) 34%, transparent); box-shadow: 0 0 0 2px color-mix(in oklch, var(--neuron-tint) 34%, transparent); color: var(--fg); }
  .empty { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); }

  .more { display: grid; gap: var(--space-5); }
  .agree { display: grid; gap: var(--space-3); }
  .agree-row { display: grid; gap: var(--space-2); margin: 0; }
  .agree-key { display: inline-flex; align-items: center; gap: var(--space-2); color: var(--fg-muted); font-size: var(--text-xs); }
  .dot { width: 6px; height: 6px; border-radius: 50%; }
  .dot.measured { background: var(--accent-green); }
  .dot.shared { background: var(--neuron-tint); }
  .chips { display: flex; flex-wrap: wrap; gap: var(--space-1); min-width: 0; }
  .facts { display: flex; flex-wrap: wrap; gap: var(--space-5) var(--space-8); margin: 0; }
  .facts > div { display: grid; gap: var(--space-1); align-content: start; }
  dt { color: var(--fg-muted); font-size: var(--text-xs); }
  dd { margin: 0; color: var(--fg); font-family: var(--font-mono); font-size: var(--text-sm); font-variant-numeric: tabular-nums; white-space: nowrap; }
  .slice-bars { display: flex; align-items: flex-end; gap: var(--space-2); }
  .slice { display: grid; justify-items: center; gap: 0; }
  .slice-bar { position: relative; display: block; width: 10px; height: 22px; border-radius: var(--data-mark-radius); background: var(--glass-strong); overflow: hidden; }
  .slice-bar span { position: absolute; inset: 0; background: var(--neuron-tint); border-radius: inherit; transform-origin: bottom; scale: 1 var(--h); }
  .slice-name { color: var(--fg-muted); font-family: var(--font-reading); font-size: var(--text-2xs); }
  .drivers { display: grid; gap: var(--space-2); }
  .note { margin: 0; color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.55; text-wrap: pretty; }
  .others { display: grid; }
  @media (prefers-reduced-motion: reduce) { .contexts li { animation: none; } }
</style>
