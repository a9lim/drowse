<script lang="ts">
  import LensReadView from "./LensReadView.svelte";
  import Token from "./Token.svelte";
  import { LENSES, bandOf, percent, poleWord, type Pole } from "./data";

  let { pole, layer, clearer }: { pole: Pole; layer: number; clearer: boolean } = $props();

  const band = $derived(bandOf(layer));
  const inputs = $derived(pole.reads.filter((read) => LENSES[read.col].side === "input" && (read.trust === "primary" || read.trust === "trusted")));
  const outputs = $derived(pole.reads.filter((read) => LENSES[read.col].side === "output" && (read.trust === "primary" || read.trust === "trusted")));
  const others = $derived(pole.reads.filter((read) => read.trust === "weak" || read.trust === "compare"));
  const firesFirst = $derived(band === "dark");
  let allLenses = $state(false);

  const slices = $derived(pole.slices ?? [0, 0, 0, 0]);
  const sliceMax = $derived(Math.max(...slices, 1e-6));
  const sliceNames = ["Web", "Chinese", "Code", "Chat"];

  function twinName(index: number, list: typeof pole.reads): string | undefined {
    const read = list[index];
    if (!read.twin) return undefined;
    const family = read.col.replace(/_z$/, "");
    const first = pole.reads.find((other) => other !== read && other.col.replace(/_z$/, "") === family);
    return first ? LENSES[first.col].short : undefined;
  }

  const confidence = $derived(pole.label ? pole.label.confidence : 0);
  const clip = (text: string, length: number, fromEnd: boolean) =>
    text.length <= length ? text : fromEnd ? `…${text.slice(-length)}` : `${text.slice(0, length)}…`;
</script>

<article class="pole" class:silent={pole.silent} aria-label={`${poleWord(pole.pole)}: fires ${pole.pole > 0 ? "positive" : "negative"}`}>
  <header class="pole-head">
    <span class="sign" aria-hidden="true">{pole.pole > 0 ? "+" : "−"}</span>
    <div class="pole-title">
      <h3>{poleWord(pole.pole)}<span class="sub">{` · fires ${pole.pole > 0 ? "positive" : "negative"}`}</span></h3>
      {#if clearer}<span class="badge clear">Clearer pole</span>{/if}
      {#if pole.silent}<span class="badge">Silent</span>{/if}
      {#if pole.detector === "token"}<span class="badge">Token detector</span>{/if}
      {#if pole.detector === "next"}<span class="badge">Next-token detector</span>{/if}
      {#if pole.shape?.kind === "predicts"}<span class="badge tint">Predicts the next word</span>{/if}
      {#if pole.shape?.kind === "suppresses"}<span class="badge warn">Suppresses the next word</span>{/if}
    </div>
  </header>

  {#if pole.label}
    <div class="label" class:low={confidence < 0.5}>
      <p class="label-text">“{pole.label.text}”</p>
      <p class="label-meta">
        <span class="conf" role="img" aria-label={`Confidence ${Math.round(confidence * 100)}%`}><span style:--c={`${confidence * 100}%`}></span></span>
        {Math.round(confidence * 100)}% confident · {pole.label.source}{pole.label.relation && pole.label.relation !== "unclear" ? ` · ${pole.label.relation}s what it reads` : ""}
      </p>
      {#if pole.label.note}<p class="note">{pole.label.note}</p>{/if}
    </div>
  {/if}

  <dl class="facts">
    <div><dt>Typical peak</dt><dd>{pole.act_top?.toFixed(2) ?? "–"}</dd></div>
    <div><dt>Fires on</dt><dd>{percent(pole.rate)}{" "}<span class="unit">of tokens</span></dd></div>
    <div class="slices">
      <dt>Where it fires</dt>
      <dd>
        <span class="slice-bars" role="img" aria-label={sliceNames.map((name, index) => `${name} ${percent(slices[index], 2)}`).join(", ")}>
          {#each slices as value, index}
            <span class="slice">
              <span class="slice-bar"><span style:--h={`${Math.max(4, (value / sliceMax) * 100)}%`}></span></span>
              <span class="slice-name">{sliceNames[index]}</span>
            </span>
          {/each}
        </span>
      </dd>
    </div>
  </dl>

  {#if pole.measuredAgree.length || pole.shared.length}
    <section class="agree" aria-label="Agreement">
      {#if pole.measuredAgree.length}
        <p class="agree-row"><span class="agree-key"><span class="dot measured" aria-hidden="true"></span>Confirmed on real text</span>
          <span class="chips">{#each pole.measuredAgree.slice(0, 8) as token}<Token token={token} flags={2} size="sm" />{/each}</span></p>
      {/if}
      {#if pole.shared.length}
        <p class="agree-row"><span class="agree-key"><span class="dot shared" aria-hidden="true"></span>Lenses agree</span>
          <span class="chips">{#each pole.shared.slice(0, 8) as token}<Token token={token} flags={1} size="sm" />{/each}</span></p>
      {/if}
    </section>
  {/if}

  {#snippet fires()}
    <section class="side" aria-label="Where it fires">
      <h4>Fires on</h4>
      {#if pole.contexts?.length}
        <ol class="contexts">
          {#each pole.contexts as [before, hit, after]}
            <li><span class="ctx-pre">{clip(before, 70, true)}</span><mark>{hit || "␣"}</mark><span class="ctx-post">{clip(after, 44, false)}</span></li>
          {/each}
        </ol>
      {:else}
        <p class="empty">No firing contexts for this pole.</p>
      {/if}
      {#if pole.drivers?.length}
        <h5>Drivers</h5>
        <div class="chips">{#each pole.drivers.slice(0, 12) as [token, value]}<Token {token} {value} size="sm" />{/each}</div>
      {/if}
      {#if pole.driversSum?.length}
        <h5>Lingering drivers</h5>
        <div class="chips">{#each pole.driversSum.slice(0, 8) as [token, value]}<Token {token} {value} size="sm" />{/each}</div>
      {/if}
    </section>
  {/snippet}

  {#if firesFirst}{@render fires()}{/if}

  <section class="side" aria-label="What drives it">
    <h4>Reads</h4>
    {#each inputs as read, index (read.col)}<LensReadView {read} twinOf={twinName(index, inputs)} />{/each}
  </section>

  <section class="side" aria-label="What it writes">
    <h4>Writes</h4>
    {#each outputs as read, index (read.col)}<LensReadView {read} twinOf={twinName(index, outputs)} />{/each}
  </section>

  {#if !firesFirst}{@render fires()}{/if}

  {#if others.length}
    <section class="side others">
      <button type="button" class="toggle" aria-expanded={allLenses} onclick={() => (allLenses = !allLenses)}>
        {allLenses ? "Hide other lenses" : `Show ${others.length} other lenses`}
      </button>
      {#if allLenses}
        <div class="other-reads">{#each others as read, index (read.col)}<LensReadView {read} twinOf={twinName(index, others)} />{/each}</div>
      {/if}
    </section>
  {/if}

</article>

<style>
  .pole { display: grid; align-content: start; gap: var(--space-4); min-width: 0; }
  .silent { opacity: 0.6; }
  .pole-head { display: flex; align-items: center; gap: var(--space-3); }
  .sign {
    display: grid; place-items: center; flex: none; width: 28px; height: 28px; border-radius: var(--radius);
    background: color-mix(in oklch, var(--neuron-tint) 20%, var(--glass)); color: var(--fg);
    font-family: var(--font-structure); font-size: var(--text-md); font-weight: var(--weight-display); line-height: 1;
  }
  .pole-title { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-1) var(--space-2); }
  h3 { margin: 0; color: var(--fg); font-size: var(--text-md); font-weight: var(--weight-structure-bold); }
  .sub { color: var(--fg-muted); font-size: var(--text-sm); font-weight: normal; }
  .badge { display: inline-flex; align-items: center; min-height: 20px; padding-inline: var(--space-2); border-radius: var(--radius-pill); background: var(--glass-strong); color: var(--fg-dim); font-size: var(--text-xs); font-weight: var(--weight-structure); white-space: nowrap; }
  .badge.clear { color: var(--fg); box-shadow: inset 0 0 0 1px color-mix(in oklch, var(--neuron-tint) 60%, transparent); }
  .badge.tint { background: color-mix(in oklch, var(--neuron-tint) 18%, var(--glass)); color: var(--fg); }
  .badge.warn { background: var(--glass-strong); color: var(--fg); }

  .label { display: grid; gap: var(--space-1); padding: var(--space-3) var(--space-4); border-radius: var(--radius); background: var(--glass); }
  .label.low .label-text { color: var(--fg-dim); }
  .label-text { margin: 0; color: var(--fg); font-size: var(--text-sm); font-weight: var(--weight-structure); line-height: 1.4; text-wrap: pretty; }
  .label-meta { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2); margin: 0; color: var(--fg-muted); font-size: var(--text-xs); }
  .conf { position: relative; display: inline-block; width: 32px; height: 3px; border-radius: var(--radius-pill); background: var(--glass-strong); overflow: hidden; }
  .conf span { position: absolute; inset: 0 auto 0 0; width: var(--c); background: var(--neuron-tint); border-radius: inherit; }
  .note { margin: 0; color: var(--fg-dim); font-size: var(--text-xs); line-height: 1.5; text-wrap: pretty; }

  .facts { display: grid; grid-template-columns: auto auto minmax(0, 1fr); gap: var(--space-5); margin: 0; }
  .facts > div { display: grid; gap: 0; align-content: start; }
  dt { color: var(--fg-muted); font-size: var(--text-xs); }
  dd { margin: 0; color: var(--fg); font-family: var(--font-mono); font-size: var(--text-xs); font-variant-numeric: tabular-nums; white-space: nowrap; }
  .unit { color: var(--fg-muted); font-family: var(--font-reading); }
  .slices { justify-self: end; }
  .slice-bars { display: flex; align-items: flex-end; gap: var(--space-2); }
  .slice { display: grid; justify-items: center; gap: 0; }
  .slice-bar { position: relative; display: block; width: 8px; height: 18px; border-radius: var(--data-mark-radius); background: var(--glass-strong); overflow: hidden; }
  .slice-bar span { position: absolute; inset: auto 0 0; height: var(--h); background: var(--neuron-tint); border-radius: inherit; }
  .slice-name { color: var(--fg-muted); font-family: var(--font-reading); font-size: var(--text-2xs); }

  .agree { display: grid; gap: var(--space-2); }
  .agree-row { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-2); margin: 0; }
  .agree-key { display: inline-flex; align-items: center; gap: var(--space-2); min-width: 9rem; color: var(--fg-muted); font-size: var(--text-xs); }
  .dot { width: 6px; height: 6px; border-radius: 50%; }
  .dot.measured { background: var(--accent-green); }
  .dot.shared { background: var(--neuron-tint); }
  .chips { display: flex; flex-wrap: wrap; gap: var(--space-1); min-width: 0; }

  .side { display: grid; gap: var(--space-3); min-width: 0; padding: var(--space-4); border-radius: var(--radius); background: var(--glass); }
  h4 { margin: 0; color: var(--fg-dim); font-size: var(--text-xs); font-weight: var(--weight-structure-bold); letter-spacing: 0.02em; }
  h5 { margin: var(--space-1) 0 0; color: var(--fg-muted); font-size: var(--text-xs); font-weight: var(--weight-structure); }
  .contexts { display: grid; gap: var(--space-1); margin: 0; padding: 0; list-style: none; }
  .contexts li { padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm); background: var(--input-well, var(--bg)); color: var(--fg-dim); font-family: var(--font-mono); font-size: var(--text-xs); line-height: 1.65; overflow-wrap: anywhere; white-space: pre-wrap; }
  mark { padding: 0; border-radius: var(--radius-sm); background: color-mix(in oklch, var(--neuron-tint) 34%, transparent); color: var(--fg); }
  .empty { margin: 0; color: var(--fg-muted); font-size: var(--text-xs); }

  .others { padding: 0; background: transparent; }
  .toggle { position: relative; justify-self: start; min-height: 32px; padding: 0 var(--space-3); border: 0; border-radius: var(--radius); background: transparent; box-shadow: inset 0 0 0 1px var(--glass-line); color: var(--fg-dim); font: inherit; font-size: var(--text-xs); cursor: pointer; transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out); }
  .toggle::after { content: ""; position: absolute; inset: calc(var(--space-xs) * -0.5) 0; }
  .toggle:hover { background: var(--bg-hover); color: var(--fg); }
  .toggle:active { scale: var(--press-scale); }
  .toggle:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .other-reads { display: grid; padding: var(--space-4); border-radius: var(--radius); background: var(--glass); animation: reveal var(--dur-slow) var(--ease-enter) backwards; }
  @keyframes reveal { from { opacity: 0; transform: translateY(-4px); } }
  @media (max-width: 520px) {
    .facts { grid-template-columns: 1fr 1fr; }
    .slices { grid-column: 1 / -1; justify-self: start; }
  }
  @media (prefers-reduced-motion: reduce) { .other-reads { animation: none; } .toggle { transition: none; } .toggle:active { scale: none; } }
</style>
