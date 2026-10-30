<script lang="ts">
  import { hasChinese, stem, tokenParts } from "./data";
  import MorphText from "../../../lib/ui/MorphText.svelte";
  import { gloss, shortGloss } from "./gloss.svelte";

  let { token, count = 1, flags = 0, size = "md", value }: {
    token: string;
    /** Variants merged into this token (release, released, Release…). */
    count?: number;
    /** 1: another lens names it too. 2: it measurably drives or peaks the neuron. */
    flags?: number;
    size?: "lg" | "md" | "sm";
    /** Optional measured value shown after the token (an activation). */
    value?: number;
  } = $props();

  const parts = $derived(tokenParts(token));
  const shared = $derived((flags & 1) !== 0);
  const measured = $derived((flags & 2) !== 0);
  const zh = $derived(!parts.byte && hasChinese(token) ? token.trim() : undefined);
  // With "Show Chinese in English" on, the token's text morphs into its gloss and takes the translated colour.
  const english = $derived(gloss.inline && zh && gloss.dict[zh] ? shortGloss(gloss.dict[zh]) : "");
  const shownText = $derived(english || (parts.empty ? "∅" : parts.text));
  const description = $derived([
    parts.lead ? "word start" : "",
    parts.byte ? "byte piece" : "",
    count > 1 ? `${count} variants` : "",
    shared ? "lenses agree" : "",
    measured ? "confirmed on real text" : "",
    english ? `translated from ${zh}` : "",
  ].filter(Boolean).join(", "));
</script>

<!-- Chinese tokens are buttons, so keyboard users can focus them to read the gloss. -->
<svelte:element
  this={zh ? "button" : "span"}
  type={zh ? "button" : undefined}
  class={`tok ${size}`}
  class:byte={parts.byte}
  class:shared
  class:measured
  class:zh={zh !== undefined}
  class:translated={english !== ""}
  data-stem={stem(token)}
  data-zh={zh}
  lang={zh ? (english ? "en" : "zh") : undefined}
><span class="mark" aria-hidden="true">{parts.lead ? "·" : ""}</span><span class="text">{#if zh}<MorphText text={shownText} numbers={false} duration={260} />{:else}{shownText}{/if}</span>{#if description}<span class="sr-only">{` (${description})`}</span>{/if}{#if count > 1}<span class="variants" aria-hidden="true">×{count}</span>{/if}{#if value !== undefined}<span class="value">{value.toFixed(1)}</span>{/if}{#if measured}<span class="dot" aria-hidden="true"></span>{/if}</svelte:element>

<style>
  .tok {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 0;
    max-width: 100%;
    min-height: 22px;
    padding: 0 var(--space-xs);
    border-radius: var(--radius-sm);
    background: var(--glass);
    box-shadow: inset 0 0 0 1px var(--glass-line);
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    line-height: 1.4;
    white-space: pre;
    transition: background-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out);
  }
  .tok.lg { min-height: 26px; font-size: var(--text-sm); }
  .tok.sm { min-height: 20px; font-size: var(--text-xs); color: var(--fg-dim); }
  .tok.zh { border: 0; font-family: var(--font-reading); cursor: help; }
  .tok.zh:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 1px; }
  /* Long tokens (runs of punctuation, URLs) wrap inside the chip rather than being cut off. */
  .text { white-space: pre-wrap; overflow-wrap: anywhere; }
  .mark { color: var(--fg-muted); }
  .mark:empty { display: none; }
  .byte { color: var(--fg-muted); background: transparent; box-shadow: none; outline: 1px dashed var(--glass-line); outline-offset: -1px; }
  /* Translated tokens wear the amber "edited" hue, distinct from the depth tint and the measured green. */
  .translated { background: color-mix(in srgb, var(--accent-amber) 16%, var(--glass)); box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent-amber) 55%, transparent); color: var(--neuron-translated-ink, var(--fg)); font-style: italic; }
  .translated .mark { color: var(--accent-amber); }
  .variants, .value { color: var(--fg-muted); font-size: var(--text-2xs); font-variant-numeric: tabular-nums; }
  .value { margin-inline-start: var(--space-xs); }
  /* An underline bar, so agreement doesn't rest on colour alone. */
  .shared { box-shadow: inset 0 0 0 1px color-mix(in oklch, var(--neuron-tint) 70%, transparent), inset 0 -2px 0 var(--neuron-tint); background: color-mix(in oklch, var(--neuron-tint) 14%, var(--glass)); }
  .dot { align-self: center; width: 0.38em; height: 0.38em; margin-inline-start: var(--space-xs); border-radius: 50%; background: var(--accent-green); }
  /* Hovering any token lights up its variants across every read of the neuron (see LinkedTokens). */
  :global(.tok.linked) { background: color-mix(in oklch, var(--neuron-tint) 30%, var(--glass)); box-shadow: inset 0 0 0 1px var(--neuron-tint); color: var(--fg); }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  @media (prefers-reduced-motion: reduce) { .tok { transition: none; } }
  @media (forced-colors: active) { .tok { border: 1px solid CanvasText; } :global(.tok.linked) { outline: 2px solid Highlight; } }
</style>
