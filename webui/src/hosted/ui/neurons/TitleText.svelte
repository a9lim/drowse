<script lang="ts">
  import MorphText from "../../../lib/ui/MorphText.svelte";
  import { hasChinese } from "./data";
  import { gloss, shortGloss } from "./gloss.svelte";

  // Neuron titles and overviews share the same per-term translation and glossary.
  // Cards already wrapped in links keep plain spans; standalone summaries can focus each term.
  let { text, fallback = "Unclear", interactive = false }: { text: string; fallback?: string; interactive?: boolean } = $props();

  const parts = $derived((text || fallback).split(" · ").map((part) => {
    const entry = gloss.inline && hasChinese(part) ? gloss.dict[part.trim()] : undefined;
    return { source: part, shown: entry ? shortGloss(entry) : part, translated: !!entry, chinese: hasChinese(part) };
  }));
</script>

<span class="title-text">{#each parts as part, i}{#if i}<span class="sep" aria-hidden="true">{" · "}</span><span class="sr-only">{", "}</span>{/if}<svelte:element this={interactive && part.chinese ? "button" : "span"} type={interactive && part.chinese ? "button" : undefined} class:translated={part.translated} class:zh-term={part.chinese} data-zh={part.chinese ? part.source.trim() : undefined} lang={part.chinese ? part.translated ? "en" : "zh" : undefined}><MorphText text={part.shown} numbers={false} duration={260} />{#if part.translated}<span class="sr-only">{` (translated from ${part.source})`}</span>{/if}</svelte:element>{/each}</span>

<style>
  .title-text { overflow-wrap: anywhere; }
  /* Inline so a long translated part wraps with the title instead of overflowing its box. */
  .title-text :global(.morph-text) { display: inline; }
  button { display: inline; margin: 0; padding: 0; border: 0; border-radius: var(--radius-sm); background: transparent; color: inherit; font: inherit; text-align: inherit; cursor: help; }
  button:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .sep { color: var(--fg-muted); }
  .translated { color: var(--neuron-translated-ink, var(--accent-amber)); font-style: italic; }
  .sr-only { position: absolute; inset-inline-start: 0; top: 0; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
</style>
