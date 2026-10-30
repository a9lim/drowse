<script lang="ts">
  import MorphText from "../../../lib/ui/MorphText.svelte";
  import { hasChinese } from "./data";
  import { gloss, shortGloss } from "./gloss.svelte";

  // A neuron title ("garden · 树林 · 耕地"): with translation on, each Chinese part morphs into English.
  let { text, fallback = "Unclear" }: { text: string; fallback?: string } = $props();

  const parts = $derived((text || fallback).split(" · ").map((part) => {
    const entry = gloss.inline && hasChinese(part) ? gloss.dict[part.trim()] : undefined;
    return { source: part, shown: entry ? shortGloss(entry) : part, translated: !!entry };
  }));
</script>

<span class="title-text">{#each parts as part, i}{#if i}<span class="sep" aria-hidden="true">{" · "}</span><span class="sr-only">{", "}</span>{/if}<span class:translated={part.translated} class:zh-term={hasChinese(part.source)} data-zh={hasChinese(part.source) ? part.source.trim() : undefined} lang={hasChinese(part.source) && !part.translated ? "zh" : undefined}><MorphText text={part.shown} numbers={false} duration={260} />{#if part.translated}<span class="sr-only">{` (translated from ${part.source})`}</span>{/if}</span>{/each}</span>

<style>
  .title-text { overflow-wrap: anywhere; }
  .sep { color: var(--fg-muted); }
  .translated { color: var(--neuron-translated-ink, var(--accent-amber)); font-style: italic; }
  .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
</style>
