<script lang="ts">
  import { flip } from "svelte/animate";
  import { motionDuration } from "../../lib/motion";
  import MorphText from "../../lib/ui/MorphText.svelte";

  let { text }: { text: string } = $props();
  const words = $derived(text.match(/\S+\s*/g) ?? []);
</script>

{#each words as word, index (index)}
  <span class="word" animate:flip={{ duration: motionDuration(240) }}><MorphText text={word.trimEnd()} numbers={false} duration={240} />{word.slice(word.trimEnd().length)}</span>
{/each}

<style>
  .word { display: inline-block; max-width: 100%; vertical-align: baseline; }
</style>
