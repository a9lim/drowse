<script lang="ts">
  import Select from "../Select.svelte";
  import MorphText from "./MorphText.svelte";
  let { sentence, slot, values }: { sentence: string; slot: string; values: string[] } = $props();
  let chosen = $state("");
  const candidate = $derived(values.includes(chosen) ? chosen : values[0] ?? "");
  const parts = $derived(slot ? sentence.split(slot) : []);
</script>
{#if parts.length === 2 && values.length > 0}
  <section class="template-preview" aria-label="Template slot preview">
    <label>Preview candidate<Select value={candidate} options={values.map(value => ({value, label: value}))} onchange={value => chosen = value} ariaLabel="Preview candidate" /></label>
    <p>{parts[0]}<mark><MorphText text={candidate} numbers={false} /></mark>{parts[1]}</p>
  </section>
{/if}
<style>
  .template-preview { padding: var(--surface-padding); background: var(--input-well); border-radius: var(--radius); min-width: 0; }
  label { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: var(--text-xs); }
  p { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.7; }
  mark { color: var(--accent); background: var(--accent-subtle); padding: var(--space-xs) var(--space-1); border-radius: var(--radius-sm); }
</style>
