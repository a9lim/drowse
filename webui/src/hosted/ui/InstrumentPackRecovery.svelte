<script lang="ts">
  import { onMount } from "svelte";
  import Button from "../../lib/ui/Button.svelte";
  import { sessionState } from "../../lib/stores/session.svelte";
  import {
    currentCatalogInstrumentAvailability,
    type CatalogInstrumentAvailability,
  } from "../../lib/runtime/instrumentAvailability";

  let { kind }: { kind: "jlens" | "sae" } = $props();
  let availability = $state<CatalogInstrumentAvailability>("unknown");
  const name = $derived(kind === "jlens" ? "J-lens" : "SAE");
  const message = $derived(
    availability === "unavailable"
      ? `No compatible ${name} pack is published for this model.`
      : availability === "context-unavailable"
        ? `No ${name} pack supports this conversation length. Open Models to choose a supported length.`
        : `No ${name} pack is loaded. Open Models to download or update the compatible pack, then reopen the model. Your chats stay saved.`,
  );

  onMount(() => {
    let mounted = true;
    void currentCatalogInstrumentAvailability(kind).then((value) => {
      if (mounted) availability = value;
    });
    return () => { mounted = false; };
  });
</script>

<div class="pack-recovery">
  <p>{message}</p>
  {#if kind === "jlens" && sessionState.info?.is_base_model}
    <p>Word insights are optional for base models. Text completion works without a compatible J-lens pack.</p>
  {/if}
  <Button onclick={() => window.dispatchEvent(new Event("drowse:models"))}>Open Models</Button>
</div>

<style>
  .pack-recovery { display: grid; justify-items: start; gap: var(--space-3); }
  p { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); line-height: 1.5; }
</style>
