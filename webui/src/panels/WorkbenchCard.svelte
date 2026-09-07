<script lang="ts">
  import { sessionState } from "../lib/stores.svelte";
  import { DROWSE_UI_VERSION } from "../lib/version";
  import ModelProviderLogo from "../hosted/ui/ModelProviderLogo.svelte";
  import BaseModelTag from "../lib/ui/BaseModelTag.svelte";

  const model = $derived(sessionState.info?.model_id ?? "");
  const shortId = $derived(model.split("/").at(-1) ?? model);
  const modelName = $derived.by(() => {
    const known = shortId.match(/^(gemma|qwen)-?(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?b)(?:-|$)/i);
    if (!known) return shortId || "No model open";
    const family = known[1].toLowerCase() === "gemma" ? `Gemma ${known[2]}` : `Qwen${known[2]}`;
    return `${family} ${known[3].toUpperCase()}`;
  });
</script>

<span class="workbench">
  <span class="model-identity">
    <ModelProviderLogo modelId={shortId} />
    <span class="model">{modelName}{#if sessionState.info?.is_base_model}<BaseModelTag />{/if}</span>
  </span>
  <span class="version">Drowse {DROWSE_UI_VERSION}</span>
</span>

<style>
  .workbench {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: var(--space-2);
    min-width: 0;
    padding: var(--space-3);
  }

  .model-identity {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    max-width: 100%;
  }

  .version {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    line-height: 1.5;
  }

  .model {
    margin: 0;
    flex: 1 1 auto;
    min-width: 0;
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-structure);
    line-height: 1.25;
    color: var(--fg-strong);
    overflow-wrap: anywhere;
  }
</style>
