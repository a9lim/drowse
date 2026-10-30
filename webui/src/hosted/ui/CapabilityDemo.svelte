<script lang="ts">
  import { recoverFromChunkLoadError } from "../runtime/chunkRecovery";

  let { onready }: { onready?: () => void } = $props();
  const load = () => import("./RecordedDemo.svelte").then((module) => { queueMicrotask(() => onready?.()); return module; });
  let example = $state(load());
  let retrying = $state(false);

  // Chromium caches a failed dynamic import for the page's lifetime, so a plain retry
  // re-reads the same failure. Refresh the app files once; otherwise retry in place.
  // The landing demo is part of the bootstrap bundle, so it shares that recovery stage.
  async function retry(error: unknown) {
    retrying = true;
    try {
      if (await recoverFromChunkLoadError(error, "bootstrap")) return;
      example = load();
    } finally {
      retrying = false;
    }
  }
</script>

{#await example}
  <p role="status">Loading the recorded example…</p>
{:then { default: Example }}
  <Example />
{:catch error}
  <p role="status">The recorded example could not load.</p>
  <button type="button" disabled={retrying} onclick={() => void retry(error)}>Try again</button>
{/await}
