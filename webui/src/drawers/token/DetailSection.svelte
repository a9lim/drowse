<script lang="ts">
  import MorphText from "../../lib/ui/MorphText.svelte";
  // Canonical detailed-evidence section. It mirrors the main rack's compact
  // STEER / PROBE section headers.

  import type { Snippet } from "svelte";

  let {
    title,
    count = null,
    accent = "var(--accent)",
    children,
  }: {
    title: string;
    count?: string | null;
    accent?: string;
    children: Snippet;
  } = $props();
</script>

<section class="section" style:--section-accent={accent}>
  <header>
    <div class="title-row">
      <h3>{title}</h3>
      {#if count}<span class="count"><MorphText text={count} /></span>{/if}
    </div>
  </header>
  <div class="content">{@render children()}</div>
</section>

<style>
  .section {
    --section-accent: var(--accent);
    display: flex;
    flex-direction: column;
    gap: var(--space-4);
    min-width: 0;
  }
  header {
    min-width: 0;
  }
  .title-row {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    flex: 0 0 auto;
  }
  h3 {
    margin: 0;
    color: var(--section-accent);
    font-size: var(--text-sm);
    font-weight: var(--weight-bold);
    text-transform: uppercase;
    letter-spacing: 0;
  }
  .count {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    font-variant-numeric: tabular-nums;
  }
  .content {
    min-width: 0;
  }
</style>
