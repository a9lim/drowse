<script lang="ts">
  import type { Snippet } from "svelte";

  let {
    left,
    bar,
    middle,
    right,
    ariaLabel,
  }: {
    left: Snippet;
    bar: Snippet;
    middle: Snippet;
    right: Snippet;
    ariaLabel?: string;
  } = $props();
</script>

<div class="reading" role="group" aria-label={ariaLabel}>
  <div class="label">{@render left()}</div>
  <div class="bar">{@render bar()}</div>
  <div class="middle">{@render middle()}</div>
  <div class="value">{@render right()}</div>
</div>

<style>
  .reading {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--space-2) var(--space-3);
    padding-block: var(--space-2);
    min-width: 0;
    min-height: var(--control-target);
  }
  .label,
  .middle,
  .value {
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .middle { text-align: end; }
  .value { text-align: end; }
  .bar {
    min-width: 0;
    grid-column: 1 / -1;
    grid-row: 2;
    padding-block: var(--space-1);
  }
  .label :global(*), .middle :global(*), .value :global(*) {
    white-space: normal;
  }
  .bar :global(.bar) {
    width: 100%;
    height: var(--data-bar-height);
    display: block;
  }
</style>
