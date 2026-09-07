<script lang="ts">
  import FluentIcon from "./ui/FluentIcon.svelte";
  // Themed collapsible — the existing ``▸/▾ caret + button row`` pattern
  // packaged as one component.  Backs the ad-hoc ``advancedOpen`` toggles
  // (e.g. ManifoldBuilderDrawer) and inline grammar-reference disclosures.
  //
  // Default slot is the body; the trigger row shows the caret + summary.

  import type { Snippet } from "svelte";
  import { slide } from "svelte/transition";
  import { collapseIn, collapseOut } from "./motion";

  interface Props {
    /** Bindable open state. */
    expanded: boolean;
    /** Trigger label. */
    summary: string;
    /** Optional dense styling — sits flush in tight grids without the
     *  outer border / padding. */
    flush?: boolean;
    children: Snippet;
  }

  let {
    expanded = $bindable(),
    summary,
    flush = false,
    children,
  }: Props = $props();

  function toggle(): void {
    expanded = !expanded;
  }
</script>

<section class="sk-disclosure" class:is-open={expanded} class:is-flush={flush}>
  <button
    type="button"
    class="sk-disclosure-trigger"
    aria-expanded={expanded}
    onclick={toggle}
  >
    <span class="sk-disclosure-caret" aria-hidden="true">
      <FluentIcon name="next" />
    </span>
    <span class="sk-disclosure-summary">{summary}</span>
  </button>
  {#if expanded}
    <div
      class="sk-disclosure-body"
      in:slide={collapseIn()}
      out:slide={collapseOut()}
    >
      {@render children()}
    </div>
  {/if}
</section>

<style>
  .sk-disclosure {
    border: 1px solid transparent;
    border-radius: var(--radius-lg);
    background: var(--surface-sheen), var(--glass);
    box-shadow: var(--shadow-well);
    overflow: hidden;
  }
  .sk-disclosure.is-flush {
    border: none;
    background: transparent;
    border-radius: 0;
  }

  .sk-disclosure-trigger {
    display: flex;
    align-items: center;
    min-height: var(--control-target);
    gap: var(--space-3);
    width: 100%;
    padding: var(--space-sm) var(--surface-padding);
    background: transparent;
    color: var(--fg-strong);
    border: 0;
    border-bottom: 1px solid transparent;
    text-align: start;
    cursor: pointer;
    font-family: var(--font-structure);
    font-size: var(--text-sm);
    font-weight: var(--weight-structure);
    transition:
      background var(--dur-fast) var(--ease-out),
      scale var(--dur-fast) var(--ease-out);
  }
  .sk-disclosure.is-flush .sk-disclosure-trigger {
    padding: var(--space-2) 0;
  }
  .sk-disclosure-trigger:hover {
    background: var(--bg-hover);
  }
  .sk-disclosure-trigger:active {
    scale: var(--press-scale);
  }
  .sk-disclosure-trigger:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: -2px;
  }
  .sk-disclosure.is-open .sk-disclosure-trigger {
    background: var(--bg-hover);
  }
  .sk-disclosure.is-flush.is-open .sk-disclosure-trigger {
    border-bottom-color: transparent;
  }

  .sk-disclosure-caret {
    flex: 0 0 auto;
    color: var(--fg-muted);
    font-size: var(--text-sm);
    line-height: 1;
    transform: rotate(0deg);
    transition:
      color var(--dur-fast) var(--ease-enter),
      transform var(--dur) var(--ease-enter);
  }
  .sk-disclosure.is-open .sk-disclosure-caret {
    color: var(--accent);
    transform: rotate(90deg);
  }

  .sk-disclosure-summary {
    flex: 1 1 0;
    color: var(--fg);
  }

  .sk-disclosure-body {
    padding: var(--surface-padding);
    background: var(--surface-sheen), color-mix(in srgb, var(--input-well) 72%, transparent);
  }
  .sk-disclosure.is-flush .sk-disclosure-body {
    padding: var(--space-3) 0;
  }
</style>
