<script lang="ts">
  import type { Snippet } from "svelte";
  import FluentIcon from "../../../lib/ui/FluentIcon.svelte";

  // A disclosure whose body slides open: grid rows transition 0fr → 1fr, so the motion is interruptible.
  let { label, openLabel, children, open = $bindable(false) }: {
    label: string;
    openLabel?: string;
    children: Snippet;
    open?: boolean;
  } = $props();
  const id = $props.id();
</script>

<div class="expander" class:open>
  <button type="button" class="toggle" aria-expanded={open} aria-controls={id} onclick={() => (open = !open)}>
    <span>{open ? (openLabel ?? label) : label}</span>
    <span class="chev" aria-hidden="true"><FluentIcon name="down" size={16} /></span>
  </button>
  <div class="body" {id} inert={!open}>
    <div class="inner">{@render children()}</div>
  </div>
</div>

<style>
  .expander { display: grid; min-width: 0; }
  .toggle {
    display: inline-flex; align-items: center; justify-self: start; gap: var(--space-2); min-height: var(--control-target);
    padding-inline: var(--space-4); border: 0; border-radius: var(--radius-pill); background: var(--neuron-chip-bg, var(--glass)); box-shadow: inset 0 0 0 1px var(--glass-line);
    color: var(--fg-dim); font: inherit; font-size: var(--text-sm); cursor: pointer;
    transition: background-color var(--dur-fast) var(--ease-out), color var(--dur-fast) var(--ease-out), scale var(--dur-fast) var(--ease-out);
  }
  .toggle:active { scale: var(--press-scale); }
  .toggle:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  @media (hover: hover) { .toggle:hover { background: var(--glass-strong); color: var(--fg); } }
  .chev { display: grid; transition: rotate var(--dur-slow) var(--ease-out); }
  .open .chev { rotate: 180deg; }
  .body { display: grid; grid-template-rows: 0fr; opacity: 0; transition: grid-template-rows var(--dur-slow) var(--ease-out), opacity var(--dur-slow) var(--ease-out); }
  .open .body { grid-template-rows: 1fr; opacity: 1; }
  .inner { min-height: 0; overflow: hidden; }
  .open .inner { padding-top: var(--space-4); }
  @media (prefers-reduced-motion: reduce) { .body, .chev, .toggle { transition: none; } .toggle:active { scale: none; } }
</style>
