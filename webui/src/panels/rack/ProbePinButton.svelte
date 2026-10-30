<script lang="ts">
  // Shared pin/unpin control for J-lens and SAE discovery/persistent cards.
  // The shape is the pillar identity; the hit target, baseline, hover, and
  // disabled behavior are deliberately identical.

  import RackMarker, { type RackMarkerShape } from "./RackMarker.svelte";

  let {
    shape,
    pinned,
    disabled = false,
    onclick,
    ariaLabel,
    title,
  }: {
    shape: RackMarkerShape;
    pinned: boolean;
    disabled?: boolean;
    onclick: () => void;
    ariaLabel: string;
    title: string;
  } = $props();
</script>

<button
  type="button"
  class="pin"
  class:pinned
  {disabled}
  {onclick}
  {...{ "aria-description": (title) }}
  aria-label={ariaLabel}
  aria-pressed={pinned}
>
  <span class="pin-icon" class:visible={!pinned} aria-hidden="true"><RackMarker {shape} /></span>
  <span class="pin-icon" class:visible={pinned} aria-hidden="true"><RackMarker {shape} filled /></span>
</button>

<style>
  .pin {
    display: inline-grid;
    place-items: center;
    inline-size: var(--control-target);
    block-size: var(--control-target);
    padding: 0;
    color: var(--fg-muted);
    background: transparent;
    border: 0;
    border-radius: var(--radius-sm);
    flex: 0 0 var(--control-target);
    cursor: pointer;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      scale var(--dur-fast) var(--ease-out);
  }
  .pin:active:not(:disabled) {
    scale: var(--press-scale);
  }
  .pin:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }
  .pin-icon {
    grid-area: 1 / 1;
    display: inline-grid;
    pointer-events: none;
    opacity: 0;
    scale: 0.25;
    filter: blur(4px);
    transition:
      opacity var(--dur-slow) cubic-bezier(0.2, 0, 0, 1),
      scale var(--dur-slow) cubic-bezier(0.2, 0, 0, 1),
      filter var(--dur-slow) cubic-bezier(0.2, 0, 0, 1);
  }
  .pin-icon.visible {
    opacity: 1;
    scale: 1;
    filter: blur(0);
  }
  .pin.pinned {
    color: var(--card-accent);
  }
  .pin:hover:not(:disabled) {
    color: var(--card-accent);
    background: color-mix(in srgb, var(--card-accent) 8%, transparent);
  }
  .pin.pinned:hover:not(:disabled) {
    color: var(--accent-red);
    background: color-mix(in srgb, var(--accent-red) 8%, transparent);
  }
  .pin:disabled {
    cursor: default;
    opacity: var(--disabled-opacity);
  }
</style>
