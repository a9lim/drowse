<script lang="ts">
  import FluentIcon from "./FluentIcon.svelte";
  // v2 chip — the recipe-bar term, the depth badge, the role tag. A small
  // mono capsule washed in its hue.
  //
  // ``color`` is the hue (pillar or state — pass the CSS color; defaults
  // to chrome white). ``onremove`` grows the trailing ×; ``onclick`` makes
  // the body interactive (recipe chips jump to their card). Dumb about
  // content — the caller renders the label via children.

  import type { Snippet } from "svelte";

  interface Props {
    children: Snippet;
    /** Hue — any CSS color; chrome white when unset. */
    color?: string;
    /** Inactive term — readable, with a hollow/struck state treatment. */
    muted?: boolean;
    title?: string;
    onclick?: (ev: MouseEvent) => void;
    /** Grows a trailing × that fires independently of onclick. */
    onremove?: (ev: MouseEvent) => void;
    /** Accessible name for the trailing remove action. */
    removeLabel?: string;
  }

  let {
    children,
    color,
    muted = false,
    title,
    onclick,
    onremove,
    removeLabel = "Remove chip",
  }: Props = $props();
</script>

<span
  class="sk-chip"
  class:muted
  class:clickable={!!onclick}
  style:--chip-c={color}
  {...{ "aria-description": (title) }}
  role={onremove ? "group" : undefined}
>
  {#if onclick}
    <button type="button" class="body body-button" onclick={onclick}>
      {@render children()}
    </button>
  {:else}
    <span class="body">{@render children()}</span>
  {/if}
  {#if onremove}
    <button
      class="x"
      type="button"
      aria-label={removeLabel}
      onclick={(ev) => {
        ev.stopPropagation();
        onremove(ev);
      }}><FluentIcon name="dismiss" /></button
    >
  {/if}
</span>

<style>
  .sk-chip {
    --chip-c: var(--accent);
    display: inline-flex;
    align-items: center;
    min-width: 0;
    max-width: 100%;
    gap: var(--space-2);
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: 1;
    padding: var(--space-xs);
    border-radius: var(--radius-group);
    color: var(--chip-c);
    /* Borderless: the hue wash IS the chip — a touch deeper than the old
     * outlined version so the shape holds without its hairline. */
    background: var(--surface-sheen), color-mix(in srgb, var(--chip-c) 14%, transparent);
    border: 0;
    white-space: nowrap;
    transition:
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out);
  }
  .sk-chip.muted {
    background: var(--surface-sheen), color-mix(in srgb, var(--chip-c) 5%, transparent);
    box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--chip-c) 32%, transparent);
  }
  .sk-chip.muted .body {
    text-decoration: line-through;
    text-decoration-color: color-mix(in srgb, var(--chip-c) 70%, transparent);
    text-decoration-thickness: 1px;
  }
  .sk-chip.clickable {
    cursor: pointer;
  }
  .sk-chip.clickable:hover {
    background: var(--surface-sheen), color-mix(in srgb, var(--chip-c) 20%, transparent);
  }
  .body {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    min-height: var(--control-target);
  }
  .body-button {
    min-width: var(--control-target);
    background: none;
    background-image: none !important;
    border: 0;
    border-radius: var(--radius-inset);
    padding: 0;
    margin: 0;
    color: inherit;
    font: inherit;
    font-family: inherit !important;
    font-weight: inherit !important;
    text-align: start;
    cursor: pointer;
    transition: scale var(--dur-fast) var(--ease-out);
  }
  .body-button:focus-visible,
  .x:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }

  .x {
    flex: none;
    min-width: var(--control-target);
    min-height: var(--control-target);
    background: none;
    background-image: none !important;
    border: none;
    border-radius: var(--radius-inset);
    padding: 0;
    margin: 0;
    font-size: var(--text-sm);
    line-height: 1;
    color: color-mix(in srgb, var(--chip-c) 65%, transparent);
    transition:
      color var(--dur-fast) var(--ease-out),
      scale var(--dur-fast) var(--ease-out);
  }
  .body-button:active,
  .x:active {
    scale: var(--press-scale);
  }
  .x:hover {
    color: var(--chip-c);
  }
</style>
