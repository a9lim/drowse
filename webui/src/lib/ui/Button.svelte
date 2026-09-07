<script lang="ts">
  // v2 button — the one button. Four variants on one skeleton:
  //   solid  — accent-filled, dark text; the primary action of a surface
  //   ghost  — borderless glass fill + wash on hover; the default workhorse
  //   flat   — transparent compact action; color-only hover feedback
  //   danger — red ghost; destructive affordances
  //
  // ``accent`` retints a button to a pillar hue (pass the CSS color) —
  // solid fills with it, ghost/danger tint text + hover wash.
  // Busy feedback is decorative; callers still own labels and disabling.

  import type { Snippet } from "svelte";

  interface Props {
    children: Snippet;
    variant?: "solid" | "ghost" | "flat" | "danger";
    size?: "sm" | "md";
    /** Pillar/state hue override — any CSS color. */
    accent?: string;
    disabled?: boolean;
    busy?: boolean;
    static?: boolean;
    title?: string;
    ariaLabel?: string;
    type?: "button" | "submit";
    onclick?: (ev: MouseEvent) => void;
  }

  let {
    children,
    variant = "ghost",
    size = "md",
    accent,
    disabled = false,
    busy = false,
    static: isStatic = false,
    title,
    ariaLabel,
    type = "button",
    onclick,
  }: Props = $props();
</script>

<button
  class="sk-btn {variant} {size}"
  class:accented={accent !== undefined}
  class:loading-pulse={busy}
  class:static={isStatic}
  style:--btn-accent={accent}
  style:--btn-solid-fill={accent ?? "var(--action-bg)"}
  style:--btn-solid-ink={accent ? "var(--text-on-accent)" : "var(--action-ink)"}
  {disabled}
  {title}
  aria-label={ariaLabel}
  aria-busy={busy || undefined}
  {type}
  {onclick}
>
  {@render children()}
</button>

<style>
  .sk-btn {
    --btn-accent: var(--accent);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--control-label-gap);
    border-radius: var(--radius);
    font-family: var(--font-structure);
    font-weight: var(--weight-structure);
    min-width: var(--control-target);
    max-width: 100%;
    line-height: 1.4;
    overflow-wrap: anywhere;
    border: 1px solid transparent;
    transition:
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out),
      box-shadow var(--dur-fast) var(--ease-out),
      color var(--dur-fast) var(--ease-out),
      filter var(--dur-fast) var(--ease-out),
      scale var(--dur-fast) var(--ease-out);
  }
  .sk-btn:active:not(:disabled):not(.static) {
    scale: var(--press-scale);
  }
  .sk-btn:disabled {
    cursor: not-allowed;
    opacity: 0.45;
  }

  .md {
    font-size: var(--text-sm);
    padding: var(--space-xs) var(--space-sm);
    min-height: var(--control-field);
  }
  .sm {
    min-height: var(--control-target);
    font-size: var(--text-sm);
    padding: var(--space-xs) var(--space-xs);
    border-radius: var(--radius-sm);
  }

  /* Solid buttons use one quiet, legible accent fill. */
  .solid {
    background: var(--control-sheen), var(--btn-solid-fill);
    color: var(--btn-solid-ink);
    border-color: transparent;
    box-shadow: var(--shadow-control);
  }
  .solid:hover:not(:disabled) {
    background: var(--control-sheen), color-mix(in srgb, var(--btn-solid-fill) 94%, white);
    box-shadow: var(--shadow-control-hover);
  }

  /* Borderless doctrine: the control floats UP on a glass fill — shape
   * without an outline. Hover deepens the wash toward the accent. */
  .ghost {
    background: var(--control-sheen), var(--glass);
    color: var(--fg-dim);
    border-color: transparent;
    box-shadow: var(--shadow-control);
  }
  .ghost:hover:not(:disabled) {
    background: var(--control-sheen), var(--glass-strong);
    color: var(--fg);
    box-shadow: var(--shadow-control-hover);
  }
  .ghost.accented {
    background: var(--control-sheen), color-mix(in srgb, var(--btn-accent) 7%, var(--glass));
    color: var(--btn-accent);
  }
  .ghost.accented:hover:not(:disabled) {
    background: var(--control-sheen), color-mix(in srgb, var(--btn-accent) 10%, var(--glass));
    color: var(--btn-accent);
  }

  .flat {
    background: transparent;
    color: var(--fg-muted);
    border-color: transparent;
  }
  .flat:hover:not(:disabled) {
    background: transparent;
    color: var(--btn-accent);
  }

  .danger {
    background: var(--control-sheen), var(--danger-bg);
    color: var(--accent-red);
    border-color: transparent;
    box-shadow: var(--shadow-control);
  }
  .danger:hover:not(:disabled) {
    background: var(--control-sheen), var(--danger-hover);
  }

  .sk-btn:active:not(:disabled):not(.flat) { box-shadow: var(--shadow-control-pressed); }

  .sk-btn:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }
</style>
