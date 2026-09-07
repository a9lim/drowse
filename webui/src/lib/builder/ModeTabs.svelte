<script lang="ts" generics="T extends string">
  import { slidingSelection } from "../slidingSelection";
  // Builder-drawer mode-tab row — one consistent shape across the
  // extract-vector and build-manifold drawers.  Replaces the ad-hoc
  // ``.mode-switch`` (extract) and ``.mode-tabs`` (manifold) markup so
  // the two surfaces read as the same control.
  //
  // Generic over tab-value type so callsites keep strongly-typed enums.

  interface Tab<U extends string> {
    value: U;
    label: string;
  }

  interface Props {
    value: T;
    tabs: Tab<T>[];
    /** ARIA label for the whole tablist — e.g. "Input mode". */
    ariaLabel?: string;
    onchange?: (value: T) => void;
  }

  let { value = $bindable(), tabs, ariaLabel = "Mode", onchange }: Props = $props();

  function pick(v: T): void {
    if (v === value) return;
    value = v;
    onchange?.(v);
  }

  function onKeydown(ev: KeyboardEvent): void {
    const currentButton = ev.currentTarget as HTMLButtonElement;
    const tablist = currentButton.closest('[role="tablist"]');
    if (!tablist) return;
    const buttons = [...tablist.querySelectorAll<HTMLButtonElement>('[role="tab"]')];
    const current = buttons.indexOf(currentButton);
    if (current < 0 || buttons.length === 0) return;
    let next = current;
    if (ev.key === "ArrowRight") next = (current + 1) % buttons.length;
    else if (ev.key === "ArrowLeft") next = (current - 1 + buttons.length) % buttons.length;
    else if (ev.key === "Home") next = 0;
    else if (ev.key === "End") next = buttons.length - 1;
    else return;
    ev.preventDefault();
    buttons[next].focus();
    buttons[next].click();
  }
</script>

<div
  class="sk-mode-tabs"
  use:slidingSelection
  role="tablist"
  aria-label={ariaLabel}
>
  {#each tabs as tab (tab.value)}
    <button
      type="button"
      role="tab"
      class="sk-mode-tab"
      class:active={tab.value === value}
      aria-selected={tab.value === value}
      tabindex={tab.value === value ? 0 : -1}
      onclick={() => pick(tab.value)}
      onkeydown={onKeydown}
    >{tab.label}</button>
  {/each}
</div>

<style>
  .sk-mode-tabs {
    display: flex;
    gap: var(--space-2);
    padding: var(--space-1);
    background: var(--surface-sheen), var(--glass);
    border: 0;
    border-radius: var(--radius-group);
    box-shadow: var(--shadow-well);
  }

  .sk-mode-tab {
    min-width: 0;
    min-height: var(--control-target);
    flex: 1 1 0;
    padding: var(--space-3) var(--space-4);
    background: transparent;
    color: var(--fg-dim);
    border: 0;
    border-radius: var(--radius-inset);
    font: inherit;
    font-family: var(--font-structure);
    font-weight: var(--weight-structure);
    font-size: var(--text-sm);
    text-transform: lowercase;
    cursor: pointer;
    transition: background var(--dur-fast) var(--ease-out),
      color var(--dur-fast) var(--ease-out);
  }
  .sk-mode-tab:hover:not(.active) {
    color: var(--fg);
    background: var(--bg-hover);
  }
  .sk-mode-tab.active {
    background: color-mix(in srgb, var(--accent) 9%, var(--glass-strong));
    color: var(--accent);
  }
  .sk-mode-tab:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 1px;
  }
</style>
