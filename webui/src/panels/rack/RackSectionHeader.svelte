<script lang="ts" generics="T extends string">
  import MorphText from "../../lib/ui/MorphText.svelte";
  // Canonical STEER/PROBE header used by all four inspector pillars. This
  // owns the title/live/count/sort rhythm so future tabs cannot drift by a
  // pixel or silently omit one of the shared controls.

  import Select from "../../lib/Select.svelte";
  import InfoTip from "../../lib/ui/InfoTip.svelte";

  interface SortOption<U> {
    value: U;
    label: string;
    disabled?: boolean;
  }

  let {
    title,
    help,
    count,
    live = null,
    liveBusy = false,
    liveTitle = "",
    liveLabel,
    liveHelp,
    onLiveToggle,
    sortValue,
    sortOptions = [],
    sortAriaLabel = "Sort cards by",
    onSortChange,
  }: {
    title: string;
    help?: string;
    count?: string;
    live?: boolean | null;
    liveBusy?: boolean;
    liveTitle?: string;
    liveLabel?: string;
    liveHelp?: string;
    onLiveToggle?: () => void;
    sortValue?: T;
    sortOptions?: SortOption<T>[];
    sortAriaLabel?: string;
    onSortChange?: (value: T) => void;
  } = $props();

  const hasSort = $derived(sortValue !== undefined && sortOptions.length > 0);
  const contextualLiveLabel = $derived(liveLabel ?? `${title} live readings`);
</script>

<header class="header">
  <div class="header-text">
    <span class="title">{title}</span>
    {#if help}<InfoTip text={help} label={`About ${title}`} />{/if}
    {#if live !== null}
      <span class="live-control">
        <button
          type="button"
          class="toggle"
          class:on={live}
          disabled={liveBusy}
          onclick={onLiveToggle}
          {...{ "aria-description": (liveTitle) }}
          aria-label={`${live ? "Turn off" : "Turn on"} ${contextualLiveLabel}`}
          aria-pressed={live}
        >
          <MorphText text={live ? "Live on" : "Live off"} numbers={false} />
        </button>
        {#if liveHelp}
          <InfoTip text={liveHelp} label={`About ${contextualLiveLabel}`} />
        {/if}
      </span>
    {/if}
    {#if count}<span class="count" aria-live="polite"><MorphText text={count} /></span>{/if}
  </div>

  {#if hasSort}
    <label class="sort">
      <span class="sort-select">
        <Select
          value={sortValue as T}
          options={sortOptions}
          onchange={onSortChange}
          ariaLabel={sortAriaLabel}
        />
      </span>
    </label>
  {/if}
</header>

<style>
  .header {
    display: flex;
    flex: 0 0 auto;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
    min-height: var(--control-compact);
    padding-bottom: var(--space-3);
  }
  .header-text {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--control-label-gap);
    min-width: 0;
  }
  .title {
    white-space: nowrap;
    color: var(--accent);
    font-size: var(--text-sm);
    font-weight: var(--weight-bold);
    letter-spacing: 0;
    text-transform: uppercase;
  }
  .count {
    color: var(--fg-muted);
    font-size: var(--text-sm);
    flex: 0 0 auto;
  }
  .toggle {
    white-space: nowrap;
    min-height: var(--control-target);
    padding: var(--space-xs) var(--space-3);
    color: var(--fg-muted);
    background: var(--glass);
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    font-size: var(--text-sm);
    cursor: pointer;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out);
  }
  .live-control {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
  }
  .toggle:hover:not(:disabled) {
    color: var(--fg);
    background: var(--glass-strong);
  }
  .toggle.on {
    color: var(--accent);
    background: var(--accent-subtle);
  }
  .toggle:disabled {
    cursor: default;
    opacity: 0.5;
  }
  .sort {
    display: inline-flex;
    flex: 0 0 auto;
    align-items: center;
    gap: var(--space-3);
    min-width: 0;
  }
  .sort-select {
    display: inline-flex;
    min-width: 8em;
  }

  @media (max-width: 620px) {
    .header {
      align-items: flex-start;
      flex-direction: column;
      gap: var(--space-2);
    }

    .header-text {
      flex-wrap: wrap;
      width: 100%;
    }

    .title { white-space: nowrap; }
    .sort { align-self: flex-end; }
  }
</style>
