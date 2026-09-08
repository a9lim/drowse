<script lang="ts">
  import MorphText from "../lib/ui/MorphText.svelte";
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  // Correlation overlay — N×N magnitude-weighted cosine matrix across the
  // profiles the runtime can represent as one residual direction. Readout,
  // curved, and multidimensional probes are named explicitly as omissions.
  //
  // Data: GET /drowse/v1/sessions/{id}/correlation with no ``names=`` filter —
  // the runtime returns its supported direction pool. It deliberately omits
  // probe families/shapes for which a direction cosine is not defined.
  //
  // Layout mirrors TokenDrilldownDrawer: header (title + ✕) · sticky-
  // header table body · footer hint.  Cells reuse <HeatmapCell showValue>
  // for the printed cosine — same color mapping as the click-token grid
  // so reading a row across both surfaces stays consistent.

  import {
    closeDrawer,
    probeRack,
    refreshCorrelation,
    steerRack,
  } from "../lib/stores.svelte";
  import HeatmapCell from "../lib/charts/HeatmapCell.svelte";
  import Button from "../lib/ui/Button.svelte";
  import { omittedAnalyticsProbes } from "../lib/profileAnalytics";
  import { userFacingError } from "../lib/runtime/userFacingError";

  // Drawer host forwards { params } — unused here, but the prop must
  // exist so the host's switch can pass it uniformly.
  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  // Lazy-fetch on mount when no snapshot exists; reopens reuse the
  // cached matrix so the drawer lands instantly.
  let loading = $state(false);
  let error = $state<string | null>(null);

  async function reload(): Promise<void> {
    loading = true;
    error = null;
    try {
      // No names filter asks the runtime for every supported direction.
      await refreshCorrelation(null);
    } catch (e) {
      error = userFacingError(e, "Unable to compare the active directions. Try again.");
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    // Trigger initial load only if the snapshot is empty.  Subsequent
    // reopens read straight off ``steerRack.correlation``.
    if (!steerRack.correlation) void reload();
  });

  const data = $derived(steerRack.correlation);
  const names = $derived<string[]>(data?.names ?? []);
  const omissions = $derived(data
    ? omittedAnalyticsProbes(probeRack.active, probeRack.entries, data.names)
    : []);

  function cellTitle(a: string, b: string, v: number | null): string {
    return `${a} vs ${b}: ${v == null ? "-" : v.toFixed(3)}`;
  }

  function onClose(): void {
    closeDrawer();
  }

  function onKeydown(ev: KeyboardEvent): void {
    if (ev.key === "Escape") {
      ev.preventDefault();
      onClose();
    }
  }

  /** Cell pixel size — wider than the click-drilldown's grid because
   * we want to read the printed cosine value inside each cell, and
   * narrow column count (typical N=20-40) leaves room. */
  let focusedCell = $state("Select a cell to inspect its value");
  const CELL_SIZE = 26;
</script>

<svelte:window onkeydown={onKeydown} />

<aside class="drawer" aria-label="Correlation matrix">
  <header class="drawer-header">
    <div class="title">
      <h2 class="eyebrow">Direction correlations</h2>
      <div class="name-row">
        <span class="meta">
          {names.length} comparable {names.length === 1 ? "direction" : "directions"}
          {#if omissions.length > 0}
            · {omissions.length} unsupported {omissions.length === 1 ? "probe" : "probes"} omitted
          {/if}
        </span>
      </div>
    </div>
    <div class="actions">
      <Button
        size="sm"
        onclick={() => void reload()}
        disabled={loading}
        title="refresh"
      >{loading ? "…" : "refresh"}</Button>
      <DrawerCloseButton onclick={onClose} />
    </div>
  </header>

  {#if data && omissions.length > 0}
    <div class="omissions" role="note">
      <span>Omitted unsupported probes:</span>
      {#each omissions as omission, index (omission.name)}
        {#if index > 0}<span aria-hidden="true">, </span>{/if}
        <code {...{ "aria-description": (omission.reason) }}>{omission.name}</code>
        <span class="omission-reason"> ({omission.reason})</span>
      {/each}
    </div>
  {/if}

  <div class="body" aria-busy={loading}>
    {#if error}
      <div class="empty err" role="alert">Correlation failed: {error}</div>
    {:else if loading && !data}
      <div class="empty loading-pulse loading-placeholder" role="status">Loading correlations…</div>
    {:else if !data || names.length === 0}
      <div class="empty">No single-direction profiles are available for correlation.</div>
    {:else}
      <p class="focused-cell" role="status"><MorphText text={focusedCell} numbers={false} /></p>
      <div class="grid-scroll">
        <table class="grid" style="--cell: {CELL_SIZE}px;">
          <thead>
            <tr>
              <th class="corner" scope="col">name</th>
              {#each names as col (col)}
                <th class="col-label" scope="col" {...{ "aria-description": (col) }}>
                  <span>{col}</span>
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each names as a (a)}
              <tr>
                <th class="row-label" scope="row" {...{ "aria-description": (a) }}>{a}</th>
                {#each names as b (b)}
                  {@const v = data.matrix[a]?.[b] ?? null}
                  <td class="cell-td">
                    <button class="matrix-pick" type="button" onpointerenter={() => focusedCell = cellTitle(a, b, v)} onfocus={() => focusedCell = cellTitle(a, b, v)} onclick={() => focusedCell = cellTitle(a, b, v)} aria-label={cellTitle(a, b, v)}>
                    <HeatmapCell
                      value={v}
                      size={CELL_SIZE}
                      title={cellTitle(a, b, v)}
                    />
                    </button>
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}
  </div>

</aside>

<style>
  .matrix-pick { display: grid; place-items: center; min-width: 44px; min-height: 44px; padding: 0; border: 0; background: transparent; cursor: crosshair; }
  .focused-cell { min-height: 2em; font: var(--text-sm)/1.5 var(--font-mono); overflow-wrap: anywhere; }
  /* v2 sheet interior — the host paints the sheet surface (glass hairline,
   * radius, --bg-alt fill), so the root is transparent; chrome speaks sans
   * and every value/identifier/expression sits in mono. */
  .drawer {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    background: transparent;
    color: var(--fg);
    font-family: var(--font-ui);
    font-size: var(--text);
  }

  .drawer-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--space-5);
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .title {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
    min-width: 0;
  }
  .eyebrow {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .name-row {
    display: flex;
    align-items: baseline;
    gap: var(--space-3);
    min-width: 0;
  }
  .meta {
    color: var(--fg-subtle);
    font-size: var(--text-sm);
    white-space: nowrap;
  }
  .actions {
    display: flex;
    gap: var(--space-3);
    align-items: center;
    flex: none;
  }
  .omissions {
    margin: 0 var(--space-6);
    padding: var(--surface-padding);
    border-radius: var(--radius);
    background: var(--surface-sheen), var(--glass);
    color: var(--fg-muted);
    font-size: var(--text-xs);
    line-height: 1.6;
  }
  .omissions code {
    color: var(--fg-dim);
  }
  .omission-reason {
    color: var(--fg-subtle);
  }

  .body {
    flex: 1 1 auto;
    overflow: auto;
    min-height: 0;
    padding: var(--drawer-gutter-block) var(--drawer-gutter-inline);
  }
  .empty {
    color: var(--fg-muted);
    padding: var(--space-6) 0;
    line-height: 1.5;
    max-width: 62ch;
  }
  .empty.err {
    color: var(--accent-red);
  }

  /* Data well — recessed matrix.  Every column/row label here IS an
   * identifier (vector/probe name), not a category header, so the whole
   * grid speaks mono (matches the probes tab in TokenDrilldownDrawer). */
  .grid-scroll {
    overflow: auto;
    max-height: 100%;
    border-radius: var(--radius);
    background: var(--bg);
  }
  .grid {
    border-collapse: separate;
    border-spacing: 1px;
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
  }
  .grid th,
  .grid td {
    padding: 0;
    margin: 0;
    background: var(--bg);
  }
  .grid thead th {
    position: sticky;
    top: 0;
    z-index: 2;
    box-shadow: var(--shadow-sticky);
  }
  .grid .row-label {
    position: sticky;
    inset-inline-start: 0;
    z-index: 1;
    text-align: end;
    padding: 0 var(--space-3) 0 var(--space-2);
    color: var(--fg-dim);
    font-size: var(--text-xs);
    box-shadow: var(--shadow-sticky-inline);
    white-space: nowrap;
  }
  .grid .corner {
    position: sticky;
    top: 0;
    inset-inline-start: 0;
    z-index: 3;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    text-align: start;
    padding: var(--space-1) var(--space-3);
    box-shadow: var(--shadow-sticky), var(--shadow-sticky-inline);
  }
  .grid .col-label {
    color: var(--fg-dim);
    font-size: var(--text-xs);
    padding: 0;
    height: 7em;
    vertical-align: bottom;
    width: var(--cell);
    min-width: var(--cell);
    max-width: var(--cell);
  }
  .grid .col-label > span {
    display: inline-block;
    transform: rotate(-60deg);
    transform-origin: left bottom;
    white-space: nowrap;
    padding-bottom: var(--space-2);
  }
  .grid .cell-td {
    line-height: 0;
  }

</style>
