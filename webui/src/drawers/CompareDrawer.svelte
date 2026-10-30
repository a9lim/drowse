<script lang="ts">
  import { onMount as onInterfaceMount } from "svelte";
  import { registerInterfaceController } from "../lib/workspaceController";
  import { profileCompareViewSchema } from "../lib/interfaceSchemas";
  import { ToolError } from "../lib/webmcp/types";

  import MorphText from "../lib/ui/MorphText.svelte";
  import DrawerCloseButton from "../lib/ui/DrawerCloseButton.svelte";
  // Pairwise compare drawer — cross-layer cosine matrix between two
  // named steering profiles / probes. Two dropdowns pick from registered
  // profiles and active probes; the body renders an L_A × L_B heatmap akin to
  // the correlation matrix, but indexed by layer rather than by name.
  //
  // Data: GET /drowse/v1/sessions/{id}/profiles/pairwise?a=&b= — the server
  // falls back to monitor profiles when a name isn't a registered steering
  // profile, so probe names resolve cleanly without a new endpoint.

  import { apiProfiles, ApiError } from "../lib/runtime/services";
  import { getRuntimeClient } from "../lib/runtime/registry";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import { comparisonProfileNames } from "../lib/profileAnalytics";
  import {
    closeDrawer,
    probeRack,
    vectorsState,
    refreshVectorList,
  } from "../lib/stores.svelte";
  import HeatmapCell from "../lib/charts/HeatmapCell.svelte";
  import Select from "../lib/Select.svelte";
  import type { PairwiseCompareResponse } from "../lib/types";

  // Drawer host forwards { params } — unused here, but the prop must
  // exist so the host's switch can pass it uniformly.
  let _drawerProps: { params?: unknown } = $props();
  $effect(() => { void _drawerProps.params; });

  const inferHttpProbeProfiles = getRuntimeClient().mode === "http";

  // Hosted profile lists already include the exact attached aliases that can
  // be folded to one direction.  Do not advertise lens, SAE, curved, or
  // multidimensional probes: pairwise analytics cannot compare them.
  const names = $derived(comparisonProfileNames(
    vectorsState.names,
    probeRack.active,
    probeRack.entries,
    inferHttpProbeProfiles,
  ));

  let conceptA = $state<string>("");
  let conceptB = $state<string>("");

  /** Options for the A / B pickers — same list both sides; the
   *  "(empty)" fallback only renders when the catalog is empty. */
  const nameOptions = $derived(
    names.length === 0
      ? [{ value: "", label: "(empty)" }]
      : names.map((n) => ({ value: n, label: n })),
  );
  let data = $state<PairwiseCompareResponse | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);

  // Refresh the rack on mount so newly extracted vectors show up in the
  // picker without requiring a full drawer reopen.  Cheap idempotent.
  $effect(() => {
    void refreshVectorList().catch(() => {/* non-fatal */});
  });

  // Auto-pick: first two distinct names when nothing is selected yet
  // (or when the prior selections drop out of the pool).  A drives B's
  // default to "next available != A" so the matrix renders on open
  // instead of waiting for a second click.
  $effect(() => {
    if (names.length === 0) {
      conceptA = "";
      conceptB = "";
      return;
    }
    if (!conceptA || !names.includes(conceptA)) {
      conceptA = names[0];
    }
    if (!conceptB || !names.includes(conceptB)) {
      conceptB = names.find((n) => n !== conceptA) ?? conceptA;
    }
  });

  async function load(a: string, b: string): Promise<void> {
    if (!a || !b) {
      data = null;
      return;
    }
    loading = true;
    error = null;
    try {
      data = await apiProfiles.pairwise(a, b);
    } catch (e) {
      if (e instanceof ApiError) {
        const detail =
          e.body && typeof e.body === "object" && "detail" in (e.body as object)
            ? String((e.body as { detail: unknown }).detail)
            : e.message;
        error = userFacingError(e, `Unable to compare these directions. ${detail}`);
      } else {
        error = userFacingError(e, "Unable to compare these directions. Try again.");
      }
      data = null;
    } finally {
      loading = false;
    }
  }

  // Re-fetch when either selection changes.  Idempotent server-side; no
  // need to dedupe identical (a, b) pairs.
  $effect(() => {
    void load(conceptA, conceptB);
  });

  function cellTitle(la: number, lb: number, v: number | null): string {
    return `${conceptA} L${la} × ${conceptB} L${lb}: ${v == null ? "-" : v.toFixed(3)}`;
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

  /** Cell pixel size — matches the correlation matrix.  Typical model
   * is ~30 layers so the matrix lands ~900px square; the scroll
   * container handles larger models. */
  let focusedCell = $state("Select a cell to inspect its value");
  const CELL_SIZE = 26;

  const matrix = $derived(data?.matrix ?? null);
  const layersA = $derived<number[]>(data?.layers_a ?? []);
  const layersB = $derived<number[]>(data?.layers_b ?? []);

  onInterfaceMount(() => registerInterfaceController("profile_compare", {
    schema: profileCompareViewSchema,
    read: () => ({ busy: false, values: { profile_a: conceptA, profile_b: conceptB }, available_profiles: names, loading, error }),
    update: async (change) => {
      if (false) throw new ToolError("BUSY", "Wait for this interface operation to finish.");
      for (const key of ["profile_a", "profile_b"]) if (change[key] !== undefined && !names.includes(change[key] as string)) throw new ToolError("NOT_FOUND", "Choose a profile from available_profiles.");
      if (change.profile_a !== undefined) conceptA = change.profile_a as string;
      if (change.profile_b !== undefined) conceptB = change.profile_b as string;
    },
  }));
</script>

<svelte:window onkeydown={onKeydown} />

<aside class="drawer" aria-label="Pairwise compare">
  <header class="drawer-header">
    <div class="title">
      <h2 class="eyebrow">Compare controls by layer</h2>
      <div class="name-row">
        {#if data}
          <code class="name">{conceptA} × {conceptB}</code>
          <span class="meta">{layersA.length} × {layersB.length} layers · model {data.model ?? "-"}</span>
        {:else if names.length < 2}
          <span class="meta">Choose at least two controls</span>
        {:else}
          <span class="meta">Choose two controls</span>
        {/if}
      </div>
    </div>
    <DrawerCloseButton onclick={onClose} />
  </header>

  <div class="picker-row">
    <label class="picker">
      <span class="picker-label">a</span>
      <Select
        bind:value={conceptA}
        options={nameOptions}
        disabled={names.length === 0}
        ariaLabel="Concept A"
      />
    </label>
    <label class="picker">
      <span class="picker-label">b</span>
      <Select
        bind:value={conceptB}
        options={nameOptions}
        disabled={names.length === 0}
        ariaLabel="Concept B"
      />
    </label>
  </div>

  <div class="body" aria-busy={loading}>
    {#if error}
      <div class="empty err" role="alert">Comparison failed: {error}</div>
    {:else if loading && !matrix}
      <div class="empty loading-pulse loading-placeholder" role="status">Loading layer comparison…</div>
    {:else if !matrix || layersA.length === 0 || layersB.length === 0}
      <div class="empty">Choose two available profiles or probes to compare their layers.</div>
    {:else}
      <p class="focused-cell" role="status"><MorphText text={focusedCell} numbers={false} /></p>
      <div class="grid-scroll">
        <table class="grid" style="--cell: {CELL_SIZE}px;">
          <thead>
            <tr>
              <th class="corner" scope="col">
                <span class="axis-a">{conceptA}</span>
                <span class="axis-sep">/</span>
                <span class="axis-b">{conceptB}</span>
              </th>
              {#each layersB as lb (lb)}
                <th class="col-label" scope="col" {...{ "aria-description": (conceptB) + " L" + (lb) }}>
                  <span>L{lb}</span>
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each layersA as la, i (la)}
              <tr>
                <th class="row-label" scope="row" {...{ "aria-description": (conceptA) + " L" + (la) }}>L{la}</th>
                {#each layersB as lb, j (lb)}
                  {@const v = matrix[i]?.[j] ?? null}
                  <td class="cell-td">
                    <button class="matrix-pick" type="button" onpointerenter={() => focusedCell = cellTitle(la, lb, v)} onfocus={() => focusedCell = cellTitle(la, lb, v)} onclick={() => focusedCell = cellTitle(la, lb, v)} aria-label={cellTitle(la, lb, v)}>
                    <HeatmapCell
                      value={v}
                      size={CELL_SIZE}
                      title={cellTitle(la, lb, v)}
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
  /* v2 sheet interior — the host paints the sheet surface, so the root
   * stays transparent and chrome speaks sans (data stays mono). */
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
    flex-wrap: wrap;
  }
  .name {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-md);
    font-weight: var(--weight-medium);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .meta {
    color: var(--fg-subtle);
    font-size: var(--text-sm);
    white-space: nowrap;
  }

  .picker-row {
    display: flex;
    align-items: center;
    gap: var(--space-5);
    padding: var(--space-6) var(--drawer-gutter-inline);
  }
  .picker {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    flex: 1 1 0;
    min-width: 0;
  }
  .picker-label {
    color: var(--fg-muted);
    font-size: var(--text-sm);
  }
  /* The themed Select owns its own chrome — the picker label provides
   * the only host styling. */

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

  /* Data well — sticky label cells stay OPAQUE (they occlude scrolled
   * cells), so they paint --bg rather than glass. */
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
    white-space: nowrap;
  }
  .corner .axis-a,
  .corner .axis-b {
    color: var(--fg-strong);
  }
  .corner .axis-sep {
    color: var(--fg-dim);
    padding: 0 var(--space-1);
  }
  .grid .col-label {
    color: var(--fg-dim);
    font-size: var(--text-xs);
    padding: 0;
    height: 3em;
    vertical-align: bottom;
    width: var(--cell);
    min-width: var(--cell);
    max-width: var(--cell);
    text-align: center;
  }
  .grid .col-label > span {
    display: inline-block;
    padding-bottom: var(--space-2);
  }
  .grid .cell-td {
    line-height: 0;
  }

</style>
