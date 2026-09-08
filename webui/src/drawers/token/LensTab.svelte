<script lang="ts">
  import MorphText from "../../lib/ui/MorphText.svelte";
  import { chartValue } from "../../lib/charts/chartValues";
  import RollingNumber from "../../lib/ui/RollingNumber.svelte";
  // J-lens tab — pinned ``jlens/<word>`` probe readings (when captured
  // live), the layer-aggregated chip row, then the all-fitted-layer
  // readout matrix: each row ranks softmax(W_U · norm(J_l h)) at the
  // forward that produced this token — what that layer's residual was
  // disposed to make the model say.

  import type {
    LensAggregateTokenJSON,
    LensTokenReadoutJSON,
    ScalarReadingJSON,
  } from "../../lib/types";
  import Bar from "../../lib/charts/Bar.svelte";
  import LayerStrip from "../../panels/rack/LayerStrip.svelte";
  import ProbeReadingRow from "../../panels/rack/ProbeReadingRow.svelte";
  import RackCard from "../../panels/rack/RackCard.svelte";
  import type { ReplayReadout } from "./readout.svelte";
  import EmptyState from "./EmptyState.svelte";
  import PinnedReadings from "./PinnedReadings.svelte";
  import InstrumentHeader from "./InstrumentHeader.svelte";
  import DetailSection from "./DetailSection.svelte";
  import DetailCardHeader from "./DetailCardHeader.svelte";
  import JLensMissingState from "@runtime-jlens-missing";

  let {
    readout,
    steered = $bindable(),
    jlensFitted,
    hasReplayContext,
    pinned,
    modelId,
    replayAvailable,
  }: {
    readout: ReplayReadout<LensTokenReadoutJSON>;
    steered: boolean;
    jlensFitted: boolean;
    hasReplayContext: boolean;
    /** Live-captured pinned-probe readings from the token's envelope. */
    pinned: Record<string, ScalarReadingJSON> | null;
    /** For the unfitted-state CLI hint. */
    modelId: string | null;
    replayAvailable: boolean;
  } = $props();

  const showToggle = $derived(
    replayAvailable && ((readout.data?.steering ?? null) !== null || !steered),
  );

  const columnCount = $derived(
    Math.max(0, ...(readout.data?.layers.map((row) => row.tokens.length) ?? [])),
  );

  const layerCount = $derived(readout.data?.layers.length ?? 0);

  const determinateProgress = $derived(
    readout.progress !== null && readout.progress.progress > 0,
  );

  const progressPercent = $derived(
    Math.round((readout.progress?.progress ?? 0) * 100),
  );

  const progressTitle = $derived(
    readout.progress?.phase === "readout"
      ? "Reading the J-lens"
      : readout.progress?.phase === "queued"
        ? "Waiting for the model"
        : "Preparing this token",
  );

  function cellStyle(logprob: number): string {
    const p = Math.min(1, Math.exp(logprob));
    const pct = Math.round(p * 30);
    return `background: color-mix(in srgb, var(--pillar-lens) ${pct}%, var(--bg));`;
  }

  function cellTitle(layer: number, t: { token: string; logprob: number }): string {
    const p = Math.exp(t.logprob);
    const pTxt = p >= 0.001 ? p.toFixed(4) : p.toExponential(2);
    return `L${layer} · ${JSON.stringify(t.token)} · ${chartValue(p, true)} · p=${pTxt} · logprob=${t.logprob.toFixed(3)}`;
  }

  function cellText(t: { token: string }): string {
    const trimmed = t.token.trim();
    return trimmed.length > 0 ? trimmed : JSON.stringify(t.token);
  }

  function displayToken(token: string): string {
    return token.trim() || JSON.stringify(token);
  }

  interface AggregateCell {
    layer: number;
    value: number | null;
    title: string;
  }

  function aggregateCells(chip: LensAggregateTokenJSON): AggregateCell[] {
    return (readout.data?.layers ?? []).map((row) => {
      const hit = row.tokens.find((token) => token.token === chip.token);
      const value = hit ? Math.exp(hit.logprob) : null;
      return {
        layer: row.layer,
        value,
        title: value == null
          ? `L${row.layer} · below top-${columnCount}`
          : `L${row.layer} · ${chartValue(value, true)} · p ${value.toPrecision(3)}`,
      };
    });
  }

  let focusedCell = $state("Select a vocabulary cell to inspect it");
  let focusedPosition = $state(0);
  $effect(() => { void readout.data; focusedPosition = 0; focusedCell = "Select a vocabulary cell to inspect it"; });
  function inspectCell(event: KeyboardEvent) {
    const cells = [...(event.currentTarget as HTMLElement).querySelectorAll<HTMLElement>(".lens-cell")];
    if (!cells.length) return;
    if (event.key === "ArrowRight") focusedPosition = Math.min(cells.length - 1, focusedPosition + 1);
    else if (event.key === "ArrowLeft") focusedPosition = Math.max(0, focusedPosition - 1);
    else if (event.key === "ArrowDown") focusedPosition = Math.min(cells.length - 1, focusedPosition + columnCount);
    else if (event.key === "ArrowUp") focusedPosition = Math.max(0, focusedPosition - columnCount);
    else if (event.key === "Home") focusedPosition = 0;
    else if (event.key === "End") focusedPosition = cells.length - 1;
    else return;
    event.preventDefault();
    focusedCell = cells[focusedPosition].getAttribute('aria-description') ?? '';
    cells[focusedPosition].scrollIntoView({block: "nearest", inline: "nearest"});
  }

  function aggregateScale(cells: AggregateCell[]): number {
    return Math.max(...cells.map((cell) => cell.value ?? 0), 1e-12);
  }

  function visibleProbability(logprob: number): string {
    const p = Math.exp(logprob);
    return p >= 0.001 ? p.toFixed(3) : p.toExponential(1);
  }

  function handVerticalWheelToDrawer(event: WheelEvent): void {
    if (event.ctrlKey || Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
    const drawerBody = (event.currentTarget as HTMLElement).closest<HTMLElement>("[data-token-details-scroll]");
    if (!drawerBody) return;
    const unit = event.deltaMode === WheelEvent.DOM_DELTA_LINE
      ? 16
      : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
        ? drawerBody.clientHeight
        : 1;
    const before = drawerBody.scrollTop;
    drawerBody.scrollTop += event.deltaY * unit;
    if (drawerBody.scrollTop !== before) event.preventDefault();
  }
</script>

{#if readout.loading}
  <div class="readout-progress loading-pulse" role="status" aria-live="polite">
    <div class="progress-heading">
      <span><MorphText text={progressTitle} numbers={false} /></span>
      <code>{#if determinateProgress}<RollingNumber value={progressPercent} />%{:else}starting{/if}</code>
    </div>
    <div
      class="progress-track"
      class:indeterminate={!determinateProgress}
      role="progressbar"
      aria-label="J-lens readout progress"
      aria-valuemin="0"
      aria-valuemax="100"
      aria-valuenow={determinateProgress ? progressPercent : undefined}
      aria-valuetext={determinateProgress ? `${progressPercent}%` : "Starting"}
    >
      <span style={determinateProgress ? `width: ${progressPercent}%` : undefined}></span>
    </div>
    <p>{readout.progress?.message ?? "Waiting for the model"}</p>
    <p class="progress-note">You can inspect another token while this finishes. This result will be kept.</p>
  </div>
{:else if readout.error}
  <EmptyState title={`readout: ${readout.error}`} />
{:else if readout.data}
  <InstrumentHeader
    origin={readout.origin}
    source={readout.source}
    steering={readout.data.steering}
    bind:steered
    {showToggle}
    accent="var(--pillar-lens)"
  />
  {#if readout.origin === "captured" && pinned && Object.keys(pinned).length > 0}
    <PinnedReadings readings={pinned} accent="--pillar-lens" shape="square" />
  {/if}
  {#if (readout.data.aggregate ?? []).length > 0}
    <DetailSection
      title="AGGREGATE WORKSPACE"
      count={`${readout.data.aggregate?.length ?? 0} tokens`}
      accent="var(--pillar-lens)"
    >
      <div class="aggregate-grid" role="list" aria-label="Aggregate lens tokens">
        {#each readout.data.aggregate ?? [] as chip, i (i)}
          {@const cells = aggregateCells(chip)}
          {@const hitCount = cells.filter((cell) => cell.value != null).length}
          <div role="listitem">
            <RackCard
              accent="--pillar-lens"
              disabled={false}
              active={chip.token === readout.data.token_text}
            >
              {#snippet statline()}
                <DetailCardHeader
                  primary={displayToken(chip.token)}
                  meta={`@${chip.com.toFixed(2)} ±${chip.spread.toFixed(2)}`}
                  metaTitle="Where this word’s probability is concentrated across layers: 0 is the first layer, 1 is the last. The ± value shows how widely it is spread."
                  badge={chip.token === readout.data?.token_text ? "generated" : null}
                >
                  {#snippet lead()}<span>#{i + 1}</span>{/snippet}
                </DetailCardHeader>
              {/snippet}
              {#snippet body()}
                <ProbeReadingRow ariaLabel={`Strength ${chip.strength.toFixed(3)}`}>
                  {#snippet left()}<span class="row-label">strength</span>{/snippet}
                  {#snippet bar()}
                    <Bar percentage value={chip.strength} max={1} color="var(--pillar-lens)" />
                  {/snippet}
                  {#snippet middle()}<span class="row-context">{hitCount}/{layerCount} layers</span>{/snippet}
                  {#snippet right()}<span class="row-value"><MorphText text={chip.strength.toFixed(3)} /></span>{/snippet}
                </ProbeReadingRow>
                <LayerStrip
                  {cells}
                  scale={aggregateScale(cells)}
                  positiveColor="var(--layer-cell-lens)"
                  ariaLabel={`Per-layer strength for ${displayToken(chip.token)}`}
                />
              {/snippet}
            </RackCard>
          </div>
        {/each}
      </div>
    </DetailSection>
  {/if}
  <DetailSection
    title="LAYER × VOCABULARY"
    count={`${readout.data.layers.length} layers × ${columnCount} ranks`}
    accent="var(--pillar-lens)"
  >
    <p class="focused-readout"><MorphText text={focusedCell} numbers={false} /></p>
    <div class="grid-scroll" role="slider" aria-label="Vocabulary matrix cell" aria-valuemin={0} aria-valuemax={Math.max(0, readout.data.layers.reduce((sum, row) => sum + row.tokens.length, 0) - 1)} aria-valuenow={focusedPosition} aria-valuetext={focusedCell} tabindex="0" onkeydown={inspectCell} onwheel={handVerticalWheelToDrawer}
      onpointermove={(event) => { const cell = (event.target as Element).closest<HTMLElement>(".lens-cell"); if (cell) { focusedCell = cell.getAttribute('aria-description') ?? ''; focusedPosition = [...event.currentTarget.querySelectorAll(".lens-cell")].indexOf(cell); } }}
      onpointerdown={(event) => { const cell = (event.target as Element).closest<HTMLElement>(".lens-cell"); if (cell) focusedCell = cell.getAttribute('aria-description') ?? ''; }}>
      <table class="lens-table">
        <thead>
          <tr>
            <th class="corner">L \ rank</th>
            {#each { length: columnCount } as _, i (i)}
              <th class="num">{i + 1}</th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each readout.data.layers as row (row.layer)}
            <tr>
              <th class="row-label">
                L{row.layer}
              </th>
              {#each row.tokens as cell (cell.id)}
                <td
                  class="lens-cell"
                  class:hit={cell.id === readout.data.token_id}
                  style={cellStyle(cell.logprob)}
                  {...{ "aria-description": (cellTitle(row.layer, cell)) }}
                >
                  <span class="cell-token">{cellText(cell)}</span>
                  <span class="cell-prob">p {visibleProbability(cell.logprob)}</span>
                </td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </DetailSection>
{:else if !jlensFitted}
  <JLensMissingState {modelId} />
{:else if !hasReplayContext}
  <EmptyState
    title="no raw decode record"
    detail="replay needs a loom node generated with raw-decode capture in this session"
  />
{:else}
  <EmptyState title="no readout" />
{/if}

<style>
  .focused-readout { min-height: 3em; margin: 0; color: var(--fg-dim); font: var(--text-xs)/1.5 var(--font-mono); }
  .readout-progress {
    max-width: 62ch;
    padding: var(--surface-padding);
    border-radius: var(--radius);
  }
  .progress-heading {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-4);
    color: var(--fg-dim);
    font-family: var(--font-ui);
    font-weight: var(--weight-medium);
  }
  .progress-heading code {
    color: var(--pillar-lens);
    font-family: var(--font-mono);
    font-weight: var(--weight-medium);
    font-variant-numeric: tabular-nums;
  }
  .progress-track {
    height: 6px;
    margin-top: var(--space-3);
    overflow: hidden;
    border-radius: var(--radius-pill);
    background: var(--bg);
    box-shadow: var(--shadow-rack);
  }
  .progress-track span {
    display: block;
    height: 100%;
    min-width: 2px;
    border-radius: inherit;
    background: var(--pillar-lens);
    transition: width var(--dur) linear;
  }
  .progress-track.indeterminate span {
    width: 38%;
    min-width: 72px;
    animation: readout-wait 1.15s linear infinite;
  }
  @keyframes readout-wait {
    from { transform: translateX(-110%); }
    to { transform: translateX(290%); }
  }
  .readout-progress p {
    margin: var(--space-2) 0 0;
    color: var(--fg-muted);
    font-size: var(--text-sm);
    line-height: 1.5;
  }
  .readout-progress .progress-note {
    color: var(--fg-subtle);
    font-size: var(--text-xs);
  }
  @media (prefers-reduced-motion: reduce) {
    .progress-track.indeterminate span {
      animation: none;
      transform: none;
    }
  }
  .aggregate-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3);
  }
  .row-label,
  .row-context,
  .row-value {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
  }
  .row-label,
  .row-value {
    text-align: end;
  }
  .row-value {
    color: var(--pillar-lens);
  }

  .grid-scroll {
    overflow-x: auto;
    border-radius: var(--radius-lg);
    background: var(--bg);
    box-shadow: var(--shadow-rack);
  }
  .lens-table {
    width: max-content;
    min-width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-variant-numeric: tabular-nums;
    font-size: var(--text-sm);
  }
  .lens-table th,
  .lens-table td {
    padding: var(--space-2) var(--space-3);
    text-align: start;
    background: var(--bg);
  }
  .lens-table tbody tr:last-child > :last-child {
    border-end-end-radius: var(--radius-lg);
  }
  .lens-table thead th {
    position: sticky;
    top: 0;
    z-index: 2;
    color: var(--fg-muted);
    font-family: var(--font-ui);
    font-weight: var(--weight-medium);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    box-shadow: var(--shadow-sticky);
  }
  .lens-table .corner {
    position: sticky;
    inset-inline-start: 0;
    z-index: 3;
    box-shadow: var(--shadow-sticky), var(--shadow-sticky-inline);
  }
  .lens-table .row-label {
    position: sticky;
    inset-inline-start: 0;
    z-index: 1;
    text-align: end;
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-weight: var(--weight-normal);
    font-size: var(--text-xs);
    box-shadow: var(--shadow-sticky-inline);
    white-space: nowrap;
  }
  .lens-cell {
    font-family: var(--font-mono);
    color: var(--fg-strong);
    min-width: 92px;
    max-width: 13ch;
    overflow: hidden;
    vertical-align: top;
  }
  .cell-token,
  .cell-prob {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .cell-prob {
    margin-top: var(--space-1);
    color: var(--fg-strong);
    font-size: var(--text-2xs);
  }
  .lens-cell.hit {
    box-shadow: inset 0 0 0 2px var(--pillar-lens);
  }
  @media (forced-colors: active) {
    .lens-cell.hit {
      outline: 2px solid Highlight;
      outline-offset: -2px;
    }
  }
  @media (max-width: 760px) {
    .aggregate-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
