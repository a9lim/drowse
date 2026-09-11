<script lang="ts">
  import MorphText from "../../lib/ui/MorphText.svelte";
  import RollingNumber from "../../lib/ui/RollingNumber.svelte";
  // Geometry tab — the full whitened Monitor reading for every attached
  // geometry probe at the forward that produced this token: all
  // coordinate axes, subspace fraction, nearest nodes, soft assignment,
  // tube membership, per-layer strip, depth CoM.  Captured envelopes
  // render directly; the replay endpoint covers aggregate-only
  // generations and probes attached after the fact.  Achromatic like the
  // rack's monitor cards — the family splits subspace-white /
  // manifold-violet, so no single pillar hue applies.

  import Bar from "../../lib/charts/Bar.svelte";
  import LayerStrip from "../../panels/rack/LayerStrip.svelte";
  import GeometryLayerReadings from "./GeometryLayerReadings.svelte";
  import ProbeReadingRow from "../../panels/rack/ProbeReadingRow.svelte";
  import RackCard from "../../panels/rack/RackCard.svelte";
  import RackMarker from "../../panels/rack/RackMarker.svelte";
  import { probeRack, probeAxisScale, openDrawer } from "../../lib/stores.svelte";
  import Button from "../../lib/ui/Button.svelte";
  import { drawerAvailability } from "../../lib/runtime/ui-capabilities";
  import type { ProbeReadingJSON } from "../../lib/types";
  import type { GeometryTokenReadout, ReplayReadout } from "./readout.svelte";
  import EmptyState from "./EmptyState.svelte";
  import InstrumentHeader from "./InstrumentHeader.svelte";
  import DetailSection from "./DetailSection.svelte";
  import DetailCardHeader from "./DetailCardHeader.svelte";
  import EvidenceChips from "./EvidenceChips.svelte";

  let {
    readout,
    steered = $bindable(),
    hasGeometryProbes,
    hasReplayContext,
    replayAvailable,
    returnToToken,
  }: {
    readout: ReplayReadout<GeometryTokenReadout>;
    steered: boolean;
    /** ≥1 attached Monitor probe (the replay endpoint 400s on an empty
     *  roster, so absence renders the attach hint instead). */
    hasGeometryProbes: boolean;
    /** The token has a raw decode index + backing loom node. */
    hasReplayContext: boolean;
    replayAvailable: boolean;
    returnToToken: { turnIdx: number; tokenIdx: number; isThinking: boolean };
  } = $props();

  const rows = $derived<[string, ProbeReadingJSON][]>(
    Object.entries(readout.data?.readings ?? {}).sort(([a], [b]) =>
      a.localeCompare(b, undefined, { sensitivity: "base" }),
    ),
  );
  const authoring = drawerAvailability("manifold_builder");

  function setupProbe(create = false): void {
    openDrawer(create ? "manifold_builder" : "subspace", {
      returnToToken,
      ...(create ? { mode: "discover" } : {}),
    });
  }

  const showToggle = $derived(
    replayAvailable && ((readout.data?.steering ?? null) !== null || !steered),
  );

  const determinateProgress = $derived(
    readout.progress !== null && readout.progress.progress > 0,
  );

  const progressPercent = $derived(
    Math.round((readout.progress?.progress ?? 0) * 100),
  );

  const progressTitle = $derived(
    readout.progress?.phase === "readout"
      ? "Reading geometry"
      : readout.progress?.phase === "queued"
        ? "Waiting for the model"
        : "Preparing this token",
  );

  function affineOf(name: string): boolean | null {
    const info = probeRack.entries.get(name)?.info;
    if (info?.family === "geometry") return info.is_affine;
    return null;
  }

  /** Axis label: the positive pole for a rank-1 two-node concept axis
   *  (coords axis 0 is pole-normalized, +1 at node 0), ``c<i>`` otherwise. */
  function axisLabel(name: string, axis: number, rank: number): string {
    const info = probeRack.entries.get(name)?.info;
    const labels = info?.family === "geometry" ? info.node_labels : undefined;
    if (rank === 1 && axis === 0 && labels && labels.length === 2) {
      return labels[0];
    }
    return `c${axis}`;
  }

  /** Per-layer strip source, mirroring the rack's primary-per-layer rule:
   *  axis-0 ``coords_per_layer`` for a flat probe, ``fraction_per_layer``
   *  for a curved one (no single signed coordinate to strip). */
  function stripCells(
    name: string,
    reading: ProbeReadingJSON,
  ): { layer: number; value: number | null; title: string }[] {
    const curved = affineOf(name) === false;
    const source: Record<string, number | null> = {};
    if (curved) {
      Object.assign(source, reading.fraction_per_layer ?? {});
    } else {
      for (const [layer, c] of Object.entries(reading.coords_per_layer ?? {})) {
        source[layer] = c[0] ?? null;
      }
    }
    return Object.keys(source)
      .sort((a, b) => Number(a) - Number(b))
      .map((layer) => {
        const v = source[layer];
        const sign = v !== null && v >= 0 ? "+" : "";
        return {
          layer: Number(layer),
          value: v,
          title: v === null ? `L${layer} · unavailable` : `L${layer} · ${sign}${v.toFixed(3)}`,
        };
      });
  }

  function stripScale(name: string, reading: ProbeReadingJSON): number {
    if (affineOf(name) === false) return 1; // fraction strip is [0, 1]
    return probeAxisScale(name, 0);
  }

  function geometryEvidence(reading: ProbeReadingJSON, affine: boolean | null) {
    return [
      ...(reading.nearest ?? []).map(([label, dist]) => ({
        label,
        value: `d=${dist.toFixed(2)}`,
        title: `${dist.toFixed(3)} typical label spacings away · smaller is closer`,
      })),
      ...(reading.assignment ?? []).map(([label, prob]) => ({
        label: `~${label}`,
        value: `${(prob * 100).toFixed(0)}%`,
        title: `geometric assignment · ${(prob * 100).toFixed(1)}% · relative fit among nodes, not semantic confidence`,
        soft: true,
      })),
      ...(affine === false || reading.residual !== 0
        ? [{
            label: "residual",
            value: reading.residual.toFixed(3),
            title: "off-surface distance divided by the in-subspace activation norm · smaller is closer to the surface",
          }]
        : []),
      ...(affine !== true && reading.membership != null
        ? [{
            label: "membership",
            value: reading.membership.toFixed(3),
            title: "fit inside the learned tube · 0 to 1 · not semantic confidence; without a learned tube this defaults to 1",
          }]
        : []),
    ];
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
      aria-label="Geometry readout progress"
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
{:else if readout.data && rows.length > 0}
  <InstrumentHeader
    origin={readout.origin}
    source={readout.source}
    steering={readout.data.steering}
    bind:steered
    {showToggle}
    accent="var(--pillar-subspace)"
  />
  <DetailSection
    title="PROBE READINGS"
    count={`${rows.length} attached`}
  >
    <details class="geometry-guide">
      <summary>How to read geometry</summary>
      <dl>
        <dt>Coordinates</dt>
        <dd>Position in the fitted domain. On a two-pole axis, 0 is neutral and +1 is the named pole. Values can extend beyond a pole.</dd>
        <dt>Subspace</dt>
        <dd>Share of the centered activation in this subspace, from 0 to 1. Larger subspaces can capture more; this is not confidence.</dd>
        <dt>Distance · d</dt>
        <dd>Distance to a node in typical label spacings. Smaller is closer; 1 means one typical spacing away.</dd>
        <dt>Assignment · ~</dt>
        <dd>Relative geometric fit among nodes, not the probability that a trait is true.</dd>
        <dt>Membership</dt>
        <dd>Fit inside a curved manifold’s learned tube. Flat fits always return 1, so that value is hidden.</dd>
      </dl>
    </details>
    <div class="geo-list">
      {#each rows as [name, reading] (name)}
        {@const rank = reading.coords.length}
        {@const cells = stripCells(name, reading)}
        {@const affine = affineOf(name)}
        {@const cardAccent = affine === false ? "--pillar-manifold" : "--pillar-subspace"}
        {@const evidence = geometryEvidence(reading, affine)}
        <RackCard accent={cardAccent} disabled={false}>
          {#snippet statline()}
            <DetailCardHeader
              primary={name}
              primaryTitle={name}
              secondary={affine === null ? "geometry" : affine ? "subspace" : "manifold"}
              secondaryAccent
              meta={reading.depth_com?.[0] != null
                ? `@${reading.depth_com[0].toFixed(2)} ±${(reading.depth_spread?.[0] ?? 0).toFixed(2)}`
                : null}
              metaTitle="Where this probe’s signal is concentrated across layers: 0 is the first layer, 1 is the last. The ± value shows how widely it is spread."
            >
              {#snippet lead()}
                <RackMarker shape={affine === false ? "diamond" : "circle"} filled />
              {/snippet}
            </DetailCardHeader>
          {/snippet}
          {#snippet body()}
            <p class="aggregate-label">Across fitted layers</p>
            <ProbeReadingRow ariaLabel={`Subspace fraction ${reading.fraction.toFixed(3)}`}>
              {#snippet left()}<span class="geo-axis-label" {...{ "aria-description": "Share of the centered activation in this subspace · 0 to 1 · not confidence" }}>subspace</span>{/snippet}
              {#snippet bar()}
                <Bar percentage value={reading.fraction} max={1} color="var(--fg)" />
              {/snippet}
              {#snippet middle()}
                {#if (reading.nearest ?? []).length > 0}
                  <span class="nearest">
                    {reading.nearest[0][0]}
                  </span>
                {/if}
              {/snippet}
              {#snippet right()}<span class="geo-value"><MorphText text={reading.fraction.toFixed(3)} /></span>{/snippet}
            </ProbeReadingRow>
            {#each reading.coords as coord, axis (axis)}
            <ProbeReadingRow ariaLabel={`${name} axis ${axis}`}>
              {#snippet left()}
                <span class="geo-axis-label" {...{ "aria-description": (`coordinate axis ${axis}`) }}>
                  {axisLabel(name, axis, rank)}
                </span>
              {/snippet}
              {#snippet bar()}
                <Bar
                  value={coord}
                  max={probeAxisScale(name, axis)}
                  bipolar
                />
              {/snippet}
              {#snippet middle()}
                {#if reading.depth_com && reading.depth_com[axis] != null}
                  <span
                    class="geo-depth"
                    {...{ "aria-description": (`depth center ±${(reading.depth_spread?.[axis] ?? 0).toFixed(2)} · 0 first, 1 last`) }}
                  >
                    @<MorphText text={reading.depth_com[axis].toFixed(2)} />
                  </span>
                {/if}
              {/snippet}
              {#snippet right()}
                <span class="geo-value"><MorphText text={coord.toFixed(3)} /></span>
              {/snippet}
            </ProbeReadingRow>
            {/each}
            {#if cells.length > 0}
              <LayerStrip
                {cells}
                scale={stripScale(name, reading)}
                positiveColor={affine === false ? "var(--pillar-manifold)" : undefined}
                ariaLabel={`${name} per-layer readings`}
              />
            {/if}
            <EvidenceChips items={evidence} ariaLabel={`Geometry evidence for ${name}`} />
            {#if affine}
              <GeometryLayerReadings {name} {reading} axisLabels={reading.coords.map((_, axis) => axisLabel(name, axis, rank))} />
            {/if}
          {/snippet}
        </RackCard>
      {/each}
    </div>
  </DetailSection>
{:else if !hasGeometryProbes}
  <EmptyState
    title="Add a probe to see concept readings"
    detail="Choose a fitted concept, or create and train one for this model. Add it as a probe to inspect its readings here."
  >
    <div class="setup-actions">
      <Button variant="solid" onclick={() => setupProbe()}>Add a probe</Button>
      {#if authoring.available}
        <Button onclick={() => setupProbe(true)}>Create a concept</Button>
      {/if}
    </div>
    {#if !authoring.available}<p class="setup-note">Concept training isn’t available in this session. You can still add a fitted concept as a probe.</p>{/if}
  </EmptyState>
{:else if !hasReplayContext}
  <EmptyState
    title="no raw decode record"
    detail="replay needs a loom node generated with raw-decode capture in this session"
  />
{:else}
  <EmptyState title="no readings" />
{/if}

<style>
  .aggregate-label { margin: 0 0 var(--space-3); color: var(--fg-muted); font-size: var(--text-xs); }
  .setup-actions { display: flex; flex-wrap: wrap; gap: var(--space-3); }
  .setup-note { margin-top: var(--space-3); color: var(--fg-muted); font-size: var(--text-sm); }
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
    color: var(--pillar-subspace);
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
    background: var(--pillar-subspace);
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
  .geo-list {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3);
  }
  .geometry-guide {
    margin-bottom: var(--space-4);
    color: var(--fg-muted);
    font-size: var(--text-xs);
    line-height: 1.5;
  }
  .geometry-guide summary {
    color: var(--fg);
    cursor: pointer;
    padding-block: var(--space-2);
  }
  .geometry-guide dl { margin: var(--space-2) 0 0; }
  .geometry-guide dt { color: var(--fg); font-weight: var(--weight-medium); }
  .geometry-guide dd { margin: 0 0 var(--space-3); }
  @media (prefers-reduced-motion: reduce) {
    .progress-track.indeterminate span {
      animation: none;
      transform: none;
    }
  }
  .geo-axis-label {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .geo-depth {
    color: var(--fg-dim);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    font-variant-numeric: tabular-nums;
  }
  .geo-value {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: end;
  }
  .nearest {
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  @media (max-width: 820px) {
    .geo-list {
      grid-template-columns: 1fr;
    }
  }
</style>
