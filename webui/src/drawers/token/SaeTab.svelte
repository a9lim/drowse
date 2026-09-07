<script lang="ts">
  import Select from "../../lib/Select.svelte";
  import RollingNumber from "../../lib/ui/RollingNumber.svelte";
  // Feature descriptions and reference maxima belong to the readout's SAE.
  // Relative activation may exceed 1; raw bars share this readout's scale.

  import Bar from "../../lib/charts/Bar.svelte";
  import { onMount, untrack } from "svelte";
  import { refreshSaeSources, saeSourceState } from "../../lib/stores.svelte";
  import { loadSaeDescription, type SaeDescription, type SaeDescriptionSource } from "../../lib/saeDescriptions";
  import RackCard from "../../panels/rack/RackCard.svelte";
  import type {
    ScalarReadingJSON,
    SaeFeatureJSON,
    SaeTokenReadoutJSON,
  } from "../../lib/types";
  import type { ReplayReadout } from "./readout.svelte";
  import type { CatalogInstrumentAvailability } from "../../lib/runtime/instrumentAvailability";
  import EmptyState from "./EmptyState.svelte";
  import PinnedReadings from "./PinnedReadings.svelte";
  import InstrumentHeader from "./InstrumentHeader.svelte";
  import DetailSection from "./DetailSection.svelte";
  import DetailCardHeader from "./DetailCardHeader.svelte";
  import EvidenceChips from "./EvidenceChips.svelte";

  let {
    readout,
    steered = $bindable(),
    saeLoaded,
    availability,
    hasReplayContext,
    pinned,
    replayAvailable,
  }: {
    readout: ReplayReadout<SaeTokenReadoutJSON>;
    steered: boolean;
    saeLoaded: boolean;
    availability: CatalogInstrumentAvailability;
    hasReplayContext: boolean;
    replayAvailable: boolean;
    /** Live-captured pinned-probe readings from the token's envelope. */
    pinned: Record<string, ScalarReadingJSON> | null;
  } = $props();

  onMount(() => { void refreshSaeSources(); });
  const sourceInfo = $derived(saeSourceState.sources.find(source =>
    source.source === readout.source && source.layer === readout.data?.layer));
  const descriptionSource = $derived(sourceInfo?.description_source as SaeDescriptionSource | null | undefined);
  const descriptionIdentity = $derived(JSON.stringify(descriptionSource ?? null));
  let query = $state("");
  let sort = $state("activation");
  let descriptions = $state<Record<number, SaeDescription>>({});
  let descriptionsLoading = $state(false);
  let descriptionsError = $state("");
  let metadataController: AbortController | null = null;
  const missingDescriptions = $derived((readout.data?.features ?? []).some(feature =>
    !feature.label?.trim() && !descriptions[feature.id]));
  const featureRanks = $derived(new Map([...(readout.data?.features ?? [])]
    .sort((a, b) => b.activation - a.activation)
    .map((feature, index) => [feature.id, index + 1])));
  $effect(() => {
    const identity = `${readout.data?.node_id}/${readout.data?.raw_index}/${readout.source}/${readout.data?.layer}`;
    void identity;
    void descriptionIdentity;
    void readout.data?.features.map(feature => feature.id).join(",");
    descriptions = {};
    descriptionsError = "";
    descriptionsLoading = false;
    query = "";
    untrack(() => { void loadDescriptions(); });
    return () => { metadataController?.abort(); metadataController = null; };
  });
  const visibleFeatures = $derived.by(() => {
    const text = query.trim().toLowerCase();
    return [...(readout.data?.features ?? [])]
      .filter(feature => !text || `sae/${feature.id} ${feature.label ?? descriptions[feature.id]?.label ?? ""}`.toLowerCase().includes(text))
      .sort((a, b) => sort === "id" ? a.id - b.id : b.activation - a.activation);
  });

  async function loadDescriptions(): Promise<void> {
    const binding = descriptionSource;
    if (!binding || !readout.data || descriptionsLoading) return;
    const controller = new AbortController();
    metadataController = controller;
    descriptionsLoading = true;
    descriptionsError = "";
    const timeout = setTimeout(() => controller.abort(), 20_000);
    const features = readout.data.features.filter(feature => !feature.label?.trim() && !descriptions[feature.id]);
    try {
      for (let start = 0; start < features.length; start += 4) {
        const results = await Promise.allSettled(features.slice(start, start + 4).map(async feature =>
          [feature.id, await loadSaeDescription(binding, feature.id, controller.signal)] as const));
        if (metadataController !== controller) return;
        for (const result of results) {
          if (result.status === "fulfilled") {
            descriptions = { ...descriptions, [result.value[0]]: result.value[1] };
          } else {
            descriptionsError = "Some descriptions could not be loaded. Check your connection and try again.";
          }
        }
        if (controller.signal.aborted) break;
      }
    } catch {
      if (metadataController === controller) descriptionsError = "Some descriptions could not be loaded. Check your connection and try again.";
    } finally {
      clearTimeout(timeout);
      if (metadataController === controller) descriptionsLoading = false;
    }
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
      ? "Reading model features"
      : readout.progress?.phase === "queued"
        ? "Waiting for the model"
        : "Preparing this token",
  );

  /** Shared raw unit for metadata-less features — the largest raw
   *  activation in THIS readout (historical view, so the panel-wide live
   *  scale doesn't apply). */
  const rawScale = $derived(
    Math.max(
      ...(readout.data?.features ?? [])
        .filter((f) => !(f.max_act != null && f.max_act > 0))
        .map((f) => f.activation),
      1,
    ),
  );

  function strengthOf(f: SaeFeatureJSON): number | null {
    return f.max_act != null && f.max_act > 0 ? f.activation / f.max_act : null;
  }

  function featureEvidence(f: SaeFeatureJSON) {
    if (!(f.max_act != null && f.max_act > 0)) return [];
    return [
      {
        label: "activation",
        value: f.activation.toFixed(3),
        title: `raw activation · ${f.activation.toFixed(3)}`,
      },
      {
        label: "reference maximum",
        value: f.max_act.toFixed(3),
        title: `Reference maximum supplied with this feature: ${f.max_act.toFixed(3)}`,
      },
    ];
  }

</script>

{#if readout.loading}
  <div class="readout-progress loading-pulse" role="status" aria-live="polite">
    <div class="progress-heading">
      <span>{progressTitle}</span>
      <code>{#if determinateProgress}<RollingNumber value={progressPercent} />%{:else}starting{/if}</code>
    </div>
    <div
      class="progress-track"
      class:indeterminate={!determinateProgress}
      role="progressbar"
      aria-label="Model-feature readout progress"
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
    layer={readout.data.layer}
    steering={readout.data.steering}
    bind:steered
    {showToggle}
    accent="var(--pillar-sae)"
  />
  <div class="description-context">
    <p>Feature descriptions are published interpretations, not definitive meanings. Activations are not probabilities.</p>
    {#if descriptionSource}
      <button class="description-action" disabled={descriptionsLoading || !missingDescriptions} aria-busy={descriptionsLoading} onclick={loadDescriptions}>
        {descriptionsLoading ? "Loading descriptions…" : descriptionsError ? "Retry descriptions" : !missingDescriptions ? "Descriptions checked" : "Load published descriptions"}
      </button>
      <p class="metadata-privacy">Automatically looks up this SAE’s feature IDs on Neuronpedia. Your conversation and activation values are not sent.</p>
    {:else}
      <p class="metadata-privacy">Only labels included with this exact SAE pack are shown. An exact published dictionary match is required for additional descriptions.</p>
    {/if}
    {#if descriptionsLoading}<p role="status">Checking the published feature records…</p>{/if}
    {#if descriptionsError}<p class="metadata-error" role="alert">{descriptionsError}</p>{/if}
  </div>
  {#if readout.origin === "captured" && pinned && Object.keys(pinned).length > 0}
    <PinnedReadings readings={pinned} accent="--pillar-sae" shape="triangle" />
  {/if}
  {#if readout.data.features.length === 0}
    <EmptyState title="no features fired at this position" />
  {:else}
    <div class="feature-tools">
      <label>Find a feature<input type="search" placeholder="Description or feature ID" bind:value={query} /></label>
      <label>Sort by<Select bind:value={sort} ariaLabel="Sort by" options={[{ value: "activation", label: "Activation" }, { value: "id", label: "Feature ID" }]} /></label>
    </div>
    <DetailSection
      title="SAE activations"
      count={`${visibleFeatures.length} of ${readout.data.features.length} recorded`}
      accent="var(--pillar-sae)"
    >
      <div class="sae-list" role="list" aria-label="Top SAE features">
        {#each visibleFeatures as feature (feature.id)}
          {@const strength = strengthOf(feature)}
          {@const evidence = featureEvidence(feature)}
          {@const published = descriptions[feature.id]}
          {@const label = feature.label?.trim() || published?.label}
          <div role="listitem" aria-label={`SAE feature ${feature.id}, layer ${readout.data.layer}`}>
            <RackCard accent="--pillar-sae" disabled={false}>
              {#snippet statline()}
                <DetailCardHeader
                  primary={`sae/${feature.id}`}
                  tail={`Layer ${readout.data?.layer ?? "-"}`}
                  tailTitle="SAE layer"
                >
                  {#snippet lead()}<span>#{featureRanks.get(feature.id)}</span>{/snippet}
                </DetailCardHeader>
              {/snippet}
              {#snippet body()}
                <p class="feature-description" class:missing={!label}>
                  {label || (published ? "No description published for this feature" : descriptionSource ? descriptionsLoading ? "Loading description…" : "Description could not be loaded" : "No description included in this pack")}
                </p>
                {#if published}
                  <a class="description-source" href={published.url} target="_blank" rel="noreferrer">Neuronpedia{published.explanationModel ? ` · ${published.explanationModel}` : " · feature record"}</a>
                {/if}
                <div class="feature-reading" role="group" aria-label={`Activation for sae/${feature.id}`}>
                  <span class="row-label">{strength != null ? "Relative activation" : "Raw activation"}</span>
                  <span class="sae-value">
                    {strength != null ? strength.toFixed(3) : feature.activation.toFixed(2)}
                  </span>
                  <div class="feature-bar">
                    {#if strength != null}
                      <Bar
                        value={Math.max(strength, 0)}
                        max={1}
                        color="var(--pillar-sae)"
                      />
                    {:else}
                      <Bar
                        value={Math.max(feature.activation, 0)}
                        max={rawScale}
                        color="color-mix(in srgb, var(--pillar-sae) 55%, transparent)"
                      />
                    {/if}
                  </div>
                </div>
                <EvidenceChips
                  items={evidence}
                  ariaLabel={`Metadata for feature sae/${feature.id}`}
                />
              {/snippet}
            </RackCard>
          </div>
        {/each}
      </div>
      {#if visibleFeatures.length === 0}<p class="no-matches">No recorded features match “{query}”. <button onclick={() => { query = ""; }}>Clear search</button></p>{/if}
    </DetailSection>
  {/if}
{:else if !saeLoaded}
  {#if availability === "unavailable"}
    <EmptyState title="No SAE is available for this model" />
  {:else if availability === "context-unavailable"}
    <EmptyState title="No SAE is available for this conversation length" />
  {:else if availability === "available"}
    <EmptyState
      title="No SAE is installed"
      detail="Add the available SAE in Model settings, then reopen the model."
    />
  {:else}
    <EmptyState
      title="No SAE is loaded"
      detail="Check Model settings for a compatible SAE."
    />
  {/if}
{:else if !hasReplayContext}
  <EmptyState
    title="no raw decode record"
    detail="replay needs a loom node generated with raw-decode capture in this session"
  />
{:else}
  <EmptyState title="no readout" />
{/if}

<style>
  .description-context { margin-block: var(--space-4) var(--space-5); }
  .description-context p { margin: var(--space-2) 0; max-width: 76ch; color: var(--fg-muted); font-size: var(--text-xs); line-height: 1.6; }
  .description-context .metadata-privacy { color: var(--fg-subtle); font-size: var(--text-2xs); }
  .description-context .metadata-error { color: var(--accent-red); }
  .description-action, .no-matches button { min-height: var(--control-target); padding-inline: var(--space-4); border: 0; border-radius: var(--radius-sm); background: var(--glass); color: var(--pillar-sae); cursor: pointer; }
  .description-action:disabled { opacity: 0.6; cursor: wait; }
  .feature-tools { display: flex; gap: var(--space-4); flex-wrap: wrap; margin-block: var(--space-5); }
  .feature-tools label { display: grid; gap: var(--space-2); color: var(--fg-muted); font-size: var(--text-xs); }
  .feature-tools label:first-child { flex: 1 1 200px; }
  .feature-tools input { min-height: var(--control-target); min-width: 0; width: 100%; padding-inline: var(--space-4); border: 0; border-radius: var(--radius-sm); background: var(--input-well); color: var(--fg); font: inherit; }
  .description-source { display: inline-block; margin-block: calc(-1 * var(--space-2)) var(--space-3); color: var(--fg-muted); font-size: var(--text-2xs); overflow-wrap: anywhere; }
  .no-matches { color: var(--fg-muted); font-size: var(--text-sm); }
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
    color: var(--pillar-sae);
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
    background: var(--pillar-sae);
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
  .sae-list {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3);
  }
  .feature-description {
    margin: 0 0 var(--space-3);
    color: var(--fg-strong);
    font-size: var(--text-sm);
    line-height: 1.5;
    overflow-wrap: anywhere;
  }
  .feature-description.missing {
    color: var(--fg-muted);
  }
  .feature-reading {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: baseline;
    gap: var(--space-2) var(--space-3);
    min-width: 0;
  }
  .feature-bar {
    grid-column: 1 / -1;
    min-width: 0;
    padding-block: var(--space-1);
  }
  .feature-bar :global(.bar) {
    display: block;
    width: 100%;
    height: var(--data-bar-height);
  }
  .row-label {
    color: var(--fg-muted);
    font-size: var(--text-xs);
    line-height: 1.5;
  }
  .sae-value {
    color: var(--pillar-sae);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-variant-numeric: tabular-nums;
    text-align: end;
    white-space: nowrap;
  }
  @media (max-width: 760px) {
    .sae-list {
      grid-template-columns: 1fr;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .progress-track.indeterminate span {
      animation: none;
      transform: none;
    }
  }
</style>
