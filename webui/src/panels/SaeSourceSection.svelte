<script lang="ts">
  import { onMount } from "svelte";
  import Bar from "../lib/charts/Bar.svelte";
  import Select from "../lib/Select.svelte";
  import Button from "../lib/ui/Button.svelte";
  import { apiInstruments } from "../lib/runtime/services";
  import { runtimeOperationAvailability } from "../lib/runtime/ui-capabilities";
  import { userFacingError } from "../lib/runtime/userFacingError";
  import {
    instrumentFamily,
    loadSae,
    refreshSaeSources,
    saeLoad,
    saeLoaded,
    saeSourceState,
  } from "../lib/stores.svelte";
  import { saeTrain } from "../lib/stores/instrumentAuthoring.svelte";
  import InstrumentSourceSection from "./rack/InstrumentSourceSection.svelte";

  const loaded = $derived(saeLoaded());
  const allowLocalTraining = runtimeOperationAvailability("sae_training").available;
  const allowPackFetch = $derived(
    instrumentFamily("sae")?.capabilities.preparations.includes("fetch") === true,
  );
  const allowSourceSwitch = $derived(
    instrumentFamily("sae")?.capabilities.source_switch === true,
  );
  const visibleSources = $derived(
    allowSourceSwitch
      ? saeSourceState.sources
      : saeSourceState.sources.filter((source) => source.active === true),
  );
  const residentSource = $derived(instrumentFamily("sae")?.source ?? null);
  const residentLayer = $derived.by((): number | null => {
    const active = saeSourceState.sources.find((row) => row.active);
    if (active?.layer != null) return active.layer;
    const live = instrumentFamily("sae")?.live;
    return live && "layer" in live ? live.layer : null;
  });
  let selectedSource = $state("");
  let selectedLayer = $state("");
  let layerSource = $state("");
  let localName = $state("my-sae");
  let trainTokens = $state(1_000_000);
  let trainLayer = $state("");
  let trainConfirm = $state(false);
  let releases = $state<{
    release: string;
    layers: number[];
    source?: "local" | "saelens";
  }[]>([]);
  let discoverError = $state<string | null>(null);
  const providerOptions = $derived((allowPackFetch ? releases : []).map((row) => ({
    value: `saelens:${row.release}`,
    label: row.release,
  })));
  const sourceBusy = $derived(
    saeSourceState.loading || saeSourceState.busy || saeLoad.state.running ||
      (allowLocalTraining && saeTrain.state.running),
  );
  const selectedPreparedSource = $derived(
    saeSourceState.sources.find((source) => source.source === selectedSource),
  );
  const selectedRelease = $derived(
    selectedSource.startsWith("saelens:")
      ? selectedSource.slice("saelens:".length)
      : selectedSource,
  );
  const availableLayers = $derived.by(() => {
    const registry = releases.find((row) => row.release === selectedRelease);
    const layers = registry?.layers ?? (
      selectedPreparedSource?.layer == null ? [] : [selectedPreparedSource.layer]
    );
    return [...new Set(layers)].sort((a, b) => a - b);
  });
  const layerOptions = $derived(availableLayers.map((layer) => ({
    value: String(layer),
    label: `layer ${layer}`,
  })));
  const sourceMatchesLoaded = $derived(
    loaded && residentSource !== null && selectedSource === residentSource,
  );
  const selectedLayerNumber = $derived(
    selectedLayer === "" ? null : Number(selectedLayer),
  );
  const sourceSelectionCurrent = $derived(
    sourceMatchesLoaded && selectedLayerNumber === residentLayer,
  );

  onMount(() => {
    void refreshSaeSources();
    if (allowLocalTraining) void saeTrain.check();
    if (allowPackFetch) void saeLoad.check();
  });

  $effect(() => {
    const source = selectedSource;
    const layers = availableLayers;
    if (source !== layerSource) {
      layerSource = source;
      const resident = sourceMatchesLoaded ? residentLayer : null;
      const cached = selectedPreparedSource?.layer ?? null;
      const preferred = resident != null && layers.includes(resident)
        ? resident
        : cached != null && layers.includes(cached)
        ? cached
        : preferredLayer(layers);
      selectedLayer = preferred == null ? "" : String(preferred);
    } else if (
      layers.length > 0 &&
      (selectedLayer === "" || !layers.includes(Number(selectedLayer)))
    ) {
      const preferred = preferredLayer(layers);
      selectedLayer = preferred == null ? "" : String(preferred);
    }
  });

  $effect(() => {
    if (selectedSource !== "local") trainConfirm = false;
  });

  $effect(() => {
    if (releases.length > 0) return;
    void apiInstruments.sources("sae").then((result) => {
      releases = (result.releases ?? []).filter((row) => row.source !== "local");
      if (!selectedSource && releases.length > 0) {
        selectedSource = `saelens:${releases[0].release}`;
      }
    }).catch((error) => {
      discoverError = userFacingError(
        error,
        "Unable to check available feature packs. Check your connection and try again.",
      );
    });
  });

  function requestTrain(): void {
    if (!allowLocalTraining) return;
    if (!localName.trim() || saeTrain.state.running) return;
    if (!trainConfirm) {
      trainConfirm = true;
      return;
    }
    trainConfirm = false;
    const parsedLayer = trainLayer.trim() === "" ? null : Number(trainLayer);
    void saeTrain.start({
      name: localName.trim(),
      tokens: trainTokens,
      layer: parsedLayer != null && Number.isInteger(parsedLayer) ? parsedLayer : null,
    });
  }

  function preferredLayer(layers: number[]): number | null {
    if (layers.length === 0) return null;
    const depth = Math.max(...layers, 1);
    const band = layers.filter((layer) => {
      const fraction = layer / depth;
      return fraction >= 0.4 && fraction <= 0.9;
    });
    const pool = band.length > 0 ? band : layers;
    const target = 0.65 * depth;
    return [...pool].sort((a, b) =>
      Math.abs(a - target) - Math.abs(b - target) || a - b
    )[0] ?? null;
  }

  function loadSelectedSae(source: string): void {
    void loadSae(source, selectedLayerNumber);
  }
</script>

<InstrumentSourceSection
  ready={loaded}
  sources={visibleSources}
  bind:value={selectedSource}
  busy={sourceBusy}
  sourceError={saeSourceState.error}
  working={allowLocalTraining && saeTrain.state.running}
  allowLocal={allowLocalTraining}
  selectionCurrent={sourceSelectionCurrent}
  onuse={loadSelectedSae}
  {providerOptions}
  providerPlaceholder="SAELens release"
  onfetch={loadSelectedSae}
  unavailableMessage="This model has no compatible SAE pack installed. Add one in Model settings if a pack is available."
  localSectionLabel="Create on this device"
  localActionLabel={trainConfirm ? "confirm train" : "train"}
  localActionDisabled={!localName.trim() || sourceBusy}
  onlocal={requestTrain}
>
  {#snippet sourceControls()}
    <Select
      bind:value={selectedLayer}
      options={layerOptions}
      placeholder="layer"
      disabled={sourceBusy || layerOptions.length === 0}
      ariaLabel="SAE measurement layer"
    />
  {/snippet}
  {#snippet localControls()}
    <label class="setup-field setup-field-wide">
      <span class="setup-field-label">name</span>
      <input class="add-input" bind:value={localName} placeholder="name" aria-label="Local SAE name" />
    </label>
    <label class="setup-field setup-field-medium">
      <span class="setup-field-label">tokens</span>
      <input
        class="add-input"
        type="number"
        min="1"
        step="10000"
        bind:value={trainTokens}
        aria-label="SAE training tokens"
        title="tokens"
      />
    </label>
    <label class="setup-field setup-field-narrow">
      <span class="setup-field-label">layer</span>
      <input
        class="add-input"
        inputmode="numeric"
        bind:value={trainLayer}
        placeholder="auto"
        aria-label="Residual layer (blank for automatic)"
      />
    </label>
  {/snippet}
  {#snippet progress()}
    <div class="train-progress loading-pulse loading-placeholder" role="status" aria-live="polite">
      <div class="train-line">
        <span class="work-status">{saeTrain.state.message ?? "training…"}</span>
        <span class="train-count">
          {saeTrain.state.current.toLocaleString()}/{saeTrain.state.total.toLocaleString()}
        </span>
      </div>
      <Bar
        value={saeTrain.state.current}
        max={Math.max(saeTrain.state.total, 1)}
        width={160}
        height={8}
        color="var(--pillar-sae)"
      />
      <Button
        size="sm"
        variant="danger"
        disabled={saeTrain.state.cancelling}
        onclick={() => void saeTrain.cancel()}
      >
        {saeTrain.state.cancelling ? "cancelling…" : "cancel"}
      </Button>
    </div>
  {/snippet}
  {#snippet warning()}
    {#if trainConfirm}
      <p class="hint train-warning" role="alert">Blocks generation; uses FineWeb-Edu. Confirm again.</p>
    {/if}
  {/snippet}
  {#snippet messages()}
    {#if saeLoad.state.running && saeLoad.state.message}
      <p class="hint" role="status" aria-live="polite">{saeLoad.state.message}</p>
    {/if}
    {#if saeLoad.state.error}
      <p class="hint load-error" role="alert">{saeLoad.state.error}</p>
    {/if}
    {#if saeTrain.state.error}
      <p class="hint load-error" role="alert">local train: {saeTrain.state.error}</p>
    {/if}
    {#if discoverError}
      <p class="hint" role="alert">registry: {discoverError}</p>
    {/if}
  {/snippet}
</InstrumentSourceSection>

<style>
  .train-line,
  .train-progress {
    display: flex;
    gap: var(--space-2);
  }
  .train-progress {
    flex-direction: column;
    align-items: flex-start;
  }
  .train-line {
    width: 100%;
    min-width: 0;
    align-items: center;
    justify-content: space-between;
  }
  .work-status,
  .train-count {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
  }
  .work-status {
    overflow: hidden;
    margin: 0;
    color: var(--fg-dim);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .train-count {
    flex: 0 0 auto;
    color: var(--fg-muted);
    font-variant-numeric: tabular-nums;
  }
  .hint { margin: 0; color: var(--fg-muted); font-size: var(--text-sm); }
  .train-warning { color: var(--accent-yellow); }
  .load-error { color: var(--accent-red); }
</style>
