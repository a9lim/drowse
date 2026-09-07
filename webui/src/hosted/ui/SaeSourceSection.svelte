<script lang="ts">
  import { onMount } from "svelte";
  import {
    instrumentFamily,
    loadSae,
    refreshSaeSources,
    saeLoaded,
    saeSourceState,
  } from "../../lib/stores.svelte";
  import InstrumentSourceSection from "../../panels/rack/InstrumentSourceSection.svelte";
  import {
    currentCatalogInstrumentAvailability,
    type CatalogInstrumentAvailability,
  } from "../../lib/runtime/instrumentAvailability";

  const loaded = $derived(saeLoaded());
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
  let availability = $state<CatalogInstrumentAvailability>("unknown");
  let selectedLayer = $state("");
  let layerSource = $state("");
  const sourceBusy = $derived(saeSourceState.loading || saeSourceState.busy);
  const selectedPreparedSource = $derived(
    saeSourceState.sources.find((source) => source.source === selectedSource),
  );
  const availableLayers = $derived.by(() => {
    const layers = selectedPreparedSource?.layer == null
      ? []
      : [selectedPreparedSource.layer];
    return [...new Set(layers)].sort((a, b) => a - b);
  });
  const layerOptions = $derived(availableLayers.map((layer) => ({
    value: String(layer),
    label: `layer ${layer}`,
  })));
  const sourceMatchesLoaded = $derived(
    loaded && residentSource !== null && selectedSource === residentSource,
  );
  const selectedLayerNumber = $derived(selectedLayer === "" ? null : Number(selectedLayer));
  const sourceSelectionCurrent = $derived(
    sourceMatchesLoaded && selectedLayerNumber === residentLayer,
  );

  onMount(() => {
    void refreshSaeSources();
    let mounted = true;
    void currentCatalogInstrumentAvailability("sae").then((value) => {
      if (mounted) availability = value;
    });
    return () => {
      mounted = false;
    };
  });

  const unavailableMessage = $derived(
    availability === "unavailable"
      ? "No SAE is available for this model"
      : availability === "context-unavailable"
        ? "No SAE is available for this conversation length"
        : availability === "available"
          ? "No SAE is installed. Add the available SAE in Model settings, then reopen the model."
          : "No SAE is loaded. Check Model settings for a compatible SAE.",
  );

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

  function preferredLayer(layers: number[]): number | null {
    if (layers.length === 0) return null;
    const depth = Math.max(...layers, 1);
    const pool = layers.filter((layer) => {
      const fraction = layer / depth;
      return fraction >= 0.4 && fraction <= 0.9;
    });
    const candidates = pool.length > 0 ? pool : layers;
    const target = 0.65 * depth;
    return [...candidates].sort((a, b) =>
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
  working={false}
  allowLocal={false}
  selectionCurrent={sourceSelectionCurrent}
  onuse={loadSelectedSae}
  providerOptions={[]}
  providerPlaceholder="compatible SAE pack"
  onfetch={() => undefined}
  {unavailableMessage}
  localActionLabel=""
  localActionDisabled
  onlocal={() => undefined}
/>
