<script lang="ts">
  import { onMount } from "svelte";
  import {
    instrumentFamily,
    lensSourceState,
    refreshLensSources,
    sessionState,
    useLensSource,
  } from "../../lib/stores.svelte";
  import InstrumentSourceSection from "../../panels/rack/InstrumentSourceSection.svelte";

  const fitted = $derived(sessionState.info?.jlens_fitted === true);
  const allowSourceSwitch = $derived(
    instrumentFamily("lens")?.capabilities.source_switch === true,
  );
  const visibleSources = $derived(
    allowSourceSwitch
      ? lensSourceState.sources
      : lensSourceState.sources.filter((source) => source.active === true),
  );
  const sourceBusy = $derived(lensSourceState.loading || lensSourceState.busy);
  let selectedSource = $state("");

  onMount(() => void refreshLensSources());
</script>

<InstrumentSourceSection
  ready={fitted}
  sources={visibleSources}
  bind:value={selectedSource}
  busy={sourceBusy}
  sourceError={lensSourceState.error}
  working={false}
  allowLocal={false}
  onuse={(source) => void useLensSource(source)}
  providerOptions={[]}
  providerPlaceholder="compatible J-lens pack"
  onfetch={() => undefined}
  unavailableMessage={sessionState.info?.is_base_model
    ? "Word insights are optional for base models. This view needs a compatible J-lens pack; text completion works without it."
    : "Word insights are missing. Switch models, then download this one again."}
  localActionLabel=""
  localActionDisabled
  onlocal={() => undefined}
/>
