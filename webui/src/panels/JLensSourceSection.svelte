<script lang="ts">
  import { onMount } from "svelte";
  import Bar from "../lib/charts/Bar.svelte";
  import Button from "../lib/ui/Button.svelte";
  import { runtimeOperationAvailability } from "../lib/runtime/ui-capabilities";
  import {
    instrumentFamily,
    lensFetch,
    lensSourceState,
    refreshLensSources,
    sessionState,
    useLensSource,
  } from "../lib/stores.svelte";
  import { lensFit } from "../lib/stores/instrumentAuthoring.svelte";
  import InstrumentSourceSection from "./rack/InstrumentSourceSection.svelte";

  const fitted = $derived(sessionState.info?.jlens_fitted === true);
  const allowLocalFitting = runtimeOperationAvailability("jlens_fitting").available;
  const allowPackFetch = $derived(
    instrumentFamily("lens")?.capabilities.preparations.includes("fetch") === true,
  );
  const allowSourceSwitch = $derived(
    instrumentFamily("lens")?.capabilities.source_switch === true,
  );
  const visibleSources = $derived(
    allowSourceSwitch
      ? lensSourceState.sources
      : lensSourceState.sources.filter((source) => source.active === true),
  );
  const sourceBusy = $derived(
    lensSourceState.loading || lensSourceState.busy ||
      lensFetch.state.running || (allowLocalFitting && lensFit.state.running),
  );
  const providerOptions = $derived(allowPackFetch ? [
    { value: "workspace-r", label: "workspace-r (RelP)" },
    { value: "neuronpedia", label: "neuronpedia" },
    { value: "workspace-j", label: "workspace-j" },
  ] : []);
  let fitPrompts = $state(100);
  let fitLayers = $state("all");
  let fitRelp = $state(true);
  let fitConfirm = $state(false);
  let selectedSource = $state("");
  const fitReady = $derived(
    Number.isInteger(fitPrompts) && fitPrompts >= 1 && fitPrompts <= 5000 &&
      fitLayers.trim().length > 0,
  );
  const fitIsPreparing = $derived(
    (lensFit.state.message ?? "").startsWith("streaming "),
  );

  function requestFit(): void {
    if (!allowLocalFitting) return;
    if (!fitConfirm) {
      fitConfirm = true;
      return;
    }
    fitConfirm = false;
    void lensFit.start({
      prompts: fitPrompts,
      layers: fitLayers.trim(),
      relp: fitRelp,
    });
  }

  onMount(() => {
    if (allowLocalFitting) void lensFit.check();
    if (allowPackFetch) void lensFetch.check();
    void refreshLensSources();
  });

  $effect(() => {
    if (selectedSource !== "local") fitConfirm = false;
  });
</script>

<InstrumentSourceSection
  ready={fitted}
  sources={visibleSources}
  bind:value={selectedSource}
  busy={sourceBusy}
  sourceError={lensSourceState.error}
  working={lensFetch.state.running || (allowLocalFitting && lensFit.state.running)}
  allowLocal={allowLocalFitting}
  onuse={(source) => void useLensSource(source)}
  {providerOptions}
  providerPlaceholder="lens provider"
  onfetch={(source) => void lensFetch.start({ source })}
  unavailableMessage="Word insights are missing. Switch models, then download this one again."
  localSectionLabel="Create on this device"
  localActionLabel={fitConfirm ? "confirm fit" : "fit"}
  localActionDisabled={sourceBusy || !fitReady}
  onlocal={requestFit}
>
  {#snippet localControls()}
    <label class="setup-field setup-field-medium">
      <span class="setup-field-label">prompts</span>
      <input
        class="add-input"
        type="number"
        min="1"
        max="5000"
        step="25"
        bind:value={fitPrompts}
        placeholder="100"
        aria-label="J-lens corpus prompts"
        title="1–5000"
      />
    </label>
    <label class="setup-field setup-field-wide">
      <span class="setup-field-label">layers</span>
      <input
        class="add-input"
        bind:value={fitLayers}
        placeholder="workspace | all | 13,14,…"
        aria-label="J-lens source layers"
        title="workspace | all | layer ids"
      />
    </label>
    <label class="setup-field setup-field-medium">
      <span class="setup-field-label">estimator</span>
      <button
        type="button"
        class="add-input relp-toggle"
        class:relp-on={fitRelp}
        onclick={() => (fitRelp = !fitRelp)}
        aria-pressed={fitRelp}
        title="RelP (default) · saves as local:relp"
      >{fitRelp ? "relp (R-lens)" : "standard"}</button>
    </label>
  {/snippet}
  {#snippet progress()}
    {#if lensFetch.state.running}
      <p class="work-status loading-pulse loading-placeholder" role="status" aria-live="polite">
        {lensFetch.state.message ?? "fetching official lens…"}
      </p>
    {:else}
      <div class="fit-progress loading-pulse loading-placeholder" role="status" aria-live="polite" aria-label="Lens fit progress">
        <div class="fit-line">
          <span class="fit-msg">{lensFit.state.message ?? "fitting…"}</span>
          {#if lensFit.state.total > 0}
            <span class="fit-count">{lensFit.state.current}/{lensFit.state.total}</span>
          {/if}
        </div>
        <div
          class="fit-bar"
          role="progressbar"
          aria-label="J-lens prompts fitted"
          aria-valuemin="0"
          aria-valuemax={Math.max(lensFit.state.total, 1)}
          aria-valuenow={lensFit.state.current}
        >
          <Bar
            value={lensFit.state.current}
            max={Math.max(lensFit.state.total, 1)}
            width={160}
            height={8}
            color="var(--pillar-lens)"
          />
        </div>
        <p class="hint">
          {#if lensFit.state.cancelling}
            stopping background work…
          {:else if fitIsPreparing}
            generation available during corpus setup
          {:else}
            generation paused during model fitting
          {/if}
        </p>
        <Button
          size="sm"
          variant="danger"
          disabled={lensFit.state.cancelling}
          onclick={() => void lensFit.cancel()}
        >
          {lensFit.state.cancelling ? "cancelling…" : "cancel"}
        </Button>
      </div>
    {/if}
  {/snippet}
  {#snippet warning()}
    {#if fitConfirm && !lensFit.state.running}
      <p class="hint fit-warning" role="alert">Blocks generation; may take hours. Confirm again.</p>
    {/if}
  {/snippet}
  {#snippet messages()}
    {#if lensFit.state.error}
      <p class="hint fit-error" role="alert">local fit: {lensFit.state.error}</p>
    {/if}
    {#if lensFetch.state.error}
      <p class="hint fit-error" role="alert">official fetch: {lensFetch.state.error}</p>
    {/if}
  {/snippet}
</InstrumentSourceSection>

<style>
  .work-status,
  .hint {
    margin: 0;
    color: var(--fg);
    font-size: var(--text-sm);
  }
  .work-status,
  .fit-msg {
    font-family: var(--font-mono);
  }
  .fit-error { color: var(--accent-red); }
  .fit-warning { color: var(--accent-yellow); }
  .fit-progress {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
  .fit-line {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-3);
  }
  .fit-msg {
    overflow: hidden;
    min-width: 0;
    color: var(--fg);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .fit-count {
    flex: 0 0 auto;
    color: var(--fg-muted);
    font-variant-numeric: tabular-nums;
  }
  .fit-bar :global(.bar) {
    display: block;
    width: 100%;
    height: var(--data-bar-height);
  }
  .relp-toggle {
    cursor: pointer;
    text-align: start;
  }
  .relp-on {
    color: var(--pillar-lens);
    border-color: var(--pillar-lens);
  }
</style>
