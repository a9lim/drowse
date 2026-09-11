<script lang="ts">
  import MorphText from "../../lib/ui/MorphText.svelte";
  import InfoTip from "../../lib/ui/InfoTip.svelte";
  import ModelProviderLogo from "./ModelProviderLogo.svelte";
  import { formatEtaRange } from "../../lib/runtime/eta";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import type { HostedModelOption, HostedShellController, HostedShellSnapshot } from "./types";

  let { controller, snapshot, disabled = false, onSelect }: {
    controller: HostedShellController;
    snapshot: HostedShellSnapshot;
    disabled?: boolean;
    onSelect: (model: HostedModelOption) => void;
  } = $props();

  let armedId = $state<string | null>(null);
  let pendingId = $state<string | null>(null);
  let pausing = $state(false);
  let failure = $state<{ id: string; message: string } | null>(null);
  const busy = $derived(pendingId !== null || ["requesting_persistence", "downloading", "cancelling"].includes(snapshot.download.phase));

  function unavailable(model: HostedModelOption): boolean {
    return disabled || busy || model.fit === "blocked" || (model.setupComplete
      ? !snapshot.runtime.available
      : !snapshot.download.available || model.catalogAvailable === false);
  }

  function downloadSize(model: HostedModelOption): string {
    return `${(model.remainingDownloadBytes / 1_000_000_000).toFixed(2)} GB`;
  }

  async function choose(model: HostedModelOption): Promise<void> {
    if (unavailable(model)) return;
    if (model.setupComplete) {
      armedId = null;
      onSelect(model);
      return;
    }
    if (armedId !== model.id) {
      armedId = model.id;
      return;
    }
    armedId = null;
    pendingId = model.id;
    failure = null;
    try {
      await controller.download(model.id, {
        explicitUnsafeOverride: model.fit === "uncertain" || model.requiresOomRetry === true,
      });
    } catch (cause) {
      failure = { id: model.id, message: userFacingError(cause, "Download failed. Try again.") };
    } finally {
      pendingId = null;
    }
  }

  async function pause(id: string): Promise<void> {
    if (pausing || snapshot.download.phase !== "downloading") return;
    pausing = true;
    try {
      await controller.cancelDownload();
    } catch (cause) {
      failure = { id, message: userFacingError(cause, "Could not pause the download. Try again.") };
    } finally {
      pausing = false;
    }
  }
</script>

<div class="model-choice-groups">
  {#each ["chat", "base"] as kind}
    {@const models = snapshot.models.filter(model => (model.modelType === "base" ? "base" : "chat") === kind)}
    {#if models.length}
      <section class="model-choice-group" aria-labelledby={`new-chat-${kind}-title`}>
        <div class="model-group-heading">
          <h3 id={`new-chat-${kind}-title`}>{kind === "base" ? "Base models" : "Chat models"}</h3>
          {#if kind === "base"}<InfoTip label="What are base models?" text="Base models predict what comes next in a piece of text. For conversation or instructions, choose a chat model." />{/if}
        </div>
        <div class="model-choices">
          {#each models as model (model.id)}
            {@const download = snapshot.download.modelVariantId === model.id ? snapshot.download : null}
            {@const active = !model.setupComplete && (pendingId === model.id || (download !== null && ["requesting_persistence", "downloading", "cancelling"].includes(download.phase)))}
            {@const progress = download?.progress}
            {@const determinate = progress !== undefined && progress.bytesTotal > 0}
            {@const percent = determinate ? Math.max(0, Math.min(100, Math.round(progress.bytesReceived / progress.bytesTotal * 100))) : 0}
            {@const failed = failure?.id === model.id ? failure.message : download?.phase === "failed" ? download.reason : null}
            {@const paused = download?.phase === "paused"}
            {@const armed = armedId === model.id && !model.setupComplete}
            {@const status = download?.phase === "cancelling" ? "Pausing…"
              : download?.phase === "requesting_persistence" || (pendingId === model.id && download?.phase !== "downloading") ? "Preparing download…"
              : progress?.offline ? "Offline. Waiting for a connection"
              : progress?.stalled ? "Waiting for data…"
              : determinate && progress.bytesReceived >= progress.bytesTotal ? "Verifying files…"
              : progress?.etaSeconds ? `${formatEtaRange(progress.etaSeconds)} remaining` : "Downloading…"}
            <div class="model-choice" class:installed={model.installed} class:armed class:active data-model-id={model.id}>
              <span class="model-download-size"><span class="sr-only">Total download: </span>{model.size}</span>
              <button type="button" class="model-choice-button" disabled={unavailable(model)}
                onclick={() => void choose(model)} onkeydown={(event) => { if (event.key === "Escape") armedId = null; }}
                aria-describedby={armed && (model.fit === "uncertain" || model.requiresOomRetry) ? `model-warning-${model.id}` : undefined}>
                <ModelProviderLogo modelId={model.modelId} size={28} />
                <span class="model-choice-copy">
                  <span class="model-name">{model.name}</span>
                  <span class="model-status" aria-live="polite"><MorphText text={model.fit === "blocked" ? "Not supported on this device"
                    : model.setupComplete ? "Ready to use"
                    : active ? "Download in progress"
                    : model.catalogAvailable === false || !snapshot.download.available ? "Download unavailable"
                    : armed ? `Click again to download - ${downloadSize(model)}`
                    : paused ? "Download paused. Click to resume"
                    : failed ? "Download failed. Click to retry"
                    : model.installed ? "Model downloaded. Download remaining tools"
                    : "Click to download"} /></span>
                </span>
              </button>
              {#if armed && (model.fit === "uncertain" || model.requiresOomRetry)}
                <p class="model-warning" id={`model-warning-${model.id}`} role="alert">{model.reason} The browser may run out of memory. Click again only if you want to continue.</p>
              {/if}
              {#if active || (paused && !model.setupComplete)}
                <div class="model-download">
                  <div class="progress-track" role="progressbar" aria-label={`Download ${model.name}`}
                    aria-valuemin="0" aria-valuemax="100" aria-valuenow={determinate ? percent : undefined}
                    aria-valuetext={paused ? "Download paused" : `${determinate ? `${percent}% · ` : ""}${status}`}>
                    <span class:indeterminate={!determinate} style:width={`${determinate ? percent : 18}%`}></span>
                  </div>
                  <div class="download-detail">
                    <span><MorphText text={paused ? "Downloaded files kept" : status} numbers={false} />{#if determinate} · <MorphText text={`${percent}%`} />{/if}</span>
                    {#if active}<button type="button" class="pause-download" disabled={pausing || download?.phase !== "downloading"} onclick={() => void pause(model.id)}>Pause</button>{/if}
                  </div>
                </div>
              {/if}
              {#if failed && !model.setupComplete}<p class="model-error" role="alert">{failed}</p>{/if}
            </div>
          {/each}
        </div>
      </section>
    {/if}
  {/each}
</div>

<style>
  .model-choice-groups { --model-card-padding: var(--space-8); display: grid; gap: calc(var(--space-7) * 2); container-type: inline-size; }
  .model-choice-group { min-width: 0; display: grid; gap: var(--space-7); }
  .model-group-heading { display: flex; align-items: center; gap: var(--space-xs); }
  h3 { margin: 0; font-size: var(--text-sm); color: var(--fg-muted); }
  .model-choices { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--space-7); align-items: start; }
  .model-choice { position: relative; min-width: 0; border-radius: var(--radius); background: var(--control-sheen), var(--glass-strong); box-shadow: var(--shadow-control); }
  .model-download-size { position: absolute; inset-block-start: var(--space-sm); inset-inline-end: var(--space-sm); z-index: 1; pointer-events: none; color: var(--fg-muted); font-size: var(--text-xs); line-height: 1.5; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .installed .model-download-size { color: var(--action-ink); }
  .model-choice.installed { --fg: var(--action-ink); color: var(--action-ink); background: var(--action-sheen), var(--action-bg); }
  .model-choice.armed { box-shadow: inset 0 0 0 2px var(--accent), var(--shadow-control); }
  button { font: inherit; color: inherit; border: 0; border-radius: var(--radius); background: transparent; cursor: pointer; }
  .model-choice-button { display: flex; align-items: center; gap: var(--space-7); width: 100%; min-height: 7.25rem; padding: var(--model-card-padding); padding-block-start: calc(var(--model-card-padding) + var(--space-xs)); text-align: start; }
  .model-choice-button:hover:not(:disabled) { background: var(--bg-hover); }
  .installed .model-choice-button:hover:not(:disabled) { background: var(--action-hover); }
  .model-choice-button:disabled { cursor: default; }
  .model-choice:not(.active) .model-choice-button:disabled { opacity: var(--disabled-opacity); }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
  .model-choice-copy { min-width: 0; display: grid; gap: var(--space-2); overflow-wrap: anywhere; }
  .model-name { font-weight: var(--weight-medium); color: var(--fg); }
  .model-status { color: var(--fg-muted); font-size: var(--text-xs); line-height: 1.5; }
  .installed .model-status { color: var(--action-ink); }
  .model-download { display: grid; gap: var(--space-3); padding: 0 var(--model-card-padding) var(--space-7); }
  .progress-track { height: 7px; overflow: hidden; border-radius: var(--radius-pill); background: var(--glass-strong); }
  .progress-track > span { display: block; height: 100%; background: var(--accent); }
  .progress-track .indeterminate { opacity: 0.6; }
  .download-detail { display: flex; align-items: center; justify-content: space-between; gap: var(--space-3); font-size: var(--text-xs); line-height: 1.5; }
  .download-detail > span { min-width: 0; overflow-wrap: anywhere; }
  .pause-download { min-height: var(--control-target); padding-inline: var(--space-4); }
  .pause-download:hover:not(:disabled) { background: var(--bg-hover); }
  .pause-download:disabled { opacity: var(--disabled-opacity); cursor: default; }
  .model-warning, .model-error { margin: 0; padding: 0 var(--model-card-padding) var(--model-card-padding); font-size: var(--text-xs); line-height: 1.5; overflow-wrap: anywhere; }
  .model-error { color: var(--accent-red); }
  .installed .model-error { color: var(--action-ink); }
  @container (max-width: 34rem) { .model-choices { grid-template-columns: minmax(0, 1fr); } }
  @media (forced-colors: active) {
    .model-choice { border: 1px solid ButtonText; }
    .model-choice.armed, .model-choice.installed { border: 2px solid Highlight; }
    .progress-track { border: 1px solid CanvasText; }
    .progress-track > span { background: Highlight; forced-color-adjust: none; }
  }
</style>
