<script lang="ts">
  import MorphText from "../../lib/ui/MorphText.svelte";
  import FluentIcon from "../../lib/ui/FluentIcon.svelte";
  import StateIcon from "../../lib/ui/StateIcon.svelte";
  import BaseModelTag from "../../lib/ui/BaseModelTag.svelte";
  import { animatedDetails } from "../../lib/animatedDetails";
  import RollingNumber from "../../lib/ui/RollingNumber.svelte";
  import TabIdentity from "../../lib/ui/TabIdentity.svelte";
  import { setupTabState } from "../../lib/tabIdentity";
  import { onMount } from "svelte";
  import { fade, fly } from "svelte/transition";
  import InfoTip from "../../lib/ui/InfoTip.svelte";
  import PageHeader from "./PageHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
  import ModelProviderLogo from "./ModelProviderLogo.svelte";
  import { formatEtaRange } from "../../lib/runtime/eta";
  import type {
    HostedCheckState,
    HostedModelOption,
    HostedShellController,
    HostedShellSnapshot,
  } from "./types";
  import type { BrowserRuntimeClass } from "../../lib/runtime/contracts";
  import { contentIn, contentOut } from "../../lib/motion";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import { reloadInstructions, appRefreshSafetyMessage } from "../runtime/chunkRecovery";
  import { clearConversationOpen } from "../runtime/entryExperience";
  import { downloadUnavailableReason, modelsBySize } from "./modelSelection";

  let {
    controller,
    firstRun = true,
    onBack,
    workbenchError = null,
    workbenchNeedsReload = false,
    workbenchReloading = false,
    onRetryWorkbench,
  }: {
    controller: HostedShellController;
    firstRun?: boolean;
    onBack?: () => void;
    workbenchError?: string | null;
    workbenchNeedsReload?: boolean;
    workbenchReloading?: boolean;
    onRetryWorkbench?: () => void;
  } = $props();
  let snapshot = $state<HostedShellSnapshot>({
    phase: "idle",
    headline: "Check this device",
    detail: "Preparing the compatibility check.",
    checks: [],
    models: [],
    download: {
      available: false,
      phase: "locked",
      reason: "Checking the signed distribution configuration.",
    },
    runtime: {
      available: false,
      phase: "locked",
      reason: "Checking the local model engine.",
    },
  });
  let selectedModel = $state<string | null>(null);
  let baseModelsOpen = $state(false);
  let unsafeConfirmation = $state<string | null>(null);
  let confirmSessionReset = $state(false);
  let deleteConfirmation = $state<string | null>(null);
  let deletingModel = $state(false);
  let deletionError = $state<string | null>(null);
  let deletionNotice = $state<string | null>(null);
  const modelOperationBusy = $derived(deletingModel || snapshot.runtime.phase === "loading" || ["requesting_persistence", "downloading", "cancelling"].includes(snapshot.download.phase));
  async function deleteSelectedModel() {
    const id = deleteConfirmation;
    if (!id || modelOperationBusy) return;
    const name = snapshot.models.find(model => model.id === id)?.name ?? "Model";
    deletingModel = true;
    deletionError = null;
    try {
      await controller.deleteModel(id);
      deleteConfirmation = null;
      deletionNotice = `${name} removed from this device. Your chats and analysis packs are kept. Download the model again to continue its chats.`;
    } catch (error) {
      deletionError = userFacingError(error, "The model could not be removed. Close other Drowse tabs and try again.");
    } finally {
      deletingModel = false;
    }
  }
  let packSelectionError = $state<string | null>(null);
  let persistenceRetrying = $state(false);
  let persistenceRetryAttempted = $state(false);
  const persistenceRequestAvailable = typeof navigator.storage?.persist === "function";
  const storageUnprotected = $derived((snapshot.storage !== undefined && snapshot.storage.persisted !== true) || snapshot.download.persistenceDenied === true);
  let appleMobile = $state(false);
  let runtimeClass = $state<BrowserRuntimeClass | null>(null);
  let automaticOpenStarted = false;
  const workbenchReopenRequested = new URL(window.location.href).searchParams.get("reopen") === "1";
  const requestedModelId = new URL(window.location.href).searchParams.get("model");
  const selectedOption = $derived(
    snapshot.models.find((model) => model.id === selectedModel) ?? null,
  );
  const displayHeadline = $derived(
    snapshot.phase === "supported" && snapshot.models.some((model) => model.fit !== "blocked")
      ? firstRun
        ? "Choose your first model"
        : "Models"
      : snapshot.headline,
  );
  const displayDetail = $derived(
    snapshot.phase === "supported" && snapshot.models.some((model) => model.fit !== "blocked")
      ? firstRun
        ? "Choose a model to download to this device. The workbench opens when setup finishes."
        : "Open an installed model or download another one. Your saved chats are kept separately."
      : snapshot.detail,
  );
  const showDownloadAction = $derived(
    selectedOption !== null && selectedOption.fit !== "blocked" &&
    snapshot.download.available && (
      !selectedOption?.setupComplete ||
      (selectedOption?.remainingDownloadBytes ?? 0) > 0 ||
      ["requesting_persistence", "downloading", "cancelling", "paused", "failed"].includes(
        snapshot.download.phase,
      )
    ),
  );
  const downloadComplete = $derived(
    snapshot.phase === "supported" && selectedOption?.setupComplete === true &&
    selectedOption.fit !== "blocked" && !showDownloadAction &&
    !["requesting_persistence", "downloading", "cancelling", "paused", "failed"].includes(snapshot.download.phase),
  );
  const passedChecks = $derived(
    snapshot.checks.filter((check) => check.state === "pass").length,
  );
  const advisoryChecks = $derived(
    snapshot.checks.filter((check) => check.state === "warn").length,
  );
  const checkStatus = $derived(
    snapshot.phase === "checking" || snapshot.phase === "idle"
      ? "checking"
      : snapshot.checks.length > 0
        ? passedChecks === snapshot.checks.length
          ? "ready"
          : passedChecks > 0 ? "partial" : "unavailable"
        : snapshot.phase === "supported" ? "ready" : "unavailable",
  );
  const previewBrowser = $derived(
    runtimeClass === "desktop-webkit"
      ? "Safari preview"
      : runtimeClass === "desktop-gecko"
        ? "Firefox preview"
        : null,
  );
  const checkSummaryLabel = $derived(
    snapshot.phase === "checking"
      ? "Checking browser and graphics"
      : snapshot.phase === "supported"
          ? appleMobile
            ? advisoryChecks
              ? `iPhone preview checks passed with ${advisoryChecks} note${advisoryChecks === 1 ? "" : "s"}`
              : "iPhone preview checks passed"
            : previewBrowser
              ? advisoryChecks
                ? `${previewBrowser} checks passed with ${advisoryChecks} note${advisoryChecks === 1 ? "" : "s"}`
                : `${previewBrowser} checks passed`
          : advisoryChecks
            ? `Device ready with ${advisoryChecks} note${advisoryChecks === 1 ? "" : "s"}`
            : "Device ready"
        : snapshot.phase === "unsupported" || snapshot.phase === "failed"
          ? "Review these checks"
          : "Device check",
  );

  const checkLabel: Record<HostedCheckState, string> = {
    pending: "Waiting",
    running: "Checking",
    pass: "Ready",
    warn: "Needs verification",
    fail: "Unavailable",
  };

  const tierLabel: Record<HostedModelOption["tier"], string> = {
    fastest: "Fastest",
    balanced: "Balanced",
    quality: "Best quality",
  };

  const fitLabel: Record<HostedModelOption["fit"], string> = {
    recommended: "Recommended",
    eligible: "Good fit",
    uncertain: "Needs confirmation",
    blocked: "Not supported",
  };

  const formatBytes = (bytes?: number) => {
    if (bytes == null) return "Storage estimate unavailable";
    if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB available`;
    return `${Math.round(bytes / 1_000_000)} MB available`;
  };

  const formatByteCount = (bytes: number) => {
    if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
    if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
    if (bytes >= 1_000) return `${Math.round(bytes / 1_000)} KB`;
    return `${bytes} B`;
  };

  const packName = (pack: HostedModelOption["firstRunPacks"][number]) =>
    pack.name;

  const packTitle = (pack: HostedModelOption["firstRunPacks"][number]) =>
    pack.kind === "jlens"
      ? "Word insights"
      : /gemma scope/i.test(pack.name)
        ? "Gemma Scope features"
        : "Feature explorer";

  const packDescription = (pack: HostedModelOption["firstRunPacks"][number]) =>
    pack.kind === "jlens"
      ? `${pack.name}. Read word predictions across layers. This lens is already fitted; your browser doesn't train it.`
      : `${pack.name}. Inspect and steer learned features. This SAE is already trained; your browser doesn't train it.`;
  const appleStorageHelp =
    "Adding Drowse to your iPhone or iPad Home Screen can help it keep data. Back up your chats first. Existing chats and models won't transfer to the Home Screen app.";
  const macDesktop = /Macintosh|Mac OS X/i.test(navigator.userAgent) ||
    /^Mac/i.test(navigator.platform);
  const firefoxSwitchGuidance = macDesktop
    ? "Update Firefox to the latest version and check again. If it remains incompatible, switch to Safari or a Chromium-based browser such as Chrome, Edge, Brave, or Arc."
    : "Update Firefox to the latest version and check again. If it remains incompatible, switch to a Chromium-based browser such as Chrome, Edge, or Brave.";
  const firefoxIncompatible = $derived(
    runtimeClass === "desktop-gecko" &&
      snapshot.headline === "Firefox is incompatible with the current model runtime",
  );
  const recoveryGuidance = $derived(
    appleMobile
      ? "Review the failed check above. On iPhone or iPad, Drowse needs Safari 26 or later, WebGPU, and working local storage."
      : runtimeClass === "desktop-webkit"
        ? "Review the failed check above. Drowse needs Safari 26 on macOS 26, working local storage, and a graphics adapter that passes the WebGPU compute check. Lockdown Mode can disable WebGPU."
        : runtimeClass === "desktop-gecko"
          ? firefoxSwitchGuidance
          : "Use a current Safari, Firefox, Chrome, or Edge browser on a device with WebGPU, then check again.",
  );
  const runtimeGateTitle = $derived(
    snapshot.runtime.phase === "loading"
      ? "Loading the model"
      : snapshot.runtime.phase === "failed"
        ? "The model could not open"
        : "Ready to open",
  );
  const runtimeGateAction = $derived(
    snapshot.runtime.phase === "failed" ? "Try opening again" : "Open Drowse",
  );

  const speedClass = (speed: string) => {
    const lowerBound = Number.parseFloat(speed);
    if (!Number.isFinite(lowerBound)) return "Measured after the first reply";
    if (lowerBound >= 40) return "Very fast";
    if (lowerBound >= 20) return "Fast";
    if (lowerBound >= 8) return "Moderate";
    return "Slower";
  };

  const clearModelChoiceRequest = () => {
    const url = new URL(window.location.href);
    url.searchParams.delete("choose");
    url.searchParams.delete("model");
    url.searchParams.set("reopen", "1");
    window.history.replaceState(window.history.state, "", url);
  };

  const openInstalledModel = async (next: HostedShellSnapshot) => {
    if (
      !workbenchReopenRequested || automaticOpenStarted || next.phase !== "supported" ||
      !next.runtime.available || next.runtime.phase !== "unloaded"
    ) return;
    const model = next.models.find(
      (candidate) => candidate.id === next.selectedModelVariantId,
    ) ?? next.models.find((candidate) => candidate.setupComplete);
    if (!model?.setupComplete || model.fit === "blocked" || model.requiresOomRetry) return;
    automaticOpenStarted = true;
    selectedModel = model.id;
    try {
      await controller.open(model.id);
      clearModelChoiceRequest();
    } catch {}
  };

  onMount(() => {
    snapshot = controller.current();
    const unsubscribe = controller.subscribe((next) => {
      snapshot = next;
      const capabilities = controller.capabilities();
      appleMobile = capabilities?.signals.appleMobile === true;
      runtimeClass = capabilities?.signals.runtimeClass ?? null;
      if (!selectedModel || !next.models.some((model) => model.id === selectedModel)) {
        const requestedSelection = requestedModelId ?? (next.download.phase !== "idle" ? next.download.modelVariantId : undefined);
        selectedModel = next.models.find((model) => model.id === requestedSelection)?.id ?? null;
        if (next.models.find((model) => model.id === selectedModel)?.modelType === "base") {
          baseModelsOpen = true;
        }
      }
      void openInstalledModel(next);
    });
    return () => unsubscribe();
  });

  const etaLabel = (progress = snapshot.download.progress) => {
    if (!progress) return "Preparing download";
    if (progress.bytesTotal > 0 && progress.bytesReceived >= progress.bytesTotal) {
      return "Complete";
    }
    if (progress.offline) return "Offline. Waiting for a connection";
    if (progress.stalled) return "Waiting for download data";
    if (progress.calculatingEta || !progress.etaSeconds) return "Calculating…";
    return `${formatEtaRange(progress.etaSeconds)} remaining`;
  };

  const progressPercent = (progress = snapshot.download.progress) =>
    progress && progress.bytesTotal > 0
      ? Math.min(100, (progress.bytesReceived / progress.bytesTotal) * 100)
      : 0;

  const startDownload = async (explicitUnsafeOverride = false) => {
    if (!selectedModel || modelOperationBusy) return;
    if (selectedOption?.fit === "uncertain" && !explicitUnsafeOverride) {
      unsafeConfirmation = selectedModel;
      return;
    }
    const modelId = selectedModel;
    unsafeConfirmation = null;
    try {
      await controller.download(modelId, { explicitUnsafeOverride });
      const next = controller.current();
      const installed = next.models.find((model) => model.id === modelId);
      if (
        installed?.installed && next.download.phase === "installed" &&
        next.runtime.available &&
        next.runtime.phase !== "ready"
      ) {
        await controller.open(modelId);
        clearModelChoiceRequest();
      }
    } catch {}
  };

  const setPackSelected = (packId: string, selected: boolean) => {
    if (!selectedModel) return;
    try {
      controller.setOptionalPackSelected(selectedModel, packId, selected);
      packSelectionError = null;
    } catch (error) {
      packSelectionError = userFacingError(
        error,
        "That model tool could not be selected. Choose another tool or model and try again.",
      );
    }
  };

  const retryPersistence = async () => {
    if (persistenceRetrying) return;
    persistenceRetrying = true;
    try {
      persistenceRetryAttempted = !(await controller.retryPersistence());
    } catch {
      persistenceRetryAttempted = true;
    } finally {
      persistenceRetrying = false;
    }
  };

  const openModel = async (resetSession = false) => {
    if (!selectedModel) return;
    try {
      await controller.open(selectedModel, { resetSession });
      confirmSessionReset = false;
      clearModelChoiceRequest();
    } catch {}
  };
</script>

{#snippet storageProtectionNotice()}
  <div class="persistence-notice" role="status" aria-label="Storage protection">
    <div>
      <strong>Local storage is available</strong>
      <p class="persistence-help">
        {persistenceRetryAttempted
          ? "Downloads and chats still save on this device. Your browser may remove them during automatic cleanup, so back up your chats."
          : persistenceRequestAvailable
            ? "Downloads and chats save on this device. Storage protection is optional and helps prevent automatic cleanup."
            : "Downloads and chats save on this device, but the browser may remove them during automatic cleanup. Back up your chats."}
      </p>
      {#if appleMobile}<p class="persistence-help">{appleStorageHelp}</p>{/if}
    </div>
    {#if persistenceRequestAvailable}
      <button
        class="quiet-action"
        type="button"
        disabled={persistenceRetrying}
        aria-busy={persistenceRetrying}
        onclick={() => void retryPersistence()}
      >{persistenceRetrying ? "Requesting protection…" : "Protect storage"}</button>
    {/if}
  </div>
{/snippet}

<TabIdentity state={setupTabState(snapshot)} />

<a class="skip-link" href="#device-check">Skip to device check</a>

<div class="app-shell" class:hardware-compatible={snapshot.phase === "supported"} class:download-complete={downloadComplete}>
  <PageHeader current="models" onChats={onBack} />

  <main id="device-check">
    <div class="intro" data-page-group="0">
      <h1><MorphText text={displayHeadline} numbers={false} /></h1>
      <p>{displayDetail}</p>
      {#if firefoxIncompatible}<p>{firefoxSwitchGuidance}</p>{/if}
    </div>

    <div class="status-region" role="status" aria-live="polite" aria-atomic="true">
      <MorphText text={snapshot.download.phase === "downloading"
        ? "Local setup download in progress."
        : snapshot.download.phase === "cancelling"
          ? "Pausing local setup download."
          : snapshot.download.phase === "paused"
            ? `Local setup download paused. ${snapshot.download.reason}`
          : snapshot.download.phase === "installed"
            ? "Local setup download complete and verified."
            : snapshot.runtime.phase === "loading"
              ? "Loading the model on this device."
              : snapshot.runtime.phase === "ready"
                ? "The local model is ready."
        : snapshot.phase === "checking"
        ? "Device check in progress."
        : snapshot.phase === "supported"
          ? snapshot.runtime.available &&
              (snapshot.download.available || snapshot.models.some((model) => model.setupComplete))
            ? "Check complete."
            : "Check complete. Review model compatibility below."
          : snapshot.phase === "unsupported"
            ? "Check failed. Review the items below."
            : ""} />
    </div>
    <div class="alert-region" role="alert" aria-live="assertive" aria-atomic="true">
      {snapshot.phase === "failed"
        ? snapshot.detail
        : snapshot.download.phase === "failed"
          ? `Local setup download failed. ${snapshot.download.reason}`
        : snapshot.runtime.phase === "failed"
          ? snapshot.runtime.reason
          : ""}
    </div>

    {#if workbenchError}
      <section
        class="workbench-error"
        role="alert"
        in:fly={contentIn()}
        out:fly={contentOut()}
      >
        <div>
          <strong>The workbench could not open.</strong>
          <p>{workbenchError}</p>
          {#if workbenchNeedsReload}
            <p>{appRefreshSafetyMessage}</p>
            <p>{reloadInstructions()}</p>
          {/if}
        </div>
        <button
          type="button"
          disabled={workbenchReloading}
          onclick={onRetryWorkbench}
        ><MorphText text={workbenchReloading ? "Refreshing app…" : workbenchNeedsReload ? "Reload Drowse" : "Retry workbench"} /></button>
      </section>
    {/if}

    <section class="check-panel" data-status={checkStatus} data-page-group="1" class:loading-pulse={snapshot.phase === "checking"} aria-label="Device check" aria-busy={snapshot.phase === "checking"}>
      <h2 class="sr-only">Device checks</h2>
      <details use:animatedDetails class="device-details" open={snapshot.phase === "unsupported" || snapshot.phase === "failed"}>
        <summary>
          <span class="check-summary-copy">
            <strong><MorphText text={checkSummaryLabel} numbers={false} /></strong>
            {#if snapshot.checks.length > 0}<small><MorphText text={`${passedChecks} checks complete`} /></small>{/if}
          </span>
          <span>View details</span>
        </summary>
        <ul class="check-list">
          {#each snapshot.checks as check}
            <li class:failed={check.state === "fail"} class:warning={check.state === "warn"}>
              <span class="check-mark {check.state}" aria-hidden="true">
                <StateIcon icons={["ready", "error", "warning", "info"]} name={check.state === "pass" ? "ready" : check.state === "fail" ? "error" : check.state === "warn" ? "warning" : "info"} size={18} />
              </span>
              <div>
                <h3>{check.label}</h3>
                <p>{check.detail}</p>
              </div>
              <span class="check-label">{checkLabel[check.state]}</span>
            </li>
          {/each}
        </ul>
      </details>

      {#if snapshot.phase === "unsupported" || snapshot.phase === "failed"}
        <div class="recovery" in:fly={contentIn()} out:fly={contentOut()}>
          <p>{recoveryGuidance}</p>
          <button type="button" onclick={() => controller.check()}>Check again</button>
        </div>
      {/if}
    </section>

    {#if snapshot.phase === "supported"}
      <section
        class="model-panel"
        data-page-group="2"
        aria-labelledby="models-title"
        in:fly={contentIn(8)}
        out:fly={contentOut()}
      >
        <div
          class="panel-heading"
          class:choice-heading={snapshot.models.some((model) => model.fit !== "blocked")}
        >
            <h2 id="models-title" class:sr-only={snapshot.models.some((model) => model.fit !== "blocked")}>
              {snapshot.models.some((model) => model.fit !== "blocked")
                ? appleMobile
                  ? "Choose a compact model"
                  : "Choose a model"
                : "Current model requirements"}
            </h2>
          <div class="heading-side">
            <InfoTip label="About model choices" text="Smaller models use less memory and download faster. Larger models usually give better answers. The recommendation is an estimate of what this device can run." />
            <code>{formatBytes(snapshot.storage?.availableBytes)}</code>
          </div>
        </div>

        {#snippet modelCard(model: HostedModelOption)}
            <button
              type="button"
              class:selected={selectedModel === model.id}
              class:blocked={model.fit === "blocked"}
              disabled={modelOperationBusy}
              aria-disabled={model.fit === "blocked" && model.catalogAvailable !== false}
              aria-pressed={selectedModel === model.id}
              onclick={() => {
                if (model.id !== requestedModelId) clearConversationOpen();
                selectedModel = model.id;
                unsafeConfirmation = null;
                confirmSessionReset = false;
                deleteConfirmation = null;
                deletionError = null;
                packSelectionError = null;
              }}
            >
              <span class="model-topline">
                <span>{tierLabel[model.tier]}</span>
                <code>{model.catalogAvailable === false ? "Chat model" : model.size}</code>
              </span>
              <span class="model-name">
                <ModelProviderLogo modelId={model.id} />
                <strong><span>{model.name}</span>{#if model.modelType === "base"}<BaseModelTag plain />{/if}</strong>
                {#if model.firstRunPacks.some((pack) => pack.kind === "sae")}
                  <span class="sae-available" {...{ "aria-description": "Sparse autoencoder available" }}>SAE</span>
                {/if}
              </span>
              {#if selectedModel === model.id}
                <span class="model-meta">{model.context} · {model.language}</span>
                {#if model.expectedSpeed}
                  <span class="model-detail" {...{ "aria-description": (`Measured estimate: ${model.expectedSpeed}. A token is a short piece of text.`) }}>
                    Expected response speed: {speedClass(model.expectedSpeed)}
                  </span>
                {/if}
                {#if model.estimatedDownloadSeconds}
                  <span class="model-detail">Estimated download: <MorphText text={formatEtaRange(model.estimatedDownloadSeconds)} /></span>
                {/if}
                {#if model.catalogAvailable !== false}
                <span class="model-detail">
                  Setup download: <MorphText text={formatByteCount(model.modelDownloadBytes + model.firstRunPacks.filter((pack) => pack.requiredForSetup).reduce((total, pack) => total + pack.bytes, 0))} />
                </span>
                {/if}
                {#if model.firstRunPacks.some((pack) => pack.selected && !pack.requiredForSetup)}
                  <span class="model-detail">
                    Also selected: {model.firstRunPacks.filter((pack) => pack.selected && !pack.requiredForSetup).map(packName).join(", ")}
                  </span>
                {/if}
                <span class="model-reason">{model.reason}</span>
              {/if}
              <span class="model-fit {model.fit}">{model.catalogAvailable === false ? "Download unavailable" : model.requiresOomRetry ? "Previously ran out of memory" : fitLabel[model.fit]}</span>
              {#if model.installed}
                <span class="installed-mark">
                  {model.setupComplete ? "Installed and verified" : "Model downloaded · tools still needed"}
                </span>
              {/if}
            </button>
        {/snippet}

        <div class="model-grid">
          {#each modelsBySize(snapshot.models.filter((model) => model.modelType !== "base")) as model}
            {@render modelCard(model)}
          {/each}
        </div>

        <details use:animatedDetails class="base-models" bind:open={baseModelsOpen}>
          <summary>Base models <span class="base-models-note">Advanced · text completion</span></summary>
          <p class="base-model-warning">
            Base models predict what comes next in a piece of text. They don't follow instructions reliably
            and can produce unreliable or offensive text. Use a chat model for everyday conversations.
          </p>
          {#if snapshot.models.some((model) => model.modelType === "base")}
            <p class="base-model-description">Each download includes the core pack for generation and concept steering, plus compatible SAE features when available. J-lens packs are optional for base models.</p>
            <div class="model-grid">
              {#each modelsBySize(snapshot.models.filter((model) => model.modelType === "base")) as model}
                {@render modelCard(model)}
              {/each}
            </div>
          {:else}
            <p class="base-model-description">No verified base-model downloads are available in this catalog yet. Choose a chat model above to continue.</p>
          {/if}
        </details>

        {#if selectedOption}
          {#key selectedOption.id}
          <div class="selected-model-details" in:fly={contentIn()}>
          <p class="model-provenance">
            <a href={selectedOption.sourceUrl} target="_blank" rel="noreferrer">
              {selectedOption.name} source
            </a>
            <span aria-hidden="true">·</span>
            <span>{selectedOption.license}</span>
          </p>

          {#if selectedOption.fit !== "blocked"}
          <section class="tool-picker" aria-labelledby="tools-title">
            <div class="tool-picker-heading">
              <div>
                <h3 id="tools-title">Choose {selectedOption.name}</h3>
              </div>
              <InfoTip
                label="About model tools"
                text={selectedOption.modelType === "base"
                  ? "This setup includes response controls and compatible SAE features when available. Word insights are optional for base models."
                  : "This setup includes response controls, word insights, and compatible SAE features when available. These packs are already trained."}
              />
            </div>
            <div class="tool-option required-tool">
              <span class="tool-check" aria-hidden="true"><FluentIcon name="check" /></span>
              <div>
                <strong>Response controls</strong>
                <p>Measure and shape replies.</p>
              </div>
              <span>Included</span>
            </div>
            {#each selectedOption.firstRunPacks.filter((pack) => pack.requiredForSetup) as pack (pack.id)}
              <div class="tool-option required-tool" class:tool-installed={pack.installed}>
                <span class="tool-check" aria-hidden="true"><FluentIcon name="check" /></span>
                <div>
                  <strong>{pack.kind === "jlens" ? "Word insights" : "Feature insights"} <span>({pack.kind === "jlens" ? "J-lens" : "SAE"})</span></strong>
                  <p>{pack.kind === "jlens" ? `${pack.name}. Read word predictions across layers and steer toward a word.` : packDescription(pack)}</p>
                </div>
                <span><MorphText text={pack.installed ? "Already downloaded" : formatByteCount(pack.bytes)} /></span>
              </div>
            {/each}
            {#if selectedOption.setupIssue}
              <p class="tool-error" role="alert">{selectedOption.setupIssue}</p>
            {:else}
              {#if selectedOption.toolNotice}
                <p class="tool-empty">{selectedOption.toolNotice}</p>
              {/if}
              {#if selectedOption.firstRunPacks.some((pack) => !pack.requiredForSetup)}
                {#each selectedOption.firstRunPacks.filter((pack) => !pack.requiredForSetup) as pack (pack.id)}
                  <label class="tool-option" class:tool-installed={pack.installed}>
                    <input
                      type="checkbox"
                      checked={pack.selected}
                      disabled={pack.installed || snapshot.download.phase === "requesting_persistence" || snapshot.download.phase === "downloading" || snapshot.download.phase === "cancelling"}
                      onchange={(event) => setPackSelected(pack.id, event.currentTarget.checked)}
                    />
                    <div>
                      <strong>{pack.kind === "jlens" ? selectedOption.modelType === "base" ? pack.name : "Alternative R-lens" : packTitle(pack)} <span>({pack.kind === "jlens" ? "optional readout" : "optional SAE"})</span></strong>
                      <p>{packDescription(pack)}</p>
                    </div>
                    <span><MorphText text={pack.installed ? "Already downloaded" : formatByteCount(pack.bytes)} /></span>
                  </label>
                {/each}
              {:else if !selectedOption.toolNotice}
                <p class="tool-empty">No other tools are available for this model and context size.</p>
              {/if}
            {/if}
            <div class="tool-total">
              <span>{selectedOption.installed ? "Download remaining" : "First download"}</span>
              <strong><MorphText text={formatByteCount(selectedOption.remainingDownloadBytes)} /></strong>
            </div>
            {#if packSelectionError}
              <p class="tool-error" role="alert">{packSelectionError}</p>
            {/if}
          </section>
          {/if}
          </div>
          {/key}
        {/if}

        {#if selectedOption?.installed}
          <div class="model-storage">
            <div>
              <strong>Stored on this device</strong>
              <p>Remove the model files to free storage. Your chats and analysis packs are kept.</p>
            </div>
            {#if deleteConfirmation === selectedModel}
              <div class="delete-confirmation">
                <p>Delete {selectedOption.name} from this device? You will need to download it again to continue its chats.</p>
                <div class="warning-actions">
                  <button type="button" class="secondary" disabled={deletingModel} onclick={() => deleteConfirmation = null}>Cancel</button>
                  <button type="button" class="delete-model" disabled={modelOperationBusy} onclick={() => void deleteSelectedModel()}><MorphText text={deletingModel ? "Deleting…" : "Delete model"} /></button>
                </div>
              </div>
            {:else}
              <button type="button" class="secondary" disabled={modelOperationBusy} onclick={() => { deleteConfirmation = selectedModel; deletionNotice = null; }}>Delete from device</button>
            {/if}
            {#if deletionError}<p role="alert" class="tool-error">{deletionError}</p>{/if}
          </div>
        {/if}
        {#if deletionNotice}<p role="status">{deletionNotice}</p>{/if}

        {#if selectedOption?.installed && !showDownloadAction}
          <div class="release-gate runtime-gate" class:loading-pulse={snapshot.runtime.phase === "loading"}>
            <div>
              <strong><MorphText text={runtimeGateTitle} /></strong>
              <p>{snapshot.runtime.reason}</p>
              {#if storageUnprotected}
                {@render storageProtectionNotice()}
              {/if}
            </div>
            <button
              type="button"
              disabled={modelOperationBusy || !selectedOption.setupComplete || !snapshot.runtime.available || snapshot.runtime.resetSessionAvailable}
              onclick={() => {
                if (selectedOption.fit === "uncertain") unsafeConfirmation = selectedModel;
                else void openModel();
              }}
            >
              <MorphText text={!selectedOption.setupComplete ? "Add selected tools first" : snapshot.runtime.phase === "loading" ? "Loading…" : snapshot.download.phase === "requesting_persistence" || snapshot.download.phase === "downloading" || snapshot.download.phase === "cancelling" ? "Finishing setup…" : snapshot.runtime.resetSessionAvailable ? "Conversation needs attention" : selectedOption.fit === "uncertain" ? "Review warning" : runtimeGateAction} />
            </button>
          </div>
          {#if selectedOption.fit === "uncertain" && unsafeConfirmation === selectedModel}
            <div
              class="unsafe-warning"
              role="alert"
              in:fly={contentIn()}
              out:fly={contentOut()}
            >
              <div>
                <strong>{selectedOption.requiresOomRetry ? "This model ran out of memory with these settings." : "Check this model on your device"}</strong>
                <p>{selectedOption.reason} The model may run out of memory or lose its graphics connection.</p>
              </div>
              <div class="warning-actions">
                <button class="safe-action" type="button" onclick={() => (unsafeConfirmation = null)}>Choose another model</button>
                <button class="warning-action" type="button" onclick={() => void openModel()}>{selectedOption.requiresOomRetry ? "Retry this model" : "Load anyway"}</button>
              </div>
            </div>
          {/if}
        {/if}

        {#if selectedOption?.installed && snapshot.runtime.resetSessionAvailable}
          <div
            class="unsafe-warning"
            role="alert"
            in:fly={contentIn()}
            out:fly={contentOut()}
          >
            <div>
              <strong>The saved conversation cannot open with this runtime.</strong>
              <p>Starting over removes the conversation but keeps the model.</p>
            </div>
            <div class="warning-actions">
              {#if confirmSessionReset}
                <button class="safe-action" type="button" onclick={() => (confirmSessionReset = false)}>Keep conversation</button>
                <button class="danger-action" type="button" onclick={() => void openModel(true)}>Remove conversation and start over</button>
              {:else}
                <button class="safe-action" type="button" onclick={() => (confirmSessionReset = true)}>Review start-over option</button>
              {/if}
            </div>
          </div>
        {/if}

        {#if showDownloadAction}
          <div class="release-gate download-gate" class:loading-pulse={snapshot.download.phase === "requesting_persistence" || snapshot.download.phase === "cancelling" || (snapshot.download.phase === "downloading" && !snapshot.download.progress?.offline && !snapshot.download.progress?.stalled)}>
            <div>
              {#if snapshot.download.phase !== "idle"}
              <strong><MorphText text={snapshot.download.phase === "installed" ? "Setup complete" : snapshot.download.phase === "requesting_persistence" ? "Preparing download" : snapshot.download.phase === "downloading" ? "Downloading model files" : snapshot.download.phase === "cancelling" ? "Pausing download" : snapshot.download.phase === "paused" ? "Download paused" : "Download interrupted"} /></strong>
              {#if snapshot.download.reason !== "Ready to download."}<p>{snapshot.download.reason}</p>{/if}
              {/if}
              {#if storageUnprotected}
                {@render storageProtectionNotice()}
              {/if}
              {#if snapshot.download.progress}
                <div
                  class="progress-track"
                  role="progressbar"
                  aria-label="Local setup download"
                  aria-valuemin="0"
                  aria-valuemax="100"
                  aria-valuenow={Math.round(progressPercent())}
                  aria-valuetext={`${Math.round(progressPercent())}% complete. ${etaLabel()}.`}
                  in:fade={{ duration: contentIn().duration, easing: contentIn().easing }}
                  out:fade={{ duration: contentOut().duration, easing: contentOut().easing }}
                >
                  <span style={`width: ${progressPercent()}%`}></span>
                </div>
                <p class="progress-copy">
                  <RollingNumber value={Math.round(progressPercent())} />% · <MorphText text={etaLabel()} />
                </p>
              {/if}
            </div>
            {#if snapshot.download.phase === "downloading" || snapshot.download.phase === "cancelling"}
              <button
                type="button"
                disabled={snapshot.download.phase === "cancelling"}
                onclick={() => void controller.cancelDownload()}
              >
                <MorphText text={snapshot.download.phase === "cancelling" ? "Pausing…" : "Pause download"} />
              </button>
            {:else}
              <button
                type="button"
                disabled={modelOperationBusy || !selectedModel || selectedOption?.setupComplete || (snapshot.download.phase === "installed" && snapshot.download.modelVariantId === selectedModel)}
                onclick={() => void startDownload(false)}
              >
                <MorphText text={selectedOption?.setupComplete ? "Installed" : snapshot.download.phase === "paused" ? "Resume download" : snapshot.download.phase === "failed" ? "Retry download" : selectedOption?.fit === "uncertain" ? "Review warning" : snapshot.download.phase === "requesting_persistence" ? "Preparing…" : selectedOption?.installed ? "Add tools and open" : "Download and open"} />
              </button>
            {/if}
          </div>
          {#if selectedOption?.fit === "uncertain" && !selectedOption.installed && unsafeConfirmation === selectedModel}
            <div
              class="unsafe-warning"
              role="alert"
              in:fly={contentIn()}
              out:fly={contentOut()}
            >
              <div>
                <strong>{selectedOption.requiresOomRetry ? "This model ran out of memory with these settings." : "Check this model on your device"}</strong>
                <p>{selectedOption.reason} The model may run out of memory or lose its graphics connection.</p>
              </div>
              <div class="warning-actions">
                <button class="safe-action" type="button" onclick={() => (unsafeConfirmation = null)}>Choose another model</button>
                <button class="warning-action" type="button" onclick={() => void startDownload(true)}>{selectedOption.requiresOomRetry ? "Retry download" : "Download anyway"}</button>
              </div>
            </div>
          {/if}
        {:else if (!snapshot.download.available || selectedOption?.fit === "blocked") && !selectedOption?.setupComplete}
          <div class="release-gate unavailable-gate">
            <div>
              <strong>{selectedOption?.fit === "blocked" ? "This model is unavailable" : "Downloads are unavailable"}</strong>
              <p>{downloadUnavailableReason(selectedOption, snapshot.download)}</p>
            </div>
            <button type="button" disabled>Download and open</button>
          </div>
        {/if}
      </section>
    {/if}

  </main>

  <PageFooter />
</div>

<style>
  .skip-link {
    position: fixed;
    z-index: 1000;
    top: 8px;
    inset-inline-start: 8px;
    padding: var(--space-xs) var(--space-sm);
    border-radius: var(--radius);
    background: var(--accent);
    color: var(--text-on-accent);
    transform: translateY(-160%);
  }
  .skip-link:focus { transform: translateY(0); }

  .app-shell {
    display: flex;
    flex-direction: column;
    position: relative;
    isolation: isolate;
    min-height: 100vh;
    min-height: 100dvh;
    box-sizing: border-box;
    background: var(--ambient-canvas);
  }

  main {
    position: relative;
    z-index: 1;
    width: min(100%, var(--page-max));
    margin-inline: auto;
  }


  main { flex: 1; padding-block: var(--surface-padding) var(--surface-padding); }
  main {
    padding-inline: max(var(--surface-padding), env(safe-area-inset-left)) max(var(--surface-padding), env(safe-area-inset-right));
  }
  .intro { max-width: var(--copy-measure); }
  h1 {
    margin: 0;
    color: var(--fg);
    font-size: var(--text-onboarding-heading);
    font-weight: var(--weight-display);
    letter-spacing: -0.065em;
    line-height: 1.05;
  }
  .intro > p:last-child {
    max-width: var(--copy-measure);
    margin: var(--space-md) 0 0;
    color: var(--fg-dim);
    font-size: var(--text-onboarding-lede);
    line-height: 1.6;
  }

  .status-region,
  .alert-region {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
  .workbench-error {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-md);
    margin-top: var(--space-lg);
    padding: var(--surface-padding);
    border: 1px solid color-mix(in srgb, var(--accent-red) 38%, transparent);
    border-radius: var(--radius-lg);
    background: color-mix(in srgb, var(--accent-red) 8%, var(--input-well));
  }
  .workbench-error > div { flex: 1 1 24rem; min-width: 0; }
  .workbench-error > button { flex: 0 0 auto; max-width: 100%; padding: var(--space-4) var(--space-6); }
  .workbench-error strong { color: var(--accent-red); }
  .workbench-error p { margin: var(--space-xs) 0 0; color: var(--fg-dim); }

  .check-panel { margin-top: var(--space-xl); }
  .model-panel { margin-top: var(--space-xl); }
  .check-panel,
  .model-panel { padding: 0; }
  .check-panel {
    --check-bg: var(--bg-alt);
    --check-ink: var(--fg-strong);
    padding: var(--surface-padding);
    border-radius: var(--radius-lg);
    background: var(--check-bg);
    color: var(--check-ink);
    box-shadow: var(--shadow-card);
  }
  .check-panel[data-status="ready"] { --check-bg: var(--success-bg); --check-ink: var(--success-ink); }
  .check-panel[data-status="partial"] { --check-bg: var(--warning-bg); --check-ink: var(--warning-ink); }
  .check-panel[data-status="unavailable"] { --check-bg: var(--danger-bg); --check-ink: var(--danger-ink); }
  @media (prefers-contrast: more) {
    .check-panel { outline: 2px solid var(--check-ink); }
  }
  @media (forced-colors: active) {
    .check-panel[data-status] { --check-bg: Canvas; --check-ink: CanvasText; outline: 1px solid CanvasText; }
  }

  .panel-heading {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: var(--space-md);
    padding-bottom: var(--space-md);
    border-bottom: 1px solid var(--glass-line);
  }
  .panel-heading.choice-heading { border-bottom: 0; }
  h2 { margin: 0; color: var(--fg); font-size: var(--text-onboarding-heading); font-weight: var(--weight-structure-bold); letter-spacing: -0.035em; }
  .heading-side {
    margin-inline-start: auto;
    display: flex;
    align-items: center;
    gap: var(--space-3);
    color: var(--fg-muted);
    font-family: var(--font-data);
    font-size: var(--text-xs);
  }
  .device-details > summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 3.5rem;
    padding-block: var(--space-4);
    color: var(--check-ink);
    cursor: pointer;
    font-family: var(--font-structure);
    list-style: none;
  }
  .device-details > summary::-webkit-details-marker { display: none; }
  .device-details > summary span:last-child {
    color: var(--check-ink);
    font-family: var(--font-data);
    font-size: var(--text-xs);
  }
  .check-summary-copy {
    display: flex;
    align-items: baseline;
    gap: var(--space-sm);
  }
  .check-summary-copy strong {
    color: var(--check-ink);
    font-family: var(--font-structure);
    font-size: var(--text-md);
    font-weight: var(--weight-structure-bold);
  }
  .check-summary-copy small {
    color: var(--check-ink);
    font-family: var(--font-data);
    font-size: var(--text-xs);
  }
  .device-details[open] > summary span:last-child { font-size: 0; }
  .device-details[open] > summary span:last-child::before {
    content: "Hide details";
    font-size: var(--text-xs);
  }

  .check-list {
    display: grid;
    gap: var(--space-6);
    margin: var(--space-6) 0 0;
    padding: 0;
    list-style: none;
  }
  .check-list li {
    display: grid;
    grid-template-columns: 30px minmax(0, 1fr) auto;
    gap: var(--space-sm);
    align-items: start;
    min-height: 3.5rem;
    padding: var(--space-3) 0;
  }
  .check-mark {
    display: grid;
    place-items: center;
    margin-top: var(--space-xs);
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--glass-strong);
    color: var(--check-ink);
    font-family: var(--font-mono);
    font-weight: var(--weight-bold);
    transform: scale(1);
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      transform var(--dur) var(--ease-out);
  }
  .check-mark.pending,
  .check-mark.running { transform: scale(0.97); }
  .check-mark.pass { background: color-mix(in srgb, var(--live) 17%, transparent); color: var(--live); }
  .check-mark.fail { background: color-mix(in srgb, var(--accent-red) 17%, transparent); color: var(--accent-red); }
  .check-mark.warn { background: color-mix(in srgb, var(--accent-yellow) 17%, transparent); color: var(--accent-yellow); }
  .check-list h3 { margin: var(--space-xs) 0 0; color: var(--check-ink); font-size: var(--text-md); font-weight: var(--weight-medium); }
  .check-list p { margin: var(--space-xs) 0 0; color: var(--check-ink); line-height: 1.5; }
  .check-label { margin-top: var(--space-xs); color: var(--check-ink); font-family: var(--font-mono); font-size: var(--text-xs); }

  .recovery,
  .release-gate {
    display: grid;
    gap: var(--space-md);
    margin-top: calc(var(--space-unit) * 7);
    padding: var(--surface-padding) 0 0;
    border-top: 1px solid var(--glass-line);
    border-radius: 0;
    background: transparent;
  }
  .recovery p,
  .release-gate p { margin: 0; color: var(--fg-dim); line-height: 1.5; }
  .recovery p { color: var(--check-ink); }
  .release-gate { padding: var(--surface-padding); border-radius: var(--radius); }
  .recovery > button,
  .release-gate > button {
    width: 100%;
    min-height: 56px;
    font-size: var(--text-md);
  }
  button {
    min-height: 42px;
    padding: 0 var(--space-sm);
    border: 0;
    border-radius: var(--radius);
    background: var(--control-sheen), var(--action-bg);
    box-shadow: var(--shadow-control);
    color: var(--action-ink);
    font-family: var(--font-structure);
    font-weight: var(--weight-structure);
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out),
      box-shadow var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }
  button:disabled { cursor: not-allowed; background: var(--glass-strong); color: var(--fg-muted); box-shadow: none; }
  button:hover:not(:disabled) { box-shadow: var(--shadow-control-hover); }
  button:active:not(:disabled) { transform: scale(var(--press-scale)); box-shadow: var(--shadow-control-pressed); }

  .model-grid { display: grid; grid-template-columns: 1fr; gap: var(--space-6); }
  .model-storage { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-5); margin-block: var(--space-7); padding-block: var(--space-5); }
  .model-storage > div:first-child { flex: 1 1 18rem; }
  .model-storage p { margin-block: var(--space-3); overflow-wrap: anywhere; }
  .delete-confirmation { flex-basis: 100%; }
  .delete-model { color: var(--accent-red); background: var(--control-sheen), var(--surface-hi); }
  .base-models { margin-block: var(--space-7); }
  .base-models > summary {
    padding-block: var(--space-5);
    cursor: pointer;
    color: var(--fg);
    font-weight: var(--weight-medium);
  }
  .base-models-note {
    display: inline-block;
    margin-inline-start: var(--space-4);
    color: var(--fg-muted);
    font-size: var(--text-sm);
    font-weight: var(--weight-normal);
  }
  .base-model-warning,
  .base-model-description {
    margin: 0 0 var(--space-6);
    max-width: 70ch;
    color: var(--fg-dim);
    line-height: 1.6;
  }
  .base-model-warning { color: var(--accent-yellow); }
  .model-grid > button {
    display: grid;
    grid-template-columns: minmax(115px, 0.32fr) minmax(0, 1fr) minmax(130px, auto);
    gap: var(--space-xs) var(--space-md);
    align-items: baseline;
    min-height: 100px;
    padding: var(--surface-padding);
    border: 0;
    border-radius: var(--radius);
    background: var(--surface-card);
    background-image: var(--surface-sheen) !important;
    box-shadow: var(--shadow-card);
    color: var(--fg);
    text-align: start;
  }
  .model-grid > button:hover:not(:disabled) {
    background: var(--surface-card-hover);
    box-shadow: var(--shadow-rack-hover);
    transform: translateY(-1px);
  }
  .model-grid > button:active:not(:disabled) {
    transform: scale(var(--press-scale));
  }
  .model-grid > button.selected,
  .model-grid > button.selected:hover:not(:disabled),
  .model-grid > button.selected:active:not(:disabled) {
    background: color-mix(in srgb, var(--pillar-lens) 8%, var(--surface-card));
    box-shadow: 0 0 0 2px var(--pillar-lens), 0 10px 28px -8px color-mix(in srgb, var(--pillar-lens) 40%, transparent), var(--shadow-card);
  }
  .model-grid > button.blocked {
    background: var(--danger-bg);
  }
  .model-topline { grid-row: 1 / span 5; display: flex; flex-direction: column; gap: var(--space-xs); color: var(--fg-muted); font-family: var(--font-mono); font-size: var(--text-xs); }
  .model-topline code { color: var(--fg-muted); text-transform: none; }
  .model-name {
    grid-column: 2;
    display: flex;
    min-width: 0;
    align-items: center;
    gap: var(--space-sm);
  }
  .model-name strong { display: flex; align-items: baseline; gap: var(--space-xs); min-width: 0; margin: 0; font-family: var(--font-structure); font-size: var(--text-onboarding-card-title); font-weight: var(--weight-structure-bold); line-height: 1.2; }
  .model-name strong > span { min-width: 0; overflow-wrap: anywhere; }
  .sae-available {
    flex: 0 0 auto;
    padding: var(--space-xs) var(--space-xs);
    border-radius: var(--radius-pill);
    background: color-mix(in srgb, var(--pillar-sae) 12%, transparent);
    color: var(--pillar-sae);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
    font-weight: var(--weight-data-bold);
    letter-spacing: 0.08em;
    line-height: 1;
  }
  .model-meta { margin-top: var(--space-xs); color: var(--fg-muted); font-family: var(--font-mono); font-size: var(--text-xs); }
  .model-meta,
  .model-detail,
  .model-reason { grid-column: 2; }
  .model-detail { margin-top: var(--space-xs); color: var(--fg-muted); font-family: var(--font-mono); font-size: var(--text-xs); }
  .model-reason { max-width: 58ch; margin-top: var(--space-xs); color: var(--fg-dim); font-size: var(--text-sm); line-height: 1.5; }
  .model-fit { grid-column: 3; grid-row: 1; text-align: end; font-family: var(--font-mono); font-size: var(--text-xs); }
  .model-fit.recommended,
  .model-fit.eligible { color: var(--live); }
  .model-fit.uncertain { color: var(--accent-yellow); }
  .model-fit.blocked { color: var(--accent-red); }
  .installed-mark { grid-column: 3; color: var(--live); font-family: var(--font-mono); font-size: var(--text-xs); text-align: end; }
  .model-provenance {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: var(--space-xs);
    margin: var(--space-sm) var(--space-xs) 0;
    color: var(--fg-muted);
    font-family: var(--font-mono);
    font-size: var(--text-xs);
  }
  .selected-model-details { min-width: 0; }
  .model-provenance a { display: inline-flex; align-items: center; min-height: var(--control-target); color: var(--fg-dim); text-underline-offset: 3px; }
  .model-provenance a:hover { color: var(--fg); }
  .tool-picker {
    display: grid;
    gap: var(--space-5);
    margin-top: var(--space-lg);
    padding: var(--space-lg) 0 0;
    border-top: 1px solid var(--glass-line);
  }
  .tool-picker-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-4);
  }
  .tool-picker h3 {
    margin: 0;
    color: var(--fg);
    font-family: var(--font-structure);
    font-size: var(--text-onboarding-subheading);
    font-weight: var(--weight-structure-bold);
  }
  .tool-option {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    gap: var(--space-5);
    align-items: start;
    min-height: var(--control-target);
    padding: var(--space-4) 0;
    border-bottom: 1px solid var(--glass-line);
    border-radius: 0;
    background: transparent;
    color: var(--fg);
  }
  label.tool-option { cursor: pointer; }
  label.tool-option:has(input:focus-visible) {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }
  .tool-option input {
    width: 18px;
    height: 18px;
    margin: var(--space-xs) 0 0;
    accent-color: var(--accent);
  }
  .tool-option strong {
    display: block;
    font-family: var(--font-structure);
    font-weight: var(--weight-structure);
  }
  .tool-option strong span {
    color: var(--fg-muted);
    font-family: var(--font-data);
    font-size: var(--text-xs);
    font-weight: var(--weight-medium);
  }
  .tool-option p,
  .tool-empty {
    margin: var(--space-1) 0 0;
    color: var(--fg-dim);
    font-family: var(--font-reading);
    line-height: 1.5;
  }
  .tool-option > span:last-child,
  .tool-total {
    color: var(--fg-muted);
    font-family: var(--font-data);
    font-size: var(--text-xs);
  }
  .tool-check {
    display: grid;
    place-items: center;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: color-mix(in srgb, var(--live) 17%, transparent);
    color: var(--live);
    font-family: var(--font-data);
  }
  .tool-installed { opacity: 0.78; }
  .tool-total {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--space-4);
    padding: var(--space-2) var(--space-1) 0;
  }
  .tool-total strong { color: var(--fg); font-size: var(--text-sm); }
  .tool-error { margin: 0; color: var(--accent-red); }
  .runtime-gate { margin-top: calc(var(--space-unit) * 7); border: 0; }
  .unavailable-gate { border-top: 0; }
  .unsafe-warning {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-6);
    margin-top: var(--space-6);
    padding: var(--surface-padding);
    border: 1px solid var(--warning-border);
    border-radius: var(--radius-lg);
    background: var(--warning-bg);
  }
  .unsafe-warning strong { color: var(--warning-ink); }
  .unsafe-warning p { margin: var(--space-xs) 0 0; color: var(--warning-ink); line-height: 1.5; }
  .warning-actions { display: flex; flex-wrap: wrap; gap: var(--space-6); flex: 0 0 auto; }
  .unsafe-warning .safe-action {
    background: var(--action-bg);
    color: var(--action-ink);
  }
  .unsafe-warning .danger-action {
    background: var(--danger-bg);
    color: var(--accent-red);
  }
  .unsafe-warning .warning-action {
    background: var(--warning-action);
    color: var(--warning-action-ink);
  }
  .unsafe-warning .warning-action:hover:not(:disabled) { background: var(--warning-action-hover); }
  .release-gate strong { color: var(--fg); }
  .release-gate p { margin-top: var(--space-xs); max-width: 680px; }
  .persistence-notice { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: start; gap: var(--space-6); margin-top: var(--space-6); }
  .persistence-notice strong { color: var(--accent-yellow); }
  .persistence-notice p { margin: var(--space-1) 0 0; }
  .warning-copy { color: var(--accent-yellow) !important; }
  .persistence-help { color: var(--fg-dim); }
  .quiet-action { min-height: var(--control-target); border: 0; background: var(--control-sheen), var(--bg-elev); color: var(--fg); }
  .progress-track {
    width: min(520px, 100%);
    height: 7px;
    margin-top: var(--space-sm);
    overflow: hidden;
    border-radius: var(--radius-pill);
    background: var(--glass-strong);
  }
  .progress-track span {
    display: block;
    height: 100%;
    border-radius: inherit;
    background: var(--accent);
    transition: width var(--dur) linear;
  }
  .progress-copy { font-family: var(--font-mono); font-size: var(--text-xs); }

  @keyframes pulse { 50% { opacity: 0.35; } }

  @media (max-width: 760px) {
    .model-grid > button { grid-template-columns: 1fr auto; min-height: 0; }
    .model-topline { grid-column: 1 / -1; grid-row: auto; flex-direction: row; justify-content: space-between; }
    .model-name { grid-column: 1 / -1; }
    .model-meta, .model-detail, .model-reason { grid-column: 1 / -1; }
    .model-fit, .installed-mark { grid-column: auto; grid-row: auto; text-align: start; }
    .unsafe-warning { align-items: stretch; flex-direction: column; }
    .persistence-notice { grid-template-columns: 1fr; }
    .persistence-notice .quiet-action { width: 100%; }
    .tool-option { grid-template-columns: auto minmax(0, 1fr); }
    .tool-option > span:last-child { grid-column: 2; }
    .warning-actions { flex-direction: column; }
    button { min-height: 44px; }
    .check-list li { grid-template-columns: 30px minmax(0, 1fr); }
    .check-label { grid-column: 2; margin-top: 0; }
  }

  @media (max-width: 460px) {
    .panel-heading { align-items: start; flex-direction: column; }
    .device-details > summary { align-items: flex-start; flex-direction: column; justify-content: center; gap: var(--space-xs); }
    .check-summary-copy { align-items: flex-start; flex-direction: column; gap: var(--space-xs); }
  }

  @media (max-height: 600px) and (orientation: landscape) {
    main {
      padding-top: var(--surface-padding);
    }
  }

  @media (prefers-reduced-motion: no-preference) {
    .recovery button,
    .release-gate button,
    .unsafe-warning button,
    .workbench-error button {
      transition: transform var(--dur-fast) var(--ease-out);
    }
    .check-mark.running { animation: pulse 1.2s var(--ease-in-out) infinite; }
  }
</style>
