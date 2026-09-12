<script lang="ts">
  import TabIdentity from "../../lib/ui/TabIdentity.svelte";
  import { onDestroy, onMount, tick, type Component } from "svelte";
  import { fade, fly } from "svelte/transition";
  import { createShellController } from "../../../hosted/shell-controller";
  import { mountHostedWebMcp } from "../../lib/webmcp";
  import { registerPageActions } from "../../lib/webmcp/lifecycle";
  import {
    installHostedController,
    installRuntimeCapabilities,
    installRuntimeClient,
  } from "../../lib/runtime/registry";
  import HostedApp from "./HostedApp.svelte";
  import HostedHome from "./HostedHome.svelte";
  import PageHeader from "./PageHeader.svelte";
  import PageFooter from "./PageFooter.svelte";
  import PwaUpdatePrompt from "./PwaUpdatePrompt.svelte";
  import { conversationLibrary, flushConversationAutosave } from "../../lib/stores/savedConversations.svelte";
  import type { SavedConversationSummary } from "../../lib/conversationLibrary";
  import type { HostedShellController, HostedShellSnapshot } from "./types";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import {
    chunkLoadFailureMessage,
    completeChunkRecovery,
    isChunkLoadError,
    recoverFromChunkLoadError,
    reloadHostedApp,
  } from "../runtime/chunkRecovery";
  import {
    contentIn,
    contentOut,
    pageTransition,
    modalIn,
    modalOut,
    scrimIn,
    scrimOut,
  } from "../../lib/motion";
  import {
    clearConversationOpen,
    queueConversationOpen,
    readEntryMemory,
    rememberCompletedOnboarding,
  } from "../runtime/entryExperience";

  type EntryView = "resolving" | "home" | "models";

  let { controller }: { controller?: HostedShellController } = $props();
  const appController = $derived(controller ?? createShellController());
  let Workbench = $state<Component | null>(null);
  let workbenchError = $state<string | null>(null);
  let workbenchNeedsReload = $state(false);
  let workbenchReloading = $state(false);
  let runtimeRecovery = $state<string | null>(null);
  let shellSnapshot = $state<HostedShellSnapshot | null>(null);
  let entryView: EntryView = $state("resolving");
  let firstRun = $state(true);
  let runtimeDialog: HTMLElement | null = $state(null);
  let dialogInitialFocus: HTMLButtonElement | null = $state(null);
  let allowingTakeover = $state(false);
  let preparingPwaUpdate = $state(false);
  let returningHome = $state(false);
  let disposed = false;
  let reloading = false;
  let focusedDialogKey: string | null = null;
  let previousFocus: HTMLElement | null = null;
  let restoreFocusOnClose = true;
  let workbenchPromise: Promise<void> | null = null;
  const initialUrl = new URL(window.location.href);
  const requestedModelPicker = initialUrl.searchParams.get("choose") === "1";
  const requestedWorkbenchReopen = initialUrl.searchParams.get("reopen") === "1";
  const takeoverPhase = $derived(shellSnapshot?.takeover?.phase ?? "idle");
  const takeoverReason = $derived(shellSnapshot?.takeover?.reason ?? null);
  const dialogKey = $derived(
    runtimeRecovery !== null
      ? "recovery"
      : takeoverPhase === "idle"
        ? null
        : takeoverPhase,
  );
  const dialogOpen = $derived(dialogKey !== null);
  const route = $derived(Workbench ? "workbench" : shellSnapshot === null ? "resolving" : entryView);
  let routeMounted = $state(false);

  const FOCUSABLE = [
    "button:not([disabled])",
    "[href]",
    "input:not([disabled])",
    "select:not([disabled])",
    "textarea:not([disabled])",
    '[tabindex]:not([tabindex="-1"])',
  ].join(",");

  $effect(() => {
    const nextKey = dialogKey;
    if (nextKey === focusedDialogKey) return;
    const wasOpen = focusedDialogKey !== null;
    focusedDialogKey = nextKey;
    if (nextKey !== null) {
      if (!wasOpen) {
        previousFocus = document.activeElement instanceof HTMLElement
          ? document.activeElement
          : null;
        restoreFocusOnClose = nextKey === "requested";
      }
      void tick().then(() => {
        if (!disposed && focusedDialogKey === nextKey) {
          (dialogInitialFocus ?? runtimeDialog)?.focus();
        }
      });
      return;
    }
    const target = previousFocus;
    previousFocus = null;
    if (wasOpen && restoreFocusOnClose) {
      queueMicrotask(() => {
        if (!disposed && focusedDialogKey === null && target?.isConnected) {
          target.focus();
        }
      });
    }
  });

  function reloadToOnboarding(chooseModels = false): void {
    if (disposed || reloading) return;
    reloading = true;
    const url = new URL("/app", window.location.href);
    const currentUrl = new URL(window.location.href);
    if (currentUrl.searchParams.get("fixture") === "1") {
      url.searchParams.set("fixture", "1");
      for (const key of ["fixtureSae", "fixtureSlow"]) {
        const value = currentUrl.searchParams.get(key);
        if (value !== null) url.searchParams.set(key, value);
      }
    }
    if (currentUrl.searchParams.get("choose") === "1") {
      url.searchParams.set("choose", "1");
    } else if (currentUrl.searchParams.get("reopen-after-unload") === "1") {
      url.searchParams.set("reopen", "1");
    } else if (chooseModels) {
      url.searchParams.set("choose", "1");
    }
    window.location.replace(url.href);
  }

  function prepareWorkbenchUrl(): void {
    const url = new URL(window.location.href);
    url.searchParams.delete("choose");
    url.searchParams.delete("model");
    url.searchParams.delete("reopen-after-unload");
    url.searchParams.set("reopen", "1");
    window.history.replaceState(window.history.state, "", url);
  }

  function showModels(
    modelVariantId?: string,
    conversation?: SavedConversationSummary,
  ): void {
    if (conversation) queueConversationOpen(conversation.id, conversation.modelId);
    else clearConversationOpen();
    const url = new URL(window.location.href);
    url.searchParams.delete("reopen");
    url.searchParams.delete("reopen-after-unload");
    url.searchParams.set("choose", "1");
    if (modelVariantId) url.searchParams.set("model", modelVariantId);
    else url.searchParams.delete("model");
    window.history.replaceState(window.history.state, "", url);
    firstRun = false;
    entryView = "models";
  }

  function showHome(): void {
    clearConversationOpen();
    const url = new URL(window.location.href);
    url.searchParams.delete("choose");
    url.searchParams.delete("model");
    url.searchParams.delete("reopen");
    url.searchParams.delete("reopen-after-unload");
    window.history.replaceState(window.history.state, "", url);
    entryView = "home";
  }

  function enterPage(event: Event): void {
    (event.currentTarget as HTMLElement).inert = false;
    if (!routeMounted) return;
    (event.currentTarget as HTMLElement).classList.add("page-entering");
    window.scrollTo({ top: 0, behavior: "instant" });
    const heading = (event.currentTarget as HTMLElement).querySelector("h1");
    if (heading) {
      heading.tabIndex = -1;
      heading.focus({ preventScroll: true });
    }
  }

  async function returnToChats(destination: "chats" | "models" = "chats"): Promise<void> {
    if (returningHome) return;
    returningHome = true;
    try {
      await appController.prepareForReload();
      await flushConversationAutosave();
      Workbench = null;
      if (destination === "models") showModels();
      else showHome();
      await tick();
      document.querySelector<HTMLElement>(destination === "models" ? ".app-shell h1" : ".home-shell h1")?.focus({ preventScroll: true });
    } finally {
      returningHome = false;
    }
  }

  async function resolveEntryView(): Promise<void> {
    const memory = readEntryMemory();
    let hasSavedChats = false;
    if (!memory && !requestedModelPicker && !requestedWorkbenchReopen) {
      try {
        hasSavedChats = await conversationLibrary.hasAny();
      } catch {
        hasSavedChats = false;
      }
    }
    await appController?.check();
    if (disposed || !appController) return;
    const checked = appController.current();
    const installedModel = checked.models.find((model) => model.setupComplete);
    const returning = memory !== null || hasSavedChats || installedModel !== undefined;
    if (!memory && installedModel) rememberCompletedOnboarding(installedModel.id);
    firstRun = !returning;
    entryView = requestedModelPicker || requestedWorkbenchReopen || !returning
      ? "models"
      : "home";
  }

  function denyTakeover(): void {
    if (takeoverPhase !== "requested") return;
    appController?.denyBusyTakeover();
  }

  async function allowTakeover(): Promise<void> {
    if (!appController || takeoverPhase !== "requested") return;
    allowingTakeover = true;
    restoreFocusOnClose = false;
    await appController.allowBusyTakeover();
    reloadToOnboarding(true);
  }

  async function prepareForPwaUpdate(): Promise<void> {
    await appController?.prepareForReload();
  }

  function onPwaApplyingChange(applying: boolean): void {
    preparingPwaUpdate = applying;
    if (
      !applying &&
      Workbench &&
      appController?.current().runtime.phase === "unloaded"
    ) {
      reloadToOnboarding();
    }
  }

  function onDialogKeydown(event: KeyboardEvent): void {
    event.stopPropagation();
    if (event.key === "Escape") {
      event.preventDefault();
      denyTakeover();
      return;
    }
    if (event.key !== "Tab" || !runtimeDialog) return;
    const focusable = [...runtimeDialog.querySelectorAll<HTMLElement>(FOCUSABLE)]
      .filter((element) => element.offsetParent !== null);
    if (focusable.length === 0) {
      event.preventDefault();
      runtimeDialog.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    if (
      event.shiftKey &&
      (active === first || active === runtimeDialog || !runtimeDialog.contains(active))
    ) {
      event.preventDefault();
      last.focus();
    } else if (
      !event.shiftKey &&
      (active === last || !runtimeDialog.contains(active))
    ) {
      event.preventDefault();
      first.focus();
    }
  }

  async function mountWorkbench(): Promise<void> {
    if (!appController || Workbench || workbenchPromise) return;
    const capabilities = appController.capabilities();
    if (!capabilities) return;
    workbenchPromise = (async () => {
      workbenchError = null;
      workbenchNeedsReload = false;
      try {
        installRuntimeClient(appController.runtimeClient());
        installRuntimeCapabilities(capabilities);
        installHostedController(appController.hostedController());
        const component = (await import("../../App.svelte")).default;
        if (!disposed) {
          if (appController.current().runtime.phase === "ready") {
            Workbench = component;
            completeChunkRecovery("workbench");
          } else {
            reloadToOnboarding();
          }
        }
      } catch (error) {
        if (
          await recoverFromChunkLoadError(
            error,
            "workbench",
            () => appController.prepareForReload(),
          )
        ) return;
        workbenchNeedsReload = isChunkLoadError(error);
        workbenchError = workbenchNeedsReload
          ? chunkLoadFailureMessage
          : userFacingError(
              error,
              "The workbench could not open. Reopen the model and try again.",
            );
      }
    })();
    try {
      await workbenchPromise;
    } finally {
      workbenchPromise = null;
    }
  }

  async function retryWorkbench(): Promise<void> {
    if (!workbenchNeedsReload) return mountWorkbench();
    if (workbenchReloading) return;
    workbenchReloading = true;
    try {
      await reloadHostedApp("workbench", async () => {
        await appController.prepareForReload();
        await flushConversationAutosave();
      });
    } catch (error) {
      workbenchError = userFacingError(error, "The app could not refresh. Please try again.");
      workbenchReloading = false;
    }
  }

  onMount(() => {
    if (!appController) return;
    const disposeTools = mountHostedWebMcp(appController);
    const disposeActions = registerPageActions({
      state: () => ({ route, recovery: runtimeRecovery ?? workbenchError }),
      home: () => Workbench ? returnToChats("chats") : showHome(),
      models: () => Workbench ? returnToChats("models") : showModels(),
      leave: async url => {
        returningHome = true;
        try {
          await appController.prepareForReload();
          await flushConversationAutosave();
          window.setTimeout(() => window.location.assign(url), 0);
        } catch (error) {
          returningHome = false;
          throw error;
        }
      },
      retry: retryWorkbench, allowTakeover, denyTakeover,
    });
    const unsubscribe = appController.subscribe((snapshot) => {
      shellSnapshot = snapshot;
      if (snapshot.runtime.phase === "ready") {
        const modelVariantId = snapshot.runtime.modelVariantId ?? snapshot.selectedModelVariantId;
        if (modelVariantId) rememberCompletedOnboarding(modelVariantId);
        prepareWorkbenchUrl();
        runtimeRecovery = null;
        void mountWorkbench();
      } else if (Workbench && snapshot.runtime.phase === "failed") {
        runtimeRecovery = snapshot.runtime.reason;
      } else if (
        Workbench &&
        snapshot.runtime.phase === "unloaded" &&
        !allowingTakeover &&
        !preparingPwaUpdate &&
        !returningHome
      ) {
        reloadToOnboarding(true);
      }
    });
    void resolveEntryView();
    return () => { unsubscribe(); disposeTools(); disposeActions(); };
  });

  onDestroy(() => {
    disposed = true;
    appController?.dispose();
  });
</script>


<div class="hosted-background" inert={dialogOpen}>
  <PwaUpdatePrompt
    showOfflineReady={!Workbench}
    onPrepareReload={prepareForPwaUpdate}
    onApplyingChange={onPwaApplyingChange}
  />

  <div class="route-stage">
  {#key route}
  <div class="page-route" data-route={route}
    in:pageTransition={{ animate: routeMounted }}
    out:pageTransition={{ animate: routeMounted }}
    onintrostart={enterPage}
    onintroend={(event) => {
      if (event.currentTarget.dataset.route !== "resolving") routeMounted = true;
      event.currentTarget.classList.remove("page-entering");
    }}
    onoutrostart={(event) => { event.currentTarget.inert = true; }}
  >
  {#if Workbench}
    <div class="workbench-route" inert={returningHome}>
      <Workbench onhome={() => returnToChats()} onmodels={() => returnToChats("models")} />
    </div>
  {:else if entryView === "resolving" || shellSnapshot === null}
    <TabIdentity state="loading" title="Opening workspace · Drowse" />
    <div class="entry-loading">
      <PageHeader />
      <main aria-busy="true" aria-live="polite">
        <p class="loading-pulse loading-placeholder">Opening your local workspace…</p>
      </main>
      <PageFooter />
    </div>
  {:else if entryView === "home"}
    <div class="home-route">
      <HostedHome
        controller={appController}
        snapshot={shellSnapshot}
        onChooseModels={showModels}
        {workbenchError}
        {workbenchNeedsReload}
        {workbenchReloading}
        onRetryWorkbench={() => void retryWorkbench()}
      />
    </div>
  {:else}
    <HostedApp
      controller={appController}
      {firstRun}
      onBack={firstRun ? undefined : showHome}
      {workbenchError}
      {workbenchNeedsReload}
      {workbenchReloading}
      onRetryWorkbench={() => void retryWorkbench()}
    />
  {/if}
  </div>
  {/key}
  </div>
</div>

{#if dialogOpen}
  <div
    class="runtime-dialog-backdrop"
    in:fade={scrimIn()}
    out:fade={scrimOut()}
  >
    <div
      bind:this={runtimeDialog}
      class="runtime-dialog"
      role="alertdialog"
      aria-modal="true"
      aria-labelledby="runtime-dialog-title"
      aria-describedby="runtime-dialog-description"
      aria-busy={takeoverPhase === "allowing"}
      tabindex="-1"
      onkeydown={onDialogKeydown}
      in:fly={modalIn(8)}
      out:fly={modalOut(4)}
      onintrostart={(event) => { event.currentTarget.inert = false; }}
      onoutrostart={(event) => { event.currentTarget.inert = true; }}
    >
      {#key dialogKey}
        <div
          class="runtime-dialog-content"
          in:fly={contentIn()}
          out:fly={contentOut()}
        >
          {#if runtimeRecovery}
            <h1 id="runtime-dialog-title">Reload Drowse to recover</h1>
            <div id="runtime-dialog-description">
              <p>{runtimeRecovery}</p>
              <p>Your installed model and saved conversation remain on this device.</p>
            </div>
            <div class="runtime-dialog-actions">
              <button
                bind:this={dialogInitialFocus}
                type="button"
                class="primary"
                onclick={() => reloadToOnboarding()}
              >Reload Drowse</button>
            </div>
          {:else if takeoverPhase === "requested"}
            <h1 id="runtime-dialog-title">Another tab wants to use the GPU</h1>
            <div id="runtime-dialog-description">
              <p>{takeoverReason}</p>
              <p>Stop the current task, unload the model, and move it to the other tab.</p>
            </div>
            <div class="runtime-dialog-actions">
              <button
                bind:this={dialogInitialFocus}
                type="button"
                class="secondary"
                onclick={denyTakeover}
              >Keep working here</button>
              <button
                type="button"
                class="primary"
                onclick={() => void allowTakeover()}
              >Release runtime</button>
            </div>
          {:else}
            <h1 id="runtime-dialog-title">Releasing the local runtime</h1>
            <div id="runtime-dialog-description" role="status" aria-live="polite">
              <p>{takeoverReason}</p>
              <p>Drowse is stopping the current task and unloading the model before returning to setup.</p>
            </div>
          {/if}
        </div>
      {/key}
    </div>
  </div>
{/if}

<style>
  .hosted-background {
    display: contents;
  }

  .route-stage { position: relative; display: grid; min-width: 0; }
  .route-stage > :global(*) { grid-area: 1 / 1; min-width: 0; }
  .page-route { min-width: 0; }
  .workbench-route, .home-route { min-width: 0; }

  .entry-loading {
    min-height: 100dvh;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    color: var(--fg);
    background: var(--ambient-canvas);
  }

  .entry-loading > main {
    min-height: 50dvh;
    display: grid;
    place-content: center;
    justify-items: center;
    gap: var(--space-5);
    color: var(--fg-dim);
  }

  .entry-loading p { margin: 0; }

  .runtime-dialog-backdrop {
    position: fixed;
    inset: 0;
    z-index: 10000;
    display: grid;
    place-items: center;
    padding:
      max(var(--surface-padding), env(safe-area-inset-top))
      max(var(--surface-padding), env(safe-area-inset-right))
      max(var(--surface-padding), env(safe-area-inset-bottom))
      max(var(--surface-padding), env(safe-area-inset-left));
    overflow-y: auto;
    overscroll-behavior: contain;
    background: var(--scrim-strong);
    backdrop-filter: blur(8px);
  }

  .runtime-dialog {
    width: min(100%, 32rem);
    padding: var(--surface-padding);
    border: 1px solid var(--popup-border);
    border-radius: var(--popup-radius);
    color: var(--fg);
    background: var(--surface-sheen), var(--popup-bg);
    box-shadow: var(--popup-shadow);
  }

  .runtime-dialog-content {
    min-width: 0;
  }

  .runtime-dialog h1 {
    margin: 0;
    font-size: var(--text-page-title);
  }

  .runtime-dialog p {
    color: var(--fg-strong);
    line-height: 1.55;
  }

  .runtime-dialog-actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-6);
    margin-top: var(--space-6);
  }

  .runtime-dialog button {
    min-height: 2.75rem;
    padding: var(--space-4) var(--space-6);
    border: 1px solid transparent;
    border-radius: var(--radius);
    font-family: var(--font-structure);
    font-size: inherit;
    font-weight: var(--weight-structure-bold);
    cursor: pointer;
  }

  .runtime-dialog button.primary {
    color: var(--text-on-accent);
    background: var(--accent);
  }

  .runtime-dialog button.secondary {
    border-color: var(--glass-line);
    color: var(--fg);
    background: var(--glass);
  }

  .runtime-dialog button:focus-visible,
  .runtime-dialog:focus-visible {
    outline: 3px solid var(--focus-ring);
    outline-offset: 3px;
  }

  @media (prefers-reduced-motion: no-preference) {
    .runtime-dialog button {
      transition: transform var(--dur-fast) var(--ease-out);
    }

    .runtime-dialog button:active {
      transform: scale(var(--press-scale));
    }
  }

  @media (max-width: 32rem) {
    .runtime-dialog-backdrop {
      align-items: end;
      padding:
        max(var(--surface-padding), env(safe-area-inset-top))
        max(var(--surface-padding), env(safe-area-inset-right))
        max(var(--surface-padding), env(safe-area-inset-bottom))
        max(var(--surface-padding), env(safe-area-inset-left));
    }

    .runtime-dialog-actions,
    .runtime-dialog button {
      width: 100%;
    }
  }
</style>
