<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { fly } from "svelte/transition";
  import { registerSW } from "virtual:pwa-register";
  import { userFacingError } from "../../lib/runtime/userFacingError";
  import { hostedNetworkIsOffline } from "../runtime/networkState";
  import { contentIn, contentOut } from "../../lib/motion";
  import {
    UPDATE_REMINDER_KEY, readUpdateReminder, markUpdateShown,
    deferUpdate, clearUpdateReminder,
  } from "../runtime/updateReminder";

  const UPDATE_CONTROL_TIMEOUT_MS = 10_000;
  const CONFIRMATION_MS = 5_000;

  let {
    onPrepareReload,
    onApplyingChange,
  }: {
    onPrepareReload?: () => Promise<void>;
    onApplyingChange?: (applying: boolean) => void;
  } = $props();

  let updateAvailable = $state(false);
  let offlineReady = $state(false);
  let applying = $state(false);
  let registrationError = $state<string | null>(null);
  let updateServiceWorker: ((reloadPage?: boolean) => Promise<void>) | null = null;
  let noticeEl: HTMLElement | null = $state(null);
  let returnFocus: HTMLElement | null = null;
  let registrationStarted = false;
  let waitingForOnline = false;
  let destroyed = false;
  let updatePending = false;
  let firstUpdateEntrance = $state(false);
  let confirmation = $state<string | null>(null);
  let reminderTimer: number | undefined;
  let confirmationTimer: number | undefined;

  onMount(() => {
    void initializeServiceWorker();
    window.addEventListener("storage", syncReminderStorage);
    window.addEventListener("focus", refreshUpdateNotice);
    document.addEventListener("visibilitychange", refreshUpdateNotice);
    navigator.serviceWorker?.addEventListener("controllerchange", updateControlled);
  });

  onDestroy(() => {
    destroyed = true;
    stopWaitingForOnline();
    window.clearTimeout(reminderTimer);
    window.clearTimeout(confirmationTimer);
    window.removeEventListener("storage", syncReminderStorage);
    window.removeEventListener("focus", refreshUpdateNotice);
    document.removeEventListener("visibilitychange", refreshUpdateNotice);
    navigator.serviceWorker?.removeEventListener("controllerchange", updateControlled);
  });

  function refreshUpdateNotice(): void {
    window.clearTimeout(reminderTimer);
    if (destroyed || !updatePending || applying) return;
    const reminder = readUpdateReminder();
    const remaining = reminder.remindAt - Date.now();
    if (remaining > 0) {
      updateAvailable = false;
      reminderTimer = window.setTimeout(refreshUpdateNotice, Math.min(remaining, 2_147_483_647));
    } else if (!document.hidden && !updateAvailable) {
      firstUpdateEntrance = !reminder.shown;
      markUpdateShown();
      confirmation = null;
      offlineReady = false;
      updateAvailable = true;
    }
  }

  function syncReminderStorage(event: StorageEvent): void {
    if (event.key === UPDATE_REMINDER_KEY || event.key === null) refreshUpdateNotice();
  }

  function updateControlled(): void {
    clearUpdateReminder();
    updatePending = false;
    updateAvailable = false;
    confirmation = null;
    window.clearTimeout(reminderTimer);
    window.clearTimeout(confirmationTimer);
  }

  function remindLater(): void {
    if (!updateAvailable || applying) return;
    confirmation = `Okay! We'll remind you in ${deferUpdate()}.`;
    offlineReady = false;
    refreshUpdateNotice();
    window.clearTimeout(confirmationTimer);
    confirmationTimer = window.setTimeout(() => { confirmation = null; }, CONFIRMATION_MS);
    restoreFocus();
  }

  async function initializeServiceWorker(): Promise<void> {
    if ("serviceWorker" in navigator && hostedNetworkIsOffline()) {
      const registration = await navigator.serviceWorker.getRegistration("/");
      if (destroyed) return;
      if (registration?.active && navigator.serviceWorker.controller) {
        offlineReady = true;
        waitForOnline();
        return;
      }
    }
    startUpdateMonitoring();
  }

  function waitForOnline(): void {
    if (destroyed || registrationStarted || waitingForOnline) return;
    waitingForOnline = true;
    window.addEventListener("online", resumeUpdateMonitoring);
    if (!hostedNetworkIsOffline()) resumeUpdateMonitoring();
  }

  function stopWaitingForOnline(): void {
    if (!waitingForOnline) return;
    waitingForOnline = false;
    window.removeEventListener("online", resumeUpdateMonitoring);
  }

  function resumeUpdateMonitoring(): void {
    stopWaitingForOnline();
    startUpdateMonitoring();
  }

  function startUpdateMonitoring(): void {
    if (destroyed || registrationStarted) return;
    registrationStarted = true;
    stopWaitingForOnline();
    updateServiceWorker = registerSW({
      immediate: true,
      onNeedRefresh: () => {
        if (destroyed) return;
        updatePending = true;
        offlineReady = false;
        refreshUpdateNotice();
      },
      onRegisteredSW: (_url, registration) => {
        if (destroyed || !registration) return;
        if (!registration.waiting && !registration.installing) clearUpdateReminder();
      },
      onNeedReload: () => undefined,
      onOfflineReady: () => {
        if (!destroyed && !updatePending) offlineReady = true;
      },
      onRegisterError: (error) => {
        registrationError = userFacingError(
          error,
          "Offline support could not start. Reload while connected and try again.",
        );
      },
    });
  }

  async function applyUpdate(): Promise<void> {
    if (!updateServiceWorker || applying) return;
    applying = true;
    try {
      onApplyingChange?.(true);
      await onPrepareReload?.();
      const controlling = waitForUpdatedController();
      await updateServiceWorker(false);
      await controlling;
      window.location.reload();
    } catch (error) {
      registrationError = userFacingError(
        error,
        "Unable to apply the update safely. Reload the page and try again.",
      );
      updateAvailable = false;
      applying = false;
      onApplyingChange?.(false);
    }
  }

  function waitForUpdatedController(): Promise<void> {
    return new Promise((resolve, reject) => {
      const previousController = navigator.serviceWorker.controller;
      const onControllerChange = () => {
        const nextController = navigator.serviceWorker.controller;
        if (!nextController || nextController === previousController) return;
        window.clearTimeout(timeout);
        navigator.serviceWorker.removeEventListener("controllerchange", onControllerChange);
        resolve();
      };
      const timeout = window.setTimeout(() => {
        navigator.serviceWorker.removeEventListener("controllerchange", onControllerChange);
        reject(new Error("The updated app could not take control. Try reloading this page."));
      }, UPDATE_CONTROL_TIMEOUT_MS);
      navigator.serviceWorker.addEventListener("controllerchange", onControllerChange);
    });
  }

  function rememberReturnFocus(event: FocusEvent): void {
    if (event.relatedTarget instanceof HTMLElement && !noticeEl?.contains(event.relatedTarget)) {
      returnFocus = event.relatedTarget;
    }
  }

  function dismissNotice(): void {
    offlineReady = false;
    registrationError = null;
    restoreFocus();
  }

  function restoreFocus(): void {
    const target = returnFocus;
    queueMicrotask(() => {
      if (target?.isConnected) {
        target.focus();
        return;
      }
      const main = document.querySelector<HTMLElement>("main");
      if (!main) return;
      const heroAction = main.querySelector<HTMLAnchorElement>(".hero .primary-action");
      if (heroAction) {
        heroAction.focus({ preventScroll: true });
        return;
      }
      main.tabIndex = -1;
      main.focus();
    });
  }
</script>

{#if updateAvailable || offlineReady || registrationError}
  <div
    bind:this={noticeEl}
    class="pwa-notice"
    class:first-update={updateAvailable && firstUpdateEntrance}
    class:passive={offlineReady && !updateAvailable && !registrationError}
    role={registrationError ? "alert" : "status"}
    aria-live={registrationError ? "assertive" : "polite"}
    aria-atomic="true"
    aria-busy={applying}
    onfocusin={rememberReturnFocus}
    in:fly={updateAvailable ? { duration: 0 } : contentIn(10)}
    out:fly={contentOut(6)}
  >
    <div>
      {#if updateAvailable}
        <strong>A Drowse update is ready.</strong>
        <span>Your installed models and conversations will stay on this device.</span>
      {:else if offlineReady}
        <strong>The interface is ready offline.</strong>
        <span>Once a model is installed, Drowse can reopen and generate offline.</span>
      {:else}
        <strong>Offline app setup failed.</strong>
        <span>{registrationError}</span>
      {/if}
    </div>
    <div class="pwa-actions">
      {#if updateAvailable}
        <button type="button" class="primary" disabled={applying} aria-busy={applying} onclick={() => void applyUpdate()}>
          {applying ? "Saving local work…" : "Update and reload"}
        </button>
        <button type="button" disabled={applying} onclick={remindLater}>
          Update later
        </button>
      {:else}
        <button
          type="button"
          onclick={dismissNotice}
        >Dismiss</button>
      {/if}
    </div>
  </div>
{/if}

{#if confirmation}
  <div
    class="pwa-notice pwa-confirmation"
    role="status"
    aria-live="polite"
    aria-atomic="true"
    style:--confirmation-duration={`${CONFIRMATION_MS}ms`}
    in:fly={contentIn(6)}
    out:fly={contentOut(4)}
  >
    <svg class="countdown" viewBox="0 0 24 24" aria-hidden="true">
      <circle class="countdown-track" cx="12" cy="12" r="9" />
      <circle class="countdown-remaining" cx="12" cy="12" r="9" pathLength="1" />
    </svg>
    <span>{confirmation}</span>
  </div>
{/if}

<style>
  .pwa-notice {
    position: fixed;
    z-index: calc(var(--z-drawer) - 1);
    inset-inline-end: calc(var(--space-6) + env(safe-area-inset-right, 0px));
    bottom: calc(var(--space-6) + env(safe-area-inset-bottom, 0px));
    display: flex;
    flex-direction: column;
    width: min(28rem, calc(100vw - var(--space-8) - env(safe-area-inset-left, 0px) - env(safe-area-inset-right, 0px)));
    max-height: calc(100dvh - var(--space-8) - env(safe-area-inset-top, 0px) - env(safe-area-inset-bottom, 0px));
    overflow-y: auto;
    gap: var(--space-6);
    padding: var(--surface-padding);
    border: 1px solid var(--popup-border);
    border-radius: var(--popup-radius);
    box-shadow: var(--popup-shadow);
    color: var(--fg);
    background: var(--surface-sheen), var(--popup-bg);
  }

  .pwa-notice div:first-child {
    min-width: 0;
    display: grid;
    gap: var(--space-1);
  }

  .pwa-notice span {
    color: var(--fg-strong);
    overflow-wrap: anywhere;
    line-height: 1.45;
  }

  .pwa-actions {
    display: flex;
    flex: 0 0 auto;
    gap: var(--space-6);
    flex-wrap: wrap;
    justify-content: flex-end;
  }

  button {
    white-space: nowrap;
    min-height: var(--control-target);
    padding: var(--space-2) var(--space-4);
    border: 1px solid var(--glass-line);
    border-radius: var(--radius);
    color: var(--fg);
    background: var(--control-sheen), var(--glass);
    box-shadow: var(--shadow-control);
    font-family: var(--font-structure);
    font-size: inherit;
    font-weight: var(--weight-structure);
    cursor: pointer;
  }

  button.primary {
    color: var(--action-ink);
    background: var(--control-sheen), var(--action-bg);
  }

  button:hover:not(:disabled) { background: var(--control-sheen), var(--glass-bright); box-shadow: var(--shadow-control-hover); }
  button.primary:hover:not(:disabled) { background: var(--control-sheen), var(--action-hover); }

  .pwa-confirmation {
    flex-direction: row;
    align-items: center;
    gap: var(--space-3);
    width: fit-content;
    max-width: calc(100vw - var(--space-8));
  }

  .countdown {
    flex: 0 0 auto;
    width: 24px;
    height: 24px;
    transform: rotate(-90deg);
    fill: none;
    stroke-width: 2.5;
  }

  .countdown-track { stroke: var(--glass-line); }
  .countdown-remaining {
    stroke: var(--accent);
    stroke-dasharray: 1;
    stroke-dashoffset: 0;
    animation: countdown var(--confirmation-duration) linear forwards;
  }

  @keyframes countdown { to { stroke-dashoffset: 1; } }

  @keyframes update-pop {
    from { opacity: 0; transform: translateY(14px) scale(0.96); }
    to { opacity: 1; transform: none; }
  }

  @keyframes update-glow {
    from { box-shadow: var(--popup-shadow), 0 0 44px 8px color-mix(in srgb, var(--accent) 35%, transparent); }
    to { box-shadow: var(--popup-shadow), 0 0 44px 8px transparent; }
  }

  button:disabled {
    cursor: wait;
    opacity: 0.6;
  }

  button:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 2px;
  }

  @media (prefers-reduced-motion: no-preference) {
    .first-update {
      animation: update-pop 360ms var(--ease-out), update-glow 4s ease-out;
    }
    button {
      transition: transform var(--dur-fast) var(--ease-out), background var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out);
    }

    button:active:not(:disabled) {
      transform: scale(var(--press-scale));
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .countdown-remaining { animation-timing-function: steps(5, end); }
  }

  @media (max-width: 42rem) {
    .pwa-actions button {
      flex: 1 1 auto;
    }

  }

  @media (forced-colors: active) {
    .first-update { animation: none; }
    .countdown-track { stroke: GrayText; }
    .countdown-remaining { stroke: CanvasText; }
    .pwa-notice {
      outline: 1px solid CanvasText;
    }
  }
</style>
