<script lang="ts">
  import FluentIcon from "./ui/FluentIcon.svelte";
  // Toast host — renders ``toasts.entries`` in the bottom-right corner.
  // Each toast auto-dismisses after its ``ttlMs`` fires; clicking the
  // ✕ dismisses early.  Toasts with ``ttlMs === null`` are sticky —
  // used by long-running async work (extract / clone) that drives a
  // single chip from kickoff to completion via ``updateToast``.

  import { onDestroy } from "svelte";
  import { cubicIn, cubicOut } from "svelte/easing";
  import { fly } from "svelte/transition";
  import { dismissToast, toasts } from "./stores.svelte";
  import { motionDuration } from "./motion";

  // Track which toast ids have an active timer so we don't re-schedule
  // dismissal every time the entries array reshuffles.  Sticky entries
  // (``ttlMs === null``) never enter the set and never get scheduled.
  interface ToastTimer {
    handle: ReturnType<typeof setTimeout> | null;
    remaining: number;
    startedAt: number;
    paused: boolean;
  }

  const timers = new Map<number, ToastTimer>();

  function finish(id: number): void {
    const timer = timers.get(id);
    if (timer && timer.handle !== null) clearTimeout(timer.handle);
    timers.delete(id);
    dismissToast(id);
  }

  function arm(id: number, timer: ToastTimer): void {
    if (timer.remaining <= 0) {
      finish(id);
      return;
    }
    timer.startedAt = performance.now();
    timer.paused = false;
    timer.handle = setTimeout(() => finish(id), timer.remaining);
  }

  function pause(id: number): void {
    const timer = timers.get(id);
    if (!timer || timer.paused || timer.handle === null) return;
    clearTimeout(timer.handle);
    timer.handle = null;
    timer.remaining = Math.max(0, timer.remaining - (performance.now() - timer.startedAt));
    timer.paused = true;
  }

  function resume(id: number): void {
    const timer = timers.get(id);
    if (!timer || !timer.paused) return;
    arm(id, timer);
  }

  $effect(() => {
    for (const t of toasts.entries) {
      if (t.ttlMs === null || timers.has(t.id)) continue;
      const timer: ToastTimer = {
        handle: null,
        remaining: t.ttlMs,
        startedAt: 0,
        paused: false,
      };
      timers.set(t.id, timer);
      arm(t.id, timer);
    }
    const visibleIds = new Set(toasts.entries.map((toast) => toast.id));
    for (const [id, timer] of timers) {
      if (visibleIds.has(id)) continue;
      if (timer.handle !== null) clearTimeout(timer.handle);
      timers.delete(id);
    }
  });

  onDestroy(() => {
    for (const timer of timers.values()) {
      if (timer.handle !== null) clearTimeout(timer.handle);
    }
    timers.clear();
  });
</script>

{#if toasts.entries.length > 0}
  <div class="toaster" role="region" aria-label="Notifications">
    {#each toasts.entries as t (t.id)}
      <div
        class="toast"
        class:warning={t.kind === "warning"}
        class:error={t.kind === "error"}
        role={t.kind === "error" ? "alert" : "status"}
        aria-live={t.kind === "error" ? "assertive" : "polite"}
        aria-atomic="true"
        in:fly={{ y: -8, duration: motionDuration(220), easing: cubicOut }}
        out:fly={{ y: -4, duration: motionDuration(140), easing: cubicIn }}
        onpointerenter={() => pause(t.id)}
        onpointerleave={() => resume(t.id)}
        onfocusin={() => pause(t.id)}
        onfocusout={(event) => {
          if (!(event.currentTarget as HTMLElement).contains(event.relatedTarget as Node | null)) {
            resume(t.id);
          }
        }}
      >
        <div class="body">
          <span class="msg">{t.message}</span>
          {#if t.detail}
            <span class="detail">{t.detail}</span>
          {/if}
        </div>
        <button
          type="button"
          class="dismiss"
          aria-label={`Dismiss notification: ${t.message}`}
          onclick={() => finish(t.id)}
        ><FluentIcon name="dismiss" /></button>
      </div>
    {/each}
  </div>
{/if}

<style>
  .toaster {
    position: fixed;
    inset-inline-end: calc(var(--space-6) + env(safe-area-inset-right, 0px));
    bottom: calc(var(--space-6) + env(safe-area-inset-bottom, 0px));
    display: flex;
    flex-direction: column;
    gap: var(--space-3);
    z-index: calc(var(--z-modal) + 10);
    max-width: min(32em, calc(100vw - 2 * var(--space-6)));
    pointer-events: none;
  }
  .toast {
    pointer-events: auto;
    background: var(--surface-sheen), var(--popup-bg);
    color: var(--fg);
    border: 1px solid var(--popup-border);
    border-radius: var(--popup-radius);
    padding: var(--surface-padding);
    font-family: var(--font-reading);
    font-size: var(--text);
    box-shadow: var(--popup-shadow);
    display: flex;
    align-items: flex-start;
    gap: var(--space-4);
    line-height: 1.5;
  }
  .toast.warning {
    color: var(--accent-yellow);
  }
  .toast.error {
    color: var(--error-text);
  }
  .toast.error .detail,
  .toast.error .dismiss {
    color: inherit;
  }
  @media (prefers-contrast: more) {
    .toast.error {
      background: var(--bg);
      border-color: currentColor;
    }
  }
  @media (forced-colors: active) {
    .toast.error {
      background: Canvas;
      color: CanvasText;
      border-color: CanvasText;
    }
  }
  .body {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    gap: var(--space-1);
    min-width: 0;
  }
  .msg {
    word-break: break-word;
  }
  .detail {
    color: var(--fg-dim);
    font-size: var(--text-sm);
    word-break: break-word;
  }
  .dismiss {
    background: transparent;
    border: 0;
    border-radius: var(--radius);
    color: var(--fg-dim);
    cursor: pointer;
    min-width: 44px;
    min-height: 44px;
    padding: 0 var(--space-2);
    font: inherit;
    font-family: var(--font-mono);
    transition:
      color var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }
  .dismiss:hover {
    color: var(--accent-red);
  }
  .dismiss:active { transform: scale(var(--press-scale)); }

  @media (prefers-reduced-motion: reduce) {
    .dismiss { transition: none; }
  }
</style>
