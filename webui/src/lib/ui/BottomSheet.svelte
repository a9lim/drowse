<script lang="ts">
  import { onMount, tick, untrack, type Snippet } from "svelte";
  import { SHEET_STOPS, canScrollSheetContent, nearestSheetStop, releaseSheet, rubberBandSheet, sampleSheetVelocity, sheetPositions, sheetSpring, sheetSpringVelocity, type SheetStop, type VelocitySample } from "../bottomSheet";

  let { enabled, open, onclose, children }: { enabled: boolean; open: boolean; onclose: () => void; children: Snippet<[() => void]> } = $props();
  let surface: HTMLDivElement;
  let handle = $state<HTMLButtonElement>();
  let present = $state(false);
  let y = $state(0);
  let viewportHeight = $state(0);
  let viewportTop = $state(0);
  let safeTop = $state(0);
  let stop = $state<SheetStop>("half");
  let dragging = $state(false);
  let reduceMotion = false;
  let frame = 0;
  let motionVelocity = 0;
  let closing = false;
  let trigger: HTMLElement | null = null;
  let suppressClick = false;
  let gesture: { x: number; y: number; sheetY: number; start: SheetStop; target: Element; samples: VelocitySample[]; active: boolean; handle: boolean; pointer: number | null } | null = null;
  const positions = $derived(sheetPositions(viewportHeight, safeTop));
  const expansion = $derived(Math.max(0, Math.min(1, (viewportHeight - y) / Math.max(1, viewportHeight))));
  const FOCUSABLE = 'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

  function viewport() {
    viewportHeight = Math.min(window.innerHeight, window.visualViewport?.height ?? window.innerHeight);
    viewportTop = window.visualViewport?.offsetTop ?? 0;
    safeTop = parseFloat(getComputedStyle(surface).getPropertyValue("--sheet-safe-top")) || 0;
  }

  function animateTo(target: number, velocity = motionVelocity, done?: () => void) {
    cancelAnimationFrame(frame);
    if (reduceMotion) { y = target; motionVelocity = 0; done?.(); return; }
    const from = y;
    const start = performance.now();
    const update = (time: number) => {
      const elapsed = (time - start) / 1000;
      y = sheetSpring(from, target, velocity, elapsed);
      motionVelocity = sheetSpringVelocity(from, target, velocity, elapsed);
      if (elapsed >= 0.3 || (Math.abs(y - target) < 0.35 && elapsed > 0.12)) {
        y = target;
        motionVelocity = 0;
        frame = 0;
        done?.();
      } else frame = requestAnimationFrame(update);
    };
    frame = requestAnimationFrame(update);
  }

  function snap(next: SheetStop, velocity = motionVelocity) {
    closing = false;
    stop = next;
    animateTo(positions[next], velocity);
  }

  function dismiss() {
    if (!enabled) { onclose(); return; }
    gesture = null;
    dragging = false;
    finishDismiss(motionVelocity);
  }

  function finishDismiss(velocity: number) {
    closing = true;
    animateTo(viewportHeight, velocity, onclose);
  }

  function restoreFocus() {
    const previous = trigger;
    trigger = null;
    void tick().then(() => {
      if (previous?.isConnected && !previous.closest('[inert]')) previous.focus({ preventScroll: true });
      else document.querySelector<HTMLElement>('[aria-controls="workspace-token-sidebar"]')?.focus({ preventScroll: true });
    });
  }

  $effect(() => {
    const active = enabled && open;
    untrack(() => {
      if (active) {
        viewport();
        trigger = document.activeElement instanceof HTMLElement && document.activeElement !== document.body ? document.activeElement : null;
        stop = "half";
        y = viewportHeight;
        present = true;
        void tick().then(() => {
          if (!enabled || !open) return;
          handle?.focus({ preventScroll: true });
          snap("half");
        });
      } else {
        const wasPresent = present;
        closing = false;
        cancelAnimationFrame(frame);
        motionVelocity = 0;
        gesture = null;
        dragging = false;
        present = false;
        if (wasPresent || trigger) restoreFocus();
      }
    });
  });

  function begin(x: number, touchY: number, target: Element, time: number, isHandle: boolean, pointer: number | null = null) {
    gesture = { x, y: touchY, sheetY: y, start: nearestSheetStop(y, positions), target, samples: [{ y: touchY, time }], active: false, handle: isHandle, pointer };
  }

  function move(x: number, touchY: number, time: number, event: Event) {
    const current = gesture;
    if (!current) return;
    const delta = touchY - current.y;
    if (!current.active) {
      if (Math.abs(x - current.x) > Math.abs(delta) && Math.abs(x - current.x) > 8) { gesture = null; return; }
      if (Math.abs(delta) < 8) return;
      if (!current.handle && canScrollSheetContent(current.target, surface, delta)) {
        current.y = touchY;
        current.samples = [{ y: touchY, time }];
        return;
      }
      current.active = true;
      closing = false;
      cancelAnimationFrame(frame);
      motionVelocity = 0;
      current.sheetY = y;
      dragging = true;
    }
    event.preventDefault();
    y = rubberBandSheet(current.sheetY + delta, positions);
    sampleSheetVelocity(current.samples, touchY, time);
  }

  function end(touchY: number, time: number) {
    const current = gesture;
    gesture = null;
    dragging = false;
    if (!current?.active) return;
    suppressClick = true;
    setTimeout(() => { suppressClick = false; }, 0);
    const velocity = sampleSheetVelocity(current.samples, touchY, time);
    const next = releaseSheet(current.start, y, touchY - current.y, velocity, positions, viewportHeight);
    if (next === "dismiss") finishDismiss(velocity);
    else snap(next, velocity);
  }

  function cancel() {
    if (gesture?.active) snap(nearestSheetStop(y, positions), 0);
    gesture = null;
    dragging = false;
  }

  function pointerDown(event: PointerEvent) {
    if (!handle || event.pointerType === "touch" || event.button !== 0) return;
    handle.setPointerCapture(event.pointerId);
    begin(event.clientX, event.clientY, handle, event.timeStamp, true, event.pointerId);
  }

  function handleKey(event: KeyboardEvent) {
    const index = SHEET_STOPS.indexOf(stop);
    const next = event.key === "ArrowUp" ? SHEET_STOPS[Math.max(0, index - 1)]
      : event.key === "ArrowDown" ? SHEET_STOPS[Math.min(2, index + 1)]
      : event.key === "Home" ? "full" : event.key === "End" ? "peek" : null;
    if (next) { event.preventDefault(); event.stopPropagation(); snap(next); }
  }

  function keydown(event: KeyboardEvent) {
    if (!enabled || !open || event.defaultPrevented) return;
    if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); dismiss(); return; }
    if (event.key !== "Tab") return;
    const items = [...surface.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(el => el.offsetParent !== null && !el.closest('[inert]'));
    const first = items[0];
    const last = items.at(-1);
    if ((event.shiftKey && document.activeElement === first) || (!event.shiftKey && document.activeElement === last)) {
      event.preventDefault();
      (event.shiftKey ? last : first)?.focus({ preventScroll: true });
    }
  }

  onMount(() => {
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const updateMotion = () => {
      reduceMotion = motion.matches;
      if (reduceMotion && enabled && open) {
        cancelAnimationFrame(frame);
        motionVelocity = 0;
        if (closing) { y = viewportHeight; onclose(); }
        else y = positions[stop];
      }
    };
    updateMotion();
    const resize = () => {
      if (!enabled || !open) return;
      viewport();
      cancelAnimationFrame(frame);
      motionVelocity = 0;
      gesture = null;
      dragging = false;
      if (closing) { y = viewportHeight; onclose(); return; }
      y = sheetPositions(viewportHeight, safeTop)[stop];
    };
    const startTouch = (event: TouchEvent) => {
      if (!enabled || !open) return;
      if (event.touches.length !== 1) { cancel(); return; }
      const target = event.target as Element;
      if (target.closest('input, textarea, select, [contenteditable="true"], [role="slider"], [role="listbox"], canvas, [data-sheet-no-drag]')) return;
      const touch = event.touches[0];
      begin(touch.clientX, touch.clientY, target, event.timeStamp, handle?.contains(target) ?? false);
    };
    const moveTouch = (event: TouchEvent) => {
      if (event.touches.length !== 1) { cancel(); return; }
      move(event.touches[0].clientX, event.touches[0].clientY, event.timeStamp, event);
    };
    const endTouch = (event: TouchEvent) => { if (event.changedTouches[0]) end(event.changedTouches[0].clientY, event.timeStamp); };
    const click = (event: MouseEvent) => { if (suppressClick) { event.preventDefault(); event.stopImmediatePropagation(); } };
    const focus = (event: FocusEvent) => {
      if (enabled && open && (event.target as Element).matches('input, textarea, [contenteditable="true"]')) snap("full");
    };
    surface.addEventListener("touchstart", startTouch, { passive: true });
    surface.addEventListener("touchmove", moveTouch, { passive: false });
    surface.addEventListener("touchend", endTouch);
    surface.addEventListener("touchcancel", cancel);
    surface.addEventListener("click", click, true);
    surface.addEventListener("focusin", focus);
    window.addEventListener("resize", resize);
    window.visualViewport?.addEventListener("resize", resize);
    window.visualViewport?.addEventListener("scroll", resize);
    motion.addEventListener("change", updateMotion);
    return () => {
      cancelAnimationFrame(frame);
      surface.removeEventListener("touchstart", startTouch);
      surface.removeEventListener("touchmove", moveTouch);
      surface.removeEventListener("touchend", endTouch);
      surface.removeEventListener("touchcancel", cancel);
      surface.removeEventListener("click", click, true);
      surface.removeEventListener("focusin", focus);
      window.removeEventListener("resize", resize);
      window.visualViewport?.removeEventListener("resize", resize);
      window.visualViewport?.removeEventListener("scroll", resize);
      motion.removeEventListener("change", updateMotion);
      if (present || trigger) restoreFocus();
    };
  });
</script>

{#if enabled && present}
  <div class="sheet-backdrop" aria-hidden="true" onclick={dismiss} style:opacity={0.45 + expansion * 0.3}></div>
{/if}
<div bind:this={surface} class="sheet-host" class:enabled class:dragging class:present
  data-mobile-sheet={enabled ? "true" : undefined} data-detent={enabled ? stop : undefined}
  role={enabled && present ? "dialog" : undefined} aria-modal={enabled && present ? "true" : undefined}
  aria-label={enabled && present ? "Generated word details" : undefined}
  onkeydown={keydown}
  style:--sheet-y={`${y}px`} style:--sheet-top={`${viewportTop}px`} style:--sheet-height={`${Math.max(0, viewportHeight - Math.max(0, y))}px`}>
  {#if enabled}
    <button bind:this={handle} type="button" class="sheet-handle" aria-label={`Resize word details: ${stop === "full" ? "expanded" : stop === "half" ? "middle" : "compact"}. Drag to resize. Arrow up and down move one step; Home expands; End makes compact.`}
      onkeydown={handleKey} onclick={() => { if (!suppressClick) snap(SHEET_STOPS[(SHEET_STOPS.indexOf(stop) + 2) % 3]); }}
      onpointerdown={pointerDown}
      onpointermove={event => { if (gesture?.pointer === event.pointerId) move(event.clientX, event.clientY, event.timeStamp, event); }}
      onpointerup={event => { if (gesture?.pointer === event.pointerId) { end(event.clientY, event.timeStamp); handle?.releasePointerCapture(event.pointerId); } }}
      onpointercancel={cancel} onlostpointercapture={() => { if (gesture?.pointer !== null) cancel(); }}>
      <span></span>
    </button>
  {/if}
  {@render children(dismiss)}
</div>

<style>
  .sheet-host { display: contents; }
  .sheet-host.enabled {
    --sheet-safe-top: env(safe-area-inset-top, 0px);
    position: fixed;
    inset: var(--sheet-top) 0 auto;
    height: var(--sheet-height);
    max-height: calc(100dvh - var(--sheet-y));
    padding-inline-start: env(safe-area-inset-left, 0px);
    padding-inline-end: env(safe-area-inset-right, 0px);
    padding-bottom: env(safe-area-inset-bottom, 0px);
    box-sizing: border-box;
    transform: translate3d(0, var(--sheet-y), 0);
    display: none;
    flex-direction: column;
    min-width: 0;
    overflow: hidden;
    z-index: calc(var(--z-drawer) + 1);
    border: 1px solid var(--popup-border);
    border-bottom: 0;
    border-radius: var(--popup-radius) var(--popup-radius) 0 0;
    background: var(--surface-sheen), var(--popup-bg);
    box-shadow: var(--popup-shadow);
  }
  .sheet-host.enabled.present { display: flex; }
  .sheet-host.enabled :global(.mobile-sheet-content) {
    position: relative;
    inset: auto;
    width: 100%;
    height: auto;
    flex: 1;
    min-height: 0;
    border: 0;
    border-radius: 0;
    box-shadow: none;
    background: transparent !important;
    backdrop-filter: none;
    transform: none !important;
    transition: none !important;
    opacity: 1;
    filter: none;
  }
  .sheet-handle {
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 44px;
    width: 100%;
    padding: 0;
    border: 0;
    border-radius: 0;
    background: transparent;
    touch-action: none;
    cursor: grab;
    user-select: none;
  }
  .sheet-handle span { width: 36px; height: 4px; border-radius: var(--radius); background: var(--fg-muted); opacity: 0.55; }
  .sheet-handle:focus-visible { outline-offset: -4px; }
  .dragging .sheet-handle { cursor: grabbing; }
  .sheet-backdrop { position: fixed; inset: 0; z-index: var(--z-drawer); background: var(--scrim-soft); backdrop-filter: blur(2px); -webkit-backdrop-filter: blur(2px); }
  @media (prefers-reduced-transparency: reduce), (prefers-contrast: more), (forced-colors: active) {
    .sheet-host.enabled { background: Canvas; }
    .sheet-backdrop { backdrop-filter: none; -webkit-backdrop-filter: none; }
  }
</style>
