<script lang="ts">
  import { onMount, tick } from "svelte";
  import MorphText from "./ui/MorphText.svelte";
  // Shared range slider — one consistent thumb / track across the whole
  // webui (sampling strip, steering strips, the steering picker).
  //
  // Deliberately dumb: it reports the raw value through ``oninput`` and
  // ``bind:value``; consumers own any snapping (e.g. the steering strip's
  // 0-detent).  The thumb tints to ``accent`` so a slider reads as the
  // same control wherever it appears.

  interface Props {
    /** Current value — bindable. */
    value: number;
    min?: number;
    max?: number;
    step?: number;
    disabled?: boolean;
    ariaLabel?: string;
    title?: string;
    displayValue?: string;
    /** Fired on every drag tick with the raw (un-snapped) value. */
    oninput?: (value: number) => void;
  }

  let {
    value = $bindable(),
    min = 0,
    max = 1,
    step = 0.01,
    disabled = false,
    ariaLabel,
    title,
    displayValue,
    oninput,
  }: Props = $props();

  function handle(ev: Event): void {
    const v = parseFloat((ev.currentTarget as HTMLInputElement).value);
    if (!Number.isFinite(v)) return;
    active = true;
    value = v;
    oninput?.(v);
  }

  let inputEl: HTMLInputElement;
  let bubble: HTMLSpanElement;
  let active = $state(false);
  const formattedValue = $derived(displayValue ?? value.toFixed(Math.min(6, (String(step).split(".")[1] ?? "").length)));

  function placeBubble() {
    if (!active || !inputEl || !bubble) return;
    const rect = inputEl.getBoundingClientRect();
    const fraction = max === min ? 0 : Math.max(0, Math.min(1, (value - min) / (max - min)));
    const x = rect.left + 10 + (getComputedStyle(inputEl).direction === "rtl" ? 1 - fraction : fraction) * (rect.width - 20);
    const half = bubble.getBoundingClientRect().width / 2;
    const viewport = window.visualViewport;
    const left = viewport?.offsetLeft ?? 0;
    const top = viewport?.offsetTop ?? 0;
    const width = viewport?.width ?? innerWidth;
    const height = viewport?.height ?? innerHeight;
    if (rect.bottom <= top || rect.top >= top + height) { bubble.hidePopover(); return; }
    if (!bubble.matches(":popover-open")) bubble.showPopover();
    bubble.style.left = `${Math.max(left + half + 8, Math.min(left + width - half - 8, x))}px`;
    bubble.style.top = `${Math.max(top + 8, Math.min(top + height - bubble.offsetHeight - 8, rect.top - bubble.offsetHeight + 2))}px`;
  }

  onMount(() => {
    window.addEventListener("resize", placeBubble);
    document.addEventListener("scroll", placeBubble, true);
    window.visualViewport?.addEventListener("resize", placeBubble);
    window.visualViewport?.addEventListener("scroll", placeBubble);
    return () => {
      window.removeEventListener("resize", placeBubble);
      document.removeEventListener("scroll", placeBubble, true);
      window.visualViewport?.removeEventListener("resize", placeBubble);
      window.visualViewport?.removeEventListener("scroll", placeBubble);
    };
  });

  $effect(() => {
    void value;
    void formattedValue;
    if (active && !disabled) {
      void tick().then(() => {
        if (!active || disabled || !bubble?.isConnected) return;
        bubble.showPopover();
        placeBubble();
      });
    } else bubble?.hidePopover();
  });

  let drag: { id: number; offset: number } | null = null;

  function move(event: PointerEvent): void {
    if (!drag || event.pointerId !== drag.id || disabled) return;
    const input = event.currentTarget as HTMLInputElement;
    const rect = input.getBoundingClientRect();
    const width = Math.max(1, rect.width - 20);
    const rtl = getComputedStyle(input).direction === "rtl";
    const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left - 10 - drag.offset) / width));
    const raw = min + (rtl ? 1 - fraction : fraction) * (max - min);
    input.value = String(Math.max(min, Math.min(max, min + Math.round((raw - min) / step) * step)));
    value = input.valueAsNumber;
    oninput?.(value);
  }

  function start(event: PointerEvent): void {
    if (disabled || event.button !== 0 || !event.isPrimary) return;
    event.preventDefault();
    const input = event.currentTarget as HTMLInputElement;
    const rect = input.getBoundingClientRect();
    const fraction = max === min ? 0 : (value - min) / (max - min);
    const center = rect.left + 10 + (getComputedStyle(input).direction === "rtl" ? 1 - fraction : fraction) * (rect.width - 20);
    const offset = event.clientX - center;
    active = true;
    drag = { id: event.pointerId, offset: Math.abs(offset) <= 12 ? offset : 0 };
    input.focus({ preventScroll: true });
    input.setPointerCapture(event.pointerId);
    move(event);
  }

  function end(event: PointerEvent): void {
    if (drag?.id !== event.pointerId) return;
    drag = null;
    active = false;
    const input = event.currentTarget as HTMLInputElement;
    if (input.hasPointerCapture(event.pointerId)) input.releasePointerCapture(event.pointerId);
  }
</script>

<input
  bind:this={inputEl}
  onfocus={() => active = true}
  onblur={() => active = false}
  class="sk-slider"
  type="range"
  {min}
  {max}
  {step}
  value={value}
  {disabled}
  {...{ "aria-description": (title) }}
  aria-label={ariaLabel}
  aria-valuetext={formattedValue}
  oninput={handle}
  onpointerdown={start}
  onpointermove={move}
  onpointerup={end}
  onpointercancel={end}
  onlostpointercapture={end}
/>

<span bind:this={bubble} class="slider-bubble" popover="manual" aria-hidden="true"><MorphText text={formattedValue} duration={120} /></span>

<style>
  .slider-bubble { position: fixed; inset: auto; margin: 0; transform: translateX(-50%); pointer-events: none; padding: var(--space-1) var(--space-2); border: 1px solid var(--glass-line); border-radius: var(--radius); background: var(--bg-deep); color: var(--fg); box-shadow: var(--shadow-control); font: var(--text-sm)/1.4 var(--font-mono); font-variant-numeric: tabular-nums; max-width: calc(100vw - 16px); }

  .sk-slider {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: var(--control-target);
    min-height: 44px;
    touch-action: none;
    -webkit-user-select: none;
    user-select: none;
    margin: 0;
    /* Borderless: the track is a recessed groove — fill only.  The thumb
     * keeps its --bg-deep cutout ring (a glyph stroke, not chrome). */
    background: transparent;
    border: 0;
    border-radius: var(--radius-pill);
    cursor: ew-resize;
  }
  .sk-slider::-webkit-slider-runnable-track {
    height: 4px;
    background: var(--data-track);
    border: 0;
    border-radius: var(--radius-pill);
  }
  .sk-slider::-moz-range-track {
    height: 4px;
    background: var(--data-track);
    border: 0;
    border-radius: var(--radius-pill);
  }
  .sk-slider:disabled {
    cursor: not-allowed;
    opacity: var(--disabled-opacity);
  }

  .sk-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    box-sizing: border-box;
    margin-top: -8px;
    border-radius: 50%;
    background: var(--accent);
    border: 1px solid var(--bg-deep);
    cursor: inherit;
    transition: transform var(--selection-dur) var(--selection-ease),
      box-shadow var(--selection-dur) var(--selection-ease);
  }
  .sk-slider::-moz-range-thumb {
    width: 20px;
    height: 20px;
    box-sizing: border-box;
    border-radius: 50%;
    background: var(--accent);
    border: 1px solid var(--bg-deep);
    cursor: inherit;
    transition: transform var(--selection-dur) var(--selection-ease),
      box-shadow var(--selection-dur) var(--selection-ease);
  }
  .sk-slider:hover:not(:disabled)::-webkit-slider-thumb,
  .sk-slider:active:not(:disabled)::-webkit-slider-thumb {
    transform: scale(1.05);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent);
  }
  .sk-slider:hover:not(:disabled)::-moz-range-thumb,
  .sk-slider:active:not(:disabled)::-moz-range-thumb {
    transform: scale(1.05);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent);
  }
  .sk-slider:disabled::-webkit-slider-thumb {
    background: var(--fg-muted);
  }
  .sk-slider:disabled::-moz-range-thumb {
    background: var(--fg-muted);
  }
  .sk-slider:focus-visible {
    outline: 2px solid var(--focus-ring);
    outline-offset: 0;
  }
</style>
