<script lang="ts">
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
    oninput,
  }: Props = $props();

  function handle(ev: Event): void {
    const v = parseFloat((ev.currentTarget as HTMLInputElement).value);
    if (!Number.isFinite(v)) return;
    value = v;
    oninput?.(v);
  }

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
    drag = { id: event.pointerId, offset: Math.abs(offset) <= 12 ? offset : 0 };
    input.focus({ preventScroll: true });
    input.setPointerCapture(event.pointerId);
    move(event);
  }

  function end(event: PointerEvent): void {
    if (drag?.id !== event.pointerId) return;
    drag = null;
    const input = event.currentTarget as HTMLInputElement;
    if (input.hasPointerCapture(event.pointerId)) input.releasePointerCapture(event.pointerId);
  }
</script>

<input
  class="sk-slider"
  type="range"
  {min}
  {max}
  {step}
  value={value}
  {disabled}
  {title}
  aria-label={ariaLabel}
  oninput={handle}
  onpointerdown={start}
  onpointermove={move}
  onpointerup={end}
  onpointercancel={end}
  onlostpointercapture={end}
/>

<style>
  .sk-slider {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: var(--control-target);
    min-height: 44px;
    touch-action: none;
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
    opacity: 0.5;
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
