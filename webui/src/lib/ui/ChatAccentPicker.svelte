<script lang="ts">
  import MorphText from "./MorphText.svelte";
  import { onMount, tick } from "svelte";
  import FluentIcon from "./FluentIcon.svelte";
  import { CHAT_ACCENTS, chatAccentPalette, type ChatAccent } from "../chatAccent";
  import { dropdownMotion } from "../dropdownMotion.svelte";

  let { value = "purple", disabled = false, onchange }: {
    value?: ChatAccent;
    disabled?: boolean;
    onchange: (value: ChatAccent) => void | Promise<void>;
  } = $props();
  const id = $props.id();
  const selected = $derived(chatAccentPalette(value));
  let choice: ChatAccent = $state("purple");
  let open = $state(false);
  let trigger: HTMLButtonElement | null = $state(null);
  let popover: HTMLDivElement | null = $state(null);
  let surface: HTMLDivElement | null = $state(null);
  let placement = $state("");
  let tabTimer: ReturnType<typeof setTimeout> | undefined;
  const presence = dropdownMotion();
  $effect(() => { choice = value; });

  function place() {
    if (!open || !trigger || !popover) return;
    const rect = trigger.getBoundingClientRect();
    const viewport = window.visualViewport;
    const x = viewport?.offsetLeft ?? 0;
    const y = viewport?.offsetTop ?? 0;
    const width = viewport?.width ?? window.innerWidth;
    const bottom = y + (viewport?.height ?? window.innerHeight);
    const popupWidth = Math.min(432, width - 16);
    const left = Math.max(x + 8, Math.min(rect.left, x + width - popupWidth - 8));
    const height = surface?.offsetHeight ?? 0;
    const flip = rect.bottom + 8 + height > bottom - 8 && rect.top - height - 8 >= y + 8;
    const top = flip ? rect.top - height - 8 : rect.bottom + 8;
    const bounds = popover.getBoundingClientRect();
    const styles = getComputedStyle(popover);
    // Safari can pan the visual viewport independently of top-layer coordinates.
    const offsetX = parseFloat(styles.left) - bounds.left;
    const offsetY = parseFloat(styles.top) - bounds.top;
    placement = `left:${left + offsetX}px;top:${top + offsetY}px;width:${popupWidth}px;max-height:${Math.max(0, bottom - top - 8)}px`;
    if (surface) surface.dataset.origin = flip ? "bottom-left" : "top-left";
  }

  async function show(focusChoice: boolean) {
    if (disabled) return;
    clearTimeout(tabTimer);
    open = true;
    presence.mount();
    await tick();
    if (!open || !popover) return;
    popover.showPopover();
    place();
    await tick();
    if (!open || !popover || !surface) return;
    presence.show(surface);
    if (focusChoice) popover.querySelector<HTMLInputElement>("input:checked")?.focus({ preventScroll: true });
  }

  function close(restoreFocus = false) {
    open = false;
    if (restoreFocus) trigger?.focus({ preventScroll: true });
    presence.close(surface);
  }

  function outside(event: Event) {
    if (!open) return;
    const target = event.target as Node;
    if (!trigger?.contains(target) && !popover?.contains(target)) close();
  }

  function onKeydown(event: KeyboardEvent) {
    if (!open) return;
    if (event.key === "Tab" && !trigger?.contains(event.target as Node)) {
      tabTimer = setTimeout(() => close(), 0);
      return;
    }
    if (event.key !== "Escape") return;
    event.preventDefault();
    event.stopPropagation();
    close(true);
  }

  onMount(() => {
    document.addEventListener("pointerdown", outside, true);
    document.addEventListener("focusin", outside);
    document.addEventListener("keydown", onKeydown, true);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    window.visualViewport?.addEventListener("resize", place);
    window.visualViewport?.addEventListener("scroll", place);
    return () => {
      open = false;
      clearTimeout(tabTimer);
      document.removeEventListener("pointerdown", outside, true);
      document.removeEventListener("focusin", outside);
      document.removeEventListener("keydown", onKeydown, true);
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
      window.visualViewport?.removeEventListener("resize", place);
      window.visualViewport?.removeEventListener("scroll", place);
      presence.destroy();
    };
  });

  async function choose(accent: ChatAccent): Promise<void> {
    choice = accent;
    await onchange(accent);
    choice = value;
  }
</script>

<div class="accent-picker">
  <button bind:this={trigger} type="button" {disabled} aria-haspopup="dialog" aria-expanded={open} aria-controls={presence.mounted ? `${id}-colors` : undefined} onclick={(event) => open ? close() : void show(event.detail === 0)}>
    <span class="current-swatch" style:background={selected.dark} aria-hidden="true"></span>
    <span>Color <span class="color-name">· <MorphText text={selected.name} numbers={false} /></span></span>
    <span class="chevron"><FluentIcon name="down" size={16} /></span>
  </button>
  {#if presence.mounted}
    <div bind:this={popover} id={`${id}-colors`} class="accent-popover" popover="manual" role="dialog" aria-label="Chat accent color" style={placement} tabindex="-1" inert={!open}>
      <div bind:this={surface} class="accent-surface t-dropdown">
        <fieldset {disabled}>
          <legend class="sr-only">Chat accent color</legend>
          {#each CHAT_ACCENTS as accent}
            <label class="color-option">
              <input type="radio" name={`chat-accent-${id}`} value={accent.id} checked={choice === accent.id} onchange={() => void choose(accent.id)} />
              <span class="swatch" style:background={accent.dark}>
                <span class="swatch-check" class:chosen={choice === accent.id}><FluentIcon name="check" size={16} /></span>
              </span>
              <span>{accent.name}</span>
            </label>
          {/each}
        </fieldset>
      </div>
    </div>
  {/if}
</div>

<style>
  .accent-picker { min-width: 0; max-width: 27rem; }
  button { display: flex; align-items: center; gap: var(--control-label-gap); width: fit-content; max-width: 100%; box-sizing: border-box; min-height: var(--control-compact); padding: var(--space-3) var(--space-4); border: 0; border-radius: var(--radius); cursor: pointer; color: var(--fg); background: var(--control-sheen), var(--glass-strong); box-shadow: var(--shadow-control); font-size: var(--text-sm); }
  button:hover { background: var(--control-sheen), var(--glass-bright); box-shadow: var(--shadow-control-hover); }
  button { transition: background var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out); }
  .color-name { color: var(--fg-muted); }
  .accent-surface { --popup-radius: calc(var(--radius) + var(--space-2)); }
  .swatch-check { display: grid; place-items: center; opacity: 0; scale: 0.25; filter: blur(4px); transition: opacity var(--dur-slow) cubic-bezier(0.2, 0, 0, 1), scale var(--dur-slow) cubic-bezier(0.2, 0, 0, 1), filter var(--dur-slow) cubic-bezier(0.2, 0, 0, 1); }
  .swatch-check.chosen { opacity: 1; scale: 1; filter: blur(0); }
  .current-swatch { width: 14px; height: 14px; flex-shrink: 0; border-radius: 50%; box-shadow: var(--pastel-swatch-rim); }
  .chevron { transition: rotate var(--dur-fast) var(--ease-out); }
  button[aria-expanded="true"] .chevron { rotate: 180deg; }
  .accent-popover { position: fixed; inset: auto; margin: 0; width: min(432px, calc(100vw - 16px)); padding: 0; border: 0; background: transparent; overflow: visible; pointer-events: none; z-index: var(--z-modal); }
  .accent-surface { max-height: inherit; padding: var(--space-2); box-sizing: border-box; overflow: auto; border: 1px solid var(--popup-border); border-radius: var(--popup-radius); background: var(--surface-sheen), var(--popup-bg); color: var(--fg); box-shadow: var(--popup-shadow); }
  fieldset { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: var(--space-xs); min-width: 0; margin: 0; padding: 0; border: 0; }
  .color-option { position: relative; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: var(--space-xs); min-height: 60px; padding: var(--space-xs); border-radius: var(--radius); color: var(--fg-muted); font-size: var(--text-xs); cursor: pointer; }
  .color-option input { position: absolute; opacity: 0; width: 1px; height: 1px; }
  .color-option:has(:checked) { color: var(--fg); background: var(--surface-sheen), var(--glass); }
  .color-option:has(:focus-visible) { outline: 2px solid var(--focus-ring); outline-offset: 2px; }
  .color-option:hover { background: var(--surface-sheen), var(--glass-bright); }
  .color-option:has(:disabled) { cursor: wait; }
  @media (max-width: 380px) { .color-option { font-size: var(--text-2xs); } }
  .swatch { display: grid; place-items: center; width: 28px; height: 28px; border-radius: 50%; color: var(--pastel-swatch-ink); box-shadow: var(--pastel-swatch-rim); transition: transform var(--dur-fast) var(--ease-out); }
  .color-option:hover .swatch { transform: translateY(-2px); }
  .color-option:active .swatch { transform: scale(0.96); }
  @media (prefers-reduced-motion: reduce) { button, .swatch-check, .chevron, .swatch { transition: none; } .color-option:hover .swatch, .color-option:active .swatch { transform: none; } }
</style>
