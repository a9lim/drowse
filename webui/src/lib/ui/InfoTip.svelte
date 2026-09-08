<script lang="ts">
  import FluentIcon from "./FluentIcon.svelte";
  import { onMount, tick } from "svelte";
  import { dropdownMotion } from "../dropdownMotion.svelte";

  let {
    text,
    label = "About this setting",
  }: {
    text: string;
    label?: string;
  } = $props();

  let open = $state(false);
  let trigger: HTMLButtonElement;
  let tip: HTMLSpanElement;
  let position = $state("");
  let disposed = false;
  let hovered = false;
  let touchInteraction = false;
  let hoverOpenTimer: ReturnType<typeof setTimeout> | undefined;
  let hoverCloseTimer: ReturnType<typeof setTimeout> | undefined;
  const id = $props.id();
  const presence = dropdownMotion();

  function enter(event: PointerEvent): void {
    if (event.pointerType === "touch") return;
    clearTimeout(hoverCloseTimer);
    hovered = true;
    clearTimeout(hoverOpenTimer);
    hoverOpenTimer = setTimeout(() => { open = true; }, 300);
  }

  function leave(event: PointerEvent): void {
    if (event.pointerType === "touch") return;
    hovered = false;
    clearTimeout(hoverOpenTimer);
    clearTimeout(hoverCloseTimer);
    hoverCloseTimer = setTimeout(() => {
      if (document.activeElement !== trigger || !trigger.matches(":focus-visible")) open = false;
    }, 150);
  }

  function place(): void {
    if (!open || !trigger || !tip) return;
    const viewport = window.visualViewport;
    const left = viewport?.offsetLeft ?? 0;
    const top = viewport?.offsetTop ?? 0;
    const width = viewport?.width ?? window.innerWidth;
    const height = viewport?.height ?? window.innerHeight;
    tip.style.maxWidth = `min(18rem, ${width - 16}px)`;
    tip.style.maxHeight = `${height - 16}px`;
    const anchor = trigger.getBoundingClientRect();
    const box = tip.getBoundingClientRect();
    const x = Math.max(left + 8, Math.min(anchor.left, left + width - box.width - 8));
    const below = anchor.bottom + 6;
    const y = below + box.height <= top + height - 8
      ? below : Math.max(top + 8, anchor.top - box.height - 6);
    position = `left:${x}px;top:${y}px;max-width:min(18rem, ${width - 16}px);max-height:${height - 16}px`;
  }

  $effect(() => {
    if (open) {
      presence.mount();
      void tick().then(() => {
        if (!open || disposed) return;
        tip.showPopover();
        place();
        presence.show(tip);
      });
    } else if (tip?.matches(":popover-open")) {
      presence.close(tip);
    }
  });

  $effect(() => {
    if (!presence.mounted) tip?.hidePopover();
  });

  onMount(() => {
    const outside = (event: PointerEvent) => {
      if (event.target instanceof Node && !trigger.contains(event.target) && !tip.contains(event.target)) {
        clearTimeout(hoverOpenTimer);
        open = false;
      }
    };
    document.addEventListener("pointerdown", outside, true);
    document.addEventListener("keydown", closeOnEscape, true);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    window.visualViewport?.addEventListener("resize", place);
    window.visualViewport?.addEventListener("scroll", place);
    return () => {
      disposed = true;
      clearTimeout(hoverOpenTimer);
      clearTimeout(hoverCloseTimer);
      presence.destroy();
      document.removeEventListener("pointerdown", outside, true);
      document.removeEventListener("keydown", closeOnEscape, true);
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
      window.visualViewport?.removeEventListener("resize", place);
      window.visualViewport?.removeEventListener("scroll", place);
    };
  });

  function closeOnEscape(event: KeyboardEvent): void {
    if (event.key !== "Escape") return;
    clearTimeout(hoverOpenTimer);
    if (!open) return;
    open = false;
    event.preventDefault();
    event.stopPropagation();
  }
</script>

<span
  class="info-tip"
  class:open
  role="presentation"
  onpointerenter={enter}
  onpointerleave={leave}
  onfocusout={(event) => {
    if (!hovered && !(event.currentTarget as HTMLElement).contains(event.relatedTarget as Node | null)) {
      open = false;
    }
  }}
>
  <button
    bind:this={trigger}
    type="button"
    class="info-trigger"
    aria-label={label}
    aria-describedby={id}
    aria-expanded={open}
    onfocus={() => { if (trigger.matches(":focus-visible")) { clearTimeout(hoverOpenTimer); open = true; } }}
    onpointerdown={(event) => { touchInteraction = event.pointerType === "touch"; }}
    onclick={(event) => { clearTimeout(hoverOpenTimer); clearTimeout(hoverCloseTimer); open = event.detail > 0 && touchInteraction ? !open : true; }}
  ><FluentIcon name="help" size={20} /></button>
  <span bind:this={tip} id={id} class="info-popover t-dropdown" role="tooltip" popover="manual" style={position}>{text}</span>
</span>

<style>
  .info-tip {
    position: relative;
    display: inline-flex;
    flex: none;
  }

  .info-trigger {
    cursor: help;
    position: relative;
    display: inline-grid;
    width: var(--control-target);
    height: var(--control-target);
    place-items: center;
    padding: 0;
    border: 0;
    border-radius: var(--radius);
    background: transparent;
    color: var(--fg-muted);
    font-family: var(--font-data) !important;
    font-size: var(--text-2xs);
    font-weight: var(--weight-data-bold) !important;
    line-height: 1;
    transition:
      color var(--dur-fast) var(--ease-out),
      border-color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }

  .info-trigger:hover,
  .open .info-trigger {
    color: var(--fg);
  }

  .info-trigger:hover,
  .open .info-trigger {
    background: var(--glass-strong);
  }

  .info-trigger:active {
    transform: scale(var(--press-scale));
  }

  .info-popover {
    --dropdown-pre-scale: 1;
    --dropdown-closing-scale: 1;
    position: fixed;
    z-index: calc(var(--z-modal) + 30);
    inset: auto;
    margin: 0;
    width: max-content;
    max-width: min(18rem, calc(100vw - 2rem));
    padding: calc(var(--space-sm) / 2) var(--space-sm);
    border: 0;
    border-radius: var(--popup-radius);
    background: var(--surface-sheen), var(--popup-bg);
    box-shadow: 0 0 0 1px var(--popup-border), var(--popup-shadow);
    color: var(--fg-strong);
    font-family: var(--font-reading);
    font-size: var(--text-sm);
    font-weight: var(--weight-reading);
    line-height: 1.45;
    text-align: start;
    text-wrap: pretty;
    overflow-wrap: anywhere;
    overflow: auto;
  }

  @media (prefers-reduced-motion: reduce) {
    .info-trigger,
    .info-popover {
      transition: none;
    }
  }
</style>
