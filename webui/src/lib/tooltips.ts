// One delegated tooltip surface for the whole dashboard. Existing and future
// `title` attributes remain the authoring API, but while an element is
// hovered/focused we remove the attribute synchronously (suppressing the
// OS/browser bubble) and render its text through Drowse chrome instead.

const TOOLTIP_ID = "drowse-tooltip";
const SHOW_DELAY_MS = 260;
const VIEWPORT_MARGIN = 8;
const ANCHOR_GAP = 8;

export function installTooltipLayer(): () => void {
  const tooltip = document.createElement("div");
  tooltip.id = TOOLTIP_ID;
  tooltip.className = "drowse-tooltip";
  tooltip.setAttribute("role", "tooltip");
  tooltip.setAttribute("popover", "manual");
  tooltip.setAttribute("aria-hidden", "true");
  document.body.append(tooltip);

  let activeEl: HTMLElement | null = null;
  let activeTitle = "";
  let previousDescribedBy: string | null = null;
  let showTimer: ReturnType<typeof setTimeout> | null = null;
  let hideTimer: ReturnType<typeof setTimeout> | null = null;
  let keyboardFocus = false;

  function titledAncestor(target: EventTarget | null): HTMLElement | null {
    if (!(target instanceof Element)) return null;
    return target.closest<HTMLElement>("[title]");
  }

  function place(anchor: HTMLElement): void {
    if (activeEl !== anchor) return;
    const rect = anchor.getBoundingClientRect();
    const tip = tooltip.getBoundingClientRect();
    const roomAbove = rect.top - VIEWPORT_MARGIN;
    const roomBelow = window.innerHeight - rect.bottom - VIEWPORT_MARGIN;
    const below = roomAbove < tip.height + ANCHOR_GAP && roomBelow > roomAbove;
    const desiredX = rect.left + rect.width / 2;
    const x = Math.max(
      VIEWPORT_MARGIN + tip.width / 2,
      Math.min(window.innerWidth - VIEWPORT_MARGIN - tip.width / 2, desiredX),
    );
    const desiredY = below
      ? rect.bottom + ANCHOR_GAP
      : rect.top - ANCHOR_GAP - tip.height;
    const y = Math.max(
      VIEWPORT_MARGIN,
      Math.min(window.innerHeight - VIEWPORT_MARGIN - tip.height, desiredY),
    );
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
    tooltip.classList.toggle("below", below);
  }

  function restoreActiveTitle(): void {
    if (!activeEl) return;
    // A reactive update may have authored a newer title while the tooltip was
    // open. Prefer it, then restore the latest text after leaving so the DOM
    // keeps its ordinary accessibility/source contract between interactions.
    const updated = activeEl.getAttribute("title");
    if (updated !== null) activeTitle = updated;
    activeEl.setAttribute("title", activeTitle);
    if (previousDescribedBy === null) activeEl.removeAttribute("aria-describedby");
    else activeEl.setAttribute("aria-describedby", previousDescribedBy);
  }

  function hide(): void {
    if (showTimer !== null) clearTimeout(showTimer);
    showTimer = null;
    if (hideTimer !== null) clearTimeout(hideTimer);
    hideTimer = null;
    tooltip.hidePopover();
    tooltip.classList.remove("visible");
    tooltip.setAttribute("aria-hidden", "true");
    restoreActiveTitle();
    activeEl = null;
    activeTitle = "";
    previousDescribedBy = null;
  }

  function activate(anchor: HTMLElement, immediate: boolean): void {
    const authored = anchor.getAttribute("title") ?? "";
    if (!authored.trim() || activeEl === anchor) return;
    hide();
    activeEl = anchor;
    activeTitle = authored;
    previousDescribedBy = anchor.getAttribute("aria-describedby");
    const ids = new Set((previousDescribedBy ?? "").split(/\s+/).filter(Boolean));
    ids.add(TOOLTIP_ID);
    anchor.setAttribute("aria-describedby", [...ids].join(" "));
    // Removing in the pointer/focus event prevents a native tooltip from ever
    // reaching its display delay. It is restored by `hide()`.
    anchor.removeAttribute("title");
    tooltip.textContent = authored;
    const show = () => {
      if (activeEl !== anchor) return;
      tooltip.showPopover();
      tooltip.classList.add("visible");
      tooltip.setAttribute("aria-hidden", "false");
      place(anchor);
    };
    if (immediate) show();
    else showTimer = setTimeout(show, SHOW_DELAY_MS);
  }

  function onMouseOver(event: MouseEvent): void {
    if (!window.matchMedia("(hover: hover)").matches) return;
    if (event.target instanceof Node && activeEl?.contains(event.target)) {
      onTooltipEnter();
      return;
    }
    const anchor = titledAncestor(event.target);
    if (anchor) activate(anchor, false);
  }

  function onMouseOut(event: MouseEvent): void {
    if (!activeEl) return;
    const related = event.relatedTarget;
    if (related instanceof Node && (activeEl.contains(related) || tooltip.contains(related))) return;
    if (document.activeElement === activeEl) return;
    if (hideTimer !== null) clearTimeout(hideTimer);
    hideTimer = setTimeout(hide, 120);
  }

  function onTooltipEnter(): void {
    if (hideTimer !== null) clearTimeout(hideTimer);
    hideTimer = null;
  }

  function onTooltipLeave(): void {
    if (document.activeElement !== activeEl) hide();
  }

  function onFocusIn(event: FocusEvent): void {
    if (!keyboardFocus) return;
    const anchor = titledAncestor(event.target);
    if (anchor) activate(anchor, true);
  }

  function onKeyDown(event: KeyboardEvent): void {
    if (event.key === "Tab") keyboardFocus = true;
    if (event.key === "Escape" && activeEl) {
      hide();
      event.preventDefault();
      event.stopPropagation();
    }
  }

  function onPointerDown(): void {
    keyboardFocus = false;
    hide();
  }

  function onFocusOut(event: FocusEvent): void {
    if (!activeEl) return;
    const related = event.relatedTarget;
    if (related instanceof Node && activeEl.contains(related)) return;
    hide();
  }

  function onViewportChange(): void {
    if (activeEl && tooltip.classList.contains("visible")) place(activeEl);
  }

  document.addEventListener("mouseover", onMouseOver, true);
  tooltip.addEventListener("mouseenter", onTooltipEnter);
  tooltip.addEventListener("mouseleave", onTooltipLeave);
  document.addEventListener("mouseout", onMouseOut, true);
  document.addEventListener("focusin", onFocusIn, true);
  document.addEventListener("focusout", onFocusOut, true);
  document.addEventListener("keydown", onKeyDown, true);
  document.addEventListener("pointerdown", onPointerDown, true);
  window.addEventListener("resize", onViewportChange);
  window.addEventListener("scroll", onViewportChange, true);

  return () => {
    document.removeEventListener("mouseover", onMouseOver, true);
    document.removeEventListener("mouseout", onMouseOut, true);
    document.removeEventListener("focusin", onFocusIn, true);
    document.removeEventListener("focusout", onFocusOut, true);
    document.removeEventListener("keydown", onKeyDown, true);
    document.removeEventListener("pointerdown", onPointerDown, true);
    window.removeEventListener("resize", onViewportChange);
    window.removeEventListener("scroll", onViewportChange, true);
    hide();
    tooltip.remove();
  };
}
