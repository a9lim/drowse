// Focus containment for modal surfaces (drawer dialog, command palette, loom
// dialogs, bottom sheet): one definition of a tab stop and one wrap rule.

const CANDIDATES = "a[href], button, input, select, textarea, [tabindex]";

/** The visible, enabled, non-inert tab stops inside ``container``, in DOM order. */
export function tabStops(container: HTMLElement): HTMLElement[] {
  return [...container.querySelectorAll<HTMLElement>(CANDIDATES)].filter(element =>
    element.tabIndex >= 0 &&
    !(element as HTMLButtonElement).disabled &&
    element.offsetParent !== null &&
    !element.closest("[inert]"));
}

/** Keep Tab and Shift+Tab inside ``container``.  Focus wraps from the last
 *  stop to the first (and back); focus on the container itself, or anywhere
 *  outside its stops, enters at the near end; a container with no stops
 *  holds focus on itself.  Call from the container's keydown handler. */
export function containTab(event: KeyboardEvent, container: HTMLElement): void {
  if (event.key !== "Tab") return;
  const stops = tabStops(container);
  if (stops.length === 0) {
    event.preventDefault();
    container.focus({ preventScroll: true });
    return;
  }
  const index = stops.indexOf(document.activeElement as HTMLElement);
  if (event.shiftKey ? index <= 0 : index < 0 || index === stops.length - 1) {
    event.preventDefault();
    stops[event.shiftKey ? stops.length - 1 : 0].focus({ preventScroll: true });
  }
}
