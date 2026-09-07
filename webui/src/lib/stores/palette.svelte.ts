// Tool-directory state shared by the app shell and workspace menu.

export const paletteState: { open: boolean } = $state({ open: false });

let opener: HTMLElement | null = null;

export function openPalette(): void {
  if (
    !paletteState.open &&
    typeof document !== "undefined" &&
    document.activeElement instanceof HTMLElement
  ) {
    opener = document.activeElement;
  }
  paletteState.open = true;
}

export function closePalette(): void {
  const restore = opener;
  opener = null;
  paletteState.open = false;
  queueMicrotask(() => {
    if (restore?.isConnected) restore.focus();
  });
}
