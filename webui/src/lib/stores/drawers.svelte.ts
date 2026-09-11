import type { DrawerName, DrawerState } from "../types";
import { drawerAvailability } from "../runtime/ui-capabilities";
import { pushToast } from "./toasts.svelte";

export const drawerState: DrawerState = $state({
  open: null,
  params: null,
});

export const toolPresentation = $state({ mode: "dialog" as "dialog" | "sidebar", expanded: false });

export function setToolPresentation(mode: "dialog" | "sidebar"): void {
  toolPresentation.mode = mode;
  try { localStorage.setItem("drowse.tools.presentation", mode); } catch { /* Presentation still works without storage. */ }
}

export function restoreToolPresentation(): void {
  try {
    const mode = localStorage.getItem("drowse.tools.presentation");
    if (mode === "dialog" || mode === "sidebar") toolPresentation.mode = mode;
  } catch { /* Use the default when browser storage is unavailable. */ }
}

export function openToolInSidebar(name: DrawerName, params: unknown = null): void {
  setToolPresentation("sidebar");
  openDrawer(name, params);
}

let opener: HTMLElement | null = null;

export const tokenInspectorUi = $state({ docked: false, visible: false, params: null as unknown });

export function dockTokenDetails(params?: unknown): void {
  if (drawerState.open === "token_drilldown") {
    tokenInspectorUi.params = params ?? drawerState.params;
    closeDrawer();
  }
  tokenInspectorUi.docked = true;
  tokenInspectorUi.visible = true;
}

export function hideTokenDetails(): void {
  tokenInspectorUi.visible = false;
  if (document.getElementById("workspace-token-sidebar")?.contains(document.activeElement)) {
    document.querySelector<HTMLElement>('[aria-controls="workspace-token-sidebar"]')?.focus({ preventScroll: true });
  }
}

export function undockTokenDetails(params: unknown): void {
  hideTokenDetails();
  tokenInspectorUi.docked = false;
  openDrawer("token_drilldown", params);
}

export function openDrawer(name: DrawerName, params: unknown = null): void {
  const availability = drawerAvailability(name);
  if (!availability.available) {
    pushToast(availability.reason ?? "This tool is unavailable.", {
      kind: "warning",
    });
    return;
  }
  if (name === "token_drilldown" && tokenInspectorUi.docked) {
    tokenInspectorUi.params = params;
    tokenInspectorUi.visible = true;
    return;
  }
  if (
    drawerState.open === null &&
    typeof document !== "undefined" &&
    document.activeElement instanceof HTMLElement
  ) {
    opener = document.activeElement;
  }
  drawerState.open = name;
  drawerState.params = params;
}

export function closeDrawer(): void {
  const restore = opener;
  opener = null;
  drawerState.open = null;
  drawerState.params = null;
  queueMicrotask(() => {
    if (restore?.isConnected) restore.focus();
  });
}
