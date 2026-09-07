import { tick } from "svelte";
import { migrateLegacyStorageItem } from "../hosted/runtime/brandMigration";

export type Theme = "light" | "dark";

export const THEME_STORAGE_KEY = "drowse.theme";

const themeColors: Record<Theme, string> = {
  light: "#f2f4f8",
  dark: "#0b0e17",
};

const listeners = new Set<(theme: Theme) => void>();
let activeTheme: Theme = "dark";
let initialized = false;
let themeTransition: ViewTransition | undefined;
let transitionRevision = 0;
let fallbackTimer: ReturnType<typeof setTimeout> | undefined;

function isTheme(value: string | null | undefined): value is Theme {
  return value === "light" || value === "dark";
}

function storedTheme(): Theme | null {
  try {
    const value = migrateLegacyStorageItem(window.localStorage, THEME_STORAGE_KEY);
    return isTheme(value) ? value : null;
  } catch {
    return null;
  }
}

function preferredTheme(): Theme {
  const fromDocument = document.documentElement.dataset.theme;
  if (isTheme(fromDocument)) return fromDocument;
  return storedTheme() ??
    (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
}

function applyTheme(theme: Theme): void {
  activeTheme = theme;
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
  document.querySelectorAll<HTMLMetaElement>('meta[name="theme-color"]').forEach((meta) => {
    meta.content = themeColors[theme];
  });
  for (const listener of listeners) listener(theme);
}

function changeTheme(theme: Theme): void {
  if (theme === activeTheme && !themeTransition) return;
  const revision = ++transitionRevision;
  themeTransition?.skipTransition();
  themeTransition = undefined;
  clearTimeout(fallbackTimer);
  const root = document.documentElement;
  delete root.dataset.themeTransition;

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches || document.hidden) {
    applyTheme(theme);
    return;
  }

  const cleanup = () => {
    if (revision !== transitionRevision) return;
    delete root.dataset.themeTransition;
    themeTransition = undefined;
  };

  if (typeof document.startViewTransition !== "function") {
    root.dataset.themeTransition = "fallback";
    const duration = Number.parseFloat(getComputedStyle(root).getPropertyValue("--dur-theme")) * 1000;
    applyTheme(theme);
    fallbackTimer = setTimeout(cleanup, duration);
    return;
  }

  root.dataset.themeTransition = "snapshot";
  themeTransition = document.startViewTransition(async () => {
    if (revision !== transitionRevision) return;
    applyTheme(theme);
    await tick();
  });
  // Skipping an in-flight transition rejects ready, but still applies its update.
  void themeTransition.ready.catch(() => {});
  void themeTransition.finished.then(cleanup, cleanup);
}

export function initializeTheme(): Theme {
  if (typeof document === "undefined") return activeTheme;
  if (initialized) return activeTheme;
  initialized = true;
  applyTheme(preferredTheme());

  const media = window.matchMedia("(prefers-color-scheme: light)");
  media.addEventListener("change", (event) => {
    if (storedTheme() === null) changeTheme(event.matches ? "light" : "dark");
  });
  window.addEventListener("storage", (event) => {
    if (event.key !== THEME_STORAGE_KEY) return;
    changeTheme(isTheme(event.newValue) ? event.newValue : (media.matches ? "light" : "dark"));
  });
  return activeTheme;
}

export function subscribeTheme(listener: (theme: Theme) => void): () => void {
  initializeTheme();
  listeners.add(listener);
  listener(activeTheme);
  return () => listeners.delete(listener);
}

export function setTheme(theme: Theme): void {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // The appearance still changes for this page when storage is unavailable.
  }
  changeTheme(theme);
}
