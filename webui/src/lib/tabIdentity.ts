import type { HostedShellSnapshot } from "../hosted/ui/types";
import { chatAccentPalette, isChatAccent } from "./chatAccent";
import { icon as tabArtwork, frameCount, frameInterval, animatedStates } from "./tab-icon-artwork.mjs";

export type TabState = "home" | "credits" | "chats" | "conversation" | "models" | "controls" | "chat-settings" | "tokens" | "comparison" | "checking" | "ready" | "unsupported" | "downloading" | "loading" | "working" | "thinking" | "analyzing" | "loom" | "loom-working" | "saving" | "training" | "paused" | "error" | "offline";
export type TabTheme = "light" | "dark";
export const TAB_FRAME_COUNT = frameCount;

export const tabLabels: Record<TabState, string> = {
  home: "", credits: "Credits", chats: "Your chats", models: "Choose a model", checking: "Checking this device",
  ready: "Device ready", unsupported: "Device not supported", downloading: "Downloading files",
  loading: "Loading model", working: "Generating reply", loom: "Loom",
  analyzing: "Computing insights", "loom-working": "Loom · Generating reply", paused: "Download paused", error: "Action needed",
  offline: "Download waiting for connection",
  conversation: "", controls: "Response settings", "chat-settings": "Chat settings", tokens: "Token details",
  comparison: "Compare replies", thinking: "Thinking", saving: "Saving chat", training: "Building model tools",
};

export function setupTabState(snapshot: HostedShellSnapshot): TabState {
  if (snapshot.phase === "checking") return "checking";
  if (snapshot.phase === "unsupported") return "unsupported";
  if (snapshot.phase === "failed" || snapshot.runtime.phase === "failed" || snapshot.download.phase === "failed") return "error";
  if (snapshot.runtime.phase === "loading") return "loading";
  if (snapshot.download.phase === "downloading") {
    if (snapshot.download.progress?.offline) return "offline";
    if (snapshot.download.progress?.stalled) return "paused";
    return "downloading";
  }
  if (snapshot.download.phase === "paused" || snapshot.download.phase === "cancelling") return "paused";
  if (snapshot.download.phase === "requesting_persistence") return "loading";
  if (snapshot.phase === "supported") return "models";
  return "models";
}

export function tabTitle(state: TabState): string {
  return !tabLabels[state] ? "Drowse" : `${tabLabels[state]} · Drowse`;
}

export function workbenchTabState(input: {
  boot: "loading" | "ready" | "failed";
  runtime: TabState | null;
  active: boolean;
  replay: boolean;
  thinking: boolean;
  view: "conversation" | "branches" | "controls";
  section: "response" | "model" | "chat";
  drawer: string | null;
  saveStatus: string;
}): TabState {
  if (input.boot === "failed" || input.saveStatus === "error") return "error";
  if (input.boot === "loading") return "loading";
  if (input.runtime) return input.runtime;
  if (input.active) {
    if (input.replay) return "analyzing";
    if (input.thinking) return "thinking";
    return input.view === "branches" ? "loom-working" : "working";
  }
  if (input.drawer === "token_drilldown" || input.drawer === "probe_inspector") return "tokens";
  if (input.drawer === "compare" || input.drawer === "node_compare" || input.drawer === "correlation") return "comparison";
  if (input.drawer === "load_conversation") return "chats";
  if (input.drawer === "save_conversation") return input.saveStatus === "saving" ? "saving" : "chat-settings";
  if (input.drawer === "manifold_builder" || input.drawer === "advanced_sampling" || input.drawer === "system_prompt") return "controls";
  if (input.view === "branches") return "loom";
  if (input.view === "controls") return input.section === "model" ? "models" : input.section === "chat" ? "chat-settings" : "controls";
  return "conversation";
}

export function faviconPath(state: TabState, frame = 0, theme: TabTheme = "light"): string {
  return `/icons/tab-${state}-${theme}${animatedStates.has(state) ? `-${((frame % TAB_FRAME_COUNT) + TAB_FRAME_COUNT) % TAB_FRAME_COUNT}` : ""}.png`;
}

let releaseActiveTabIcon: (() => void) | undefined;

export function createTabIcon() {
  releaseActiveTabIcon?.();
  let disposed = false;
  const existing = document.querySelector<HTMLLinkElement>('link[rel="icon"]');
  const link = existing ?? document.createElement("link");
  const previous = { href: link.getAttribute("href"), type: link.getAttribute("type"), sizes: link.getAttribute("sizes") };
  link.rel = "icon";
  link.type = "image/png";
  link.sizes.value = "96x96";
  if (!existing) document.head.append(link);
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  const systemTheme = matchMedia("(prefers-color-scheme: dark)");
  const theme = (): TabTheme => {
    const value = document.documentElement.dataset.theme;
    return value === "light" || value === "dark" ? value : systemTheme.matches ? "dark" : "light";
  };
  let state: TabState = "home";
  let timer: ReturnType<typeof setInterval> | undefined;
  let frame = 0;
  let revision = 0;
  const rendered = new Map<string, Promise<string>>();
  const coloredIcon = (color: string, appearance: TabTheme): Promise<string> => {
    const svg = tabArtwork(state, frame, appearance, false, color);
    const cached = rendered.get(svg);
    if (cached) return cached;
    const result = new Promise<string>((resolve, reject) => {
      const image = new Image();
      image.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = canvas.height = 96;
        const context = canvas.getContext("2d");
        if (!context) { reject(new Error("Favicon canvas unavailable")); return; }
        context.drawImage(image, 0, 0, 96, 96);
        resolve(canvas.toDataURL("image/png"));
      };
      image.onerror = () => reject(new Error("Favicon artwork unavailable"));
      image.src = `data:image/svg+xml,${encodeURIComponent(svg)}`;
    });
    if (rendered.size >= TAB_FRAME_COUNT * 3) rendered.delete(rendered.keys().next().value!);
    rendered.set(svg, result);
    return result;
  };
  const paint = () => {
    if (disposed) return;
    const ticket = ++revision;
    const appearance = theme();
    const selected = document.documentElement.dataset.chatAccent;
    const palette = chatAccentPalette(isChatAccent(selected) ? selected : "purple");
    const fallback = `${faviconPath(state, frame, appearance)}?v=fluent`;
    if (palette.id === "purple" && (!animatedStates.has(state) || motion.matches || document.hidden)) link.setAttribute("href", fallback);
    else void coloredIcon(palette[appearance], appearance).then(href => {
      if (!disposed && ticket === revision) link.setAttribute("href", href);
    }).catch(() => {
      if (!disposed && ticket === revision) link.setAttribute("href", fallback);
    });
    link.dataset.state = state;
    link.dataset.theme = appearance;
    link.dataset.accent = palette.id;
  };
  const refresh = () => {
    clearInterval(timer);
    timer = undefined;
    frame = 0;
    paint();
    if (!motion.matches && !document.hidden && faviconPath(state, 1) !== faviconPath(state)) {
      const started = performance.now();
      timer = setInterval(() => {
        const next = Math.floor((performance.now() - started) / frameInterval) % TAB_FRAME_COUNT;
        if (next !== frame) { frame = next; paint(); }
      }, frameInterval);
    }
  };
  motion.addEventListener("change", refresh);
  document.addEventListener("visibilitychange", refresh);
  systemTheme.addEventListener("change", paint);
  const themeObserver = new MutationObserver(paint);
  themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "data-chat-accent"] });
  const dispose = () => {
      if (disposed) return;
      disposed = true;
      clearInterval(timer);
      rendered.clear();
      motion.removeEventListener("change", refresh);
      document.removeEventListener("visibilitychange", refresh);
      systemTheme.removeEventListener("change", paint);
      themeObserver.disconnect();
      delete link.dataset.state;
      delete link.dataset.theme;
      delete link.dataset.accent;
      if (!existing) link.remove();
      else for (const key of ["href", "type", "sizes"] as const) {
        if (previous[key] === null) link.removeAttribute(key);
        else link.setAttribute(key, previous[key]);
      }
      if (releaseActiveTabIcon === dispose) releaseActiveTabIcon = undefined;
  };
  releaseActiveTabIcon = dispose;
  return {
    update(next: TabState) { if (!disposed && (next !== state || !link.dataset.state)) { state = next; refresh(); } },
    dispose,
  };
}
