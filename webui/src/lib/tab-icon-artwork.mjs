export const animatedStates = new Set(["checking", "downloading", "loading", "working", "thinking", "analyzing", "loom-working", "saving", "training"]);
import { fluentPaths } from "./fluent-icons.mjs";

export const frameCount = 48;
export const frameInterval = 1000 / 24;
export const states = ["home", "credits", "chats", "conversation", "models", "controls", "chat-settings", "tokens", "comparison", "checking", "ready", "unsupported", "downloading", "loading", "working", "thinking", "analyzing", "loom", "loom-working", "saving", "training", "paused", "error", "offline"];

export function icon(state, frame = 0, theme = "light", maskable = false, accent) {
  const color = accent ?? (theme === "dark" ? "#c5b3ff" : "#5b3fbf");
  const ink = theme === "dark" ? "#141822" : "#ffffff";
  frame = ((frame % frameCount) + frameCount) % frameCount;
  const pulse = (0.75 + 0.25 * Math.cos(frame * Math.PI * 2 / frameCount)).toFixed(3);
  const shift = Math.sin(frame * Math.PI * 2 / frameCount) * 3;
  const paths = {
    checking: `<circle cx="27" cy="27" r="17"/><path d="m40 40 13 13"/><circle cx="27" cy="27" r="5" fill="currentColor" opacity="${pulse}" stroke="none"/>`,
    downloading: `<path d="M12 48v7h40v-7"/><g transform="translate(0 ${shift})"><path d="M32 9v30m-13-12 13 13 13-13" stroke-width="7"/></g>`,
    loading: `<rect x="11" y="22" width="42" height="20" rx="10"/><path d="M20 32h24" opacity="${pulse}" stroke-width="8"/>`,
    working: `<path d="M14 23h36M14 33h25M14 43h16"/><path d="M45 35v12" opacity="${pulse}"/>`,
    thinking: `<path d="M16 38a17 17 0 1 1 29-2l-4 5v5H24v-5ZM25 54h15"/><path d="M32 20v10" opacity="${pulse}"/>`,
    analyzing: `<path d="M14 48V30M32 48V14M50 48V23" stroke-width="8" opacity="${pulse}"/>`,
    "loom-working": `<path d="M10 32h17q7 0 7-10t10-10h10M27 32q7 0 7 10t10 10h10" stroke-dasharray="16 8" stroke-dashoffset="${-frame * 24 / frameCount}"/>`,
    saving: `<path d="M12 10h34l8 8v36H10V10zM20 10v16h23V10M21 54V38h22v16"/><path d="M27 45h10" opacity="${pulse}"/>`,
    training: `<path d="M10 53V11M10 53h45M19 43l11-11 10 5 13-21"/><circle cx="53" cy="16" r="4" fill="currentColor" opacity="${pulse}"/>`,
  };
  if (!animatedStates.has(state)) {
    const name = state === "chat-settings" ? "settings" : state;
    if (fluentPaths[name]) paths[state] = `<g data-fluent="${name}" transform="translate(4 4) scale(2.333333)" fill="currentColor" stroke="none">${fluentPaths[name].map(d => `<path d="${d}"/>`).join("")}</g>`;
  }
  if (!(state in paths)) throw new Error(`Unknown tab icon: ${state}`);
  const tile = maskable ? '<rect width="64" height="64" fill="var(--tile)"/>' : '<path data-squircle="true" d="M32 2C8 2 2 8 2 32s6 30 30 30 30-6 30-30S56 2 32 2Z" fill="var(--tile)"/>';
  return `<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 64 64" style="--tile:${color};--ink:${ink}">${tile}<g style="color:var(--ink);--well:var(--tile)" fill="none" stroke="currentColor" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" transform="translate(10 10) scale(.6875)">${paths[state]}</g></svg>`;
}
