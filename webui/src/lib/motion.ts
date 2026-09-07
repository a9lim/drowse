import { cubicIn, cubicOut } from "svelte/easing";
import type { TransitionConfig } from "svelte/transition";

const prefersReducedMotion = (): boolean =>
  typeof window !== "undefined"
  && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export function motionDuration(milliseconds: number): number {
  return prefersReducedMotion() ? 0 : milliseconds;
}

export function pageTransition(
  node: Element,
  { animate = true }: { animate?: boolean },
  { direction }: { direction: "in" | "out" | "both" },
): TransitionConfig {
  const leaving = direction === "out";
  const grouped = !leaving && node.querySelector("[data-page-group]") !== null;
  return {
    duration: animate ? motionDuration(leaving ? 180 : grouped ? 440 : 240) : 0,
    easing: leaving ? cubicIn : cubicOut,
    css: (t) => grouped ? "" : `opacity: ${t}; filter: blur(${(1 - t) * 4}px); ${leaving ? "position: absolute; inset: 0; overflow: clip; pointer-events: none;" : ""}`,
  };
}

export function scrimIn() {
  return { duration: motionDuration(160), easing: cubicOut };
}

export function scrimOut() {
  return { duration: motionDuration(110), easing: cubicIn };
}

export function panelIn(x = 18, y = 0) {
  return { x, y, duration: motionDuration(220), easing: cubicOut };
}

export function panelOut(x = 10, y = 0) {
  return { x, y, duration: motionDuration(150), easing: cubicIn };
}

export function modalIn(y = -8) {
  return { y, duration: motionDuration(200), easing: cubicOut };
}

export function modalOut(y = -4) {
  return { y, duration: motionDuration(130), easing: cubicIn };
}

export function contentIn(y = 4) {
  return { y, duration: motionDuration(170), easing: cubicOut };
}

export function contentOut(y = -2) {
  return { y, duration: motionDuration(100), easing: cubicIn };
}

export function selectionIn(node: Element): TransitionConfig {
  const styles = getComputedStyle(node);
  return {
    duration: motionDuration(Number.parseFloat(styles.getPropertyValue("--selection-dur")) * 1000),
    easing: cubicOut,
    css: (t) => `opacity: ${t}; transform: translateY(${(1 - t) * 4}px);`,
  };
}

export function collapseIn() {
  return { duration: motionDuration(200), easing: cubicOut };
}

export function collapseOut() {
  return { duration: motionDuration(130), easing: cubicIn };
}
