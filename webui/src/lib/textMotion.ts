import { TextMorph } from "torph";

export type TextMotionOptions = {
  text: string;
  numbers: boolean;
  duration: number;
  disabled: boolean;
  identity?: string | number | null;
};

const work = new Map<Element, () => void>();
let frame = 0;

function schedule(element: Element, run: () => void) {
  work.set(element, run);
  if (frame) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    const jobs = [...work.values()];
    work.clear();
    for (const job of jobs) job();
  });
}

const subscribers = new Map<Element, { visible: boolean; update: () => void }>();
let intersection: IntersectionObserver | null = null;
let resize: ResizeObserver | null = null;
let reduced: MediaQueryList | null = null;

function refreshMotion() {
  for (const { update } of subscribers.values()) update();
}

function observe(element: Element, update: () => void) {
  if (!intersection) {
    reduced = matchMedia("(prefers-reduced-motion: reduce)");
    reduced.addEventListener("change", refreshMotion);
    document.addEventListener("visibilitychange", refreshMotion);
    document.addEventListener("selectionchange", refreshMotion);
    intersection = new IntersectionObserver(entries => {
      for (const entry of entries) {
        const subscriber = subscribers.get(entry.target);
        if (!subscriber) continue;
        subscriber.visible = entry.isIntersecting;
        subscriber.update();
      }
    });
    resize = new ResizeObserver(entries => {
      for (const entry of entries) subscribers.get(entry.target)?.update();
    });
  }
  subscribers.set(element, { visible: false, update });
  intersection.observe(element);
  resize!.observe(element);
  return () => {
    work.delete(element);
    subscribers.delete(element);
    intersection!.unobserve(element);
    resize!.unobserve(element);
    if (subscribers.size) return;
    intersection!.disconnect();
    resize!.disconnect();
    intersection = null;
    resize = null;
    reduced!.removeEventListener("change", refreshMotion);
    reduced = null;
    document.removeEventListener("visibilitychange", refreshMotion);
    document.removeEventListener("selectionchange", refreshMotion);
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
  };
}

// The real text owns layout, selection, and accessibility. Torph only paints
// a non-interactive overlay, so interrupted motion cannot change the value.
export function textMotion(node: HTMLElement, initial: TextMotionOptions) {
  const host = node.parentElement!;
  const source = host.querySelector<HTMLElement>(".morph-source")!;
  let options = initial;
  let morph: TextMorph | null = null;
  let config = "";
  let disposed = false;

  function clear() {
    morph?.destroy();
    morph = null;
    node.replaceChildren();
    node.removeAttribute("style");
    host.removeAttribute("data-morph-active");
  }

  function paint() {
    if (disposed) return;
    const selected = document.getSelection();
    const enabled = !options.disabled && !reduced?.matches && !document.hidden
      && subscribers.get(host)?.visible && !host.closest("[inert]")
      && (!selected || selected.isCollapsed) && options.text.length <= 100
      && !options.text.includes("\n");
    if (!enabled) { clear(); return; }
    const style = getComputedStyle(source);
    const line = Number.parseFloat(style.lineHeight) || Number.parseFloat(style.fontSize) * 1.4;
    if (source.getBoundingClientRect().height > line * 1.5) { clear(); return; }
    const numbers = options.numbers && !/[eE][+-]?\d/.test(options.text);
    const nextConfig = JSON.stringify([options.duration, numbers, options.identity, host.closest("[data-morph-snapshot]")?.getAttribute("data-morph-snapshot")]);
    if (!morph || config !== nextConfig) {
      clear();
      config = nextConfig;
      morph = new TextMorph({
        element: node,
        numbers,
        duration: options.duration,
        ease: "cubic-bezier(0.2, 0, 0, 1)",
        scale: false,
        respectReducedMotion: false,
        onAnimationComplete: () => requestAnimationFrame(() => {
          if (disposed) return;
          for (const animation of node.getAnimations({ subtree: true })) {
            if (animation.playState === "finished") animation.cancel();
          }
        }),
      });
    }
    morph.update(options.text);
    host.setAttribute("data-morph-active", "");
  }

  const queue = () => schedule(host, paint);
  const stop = observe(host, queue);
  void document.fonts.ready.then(() => { if (!disposed) queue(); });
  return {
    update(next: TextMotionOptions) { options = next; host.removeAttribute("data-morph-active"); queue(); },
    destroy() { disposed = true; stop(); clear(); },
  };
}
