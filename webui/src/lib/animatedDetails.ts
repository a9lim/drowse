import { motionDuration } from "./motion";

export function animatedDetails(node: HTMLDetailsElement) {
  const summary = node.querySelector("summary")!;
  let animation: Animation | null = null;
  let targetOpen = node.open;
  const originalOverflow = node.style.overflow;
  const originalSizing = node.style.boxSizing;

  function clear() {
    animation?.cancel();
    animation = null;
    node.style.overflow = originalOverflow;
    node.style.boxSizing = originalSizing;
    summary.removeAttribute("aria-expanded");
  }

  function toggle(event: MouseEvent) {
    if ((event.target as Element).closest("a, button, input")) return;
    event.preventDefault();
    const from = node.getBoundingClientRect().height;
    targetOpen = animation ? !targetOpen : !node.open;
    clear();
    node.style.boxSizing = "border-box";
    node.open = targetOpen;
    const to = node.getBoundingClientRect().height;
    const duration = motionDuration(targetOpen ? 250 : 150);
    if (!duration || from === to) { clear(); return; }
    node.open = true;
    summary.setAttribute("aria-expanded", String(targetOpen));
    node.style.overflow = "clip";
    animation = node.animate([{ height: `${from}px` }, { height: `${to}px` }], {
      duration,
      easing: targetOpen ? "cubic-bezier(0.22, 1, 0.36, 1)" : "cubic-bezier(0.4, 0, 1, 1)",
    });
    animation.onfinish = () => {
      node.open = targetOpen;
      clear();
    };
  }

  summary.addEventListener("click", toggle);
  return { destroy() { summary.removeEventListener("click", toggle); clear(); } };
}
