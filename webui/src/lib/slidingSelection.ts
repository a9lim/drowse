import "./style/sliding-selection.css";

let selectionId = 0;

export function slidingSelection(node: HTMLElement) {
  const themeToggle = node.classList.contains("theme-toggle");
  node.classList.add("sliding-selection");
  const indicator = document.createElement("span");
  indicator.className = "selection-indicator";
  indicator.setAttribute("aria-hidden", "true");
  if (themeToggle) {
    const id = ++selectionId;
    indicator.style.viewTransitionName = `theme-selection-${id}`;
    node.querySelectorAll("button").forEach((button, index) => {
      button.style.viewTransitionName = `theme-option-${id}-${index}`;
    });
  }
  node.prepend(indicator);
  let frame = 0;
  let readyFrame = 0;
  let previousSelection: HTMLElement | null = null;

  function measure() {
    frame = 0;
    const selected = node.querySelector<HTMLElement>(
      'button[aria-pressed="true"], button[aria-selected="true"], button[aria-current="page"]',
    );
    if (!selected || !selected.getClientRects().length) {
      node.removeAttribute("data-selection-visible");
      return;
    }
    if (themeToggle && previousSelection && previousSelection !== selected) {
      node.dataset.selectionReady = "";
    }
    previousSelection = selected;
    let x = 0;
    let y = 0;
    let element: HTMLElement | null = selected;
    while (element && element !== node) {
      x += element.offsetLeft;
      y += element.offsetTop;
      element = element.offsetParent as HTMLElement | null;
    }
    node.style.setProperty("--selection-x", `${x}px`);
    node.style.setProperty("--selection-y", `${y}px`);
    node.style.setProperty("--selection-width", `${selected.offsetWidth}px`);
    node.style.setProperty("--selection-height", `${selected.offsetHeight}px`);
    node.dataset.selectionVisible = "";
    if (!themeToggle && !node.hasAttribute("data-selection-ready") && !readyFrame) {
      readyFrame = requestAnimationFrame(() => {
        node.dataset.selectionReady = "";
        readyFrame = 0;
      });
    }
  }

  function schedule() {
    if (!frame) frame = requestAnimationFrame(measure);
  }

  const resize = new ResizeObserver(schedule);
  function observeSizes() {
    resize.disconnect();
    resize.observe(node);
    node.querySelectorAll("button").forEach((button) => resize.observe(button));
  }
  const changes = new MutationObserver((records) => {
    if (records.some((record) => record.type === "childList")) observeSizes();
    measure();
  });
  changes.observe(node, {
    subtree: true, childList: true,
    attributes: true, attributeFilter: ["aria-pressed", "aria-selected", "aria-current", "class"],
  });
  observeSizes();
  measure();

  return {
    destroy() {
      changes.disconnect();
      resize.disconnect();
      cancelAnimationFrame(frame);
      cancelAnimationFrame(readyFrame);
      indicator.remove();
    },
  };
}
