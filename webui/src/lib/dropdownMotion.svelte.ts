import { motionDuration } from "./motion";

export function dropdownMotion() {
  let mounted = $state(false);
  let closeTimer: ReturnType<typeof setTimeout> | undefined;

  function cancelClose() {
    clearTimeout(closeTimer);
    closeTimer = undefined;
  }

  return {
    get mounted() { return mounted; },
    mount() {
      cancelClose();
      mounted = true;
    },
    show(node: HTMLElement) {
      cancelClose();
      node.inert = false;
      void node.offsetHeight;
      node.classList.remove("is-closing");
      node.classList.add("is-open");
    },
    close(node: HTMLElement | null) {
      cancelClose();
      if (!node) { mounted = false; return; }
      node.inert = true;
      node.classList.remove("is-open");
      node.classList.add("is-closing");
      const value = getComputedStyle(node).getPropertyValue("--dropdown-close-dur").trim();
      const milliseconds = parseFloat(value) * (value.endsWith("ms") ? 1 : 1000);
      const finish = () => {
        node.classList.remove("is-closing");
        mounted = false;
      };
      const duration = motionDuration(milliseconds);
      if (duration === 0) finish();
      else closeTimer = setTimeout(finish, duration);
    },
    destroy() { cancelClose(); },
  };
}
