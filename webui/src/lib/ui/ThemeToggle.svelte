<script lang="ts">
  import { onMount } from "svelte";
  import { initializeTheme, setTheme, subscribeTheme, type Theme } from "../theme";
  import ThemeIcon from "./ThemeIcon.svelte";

  let theme: Theme = $state(initializeTheme());

  onMount(() => {
    return subscribeTheme((next) => {
      theme = next;
    });
  });
</script>

<div class="theme-toggle" role="group" aria-label="Appearance">
  <button
    type="button"
    aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
    onclick={() => setTheme(theme === "dark" ? "light" : "dark")}
  >
    <ThemeIcon name={theme === "dark" ? "moon" : "sunny"} class="theme-icon" />
  </button>
</div>

<style>
  .theme-toggle {
    display: inline-flex;
    flex: none;
  }

  button {
    --control-sheen: none;
    position: relative;
    display: inline-grid;
    place-items: center;
    width: var(--control-target);
    height: var(--control-target);
    min-width: 44px;
    min-height: 44px;
    touch-action: manipulation;
    padding: 0;
    border: 0;
    border-radius: var(--radius-pill);
    background: transparent;
    box-shadow: none;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    transition:
      color var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }

  :global(.theme-icon) {
    width: 1.5rem;
    height: 1.5rem;
  }

  button:hover {
    color: var(--fg-strong);
  }

  button:active { transform: scale(var(--press-scale)); }

  @media (prefers-reduced-motion: reduce) {
    button { transition: none; }
    button:active { transform: none; }
  }

</style>
