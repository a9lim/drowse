<script lang="ts">
  import { slidingSelection } from "../slidingSelection";
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

<div class="theme-toggle" role="group" aria-label="Appearance" use:slidingSelection>
  <button
    type="button"
    class:active={theme === "light"}
    aria-label="Light"
    aria-pressed={theme === "light"}
    onclick={() => setTheme("light")}
  ><ThemeIcon name="sunny" class="theme-icon" /></button>
  <button
    type="button"
    class:active={theme === "dark"}
    aria-label="Dark"
    aria-pressed={theme === "dark"}
    onclick={() => setTheme("dark")}
  ><ThemeIcon name="moon" class="theme-icon" /></button>
</div>

<style>
  .theme-toggle {
    --radius-inset: var(--radius-pill);
    display: inline-grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    flex: none;
    gap: calc(var(--space-xs) / 4);
    padding: calc(var(--space-xs) / 4);
    border-radius: var(--radius-pill);
    background: var(--surface-sheen), var(--glass);
    box-shadow: var(--shadow-control);
  }

  button {
    position: relative;
    display: inline-grid;
    place-items: center;
    width: var(--control-target);
    height: var(--control-target);
    min-width: 40px;
    min-height: 40px;
    padding: 0;
    border: 0;
    border-radius: var(--radius-inset);
    background: transparent;
    color: var(--fg-muted);
    font-size: var(--text-xs);
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      transform var(--dur-fast) var(--ease-out);
  }

  :global(.theme-icon) {
    width: 1rem;
    height: 1rem;
  }

  button:hover:not(.active) {
    background: var(--bg-hover);
    color: var(--fg-strong);
  }

  button:active { transform: scale(var(--press-scale)); }

  button.active {
    background: var(--glass-bright);
    color: var(--fg);
    box-shadow: inset 0 0 0 1px var(--glass-line);
  }

</style>
