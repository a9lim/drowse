<script lang="ts">
  import FluentIcon from "./FluentIcon.svelte";

  interface Props {
    onclick: (ev: MouseEvent) => void;
    label?: string;
  }

  let { onclick, label = "Close drawer" }: Props = $props();
</script>

<button type="button" class="drawer-close" aria-label={label} {onclick}>
  <FluentIcon name="dismiss" />
</button>

<style>
  .drawer-close {
    display: inline-grid;
    place-items: center;
    flex: none;
    width: var(--control-compact);
    height: var(--control-compact);
    padding: 0;
    border: 1px solid transparent;
    border-radius: var(--radius);
    background: var(--glass);
    color: var(--fg-muted);
    cursor: pointer;
    font-size: var(--text-sm);
    line-height: 1;
    transition:
      color var(--dur-fast) var(--ease-out),
      background var(--dur-fast) var(--ease-out),
      box-shadow var(--dur-fast) var(--ease-out),
      scale var(--dur-fast) var(--ease-out);
    box-shadow: var(--shadow-control);
  }

  .drawer-close:hover {
    background: var(--glass-strong);
    color: var(--fg);
    box-shadow: var(--shadow-control-hover);
  }

  .drawer-close:active {
    scale: var(--press-scale);
  }

  :global(.drawer-close svg) {
    width: 1rem;
    height: 1rem;
  }
</style>
