<script lang="ts">
  import { fluentPaths, type FluentIconName } from "../fluent-icons.mjs";

  let { name, class: className = "", size = 16, spin = false }: {
    name: FluentIconName;
    class?: string;
    size?: string | number;
    spin?: boolean;
  } = $props();
</script>

<svg class={`fluent-icon ${className}`} class:spinning={spin} width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" focusable="false" data-icon={name}>
  {#each fluentPaths[name] as path}<path d={path} />{/each}
</svg>

<style>
  svg { display: inline-block; flex: none; vertical-align: -0.125em; fill: currentColor; }
  .spinning { animation: icon-spin var(--dur-spin) linear infinite; }
  @keyframes icon-spin { to { transform: rotate(360deg); } }
  @media (prefers-reduced-motion: reduce) { .spinning { animation: none; } }
</style>
