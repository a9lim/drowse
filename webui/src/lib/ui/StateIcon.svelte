<script lang="ts">
  import FluentIcon from "./FluentIcon.svelte";
  import type { FluentIconName } from "../fluent-icons.mjs";

  let { name, icons, size = 18 }: {
    name: FluentIconName;
    icons: FluentIconName[];
    size?: number;
  } = $props();
</script>

<span class="state-icon" style:width={`${size}px`} style:height={`${size}px`} aria-hidden="true">
  {#each icons as icon (icon)}
    <span class="variant" class:active={name === icon}><FluentIcon name={icon} {size} /></span>
  {/each}
</span>

<style>
  .state-icon { position: relative; display: inline-block; flex: none; vertical-align: -0.125em; }
  .variant {
    position: absolute; inset: 0; display: grid; place-items: center;
    opacity: 0; scale: 0.25; filter: blur(4px);
    transition: opacity var(--dur-slow) cubic-bezier(0.2, 0, 0, 1), scale var(--dur-slow) cubic-bezier(0.2, 0, 0, 1), filter var(--dur-slow) cubic-bezier(0.2, 0, 0, 1);
  }
  .active { opacity: 1; scale: 1; filter: blur(0); }
  @media (prefers-reduced-motion: reduce) { .variant { transition: none; } }
</style>
