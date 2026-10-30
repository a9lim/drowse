<script lang="ts">
  import { cubicOut } from "svelte/easing";
  import { prefersReducedMotion, Tween } from "svelte/motion";

  interface Props {
    name: "sunny" | "moon";
    class?: string;
  }

  let { name, class: className = "" }: Props = $props();
  const clipId = $props.id();
  const morph = Tween.of(() => name === "moon" ? 1 : 0, {
    duration: () => prefersReducedMotion.current ? 0 : 320,
    easing: cubicOut,
  });
  const progress = $derived(prefersReducedMotion.current ? (name === "moon" ? 1 : 0) : morph.current);
  const cutout = $derived(-6 + 14 * progress);
</script>

<svg class={className} viewBox="0 0 24 24" aria-hidden="true" focusable="false" data-theme-icon={name}>
  <defs>
    <clipPath id={clipId}>
      <circle cx="12" cy="12" r={5 + 3 * progress} />
    </clipPath>
  </defs>
  <path
    class="disc"
    clip-path={`url(#${clipId})`}
    fill-rule="evenodd"
    d={`M0 0h24v24H0z M${cutout - 7} ${cutout}a7 7 0 1 0 14 0a7 7 0 1 0-14 0z`}
  />
  <g class="rays" opacity={1 - progress} stroke-linecap="round">
    {#each [0, 45, 90, 135, 180, 225, 270, 315] as angle}
      <line
        x1="12" y1={4 + 3 * progress}
        x2="12" y2={2 + 5 * progress}
        transform={`rotate(${angle} 12 12)`}
      />
    {/each}
  </g>
</svg>

<style>
  svg { display: block; overflow: visible; }
  .disc { fill: currentColor; }
  .rays { fill: none; stroke: currentColor; stroke-width: 1.5; }
</style>
