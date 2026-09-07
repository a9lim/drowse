<script lang="ts">
  // Tiny inline sparkline.  Renders a polyline over the points list,
  // auto-scaled to the visible range.  Designed for live append — the
  // store's probe rack drops trailing values when the buffer grows
  // past N, so the component is fed a fresh array each tick and the
  // SVG re-renders cheaply.
  //
  // Hover exposes the untruncated readings behind the compact trace.

  import { onMount } from "svelte";
  import { sparklinePaths } from "./sparklinePath";
  import { chartValue } from "./chartValues";

  interface Props {
    points: (number | null)[];
    width?: number;
    height?: number;
    /** When set, points are clamped to ``[-cap, +cap]`` before scaling
     * so a single outlier doesn't squash the rest of the trace.  Probe
     * scores live in [-1, 1] so 1 is a sensible default. */
    cap?: number;
    /** Stroke color override; defaults to fg-dim. */
    color?: string;
    percentage?: boolean;
  }

  let {
    points,
    width = 60,
    height = 16,
    cap = 1,
    color,
    percentage = false,
  }: Props = $props();

  const stroke = $derived(color ?? "var(--fg-dim)");

  const paths = $derived(sparklinePaths(points, width, height, cap));
  const tip = $derived.by(() => {
    const values = points.filter((v): v is number => v !== null && Number.isFinite(v));
    if (!values.length) return "No readings yet";
    return `Latest ${chartValue(values[values.length - 1], percentage)} · range ${chartValue(Math.min(...values), percentage)} to ${chartValue(Math.max(...values), percentage)} · ${values.length} readings`;
  });

  // Draw-in runs once, gated on mount — this component re-renders every
  // streamed token, so the animation is bound to a class flipped a single
  // time (never to the per-tick data), and the dasharray uses a fixed
  // over-length value so path updates can't restart it.
  let mounted = $state(false);
  onMount(() => {
    mounted = true;
  });
</script>

<svg
  class="sparkline"
  {width}
  {height}
  viewBox="0 0 {width} {height}"
  preserveAspectRatio="none"
  role="img"
  aria-label={tip}
>
  <title>{tip}</title>
  {#if paths.area}
    <path d={paths.area} fill={stroke} fill-opacity="0.14" stroke="none" />
  {/if}
  {#if paths.line}
    <path
      class="sparkline-line"
      class:draw={mounted}
      d={paths.line}
      fill="none"
      stroke={stroke}
      stroke-width="1"
      stroke-linecap="round"
      stroke-linejoin="round"
    />
  {/if}
</svg>

<style>
  .sparkline {
    display: inline-block;
    vertical-align: middle;
  }
  /* Fixed over-length dash (path is ≪400 units) so per-token ``d`` updates
     never re-measure or restart the draw-in; the animation is carried by
     the ``.draw`` class, applied once on mount. */
  .sparkline-line {
    stroke-dasharray: 400;
    stroke-dashoffset: 0;
  }
  .sparkline-line.draw {
    animation: sparkline-draw var(--dur-draw) var(--ease-out);
  }
  @keyframes sparkline-draw {
    from {
      stroke-dashoffset: 400;
    }
    to {
      stroke-dashoffset: 0;
    }
  }
</style>
