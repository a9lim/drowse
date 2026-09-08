<script lang="ts">
  import { barExtent, barTooltip } from "./chartValues";

  interface Props {
    value: number;
    max: number;
    width?: number;
    height?: number;
    showBaseline?: boolean;
    color?: string;
    bipolar?: boolean;
    percentage?: boolean;
    title?: string;
  }

  let {
    value, max, width = 144, height = 8, showBaseline = false,
    color, bipolar = false, percentage = false, title,
  }: Props = $props();

  const filled = $derived(barExtent(value, max) * (bipolar ? 50 : 100));
  const fillX = $derived(bipolar ? (value < 0 ? 50 - filled : 50) : 0);
  const fill = $derived(color ?? (value > 0 ? "var(--accent-green)" : value < 0 ? "var(--accent-red)" : "var(--fg-muted)"));
  const tip = $derived(title ?? barTooltip(value, max, percentage));
</script>

<span
  class="bar"
  class:baseline={showBaseline}
  style:--bar-width={`${width}px`}
  style:height={`${height}px`}

  role="img"
  aria-label={tip}
>
  <span class="fill" style:left={`${fillX}%`} style:width={`${filled}%`} style:--fill={fill}></span>
  {#if bipolar}<span class="midline"></span>{/if}
</span>

<style>
  .bar {
    position: relative;
    display: inline-block;
    vertical-align: middle;
    width: var(--bar-width);
    max-width: 100%;
    border-radius: var(--radius-pill);
    background: var(--data-track-fill);
    box-shadow: inset 0 0 0 1px var(--data-track);
    overflow: hidden;
  }
  .fill {
    position: absolute;
    top: 0;
    bottom: 0;
    border-radius: inherit;
    background: linear-gradient(to bottom, color-mix(in srgb, var(--fill) 80%, white), var(--fill));
    transition: width var(--dur) var(--ease-out), left var(--dur) var(--ease-out);
  }
  .midline {
    position: absolute;
    inset-block: 0;
    left: 50%;
    width: 1px;
    background: var(--glass-line);
  }
  .baseline {
    border-bottom: 1px solid var(--glass-line);
  }
</style>
